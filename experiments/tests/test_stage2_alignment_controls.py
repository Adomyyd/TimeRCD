import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.artifacts import sha256
from experiments.stage2_alignment_controls.data import generate, verify_pair
from experiments.stage2_alignment_controls.inference import predict, run_case
from experiments.stage2_alignment_controls.statistics import alignment_interactions, paired_effects
from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_paths.attention import model_fingerprint
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig


def settings():
    return json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())


class AlignmentDataTests(unittest.TestCase):
    def test_full_design_paired_background_and_common_positions(self):
        config = settings()
        cases = set()
        pairs = 0
        for pair in generate(config["synthetic"], 16):
            check = verify_pair(pair, config["synthetic"], 16)
            self.assertLessEqual(check["normalized_translation_max_abs"], 1e-12)
            pairs += 1
            for meta, arrays in pair:
                cases.add(meta["case"])
                self.assertEqual(arrays["labels"].sum(), 640)
                self.assertEqual(meta["common_patches"], 20 if meta["event_length"] == 32 else 35)
                self.assertEqual(meta["P"], 40 if meta["offset"] == 0 else meta["common_patches"])
            for start0, start8 in zip(pair[0][1]["starts"], pair[1][1]["starts"]):
                length = pair[0][0]["event_length"]
                np.testing.assert_array_equal(pair[0][1]["clean"][start0:start0 + length],
                                              pair[1][1]["clean"][start8:start8 + length])
        self.assertEqual(pairs, 192)
        self.assertEqual(len(cases), 384)

    def test_pair_corruption_is_rejected(self):
        config = settings()
        pair = next(generate(config["synthetic"], 16))
        pair[1][1]["corrupt"][100] += .01
        with self.assertRaises(ValueError):
            verify_pair(pair, config["synthetic"], 16)
        pair = next(generate(config["synthetic"], 16))
        pair[1][1]["common"][0] = True
        with self.assertRaises(ValueError):
            verify_pair(pair, config["synthetic"], 16)


class AlignmentStatisticsTests(unittest.TestCase):
    def frame(self):
        rows = []
        for seed, scale in ((1, 1), (2, 2)):
            for offset in (0, 8):
                for variant, effect in (("A", 0), ("full_last", .01),
                                        ("full_all", .02 if offset == 0 else .08),
                                        ("common_last", .01), ("common_all", .04 if offset == 0 else .08)):
                    rows.append({"case": f"{seed}_{offset}", "seed": seed, "phase": 0, "shape": "positive",
                                 "event_length": 32, "offset": offset, "M": 0, "P": 2, "common_patches": 1,
                                 "score_kind": "scores", "variant": variant, "metric_status": "ok",
                                 "average_precision": .5 + scale * effect,
                                 "missing_group": np.nan if offset == 0 else .4 + effect})
        return pd.DataFrame(rows).sample(frac=1, random_state=3)

    def test_signed_interactions_and_seed_pairing(self):
        effects, columns = paired_effects(self.frame())
        _, interactions = alignment_interactions(effects, columns)
        first = interactions[interactions.seed == 1].set_index("contrast")
        self.assertAlmostEqual(first.loc["I_common", "average_precision"], .04)
        self.assertAlmostEqual(first.loc["I_full", "average_precision"], .06)
        self.assertAlmostEqual(first.loc["J_common_minus_full", "average_precision"], -.02)
        self.assertTrue(interactions.missing_group.isna().all())
        _, stats = seed_summary(interactions, [], columns, {"synthetic": {"seeds": [1, 2]},
                               "bootstrap_seed": 2024, "bootstrap_repetitions": 100})
        row = stats[(stats.contrast == "I_common") & (stats.metric == "average_precision")].iloc[0]
        self.assertEqual(row.seeds, 2)
        self.assertAlmostEqual(row["mean"], .06)

    def test_missing_duplicate_and_unpaired_arms_fail(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            paired_effects(frame.iloc[1:])
        with self.assertRaises(ValueError):
            paired_effects(pd.concat([frame, frame.iloc[:1]]))
        effects, columns = paired_effects(frame)
        with self.assertRaises(ValueError):
            alignment_interactions(effects.iloc[1:], columns)


class AlignmentModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 3
        config.ts_config.num_heads = 2
        self.model = TimeSeriesPretrainModel(config).eval()

    def test_miniature_full_pipeline_and_common_identity(self):
        config = settings()
        synthetic = {**config["synthetic"], "length": 256, "anomaly_count": 64,
                     "seeds": [1], "phases": [16], "event_lengths": [32], "shapes": ["negative"]}
        pair = next(generate(synthetic, 16))
        verify_pair(pair, synthetic, 16)
        fingerprint = model_fingerprint(self.model)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "arrays").mkdir()
            for meta, arrays in pair:
                path = directory / "arrays" / f"input_{meta['case']}.npz"
                np.savez_compressed(path, **arrays)
                meta.update(input_snapshot=str(path.relative_to(directory)), input_sha256=sha256(path))
                self.assertEqual(run_case(directory, directory, {**config, "window": 256}, self.model, "cpu", meta, gate=True), 5)
                self.assertEqual(run_case(directory, directory, {**config, "window": 256}, self.model, "cpu", meta), 5)
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())
            self.assertEqual(len(pd.read_csv(directory / "metrics.csv")), 20)
        self.assertEqual(model_fingerprint(self.model), fingerprint)

    def test_padding_empty_selection_and_adapter_restoration(self):
        values, valid = torch.randn(64, 1), torch.arange(64) < 33
        present = np.array([True, True, True, False])
        empty = np.zeros(4, dtype=bool)
        a, _ = predict(self.model, values, valid, present, empty, "A", "all", "cpu")
        for scope in ("last", "all"):
            c, _ = predict(self.model, values, valid, present, empty, "C", scope, "cpu")
            np.testing.assert_array_equal(a["scores"], c["scores"])
        with self.assertRaises(ValueError):
            predict(self.model, values, valid, present, np.ones(4, dtype=bool), "C", "all", "cpu")
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in self.model.ts_encoder.transformer_encoder.layers))


if __name__ == "__main__":
    unittest.main()
