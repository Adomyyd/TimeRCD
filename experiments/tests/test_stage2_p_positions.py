import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.artifacts import sha256, write_json
from experiments.stage2_alignment_controls.data import generate
from experiments.stage2_alignment_controls.inference import predict
from experiments.stage2_p_positions.data import select_positions, verify_parents
from experiments.stage2_p_positions.inference import run_case
from experiments.stage2_p_positions.statistics import paired_effects
from experiments.stage2_paths.attention import model_fingerprint
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig


def fixture(length=32):
    config = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
    settings = {**config["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                "phases": [16], "event_lengths": [length], "shapes": ["negative"]}
    return next(generate(settings, 16))[0]


class PositionDataTests(unittest.TestCase):
    def test_per_event_counts_direction_and_random_reproducibility(self):
        for length in (32, 128):
            meta, arrays = fixture(length)
            masks, kept = select_positions(meta, arrays, 16, [2024, 2025])
            other, indices = select_positions(meta, arrays, 16, [2025, 2024])
            for name, mask in masks.items():
                np.testing.assert_array_equal(mask, other[name])
                if name == "full":
                    continue
                self.assertEqual(kept[name], indices[name])
                for start, keep in zip(arrays["starts"], kept[name]):
                    patches = np.arange(start // 16, (start + length) // 16)
                    self.assertIn(keep, patches)
                    self.assertEqual(mask[patches].sum(), len(patches) - 1)
                    self.assertFalse(mask[keep])
            np.testing.assert_array_equal(kept["keep_first"], arrays["starts"] // 16)
            np.testing.assert_array_equal(kept["keep_last"], (arrays["starts"] + length) // 16 - 1)
            np.testing.assert_array_equal(masks["keep_first"], arrays["common"])

    def test_invalid_events_labels_and_common_fail(self):
        meta, arrays = fixture()
        for mutation in ("labels", "overlap", "unaligned", "common", "type"):
            bad = copy.deepcopy(arrays)
            if mutation == "labels":
                bad["labels"][0] = 1
            elif mutation == "overlap":
                bad["starts"][1] = bad["starts"][0]
            elif mutation == "unaligned":
                bad["starts"][0] += 1
            elif mutation == "common":
                bad["common"][0] = ~bad["common"][0]
            else:
                bad["type_P"][0] = ~bad["type_P"][0]
            with self.assertRaises(ValueError, msg=mutation):
                select_positions(meta, bad, 16, [2024])

    def test_parent_recovery_identity_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, recovery = root / "parent", root / "recovery"
            parent.mkdir()
            recovery.mkdir()
            write_json(parent / "manifest.json", {"status": "interrupted", "checkpoint": {"sha256": "frozen"}})
            write_json(recovery / "manifest.json", {"status": "completed", "recovery_parent": str(parent),
                                                     "checkpoint": {"sha256": "frozen"}})
            for path in (parent / "data_manifest.json", recovery / "summary.json"):
                write_json(path, {})
            for path in (parent / "checks.csv", parent / "gates/checks.csv", parent / "regression/checks.csv", recovery / "checks.csv"):
                path.parent.mkdir(exist_ok=True)
                pd.DataFrame([{"close": True}]).to_csv(path, index=False)
            write_json(recovery / "parent_artifacts.sha256.json", {"data_manifest.json": sha256(parent / "data_manifest.json")})
            config = {"parent_run": str(parent), "recovery_run": str(recovery)}
            self.assertTrue(verify_parents(config, "frozen"))
            write_json(parent / "data_manifest.json", {"changed": True})
            with self.assertRaises(ValueError):
                verify_parents(config, "frozen")


class PositionStatisticsTests(unittest.TestCase):
    def frame(self):
        values = {"A": .1, "full_last": .2, "full_all": .9,
                  "keep_first_last": .3, "keep_first_all": .4,
                  "keep_last_last": .35, "keep_last_all": .7,
                  "random_2024_last": .2, "random_2024_all": .5,
                  "random_2025_last": .6, "random_2025_all": .9}
        return pd.DataFrame([{"case": "one", "seed": 1, "phase": 0, "shape": "negative",
                              "event_length": 32, "offset": 0, "M": 0, "P": 8, "common_patches": 4,
                              "score_kind": "scores", "metric_status": "ok", "variant": variant,
                              "average_precision": value, "missing": np.nan} for variant, value in values.items()])

    def test_signed_pairing_and_input_level_random_metric_mean(self):
        effects, absolute, ranges, _ = paired_effects(self.frame().sample(frac=1, random_state=9), [2024, 2025])
        effect = effects.set_index("contrast")
        self.assertAlmostEqual(effect.loc["keep_last_all_minus_keep_first_all", "average_precision"], .3)
        self.assertAlmostEqual(effect.loc["position_by_scope", "average_precision"], .25)
        self.assertAlmostEqual(absolute.set_index("contrast").loc["random_mean_all", "average_precision"], .7)
        self.assertAlmostEqual(effect.loc["keep_first_all_minus_random_mean_all", "average_precision"], -.3)
        self.assertTrue(effects.missing.isna().all())
        self.assertAlmostEqual(ranges[(ranges.scope == "all") & (ranges.metric == "average_precision")].iloc[0]["range"], .4)

    def test_missing_duplicate_and_condition_dependent_missingness_fail(self):
        frame = self.frame()
        bad_missing = frame.copy()
        bad_missing.loc[0, "average_precision"] = np.nan
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), bad_missing):
            with self.assertRaises(ValueError):
                paired_effects(bad, [2024, 2025])


class PositionModelTests(unittest.TestCase):
    def test_miniature_pipeline_parent_equivalence_and_empty_masks(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        settings = TimeRCDConfig()
        settings.ts_config.d_model = 16
        settings.ts_config.d_proj = 8
        settings.ts_config.patch_size = 16
        settings.ts_config.num_layers = 3
        settings.ts_config.num_heads = 2
        model = TimeSeriesPretrainModel(settings).eval()
        fingerprint = model_fingerprint(model)
        meta, arrays = fixture()
        config = json.loads(Path("experiments/configs/stage2_p_positions_v1.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "arrays").mkdir()
            parent = directory / "parent"
            (parent / "arrays").mkdir(parents=True)
            config.update(parent_run=str(parent), window=512)
            values = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32)
            valid = torch.ones(len(values), dtype=torch.bool)
            old = {}
            for variant, condition, scope, mask in (("A", "A", "all", arrays["type_P"]),
                    ("full_last", "C", "last", arrays["type_P"]), ("full_all", "C", "all", arrays["type_P"]),
                    ("common_last", "C", "last", arrays["common"]), ("common_all", "C", "all", arrays["common"])):
                prediction, _ = predict(model, values, valid, arrays["type_valid"], mask, condition, scope, "cpu")
                old.update({f"{variant}_{k}": v for k, v in prediction.items()})
            np.savez_compressed(parent / f"arrays/{meta['case']}.npz", **old)
            path = directory / "arrays/input.npz"
            np.savez_compressed(path, **arrays)
            meta.update(input_snapshot="arrays/input.npz", input_sha256=sha256(path))
            masks, kept = select_positions(meta, arrays, 16, config["random_seeds"])
            path = directory / "arrays/selection.npz"
            np.savez_compressed(path, **masks, **{f"kept_{k}": v for k, v in kept.items()})
            meta.update(selection_snapshot="arrays/selection.npz", selection_sha256=sha256(path))
            self.assertEqual(run_case(directory, directory, config, model, "cpu", meta, gate=True), 17)
            self.assertEqual(run_case(directory, directory, config, model, "cpu", meta), 17)
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())
            self.assertEqual(len(pd.read_csv(directory / "metrics.csv")), 34)
            with np.load(directory / f"arrays/{meta['case']}.npz") as saved:
                self.assertEqual(len(saved.files), 34)
        self.assertEqual(model_fingerprint(model), fingerprint)
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in model.ts_encoder.transformer_encoder.layers))


if __name__ == "__main__":
    unittest.main()
