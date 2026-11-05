import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.artifacts import sha256
from experiments.stage1.inference import score_window
from experiments.stage2_clean_reference.attention import last_reference, reference_attention
from experiments.stage2_clean_reference.data import generate, shared_normalize
from experiments.stage2_clean_reference.inference import run_case
from experiments.stage2_clean_reference.statistics import paired_contrasts, seed_summary
from experiments.stage2_paths.attention import model_fingerprint
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig
from time_rcd._inference import _WindowDataset


class ReferenceKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(1)
        self.q, self.k, self.v, self.dk, self.dv = [torch.randn(1, 2, 4, 3, dtype=torch.float64) for _ in range(5)]
        self.valid = np.array([True, True, True, False])
        self.selected = np.array([True, False, True, False])

    def test_query_specific_key_value_matches_explicit_edges(self):
        for mode in ("A", "C", "V", "KV"):
            actual, _ = reference_attention(self.q, self.k, self.v, self.valid, self.selected, mode, self.dk, self.dv)
            expected = torch.zeros_like(actual)
            for head in range(2):
                for i in range(4):
                    logits, values = [], []
                    for j in range(4):
                        replace = self.selected[j] and i != j
                        key = self.dk[0, head, j] if replace and mode == "KV" else self.k[0, head, j]
                        value = self.dv[0, head, j] if replace and mode in ("V", "KV") else self.v[0, head, j]
                        logit = self.q[0, head, i] @ key / math.sqrt(3)
                        logits.append(logit if self.valid[j] and not (mode == "C" and replace) else torch.tensor(-torch.inf))
                        values.append(value)
                    expected[0, head, i] = torch.stack(logits).softmax(0) @ torch.stack(values)
            torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)

    def test_noop_replay_fixed_weights_and_single_target_self(self):
        baseline, original = reference_attention(self.q, self.k, self.v, self.valid, self.selected, "A")
        for mode in ("V", "KV"):
            empty, _ = reference_attention(self.q, self.k, self.v, self.valid, np.zeros(4, dtype=bool), mode, self.dk, self.dv)
            replay, _ = reference_attention(self.q, self.k, self.v, self.valid, self.selected, mode, self.k, self.v)
            torch.testing.assert_close(empty, baseline, atol=0, rtol=0)
            torch.testing.assert_close(replay, baseline, atol=0, rtol=0)
            one, record = reference_attention(self.q, self.k, self.v, self.valid, np.array([True, False, False, False]), mode, self.dk, self.dv)
            torch.testing.assert_close(one[:, :, 0], baseline[:, :, 0], atol=0, rtol=0)
            if mode == "V":
                torch.testing.assert_close(record["weights"], original["weights"], atol=0, rtol=0)
            else:
                torch.testing.assert_close(torch.diagonal(record["logits"], dim1=-2, dim2=-1),
                                           torch.diagonal(original["logits"], dim1=-2, dim2=-1), atol=0, rtol=0)

    def test_invalid_donor_and_padded_selection_rejected(self):
        with self.assertRaises(ValueError):
            reference_attention(self.q, self.k, self.v, self.valid, self.selected, "KV")
        with self.assertRaises(ValueError):
            reference_attention(self.q, self.k, self.v, self.valid, np.ones(4, dtype=bool), "A")


class ReferenceDataTests(unittest.TestCase):
    def test_frozen_design_energy_coverage_and_shared_scale(self):
        config = json.loads(Path("experiments/configs/stage2_clean_reference_v1.json").read_text())
        records = list(generate(config["synthetic"], 16))
        self.assertEqual(len(records), 144)
        clean_by_seed = {}
        for meta, arrays in records:
            self.assertEqual(meta["anomaly_count"], 640)
            self.assertEqual(meta["perturbation_energy"], 10240)
            self.assertEqual(meta["event_count"] * meta["event_length"], 640)
            self.assertEqual(meta["P"] == 0, meta["event_length"] == 5)
            if meta["event_length"] > 5 and meta["offset"] == 0:
                self.assertEqual(meta["M"], 0)
                self.assertEqual(meta["P"], 40)
            normalized, _ = _WindowDataset(arrays["corrupt"], 5000)[0]
            np.testing.assert_array_equal(normalized.numpy(), arrays["corrupt_normalized"].astype(np.float32))
            delta = arrays["corrupt_normalized"] - arrays["clean_normalized"]
            np.testing.assert_allclose(delta, (arrays["corrupt"] - arrays["clean"]) / meta["normalization_std"], atol=1e-14)
            if meta["seed"] in clean_by_seed:
                np.testing.assert_array_equal(arrays["clean"], clean_by_seed[meta["seed"]])
            clean_by_seed[meta["seed"]] = arrays["clean"]
        with self.assertRaises(ValueError):
            shared_normalize(np.ones((4, 1)), np.ones((5, 1)))

    def test_seed_is_the_bootstrap_unit_and_missing_conditions_fail(self):
        rows = []
        for seed, value in ((1, .1), (2, .3)):
            for offset in (0, 8):
                for mode, delta in (("A", 0), ("C", -value), ("V", value), ("KV", 2 * value), ("C_all", -2 * value)):
                    rows.append({"case": f"{seed}_{offset}", "seed": seed, "shape": "positive", "event_length": 32,
                                 "offset": offset, "M": 0, "P": 2, "score_kind": "scores", "variant": mode,
                                 "metric_status": "ok", "average_precision": .5 + delta})
        frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
        paired, columns = paired_contrasts(frame)
        seeds, summary = seed_summary(paired, [], columns, {"synthetic": {"seeds": [1, 2]},
                                     "bootstrap_seed": 2024, "bootstrap_repetitions": 100})
        row = summary[summary.contrast == "V_minus_A"].iloc[0]
        self.assertEqual(row.seeds, 2)
        self.assertAlmostEqual(row["mean"], .2)
        self.assertEqual(len(seeds[seeds.contrast == "V_minus_A"]), 2)
        with self.assertRaises(ValueError):
            paired_contrasts(frame.iloc[1:])
        with self.assertRaises(ValueError):
            paired_contrasts(pd.concat([frame, frame.iloc[:1]]))


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 3
        config.ts_config.num_heads = 2
        self.model = TimeSeriesPretrainModel(config).eval()

    def test_full_pipeline_and_snapshot_checks(self):
        config = json.loads(Path("experiments/configs/stage2_clean_reference_v1.json").read_text())
        settings = {**config["synthetic"], "length": 256, "anomaly_count": 64, "seeds": [1],
                    "event_lengths": [32], "offsets": [8], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))
        fingerprint = model_fingerprint(self.model)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "arrays").mkdir()
            path = directory / "arrays/input.npz"
            np.savez_compressed(path, **arrays)
            meta.update(input_snapshot="arrays/input.npz", input_sha256=sha256(path))
            self.assertEqual(run_case(directory, directory, {**config, "window": 256}, self.model, "cpu", meta, gate=True), 5)
            self.assertEqual(run_case(directory, directory, {**config, "window": 256}, self.model, "cpu", meta), 5)
            checks = pd.read_csv(directory / "checks.csv")
            self.assertTrue(checks.close.all())
            self.assertEqual(len(pd.read_csv(directory / "metrics.csv")), 10)
        self.assertEqual(model_fingerprint(self.model), fingerprint)
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in self.model.ts_encoder.transformer_encoder.layers))

    def test_partial_patch_and_hook_restoration_after_failure(self):
        values, valid = torch.randn(49, 1), torch.ones(49, dtype=torch.bool)
        present, selected = np.ones(4, dtype=bool), np.array([False, True, False, False])
        official = score_window(self.model, values, valid, "cpu")
        with last_reference(self.model, present, selected, "A"):
            adapted = score_window(self.model, values, valid, "cpu")
        np.testing.assert_allclose(official["logits"], adapted["logits"], atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            with last_reference(self.model, present, selected, "KV"):
                score_window(self.model, values, valid, "cpu")
        self.assertNotIn("forward", self.model.ts_encoder.transformer_encoder.layers[-1].self_attn.__dict__)


if __name__ == "__main__":
    unittest.main()
