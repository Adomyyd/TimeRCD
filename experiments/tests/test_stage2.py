from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.data import Series
from experiments.stage2.inference import run_case, windows
from experiments.stage2.masking import reference_intervention, reference_masks, random_reference_mask
from experiments.stage2.synthetic import generate
from experiments.stage2.analyze import paired_cases, aggregate
from time_rcd import TimeRCDDetector
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig
from time_rcd._inference import TimeRCDPretrainTester


class OracleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 3
        config.ts_config.num_heads = 2
        self.model = TimeSeriesPretrainModel(config).eval()

    def test_mapping_mixed_partial_and_invalid_points(self):
        labels = np.zeros(49, dtype=int)
        labels[[15, 16, 48]] = 1
        valid = np.arange(49) < 33
        present, clean, mixed = reference_masks(labels, valid, 16)
        np.testing.assert_array_equal(present, [True, True, True, False])
        np.testing.assert_array_equal(clean, [False, False, True, False])
        np.testing.assert_array_equal(mixed, [True, True, False, False])
        for seed in range(5):
            random = random_reference_mask(present, clean, np.random.default_rng(seed))
            self.assertEqual(random.sum(), clean.sum())
            self.assertFalse((random & ~present).any())
        with self.assertRaisesRegex(ValueError, "No clean"):
            random_reference_mask(present, np.zeros(4, dtype=bool), np.random.default_rng(1))

    def test_blocked_information_cannot_relay_but_queries_survive(self):
        values = torch.randn(1, 64, 1)
        changed = values.clone()
        changed[:, 16:32] += torch.linspace(1, 10, 16)[None, :, None]
        valid = torch.ones(1, 64, dtype=torch.bool)
        keep = [True, False, True, True]
        with torch.inference_mode():
            original = self.model(values, valid)
            with reference_intervention(self.model, [True] * 4):
                replay = self.model(values, valid)
            torch.testing.assert_close(original, replay, rtol=0, atol=0)
            with reference_intervention(self.model, keep):
                first = self.model(values, valid)
            with reference_intervention(self.model, keep):
                second = self.model(changed, valid)
            clean_points = torch.ones(64, dtype=torch.bool)
            clean_points[16:32] = False
            torch.testing.assert_close(first[:, clean_points], second[:, clean_points], rtol=0, atol=0)
            self.assertGreater(float((first[:, 16:32] - second[:, 16:32]).abs().max()), .01)
            self.assertGreater(float((original - self.model(changed, valid)).abs().max()), .01)
        self.assertEqual(first.shape, (1, 64, 1, 8))
        self.assertFalse(self.model.ts_encoder.transformer_encoder._forward_pre_hooks)
        for layer in self.model.ts_encoder.transformer_encoder.layers:
            self.assertFalse(layer.self_attn._forward_pre_hooks)

    def test_hooks_removed_on_failure_and_invalid_mask_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with reference_intervention(self.model, [True]):
                raise RuntimeError("test failure")
        self.assertFalse(self.model.ts_encoder.transformer_encoder._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with reference_intervention(self.model, [False]):
                pass

    def test_full_pipeline_fallback_preserves_all_points(self):
        tester = object.__new__(TimeRCDPretrainTester)
        tester.model, tester.device = self.model, torch.device("cpu")
        tester.win_size, tester.batch_size = 32, 1
        detector = TimeRCDDetector("unused.pth", win_size=32, batch_size=1, device="cpu")
        detector._tester, detector._num_features = tester, 1
        values = np.random.default_rng(5).normal(size=(65, 1))
        labels = np.zeros(65, dtype=int)
        labels[:32] = 1
        labels[40] = 1
        series = Series("001_Test_tr_1", values, labels, {"kind": "tsb"})
        config = {"random_seeds": [1, 2], "atol": 1e-6, "rtol": 1e-5}
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            directory = Path(temporary)
            (directory / "arrays").mkdir()
            self.assertTrue(run_case(directory, config, detector, series, 0, 32))
            with np.load(directory / "arrays/s0000_w32.npz") as arrays:
                self.assertEqual(arrays["oracle_scores"].shape, (65,))
                np.testing.assert_array_equal(arrays["eligible"][:32], False)
                for name in ("oracle", "random_1", "random_2"):
                    np.testing.assert_array_equal(arrays[f"{name}_scores"][:32], arrays["baseline_scores"][:32])
            audit = pd.read_csv(directory / "windows.csv")
            self.assertEqual(audit.eligible.tolist(), [False, True, True])
            self.assertEqual(audit.mixed_patches.tolist(), [0, 1, 0])

    def test_synthetic_counts_nested_events_and_fixed_normalization(self):
        settings = json.loads(Path("experiments/configs/stage2_oracle_v1.json").read_text())["synthetic"]
        series = list(generate(settings))
        self.assertEqual(len(series), 27)
        for i in range(0, len(series), 3):
            group = series[i:i + 3]
            for sample, count in zip(group, [5, 500, 2000]):
                self.assertEqual(sample.labels.sum(), count)
                events = np.diff(np.pad(sample.labels, (1, 1)))
                starts, ends = np.flatnonzero(events == 1), np.flatnonzero(events == -1)
                np.testing.assert_array_equal(ends - starts, 5)
            self.assertTrue(np.all(group[0].labels <= group[1].labels))
            self.assertTrue(np.all(group[1].labels <= group[2].labels))
            normal = [s.values[:, 0] - 4 * s.labels for s in group]
            np.testing.assert_allclose(normal[0], normal[2], atol=1e-15)
            a, _ = next(windows(group[0], 5000))
            b, _ = next(windows(group[2], 5000))
            unchanged = group[2].labels == group[0].labels
            torch.testing.assert_close(a[unchanged], b[unchanged], rtol=0, atol=0)


class OracleAnalysisTests(unittest.TestCase):
    def test_pairing_averages_metrics_not_predictions_and_rejects_missing_arms(self):
        rows = []
        for variant, ap in (("baseline", .2), ("oracle", .6), ("random_1", .3), ("random_2", .5)):
            rows.append({"case": "a", "scope": "full_fallback", "score_kind": "scores",
                         "series": "a", "window": 5000, "kind": "tsb", "source": "Test",
                         "density_bin": "high", "generator_seed": None, "placement": None,
                         "anomaly_count": 20, "anomaly_ratio": .2, "eligible_points": 100,
                         "length": 100, "variant": variant, "average_precision": ap, "roc_auc": .8})
        frame = pd.DataFrame(rows)
        paired = paired_cases(frame, [1, 2])
        self.assertAlmostEqual(paired.iloc[0].oracle_minus_random_ap, .2)
        self.assertAlmostEqual(paired.iloc[0].oracle_minus_baseline_ap, .4)
        self.assertAlmostEqual(paired.iloc[0].random_max_ap, .5)
        with self.assertRaises(ValueError):
            paired_cases(frame.iloc[:-1], [1, 2])
        with self.assertRaises(ValueError):
            paired_cases(pd.concat([frame, frame.iloc[:1]]), [1, 2])
        group = pd.concat([paired.assign(generator_seed=i, placement=j)
                           for i in [71, 72, 73] for j in ["early", "late", "spread"]])
        result = aggregate(group, "generator_seed", 2024, 100)
        self.assertEqual(result["units"], 3)
        self.assertEqual(result["cases"], 9)
        self.assertAlmostEqual(result["oracle_minus_random_ap"]["mean"], .2)


if __name__ == "__main__":
    unittest.main()
