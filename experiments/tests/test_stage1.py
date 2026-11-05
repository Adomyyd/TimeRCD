from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.artifacts import create_run
from experiments.common.config import ROOT, load_config
from experiments.common.data import Series, geometry, load_series
from experiments.common.metrics import detection_metrics, difference
from experiments.stage1.inference import compare_backends, probe_tail, probe_short
from experiments.stage1.effects import assemble_conditions
from experiments.stage1.select import choose
from time_rcd import TimeRCDDetector
from time_rcd._core.time_rcd_config import TimeRCDConfig
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._inference import TimeRCDPretrainTester, _WindowDataset
from utils.dataset import TimeRCDDataset


class StageOneTests(unittest.TestCase):
    def test_window_and_patch_geometry(self):
        short = geometry(17, 64)
        self.assertEqual(short["actual_window"], 17)
        self.assertEqual(short["tail_valid_ratio"], 1)
        self.assertEqual(short["valid_patch_count"], 2)
        self.assertEqual(short["length_group"], "short")
        tail = geometry(65, 32)
        self.assertEqual(tail["window_count"], 3)
        self.assertEqual(tail["tail_valid_length"], 1)
        self.assertEqual(tail["valid_patch_count"], 5)
        self.assertTrue(tail["tail_partial_patch"])
        self.assertEqual(geometry(64, 32)["length_group"], "multiple_full")

    def test_data_windows_match_at_boundaries_and_for_constants(self):
        rng = np.random.default_rng(19)
        for length in (1, 15, 16, 17, 32, 33, 65):
            for configured in (16, 31, 32):
                window = min(length, configured)
                for values in (rng.normal(size=(length, 1)), np.ones((length, 1))):
                    with self.subTest(length=length, window=window), redirect_stdout(io.StringIO()):
                        public = _WindowDataset(values, window)
                        legacy = TimeRCDDataset(values, window, stride=window, normalize=True)
                        self.assertEqual(len(public), len(legacy))
                        for index in range(len(public)):
                            for left, right in zip(public[index], legacy[index]):
                                torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_synthetic_data_is_reproducible(self):
        settings = json.loads((ROOT / "experiments/configs/stage1_smoke.json").read_text())["data"]
        first = list(load_series(settings, 2024))
        second = list(load_series(settings, 2024))
        other_seed = list(load_series(settings, 2025))
        for a, b, c in zip(first, second, other_seed):
            np.testing.assert_array_equal(a.values, b.values)
            np.testing.assert_array_equal(a.labels, b.labels)
            self.assertFalse(np.array_equal(a.values, c.values))

    def test_tsb_split_preserves_coordinates_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "001_Test_tr_2_1st_3.csv"
            path.write_text("value,Label\n10,0\n11,0\n12,1\n13,0\n")
            settings = {"kind": "tsb", "root": temporary, "files": [path.name], "split": "test"}
            series = list(load_series(settings, 1))[0]
            np.testing.assert_array_equal(series.values[:, 0], [12, 13])
            self.assertEqual(series.source["offset"], 2)
            self.assertEqual(len(series.source["sha256"]), 64)
            path.write_text("value,Label\n10,0\n11,0\n,1\n13,0\n")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                list(load_series(settings, 1))

    def test_invalid_series_and_predictions_fail_explicitly(self):
        with self.assertRaises(ValueError):
            Series("bad", np.ones((3, 1)), np.array([0, -1, 1]), {})
        with self.assertRaises(ValueError):
            difference(np.ones(2), np.ones(3), 1e-5, 1e-5)
        with self.assertRaises(ValueError):
            difference(np.ones(2), np.array([np.nan, 1]), 1e-5, 1e-5)

    def test_single_class_is_not_reported_as_zero_auc(self):
        result = detection_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(result["metric_status"], "single_class")
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["average_precision"])
        result = detection_metrics(np.array([0, 1]), np.array([0.1, 0.9]))
        self.assertEqual(result["roc_auc"], 1)

    def test_tail_replacement_preserves_prefix_and_noops(self):
        baseline = np.arange(5, dtype=float)
        arrays = {f"public_{kind}": baseline.copy() for kind in ("scores", "logits")}
        for variant in ("current", "zero", "compact"):
            for kind in ("scores", "logits"):
                arrays[f"tail_{variant}_{kind}"] = np.array([8., 9.])
        conditions = assemble_conditions(arrays, 2, True)
        np.testing.assert_array_equal(conditions["zero"]["scores"], [0, 1, 2, 8, 9])
        np.testing.assert_array_equal(arrays["public_scores"], baseline)
        no_op = assemble_conditions(arrays, 2, False)
        np.testing.assert_array_equal(no_op["zero"]["scores"], baseline)
        with self.assertRaises(ValueError):
            assemble_conditions(arrays, 3, True)

    def test_selection_is_deterministic_and_uses_no_scores(self):
        candidates = [{"filename": f"file_{i}", "source": source, "stratum": "short",
                       "density_bin": "low", "patch_aligned": False}
                      for i, source in enumerate(("A", "A", "B", "C"))]
        selected = choose(candidates, 2024, {"short": 3})
        self.assertEqual(len({row["source"] for row in selected}), 3)
        self.assertEqual(selected, choose(list(reversed(candidates)), 2024, {"short": 3}))
        with self.assertRaises(ValueError):
            choose(candidates, 2024, {"short": 5})

    def test_frozen_selection_rejects_modified_data(self):
        from experiments.common.artifacts import sha256

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "001_Test_tr_1_1st_2.csv"
            path.write_text("value,Label\n0,0\n1,0\n2,1\n")
            manifest = Path(temporary) / "selection.json"
            manifest.write_text(json.dumps({"selected": [{"filename": path.name,
                                                          "sha256": sha256(path), "length": 2}]}))
            settings = {"kind": "tsb", "root": temporary, "files": [path.name],
                        "split": "test", "selection_manifest": str(manifest)}
            self.assertEqual(len(list(load_series(settings, 0))), 1)
            path.write_text("value,Label\n0,0\n3,0\n2,1\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                list(load_series(settings, 0))

    def test_bad_config_and_duplicate_windows_are_rejected(self):
        template = json.loads((ROOT / "experiments/configs/stage1_smoke.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            checkpoint = Path(temporary) / "weights.pth"
            checkpoint.touch()
            template["checkpoint"] = str(checkpoint)
            for update in ({"window_sizes": [32, 32]}, {"window_sizes": [0]},
                           {"atol": -1}, {"seed": True}, {"unknown": 1}):
                config = {**template, **update}
                path.write_text(json.dumps(config))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_config(path)

    def test_repeated_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"name": "test", "output_root": temporary}
            first, _ = create_run(config, "config.json")
            (first / "sentinel").write_text("original")
            second, _ = create_run(config, "config.json")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "sentinel").read_text(), "original")

    def test_failed_run_preserves_traceback_and_status(self):
        from experiments.stage1.run import main

        with tempfile.TemporaryDirectory() as temporary:
            config = {"name": "failure_test", "output_root": temporary}
            with patch("sys.argv", ["run", "--config", "unused.json"]), \
                    patch("experiments.stage1.run.load_config", return_value=config), \
                    patch("experiments.stage1.run.configure_runtime", return_value={}), \
                    patch("experiments.stage1.run.capture_provenance"), \
                    patch("experiments.stage1.run.run_cases", side_effect=ValueError("bad input")), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "bad input"):
                main()
            directory = next(Path(temporary).iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("finished_utc", manifest)
            self.assertIn("ValueError: bad input", (directory / "error.txt").read_text())


class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 2
        config.ts_config.num_heads = 2
        config.win_size = 32
        config.batch_size = 2
        model = TimeSeriesPretrainModel(config).eval()
        # Supply a tiny, freshly initialized model; unit tests need no checkpoint.
        tester = object.__new__(TimeRCDPretrainTester)
        tester.config = deepcopy(config)
        tester.device = torch.device("cpu")
        tester.win_size = config.win_size
        tester.batch_size = config.batch_size
        tester.model = model
        self.detector = TimeRCDDetector("unused.pth", win_size=32, batch_size=2, device="cpu")
        self.detector._tester = tester
        self.detector._num_features = 1

    def test_real_backends_match_after_short_then_long_series(self):
        rng = np.random.default_rng(3)
        for length in (15, 65, 16, 17, 32, 33):
            with self.subTest(length=length), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                arrays, raw = compare_backends(self.detector, rng.normal(size=(length, 1)))
                self.assertEqual(raw, ((length + min(length, 32) - 1) // min(length, 32)) * min(length, 32))
                for key in ("scores", "logits"):
                    np.testing.assert_allclose(arrays[f"public_{key}"], arrays[f"legacy_{key}"], atol=1e-6)
                self.assertEqual(self.detector._tester.win_size, 32)

    def test_padding_only_affects_valid_partial_patch(self):
        rng = np.random.default_rng(7)
        for length in (48, 49):
            tails, valid = probe_tail(self.detector, rng.normal(size=(length, 1)))
            self.assertEqual(valid, length - 32)
            for key in ("scores", "logits"):
                np.testing.assert_allclose(tails[f"tail_zero_{key}"], tails[f"tail_compact_{key}"], atol=1e-6)
                if length == 48:
                    np.testing.assert_allclose(tails[f"tail_current_{key}"], tails[f"tail_zero_{key}"], atol=1e-6)
            if length == 49:
                self.assertGreater(float(np.max(np.abs(tails["tail_current_logits"] - tails["tail_zero_logits"]))), 1e-6)

    def test_short_forced_padding_preserves_valid_data_and_length(self):
        rng = np.random.default_rng(11)
        for length in (16, 17):
            values = rng.normal(size=(length, 1))
            baseline, logits = self.detector.predict(values, return_logits=True)
            forced = probe_short(self.detector, values)
            self.assertEqual(forced["short_forced_zero"]["scores"].shape, (length,))
            np.testing.assert_allclose(forced["short_forced_zero"]["scores"], baseline, atol=1e-6)
            np.testing.assert_allclose(forced["short_forced_zero"]["logits"], logits, atol=1e-6)
            if length == 16:
                np.testing.assert_allclose(forced["short_forced_edge"]["logits"], logits, atol=1e-6)
        self.assertEqual(probe_short(self.detector, np.ones((32, 1))), {})


if __name__ == "__main__":
    unittest.main()
