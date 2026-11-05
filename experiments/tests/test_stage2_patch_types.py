from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.data import Series
from experiments.stage1.inference import score_window
from experiments.stage2.inference import windows
from experiments.stage2_paths.attention import attention_edges, model_fingerprint
from experiments.stage2_patch_types.selections import patch_types, split_random, make_selections
from experiments.stage2_patch_types.inference import run_case
from experiments.stage2_patch_types.analyze import contrasts, aggregate
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig


class PatchTypeTests(unittest.TestCase):
    def test_mapping_uses_only_valid_points_including_partial(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1])
        types = patch_types(labels, np.arange(10) < 7, 2)
        np.testing.assert_array_equal(types["N"], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(types["P"], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(types["M"], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(types["partial"], [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(types["U"], types["M"] | types["P"])
        self.assertFalse((types["M"] & types["P"]).any())
        for labels in ([0, 2], [0, np.nan]):
            with self.assertRaises(ValueError):
                patch_types(labels, np.ones(2, dtype=bool), 2)
        with self.assertRaises(ValueError):
            patch_types([0, 1], [1, 1], 2)
        self.assertFalse(patch_types(np.zeros(3), np.ones(3, dtype=bool), 2)["U"].any())
        self.assertTrue(patch_types(np.ones(3), np.ones(3, dtype=bool), 2)["P"].all())

    def test_random_partition_is_paired_repeatable_and_count_matched(self):
        union = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
        for count in (0, 2, 4):
            first = split_random(union, count, 71, "series", 0)
            second = split_random(union, count, 71, "series", 0)
            for t in "MPU":
                np.testing.assert_array_equal(first[t], second[t])
            self.assertEqual(first["M"].sum(), count)
            self.assertEqual(first["P"].sum(), 4 - count)
            self.assertFalse((first["M"] & first["P"]).any())
            np.testing.assert_array_equal(first["M"] | first["P"], union)

    def test_parent_mismatch_and_empty_references_stop(self):
        types = patch_types([0, 0, 1, 0], np.ones(4, dtype=bool), 2)
        saved = {"w0_valid": types["valid"], "w0_clean": types["N"],
                 "w0_random_1": np.array([False, True])}
        selections = make_selections(types, saved, 0, [1], "test")
        self.assertFalse(selections["P_random_1"].any())
        np.testing.assert_array_equal(selections["M_random_1"], [True, False])
        saved["w0_random_1"] = np.ones(2, dtype=bool)
        with self.assertRaises(ValueError):
            make_selections(types, saved, 0, [1], "test")
        full = patch_types(np.ones(4), np.ones(4, dtype=bool), 2)
        with self.assertRaisesRegex(ValueError, "empty reference"):
            make_selections(full, {"w0_valid": full["valid"], "w0_clean": full["N"]}, 0, [], "test")

    def test_miniature_model_full_pipeline_and_parent_regression(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 2
        config.ts_config.num_heads = 2
        model = TimeSeriesPretrainModel(config).eval()
        before = model_fingerprint(model)
        detector = SimpleNamespace(_tester=SimpleNamespace(model=model, device="cpu"))
        settings = {"random_seeds": [1, 2], "atol": 1e-6, "rtol": 1e-5}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "arrays").mkdir()
            for index, pure in enumerate((False, True)):
                labels = np.zeros(81, dtype=int)
                labels[17] = 1
                if pure:
                    labels[32:48] = 1
                    labels[80] = 1
                series = Series(f"00{index}_Test_tr_1", np.random.default_rng(3).normal(size=(81, 1)), labels, {"kind": "tsb"})
                values, valid = next(windows(series, 5000))
                types = patch_types(labels, valid.numpy(), 16)
                old = {"w0_valid": types["valid"], "w0_clean": types["N"]}
                for seed in settings["random_seeds"]:
                    selected = np.random.default_rng(seed).choice(np.flatnonzero(types["valid"]), int(types["U"].sum()), replace=False)
                    keep = types["valid"].copy()
                    keep[selected] = False
                    old[f"w0_random_{seed}"] = keep
                selections = make_selections(types, old, 0, settings["random_seeds"], series.name)
                baseline = score_window(model, values, valid, "cpu")
                parent = {f"A_{k}": v for k, v in baseline.items()}
                for selector in ("label", "random_1", "random_2"):
                    for c in "CD":
                        with attention_edges(model, types["valid"], selections[f"U_{selector}"], c, "all"):
                            pred = score_window(model, values, valid, "cpu")
                        parent.update({f"{selector}__all__{c}_{k}": v for k, v in pred.items()})
                path = directory / f"parent{index}.npz"
                np.savez_compressed(path, **parent)
                item = (index, series, {0: (types, selections)}, path)
                gate = directory / f"gate{index}"
                (gate / "arrays").mkdir(parents=True)
                self.assertEqual(run_case(gate, settings, detector, item, gate=True), 19)
                self.assertEqual(run_case(directory, settings, detector, item), 19)
                with np.load(directory / f"arrays/s{index:04d}_w5000.npz") as result:
                    self.assertEqual(result["M_label__C_scores"].shape, (81,))
                    if not pure:
                        np.testing.assert_array_equal(result["P_label__C_scores"], result["A_scores"])
                        np.testing.assert_array_equal(result["M_label__D_scores"], result["U_label__D_scores"])
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())
            metrics = pd.read_csv(directory / "metrics.csv")
            seeds, paired = contrasts(metrics, [1, 2])
            self.assertEqual(len(seeds), 6)
            self.assertEqual(len(paired), 2)
            self.assertEqual(model_fingerprint(model), before)

    def test_analysis_paired_interaction_missing_and_duplicate_conditions(self):
        base = {"case": "s0", "series": "test", "kind": "tsb", "source": "Test", "generator_seed": 1,
                "placement": "early", "anomaly_count": 2, "length": 64, "mixed_patches": 1, "pure_patches": 1,
                "score_kind": "scores", "metric_status": "ok"}
        rows = [{**base, "variant": "A", "average_precision": .5}]
        for selector, shift in (("label", .1), ("random_1", .02), ("random_2", .04)):
            for t, value in zip("MPU", [.3, .4, .2]):
                for c, diff in (("C", 0), ("D", -.05)):
                    rows.append({**base, "variant": f"{t}_{selector}__{c}", "average_precision": value + shift + diff})
        frame = pd.DataFrame(rows)
        _, paired = contrasts(frame, [1, 2])
        row = paired.iloc[0]
        self.assertAlmostEqual(row.label_M_C_minus_A, -.1)
        self.assertAlmostEqual(row.label_minus_random_M_C_minus_A, .07)
        self.assertAlmostEqual(row.label_interaction_C, -.1)
        for bad in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                contrasts(bad, [1, 2])
        grouped = pd.concat([paired.assign(generator_seed=seed, placement=layout) for seed in [1, 2, 3] for layout in ["early", "late", "spread"]])
        stats = aggregate(grouped, "generator_seed", {"bootstrap_seed": 2024, "bootstrap_repetitions": 100})
        self.assertEqual(stats["units"], 3)
        self.assertAlmostEqual(stats["label_M_C_minus_A"]["mean"], -.1)


if __name__ == "__main__":
    unittest.main()
