import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.common.metrics import detection_metrics
from experiments.stage2_alignment_controls.data import generate
from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_p_positions.data import select_positions
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.analyze import analyze_case
from experiments.stage2_rank_localization.data import load_case, load_parent
from experiments.stage2_rank_localization.diagnostics import point_groups, readout, relative_ranks
from experiments.stage2_rank_localization.statistics import paired_readouts


def simple_groups():
    first = np.array([True, False, False, False])
    last = np.array([False, False, True, False])
    return {"F": first, "L": last, "I": np.zeros(4, dtype=bool), "N": ~(first | last), "K": first, "B": last}


class RankReadoutTests(unittest.TestCase):
    def test_hand_computed_thresholds_group_composition_and_midrank(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([.9, .8, .8, .1])
        result = readout(labels, scores, simple_groups(), [.5, .75, 1])
        self.assertEqual(result["r050_precision"], 1)
        self.assertAlmostEqual(result["r075_precision"], 2 / 3)
        self.assertEqual(result["r075_recall"], 1)
        self.assertEqual(result["r075_strict_recall"], .5)
        self.assertEqual(result["r075_fpr"], .5)
        self.assertAlmostEqual(result["r075_N_share"], 1 / 3)
        self.assertEqual(result["r075_F_capture"], 1)
        np.testing.assert_array_equal(relative_ranks(scores), np.array([1, 2.5, 2.5, 4]) / 4)
        self.assertTrue(np.isnan(result["I_rank_mean"]))
        self.assertEqual(result["r075_I_share"], 0)
        self.assertTrue(np.isnan(result["r075_I_capture"]))

    def test_all_ties_not_split_and_empty_role_groups(self):
        groups = simple_groups()
        groups["K"] = groups["F"] | groups["L"]
        groups["B"] = np.zeros(4, dtype=bool)
        result = readout(np.array([1, 0, 1, 0]), np.ones(4), groups, [.1, .5, 1])
        self.assertEqual(result["r010_recall"], 1)
        self.assertEqual(result["r010_precision"], .5)
        self.assertEqual(result["r010_selected_fraction"], 1)
        self.assertTrue(np.isnan(result["B_score_mean"]))
        np.testing.assert_array_equal(relative_ranks(np.ones(4)), np.full(4, .625))

    def test_invalid_scores_labels_groups_and_targets_fail(self):
        groups, labels, scores = simple_groups(), np.array([1, 0, 1, 0]), np.arange(4.)
        bad_groups = {**groups, "N": np.ones(4, dtype=bool)}
        for y, s, g, recalls in ((labels, np.array([1, 2, np.nan, 4]), groups, [.5]),
                                 (np.array([1, 0, 2, 0]), scores, groups, [.5]),
                                 (labels, scores, bad_groups, [.5]), (labels, scores, groups, [0]),
                                 (labels, scores, groups, [.5, .5])):
            with self.assertRaises(ValueError):
                readout(y, s, g, recalls)


class RankStatisticsTests(unittest.TestCase):
    def frame(self):
        rows = []
        values = {"A": .1, "full_last": .2, "full_all": .9,
                  "keep_first_last": .3, "keep_first_all": .4,
                  "keep_last_last": .35, "keep_last_all": .7,
                  "random_2024_last": .2, "random_2024_all": .5,
                  "random_2025_last": .6, "random_2025_all": .9}
        for seed in (1, 2):
            for variant, value in values.items():
                rows.append({"case": f"case{seed}", "seed": seed, "phase": 0, "shape": "positive",
                             "event_length": 32, "offset": 0, "score_kind": "scores", "variant": variant,
                             "metric_status": "ok", "r050_precision": value, "I_rank_mean": np.nan})
        return pd.DataFrame(rows)

    def test_random_metric_average_pair_sign_and_seed_unit(self):
        absolute, effects, columns = paired_readouts(self.frame().sample(frac=1, random_state=1), [2024, 2025])
        self.assertTrue(absolute[absolute.contrast == "random_mean_all"].r050_precision.eq(.7).all())
        self.assertTrue(np.allclose(effects[effects.contrast == "position_by_scope"].r050_precision, .25))
        self.assertTrue(effects.I_rank_mean.isna().all())
        seeds, stats = seed_summary(effects, [], columns, {"synthetic": {"seeds": [1, 2]}, "bootstrap_seed": 2024,
                                                          "bootstrap_repetitions": 100})
        chosen = stats[(stats.contrast == "position_by_scope") & (stats.metric == "r050_precision")].iloc[0]
        self.assertEqual(chosen.seeds, 2)
        self.assertAlmostEqual(chosen["mean"], .25)
        self.assertEqual(len(seeds[seeds.contrast == "position_by_scope"]), 2)

    def test_incomplete_duplicate_and_random_missingness_fail(self):
        frame = self.frame()
        missing = frame.copy()
        missing.loc[missing.variant == "random_2024_all", "r050_precision"] = np.nan
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), missing):
            with self.assertRaises(ValueError):
                paired_readouts(bad, [2024, 2025])


class RankPipelineTests(unittest.TestCase):
    def test_miniature_saved_output_pipeline_and_mask_mismatch(self):
        source = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
        settings = {**source["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                    "phases": [0], "event_lengths": [32], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))[0]
        config = {"patch_size": 16, "random_seeds": [2024, 2025], "recalls": [.1, .5, 1]}
        masks, kept = select_positions(meta, arrays, 16, config["random_seeds"])
        fixed, roles = point_groups(meta, arrays, masks, 16, config["random_seeds"])
        self.assertEqual(fixed["I"].sum(), 0)
        np.testing.assert_array_equal(roles["keep_first"]["K"], fixed["F"])
        np.testing.assert_array_equal(roles["keep_last"]["K"], fixed["L"])
        bad = {**masks, "keep_first": masks["keep_last"]}
        with self.assertRaises(ValueError):
            point_groups(meta, arrays, bad, 16, config["random_seeds"])
        predictions, metric_rows = {}, []
        variants = ["A"] + [f"{s}_{scope}" for s in masks for scope in ("last", "all")]
        for index, variant in enumerate(variants):
            for kind in ("scores", "logits"):
                values = np.round(np.sin(np.arange(512) / 13 + index), 1)
                predictions[f"{variant}_{kind}"] = values
                metric_rows.append({"case": meta["case"], "variant": variant, "score_kind": kind,
                                    **detection_metrics(arrays["labels"], values)})
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            (parent / "arrays").mkdir()
            for kind, values in (("input", arrays), ("selection", {**masks, **{f"kept_{k}": v for k, v in kept.items()}})):
                path = parent / f"arrays/{kind}.npz"
                np.savez_compressed(path, **values)
                meta.update({f"{kind}_snapshot": str(path.relative_to(parent)), f"{kind}_sha256": sha256(path)})
            np.savez_compressed(parent / f"arrays/{meta['case']}.npz", **predictions)
            labels, fixed, roles, predictions = load_case(parent, meta, config)
            audit = Audit(parent, 1e-12)
            metrics = pd.DataFrame(metric_rows).set_index(["case", "variant", "score_kind"])
            rows = analyze_case(meta, labels, fixed, roles, predictions, config, metrics, audit)
            self.assertEqual(len(rows), 22)
            self.assertEqual(audit.count, 22 * 11)
            self.assertTrue(pd.read_csv(parent / "checks.csv").close.all())

    def test_parent_validator_rejects_previously_hashed_artifact_change(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            (parent / "output_validation").mkdir()
            config = {"parent_run": str(parent), "patch_size": 16, "random_seeds": [2024]}
            write_json(parent / "manifest.json", {"status": "completed", "config": config})
            write_json(parent / "output_validation/summary.json", {"status": "passed"})
            artifact = parent / "metrics.csv"
            artifact.write_text("before\n")
            write_json(parent / "output_validation/artifacts.sha256.json", {str(artifact): sha256(artifact)})
            artifact.write_text("after\n")
            with self.assertRaisesRegex(ValueError, "Previously validated artifact changed"):
                load_parent(config)


if __name__ == "__main__":
    unittest.main()
