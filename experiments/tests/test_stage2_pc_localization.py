import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.stage2_pc_localization.diagnostics import GROUPS, pair_decomposition, point_partition
from experiments.stage2_pc_localization.statistics import paired_statistics, summarize


class PCLocalizationTests(unittest.TestCase):
    def test_window_patch_reset_and_partial_valid_points(self):
        labels = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
        groups, membership, windows = point_partition(labels, 7, 4)
        np.testing.assert_array_equal(np.flatnonzero(groups["P_anomaly"]), [0, 1, 2, 3, 7, 8, 9, 10])
        np.testing.assert_array_equal(np.flatnonzero(groups["M_anomaly"]), [5])
        np.testing.assert_array_equal(np.flatnonzero(groups["M_normal"]), [4, 6])
        np.testing.assert_array_equal(np.flatnonzero(groups["N_normal"]), [11, 12])
        np.testing.assert_array_equal(membership, [0] * 7 + [1] * 6)
        np.testing.assert_array_equal(windows[1][2]["valid_counts"], [4, 2])
        groups, _, _ = point_partition(np.array([0, 0, 0, 0, 1]), 7, 4)
        self.assertEqual(int(groups["P_anomaly"].sum()), 1)

    def test_decomposition_matches_enumerated_pairs_with_ties(self):
        rng = np.random.default_rng(17)
        labels = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
        groups, membership, _ = point_partition(labels, 7, 2)
        total_pairs = int(labels.sum() * (1 - labels).sum())
        for _ in range(20):
            scores = rng.integers(0, 4, size=len(labels)).astype(float)
            rows = pair_decomposition(labels, scores, groups, membership)
            for row in rows:
                p, n = row["pair"].split("__")
                correct, count = 0.0, 0
                for i in np.flatnonzero(groups[p]):
                    for j in np.flatnonzero(groups[n]):
                        within = membership[i] == membership[j]
                        if row["scope"] == "within" and not within or row["scope"] == "across" and within:
                            continue
                        count += 1
                        correct += float(scores[i] > scores[j]) + .5 * float(scores[i] == scores[j])
                self.assertEqual(row["pairs"], count)
                self.assertEqual(row["correct"], correct)
                self.assertEqual(row["contribution"], correct / total_pairs)
                if count:
                    self.assertEqual(row["auc"], correct / count)
                else:
                    self.assertIsNone(row["auc"])
            for scopes in (("all",), ("within", "across")):
                self.assertAlmostEqual(sum(r["contribution"] for r in rows if r["scope"] in scopes), roc_auc_score(labels, scores))

    def test_absent_group_zero_contribution_missing_conditional_auc(self):
        labels = np.array([1, 0, 0, 0])
        groups, membership, _ = point_partition(labels, 4, 2)
        rows = pair_decomposition(labels, np.ones(4), groups, membership)
        for row in rows:
            if row["pair"].startswith("P_anomaly") or row["scope"] == "across":
                self.assertEqual(row["contribution"], 0)
                self.assertIsNone(row["auc"])
        self.assertEqual(sum(r["contribution"] for r in rows if r["scope"] == "all"), .5)

    def test_invalid_partitions_and_inputs_fail(self):
        labels = np.array([1, 0, 0, 0])
        groups, membership, _ = point_partition(labels, 4, 2)
        wrong = {k: v.copy() for k, v in groups.items()}
        wrong["P_anomaly"][0] = True
        with self.assertRaises(ValueError):
            pair_decomposition(labels, np.ones(4), wrong, membership)
        with self.assertRaises(ValueError):
            pair_decomposition(labels, np.array([np.nan, 0, 0, 0]), groups, membership)
        with self.assertRaises(ValueError):
            point_partition(np.array([0, 2]), 4, 2)

    def test_random_metrics_are_averaged_after_scoring_and_paired_by_case(self):
        labels = np.array([1, 0, 0])
        predictions = [np.array([.9, .8, .1]), np.array([.1, .2, .9])]
        aps = [average_precision_score(labels, p) for p in predictions]
        rows = []
        for case, shift in (("a", 0), ("b", .1)):
            for selector, value in (("label", .7 + shift), ("random_1", aps[0] + shift), ("random_2", aps[1] + shift)):
                rows.append({"case": case, "series": case, "source": "src", "has_P": True,
                             "score_kind": "scores", "selector": selector, "ap": value})
        frame = pd.DataFrame(rows).sample(frac=1, random_state=2)
        result = paired_statistics(frame, [], ["ap"], ["label", "random_1", "random_2"]).set_index("case")
        self.assertAlmostEqual(result.loc["a", "random_ap"], np.mean(aps))
        self.assertNotAlmostEqual(result.loc["a", "random_ap"], average_precision_score(labels, np.mean(predictions, axis=0)))
        self.assertAlmostEqual(result.loc["a", "label_minus_random_ap"], result.loc["b", "label_minus_random_ap"])
        with self.assertRaises(ValueError):
            paired_statistics(pd.concat([frame, frame.iloc[:1]]), [], ["ap"], ["label", "random_1", "random_2"])
        with self.assertRaises(ValueError):
            paired_statistics(frame.iloc[1:], [], ["ap"], ["label", "random_1", "random_2"])

    def test_summary_keeps_zero_contributions_but_excludes_missing_groups(self):
        frame = pd.DataFrame([
            {"case": "a", "has_P": True, "score_kind": "scores", "delta": -.2, "shift": -.5},
            {"case": "b", "has_P": False, "score_kind": "scores", "delta": 0, "shift": np.nan},
        ])
        summary = summarize(frame, [], ["delta", "shift"], {"bootstrap_seed": 2024, "bootstrap_repetitions": 100})
        all_rows = summary[summary.subset == "all"].set_index("statistic")
        self.assertAlmostEqual(all_rows.loc["delta", "mean"], -.1)
        self.assertEqual(all_rows.loc["delta", "present_cases"], 2)
        self.assertEqual(all_rows.loc["shift", "present_cases"], 1)
        absent = summary[(summary.subset == "without_P") & (summary.statistic == "shift")].iloc[0]
        self.assertEqual(absent.present_cases, 0)
        self.assertTrue(pd.isna(absent["mean"]))


if __name__ == "__main__":
    unittest.main()
