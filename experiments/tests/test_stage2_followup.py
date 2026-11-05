from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from experiments.common.data import Series
from experiments.stage2.inference import windows
from experiments.stage2.synthetic import generate
from experiments.stage2_followup.diagnostics import concordance, ranking_decomposition, shift_statistics
from experiments.stage2_followup.analyze import normalization_pairs
from time_rcd._inference import _WindowDataset


class FollowupTests(unittest.TestCase):
    def test_crossover_pairs_by_identity_and_computes_difference_of_effects(self):
        common = {"case": "s1_w5000", "series": "synthetic", "anomaly_count": 500,
                  "generator_seed": 71, "placement": "early"}
        old = pd.DataFrame([{**common, "baseline_ap": .2, "oracle_ap": .6, "random_mean_ap": .3,
                             "oracle_minus_baseline_ap": .4, "oracle_minus_random_ap": .3}])
        new = pd.DataFrame([{**common, "baseline_ap": .3, "oracle_ap": .5, "random_mean_ap": .3,
                             "oracle_minus_baseline_ap": .2, "oracle_minus_random_ap": .2}])
        result = normalization_pairs(old, new)
        self.assertAlmostEqual(result.iloc[0].interaction_oracle_minus_baseline_ap, -.2)
        self.assertAlmostEqual(result.iloc[0].interaction_oracle_minus_random_ap, -.1)
        with self.assertRaises(ValueError):
            normalization_pairs(old, new.assign(placement="late"))

    def test_exact_pair_decomposition_with_ties_and_single_class_windows(self):
        labels = np.array([0, 1, 0, 1, 0, 0, 1])
        scores = np.array([.2, .2, .5, .9, .3, .3, .7])
        groups = np.array([0, 0, 0, 1, 2, 2, 3])
        correct, pairs = concordance(labels, scores)
        brute = [float(a > b) + .5 * float(a == b)
                 for a in scores[labels == 1] for b in scores[labels == 0]]
        self.assertEqual(pairs, len(brute))
        self.assertEqual(correct, sum(brute))
        result = ranking_decomposition(labels, scores, groups)
        self.assertAlmostEqual(result["auc"], roc_auc_score(labels, scores))
        self.assertAlmostEqual(result["within_contribution"] + result["across_contribution"], result["auc"])
        single = ranking_decomposition(labels, scores, np.zeros(len(labels)))
        self.assertEqual(single["across_pairs"], 0)
        self.assertIsNone(single["across_auc"])
        self.assertIsNone(ranking_decomposition(np.zeros(3), np.ones(3), np.zeros(3))["auc"])

    def test_cross_window_shift_can_hurt_global_ranking_without_local_change(self):
        labels = np.array([0, 1, 0, 1])
        groups = np.array([0, 0, 1, 1])
        before = ranking_decomposition(labels, np.array([0., 1., 0., 1.]), groups)
        after = ranking_decomposition(labels, np.array([10., 11., 0., 1.]), groups)
        self.assertEqual(before["within_auc"], after["within_auc"])
        self.assertLess(after["across_auc"], before["across_auc"])
        self.assertLess(after["auc"], before["auc"])

    def test_paired_shift_and_empty_group(self):
        result = shift_statistics(np.array([1., 2., 8.]), np.array([0., 4., 1.]), np.array([True, True, False]))
        self.assertEqual(result["mean_shift"], .5)
        self.assertEqual(result["fraction_decreased"], .5)
        self.assertEqual(result["points"], 2)
        self.assertIsNone(shift_statistics(np.ones(2), np.ones(2), np.zeros(2, dtype=bool))["mean_shift"])
        with self.assertRaises(ValueError):
            concordance(np.array([0, 2]), np.ones(2))

    def test_official_crossover_changes_only_normalization(self):
        settings = json.loads(Path("experiments/configs/stage2_oracle_v1.json").read_text())["synthetic"]
        original = list(generate(settings))[2]
        source = deepcopy(original.source)
        source["normalization"] = "official_series"
        crossed = Series(original.name, original.values.copy(), original.labels.copy(), source)
        actual = list(windows(crossed, 256))
        official = _WindowDataset(original.values, 256)
        self.assertEqual(len(actual), len(official))
        for index, (values, valid) in enumerate(actual):
            expected_values, expected_valid = official[index]
            torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
            torch.testing.assert_close(valid, expected_valid, rtol=0, atol=0)
        fixed, _ = next(windows(original, 256))
        self.assertGreater(float((fixed - actual[0][0]).abs().max()), .1)
        np.testing.assert_array_equal(original.labels, crossed.labels)
        np.testing.assert_array_equal(original.values, crossed.values)
        crossed.source["normalization"] = "typo"
        with self.assertRaises(ValueError):
            list(windows(crossed, 256))


if __name__ == "__main__":
    unittest.main()
