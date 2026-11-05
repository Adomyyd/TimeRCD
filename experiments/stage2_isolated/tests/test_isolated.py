import json
from pathlib import Path
import unittest

import numpy as np

from experiments.stage2_isolated.data import generate_isolated, random_selection
from experiments.stage2_isolated.statistics import decision, interval, readout


class IsolatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("experiments/stage2_isolated/configs/isolated_v1.json").read_text())
        cls.cases = list(generate_isolated(cls.config))

    def test_design(self):
        self.assertEqual(len(self.cases), 40)
        self.assertEqual(len({r["case"] for r, _ in self.cases}), 40)
        for record, arrays in self.cases:
            self.assertEqual(arrays["labels"].sum(), 16)
            self.assertEqual(np.flatnonzero(arrays["type_P"]).tolist(), [record["position"]])
            self.assertFalse(arrays["type_M"].any())

    def test_shared_background_and_normalization(self):
        for start in range(0, 40, 5):
            for _, arrays in self.cases[start:start + 5]:
                np.testing.assert_array_equal(arrays["clean"], self.cases[start][1]["clean"])
                self.assertAlmostEqual(arrays["corrupt_normalized"].mean(), 0)
                self.assertAlmostEqual(arrays["corrupt_normalized"].std(), 1)

    def test_random_normal_only_deterministic(self):
        for record, arrays in self.cases:
            types = {k.removeprefix("type_"): v for k, v in arrays.items() if k.startswith("type_")}
            for k in range(1, 6):
                chosen = random_selection(types, record["seed"], record["position"], k)
                np.testing.assert_array_equal(chosen, arrays[f"selection_D_rand_{k}"])
                self.assertEqual(chosen.sum(), 1)
                self.assertFalse((chosen & (~types["N"] | types["partial"])).any())

    def test_random_invalid_count(self):
        types = {"N": np.zeros(3, dtype=bool), "partial": np.zeros(3, dtype=bool)}
        with self.assertRaises(ValueError):
            random_selection(types, 1, 1, 1)

    def test_dose_selections(self):
        record, arrays = self.cases[0]
        types = {k.removeprefix("type_"): v for k, v in arrays.items() if k.startswith("type_")}
        for count in (1, 16, 64):
            chosen = random_selection(types, record["seed"], record["position"], 1, count)
            self.assertEqual(chosen.sum(), count)
            self.assertFalse((chosen & ~types["N"]).any())

    def test_readout_and_ties(self):
        record, arrays = self.cases[0]
        result = readout(arrays["labels"], arrays["labels"].astype(float), record["position"])
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["average_precision"], 1)
        self.assertEqual(result["anomaly_mean"], 1)
        result = readout(arrays["labels"], np.zeros(5000), record["position"])
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["max_anomaly_percentile"], .5)

    def test_bootstrap(self):
        a, b = interval(np.arange(8)), interval(np.arange(8))
        self.assertEqual(a, b)
        self.assertEqual(a["mean"], 3.5)
        self.assertEqual(interval(np.zeros(8)), {"mean": 0., "low": 0., "high": 0.})

    def test_decisions(self):
        negative = {"mean": -1, "low": -2, "high": -.1}
        positive = {"mean": 1, "low": .1, "high": 2}
        null = {"mean": 0, "low": -.1, "high": .1}
        self.assertEqual(decision(negative, negative, positive), "pause_H_isolated_specific_harm")
        self.assertEqual(decision(null, null, null), "no_extra_harm_detected_not_equivalence")
        self.assertEqual(decision(positive, positive, positive), "inconclusive_H_remains_locked")


if __name__ == "__main__":
    unittest.main()
