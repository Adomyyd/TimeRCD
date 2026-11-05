import unittest

import pandas as pd

from experiments.stage1.analyze import bootstrap_mean, window_ranges


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_is_repeatable_at_series_level(self):
        first = bootstrap_mean([0.1, -0.2, 0.4], seed=7)
        self.assertEqual(first, bootstrap_mean([0.1, -0.2, 0.4], seed=7))
        self.assertAlmostEqual(first["mean"], 0.1)
        self.assertLessEqual(first["ci95_low"], first["mean"])
        self.assertGreaterEqual(first["ci95_high"], first["mean"])
        with self.assertRaises(ValueError):
            bootstrap_mean([])

    def test_ranges_use_each_series_and_require_every_window(self):
        rows = [{"series": name, "score_kind": "scores", "variant": "current",
                 "configured_window": window, "average_precision": ap}
                for name, values in (("a", [0.1, 0.9]), ("b", [0.7, 0.6]))
                for window, ap in zip([128, 5000], values)]
        frame = pd.DataFrame(rows)
        result = window_ranges(frame, [128, 5000]).set_index("series")
        self.assertAlmostEqual(result.loc["a", "ap_range"], 0.8)
        self.assertAlmostEqual(result.loc["b", "ap_range"], 0.1)
        self.assertEqual(result.loc["b", "best_window"], 128)
        with self.assertRaises(ValueError):
            window_ranges(frame.iloc[:-1], [128, 5000])


if __name__ == "__main__":
    unittest.main()
