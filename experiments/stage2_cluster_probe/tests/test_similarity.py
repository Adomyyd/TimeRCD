import unittest

import numpy as np

from experiments.stage2_cluster_probe.analysis import patch_correlation, rank_correlation


class RealSimilarityTests(unittest.TestCase):
    def test_identical(self):
        value, status = patch_correlation(np.tile(np.arange(16), (3, 1)))
        self.assertAlmostEqual(value, 1)
        self.assertEqual(status, "ok")

    def test_opposite(self):
        x = np.arange(16)
        self.assertAlmostEqual(patch_correlation(np.array([x, x, -x]))[0], -1 / 3)

    def test_missing(self):
        self.assertIsNone(patch_correlation(np.ones((2, 16)))[0])
        self.assertEqual(patch_correlation(np.ones((3, 16)))[1], "constant_P_segment")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            patch_correlation(np.ones((3, 15)))

    def test_rank_ties(self):
        self.assertAlmostEqual(rank_correlation([1, 1, 3], [3, 3, 1]), -1)

    def test_rank_invalid(self):
        for a, b in [([1, 1, 1], [1, 2, 3]), ([1, np.nan, 3], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                rank_correlation(a, b)


if __name__ == "__main__":
    unittest.main()
