import unittest

import numpy as np

from experiments.stage2_isolated.boundary_audit import boundary_cases
from experiments.stage2_patch_types.selections import patch_types


class BoundaryTests(unittest.TestCase):
    def test_one_based_singletons(self):
        cases = list(boundary_cases())
        self.assertEqual(len(cases), 3)
        for points, expected, _, labels in cases:
            types = patch_types(labels, np.ones(5000, dtype=bool), 16)
            self.assertEqual(np.flatnonzero(types["U"]).tolist(), expected)
            self.assertEqual(np.flatnonzero(labels).tolist(), [point - 1 for point in points])
            self.assertFalse(types["P"].any())
        self.assertEqual(cases[0][1], [0])
        self.assertEqual(cases[1][1], [1])


if __name__ == "__main__":
    unittest.main()
