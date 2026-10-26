"""The E95 receiver-row patch preserves existing signed topology."""

import unittest
from types import SimpleNamespace

import numpy as np

from tools.prepare_c_banc_m3_RM_e2_source import selected_row_patch


class RMSourcePatchTests(unittest.TestCase):
    def test_only_existing_selected_positive_edge_is_strengthened(self):
        graph = SimpleNamespace(indptr=np.array([0, 0, 3]),
            indices=np.array([4, 5, 6]))
        matrix = SimpleNamespace(data=np.array([.18, -2., .5], dtype=np.float32),
            indices=graph.indices)
        edges, values, chosen, efficacy = selected_row_patch(graph, matrix,
            np.array([], dtype=np.int64), 4, 1)
        np.testing.assert_array_equal(edges, [0, 1, 2])
        np.testing.assert_array_equal(values[1:],
            matrix.data[1:] * np.float32(.001))
        self.assertEqual(chosen, 0)
        self.assertAlmostEqual(float(values[0]), float(np.float32(1.1707944)))
        self.assertTrue(.001 <= efficacy <= 10)

    def test_overlap_or_missing_edge_rejected(self):
        graph = SimpleNamespace(indptr=np.array([0, 0, 1]),
            indices=np.array([4]))
        matrix = SimpleNamespace(data=np.array([.18], dtype=np.float32),
            indices=graph.indices)
        with self.assertRaises(ValueError):
            selected_row_patch(graph, matrix, np.array([0]), 4, 1)
        with self.assertRaises(ValueError):
            selected_row_patch(graph, matrix, np.array([], dtype=np.int64), 5, 1)


if __name__ == '__main__':
    unittest.main()
