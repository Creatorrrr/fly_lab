import unittest

import numpy as np

from tools.probe_c_banc_gate import remove_external_dn


class ExternalDescendingGateTests(unittest.TestCase):
    def test_only_the_named_external_channel_changes_without_mutating_saved_input(self):
        ids = np.array([17, 4, 9], np.int64)
        values = np.array([[2, 400, 7], [3, 0, 8]], np.float32)
        before = values.copy()
        result = remove_external_dn(ids, values, 4)
        np.testing.assert_array_equal(result[:, [0, 2]], before[:, [0, 2]])
        self.assertTrue(np.all(result[:, 1] == 0))
        np.testing.assert_array_equal(values, before)
        self.assertFalse(np.shares_memory(result, values))
        for invalid_ids, invalid_values in (
            (np.array([17, 4, 4]), values),
            (np.array([17, 3, 9]), values),
            (ids, np.array([[2, float("nan"), 7]], np.float32)),
            (ids, np.zeros((2, 2), np.float32)),
        ):
            with self.assertRaises(ValueError):
                remove_external_dn(invalid_ids, invalid_values, 4)
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
