import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flylab.c.rate_observation import RateInputObserver
from tools.analyze_c_late_motor_activity import aligned_states, edge_observations


class LateMotorActivityTests(unittest.TestCase):
    def raw(self):
        raw = {
            "input_indices": np.array([2, 0]),
            "input_drive": np.zeros((600, 2)),
            "output_mask": np.array([1, 1, 0]),
        }
        raw["input_drive"][:200] = [50, 7]
        for stamp in (199, 200, 201, 400, 600):
            raw[f"rate_at_{stamp}ms"] = np.array([stamp, 2, 3], dtype=float)
        return raw

    def test_boundary_uses_interval_start_and_terminal_is_zero_input(self):
        rates, drive, mask = aligned_states(self.raw(), 3)
        np.testing.assert_array_equal(rates[:, 0], [199, 200, 201, 400, 600])
        np.testing.assert_array_equal(drive[0], [7, 0, 50])
        np.testing.assert_array_equal(drive[1:], np.zeros((4, 3)))
        # Input/output cut does not zero the saved state of the muted cell.
        np.testing.assert_array_equal(rates[:, 2], [3] * 5)
        np.testing.assert_array_equal(mask, [1, 1, 0])

    def test_sparse_input_sums_match_independent_edges_with_cancellation(self):
        weights = csr_matrix([[0, 2, -3], [4, 0, 0], [0, -1, 0]], dtype=float)
        parameters = (
            np.array([0.02, 0.03, 0.04]),
            np.ones(3),
            np.array([7.5, 8, 9]),
            np.full(3, 200),
        )
        rates = np.array([[10, 20, 5], [0, 1, 1]], dtype=float)
        drive = np.array([[0, 0, 50], [0, 0, 0]], dtype=float)
        indices = [2, 0]
        for mask in (np.ones(3), np.array([1, 1, 0])):
            observed = RateInputObserver(weights, *parameters, indices).observe(
                rates, drive, mask
            )
            independent, edges = edge_observations(
                weights, parameters, indices, rates, drive, mask
            )
            for key in observed:
                np.testing.assert_allclose(
                    observed[key], independent[key], rtol=0, atol=1e-12
                )
            np.testing.assert_array_equal(edges[0][0], [1, 2])
        self.assertGreater(observed["threshold_margin"][0, 1], 0)
        self.assertEqual(observed["inhibitory_input"][0, 1], 0)

    def test_missing_states_and_input_after_removal_are_invalid(self):
        raw = self.raw()
        del raw["rate_at_201ms"]
        with self.assertRaises(KeyError):
            aligned_states(raw, 3)
        raw = self.raw()
        raw["input_drive"][200, 0] = 1
        with self.assertRaises(ValueError):
            aligned_states(raw, 3)
        raw = self.raw()
        raw["input_indices"] = np.array([0, 0])
        with self.assertRaises(ValueError):
            aligned_states(raw, 3)


if __name__ == "__main__":
    unittest.main()
