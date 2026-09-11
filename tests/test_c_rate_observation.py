import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flylab.c.rate_observation import RateInputObserver


class RateInputObservationTests(unittest.TestCase):
    def setUp(self):
        # Asymmetric connectivity makes an accidental transpose observable.
        self.weights = csr_matrix([[0, 2, -3], [4, 0, 0], [0, -1, 0]], dtype=float)
        self.parameters = (
            np.array([0.02, 0.03, 0.04]),
            np.ones(3),
            np.array([7.5, 8, 9]),
            np.full(3, 200.0),
        )
        self.observer = RateInputObserver(self.weights, *self.parameters, [2, 0])

    def test_subthreshold_inputs_nonlinearity_and_outgoing_cut(self):
        rates = np.array([[10.0, 20.0, 5.0], [0.0, 1.0, 1.0]])
        drives = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
        originals = (rates.copy(), drives.copy(), self.weights.toarray().copy())
        full = self.observer.observe(rates, drives, np.ones(3))
        np.testing.assert_array_equal(full["excitatory_input"], [[0, 40], [0, 2]])
        np.testing.assert_array_equal(full["inhibitory_input"], [[-20, -15], [-1, -3]])
        np.testing.assert_array_equal(
            full["threshold_margin"], [[21, 17.5], [-10, -8.5]]
        )
        expected = np.maximum(
            200 * np.tanh(np.array([[21, 17.5], [-10, -8.5]]) / 200), 0
        )
        np.testing.assert_allclose(full["instantaneous_target_rate"], expected)
        np.testing.assert_allclose(
            full["rate_derivative"], (expected - rates[:, [2, 0]]) / [0.04, 0.02]
        )
        cut = self.observer.observe(rates, drives, [1, 1, 0])
        # Cutting target 2's outgoing edges does not erase its own state/drive.
        np.testing.assert_array_equal(cut["rate"], rates[:, [2, 0]])
        np.testing.assert_array_equal(cut["external_input"], drives[:, [2, 0]])
        np.testing.assert_array_equal(cut["inhibitory_input"], [[-20, 0], [-1, 0]])
        for got, before in zip((rates, drives, self.weights.toarray()), originals):
            np.testing.assert_array_equal(got, before)

    def test_alignment_and_invalid_raw_states_are_rejected(self):
        with self.assertRaises(ValueError):
            RateInputObserver(self.weights.astype(complex), *self.parameters, [0])
        for ids in ([0, 0], [-1], [3], [0.5], []):
            with self.assertRaises(ValueError):
                RateInputObserver(self.weights, *self.parameters, ids)
        rates = np.ones((2, 3))
        for raw, drive, mask in (
            (rates[:, :2], rates, [1, 1, 1]),
            (rates, rates[:1], [1, 1, 1]),
            (rates, rates, [1, 0.5, 1]),
            (rates * np.nan, rates, [1, 1, 1]),
            (-rates, rates, [1, 1, 1]),
            (rates, rates.astype(complex), [1, 1, 1]),
        ):
            with self.assertRaises(ValueError):
                self.observer.observe(raw, drive, mask)


if __name__ == "__main__":
    unittest.main()
