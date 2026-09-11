"""Input projection must not silently discard a sensory intervention."""

import unittest

import numpy as np

from tools.probe_c_banc_reference import project_inputs


class ReferenceProjectionTests(unittest.TestCase):
    def test_empty_input_keeps_integer_indices(self):
        targets, values, absent = project_inputs(
            ["10"], ["10"], np.array([], np.int64), np.zeros((2, 0))
        )
        self.assertEqual(targets.dtype, np.dtype("int64"))
        self.assertEqual(values.shape, (2, 0))
        self.assertEqual(absent.shape, (0,))

    def test_permuted_root_ids_and_missing_zero_inputs(self):
        inputs = np.array([[0, 8, 2], [0, 0, 4]], np.float32)
        targets, values, absent = project_inputs(
            ["10", "20", "30", "40"],
            ["40", "20", "50"],
            np.array([0, 3, 1]),
            inputs,
        )
        np.testing.assert_array_equal(targets, [0, 1])
        np.testing.assert_array_equal(values, [[8, 2], [0, 4]])
        np.testing.assert_array_equal(absent, [0])
        values[0] = -99
        self.assertEqual(inputs[0, 1], 8)

    def test_absent_nonzero_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "drops nonzero"):
            project_inputs(
                ["10", "20"], ["20"], np.array([0, 1]), np.array([[0.01, 2]])
            )

    def test_ambiguous_or_malformed_alignment_rejected(self):
        for targets, ids, inputs in (
            (["20", "20"], [0], [[0]]),
            (["20"], [0, 0], [[0, 0]]),
            (["20"], [-1], [[0]]),
            (["20"], [2], [[0]]),
            (["20"], [0.0], [[0]]),
            (["20"], [0], [[float("nan")]]),
        ):
            with (
                self.subTest(targets=targets, ids=ids, inputs=inputs),
                self.assertRaises(ValueError),
            ):
                project_inputs(["10", "20"], targets, np.array(ids), np.array(inputs))
