import copy
import importlib.util
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flylab.c.research_rate import ResearchRateNetwork


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch runtime")
class ResearchRateContract(unittest.TestCase):
    def setUp(self):
        self.network = ResearchRateNetwork(
            csr_matrix(np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], np.float32)),
            np.full(3, 0.02),
            np.ones(3),
            np.full(3, 7.5),
            np.full(3, 200),
            device="cpu",
        )
        self.drive = np.array([50, 0, 0], np.float32)

    def test_outgoing_cut_keeps_source_response_and_stops_downstream_recruitment(self):
        n = self.network
        n.advance(self.drive, 400)
        connected = n.readout()
        self.assertGreater(connected[1], 1)
        n.reset()
        n.set_muted([0])
        n.advance(self.drive, 400)
        cut = n.readout()
        self.assertEqual(cut[0], connected[0])
        np.testing.assert_array_equal(cut[1:], [0, 0])

    def test_checkpoint_preserves_held_drive_cut_and_future(self):
        n = self.network
        n.set_muted([2])
        n.advance(self.drive, 20)
        saved = n.snapshot()
        n.advance(self.drive * 0.5, 30)
        expected = n.snapshot()
        n.reset()
        n.restore(saved)
        n.advance(self.drive * 0.5, 30)
        actual = n.snapshot()
        self.assertEqual(expected["tick"], actual["tick"])
        for key in ("rate", "drive", "output_mask"):
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_invalid_drive_and_checkpoint_do_not_mutate_state(self):
        n = self.network
        n.advance(self.drive, 10)
        saved = n.snapshot()
        for drive in (
            np.ones(2),
            np.ones(3, dtype=bool),
            np.full(3, np.nan),
            np.full(3, 1e100),
        ):
            with self.assertRaises(ValueError):
                n.advance(drive)
        invalid = copy.deepcopy(saved)
        invalid["output_mask"][1] = 0.5
        with self.assertRaises(ValueError):
            n.restore(invalid)
        actual = n.snapshot()
        self.assertEqual(saved["tick"], actual["tick"])
        for key in ("rate", "drive", "output_mask"):
            np.testing.assert_array_equal(saved[key], actual[key])

    def test_fast_profile_matches_analytic_rate_rise_and_decay(self):
        n = ResearchRateNetwork(
            csr_matrix((1, 1), dtype=np.float32),
            np.array([0.002]),
            np.ones(1),
            np.array([7.5]),
            np.array([200.0]),
            device="cpu",
            dt=0.00002,
        )
        steady = 200 * np.tanh((50 - 7.5) / 200)
        n.advance(np.array([50.0]), 50)
        expected = steady * (1 - np.exp(-0.001 / 0.002))
        self.assertAlmostEqual(float(n.readout()[0]), expected, delta=5e-5)
        n.advance(np.array([0.0]), 50)
        self.assertAlmostEqual(
            float(n.readout()[0]), expected * np.exp(-0.5), delta=5e-5
        )
        self.assertAlmostEqual(n.tick * n.dt, 0.002)
