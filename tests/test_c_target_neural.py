"""Anatomical port isolation, source-edge preservation and target state contracts."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from flylab.c.target_neural import LEFT, RIGHT, BancTargetEncoder
from flylab.c.target_sense import TargetObservation


def graph_fixture():
    roots = LEFT + RIGHT + ("720575941500851362", "720575941626500746", "1", "2")
    types = (
        (("DNg105",) + ("IN08A006",) * 3 + ("IN19A003",) * 3 + ("IN09A002",) * 3) * 2
        + ("DNg100",) * 2
        + ("motor",) * 2
    )
    nodes = [
        {
            "cell_type": name,
            "nt_type": "GABA" if i < 20 else "ACH",
            "super_class": "motor" if i >= 22 else "intrinsic",
        }
        for i, name in enumerate(types)
    ]
    index = {"flywire:banc:888:" + root: i for i, root in enumerate(roots)}
    return SimpleNamespace(
        n=24,
        hash="fixture",
        nodes=nodes,
        resolve=lambda ids: np.array([index[i] for i in ids]),
    )


class TargetNeuralContracts(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.encoder = BancTargetEncoder(self.graph, forward_drive=5000)

    def test_only_existing_motor_edges_change_without_rewiring(self):
        source = csr_matrix(
            (
                np.array([-0.03, -0.05, -0.1, 0.2], np.float32),
                ([22, 23, 1, 23], [0, 11, 0, 20]),
            ),
            shape=(24, 24),
        )
        before = source.copy()
        result = self.encoder.calibrate_weights(source, np.array([22, 23]))
        np.testing.assert_array_equal(source.data, before.data)
        np.testing.assert_array_equal(result.indptr, source.indptr)
        np.testing.assert_array_equal(result.indices, source.indices)
        self.assertEqual(result[22, 0], source[22, 0] * 64)
        self.assertEqual(result[23, 11], source[23, 11] * 64)
        self.assertEqual(result[1, 0], source[1, 0])
        self.assertEqual(result[23, 20], source[23, 20])

    def test_goal_cue_drives_neural_ports_and_never_motor_cells(self):
        for bearing, active in ((0.4, [4, 6, 11, 13]), (-0.4, [1, 3, 14, 16])):
            drive = self.encoder.drive(TargetObservation(bearing, 6.0, True))
            np.testing.assert_array_equal(drive[list(active)], np.full(4, 5000.0))
            quiet = sorted(set(range(20)) - set(active))
            np.testing.assert_array_equal(drive[quiet], np.full(16, -5000.0))
            np.testing.assert_array_equal(drive[22:], np.zeros(2))
        stop = self.encoder.drive(TargetObservation(0.5, 1.0, True))
        np.testing.assert_array_equal(stop[:20], np.full(20, 5000.0))
        self.assertTrue(self.encoder.braking)
        cut = self.encoder.drive(TargetObservation(0.0, 0.0, False))
        np.testing.assert_array_equal(cut[:20], np.full(20, -5000.0))
        self.assertFalse(self.encoder.braking)
        self.assertEqual(self.encoder.turn, 0)

    def test_checkpoint_keeps_hysteresis_and_rejects_malformed_state(self):
        self.encoder.drive(TargetObservation(0.4, 4.0, True))
        saved = self.encoder.snapshot()
        other = BancTargetEncoder(self.graph, forward_drive=5000)
        other.restore(saved)
        cue = TargetObservation(0.1, 4.0, True)
        np.testing.assert_array_equal(self.encoder.drive(cue), other.drive(cue))
        for key, value in (
            ("turn", True),
            ("turn", 2),
            ("braking", 1),
            ("braking", True),
            ("hash", "wrong"),
        ):
            bad = copy.deepcopy(saved)
            bad[key] = value
            with self.assertRaises(ValueError):
                other.restore(bad)
        self.assertEqual(other.snapshot(), saved)
        with self.assertRaises(ValueError):
            BancTargetEncoder(self.graph).restore(saved)

    def test_wrong_anatomy_and_invalid_sensory_values_are_rejected(self):
        self.graph.nodes[0]["nt_type"] = "ACH"
        with self.assertRaises(ValueError):
            BancTargetEncoder(self.graph)
        for cue in (
            TargetObservation(float("nan"), 2.0, True),
            TargetObservation(0.0, -1.0, True),
            TargetObservation(0.0, 2.0, 1),
        ):
            with self.assertRaises(ValueError):
                self.encoder.drive(cue)


if __name__ == "__main__":
    unittest.main()
