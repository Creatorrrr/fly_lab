"""Physical dwell confirmation and complete target checkpoint validation."""

import copy
import unittest

import numpy as np

from flylab.c.target_navigation import BancTargetNavigation
from tests.test_c_target_neural import graph_fixture


class TargetNavigationContracts(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.config = {"target_xz_mm": [6.0, 0.0], "forward_drive": 5000.0}
        self.navigation = BancTargetNavigation(self.graph, self.config)
        self.navigation.set_target(
            [6.0, 0.0], 0.0, np.array([0.0, 0.0, 1.0]), np.eye(3)
        )

    def test_brake_request_does_not_count_as_physical_arrival(self):
        nav = self.navigation
        nav.drive(np.array([6.0, 0.0, 1.0]), np.eye(3))
        self.assertTrue(nav.encoder.braking)
        self.assertFalse(
            nav.view(np.array([6.0, 0.0, 1.0]), np.eye(3))["physical_hold_confirmed"]
        )
        for step in range(1, 302):
            nav.record(step * 0.005, [6.0, 0.0, 1.0], True)
        self.assertTrue(nav.view([6.0, 0.0, 1.0], np.eye(3))["physical_hold_confirmed"])
        nav.record(1.51, [6.0, 0.0, 1.0], False)
        self.assertFalse(
            nav.view([6.0, 0.0, 1.0], np.eye(3))["physical_hold_confirmed"]
        )

    def test_motion_inside_target_radius_does_not_count_as_stopped(self):
        for step in range(1, 302):
            position = [6.0 + 0.4 * np.sin(step * 0.08), 0.4 * np.cos(step * 0.08), 1.0]
            self.navigation.record(step * 0.005, position, True)
        observation = self.navigation.view(position, np.eye(3))
        self.assertLess(observation["distance_mm"], 2.0)
        self.assertGreater(observation["stop_window_path_mm"], 0.5)
        self.assertFalse(observation["physical_hold_confirmed"])

    def test_restore_keeps_cue_policy_and_history_and_rejects_bad_clock(self):
        nav = self.navigation
        nav.drive(np.array([0.0, 2.0, 1.0]), np.eye(3))
        nav.record(0.005, [0.0, 2.0, 1.0], True)
        saved = nav.snapshot()
        restored = BancTargetNavigation(self.graph, self.config)
        restored.restore(saved, time_s=0.005)
        np.testing.assert_array_equal(
            nav.drive([0.0, 2.0, 1.0], np.eye(3)),
            restored.drive([0.0, 2.0, 1.0], np.eye(3)),
        )
        self.assertEqual(nav.snapshot(), restored.snapshot())
        before = restored.snapshot()
        for bad in (
            dict(saved, start_time=float("nan")),
            dict(saved, cut=1),
            dict(saved, history=[]),
        ):
            with self.assertRaises(ValueError):
                restored.restore(bad, time_s=0.005)
        bad = copy.deepcopy(saved)
        bad["history"][-1][0] = 0.006
        with self.assertRaises(ValueError):
            restored.restore(bad, time_s=0.005)
        self.assertEqual(before, restored.snapshot())

    def test_target_change_resets_only_goal_history_and_invalid_change_is_atomic(self):
        self.navigation.record(0.005, [0.1, 0.0, 1.0], True)
        before = self.navigation.snapshot()
        with self.assertRaises(ValueError):
            self.navigation.set_target(
                [float("nan"), 0.0], 0.005, [0.1, 0.0, 1.0], np.eye(3)
            )
        self.assertEqual(before, self.navigation.snapshot())
        self.navigation.set_target([-6.0, 0.0], 0.005, [0.1, 0.0, 1.0], np.eye(3))
        self.assertEqual(len(self.navigation.history), 1)
        self.assertEqual(self.navigation.sensor.target_xz, (-6.0, 0.0))


if __name__ == "__main__":
    unittest.main()
