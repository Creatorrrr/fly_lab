import unittest

import numpy as np

from flylab.c.navigation_tasks import evaluate_navigation


class NavigationTasksTests(unittest.TestCase):
    def trace(self, x):
        return [
            {
                "simTime": i * 0.005,
                "position": [float(v), 0.7, 0.0],
                "contact": False,
                "fault": None,
            }
            for i, v in enumerate(x)
        ]

    def test_physical_arrival_and_dwell(self):
        rows = self.trace(np.r_[np.linspace(0, 8, 401), np.full(301, 8.0)])
        result = evaluate_navigation(rows, [8.0, 0.0])
        self.assertEqual(result["task_status"], "PASS")
        self.assertLessEqual(result["stop_path_mm"], 0.5)

    def test_crossing_or_policy_arrived_flag_does_not_pass(self):
        rows = self.trace(np.linspace(0, 126, 6301))
        for r in rows:
            r["arrived"] = True
        self.assertEqual(evaluate_navigation(rows, [8.0, 0.0])["task_status"], "FAIL")

    def test_no_approach_and_missing_samples(self):
        rows = self.trace(np.zeros(401))
        self.assertEqual(
            evaluate_navigation(rows, [0.0, 0.0])["task_status"], "NOT_APPLICABLE"
        )
        self.assertEqual(
            evaluate_navigation(rows[::2], [8.0, 0.0])["technical_status"], "INCOMPLETE"
        )

    def test_collision_disqualifies_apparent_arrival(self):
        rows = self.trace(np.r_[np.linspace(0, 8, 401), np.full(301, 8.0)])
        rows[100]["contact"] = True
        self.assertEqual(evaluate_navigation(rows, [8.0, 0.0])["task_status"], "FAIL")
