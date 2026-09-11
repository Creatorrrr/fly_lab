import math
import unittest

import numpy as np

from flylab.c.receptors import ClawPositionTuning, JointReceptors, ReceptorParameters


class ClawTuningTests(unittest.TestCase):
    def setUp(self):
        self.parameters = ReceptorParameters(0.4789, 2.502)

    def test_rest_is_quiet_and_opponent_position_inputs_are_separate(self):
        tuning = ClawPositionTuning()
        np.testing.assert_array_equal(tuning.encode(math.pi / 2), [0, 0])
        np.testing.assert_array_equal(tuning.encode(math.pi / 10), [1, 0])
        np.testing.assert_array_equal(tuning.encode(math.pi), [0, 1])
        for angle in np.linspace(0, math.pi, 101):
            flexion, extension = tuning.encode(float(angle))
            self.assertEqual(flexion * extension, 0)
            self.assertGreaterEqual(flexion + extension, 0)
            self.assertLessEqual(flexion + extension, 1)
        receptors = JointReceptors(self.parameters, claw_tuning=tuning)
        for _ in range(200):
            sample = receptors.step(1.64, 0, interior_angle_rad=math.pi / 2)
            self.assertEqual(sum(sample["output"].values()), 0)

    def test_invalid_geometry_does_not_advance_receptor_state(self):
        receptors = JointReceptors(self.parameters, claw_tuning=ClawPositionTuning())
        receptors.step(1.4, 2, interior_angle_rad=1.8)
        before = receptors.snapshot()
        for invalid in (None, True, float("nan"), float("inf"), -0.01, math.pi + 0.01):
            with self.assertRaises(ValueError):
                receptors.step(1.4, 2, interior_angle_rad=invalid)
            after = receptors.snapshot()
            self.assertEqual(after["tick"], before["tick"])
            self.assertEqual(after["velocity_lowpass"], before["velocity_lowpass"])
            for key in ("filtered", "queue"):
                np.testing.assert_array_equal(before[key], after[key])

    def test_legacy_identity_and_new_profile_checkpoint_are_distinct(self):
        legacy = JointReceptors(self.parameters)
        self.assertEqual(
            legacy.identity,
            "e0ab21b494aebdfcaabf84a3a0fee2cdf31e7eb9e65e99075277b7769b6b0d5a",
        )
        candidate = JointReceptors(self.parameters, claw_tuning=ClawPositionTuning())
        self.assertNotEqual(candidate.identity, legacy.identity)
        with self.assertRaises(ValueError):
            candidate.restore(legacy.snapshot())
        with self.assertRaises(ValueError):
            legacy.step(1.4, 0, interior_angle_rad=1.8)
        self.assertEqual(legacy.tick, 0)
        for i in range(17):
            candidate.step(1.4, i - 8, interior_angle_rad=1.8)
        saved = candidate.snapshot()
        expected = candidate.step(1.3, -10, interior_angle_rad=1.9)
        candidate.restore(saved)
        self.assertEqual(candidate.step(1.3, -10, interior_angle_rad=1.9), expected)


if __name__ == "__main__":
    unittest.main()
