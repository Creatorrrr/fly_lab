import math
import unittest

import numpy as np

from flylab.c.target_sense import TargetSensor


class TargetSensorTests(unittest.TestCase):
    def test_ui_left_is_positive_body_bearing(self):
        sensor = TargetSensor([6.0, -6.0])
        result = sensor.observe([0.0, 0.0, 0.8], np.eye(3))
        self.assertAlmostEqual(result.bearing_rad, math.pi / 4)
        self.assertAlmostEqual(result.distance_mm, math.sqrt(72))
        left = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(
            sensor.observe([0.0, 0.0, 0.8], left).bearing_rad, -math.pi / 4
        )

    def test_cut_cannot_leak_target_information(self):
        a = TargetSensor([8.0, 0.0]).observe([0.0, 0.0, 0.8], np.eye(3), cut=True)
        b = TargetSensor([-6.0, 6.0]).observe([0.0, 0.0, 0.8], np.eye(3), cut=True)
        self.assertEqual(a, b)
        self.assertFalse(a.visible)

    def test_checkpoint_and_bad_coordinates(self):
        sensor = TargetSensor([-6.0, 0.0])
        self.assertEqual(
            TargetSensor.from_snapshot(sensor.snapshot()).target_xz, sensor.target_xz
        )
        for point in ([True, 2.0], [float("nan"), 2.0], [101.0, 0.0], [1.0], "1,2"):
            with self.assertRaises(ValueError):
                TargetSensor(point)
