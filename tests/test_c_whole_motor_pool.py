import copy
import unittest

import numpy as np

from tools.probe_c_whole_motor_pool import POSES, assess_motor_pairs


class WholeMotorPoolTests(unittest.TestCase):
    def test_other_flexor_can_respond_but_opponent_cut_or_residual_prevents_candidate(
        self,
    ):
        motors = [
            {"id": "fast", "target": "tibia_flexor_muscle"},
            {"id": "accessory", "target": "accessory_tibia_flexor_muscle"},
            {"id": "extensor", "target": "tibia_extensor_muscle"},
        ]
        zero = {q: np.zeros(3) for q in POSES}
        full = copy.deepcopy(zero)
        full["q1"][1] = 3
        full["q2"][2] = 4
        result = assess_motor_pairs(motors, full, zero, zero, zero)
        self.assertEqual(result["candidate_pairs"], 1)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertFalse(result["physical_qualified"])
        for condition in ("opponent", "cut", "residual"):
            altered, cut, final = (
                copy.deepcopy(full),
                copy.deepcopy(zero),
                copy.deepcopy(zero),
            )
            if condition == "opponent":
                altered["q1"][2] = 1
            elif condition == "cut":
                cut["q1"][1] = 2
            else:
                final["q2.3"][0] = 2
            self.assertEqual(
                assess_motor_pairs(motors, altered, cut, final, zero)[
                    "candidate_pairs"
                ],
                0,
            )
        duplicate = copy.deepcopy(motors)
        duplicate[1]["id"] = "fast"
        with self.assertRaises(ValueError):
            assess_motor_pairs(duplicate, full, zero, zero, zero)


if __name__ == "__main__":
    unittest.main()
