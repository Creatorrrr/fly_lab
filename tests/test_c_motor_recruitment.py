import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and importlib.util.find_spec("flygym"),
    "Native FlyGym/MuJoCo required",
)
class RecruitmentTests(unittest.TestCase):
    def test_calibration_preserves_body_and_balances_reference_active_torque(self):
        from flylab.c.motor_recruitment import TorqueBalancedRecruitment
        from flylab.c.muscles import MuscleRig

        body = MuscleRig()
        body.step((0.01, 0.02), steps=20)
        before = body.snapshot()
        calibration = TorqueBalancedRecruitment(body)
        after = body.snapshot()
        np.testing.assert_array_equal(before["state"], after["state"])
        self.assertEqual(before["tick"], after["tick"])
        self.assertAlmostEqual(
            calibration.metadata["reference_interior_angle_rad"], math.pi / 2, places=12
        )
        coefficients = np.array(calibration.metadata["torque_per_activation"])
        self.assertTrue((calibration.scale > 0).all())
        self.assertTrue((calibration.scale <= 1).all())
        for rate in (0, 1, 25, 100, 200, 1e30):
            excitation = calibration.encode([rate, rate])
            self.assertAlmostEqual(float(excitation @ coefficients), 0, places=12)
            self.assertTrue((excitation >= 0).all())
            self.assertTrue((excitation <= calibration.scale).all())
        with self.assertRaises(ValueError):
            calibration.scale[0] = 0
        for rates in (
            [1],
            [True, False],
            [True, 1.0],
            [1, np.bool_(False)],
            [-1, 1],
            [0, float("nan")],
        ):
            with self.assertRaises(ValueError):
                calibration.encode(rates)


class ExperimentArgumentsTests(unittest.TestCase):
    def test_unknown_profiles_rejected_before_writing_or_loading_graph(self):
        from tools.compare_c_rate_joint import run
        from tools.probe_c_rate_posture import run as posture_run

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "unused"
            for arguments in (
                {"claw_profile": "typo"},
                {"recruitment": "typo"},
                {"temporal_profile": "typo"},
                {"profiles": ["uniform", "uniform"]},
                {"profiles": ["typo"]},
                {"descending_inputs": [False]},
                {"descending_inputs": []},
                {"descending_inputs": [0, 0]},
                {"annotation_reference": Path("unused.csv")},
                {"annotation_sha256": "0" * 64},
                {
                    "annotation_reference": Path("unused.csv"),
                    "annotation_sha256": "0" * 64,
                    "temporal_profile": "fast_hypothesis",
                },
            ):
                with self.assertRaises(ValueError):
                    run(None, None, out, **arguments)
                self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                posture_run(None, None, out, claw_profile="typo")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
