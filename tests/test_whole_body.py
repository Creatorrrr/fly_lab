import os
import unittest

import numpy as np

from flylab.body_options import BodyOptions


class BodyScope(unittest.TestCase):
    def test_legacy_identity_omits_new_default(self):
        self.assertNotIn("actuation", BodyOptions(model="flybody").model_identity())
        self.assertEqual(
            BodyOptions(model="flybody", actuation="whole_body").model_identity()[
                "actuation"
            ],
            "whole_body",
        )

    def test_unsupported_scope_rejected(self):
        for value in (
            {"actuation": "whole_body"},
            {"actuation": "invalid"},
            {"actuation": True},
        ):
            with self.assertRaises(ValueError):
                BodyOptions(**value)


@unittest.skipUnless(
    os.environ.get("FLYLAB_NATIVE_TESTS") == "1", "Explicit native physics verification"
)
class NativeWholeBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flylab.body import FlyGymBody
        from flylab.engine import config_values
        from flylab.sensors import default_world

        cls.body = FlyGymBody(
            42,
            default_world(),
            config_values(),
            body_options=BodyOptions(model="flybody", actuation="whole_body"),
        )
        cls.initial = cls.body.snapshot()

    @classmethod
    def tearDownClass(cls):
        cls.body.close()

    def setUp(self):
        self.body.restore(self.initial)

    def test_sparse_command_preserved_by_legacy_leg_path_and_restore(self):
        b = self.body
        name = "c_thorax-c_head-yaw"
        b.step_body_targets({name: 0.1})
        saved = b.snapshot()
        b.step_joint_targets(b.neutral, b.last_action.adhesion_onoff)
        expected = b.snapshot()
        self.assertAlmostEqual(
            b.whole_body_observation()["targets_rad"][b.whole_body_control.index[name]],
            0.1,
        )
        b.restore(saved)
        b.step_joint_targets(b.neutral, b.last_action.adhesion_onoff)
        self.assertEqual(expected, b.snapshot())

    def test_invalid_adhesion_dt_and_mixed_targets_do_not_mutate(self):
        b = self.body
        name = "c_thorax-c_head-yaw"
        for command, kwargs in (
            ({name: 0.1, "bad": 0.0}, {}),
            ({name: 0.1}, {"adhesion": np.ones(6)}),
            ({name: 0.1}, {"dt": 0.00015}),
            ({name: float("inf")}, {}),
        ):
            before = b.snapshot()
            with self.assertRaises(ValueError):
                b.step_body_targets(command, **kwargs)
            self.assertEqual(before, b.snapshot())

    def test_observation_is_read_only_and_frame_leg_forces_align(self):
        b = self.body
        before = b.snapshot()
        obs = b.whole_body_observation()
        frame = b.frame()[1]
        self.assertEqual(len(obs["names"]), 78)
        self.assertEqual(len(obs["passive"]["names"]), 24)
        self.assertFalse(obs["root_actuated"])
        self.assertEqual(len(frame["actuatorForces"]), len(frame["jointTargets"]))
        self.assertEqual(before, b.snapshot())

    def test_initialization_rejects_mujoco_automatic_reset(self):
        from unittest.mock import patch

        import flylab.whole_body as control
        from flylab.body import FlyGymBody
        from flylab.engine import config_values
        from flylab.sensors import default_world

        configure = control.configure_tracking_servos

        def unstable_feedback(body):
            configure(body)
            ids = body.sim._intern_actuatorids_by_type_by_fly[
                body.ActuatorType.POSITION
            ][body.fly.name]
            for dof, actuator in zip(body.full_order, ids):
                if dof.child.link == "haltere":
                    # Reproduce the rejected candidate's explicit-damping failure.
                    body.m.actuator_biasprm[actuator, 2] = -1.7

        with (
            patch.object(control, "configure_tracking_servos", unstable_feedback),
            self.assertRaisesRegex(RuntimeError, "Invalid neutral settling"),
        ):
            FlyGymBody(
                42,
                default_world(),
                config_values(),
                body_options=BodyOptions(
                    model="flybody",
                    actuation="whole_body",
                    servo_profile="tracking",
                ),
            )
