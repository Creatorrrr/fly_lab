"""Control separation, exact CPU continuation, and shared-world isolation."""

import copy
import os
import unittest

import numpy as np

from flylab.body_options import BodyOptions


class ExtensionOptions(unittest.TestCase):
    def test_old_model_identity_and_invalid_combinations(self):
        self.assertEqual(
            BodyOptions(model="flybody").model_identity(),
            {
                "model": "flybody",
                "terrain": "flat",
                "terrain_seed": 0,
                "slope_degrees": 10.0,
                "render_camera": False,
            },
        )
        for options in (
            {"attachment": "unknown"},
            {"tendons": "tarsi"},
            {"model": "flybody", "tendons": "abdomen"},
            {"attachment": "tethered", "terrain": "slope"},
        ):
            with self.assertRaises(ValueError):
                BodyOptions(**options)


@unittest.skipUnless(
    os.environ.get("FLYLAB_NATIVE_TESTS") == "1", "Native MuJoCo tests"
)
class NativeExtensions(unittest.TestCase):
    def test_all_tendons_have_no_competing_servo_and_restore_commands(self):
        from flylab.body import FlyGymBody
        from flylab.engine import config_values
        from flylab.sensors import default_world

        b = FlyGymBody(
            42,
            default_world(),
            config_values(),
            body_options=BodyOptions(
                model="flybody",
                actuation="whole_body",
                servo_profile="tracking_all",
                attachment="tethered",
                tendons="all",
            ),
        )
        try:
            self.assertEqual(
                (
                    len(b.whole_body_control.names),
                    len(b.whole_body_control.passive_ids),
                    b.m.ntendon,
                ),
                (66, 0, 8),
            )
            self.assertEqual(len(b.tendon_control.joint_ids), 36)
            self.assertEqual(np.sum(b.m.jnt_type == b.mj.mjtJoint.mjJNT_FREE), 0)
            positions = set(b.whole_body_control.joint_ids)
            self.assertFalse(positions & set(b.tendon_control.joint_ids))
            initial = b.snapshot()
            for invalid in (
                {"lf_tarsus": float("nan")},
                {"abdomen_pitch": 1.1},
                {"bad": 0.1},
            ):
                with self.assertRaises(ValueError):
                    b.step_body_targets(
                        {"c_thorax-c_head-yaw": 0.1}, tendon_inputs=invalid
                    )
                self.assertEqual(initial, b.snapshot())
            with self.assertRaises(ValueError):
                b.step_body_targets(
                    tendon_inputs={"lf_tarsus": 0.2}, adhesion=np.ones(6)
                )
            self.assertEqual(initial, b.snapshot())
            b.set_body_actuation(tendon_inputs={"lf_tarsus": 0.2, "abdomen_yaw": -0.2})
            saved = b.snapshot()
            root = b.pose()[0].copy()
            b.step_joint_targets(b.neutral, b.last_action.adhesion_onoff)
            future = b.snapshot()
            b.restore(saved)
            b.step_joint_targets(b.neutral, b.last_action.adhesion_onoff)
            self.assertEqual(future, b.snapshot())
            np.testing.assert_array_equal(root, b.pose()[0])
            self.assertEqual(
                b.tendon_control.observation()["neural_mapping"], "unmapped"
            )
        finally:
            b.close()

    def test_new_model_cannot_restore_old_attachment(self):
        from flylab.body import FlyGymBody
        from flylab.engine import config_values
        from flylab.sensors import default_world

        bodies = []
        try:
            for attachment in ("free", "tethered"):
                bodies.append(
                    FlyGymBody(
                        42,
                        default_world(),
                        config_values(),
                        body_options=BodyOptions(attachment=attachment),
                    )
                )
            saved = bodies[1].snapshot()
            with self.assertRaises(ValueError):
                bodies[1].restore(bodies[0].snapshot())
            self.assertEqual(saved, bodies[1].snapshot())
        finally:
            for b in bodies:
                b.close()

    def test_shared_control_is_named_atomic_and_restorable(self):
        from flylab.shared_arena import SharedFlyArena

        arena = SharedFlyArena()
        try:
            before = arena.snapshot()
            with self.assertRaises(ValueError):
                arena.set_drives({"fly_1": [0.6, 0.6], "fly_2": [float("inf"), 0.0]})
            self.assertEqual(before, arena.snapshot())
            arena.set_drives({"fly_1": [0.6, 0.6]})
            np.testing.assert_array_equal(arena.drives["fly_2"], [0.0, 0.0])
            arena.advance(2)
            saved = arena.snapshot()
            arena.advance(2)
            future = arena.snapshot()
            arena.restore(saved)
            arena.advance(2)
            self.assertEqual(future, arena.snapshot())
            broken = copy.deepcopy(saved)
            broken["controllers"]["fly_2"]["cpg"]["curr_phases"] = [1.0]
            current = arena.snapshot()
            with self.assertRaises(ValueError):
                arena.restore(broken)
            self.assertEqual(current, arena.snapshot())
            self.assertEqual(arena.observation()["physical_worlds"], 1)
            self.assertFalse(arena.observation()["neural_connected"])
        finally:
            arena.close()


if __name__ == "__main__":
    unittest.main()
