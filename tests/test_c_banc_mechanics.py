import unittest

import numpy as np

from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.banc_mechanics import configure_activation_trajectory, configure_active_length, configure_adhesion_units, configure_finite_pad_contact, set_active_length_scale
from flylab.engine import config_values
from flylab.sensors import default_world


class ContinuousActivationTests(unittest.TestCase):
    def test_adhesion_force_units_and_restored_future(self):
        body = FlyGymBody(42, default_world(), config_values(None),
                          body_options=BodyOptions(model="flybody", actuation="whole_body",
                                                   servo_profile="tracking_all", tendons="all"))
        try:
            original_hash = body.model_hash
            gain = body.m.actuator_gainprm.copy()
            bias, friction = body.m.actuator_biasprm.copy(), body.m.pair_friction.copy()
            ids = np.asarray(body.sim._intern_adhesionactuatorids_by_fly[body.fly.name])
            configure_adhesion_units(body, False)
            self.assertEqual(body.model_hash, original_hash)
            configure_adhesion_units(body, True)
            self.assertNotEqual(body.model_hash, original_hash)
            gain[ids, 0] *= 10
            np.testing.assert_array_equal(body.m.actuator_gainprm, gain)
            np.testing.assert_array_equal(body.m.actuator_biasprm, bias)
            np.testing.assert_array_equal(body.m.pair_friction, friction)
            body.d.ctrl[ids] = 1
            body.mj.mj_forward(body.m, body.d)
            np.testing.assert_array_equal(body.d.actuator_force[ids], np.full(6, .985 * 10))
            body.d.ctrl[ids] = 0
            body.mj.mj_forward(body.m, body.d)
            np.testing.assert_array_equal(body.d.actuator_force[ids], np.zeros(6))
            saved = body.snapshot()
            body.step_joint_targets(body.neutral, np.ones(6, bool))
            future = body.snapshot()['state']
            body.restore(saved)
            body.step_joint_targets(body.neutral, np.ones(6, bool))
            np.testing.assert_array_equal(body.snapshot()['state'], future)
            body.graded_adhesion = True
            fractions = np.arange(6) / 5
            body.step_joint_targets(body.neutral, fractions > 0, adhesion_control=fractions)
            np.testing.assert_array_equal(body.d.actuator_force[ids], .985 * 10 * fractions)
            saved = body.snapshot()
            body.step_joint_targets(body.neutral, fractions > 0, adhesion_control=fractions)
            future = body.snapshot()['state']
            body.restore(saved)
            body.step_joint_targets(body.neutral, fractions > 0, adhesion_control=fractions)
            np.testing.assert_array_equal(body.snapshot()['state'], future)
            with self.assertRaises(ValueError):
                body.step_joint_targets(body.neutral, np.ones(6, bool))
            with self.assertRaises(ValueError):
                body.step_joint_targets(body.neutral, np.zeros(6, bool), adhesion_control=fractions)
            body.step_joint_targets(body.neutral, np.zeros(6, bool), adhesion_control=np.zeros(6))
            np.testing.assert_array_equal(body.d.actuator_force[ids], np.zeros(6))
        finally:
            body.close()

    def test_finite_pad_uses_existing_friction_and_replays_native_future(self):
        body = FlyGymBody(42, default_world(), config_values(None),
                          body_options=BodyOptions(model="flybody", actuation="whole_body",
                                                   servo_profile="tracking_all", tendons="all"))
        try:
            original_hash = body.model_hash
            dimensions, friction = body.m.pair_dim.copy(), body.m.pair_friction.copy()
            configure_finite_pad_contact(body, False)
            self.assertEqual(body.model_hash, original_hash)
            configure_finite_pad_contact(body, True)
            self.assertNotEqual(body.model_hash, original_hash)
            np.testing.assert_array_equal(body.m.pair_friction, friction)
            mapping, _ = body._force_maps['feet']
            changed = body.m.pair_dim != dimensions
            self.assertTrue(changed.any())
            for p in np.flatnonzero(changed):
                a, b = body.m.pair_geom1[p], body.m.pair_geom2[p]
                self.assertTrue((mapping[a] >= 0 and b in body.support_geom_ids)
                                or (mapping[b] >= 0 and a in body.support_geom_ids))
                self.assertEqual(body.m.pair_dim[p], 6)
            saved = body.snapshot()
            body.step_joint_targets(body.neutral, np.ones(6, bool))
            future = body.snapshot()['state']
            body.restore(saved)
            body.step_joint_targets(body.neutral, np.ones(6, bool))
            np.testing.assert_array_equal(body.snapshot()['state'], future)
            for contact in body.d.contact[:body.d.ncon]:
                a, b = contact.geom1, contact.geom2
                if ((mapping[a] >= 0 and b in body.support_geom_ids)
                        or (mapping[b] >= 0 and a in body.support_geom_ids)):
                    self.assertEqual(contact.dim, 6)
        finally:
            body.close()

    def test_active_length_native_force_and_restored_future(self):
        body = FlyGymBody(42, default_world(), config_values(None),
                          body_options=BodyOptions(model="flybody", actuation="whole_body",
                                                   servo_profile="tracking_all", tendons="all"))
        try:
            original_hash = body.model_hash
            configure_active_length(body, False)
            self.assertEqual(body.model_hash, original_hash)
            self.assertNotIn('joint_stiffness_scale', body.snapshot())
            configure_active_length(body, True)
            self.assertNotEqual(body.model_hash, original_hash)
            active = np.linspace(.1, .9, 42)
            displacement = np.linspace(-.04, .04, 42)
            target = body.neutral + displacement / (1 + active)
            set_active_length_scale(body, 1 + active)
            body.d.ctrl[body.act_ids] = target
            body.mj.mj_forward(body.m, body.d)
            actual_target = np.where(body.m.actuator_ctrllimited[body.act_ids],
                                     np.clip(target, *body.m.actuator_ctrlrange[body.act_ids].T), target)
            expected = body._active_length_base_gain * (1 + active) * (
                actual_target - body.d.actuator_length[body.act_ids]
            ) + body.m.actuator_biasprm[body.act_ids, 2] * body.d.actuator_velocity[body.act_ids]
            np.testing.assert_allclose(body.d.actuator_force[body.act_ids], expected, atol=1e-12, rtol=0)
            saved = body.snapshot()
            body.step_joint_targets(target, np.zeros(6, bool), joint_stiffness_scale=1+active)
            expected_state = body.snapshot()['state']
            set_active_length_scale(body, np.ones(42))
            body.restore(saved)
            np.testing.assert_array_equal(body.active_length_scale, 1+active)
            body.step_joint_targets(target, np.zeros(6, bool), joint_stiffness_scale=1+active)
            np.testing.assert_array_equal(body.snapshot()['state'], expected_state)
            with self.assertRaises(ValueError):
                body.step_joint_targets(target, np.zeros(6, bool))
        finally:
            body.close()

    def test_native_commands_follow_activation_and_motor_cut_is_immediate(self):
        body = FlyGymBody(42, default_world(), config_values(None),
                          body_options=BodyOptions(model="flybody", actuation="whole_body",
                                                   servo_profile="tracking_all", tendons="all"))
        try:
            original_hash = body.model_hash
            configure_activation_trajectory(body, 0.0)
            self.assertEqual(body.model_hash, original_hash)
            initial = body.snapshot()
            start = body.neutral.copy()
            target = start.copy()
            target[5] += .04
            controls = []
            original_step = body._step_physics

            def observed_step():
                controls.append(body.d.ctrl[body.act_ids].copy())
                original_step()

            body._step_physics = observed_step
            body.step_joint_targets(target, np.zeros(6, bool))
            np.testing.assert_array_equal(controls, np.tile(target, (50, 1)))
            body.restore(initial)
            configure_activation_trajectory(body, .03)
            self.assertNotEqual(body.model_hash, original_hash)
            controls.clear()
            body.step_joint_targets(target, np.zeros(6, bool))
            # Independent activation ODE solution for the held neural request
            # whose 5ms endpoint is target. Check what native steps received.
            request = start + (target - start) / (1 - np.exp(-.005 / .03))
            expected = np.array([request + (start - request) * np.exp(-t / .03)
                                 for t in np.arange(1, 51) * .0001])
            np.testing.assert_allclose(controls, expected, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(controls[-1], target)
            controls.clear()
            body.step_joint_targets(start, np.zeros(6, bool), interpolate=False)
            np.testing.assert_array_equal(controls, np.tile(start, (50, 1)))
        finally:
            body.close()


if __name__ == "__main__":
    unittest.main()
