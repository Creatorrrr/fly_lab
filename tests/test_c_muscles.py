import copy
import importlib.util
import math
import unittest
import numpy as np
from flylab.c.receptors import JointReceptors, ReceptorParameters


class ReceptorTests(unittest.TestCase):
    def test_direction_delay_and_sensor_off(self):
        p = ReceptorParameters(0., 2., delay_steps=2)
        positive, negative = JointReceptors(p), JointReceptors(p)
        for tick in range(30):
            a = positive.step(1.5, 10., support_bw=.5)
            b = negative.step(.5, -10., contact_bw=.05)
            if tick < 2: self.assertEqual(sum(a['output'].values()), 0.)
        self.assertGreater(a['output']['claw_positive'], a['output']['claw_negative'])
        self.assertGreater(b['output']['claw_negative'], b['output']['claw_positive'])
        self.assertGreater(a['output']['hook_positive'], 0.); self.assertEqual(a['output']['hook_negative'], 0.)
        self.assertGreater(b['output']['hook_negative'], 0.); self.assertEqual(b['output']['hook_positive'], 0.)
        self.assertGreater(a['output']['load'], 0.); self.assertEqual(a['output']['touch'], 0.)
        self.assertGreater(b['output']['touch'], 0.); self.assertEqual(b['output']['load'], 0.)
        disabled = positive.step(1.5, 10., enabled=False)
        self.assertEqual(sum(disabled['output'].values()), 0.)
        self.assertGreater(sum(disabled['raw'].values()), 0.)

    def test_stationary_sustained_motion_and_vibration_are_separate(self):
        p = ReceptorParameters(0., 2.)
        stationary, moving, vibration = [JointReceptors(p) for _ in range(3)]
        for tick in range(500):
            a = stationary.step(1., 0.)
            b = moving.step(1., 10.)
            c = vibration.step(1.+.05*math.sin(tick*.5), 25*math.cos(tick*.5))
        self.assertEqual(a['raw']['club_highpass'], 0.)
        self.assertLess(b['raw']['club_highpass'], 1e-6)
        self.assertGreater(c['output']['club_highpass'], .1)
        self.assertGreater(b['output']['hook_positive'], .4)

    def test_receptor_restore_is_atomic_and_continuation_exact(self):
        r = JointReceptors(ReceptorParameters(.1, 2.))
        for _ in range(20): r.step(1., 5.)
        saved = r.snapshot()
        for key, value in [('tick', -1), ('velocity_lowpass', float('nan')),
                           ('filtered', np.full(7, 2.)), ('queue', np.zeros((1, 7)))]:
            bad = copy.deepcopy(saved); bad[key] = value
            with self.assertRaises(ValueError): r.restore(bad)
            for field in ('filtered', 'queue'): np.testing.assert_array_equal(saved[field], r.snapshot()[field])
            self.assertEqual(saved['tick'], r.tick)
        expected = r.step(.5, -2.)
        r.restore(saved); self.assertEqual(expected, r.step(.5, -2.))
        zero = JointReceptors(r.p); bad = zero.snapshot(); bad['queue'][0, 0] = 1.
        with self.assertRaises(ValueError): zero.restore(bad)


@unittest.skipUnless(importlib.util.find_spec('mujoco') and importlib.util.find_spec('flygym'),
                     'Native FlyGym/MuJoCo dependencies required')
class MuscleTests(unittest.TestCase):
    def setUp(self):
        from flylab.c.muscles import MuscleRig
        self.rig = MuscleRig()

    def test_derivation_activation_and_force_consistency(self):
        r = self.rig
        self.assertEqual((r.model.nq, r.model.nv, r.model.nu), (1, 1, 2))
        self.assertLess(max(r.metadata['derivation_error'].values()), 1e-9)
        for excitation in ((1., 0.), (0., 1.), (0., 0.)):
            for _ in range(100):
                frame = r.step(excitation, steps=1)
                self.assertTrue(all(0 <= v <= 1 for v in frame['activation']))
                self.assertLess(frame['torque_projection_error'], 1e-9)
                self.assertLess(frame['dynamics_balance_error'], 1e-8)
                self.assertEqual(frame['contacts'], 0)

    def test_moment_arm_matches_tendon_length_derivative(self):
        from flylab.c.muscles import moment_matrix
        r = self.rig; lengths = []
        for sign in (-1, 1):
            candidate = r.mj.MjData(r.model)
            candidate.qpos[:] = r.data.qpos+sign*1e-6
            r.mj.mj_forward(r.model, candidate); lengths.append(candidate.actuator_length.copy())
        np.testing.assert_allclose((lengths[1]-lengths[0])/2e-6,
                                   moment_matrix(r.mj, r.model, r.data)[:, 0], rtol=1e-6, atol=1e-9)

    def test_invalid_commands_and_restore_preserve_state_and_replay(self):
        r = self.rig; r.step((.2, .1), steps=30); saved = r.snapshot()
        for kw in (dict(excitation=(2., 0.)), dict(torque=float('nan')), dict(steps=True), dict(connected=1)):
            with self.assertRaises(ValueError): r.step(**kw)
            np.testing.assert_array_equal(saved['state'], r.snapshot()['state'])
        for key, value in [('tick', 0), ('requested', np.zeros(3)), ('connected', False), ('model_hash', 'other')]:
            bad = copy.deepcopy(saved); bad[key] = value
            with self.assertRaises(ValueError): r.restore(bad)
            np.testing.assert_array_equal(saved['state'], r.snapshot()['state'])
        r.step((.1, .3), torque=.01, steps=100); expected = r.snapshot()['state']
        r.restore(saved); r.step((.1, .3), torque=.01, steps=100)
        np.testing.assert_array_equal(expected, r.snapshot()['state'])

    def test_walking_body_instrumentation_is_read_only(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        body = FlyGymBody(7, default_world(), config_values())
        try:
            before = body.d.qpos.copy(); clock = body.physics_time()
            sample = body.joint_observation()
            self.assertEqual(len(sample['names']), 42)
            self.assertEqual(sample['positive_axes_world'].shape, (42, 3))
            np.testing.assert_allclose(np.linalg.norm(sample['positive_axes_world'], axis=1), 1.)
            np.testing.assert_array_equal(sample['actuator_torque'], body.d.qfrc_actuator[body.qvel_ids])
            np.testing.assert_array_equal(before, body.d.qpos); self.assertEqual(clock, body.physics_time())
        finally: body.close()
