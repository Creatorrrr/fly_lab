import copy
import unittest

import numpy as np

from flylab.c.neural import ExpLIF
from flylab.c.neuromuscular import NeuromuscularLoop, build_spec, motor_unit_response
from flylab.engine import config_values
from flylab.sensors import default_world
from tests.test_c import same_state
from tests.test_c_repairs import ContactBody
from tests.test_c_sensorimotor import banc_fixture


class PadBody(ContactBody):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift = np.zeros(6)

    def foot_kinematics(self, targets):
        return dict(target_lift_mm=self.lift.copy(), velocity_mm_s=np.zeros((6, 3)))


class MotorUnitTests(unittest.TestCase):
    def test_comparison_refuses_changed_gates_or_candidate_parameters(self):
        from tools.compare_c_banc_gait import validate_protocol, PROFILES
        protocol = dict(schema='flylab.banc-gait-improvement.v1',
            fixed=dict(seconds=10., seed=42, initial_pose=None, rate_half_Hz=100.,
                       lift_release_mm=.02, neural_dt_s=.0001, physical_dt_s=.0001,
                       control_dt_s=.005),
            hypotheses=[dict(name=name) for name in PROFILES[1:]],
            walking_gate=dict(seconds=10., horizontal_net_mm=5., signed_forward_mm=5.,
                              stuck='NOT_DETECTED', fault=None))
        validate_protocol(protocol)
        for section, key, value in [('fixed', 'seconds', 1.), ('fixed', 'rate_half_Hz', 50.),
                                    ('walking_gate', 'horizontal_net_mm', .05)]:
            bad = copy.deepcopy(protocol)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(bad)

    def setup_loop(self, pooling='sum-saturating-units-v3', policy='neural-tarsus-lift-release-v3'):
        graph = banc_fixture()
        body = PadBody(42, default_world(), config_values(None))
        # A two-unit flexor pool: changing the number of silent members must
        # not dilute the active member in the additive candidate.
        for node in graph.nodes:
            a = node['source_annotations']
            if (node['soma_side'] == 'left' and a.get('body_part_effector') == 'front_leg'
                    and a['peripheral_target_type'] == 'accessory_tibia_flexor_muscle'):
                a['peripheral_target_type'] = 'tibia_flexor_muscle'
        spec = build_spec(graph, 3)
        spec.update(motor_pooling=pooling, adhesion_policy=policy)
        return graph, body, NeuromuscularLoop(graph, spec, body)

    @staticmethod
    def muscle_ids(loop, name):
        return next(ids for muscle, ids, _ in loop.muscles[0] if muscle == name)

    def test_individual_response_is_monotone_bounded_and_zero_preserving(self):
        rates = np.array([0., 1., 10., 100., 1e6])
        response = motor_unit_response(rates, 100.)
        self.assertEqual(response[0], 0.)
        self.assertTrue(np.all(np.diff(response) > 0))
        self.assertTrue(np.all(response < 100.))
        self.assertEqual(response[3], 50.)

    def test_silent_members_do_not_dilute_and_active_members_add(self):
        graph, _, loop = self.setup_loop()
        neural = ExpLIF(graph)
        ids = self.muscle_ids(loop, 'tibia_flexor_muscle')
        self.assertEqual(len(ids), 2)
        neural.rate[ids[0]] = 100.
        loop.decode(neural)
        first = loop.last_motor[0]['muscle_output_equivalent_Hz']['tibia_flexor_muscle']
        self.assertEqual(first, 50.)
        neural.rate[ids[1]] = 100.
        loop.decode(neural)
        self.assertEqual(loop.last_motor[0]['muscle_output_equivalent_Hz']['tibia_flexor_muscle'], 2*first)
        units = [u for row in loop.last_motor for u in row['motor_units']]
        self.assertEqual(len(units), len(loop.motor_indices))
        self.assertEqual(len({u['id'] for u in units}), len(units))
        self.assertTrue(all(u['cell_type'] == 'SYNTHETIC_ONLY' for u in units))

    def test_release_requires_neural_attachment_and_upward_target(self):
        graph, body, loop = self.setup_loop()
        neural = ExpLIF(graph)
        body.lift[:] = 1.
        self.assertFalse(loop.decode(neural)[1].any())
        neural.rate[self.muscle_ids(loop, 'tarsus_depressor_muscle')] = 100.
        self.assertFalse(loop.decode(neural)[1][0])
        self.assertTrue(loop.last_motor[0]['adhesion_requested'])
        self.assertTrue(loop.last_motor[0]['lift_release'])
        body.lift[0] = -.1
        self.assertTrue(loop.decode(neural)[1][0])
        targets, adhesion = loop.decode(neural, disconnected=True)
        self.assertFalse(adhesion.any())
        self.assertFalse(loop.last_motor[0]['lift_release'])
        self.assertTrue(loop.last_motor[0]['motor_disconnected'])
        np.testing.assert_array_equal(targets, body.neutral)

    def test_invalid_pad_observation_rejects_without_advancing_motor_state(self):
        graph, body, loop = self.setup_loop()
        neural = ExpLIF(graph)
        neural.rate[:] = 30.
        loop.decode(neural)
        before = loop.snapshot()
        body.lift[2] = float('nan')
        with self.assertRaises(ValueError):
            loop.decode(neural)
        same_state(before, loop.snapshot())

    def test_neutral_candidate_matches_v2_when_both_changes_are_disabled(self):
        graph, body, loop = self.setup_loop('mean-linear-v2', 'neural-tarsus-active-v2')
        old = NeuromuscularLoop(graph, build_spec(graph, 2), body)
        neural = ExpLIF(graph)
        rng = np.random.default_rng(15)
        for _ in range(12):
            neural.rate[:] = rng.uniform(0., 250., graph.n)
            a, apad = old.decode(neural)
            b, bpad = loop.decode(neural)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(apad, bpad)

    def test_unit_annotations_and_checkpoint_profile_are_bound(self):
        graph, body, loop = self.setup_loop()
        bad = copy.deepcopy(loop.spec)
        bad['rows'][0]['muscles'][0]['units'][0]['cell_type'] = 'invented_slow'
        with self.assertRaises(ValueError):
            NeuromuscularLoop(graph, bad, body)
        neural = ExpLIF(graph)
        neural.rate[:] = 30.
        loop.encode(); loop.decode(neural)
        before = loop.snapshot()
        loop.encode(); expected = loop.decode(neural)
        after = loop.snapshot()
        loop.restore(before)
        loop.encode(); actual = loop.decode(neural)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        same_state(after, loop.snapshot())
        other_spec = copy.deepcopy(loop.spec)
        other_spec['rate_half_Hz'] = 50.
        other = NeuromuscularLoop(graph, other_spec, body)
        with self.assertRaises(ValueError):
            other.restore(before)

    def test_native_pad_jacobian_matches_small_displacements_without_state_changes(self):
        from flylab.body import FlyGymBody, dependency_report
        from flylab.engine import config_values
        from flylab.sensors import default_world
        if not dependency_report()['ready']:
            self.skipTest('Pinned FlyGym/MuJoCo runtime required')
        body = FlyGymBody(42, default_world(), config_values(None))
        try:
            before = body.snapshot()
            q = body.d.qpos[body.qpos_ids].copy()
            delta = np.random.default_rng(5).uniform(-1e-6, 1e-6, 42)
            prediction = body.foot_kinematics(q+delta)
            same_state(before, body.snapshot())
            self.assertTrue(np.isfinite(prediction['velocity_mm_s']).all())
            initial_z = body.d.xpos[body._tip_ids, 2].copy()
            body.d.qpos[body.qpos_ids] += delta
            body.mj.mj_forward(body.m, body.d)
            actual = body.d.xpos[body._tip_ids, 2] - initial_z
            np.testing.assert_allclose(prediction['target_lift_mm'], actual, rtol=1e-4, atol=2e-11)
        finally:
            body.close()


if __name__ == '__main__':
    unittest.main()
