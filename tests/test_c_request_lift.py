"""Neural lift release must preserve commands, physical bounds and saved state."""
import unittest
from types import SimpleNamespace

import numpy as np

from flylab.c.banc_walking import BancWalkingParameters
from flylab.c.rate_body import RateBodyAdapter
from tests.test_c_rate_body import fixture


class RequestLiftContracts(unittest.TestCase):
    def setUp(self):
        self.graph, self.body, _ = fixture()
        self.body.act_ids = np.arange(42)
        self.body.m = SimpleNamespace(
            actuator_ctrllimited=np.ones(42, bool),
            actuator_ctrlrange=np.column_stack((self.body.neutral-.5, self.body.neutral+.5)),
        )
        self.body.foot_kinematics = lambda targets: {
            'target_lift_mm': (targets-self.body.neutral).reshape(6, 7)[:, 5].copy()
        }
        self.kwargs = {'joint_activation_tau_s': .09, 'adhesion_model': 'request_lift'}

    def rates(self, adapter):
        rates = np.zeros(len(adapter.motor_ids), np.float32)
        rates[adapter.motor_columns[0]] = 100  # LF knee flexion/lift.
        rates[adapter.motor_columns[2]] = 100  # LF neural adhesion vote.
        return rates

    def test_neural_lift_releases_early_without_changing_joint_targets(self):
        legacy = RateBodyAdapter(self.graph, self.body, joint_activation_tau_s=.09)
        candidate = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        rates = self.rates(candidate)
        lt, la = legacy.decode(rates)
        ct, ca = candidate.decode(rates)
        np.testing.assert_array_equal(ct, lt)
        np.testing.assert_array_equal(candidate.offset, legacy.offset)
        self.assertTrue(la[0])
        self.assertFalse(ca[0])
        self.assertTrue(np.all(ca <= la))
        self.assertLess(ct[5]-self.body.neutral[5], .02)

    def test_unreachable_lift_is_not_used_and_unlimited_actuators_stay_unclipped(self):
        self.body.m.actuator_ctrlrange[5, 1] = self.body.neutral[5]+.01
        capped = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        self.assertTrue(capped.decode(self.rates(capped))[1][0])
        self.body.m.actuator_ctrllimited[5] = False
        unlimited = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        self.assertFalse(unlimited.decode(self.rates(unlimited))[1][0])

    def test_original_filtered_release_survives_when_new_request_has_no_lift(self):
        candidate = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        candidate.offset[5] = .1
        rates = self.rates(candidate)
        rates[candidate.motor_columns[0]] = 0
        target, adhesion = candidate.decode(rates)
        self.assertGreater(target[5]-self.body.neutral[5], .02)
        self.assertFalse(adhesion[0])

    def test_saved_future_is_exact_cut_is_immediate_and_profiles_do_not_cross_restore(self):
        a = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        b = RateBodyAdapter(self.graph, self.body, **self.kwargs)
        rates = self.rates(a)
        a.decode(rates)
        b.restore(a.snapshot())
        for _ in range(6):
            ta, aa = a.decode(rates)
            tb, ab = b.decode(rates)
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(aa, ab)
        target, adhesion = a.decode(rates, motor_cut=True)
        np.testing.assert_array_equal(target, self.body.neutral)
        self.assertFalse(adhesion.any())
        old = RateBodyAdapter(self.graph, self.body, joint_activation_tau_s=.09)
        with self.assertRaises(ValueError):
            old.restore(b.snapshot())

    def test_explicit_configuration_and_declared_filter_path_are_required(self):
        self.assertNotIn('adhesion_model', BancWalkingParameters().state_dict())
        p = BancWalkingParameters(adhesion_model='request_lift')
        self.assertEqual(p, BancWalkingParameters(**p.state_dict()))
        for name in ('muscle_activation_first', 'active_length'):
            with self.assertRaises(ValueError):
                BancWalkingParameters(adhesion_model='request_lift', **{name: True})
            with self.assertRaises(ValueError):
                RateBodyAdapter(self.graph, self.body, **self.kwargs, **{name: True})


if __name__ == '__main__':
    unittest.main()
