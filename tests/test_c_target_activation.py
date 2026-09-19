"""Analytic dynamics and adapter-state contracts for optional target release."""

import copy
import math
import unittest

import numpy as np

from flylab.c.banc_walking import BancWalkingParameters
from flylab.c.rate_body import RateBodyAdapter
from flylab.c.target_activation import advance_target_activation
from tests.test_c_rate_body import fixture


class TargetActivationContracts(unittest.TestCase):
    def test_growth_and_neutral_release_have_distinct_time_constants(self):
        result = advance_target_activation(np.array([0., .4]), np.array([.4, 0.]), .03, .09)
        np.testing.assert_allclose(result, [.4 * (1 - math.exp(-1/3)), .4 / math.e], atol=1e-15)

    def test_reversal_changes_response_at_zero_not_at_control_boundary(self):
        crossing = .03 * math.log(2)
        result = advance_target_activation(np.array([.4]), np.array([-.4]), .03, .09)
        expected = -.4 * (1 - math.exp(-(.03-crossing)/.09))
        self.assertAlmostEqual(float(result[0]), expected, places=15)
        zero = advance_target_activation(np.array([.4]), np.array([-.4]), crossing, .09)
        self.assertAlmostEqual(float(zero[0]), 0., places=15)

    def test_constant_request_subdivision_is_equivalent_across_reversals(self):
        start = np.array([-.4, -.2, -.1, 0., .1, .2, .4])
        target = np.array([.1, 0., -.4, .3, .05, -.3, -.4])
        whole = advance_target_activation(start, target, .05, .09)
        split = start.copy()
        for _ in range(10):
            split = advance_target_activation(split, target, .005, .09)
        np.testing.assert_allclose(whole, split, rtol=0, atol=5e-16)

    def test_directions_are_symmetric_and_never_overshoot(self):
        start = np.linspace(-.5, .5, 41)
        target = np.linspace(.3, -.4, 41)
        result = advance_target_activation(start, target, .05, .09)
        np.testing.assert_array_equal(result, -advance_target_activation(-start, -target, .05, .09))
        self.assertTrue(np.all(result >= np.minimum(start, target)))
        self.assertTrue(np.all(result <= np.maximum(start, target)))

    def test_reversal_only_preserves_same_sign_and_zero_request_decay(self):
        start, target = np.array([.4, -.3, .2, .4]), np.array([.1, -.1, 0., -.4])
        result = advance_target_activation(start, target, .03, .09, reversals_only=True)
        legacy = start + (1-math.exp(-1/3)) * (target-start)
        np.testing.assert_array_equal(result[:3], legacy[:3])
        full = advance_target_activation(start, target, .03, .09)
        self.assertEqual(result[3], full[3])
        split = start.copy()
        for _ in range(6):
            split = advance_target_activation(split, target, .005, .09, reversals_only=True)
        np.testing.assert_allclose(result, split, rtol=0, atol=5e-16)

    def test_release_modes_have_distinct_restorable_identities(self):
        with self.assertRaises(ValueError):
            BancWalkingParameters(joint_release_mode="reversal_only")
        self.assertNotIn("joint_release_mode", BancWalkingParameters().state_dict())
        p = BancWalkingParameters(joint_activation_tau_s=.09, fast_joint_release=True,
                                  joint_release_mode="reversal_only")
        self.assertEqual(p, BancWalkingParameters(**p.state_dict()))
        graph, body, _ = fixture()
        kwargs = {"joint_activation_tau_s": .09, "fast_joint_release": True}
        all_release = RateBodyAdapter(graph, body, **kwargs)
        reversal = RateBodyAdapter(graph, body, **kwargs, joint_release_mode="reversal_only")
        clone = RateBodyAdapter(graph, body, **kwargs, joint_release_mode="reversal_only")
        self.assertNotEqual(all_release.identity, reversal.identity)
        reversal.offset[:] = np.linspace(-.4, .4, 42)
        clone.restore(reversal.snapshot())
        rates = np.zeros(len(reversal.motor_ids), np.float32)
        np.testing.assert_array_equal(reversal.decode(rates)[0], clone.decode(rates)[0])
        with self.assertRaises(ValueError):
            all_release.restore(reversal.snapshot())

    def test_option_is_explicit_and_requires_the_declared_filter_path(self):
        self.assertNotIn("fast_joint_release", BancWalkingParameters().state_dict())
        good = BancWalkingParameters(fast_joint_release=True, joint_activation_tau_s=.09)
        self.assertEqual(BancWalkingParameters(**good.state_dict()), good)
        for changes in ({}, {"muscle_activation_first": True}, {"continuous_joint_activation": True}):
            params = {"fast_joint_release": True}
            if changes:
                params["joint_activation_tau_s"] = .09
            with self.assertRaises(ValueError):
                BancWalkingParameters(**params, **changes)

    def test_adapter_restore_keeps_release_future_and_motor_cut_is_immediate(self):
        graph, body, _ = fixture()
        kwargs = {"joint_activation_tau_s": .09, "fast_joint_release": True}
        a, b = RateBodyAdapter(graph, body, **kwargs), RateBodyAdapter(graph, body, **kwargs)
        rates = np.zeros(len(a.motor_ids), np.float32)
        rates[a.motor_columns[0]] = 100.
        for _ in range(12):
            a.decode(rates)
        self.assertGreater(float(np.max(np.abs(a.offset))), .01)
        b.restore(copy.deepcopy(a.snapshot()))
        rates.fill(0)
        for _ in range(8):
            ta, aa = a.decode(rates)
            tb, ab = b.decode(rates)
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(aa, ab)
        target, adhesion = a.decode(rates, motor_cut=True)
        np.testing.assert_array_equal(target, body.neutral)
        self.assertFalse(np.any(adhesion))
        legacy = RateBodyAdapter(graph, body, joint_activation_tau_s=.09)
        with self.assertRaises(ValueError):
            legacy.restore(b.snapshot())

    def test_fast_coxa_axes_keep_the_existing_arithmetic(self):
        graph, body, _ = fixture()
        kwargs = {"joint_activation_tau_s": .09, "fast_coxa_activation": True}
        legacy = RateBodyAdapter(graph, body, **kwargs)
        candidate = RateBodyAdapter(graph, body, **kwargs, fast_joint_release=True)
        legacy.offset[:] = np.linspace(-.4, .4, 42)
        candidate.offset[:] = legacy.offset
        rates = np.zeros(len(legacy.motor_ids), np.float32)
        for _ in range(20):
            legacy.decode(rates)
            candidate.decode(rates)
            np.testing.assert_array_equal(candidate.offset[candidate.joints[:, 0]],
                                          legacy.offset[legacy.joints[:, 0]])
        self.assertFalse(np.array_equal(candidate.offset, legacy.offset))


if __name__ == "__main__":
    unittest.main()
