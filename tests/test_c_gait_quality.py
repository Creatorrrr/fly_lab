"""Physical quality scores cannot improve through missing load or longer stops."""

import copy
import unittest

from tools.analyze_c_banc_gait_quality import compare, quality


def recording():
    rows = []
    for i in range(801):
        t = i * .005
        braking = t > 3
        rows.append({'time_s': t, 'body': {'position': [min(t, 3), 0, 0]},
                     'navigation': {'distance_mm': max(1, 4 - t)},
                     'neural_request': {'navigation_at_start': {'brake_request': braking}},
                     'feet': {'normal_load_bw': [.1] * 6, 'slip_mm_s': [1.0 if not braking else 0] * 6},
                     'upright': 1, 'fault': None})
    return {'samples': rows}


class QualityContracts(unittest.TestCase):
    def test_braking_does_not_dilute_slip_and_boundary_is_fixed(self):
        result = quality(recording())
        self.assertAlmostEqual(result['window_end_s'], 3)
        self.assertAlmostEqual(result['moving_observed_s'], 2.5)
        self.assertEqual(result['balanced_mean_slip_mm_s'], 1)
        self.assertTrue(result['observation_valid'])

    def test_missing_supported_velocity_is_invalid(self):
        payload = recording()
        payload['samples'][200]['feet']['slip_mm_s'][0] = None
        with self.assertRaisesRegex(ValueError, 'Missing supported'):
            quality(payload)

    def test_removing_a_leg_does_not_pass_quality(self):
        payload = recording()
        for row in payload['samples']:
            row['feet']['normal_load_bw'][0] = 0
            row['feet']['slip_mm_s'][0] = None
        result = quality(payload)
        self.assertFalse(result['observation_valid'])
        self.assertIsNone(result['balanced_mean_slip_mm_s'])
        self.assertFalse(compare(quality(recording()), result)['Q2'])

    def test_lower_slip_does_not_excuse_slow_approach(self):
        baseline = quality(recording())
        candidate = copy.deepcopy(baseline)
        candidate['balanced_mean_slip_mm_s'] *= .5
        candidate['approach_speed_mm_s'] *= .5
        verdict = compare(baseline, candidate)
        self.assertTrue(verdict['Q1'])
        self.assertFalse(verdict['Q2'])
