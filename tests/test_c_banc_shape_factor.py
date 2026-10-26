"""Pure E107 reference-shape, prefix and terminal gates; no body creation."""

import unittest

import numpy as np

from flylab.body import LEGS
from tools.probe_c_banc_m3_shape_factor import (
    arm_terminal,
    prefix_exact,
    resource_screen,
    shape_commands,
    shape_screen,
)
from tools.probe_c_banc_m3_slow_phase_replay import CONTROLS


class ShapeFactorTests(unittest.TestCase):
    @staticmethod
    def source():
        time_s = np.arange(2001)*.005
        base = np.arange(.2, 9.801, .4)
        peaks = [base.copy() for _ in range(6)]
        for leg in (1, 3, 5):
            peaks[leg] = base-.2
        query = np.broadcast_to(time_s[1:CONTROLS+1, None], (CONTROLS, 6)).copy()
        ref_time = time_s[1:]
        reference = {'time_s': ref_time,
                     'phase': (2*np.pi*ref_time[:, None]/.4
                               + 2*np.pi*np.array([0., .5, 0., .5, 0., .5])),
                     'command': np.tile(np.arange(42, dtype=float)/100, (2000, 1))}
        return time_s, peaks, query, reference

    def test_smooth_blend_and_original_prefix(self):
        time_s, peaks, query, reference = self.source()
        base = np.zeros((CONTROLS, 42))
        output, phase, blend, target = shape_commands(time_s, query, base, peaks, reference)
        np.testing.assert_array_equal(output[:500], base[:500])
        self.assertEqual(blend[499], 0.)
        self.assertAlmostEqual(blend[549], .5)  # control t=2.75
        self.assertEqual(blend[599], 1.)
        self.assertEqual(phase.shape, (1400, 6))
        self.assertTrue(np.isfinite(phase).all())
        self.assertTrue(np.allclose(output[549], .5*target[49]))
        self.assertTrue(np.allclose(output[600], target[100]))

    def test_shape_and_timing_factor_use_own_query_without_changing_prefix(self):
        time_s, peaks, query, reference = self.source()
        phase = reference['phase'][:, 0]
        reference['command'][:, 7:14] = np.sin(phase)[:, None]
        base = np.zeros((CONTROLS, 42))
        shape_only, _, _, _ = shape_commands(time_s, query, base, peaks, reference)
        timing_query = query.copy()
        timing_query[500:, 1] += .02
        combined, _, _, _ = shape_commands(time_s, timing_query, base, peaks, reference)
        np.testing.assert_array_equal(shape_only[:500], combined[:500])
        np.testing.assert_array_equal(shape_only[:, :7], combined[:, :7])
        self.assertGreater(np.max(abs(shape_only[600:, 7:14]-combined[600:, 7:14])), .05)
        np.testing.assert_array_equal(shape_only[:, 14:], combined[:, 14:])
        with self.assertRaises(ValueError):
            bad = timing_query.copy()
            bad[700, 1] = np.nan
            shape_commands(time_s, bad, base, peaks, reference)
        with self.assertRaises(ValueError):
            short = [p.copy() for p in peaks]
            short[5] = short[5][:-2]
            shape_commands(time_s, query, base, short, reference)

    def test_exact_original_boundary_and_native_prefix(self):
        rows = [{'qpos': [float(i)], 'qvel': [float(i+1)], 'ctrl': [float(i+2)]}
                for i in range(501)]
        reference_rows = [dict(row) for row in rows]
        native = {'time_s': np.arange(25000)*.0001,
                  'normal_load_bw': np.zeros((25000, 6)),
                  'slip_mm_s': np.full((25000, 6), np.nan),
                  'nonfoot_contact': np.zeros(25000, bool),
                  'upright': np.ones(25000)}
        reference_native = {key: value.copy() for key, value in native.items()}
        self.assertTrue(prefix_exact(rows, reference_rows, native, reference_native))
        rows[500]['ctrl'] = [1000.]
        with self.assertRaises(ValueError):
            prefix_exact(rows, reference_rows, native, reference_native)
        rows[500]['ctrl'] = reference_rows[500]['ctrl']
        native['normal_load_bw'][24999, 5] = .1
        with self.assertRaises(ValueError):
            prefix_exact(rows, reference_rows, native, reference_native)

    def test_health_rejection_and_full_shape_screen_separate_from_yaw(self):
        complete = {'actual_physical_s': 9.5, 'stop_reason': 'duration',
                    'controls': CONTROLS, 'native_samples': 95000,
                    'native_nonfoot_samples': 0, 'minimum_upright': .95, 'fault': None}
        self.assertEqual(arm_terminal(complete),
                         ('valid_complete_body_only_diagnostic', 'evaluate', True))
        self.assertEqual(arm_terminal({**complete, 'stop_reason': 'health_guard'}),
                         ('valid_measured_native_health_rejection', 'not_supported', False))
        self.assertEqual(arm_terminal({**complete, 'actual_physical_s': None})[1], 'not_tested')
        quality = {'events': {leg: {'complete_cycles': 10, 'period_cv': .2} for leg in LEGS},
                   'pair_phase': {pair: {'mean_cycles': .5, 'resultant_length': .7}
                                  for pair in ('lf-rf', 'lm-rm', 'lh-rh')}}
        metrics = {'quality': quality,
                   'four_knee_widths_deg': {'lf': 25., 'rf': 30., 'lh': 40., 'rh': 35.},
                   'late_yaw_drift_rad': 100.}
        self.assertTrue(shape_screen(metrics)['supported'])  # yaw is descriptive
        quality['events']['lh']['period_cv'] = .31
        self.assertFalse(shape_screen(metrics)['supported'])

    def test_resource_stops_second_arm_on_limit_or_unknown_clock(self):
        ledger = {'actual_physical_s': 9.5,
                  'shape_only': {'wall_s_including_late_diagnostics': 151.,
                                 'wall_s_inclusive_close_persistence': 100.},
                  'shape_plus_timing': None}
        self.assertEqual(resource_screen(ledger, 200.), 'limit_exceeded_or_unverifiable')
        ledger['shape_only']['wall_s_including_late_diagnostics'] = 149.
        self.assertEqual(resource_screen(ledger, 200.), 'within_limits')
        ledger['actual_physical_s'] = None
        self.assertEqual(resource_screen(ledger, 200.), 'limit_exceeded_or_unverifiable')


if __name__ == '__main__':
    unittest.main()
