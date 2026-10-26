"""No body construction: E106 phase and recorded-command mapping contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from flylab.body import LEGS
from tools.probe_c_banc_m3_slow_phase_replay import (
    CONTROLS,
    add_physical_advance,
    body_arm,
    inclusive_arm_wall,
    map_commands,
    phase_queries,
    physical_advance_from_clocks,
    planar_initial_heading_forward,
    require_accepted_review,
    require_grid,
    require_source_e1_rates,
    require_source_tendons,
    resource_screen,
    terminal_classification,
)


def clock():
    return np.arange(2001) * .005


class SlowPhaseReplayTests(unittest.TestCase):
    def test_phase_advance_sign_and_integer_peak_counter_invariance(self):
        t = clock()
        base = np.arange(.2, 9.801, .4)
        peaks = [base.copy() for _ in range(6)]
        for leg in (1, 3, 5):
            peaks[leg] = base - .2
        peaks[1] += .02  # lagging actual peaks need a later source query to advance
        offsets = np.array([0., .5, 0., .5, 0., .5])
        expected = np.zeros(6, dtype=int)
        query, _, counters, _ = phase_queries(t, peaks, offsets=offsets,
                                               expected_counters=expected)
        np.testing.assert_array_equal(counters, expected)
        np.testing.assert_array_equal(query[:, 0], t[1:CONTROLS + 1])
        self.assertTrue(np.all(query[600:, 1] > t[601:CONTROLS + 1]))
        # Index the very same physical LM peaks from one preceding peak instead.
        shifted = [p.copy() for p in peaks]
        shifted[1] = np.r_[shifted[1][0] - .4, shifted[1]]
        expected[1] = 1
        reindexed, _, found, _ = phase_queries(t, shifted, offsets=offsets,
                                                expected_counters=expected)
        np.testing.assert_array_equal(found, expected)
        np.testing.assert_allclose(reindexed, query, atol=1e-12, rtol=0)

    def test_recorded_prefix_and_shared_joint_tendon_query(self):
        t = clock()
        query = np.broadcast_to(t[1:CONTROLS + 1, None], (CONTROLS, 6)).copy()
        query[500:, 1] += .02
        source = np.stack([t*(joint+1) for joint in range(42)], axis=1)
        tendons = {leg + '_tarsus': t*(i+1)/10 for i, leg in enumerate(LEGS)}
        tendons.update(abdomen_pitch=t*.01, abdomen_yaw=t*.02)
        mapped, mapped_tendon = map_commands(t, query, source, tendons)
        np.testing.assert_array_equal(mapped[:500], source[1:501])
        for key in tendons:
            np.testing.assert_array_equal(mapped_tendon[key][:500], tendons[key][1:501])
        for joint in range(7, 14):
            np.testing.assert_allclose(mapped[501:, joint], query[501:, 1]*(joint+1))
        np.testing.assert_allclose(mapped_tendon['lm_tarsus'][501:], query[501:, 1]*.2)
        np.testing.assert_array_equal(mapped_tendon['abdomen_pitch'], tendons['abdomen_pitch'][1:CONTROLS+1])
        np.testing.assert_array_equal(mapped[:, :7], source[1:CONTROLS+1, :7])

    def test_reject_invalid_query_grids(self):
        for damage in ('missing', 'nonfinite', 'extrapolate', 'nonincreasing',
                       'big_shift', 'prefix', 'lf'):
            with self.subTest(damage=damage):
                t = clock()
                q = np.broadcast_to(t[1:CONTROLS+1, None], (CONTROLS, 6)).copy()
                if damage == 'missing':
                    t = np.delete(t, 17)
                elif damage == 'nonfinite':
                    q[700, 2] = np.nan
                elif damage == 'extrapolate':
                    q[-1, 2] = 10.001
                elif damage == 'nonincreasing':
                    q[700, 2] = q[699, 2]
                elif damage == 'big_shift':
                    q[700, 2] += .13
                elif damage == 'prefix':
                    q[20, 2] += .005
                else:
                    q[700, 0] += .005
                with self.assertRaises(ValueError):
                    require_grid(t, q)

    def test_planar_initial_heading_ignores_vertical_displacement(self):
        basis = np.eye(3)
        basis[:, 0] = np.array([.6, .8, 0.])
        positions = np.array([[0., 0., 0.], [3., 100., 4.]])
        self.assertEqual(planar_initial_heading_forward(positions, basis, 0, 1), 3.)
        positions[1, 1] = -100.
        self.assertEqual(planar_initial_heading_forward(positions, basis, 0, 1), 3.)

    def test_review_requires_explicit_true_and_empty_defects(self):
        require_accepted_review({'accepted_as_valid_failed_body_diagnostic': True,
                                 'blocking_defects': []})
        for review in ({'accepted_as_valid_failed_body_diagnostic': 1, 'blocking_defects': []},
                       {'accepted_as_valid_failed_body_diagnostic': True},
                       {'accepted_as_valid_failed_body_diagnostic': False, 'blocking_defects': []},
                       {'accepted_as_valid_failed_body_diagnostic': True, 'blocking_defects': ['x']}):
            with self.subTest(review=review), self.assertRaises(ValueError):
                require_accepted_review(review)

    def test_entire_source_rate_and_tendon_arrays_finite(self):
        rates = np.zeros((2000, 6))
        require_source_e1_rates(rates)
        bad_rates = rates.copy()
        bad_rates[-1, -1] = np.nan
        for value in (bad_rates, rates[:-1], rates[:, :-1]):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                require_source_e1_rates(value)
        t = clock()
        query = np.broadcast_to(t[1:CONTROLS+1, None], (CONTROLS, 6)).copy()
        commands = np.zeros((2001, 42))
        tendons = {leg + '_tarsus': np.zeros(2001) for leg in LEGS}
        tendons.update(abdomen_pitch=np.zeros(2001), abdomen_yaw=np.zeros(2001))
        require_source_tendons(tendons)
        corrupt = {key: value.copy() for key, value in tendons.items()}
        corrupt['rh_tarsus'][-1] = np.nan  # outside every E106 source query
        with self.assertRaises(ValueError):
            map_commands(t, query, commands, corrupt)
        missing = dict(tendons)
        del missing['abdomen_yaw']
        with self.assertRaises(ValueError):
            require_source_tendons(missing)

    def test_native_error_accounting_and_measured_health_terminal(self):
        # Observer failed after one native step advanced, before recording it.
        self.assertAlmostEqual(physical_advance_from_clocks(0., .0001, 0), .0001)
        self.assertEqual(inclusive_arm_wall(10., 12., 1.5), 2.)
        with self.assertRaises(ValueError):
            inclusive_arm_wall(10., 11., 1.5)
        baseline = {'stop_reason': 'duration', 'controls': CONTROLS,
                    'native_reference_exact': True}
        self.assertEqual(terminal_classification(baseline, {'stop_reason': 'health_guard'}),
                         ('valid_measured_health_rejection', 'not_supported', False))
        self.assertEqual(terminal_classification(baseline, {'stop_reason': 'execution_error'})[1],
                         'not_tested')
        self.assertEqual(terminal_classification({'stop_reason': 'health_guard'}, None)[1],
                         'not_tested')

    def test_rejected_candidate_inclusive_wall_and_unknown_clock_fail_resource_gate(self):
        baseline = {'wall_s_inclusive_close_persistence': 48.,
                    'wall_s_including_late_diagnostics': 50.}
        candidate = {'stop_reason': 'health_guard',
                     'wall_s_inclusive_close_persistence': 140.,
                     'wall_s_including_late_diagnostics': 151.}
        ledger = {'actual_physical_s': 19., 'baseline': baseline, 'candidate': candidate}
        self.assertNotEqual(resource_screen(ledger, 210.), 'within_limits')
        candidate['wall_s_including_late_diagnostics'] = 149.
        self.assertEqual(resource_screen(ledger, 210.), 'within_limits')
        ledger['actual_physical_s'] = add_physical_advance(9.5, None)
        self.assertIsNone(ledger['actual_physical_s'])
        self.assertNotEqual(resource_screen(ledger, 210.), 'within_limits')

    def test_invalid_physics_clock_still_preserves_raw_and_closes_body(self):
        fake = SimpleNamespace(
            physics_time=Mock(side_effect=[0., float('nan')]),
            _step_physics=Mock(), m=object(), fault=None, close=Mock(),
            mj=SimpleNamespace(MjData=lambda _: object(),
                               mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
                               mj_stateSize=Mock(side_effect=RuntimeError('setup failed'))))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'arm'
            with patch('tools.probe_c_banc_m3_slow_phase_replay.new_body', return_value=fake):
                result = body_arm({}, {'samples': []}, [], {}, out,
                                  baseline=False, original_native=None, whole_start=0.)
            self.assertIsNone(result['actual_physical_s'])
            self.assertEqual(result['controls'], 0)
            self.assertEqual(result['error']['message'], 'setup failed')
            self.assertTrue((out/'native-contact.npz').is_file())
            self.assertEqual(json.loads((out/'trace.json').read_text())['samples'], [])
            fake.close.assert_called_once()
            json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
