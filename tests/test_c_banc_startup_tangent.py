"""Synthetic E116 LP/receipt tests; never solve a saved E115 gradient."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from tools.analyze_c_banc_m3_startup_tangent import (
    SELECTOR_CAP,
    SOLVER_OPTIONS,
    build_lp,
    returned_arrays,
    run_once,
    solve_recorded_lp,
    validate_solution,
    warning_rows,
)


def toy_origin():
    jac = np.zeros((10, 1), dtype=np.float64)
    jac[0, 0] = 1.
    jac[1, 0] = -1.
    jac[2, 0] = 1.
    ratios = np.zeros((2, 4))
    ratios[0, 0] = SELECTOR_CAP
    return {'leg': 'lf', 'delta': np.array([0.]),
        'losses': np.array([1., 2.]), 'ratios': ratios,
        'jacobian': jac, 'bounds': np.array([[-.1, .1]]),
        'source_sha256': 'synthetic'}


def toy_result(problem):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = linprog(problem['c'], A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=list(zip(problem['lower'][:-1],
                            problem['upper'][:-1]))+[(0., None)],
            method='highs-ds', options=SOLVER_OPTIONS.copy())
    return result, warning_rows(caught)


def ledger():
    return {'lp_input_raw_saved': 0, 'lp_started': 0,
        'lp_returned': 0, 'lp_raw_saved': 0, 'active_lp': None}


class StartupTangentTests(unittest.TestCase):
    def test_lp_rows_objective_prior_and_bounds_order(self):
        origin = toy_origin()
        origin['delta'] = np.array([.1])
        origin['bounds'] = np.array([[-.1, .1]])
        shape = build_lp(origin, 'shape_box')
        startup = build_lp(origin, 'startup_box')
        np.testing.assert_array_equal(shape['A_ub'], [[1., -1.], [-1., -1.]])
        np.testing.assert_array_equal(shape['b_ub'], [-1., -2.])
        np.testing.assert_array_equal(startup['A_ub'][:2], shape['A_ub'])
        np.testing.assert_array_equal(startup['A_ub'][2:, :-1],
                                      origin['jacobian'][2:])
        np.testing.assert_array_equal(startup['A_ub'][2:, -1], 0.)
        np.testing.assert_array_equal(startup['b_ub'][2:],
                                      SELECTOR_CAP-origin['ratios'].ravel())
        self.assertEqual(startup['row_labels'], ['loss_E110', 'loss_E113',
            'startup_E110_q_RMS', 'startup_E110_q_max',
            'startup_E110_LTM_RMS', 'startup_E110_LTM_max',
            'startup_E113_q_RMS', 'startup_E113_q_max',
            'startup_E113_LTM_RMS', 'startup_E113_LTM_max'])
        self.assertEqual(shape['upper'][0], 0.)
        self.assertEqual(shape['lower'][0], -.01)
        self.assertEqual(shape['lower'][-1], 0.)
        self.assertTrue(np.isinf(shape['upper'][-1]))
        self.assertAlmostEqual(shape['prior'], 6e-5*.01/1572)
        self.assertAlmostEqual(shape['c'][0], 1.2e-4*.1/1572)
        self.assertEqual(shape['c'][-1], 1.)

    def test_bad_bounds_and_variant_rejected_before_solver(self):
        origin = toy_origin()
        with self.assertRaisesRegex(ValueError, 'variant'):
            build_lp(origin, 'wrong')
        origin['delta'] = np.array([.11])
        with self.assertRaisesRegex(ValueError, 'origin'):
            build_lp(origin, 'shape_box')

    def test_analytical_shape_vs_startup_and_kkt_signs(self):
        origin = toy_origin()
        shape = build_lp(origin, 'shape_box')
        startup = build_lp(origin, 'startup_box')
        shape_result, shape_warnings = toy_result(shape)
        startup_result, startup_warnings = toy_result(startup)
        self.assertEqual(shape_result.status, 0)
        self.assertEqual(startup_result.status, 0)
        shape_raw = returned_arrays(shape_result)
        startup_raw = returned_arrays(startup_result)
        shape_check = validate_solution(shape, shape_result, shape_raw,
                                        shape_warnings)
        startup_check = validate_solution(startup, startup_result,
                                          startup_raw, startup_warnings)
        self.assertAlmostEqual(shape_result.x[0], .01)
        self.assertAlmostEqual(shape_result.fun, 1.99)
        self.assertAlmostEqual(startup_result.x[0], 0.)
        self.assertAlmostEqual(startup_result.fun, 2.)
        self.assertAlmostEqual(startup_result.fun-shape_result.fun, .01)
        self.assertFalse(shape_check['startup_prediction_feasible'])
        self.assertTrue(startup_check['startup_prediction_feasible'])
        self.assertEqual(shape_check['active_upper_h_count'], 1)
        self.assertEqual(startup_check['row_labels'][2], 'startup_E110_q_RMS')
        self.assertTrue(all(shape_check['checks'].values()))
        self.assertTrue(all(startup_check['checks'].values()))
        self.assertLessEqual(np.max(shape_raw['ineqlin_marginal']), 0.)
        self.assertGreaterEqual(np.min(shape_raw['lower_marginal']), 0.)
        self.assertLessEqual(np.max(shape_raw['upper_marginal']), 0.)
        self.assertTrue(all(row['expected_threads_forwarding']
                            for row in shape_warnings+startup_warnings))

    def test_wrong_dual_sign_and_stationarity_rejected(self):
        problem = build_lp(toy_origin(), 'startup_box')
        result, seen = toy_result(problem)
        raw = returned_arrays(result)
        raw['ineqlin_marginal'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'KKT'):
            validate_solution(problem, result, raw, seen)

    def test_unexpected_warning_rejected_after_result_exists(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, seen = toy_result(problem)
        seen.append({'category': 'RuntimeWarning', 'message': 'unexpected',
                     'expected_threads_forwarding': False})
        with self.assertRaisesRegex(ValueError, 'warning'):
            validate_solution(problem, result, returned_arrays(result), seen)

    def test_started_receipt_failure_prevents_solver_and_rolls_back(self):
        problem = build_lp(toy_origin(), 'shape_box')
        state, calls = ledger(), []
        original = OSError('synthetic started ledger locked')
        def persist():
            if state['lp_started']:
                raise original
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                solve_recorded_lp(problem, out, state, persist,
                    lambda: None, solver=lambda *_a, **_k: calls.append(True))
            self.assertTrue((out/'lp-lf-shape_box-input.npz').exists())
            self.assertTrue((out/'error-raw-lp-lf-shape_box/input/c.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertFalse(calls)
        self.assertEqual(state['lp_input_raw_saved'], 1)
        self.assertEqual(state['lp_started'], 0)
        self.assertEqual(state['lp_returned'], 0)
        self.assertEqual(state['start_receipt_failed'], 'lp-lf-shape_box')

    def test_input_raw_failure_prevents_solver_and_rescues_matrix(self):
        problem = build_lp(toy_origin(), 'shape_box')
        state, calls = ledger(), []
        original = OSError('synthetic input archive locked')
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.analyze_c_banc_m3_startup_tangent.np.savez_compressed',
                   side_effect=original):
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                solve_recorded_lp(problem, out, state, lambda: None,
                    lambda: None, solver=lambda *_a, **_k: calls.append(True))
            rescued = out/'error-raw-lp-lf-shape_box/input'
            self.assertTrue((rescued/'A_ub.npy').exists())
            self.assertTrue((rescued/'b_ub.npy').exists())
            self.assertTrue((rescued/'row_labels.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertFalse(calls)
        self.assertEqual(state['lp_started'], 0)

    def test_returned_raw_save_failure_rescues_all_available_duals(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, _ = toy_result(problem)
        result.nit, result.fun = 7, 2.
        state, calls = ledger(), []
        original = OSError('synthetic returned archive locked')
        actual_save = np.savez_compressed
        def selective_save(path, **kwargs):
            if str(path).endswith('-returned.npz'):
                raise original
            return actual_save(path, **kwargs)
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.analyze_c_banc_m3_startup_tangent.np.savez_compressed',
                   side_effect=selective_save):
            out = Path(temp)
            def solver(*_args, **_kwargs):
                calls.append(True)
                return result
            with self.assertRaises(OSError) as caught:
                solve_recorded_lp(problem, out, state, lambda: None,
                                  lambda: None, solver=solver)
            rescued = out/'error-raw-lp-lf-shape_box/returned'
            for key in ('x', 'ineqlin_residual', 'ineqlin_marginal',
                        'lower_residual', 'lower_marginal',
                        'upper_residual', 'upper_marginal'):
                self.assertTrue((rescued/f'{key}.npy').exists(), key)
            metadata = json.loads((out/'error-raw-lp-lf-shape_box/'
                'error-metadata.json').read_text())
            self.assertEqual({key: metadata[key] for key in
                ('status', 'success', 'nit', 'fun')},
                {'status': 0, 'success': True, 'nit': 7, 'fun': 2.})
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['lp_started'], 1)
        self.assertEqual(state['lp_returned'], 1)
        self.assertEqual(state['lp_raw_saved'], 0)

    def test_returned_receipt_failure_rescues_result_before_raw_save(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, _ = toy_result(problem)
        result.nit, result.fun = 7, 2.
        state = ledger()
        original = OSError('synthetic returned receipt locked')
        def persist():
            if state['lp_returned'] == 1 and state['lp_raw_saved'] == 0:
                raise original
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                solve_recorded_lp(problem, out, state, persist, lambda: None,
                                  solver=lambda *_a, **_k: result)
            rescued = out/'error-raw-lp-lf-shape_box/returned'
            self.assertTrue((rescued/'x.npy').exists())
            self.assertTrue((rescued/'upper_marginal.npy').exists())
            metadata = json.loads((out/'error-raw-lp-lf-shape_box/'
                'error-metadata.json').read_text())
            self.assertEqual({key: metadata[key] for key in
                ('status', 'success', 'message', 'nit', 'fun')},
                {'status': 0, 'success': True, 'message': result.message,
                 'nit': 7, 'fun': 2.})
        self.assertIs(caught.exception, original)
        self.assertEqual(state['lp_started'], 1)
        self.assertEqual(state['lp_returned'], 1)
        self.assertEqual(state['lp_raw_saved'], 0)

    def test_rescue_secondary_preserves_original_error(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, _ = toy_result(problem)
        original = OSError('primary returned save locked')
        actual_savez = np.savez_compressed
        actual_save = np.save
        def fail_npz(path, **kwargs):
            if str(path).endswith('-returned.npz'):
                raise original
            return actual_savez(path, **kwargs)
        def fail_one(path, value, **kwargs):
            if Path(path).name == 'x.npy':
                raise OSError('secondary x rescue locked')
            return actual_save(path, value, **kwargs)
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.analyze_c_banc_m3_startup_tangent.np.savez_compressed',
                   side_effect=fail_npz), \
             patch('tools.analyze_c_banc_m3_startup_tangent.np.save',
                   side_effect=fail_one):
            with self.assertRaises(OSError) as caught:
                solve_recorded_lp(problem, Path(temp), ledger(), lambda: None,
                                  lambda: None,
                                  solver=lambda *_a, **_k: result)
            rescued = Path(temp)/'error-raw-lp-lf-shape_box/returned'
            self.assertFalse((rescued/'x.npy').exists())
            self.assertTrue((rescued/'lower_marginal.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertTrue(any('secondary x rescue locked' in note
                            for note in original.__notes__))

    def test_returned_status_one_raw_before_failure_without_retry(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, _ = toy_result(problem)
        result.status, result.success, result.message = 1, False, 'time limit'
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def solver(*_args, **_kwargs):
                calls.append(True)
                return result
            state = ledger()
            with self.assertRaisesRegex(ValueError, 'did not finish'):
                solve_recorded_lp(problem, out, state, lambda: None,
                                  lambda: None, solver=solver)
            self.assertTrue((out/'lp-lf-shape_box-returned.npz').exists())
            meta = json.loads((out/'lp-lf-shape_box-returned.json').read_text())
            self.assertEqual(meta['status'], 1)
            self.assertEqual(state['lp_raw_saved'], 1)
        self.assertEqual(len(calls), 1)

    def test_postreturn_deadline_keeps_raw_and_original_timeout(self):
        problem = build_lp(toy_origin(), 'shape_box')
        result, _ = toy_result(problem)
        state, clocks = ledger(), []
        original = TimeoutError('synthetic E116 terminal wall')
        def deadline():
            clocks.append(True)
            if len(clocks) == 2:
                raise original
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(TimeoutError) as caught:
                solve_recorded_lp(problem, out, state, lambda: None,
                    deadline, solver=lambda *_a, **_k: result)
            self.assertTrue((out/'lp-lf-shape_box-returned.npz').exists())
            self.assertEqual(state['lp_started'], 1)
            self.assertEqual(state['lp_returned'], 1)
            self.assertEqual(state['lp_raw_saved'], 1)
        self.assertIs(caught.exception, original)

    def test_outer_preflight_error_has_fresh_wall_and_no_lp(self):
        clock = [0.]
        original = RuntimeError('synthetic preflight failed')
        def fail_inputs(_):
            clock[0] = 4.
            raise original
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.analyze_c_banc_m3_startup_tangent.shutil.copy2'), \
             patch('tools.analyze_c_banc_m3_startup_tangent.reviewed_inputs',
                   side_effect=fail_inputs), \
             patch('tools.analyze_c_banc_m3_startup_tangent.time.monotonic',
                   side_effect=lambda: clock[0]):
            out = Path(temp)/'out'
            with self.assertRaises(RuntimeError) as caught:
                run_once(Path(temp), out)
            saved = json.loads((out/'execution-ledger.json').read_text())
        self.assertIs(caught.exception, original)
        self.assertEqual(saved['process_wall_s'], 4.)
        self.assertEqual(saved['lp_started'], 0)

    def test_full_runner_uses_exactly_twelve_toy_lps_and_no_calibration(self):
        records = {name: dict(toy_origin(), leg=name) for name in
                   ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')}
        fixture = {'records': records, 'freeze': {'input_sha256': {}},
                   'manifest_sha256': 'synthetic'}
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.analyze_c_banc_m3_startup_tangent.shutil.copy2'), \
             patch('tools.analyze_c_banc_m3_startup_tangent.reviewed_inputs',
                   return_value=fixture), \
             patch.dict(os.environ, {'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}):
            out = Path(temp)/'out'
            result = run_once(Path(temp), out)
            self.assertEqual(result['status'],
                             'completed_saved_gradient_diagnostic')
            saved = json.loads((out/'execution-ledger.json').read_text())
            self.assertEqual(saved['lp_input_raw_saved'], 12)
            self.assertEqual(saved['lp_started'], 12)
            self.assertEqual(saved['lp_returned'], 12)
            self.assertEqual(saved['lp_raw_saved'], 12)
            self.assertEqual(saved['conditional_responses'], 0)
            self.assertEqual(saved['actual_neural_s'], 0.)
            self.assertEqual(saved['actual_physical_s'], 0.)
            self.assertEqual(len(list(out.glob('lp-*-input.npz'))), 12)
            self.assertEqual(len(list(out.glob('lp-*-returned.npz'))), 12)
            self.assertFalse((out/'calibration.npz').exists())


if __name__ == '__main__':
    unittest.main()
