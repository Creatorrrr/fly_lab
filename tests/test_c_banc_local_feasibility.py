"""Small CPU-only E120 local LP, cut, certificate and failure tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from tools import analyze_c_banc_m3_local_feasibility as mod


def toy_origin():
    q = np.zeros((2, 99, 7), dtype=np.float64)
    z = np.zeros((2, 99), dtype=np.float64)
    q[0, 0, 0] = .03998
    q[0, 1, 0] = -.038
    q[1, 2, 1] = .024
    z[0, 0] = .023
    z[1, 1] = -.022
    v = np.zeros((2, 99, 2), dtype=np.float64)
    v[0, :3, 0] = [1., 2., -1.]
    v[1, :4, 1] = [1., -1., 2., .5]
    qco = np.zeros((7, 2), dtype=np.float64)
    qco[0] = [1., .2]
    qco[1] = [.1, -.3]
    return {'leg': 'lf', 'source_sha256': 'synthetic',
        'delta': np.array([0., 0.]),
        'bounds': np.array([[-.02, .02], [-.005, .006]]),
        'losses': np.array([1., 1.2]),
        'loss_gradient': np.array([[-1., .1], [.5, -.2]]),
        'q_error': q, 'tendon_error': z, 'v': v,
        'q_coefficient': qco,
        'ltm_coefficient': np.array([.1, .5]),
        'ltm_gain': np.ones((2, 99))}


def toy_ledger():
    return {'lp_input_raw_saved': 0, 'lp_started': 0,
        'lp_returned': 0, 'lp_materialized': 0,
        'lp_raw_saved': 0, 'active_lp': None}


class LocalFeasibilityTests(unittest.TestCase):
    def test_finite_t_bound_and_3174_ordered_rows(self):
        base = mod.base_problem(toy_origin())
        self.assertEqual(base['h_lower'].tolist(), [-.01, -.005])
        self.assertEqual(base['h_upper'].tolist(), [.01, .006])
        self.assertTrue(np.isfinite(base['upper']).all())
        self.assertGreater(base['upper'][-1],
            np.max(base['losses']+np.maximum(
                base['loss_gradient']*base['h_lower'],
                base['loss_gradient']*base['h_upper']).sum(axis=1)))
        center = np.zeros(2)
        cuts = [mod.rms_cut(base, metric, context, center, 0)
                for metric, context in mod.RMS_ORDER]
        shape = mod.build_lp(base, 'shape_box')
        startup = mod.build_lp(base, 'startup_box', cuts)
        self.assertEqual(shape['A_ub'].shape, (2, 3))
        self.assertEqual(startup['A_ub'].shape, (3174, 3))
        np.testing.assert_array_equal(startup['A_ub'][:2, :2],
                                      base['loss_gradient'])
        self.assertEqual(startup['row_labels'][2], 'plus_q_0')
        self.assertEqual(startup['row_labels'][1388], 'minus_q_0')
        self.assertEqual(startup['row_labels'][2774], 'plus_z_0')
        self.assertEqual(startup['row_labels'][2972], 'minus_z_0')
        self.assertEqual(startup['row_labels'][3170], 'rms_00_q_E110')
        self.assertEqual(startup['row_labels'][3173], 'rms_03_z_E113')
        self.assertAlmostEqual(startup['b_ub'][2],
                               np.sqrt(mod.CAP)-.03998/.04)
        self.assertAlmostEqual(startup['b_ub'][1388],
                               np.sqrt(mod.CAP)+.03998/.04)
        self.assertAlmostEqual(startup['b_ub'][2774],
                               np.sqrt(mod.CAP)-.023/.025)

    def test_sample_switch_and_exact_cap_without_padding(self):
        base = mod.base_problem(toy_origin())
        zero = mod.strict_feasible(base, np.zeros(2), 'startup_box')
        self.assertTrue(zero['feasible'])
        h = np.array([.005, 0.])
        positive = mod.strict_feasible(base, h, 'startup_box')
        self.assertFalse(positive['startup_feasible'])
        self.assertGreater(positive['ratios'][0, 1], mod.CAP)
        negative = mod.strict_feasible(base, -h, 'startup_box')
        self.assertFalse(negative['startup_feasible'])
        self.assertLess(negative['q_residual'][0, 1, 0],
                        zero['q_residual'][0, 1, 0])
        self.assertTrue(mod.strict_feasible(base, h, 'shape_box')['feasible'])

    def test_four_rms_supporting_cuts_and_padding_are_outer(self):
        base = mod.base_problem(toy_origin())
        center = np.array([.003, -.001])
        other = np.array([-.004, .002])
        for metric, context in mod.RMS_ORDER:
            cut = mod.rms_cut(base, metric, context, center, 3)
            g_center, grad, _ = mod.rms_state(base, metric, context, center)
            g_other, _, _ = mod.rms_state(base, metric, context, other)
            np.testing.assert_array_equal(cut['gradient'], grad)
            self.assertGreaterEqual(g_other+1e-15,
                                    g_center+grad@(other-center))
            self.assertAlmostEqual(cut['rhs'], mod.CAP-g_center+
                                   grad@center+cut['padding'])
            self.assertGreater(cut['padding'], 0.)

    def test_exact_sixty_step_ray_repairs_startup_and_box(self):
        base = mod.base_problem(toy_origin())
        h = np.array([.01, .006])
        repair = mod.ray_repair(base, h, 'startup_box')
        self.assertEqual(repair['bisection_steps'], 60)
        self.assertLess(repair['alpha'], 1.)
        self.assertFalse(repair['raw_feasibility']['feasible'])
        self.assertTrue(repair['repaired_feasibility']['feasible'])
        np.testing.assert_allclose(repair['repaired_h'], repair['alpha']*h)
        shape_h = np.array([.01+1e-8, 0.])
        shape = mod.ray_repair(base, shape_h, 'shape_box')
        self.assertEqual(shape['bisection_steps'], 60)
        self.assertTrue(shape['repaired_feasibility']['box_feasible'])
        self.assertLessEqual(shape['repaired_h'][0], base['h_upper'][0])

    def test_dual_sign_projection_and_finite_box_allowance(self):
        base = mod.base_problem(toy_origin())
        problem = mod.build_lp(base, 'shape_box')
        result = linprog(problem['c'], A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=list(zip(problem['lower'], problem['upper'])),
            method='highs-ds', options=mod.LP_OPTIONS.copy())
        self.assertTrue(result.success, result.message)
        raw = mod.returned_arrays(result)
        cert = mod.dual_certificate(problem, raw)
        self.assertTrue(np.all(cert['marginal_projected'] <= 0.))
        self.assertLessEqual(cert['lower_bound'],
                             float(result.fun)+base['p0'])
        self.assertGreater(cert['numerical_allowance'], 0.)
        self.assertTrue(np.isfinite(cert['stationarity_remainder']).all())
        raw['ineqlin_marginal'][0] = 1e-12
        projected = mod.dual_certificate(problem, raw)
        self.assertEqual(projected['marginal_projected'][0], 0.)
        self.assertEqual(projected['marginal_raw'][0], 1e-12)

    def test_toy_oa_is_bounded_and_preserves_all_lp_inputs(self):
        base = mod.base_problem(toy_origin())
        ledger = toy_ledger()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            report = mod.analyze_leg(base, out, ledger, lambda: None,
                                     lambda: None)
            self.assertTrue((out/'lp-lf-shape_box-input.npz').exists())
            self.assertTrue((out/'lp-lf-startup_box-01-input.npz').exists())
            self.assertTrue((out/'comparison-lf.json').exists())
            self.assertGreaterEqual(len(report['startup_box']['iterations']), 1)
            self.assertLessEqual(len(report['startup_box']['iterations']), 8)
            self.assertLessEqual(ledger['lp_started'], 9)
            self.assertEqual(ledger['lp_started'], ledger['lp_raw_saved'])
            self.assertLessEqual(report['startup_box']['lower_bound'],
                                 report['startup_box']['upper_bound']+
                                 1e-6*(1+abs(base['origin_objective'])))

    def test_start_receipt_failure_prevents_toy_solver_call(self):
        problem = mod.build_lp(mod.base_problem(toy_origin()), 'shape_box')
        ledger = toy_ledger()
        primary = OSError('synthetic durable start receipt blocked')
        saved = []

        def persist():
            saved.append(True)
            if len(saved) == 2:
                raise primary

        calls = []
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                mod.solve_recorded_lp(problem, 'toy-start-fail', out, ledger,
                    persist, lambda: None,
                    solver=lambda *_args, **_kw: calls.append(True))
            self.assertTrue((out/'toy-start-fail-input.npz').exists())
            self.assertTrue((out/'error-raw-toy-start-fail/input/c.npy').exists())
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, [])
        self.assertEqual(ledger['lp_started'], 0)
        self.assertEqual(ledger['lp_returned'], 0)

    def test_post_return_receipt_failure_keeps_result_and_primary(self):
        problem = mod.build_lp(mod.base_problem(toy_origin()), 'shape_box')
        result = linprog(problem['c'], A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=list(zip(problem['lower'], problem['upper'])),
            method='highs-ds', options=mod.LP_OPTIONS.copy())
        primary = OSError('synthetic returned receipt blocked')
        ledger = toy_ledger()
        saved = []

        def persist():
            saved.append(True)
            if len(saved) == 3:
                raise primary

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                mod.solve_recorded_lp(problem, 'toy-return-fail', out, ledger,
                    persist, lambda: None, solver=lambda *_args, **_kw: result)
            destination = out/'error-raw-toy-return-fail'
            self.assertTrue((destination/'returned/x.npy').exists())
            metadata = json_load(destination/'error-metadata.json')
            self.assertEqual(metadata['status'], 0)
            self.assertTrue(metadata['success'])
            self.assertIsNotNone(metadata['fun'])
        self.assertIs(caught.exception, primary)
        self.assertEqual(ledger['lp_started'], 1)
        self.assertEqual(ledger['lp_returned'], 1)
        self.assertEqual(ledger['lp_materialized'], 0)

    def test_returned_raw_save_failure_rescues_arrays_no_retry(self):
        problem = mod.build_lp(mod.base_problem(toy_origin()), 'shape_box')
        result = linprog(problem['c'], A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=list(zip(problem['lower'], problem['upper'])),
            method='highs-ds', options=mod.LP_OPTIONS.copy())
        primary = OSError('synthetic returned npz locked')
        real_save = mod.np.savez_compressed
        ledger = toy_ledger()

        def save(path, **arrays):
            if str(path).endswith('-returned.npz'):
                raise primary
            return real_save(path, **arrays)

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod.np, 'savez_compressed', side_effect=save):
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                mod.solve_recorded_lp(problem, 'toy-save-fail', out, ledger,
                    lambda: None, lambda: None,
                    solver=lambda *_args, **_kw: result)
            destination = out/'error-raw-toy-save-fail'
            self.assertTrue((destination/'returned/x.npy').exists())
            self.assertTrue((destination/'input/A_ub.npy').exists())
            self.assertTrue(json_load(destination/'error-metadata.json')['success'])
        self.assertIs(caught.exception, primary)
        self.assertEqual(ledger['lp_materialized'], 1)
        self.assertEqual(ledger['lp_raw_saved'], 0)

    def test_final_result_save_crossing_deadline_preserves_result_and_wall(self):
        clock = [0.]
        real_save = mod.save_json

        def delayed_save(path, value):
            real_save(path, value)
            if Path(path).name == 'result.json':
                clock[0] = 81.

        def fake_base(origin):
            return {'leg': origin['leg'], 'source_sha256': 'synthetic',
                'origin_objective': 1., 'h_lower': np.zeros(1),
                'h_upper': np.ones(1), 'upper': np.ones(2)}

        def fake_leg(_base, _out, ledger, _persist, _deadlines):
            for key in ('lp_input_raw_saved', 'lp_started', 'lp_returned',
                        'lp_materialized', 'lp_raw_saved'):
                ledger[key] += 2
            return {'synthetic': True}

        inputs = {'records': {leg: {'leg': leg} for leg in mod.LEGS},
            'freeze': {'input_sha256': {}, 'source_sha256': {}},
            'manifest_sha256': 'synthetic'}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(mod, 'reviewed_inputs', return_value=inputs), \
             patch.object(mod, 'base_problem', side_effect=fake_base), \
             patch.object(mod, 'analyze_leg', side_effect=fake_leg), \
             patch.object(mod.shutil, 'copy2'), \
             patch.object(mod, 'sha256', return_value='synthetic'), \
             patch.object(mod, 'save_json', side_effect=delayed_save), \
             patch.dict(mod.os.environ, {'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}):
            out = Path(temp)/'e120-synthetic'
            with self.assertRaises(TimeoutError):
                mod.run_once(Path(temp), out)
            self.assertTrue((out/'result.json').exists())
            ledger = json_load(out/'execution-ledger.json')
            self.assertEqual(ledger['stage'], 'error')
            self.assertGreaterEqual(ledger['process_wall_s'], 81.)

    def test_materialization_and_rescue_failure_preserve_first_error(self):
        base = mod.base_problem(toy_origin())
        problem = mod.build_lp(base, 'shape_box')
        ledger = toy_ledger()
        primary = MemoryError('primary returned materialization failure')
        secondary = MemoryError('secondary returned rescue failure')
        result = SimpleNamespace(status=0, success=True, message='synthetic',
                                 nit=0, fun=1.)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod, 'returned_arrays', side_effect=[primary,
                                                                secondary]):
            out = Path(temp)
            with self.assertRaises(MemoryError) as caught:
                mod.solve_recorded_lp(problem, 'toy-double-failure', out,
                    ledger, lambda: None, lambda: None,
                    solver=lambda *_args, **_kw: result)
            destination = out/'error-raw-toy-double-failure'
            self.assertTrue((destination/'input/c.npy').exists())
            metadata = json_load(destination/'error-metadata.json')
            self.assertEqual(metadata['status'], 0)
            self.assertTrue(metadata['success'])
            self.assertEqual(metadata['primary_message'], str(primary))
        self.assertIs(caught.exception, primary)
        self.assertTrue(any('secondary returned rescue' in note for note in
                            getattr(primary, '__notes__', [])))
        self.assertEqual(ledger['lp_started'], 1)
        self.assertEqual(ledger['lp_returned'], 1)
        self.assertEqual(ledger['lp_materialized'], 0)


def json_load(path):
    import json
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
