"""E109 native tau, optimizer, provenance, and lifecycle tests without a model."""

import ast
import inspect
import json
import math
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import OptimizeResult

from tools import fit_c_banc_m3_motor_kinetics as runner
from tools.banc_m3_causal_recruitment import linear_response
from tools.fit_c_banc_m3_block_constrained_recruitment import (
    NativeCache,
    choose_records,
    shadow_body_lifetime,
)
from tools.fit_c_banc_m3_constrained_startup import (
    native_shape_screen,
    startup_screen,
)
from tools.fit_c_banc_m3_motor_kinetics import (
    BASE,
    OPTIMIZER_RESERVE,
    SELECTOR_RESERVE,
    main,
    native_tau_patch,
    resource_screen,
    solve_block,
)


def analytic_piecewise_linear(forcing, dt, tau):
    result = np.zeros_like(forcing, dtype=np.float64)
    beta = np.exp(-dt/tau)
    alpha = 1-beta
    c = 1-alpha/(dt/tau)
    for t in range(1, len(forcing)):
        result[t] = (beta*result[t-1] + (alpha-c)*forcing[t-1]
                     + c*forcing[t])
    return result


class MotorKineticsTests(unittest.TestCase):
    def test_tau_gradient_matches_cpu_causal_recurrence_central_difference(self):
        torch.set_default_dtype(torch.float64)
        forcing = np.array([[0., 1., 2.], [1., 1.5, 3.], [2., .5, 4.],
                            [1.5, 2., 1.], [3., 2.5, 0.]])
        native = np.array([np.float32(.02), np.float32(.014046886935830116),
                           np.float32(.02)], dtype=np.float64)
        log_scale = torch.tensor(math.log(2.3), requires_grad=True)
        response = linear_response(torch.as_tensor(forcing), .005,
                                   torch.as_tensor(native)*log_scale.exp())
        coeff = torch.as_tensor(np.arange(15).reshape(5, 3)/15+1)
        loss = (response*coeff).sum()
        derivative = float(torch.autograd.grad(loss, log_scale)[0])
        def independent(x):
            scale = math.exp(x)
            return float(np.sum(analytic_piecewise_linear(
                forcing, .005, native*scale)*coeff.numpy()))
        h = 1e-5
        origin = float(log_scale.detach())
        finite_difference = (independent(origin+h)
                             -independent(origin-h))/(2*h)
        self.assertAlmostEqual(derivative, finite_difference, delta=3e-8)
        np.testing.assert_allclose(response.detach().numpy(),
            analytic_piecewise_linear(forcing, .005, native*2.3),
            rtol=0, atol=1e-12)

    def test_scale_one_exact_and_428_mapping_matches_parent_independent_fixture(self):
        fixture = np.load(BASE/'parent-e109-tau-mapping-fixture.npz')
        cal = {'tau_ids': fixture['original_patch_ids'],
               'tau_values': fixture['original_patch_tau']}
        mask = np.zeros((6, 391), dtype=bool)
        mask[fixture['motor_leg_group'], np.arange(391)] = True
        ids, values, native, motor = native_tau_patch(cal, fixture['motor_ids'],
            fixture['original_motor_tau'], mask, np.ones(6))
        np.testing.assert_array_equal(ids, fixture['combined_ids'])
        np.testing.assert_array_equal(motor,
            fixture['original_motor_tau'].astype(np.float32))
        self.assertEqual(len(ids), 428)
        self.assertEqual(len(np.unique(ids)), 428)
        self.assertEqual(int(fixture['motor_ids'][np.argmin(fixture['original_motor_tau'])]),
                         92104)
        nonmotor = fixture['nonmotor_original_mask']
        positions = fixture['original_positions'][nonmotor]
        np.testing.assert_array_equal(values[positions],
            fixture['original_patch_tau'][nonmotor])
        np.testing.assert_array_equal(native[positions],
            fixture['original_patch_tau'][nonmotor].astype(np.float32))
        ids, values, native, motor = native_tau_patch(cal, fixture['motor_ids'],
            fixture['original_motor_tau'], mask, fixture['fixture_scales'])
        np.testing.assert_array_equal(values,
            fixture['fixture_calibration_patch_tau'])
        np.testing.assert_array_equal(native,
            fixture['fixture_actual_native_float32_patch_tau'])
        np.testing.assert_array_equal(motor,
            fixture['fixture_actual_float32_motor_tau'])
        with self.assertRaises(ValueError):
            native_tau_patch(cal, fixture['motor_ids'],
                fixture['original_motor_tau'], mask, np.array([.9, 1, 1, 1, 1, 1]))

    def test_bit_exact_native_cache_includes_log_tau_parameter(self):
        seen = []
        cache = NativeCache(lambda x, i: seen.append((i, x.copy())) or {'objective': 1.},
            2, max_calls=2, deadline=time.perf_counter()+10,
            total_deadline=time.perf_counter()+10, on_start=lambda *_: None)
        cache.get(np.array([0., 0.]))
        cache.get(np.array([0., 0.]))
        cache.get(np.array([0., -0.]))
        self.assertEqual(cache.calls, 2)
        self.assertEqual(cache.hits, 1)
        self.assertEqual([v[0] for v in seen], [1, 2])

    def test_optimizer_has_inner_reserve_and_finite_record_selection_margin(self):
        observed = {}
        class Cache:
            def get(self, x):
                return {'objective': 1., 'jac': np.zeros(2),
                    'ratios': np.array([.7, .8, .9, 1.]),
                    'ratio_jac': np.ones((4, 2))}
        def fake_minimize(fun, x0, *, method, jac, constraints, bounds, options):
            observed['method'] = method
            observed['inequalities'] = constraints[0]['fun'](x0)
            observed['jac'] = constraints[0]['jac'](x0)
            observed['bounds'] = bounds
            observed['options'] = options
            return OptimizeResult(success=False, status=9, message='iteration limit',
                nit=1000, nfev=1000, njev=1000, fun=fun(x0), jac=jac(x0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'optimizer.json'
            with patch('tools.fit_c_banc_m3_motor_kinetics.minimize',
                       side_effect=fake_minimize):
                _, record = solve_block(Cache(), np.zeros(2), 1, path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved, record)
            self.assertEqual(saved['message'], 'iteration limit')
        np.testing.assert_array_equal(observed['inequalities'],
            OPTIMIZER_RESERVE-np.array([.7, .8, .9, 1.]))
        np.testing.assert_array_equal(observed['jac'], -np.ones((4, 2)))
        self.assertLess(OPTIMIZER_RESERVE, SELECTOR_RESERVE)
        self.assertEqual(observed['bounds'][-1], (0., math.log(5.)))
        self.assertEqual(observed['options']['maxiter'], 1000)
        self.assertEqual(observed['options']['ftol'], 1e-9)

    def test_selector_margin_and_original_final_gates_remain_separate(self):
        middle = (OPTIMIZER_RESERVE+SELECTOR_RESERVE)/2
        records = [
            {'objective': 1., 'ratios': [middle]*4},
            {'objective': .5, 'ratios': [SELECTOR_RESERVE+1e-6]*4},
        ]
        best_any, best_feasible = choose_records(records)
        self.assertIs(best_any, records[1])
        self.assertIs(best_feasible, records[0])
        baseline = {'q_rms': .1, 'foot_rms': .1, 'LF_row6_rms': .1,
                    'per_leg_q_rms': [1.]*6}
        final = {'q_rms': .08, 'foot_rms': .08, 'LF_row6_rms': .05,
                 'per_leg_q_rms': [1.05]*6}
        self.assertTrue(native_shape_screen(final, baseline)['supported'])
        final['LF_row6_rms'] = .05001
        self.assertFalse(native_shape_screen(final, baseline)['supported'])
        times = np.arange(2001)*.005
        mask = np.zeros(2001, dtype=bool)
        mask[1:100] = True
        q = np.zeros((2001, 42))
        t = np.zeros((2001, 6))
        self.assertTrue(startup_screen(q, t, np.zeros((99, 42)),
            np.zeros((99, 6)), times, mask)['supported'])
        q[1, 0] = .050001
        self.assertFalse(startup_screen(q, t, np.zeros((99, 42)),
            np.zeros((99, 6)), times, mask)['supported'])

    def test_nonfinite_scipy_return_is_saved_before_rejection(self):
        bad = OptimizeResult(success=False, status=2, message='bad native return',
            nit=1, nfev=3, njev=2, fun=np.nan, jac=np.array([np.nan, np.inf]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'optimizer.json'
            with (patch('tools.fit_c_banc_m3_motor_kinetics.minimize',
                        return_value=bad),
                  self.assertRaisesRegex(ValueError, 'preserved')):
                solve_block(None, np.zeros(2), 1, path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['message'], 'bad native return')
            self.assertEqual(saved['stop'], 'scipy_nonfinite_result')
            self.assertIsNone(saved['objective'])
            self.assertEqual(saved['nfev'], 3)

    def test_exact_prior_charge_and_started_error_cost(self):
        ledger = {'prior_shared_wall_s': 40., 'whole_wall_s': 184.,
            'fit_start_elapsed_s': 5., 'fit_wall_s': 149.,
            'leg_started_evaluations': {'lf': 1000},
            'leg_wall_s': {'lf': 23.9}}
        self.assertEqual(resource_screen(ledger), 'within_limits')
        ledger['whole_wall_s'] = 185.001
        self.assertEqual(resource_screen(ledger), 'limit_exceeded')
        ledger['whole_wall_s'] = 100.
        ledger['leg_started_evaluations']['lf'] = 1001
        self.assertEqual(resource_screen(ledger), 'limit_exceeded')
        ledger['leg_started_evaluations']['lf'] = 1
        ledger['fit_wall_s'] = None
        self.assertEqual(resource_screen(ledger), 'limit_exceeded_or_unverifiable')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'e109'
            with self.assertRaisesRegex(ValueError, 'exactly 40s'):
                main(Path(folder), Path(folder), out, prior_shared_wall_s=39.9)
            self.assertFalse(out.exists())
            with (patch('tools.fit_c_banc_m3_motor_kinetics._run',
                        side_effect=RuntimeError('preflight provenance failure')),
                  self.assertRaisesRegex(RuntimeError, 'preflight provenance failure')):
                main(Path(folder), Path(folder), out, prior_shared_wall_s=40.)
            raw = json.loads((out/'execution-error.json').read_text(encoding='utf-8'))
            cost = json.loads((out/'execution-ledger.json').read_text(encoding='utf-8'))
            self.assertEqual(raw['error_type'], 'RuntimeError')
            self.assertEqual(cost['prior_shared_wall_s'], 40.)
            self.assertGreaterEqual(cost['whole_wall_s'], 0.)
            self.assertEqual(cost['resource_screen'], 'within_limits')

    def test_post_block_summary_save_error_charges_inclusive_leg_wall(self):
        # Execute the production loop tail through main's real error ledger.
        source = inspect.getsource(runner._fit)
        begin = source.index("        ledger['leg_wall_s'][name] = wall\n")
        end = source.index("    ledger['fit_wall_s'] = time.perf_counter()-fit_started", begin)
        actual_tail = textwrap.dedent(source[begin:end])
        compiled = compile(ast.parse(actual_tail), '<production E109 post-block>', 'exec')

        class Clock:
            now = 0.
            def read(self):
                return self.now

        clock = Clock()
        original_save = runner.save_json
        def interrupted_save(path, value):
            if Path(path).name == 'all-block-summaries.json':
                clock.now = 24.3
                raise OSError('mock summary persistence failure')
            original_save(path, value)

        def interrupted_run(_root, _graph, out, _started, ledger):
            ledger['fit_start_elapsed_s'] = 0.
            ledger['active_leg'] = 'lf'
            ledger['active_leg_start_elapsed_s'] = 0.
            ledger['leg_started_evaluations']['lf'] = 1
            ledger['leg_completed_evaluations']['lf'] = 1
            clock.now = 23.7
            scope = {'ledger': ledger, 'name': 'lf', 'wall': 23.7,
                'out': out, 'summaries': {'lf': {'stop': 'scipy_failure'}},
                'block_started': 0., 'best_feasible': None, 'chosen': {},
                'save_json': interrupted_save, 'time': runner.time,
                'LEG_LIMIT_S': runner.LEG_LIMIT_S}
            exec(compiled, scope)  # noqa: S102 - fixed production source under a synthetic clock
            self.fail('Summary persistence must raise before block deactivation')

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'e109'
            with (patch.object(runner.time, 'perf_counter', side_effect=clock.read),
                  patch.object(runner, 'save_json', side_effect=interrupted_save),
                  patch.object(runner, '_run', side_effect=interrupted_run),
                  self.assertRaisesRegex(OSError, 'mock summary persistence failure')):
                main(Path(folder), Path(folder), out, prior_shared_wall_s=40.)
            error = json.loads((out/'execution-error.json').read_text(encoding='utf-8'))
            ledger = json.loads((out/'execution-ledger.json').read_text(encoding='utf-8'))
            self.assertEqual(error['error_type'], 'OSError')
            self.assertEqual(error['resource_screen'], 'limit_exceeded')
            self.assertEqual(ledger['resource_screen'], 'limit_exceeded')
            self.assertGreaterEqual(ledger['leg_wall_s']['lf'], 24.3)
            self.assertGreaterEqual(error['leg_wall_s']['lf'], 24.3)

    def test_shadow_body_closes_on_initial_snapshot_failure(self):
        class FakeBody:
            closed = False
            def snapshot(self):
                raise RuntimeError('snapshot failed')
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as folder:
            body = FakeBody()
            with (self.assertRaisesRegex(RuntimeError, 'snapshot failed'),
                  shadow_body_lifetime(None, Path(folder),
                                       body_factory=lambda _: body)):
                self.fail('unexpected body exposure')
            self.assertTrue(body.closed)
            record = json.loads((Path(folder)/'shadow-body-final.json').read_text(encoding='utf-8'))
            self.assertTrue(record['closed'])


if __name__ == '__main__':
    unittest.main()
