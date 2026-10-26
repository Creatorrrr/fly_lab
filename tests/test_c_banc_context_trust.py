"""E115 synthetic CPU checks; no 2001-row response, CUDA fit or body."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.banc_m3_context_response import ContextResponse
from tools.fit_c_banc_m3_context_trust import (
    DeltaCache,
    LegFit,
    LegSoftStop,
    LegStage,
    candidate_weights,
    choose_records,
    efficacy_boundary_roundoff,
    epigraph_terms,
    leg_losses_and_startup,
    native_rounding_comparison,
    run_once,
    selected_edge_values,
    solve_leg,
    trust_bounds,
)


def fake_cached(n=2):
    return {'losses': np.array([1.2, 1.5]),
        'parts': np.array([[.2, .3, .4, .3], [.3, .4, .5, .3]]),
        'ratios': np.array([[.2, .3, .4, .5], [.6, .7, .8, .9]]),
        'jacobian': np.arange(10*n, dtype=np.float64).reshape(10, n)/100}


def fake_stage_response(parameter):
    value = parameter[0]
    response = {key: value*torch.ones((2, 2, 1), dtype=torch.float64)
                for key in ('current', 'motor_rate', 'muscle_activation',
                            'command', 'tendon')}
    return {'response': response,
        'losses': torch.stack((value*value, value+1)),
        'parts': value*torch.ones((2, 4), dtype=torch.float64),
        'ratios': value*torch.ones((2, 4), dtype=torch.float64),
        'feet': value*torch.ones((2, 2, 3), dtype=torch.float64)}


def one_leg_ledger(started=0):
    return {'leg_started_evaluations': {'lf': started},
        'leg_completed_evaluations': {'lf': started},
        'leg_raw_saved_evaluations': {'lf': started},
        'leg_context_pairs_started': {'lf': 2*started},
        'leg_context_pairs_completed': {'lf': 2*started},
        'leg_context_pairs_raw_saved': {'lf': 2*started},
        'leg_conditional_model_s_started': {'lf': 20.*started},
        'leg_conditional_model_s_completed': {'lf': 20.*started},
        'leg_conditional_model_s_raw_saved': {'lf': 20.*started},
        'leg_cache_hits': {'lf': 0}, 'response_call_entered': {'lf': started}}


class ContextTrustTests(unittest.TestCase):
    def test_trust_bounds_and_empty_intersection(self):
        bounds = trust_bounds(np.array([.001, 10., 1.]))
        self.assertAlmostEqual(bounds[0, 0], 0.)
        self.assertAlmostEqual(bounds[1, 1], 0.)
        self.assertAlmostEqual(bounds[2, 0], -math.log(1.1))
        self.assertAlmostEqual(bounds[2, 1], +math.log(1.1))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            trust_bounds(np.array([.0001]))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            trust_bounds(np.array([11.]))
        with self.assertRaisesRegex(ValueError, 'positive'):
            trust_bounds(np.array([np.nan]))

    def test_epigraph_true_worst_and_exact_jacobian(self):
        cached = fake_cached()
        delta = np.array([.02, -.03])
        t = 1.7
        result = epigraph_terms(delta, t, cached)
        self.assertAlmostEqual(result['true_worst_objective'],
                               1.5+result['prior'])
        np.testing.assert_allclose(result['inequalities'][:2], [.5, .2])
        np.testing.assert_allclose(result['inequality_jac'][:2, -1], [1., 1.])
        np.testing.assert_allclose(result['inequality_jac'][2:, -1], 0.)
        np.testing.assert_allclose(result['inequality_jac'][:, :-1],
                                   -cached['jacobian'])
        step = 1e-6
        for i in range(2):
            plus, minus = delta.copy(), delta.copy()
            plus[i] += step
            minus[i] -= step
            numeric = (epigraph_terms(plus, t, cached)['objective']-
                       epigraph_terms(minus, t, cached)['objective'])/(2*step)
            self.assertAlmostEqual(result['objective_jac'][i], numeric,
                                   delta=1e-10)
        self.assertEqual(result['objective_jac'][-1], 1.)

    def test_delta_cache_ignores_t_but_keeps_delta_bit_pattern(self):
        calls, hits = [], []
        def evaluate(delta):
            calls.append(delta.copy())
            return fake_cached(len(delta))
        cache = DeltaCache(evaluate, lambda: hits.append(True))
        first = cache.get(np.array([0., 0., 2.]))
        second = cache.get(np.array([0., 0., 7.]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(first['true_worst_objective'],
                         second['true_worst_objective'])
        self.assertEqual(second['objective']-first['objective'], 5.)
        cache.get(np.array([-0., 0., 7.]))
        self.assertEqual(len(calls), 2)  # literal bytes distinguish signed zero

    def test_selected_calibration_ids_use_mapping_not_sort_order(self):
        values = selected_edge_values(np.array([9, 3, 7]),
                                      np.array([.9, .3, .7]),
                                      np.array([7, 9]))
        np.testing.assert_array_equal(values, [.7, .9])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            selected_edge_values(np.array([9, 9]), np.array([.9, .9]),
                                 np.array([9]))

    def test_two_context_four_parts_and_eight_distinct_startup_ratios(self):
        command = torch.zeros((2, 4, 1), dtype=torch.float64,
                              requires_grad=True)
        tendon = torch.zeros((2, 4, 1), dtype=torch.float64)
        tendon[:, 2:, 0] = .2
        foot = torch.zeros((2, 2, 3), dtype=torch.float64)
        foot[0, 1] = 1.
        foot[1, 1] = 2.
        target_q = torch.zeros((2, 2, 1), dtype=torch.float64)
        target_feet = torch.zeros_like(foot)
        start_q = torch.zeros((2, 1), dtype=torch.float64)
        start_tendon = torch.zeros(2, dtype=torch.float64)
        command.data[0, 1, 0] = .018
        command.data[1, 1, 0] = .04
        parts, ratios = leg_losses_and_startup(command, tendon, foot,
            common=torch.tensor([2, 3]), startup=torch.tensor([1, 2]),
            target_q=target_q, target_feet=target_feet,
            target_start_q=start_q, target_start_tendon=start_tendon,
            q_scale=torch.ones(1), foot_scale=torch.ones(3),
            velocity_scale=torch.ones(3))
        self.assertEqual(parts.shape, (2, 4))
        self.assertEqual(ratios.shape, (2, 4))
        self.assertGreater(parts[1, 1], parts[0, 1])
        self.assertGreater(parts[1, 2], parts[0, 2])
        ratios_np = ratios.detach().numpy()
        self.assertNotEqual(ratios_np[0, 0], ratios_np[1, 0])
        self.assertNotEqual(ratios_np[0, 1], ratios_np[1, 1])
        self.assertEqual(ratios_np[0, 2], ratios_np[1, 2])
        self.assertEqual(ratios_np[0, 3], ratios_np[1, 3])

    def test_batched_autograd_has_ten_gradients(self):
        delta = torch.tensor([.03, -.02], dtype=torch.float64,
                             requires_grad=True)
        metric = torch.stack([delta[0]**2, delta[1]**2,
            *[(i+1)*delta[0]+(i+2)*delta[1] for i in range(8)]])
        jac = torch.autograd.grad(metric, delta,
            grad_outputs=torch.eye(10, dtype=torch.float64),
            is_grads_batched=True)[0]
        self.assertEqual(jac.shape, (10, 2))
        np.testing.assert_allclose(jac.detach().numpy()[:2],
                                   [[.06, 0.], [0., -.04]])
        np.testing.assert_allclose(jac.detach().numpy()[2], [1., 2.])

    def test_batched_autograd_through_actual_fft_response_helper(self):
        response = ContextResponse(time_s=np.arange(4, dtype=np.float64)*.005,
            tau=np.array([.02]), a=np.array([1.]), threshold=np.array([0.]),
            cap=np.array([10.]), pool=np.array([[1.]]),
            moment=np.array([[1.]]), rest=np.array([0.]),
            ltm=np.array([0]), device='cpu')
        source = torch.tensor([[[0.], [1.], [1.], [1.]],
                               [[0.], [2.], [2.], [2.]]], dtype=torch.float64)
        background = torch.zeros((2, 4, 1), dtype=torch.float64)
        delta = torch.tensor([.02], dtype=torch.float64, requires_grad=True)
        weights = torch.exp(delta).reshape(1, 1, 1)
        result = response.evaluate(source, background, weights,
            torch.tensor([0, 1]), torch.tensor([0, 0]))
        command = result['command']
        values = torch.stack([command[i % 2, 1+(i % 3), 0]
                              for i in range(10)])
        jac = torch.autograd.grad(values, delta,
            grad_outputs=torch.eye(10, dtype=torch.float64),
            is_grads_batched=True)[0]
        self.assertEqual(jac.shape, (10, 1))
        self.assertTrue(torch.isfinite(jac).all())
        self.assertTrue(torch.all(jac > 0))

    def test_solver_W_starts_at_actual_float32_and_native_rounding_is_separate(self):
        signed = np.array([-.1, .3], dtype=np.float32)
        efficacy = np.array([1.23456789, 2.45678901])
        actual = (signed*efficacy).astype(np.float32).astype(np.float64)
        delta = np.array([.04, -.05])
        base = torch.tensor(actual.reshape(1, 2), requires_grad=False)
        parameter = torch.tensor(delta, dtype=torch.float64, requires_grad=True)
        candidate = candidate_weights(base, torch.tensor([0, 0]),
                                      torch.tensor([0, 1]), parameter)
        np.testing.assert_allclose(candidate.detach().numpy().ravel(),
                                   actual*np.exp(delta))
        candidate.sum().backward()
        np.testing.assert_allclose(parameter.grad.numpy(),
                                   actual*np.exp(delta))
        new_eff, solver, native, difference = native_rounding_comparison(
            signed, efficacy, actual, delta)
        np.testing.assert_allclose(new_eff, efficacy*np.exp(delta))
        np.testing.assert_allclose(solver, candidate.detach().numpy().ravel())
        np.testing.assert_array_equal(native, native.astype(np.float32).astype(np.float64))
        self.assertTrue(np.any(difference != 0))
        zero = native_rounding_comparison(signed, efficacy, actual,
                                          np.zeros(2))
        np.testing.assert_array_equal(zero[1], actual)
        np.testing.assert_array_equal(zero[2], actual)

    def test_selector_requires_all_eight_and_uses_true_worst_not_old_t(self):
        okay = {'true_worst_objective': 2., 'ratios': [[.1]*4]*2,
                'old_t': 100.}
        bad = {'true_worst_objective': 1.,
               'ratios': [[.1]*4, [.1, .1, .1, 1.]], 'old_t': 0.}
        best_any, best_feasible = choose_records([okay, bad])
        self.assertIs(best_any, bad)
        self.assertIs(best_feasible, okay)
        self.assertEqual(choose_records([bad])[1], None)

    def test_initial_response_charged_once_before_single_solver(self):
        calls = []
        def evaluator(delta):
            calls.append(delta.copy())
            return {'losses': np.array([.1, .2]),
                'parts': np.array([[.025]*4, [.05]*4]),
                'ratios': np.zeros((2, 4)),
                'jacobian': np.zeros((10, len(delta)))}
        fake = SimpleNamespace(cache=DeltaCache(evaluator))
        with patch('tools.fit_c_banc_m3_context_trust.minimize') as minimize:
            minimize.return_value = SimpleNamespace(success=True)
            solve_leg(fake, np.array([[-.1, .1]]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(minimize.call_count, 1)
        args, kwargs = minimize.call_args
        self.assertAlmostEqual(args[1][-1], .21)
        self.assertEqual(kwargs['options']['maxiter'], 500)
        self.assertEqual(len(kwargs['constraints']), 1)

    def test_failed_pre_response_receipt_never_calls_stage(self):
        calls = []
        stage = SimpleNamespace(name='lf', edges=np.array([0]),
            device=torch.device('cpu'), tensors=lambda _: calls.append(True))
        ledger = {'leg_started_evaluations': {'lf': 0},
            'leg_completed_evaluations': {'lf': 0},
            'leg_raw_saved_evaluations': {'lf': 0},
            'leg_context_pairs_started': {'lf': 0},
            'leg_context_pairs_completed': {'lf': 0},
            'leg_context_pairs_raw_saved': {'lf': 0},
            'leg_conditional_model_s_started': {'lf': 0.},
            'leg_conditional_model_s_completed': {'lf': 0.},
            'leg_conditional_model_s_raw_saved': {'lf': 0.},
            'leg_cache_hits': {'lf': 0}, 'response_call_entered': {'lf': 0},
            'active_leg': None, 'active_delta': None, 'active_delta_index': None}
        original = OSError('synthetic ledger locked')
        with tempfile.TemporaryDirectory() as temp:
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: (_ for _ in ()).throw(original),
                lambda _: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.array([0.]))
            with np.load(Path(temp)/'error-input-lf-0001.npz') as saved:
                np.testing.assert_array_equal(saved['delta'], [0.])
        self.assertIs(caught.exception, original)
        self.assertFalse(calls)
        self.assertEqual(ledger['leg_started_evaluations']['lf'], 0)
        self.assertEqual(ledger['leg_conditional_model_s_started']['lf'], 0.)
        self.assertEqual(ledger['leg_context_pairs_started']['lf'], 0)
        self.assertEqual(ledger['response_call_entered']['lf'], 0)
        self.assertEqual(ledger['start_receipt_failed']['leg'], 'lf')

    def test_failure_raw_secondary_preserves_primary(self):
        original = RuntimeError('synthetic response failure')
        def fail_response(_):
            raise original
        stage = SimpleNamespace(name='lf', edges=np.array([0]),
            device=torch.device('cpu'), tensors=fail_response)
        ledger = {'leg_started_evaluations': {'lf': 0},
            'leg_completed_evaluations': {'lf': 0},
            'leg_raw_saved_evaluations': {'lf': 0},
            'leg_context_pairs_started': {'lf': 0},
            'leg_context_pairs_completed': {'lf': 0},
            'leg_context_pairs_raw_saved': {'lf': 0},
            'leg_conditional_model_s_started': {'lf': 0.},
            'leg_conditional_model_s_completed': {'lf': 0.},
            'leg_conditional_model_s_raw_saved': {'lf': 0.},
            'leg_cache_hits': {'lf': 0}, 'response_call_entered': {'lf': 0}}
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.fit_c_banc_m3_context_trust.np.save',
                   side_effect=OSError('synthetic raw locked')):
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: None, lambda _: None)
            with self.assertRaises(RuntimeError) as caught:
                fit.evaluate_delta(np.array([0.]))
        self.assertIs(caught.exception, original)
        self.assertEqual(ledger['response_call_entered']['lf'], 1)
        self.assertEqual(ledger['leg_conditional_model_s_started']['lf'], 20.)
        self.assertEqual(ledger['leg_context_pairs_started']['lf'], 2)
        self.assertEqual(ledger['leg_completed_evaluations']['lf'], 0)
        self.assertTrue(any('synthetic raw locked' in note
                            for note in original.__notes__))

    def test_completed_two_context_accounting_and_cap_soft_stop(self):
        stage = SimpleNamespace(name='lf', edges=np.array([0]),
            device=torch.device('cpu'), tensors=fake_stage_response)
        ledger = one_leg_ledger(started=1)
        with tempfile.TemporaryDirectory() as temp:
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: None, lambda _: None)
            fit.evaluate_delta(np.array([.01]))
            self.assertEqual(ledger['leg_started_evaluations']['lf'], 2)
            self.assertEqual(ledger['leg_completed_evaluations']['lf'], 2)
            self.assertEqual(ledger['leg_raw_saved_evaluations']['lf'], 2)
            self.assertEqual(ledger['leg_context_pairs_started']['lf'], 4)
            self.assertEqual(ledger['leg_context_pairs_completed']['lf'], 4)
            self.assertEqual(ledger['leg_context_pairs_raw_saved']['lf'], 4)
            self.assertEqual(ledger['leg_conditional_model_s_started']['lf'], 40.)
            self.assertEqual(ledger['leg_conditional_model_s_completed']['lf'], 40.)
            self.assertEqual(ledger['leg_conditional_model_s_raw_saved']['lf'], 40.)
            ledger['leg_started_evaluations']['lf'] = 500
            with self.assertRaises(LegSoftStop) as caught:
                fit.evaluate_delta(np.array([.02]))
            self.assertEqual(caught.exception.reason, 'leg_unique_delta_limit')

    def test_post_response_save_failure_rescues_nested_traces_without_rerun(self):
        calls = []
        def response(parameter):
            calls.append(True)
            return fake_stage_response(parameter)
        stage = SimpleNamespace(name='lf', edges=np.array([0]),
            device=torch.device('cpu'), tensors=response)
        ledger = one_leg_ledger()
        original = OSError('synthetic eval archive locked')
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.fit_c_banc_m3_context_trust.np.savez_compressed',
                   side_effect=original):
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: None, lambda _: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.array([.01]))
            rescued = Path(temp)/'error-raw-lf-0001'
            for key in ('current', 'motor_rate', 'muscle_activation',
                        'command', 'tendon'):
                self.assertTrue((rescued/f'response_{key}.npy').exists(), key)
            self.assertTrue((rescued/'jacobian.npy').exists())
            self.assertTrue((rescued/'feet.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger['leg_context_pairs_started']['lf'], 2)
        self.assertEqual(ledger['leg_context_pairs_completed']['lf'], 2)
        self.assertEqual(ledger['leg_context_pairs_raw_saved']['lf'], 0)
        self.assertEqual(ledger['leg_conditional_model_s_completed']['lf'], 20.)

    def test_post_response_rescue_secondary_keeps_primary_and_other_traces(self):
        stage = SimpleNamespace(name='lf', edges=np.array([0]),
            device=torch.device('cpu'), tensors=fake_stage_response)
        ledger = one_leg_ledger()
        original = OSError('primary eval archive locked')
        actual_save = np.save
        def selective_save(path, value, **kwargs):
            if Path(path).name == 'response_current.npy':
                raise OSError('secondary current raw locked')
            return actual_save(path, value, **kwargs)
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.fit_c_banc_m3_context_trust.np.savez_compressed',
                   side_effect=original), \
             patch('tools.fit_c_banc_m3_context_trust.np.save',
                   side_effect=selective_save):
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: None, lambda _: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.array([.01]))
            rescued = Path(temp)/'error-raw-lf-0001'
            self.assertTrue((rescued/'response_command.npy').exists())
            self.assertTrue((rescued/'response_tendon.npy').exists())
            self.assertFalse((rescued/'response_current.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertTrue(any('secondary current raw locked' in note
                            for note in original.__notes__))

    def test_geometry_failure_rescues_response_created_inside_real_stage_method(self):
        primary = RuntimeError('synthetic geometry failure')
        def response(*arguments):
            value = arguments[2][0, 0, 0]
            return {key: value*torch.ones((2, 2, 7), dtype=torch.float64)
                    for key in ('current', 'motor_rate', 'muscle_activation',
                                'command', 'tendon')}
        stage = object.__new__(LegStage)
        stage.name, stage.leg = 'lf', 0
        stage.edges = np.array([0])
        stage.device = torch.device('cpu')
        stage.base_w = torch.ones((1, 1), dtype=torch.float64)
        stage.rows = torch.tensor([0])
        stage.cols = torch.tensor([0])
        stage.source = stage.background = torch.zeros(1)
        stage.context_indices = torch.tensor([0, 1])
        stage.weight_indices = torch.tensor([0, 0])
        stage.common = torch.tensor([0, 1])
        stage.response = SimpleNamespace(evaluate=response)
        stage.geometry = SimpleNamespace(leg=lambda *_: (_ for _ in ()).throw(primary))
        ledger = one_leg_ledger()
        with tempfile.TemporaryDirectory() as temp:
            fit = LegFit(stage, {'bounds': np.array([[-.1, .1]])},
                Path(temp), ledger, lambda: None, lambda _: None)
            with self.assertRaises(RuntimeError) as caught:
                fit.evaluate_delta(np.array([.01]))
            rescued = Path(temp)/'error-raw-lf-0001'
            self.assertTrue((rescued/'response_current.npy').exists())
            self.assertTrue((rescued/'response_command.npy').exists())
            self.assertTrue((rescued/'response_tendon.npy').exists())
        self.assertIs(caught.exception, primary)
        self.assertEqual(ledger['leg_completed_evaluations']['lf'], 0)

    def test_actual_frozen_upper_edge_roundoff_and_real_excess_rejected(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'verification/banc-natural-gait-m3-20260919/' \
            'parent-e115-data-review-raw.npz'
        with np.load(path, allow_pickle=False) as saved:
            base = saved['initial_efficacy'].copy()
            bounds = np.stack((saved['delta_lower'], saved['delta_upper']), axis=1)
        delta = np.zeros(len(base))
        delta[1439] = bounds[1439, 1]
        result = efficacy_boundary_roundoff(base, delta, bounds)
        self.assertGreater(result['pre'][1439], 10.)
        self.assertEqual(result['post'][1439], 10.)
        self.assertTrue(result['valid'].all())
        self.assertLessEqual(abs(result['correction'][1439]),
                             result['tolerance'][1439])
        outside = delta.copy()
        outside[1439] += 1e-6
        invalid = efficacy_boundary_roundoff(base, outside, bounds)
        self.assertFalse(invalid['valid'][1439])
        beyond = efficacy_boundary_roundoff(np.array([10.01]),
            np.array([0.]), np.array([[-.1, .1]]))
        self.assertFalse(beyond['original_valid'][0])

    def test_soft_leg_stop_preserves_selection_and_continues_remaining_legs(self):
        names = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
        solution = SimpleNamespace(success=True, status=0, message='done',
            nit=1, nfev=1, njev=1, x=np.array([0., .2]), fun=.2)
        class FakeStage:
            def __init__(self, _data, leg, _geometry, device):
                self.name = names[leg]
                self.edges = np.array([leg])
        class FakeFit:
            def __init__(self, stage, *_args):
                self.stage = stage
                ratio = 1. if stage.name == 'lf' else .1
                self.records = [{'true_worst_objective': .2,
                    'ratios': [[ratio]*4]*2, 'delta': [0.]}]
                self.cache = SimpleNamespace(get_delta=lambda _: fake_cached(1))
        def solve(fit, _bounds):
            if fit.stage.name == 'lf':
                raise LegSoftStop('leg_unique_delta_limit')
            return solution
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.fit_c_banc_m3_context_trust.shutil.copy2'), \
             patch('tools.fit_c_banc_m3_context_trust.reviewed_inputs',
                   return_value={'bounds': np.tile([[-.1, .1]], (6, 1))}), \
             patch('tools.fit_c_banc_m3_context_trust.TorchFootGeometry.load',
                   return_value=SimpleNamespace(geometry_hash=
                       '36a47d2f0f0d4e83d5ff8ecf1af95b9ce63269d5158d98642b4f89e0bc230953')), \
             patch('tools.fit_c_banc_m3_context_trust.LegStage', FakeStage), \
             patch('tools.fit_c_banc_m3_context_trust.LegFit', FakeFit), \
             patch('tools.fit_c_banc_m3_context_trust.solve_leg', solve), \
             patch('tools.fit_c_banc_m3_context_trust.torch.cuda.is_available',
                   return_value=True), \
             patch('tools.fit_c_banc_m3_context_trust.torch.cuda.get_device_name',
                   return_value='NVIDIA RTX 4080'), \
             patch.dict('os.environ', {'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}):
            out = Path(temp)/'result'
            result = run_once(Path(temp), out)
            self.assertEqual(result['status'], 'no_joint_feasible_candidate')
            selections = [json.loads((out/f'selection-{name}.json').read_text())
                          for name in names]
            self.assertEqual(selections[0]['termination_reason'],
                             'leg_unique_delta_limit')
            self.assertIsNone(selections[0]['solver'])
            self.assertIsNotNone(selections[0]['best_any'])
            self.assertEqual([row['termination_reason'] for row in selections[1:]],
                             ['scipy_returned']*5)
            self.assertEqual(result['ledger']['leg_solver_calls'],
                             {name: 1 for name in names})

    def test_outer_error_has_fresh_wall_and_original_exception(self):
        clock = [0.]
        original = RuntimeError('synthetic preflight failure')
        def fail_preflight(_):
            clock[0] = 4.
            raise original
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.fit_c_banc_m3_context_trust.shutil.copy2'), \
             patch('tools.fit_c_banc_m3_context_trust.reviewed_inputs',
                   fail_preflight), \
             patch('tools.fit_c_banc_m3_context_trust.time.monotonic',
                   lambda: clock[0]):
            out = Path(temp)/'result'
            try:
                run_once(Path(temp), out)
            except RuntimeError as caught:
                self.assertIs(caught, original)
            else:
                self.fail('Original error was swallowed')
            ledger = json.loads((out/'execution-ledger.json').read_text())
            self.assertEqual(ledger['process_wall_s'], 4.)
            self.assertEqual(ledger['fit_calls'], 0)


if __name__ == '__main__':
    unittest.main()
