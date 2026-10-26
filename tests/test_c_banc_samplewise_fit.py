"""Synthetic CPU E119 constraint, cache and failure-path tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools import fit_c_banc_m3_samplewise_startup as mod
from tools.analyze_c_banc_m3_samplewise_startup import derivative_bank
from tools.banc_m3_context_response import ContextResponse
from tools.fit_c_banc_m3_context_trust import choose_records


def toy_terms():
    time = np.arange(100)*.005
    source = np.zeros((2, 100, 2), dtype=np.float64)
    source[0, 1:, 0] = 1.
    source[0, 2::2, 1] = 1.5
    source[1, 1:, 1] = 1.
    source[1, 2::2, 0] = 1.25
    base = torch.tensor([[1.2, -.5], [-.7, .9]], dtype=torch.float64)
    rows = torch.tensor([1, 0, 1])  # Explicitly unsorted post and signed W.
    cols = torch.tensor([1, 0, 0])
    pool = np.array([[1., .5], [.3, .8]])
    moment = np.array([[1., 0.], [0., 1.], [.4, -.3], [0., 0.],
                       [0., 0.], [0., 0.], [0., 0.]])
    stage = ContextResponse(time_s=time, tau=np.array([.01, .025]),
        a=np.array([1.2, .8]), threshold=np.zeros(2),
        cap=np.array([5., 4.]), pool=pool, moment=moment,
        rest=np.zeros(7), ltm=np.array([0]), device='cpu')
    delta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    candidate = base.clone()
    candidate[rows, cols] = base[rows, cols]*torch.exp(delta)
    output = stage.evaluate(torch.from_numpy(source),
        torch.zeros((2, 100, 2), dtype=torch.float64),
        candidate[None], torch.tensor([0, 1]), torch.tensor([0, 0]))
    q = output['command'][:, 1:100]
    z = output['tendon'][:, 1:100, 0]
    losses = torch.stack((q.square().mean(dim=(1, 2)),
                         z.square().mean(dim=1)), dim=1).sum(dim=1)
    loss_grad = torch.autograd.grad(losses, delta,
        grad_outputs=torch.eye(2, dtype=torch.float64),
        is_grads_batched=True, retain_graph=True)[0]
    operands = {'source': torch.from_numpy(source),
        'current': output['current'].detach(), 'weights': base,
        'rows': rows, 'cols': cols, 'tau': stage.tau, 'a': stage.a,
        'threshold': stage.threshold, 'cap': stage.cap}
    bank = derivative_bank(operands)
    v = bank['activation_derivative'][:, 1:100].numpy()
    coefficient = .02*(stage.moment @ stage.pool)[:, rows].numpy()
    muscle = int(stage.ltm[0])
    z_coefficient = (.02*stage.pool[muscle, rows]).numpy()
    activation = output['muscle_activation'][:, 1:100, muscle].detach()
    gain = (1-torch.tanh(activation/10)**2).numpy()
    cached = {'losses': losses.detach().numpy(),
        'loss_gradient': loss_grad.detach().numpy(),
        'parts': np.zeros((2, 4)), 'ratios': np.zeros((2, 4)),
        'q_error': q.detach().numpy(), 'tendon_error': z.detach().numpy(),
        'v': v, 'q_coefficient': coefficient,
        'ltm_coefficient': z_coefficient, 'ltm_gain': gain,
        'forcing_derivative': bank['forcing_derivative'].numpy(),
        'forcing_sign': output['current'][:, :, rows].detach().numpy()}
    return cached, output, delta, q, z


def ledger_one():
    name = 'lf'
    keys = ('leg_started', 'leg_materialized', 'leg_raw_saved',
            'response_call_entered', 'leg_pairs_started',
            'leg_pairs_materialized', 'leg_pairs_raw_saved',
            'leg_conditional_s_started', 'leg_conditional_s_materialized',
            'leg_conditional_s_raw_saved', 'bank_started',
            'bank_call_entered', 'bank_materialized', 'bank_raw_saved',
            'linear_filter_started', 'linear_filter_returned',
            'linear_filter_completed', 'endpoint_filter_started',
            'endpoint_filter_returned', 'endpoint_filter_completed',
            'cache_hits')
    ledger = {key: {name: 0} for key in keys}
    ledger['warm_trace_saved'] = {name: False}
    return ledger


class FakeLegStage:
    def __init__(self):
        self.name, self.leg = 'lf', 0
        self.device = torch.device('cpu')
        self.motor = np.array([0])
        self.edges = np.array([0])
        self.rows = torch.tensor([0])
        self.cols = torch.tensor([0])
        self.base_w = torch.ones((1, 1), dtype=torch.float64)
        self.source = torch.ones((2, 100, 1), dtype=torch.float64)
        self.source[:, 0] = 0
        self.background = torch.zeros((2, 100, 1), dtype=torch.float64)
        self.response = ContextResponse(time_s=np.arange(100)*.005,
            tau=np.array([.02]), a=np.array([1.]),
            threshold=np.array([0.]), cap=np.array([5.]),
            pool=np.array([[1.]]), moment=np.ones((7, 1)),
            rest=np.zeros(7), ltm=np.array([0]), device='cpu')
        self.start_q = torch.zeros((2, 99, 7), dtype=torch.float64)
        self.start_tendon = torch.zeros((2, 99), dtype=torch.float64)
        self.last_response = None
        self.calls = 0

    def tensors(self, parameter):
        self.calls += 1
        candidate = mod.candidate_weights(self.base_w, self.rows,
                                          self.cols, parameter)
        response = self.response.evaluate(self.source, self.background,
            candidate[None], torch.tensor([0, 1]), torch.tensor([0, 0]))
        self.last_response = response
        q = response['command'][:, 1:100]
        parts = torch.stack((q.square().mean(dim=(1, 2)),
            q.abs().mean(dim=(1, 2)), q.square().mean(dim=(1, 2)),
            q.abs().mean(dim=(1, 2))), dim=1)
        return {'response': response, 'parts': parts,
            'losses': parts.sum(dim=1),
            'ratios': torch.zeros((2, 4), dtype=torch.float64)}


def fake_data():
    return {'bounds': np.array([[-.1, .1]]),
        'fixture': {'weights': np.ones((2, 1, 1)),
            'edge_ids': np.array([17]), 'time_s': np.arange(100)*.005,
            'context_names': np.array(['E110', 'E113'])}}


class SamplewiseFitTests(unittest.TestCase):
    def test_3174_signed_constraints_match_direct_context_autograd(self):
        cached, _, delta, q, z = toy_terms()
        t = 2.
        row = mod.samplewise_terms(np.zeros(3), t, cached)
        self.assertEqual(row['inequalities'].shape, (3174,))
        self.assertEqual(row['inequality_jac'].shape, (3174, 4))
        self.assertAlmostEqual(row['inequalities'][0], t-cached['losses'][0])
        root = np.sqrt(mod.OPTIMIZER_CAP)
        self.assertAlmostEqual(row['inequalities'][2], root-q[0, 0, 0].item()/.04)
        self.assertAlmostEqual(row['inequalities'][2+1386],
                               root+q[0, 0, 0].item()/.04)
        self.assertAlmostEqual(row['inequalities'][2+2772],
                               root-z[0, 0].item()/.025)
        for index, scalar in (
            (2, root-q[0, 0, 0]/.04),
            (2+1386, root+q[0, 0, 0]/.04),
            (2+693, root-q[1, 0, 0]/.04),
            (2+2772, root-z[0, 0]/.025),
            (2+2772+198, root+z[0, 0]/.025),
            (2+2772+99, root-z[1, 0]/.025),
            (3170, mod.OPTIMIZER_CAP-q[0].square().mean()/.018**2),
            (3171, mod.OPTIMIZER_CAP-q[1].square().mean()/.018**2),
            (3172, mod.OPTIMIZER_CAP-z[0].square().mean()/.009**2),
            (3173, mod.OPTIMIZER_CAP-z[1].square().mean()/.009**2)):
            gradient = torch.autograd.grad(scalar, delta,
                retain_graph=True)[0].numpy()
            np.testing.assert_allclose(row['inequality_jac'][index, :3],
                                       gradient, rtol=0, atol=1e-9)
        zero_relu = cached['forcing_sign'] <= 0.
        self.assertTrue(np.any(zero_relu))
        np.testing.assert_array_equal(cached['forcing_derivative'][zero_relu],
                                      0.)

    def test_signed_feasibility_matches_squared_max_and_switch(self):
        q = np.zeros((2, 99, 7))
        z = np.zeros((2, 99))
        q[0, 0, 0] = .039
        q[0, 1, 0] = -.041  # Changed peak must be constrained.
        z[1, 0] = .024
        cached = {'losses': np.array([.1, .2]),
            'loss_gradient': np.zeros((2, 1)),
            'q_error': q, 'tendon_error': z,
            'v': np.ones((2, 99, 1)),
            'q_coefficient': np.zeros((7, 1)),
            'ltm_coefficient': np.zeros(1),
            'ltm_gain': np.ones((2, 99))}
        row = mod.samplewise_terms(np.zeros(1), .3, cached)
        signed_max = row['inequalities'][2:3170]
        self.assertFalse(np.all(signed_max >= 0))
        self.assertLess(row['inequalities'][2+1386+7], 0.)
        squared_q = np.max(q*q)/.04**2
        self.assertEqual(bool(np.all(signed_max >= 0)),
                         bool(squared_q <= mod.OPTIMIZER_CAP))
        cached['q_error'][0, 1, 0] = -.02
        row2 = mod.samplewise_terms(np.zeros(1), .3, cached)
        self.assertTrue(np.all(row2['inequalities'][2:3170] >= 0))

    def test_t_only_and_backtracking_cache_never_repeat_evaluator(self):
        calls, hits = [], []
        cached, *_ = toy_terms()
        cache = mod.DeltaCache(lambda delta: (calls.append(delta.copy()),
                               cached)[1], lambda: hits.append(True))
        a = np.zeros(3)
        b = np.array([.01, 0., 0.])
        first = cache.get(np.r_[a, 1.])
        cache.get(np.r_[a, 2.])
        cache.get(np.r_[b, 2.])
        cache.get(np.r_[a, 1.])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(first['objective'], 1.)
        self.assertEqual(len(cache.values), 2)

    def test_selector_uses_strict_saved_ratios_not_optimizer_start(self):
        near = (mod.SELECTOR_CAP+mod.OPTIMIZER_CAP)/2
        cached = {'parts': np.zeros((2, 4)),
            'losses': np.array([.2, .3]),
            'ratios': np.full((2, 4), near)}
        record = mod.selector_record(1, 'lf', np.zeros(2), cached)
        self.assertTrue(record['selector_feasible'])
        self.assertFalse(record['optimizer_startup_feasible'])
        best_any, feasible = choose_records([record])
        self.assertIs(best_any, record)
        self.assertIs(feasible, record)
        cached['ratios'] = np.full((2, 4), mod.SELECTOR_CAP+1e-5)
        reject = mod.selector_record(2, 'lf', np.zeros(2), cached)
        _, feasible = choose_records([reject])
        self.assertIsNone(feasible)
        no_joint = mod.no_joint_result([record, None, None, None, None, None], {})
        self.assertEqual(no_joint['status'], 'no_joint_feasible_candidate')
        self.assertEqual(no_joint['feasible_legs'], ['lf'])

    def test_small_real_slsqp_single_start_and_limits(self):
        def evaluator(delta):
            d = float(delta[0])
            return {'losses': np.array([d*d, (d-.1)**2]),
                'loss_gradient': np.array([[2*d], [2*(d-.1)]]),
                'q_error': np.zeros((2, 99, 7)),
                'tendon_error': np.zeros((2, 99)),
                'v': np.zeros((2, 99, 1)),
                'q_coefficient': np.zeros((7, 1)),
                'ltm_coefficient': np.zeros(1), 'ltm_gain': np.ones((2, 99))}
        fit = SimpleNamespace(cache=mod.DeltaCache(evaluator))
        solution = mod.solve_leg(fit, np.array([[-.2, .2]]), np.zeros(1))
        self.assertTrue(solution.success, solution.message)
        self.assertLessEqual(solution.nit, 100)
        self.assertAlmostEqual(solution.x[0], .05, delta=2e-3)
        self.assertEqual(solution.x.shape, (2,))

    def test_start_receipt_failure_has_no_forward_or_derivative(self):
        stage, ledger = FakeLegStage(), ledger_one()
        original = OSError('synthetic response receipt locked')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                lambda: (_ for _ in ()).throw(original),
                lambda *_args, **_kwargs: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.zeros(1))
            for key in ('delta', 'edge_ids', 'local_candidate_weights',
                        'time_s', 'context_names'):
                self.assertTrue((out/f'error-input-lf-001/{key}.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertEqual(stage.calls, 0)
        self.assertEqual(ledger['leg_started']['lf'], 0)
        self.assertEqual(ledger['bank_started']['lf'], 0)

    def test_post_forward_bank_failure_rescues_trace_and_primary(self):
        stage, ledger = FakeLegStage(), ledger_one()
        original = RuntimeError('synthetic derivative bank failed')
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod, 'derivative_bank', side_effect=original):
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                             lambda: None, lambda *_args, **_kwargs: None)
            with self.assertRaises(RuntimeError) as caught:
                fit.evaluate_delta(np.zeros(1))
            for key in ('delta', 'edge_ids', 'local_candidate_weights',
                        'parts', 'losses', 'ratios', 'loss_gradient',
                        'q_error', 'tendon_error', 'response_current',
                        'response_muscle_activation'):
                self.assertTrue((out/f'error-raw-lf-001/{key}.npy').exists(), key)
        self.assertIs(caught.exception, original)
        self.assertEqual(stage.calls, 1)
        self.assertEqual(ledger['response_call_entered']['lf'], 1)
        self.assertEqual(ledger['leg_materialized']['lf'], 1)
        self.assertEqual(ledger['bank_call_entered']['lf'], 1)
        self.assertEqual(ledger['bank_raw_saved']['lf'], 0)

    def test_post_materialization_raw_failure_keeps_factors_and_metadata(self):
        stage, ledger = FakeLegStage(), ledger_one()
        original = OSError('synthetic eval raw write failed')
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod.np, 'savez', side_effect=original):
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                             lambda: None, lambda *_args, **_kwargs: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.zeros(1))
            raw = out/'error-raw-lf-001'
            for key in ('delta', 'bounds', 'edge_positions', 'edge_ids',
                        'local_candidate_weights', 'context_names', 'time_s',
                        'parts', 'losses', 'ratios', 'loss_gradient',
                        'q_error', 'tendon_error', 'v', 'q_coefficient',
                        'ltm_coefficient', 'ltm_gain', 'response_current',
                        'response_command', 'response_tendon'):
                self.assertTrue((raw/f'{key}.npy').exists(), key)
        self.assertIs(caught.exception, original)
        self.assertEqual(ledger['response_call_entered']['lf'], 1)
        self.assertEqual(ledger['bank_call_entered']['lf'], 1)
        self.assertEqual(ledger['bank_materialized']['lf'], 1)
        self.assertEqual(ledger['leg_raw_saved']['lf'], 0)

    def test_post_forward_gradient_failure_rescues_available_tensors(self):
        stage, ledger = FakeLegStage(), ledger_one()
        original = RuntimeError('synthetic autograd failure')
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod.torch.autograd, 'grad', side_effect=original):
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                             lambda: None, lambda *_args, **_kwargs: None)
            with self.assertRaises(RuntimeError) as caught:
                fit.evaluate_delta(np.zeros(1))
            raw = out/'error-raw-lf-001'
            for key in ('parts', 'losses', 'ratios', 'response_current',
                        'response_command', 'response_tendon'):
                self.assertTrue((raw/f'{key}.npy').exists(), key)
        self.assertIs(caught.exception, original)
        self.assertEqual(stage.calls, 1)
        self.assertEqual(ledger['leg_materialized']['lf'], 1)
        self.assertEqual(ledger['bank_started']['lf'], 0)

    def test_gradient_transfer_failure_rescues_computed_gradient_and_clock(self):
        stage, ledger = FakeLegStage(), ledger_one()
        original = OSError('synthetic one-off gradient transfer failure')
        original_grad = torch.autograd.grad
        original_cpu = torch.Tensor.cpu
        gradients, failed = [], []

        def capture_grad(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            gradients.append(result[0])
            return result

        def fail_once(tensor, *args, **kwargs):
            if gradients and not failed and \
                    tensor.data_ptr() == gradients[0].data_ptr():
                failed.append(True)
                raise original
            return original_cpu(tensor, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(torch.autograd, 'grad', side_effect=capture_grad), \
             patch.object(torch.Tensor, 'cpu', new=fail_once):
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                             lambda: None, lambda *_args, **_kwargs: None)
            with self.assertRaises(OSError) as caught:
                fit.evaluate_delta(np.zeros(1))
            recovered = np.load(out/'error-raw-lf-001/loss_gradient.npy')
            np.testing.assert_array_equal(recovered,
                                          gradients[0].detach().numpy())
            self.assertTrue((out/'error-raw-lf-001/response_current.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertEqual(len(gradients), 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(stage.calls, 1)
        self.assertEqual(ledger['leg_materialized']['lf'], 1)
        self.assertEqual(ledger['leg_pairs_materialized']['lf'], 2)
        self.assertEqual(ledger['leg_conditional_s_materialized']['lf'], 20.)
        self.assertEqual(ledger['bank_started']['lf'], 0)

    def test_response_created_before_geometry_failure_is_charged(self):
        original = RuntimeError('synthetic post-response geometry failure')

        class FailingGeometryStage(FakeLegStage):
            def tensors(self, parameter):
                super().tensors(parameter)
                raise original

        stage, ledger = FailingGeometryStage(), ledger_one()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            fit = mod.LegFit(stage, fake_data(), out, ledger,
                             lambda: None, lambda *_args, **_kwargs: None)
            with self.assertRaises(RuntimeError) as caught:
                fit.evaluate_delta(np.zeros(1))
            for key in ('response_current', 'response_forcing',
                        'response_motor_rate', 'response_muscle_activation',
                        'response_command', 'response_tendon'):
                self.assertTrue((out/f'error-raw-lf-001/{key}.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertEqual(stage.calls, 1)
        self.assertEqual(ledger['leg_materialized']['lf'], 1)
        self.assertEqual(ledger['leg_pairs_materialized']['lf'], 2)
        self.assertEqual(ledger['leg_conditional_s_materialized']['lf'], 20.)
        self.assertEqual(ledger['bank_started']['lf'], 0)

    def test_final_feet_tensor_rescue_uses_same_geometry_result(self):
        tensor = torch.arange(2*3*6*3, dtype=torch.float64).reshape(2*3, 6, 3)
        original_cpu = torch.Tensor.cpu
        primary = OSError('synthetic final feet transfer failure')
        failed, raw, secondary_errors = [], {'fixed_mask': np.ones(3)}, []

        def fail_once(value, *args, **kwargs):
            if value.data_ptr() == tensor.data_ptr() and not failed:
                failed.append(True)
                raise primary
            return original_cpu(value, *args, **kwargs)

        with patch.object(torch.Tensor, 'cpu', new=fail_once):
            with self.assertRaises(OSError) as caught:
                mod.materialize_feet(tensor, 3)
            mod.rescue_available_feet(tensor, 3, raw, caught.exception,
                                      secondary_errors)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(failed), 1)
        np.testing.assert_array_equal(raw['feet_fixed'],
                                      tensor.numpy().reshape(2, 3, 6, 3))
        self.assertEqual(secondary_errors, [])


if __name__ == '__main__':
    unittest.main()
