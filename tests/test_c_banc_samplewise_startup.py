"""Synthetic CPU E118 derivative and receipt tests; no original bank."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools import analyze_c_banc_m3_samplewise_startup as mod
from tools.banc_m3_context_response import ContextResponse


def synthetic_response():
    """Two independent contexts, unsorted excitatory/inhibitory edge order."""
    time_s = np.arange(100, dtype=np.float64)*.005
    source = np.zeros((2, 100, 2), dtype=np.float64)
    source[0, 1:, 0] = 1.
    source[0, 2::2, 1] = 1.5
    source[0, 3::3, 0] = .25
    source[1, 1:, 1] = 1.
    source[1, 2::2, 0] = 1.25
    source[1, 4::4, 1] = .2
    base = torch.tensor([[1.2, -.5], [-.7, .9]], dtype=torch.float64)
    rows = np.array([1, 0, 1], dtype=np.int64)
    cols = np.array([1, 0, 0], dtype=np.int64)
    pool = np.zeros((6, 2), dtype=np.float64)
    pool[0] = [1., .5]
    pool[1] = [.4, 1.]
    moment = np.zeros((42, 6), dtype=np.float64)
    moment[:7, 0] = np.arange(1., 8.)/7
    moment[:7, 1] = .25
    stage = ContextResponse(time_s=time_s, tau=np.array([.01, .025]),
        a=np.array([1.2, .8]), threshold=np.zeros(2),
        cap=np.array([5., 4.]), pool=pool, moment=moment,
        rest=np.zeros(42), ltm=np.arange(6), device='cpu')
    delta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    candidate = base.clone()
    candidate[rows, cols] = base[rows, cols]*torch.exp(delta)
    output = stage.evaluate(torch.from_numpy(source),
        torch.zeros((2, 100, 2), dtype=torch.float64),
        candidate.unsqueeze(0), torch.tensor([0, 1]), torch.tensor([0, 0]))
    staged = {'source': torch.from_numpy(source),
        'current': output['current'].detach(), 'weights': base,
        'rows': torch.from_numpy(rows), 'cols': torch.from_numpy(cols),
        'tau': stage.tau, 'a': stage.a, 'threshold': stage.threshold,
        'cap': stage.cap}
    return staged, output, delta, rows, cols, pool, moment, time_s


def small_ledger():
    return {'stage': 'staged', 'bank_started': 0, 'bank_materialized': 0,
        'bank_raw_saved': 0, 'derivative_bank_reservation_requested': 0,
        'bank_call_entered': False, 'linear_filter_started': 0,
        'linear_filter_returned': 0, 'linear_filter_completed': 0,
        'endpoint_filter_started': 0, 'endpoint_filter_returned': 0,
        'endpoint_filter_completed': 0}


class SamplewiseStartupTests(unittest.TestCase):
    def test_cpu_autograd_signed_unsorted_relu_zero_and_filter_call_counts(self):
        staged, response, delta, rows, cols, pool, moment, _ = \
            synthetic_response()
        counts = {'linear': 0, 'endpoint': 0}
        original_linear = mod.linear_response
        original_endpoint = mod.endpoint_response
        def count_linear(*args, **kwargs):
            counts['linear'] += 1
            return original_linear(*args, **kwargs)
        def count_endpoint(*args, **kwargs):
            counts['endpoint'] += 1
            return original_endpoint(*args, **kwargs)
        with patch.object(mod, 'linear_response', side_effect=count_linear), \
             patch.object(mod, 'endpoint_response', side_effect=count_endpoint):
            bank = mod.derivative_bank(staged)
        self.assertEqual(counts, {'linear': 1, 'endpoint': 1})
        self.assertEqual(bank['activation_derivative'].shape, (2, 100, 3))
        self.assertEqual(staged['weights'][rows[2], cols[2]].item(), -.7)
        np.testing.assert_array_equal(bank['forcing_derivative'][:, 0].numpy(),
                                      np.zeros((2, 3)))
        self.assertTrue((bank['forcing_derivative'][0, 1:10] != 0).any())
        self.assertTrue((bank['forcing_derivative'][0, 2:10] == 0).any())
        for context in range(2):
            for tick in (0, 1, 2, 17, 99):
                for edge in range(3):
                    current_grad = torch.autograd.grad(
                        response['current'][context, tick, rows[edge]],
                        delta, retain_graph=True)[0][edge]
                    forcing_grad = torch.autograd.grad(
                        response['forcing'][context, tick, rows[edge]],
                        delta, retain_graph=True)[0][edge]
                    motor_grad = torch.autograd.grad(
                        response['motor_rate'][context, tick, rows[edge]],
                        delta, retain_graph=True)[0][edge]
                    self.assertAlmostEqual(
                        bank['current_derivative'][context, tick, edge].item(),
                        current_grad.item(), delta=1e-12)
                    self.assertAlmostEqual(
                        bank['forcing_derivative'][context, tick, edge].item(),
                        forcing_grad.item(), delta=1e-12)
                    self.assertAlmostEqual(
                        bank['motor_derivative'][context, tick, edge].item(),
                        motor_grad.item(), delta=1e-11)
        f = {'edge_rows': rows, 'edge_cols': cols,
            'edge_ids': np.array([8, 3, 5]), 'moment': moment,
            'pool': pool, 'ltm': np.arange(6),
            'startup_target_command': np.zeros((99, 42)),
            'startup_target_tendon': np.zeros((99, 6))}
        saved = {key: value.detach().numpy().copy()
                 for key, value in response.items()}
        mask = np.zeros((6, 3), dtype=bool)
        mask[0] = True
        data = {'fixture': f, 'response': saved,
            'step': {'leg_edge_mask': mask},
            'baseline': {'lf': {'jacobian': np.zeros((10, 3))}}}
        bank_np = {key: tensor.detach().numpy().copy()
                   for key, tensor in bank.items()}
        jacobian = mod.leg_jacobian(bank_np, data, 0)
        self.assertEqual(jacobian['q_jacobian'].shape, (2, 99, 7, 3))
        self.assertEqual(jacobian['tendon_jacobian'].shape, (2, 99, 3))
        np.testing.assert_array_equal(jacobian['edge_ids'], [8, 3, 5])
        for context in range(2):
            for tick in (1, 2, 17, 99):
                for joint in (0, 4):
                    gradient = torch.autograd.grad(
                        response['command'][context, tick, joint], delta,
                        retain_graph=True)[0].numpy()
                    np.testing.assert_allclose(jacobian['q_jacobian'][
                        context, tick-1, joint], gradient, rtol=0, atol=1e-11)
                gradient = torch.autograd.grad(response['tendon'][
                    context, tick, 0], delta, retain_graph=True)[0].numpy()
                np.testing.assert_allclose(jacobian['tendon_jacobian'][
                    context, tick-1], gradient, rtol=0, atol=1e-11)

    def test_future_sample_and_other_context_do_not_change_prefix(self):
        staged, _, _, _, _, _, _, _ = synthetic_response()
        first = mod.derivative_bank(staged)
        altered = dict(staged)
        altered['source'] = staged['source'].clone()
        altered['current'] = staged['current'].clone()
        altered['source'][1, 99, 0] += 9.
        altered['current'][1, 99, 0] += 10.
        second = mod.derivative_bank(altered)
        for key in mod.BANK_FIELDS:
            np.testing.assert_allclose(first[key][0].numpy(),
                                       second[key][0].numpy(), atol=1e-12)
            np.testing.assert_allclose(first[key][1, :99].numpy(),
                                       second[key][1, :99].numpy(), atol=1e-12)
        self.assertFalse(torch.allclose(first['current_derivative'][1, 99],
                                        second['current_derivative'][1, 99]))

    def test_exact_amax_ties_are_averaged_not_near_ties(self):
        q = np.zeros((2, 3, 7))
        tendon = np.zeros((2, 3))
        jq = np.zeros((2, 3, 7, 2))
        jt = np.zeros((2, 3, 2))
        q[0, 0, 0] = 2.
        q[0, 1, 1] = -2.
        jq[0, 0, 0, 0] = 1.
        jq[0, 1, 1, 1] = 2.
        tendon[0, 0] = 1.
        tendon[0, 1] = -1.
        jt[0, 0, 0] = 3.
        jt[0, 1, 1] = 4.
        q[0, 2, 2] = 2.-1e-12
        gradient, exact, explained = mod.startup_gradient(
            q, tendon, jq, jt, inner=np.ones(4))
        np.testing.assert_allclose(gradient[1], [2., -4.])
        np.testing.assert_allclose(gradient[3], [3., -4.])
        self.assertEqual(exact[0].tolist(), [2, 2])
        self.assertEqual(explained[0].tolist(), [3, 2])
        self.assertEqual(gradient.shape, (8, 2))

    def test_adjacent_float_peaks_match_original_torch_amax_before_cap(self):
        q = np.zeros((2, 99, 7))
        tendon = np.zeros((2, 99))
        jq = np.zeros((2, 99, 7, 1))
        jt = np.zeros((2, 99, 1))
        q_lower = np.float64(.010076007600760077)
        t_lower = np.float64(.012565256525652565)
        q[0, 0, 0], q[0, 1, 0] = q_lower, np.nextafter(q_lower, np.inf)
        tendon[0, 0], tendon[0, 1] = t_lower, np.nextafter(t_lower, np.inf)
        jq[0, 0, 0, 0] = jt[0, 0, 0] = 1.
        jq[0, 1, 0, 0] = jt[0, 1, 0] = 3.
        self.assertLess(q[0, 0, 0]**2, q[0, 1, 0]**2)
        self.assertLess(tendon[0, 0]**2, tendon[0, 1]**2)
        self.assertEqual((q[0, 0, 0]/.04)**2, (q[0, 1, 0]/.04)**2)
        self.assertEqual((tendon[0, 0]/.025)**2,
                         (tendon[0, 1]/.025)**2)
        gradient, exact, explanatory = mod.startup_gradient(
            q, tendon, jq, jt)
        delta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        q_torch = torch.from_numpy(q)+torch.from_numpy(jq)@delta
        t_torch = torch.from_numpy(tendon)+torch.from_numpy(jt)@delta
        expected_q = torch.autograd.grad(
            q_torch[0].square().amax()/.04**2, delta,
            retain_graph=True)[0].item()
        expected_t = torch.autograd.grad(
            t_torch[0].square().amax()/.025**2, delta)[0].item()
        self.assertAlmostEqual(gradient[1, 0], expected_q, delta=1e-10)
        self.assertAlmostEqual(gradient[3, 0], expected_t, delta=1e-10)
        np.testing.assert_array_equal(exact[0], [1, 1])
        np.testing.assert_array_equal(explanatory[0], [2, 2])

    def test_jacobian_gradient_error_retains_available_numeric_axes(self):
        f = {'edge_rows': np.array([0]), 'edge_cols': np.array([0]),
            'edge_ids': np.array([17]), 'moment': np.zeros((42, 6)),
            'pool': np.zeros((6, 1)), 'ltm': np.arange(6),
            'startup_target_command': np.zeros((99, 42)),
            'startup_target_tendon': np.zeros((99, 6))}
        f['pool'][0, 0] = 1.
        f['moment'][0, 0] = 1.
        mask = np.zeros((6, 1), dtype=bool)
        mask[0, 0] = True
        data = {'fixture': f, 'step': {'leg_edge_mask': mask},
            'response': {'muscle_activation': np.zeros((2, 100, 6)),
                'command': np.zeros((2, 100, 42)),
                'tendon': np.zeros((2, 100, 6))},
            'baseline': {'lf': {'jacobian': np.zeros((10, 1))}}}
        bank = {'activation_derivative': np.ones((2, 100, 1))}
        available = {}
        original = RuntimeError('synthetic gradient reconstruction failure')
        with patch.object(mod, 'startup_gradient', side_effect=original), \
             self.assertRaises(RuntimeError) as caught:
            mod.leg_jacobian(bank, data, 0, record=available.update)
        self.assertIs(caught.exception, original)
        for key in ('edge_ids', 'edge_positions', 'q_jacobian',
                    'tendon_jacobian', 'q_error', 'tendon_error'):
            self.assertIn(key, available)
        self.assertEqual(available['q_jacobian'].shape, (2, 99, 7, 1))

    def test_saved_h_signed_prediction_and_ratios_axes(self):
        jacobians = {}
        for leg, name in enumerate(mod.LEGS):
            jq = np.zeros((2, 99, 7, 1))
            jt = np.zeros((2, 99, 1))
            jq[:, :, 0, 0] = leg+1
            jt[:, :, 0] = -(leg+1)
            jacobians[name] = {'edge_positions': np.array([leg]),
                'q_error': np.zeros((2, 99, 7)),
                'tendon_error': np.zeros((2, 99)),
                'q_jacobian': jq, 'tendon_jacobian': jt}
        h = np.array([[0.]*6, [.01]*6, [-.01]*6])
        data = {'step': {'increments': h},
            'fixture': {'context_names': np.array(['E110', 'E113']),
                        'time_s': np.arange(100)*.005}}
        raw = mod.signed_predictions(jacobians, data)
        self.assertEqual(raw['q_error_predicted'].shape, (3, 2, 99, 42))
        self.assertEqual(raw['tendon_error_predicted'].shape, (3, 2, 99, 6))
        self.assertEqual(raw['startup_ratios'].shape, (3, 6, 2, 4))
        self.assertEqual(raw['q_error_predicted'][0].max(), 0.)
        self.assertAlmostEqual(raw['q_error_predicted'][1, 0, 0, 7], .02)
        self.assertAlmostEqual(raw['tendon_error_predicted'][1, 0, 0, 1], -.02)
        self.assertTrue(np.isfinite(raw['startup_ratios']).all())

    def test_start_receipt_failure_rolls_back_without_filter(self):
        calls = []
        original = OSError('synthetic bank start receipt locked')
        metadata = {'edge_ids': np.array([9, 2]),
                    'time_s': np.arange(4)*.005}
        def bank_fn(*_args, **_kwargs):
            calls.append(True)
            raise AssertionError('should not enter bank')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                mod.execute_bank({}, metadata, out, small_ledger(),
                    lambda: (_ for _ in ()).throw(original), lambda: None,
                    bank_fn=bank_fn)
            self.assertTrue((out/'error-bank-input/edge_ids.npy').exists())
            self.assertTrue((out/'error-bank-input/time_s.npy').exists())
        self.assertIs(caught.exception, original)
        self.assertFalse(calls)

    def test_returned_receipt_failure_rescues_all_fields_and_primary(self):
        calls, receipts = [], []
        original = OSError('synthetic materialized receipt locked')
        metadata = {'edge_ids': np.array([9, 2]),
                    'selected_weights': np.array([1., -2.]),
                    'context_names': np.array(['E110', 'E113']),
                    'time_s': np.arange(4)*.005}
        def bank_fn(_staged, *, record, on_filter):
            calls.append(True)
            values = {}
            for name in mod.BANK_FIELDS:
                values[name] = torch.ones((2, 4, 3), dtype=torch.float64)
                record(name, values[name])
            for name in ('linear', 'endpoint'):
                on_filter(name, 'started')
                on_filter(name, 'returned')
            return values
        def persist():
            receipts.append(True)
            if len(receipts) == 2:
                raise original
        state = small_ledger()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod, 'PREFIX_ROWS', 4), patch.object(mod, 'EDGE_COUNT', 3):
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                mod.execute_bank({}, metadata, out, state, persist,
                                 lambda: None, bank_fn=bank_fn)
            for key in (*mod.BANK_FIELDS, *metadata):
                self.assertTrue((out/f'error-bank-raw/{key}.npy').exists(), key)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual([state[k] for k in
            ('bank_started', 'bank_materialized', 'bank_raw_saved')], [1, 1, 0])
        self.assertEqual([state[k] for k in ('linear_filter_started',
            'linear_filter_returned', 'linear_filter_completed',
            'endpoint_filter_started', 'endpoint_filter_returned',
            'endpoint_filter_completed')], [1, 1, 1, 1, 1, 1])

    def test_malformed_return_is_saved_before_shape_assertion(self):
        def bank_fn(_staged, *, record, on_filter):
            returned = {key: torch.ones((2, 4, 2), dtype=torch.float64)
                        for key in mod.BANK_FIELDS}
            for key, value in returned.items():
                record(key, value)
            for name in ('linear', 'endpoint'):
                on_filter(name, 'started')
                on_filter(name, 'returned')
            return returned
        state = small_ledger()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod, 'PREFIX_ROWS', 4), patch.object(mod, 'EDGE_COUNT', 3):
            out = Path(temp)
            with self.assertRaisesRegex(ValueError, 'axes changed'):
                mod.execute_bank({}, {'edge_ids': np.array([1, 2])}, out,
                    state, lambda: None, lambda: None, bank_fn=bank_fn)
            with np.load(out/'derivative-bank-raw.npz', allow_pickle=False) as z:
                self.assertEqual(z['current_derivative'].shape, (2, 4, 2))
                np.testing.assert_array_equal(z['edge_ids'], [1, 2])
        self.assertEqual(state['bank_raw_saved'], 1)

    def test_staging_failure_precedes_bank_accounting(self):
        f = {'source_rate': np.zeros((2, 100, 1)),
             'edge_rows': np.array([0]), 'edge_cols': np.array([0]),
             'tau': np.array([.01]), 'a': np.array([1.]),
             'threshold': np.array([0.]), 'cap': np.array([1.])}
        d = {'fixture': f, 'response': {'current': np.zeros((2, 100, 1))},
             'step': {'weights': np.ones((3, 1, 1))}}
        original = OSError('synthetic last tensor transfer')
        actual = torch.as_tensor
        def transfer(value, *args, **kwargs):
            if value is f['cap']:
                raise original
            return actual(value, *args, **kwargs)
        state = small_ledger()
        with patch.object(mod.torch, 'as_tensor', side_effect=transfer), \
             self.assertRaises(OSError) as caught:
            mod.stage_bank_inputs(d, 'cpu')
        self.assertIs(caught.exception, original)
        self.assertEqual(state['bank_started'], 0)
        self.assertEqual(state['linear_filter_started'], 0)

    def test_outer_preflight_error_has_fresh_wall_and_zero_bank(self):
        clock = [0.]
        original = RuntimeError('synthetic preflight failure')
        def fail_preflight(_):
            clock[0] = 4.
            raise original
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(mod.shutil, 'copy2'), \
             patch.object(mod, 'reviewed_inputs', side_effect=fail_preflight), \
             patch.object(mod.time, 'monotonic', side_effect=lambda: clock[0]):
            out = Path(temp)/'out'
            with self.assertRaises(RuntimeError) as caught:
                mod.run_once(Path(temp), out)
            saved = json.loads((out/'execution-ledger.json').read_text())
        self.assertIs(caught.exception, original)
        self.assertEqual(saved['process_wall_s'], 4.)
        self.assertEqual(saved['bank_started'], 0)
        self.assertEqual(saved['linear_filter_started'], 0)


if __name__ == '__main__':
    unittest.main()
