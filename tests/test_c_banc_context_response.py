"""Small CPU-only checks for E114; no original 2001-row response is run."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools import verify_c_banc_m3_e114_contexts as runner
from tools.banc_m3_context_response import (
    ContextResponse,
    audit_own_background,
    id_positions,
    validate_time_grid,
)
from tools.verify_c_banc_m3_e114_contexts import (
    check_calibration_weight_map,
    error_stats,
    run_once,
)


def staged(n_time=3):
    return ContextResponse(time_s=np.arange(n_time, dtype=np.float64)*.005,
        tau=np.array([.02]), a=np.array([1.]), threshold=np.array([0.]),
        cap=np.array([10.]), pool=np.array([[1.]]), moment=np.array([[1.]]),
        rest=np.array([.3]), ltm=np.array([0]), device='cpu:0')


def input_batch(n_time=3, *, candidate=None):
    source = torch.zeros((2, n_time, 1), dtype=torch.float64)
    source[0, 1:, 0] = 1.
    source[1, 1:, 0] = 2.
    background = torch.zeros((2, n_time, 1), dtype=torch.float64)
    if candidate is None:
        candidate = torch.tensor([[[1.]], [[2.]]], dtype=torch.float64,
                                 requires_grad=True)
    contexts = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return source, background, candidate, contexts, weights


def tiny_fixture():
    return {'time_s': np.arange(3, dtype=np.float64)*.005,
        'source_rate': np.zeros((2, 3, 1)), 'current': np.zeros((2, 3, 1)),
        'background': np.zeros((2, 3, 1)), 'weights': np.ones((2, 1, 1)),
        'tau': np.array([.02]), 'a': np.ones(1), 'threshold': np.ones(1),
        'cap': np.array([200.]), 'pool': np.ones((1, 1)),
        'moment': np.ones((42, 1)), 'rest': np.zeros(42),
        'ltm': np.array([0]), 'pair_context_indices': np.array([0, 0, 1, 1]),
        'pair_weight_indices': np.array([0, 1, 0, 1]),
        'fixed_mask': np.array([False, True, True])}


def mock_production_response_failure(kind):
    """Exercise actual run_once bookkeeping with synthetic 3-row tensors."""
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        out = Path(temp)/'output'
        original_error = OSError(f'synthetic {kind} receipt failure')
        real_as_tensor = torch.as_tensor
        real_save_npz = np.savez_compressed
        real_save_npy = np.save
        real_save_json = runner.save_json
        entered = []
        fixture = tiny_fixture()

        class FakeResponse:
            def __init__(self, **kwargs):
                pass

            def evaluate(self, *args):
                entered.append(True)
                return {key: torch.full((4, 3, width), .375, dtype=torch.float64)
                        for key, width in (('current', 1), ('forcing', 1),
                            ('motor_rate', 1), ('muscle_activation', 1),
                            ('command', 42), ('tendon', 1))}

        def cpu_tensor(value, *args, **kwargs):
            if kind.startswith('stage:') and value is fixture[kind.split(':')[1]]:
                raise original_error
            kwargs['device'] = 'cpu'
            return real_as_tensor(value, *args, **kwargs)

        def save_npz(path, **values):
            if kind in ('raw', 'raw_rescue') and \
                    Path(path).name == 'response-raw.npz':
                raise original_error
            return real_save_npz(path, **values)

        def save_npy(path, value, **kwargs):
            if kind == 'raw_rescue' and Path(path).name == 'current.npy':
                raise OSError('synthetic secondary rescue failure')
            return real_save_npy(path, value, **kwargs)

        def save_json(path, value):
            if Path(path).name == 'execution-ledger.json' and (
                    (kind == 'start' and value['stage'] == 'response') or
                    (kind == 'materialized' and
                     value['stage'] == 'response_materialized')):
                raise original_error
            return real_save_json(path, value)

        def stop_geometry(*args):
            raise RuntimeError('synthetic geometry stop')

        patches = (
            ('shutil.copy2', lambda *args: None),
            ('reviewed_inputs', lambda *args: (fixture, {})),
            ('ContextResponse', FakeResponse),
            ('TorchFootGeometry.load', lambda *args, **kwargs:
                SimpleNamespace(geometry_hash=runner.GEOMETRY_HASH,
                                full=stop_geometry)),
            ('torch.cuda.is_available', lambda: True),
            ('torch.cuda.device_count', lambda: 1),
            ('torch.cuda.get_device_name', lambda *args: 'synthetic RTX 4080'),
            ('torch.cuda.synchronize', lambda: None),
            ('torch.as_tensor', cpu_tensor),
            ('np.savez_compressed', save_npz),
            ('np.save', save_npy),
            ('save_json', save_json),
        )
        for name, replacement in patches:
            stack.enter_context(patch('tools.verify_c_banc_m3_e114_contexts.'+name,
                                      replacement))
        stack.enter_context(patch.dict(os.environ, {
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'}))
        caught = None
        try:
            run_once(Path(temp), out)
        except Exception as error:  # noqa: BLE001 - verify identity below
            caught = error
        ledger = json.loads((out/'execution-ledger.json').read_text())
        raw_files = sorted((out/'error-raw').glob('*.npy'))
        saved = {path.stem: np.load(path) for path in raw_files}
        return caught, original_error, len(entered), ledger, saved


class ContextResponseTests(unittest.TestCase):
    def test_explicit_pair_axis_independent_zero_state_and_shape(self):
        model = staged()
        source, background, candidate, contexts, weights = input_batch()
        result = model.evaluate(source, background, candidate, contexts, weights)
        self.assertEqual(model.device, torch.device('cpu'))
        self.assertEqual(result['current'].shape, (4, 3, 1))
        self.assertEqual(result['muscle_activation'].shape, (4, 3, 1))
        self.assertEqual(result['command'].shape, (4, 3, 1))
        self.assertEqual(result['tendon'].shape, (4, 3, 1))
        np.testing.assert_array_equal(result['current'][:, 1, 0].detach().numpy(),
                                      [1, 2, 2, 4])
        for key in ('motor_rate', 'muscle_activation', 'tendon'):
            self.assertTrue(torch.all(torch.abs(result[key][:, 0]) < 1e-14))
        self.assertTrue(torch.all(result['command'][:, 0, 0] == .3))
        # Identical current for pairs 01 and 10 must give identical response.
        for key in ('current', 'motor_rate', 'muscle_activation', 'command', 'tendon'):
            torch.testing.assert_close(result[key][1], result[key][2], atol=1e-14,
                                       rtol=0)

    def test_two_sample_closed_form_not_reusing_response_implementation(self):
        model = staged(2)
        source, background, candidate, contexts, weights = input_batch(2)
        result = model.evaluate(source, background, candidate, contexts, weights)
        dt, tau = .005, .02
        beta = np.exp(-dt/tau)
        alpha = 1-beta
        c = 1-alpha/(dt/tau)
        forcing = 10*np.tanh(np.array([1., 2., 2., 4.])/10)
        expected_rate = c*forcing
        expected_activation = (1-np.exp(-dt/.01))*expected_rate
        np.testing.assert_allclose(result['motor_rate'][:, 1, 0].detach().numpy(),
                                   expected_rate, rtol=0, atol=2e-15)
        np.testing.assert_allclose(result['muscle_activation'][:, 1, 0]
                                   .detach().numpy(), expected_activation,
                                   rtol=0, atol=2e-15)
        np.testing.assert_allclose(result['command'][:, 1, 0].detach().numpy(),
                                   .3+.02*expected_activation, rtol=0, atol=2e-15)

    def test_candidate_autograd_matches_finite_difference(self):
        model = staged()
        source, background, candidate, contexts, weights = input_batch()
        loss = model.evaluate(source, background, candidate, contexts, weights)[
            'command'][:, 2].sum()
        gradient = torch.autograd.grad(loss, candidate)[0].detach().numpy()
        base = candidate.detach().numpy().copy()
        for i in range(2):
            plus, minus = base.copy(), base.copy()
            plus[i, 0, 0] += 1e-5
            minus[i, 0, 0] -= 1e-5
            def value(array):
                return float(model.evaluate(source, background,
                    torch.as_tensor(array), contexts, weights)['command'][:, 2].sum())
            expected = (value(plus)-value(minus))/2e-5
            self.assertAlmostEqual(gradient[i, 0, 0], expected, delta=1e-10)
        self.assertTrue(np.all(gradient[:, 0, 0] > 0))

    def test_fixed_background_is_not_recomputed_with_candidate(self):
        source = np.array([[0.], [1.], [2.]], dtype=np.float64)
        own_w = np.array([[2.]], dtype=np.float64)
        candidate_w = np.array([[5.]], dtype=np.float64)
        current = source@own_w.T+np.array([[0.], [3.], [4.]])
        stored = current-source@own_w.T
        audit = audit_own_background(current, source, own_w, stored)
        self.assertTrue(audit['passed'])
        self.assertFalse(audit_own_background(current, source, candidate_w,
                                              stored)['passed'])
        model = staged()
        ctx = torch.as_tensor(source[None])
        residual = torch.as_tensor(stored[None])
        cand = torch.as_tensor(candidate_w[None])
        out = model.evaluate(ctx, residual, cand, torch.tensor([0]),
                             torch.tensor([0]))
        np.testing.assert_array_equal(out['current'][0].detach().numpy(),
                                      source@candidate_w.T+stored)

    def test_time_and_input_rejection(self):
        with self.assertRaisesRegex(ValueError, 'time grid'):
            validate_time_grid(np.array([0., .005, .011]))
        with self.assertRaisesRegex(ValueError, 'time grid'):
            validate_time_grid(np.array([0., np.nan]))
        model = staged()
        source, background, candidate, contexts, weights = input_batch()
        bad_source = source.clone()
        bad_source[0, 1, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            model.evaluate(bad_source, background, candidate, contexts, weights)
        with self.assertRaisesRegex(ValueError, 'fixed background'):
            model.evaluate(source, background.requires_grad_(), candidate,
                           contexts, weights)
        with self.assertRaisesRegex(ValueError, 'pair indices'):
            model.evaluate(source, background.detach(), candidate,
                           torch.tensor([0, 2]), torch.tensor([0, 1]))

    def test_unsorted_edge_id_mapping_and_duplicates(self):
        np.testing.assert_array_equal(id_positions(np.array([9, 3, 7]),
                                                   np.array([7, 9])), [2, 0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            id_positions(np.array([9, 9]), np.array([9]))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            id_positions(np.array([9]), np.array([7]))

    def test_actual_float32_signed_calibration_mapping(self):
        fixture = {'edge_ids': np.array([12, 10]),
                   'edge_rows': np.array([0, 0]),
                   'edge_cols': np.array([1, 0]),
                   'motor_ids': np.array([80]), 'source_ids': np.array([30, 40]),
                   'original_signed_edge_weights': np.array([-.25, .5],
                                                               dtype=np.float32),
                   'weights': np.array([[[-.0, -.5]]], dtype=np.float64)}
        fixture['weights'][0, 0, 0] = .5
        cal = {'edge_ids': np.array([10, 12]),
               'source_values': np.array([.5, -.25], dtype=np.float32),
               'values': np.array([.5, -.5], dtype=np.float32),
               'individual_edge_ids': np.array([12, 10]),
               'individual_edge_motor_ids': np.array([80, 80]),
               'individual_edge_source_ids': np.array([40, 30]),
               'individual_source_basis_ids': np.array([30, 40]),
               'individual_edge_efficacies': np.array([2., 1.])}
        check_calibration_weight_map(cal, fixture, 0)
        wrong = dict(cal, values=np.array([.5, +.5], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'actual float32'):
            check_calibration_weight_map(wrong, fixture, 0)

    def test_nonfinite_error_never_passes_as_zero(self):
        result = error_stats(np.array([np.inf]), np.array([0.]))
        self.assertFalse(result['finite'])
        self.assertIsNone(result['rms'])
        self.assertIsNone(result['max'])

    def test_original_failure_survives_execution_receipt_save_failure(self):
        original = RuntimeError('synthetic preflight failure')
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'new-output'
            def save(path, value):
                if Path(path).name == 'error.json':
                    raise OSError('receipt locked')
            with patch('tools.verify_c_banc_m3_e114_contexts.shutil.copy2'), \
                 patch('tools.verify_c_banc_m3_e114_contexts.reviewed_inputs',
                       side_effect=original), \
                 patch('tools.verify_c_banc_m3_e114_contexts.save_json',
                       side_effect=save), \
                 self.assertRaises(RuntimeError) as caught:
                run_once(Path(temp), destination)
            self.assertIs(caught.exception, original)
            self.assertTrue(any('receipt locked' in note for note in original.__notes__))

    def test_materialized_response_rescued_when_raw_npz_fails(self):
        caught, original, entered, ledger, saved = mock_production_response_failure(
            'raw')
        self.assertIs(caught, original)
        self.assertEqual(entered, 1)
        self.assertEqual(ledger['batch_materialized'], 1)
        self.assertEqual(ledger['batch_raw_saved'], 0)
        self.assertEqual(ledger['pairs_materialized'], 4)
        self.assertEqual(ledger['pairs_raw_saved'], 0)
        for field in ('current', 'motor_rate', 'muscle_activation', 'command',
                      'tendon', 'pair_context_indices', 'pair_weight_indices'):
            self.assertIn(field, saved)
        self.assertEqual(saved['command'].shape, (4, 3, 42))

    def test_materialized_response_rescued_when_completion_receipt_fails(self):
        caught, original, entered, ledger, saved = mock_production_response_failure(
            'materialized')
        self.assertIs(caught, original)
        self.assertEqual(entered, 1)
        self.assertEqual(ledger['batch_materialized'], 1)
        self.assertEqual(ledger['batch_raw_saved'], 0)
        self.assertEqual(saved['current'].shape, (4, 3, 1))

    def test_rescue_secondary_failure_preserves_primary_and_other_fields(self):
        caught, original, entered, ledger, saved = mock_production_response_failure(
            'raw_rescue')
        self.assertIs(caught, original)
        self.assertEqual(entered, 1)
        self.assertNotIn('current', saved)
        self.assertIn('command', saved)
        self.assertIn('tendon', saved)
        self.assertTrue(any(item['at'] == 'error-raw:current'
                            for item in ledger['secondary_errors']))
        self.assertTrue(any('secondary rescue failure' in note
                            for note in original.__notes__))

    def test_failed_start_receipt_prevents_evaluation_and_charges_zero(self):
        caught, original, entered, ledger, saved = mock_production_response_failure(
            'start')
        self.assertIs(caught, original)
        self.assertEqual(entered, 0)
        self.assertEqual(ledger['batch_reservation_requested'], 4)
        self.assertTrue(ledger['batch_start_receipt_failed'])
        self.assertFalse(ledger['response_call_entered'])
        self.assertEqual(ledger['batch_started'], 0)
        self.assertEqual(ledger['pairs_started'], 0)
        self.assertEqual(ledger['conditional_model_s_started'], 0.)
        self.assertFalse(saved)

    def test_input_transfer_failure_prevents_evaluation_and_charges_zero(self):
        for key in ('source_rate', 'background', 'weights',
                    'pair_context_indices', 'pair_weight_indices'):
            with self.subTest(input=key):
                caught, original, entered, ledger, saved = \
                    mock_production_response_failure('stage:'+key)
                self.assertIs(caught, original)
                self.assertEqual(entered, 0)
                self.assertFalse(ledger['response_call_entered'])
                self.assertEqual(ledger['batch_reservation_requested'], 0)
                self.assertEqual(ledger['batch_started'], 0)
                self.assertEqual(ledger['pairs_started'], 0)
                self.assertEqual(ledger['conditional_model_s_started'], 0.)
                self.assertFalse(saved)

    def test_failure_wall_refreshed_and_clock_failure_is_unavailable(self):
        clock = [0.]
        original = RuntimeError('synthetic four-second preflight failure')
        def fail_preflight(*args):
            clock[0] = 4.
            raise original
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.verify_c_banc_m3_e114_contexts.time.monotonic',
                   lambda: clock[0]), \
             patch('tools.verify_c_banc_m3_e114_contexts.shutil.copy2'), \
             patch('tools.verify_c_banc_m3_e114_contexts.reviewed_inputs',
                   fail_preflight):
            out = Path(temp)/'output'
            try:
                run_once(Path(temp), out)
            except RuntimeError as caught:
                self.assertIs(caught, original)
            else:
                self.fail('Original preflight failure was swallowed')
            saved = json.loads((out/'execution-ledger.json').read_text())
        self.assertEqual(saved['internal_wall_s'], 4.)
        self.assertEqual(saved['internal_wall_status'], 'available')

        ticks = [0]
        second = RuntimeError('primary preflight failure')
        def broken_clock():
            ticks[0] += 1
            if ticks[0] <= 2:
                return 0.
            raise OSError('synthetic clock unavailable')
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.verify_c_banc_m3_e114_contexts.time.monotonic',
                   broken_clock), \
             patch('tools.verify_c_banc_m3_e114_contexts.shutil.copy2'), \
             patch('tools.verify_c_banc_m3_e114_contexts.reviewed_inputs',
                   side_effect=second):
            out = Path(temp)/'output'
            try:
                run_once(Path(temp), out)
            except RuntimeError as caught:
                self.assertIs(caught, second)
            else:
                self.fail('Primary preflight failure was swallowed')
            unavailable = json.loads((out/'execution-ledger.json').read_text())
        self.assertIsNone(unavailable['internal_wall_s'])
        self.assertEqual(unavailable['internal_wall_status'], 'unavailable')
        self.assertTrue(any('synthetic clock unavailable' in note
                            for note in second.__notes__))


if __name__ == '__main__':
    unittest.main()
