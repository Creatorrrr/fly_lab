"""Synthetic E117 response, metric, peak and receipt tests; no CUDA run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.banc_m3_context_response import ContextResponse
from tools.verify_c_banc_m3_e117_nonlinear_steps import (
    PAIR_CONTEXT,
    PAIR_WEIGHT,
    all_peak_decompositions,
    baseline_parity,
    compose_solver_weights,
    execute_response_batch,
    leg_metrics,
    peak_decomposition,
    rescue_available_feet,
    run_once,
    stage_response_inputs,
    validate_peak_records,
)


def small_fixture():
    return {'source_rate': np.array([[[0.], [1.], [1.], [1.]],
        [[0.], [2.], [2.], [2.]]], dtype=np.float64),
        'background': np.zeros((2, 4, 1), dtype=np.float64),
        'time_s': np.arange(4, dtype=np.float64)*.005}


def small_steps():
    return {'weights': np.array([[[1.]], [[2.]], [[3.]]]),
        'pair_context_indices': PAIR_CONTEXT.copy(),
        'pair_weight_indices': PAIR_WEIGHT.copy(),
        'variant_names': np.array(['selected_solver', 'shape_box', 'startup_box'])}


def small_ledger():
    return {'stage': 'input_staged', 'batch_started': 0,
        'batch_materialized': 0, 'batch_raw_saved': 0,
        'pairs_started': 0, 'pairs_materialized': 0,
        'pairs_raw_saved': 0, 'conditional_model_s_started': 0.,
        'conditional_model_s_materialized': 0.,
        'conditional_model_s_raw_saved': 0.,
        'response_call_entered': False, 'batch_reservation_requested': 0}


def fake_stage(calls):
    def evaluate(*_inputs):
        calls.append(True)
        return {key: torch.ones((6, 4, 1), dtype=torch.float64)*(i+1)
                for i, key in enumerate(('current', 'forcing', 'motor_rate',
                    'muscle_activation', 'command', 'tendon'))}
    return SimpleNamespace(evaluate=evaluate)


class NonlinearStepTests(unittest.TestCase):
    def test_pair_order_and_independent_zero_state_with_actual_cpu_helper(self):
        f, step = small_fixture(), small_steps()
        stage = ContextResponse(time_s=f['time_s'], tau=np.array([.02]),
            a=np.array([1.]), threshold=np.array([0.]),
            cap=np.array([10.]), pool=np.array([[1.]]),
            moment=np.array([[1.]]), rest=np.array([0.]),
            ltm=np.array([0]), device='cpu')
        inputs = stage_response_inputs(f, step, 'cpu')
        output = stage.evaluate(*inputs)
        np.testing.assert_array_equal(inputs[3].numpy(), PAIR_CONTEXT)
        np.testing.assert_array_equal(inputs[4].numpy(), PAIR_WEIGHT)
        self.assertEqual(output['current'].shape, (6, 4, 1))
        np.testing.assert_array_equal(output['current'][:, 0, 0].numpy(),
                                      np.zeros(6))
        np.testing.assert_array_equal(output['current'][:, 1, 0].numpy(),
                                      [1., 2., 2., 4., 3., 6.])
        np.testing.assert_allclose(output['motor_rate'][:, 0, 0].numpy(),
                                   np.zeros(6), atol=1e-14)
        self.assertFalse(torch.allclose(output['motor_rate'][0, 1],
                                        output['motor_rate'][1, 1]))

    def test_three_solver_weights_preserve_unselected_and_selected_order(self):
        base = np.array([[1., 2., 3.], [4., 5., 6.]])
        delta = np.array([[0., 0.], [.01, -.02], [-.01, .03]])
        weights = compose_solver_weights(base, np.array([0, 1]),
                                         np.array([1, 2]), delta)
        self.assertEqual(weights.shape, (3, 2, 3))
        np.testing.assert_array_equal(weights[:, 0, 0], np.ones(3))
        np.testing.assert_array_equal(weights[:, 1, 0], np.ones(3)*4)
        np.testing.assert_allclose(weights[:, 0, 1], 2*np.exp(delta[:, 0]))
        np.testing.assert_allclose(weights[:, 1, 2], 6*np.exp(delta[:, 1]))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            compose_solver_weights(base, np.array([0, 0]),
                                   np.array([1, 1]), delta)

    def test_qmax_peak_switch_signed_errors_and_exact_decomposition(self):
        baseline = np.array([[2., 0.], [0., 0.], [0., 0.]])
        trial = np.array([[-1., 0.], [0., -3.], [0., 0.]])
        result = peak_decomposition(baseline, trial, 4.5, threshold=1.,
            startup_indices=np.array([1, 2, 3]),
            time_s=np.arange(4)*.005, leg=2, metric='qmax')
        self.assertTrue(result['argmax_changed'])
        self.assertEqual(result['baseline_max_sample']['global_joint'], 14)
        self.assertEqual(result['trial_max_sample']['global_joint'], 15)
        self.assertEqual(result['baseline_max_signed_error'], 2.)
        self.assertEqual(result['trial_same_sample_signed_error'], -1.)
        self.assertEqual(result['trial_max_signed_error'], -3.)
        self.assertEqual(result['max_sample_switch_term'], 8.)
        self.assertEqual(result['same_sample_affine_gap_term'], -3.5)
        self.assertEqual(result['total_nonlinear_minus_affine'], 4.5)
        self.assertEqual(result['identity_residual'], 0.)

    def test_ltm_peak_tie_is_explicitly_ambiguous(self):
        baseline = np.array([2., -2., 0.])
        trial = np.array([1., 1., -3.])
        result = peak_decomposition(baseline, trial, 3., threshold=1.,
            startup_indices=np.array([1, 2, 3]),
            time_s=np.arange(4)*.005, leg=4, metric='LTMmax')
        self.assertEqual(result['baseline_tie_count'], 2)
        self.assertTrue(result['baseline_tie_limits_curvature_interpretation'])
        self.assertEqual(result['trial_tie_count'], 1)
        self.assertEqual(result['baseline_max_sample']['leg'], 'rm')
        self.assertNotIn('global_joint', result['baseline_max_sample'])
        self.assertEqual(result['identity_residual'], 0.)

    def test_all_48_peak_records_follow_pair_axes(self):
        f = {'startup_mask': np.zeros(2001, dtype=bool),
            'startup_target_command': np.zeros((99, 42)),
            'startup_target_tendon': np.ones((99, 6))*.2,
            'time_s': np.arange(2001)*.005,
            'context_names': np.array(['E110', 'E113'])}
        f['startup_mask'][1:100] = True
        raw = {'command': np.zeros((6, 2001, 42)),
            'tendon': np.ones((6, 2001, 6))*.2}
        metrics = {'startup_ratios': np.zeros((3, 6, 2, 4))}
        peaks = all_peak_decompositions(raw, metrics, {'fixture': f,
            'affine_ratios': np.zeros((3, 6, 2, 4))})
        self.assertEqual(validate_peak_records(peaks, metrics), 48)
        self.assertEqual(peaks['startup_box']['rh']['E113']['qmax'][
            'trial_max_sample']['global_joint'], 35)

    def test_original_e115_loss_and_startup_axis_normalization(self):
        f = {'fixed_mask': np.zeros(2001, dtype=bool),
            'common_mask': np.zeros(2001, dtype=bool),
            'startup_mask': np.zeros(2001, dtype=bool),
            'own_target': np.zeros((2, 2001, 42)),
            'startup_target_command': np.zeros((99, 42)),
            'startup_target_tendon': np.ones((99, 6))*.2,
            'joint_scale': np.ones(42), 'foot_scale': np.ones((6, 3)),
            'velocity_scale': np.ones((6, 3))}
        f['fixed_mask'][100:1928] = True
        f['common_mask'][100:1923] = True
        f['startup_mask'][1:100] = True
        command = np.zeros((6, 2001, 42))
        tendon = np.ones((6, 2001, 6))*.2
        command[0, 100:1923, 0] = 2.
        command[0, 1:100, 0] = .018
        feet = np.zeros((6, 1828, 6, 3))
        metrics = leg_metrics({'command': command, 'tendon': tendon}, feet,
            {'fixture': f, 'independent': {
                'common_target_feet': np.zeros((2, 1823, 6, 3))}})
        self.assertEqual(metrics['loss_parts'].shape, (3, 6, 2, 4))
        self.assertEqual(metrics['startup_ratios'].shape, (3, 6, 2, 4))
        self.assertAlmostEqual(metrics['loss_parts'][0, 0, 0, 0], 4/7)
        self.assertAlmostEqual(metrics['startup_ratios'][0, 0, 0, 1],
                               (.018/.04)**2)
        self.assertEqual(metrics['loss_parts'][0, 0, 0, 1], 0.)
        self.assertEqual(metrics['loss_parts'][1, 0, 0, 0], 0.)

    def test_baseline_mismatch_is_reported_without_mutating_raw(self):
        metrics = {'loss_parts': np.zeros((3, 6, 2, 4)),
                   'startup_ratios': np.zeros((3, 6, 2, 4))}
        data = {'baseline': {name: {'parts': np.zeros((2, 4)),
                                   'ratios': np.zeros((2, 4))}
                             for name in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')}}
        metrics['loss_parts'][0, 0, 0, 0] = 1e-7
        report = baseline_parity(metrics, data)
        self.assertFalse(report['passed'])
        self.assertEqual(report['loss_parts_max_abs'], 1e-7)
        self.assertEqual(metrics['loss_parts'][0, 0, 0, 0], 1e-7)

    def test_failed_start_receipt_rolls_back_and_never_calls_response(self):
        calls, state = [], small_ledger()
        original = OSError('synthetic start receipt locked')
        with tempfile.TemporaryDirectory() as temp:
            try:
                execute_response_batch(fake_stage(calls), (), small_fixture(),
                    small_steps(), Path(temp), state,
                    lambda: (_ for _ in ()).throw(original), lambda: None)
            except OSError as caught:
                self.assertIs(caught, original)
            else:
                self.fail('Mandatory start receipt failure was swallowed')
            self.assertTrue((Path(temp)/'error-response-input/weights.npy').exists())
        self.assertFalse(calls)
        self.assertEqual(state['batch_started'], 0)
        self.assertEqual(state['pairs_started'], 0)
        self.assertEqual(state['conditional_model_s_started'], 0.)
        self.assertEqual(state['batch_reservation_requested'], 6)

    def test_completed_response_raw_order_and_six_pair_accounting(self):
        calls, state = [], small_ledger()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            raw = execute_response_batch(fake_stage(calls), (),
                small_fixture(), small_steps(), out, state,
                lambda: None, lambda: None)
            with np.load(out/'response-raw.npz', allow_pickle=False) as saved:
                self.assertEqual({'current', 'forcing', 'motor_rate',
                    'muscle_activation', 'command', 'tendon', 'weights',
                    'pair_context_indices', 'pair_weight_indices', 'time_s'}
                    -set(saved.files), set())
                np.testing.assert_array_equal(saved['pair_context_indices'],
                                              PAIR_CONTEXT)
                np.testing.assert_array_equal(saved['pair_weight_indices'],
                                              PAIR_WEIGHT)
            self.assertEqual(raw['current'].shape, (6, 4, 1))
        self.assertEqual(len(calls), 1)
        self.assertEqual([state[key] for key in
            ('batch_started', 'batch_materialized', 'batch_raw_saved')],
            [1, 1, 1])
        self.assertEqual([state[key] for key in
            ('pairs_started', 'pairs_materialized', 'pairs_raw_saved')],
            [6, 6, 6])
        self.assertEqual([state[key] for key in
            ('conditional_model_s_started', 'conditional_model_s_materialized',
             'conditional_model_s_raw_saved')], [60., 60., 60.])

    def test_postmaterialization_save_failure_rescues_raw_and_primary(self):
        calls, state = [], small_ledger()
        original = OSError('synthetic response archive locked')
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.verify_c_banc_m3_e117_nonlinear_steps.np.savez_compressed',
                   side_effect=original):
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                execute_response_batch(fake_stage(calls), (), small_fixture(),
                    small_steps(), out, state, lambda: None, lambda: None)
            for key in ('current', 'forcing', 'motor_rate',
                        'muscle_activation', 'command', 'tendon', 'weights'):
                self.assertTrue((out/f'error-response-raw/{key}.npy').exists(), key)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['pairs_started'], 6)
        self.assertEqual(state['pairs_materialized'], 6)
        self.assertEqual(state['pairs_raw_saved'], 0)

    def test_postmaterialization_receipt_failure_rescues_values_and_metadata(self):
        calls, state, receipts = [], small_ledger(), []
        original = OSError('synthetic materialized receipt locked')
        def persist():
            receipts.append(True)
            if len(receipts) == 2:
                raise original
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(OSError) as caught:
                execute_response_batch(fake_stage(calls), (), small_fixture(),
                    small_steps(), out, state, persist, lambda: None)
            rescued = out/'error-response-raw'
            for key in ('current', 'forcing', 'motor_rate',
                        'muscle_activation', 'command', 'tendon', 'weights',
                        'pair_context_indices', 'pair_weight_indices',
                        'variant_names', 'time_s'):
                self.assertTrue((rescued/f'{key}.npy').exists(), key)
            np.testing.assert_array_equal(
                np.load(rescued/'pair_context_indices.npy'), PAIR_CONTEXT)
            np.testing.assert_array_equal(
                np.load(rescued/'pair_weight_indices.npy'), PAIR_WEIGHT)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['pairs_started'], 6)
        self.assertEqual(state['pairs_materialized'], 6)
        self.assertEqual(state['pairs_raw_saved'], 0)

    def test_geometry_return_rescue_uses_existing_tensor_only(self):
        original = OSError('synthetic first geometry transfer failure')
        tensor = torch.arange(108, dtype=torch.float64).reshape(6, 1, 6, 3)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            rescue_available_feet(tensor, None, 1, out, original)
            saved = np.load(out/'error-available/geometry/feet_fixed.npy')
            np.testing.assert_array_equal(saved, tensor.numpy())
            self.assertEqual(saved.shape, (6, 1, 6, 3))
        self.assertEqual(str(original), 'synthetic first geometry transfer failure')
        def transfer_fails():
            raise RuntimeError('synthetic rescue transfer failure')
        unavailable = SimpleNamespace(detach=transfer_fails)
        with tempfile.TemporaryDirectory() as temp:
            rescue_available_feet(unavailable, None, 1, Path(temp), original)
        self.assertIn('secondary geometry materialization', original.__notes__[0])

    def test_input_staging_failure_happens_before_any_batch_charge(self):
        calls, state = [], small_ledger()
        actual = torch.as_tensor
        def fail_last(value, *args, **kwargs):
            if isinstance(value, np.ndarray) and np.array_equal(value,
                    PAIR_WEIGHT):
                raise OSError('synthetic fifth tensor transfer')
            return actual(value, *args, **kwargs)
        with patch('tools.verify_c_banc_m3_e117_nonlinear_steps.torch.as_tensor',
                   side_effect=fail_last), \
             self.assertRaisesRegex(OSError, 'fifth tensor'):
            stage_response_inputs(small_fixture(), small_steps(), 'cpu')
        self.assertFalse(calls)
        self.assertEqual(state['pairs_started'], 0)

    def test_outer_preflight_error_has_fresh_wall_and_zero_response(self):
        clock = [0.]
        original = RuntimeError('synthetic preflight failure')
        def fail_preflight(_):
            clock[0] = 4.
            raise original
        with tempfile.TemporaryDirectory() as temp, \
             patch('tools.verify_c_banc_m3_e117_nonlinear_steps.shutil.copy2'), \
             patch('tools.verify_c_banc_m3_e117_nonlinear_steps.reviewed_inputs',
                   side_effect=fail_preflight), \
             patch('tools.verify_c_banc_m3_e117_nonlinear_steps.time.monotonic',
                   side_effect=lambda: clock[0]):
            out = Path(temp)/'out'
            with self.assertRaises(RuntimeError) as caught:
                run_once(Path(temp), out)
            saved = json.loads((out/'execution-ledger.json').read_text())
        self.assertIs(caught.exception, original)
        self.assertEqual(saved['process_wall_s'], 4.)
        self.assertEqual(saved['pairs_started'], 0)


if __name__ == '__main__':
    unittest.main()
