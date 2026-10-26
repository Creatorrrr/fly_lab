"""E112 model-free CPU contract tests; no CUDA, FlyBody, or optimizer fit."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import OptimizeResult

from tools import fit_c_banc_m3_dense_foot_recruitment as e112


class DenseFootRecruitmentTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(2001)*.005
        self.selected = np.arange(100, 104)

    def test_adjacent_constant_and_linear_velocity(self):
        constant = torch.full((4, 3), 2., dtype=torch.float64)
        self.assertTrue(torch.equal(e112.adjacent_velocity(
            constant, self.selected, self.times), torch.zeros((3, 3))))
        slope = torch.tensor([1., -2., 3.], dtype=torch.float64)
        position = .005*torch.arange(4, dtype=torch.float64)[:, None]*slope
        expected = slope.repeat(3, 1)
        torch.testing.assert_close(e112.adjacent_velocity(
            position, self.selected, self.times), expected, rtol=0, atol=1e-13)

    def test_adjacent_velocity_gradient_reaches_both_sides(self):
        point = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
        velocity = e112.adjacent_velocity(point, self.selected[:3], self.times)
        gradient, = torch.autograd.grad(velocity[0, 0]+2*velocity[1, 0], point)
        np.testing.assert_array_equal(gradient.detach().numpy()[:, 0],
                                      np.array([-200., -200., 400.]))
        self.assertTrue(torch.equal(gradient[:, 1:], torch.zeros((3, 2))))

    def test_adjacent_rejects_shape_nonfinite_gap_and_empty(self):
        good = torch.zeros((4, 3), dtype=torch.float64)
        bad_cases = [
            (good[:1], self.selected[:1], self.times),
            (good.float(), self.selected, self.times),
            (good[:, :2], self.selected, self.times),
            (good, np.array([100, 101, 103, 104]), self.times),
            (good, self.selected, self.times*2),
            (good.clone().fill_(float('nan')), self.selected, self.times),
        ]
        for points, selected, times in bad_cases:
            with (self.subTest(shape=tuple(points.shape),
                               selected=selected.tolist()),
                  self.assertRaises(ValueError)):
                e112.adjacent_velocity(points, selected, times)

    def test_infeasible_initial_record_never_selected(self):
        records = [
            {'objective': .1, 'ratios': [1.2, .9, .9, .9], 'log_theta': [0.]},
            {'objective': .3, 'ratios': [.5, .5, .5, .5], 'log_theta': [1.]},
        ]
        best_any, feasible = e112.choose_records(records)
        self.assertEqual(best_any['objective'], .1)
        self.assertEqual(feasible['objective'], .3)
        self.assertFalse(e112.selector_feasible(records[0]['ratios']))
        self.assertFalse(e112.selector_feasible([1., 0., 0., 0.]))

    def test_cache_counts_unique_start_before_failure(self):
        starts = []
        calls = []
        cache = e112.EvaluationCache(lambda x, i: calls.append(i) or {'x': x.copy()},
            2, lambda i, x: starts.append((i, x.copy())), max_calls=3,
            deadline=time.perf_counter()+5, total_deadline=time.perf_counter()+5)
        x = np.array([.1, -.2], dtype=np.float64)
        first = cache.get(x)
        self.assertIs(cache.get(x.copy()), first)
        self.assertEqual((cache.started, cache.completed, cache.hits), (1, 1, 1))
        self.assertEqual(starts[0][0], 1)
        self.assertEqual(calls, [1])
        def fail(x, i):
            raise RuntimeError('synthetic evaluation fault')
        failing = e112.EvaluationCache(fail, 2, lambda i, x: starts.append((i, x)),
            max_calls=1, deadline=time.perf_counter()+5,
            total_deadline=time.perf_counter()+5)
        with self.assertRaisesRegex(RuntimeError, 'synthetic evaluation fault'):
            failing.get(x)
        self.assertEqual((failing.started, failing.completed), (1, 0))

    def test_slsqp_callbacks_share_one_exact_vector_and_four_constraints(self):
        calls = []
        cache = e112.EvaluationCache(
            lambda x, i: calls.append(i) or {'objective': 2.,
                'jac': np.array([3., 4.]),
                'ratios': np.array([.1, .2, .3, .4]),
                'ratio_jac': np.ones((4, 2))},
            2, lambda i, x: None, max_calls=3,
            deadline=time.perf_counter()+5, total_deadline=time.perf_counter()+5)
        observed = {}

        def fake_minimize(fun, x0, **kwargs):
            observed['objective'] = fun(x0)
            observed['jac'] = kwargs['jac'](x0)
            observed['constraints'] = kwargs['constraints'][0]['fun'](x0)
            observed['constraint_jac'] = kwargs['constraints'][0]['jac'](x0)
            observed['options'] = kwargs['options']
            return OptimizeResult(x=x0, fun=2., jac=np.array([3., 4.]),
                success=False, status=9, message='synthetic stop',
                nit=1, nfev=1, njev=1)

        with patch.object(e112, 'minimize', side_effect=fake_minimize):
            _, termination = e112.optimize_block(cache, np.zeros(2))
        self.assertEqual(calls, [1])
        self.assertEqual((cache.started, cache.completed, cache.hits), (1, 1, 3))
        np.testing.assert_allclose(observed['constraints'],
                                   e112.OPTIMIZER_CAP-np.array([.1, .2, .3, .4]))
        np.testing.assert_array_equal(observed['constraint_jac'], -np.ones((4, 2)))
        self.assertEqual(observed['options']['maxiter'], 1000)
        self.assertEqual(termination['stop'], 'scipy_finished')

    def test_synthetic_native_response_velocity_loss_reaches_log_weight(self):
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
        source = torch.linspace(0., 1., 2001, dtype=torch.float64)[:, None]
        selected = np.arange(100, 104)
        staged = SimpleNamespace(source_rate=source, selected=selected,
            startup=np.arange(1, 100), train={'time_s': self.times},
            geometry=SimpleNamespace(leg=lambda q, leg: q[:, :3]))
        block = {'leg': 0, 'original_w': tensor([[0.]]),
            'rows': torch.tensor([0]), 'cols': torch.tensor([0]),
            'signed': tensor([1.]), 'background': tensor(np.zeros((2001, 1))),
            'tau': tensor([.02]), 'a': tensor([1.]),
            'threshold': tensor([0.]), 'cap': tensor([1.]),
            'pool': tensor([[1.]]), 'moment': tensor(np.ones((7, 1))),
            'rest': tensor(np.zeros(7)), 'ltm': 0,
            'target_q': tensor(np.zeros((4, 7))),
            'target_foot': tensor(np.zeros((4, 3))),
            'target_velocity': tensor(np.zeros((3, 3))),
            'q_scale': tensor(np.ones(7)), 'foot_scale': tensor(np.ones(3)),
            'velocity_scale': tensor(np.ones(3)),
            'startup_q': tensor(np.zeros((99, 7))),
            'startup_tendon': tensor(np.zeros(99))}
        x = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        result = e112.block_tensors(staged, block, x)
        self.assertGreater(float(result['parts'][2].detach()), 0.)
        gradient, = torch.autograd.grad(result['parts'][2], x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertNotEqual(float(gradient[0]), 0.)
        self.assertEqual(result['velocity'].shape, (3, 3))

    def test_real_frozen_inputs_and_unsorted_float32_replay_are_model_free(self):
        root = Path(__file__).resolve().parents[1]/(
            'verification/banc-natural-gait-m3-20260919')
        with tempfile.TemporaryDirectory() as temporary:
            train, cal, parent, geometry = e112.reviewed_inputs(root, Path(temporary))
            audit = json.loads((Path(temporary)/'background-roundoff.json').read_text())
            self.assertTrue(audit['passed'])
            self.assertFalse(audit['stored_background_replaced'])
        self.assertEqual(train['selected'].shape, (1828,))
        self.assertEqual(geometry.geometry_hash, e112.EXPECTED_GEOMETRY_HASH)
        self.assertEqual(parent['command'].shape, (2001, 42))
        self.assertFalse(np.all(np.diff(cal['edge_ids']) > 0))
        dense, restored = e112.selected_actual_float32(
            train, cal, train['initial_efficacy'])
        np.testing.assert_array_equal(dense, train['current_w'])
        np.testing.assert_array_equal(restored['values'], cal['values'])
        np.testing.assert_array_equal(restored['tau_ids'], cal['tau_ids'])

    def test_parent_background_roundoff_accepts_ulp_and_preserves_data(self):
        current = np.full((2, 1), 2., dtype=np.float64)
        source = np.full((2, 3), .1, dtype=np.float64)
        weights = np.ones((1, 3), dtype=np.float64)
        background = np.nextafter(current-source@weights.T, np.inf)
        before = background.tobytes()
        with tempfile.TemporaryDirectory() as temporary:
            report = e112.audit_background_roundoff(current, source, weights,
                                                    background, Path(temporary))
            self.assertTrue(report['passed'])
            self.assertEqual(report['bit_unequal_elements'], 2)
        self.assertEqual(background.tobytes(), before)

    def test_parent_background_real_discrepancy_fails_after_raw_saved(self):
        current = np.full((2, 1), 2., dtype=np.float64)
        source = np.full((2, 3), .1, dtype=np.float64)
        weights = np.ones((1, 3), dtype=np.float64)
        background = current-source@weights.T
        background[0, 0] += 1e-6
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'exceeds float64 roundoff'):
                e112.audit_background_roundoff(current, source, weights, background, out)
            self.assertFalse(json.loads((out/'background-roundoff.json').read_text())['passed'])
            with np.load(out/'background-roundoff-raw.npz') as raw:
                np.testing.assert_array_equal(raw['stored_background'], background)
                self.assertGreater(raw['absolute_difference'][0, 0], raw['bound'][0, 0])

    def test_parent_exact_input_hash_still_rejects_mutation(self):
        root = Path(__file__).resolve().parents[1]/(
            'verification/banc-natural-gait-m3-20260919')
        original = e112.sha256

        def changed_digest(path):
            return '0'*64 if Path(path).name == 'parent-e112-training-inputs.npz' \
                else original(path)

        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(e112, 'sha256', side_effect=changed_digest),
              self.assertRaisesRegex(ValueError, 'training input hash changed')):
            e112.reviewed_inputs(root, Path(temporary))

    def test_float32_weight_rounding_and_final_inner_recheck(self):
        root = Path(__file__).resolve().parents[1]/(
            'verification/banc-natural-gait-m3-20260919')
        train = e112.load_npz(root/'parent-e112-training-inputs.npz')
        cal = e112.load_npz(root/'e110-e109-native-body-diagnostic/calibration.npz')
        efficacy = train['initial_efficacy'].copy()
        efficacy[0] *= 1.000001
        dense, updated = e112.selected_actual_float32(train, cal, efficacy)
        self.assertEqual(dense[train['edge_rows'][0], train['edge_cols'][0]],
                         float(np.float32(train['original_signed_edge_weights'][0]
                                          *efficacy[0])))
        self.assertTrue(np.array_equal(updated['tau_values'], cal['tau_values']))
        command = np.zeros((2001, 42))
        tendon = np.zeros((2001, 6))
        fake = {'startup_mask': np.zeros(2001, dtype=bool),
                'startup_target_command': np.zeros((99, 42)),
                'startup_target_tendon': np.zeros((99, 6))}
        fake['startup_mask'][1:100] = True
        command[1, 0] = float(np.float32(.04+1e-8))
        result = e112.startup_details(command, tendon, fake, e112.INNER)
        self.assertFalse(result['supported'])
        self.assertFalse(result['selector_margin_supported'])

    def test_error_receipt_before_any_cuda_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'single'
            with (patch.object(e112, 'reviewed_inputs',
                               side_effect=ValueError('synthetic input failure')),
                  self.assertRaisesRegex(ValueError, 'synthetic input failure')):
                e112.run(Path(temporary), output)
            receipt = json.loads((output/'execution-error.json').read_text())
            ledger = json.loads((output/'execution-ledger.json').read_text())
            self.assertEqual(receipt['error_type'], 'ValueError')
            self.assertEqual(ledger['baseline_validation_calls'], 0)
            self.assertEqual(sum(ledger['leg_started_evaluations'].values()), 0)
            self.assertEqual(ledger['actual_neural_s'], 0.)
            self.assertEqual(ledger['actual_physical_s'], 0.)

    def test_full_validation_faults_charge_each_production_boundary(self):
        for phase in ('baseline', 'final'):
            for fault in ('forward', 'materialize', 'raw_save'):
                with (self.subTest(phase=phase, fault=fault),
                      tempfile.TemporaryDirectory() as temporary):
                    out = Path(temporary)
                    ledger = {name+suffix: 0 for name in ('baseline', 'final')
                        for suffix in ('_validation_calls', '_response_completed',
                                       '_raw_saved')}

                    def full(_weights, failure=fault):
                        if failure == 'forward':
                            raise ValueError('original forward failure')
                        return {'command': torch.ones((2, 3), dtype=torch.float64)}

                    staged = SimpleNamespace(full=full)
                    context = (patch.object(e112, 'detach_np',
                        side_effect=ValueError('original materialization failure'))
                        if fault == 'materialize' else
                        patch.object(e112.np, 'savez_compressed',
                        side_effect=OSError('original raw save failure'))
                        if fault == 'raw_save' else patch.object(e112, 'detach_np',
                        wraps=e112.detach_np))
                    expected = {'forward': ValueError,
                        'materialize': ValueError, 'raw_save': OSError}[fault]
                    with (context, self.assertRaises(expected) as raised):
                        e112.full_validation_response(staged, np.zeros(1), out,
                            ledger, phase=phase, payload={})
                    description = {'forward': 'forward',
                        'materialize': 'materialization',
                        'raw_save': 'raw save'}[fault]
                    self.assertIn('original '+description+' failure',
                                  str(raised.exception))
                    saved = json.loads((out/'execution-ledger.json').read_text())
                    for counts in (ledger, saved):
                        self.assertEqual(counts[phase+'_validation_calls'], 1)
                        self.assertEqual(counts[phase+'_response_completed'],
                                         int(fault == 'raw_save'))
                        self.assertEqual(counts[phase+'_raw_saved'], 0)
                        other = 'final' if phase == 'baseline' else 'baseline'
                        self.assertEqual(counts[other+'_validation_calls'], 0)

    def test_full_validation_success_charges_materialization_and_raw_save(self):
        for phase in ('baseline', 'final'):
            with (self.subTest(phase=phase),
                  tempfile.TemporaryDirectory() as temporary):
                out = Path(temporary)
                ledger = {name+suffix: 0 for name in ('baseline', 'final')
                    for suffix in ('_validation_calls', '_response_completed',
                                   '_raw_saved')}
                staged = SimpleNamespace(full=lambda _: {
                    'command': torch.ones((2, 3), dtype=torch.float64)})
                arrays = e112.full_validation_response(staged, np.zeros(1), out,
                    ledger, phase=phase, payload={})
                np.testing.assert_array_equal(arrays['command'], np.ones((2, 3)))
                path = out/('baseline-cuda-raw.npz' if phase == 'baseline'
                            else 'prediction.npz')
                self.assertTrue(path.exists())
                saved = json.loads((out/'execution-ledger.json').read_text())
                self.assertEqual([saved[phase+suffix] for suffix in (
                    '_validation_calls', '_response_completed', '_raw_saved')],
                    [1, 1, 1])


    def test_parent_raw_survives_completion_ledger_failure(self):
        for phase in ('baseline', 'final'):
            with (self.subTest(phase=phase),
                  tempfile.TemporaryDirectory() as temporary):
                out = Path(temporary)
                ledger = {name+suffix: 0 for name in ('baseline', 'final')
                    for suffix in ('_validation_calls', '_response_completed',
                                   '_raw_saved')}
                staged = SimpleNamespace(full=lambda _: {
                    'command': torch.ones((2, 3), dtype=torch.float64)})
                with (patch.object(e112, 'save_json', side_effect=[
                        None, OSError('completion ledger locked')]),
                      self.assertRaisesRegex(OSError, 'completion ledger locked')):
                    e112.full_validation_response(staged, np.zeros(1), out,
                        ledger, phase=phase, payload={})
                path = out/('baseline-cuda-raw.npz' if phase == 'baseline'
                            else 'prediction.npz')
                with np.load(path) as raw:
                    np.testing.assert_array_equal(raw['command'], np.ones((2, 3)))
                self.assertEqual([ledger[phase+suffix] for suffix in (
                    '_validation_calls', '_response_completed', '_raw_saved')],
                    [1, 1, 1])

    def test_parent_raw_failure_remains_primary_if_completion_ledger_fails(self):
        for phase in ('baseline', 'final'):
            with (self.subTest(phase=phase),
                  tempfile.TemporaryDirectory() as temporary):
                ledger = {name+suffix: 0 for name in ('baseline', 'final')
                    for suffix in ('_validation_calls', '_response_completed',
                                   '_raw_saved')}
                staged = SimpleNamespace(full=lambda _: {
                    'command': torch.ones((2, 3), dtype=torch.float64)})
                with (patch.object(e112, 'save_json', side_effect=[
                        None, OSError('completion ledger locked')]),
                      patch.object(e112.np, 'savez_compressed',
                        side_effect=OSError('original raw failure')),
                      self.assertRaisesRegex(OSError, 'original raw failure') as raised):
                    e112.full_validation_response(staged, np.zeros(1),
                        Path(temporary), ledger, phase=phase, payload={})
                self.assertIn('completion ledger locked',
                              ' '.join(raised.exception.__notes__))
                self.assertEqual([ledger[phase+suffix] for suffix in (
                    '_validation_calls', '_response_completed', '_raw_saved')],
                    [1, 1, 0])


if __name__ == '__main__':
    unittest.main()
