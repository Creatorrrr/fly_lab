"""Model-free E113 guards for provenance, partial clocks and fail-closed screens."""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.sparse import csr_matrix

from tools import verify_c_banc_m3_e113_body as e113
from tools.verify_c_banc_m3_e113_body import (
    AdvanceLedger,
    NativeHealthStop,
    body_quality,
    finite_error,
    install_native_observer,
    jsonable,
    prepare_neutral_checkpoint,
    resource_screen,
    reviewed_inputs,
    run_body,
    save_json,
    save_raw,
    startup_diagnostic,
    supported_slip,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'verification/banc-natural-gait-m3-20260919'


class TestE113Body(unittest.TestCase):
    def test_actual_reviewed_input_and_independent_fixture_without_model(self):
        calibration, baseline, selected, prediction, expected, parent = reviewed_inputs(EVIDENCE)
        self.assertEqual(len(calibration['tau_ids']), 38)
        np.testing.assert_array_equal(calibration['tau_ids'], baseline['tau_ids'])
        np.testing.assert_array_equal(calibration['tau_values'], baseline['tau_values'])
        self.assertEqual(calibration['theta'].shape, (96, 48))
        self.assertEqual({key for key in calibration if not np.array_equal(
            calibration[key], baseline[key])},
            {'values', 'individual_edge_efficacies'})
        self.assertEqual(len(selected['edge_ids']), 1572)
        self.assertFalse(np.all(np.diff(calibration['edge_ids']) >= 0))
        positions = {int(edge): index for index, edge in
                     enumerate(calibration['edge_ids'])}
        mapped = np.asarray([positions[int(edge)] for edge in selected['edge_ids']])
        np.testing.assert_array_equal(calibration['values'][mapped], np.asarray(
            selected['original_signed_weights'].astype(float)*selected['efficacy'],
            dtype=np.float32))
        native = e113.patch_tau(np.full(158706, .02, np.float32),
                                calibration['tau_ids'], calibration['tau_values'])
        previous = e113.patch_tau(np.full(158706, .02, np.float32),
                                  baseline['tau_ids'], baseline['tau_values'])
        np.testing.assert_array_equal(native, previous)
        np.testing.assert_array_equal(native[calibration['motor_ids']],
                                      expected['native_motor_tau'])
        self.assertEqual(len(expected['basic_ids']), 442)
        self.assertEqual(len(expected['detailed_ids']), 446)
        self.assertEqual(len(expected['metadata_source_ids']), 55)
        self.assertEqual(prediction['command'].shape, (2001, 42))
        self.assertEqual(len(parent['fixture_sha256']), 2)
        manifest = json.loads((EVIDENCE/'e112-dense-foot-recruitment-v2'
            /'parent-output-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['original_fit_artifact_sha256']), 59)
        self.assertNotIn('parent-review-stdout.txt',
                         manifest['original_fit_artifact_sha256'])

    def test_reviewed_input_rejects_changed_original_fit_artifact_without_model(self):
        original = e113.sha256
        changed = EVIDENCE/'e112-dense-foot-recruitment-v2/calibration.npz'

        def fake_hash(path):
            return '0'*64 if Path(path) == changed else original(path)

        with (patch.object(e113, 'sha256', side_effect=fake_hash),
              self.assertRaisesRegex(ValueError, 'Parent E113 input changed')):
            reviewed_inputs(EVIDENCE)

    def test_prepare_uses_original_full_tau_and_only_selected_signed_edge(self):
        base = csr_matrix((np.array([1.], dtype=np.float32),
            (np.array([0]), np.array([0]))), shape=(158706, 158706))
        graph = SimpleNamespace(n=158706, indptr=base.indptr,
                                indices=base.indices)
        original = {'body': {'state': np.array([0.])},
            'world': {'flat': True}, 'control_tick': 0,
            'network': {key: np.array([0.]) for key in
                ('rate', 'adaptation', 'drive', 'output_mask', 'tick')}}
        baseline = {'edge_ids': np.array([0]),
            'source_values': np.array([1.], dtype=np.float32),
            'values': np.array([1.], dtype=np.float32),
            'tau_ids': np.array([0]), 'tau_values': np.array([.02])}
        cal = {**baseline, 'values': np.array([2.], dtype=np.float32),
            'motor_ids': np.array([0]), 'individual_edge_ids': np.array([0]),
            'individual_edge_motor_ids': np.array([0]),
            'individual_edge_source_ids': np.array([0]),
            'individual_source_basis_ids': np.array([0])}
        selected = {'edge_rows': np.array([0]), 'edge_cols': np.array([0]),
            'original_signed_weights': np.array([1.], dtype=np.float32),
            'efficacy': np.array([2.])}
        prediction = {'actual_float32_source_weights': np.array([[2.]])}
        expected = {'native_motor_tau': np.array([.02], dtype=np.float32),
            'actual_float32_source_weights': np.array([[2.]])}
        fixture = {'fixture_sha256': {'cal': 'synthetic'}}
        encoder = SimpleNamespace(calibrate_weights=lambda weights, ids: weights)
        frozen_anatomy_path = EVIDENCE/'e110-e109-native-body-diagnostic/muscle-provenance.json'
        unreviewed_path = EVIDENCE/'e105-startup-native-body-diagnostic-v2/muscle-provenance.json'
        frozen_anatomy = json.loads(frozen_anatomy_path.read_text(encoding='utf-8'))
        original_read_text = Path.read_text
        read_paths = []

        def read_with_unreviewed_sentinel(path, *args, **kwargs):
            read_paths.append(path)
            if path == unreviewed_path:
                return '{"muscles": ["UNFROZEN_E105_SENTINEL"]}'
            return original_read_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (patch.object(e113, 'reviewed_inputs', return_value=(
                    cal, baseline, selected, prediction, expected, fixture)),
                  patch.object(e113.GraphStore, 'load', return_value=graph),
                  patch.object(e113.StateStore, 'load', return_value=original),
                  patch.object(e113, 'rate_weights', return_value=(base, None)),
                  patch.object(e113, 'BancTargetEncoder', return_value=encoder),
                  patch.object(Path, 'read_text', read_with_unreviewed_sentinel),
                  patch.object(e113, 'prepare_neutral_checkpoint') as checkpoint):
                prepared = e113.prepare(EVIDENCE, out, out, time.perf_counter())
            self.assertEqual(prepared[4]['validity'],
                             'valid_prepared_E112b_body_diagnostic')
            self.assertTrue(prepared[4]['full_158706_tau_unchanged'])
            self.assertEqual(checkpoint.call_count, 1)
            self.assertEqual(checkpoint.call_args.args[3], frozen_anatomy)
            self.assertIn(frozen_anatomy_path, read_paths)
            self.assertNotIn(unreviewed_path, read_paths)
            np.testing.assert_array_equal(checkpoint.call_args.args[4].data,
                                          np.array([2.], dtype=np.float32))
            self.assertEqual(checkpoint.call_args.args[5].shape, (158706,))
            with patch.object(e113, 'reviewed_inputs', return_value=(
                    cal, {**baseline, 'tau_values': np.array([.03])}, selected,
                    prediction, expected, fixture)), patch.object(
                    e113.GraphStore, 'load', return_value=graph), patch.object(
                    e113.StateStore, 'load', return_value=original), self.assertRaisesRegex(
                    AssertionError, 'full158706 E110/E112b tau equivalence'):
                e113.prepare(EVIDENCE, out, out, time.perf_counter())

    def test_main_writes_e113_provenance_without_constructing_a_model(self):
        prepared = {'validity': 'valid_prepared_E112b_body_diagnostic',
                    'resource_limit_met': True, 'wall_s': 0.}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'e113-only'
            with (patch.object(e113, 'prepare', return_value=(
                    None, {}, {}, {}, prepared)) as prepare,
                  patch.object(e113, 'run_body', return_value={'done': True}) as body):
                self.assertEqual(e113.main(EVIDENCE, Path(temp)/'unused-graph', out),
                                 {'done': True})
            prepare.assert_called_once()
            body.assert_called_once()
            spec = json.loads((out/'spec.json').read_text(encoding='utf-8'))
            self.assertEqual(spec['experiment'], 'M3-E113-dense-foot-native-body')
            self.assertEqual(spec['limits']['prior_conservative_wall_s'], 450.)
            self.assertEqual(spec['limits']['shared_wall_s'], 630.)
            self.assertEqual(spec['E112b_calibration_sha256'],
                '628e5099bf6ea4372a81d5041a957785f75677020743215b2455af38ce9254eb')
            self.assertEqual(spec['parent_input_manifest_sha256'],
                '96789069a39aaf4192becc0b4491c465a871691ca89bfd0de00509c04026984a')
            self.assertEqual((out/'runner.py').read_bytes(),
                             Path(e113.__file__).read_bytes())

    def test_main_receipt_failure_preserves_original_prepare_error_identity(self):
        original = RuntimeError('original prepare failure')
        original_save = e113.save_json

        def failing_receipt(path, value):
            if path.name == 'execution-error.json':
                raise OSError('synthetic execution receipt write locked')
            return original_save(path, value)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'single-e113'
            with (patch.object(e113, 'prepare', side_effect=original),
                  patch.object(e113, 'save_json', side_effect=failing_receipt),
                  self.assertRaises(RuntimeError) as raised):
                e113.main(EVIDENCE, Path(temp)/'unused-graph', out)
            self.assertIs(raised.exception, original)
            self.assertIn('synthetic execution receipt write locked',
                          ' '.join(original.__notes__))
            self.assertTrue((out/'spec.json').is_file())

    def test_nonfinite_values_are_null_and_cannot_become_zero_error(self):
        error = finite_error(np.array([1e308, -1e308]), np.zeros(2))
        self.assertFalse(error['available'])
        self.assertIsNone(error['RMS'])
        self.assertIsNone(error['max_abs'])
        self.assertEqual(jsonable({'x': np.nan, 'y': np.inf}),
                         {'x': None, 'y': None})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            save_json(path, {'x': np.nan})
            self.assertIn('null', path.read_text(encoding='utf-8'))

    def test_original99_incomplete_and_nonfinite_fail_closed(self):
        prediction = {'command': np.zeros((2001, 42)),
                      'tendon': np.zeros((2001, 6))}
        startup = {'target_command': np.zeros((99, 42)),
                   'target_tendon': np.zeros((99, 6))}
        rows = [{'time_s': i*.005, 'joint_targets_rad': [0.]*42,
                 'tendon_inputs': {f'{leg}_tarsus': 0. for leg in
                     ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')}} for i in range(100)]
        partial = startup_diagnostic(rows[:-1], [None]*4900, prediction, startup)
        self.assertFalse(partial['complete_original99_prefix'])
        self.assertIsNone(partial['E112b_conditional_q']['RMS'])
        self.assertFalse(partial['E112b_conditional_transfer_screen'])
        rows[-1]['joint_targets_rad'][0] = math.inf
        nonfinite = startup_diagnostic(rows, [None]*4950, prediction, startup)
        self.assertTrue(nonfinite['complete_original99_prefix'])
        self.assertFalse(nonfinite['E112b_conditional_q']['available'])
        self.assertFalse(nonfinite['original_E100_E103_startup_screen'])

    def test_partial_real_advance_charges_neural_and_physics_separately(self):
        class FakeSession:
            def __init__(self):
                self.network = SimpleNamespace(tick=0)
                self.body = SimpleNamespace(physics_time=lambda: self.physics)
                self.physics = 0.
                self.control_tick = 0

            def advance(self, controls):
                self.network.tick += 20
                self.physics += .0003
                raise NativeHealthStop('partial native step')

        trial = FakeSession()
        ledger = AdvanceLedger()
        with self.assertRaises(NativeHealthStop):
            ledger.advance(trial, 1, 'main')
        self.assertAlmostEqual(ledger.neural_s, .005)
        self.assertAlmostEqual(ledger.physical_s, .0003)
        self.assertEqual(ledger.events[0]['control_tick_after'], 0)
        self.assertEqual(ledger.events[0]['advance_error'], 'NativeHealthStop')
        self.assertAlmostEqual(ledger.by_owner()['main']['actual_physical_s'], .0003)
        self.assertEqual(ledger.by_owner()['main']['incomplete_or_failed_calls'], 1)

    def test_nonfinite_physics_clock_keeps_neural_cost_and_null_physics(self):
        class FakeSession:
            def __init__(self):
                self.network = SimpleNamespace(tick=0)
                self.time = 0.
                self.body = SimpleNamespace(physics_time=lambda: self.time)
                self.control_tick = 0

            def advance(self, controls):
                self.network.tick += 20
                self.time = math.nan
                raise RuntimeError('nonfinite body')

        ledger = AdvanceLedger()
        with self.assertRaises(RuntimeError):
            ledger.advance(FakeSession(), 1, 'main')
        self.assertAlmostEqual(ledger.neural_s, .005)
        self.assertIsNone(ledger.physical_s)
        self.assertFalse(ledger.clock_valid)

    def test_after_neural_clock_failure_retains_primary_error_and_physics_cost(self):
        class Network:
            reads = 0

            @property
            def tick(self):
                self.reads += 1
                if self.reads > 1:
                    raise OSError('neural clock read failed')
                return 0

        class Trial:
            def __init__(self):
                self.network = Network()
                self.physics = 0.
                self.body = SimpleNamespace(physics_time=lambda: self.physics)
                self.control_tick = 0

            def advance(self, controls):
                self.physics += .0003
                raise NativeHealthStop('native partial fault')

        ledger = AdvanceLedger()
        with self.assertRaisesRegex(NativeHealthStop, 'native partial fault'):
            ledger.advance(Trial(), 1, 'main')
        self.assertEqual(len(ledger.events), 1)
        self.assertFalse(ledger.clock_valid)
        self.assertIsNone(ledger.neural_s)
        self.assertAlmostEqual(ledger.physical_s, .0003)
        self.assertEqual(ledger.events[0]['advance_error'], 'NativeHealthStop')
        self.assertIn('neural clock read failed',
                      ledger.events[0]['clock_read_errors'][0])

    def test_preparation_original_snapshot_error_closes_clone_and_main(self):
        original = {'body': {'state': np.array([0.])}, 'world': {'flat': True},
                    'network': {key: np.array([0.]) for key in
                        ('rate', 'adaptation', 'drive', 'output_mask', 'tick')}}
        neutral = {'body': original['body'], 'world': original['world'],
                   'network': original['network']}
        weights = SimpleNamespace(indptr=np.array([0, 1]),
                                  indices=np.array([0]), data=np.array([1.]))
        main = MagicMock()
        main.network.device = 'cuda'
        main.network.tau.cpu.return_value.numpy.return_value = np.array([.02],
                                                                       dtype=np.float32)
        main.snapshot.return_value = neutral
        clone = MagicMock()
        clone.snapshot.side_effect = OSError('clone snapshot failed')
        clone.close.side_effect = OSError('clone close failed')
        calibration = {'edge_ids': np.array([0]), 'source_values': np.array([1.]),
            'values': np.array([1.]), 'zero_adaptation_ids': np.array([]),
            'tau_ids': np.array([0]), 'tau_values': np.array([.02])}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (patch('tools.verify_c_banc_m3_e113_body.EfficacyTrial'),
                  patch('tools.verify_c_banc_m3_e113_body.RecruitmentTrial'),
                  patch('tools.verify_c_banc_m3_e113_body.IntrinsicRecruitmentTrial')
                    as trial_class,
                  patch('tools.verify_c_banc_m3_e113_body.source_matrix',
                        return_value=weights),
                  patch('tools.verify_c_banc_m3_e113_body.StateStore.save'),
                  patch('tools.verify_c_banc_m3_e113_body.StateStore.load',
                        return_value=neutral),
                  self.assertRaisesRegex(OSError, 'clone snapshot failed') as caught):
                trial_class.return_value = main
                trial_class.from_checkpoint.return_value = clone
                prepare_neutral_checkpoint(SimpleNamespace(), original,
                    calibration, {'muscles': []}, weights,
                    np.array([.02], dtype=np.float32), out)
            main.close.assert_called_once()
            clone.close.assert_called_once()
            self.assertIn('clone close failed', ' '.join(caught.exception.__notes__))
            self.assertTrue((out/'preparation-cleanup-errors.json').is_file())

    def test_trace_write_failure_preserves_other_raw_artifacts_and_clocks(self):
        original_save = save_json

        def failing_trace(path, value):
            if path.name == 'trace.json':
                raise OSError('trace persistence failed')
            return original_save(path, value)

        session = SimpleNamespace(network=SimpleNamespace(identity='network'),
            body=SimpleNamespace(model_hash='body'),
            adapter=SimpleNamespace(identity='adapter'))
        basic = np.arange(442)
        detailed = np.arange(446)
        ledger = AdvanceLedger()
        ledger.events.append({'owner': 'main', 'advance_error': 'NativeHealthStop',
            'clock_read_errors': [], 'actual_neural_s': .005,
            'actual_physical_s': .0003})
        ledger.neural_s = .005
        ledger.physical_s = .0003
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch('tools.verify_c_banc_m3_e113_body.save_json',
                       side_effect=failing_trace):
                errors = save_raw(out, SimpleNamespace(hash='graph'),
                    {'parameters': {}}, session, [], basic, detailed,
                    np.array([440, 441]), np.arange(55), [], [], [], ledger, {})
            self.assertEqual([error['artifact'] for error in errors], ['trace.json'])
            for name in ('clock-events.json', 'clock-summary.json',
                         'native-contact.npz', 'clone-native-contact.npz',
                         'signals.npz', 'recruitment-signals.npz'):
                self.assertTrue((out/name).is_file(), name)
            receipt = json.loads((out/'clock-summary.json').read_text())
            self.assertEqual(receipt['actual_physical_s'], .0003)

    def test_original_process_and_shared_resource_bounds(self):
        ledger = AdvanceLedger()
        ledger.neural_s = 10.1
        ledger.physical_s = 10.1
        preparation = {'resource_limit_met': True}
        self.assertTrue(resource_screen(preparation, 140, 180, ledger))
        self.assertFalse(resource_screen(preparation, 140, 180.001, ledger))
        self.assertFalse(resource_screen(preparation, 140.001, 170, ledger))
        ledger.physical_s = None
        self.assertFalse(resource_screen(preparation, 139, 179, ledger))

    def test_body_session_closes_when_initial_snapshot_fails(self):
        class FakeSession:
            def __init__(self):
                self.network = SimpleNamespace(device='cuda')
                self.body = SimpleNamespace(fault=None)
                self.fault = None
                self.closed = False

            def snapshot(self):
                raise OSError('snapshot failed before advance')

            def close(self):
                self.closed = True

        session = FakeSession()
        prediction = {'command': np.zeros((2001, 42)),
                      'tendon': np.zeros((2001, 6))}
        preparation = {'validity': 'valid_prepared_E112b_body_diagnostic',
                       'resource_limit_met': True, 'wall_s': 0.}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (patch('tools.verify_c_banc_m3_e113_body.StateStore.load',
                        return_value={'network': {'device': 'cuda'}, 'parameters': {}}),
                  patch('tools.verify_c_banc_m3_e113_body.'
                        'IntrinsicRecruitmentTrial.from_checkpoint',
                        return_value=session),
                  self.assertRaisesRegex(OSError, 'snapshot failed')):
                run_body(EVIDENCE, out, SimpleNamespace(), {}, prediction, {},
                         preparation, {}, time.perf_counter())
            self.assertTrue(session.closed)
            self.assertTrue((out/'clock-summary.json').is_file())
            self.assertTrue((out/'result.json').is_file())
            self.assertEqual(json.loads((out/'result.json').read_text())['validity'],
                             'incomplete')

    def test_shadow_native_guard_stops_first_bad_substep_without_actor_forward(self):
        class Mj:
            mjtState = SimpleNamespace(mjSTATE_INTEGRATION=0)

            @staticmethod
            def MjData(model):
                return SimpleNamespace(qpos=np.array([0.]), qvel=np.array([0.]))

            @staticmethod
            def mj_stateSize(model, kind):
                return 1

            @staticmethod
            def mj_getState(model, data, state, kind):
                state[0] = data.qpos[0]

            @staticmethod
            def mj_setState(model, data, state, kind):
                data.qpos[0] = state[0]

            @staticmethod
            def mj_forward(model, data):
                data.qpos[0] += 100  # would corrupt actor if its MjData were reused

        class Body:
            def __init__(self):
                self.mj = Mj()
                self.m = object()
                self.d = SimpleNamespace(qpos=np.array([0.]),
                                         qvel=np.array([0.]), time=0.)
                self.fault = None

            def _step_physics(self):
                self.d.qpos[0] += 1
                self.d.time += .0001

            def physics_time(self):
                return self.d.time

            def pose(self):
                return None, np.eye(3), None

            def nonfoot_contact(self):
                return True

        body = Body()
        samples = []
        with patch('tools.verify_c_banc_m3_e113_body.contact_diagnostics',
                   return_value={'normal_load_bw': np.ones(6),
                                 'slip_mm_s': [None]*6}):
            restore = install_native_observer(body, samples)
            with self.assertRaises(NativeHealthStop):
                body._step_physics()
            restore()
        self.assertEqual(len(samples), 1)
        self.assertEqual(body.d.qpos[0], 1.)
        self.assertEqual(body.physics_time(), .0001)

    def test_incomplete_body_never_uses_plausible_partial_quality(self):
        rows = [{'time_s': i*.005, 'body': {'position': [i*.1, 0., 0.],
                 'basis': np.eye(3).tolist()}, 'fault': None} for i in range(100)]
        quality = {'events': {leg: {'complete_cycles': 20, 'period_cv': .1}
                    for leg in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')},
                   'pair_phase': {pair: {'mean_cycles': .5,
                                        'resultant_length': 1., 'samples': 100}
                     for pair in ('lf-rf', 'lm-rm', 'lh-rh')},
                   'knee_interior_deg': {leg: {'p05': 0., 'p95': 90.}
                    for leg in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')}}
        summary = body_quality(rows, [], quality, reason='native_health_guard',
            fault=True, clock_valid=True, future_exact=False)
        self.assertFalse(summary['complete_10s_native_health'])
        self.assertIsNone(summary['fixed_initial_heading_net_forward_mm'])
        self.assertTrue(all(x is False for x in summary['development_screens'].values()))
        self.assertTrue(all(x is None for x in summary['period_cv'].values()))

    def test_supported_slip_uses_only_valid_loaded_original_window(self):
        times = np.arange(1, 100001)*.0001
        loads = np.zeros((100000, 6))
        slip = np.full((100000, 6), np.nan)
        valid = np.zeros((100000, 6), dtype=bool)
        loads[6000, 0], slip[6000, 0], valid[6000, 0] = 1., 2., True
        loads[7000, 0], slip[7000, 0], valid[7000, 0] = 3., 4., True
        data = {'time_s': times, 'normal_load_bw': loads,
                'slip_mm_s': slip, 'slip_valid': valid}
        self.assertIsNone(supported_slip(data, complete=False)['load_weighted_mm_s'])
        result = supported_slip(data, complete=True)
        self.assertTrue(result['available'])
        self.assertEqual(result['support_samples'], 2)
        self.assertEqual(result['load_weighted_mm_s'], 3.5)
        loads[8000, 0] = 1.
        self.assertIsNone(supported_slip(data, complete=True)['load_weighted_mm_s'])


if __name__ == '__main__':
    unittest.main()

