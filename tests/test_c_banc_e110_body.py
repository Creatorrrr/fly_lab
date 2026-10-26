"""Model-free E110 guards for provenance, partial clocks and fail-closed screens."""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from tools.verify_c_banc_m3_e110_body import (
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


class TestE110Body(unittest.TestCase):
    def test_actual_reviewed_input_and_independent_fixture_without_model(self):
        calibration, fitted, selected, prediction, expected, parent = reviewed_inputs(EVIDENCE)
        self.assertEqual(len(calibration['tau_ids']), 38)
        self.assertEqual(len(fitted['tau_ids']), 428)
        self.assertEqual(len(selected['edge_ids']), 1572)
        self.assertEqual(len(expected['basic_ids']), 442)
        self.assertEqual(len(expected['detailed_ids']), 446)
        self.assertEqual(len(expected['metadata_source_ids']), 55)
        self.assertEqual(prediction['command'].shape, (2001, 42))
        self.assertEqual(len(parent['fixture_sha256']), 2)

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
        self.assertIsNone(partial['E109_conditional_q']['RMS'])
        self.assertFalse(partial['E109_conditional_transfer_screen'])
        rows[-1]['joint_targets_rad'][0] = math.inf
        nonfinite = startup_diagnostic(rows, [None]*4950, prediction, startup)
        self.assertTrue(nonfinite['complete_original99_prefix'])
        self.assertFalse(nonfinite['E109_conditional_q']['available'])
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
            with (patch('tools.verify_c_banc_m3_e110_body.EfficacyTrial'),
                  patch('tools.verify_c_banc_m3_e110_body.RecruitmentTrial'),
                  patch('tools.verify_c_banc_m3_e110_body.IntrinsicRecruitmentTrial')
                    as trial_class,
                  patch('tools.verify_c_banc_m3_e110_body.source_matrix',
                        return_value=weights),
                  patch('tools.verify_c_banc_m3_e110_body.StateStore.save'),
                  patch('tools.verify_c_banc_m3_e110_body.StateStore.load',
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
            with patch('tools.verify_c_banc_m3_e110_body.save_json',
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
        preparation = {'validity': 'valid_prepared_E109_body_diagnostic',
                       'resource_limit_met': True, 'wall_s': 0.}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (patch('tools.verify_c_banc_m3_e110_body.StateStore.load',
                        return_value={'network': {'device': 'cuda'}, 'parameters': {}}),
                  patch('tools.verify_c_banc_m3_e110_body.'
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
        with patch('tools.verify_c_banc_m3_e110_body.contact_diagnostics',
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
