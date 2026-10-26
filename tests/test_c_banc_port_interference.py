"""Model-free checks for E121 identity, observations, and failure receipts."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from scipy.sparse import csr_matrix

from tools import verify_c_banc_m3_e121_port_interference as e121


def toy_projection():
    """382 distinct MN rows, 872 port edges, and deliberately unsorted cal IDs."""
    n = 411
    ports = np.arange(391, 411, dtype=np.int32)
    motors = np.arange(391, dtype=np.int64)
    post = np.r_[np.arange(382), np.arange(872-382) % 382]
    pre = ports[np.arange(872) % 20]
    matrix = csr_matrix((np.ones(872, np.float32), (post, pre)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    # The construction has distinct (post, pre) pairs for these coordinates.
    assert len(matrix.data) == 872
    unscaled = matrix.copy()
    unscaled.data[:] = -np.arange(1, 873, dtype=np.float32) / 8
    scaled = unscaled.copy()
    scaled.data *= np.float32(64)
    ids = np.arange(871, -1, -1, dtype=np.int64)
    cal = {'edge_ids': ids,
           'source_values': scaled.data[ids].copy(),
           'values': scaled.data[ids].copy(),
           'motor_ids': motors, 'individual_edge_ids': np.array([], np.int64),
           'tau_ids': np.array([1, 2], np.int64)}
    graph = SimpleNamespace(n=n, indices=matrix.indices, indptr=matrix.indptr)
    return graph, cal, ports, unscaled, scaled


class PortPatchTests(unittest.TestCase):
    def test_exact_registered_patch_with_unsorted_calibration(self):
        graph, cal, ports, unscaled, scaled = toy_projection()
        changed, patch = e121.derive_port_patch(graph, cal, ports, unscaled, scaled)
        self.assertEqual(len(patch['edge_ids']), 872)
        self.assertEqual(len(np.unique(patch['edge_post'])), 382)
        np.testing.assert_array_equal(patch['calibration_positions'],
                                      np.arange(871, -1, -1))
        np.testing.assert_array_equal(changed['values'], unscaled.data[cal['edge_ids']])
        np.testing.assert_array_equal(changed['source_values'], cal['source_values'])
        np.testing.assert_array_equal(changed['edge_ids'], cal['edge_ids'])
        np.testing.assert_array_equal(changed['tau_ids'], cal['tau_ids'])
        expected = {k: v.copy() for k, v in changed.items()}
        e121.derive_port_patch(graph, cal, ports, unscaled, scaled,
                               expected, patch)
        expected['values'][0] *= 2
        with self.assertRaises(AssertionError):
            e121.derive_port_patch(graph, cal, ports, unscaled, scaled,
                                   expected, patch)

    def test_missing_registration_or_fit_overlap_rejected(self):
        graph, cal, ports, unscaled, scaled = toy_projection()
        cal['edge_ids'] = cal['edge_ids'][1:]
        cal['values'] = cal['values'][1:]
        cal['source_values'] = cal['source_values'][1:]
        with self.assertRaisesRegex(ValueError, 'already be calibrated'):
            e121.derive_port_patch(graph, cal, ports, unscaled, scaled)
        graph, cal, ports, unscaled, scaled = toy_projection()
        cal['individual_edge_ids'] = np.array([0], np.int64)
        with self.assertRaisesRegex(ValueError, 'projection'):
            e121.derive_port_patch(graph, cal, ports, unscaled, scaled)

    def test_exact_float32_and_no_sign_change(self):
        graph, cal, ports, unscaled, scaled = toy_projection()
        scaled.data[0] = abs(scaled.data[0])
        with self.assertRaisesRegex(ValueError, 'weights'):
            e121.derive_port_patch(graph, cal, ports, unscaled, scaled)


class ObservationsAndFailureTests(unittest.TestCase):
    def test_one_current_supplies_details_and_ports(self):
        class Network:
            rate = torch.arange(25, dtype=torch.float32)
            drive = torch.arange(25, dtype=torch.float32) + 2
            adaptation = torch.arange(25, dtype=torch.float32) + 3

            def __init__(self):
                self.calls = 0

            def _current(self, rate):
                self.calls += 1
                return rate * -2

            def readout(self, ids):
                return self.rate[ids].numpy().copy()

        session = SimpleNamespace(network=Network(),
                                  adapter=SimpleNamespace(activation=np.zeros(96)))
        detailed = np.array([1, 4, 20])
        ports = np.arange(20)
        detail, port = e121.observed_signals(session, detailed, ports)
        self.assertEqual(session.network.calls, 1)
        np.testing.assert_array_equal(detail['recurrent_current'], [-2, -8, -40])
        np.testing.assert_array_equal(port['drive'], np.arange(20)+2)

    def test_port_contribution_is_fixed_rate_arithmetic(self):
        rate = np.full((2, 20), 2, dtype=float)
        actual = np.full((391, 20), -1, dtype=float)
        before = actual * 64
        now, fixed_before = e121.port_contribution(rate, actual, before)
        self.assertEqual(now.shape, (2, 391))
        self.assertTrue(np.all(now == -40))
        self.assertTrue(np.all(fixed_before == -2560))
        with self.assertRaisesRegex(ValueError, 'Finite'):
            e121.port_contribution(np.full((2, 20), np.nan), actual, before)

    def test_missing_startup_is_unavailable_and_false(self):
        startup = {'target_command': np.zeros((99, 42)),
                   'target_tendon': np.zeros((99, 6))}
        result = e121.original_startup([{'time_s': 0}], [], startup)
        self.assertFalse(result['complete_original99_prefix'])
        self.assertIsNone(result['original_E100_E103_q']['RMS'])
        self.assertFalse(result['original_E100_E103_startup_screen'])

    def test_partial_native_advance_charges_two_actual_clocks(self):
        class Body:
            time_s = 0.0

            def physics_time(self):
                return self.time_s

        class Session:
            def __init__(self):
                self.body = Body()
                self.network = SimpleNamespace(tick=0)
                self.control_tick = 0

            def advance(self, controls):
                self.network.tick = 20
                self.body.time_s = .0003
                raise e121.NativeHealthStop('third native microstep')

        ledger = e121.AdvanceLedger()
        with self.assertRaises(e121.NativeHealthStop):
            ledger.advance(Session(), 1, 'main')
        self.assertEqual(len(ledger.events), 1)
        self.assertEqual(ledger.events[0]['control_tick_after'], 0)
        self.assertEqual(ledger.events[0]['actual_neural_s'], .005)
        self.assertAlmostEqual(ledger.events[0]['actual_physical_s'], .0003)
        self.assertEqual(ledger.by_owner()['main']['incomplete_or_failed_calls'], 1)

    def test_save_raw_attempts_independent_port_and_clock_files(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            graph = SimpleNamespace(nodes=[{'id': str(i)} for i in range(20)])
            rows = [{'time_s': 0, 'navigation': {'cue_cut': True,
                                                 'turn_request': 0,
                                                 'brake_request': False}}]
            port_rows = [{key: np.zeros(20, np.float32) for key in
                          ('rate', 'recurrent_current', 'drive', 'adaptation')}]
            projection = {'motor_ids': np.arange(391),
                          'actual_W': np.zeros((391, 20)),
                          'before_64x_W': np.zeros((391, 20))}
            true_save = e121.np.savez_compressed

            def fail_port(path, *args, **kwargs):
                if Path(path).name == 'target-port-signals.npz':
                    raise OSError('locked target-port artifact')
                return true_save(path, *args, **kwargs)

            with mock.patch.object(e121.np, 'savez_compressed', side_effect=fail_port):
                errors = e121.save_raw(out, graph, {}, None, rows, None, None,
                    None, None, [], np.arange(20), port_rows, [], [],
                    e121.AdvanceLedger(), {}, projection)
            self.assertEqual([e['artifact'] for e in errors],
                             ['target-port-signals.npz'])
            self.assertTrue((out/'clock-summary.json').exists())
            self.assertTrue((out/'port-MN-contribution.npz').exists())
            self.assertTrue((out/'target-port-identity.json').exists())

    def test_close_preserves_primary_and_attempts_all(self):
        primary = RuntimeError('primary body failure')
        called = []

        def bad():
            called.append('bad')
            raise OSError('close failed')

        def good():
            called.append('good')

        e121.close_owned(primary, ('clone', bad), ('main', good))
        self.assertEqual(called, ['bad', 'good'])
        self.assertIn('close failed', '\n'.join(primary.__notes__))

    def test_main_receipt_failure_keeps_prepare_exception(self):
        primary = RuntimeError('original prepare failure')
        true_save = e121.save_json

        def save(path, value):
            if path.name == 'execution-error.json':
                raise OSError('receipt locked')
            return true_save(path, value)

        with tempfile.TemporaryDirectory() as name:
            out = Path(name)/'new-result'
            root = e121.ROOT/'verification/banc-natural-gait-m3-20260919'
            with (mock.patch.object(e121, 'prepare', side_effect=primary),
                  mock.patch.object(e121, 'save_json', side_effect=save),
                  self.assertRaises(RuntimeError) as caught):
                e121.main(root, Path('unused-graph'), out)
            self.assertIs(caught.exception, primary)
            self.assertIn('receipt locked', '\n'.join(primary.__notes__))
            self.assertEqual(json.loads((out/'spec.json').read_text())['experiment'],
                             e121.EXPERIMENT)


def fake_body_probe(*, secondary=False, clone_late=False, clone_mid=False):
    """Exercise the production coordinator with CPU fakes and no model creation."""
    ports = np.arange(446, 466, dtype=np.int32)
    primary = RuntimeError('port transfer after complete control')
    timer = [0.0]
    calls, closed, saved = [], [], {}
    neutral = {'parameters': {}, 'network': {'device': 'cuda'}, 'control_tick': 0}

    class Network:
        device, identity = 'cuda', 'fake network'

        def __init__(self):
            self.tick = self.current_calls = self.port_reads = 0
            for key in ('rate', 'drive', 'adaptation', 'tau', 'a', 'threshold',
                        'cap', '_cell_gain'):
                setattr(self, key, torch.zeros(466, dtype=torch.float32))

        def _current(self, rate):
            self.current_calls += 1
            return rate-2

        def readout(self, ids):
            if np.array_equal(ids, ports):
                self.port_reads += 1
                if self.port_reads == 2 and not (clone_late or clone_mid):
                    raise primary
            return self.rate[ids].numpy().copy()

    class Session:
        fault = None

        def __init__(self, state, owner):
            self.owner = owner
            self.control_tick = state['control_tick']
            self.network = Network()
            self.network.tick = self.control_tick*20
            self.body = SimpleNamespace(fault=None, model_hash='fake body',
                physics_time=lambda: self.control_tick*.005)
            self.adapter = SimpleNamespace(identity='fake adapter', activation=np.zeros(96))
            self.navigation = SimpleNamespace(encoder=SimpleNamespace(ports=ports))
            self.native = None

        def snapshot(self):
            return {**neutral, 'control_tick': self.control_tick}

        def advance(self, controls):
            calls.append((self.owner, controls, timer[0]))
            before = self.control_tick
            self.control_tick += controls
            self.network.tick = self.control_tick*20
            self.network.rate[:] = self.control_tick
            for i in range(before*50+1, self.control_tick*50+1):
                self.native.append((i*.0001, np.zeros(6), np.full(6, np.nan),
                                    np.zeros(6, bool), False, 1.))
            if (clone_mid and self.owner == 'clone'
                    and sum(owner == 'clone' for owner, _, _ in calls) == 5):
                timer[0] = 143.0

        def close(self):
            closed.append(self.owner)

    instances = []

    def construct(graph, state):
        owner = 'main' if not instances else 'clone'
        result = Session(state, owner)
        instances.append(result)
        if clone_late and owner == 'clone':
            timer[0] = 143.0
        return result

    def capture(session):
        return {'time_s': session.control_tick*.005, 'fault': None,
                'navigation': {'cue_cut': True, 'turn_request': 0,
                               'brake_request': False},
                'joint_targets_rad': np.zeros(42).tolist(),
                'tendon_inputs': {leg+'_tarsus': 0. for leg in e121.LEGS}}

    def install(body, native):
        next(s for s in instances if s.body is body).native = native
        return lambda: None

    graph = SimpleNamespace(hash='fake', nodes=[{'id': str(i)} for i in range(466)])
    projection = {'port_ids': ports, 'motor_ids': np.arange(391),
                  'actual_W': np.zeros((391, 20), np.float32),
                  'before_64x_W': np.zeros((391, 20), np.float32)}
    load_npz = e121.load_npz

    def fake_npz(path):
        return projection if Path(path).name == 'port-projection.npz' else load_npz(path)

    def fake_load(path):
        return copy.deepcopy(saved.get(Path(path).name, neutral))

    def fake_save(path, state):
        saved[Path(path).name] = copy.deepcopy(state)

    with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
        out = Path(name)
        for target, field, replacement in (
            (e121.IntrinsicRecruitmentTrial, 'from_checkpoint', construct),
            (e121.StateStore, 'load', fake_load),
            (e121.StateStore, 'save', fake_save),
            (e121, 'load_npz', fake_npz),
            (e121, 'source_matrix', lambda _: csr_matrix((466, 466), dtype=np.float32)),
            (e121, 'observation_ids', lambda *a: (
                np.arange(442), np.arange(446), np.arange(55), np.array([440, 441]))),
            (e121, 'capture_control', capture),
            (e121, 'install_native_observer', install),
            (e121, 'source_diagnostics', lambda *a: (
                {'source_screen': False}, {'source_preparation_screen': False})),
            (e121.time, 'perf_counter', lambda: timer[0])):
            stack.enter_context(mock.patch.object(target, field, replacement))
        if secondary:
            stack.enter_context(mock.patch.object(e121, 'save_startup_raw',
                side_effect=OSError('secondary startup save failed')))
        caught = None
        try:
            e121.run_body(e121.ROOT/'verification/banc-natural-gait-m3-20260919',
                out, graph, {'motor_ids': np.arange(391)}, {},
                {'target_command': np.zeros((99, 42)),
                 'target_tendon': np.zeros((99, 6))}, {},
                {'wall_s': 0., 'resource_limit_met': True}, {}, 0.)
        except Exception as error:  # noqa: BLE001 - inspect production failure identity
            caught = error
        trace = json.loads((out/'trace.json').read_text(encoding='utf-8'))
        detail = e121.load_npz(out/'recruitment-signals.npz')
        clock = json.loads((out/'clock-summary.json').read_text(encoding='utf-8'))
        runtime_path = out/'runtime-error.json'
        runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if (
            runtime_path.exists()) else None
        partial_path = out/'partial-observation.json'
        partial = json.loads(partial_path.read_text(encoding='utf-8')) if (
            partial_path.exists()) else None
        return SimpleNamespace(caught=caught, primary=primary, calls=calls,
            closed=closed, trace=trace, detail=detail, clock=clock,
            runtime=runtime, partial=partial,
            currents=sum(s.network.current_calls for s in instances))


class ProductionFailureTests(unittest.TestCase):
    def test_post_control_port_error_preserves_capture_and_detail(self):
        result = fake_body_probe()
        self.assertIs(result.caught, result.primary)
        self.assertEqual(len(result.trace['samples']), 2)
        self.assertEqual(len(result.detail['rate']), 1)
        self.assertEqual(result.clock['main_completed_controls'], 1)
        self.assertEqual(result.clock['actual_neural_s'], .005)
        self.assertEqual(result.clock['actual_physical_s'], .005)
        self.assertEqual(result.partial[0]['control_tick'], 1)
        self.assertTrue(result.partial[0]['detail_complete'])
        self.assertFalse(result.partial[0]['port_complete'])
        self.assertEqual(result.currents, 2)
        self.assertEqual(result.closed, ['main'])

    def test_secondary_startup_save_keeps_original_error_and_notes(self):
        result = fake_body_probe(secondary=True)
        self.assertIs(result.caught, result.primary)
        self.assertIn('secondary startup', '\n'.join(result.primary.__notes__))
        self.assertIn('secondary startup', '\n'.join(result.runtime['notes']))
        self.assertEqual(len(result.trace['samples']), 2)

    def test_late_clone_construction_starts_zero_clone_controls(self):
        result = fake_body_probe(clone_late=True)
        self.assertEqual([call for call in result.calls if call[0] == 'clone'], [])
        self.assertAlmostEqual(result.clock['actual_neural_s'], 1.0)
        self.assertAlmostEqual(result.clock['actual_physical_s'], 1.0)
        self.assertEqual(result.clock['clone_native_samples'], 0)
        self.assertEqual(result.closed, ['clone', 'main'])

    def test_clone_stops_between_controls_when_reserve_expires(self):
        result = fake_body_probe(clone_mid=True)
        self.assertEqual(len([call for call in result.calls if call[0] == 'clone']), 5)
        self.assertAlmostEqual(result.clock['actual_neural_s'], 1.025)
        self.assertAlmostEqual(result.clock['actual_physical_s'], 1.025)
        self.assertEqual(result.clock['clone_native_samples'], 250)
        self.assertEqual(result.closed, ['clone', 'main'])

    def test_source_diagnostic_assets_are_hash_guarded(self):
        real_sha = e121.sha256

        def changed(path):
            if Path(path).parts[-2:] == (
                    'e85-space-compatible-cadence-reference', 'commands.npz'):
                return 'bad'
            return real_sha(path)

        root = e121.ROOT/'verification/banc-natural-gait-m3-20260919'
        with (mock.patch.object(e121, 'sha256', side_effect=changed),
              self.assertRaisesRegex(ValueError, 'source diagnostic input changed')):
            e121.reviewed_inputs(root)


if __name__ == '__main__':
    unittest.main()
