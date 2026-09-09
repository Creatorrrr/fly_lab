import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flylab.c.graph import GraphStore, external_id
from flylab.c.integrity import write_json
from flylab.c.neural import ExpLIF, LIFParameters
from flylab.c.ports import PortBindings, SensoryEncoder, MotorDecoder, RecoverySupervisor, MotorArbiter, zero_command
from flylab.c.engine import CEngine
from flylab.c.storage import StateStore
from flylab.c.protocol import signal_frame, decode_signals
from flylab.engine import Engine as BEngine
from tests.fixture_body import FixtureBody
from tests.c_fixtures import graph_fixture, bindings_fixture


def same_state(a, b):
    if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a: same_state(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): same_state(x, y)
    else: assert a == b, (a, b)


class GraphTests(unittest.TestCase):
    def test_direction_counts_and_inhibition(self):
        g = graph_fixture()
        self.assertAlmostEqual(g.matrix[1, 0], 1.1, places=6)
        self.assertLess(g.matrix[1, 2], 0)
        self.assertEqual(g.matrix[0, 1], 0)
        self.assertEqual(g.manifest['anatomical_synapse_count'], 11)

    def test_isolated_nodes_survive_aggregation(self):
        g = graph_fixture(pre=(0, 0), post=(1, 1), counts=(3, 7))
        self.assertEqual(g.n, 6)
        self.assertEqual(len(g.weights), 1)
        self.assertEqual(g.counts[0], 10)
        self.assertFalse(g.full_brain)

    def test_ids_cannot_pass_through_float_or_number(self):
        for value in (720575940604737708, 7.205759e17, '1e18', '-1', '1.0'):
            with self.assertRaises(ValueError): external_id(value)
        self.assertEqual(external_id('720575940604737708'), 'flywire:fafb:783:720575940604737708')

    def test_unknown_transmitter_requires_explicit_policy(self):
        g = graph_fixture(); nodes = copy.deepcopy(g.nodes); nodes[0]['nt_type'] = ''
        with self.assertRaisesRegex(ValueError, 'BLOCKED_NEUROTRANSMITTER'):
            GraphStore.from_edges(nodes, [0], [1], [5])
        masked = GraphStore.from_edges(nodes, [0], [1], [5], unknown_policy='mask_zero')
        self.assertEqual(masked.manifest['masked_edge_count'], 1)
        self.assertEqual(masked.manifest['anatomical_synapse_count'], 5)

    def test_glutamate_is_inhibitory_and_modulators_explicit(self):
        nodes = graph_fixture().nodes
        nodes = copy.deepcopy(nodes); nodes[0]['nt_type'] = 'GLUT'; nodes[1]['nt_type'] = 'DA'
        g = GraphStore.from_edges(nodes, [0, 1], [1, 2], [5, 5])
        self.assertLess(g.matrix[1, 0], 0); self.assertEqual(g.matrix[2, 1], 0)
        self.assertEqual(g.manifest['modulatory_neuron_count'], 1)

    def test_roundtrip_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bundle'; g = graph_fixture(); g.save(path)
            loaded = GraphStore.load(path)
            self.assertEqual(g.hash, loaded.hash)
            np.testing.assert_array_equal(g.matrix.toarray(), loaded.matrix.toarray())
            with self.assertRaises(FileExistsError): g.save(path)
            with (path/'weights.npy').open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): GraphStore.load(path)


class LIFTests(unittest.TestCase):
    def setUp(self):
        self.g = graph_fixture(pre=(), post=(), counts=())

    def test_exact_constant_drive_against_analytic_solution(self):
        p = LIFParameters(dtype='float64'); b = ExpLIF(self.g, p)
        b.advance(np.full(6, 3.), 100)
        expected = -52 + 3*(1-math.exp(-.01/.02))
        np.testing.assert_allclose(b.v, expected, rtol=0, atol=1e-11)
        self.assertEqual(b.spike_count.sum(), 0)

    def test_exact_synaptic_decay_and_membrane_response(self):
        b = ExpLIF(self.g, LIFParameters(dtype='float64')); b.h[0] = 2
        b.advance(np.zeros(6), 50)
        self.assertAlmostEqual(b.h[0], 2*math.exp(-.005/.005), places=12)
        expected = -52 + 2*.005/(.02-.005)*(math.exp(-.005/.02)-math.exp(-.005/.005))
        self.assertAlmostEqual(b.v[0], expected, places=11)

    def test_equal_time_constants_has_finite_exact_limit(self):
        b = ExpLIF(self.g, LIFParameters(tau_syn_s=.02, dtype='float64')); b.h[0] = 2
        b.advance(np.zeros(6), 50)
        self.assertAlmostEqual(b.v[0], -52 + 2*.005/.02*math.exp(-.005/.02), places=11)

    def test_spike_reset_and_refractory(self):
        b = ExpLIF(self.g); drive = np.zeros(6); drive[0] = 100
        b.advance(drive, 100, [0])
        ticks = [e['tick'] for e in b.last_events]
        self.assertGreater(len(ticks), 1)
        self.assertTrue(all(y-x > b.refractory_ticks for x, y in zip(ticks, ticks[1:])))
        self.assertLess(b.v[0], b.p.threshold_mV)

    def test_delayed_arrival_and_orientation(self):
        g = graph_fixture(pre=(0,), post=(1,), counts=(20,))
        b = ExpLIF(g, LIFParameters(dtype='float64')); b.v[0] = -44.
        b.advance(np.zeros(6), 1)
        self.assertEqual(b.spike_count[0], 1); self.assertEqual(b.h[1], 0)
        b.advance(np.zeros(6), b.delay_ticks)
        self.assertEqual(b.h[1], 0)
        b.advance(np.zeros(6), 1)
        self.assertGreater(b.h[1], 0); self.assertEqual(b.h[0], 0)

    def test_inflight_event_survives_mute_and_suppression(self):
        g = graph_fixture(pre=(0,), post=(1,), counts=(20,))
        b = ExpLIF(g); b.v[0] = -44.; b.advance(np.zeros(6), 1)
        b.set_interventions(suppress=[0], mute=[0], edges=[0])
        b.advance(np.zeros(6), b.delay_ticks+1)
        self.assertGreater(b.h[1], 0)

    def test_mute_outgoing_preserves_spikes_but_stops_future_propagation(self):
        g = graph_fixture(pre=(0,), post=(1,), counts=(20,)); b = ExpLIF(g)
        b.set_interventions(mute=[0]); drive = np.zeros(6); drive[0] = 100
        b.advance(drive, 100)
        self.assertGreater(b.spike_count[0], 0); self.assertEqual(b.h[1], 0)

    def test_suppression_preserves_existing_rate_filter(self):
        b = ExpLIF(self.g); b.rate[0] = 100; b.set_interventions(suppress=[0])
        b.advance(np.full(6, 100.), 50)
        self.assertEqual(b.spike_count[0], 0)
        self.assertGreater(b.rate[0], 0); self.assertLess(b.rate[0], 100)

    def test_substep_convergence(self):
        a = ExpLIF(self.g, LIFParameters(dtype='float64'))
        b = ExpLIF(self.g, LIFParameters(dt=.00005, dtype='float64'))
        a.h[0] = b.h[0] = 1
        a.advance(np.full(6, 2.), 100); b.advance(np.full(6, 2.), 200)
        np.testing.assert_allclose(a.v, b.v, atol=1e-11, rtol=0)

    def test_snapshot_restores_queue_and_continues_exactly(self):
        g = graph_fixture(); a = ExpLIF(g); drive = np.zeros(6); drive[0] = 100
        a.advance(drive, 17); state = a.snapshot(); b = ExpLIF(g); b.restore(state)
        a.advance(drive, 70); b.advance(drive, 70)
        same_state(a.snapshot(), b.snapshot())

    def test_restore_bounds_are_atomic(self):
        b = ExpLIF(self.g); before = b.snapshot()
        for key in ('rate', 'queue', 'refractory_until'):
            invalid = copy.deepcopy(before)
            invalid[key].flat[0] = -1 if key == 'rate' else 10**8
            with self.assertRaises(ValueError): b.restore(invalid)
            same_state(before, b.snapshot())

    def test_cuda_is_explicitly_blocked_without_device(self):
        import importlib.util
        if importlib.util.find_spec('cupy'): self.skipTest('CUDA checked by tools/verify_c.py')
        with self.assertRaisesRegex(RuntimeError, 'BLOCKED_CUDA'): ExpLIF(self.g, backend='exp_lif_cuda')


class PortTests(unittest.TestCase):
    def setUp(self):
        self.g = graph_fixture(); self.b = bindings_fixture(self.g)

    def test_unknown_id_blocks_instead_of_fallback(self):
        spec = copy.deepcopy(self.b.spec); spec['sensory'][0]['ids'] = [external_id('999')]
        with self.assertRaisesRegex(ValueError, 'BLOCKED_PORT_BINDING'): PortBindings(self.g, spec)

    def test_missing_forward_port_blocks(self):
        spec = copy.deepcopy(self.b.spec); spec['motor']['forward']['ids'] = []
        with self.assertRaisesRegex(ValueError, 'BLOCKED_PORT_BINDING'): PortBindings(self.g, spec)

    def test_truth_channel_and_side_inference_rejected(self):
        for field, value in [('channel', 'yaw'), ('review_status', 'pending')]:
            spec = copy.deepcopy(self.b.spec); spec['sensory'][0][field] = value
            with self.assertRaises(ValueError): PortBindings(self.g, spec)
        spec = copy.deepcopy(self.b.spec); del spec['motor']['yaw_left']['output_side']
        with self.assertRaises(ValueError): PortBindings(self.g, spec)

    def test_decoder_uses_only_rates_with_signed_direction(self):
        n = ExpLIF(self.g); d = MotorDecoder(self.b)
        self.assertEqual(d.decode(n)[0], zero_command())
        n.rate[1] = 20; n.rate[5] = 10
        c, _ = d.decode(n)
        self.assertGreater(c['forwardSpeed'], 0); self.assertGreater(c['yawRate'], 0)
        n.rate[1] = 0; n.rate[2] = 20; n.rate[5] = 0; n.rate[4] = 10
        c, _ = d.decode(n)
        self.assertLess(c['forwardSpeed'], 0); self.assertLess(c['yawRate'], 0)

    def test_strict_and_shadow_command_source(self):
        n = dict(forwardSpeed=1., yawRate=.1, verticalSpeed=0.); assist = dict(forwardSpeed=-1., yawRate=2., verticalSpeed=0.)
        self.assertEqual(MotorArbiter.choose(n, assist, mode='C_STRICT')['u_final'], n)
        self.assertEqual(MotorArbiter.choose(n, assist, mode='C_ASSISTED')['u_final'], assist)
        self.assertEqual(MotorArbiter.choose(n, assist, mode='C_SHADOW', legacy=zero_command())['command_source'], 'legacy_b_rate')
        self.assertEqual(MotorArbiter.choose(n, assist, mode='C_ASSISTED', motor_coupled=False)['u_final'], zero_command())


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.g = graph_fixture(); self.bindings = bindings_fixture(self.g); self.engines = []

    def engine(self, **kwargs):
        e = CEngine(self.g, self.bindings, body_factory=FixtureBody, mode=kwargs.pop('mode', 'C_STRICT'), **kwargs)
        self.engines.append(e); return e

    def tearDown(self):
        for e in self.engines: e.close()

    def test_b_compat_and_shadow_match_actual_b_model(self):
        original = BEngine(body_factory=FixtureBody)
        try:
            compatible = self.engine(mode='B_COMPAT'); shadow = self.engine(mode='C_SHADOW')
            original.step(80); compatible.step(80); shadow.step(80)
            same_state(original.body.snapshot(), compatible.body.snapshot())
            same_state(original.body.snapshot(), shadow.body.snapshot())
            same_state(original.brain.snapshot(), compatible.legacy.snapshot())
            self.assertEqual(shadow.neural.tick, 4000)
        finally: original.close()

    def test_causal_readout_and_no_tonic_movement(self):
        e = self.engine(); e.step(1)
        self.assertEqual(e.last_command['u_final'], zero_command())
        e.schedule(dict(kind='stimulate', ids=[self.g.nodes[1]['id']], amplitude_mV=100., duration_controls=10))
        start = e.tick; e.step(1)
        self.assertEqual(e.last_command['u_neural'], zero_command())
        self.assertEqual(e.last_command['neural_readout_tick'], start)
        e.step(1)
        self.assertGreater(e.last_command['u_neural']['forwardSpeed'], 0)
        self.assertEqual(e.neural.tick, e.tick)

    def test_schedule_applies_at_exact_control_boundary(self):
        e = self.engine(); e.schedule(dict(kind='stimulate', ids=[self.g.nodes[1]['id']], amplitude_mV=100., duration_controls=1, at_tick=100))
        e.step(2); self.assertEqual(e.neural.spike_count.sum(), 0)
        e.step(1); self.assertGreater(e.neural.spike_count[1], 0)
        e.step(1); self.assertEqual(len(e.active), 0)
        self.assertTrue(any(x['kind'] == 'intervention_applied' and x['tick'] == 100 for x in e.events))

    def test_invalid_intervention_does_not_mutate_queue(self):
        e = self.engine(); before = e.checkpoint()
        with self.assertRaises(ValueError): e.schedule(dict(kind='stimulate', ids=[self.g.nodes[1]['id']], at_tick=1))
        same_state(before, e.checkpoint())

    def test_subscription_and_recording_do_not_change_dynamics(self):
        a = self.engine(); b = self.engine()
        a.subscribe([]); b.subscribe([self.g.nodes[1]['id']])
        for e in (a, b): e.schedule(dict(kind='stimulate', ids=[self.g.nodes[1]['id']], amplitude_mV=30.))
        with tempfile.TemporaryDirectory() as tmp:
            b.start_recording(Path(tmp)/'run')
            a.step(40); b.step(40); b.stop_recording()
            same_state(a.neural.snapshot(), b.neural.snapshot())
            same_state(a.body.snapshot(), b.body.snapshot())
            manifest = json.loads((Path(tmp)/'run/manifest.json').read_text())
            self.assertEqual(manifest['dropped_records'], 0)
            self.assertEqual(sum(x['rows'] for x in manifest['chunks']), 20)

    def test_checkpoint_saved_arrays_pending_events_and_continuation(self):
        a = self.engine(); a.schedule(dict(kind='stimulate', ids=[self.g.nodes[1]['id']], amplitude_mV=50., at_tick=200)); a.step(3)
        with tempfile.TemporaryDirectory() as tmp:
            StateStore.save(Path(tmp)/'cp', a.checkpoint())
            b = CEngine.from_checkpoint(self.g, self.bindings, StateStore.load(Path(tmp)/'cp'), FixtureBody)
            self.engines.append(b)
            a.step(20); b.step(20)
            same_state(a.neural.snapshot(), b.neural.snapshot())
            same_state(a.body.snapshot(), b.body.snapshot())
            same_state(a.encoder.snapshot(), b.encoder.snapshot())

    def test_restore_rejects_bad_state_and_preserves_live_instance(self):
        a = self.engine(); a.step(2); before = a.checkpoint(); bad = copy.deepcopy(before)
        bad['neural']['refractory_until'][0] = 10**9
        with self.assertRaises(ValueError): CEngine.from_checkpoint(self.g, self.bindings, bad, FixtureBody)
        same_state(before, a.checkpoint())

    def test_environment_mutation_preserves_neural_state(self):
        e = self.engine(); e.step(2); before = e.neural.snapshot()
        e.command('place', dict(kind='obstacle', position=[10, .8, 0], radius=.8))
        same_state(before, e.neural.snapshot())

    def test_sensor_packet_rejects_ground_truth(self):
        e = self.engine(); p = copy.deepcopy(e.last_sensors); p['yaw'] = 0
        with self.assertRaises(ValueError): e.encoder.encode(p, .005, .0001)

    def test_poisson_rng_restore_and_sensor_off_draw_count(self):
        e = self.engine(); spec = bindings_fixture(self.g, 'poisson_Hz').spec
        spec['sensory'][0].update(baseline=50., gain=0., delay_controls=2, tau_s=.02)
        bindings = PortBindings(self.g, spec); a = SensoryEncoder(bindings, 42); b = SensoryEncoder(bindings, 42)
        for _ in range(5):
            a.encode(e.last_sensors, .005, .0001)
            b.encode(e.last_sensors, .005, .0001, ['*'])
        same_state(a.snapshot(), b.snapshot())
        c = SensoryEncoder(bindings, 1); c.restore(a.snapshot())
        x = a.encode(e.last_sensors, .005, .0001); y = c.encode(e.last_sensors, .005, .0001)
        np.testing.assert_array_equal(x[1][1], y[1][1])

    def test_binary_units_epoch_and_string_ids(self):
        e = self.engine(); e.subscribe([self.g.nodes[1]['id']]); e.step(2)
        h, voltage, rate, spikes = decode_signals(signal_frame(e))
        self.assertEqual(h['unit_descriptor'], ['mV', 'Hz'])
        self.assertEqual(h['ids'], [self.g.nodes[1]['id']])
        self.assertLess(voltage[0], 0)
        old = h['subscription_epoch']; e.subscribe([])
        new = decode_signals(signal_frame(e))[0]
        self.assertGreater(new['subscription_epoch'], old); self.assertEqual(new['channel_count'], 0)
        with self.assertRaises(ValueError): decode_signals(signal_frame(e)[:-1])

    def test_recording_capacity_fault_is_not_silently_dropped(self):
        e = self.engine()
        with tempfile.TemporaryDirectory() as tmp:
            e.start_recording(Path(tmp)/'run', max_bytes=1000)
            with self.assertRaisesRegex(RuntimeError, 'FAULT_RECORDING_CAPACITY'): e.step(5)
            self.assertIsNotNone(e.fault)
            with self.assertRaises(ValueError): e.command('resume', {})


if __name__ == '__main__': unittest.main()
