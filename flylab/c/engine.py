"""One worker owns clocks, brain, body, ports, interventions and recording."""
from dataclasses import asdict
import copy
import math
import time
import numpy as np
from . import VERSION, MODES, CONTROL_DT
from .integrity import digest, bounded_int, finite, boolean
from .neural import create_backend, LIFParameters, NEURAL_BACKENDS
from .ports import SensoryEncoder, MotorDecoder, RecoverySupervisor, MotorArbiter, zero_command
from .storage import Recorder, StateStore, runtime_versions
from .inputs import InputRejected, validate_input, validate_stimulus_schedule, validate_schedule
from ..body import FlyGymBody
from ..brain import Circuit
from ..legacy import LegacyBRate
from ..engine import config_values, default_graph, validate_sensor_packet
from ..sensors import SensorAdapter, default_world, validate_world
from .sensors import CSensorAdapter
from .metabolism import Metabolism
from ..common import to_ui
from .workbench import (ENVIRONMENT_COMMANDS, edit_environment, intervention_rows,
                        cancel_intervention, terminal_intervention, restore_workbench)

KINDS = {'stimulate', 'suppress_spiking', 'mute_outgoing', 'mute_edges',
         'sensor_off', 'motor_disconnect', 'assist_off'}


class CEngine:
    def __init__(self, graph, bindings, *, mode='C_SHADOW', seed=42, config=None,
                 world=None, backend='exp_lif_cpu_reference', parameters=None,
                 body_factory=FlyGymBody, motion_expected=True, metabolism=None):
        if mode not in MODES: raise ValueError('Unknown C control mode')
        self.graph, self.bindings = graph, bindings
        if graph.hash != bindings.graph.hash: raise ValueError('Graph/port mismatch')
        self.seed = bounded_int(seed, 'seed', 0, 2**32-1)
        self.mode = mode
        self.motion_expected = boolean(motion_expected, 'motion_expected')
        self.metabolism = Metabolism(metabolism) if metabolism is not None else None
        self.config = config_values(config)
        self.world = copy.deepcopy(validate_world(world if world is not None else default_world()))
        self.parameters = parameters or LIFParameters()
        self.substeps = round(CONTROL_DT / self.parameters.dt)
        self.neural = None if mode == 'B_COMPAT' else create_backend(graph, self.parameters, backend)
        if hasattr(self.neural, 'set_readout_cohort') and len(bindings.motor_indices)<=512:
            self.neural.set_readout_cohort(bindings.motor_indices)
        self.encoder = SensoryEncoder(bindings, seed)
        self.decoder = MotorDecoder(bindings)
        self.supervisor = RecoverySupervisor()
        self.legacy = LegacyBRate(Circuit(default_graph()), seed)
        self.sensors = CSensorAdapter(seed, bindings.spec.get('sensor_model'))
        self.body_factory = body_factory
        self.body = body_factory(seed, self.world, self.config)
        self.control_tick = 0
        self.sensor_tick = 0
        try:
            self.last_sensors = self.sensors.observe(self.body, self.world, CONTROL_DT, self.config)
        except Exception:
            self.body.close()
            raise
        self.subscription = graph.resolve([graph.nodes[int(i)]['id'] for i in bindings.motor_indices])
        self.subscription_epoch = 1
        self.sequence = 0
        self.selected_events = []
        self.signal_start_tick = 0
        self.pending = []
        self.active = []
        self.event_serial = 0
        self.object_serial = 0
        self.environment_history = []
        self.environment_updated_tick = None
        self.intervention_history = []
        self.events = []
        self.recorder = None
        self.record_indices = bindings.motor_indices.copy()
        self.fault = None
        self.closed = False
        self.stopped = False
        self.last_ports = []
        self.last_motor_rates = {}
        self.last_command = MotorArbiter.choose(zero_command(), None, mode=mode,
                                               legacy=zero_command(), motor_coupled=self.config['motorCoupled'])
        self.last_command.update(sensor_tick=0, neural_readout_tick=0, interval_start_tick=0, interval_end_tick=0)
        self.timings = dict(neural_s=0., physics_s=0., ports_s=0., recording_s=0., total_s=0.)
        self.timing_controls = 0

    @property
    def tick(self):
        return self.control_tick * self.substeps

    def provenance(self):
        return dict(app_version=VERSION, graph_hash=self.graph.hash, binding_hash=self.bindings.hash,
                    sensory_binding_hash=self.bindings.sensory_hash, motor_binding_hash=self.bindings.motor_hash,
                    model_parameter_hash=self.parameters.hash, model_parameters=asdict(self.parameters),
                    body_model_hash=self.body.frame()[1]['bodyModelHash'], environment_hash=digest(self.world),
                    config_hash=digest(self.config), mode=self.mode, seed=self.seed,
                    neural_backend=self.neural.backend if self.neural else 'legacy_b_rate',
                    neural_dt=self.parameters.dt if self.neural else CONTROL_DT, physical_dt=.0001,
                    control_dt=CONTROL_DT, coupling_profile='readout-at-start-held-sensor-one-control-latency-v1' if self.neural else 'legacy-B-step-order',
                    motion_expected=self.motion_expected, dataset=self.graph.manifest,
                    bindings=self.bindings.summary(), physical=not self.body.test_double,
                    full_brain=self.graph.full_brain and self.neural is not None,
                    metabolism_model=(dict(parameters=asdict(self.metabolism.p), parameter_hash=digest(asdict(self.metabolism.p))) if self.metabolism else None),
                    biological_validation=False, versions=runtime_versions())

    def event(self, kind, details):
        record = dict(tick=self.tick, kind=kind, details=copy.deepcopy(details))
        self.events.append(record)
        self.events = self.events[-80:]
        if self.recorder:
            try: self.recorder.event(record)
            except Exception as exc:
                self.fault = str(exc)
                raise

    def start_recording(self, path, ids=None, max_bytes=2*1024**3):
        if self.recorder: raise ValueError('Recording already active')
        selected = self.bindings.motor_indices.copy() if ids is None else self.graph.resolve(ids)
        if self.neural and not len(selected): raise ValueError('Choose a nonempty fixed recording cohort')
        names = [self.graph.nodes[int(i)]['id'] for i in selected]
        recorder = Recorder(path, self.provenance(), names, max_bytes=max_bytes)
        previous_cohort = self.record_indices
        try:
            from pathlib import Path
            self.record_indices = selected
            StateStore.save(Path(path)/'initial_checkpoint', self.checkpoint())
        except Exception:
            self.record_indices = previous_cohort
            recorder.close('FAILED', 'Initial checkpoint could not be saved')
            raise
        self.record_indices, self.recorder = selected, recorder
        self.event('recording_start', {'path': str(path), 'cohort_ids': names})

    def stop_recording(self):
        if self.recorder:
            self.recorder.manifest['end_tick'] = self.tick
            fault = self.fault or self.body.fault
            self.recorder.close('FAILED' if fault else 'COMPLETE', fault)
            self.recorder = None

    def subscribe(self, ids):
        self.subscription = self.graph.resolve(ids)
        self.subscription_epoch += 1
        self.selected_events = []
        self.signal_start_tick = self.tick

    def _validate_intervention(self, data):
        if not isinstance(data, dict) or set(data) - {'kind', 'ids', 'edges', 'channels', 'amplitude_mV', 'duration_controls', 'at_tick'}:
            raise ValueError('Unsupported intervention fields')
        kind = data.get('kind')
        if kind not in KINDS: raise ValueError('Unknown intervention kind')
        at = bounded_int(data.get('at_tick', self.tick), 'intervention tick', self.tick)
        if at % self.substeps: raise ValueError('Interventions must be scheduled on a 5ms control boundary')
        duration = bounded_int(data.get('duration_controls', 400), 'duration_controls', 1, 720000)
        item = dict(kind=kind, at_tick=at, expires_tick=at + duration * self.substeps)
        if kind in ('stimulate', 'suppress_spiking', 'mute_outgoing'):
            indices = self.graph.resolve(data.get('ids'))
            if len(indices) == 0: raise ValueError('Intervention targets required')
            item.update(ids=list(data['ids']), indices=indices.tolist())
            if kind == 'stimulate': item['amplitude_mV'] = finite(data.get('amplitude_mV', 12.), 'stimulus mV', -100, 100)
        elif kind == 'mute_edges':
            edges = data.get('edges')
            if not isinstance(edges, list) or not 1 <= len(edges) <= 100000 or len(set(edges)) != len(edges):
                raise ValueError('Unique bounded edge indices required')
            for i in edges: bounded_int(i, 'edge index', 0, len(self.graph.weights)-1)
            item['edges'] = list(edges)
        elif kind == 'sensor_off':
            channels = data.get('channels')
            known = {p['name'] for p, _ in self.bindings.sensory} | {p['channel'] for p, _ in self.bindings.sensory} | {'*'}
            if not isinstance(channels, list) or not channels or any(not isinstance(c, str) or c not in known for c in channels):
                raise ValueError('Known sensory channels required')
            item['channels'] = list(channels)
        return item

    def schedule(self, data):
        if self.mode == 'B_COMPAT': raise ValueError('Use B viewer for legacy interventions')
        if len(self.pending) + len(self.active) >= 20000: raise ValueError('Intervention queue capacity reached')
        item = self._validate_intervention(data)
        validate_stimulus_schedule(item, self.pending + self.active)
        self.event_serial += 1
        item['serial'] = self.event_serial
        self.pending.append(item)
        self.pending.sort(key=lambda i: (i['at_tick'], i['serial']))
        self.event('intervention_scheduled', item)
        return item

    def _prepare_interventions(self):
        expired = [e for e in self.active if e['expires_tick'] <= self.tick]
        if self.pending and self.pending[0]['at_tick'] < self.tick:
            raise RuntimeError('FAULT_INTERVENTION_CLOCK: a scheduled command was not applied')
        applied = [e for e in self.pending if e['at_tick'] == self.tick]
        active = [e for e in self.active if e['expires_tick'] > self.tick] + applied
        suppress, mute, edges, disabled = set(), set(), set(), set()
        current = np.zeros(self.graph.n, dtype=np.float32)
        motor, assist = self.config['motorCoupled'], True
        for e in active:
            kind = e['kind']
            if kind == 'suppress_spiking': suppress.update(e['indices'])
            elif kind == 'mute_outgoing': mute.update(e['indices'])
            elif kind == 'mute_edges': edges.update(e['edges'])
            elif kind == 'sensor_off': disabled.update(e['channels'])
            elif kind == 'motor_disconnect': motor = False
            elif kind == 'assist_off': assist = False
            elif kind == 'stimulate': np.add.at(current, e['indices'], e['amplitude_mV'])
        return dict(active=active, applied=applied, expired=expired, stimulus=current,
                    suppress=sorted(suppress), mute=sorted(mute), edges=sorted(edges),
                    disabled=disabled, coupled=motor, assist_enabled=assist)

    def _commit_interventions(self, prepared):
        self.active = prepared['active']
        self.pending = self.pending[len(prepared['applied']):]
        for event in prepared['expired']:
            terminal_intervention(self, event, 'expired', event['expires_tick'])
            self.event('intervention_expired', event)
        for event in prepared['applied']:
            self.event('intervention_applied', event)
        if self.neural:
            self.neural.set_interventions(prepared['suppress'], prepared['mute'], prepared['edges'])

    def _prepare_control(self, capture):
        prepared = self._prepare_interventions()
        sensor_state = self.sensors.snapshot()
        encoder_state = self.encoder.snapshot() if self.neural else None
        try:
            packet = self.sensors.observe(self.body, self.world, CONTROL_DT, self.config)
            drive = pulses = None
            ports = []
            if self.neural:
                drive, pulses, ports = self.encoder.encode(packet, CONTROL_DT, self.parameters.dt, prepared['disabled'],
                                                           self.sensors.diagnostics.get('features'))
                drive += prepared['stimulus']
                validate_input(self.graph.n, drive, self.substeps, capture, pulses)
        except Exception:
            self.sensors.restore(sensor_state)
            if self.neural: self.encoder.restore(encoder_state)
            raise
        return prepared, packet, drive, pulses, ports

    def step(self, controls=1):
        bounded_int(controls, 'control steps', 0, 200000)
        start_wall = time.perf_counter()
        self.selected_events = []
        self.signal_start_tick = self.tick
        try:
            for _ in range(controls):
                if self.fault or self.body.fault or self.stopped: break
                start = self.tick
                began = time.perf_counter()
                capture = np.unique(np.concatenate([self.subscription, self.record_indices if self.recorder else np.empty(0, dtype=np.int32)])) if self.neural else ()
                prepared, packet, drive, pulses, ports = self._prepare_control(capture)
                self._commit_interventions(prepared)
                coupled, assist_enabled = prepared['coupled'], prepared['assist_enabled']
                self.last_sensors = packet
                self.sensor_tick = start
                legacy_command = self.legacy.step(self.last_sensors, CONTROL_DT, self.config) if self.mode in ('B_COMPAT', 'C_SHADOW') else None
                # Extract at t_k, before integrating either model into the future.
                neural_command, self.last_motor_rates = self.decoder.decode(self.neural) if self.neural else (zero_command(), {})
                assist, reason = (self.supervisor.step(self.last_sensors, CONTROL_DT)
                                  if self.mode == 'C_ASSISTED' and assist_enabled and coupled else (None, None))
                command = MotorArbiter.choose(neural_command, assist, mode=self.mode, legacy=legacy_command,
                                              motor_coupled=coupled, stopped=self.stopped, assist_reason=reason)
                command.update(sensor_tick=start, neural_readout_tick=start,
                               interval_start_tick=start, interval_end_tick=start + self.substeps)
                self.last_command = command
                if self.neural:
                    self.last_ports = ports
                self.timings['ports_s'] += time.perf_counter() - began
                overlap = self.neural is not None and hasattr(self.neural, 'begin_advance')
                if overlap:
                    began = time.perf_counter()
                    self.neural.begin_advance(drive, self.substeps, capture, pulses)
                    self.timings['neural_s'] += time.perf_counter() - began
                began = time.perf_counter()
                try:
                    self.body.step(command['u_final'], CONTROL_DT)
                finally:
                    self.timings['physics_s'] += time.perf_counter() - began
                    if overlap:
                        began = time.perf_counter()
                        self.neural.finish_advance()
                        self.timings['neural_s'] += time.perf_counter() - began
                if self.neural:
                    began = time.perf_counter()
                    if not overlap:self.neural.advance(drive, self.substeps, capture, pulses)
                    subscribed = set(self.subscription.tolist())
                    self.selected_events.extend(e for e in self.neural.last_events if e['index'] in subscribed)
                    self.timings['neural_s'] += time.perf_counter() - began
                self.control_tick += 1
                if self.metabolism:
                    p, R, velocity=self.body.pose()
                    eaten=self.metabolism.step(CONTROL_DT,to_ui(p+R@np.array([.65,0.,-.1])),float(velocity[3:]@R[:,0]),self.world)
                    if eaten:self.event('virtual_food_intake',dict(amount=eaten,energy=self.metabolism.energy))
                self.timing_controls += 1
                if self.body.fault: self.fault = self.body.fault
                began = time.perf_counter()
                if self.recorder:
                    if self.neural:
                        cohort = set(self.record_indices.tolist())
                        spikes = [e for e in self.neural.last_events if e['index'] in cohort]
                        if spikes: self.recorder.event(dict(kind='spikes', tick=self.tick, events=spikes))
                        if self.control_tick % 2 == 0:
                            r = self.neural.readout(self.record_indices)
                            self.recorder.signal(self.tick, r['voltage_mV'], r['rate_Hz'])
                    # Every motor command is scientifically recorded at 200 Hz.
                    self.recorder.event(dict(kind='motor_command', tick=start, details=command))
                    if self.control_tick % 20 == 0:
                        body, physics = self.body.frame()
                        self.recorder.body(dict(tick=self.tick, simTime=self.control_tick*CONTROL_DT,
                                                body=body, physics=physics, sensors=self.last_sensors,
                                                sensor_tick=self.sensor_tick, command=command,
                                                motion_expected=self.motion_expected))
                self.timings['recording_s'] += time.perf_counter() - began
                self._clocks()
                if self.fault:
                    self.stop_recording()
                    break
        except InputRejected:
            # Preparation restores the sensory state; both physical/neural
            # clocks and the accepted intervention journal remain unchanged.
            raise
        except Exception as e:
            self.fault = str(e)
            if self.recorder:
                self.recorder.manifest['end_tick'] = self.tick
                try: self.recorder.close('FAILED', self.fault)
                finally: self.recorder = None
            raise
        finally:
            self.timings['total_s'] += time.perf_counter() - start_wall
        began = time.perf_counter()
        frame = self.frame()
        self.timings['total_s'] += time.perf_counter()-began
        frame['performance'] = self.performance()
        return frame

    def _clocks(self):
        expected = self.control_tick * CONTROL_DT
        if self.neural and self.neural.tick != self.tick: raise RuntimeError('Neural/control clock mismatch')
        body_time = self.body.physics_time() if hasattr(self.body,'physics_time') else self.body.frame()[1]['physicsTime']
        if abs(body_time - expected) > 1e-5: raise RuntimeError('Physics/control clock mismatch')
        if self.neural and self.encoder.control_tick != self.control_tick: raise RuntimeError('Encoder/control clock mismatch')

    def frame(self):
        body, physics = self.body.frame()
        return dict(schema='flylab.frame.v3', tick=self.tick, controlTick=self.control_tick,
                    simTime=self.control_tick * CONTROL_DT, neuralDt=self.parameters.dt, mode=self.mode, seed=self.seed, config=copy.deepcopy(self.config),
                    world=copy.deepcopy(self.world), body=body, physics=physics,
                    sensors=copy.deepcopy(self.last_sensors), sensorTick=self.sensor_tick,
                    sensor_diagnostics=copy.deepcopy(self.sensors.diagnostics),
                    metabolism=self.metabolism.summary() if self.metabolism else dict(enabled=False),
                    command=copy.deepcopy(self.last_command), sensory_ports=copy.deepcopy(self.last_ports),
                    motor_rates_Hz=dict(self.last_motor_rates), neural=self.neural.summary() if self.neural else None,
                    motor_diagnostics=copy.deepcopy(self.decoder.diagnostics),
                    legacy=self.legacy.readout([]) if self.mode in ('B_COMPAT', 'C_SHADOW') else None,
                    scope=dict(fullBrain=self.graph.full_brain and self.neural is not None,
                               sourceNodes=self.graph.n, simulatedNodes=self.graph.n if self.neural else 0,
                               pairEdges=len(self.graph.weights), effectiveEdges=self.graph.manifest['effective_edge_count'],
                               unknownNeurotransmitters=self.graph.manifest['unknown_neurotransmitter_count'],
                               maskedEdges=self.graph.manifest['masked_edge_count'], biologicalValidation=False),
                    binding=self.bindings.summary(), subscription=dict(epoch=self.subscription_epoch,
                        ids=[self.graph.nodes[int(i)]['id'] for i in self.subscription]),
                    events=copy.deepcopy(self.events), interventions=dict(pending=len(self.pending),
                        active=copy.deepcopy(self.active), **intervention_rows(self)),
                    environment=dict(undo_depth=len(self.environment_history), updated_tick=self.environment_updated_tick,
                        sensor_refresh_pending=self.environment_updated_tick is not None and self.tick <= self.environment_updated_tick),
                    fault=self.fault or self.body.fault, stopped=self.stopped,
                    recording=dict(active=self.recorder is not None, dropped=0,
                                   cohort_ids=[self.graph.nodes[int(i)]['id'] for i in self.record_indices],
                                   path=str(self.recorder.path) if self.recorder else None,
                                   signals_hz=100, body_hz=10, motor_hz=200,
                                   bytes=self.recorder.bytes if self.recorder else 0),
                    performance=self.performance())

    def performance(self):
        return dict(**self.timings, measured_model_s=self.timing_controls*CONTROL_DT,
                    neural_overlaps_physics=self.neural is not None and hasattr(self.neural,'begin_advance'),
                    timing_policy='host-stage-wall; neural_s includes submission and remaining wait',
                    sim_wall_ratio=(self.timing_controls*CONTROL_DT/self.timings['total_s']) if self.timings['total_s'] else 0.)

    def region_summary(self):
        if not self.neural: return []
        return self.neural.region_summary(self.graph.groups)

    def command(self, kind, payload):
        if not isinstance(payload, dict): raise ValueError('Command payload object required')
        if kind == 'intervene': return self.schedule(payload)
        if kind == 'release_all':
            for e in self.pending+self.active:
                expired = e['expires_tick'] <= self.tick
                terminal_intervention(self, e, 'expired' if expired else 'cancelled', e['expires_tick'] if expired else self.tick)
            self.pending.clear(); self.active.clear()
            if self.neural: self.neural.set_interventions()
        elif kind == 'cancel_intervention':
            if set(payload) != {'serial'}: raise ValueError('Intervention serial required')
            cancel_intervention(self, payload['serial'])
        elif kind == 'stop': self.stopped = True
        elif kind == 'resume':
            if self.fault or self.body.fault: raise ValueError('Faulted experiment must be restored or restarted')
            self.stopped = False
        elif kind == 'configure':
            new = config_values({**self.config, **payload})
            self.body.set_config(new); self.config = new
        elif kind in ('cue', 'food'):
            value = boolean(payload.get('enabled'), 'enabled')
            self.world['cueOn' if kind == 'cue' else 'foodOn'] = value
            self.body.set_world(self.world)
        elif kind == 'push':
            self.body.perturb(finite(payload.get('bw', .5), 'bw', -2, 2), finite(payload.get('duration', .05), 'duration', .005, .2))
        elif kind in ENVIRONMENT_COMMANDS: edit_environment(self, kind, payload)
        else: raise ValueError('Unknown C command')
        self.event(kind, payload)
        return self.frame()

    def checkpoint(self):
        if self.fault or self.body.fault: raise ValueError('Faulted state is not a restorable checkpoint')
        self._clocks()
        return dict(schema='flylab.checkpoint.v3', app_version=VERSION, versions=runtime_versions(),
                    graph_hash=self.graph.hash, binding_hash=self.bindings.hash,
                    parameters=asdict(self.parameters), parameter_hash=self.parameters.hash,
                    mode=self.mode, seed=self.seed, config=copy.deepcopy(self.config), world=copy.deepcopy(self.world),
                    config_hash=digest(self.config), environment_hash=digest(self.world),
                    body_model_hash=self.body.frame()[1]['bodyModelHash'],
                    control_tick=self.control_tick, sensor_tick=self.sensor_tick,
                    body=self.body.snapshot(), neural=self.neural.snapshot() if self.neural else None,
                    encoder=self.encoder.snapshot(), sensors=self.sensors.snapshot(), legacy=self.legacy.snapshot(),
                    supervisor=self.supervisor.snapshot(), last_sensors=copy.deepcopy(self.last_sensors),
                    last_command=copy.deepcopy(self.last_command), last_ports=copy.deepcopy(self.last_ports),
                    last_motor_rates=dict(self.last_motor_rates), pending=copy.deepcopy(self.pending),
                    active=copy.deepcopy(self.active), event_serial=self.event_serial, object_serial=self.object_serial,
                    metabolism=self.metabolism.snapshot() if self.metabolism else None,
                    environment_history=copy.deepcopy(self.environment_history), environment_updated_tick=self.environment_updated_tick,
                    intervention_history=copy.deepcopy(self.intervention_history),
                    record_cohort_ids=[self.graph.nodes[int(i)]['id'] for i in self.record_indices],
                    motion_expected=self.motion_expected, stopped=self.stopped,
                    subscription=[self.graph.nodes[int(i)]['id'] for i in self.subscription],
                    subscription_epoch=self.subscription_epoch)

    @classmethod
    def from_checkpoint(cls, graph, bindings, state, body_factory=FlyGymBody, *, backend_override=None):
        s = state
        if backend_override is not None and (backend_override not in NEURAL_BACKENDS or not s.get('neural')):
            raise ValueError('A C neural checkpoint and known backend are required for transfer')
        # 0.3 checkpoints have the same baseline equations/coupling/state.
        # Optional new state is validated below. New saves use 0.4 so an older
        # executable cannot silently discard metabolism or sensory history.
        if s.get('schema') != 'flylab.checkpoint.v3' or s.get('app_version') not in ('0.3.0', VERSION) or s.get('versions') != runtime_versions():
            raise ValueError('C version/runtime checkpoint required; B neural state cannot be converted')
        if s.get('graph_hash') != graph.hash or s.get('binding_hash') != bindings.hash:
            raise ValueError('Checkpoint data or binding mismatch')
        if s.get('config_hash') != digest(s.get('config')) or s.get('environment_hash') != digest(s.get('world')):
            raise ValueError('Checkpoint environment/configuration mismatch')
        parameters = LIFParameters(**s['parameters'])
        if parameters.hash != s.get('parameter_hash'): raise ValueError('Parameter hash mismatch')
        ct = bounded_int(s.get('control_tick'), 'control tick', 0, 10**10)
        substeps = round(CONTROL_DT / parameters.dt)
        st = bounded_int(s.get('sensor_tick'), 'sensor tick', 0, ct*substeps)
        if st != max(0, (ct-1)*substeps): raise ValueError('Sensor timestamp mismatch')
        e = cls(graph, bindings, mode=s['mode'], seed=s['seed'], config=s['config'], world=s['world'],
                backend=backend_override or (s['neural']['backend'] if s.get('neural') else 'exp_lif_cpu_reference'),
                parameters=parameters, body_factory=body_factory, motion_expected=s['motion_expected'],
                metabolism=s['metabolism']['parameters'] if s.get('metabolism') else None)
        try:
            if e.body.frame()[1]['bodyModelHash'] != s.get('body_model_hash'): raise ValueError('Body model hash mismatch')
            e.body.restore(s['body'])
            if e.neural:
                neural_state = s['neural']
                if backend_override is not None:
                    neural_state = copy.deepcopy(neural_state)
                    neural_state['backend'] = backend_override
                    neural_state.pop('backend_runtime', None)
                    if hasattr(e.neural, 'runtime_identity'):
                        neural_state['backend_runtime'] = e.neural.runtime_identity
                e.neural.restore(neural_state); e.encoder.restore(s['encoder'])
                if backend_override is not None:
                    transferred = e.neural.snapshot()
                    for key in ('v', 'h', 'rate', 'queue', 'refractory_until', 'spike_count', 'suppress', 'mute'):
                        if not np.array_equal(transferred[key], s['neural'][key]):
                            raise ValueError('Backend transfer changed neural state: '+key)
            elif s.get('neural') is not None: raise ValueError('B_COMPAT cannot restore a C neural state')
            e.sensors.restore(s['sensors']); e.legacy.restore(s['legacy']); e.supervisor.restore(s['supervisor'])
            e.control_tick, e.sensor_tick = ct, st
            if e.metabolism:
                e.metabolism.restore(s['metabolism'])
                if e.metabolism.tick!=ct:raise ValueError('Metabolism/control clock mismatch')
            e.last_sensors = copy.deepcopy(s['last_sensors']); validate_sensor_packet(e.last_sensors)
            if e.sensors.last != e.last_sensors: raise ValueError('Sensor snapshot/packet mismatch')
            e.pending, e.active = copy.deepcopy(s['pending']), copy.deepcopy(s['active'])
            if not isinstance(e.pending, list) or not isinstance(e.active, list) or len(e.pending)+len(e.active) > 20000:
                raise ValueError('Invalid intervention queues')
            serials = set()
            last_order = (-1, -1)
            for group, entries in (('pending', e.pending), ('active', e.active)):
                for item in entries:
                    at = bounded_int(item.get('at_tick'), 'scheduled tick')
                    expires = bounded_int(item.get('expires_tick'), 'expiry tick', at+substeps, at+720000*substeps)
                    if at % substeps or expires % substeps or (group == 'pending' and at < e.tick) or (group == 'active' and not at <= e.tick <= expires):
                        raise ValueError('Intervention timing outside reachable range')
                    serial = bounded_int(item.get('serial'), 'event serial', 1)
                    if serial in serials: raise ValueError('Duplicate intervention serial')
                    serials.add(serial)
                    data = {k: v for k, v in item.items() if k in ('kind', 'ids', 'edges', 'channels', 'amplitude_mV')}
                    data.update(at_tick=max(at, e.tick), duration_controls=(expires-at)//substeps)
                    expected = e._validate_intervention(data)
                    if item.get('indices') != expected.get('indices'): raise ValueError('Intervention ID/index mismatch')
                    if group == 'pending':
                        order = (at, serial)
                        if order < last_order: raise ValueError('Unsorted intervention queue')
                        last_order = order
            validate_schedule(e.pending + e.active, e.tick)
            e.event_serial = bounded_int(s['event_serial'], 'event serial', max(serials, default=0))
            e.object_serial = bounded_int(s['object_serial'], 'object serial', 0, 1000000)
            restore_workbench(e, s)
            e.last_command = copy.deepcopy(s['last_command'])
            for key in ('u_neural', 'u_final', 'u_assist', 'u_legacy'):
                c = e.last_command.get(key)
                if c is not None:
                    if not isinstance(c, dict) or set(c) != {'forwardSpeed', 'yawRate', 'verticalSpeed'}: raise ValueError('Invalid saved motor command')
                    finite(c['forwardSpeed'], 'saved speed', -10, 10); finite(c['yawRate'], 'saved yaw', -5, 5)
                    if c['verticalSpeed'] != 0: raise ValueError('C supports physical walking only')
            expected_start = max(0, e.tick-substeps)
            if e.last_command.get('interval_start_tick') != expected_start or e.last_command.get('interval_end_tick') != e.tick:
                raise ValueError('Saved motor command clock mismatch')
            e.last_ports = copy.deepcopy(s['last_ports']); e.last_motor_rates = copy.deepcopy(s['last_motor_rates'])
            e.stopped = boolean(s['stopped'], 'stopped')
            e.subscribe(s['subscription'])
            e.subscription_epoch = bounded_int(s['subscription_epoch'], 'subscription epoch', 1) + 1
            if backend_override is not None:
                e.event('backend_transfer', dict(source=s['neural']['backend'], target=backend_override,
                                               neural_state_equal=True, model_parameters_changed=False))
            e._clocks()
            return e
        except Exception:
            e.close()
            raise

    def close(self):
        if not self.closed:
            try: self.stop_recording()
            finally: self.body.close(); self.closed = True
