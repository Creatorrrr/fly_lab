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
from ..body import FlyGymBody
from ..brain import Circuit
from ..legacy import LegacyBRate
from ..engine import config_values, default_graph, validate_sensor_packet, Engine as BEngine
from ..sensors import SensorAdapter, default_world, validate_world

KINDS = {'stimulate', 'suppress_spiking', 'mute_outgoing', 'mute_edges',
         'sensor_off', 'motor_disconnect', 'assist_off'}


class CEngine:
    def __init__(self, graph, bindings, *, mode='C_SHADOW', seed=42, config=None,
                 world=None, backend='exp_lif_cpu_reference', parameters=None,
                 body_factory=FlyGymBody, motion_expected=True):
        if mode not in MODES: raise ValueError('Unknown C control mode')
        self.graph, self.bindings = graph, bindings
        if graph.hash != bindings.graph.hash: raise ValueError('Graph/port mismatch')
        self.seed = bounded_int(seed, 'seed', 0, 2**32-1)
        self.mode = mode
        self.motion_expected = boolean(motion_expected, 'motion_expected')
        self.config = config_values(config)
        self.world = copy.deepcopy(validate_world(world if world is not None else default_world()))
        self.parameters = parameters or LIFParameters()
        self.substeps = round(CONTROL_DT / self.parameters.dt)
        self.neural = None if mode == 'B_COMPAT' else create_backend(graph, self.parameters, backend)
        self.encoder = SensoryEncoder(bindings, seed)
        self.decoder = MotorDecoder(bindings)
        self.supervisor = RecoverySupervisor()
        self.legacy = LegacyBRate(Circuit(default_graph()), seed)
        self.sensors = SensorAdapter(seed)
        self.body_factory = body_factory
        self.body = body_factory(seed, self.world, self.config)
        self.control_tick = 0
        self.sensor_tick = 0
        self.last_sensors = self.sensors.observe(self.body, self.world, CONTROL_DT, self.config)
        self.subscription = graph.resolve([graph.nodes[int(i)]['id'] for i in bindings.motor_indices])
        self.subscription_epoch = 1
        self.sequence = 0
        self.selected_events = []
        self.signal_start_tick = 0
        self.pending = []
        self.active = []
        self.event_serial = 0
        self.object_serial = 0
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
        names = [self.graph.nodes[int(i)]['id'] for i in selected]
        recorder = Recorder(path, self.provenance(), names, max_bytes=max_bytes)
        try:
            from pathlib import Path
            StateStore.save(Path(path)/'initial_checkpoint', self.checkpoint())
        except Exception:
            recorder.close('FAILED', 'Initial checkpoint could not be saved')
            raise
        self.record_indices, self.recorder = selected, recorder
        self.event('recording_start', {'path': str(path), 'cohort_ids': names})

    def stop_recording(self):
        if self.recorder:
            self.recorder.manifest['end_tick'] = self.tick
            self.recorder.close('FAILED' if self.fault else 'COMPLETE', self.fault)
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
        self.event_serial += 1
        item['serial'] = self.event_serial
        self.pending.append(item)
        self.pending.sort(key=lambda i: (i['at_tick'], i['serial']))
        self.event('intervention_scheduled', item)
        return item

    def _interventions(self):
        expired = [e for e in self.active if e['expires_tick'] <= self.tick]
        self.active = [e for e in self.active if e['expires_tick'] > self.tick]
        for e in expired: self.event('intervention_expired', e)
        while self.pending and self.pending[0]['at_tick'] == self.tick:
            item = self.pending.pop(0)
            self.active.append(item)
            self.event('intervention_applied', item)
        if self.pending and self.pending[0]['at_tick'] < self.tick:
            raise RuntimeError('FAULT_INTERVENTION_CLOCK: a scheduled command was not applied')
        suppress, mute, edges, disabled = set(), set(), set(), set()
        current = np.zeros(self.graph.n, dtype=np.float32)
        motor, assist = self.config['motorCoupled'], True
        for e in self.active:
            kind = e['kind']
            if kind == 'suppress_spiking': suppress.update(e['indices'])
            elif kind == 'mute_outgoing': mute.update(e['indices'])
            elif kind == 'mute_edges': edges.update(e['edges'])
            elif kind == 'sensor_off': disabled.update(e['channels'])
            elif kind == 'motor_disconnect': motor = False
            elif kind == 'assist_off': assist = False
            elif kind == 'stimulate': np.add.at(current, e['indices'], e['amplitude_mV'])
        if self.neural: self.neural.set_interventions(sorted(suppress), sorted(mute), sorted(edges))
        return current, disabled, motor, assist

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
                stimulus, disabled, coupled, assist_enabled = self._interventions()
                self.last_sensors = self.sensors.observe(self.body, self.world, CONTROL_DT, self.config)
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
                    drive, pulses, self.last_ports = self.encoder.encode(self.last_sensors, CONTROL_DT, self.parameters.dt, disabled)
                    drive += stimulus
                self.timings['ports_s'] += time.perf_counter() - began
                began = time.perf_counter()
                self.body.step(command['u_final'], CONTROL_DT)
                self.timings['physics_s'] += time.perf_counter() - began
                began = time.perf_counter()
                if self.neural:
                    capture = np.unique(np.concatenate([self.subscription, self.record_indices if self.recorder else np.empty(0, dtype=np.int32)]))
                    self.neural.advance(drive, self.substeps, capture, pulses)
                    subscribed = set(self.subscription.tolist())
                    self.selected_events.extend(e for e in self.neural.last_events if e['index'] in subscribed)
                self.timings['neural_s'] += time.perf_counter() - began
                self.control_tick += 1
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
        except Exception as e:
            self.fault = str(e)
            if self.recorder:
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
        if abs(self.body.frame()[1]['physicsTime'] - expected) > 1e-5: raise RuntimeError('Physics/control clock mismatch')
        if self.neural and self.encoder.control_tick != self.control_tick: raise RuntimeError('Encoder/control clock mismatch')

    def frame(self):
        body, physics = self.body.frame()
        return dict(schema='flylab.frame.v3', tick=self.tick, controlTick=self.control_tick,
                    simTime=self.control_tick * CONTROL_DT, neuralDt=self.parameters.dt, mode=self.mode, seed=self.seed, config=copy.deepcopy(self.config),
                    world=copy.deepcopy(self.world), body=body, physics=physics,
                    sensors=copy.deepcopy(self.last_sensors), sensorTick=self.sensor_tick,
                    command=copy.deepcopy(self.last_command), sensory_ports=copy.deepcopy(self.last_ports),
                    motor_rates_Hz=dict(self.last_motor_rates), neural=self.neural.summary() if self.neural else None,
                    legacy=self.legacy.readout([]) if self.mode in ('B_COMPAT', 'C_SHADOW') else None,
                    scope=dict(fullBrain=self.graph.full_brain and self.neural is not None,
                               sourceNodes=self.graph.n, simulatedNodes=self.graph.n if self.neural else 0,
                               pairEdges=len(self.graph.weights), effectiveEdges=self.graph.manifest['effective_edge_count'],
                               unknownNeurotransmitters=self.graph.manifest['unknown_neurotransmitter_count'],
                               maskedEdges=self.graph.manifest['masked_edge_count'], biologicalValidation=False),
                    binding=self.bindings.summary(), subscription=dict(epoch=self.subscription_epoch,
                        ids=[self.graph.nodes[int(i)]['id'] for i in self.subscription]),
                    events=copy.deepcopy(self.events), interventions=dict(pending=len(self.pending), active=copy.deepcopy(self.active)),
                    fault=self.fault or self.body.fault, stopped=self.stopped,
                    recording=dict(active=self.recorder is not None, dropped=0,
                                   bytes=self.recorder.bytes if self.recorder else 0),
                    performance=self.performance())

    def performance(self):
        return dict(**self.timings, measured_model_s=self.timing_controls*CONTROL_DT,
                    sim_wall_ratio=(self.timing_controls*CONTROL_DT/self.timings['total_s']) if self.timings['total_s'] else 0.)

    def region_summary(self):
        if not self.neural: return []
        return self.neural.region_summary(self.graph.groups)

    def command(self, kind, payload):
        if not isinstance(payload, dict): raise ValueError('Command payload object required')
        if kind == 'intervene': return self.schedule(payload)
        if kind == 'release_all':
            self.pending.clear(); self.active.clear()
            if self.neural: self.neural.set_interventions()
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
        elif kind == 'place':
            category = payload.get('kind')
            if category not in ('food', 'hazard', 'obstacle'): raise ValueError('Unknown object kind')
            w = copy.deepcopy(self.world)
            p = payload.get('position')
            if not isinstance(p, list) or len(p) != 3: raise ValueError('Position required')
            p = [finite(v, 'position', -100, 100) for v in p]
            obj = dict(id='user-' + str(self.object_serial+1), p=p)
            if category == 'obstacle':
                r = finite(payload.get('radius', .8), 'radius', .2, 3)
                obj.update(r=r); obj['p'][1] = r
                if np.linalg.norm(np.asarray(obj['p']) - self.body.frame()[0]['position']) < r + 3:
                    raise ValueError('Place obstacle at least 3 mm clear of body')
                w['obstacles'].append(obj)
            else:
                obj.update(kind=category, strength=1.2); w['sources'].append(obj)
            validate_world(w); BEngine._set_world(self, w); self.object_serial += 1
        elif kind == 'clear_added':
            w = copy.deepcopy(self.world)
            for key in ('obstacles', 'sources'): w[key] = [o for o in w[key] if not o['id'].startswith('user-')]
            BEngine._set_world(self, w)
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
                    motion_expected=self.motion_expected, stopped=self.stopped,
                    subscription=[self.graph.nodes[int(i)]['id'] for i in self.subscription],
                    subscription_epoch=self.subscription_epoch)

    @classmethod
    def from_checkpoint(cls, graph, bindings, state, body_factory=FlyGymBody, *, backend_override=None):
        s = state
        if backend_override is not None and (backend_override not in NEURAL_BACKENDS or not s.get('neural')):
            raise ValueError('A C neural checkpoint and known backend are required for transfer')
        if s.get('schema') != 'flylab.checkpoint.v3' or s.get('app_version') != VERSION or s.get('versions') != runtime_versions():
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
                parameters=parameters, body_factory=body_factory, motion_expected=s['motion_expected'])
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
            e.event_serial = bounded_int(s['event_serial'], 'event serial', max(serials, default=0))
            e.object_serial = bounded_int(s['object_serial'], 'object serial', 0, 1000000)
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
