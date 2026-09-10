"""Reviewed ID bindings. Sensors never contain position, truth yaw or goals."""
import copy
import math
import numpy as np
from .integrity import digest, finite, bounded_int
from ..engine import validate_sensor_packet

CHANNELS = {'odor_mean', 'odor_left', 'odor_right', 'panorama_mean', 'panorama_bin',
            'front_proximity', 'contact', 'angular_velocity', 'danger', 'loom_left','loom_right','head_contact'}
INPUT_KINDS = {'sensory', 'direct_injection', 'assisted_input'}
MOTOR_GROUPS = ('forward', 'backward', 'stop', 'yaw_left', 'yaw_right')


class PortBindings:
    def __init__(self, graph, specification):
        self.graph = graph
        self.spec = copy.deepcopy(specification)
        s = self.spec
        if s.get('neural_parameters') is not None:
            from .neural import LIFParameters
            if not isinstance(s['neural_parameters'], dict): raise ValueError('Neural parameter object required')
            LIFParameters(**s['neural_parameters'])
        if s.get('schema') != 'flylab.bindings.v3' or s.get('graph_hash') != graph.hash:
            raise ValueError('BLOCKED_PORT_BINDING: schema or graph identity mismatch')
        if not isinstance(s.get('sensory'), list) or not 1 <= len(s['sensory']) <= 128:
            raise ValueError('BLOCKED_PORT_BINDING: explicit sensory ports required')
        seen = set()
        self.sensory = []
        for p in s['sensory']:
            name = p.get('name')
            if not isinstance(name, str) or not 1 <= len(name) <= 100 or name in seen:
                raise ValueError('Duplicate/invalid sensory port name')
            seen.add(name)
            if p.get('channel') not in CHANNELS or p.get('input_kind') not in INPUT_KINDS:
                raise ValueError('BLOCKED_PORT_BINDING: unsupported observation boundary')
            if p['channel'] in ('loom_left','loom_right','head_contact') and not s.get('sensor_model',{}).get('extended_observations'):
                raise ValueError('BLOCKED_PORT_BINDING: extended observations must be enabled explicitly')
            self._review(p)
            ids = graph.resolve(p.get('ids'), maximum=10000)
            if len(ids) == 0:
                raise ValueError('BLOCKED_PORT_BINDING: empty required input targets')
            if p['input_kind'] == 'sensory' and any(graph.nodes[i].get('super_class') not in ('sensory', 'optic', 'ascending') and graph.nodes[i].get('flow') != 'afferent' for i in ids):
                raise ValueError('Sensory targets must have sensory/early-relay annotations; use direct_injection explicitly')
            if p.get('method') not in ('drive_mV', 'poisson_Hz'):
                raise ValueError('Input method must include mV or Hz units')
            for key, low, high in [('gain', -1000, 1000), ('baseline', 0, 1000),
                                   ('cap', .001, 1000), ('tau_s', 0, 10), ('offset', -100, 100), ('scale', -100, 100)]:
                finite(p.get(key), key, low, high)
            bounded_int(p.get('delay_controls'), 'delay_controls', 0, 200)
            if p['channel'] == 'panorama_bin':
                bounded_int(p.get('bin'), 'panorama bin', 0, 63)
            if p['method'] == 'poisson_Hz':
                finite(p.get('pulse_mV'), 'pulse_mV', -100, 100)
            self.sensory.append((p, ids))
        motor = s.get('motor')
        if not isinstance(motor, dict) or set(motor) != set(MOTOR_GROUPS):
            raise ValueError('All motor roles must be declared (optional roles may be empty)')
        self.motor = {}
        for role in MOTOR_GROUPS:
            p = motor[role]
            if not isinstance(p, dict):
                raise ValueError('Invalid motor port')
            ids = graph.resolve(p.get('ids'), maximum=10000)
            if len(ids):
                self._review(p)
                if any(graph.nodes[i].get('super_class') != 'descending' for i in ids):
                    raise ValueError('Motor outputs must be annotated descending neurons')
            elif role in ('forward', 'yaw_left', 'yaw_right'):
                raise ValueError('BLOCKED_PORT_BINDING: required motor role ' + role)
            if role.startswith('yaw_') and p.get('output_side') != role[4:]:
                raise ValueError('Explicit output projection side required; never infer from cell name')
            finite(p.get('gain'), 'motor gain', 0, 10)
            self.motor[role] = (p, ids)
        self.motor_indices = np.asarray(sorted({int(i) for _, ids in self.motor.values() for i in ids}), dtype=np.int32)
        decoder = s.get('motor_decoder')
        if decoder is not None:
            if decoder.get('kind') not in ('bounded-opponent-v1','bounded-walk-off-v2'):
                raise ValueError('Unknown motor decoder model')
            finite(decoder.get('speed_limit'), 'speed limit', .01, 3.3)
            finite(decoder.get('yaw_limit'), 'yaw limit', .01, 3.)
            key='stop_half_drive' if decoder['kind']=='bounded-opponent-v1' else 'stop_full_drive'
            finite(decoder.get(key), key, .001, 100.)
        self.research_cohorts=[]
        cohorts=s.get('research_cohorts',[])
        if not isinstance(cohorts,list) or len(cohorts)>32:raise ValueError('At most 32 reviewed research cohorts')
        cohort_names=set()
        for cohort in cohorts:
            if not isinstance(cohort,dict):raise ValueError('Research cohort object required')
            self._review(cohort)
            name=cohort.get('name')
            if not isinstance(name,str) or not 1<=len(name)<=100 or name in cohort_names:
                raise ValueError('Invalid research cohort name')
            ids=graph.resolve(cohort.get('ids'),maximum=512)
            if not len(ids):raise ValueError('Empty research cohort')
            cohort_names.add(name);self.research_cohorts.append((cohort,ids))
        self.hash = digest(s)
        self.sensory_hash = digest(s['sensory'])
        self.motor_hash = digest(s['motor'])
        if s.get('motor_decoder'):
            self.motor_hash = digest(dict(ports=s['motor'], decoder=s['motor_decoder']))
        if s.get('neuromuscular'):
            self.sensory_hash = digest(dict(ports=s['sensory'], neuromuscular=s['neuromuscular']))
            self.motor_hash = digest(dict(ports=s['motor'], neuromuscular=s['neuromuscular']))

    @staticmethod
    def _review(p):
        if p.get('review_status') not in ('engineering_reviewed', 'biologically_validated'):
            raise ValueError('BLOCKED_PORT_BINDING: unreviewed mapping')
        if not isinstance(p.get('evidence'), list) or not p['evidence'] or any(not isinstance(x, str) or not x for x in p['evidence']):
            raise ValueError('Mapping evidence required')
        if not isinstance(p.get('uncertainty'), str) or not p['uncertainty']:
            raise ValueError('Mapping uncertainty must be explicit')

    @property
    def motor_execution(self):
        model = self.spec.get('neuromuscular')
        if not model:
            return 'descending-CPG-adapter'
        return 'BANC-neuron-muscle-joints-' + model['schema'].rsplit('.', 1)[-1]

    def summary(self):
        return dict(hash=self.hash, profile=self.spec.get('profile', 'custom'),
                    hazard_semantics=self.spec.get('hazard_semantics', 'unbound'),
                    sensory_binding_hash=self.sensory_hash, motor_binding_hash=self.motor_hash,
                    sensory=[dict(name=p['name'], channel=p['channel'], input_kind=p['input_kind'],
                                  targets=len(ids), method=p['method'], review_status=p['review_status']) for p, ids in self.sensory],
                    motor={k: dict(targets=len(ids), gain=p['gain']) for k, (p, ids) in self.motor.items()},
                    motor_decoder=self.spec.get('motor_decoder', {'kind':'legacy-subtractive-v1'}),
                    research_cohorts=[dict(name=c['name'],targets=len(ids),review_status=c['review_status'])
                                      for c,ids in self.research_cohorts],
                    motor_execution=self.motor_execution,
                    unused_observations=self.spec.get('unused_observations', []),
                    biological_validation=all(p['review_status'] == 'biologically_validated' for p, _ in self.sensory)
                    and all(p.get('review_status') == 'biologically_validated' for p, ids in self.motor.values() if len(ids)))


def feature(packet, port, supplemental=None):
    channel = port['channel']
    if channel in ('loom_left','loom_right','head_contact'):
        if supplemental is None or channel not in supplemental:raise ValueError('Extended observation unavailable: '+channel)
        return finite(supplemental[channel],channel,0.,1000.)
    if channel == 'odor_mean': return sum(packet['odor']) / 2
    if channel == 'odor_left': return packet['odor'][0]
    if channel == 'odor_right': return packet['odor'][1]
    if channel == 'panorama_mean': return sum(packet['panorama']) / 64
    if channel == 'panorama_bin': return packet['panorama'][port['bin']]
    if channel == 'front_proximity': return max(0., 1 - packet['nearRanges'][4] / 10)
    if channel == 'angular_velocity': return packet['angularVelocity']
    return packet[channel]


class SensoryEncoder:
    def __init__(self, bindings, seed):
        self.bindings = bindings
        self.rng = np.random.default_rng(seed ^ 0xEC03)
        self.filtered = np.zeros(len(bindings.sensory), dtype=np.float64)
        self.delays = [[0.] * p['delay_controls'] for p, _ in bindings.sensory]
        self.control_tick = 0

    def encode(self, packet, dt, neural_dt, disabled=(), supplemental=None):
        validate_sensor_packet(packet)
        count = round(dt / neural_dt)
        if abs(count * neural_dt - dt) > 1e-12:
            raise ValueError('Encoder requires integral neural ticks')
        drive = np.zeros(self.bindings.graph.n, dtype=np.float32)
        pulse_ids, pulse_values = [], []
        values = []
        for i, (p, ids) in enumerate(self.bindings.sensory):
            raw_feature = feature(packet, p, supplemental)
            requested = p['baseline'] + p['gain'] * (raw_feature - p['offset']) * p['scale']
            value = float(np.clip(requested, 0, p['cap']))
            alpha = 1 if p['tau_s'] == 0 else 1 - math.exp(-dt / p['tau_s'])
            self.filtered[i] += alpha * (value - self.filtered[i])
            value = float(self.filtered[i])
            if p['delay_controls']:
                self.delays[i].append(value)
                value = self.delays[i].pop(0)
            enabled = p['name'] not in disabled and p['channel'] not in disabled and '*' not in disabled
            if p['method'] == 'drive_mV':
                if enabled:
                    np.add.at(drive, ids, value)
            else:
                # Identical RNG draw shape with a channel enabled or disabled.
                pulses = self.rng.poisson(value * neural_dt, (count, len(ids))) * p['pulse_mV']
                pulse_ids.extend(ids.tolist()); pulse_values.append(pulses if enabled else np.zeros_like(pulses))
            values.append(dict(name=p['name'], value=value if enabled else 0.,
                               unit='mV' if p['method'] == 'drive_mV' else 'Hz', enabled=enabled,
                               feature=float(raw_feature), requested=float(requested), clipped=bool(requested<0 or requested>p['cap'])))
        self.control_tick += 1
        pulses = None if not pulse_values else (np.asarray(pulse_ids, dtype=np.int32), np.concatenate(pulse_values, axis=1))
        return drive, pulses, values

    def snapshot(self):
        return dict(binding_hash=self.bindings.hash, control_tick=self.control_tick,
                    filtered=self.filtered.copy(), delays=copy.deepcopy(self.delays),
                    rng=copy.deepcopy(self.rng.bit_generator.state))

    def restore(self, s):
        if s.get('binding_hash') != self.bindings.hash:
            raise ValueError('Encoder binding mismatch')
        tick = bounded_int(s.get('control_tick'), 'encoder clock')
        a = np.asarray(s.get('filtered'))
        if a.dtype != self.filtered.dtype or a.shape != self.filtered.shape or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid encoder filter')
        delays = s.get('delays')
        if not isinstance(delays, list) or len(delays) != len(self.delays):
            raise ValueError('Invalid encoder delays')
        for i, ((p, _), q) in enumerate(zip(self.bindings.sensory, delays)):
            if not isinstance(q, list) or len(q) != p['delay_controls'] or a[i] > p['cap']:
                raise ValueError('Invalid encoder delay length or filter bound')
            for v in q: finite(v, 'encoder delayed value', 0, p['cap'])
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(s['rng'])
        self.filtered[:] = a; self.delays = copy.deepcopy(delays); self.rng = rng; self.control_tick = tick


def zero_command():
    return dict(forwardSpeed=0., yawRate=0., verticalSpeed=0.)


class MotorDecoder:
    def __init__(self, bindings):
        self.bindings = bindings
        self.diagnostics = {}

    def decode(self, backend):
        indices = self.bindings.motor_indices
        rates = backend.readout(indices)['rate_Hz']
        table = {int(i): float(v) for i, v in zip(indices, rates)}
        values, scaled = {}, {}
        for role, (p, ids) in self.bindings.motor.items():
            value = float(np.mean([table[int(i)] for i in ids])) if len(ids) else 0.
            values[role] = value
            scaled[role] = value * p['gain']
        command = dict(forwardSpeed=float(np.clip(scaled['forward'] - scaled['backward'] - scaled['stop'], -3.3, 3.3)),
                       yawRate=float(np.clip(scaled['yaw_right'] - scaled['yaw_left'], -3., 3.)), verticalSpeed=0.)
        raw = dict(forwardSpeed=scaled['forward']-scaled['backward']-scaled['stop'],
                   yawRate=scaled['yaw_right']-scaled['yaw_left'])
        model = self.bindings.spec.get('motor_decoder')
        if model:
            if model['kind']=='bounded-walk-off-v2':
                # BB is a Walk-OFF population, not a universal brake. This
                # explicit adapter hypothesis gates only forward recruitment.
                gate=float(np.clip(1-scaled['stop']/model['stop_full_drive'],0.,1.))
                longitudinal=scaled['forward']-scaled['backward']
                raw=dict(forwardSpeed=longitudinal*gate if longitudinal>0 else longitudinal,
                         yawRate=scaled['yaw_right']-scaled['yaw_left'])
            else:
                # Stop reduces all locomotion. It cannot create reverse movement.
                gate = 1. / (1. + scaled['stop'] / model['stop_half_drive'])
                raw = dict(forwardSpeed=(scaled['forward']-scaled['backward'])*gate,
                           yawRate=(scaled['yaw_right']-scaled['yaw_left'])*gate)
            command = dict(forwardSpeed=float(np.clip(raw['forwardSpeed'], -model['speed_limit'], model['speed_limit'])),
                           yawRate=float(np.clip(raw['yawRate'], -model['yaw_limit'], model['yaw_limit'])), verticalSpeed=0.)
        self.diagnostics = dict(scaled=scaled, before_limits=raw, after_limits=command,
                                clipped={k:raw[k]!=command[k] for k in raw})
        if model:self.diagnostics.update(stop_gate=gate,stop_scope='forward_only' if model['kind']=='bounded-walk-off-v2' else 'all_locomotion')
        return command, values


class RecoverySupervisor:
    """Explicit engineering assistance; active only in C_ASSISTED mode."""
    def __init__(self):
        self.remaining_turn = 0.
        self.side = 1
        self.contact_s = 0.
        self.reverse_s = 0.

    def step(self, packet, dt):
        front = packet['nearRanges'][4]
        contact = bool(packet['contact'])
        if not self.remaining_turn and (front < 3.5 or contact):
            left, right = sum(packet['nearRanges'][:4]), sum(packet['nearRanges'][5:])
            self.side = 1 if right >= left else -1
            self.remaining_turn = math.pi / 2
        self.contact_s = self.contact_s + dt if contact else 0.
        if self.contact_s >= .15 and self.reverse_s <= 0:
            self.reverse_s = .3
            self.contact_s = 0.
        if self.remaining_turn:
            self.remaining_turn = max(.001, self.remaining_turn - max(0., packet['angularVelocity'] * self.side) * dt)
            if self.remaining_turn <= .001 and front > 4.5 and not contact:
                self.remaining_turn = 0.
            elif self.reverse_s > 0:
                self.reverse_s = max(0., self.reverse_s - dt)
                return dict(forwardSpeed=-1.3, yawRate=self.side * 1., verticalSpeed=0.), 'contact_reverse'
            else:
                return dict(forwardSpeed=0., yawRate=self.side * 3., verticalSpeed=0.), 'proximity_turn'
        return None, None

    def snapshot(self):
        return dict(remaining_turn=self.remaining_turn, side=self.side, contact_s=self.contact_s, reverse_s=self.reverse_s)

    def restore(self, s):
        r = finite(s.get('remaining_turn'), 'assist remaining turn', 0, math.pi/2)
        c = finite(s.get('contact_s'), 'assist contact timer', 0, .3)
        b = finite(s.get('reverse_s'), 'assist reverse timer', 0, .3)
        side = s.get('side')
        if type(side) is not int or side not in (-1, 1):
            raise ValueError('Invalid assist side')
        self.remaining_turn, self.contact_s, self.reverse_s, self.side = r, c, b, side


class MotorArbiter:
    @staticmethod
    def choose(neural, assist, *, mode, legacy=None, motor_coupled=True, stopped=False, assist_reason=None):
        if stopped or not motor_coupled:
            final, source = zero_command(), 'safety_stop' if stopped else 'motor_disconnect'
        elif mode in ('B_COMPAT', 'C_SHADOW'):
            if legacy is None: raise ValueError('Legacy command required for explicit B mode')
            final, source = legacy, 'legacy_b_rate'
        elif mode == 'C_ASSISTED' and assist is not None:
            final, source = assist, 'recovery_supervisor'
        elif mode in ('C_STRICT', 'C_ASSISTED'):
            final, source = neural, 'connectome_lif'
        else:
            raise ValueError('Unknown control mode')
        return dict(u_neural=copy.deepcopy(neural), u_assist=copy.deepcopy(assist),
                    u_final=copy.deepcopy(final), command_source=source,
                    assist_reason=assist_reason, motor_coupled=motor_coupled,
                    u_legacy=copy.deepcopy(legacy))
