"""Causal whole-BANC neural / one-joint Hill-muscle experimental loop.

This is a separate calibration profile, not a replacement for the walking
engine. Motor identities are anatomical; rate recruitment and FeCO tuning
remain explicit, versioned hypotheses. No target angle, servo or CPG is used.
"""
import copy
from dataclasses import asdict, dataclass
import numpy as np
from .integrity import boolean, bounded_int, digest, finite
from .model_config import model_ticks
from .muscles import JOINT, MUSCLES, MuscleRig
from .muscle_calibration import direction_status, interior_angle
from .neural import LIFParameters, create_backend
from .neuromuscular import FECO, FECO_CIRCUIT, SOURCE, sensory_kind
from .receptors import JointReceptors, ReceptorParameters

MOTOR_TYPES = ('tibia_flexor_Fast', 'tibia_extensor_FETi')
TARGET_TYPES = ('tibia_flexor_muscle', 'tibia_extensor_muscle')
FEATURES = {'SNpp50':'claw_positive', 'SNpp51':'claw_negative',
            'SNpp41':'hook_positive', 'SNpp39':'hook_negative'}
SWAPPED = dict(claw_positive='claw_negative', claw_negative='claw_positive',
               hook_positive='hook_negative', hook_negative='hook_positive')


@dataclass(frozen=True)
class JointParameters:
    control_dt: float = .001
    sensory_gain_mV: float = 18.
    rate_half_Hz: float = 100.
    receptor_delay_steps: int = 1
    polarity_swapped: bool = False
    initial_q_rad: float = 1.862

    def __post_init__(self):
        finite(self.control_dt, 'joint control dt', .0001, .005)
        finite(self.sensory_gain_mV, 'sensory gain', 0., 100.)
        finite(self.rate_half_Hz, 'recruitment half-rate', 1., 1000.)
        bounded_int(self.receptor_delay_steps, 'receptor delay', 0, 1000)
        boolean(self.polarity_swapped, 'polarity swap')
        finite(self.initial_q_rad, 'initial joint angle', -10., 10.)


def anatomical_bindings(graph):
    if (graph.manifest.get('dataset_id'), graph.manifest.get('snapshot_id')) != ('flywire_banc', '888'):
        raise ValueError('Single-joint profile requires BANC v888, without cross-specimen IDs')
    motors = [[] for _ in MUSCLES]; sensory = {}; excluded = []
    for n in graph.nodes:
        a = n.get('source_annotations', {})
        if n.get('soma_side') != 'left': continue
        if n.get('super_class') == 'motor' and a.get('body_part_effector') == 'front_leg':
            cell = n.get('cell_type')
            if cell in MOTOR_TYPES:
                i = MOTOR_TYPES.index(cell)
                if a.get('peripheral_target_type') != TARGET_TYPES[i]:
                    raise ValueError('Motor cell type and anatomical muscle disagree: '+n['id'])
                motors[i].append(n['id'])
            else: excluded.append(dict(id=n['id'], reason='No corresponding actuator in this two-MTU fixture'))
        elif n.get('super_class') == 'sensory' and a.get('body_part_sensory') == 'front_leg':
            kind, cell = sensory_kind(n), n.get('cell_type')
            channel = FEATURES.get(cell) if kind in ('claw','hook') else 'club_highpass' if kind == 'club' else None
            if channel and (kind == 'club' or channel.startswith(kind+'_')):
                sensory.setdefault((cell, channel), []).append(n['id'])
            else: excluded.append(dict(id=n['id'], reason='Receptor tuning or local receptive field unresolved'))
    if any(len(ids) != 1 for ids in motors): raise ValueError('Exactly one Fast flexor and one FETi neuron required')
    channels = {feature for cell, feature in sensory}
    if not set(FEATURES.values()) <= channels: raise ValueError('Missing opponent claw/hook cohorts')
    return dict(motor=[dict(cell_type=cell, ids=ids, actuator=muscle, anatomical_target=target)
                for cell,ids,muscle,target in zip(MOTOR_TYPES,motors,MUSCLES,TARGET_TYPES)],
        sensory=[dict(cell_type=cell, feature=feature, ids=sorted(ids)) for (cell,feature),ids in sorted(sensory.items())],
        excluded_from_external_ports=excluded,
        all_excluded_neurons_still_computed=True,
        motor_evidence='BANC side, cell type and peripheral target; fast MTU model correspondence across specimens',
        sensory_evidence='Existing v2 MANC-to-BANC cell-type polarity hypothesis, not measured BANC response tuning',
        motor_unit_recruitment='positive bounded rate-to-activation hypothesis, not a receptor-aware NMJ model',
        central_nt_weights='Unchanged graph weights, including uncertain NT predictions; no peripheral sign inference from CNS NT',
        evidence=[SOURCE, FECO, FECO_CIRCUIT, 'https://arxiv.org/html/2509.06426v1'])


def build_profile(graph, body, parameters=None, neural_parameters=None):
    p = parameters or JointParameters(); n = neural_parameters or LIFParameters()
    model_ticks(p.control_dt, n.dt); model_ticks(p.control_dt, body.model.opt.timestep)
    lo, hi = map(float, body.model.jnt_range[0])
    finite(p.initial_q_rad, 'initial joint angle', lo, hi)
    return dict(schema='flylab.single-joint.v1', graph_hash=graph.hash, body_hash=body.identity,
        parameters=asdict(p), neural_parameters=asdict(n), bindings=anatomical_bindings(graph),
        receptor_parameters=asdict(ReceptorParameters(lo, hi, dt=p.control_dt, delay_steps=p.receptor_delay_steps)),
        joint=JOINT, neural_scope='all declared graph nodes and retained central weights',
        body_scope='fixed-body left Fe-Ti only; no contacts or other controlled joints',
        order='At t: observe q/v, update delayed receptors, read rates(t); advance neurons and physics over [t,t+dt] using held input/rates(t)',
        motor_lag_controls=1, recruitment='u = rate_Hz / (rate_half_Hz + rate_Hz)',
        guard='stop on singular/reversed moment arm at every physics tick; no corrective force',
        unit_contract=body.metadata['unit_contract'],
        biological_validation=False, physiological_status='CALIBRATION_REQUIRED')


def encode_feedback(size, ports, sample, gain, swapped=False):
    """Shared encoder for moving-body experiments and fixed-posture probes."""
    drive = np.zeros(size, np.float32)
    for feature, ids in ports:
        feature = SWAPPED.get(feature, feature) if swapped else feature
        drive[ids] += gain*sample['output'][feature]
    return drive


class SingleJointLoop:
    def __init__(self, graph, profile=None, backend='exp_lif_cpu_reference', *, body=None):
        self.graph = graph; self.body = body or MuscleRig()
        if profile is None: profile = build_profile(graph, self.body)
        p = JointParameters(**profile['parameters']); neural_p = LIFParameters(**profile['neural_parameters'])
        if profile != build_profile(graph, self.body, p, neural_p):
            raise ValueError('Single-joint profile identity or anatomical binding mismatch')
        self.profile = copy.deepcopy(profile); self.identity = digest(profile); self.p = p
        self.neural_steps = model_ticks(p.control_dt, neural_p.dt)
        self.physics_steps = model_ticks(p.control_dt, self.body.model.opt.timestep)
        if self.body.tick != 0: raise ValueError('A fresh muscle fixture is required')
        self.body.data.qpos[0] = p.initial_q_rad
        self.body.mj.mj_forward(self.body.model, self.body.data)
        if direction_status(self.body.frame())['status'] != 'CONSISTENT':
            raise ValueError('Initial pose has reversed or singular muscle geometry')
        self.receptors = JointReceptors(ReceptorParameters(**profile['receptor_parameters']))
        binding = profile['bindings']
        self.motor_indices = graph.resolve([m['ids'][0] for m in binding['motor']])
        self.ports = [(row['feature'], graph.resolve(row['ids'], maximum=10000)) for row in binding['sensory']]
        self.observed = np.unique(np.concatenate([self.motor_indices]+[ids for _, ids in self.ports])).astype(np.int32)
        self.neural = create_backend(graph, neural_p, backend)
        self.control_tick = 0; self.fault = None
        self.last_input = None; self.last_receptors = None
        self.last_motor_rates = np.zeros(2); self.last_excitation = np.zeros(2)
        self.last_motor_tick = 0; self.last_drive_max = 0.

    def step(self, *, stimulation=None, suppress_ids=(), mute_ids=(), feedback=True, motor_connected=True, torque=0.):
        if self.fault: raise RuntimeError('FAULTED_SINGLE_JOINT_LOOP: restore a healthy checkpoint')
        boolean(feedback, 'feedback'); boolean(motor_connected, 'motor connection')
        torque = finite(torque, 'external torque in model units', -1., 1.)
        stimulation = {} if stimulation is None else stimulation
        if not isinstance(stimulation, dict): raise ValueError('Stimulation must map real neuron IDs to held mV')
        stimulus_ids = list(stimulation)
        indices = self.graph.resolve(stimulus_ids)
        values = [finite(stimulation[i], 'stimulation mV', -100., 100.) for i in stimulus_ids]
        suppress_ids, mute_ids = list(suppress_ids), list(mute_ids)
        suppress = self.graph.resolve(suppress_ids).tolist()
        mute = self.graph.resolve(mute_ids).tolist()
        bounded_int(self.control_tick+1, 'joint control tick', 1, 2**31//self.physics_steps)
        self.neural.set_interventions(suppress=suppress, mute=mute)
        self.last_input = dict(stimulation=dict(zip(stimulus_ids,values)), suppress_ids=list(suppress_ids),
                               mute_ids=mute_ids, feedback=feedback, motor_connected=motor_connected, torque=torque)
        # Read the previous boundary rates before advancing either subsystem.
        self.last_motor_tick = self.neural.tick
        rates = self.neural.readout(self.motor_indices)['rate_Hz'].astype(float)
        if not np.isfinite(rates).all() or np.any(rates < 0): raise RuntimeError('FAULT_MOTOR_RATES')
        self.last_motor_rates = rates.copy(); self.last_excitation = rates/(self.p.rate_half_Hz+rates)
        self.last_receptors = self.receptors.step(float(self.body.data.qpos[0]), float(self.body.data.qvel[0]), enabled=feedback)
        drive = encode_feedback(self.graph.n,self.ports,self.last_receptors,self.p.sensory_gain_mV,self.p.polarity_swapped)
        self.last_drive_max = float(drive.max())
        drive[indices] += values
        try:
            self.neural.advance(drive, self.neural_steps, capture=self.observed)
            frame = self.body.step(self.last_excitation, steps=self.physics_steps, connected=motor_connected,
                                   torque=torque, direction_guard=True)
            if (not np.isfinite(np.array([frame[k] for k in ('q_rad','qdot_rad_s','qacc_rad_s2','actuator_torque')])).all()
                or any(not 0 <= a <= 1 for a in frame['activation'])
                or abs(self.body.data.time-self.body.tick*self.body.model.opt.timestep) > 1e-8):
                raise RuntimeError('FAULT_JOINT_NUMERICS')
            self.control_tick += 1
        except RuntimeError as exc:
            self.fault = dict(code=str(exc), completed_controls=self.control_tick,
                neural_tick=self.neural.tick, physics_tick=self.body.tick, receptor_tick=self.receptors.tick)
        return self.frame()

    def frame(self):
        physics = self.body.frame()
        return dict(profile_hash=self.identity, control_tick=self.control_tick, seconds=physics['seconds'],
            neural_tick=self.neural.tick, neural_seconds=self.neural.tick*self.neural.p.dt,
            simulated_neurons=self.graph.n, fault=copy.deepcopy(self.fault), physics=physics,
            direction=direction_status(physics), interior_angle_rad=interior_angle(self.body.model,self.body.data),
            receptors=copy.deepcopy(self.last_receptors), feedback_drive_max_mV=self.last_drive_max,
            motor_rate_sample_tick=self.last_motor_tick, motor_rates_Hz=self.last_motor_rates.tolist(),
            requested_excitation=self.last_excitation.tolist(), input=copy.deepcopy(self.last_input),
            signals=dict(ids=[self.graph.nodes[int(i)]['id'] for i in self.observed],
                         **{k:v.tolist() for k,v in self.neural.readout(self.observed).items()}),
            selected_spike_events=copy.deepcopy(self.neural.last_events), biological_validation=False)

    def snapshot(self):
        return dict(schema='flylab.single-joint-state.v1', profile_hash=self.identity,
            backend=self.neural.backend, control_tick=self.control_tick, fault=copy.deepcopy(self.fault),
            body=self.body.snapshot(), receptors=self.receptors.snapshot(), neural=self.neural.snapshot(),
            diagnostics=dict(last_input=copy.deepcopy(self.last_input), last_receptors=copy.deepcopy(self.last_receptors),
                last_motor_rates=self.last_motor_rates.copy(), last_excitation=self.last_excitation.copy(),
                last_motor_tick=self.last_motor_tick, last_drive_max=self.last_drive_max))

    def restore(self, saved):
        # Stage a complete replacement. Even a late neural validation failure
        # must leave the current physics, delay filters and GPU state intact.
        if (saved.get('schema') != 'flylab.single-joint-state.v1' or saved.get('profile_hash') != self.identity
            or saved.get('backend') != self.neural.backend):
            raise ValueError('Single-joint checkpoint identity mismatch')
        tick = bounded_int(saved.get('control_tick'), 'joint control tick', 0, 2**31//self.physics_steps)
        if saved.get('fault') is not None:
            raise ValueError('Fault snapshots are evidence only; resume from a healthy checkpoint')
        if (saved['body'].get('tick') != tick*self.physics_steps or saved['neural'].get('tick') != tick*self.neural_steps
            or saved['receptors'].get('tick') != tick):
            raise ValueError('Single-joint checkpoint clocks disagree')
        candidate = SingleJointLoop(self.graph, self.profile, self.neural.backend, body=MuscleRig(self.body.source))
        candidate.body.restore(saved['body']); candidate.receptors.restore(saved['receptors'])
        if direction_status(candidate.body.frame())['status'] != 'CONSISTENT':
            raise ValueError('Restored muscle geometry is reversed or singular')
        diag = saved.get('diagnostics', {})
        if set(diag) != {'last_input','last_receptors','last_motor_rates','last_excitation','last_motor_tick','last_drive_max'}:
            raise ValueError('Joint diagnostics schema mismatch')
        for key in ('last_motor_rates', 'last_excitation'):
            value = diag.get(key)
            if (not isinstance(value,np.ndarray) or value.dtype != np.float64 or value.shape != (2,)
                or not np.isfinite(value).all() or np.any(value < 0)):
                raise ValueError('Invalid joint diagnostic array: '+key)
        expected = diag['last_motor_rates']/(self.p.rate_half_Hz+diag['last_motor_rates'])
        if not np.array_equal(diag['last_excitation'],expected): raise ValueError('Invalid recruitment state')
        expected_tick = max(0,tick-1)*self.neural_steps
        if bounded_int(diag.get('last_motor_tick'),'motor sample tick') != expected_tick:
            raise ValueError('Motor sample clock mismatch')
        finite(diag.get('last_drive_max'),'feedback drive maximum',0.,self.p.sensory_gain_mV)
        if not np.array_equal(diag['last_excitation'],saved['body']['requested']):
            raise ValueError('Muscle command and recorded recruitment disagree')
        # Diagnostics are descriptive, but still must be finite JSON and agree
        # with whether any sample has actually been taken.
        digest(diag.get('last_input')); digest(diag.get('last_receptors'))
        if (tick == 0) != (diag.get('last_input') is None and diag.get('last_receptors') is None):
            raise ValueError('Invalid initial diagnostic state')
        if np.any(diag['last_motor_rates'] > 1./candidate.neural.p.dt+1e-3):
            raise ValueError('Motor rate exceeds the numerical event limit')
        if tick == 0:
            if np.any(diag['last_motor_rates']) or diag['last_drive_max'] != 0.:
                raise ValueError('Nonzero initial motor or sensory diagnostic')
        else:
            command, sensory = diag['last_input'], diag['last_receptors']
            if not isinstance(command,dict) or set(command) != {'stimulation','suppress_ids','mute_ids','feedback','motor_connected','torque'}:
                raise ValueError('Invalid recorded joint command')
            for key in ('feedback','motor_connected'): boolean(command[key],key)
            torque=finite(command['torque'],'saved external torque',-1.,1.)
            if command['motor_connected'] != saved['body']['connected'] or torque != float(candidate.body.data.qfrc_applied[0]):
                raise ValueError('Recorded command and physical input disagree')
            if not isinstance(command['stimulation'],dict): raise ValueError('Invalid recorded stimulation')
            self.graph.resolve(list(command['stimulation']))
            for value in command['stimulation'].values(): finite(value,'saved stimulation',-100.,100.)
            for names,mask in (('suppress_ids','suppress'),('mute_ids','mute')):
                indices=self.graph.resolve(command[names]);expected_mask=np.zeros(self.graph.n,bool);expected_mask[indices]=True
                if not np.array_equal(expected_mask,saved['neural'][mask]): raise ValueError('Recorded neural intervention mismatch')
            if not isinstance(sensory,dict) or sensory.get('tick') != tick or sensory.get('profile_hash') != candidate.receptors.identity:
                raise ValueError('Invalid recorded receptor clock or identity')
            if sensory.get('enabled') is not command['feedback'] or sensory.get('seconds') != tick*self.p.control_dt:
                raise ValueError('Recorded receptor enablement or time mismatch')
            from .receptors import CHANNELS
            for key in ('raw','filtered','output'):
                values=sensory.get(key)
                if not isinstance(values,dict) or set(values)!=set(CHANNELS): raise ValueError('Invalid recorded receptor channels')
                for v in values.values(): finite(v,'recorded receptor value',0.,1.)
            if not np.array_equal([sensory['filtered'][key] for key in CHANNELS],candidate.receptors.filtered):
                raise ValueError('Recorded receptor filter mismatch')
            delay=candidate.receptors.p.delay_steps;queue=candidate.receptors.queue
            expected_output=queue[(tick-1-delay)%len(queue)] if command['feedback'] else np.zeros(len(CHANNELS))
            if not np.array_equal([sensory['output'][key] for key in CHANNELS],expected_output):
                raise ValueError('Recorded receptor delayed output mismatch')
            expected_drive=encode_feedback(self.graph.n,self.ports,sensory,self.p.sensory_gain_mV,self.p.polarity_swapped)
            if float(expected_drive.max()) != diag['last_drive_max']: raise ValueError('Recorded feedback magnitude mismatch')
        candidate.neural.restore(saved['neural'])
        candidate.control_tick = tick
        for key,value in diag.items():
            setattr(candidate,key,copy.deepcopy(value))
        self.__dict__.update(candidate.__dict__)
