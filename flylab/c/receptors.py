"""Inspectable joint receptor hypotheses for calibration, without neural IDs.

Default claw and hook proxies distinguish raw joint-coordinate directions;
their anatomical polarity is unassigned. Optional claw position tuning uses
a measured interior angle and an explicit flexion-positive convention. The
club proxy is a velocity high-pass, not a measured vibration tuning curve.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .integrity import bounded_int, boolean, digest, finite

CHANNELS = ('claw_positive', 'claw_negative', 'hook_positive', 'hook_negative',
            'club_highpass', 'load', 'touch')


@dataclass(frozen=True)
class ReceptorParameters:
    angle_low_rad: float
    angle_high_rad: float
    velocity_scale_rad_s: float = 20.
    vibration_scale_rad_s: float = 20.
    velocity_tau_s: float = .03
    response_tau_s: float = .02
    delay_steps: int = 1
    dt: float = .001
    load_half_bw: float = .5
    contact_half_bw: float = .05

    def __post_init__(self):
        finite(self.angle_low_rad, 'low joint angle', -10., 10.)
        finite(self.angle_high_rad, 'high joint angle', self.angle_low_rad+1e-6, 10.)
        for key in ('velocity_scale_rad_s', 'vibration_scale_rad_s', 'velocity_tau_s',
                    'response_tau_s', 'load_half_bw', 'contact_half_bw'):
            finite(getattr(self, key), key, 1e-6, 1000.)
        finite(self.dt, 'receptor dt', 1e-5, .05)
        bounded_int(self.delay_steps, 'receptor delay steps', 0, 1000)


@dataclass(frozen=True)
class ClawPositionTuning:
    """Optional position proxy based on measured femur-tibia interior angle.

    Mamiya et al. (2018), Figure 5: little claw activity near 90 degrees,
    flexion/extension populations active on opposite sides. Piecewise-linear
    amplitudes are an engineering hypothesis; calcium is not a firing rate.
    """
    neutral_rad: float = math.pi / 2
    flexion_limit_rad: float = math.pi / 10
    extension_limit_rad: float = math.pi

    def __post_init__(self):
        finite(self.flexion_limit_rad, 'claw flexion limit', 0., math.pi)
        finite(self.neutral_rad, 'claw neutral angle', self.flexion_limit_rad+1e-6, math.pi)
        finite(self.extension_limit_rad, 'claw extension limit', self.neutral_rad+1e-6, math.pi)

    def encode(self, interior_angle_rad):
        angle = finite(interior_angle_rad, 'measured joint interior angle', 0., math.pi)
        return np.clip([
            (self.neutral_rad-angle)/(self.neutral_rad-self.flexion_limit_rad),
            (angle-self.neutral_rad)/(self.extension_limit_rad-self.neutral_rad),
        ], 0., 1.)


class JointReceptors:
    def __init__(self, parameters, *, claw_tuning=None):
        if claw_tuning is not None and not isinstance(claw_tuning, ClawPositionTuning):
            raise ValueError('A declared claw position tuning profile is required')
        self.claw_tuning = claw_tuning
        self.p = parameters; self.tick = 0; self.velocity_lowpass = 0.
        self.filtered = np.zeros(len(CHANNELS))
        self.queue = np.zeros((parameters.delay_steps+1, len(CHANNELS)))
        self.metadata = dict(schema='flylab.joint-receptors.v1', parameters=asdict(parameters),
            channels=list(CHANNELS), unit='normalized proxy 0..1', neural_targets=[],
            evidence_status='engineering hypotheses; physiological calibration pending',
            polarity='positive/negative model q; anatomical polarity unassigned',
            club_model='abs(velocity - lowpass_velocity); no measured frequency tuning',
            source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC8665017/', biological_validation=False)
        if claw_tuning is not None:
            # Default metadata/hash remains byte-for-byte compatible.
            self.metadata.update(schema='flylab.joint-receptors.v2',
                claw_position_tuning=asdict(claw_tuning),
                position_encoding='opponent half-wave interior-angle proxy; no hysteresis model',
                polarity='positive model q is anatomical flexion; caller supplies measured interior angle',
                position_source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/')
        self.identity = digest(self.metadata)

    def step(self, q, velocity, *, support_bw=0., contact_bw=0., enabled=True,
             interior_angle_rad=None):
        q = finite(q, 'receptor angle', -1000., 1000.)
        velocity = finite(velocity, 'receptor velocity', -1e6, 1e6)
        support_bw = finite(support_bw, 'support load', 0., 1e6)
        contact_bw = finite(contact_bw, 'non-support contact', 0., 1e6)
        boolean(enabled, 'receptors enabled')
        bounded_int(self.tick+1, 'receptor tick')
        p = self.p
        angle = float(np.clip((q-p.angle_low_rad)/(p.angle_high_rad-p.angle_low_rad), 0., 1.))
        if self.claw_tuning is None:
            if interior_angle_rad is not None:
                raise ValueError('Interior angle requires an explicit claw tuning profile')
            position = (angle, 1-angle)
        else:
            # Validate before updating filters, delay queue or clocks.
            position = self.claw_tuning.encode(interior_angle_rad)
        raw = np.array([*position, max(0., velocity)/p.velocity_scale_rad_s,
            max(0., -velocity)/p.velocity_scale_rad_s,
            abs(velocity-self.velocity_lowpass)/p.vibration_scale_rad_s,
            support_bw/(p.load_half_bw+support_bw), contact_bw/(p.contact_half_bw+contact_bw)])
        np.clip(raw, 0., 1., out=raw)
        self.filtered += -math.expm1(-p.dt/p.response_tau_s)*(raw-self.filtered)
        self.queue[self.tick % len(self.queue)] = self.filtered
        output = self.queue[(self.tick-p.delay_steps) % len(self.queue)].copy()
        self.velocity_lowpass += -math.expm1(-p.dt/p.velocity_tau_s)*(velocity-self.velocity_lowpass)
        self.tick += 1
        if not enabled: output.fill(0.)
        stimulus = dict(q_rad=q, velocity_rad_s=velocity, support_bw=support_bw, contact_bw=contact_bw)
        if self.claw_tuning is not None:
            stimulus['interior_angle_rad'] = float(interior_angle_rad)
        return dict(tick=self.tick, seconds=self.tick*p.dt, profile_hash=self.identity,
            raw=dict(zip(CHANNELS, raw.tolist())), filtered=dict(zip(CHANNELS, self.filtered.tolist())),
            output=dict(zip(CHANNELS, output.tolist())), enabled=enabled,
            stimulus=stimulus)

    def snapshot(self):
        return dict(schema='flylab.joint-receptor-state.v1', profile_hash=self.identity,
            tick=self.tick, velocity_lowpass=self.velocity_lowpass,
            filtered=self.filtered.copy(), queue=self.queue.copy())

    def restore(self, state):
        if state.get('schema') != 'flylab.joint-receptor-state.v1' or state.get('profile_hash') != self.identity:
            raise ValueError('Receptor profile mismatch')
        tick = bounded_int(state.get('tick'), 'receptor tick')
        velocity = finite(state.get('velocity_lowpass'), 'velocity lowpass', -1e6, 1e6)
        staged = {}
        for key, current in (('filtered', self.filtered), ('queue', self.queue)):
            value = state.get(key)
            if (not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != current.shape or
                not np.isfinite(value).all() or np.any((value < 0) | (value > 1))):
                raise ValueError('Invalid receptor state array: '+key)
            staged[key] = value.copy()
        if tick == 0 and (velocity != 0. or any(np.any(v) for v in staged.values())):
            raise ValueError('Nonzero receptor state at time zero')
        if tick and not np.array_equal(staged['queue'][(tick-1)%len(self.queue)],staged['filtered']):
            raise ValueError('Latest receptor queue entry and filter disagree')
        if tick < len(self.queue) and np.any(staged['queue'][tick:]):
            raise ValueError('Unwritten receptor delay slots must be zero')
        self.filtered = staged['filtered']; self.queue = staged['queue']
        self.tick = tick; self.velocity_lowpass = velocity
