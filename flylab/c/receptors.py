"""Inspectable joint receptor hypotheses for calibration, without neural IDs.

Claw and hook proxies distinguish raw joint-coordinate directions. Their
anatomical flexion/extension polarity is deliberately left unassigned. The
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


class JointReceptors:
    def __init__(self, parameters):
        self.p = parameters; self.tick = 0; self.velocity_lowpass = 0.
        self.filtered = np.zeros(len(CHANNELS))
        self.queue = np.zeros((parameters.delay_steps+1, len(CHANNELS)))
        self.metadata = dict(schema='flylab.joint-receptors.v1', parameters=asdict(parameters),
            channels=list(CHANNELS), unit='normalized proxy 0..1', neural_targets=[],
            evidence_status='engineering hypotheses; physiological calibration pending',
            polarity='positive/negative model q; anatomical polarity unassigned',
            club_model='abs(velocity - lowpass_velocity); no measured frequency tuning',
            source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC8665017/', biological_validation=False)
        self.identity = digest(self.metadata)

    def step(self, q, velocity, *, support_bw=0., contact_bw=0., enabled=True):
        q = finite(q, 'receptor angle', -1000., 1000.)
        velocity = finite(velocity, 'receptor velocity', -1e6, 1e6)
        support_bw = finite(support_bw, 'support load', 0., 1e6)
        contact_bw = finite(contact_bw, 'non-support contact', 0., 1e6)
        boolean(enabled, 'receptors enabled')
        bounded_int(self.tick+1, 'receptor tick')
        p = self.p
        angle = float(np.clip((q-p.angle_low_rad)/(p.angle_high_rad-p.angle_low_rad), 0., 1.))
        raw = np.array([angle, 1-angle, max(0., velocity)/p.velocity_scale_rad_s,
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
        return dict(tick=self.tick, seconds=self.tick*p.dt, profile_hash=self.identity,
            raw=dict(zip(CHANNELS, raw.tolist())), filtered=dict(zip(CHANNELS, self.filtered.tolist())),
            output=dict(zip(CHANNELS, output.tolist())), enabled=enabled,
            stimulus=dict(q_rad=q, velocity_rad_s=velocity, support_bw=support_bw, contact_bw=contact_bw))

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
