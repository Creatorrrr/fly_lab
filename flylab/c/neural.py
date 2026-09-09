"""Exponential-current LIF, exact subthreshold integration with fixed input.

This is a C model with literature-informed initial parameters, not a bitwise
reproduction of Shiu et al. h is an mV drive, not a conductance. Events arriving
at tick t enter h before integrating [t,t+1). Spikes are dated at the interval
end, then scheduled at spike_tick + delay_ticks. Refractory cells keep v_reset
while synaptic currents continue to decay and arrive. Old arrivals survive
suppress_spiking/mute_outgoing/mute_edges; only future emissions are altered.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .integrity import digest, bounded_int, finite
from .inputs import validate_input

NEURAL_BACKENDS = ('exp_lif_cpu_reference', 'exp_lif_mps', 'exp_lif_cuda')


def create_backend(graph, parameters=None, backend='exp_lif_cpu_reference'):
    if backend == 'exp_lif_mps':
        from .neural_mps import MetalLIF
        return MetalLIF(graph, parameters)
    return ExpLIF(graph, parameters, backend)


@dataclass(frozen=True)
class LIFParameters:
    dt: float = .0001
    rest_mV: float = -52.
    reset_mV: float = -52.
    threshold_mV: float = -45.
    tau_m_s: float = .020
    tau_syn_s: float = .005
    refractory_s: float = .0022
    delay_s: float = .0018
    rate_tau_s: float = .050
    dtype: str = 'float32'
    integration: str = 'exact-exponential-held-drive-v1'

    def __post_init__(self):
        for key in ('dt', 'tau_m_s', 'tau_syn_s', 'refractory_s', 'delay_s', 'rate_tau_s'):
            finite(getattr(self, key), key, 1e-6, 1.)
        for key in ('rest_mV', 'reset_mV', 'threshold_mV'):
            finite(getattr(self, key), key, -200, 200)
        if self.threshold_mV <= max(self.rest_mV, self.reset_mV):
            raise ValueError('Threshold must exceed rest and reset')
        if self.dtype not in ('float32', 'float64') or self.integration != 'exact-exponential-held-drive-v1':
            raise ValueError('Unsupported LIF precision or integration scheme')
        for value in (self.delay_s, self.refractory_s, .005):
            if abs(value / self.dt - round(value / self.dt)) > 1e-8:
                raise ValueError('Delay, refractory and control period must be integer neural ticks')

    @property
    def hash(self):
        return digest(asdict(self))


class ExpLIF:
    def __init__(self, graph, parameters=None, backend='exp_lif_cpu_reference'):
        self.graph = graph
        self.p = parameters or LIFParameters()
        self.backend = backend
        if backend == 'exp_lif_cpu_reference':
            self.xp = np
            self.W = graph.matrix.astype(self.p.dtype, copy=True)
        elif backend == 'exp_lif_cuda':
            try:
                import cupy as cp
                from cupyx.scipy.sparse import csr_matrix
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError('No CUDA device')
                self.xp = cp
                self.W = csr_matrix((cp.asarray(graph.weights, dtype=self.p.dtype),
                                     cp.asarray(graph.indices), cp.asarray(graph.indptr)),
                                    shape=(graph.n, graph.n))
            except (ImportError, RuntimeError) as e:
                raise RuntimeError('BLOCKED_CUDA: install matching CuPy on an NVIDIA CUDA host') from e
        else:
            raise ValueError('Unknown neural backend: ' + str(backend))
        xp = self.xp
        self.n = graph.n
        self.tick = 0
        self.delay_ticks = round(self.p.delay_s / self.p.dt)
        self.refractory_ticks = round(self.p.refractory_s / self.p.dt)
        self.slots = self.delay_ticks + 1
        self.v = xp.full(self.n, self.p.rest_mV, dtype=self.p.dtype)
        self.h = xp.zeros(self.n, dtype=self.p.dtype)
        self.rate = xp.zeros(self.n, dtype=self.p.dtype)
        self.refractory_until = xp.zeros(self.n, dtype=xp.int64)
        self.queue = xp.zeros((self.slots, self.n), dtype=self.p.dtype)
        self.spike_count = xp.zeros(self.n, dtype=xp.int64)
        self.suppress = xp.zeros(self.n, dtype=bool)
        self.mute = xp.zeros(self.n, dtype=bool)
        self.edge_mutes = []
        self.em = math.exp(-self.p.dt / self.p.tau_m_s)
        self.es = math.exp(-self.p.dt / self.p.tau_syn_s)
        self.er = math.exp(-self.p.dt / self.p.rate_tau_s)
        tm, ts = self.p.tau_m_s, self.p.tau_syn_s
        self.syn_factor = self.p.dt / tm * self.em if tm == ts else ts / (tm - ts) * (self.em - self.es)
        self.last_events = []

    def host(self, array):
        return np.asarray(array) if self.xp is np else self.xp.asnumpy(array)

    def set_interventions(self, suppress=(), mute=(), edges=()):
        # Called only on a control boundary. Updating W does not erase in-flight currents.
        suppress, mute = list(suppress), list(mute)
        for ids in (suppress, mute):
            if any(type(i) is not int or not 0 <= i < self.n for i in ids):
                raise ValueError('Invalid intervention neuron index')
        edges = sorted(set(edges))
        if any(type(i) is not int or not 0 <= i < len(self.graph.weights) for i in edges):
            raise ValueError('Invalid intervention edge index')
        self.suppress[:] = False; self.mute[:] = False
        self.suppress[self.xp.asarray(suppress, dtype=self.xp.int32)] = True
        self.mute[self.xp.asarray(mute, dtype=self.xp.int32)] = True
        if edges != self.edge_mutes:
            self.W.data[:] = self.xp.asarray(self.graph.weights, dtype=self.p.dtype)
            self.W.data[self.xp.asarray(edges, dtype=self.xp.int32)] = 0
            self.edge_mutes = edges

    def advance(self, drive, steps, capture=(), pulses=None):
        drive, capture = validate_input(self.n, drive, steps, capture, pulses)
        xp = self.xp
        x = xp.asarray(drive, dtype=self.p.dtype)
        selected = xp.asarray(capture)
        spike_samples = []
        if pulses is not None:
            pi, pv = pulses
            pi = np.asarray(pi)
            pv = np.asarray(pv)
            if pi.ndim != 1 or pi.dtype.kind not in 'iu' or pv.shape != (steps, len(pi)) or not np.isfinite(pv).all():
                raise ValueError('Invalid external pulse matrix')
            if len(pi) and (pi.min() < 0 or pi.max() >= self.n):
                raise ValueError('Invalid external pulse targets')
            pi, pv = xp.asarray(pi), xp.asarray(pv, dtype=self.p.dtype)
        start = self.tick
        for k in range(steps):
            slot = self.tick % self.slots
            self.h += self.queue[slot]
            self.queue[slot].fill(0)
            if pulses is not None:
                xp.add.at(self.h, pi, pv[k])
            eligible = self.tick >= self.refractory_until
            v_next = self.p.rest_mV + x + (self.v - self.p.rest_mV - x) * self.em + self.h * self.syn_factor
            self.h *= self.es
            self.v[:] = xp.where(eligible, v_next, self.p.reset_mV)
            spikes = eligible & (self.v >= self.p.threshold_mV) & ~self.suppress
            self.tick += 1
            emitted = (spikes & ~self.mute).astype(self.p.dtype)
            # CPU can skip an empty emission. CUDA always launches SpMV, without
            # device -> host transfers or scalar synchronizations inside a tick.
            if xp is not np or np.any(emitted):
                self.queue[(self.tick + self.delay_ticks) % self.slots] += self.W @ emitted
            self.v[:] = xp.where(spikes, self.p.reset_mV, self.v)
            self.refractory_until[:] = xp.where(spikes, self.tick + self.refractory_ticks, self.refractory_until)
            self.rate *= self.er
            self.rate += spikes * ((1 - self.er) / self.p.dt)
            self.spike_count += spikes
            if len(capture):
                spike_samples.append(spikes[selected].copy())
        # A single bounded transfer at the control boundary, never per substep.
        self.last_events = []
        if spike_samples:
            events = self.host(xp.stack(spike_samples))
            times, cells = np.nonzero(events)
            self.last_events = [{'tick': start + int(t) + 1, 'index': int(capture[j])} for t, j in zip(times, cells)]
        if not bool(self.host(xp.isfinite(self.v).all() & xp.isfinite(self.h).all() & xp.isfinite(self.rate).all())):
            raise RuntimeError('FAULT_NEURAL_NONFINITE')

    def readout(self, indices):
        i = self.xp.asarray(indices, dtype=self.xp.int32)
        return dict(voltage_mV=self.host(self.v[i]).copy(), rate_Hz=self.host(self.rate[i]).copy(),
                    spike_count=self.host(self.spike_count[i]).copy())

    def summary(self):
        xp = self.xp
        stats = self.host(xp.stack([self.v.min(), self.v.max(), self.rate.mean()]))
        count = int(self.host(self.spike_count.sum()))
        return dict(min_voltage_mV=float(stats[0]), max_voltage_mV=float(stats[1]),
                    mean_rate_Hz=float(stats[2]), cumulative_spikes=count,
                    simulated_node_count=self.n, tick=self.tick)

    def snapshot(self):
        return dict(backend=self.backend, parameter_hash=self.p.hash, graph_hash=self.graph.hash,
                    tick=self.tick, slot=self.tick % self.slots, edge_mutes=list(self.edge_mutes),
                    **{k: self.host(getattr(self, k)).copy() for k in
                       ('v', 'h', 'rate', 'refractory_until', 'queue', 'spike_count', 'suppress', 'mute')})

    def region_summary(self, groups):
        xp = self.xp
        means = [xp.mean(self.rate[xp.asarray(ids)]) for ids in groups.values()]
        values = self.host(xp.stack(means)) if means else np.empty(0)
        return [dict(region=name, neurons=len(groups[name]), mean_rate_Hz=float(v))
                for name, v in zip(groups, values)]

    def restore(self, state):
        if state.get('backend') != self.backend or state.get('parameter_hash') != self.p.hash or state.get('graph_hash') != self.graph.hash:
            raise ValueError('Neural checkpoint identity mismatch')
        tick = bounded_int(state.get('tick'), 'neural tick')
        if state.get('slot') != tick % self.slots:
            raise ValueError('Delay queue slot mismatch')
        validated = {}
        for key in ('v', 'h', 'rate', 'refractory_until', 'queue', 'spike_count', 'suppress', 'mute'):
            a = np.asarray(state.get(key))
            dest = getattr(self, key)
            dtype = ('bool' if key in ('suppress', 'mute') else
                     'int64' if key in ('refractory_until', 'spike_count') else self.p.dtype)
            if a.shape != dest.shape or a.dtype != np.dtype(dtype) or not np.isfinite(a).all():
                raise ValueError('Invalid neural checkpoint array: ' + key)
            validated[key] = a
        if np.any(validated['rate'] < 0) or np.any(validated['spike_count'] < 0):
            raise ValueError('Negative neural rates or spike counters')
        if np.any(validated['rate'] > 1 / self.p.dt + 1) or np.any(validated['spike_count'] > tick):
            raise ValueError('Neural rate or spike count out of range')
        refractory = validated['refractory_until']
        if np.any(refractory < 0) or np.any(refractory > tick + self.refractory_ticks):
            raise ValueError('Refractory timer outside reachable range')
        if np.max(np.abs(validated['v'])) > 1e6 or np.max(np.abs(validated['h'])) > 1e7 or np.max(np.abs(validated['queue'])) > 1e7:
            raise ValueError('Neural checkpoint drive out of range')
        edges = state.get('edge_mutes')
        if not isinstance(edges, list) or any(type(i) is not int or not 0 <= i < len(self.graph.weights) for i in edges):
            raise ValueError('Invalid edge mask')
        self.set_interventions(np.flatnonzero(validated['suppress']).tolist(), np.flatnonzero(validated['mute']).tolist(), edges)
        for key, a in validated.items():
            getattr(self, key)[:] = self.xp.asarray(a)
        self.tick = tick
        self.last_events = []
