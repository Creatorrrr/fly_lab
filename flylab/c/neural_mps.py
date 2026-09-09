"""Apple GPU LIF: persistent MPS tensors and two ordered Metal kernels per tick.

CSR rows are reduced serially in their original order, without atomic float
addition. This retains the CPU summation order. The CPU reference remains a
separate backend; sparse MPS operations never silently fall back to the CPU.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .neural import ExpLIF, LIFParameters
from .integrity import bounded_int, file_hash


class _MPSArrays:
    """Only the array helpers needed by shared readout/checkpoint operations."""
    int32 = np.int32

    def __init__(self, torch): self.torch = torch

    def asarray(self, value, dtype=None):
        t = self.torch
        if isinstance(value, t.Tensor):
            if dtype is None: return value.contiguous()
            return value.to(dtype=t.from_numpy(np.empty(0, dtype=dtype)).dtype).contiguous()
        return t.from_numpy(np.array(value, dtype=dtype, copy=True, order='C')).to('mps')

    def asnumpy(self, value): return value.detach().cpu().numpy()

    def stack(self, arrays): return self.torch.stack(arrays)


class MetalLIF(ExpLIF):
    def __init__(self, graph, parameters=None):
        p = parameters or LIFParameters()
        if p.dtype != 'float32': raise ValueError('MPS backend requires float32; no silent precision conversion')
        if graph.n >= 2**31 or len(graph.weights) >= 2**31: raise ValueError('MPS CSR indices exceed int32 capacity')
        try:
            import torch
            if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
                raise RuntimeError('MPS shader runtime unavailable')
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('BLOCKED_MPS: install requirements-mps.txt on Apple silicon') from exc
        super().__init__(graph, p)
        outgoing = self.W.tocsc()
        self.backend = 'exp_lif_mps'
        self.torch = torch
        self.xp = _MPSArrays(torch)
        for key in ('v', 'h', 'rate', 'refractory_until', 'queue', 'spike_count', 'suppress', 'mute'):
            setattr(self, key, self.xp.asarray(getattr(self, key)))
        self.W = SimpleNamespace(data=self.xp.asarray(graph.weights, np.float32))
        self.indptr = self.xp.asarray(graph.indptr, np.int32)
        self.indices = self.xp.asarray(graph.indices, np.int32)
        self.out_ptr = self.xp.asarray(outgoing.indptr, np.int32)
        self.out_indices = self.xp.asarray(outgoing.indices, np.int32)
        self.active_rows = torch.zeros(self.n, dtype=torch.int32, device='mps')
        self.emitted = torch.zeros(self.n, dtype=torch.uint8, device='mps')
        self.capture_map = torch.full((self.n,), -1, dtype=torch.int32, device='mps')
        self.capture_cache = None
        self.empty_pulse_ptr = torch.zeros(self.n+1, dtype=torch.int32, device='mps')
        self.empty_pulses = torch.zeros(1, dtype=torch.float32, device='mps')
        self.constants = self.xp.asarray([p.rest_mV, p.reset_mV, p.threshold_mV,
                                         self.em, self.es, self.er, self.syn_factor,
                                         (1-self.er)/p.dt], np.float32)
        self.kernel_path = Path(__file__).with_name('lif.metal')
        self.library = torch.mps.compile_shader(self.kernel_path.read_text())
        self.runtime_identity = dict(torch=torch.__version__, shader_sha256=file_hash(self.kernel_path),
                                     numerical_policy='float32-serial-csr-no-fma-v1')

    def advance(self, drive, steps, capture=(), pulses=None):
        bounded_int(steps, 'neural steps', 0, 10000)
        drive = np.asarray(drive)
        if drive.shape != (self.n,) or not np.isfinite(drive).all() or np.max(np.abs(drive)) > 1000:
            raise ValueError('Finite mV input per included neuron required')
        capture = np.asarray(capture, dtype=np.int32)
        if capture.ndim != 1 or len(capture) > 1024 or (len(capture) and (capture.min() < 0 or capture.max() >= self.n)):
            raise ValueError('Invalid bounded readout')
        # Keep duplicate capture indices legal, as in the CPU reference.
        unique, inverse = np.unique(capture, return_inverse=True)
        if self.capture_cache != tuple(unique):
            mapping = np.full(self.n, -1, dtype=np.int32)
            mapping[unique] = np.arange(len(unique), dtype=np.int32)
            self.capture_map = self.xp.asarray(mapping)
            self.capture_cache = tuple(unique)
        pulse_ptr, pv, pulse_width = self.empty_pulse_ptr, self.empty_pulses, 0
        if pulses is not None:
            pi, values = map(np.asarray, pulses)
            if pi.ndim != 1 or pi.dtype.kind not in 'iu' or values.shape != (steps, len(pi)) or not np.isfinite(values).all():
                raise ValueError('Invalid external pulse matrix')
            if len(pi) and (pi.min() < 0 or pi.max() >= self.n): raise ValueError('Invalid external pulse targets')
            if len(pi):
                # Stable order preserves repeated additions to the same neuron.
                order = np.argsort(pi, kind='stable')
                ptr = np.zeros(self.n+1, dtype=np.int32)
                ptr[1:] = np.cumsum(np.bincount(pi.astype(np.int64), minlength=self.n))
                pulse_ptr = self.xp.asarray(ptr)
                pv = self.xp.asarray(values[:, order], np.float32)
                pulse_width = len(pi)
        x = self.xp.asarray(drive, np.float32)
        events = self.torch.empty(max(1, steps*len(unique)), dtype=self.torch.uint8, device='mps')
        health = self.torch.zeros(1, dtype=self.torch.int32, device='mps')
        start = self.tick
        for k in range(steps):
            slot = self.tick % self.slots
            self.library.integrate(self.v, self.h, self.rate, self.refractory_until,
                self.spike_count, self.queue, self.suppress, self.mute, self.emitted,
                x, self.constants, self.capture_map, events, pulse_ptr, pv, health,
                self.out_ptr, self.out_indices, self.active_rows,
                self.n, self.tick, slot, self.refractory_ticks, k, len(unique), pulse_width,
                threads=self.n)
            # With slots=delay_ticks+1, the just-consumed slot is also the
            # destination for emissions at the end of this interval.
            self.library.propagate(self.indptr, self.indices, self.W.data, self.emitted, self.active_rows,
                                   self.queue, self.n, slot, threads=self.n)
            self.tick += 1
        # The first host read synchronizes all queued work. No substep readback.
        if int(health.cpu()[0]): raise RuntimeError('FAULT_NEURAL_NONFINITE')
        self.last_events = []
        if steps and len(capture):
            samples = events.cpu().numpy().reshape(steps, len(unique))[:, inverse]
            times, cells = np.nonzero(samples)
            self.last_events = [{'tick': start+int(t)+1, 'index': int(capture[j])} for t,j in zip(times,cells)]

    def region_summary(self, groups):
        # One bounded host transfer, rather than a synchronized operation per ROI.
        rates = self.host(self.rate)
        return [dict(region=name, neurons=len(ids), mean_rate_Hz=float(rates[ids].mean()))
                for name, ids in groups.items()]

    def snapshot(self):
        state = super().snapshot()
        state['backend_runtime'] = dict(self.runtime_identity)
        return state

    def restore(self, state):
        if state.get('backend_runtime') != self.runtime_identity:
            raise ValueError('MPS runtime or shader differs; explicit backend transfer required')
        return super().restore(state)
