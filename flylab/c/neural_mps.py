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
        self.observation_library = torch.mps.compile_shader(Path(__file__).with_name('observation.metal').read_text())
        self._readout_indices = None
        self._readout_device = None
        self._pending_advance = None
        self.submitted_event = torch.mps.Event()
        self._observed_tick = None

    def advance(self, drive, steps, capture=(), pulses=None):
        self.begin_advance(drive, steps, capture, pulses)
        self.finish_advance()

    def begin_advance(self, drive, steps, capture=(), pulses=None):
        """Submit a whole control period, allowing independent CPU body work."""
        if self._pending_advance is not None: raise RuntimeError('Previous neural period still pending')
        self._observed_tick = None
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
            self.capture_device = self.xp.asarray(unique,np.int32)
            self._capture_lookup = {int(j):i for i,j in enumerate(unique)}
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
        self._observed_tick = None
        event_bytes=max(1,steps*len(unique))
        read_start=(8+event_bytes+7)//8*8
        stats_start=read_start+16*len(unique)
        blocks=(self.n+255)//256
        collected=self.torch.empty(stats_start+24*blocks,dtype=self.torch.uint8,device='mps')
        health=collected[:4].view(self.torch.int32);health.zero_()
        events=collected[8:8+event_bytes]
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
        if len(unique):
            self.observation_library.read_state(self.v,self.rate,self.spike_count,self.capture_device,
                collected[read_start:stats_start],len(unique),threads=len(unique))
        self.observation_library.summarize(self.v,self.rate,self.spike_count,collected[stats_start:],self.n,threads=blocks)
        self._pending_advance = (collected,read_start,stats_start,start,steps,capture,unique,inverse)
        # Event.record commits the MPS command buffer without waiting for GPU
        # completion. Merely encoding kernels would postpone work until readback.
        self.submitted_event.record()

    def finish_advance(self):
        if self._pending_advance is None: raise RuntimeError('No pending neural period')
        collected,read_start,stats_start,start,steps,capture,unique,inverse = self._pending_advance
        self._pending_advance = None
        # Health, events, selected state and aggregate statistics share one copy.
        raw=collected.cpu().numpy()
        if int(raw[:4].view(np.int32)[0]): raise RuntimeError('FAULT_NEURAL_NONFINITE')
        self.last_events = []
        if steps and len(capture):
            samples = raw[8:8+steps*len(unique)].reshape(steps, len(unique))[:, inverse]
            times, cells = np.nonzero(samples)
            self.last_events = [{'tick': start+int(t)+1, 'index': int(capture[j])} for t,j in zip(times,cells)]
        self._observed_values=self._unpack_readout(raw[read_start:stats_start],len(unique))
        self._observed_summary=self._unpack_summary(raw[stats_start:])
        self._observed_tick=self.tick

    @staticmethod
    def _unpack_readout(raw,n):
        return dict(voltage_mV=raw[:n*4].view(np.float32).copy(),
                    rate_Hz=raw[n*4:n*8].view(np.float32).copy(),
                    spike_count=raw[n*8:].view(np.int64).copy())

    def readout(self, indices):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before observing it')
        indices=np.asarray(indices,dtype=np.int32)
        if indices.ndim!=1 or (len(indices) and (indices.min()<0 or indices.max()>=self.n)):
            raise ValueError('Invalid readout indices')
        n=len(indices)
        if not n:return dict(voltage_mV=np.empty(0,np.float32),rate_Hz=np.empty(0,np.float32),spike_count=np.empty(0,np.int64))
        if self._observed_tick==self.tick and all(int(j) in self._capture_lookup for j in indices):
            order=np.array([self._capture_lookup[int(j)] for j in indices])
            return {k:v[order].copy() for k,v in self._observed_values.items()}
        key=tuple(indices)
        if key!=self._readout_indices:
            self._readout_indices=key;self._readout_device=self.xp.asarray(indices,np.int32)
        packed=self.torch.empty(n*16,dtype=self.torch.uint8,device='mps')
        self.observation_library.read_state(self.v,self.rate,self.spike_count,self._readout_device,packed,n,threads=n)
        raw=packed.cpu().numpy()
        return self._unpack_readout(raw,n)

    def summary(self):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before observing it')
        if self._observed_tick==self.tick:return dict(self._observed_summary)
        blocks=(self.n+255)//256
        packed=self.torch.empty(blocks*24,dtype=self.torch.uint8,device='mps')
        self.observation_library.summarize(self.v,self.rate,self.spike_count,packed,self.n,threads=blocks)
        raw=packed.cpu().numpy()
        return self._unpack_summary(raw)

    def _unpack_summary(self,raw):
        values=raw.view(np.dtype([('stats','<f4',(4,)),('count','<i8')]))
        return dict(min_voltage_mV=float(values['stats'][:,0].min()),
                    max_voltage_mV=float(values['stats'][:,1].max()),
                    mean_rate_Hz=float(values['stats'][:,2].sum(dtype=np.float64)/self.n),
                    cumulative_spikes=int(values['count'].sum()),simulated_node_count=self.n,tick=self.tick)

    def region_summary(self, groups):
        # One bounded host transfer, rather than a synchronized operation per ROI.
        rates = self.host(self.rate)
        return [dict(region=name, neurons=len(ids), mean_rate_Hz=float(rates[ids].mean()))
                for name, ids in groups.items()]

    def snapshot(self):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before checkpointing')
        state = super().snapshot()
        state['backend_runtime'] = dict(self.runtime_identity)
        return state

    def restore(self, state):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before restoring')
        if state.get('backend_runtime') != self.runtime_identity:
            raise ValueError('MPS runtime or shader differs; explicit backend transfer required')
        result=super().restore(state)
        self._observed_tick=None
        return result
