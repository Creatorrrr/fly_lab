"""Transparent CUDA-stage total-current observation of native core adaptation.

Only a lane-0 copy after the existing drive addition is added. Observation
buffers are outside the neural identity and checkpoint. No CPU neural fallback.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from flylab.c.adaptive_fused import SOURCE
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork


def _observed_csr_source(columns, capacity):
    before = 'float* input, int n) {'
    after = ('float* input, const int* observed_col, '
             'unsigned long long* counters, const int* armed, '
             'int* overflow, float* observed, int n) {')
    point = 'value = add_exact(drive[row], value);'
    if SOURCE.count(before) != 1 or SOURCE.count(point) != 1:
        raise RuntimeError('Native CSR observation insertion point changed')
    extra = f'''{point}
        int column = observed_col[row];
        if (armed[0] != 0 && column >= 0) {{
            unsigned long long position = counters[column]++;
            if (position < {capacity}ull)
                observed[position * {columns}ull + column] = value;
            else
                atomicExch(overflow, 1);
        }}'''
    return SOURCE.replace(before, after, 1).replace(point, extra, 1)


class ObservedCoreAdaptationNetwork(CoreAdaptationNetwork):
    """Core neural equations/identity with separately armed GPU current trace."""

    def __init__(self, *args, observation_ids, capacity_stages,
                 core_ids, core_gain, **kwargs):
        import torch

        n = args[0].shape[0]
        ids = np.asarray(observation_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 1
                or len(np.unique(ids)) != len(ids)
                or np.any((ids < 0) | (ids >= n))
                or type(capacity_stages) is not int
                or not 1 <= capacity_stages <= 20000):
            raise ValueError('Existing unique observation rows and bounded capacity required')
        if kwargs.get('device') != 'cuda' or kwargs.get('cuda_implementation') != 'packed':
            raise ValueError('Actual packed CUDA neural implementation required')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA GPU required; no fallback')
        self._construction_stream = torch.cuda.current_stream().cuda_stream
        self.observation_ids = ids.astype(np.int64, copy=True)
        self.capacity_stages = capacity_stages
        mapping = np.full(n, -1, np.int32)
        mapping[self.observation_ids] = np.arange(len(ids), dtype=np.int32)
        self._observed_col = torch.as_tensor(mapping, device='cuda')
        self._observed_counters = torch.zeros(len(ids), dtype=torch.int64,
                                                device='cuda')
        self._observed_armed = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._observed_overflow = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._observed_values = torch.full((capacity_stages, len(ids)),
            float('nan'), dtype=torch.float32, device='cuda')
        self._observation_start_tick = None
        self._observation_end_tick = None
        self._observation_stream = None
        self._observation_active = False
        self._capture_complete = False
        super().__init__(*args, core_ids=core_ids, core_gain=core_gain, **kwargs)
        if self.cuda_implementation != 'packed' or self.graph is None:
            raise ValueError('Native packed CUDA graph required for observation')
        self._capture_complete = True

    def _init_fusion(self):
        super()._init_fusion()
        source = _observed_csr_source(len(self.observation_ids),
                                      self.capacity_stages)
        kernel = self._cupy.RawKernel(source, 'csr_input',
                                      options=('--fmad=false',))
        kernel.compile()
        self._fused_kernels['csr_input'] = kernel

    def _launch_fused(self, name, tensors, scalars=()):
        if name == 'csr_input':
            tensors = (*tensors, self._observed_col, self._observed_counters,
                       self._observed_armed, self._observed_overflow,
                       self._observed_values)
        return super()._launch_fused(name, tensors, scalars)

    def _check_stream(self):
        stream = self.torch.cuda.current_stream().cuda_stream
        if stream != self._construction_stream:
            raise RuntimeError('Native observation requires constructor CUDA stream')
        if self._observation_stream is not None and stream != self._observation_stream:
            raise RuntimeError('Native observation belongs to one CUDA stream')
        return stream

    def _advance(self, drive, steps, *, readout_indices=None):
        self._check_stream()
        return super()._advance(drive, steps,
                                readout_indices=readout_indices)

    def restore(self, saved):
        self._check_stream()
        if self._observation_start_tick is not None:
            raise RuntimeError('Cannot restore after native observation starts')
        return super().restore(saved)

    def arm_observation(self):
        """Reset after graph capture and arm on the single execution stream."""
        if not self._capture_complete or self._observation_start_tick is not None:
            raise RuntimeError('Arm once only after CUDA graph capture')
        self._check_stream()
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._observed_armed.zero_()
        self._observed_counters.zero_()
        self._observed_overflow.zero_()
        self._observed_values.fill_(float('nan'))
        stream.synchronize()
        self._observation_stream = stream.cuda_stream
        self._observation_start_tick = int(self.tick)
        self._observed_armed.fill_(1)
        self._observation_active = True

    def disable_observation(self):
        self._check_stream()
        if self._observation_start_tick is None:
            raise RuntimeError('Native observation was not armed')
        if not self._observation_active:
            return
        self.torch.cuda.current_stream().synchronize()
        self._observed_armed.zero_()
        self.torch.cuda.current_stream().synchronize()
        self._observation_end_tick = int(self.tick)
        self._observation_active = False

    def observation_raw(self):
        """Copy full NaN-padded raw once; counter retains overflow requests."""
        if self._observation_start_tick is None:
            raise RuntimeError('Native observation was not armed')
        self.disable_observation()
        counts = self._observed_counters.cpu().numpy().copy()
        capacity = self.capacity_stages
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {'observation_ids': self.observation_ids.copy(),
                'observer_source_sha256': np.asarray(source_sha),
                'start_tick': np.asarray(self._observation_start_tick,
                                         dtype=np.int64),
                'end_tick': np.asarray(self._observation_end_tick, dtype=np.int64),
                'counter_requested': counts,
                'counter_completed': np.minimum(counts, capacity),
                'capacity_stages': np.asarray(capacity, dtype=np.int64),
                'overflow': self._observed_overflow.cpu().numpy().copy(),
                'armed': self._observed_armed.cpu().numpy().copy(),
                'total_current': self._observed_values.cpu().numpy().copy()}
