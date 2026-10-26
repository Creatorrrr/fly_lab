"""Passive native RK4 stage input recorder for existing BANC motor cells.

The two GPU traces are outside neural identity and checkpoint state. The
original CSR sum, drive addition, stage and finish equations are unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from flylab.c.adaptive_fused import SOURCE
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_native_current import ObservedCoreAdaptationNetwork

MAX_STAGES = 160_000
MAX_OBSERVATION_BYTES = 512 * 1024**2


def _stage_input_csr_source(current_columns, source_columns, capacity):
    """Insert two lane-0 stores after the existing total-current addition."""
    before = 'float* input, int n) {'
    after = (
        'float* input, const int* current_col, '
        'unsigned long long* current_count, const int* source_col, '
        'unsigned long long* source_count, const int* armed, '
        'int* current_overflow, int* source_overflow, '
        'float* current_values, float* source_values, int n) {'
    )
    point = 'value = add_exact(drive[row], value);'
    if SOURCE.count(before) != 1 or SOURCE.count(point) != 1:
        raise RuntimeError('Native CSR insertion point changed')
    extra = f'''{point}
        if (armed[0] != 0) {{
            int column = current_col[row];
            if (column >= 0) {{
                unsigned long long position = current_count[column]++;
                if (position < {capacity}ull)
                    current_values[position * {current_columns}ull + column] = value;
                else
                    atomicExch(current_overflow, 1);
            }}
            column = source_col[row];
            if (column >= 0) {{
                unsigned long long position = source_count[column]++;
                if (position < {capacity}ull)
                    source_values[position * {source_columns}ull + column] = masked[row];
                else
                    atomicExch(source_overflow, 1);
            }}
        }}'''
    return SOURCE.replace(before, after, 1).replace(point, extra, 1)


def _unique_rows(ids, n, label):
    ids = np.asarray(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 1
            or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= n))):
        raise ValueError('Existing unique '+label+' rows required')
    return ids.astype(np.int64, copy=True)


class StageInputNetwork(ObservedCoreAdaptationNetwork):
    """Core dynamics with independent current and masked-source GPU traces."""

    def __init__(self, *args, motor_ids, source_ids, capacity_stages,
                 core_ids, core_gain, **kwargs):
        import torch

        n = args[0].shape[0]
        motors = _unique_rows(motor_ids, n, 'motor')
        sources = _unique_rows(source_ids, n, 'source')
        if type(capacity_stages) is not int or not 1 <= capacity_stages <= MAX_STAGES:
            raise ValueError('Bounded native stage capacity required')
        # Account for both traces and all instrument-only maps/counters/flags.
        bytes_required = (4*capacity_stages*(len(motors)+len(sources))
            + 8*n + 8*(len(motors)+len(sources)) + 12)
        if bytes_required > MAX_OBSERVATION_BYTES:
            raise ValueError('Native observation exceeds 512 MiB GPU buffer limit')
        if kwargs.get('device') != 'cuda' or kwargs.get('cuda_implementation') != 'packed':
            raise ValueError('Actual packed CUDA neural implementation required')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA GPU required; no fallback')

        self._construction_stream = torch.cuda.current_stream().cuda_stream
        self.observation_ids = motors
        self.source_ids = sources
        self.capacity_stages = capacity_stages
        self.observation_buffer_bytes = bytes_required
        current_map = np.full(n, -1, np.int32)
        source_map = np.full(n, -1, np.int32)
        current_map[motors] = np.arange(len(motors), dtype=np.int32)
        source_map[sources] = np.arange(len(sources), dtype=np.int32)
        self._observed_col = torch.as_tensor(current_map, device='cuda')
        self._source_col = torch.as_tensor(source_map, device='cuda')
        self._observed_counters = torch.zeros(len(motors), dtype=torch.int64,
                                              device='cuda')
        self._source_counters = torch.zeros(len(sources), dtype=torch.int64,
                                            device='cuda')
        self._observed_armed = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._observed_overflow = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._source_overflow = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._observed_values = torch.full((capacity_stages, len(motors)),
            float('nan'), dtype=torch.float32, device='cuda')
        self._source_values = torch.full((capacity_stages, len(sources)),
            float('nan'), dtype=torch.float32, device='cuda')
        self._observation_start_tick = None
        self._observation_end_tick = None
        self._observation_stream = None
        self._observation_active = False
        self._capture_complete = False

        # Deliberately bypass E125's 20,000-stage constructor cap while keeping
        # its reviewed one-stream/one-arm/restore/disable lifecycle methods.
        CoreAdaptationNetwork.__init__(self, *args, core_ids=core_ids,
                                       core_gain=core_gain, **kwargs)
        if self.cuda_implementation != 'packed' or self.graph is None:
            raise RuntimeError('Native packed CUDA graph required')
        self._capture_complete = True

    def _init_fusion(self):
        CoreAdaptationNetwork._init_fusion(self)
        source = _stage_input_csr_source(len(self.observation_ids),
            len(self.source_ids), self.capacity_stages)
        kernel = self._cupy.RawKernel(source, 'csr_input',
                                      options=('--fmad=false',))
        kernel.compile()
        self._fused_kernels['csr_input'] = kernel

    def _launch_fused(self, name, tensors, scalars=()):
        if name == 'csr_input':
            tensors = (*tensors, self._observed_col, self._observed_counters,
                       self._source_col, self._source_counters,
                       self._observed_armed, self._observed_overflow,
                       self._source_overflow, self._observed_values,
                       self._source_values)
        return CoreAdaptationNetwork._launch_fused(self, name, tensors, scalars)

    def arm_observation(self):
        if not self._capture_complete or self._observation_start_tick is not None:
            raise RuntimeError('Arm once only after CUDA graph capture')
        self._check_stream()
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._source_counters.zero_()
        self._source_overflow.zero_()
        self._source_values.fill_(float('nan'))
        stream.synchronize()
        super().arm_observation()

    def observation_static_metadata(self):
        """Available even if either GPU readback fails; unknown end stays null."""
        return {'motor_ids': self.observation_ids.copy(),
                'source_ids': self.source_ids.copy(),
                'start_tick': self._observation_start_tick,
                'end_tick': self._observation_end_tick,
                'capacity_stages': self.capacity_stages,
                'observation_buffer_bytes': self.observation_buffer_bytes,
                'observer_source_sha256': hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()}

    def _require_stopped(self):
        self._check_stream()
        if self._observation_end_tick is None or self._observation_active:
            raise RuntimeError('Stop native observation before host readback')

    def copy_current_values(self):
        self._require_stopped()
        return self._observed_values.cpu().numpy().copy()

    def copy_source_values(self):
        self._require_stopped()
        return self._source_values.cpu().numpy().copy()

    def copy_current_counters(self):
        self._require_stopped()
        return self._observed_counters.cpu().numpy().copy()

    def copy_source_counters(self):
        self._require_stopped()
        return self._source_counters.cpu().numpy().copy()

    def copy_current_overflow(self):
        self._require_stopped()
        return self._observed_overflow.cpu().numpy().copy()

    def copy_source_overflow(self):
        self._require_stopped()
        return self._source_overflow.cpu().numpy().copy()

    def copy_armed(self):
        self._require_stopped()
        return self._observed_armed.cpu().numpy().copy()
