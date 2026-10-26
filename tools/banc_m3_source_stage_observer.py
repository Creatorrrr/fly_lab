"""E268 passive native stage acquisition on the actual E267 neural subclass.

Observer buffers are outside scientific state and identity. No body, teacher
runtime input, interpolation, masked-rate history substitution or fitter lives
here. Source histories are read before the existing E267 history RK4 updates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools.banc_m3_source_filtered_synapses import (
    SourceFilteredSynapseNetwork,
    checked_selection,
    source_filtered_cuda_source,
)

MAX_STAGES = 160_000
MAX_OBSERVATION_BYTES = 512 * 1024**2


def observation_plan(n, motor_ids, source_ids, native_motor_ids, native_source_ids,
                     capacity_stages):
    """Pure CPU validation and exact instrument-only GPU byte accounting."""
    if type(n) is not int or not 1 <= n <= 2**18:
        raise ValueError("Native packed neuron count required")
    if type(capacity_stages) is not int or not 1 <= capacity_stages <= MAX_STAGES:
        raise ValueError("Bounded integer stage capacity required")
    motors, sources = np.asarray(motor_ids), np.asarray(source_ids)
    for label, ids, expected in (("motor", motors, native_motor_ids),
                                  ("source", sources, native_source_ids)):
        expected = np.asarray(expected)
        if (ids.ndim != 1 or ids.dtype != np.int64 or not len(ids)
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
                or np.any(ids >= n) or expected.dtype != np.int64
                or expected.shape != ids.shape or not np.array_equal(ids, expected)):
            raise ValueError(f"Exact ordered original int64 {label} roster required")
    if np.any(sources[1:] <= sources[:-1]):
        raise ValueError("Exact sorted E267 unique source roster required")
    m, p, s = len(motors), len(sources), capacity_stages
    sizes = {"motor_map": 4*n, "source_map": 4*n,
        "motor_counts": 8*m, "source_counts": 8*p, "armed": 4, "overflow": 4,
        "total_input": 4*s*m, "source_rate": 4*s*p,
        "source_mask": 4*s*p, "source_history": 16*s*p}
    total = sum(sizes.values())
    if total > MAX_OBSERVATION_BYTES:
        raise ValueError("Native observation exceeds 512 MiB GPU buffer limit")
    motor_map, source_map = np.full(n, -1, np.int32), np.full(n, -1, np.int32)
    motor_map[motors] = np.arange(m, dtype=np.int32)
    source_map[sources] = np.arange(p, dtype=np.int32)
    return {"motor_ids": motors.copy(), "source_ids": sources.copy(),
            "capacity_stages": s, "buffer_bytes": sizes,
            "observation_buffer_bytes": total,
            "motor_map": motor_map, "source_map": source_map}


def observation_status(motor_counts, source_counts, overflow, capacity, expected_stages):
    """Pure CPU common-prefix adjudication; counts retain attempted stage stores."""
    if (type(capacity) is not int or not 1 <= capacity <= MAX_STAGES
            or type(expected_stages) is not int or expected_stages < 0):
        raise ValueError("Valid capacity and expected native stage count required")
    counts = [np.asarray(motor_counts), np.asarray(source_counts)]
    flag = np.asarray(overflow)
    if (any(v.ndim != 1 or v.dtype != np.int64 or not len(v) or np.any(v < 0)
            for v in counts) or flag.shape != (1,) or flag.dtype != np.int32
            or flag[0] not in (0, 1)):
        raise ValueError("Native nonnegative int64 counts and int32 overflow required")
    valid = [np.minimum(v, capacity) for v in counts]
    prefix = min(int(v.min()) for v in valid)
    complete = (int(flag[0]) == 0 and expected_stages <= capacity
                and all(np.all(v == expected_stages) for v in counts))
    return {"complete": bool(complete), "expected_stages": expected_stages,
            "valid_prefix_stages": prefix,
            "motor_valid_counts": valid[0].copy(), "source_valid_counts": valid[1].copy()}


def observed_source_cuda_source(fuse_tanh, motor_columns, source_columns, capacity):
    """Add guarded lane-zero stores immediately after existing drive addition."""
    for value in (motor_columns, source_columns):
        if type(value) is not int or not 1 <= value <= 2**18:
            raise ValueError("Positive bounded native observation column count required")
    if type(capacity) is not int or not 1 <= capacity <= MAX_STAGES:
        raise ValueError("Bounded native stage capacity required")
    source = source_filtered_cuda_source(fuse_tanh)
    before = "const double* history, const float* rate, const float* mask, int n) {"
    after = ("const double* history, const float* rate, const float* mask, "
        "const int* motor_map, const int* source_map, "
        "unsigned long long* motor_counts, unsigned long long* source_counts, "
        "const int* armed, int* overflow, float* total_input, "
        "float* source_rate, float* source_mask, double* source_history, int n) {")
    point = "value = add_exact(drive[row], value);"
    if source.count(before) != 1 or source.count(point) != 1:
        raise RuntimeError("E267 native CSR observer insertion point changed")
    stores = f"""{point}
        if (armed[0] != 0) {{
            int column = motor_map[row];
            if (column >= 0) {{
                unsigned long long position = motor_counts[column]++;
                if (position < {capacity}ull)
                    total_input[position * {motor_columns}ull + column] = value;
                else
                    atomicExch(overflow, 1);
            }}
            column = source_map[row];
            if (column >= 0) {{
                unsigned long long position = source_counts[column]++;
                if (position < {capacity}ull) {{
                    unsigned long long offset = position * {source_columns}ull + column;
                    source_rate[offset] = rate[row];
                    source_mask[offset] = mask[row];
                    source_history[2*offset] = history[2*column];
                    source_history[2*offset+1] = history[2*column+1];
                }} else
                    atomicExch(overflow, 1);
            }}
        }}"""
    return source.replace(before, after, 1).replace(point, stores, 1)


class SourceStageObserverNetwork(SourceFilteredSynapseNetwork):
    """Single-use E267 native acquisition; scientific checkpoint remains E267."""

    def __init__(self, *args, observation_ids, source_ids, capacity_stages, **kwargs):
        import torch

        selection = checked_selection(args[0], kwargs["edge_ids"],
            kwargs["coefficients"], kwargs["free_motor_ids"])
        plan = observation_plan(args[0].shape[0], observation_ids, source_ids,
            kwargs["motor_ids"], selection["source_ids"], capacity_stages)
        if kwargs.get("device") != "cuda" or kwargs.get("cuda_implementation") != "packed":
            raise ValueError("Actual packed CUDA observer required; no CPU fallback")
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA GPU required; no fallback")
        self._construction_stream = torch.cuda.current_stream().cuda_stream
        self._observation_plan = plan
        self.capacity_stages = plan["capacity_stages"]
        self.observation_buffer_bytes = plan["observation_buffer_bytes"]
        m, p, s = len(plan["motor_ids"]), len(plan["source_ids"]), self.capacity_stages
        with torch.inference_mode(False):
            self._observer_buffers = {
                "motor_map": torch.as_tensor(plan["motor_map"].copy(), device="cuda"),
                "source_map": torch.as_tensor(plan["source_map"].copy(), device="cuda"),
                "motor_counts": torch.zeros(m, dtype=torch.int64, device="cuda"),
                "source_counts": torch.zeros(p, dtype=torch.int64, device="cuda"),
                "armed": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "overflow": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "total_input": torch.full((s, m), float("nan"), dtype=torch.float32, device="cuda"),
                "source_rate": torch.full((s, p), float("nan"), dtype=torch.float32, device="cuda"),
                "source_mask": torch.full((s, p), float("nan"), dtype=torch.float32, device="cuda"),
                "source_history": torch.full((s, p, 2), float("nan"),
                    dtype=torch.float64, device="cuda")}
        self._observer_specs = {key: (value.data_ptr(), value.shape, value.dtype, value.device)
            for key, value in self._observer_buffers.items()}
        if sum(v.numel()*v.element_size() for v in self._observer_buffers.values()) != (
                self.observation_buffer_bytes):
            raise RuntimeError("Observer allocation accounting mismatch")
        self._observation_start_tick = None
        self._observation_end_tick = None
        self._observation_active = False
        self._capture_complete = False
        super().__init__(*args, **kwargs)
        if self.cuda_implementation != "packed" or self.graph is None:
            raise RuntimeError("Actual native packed graph required")
        self._capture_complete = True

    def _init_fusion(self):
        super()._init_fusion()
        source = observed_source_cuda_source(self.fuse_tanh,
            len(self._observation_plan["motor_ids"]), len(self._observation_plan["source_ids"]),
            self.capacity_stages)
        kernel = self._cupy.RawKernel(source, "csr_input", options=("--fmad=false",))
        kernel.compile()
        self._fused_kernels["csr_input"] = kernel

    def _launch_fused(self, name, tensors, scalars=()):
        if name == "csr_input":
            tensors = (*tensors, *(self._observer_buffers[key] for key in (
                "motor_map", "source_map", "motor_counts", "source_counts", "armed",
                "overflow", "total_input", "source_rate", "source_mask", "source_history")))
        return super()._launch_fused(name, tensors, scalars)

    def _check_observer(self):
        if self.torch.cuda.current_stream().cuda_stream != self._construction_stream:
            raise RuntimeError("Native observation requires constructor CUDA stream")
        if self._observer_buffers.keys() != self._observer_specs.keys():
            raise RuntimeError("Captured observer buffer inventory changed")
        for key, spec in self._observer_specs.items():
            value = self._observer_buffers[key]
            if (value.data_ptr(), value.shape, value.dtype, value.device) != spec:
                raise RuntimeError("Captured observer buffer pointer or layout changed")

    def _step_fused(self):
        if self._capture_complete:
            self._check_observer()
        return super()._step_fused()

    def _advance(self, drive, steps, *, readout_indices=None):
        self._check_observer()
        return super()._advance(drive, steps, readout_indices=readout_indices)

    def set_muted(self, indices=()):
        self._check_observer()
        return super().set_muted(indices)

    def snapshot(self):
        self._check_observer()
        return super().snapshot()

    def restore(self, saved):
        self._check_observer()
        if self._observation_start_tick is not None:
            raise RuntimeError("Cannot restore after native observation starts")
        return super().restore(saved)

    def reset(self):
        self._check_observer()
        if self._observation_start_tick is not None:
            raise RuntimeError("Cannot reset after native observation starts")
        return super().reset()

    def arm_observation(self):
        self._check_observer()
        if not self._capture_complete or self._observation_start_tick is not None:
            raise RuntimeError("Arm once only after native graph capture")
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._assert_parameters()
        for key in ("motor_map", "source_map"):
            if not np.array_equal(self._observer_buffers[key].cpu().numpy(),
                                  self._observation_plan[key]):
                raise RuntimeError("Owned observer mapping values changed")
        for key in ("armed", "overflow", "motor_counts", "source_counts"):
            self._observer_buffers[key].zero_()
        for key in ("total_input", "source_rate", "source_mask", "source_history"):
            self._observer_buffers[key].fill_(float("nan"))
        stream.synchronize()
        self._observation_start_tick = int(self.tick)
        self._observation_active = True
        self._observer_buffers["armed"].fill_(1)

    def disable_observation(self):
        self._check_observer()
        if self._observation_start_tick is None:
            raise RuntimeError("Native observation has not been armed")
        if not self._observation_active:
            return
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._observer_buffers["armed"].zero_()
        stream.synchronize()
        self._observation_end_tick = int(self.tick)
        self._observation_active = False

    def observation_static_metadata(self):
        plan = self._observation_plan
        return {"motor_ids": plan["motor_ids"].copy(), "source_ids": plan["source_ids"].copy(),
            "capacity_stages": self.capacity_stages,
            "observation_buffer_bytes": self.observation_buffer_bytes,
            "buffer_bytes": plan["buffer_bytes"].copy(),
            "start_tick": self._observation_start_tick, "end_tick": self._observation_end_tick,
            "stage_order": ["start", "half1", "half2", "end"],
            "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def export_observation(self, *, expected_stages=None):
        """Read raw capacity buffers plus attempted counts and safe common prefix.

        Caller owns persistence. After overflow or a scientific advance error,
        disable first; export reports partial status without discarding raw data.
        Pass the planned stage count to assess an interrupted acquisition against
        its requested extent; the default assesses completed logical ticks.
        """
        self._check_observer()
        if self._observation_active or self._observation_end_tick is None:
            raise RuntimeError("Stop native observation before export")
        values = {key: value.cpu().numpy().copy() for key, value in self._observer_buffers.items()}
        logical = 4*(self._observation_end_tick-self._observation_start_tick)
        expected = logical if expected_stages is None else expected_stages
        status = observation_status(values["motor_counts"], values["source_counts"],
            values["overflow"], self.capacity_stages, expected)
        return {**self.observation_static_metadata(), **values, **status,
                "logical_stages": logical}
