"""Passive E150 packed-CUDA signed recurrent input recorder.

The scientific state, identity, CSR reduction, and stage/finish kernels remain
those of MotorTimescaleNetwork. Separate stores never feed back into the net.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

MAX_STAGES = 160_000
MAX_OBSERVATION_BYTES = 1024**3


def _signed_timescale_source(fuse_tanh: bool, columns: int, capacity: int) -> str:
    """Retain original CSR arithmetic; sum signed products in other registers."""
    source = _timescale_source(fuse_tanh)
    before = "float* input, int n) {"
    after = ("float* input, const int* observed_col, "
             "unsigned long long* counters, const int* armed, "
             "int* overflow, float* observed, float* positive_observed, "
             "float* negative_observed, float* drive_observed, int n) {")
    point = "value = add_exact(drive[row], value);"
    loop = "value += lookup[item >> 18] * masked[item & 0x3ffff];"
    reduction = "value += __shfl_down_sync(0xffffffff, value, offset);"
    if any(source.count(s) != 1 for s in (before, point, loop, reduction)):
        raise RuntimeError("E150 native CSR insertion point changed")
    source = source.replace(before, after, 1)
    source = source.replace("float value = 0.0f;", """int column = observed_col[row];
    float positive = 0.0f, negative = 0.0f;
    float value = 0.0f;""", 1)
    source = source.replace(loop, """value += lookup[item >> 18] * masked[item & 0x3ffff];
        if (column >= 0 && armed[0] != 0) {
            float weight = lookup[item >> 18];
            float product = weight * masked[item & 0x3ffff];
            if (weight > 0.0f) positive += product;
            if (weight < 0.0f) negative += -product;
        }""", 1)
    source = source.replace(reduction, """{
        value += __shfl_down_sync(0xffffffff, value, offset);
        positive += __shfl_down_sync(0xffffffff, positive, offset);
        negative += __shfl_down_sync(0xffffffff, negative, offset);
    }""", 1)
    extra = f"""{point}
        if (armed[0] != 0 && column >= 0) {{
            unsigned long long position = counters[column]++;
            if (position < {capacity}ull) {{
                observed[position * {columns}ull + column] = value;
                positive_observed[position * {columns}ull + column] = positive;
                negative_observed[position * {columns}ull + column] = negative;
                drive_observed[position * {columns}ull + column] = drive[row];
            }} else
                atomicExch(overflow, 1);
        }}"""
    return source.replace(point, extra, 1)


class SignedInputChannelNetwork(MotorTimescaleNetwork):
    """E150 network plus four out-of-checkpoint single-use stage buffers."""

    def __init__(self, *args, observation_ids, capacity_stages,
                 core_ids, core_gain, motor_ids, motor_gains,
                 motor_log_tau, **kwargs):
        import torch

        n = args[0].shape[0]
        ids = np.asarray(observation_ids)
        if (ids.ndim != 1 or ids.dtype != np.int64 or len(ids) < 1
                or len(np.unique(ids)) != len(ids)
                or np.any((ids < 0) | (ids >= n))):
            raise ValueError("Ordered unique native int64 observation IDs required")
        if type(capacity_stages) is not int or not 1 <= capacity_stages <= MAX_STAGES:
            raise ValueError("Bounded native stage capacity required")
        bytes_required = 4 * 4 * capacity_stages * len(ids) + 4 * n + 8 * len(ids) + 8
        if bytes_required > MAX_OBSERVATION_BYTES:
            raise ValueError("Native observation exceeds 1 GiB GPU limit")
        if kwargs.get("device") != "cuda" or kwargs.get("cuda_implementation") != "packed":
            raise ValueError("Actual packed CUDA neural implementation required")
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA GPU required; no fallback")

        self._construction_stream = torch.cuda.current_stream().cuda_stream
        self.observation_ids = ids.copy()
        self.capacity_stages = capacity_stages
        self.observation_buffer_bytes = bytes_required
        mapping = np.full(n, -1, np.int32)
        mapping[ids] = np.arange(len(ids), dtype=np.int32)
        self._observed_col = torch.as_tensor(mapping, device="cuda")
        self._observed_counters = torch.zeros(len(ids), dtype=torch.int64, device="cuda")
        self._observed_armed = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._observed_overflow = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._observed_values = torch.full((capacity_stages, len(ids)),
            float("nan"), dtype=torch.float32, device="cuda")
        self._positive_values = torch.full_like(self._observed_values, float("nan"))
        self._negative_values = torch.full_like(self._observed_values, float("nan"))
        self._drive_values = torch.full_like(self._observed_values, float("nan"))
        self._observation_start_tick = None
        self._observation_end_tick = None
        self._observation_stream = None
        self._observation_active = False
        self._capture_complete = False
        super().__init__(*args, core_ids=core_ids, core_gain=core_gain,
            motor_ids=motor_ids, motor_gains=motor_gains,
            motor_log_tau=motor_log_tau, **kwargs)
        if self.cuda_implementation != "packed" or self.graph is None:
            raise RuntimeError("Native packed CUDA graph required")
        self._capture_complete = True

    def _init_fusion(self):
        super()._init_fusion()
        source = _signed_timescale_source(self.fuse_tanh,
            len(self.observation_ids), self.capacity_stages)
        kernel = self._cupy.RawKernel(source, "csr_input", options=("--fmad=false",))
        kernel.compile()
        self._fused_kernels["csr_input"] = kernel

    def _launch_fused(self, name, tensors, scalars=()):
        if name == "csr_input":
            tensors = (*tensors, self._observed_col, self._observed_counters,
                self._observed_armed, self._observed_overflow,
                self._observed_values, self._positive_values,
                self._negative_values, self._drive_values)
        return super()._launch_fused(name, tensors, scalars)

    def _check_stream(self):
        stream = self.torch.cuda.current_stream().cuda_stream
        if stream != self._construction_stream:
            raise RuntimeError("Native observation requires constructor CUDA stream")
        if self._observation_stream is not None and stream != self._observation_stream:
            raise RuntimeError("Native observation belongs to one CUDA stream")
        return stream

    def _advance(self, drive, steps, *, readout_indices=None):
        self._check_stream()
        return super()._advance(drive, steps, readout_indices=readout_indices)

    def restore(self, saved):
        self._check_stream()
        if self._observation_start_tick is not None:
            raise RuntimeError("Cannot restore after native observation starts")
        return super().restore(saved)

    def arm_observation(self):
        if not self._capture_complete or self._observation_start_tick is not None:
            raise RuntimeError("Arm once only after CUDA graph capture")
        self._check_stream()
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._observed_armed.zero_()
        self._observed_counters.zero_()
        self._observed_overflow.zero_()
        self._observed_values.fill_(float("nan"))
        self._positive_values.fill_(float("nan"))
        self._negative_values.fill_(float("nan"))
        self._drive_values.fill_(float("nan"))
        stream.synchronize()
        self._observation_stream = stream.cuda_stream
        self._observation_start_tick = int(self.tick)
        self._observed_armed.fill_(1)
        self._observation_active = True

    def disable_observation(self):
        self._check_stream()
        if self._observation_start_tick is None:
            raise RuntimeError("Native observation was not armed")
        if not self._observation_active:
            return
        stream = self.torch.cuda.current_stream()
        stream.synchronize()
        self._observed_armed.zero_()
        stream.synchronize()
        self._observation_end_tick = int(self.tick)
        self._observation_active = False

    def observation_static_metadata(self):
        return {"motor_ids": self.observation_ids.copy(),
                "start_tick": self._observation_start_tick,
                "end_tick": self._observation_end_tick,
                "capacity_stages": self.capacity_stages,
                "observation_buffer_bytes": self.observation_buffer_bytes,
                "observer_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()}

    def _require_stopped(self):
        self._check_stream()
        if self._observation_end_tick is None or self._observation_active:
            raise RuntimeError("Stop native observation before host readback")

    def copy_values(self, channel="total_current"):
        self._require_stopped()
        buffers = {"total_current": self._observed_values,
                   "positive_recurrent": self._positive_values,
                   "negative_magnitude": self._negative_values,
                   "drive": self._drive_values}
        return buffers[channel].cpu().numpy().copy()

    def copy_counters(self):
        self._require_stopped()
        return self._observed_counters.cpu().numpy().copy()

    def copy_overflow(self):
        self._require_stopped()
        return self._observed_overflow.cpu().numpy().copy()

    def copy_armed(self):
        self._require_stopped()
        return self._observed_armed.cpu().numpy().copy()


def replace_with_signed_input_network(session, capacity_stages=MAX_STAGES):
    """Replace only the restored E150 neural object, preserving whole state."""
    old = session.network
    if type(old) is not MotorTimescaleNetwork:
        raise TypeError("Original E150 MotorTimescaleNetwork required")
    saved = old.snapshot()
    ids = np.asarray(session.adapter.calibration["motor_ids"], dtype=np.int64)
    if ids.shape != (391,) or not np.array_equal(ids, old.motor_ids):
        raise ValueError("E150 native MN roster changed")
    candidate = SignedInputChannelNetwork(source_matrix(old),
        *(getattr(old, key).cpu().numpy().copy()
          for key in ("tau", "a", "threshold", "cap")),
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
        device=old.device, cuda_implementation=old.cuda_implementation,
        capture_steps=old.interval_steps, core_ids=old.core_ids,
        core_gain=old.core_gain, motor_ids=old.motor_ids,
        motor_gains=old.motor_gains, motor_log_tau=old.motor_log_tau,
        observation_ids=ids, capacity_stages=capacity_stages)
    exact(old.identity, candidate.identity, "E208 E150 neural identity")
    candidate.restore(saved)
    exact(saved, candidate.snapshot(), "E208 E150 neural state")
    session.network = candidate
    del old
    return candidate
