"""E223 parallel current-conditioned effector with intact neuronal dynamics.

The float64 response is a muscle effector signal, not the firing rate of a
BANC motor neuron. Its outputs never replace the original full-neuron RK state
or the masked rate used by the recurrent CSR calculation.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _same_bits
from tools.banc_m3_native_grouped_response import NativeGroupedResponseNetwork

SCHEMA = "flylab.m3.neural-current-effector.v1"
PRECISION_CONTRACT = (
    "packed-signed-csr-and-original-neuronal-r-z-masked-f32;"
    "parallel-effector-fast-slow-f64;effector-readout-cast-f32;"
    "fixed-motor-original-rate-f32;rk4-dt-f64"
)
ROUTING_CONTRACT = "native-stage-and-finish-output-to-fixed-scratch-v1"


def route_native_destinations(name, tensors, scratch_full, scratch_masked):
    """Replace only E221 native output pointers, preserving every input."""
    routed = list(tensors)
    if name == "native_stage" and len(routed) == 27:
        routed[-3] = scratch_full
        routed[-1] = scratch_masked
    elif name == "native_finish" and len(routed) == 22:
        routed[-1] = scratch_full
    else:
        raise ValueError("Unexpected E221 native kernel argument layout")
    return tuple(routed)


def ordered_effector_from_selected(motor_rate32, motor_ids, free_ids, effector64):
    """Replace free slots in selected original MN rates, preserving their order."""
    selected = np.asarray(motor_rate32)
    motors = np.asarray(motor_ids)
    free = np.asarray(free_ids)
    effector = np.asarray(effector64)
    if (selected.dtype != np.float32 or selected.shape != motors.shape
            or motors.dtype != np.int64 or motors.ndim != 1
            or free.dtype != np.int64 or free.ndim != 1
            or effector.dtype != np.float64 or effector.shape != free.shape
            or not np.isfinite(selected).all() or not np.isfinite(effector).all()
            or np.any(selected < -1e-4) or np.any(effector < 0)
            or len(np.unique(motors)) != len(motors)
            or not np.array_equal(free, motors[np.isin(motors, free)])):
        raise ValueError("Finite ordered neuronal and effector states required")
    cast = effector.astype(np.float32)
    if not np.isfinite(cast).all():
        raise ValueError("Effector exceeds float32 readout precision")
    result = selected.copy()
    result[np.isin(motors, free)] = cast
    return result


def ordered_effector_readout(rate32, motor_ids, free_ids, effector64):
    """Pure full-array helper; runtime uses selected MN values only."""
    full = np.asarray(rate32)
    motors = np.asarray(motor_ids)
    if (full.dtype != np.float32 or full.ndim != 1
            or motors.dtype != np.int64 or motors.ndim != 1
            or np.any((motors < 0) | (motors >= len(full)))):
        raise ValueError("Finite ordered neuronal and effector states required")
    return ordered_effector_from_selected(full[motors], motors,
                                          free_ids, effector64)


def _same_metadata(actual, expected):
    """Strict recursive comparison for immutable inherited checkpoint fields."""
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and _same_bits(actual, expected)
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_same_metadata(actual[key], value)
                        for key, value in expected.items()))
    return type(actual) is type(expected) and actual == expected


class NeuralCurrentEffectorNetwork(NativeGroupedResponseNetwork):
    """E221 response alongside, never substituted for, the original rate."""

    def __init__(self, *args, **kwargs):
        import torch

        if (kwargs.get("device") != "cuda" or kwargs.get("cuda_implementation") != "packed"
                or kwargs.get("dt") != .00025 or not torch.cuda.is_available()):
            raise ValueError("Neural current effector requires packed CUDA dt=.00025")
        n = args[0].shape[0]
        # Allocate before E221's parent captures graph addresses. These are
        # destinations only; the original full rate and masked buffers remain
        # owned by the unchanged neuronal RK stage/finish kernels.
        self._effector_full_scratch = torch.zeros(n, dtype=torch.float32, device="cuda")
        self._effector_masked_scratch = torch.zeros_like(self._effector_full_scratch)
        self._scratch_pointers = (
            self._effector_full_scratch.data_ptr(),
            self._effector_masked_scratch.data_ptr())
        super().__init__(*args, **kwargs)
        self._motor_count = len(self.native_motor_ids)
        self.source_grouped_identity = self.identity
        digest = hashlib.sha256()
        for part in (SCHEMA, PRECISION_CONTRACT, ROUTING_CONTRACT,
                     self.source_timescale_identity, self.source_grouped_identity,
                     self._native_fingerprint):
            digest.update(part.encode("ascii"))
        self.identity = digest.hexdigest()
        self._effector_full_scratch.zero_()
        self._effector_masked_scratch.zero_()
        self._assert_effector_routing()

    def _assert_effector_routing(self):
        full, masked = self._effector_full_scratch, self._effector_masked_scratch
        if ((full.data_ptr(), masked.data_ptr()) != self._scratch_pointers
                or full.data_ptr() == masked.data_ptr()
                or full.dtype != self.torch.float32 or masked.dtype != self.torch.float32
                or tuple(full.shape) != (self.n,) or tuple(masked.shape) != (self.n,)
                or full.device.type != "cuda" or masked.device.type != "cuda"):
            raise ValueError("Captured effector scratch pointer or layout changed")
        if hasattr(self, "rate") and full.data_ptr() == self.rate.data_ptr():
            raise ValueError("Effector scratch aliases original neuronal rate")
        if hasattr(self, "_fused_buffers") and masked.data_ptr() == self._fused_buffers[0].data_ptr():
            raise ValueError("Effector scratch aliases original masked rate")

    def _launch_native(self, name, tensors, step):
        self._assert_effector_routing()
        routed = route_native_destinations(name, tensors,
            self._effector_full_scratch, self._effector_masked_scratch)
        return super()._launch_native(name, routed, step)

    def _assert_hot_path_layout(self):
        """Check captured addresses and shapes without copying GPU parameters."""
        if (any(getattr(self, key).data_ptr() != ptr
                for key, ptr in self._native_pointers.items())
                or self.rate.dtype != self.torch.float32
                or tuple(self.rate.shape) != (self.n,)
                or self._free_rate64.dtype != self.torch.float64
                or tuple(self._free_rate64.shape) != (self._free_count,)
                or self.native_motor_ids.shape != (self._motor_count,)
                or self.free_motor_ids.shape != (self._free_count,)):
            raise ValueError("Captured neuronal or effector hot-path layout changed")
        self._assert_effector_routing()

    def effector_readout(self):
        """Ordered float32 muscle effector signal; never writes network.rate."""
        self._assert_hot_path_layout()
        return ordered_effector_from_selected(
            self.readout(self.native_motor_ids), self.native_motor_ids,
            self.free_motor_ids, self._free_rate64.detach().cpu().numpy().copy())

    def reset(self):
        super().reset()
        self._effector_full_scratch.zero_()
        self._effector_masked_scratch.zero_()

    def snapshot(self):
        self._assert_native_parameters()
        self._assert_effector_routing()
        saved = MotorTimescaleNetwork.snapshot(self)
        saved["neural_current_effector"] = {
            "schema": SCHEMA,
            "precision_contract": PRECISION_CONTRACT,
            "routing_contract": ROUTING_CONTRACT,
            "source_timescale_identity": self.source_timescale_identity,
            "source_grouped_identity": self.source_grouped_identity,
            "parameter_fingerprint": self._native_fingerprint,
            "motor_ids": self.native_motor_ids.copy(),
            "free_ids": self.free_motor_ids.copy(),
            "fixed_ids": self.fixed_motor_ids.copy(),
            "membership": self.membership.copy(),
            "scale": self.scale.copy(),
            "centers": self.centers.copy(),
            "sigma2": self.sigma2.copy(),
            "beta": self.beta.copy(),
            "tau": self.free_tau.copy(),
            "cap": self.free_cap.copy(),
            "inverse_slow": self.inverse_slow.copy(),
            "effector64": self._free_rate64.detach().cpu().numpy().copy(),
            "fast64": self._free_fast64.detach().cpu().numpy().copy(),
            "slow64": self._free_slow64.detach().cpu().numpy().copy(),
        }
        return saved

    def restore(self, saved):
        """Preflight all provenance and state before any live tensor is touched."""
        self._assert_native_parameters()
        self._assert_effector_routing()
        if not isinstance(saved, dict):
            raise TypeError("Complete effector checkpoint dictionary required")
        template = MotorTimescaleNetwork.snapshot(self)
        patch = saved.get("neural_current_effector")
        expected = {
            "schema": SCHEMA,
            "precision_contract": PRECISION_CONTRACT,
            "routing_contract": ROUTING_CONTRACT,
            "source_timescale_identity": self.source_timescale_identity,
            "source_grouped_identity": self.source_grouped_identity,
            "parameter_fingerprint": self._native_fingerprint,
            "motor_ids": self.native_motor_ids,
            "free_ids": self.free_motor_ids,
            "fixed_ids": self.fixed_motor_ids,
            "membership": self.membership,
            "scale": self.scale,
            "centers": self.centers,
            "sigma2": self.sigma2,
            "beta": self.beta,
            "tau": self.free_tau,
            "cap": self.free_cap,
            "inverse_slow": self.inverse_slow,
        }
        state_keys = ("effector64", "fast64", "slow64")
        if (set(saved) != set(template) | {"neural_current_effector"}
                or not isinstance(patch, dict)
                or set(patch) != set(expected) | set(state_keys)
                or any(not _same_metadata(patch[key], value)
                       for key, value in expected.items())):
            raise ValueError("Complete independent effector identity required")
        for key, value in template.items():
            if (key not in ("tick", "rate", "drive", "output_mask", "adaptation")
                    and not _same_metadata(saved[key], value)):
                raise ValueError("Inherited neuronal provenance changed: " + key)
        n = self.n
        if type(saved["tick"]) is not int or saved["tick"] < 0:
            raise ValueError("Invalid neuronal tick")
        for key in ("rate", "drive", "output_mask", "adaptation"):
            value = saved[key]
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != (n,) or not np.isfinite(value).all()):
                raise ValueError("Finite float32 neuronal checkpoint field required: " + key)
        if (np.any(saved["rate"] < -1e-4)
                or np.any(saved["adaptation"] < 0)
                or not np.isin(saved["output_mask"], (0, 1)).all()):
            raise ValueError("Neuronal checkpoint range invalid")
        shapes = ((self._free_count,), (self._free_count, 3), (self._free_count, 3))
        states = tuple(patch[key] for key in state_keys)
        if any(not isinstance(value, np.ndarray) or value.dtype != np.float64
               or value.shape != shape or not np.isfinite(value).all()
               for value, shape in zip(states, shapes)) or np.any(states[0] < 0):
            raise ValueError("Finite independent float64 effector state required")
        # E221 restore asserts rate32 == effector64. That relation is false here;
        # invoke the actual parent timescale restore after complete preflight.
        MotorTimescaleNetwork.restore(self, saved)
        for target, value in zip((self._free_rate64, self._free_fast64,
                                  self._free_slow64), states):
            target.copy_(self.torch.as_tensor(value, device=self.device))


__all__ = ("NeuralCurrentEffectorNetwork", "ordered_effector_from_selected",
           "ordered_effector_readout", "route_native_destinations")
