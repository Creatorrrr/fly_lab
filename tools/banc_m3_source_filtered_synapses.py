"""E267 source-resolved causal filters on existing CSR edges only.

This engineering hypothesis neither supplies a gait controller nor establishes
biological physiology. CUDA uses the original packed warp CSR and RK4 kernels.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from scipy.sparse import csr_matrix

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source

SCHEMA = "flylab.m3-source-filtered-synapses.v1"
TAU = np.array([.005, .040], np.float64)
PRECISION = ("coeff:f64-checked-to-f32;history:f64;history-input:stage-f32;"
             "selected:((c0*r+c1*f32(hfast))+c2*f32(hslow))*mask;"
             "selected-products-and-sums:f32-rn-ftz;mask:f32-rn-no-ftz;"
             "csr:original-warp-order-f32-ftz;neutral:original-expression;"
             "history-rk4:f64-no-fma-before-rate-overwrite")


def checked_selection(weights, edge_ids, coefficients, free_motor_ids):
    """Validate without canonicalizing topology or silently casting inputs."""
    if (not isinstance(weights, csr_matrix) or weights.dtype != np.float32
            or weights.shape[0] != weights.shape[1] or not weights.shape[0]
            or not weights.has_canonical_format or not weights.has_sorted_indices
            or not np.isfinite(weights.data).all()):
        raise ValueError("Canonical finite square float32 CSR required")
    weights.check_format(full_check=True)
    ids, c, free = map(np.asarray, (edge_ids, coefficients, free_motor_ids))
    n = weights.shape[0]
    for name, values, bound in (("edge", ids, weights.nnz), ("free MN", free, n)):
        if (values.ndim != 1 or values.dtype != np.int64 or not len(values)
                or np.any(values < 0) or np.any(values >= bound)
                or np.any(values[1:] <= values[:-1])):
            raise ValueError(f"Sorted unique in-range native int64 {name} IDs required")
    if (c.shape != (len(ids), 3) or c.dtype != np.float64
            or not np.isfinite(c).all() or np.any(c < 0)
            or np.any(c.sum(axis=1) < .125) or np.any(c.sum(axis=1) > 8)):
        raise ValueError("Nonnegative finite float64 triples with sum in [.125,8] required")
    c32 = c.astype(np.float32)
    sums = c32.astype(np.float64).sum(axis=1)
    if (not np.isfinite(c32).all() or np.any(c32 < 0)
            or np.any(sums < .125) or np.any(sums > 8)):
        raise ValueError("Converted coefficient triples exceed bounds")
    posts = np.searchsorted(weights.indptr, ids, side="right") - 1
    pres = weights.indices[ids].astype(np.int64)
    if not np.isin(posts, free).all() or np.any(weights.data[ids] == 0):
        raise ValueError("Selected original nonzero edges must target provided free MNs")
    sources = np.unique(pres)
    return {"edge_ids": ids.copy(), "coefficients64": c.copy(),
            "coefficients32": c32.copy(), "free_motor_ids": free.copy(),
            "pre_ids": pres.copy(), "post_ids": posts.astype(np.int64),
            "weights32": weights.data[ids].copy(), "source_ids": sources,
            "edge_source_slots": np.searchsorted(sources, pres).astype(np.int32)}


def _ftz(value):
    value = np.float32(value)
    if abs(value) < np.finfo(np.float32).tiny:
        return np.copysign(np.float32(0), value)
    return value


def _mul(a, b):
    return _ftz(_ftz(a) * _ftz(b))


def _add(a, b):
    return _ftz(_ftz(a) + _ftz(b))


def filtered_edge_value(weight, rate, mask, history, coefficients):
    """Scalar CUDA-expression oracle including CSR FTZ and neutral branch."""
    c = np.asarray(coefficients, np.float32)
    if np.array_equal(c, np.array([1, 0, 0], np.float32)):
        return _mul(weight, np.float32(np.float32(rate) * np.float32(mask)))
    fast, slow = np.asarray(history, np.float64).astype(np.float32)
    mixture = _add(_add(_mul(c[0], rate), _mul(c[1], fast)), _mul(c[2], slow))
    return _mul(weight, np.float32(mixture * np.float32(mask)))


def packed_current_oracle(weights, rate, mask, selection, history):
    """Independent host scalar fixed-warp CSR reduction, in original edge order."""
    selected = {int(e): i for i, e in enumerate(selection["edge_ids"])}
    result = np.zeros(weights.shape[0], np.float32)
    for row in range(len(result)):
        lanes = np.zeros(32, np.float32)
        start, end = weights.indptr[row:row + 2]
        for lane in range(32):
            for edge in range(start + lane, end, 32):
                pre = weights.indices[edge]
                if edge in selected:
                    i = selected[edge]
                    value = filtered_edge_value(weights.data[edge], rate[pre], mask[pre],
                        history[selection["edge_source_slots"][i]],
                        selection["coefficients32"][i])
                else:
                    value = _mul(weights.data[edge], np.float32(rate[pre] * mask[pre]))
                lanes[lane] = _add(lanes[lane], value)
        for offset in (16, 8, 4, 2, 1):
            old = lanes.copy()
            for lane in range(32 - offset):
                lanes[lane] = _add(old[lane], old[lane + offset])
        result[row] = lanes[0]
    return result


def history_derivative(rate, history, source_ids):
    return (np.asarray(rate, np.float32)[source_ids].astype(np.float64)[:, None]
            - np.asarray(history, np.float64)) / TAU


_HISTORY_SOURCE = r"""
extern "C" __global__ void history_stage(const long long* sources,
    const float* rate, const double* h, const double* base, double* derivative,
    double* next, double step, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count*2) return;
    double tau = (i & 1) ? 0.040 : 0.005;
    double k = ((double)rate[sources[i/2]] - h[i]) / tau;
    derivative[i] = k;
    next[i] = base[i] + step*k;
}
extern "C" __global__ void history_finish(const long long* sources,
    const float* rate, const double* h, const double* k1, const double* k2,
    const double* k3, double* base, double sixth_step, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count*2) return;
    double tau = (i & 1) ? 0.040 : 0.005;
    double k4 = ((double)rate[sources[i/2]] - h[i]) / tau;
    double sum = ((k1[i] + 2.0*k2[i]) + 2.0*k3[i]) + k4;
    base[i] = base[i] + sixth_step*sum;
}
"""


def source_filtered_cuda_source(fuse_tanh=True):
    source = _timescale_source(fuse_tanh)
    before = "const float* adaptation, const float* coefficient, float* input, int n) {"
    after = ("const float* adaptation, const float* coefficient, float* input, "
             "const int* edge_slot, const int* source_slot, const float* c, "
             "const double* history, const float* rate, const float* mask, int n) {")
    point = "value += lookup[item >> 18] * masked[item & 0x3ffff];"
    replacement = r"""int slot = edge_slot[edge];
        if (slot < 0 || (c[3*slot] == 1.0f && c[3*slot+1] == 0.0f && c[3*slot+2] == 0.0f)) {
            value += lookup[item >> 18] * masked[item & 0x3ffff];
        } else {
            int pre = item & 0x3ffff;
            int s = source_slot[slot];
            float mixture = (c[3*slot]*rate[pre] + c[3*slot+1]*(float)history[2*s])
                            + c[3*slot+2]*(float)history[2*s+1];
            value += lookup[item >> 18] * mul_exact(mixture, mask[pre]);
        }"""
    if source.count(before) != 1 or source.count(point) != 1:
        raise RuntimeError("Frozen packed CSR transformation points changed")
    return source.replace(before, after).replace(point, replacement) + _HISTORY_SOURCE


def _equal(left, right):
    if isinstance(right, dict):
        return (isinstance(left, dict) and left.keys() == right.keys()
                and all(_equal(left[k], v) for k, v in right.items()))
    if isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and left.tobytes() == right.tobytes())
    return type(left) is type(right) and left == right


class SourceFilteredSynapseNetwork(MotorTimescaleNetwork):
    """Shared source histories, edge-specific nonnegative input mixtures."""

    def __init__(self, *args, edge_ids, coefficients, free_motor_ids, **kwargs):
        self._selection = checked_selection(args[0], edge_ids, coefficients, free_motor_ids)
        motor_ids = np.asarray(kwargs.get("motor_ids"))
        if (motor_ids.ndim != 1 or motor_ids.dtype != np.int64
                or not np.isin(self._selection["free_motor_ids"], motor_ids).all()):
            raise ValueError("Free MN roster must belong to actual native motor roster")
        if kwargs.get("device") not in ("cpu", "cuda"):
            raise ValueError("Explicit CPU oracle or packed CUDA device required")
        if kwargs.get("device") == "cuda" and kwargs.get("cuda_implementation") != "packed":
            raise ValueError("Source filtering requires packed CUDA")
        if kwargs.get("device") == "cuda" and (
                args[0].shape[0] > 2**18 or args[0].nnz >= 2**31
                or len(np.unique(args[0].data.view(np.uint32))) > 2**14):
            raise ValueError("Source graph exceeds packed CUDA representation limits")
        if kwargs.get("dt", .0001) > TAU.min() / 10:
            raise ValueError("Integration step must resolve fast source history")
        self._source_weights = args[0].copy()
        self._history_tau = TAU.copy()
        self._neutral = np.array_equal(self._selection["coefficients32"],
            np.tile(np.array([1, 0, 0], np.float32), (len(edge_ids), 1)))
        owned_args = (args[0].copy(), *(np.asarray(p).copy() for p in args[1:]))
        super().__init__(*owned_args, **kwargs)
        if self.device == "cuda" and self.cuda_implementation != "packed":
            raise ValueError("Native graph exceeded packed representation limits")
        self._init_history()
        self.history.zero_()  # Constructor warmup/capture is not scientific state.
        self.source_timescale_identity = self.identity
        self._metadata = {"schema": SCHEMA, "source_model_hash": self.identity,
                          "precision": PRECISION, "tau": TAU.copy(),
                          **copy.deepcopy(self._selection)}
        digest = hashlib.sha256((self.identity + SCHEMA + PRECISION).encode())
        for key in sorted(self._metadata):
            value = self._metadata[key]
            if isinstance(value, np.ndarray):
                digest.update(key.encode() + value.dtype.str.encode()
                              + repr(value.shape).encode() + value.tobytes())
        self.identity = digest.hexdigest()
        self._history_pointer = self.history.data_ptr()
        self._parameter_pointers = {k: getattr(self, k).data_ptr() for k in
            ("_edge_slot", "_source_slot", "_filter_coefficients", "_source_ids")}
        self._inherited_metadata = {k: v for k, v in super().snapshot().items()
            if k not in ("rate", "adaptation", "drive", "output_mask", "tick")}

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    def _init_history(self):
        if hasattr(self, "history"):
            return
        t = self.torch
        count = len(self._selection["source_ids"])
        slots = np.full(self._source_weights.nnz, -1, np.int32)
        slots[self._selection["edge_ids"]] = np.arange(len(self._selection["edge_ids"]))
        # CUDA warmup invokes this inside inference_mode. Allocate ordinary
        # tensors so later constructor cleanup, reset and restore may mutate
        # them normally, without rebinding any captured addresses.
        with t.inference_mode(False):
            self.history = t.zeros((count, 2), dtype=t.float64, device=self.device)
            self._history_buffers = [t.empty_like(self.history) for _ in range(4)]
            for name, value in (("_edge_slot", slots),
                ("_source_slot", self._selection["edge_source_slots"]),
                ("_filter_coefficients", self._selection["coefficients32"]),
                ("_source_ids", self._selection["source_ids"])):
                setattr(self, name, t.as_tensor(value.copy(), device=self.device))

    def _init_fusion(self):
        self._init_history()
        super()._init_fusion()
        source = source_filtered_cuda_source(self.fuse_tanh)
        for name in ("csr_input", "history_stage", "history_finish"):
            kernel = self._cupy.RawKernel(source, name, options=("--fmad=false",))
            kernel.compile()
            self._fused_kernels[name] = kernel

    def _history_launch(self, name, tensors, step):
        count = len(self._selection["source_ids"])
        self._fused_kernels[name](((2*count + 255)//256,), (256,),
            tuple(np.uint64(x.data_ptr()) for x in tensors)
            + (np.float64(step), np.int32(count)))

    def _step_fused(self):
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, work_r, work_z, inputs, activation, *derivatives = self._fused_buffers
        ks, azs = derivatives[:3], derivatives[3:]
        work_h, *hs = self._history_buffers
        self.torch.mul(self.rate, self.output_mask, out=masked)
        r, z, h = self.rate, self.adaptation, self.history
        with self._cupy.cuda.ExternalStream(self.torch.cuda.current_stream().cuda_stream):
            for stage in range(4):
                self._launch_fused("csr_input", (self._rowptr, self._columns,
                    self._weight_values, masked, self.drive, self.threshold, z,
                    self.coefficient, inputs, self._edge_slot, self._source_slot,
                    self._filter_coefficients, h, r, self.output_mask))
                if self.fuse_tanh:
                    activation = inputs
                else:
                    self.torch.tanh(inputs, out=activation)
                # History reads the current rate BEFORE stage overwrites work_r.
                if stage < 3:
                    step = self.dt * (.5 if stage < 2 else 1)
                    self._history_launch("history_stage", (self._source_ids, r, h,
                        self.history, hs[stage], work_h), step)
                    self._launch_fused("stage", (activation, self.cap, r, z, self.tau,
                        self.rate, self.adaptation, self.output_mask, ks[stage],
                        azs[stage], work_r, work_z, masked),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, step))
                    r, z, h = work_r, work_z, work_h
                else:
                    self._history_launch("history_finish", (self._source_ids, r, h,
                        *hs, self.history), self.dt/6)
                    self._launch_fused("finish", (activation, self.cap, r, z, self.tau,
                        *ks, *azs, self.rate, self.adaptation),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, self.dt/6))

    def _step(self):
        if self.device == "cuda":
            self._step_fused()
            return
        self._init_history()
        t, dt = self.torch, self.dt
        r0, z0, h0 = self.rate, self.adaptation, self.history
        r, z, h = r0, z0, h0
        kr, kz, kh = [], [], []
        for stage in range(4):
            if self._neutral:
                dr, dz = super()._derivative_pair(r, z)
            else:
                current = packed_current_oracle(self._source_weights, r.numpy(),
                    self.output_mask.numpy(), self._selection, h.numpy())
                current = t.as_tensor(current)
                target = t.clamp_min(self.cap*t.tanh(self.coefficient*(
                    self.drive + current - self.threshold - z)), 0.)
                dr = (target-r)/self.tau
                dz = (self._cell_gain*r-z)/self._cell_adaptation_tau
            dh = (r[self._source_ids].to(t.float64)[:, None]-h)/t.as_tensor(self._history_tau)
            kr.append(dr)
            kz.append(dz)
            kh.append(dh)
            if stage < 3:
                step = dt*(.5 if stage < 2 else 1)
                r, z, h = r0+step*dr, z0+step*dz, h0+step*dh
        for state, values in ((r0, kr), (z0, kz), (h0, kh)):
            state.add_(dt/6*(values[0]+2*values[1]+2*values[2]+values[3]))

    def _current(self, rate):
        if not self._neutral:
            raise RuntimeError("Nonneutral current requires coupled stage history; use source_current_oracle")
        return super()._current(rate)

    def _derivative_pair(self, rate, adaptation):
        if not self._neutral:
            raise RuntimeError("Nonneutral derivative requires coupled RK4 source history")
        return super()._derivative_pair(rate, adaptation)

    def _derivative(self, rate):
        raise RuntimeError("Source filter requires joint rate/adaptation/history integration")

    def source_current_oracle(self, *, rate=None, history=None):
        """Host scalar packed-current oracle, not a separate native current kernel.

        Defaults to the current model endpoint. Explicit arguments describe one
        RK stage together and must use original f32 rate and f64 history.
        """
        self._assert_parameters()
        r = self.rate.cpu().numpy().copy() if rate is None else np.asarray(rate)
        h = self.history.cpu().numpy().copy() if history is None else np.asarray(history)
        if (r.dtype != np.float32 or r.shape != (self.n,) or not np.isfinite(r).all()
                or h.dtype != np.float64 or h.shape != tuple(self.history.shape)
                or not np.isfinite(h).all()):
            raise ValueError("Joint finite f32 rate and f64 source history required")
        return packed_current_oracle(self._source_weights, r,
            self.output_mask.cpu().numpy(), self._selection, h)

    def _assert_parameters(self):
        expected = {"_source_ids": self._selection["source_ids"],
            "_source_slot": self._selection["edge_source_slots"],
            "_filter_coefficients": self._selection["coefficients32"]}
        slots = np.full(self._source_weights.nnz, -1, np.int32)
        slots[self._selection["edge_ids"]] = np.arange(len(self._selection["edge_ids"]))
        expected["_edge_slot"] = slots
        if self.history.data_ptr() != self._history_pointer:
            raise ValueError("History captured pointer changed")
        for key, value in expected.items():
            actual = getattr(self, key)
            if (actual.data_ptr() != self._parameter_pointers[key]
                    or not _equal(actual.cpu().numpy(), value)):
                raise ValueError("Owned source filter parameter bytes or pointer changed")

    def reset(self):
        super().reset()
        self.history.zero_()

    def snapshot(self):
        self._assert_parameters()
        return {**super().snapshot(), "source_filter": copy.deepcopy(self._metadata),
                "source_history": self.history.cpu().numpy().copy()}

    def _prevalidate(self, saved):
        self._assert_parameters()
        if not isinstance(saved, dict):
            raise TypeError("Complete source filter checkpoint required")
        expected_keys = set(self._inherited_metadata) | {
            "rate", "adaptation", "drive", "output_mask", "tick",
            "source_filter", "source_history"}
        if set(saved) != expected_keys or not _equal(saved["source_filter"], self._metadata):
            raise ValueError("Source filter metadata mismatch")
        if any(not _equal(saved[k], v) for k, v in self._inherited_metadata.items()):
            raise ValueError("Inherited metadata mismatch")
        if type(saved["tick"]) is not int or saved["tick"] < 0:
            raise ValueError("Invalid saved tick")
        for key in ("rate", "adaptation", "drive", "output_mask", "source_history"):
            value = saved[key]
            shape = self.history.shape if key == "source_history" else (self.n,)
            dtype = np.float64 if key == "source_history" else np.float32
            if (not isinstance(value, np.ndarray) or value.dtype != dtype
                    or value.shape != shape or not np.isfinite(value).all()):
                raise ValueError(f"Invalid saved {key}")
        if (np.any(saved["rate"] < -1e-4) or np.any(saved["adaptation"] < 0)
                or not np.isin(saved["output_mask"], (0, 1)).all()):
            raise ValueError("Invalid inherited state")
        self._assert_actual_timescales()
        # Prepare every tensor before copying anything into the live model.
        return {key: self.torch.as_tensor(saved[key].copy(), device=self.device)
                for key in ("rate", "adaptation", "drive", "output_mask", "source_history")}

    def restore(self, saved):
        values = self._prevalidate(saved)
        for key, value in values.items():
            target = self.history if key == "source_history" else getattr(self, key)
            target.copy_(value)
        self.tick = saved["tick"]

    def import_neutral_base_snapshot(self, saved, *, history=None):
        """Neutral-only import with steady prehistory (both h = saved source rate).

        This assumption uses only the saved instantaneous source state. Reset
        zeros histories; ordinary restore retains the explicitly saved histories.
        """
        if not self._neutral:
            raise ValueError("Base snapshot import requires neutral coefficients")
        if not isinstance(saved, dict) or saved.get("model_hash") != self.source_timescale_identity:
            raise ValueError("Exact original MotorTimescale identity required")
        candidate = copy.deepcopy(saved)
        candidate["model_hash"] = self.identity
        candidate["source_filter"] = copy.deepcopy(self._metadata)
        candidate["source_history"] = (np.repeat(np.asarray(saved["rate"])[
            self._selection["source_ids"]].astype(np.float64)[:, None], 2, axis=1)
            if history is None else np.asarray(history).copy())
        self.restore(candidate)

    import_base_snapshot = import_neutral_base_snapshot
