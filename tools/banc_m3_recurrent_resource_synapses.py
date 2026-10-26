"""Shared presynaptic resources on selected existing premotor recurrence.

Engineering dynamics only; no measured STP or gait claim. All CUDA pointwise
resource arithmetic is float64 without FMA; CSR keeps native warp FTZ order.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from scipy.sparse import csr_matrix

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source
from tools.banc_m3_source_filtered_synapses import _add, _equal, _mul

SCHEMA = "flylab.m3-recurrent-resource-synapses.v1"
PRECISION = ("resource/u/tau:f64;rate-input:stage-f32-to-f64;"
             "selected:weight*((rate*mask)*f32(R));mask:f32-rn-no-ftz;"
             "selected-products-and-csr-sums:f32-rn-ftz-original-warp-order;"
             "u0:original-expression;resource-rk4:f64-no-fma-before-rate-overwrite")
SELECTION_KEYS = {"edge_ids", "premotor_ids", "motor_ids", "pre_ids", "post_ids",
                  "weights32", "source_ids", "edge_source_slots", "utilization", "recovery_tau"}


def _ids(value, name, *, bound=None, sorted_ids=True):
    a = np.asarray(value)
    if (a.ndim != 1 or a.dtype != np.int64 or not len(a) or np.any(a < 0)
            or len(np.unique(a)) != len(a)
            or (sorted_ids and np.any(a[1:] <= a[:-1]))
            or (bound is not None and np.any(a >= bound))):
        raise ValueError("Native unique in-range int64 "+name+" required")
    return a.copy()


def _selection_shape(s):
    if not isinstance(s, dict) or set(s) != SELECTION_KEYS:
        raise ValueError("Complete recurrent selection required")
    edge, prepop, motor, sources = (_ids(s[k], k, sorted_ids=k != "motor_ids")
        for k in ("edge_ids", "premotor_ids", "motor_ids", "source_ids"))
    pre, post = s["pre_ids"], s["post_ids"]
    slots, w, u, tau = (s[k] for k in ("edge_source_slots", "weights32", "utilization", "recovery_tau"))
    if (any(not isinstance(v, np.ndarray) or v.dtype != np.int64 or v.shape != edge.shape for v in (pre, post))
            or not np.isin(pre, prepop).all() or not np.isin(post, prepop).all()
            or np.intersect1d(prepop, motor).size or not np.array_equal(np.unique(pre), sources)
            or not isinstance(slots, np.ndarray) or slots.dtype != np.int32 or slots.shape != edge.shape
            or not np.array_equal(slots, np.searchsorted(sources, pre))
            or not isinstance(w, np.ndarray) or w.dtype != np.float32 or w.shape != edge.shape
            or not np.isfinite(w).all() or np.any(w == 0)
            or any(not isinstance(v, np.ndarray) or v.dtype != np.float64 or v.shape != sources.shape
                   or not np.isfinite(v).all() for v in (u, tau))
            or np.any((u < 0) | (u > .5)) or np.any((tau < .02) | (tau > .5))):
        raise ValueError("Original premotor edges and bounded shared source parameters required")
    return copy.deepcopy(s)


def checked_selection(weights, edge_ids, premotor_ids, utilization, recovery_tau, motor_ids):
    """Reject topology changes; CSR positions are original edge IDs, not node IDs."""
    if (not isinstance(weights, csr_matrix) or weights.dtype != np.float32
            or weights.shape[0] != weights.shape[1] or not weights.shape[0]
            or not weights.has_canonical_format or not weights.has_sorted_indices
            or not np.isfinite(weights.data).all()):
        raise ValueError("Canonical finite square float32 original CSR required")
    if (weights.indices.ndim != 1 or weights.indptr.ndim != 1 or weights.data.ndim != 1
            or weights.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
            or weights.indptr.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
            or len(weights.indptr) != weights.shape[0]+1 or weights.indptr[0] != 0
            or weights.indptr[-1] != len(weights.data) or len(weights.indices) != len(weights.data)):
        raise ValueError("Exact original CSR storage lengths required")
    # SciPy's flags can be stale after direct index edits; check the actual rows.
    # check_format may prune/cast its input, so use an owned copy for that check.
    weights.copy().check_format(full_check=True)
    same_row = np.ones(max(0, weights.nnz-1), bool)
    boundaries = weights.indptr[1:-1]-1
    same_row[boundaries[(boundaries >= 0) & (boundaries < len(same_row))]] = False
    if np.any((weights.indices[1:] <= weights.indices[:-1]) & same_row):
        raise ValueError("Original CSR rows must be sorted without duplicate entries")
    edge = _ids(edge_ids, "edge_ids", bound=weights.nnz)
    population = _ids(premotor_ids, "premotor_ids", bound=weights.shape[0])
    motor = _ids(motor_ids, "motor_ids", bound=weights.shape[0], sorted_ids=False)
    pre = weights.indices[edge].astype(np.int64)
    sources = np.unique(pre)
    return _selection_shape({"edge_ids": edge, "premotor_ids": population, "motor_ids": motor,
        "pre_ids": pre.copy(), "post_ids": (np.searchsorted(weights.indptr, edge, side="right")-1).astype(np.int64),
        "weights32": weights.data[edge].copy(), "source_ids": sources,
        "edge_source_slots": np.searchsorted(sources, pre).astype(np.int32),
        "utilization": np.asarray(utilization).copy(), "recovery_tau": np.asarray(recovery_tau).copy()})


def resource_metadata(source_model_hash, selection):
    if not isinstance(source_model_hash, str) or len(source_model_hash) != 64 or any(c not in "0123456789abcdef" for c in source_model_hash):
        raise ValueError("Original neural SHA256 identity required")
    return {"schema": SCHEMA, "source_model_hash": source_model_hash, "precision": PRECISION,
            **_selection_shape(selection)}


def checked_resource_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) != SELECTION_KEYS | {"schema", "source_model_hash", "precision"}:
        raise ValueError("Complete recurrent resource metadata required")
    if metadata["schema"] != SCHEMA or metadata["precision"] != PRECISION:
        raise ValueError("Recurrent schema/precision mismatch")
    return resource_metadata(metadata["source_model_hash"], {k: metadata[k] for k in SELECTION_KEYS})


def resource_identity(metadata):
    m = checked_resource_metadata(metadata)
    checksum = hashlib.sha256((m["source_model_hash"]+SCHEMA+PRECISION).encode())
    for key in sorted(SELECTION_KEYS):
        v = m[key]
        checksum.update(key.encode()+v.dtype.str.encode()+repr(v.shape).encode()+v.tobytes(order="C"))
    return checksum.hexdigest()


def resource_derivative(rate, resource, source_ids, utilization, recovery_tau):
    """Same-stage actual rate, independent of the outgoing transmission mask."""
    r, R, u, tau = map(np.asarray, (rate, resource, utilization, recovery_tau))
    sources = _ids(source_ids, "source_ids", bound=len(r))
    if (r.ndim != 1 or r.dtype != np.float32 or not np.isfinite(r).all()
            or any(v.dtype != np.float64 or v.shape != sources.shape or not np.isfinite(v).all() for v in (R, u, tau))
            or np.any(tau <= 0)):
        raise ValueError("Finite f32 stage rate and aligned f64 resource parameters required")
    return (1.-R)/tau-(u*r[sources].astype(np.float64))*R


def resource_edge_value(weight, rate, mask, resource, utilization):
    """Scalar oracle: native masked rate first, then CSR FTZ products."""
    masked = np.float32(np.float32(rate)*np.float32(mask))
    if utilization == 0: return _mul(weight, masked)
    return _mul(weight, _mul(masked, np.float32(resource)))


def packed_current_oracle(weights, rate, mask, selection, resource):
    chosen = {int(edge): i for i, edge in enumerate(selection["edge_ids"])}
    out = np.zeros(weights.shape[0], np.float32)
    for row in range(len(out)):
        lane = np.zeros(32, np.float32)
        begin, end = weights.indptr[row:row+2]
        for j in range(32):
            for edge in range(begin+j, end, 32):
                pre = weights.indices[edge]
                slot = chosen.get(edge)
                if slot is None: value = resource_edge_value(weights.data[edge], rate[pre], mask[pre], 1., 0.)
                else:
                    source = selection["edge_source_slots"][slot]
                    value = resource_edge_value(weights.data[edge], rate[pre], mask[pre], resource[source], selection["utilization"][source])
                lane[j] = _add(lane[j], value)
        for offset in (16, 8, 4, 2, 1):
            old = lane.copy()
            for j in range(32-offset): lane[j] = _add(old[j], old[j+offset])
        out[row] = lane[0]
    return out


_RESOURCE_SOURCE = r"""
extern "C" __global__ void resource_stage(const long long* sources, const float* rate,
    const double* resource, const double* base, const double* u, const double* tau,
    double* derivative, double* next, double step, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    double k = (1.0-resource[i])/tau[i] - (u[i]*(double)rate[sources[i]])*resource[i];
    derivative[i] = k;
    next[i] = base[i]+step*k;
}
extern "C" __global__ void resource_finish(const long long* sources, const float* rate,
    const double* resource, const double* u, const double* tau,
    const double* k1, const double* k2, const double* k3,
    double* base, double sixth_step, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    double k4 = (1.0-resource[i])/tau[i] - (u[i]*(double)rate[sources[i]])*resource[i];
    double sum = ((k1[i]+2.0*k2[i])+2.0*k3[i])+k4;
    base[i] = base[i]+sixth_step*sum;
}
"""


def resource_cuda_source(fuse_tanh=True):
    source = _timescale_source(fuse_tanh)
    before = "const float* adaptation, const float* coefficient, float* input, int n) {"
    point = "value += lookup[item >> 18] * masked[item & 0x3ffff];"
    after = ("const float* adaptation, const float* coefficient, float* input, "
             "const int* edge_source, const double* u, const double* resource, int n) {")
    replacement = r"""int s = edge_source[edge];
        if (s < 0 || u[s] == 0.0) {
            value += lookup[item >> 18] * masked[item & 0x3ffff];
        } else {
            value += lookup[item >> 18] * (masked[item & 0x3ffff] * (float)resource[s]);
        }"""
    if source.count(before) != 1 or source.count(point) != 1:
        raise RuntimeError("Frozen packed CSR transformation points changed")
    return source.replace(before, after).replace(point, replacement)+_RESOURCE_SOURCE


class RecurrentResourceNetwork(MotorTimescaleNetwork):
    """Joint RK4 resource/rate/adaptation with unchanged native neural parameters."""

    def __init__(self, *args, edge_ids, premotor_ids, utilization, recovery_tau, **kwargs):
        self._selection = checked_selection(args[0], edge_ids, premotor_ids, utilization, recovery_tau, kwargs.get("motor_ids"))
        self._source_weights = args[0].copy()
        self._neutral = bool(np.all(self._selection["utilization"] == 0))
        dt = kwargs.get("dt", .0001)
        cap = np.asarray(args[4])
        tau, u = self._selection["recovery_tau"], self._selection["utilization"]
        if (type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0
                or not np.isfinite(cap).all() or np.any(cap <= 0)
                or dt > tau.min()/20 or dt*(1/tau.min()+u.max()*cap.max()) > .1):
            raise ValueError("Integration interval must resolve depletion and recovery")
        if kwargs.get("device") not in ("cpu", "cuda"):
            raise ValueError("Explicit CPU or CUDA resource device required")
        if kwargs.get("device") == "cuda" and (kwargs.get("cuda_implementation") != "packed"
                or args[0].shape[0] > 2**18 or args[0].nnz >= 2**31
                or len(np.unique(args[0].data.view(np.uint32))) > 2**14):
            raise ValueError("Original packed CUDA representation required")
        super().__init__(args[0].copy(), *(np.asarray(v).copy() for v in args[1:]), **kwargs)
        self._init_resource()
        self.resource.fill_(1)  # Warmup and graph capture are not scientific state.
        self.source_timescale_identity = self.identity
        self._metadata = resource_metadata(self.identity, self._selection)
        self.identity = resource_identity(self._metadata)
        self._parameter_pointers = {k: getattr(self, k).data_ptr() for k in
            ("resource", "_resource_sources", "_resource_edge_source", "_resource_u", "_resource_tau")}
        self._buffer_pointers = [v.data_ptr() for v in self._resource_buffers]
        self._inherited_metadata = {k: v for k, v in super().snapshot().items()
            if k not in ("rate", "adaptation", "drive", "output_mask", "tick")}

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    def _init_resource(self):
        if hasattr(self, "resource"): return
        t, count = self.torch, len(self._selection["source_ids"])
        edge_source = np.full(self._source_weights.nnz, -1, np.int32)
        edge_source[self._selection["edge_ids"]] = self._selection["edge_source_slots"]
        with t.inference_mode(False):
            self.resource = t.ones(count, dtype=t.float64, device=self.device)
            self._resource_buffers = [t.empty_like(self.resource) for _ in range(4)]
            for key, value in (("_resource_sources", self._selection["source_ids"]),
                ("_resource_edge_source", edge_source), ("_resource_u", self._selection["utilization"]),
                ("_resource_tau", self._selection["recovery_tau"])):
                setattr(self, key, t.as_tensor(value.copy(), device=self.device))

    def _init_fusion(self):
        self._init_resource()
        super()._init_fusion()
        if self._neutral: return
        source = resource_cuda_source(self.fuse_tanh)
        for name in ("csr_input", "resource_stage", "resource_finish"):
            kernel = self._cupy.RawKernel(source, name, options=("--fmad=false",))
            kernel.compile()
            self._fused_kernels[name] = kernel

    def _resource_launch(self, name, tensors, step):
        count = len(self._selection["source_ids"])
        self._fused_kernels[name](((count+255)//256,), (256,),
            tuple(np.uint64(v.data_ptr()) for v in tensors)+(np.float64(step), np.int32(count)))

    def _step_fused(self):
        if self._neutral: return super()._step_fused()
        if not hasattr(self, "_fused_kernels"): self._init_fusion()
        masked, work_r, work_z, inputs, activation, *derivatives = self._fused_buffers
        kr, kz = derivatives[:3], derivatives[3:]
        work_R, *kR = self._resource_buffers
        self.torch.mul(self.rate, self.output_mask, out=masked)
        r, z, R = self.rate, self.adaptation, self.resource
        with self._cupy.cuda.ExternalStream(self.torch.cuda.current_stream().cuda_stream):
            for stage in range(4):
                self._launch_fused("csr_input", (self._rowptr, self._columns, self._weight_values,
                    masked, self.drive, self.threshold, z, self.coefficient, inputs,
                    self._resource_edge_source, self._resource_u, R))
                if self.fuse_tanh: activation = inputs
                else: self.torch.tanh(inputs, out=activation)
                if stage < 3:
                    step = self.dt*(.5 if stage < 2 else 1)
                    self._resource_launch("resource_stage", (self._resource_sources, r, R,
                        self.resource, self._resource_u, self._resource_tau, kR[stage], work_R), step)
                    self._launch_fused("stage", (activation, self.cap, r, z, self.tau,
                        self.rate, self.adaptation, self.output_mask, kr[stage], kz[stage], work_r, work_z, masked),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, step))
                    r, z, R = work_r, work_z, work_R
                else:
                    self._resource_launch("resource_finish", (self._resource_sources, r, R,
                        self._resource_u, self._resource_tau, *kR, self.resource), self.dt/6)
                    self._launch_fused("finish", (activation, self.cap, r, z, self.tau,
                        *kr, *kz, self.rate, self.adaptation),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, self.dt/6))

    def _step(self):
        if self._neutral: return super()._step()
        if self.device == "cuda": return self._step_fused()
        t, dt = self.torch, self.dt
        r0, z0, R0 = self.rate, self.adaptation, self.resource
        r, z, R = r0, z0, R0
        kr, kz, kR = [], [], []
        for stage in range(4):
            current = t.as_tensor(packed_current_oracle(self._source_weights, r.numpy(), self.output_mask.numpy(), self._selection, R.numpy()))
            target = t.clamp_min(self.cap*t.tanh(self.coefficient*(self.drive+current-self.threshold-z)), 0.)
            dr, dz = (target-r)/self.tau, (self._cell_gain*r-z)/self._cell_adaptation_tau
            dR = (1.-R)/self._resource_tau-(self._resource_u*r[self._resource_sources].to(t.float64))*R
            kr.append(dr); kz.append(dz); kR.append(dR)
            if stage < 3:
                step = dt*(.5 if stage < 2 else 1)
                r, z, R = r0+step*dr, z0+step*dz, R0+step*dR
        for state, values in ((r0, kr), (z0, kz), (R0, kR)):
            state.add_(dt/6*(values[0]+2*values[1]+2*values[2]+values[3]))

    def _current(self, rate):
        if not self._neutral: raise RuntimeError("Resource current requires joint RK-stage state")
        return super()._current(rate)

    def _derivative_pair(self, rate, adaptation):
        if not self._neutral: raise RuntimeError("Resource derivative requires joint RK-stage state")
        return super()._derivative_pair(rate, adaptation)

    def _derivative(self, rate):
        raise RuntimeError("Resource dynamics require joint rate/adaptation/resource integration")

    def resource_current_oracle(self, *, rate=None, resource=None):
        self._assert_parameters()
        r = self.rate.cpu().numpy().copy() if rate is None else np.asarray(rate)
        R = self.resource.cpu().numpy().copy() if resource is None else np.asarray(resource)
        if (r.dtype != np.float32 or r.shape != (self.n,) or not np.isfinite(r).all()
                or R.dtype != np.float64 or R.shape != tuple(self.resource.shape) or not np.isfinite(R).all()):
            raise ValueError("Finite simultaneous f32 rate and f64 resource required")
        return packed_current_oracle(self._source_weights, r, self.output_mask.cpu().numpy(), self._selection, R)

    def _assert_parameters(self):
        slots = np.full(self._source_weights.nnz, -1, np.int32)
        slots[self._selection["edge_ids"]] = self._selection["edge_source_slots"]
        expected = {"_resource_sources": self._selection["source_ids"], "_resource_edge_source": slots,
                    "_resource_u": self._selection["utilization"], "_resource_tau": self._selection["recovery_tau"]}
        if (self.resource.data_ptr() != self._parameter_pointers["resource"]
                or [v.data_ptr() for v in self._resource_buffers] != self._buffer_pointers):
            raise ValueError("Captured resource/buffer pointer changed")
        for key, value in expected.items():
            actual = getattr(self, key)
            if actual.data_ptr() != self._parameter_pointers[key] or not _equal(actual.cpu().numpy(), value):
                raise ValueError("Owned resource parameter pointer/bytes changed")
        self._assert_actual_timescales()

    def reset(self):
        super().reset()
        self.resource.fill_(1)

    def snapshot(self):
        self._assert_parameters()
        return {**super().snapshot(), "recurrent_resource": copy.deepcopy(self._metadata),
                "synaptic_resource": self.resource.cpu().numpy().copy()}

    def _prevalidate(self, saved):
        self._assert_parameters()
        if not isinstance(saved, dict): raise TypeError("Complete recurrent checkpoint required")
        state_keys = {"rate", "adaptation", "drive", "output_mask", "tick", "recurrent_resource", "synaptic_resource"}
        if (set(saved) != set(self._inherited_metadata) | state_keys
                or not _equal(saved["recurrent_resource"], self._metadata)
                or any(not _equal(saved[k], v) for k, v in self._inherited_metadata.items())
                or type(saved["tick"]) is not int or saved["tick"] < 0):
            raise ValueError("Exact inherited and recurrent identity required")
        for key in ("rate", "adaptation", "drive", "output_mask", "synaptic_resource"):
            v = saved[key]; shape = tuple(self.resource.shape) if key == "synaptic_resource" else (self.n,)
            dtype = np.float64 if key == "synaptic_resource" else np.float32
            if not isinstance(v, np.ndarray) or v.dtype != dtype or v.shape != shape or not np.isfinite(v).all():
                raise ValueError("Invalid saved "+key)
        R = saved["synaptic_resource"]
        if (np.any(saved["rate"] < -1e-4) or np.any(saved["adaptation"] < 0)
                or not np.isin(saved["output_mask"], (0, 1)).all() or np.any((R < 0) | (R > 1))
                or np.any(R[self._selection["utilization"] == 0] != 1)
                or (saved["tick"] == 0 and np.any(R != 1))):
            raise ValueError("Invalid neural/resource state bounds")
        return {k: self.torch.as_tensor(saved[k].copy(), device=self.device) for k in
                ("rate", "adaptation", "drive", "output_mask", "synaptic_resource")}

    def restore(self, saved):
        staged = self._prevalidate(saved)
        for key, value in staged.items():
            target = self.resource if key == "synaptic_resource" else getattr(self, key)
            target.copy_(value)
        self.tick = saved["tick"]

    def import_base_snapshot(self, saved):
        """First application is neutral-only; ordinary restore never ignores identity."""
        if not isinstance(saved, dict) or saved.get("model_hash") != self.source_timescale_identity:
            raise ValueError("Exact original MotorTimescale identity required")
        if (type(saved.get("tick")) is not int or saved["tick"] != 0
                or any(not isinstance(saved.get(k), np.ndarray) or saved[k].dtype != np.float32
                    or saved[k].shape != (self.n,) or not np.all(saved[k] == 0) for k in ("rate", "adaptation", "drive"))
                or not isinstance(saved.get("output_mask"), np.ndarray) or saved["output_mask"].dtype != np.float32
                or saved["output_mask"].shape != (self.n,) or not np.all(saved["output_mask"] == 1)):
            raise ValueError("Base import requires tick0/rate0/adaptation0/drive0/mask1")
        candidate = copy.deepcopy(saved)
        candidate.update(model_hash=self.identity, recurrent_resource=copy.deepcopy(self._metadata),
                         synaptic_resource=np.ones(len(self._selection["source_ids"]), np.float64))
        self.restore(candidate)
