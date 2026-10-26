"""Same-stage recovery-only attenuation on existing recurrent synapses.

Engineering dynamics, not measured synaptic physiology or a gait claim.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from tools import banc_m3_recurrent_resource_synapses as original
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source
from tools.banc_m3_source_filtered_synapses import _add, _equal, _mul

SCHEMA = "flylab.m3-recovery-resource-synapses.v1"
PRECISION = (
    "R/u/tau:f64;unmasked-stage-rate:f32-to-f64;"
    "a=R*(1+(tau*u)*r);g32=f32(min(1,a));source-once-per-stage;"
    "u0-or-g32==1:original-weight*masked;otherwise:weight*(masked*g32);"
    "mask:f32-rn-no-ftz;csr-products/sums:f32-rn-ftz-original-warp-order;"
    "Rdot=(1-R)/tau-(u*r)*R;RK4:f64-no-fma-before-rate-overwrite;"
    "guards:original-resource-dt;R:[0,1];u0R:1;active-stage-r:finite>=0;"
    "sticky-fault:stage/final;clear:reset-or-valid-restore;readout:one-packet"
)
SELECTION_KEYS = original.SELECTION_KEYS
checked_selection = original.checked_selection
FAULT_RESOURCE = 1
FAULT_NEUTRAL_RESOURCE = 2
FAULT_ACTIVE_RATE = 4
FAULT_FACTOR = 8
FAULT_FINAL_RESOURCE = 16


def resource_metadata(source_model_hash, selection):
    validated = original.resource_metadata(source_model_hash, selection)
    return {**validated, "schema": SCHEMA, "precision": PRECISION}


def checked_resource_metadata(metadata):
    if (not isinstance(metadata, dict)
            or set(metadata) != SELECTION_KEYS | {"schema", "source_model_hash", "precision"}
            or metadata["schema"] != SCHEMA or metadata["precision"] != PRECISION):
        raise ValueError("Exact recovery resource schema/precision/fields required")
    return resource_metadata(metadata["source_model_hash"],
                             {key: metadata[key] for key in SELECTION_KEYS})


def resource_identity(metadata):
    m = checked_resource_metadata(metadata)
    checksum = hashlib.sha256((m["source_model_hash"]+SCHEMA+PRECISION).encode())
    for key in sorted(SELECTION_KEYS):
        value = m[key]
        checksum.update(key.encode()+value.dtype.str.encode()+repr(value.shape).encode()
                        +value.tobytes(order="C"))
    return checksum.hexdigest()


def _fault_error(flags):
    error = RuntimeError("Sticky recovery resource fault: "+str(flags))
    error.resource_fault_flags = int(flags)
    return error


def _stage_flags(rate, resource, source_ids, utilization):
    flags = 0
    if not np.isfinite(resource).all() or np.any((resource < 0) | (resource > 1)):
        flags |= FAULT_RESOURCE
    if np.any(resource[utilization == 0] != 1):
        flags |= FAULT_NEUTRAL_RESOURCE
    active_rate = rate[source_ids[utilization > 0]]
    if not np.isfinite(active_rate).all() or np.any(active_rate < 0):
        flags |= FAULT_ACTIVE_RATE
    return flags


def resource_factors(rate, resource, source_ids, utilization, recovery_tau):
    """Owned f32 factors; literal f64 expression, never rate/resource clipping."""
    r, R, u, tau = map(np.asarray, (rate, resource, utilization, recovery_tau))
    if r.ndim != 1 or r.dtype != np.float32:
        raise ValueError("Native one-dimensional float32 stage rate required")
    sources = original._ids(source_ids, "source_ids", bound=len(r))
    if (r.ndim != 1 or r.dtype != np.float32
            or any(v.dtype != np.float64 or v.shape != sources.shape
                   or not np.isfinite(v).all() for v in (u, tau))
            or R.dtype != np.float64 or R.shape != sources.shape
            or np.any((u < 0) | (u > .5)) or np.any((tau < .02) | (tau > .5))):
        raise ValueError("Native stage rate and bounded aligned resource parameters required")
    flags = _stage_flags(r, R, sources, u)
    if flags:
        raise _fault_error(flags)
    factors = np.ones(len(sources), np.float32)
    active = u > 0
    a = R[active]*(1.+(tau[active]*u[active])*r[sources[active]].astype(np.float64))
    if not np.isfinite(a).all() or np.any(a < 0):
        raise _fault_error(FAULT_FACTOR)
    factors[active] = np.minimum(1., a).astype(np.float32)
    if not np.isfinite(factors).all() or np.any((factors < 0) | (factors > 1)):
        raise _fault_error(FAULT_FACTOR)
    return factors


def resource_derivative(rate, resource, source_ids, utilization, recovery_tau):
    resource_factors(rate, resource, source_ids, utilization, recovery_tau)
    r, R, u, tau = map(np.asarray, (rate, resource, utilization, recovery_tau))
    source_ids = np.asarray(source_ids)
    result = np.zeros(R.shape, np.float64)
    active = u > 0
    result[active] = (1.-R[active])/tau[active]-(u[active]*r[source_ids[active]].astype(np.float64))*R[active]
    return result


def resource_edge_value(weight, rate, mask, resource, utilization, recovery_tau):
    factor = resource_factors(np.asarray([rate], np.float32), np.asarray([resource], np.float64),
        np.asarray([0], np.int64), np.asarray([utilization], np.float64),
        np.asarray([recovery_tau], np.float64))[0]
    return _factor_edge_value(weight, rate, mask, factor)


def _factor_edge_value(weight, rate, mask, factor):
    masked = np.float32(np.float32(rate)*np.float32(mask))
    if factor == np.float32(1):
        return _mul(weight, masked)
    return _mul(weight, _mul(masked, factor))


def packed_current_oracle(weights, rate, mask, selection, resource, *, factors=None):
    if factors is None:
        factors = resource_factors(rate, resource, selection["source_ids"],
                                    selection["utilization"], selection["recovery_tau"])
    chosen = dict(zip(selection["edge_ids"], selection["edge_source_slots"], strict=True))
    out = np.zeros(weights.shape[0], np.float32)
    for row in range(len(out)):
        lane = np.zeros(32, np.float32)
        begin, end = weights.indptr[row:row+2]
        for j in range(32):
            for edge in range(begin+j, end, 32):
                pre = weights.indices[edge]
                slot = chosen.get(edge)
                factor = np.float32(1) if slot is None else factors[slot]
                value = _factor_edge_value(weights.data[edge], rate[pre], mask[pre], factor)
                lane[j] = _add(lane[j], value)
        for offset in (16, 8, 4, 2, 1):
            old = lane.copy()
            for j in range(32-offset):
                lane[j] = _add(old[j], old[j+offset])
        out[row] = lane[0]
    return out


_RECOVERY_SOURCE = r"""
// PTX ISA cvt (sm_20+): without .ftz, conversions preserve subnormals.
// Inline PTX keeps CuPy's surrounding -ftz=true out of these conversions.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
__device__ __forceinline__ double recovery_f32_to_f64(float value) {
    double result;
    asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));
    return result;
}
__device__ __forceinline__ float recovery_f64_to_f32(double value) {
    float result;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
    return result;
}
__device__ __forceinline__ bool recovery_bad_rate(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int magnitude = bits & 0x7fffffffu;
    // Negative zero is valid; every negative nonzero bit pattern is invalid.
    return magnitude >= 0x7f800000u || ((bits & 0x80000000u) && magnitude != 0u);
}
__device__ __forceinline__ int recovery_state_flags(double R, double u, float r) {
    int flags = 0;
    if (!isfinite(R) || R < 0.0 || R > 1.0) flags |= 1;
    if (u == 0.0 && R != 1.0) flags |= 2;
    if (u > 0.0 && recovery_bad_rate(r)) flags |= 4;
    return flags;
}
extern "C" __global__ void recovery_factor(const long long* sources, const float* rate,
    const double* resource, const double* u, const double* tau,
    float* factor, int* fault, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    float r = rate[sources[i]];
    int flags = recovery_state_flags(resource[i], u[i], r);
    double a = u[i] == 0.0 ? 1.0 : resource[i]*(1.0+(tau[i]*u[i])*recovery_f32_to_f64(r));
    if (!isfinite(a) || a < 0.0) flags |= 8;
    float g = recovery_f64_to_f32(a < 1.0 ? a : 1.0);
    unsigned int g_bits = __float_as_uint(g);
    unsigned int g_magnitude = g_bits & 0x7fffffffu;
    if (recovery_bad_rate(g) || g_magnitude > 0x3f800000u) flags |= 8;
    if (flags) atomicOr(fault, flags);
    // Invalid states remain invalid; no state repair or successful publication.
    factor[i] = flags ? __int_as_float(0x7fffffff) : g;
}
extern "C" __global__ void recovery_stage(const long long* sources, const float* rate,
    const double* resource, const double* base, const double* u, const double* tau,
    double* derivative, double* next, double step, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    double k = u[i] == 0.0 ? 0.0 :
        (1.0-resource[i])/tau[i]-(u[i]*recovery_f32_to_f64(rate[sources[i]]))*resource[i];
    derivative[i] = k;
    next[i] = base[i]+step*k;
}
extern "C" __global__ void recovery_finish(const long long* sources, const float* rate,
    const double* resource, const double* u, const double* tau,
    const double* k1, const double* k2, const double* k3,
    double* base, int* fault, double sixth_step, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    double k4 = u[i] == 0.0 ? 0.0 :
        (1.0-resource[i])/tau[i]-(u[i]*recovery_f32_to_f64(rate[sources[i]]))*resource[i];
    double sum = ((k1[i]+2.0*k2[i])+2.0*k3[i])+k4;
    double R = base[i]+sixth_step*sum;
    base[i] = R;
    int flags = 0;
    if (!isfinite(R) || R < 0.0 || R > 1.0) flags |= 16;
    if (u[i] == 0.0 && R != 1.0) flags |= 2;
    if (flags) atomicOr(fault, flags);
}
"""


def resource_cuda_source(fuse_tanh=True):
    source = _timescale_source(fuse_tanh)
    before = "const float* adaptation, const float* coefficient, float* input, int n) {"
    point = "value += lookup[item >> 18] * masked[item & 0x3ffff];"
    after = ("const float* adaptation, const float* coefficient, float* input, "
             "const int* edge_source, const float* factor, int n) {")
    replacement = r"""int s = edge_source[edge];
        if (s < 0 || factor[s] == 1.0f) {
            value += lookup[item >> 18] * masked[item & 0x3ffff];
        } else {
            value += lookup[item >> 18] * (masked[item & 0x3ffff] * factor[s]);
        }"""
    if source.count(before) != 1 or source.count(point) != 1:
        raise RuntimeError("Frozen native packed CSR transformation points changed")
    return source.replace(before, after).replace(point, replacement)+_RECOVERY_SOURCE


class RecoveryResourceNetwork(original.RecurrentResourceNetwork):
    """Joint RK4 with recovery-only transmission and sticky stage faults."""

    def __init__(self, *args, edge_ids, premotor_ids, utilization, recovery_tau, **kwargs):
        super().__init__(*args, edge_ids=edge_ids, premotor_ids=premotor_ids,
                         utilization=utilization, recovery_tau=recovery_tau, **kwargs)
        self._metadata = resource_metadata(self.source_timescale_identity, self._selection)
        self.identity = resource_identity(self._metadata)
        self._inherited_metadata["model_hash"] = self.identity
        self.resource_factor.fill_(1)
        # Capture/setup flags cannot carry into the scientific reset state.
        self._resource_fault.zero_()
        names = ("resource_factor", "_resource_fault", "rate", "adaptation", "drive", "output_mask",
                 "tau", "a", "cap", "threshold", "coefficient", "_cell_gain",
                 "_cell_adaptation_tau", "_cell_inverse_tau", *self._parameter_pointers)
        if self.device == "cuda":
            names += ("_rowptr", "_columns", "_weight_values")
        self._recovery_pointers = {name: getattr(self, name).data_ptr() for name in names}
        self._constant_versions = {name: getattr(self, name)._version for name in names
            if name not in {"resource", "resource_factor", "_resource_fault", "rate", "adaptation", "drive", "output_mask"}
            and not getattr(self, name).is_inference()}
        self._csr_hashes = self._source_csr_hashes()
        self._native_csr_pointers = self._live_csr_pointers()
        self._native_csr_version = None if self.weights.is_inference() else self.weights._version
        self._native_constants = {name: getattr(self, name).cpu().numpy().copy() for name in
            ("tau", "a", "cap", "threshold", "coefficient", "_cell_gain")}
        self._fused_buffer_pointers = (tuple(value.data_ptr() for value in self._fused_buffers)
                                      if hasattr(self, "_fused_buffers") else None)

    def _init_resource(self):
        super()._init_resource()
        if not hasattr(self, "resource_factor"):
            with self.torch.inference_mode(False):
                self.resource_factor = self.torch.ones_like(self.resource, dtype=self.torch.float32)
                self._resource_fault = self.torch.zeros(1, dtype=self.torch.int32, device=self.device)

    def _source_csr_hashes(self):
        return tuple(hashlib.sha256(value.tobytes()).digest() for value in
                     (self._source_weights.indptr, self._source_weights.indices, self._source_weights.data))

    def _assert_pointers(self):
        if hasattr(self, "_recovery_pointers") and (
                any(getattr(self, name).data_ptr() != pointer for name, pointer in self._recovery_pointers.items())
                or any(getattr(self, name)._version != version for name, version in self._constant_versions.items())):
            raise ValueError("Captured recovery/native pointer or constant changed")
        if hasattr(self, "_buffer_pointers") and tuple(
                value.data_ptr() for value in self._resource_buffers) != tuple(self._buffer_pointers):
            raise ValueError("Captured resource scratch buffer pointer changed")
        if getattr(self, "_fused_buffer_pointers", None) is not None and tuple(
                value.data_ptr() for value in getattr(self, "_fused_buffers", ())) != self._fused_buffer_pointers:
            raise ValueError("Captured native fused scratch buffer pointer changed")
        if hasattr(self, "_native_csr_pointers") and (
                self._live_csr_pointers() != self._native_csr_pointers
                or (self._native_csr_version is not None and self.weights._version != self._native_csr_version)):
            raise ValueError("Original live CSR pointer or bytes changed")

    def _live_csr_pointers(self):
        return tuple(value.data_ptr() for value in
                     (self.weights.crow_indices(), self.weights.col_indices(), self.weights.values()))

    def _assert_parameters(self):
        self._assert_pointers()
        super()._assert_parameters()
        if hasattr(self, "_csr_hashes") and self._source_csr_hashes() != self._csr_hashes:
            raise ValueError("Original full CSR bytes changed")
        if hasattr(self, "_native_constants") and any(
                not _equal(getattr(self, name).cpu().numpy(), expected)
                for name, expected in self._native_constants.items()):
            raise ValueError("Original neural parameter bytes changed")

    @property
    def resource_fault_flags(self):
        """Explicit diagnostic host copy; successful control uses the merged packet."""
        self._assert_pointers()
        return int(self._resource_fault.detach().cpu().numpy()[0])

    def _cpu_factors(self, rate, resource):
        flags = int(self._resource_fault.numpy()[0])
        if flags:
            raise _fault_error(flags)
        try:
            return resource_factors(rate.numpy(), resource.numpy(), self._selection["source_ids"],
                                    self._selection["utilization"], self._selection["recovery_tau"])
        except RuntimeError as error:
            self._resource_fault.bitwise_or_(error.resource_fault_flags)
            raise

    def _init_fusion(self):
        self._init_resource()
        MotorTimescaleNetwork._init_fusion(self)
        source = resource_cuda_source(self.fuse_tanh)
        for name in ("csr_input", "recovery_factor", "recovery_stage", "recovery_finish"):
            kernel = self._cupy.RawKernel(source, name, options=("--fmad=false",))
            kernel.compile()
            self._fused_kernels[name] = kernel

    def _recovery_launch(self, name, tensors, step=None):
        count = len(self._selection["source_ids"])
        arguments = tuple(np.uint64(value.data_ptr()) for value in tensors)
        if step is not None:
            arguments += (np.float64(step),)
        self._fused_kernels[name](((count+255)//256,), (256,), arguments+(np.int32(count),))

    def _step_fused(self):
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, work_r, work_z, inputs, activation, *derivatives = self._fused_buffers
        kr, kz = derivatives[:3], derivatives[3:]
        work_R, *kR = self._resource_buffers
        self.torch.mul(self.rate, self.output_mask, out=masked)
        r, z, R = self.rate, self.adaptation, self.resource
        with self._cupy.cuda.ExternalStream(self.torch.cuda.current_stream().cuda_stream):
            for stage in range(4):
                self._recovery_launch("recovery_factor", (self._resource_sources, r, R,
                    self._resource_u, self._resource_tau, self.resource_factor, self._resource_fault))
                self._launch_fused("csr_input", (self._rowptr, self._columns, self._weight_values,
                    masked, self.drive, self.threshold, z, self.coefficient, inputs,
                    self._resource_edge_source, self.resource_factor))
                if self.fuse_tanh:
                    activation = inputs
                else:
                    self.torch.tanh(inputs, out=activation)
                if stage < 3:
                    step = self.dt*(.5 if stage < 2 else 1)
                    self._recovery_launch("recovery_stage", (self._resource_sources, r, R, self.resource,
                        self._resource_u, self._resource_tau, kR[stage], work_R), step)
                    self._launch_fused("stage", (activation, self.cap, r, z, self.tau,
                        self.rate, self.adaptation, self.output_mask, kr[stage], kz[stage], work_r, work_z, masked),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, step))
                    r, z, R = work_r, work_z, work_R
                else:
                    self._recovery_launch("recovery_finish", (self._resource_sources, r, R,
                        self._resource_u, self._resource_tau, *kR, self.resource, self._resource_fault), self.dt/6)
                    self._launch_fused("finish", (activation, self.cap, r, z, self.tau,
                        *kr, *kz, self.rate, self.adaptation),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, self.dt/6))

    def _step(self):
        if self.device == "cuda":
            return self._step_fused()
        if self._neutral:
            self._cpu_factors(self.rate, self.resource)
            return MotorTimescaleNetwork._step(self)
        t, dt = self.torch, self.dt
        r0, z0, R0 = self.rate, self.adaptation, self.resource
        r, z, R = r0, z0, R0
        kr, kz, kR = [], [], []
        for stage in range(4):
            factors = self._cpu_factors(r, R)
            self.resource_factor.copy_(t.as_tensor(factors))
            current = t.as_tensor(packed_current_oracle(self._source_weights, r.numpy(),
                self.output_mask.numpy(), self._selection, R.numpy(), factors=factors))
            target = t.clamp_min(self.cap*t.tanh(self.coefficient*(self.drive+current-self.threshold-z)), 0.)
            dr, dz = (target-r)/self.tau, (self._cell_gain*r-z)/self._cell_adaptation_tau
            dR = t.zeros_like(R)
            active = self._resource_u > 0
            dR[active] = (1.-R[active])/self._resource_tau[active]-(
                self._resource_u[active]*r[self._resource_sources[active]].to(t.float64))*R[active]
            kr.append(dr)
            kz.append(dz)
            kR.append(dR)
            if stage < 3:
                step = dt*(.5 if stage < 2 else 1)
                r, z, R = r0+step*dr, z0+step*dz, R0+step*dR
        for state, values in ((r0, kr), (z0, kz), (R0, kR)):
            state.add_(dt/6*(values[0]+2*values[1]+2*values[2]+values[3]))
        flags = _stage_flags(r0.numpy(), R0.numpy(), self._selection["source_ids"], self._selection["utilization"])
        if flags:
            if flags & FAULT_RESOURCE:
                flags = (flags & ~FAULT_RESOURCE) | FAULT_FINAL_RESOURCE
            self._resource_fault.bitwise_or_(flags)
            raise _fault_error(flags)

    def _record_final_faults(self):
        t = self.torch
        R, r, u = self.resource, self.rate[self._resource_sources], self._resource_u
        badR = (~t.isfinite(R)).any() | ((R < 0) | (R > 1)).any()
        bad0 = ((u == 0) & (R != 1)).any()
        # Keep the endpoint guard literal for subnormals as well as the stage
        # kernel; a floating comparison may be compiled with native CUDA FTZ.
        bits = r.view(t.int32)
        magnitude = bits & 0x7fffffff
        badr = ((u > 0) & ((magnitude >= 0x7f800000) |
                ((bits < 0) & (magnitude != 0)))).any()
        flags = badR.to(t.int32)*FAULT_FINAL_RESOURCE
        flags = flags | (bad0.to(t.int32)*FAULT_NEUTRAL_RESOURCE) | (badr.to(t.int32)*FAULT_ACTIVE_RATE)
        self._resource_fault.bitwise_or_(flags.reshape(1))

    def _readout_packet(self, indices=None):
        self._assert_pointers()
        self._record_final_faults()
        t = self.torch
        native_bad = (~t.isfinite(self.rate)).any() | (self.rate < -1e-4).any()
        selected = self.rate if indices is None else t.index_select(self.rate, 0, indices)
        packet = t.cat((selected, native_bad.reshape(1).to(self.rate.dtype),
                        self._resource_fault.to(self.rate.dtype)))
        result = packet.detach().cpu().numpy()
        if result[-1] != 0:
            raise _fault_error(int(result[-1]))
        if result[-2] != 0:
            raise RuntimeError("Invalid raw complete-network rate state")
        return result[:-2].copy()

    def _advance(self, drive, steps, *, readout_indices=None):
        self._assert_pointers()
        values = np.asarray(drive)
        if (values.shape != (self.n,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
            raise ValueError("Finite held rate-model input vector required")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Rate input exceeds float32 range")
        if type(steps) is not int or not 0 < steps <= 10000:
            raise ValueError("Bounded positive integer rate steps required")
        with self.torch.inference_mode():
            self.drive.copy_(self.torch.as_tensor(values, dtype=self.torch.float32))
            for _ in range(steps // self.interval_steps):
                if self.graph is None:
                    self._interval()
                else:
                    self.graph.replay()
            for _ in range(steps % self.interval_steps):
                self._step()
            self.tick += steps
            result = self._readout_packet(readout_indices)
            return result if readout_indices is not None else None

    def readout(self, indices=None):
        if indices is not None:
            ids = np.asarray(indices)
            if (ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu"
                    or np.any(ids < 0) or np.any(ids >= self.n)))):
                raise ValueError("Invalid readout neuron indices")
            indices = self.torch.as_tensor(ids.astype(np.int64), device=self.device)
        return self._readout_packet(indices)

    def resource_current_oracle(self, *, rate=None, resource=None):
        self._assert_parameters()
        if self.resource_fault_flags:
            raise _fault_error(self.resource_fault_flags)
        r = self.rate.cpu().numpy().copy() if rate is None else np.asarray(rate)
        R = self.resource.cpu().numpy().copy() if resource is None else np.asarray(resource)
        if (r.dtype != np.float32 or r.shape != (self.n,) or not np.isfinite(r).all()
                or R.dtype != np.float64 or R.shape != tuple(self.resource.shape) or not np.isfinite(R).all()):
            raise ValueError("Finite simultaneous native rate and aligned resource required")
        return packed_current_oracle(self._source_weights, r, self.output_mask.cpu().numpy(), self._selection, R)

    def snapshot(self):
        self._assert_parameters()
        return {**MotorTimescaleNetwork.snapshot(self), "recurrent_resource": copy.deepcopy(self._metadata),
                "synaptic_resource": self.resource.cpu().numpy().copy()}

    def _prevalidate(self, saved):
        staged = super()._prevalidate(saved)
        active = self._selection["source_ids"][self._selection["utilization"] > 0]
        if np.any(saved["rate"][active] < 0):
            raise ValueError("Active recovery source rate must be nonnegative")
        return staged

    def restore(self, saved):
        staged = self._prevalidate(saved)
        for key, value in staged.items():
            getattr(self, "resource" if key == "synaptic_resource" else key).copy_(value)
        self.tick = saved["tick"]
        self._resource_fault.zero_()

    def reset(self):
        self._assert_parameters()
        super().reset()
        self._resource_fault.zero_()
