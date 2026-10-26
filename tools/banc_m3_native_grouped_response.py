"""E221 native recurrent, muscle-shared response on the packed rate network.

This class is a neural component only.  It neither builds a walking session nor
changes the original adapter.  The float64 state is kept beside, never in place
of, the original float32 full-neuron rate and adaptation tensors.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tools.banc_m3_motor_timescale import (
    MotorTimescaleNetwork,
    _same_bits,
    _timescale_source,
)

SCHEMA = "flylab.m3.native-grouped-response.v1"
PRECISION_CONTRACT = (
    "packed-csr-f32-total-and-signed;f32-full-rate-adaptation-mask-readout;"
    "free-r-fast-slow-and-reaction-f64;free-stage-cast-f32;rk4-dt-f64"
)
BASIS_COUNT = 64
HISTORY_TAU = .016


def _parameter_fingerprint(values):
    digest = hashlib.sha256(SCHEMA.encode() + PRECISION_CONTRACT.encode())
    for value in values:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def validate_native_parameters(n, motor_ids, free_motor_ids, fixed_motor_ids,
                               membership, scale, centers, sigma2, beta,
                               tau, cap, inverse_slow):
    """Return byte-preserving copies of the ordered, owned runtime parameters."""
    ids, free, fixed, owner = (np.asarray(v) for v in
        (motor_ids, free_motor_ids, fixed_motor_ids, membership))
    if any(v.dtype != np.int64 or v.ndim != 1 for v in (ids, free, fixed, owner)):
        raise ValueError("Ordered int64 motor/free/fixed/owner arrays required")
    if (len(ids) < 2 or len(free) < 1 or len(fixed) < 1
            or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= n))
            or not np.array_equal(free, ids[np.isin(ids, free)])
            or not np.array_equal(fixed, ids[np.isin(ids, fixed)])
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.sort(ids))):
        raise ValueError("Free and fixed motor IDs must partition the ordered roster")
    f = len(free)
    arrays = [np.asarray(v) for v in (scale, centers, sigma2, beta, tau, cap,
                                      inverse_slow)]
    if any(v.dtype != np.float64 or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Finite float64 response parameters required")
    scale, centers, sigma2, beta, tau, cap, inverse_slow = arrays
    groups = len(beta)
    if (owner.shape != (f,) or groups < 1 or np.any((owner < 0) | (owner >= groups))
            or scale.shape != (f, 3) or centers.shape != (f, 64, 9)
            or sigma2.shape != (f,) or beta.shape != (groups, 64)
            or any(v.shape != (f,) for v in (tau, cap, inverse_slow))
            or np.any(scale <= 0) or np.any(sigma2 <= 0)
            or np.any(tau <= 0) or np.any(cap <= 0) or np.any(inverse_slow <= 0)
            or np.any((beta < 0) | (beta > 1))):
        raise ValueError("Owned, bounded 64-center native response required")
    unused = np.bincount(owner, minlength=groups) == 0
    if np.any(beta[unused] != 0):
        raise ValueError("Unused fixed-only muscle groups require zero beta rows")
    return tuple(v.copy() for v in (ids, free, fixed, owner, *arrays))


def native_response_derivative(positive, negative, drive, rate, fast, slow,
                               scale, centers, sigma2, beta, membership, tau,
                               cap, inverse_slow):
    """Small pure-NumPy stage oracle; no recurrent network is constructed."""
    e, i, d = (np.asarray(v, np.float64) for v in (positive, negative, drive))
    x = np.stack((np.log1p(e) / scale[:, 0], np.log1p(i) / scale[:, 1],
                  np.arcsinh(d) / scale[:, 2]), axis=1)
    descriptor = np.concatenate((x, fast, slow), axis=1)
    logits = -np.sum((descriptor[:, None] - centers)**2, axis=2) / (2 * sigma2[:, None])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    gate = -np.expm1(-8 * (x[:, 0] + np.abs(x[:, 2])
                            + np.maximum(fast[:, 0], 0) + np.abs(fast[:, 2])))
    phi = cap * gate * np.sum(p * beta[membership], axis=1)
    return ((phi - rate) / tau, (x - fast) / HISTORY_TAU,
            (x - slow) * inverse_slow[:, None])


def _native_source(fuse_tanh):
    """Preserve original CSR total arithmetic and original stage/finish code."""
    source = _timescale_source(fuse_tanh)
    signature = "float* input, int n) {"
    loop = "value += lookup[item >> 18] * masked[item & 0x3ffff];"
    reduction = "value += __shfl_down_sync(0xffffffff, value, offset);"
    point = "value = add_exact(drive[row], value);"
    if any(source.count(s) != 1 for s in (signature, loop, reduction, point)):
        raise RuntimeError("Frozen packed CSR insertion points changed")
    source = source.replace(signature,
        "float* input, const int* free_index, float* positive, float* negative, int n) {", 1)
    source = source.replace("float value = 0.0f;",
        "int column = free_index[row]; float e = 0.0f, inhibitory = 0.0f;\n"
        "    float value = 0.0f;", 1)
    source = source.replace(loop, loop + "\n"
        "        if (column >= 0) { float weight = lookup[item >> 18];\n"
        "            float product = weight * masked[item & 0x3ffff];\n"
        "            if (weight > 0.0f) e += product;\n"
        "            if (weight < 0.0f) inhibitory += -product; }", 1)
    source = source.replace(reduction,
        "{ value += __shfl_down_sync(0xffffffff, value, offset);\n"
        "      e += __shfl_down_sync(0xffffffff, e, offset);\n"
        "      inhibitory += __shfl_down_sync(0xffffffff, inhibitory, offset); }", 1)
    source = source.replace(point,
        "if (column >= 0) { positive[column] = e; negative[column] = inhibitory; }\n"
        "        " + point, 1)
    return source + _NATIVE_KERNELS


_NATIVE_KERNELS = r"""
__device__ __forceinline__ void response(
    int j, const float* e, const float* inhibitory, const float* drive,
    const int* ids, const int* owner, const double* scale, const double* center,
    const double* sigma, const double* beta, const double* tau, const double* cap,
    const double* inverse, const double* r, const double* fast, const double* slow,
    double* kr, double* kf, double* ks) {
    double x[3] = {log1p((double)e[j])/scale[3*j],
                   log1p((double)inhibitory[j])/scale[3*j+1],
                   asinh((double)drive[ids[j]])/scale[3*j+2]};
    double logits[64], largest = -1.0e300;
    for (int a=0; a<64; a++) {
        double distance=0.;
        for (int b=0; b<3; b++) {
            double q=x[b]-center[576*j+9*a+b];
            double f=fast[3*j+b]-center[576*j+9*a+3+b];
            double s=slow[3*j+b]-center[576*j+9*a+6+b];
            distance += q*q+f*f+s*s;
        }
        logits[a]=-distance/(2.*sigma[j]);
        if (logits[a]>largest) largest=logits[a];
    }
    double sum=0., weighted=0.;
    for (int a=0; a<64; a++) sum += exp(logits[a]-largest);
    for (int a=0; a<64; a++)
        weighted += exp(logits[a]-largest)/sum*beta[64*owner[j]+a];
    double argument=x[0]+fabs(x[2])+fmax(fast[3*j],0.)+fabs(fast[3*j+2]);
    double gate=-expm1(-8.*argument);
    *kr=(cap[j]*gate*weighted-r[j])/tau[j];
    for (int a=0; a<3; a++) {
        kf[a]=(x[a]-fast[3*j+a])/.016;
        ks[a]=(x[a]-slow[3*j+a])*inverse[j];
    }
}
extern "C" __global__ void native_stage(
    const float* e,const float* inhibitory,const float* drive,
    const int* ids,const int* owner,const double* scale,const double* center,
    const double* sigma,const double* beta,const double* tau,const double* cap,
    const double* inverse,const double* base_r,const double* base_f,const double* base_s,
    const double* stage_r,const double* stage_f,const double* stage_s,
    double* kr,double* kf,double* ks,double* next_r,double* next_f,double* next_s,
    float* full_rate,const float* mask,float* masked,double step,int f) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=f) return;
    double dr,df[3],ds[3];
    response(j,e,inhibitory,drive,ids,owner,scale,center,sigma,beta,tau,cap,
             inverse,stage_r,stage_f,stage_s,&dr,df,ds);
    kr[j]=dr; next_r[j]=base_r[j]+step*dr;
    int row=ids[j];
    float cast=(float)next_r[j];
    full_rate[row]=cast; masked[row]=mul_exact(cast,mask[row]);
    for(int a=0;a<3;a++) {
        kf[3*j+a]=df[a]; ks[3*j+a]=ds[a];
        next_f[3*j+a]=base_f[3*j+a]+step*df[a];
        next_s[3*j+a]=base_s[3*j+a]+step*ds[a];
    }
}
extern "C" __global__ void native_finish(
    const float* e,const float* inhibitory,const float* drive,
    const int* ids,const int* owner,const double* scale,const double* center,
    const double* sigma,const double* beta,const double* tau,const double* cap,
    const double* inverse,double* base_r,double* base_f,double* base_s,
    const double* stage_r,const double* stage_f,const double* stage_s,
    const double* kr,const double* kf,const double* ks,
    float* full_rate,double sixth,int f) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=f) return;
    double dr,df[3],ds[3];
    response(j,e,inhibitory,drive,ids,owner,scale,center,sigma,beta,tau,cap,
             inverse,stage_r,stage_f,stage_s,&dr,df,ds);
    base_r[j] += sixth*(kr[j]+2.*kr[f+j]+2.*kr[2*f+j]+dr);
    full_rate[ids[j]]=(float)base_r[j];
    for(int a=0;a<3;a++) {
        int offset=3*j+a, stride=3*f;
        base_f[offset] += sixth*(kf[offset]+2.*kf[stride+offset]
                          +2.*kf[2*stride+offset]+df[a]);
        base_s[offset] += sixth*(ks[offset]+2.*ks[stride+offset]
                          +2.*ks[2*stride+offset]+ds[a]);
    }
}
"""


class NativeGroupedResponseNetwork(MotorTimescaleNetwork):
    """Actual same-stage signed-input recurrent replacement for free MNs."""

    def __init__(self, *args, free_motor_ids, fixed_motor_ids, membership,
                 scale, centers, sigma2, beta, **kwargs):
        import torch

        if (kwargs.get("device") != "cuda" or kwargs.get("cuda_implementation") != "packed"
                or kwargs.get("dt") != .00025 or not torch.cuda.is_available()):
            raise ValueError("Native grouped response requires packed CUDA dt=.00025")
        n = args[0].shape[0]
        motor_ids = np.asarray(kwargs["motor_ids"])
        tau32 = np.asarray(args[1], np.float32)
        cap32 = np.asarray(args[4], np.float32)
        # MotorTimescaleNetwork computes the actual per-cell inverse by this rule.
        from tools.banc_m3_motor_timescale import canonical_timescales
        _, _, inverse32 = canonical_timescales(motor_ids,
            np.asarray(kwargs["motor_log_tau"]), n)
        free = np.asarray(free_motor_ids)
        values = validate_native_parameters(n, motor_ids, free_motor_ids,
            fixed_motor_ids, membership, scale, centers, sigma2, beta,
            tau32[free].astype(np.float64), cap32[free].astype(np.float64),
            inverse32[free].astype(np.float64))
        (self.native_motor_ids, self.free_motor_ids, self.fixed_motor_ids,
         self.membership, self.scale, self.centers, self.sigma2, self.beta,
         self.free_tau, self.free_cap, self.inverse_slow) = values
        self._free_count = len(self.free_motor_ids)
        self._free_row = torch.as_tensor(self.free_motor_ids.astype(np.int32), device="cuda")
        self._owner = torch.as_tensor(self.membership.astype(np.int32), device="cuda")
        mapping = np.full(n, -1, np.int32)
        mapping[self.free_motor_ids] = np.arange(self._free_count, dtype=np.int32)
        self._free_index = torch.as_tensor(mapping, device="cuda")
        self._positive = torch.zeros(self._free_count, dtype=torch.float32, device="cuda")
        self._negative = torch.zeros_like(self._positive)
        for key in ("scale", "centers", "sigma2", "beta", "free_tau",
                    "free_cap", "inverse_slow"):
            setattr(self, "_d_" + key,
                    torch.as_tensor(getattr(self, key).copy(), device="cuda"))
        self._free_rate64 = torch.zeros(self._free_count, dtype=torch.float64, device="cuda")
        self._free_fast64 = torch.zeros((self._free_count, 3), dtype=torch.float64, device="cuda")
        self._free_slow64 = torch.zeros_like(self._free_fast64)
        self._stage_rate64 = torch.empty_like(self._free_rate64)
        self._stage_fast64 = torch.empty_like(self._free_fast64)
        self._stage_slow64 = torch.empty_like(self._free_slow64)
        self._k_rate64 = torch.empty((3, self._free_count), dtype=torch.float64, device="cuda")
        self._k_fast64 = torch.empty((3, self._free_count, 3), dtype=torch.float64, device="cuda")
        self._k_slow64 = torch.empty_like(self._k_fast64)
        super().__init__(*args, **kwargs)
        self._free_rate64.zero_()
        self._free_fast64.zero_()
        self._free_slow64.zero_()
        self.rate.zero_()
        self.adaptation.zero_()
        self.drive.zero_()
        self.output_mask.fill_(1)
        self.tick = 0
        self.source_timescale_identity = self.identity
        self._native_fingerprint = _parameter_fingerprint(values)
        checksum = hashlib.sha256(SCHEMA.encode() + PRECISION_CONTRACT.encode()
                                  + self.identity.encode()
                                  + self._native_fingerprint.encode())
        self.identity = checksum.hexdigest()
        names = ("_free_row", "_owner", "_free_index", "_positive", "_negative",
                 "_free_rate64", "_free_fast64", "_free_slow64", "_stage_rate64",
                 "_stage_fast64", "_stage_slow64", "_k_rate64", "_k_fast64",
                 "_k_slow64") + tuple("_d_" + key for key in
                    ("scale", "centers", "sigma2", "beta", "free_tau",
                     "free_cap", "inverse_slow"))
        self._native_pointers = {key: getattr(self, key).data_ptr() for key in names}
        self._assert_native_parameters()

    def _assert_native_parameters(self):
        if any(getattr(self, key).data_ptr() != ptr
               for key, ptr in self._native_pointers.items()):
            raise ValueError("Captured native grouped pointer changed")
        current = (self.native_motor_ids, self.free_motor_ids,
                   self.fixed_motor_ids, self.membership, self.scale,
                   self.centers, self.sigma2, self.beta, self.free_tau,
                   self.free_cap, self.inverse_slow)
        if _parameter_fingerprint(current) != self._native_fingerprint:
            raise ValueError("Native grouped construction-time parameter drift")
        mapping = np.full(self.n, -1, np.int32)
        mapping[self.free_motor_ids] = np.arange(self._free_count, dtype=np.int32)
        expected_maps = {"_free_row": self.free_motor_ids.astype(np.int32),
                         "_owner": self.membership.astype(np.int32),
                         "_free_index": mapping}
        for key, expected in expected_maps.items():
            actual = getattr(self, key).detach().cpu().numpy().copy()
            if not _same_bits(actual, expected):
                raise ValueError("Captured native grouped motor/owner map changed")
        for key in ("scale", "centers", "sigma2", "beta", "free_tau",
                    "free_cap", "inverse_slow"):
            actual = getattr(self, "_d_" + key).detach().cpu().numpy().copy()
            if not _same_bits(actual, getattr(self, key)):
                raise ValueError("Captured native grouped parameter changed")
        if (not _same_bits(self.tau.detach().cpu().numpy()[self.free_motor_ids]
                           .astype(np.float64), self.free_tau)
                or not _same_bits(self.cap.detach().cpu().numpy()[self.free_motor_ids]
                                  .astype(np.float64), self.free_cap)
                or not _same_bits(self._full_inverse32[self.free_motor_ids]
                                  .astype(np.float64), self.inverse_slow)):
            raise ValueError("Original float32 motor physiology changed")

    def _init_fusion(self):
        super()._init_fusion()
        source = _native_source(self.fuse_tanh)
        for name in ("csr_input", "native_stage", "native_finish"):
            kernel = self._cupy.RawKernel(source, name, options=("--fmad=false",))
            kernel.compile()
            self._fused_kernels[name] = kernel

    def _launch_fused(self, name, tensors, scalars=()):
        if name == "csr_input":
            tensors = (*tensors, self._free_index, self._positive, self._negative)
        return super()._launch_fused(name, tensors, scalars)

    def _launch_native(self, name, tensors, step):
        self._fused_kernels[name](((self._free_count + 127)//128,), (128,),
            tuple(np.uint64(t.data_ptr()) for t in tensors)
            + (np.float64(step), np.int32(self._free_count)))

    def _native_common(self):
        return (self._positive, self._negative, self.drive, self._free_row,
                self._owner, self._d_scale, self._d_centers, self._d_sigma2,
                self._d_beta, self._d_free_tau, self._d_free_cap,
                self._d_inverse_slow)

    def _step_fused(self):
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, work_r, work_z, inputs, activation, *derivatives = self._fused_buffers
        ks, azs = derivatives[:3], derivatives[3:]
        self.torch.mul(self.rate, self.output_mask, out=masked)
        r, z = self.rate, self.adaptation
        rr, ff, ss = self._free_rate64, self._free_fast64, self._free_slow64
        with self._cupy.cuda.ExternalStream(self.torch.cuda.current_stream().cuda_stream):
            for stage in range(4):
                self._launch_fused("csr_input", (self._rowptr, self._columns,
                    self._weight_values, masked, self.drive, self.threshold,
                    z, self.coefficient, inputs))
                if self.fuse_tanh:
                    activation = inputs
                else:
                    self.torch.tanh(inputs, out=activation)
                if stage < 3:
                    step = self.dt * (.5 if stage < 2 else 1.)
                    self._launch_fused("stage", (activation, self.cap, r, z,
                        self.tau, self.rate, self.adaptation, self.output_mask,
                        ks[stage], azs[stage], work_r, work_z, masked),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, step))
                    self._launch_native("native_stage", (*self._native_common(),
                        self._free_rate64, self._free_fast64, self._free_slow64,
                        rr, ff, ss, self._k_rate64[stage], self._k_fast64[stage],
                        self._k_slow64[stage], self._stage_rate64,
                        self._stage_fast64, self._stage_slow64, work_r,
                        self.output_mask, masked), step)
                    r, z = work_r, work_z
                    rr, ff, ss = (self._stage_rate64, self._stage_fast64,
                                  self._stage_slow64)
                else:
                    self._launch_fused("finish", (activation, self.cap, r, z,
                        self.tau, *ks, *azs, self.rate, self.adaptation),
                        (self.adaptation_gain, 1/self.adaptation_tau_s, self.dt/6))
                    self._launch_native("native_finish", (*self._native_common(),
                        self._free_rate64, self._free_fast64, self._free_slow64,
                        rr, ff, ss, self._k_rate64, self._k_fast64,
                        self._k_slow64, self.rate), self.dt/6)

    def reset(self):
        super().reset()
        self.adaptation.zero_()
        self._free_rate64.zero_()
        self._free_fast64.zero_()
        self._free_slow64.zero_()

    def snapshot(self):
        self._assert_native_parameters()
        saved = super().snapshot()
        saved["native_grouped_response"] = {
            "schema": SCHEMA, "precision_contract": PRECISION_CONTRACT,
            "parameter_fingerprint": self._native_fingerprint,
            "source_timescale_identity": self.source_timescale_identity,
            "motor_ids": self.native_motor_ids.copy(),
            "free_ids": self.free_motor_ids.copy(),
            "fixed_ids": self.fixed_motor_ids.copy(),
            "membership": self.membership.copy(),
            "scale": self.scale.copy(), "centers": self.centers.copy(),
            "sigma2": self.sigma2.copy(), "beta": self.beta.copy(),
            "tau": self.free_tau.copy(), "cap": self.free_cap.copy(),
            "inverse_slow": self.inverse_slow.copy(),
            "rate64": self._free_rate64.cpu().numpy().copy(),
            "fast64": self._free_fast64.cpu().numpy().copy(),
            "slow64": self._free_slow64.cpu().numpy().copy()}
        return saved

    def restore(self, saved):
        self._assert_native_parameters()
        patch = saved.get("native_grouped_response")
        expected = {"motor_ids": self.native_motor_ids,
            "free_ids": self.free_motor_ids, "fixed_ids": self.fixed_motor_ids,
            "membership": self.membership, "scale": self.scale,
            "centers": self.centers, "sigma2": self.sigma2, "beta": self.beta,
            "tau": self.free_tau, "cap": self.free_cap,
            "inverse_slow": self.inverse_slow}
        keys = set(expected) | {"schema", "precision_contract",
                                 "parameter_fingerprint", "source_timescale_identity",
                                 "rate64", "fast64", "slow64"}
        if (not isinstance(patch, dict) or set(patch) != keys
                or patch["schema"] != SCHEMA
                or patch["precision_contract"] != PRECISION_CONTRACT
                or patch["parameter_fingerprint"] != self._native_fingerprint
                or patch["source_timescale_identity"] != self.source_timescale_identity
                or saved.get("model_hash") != self.identity):
            raise ValueError("Complete native grouped response identity required")
        for key, value in expected.items():
            if (not isinstance(patch[key], np.ndarray)
                    or not _same_bits(patch[key], value)):
                raise ValueError("Native grouped response parameter bytes mismatch")
        states = (patch["rate64"], patch["fast64"], patch["slow64"])
        shapes = ((self._free_count,), (self._free_count, 3),
                  (self._free_count, 3))
        if any(not isinstance(v, np.ndarray) or v.dtype != np.float64
               or v.shape != shape or not np.isfinite(v).all()
               for v, shape in zip(states, shapes)) or np.any(states[0] < 0):
            raise ValueError("Finite nonnegative native grouped state required")
        full = saved.get("rate")
        if (not isinstance(full, np.ndarray) or full.dtype != np.float32
                or full.shape != (self.n,)
                or not _same_bits(full[self.free_motor_ids],
                                  states[0].astype(np.float32))):
            raise ValueError("Native free rate must match full float32 rate")
        # Validate all inherited metadata and arrays on a scratch network-free
        # path before the inherited restore mutates any live device tensor.
        self._validate_base_snapshot(saved)
        super().restore(saved)
        for target, value in zip((self._free_rate64, self._free_fast64,
                                  self._free_slow64), states):
            target.copy_(self.torch.as_tensor(value, device=self.device))

    def _validate_base_snapshot(self, saved):
        if (saved.get("schema") != "flylab.research-rate.v1"
                or saved.get("model_hash") != self.identity):
            raise ValueError("Native base checkpoint schema/identity mismatch")
        for key in ("rate", "drive", "output_mask", "adaptation"):
            value = saved.get(key)
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != (self.n,) or not np.isfinite(value).all()):
                raise ValueError("Native base checkpoint arrays required")
        if (np.any(saved["rate"] < -1e-4)
                or not np.isin(saved["output_mask"], (0, 1)).all()
                or type(saved.get("tick")) is not int or saved["tick"] < 0
                or saved.get("dt") != self.dt or saved.get("device") != self.device):
            raise ValueError("Native base checkpoint values invalid")


__all__ = ("NativeGroupedResponseNetwork", "native_response_derivative",
           "validate_native_parameters")
