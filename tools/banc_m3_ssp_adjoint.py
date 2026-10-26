"""Functional E314 SSPRK33 step with native f32 forward and real-equation VJP.

The backward is a surrogate derivative of the unrounded SSP equations. It is
not a derivative of the packed CSR/FTZ and pointwise rounding decisions.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

_POINT_SOURCE = r"""
__device__ __forceinline__ float matching_tanh(float x) {
    if ((__float_as_uint(x) & 0x7fffffffu) < 0x00800000u) return x;
    return tanhf(x);
}
__device__ __forceinline__ float add_exact(float a,float b) {
    float v; asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a),"f"(b)); return v;
}
__device__ __forceinline__ float sub_exact(float a,float b) {
    float v; asm("sub.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a),"f"(b)); return v;
}
__device__ __forceinline__ float mul_exact(float a,float b) {
    float v; asm("mul.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a),"f"(b)); return v;
}
__device__ __forceinline__ float div_exact(float a,float b) {
    float v; asm("div.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a),"f"(b)); return v;
}
__device__ __forceinline__ float clamp_exact(float a) {
    float v; asm("{.reg .pred p; setp.lt.f32 p, %1, 0f00000000; selp.f32 %0, 0f00000000, %1, p;}"
        : "=f"(v) : "f"(a)); return v;
}
extern "C" __global__ void ssp_adjoint_point(
    const float* rate,const float* adaptation,const float* base_rate,
    const float* base_adaptation,const float* current,const float* drive,
    const float* tau,const float* coefficient,const float* threshold,
    const float* cap,const float* gain,const float* inverse,
    float* next_rate,float* next_adaptation,float* activation,
    float h,float base_weight,float work_weight,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    float value=add_exact(drive[i],current[i]);
    value=sub_exact(value,threshold[i]);value=sub_exact(value,adaptation[i]);
    float input=mul_exact(coefficient[i],value);activation[i]=input;
    float target=clamp_exact(mul_exact(cap[i],matching_tanh(input)));
    float qr=div_exact(h,tau[i]),qz=mul_exact(h,inverse[i]);
    float rfe=add_exact(mul_exact(sub_exact(1.0f,qr),rate[i]),mul_exact(qr,target));
    float zfe=add_exact(mul_exact(sub_exact(1.0f,qz),adaptation[i]),
        mul_exact(qz,mul_exact(gain[i],rate[i])));
    next_rate[i]=add_exact(mul_exact(base_weight,base_rate[i]),mul_exact(work_weight,rfe));
    next_adaptation[i]=add_exact(mul_exact(base_weight,base_adaptation[i]),mul_exact(work_weight,zfe));
}
"""


@lru_cache(maxsize=1)
def _cuda_kernel():
    import cupy

    return cupy.RawKernel(_POINT_SOURCE, "ssp_adjoint_point", options=("--fmad=false",))


def _point_cpu(rate, adaptation, base_rate, base_adaptation, current, drive,
               tau, coefficient, threshold, cap, gain, inverse, h, base_weight, work_weight):
    value = drive + current
    value = value - threshold
    value = value - adaptation
    activation = coefficient * value
    target = torch.clamp_min(cap * torch.tanh(activation), 0.)
    qr = h / tau
    qz = h * inverse
    rfe = (1. - qr) * rate + qr * target
    zfe = (1. - qz) * adaptation + qz * (gain * rate)
    next_rate = base_weight * base_rate + work_weight * rfe
    next_adaptation = base_weight * base_adaptation + work_weight * zfe
    return next_rate, next_adaptation, activation


def _require(condition, message, device):
    """Immediate CPU error; CUDA predicate stays on device without host sync."""
    if device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class _SSPPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rate, adaptation, base_rate, base_adaptation, current, drive,
                tau, coefficient, threshold, cap, gain, inverse, h, base_weight, work_weight):
        if rate.is_cuda:
            import cupy

            nr, nz, activation = (torch.empty_like(rate) for _ in range(3))
            tensors = (rate, adaptation, base_rate, base_adaptation, current, drive,
                       tau, coefficient, threshold, cap, gain, inverse, nr, nz, activation)
            with cupy.cuda.ExternalStream(torch.cuda.current_stream(rate.device).cuda_stream):
                _cuda_kernel()(((rate.numel() + 255) // 256,), (256,),
                    tuple(np.uint64(value.data_ptr()) for value in tensors) +
                    (np.float32(h), np.float32(base_weight), np.float32(work_weight), np.int32(rate.numel())))
        else:
            nr, nz, activation = _point_cpu(rate, adaptation, base_rate, base_adaptation,
                current, drive, tau, coefficient, threshold, cap, gain, inverse,
                h, base_weight, work_weight)
        ctx.save_for_backward(rate, adaptation, base_rate, base_adaptation, current, drive,
                              tau, coefficient, threshold, cap, gain, inverse, activation)
        ctx.h, ctx.base_weight, ctx.work_weight = h, base_weight, work_weight
        return nr, nz

    @staticmethod
    def backward(ctx, grad_rate, grad_adaptation):
        rate, adaptation, base_rate, base_adaptation, current, drive, tau, coefficient, threshold, cap, gain, inverse, activation = ctx.saved_tensors
        del rate, adaptation, base_rate, base_adaptation, current, drive, threshold
        if grad_rate is None:
            grad_rate = torch.zeros_like(activation)
        if grad_adaptation is None:
            grad_adaptation = torch.zeros_like(activation)
        qr = ctx.h / tau
        qz = ctx.h * inverse
        # Native clamp returns zero at its exact rectifier kink by contract.
        slope = torch.where(activation > 0, cap * (1. - torch.tanh(activation).square()), 0.)
        through_target = ctx.work_weight * qr * slope * coefficient * grad_rate
        grad_r = ctx.work_weight * (1. - qr) * grad_rate + ctx.work_weight * qz * gain * grad_adaptation
        grad_z = ctx.work_weight * (1. - qz) * grad_adaptation - through_target
        grad_r0 = ctx.base_weight * grad_rate
        grad_z0 = ctx.base_weight * grad_adaptation
        return (grad_r, grad_z, grad_r0, grad_z0, through_target, through_target,
                None, None, None, None, None, None, None, None, None)


class SSPAdjointModel:
    """One functional SSP step; prepare efficacy once for a multi-tick rollout."""

    def __init__(self, operator, tau, coefficient, threshold, cap, gain, inverse_adaptation_tau, dt):
        if not callable(getattr(operator, "current", None)) or type(getattr(operator, "n", None)) is not int or operator.n < 1:
            raise TypeError("Grouped native current operator with positive n required")
        self.operator = operator
        self.n = operator.n
        self.device = torch.device(operator.device)
        names = ("tau", "coefficient", "threshold", "cap", "gain", "inverse_adaptation_tau")
        values = (tau, coefficient, threshold, cap, gain, inverse_adaptation_tau)
        for name, value in zip(names, values, strict=True):
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32 or value.ndim != 1 or value.shape != (self.n,) or value.device != self.device:
                    raise ValueError("Resolved native float32 vector/device required: " + name)
                owned = value.detach().clone().contiguous()
            else:
                array = np.asarray(value)
                if array.dtype != np.float32 or array.shape != (self.n,):
                    raise ValueError("Resolved native float32 vector required: " + name)
                owned = torch.as_tensor(array.copy(), device=self.device)
            if not bool(torch.isfinite(owned).all()):
                raise ValueError("Finite native physiology required: " + name)
            if name in ("tau", "cap") and bool((owned <= 0).any()):
                raise ValueError("Positive native physiology required: " + name)
            if name in ("gain", "inverse_adaptation_tau") and bool((owned < 0).any()):
                raise ValueError("Nonnegative native physiology required: " + name)
            setattr(self, name, owned)
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float, np.floating)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Finite positive SSP dt required")
        self.dt = float(np.float32(dt))
        if self.dt <= 0 or bool((self.dt / self.tau > .5).any()):
            raise ValueError("Resolved SSP h/tau must be at most .5")

    def prepare_efficacy(self, efficacy):
        """Cache checked theta32 once for a trajectory without detaching its VJP."""
        prepare = getattr(self.operator, "prepare_efficacy", None)
        if not callable(prepare):
            raise TypeError("Grouped current operator does not provide prepared efficacy")
        return prepare(efficacy)

    def step(self, rate, adaptation, drive, efficacy, mask):
        for name, value in (("rate", rate), ("adaptation", adaptation), ("drive", drive), ("mask", mask)):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (self.n,)
                    or value.device != self.device or not value.is_contiguous()):
                raise ValueError("Contiguous native f32 same-device vector required: " + name)
            _require(torch.isfinite(value).all(), "Finite native vector required: " + name, self.device)
        _require(torch.all(rate >= 0) & torch.all(adaptation >= 0), "Nonnegative SSP state required", self.device)
        _require(torch.all((mask == 0) | (mask == 1)), "Binary native output mask required", self.device)
        prepared = efficacy if hasattr(efficacy, "efficacy") and hasattr(efficacy, "version") else None
        source = prepared.efficacy if prepared is not None else efficacy
        if (not isinstance(source, torch.Tensor) or source.dtype not in (torch.float32, torch.float64)
                or source.shape != (self.operator.group_count,) or source.device != self.device
                or not source.is_contiguous()):
            raise ValueError("Contiguous same-device grouped efficacy required")
        if prepared is not None and (prepared.operator is not self.operator or prepared.version != source._version):
            raise RuntimeError("Prepared efficacy belongs to another operator or was mutated")
        if prepared is None:
            _require(torch.isfinite(source).all() & torch.all(source > 0),
                "Finite positive grouped efficacy required", self.device)
        r0, z0 = rate, adaptation
        r, z = r0, z0
        if prepared is None:
            prepare = getattr(self.operator, "prepare_efficacy", None)
            prepared = prepare(source) if callable(prepare) else source
        stages = ((np.float32(0), np.float32(1)), (np.float32(.75), np.float32(.25)),
                  (np.float32(1. / 3.), np.float32(2. / 3.)))
        for base, work in stages:
            masked = (r * mask).contiguous()
            current = self.operator.current(masked, prepared)
            if (not isinstance(current, torch.Tensor) or current.dtype != torch.float32
                    or current.shape != (self.n,) or current.device != self.device or not current.is_contiguous()):
                raise ValueError("Contiguous native f32 grouped current required")
            _require(torch.isfinite(current).all(), "Finite native grouped current required", self.device)
            r, z = _SSPPoint.apply(r, z, r0, z0, current.contiguous(), drive.contiguous(),
                self.tau, self.coefficient, self.threshold, self.cap, self.gain,
                self.inverse_adaptation_tau, self.dt, float(base), float(work))
            _require(torch.isfinite(r).all() & torch.isfinite(z).all() & torch.all(r >= 0) & torch.all(z >= 0),
                "Finite nonnegative SSP state required after each stage", self.device)
        return r, z
