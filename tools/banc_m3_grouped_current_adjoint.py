"""E314 fixed-topology grouped native current with a real-valued surrogate VJP.

The forward is rounded packed-CSR float32; the VJP differentiates a separate
real sparse equation with the already-rounded selected weights. It is not a
derivative through float32 quantization or FTZ.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from tools.banc_m3_source_filtered_synapses import _add, _mul

SCHEMA = "flylab.m3.e314.grouped-current.v1"
PRECISION = (
    "selected_weight=rn-f32-noFTZ(original_weight32*theta32);"
    "neutral-original-product;packed-CSR-warp32-f32-FTZ;"
    "surrogate-VJP:real-W(rounded-theta32)^T,real-unrounded-group-sum;"
    "deterministic-f64-column/group-accumulation;not-quantization-gradient"
)

_CUDA_SOURCE = r"""
__device__ __forceinline__ float scale_exact(float a,float b) {
    float v;asm("mul.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a),"f"(b));return v;
}
__device__ __forceinline__ float to_f32(double a) {
    float v;asm("cvt.rn.f32.f64 %0, %1;" : "=f"(v) : "d"(a));return v;
}
__device__ __forceinline__ double to_f64(float a) {
    // CuPy's default FTZ also affects implicit float-to-double conversion.
    // The surrogate real VJP must retain the input's exact finite float bits.
    double v;asm("cvt.f64.f32 %0, %1;" : "=d"(v) : "f"(a));return v;
}
extern "C" __global__ void grouped_forward(const int* rowptr,
    const unsigned int* entries,const float* lookup,const int* groups,
    const float* rate,const float* theta,float* output,int n) {
    int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(row>=n)return;
    float value=0.0f;
    for(int edge=rowptr[row]+lane;edge<rowptr[row+1];edge+=32) {
        unsigned int item=entries[edge];
        int group=groups[edge];
        if(group<0||theta[group]==1.0f) {
            value+=lookup[item>>18]*rate[item&0x3ffff];
        } else {
            float weight=scale_exact(lookup[item>>18],theta[group]);
            value+=weight*rate[item&0x3ffff];
        }
    }
    for(int offset=16;offset>0;offset>>=1)
        value+=__shfl_down_sync(0xffffffff,value,offset);
    if(lane==0)output[row]=value;
}
extern "C" __global__ void grouped_rate_vjp(const int* ptr,const int* edges,
    const int* post,const float* original,const int* groups,const float* theta,
    const float* grad,float* output,int n) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n)return;
    double sum=0.0;
    for(int k=ptr[col];k<ptr[col+1];++k) {
        int edge=edges[k],group=groups[edge];
        float weight=group<0||theta[group]==1.0f?original[edge]:scale_exact(original[edge],theta[group]);
        sum+=to_f64(weight)*to_f64(grad[post[edge]]);
    }
    output[col]=to_f32(sum);
}
extern "C" __global__ void grouped_efficacy_vjp(const int* ptr,const int* edges,
    const int* pre,const int* post,const float* original,const float* rate,
    const float* grad,double* output,int n) {
    int group=blockIdx.x*blockDim.x+threadIdx.x;if(group>=n)return;
    double sum=0.0;
    for(int k=ptr[group];k<ptr[group+1];++k) {
        int edge=edges[k];
        sum+=(to_f64(grad[post[edge]])*to_f64(original[edge]))*to_f64(rate[pre[edge]]);
    }
    output[group]=sum;
}
"""


class _Current(torch.autograd.Function):
    @staticmethod
    def forward(ctx, masked_rate, efficacy, operator, prepared):
        operator._validate(masked_rate, efficacy, prepared)
        if prepared is None:
            theta = efficacy.detach().to(torch.float32).contiguous()
            operator._check_theta(theta)
        else:
            theta = prepared.theta
        ctx.save_for_backward(masked_rate, efficacy, theta)
        ctx.operator = operator
        ctx.theta = theta.clone()
        ctx.prepared = prepared
        return operator._forward(masked_rate, theta)

    @staticmethod
    def backward(ctx, upstream):
        masked_rate, efficacy, _saved_theta = (
            ctx.saved_tensors
        )  # PyTorch version guards.
        operator = ctx.operator
        operator._validate(masked_rate, efficacy, ctx.prepared)
        if (
            upstream.shape != (operator.n,)
            or upstream.device != operator.device
            or upstream.dtype != torch.float32
        ):
            raise ValueError("Aligned float32 upstream current cotangent required")
        operator._finite(upstream, positive=False)
        rate_grad, efficacy_grad = operator._vjp(
            masked_rate, ctx.theta, upstream.contiguous()
        )
        return rate_grad, efficacy_grad.to(efficacy.dtype), None, None


class PreparedEfficacy:
    """Frozen validated theta for repeated stages; checks its leaf version."""

    __slots__ = ("efficacy", "operator", "theta", "theta_version", "version")

    def __init__(self, operator, efficacy, theta):
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "efficacy", efficacy)
        owned = theta.clone()
        object.__setattr__(self, "theta", owned)
        object.__setattr__(self, "version", efficacy._version)
        object.__setattr__(self, "theta_version", owned._version)

    def __setattr__(self, name, value):
        raise AttributeError(
            "Prepared efficacy fields are immutable; prepare a new context"
        )


class GroupedCurrent:
    """Owned canonical CSR and selected existing nonzero edge-group assignment."""

    def __init__(self, weights, edge_ids, group_ids, group_count, device):
        if (
            not isinstance(weights, csr_matrix)
            or weights.dtype != np.float32
            or weights.shape[0] != weights.shape[1]
            or weights.shape[0] == 0
            or not weights.has_sorted_indices
            or not weights.has_canonical_format
            or not np.isfinite(weights.data).all()
        ):
            raise ValueError("Finite square sorted canonical float32 CSR required")
        weights.check_format(full_check=True)
        # SciPy may cache has_canonical_format before a caller mutates indices.
        # Check the actual row bytes before taking an owned copy.
        if weights.nnz > 1:
            same_row = np.ones(weights.nnz - 1, dtype=bool)
            boundaries = weights.indptr[1:-1]
            boundaries = boundaries[(boundaries > 0) & (boundaries < weights.nnz)]
            same_row[boundaries - 1] = False
            if np.any((np.diff(weights.indices) <= 0) & same_row):
                raise ValueError("Actual CSR row indices must be strictly increasing")
        if not isinstance(device, (str, torch.device)):
            raise TypeError("Explicit CPU or CUDA device required")
        try:
            requested_device = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise ValueError("Explicit CPU or CUDA device required") from error
        if requested_device.type not in ("cpu", "cuda"):
            raise ValueError("Explicit CPU or CUDA device required")
        if type(group_count) is not int or not 0 < group_count < 2**31:
            raise ValueError("Positive bounded group_count required")
        ids, groups = np.asarray(edge_ids), np.asarray(group_ids)
        if (
            ids.dtype != np.int64
            or groups.dtype != np.int64
            or ids.ndim != 1
            or groups.shape != ids.shape
            or len(ids) == 0
            or np.any((ids < 0) | (ids >= weights.nnz))
            or len(np.unique(ids)) != len(ids)
            or np.any((groups < 0) | (groups >= group_count))
            or not np.array_equal(np.unique(groups), np.arange(group_count))
            or np.any(weights.data[ids] == 0)
        ):
            raise ValueError(
                "Unique existing nonzero edge IDs and contiguous group IDs required"
            )
        ordering = np.argsort(ids, kind="stable")
        ids, groups = ids[ordering].copy(), groups[ordering].copy()
        self.n, self.group_count = weights.shape[0], group_count
        self.device = (
            torch.device(
                "cuda",
                torch.cuda.current_device()
                if requested_device.index is None
                else requested_device.index,
            )
            if requested_device.type == "cuda"
            else torch.device("cpu")
        )
        if self.n > 2**18 or weights.nnz >= 2**31:
            raise ValueError("Original packed CSR field bounds exceeded")
        self._weights = weights.copy()
        self._edge_ids, self._selected_groups = ids, groups
        self._edge_groups = np.full(weights.nnz, -1, np.int32)
        self._edge_groups[ids] = groups.astype(np.int32)
        self._pre = weights.indices.astype(np.int32, copy=True)
        self._post = (
            np.searchsorted(weights.indptr, np.arange(weights.nnz), side="right") - 1
        ).astype(np.int32)
        self._original = weights.data.copy()
        order = np.argsort(self._pre, kind="stable").astype(np.int32)
        self._column_edges = order
        self._column_ptr = np.r_[
            0, np.cumsum(np.bincount(self._pre, minlength=self.n))
        ].astype(np.int32)
        group_order = np.argsort(groups, kind="stable")
        self._group_edges = ids[group_order].astype(np.int32)
        self._group_ptr = np.r_[
            0, np.cumsum(np.bincount(groups, minlength=group_count))
        ].astype(np.int32)
        unique, inverse = np.unique(weights.data.view(np.uint32), return_inverse=True)
        if len(unique) > 2**14:
            raise ValueError("Original packed lookup exceeds 14-bit field")
        packed = weights.indices.astype(np.uint32) | (inverse.astype(np.uint32) << 18)
        self._rowptr = weights.indptr.astype(np.int32, copy=True)
        self._entries = packed.view(np.int32).copy()
        self._lookup = unique.view(np.float32).copy()
        self._gpu = None

    @property
    def selected_edge_ids(self):
        return self._edge_ids.copy()

    @property
    def selected_group_ids(self):
        return self._selected_groups.copy()

    def _validate(self, rate, efficacy, prepared=None):
        if prepared is not None and (
            prepared.operator is not self
            or prepared.efficacy is not efficacy
            or prepared.version != efficacy._version
            or prepared.theta._version != prepared.theta_version
        ):
            raise RuntimeError("Prepared efficacy tensor changed after validation")
        if (
            not isinstance(rate, torch.Tensor)
            or rate.dtype != torch.float32
            or rate.shape != (self.n,)
            or rate.device != self.device
            or not rate.is_contiguous()
            or not isinstance(efficacy, torch.Tensor)
            or efficacy.dtype not in (torch.float32, torch.float64)
            or efficacy.shape != (self.group_count,)
            or efficacy.device != self.device
            or not efficacy.is_contiguous()
        ):
            raise ValueError(
                "Finite contiguous float32 rate and positive f32/f64 efficacy on operator device required"
            )
        self._finite(rate, positive=False)
        if prepared is None:
            self._finite(efficacy, positive=True)

    def _finite(self, value, *, positive):
        valid = torch.isfinite(value).all()
        if positive:
            valid = valid & torch.all(value > 0)
        if self.device.type == "cuda":
            # Device assertion is captured/asynchronous; avoid a host sync per SSP stage.
            torch._assert_async(valid, "Nonfinite or nonpositive grouped-current input")
        elif not valid.item():
            raise ValueError("Nonfinite or nonpositive grouped-current input")

    def _check_theta(self, theta):
        if not torch.isfinite(theta).all().item() or not torch.all(theta > 0).item():
            raise ValueError(
                "Efficacy must remain finite positive after float32 conversion"
            )
        chosen = self._original[self._edge_ids]
        theta_cpu = theta.detach().cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.float32(chosen * theta_cpu[self._selected_groups])
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled selected weights exceed native float32")

    def prepare_efficacy(self, efficacy):
        """Optional once-per-trajectory validation for repeated SSP stages."""
        if (
            not isinstance(efficacy, torch.Tensor)
            or efficacy.dtype not in (torch.float32, torch.float64)
            or efficacy.shape != (self.group_count,)
            or efficacy.device != self.device
            or not efficacy.is_contiguous()
        ):
            raise ValueError("Aligned contiguous float32/float64 efficacy required")
        self._finite(efficacy, positive=True)
        theta = efficacy.detach().to(torch.float32).contiguous()
        self._check_theta(theta)
        return PreparedEfficacy(self, efficacy, theta)

    def current(self, masked_rate, efficacy):
        """Native current and real surrogate gradients; mask already applied by caller."""
        prepared = efficacy if isinstance(efficacy, PreparedEfficacy) else None
        tensor = prepared.efficacy if prepared is not None else efficacy
        return _Current.apply(masked_rate, tensor, self, prepared)

    def _forward(self, rate, theta):
        if self.device.type == "cuda":
            gpu = self._cuda()
            output = torch.empty(self.n, device=self.device, dtype=torch.float32)
            with gpu["cupy"].cuda.ExternalStream(
                torch.cuda.current_stream().cuda_stream
            ):
                tensors = (
                    gpu["rowptr"],
                    gpu["entries"],
                    gpu["lookup"],
                    gpu["groups"],
                    rate,
                    theta,
                    output,
                )
                gpu["forward"](
                    ((self.n + 7) // 8,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in tensors)
                    + (np.int32(self.n),),
                )
            return output
        r, t = rate.detach().cpu().numpy(), theta.detach().cpu().numpy()
        output = np.zeros(self.n, np.float32)
        for row in range(self.n):
            lanes = np.zeros(32, np.float32)
            for lane in range(32):
                for edge in range(self._rowptr[row] + lane, self._rowptr[row + 1], 32):
                    group = self._edge_groups[edge]
                    weight = self._original[edge]
                    if group >= 0 and t[group] != 1:
                        weight = np.float32(
                            weight * t[group]
                        )  # No FTZ in selected weight scaling.
                    lanes[lane] = _add(lanes[lane], _mul(weight, r[self._pre[edge]]))
            for offset in (16, 8, 4, 2, 1):
                old = lanes.copy()
                for lane in range(32 - offset):
                    lanes[lane] = _add(old[lane], old[lane + offset])
            output[row] = lanes[0]
        return torch.from_numpy(output.copy())

    def _vjp(self, rate, theta, upstream):
        if self.device.type == "cuda":
            gpu = self._cuda()
            rate_grad = torch.empty_like(rate)
            group_grad = torch.empty(
                self.group_count, device=self.device, dtype=torch.float64
            )
            with gpu["cupy"].cuda.ExternalStream(
                torch.cuda.current_stream().cuda_stream
            ):
                rargs = (
                    gpu["column_ptr"],
                    gpu["column_edges"],
                    gpu["post"],
                    gpu["original"],
                    gpu["groups"],
                    theta,
                    upstream,
                    rate_grad,
                )
                gpu["rate_vjp"](
                    ((self.n + 255) // 256,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in rargs) + (np.int32(self.n),),
                )
                gargs = (
                    gpu["group_ptr"],
                    gpu["group_edges"],
                    gpu["pre"],
                    gpu["post"],
                    gpu["original"],
                    rate,
                    upstream,
                    group_grad,
                )
                gpu["efficacy_vjp"](
                    ((self.group_count + 255) // 256,),
                    (256,),
                    tuple(np.uint64(v.data_ptr()) for v in gargs)
                    + (np.int32(self.group_count),),
                )
            return rate_grad, group_grad
        grad = upstream.detach().cpu().numpy()
        theta_cpu = theta.detach().cpu().numpy()
        rate_cpu = rate.detach().cpu().numpy()
        dr = np.zeros(self.n, np.float64)
        for pre in range(self.n):
            for k in range(self._column_ptr[pre], self._column_ptr[pre + 1]):
                edge = self._column_edges[k]
                group = self._edge_groups[edge]
                weight = self._original[edge]
                if group >= 0 and theta_cpu[group] != 1:
                    weight = np.float32(weight * theta_cpu[group])
                dr[pre] += float(weight) * float(grad[self._post[edge]])
        dg = np.zeros(self.group_count, np.float64)
        for group in range(self.group_count):
            for k in range(self._group_ptr[group], self._group_ptr[group + 1]):
                edge = self._group_edges[k]
                dg[group] += (
                    float(grad[self._post[edge]]) * float(self._original[edge])
                ) * float(rate_cpu[self._pre[edge]])
        return torch.from_numpy(dr.astype(np.float32)), torch.from_numpy(dg)

    def _cuda(self):
        if self._gpu is None:
            import cupy as cp

            arrays = {
                "rowptr": self._rowptr,
                "entries": self._entries,
                "lookup": self._lookup,
                "groups": self._edge_groups,
                "original": self._original,
                "post": self._post,
                "pre": self._pre,
                "column_ptr": self._column_ptr,
                "column_edges": self._column_edges,
                "group_ptr": self._group_ptr,
                "group_edges": self._group_edges,
            }
            gpu = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in arrays.items()
            }
            names = {
                "forward": "grouped_forward",
                "rate_vjp": "grouped_rate_vjp",
                "efficacy_vjp": "grouped_efficacy_vjp",
            }
            for key, name in names.items():
                gpu[key] = cp.RawKernel(_CUDA_SOURCE, name, options=("--fmad=false",))
                gpu[key].compile()
            gpu["cupy"] = cp
            self._gpu = gpu
        return self._gpu
