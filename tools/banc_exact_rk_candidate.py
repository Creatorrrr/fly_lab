"""Experimental exact RK fusion candidate; production import path is unchanged."""

import numpy as np

from flylab.c.adaptive_rate import AdaptiveRateNetwork

SOURCE = r"""
__device__ __forceinline__ float matching_tanh(float x) {
    // CuPy FTZ would discard subnormal inputs; their correctly rounded tanh
    // is the input itself. Keep the signed-zero bit as well.
    if ((__float_as_uint(x) & 0x7fffffffu) < 0x00800000u) return x;
    return tanhf(x);
}
// CuPy appends -ftz=true. CSR must keep it, whereas Torch pointwise ops
// preserve subnormals. Explicit PTX omits .ftz for pointwise arithmetic only.
__device__ __forceinline__ float add_exact(float a, float b) { float v; asm("add.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b)); return v; }
__device__ __forceinline__ float sub_exact(float a, float b) { float v; asm("sub.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b)); return v; }
__device__ __forceinline__ float mul_exact(float a, float b) { float v; asm("mul.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b)); return v; }
__device__ __forceinline__ float div_exact(float a, float b) { float v; asm("div.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b)); return v; }
__device__ __forceinline__ float clamp_exact(float a) { float v; asm("{.reg .pred p; setp.lt.f32 p, %1, 0f00000000; selp.f32 %0, 0f00000000, %1, p;}" : "=f"(v) : "f"(a)); return v; }

extern "C" __global__ void csr_input(
    const int* rowptr, const unsigned int* entries, const float* lookup,
    const float* masked, const float* drive, const float* threshold,
    const float* adaptation, const float* coefficient, float* input, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        unsigned int item = entries[edge];
        value += lookup[item >> 18] * masked[item & 0x3ffff];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) {
        value = add_exact(drive[row], value);
        value = sub_exact(value, threshold[row]);
        value = sub_exact(value, adaptation[row]);
        input[row] = mul_exact(coefficient[row], value);
    }
}

extern "C" __global__ void stage(
    const float* activation, const float* cap, const float* rate,
    const float* adaptation, const float* tau, const float* base_rate,
    const float* base_adaptation, const float* mask,
    float* k, float* a, float* next_rate, float* next_adaptation, float* next_masked,
    float gain, float inverse_adaptation_tau, float step_size, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = mul_exact(cap[i], activation[i]);
    target = clamp_exact(target);
    float kr = div_exact(sub_exact(target, rate[i]), tau[i]);
    float ka = mul_exact(sub_exact(mul_exact(gain, rate[i]), adaptation[i]), inverse_adaptation_tau);
    k[i] = kr; a[i] = ka;
    float nr = add_exact(base_rate[i], mul_exact(step_size, kr));
    float na = add_exact(base_adaptation[i], mul_exact(step_size, ka));
    next_rate[i] = nr; next_adaptation[i] = na;
    next_masked[i] = mul_exact(nr, mask[i]);
}

extern "C" __global__ void finish(
    const float* activation, const float* cap, const float* rate,
    const float* adaptation, const float* tau,
    const float* k1, const float* k2, const float* k3,
    const float* a1, const float* a2, const float* a3,
    float* base_rate, float* base_adaptation,
    float gain, float inverse_adaptation_tau, float sixth_step, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = mul_exact(cap[i], activation[i]);
    target = clamp_exact(target);
    float kr = div_exact(sub_exact(target, rate[i]), tau[i]);
    float ka = mul_exact(sub_exact(mul_exact(gain, rate[i]), adaptation[i]), inverse_adaptation_tau);
    float sr = add_exact(k1[i], mul_exact(2.0f, k2[i]));
    sr = add_exact(sr, mul_exact(2.0f, k3[i]));
    sr = add_exact(sr, kr);
    float sa = add_exact(a1[i], mul_exact(2.0f, a2[i]));
    sa = add_exact(sa, mul_exact(2.0f, a3[i]));
    sa = add_exact(sa, ka);
    base_rate[i] = add_exact(base_rate[i], mul_exact(sixth_step, sr));
    base_adaptation[i] = add_exact(base_adaptation[i], mul_exact(sixth_step, sa));
}
"""


class ExactRKCandidate(AdaptiveRateNetwork):
    fuse_tanh = False

    def _init_fusion(self):
        cp, torch = self._cupy, self.torch
        source = SOURCE
        if self.fuse_tanh:
            source = source.replace(
                "mul_exact(cap[i], activation[i])",
                "mul_exact(cap[i], matching_tanh(activation[i]))",
            )
        self._fused_kernels = {
            name: cp.RawKernel(source, name, options=("--fmad=false",))
            for name in ("csr_input", "stage", "finish")
        }
        for kernel in self._fused_kernels.values():
            kernel.compile()
        self._fused_buffers = [torch.empty_like(self.rate) for _ in range(11)]

    def _launch_fused(self, name, tensors, scalars=()):
        block = 256
        count = (
            (self.n + 7) // 8 if name == "csr_input" else (self.n + block - 1) // block
        )
        self._fused_kernels[name](
            (count,),
            (block,),
            tuple(np.uint64(t.data_ptr()) for t in tensors)
            + tuple(np.float32(s) for s in scalars)
            + (np.int32(self.n),),
        )

    def _step(self):
        if (
            self.device != "cuda"
            or self.cuda_implementation != "packed"
            or not self.adaptation_gain
        ):
            return super()._step()
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, work_r, work_z, inputs, activation, *derivatives = self._fused_buffers
        ks, azs = derivatives[:3], derivatives[3:]
        self.torch.mul(self.rate, self.output_mask, out=masked)
        r, z = self.rate, self.adaptation
        with self._cupy.cuda.ExternalStream(
            self.torch.cuda.current_stream().cuda_stream
        ):
            for stage in range(4):
                self._launch_fused(
                    "csr_input",
                    (
                        self._rowptr,
                        self._columns,
                        self._weight_values,
                        masked,
                        self.drive,
                        self.threshold,
                        z,
                        self.coefficient,
                        inputs,
                    ),
                )
                if self.fuse_tanh:
                    activation = inputs
                else:
                    self.torch.tanh(inputs, out=activation)
                if stage < 3:
                    self._launch_fused(
                        "stage",
                        (
                            activation,
                            self.cap,
                            r,
                            z,
                            self.tau,
                            self.rate,
                            self.adaptation,
                            self.output_mask,
                            ks[stage],
                            azs[stage],
                            work_r,
                            work_z,
                            masked,
                        ),
                        (
                            self.adaptation_gain,
                            1 / self.adaptation_tau_s,
                            self.dt * (0.5 if stage < 2 else 1),
                        ),
                    )
                    r, z = work_r, work_z
                else:
                    self._launch_fused(
                        "finish",
                        (
                            activation,
                            self.cap,
                            r,
                            z,
                            self.tau,
                            *ks,
                            *azs,
                            self.rate,
                            self.adaptation,
                        ),
                        (self.adaptation_gain, 1 / self.adaptation_tau_s, self.dt / 6),
                    )
