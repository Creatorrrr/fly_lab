"""Candidate-only CSR plus complete RK-stage fusion, with masked ping-pong."""

import numpy as np

from flylab.c.adaptive_fused import SOURCE as POINTWISE_SOURCE
from flylab.c.adaptive_rate import AdaptiveRateNetwork

SOURCE = (
    POINTWISE_SOURCE.split('extern "C" __global__ void csr_input', 1)[0]
    + r"""
extern "C" __global__ void csr_stage(
    const int* rowptr, const unsigned int* entries, const float* lookup,
    const float* masked, const float* drive, const float* threshold,
    const float* adaptation, const float* coefficient,
    const float* cap, const float* rate, const float* tau,
    float* base_rate, float* base_adaptation, const float* mask,
    float* k1, float* k2, float* k3, float* a1, float* a2, float* a3,
    float* next_rate, float* next_adaptation, float* next_masked,
    float gain, float inverse_adaptation_tau, float step_size, float sixth_step,
    int stage, int n) {
    int lane = threadIdx.x & 31;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (i >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[i] + lane; edge < rowptr[i + 1]; edge += 32) {
        unsigned int item = entries[edge];
        value += lookup[item >> 18] * masked[item & 0x3ffff];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane != 0) return;
    value = add_exact(drive[i], value);
    value = sub_exact(value, threshold[i]);
    value = sub_exact(value, adaptation[i]);
    value = mul_exact(coefficient[i], value);
    float target = clamp_exact(mul_exact(cap[i], matching_tanh(value)));
    float kr = div_exact(sub_exact(target, rate[i]), tau[i]);
    float ka = mul_exact(sub_exact(mul_exact(gain, rate[i]), adaptation[i]), inverse_adaptation_tau);
    if (stage == 3) {
        float sr = add_exact(k1[i], mul_exact(2.0f, k2[i]));
        sr = add_exact(sr, mul_exact(2.0f, k3[i]));
        sr = add_exact(sr, kr);
        float sa = add_exact(a1[i], mul_exact(2.0f, a2[i]));
        sa = add_exact(sa, mul_exact(2.0f, a3[i]));
        sa = add_exact(sa, ka);
        base_rate[i] = add_exact(base_rate[i], mul_exact(sixth_step, sr));
        base_adaptation[i] = add_exact(base_adaptation[i], mul_exact(sixth_step, sa));
    } else {
        if (stage == 0) { k1[i] = kr; a1[i] = ka; }
        else if (stage == 1) { k2[i] = kr; a2[i] = ka; }
        else { k3[i] = kr; a3[i] = ka; }
        float nr = add_exact(base_rate[i], mul_exact(step_size, kr));
        float na = add_exact(base_adaptation[i], mul_exact(step_size, ka));
        next_rate[i] = nr; next_adaptation[i] = na;
        // Different allocation from masked: another warp may still read it.
        next_masked[i] = mul_exact(nr, mask[i]);
    }
}
"""
)


class StageFusionCandidate(AdaptiveRateNetwork):
    def _init_stage_fusion(self):
        self._stage_kernel = self._cupy.RawKernel(
            SOURCE, "csr_stage", options=("--fmad=false",)
        )
        self._stage_kernel.compile()
        self._stage_buffers = [self.torch.empty_like(self.rate) for _ in range(10)]

    def _step(self):
        if (
            self.device != "cuda"
            or self.cuda_implementation != "packed"
            or not self.adaptation_gain
        ):
            return super()._step()
        if not hasattr(self, "_stage_kernel"):
            self._init_stage_fusion()
        masked_a, masked_b, work_r, work_z, *derivatives = self._stage_buffers
        ks, azs = derivatives[:3], derivatives[3:]
        self.torch.mul(self.rate, self.output_mask, out=masked_a)
        current, destination = masked_a, masked_b
        r, z = self.rate, self.adaptation
        with self._cupy.cuda.ExternalStream(
            self.torch.cuda.current_stream().cuda_stream
        ):
            for stage in range(4):
                assert current.data_ptr() != destination.data_ptr()
                tensors = (
                    self._rowptr,
                    self._columns,
                    self._weight_values,
                    current,
                    self.drive,
                    self.threshold,
                    z,
                    self.coefficient,
                    self.cap,
                    r,
                    self.tau,
                    self.rate,
                    self.adaptation,
                    self.output_mask,
                    *ks,
                    *azs,
                    work_r,
                    work_z,
                    destination,
                )
                self._stage_kernel(
                    ((self.n + 7) // 8,),
                    (256,),
                    tuple(np.uint64(t.data_ptr()) for t in tensors)
                    + (
                        np.float32(self.adaptation_gain),
                        np.float32(1 / self.adaptation_tau_s),
                        np.float32(self.dt * (0.5 if stage < 2 else 1)),
                        np.float32(self.dt / 6),
                        np.int32(stage),
                        np.int32(self.n),
                    ),
                )
                current, destination = destination, current
                r, z = work_r, work_z
