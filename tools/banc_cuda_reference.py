"""Frozen pre-mask-optimization CUDA baseline, for diagnostics only.

The packed kernel and its call retain the 2026-09-12 audit implementation.
Keeping it outside production lets benchmarks compare subsequent changes
without silently changing the baseline along with the candidate.
"""

import numpy as np

from flylab.c.adaptive_rate import AdaptiveRateNetwork

LEGACY_PACKED_CSR_CURRENT = r"""
extern "C" __global__ void csr_current(
    const int* rowptr, const unsigned int* entries,
    const float* lookup, const float* rate, const float* mask,
    float* current, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        unsigned int item = entries[edge];
        int column = item & 0x3ffff;
        float weight = lookup[item >> 18];
        value += weight * (rate[column] * mask[column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) current[row] = value;
}
"""


class UnfusedPackedNetwork(AdaptiveRateNetwork):
    def _derivative_pair(self, rate, adaptation):
        current = self._current(rate)
        target = self.torch.clamp_min(
            self.cap
            * self.torch.tanh(
                self.coefficient * (self.drive + current - self.threshold - adaptation)
            ),
            0.0,
        )
        return (target - rate) / self.tau, (
            self.adaptation_gain * rate - adaptation
        ) / self.adaptation_tau_s


class LegacyPackedNetwork(UnfusedPackedNetwork):
    def _current(self, rate):
        if self.device != "cuda" or self.cuda_implementation != "packed":
            raise ValueError("The frozen audit baseline requires packed CUDA")
        if not hasattr(self, "_legacy_kernel"):
            self._legacy_kernel = self._cupy.RawKernel(
                LEGACY_PACKED_CSR_CURRENT, "csr_current", options=("--fmad=false",)
            )
            self._legacy_kernel.compile()
        result = self.torch.empty_like(rate)
        with self._cupy.cuda.ExternalStream(
            self.torch.cuda.current_stream().cuda_stream
        ):
            self._legacy_kernel(
                ((self.n + 7) // 8,),
                (256,),
                tuple(
                    np.uint64(v.data_ptr())
                    for v in (
                        self._rowptr,
                        self._columns,
                        self._weight_values,
                        rate,
                        self.output_mask,
                        result,
                    )
                )
                + (np.int32(self.n),),
            )
        return result
