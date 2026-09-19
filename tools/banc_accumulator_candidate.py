"""Candidate-only RK derivative accumulation in the original float order."""

import numpy as np

from flylab.c.adaptive_fused import SOURCE as ORIGINAL_SOURCE
from flylab.c.adaptive_rate import AdaptiveRateNetwork

SOURCE = (
    ORIGINAL_SOURCE.split('extern "C" __global__ void stage(', 1)[0]
    + r"""
extern "C" __global__ void stage(
    const float* activation, const float* cap, const float* rate,
    const float* adaptation, const float* tau, const float* base_rate,
    const float* base_adaptation, const float* mask,
    float* sum_rate, float* sum_adaptation,
    float* next_rate, float* next_adaptation, float* next_masked,
    float gain, float inverse_adaptation_tau, float step_size, int stage_id, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = clamp_exact(mul_exact(cap[i], activation[i]));
    float kr = div_exact(sub_exact(target, rate[i]), tau[i]);
    float ka = mul_exact(sub_exact(mul_exact(gain, rate[i]), adaptation[i]), inverse_adaptation_tau);
    if (stage_id == 0) {
        sum_rate[i] = kr; sum_adaptation[i] = ka;
    } else {
        sum_rate[i] = add_exact(sum_rate[i], mul_exact(2.0f, kr));
        sum_adaptation[i] = add_exact(sum_adaptation[i], mul_exact(2.0f, ka));
    }
    float nr = add_exact(base_rate[i], mul_exact(step_size, kr));
    float na = add_exact(base_adaptation[i], mul_exact(step_size, ka));
    next_rate[i] = nr; next_adaptation[i] = na;
    next_masked[i] = mul_exact(nr, mask[i]);
}

extern "C" __global__ void finish(
    const float* activation, const float* cap, const float* rate,
    const float* adaptation, const float* tau,
    const float* sum_rate, const float* sum_adaptation,
    float* base_rate, float* base_adaptation,
    float gain, float inverse_adaptation_tau, float sixth_step, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = clamp_exact(mul_exact(cap[i], activation[i]));
    float kr = div_exact(sub_exact(target, rate[i]), tau[i]);
    float ka = mul_exact(sub_exact(mul_exact(gain, rate[i]), adaptation[i]), inverse_adaptation_tau);
    float sr = add_exact(sum_rate[i], kr);
    float sa = add_exact(sum_adaptation[i], ka);
    base_rate[i] = add_exact(base_rate[i], mul_exact(sixth_step, sr));
    base_adaptation[i] = add_exact(base_adaptation[i], mul_exact(sixth_step, sa));
}
"""
)


class AccumulatorCandidate(AdaptiveRateNetwork):
    def _init_fusion(self):
        source = SOURCE
        if self.fuse_tanh:
            source = source.replace(
                "mul_exact(cap[i], activation[i])",
                "mul_exact(cap[i], matching_tanh(activation[i]))",
            )
        self._fused_kernels = {
            name: self._cupy.RawKernel(source, name, options=("--fmad=false",))
            for name in ("csr_input", "stage", "finish")
        }
        for kernel in self._fused_kernels.values():
            kernel.compile()
        self._fused_buffers = [self.torch.empty_like(self.rate) for _ in range(7)]

    def _launch_accumulator(self, name, tensors, scalars, stage=None):
        args = tuple(np.uint64(t.data_ptr()) for t in tensors)
        args += tuple(np.float32(v) for v in scalars)
        if stage is not None:
            args += (np.int32(stage),)
        self._fused_kernels[name](
            ((self.n + 255) // 256,),
            (256,),
            args + (np.int32(self.n),),
        )

    def _step_fused(self):
        if not hasattr(self, "_fused_kernels"):
            self._init_fusion()
        masked, work_r, work_z, inputs, activation, sum_r, sum_z = self._fused_buffers
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
                    self._launch_accumulator(
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
                            sum_r,
                            sum_z,
                            work_r,
                            work_z,
                            masked,
                        ),
                        (
                            self.adaptation_gain,
                            1 / self.adaptation_tau_s,
                            self.dt * (0.5 if stage < 2 else 1),
                        ),
                        stage,
                    )
                    r, z = work_r, work_z
                else:
                    self._launch_accumulator(
                        "finish",
                        (
                            activation,
                            self.cap,
                            r,
                            z,
                            self.tau,
                            sum_r,
                            sum_z,
                            self.rate,
                            self.adaptation,
                        ),
                        (self.adaptation_gain, 1 / self.adaptation_tau_s, self.dt / 6),
                    )
