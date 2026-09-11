"""Opt-in activity adaptation on the complete BANC rate network.

This is a model hypothesis, not inferred individual-cell physiology. No
oscillator, timed reset, gait phase, or periodic stimulus is added. Original
ResearchRateNetwork and source anatomical arrays remain unchanged.
"""

import hashlib

import numpy as np

from .research_rate import ResearchRateNetwork

_CSR_CURRENT = r"""
extern "C" __global__ void csr_current(
    const long long* rowptr, const long long* columns,
    const float* weight, const float* rate, const float* mask,
    float* current, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (long long edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        long long column = columns[edge];
        value += weight[edge] * (rate[column] * mask[column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) current[row] = value;
}
"""

_PACKED_CSR_CURRENT = r"""
extern "C" __global__ void csr_current(
    const int* rowptr, const unsigned int* entries,
    const float* lookup, const float* masked_rate,
    float* current, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        unsigned int item = entries[edge];
        int column = item & 0x3ffff;
        float weight = lookup[item >> 18];
        value += weight * masked_rate[column];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) current[row] = value;
}
"""


class AdaptiveRateNetwork(ResearchRateNetwork):
    def __init__(
        self,
        weights,
        tau,
        a,
        threshold,
        cap,
        *,
        adaptation_gain=0.0,
        adaptation_tau_s=0.15,
        dt=0.0001,
        device="auto",
        capture_steps=10,
        cuda_implementation="reference",
    ):
        import torch

        if type(capture_steps) is not int or not 1 <= capture_steps <= 50:
            raise ValueError("Bounded integer CUDA capture interval required")
        if cuda_implementation not in ("reference", "packed"):
            raise ValueError("Unknown adaptive rate CUDA implementation")
        if not np.isfinite([adaptation_gain, adaptation_tau_s]).all() or not (
            0 <= adaptation_gain <= 100 and 0.01 <= adaptation_tau_s <= 10
        ):
            raise ValueError("Invalid adaptation hypothesis parameters")
        # Build without the parent's CUDA capture: the additional neural state
        # must exist before _step can be captured. No dynamics advance here.
        super().__init__(weights, tau, a, threshold, cap, dt=dt, device="cpu")
        self.interval_steps = capture_steps
        if adaptation_gain and dt > adaptation_tau_s / 50:
            raise ValueError("Integration step must resolve adaptation time constant")
        self.adaptation_gain = float(adaptation_gain)
        self.adaptation_tau_s = float(adaptation_tau_s)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        if self.device not in ("cpu", "cuda") or (
            self.device == "cuda" and not torch.cuda.is_available()
        ):
            raise ValueError("Requested adaptive rate device is unavailable")
        for name in (
            "weights",
            "tau",
            "a",
            "threshold",
            "cap",
            "coefficient",
            "rate",
            "drive",
            "output_mask",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        self.adaptation = torch.zeros_like(self.rate)
        self._csr_kernel = None
        self.cuda_implementation = "torch-cpu-csr"
        if self.device == "cuda":
            import cupy as cp

            self._cupy = cp
            self._rowptr = self.weights.crow_indices()
            self._columns = self.weights.col_indices()
            self._weight_values = self.weights.values()
            source = _CSR_CURRENT
            self.cuda_implementation = "reference"
            if cuda_implementation == "packed":
                # Keep the exact existing float bits; recomputing from integer
                # synapse counts changes some weights by float32 rounding.
                values = self._weight_values.cpu().numpy()
                unique, inverse = np.unique(values.view(np.uint32), return_inverse=True)
                if self.n <= 2**18 and len(unique) <= 2**14 and len(values) < 2**31:
                    columns = self._columns.cpu().numpy().astype(np.uint32)
                    entries = columns | (inverse.astype(np.uint32) << 18)
                    self._rowptr = self._rowptr.to(torch.int32)
                    self._columns = torch.as_tensor(
                        entries.view(np.int32), device=self.device
                    )
                    self._weight_values = torch.as_tensor(
                        unique.view(np.float32), device=self.device
                    )
                    source = _PACKED_CSR_CURRENT
                    self.cuda_implementation = "packed"
            self._csr_kernel = cp.RawKernel(
                source, "csr_current", options=("--fmad=false",)
            )
            self._csr_kernel.compile()
        self.identity = hashlib.sha256(
            (
                self.identity
                + repr(
                    (
                        "activity-adaptation-v1",
                        self.adaptation_gain,
                        self.adaptation_tau_s,
                        "fixed-warp-csr-v1"
                        if self.device == "cuda"
                        else "torch-cpu-csr",
                    )
                )
            ).encode()
        ).hexdigest()
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    self._interval()
                    self._interval()
                torch.cuda.current_stream().wait_stream(warmup)
                self.rate.zero_()
                self.adaptation.zero_()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._interval()
                self.rate.zero_()
                self.adaptation.zero_()

    def _current(self, rate):
        if self._csr_kernel is not None:
            # One warp owns one postsynaptic neuron. Fixed lane order, no
            # floating-point atomics. This also runs on the CUDA capture stream.
            if self.cuda_implementation == "packed":
                # Recompute for every RK stage, including CUDA Graph replays:
                # rate changes within an interval and cuts/restores change the
                # mask between intervals. Preserve the original float32 product
                # while avoiding a repeated mask gather for every synapse.
                inputs = (rate * self.output_mask,)
            else:
                inputs = (rate, self.output_mask)
            result = self.torch.empty_like(rate)
            with self._cupy.cuda.ExternalStream(
                self.torch.cuda.current_stream().cuda_stream
            ):
                self._csr_kernel(
                    ((self.n + 7) // 8,),
                    (256,),
                    tuple(
                        np.uint64(value.data_ptr())
                        for value in (
                            self._rowptr,
                            self._columns,
                            self._weight_values,
                            *inputs,
                            result,
                        )
                    )
                    + (np.int32(self.n),),
                )
            return result
        return self.torch.sparse.mm(
            self.weights, (rate * self.output_mask).unsqueeze(1)
        ).squeeze(1)

    def _derivative(self, rate):
        if self.device == "cpu":
            return super()._derivative(rate)
        target = self.torch.clamp_min(
            self.cap
            * self.torch.tanh(
                self.coefficient * (self.drive + self._current(rate) - self.threshold)
            ),
            0.0,
        )
        return (target - rate) / self.tau

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

    def _step(self):
        if not self.adaptation_gain:
            super()._step()
            return
        dt = self.dt
        r, z = self.rate, self.adaptation
        k1, a1 = self._derivative_pair(r, z)
        k2, a2 = self._derivative_pair(r + 0.5 * dt * k1, z + 0.5 * dt * a1)
        k3, a3 = self._derivative_pair(r + 0.5 * dt * k2, z + 0.5 * dt * a2)
        k4, a4 = self._derivative_pair(r + dt * k3, z + dt * a3)
        self.rate.add_(dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        self.adaptation.add_(dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4))

    def snapshot(self):
        return dict(
            super().snapshot(),
            adaptation=self.adaptation.detach().cpu().numpy().copy(),
            adaptation_gain=self.adaptation_gain,
            adaptation_tau_s=self.adaptation_tau_s,
        )

    def restore(self, state):
        z = np.asarray(state.get("adaptation"))
        if (
            state.get("adaptation_gain") != self.adaptation_gain
            or state.get("adaptation_tau_s") != self.adaptation_tau_s
            or z.shape != (self.n,)
            or z.dtype.kind not in "fiu"
            or not np.isfinite(z).all()
            or np.any(z < 0)
            or (self.adaptation_gain == 0 and np.any(z != 0))
        ):
            raise ValueError("Adaptive neural state mismatch")
        with np.errstate(over="ignore", invalid="ignore"):
            z = z.astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Adaptive neural state exceeds model precision")
        super().restore(state)
        self.adaptation.copy_(self.torch.as_tensor(z, device=self.device))

    def reset(self):
        super().reset()
        self.adaptation.zero_()
