"""Explicit rate-model research kernel; not an interchangeable LIF backend.

Equation: Pugliese_2026, commit 5626b7312ffbbe768793a148cec82f93d7530fce.
Inputs/weights are rate-model quantities, never silently reinterpreted mV.
All declared neurons and matrix entries are retained. Muting cuts outgoing
rate contributions while leaving the source neuron's own dynamics intact.
"""

import hashlib

import numpy as np
from scipy.sparse import csr_matrix


class ResearchRateNetwork:
    def __init__(self, weights, tau, a, threshold, cap, *, dt=0.0001, device="auto"):
        import torch

        self.torch = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Rate research device must be cpu/cuda/auto")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested rate CUDA device is unavailable")
        weights = csr_matrix(weights, dtype=np.float32, copy=True)
        weights.check_format(full_check=True)
        weights.sum_duplicates()
        weights.sort_indices()
        if weights.shape[0] != weights.shape[1] or not weights.shape[0]:
            raise ValueError("Square complete-neuron weight matrix required")
        self.n = weights.shape[0]
        parameters = [np.asarray(p, dtype=np.float32) for p in (tau, a, threshold, cap)]
        if (
            not np.isfinite(weights.data).all()
            or any(p.shape != (self.n,) or not np.isfinite(p).all() for p in parameters)
            or any(np.any(p <= 0) for p in parameters)
            or type(dt) not in (int, float)
            or not np.isfinite(dt)
            or not 0 < dt <= float(parameters[0].min()) / 50
        ):
            raise ValueError("Invalid rate parameters or integration interval")
        self.dt = float(dt)
        checksum = hashlib.sha256(b"flylab.research-rate.rk4.csr.v1")
        for data in (weights.indptr, weights.indices, weights.data, *parameters):
            checksum.update(np.ascontiguousarray(data).tobytes())
        checksum.update(repr(self.dt).encode())
        self.identity = checksum.hexdigest()
        self.weights = torch.sparse_csr_tensor(
            torch.as_tensor(weights.indptr, dtype=torch.int64, device=self.device),
            torch.as_tensor(weights.indices, dtype=torch.int64, device=self.device),
            torch.as_tensor(weights.data, device=self.device),
            size=weights.shape,
            check_invariants=True,
        )
        self.tau, self.a, self.threshold, self.cap = [
            torch.as_tensor(p, device=self.device) for p in parameters
        ]
        self.coefficient = self.a / self.cap
        self.rate = torch.zeros(self.n, device=self.device)
        self.drive = torch.zeros_like(self.rate)
        self.output_mask = torch.ones_like(self.rate)
        self.tick = 0
        self.graph = None
        if self.device == "cuda":
            # Same full-precision matmul setting as the fixed MANC comparison.
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    self._interval()
                    self._interval()
                torch.cuda.current_stream().wait_stream(warmup)
                self.rate.zero_()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._interval()
                self.rate.zero_()

    def _derivative(self, value):
        torch = self.torch
        current = torch.sparse.mm(
            self.weights, (value * self.output_mask).unsqueeze(1)
        ).squeeze(1)
        activation = torch.clamp_min(
            self.cap
            * torch.tanh(self.coefficient * (self.drive + current - self.threshold)),
            0.0,
        )
        return (activation - value) / self.tau

    def _step(self):
        dt = self.dt
        k1 = self._derivative(self.rate)
        k2 = self._derivative(self.rate + (dt / 2) * k1)
        k3 = self._derivative(self.rate + (dt / 2) * k2)
        k4 = self._derivative(self.rate + dt * k3)
        self.rate.add_((dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    def _interval(self):
        for _ in range(10):
            self._step()

    def set_muted(self, indices=()):
        ids = np.asarray(indices)
        if ids.ndim != 1 or (
            ids.size
            and (ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.n))
        ):
            raise ValueError("Invalid outgoing-cut neuron indices")
        self.output_mask.fill_(1)
        if ids.size:
            self.output_mask[
                self.torch.as_tensor(ids, dtype=self.torch.int64, device=self.device)
            ] = 0

    def advance(self, drive, steps=10):
        values = np.asarray(drive)
        if (
            values.shape != (self.n,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError("Finite held rate-model input vector required")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Rate input exceeds float32 range")
        if type(steps) is not int or not 0 < steps <= 10000:
            raise ValueError("Bounded positive integer rate steps required")
        with self.torch.inference_mode():
            self.drive.copy_(
                self.torch.as_tensor(
                    values, dtype=self.torch.float32, device=self.device
                )
            )
            for _ in range(steps // 10):
                if self.graph is None:
                    self._interval()
                else:
                    self.graph.replay()
            for _ in range(steps % 10):
                self._step()
            self.tick += steps
            if not bool(self.torch.isfinite(self.rate).all()) or bool(
                (self.rate < -1e-4).any()
            ):
                raise RuntimeError("Invalid raw complete-network rate state")

    def readout(self, indices=None):
        values = self.rate if indices is None else self.rate[indices]
        result = values.detach().cpu().numpy().copy()
        if not np.isfinite(result).all() or np.any(result < -1e-4):
            raise RuntimeError("Invalid raw rate state")
        return result

    def reset(self):
        self.rate.zero_()
        self.drive.zero_()
        self.output_mask.fill_(1)
        self.tick = 0

    def snapshot(self):
        return {
            "schema": "flylab.research-rate.v1",
            "model_hash": self.identity,
            "tick": self.tick,
            "dt": self.dt,
            "rate": self.readout(),
            "drive": self.drive.cpu().numpy().copy(),
            "output_mask": self.output_mask.cpu().numpy().copy(),
            "device": self.device,
        }

    def restore(self, state):
        if (
            state.get("schema") != "flylab.research-rate.v1"
            or state.get("model_hash") != self.identity
            or state.get("device") != self.device
            or state.get("dt") != self.dt
            or type(state.get("tick")) is not int
            or state["tick"] < 0
        ):
            raise ValueError("Incompatible research rate checkpoint")
        values = [np.asarray(state[k]) for k in ("rate", "drive", "output_mask")]
        if any(x.dtype.kind not in "fiu" for x in values):
            raise ValueError("Numeric research rate checkpoint arrays required")
        with np.errstate(over="ignore", invalid="ignore"):
            values = [x.astype(np.float32, copy=False) for x in values]
        if (
            any(x.shape != (self.n,) or not np.isfinite(x).all() for x in values)
            or np.any(values[0] < -1e-4)
            or not np.isin(values[2], (0, 1)).all()
        ):
            raise ValueError("Invalid research rate checkpoint arrays")
        for target, value in zip((self.rate, self.drive, self.output_mask), values):
            target.copy_(
                self.torch.as_tensor(
                    value, dtype=self.torch.float32, device=self.device
                )
            )
        self.tick = state["tick"]
