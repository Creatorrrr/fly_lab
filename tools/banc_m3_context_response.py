"""Independent, differentiable conditional BANC motor responses.

The recorded context owns its source activity and residual current.  Candidate
weights only enter the explicit source-to-MN term; this is an endpoint model,
not a native recurrent-network or physical-body trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.banc_m3_causal_recruitment import endpoint_response, linear_response


def id_positions(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return positions by ID, including for unsorted calibration edge IDs."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.ndim != 1 or expected.ndim != 1 or actual.dtype.kind not in "iu" \
            or expected.dtype.kind not in "iu":
        raise ValueError("Integer one-dimensional ID arrays required")
    lookup = {int(value): i for i, value in enumerate(actual)}
    if len(lookup) != len(actual) or len(set(map(int, expected))) != len(expected):
        raise ValueError("Duplicate IDs")
    try:
        return np.asarray([lookup[int(value)] for value in expected], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Missing ID {exc.args[0]}") from exc


def validate_time_grid(time_s: np.ndarray, dt: float = .005) -> None:
    time_s = np.asarray(time_s)
    if time_s.ndim != 1 or len(time_s) < 2 or time_s.dtype != np.float64 \
            or not np.isfinite(time_s).all() or time_s[0] != 0 \
            or not np.allclose(np.diff(time_s), dt, rtol=0, atol=2e-12):
        raise ValueError("Contiguous float64 time grid from zero required")


def audit_own_background(current: np.ndarray, source: np.ndarray,
                         weights: np.ndarray, stored: np.ndarray) -> dict:
    """Report the E112b gamma bound, without changing the stored residual."""
    if any(not isinstance(value, np.ndarray) or value.dtype != np.float64
           for value in (current, source, weights, stored)) or \
            current.ndim != 2 or source.ndim != 2 or weights.ndim != 2 or \
            stored.shape != current.shape or \
            current.shape != (source.shape[0], weights.shape[0]) or \
            source.shape[1] != weights.shape[1]:
        raise ValueError("Incompatible float64 background operands")
    k = source.shape[1] + 1
    eps = np.finfo(np.float64).eps
    gamma = k * eps / (1 - k * eps)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        derived = current - source @ weights.T
        terms = np.abs(current) + np.abs(source) @ np.abs(weights).T
        bound = np.nextafter(2 * gamma / (1 - gamma) * terms, np.inf)
        difference = np.abs(derived - stored)
    finite = all(np.isfinite(x).all() for x in
                 (current, source, weights, stored, derived, bound, difference))
    nonzero = [np.abs(x[x != 0]).min() if np.any(x != 0) else 1.
               for x in (current, source, weights, stored)]
    normal = bool(min(nonzero) >= np.finfo(np.float64).tiny and
                  nonzero[1] * nonzero[2] >= np.finfo(np.float64).tiny)
    return {"derived": derived, "difference": difference, "bound": bound,
            "passed": bool(finite and normal and np.all(difference <= bound)),
            "finite": finite, "normal_range": normal,
            "max_difference": float(np.max(difference)),
            "max_bound": float(np.max(bound)), "gamma": float(gamma),
            "bound_exceedances": int(np.count_nonzero(difference > bound))}


class ContextResponse:
    """Batch conditional responses with an independent zero state per pair."""

    def __init__(self, *, time_s: np.ndarray, tau: np.ndarray, a: np.ndarray,
                 threshold: np.ndarray, cap: np.ndarray, pool: np.ndarray,
                 moment: np.ndarray, rest: np.ndarray, ltm: np.ndarray,
                 device: str | torch.device = "cpu"):
        validate_time_grid(time_s)
        self.dt = float(time_s[1] - time_s[0])
        self.n_time = len(time_s)
        requested = torch.device(device)
        values = {name: torch.as_tensor(value, dtype=torch.float64,
                                        device=requested)
                  for name, value in (("tau", tau), ("a", a),
                                      ("threshold", threshold), ("cap", cap),
                                      ("pool", pool), ("moment", moment),
                                      ("rest", rest))}
        self.__dict__.update(values)
        self.device = self.tau.device  # Canonicalize bare cuda / cpu:0 aliases.
        n_motor = self.tau.numel()
        n_muscle, pool_motor = self.pool.shape
        n_joint, moment_muscle = self.moment.shape
        self.ltm = torch.as_tensor(ltm, dtype=torch.long, device=self.device)
        if self.tau.ndim != 1 or any(x.shape != (n_motor,) for x in
            (self.a, self.threshold, self.cap)) or pool_motor != n_motor or \
                moment_muscle != n_muscle or self.rest.shape != (n_joint,) or \
                self.ltm.ndim != 1 or torch.any(self.ltm < 0) or \
                torch.any(self.ltm >= n_muscle):
            raise ValueError("Incompatible motor, muscle, joint or LTM axes")
        if n_motor == 0 or n_muscle == 0 or n_joint == 0 or \
                any(not bool(torch.isfinite(x).all()) for x in values.values()) \
                or not bool(torch.all(self.tau > 0)) or \
                not bool(torch.all(self.cap > 0)):
            raise ValueError("Finite coefficients, positive tau and cap required")

    def evaluate(self, source_rate: torch.Tensor, background: torch.Tensor,
                 candidate_weights: torch.Tensor, context_indices: torch.Tensor,
                 weight_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return [pair,time,...] tensors; gradients flow to candidate W."""
        inputs = (source_rate, background, candidate_weights)
        if any(not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or
               x.device != self.device for x in inputs):
            raise ValueError("Float64 tensors on the coefficient device required")
        if source_rate.ndim != 3 or background.ndim != 3 or \
                candidate_weights.ndim != 3:
            raise ValueError("Context and weight tensors must have three axes")
        n_context, n_time, n_source = source_rate.shape
        n_weight, n_motor, weight_source = candidate_weights.shape
        if n_context < 1 or n_weight < 1 or n_time != self.n_time or \
                background.shape != (n_context, n_time, n_motor) or \
                n_motor != self.tau.numel() or n_source != weight_source or \
                n_source == 0 or background.requires_grad:
            raise ValueError("Incompatible contexts, candidate weights or fixed background")
        if any(not bool(torch.isfinite(x).all()) for x in inputs):
            raise ValueError("Nonfinite response inputs")
        indices = (context_indices, weight_indices)
        if any(not isinstance(x, torch.Tensor) or x.dtype != torch.long or
               x.device != self.device or x.ndim != 1 for x in indices) or \
                context_indices.numel() == 0 or \
                context_indices.shape != weight_indices.shape or \
                bool(torch.any(context_indices < 0)) or \
                bool(torch.any(context_indices >= n_context)) or \
                bool(torch.any(weight_indices < 0)) or \
                bool(torch.any(weight_indices >= n_weight)):
            raise ValueError("Valid explicit pair indices required")
        count = context_indices.numel()
        source = source_rate[context_indices]
        fixed_background = background[context_indices]
        weights = candidate_weights[weight_indices]
        current = torch.bmm(source, weights.transpose(1, 2)) + fixed_background
        z = self.cap * torch.tanh(self.a * (current - self.threshold) / self.cap)
        forcing = torch.relu(z)
        # The reused helpers convolve dim 0. Flatten PAIR and feature only;
        # no temporal concatenation and no transfer between pairs' zero states.
        flat = forcing.permute(1, 0, 2).reshape(n_time, count * n_motor)
        flat_tau = self.tau.repeat(count)
        rate = linear_response(flat, self.dt, flat_tau).reshape(
            n_time, count, n_motor).permute(1, 0, 2)
        muscle_forcing = rate @ self.pool.T
        n_muscle = self.pool.shape[0]
        muscle_flat = muscle_forcing.permute(1, 0, 2).reshape(
            n_time, count * n_muscle)
        activation = endpoint_response(muscle_flat, self.dt, .01).reshape(
            n_time, count, n_muscle).permute(1, 0, 2)
        command = self.rest + .02 * (activation @ self.moment.T)
        tendon = .2 * torch.tanh(activation[:, :, self.ltm] / 10)
        return {"current": current, "forcing": forcing,
                "motor_rate": rate, "muscle_activation": activation,
                "command": command, "tendon": tendon}
