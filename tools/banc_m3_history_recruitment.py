"""Pure numerical helpers for the E170 Stage A history recruitment hypothesis.

The caller must preserve raw candidate coefficients and solver diagnostics before
calling these helpers. ``evaluate_history`` deliberately does not clip finite
negative diagnostic coefficients; ``parameter_gates`` rejects them. No original
trajectory, fitting, adapter, or physical simulation is loaded here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_MN = 391
N_MUSCLE = 96
N_FILTER = 3
N_Q = 42
DT = 0.005
TAU = 0.07
ATOL = 1e-12


def _float64(value: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float64
        or array.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape, strict=True)
        )
    ):
        raise ValueError(f"{name} must be float64 with shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _integer(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.shape != shape:
        raise ValueError(f"{name} must be int64 with shape {shape}")
    return array


def _anatomy(membership: Any, fixed_muscles: Any) -> tuple[np.ndarray, np.ndarray]:
    members = _integer(membership, (N_MN,), "membership")
    fixed = np.asarray(fixed_muscles)
    if fixed.dtype != np.int64 or fixed.ndim != 1:
        raise ValueError("fixed_muscles must be a 1-D int64 array")
    if (
        np.any((members < 0) | (members >= N_MUSCLE))
        or np.any((fixed < 0) | (fixed >= N_MUSCLE))
        or np.unique(fixed).size != fixed.size
        or np.any(np.bincount(members, minlength=N_MUSCLE) == 0)
    ):
        raise ValueError("invalid anatomical ownership or fixed muscle indices")
    return members, fixed


def history_state(rate: Any, kappa: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, F) for float64 rate(T,391), kappa(391), T>=1.

    The fixed update uses dt=.005 and tau=.07. The available current rate
    updates state at each tick after tick zero. Inputs are never mutated.
    """
    rate_array = _float64(rate, (None, N_MN), "rate")
    kappa_array = _float64(kappa, (N_MN,), "kappa")
    if rate_array.shape[0] < 1 or np.any(rate_array < 0) or np.any(kappa_array < 0):
        raise ValueError("rate needs T>=1; rate and kappa must be nonnegative")
    with np.errstate(over="ignore"):
        denominator = rate_array + kappa_array
    s = np.divide(
        rate_array,
        denominator,
        out=np.zeros_like(rate_array),
        where=denominator != 0,
    )
    overflow = np.isinf(denominator)
    if np.any(overflow):
        large_rate = rate_array[overflow]
        large_kappa = np.broadcast_to(kappa_array, rate_array.shape)[overflow]
        rate_dominant = large_rate >= large_kappa
        safe = np.empty_like(large_rate)
        safe[rate_dominant] = 1.0 / (
            1.0 + large_kappa[rate_dominant] / large_rate[rate_dominant]
        )
        ratio = large_rate[~rate_dominant] / large_kappa[~rate_dominant]
        safe[~rate_dominant] = ratio / (1.0 + ratio)
        s[overflow] = safe
    state = np.empty_like(rate_array)
    state[0] = 1.0
    for tick in range(1, rate_array.shape[0]):
        target = 1.0 / (1.0 + s[tick])
        state[tick] = target + (state[tick - 1] - target) * np.exp(
            -(1.0 + s[tick]) * DT / TAU
        )
    return state, 2.0 * (1.0 - state)


def evaluate_history(
    bank: Any,
    original_activation: Any,
    alpha: Any,
    beta: Any,
    kappa: Any,
    membership: Any,
    fixed_muscles: Any,
    moment: Any,
    rest: Any,
    history: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate activation(T,96) and command(T,42) in adapter tick order.

    ``history`` is F(T,391), as returned by ``history_state``. All real-valued
    inputs must be finite float64; membership and fixed indices must be int64.
    T>=1. Coefficients may be finite negative diagnostic values and are never
    clipped here; call ``parameter_gates`` to determine physical admissibility.
    """
    bank_array = _float64(bank, (None, N_MN, N_FILTER), "bank")
    ticks = bank_array.shape[0]
    if ticks < 1 or np.any(bank_array < 0):
        raise ValueError("bank needs T>=1 and nonnegative values")
    original = _float64(original_activation, (ticks, N_MUSCLE), "original_activation")
    a = _float64(alpha, (N_MUSCLE, N_MN, N_FILTER, 2), "alpha")
    b = _float64(beta, (N_MUSCLE, N_MN, N_FILTER, 2), "beta")
    k = _float64(kappa, (N_MN,), "kappa")
    f = _float64(history, (ticks, N_MN), "history")
    moment_array = _float64(moment, (N_Q, N_MUSCLE), "moment")
    rest_array = _float64(rest, (N_Q,), "rest")
    members, fixed = _anatomy(membership, fixed_muscles)
    if np.any(k < 0) or np.any((f < 0) | (f > 1)) or np.any(original < 0):
        raise ValueError(
            "kappa and original activation must be nonnegative; F in [0,1]"
        )
    activation = np.empty((ticks, N_MUSCLE), dtype=np.float64)
    command = np.empty((ticks, N_Q), dtype=np.float64)
    variable = np.setdiff1d(np.arange(N_MUSCLE, dtype=np.int64), fixed)
    for tick in range(ticks):
        activation[tick, fixed] = original[tick, fixed]
        hinge = np.maximum(bank_array[tick] - k[:, None], 0.0)
        for muscle in variable:
            owned = np.flatnonzero(members == muscle)
            low = a[muscle, owned, :, 0] + f[tick, owned, None] * b[muscle, owned, :, 0]
            high = (
                a[muscle, owned, :, 1] + f[tick, owned, None] * b[muscle, owned, :, 1]
            )
            activation[tick, muscle] = np.sum(low * bank_array[tick, owned]) + np.sum(
                (high - low) * hinge[owned]
            )
        command[tick] = rest_array + 0.02 * (moment_array @ activation[tick])
    return activation, command


def parameter_gates(
    alpha: Any,
    beta: Any,
    baseline_slopes: Any,
    membership: Any,
    fixed_muscles: Any,
) -> dict[str, Any]:
    """Gate actual float64 coefficients over every F in [0,1].

    Returns per-condition booleans, a combined ``pass`` boolean, and raw
    coefficient, MN and muscle extrema. Only floor/budget gates have 1e-12
    tolerance; box, support, and fixed equality are exact.
    """
    shape = (N_MUSCLE, N_MN, N_FILTER, 2)
    a = _float64(alpha, shape, "alpha")
    b = _float64(beta, shape, "beta")
    baseline = _float64(baseline_slopes, shape, "baseline_slopes")
    members, fixed = _anatomy(membership, fixed_muscles)
    support = np.zeros((N_MUSCLE, N_MN), dtype=bool)
    support[members, np.arange(N_MN)] = True
    endpoint = a + b
    owned_a = a[members, np.arange(N_MN)]
    owned_end = endpoint[members, np.arange(N_MN)]
    motor_minimum = np.min(owned_a, axis=-1).sum(axis=1)
    motor_maximum = np.max(owned_end, axis=-1).sum(axis=1)
    muscle_minimum = np.bincount(members, weights=motor_minimum, minlength=N_MUSCLE)
    muscle_maximum = np.bincount(members, weights=motor_maximum, minlength=N_MUSCLE)
    counts = np.bincount(members, minlength=N_MUSCLE)
    gates = {
        "box": bool(
            np.all((a >= 0) & (a <= 2))
            and np.all((b >= 0) & (b <= 2))
            and np.all(endpoint <= 2)
        ),
        "support": bool(np.all(a[~support] == 0) and np.all(b[~support] == 0)),
        "fixed": bool(
            np.array_equal(a[fixed], baseline[fixed]) and np.all(b[fixed] == 0)
        ),
        "mn_floor": bool(np.all(motor_minimum >= 1e-6 / counts[members] - ATOL)),
        "muscle_budget": bool(
            np.all(muscle_minimum >= 0.5 - ATOL)
            and np.all(muscle_maximum <= 2.0 + ATOL)
        ),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "alpha_min": float(np.min(a)),
        "alpha_max": float(np.max(a)),
        "beta_min": float(np.min(b)),
        "beta_max": float(np.max(b)),
        "endpoint_min": float(np.min(endpoint)),
        "endpoint_max": float(np.max(endpoint)),
        "motor_minimum": motor_minimum,
        "motor_maximum": motor_maximum,
        "muscle_minimum": muscle_minimum,
        "muscle_maximum": muscle_maximum,
    }
