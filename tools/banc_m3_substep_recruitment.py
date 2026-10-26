"""Pure causal substep reconstruction for the frozen E149 motor transfer.

The helper observes an already-decoded monotone recruitment adapter.  It does
not advance or modify that adapter, a neural model, or a body.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.banc_m3_monotone_recruitment import ADAPTER_SCHEMA, FILTER_TAU_S
from tools.banc_m3_monotone_recruitment import _identity as _monotone_identity

ATOL = 1e-12
FIXED_MUSCLE_TAU_S = 0.01
PHYSICS_DT_S = 0.0001


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_real_array(
    value: Any, shape: tuple[int, ...], name: str, *, nonnegative: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    _require(
        array.shape == shape
        and np.issubdtype(array.dtype, np.number)
        and np.isrealobj(array)
        and np.isfinite(array).all()
        and (not nonnegative or np.all(array >= 0)),
        f"Finite real {name} with shape {shape} required",
    )
    return np.array(array, dtype=np.float64, copy=True)


def _exact_array(left: Any, right: np.ndarray) -> bool:
    candidate = np.asarray(left)
    return (
        candidate.shape == right.shape
        and candidate.dtype == right.dtype
        and np.array_equal(candidate, right)
    )


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right), initial=0.0))


@dataclass(frozen=True)
class SubstepRecruitmentResult:
    """One control interval reconstructed at native physics boundaries."""

    joint_targets: np.ndarray
    relative_time_s: np.ndarray
    endpoint_error: dict[str, float]


class CausalSubstepRecruitment:
    """Validate and evaluate the fixed E149 transfer without dynamic state."""

    def __init__(self, adapter: Any):
        _require(
            getattr(adapter, "schema", None) == ADAPTER_SCHEMA,
            "Frozen monotone adapter schema required",
        )
        identity = getattr(adapter, "identity", None)
        _require(
            isinstance(identity, str)
            and len(identity) == 64
            and all(character in "0123456789abcdef" for character in identity),
            "Frozen monotone adapter identity required",
        )
        _require(
            hasattr(adapter, "original")
            and hasattr(adapter.original, "activation")
            and isinstance(getattr(adapter.original, "identity", None), str)
            and bool(adapter.original.identity)
            and hasattr(adapter, "body")
            and hasattr(adapter.body, "neutral"),
            "Complete original adapter and body-neutral view required",
        )

        calibration = getattr(adapter, "calibration", None)
        _require(
            isinstance(calibration, dict)
            and {"moment", "rest", "pool", "motor_ids", "ltm"} <= set(calibration),
            "Complete original calibration required",
        )
        moment = _finite_real_array(calibration["moment"], (42, 96), "moment")
        rest = _finite_real_array(calibration["rest"], (42,), "rest")
        pool = _finite_real_array(
            calibration["pool"], (96, 391), "anatomical pool", nonnegative=True
        )
        _require(
            np.allclose(pool.sum(axis=1), 1.0, atol=1e-15, rtol=0),
            "Normalized anatomical pool required",
        )
        motor_ids = np.asarray(calibration["motor_ids"])
        _require(
            motor_ids.shape == (391,)
            and motor_ids.dtype.kind in "iu"
            and len(np.unique(motor_ids)) == 391
            and np.array_equal(motor_ids, np.sort(motor_ids)),
            "Sorted unique 391-MN IDs required",
        )
        ltm = np.asarray(calibration["ltm"])
        _require(
            ltm.shape == (6,)
            and ltm.dtype.kind in "iu"
            and len(np.unique(ltm)) == 6
            and np.all((ltm >= 0) & (ltm < 96)),
            "Six unique original LTM rows required",
        )

        tau = np.asarray(getattr(adapter, "tau", None))
        _require(
            tau.dtype == np.float64 and np.array_equal(tau, FILTER_TAU_S),
            "Exact fixed filter tau [0.005,0.01,0.04] required",
        )
        kappa = _finite_real_array(
            getattr(adapter, "kappa", None), (391,), "391-MN kappa", nonnegative=True
        )
        slopes = _finite_real_array(
            getattr(adapter, "slopes", None),
            (96, 391, 3, 2),
            "monotone slopes",
            nonnegative=True,
        )
        membership = np.asarray(getattr(adapter, "membership", None))
        _require(
            membership.shape == (391,)
            and membership.dtype == np.int64
            and np.all((membership >= 0) & (membership < 96)),
            "Exact int64 391-MN membership required",
        )
        _require(
            np.array_equal(np.sum(pool > 0, axis=0), np.ones(391, dtype=int))
            and np.array_equal(np.argmax(pool, axis=0), membership),
            "Membership differs from original anatomical pool",
        )
        support = np.zeros((96, 391), dtype=bool)
        support[membership, np.arange(391)] = True
        _require(
            np.all(slopes[~support] == 0),
            "Monotone slopes changed anatomical support",
        )
        supported_slopes = slopes[membership, np.arange(391)]
        minimum = np.min(supported_slopes, axis=-1)
        maximum = np.max(supported_slopes, axis=-1)
        lower = np.bincount(membership, weights=minimum.sum(axis=1), minlength=96)
        upper = np.bincount(membership, weights=maximum.sum(axis=1), minlength=96)
        counts = np.bincount(membership, minlength=96)
        _require(np.all(counts > 0), "Every original muscle requires at least one MN")
        floor = 1e-6 / counts[membership]
        _require(
            np.all(lower >= 0.5 - ATOL)
            and np.all(upper <= 2 + ATOL)
            and np.all(minimum.sum(axis=1) >= floor - ATOL),
            "Frozen monotone slope budgets changed",
        )

        groups = getattr(adapter, "groups", None)
        _require(
            isinstance(groups, list) and len(groups) == 96, "96 muscle groups required"
        )
        seen: list[int] = []
        for muscle, group in enumerate(groups):
            expected = np.flatnonzero(membership == muscle).tolist()
            _require(
                isinstance(group, dict)
                and bool(group.get("leg"))
                and bool(group.get("muscle"))
                and list(group.get("columns", [])) == expected,
                "Named muscle membership provenance changed",
            )
            seen.extend(expected)
        _require(sorted(seen) == list(range(391)), "Muscle groups must partition MNs")

        fixed = np.asarray(getattr(adapter, "fixed_muscles", None))
        expected_fixed = np.asarray(
            sorted(
                set(map(int, ltm)).union(
                    muscle
                    for muscle in range(96)
                    if np.count_nonzero(membership == muscle) == 1
                )
            ),
            dtype=np.int64,
        )
        variable = np.asarray(getattr(adapter, "variable_muscles", None))
        expected_variable = np.setdiff1d(np.arange(96, dtype=np.int64), expected_fixed)
        _require(
            fixed.dtype == np.int64
            and np.array_equal(fixed, expected_fixed)
            and variable.dtype == np.int64
            and np.array_equal(variable, expected_variable),
            "Fixed/singleton/LTM muscle mapping changed",
        )
        _require(
            _exact_array(getattr(adapter, "lower_budget", None), lower)
            and _exact_array(getattr(adapter, "upper_budget", None), upper)
            and _exact_array(
                getattr(adapter, "motor_minimum_slope", None), minimum.sum(axis=1)
            ),
            "Frozen monotone budget provenance changed",
        )
        canonical = np.zeros((96, 391, 3, 2), dtype=np.float64)
        canonical[:, :, 1, :] = pool[:, :, None]
        _require(
            all(np.array_equal(slopes[muscle], canonical[muscle]) for muscle in fixed),
            "Fixed muscle transfer differs from original pool",
        )

        roster = getattr(adapter, "motor_roster", None)
        _require(
            isinstance(roster, tuple)
            and len(roster) == 391
            and len(set(roster)) == 391
            and all(isinstance(value, str) and value for value in roster),
            "Unique ordered original MN roster required",
        )
        neutral = _finite_real_array(adapter.body.neutral, (42,), "body neutral")
        expected_identity = _monotone_identity(
            adapter.original,
            {
                "moment": moment,
                "rest": rest,
                "pool": pool,
                "motor_ids": np.array(motor_ids, copy=True),
                "ltm": np.array(ltm, copy=True),
            },
            groups,
            tau,
            kappa,
            slopes,
            membership,
            list(roster),
        )
        _require(identity == expected_identity, "Monotone adapter identity mismatch")

        self._adapter = adapter
        self._identity = identity
        self._original_identity = adapter.original.identity
        self._moment = moment
        self._rest = rest
        self._pool = pool
        self._motor_ids = np.array(motor_ids, copy=True)
        self._ltm = np.array(ltm, dtype=np.int64, copy=True)
        self._tau = np.array(tau, copy=True)
        self._kappa = kappa
        self._slopes = slopes
        self._membership = np.array(membership, copy=True)
        self._fixed = np.array(fixed, copy=True)
        self._variable = np.array(variable, copy=True)
        self._lower_budget = np.array(lower, copy=True)
        self._upper_budget = np.array(upper, copy=True)
        self._motor_minimum_slope = np.array(minimum.sum(axis=1), copy=True)
        self._groups = copy.deepcopy(groups)
        self._roster = tuple(roster)
        self._neutral = neutral

    def _validate_unchanged_configuration(self) -> None:
        adapter = self._adapter
        calibration = getattr(adapter, "calibration", {})
        body = getattr(adapter, "body", None)
        original = getattr(adapter, "original", None)
        _require(
            getattr(adapter, "schema", None) == ADAPTER_SCHEMA
            and getattr(adapter, "identity", None) == self._identity
            and getattr(original, "identity", None) == self._original_identity
            and isinstance(calibration, dict)
            and _exact_array(calibration.get("moment"), self._moment)
            and _exact_array(calibration.get("rest"), self._rest)
            and _exact_array(calibration.get("pool"), self._pool)
            and _exact_array(calibration.get("motor_ids"), self._motor_ids)
            and _exact_array(calibration.get("ltm"), self._ltm)
            and _exact_array(getattr(adapter, "tau", None), self._tau)
            and _exact_array(getattr(adapter, "kappa", None), self._kappa)
            and _exact_array(getattr(adapter, "slopes", None), self._slopes)
            and _exact_array(getattr(adapter, "membership", None), self._membership)
            and _exact_array(getattr(adapter, "fixed_muscles", None), self._fixed)
            and _exact_array(getattr(adapter, "variable_muscles", None), self._variable)
            and _exact_array(getattr(adapter, "lower_budget", None), self._lower_budget)
            and _exact_array(getattr(adapter, "upper_budget", None), self._upper_budget)
            and _exact_array(
                getattr(adapter, "motor_minimum_slope", None),
                self._motor_minimum_slope,
            )
            and getattr(adapter, "groups", None) == self._groups
            and tuple(getattr(adapter, "motor_roster", ())) == self._roster
            and _exact_array(getattr(body, "neutral", None), self._neutral),
            "Adapter identity or fixed transfer changed after helper construction",
        )

    def compute(
        self,
        bank_before: Any,
        fixed_activation_before: Any,
        rates: Any,
        endpoint_target: Any,
        *,
        motor_cut: bool,
        dt: float,
        physics_dt: float,
    ) -> SubstepRecruitmentResult:
        """Return native-boundary targets for one already-decoded control."""

        self._validate_unchanged_configuration()
        _require(type(motor_cut) is bool, "Boolean motor_cut required")
        _require(
            not isinstance(dt, (bool, np.bool_))
            and isinstance(dt, (int, float, np.integer, np.floating))
            and math.isfinite(float(dt))
            and 0 < float(dt) <= 0.05,
            "Finite control dt in (0, 0.05] required",
        )
        _require(
            not isinstance(physics_dt, (bool, np.bool_))
            and isinstance(physics_dt, (int, float, np.integer, np.floating))
            and math.isfinite(float(physics_dt))
            and float(physics_dt) == PHYSICS_DT_S,
            "Exact 0.0001-s physics dt required",
        )
        ratio = float(dt) / float(physics_dt)
        steps = round(ratio)
        represented_interval = steps * float(physics_dt)
        interval_tolerance = (
            8
            * np.finfo(np.float64).eps
            * max(abs(float(dt)), abs(represented_interval), float(physics_dt))
        )
        _require(
            steps >= 1 and abs(float(dt) - represented_interval) <= interval_tolerance,
            "dt/physics_dt must be an integer within machine precision",
        )

        bank0 = _finite_real_array(
            bank_before, (391, 3), "pre-decode filter bank", nonnegative=True
        )
        fixed0 = _finite_real_array(
            fixed_activation_before,
            (96,),
            "pre-decode fixed activation",
            nonnegative=True,
        )
        rate = _finite_real_array(rates, (391,), "391-MN rates", nonnegative=True)
        endpoint = _finite_real_array(endpoint_target, (42,), "endpoint target")
        current_bank = _finite_real_array(
            self._adapter.bank, (391, 3), "adapter endpoint bank", nonnegative=True
        )
        current_fixed = _finite_real_array(
            self._adapter.original.activation,
            (96,),
            "original endpoint activation",
            nonnegative=True,
        )
        current_activation = _finite_real_array(
            self._adapter.activation,
            (96,),
            "adapter endpoint activation",
            nonnegative=True,
        )

        times = np.arange(1, steps + 1, dtype=np.float64) * float(physics_dt)
        times[-1] = float(dt)
        if motor_cut:
            _require(
                np.all(current_bank == 0)
                and np.all(current_fixed == 0)
                and np.all(current_activation == 0)
                and np.array_equal(endpoint, self._neutral),
                "Motor cut endpoint must be exact zero state and body neutral",
            )
            targets = np.broadcast_to(self._neutral, (steps, 42)).copy()
            error = {
                "bank_max_abs": 0.0,
                "fixed_activation_max_abs": 0.0,
                "activation_max_abs": 0.0,
                "target_max_abs": 0.0,
            }
        else:
            bank_alpha = -np.expm1(-times[:, None] / self._tau[None, :])
            bank = bank0[None, :, :] + bank_alpha[:, None, :] * (
                rate[None, :, None] - bank0[None, :, :]
            )
            pooled = self._pool @ rate
            fixed_alpha = -np.expm1(-times / FIXED_MUSCLE_TAU_S)
            fixed = fixed0[None, :] + fixed_alpha[:, None] * (
                pooled[None, :] - fixed0[None, :]
            )
            activation = np.empty((steps, 96), dtype=np.float64)
            activation[:, self._fixed] = fixed[:, self._fixed]
            hinge = np.maximum(bank - self._kappa[None, :, None], 0.0)
            for muscle in self._variable:
                members = np.flatnonzero(self._membership == muscle)
                low = self._slopes[muscle, members, :, 0]
                delta = self._slopes[muscle, members, :, 1] - low
                activation[:, muscle] = np.sum(
                    low[None, :, :] * bank[:, members, :], axis=(1, 2)
                ) + np.sum(delta[None, :, :] * hinge[:, members, :], axis=(1, 2))
            _require(
                np.isfinite(activation).all() and np.all(activation >= 0),
                "Substep recruitment produced invalid activation",
            )
            targets = self._rest[None, :] + 0.02 * (activation @ self._moment.T)
            error = {
                "bank_max_abs": _max_abs(bank[-1], current_bank),
                "fixed_activation_max_abs": _max_abs(fixed[-1], current_fixed),
                "activation_max_abs": _max_abs(activation[-1], current_activation),
                "target_max_abs": _max_abs(targets[-1], endpoint),
            }
            _require(
                np.allclose(bank[-1], current_bank, atol=ATOL, rtol=0)
                and np.allclose(fixed[-1], current_fixed, atol=ATOL, rtol=0)
                and np.allclose(activation[-1], current_activation, atol=ATOL, rtol=0)
                and np.allclose(targets[-1], endpoint, atol=ATOL, rtol=0),
                "Reconstructed endpoint differs from decoded adapter state",
            )
            targets[-1] = endpoint

        targets.flags.writeable = False
        times.flags.writeable = False
        return SubstepRecruitmentResult(targets, times, error)


def compute_substep_joint_targets(
    adapter: Any,
    bank_before: Any,
    fixed_activation_before: Any,
    rates: Any,
    endpoint_target: Any,
    *,
    motor_cut: bool,
    dt: float,
    physics_dt: float,
) -> SubstepRecruitmentResult:
    """One-shot form of :class:`CausalSubstepRecruitment`."""

    return CausalSubstepRecruitment(adapter).compute(
        bank_before,
        fixed_activation_before,
        rates,
        endpoint_target,
        motor_cut=motor_cut,
        dt=dt,
        physics_dt=physics_dt,
    )


__all__ = [
    "CausalSubstepRecruitment",
    "SubstepRecruitmentResult",
    "compute_substep_joint_targets",
]
