"""Stateful native adapter for a frozen monotone motor recruitment curve.

This module changes only the recorded 391-MN to 96-muscle transfer.  It wraps
the existing anatomical adapter and the existing motor-timescale trial; it does
not construct a network, graph, body, teacher, phase clock, or target path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

from tools.banc_m3_motor_timescale import MotorTimescaleTrial
from tools.verify_c_banc_observation import exact

ADAPTER_SCHEMA = "flylab.m3-monotone-recruitment-adapter.v1"
PATCH_SCHEMA = "flylab.m3-monotone-recruitment-patch.v1"
TRIAL_SCHEMA = "flylab.m3-monotone-recruitment-walking.v1"
FILTER_TAU_S = np.array([0.005, 0.01, 0.04], dtype=np.float64)
ATOL = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly_copy(value: Any, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _canonical_slopes(pool: np.ndarray) -> np.ndarray:
    result = np.zeros((96, 391, 3, 2), dtype=np.float64)
    result[:, :, 1, 0] = pool
    result[:, :, 1, 1] = pool
    return result


def _jsonable_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_jsonable_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_jsonable_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _identity(
    original: Any,
    calibration: dict[str, np.ndarray],
    groups: list[dict[str, Any]],
    tau: np.ndarray,
    kappa: np.ndarray,
    slopes: np.ndarray,
    membership: np.ndarray,
    motor_roster: list[str],
) -> str:
    payload = {
        "schema": ADAPTER_SCHEMA,
        "original_adapter_identity": original.identity,
        "groups": groups,
        "motor_roster": motor_roster,
        "initial_state": "zero_bank_zero_activation",
        "transfer": "low*bank+(high-low)*maximum(bank-kappa,0)",
        "moment_scale": 0.02,
        "tendon_max": 0.2,
        "tendon_rate_scale": 10.0,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for name in ("moment", "rest", "pool", "motor_ids", "ltm"):
        value = np.ascontiguousarray(calibration[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    for name, value in (
        ("tau", tau),
        ("kappa", kappa),
        ("slopes", slopes),
        ("membership", membership),
    ):
        value = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class MonotoneRecruitmentAdapter:
    """Apply one fixed continuous monotone curve to existing MN rates."""

    schema = ADAPTER_SCHEMA

    def __init__(
        self,
        original: Any,
        tau: Any,
        kappa: Any,
        slopes: Any,
        membership: Any,
        motor_roster: Any,
    ):
        self.original = original
        _require(
            hasattr(original, "calibration")
            and hasattr(original, "groups")
            and hasattr(original, "identity")
            and hasattr(original, "decode"),
            "Existing anatomical recruitment adapter required",
        )
        self.calibration = {
            key: np.array(original.calibration[key], copy=True)
            for key in ("moment", "rest", "pool", "motor_ids", "ltm")
        }
        self.groups = copy.deepcopy(original.groups)
        d = self.calibration
        _require(
            d["moment"].shape == (42, 96)
            and d["rest"].shape == (42,)
            and d["pool"].shape == (96, 391)
            and d["motor_ids"].shape == (391,)
            and d["ltm"].shape == (6,)
            and len(self.groups) == 96,
            "Complete original 391-MN/96-muscle anatomy required",
        )
        _require(
            all(np.isfinite(value).all() for value in d.values())
            and np.all(d["pool"] >= 0)
            and np.allclose(d["pool"].sum(axis=1), 1, atol=1e-15, rtol=0),
            "Finite normalized original calibration required",
        )
        motor_ids = np.asarray(d["motor_ids"])
        _require(
            motor_ids.dtype.kind in "iu"
            and len(np.unique(motor_ids)) == 391
            and np.array_equal(motor_ids, np.sort(motor_ids)),
            "Sorted unique original motor IDs required",
        )

        tau_array = np.asarray(tau)
        kappa_array = np.asarray(kappa)
        slope_array = np.asarray(slopes)
        member_array = np.asarray(membership)
        _require(
            tau_array.dtype == np.float64 and np.array_equal(tau_array, FILTER_TAU_S),
            "Exact float64 filter tau [0.005,0.01,0.04] required",
        )
        _require(
            kappa_array.shape == (391,)
            and kappa_array.dtype == np.float64
            and np.isfinite(kappa_array).all()
            and np.all(kappa_array >= 0),
            "Finite nonnegative float64 391-MN kappa required",
        )
        _require(
            slope_array.shape == (96, 391, 3, 2)
            and slope_array.dtype == np.float64
            and np.isfinite(slope_array).all()
            and np.all(slope_array >= 0),
            "Finite nonnegative float64 monotone slopes required",
        )
        _require(
            member_array.shape == (391,)
            and member_array.dtype == np.int64
            and np.all((member_array >= 0) & (member_array < 96)),
            "Exact int64 391-MN membership required",
        )
        roster = copy.deepcopy(list(motor_roster))
        _require(
            len(roster) == 391
            and len(set(roster)) == 391
            and all(isinstance(value, str) and value for value in roster),
            "Unique ordered original MN root roster required",
        )

        support = np.zeros((96, 391), dtype=bool)
        support[member_array, np.arange(391)] = True
        _require(
            np.array_equal(np.sum(d["pool"] > 0, axis=0), np.ones(391, dtype=int))
            and np.array_equal(np.argmax(d["pool"], axis=0), member_array),
            "Membership differs from original anatomical pool",
        )
        _require(
            np.all(slope_array[~support] == 0),
            "Monotone slopes changed anatomical support",
        )
        seen: list[int] = []
        for muscle, group in enumerate(self.groups):
            columns = list(group.get("columns", []))
            expected = np.flatnonzero(member_array == muscle).tolist()
            _require(
                bool(group.get("leg"))
                and bool(group.get("muscle"))
                and columns == expected,
                "Named muscle column provenance changed",
            )
            seen.extend(columns)
        _require(
            sorted(seen) == list(range(391)),
            "Muscle columns do not partition MN roster",
        )

        fixed = np.asarray(
            sorted(
                set(map(int, d["ltm"])).union(
                    muscle
                    for muscle in range(96)
                    if np.count_nonzero(member_array == muscle) == 1
                )
            ),
            dtype=np.int64,
        )
        _require(
            d["ltm"].dtype.kind in "iu"
            and len(np.unique(d["ltm"])) == 6
            and np.all((d["ltm"] >= 0) & (d["ltm"] < 96)),
            "Six unique original LTM rows required",
        )
        canonical = _canonical_slopes(d["pool"])
        _require(
            all(
                np.array_equal(slope_array[muscle], canonical[muscle])
                for muscle in fixed
            ),
            "Fixed LTM and singleton weights must equal original linear transfer",
        )
        supported_slopes = slope_array[member_array, np.arange(391)]
        minimum = np.min(supported_slopes, axis=-1)
        maximum = np.max(supported_slopes, axis=-1)
        lower = np.bincount(member_array, weights=minimum.sum(axis=1), minlength=96)
        upper = np.bincount(member_array, weights=maximum.sum(axis=1), minlength=96)
        counts = np.bincount(member_array, minlength=96)
        _require(np.all(counts > 0), "Every original muscle requires at least one MN")
        floor = 1e-6 / counts[member_array]
        _require(
            np.all(lower >= 0.5 - ATOL) and np.all(upper <= 2 + ATOL),
            "Per-muscle minimum/maximum slope sums must stay within 0.5-2",
        )
        _require(
            np.all(minimum.sum(axis=1) >= floor - ATOL),
            "Original per-MN minimum slope floor changed",
        )

        self.tau = _readonly_copy(tau_array, np.float64)
        self.kappa = _readonly_copy(kappa_array, np.float64)
        self.slopes = _readonly_copy(slope_array, np.float64)
        self.membership = _readonly_copy(member_array, np.int64)
        self.motor_roster = tuple(roster)
        self.fixed_muscles = _readonly_copy(fixed, np.int64)
        self.variable_muscles = _readonly_copy(
            np.setdiff1d(np.arange(96, dtype=np.int64), fixed), np.int64
        )
        self.lower_budget = _readonly_copy(lower, np.float64)
        self.upper_budget = _readonly_copy(upper, np.float64)
        self.motor_minimum_slope = _readonly_copy(minimum.sum(axis=1), np.float64)
        self.identity = _identity(
            original,
            d,
            self.groups,
            self.tau,
            self.kappa,
            self.slopes,
            self.membership,
            roster,
        )
        self.bank = np.zeros((391, 3), dtype=np.float64)
        original_activation = np.asarray(original.activation)
        _require(
            original_activation.shape == (96,)
            and original_activation.dtype == np.float64
            and np.isfinite(original_activation).all()
            and np.all(original_activation >= 0),
            "Finite nonnegative original fixed activation state required",
        )
        self.activation = self._activation_from(self.bank, original_activation)
        self.last_motor: list[dict[str, Any]] = copy.deepcopy(
            getattr(original, "last_motor", [])
        )

    def __getattr__(self, key: str) -> Any:
        return getattr(self.original, key)

    @property
    def tendon_inputs(self) -> dict[str, float]:
        return {
            self.groups[int(index)]["leg"] + "_tarsus": float(
                0.2 * np.tanh(self.activation[int(index)] / 10)
            )
            for index in self.calibration["ltm"]
        }

    def _activation_from(
        self, bank: np.ndarray, fixed_activation: np.ndarray
    ) -> np.ndarray:
        result = np.empty(96, dtype=np.float64)
        result[self.fixed_muscles] = fixed_activation[self.fixed_muscles]
        hinge = np.maximum(bank - self.kappa[:, None], 0.0)
        for muscle in self.variable_muscles:
            members = np.flatnonzero(self.membership == muscle)
            low = self.slopes[muscle, members, :, 0]
            delta = self.slopes[muscle, members, :, 1] - low
            # Explicit stable hinge form; negative delta is a lower positive
            # high-region slope, not a negative anatomical connection.
            result[muscle] = np.sum(low * bank[members]) + np.sum(
                delta * hinge[members]
            )
        return result

    def decode(
        self, rates: Any, *, motor_cut: bool = False, dt: float = 0.005
    ) -> tuple[np.ndarray, np.ndarray]:
        rate = np.asarray(rates)
        _require(
            rate.shape == (391,)
            and np.issubdtype(rate.dtype, np.number)
            and np.isrealobj(rate)
            and np.isfinite(rate).all()
            and np.all(rate >= 0)
            and type(motor_cut) is bool
            and isinstance(dt, (int, float, np.floating))
            and math.isfinite(float(dt))
            and 0 < float(dt) <= 0.05,
            "Finite nonnegative 391-MN rates and actual timestep required",
        )
        original_target, adhesion = self.original.decode(
            np.array(rate, dtype=np.float64, copy=True),
            motor_cut=motor_cut,
            dt=float(dt),
        )
        _require(
            np.asarray(self.original.activation).shape == (96,)
            and np.isfinite(self.original.activation).all()
            and np.all(np.asarray(self.original.activation) >= 0),
            "Original fixed activation state invalid",
        )
        if motor_cut:
            self.bank.fill(0)
            self.activation.fill(0)
            _require(
                np.all(np.asarray(self.original.activation) == 0)
                and np.array_equal(np.asarray(original_target), self.body.neutral),
                "Original motor cut did not return exact neutral zero state",
            )
            target = np.array(original_target, copy=True)
        else:
            alpha = -np.expm1(-float(dt) / self.tau)
            self.bank += alpha[None, :] * (rate[:, None] - self.bank)
            self.activation[:] = self._activation_from(
                self.bank, np.asarray(self.original.activation, dtype=np.float64)
            )
            _require(
                np.isfinite(self.activation).all() and np.all(self.activation >= 0),
                "Monotone recruitment produced invalid activation",
            )
            target = self.calibration["rest"] + 0.02 * (
                self.calibration["moment"] @ self.activation
            )
        self.last_motor = copy.deepcopy(getattr(self.original, "last_motor", []))
        return np.asarray(target).copy(), np.asarray(adhesion).copy()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hash": self.identity,
            "parameters": {
                "tau": self.tau.copy(),
                "kappa": self.kappa.copy(),
                "slopes": self.slopes.copy(),
                "membership": self.membership.copy(),
                "motor_roster": list(self.motor_roster),
                "fixed_muscles": self.fixed_muscles.copy(),
                "lower_budget": self.lower_budget.copy(),
                "upper_budget": self.upper_budget.copy(),
                "motor_minimum_slope": self.motor_minimum_slope.copy(),
            },
            "calibration": copy.deepcopy(self.calibration),
            "groups": copy.deepcopy(self.groups),
            "original": self.original.snapshot(),
            "bank": self.bank.copy(),
            "activation": self.activation.copy(),
            "last_motor": copy.deepcopy(self.last_motor),
        }

    def restore(self, saved: dict[str, Any]) -> None:
        _require(
            isinstance(saved, dict)
            and set(saved)
            == {
                "schema",
                "hash",
                "parameters",
                "calibration",
                "groups",
                "original",
                "bank",
                "activation",
                "last_motor",
            },
            "Complete monotone recruitment snapshot required",
        )
        _require(
            saved["schema"] == self.schema and saved["hash"] == self.identity,
            "Monotone recruitment identity changed",
        )
        expected_parameters = self.snapshot()["parameters"]
        _require(
            _jsonable_equal(saved["parameters"], expected_parameters),
            "Fixed monotone recruitment parameters changed",
        )
        _require(
            _jsonable_equal(saved["calibration"], self.calibration)
            and _jsonable_equal(saved["groups"], self.groups),
            "Fixed monotone anatomy changed",
        )
        bank = np.asarray(saved["bank"])
        active = np.asarray(saved["activation"])
        _require(
            bank.shape == (391, 3)
            and bank.dtype == np.float64
            and np.isfinite(bank).all()
            and np.all(bank >= 0),
            "Finite nonnegative float64 filter bank required",
        )
        _require(
            active.shape == (96,)
            and active.dtype == np.float64
            and np.isfinite(active).all()
            and np.all(active >= 0),
            "Finite nonnegative float64 activation required",
        )
        original_saved = saved["original"]
        _require(
            isinstance(original_saved, dict)
            and isinstance(original_saved.get("activation"), np.ndarray)
            and original_saved["activation"].shape == (96,)
            and original_saved["activation"].dtype == np.float64
            and np.isfinite(original_saved["activation"]).all()
            and np.all(original_saved["activation"] >= 0),
            "Complete original fixed activation required",
        )
        expected_active = self._activation_from(bank, original_saved["activation"])
        _require(
            np.allclose(active, expected_active, atol=ATOL, rtol=0),
            "Activation disagrees with saved bank and fixed transfer",
        )
        _require(
            _jsonable_equal(saved["last_motor"], original_saved.get("last_motor", [])),
            "Last motor diagnostics disagree with original adapter",
        )
        before = self.original.snapshot()
        try:
            self.original.restore(original_saved)
            self.bank[:] = bank
            self.activation[:] = active
            self.last_motor = copy.deepcopy(saved["last_motor"])
        except Exception as error:
            try:
                self.original.restore(before)
            except Exception as rollback_error:  # noqa: BLE001
                error.add_note(
                    "Original adapter rollback failed: " + repr(rollback_error)
                )
            raise


class MonotoneRecruitmentTrial(MotorTimescaleTrial):
    """Wrap the unchanged motor-timescale trial with the recruitment adapter."""

    schema = TRIAL_SCHEMA

    def __init__(
        self,
        base: MotorTimescaleTrial,
        tau: Any,
        kappa: Any,
        slopes: Any,
        membership: Any,
    ):
        if not isinstance(base, MotorTimescaleTrial):
            raise TypeError("Existing motor-timescale trial layer required")
        self.__dict__.update(base.__dict__)
        original_adapter = self.adapter
        try:
            motor_ids = np.asarray(original_adapter.calibration["motor_ids"])
            roster = [self.graph.nodes[int(index)]["id"] for index in motor_ids]
            if len(roster) != 391 or len(set(roster)) != 391:
                raise ValueError("Unique original MN root provenance required")
            candidate = MonotoneRecruitmentAdapter(
                original_adapter, tau, kappa, slopes, membership, roster
            )
            self.monotone_recruitment_patch = {
                "schema": PATCH_SCHEMA,
                "source_adapter_identity": original_adapter.identity,
                "adapter_identity": candidate.identity,
                "tau": candidate.tau.copy(),
                "kappa": candidate.kappa.copy(),
                "slopes": candidate.slopes.copy(),
                "membership": candidate.membership.copy(),
                "motor_ids": motor_ids.copy(),
                "motor_roster": roster,
                "fixed_muscles": candidate.fixed_muscles.copy(),
                "lower_budget": candidate.lower_budget.copy(),
                "upper_budget": candidate.upper_budget.copy(),
                "motor_minimum_slope": candidate.motor_minimum_slope.copy(),
                "source_motor_timescale_patch": copy.deepcopy(
                    getattr(self, "motor_timescale_patch", None)
                ),
            }
            self.adapter = candidate
            self.tendon_inputs.update(candidate.tendon_inputs)
        except Exception as error:
            base.closed = True
            try:
                self.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note("Body cleanup failed: " + repr(close_error))
            raise

    def snapshot(self) -> dict[str, Any]:
        self.tendon_inputs.update(self.adapter.tendon_inputs)
        return {
            **super().snapshot(),
            "monotone_recruitment_patch": copy.deepcopy(
                self.monotone_recruitment_patch
            ),
        }

    @classmethod
    def from_checkpoint(
        cls, graph: Any, saved: dict[str, Any]
    ) -> MonotoneRecruitmentTrial:
        if saved.get("schema") != cls.schema:
            raise ValueError("Explicit monotone recruitment checkpoint required")
        patch = saved.get("monotone_recruitment_patch")
        if not isinstance(patch, dict) or patch.get("schema") != PATCH_SCHEMA:
            raise ValueError("Complete monotone recruitment provenance required")
        source = copy.deepcopy(saved)
        source.pop("monotone_recruitment_patch")
        source["schema"] = MotorTimescaleTrial.schema
        source["adapter"] = saved["adapter"]["original"]
        base = MotorTimescaleTrial.from_checkpoint(graph, source)
        trial = None
        try:
            if base.adapter.identity != patch.get("source_adapter_identity"):
                raise ValueError("Source anatomical adapter identity changed")
            if not _jsonable_equal(
                getattr(base, "motor_timescale_patch", None),
                patch.get("source_motor_timescale_patch"),
            ):
                raise ValueError("Source motor-timescale provenance changed")
            trial = cls(
                base, patch["tau"], patch["kappa"], patch["slopes"], patch["membership"]
            )
            trial.adapter.restore(saved["adapter"])
            trial.tendon_inputs.update(trial.adapter.tendon_inputs)
            exact(saved, trial.snapshot(), "Whole monotone recruitment checkpoint")
            return trial
        except Exception as error:
            owner = trial if trial is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note("Body cleanup failed: " + repr(close_error))
            raise
