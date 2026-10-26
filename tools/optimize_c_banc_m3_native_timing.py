"""E181 one-pass native timing search. Import is inert; execution is gated.

Only the saved E150 three-bank slopes change, at zero body and neural time.
Every baseline or candidate has an independent native child process.
"""

# Broad catches preserve first neural/body failures while independently saving raw
# state, clock, native contact, receipts, and secondary persistence failures.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools import banc_m3_monotone_recruitment as monotone
from tools import probe_c_banc_m3_native_objective as e180
from tools import verify_c_banc_m3_native_monotone_body as e150
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.analyze_c_banc_biological_gait import analyze, load_model

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e181.native-timing-freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e181.parent-source-review.v1"
BASELINE_REVIEW_SCHEMA = "flylab.m3.e181.parent-baseline-review.v1"
AXIS_REVIEW_SCHEMA = "flylab.m3.e181.parent-axis-review.v1"
OWNED = (
    "tools/optimize_c_banc_m3_native_timing.py",
    "tests/test_c_banc_native_timing.py",
)
LIMITS = {"q_RMS": 0.02, "q_max_abs": 0.04, "LTM_RMS": 0.02, "LTM_max_abs": 0.04}
CONTROLS = 600
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
STEPS = (0.125, 0.0625, 0.03125, 0.015625)
TAU = np.array([0.005, 0.01, 0.04], np.float64)
OUTPUT_DIRECTORY = "verification/banc-natural-gait-m3-20260919/e181-native-timing"
CHILD_INTERNAL_S = 100
CHILD_EXTERNAL_S = 105
SUPERVISOR_S = 110
AXIS_S = 270
STARTUP_S = 30
ARM_BYTES = 1 << 30
CAMPAIGN_BYTES = 14 << 30
FREE_BYTES = 20 << 30


class ResourceGuard(RuntimeError):
    """Frozen process budget exhausted; scientific arm is incomplete."""


class PreflightError(RuntimeError):
    """Saved-startup calculation failed with its attempted steps retained."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class ValidatedGate:
    freeze: dict[str, Any]
    freeze_sha256: str
    source_review_sha256: str
    mode: str


def tilt_slopes(
    slopes: Any, tau: Any, muscle_legs: Any, fixed: Any, theta: Any
) -> np.ndarray:
    """Preserve each low/high bank sum using min and separate surplus weights."""
    source = np.asarray(slopes)
    times = np.asarray(tau)
    legs = tuple(muscle_legs)
    fixed_rows = np.asarray(fixed)
    angles = np.asarray(theta)
    if (
        source.dtype != np.float64
        or source.ndim != 4
        or source.shape[2:] != (3, 2)
        or times.dtype != np.float64
        or not np.array_equal(times, TAU)
        or len(legs) != source.shape[0]
        or any(leg not in LEGS for leg in legs)
        or fixed_rows.ndim != 1
        or fixed_rows.dtype.kind not in "iu"
        or len(np.unique(fixed_rows)) != len(fixed_rows)
        or np.any(fixed_rows < 0)
        or np.any(fixed_rows >= len(legs))
        or angles.shape != (6,)
        or angles.dtype != np.float64
        or not np.isfinite(angles).all()
        or np.any(np.abs(angles) > STEPS[0])
        or not np.isfinite(source).all()
        or np.any(source < 0)
    ):
        raise ValueError("Exact finite E181 slopes/tau/legs/fixed/theta required")
    result = source.copy(order="K")
    if not np.any(angles):
        return result
    fixed_set = set(map(int, fixed_rows))
    for row, leg in enumerate(legs):
        if row in fixed_set or angles[LEGS.index(leg)] == 0:
            continue
        weights = (times / 0.01) ** angles[LEGS.index(leg)]
        low = source[row, :, :, 0]
        high = source[row, :, :, 1]
        common = np.minimum(low, high)
        low_extra = low - common
        high_extra = high - common

        def redistribute(
            component: np.ndarray, bank_weights: np.ndarray = weights
        ) -> np.ndarray:
            numerator = component * bank_weights[None, :]
            sums = component.sum(axis=1, keepdims=True)
            denominator = numerator.sum(axis=1, keepdims=True)
            return np.divide(
                numerator * sums,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )

        result[row, :, :, 0] = redistribute(common) + redistribute(low_extra)
        result[row, :, :, 1] = redistribute(common) + redistribute(high_extra)
    if not np.array_equal(result == 0, source == 0):
        raise ValueError("Slope support changed")
    original_sum = source.sum(axis=2)
    result_sum = result.sum(axis=2)
    if not np.allclose(original_sum, result_sum, atol=2e-14, rtol=2e-14):
        raise ValueError("Constant-input slope sums changed")
    if fixed_rows.size and result[fixed_rows].tobytes() != source[fixed_rows].tobytes():
        raise ValueError("Fixed-muscle bytes changed")
    return result


def forward_deficit(speed: Any, window: Any) -> float:
    """Mean physical forward-velocity shortfall on exactly [0.5,3) seconds."""
    values = np.asarray(speed)
    selected = np.asarray(window)
    if (
        values.shape != (601,)
        or selected.shape != (601,)
        or selected.dtype != np.bool_
        or int(selected.sum()) != 500
        or not np.array_equal(
            selected, (np.arange(601) * 0.005 >= 0.5) & (np.arange(601) * 0.005 < 3)
        )
        or not np.isfinite(values).all()
    ):
        raise ValueError("Complete finite E181 physical speed/window required")
    return float(np.maximum(0.0, 0.5 - values[selected]).mean())


def score_from_trace(path: Path) -> float:
    data = load_model(path)
    return forward_deficit(data["speed"], data["window"])


def propose_axis(
    center: Any, axis: str, direction: str, step: float
) -> tuple[float, ...]:
    angles = np.asarray(center, dtype=np.float64)
    if (
        angles.shape != (6,)
        or not np.isfinite(angles).all()
        or np.any(np.abs(angles) > STEPS[0])
        or axis not in LEGS
        or direction not in ("plus", "minus")
        or step not in STEPS
    ):
        raise ValueError("Frozen E181 coordinate proposal required")
    candidate = angles.copy()
    candidate[LEGS.index(axis)] += step if direction == "plus" else -step
    if np.any(np.abs(candidate) > STEPS[0]):
        raise ValueError("E181 theta box exceeded")
    return tuple(map(float, candidate))


def preflight_direction(
    center: Any,
    axis: str,
    direction: str,
    startup_callback: Callable[[tuple[float, ...]], dict[str, Any]],
    seen: set[tuple[float, ...]],
) -> dict[str, Any]:
    """Saved-startup-only backtracking; all four attempted steps are reported."""
    attempts: list[dict[str, Any]] = []
    selected = None
    for step in STEPS:
        try:
            theta = propose_axis(center, axis, direction, step)
        except ValueError:
            attempts.append({"step": step, "theta": None, "status": "box_reject"})
            continue
        if theta in seen:
            attempts.append({"step": step, "theta": list(theta), "status": "duplicate"})
            continue
        seen.add(theta)
        try:
            screen = startup_callback(theta)
        except Exception as error:
            attempts.append(
                {
                    "step": step,
                    "theta": list(theta),
                    "status": "technical_error",
                    "error": repr(error),
                }
            )
            raise PreflightError("Saved startup callback failed", attempts) from error
        if (
            not isinstance(screen, dict)
            or type(screen.get("strict_E141_startup_screen")) is not bool
        ):
            attempts.append(
                {
                    "step": step,
                    "theta": list(theta),
                    "status": "technical_error",
                    "error": "Invalid startup screen",
                }
            )
            raise PreflightError("Complete saved-startup screen required", attempts)
        feasible = screen["strict_E141_startup_screen"]
        attempts.append(
            {
                "step": step,
                "theta": list(theta),
                "status": "feasible" if feasible else "startup_reject",
                "screen": screen,
            }
        )
        if feasible:
            selected = theta
            break
    return {
        "axis": axis,
        "direction": direction,
        "attempts": attempts,
        "selected_theta": list(selected) if selected is not None else None,
    }


def _finite_metric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def screen_candidate(baseline: dict, candidate: dict) -> dict[str, Any]:
    """Eligibility against the ORIGINAL E181 baseline, never the moving center."""
    b = baseline.get("physical_3s", {})
    c = candidate.get("physical_3s", {})
    fields = (
        "fixed_initial_heading_net_forward_mm",
        "net_over_path",
        "native_slip_load_weighted_mm_s",
    )
    numeric = all(
        _finite_metric(b.get(key)) and _finite_metric(c.get(key)) for key in fields
    )
    denominators = (
        numeric and b[fields[0]] > 0 and b[fields[1]] > 0 and b[fields[2]] >= 0
    )
    screens = {
        "valid_3s_startup_health": candidate.get("validity") == "valid_3s_diagnostic"
        and candidate.get("startup_pass") is True
        and candidate.get("technical_failure") is False
        and candidate.get("measured_health_stop") is False,
        "forward_at_least_95pct": bool(
            denominators and c[fields[0]] >= 0.95 * b[fields[0]]
        ),
        "net_over_path_at_least_95pct": bool(
            denominators and c[fields[1]] >= 0.95 * b[fields[1]]
        ),
        "supported_slip_not_increased": bool(
            denominators and c[fields[2]] <= b[fields[2]]
        ),
        "objective_finite": _finite_metric(candidate.get("objective_mm_s"))
        and candidate.get("objective_mm_s", -1) >= 0,
    }
    return {
        "screens": screens,
        "eligible": bool(
            baseline.get("validity") == "valid_3s_diagnostic" and all(screens.values())
        ),
    }


def select_axis(
    center_theta: Any, center_result: dict, directions: dict, baseline_result: dict
) -> dict[str, Any]:
    """Choose once after both signed arms used the identical center."""
    center = np.asarray(center_theta, np.float64)
    center_j = center_result.get("objective_mm_s")
    if (
        center.shape != (6,)
        or not np.isfinite(center).all()
        or not _finite_metric(center_j)
        or center_j < 0
    ):
        raise ValueError("Valid center theta and physical objective required")
    threshold = max(1e-6, 0.01 * center_j)
    eligible = []
    ledger = {}
    for order, direction in enumerate(("plus", "minus")):
        entry = directions.get(direction)
        if entry is None:
            ledger[direction] = {"eligible": False, "reason": "no_native_arm"}
            continue
        theta = np.asarray(entry.get("theta"), np.float64)
        result = entry.get("result", {})
        if (
            theta.shape != (6,)
            or not np.isfinite(theta).all()
            or np.any(np.abs(theta) > STEPS[0])
        ):
            raise ValueError("Direction result lacks valid theta")
        screen = screen_candidate(baseline_result, result)
        j = result.get("objective_mm_s")
        improves = bool(screen["eligible"] and center_j - j >= threshold)
        ledger[direction] = {
            **screen,
            "improves": improves,
            "objective_mm_s": j,
            "theta": theta.tolist(),
        }
        if improves:
            eligible.append((j, float(theta @ theta), order, direction, theta.tolist()))
    selected = min(eligible) if eligible else None
    return {
        "center_theta": center.tolist(),
        "center_objective_mm_s": center_j,
        "minimum_improvement_mm_s": threshold,
        "directions": ledger,
        "selected_direction": selected[3] if selected else None,
        "selected_theta": selected[4] if selected else center.tolist(),
        "selected_objective_mm_s": selected[0] if selected else center_j,
        "candidate_adopted": False,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
    }


def saved_startup_screen(
    theta: tuple[float, ...],
    slopes: np.ndarray,
    tau: np.ndarray,
    muscle_legs: tuple[str, ...],
    fixed: np.ndarray,
    kappa: np.ndarray,
    membership: np.ndarray,
    bank: np.ndarray,
    original_activation: np.ndarray,
    moment: np.ndarray,
    rest: np.ndarray,
    ltm: np.ndarray,
    startup: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Open-loop screening using the saved E181 baseline 99-bank prefix."""
    if (
        bank.shape != (100, 391, 3)
        or original_activation.shape != (100, 96)
        or kappa.shape != (391,)
        or membership.shape != (391,)
        or moment.shape != (42, 96)
        or rest.shape != (42,)
        or ltm.shape != (6,)
        or not np.array_equal(
            np.array([muscle_legs[int(index)] for index in ltm]), np.array(LEGS)
        )
        or not set(map(int, ltm)).issubset(set(map(int, fixed)))
        or startup["target_command"].shape != (99, 42)
        or startup["target_tendon"].shape != (99, 6)
        or not all(
            np.isfinite(x).all()
            for x in (bank, original_activation, kappa, moment, rest)
        )
    ):
        raise ValueError("Complete saved 99-control E181 startup inputs required")
    tilted = tilt_slopes(slopes, tau, muscle_legs, fixed, np.asarray(theta, np.float64))
    activation = original_activation[1:].copy()
    hinge = np.maximum(bank[1:] - kappa[None, :, None], 0)
    for muscle in range(96):
        if muscle in fixed:
            continue
        members = np.flatnonzero(membership == muscle)
        low = tilted[muscle, members, :, 0]
        high = tilted[muscle, members, :, 1]
        activation[:, muscle] = np.einsum(
            "tmk,mk->t", bank[1:, members], low
        ) + np.einsum("tmk,mk->t", hinge[:, members], high - low)
    command = rest[None, :] + 0.02 * activation @ moment.T
    # LTM rows are fixed, so their original E150 tendon transfer is exact.
    tendon = 0.2 * np.tanh(activation[:, ltm] / 10)
    q_error = command - startup["target_command"]
    t_error = tendon - startup["target_tendon"]
    metrics = {
        "q_RMS": float(np.sqrt(np.mean(q_error**2))),
        "q_max_abs": float(np.max(np.abs(q_error))),
        "LTM_RMS": float(np.sqrt(np.mean(t_error**2))),
        "LTM_max_abs": float(np.max(np.abs(t_error))),
    }
    passed = bool(
        all(np.isfinite(list(metrics.values())))
        and all(metrics[key] <= LIMITS[key] for key in LIMITS)
    )
    return {
        "strict_E141_startup_screen": passed,
        "metrics": metrics,
        "limits": LIMITS,
        "source": "saved_E181_baseline_bank_open_loop",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Frozen input must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen input escaped repository")
    return resolved


def _check_hashes(mapping: dict[str, str]) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Nonempty frozen hash map required")
    for relative, expected in mapping.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(_safe(relative)) != expected
        ):
            raise ValueError("Frozen input/source changed: " + str(relative))


def verified_inputs(
    freeze_path: Path,
    freeze_sha: str,
    review_path: Path,
    review_sha: str,
    mode: str,
    baseline_review_path: Path | None = None,
    baseline_review_sha: str | None = None,
    previous_axis_sha: str | None = None,
) -> ValidatedGate:
    """Validate all immutable bytes before GraphStore or StateStore access."""
    if mode != "baseline" and mode not in LEGS:
        raise ValueError("Exact E181 baseline or axis required")
    if len(freeze_sha) != 64 or _sha256(freeze_path) != freeze_sha:
        raise ValueError("Exact E181 freeze required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "schema": SCHEMA,
        "duration_s": 3.0,
        "controls": CONTROLS,
        "legs": list(LEGS),
        "theta_box": STEPS[0],
        "backtrack_steps": list(STEPS),
        "objective_window_s": [0.5, 3.0],
        "forward_target_mm_s": 0.5,
        "improvement_floor_mm_s": 1e-6,
        "improvement_fraction": 0.01,
        "startup_limits": LIMITS,
        "neural_steps_per_control": 20,
        "physics_steps_per_control": 50,
        "control_dt_s": 0.005,
        "neural_dt_s": 0.00025,
        "physics_dt_s": 0.0001,
        "child_internal_wall_limit_s": CHILD_INTERNAL_S,
        "child_external_wall_limit_s": CHILD_EXTERNAL_S,
        "supervisor_limit_s": SUPERVISOR_S,
        "axis_wall_limit_s": AXIS_S,
        "minimum_free_disk_bytes": FREE_BYTES,
        "maximum_raw_bytes": ARM_BYTES,
        "maximum_campaign_bytes": CAMPAIGN_BYTES,
        "automatic_retries": 0,
        "original_processes_max": 13,
        "offline_solver_calls": 0,
        "coordinate_sweeps_max": 1,
        "native_objective_evaluations_max": 13,
        "F6_exposures": 0,
        "candidate_adopted": False,
        "goal_status": "active",
        "cumulative_before_s": 3533.5252058999613,
        "cumulative_limit_s": 4994,
        "owned_files": list(OWNED),
        "output_directory": OUTPUT_DIRECTORY,
    }
    if any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError("E181 fixed experiment contract changed")
    if freeze.get(
        "design_path"
    ) != "docs/BANC_M3_E181_NATIVE_TIMING_DESIGN_20260923.md" or _sha256(
        _safe(freeze["design_path"])
    ) != freeze.get("design_sha256"):
        raise ValueError("E181 design changed")
    for key in (
        "graph_path",
        "neutral_checkpoint",
        "startup_path",
        "reference_directory",
    ):
        _safe(freeze[key])
    for key in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        _check_hashes(freeze[key])
    if not isinstance(freeze.get("graph_hash"), str) or len(freeze["graph_hash"]) != 64:
        raise ValueError("Frozen graph identity required")
    if not any(
        Path(name).is_relative_to(freeze["neutral_checkpoint"])
        for name in freeze["input_sha256"]
    ):
        raise ValueError("Neutral checkpoint absent from frozen inputs")
    if freeze["startup_path"] not in freeze["input_sha256"]:
        raise ValueError("Startup absent from frozen inputs")
    if not isinstance(freeze.get("baseline_reference_sha256"), dict):
        raise TypeError("Frozen E180 baseline reference required")
    _check_hashes(freeze["baseline_reference_sha256"])
    if any(
        not Path(name).is_relative_to(freeze["reference_directory"])
        for name in freeze["baseline_reference_sha256"]
    ):
        raise ValueError("Baseline reference escaped E180 directory")
    if len(review_sha) != 64 or _sha256(review_path) != review_sha:
        raise ValueError("Exact parent source review required")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("accepts_native_timing") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("Accepted parent source gate required")
    for name, digest in review["source_sha256"].items():
        if _sha256(_safe(name)) != digest:
            raise ValueError("Reviewed E181 source changed: " + name)
    if mode != "baseline":
        if (
            baseline_review_path is None
            or baseline_review_sha is None
            or len(baseline_review_sha) != 64
            or _sha256(baseline_review_path) != baseline_review_sha
        ):
            raise ValueError("Candidate requires exact parent baseline review")
        baseline = json.loads(baseline_review_path.read_text(encoding="utf-8"))
        if (
            baseline.get("schema") != BASELINE_REVIEW_SCHEMA
            or baseline.get("accepts_baseline") is not True
            or baseline.get("permits_axes") is not True
            or baseline.get("blocking_defects") != []
            or baseline.get("freeze_sha256") != freeze_sha
            or not isinstance(baseline.get("baseline_artifact_sha256"), dict)
            or not baseline["baseline_artifact_sha256"]
        ):
            raise ValueError("Parent baseline comparison did not authorize candidate")
        _check_hashes(baseline["baseline_artifact_sha256"])
        if mode != LEGS[0]:
            previous = LEGS[LEGS.index(mode) - 1]
            result = _safe(f"{OUTPUT_DIRECTORY}/axis-{previous}/result.json")
            if previous_axis_sha is None or _sha256(result) != previous_axis_sha:
                raise ValueError("Exact previous axis result hash required")
            parent_review = result.parent / "parent-axis-review.json"
            if not parent_review.is_file():
                raise ValueError("Parent previous-axis review required")
            axis_review = json.loads(parent_review.read_text(encoding="utf-8"))
            if (
                axis_review.get("schema") != AXIS_REVIEW_SCHEMA
                or axis_review.get("accepts_next_axis") is not True
                or axis_review.get("blocking_defects") != []
                or axis_review.get("freeze_sha256") != freeze_sha
                or axis_review.get("axis_result_sha256") != previous_axis_sha
            ):
                raise ValueError("Previous axis was not parent-approved")
    return ValidatedGate(freeze, freeze_sha, review_sha, mode)


def _neutral(saved: dict[str, Any]) -> None:
    if (
        saved.get("schema") != monotone.TRIAL_SCHEMA
        or saved.get("control_tick") != 0
        or not np.array_equal(saved["adapter"]["bank"], np.zeros((391, 3)))
        or not np.array_equal(saved["adapter"]["activation"], np.zeros(96))
    ):
        raise ValueError("Whole E150 monotone zero checkpoint required")


def restore_arm(graph: Any, saved: dict[str, Any], theta: tuple[float, ...]):
    """Construct an independent original-schema trial at time zero only."""
    _neutral(saved)
    if theta == (0.0,) * 6:
        owner = monotone.MonotoneRecruitmentTrial.from_checkpoint(graph, saved)
        try:
            protocol.exact(saved, owner.snapshot(), "E181 zero whole checkpoint")
            return owner
        except BaseException:
            owner.close()
            raise
    source = copy.deepcopy(saved)
    source.pop("monotone_recruitment_patch")
    source["schema"] = protocol.MotorTimescaleTrial.schema
    source["adapter"] = copy.deepcopy(saved["adapter"]["original"])
    base = protocol.MotorTimescaleTrial.from_checkpoint(graph, source)
    owner = None
    try:
        protocol.exact(source, base.snapshot(), "E181 exact unwrapped E150 source")
        patch = saved["monotone_recruitment_patch"]
        slopes = tilt_slopes(
            patch["slopes"],
            patch["tau"],
            tuple(group["leg"] for group in base.adapter.groups),
            patch["fixed_muscles"],
            np.asarray(theta, np.float64),
        )
        owner = monotone.MonotoneRecruitmentTrial(
            base, patch["tau"], patch["kappa"], slopes, patch["membership"]
        )
        e150.verify_wrapper_provenance(saved, source, base, owner)
        if not np.array_equal(
            owner.adapter.bank, np.zeros((391, 3))
        ) or not np.array_equal(owner.adapter.activation, np.zeros(96)):
            raise ValueError("Candidate started with nonzero adapter state")
        return owner
    except BaseException:
        (owner or base).close()
        raise


def summarize_arm(
    reason: str,
    health_stop: bool,
    controls: int,
    native_samples: int,
    neural_s: float | None,
    physical_s: float | None,
    clock_valid: bool,
    startup_ok: bool | None,
    raw_errors: list,
    analysis_errors: list,
) -> dict[str, Any]:
    complete = bool(
        reason == "duration"
        and controls == CONTROLS
        and native_samples == 50 * CONTROLS
        and clock_valid
        and neural_s is not None
        and physical_s is not None
        and abs(neural_s - 3.0) <= 1e-10
        and abs(physical_s - 3.0) <= 1e-8
    )
    technical = bool(
        raw_errors or analysis_errors or reason in ("execution_error", "resource_guard")
    )
    validity = (
        "incomplete"
        if technical
        else "measured_native_health_stop"
        if health_stop
        else "valid_3s_diagnostic"
        if complete and startup_ok is True
        else "incomplete"
    )
    return {
        "validity": validity,
        "technical_failure": technical,
        "stop_reason": reason,
        "measured_health_stop": health_stop,
        "complete_3s": complete,
        "startup_pass": startup_ok,
        "completed_controls": controls,
        "native_samples": native_samples,
        "actual_neural_s": neural_s,
        "actual_physical_s": physical_s,
        "clock_valid": clock_valid,
        "raw_persistence_errors": raw_errors,
        "analysis_errors": analysis_errors,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
        "candidate_adopted": False,
    }


def run_arm(
    graph: Any,
    neutral: dict,
    theta: tuple[float, ...],
    output: Path,
    *,
    startup: dict[str, np.ndarray],
    limits: dict[str, float],
    wall_limit_s: float = 100.0,
    gate: ValidatedGate | None = None,
    process_began: float | None = None,
) -> dict[str, Any]:
    """Run exactly one reviewed arm; caller must first pass verified_inputs."""
    if (
        not isinstance(gate, ValidatedGate)
        or gate.freeze.get("schema") != SCHEMA
        or (gate.mode == "baseline" and theta != (0.0,) * 6)
        or gate.mode not in ("baseline", *LEGS)
        or len(theta) != 6
        or not np.isfinite(theta).all()
        or np.any(np.abs(theta) > STEPS[0])
        or limits != LIMITS
        or wall_limit_s > 100
    ):
        raise ValueError("Frozen E181 arm/limits required")
    _neutral(neutral)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    if process_began is None:
        process_began = began
    session = restore = journal = trace_stream = None
    rows, native, native_state, raw_errors, analysis_errors = [], [], [], [], []
    clock = protocol.AdvanceLedger()
    reason, health_stop, primary = "duration", False, None
    ledger = {
        "schema": "flylab.m3.e181.execution-ledger.v1",
        "construction_attempts": [],
        "clock": {},
        "completed_controls": 0,
        "candidate_adopted": False,
        "native_objective_evaluations": 1,
        "offline_solver_calls": 0,
    }

    def persist():
        ledger["construction_totals"] = protocol._construction_totals(
            ledger["construction_attempts"]
        )
        ledger["process_wall_s"] = time.perf_counter() - process_began
        protocol.write_json(out / "advance-ledger.json", ledger)

    persist()
    metadata = {
        "schema": "flylab.m3.e181.arm.v1",
        "theta": list(theta),
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
        "source_adapter_identity": neutral["monotone_recruitment_patch"][
            "adapter_identity"
        ],
        "source_network_identity": neutral["network"]["model_hash"],
    }
    protocol.write_json(out / "arm-provenance.json", metadata)
    try:
        if time.perf_counter() - process_began >= wall_limit_s - 10:
            raise ResourceGuard("Whole E181 process reserve exhausted before model")
        session = protocol.profiled_construction(
            "restore_e181_arm",
            lambda: restore_arm(graph, neutral, theta),
            ledger["construction_attempts"],
            persist,
            on_return=lambda _owner: None,
        )
        if session.network.device != "cuda":
            raise ValueError("Full BANC CUDA network required")
        if (
            graph.n != 158706
            or session.network.tick != 0
            or session.control_tick != 0
            or abs(session.body.physics_time()) > 1e-8
        ):
            raise ValueError(
                "Full 158706-neuron graph and zero neural/body clocks required"
            )
        metadata["actual_device"] = str(session.network.device)
        metadata["graph_neurons"] = int(graph.n)
        metadata["adapter_identity"] = session.adapter.identity
        metadata["body_model_hash"] = session.body.model_hash
        protocol.write_json(out / "arm-provenance.json", metadata)
        patch = session.monotone_recruitment_patch
        np.savez_compressed(
            out / "timing-parameters.npz",
            tau=patch["tau"],
            kappa=patch["kappa"],
            slopes=patch["slopes"],
            membership=patch["membership"],
            motor_ids=patch["motor_ids"],
            fixed_muscles=patch["fixed_muscles"],
        )
        StateStore.save(out / "start", session.snapshot())
        rows.append(capture_control(session))
        restore = protocol.install_native_observer(session.body, native)
        journal = protocol.ControlJournal(out / "advance-journal.jsonl")
        observe = e150.make_observer(native_state, original=lambda *_: {})
        detailed = np.asarray(session.adapter.calibration["motor_ids"][:1], np.int64)
        trace_stream = (out / "trace.jsonl").open("x", encoding="utf-8", buffering=1)
        trace_stream.write(
            json.dumps(protocol.jsonable(rows[0]), allow_nan=False) + "\n"
        )
        trace_stream.flush()
        observe(session, detailed, {})
        for tick in range(1, CONTROLS + 1):
            if time.perf_counter() - process_began >= wall_limit_s - 10:
                reason = "resource_guard"
                break
            event = protocol.advance_control(session, "main", tick, clock, journal)
            try:
                row = capture_control(session)
                if (
                    len(native) != 50 * tick
                    or session.network.tick != 20 * tick
                    or abs(session.body.physics_time() - 0.005 * tick) > 1e-8
                ):
                    raise ValueError("Incomplete 20-neural/50-physics control")
                rows.append(row)
                trace_stream.write(
                    json.dumps(protocol.jsonable(row), allow_nan=False) + "\n"
                )
                trace_stream.flush()
                observe(session, detailed, {})
                protocol.record_readout(journal, "main", tick, event, complete=True)
            except Exception as error:
                protocol.record_readout(
                    journal, "main", tick, event, complete=False, error=error
                )
                raise
            if session.fault or session.body.fault:
                reason, health_stop = "native_health_guard", True
                break
            if tick == 99:
                protocol.e121.save_startup_raw(out, rows, startup)
                screen = protocol.strict_startup(rows, native, startup, limits)
                protocol.write_json(out / "startup-screen.json", screen)
                if not screen["strict_E141_startup_screen"]:
                    reason = "startup_guard"
                    break
            if tick % 100 == 0:
                ledger["completed_controls"] = tick
                ledger["clock"] = {
                    "neural_s": clock.neural_s,
                    "physical_s": clock.physical_s,
                    "valid": clock.clock_valid,
                }
                persist()
    except protocol.NativeHealthStop as error:
        reason, health_stop, primary = "native_health_guard", True, error
    except ResourceGuard as error:
        reason, primary = "resource_guard", error
    except Exception as error:
        reason, primary = "execution_error", error
    finally:

        def preserve(label, operation):
            try:
                operation()
            except Exception as error:
                raw_errors.append({"artifact": label, "error": repr(error)})

        if journal is not None:
            preserve("journal_close", journal.close)
        if trace_stream is not None:
            preserve("trace_stream_close", trace_stream.close)
        if session is not None:
            preserve(
                "final_checkpoint",
                lambda: StateStore.save(out / "final", session.snapshot()),
            )
        preserve(
            "trace",
            lambda: protocol.write_json(
                out / "trace.json",
                {
                    "schema": "flylab.m3.e181.native-timing-trace.v1",
                    "controller": "E150 fixed monotone recruitment with E181 bank timing tilt",
                    "graph_hash": graph.hash,
                    "model_hash": session.network.identity
                    if session
                    else neutral["network"]["model_hash"],
                    "body_hash": session.body.model_hash if session else None,
                    "adapter_hash": session.adapter.identity if session else None,
                    "parameters": neutral["parameters"],
                    "samples": rows,
                },
            ),
        )
        preserve(
            "native_contact",
            lambda: np.savez_compressed(
                out / "native-contact.npz", **protocol.native_arrays(native)
            ),
        )
        preserve(
            "native_recruitment_state",
            lambda: e150.save_native_state(
                out / "native-recruitment-state.npz", native_state
            ),
        )
        preserve(
            "clock_events",
            lambda: protocol.write_json(out / "clock-events.json", clock.events),
        )
        if restore is not None:
            preserve("observer_restore", restore)
        if session is not None:
            preserve("session_close", session.close)
        if primary is not None:
            protocol.write_json(
                out / "primary-error.json",
                {"type": type(primary).__name__, "message": str(primary)},
            )
        ledger["completed_controls"] = max(0, len(rows) - 1)
        ledger["clock"] = {
            "neural_s": clock.neural_s,
            "physical_s": clock.physical_s,
            "valid": clock.clock_valid,
            "actual_costs_by_owner": clock.by_owner(),
        }
        ledger["stop_reason"] = reason
        preserve("ledger_final", persist)
    screen_path = out / "startup-screen.json"
    startup_ok = (
        json.loads(screen_path.read_text(encoding="utf-8"))[
            "strict_E141_startup_screen"
        ]
        if screen_path.exists()
        else None
    )
    if (
        reason == "duration"
        and not raw_errors
        and time.perf_counter() - process_began < wall_limit_s - 3
    ):
        try:
            model = load_model(out / "trace.json")
            objective = forward_deficit(model["speed"], model["window"])
            quality, _phi, _events = analyze(model)
            protocol.write_json(out / "biological-gait-3s.json", quality)
            physical = e180.physical_summary(
                rows, protocol.native_arrays(native), quality
            )
            protocol.write_json(out / "physical-3s.json", physical)
        except Exception as error:
            analysis_errors.append(
                {"artifact": "biological_gait_3s", "error": repr(error)}
            )
    elif reason == "duration":
        reason = "resource_guard"
    outcome = summarize_arm(
        reason,
        health_stop,
        len(rows) - 1,
        len(native),
        clock.neural_s,
        clock.physical_s,
        clock.clock_valid,
        startup_ok,
        raw_errors,
        analysis_errors,
    )
    outcome.update(
        theta=list(theta),
        objective_mm_s=objective
        if reason == "duration" and not analysis_errors and not raw_errors
        else None,
        freeze_sha256=gate.freeze_sha256,
        source_review_sha256=gate.source_review_sha256,
        body_wall_s=time.perf_counter() - began,
        process_wall_s=time.perf_counter() - process_began,
        advance_journal_rows=journal.rows if journal else 0,
        primary_error=repr(primary) if primary else None,
    )
    if reason == "duration" and not analysis_errors and not raw_errors:
        outcome["physical_3s"] = physical
    raw_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    outcome["raw_bytes"] = raw_bytes
    if raw_bytes > gate.freeze["maximum_raw_bytes"]:
        outcome["technical_failure"] = True
        outcome["validity"] = "incomplete"
        outcome["raw_persistence_errors"].append(
            {
                "artifact": "output_limit",
                "error": "Raw bytes exceeded frozen 1 GiB limit",
            }
        )
    if time.perf_counter() - process_began > wall_limit_s:
        outcome["validity"] = "incomplete"
        outcome["stop_reason"] = "resource_guard"
        outcome["resource_limit_exceeded"] = True
        outcome["technical_failure"] = True
    persist()
    protocol.write_json(out / "result.json", outcome)
    return outcome


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _directory_bytes(path: Path) -> int:
    return (
        sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        if path.exists()
        else 0
    )


def _require_capacity(root: Path) -> None:
    if shutil.disk_usage(root.parent).free < FREE_BYTES:
        raise ResourceGuard("E181 requires 20 GiB free before a driver")
    if _directory_bytes(root) > CAMPAIGN_BYTES:
        raise ResourceGuard("E181 campaign exceeded 14 GiB")


def compare_baseline_prefix(candidate: Path, reference: Path) -> dict[str, Any]:
    """Direct E180/E181 zero-arm raw comparison before any coordinate arm."""
    left = json.loads((candidate / "trace.json").read_text(encoding="utf-8"))["samples"]
    right = json.loads((reference / "trace.json").read_text(encoding="utf-8"))[
        "samples"
    ]
    if len(left) != 601 or len(right) != 601:
        raise ValueError("Baseline reference requires 601 control rows")
    fields = (
        "control_tick",
        "command_interval",
        "joint_targets_rad",
        "tendon_inputs",
        "qpos",
        "qvel",
        "ctrl",
        "upright",
    )
    mismatches = []
    for index, (new, old) in enumerate(zip(left, right, strict=True)):
        for key in fields:
            if new.get(key) != old.get(key):
                mismatches.append(f"row{index}.{key}")
        for key in ("position", "basis", "contactsBW"):
            if new.get("body", {}).get(key) != old.get("body", {}).get(key):
                mismatches.append(f"row{index}.body.{key}")
        if not math.isclose(new["time_s"], old["time_s"], rel_tol=0, abs_tol=2e-8):
            mismatches.append(f"row{index}.time_s")
    with (
        np.load(candidate / "native-recruitment-state.npz", allow_pickle=False) as new,
        np.load(reference / "native-recruitment-state.npz", allow_pickle=False) as old,
    ):
        state_fields = (
            "control_tick",
            "network_tick",
            "complete",
            "bank",
            "bank_available",
            "activation",
            "activation_available",
            "original_activation",
            "original_activation_available",
            "motor_rate",
            "motor_rate_available",
        )
        for key in state_fields:
            if (
                key not in new
                or key not in old
                or not np.array_equal(new[key], old[key])
            ):
                mismatches.append("state." + key)
    with (
        np.load(candidate / "native-contact.npz", allow_pickle=False) as new,
        np.load(reference / "native-contact.npz", allow_pickle=False) as old,
    ):
        if set(new.files) != set(old.files):
            mismatches.append("native.keys")
        else:
            for key in new.files:
                if len(new[key]) != 30000 or len(old[key]) != 30000:
                    mismatches.append("native.length." + key)
                elif key == "time_s":
                    if not np.allclose(new[key], old[key], atol=2e-8, rtol=0):
                        mismatches.append("native.time_s")
                elif not (
                    np.array_equal(new[key], old[key], equal_nan=True)
                    if np.issubdtype(new[key].dtype, np.inexact)
                    else np.array_equal(new[key], old[key])
                ):
                    mismatches.append("native." + key)
    return {
        "reference": str(reference.relative_to(ROOT)),
        "candidate": str(candidate.relative_to(ROOT)),
        "rows": 601,
        "native_samples": 30000,
        "mismatches": mismatches[:100],
        "mismatch_count": len(mismatches),
        "exact_reproduction": not mismatches,
    }


def _load_startup(freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = protocol.load_npz(_safe(freeze["startup_path"]))
    return {
        "target_command": raw["target_command"],
        "target_tendon": raw["target_tendon"],
    }


def driver_ancestor(request: dict[str, Any], *, child_pid: int | None = None) -> bool:
    """Accept the live driver across the Windows venv redirector layer."""
    driver_pid = request.get("parent_pid")
    driver_created = request.get("parent_created_s")
    if type(driver_pid) is not int or not _finite_metric(driver_created):
        return False
    try:
        child = psutil.Process(os.getpid() if child_pid is None else child_pid)
        return any(
            parent.pid == driver_pid
            and abs(parent.create_time() - driver_created) <= 0.001
            for parent in child.parents()
        )
    except (psutil.Error, OSError):
        return False


def _owned_children(root_pid: int, known: dict[int, float]) -> None:
    """Keep process birth times so a recycled PID cannot become owned."""
    try:
        root = psutil.Process(root_pid)
        if root_pid not in known or abs(root.create_time() - known[root_pid]) > 0.001:
            return
        for child in root.children(recursive=True):
            known.setdefault(child.pid, child.create_time())
    except (psutil.Error, OSError):
        return


def _live_owned(known: dict[int, float]) -> list[psutil.Process]:
    alive = []
    for pid, created in known.items():
        try:
            process = psutil.Process(pid)
            if (
                abs(process.create_time() - created) <= 0.001
                and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                alive.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, OSError):
            # Uninspectable owned processes are conservatively not cleared.
            raise ResourceGuard("Owned child process could not be inspected") from None
    return alive


def _terminate_owned_tree(root_pid: int, known: dict[int, float]) -> dict[str, Any]:
    _owned_children(root_pid, known)
    errors = []
    try:
        alive = _live_owned(known)
    except Exception as error:
        return {
            "terminated_pids": [],
            "alive_pids": list(known),
            "errors": [repr(error)],
        }
    # Descendants first, redirector root last.
    ordered = sorted(alive, key=lambda process: process.pid == root_pid)
    for process in ordered:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError) as error:
            errors.append(f"terminate {process.pid}: {error!r}")
    _gone, survivors = psutil.wait_procs(ordered, timeout=2)
    for process in survivors:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError) as error:
            errors.append(f"kill {process.pid}: {error!r}")
    psutil.wait_procs(survivors, timeout=2)
    try:
        remaining = [process.pid for process in _live_owned(known)]
    except Exception as error:
        remaining = list(known)
        errors.append(repr(error))
    return {
        "terminated_pids": [process.pid for process in ordered],
        "alive_pids": remaining,
        "errors": errors,
    }


def supervise_command(
    command: list[str], request_path: Path, *, timeout_s: float = CHILD_EXTERNAL_S
) -> dict[str, Any]:
    """Persist a live PID, streams, and one final receipt on every launch path."""
    request_path = Path(request_path)
    started_path = request_path.with_name(request_path.stem + "-started.json")
    final_path = request_path.with_name(request_path.stem + "-supervisor.json")
    stdout_path = request_path.with_name(request_path.stem + "-stdout.txt")
    stderr_path = request_path.with_name(request_path.stem + "-stderr.txt")
    launched = time.perf_counter()
    process = None
    known: dict[int, float] = {}
    status = "spawn_error"
    exit_code = None
    error_text = None
    timed_out = False
    termination = None
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True
            )
            root = psutil.Process(process.pid)
            known[root.pid] = root.create_time()
            _write_new(
                started_path,
                {
                    "schema": "flylab.m3.e181.child-started.v1",
                    "pid": process.pid,
                    "pid_created_s": known[process.pid],
                    "request_sha256": _sha256(request_path),
                    "started": True,
                    "started_wall_s": time.perf_counter() - launched,
                },
            )
            status = "running"
            while True:
                _owned_children(process.pid, known)
                remaining = timeout_s - (time.perf_counter() - launched)
                if remaining <= 0:
                    timed_out = True
                    status = "timed_out"
                    break
                try:
                    exit_code = process.wait(timeout=min(0.5, remaining))
                    status = "completed" if exit_code == 0 else "child_nonzero"
                    break
                except subprocess.TimeoutExpired:
                    continue
            _owned_children(process.pid, known)
            if _live_owned(known):
                if status == "completed":
                    status = "unexpected_live_descendant"
                termination = _terminate_owned_tree(process.pid, known)
                if termination["alive_pids"] or termination["errors"]:
                    status = "termination_uncertain"
        except Exception as error:
            error_text = repr(error)
            status = "supervisor_error" if process is not None else "spawn_error"
        finally:
            if process is not None:
                _owned_children(process.pid, known)
                if termination is None:
                    termination = _terminate_owned_tree(process.pid, known)
                if termination["alive_pids"] or termination["errors"]:
                    status = "termination_uncertain"
                try:
                    process.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError):
                    status = "termination_uncertain"
            elapsed = time.perf_counter() - launched
            receipt = {
                "schema": "flylab.m3.e181.child-supervisor.v1",
                "request_sha256": _sha256(request_path),
                "attempted": True,
                "started": process is not None,
                "completed": status == "completed",
                "timed_out": timed_out,
                "status": status,
                "pid": process.pid if process else None,
                "owned_pid_birth_s": known,
                "exit_code": exit_code,
                "child_wall_s": elapsed if process else None,
                "accounted_wall_s": (
                    max(elapsed, SUPERVISOR_S)
                    if process is not None and status != "completed"
                    else elapsed
                    if process is not None
                    else None
                ),
                "cost_exact": process is not None and status == "completed",
                "spawn_wall_s": elapsed if process is None else None,
                "error": error_text,
                "termination": termination,
                "stdout_path": stdout_path.name,
                "stderr_path": stderr_path.name,
                "child_external_limit_s": timeout_s,
                "supervisor_limit_s": SUPERVISOR_S,
                "automatic_retries": 0,
            }
            _write_new(final_path, receipt)
    return receipt


def _run_child(
    args: argparse.Namespace, gate: ValidatedGate, request: dict[str, Any]
) -> dict[str, Any]:
    freeze = gate.freeze
    root = _safe(OUTPUT_DIRECTORY)
    if (
        request.get("schema") != "flylab.m3.e181.child-request.v1"
        or request.get("freeze_sha256") != gate.freeze_sha256
        or request.get("source_review_sha256") != gate.source_review_sha256
    ):
        raise ValueError("Frozen child request identity changed")
    if not driver_ancestor(request) or request.get("mode") != gate.mode:
        raise ValueError("E181 child requires its live driver parent")
    arm = request.get("arm")
    if gate.mode == "baseline":
        expected_request = root / "baseline-request.json"
        expected_output = root / "baseline"
        if arm != "baseline" or request.get("theta") != [0.0] * 6:
            raise ValueError("E181 baseline child must use zero theta")
    else:
        if arm not in ("plus", "minus"):
            raise ValueError("E181 axis child requires a signed direction")
        expected_request = root / f"axis-{gate.mode}" / f"{arm}-request.json"
        expected_output = root / f"axis-{gate.mode}" / arm
        center = tuple(request.get("center_theta", []))
        if tuple(request.get("theta", [])) != propose_axis(
            center, gate.mode, arm, request.get("step")
        ):
            raise ValueError("E181 child theta differs from frozen coordinate proposal")
        intent = json.loads(
            (expected_request.parent / "intent.json").read_text(encoding="utf-8")
        )
        if request.get("intent_sha256") != _sha256(
            expected_request.parent / "intent.json"
        ):
            raise ValueError("E181 axis child intent hash changed")
        if request.get("theta") != intent["preflight"][arm]["selected_theta"]:
            raise ValueError("E181 child theta was not saved-startup selected")
    if (
        Path(args.child_request).resolve() != expected_request.resolve()
        or _sha256(expected_request) != args.child_request_sha256
    ):
        raise ValueError("Driver-created exact request hash required")
    if expected_output.exists():
        raise FileExistsError("E181 arm output is create-only")
    _require_capacity(root)
    thread_limits, _threads = protocol.joint_fit.one_cpu_thread()
    try:
        startup = _load_startup(freeze)
        graph = GraphStore.load(_safe(freeze["graph_path"]))
        if graph.hash != freeze["graph_hash"]:
            raise ValueError("Frozen graph identity changed")
        neutral = StateStore.load(_safe(freeze["neutral_checkpoint"]))
        result = run_arm(
            graph,
            neutral,
            tuple(request["theta"]),
            expected_output,
            startup=startup,
            limits=freeze["startup_limits"],
            wall_limit_s=CHILD_INTERNAL_S,
            gate=gate,
            process_began=_ENTRY_STARTED,
        )
    finally:
        thread_limits.restore_original_limits()
    if result["technical_failure"]:
        raise RuntimeError("E181 child technical failure; raw preserved")
    return result


def _launch_child(
    args: argparse.Namespace,
    gate: ValidatedGate,
    request: dict[str, Any],
    request_path: Path,
    *,
    began: float,
) -> dict[str, Any]:
    _write_new(request_path, request)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--child-request",
        str(request_path),
        "--child-request-sha256",
        _sha256(request_path),
        "--freeze",
        str(args.freeze),
        "--freeze-sha256",
        args.freeze_sha256,
        "--source-review",
        str(args.source_review),
        "--source-review-sha256",
        args.source_review_sha256,
    ]
    if gate.mode == "baseline":
        command.append("--baseline")
    else:
        command.extend(
            (
                "--axis",
                gate.mode,
                "--baseline-review",
                str(args.baseline_review),
                "--baseline-review-sha256",
                args.baseline_review_sha256,
            )
        )
        if gate.mode != LEGS[0]:
            command.extend(("--previous-axis-sha256", args.previous_axis_sha256))
    receipt = supervise_command(command, request_path)
    if receipt["status"] != "completed" or receipt["child_wall_s"] > SUPERVISOR_S:
        raise ResourceGuard(
            "E181 child failed or exceeded supervisor bound; raw preserved"
        )
    result_path = (
        request_path.with_name(
            "baseline" if gate.mode == "baseline" else request["arm"]
        )
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("freeze_sha256") != gate.freeze_sha256
        or result.get("source_review_sha256") != gate.source_review_sha256
    ):
        raise ValueError("E181 child result provenance changed")
    return result


def _attempt_accounting(request_path: Path) -> dict[str, Any]:
    """Count known starts exactly and reserve unknown starts conservatively."""
    receipt_path = request_path.with_name(request_path.stem + "-supervisor.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        started = receipt.get("started")
        if type(started) is not bool:
            raise ValueError("Supervisor receipt omitted start status")
        return {
            "started": started,
            "arms_exact": int(started),
            "arms_upper_bound": int(started),
            "accounted_wall_s": receipt["accounted_wall_s"] if started else 0.0,
            "cost_exact": receipt["cost_exact"] if started else True,
            "status": receipt["status"],
            "receipt_sha256": _sha256(receipt_path),
        }
    if request_path.is_file():
        return {
            "started": None,
            "arms_exact": None,
            "arms_upper_bound": 1,
            "accounted_wall_s": SUPERVISOR_S,
            "cost_exact": False,
            "status": "missing_supervisor_receipt",
            "receipt_sha256": None,
        }
    return {
        "started": False,
        "arms_exact": 0,
        "arms_upper_bound": 0,
        "accounted_wall_s": 0.0,
        "cost_exact": True,
        "status": "not_attempted",
        "receipt_sha256": None,
    }


def _baseline_driver(args: argparse.Namespace, gate: ValidatedGate) -> dict[str, Any]:
    root = _safe(OUTPUT_DIRECTORY)
    if root.exists():
        raise FileExistsError("E181 baseline campaign is create-only")
    _require_capacity(root)
    root.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    request = {
        "schema": "flylab.m3.e181.child-request.v1",
        "mode": "baseline",
        "arm": "baseline",
        "parent_pid": os.getpid(),
        "parent_created_s": psutil.Process(os.getpid()).create_time(),
        "theta": [0.0] * 6,
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
    }
    result = None
    error = None
    try:
        result = _launch_child(
            args, gate, request, root / "baseline-request.json", began=began
        )
        if result.get("validity") != "valid_3s_diagnostic":
            raise ValueError(
                "E181 zero-arm did not complete valid 3s native diagnostic"
            )
        comparison = compare_baseline_prefix(
            root / "baseline", _safe(gate.freeze["reference_directory"])
        )
        _write_new(root / "baseline-comparison.json", comparison)
        if not comparison["exact_reproduction"]:
            raise ValueError("E181 baseline failed direct E180 raw-prefix reproduction")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    accounting = _attempt_accounting(root / "baseline-request.json")
    if accounting["status"] != "completed" and error is None:
        error = {"type": "ResourceGuard", "message": "Baseline child did not complete"}
    receipt = {
        "schema": "flylab.m3.e181.baseline-driver.v1",
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
        "baseline_result_sha256": _sha256(root / "baseline" / "result.json")
        if (root / "baseline" / "result.json").is_file()
        else None,
        "comparison_sha256": _sha256(root / "baseline-comparison.json")
        if (root / "baseline-comparison.json").is_file()
        else None,
        "native_arms_executed": accounting["arms_exact"],
        "native_arms_upper_bound": accounting["arms_upper_bound"],
        "native_supervisor_wall_s_cumulative": accounting["accounted_wall_s"],
        "native_cost_exact": accounting["cost_exact"],
        "supervisor_status": accounting["status"],
        "startup_wall_s_cumulative": 0.0,
        "driver_wall_s": time.perf_counter() - began,
        "campaign_raw_bytes": _directory_bytes(root),
        "error": error,
        "parent_review_required": True,
        "candidate_adopted": False,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
    }
    _write_new(root / "baseline-driver.json", receipt)
    if error is not None:
        raise RuntimeError(
            "E181 baseline driver failed; raw and receipts preserved: "
            + error["message"]
        )
    return receipt


def _saved_startup_callback(
    gate: ValidatedGate,
) -> Callable[[tuple[float, ...]], dict[str, Any]]:
    root = _safe(OUTPUT_DIRECTORY)
    neutral = StateStore.load(_safe(gate.freeze["neutral_checkpoint"]))
    _neutral(neutral)
    patch = neutral["monotone_recruitment_patch"]
    adapter = neutral["adapter"]
    calibration = adapter["calibration"]
    legs = tuple(group["leg"] for group in adapter["groups"])
    ltm_raw = np.asarray(calibration["ltm"], np.int64)
    ltm = np.asarray(
        [
            next(int(index) for index in ltm_raw if legs[int(index)] == leg)
            for leg in LEGS
        ],
        np.int64,
    )
    with np.load(
        root / "baseline" / "native-recruitment-state.npz", allow_pickle=False
    ) as state:
        if (
            not np.array_equal(state["control_tick"][:100], np.arange(100))
            or not state["bank_available"][:100].all()
            or not state["original_activation_available"][:100].all()
            or not state["complete"][:100].all()
        ):
            raise ValueError("Incomplete E181 baseline saved bank startup prefix")
        bank = state["bank"][:100].copy()
        original_active = state["original_activation"][:100].copy()
    startup = _load_startup(gate.freeze)

    def evaluate(theta: tuple[float, ...]) -> dict[str, Any]:
        return saved_startup_screen(
            theta,
            patch["slopes"],
            patch["tau"],
            legs,
            patch["fixed_muscles"],
            patch["kappa"],
            patch["membership"],
            bank,
            original_active,
            calibration["moment"],
            calibration["rest"],
            ltm,
            startup,
        )

    return evaluate


def _axis_driver(args: argparse.Namespace, gate: ValidatedGate) -> dict[str, Any]:
    root = _safe(OUTPUT_DIRECTORY)
    axis = gate.mode
    out = root / f"axis-{axis}"
    if out.exists():
        raise FileExistsError("E181 axis output is create-only")
    if not (root / "baseline-comparison.json").is_file() or not json.loads(
        (root / "baseline-comparison.json").read_text(encoding="utf-8")
    ).get("exact_reproduction"):
        raise ValueError("E181 exact zero-arm baseline reproduction required")
    baseline = json.loads(
        (root / "baseline" / "result.json").read_text(encoding="utf-8")
    )
    if (
        baseline.get("validity") != "valid_3s_diagnostic"
        or baseline.get("theta") != [0.0] * 6
    ):
        raise ValueError("E181 baseline result invalid")
    if axis == LEGS[0]:
        center_theta = (0.0,) * 6
        center_result = baseline
        prior_arms = 1
        baseline_driver = json.loads(
            (root / "baseline-driver.json").read_text(encoding="utf-8")
        )
        if (
            baseline_driver.get("error") is not None
            or baseline_driver.get("native_arms_executed") != 1
            or baseline_driver.get("native_cost_exact") is not True
        ):
            raise ValueError("E181 baseline execution accounting is incomplete")
        prior_supervisor_s = baseline_driver["native_supervisor_wall_s_cumulative"]
        prior_startup_s = 0.0
    else:
        previous = json.loads(
            (root / f"axis-{LEGS[LEGS.index(axis) - 1]}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            previous.get("technical_failure")
            or previous.get("axis") != LEGS[LEGS.index(axis) - 1]
            or previous.get("native_cost_exact") is not True
        ):
            raise ValueError("Previous E181 axis did not finish cleanly")
        center_theta = tuple(previous["selection"]["selected_theta"])
        center_result = previous["selected_result"]
        prior_arms = previous["native_arms_executed_cumulative"]
        prior_supervisor_s = previous["native_supervisor_wall_s_cumulative"]
        prior_startup_s = previous["startup_wall_s_cumulative"]
    if prior_arms > 11:
        raise ResourceGuard("E181 baseline plus twelve candidate-arm limit")
    _require_capacity(root)
    began = _ENTRY_STARTED
    out.mkdir(parents=False, exist_ok=False)
    preflight = {}
    preflight_error = None
    startup_began = time.perf_counter()
    try:
        thread_limits, _threads = protocol.joint_fit.one_cpu_thread()
        try:
            startup_callback = _saved_startup_callback(gate)
            seen = {center_theta}
            for direction in ("plus", "minus"):
                try:
                    preflight[direction] = preflight_direction(
                        center_theta, axis, direction, startup_callback, seen
                    )
                except PreflightError as exc:
                    preflight[direction] = {
                        "axis": axis,
                        "direction": direction,
                        "attempts": exc.attempts,
                        "selected_theta": None,
                    }
                    raise
        finally:
            thread_limits.restore_original_limits()
    except Exception as exc:
        preflight_error = {"type": type(exc).__name__, "message": str(exc)}
    startup_elapsed = time.perf_counter() - startup_began
    if prior_startup_s + startup_elapsed > STARTUP_S:
        preflight_error = {
            "type": "ResourceGuard",
            "message": "Cumulative saved-startup work exceeded 30 seconds",
        }
    intent = {
        "schema": "flylab.m3.e181.axis-intent.v1",
        "axis": axis,
        "center_theta": list(center_theta),
        "center_objective_mm_s": center_result["objective_mm_s"],
        "baseline_result_sha256": _sha256(root / "baseline" / "result.json"),
        "previous_axis_sha256": args.previous_axis_sha256,
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
        "preflight": preflight,
        "native_arms_planned": sum(
            value["selected_theta"] is not None for value in preflight.values()
        ),
        "startup_only_attempts": sum(
            len(value["attempts"]) for value in preflight.values()
        ),
        "candidate_adopted": False,
    }
    intent["startup_wall_s"] = startup_elapsed
    intent["startup_wall_s_cumulative"] = prior_startup_s + startup_elapsed
    intent["preflight_error"] = preflight_error
    _write_new(out / "intent.json", intent)
    directions: dict[str, Any] = {}
    error = preflight_error
    for direction in ("plus", "minus"):
        if error is not None:
            break
        proposal = preflight[direction]
        if proposal["selected_theta"] is None:
            continue
        attempts_so_far = [
            _attempt_accounting(out / f"{signed}-request.json")
            for signed in ("plus", "minus")
        ]
        current_accounted = sum(row["accounted_wall_s"] for row in attempts_so_far)
        if (
            time.perf_counter() - began + SUPERVISOR_S > AXIS_S
            or 3533.5252058999613
            + prior_supervisor_s
            + current_accounted
            + prior_startup_s
            + startup_elapsed
            + SUPERVISOR_S
            > 4994
        ):
            error = {
                "type": "ResourceGuard",
                "message": "Whole axis or cumulative native reserve exhausted before child",
            }
            break
        attempt = proposal["attempts"][-1]
        request = {
            "schema": "flylab.m3.e181.child-request.v1",
            "mode": axis,
            "arm": direction,
            "parent_pid": os.getpid(),
            "parent_created_s": psutil.Process(os.getpid()).create_time(),
            "theta": proposal["selected_theta"],
            "center_theta": list(center_theta),
            "step": attempt["step"],
            "freeze_sha256": gate.freeze_sha256,
            "source_review_sha256": gate.source_review_sha256,
            "intent_sha256": _sha256(out / "intent.json"),
        }
        try:
            arm_result = _launch_child(
                args, gate, request, out / f"{direction}-request.json", began=began
            )
            directions[direction] = {
                "theta": proposal["selected_theta"],
                "result": arm_result,
                "result_sha256": _sha256(out / direction / "result.json"),
            }
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "direction": direction,
            }
            break
    selection = (
        select_axis(center_theta, center_result, directions, baseline)
        if error is None
        else None
    )
    selected_result = (
        directions[selection["selected_direction"]]["result"]
        if selection and selection["selected_direction"]
        else center_result
    )
    attempts = {
        direction: _attempt_accounting(out / f"{direction}-request.json")
        for direction in ("plus", "minus")
    }
    exact_axis_arms = (
        sum(row["arms_exact"] for row in attempts.values())
        if all(row["arms_exact"] is not None for row in attempts.values())
        else None
    )
    cumulative = prior_arms + exact_axis_arms if exact_axis_arms is not None else None
    upper_bound = prior_arms + sum(row["arms_upper_bound"] for row in attempts.values())
    supervisor_s_axis = sum(row["accounted_wall_s"] for row in attempts.values())
    supervisor_s_cumulative = prior_supervisor_s + supervisor_s_axis
    receipt = {
        "schema": "flylab.m3.e181.axis-result.v1",
        "axis": axis,
        "freeze_sha256": gate.freeze_sha256,
        "source_review_sha256": gate.source_review_sha256,
        "baseline_review_sha256": args.baseline_review_sha256,
        "previous_axis_sha256": args.previous_axis_sha256,
        "intent_sha256": _sha256(out / "intent.json"),
        "arm_result_sha256": {
            key: value["result_sha256"] for key, value in directions.items()
        },
        "selection": selection,
        "selected_result": selected_result,
        "native_arms_executed_cumulative": cumulative,
        "native_arms_upper_bound_cumulative": upper_bound,
        "native_arms_executed_axis": exact_axis_arms,
        "attempt_accounting": attempts,
        "native_cost_exact": all(row["cost_exact"] for row in attempts.values()),
        "objective_evaluations_completed_axis": sum(
            value["result"].get("objective_mm_s") is not None
            for value in directions.values()
        ),
        "startup_only_attempts_axis": intent["startup_only_attempts"],
        "startup_wall_s_cumulative": prior_startup_s + startup_elapsed,
        "native_supervisor_wall_s_cumulative": supervisor_s_cumulative,
        "axis_wall_s": time.perf_counter() - began,
        "campaign_raw_bytes": _directory_bytes(root),
        "technical_failure": error is not None,
        "error": error,
        "candidate_adopted": False,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
        "parent_review_required": True,
    }
    if (
        receipt["axis_wall_s"] > AXIS_S
        or receipt["campaign_raw_bytes"] > CAMPAIGN_BYTES
        or upper_bound > 13
        or 3533.5252058999613
        + supervisor_s_cumulative
        + prior_startup_s
        + startup_elapsed
        > 4994
    ):
        receipt["technical_failure"] = True
        receipt["error"] = {
            "type": "ResourceGuard",
            "message": "E181 axis or campaign limit exceeded",
        }
    _write_new(out / "result.json", receipt)
    if receipt["technical_failure"]:
        raise RuntimeError("E181 axis stopped after technical failure; raw preserved")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--axis", choices=LEGS)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--baseline-review", type=Path)
    parser.add_argument("--baseline-review-sha256")
    parser.add_argument("--previous-axis-sha256")
    parser.add_argument("--child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected_mode = "baseline" if args.baseline else args.axis
    gate = verified_inputs(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        selected_mode,
        args.baseline_review,
        args.baseline_review_sha256,
        args.previous_axis_sha256,
    )
    if args.child_request is not None:
        if args.child_request_sha256 is None:
            raise ValueError("Exact driver-created child request hash required")
        request = json.loads(args.child_request.read_text(encoding="utf-8"))
        result = _run_child(args, gate, request)
    elif args.child_request_sha256 is not None:
        raise ValueError("Child hash without request is invalid")
    elif args.baseline:
        result = _baseline_driver(args, gate)
    else:
        result = _axis_driver(args, gate)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
