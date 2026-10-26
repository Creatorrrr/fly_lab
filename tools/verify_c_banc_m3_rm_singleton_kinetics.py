"""E158 one-shot RM singleton muscle-kinetics body diagnostic.

The original execution is deliberately gated by an exact frozen-input and
two-source review receipt.  All arithmetic before body construction is exposed
as pure NumPy functions so it can be reviewed without loading a StateStore or a
MuJoCo model.
"""

# Broad catches below preserve a primary failure while independently saving
# partial scientific work and cleanup failures.
# ruff: noqa: BLE001  # thread environment must precede NumPy import

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "f44865a0e79b94a0686c2493e715e52c94a20727d3fe17b49392c704aaa9541c"
DESIGN_SHA = "0e58f38fca660c1d5c1e0180f4816ba9d6ebf554bb9aebb6a8df87c13d416118"
OWNED = (
    "tools/verify_c_banc_m3_rm_singleton_kinetics.py",
    "tests/test_c_banc_rm_singleton_kinetics.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
TENDONS = tuple(f"{leg}_tarsus" for leg in LEGS)
CONTROL_DT = 0.005
PHYSICS_DT = 0.0001
N_CONTROLS = 2000
N_NATIVE = 100000
CLOCK_ATOL = 2e-8
RECON_ATOL = 1e-10


class NativeHealthStop(RuntimeError):
    """A measured finite physical health rejection, not a technical failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(relative: str) -> Path:
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("E158 repository-relative path required")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def exact_hash_map(mapping: dict[str, str], label: str) -> None:
    require(isinstance(mapping, dict), f"E158 {label} hash map required")
    for name, digest in mapping.items():
        require(sha256(safe_path(name)) == digest, f"E158 {label} changed: {name}")


def verified_inputs(
    freeze_path: str | Path,
    freeze_sha256: str,
    review_path: str | Path,
    review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every immutable byte and the review gate before source data use."""
    require(freeze_sha256 == FREEZE_SHA, "Exact E158 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E158 freeze bytes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    fixed = {
        "schema": "flylab.m3.e160.rm-singleton-initial-boundary-recovery.v1",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "body_hash": "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "original_runs_before_gate": 0,
        "candidate_values_seen_before_freeze": True,
    }
    require(all(freeze.get(k) == v for k, v in fixed.items()), "E158 contract changed")
    require(
        freeze.get("intervention")
        == {
            "muscle": 67,
            "motor_column": 364,
            "graph_index": 144018,
            "joint_row": 29,
            "tau_s": 0.04,
            "branch": 2,
            "gain": 0.02,
        },
        "E158 intervention changed",
    )
    require(
        freeze.get("counts")
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_constructor_max": 1,
            "shadow_mjdata_max": 1,
            "settling_steps_max": 2000,
            "scientific_steps_max": 100000,
            "shadow_forward_calls_max": 100000,
            "neural_steps": 0,
            "fit_calls": 0,
        },
        "E158 execution counts changed",
    )
    require(
        freeze.get("resource_limits")
        == {
            "internal_s": 260,
            "external_s": 280,
            "supervisor_s": 295,
            "raw_bytes": 1073741824,
            "free_bytes": 4294967296,
            "cpu_threads": 1,
            "gpu_count": 0,
        },
        "E158 resource limits changed",
    )
    require(len(freeze.get("input_sha256", {})) == 199, "E160 input count changed")
    require(
        len(freeze.get("reused_source_sha256", {})) == 23,
        "E158 reused-source count changed",
    )
    require(
        len(freeze.get("preservation_sha256", {})) == 3122,
        "E160 preservation count changed",
    )
    require(sha256(review_path) == review_sha256, "Exact E158 source review required")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    approved = (
        review.get("approved") is True
        or review.get("accepts_single_rm_singleton_kinetics") is True
    )
    require(
        approved
        and review.get("freeze_sha256") == FREEZE_SHA
        and review.get("blocking_defects", []) == []
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E158 two-source gate required",
    )
    for name, digest in review["source_sha256"].items():
        require(
            sha256(safe_path(name)) == digest, "Reviewed E158 source changed: " + name
        )
    for key in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        exact_hash_map(freeze[key], key)
    return freeze, review


def ema_from_saved_rate(
    source_rate: Any,
    initial: float,
    *,
    dt: float = CONTROL_DT,
    tau: float = 0.04,
) -> np.ndarray:
    """Return the causal EMA at saved boundaries; index zero is the saved state."""
    rate = np.asarray(source_rate, dtype=np.float64)
    require(rate.ndim == 1 and rate.size >= 1, "One-dimensional saved rate required")
    require(
        np.isfinite(rate).all() and math.isfinite(initial), "Finite EMA inputs required"
    )
    require(
        dt > 0 and tau > 0 and math.isfinite(dt + tau),
        "Positive finite EMA times required",
    )
    result = np.empty_like(rate)
    result[0] = initial
    alpha = -math.expm1(-dt / tau)
    for index in range(1, len(rate)):
        result[index] = result[index - 1] + alpha * (rate[index] - result[index - 1])
    return result


def build_candidate_commands(
    *,
    bank: Any,
    motor_rate: Any,
    baseline_activation: Any,
    baseline_command: Any,
    moment: Any,
    rest: Any,
    muscle: int = 67,
    motor_column: int = 364,
    branch: int = 2,
    joint_row: int = 29,
    gain: float = 0.02,
    atol: float = RECON_ATOL,
    diagnostics: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, np.ndarray | float]:
    """Verify the frozen transfer and replace only RM joint row 29."""
    bank = np.asarray(bank)
    rate = np.asarray(motor_rate)
    activation = np.asarray(baseline_activation)
    command = np.asarray(baseline_command)
    moment = np.asarray(moment)
    rest = np.asarray(rest)
    n = command.shape[0] if command.ndim == 2 else -1
    require(
        n >= 1
        and command.shape == (n, 42)
        and bank.shape == (n, 391, 3)
        and rate.shape == (n, 391)
        and activation.shape == (n, 96)
        and moment.shape == (42, 96)
        and rest.shape == (42,),
        "Complete E158 command reconstruction arrays required",
    )
    require(
        all(
            value.dtype == np.float64
            for value in (bank, rate, activation, command, moment, rest)
        ),
        "E158 reconstruction inputs must be float64",
    )
    require(
        all(
            np.isfinite(value).all()
            for value in (bank, rate, activation, command, moment, rest)
        ),
        "Finite E158 arrays required",
    )
    require(
        moment[joint_row, muscle] == 1.0, "Frozen RM singleton moment must equal one"
    )
    other = np.delete(moment[joint_row], muscle)
    require(
        np.array_equal(other, np.zeros(95)), "RM singleton joint row has another muscle"
    )
    ema40 = ema_from_saved_rate(
        rate[:, motor_column], float(bank[0, motor_column, branch])
    )
    bank_error = float(np.max(np.abs(ema40 - bank[:, motor_column, branch])))
    reconstructed = rest[None, :] + gain * (activation @ moment.T)
    error = reconstructed - command
    # Boundary zero is the restored body's observation, before a neural
    # command has been decoded. Only later boundaries are command endpoints.
    reconstruction_error = float(np.max(np.abs(error[1:]))) if n > 1 else 0.0
    initial_difference = float(np.max(np.abs(error[0])))
    if diagnostics is not None:
        diagnostics(
            {
                "time_s": np.arange(n) * CONTROL_DT,
                "baseline_command": command.copy(),
                "reconstructed_command": reconstructed,
                "reconstruction_error": error,
                "ema40": ema40.copy(),
                "saved_branch": bank[:, motor_column, branch].copy(),
                "source_rate": rate[:, motor_column].copy(),
            }
        )
    require(bank_error <= atol, "Saved 40 ms bank recurrence differs")
    require(
        reconstruction_error <= atol, "Saved original target reconstruction differs"
    )
    candidate_activation = activation.copy()
    candidate_activation[:, muscle] = ema40
    candidate = command.copy()
    candidate[1:, joint_row] = rest[joint_row] + gain * ema40[1:]
    unchanged = np.ones(42, dtype=bool)
    unchanged[joint_row] = False
    require(
        np.array_equal(candidate[:, unchanged], command[:, unchanged]),
        "Non-target joints changed",
    )
    require(
        np.array_equal(
            candidate_activation[:, np.arange(96) != muscle],
            activation[:, np.arange(96) != muscle],
        ),
        "Non-target activation changed",
    )
    return {
        "candidate_command": candidate,
        "candidate_activation": candidate_activation,
        "ema40": ema40,
        "source_rate": rate[:, motor_column].copy(),
        "bank_max_abs_error": bank_error,
        "target_reconstruction_max_abs_error": reconstruction_error,
        "initial_observation_reconstruction_difference": initial_difference,
    }


def error_metrics(actual: Any, target: Any) -> dict[str, float | bool]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    require(
        actual.shape == target.shape and actual.size > 0,
        "Comparable nonempty arrays required",
    )
    delta = actual - target
    finite = bool(np.isfinite(delta).all())
    return {
        "available": finite,
        "RMS": float(np.sqrt(np.mean(delta * delta))) if finite else math.inf,
        "max_abs": float(np.max(np.abs(delta))) if finite else math.inf,
    }


def startup_compatibility(
    candidate_command: Any,
    candidate_tendon: Any,
    original_command99: Any,
    original_tendon99: Any,
    limits: dict[str, float],
) -> dict[str, Any]:
    command = np.asarray(candidate_command, dtype=np.float64)
    tendon = np.asarray(candidate_tendon, dtype=np.float64)
    original_q = np.asarray(original_command99, dtype=np.float64)
    original_ltm = np.asarray(original_tendon99, dtype=np.float64)
    require(
        command.shape == (2001, 42) and tendon.shape == (2001, 6),
        "Complete candidate commands required",
    )
    require(
        original_q.shape == (99, 42) and original_ltm.shape == (99, 6),
        "Exact original99 startup arrays required",
    )
    q = error_metrics(command[1:100], original_q)
    ltm = error_metrics(tendon[1:100], original_ltm)
    passed = bool(
        q["available"]
        and ltm["available"]
        and q["RMS"] <= limits["q_RMS"]
        and q["max_abs"] <= limits["q_max"]
        and ltm["RMS"] <= limits["LTM_RMS"]
        and ltm["max_abs"] <= limits["LTM_max"]
    )
    return {
        "complete_original99_prefix": True,
        "time_s": (np.arange(1, 100) * CONTROL_DT).tolist(),
        "q": q,
        "LTM": ltm,
        "limits": copy.deepcopy(limits),
        "startup_pass": passed,
    }


def impulse_summary(
    time_s: Any, force_forward_bw: Any, *, window=(1.0, 2.0)
) -> dict[str, Any]:
    time_s = np.asarray(time_s, dtype=np.float64)
    force = np.asarray(force_forward_bw, dtype=np.float64)
    require(
        time_s.ndim == 1 and force.shape == (len(time_s), 6),
        "Complete native force history required",
    )
    require(
        np.isfinite(time_s).all() and np.isfinite(force).all(),
        "Finite native force history required",
    )
    if len(time_s) == N_NATIVE:
        require(
            np.allclose(
                time_s, np.arange(1, N_NATIVE + 1) * PHYSICS_DT, rtol=0, atol=CLOCK_ATOL
            ),
            "Canonical native clock required",
        )
        selected = force[10000:20000]
    else:
        require(
            len(time_s) == 10000
            and np.allclose(
                time_s, np.arange(10001, 20001) * PHYSICS_DT, rtol=0, atol=CLOCK_ATOL
            ),
            "Exact E154 (1,2] force window required",
        )
        selected = force
    negative = np.minimum(selected, 0.0)
    positive = np.maximum(selected, 0.0)
    neg = -negative.sum(axis=0) * PHYSICS_DT
    pos = positive.sum(axis=0) * PHYSICS_DT
    return {
        "window_s": list(window),
        "negative_forward_impulse_bw_s": dict(zip(LEGS, neg.tolist(), strict=True)),
        "positive_forward_impulse_bw_s": dict(zip(LEGS, pos.tolist(), strict=True)),
        "all_negative_forward_impulse_bw_s": float(neg.sum()),
        "all_positive_forward_impulse_bw_s": float(pos.sum()),
    }


def horizontal_heading(rotation_world: Any) -> np.ndarray:
    """Frozen E154 world-horizontal projection (world Z is vertical)."""
    rotation = np.asarray(rotation_world, dtype=np.float64)
    require(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0),
        "Finite orthogonal body rotation required",
    )
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    magnitude = float(np.linalg.norm(heading))
    require(magnitude > 1e-12, "Body forward axis has no horizontal projection")
    return heading / magnitude


def validate_complete_counts(ledger: dict[str, Any]) -> None:
    expected = {
        "body_constructor_attempts": 1,
        "body_constructor_completed": 1,
        "settling_physics_steps": 2000,
        "body_models": 1,
        "shadow_mjdata": 1,
        "scientific_steps_requested": N_NATIVE,
        "scientific_steps_completed": N_NATIVE,
        "scientific_samples_observed": N_NATIVE,
        "shadow_forward_requested": N_NATIVE,
        "shadow_forward_completed": N_NATIVE,
    }
    require(
        all(ledger.get(key) == value for key, value in expected.items()),
        "Exact bounded E158 body work required for a complete contrast",
    )


def advancement_decision(
    *,
    complete_health: bool,
    quality: dict[str, Any],
    slip: dict[str, Any],
    baseline_quality: dict[str, Any],
    baseline_slip: dict[str, Any],
    impulses: dict[str, Any],
    baseline_impulses: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen E158 advancement rule without selecting new thresholds."""
    direct = quality.get("fixed_initial_heading_net_forward_mm")
    path_ratio = quality.get("net_over_path")
    base_direct = baseline_quality.get("fixed_initial_heading_net_forward_mm")
    base_path_ratio = baseline_quality.get("net_over_path")
    forward_ratio = (
        direct / base_direct
        if direct is not None and base_direct not in (None, 0)
        else None
    )
    net_path_ratio = (
        path_ratio / base_path_ratio
        if path_ratio is not None and base_path_ratio not in (None, 0)
        else None
    )
    slip_value = slip.get("load_weighted_mm_s")
    base_slip = baseline_slip.get("load_weighted_mm_s")
    slip_ratio = (
        slip_value / base_slip
        if slip_value is not None and base_slip not in (None, 0)
        else None
    )
    rm = impulses.get("negative_forward_impulse_bw_s", {}).get("rm")
    base_rm = baseline_impulses.get("negative_forward_impulse_bw_s", {}).get("rm")
    rm_ratio = rm / base_rm if rm is not None and base_rm not in (None, 0) else None
    cycles = quality.get("forward_eligible_complete_cycles", {})
    widths = quality.get("knee_width_deg", {})
    longest = quality.get("longest_forward_bout_s")
    criteria = {
        "complete_health": bool(complete_health),
        "F3_numeric": quality.get("development_screens", {}).get("F3_numeric")
        is thresholds["F3_numeric"],
        "forward": forward_ratio is not None
        and forward_ratio >= thresholds["forward_ratio_min"],
        "net_path": net_path_ratio is not None
        and net_path_ratio >= thresholds["net_path_ratio_min"],
        "slip": slip.get("available") is True
        and slip_ratio is not None
        and slip_ratio <= thresholds["slip_ratio_max"],
        "knee_width": all(
            widths.get(leg) is not None and widths[leg] >= limit
            for leg, limit in thresholds["knee_width_deg_min"].items()
        ),
        "longest_bout": longest is not None
        and longest >= thresholds["longest_bout_s_min"],
        "all_leg_cycles": all(
            cycles.get(leg) is not None
            and cycles[leg] >= thresholds["all_leg_cycles_min"]
            for leg in LEGS
        ),
        "rm_backward_impulse": rm_ratio is not None
        and rm_ratio <= thresholds["rm_backward_impulse_ratio_max"],
    }
    return {
        "criteria": criteria,
        "advance_to_closed_loop": all(criteria.values()),
        "forward_ratio": forward_ratio,
        "net_path_ratio": net_path_ratio,
        "slip_ratio": slip_ratio,
        "rm_backward_impulse_ratio": rm_ratio,
    }


def new_native_registry() -> dict[str, list[Any]]:
    return {
        key: []
        for key in (
            "time_s",
            "control_tick",
            "qpos",
            "qvel",
            "ctrl",
            "integration_complete",
            "shadow_complete",
            "observation_complete",
            "rotation_world",
            "force_world_bw",
            "force_forward_bw",
            "normal_load_bw",
            "slip_mm_s",
            "slip_valid",
            "nonfoot_contact",
            "upright",
        )
    }


def native_arrays(registry: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    shapes = {
        "rotation_world": (3, 3),
        "force_world_bw": (6, 3),
        "force_forward_bw": (6,),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
        "slip_valid": (6,),
    }
    result = {}
    for key, values in registry.items():
        array = np.asarray(values)
        if key in shapes:
            array = array.reshape(-1, *shapes[key])
        result[key] = array
    return result


def _partial_vector(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    out = np.full(int(np.prod(shape)), np.nan)
    source = np.asarray(value, dtype=np.float64).reshape(-1)
    out[: min(len(out), len(source))] = source[: min(len(out), len(source))]
    return out.reshape(shape)


def execute_body(
    *,
    body: Any,
    shadow: Any,
    rows: list[dict[str, Any]],
    candidate_command: np.ndarray,
    implicit_request: dict[str, Any] | None,
    replay_command: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    observe_shadow: Callable[..., tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]],
    contact_diagnostics: Callable[[Any], dict[str, Any]],
    capture_control: Callable[[Any], dict[str, Any]],
    ledger: dict[str, Any],
    deadline_check: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Replay one candidate body while retaining integration before observation."""
    require(
        len(rows) == 2001 and candidate_command.shape == (2001, 42),
        "Complete E158 controls required",
    )
    registry = new_native_registry()
    trace_rows: list[dict[str, Any]] = []
    state_type = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, state_type))
    joint_mask = np.zeros(int(body.m.nu), dtype=bool)
    joint_mask[np.asarray(body.act_ids, dtype=np.int64)] = True
    original_step = body._step_physics
    active_tick = 0
    expected_joint: np.ndarray | None = None
    expected_nonjoint: np.ndarray | None = None

    def observed_step() -> None:
        ledger["scientific_steps_requested"] += 1
        original_step()
        ledger["scientific_steps_completed"] += 1
        registry["time_s"].append(float(body.physics_time()))
        registry["control_tick"].append(active_tick)
        registry["qpos"].append(np.array(body.d.qpos, copy=True))
        registry["qvel"].append(np.array(body.d.qvel, copy=True))
        registry["ctrl"].append(np.array(body.d.ctrl, copy=True))
        registry["integration_complete"].append(True)
        for key in ("shadow_complete", "observation_complete"):
            registry[key].append(False)
        for key, shape in (
            ("rotation_world", (3, 3)),
            ("force_world_bw", (6, 3)),
            ("force_forward_bw", (6,)),
            ("normal_load_bw", (6,)),
            ("slip_mm_s", (6,)),
        ):
            registry[key].append(np.full(shape, np.nan))
        registry["slip_valid"].append(np.zeros(6, dtype=bool))
        registry["nonfoot_contact"].append(False)
        registry["upright"].append(np.nan)
        require(
            expected_joint is not None
            and np.array_equal(body.d.ctrl[body.act_ids], expected_joint),
            "Candidate joint hold changed",
        )
        require(
            expected_nonjoint is not None
            and np.array_equal(body.d.ctrl[~joint_mask], expected_nonjoint),
            "Non-joint control changed",
        )
        body.mj.mj_getState(body.m, body.d, state, state_type)
        body.mj.mj_setState(body.m, shadow.d, state, state_type)
        ledger["shadow_forward_requested"] += 1
        body.mj.mj_forward(body.m, shadow.d)
        ledger["shadow_forward_completed"] += 1
        registry["shadow_complete"][-1] = True
        force, _contacts, rotation = observe_shadow(body, shadow)
        rotation = np.asarray(rotation, dtype=np.float64)
        force = np.asarray(force, dtype=np.float64)
        for key, value in (("rotation_world", rotation), ("force_world_bw", force)):
            registry[key][-1] = _partial_vector(
                value,
                {
                    "rotation_world": (3, 3),
                    "force_world_bw": (6, 3),
                    "normal_load_bw": (6,),
                    "slip_mm_s": (6,),
                }[key],
            )
        diagnostic = contact_diagnostics(shadow)
        load = np.asarray(diagnostic["normal_load_bw"], dtype=np.float64)
        slip = np.asarray(
            [np.nan if x is None else x for x in diagnostic["slip_mm_s"]],
            dtype=np.float64,
        )
        for key, value in (("normal_load_bw", load), ("slip_mm_s", slip)):
            registry[key][-1] = _partial_vector(
                value, {"normal_load_bw": (6,), "slip_mm_s": (6,)}[key]
            )
        require(
            rotation.shape == (3, 3)
            and force.shape == (6, 3)
            and load.shape == slip.shape == (6,),
            "Complete native observation required",
        )
        heading = horizontal_heading(rotation)
        registry["force_forward_bw"][-1] = force @ heading
        registry["slip_valid"][-1] = np.isfinite(slip)
        nonfoot = bool(shadow.nonfoot_contact())
        upright = float(rotation[2, 2])
        registry["nonfoot_contact"][-1] = nonfoot
        registry["upright"][-1] = upright
        require(
            np.isfinite(rotation).all()
            and np.isfinite(force).all()
            and np.isfinite(load).all()
            and np.all(load >= 0)
            and math.isfinite(upright),
            "Finite native observation required",
        )
        registry["observation_complete"][-1] = True
        ledger["scientific_samples_observed"] += 1
        if (
            nonfoot
            or upright < 0.9
            or not np.isfinite(body.d.qpos).all()
            or not np.isfinite(body.d.qvel).all()
        ):
            raise NativeHealthStop("E158 native health stop")

    body._step_physics = observed_step
    try:
        view = SimpleNamespace(
            body=body,
            control_tick=0,
            tendon_inputs=copy.deepcopy(rows[0]["tendon_inputs"]),
            rate=np.asarray(
                rows[0].get("motor_rates", np.zeros(391)), dtype=np.float64
            ),
            fault=None,
            navigation=None,
        )
        trace_rows.append(capture_control(view))
        for tick, row in enumerate(rows[1:], 1):
            active_tick = tick
            original_joint, adhesion, command = replay_command(
                row, implicit_request=implicit_request
            )
            require(
                np.array_equal(
                    original_joint,
                    np.asarray(row["joint_targets_rad"], dtype=np.float64),
                ),
                "Recorded endpoint changed",
            )
            full = np.asarray(row["ctrl"], dtype=np.float64)
            require(full.shape == (int(body.m.nu),), "Recorded full ctrl width differs")
            expected_nonjoint = full[~joint_mask].copy()
            expected_joint = np.array(candidate_command[tick], copy=True)
            body.push_left = command["push_left_s"]
            body.push_force = command["push_force"]
            step_kwargs = copy.deepcopy(command["step_kwargs"])
            step_kwargs["interpolate"] = False
            body.step_joint_targets(candidate_command[tick], adhesion, **step_kwargs)
            require(
                np.array_equal(body.d.ctrl[body.act_ids], candidate_command[tick]),
                "Candidate joint endpoint differs",
            )
            require(
                np.array_equal(body.d.ctrl[~joint_mask], expected_nonjoint),
                "Candidate non-joint endpoint differs",
            )
            require(
                abs(float(body.physics_time()) - tick * CONTROL_DT) <= CLOCK_ATOL,
                "E158 body clock differs",
            )
            view = SimpleNamespace(
                body=body,
                control_tick=tick,
                tendon_inputs=copy.deepcopy(row["tendon_inputs"]),
                rate=np.asarray(
                    row.get("motor_rates", np.zeros(391)), dtype=np.float64
                ),
                fault=None,
                navigation=None,
            )
            trace_rows.append(capture_control(view))
            require(
                not body.fault and not np.any(np.asarray(body.d.warning.number)),
                "E158 body fault/warning",
            )
            if deadline_check is not None:
                deadline_check()
    except BaseException as error:
        error.e158_native = native_arrays(registry)
        error.e158_trace_rows = copy.deepcopy(trace_rows)
        raise
    finally:
        body._step_physics = original_step
    return trace_rows, native_arrays(registry)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _tendon_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    result = np.asarray(
        [[row["tendon_inputs"][key] for key in TENDONS] for row in rows],
        dtype=np.float64,
    )
    require(
        result.shape == (2001, 6) and np.isfinite(result).all(),
        "Complete finite tendon commands required",
    )
    return result


def _trace_commands(rows: list[dict[str, Any]]) -> np.ndarray:
    result = np.asarray([row["joint_targets_rad"] for row in rows], dtype=np.float64)
    require(
        result.shape == (2001, 42) and np.isfinite(result).all(),
        "Complete finite baseline commands required",
    )
    require(
        np.array_equal(
            np.asarray([row["time_s"] for row in rows]), np.arange(2001) * CONTROL_DT
        ),
        "Exact E150 boundary clock required",
    )
    require(
        np.array_equal(
            np.asarray([row["control_tick"] for row in rows]), np.arange(2001)
        ),
        "Exact E150 ticks required",
    )
    return result


def _startup_arrays(startup: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    q_key = (
        "original_E100_E103_command"
        if "original_E100_E103_command" in startup
        else "target_command"
    )
    ltm_key = (
        "original_E100_E103_LTM"
        if "original_E100_E103_LTM" in startup
        else "target_tendon"
    )
    q = np.asarray(startup[q_key], dtype=np.float64)
    ltm = np.asarray(startup[ltm_key], dtype=np.float64)
    require(q.shape == (99, 42) and ltm.shape == (99, 6), "E158 startup schema differs")
    return q, ltm


def validate_singleton_mapping(
    parameters: dict[str, np.ndarray],
    provenance: dict[str, np.ndarray],
    intervention: dict[str, Any],
) -> None:
    """Bind the frozen global MN column to its one-muscle anatomical row."""
    required_parameters = {"tau", "kappa", "slopes", "membership", "motor_ids"}
    required_provenance = {"motor_ids", "moment", "rest", "pool", "membership", "ltm"}
    require(
        set(parameters) == required_parameters, "E150 parameter bundle schema differs"
    )
    require(
        required_provenance <= set(provenance), "E150 adapter provenance schema differs"
    )
    membership = np.asarray(parameters["membership"])
    motor_ids = np.asarray(parameters["motor_ids"])
    require(
        membership.shape == motor_ids.shape == (391,)
        and membership.dtype.kind in "iu"
        and motor_ids.dtype.kind in "iu",
        "E150 391-MN identity arrays differ",
    )
    column = intervention["motor_column"]
    muscle = intervention["muscle"]
    require(
        int(membership[column]) == muscle
        and np.count_nonzero(membership == muscle) == 1
        and int(motor_ids[column]) == intervention["graph_index"]
        and np.array_equal(membership, np.asarray(provenance["membership"]))
        and np.array_equal(motor_ids, np.asarray(provenance["motor_ids"])),
        "E158 singleton global/local mapping differs",
    )
    require(
        np.asarray(parameters["tau"]).dtype == np.float64
        and np.array_equal(parameters["tau"], np.array([0.005, 0.01, 0.04]))
        and np.asarray(parameters["slopes"]).shape == (96, 391, 3, 2),
        "E158 frozen branch parameters differ",
    )


def _save_startup(
    out: Path,
    command: np.ndarray,
    tendon: np.ndarray,
    q: np.ndarray,
    ltm: np.ndarray,
    screen: dict[str, Any],
) -> None:
    atomic_npz(
        out / "startup-original99.npz",
        time_s=np.arange(1, 100) * CONTROL_DT,
        candidate_command=command[1:100],
        candidate_LTM=tendon[1:100],
        original_command=q,
        original_LTM=ltm,
        command_error=command[1:100] - q,
        LTM_error=tendon[1:100] - ltm,
    )
    atomic_json(out / "startup-screen.json", screen)


def output_manifest(out: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "output-manifest.json"
    ):
        files[path.relative_to(out).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "flylab.m3.e158.output-manifest.v1",
        "files": files,
        "total_bytes": sum(x["bytes"] for x in files.values()),
    }


def _default_body_dependencies() -> dict[str, Any]:
    """Import original-only dependencies after Stage A accepts the candidate."""
    from flylab.c.walking_trace import capture_control, contact_diagnostics
    from tools import analyze_c_banc_biological_gait as gait
    from tools import analyze_c_banc_m3_signed_contact as e154
    from tools import analyze_c_banc_observation as observation
    from tools import verify_c_banc_m3_e113_body as e113
    from tools.verify_c_banc_m3_native_timescale_transfer import StateStore

    return {
        "capture_control": capture_control,
        "contact_diagnostics": contact_diagnostics,
        "gait": gait,
        "e113": e113,
        "observation": observation,
        "state_load": StateStore.load,
        "construct_body": e154.construct_body_safely,
        "implicit_request": e154.implicit_request_for_trace,
        "replay_command": e154.replay_command,
        "observe_shadow": e154._observe_shadow,
        "execute_body": execute_body,
    }


def run(
    freeze_path: str | Path,
    freeze_sha256: str,
    source_review: str | Path,
    source_review_sha256: str,
    output: str | Path,
    *,
    body_runner: Callable[..., tuple[list[dict[str, Any]], dict[str, np.ndarray]]]
    | None = None,
    dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the one gated original.  ``body_runner`` is for generated stubs."""
    began = time.perf_counter()
    freeze, _review = verified_inputs(
        freeze_path, freeze_sha256, source_review, source_review_sha256
    )
    out = Path(output).resolve()
    require(not out.exists(), "E158 output must be new")
    require(
        shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
        "E158 free-space reserve unavailable",
    )
    out.mkdir(parents=True)
    ledger: dict[str, Any] = {
        "schema": "flylab.m3.e158.ledger.v1",
        "status": "created",
        "original_processes": 1,
        "neural_steps": 0,
        "fit_calls": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "settling_physics_steps": 0,
        "body_models": 0,
        "shadow_mjdata": 0,
        "scientific_steps_requested": 0,
        "scientific_steps_completed": 0,
        "scientific_samples_observed": 0,
        "shadow_forward_requested": 0,
        "shadow_forward_completed": 0,
        "cleanup_attempted": False,
        "cleanup_completed": False,
        "secondary_errors": [],
    }
    primary: BaseException | None = None
    health_stop: NativeHealthStop | None = None
    body = None
    thread_limits = None
    result: dict[str, Any] = {
        "schema": "flylab.m3.e158.result.v1",
        "execution_schema": freeze.get("schema"),
        "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested",
        "startup_pass": False,
        "body_executed": False,
        "advance_to_closed_loop": False,
        "physical_metrics": None,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "active_UI_changed": False,
    }

    def persist() -> None:
        atomic_json(out / "ledger.json", ledger)

    def deadline(label: str) -> None:
        if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E158 internal deadline: " + label)

    def add_secondary(
        error: BaseException, stage: str, owner: BaseException | None
    ) -> None:
        ledger["secondary_errors"].append({"stage": stage, "error": repr(error)})
        if owner is not None:
            owner.add_note("Secondary E158 " + stage + ": " + repr(error))

    def save_partial(error: BaseException) -> None:
        failed = False
        operations = (
            (
                "partial native raw",
                getattr(error, "e158_native", None),
                lambda value: atomic_npz(out / "candidate-native-partial.npz", **value),
            ),
            (
                "partial trace raw",
                getattr(error, "e158_trace_rows", None),
                lambda value: atomic_json(
                    out / "candidate-trace-partial.json",
                    {"schema": "flylab.m3.e158.partial-trace.v1", "samples": value},
                ),
            ),
        )
        for stage, value, operation in operations:
            if value is None:
                continue
            try:
                operation(value)
            except BaseException as secondary:
                failed = True
                add_secondary(secondary, stage, error)
        error.e158_partial_save_failed = failed

    try:
        thread_limits = threadpool_limits(limits=1)
        ledger["cpu_thread_environment"] = {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        ledger["threadpools_during"] = threadpool_info()
        require(
            all(value == "1" for value in ledger["cpu_thread_environment"].values())
            and all(
                pool.get("num_threads") == 1 for pool in ledger["threadpools_during"]
            ),
            "Actual E158 CPU thread count exceeds one",
        )
        persist()
        payload = json.loads(safe_path(freeze["trace"]).read_text(encoding="utf-8"))
        rows = payload.get("samples")
        require(
            isinstance(rows, list) and len(rows) == 2001, "Complete E150 trace required"
        )
        recruitment = _load_npz(safe_path(freeze["recruitment"]))
        provenance = _load_npz(safe_path(freeze["provenance"]))
        parameters = _load_npz(safe_path(freeze["parameters"]))
        startup = _load_npz(safe_path(freeze["startup"]))
        validate_singleton_mapping(parameters, provenance, freeze["intervention"])
        command = _trace_commands(rows)
        tendon = _tendon_matrix(rows)
        for key in ("bank", "activation", "motor_rate"):
            require(
                key in recruitment and np.asarray(recruitment[key]).dtype == np.float64,
                "E150 recruitment schema differs: " + key,
            )
        for key in (
            "bank_available",
            "activation_available",
            "motor_rate_available",
            "complete",
        ):
            require(
                np.asarray(recruitment[key]).shape == (2001,)
                and np.asarray(recruitment[key]).all(),
                "Incomplete E150 recruitment state: " + key,
            )
        built = build_candidate_commands(
            bank=recruitment["bank"],
            motor_rate=recruitment["motor_rate"],
            baseline_activation=recruitment["activation"],
            baseline_command=command,
            moment=provenance["moment"],
            rest=provenance["rest"],
            diagnostics=lambda arrays: atomic_npz(
                out / "reconstruction-diagnostic.npz", **arrays
            ),
            **{
                key: freeze["intervention"][key]
                for key in ("muscle", "motor_column", "branch", "joint_row", "gain")
            },
        )
        candidate = np.asarray(built["candidate_command"])
        candidate_activation = np.asarray(built["candidate_activation"])
        atomic_npz(
            out / "candidate-commands.npz",
            time_s=np.arange(2001) * CONTROL_DT,
            baseline_command=command,
            candidate_command=candidate,
            baseline_activation=np.asarray(recruitment["activation"]),
            candidate_activation=candidate_activation,
            ema40=built["ema40"],
            source_rate=built["source_rate"],
            baseline_tendon=tendon,
            candidate_tendon=tendon.copy(),
        )
        q99, ltm99 = _startup_arrays(startup)
        screen = startup_compatibility(
            candidate, tendon, q99, ltm99, freeze["startup_limits"]
        )
        _save_startup(out, candidate, tendon, q99, ltm99, screen)
        result.update(
            startup_pass=screen["startup_pass"],
            startup=screen,
            bank_max_abs_error=built["bank_max_abs_error"],
            target_reconstruction_max_abs_error=built[
                "target_reconstruction_max_abs_error"
            ],
            initial_observation_reconstruction_difference=built[
                "initial_observation_reconstruction_difference"
            ],
        )
        ledger["status"] = "startup_complete"
        persist()
        if not screen["startup_pass"]:
            result.update(
                validity="valid_startup_rejected",
                hypothesis_verdict="not_physically_tested_startup_rejected",
                stop_reason="startup_rejected",
            )
            ledger["status"] = "startup_rejected"
        else:
            deadline("before body import/construction")
            if body_runner is None:
                deps = (
                    _default_body_dependencies()
                    if dependencies is None
                    else dependencies
                )
                saved = deps["state_load"](safe_path(freeze["checkpoint"]))
                require(
                    saved.get("control_tick") == 0
                    and saved.get("body", {}).get("modelHash") == freeze["body_hash"],
                    "E150 start checkpoint differs",
                )
                implicit = deps["implicit_request"](payload, saved)
                body = deps["construct_body"](saved, deps["observation"], ledger)
                ledger["body_models"] = 1
                require(
                    abs(float(body.m.opt.timestep) - PHYSICS_DT) <= 1e-15,
                    "E158 physics dt differs",
                )
                for key in ("qpos", "qvel", "ctrl"):
                    require(
                        np.array_equal(
                            np.asarray(getattr(body.d, key)), np.asarray(rows[0][key])
                        ),
                        "E150 start state differs: " + key,
                    )
                require(
                    abs(float(body.physics_time())) <= CLOCK_ATOL,
                    "E150 start clock differs",
                )
                shadow = copy.copy(body)
                shadow.d = body.mj.MjData(body.m)
                ledger["shadow_mjdata"] = 1
                runner = deps["execute_body"]
                try:
                    trace_rows, native = runner(
                        body=body,
                        shadow=shadow,
                        rows=rows,
                        candidate_command=candidate,
                        implicit_request=implicit,
                        replay_command=deps["replay_command"],
                        observe_shadow=deps["observe_shadow"],
                        contact_diagnostics=deps["contact_diagnostics"],
                        capture_control=deps["capture_control"],
                        ledger=ledger,
                        deadline_check=lambda: deadline("body replay"),
                    )
                except BaseException as error:
                    save_partial(error)
                    raise
                atomic_npz(out / "candidate-native.npz", **native)
                candidate_payload = copy.deepcopy(payload)
                candidate_payload.update(
                    schema="flylab.m3.e158.rm-singleton-kinetics-trace.v1",
                    controller="E150 fixed candidate with RM muscle67 40ms activation",
                    samples=trace_rows,
                    E158_provenance={
                        "intervention": freeze["intervention"],
                        "neural_steps": 0,
                    },
                )
                atomic_json(out / "candidate-trace.json", candidate_payload)
                quality_raw, _, _ = deps["gait"].analyze(
                    deps["gait"].load_model(out / "candidate-trace.json")
                )
                quality = deps["e113"].body_quality(
                    trace_rows,
                    list(
                        zip(
                            native["time_s"],
                            native["normal_load_bw"],
                            native["slip_mm_s"],
                            native["slip_valid"],
                            native["nonfoot_contact"],
                            native["upright"],
                            strict=True,
                        )
                    ),
                    quality_raw,
                    reason="duration",
                    fault=False,
                    clock_valid=True,
                    future_exact=False,
                )
                quality["longest_forward_bout_s"] = quality_raw["window"][
                    "longest_forward_bout_s"
                ]
                slip = deps["e113"].supported_slip(
                    native, complete=quality["complete_10s_native_health"]
                )
            else:
                try:
                    stub_value = body_runner(
                        rows=rows,
                        candidate_command=candidate,
                        ledger=ledger,
                        deadline_check=lambda: deadline("stub body replay"),
                    )
                except BaseException as error:
                    save_partial(error)
                    raise
                require(
                    isinstance(stub_value, tuple) and len(stub_value) == 4,
                    "Generated E158 runner must return rows/native/quality/slip",
                )
                trace_rows, native, quality, slip = stub_value
                atomic_npz(out / "candidate-native.npz", **native)
            require(
                len(trace_rows) == 2001 and len(native["time_s"]) == N_NATIVE,
                "Complete E158 body raw required",
            )
            validate_complete_counts(ledger)
            impulses = impulse_summary(native["time_s"], native["force_forward_bw"])
            baseline_impulses = impulse_summary(
                **{
                    key: _load_npz(safe_path(freeze["baseline_forces"]))[key]
                    for key in ("time_s", "force_forward_bw")
                }
            )
            baseline_dir = safe_path(freeze["baseline_directory"])
            baseline_quality = json.loads(
                (baseline_dir / "quality-summary.json").read_text(encoding="utf-8")
            )
            baseline_slip = json.loads(
                (baseline_dir / "supported-slip.json").read_text(encoding="utf-8")
            )
            decision = advancement_decision(
                complete_health=quality.get("complete_10s_native_health") is True,
                quality=quality,
                slip=slip,
                baseline_quality=baseline_quality,
                baseline_slip=baseline_slip,
                impulses=impulses,
                baseline_impulses=baseline_impulses,
                thresholds=freeze["advancement"],
            )
            atomic_json(out / "quality-summary.json", quality)
            atomic_json(out / "supported-slip.json", slip)
            atomic_json(
                out / "impulses.json",
                {"candidate": impulses, "baseline": baseline_impulses},
            )
            atomic_json(out / "advancement.json", decision)
            result.update(
                validity="valid_body_contrast",
                hypothesis_verdict="supported"
                if decision["advance_to_closed_loop"]
                else "refuted",
                stop_reason="duration",
                body_executed=True,
                physical_metrics={
                    "quality": quality,
                    "supported_slip": slip,
                    "impulses": impulses,
                    "advancement": decision,
                },
                advance_to_closed_loop=decision["advance_to_closed_loop"],
            )
            ledger["status"] = "body_complete"
    except NativeHealthStop as error:
        if getattr(error, "e158_partial_save_failed", False):
            primary = error
            ledger["status"] = "technical_failed"
        else:
            health_stop = error
            ledger["status"] = "measured_native_health_stop"
            ledger["health_stop"] = {
                "type": type(error).__name__,
                "message": str(error),
                "notes": list(getattr(error, "__notes__", [])),
            }
    except BaseException as error:
        primary = error
        ledger["status"] = "technical_failed"
    finally:

        def record_primary(error: BaseException, stage: str) -> None:
            nonlocal primary
            if primary is None:
                primary = error
            else:
                add_secondary(error, stage, primary)
            ledger["status"] = "technical_failed"

        def update_receipts() -> None:
            elapsed = time.perf_counter() - began
            result.update(
                process_wall_s=elapsed,
                prior_conservative_actual_s=freeze["prior_actual_original_s"],
                prior_conservative_s=freeze["prior_conservative_s"],
                conservative_cumulative_s=freeze["prior_conservative_s"] + elapsed,
                counts={
                    key: ledger[key]
                    for key in (
                        "neural_steps",
                        "fit_calls",
                        "body_constructor_attempts",
                        "body_constructor_completed",
                        "settling_physics_steps",
                        "body_models",
                        "shadow_mjdata",
                        "scientific_steps_requested",
                        "scientific_steps_completed",
                        "scientific_samples_observed",
                        "shadow_forward_requested",
                        "shadow_forward_completed",
                    )
                },
                cpu_thread_environment=copy.deepcopy(
                    ledger.get("cpu_thread_environment")
                ),
                threadpools_during=copy.deepcopy(ledger.get("threadpools_during")),
                candidate_adopted=False,
                whole_goal_complete=False,
                F6_exposed=False,
                active_UI_changed=False,
                body_executed=bool(
                    result.get("body_executed")
                    or ledger["scientific_steps_completed"] > 0
                ),
            )
            if primary is not None:
                # A measured screen never authorizes advancement after the
                # experiment's recording, resource or cleanup contract fails.
                result["advance_to_closed_loop"] = False
                metrics = result.get("physical_metrics")
                decision = (
                    metrics.get("advancement") if isinstance(metrics, dict) else None
                )
                if isinstance(decision, dict):
                    decision["advance_to_closed_loop"] = False
                    decision["revoked_by"] = "technical_or_resource_failure"
                    try:
                        atomic_json(out / "advancement.json", decision)
                    except BaseException as error:
                        add_secondary(error, "revoked advancement save", primary)
                failure = {
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                }
                result.update(
                    validity="incomplete",
                    hypothesis_verdict="invalid_not_tested",
                    technical_failure=True,
                    primary_error=failure,
                )
                ledger["primary_error"] = failure
            elif health_stop is not None:
                result.update(
                    validity="measured_native_health_stop",
                    hypothesis_verdict="refuted_health_stop",
                    technical_failure=False,
                    stop_reason="native_health_stop",
                    advance_to_closed_loop=False,
                    body_executed=ledger["scientific_steps_completed"] > 0,
                    physical_metrics={
                        "complete": False,
                        "native_samples_completed": ledger[
                            "scientific_steps_completed"
                        ],
                        "health_stop": str(health_stop),
                    },
                )
            else:
                result["technical_failure"] = False

        if body is not None:
            ledger["cleanup_attempted"] = True
            try:
                body.close()
                ledger["cleanup_completed"] = True
            except BaseException as error:
                ledger["secondary_errors"].append(
                    {"stage": "body_close", "error": repr(error)}
                )
                record_primary(error, "body close")
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
                ledger["threadpools_after"] = threadpool_info()
            except BaseException as error:
                record_primary(error, "threadpool restore")

        try:
            raw_bytes = sum(
                path.stat().st_size for path in out.rglob("*") if path.is_file()
            )
            if raw_bytes > freeze["resource_limits"]["raw_bytes"]:
                raise RuntimeError("E158 raw-byte limit exceeded")
            if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
                raise TimeoutError("E158 finalization exceeded internal deadline")
        except BaseException as error:
            record_primary(error, "resource checks")

        update_receipts()
        try:
            persist()
        except BaseException as error:
            record_primary(error, "final ledger save")
            update_receipts()
            try:
                persist()
            except BaseException as secondary:
                add_secondary(secondary, "fail-closed ledger save", primary)
        update_receipts()

        result_written = False
        try:
            atomic_json(out / "result.json", result)
            result_written = True
        except BaseException as error:
            record_primary(error, "final result save")
            update_receipts()
            try:
                atomic_json(out / "result.json", result)
                result_written = True
            except BaseException as publication_error:
                add_secondary(publication_error, "fail-closed result save", primary)

        manifest = None
        if result_written:
            try:
                manifest = output_manifest(out)
                if (
                    time.perf_counter() - began
                    > freeze["resource_limits"]["internal_s"]
                ):
                    raise TimeoutError("E158 post-manifest deadline exceeded")
            except BaseException as error:
                record_primary(error, "manifest construction/deadline")
                update_receipts()
                try:
                    persist()
                except BaseException as secondary:
                    add_secondary(secondary, "late fail-closed ledger save", primary)
                try:
                    atomic_json(out / "result.json", result)
                except BaseException as secondary:
                    add_secondary(secondary, "late fail-closed result save", primary)
                # Receipts may have changed after the first hash pass. This
                # bounded recovery pass records their final failed state.
                try:
                    manifest = output_manifest(out)
                except BaseException as secondary:
                    manifest = None
                    add_secondary(secondary, "fail-closed manifest rebuild", primary)
            if manifest is not None:
                try:
                    atomic_json(out / "output-manifest.json", manifest)
                except BaseException as error:
                    record_primary(error, "manifest publication")
                    update_receipts()
                    try:
                        persist()
                    except BaseException as secondary:
                        add_secondary(
                            secondary, "manifest-failure ledger save", primary
                        )
                    try:
                        atomic_json(out / "result.json", result)
                    except BaseException as secondary:
                        add_secondary(
                            secondary, "manifest-failure result save", primary
                        )
                    try:
                        atomic_json(out / "output-manifest.json", output_manifest(out))
                    except BaseException as secondary:
                        add_secondary(secondary, "fail-closed manifest save", primary)
    if primary is not None:
        raise primary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.freeze_sha256,
                args.source_review,
                args.source_review_sha256,
                args.out,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
