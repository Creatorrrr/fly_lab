"""E154 continuation of the E153 exact saved-command signed-contact diagnostic."""

# ruff: noqa: I001  # Thread limits must precede numerical imports.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info


FREEZE_SHA = "34c04145128c2729f6687551a4c4a15108bbec1a864bdda97277cc8a0e0708a1"
DESIGN_SHA = "6f1554a29279dfac0164c3a3b8e2876e0826e7ebb207c5c9648c329e9050e0fd"
OWNED = (
    "tools/analyze_c_banc_m3_signed_contact.py",
    "tests/test_c_banc_signed_contact.py",
)
SCHEMA = "flylab.m3.e154.signed-contact-result.v1"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
SUPPORT_BW = 0.02
PHYSICS_DT = 0.0001
PHYSICS_PER_CONTROL = 50
CLOCK_ATOL = 2e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_progress(path: Path, stage: str, started: float, **values: Any) -> None:
    record = {"stage": stage, "wall_s": time.perf_counter() - started, **values}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def safe_path(relative: str) -> Path:
    value = Path(relative)
    path = (ROOT / value).resolve()
    require(
        not value.is_absolute()
        and ".." not in value.parts
        and path.is_relative_to(ROOT),
        "E153 frozen paths must be repository-relative",
    )
    return path


def verify_hash_map(entries: dict[str, str]) -> None:
    for relative, digest in entries.items():
        require(
            sha256(safe_path(relative)) == digest, "Frozen hash mismatch: " + relative
        )


def load_freeze(path: Path, expected_sha: str) -> dict[str, Any]:
    require(
        expected_sha == FREEZE_SHA and sha256(path) == FREEZE_SHA,
        "Exact E153 freeze required",
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed_replay = {
        "anchor_tick": 200,
        "first_tick": 201,
        "final_tick": 400,
        "controls": 200,
        "physics_per_control": 50,
        "physics_dt_s": 0.0001,
        "control_dt_s": 0.005,
        "scientific_physics_steps": 10000,
        "expected_settling_physics_steps": 2000,
        "start_s": 1.0,
        "end_s": 2.0,
        "native_slice": [10000, 20000],
    }
    require(
        value.get("schema") == "flylab.m3.e154.signed-contact-continuation.v1",
        "E153 freeze schema changed",
    )
    require(
        value.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(value["design"])) == DESIGN_SHA,
        "E153 design changed",
    )
    require(value.get("owned_files") == list(OWNED), "E153 owned files changed")
    require(value.get("replay") == fixed_replay, "E153 replay contract changed")
    require(
        value.get("candidate_adopted") is False
        and value.get("whole_goal_complete") is False
        and value.get("F6_exposed") is False
        and value.get("E150_original_still_incomplete") is True,
        "E153 outcome boundary changed",
    )
    limits = value.get("resource_limits", {})
    require(
        limits.get("original_processes") == 1
        and limits.get("automatic_retries") == 0
        and limits.get("CPU_threads") == 1
        and limits.get("GPU") == 0
        and limits.get("internal_wall_s") == 35
        and limits.get("external_wall_s") == 40
        and limits.get("supervisor_wall_s") == 48,
        "E153 resource contract changed",
    )
    return value


def validate_source_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha, "Parent E153 source-review SHA changed")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("accepts_single_body_diagnostic") is True,
        "Parent did not accept one E153 body diagnostic",
    )
    require(
        review.get("freeze_sha256") == FREEZE_SHA,
        "Parent E153 gate has a different freeze",
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "Parent E153 gate must allow exactly one original process",
    )
    require(
        review.get("source_sha256") == expected_sources,
        "Parent E153 gate source map differs from both owned files",
    )
    return review


def deadline(started: float, limit_s: float, operation: str) -> float:
    elapsed = time.perf_counter() - started
    if elapsed > limit_s:
        raise TimeoutError(f"E153 {operation} exceeded {limit_s}s: {elapsed:.6f}s")
    return elapsed


def horizontal_heading(rotation_world: Any) -> np.ndarray:
    rotation = np.asarray(rotation_world, dtype=np.float64)
    require(
        rotation.shape == (3, 3) and np.isfinite(rotation).all(),
        "Finite 3x3 body rotation required",
    )
    require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0),
        "Body rotation is not orthogonal",
    )
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    magnitude = float(np.linalg.norm(heading))
    require(magnitude > 1e-12, "Body forward axis has no horizontal projection")
    return heading / magnitude


def contact_force_world(
    frame: Any, wrench: Any, sign: int, weight0: float
) -> np.ndarray:
    axes = np.asarray(frame, dtype=np.float64)
    force_torque = np.asarray(wrench, dtype=np.float64)
    require(
        axes.shape == (3, 3)
        and force_torque.shape == (6,)
        and np.isfinite(axes).all()
        and np.isfinite(force_torque).all(),
        "Finite contact frame and six-axis wrench required",
    )
    require(
        np.allclose(axes @ axes.T, np.eye(3), atol=1e-9, rtol=0),
        "Contact frame is not orthogonal",
    )
    require(type(sign) is int and sign in (-1, 1), "Contact sign must be exactly +/-1")
    require(
        np.isfinite(weight0) and weight0 > 0, "Positive finite body weight required"
    )
    return sign * (axes.T @ force_torque[:3]) / weight0


def _mapped_segment(mapping: np.ndarray, geom: int) -> int:
    return int(mapping[geom]) if 0 <= geom < len(mapping) else -1


def classify_contact(
    geom1: int,
    geom2: int,
    excluded: bool,
    feet_mapping: Any,
    support_geom_ids: Any,
) -> tuple[int, int, int] | None:
    if bool(excluded):
        return None
    mapping = np.asarray(feet_mapping)
    require(
        mapping.ndim == 1 and mapping.dtype.kind in "iu",
        "Integer one-dimensional feet map required",
    )
    supports = {int(value) for value in support_geom_ids}
    a, b = int(geom1), int(geom2)
    if a in supports and _mapped_segment(mapping, b) >= 0:
        segment, sign = _mapped_segment(mapping, b), 1
    elif b in supports and _mapped_segment(mapping, a) >= 0:
        segment, sign = _mapped_segment(mapping, a), -1
    else:
        return None
    leg = segment // 5
    require(0 <= leg < 6, "Mapped foot segment is outside six legs")
    return segment, leg, sign


def transform_contacts(
    geom1: Any,
    geom2: Any,
    excluded: Any,
    frame: Any,
    wrench: Any,
    position_world: Any,
    feet_mapping: Any,
    support_geom_ids: Any,
    weight0: float,
    rotation_world: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Transform one shadow state's contacts while preserving accepted raw order."""
    first = np.asarray(geom1)
    second = np.asarray(geom2)
    omit = np.asarray(excluded)
    frames = np.asarray(frame, dtype=np.float64)
    wrenches = np.asarray(wrench, dtype=np.float64)
    positions = np.asarray(position_world, dtype=np.float64)
    count = len(first)
    require(
        first.shape == second.shape == omit.shape == (count,)
        and first.dtype.kind in "iu"
        and second.dtype.kind in "iu"
        and omit.dtype == np.dtype(bool)
        and frames.shape == (count, 3, 3)
        and wrenches.shape == (count, 6)
        and positions.shape == (count, 3),
        "Contact batch shapes or dtypes differ",
    )
    heading = horizontal_heading(rotation_world)
    force_world = np.zeros((6, 3), dtype=np.float64)
    raw: dict[str, list[Any]] = {
        key: []
        for key in (
            "contact_id",
            "geom1",
            "geom2",
            "segment",
            "leg",
            "sign",
            "frame",
            "wrench",
            "position_world",
        )
    }
    for contact_id in range(count):
        identity = classify_contact(
            int(first[contact_id]),
            int(second[contact_id]),
            bool(omit[contact_id]),
            feet_mapping,
            support_geom_ids,
        )
        if identity is None:
            continue
        segment, leg, sign = identity
        value = contact_force_world(
            frames[contact_id], wrenches[contact_id], sign, weight0
        )
        force_world[leg] += value
        for key, item in (
            ("contact_id", contact_id),
            ("geom1", int(first[contact_id])),
            ("geom2", int(second[contact_id])),
            ("segment", segment),
            ("leg", leg),
            ("sign", sign),
            ("frame", frames[contact_id].copy()),
            ("wrench", wrenches[contact_id].copy()),
            ("position_world", positions[contact_id].copy()),
        ):
            raw[key].append(item)
    arrays = {
        "contact_id": np.asarray(raw["contact_id"], dtype=np.int64),
        "geom1": np.asarray(raw["geom1"], dtype=np.int64),
        "geom2": np.asarray(raw["geom2"], dtype=np.int64),
        "segment": np.asarray(raw["segment"], dtype=np.int64),
        "leg": np.asarray(raw["leg"], dtype=np.int64),
        "sign": np.asarray(raw["sign"], dtype=np.int8),
        "frame": np.asarray(raw["frame"], dtype=np.float64).reshape(-1, 3, 3),
        "wrench": np.asarray(raw["wrench"], dtype=np.float64).reshape(-1, 6),
        "position_world": np.asarray(raw["position_world"], dtype=np.float64).reshape(
            -1, 3
        ),
    }
    return force_world, force_world @ heading, arrays


def expand_control_mask(
    control_mask: Any, samples_per_control: int = PHYSICS_PER_CONTROL
) -> np.ndarray:
    selected = np.asarray(control_mask)
    require(
        selected.ndim == 1
        and selected.dtype == np.dtype(bool)
        and type(samples_per_control) is int
        and samples_per_control > 0,
        "Boolean control mask and positive fixed expansion required",
    )
    return np.repeat(selected, samples_per_control)


def aggregate_force_group(
    selected: Any,
    force_forward_bw: Any,
    normal_load_bw: Any,
    slip_mm_s: Any,
    slip_valid: Any,
    *,
    dt: float = PHYSICS_DT,
) -> dict[str, Any]:
    force = np.asarray(force_forward_bw, dtype=np.float64)
    load = np.asarray(normal_load_bw, dtype=np.float64)
    slip = np.asarray(slip_mm_s, dtype=np.float64)
    valid = np.asarray(slip_valid)
    mask = np.asarray(selected)
    rows = len(force)
    require(
        force.shape == load.shape == slip.shape == (rows, 6)
        and mask.shape == (rows,)
        and mask.dtype == np.dtype(bool)
        and valid.shape == (rows, 6)
        and valid.dtype == np.dtype(bool)
        and np.isfinite(force).all()
        and np.isfinite(load).all()
        and np.all(load >= 0)
        and np.isfinite(dt)
        and dt > 0,
        "Invalid signed-force aggregation arrays",
    )
    require(
        np.array_equal(valid, np.isfinite(slip)),
        "slip_valid differs from finite saved slip",
    )
    support = load >= SUPPORT_BW
    require(not np.any(support & ~valid), "Supported slip is missing")
    if not mask.any():
        return {"sample_count": 0, "legs": None, "total": None}
    chosen_force = force[mask]
    chosen_load = load[mask]
    chosen_slip = slip[mask]
    chosen_support = support[mask]
    legs = {}
    for leg in range(6):
        supported = chosen_support[:, leg]
        denominator = float(chosen_load[supported, leg].sum())
        numerator = float(
            np.sum(chosen_load[supported, leg] * chosen_slip[supported, leg])
        )
        legs[LEGS[leg].upper()] = {
            "mean_force_forward_bw": float(chosen_force[:, leg].mean()),
            "positive_impulse_bw_s": float(
                np.maximum(chosen_force[:, leg], 0).sum() * dt
            ),
            "negative_impulse_bw_s": float(
                np.minimum(chosen_force[:, leg], 0).sum() * dt
            ),
            "supported_negative_fraction": float(
                np.mean(chosen_force[supported, leg] < 0)
            )
            if supported.any()
            else None,
            "mean_normal_load_bw": float(chosen_load[:, leg].mean()),
            "load_weighted_slip_mm_s": numerator / denominator
            if denominator > 0
            else None,
            "supported_samples": int(supported.sum()),
        }
    total_force = chosen_force.sum(axis=1)
    flat_support = chosen_support
    total_denominator = float(chosen_load[flat_support].sum())
    total_numerator = float(
        np.sum(chosen_load[flat_support] * chosen_slip[flat_support])
    )
    total = {
        "mean_force_forward_bw": float(total_force.mean()),
        "positive_impulse_bw_s": float(np.maximum(total_force, 0).sum() * dt),
        "negative_impulse_bw_s": float(np.minimum(total_force, 0).sum() * dt),
        "supported_negative_fraction": float(np.mean(chosen_force[flat_support] < 0))
        if flat_support.any()
        else None,
        "mean_normal_load_bw": float(chosen_load.sum(axis=1).mean()),
        "mean_normal_load_bw_per_leg": float(chosen_load.mean()),
        "load_weighted_slip_mm_s": total_numerator / total_denominator
        if total_denominator > 0
        else None,
        "supported_samples": int(flat_support.sum()),
        "supported_negative_fraction_denominator": "supported leg samples across six legs",
    }
    return {"sample_count": int(mask.sum()), "legs": legs, "total": total}


def aggregate_signed_contacts(
    force_forward_bw: Any,
    normal_load_bw: Any,
    slip_mm_s: Any,
    slip_valid: Any,
    endpoint_speed_mm_s: Any,
) -> dict[str, Any]:
    force = np.asarray(force_forward_bw, dtype=np.float64)
    speed = np.asarray(endpoint_speed_mm_s, dtype=np.float64)
    require(
        force.ndim == 2
        and force.shape[1] == 6
        and len(force) == len(speed) * PHYSICS_PER_CONTROL
        and np.isfinite(speed).all(),
        "Exact 50:1 native force/control speed alignment required",
    )
    masks = {
        "all": np.ones(len(force), dtype=bool),
        "forward": expand_control_mask(speed > 0.5),
        "interrupted": expand_control_mask(speed <= 0.5),
    }
    return {
        name: aggregate_force_group(mask, force, normal_load_bw, slip_mm_s, slip_valid)
        for name, mask in masks.items()
    }


def validate_body_parameters(saved: dict[str, Any]) -> dict[str, Any]:
    parameters = saved.get("parameters")
    require(isinstance(parameters, dict), "Checkpoint parameters are required")
    for key in (
        "continuous_joint_activation",
        "active_length",
        "finite_pad_contact",
        "adhesion_unit_conversion",
    ):
        require(
            key not in parameters or type(parameters[key]) is bool,
            "Checkpoint body parameter must be Boolean: " + key,
        )
    adhesion = parameters.get("adhesion_model", "legacy")
    require(
        adhesion in ("legacy", "feasible_lift", "activation", "graded", "request_lift"),
        "Unsupported checkpoint adhesion model",
    )
    tau = parameters.get("joint_activation_tau_s", 0.03)
    require(
        np.isfinite(tau) and tau > 0, "Positive finite joint activation tau required"
    )
    return {
        "continuous_joint_activation": parameters.get(
            "continuous_joint_activation", False
        ),
        "joint_activation_tau_s": tau,
        "active_length": parameters.get("active_length", False),
        "finite_pad_contact": parameters.get("finite_pad_contact", False),
        "adhesion_unit_conversion": parameters.get("adhesion_unit_conversion", False),
        "adhesion_model": adhesion,
        "graded_adhesion": adhesion == "graded",
    }


def validate_settling_count(t0: float, physics_dt: float, expected_steps: int) -> int:
    require(
        np.isfinite(t0)
        and t0 >= 0
        and np.isfinite(physics_dt)
        and physics_dt > 0
        and type(expected_steps) is int
        and expected_steps >= 0,
        "Finite settling clock contract required",
    )
    steps = round(t0 / physics_dt)
    require(
        steps == expected_steps and abs(t0 - expected_steps * physics_dt) <= 1e-12,
        "Body neutral settling count differs",
    )
    return steps


def construct_body_safely(
    saved: dict[str, Any], observation_module: Any, counters: dict[str, Any]
) -> Any:
    """Call the unchanged new_body while retaining a partial instance for cleanup."""
    original_class = observation_module.FlyGymBody
    partial = None

    def tracked_constructor(*args: Any, **kwargs: Any) -> Any:
        nonlocal partial
        counters["body_constructor_attempts"] += 1
        partial = original_class.__new__(original_class)
        original_settling_step = partial._step_physics

        def counted_settling_step() -> Any:
            value = original_settling_step()
            counters["settling_physics_steps"] += 1
            return value

        partial._step_physics = counted_settling_step
        try:
            original_class.__init__(partial, *args, **kwargs)
        except Exception as primary:
            try:
                partial._step_physics = original_settling_step
            except Exception as restore_error:  # noqa: BLE001
                primary.add_note(
                    "Secondary E153 constructor-step restore failure: "
                    + repr(restore_error)
                )
            raise
        partial._step_physics = original_settling_step
        counters["body_constructor_completed"] += 1
        return partial

    observation_module.FlyGymBody = tracked_constructor
    try:
        return observation_module.new_body(saved)
    except Exception as primary:
        if partial is not None:
            try:
                partial.close()
            except Exception as cleanup_error:  # noqa: BLE001
                primary.add_note(
                    "Secondary E153 partial-body cleanup failure: "
                    + repr(cleanup_error)
                )
        raise
    finally:
        observation_module.FlyGymBody = original_class


def advance_scientific_step(step: Any, counters: dict[str, Any]) -> None:
    """Count a completed actor integration before any observation can fail."""
    step()
    counters["scientific_physics_steps_advanced"] += 1
    counters["scientific_physics_steps"] = counters["scientific_physics_steps_advanced"]


def mark_scientific_observed(counters: dict[str, Any]) -> None:
    counters["scientific_samples_observed"] += 1


def mark_scientific_verified(counters: dict[str, Any]) -> None:
    counters["scientific_samples_verified"] += 1


def implicit_request_for_trace(
    payload: dict[str, Any], saved: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve only the frozen E150 no-event protocol, never generic missing data."""
    rows = payload.get("samples")
    require(isinstance(rows, list) and bool(rows), "Recorded command rows required")
    if all(row.get("neural_request") is not None for row in rows):
        return None
    require(
        payload.get("schema") == "flylab.m3.e150.native-monotone-body-trace.v1"
        and payload.get("E150_provenance", {}).get("numeric_protocol")
        == "unchanged E141 native body protocol"
        and len(rows) == 2001
        and all(
            "neural_request" in row and row["neural_request"] is None for row in rows
        ),
        "Null requests require the complete frozen E150 no-event protocol",
    )
    cuts = saved.get("cuts", {})
    body = saved.get("body", {})
    push = np.asarray(body.get("push_force"), dtype=np.float64)
    require(
        set(cuts) == {"circuit", "descending", "motor", "sensory"}
        and all(value is False for value in cuts.values())
        and body.get("push_left") == 0.0
        and push.shape == (3,)
        and np.array_equal(push, np.zeros(3)),
        "Implicit E150 requests require an uncut checkpoint with zero external force",
    )
    return {
        "cuts": copy.deepcopy(cuts),
        "external_force_at_start": {"push_left_s": 0.0, "push_force": [0.0] * 3},
    }


def replay_command(
    row: dict[str, Any], *, implicit_request: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    request = row.get("neural_request")
    if request is None and implicit_request is not None:
        request = implicit_request
    require(
        isinstance(request, dict)
        and isinstance(request.get("external_force_at_start"), dict)
        and isinstance(request.get("cuts"), dict)
        and type(request["cuts"].get("motor")) is bool,
        "Complete recorded external force and motor cut required",
    )
    push = request["external_force_at_start"]
    require(
        {"push_left_s", "push_force"} <= set(push),
        "Recorded external force fields missing",
    )
    push_left = float(push["push_left_s"])
    push_force = np.asarray(push["push_force"], dtype=np.float64)
    require(
        np.isfinite(push_left)
        and push_left >= 0
        and push_force.shape == (3,)
        and np.isfinite(push_force).all(),
        "Finite recorded external force required",
    )
    joint = np.asarray(row.get("joint_targets_rad"), dtype=np.float64)
    adhesion = np.asarray(row.get("adhesion"))
    require(
        joint.shape == (42,)
        and np.isfinite(joint).all()
        and adhesion.shape == (6,)
        and adhesion.dtype == np.dtype(bool)
        and isinstance(row.get("tendon_inputs"), dict)
        and all(
            np.asarray(value).ndim == 0 and np.isfinite(value)
            for value in row["tendon_inputs"].values()
        ),
        "Complete finite recorded body command required",
    )
    kwargs: dict[str, Any] = {
        "tendon_inputs": copy.deepcopy(row["tendon_inputs"]),
        "interpolate": not request["cuts"]["motor"],
    }
    if "joint_stiffness_scale" in row:
        kwargs["joint_stiffness_scale"] = copy.deepcopy(row["joint_stiffness_scale"])
    if "adhesion_control" in row:
        kwargs["adhesion_control"] = copy.deepcopy(row["adhesion_control"])
    return (
        joint,
        adhesion.copy(),
        {
            "push_left_s": push_left,
            "push_force": push_force.copy(),
            "step_kwargs": kwargs,
        },
    )


def new_raw_registry() -> dict[str, Any]:
    return {
        "native": {
            key: []
            for key in (
                "time_s",
                "control_tick",
                "rotation_world",
                "heading_horizontal",
                "force_world_bw",
                "force_forward_bw",
                "normal_load_bw",
                "slip_mm_s",
                "slip_valid",
                "nonfoot_contact",
                "upright",
            )
        },
        "contacts": {
            key: []
            for key in (
                "contact_id",
                "geom1",
                "geom2",
                "segment",
                "leg",
                "sign",
                "frame",
                "wrench",
                "position_world",
            )
        },
        "offsets": [0],
        "controls": {
            key: []
            for key in (
                "tick",
                "time_s",
                "qpos",
                "qvel",
                "ctrl",
                "verified",
            )
        },
    }


def append_native_raw(
    registry: dict[str, Any],
    *,
    time_s: float,
    control_tick: int,
    rotation_world: np.ndarray,
    force_world_bw: np.ndarray,
    normal_load_bw: np.ndarray,
    slip_mm_s: np.ndarray,
    slip_valid: np.ndarray,
    nonfoot_contact: bool,
    upright: float,
    contacts: dict[str, np.ndarray],
) -> None:
    native = registry["native"]
    heading = horizontal_heading(rotation_world)
    values = {
        "time_s": float(time_s),
        "control_tick": int(control_tick),
        "rotation_world": np.array(rotation_world, copy=True),
        "heading_horizontal": heading,
        "force_world_bw": np.array(force_world_bw, copy=True),
        "force_forward_bw": np.asarray(force_world_bw) @ heading,
        "normal_load_bw": np.array(normal_load_bw, copy=True),
        "slip_mm_s": np.array(slip_mm_s, copy=True),
        "slip_valid": np.array(slip_valid, copy=True),
        "nonfoot_contact": bool(nonfoot_contact),
        "upright": float(upright),
    }
    for key, value in values.items():
        native[key].append(value)
    count = len(contacts["contact_id"])
    for key in registry["contacts"]:
        registry["contacts"][key].extend(np.asarray(contacts[key]))
    registry["offsets"].append(registry["offsets"][-1] + count)


def append_control_raw(
    registry: dict[str, Any], body: Any, tick: int, verified: bool
) -> None:
    controls = registry["controls"]
    for key, value in (
        ("tick", int(tick)),
        ("time_s", float(body.physics_time())),
        ("qpos", np.array(body.d.qpos, copy=True)),
        ("qvel", np.array(body.d.qvel, copy=True)),
        ("ctrl", np.array(body.d.ctrl, copy=True)),
        ("verified", bool(verified)),
    ):
        controls[key].append(value)


def raw_arrays(
    registry: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    native_shapes = {
        "rotation_world": (3, 3),
        "heading_horizontal": (3,),
        "force_world_bw": (6, 3),
        "force_forward_bw": (6,),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
        "slip_valid": (6,),
    }
    native = {}
    for key, values in registry["native"].items():
        array = np.asarray(values)
        if key in native_shapes:
            array = array.reshape(-1, *native_shapes[key])
        native[key] = array
    contact_shapes = {"frame": (3, 3), "wrench": (6,), "position_world": (3,)}
    contacts = {"offsets": np.asarray(registry["offsets"], dtype=np.int64)}
    for key, values in registry["contacts"].items():
        array = np.asarray(values)
        if key in contact_shapes:
            array = array.reshape(-1, *contact_shapes[key])
        contacts[key] = array
    controls = {key: np.asarray(values) for key, values in registry["controls"].items()}
    return native, contacts, controls


def save_raw(output: Path, registry: dict[str, Any]) -> None:
    native, contacts, controls = raw_arrays(registry)
    atomic_npz(output / "native-forces.npz", **native)
    atomic_npz(output / "contacts.npz", **contacts)
    atomic_npz(output / "replay-controls.npz", **controls)


def _array_equal_nan(actual: Any, expected: Any) -> bool:
    return np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True)


def _observe_shadow(
    body: Any, shadow: Any
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    count = int(shadow.d.ncon)
    contacts = shadow.d.contact[:count]
    geom1 = np.asarray([int(value.geom1) for value in contacts], dtype=np.int64)
    geom2 = np.asarray([int(value.geom2) for value in contacts], dtype=np.int64)
    excluded = np.asarray([bool(value.exclude) for value in contacts], dtype=bool)
    frames = np.asarray(
        [np.asarray(value.frame).reshape(3, 3) for value in contacts], dtype=np.float64
    ).reshape(-1, 3, 3)
    positions = np.asarray(
        [np.asarray(value.pos) for value in contacts], dtype=np.float64
    ).reshape(-1, 3)
    wrenches = np.empty((count, 6), dtype=np.float64)
    for index in range(count):
        body.mj.mj_contactForce(body.m, shadow.d, index, wrenches[index])
    mapping, _ = body._force_maps["feet"]
    _, rotation, _ = shadow.pose()
    force, _, raw = transform_contacts(
        geom1,
        geom2,
        excluded,
        frames,
        wrenches,
        positions,
        mapping,
        body.support_geom_ids,
        float(body.weight0),
        rotation,
    )
    return force, raw, rotation


def output_manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and item.name != "output-manifest.json"
    ):
        files[path.relative_to(output).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "flylab.m3.e153.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def run(
    freeze_path: Path,
    freeze_sha: str,
    source_review: Path,
    source_review_sha: str,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    freeze = load_freeze(freeze_path, freeze_sha)
    limit_s = float(freeze["resource_limits"]["internal_wall_s"])
    validate_source_gate(source_review, source_review_sha)
    verify_hash_map(freeze["input_sha256"])
    verify_hash_map(freeze["reused_source_sha256"])
    verify_hash_map(freeze["preservation_sha256"])
    require(not output.exists(), "E153 output must be new")
    require(
        shutil.disk_usage(output.parent).free
        >= freeze["resource_limits"]["free_bytes"],
        "E153 free-space reserve unavailable",
    )
    pools = threadpool_info()
    require(
        all(item.get("num_threads") == 1 for item in pools),
        "Actual CPU threadpool exceeds one thread",
    )
    deadline(started, limit_s, "preflight")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA,
        "accepted": False,
        "publication_complete": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "E150_original_still_incomplete": True,
        "interpretation_limit": (
            "Exploratory contact-level signed force during an exact saved-command replay; "
            "not causal isolation of motor neurons, muscles, sensory feedback, or gait quality"
        ),
    }
    atomic_json(output / "result.json", result)
    append_progress(output / "progress.jsonl", "preflight_complete", started)
    registry = new_raw_registry()
    context: dict[str, Any] = {}
    body = None
    original_step = None
    primary: Exception | None = None
    secondary: list[str] = []
    ledger = {
        "original_processes": 1,
        "automatic_retries": 0,
        "CPU_threads": 1,
        "GPU": 0,
        "network_models": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "body_models": 0,
        "shadow_mjdata": 0,
        "settling_physics_steps": 0,
        "settling_clock_derived_steps": 0,
        "scientific_physics_steps": 0,
        "scientific_physics_steps_advanced": 0,
        "scientific_samples_observed": 0,
        "scientific_samples_verified": 0,
        "shadow_forward_calls": 0,
        "trace_reads": 0,
        "native_reads": 0,
        "features_reads": 0,
        "checkpoint_reads": 0,
        "thread_pools": pools,
    }
    try:
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import contact_diagnostics
        from tools import analyze_c_banc_observation as observation

        trace_path = safe_path(freeze["trace"])
        native_path = safe_path(freeze["native"])
        features_path = safe_path(freeze["features"])
        checkpoint_path = safe_path(freeze["checkpoint"])
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        ledger["trace_reads"] = 1
        rows = payload.get("samples")
        require(
            isinstance(rows, list) and len(rows) > 400,
            "Complete E150 control recording required",
        )
        selected_rows = rows[200:401]
        require(
            [row.get("control_tick") for row in selected_rows] == list(range(200, 401)),
            "E153 control ticks 200..400 required",
        )
        control_time = np.asarray(
            [row.get("time_s") for row in selected_rows], dtype=np.float64
        )
        require(
            np.allclose(
                control_time, np.arange(200, 401) * 0.005, atol=CLOCK_ATOL, rtol=0
            ),
            "E153 control clock differs",
        )
        require(
            payload.get("body_hash") == freeze["body_hash"], "E153 body hash differs"
        )
        with np.load(native_path, allow_pickle=False) as stored:
            native_source = {key: stored[key] for key in stored.files}
        ledger["native_reads"] = 1
        with np.load(features_path, allow_pickle=False) as stored:
            features = {key: stored[key] for key in stored.files}
        ledger["features_reads"] = 1
        expected_native = {
            key: np.asarray(native_source[key])[10000:20000]
            for key in (
                "time_s",
                "normal_load_bw",
                "slip_mm_s",
                "slip_valid",
                "nonfoot_contact",
                "upright",
            )
        }
        require(
            all(len(value) == 10000 for value in expected_native.values()),
            "Complete original E150 native slice required",
        )
        require(
            features.get("time_s") is not None
            and features.get("speed_mm_s") is not None
            and np.array_equal(features["time_s"][201:401], control_time[1:]),
            "E152 feature endpoints differ from replay controls",
        )
        endpoint_speed = np.asarray(features["speed_mm_s"][201:401], dtype=np.float64)
        require(
            endpoint_speed.shape == (200,) and np.isfinite(endpoint_speed).all(),
            "Finite 200 endpoint speeds required",
        )
        saved = StateStore.load(checkpoint_path)
        ledger["checkpoint_reads"] = 1
        parameter_context = validate_body_parameters(saved)
        require(
            saved.get("control_tick") == 200
            and saved.get("body", {}).get("modelHash") == freeze["body_hash"],
            "Checkpoint tick/body identity differs",
        )
        implicit_request = implicit_request_for_trace(payload, saved)
        prepared_commands = [
            replay_command(row, implicit_request=implicit_request)
            for row in selected_rows[1:]
        ]
        ledger["implicit_request_controls"] = (
            len(prepared_commands) if implicit_request is not None else 0
        )
        deadline(started, limit_s, "before body construction")
        body = construct_body_safely(saved, observation, ledger)
        ledger["body_models"] = 1
        actual_dt = float(body.m.opt.timestep)
        require(abs(actual_dt - PHYSICS_DT) <= 1e-15, "Actual physics dt differs")
        settling_clock_steps = validate_settling_count(
            float(body.t0),
            actual_dt,
            freeze["replay"]["expected_settling_physics_steps"],
        )
        ledger["settling_clock_derived_steps"] = settling_clock_steps
        require(
            ledger["body_constructor_attempts"] == 1
            and ledger["body_constructor_completed"] == 1
            and ledger["settling_physics_steps"]
            == freeze["replay"]["expected_settling_physics_steps"],
            "Measured body construction/settling count differs",
        )
        require(
            abs(float(body.physics_time()) - 1.0) <= CLOCK_ATOL,
            "Restored checkpoint physics time differs",
        )
        mapping, _ = body._force_maps["feet"]
        context = {
            "feet_map": np.asarray(mapping, dtype=np.int64).tolist(),
            "support_geom_ids": sorted(int(value) for value in body.support_geom_ids),
            "weight0": float(body.weight0),
            "body_hash": freeze["body_hash"],
            "physics_dt_s": actual_dt,
            "source_native_slice": [10000, 20000],
            "source_control_ticks": [200, 400],
            "force_axis_convention": (
                "force on foot: +frame.T@wrench when foot is geom2, negative when geom1; "
                "divided by weight0; forward uses normalized horizontal rotation_world[:,0]"
            ),
            "body_parameters": parameter_context,
            "neural_request_resolution": (
                "E150 trace requests are null; no-cut/no-push values are derived from "
                "its unchanged E141 main protocol and uncut zero-force checkpoint. "
                "Every body state and native contact must still match the original."
                if implicit_request is not None
                else "Explicit per-control recorded requests"
            ),
        }
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        ledger["shadow_mjdata"] = 1
        state_type = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, state_type))
        original_step = body._step_physics
        native_index = 0
        active_tick = 201

        def observed_step() -> None:
            nonlocal native_index
            advance_scientific_step(original_step, ledger)
            body.mj.mj_getState(body.m, body.d, state, state_type)
            body.mj.mj_setState(body.m, shadow.d, state, state_type)
            body.mj.mj_forward(body.m, shadow.d)
            ledger["shadow_forward_calls"] += 1
            force_world, raw_contacts, rotation = _observe_shadow(body, shadow)
            diagnostic = contact_diagnostics(shadow)
            load = np.asarray(diagnostic["normal_load_bw"], dtype=np.float64)
            slip = np.asarray(
                [
                    np.nan if value is None else value
                    for value in diagnostic["slip_mm_s"]
                ],
                dtype=np.float64,
            )
            valid = np.isfinite(slip)
            sample_time = float(body.physics_time())
            nonfoot = bool(shadow.nonfoot_contact())
            upright = float(rotation[2, 2])
            append_native_raw(
                registry,
                time_s=sample_time,
                control_tick=active_tick,
                rotation_world=rotation,
                force_world_bw=force_world,
                normal_load_bw=load,
                slip_mm_s=slip,
                slip_valid=valid,
                nonfoot_contact=nonfoot,
                upright=upright,
                contacts=raw_contacts,
            )
            mark_scientific_observed(ledger)
            for key, actual in (
                ("time_s", sample_time),
                ("normal_load_bw", load),
                ("slip_mm_s", slip),
                ("slip_valid", valid),
                ("nonfoot_contact", nonfoot),
                ("upright", upright),
            ):
                require(
                    _array_equal_nan(actual, expected_native[key][native_index]),
                    f"Original native replay mismatch at {native_index}: {key}",
                )
            mark_scientific_verified(ledger)
            native_index += 1

        body._step_physics = observed_step
        anchor = selected_rows[0]
        append_control_raw(registry, body, 200, False)
        for key in ("qpos", "qvel", "ctrl"):
            require(
                np.array_equal(
                    np.asarray(getattr(body.d, key)), np.asarray(anchor[key])
                ),
                "Checkpoint anchor differs: " + key,
            )
        require(
            abs(float(body.physics_time()) - float(anchor["time_s"])) <= CLOCK_ATOL,
            "Checkpoint anchor clock differs",
        )
        registry["controls"]["verified"][-1] = True
        append_progress(output / "progress.jsonl", "anchor_verified", started)
        for row, (joint, adhesion, command) in zip(
            selected_rows[1:], prepared_commands, strict=True
        ):
            active_tick = int(row["control_tick"])
            body.push_left = command["push_left_s"]
            body.push_force = command["push_force"]
            body.step_joint_targets(joint, adhesion, **command["step_kwargs"])
            append_control_raw(registry, body, active_tick, False)
            for key in ("qpos", "qvel", "ctrl"):
                require(
                    np.array_equal(
                        np.asarray(getattr(body.d, key)), np.asarray(row[key])
                    ),
                    f"Replay control {active_tick} differs: {key}",
                )
            require(
                abs(float(body.physics_time()) - float(row["time_s"])) <= CLOCK_ATOL,
                f"Replay control {active_tick} clock differs",
            )
            registry["controls"]["verified"][-1] = True
            require(not body.fault, "Body replay fault: " + str(body.fault))
            deadline(started, limit_s, "scientific replay")
        require(
            native_index == 10000
            and ledger["scientific_physics_steps_advanced"] == 10000
            and ledger["scientific_samples_observed"] == 10000
            and ledger["scientific_samples_verified"] == 10000
            and ledger["shadow_forward_calls"] == 10000,
            "Exact 10000-step scientific replay required",
        )
        save_raw(output, registry)
        native_arrays, _, control_arrays = raw_arrays(registry)
        require(
            len(control_arrays["tick"]) == 201 and np.all(control_arrays["verified"]),
            "Anchor plus 200 verified replay controls required",
        )
        summary = aggregate_signed_contacts(
            native_arrays["force_forward_bw"],
            native_arrays["normal_load_bw"],
            native_arrays["slip_mm_s"],
            native_arrays["slip_valid"],
            endpoint_speed,
        )
        atomic_json(
            output / "summary.json",
            {
                "schema": "flylab.m3.e153.signed-contact-summary.v1",
                "exploratory_only": True,
                "formal_F4_F5_unchanged": True,
                "groups": summary,
            },
        )
        atomic_json(output / "context.json", context)
        body._step_physics = original_step
        original_step = None
        body.close()
        body = None
        verify_hash_map(freeze["input_sha256"])
        verify_hash_map(freeze["reused_source_sha256"])
        verify_hash_map(freeze["preservation_sha256"])
        deadline(started, limit_s, "post-preservation verification")
        atomic_json(output / "ledger.json", ledger)
        atomic_json(
            output / "timing.json",
            {
                "wall_s_before_final_publication": time.perf_counter() - started,
                "internal_limit_s": limit_s,
                "external_limit_s": freeze["resource_limits"]["external_wall_s"],
                "supervisor_limit_s": freeze["resource_limits"]["supervisor_wall_s"],
                "prior_actual_s": freeze["resource_limits"]["prior_actual_s"],
                "prospective_cumulative_s": freeze["resource_limits"][
                    "prospective_cumulative_s"
                ],
            },
        )
        result.update(
            {
                "accepted": True,
                "publication_complete": True,
                "resource_limit_met": True,
                "verified_controls": 201,
                "scientific_physics_steps": 10000,
                "settling_physics_steps": 2000,
            }
        )
        append_progress(output / "progress.jsonl", "complete", started)
        atomic_json(output / "result.json", result)
        deadline(started, limit_s, "final result publication")
        manifest = output_manifest(output)
        require(
            manifest["total_bytes"] <= freeze["resource_limits"]["raw_bytes"],
            "E153 output exceeded raw byte limit",
        )
        atomic_json(output / "output-manifest.json", manifest)
        deadline(started, limit_s, "manifest publication")
        return result
    except Exception as error:  # noqa: BLE001 - preserve the exact primary failure
        primary = error
        result.update(
            {
                "accepted": False,
                "publication_complete": False,
                "resource_limit_met": (time.perf_counter() - started) <= limit_s,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
    finally:
        if body is not None and original_step is not None:
            try:
                body._step_physics = original_step
            except Exception as cleanup_error:  # noqa: BLE001
                secondary.append("restore_step: " + repr(cleanup_error))
        if body is not None:
            try:
                body.close()
            except Exception as cleanup_error:  # noqa: BLE001
                secondary.append("body_close: " + repr(cleanup_error))
        if primary is not None:
            for operation, action in (
                ("raw", lambda: save_raw(output, registry)),
                ("context", lambda: atomic_json(output / "context.json", context)),
                ("ledger", lambda: atomic_json(output / "ledger.json", ledger)),
                (
                    "timing",
                    lambda: atomic_json(
                        output / "timing.json",
                        {
                            "wall_s": time.perf_counter() - started,
                            "internal_limit_s": limit_s,
                        },
                    ),
                ),
                (
                    "error",
                    lambda: atomic_json(
                        output / "error.json",
                        {
                            "accepted": False,
                            "primary": result["error"],
                            "traceback": "".join(traceback.format_exception(primary)),
                        },
                    ),
                ),
            ):
                try:
                    action()
                except Exception as publication_error:  # noqa: BLE001
                    secondary.append(operation + ": " + repr(publication_error))
            result["resource_limit_met"] = (time.perf_counter() - started) <= limit_s
            try:
                atomic_json(output / "result.json", result)
                if time.perf_counter() - started > limit_s:
                    result["resource_limit_met"] = False
                    atomic_json(output / "result.json", result)
            except Exception as publication_error:  # noqa: BLE001
                secondary.append("result: " + repr(publication_error))
            for note in secondary:
                primary.add_note("Secondary E153 cleanup/publication failure: " + note)
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    raise RuntimeError("Unreachable E153 state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.freeze,
                arguments.freeze_sha256,
                arguments.source_review,
                arguments.source_review_sha256,
                arguments.output,
            ),
            indent=2,
        )
    )
