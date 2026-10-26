"""E152 read-only analysis of forward interruption and physical foot support.

The runner consumes only frozen trace/contact recordings.  It does not create a
graph, neural model, CUDA context, body, fit, or simulation step.
"""

# Thread limits must precede numerical imports.

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_info

from tools.analyze_c_banc_biological_gait import (
    LEGS,
    analyze,
    event_summary,
    load_model,
    phase_and_events,
    phase_summary,
)

FREEZE_SHA = "634f5baa389ed3ab776fecf1bc4b795c5eec52d3b516874ba27102718e342abf"
DESIGN_SHA = "bdfc310667cbcf6c22af4b1bc76762276d726a5187720fcf500531e5ebd447d2"
SCHEMA = "flylab.m3-e152-interrupted-support.v1"
OWNED = (
    "tools/analyze_c_banc_m3_interrupted_support.py",
    "tests/test_c_banc_interrupted_support.py",
)
DEFAULT_FREEZE = (
    ROOT
    / "verification/banc-natural-gait-m3-20260919/e152-interrupted-support-freeze.json"
)
DEFAULT_OUTPUT = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e152-interrupted-support"
)
SUPPORT_BW = 0.02
CONTROL_DT = 0.005
NATIVE_DT = 0.0001
NATIVE_PER_CONTROL = 50
CLOCK_ATOL = 2e-8
EVENT_RADIUS = 20


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
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_progress(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def verify_hash_map(entries: dict[str, str]) -> None:
    for relative, digest in entries.items():
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts,
            f"Unsafe frozen path: {relative}",
        )
        require(sha256(ROOT / path) == digest, f"Frozen hash mismatch: {relative}")


def load_freeze(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha == FREEZE_SHA, "E152 freeze SHA changed")
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == "flylab.m3-e152-interrupted-support-freeze.v1",
        "Unexpected E152 freeze schema",
    )
    require(
        freeze.get("design_sha256") == DESIGN_SHA
        and sha256(ROOT / freeze["design"]) == DESIGN_SHA,
        "E152 design changed",
    )
    require(
        freeze.get("owned_files") == list(OWNED), "E152 owned source contract changed"
    )
    require(
        freeze.get("not_adoption") is True
        and freeze.get("whole_goal_complete") is False
        and freeze.get("E150_original_still_incomplete") is True,
        "E152 outcome boundary changed",
    )
    return freeze


def validate_source_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha, "Parent E152 source-review SHA changed")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {relative: sha256(ROOT / relative) for relative in OWNED}
    require(
        review.get("accepts_single_saved_diagnostic") is True,
        "Parent did not accept one saved E152 diagnostic",
    )
    require(
        review.get("freeze_sha256") == FREEZE_SHA,
        "Parent source-review freeze SHA differs from E152 freeze",
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "Parent source-review must allow exactly one original process",
    )
    require(
        review.get("source_sha256") == expected,
        "Parent source-review SHA map differs from current E152 sources",
    )
    return review


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    require(
        result.shape == shape and np.isfinite(result).all(),
        f"Invalid finite {name}: {result.shape}",
    )
    return result


def native_blocks(
    time_s: Any,
    load_bw: Any,
    slip_mm_s: Any,
    slip_valid: Any | None,
    *,
    samples_per_control: int = NATIVE_PER_CONTROL,
    native_dt: float = NATIVE_DT,
    strict_rows: bool = False,
) -> tuple[dict[str, np.ndarray], str]:
    """Map native rows exactly to endpoint blocks (t-control_dt,t]."""
    time_array = np.asarray(time_s, dtype=np.float64)
    load = np.asarray(load_bw, dtype=np.float64)
    slip = np.asarray(slip_mm_s, dtype=np.float64)
    expected_rows = 100000 if strict_rows else len(time_array)
    require(
        time_array.shape == (expected_rows,)
        and load.shape == (expected_rows, 6)
        and slip.shape == (expected_rows, 6),
        "Native time/load/slip shape mismatch",
    )
    require(
        np.isfinite(time_array).all() and np.isfinite(load).all() and np.all(load >= 0),
        "Native time/load must be finite and load nonnegative",
    )
    require(
        type(samples_per_control) is int
        and samples_per_control > 0
        and expected_rows % samples_per_control == 0,
        "Whole fixed native blocks required",
    )
    require(
        np.allclose(
            time_array,
            np.arange(1, expected_rows + 1) * native_dt,
            atol=CLOCK_ATOL,
            rtol=0,
        ),
        "Native clock is shifted, missing, or noncontiguous",
    )
    if slip_valid is None:
        valid = np.isfinite(slip)
        provenance = "derived_isfinite_E85"
    else:
        raw_valid = np.asarray(slip_valid)
        require(
            raw_valid.dtype == np.dtype(bool) and raw_valid.shape == (expected_rows, 6),
            "Recorded slip_valid must be Boolean native shape",
        )
        valid = raw_valid.copy()
        require(
            np.array_equal(valid, np.isfinite(slip)),
            "Recorded slip_valid disagrees with finite slip",
        )
        provenance = "recorded"
    supported = load >= SUPPORT_BW
    require(not np.any(supported & ~valid), "Supported native slip is missing")
    controls = expected_rows // samples_per_control
    shaped_load = load.reshape(controls, samples_per_control, 6)
    shaped_slip = slip.reshape(controls, samples_per_control, 6)
    shaped_support = supported.reshape(controls, samples_per_control, 6)
    denominator = np.sum(np.where(shaped_support, shaped_load, 0.0), axis=1)
    # Index supported entries directly: unsupported NaN never becomes zero slip.
    numerator = np.zeros((controls, 6), dtype=np.float64)
    for leg in range(6):
        for control in range(controls):
            selected = shaped_support[control, :, leg]
            numerator[control, leg] = np.sum(
                shaped_load[control, selected, leg]
                * shaped_slip[control, selected, leg]
            )
    support_count = shaped_support.sum(axis=2)
    result = {
        "endpoint_time_s": time_array[
            samples_per_control - 1 :: samples_per_control
        ].copy(),
        "mean_load_bw": shaped_load.mean(axis=1),
        "supported_duty": shaped_support.mean(axis=1),
        "supported_load_sum_bw": denominator,
        "supported_load_slip_sum_bw_mm_s": numerator,
        "support_count_mean": support_count.mean(axis=1),
        "support_count_lt3_fraction": (support_count < 3).mean(axis=1),
        "support_count_eq0_fraction": (support_count == 0).mean(axis=1),
    }
    return result, provenance


def foot_kinematics(
    position: Any, basis: Any, foot_world: Any, *, dt: float = CONTROL_DT
) -> dict[str, np.ndarray]:
    position_array = np.asarray(position, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    foot_array = np.asarray(foot_world, dtype=np.float64)
    rows = len(position_array)
    require(
        position_array.shape == (rows, 3)
        and basis_array.shape == (rows, 3, 3)
        and foot_array.shape == (rows, 6, 3)
        and rows >= 3
        and np.isfinite(position_array).all()
        and np.isfinite(basis_array).all()
        and np.isfinite(foot_array).all(),
        "Finite physical body and tarsus5 coordinates required",
    )
    require(
        np.allclose(
            np.einsum("tji,tjk->tik", basis_array, basis_array),
            np.eye(3),
            atol=1e-9,
            rtol=0,
        ),
        "Body basis is not orthogonal",
    )
    ego = np.einsum(
        "tlj,tjk->tlk", foot_array - position_array[:, None, :], basis_array
    )
    ego_velocity = gaussian_filter1d(
        np.gradient(ego[:, :, 0], dt, axis=0), 0.02 / dt, axis=0, mode="nearest"
    )
    world_vector = np.gradient(foot_array, dt, axis=0)
    world_forward = np.einsum("tlj,tj->tl", world_vector, basis_array[:, :, 0])
    world_forward = gaussian_filter1d(world_forward, 0.02 / dt, axis=0, mode="nearest")
    return {
        "ego_ap_mm": ego[:, :, 0],
        "ego_height_mm": ego[:, :, 2],
        "ego_ap_velocity_mm_s": ego_velocity,
        "world_forward_velocity_mm_s": world_forward,
    }


def tracking_rms_by_leg(target: Any, actual: Any) -> np.ndarray:
    requested = np.asarray(target, dtype=np.float64)
    measured = np.asarray(actual, dtype=np.float64)
    require(
        requested.shape == measured.shape
        and requested.ndim == 2
        and requested.shape[1] == 42
        and np.isfinite(requested).all()
        and np.isfinite(measured).all(),
        "Finite matching 42-joint traces required",
    )
    return np.sqrt(np.mean(np.square((requested - measured).reshape(-1, 6, 7)), axis=2))


def percentile_summary(values: Any, mask: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    require(
        array.ndim in (1, 2) and selected.shape == (len(array),),
        "Percentile values and selection differ",
    )
    chosen = array[selected]
    if array.ndim == 1:
        finite = chosen[np.isfinite(chosen)]
        return (
            None
            if not len(finite)
            else {
                "p05": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
            }
        )
    return [
        percentile_summary(array[:, column], selected)
        for column in range(array.shape[1])
    ]


def aggregate_native(
    blocks: dict[str, np.ndarray], endpoint_mask: Any
) -> dict[str, Any]:
    selected = np.asarray(endpoint_mask, dtype=bool)
    rows = len(blocks["endpoint_time_s"])
    require(selected.shape == (rows,), "Native endpoint mask shape differs")
    if not selected.any():
        return {
            "endpoints": 0,
            "per_leg": None,
            "overall": None,
            "support_count_mean": None,
            "support_count_lt3_fraction": None,
            "support_count_eq0_fraction": None,
        }
    denominator = blocks["supported_load_sum_bw"][selected].sum(axis=0)
    numerator = blocks["supported_load_slip_sum_bw_mm_s"][selected].sum(axis=0)
    slip = np.divide(
        numerator, denominator, out=np.full(6, np.nan), where=denominator > 0
    )
    total_denominator = float(denominator.sum())
    total_numerator = float(numerator.sum())
    per_leg = [
        {
            "mean_load_bw": float(blocks["mean_load_bw"][selected, leg].mean()),
            "supported_duty": float(blocks["supported_duty"][selected, leg].mean()),
            "supported_load_sum_bw": float(denominator[leg]),
            "supported_load_weighted_slip_mm_s": float(slip[leg])
            if np.isfinite(slip[leg])
            else None,
        }
        for leg in range(6)
    ]
    return {
        "endpoints": int(selected.sum()),
        "per_leg": per_leg,
        "overall": {
            "mean_load_bw": float(blocks["mean_load_bw"][selected].mean()),
            "supported_duty": float(blocks["supported_duty"][selected].mean()),
            "supported_load_sum_bw": total_denominator,
            "supported_load_weighted_slip_mm_s": total_numerator / total_denominator
            if total_denominator > 0
            else None,
        },
        "support_count_mean": float(blocks["support_count_mean"][selected].mean()),
        "support_count_lt3_fraction": float(
            blocks["support_count_lt3_fraction"][selected].mean()
        ),
        "support_count_eq0_fraction": float(
            blocks["support_count_eq0_fraction"][selected].mean()
        ),
    }


def aggregate_group(
    mask: Any,
    speed: Any,
    yaw_rate: Any,
    tracking: Any,
    ego_velocity: Any,
    world_velocity: Any,
    boundary_support: Any,
    blocks: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    support = np.asarray(boundary_support, dtype=bool)
    require(
        selected.shape == (len(speed),) and support.shape == (len(speed), 6),
        "Control group shapes differ",
    )
    forward = np.asarray(ego_velocity) > 0
    fractions = []
    for leg in range(6):
        denominator = int(np.sum(selected & support[:, leg]))
        fractions.append(
            float(np.sum(selected & support[:, leg] & forward[:, leg]) / denominator)
            if denominator
            else None
        )
    overall_denominator = int(np.sum(support[selected]))
    overall_numerator = int(np.sum(support[selected] & forward[selected]))
    return {
        "control_samples": int(selected.sum()),
        "speed_mm_s": percentile_summary(speed, selected),
        "yaw_rate_rad_s": percentile_summary(yaw_rate, selected),
        "tracking_RMS_rad_by_leg": percentile_summary(tracking, selected),
        "ego_AP_velocity_mm_s_by_leg": percentile_summary(ego_velocity, selected),
        "world_forward_velocity_mm_s_by_leg": percentile_summary(
            world_velocity, selected
        ),
        "boundary_supported_forward_ego_AP_fraction_by_leg": fractions,
        "boundary_supported_forward_ego_AP_fraction_overall": overall_numerator
        / overall_denominator
        if overall_denominator
        else None,
        "native": aggregate_native(blocks, selected[1:]),
    }


def support_patterns(
    boundary_support: Any, speed: Any, window: Any
) -> list[dict[str, Any]]:
    support = np.asarray(boundary_support, dtype=bool)
    speed_array = np.asarray(speed, dtype=np.float64)
    selected = np.asarray(window, dtype=bool)
    require(
        support.shape == (len(speed_array), 6)
        and selected.shape == (len(speed_array),),
        "Boundary pattern inputs differ",
    )
    code = np.sum(
        support.astype(np.uint8) * (1 << np.arange(6, dtype=np.uint8)), axis=1
    )
    result = []
    for pattern in range(64):
        values = speed_array[selected & (code == pattern)]
        values = values[np.isfinite(values)]
        result.append(
            {
                "pattern": pattern,
                "bits_LF_LM_LH_RF_RM_RH": [
                    bool(pattern & (1 << bit)) for bit in range(6)
                ],
                "samples": len(values),
                "speed_median_mm_s": float(np.median(values)) if len(values) else None,
            }
        )
    return result


def interruption_windows(
    speed: Any,
    window: Any,
    *,
    threshold: float = 0.5,
    radius: int = EVENT_RADIUS,
) -> dict[str, Any]:
    speed_array = np.asarray(speed, dtype=np.float64)
    selected = np.asarray(window, dtype=bool)
    require(
        speed_array.ndim == 1
        and selected.shape == speed_array.shape
        and np.isfinite(speed_array).all()
        and type(radius) is int
        and radius >= 1,
        "Finite interruption speed/window required",
    )
    crossings = (
        np.flatnonzero(
            selected[1:]
            & selected[:-1]
            & (speed_array[:-1] > threshold)
            & (speed_array[1:] <= threshold)
        )
        + 1
    )
    included: list[int] = []
    records = []
    for index in crossings:
        before = index - radius
        after = index + radius
        reasons = []
        if before < 0 or not np.all(selected[max(0, before) : index + 1]):
            reasons.append("pre_outside_window")
        if after >= len(selected) or not np.all(
            selected[index : min(len(selected), after + 1)]
        ):
            reasons.append("post_outside_window")
        if not reasons:
            included.append(int(index))
        records.append(
            {"index": int(index), "included": not reasons, "exclusion_reasons": reasons}
        )
    return {
        "all": records,
        "included_indices": np.asarray(included, dtype=np.int64),
        "offsets": np.arange(-radius, radius + 1, dtype=np.int64),
    }


def stack_events(
    features: dict[str, np.ndarray], events: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    indices = events["included_indices"]
    offsets = events["offsets"]
    stacks: dict[str, np.ndarray] = {
        "event_index": indices.copy(),
        "offset_control": offsets.copy(),
        "offset_s": offsets.astype(np.float64) * CONTROL_DT,
    }
    curve_summary: dict[str, Any] = {}
    individual = []
    for name, raw in features.items():
        array = np.asarray(raw)
        shape = (len(indices), len(offsets), *array.shape[1:])
        stack = np.empty(shape, dtype=array.dtype)
        for row, index in enumerate(indices):
            stack[row] = array[index + offsets]
        stacks[name] = stack
        if len(indices):
            curve_summary[name] = {
                "mean": np.mean(stack, axis=0).tolist(),
                "median": np.median(stack, axis=0).tolist(),
            }
    for row, index in enumerate(indices):
        record: dict[str, Any] = {
            "index": int(index),
            "pre_offsets": [-20, -1],
            "post_offsets": [0, 19],
            "values": {},
        }
        for name in features:
            stack = stacks[name][row]
            record["values"][name] = {
                "pre_mean": np.mean(stack[:20], axis=0).tolist(),
                "post_mean": np.mean(stack[20:40], axis=0).tolist(),
            }
        individual.append(record)
    return stacks, {"curves": curve_summary, "events": individual}


def validate_recording(
    payload: dict[str, Any],
    model: dict[str, Any],
    native: dict[str, np.ndarray],
    expected_body_hash: str,
) -> tuple[dict[str, np.ndarray], str, dict[str, np.ndarray]]:
    rows = payload.get("samples")
    require(
        isinstance(rows, list) and len(rows) == 2001, "Exactly 2001 trace rows required"
    )
    time_s = finite_array([row["time_s"] for row in rows], (2001,), "control clock")
    require(
        np.allclose(time_s, np.arange(2001) * CONTROL_DT, atol=CLOCK_ATOL, rtol=0)
        and np.array_equal([row["control_tick"] for row in rows], np.arange(2001)),
        "Control clock/ticks differ",
    )
    require(
        payload.get("body_hash") == expected_body_hash
        and model["source"]["body_hash"] == expected_body_hash,
        "Frozen body hash differs",
    )
    position = finite_array(
        [row["body"]["position"] for row in rows], (2001, 3), "position"
    )
    basis = finite_array([row["body"]["basis"] for row in rows], (2001, 3, 3), "basis")
    foot = finite_array(
        [[row["body"]["legs"][leg][-1] for leg in LEGS] for row in rows],
        (2001, 6, 3),
        "tarsus5",
    )
    target = finite_array(
        [row["joint_targets_rad"] for row in rows], (2001, 42), "target"
    )
    actual = finite_array(
        [row["joint_angles_rad"] for row in rows], (2001, 42), "actual"
    )
    qpos = [np.asarray(row["qpos"], dtype=np.float64) for row in rows]
    qvel = [np.asarray(row["qvel"], dtype=np.float64) for row in rows]
    require(
        all(value.shape == qpos[0].shape and np.isfinite(value).all() for value in qpos)
        and all(
            value.shape == qvel[0].shape and np.isfinite(value).all() for value in qvel
        ),
        "Finite fixed physical qpos/qvel required",
    )
    boundary_load = finite_array(
        [row["feet"]["normal_load_bw"] for row in rows], (2001, 6), "boundary load"
    )
    require(np.all(boundary_load >= 0), "Boundary load must be nonnegative")
    boundary_support = np.asarray([row["feet"]["load_bearing"] for row in rows])
    require(
        boundary_support.dtype == np.dtype(bool)
        and boundary_support.shape == (2001, 6)
        and np.array_equal(boundary_support, boundary_load >= SUPPORT_BW),
        "Boundary support differs from load>=.02 BW",
    )
    blocks, provenance = native_blocks(
        native["time_s"],
        native["normal_load_bw"],
        native["slip_mm_s"],
        native.get("slip_valid"),
        strict_rows=True,
    )
    require(
        np.allclose(blocks["endpoint_time_s"], time_s[1:], atol=CLOCK_ATOL, rtol=0),
        "Native block endpoints differ from control clock",
    )
    require(
        np.array_equal(model["t"], time_s) and model["window"].sum() == 1900,
        "Official model clock/window differs",
    )
    physical = {
        "time_s": time_s,
        "position": position,
        "basis": basis,
        "foot_world": foot,
        "target": target,
        "actual": actual,
        "boundary_load": boundary_load,
        "boundary_support": boundary_support,
        "qpos0": qpos[0].copy(),
        "qvel0": qvel[0].copy(),
    }
    return blocks, provenance, physical


def recording_feature_arrays(
    model: dict[str, Any],
    physical: dict[str, np.ndarray],
    blocks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    kinematics = foot_kinematics(
        physical["position"], physical["basis"], physical["foot_world"]
    )
    tracking = tracking_rms_by_leg(physical["target"], physical["actual"])
    return {
        "time_s": physical["time_s"],
        "window": model["window"],
        "speed_mm_s": model["speed"],
        "yaw_rate_rad_s": model["yaw_rate"],
        "tracking_RMS_rad_by_leg": tracking,
        "boundary_load_bw": physical["boundary_load"],
        "boundary_support": physical["boundary_support"],
        **kinematics,
        **{"native_" + key: value for key, value in blocks.items()},
    }


def summarize_recording_features(
    model: dict[str, Any],
    physical: dict[str, np.ndarray],
    blocks: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    window = np.asarray(model["window"], dtype=bool)
    groups = {
        "all": window,
        "forward": window & (model["speed"] > 0.5),
        "interrupted": window & (model["speed"] <= 0.5),
    }
    summary = {
        name: aggregate_group(
            mask,
            model["speed"],
            model["yaw_rate"],
            arrays["tracking_RMS_rad_by_leg"],
            arrays["ego_ap_velocity_mm_s"],
            arrays["world_forward_velocity_mm_s"],
            physical["boundary_support"],
            blocks,
        )
        for name, mask in groups.items()
    }
    summary["support_patterns"] = support_patterns(
        physical["boundary_support"], model["speed"], window
    )
    return summary


def recording_features(
    model: dict[str, Any],
    physical: dict[str, np.ndarray],
    blocks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compatibility composition for generated numerical review callers."""
    arrays = recording_feature_arrays(model, physical, blocks)
    return arrays, summarize_recording_features(model, physical, blocks, arrays)


def exploratory_all_window(model: dict[str, Any]) -> dict[str, Any]:
    exploratory = copy.deepcopy(model)
    exploratory["moving"] = np.asarray(model["window"], dtype=bool).copy()
    phi, events, eligible, bouts = phase_and_events(exploratory)
    return {
        "exploratory_only": True,
        "eligible_gait_pass": False,
        "includes_backward_and_stationary": True,
        "bouts": bouts,
        "eligible_samples": int(eligible.sum()),
        "pair_phase": phase_summary(phi, eligible, exploratory["dt"]),
        "events": event_summary(events),
    }


def save_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


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
        "schema": "flylab.m3-e152-output-manifest.v1",
        "files": files,
        "total_bytes": sum(value["bytes"] for value in files.values()),
    }


def deadline(started: float, limit_s: float, operation: str) -> float:
    elapsed = time.perf_counter() - started
    if elapsed > limit_s:
        raise TimeoutError(f"E152 {operation} exceeded {limit_s}s: {elapsed:.6f}s")
    return elapsed


def publish_final_result(
    path: Path,
    result: dict[str, Any],
    started: float,
    limit_s: float,
    *,
    writer: Callable[[Path, Any], None] = atomic_json,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Publish a false provisional receipt before the final acceptance receipt."""
    desired = copy.deepcopy(result)
    desired["publication_complete"] = True
    provisional = copy.deepcopy(result)
    provisional["accepted"] = False
    provisional["resource_limit_met"] = False
    provisional["publication_complete"] = False
    writer(path, provisional)
    elapsed = clock() - started
    if elapsed > limit_s:
        failure = provisional
        failure["error"] = {
            "type": "TimeoutError",
            "message": f"E152 final result publication (provisional) exceeded {limit_s}s: {elapsed:.6f}s",
        }
        result.clear()
        result.update(failure)
        raise TimeoutError(failure["error"]["message"])
    writer(path, desired)
    elapsed = clock() - started
    if elapsed <= limit_s:
        result.clear()
        result.update(desired)
        return elapsed
    failure = copy.deepcopy(desired)
    failure["accepted"] = False
    failure["resource_limit_met"] = False
    failure["error"] = {
        "type": "TimeoutError",
        "message": f"E152 final result publication exceeded {limit_s}s: {elapsed:.6f}s",
    }
    result.clear()
    result.update(failure)
    try:
        writer(path, result)
    except Exception as publication_error:  # noqa: BLE001 - retain secondary failure
        timeout = TimeoutError(failure["error"]["message"])
        timeout.add_note(
            "Secondary E152 result correction failure: "
            f"{type(publication_error).__name__}: {publication_error}"
        )
        raise timeout from None
    raise TimeoutError(failure["error"]["message"])


def run(
    freeze_path: Path,
    freeze_sha: str,
    source_review: Path,
    source_review_sha: str,
    output: Path,
    *,
    writer: Callable[[Path, Any], None] = atomic_json,
) -> dict[str, Any]:
    started = time.perf_counter()
    freeze = load_freeze(freeze_path, freeze_sha)
    limit_s = float(freeze["resource_limits"]["internal_wall_s"])
    validate_source_gate(source_review, source_review_sha)
    verify_hash_map(freeze["input_sha256"])
    verify_hash_map(freeze["reused_source_sha256"])
    verify_hash_map(freeze["preservation_sha256"])
    require(not output.exists(), "E152 output must be a new empty directory")
    require(
        shutil.disk_usage(output.parent).free
        >= freeze["resource_limits"]["free_bytes"],
        "E152 free-space reserve unavailable",
    )
    pools = threadpool_info()
    require(
        all(item.get("num_threads") == 1 for item in pools),
        "Actual CPU threadpool exceeds one thread",
    )
    deadline(started, limit_s, "preflight")
    output.mkdir(parents=True, exist_ok=False)
    ledger: dict[str, Any] = {
        "original_processes": 1,
        "automatic_retries": 0,
        "CPU_threads": 1,
        "GPU": 0,
        "fit_calls": 0,
        "model_calls": 0,
        "physics_steps": 0,
        "trace_reads": {},
        "native_reads": {},
        "control_rows": {},
        "native_rows": {},
        "native_blocks": {},
        "thread_pools": pools,
    }
    result = {
        "schema": SCHEMA,
        "accepted": False,
        "candidate_adopted": False,
        "F6_exposed": False,
        "whole_goal_complete": False,
        "E150_original_still_incomplete": True,
        "interpretation_limit": (
            "saved-data descriptive association only; not causal or a gait gate. "
            "E85 uses externally measured-step joints and fixed 0.2 six-leg tendon "
            "commands, whereas E142/E150 use neural-activation-derived tendons; equal "
            "body and initial state do not control tendon actuation or establish a single "
            "cause. All-window results are exploratory and remain separate from unchanged "
            "official F4/F5 outputs"
        ),
    }
    primary: Exception | None = None
    secondary: list[str] = []
    anchors = None
    try:
        append_progress(output / "progress.jsonl", {"stage": "started", "wall_s": 0.0})
        for recording in freeze["recordings"]:
            deadline(started, limit_s, "recording start")
            label = recording["label"]
            destination = output / label
            destination.mkdir()
            trace_path = ROOT / recording["trace"]
            native_path = ROOT / recording["native"]
            model = load_model(trace_path)
            ledger["trace_reads"][label] = 1
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            ledger["trace_reads"][label] += 1
            with np.load(native_path, allow_pickle=False) as stored:
                native = {key: stored[key] for key in stored.files}
            ledger["native_reads"][label] = 1
            blocks, provenance, physical = validate_recording(
                payload, model, native, recording["body_hash"]
            )
            current_anchor = (physical["qpos0"], physical["qvel0"])
            if anchors is None:
                anchors = current_anchor
            else:
                require(
                    np.array_equal(current_anchor[0], anchors[0])
                    and np.array_equal(current_anchor[1], anchors[1]),
                    "Initial qpos/qvel differ across frozen recordings",
                )
            arrays = recording_feature_arrays(model, physical, blocks)
            # Complete raw numerical features precede every summary calculation.
            save_npz_new(destination / "features.npz", arrays)
            summary = summarize_recording_features(model, physical, blocks, arrays)
            official, _, _ = analyze(model)
            exploratory = exploratory_all_window(model)
            event_spec = interruption_windows(model["speed"], model["window"])
            event_features = {
                "speed_mm_s": model["speed"],
                "yaw_rate_rad_s": model["yaw_rate"],
                "tracking_RMS_rad_by_leg": arrays["tracking_RMS_rad_by_leg"],
                "ego_AP_velocity_mm_s": arrays["ego_ap_velocity_mm_s"],
                "world_forward_velocity_mm_s": arrays["world_forward_velocity_mm_s"],
                "boundary_support": arrays["boundary_support"].astype(np.float64),
            }
            for key in (
                "mean_load_bw",
                "supported_duty",
                "supported_load_sum_bw",
                "supported_load_slip_sum_bw_mm_s",
                "support_count_mean",
                "support_count_lt3_fraction",
                "support_count_eq0_fraction",
            ):
                value = blocks[key]
                expanded = np.full((2001, *value.shape[1:]), np.nan)
                expanded[1:] = value
                event_features["native_" + key] = expanded
            stacks, event_summary_value = stack_events(event_features, event_spec)
            save_npz_new(destination / "event-stacks.npz", stacks)
            summary.update(
                {
                    "label": label,
                    "role": recording["role"],
                    "slip_valid_provenance": provenance,
                    "interruption_crossings": event_spec["all"],
                    "event_stack": event_summary_value,
                    "overlapping_events_are_independent": False,
                }
            )
            writer(destination / "summary.json", summary)
            writer(destination / "official-quality.json", official)
            writer(destination / "exploratory-all-window.json", exploratory)
            ledger["control_rows"][label] = 2001
            ledger["native_rows"][label] = 100000
            ledger["native_blocks"][label] = 2000
            append_progress(
                output / "progress.jsonl",
                {
                    "stage": "recording_complete",
                    "label": label,
                    "wall_s": time.perf_counter() - started,
                },
            )
        verify_hash_map(freeze["input_sha256"])
        verify_hash_map(freeze["reused_source_sha256"])
        verify_hash_map(freeze["preservation_sha256"])
        deadline(started, limit_s, "post-preservation verification")
        result["recordings"] = [item["label"] for item in freeze["recordings"]]
        result["accepted"] = True
        result["resource_limit_met"] = True
        writer(output / "ledger.json", ledger)
        writer(
            output / "timing.json",
            {
                "wall_s_before_final_publication": time.perf_counter() - started,
                "internal_limit_s": limit_s,
                "external_limit_s": freeze["resource_limits"]["external_wall_s"],
                "prior_actual_s": freeze["resource_limits"]["prior_actual_s"],
                "prospective_conservative_s": freeze["resource_limits"][
                    "prospective_cumulative_s"
                ],
            },
        )
        deadline(started, limit_s, "pre-final result publication")
        publish_final_result(
            output / "result.json", result, started, limit_s, writer=writer
        )
        manifest = output_manifest(output)
        require(
            manifest["total_bytes"] <= freeze["resource_limits"]["raw_bytes"],
            "E152 output exceeded raw byte limit",
        )
        writer(output / "output-manifest.json", manifest)
        deadline(started, limit_s, "post-manifest publication")
        return result
    except Exception as error:  # noqa: BLE001 - persist every primary analysis failure
        primary = error
        result["accepted"] = False
        result["resource_limit_met"] = (time.perf_counter() - started) <= limit_s
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if primary is not None:
            error_value = {
                "accepted": False,
                "primary": result["error"],
                "traceback": "".join(traceback.format_exception(primary)),
            }
            for name, value in (
                ("error.json", error_value),
                ("ledger.json", ledger),
                (
                    "timing.json",
                    {
                        "wall_s": time.perf_counter() - started,
                        "internal_limit_s": limit_s,
                    },
                ),
            ):
                try:
                    writer(output / name, value)
                except Exception as publication_error:  # noqa: BLE001
                    secondary.append(
                        f"{name}: {type(publication_error).__name__}: {publication_error}"
                    )
            result["resource_limit_met"] = (time.perf_counter() - started) <= limit_s
            try:
                publish_final_result(
                    output / "result.json", result, started, limit_s, writer=writer
                )
            except Exception as publication_error:  # noqa: BLE001
                result["resource_limit_met"] = False
                secondary.append(
                    "result.json: "
                    f"{type(publication_error).__name__}: {publication_error}"
                )
            if secondary:
                for note in secondary:
                    primary.add_note("Secondary E152 publication failure: " + note)
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    raise RuntimeError("Unreachable E152 runner state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.freeze_sha256,
                args.source_review,
                args.source_review_sha256,
                args.output,
            ),
            indent=2,
        )
    )
