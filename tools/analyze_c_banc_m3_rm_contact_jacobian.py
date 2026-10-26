"""E155 saved-state RM contact-point Jacobian decomposition.

The original entry point is deliberately orchestration-only: it restores one
already-recorded body solely to construct its MuJoCo model, forwards 200 saved
states in a separate ``MjData``, and never advances scientific physics.
"""

# ruff: noqa: I001  # Thread limits must precede numerical imports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from threadpoolctl import threadpool_info


FREEZE_SHA = "4c7f911a0bdd775d4291c176f53bbb7ac67112455aab490099fedb84cd4f5833"
DESIGN_SHA = "c820489fcefb978d46f2b67266efa3e1371311208c55a13b4833d4a54e742177"
OWNED = (
    "tools/analyze_c_banc_m3_rm_contact_jacobian.py",
    "tests/test_c_banc_rm_contact_jacobian.py",
)
SCHEMA = "flylab.m3.e155.rm-contact-jacobian-result.v1"
SUPPORT_BW = 0.02
CLOCK_ATOL = 2e-8
NUMERIC_ATOL = 1e-11
NUMERIC_RTOL = 1e-12
RM_LEG = 4
RM_JOINT_SLICE = slice(28, 35)
GROUP_NAMES = (
    "all",
    "forward",
    "interrupted",
    "supported_RM_braking",
    "supported_RM_propulsion",
)


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
        "E155 frozen paths must be repository-relative",
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
        "Exact E155 freeze required",
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        value.get("schema") == "flylab.m3.e155.rm-contact-jacobian.v1",
        "E155 freeze schema changed",
    )
    require(
        value.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(value["design"])) == DESIGN_SHA,
        "E155 design changed",
    )
    require(value.get("owned_files") == list(OWNED), "E155 owned files changed")
    require(
        value.get("sample_contract")
        == {
            "first_tick": 201,
            "final_tick": 400,
            "native_indices_start": 49,
            "native_stride": 50,
            "controls": 200,
            "rm_leg_index": 4,
            "rm_joint_slice": [28, 35],
            "support_threshold_bw": 0.02,
        },
        "E155 sample contract changed",
    )
    limits = value.get("resource_limits", {})
    require(
        limits.get("original_processes") == 1
        and limits.get("automatic_retries") == 0
        and limits.get("CPU_threads") == 1
        and limits.get("GPU") == 0
        and limits.get("internal_wall_s") == 35
        and limits.get("external_wall_s") == 45
        and limits.get("supervisor_wall_s") == 55
        and limits.get("raw_bytes") == 536870912
        and limits.get("free_bytes") == 12884901888
        and limits.get("prospective_cumulative_s") == 2567,
        "E155 resource contract changed",
    )
    require(
        value.get("candidate_adopted") is False
        and value.get("whole_goal_complete") is False
        and value.get("F6_exposed") is False
        and value.get("E150_original_still_incomplete") is True
        and value.get("E153_original_still_failed") is True,
        "E155 outcome boundary changed",
    )
    return value


def validate_source_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha, "Parent E155 source-review SHA changed")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_kinematic_diagnostic") is True,
        "Parent did not accept one original E155 RM contact Jacobian diagnostic",
    )
    require(
        review.get("freeze_sha256") == FREEZE_SHA,
        "Parent E155 gate has a different freeze",
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "Parent E155 gate must allow exactly one original process",
    )
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("source_sha256") == expected_sources,
        "Parent E155 gate source map differs from both owned files",
    )
    return review


def deadline(started: float, limit_s: float, operation: str) -> float:
    elapsed = time.perf_counter() - started
    if elapsed > limit_s:
        raise TimeoutError(f"E155 {operation} exceeded {limit_s}s: {elapsed:.6f}s")
    return elapsed


def horizontal_heading(rotation_world: Any) -> np.ndarray:
    rotation = np.asarray(rotation_world, dtype=np.float64)
    require(
        rotation.shape == (3, 3) and np.isfinite(rotation).all(),
        "Finite 3x3 Thorax rotation required",
    )
    require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0),
        "Thorax rotation is not orthogonal",
    )
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    magnitude = float(np.linalg.norm(heading))
    require(magnitude > 1e-12, "Thorax forward axis has no horizontal projection")
    return heading / magnitude


def validate_recorded_inputs(
    payload: dict[str, Any], body_hash: str
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    """Validate every selected command/state before model construction."""
    require(
        payload.get("schema") == "flylab.m3.e150.native-monotone-body-trace.v1"
        and payload.get("body_hash") == body_hash,
        "Frozen E150 trace identity differs",
    )
    rows = payload.get("samples")
    require(
        isinstance(rows, list) and len(rows) > 400,
        "Complete E150 control recording required",
    )
    selected = rows[201:401]
    require(
        [row.get("control_tick") for row in selected] == list(range(201, 401)),
        "E155 control ticks 201..400 required",
    )
    times = np.asarray([row.get("time_s") for row in selected], dtype=np.float64)
    require(
        np.allclose(times, np.arange(201, 401) * 0.005, atol=CLOCK_ATOL, rtol=0),
        "E155 control clock differs",
    )
    shapes: dict[str, tuple[int, ...]] = {}
    for position, row in enumerate(selected):
        for key in ("qpos", "qvel", "ctrl"):
            array = np.asarray(row.get(key), dtype=np.float64)
            require(
                array.ndim == 1 and array.size > 0 and np.isfinite(array).all(),
                f"Control {201 + position} has invalid {key}",
            )
            if key not in shapes:
                shapes[key] = array.shape
            require(
                array.shape == shapes[key],
                f"Control {201 + position} changed {key} shape",
            )
        joint = np.asarray(row.get("joint_targets_rad"), dtype=np.float64)
        adhesion = np.asarray(row.get("adhesion"))
        require(
            joint.shape == (42,) and np.isfinite(joint).all(),
            f"Control {201 + position} has invalid joint command",
        )
        require(
            adhesion.shape == (6,) and adhesion.dtype == np.dtype(bool),
            f"Control {201 + position} has invalid adhesion command",
        )
        require(
            isinstance(row.get("tendon_inputs"), dict)
            and all(
                np.asarray(value).ndim == 0 and np.isfinite(value)
                for value in row["tendon_inputs"].values()
            ),
            f"Control {201 + position} has invalid tendon command",
        )
        require(
            "neural_request" in row and row["neural_request"] is None,
            "E150 original must retain its null neural requests",
        )
    return selected, shapes


def validate_source_arrays(
    rows: list[dict[str, Any]],
    shapes: dict[str, tuple[int, ...]],
    native: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> np.ndarray:
    count = len(rows)
    require(count == 200, "Exactly 200 E155 controls required")
    indices = 49 + 50 * np.arange(count, dtype=np.int64)
    required_native = (
        "time_s",
        "rotation_world",
        "heading_horizontal",
        "force_forward_bw",
        "normal_load_bw",
        "slip_mm_s",
        "slip_valid",
    )
    require(all(key in native for key in required_native), "E154 native fields missing")
    native_suffix = {
        "time_s": (),
        "rotation_world": (3, 3),
        "heading_horizontal": (3,),
        "force_forward_bw": (6,),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
        "slip_valid": (6,),
    }
    require(
        all(
            np.asarray(native[key]).ndim == 1 + len(native_suffix[key])
            and np.asarray(native[key]).shape[1:] == native_suffix[key]
            and len(np.asarray(native[key])) > int(indices[-1])
            for key in required_native
        ),
        "E154 native rows or shapes incomplete",
    )
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    require(
        np.allclose(
            np.asarray(native["time_s"])[indices],
            times,
            atol=CLOCK_ATOL,
            rtol=0,
        ),
        "E154 native endpoint clocks differ",
    )
    require(
        features.get("time_s") is not None
        and features.get("speed_mm_s") is not None
        and np.array_equal(np.asarray(features["time_s"])[201:401], times),
        "E152 endpoint clocks differ",
    )
    speed = np.asarray(features["speed_mm_s"])[201:401]
    require(
        speed.shape == (200,) and np.isfinite(speed).all(),
        "Finite E152 endpoint speeds required",
    )
    offsets = np.asarray(contacts.get("offsets"))
    require(
        offsets.ndim == 1
        and offsets.dtype.kind in "iu"
        and len(offsets) > int(indices[-1]) + 1,
        "E154 contact offsets incomplete",
    )
    require(np.all(np.diff(offsets) >= 0), "E154 contact offsets are not monotone")
    total = int(offsets[-1])
    contact_shapes = {
        "contact_id": (),
        "geom1": (),
        "geom2": (),
        "segment": (),
        "leg": (),
        "sign": (),
        "frame": (3, 3),
        "wrench": (6,),
        "position_world": (3,),
    }
    for key, suffix in contact_shapes.items():
        value = np.asarray(contacts.get(key))
        require(
            value.shape == (total, *suffix), "E154 contact field shape differs: " + key
        )
    require(
        all(
            np.asarray(contacts[key]).dtype.kind in "iu"
            for key in ("contact_id", "geom1", "geom2", "segment", "leg", "sign")
        ),
        "E154 contact identity fields must be integers",
    )
    for key in (
        "time_s",
        "rotation_world",
        "heading_horizontal",
        "force_forward_bw",
        "normal_load_bw",
    ):
        require(
            np.isfinite(np.asarray(native[key])).all(),
            "E154 native field is nonfinite: " + key,
        )
    require(
        np.asarray(native["slip_valid"]).dtype == np.dtype(bool)
        and np.array_equal(
            np.asarray(native["slip_valid"]),
            np.isfinite(np.asarray(native["slip_mm_s"])),
        )
        and np.all(np.asarray(native["normal_load_bw"]) >= 0),
        "E154 native slip/load validity differs",
    )
    require(
        all(
            np.isfinite(np.asarray(contacts[key], dtype=np.float64)).all()
            for key in ("frame", "wrench", "position_world")
        ),
        "E154 contact geometry is nonfinite",
    )
    return indices


def validate_layout(body: Any) -> dict[str, Any]:
    m, mj = body.m, body.mj
    act_ids = np.asarray(body.act_ids, dtype=np.int64)
    require(
        act_ids.shape == (42,) and len(np.unique(act_ids)) == 42,
        "Exactly 42 unique leg actuators required",
    )
    trnid = np.asarray(m.actuator_trnid)[act_ids, 0].astype(np.int64)
    trntype = np.asarray(m.actuator_trntype)[act_ids]
    require(
        np.all(trntype == int(mj.mjtTrn.mjTRN_JOINT)),
        "Every leg actuator must use joint transmission",
    )
    require(
        np.array_equal(trnid, np.asarray(body.joint_ids, dtype=np.int64)),
        "Body joint IDs differ from actuator transmissions",
    )
    require(
        np.all(np.asarray(m.jnt_type)[trnid] == int(mj.mjtJoint.mjJNT_HINGE)),
        "Every leg actuator must drive a hinge",
    )
    dofs = np.asarray(m.jnt_dofadr)[trnid].astype(np.int64)
    require(len(np.unique(dofs)) == 42, "Leg actuator DOF addresses must be unique")
    require(
        np.array_equal(dofs, np.asarray(body.qvel_ids, dtype=np.int64)),
        "Body qvel IDs differ from actuator joint DOFs",
    )
    rm_joint_ids = trnid[RM_JOINT_SLICE]
    rm_dofs = dofs[RM_JOINT_SLICE]
    require(rm_dofs.shape == (7,), "RM must expose seven actuated DOFs")
    free_joints = np.flatnonzero(
        np.asarray(m.jnt_type) == int(mj.mjtJoint.mjJNT_FREE)
    ).astype(np.int64)
    require(free_joints.shape == (1,), "Exactly one free joint required")
    root_start = int(np.asarray(m.jnt_dofadr)[free_joints[0]])
    root_stop = (
        int(np.asarray(m.jnt_dofadr)[free_joints[0] + 1])
        if free_joints[0] + 1 < int(m.njnt)
        else int(m.nv)
    )
    require(
        root_stop - root_start == 6, "The unique free joint must own exactly six DOFs"
    )
    root_dofs = np.arange(root_start, root_start + 6, dtype=np.int64)
    require(
        np.all(root_dofs < int(m.nv)) and not np.intersect1d(root_dofs, rm_dofs).size,
        "Root6 and RM7 DOFs must be valid and disjoint",
    )

    joint_names = [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, index) for index in range(int(m.njnt))
    ]
    require(
        all(isinstance(name, str) and name for name in joint_names),
        "Every model joint must have a name",
    )
    require(
        list(body.joint_names) == [joint_names[int(value)] for value in trnid],
        "Body joint names differ from actuator transmissions",
    )
    starts = np.asarray(m.jnt_dofadr, dtype=np.int64)
    dof_joint_ids = np.empty(int(m.nv), dtype=np.int64)
    for joint_id, start in enumerate(starts):
        stop = int(starts[joint_id + 1]) if joint_id + 1 < len(starts) else int(m.nv)
        require(0 <= int(start) <= stop <= int(m.nv), "Joint DOF address order differs")
        dof_joint_ids[int(start) : stop] = joint_id
    geom_names = [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, index) for index in range(int(m.ngeom))
    ]
    body_names = [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, index) for index in range(int(m.nbody))
    ]
    return {
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "actuator_ids": act_ids.tolist(),
        "actuator_joint_ids": trnid.tolist(),
        "actuator_dof_addresses": dofs.tolist(),
        "actuator_names": [
            mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, int(value)) for value in act_ids
        ],
        "joint_names": joint_names,
        "dof_joint_ids": dof_joint_ids.tolist(),
        "dof_joint_names": [joint_names[int(value)] for value in dof_joint_ids],
        "rm_actuator_indices": list(range(28, 35)),
        "rm_joint_ids": rm_joint_ids.tolist(),
        "rm_joint_names": [joint_names[int(value)] for value in rm_joint_ids],
        "rm_dof_addresses": rm_dofs.tolist(),
        "free_joint_id": int(free_joints[0]),
        "free_joint_name": joint_names[int(free_joints[0])],
        "root_dof_addresses": root_dofs.tolist(),
        "geom_names": geom_names,
        "geom_body_ids": np.asarray(m.geom_bodyid, dtype=np.int64).tolist(),
        "body_names": body_names,
    }


def classify_saved_rm_contact(
    geom1: int,
    geom2: int,
    segment: int,
    leg: int,
    sign: int,
    feet_map: Any,
    support_geom_ids: Any,
) -> tuple[int, int]:
    mapping = np.asarray(feet_map)
    supports = {int(value) for value in support_geom_ids}
    require(
        mapping.ndim == 1 and mapping.dtype.kind in "iu",
        "Integer one-dimensional feet map required",
    )
    require(
        leg == RM_LEG and segment in (21, 22),
        "E155 positive RM contacts must be tarsus2/3",
    )
    if sign == 1:
        support_geom, foot_geom = geom1, geom2
    elif sign == -1:
        support_geom, foot_geom = geom2, geom1
    else:
        raise ValueError("Saved contact sign must be exactly +/-1")
    require(support_geom in supports, "Saved support geometry order differs")
    require(
        0 <= foot_geom < len(mapping) and int(mapping[foot_geom]) == segment,
        "Saved foot geometry/segment differs",
    )
    return foot_geom, support_geom


def decompose_contact(
    frame: Any,
    qvel: Any,
    jfoot: Any,
    jsupport: Any,
    heading: Any,
    root_dofs: Any,
    rm_dofs: Any,
) -> dict[str, Any]:
    axes = np.asarray(frame, dtype=np.float64)
    velocity = np.asarray(qvel, dtype=np.float64)
    foot = np.asarray(jfoot, dtype=np.float64)
    support = np.asarray(jsupport, dtype=np.float64)
    forward = np.asarray(heading, dtype=np.float64)
    root = np.asarray(root_dofs, dtype=np.int64)
    rm = np.asarray(rm_dofs, dtype=np.int64)
    nv = velocity.size
    require(
        axes.shape == (3, 3)
        and foot.shape == support.shape == (3, nv)
        and forward.shape == (3,)
        and root.shape == (6,)
        and rm.shape == (7,)
        and all(
            np.isfinite(value).all()
            for value in (axes, velocity, foot, support, forward)
        ),
        "Finite contact decomposition arrays have different shapes",
    )
    require(
        np.allclose(axes @ axes.T, np.eye(3), atol=1e-9, rtol=0),
        "Contact frame is not orthogonal",
    )
    require(
        abs(float(np.linalg.norm(forward)) - 1.0) <= 1e-11
        and abs(float(forward[2])) <= 1e-11,
        "Unit horizontal heading required",
    )
    require(
        np.all((0 <= root) & (root < nv))
        and np.all((0 <= rm) & (rm < nv))
        and len(np.unique(root)) == 6
        and len(np.unique(rm)) == 7
        and not np.intersect1d(root, rm).size,
        "Root6/RM7 decomposition indices differ",
    )
    normal = axes[0]
    relative = foot - support
    tangent = (np.eye(3) - np.outer(normal, normal)) @ relative
    tangent_velocity = tangent @ velocity
    projected_forward = forward @ tangent
    ap_dof = projected_forward * velocity
    total = float(ap_dof.sum())
    root_value = float(ap_dof[root].sum())
    rm_values = ap_dof[rm].copy()
    residual_mask = np.ones(nv, dtype=bool)
    residual_mask[root] = False
    residual_mask[rm] = False
    residual = float(ap_dof[residual_mask].sum())
    direct = float(forward @ tangent_velocity)
    closure_error = abs(total - (root_value + float(rm_values.sum()) + residual))
    direct_error = abs(total - direct)
    projection_error = abs(float(normal @ tangent_velocity))
    return {
        "relative_jacobian": relative,
        "tangent_jacobian": tangent,
        "tangent_velocity": tangent_velocity,
        "ap_dof": ap_dof,
        "ap_total": total,
        "ap_root": root_value,
        "ap_rm7": rm_values,
        "ap_residual": residual,
        "closure_error": closure_error,
        "direct_error": direct_error,
        "projection_error": projection_error,
        "slip_speed": float(np.linalg.norm(tangent_velocity)),
    }


def combine_contact_decompositions(
    loads: Any,
    slip_speeds: Any,
    ap_total: Any,
    ap_root: Any,
    ap_rm7: Any,
    ap_residual: Any,
) -> dict[str, Any]:
    weight = np.asarray(loads, dtype=np.float64)
    slip = np.asarray(slip_speeds, dtype=np.float64)
    total = np.asarray(ap_total, dtype=np.float64)
    root = np.asarray(ap_root, dtype=np.float64)
    rm7 = np.asarray(ap_rm7, dtype=np.float64).reshape(-1, 7)
    residual = np.asarray(ap_residual, dtype=np.float64)
    count = len(weight)
    require(
        weight.shape
        == slip.shape
        == total.shape
        == root.shape
        == residual.shape
        == (count,)
        and rm7.shape == (count, 7)
        and all(
            np.isfinite(value).all()
            for value in (weight, slip, total, root, rm7, residual)
        )
        and np.all(weight >= 0),
        "Finite nonnegative contact weights and aligned decompositions required",
    )
    denominator = float(weight.sum())
    supported = denominator >= SUPPORT_BW
    return {
        "normal_load": denominator,
        "slip": float(weight @ slip / denominator) if supported else math.nan,
        "slip_valid": supported,
        "ap_numerator": float(weight @ total),
        "root_numerator": float(weight @ root),
        "rm7_numerator": np.sum(weight[:, None] * rm7, axis=0),
        "residual_numerator": float(weight @ residual),
    }


def new_registry(nv: int) -> dict[str, Any]:
    return {
        "nv": int(nv),
        "attempts": {
            key: []
            for key in (
                "control_offset",
                "source_native_index",
                "source_raw_index",
                "contact_id",
                "geom1",
                "geom2",
                "foot_geom",
                "support_geom",
                "segment",
                "sign",
                "position_world",
                "frame",
                "wrench",
                "qvel",
                "jfoot",
                "jsupport",
                "jfoot_obtained",
                "jsupport_obtained",
                "decomposition_complete",
                "completed_contact_row",
            )
        },
        "contacts": {
            key: []
            for key in (
                "control_offset",
                "source_native_index",
                "source_raw_index",
                "contact_id",
                "geom1",
                "geom2",
                "foot_geom",
                "support_geom",
                "segment",
                "sign",
                "position_world",
                "frame",
                "wrench",
                "rotation_world",
                "heading_horizontal",
                "qvel",
                "jfoot",
                "jsupport",
                "relative_jacobian",
                "tangent_jacobian",
                "tangent_velocity",
                "ap_dof",
                "normal_load_bw",
                "slip_speed",
                "ap_total",
                "ap_root",
                "ap_rm7",
                "ap_residual",
                "closure_error",
                "direct_error",
                "projection_error",
            )
        },
        "offsets": [0],
        "controls": {
            key: []
            for key in (
                "tick",
                "time_s",
                "source_native_index",
                "rotation_world",
                "heading_horizontal",
                "endpoint_speed_mm_s",
                "original_force_forward_bw",
                "original_normal_load_bw",
                "original_slip_mm_s",
                "original_slip_valid",
                "reconstructed_normal_load_bw",
                "reconstructed_slip_mm_s",
                "reconstructed_slip_valid",
                "ap_numerator",
                "dof_numerator",
                "root_numerator",
                "rm7_numerator",
                "residual_numerator",
                "normal_load_denominator",
                "rotation_match_max_abs",
                "load_error",
                "slip_error",
                "numeric_verified",
            )
        },
    }


def _attempt_arrays(registry: dict[str, Any]) -> dict[str, np.ndarray]:
    nv = registry["nv"]
    suffix = {
        "position_world": (3,),
        "frame": (3, 3),
        "wrench": (6,),
        "qvel": (nv,),
        "jfoot": (3, nv),
        "jsupport": (3, nv),
    }
    boolean = {"jfoot_obtained", "jsupport_obtained", "decomposition_complete"}
    result = {}
    for key, values in registry["attempts"].items():
        dtype = bool if key in boolean else np.float64 if key in suffix else np.int64
        array = np.asarray(values, dtype=dtype)
        if key in suffix:
            array = array.reshape(len(values), *suffix[key])
        result[key] = array
    return result


def _contact_arrays(registry: dict[str, Any]) -> dict[str, np.ndarray]:
    nv = registry["nv"]
    suffix = {
        "position_world": (3,),
        "frame": (3, 3),
        "wrench": (6,),
        "rotation_world": (3, 3),
        "heading_horizontal": (3,),
        "qvel": (nv,),
        "jfoot": (3, nv),
        "jsupport": (3, nv),
        "relative_jacobian": (3, nv),
        "tangent_jacobian": (3, nv),
        "tangent_velocity": (3,),
        "ap_dof": (nv,),
        "ap_rm7": (7,),
    }
    integer = {
        "control_offset",
        "source_native_index",
        "source_raw_index",
        "contact_id",
        "geom1",
        "geom2",
        "foot_geom",
        "support_geom",
        "segment",
    }
    result = {"offsets": np.asarray(registry["offsets"], dtype=np.int64)}
    for key, values in registry["contacts"].items():
        dtype = np.int8 if key == "sign" else np.int64 if key in integer else np.float64
        array = np.asarray(values, dtype=dtype)
        if key in suffix:
            array = array.reshape(len(values), *suffix[key])
        result[key] = array
    return result


def _control_arrays(registry: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    suffix = {
        "rotation_world": (3, 3),
        "heading_horizontal": (3,),
        "rm7_numerator": (7,),
        "dof_numerator": (registry["nv"],),
    }
    integer = {"tick", "source_native_index"}
    boolean = {"original_slip_valid", "reconstructed_slip_valid", "numeric_verified"}
    for key, values in registry["controls"].items():
        dtype = bool if key in boolean else np.int64 if key in integer else np.float64
        array = np.asarray(values, dtype=dtype)
        if key in suffix:
            array = array.reshape(len(values), *suffix[key])
        result[key] = array
    return result


def save_raw(output: Path, registry: dict[str, Any]) -> None:
    atomic_npz(output / "contact-attempts.npz", **_attempt_arrays(registry))
    atomic_npz(output / "contact-kinematics.npz", **_contact_arrays(registry))
    atomic_npz(output / "control-kinematics.npz", **_control_arrays(registry))


def append_contact_attempt(registry: dict[str, Any], values: dict[str, Any]) -> int:
    index = len(registry["attempts"]["control_offset"])
    for key in registry["attempts"]:
        value = values[key]
        registry["attempts"][key].append(
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )
    return index


def append_contact(registry: dict[str, Any], values: dict[str, Any]) -> None:
    for key in registry["contacts"]:
        value = values[key]
        registry["contacts"][key].append(
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )


def append_control(registry: dict[str, Any], values: dict[str, Any]) -> None:
    for key in registry["controls"]:
        value = values[key]
        registry["controls"][key].append(
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )
    registry["offsets"].append(len(registry["contacts"]["control_offset"]))


def aggregate_group(selected: Any, controls: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = np.asarray(selected)
    rows = len(controls["tick"])
    require(
        mask.shape == (rows,) and mask.dtype == np.dtype(bool),
        "Boolean aggregation mask required",
    )
    if not mask.any():
        return {
            "sample_count": 0,
            "supported_samples": 0,
            "normal_load_denominator": 0.0,
            "numerator_sums": {
                "all_dof_forward_velocity": 0.0,
                "per_dof_forward_velocity": np.zeros(
                    np.asarray(controls["dof_numerator"]).shape[1]
                ).tolist(),
                "root6_forward_velocity": 0.0,
                "rm7_forward_velocity": np.zeros(7).tolist(),
                "residual_forward_velocity": 0.0,
            },
            "weighted_contributions": None,
        }
    load = np.asarray(controls["normal_load_denominator"], dtype=np.float64)[mask]
    denominator = float(load.sum())
    supported = (
        np.asarray(controls["original_normal_load_bw"], dtype=np.float64)[mask]
        >= SUPPORT_BW
    )

    def numerator(key: str) -> float | list[float]:
        value = np.asarray(controls[key], dtype=np.float64)[mask].sum(axis=0)
        return value.tolist() if value.ndim else float(value)

    numerators = {
        "all_dof_forward_velocity": numerator("ap_numerator"),
        "per_dof_forward_velocity": numerator("dof_numerator"),
        "root6_forward_velocity": numerator("root_numerator"),
        "rm7_forward_velocity": numerator("rm7_numerator"),
        "residual_forward_velocity": numerator("residual_numerator"),
    }

    def ratio(key: str) -> float | list[float]:
        if denominator == 0:
            raise ValueError("Internal zero-denominator ratio request")
        value = (
            np.asarray(controls[key], dtype=np.float64)[mask].sum(axis=0) / denominator
        )
        return value.tolist() if value.ndim else float(value)

    return {
        "sample_count": int(mask.sum()),
        "supported_samples": int(supported.sum()),
        "normal_load_denominator": denominator,
        "numerator_sums": numerators,
        "weighted_contributions": {
            "all_dof_forward_velocity": ratio("ap_numerator"),
            "per_dof_forward_velocity": ratio("dof_numerator"),
            "root6_forward_velocity": ratio("root_numerator"),
            "rm7_forward_velocity": ratio("rm7_numerator"),
            "residual_forward_velocity": ratio("residual_numerator"),
        }
        if denominator > 0
        else None,
    }


def aggregate_groups(controls: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = len(controls["tick"])
    speed = np.asarray(controls["endpoint_speed_mm_s"], dtype=np.float64)
    load = np.asarray(controls["original_normal_load_bw"], dtype=np.float64)
    force = np.asarray(controls["original_force_forward_bw"], dtype=np.float64)
    require(
        speed.shape == load.shape == force.shape == (rows,)
        and np.isfinite(speed).all()
        and np.isfinite(load).all()
        and np.isfinite(force).all(),
        "Finite control group values required",
    )
    masks = {
        "all": np.ones(rows, dtype=bool),
        "forward": speed > 0.5,
        "interrupted": speed <= 0.5,
        "supported_RM_braking": (load >= SUPPORT_BW) & (force < 0),
        "supported_RM_propulsion": (load >= SUPPORT_BW) & (force > 0),
    }
    return {name: aggregate_group(masks[name], controls) for name in GROUP_NAMES}


def process_saved_control(
    *,
    body: Any,
    shadow: Any,
    row: dict[str, Any],
    native: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    native_index: int,
    speed: float,
    control_offset: int,
    context: dict[str, Any],
    layout: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
) -> list[str]:
    m, mj, d = body.m, body.mj, shadow.d
    qpos = np.asarray(row["qpos"], dtype=np.float64)
    qvel = np.asarray(row["qvel"], dtype=np.float64)
    ctrl = np.asarray(row["ctrl"], dtype=np.float64)
    require(
        qpos.shape == (int(m.nq),)
        and qvel.shape == (int(m.nv),)
        and ctrl.shape == (int(m.nu),),
        "Saved state shape differs from model",
    )
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    d.time = float(row["time_s"])
    ledger["shadow_forward_calls_entered"] += 1
    mj.mj_forward(m, d)
    ledger["shadow_forward_calls_completed"] += 1
    _, actual_rotation, _ = shadow.pose()
    expected_rotation = np.asarray(native["rotation_world"])[native_index]
    rotation_error = float(
        np.max(np.abs(np.asarray(actual_rotation) - expected_rotation))
    )
    heading = horizontal_heading(np.asarray(native["rotation_world"])[native_index])
    require(
        np.array_equal(heading, np.asarray(native["heading_horizontal"])[native_index]),
        "Saved E154 heading differs from rotation",
    )

    offsets = np.asarray(contacts["offsets"], dtype=np.int64)
    contact_start, contact_stop = (
        int(offsets[native_index]),
        int(offsets[native_index + 1]),
    )
    root_dofs = np.asarray(layout["root_dof_addresses"], dtype=np.int64)
    rm_dofs = np.asarray(layout["rm_dof_addresses"], dtype=np.int64)
    loads, slips, totals, dof_values, roots, residuals, rm_values = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    scalar_errors: list[str] = []
    for raw_index in range(contact_start, contact_stop):
        if (
            int(contacts["leg"][raw_index]) != RM_LEG
            or float(contacts["wrench"][raw_index, 0]) <= 0
        ):
            continue
        geom1, geom2 = (
            int(contacts["geom1"][raw_index]),
            int(contacts["geom2"][raw_index]),
        )
        segment, sign = (
            int(contacts["segment"][raw_index]),
            int(contacts["sign"][raw_index]),
        )
        foot_geom, support_geom = classify_saved_rm_contact(
            geom1,
            geom2,
            segment,
            RM_LEG,
            sign,
            context["feet_map"],
            context["support_geom_ids"],
        )
        point = np.asarray(contacts["position_world"][raw_index], dtype=np.float64)
        jfoot = np.zeros((3, int(m.nv)), dtype=np.float64)
        jsupport = np.zeros((3, int(m.nv)), dtype=np.float64)
        attempt_index = append_contact_attempt(
            registry,
            {
                "control_offset": control_offset,
                "source_native_index": native_index,
                "source_raw_index": raw_index,
                "contact_id": int(contacts["contact_id"][raw_index]),
                "geom1": geom1,
                "geom2": geom2,
                "foot_geom": foot_geom,
                "support_geom": support_geom,
                "segment": segment,
                "sign": sign,
                "position_world": point,
                "frame": np.asarray(contacts["frame"][raw_index]),
                "wrench": np.asarray(contacts["wrench"][raw_index]),
                "qvel": qvel,
                "jfoot": jfoot,
                "jsupport": jsupport,
                "jfoot_obtained": False,
                "jsupport_obtained": False,
                "decomposition_complete": False,
                "completed_contact_row": -1,
            },
        )
        ledger["contact_attempts"] += 1
        ledger["contact_attempts_incomplete"] += 1
        ledger["jacobian_calls_requested"] += 1
        mj.mj_jac(m, d, jfoot, None, point, int(m.geom_bodyid[foot_geom]))
        ledger["jacobian_calls_completed"] += 1
        registry["attempts"]["jfoot"][attempt_index] = jfoot.copy()
        registry["attempts"]["jfoot_obtained"][attempt_index] = True
        ledger["jacobian_calls_requested"] += 1
        mj.mj_jac(m, d, jsupport, None, point, int(m.geom_bodyid[support_geom]))
        ledger["jacobian_calls_completed"] += 1
        registry["attempts"]["jsupport"][attempt_index] = jsupport.copy()
        registry["attempts"]["jsupport_obtained"][attempt_index] = True
        values = decompose_contact(
            contacts["frame"][raw_index],
            qvel,
            jfoot,
            jsupport,
            heading,
            root_dofs,
            rm_dofs,
        )
        load = max(float(contacts["wrench"][raw_index, 0]), 0.0) / float(
            context["weight0"]
        )
        append_contact(
            registry,
            {
                "control_offset": control_offset,
                "source_native_index": native_index,
                "source_raw_index": raw_index,
                "contact_id": int(contacts["contact_id"][raw_index]),
                "geom1": geom1,
                "geom2": geom2,
                "foot_geom": foot_geom,
                "support_geom": support_geom,
                "segment": segment,
                "sign": sign,
                "position_world": point,
                "frame": np.asarray(contacts["frame"][raw_index]),
                "wrench": np.asarray(contacts["wrench"][raw_index]),
                "rotation_world": np.asarray(native["rotation_world"])[native_index],
                "heading_horizontal": heading,
                "qvel": qvel,
                "jfoot": jfoot,
                "jsupport": jsupport,
                "relative_jacobian": values["relative_jacobian"],
                "tangent_jacobian": values["tangent_jacobian"],
                "tangent_velocity": values["tangent_velocity"],
                "ap_dof": values["ap_dof"],
                "normal_load_bw": load,
                "slip_speed": values["slip_speed"],
                "ap_total": values["ap_total"],
                "ap_root": values["ap_root"],
                "ap_rm7": values["ap_rm7"],
                "ap_residual": values["ap_residual"],
                "closure_error": values["closure_error"],
                "direct_error": values["direct_error"],
                "projection_error": values["projection_error"],
            },
        )
        registry["attempts"]["decomposition_complete"][attempt_index] = True
        registry["attempts"]["completed_contact_row"][attempt_index] = (
            len(registry["contacts"]["control_offset"]) - 1
        )
        ledger["contact_attempts_incomplete"] -= 1
        ledger["contacts_completed"] += 1
        loads.append(load)
        slips.append(values["slip_speed"])
        totals.append(values["ap_total"])
        dof_values.append(values["ap_dof"])
        roots.append(values["ap_root"])
        residuals.append(values["ap_residual"])
        rm_values.append(values["ap_rm7"])
        if (
            max(
                values["closure_error"],
                values["direct_error"],
                values["projection_error"],
            )
            > NUMERIC_ATOL
        ):
            scalar_errors.append(f"contact {raw_index} decomposition scalar mismatch")
    combined = combine_contact_decompositions(
        loads, slips, totals, roots, rm_values, residuals
    )
    load_sum = combined["normal_load"]
    slip_value = combined["slip"]
    slip_valid = combined["slip_valid"]
    original_load = float(native["normal_load_bw"][native_index, RM_LEG])
    original_slip = float(native["slip_mm_s"][native_index, RM_LEG])
    original_valid = bool(native["slip_valid"][native_index, RM_LEG])
    original_force = float(native["force_forward_bw"][native_index, RM_LEG])
    load_error = abs(load_sum - original_load)
    slip_error = (
        abs(slip_value - original_slip) if slip_valid and original_valid else 0.0
    )
    if rotation_error > 1e-12:
        scalar_errors.append(f"control {row['control_tick']} Thorax rotation mismatch")
    if not np.isclose(load_sum, original_load, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL):
        scalar_errors.append(f"control {row['control_tick']} RM normal load mismatch")
    if slip_valid != original_valid:
        scalar_errors.append(f"control {row['control_tick']} RM slip validity mismatch")
    if slip_valid and not np.isclose(
        slip_value, original_slip, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL
    ):
        scalar_errors.append(f"control {row['control_tick']} RM slip mismatch")
    append_control(
        registry,
        {
            "tick": int(row["control_tick"]),
            "time_s": float(row["time_s"]),
            "source_native_index": native_index,
            "rotation_world": np.asarray(native["rotation_world"])[native_index],
            "heading_horizontal": heading,
            "endpoint_speed_mm_s": float(speed),
            "original_force_forward_bw": original_force,
            "original_normal_load_bw": original_load,
            "original_slip_mm_s": original_slip,
            "original_slip_valid": original_valid,
            "reconstructed_normal_load_bw": load_sum,
            "reconstructed_slip_mm_s": slip_value,
            "reconstructed_slip_valid": slip_valid,
            "ap_numerator": combined["ap_numerator"],
            "dof_numerator": np.sum(
                np.asarray(loads, dtype=np.float64)[:, None]
                * np.asarray(dof_values, dtype=np.float64).reshape(-1, int(m.nv)),
                axis=0,
            )
            if loads
            else np.zeros(int(m.nv)),
            "root_numerator": combined["root_numerator"],
            "rm7_numerator": combined["rm7_numerator"],
            "residual_numerator": combined["residual_numerator"],
            "normal_load_denominator": load_sum,
            "rotation_match_max_abs": rotation_error,
            "load_error": load_error,
            "slip_error": slip_error,
            "numeric_verified": not scalar_errors,
        },
    )
    ledger["controls_completed"] += 1
    return scalar_errors


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
        "schema": "flylab.m3.e155.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def _save_failure_artifacts(
    output: Path,
    registry: dict[str, Any],
    layout: dict[str, Any],
    ledger: dict[str, Any],
    result: dict[str, Any],
    started: float,
    limit_s: float,
    primary: BaseException,
    secondary: list[str],
) -> None:
    operations: list[tuple[str, Callable[[], None]]] = [
        ("raw", lambda: save_raw(output, registry)),
        ("layout", lambda: atomic_json(output / "layout.json", layout)),
        ("ledger", lambda: atomic_json(output / "ledger.json", ledger)),
        (
            "timing",
            lambda: atomic_json(
                output / "timing.json",
                {"wall_s": time.perf_counter() - started, "internal_limit_s": limit_s},
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
    ]
    for name, action in operations:
        try:
            action()
        except Exception as error:  # noqa: BLE001
            secondary.append(name + ": " + repr(error))
    result["resource_limit_met"] = (time.perf_counter() - started) <= limit_s
    result["secondary_errors"] = list(secondary)
    try:
        atomic_json(output / "result.json", result)
    except Exception as error:  # noqa: BLE001
        secondary.append("result: " + repr(error))
    for note in secondary:
        primary.add_note("Secondary E155 cleanup/publication failure: " + note)


def run(
    freeze_path: Path,
    freeze_sha: str,
    source_review: Path,
    source_review_sha: str,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    freeze = load_freeze(freeze_path, freeze_sha)
    validate_source_gate(source_review, source_review_sha)
    verify_hash_map(freeze["input_sha256"])
    verify_hash_map(freeze["reused_source_sha256"])
    verify_hash_map(freeze["preservation_sha256"])
    require(not output.exists(), "E155 output must be new")
    require(
        shutil.disk_usage(output.parent).free
        >= freeze["resource_limits"]["free_bytes"],
        "E155 free-space reserve unavailable",
    )
    pools = threadpool_info()
    require(
        all(item.get("num_threads") == 1 for item in pools),
        "Actual CPU threadpool exceeds one thread",
    )
    limit_s = float(freeze["resource_limits"]["internal_wall_s"])
    deadline(started, limit_s, "preflight")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA,
        "accepted": False,
        "publication_complete": False,
        "resource_limit_met": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "E150_original_still_incomplete": True,
        "E153_original_still_failed": True,
        "interpretation_limit": "Kinematic identity at saved RM contact points; not a causal neural, muscle, joint-intervention, or gait-quality percentage.",
    }
    atomic_json(output / "result.json", result)
    append_progress(output / "progress.jsonl", "preflight_complete", started)
    ledger = {
        "original_processes": 1,
        "automatic_retries": 0,
        "CPU_threads": 1,
        "GPU": 0,
        "network_models": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "scientific_physics_steps": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "body_models": 0,
        "settling_physics_steps": 0,
        "settling_clock_derived_steps": 0,
        "shadow_mjdata": 0,
        "shadow_forward_calls_requested": 0,
        "shadow_forward_calls_entered": 0,
        "shadow_forward_calls_completed": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_completed": 0,
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "trace_reads": 0,
        "native_reads": 0,
        "contacts_reads": 0,
        "features_reads": 0,
        "context_reads": 0,
        "checkpoint_reads": 0,
        "controls_validated_before_model": 0,
        "controls_completed": 0,
        "contacts_completed": 0,
        "thread_pools": pools,
    }
    registry = new_registry(0)
    layout: dict[str, Any] = {}
    body = None
    primary: BaseException | None = None
    secondary: list[str] = []
    try:
        from flylab.c.storage import StateStore
        from tools import analyze_c_banc_observation as observation
        from tools import analyze_c_banc_m3_signed_contact as e154

        trace = json.loads(safe_path(freeze["trace"]).read_text(encoding="utf-8"))
        ledger["trace_reads"] = 1
        rows, shapes = validate_recorded_inputs(trace, freeze["body_hash"])
        ledger["controls_validated_before_model"] = len(rows)
        with np.load(safe_path(freeze["native"]), allow_pickle=False) as stored:
            native = {key: stored[key] for key in stored.files}
        ledger["native_reads"] = 1
        with np.load(safe_path(freeze["contacts"]), allow_pickle=False) as stored:
            contacts = {key: stored[key] for key in stored.files}
        ledger["contacts_reads"] = 1
        with np.load(safe_path(freeze["features"]), allow_pickle=False) as stored:
            features = {key: stored[key] for key in stored.files}
        ledger["features_reads"] = 1
        context = json.loads(safe_path(freeze["context"]).read_text(encoding="utf-8"))
        ledger["context_reads"] = 1
        indices = validate_source_arrays(rows, shapes, native, contacts, features)
        require(
            context.get("body_hash") == freeze["body_hash"],
            "E154 context body identity differs",
        )
        saved = StateStore.load(safe_path(freeze["checkpoint"]))
        ledger["checkpoint_reads"] = 1
        require(
            saved.get("control_tick") == 200
            and saved.get("body", {}).get("modelHash") == freeze["body_hash"],
            "Checkpoint tick/body identity differs",
        )
        require(
            e154.validate_body_parameters(saved) == context.get("body_parameters"),
            "Checkpoint body parameters differ from E154",
        )
        deadline(started, limit_s, "before body construction")
        body = e154.construct_body_safely(saved, observation, ledger)
        ledger["body_models"] = 1
        require(
            ledger["settling_physics_steps"] == 2000,
            "Measured constructor settling count differs",
        )
        ledger["settling_clock_derived_steps"] = e154.validate_settling_count(
            float(body.t0), float(body.m.opt.timestep), 2000
        )
        require(
            getattr(body, "model_hash", freeze["body_hash"]) == freeze["body_hash"],
            "Constructed body identity differs",
        )
        actual_feet_map, _ = body._force_maps["feet"]
        require(
            np.array_equal(
                np.asarray(actual_feet_map, dtype=np.int64),
                np.asarray(context["feet_map"], dtype=np.int64),
            )
            and sorted(int(value) for value in body.support_geom_ids)
            == list(context["support_geom_ids"])
            and float(body.weight0) == float(context["weight0"]),
            "Constructed body contact mapping differs from E154",
        )
        layout = validate_layout(body)
        layout.update(
            {
                "schema": "flylab.m3.e155.rm-contact-layout.v1",
                "body_hash": freeze["body_hash"],
                "feet_map": list(context["feet_map"]),
                "support_geom_ids": list(context["support_geom_ids"]),
                "weight0": float(context["weight0"]),
                "resource_contract": freeze["resource_limits"],
            }
        )
        atomic_json(output / "layout.json", layout)
        shadow = type("Shadow", (), {})()
        shadow.m = body.m
        shadow.mj = body.mj
        shadow.thorax = body.thorax
        shadow.d = body.mj.MjData(body.m)
        shadow.pose = lambda: (
            shadow.d.xpos[shadow.thorax].copy(),
            shadow.d.xmat[shadow.thorax].reshape(3, 3).copy(),
            None,
        )
        ledger["shadow_mjdata"] = 1
        registry = new_registry(int(body.m.nv))
        speed = np.asarray(features["speed_mm_s"], dtype=np.float64)[201:401]
        errors: list[str] = []
        for control_offset, (row, native_index) in enumerate(
            zip(rows, indices, strict=True)
        ):
            ledger["shadow_forward_calls_requested"] += 1
            control_errors = process_saved_control(
                body=body,
                shadow=shadow,
                row=row,
                native=native,
                contacts=contacts,
                native_index=int(native_index),
                speed=float(speed[control_offset]),
                control_offset=control_offset,
                context=context,
                layout=layout,
                registry=registry,
                ledger=ledger,
            )
            errors.extend(control_errors)
            deadline(started, limit_s, "saved-state kinematics")
        save_raw(output, registry)
        append_progress(
            output / "progress.jsonl",
            "raw_saved_before_assertions",
            started,
            contacts=ledger["contacts_completed"],
        )
        require(not errors, "; ".join(errors[:10]))
        controls = _control_arrays(registry)
        require(
            len(controls["tick"]) == 200 and np.all(controls["numeric_verified"]),
            "All 200 saved controls must pass numeric verification",
        )
        groups = aggregate_groups(controls)
        summary = {
            "schema": "flylab.m3.e155.rm-contact-jacobian-summary.v1",
            "exploratory_only": True,
            "kinematic_identity_only": True,
            "causal_percentage": False,
            "formal_F4_F5_unchanged": True,
            "groups": groups,
        }
        atomic_json(output / "summary.json", summary)
        body.close()
        body = None
        verify_hash_map(freeze["input_sha256"])
        verify_hash_map(freeze["reused_source_sha256"])
        verify_hash_map(freeze["preservation_sha256"])
        atomic_json(output / "ledger.json", ledger)
        atomic_json(
            output / "timing.json",
            {
                "wall_s_before_final_publication": time.perf_counter() - started,
                "internal_limit_s": limit_s,
                "external_limit_s": freeze["resource_limits"]["external_wall_s"],
                "supervisor_limit_s": freeze["resource_limits"]["supervisor_wall_s"],
                "prior_actual_s": freeze["resource_limits"]["prior_actual_s"],
                "prior_conservative_s": freeze["resource_limits"][
                    "prior_conservative_s"
                ],
                "previous_allocation_s": freeze["resource_limits"][
                    "previous_allocation_s"
                ],
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
                "verified_controls": 200,
                "contact_rows": ledger["contacts_completed"],
                "body_models": 1,
                "shadow_mjdata": 1,
                "settling_physics_steps": 2000,
                "kinematic_forward_calls": 200,
                "scientific_physics_steps": 0,
            }
        )
        atomic_json(output / "result.json", result)
        append_progress(output / "progress.jsonl", "complete", started)
        deadline(started, limit_s, "complete progress publication")
        manifest = output_manifest(output)
        require(
            manifest["total_bytes"] <= freeze["resource_limits"]["raw_bytes"],
            "E155 output exceeded raw byte limit",
        )
        atomic_json(output / "output-manifest.json", manifest)
        deadline(started, limit_s, "manifest publication")
        return result
    except Exception as error:  # noqa: BLE001
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
        if body is not None:
            try:
                body.close()
            except Exception as error:  # noqa: BLE001
                secondary.append("body_close: " + repr(error))
        if primary is not None:
            _save_failure_artifacts(
                output,
                registry,
                layout,
                ledger,
                result,
                started,
                limit_s,
                primary,
                secondary,
            )
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    raise RuntimeError("Unreachable E155 state")


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
