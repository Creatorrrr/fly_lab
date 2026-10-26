"""E157 native contact-point velocity and tangential-power decomposition.

The original entry point replays the already frozen E150 body commands through
the reviewed E154 body path.  It does not construct a neural model.  Every
positive native contact is recorded before numerical checks and is decomposed
from two MuJoCo point Jacobians into root, owning-leg, and residual DOFs.
"""

# ruff: noqa: I001  # Thread limits must precede numerical imports.

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


FREEZE_SHA = "513f6445b9e918b4c319b2ff1c1c2585964b10a971043a55d4e6fb0e929c6fcc"
DESIGN_SHA = "5d8587088cb8b5ddb70d27b07fba58d520d8a2eb37a5644ca0642bac26749a38"
OWNED = (
    "tools/analyze_c_banc_m3_native_contact_decomposition.py",
    "tests/test_c_banc_native_contact_decomposition.py",
)
SCHEMA = "flylab.m3.e157.native-contact-decomposition-result.v1"
LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")
GROUP_NAMES = (
    "all",
    "forward",
    "interrupted",
    "early",
    "late",
    "interrupted_early",
    "interrupted_late",
)
PHYSICS_DT = 0.0001
CONTROL_DT = 0.005
SUPPORT_BW = 0.02
CLOCK_ATOL = 2e-8
LOAD_SLIP_ATOL = 1e-8
DECOMPOSITION_ATOL = 1e-10


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
        "E157 frozen paths must be repository-relative",
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
        "Exact E157 freeze required",
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        value.get("schema") == "flylab.m3.e157.native-contact-decomposition.v1",
        "E157 freeze schema changed",
    )
    require(
        value.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(value["design"])) == DESIGN_SHA,
        "E157 design changed",
    )
    require(value.get("owned_files") == list(OWNED), "E157 owned files changed")
    require(
        value.get("sample_contract")
        == {
            "anchor_tick": 200,
            "final_tick": 400,
            "control_dt": 0.005,
            "physics_dt": 0.0001,
            "native_samples": 10000,
            "all_contacts": 24836,
            "positive_contacts": 24420,
            "positive_contacts_by_leg": [3808, 3720, 5093, 2363, 6156, 3280],
            "support_threshold_bw": 0.02,
            "clock_atol": 2e-08,
            "load_slip_atol": 1e-08,
            "decomposition_atol": 1e-10,
        },
        "E157 sample contract changed",
    )
    require(value.get("groups") == list(GROUP_NAMES), "E157 groups changed")
    require(
        len(value.get("input_sha256", {})) == 94
        and len(value.get("reused_source_sha256", {})) == 20
        and len(value.get("preservation_sha256", {})) == 3002,
        "E157 frozen hash inventory changed",
    )
    require(
        value.get("counts")
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_constructor": 1,
            "shadow_mjdata": 1,
            "settling_steps": 2000,
            "scientific_steps": 10000,
            "shadow_forward_calls": 10000,
            "positive_contacts": 24420,
            "jacobian_calls": 48840,
            "neural_steps": 0,
            "fit_calls": 0,
        },
        "E157 work contract changed",
    )
    require(
        value.get("resource_limits")
        == {
            "internal_s": 90,
            "external_s": 100,
            "supervisor_s": 110,
            "raw_bytes": 805306368,
            "free_bytes": 4294967296,
            "cpu_threads": 1,
            "gpu_count": 0,
        }
        and value.get("prior_actual_original_s") == 2575.245951900026
        and value.get("prior_conservative_s") == 2576
        and value.get("previous_allocation_s") == 2621
        and value.get("prospective_cumulative_s") == 2686,
        "E157 resource contract changed",
    )
    require(
        value.get("candidate_adopted") is False
        and value.get("whole_goal_complete") is False
        and value.get("F6_exposed") is False
        and value.get("active_UI_changed") is False,
        "E157 outcome boundary changed",
    )
    require(
        value.get("original_runs_before_gate") == 0
        and value.get("preparation_original_models") == 0
        and value.get("preparation_original_physics") == 0
        and value.get("preparation_neural_steps") == 0,
        "E157 preparation boundary changed",
    )
    return value


def validate_source_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha, "Parent E157 source-review SHA changed")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_native_decomposition") is True,
        "Parent did not accept one original E157 native decomposition",
    )
    require(
        review.get("blocking_defects") == [], "Parent E157 gate has blocking defects"
    )
    require(
        review.get("freeze_sha256") == FREEZE_SHA, "Parent E157 gate freeze differs"
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "Parent E157 gate must allow exactly one original process",
    )
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("source_sha256") == expected_sources,
        "Parent E157 gate source map differs from both owned files",
    )
    return review


def deadline(started: float, limit_s: float, operation: str) -> float:
    elapsed = time.perf_counter() - started
    if elapsed > limit_s:
        raise TimeoutError(f"E157 {operation} exceeded {limit_s}s: {elapsed:.6f}s")
    return elapsed


def horizontal_heading(rotation_world: Any) -> np.ndarray:
    rotation = np.asarray(rotation_world, dtype=np.float64)
    require(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0),
        "Finite orthogonal Thorax rotation required",
    )
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    magnitude = float(np.linalg.norm(heading))
    require(magnitude > 1e-12, "Thorax forward axis has no horizontal projection")
    return heading / magnitude


def validate_layout(body: Any) -> dict[str, Any]:
    m, mj = body.m, body.mj
    act_ids = np.asarray(body.act_ids, dtype=np.int64)
    require(
        act_ids.shape == (42,) and len(np.unique(act_ids)) == 42,
        "Exactly 42 unique leg actuators required",
    )
    joint_ids = np.asarray(m.actuator_trnid)[act_ids, 0].astype(np.int64)
    require(
        np.all(np.asarray(m.actuator_trntype)[act_ids] == int(mj.mjtTrn.mjTRN_JOINT)),
        "Every leg actuator must use joint transmission",
    )
    require(
        np.array_equal(joint_ids, np.asarray(body.joint_ids, dtype=np.int64)),
        "Body joint IDs differ",
    )
    require(
        np.all(np.asarray(m.jnt_type)[joint_ids] == int(mj.mjtJoint.mjJNT_HINGE)),
        "Every leg actuator must drive a hinge",
    )
    dofs = np.asarray(m.jnt_dofadr)[joint_ids].astype(np.int64)
    require(
        len(np.unique(dofs)) == 42
        and np.array_equal(dofs, np.asarray(body.qvel_ids, dtype=np.int64)),
        "Body actuator DOF mapping differs",
    )
    free = np.flatnonzero(np.asarray(m.jnt_type) == int(mj.mjtJoint.mjJNT_FREE))
    require(free.shape == (1,), "Exactly one free joint required")
    root_start = int(np.asarray(m.jnt_dofadr)[free[0]])
    root_dofs = np.arange(root_start, root_start + 6, dtype=np.int64)
    require(np.all(root_dofs < int(m.nv)), "Root6 DOFs differ")
    leg_dofs = dofs.reshape(6, 7)
    require(not np.intersect1d(root_dofs, leg_dofs).size, "Root and leg DOFs overlap")
    joint_names = [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, index) for index in range(int(m.njnt))
    ]
    require(
        all(isinstance(name, str) and name for name in joint_names),
        "Every joint needs a name",
    )
    require(
        list(body.joint_names) == [joint_names[int(value)] for value in joint_ids],
        "Body joint names differ",
    )
    return {
        "schema": "flylab.m3.e157.native-contact-layout.v1",
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "actuator_ids": act_ids.tolist(),
        "actuator_joint_ids": joint_ids.tolist(),
        "actuator_dof_addresses": dofs.tolist(),
        "actuator_joint_names": [joint_names[int(value)] for value in joint_ids],
        "root_dof_addresses": root_dofs.tolist(),
        "leg_actuator_indices": [list(range(7 * leg, 7 * leg + 7)) for leg in range(6)],
        "leg_joint_ids": joint_ids.reshape(6, 7).tolist(),
        "leg_dof_addresses": leg_dofs.tolist(),
        "leg_joint_names": [
            [joint_names[int(value)] for value in joint_ids[7 * leg : 7 * leg + 7]]
            for leg in range(6)
        ],
        "geom_body_ids": np.asarray(m.geom_bodyid, dtype=np.int64).tolist(),
    }


def classify_saved_contact(
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
        mapping.ndim == 1 and mapping.dtype.kind in "iu", "Integer feet map required"
    )
    require(
        0 <= segment < 30 and leg == segment // 5, "Saved contact leg/segment differs"
    )
    if sign == 1:
        support_geom, foot_geom = geom1, geom2
    elif sign == -1:
        support_geom, foot_geom = geom2, geom1
    else:
        raise ValueError("Saved contact sign must be exactly +/-1")
    require(support_geom in supports, "Saved support geometry differs")
    require(
        0 <= foot_geom < len(mapping) and int(mapping[foot_geom]) == segment,
        "Saved foot geometry differs",
    )
    return foot_geom, support_geom


def decompose_contact(
    frame: Any,
    qvel: Any,
    jfoot: Any,
    jsupport: Any,
    heading: Any,
    root_dofs: Any,
    own_dofs: Any,
    wrench: Any,
    sign: int,
    weight0: float,
) -> dict[str, Any]:
    axes = np.asarray(frame, dtype=np.float64)
    velocity = np.asarray(qvel, dtype=np.float64)
    foot = np.asarray(jfoot, dtype=np.float64)
    support = np.asarray(jsupport, dtype=np.float64)
    forward = np.asarray(heading, dtype=np.float64)
    root = np.asarray(root_dofs, dtype=np.int64)
    own = np.asarray(own_dofs, dtype=np.int64)
    contact_wrench = np.asarray(wrench, dtype=np.float64)
    nv = velocity.size
    require(
        axes.shape == (3, 3)
        and foot.shape == support.shape == (3, nv)
        and forward.shape == (3,)
        and root.shape == (6,)
        and own.shape == (7,)
        and contact_wrench.shape == (6,)
        and all(
            np.isfinite(value).all()
            for value in (axes, velocity, foot, support, forward, contact_wrench)
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
        sign in (-1, 1)
        and np.isfinite(weight0)
        and weight0 > 0
        and np.all((0 <= root) & (root < nv))
        and np.all((0 <= own) & (own < nv))
        and len(np.unique(root)) == 6
        and len(np.unique(own)) == 7
        and not np.intersect1d(root, own).size,
        "Root6/own7 indices, sign, or body weight differ",
    )
    normal = axes[0]
    projector = np.eye(3) - np.outer(normal, normal)
    relative = foot - support
    tangent = projector @ relative
    tangent_velocity = tangent @ velocity
    force_world_bw = sign * axes.T @ contact_wrench[:3] / float(weight0)
    tangent_force_bw = projector @ force_world_bw
    ap_dof = (forward @ tangent) * velocity
    power_dof = (tangent_force_bw @ tangent) * velocity
    residual_mask = np.ones(nv, dtype=bool)
    residual_mask[root] = False
    residual_mask[own] = False
    ap_total = float(ap_dof.sum())
    power_total = float(power_dof.sum())
    return {
        "relative_jacobian": relative,
        "tangent_jacobian": tangent,
        "tangent_velocity": tangent_velocity,
        "force_world_bw": force_world_bw,
        "tangent_force_bw": tangent_force_bw,
        "ap_dof": ap_dof,
        "ap_total": ap_total,
        "ap_root": float(ap_dof[root].sum()),
        "ap_own7": ap_dof[own].copy(),
        "ap_residual": float(ap_dof[residual_mask].sum()),
        "power_dof": power_dof,
        "power_total": power_total,
        "power_root": float(power_dof[root].sum()),
        "power_own7": power_dof[own].copy(),
        "power_residual": float(power_dof[residual_mask].sum()),
        "slip_speed": float(np.linalg.norm(tangent_velocity)),
        "normal_load_bw": max(float(contact_wrench[0]), 0.0) / float(weight0),
        "ap_closure_error": abs(
            ap_total
            - (
                float(ap_dof[root].sum())
                + float(ap_dof[own].sum())
                + float(ap_dof[residual_mask].sum())
            )
        ),
        "ap_direct_error": abs(ap_total - float(forward @ tangent_velocity)),
        "power_closure_error": abs(
            power_total
            - (
                float(power_dof[root].sum())
                + float(power_dof[own].sum())
                + float(power_dof[residual_mask].sum())
            )
        ),
        "power_direct_error": abs(
            power_total - float(tangent_force_bw @ tangent_velocity)
        ),
        "projection_error": abs(float(normal @ tangent_velocity)),
    }


ATTEMPT_KEYS = (
    "native_index",
    "source_raw_index",
    "contact_id",
    "leg",
    "segment",
    "geom1",
    "geom2",
    "foot_geom",
    "support_geom",
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
CONTACT_KEYS = (
    "native_index",
    "source_raw_index",
    "contact_id",
    "leg",
    "segment",
    "geom1",
    "geom2",
    "foot_geom",
    "support_geom",
    "sign",
    "position_world",
    "frame",
    "wrench",
    "qvel",
    "jfoot",
    "jsupport",
    "relative_jacobian",
    "tangent_jacobian",
    "tangent_velocity",
    "force_world_bw",
    "tangent_force_bw",
    "ap_dof",
    "power_dof",
    "normal_load_bw",
    "slip_speed",
    "ap_total",
    "ap_root",
    "ap_own7",
    "ap_residual",
    "power_total",
    "power_root",
    "power_own7",
    "power_residual",
    "ap_closure_error",
    "ap_direct_error",
    "power_closure_error",
    "power_direct_error",
    "projection_error",
)
NATIVE_KEYS = (
    "native_index",
    "time_s",
    "control_tick",
    "qpos",
    "qvel",
    "ctrl",
    "rotation_world",
    "heading_horizontal",
    "force_world_bw",
    "force_forward_bw",
    "normal_load_bw",
    "slip_mm_s",
    "slip_valid",
    "nonfoot_contact",
    "upright",
    "step_requested",
    "integration_complete",
    "state_captured",
    "state_copy_complete",
    "shadow_forward_complete",
    "observer_returned",
    "diagnostics_returned",
    "nonfoot_returned",
    "observation_complete",
    "source_verified",
)
OBSERVED_KEYS = (
    "native_index",
    "source_raw_index",
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


def new_registry(nq: int = 0, nv: int = 0, nu: int = 0) -> dict[str, Any]:
    return {
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(nu),
        "native": {key: [] for key in NATIVE_KEYS},
        "observed": {key: [] for key in OBSERVED_KEYS},
        "observed_offsets": [0],
        "attempts": {key: [] for key in ATTEMPT_KEYS},
        "contacts": {key: [] for key in CONTACT_KEYS},
        "offsets": [0],
        "controls": {
            key: [] for key in ("tick", "time_s", "qpos", "qvel", "ctrl", "verified")
        },
    }


def _append(section: dict[str, list[Any]], values: dict[str, Any]) -> None:
    require(set(section) == set(values), "Registry row fields differ")
    for key, value in values.items():
        section[key].append(
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )


def append_attempt(registry: dict[str, Any], values: dict[str, Any]) -> int:
    index = len(registry["attempts"]["native_index"])
    _append(registry["attempts"], values)
    return index


def append_native_attempt(
    registry: dict[str, Any], native_index: int, control_tick: int
) -> int:
    index = len(registry["native"]["native_index"])
    _append(
        registry["native"],
        {
            "native_index": native_index,
            "time_s": math.nan,
            "control_tick": control_tick,
            "qpos": np.full(registry["nq"], np.nan),
            "qvel": np.full(registry["nv"], np.nan),
            "ctrl": np.full(registry["nu"], np.nan),
            "rotation_world": np.full((3, 3), np.nan),
            "heading_horizontal": np.full(3, np.nan),
            "force_world_bw": np.full((6, 3), np.nan),
            "force_forward_bw": np.full(6, np.nan),
            "normal_load_bw": np.full(6, np.nan),
            "slip_mm_s": np.full(6, np.nan),
            "slip_valid": np.zeros(6, dtype=bool),
            "nonfoot_contact": False,
            "upright": math.nan,
            "step_requested": True,
            "integration_complete": False,
            "state_captured": False,
            "state_copy_complete": False,
            "shadow_forward_complete": False,
            "observer_returned": False,
            "diagnostics_returned": False,
            "nonfoot_returned": False,
            "observation_complete": False,
            "source_verified": False,
        },
    )
    return index


def capture_native_state(registry: dict[str, Any], attempt: int, body: Any) -> None:
    native = registry["native"]
    native["time_s"][attempt] = float(body.physics_time())
    native["qpos"][attempt] = np.array(body.d.qpos, copy=True)
    native["qvel"][attempt] = np.array(body.d.qvel, copy=True)
    native["ctrl"][attempt] = np.array(body.d.ctrl, copy=True)
    native["state_captured"][attempt] = True


def append_contact(registry: dict[str, Any], values: dict[str, Any]) -> int:
    index = len(registry["contacts"]["native_index"])
    _append(registry["contacts"], values)
    return index


def _stack(
    values: list[Any], suffix: tuple[int, ...], dtype: Any | None = None
) -> np.ndarray:
    if not values:
        return np.empty((0, *suffix), dtype=np.float64 if dtype is None else dtype)
    return np.asarray(values, dtype=dtype).reshape(-1, *suffix)


def registry_arrays(
    registry: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    nq, nv, nu = registry["nq"], registry["nv"], registry["nu"]
    native_shapes = {
        "qpos": (nq,),
        "qvel": (nv,),
        "ctrl": (nu,),
        "rotation_world": (3, 3),
        "heading_horizontal": (3,),
        "force_world_bw": (6, 3),
        "force_forward_bw": (6,),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
        "slip_valid": (6,),
    }
    native = {
        key: _stack(values, native_shapes.get(key, ()))
        for key, values in registry["native"].items()
    }
    common_shapes = {
        "position_world": (3,),
        "frame": (3, 3),
        "wrench": (6,),
        "qvel": (nv,),
        "jfoot": (3, nv),
        "jsupport": (3, nv),
    }
    attempts = {
        key: _stack(values, common_shapes.get(key, ()))
        for key, values in registry["attempts"].items()
    }
    contact_shapes = {
        **common_shapes,
        "relative_jacobian": (3, nv),
        "tangent_jacobian": (3, nv),
        "tangent_velocity": (3,),
        "force_world_bw": (3,),
        "tangent_force_bw": (3,),
        "ap_dof": (nv,),
        "power_dof": (nv,),
        "ap_own7": (7,),
        "power_own7": (7,),
    }
    contacts = {"offsets": np.asarray(registry["offsets"], dtype=np.int64)}
    contacts.update(
        {
            key: _stack(values, contact_shapes.get(key, ()))
            for key, values in registry["contacts"].items()
        }
    )
    control_shapes = {"qpos": (nq,), "qvel": (nv,), "ctrl": (nu,)}
    controls = {
        key: _stack(values, control_shapes.get(key, ()))
        for key, values in registry["controls"].items()
    }
    return native, attempts, contacts, controls


def observed_arrays(registry: dict[str, Any]) -> dict[str, np.ndarray]:
    shapes = {"frame": (3, 3), "wrench": (6,), "position_world": (3,)}
    arrays = {"offsets": np.asarray(registry["observed_offsets"], dtype=np.int64)}
    arrays.update(
        {
            key: _stack(values, shapes.get(key, ()))
            for key, values in registry["observed"].items()
        }
    )
    return arrays


def append_observed_contacts(
    registry: dict[str, Any],
    native_index: int,
    source_start: int,
    raw_contacts: dict[str, np.ndarray],
) -> None:
    count = len(raw_contacts["contact_id"])
    for local_index in range(count):
        _append(
            registry["observed"],
            {
                "native_index": native_index,
                "source_raw_index": source_start + local_index,
                **{
                    key: np.asarray(raw_contacts[key])[local_index]
                    for key in OBSERVED_KEYS
                    if key not in {"native_index", "source_raw_index"}
                },
            },
        )
    registry["observed_offsets"].append(len(registry["observed"]["native_index"]))


def save_raw(output: Path, registry: dict[str, Any]) -> None:
    native, attempts, contacts, controls = registry_arrays(registry)
    atomic_npz(output / "native-replay.npz", **native)
    atomic_npz(output / "observed-contacts.npz", **observed_arrays(registry))
    atomic_npz(output / "contact-attempts.npz", **attempts)
    atomic_npz(output / "contact-decomposition.npz", **contacts)
    atomic_npz(output / "replay-controls.npz", **controls)


def _equal_nan(actual: Any, expected: Any) -> bool:
    return np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True)


def validate_source_arrays(
    trace: dict[str, Any],
    native: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    controls: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    body_hash: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = trace.get("samples")
    require(
        isinstance(rows, list)
        and len(rows) > 400
        and trace.get("body_hash") == body_hash,
        "Complete E150 trace required",
    )
    rows = rows[200:401]
    require(
        [row.get("control_tick") for row in rows] == list(range(200, 401)),
        "Control ticks 200..400 required",
    )
    times = np.asarray([row.get("time_s") for row in rows], dtype=np.float64)
    require(
        np.allclose(times, np.arange(200, 401) * CONTROL_DT, atol=CLOCK_ATOL, rtol=0),
        "Control clock differs",
    )
    required_native = {
        "time_s",
        "rotation_world",
        "force_world_bw",
        "force_forward_bw",
        "normal_load_bw",
        "slip_mm_s",
        "slip_valid",
        "nonfoot_contact",
        "upright",
    }
    require(
        required_native <= set(native)
        and all(len(native[key]) == 10000 for key in required_native),
        "Complete E154 native arrays required",
    )
    offsets = np.asarray(contacts.get("offsets"), dtype=np.int64)
    require(
        offsets.shape == (10001,)
        and offsets[0] == 0
        and offsets[-1] == 24836
        and np.all(np.diff(offsets) >= 0),
        "E154 contact offsets differ",
    )
    required_contact = {
        "contact_id",
        "geom1",
        "geom2",
        "segment",
        "leg",
        "sign",
        "frame",
        "wrench",
        "position_world",
    }
    require(
        required_contact <= set(contacts)
        and all(len(contacts[key]) == 24836 for key in required_contact),
        "Complete E154 contacts required",
    )
    positive = np.asarray(contacts["wrench"], dtype=np.float64)[:, 0] > 0
    require(
        int(positive.sum()) == 24420
        and np.array_equal(
            np.bincount(np.asarray(contacts["leg"])[positive], minlength=6),
            [3808, 3720, 5093, 2363, 6156, 3280],
        ),
        "Positive contact contract differs",
    )
    require(
        all(
            key in controls
            for key in ("tick", "time_s", "qpos", "qvel", "ctrl", "verified")
        )
        and len(controls["tick"]) == 201
        and np.array_equal(controls["tick"], np.arange(200, 401))
        and np.all(controls["verified"]),
        "E154 replay controls differ",
    )
    require(
        np.array_equal(features["time_s"][201:401], times[1:])
        and np.isfinite(features["speed_mm_s"][201:401]).all(),
        "E152 feature endpoints differ",
    )
    return rows, np.asarray(features["speed_mm_s"][201:401], dtype=np.float64)


def process_positive_contacts(
    *,
    body: Any,
    shadow: Any,
    native_index: int,
    raw_contacts: dict[str, np.ndarray],
    source_contacts: dict[str, np.ndarray],
    source_start: int,
    heading: np.ndarray,
    qvel: np.ndarray,
    context: dict[str, Any],
    layout: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    count = len(raw_contacts["contact_id"])
    source_stop = source_start + count
    require(
        source_stop == int(np.asarray(source_contacts["offsets"])[native_index + 1]),
        f"Original contact count mismatch at {native_index}",
    )
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
    ):
        require(
            _equal_nan(
                raw_contacts[key],
                np.asarray(source_contacts[key])[source_start:source_stop],
            ),
            f"Original contact replay mismatch at {native_index}: {key}",
        )
    for local_index in range(count):
        source_index = source_start + local_index
        wrench = np.asarray(raw_contacts["wrench"])[local_index]
        if float(wrench[0]) <= 0:
            continue
        leg = int(np.asarray(raw_contacts["leg"])[local_index])
        segment = int(np.asarray(raw_contacts["segment"])[local_index])
        geom1 = int(np.asarray(raw_contacts["geom1"])[local_index])
        geom2 = int(np.asarray(raw_contacts["geom2"])[local_index])
        sign = int(np.asarray(raw_contacts["sign"])[local_index])
        nv = int(body.m.nv)
        jfoot = np.zeros((3, nv), dtype=np.float64)
        jsupport = np.zeros((3, nv), dtype=np.float64)
        base = {
            "native_index": native_index,
            "source_raw_index": source_index,
            "contact_id": int(np.asarray(raw_contacts["contact_id"])[local_index]),
            "leg": leg,
            "segment": segment,
            "geom1": geom1,
            "geom2": geom2,
            "foot_geom": -1,
            "support_geom": -1,
            "sign": sign,
            "position_world": np.asarray(raw_contacts["position_world"])[local_index],
            "frame": np.asarray(raw_contacts["frame"])[local_index],
            "wrench": wrench,
            "qvel": qvel,
            "jfoot": jfoot,
            "jsupport": jsupport,
            "jfoot_obtained": False,
            "jsupport_obtained": False,
            "decomposition_complete": False,
            "completed_contact_row": -1,
        }
        attempt = append_attempt(registry, base)
        ledger["contact_attempts"] += 1
        ledger["contact_attempts_incomplete"] += 1
        foot_geom, support_geom = classify_saved_contact(
            geom1,
            geom2,
            segment,
            leg,
            sign,
            context["feet_map"],
            context["support_geom_ids"],
        )
        base["foot_geom"] = foot_geom
        base["support_geom"] = support_geom
        registry["attempts"]["foot_geom"][attempt] = foot_geom
        registry["attempts"]["support_geom"][attempt] = support_geom
        point = np.asarray(base["position_world"], dtype=np.float64)
        ledger["jacobian_calls_requested"] += 1
        body.mj.mj_jac(
            body.m, shadow.d, jfoot, None, point, int(body.m.geom_bodyid[foot_geom])
        )
        ledger["jacobian_calls_returned"] += 1
        registry["attempts"]["jfoot"][attempt] = jfoot.copy()
        registry["attempts"]["jfoot_obtained"][attempt] = True
        ledger["jacobian_calls_requested"] += 1
        body.mj.mj_jac(
            body.m,
            shadow.d,
            jsupport,
            None,
            point,
            int(body.m.geom_bodyid[support_geom]),
        )
        ledger["jacobian_calls_returned"] += 1
        registry["attempts"]["jsupport"][attempt] = jsupport.copy()
        registry["attempts"]["jsupport_obtained"][attempt] = True
        decomposed = decompose_contact(
            base["frame"],
            qvel,
            jfoot,
            jsupport,
            heading,
            layout["root_dof_addresses"],
            layout["leg_dof_addresses"][leg],
            wrench,
            sign,
            context["weight0"],
        )
        require(
            max(
                decomposed[key]
                for key in (
                    "ap_closure_error",
                    "ap_direct_error",
                    "power_closure_error",
                    "power_direct_error",
                    "projection_error",
                )
            )
            <= DECOMPOSITION_ATOL,
            "Contact decomposition closure differs",
        )
        contact_row = append_contact(
            registry,
            {**{key: base[key] for key in CONTACT_KEYS if key in base}, **decomposed},
        )
        registry["attempts"]["decomposition_complete"][attempt] = True
        registry["attempts"]["completed_contact_row"][attempt] = contact_row
        ledger["contacts_completed"] += 1
        ledger["contact_attempts_incomplete"] -= 1


def aggregate_contact_subset(
    contacts: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, Any]:
    chosen = np.asarray(selected)
    count = len(contacts["native_index"])
    require(
        chosen.shape == (count,) and chosen.dtype == np.dtype(bool),
        "Boolean contact aggregation mask required",
    )
    load = np.asarray(contacts["normal_load_bw"], dtype=np.float64)[chosen]
    denominator = float(load.sum())

    def total(key: str, *, weight: bool, integrate: bool) -> Any:
        values = np.asarray(contacts[key], dtype=np.float64)[chosen]
        if weight:
            values = values * load.reshape((-1,) + (1,) * (values.ndim - 1))
        value = np.sum(values, axis=0)
        if integrate:
            value = value * PHYSICS_DT
        return value.tolist() if np.ndim(value) else float(value)

    weighted = None
    if denominator > 0:
        weighted = {
            "slip": float(
                total("slip_speed", weight=True, integrate=False) / denominator
            ),
            "ap_total": float(
                total("ap_total", weight=True, integrate=False) / denominator
            ),
            "ap_dof": (
                np.asarray(total("ap_dof", weight=True, integrate=False)) / denominator
            ).tolist(),
            "ap_root": float(
                total("ap_root", weight=True, integrate=False) / denominator
            ),
            "ap_own7": (
                np.asarray(total("ap_own7", weight=True, integrate=False)) / denominator
            ).tolist(),
            "ap_residual": float(
                total("ap_residual", weight=True, integrate=False) / denominator
            ),
        }
    return {
        "contact_count": int(chosen.sum()),
        "normal_load_sum": denominator,
        "load_weighted_velocity": weighted,
        "tangential_work_bw_mm": {
            "total": total("power_total", weight=False, integrate=True),
            "dof": total("power_dof", weight=False, integrate=True),
            "root": total("power_root", weight=False, integrate=True),
            "own7": total("power_own7", weight=False, integrate=True),
            "residual": total("power_residual", weight=False, integrate=True),
        },
    }


def aggregate_summary(
    native: dict[str, np.ndarray],
    contacts: dict[str, np.ndarray],
    endpoint_speed: np.ndarray,
) -> dict[str, Any]:
    native_count = len(native["time_s"])
    require(
        native_count == 10000 and endpoint_speed.shape == (200,),
        "Complete aggregation inputs required",
    )
    control_speed = np.repeat(endpoint_speed, 50)
    step_in_control = np.arange(native_count) % 50
    masks = {
        "all": np.ones(native_count, dtype=bool),
        "forward": control_speed > 0.5,
        "interrupted": control_speed <= 0.5,
        "early": step_in_control < 10,
        "late": step_in_control >= 10,
        "interrupted_early": (control_speed <= 0.5) & (step_in_control < 10),
        "interrupted_late": (control_speed <= 0.5) & (step_in_control >= 10),
    }
    c_native = np.asarray(contacts["native_index"], dtype=np.int64)
    c_leg = np.asarray(contacts["leg"], dtype=np.int64)
    dt = PHYSICS_DT
    result: dict[str, Any] = {}
    for group in GROUP_NAMES:
        legs: dict[str, Any] = {}
        for leg, name in enumerate(LEG_NAMES):
            chosen = masks[group][c_native] & (c_leg == leg)
            native_mask = masks[group]
            supported_native = native_mask & (
                np.asarray(native["normal_load_bw"])[:, leg] >= SUPPORT_BW
            )
            force = np.asarray(native["force_forward_bw"])[:, leg]
            supported_contacts = chosen & supported_native[c_native]
            braking_contacts = supported_contacts & (force[c_native] < 0)
            propulsion_contacts = supported_contacts & (force[c_native] > 0)
            legs[name] = {
                "supported_native_count": int(supported_native.sum()),
                "supported_braking_native_count": int(
                    np.sum(supported_native & (force < 0))
                ),
                "supported_propulsion_native_count": int(
                    np.sum(supported_native & (force > 0))
                ),
                "negative_forward_impulse_bw_s": float(
                    np.minimum(force[native_mask], 0).sum() * dt
                ),
                "positive_forward_impulse_bw_s": float(
                    np.maximum(force[native_mask], 0).sum() * dt
                ),
                "all_positive_contacts": aggregate_contact_subset(contacts, chosen),
                "supported_contacts": aggregate_contact_subset(
                    contacts, supported_contacts
                ),
                "supported_braking_contacts": aggregate_contact_subset(
                    contacts, braking_contacts
                ),
                "supported_propulsion_contacts": aggregate_contact_subset(
                    contacts, propulsion_contacts
                ),
            }
        result[group] = {"native_count": int(masks[group].sum()), "legs": legs}
    return result


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
        "schema": "flylab.m3.e157.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def _save_failure(
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
    operations = (
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
    )
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
    try:
        atomic_json(output / "output-manifest.json", output_manifest(output))
    except Exception as error:  # noqa: BLE001
        secondary.append("manifest: " + repr(error))
    for note in secondary:
        primary.add_note("Secondary E157 cleanup/publication failure: " + note)


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
    require(not output.exists(), "E157 output must be new")
    require(
        shutil.disk_usage(output.parent).free
        >= freeze["resource_limits"]["free_bytes"],
        "E157 free-space reserve unavailable",
    )
    pools = threadpool_info()
    require(
        all(item.get("num_threads") == 1 for item in pools),
        "Actual CPU threadpool exceeds one thread",
    )
    limit_s = float(freeze["resource_limits"]["internal_s"])
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
        "active_UI_changed": False,
        "interpretation_limit": "Linearized native contact kinematics and tangential power; not a causal neural, muscle, joint-intervention, or gait-quality result.",
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
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "body_models": 0,
        "settling_physics_steps": 0,
        "settling_clock_derived_steps": 0,
        "scientific_physics_steps_requested": 0,
        "scientific_physics_steps_advanced": 0,
        "scientific_samples_observed": 0,
        "scientific_samples_verified": 0,
        "shadow_mjdata": 0,
        "state_copy_calls_requested": 0,
        "state_copy_calls_returned": 0,
        "shadow_forward_calls_requested": 0,
        "shadow_forward_calls_returned": 0,
        "observer_calls_requested": 0,
        "observer_calls_returned": 0,
        "diagnostic_calls_requested": 0,
        "diagnostic_calls_returned": 0,
        "nonfoot_calls_requested": 0,
        "nonfoot_calls_returned": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_returned": 0,
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "contacts_completed": 0,
        "controls_completed": 0,
        "trace_reads": 0,
        "native_reads": 0,
        "contacts_reads": 0,
        "baseline_controls_reads": 0,
        "features_reads": 0,
        "context_reads": 0,
        "checkpoint_reads": 0,
        "thread_pools": pools,
    }
    registry = new_registry()
    layout: dict[str, Any] = {}
    body = None
    original_step = None
    primary: BaseException | None = None
    secondary: list[str] = []
    try:
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import contact_diagnostics
        from tools import analyze_c_banc_observation as observation
        from tools import analyze_c_banc_m3_signed_contact as e154

        trace = json.loads(safe_path(freeze["trace"]).read_text(encoding="utf-8"))
        ledger["trace_reads"] = 1
        with np.load(safe_path(freeze["native"]), allow_pickle=False) as stored:
            native_source = {key: stored[key] for key in stored.files}
        ledger["native_reads"] = 1
        with np.load(safe_path(freeze["contacts"]), allow_pickle=False) as stored:
            contacts_source = {key: stored[key] for key in stored.files}
        ledger["contacts_reads"] = 1
        with np.load(
            safe_path(freeze["baseline_controls"]), allow_pickle=False
        ) as stored:
            baseline_controls = {key: stored[key] for key in stored.files}
        ledger["baseline_controls_reads"] = 1
        with np.load(safe_path(freeze["features"]), allow_pickle=False) as stored:
            features = {key: stored[key] for key in stored.files}
        ledger["features_reads"] = 1
        context = json.loads(safe_path(freeze["context"]).read_text(encoding="utf-8"))
        ledger["context_reads"] = 1
        rows, endpoint_speed = validate_source_arrays(
            trace,
            native_source,
            contacts_source,
            baseline_controls,
            features,
            freeze["body_hash"],
        )
        require(
            context.get("body_hash") == freeze["body_hash"],
            "E154 context body identity differs",
        )
        saved = StateStore.load(safe_path(freeze["checkpoint"]))
        ledger["checkpoint_reads"] = 1
        require(
            saved.get("control_tick") == 200
            and saved.get("body", {}).get("modelHash") == freeze["body_hash"],
            "Checkpoint identity differs",
        )
        require(
            e154.validate_body_parameters(saved) == context.get("body_parameters"),
            "Checkpoint body parameters differ",
        )
        implicit_request = e154.implicit_request_for_trace(trace, saved)
        prepared = [
            e154.replay_command(row, implicit_request=implicit_request)
            for row in rows[1:]
        ]
        deadline(started, limit_s, "before body construction")
        body = e154.construct_body_safely(saved, observation, ledger)
        ledger["body_models"] = 1
        require(
            abs(float(body.m.opt.timestep) - PHYSICS_DT) <= 1e-15,
            "Actual physics dt differs",
        )
        ledger["settling_clock_derived_steps"] = e154.validate_settling_count(
            float(body.t0), PHYSICS_DT, 2000
        )
        require(
            ledger["body_constructor_completed"] == 1
            and ledger["settling_physics_steps"] == 2000,
            "Measured constructor work differs",
        )
        mapping, _ = body._force_maps["feet"]
        require(
            np.array_equal(mapping, context["feet_map"])
            and sorted(body.support_geom_ids) == list(context["support_geom_ids"])
            and float(body.weight0) == float(context["weight0"]),
            "Constructed contact mapping differs",
        )
        layout = validate_layout(body)
        layout.update(
            {
                "body_hash": freeze["body_hash"],
                "feet_map": list(context["feet_map"]),
                "support_geom_ids": list(context["support_geom_ids"]),
                "weight0": float(context["weight0"]),
                "resource_contract": freeze["resource_limits"],
            }
        )
        atomic_json(output / "layout.json", layout)
        registry = new_registry(int(body.m.nq), int(body.m.nv), int(body.m.nu))
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
            attempt = append_native_attempt(registry, native_index, active_tick)
            ledger["scientific_physics_steps_requested"] += 1
            try:
                original_step()
            except BaseException as primary_step:
                try:
                    capture_native_state(registry, attempt, body)
                except BaseException as capture_error:  # noqa: BLE001
                    primary_step.add_note(
                        "Secondary E157 post-integration state capture failure: "
                        + repr(capture_error)
                    )
                raise
            ledger["scientific_physics_steps_advanced"] += 1
            registry["native"]["integration_complete"][attempt] = True
            capture_native_state(registry, attempt, body)
            ledger["state_copy_calls_requested"] += 1
            body.mj.mj_getState(body.m, body.d, state, state_type)
            body.mj.mj_setState(body.m, shadow.d, state, state_type)
            ledger["state_copy_calls_returned"] += 1
            registry["native"]["state_copy_complete"][attempt] = True
            ledger["shadow_forward_calls_requested"] += 1
            body.mj.mj_forward(body.m, shadow.d)
            ledger["shadow_forward_calls_returned"] += 1
            registry["native"]["shadow_forward_complete"][attempt] = True
            ledger["observer_calls_requested"] += 1
            force_world, raw_contacts, rotation = e154._observe_shadow(body, shadow)
            ledger["observer_calls_returned"] += 1
            registry["native"]["observer_returned"][attempt] = True
            registry["native"]["rotation_world"][attempt] = np.array(
                rotation, copy=True
            )
            registry["native"]["force_world_bw"][attempt] = np.array(
                force_world, copy=True
            )
            registry["native"]["upright"][attempt] = float(rotation[2, 2])
            source_start = int(contacts_source["offsets"][native_index])
            append_observed_contacts(registry, native_index, source_start, raw_contacts)
            ledger["scientific_samples_observed"] += 1
            ledger["diagnostic_calls_requested"] += 1
            diagnostic = contact_diagnostics(shadow)
            ledger["diagnostic_calls_returned"] += 1
            registry["native"]["diagnostics_returned"][attempt] = True
            load = np.asarray(diagnostic["normal_load_bw"], dtype=np.float64)
            slip = np.asarray(
                [
                    np.nan if value is None else value
                    for value in diagnostic["slip_mm_s"]
                ],
                dtype=np.float64,
            )
            valid = np.isfinite(slip)
            native_values = registry["native"]
            native_values["normal_load_bw"][attempt] = load.copy()
            native_values["slip_mm_s"][attempt] = slip.copy()
            native_values["slip_valid"][attempt] = valid.copy()
            ledger["nonfoot_calls_requested"] += 1
            nonfoot = bool(shadow.nonfoot_contact())
            ledger["nonfoot_calls_returned"] += 1
            registry["native"]["nonfoot_returned"][attempt] = True
            time_s = float(registry["native"]["time_s"][attempt])
            native_values["nonfoot_contact"][attempt] = nonfoot
            native_values["observation_complete"][attempt] = True
            heading = horizontal_heading(rotation)
            force_forward = np.asarray(force_world) @ heading
            native_values["heading_horizontal"][attempt] = heading.copy()
            native_values["force_forward_bw"][attempt] = force_forward.copy()
            expected = {
                "time_s": native_source["time_s"][native_index],
                "rotation_world": native_source["rotation_world"][native_index],
                "force_world_bw": native_source["force_world_bw"][native_index],
                "force_forward_bw": native_source["force_forward_bw"][native_index],
                "normal_load_bw": native_source["normal_load_bw"][native_index],
                "slip_mm_s": native_source["slip_mm_s"][native_index],
                "slip_valid": native_source["slip_valid"][native_index],
                "nonfoot_contact": native_source["nonfoot_contact"][native_index],
                "upright": native_source["upright"][native_index],
            }
            for key, wanted in expected.items():
                actual = native_values[key][attempt]
                okay = _equal_nan(actual, wanted)
                require(
                    okay, f"Original native replay mismatch at {native_index}: {key}"
                )
            require(
                abs(time_s - (1.0 + (native_index + 1) * PHYSICS_DT)) <= CLOCK_ATOL,
                f"Native ideal clock mismatch at {native_index}",
            )
            process_positive_contacts(
                body=body,
                shadow=shadow,
                native_index=native_index,
                raw_contacts=raw_contacts,
                source_contacts=contacts_source,
                source_start=source_start,
                heading=heading,
                qvel=np.asarray(body.d.qvel),
                context=context,
                layout=layout,
                registry=registry,
                ledger=ledger,
            )
            registry["offsets"].append(len(registry["contacts"]["native_index"]))
            registry["native"]["source_verified"][attempt] = True
            ledger["scientific_samples_verified"] += 1
            native_index += 1

        body._step_physics = observed_step
        anchor = rows[0]
        _append(
            registry["controls"],
            {
                "tick": 200,
                "time_s": float(body.physics_time()),
                "qpos": np.array(body.d.qpos),
                "qvel": np.array(body.d.qvel),
                "ctrl": np.array(body.d.ctrl),
                "verified": False,
            },
        )
        for key in ("qpos", "qvel", "ctrl"):
            require(
                np.array_equal(getattr(body.d, key), np.asarray(anchor[key]))
                and np.array_equal(getattr(body.d, key), baseline_controls[key][0]),
                "Checkpoint anchor differs: " + key,
            )
        require(
            abs(float(body.physics_time()) - float(anchor["time_s"])) <= CLOCK_ATOL,
            "Checkpoint anchor clock differs",
        )
        registry["controls"]["verified"][-1] = True
        for offset, (row, command_data) in enumerate(
            zip(rows[1:], prepared, strict=True), start=1
        ):
            joint, adhesion, command = command_data
            active_tick = int(row["control_tick"])
            body.push_left = command["push_left_s"]
            body.push_force = command["push_force"]
            body.step_joint_targets(joint, adhesion, **command["step_kwargs"])
            values = {
                "tick": active_tick,
                "time_s": float(body.physics_time()),
                "qpos": np.array(body.d.qpos),
                "qvel": np.array(body.d.qvel),
                "ctrl": np.array(body.d.ctrl),
                "verified": False,
            }
            _append(registry["controls"], values)
            for key in ("qpos", "qvel", "ctrl"):
                require(
                    np.array_equal(values[key], np.asarray(row[key]))
                    and np.array_equal(values[key], baseline_controls[key][offset]),
                    f"Replay control {active_tick} differs: {key}",
                )
            require(
                abs(values["time_s"] - float(row["time_s"])) <= CLOCK_ATOL,
                f"Replay control {active_tick} clock differs",
            )
            registry["controls"]["verified"][-1] = True
            ledger["controls_completed"] += 1
            require(not body.fault, "Body replay fault: " + str(body.fault))
            deadline(started, limit_s, "scientific replay")
        save_raw(output, registry)
        append_progress(
            output / "progress.jsonl",
            "raw_saved_before_assertions",
            started,
            native=native_index,
            contacts=ledger["contacts_completed"],
        )
        require(
            native_index == 10000
            and ledger["scientific_physics_steps_requested"] == 10000
            and ledger["scientific_physics_steps_advanced"] == 10000
            and ledger["scientific_samples_observed"] == 10000
            and ledger["scientific_samples_verified"] == 10000
            and ledger["shadow_forward_calls_requested"]
            == ledger["shadow_forward_calls_returned"]
            == 10000
            and ledger["state_copy_calls_requested"]
            == ledger["state_copy_calls_returned"]
            == 10000
            and ledger["observer_calls_requested"]
            == ledger["observer_calls_returned"]
            == 10000
            and ledger["diagnostic_calls_requested"]
            == ledger["diagnostic_calls_returned"]
            == 10000
            and ledger["nonfoot_calls_requested"]
            == ledger["nonfoot_calls_returned"]
            == 10000
            and ledger["contact_attempts"] == 24420
            and ledger["contact_attempts_incomplete"] == 0
            and ledger["contacts_completed"] == 24420
            and ledger["jacobian_calls_requested"]
            == ledger["jacobian_calls_returned"]
            == 48840,
            "Exact E157 original work required",
        )
        native_arrays, _, contact_arrays, control_arrays = registry_arrays(registry)
        require(
            all(
                np.all(native_arrays[key])
                for key in (
                    "step_requested",
                    "integration_complete",
                    "state_captured",
                    "state_copy_complete",
                    "shadow_forward_complete",
                    "observer_returned",
                    "diagnostics_returned",
                    "nonfoot_returned",
                    "observation_complete",
                    "source_verified",
                )
            ),
            "Every E157 native receipt stage must complete",
        )
        require(
            len(control_arrays["tick"]) == 201 and np.all(control_arrays["verified"]),
            "Anchor plus 200 controls required",
        )
        flat = np.asarray(
            contact_arrays["native_index"], dtype=np.int64
        ) * 6 + np.asarray(contact_arrays["leg"], dtype=np.int64)
        contact_load = np.asarray(contact_arrays["normal_load_bw"], dtype=np.float64)
        reconstructed_load = np.bincount(
            flat, weights=contact_load, minlength=60000
        ).reshape(10000, 6)
        reconstructed_slip = np.full((10000, 6), np.nan)
        valid_load = reconstructed_load >= SUPPORT_BW
        slip_numerator = np.bincount(
            flat,
            weights=contact_load
            * np.asarray(contact_arrays["slip_speed"], dtype=np.float64),
            minlength=60000,
        ).reshape(10000, 6)
        reconstructed_slip[valid_load] = (
            slip_numerator[valid_load] / reconstructed_load[valid_load]
        )
        require(
            np.allclose(
                reconstructed_load,
                native_arrays["normal_load_bw"],
                atol=LOAD_SLIP_ATOL,
                rtol=0,
            ),
            "Reconstructed native load differs",
        )
        require(
            np.array_equal(np.isfinite(reconstructed_slip), native_arrays["slip_valid"])
            and np.allclose(
                reconstructed_slip[np.isfinite(reconstructed_slip)],
                native_arrays["slip_mm_s"][np.isfinite(reconstructed_slip)],
                atol=LOAD_SLIP_ATOL,
                rtol=0,
            ),
            "Reconstructed native slip differs",
        )
        summary = {
            "schema": "flylab.m3.e157.native-contact-decomposition-summary.v1",
            "exploratory_only": True,
            "causal_intervention": False,
            "groups": aggregate_summary(native_arrays, contact_arrays, endpoint_speed),
        }
        atomic_json(output / "summary.json", summary)
        body._step_physics = original_step
        original_step = None
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
                "external_limit_s": freeze["resource_limits"]["external_s"],
                "supervisor_limit_s": freeze["resource_limits"]["supervisor_s"],
                "prior_actual_original_s": freeze["prior_actual_original_s"],
                "prior_conservative_s": freeze["prior_conservative_s"],
                "previous_allocation_s": freeze["previous_allocation_s"],
                "prospective_cumulative_s": freeze["prospective_cumulative_s"],
            },
        )
        result.update(
            {
                "accepted": True,
                "publication_complete": True,
                "resource_limit_met": True,
                "native_samples": 10000,
                "positive_contacts": 24420,
                "jacobian_calls": 48840,
                "body_models": 1,
                "shadow_mjdata": 1,
                "scientific_physics_steps": 10000,
            }
        )
        atomic_json(output / "result.json", result)
        append_progress(output / "progress.jsonl", "complete", started)
        deadline(started, limit_s, "complete progress publication")
        manifest = output_manifest(output)
        require(
            manifest["total_bytes"] <= freeze["resource_limits"]["raw_bytes"],
            "E157 output exceeded raw byte limit",
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
            if original_step is not None:
                try:
                    body._step_physics = original_step
                except Exception as error:  # noqa: BLE001
                    secondary.append("step_restore: " + repr(error))
            try:
                body.close()
            except Exception as error:  # noqa: BLE001
                secondary.append("body_close: " + repr(error))
        if primary is not None:
            _save_failure(
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
    raise RuntimeError("Unreachable E157 state")


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
