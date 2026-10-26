"""E165 static command-lift diagnostic at saved E150 body states.

Tarsal body origins are not contact points; J*(request-current) is a local
kinematic direction, not a predicted trajectory or a gait result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "flylab.m3.e165.command-lift.v1"
FREEZE_SHA = "8d5a040ad0359b31ceb853ff9a4192ac30f5edba457bbd658c0b4004ee737102"
OWNED = ("tools/analyze_c_banc_m3_command_lift.py", "tests/test_c_banc_command_lift.py")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
KINDS = ("requested", "delivered", "effective")
GROUPS = (
    "all",
    "forward",
    "interrupted",
    "supported",
    "supported_interrupted",
    "supported_interrupted_recovery",
)
S = np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=np.float64)


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(relative: str) -> Path:
    require(isinstance(relative, str), "Frozen path must be a string")
    path = (ROOT / relative).resolve()
    require(
        path.is_relative_to(ROOT)
        and path.is_file()
        or path.is_relative_to(ROOT)
        and path.is_dir(),
        "Frozen path escapes workspace or is missing",
    )
    return path


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        np.savez(stream, **arrays)
    os.replace(tmp, path)


def hash_map(freeze: Mapping[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        for name, digest in freeze[key].items():
            require(
                name not in merged or merged[name] == digest,
                f"Conflicting frozen hash: {name}",
            )
            merged[name] = digest
    return merged


def verify_hashes(entries: Mapping[str, str]) -> None:
    for name, digest in entries.items():
        require(sha256(safe_path(name)) == digest, f"Frozen file changed: {name}")


def validate_freeze(freeze: Mapping[str, Any], freeze_path: Path, digest: str) -> None:
    require(
        digest == FREEZE_SHA and sha256(freeze_path) == FREEZE_SHA,
        "E165 freeze SHA differs",
    )
    require(
        freeze.get("schema") == SCHEMA and freeze.get("owned_files") == list(OWNED),
        "E165 freeze identity differs",
    )
    require(
        sha256(safe_path(freeze["design"])) == freeze["design_sha256"],
        "E165 design SHA differs",
    )
    contract = freeze["sample_contract"]
    require(
        contract["legs"] == list(LEGS) and contract["segments"] == [1, 2, 3, 4, 5],
        "Leg order differs",
    )
    require(
        contract["request_kinds"] == list(KINDS)
        and contract["group_names"] == list(GROUPS),
        "Diagnostic labels differ",
    )
    require(
        np.array_equal(np.asarray(contract["ui_from_world"]), S),
        "World-to-UI coordinate frame differs",
    )
    require(
        contract["recorded_rows"] == 2001
        and contract["diagnostic_rows"] == 2000
        and contract["joints"] == 42,
        "E165 row contract differs",
    )
    limits = freeze["resource_limits"]
    require(
        limits["original_processes"] == 1
        and limits["automatic_retries"] == 0
        and limits["cpu_threads"] == 1
        and limits["gpu_count"] == 0,
        "E165 process contract differs",
    )
    require(
        freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "E165 outcome boundary differs",
    )


def validate_gate(
    review_path: Path, review_sha: str, freeze_sha: str
) -> dict[str, Any]:
    require(sha256(review_path) == review_sha, "Parent source gate SHA differs")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    expected = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("approved") is True and review.get("blocking_defects") == [],
        "Parent source gate has blocking defects",
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "Parent did not allow exactly one process",
    )
    require(
        review.get("accepts_single_command_lift_diagnostic") is True,
        "Parent did not accept E165 diagnostic",
    )
    require(
        review.get("freeze_sha256") == freeze_sha
        and review.get("source_sha256") == expected,
        "Parent gate source/freeze identity differs",
    )
    return review


def horizontal_heading(rotation: Any) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    require(r.shape == (3, 3) and np.isfinite(r).all(), "Finite rotation required")
    require(
        np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0),
        "Thorax rotation is not orthogonal",
    )
    h = r[:, 0].copy()
    h[2] = 0
    norm = float(np.linalg.norm(h))
    require(norm > 1e-12, "Thorax heading has no horizontal component")
    return h / norm


def effective_requests(
    requested: Any, delivered: Any, limited: Any, ranges: Any
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(delivered, dtype=np.float64)
    req = np.asarray(requested, dtype=np.float64)
    lim = np.asarray(limited, dtype=bool)
    rng = np.asarray(ranges, dtype=np.float64)
    require(
        raw.shape == req.shape
        and raw.ndim == 1
        and lim.shape == raw.shape
        and rng.shape == (len(raw), 2),
        "Control mapping shape differs",
    )
    require(
        np.isfinite(raw).all() and np.isfinite(req).all() and np.isfinite(rng).all(),
        "Control mapping is nonfinite",
    )
    require(np.all(rng[lim, 0] <= rng[lim, 1]), "Invalid actuator control range")
    effective = raw.copy()
    effective[lim] = np.clip(raw[lim], rng[lim, 0], rng[lim, 1])
    return np.stack((req, raw, effective)), lim & (effective != raw)


def project_command(j7: Any, delta: Any, heading: Any) -> dict[str, np.ndarray]:
    j = np.asarray(j7, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    h = np.asarray(heading, dtype=np.float64)
    require(
        j.shape == (3, 7) and d.shape == (3, 7) and h.shape == (3,),
        "J7/delta/heading shape differs",
    )
    require(
        np.isfinite(j).all() and np.isfinite(d).all() and np.isfinite(h).all(),
        "J7/delta/heading nonfinite",
    )
    # FlyGym's native MuJoCo body coordinates and Jacobians are already in mm.
    xyz = np.einsum("ij,kj->ki", j, d)
    z_joint = d * j[2][None, :]
    ap_joint = d * (h @ j)[None, :]
    return {
        "xyz_mm": xyz,
        "z_joint_mm": z_joint,
        "ap_joint_mm": ap_joint,
        "z_mm": z_joint.sum(axis=1),
        "ap_mm": ap_joint.sum(axis=1),
    }


def segment_signs(values: Any, zero_mm: float) -> dict[str, np.ndarray]:
    z = np.asarray(values, dtype=np.float64)
    require(
        z.shape[-1] == 5 and np.isfinite(z).all(), "Five finite tarsal values required"
    )
    positive = np.all(z > zero_mm, axis=-1)
    nonpositive = np.all(z <= zero_mm, axis=-1)
    return {
        "minimum": z.min(axis=-1),
        "maximum": z.max(axis=-1),
        "mean": z.mean(axis=-1),
        "all_positive": positive,
        "all_nonpositive": nonpositive,
        "mixed": ~(positive | nonpositive),
    }


def weighted_mean(values: Any, weights: Any) -> float | None:
    v, w = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    require(
        v.shape == w.shape and np.isfinite(w).all() and np.all(w >= 0),
        "Invalid weighted set",
    )
    selected = np.isfinite(v) & (w > 0)
    denominator = float(w[selected].sum())
    return (
        float(np.dot(v[selected], w[selected]) / denominator)
        if denominator > 0
        else None
    )


def weighted_evidence(values: Any, weights: Any) -> dict[str, float | int | None]:
    v, w = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    require(
        v.shape == w.shape and np.isfinite(w).all() and np.all(w >= 0),
        "Invalid weighted set",
    )
    selected = np.isfinite(v) & (w > 0)
    numerator = float(np.dot(v[selected], w[selected]))
    denominator = float(w[selected].sum())
    return {
        "available_count": int(selected.sum()),
        "missing_count": int((~np.isfinite(v)).sum()),
        "numerator_sum_mm_bw": numerator,
        "denominator_sum_bw": denominator,
        "mean_mm": numerator / denominator if denominator > 0 else None,
    }


def stats(values: Any, zero: float = 0.0) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    return {
        "count": len(finite),
        "missing": int(len(a) - len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "quantiles": np.quantile(finite, [0.05, 0.5, 0.95]).tolist()
        if len(finite)
        else None,
        "positive_fraction": float(np.mean(finite > zero)) if len(finite) else None,
        "nonpositive_fraction": float(np.mean(finite <= zero)) if len(finite) else None,
    }


def event_stack(
    values: Any, indices: Any, offsets: Any, available: Any
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(values, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    valid = np.asarray(available, dtype=bool)
    require(
        a.ndim >= 1 and valid.shape == (len(a),), "Event source availability differs"
    )
    grid = idx[:, None] + off[None, :]
    mask = (grid >= 0) & (grid < len(a))
    mask &= valid[np.clip(grid, 0, len(a) - 1)]
    result = np.full((*grid.shape, *a.shape[1:]), np.nan)
    result[mask] = a[grid[mask]]
    return result, mask


def validate_recordings(
    trace: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    events: Mapping[str, np.ndarray],
    freeze: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    c = freeze["sample_contract"]
    rows = trace.get("samples")
    require(
        trace.get("schema") == "flylab.m3.e150.native-monotone-body-trace.v1"
        and trace.get("body_hash") == freeze["body_hash"],
        "E150 trace identity differs",
    )
    require(
        isinstance(rows, list) and len(rows) == c["recorded_rows"],
        "E150 trace row count differs",
    )
    ticks = np.asarray([row.get("control_tick") for row in rows])
    times = np.asarray([row.get("time_s") for row in rows], dtype=np.float64)
    require(
        np.array_equal(ticks, np.arange(2001))
        and np.allclose(
            times, np.arange(2001) * c["control_dt_s"], atol=c["clock_atol"], rtol=0
        ),
        "Trace tick/clock differs",
    )
    for tick, row in enumerate(rows):
        interval = (
            None
            if tick == 0
            else {
                "start_tick": tick - 1,
                "end_tick": tick,
                "start_s": (tick - 1) * c["control_dt_s"],
                "end_s": tick * c["control_dt_s"],
            }
        )
        require(
            row.get("command_interval") == interval,
            f"Trace command interval differs at {tick}",
        )
    require(c["neural_ticks_per_control"] == 20, "E165 neural-to-control clock differs")
    shape = {
        "qpos": (2001, c["nq"]),
        "qvel": (2001, c["nv"]),
        "ctrl": (2001, c["nu"]),
        "joint_angles_rad": (2001, 42),
        "joint_targets_rad": (2001, 42),
        "joint_tracking_error_rad": (2001, 42),
    }
    source: dict[str, np.ndarray] = {"time_s": times}
    for key, expected in shape.items():
        a = np.asarray([row.get(key) for row in rows], dtype=np.float64)
        require(
            a.shape == expected and np.isfinite(a).all(),
            f"E150 {key} shape/finite differs",
        )
        source[key] = a
    require(
        np.allclose(
            source["joint_targets_rad"] - source["joint_angles_rad"],
            source["joint_tracking_error_rad"],
            atol=c["saved_atol"],
            rtol=0,
        ),
        "Recorded tracking error differs",
    )
    require(
        all("neural_request" in row and row["neural_request"] is None for row in rows),
        "E150 null neural requests differ",
    )
    positions = np.asarray(
        [
            [
                [row["body"]["legs"][leg][3 + segment] for segment in range(5)]
                for leg in LEGS
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    require(
        positions.shape == (2001, 6, 5, 3) and np.isfinite(positions).all(),
        "Saved tarsal UI origins differ",
    )
    source["tarsal_ui"] = positions
    source["basis"] = np.asarray(
        [row["body"]["basis"] for row in rows], dtype=np.float64
    )
    require(
        source["basis"].shape == (2001, 3, 3) and np.isfinite(source["basis"]).all(),
        "Saved body basis differs",
    )
    source["normal_load_bw"] = np.asarray(
        [row["feet"]["normal_load_bw"] for row in rows], dtype=np.float64
    )
    require(
        source["normal_load_bw"].shape == (2001, 6)
        and np.isfinite(source["normal_load_bw"]).all()
        and np.all(source["normal_load_bw"] >= 0),
        "Saved load differs",
    )
    for key, shape in (
        ("time_s", (2001,)),
        ("window", (2001,)),
        ("speed_mm_s", (2001,)),
        ("boundary_load_bw", (2001, 6)),
        ("ego_ap_velocity_mm_s", (2001, 6)),
        ("world_forward_velocity_mm_s", (2001, 6)),
    ):
        a = np.asarray(features[key])
        require(a.shape == shape, f"E152 {key} shape differs")
        source["feature_" + key] = a
    require(
        np.allclose(source["feature_time_s"], times, atol=c["clock_atol"], rtol=0),
        "E152 clock differs",
    )
    require(
        source["feature_window"].dtype == bool
        and np.array_equal(source["feature_window"], (times >= 0.5) & (times < 10.0)),
        "E152 window differs",
    )
    require(
        np.isfinite(source["feature_speed_mm_s"]).all()
        and np.isfinite(source["feature_ego_ap_velocity_mm_s"]).all()
        and np.isfinite(source["feature_world_forward_velocity_mm_s"]).all(),
        "E152 velocity nonfinite",
    )
    require(
        np.allclose(
            source["feature_boundary_load_bw"],
            source["normal_load_bw"],
            atol=c["saved_atol"],
            rtol=0,
        ),
        "E152 boundary load differs",
    )
    ei = np.asarray(events["event_index"])
    eo = np.asarray(events["offset_control"])
    require(
        ei.shape == (c["events"],)
        and ei.dtype.kind in "iu"
        and eo.shape == (41,)
        and np.array_equal(eo, c["event_offsets"]),
        "E152 events/offsets differ",
    )
    require(
        np.array_equal(np.asarray(events["offset_s"]), eo * c["control_dt_s"]),
        "E152 event clock differs",
    )
    source["event_index"] = ei
    source["event_offset"] = eo
    return rows, source


def validate_position_servo(
    gain_values: Any, bias_values: Any
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.asarray(gain_values, dtype=np.float64)
    bias = np.asarray(bias_values, dtype=np.float64)
    require(
        gain.ndim == 2
        and bias.ndim == 2
        and gain.shape[0] == bias.shape[0]
        and gain.shape[1] >= 1
        and bias.shape[1] >= 3,
        "Leg actuator gain/bias shape differs",
    )
    require(
        np.isfinite(gain).all() and np.isfinite(bias).all(),
        "Leg actuator gain/bias nonfinite",
    )
    require(
        np.all(gain[:, 0] > 0)
        and np.allclose(bias[:, 0], 0, atol=1e-12, rtol=0)
        and np.allclose(bias[:, 1], -gain[:, 0], atol=1e-12, rtol=0)
        and np.all(bias[:, 2] <= 0),
        "Leg actuator is not a position servo",
    )
    return gain, bias


def validate_layout(body: Any, prior: Mapping[str, Any]) -> dict[str, Any]:
    from tools import analyze_c_banc_m3_rm_contact_jacobian as e155

    m, mj = body.m, body.mj
    require(body.body_options.terrain == "flat", "E165 requires frozen flat terrain")
    base = e155.validate_layout(body)
    require(
        all(
            base[key] == prior[key]
            for key in (
                "actuator_ids",
                "actuator_joint_ids",
                "actuator_dof_addresses",
                "joint_names",
                "root_dof_addresses",
            )
        ),
        "E155 actuator/joint mapping differs",
    )
    require((m.nq, m.nv, m.nu) == (109, 108, 82), "Body dimensions differ")
    act = np.asarray(base["actuator_ids"], dtype=np.int64)
    joint = np.asarray(base["actuator_joint_ids"], dtype=np.int64)
    qpos = np.asarray(m.jnt_qposadr)[joint].astype(np.int64)
    require(
        np.array_equal(qpos, np.asarray(body.qpos_ids, dtype=np.int64))
        and len(np.unique(qpos)) == 42,
        "Actuator qpos mapping differs",
    )
    require(
        np.all(np.asarray(m.actuator_gaintype)[act] == int(mj.mjtGain.mjGAIN_FIXED))
        and np.all(
            np.asarray(m.actuator_biastype)[act] == int(mj.mjtBias.mjBIAS_AFFINE)
        ),
        "Leg actuator position type differs",
    )
    gain, bias = validate_position_servo(
        m.actuator_gainprm[act], m.actuator_biasprm[act]
    )
    gear = np.asarray(m.actuator_gear)[act]
    require(
        np.allclose(gear[:, 0], 1, atol=1e-12, rtol=0)
        and np.allclose(gear[:, 1:], 0, atol=1e-12, rtol=0),
        "Leg actuator gear is not unit position transmission",
    )
    require(
        not bool(int(m.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_CLAMPCTRL)),
        "MuJoCo control clamping disabled",
    )
    require(
        int(m.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_CONTACT) == 0,
        "Contact disabled",
    )
    ids = np.empty((6, 5), dtype=np.int64)
    names = []
    for li, leg in enumerate(LEGS):
        leg_names = []
        for si in range(5):
            name = f"{leg}_tarsus{si + 1}"
            require(name in body.body_indices, f"Missing body {name}")
            body_id = int(body.body_ids[body.body_indices[name]])
            model_name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body_id)
            require(
                model_name == prior["body_names"][body_id]
                and model_name.endswith("/" + name),
                f"Body ID/name differs: {name}",
            )
            ids[li, si] = body_id
            leg_names.append(name)
        names.append(leg_names)
    base.update(
        {
            "joint_qpos_addresses": qpos.tolist(),
            "tarsal_body_ids": ids.tolist(),
            "tarsal_body_names": names,
            "actuator_gainprm": gain.tolist(),
            "actuator_biasprm": bias.tolist(),
            "actuator_ctrllimited": np.asarray(m.actuator_ctrllimited)[act]
            .astype(bool)
            .tolist(),
            "actuator_ctrlrange": np.asarray(m.actuator_ctrlrange)[act].tolist(),
            "actuator_gear": gear.tolist(),
        }
    )
    return base


def new_raw(rows: int, nv: int) -> dict[str, np.ndarray]:
    return {
        "completed": np.zeros(rows, bool),
        "forward_completed": np.zeros(rows, bool),
        "jacobian_completed": np.zeros((rows, 6, 5), bool),
        "tick": np.arange(1, rows + 1),
        "time_s": np.full(rows, np.nan),
        "actual_rad": np.full((rows, 42), np.nan),
        "requests_rad": np.full((rows, 3, 42), np.nan),
        "delta_rad": np.full((rows, 3, 42), np.nan),
        "clipped": np.zeros((rows, 42), bool),
        "rotation_world": np.full((rows, 3, 3), np.nan),
        "heading_world": np.full((rows, 3), np.nan),
        "basis_residual": np.full(rows, np.nan),
        "angle_residual": np.full(rows, np.nan),
        "origin_residual": np.full((rows, 6, 5), np.nan),
        "body_origin_world_mm": np.full((rows, 6, 5, 3), np.nan),
        "body_origin_ui_mm": np.full((rows, 6, 5, 3), np.nan),
        "jacobian7_mm_per_rad": np.full((rows, 6, 5, 3, 7), np.nan),
        "other_leg_jacobian_max": np.full((rows, 6, 5), np.nan),
        "z_joint_mm": np.full((rows, 3, 6, 5, 7), np.nan),
        "ap_joint_mm": np.full((rows, 3, 6, 5, 7), np.nan),
        "xyz_mm": np.full((rows, 3, 6, 5, 3), np.nan),
        "z_mm": np.full((rows, 3, 6, 5), np.nan),
        "ap_mm": np.full((rows, 3, 6, 5), np.nan),
        "normal_load_bw": np.full((rows, 6), np.nan),
        "speed_mm_s": np.full(rows, np.nan),
        "ego_ap_velocity_mm_s": np.full((rows, 6), np.nan),
        "world_forward_velocity_mm_s": np.full((rows, 6), np.nan),
    }


def process_rows(
    rows: list[dict[str, Any]],
    source: Mapping[str, np.ndarray],
    body: Any,
    layout: Mapping[str, Any],
    raw: dict[str, np.ndarray],
    ledger: dict[str, Any],
    started: float,
    limit_s: float,
    reserve_s: float,
    errors: list[str] | None = None,
) -> list[str]:
    m, mj = body.m, body.mj
    d = mj.MjData(m)
    ledger["shadow_mjdata"] += 1
    act = np.asarray(layout["actuator_ids"], dtype=np.int64)
    qids = np.asarray(layout["joint_qpos_addresses"], dtype=np.int64)
    dofs = np.asarray(layout["actuator_dof_addresses"], dtype=np.int64).reshape(6, 7)
    ids = np.asarray(layout["tarsal_body_ids"], dtype=np.int64)
    all_dofs = dofs.reshape(-1)
    limited = np.asarray(layout["actuator_ctrllimited"], dtype=bool)
    ranges = np.asarray(layout["actuator_ctrlrange"], dtype=np.float64)
    if errors is None:
        errors = []
    for index, row in enumerate(rows[1:]):
        if time.perf_counter() - started > limit_s - reserve_s:
            raise TimeoutError("E165 reserve elapsed before next shadow forward")
        tick = index + 1
        raw["time_s"][index] = float(row["time_s"])
        actual = source["joint_angles_rad"][tick]
        raw["actual_rad"][index] = actual
        requests, clipped = effective_requests(
            source["joint_targets_rad"][tick],
            source["ctrl"][tick, act],
            limited,
            ranges,
        )
        raw["requests_rad"][index] = requests
        raw["delta_rad"][index] = requests - actual[None, :]
        raw["clipped"][index] = clipped
        raw["normal_load_bw"][index] = source["normal_load_bw"][tick]
        raw["speed_mm_s"][index] = source["feature_speed_mm_s"][tick]
        raw["ego_ap_velocity_mm_s"][index] = source["feature_ego_ap_velocity_mm_s"][
            tick
        ]
        raw["world_forward_velocity_mm_s"][index] = source[
            "feature_world_forward_velocity_mm_s"
        ][tick]
        d.qpos[:] = source["qpos"][tick]
        d.qvel[:] = source["qvel"][tick]
        d.ctrl[:] = source["ctrl"][tick]
        d.time = float(getattr(body, "t0", 0.0)) + float(row["time_s"])
        ledger["forward_entered"] += 1
        mj.mj_forward(m, d)
        ledger["forward_completed"] += 1
        raw["forward_completed"][index] = True
        rotation = np.asarray(d.xmat[body.thorax]).reshape(3, 3).copy()
        raw["rotation_world"][index] = rotation
        heading = horizontal_heading(rotation)
        raw["heading_world"][index] = heading
        basis_residual = float(np.max(np.abs(S @ rotation - source["basis"][tick])))
        raw["basis_residual"][index] = basis_residual
        if basis_residual > 1e-12:
            errors.append(f"control {tick} thorax basis mismatch")
        angle_residual = float(np.max(np.abs(source["qpos"][tick, qids] - actual)))
        raw["angle_residual"][index] = angle_residual
        if angle_residual > 1e-12:
            errors.append(f"control {tick} actual qpos mismatch")
        for li in range(6):
            other = np.setdiff1d(all_dofs, dofs[li], assume_unique=True)
            for si in range(5):
                body_id = int(ids[li, si])
                point = np.asarray(d.xpos[body_id]).copy()
                raw["body_origin_world_mm"][index, li, si] = point
                raw["body_origin_ui_mm"][index, li, si] = S @ point
                origin_residual = float(
                    np.max(np.abs(S @ point - source["tarsal_ui"][tick, li, si]))
                )
                raw["origin_residual"][index, li, si] = origin_residual
                if origin_residual > 1e-12:
                    errors.append(
                        f"control {tick} {LEGS[li]} tarsus{si + 1} UI origin mismatch"
                    )
                jac = np.zeros((3, m.nv), dtype=np.float64)
                ledger["jacobian_entered"] += 1
                mj.mj_jacBody(m, d, jac, None, body_id)
                ledger["jacobian_completed"] += 1
                raw["jacobian_completed"][index, li, si] = True
                j7 = jac[:, dofs[li]]
                raw["jacobian7_mm_per_rad"][index, li, si] = j7
                residual = float(np.max(np.abs(jac[:, other])))
                raw["other_leg_jacobian_max"][index, li, si] = residual
                if residual > 1e-12:
                    errors.append(
                        f"control {tick} {LEGS[li]} tarsus{si + 1} other-leg Jacobian residual"
                    )
                values = project_command(
                    j7, raw["delta_rad"][index, :, li * 7 : (li + 1) * 7], heading
                )
                for key in ("xyz_mm", "z_joint_mm", "ap_joint_mm", "z_mm", "ap_mm"):
                    raw[key][index, :, li, si] = values[key]
        raw["completed"][index] = True
        ledger["rows_completed"] += 1
    return errors


def numeric_checks(raw: Mapping[str, np.ndarray], contract: Mapping[str, Any]) -> None:
    require(
        raw["completed"].all()
        and raw["forward_completed"].all()
        and raw["jacobian_completed"].all(),
        "Incomplete E165 endpoint or Jacobian",
    )
    for name, values in raw.items():
        if values.dtype.kind in "fc":
            require(np.isfinite(values).all(), f"Nonfinite E165 raw field: {name}")
    require(
        np.isfinite(raw["jacobian7_mm_per_rad"]).all()
        and np.isfinite(raw["z_mm"]).all()
        and np.isfinite(raw["ap_mm"]).all(),
        "Nonfinite E165 Jacobian/direction",
    )
    require(
        np.max(raw["other_leg_jacobian_max"]) <= contract["saved_atol"],
        "Other-leg Jacobian residual",
    )
    require(
        np.allclose(
            raw["z_joint_mm"].sum(axis=-1),
            raw["z_mm"],
            atol=contract["saved_atol"],
            rtol=0,
        ),
        "Z per-joint closure differs",
    )
    require(
        np.allclose(
            raw["ap_joint_mm"].sum(axis=-1),
            raw["ap_mm"],
            atol=contract["saved_atol"],
            rtol=0,
        ),
        "AP per-joint closure differs",
    )


def summaries(
    raw: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    signs = segment_signs(raw["z_mm"], contract["zero_mm"])
    ap_extent = segment_signs(raw["ap_mm"], contract["zero_mm"])
    speed = raw["speed_mm_s"]
    window = source["feature_window"][1:]
    output: dict[str, Any] = {}
    for li, leg in enumerate(LEGS):
        load = raw["normal_load_bw"][:, li]
        supported = load >= contract["support_threshold_bw"]
        masks = {
            "all": window,
            "forward": window & (speed > contract["forward_threshold_mm_s"]),
            "interrupted": window & (speed <= contract["forward_threshold_mm_s"]),
            "supported": window & supported,
            "supported_interrupted": window
            & supported
            & (speed <= contract["forward_threshold_mm_s"]),
            "supported_interrupted_recovery": window
            & supported
            & (speed <= contract["forward_threshold_mm_s"])
            & (raw["ego_ap_velocity_mm_s"][:, li] > 0),
        }
        output[leg] = {}
        for name, mask in masks.items():
            group: dict[str, Any] = {"rows": int(mask.sum()), "kinds": {}}
            for ki, kind in enumerate(KINDS):
                z = raw["z_mm"][mask, ki, li]
                ap = raw["ap_mm"][mask, ki, li]
                item = {
                    "z_mm": stats(z, contract["zero_mm"]),
                    "ap_mm": stats(ap, contract["zero_mm"]),
                    "z_by_segment_mm": [
                        stats(z[:, si], contract["zero_mm"]) for si in range(5)
                    ],
                    "ap_by_segment_mm": [
                        stats(ap[:, si], contract["zero_mm"]) for si in range(5)
                    ],
                    "segment_min_z_mm": stats(
                        signs["minimum"][mask, ki, li], contract["zero_mm"]
                    ),
                    "segment_max_z_mm": stats(
                        signs["maximum"][mask, ki, li], contract["zero_mm"]
                    ),
                    "segment_mean_z_mm": stats(
                        signs["mean"][mask, ki, li], contract["zero_mm"]
                    ),
                    "segment_min_ap_mm": stats(
                        ap_extent["minimum"][mask, ki, li], contract["zero_mm"]
                    ),
                    "segment_max_ap_mm": stats(
                        ap_extent["maximum"][mask, ki, li], contract["zero_mm"]
                    ),
                    "segment_mean_ap_mm": stats(
                        ap_extent["mean"][mask, ki, li], contract["zero_mm"]
                    ),
                    "all_positive_fraction": float(
                        signs["all_positive"][mask, ki, li].mean()
                    )
                    if mask.any()
                    else None,
                    "all_nonpositive_fraction": float(
                        signs["all_nonpositive"][mask, ki, li].mean()
                    )
                    if mask.any()
                    else None,
                    "mixed_fraction": float(signs["mixed"][mask, ki, li].mean())
                    if mask.any()
                    else None,
                }
                if name.startswith("supported"):
                    item["load_weighted_z_mm"] = [
                        weighted_mean(z[:, si], load[mask]) for si in range(5)
                    ]
                    item["load_weighted_ap_mm"] = [
                        weighted_mean(ap[:, si], load[mask]) for si in range(5)
                    ]
                    item["load_weighted_z_evidence"] = [
                        weighted_evidence(z[:, si], load[mask]) for si in range(5)
                    ]
                    item["load_weighted_ap_evidence"] = [
                        weighted_evidence(ap[:, si], load[mask]) for si in range(5)
                    ]
                group["kinds"][kind] = item
            group["requested_delivered_delta_rad"] = stats(
                (
                    raw["requests_rad"][mask, 0, li * 7 : (li + 1) * 7]
                    - raw["requests_rad"][mask, 1, li * 7 : (li + 1) * 7]
                ).ravel()
            )
            group["delivered_effective_delta_rad"] = stats(
                (
                    raw["requests_rad"][mask, 1, li * 7 : (li + 1) * 7]
                    - raw["requests_rad"][mask, 2, li * 7 : (li + 1) * 7]
                ).ravel()
            )
            group["clipped_control_count"] = int(
                raw["clipped"][mask, li * 7 : (li + 1) * 7].sum()
            )
            group["clipped_sample_count"] = int(
                np.any(raw["clipped"][mask, li * 7 : (li + 1) * 7], axis=1).sum()
            )
            output[leg][name] = group
    return output


def make_events(
    raw: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    zero_mm: float = 1e-12,
) -> dict[str, np.ndarray]:
    idx = source["event_index"]
    off = source["event_offset"]
    available = np.zeros(2001, bool)
    available[1:] = raw["completed"]
    fields = {
        "z_mm": raw["z_mm"],
        "ap_mm": raw["ap_mm"],
        "normal_load_bw": raw["normal_load_bw"],
        "ego_ap_velocity_mm_s": raw["ego_ap_velocity_mm_s"],
        "world_forward_velocity_mm_s": raw["world_forward_velocity_mm_s"],
    }
    signs = segment_signs(raw["z_mm"], zero_mm)
    ap_extent = segment_signs(raw["ap_mm"], zero_mm)
    fields.update(
        {
            "segment_min_z_mm": signs["minimum"],
            "segment_max_z_mm": signs["maximum"],
            "segment_mean_z_mm": signs["mean"],
            "segment_min_ap_mm": ap_extent["minimum"],
            "segment_max_ap_mm": ap_extent["maximum"],
            "segment_mean_ap_mm": ap_extent["mean"],
        }
    )
    result: dict[str, np.ndarray] = {
        "event_index": idx,
        "offset_control": off,
        "offset_s": off * 0.005,
    }
    for name, values in fields.items():
        expanded = np.full((2001, *values.shape[1:]), np.nan)
        expanded[1:] = values
        stack, mask = event_stack(expanded, idx, off, available)
        result[name] = stack
        result["available"] = mask
    return result


def output_manifest(output: Path) -> dict[str, Any]:
    files = {
        p.relative_to(output).as_posix(): {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "output-manifest.json"
    }
    return {
        "schema": SCHEMA + ".output-manifest",
        "files": files,
        "total_bytes": sum(v["bytes"] for v in files.values()),
    }


def snapshot_sources(
    out: Path,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    source_review: Path,
    source_review_sha: str,
    review: Mapping[str, Any],
) -> dict[str, str]:
    sources = {
        "design.md": safe_path(freeze["design"]),
        "freeze.json": freeze_path,
        "source-review.json": source_review,
        "analyzer.py": ROOT / OWNED[0],
        "tests.py": ROOT / OWNED[1],
    }
    expected = {
        "design.md": freeze["design_sha256"],
        "freeze.json": FREEZE_SHA,
        "source-review.json": source_review_sha,
        "analyzer.py": review["source_sha256"][OWNED[0]],
        "tests.py": review["source_sha256"][OWNED[1]],
    }
    directory = out / "source-snapshot"
    directory.mkdir()
    for name, path in sources.items():
        require(
            sha256(path) == expected[name],
            f"Approved source changed before copy: {name}",
        )
        (directory / name).write_bytes(path.read_bytes())
        require(
            sha256(directory / name) == expected[name],
            f"Approved source copy differs: {name}",
        )
    return expected


def verify_snapshot_sources(
    out: Path,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    source_review: Path,
    approved: Mapping[str, str],
) -> None:
    sources = {
        "design.md": safe_path(freeze["design"]),
        "freeze.json": freeze_path,
        "source-review.json": source_review,
        "analyzer.py": ROOT / OWNED[0],
        "tests.py": ROOT / OWNED[1],
    }
    for name, path in sources.items():
        require(sha256(path) == approved[name], f"Approved live source changed: {name}")
        require(
            sha256(out / "source-snapshot" / name) == approved[name],
            f"Approved source snapshot changed: {name}",
        )


def actor_integration_state(body: Any) -> np.ndarray:
    mj, m = body.mj, body.m
    state_type = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(m, state_type), dtype=np.float64)
    mj.mj_getState(m, body.d, state, state_type)
    return state


def validate_parameter_identity(
    trace_parameters: Any, saved_parameters: Any
) -> dict[str, Any]:
    require(
        isinstance(trace_parameters, dict) and isinstance(saved_parameters, dict),
        "Checkpoint/trace parameters must be dictionaries",
    )
    # E141's raw saver appends diagnostic=spec; E150 reuses that saver unchanged.
    recorded = dict(trace_parameters)
    diagnostic = recorded.pop("diagnostic", None)
    require(
        isinstance(diagnostic, dict) and diagnostic.get("experiment") == "E150",
        "E150 trace diagnostic provenance differs",
    )
    require(recorded == saved_parameters, "Checkpoint/trace model parameters differ")
    return diagnostic


def publish_success(
    out: Path,
    pending: Mapping[str, Any],
    limits: Mapping[str, Any],
    started: float,
    *,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Commit success only after every terminal write and resource check succeeds."""
    success = dict(pending)
    success.update(
        {
            "valid_diagnostic": True,
            "publication_complete": True,
            "resource_limit_met": True,
            "diagnostic_rows": 2000,
            "kinematic_forward_calls": 2000,
            "jacobian_calls": 60000,
        }
    )
    try:
        atomic_json(out / "result.json", success)
        manifest = output_manifest(out)
        atomic_json(out / "output-manifest.json", manifest)
        require(
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            <= limits["raw_bytes"],
            "E165 raw output byte limit exceeded",
        )
        require(
            shutil.disk_usage(out).free >= limits["free_bytes"],
            "E165 final free-space reserve unavailable",
        )
        require(
            clock() - started <= limits["internal_s"],
            "E165 final save exceeded internal time",
        )
    except BaseException as primary:  # Invalidate receipts even on interruption.
        manifest_path = out / "output-manifest.json"
        if manifest_path.exists():
            try:
                os.replace(manifest_path, out / "output-manifest.invalidated.json")
            except BaseException as secondary:  # noqa: BLE001 - retain the first publication error
                primary.add_note(
                    "Secondary E165 manifest invalidation failure: " + repr(secondary)
                )
        receipt = out / "result.json"
        if receipt.exists():
            try:
                current = json.loads(receipt.read_text(encoding="utf-8"))
                if current.get("valid_diagnostic") is True:
                    os.replace(receipt, out / "result.invalidated.json")
                    atomic_json(
                        receipt,
                        dict(
                            pending, valid_diagnostic=False, publication_complete=False
                        ),
                    )
            except BaseException as secondary:  # noqa: BLE001 - retain the first publication error
                primary.add_note(
                    "Secondary E165 result invalidation failure: " + repr(secondary)
                )
        raise
    return success


def run(
    freeze_path: Path,
    freeze_sha: str,
    source_review: Path,
    source_review_sha: str,
    out: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    require(not out.exists(), "E165 output must be a new directory")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validate_freeze(freeze, freeze_path, freeze_sha)
    review = validate_gate(source_review, source_review_sha, freeze_sha)
    hashes = hash_map(freeze)
    verify_hashes(hashes)
    limits = freeze["resource_limits"]
    require(
        shutil.disk_usage(out.parent).free >= limits["free_bytes"],
        "E165 free-space reserve unavailable",
    )
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    require(
        all(p.get("num_threads") == 1 for p in pools), "E165 CPU threadpool exceeds one"
    )
    require(
        time.perf_counter() - started <= limits["internal_s"] - limits["reserve_s"],
        "E165 preflight reserve elapsed",
    )
    out.mkdir(parents=True, exist_ok=False)
    ledger = {
        "original_processes": 1,
        "automatic_retries": 0,
        "cpu_threads": 1,
        "gpu_count": 0,
        "network_models": 0,
        "state_store_reads": 0,
        "graph_store_instances": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "body_models": 0,
        "settling_physics_steps": 0,
        "shadow_mjdata": 0,
        "forward_entered": 0,
        "forward_completed": 0,
        "jacobian_entered": 0,
        "jacobian_completed": 0,
        "rows_completed": 0,
        "scientific_physics_steps": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "thread_pools": pools,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "valid_diagnostic": False,
        "publication_complete": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "interpretation_limit": "Static local body-origin directions only; no contact-point, future-clearance, neural-causal or gait-quality claim.",
    }
    raw = new_raw(2000, 108)
    body = None
    primary: BaseException | None = None
    secondary: list[str] = []
    errors: list[str] = []
    actor_before: np.ndarray | None = None
    approved: dict[str, str] | None = None
    posthash_attempted = False
    postsnapshot_attempted = False
    try:
        approved = snapshot_sources(
            out, freeze, freeze_path, source_review, source_review_sha, review
        )
        snapshot = {
            "freeze_sha256": freeze_sha,
            "design_sha256": freeze["design_sha256"],
            "source_review_sha256": source_review_sha,
            "source_sha256": dict(review["source_sha256"]),
            "frozen_sha256": hashes,
            "copied_sha256": approved,
        }
        atomic_json(out / "source-snapshot.json", snapshot)
        atomic_json(out / "result.json", result)
        from flylab.c.storage import StateStore
        from tools import analyze_c_banc_m3_signed_contact as e154
        from tools import analyze_c_banc_observation as observation

        trace = json.loads(
            safe_path(freeze["paths"]["trace"]).read_text(encoding="utf-8")
        )
        context = json.loads(
            safe_path(freeze["paths"]["context"]).read_text(encoding="utf-8")
        )
        prior = json.loads(
            safe_path(freeze["paths"]["prior_layout"]).read_text(encoding="utf-8")
        )
        with np.load(
            safe_path(freeze["paths"]["features"]), allow_pickle=False
        ) as pack:
            features = {k: pack[k] for k in pack.files}
        with np.load(safe_path(freeze["paths"]["events"]), allow_pickle=False) as pack:
            events = {k: pack[k] for k in pack.files}
        rows, source = validate_recordings(trace, features, events, freeze)
        require(
            context["body_hash"] == freeze["body_hash"]
            and prior["body_hash"] == freeze["body_hash"],
            "E154/E155 body identity differs",
        )
        saved = StateStore.load(safe_path(freeze["paths"]["checkpoint"]))
        ledger["state_store_reads"] = 1
        require(
            saved.get("body", {}).get("modelHash") == freeze["body_hash"],
            "Checkpoint body hash differs",
        )
        require(
            type(saved.get("seed")) is int
            and isinstance(saved.get("world"), dict)
            and isinstance(saved.get("parameters"), dict),
            "Checkpoint seed/world/parameters identity differs",
        )
        diagnostic = validate_parameter_identity(
            trace.get("parameters"), saved["parameters"]
        )
        require(
            e154.validate_body_parameters(saved) == context["body_parameters"],
            "Checkpoint body parameters differ",
        )
        require(
            time.perf_counter() - started <= limits["internal_s"] - limits["reserve_s"],
            "E165 construction reserve elapsed",
        )
        body = e154.construct_body_safely(saved, observation, ledger)
        ledger["body_models"] = 1
        require(
            ledger["body_constructor_completed"] == 1
            and ledger["settling_physics_steps"] == 2000,
            "Constructor settling count differs",
        )
        require(
            e154.validate_settling_count(
                float(body.t0), float(body.m.opt.timestep), 2000
            )
            == 2000,
            "Settling clock differs",
        )
        require(
            body.model_hash == freeze["body_hash"]
            and body.world_spec == saved["world"],
            "Constructed body/world identity differs",
        )
        require(
            saved.get("control_tick") == 0,
            "Checkpoint must be the E150 control-zero anchor",
        )
        for key in ("qpos", "qvel", "ctrl"):
            require(
                np.allclose(
                    np.asarray(getattr(body.d, key)),
                    source[key][0],
                    atol=freeze["sample_contract"]["saved_atol"],
                    rtol=0,
                ),
                f"Checkpoint control-zero {key} differs",
            )
        require(
            abs(float(body.physics_time()) - float(rows[0]["time_s"]))
            <= freeze["sample_contract"]["clock_atol"],
            "Checkpoint control-zero time differs",
        )
        feet, _ = body._force_maps["feet"]
        require(
            np.array_equal(feet, context["feet_map"])
            and sorted(map(int, body.support_geom_ids)) == context["support_geom_ids"]
            and float(body.weight0) == float(context["weight0"]),
            "E154 contact model differs",
        )
        layout = validate_layout(body, prior)
        layout["checkpoint_seed"] = saved["seed"]
        layout["checkpoint_world_sha256"] = hashlib.sha256(
            json.dumps(saved["world"], sort_keys=True).encode()
        ).hexdigest()
        layout["checkpoint_parameters_sha256"] = hashlib.sha256(
            json.dumps(saved["parameters"], sort_keys=True).encode()
        ).hexdigest()
        layout["trace_diagnostic_provenance"] = diagnostic
        atomic_json(out / "layout.json", layout)
        actor_before = actor_integration_state(body)
        errors = process_rows(
            rows,
            source,
            body,
            layout,
            raw,
            ledger,
            started,
            limits["internal_s"],
            limits["reserve_s"],
            errors,
        )
        atomic_npz(out / "command-lift-raw.npz", raw)
        require(not errors, "; ".join(errors[:12]))
        numeric_checks(raw, freeze["sample_contract"])
        event_arrays = make_events(raw, source, freeze["sample_contract"]["zero_mm"])
        atomic_npz(out / "event-stacks-raw.npz", event_arrays)
        summary = {
            "schema": SCHEMA + ".summary",
            "exploratory_only": True,
            "body_origins_not_contacts": True,
            "groups": summaries(raw, source, freeze["sample_contract"]),
            "events": {
                "count": len(source["event_index"]),
                "complete_event_count": int(
                    event_arrays["available"].all(axis=1).sum()
                ),
                "unavailable_samples": int((~event_arrays["available"]).sum()),
            },
        }
        atomic_json(out / "summary.json", summary)
        actor_after = actor_integration_state(body)
        atomic_npz(
            out / "actor-preservation-raw.npz",
            {"before": actor_before, "after": actor_after},
        )
        require(
            np.array_equal(actor_before, actor_after),
            "Original actor integration state changed",
        )
        body.close()
        body = None
        posthash_attempted = True
        verify_hashes(hashes)
        postsnapshot_attempted = True
        verify_snapshot_sources(out, freeze, freeze_path, source_review, approved)
        require(
            ledger["forward_completed"] == 2000
            and ledger["jacobian_completed"] == 60000
            and ledger["rows_completed"] == 2000,
            "E165 execution count differs",
        )
        atomic_json(out / "ledger.json", ledger)
        atomic_json(
            out / "timing.json",
            {
                "wall_s_before_final_publication": time.perf_counter() - started,
                "internal_s": limits["internal_s"],
                "external_s": limits["external_s"],
                "supervisor_s": limits["supervisor_s"],
                "prior_conservative_s": freeze["prior_conservative_s"],
                "prospective_cumulative_s": freeze["prospective_cumulative_s"],
            },
        )
        require(
            time.perf_counter() - started <= limits["internal_s"],
            "E165 pre-publication exceeded internal time",
        )
        result = publish_success(out, result, limits, started)
        return result
    except BaseException as error:  # noqa: BLE001 - preserve primary error through mandatory cleanup
        primary = error
    finally:
        if body is not None:
            if actor_before is not None:
                try:
                    actor_after = actor_integration_state(body)
                    atomic_npz(
                        out / "actor-preservation-partial.npz",
                        {"before": actor_before, "after": actor_after},
                    )
                    if not np.array_equal(actor_before, actor_after):
                        secondary.append("Original actor integration state changed")
                except BaseException as error:  # noqa: BLE001 - secondary actor check must not mask primary
                    secondary.append("actor_state: " + repr(error))
            try:
                body.close()
            except BaseException as error:  # noqa: BLE001 - secondary cleanup must not mask primary
                secondary.append("body_close: " + repr(error))
        if not posthash_attempted:
            try:
                verify_hashes(hashes)
            except BaseException as error:  # noqa: BLE001 - retain primary on post-hash failure
                secondary.append("post_hashes: " + repr(error))
        if approved is not None and not postsnapshot_attempted:
            try:
                verify_snapshot_sources(
                    out, freeze, freeze_path, source_review, approved
                )
            except BaseException as error:  # noqa: BLE001 - retain primary on snapshot failure
                secondary.append("post_snapshot: " + repr(error))
        if primary is not None or secondary:
            if primary is None:
                primary = RuntimeError("E165 body cleanup failed")
            result.update(
                {
                    "valid_diagnostic": False,
                    "publication_complete": False,
                    # Failed publication did not verify all final resource limits.
                    "resource_limit_met": None,
                    "elapsed_s_at_failure": time.perf_counter() - started,
                    "error": {"type": type(primary).__name__, "message": str(primary)},
                    "secondary_errors": secondary,
                }
            )
            try:
                atomic_json(
                    out / "input-failure-evidence.json",
                    {
                        "stage_error": str(primary),
                        "source_paths": freeze["paths"],
                        "approved_source_sha256": review["source_sha256"],
                        "comparison_errors": errors,
                        "secondary_errors": secondary,
                    },
                )
            except BaseException as error:  # noqa: BLE001 - preserve primary if evidence cannot save
                secondary.append("input_evidence: " + repr(error))
            for name, value in (("command-lift-partial.npz", raw),):
                try:
                    atomic_npz(out / name, value)
                except BaseException as error:  # noqa: BLE001 - preserve primary if partial cannot save
                    secondary.append(name + ": " + repr(error))
            for name, value in (("ledger.json", ledger), ("result.json", result)):
                try:
                    atomic_json(out / name, value)
                except BaseException as error:  # noqa: BLE001 - preserve primary if receipt cannot save
                    secondary.append(name + ": " + repr(error))
            try:
                (out / "traceback.txt").write_text(
                    "".join(traceback.format_exception(primary)), encoding="utf-8"
                )
            except BaseException as error:  # noqa: BLE001 - preserve primary if traceback cannot save
                secondary.append("traceback: " + repr(error))
            for note in secondary:
                primary.add_note("Secondary E165 persistence/cleanup error: " + note)
    assert primary is not None
    raise primary.with_traceback(primary.__traceback__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
