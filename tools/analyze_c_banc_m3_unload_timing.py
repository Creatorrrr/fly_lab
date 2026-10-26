"""E161 read-only analysis of saved RM unloading timing.

This module consumes frozen E157/E150/E152 arrays only.  Its muscle-command
projection is a kinematic diagnostic at the recorded geometry; it is not a
muscle force, an individual-muscle velocity, a causal intervention, or a
biological validation.
"""

# Thread limits must precede NumPy.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    "tools/analyze_c_banc_m3_unload_timing.py",
    "tests/test_c_banc_unload_timing.py",
)
SCHEMA = "flylab.m3.e161.unload-timing-result.v1"
FREEZE_SCHEMA = "flylab.m3.e161.unload-timing.v1"
FREEZE_SHA = "b7928b44eafa39f2b5a9fb5d0b7bfc2425bf292c17872de425cf89cecf37b947"
DESIGN_SHA = "de7be1e71daa785464b6bef6a4ed69bb6cada0b88a115f38d81a0b1abaaaec61"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
RM = 4
SUPPORT_BW = 0.02
NATIVE_DT = 0.0001
CONTROL_DT = 0.005
STABLE_RADIUS = 100
WINDOW_RADIUS = 200
CLOSURE_ATOL = 1e-10
LOAD_ATOL = 1e-8
RAW_LIMIT = 128 * 1024 * 1024
FREE_LIMIT = 1 * 1024 * 1024 * 1024
INTERNAL_S = 35.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(relative: str) -> Path:
    item = Path(relative)
    resolved = (ROOT / item).resolve()
    require(not item.is_absolute() and ".." not in item.parts, "Unsafe frozen path")
    require(resolved.is_relative_to(ROOT), "Frozen path leaves repository")
    return resolved


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    require(
        out.shape == shape and np.isfinite(out).all(), f"Invalid {label} shape/value"
    )
    return out


def _verify_map(entries: Mapping[str, str], label: str) -> None:
    require(type(entries) is dict and entries, f"Missing frozen {label} hash map")
    for relative, digest in entries.items():
        require(
            type(relative) is str and type(digest) is str and len(digest) == 64,
            f"Malformed {label} hash entry",
        )
        require(
            sha256(safe_path(relative)) == digest, f"Frozen hash mismatch: {relative}"
        )


def verified_inputs(
    freeze_path: Path,
    freeze_sha256: str,
    source_review_path: Path,
    source_review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify only hashes/contracts; deliberately does not open any NPZ source."""
    require(
        freeze_sha256 == FREEZE_SHA and sha256(freeze_path) == FREEZE_SHA,
        "Exact E161 freeze SHA argument required",
    )
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    require(freeze.get("schema") == FREEZE_SCHEMA, "Unexpected E161 freeze schema")
    require(
        freeze.get("owned_files") == list(OWNED), "E161 owned file contract changed"
    )
    design = safe_path(freeze["design"])
    require(
        freeze.get("design_sha256") == DESIGN_SHA and sha256(design) == DESIGN_SHA,
        "E161 design changed",
    )
    contract = freeze.get("sample_contract", {})
    expected = {
        "native_samples": 10000,
        "positive_contacts": 24420,
        "first_control_tick": 201,
        "last_control_tick": 400,
        "native_dt": NATIVE_DT,
        "control_dt": CONTROL_DT,
        "support_threshold_bw": SUPPORT_BW,
        "stable_before_steps": STABLE_RADIUS,
        "stable_after_steps": STABLE_RADIUS,
        "event_half_window_steps": WINDOW_RADIUS,
        "closure_atol": CLOSURE_ATOL,
        "load_atol": LOAD_ATOL,
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"E161 sample contract differs: {key}")
    resources = freeze.get("resource_limits", {})
    require(
        resources.get("internal_s") == 35
        and resources.get("external_s") == 45
        and resources.get("supervisor_s") == 55
        and resources.get("raw_bytes") == RAW_LIMIT
        and resources.get("free_bytes") == FREE_LIMIT
        and resources.get("cpu_threads") == 1
        and resources.get("gpu_count") == 0,
        "E161 resource contract changed",
    )
    counts = freeze.get("counts", {})
    require(
        counts.get("original_processes") == 1
        and counts.get("automatic_retries") == 0
        and counts.get("body_models") == 0
        and counts.get("physics_steps") == 0
        and counts.get("neural_steps") == 0
        and counts.get("fit_calls") == 0,
        "E161 execution boundary changed",
    )
    require(
        freeze.get("diagnostic_only") is True and counts.get("F6_exposed") is False,
        "E161 outcome boundary changed",
    )
    expected_counts = {
        "input_sha256": 206,
        "reused_source_sha256": 23,
        "preservation_sha256": 3136,
    }
    for name, count in expected_counts.items():
        require(len(freeze[name]) == count, f"Frozen {name} inventory count differs")
        _verify_map(freeze[name], name)

    require(
        len(source_review_sha256) == 64
        and sha256(source_review_path) == source_review_sha256,
        "Exact E161 source-review SHA argument required",
    )
    review = json.loads(Path(source_review_path).read_text(encoding="utf-8"))
    require(
        review.get("freeze_sha256") == FREEZE_SHA,
        "E161 source-review freeze SHA differs",
    )
    require(
        review.get("blocking_defects") == [], "E161 source gate has blocking defects"
    )
    require(
        review.get("accepts_single_saved_diagnostic") is True,
        "Parent did not approve one E161 saved diagnostic",
    )
    require(
        type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1,
        "E161 gate must allow exactly one original process",
    )
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("source_sha256") == expected_sources,
        "E161 source-review source map differs",
    )
    return freeze, review


def decompose_world_z(
    relative_jacobian: Any, qvel: Any, leg: Any, root_dofs: Any, leg_dofs: Any
) -> dict[str, np.ndarray]:
    jac = np.asarray(relative_jacobian, dtype=np.float64)
    require(
        jac.ndim == 3 and jac.shape[1] == 3 and np.isfinite(jac).all(),
        "Relative Jacobian shape/value differs",
    )
    n, _, nv = jac.shape
    velocity = finite(qvel, (n, nv), "contact qvel")
    owner = np.asarray(leg, dtype=np.int64)
    roots = np.asarray(root_dofs, dtype=np.int64)
    legs = np.asarray(leg_dofs, dtype=np.int64)
    require(
        owner.shape == (n,) and np.all((owner >= 0) & (owner < 6)),
        "Contact leg differs",
    )
    require(roots.shape == (6,) and legs.shape == (6, 7), "DOF layout differs")
    require(len(np.unique(np.r_[roots, legs.ravel()])) == 48, "DOF layout overlaps")
    require(
        np.all((roots >= 0) & (roots < nv)) and np.all((legs >= 0) & (legs < nv)),
        "DOF address out of range",
    )
    product = jac[:, 2, :] * velocity
    total = product.sum(axis=1)
    root = product[:, roots].sum(axis=1)
    own7 = np.empty((n, 7), dtype=np.float64)
    residual = np.empty(n, dtype=np.float64)
    all_dofs = np.arange(nv)
    for leg_index in range(6):
        rows = owner == leg_index
        own7[rows] = product[rows][:, legs[leg_index]]
        residual_dofs = np.setdiff1d(
            all_dofs, np.r_[roots, legs[leg_index]], assume_unique=False
        )
        residual[rows] = product[rows][:, residual_dofs].sum(axis=1)
    return {
        "contact_z_dof": product,
        "contact_z_total": total,
        "contact_z_root": root,
        "contact_z_own7": own7,
        "contact_z_residual": residual,
        "contact_z_closure_error": root + own7.sum(axis=1) + residual - total,
    }


def weighted_native_leg(
    native_index: Any,
    leg: Any,
    weight: Any,
    values: Mapping[str, Any],
    native_samples: int,
) -> dict[str, np.ndarray]:
    index = np.asarray(native_index, dtype=np.int64)
    owner = np.asarray(leg, dtype=np.int64)
    load = np.asarray(weight, dtype=np.float64)
    n = len(index)
    require(
        index.shape == owner.shape == load.shape == (n,), "Contact index shape differs"
    )
    require(
        np.all((index >= 0) & (index < native_samples))
        and np.all((owner >= 0) & (owner < 6)),
        "Contact index out of range",
    )
    require(
        np.isfinite(load).all() and np.all(load > 0),
        "Positive finite contact loads required",
    )
    flat = index * 6 + owner
    load_sum = np.bincount(flat, weights=load, minlength=native_samples * 6).reshape(
        native_samples, 6
    )
    available = load_sum > 0
    out: dict[str, np.ndarray] = {
        "contact_load_sum": load_sum,
        "contact_available": available,
    }
    for name, raw in values.items():
        arr = np.asarray(raw, dtype=np.float64)
        require(
            arr.shape[0] == n and np.isfinite(arr).all(),
            f"Invalid contact value: {name}",
        )
        tail = arr.shape[1:]
        mean = np.full((native_samples, 6) + tail, np.nan, dtype=np.float64)
        for suffix in np.ndindex(tail or (1,)):
            source = arr if not tail else arr[(slice(None),) + suffix]
            sums = np.bincount(
                flat, weights=load * source, minlength=native_samples * 6
            ).reshape(native_samples, 6)
            target = mean if not tail else mean[(slice(None), slice(None)) + suffix]
            np.divide(sums, load_sum, out=target, where=available)
        out[name] = mean
    return out


def find_unload_transitions(
    load: Any,
    *,
    threshold: float = SUPPORT_BW,
    stable_radius: int = STABLE_RADIUS,
    window_radius: int = WINDOW_RADIUS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(load, dtype=np.float64)
    require(
        values.ndim == 1 and np.isfinite(values).all() and np.all(values >= 0),
        "Finite nonnegative RM loads required",
    )
    raw = np.flatnonzero((values[:-1] >= threshold) & (values[1:] < threshold)) + 1
    included = []
    reasons: list[str] = []
    for event in raw:
        if event - window_radius < 0 or event + window_radius >= len(values):
            reasons.append("incomplete_alignment_window")
        elif not np.all(values[event - stable_radius : event] >= threshold):
            reasons.append("unstable_pre_support")
        elif not np.all(values[event : event + stable_radius] < threshold):
            reasons.append("unstable_post_nonsupport")
        else:
            reasons.append("included")
            included.append(int(event))
    return raw.astype(np.int64), np.asarray(included, dtype=np.int64), reasons


def command_rate_projection(
    activation: Any,
    control_ticks: Any,
    moment: Any,
    contact_ticks: Any,
    contact_jz7: Any,
    *,
    rows: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    act = np.asarray(activation, dtype=np.float64)
    ticks = np.asarray(control_ticks, dtype=np.int64)
    mom = np.asarray(moment, dtype=np.float64)
    ct = np.asarray(contact_ticks, dtype=np.int64)
    jz = np.asarray(contact_jz7, dtype=np.float64)
    rows = np.arange(28, 35) if rows is None else np.asarray(rows, dtype=np.int64)
    groups = np.arange(65, 80) if groups is None else np.asarray(groups, dtype=np.int64)
    require(
        act.ndim == 2 and act.shape[1] >= 80 and np.isfinite(act).all(),
        "Activation schema differs",
    )
    require(
        ticks.shape == (len(act),)
        and np.array_equal(ticks, np.arange(ticks[0], ticks[0] + len(ticks))),
        "Activation control clock differs",
    )
    require(
        mom.ndim == 2
        and mom.shape[0] >= 35
        and mom.shape[1] >= 80
        and np.isfinite(mom).all(),
        "Moment schema differs",
    )
    require(jz.shape == (len(ct), 7) and np.isfinite(jz).all(), "RM Jz7 shape differs")
    lookup = ct - ticks[0]
    require(
        np.all((lookup > 0) & (lookup < len(ticks)))
        and np.array_equal(ticks[lookup], ct),
        "Contact/control tick alignment differs",
    )
    activation_derivative = (
        act[lookup][:, groups] - act[lookup - 1][:, groups]
    ) / CONTROL_DT
    joint_command_derivative = (
        0.02 * activation_derivative[:, None, :] * mom[np.ix_(rows, groups)][None, :, :]
    )
    projected = np.einsum("nj,njg->ng", jz, joint_command_derivative)
    return activation_derivative, joint_command_derivative, projected


def load_weighted_subset(
    mask: np.ndarray,
    load: np.ndarray,
    available: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    require(mask.shape == load.shape == available.shape, "Subset mask shape differs")
    valid = mask & available
    weights = np.where(valid, load, 0.0)
    denominator = weights.sum(axis=0)
    out: dict[str, np.ndarray] = {
        "load_sum_by_leg": denominator,
        "available_by_leg": denominator > 0,
        "native_leg_count": valid.sum(axis=0).astype(np.int64),
    }
    for name, raw in values.items():
        value = np.asarray(raw, dtype=np.float64)
        require(value.shape[:2] == load.shape, f"Subset value shape differs: {name}")
        tail = value.shape[2:]
        numerator = np.where(
            valid[(...,) + (None,) * len(tail)],
            value * weights[(...,) + (None,) * len(tail)],
            0.0,
        ).sum(axis=0)
        mean = np.full((6,) + tail, np.nan, dtype=np.float64)
        np.divide(
            numerator,
            denominator[(...,) + (None,) * len(tail)],
            out=mean,
            where=(denominator > 0)[(...,) + (None,) * len(tail)],
        )
        out[name] = mean
    return out


def aggregate_event_windows(
    window_index: np.ndarray,
    available: np.ndarray,
    load: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    radius_width = 2 * WINDOW_RADIUS + 1
    require(
        window_index.ndim == 2 and window_index.shape[1] == radius_width,
        "Event window index shape differs",
    )
    event_available = (
        available[window_index]
        if len(window_index)
        else np.empty((0, radius_width), bool)
    )
    event_load = (
        load[window_index] if len(window_index) else np.empty((0, radius_width), float)
    )
    weights = np.where(event_available, event_load, 0.0)
    denominator = weights.sum(axis=0)
    out: dict[str, np.ndarray] = {
        "event_window_available": event_available,
        "event_available_count": event_available.sum(axis=0).astype(np.int64),
        "event_load_sum": denominator,
        "event_aggregate_available": denominator > 0,
    }
    for name, raw in values.items():
        value = np.asarray(raw, dtype=np.float64)
        event = (
            value[window_index]
            if len(window_index)
            else np.empty((0, radius_width) + value.shape[1:])
        )
        tail = value.shape[1:]
        numerator = np.where(
            event_available[(...,) + (None,) * len(tail)],
            event * weights[(...,) + (None,) * len(tail)],
            0.0,
        ).sum(axis=0)
        mean = np.full((radius_width,) + tail, np.nan, dtype=np.float64)
        np.divide(
            numerator,
            denominator[(...,) + (None,) * len(tail)],
            out=mean,
            where=(denominator > 0)[(...,) + (None,) * len(tail)],
        )
        out["event_mean_" + name] = mean
    return out


def analyze_arrays(
    contact: Mapping[str, Any],
    native: Mapping[str, Any],
    layout: Mapping[str, Any],
    recruitment: Mapping[str, Any],
    adapter: Mapping[str, Any],
    features: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    ni = np.asarray(native["native_index"], dtype=np.int64)
    require(
        np.array_equal(ni, np.arange(10000)), "Native index is shifted or noncontiguous"
    )
    times = finite(native["time_s"], (10000,), "native time")
    require(
        np.allclose(times, 1.0001 + np.arange(10000) * NATIVE_DT, atol=2e-8, rtol=0),
        "Native clock differs",
    )
    ticks = np.asarray(native["control_tick"], dtype=np.int64)
    require(
        ticks.shape == (10000,) and np.array_equal(ticks, 201 + np.arange(10000) // 50),
        "Native/control tick mapping differs",
    )
    receipt_names = (
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
    for key in receipt_names:
        receipt = np.asarray(native[key])
        require(
            receipt.dtype == np.dtype(bool)
            and receipt.shape == (10000,)
            and receipt.all(),
            f"Native completeness receipt differs: {key}",
        )
    native_qvel = finite(native["qvel"], (10000, 108), "native qvel")
    native_ctrl = finite(native["ctrl"], (10000, 82), "native ctrl")
    slip = np.asarray(native["slip_mm_s"], dtype=np.float64)
    slip_valid = np.asarray(native["slip_valid"])
    require(
        slip.shape == (10000, 6)
        and slip_valid.dtype == np.dtype(bool)
        and slip_valid.shape == (10000, 6),
        "Native slip schema differs",
    )
    require(
        np.isfinite(slip[slip_valid]).all() and np.isnan(slip[~slip_valid]).all(),
        "Slip availability/value rule differs",
    )
    activation = finite(recruitment["activation"], (2001, 96), "activation")
    recruitment_ticks = np.asarray(recruitment["control_tick"], dtype=np.int64)
    require(
        np.array_equal(recruitment_ticks, np.arange(2001)),
        "Recruitment control clock differs",
    )
    complete = np.asarray(recruitment["complete"])
    activation_available = np.asarray(recruitment["activation_available"])
    require(
        complete.dtype == activation_available.dtype == np.dtype(bool)
        and complete.shape == activation_available.shape == (2001,)
        and complete.all()
        and activation_available.all(),
        "Recruitment completeness/availability differs",
    )
    feature_time = finite(features["time_s"], (2001,), "E152 feature time")
    require(
        np.allclose(feature_time, np.arange(2001) * CONTROL_DT, atol=2e-8, rtol=0),
        "E152 feature clock differs",
    )
    speed = finite(features["speed_mm_s"], (2001,), "interrupted speed")
    root = np.asarray(layout["root_dof_addresses"], dtype=np.int64)
    leg_dofs = np.asarray(layout["leg_dof_addresses"], dtype=np.int64)
    decomposition = decompose_world_z(
        contact["relative_jacobian"], contact["qvel"], contact["leg"], root, leg_dofs
    )
    cindex = np.asarray(contact["native_index"], dtype=np.int64)
    cleg = np.asarray(contact["leg"], dtype=np.int64)
    loads = np.asarray(contact["normal_load_bw"], dtype=np.float64)
    offsets = np.asarray(contact["offsets"], dtype=np.int64)
    require(
        len(cindex) == 24420
        and offsets.shape == (10001,)
        and offsets[0] == 0
        and offsets[-1] == 24420
        and np.all(np.diff(offsets) >= 0)
        and np.array_equal(cindex, np.repeat(np.arange(10000), np.diff(offsets))),
        "Positive-contact offset/index schema differs",
    )
    require(
        loads.shape == (24420,) and np.isfinite(loads).all() and np.all(loads > 0),
        "Positive contact load schema differs",
    )
    contact_qvel = finite(contact["qvel"], (24420, 108), "contact qvel")
    require(
        np.array_equal(contact_qvel, native_qvel[cindex]),
        "Contact qvel does not align with native source",
    )
    rm_mask = cleg == RM
    ap = np.asarray(contact["ap_own7"], dtype=np.float64)
    require(
        ap.shape == (len(cindex), 7) and np.isfinite(ap).all(), "AP own7 schema differs"
    )
    activation_derivative, joint_command_derivative, projected = (
        command_rate_projection(
            activation,
            recruitment_ticks,
            adapter["moment"],
            ticks[cindex[rm_mask]],
            np.asarray(contact["relative_jacobian"])[rm_mask, 2][:, leg_dofs[RM]],
        )
    )
    contact_values = {
        "native_z_total": decomposition["contact_z_total"],
        "native_z_root": decomposition["contact_z_root"],
        "native_z_own7": decomposition["contact_z_own7"],
        "native_z_residual": decomposition["contact_z_residual"],
        "native_z_closure_error": decomposition["contact_z_closure_error"],
        "native_ap_coxa_roll": ap[:, 1],
    }
    aggregates = weighted_native_leg(cindex, cleg, loads, contact_values, 10000)
    rm_projection = weighted_native_leg(
        cindex[rm_mask],
        cleg[rm_mask],
        loads[rm_mask],
        {
            "native_rm_activation_derivative": activation_derivative,
            "native_rm_joint_command_derivative": joint_command_derivative,
            "native_rm_command_projection_z": projected,
        },
        10000,
    )
    native_load = finite(native["normal_load_bw"], (10000, 6), "native load")
    require(np.all(native_load >= 0), "Native load must be nonnegative")
    require(
        slip_valid[native_load >= SUPPORT_BW].all(),
        "Every supported native sample requires valid slip",
    )
    interrupted = speed[ticks] <= 0.5
    force_forward = finite(native["force_forward_bw"], (10000, 6), "forward force")
    support = native_load >= SUPPORT_BW
    braking = support & (force_forward < 0)
    rm_positive_roll = (
        braking[:, RM]
        & aggregates["contact_available"][:, RM]
        & (aggregates["native_ap_coxa_roll"][:, RM] > 0)
    )
    raw, selected, reasons = find_unload_transitions(native_load[:, RM])
    # These requests exist on the full control clock, even without contact.
    # Only the geometric projection needs a recorded contact Jacobian.
    native_activation_derivative = (
        activation[ticks[:, None], np.arange(65, 80)[None, :]]
        - activation[ticks[:, None] - 1, np.arange(65, 80)[None, :]]
    ) / CONTROL_DT
    native_joint_command_derivative = (
        0.02
        * native_activation_derivative[:, None, :]
        * np.asarray(adapter["moment"])[np.ix_(np.arange(28, 35), np.arange(65, 80))][
            None
        ]
    )
    offsets = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1, dtype=np.int64)
    window_index = (
        selected[:, None] + offsets[None, :]
        if len(selected)
        else np.empty((0, len(offsets)), dtype=np.int64)
    )
    result = {
        **decomposition,
        **aggregates,
        "native_rm_activation_derivative": native_activation_derivative,
        "native_rm_joint_command_derivative": native_joint_command_derivative,
        "native_rm_command_projection_z": rm_projection[
            "native_rm_command_projection_z"
        ][:, RM],
        "support": support,
        "braking": braking,
        "interrupted": interrupted,
        "rm_braking_positive_coxa_roll": rm_positive_roll,
        "raw_transition_index": raw,
        "included_transition_index": selected,
        "transition_exclusion_reason": np.asarray(reasons, dtype="U40"),
        "window_offset": offsets,
        "window_native_index": window_index,
        "native_rm_target_ctrl": native_ctrl[
            :, np.asarray(layout["actuator_ids"])[28:35]
        ],
        "native_rm_actual_qvel": native_qvel[:, leg_dofs[RM]],
        "native_rm_activation": activation[
            ticks[:, None],
            np.arange(65, 80)[None, :],
        ],
        "native_load": native_load,
        "native_force_forward": force_forward,
        "native_slip_mm_s": slip,
        "native_slip_valid": slip_valid,
    }
    window_values = {}
    for name in (
        "native_z_total",
        "native_z_root",
        "native_z_own7",
        "native_z_residual",
        "native_ap_coxa_roll",
    ):
        value = aggregates[name][:, RM]
        result["window_" + name.removeprefix("native_")] = value[window_index]
        if name.startswith("native_z_"):
            window_values[name.removeprefix("native_")] = value
    for name in (
        "native_load",
        "native_force_forward",
        "native_slip_mm_s",
        "native_slip_valid",
        "native_rm_target_ctrl",
        "native_rm_actual_qvel",
        "native_rm_activation",
        "native_rm_activation_derivative",
        "native_rm_joint_command_derivative",
        "native_rm_command_projection_z",
    ):
        value = result[name]
        if name in (
            "native_load",
            "native_force_forward",
            "native_slip_mm_s",
            "native_slip_valid",
        ):
            value = value[:, RM]
        result["window_" + name.removeprefix("native_")] = value[window_index]
    window_values.update(
        {
            "rm_activation_derivative": result["native_rm_activation_derivative"],
            "rm_joint_command_derivative": result["native_rm_joint_command_derivative"],
            "rm_command_projection_z": result["native_rm_command_projection_z"],
        }
    )
    event_aggregate = aggregate_event_windows(
        window_index,
        aggregates["contact_available"][:, RM],
        native_load[:, RM],
        window_values,
    )
    result.update(event_aggregate)
    # Unconditional event means retain timing for quantities that remain
    # observable when the foot is unloaded. Z/projection means stay weighted.
    for name in (
        "load",
        "force_forward",
        "rm_target_ctrl",
        "rm_actual_qvel",
        "rm_activation",
        "rm_activation_derivative",
        "rm_joint_command_derivative",
    ):
        windows = result["window_" + name]
        result["event_all_mean_" + name] = (
            windows.mean(axis=0)
            if len(selected)
            else np.full(windows.shape[1:], np.nan)
        )
    # Fixed subset summaries use native leg load, equivalent to pooled-contact weighting.
    subset_masks = {
        "supported": support,
        "supported_braking": braking,
        "rm_supported_braking_positive_coxa_roll": np.column_stack(
            [
                np.zeros((len(ni), 4), dtype=bool),
                rm_positive_roll,
                np.zeros((len(ni), 1), dtype=bool),
            ]
        ),
        "interrupted_supported": support & interrupted[:, None],
        "interrupted_supported_braking": braking & interrupted[:, None],
        "interrupted_rm_supported_braking_positive_coxa_roll": np.column_stack(
            [
                np.zeros((len(ni), 4), dtype=bool),
                rm_positive_roll & interrupted,
                np.zeros((len(ni), 1), dtype=bool),
            ]
        ),
    }
    subset_values = {
        "z_total": aggregates["native_z_total"],
        "z_root": aggregates["native_z_root"],
        "z_own7": aggregates["native_z_own7"],
        "z_residual": aggregates["native_z_residual"],
    }
    for subset, mask in subset_masks.items():
        result["subset_mask_" + subset] = mask
        summary = load_weighted_subset(
            mask, native_load, aggregates["contact_available"], subset_values
        )
        for key, value in summary.items():
            result[f"subset_{subset}_{key}"] = value
        rm_valid = mask[:, RM] & aggregates["contact_available"][:, RM]
        rm_weights = np.where(rm_valid, native_load[:, RM], 0.0)
        denom = rm_weights.sum()
        for name in (
            "native_rm_activation_derivative",
            "native_rm_joint_command_derivative",
            "native_rm_command_projection_z",
        ):
            value = result[name]
            tail = value.shape[1:]
            mean = np.full(tail, np.nan)
            if denom > 0:
                mean = (
                    np.where(
                        rm_valid[(...,) + (None,) * len(tail)],
                        value * rm_weights[(...,) + (None,) * len(tail)],
                        0.0,
                    ).sum(axis=0)
                    / denom
                )
            result[f"subset_{subset}_{name.removeprefix('native_')}"] = mean
    return result


def _source_path(freeze: Mapping[str, Any], key: str) -> Path:
    paths = freeze.get("paths", freeze.get("source_paths", {}))
    require(type(paths) is dict and key in paths, f"Missing frozen source path: {key}")
    return safe_path(paths[key])


def trace_arrays(
    trace: Mapping[str, Any], adapter: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    samples = trace.get("samples")
    require(
        type(samples) is list and len(samples) == 2001, "Trace sample count differs"
    )
    ticks = np.asarray([row.get("control_tick") for row in samples], dtype=np.int64)
    require(np.array_equal(ticks, np.arange(2001)), "Trace control clock differs")
    times = np.asarray([row.get("time_s") for row in samples], dtype=np.float64)
    require(
        times.shape == (2001,)
        and np.isfinite(times).all()
        and np.allclose(times, ticks * CONTROL_DT, atol=2e-8, rtol=0),
        "Trace source clock differs",
    )
    targets = np.asarray(
        [row.get("joint_targets_rad") for row in samples], dtype=np.float64
    )
    require(
        targets.shape == (2001, 42) and np.isfinite(targets).all(),
        "Trace joint target schema differs",
    )
    tendon = np.asarray(
        [row.get("tendon_inputs", {}).get("rm_tarsus") for row in samples],
        dtype=np.float64,
    )
    require(
        tendon.shape == (2001,) and np.isfinite(tendon).all(),
        "RM LTM tendon schema differs",
    )
    ltm = np.asarray(adapter["ltm"], dtype=np.int64)
    require(ltm.shape == (6,) and int(ltm[RM]) == 78, "RM LTM group mapping differs")
    return {
        "trace_control_tick": ticks,
        "trace_time_s": times,
        "rm_trace_joint_targets": targets[:, 28:35],
        "rm_ltm_tendon": tendon,
    }


def validate_numerical_contract(
    arrays: Mapping[str, np.ndarray], recorded_targets: np.ndarray
) -> None:
    require(
        np.isfinite(arrays["contact_z_closure_error"]).all()
        and np.max(np.abs(arrays["contact_z_closure_error"])) <= CLOSURE_ATOL,
        "World-Z decomposition closure differs",
    )
    require(
        np.array_equal(
            np.isfinite(arrays["native_z_total"]), arrays["contact_available"]
        ),
        "Native Z availability/NaN rule differs",
    )
    require(
        np.max(np.abs(arrays["contact_load_sum"] - arrays["native_load"])) <= LOAD_ATOL,
        "Positive contact load does not match native load",
    )
    require(
        np.array_equal(arrays["native_rm_target_ctrl"], recorded_targets),
        "Native ctrl and recorded RM targets differ",
    )


def _json_numbers(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return None if not np.isfinite(array) else float(array)
    return [_json_numbers(item) for item in array]


def finalize_publication(
    output: Path, result: dict[str, Any], ledger: dict[str, Any], started: float
) -> dict[str, Any]:
    """Publish final artifacts while revoking success after any late failure."""
    publication_errors: list[dict[str, str]] = list(
        result.get("publication_errors", [])
    )

    def record(stage: str, exc: BaseException) -> None:
        publication_errors.append(
            {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}
        )
        result.update(
            {
                "accepted": False,
                "publication_complete": False,
                "diagnostic_valid": False,
            }
        )

    result["wall_s"] = time.perf_counter() - started
    if result["wall_s"] > INTERNAL_S:
        record("entire_run_deadline", TimeoutError("E161 internal deadline exceeded"))
    result["publication_errors"] = publication_errors
    ledger["publication_errors"] = publication_errors
    try:
        atomic_json(output / "ledger.json", ledger)
    except Exception as exc:  # noqa: BLE001
        record("ledger", exc)
    result["publication_errors"] = publication_errors
    try:
        atomic_json(output / "result.json", result)
    except Exception as exc:  # noqa: BLE001
        record("result", exc)
    result["publication_errors"] = publication_errors
    if publication_errors:
        try:
            atomic_json(output / "result.json", result)
        except Exception as exc:  # noqa: BLE001
            if not any(item["stage"] == "result" for item in publication_errors):
                record("result", exc)
    try:
        manifest = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "output-manifest.json"
        }
        atomic_json(output / "output-manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001
        record("manifest", exc)
        result["publication_errors"] = publication_errors
        try:
            atomic_json(output / "result.json", result)
        except Exception as result_exc:  # noqa: BLE001
            record("result_after_manifest_failure", result_exc)
    total_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    result["total_output_bytes"] = total_bytes
    if total_bytes > RAW_LIMIT:
        record("total_output_limit", ValueError("Total E161 output exceeds 128 MiB"))
    elapsed = time.perf_counter() - started
    result["wall_s"] = elapsed
    if elapsed > INTERNAL_S and not any(
        item["stage"] == "entire_run_deadline" for item in publication_errors
    ):
        record("entire_run_deadline", TimeoutError("E161 internal deadline exceeded"))
    result["publication_errors"] = publication_errors
    ledger["publication_errors"] = publication_errors
    try:
        atomic_json(output / "ledger.json", ledger)
        atomic_json(output / "result.json", result)
        manifest = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "output-manifest.json"
        }
        atomic_json(output / "output-manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001
        if not any(
            item["stage"] == "final_failure_publication" for item in publication_errors
        ):
            record("final_failure_publication", exc)
    # Revoke persisted success after any final-pass failure, not only the
    # in-memory return value. Include final hashes/writes in resource checks.
    elapsed = time.perf_counter() - started
    result["wall_s"] = elapsed
    if elapsed > INTERNAL_S and not any(
        item["stage"] == "entire_run_deadline" for item in publication_errors
    ):
        record("entire_run_deadline", TimeoutError("E161 internal deadline exceeded"))
    total_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if total_bytes > RAW_LIMIT and not any(
        item["stage"] == "total_output_limit" for item in publication_errors
    ):
        record("total_output_limit", ValueError("Total E161 output exceeds 128 MiB"))
    if publication_errors:
        for filename, payload in (("ledger.json", ledger), ("result.json", result)):
            try:
                atomic_json(output / filename, payload)
            except Exception as exc:  # noqa: BLE001 - preserve other artifacts.
                record("failure_recovery_" + filename, exc)
                if filename == "result.json" and (output / filename).exists():
                    # Preserve a stale receipt as explicitly invalid if its
                    # replacement cannot be written. No scientific work repeats.
                    try:
                        (output / filename).replace(
                            output / ("invalid-result-" + uuid4().hex + ".json")
                        )
                    except Exception as move_exc:  # noqa: BLE001
                        record("invalidate_stale_result", move_exc)
        try:
            manifest = {
                p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "output-manifest.json"
            }
            atomic_json(output / "output-manifest.json", manifest)
        except Exception as exc:  # noqa: BLE001
            record("failure_recovery_manifest", exc)
    return result


def run(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    require(
        not output.exists(), "E161 output directory already exists; no retry/overwrite"
    )
    require(shutil.disk_usage(output.parent).free >= FREE_LIMIT, "Less than 1 GiB free")
    output.mkdir(parents=False)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "accepted": False,
        "publication_complete": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "interpretation_limit": "Saved-array diagnostic only; no causal, muscle-force, biological-mechanism, natural-gait, M2, or F6 validation.",
    }
    source_before: dict[str, str] = {}
    owned_before: dict[str, str] = {}
    freeze: dict[str, Any] | None = None
    analysis_ready = False
    primary_error: dict[str, str] | None = None
    try:
        freeze, review = verified_inputs(
            freeze_path, freeze_sha, review_path, review_sha
        )
        atomic_json(output / "freeze-snapshot.json", freeze)
        atomic_json(output / "source-review-snapshot.json", review)
        owned_before = {name: sha256(ROOT / name) for name in OWNED}
        sources = {
            key: _source_path(freeze, key)
            for key in (
                "contacts",
                "native",
                "layout",
                "recruitment",
                "provenance",
                "trace",
                "features",
            )
        }
        source_before = {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sources.values()
        }
        with np.load(sources["contacts"], allow_pickle=False) as z:
            contact = {key: z[key] for key in z.files}
        with np.load(sources["native"], allow_pickle=False) as z:
            native = {key: z[key] for key in z.files}
        layout = json.loads(sources["layout"].read_text(encoding="utf-8"))
        contract = freeze["sample_contract"]
        require(
            layout.get("root_dof_addresses") == contract["root_dof_addresses"]
            and layout.get("leg_dof_addresses") == contract["leg_dof_addresses"]
            and layout.get("actuator_ids") == contract["actuator_ids"]
            and layout.get("nq") == contract["nq"]
            and layout.get("nv") == contract["nv"]
            and layout.get("nu") == contract["nu"]
            and layout.get("body_hash") == freeze["body_hash"],
            "Frozen layout mapping differs",
        )
        with np.load(sources["recruitment"], allow_pickle=False) as z:
            recruitment = {key: z[key] for key in z.files}
        with np.load(sources["provenance"], allow_pickle=False) as z:
            adapter = {key: z[key] for key in z.files}
        with np.load(sources["features"], allow_pickle=False) as z:
            features = {key: z[key] for key in z.files}
        arrays = analyze_arrays(contact, native, layout, recruitment, adapter, features)
        trace = json.loads(sources["trace"].read_text(encoding="utf-8"))
        trace_output = trace_arrays(trace, adapter)
        arrays.update(trace_output)
        native_ticks = np.asarray(native["control_tick"], dtype=np.int64)
        window_index = arrays["window_native_index"]
        arrays["rm_ltm_activation"] = np.asarray(recruitment["activation"])[
            native_ticks, 78
        ]
        arrays["rm_ltm_tendon_native"] = trace_output["rm_ltm_tendon"][native_ticks]
        arrays["window_rm_ltm_activation"] = arrays["rm_ltm_activation"][window_index]
        arrays["window_rm_ltm_tendon"] = arrays["rm_ltm_tendon_native"][window_index]
        for name in ("rm_ltm_activation", "rm_ltm_tendon"):
            values = arrays["window_" + name]
            arrays["event_all_mean_" + name] = (
                values.mean(axis=0)
                if len(window_index)
                else np.full(values.shape[1:], np.nan)
            )
        arrays["window_rm_trace_joint_targets"] = trace_output[
            "rm_trace_joint_targets"
        ][native_ticks][window_index]
        # Raw derived arrays precede all numerical assertion summaries/publication.
        atomic_npz(output / "raw-derived.npz", **arrays)
        require(
            (output / "raw-derived.npz").stat().st_size <= RAW_LIMIT,
            "Raw output exceeds 128 MiB",
        )
        validate_numerical_contract(
            arrays, trace_output["rm_trace_joint_targets"][native_ticks]
        )
        subset_summary = {}
        for subset in (
            "supported",
            "supported_braking",
            "rm_supported_braking_positive_coxa_roll",
            "interrupted_supported",
            "interrupted_supported_braking",
            "interrupted_rm_supported_braking_positive_coxa_roll",
        ):
            subset_summary[subset] = {
                name: _json_numbers(arrays[f"subset_{subset}_{name}"])
                for name in (
                    "z_total",
                    "z_root",
                    "z_own7",
                    "z_residual",
                    "rm_activation_derivative",
                    "rm_joint_command_derivative",
                    "rm_command_projection_z",
                )
            }
            subset_summary[subset]["load_sum_by_leg"] = _json_numbers(
                arrays[f"subset_{subset}_load_sum_by_leg"]
            )
            subset_summary[subset]["available_by_leg"] = arrays[
                f"subset_{subset}_available_by_leg"
            ].tolist()
        summary = {
            "raw_transitions": len(arrays["raw_transition_index"]),
            "included_transitions": len(arrays["included_transition_index"]),
            "max_abs_closure_error": _json_numbers(
                np.max(np.abs(arrays["contact_z_closure_error"]))
            ),
            "max_abs_load_error": _json_numbers(
                np.max(np.abs(arrays["contact_load_sum"] - arrays["native_load"]))
            ),
            "contact_available_count": int(arrays["contact_available"].sum()),
            "contact_unavailable_count": int((~arrays["contact_available"]).sum()),
            "subset_load_weighted": subset_summary,
            "event_aggregate": {
                "event_count": len(arrays["included_transition_index"]),
                "available_count": arrays["event_available_count"].tolist(),
                "available": arrays["event_aggregate_available"].tolist(),
                "z_total": _json_numbers(arrays["event_mean_z_total"]),
                "z_root": _json_numbers(arrays["event_mean_z_root"]),
                "z_own7": _json_numbers(arrays["event_mean_z_own7"]),
                "z_residual": _json_numbers(arrays["event_mean_z_residual"]),
                "rm_command_projection_z": _json_numbers(
                    arrays["event_mean_rm_command_projection_z"]
                ),
            },
            "ltm_group": 78,
            "ltm_limit": "Position moment may be zero; this analysis makes no zero physical influence claim.",
        }
        atomic_json(output / "summary.json", summary)
        result.update(
            {
                "source_hashes_pre": source_before,
                "freeze_sha256": freeze_sha,
                "source_review_sha256": review_sha,
                **summary,
            }
        )
        analysis_ready = True
    except Exception as exc:  # noqa: BLE001 - fail-closed artifacts are required.
        primary_error = {"error_type": type(exc).__name__, "error": str(exc)}
        result.update(
            {
                "diagnostic_valid": False,
                "primary_error": primary_error,
            }
        )
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        secondary: list[dict[str, str]] = []
        if freeze is not None:
            for name in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
                try:
                    _verify_map(freeze[name], "post-" + name)
                except Exception as exc:  # noqa: BLE001
                    secondary.append(
                        {
                            "stage": "post_" + name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        try:
            source_after = {name: sha256(ROOT / name) for name in source_before}
            owned_after = {name: sha256(ROOT / name) for name in OWNED}
            require(source_after == source_before, "Frozen source changed during E161")
            require(
                not owned_before or owned_after == owned_before,
                "Owned E161 source changed during run",
            )
            result["source_hashes_post"] = source_after
            result["owned_source_hashes_post"] = owned_after
        except Exception as exc:  # noqa: BLE001
            secondary.append(
                {
                    "stage": "post_source_hashes",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        pools = threadpool_info()
        result["threadpools"] = pools
        if any(int(pool.get("num_threads", 0)) != 1 for pool in pools):
            secondary.append(
                {
                    "stage": "thread_limit",
                    "error_type": "ValueError",
                    "error": "Numerical threadpool count differs from one",
                }
            )
        result["secondary_errors"] = secondary
        success = analysis_ready and primary_error is None and not secondary
        result.update(
            {
                "accepted": success,
                "publication_complete": success,
                "diagnostic_valid": success,
            }
        )
        ledger = {
            "schema": "flylab.m3.e161.ledger.v1",
            "original_processes": 1,
            "automatic_retries": 0,
            "body_models": 0,
            "physics_steps": 0,
            "neural_steps": 0,
            "fit_calls": 0,
            "candidate_adopted": False,
            "whole_goal_complete": False,
            "F6_exposed": False,
            "primary_error": primary_error,
            "secondary_errors": secondary,
        }
        result = finalize_publication(output, result, ledger, started)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.out,
    )
    print(json.dumps(result, allow_nan=False))
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
