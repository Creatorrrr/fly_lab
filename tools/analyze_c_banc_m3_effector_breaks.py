"""E226 saved-only, descriptive effector forward-bout interruption analysis."""

# Preserve the primary data/analysis failure across partial output receipts.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.storage import StateStore
from tools import analyze_c_banc_biological_gait as gait
from tools import verify_c_banc_m3_neural_current_body as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e226-effector-breaks"
FREEZE_SHA = "e2b5009615d7de8d920e0e7126e2f9d15e254d46e0115061ea2668e252cf5668"
FREEZE_SCHEMA = "flylab.m3.e226.effector-breaks.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e226.effector-breaks.review.v1"
OWNED = ("tools/analyze_c_banc_m3_effector_breaks.py",
         "tests/test_c_banc_effector_breaks.py")
LEGS = gait.LEGS
CONTROL_KEYS = prior.COMMON_FIELDS + prior.EFFECTOR_FIELDS
NATIVE_KEYS = ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid",
               "nonfoot_contact", "upright")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository-relative E226 input required")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """No saved analysis before exact input/source/resource approval."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E226 freeze and parent review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    analysis = freeze.get("analysis", {})
    resources = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or analysis != {"control_rows": 1664, "native_samples": 83150,
                "control_dt_s": .005, "native_dt_s": .0001,
                "window_start_s": .5, "window_end_s": 8.315,
                "main_entry_min_s": 3., "moving_speed_threshold_mm_s": .5,
                "moving_sigma_s": .02, "preceding_bout_min_s": .3,
                "pre_window_s": [-.08, -.02], "post_window_s": [.02, .08],
                "aligned_half_window_s": .1, "support_threshold_bw": .02}
            or resources != {"cpu_threads": 1, "processes": 1, "retries": 0,
                "internal_s": 20., "external_s": 25., "supervisor_s": 30.,
                "resident_bytes": 6442450944, "output_bytes": 134217728,
                "original_dynamics_calls": 0, "cuda_calls": 0,
                "physics_steps": 0, "solver_calls": 0, "fit_calls": 0}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete"))):
        raise ValueError("E226 fixed analysis/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Exact approved E226 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E226 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E226 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("E226 reviewed source hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E226 source changed: " + name)
    if output.exists() or not output.parent.is_dir() or output != OUT.resolve():
        raise FileExistsError("Exact new E226 output directory required")
    if shutil.disk_usage(output.parent).free < resources["output_bytes"]:
        raise OSError("E226 output disk reserve unavailable")
    return freeze, review


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_joined_inputs(freeze):
    """Join only the pinned saved prefix/suffix; reject a changed seam."""
    paths = {key: safe(name) for key, name in freeze["paths"].items()}
    prefix_control = load_npz(paths["prefix_control"])
    suffix_control = load_npz(paths["suffix_control"])
    prefix_native = load_npz(paths["prefix_native"])
    suffix_native = load_npz(paths["suffix_native"])
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    rows = trace["samples"]
    control_shapes = {"control_tick": (), "network_tick": (),
        "physics_time_s": (), "motor_rate32": (391,),
        "decoder_input32": (391,), "activation": (96,),
        "command": (42,), "tendon": (6,), "free_effector64": (343,),
        "h_fast": (343, 3), "h_slow": (343, 3)}
    native_shapes = {"time_s": (), "normal_load_bw": (6,),
        "slip_mm_s": (6,), "slip_valid": (6,),
        "nonfoot_contact": (), "upright": ()}
    if (set(prefix_control) != set(CONTROL_KEYS)
            or set(suffix_control) != set(CONTROL_KEYS)
            or set(prefix_native) != set(NATIVE_KEYS)
            or set(suffix_native) != set(NATIVE_KEYS)
            or len(rows) != 1664
            or any(prefix_control[key].shape != (601, *control_shapes[key])
                   or suffix_control[key].shape != (1064, *control_shapes[key])
                   or prefix_control[key].dtype != suffix_control[key].dtype
                   for key in CONTROL_KEYS)
            or any(prefix_native[key].shape != (30000, *native_shapes[key])
                   or suffix_native[key].shape != (53150, *native_shapes[key])
                   or prefix_native[key].dtype != suffix_native[key].dtype
                   for key in NATIVE_KEYS)
            or any(prefix_control[key].dtype != np.int64 for key in
                   ("control_tick", "network_tick"))
            or any(prefix_control[key].dtype != np.float32 for key in
                   ("motor_rate32", "decoder_input32"))
            or any(prefix_control[key].dtype != np.float64 for key in CONTROL_KEYS
                   if key not in ("control_tick", "network_tick", "motor_rate32",
                                  "decoder_input32"))
            or any(prefix_native[key].dtype != np.bool_ for key in
                   ("slip_valid", "nonfoot_contact"))
            or any(prefix_native[key].dtype != np.float64 for key in NATIVE_KEYS
                   if key not in ("slip_valid", "nonfoot_contact"))):
        raise ValueError("E226 exact E224/E225 raw shapes and dtypes required")
    for key in CONTROL_KEYS:
        if (prefix_control[key][-1].tobytes()
                != suffix_control[key][0].tobytes()):
            raise ValueError("E226 saved control600 seam changed: " + key)
    control = {key: np.concatenate((prefix_control[key], suffix_control[key][1:]))
               for key in CONTROL_KEYS}
    native = {key: np.concatenate((prefix_native[key], suffix_native[key]))
              for key in NATIVE_KEYS}
    t = np.arange(1664) * .005
    tn = np.arange(1, 83151) * .0001
    if (not np.array_equal(control["control_tick"], np.arange(1664))
            or not np.array_equal(control["network_tick"], np.arange(1664) * 20)
            or not np.allclose(control["physics_time_s"], t, atol=2e-8, rtol=0)
            or not np.allclose(native["time_s"], tn, atol=2e-8, rtol=0)
            or not np.isfinite(control["physics_time_s"]).all()
            or not np.isfinite(native["time_s"]).all()
            or any(not np.isfinite(control[key]).all() for key in CONTROL_KEYS)
            or any(row["control_tick"] != i
                   or not np.isclose(row["time_s"], t[i], atol=2e-8, rtol=0)
                   or row["feet"]["support_threshold_bw"] != freeze[
                       "analysis"]["support_threshold_bw"]
                   or not np.array_equal(np.asarray(row["joint_targets_rad"]),
                                         control["command"][i])
                   or not np.array_equal(np.asarray(row["motor_rates"], np.float32),
                                         control["motor_rate32"][i])
                   or not np.array_equal(np.asarray([row["tendon_inputs"][
                       leg + "_tarsus"] for leg in LEGS]), control["tendon"][i])
                   for i, row in enumerate(rows))):
        raise ValueError("E226 joined clocks or trace/control values changed")
    return paths, control, native, trace


def aggregate_native_intervals(control_time, native, support_threshold=.02):
    """Map exactly 50 native samples to (previous,current] for each control."""
    t = np.asarray(control_time, np.float64)
    count = len(t)
    if (t.ndim != 1 or count < 2 or not np.isfinite(t).all()
            or not np.allclose(np.diff(t), .005, atol=2e-8, rtol=0)
            or set(native) != set(NATIVE_KEYS)
            or any(native[key].shape != ((count - 1) * 50, *shape)
                   for key, shape in (("time_s", ()), ("normal_load_bw", (6,)),
                                      ("slip_mm_s", (6,)), ("slip_valid", (6,)),
                                      ("nonfoot_contact", ()), ("upright", ())))):
        raise ValueError("Exactly 50 native 100us samples per completed interval required")
    expected = t[:-1, None] + np.arange(1, 51)[None, :] * .0001
    actual = native["time_s"].reshape(count - 1, 50)
    if (not np.isfinite(actual).all()
            or not np.allclose(actual, expected, atol=2e-8, rtol=0)):
        raise ValueError("Native/control (previous,current] boundary changed")
    loads = native["normal_load_bw"].reshape(count - 1, 50, 6)
    slip = native["slip_mm_s"].reshape(count - 1, 50, 6)
    valid = native["slip_valid"].reshape(count - 1, 50, 6)
    if (not np.isfinite(loads).all() or np.any(loads < 0)
            or not np.isfinite(slip[valid]).all()):
        raise ValueError("Finite measured load and valid-slip values required")
    support = loads >= support_threshold
    selected = support & valid
    weighted = loads * selected
    denominator = weighted.sum(axis=1)
    per_leg_slip = np.full((count - 1, 6), np.nan)
    np.divide(np.where(selected, loads * slip, 0.).sum(axis=1), denominator,
              out=per_leg_slip, where=denominator > 0)
    global_denom = denominator.sum(axis=1)
    global_slip = np.full(count - 1, np.nan)
    np.divide(np.where(selected, loads * slip, 0.).sum(axis=(1, 2)),
              global_denom, out=global_slip, where=global_denom > 0)

    def prepend(data, *, absent=np.nan):
        return np.concatenate((np.full((1, *data.shape[1:]), absent,
                                       dtype=data.dtype), data), axis=0)

    return {"available": np.r_[False, np.ones(count - 1, bool)],
        "mean_normal_load_bw": prepend(loads.mean(axis=1)),
        "max_normal_load_bw": prepend(loads.max(axis=1)),
        "support_fraction": prepend(support.mean(axis=1)),
        "support_leg_count_mean": prepend(support.sum(axis=2).mean(axis=1)),
        "valid_support_slip_fraction": prepend(selected.mean(axis=1)),
        "weighted_slip_mm_s": prepend(per_leg_slip),
        "weighted_slip_available": np.vstack((np.zeros((1, 6), bool),
                                                denominator > 0)),
        "global_weighted_slip_mm_s": prepend(global_slip),
        "global_weighted_slip_available": np.r_[False, global_denom > 0],
        "nonfoot_any": np.r_[False, native["nonfoot_contact"].reshape(
            count - 1, 50).any(axis=1)],
        "minimum_upright": prepend(native["upright"].reshape(
            count - 1, 50).min(axis=1))}


def derive_kinematics(rows, model):
    """Separate true horizontal translation from rotating-body projection."""
    n = len(rows)
    time_s = np.asarray([row["time_s"] for row in rows], np.float64)
    dt = gait.validate_clock(time_s)
    position = np.asarray([row["body"]["position"] for row in rows], np.float64)
    basis = np.asarray([row["body"]["basis"] for row in rows], np.float64)
    foot = np.asarray([[row["body"]["legs"][leg][-1] for leg in LEGS]
                       for row in rows], np.float64)
    actual = np.asarray([row["joint_angles_rad"] for row in rows], np.float64)
    target = np.asarray([row["joint_targets_rad"] for row in rows], np.float64)
    tracking = np.asarray([row["joint_tracking_error_rad"] for row in rows],
                          np.float64)
    qvel = np.asarray([row["qvel"] for row in rows], np.float64)
    ctrl = np.asarray([row["ctrl"] for row in rows], np.float64)
    if (n < 4 or position.shape != (n, 3) or basis.shape != (n, 3, 3)
            or foot.shape != (n, 6, 3) or actual.shape != (n, 42)
            or target.shape != (n, 42) or tracking.shape != (n, 42)
            or qvel.ndim != 2 or ctrl.ndim != 2
            or any(not np.isfinite(value).all() for value in (
                position, basis, foot, actual, target, tracking, qvel, ctrl))
            or not np.allclose(tracking, target - actual, atol=2e-8, rtol=0)):
        raise ValueError("Complete finite actual pose, foot, and 42-joint trace required")
    velocity = np.gradient(position, dt, axis=0)
    foot_velocity = np.gradient(foot, dt, axis=0)
    forward = basis[:, :, 0]
    initial = forward[0, [0, 2]]
    if np.linalg.norm(initial) <= 1e-12:
        raise ValueError("Measured initial horizontal forward axis required")
    initial /= np.linalg.norm(initial)
    relative = foot - position[:, None, :]
    ego_fore = np.einsum("tlj,tj->tl", relative, forward)
    if (not np.array_equal(model["t"], time_s)
            or len(model["speed"]) != n or len(model["yaw_rate"]) != n):
        raise ValueError("Frozen gait model and trace clock differ")
    return {"time_s": time_s,
        "original_smoothed_forward_speed_mm_s": np.asarray(model["speed"]),
        "world_velocity_xyz_mm_s": velocity,
        "forward_basis_xyz": forward,
        "initial_horizontal_forward_xz": initial,
        "instant_body_forward_velocity_mm_s": np.einsum(
            "tj,tj->t", velocity, forward),
        "initial_forward_velocity_mm_s": velocity[:, [0, 2]] @ initial,
        "world_horizontal_speed_mm_s": np.linalg.norm(
            velocity[:, [0, 2]], axis=1),
        "yaw_rate_rad_s": np.asarray(model["yaw_rate"]),
        "ego_foot_fore_mm": ego_fore,
        "ego_foot_fore_velocity_mm_s": np.gradient(ego_fore, dt, axis=0),
        "foot_world_horizontal_speed_mm_s": np.linalg.norm(
            foot_velocity[:, :, [0, 2]], axis=2),
        "foot_world_forward_velocity_mm_s": np.einsum(
            "tlj,tj->tl", foot_velocity, forward),
        "joint_angles_rad": actual, "joint_targets_rad": target,
        "joint_tracking_error_rad": tracking,
        "joint_velocity_rad_s": np.gradient(actual, dt, axis=0),
        "target_velocity_rad_s": np.gradient(target, dt, axis=0),
        "qvel": qvel, "ctrl": ctrl}


def command_contributions(activation, command, moment, rest):
    """Decompose the canonical command on active rows; row0 is held neutral."""
    active = np.asarray(activation)
    actual = np.asarray(command)
    moment = np.asarray(moment)
    rest = np.asarray(rest)
    if (active.ndim != 2 or active.shape[1] != 96
            or actual.shape != (len(active), 42)
            or moment.shape != (42, 96) or rest.shape != (42,)
            or any(not np.isfinite(value).all() for value in (
                active, actual, moment, rest))):
        raise ValueError("Finite 96-muscle canonical command inputs required")
    contribution = .02 * active[:, None, :] * moment[None, :, :]
    canonical = rest[None, :] + .02 * active @ moment.T
    if (not np.allclose(contribution.sum(axis=2),
                        canonical - rest[None, :], atol=2e-12, rtol=0)
            or not np.allclose(actual[1:], canonical[1:], atol=2e-12, rtol=0)):
        raise ValueError("E226 canonical muscle contribution reconstruction changed")
    return {"contribution_rad": contribution,
            "canonical_command_rad": canonical,
            "command_reconstruction_error_rad": actual - canonical,
            "canonical_valid": np.r_[False, np.ones(len(active) - 1, bool)],
            "row0_held_command_rad": actual[0].copy()}


def detect_breaks(time_s, moving, analysis):
    """All below-threshold gaps bracketed by observed moving bouts."""
    t = np.asarray(time_s)
    moving = np.asarray(moving, bool)
    if t.ndim != 1 or moving.shape != t.shape:
        raise ValueError("A complete frozen movement mask is required")
    dt = gait.validate_clock(t)
    if abs(dt - analysis["control_dt_s"]) > 1e-12:
        raise ValueError("E226 original 5ms movement clock required")
    window = (t >= analysis["window_start_s"] - 1e-12) & (
        t < analysis["window_end_s"] - 1e-12)
    runs = gait.runs(moving & window)
    result = []
    for (before_start, before_end), (after_start, after_end) in itertools.pairwise(runs):
        del after_end
        if after_start <= before_end:
            continue
        entry, exit_ = before_end, after_start
        before_samples = before_end - before_start
        result.append({"event_index": len(result),
            "entry_index": int(entry), "exit_index": int(exit_),
            "entry_s": float(t[entry]), "exit_s": float(t[exit_]),
            "duration_samples": int(exit_ - entry),
            "duration_s": float((exit_ - entry) * dt),
            "preceding_bout_samples": int(before_samples),
            "preceding_bout_s": float(before_samples * dt),
            "main_comparison": bool(t[entry] >= analysis["main_entry_min_s"]
                and before_samples * dt + 1e-12 >= analysis["preceding_bout_min_s"])})
    return result


def aligned_windows(event, count, analysis):
    """Exact integer-index windows; absent right tails remain unavailable."""
    dt = analysis["control_dt_s"]
    entry, exit_ = event["entry_index"], event["exit_index"]
    def offset(value):
        ticks = value / dt
        if abs(ticks - round(ticks)) > 1e-9:
            raise ValueError("Preregistered E226 window must align to 5ms")
        return round(ticks)
    pre = (entry + offset(analysis["pre_window_s"][0]),
           entry + offset(analysis["pre_window_s"][1]))
    post = (exit_ + offset(analysis["post_window_s"][0]),
            exit_ + offset(analysis["post_window_s"][1]))
    during = (entry, exit_)
    center = (entry + exit_) // 2
    half = offset(analysis["aligned_half_window_s"])
    aligned = (center - half, center + half + 1)
    return {key: {"indices": list(range(a, b)) if 0 <= a <= b <= count else None,
                  "available": bool(0 <= a < b <= count),
                  "start_index": a, "stop_index_exclusive": b}
            for key, (a, b) in (("pre", pre), ("during", during),
                                ("post", post), ("aligned", aligned))}


def analyze_events(events, motion, native_interval, activation,
                   contribution, groups, analysis):
    """Descriptive pre/break/post summaries and noncausal muscle rankings."""
    count = len(motion["time_s"])
    features = {"forward_speed_mm_s": motion[
        "original_smoothed_forward_speed_mm_s"],
        "actual_horizontal_speed_mm_s": motion["world_horizontal_speed_mm_s"],
        "initial_forward_velocity_mm_s": motion["initial_forward_velocity_mm_s"],
        "yaw_rate_rad_s": motion["yaw_rate_rad_s"],
        "ego_foot_fore_velocity_mm_s": motion["ego_foot_fore_velocity_mm_s"],
        "foot_world_horizontal_speed_mm_s": motion[
            "foot_world_horizontal_speed_mm_s"],
        "mean_normal_load_bw": native_interval["mean_normal_load_bw"],
        "support_fraction": native_interval["support_fraction"],
        "support_leg_count_mean": native_interval["support_leg_count_mean"],
        "weighted_slip_mm_s": native_interval["weighted_slip_mm_s"],
        "joint_tracking_error_rad": motion["joint_tracking_error_rad"],
        "joint_angles_rad": motion["joint_angles_rad"],
        "joint_targets_rad": motion["joint_targets_rad"],
        "joint_velocity_rad_s": motion["joint_velocity_rad_s"],
        "target_velocity_rad_s": motion["target_velocity_rad_s"],
        "muscle_activation": activation}

    def average(value, indices):
        if indices is None:
            return None
        block = np.asarray(value)[indices]
        if block.size == 0:
            raise ValueError("A registered nonempty E226 window is required")
        finite = np.isfinite(block)
        counts = finite.sum(axis=0)
        means = np.divide(np.where(finite, block, 0.).sum(axis=0), counts,
                          out=np.full(counts.shape, np.nan), where=counts > 0)
        return {"mean": prior.finite_receipt(means.tolist())[0],
                "valid_sample_count": counts.tolist(),
                "window_sample_count": len(indices),
                "method": "finite interval values only; missing values not imputed"}

    all_events = []
    aligned_indices = np.full((len(events), 41), -1, np.int64)
    aligned_valid = np.zeros((len(events), 41), bool)
    for event in events:
        windows = aligned_windows(event, count, analysis)
        summaries = {label: {name: average(value, item["indices"])
                             for name, value in features.items()}
                     if item["available"] else None
                     for label, item in windows.items() if label != "aligned"}
        pre = windows["pre"]["indices"]
        during = windows["during"]["indices"]
        ranking = None
        if pre is not None and during is not None:
            delta = contribution[during].mean(axis=0) - contribution[pre].mean(axis=0)
            magnitude = np.linalg.norm(delta, axis=0)
            ranking = [{"muscle_index": int(i), "leg": groups[i]["leg"],
                        "muscle": groups[i]["muscle"],
                        "command_delta_l2_rad": float(magnitude[i])}
                       for i in np.argsort(-magnitude, kind="stable")[:10]]
        center = (event["entry_index"] + event["exit_index"]) // 2
        candidate = center + np.arange(-20, 21)
        valid = (candidate >= 0) & (candidate < count)
        aligned_indices[event["event_index"]] = np.where(valid, candidate, -1)
        aligned_valid[event["event_index"]] = valid
        all_events.append({**event, "windows": windows,
                           "means": summaries,
                           "noncausal_command_contribution_rank": ranking})
    return all_events, {"control_index": aligned_indices,
                        "available": aligned_valid,
                        "offset_s": np.arange(-20, 21) * .005}


def aggregate_main_event_means(events):
    """Equal-event means of available per-event window means, not pooled samples."""
    main = [event for event in events if event["main_comparison"]]
    names = sorted({name for event in main for phase in ("pre", "during", "post")
                    if event["means"][phase] is not None
                    for name in event["means"][phase]})
    result = {"method": "equal weight per main event; finite event means only",
              "main_event_count": len(main), "phases": {}}
    for phase in ("pre", "during", "post"):
        result["phases"][phase] = {}
        for name in names:
            available = [event["means"][phase][name]["mean"]
                         for event in main if event["means"][phase] is not None]
            reference = next(event["means"][other][name]["mean"]
                for event in main for other in ("pre", "during", "post")
                if event["means"][other] is not None)
            shape = np.asarray(reference, dtype=float).shape
            matrix = np.asarray(available, dtype=float).reshape((-1, *shape))
            count = np.isfinite(matrix).sum(axis=0)
            mean = np.divide(np.where(np.isfinite(matrix), matrix, 0.).sum(axis=0),
                             count, out=np.full(shape, np.nan), where=count > 0)
            result["phases"][phase][name] = {
                "mean_of_event_means": prior.finite_receipt(mean.tolist())[0],
                "available_event_count": count.tolist(),
                "total_main_events": len(main)}
    return result


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    resources = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e226.effector-breaks.ledger.v1",
              "status": "gate_passed", "processes": 1, "retries": 0,
              "original_dynamics_calls": 0, "cuda_calls": 0,
              "physics_steps": 0, "solver_calls": 0, "fit_calls": 0,
              "phase_wall_s": {}, "storage_errors": []}
    result = {"schema": "flylab.m3.e226.effector-breaks.result.v1",
              "validity": "incomplete", "candidate_adopted": False,
              "F2_complete": False, "F6_exposed": False,
              "whole_goal_complete": False}
    primary = None
    process = None

    def receipt():
        ledger["wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "cost-receipt.json", ledger)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E226 " + name + ": " + repr(error))

    def budget(label):
        elapsed = time.perf_counter() - began
        ledger["phase_wall_s"][label] = elapsed
        if elapsed > resources["internal_s"]:
            raise TimeoutError("E226 saved-only analysis wall limit reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > resources["resident_bytes"]:
                raise MemoryError("E226 resident memory cap exceeded")
        receipt()

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            import psutil
            process = psutil.Process()
            paths, control, native, trace = load_joined_inputs(freeze)
            rows = trace["samples"]
            budget("joined_inputs")
            model = gait.load_model(paths["trace"])
            a = freeze["analysis"]
            if (model["start_s"] != a["window_start_s"]
                    or model["end_s"] != a["window_end_s"]
                    or not np.array_equal(model["moving"],
                        (model["t"] >= a["window_start_s"])
                        & (model["t"] < a["window_end_s"])
                        & (model["speed"] > a["moving_speed_threshold_mm_s"]))):
                raise ValueError("E226 original gait movement window changed")
            interval = aggregate_native_intervals(control["physics_time_s"],
                native, a["support_threshold_bw"])
            motion = derive_kinematics(rows, model)
            start = StateStore.load(paths["start"])
            adapter = start["adapter"]["canonical"]
            cal = adapter["calibration"]
            groups = adapter["groups"]
            if (len(groups) != 96 or cal["moment"].shape != (42, 96)
                    or cal["rest"].shape != (42,)
                    or not np.array_equal(cal["moment"], start[
                        "neural_current_trial_patch"]["parameters"]["moment"])
                    or not np.array_equal(cal["rest"], start[
                        "neural_current_trial_patch"]["parameters"]["rest"])):
                raise ValueError("E226 frozen canonical muscle anatomy changed")
            contribution = command_contributions(control["activation"],
                control["command"], cal["moment"], cal["rest"])
            budget("derived_raw")
            np.savez(out / "native-interval.npz", **interval)
            np.savez(out / "control-raw.npz", **motion,
                activation=control["activation"],
                recorded_command=control["command"],
                contribution_rad=contribution["contribution_rad"],
                canonical_command_rad=contribution["canonical_command_rad"],
                command_reconstruction_error_rad=contribution[
                    "command_reconstruction_error_rad"],
                canonical_valid=contribution["canonical_valid"],
                row0_held_command_rad=contribution["row0_held_command_rad"])
            budget("raw_saved")
            breaks = detect_breaks(model["t"], model["moving"], a)
            events, aligned = analyze_events(breaks, motion, interval,
                control["activation"], contribution["contribution_rad"], groups, a)
            np.savez(out / "aligned-raw.npz", **aligned)
            prior.write_receipt(out / "events.json", {"events": events,
                "all_event_count": len(events),
                "main_comparison_count": sum(e["main_comparison"] for e in events)})
            budget("events_saved")
            quality = json.loads(paths["quality"].read_text(encoding="utf-8"))
            summary = {"schema": "flylab.m3.e226.effector-breaks.summary.v1",
                "window_s": [a["window_start_s"], a["window_end_s"]],
                "control_rows": len(rows), "native_samples": len(native["time_s"]),
                "all_event_count": len(events),
                "main_comparison_count": sum(e["main_comparison"] for e in events),
                "main_event_indices": [e["event_index"] for e in events
                                       if e["main_comparison"]],
                "main_event_equal_weighted_descriptive":
                    aggregate_main_event_means(events),
                "original_quality_sha256": sha256(paths["quality"]),
                "original_quality_window": quality.get("window"),
                "row0_held_command": True,
                "native_control0_interval_available": False,
                "horizontal_ground_reaction_recorded": False,
                "world_axes": "x,z horizontal; y vertical; position millimeters",
                "initial_projection": "horizontal world velocity dotted with normalized initial body forward xz",
                "original_forward_speed": "3D physical body forward projection, Gaussian sigma 20 ms",
                "attribution_scope": "descriptive co-occurrence, not causal force",
                "slip_missing_preserved": True,
                "F1_F8_complete": False, "candidate_adopted": False}
            prior.write_receipt(out / "summary.json", summary)
            result.update(validity="valid_saved_descriptive_analysis",
                          summary=summary)
            budget("analysis_complete")
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        ledger["wall_s"] = time.perf_counter() - began
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                     if path.is_file())
        if ledger["output_bytes"] > resources["output_bytes"]:
            error = OSError("E226 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E226 output cap: " + repr(error))
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("cost-receipt.json", receipt)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.freeze, args.freeze_sha256, args.review,
                     args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
