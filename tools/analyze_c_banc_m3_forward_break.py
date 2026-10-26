"""E184 offline, event-aligned reading of two already recorded native arms.

No simulation, checkpoint restoration, neural execution, or model construction.
The output describes temporal association in one trajectory per arm, not causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path, PurePosixPath

import numpy as np

try:
    from tools.analyze_c_banc_biological_gait import position_forward_speed, runs
except ModuleNotFoundError:  # direct `python tools/analyze_...py` entry point
    from analyze_c_banc_biological_gait import position_forward_speed, runs

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/BANC_M3_E184_FORWARD_BREAK_DESIGN_20260924.md"
SOURCE_FILES = (
    "tools/analyze_c_banc_m3_forward_break.py",
    "tests/test_c_banc_forward_break.py",
)
METADATA = "verification/banc-natural-gait-m3-20260919/parent-e184-frozen-metadata.json"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
ARMS = ("baseline", "lh_plus")
ARM_ROOTS = {
    "baseline": "verification/banc-natural-gait-m3-20260919/e182-native-timing-recovery/baseline",
    "lh_plus": "verification/banc-natural-gait-m3-20260919/e183-remaining-native-timing/axis-lh/plus",
}
WINDOWS = {"pre": (-20, -10), "lead": (-10, 0), "post": (0, 10), "recover": (10, 21)}
FREE_BYTES = 1 << 30
OUTPUT_BYTES = 256 << 20
INTERNAL_S = 50.0
OUTPUT_DIR = ROOT / "verification/banc-natural-gait-m3-20260919/e184-forward-break"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Unsafe E184 relative path")
    part = PurePosixPath(name)
    if part.is_absolute() or any(p in ("", ".", "..") for p in part.parts):
        raise ValueError("Unsafe E184 relative path")
    if ":" in name or name.startswith("//"):
        raise ValueError("Unsafe E184 relative path")
    return Path(*part.parts)


def _within(base: Path, rel: str) -> Path:
    target = base / _safe_rel(rel)
    if any(p.is_symlink() for p in (base, *list(target.parents)[:-1], target)):
        raise ValueError("Symlink in E184 reviewed input")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("E184 path escaped root")
    return target


def gate(freeze_path: Path, review_path: Path):
    """Authenticate design/source and every file in both full raw inventories."""
    freeze, review = json_read(freeze_path), json_read(review_path)
    if freeze.get("schema") != "flylab.m3.e184.forward-break.freeze.v1":
        raise ValueError("E184 freeze schema required")
    if review.get("schema") != "flylab.m3.e184.forward-break.source-review.v1":
        raise ValueError("E184 parent source review required")
    if review.get("approved") is not True or review.get("freeze_sha256") != digest(
        freeze_path
    ):
        raise ValueError("E184 freeze was not parent reviewed")
    if freeze.get("design_path") != DESIGN.relative_to(ROOT).as_posix() or freeze.get(
        "design_sha256"
    ) != digest(DESIGN):
        raise ValueError("E184 design changed")
    if freeze.get("metadata_path") != METADATA:
        raise ValueError("E184 parent metadata path changed")
    metadata_path = _within(ROOT, METADATA)
    if digest(metadata_path) != freeze.get("metadata_sha256"):
        raise ValueError("E184 parent metadata changed")
    metadata = json_read(metadata_path)
    if (
        metadata.get("design_path") != freeze["design_path"]
        or metadata.get("design_sha256") != freeze["design_sha256"]
        or metadata.get("cumulative_before_s") != 4233.282374799717
        or metadata.get("cumulative_limit_s") != 4994
        or metadata.get("stage_reserve_s") != 60
        or metadata.get("event_metrics_computed") is not False
        or metadata.get("new_model_constructors") != 0
        or metadata.get("neural_steps") != 0
        or metadata.get("physics_steps") != 0
    ):
        raise ValueError("E184 parent metadata contract differs")
    reused = metadata.get("reused_source_sha256")
    if not isinstance(reused, dict) or len(reused) != 6:
        raise ValueError("E184 six reused semantics sources required")
    actual_sources = {
        name: digest(_within(ROOT, name)) for name in (*SOURCE_FILES, *reused)
    }
    if freeze.get("source_hashes") != actual_sources:
        raise ValueError("E184 source changed")
    if reused != {name: actual_sources[name] for name in reused}:
        raise ValueError("E184 reused semantics source changed")
    if (
        review.get("design_sha256") != freeze["design_sha256"]
        or review.get("source_hashes") != freeze["source_hashes"]
        or review.get("metadata_sha256") != freeze["metadata_sha256"]
    ):
        raise ValueError("E184 source review identity differs")
    parameters = {
        "control_dt_s": 0.005,
        "native_dt_s": 0.0001,
        "gaussian_sigma_s": 0.02,
        "support_threshold_bw": 0.02,
        "forward_threshold_mm_s": 0.5,
        "analysis_window_s": [0.5, 3.0],
        "event_half_window_controls": 20,
        "internal_wall_s": INTERNAL_S,
        "external_wall_s": 55,
        "cleanup_reserve_s": 5,
        "stage_wall_cap_s": 60,
        "output_bytes_cap": OUTPUT_BYTES,
        "minimum_free_bytes": FREE_BYTES,
        "inherited_cumulative_wall_s": 4233.282374799717,
        "cumulative_cap_s": 4994.0,
    }
    if freeze.get("parameters") != parameters:
        raise ValueError("E184 frozen analysis parameters differ")
    arms = metadata.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("Exact two E184 arms required")
    required = {
        "trace.json",
        "native-contact.npz",
        "native-recruitment-state.npz",
        "timing-parameters.npz",
        "result.json",
        "biological-gait-3s.json",
        "start/state.json",
        "start/manifest.json",
    }
    normalized_arms = {}
    expected_input = {}
    for arm in ARMS:
        entry = arms[arm]
        if entry.get("directory") != ARM_ROOTS[arm]:
            raise ValueError("E184 arm root changed")
        base = _within(ROOT, entry["directory"])
        raw = entry.get("raw_sha256")
        if not isinstance(raw, dict):
            raise TypeError("E184 full raw inventory required")
        prefix = entry["directory"] + "/"
        if any(not name.startswith(prefix) for name in raw):
            raise ValueError("E184 raw inventory path outside arm")
        files = {name[len(prefix) :]: sha for name, sha in raw.items()}
        if not required.issubset(files):
            raise ValueError("E184 required raw files missing from inventory")
        listed = list(base.rglob("*"))
        if any(p.is_symlink() for p in listed):
            raise ValueError("Symlink in E184 reviewed input")
        observed = {p.relative_to(base).as_posix() for p in listed if p.is_file()}
        if observed != set(files):
            raise ValueError("E184 full raw file inventory changed")
        for name, sha in files.items():
            path = _within(base, name)
            if not path.is_file() or digest(path) != sha:
                raise ValueError("E184 reviewed raw changed: " + arm + "/" + name)
        normalized_arms[arm] = {"root": entry["directory"], "files": files}
        expected_input.update(raw)
        path = entry["raw_review_path"]
        sha = entry["raw_review_sha256"]
        if digest(_within(ROOT, path)) != sha:
            raise ValueError("E184 parent arm raw review changed")
        expected_input[path] = sha
    if metadata.get("input_sha256") != expected_input:
        raise ValueError("E184 parent full input inventory differs")
    if (
        digest(_within(ROOT, metadata["previous_review_path"]))
        != metadata["previous_review_sha256"]
    ):
        raise ValueError("E184 E183 final review changed")
    joint = metadata.get("joint_row_provenance")
    if not isinstance(joint, dict) or set(joint) != {"path", "sha256", "method"}:
        raise ValueError("E184 joint row provenance required")
    source = _within(ROOT, joint["path"])
    if not source.is_file() or digest(source) != joint["sha256"] or not joint["method"]:
        raise ValueError("E184 joint row provenance changed")
    provenance = json_read(source)
    joint_map = metadata.get("joint_row_leg_index")
    if (
        not isinstance(joint_map, list)
        or len(joint_map) != 42
        or any(type(x) is not int or x not in range(6) for x in joint_map)
        or any(joint_map.count(i) != 7 for i in range(6))
        or joint_map[30] != LEGS.index("rm")
    ):
        raise ValueError("E184 complete parent-reviewed joint row map required")
    if (
        provenance.get("schema") != "flylab.m3.e184.joint-row-provenance.v1"
        or provenance.get("joint_row_leg_index") != joint_map
        or provenance.get("zero_moment_rows") != [30]
        or provenance.get("zero_row_name") != "c_thorax-rm_coxa-pitch"
        or len(provenance.get("joint_name_suffixes", [])) != 42
        or provenance["joint_name_suffixes"][30] != provenance["zero_row_name"]
    ):
        raise ValueError("E184 joint row provenance mapping differs")
    source_hashes = provenance.get("source_hashes")
    if not isinstance(source_hashes, dict) or len(source_hashes) != 8:
        raise ValueError("E184 pinned joint order sources required")
    for name, sha in source_hashes.items():
        if digest(_within(ROOT, name)) != sha:
            raise ValueError("E184 pinned joint order source changed")
    return {"arms": normalized_arms, "joint_row_leg_index": joint_map}, review


def _array(ref, base: Path, files: dict[str, str]):
    if not isinstance(ref, dict) or set(ref) != {"__npy__"}:
        raise ValueError("E184 checkpoint array reference required")
    name = ref["__npy__"]
    if (
        not isinstance(name, str)
        or not name.startswith("array_")
        or not name.endswith(".npy")
    ):
        raise ValueError("Unsafe E184 checkpoint array reference")
    key = "start/" + name
    if key not in files:
        raise ValueError("Unreviewed E184 checkpoint array")
    return np.load(_within(base / "start", name), allow_pickle=False)


def leg_maps(base: Path, files: dict[str, str], frozen_joint_rows):
    state = json_read(base / "start/state.json")
    adapter = state["adapter"]
    groups = adapter["groups"]
    if len(groups) != 96:
        raise ValueError("E184 requires 96 muscle groups")
    muscle_leg = tuple(group["leg"] for group in groups)
    if any(leg not in LEGS for leg in muscle_leg):
        raise ValueError("Unknown E184 muscle leg")
    if tuple(dict.fromkeys(muscle_leg)) != LEGS:
        raise ValueError("E184 muscle leg order differs")
    moment = np.asarray(_array(adapter["calibration"]["moment"], base, files))
    with np.load(base / "timing-parameters.npz", allow_pickle=False) as p:
        membership = np.asarray(p["membership"])
    if moment.shape != (42, 96) or not np.isfinite(moment).all():
        raise ValueError("Invalid E184 calibration moment")
    if membership.shape != (391,) or not np.issubdtype(membership.dtype, np.integer):
        raise ValueError("Invalid E184 MN membership")
    if np.any((membership < 0) | (membership >= 96)):
        raise ValueError("E184 MN membership outside muscle groups")
    assigned = np.full(391, -1, dtype=np.int64)
    for group_index, group in enumerate(groups):
        columns = np.asarray(group.get("columns"))
        if columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer):
            raise ValueError("E184 group columns require integer MN indices")
        if np.any((columns < 0) | (columns >= 391)):
            raise ValueError("E184 group column outside 391 MNs")
        if np.any(assigned[columns] != -1) or len(np.unique(columns)) != len(columns):
            raise ValueError("E184 group columns duplicate MN")
        assigned[columns] = group_index
    if not np.array_equal(assigned, membership):
        raise ValueError("E184 group columns do not partition and match membership")
    mn_leg = np.array([LEGS.index(muscle_leg[int(x)]) for x in membership])
    muscle_leg_index = np.array([LEGS.index(x) for x in muscle_leg])
    if len(frozen_joint_rows) != 42:
        raise ValueError("E184 requires complete frozen joint rows")
    row_leg = []
    zero_rows = []
    for index, row in enumerate(moment):
        columns = np.flatnonzero(row != 0)
        identities = set(muscle_leg_index[columns].tolist())
        if not columns.size:
            zero_rows.append(index)
            row_leg.append(int(frozen_joint_rows[index]))
        elif len(identities) != 1 or identities != {frozen_joint_rows[index]}:
            raise ValueError("Every E184 joint row needs exactly one leg")
        else:
            row_leg.append(identities.pop())
    if zero_rows != [30] or frozen_joint_rows[30] != LEGS.index("rm"):
        raise ValueError("E184 requires exact documented zero-moment RM row30")
    row_leg = np.asarray(row_leg, dtype=np.int64)
    if any(np.count_nonzero(row_leg == i) != 7 for i in range(6)):
        raise ValueError("E184 requires seven joint rows per leg")
    if any(np.count_nonzero(mn_leg == i) == 0 for i in range(6)):
        raise ValueError("E184 MN leg has no members")
    return muscle_leg_index, mn_leg, row_leg


def check_control(rows):
    if len(rows) != 601:
        raise ValueError("E184 requires 601 control rows")
    t = np.asarray([r["time_s"] for r in rows], dtype=float)
    ticks = [r["control_tick"] for r in rows]
    if (
        ticks != list(range(601))
        or not np.isfinite(t).all()
        or not np.allclose(t, np.arange(601) * 0.005, atol=1e-8, rtol=0)
    ):
        raise ValueError("E184 control clock has a gap")
    for i, row in enumerate(rows):
        interval = row.get("command_interval")
        if i == 0:
            if interval is not None:
                raise ValueError("E184 row0 has no driven interval")
        elif (
            not isinstance(interval, dict)
            or interval.get("start_tick") != i - 1
            or interval.get("end_tick") != i
            or not math.isclose(
                interval.get("start_s", float("nan")), (i - 1) * 0.005, abs_tol=1e-8
            )
            or not math.isclose(
                interval.get("end_s", float("nan")), i * 0.005, abs_tol=1e-8
            )
        ):
            raise ValueError("E184 command interval differs")
    return t


def native_bins(native):
    t = np.asarray(native["time_s"])
    load = np.asarray(native["normal_load_bw"])
    slip = np.asarray(native["slip_mm_s"])
    valid = np.asarray(native["slip_valid"])
    if (
        t.shape != (30000,)
        or not np.isfinite(t).all()
        or not np.allclose(t, np.arange(1, 30001) * 0.0001, atol=1e-8, rtol=0)
    ):
        raise ValueError("E184 native clock has a gap")
    if (
        load.shape != (30000, 6)
        or slip.shape != (30000, 6)
        or valid.shape != (30000, 6)
    ):
        raise ValueError("E184 native contact shape differs")
    if valid.dtype != np.bool_ or not np.isfinite(load).all() or np.any(load < 0):
        raise ValueError("Invalid E184 contact load or validity")
    support = load >= 0.02
    if np.any(support & (~valid | ~np.isfinite(slip))):
        raise ValueError("E184 supported slip is invalid")
    # Unloaded slip is missing, even if a recording happens to contain zero.
    load, support, slip = (a.reshape(600, 50, 6) for a in (load, support, slip))
    numerator = np.sum(np.where(support, load * np.where(support, slip, 0), 0), axis=1)
    denominator = np.sum(np.where(support, load, 0), axis=1)
    weighted = np.full((601, 6), np.nan)
    weighted[1:] = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    mean_load = np.full((601, 6), np.nan)
    mean_load[1:] = load.mean(axis=1)
    fraction = np.full((601, 6), np.nan)
    fraction[1:] = support.mean(axis=1)
    support_count = np.full(601, np.nan)
    support_count[1:] = support.sum(axis=2).mean(axis=1)
    zero_fraction = np.full(601, np.nan)
    zero_fraction[1:] = (support.sum(axis=2) == 0).mean(axis=1)
    supported_samples = np.full((601, 6), np.nan)
    supported_samples[1:] = support.sum(axis=1)
    slip_numerator = np.full((601, 6), np.nan)
    slip_numerator[1:] = numerator
    slip_denominator = np.full((601, 6), np.nan)
    slip_denominator[1:] = denominator
    return {
        "normal_load_bw": mean_load,
        "support_fraction": fraction,
        "support_count": support_count,
        "zero_support_fraction": zero_fraction,
        "supported_slip_mm_s": weighted,
        "supported_native_samples": supported_samples,
        "supported_slip_weighted_sum_bw_mm_s": slip_numerator,
        "supported_slip_load_sum_bw": slip_denominator,
    }


def select_events(speed):
    """Preserve every downward crossing; select complete, nonoverlapping windows."""
    speed = np.asarray(speed, dtype=float)
    if speed.shape != (601,) or not np.isfinite(speed).all():
        raise ValueError("E184 speed shape or finiteness")
    events = []
    previous_end = -1
    for i in range(100, 600):
        if speed[i - 1] > 0.5 and speed[i] <= 0.5:
            first, last = i - 20, i + 20
            reason = None
            if first < 100 or last >= 600:
                reason = "edge_window"
            elif first <= previous_end:
                reason = "overlap_selected_window"
            else:
                previous_end = last
            events.append(
                {
                    "index": i,
                    "time_s": round(i * 0.005, 10),
                    "first_index": first,
                    "last_index": last,
                    "selected": reason is None,
                    "exclusion_reason": reason,
                }
            )
    return events


def _finite_summary(values):
    a = np.asarray(values, dtype=float)
    valid = np.isfinite(a)
    return {
        "mean": float(a[valid].mean()) if valid.any() else None,
        "valid_samples": int(valid.sum()),
        "missing_samples": int(a.size - valid.sum()),
    }


def _weighted_slip_summary(signals, first, end):
    weighted = np.asarray(signals["supported_slip_weighted_sum_bw_mm_s"][first:end])
    load = np.asarray(signals["supported_slip_load_sum_bw"][first:end])
    native_count = np.asarray(signals["supported_native_samples"][first:end])
    result = []
    for leg in range(6):
        valid = np.isfinite(load[:, leg]) & (load[:, leg] > 0)
        denominator = float(load[valid, leg].sum())
        result.append(
            {
                "mean": float(weighted[valid, leg].sum() / denominator)
                if denominator > 0
                else None,
                "valid_samples": int(valid.sum()),
                "missing_samples": int(len(load) - valid.sum()),
                "supported_native_samples": int(native_count[valid, leg].sum()),
                "native_load_sum_bw": denominator,
                "aggregation": "sum(native supported load * slip) / sum(native supported load)",
            }
        )
    return result


def event_windows(signals, events):
    selected = [e for e in events if e["selected"]]
    output = []
    for event in selected:
        i = event["index"]
        periods = {}
        for label, (start, end) in WINDOWS.items():
            periods[label] = {}
            for key, value in signals.items():
                if key == "supported_slip_mm_s":
                    periods[label][key] = _weighted_slip_summary(
                        signals, i + start, i + end
                    )
                    continue
                sliced = np.asarray(value[i + start : i + end])
                periods[label][key] = (
                    [_finite_summary(col) for col in sliced.T]
                    if sliced.ndim == 2
                    else _finite_summary(sliced)
                )
        output.append({"index": i, "time_s": event["time_s"], "periods": periods})
    return output


def align_curves(signals, events):
    indices = [e["index"] for e in events if e["selected"]]
    curves = {}
    means = {}
    for key, value in signals.items():
        arr = np.asarray(value, dtype=float)
        curve = (
            np.stack([arr[i - 20 : i + 21] for i in indices])
            if indices
            else np.empty((0, 41, *arr.shape[1:]))
        )
        curves[key] = curve
        count = np.isfinite(curve).sum(axis=0)
        total = np.nansum(curve, axis=0)
        means[key] = np.divide(
            total, count, out=np.full_like(total, np.nan), where=count > 0
        )
    if "supported_slip_mm_s" in signals:
        numerator = curves["supported_slip_weighted_sum_bw_mm_s"]
        denominator = curves["supported_slip_load_sum_bw"]
        total_num = np.nansum(numerator, axis=0)
        total_den = np.nansum(denominator, axis=0)
        means["supported_slip_mm_s"] = np.divide(
            total_num,
            total_den,
            out=np.full_like(total_num, np.nan),
            where=total_den > 0,
        )
    return curves, means


def _read_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in keys}


def read_arm(base: Path, files: dict[str, str], frozen_joint_rows):
    trace = json_read(base / "trace.json")
    rows = trace["samples"]
    t = check_control(rows)
    position = np.asarray([r["body"]["position"] for r in rows], dtype=float)
    basis = np.asarray([r["body"]["basis"] for r in rows], dtype=float)
    if (
        position.shape != (601, 3)
        or basis.shape != (601, 3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(basis).all()
    ):
        raise ValueError("Invalid E184 body coordinates")
    if not np.allclose(np.einsum("tji,tjk->tik", basis, basis), np.eye(3), atol=1e-8):
        raise ValueError("Invalid E184 body forward basis")
    forward = basis[:, :, 0]
    speed = position_forward_speed(position, forward, 0.005)
    if any(len(row["body"]["legs"][leg]) != 8 for row in rows for leg in LEGS):
        raise ValueError("E184 requires all eight ordered leg body origins")
    leg_origins = np.asarray(
        [[r["body"]["legs"][leg] for leg in LEGS] for r in rows], dtype=float
    )
    if leg_origins.shape != (601, 6, 8, 3) or not np.isfinite(leg_origins).all():
        raise ValueError("E184 requires all eight ordered leg body origins")
    feet = leg_origins[:, :, -1, :]
    foot_world_forward = np.sum(
        np.gradient(feet, 0.005, axis=0) * forward[:, None, :], axis=2
    )
    foot_body_forward = np.sum(
        (feet - position[:, None, :]) * forward[:, None, :], axis=2
    )
    foot_relative_rate = np.gradient(foot_body_forward, 0.005, axis=0)
    targets = np.asarray([r["joint_targets_rad"] for r in rows], dtype=float)
    actual = np.asarray([r["joint_angles_rad"] for r in rows], dtype=float)
    if (
        targets.shape != (601, 42)
        or actual.shape != (601, 42)
        or not np.isfinite(targets).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Invalid E184 joint observations")
    native = _read_npz(
        base / "native-contact.npz",
        ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid"),
    )
    binned = native_bins(native)
    recruitment = _read_npz(
        base / "native-recruitment-state.npz",
        (
            "control_tick",
            "network_tick",
            "physics_time_s",
            "complete",
            "motor_rate",
            "motor_rate_available",
            "activation",
            "activation_available",
        ),
    )
    if (
        not np.array_equal(recruitment["control_tick"], np.arange(601))
        or not np.array_equal(recruitment["network_tick"], np.arange(601) * 20)
        or not np.allclose(recruitment["physics_time_s"], t, atol=1e-8, rtol=0)
        or not np.array_equal(recruitment["complete"], np.ones(601, bool))
        or not np.array_equal(recruitment["motor_rate_available"], np.ones(601, bool))
        or not np.array_equal(recruitment["activation_available"], np.ones(601, bool))
    ):
        raise ValueError("E184 recruitment clock or availability has a gap")
    motor = np.asarray(recruitment["motor_rate"], dtype=float)
    active = np.asarray(recruitment["activation"], dtype=float)
    if (
        motor.shape != (601, 391)
        or active.shape != (601, 96)
        or not np.isfinite(motor).all()
        or not np.isfinite(active).all()
    ):
        raise ValueError("Invalid E184 MN or activation array")
    muscle_leg, mn_leg, row_leg = leg_maps(base, files, frozen_joint_rows)
    mn_mean = np.column_stack([motor[:, mn_leg == j].mean(axis=1) for j in range(6)])
    muscle_mean = np.column_stack(
        [active[:, muscle_leg == j].mean(axis=1) for j in range(6)]
    )
    muscle_change = np.r_[
        np.full((1, 6), np.nan),
        np.column_stack(
            [
                np.sqrt(
                    np.mean(np.diff(active[:, muscle_leg == j], axis=0) ** 2, axis=1)
                )
                for j in range(6)
            ]
        ),
    ]
    error_rms = np.column_stack(
        [
            np.sqrt(
                np.mean(
                    (targets[:, row_leg == j] - actual[:, row_leg == j]) ** 2, axis=1
                )
            )
            for j in range(6)
        ]
    )
    target_change = np.r_[
        np.full((1, 6), np.nan),
        np.column_stack(
            [
                np.sqrt(np.mean(np.diff(targets[:, row_leg == j], axis=0) ** 2, axis=1))
                for j in range(6)
            ]
        ),
    ]
    signals = {
        "speed_mm_s": speed,
        **binned,
        "tarsus5_world_forward_mm_s": foot_world_forward,
        "tarsus5_body_relative_forward_mm_s": foot_relative_rate,
        "mn_mean_model_rate": mn_mean,
        "muscle_activation_mean": muscle_mean,
        "muscle_activation_change_rms": muscle_change,
        "joint_tracking_error_rms_rad": error_rms,
        "joint_command_change_rms_rad": target_change,
    }
    events = select_events(speed)
    curves, means = align_curves(signals, events)
    result = json_read(base / "result.json")
    quality = json_read(base / "biological-gait-3s.json")
    window = (t >= 0.5) & (t < 3)
    objective = float(np.maximum(0, 0.5 - speed[window]).mean())
    longest = max(
        ((b - a - 1) * 0.005 for a, b in runs(window & (speed > 0.5))), default=0.0
    )
    previous_longest = quality["window"]["longest_forward_bout_s"]
    if not math.isclose(longest, previous_longest, abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError("E184 forward bout does not reproduce prior result")
    old_objective = result.get("objective_mm_s")
    if old_objective is not None and not math.isclose(
        objective, old_objective, abs_tol=1e-9, rel_tol=1e-9
    ):
        raise ValueError("E184 forward objective does not reproduce prior result")
    summary = {
        "events": events,
        "selected_event_count": sum(e["selected"] for e in events),
        "window_s": [0.5, 3.0],
        "forward_threshold_mm_s": 0.5,
        "objective_mm_s_recomputed": objective,
        "objective_mm_s_previous": old_objective,
        "longest_forward_bout_s_recomputed": longest,
        "longest_forward_bout_s_previous": previous_longest,
        "event_periods": event_windows(signals, events),
        "event_period_definition_ms": {
            "pre": {"range": [-100, -50], "right_inclusive": False, "samples": 10},
            "lead": {"range": [-50, 0], "right_inclusive": False, "samples": 10},
            "post": {"range": [0, 50], "right_inclusive": False, "samples": 10},
            "recover": {"range": [50, 100], "right_inclusive": True, "samples": 11},
        },
        "signal_names": list(signals),
        "signal_semantics": {
            "speed_mm_s": "mm/s; centered thorax position derivative projected on body forward and symmetric Gaussian sigma 20 ms; noncausal smoothing",
            "normal_load_bw": "body weight units; arithmetic mean of 50 native loads; includes adhesion",
            "support_fraction": "fraction of 50 native samples with leg load >= 0.02 bw",
            "support_count": "legs; mean native count of six supported legs per control interval",
            "zero_support_fraction": "fraction of 50 native samples with no supported leg",
            "supported_slip_mm_s": "mm/s; native supported-load weighted slip; undefined without support; event bands and mean curves pool original native weights",
            "supported_native_samples": "count of native supported samples in each control interval; row0 missing",
            "supported_slip_weighted_sum_bw_mm_s": "sum(native supported load_bw * slip_mm_s), sufficient statistic; zero when no supported sample after row0",
            "supported_slip_load_sum_bw": "sum(native supported load_bw), sufficient statistic; zero when no supported sample after row0",
            "tarsus5_world_forward_mm_s": "mm/s; last tarsus body-origin world velocity projected on current body forward; not contact slip",
            "tarsus5_body_relative_forward_mm_s": "mm/s; derivative of last tarsus origin body-relative forward position including body rotation",
            "mn_mean_model_rate": "model rate units; arithmetic mean of every assigned MN per leg",
            "muscle_activation_mean": "model activation units; arithmetic mean of assigned muscles per leg",
            "muscle_activation_change_rms": "model activation units per 5 ms step; root mean square change across assigned muscles",
            "joint_tracking_error_rms_rad": "rad; root mean square of requested minus observed angles across seven joints per leg",
            "joint_command_change_rms_rad": "rad per 5 ms step; root mean square command change across seven joints per leg",
        },
        "aggregation_semantics": "Except supported_slip_mm_s, event band means are arithmetic means of finite 5 ms rows; mean curves are arithmetic across selected events, which are repeated observations from one trajectory. Supported slip is pooled from native supported-load numerator and denominator, first per 5 ms bin, then per event band and per lag across events. No unsupported slip is imputed.",
        "missingness": {
            key: int(np.isnan(value).sum()) for key, value in signals.items()
        },
        "limitations": [
            "single repeated trajectory; no independent biological replicates or causal test",
            "adhesion contributes to measured normal load",
            "tarsus5 body origin is not the contact point",
            "ground tangential force and fore-aft impulse unavailable",
            "row i command drove the preceding (t[i-1], t[i]] interval",
            "centered position derivative and symmetric Gaussian 20 ms smoothing use future samples (approximately 80 ms kernel support); apparent lead to the smoothed speed crossing is not biological causal onset",
        ],
    }
    arrays = {
        "time_s": t,
        "relative_time_s": np.arange(-20, 21) * 0.005,
        "body_position_mm": position,
        "body_basis_ui": basis,
        "tarsus5_origin_mm": feet,
        "joint_command_rad": targets,
        "joint_actual_rad": actual,
        "motor_rate_391": motor,
        "muscle_activation_96": active,
        "muscle_leg_index": muscle_leg,
        "mn_leg_index": mn_leg,
        "joint_row_leg_index": row_leg,
        "event_indices_all": np.array([e["index"] for e in events], dtype=int),
        "event_indices_selected": np.array(
            [e["index"] for e in events if e["selected"]], dtype=int
        ),
    }
    arrays.update({"signal_" + key: value for key, value in signals.items()})
    arrays.update({"curve_" + key: value for key, value in curves.items()})
    arrays.update({"mean_curve_" + key: value for key, value in means.items()})
    return summary, arrays


def _json_new(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _available_leg_mean(value):
    value = np.asarray(value, dtype=float)
    count = np.isfinite(value).sum(axis=1)
    return np.divide(
        np.nansum(value, axis=1),
        count,
        out=np.full(value.shape[0], np.nan),
        where=count > 0,
    )


def _pooled_leg_slip(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    total_num = np.nansum(numerator, axis=1)
    total_den = np.nansum(denominator, axis=1)
    return np.divide(
        total_num, total_den, out=np.full_like(total_num, np.nan), where=total_den > 0
    )


def _plots(out: Path, outputs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for arm in ARMS:
        _, a = outputs[arm]
        t = a["time_s"]
        axes[0].plot(t, a["signal_speed_mm_s"], label=arm)
        axes[1].plot(t, a["signal_support_count"], label=arm)
        axes[2].plot(t, _available_leg_mean(a["signal_support_fraction"]), label=arm)
        for event in outputs[arm][0]["events"]:
            axes[0].axvline(
                event["time_s"], color="k" if event["selected"] else "gray", alpha=0.18
            )
    axes[0].axhline(0.5, color="red", linestyle="--", label="0.5 mm/s")
    for ax, label in zip(
        axes,
        (
            "body forward speed (mm/s)",
            "mean native supported legs",
            "mean leg support fraction",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.legend()
    axes[-1].set_xlabel("model time (s)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    event_panels = (
        ("speed_mm_s", "body forward speed (mm/s)"),
        ("support_count", "mean native supported legs"),
        ("supported_slip_mm_s", "native-load pooled contact slip (mm/s)"),
        (
            "tarsus5_body_relative_forward_mm_s",
            "mean tarsus5 body-relative rate (mm/s)",
        ),
        ("mn_mean_model_rate", "mean leg MN model rate"),
        ("joint_tracking_error_rms_rad", "mean leg joint tracking RMS (rad)"),
    )
    for arm in ARMS:
        summary, a = outputs[arm]
        relative = a["relative_time_s"] * 1000
        if summary["selected_event_count"]:
            for ax, (key, _) in zip(axes, event_panels, strict=True):
                values = a["mean_curve_" + key]
                if key == "supported_slip_mm_s":
                    values = _pooled_leg_slip(
                        a["mean_curve_supported_slip_weighted_sum_bw_mm_s"],
                        a["mean_curve_supported_slip_load_sum_bw"],
                    )
                elif values.ndim == 2:
                    values = _available_leg_mean(values)
                ax.plot(
                    relative,
                    values,
                    label=f"{arm} (n={summary['selected_event_count']})",
                )
    axes[0].axhline(0.5, color="red", linestyle="--", label="0.5 mm/s")
    for ax, (_, label) in zip(axes, event_panels, strict=True):
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    if not any(outputs[arm][0]["selected_event_count"] for arm in ARMS):
        axes[2].text(
            0.5,
            0.5,
            "No eligible complete, disjoint crossing windows",
            transform=axes[2].transAxes,
            ha="center",
        )
    axes[-1].set_xlabel("time from downward crossing (ms)")
    fig.tight_layout()
    fig.savefig(out / "event_mean.png", dpi=140)
    plt.close(fig)


def execute(freeze_path: Path, review_path: Path, output: Path):
    started = time.perf_counter()
    if output.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("E184 production output must be the fixed planned directory")
    if output.exists():
        raise FileExistsError("E184 output is create-only")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < FREE_BYTES:
        raise ValueError("E184 requires 1 GiB free and an existing output parent")
    output.mkdir(exist_ok=False)
    receipt = {
        "schema": "flylab.m3.e184.forward-break.terminal.v1",
        "status": "failed",
        "native_runs": 0,
        "neural_steps": 0,
        "physics_steps": 0,
        "inherited_cumulative_wall_s": 4233.282374799717,
        "cumulative_cap_s": 4994.0,
        "elapsed_scope": "execute entry until terminal serialization begins; excludes Python/module imports and terminal write",
        "charged_wall_source": "parent outer launch-to-exit supervisor receipt",
    }
    try:
        freeze, _ = gate(freeze_path, review_path)
        if time.perf_counter() - started > INTERNAL_S:
            raise TimeoutError("E184 internal 50-second wall limit")
        receipt.update(
            freeze_sha256=digest(freeze_path), source_review_sha256=digest(review_path)
        )
        outputs = {}
        for arm in ARMS:
            entry = freeze["arms"][arm]
            outputs[arm] = read_arm(
                _within(ROOT, entry["root"]),
                entry["files"],
                freeze["joint_row_leg_index"],
            )
            if time.perf_counter() - started > INTERNAL_S:
                raise TimeoutError("E184 internal 50-second wall limit")
        combined = {
            "schema": "flylab.m3.e184.forward-break.analysis.v1",
            "arms": {arm: outputs[arm][0] for arm in ARMS},
            "interpretation": "temporal observation only; no natural-gait success or causal verdict",
        }
        _json_new(output / "analysis.json", combined)
        arrays = {
            arm + "__" + key: value
            for arm in ARMS
            for key, value in outputs[arm][1].items()
        }
        np.savez_compressed(output / "aligned.npz", **arrays)
        _plots(output, outputs)
        if time.perf_counter() - started > INTERNAL_S:
            raise TimeoutError("E184 internal 50-second wall limit")
        size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        if size > OUTPUT_BYTES:
            raise ValueError("E184 output exceeded 256 MiB")
        receipt.update(
            status="complete",
            output_bytes=size,
            selected_events={
                arm: outputs[arm][0]["selected_event_count"] for arm in ARMS
            },
        )
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["elapsed_wall_s"] = time.perf_counter() - started
        receipt["cumulative_if_charged_wall_s"] = (
            receipt["inherited_cumulative_wall_s"] + receipt["elapsed_wall_s"]
        )
        _json_new(output / "terminal.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execute(args.freeze, args.source_review, args.output)


if __name__ == "__main__":
    main()
