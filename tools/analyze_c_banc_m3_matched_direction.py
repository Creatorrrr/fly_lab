"""E166 saved-data comparison of effective body-origin Z request directions.

Forward means an instantaneous speed > 0.5 mm/s, not validated natural gait.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e166.matched-direction.v1"
GATE_SCHEMA = "flylab.m3.e166.parent-source-review.v1"
OWNED = (
    "tools/analyze_c_banc_m3_matched_direction.py",
    "tests/test_c_banc_matched_direction.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
WINDOWS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0))


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def frozen_path(name: str) -> Path:
    require(isinstance(name, str), "Frozen path is not a string")
    path = Path(name)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def frozen_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for section in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        for name, digest in freeze[section].items():
            require(
                name not in hashes or hashes[name] == digest,
                f"Conflicting hash: {name}",
            )
            hashes[name] = digest
    return hashes


def verify_hashes(hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        require(sha256(frozen_path(name)) == digest, f"Frozen file changed: {name}")


def validate_freeze(freeze: dict[str, Any], path: Path, digest: str) -> dict[str, Any]:
    require(sha256(path) == digest, "Freeze SHA differs")
    require(freeze.get("schema") == SCHEMA, "Freeze schema differs")
    require(freeze.get("owned_files") == list(OWNED), "Owned files differ")
    require(
        sha256(frozen_path(freeze["design"])) == freeze["design_sha256"],
        "Design SHA differs",
    )
    c = freeze["sample_contract"]
    expected = {
        "legs": list(LEGS),
        "rows": 2000,
        "phase_rows": 2001,
        "first_tick": 1,
        "final_tick": 2000,
        "control_dt_s": 0.005,
        "clock_atol": 2e-8,
        "zero_mm": 1e-12,
        "request_index": 2,
        "segments": 5,
        "common_first_tick": 100,
        "common_last_tick": 1923,
        "common_count": 1824,
        "window_s": [0.5, 10.0],
        "time_windows_s": [[0.5, 2.0], [2.0, 5.0], [5.0, 10.0]],
        "support_threshold_bw": 0.02,
        "forward_threshold_mm_s": 0.5,
        "ap_threshold_mm_s": 0.0,
        "phase_caliper_cycle": 0.025,
        "load_caliper_bw": 0.02,
        "ap_caliper_mm_s": 0.5,
        "matching": "nearest_normalized_squared_with_replacement_same_leg_same_window",
        "tie_break": "smallest_control_tick",
        "unmatched_tick": -1,
        "minimum_matched_cases": 20,
        "minimum_unique_controls": 20,
        "minimum_case_coverage": 0.5,
        "shared_control_nonpositive_fraction": 0.9,
        "separated_fraction_difference": 0.1,
        "quantiles": [0.05, 0.5, 0.95],
    }
    require(c == expected, "Sample/matching contract differs")
    limits = freeze["resource_limits"]
    require(
        limits["original_processes"] == 1 and limits["automatic_retries"] == 0,
        "Process contract differs",
    )
    require(
        limits["cpu_threads"] == 1 and limits["gpu_count"] == 0,
        "CPU/GPU contract differs",
    )
    require(
        freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "Outcome boundary differs",
    )
    return c


def validate_gate(
    path: Path, digest: str, freeze_sha: str, design_sha: str
) -> dict[str, Any]:
    require(sha256(path) == digest, "Parent gate SHA differs")
    gate = json.loads(path.read_text(encoding="utf-8"))
    source = {name: sha256(ROOT / name) for name in OWNED}
    require(gate.get("schema") == GATE_SCHEMA, "Parent gate schema differs")
    require(
        gate.get("approved") is True and gate.get("blocking_defects") == [],
        "Parent gate has blocking defects",
    )
    require(
        gate.get("accepts_single_matched_direction") is True,
        "Parent did not approve this comparison",
    )
    require(
        type(gate.get("original_processes_allowed")) is int
        and gate["original_processes_allowed"] == 1,
        "Parent process allowance differs",
    )
    require(
        type(gate.get("automatic_retries")) is int and gate["automatic_retries"] == 0,
        "Parent retry allowance differs",
    )
    require(
        gate.get("freeze_sha256") == freeze_sha
        and gate.get("design_sha256") == design_sha,
        "Parent gate freeze/design differs",
    )
    require(gate.get("source_sha256") == source, "Parent gate source differs")
    return gate


def snapshot(
    out: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    gate_path: Path,
    gate_sha: str,
    gate: dict[str, Any],
) -> dict[str, str]:
    files = {
        "design.md": (frozen_path(freeze["design"]), freeze["design_sha256"]),
        "freeze.json": (freeze_path, sha256(freeze_path)),
        "source-review.json": (gate_path, gate_sha),
        "analyzer.py": (ROOT / OWNED[0], gate["source_sha256"][OWNED[0]]),
        "tests.py": (ROOT / OWNED[1], gate["source_sha256"][OWNED[1]]),
    }
    directory = out / "source-snapshot"
    directory.mkdir()
    for name, (path, digest) in files.items():
        require(sha256(path) == digest, f"Live source changed: {name}")
        (directory / name).write_bytes(path.read_bytes())
        require(sha256(directory / name) == digest, f"Snapshot differs: {name}")
    return {name: digest for name, (_, digest) in files.items()}


def verify_snapshot(
    out: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    gate_path: Path,
    approved: dict[str, str],
) -> None:
    paths = {
        "design.md": frozen_path(freeze["design"]),
        "freeze.json": freeze_path,
        "source-review.json": gate_path,
        "analyzer.py": ROOT / OWNED[0],
        "tests.py": ROOT / OWNED[1],
    }
    for name, path in paths.items():
        require(sha256(path) == approved[name], f"Post live source changed: {name}")
        require(
            sha256(out / "source-snapshot" / name) == approved[name],
            f"Post snapshot changed: {name}",
        )


def as_array(
    source: dict[str, np.ndarray], key: str, shape: tuple[int, ...], kind: str
) -> np.ndarray:
    value = source[key]
    require(value.shape == shape, f"{key} shape differs: {value.shape}")
    valid_dtype = {
        "integer": np.issubdtype(value.dtype, np.integer),
        "boolean": np.issubdtype(value.dtype, np.bool_),
        "floating": np.issubdtype(value.dtype, np.floating),
    }[kind]
    require(valid_dtype, f"{key} dtype differs: {value.dtype}; expected {kind}")
    return value


def raw_inputs(direction: Any, phase_source: Any) -> dict[str, np.ndarray]:
    """Keep required original fields before shape, dtype or value assertions."""
    direction_keys = (
        "tick",
        "time_s",
        "z_mm",
        "normal_load_bw",
        "ego_ap_velocity_mm_s",
        "speed_mm_s",
        "completed",
        "forward_completed",
        "jacobian_completed",
    )
    phase_keys = ("control_time_s", "e1_phase", "e1_phase_available")
    return {
        **{f"direction_{key}": np.asarray(direction[key]) for key in direction_keys},
        **{f"phase_{key}": np.asarray(phase_source[key]) for key in phase_keys},
    }


def aligned_inputs(
    raw: dict[str, np.ndarray], contract: dict[str, Any]
) -> dict[str, np.ndarray]:
    n = contract["rows"]
    tick = as_array(raw, "direction_tick", (n,), "integer")
    time_s = as_array(raw, "direction_time_s", (n,), "floating")
    phase_time = as_array(raw, "phase_control_time_s", (n + 1,), "floating")
    phase = as_array(raw, "phase_e1_phase", (n + 1, 6), "floating")
    available = as_array(raw, "phase_e1_phase_available", (n + 1, 6), "boolean")
    z = as_array(raw, "direction_z_mm", (n, 3, 6, 5), "floating")[:, 2]
    return {
        "tick": tick,
        "time_s": time_s,
        "phase_time_s": phase_time[1:],
        "phase_cycle": phase[1:],
        "phase_available": available[1:],
        "effective_z_mm": z,
        "normal_load_bw": as_array(raw, "direction_normal_load_bw", (n, 6), "floating"),
        "ego_ap_velocity_mm_s": as_array(
            raw, "direction_ego_ap_velocity_mm_s", (n, 6), "floating"
        ),
        "speed_mm_s": as_array(raw, "direction_speed_mm_s", (n,), "floating"),
        "completed": as_array(raw, "direction_completed", (n,), "boolean"),
        "forward_completed": as_array(
            raw, "direction_forward_completed", (n,), "boolean"
        ),
        "jacobian_completed": as_array(
            raw, "direction_jacobian_completed", (n, 6, 5), "boolean"
        ),
        "phase_time_zero_s": np.asarray(phase_time[0]),
        "phase_zero_cycle": phase[0],
        "phase_zero_available": available[0],
    }


def diagnostics(
    data: dict[str, np.ndarray], c: dict[str, Any]
) -> dict[str, np.ndarray]:
    ticks = data["tick"]
    residual = data["time_s"] - data["phase_time_s"]
    expected = ticks * c["control_dt_s"]
    phase = data["phase_cycle"]
    available = data["phase_available"]
    return {
        **data,
        "clock_residual_s": residual,
        "expected_time_residual_s": data["time_s"] - expected,
        "phase_time_residual_s": data["phase_time_s"] - expected,
        "phase_finite_mask": np.isfinite(phase),
        "phase_in_cycle_mask": (phase >= 0) & (phase < 1),
        "unavailable_nan_mask": ~available & np.isnan(phase),
        "common_tick_mask": (ticks >= 100) & (ticks <= 1923),
        "observed_common_mask": (
            (data["time_s"] >= c["window_s"][0])
            & (data["time_s"] < c["window_s"][1])
            & available.all(axis=1)
        ),
    }


def validate_inputs(d: dict[str, np.ndarray], c: dict[str, Any]) -> np.ndarray:
    n = c["rows"]
    atol = c["clock_atol"]
    require(np.array_equal(d["tick"], np.arange(1, n + 1)), "Tick sequence differs")
    for key in (
        "time_s",
        "phase_time_s",
        "normal_load_bw",
        "ego_ap_velocity_mm_s",
        "speed_mm_s",
        "effective_z_mm",
    ):
        require(np.isfinite(d[key]).all(), f"{key} contains nonfinite values")
    require(
        np.isfinite(d["phase_time_zero_s"])
        and abs(float(d["phase_time_zero_s"])) <= atol,
        "Phase t0 differs",
    )
    require(np.all(np.abs(d["clock_residual_s"]) <= atol), "E165/E163 clocks differ")
    require(
        np.all(np.abs(d["expected_time_residual_s"]) <= atol), "E165 tick clock differs"
    )
    require(
        np.all(np.abs(d["phase_time_residual_s"]) <= atol), "E163 tick clock differs"
    )
    require(np.all(np.diff(d["time_s"]) > 0), "Time is not increasing")
    require(
        d["completed"].all()
        and d["forward_completed"].all()
        and d["jacobian_completed"].all(),
        "E165 completion differs",
    )
    require(np.all(d["normal_load_bw"] >= 0), "Negative support load")
    available = d["phase_available"]
    phase = d["phase_cycle"]
    require(
        np.all(
            np.isfinite(phase[available])
            & (phase[available] >= 0)
            & (phase[available] < 1)
        ),
        "Available phase outside cycle",
    )
    require(np.isnan(phase[~available]).all(), "Unavailable phase must be NaN")
    require(
        np.all(
            np.isfinite(d["phase_zero_cycle"][d["phase_zero_available"]])
            & (d["phase_zero_cycle"][d["phase_zero_available"]] >= 0)
            & (d["phase_zero_cycle"][d["phase_zero_available"]] < 1)
        ),
        "Initial phase outside cycle",
    )
    require(
        np.isnan(d["phase_zero_cycle"][~d["phase_zero_available"]]).all(),
        "Initial unavailable phase must be NaN",
    )
    common = d["common_tick_mask"]
    require(
        np.array_equal(d["observed_common_mask"], common),
        "Observed common phase mask differs from frozen ticks",
    )
    require(int(common.sum()) == c["common_count"], "Common tick count differs")
    require(available[common].all(), "Common tick phase unavailable")
    require(
        np.all(
            (d["time_s"][common] >= c["window_s"][0])
            & (d["time_s"][common] < c["window_s"][1])
        ),
        "Common time window differs",
    )
    return common


def validate_metadata(freeze: dict[str, Any]) -> None:
    paths = freeze["paths"]
    layout = json.loads(frozen_path(paths["layout"]).read_text(encoding="utf-8"))
    names = [[f"{leg}_tarsus{number}" for number in range(1, 6)] for leg in LEGS]
    require(
        layout.get("tarsal_body_names") == names, "Tarsal body origin order differs"
    )
    direction_review = json.loads(
        frozen_path(paths["direction_review"]).read_text(encoding="utf-8")
    )
    phase_review = json.loads(
        frozen_path(paths["phase_review"]).read_text(encoding="utf-8")
    )
    require(
        direction_review.get("schema") == "flylab.m3.e165.parent-final-review.v1"
        and direction_review.get("valid") is True,
        "E165 review differs",
    )
    require(
        phase_review.get("original_execution_valid") is True
        and phase_review.get("original_processes") == 1,
        "E163 review differs",
    )


def match_leg(
    tick: np.ndarray,
    time_s: np.ndarray,
    phase: np.ndarray,
    load: np.ndarray,
    ap: np.ndarray,
    z: np.ndarray,
    case_mask: np.ndarray,
    control_mask: np.ndarray,
    c: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Pure one-leg matching, with signed differences defined as case minus control."""
    cases = np.flatnonzero(case_mask)
    controls = np.flatnonzero(control_mask)
    m = len(cases)
    selected = np.full(m, -1, np.int64)
    eligible = np.zeros(m, np.int64)
    score = np.full(m, np.nan)
    differences = np.full((m, 4), np.nan)
    control_z = np.full((m, 5), np.nan)
    for row, i in enumerate(cases):
        pool = controls[
            np.logical_and(
                time_s[controls]
                >= next(w[0] for w in WINDOWS if w[0] <= time_s[i] < w[1]),
                time_s[controls]
                < next(w[1] for w in WINDOWS if w[0] <= time_s[i] < w[1]),
            )
        ]
        if not len(pool):
            continue
        phase_diff = phase[i] - phase[pool]
        phase_absolute = np.abs(phase_diff)
        phase_distance = np.minimum(phase_absolute, 1.0 - phase_absolute)
        phase_signed = np.where(phase_diff > 0.5, phase_diff - 1.0, phase_diff)
        phase_signed = np.where(phase_signed < -0.5, phase_signed + 1.0, phase_signed)
        load_diff = load[i] - load[pool]
        ap_diff = ap[i] - ap[pool]
        ok = (
            (phase_distance <= c["phase_caliper_cycle"])
            & (np.abs(load_diff) <= c["load_caliper_bw"])
            & (np.abs(ap_diff) <= c["ap_caliper_mm_s"])
        )
        eligible[row] = int(ok.sum())
        if not eligible[row]:
            continue
        candidates = np.flatnonzero(ok)
        costs = (
            (phase_distance[candidates] / c["phase_caliper_cycle"]) ** 2
            + (load_diff[candidates] / c["load_caliper_bw"]) ** 2
            + (ap_diff[candidates] / c["ap_caliper_mm_s"]) ** 2
        )
        order = np.lexsort((tick[pool[candidates]], costs))
        k = int(candidates[order[0]])
        j = int(pool[k])
        selected[row] = int(tick[j])
        score[row] = float(costs[order[0]])
        differences[row] = (
            phase_signed[k],
            load_diff[k],
            ap_diff[k],
            time_s[i] - time_s[j],
        )
        control_z[row] = z[j]
    case_z = z[cases]
    return {
        "case_tick": tick[cases],
        "control_tick": selected,
        "eligible_control_count": eligible,
        "score": score,
        "phase_difference_cycle": differences[:, 0],
        "load_difference_bw": differences[:, 1],
        "ap_difference_mm_s": differences[:, 2],
        "time_difference_s": differences[:, 3],
        "case_z_mm": case_z,
        "control_z_mm": control_z,
        "case_mean_z_mm": case_z.mean(axis=1),
        "case_min_z_mm": case_z.min(axis=1),
        "case_max_z_mm": case_z.max(axis=1),
        "control_mean_z_mm": control_z.mean(axis=1),
        "control_min_z_mm": control_z.min(axis=1),
        "control_max_z_mm": control_z.max(axis=1),
        "case_all_nonpositive": np.all(case_z <= c["zero_mm"], axis=1),
        "control_all_nonpositive": np.where(
            selected >= 0, np.all(control_z <= c["zero_mm"], axis=1), False
        ),
    }


def scalar_stats(x: np.ndarray) -> dict[str, float | None]:
    return {
        name: (float(fn(x)) if len(x) else None)
        for name, fn in (
            ("mean", np.mean),
            ("median", np.median),
            ("p05", lambda a: np.quantile(a, 0.05)),
            ("p95", lambda a: np.quantile(a, 0.95)),
        )
    }


def z_stats(z: np.ndarray, zero: float) -> dict[str, Any]:
    return {
        "count": len(z),
        "origin_mean_z_mm": scalar_stats(z.mean(axis=1))
        if len(z)
        else scalar_stats(np.asarray([])),
        "five_origin_mean_z_mm": [float(v) if len(z) else None for v in z.mean(axis=0)]
        if len(z)
        else [None] * 5,
        "five_origin_z_mm": [scalar_stats(z[:, j]) for j in range(5)],
        "origin_min_z_mm": scalar_stats(z.min(axis=1))
        if len(z)
        else scalar_stats(np.asarray([])),
        "origin_max_z_mm": scalar_stats(z.max(axis=1))
        if len(z)
        else scalar_stats(np.asarray([])),
        "all_nonpositive_fraction": float(np.mean(np.all(z <= zero, axis=1)))
        if len(z)
        else None,
    }


def summarize_pairs(
    pairs: dict[str, np.ndarray],
    all_case_z: np.ndarray,
    all_control_z: np.ndarray,
    control_ticks: np.ndarray,
    control_z: np.ndarray,
    c: dict[str, Any],
    *,
    decision: bool,
) -> dict[str, Any]:
    """Summarize preserved pairs and unmatched cases without independent-sample claims."""
    matched = pairs["control_tick"] >= 0
    choices = pairs["control_tick"][matched]
    frequency = Counter(int(v) for v in choices)
    unique = np.array(sorted(frequency), dtype=np.int64)
    unique_z = control_z[np.isin(control_ticks, unique)]
    case_z = pairs["case_z_mm"][matched]
    paired_control_z = pairs["control_z_mm"][matched]
    count = len(pairs["case_tick"])
    coverage = len(choices) / count if count else None
    case_fraction = (
        float(np.mean(pairs["case_all_nonpositive"][matched])) if len(choices) else None
    )
    control_fraction = (
        float(np.mean(pairs["control_all_nonpositive"][matched]))
        if len(choices)
        else None
    )
    verdict = None
    if decision:
        verdict = "inconclusive"
        if (
            len(choices) >= c["minimum_matched_cases"]
            and len(unique) >= c["minimum_unique_controls"]
            and coverage is not None
            and coverage >= c["minimum_case_coverage"]
        ):
            if (
                control_fraction is not None
                and control_fraction >= c["shared_control_nonpositive_fraction"]
            ):
                verdict = "nonpositive_sign_shared"
            elif (
                case_fraction is not None
                and control_fraction is not None
                and case_fraction - control_fraction
                >= c["separated_fraction_difference"]
            ):
                verdict = "sign_separation_in_matched_subset"
    return {
        "eligible_case_count": count,
        "eligible_control_count": len(all_control_z),
        "matched_count": len(choices),
        "unmatched_count": int(count - len(choices)),
        "case_coverage": coverage,
        "unique_control_count": len(unique),
        "control_reuse_count_by_tick": {
            str(k): v for k, v in sorted(frequency.items())
        },
        "control_reuse_frequency": {
            str(k): v for k, v in sorted(Counter(frequency.values()).items())
        },
        "pre_match_case": z_stats(all_case_z, c["zero_mm"]),
        "pre_match_control": z_stats(all_control_z, c["zero_mm"]),
        "matched_case": z_stats(case_z, c["zero_mm"]),
        "paired_control": z_stats(paired_control_z, c["zero_mm"]),
        "unique_control": z_stats(unique_z, c["zero_mm"]),
        "paired_mean_z_difference_mm": scalar_stats(
            case_z.mean(axis=1) - paired_control_z.mean(axis=1)
        )
        if len(choices)
        else scalar_stats(np.asarray([])),
        "paired_five_origin_difference_mm": [
            scalar_stats(case_z[:, j] - paired_control_z[:, j]) for j in range(5)
        ],
        "paired_time_difference_s": scalar_stats(pairs["time_difference_s"][matched]),
        "matched_case_nonpositive_fraction": case_fraction,
        "paired_control_nonpositive_fraction": control_fraction,
        "decision": verdict,
    }


def output_manifest(out: Path) -> dict[str, Any]:
    return {
        str(p.relative_to(out)).replace("\\", "/"): {
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "output-manifest.json"
    }


def publish_success(
    out: Path,
    pending: dict[str, Any],
    limits: dict[str, Any],
    started: float,
    *,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Success receipt is written after resource/output records, then rechecked."""
    try:
        atomic_json(
            out / "timing.json",
            {"elapsed_s": clock() - started, "internal_limit_s": limits["internal_s"]},
        )
        manifest = output_manifest(out)
        atomic_json(out / "output-manifest.json", manifest)
        total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        require(total <= limits["raw_bytes"], "Final output byte limit exceeded")
        require(
            shutil.disk_usage(out).free >= limits["free_bytes"],
            "Final free-space reserve unavailable",
        )
        require(
            clock() - started <= limits["internal_s"], "Final internal time exceeded"
        )
        result = dict(
            pending,
            valid_diagnostic=True,
            publication_complete=True,
            resource_limit_met=True,
            output_bytes=total,
        )
        atomic_json(out / "result.json", result)
        require(
            clock() - started <= limits["internal_s"], "Post-publication time exceeded"
        )
        require(
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            <= limits["raw_bytes"],
            "Post-publication output byte limit exceeded",
        )
        require(
            shutil.disk_usage(out).free >= limits["free_bytes"],
            "Post-publication free-space reserve unavailable",
        )
        return result
    except BaseException:
        for name in ("result.json", "output-manifest.json"):
            path = out / name
            if path.exists():
                os.replace(path, out / (name + ".invalidated"))
        raise


def run(
    freeze_path: Path, freeze_sha: str, gate_path: Path, gate_sha: str, out: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    freeze_path, gate_path = freeze_path.resolve(), gate_path.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    partial: dict[str, np.ndarray] = {}
    approved: dict[str, str] | None = None
    hashes: dict[str, str] = {}
    freeze: dict[str, Any] | None = None
    error: BaseException | None = None
    secondary: list[str] = []
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        c = validate_freeze(freeze, freeze_path, freeze_sha)
        gate = validate_gate(gate_path, gate_sha, freeze_sha, freeze["design_sha256"])
        hashes = frozen_hashes(freeze)
        verify_hashes(hashes)
        validate_metadata(freeze)
        approved = snapshot(out, freeze, freeze_path, gate_path, gate_sha, gate)
        atomic_json(out / "source-snapshot.json", approved)
        direction_path = frozen_path(freeze["paths"]["direction"])
        phase_path = frozen_path(freeze["paths"]["phase"])
        with (
            np.load(direction_path, allow_pickle=False) as direction,
            np.load(phase_path, allow_pickle=False) as phase_source,
        ):
            partial = raw_inputs(direction, phase_source)
        atomic_npz(out / "raw-input-partial.npz", partial)
        partial = aligned_inputs(partial, c)
        partial = diagnostics(partial, c)
        atomic_npz(out / "aligned-partial.npz", partial)
        common = validate_inputs(partial, c)
        summaries: dict[str, Any] = {}
        pair_arrays: dict[str, np.ndarray] = {}
        count_arrays: dict[str, np.ndarray] = {}
        for li, leg in enumerate(LEGS):
            t = partial["time_s"]
            phase = partial["phase_cycle"][:, li]
            load = partial["normal_load_bw"][:, li]
            ap = partial["ego_ap_velocity_mm_s"][:, li]
            speed = partial["speed_mm_s"]
            eligible = (
                common
                & (load >= c["support_threshold_bw"])
                & (ap > c["ap_threshold_mm_s"])
            )
            case = eligible & (speed <= c["forward_threshold_mm_s"])
            control = eligible & (speed > c["forward_threshold_mm_s"])
            z = partial["effective_z_mm"][:, li]
            count_arrays[f"{leg}_eligible"] = eligible
            count_arrays[f"{leg}_case"] = case
            count_arrays[f"{leg}_control"] = control
            per_window = {}
            all_pairs = []
            for wi, (lo, hi) in enumerate(WINDOWS):
                window = (t >= lo) & (t < hi)
                p = match_leg(
                    partial["tick"],
                    t,
                    phase,
                    load,
                    ap,
                    z,
                    case & window,
                    control & window,
                    c,
                )
                all_pairs.append(p)
                for key, value in p.items():
                    pair_arrays[f"{leg}_w{wi}_{key}"] = value
                # Every completed window survives a later match/summary failure.
                atomic_npz(out / "pairs-partial.npz", pair_arrays)
            for wi, (lo, hi) in enumerate(WINDOWS):
                window = (t >= lo) & (t < hi)
                p = all_pairs[wi]
                per_window[f"{lo:g}-{hi:g}"] = summarize_pairs(
                    p,
                    z[case & window],
                    z[control & window],
                    partial["tick"][control & window],
                    z[control & window],
                    c,
                    decision=False,
                )
            joined = {
                key: np.concatenate([p[key] for p in all_pairs]) for key in all_pairs[0]
            }
            summaries[leg] = {
                "all": summarize_pairs(
                    joined,
                    z[case],
                    z[control],
                    partial["tick"][control],
                    z[control],
                    c,
                    decision=True,
                ),
                "windows": per_window,
                "exclusions": {
                    "outside_common": int((~common).sum()),
                    "phase_unavailable_outside_common": int(
                        (~partial["phase_available"][:, li]).sum()
                    ),
                    "noncontact_in_common": int(
                        (common & (load < c["support_threshold_bw"])).sum()
                    ),
                    "nonpositive_ap_after_contact": int(
                        (common & (load >= c["support_threshold_bw"]) & (ap <= 0)).sum()
                    ),
                },
            }
        atomic_npz(out / "selection-masks.npz", count_arrays)
        atomic_json(
            out / "summary.json",
            {
                "schema": SCHEMA + ".summary",
                "common_tick_count": int(common.sum()),
                "legs": summaries,
                "interpretation": "Instantaneous speed classes and matched association only; no natural-gait normal comparator, causal exhaustion inference, or F1-F8 acceptance.",
            },
        )
        verify_hashes(hashes)
        verify_snapshot(out, freeze, freeze_path, gate_path, approved)
        return publish_success(
            out,
            {
                "schema": SCHEMA + ".result",
                "freeze_sha256": freeze_sha,
                "source_review_sha256": gate_sha,
                "input_sha256": {name: hashes[name] for name in freeze["input_sha256"]},
                "original_processes": 1,
                "automatic_retries": 0,
                "candidate_adopted": False,
                "whole_goal_complete": False,
                "F6_exposed": False,
            },
            freeze["resource_limits"],
            started,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve interrupted publication evidence
        error = exc
    if hashes:
        try:
            verify_hashes(hashes)
        except BaseException as exc:  # noqa: BLE001 - retain primary error
            secondary.append("post_hashes: " + repr(exc))
    if approved is not None and freeze is not None:
        try:
            verify_snapshot(out, freeze, freeze_path, gate_path, approved)
        except BaseException as exc:  # noqa: BLE001 - retain primary error
            secondary.append("post_snapshot: " + repr(exc))
    try:
        atomic_json(
            out / "failure.json",
            {
                "schema": SCHEMA + ".failure",
                "primary_error": {"type": type(error).__name__, "message": str(error)},
                "secondary_errors": secondary,
                "elapsed_s": time.perf_counter() - started,
                "partial_keys": list(partial),
            },
        )
    except BaseException as exc:  # noqa: BLE001 - retain primary error
        secondary.append("failure_persistence: " + repr(exc))
    try:
        (out / "traceback.txt").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
    except BaseException as exc:  # noqa: BLE001 - retain primary error
        secondary.append("traceback_persistence: " + repr(exc))
    assert error is not None
    for note in secondary:
        error.add_note(note)
    raise error.with_traceback(error.__traceback__)


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
