"""E163 read-only six-leg phase, event, and signed-force diagnostics.

The quadratic peak formula is copied from ``probe_c_banc_m3_tonic_onset.py``.
Phase is a diagnostic label only; this module performs no model, neural,
physics, StateStore, CUDA, fitting, lag search, or candidate selection work.
"""

# Numerical thread limits must precede NumPy/SciPy imports.
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

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.signal import find_peaks
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    "tools/analyze_c_banc_m3_six_leg_timing.py",
    "tests/test_c_banc_six_leg_timing.py",
)
FREEZE_SHA = "3c103236e853dd5457cc599a5f77e4115340877f4fb6854d538e458dfe4f7af9"
DESIGN_SHA = "79044c6799c40aff6b71f00df451f4e25287357706049d9b0fe4acbbd052b4f0"
FREEZE_SCHEMA = "flylab.m3.e163.six-leg-timing.v1"
RESULT_SCHEMA = "flylab.m3.e163.six-leg-timing-result.v1"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
CORE_GROUPS = np.asarray((2, 5, 4, 1, 3, 0), dtype=np.int64)
E1_IDS = np.asarray((53861, 71439, 152280, 92648, 97727, 91167), dtype=np.int64)
SOURCE_COLUMNS = np.asarray((9, 12, 23, 16, 18, 15), dtype=np.int64)
LTM_GROUPS = np.asarray((15, 30, 46, 63, 78, 94), dtype=np.int64)
CONTROL_DT = 0.005
NATIVE_DT = 0.0001
NATIVE_PER_CONTROL = 50
CLOCK_ATOL = 2e-8
VALUE_ATOL = 1e-10
IMPULSE_ATOL = 1e-12
EVENT_RADIUS = 20
PHASE_WINDOWS = ((0.5, 2.5), (2.5, 5.0), (5.0, 10.0))
RAW_LIMIT = 64 * 1024 * 1024
FREE_LIMIT = 1024 * 1024 * 1024
INTERNAL_S = 55.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative: str) -> Path:
    item = Path(relative)
    resolved = (ROOT / item).resolve()
    require(not item.is_absolute() and ".." not in item.parts, "Unsafe frozen path")
    require(resolved.is_relative_to(ROOT), "Frozen path leaves repository")
    return resolved


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all(), f"Invalid {label}")
    return array


def verify_hash_map(entries: Mapping[str, str], label: str) -> None:
    require(type(entries) is dict and entries, f"Missing {label} hash inventory")
    for relative, expected in entries.items():
        require(
            type(expected) is str and len(expected) == 64, f"Malformed {label} hash"
        )
        require(
            sha256(safe_path(relative)) == expected, f"Frozen hash mismatch: {relative}"
        )


def frozen_hashes(freeze: Mapping[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for name in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        for relative, digest in freeze[name].items():
            require(
                relative not in merged or merged[relative] == digest,
                f"Conflicting frozen digest: {relative}",
            )
            merged[relative] = digest
    return merged


def verified_inputs(
    freeze_path: Path,
    freeze_sha256: str,
    source_review_path: Path,
    source_review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Contract/hash-only preflight.  No NPZ source is opened here."""
    require(
        freeze_sha256 == FREEZE_SHA and sha256(freeze_path) == FREEZE_SHA,
        "Exact E163 freeze SHA required",
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(freeze.get("schema") == FREEZE_SCHEMA, "Unexpected E163 freeze schema")
    require(
        freeze.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(freeze["design"])) == DESIGN_SHA,
        "E163 design changed",
    )
    require(freeze.get("owned_files") == list(OWNED), "E163 owned files changed")
    contract = freeze.get("sample_contract", {})
    exact = {
        "legs": list(LEGS),
        "core_groups_by_leg": CORE_GROUPS.tolist(),
        "e1_ids_by_leg": E1_IDS.tolist(),
        "source_columns_by_leg": SOURCE_COLUMNS.tolist(),
        "control_rows": 2001,
        "signal_rows": 2000,
        "observed_cells": 446,
        "motor_cells": 391,
        "muscles": 96,
        "joint_targets": 42,
        "ltm_groups": LTM_GROUPS.tolist(),
        "control_dt": CONTROL_DT,
        "native_dt": NATIVE_DT,
        "native_samples": 10000,
        "native_per_control": NATIVE_PER_CONTROL,
        "first_native_time_s": 1.0001,
        "native_first_control": 201,
        "native_last_control": 400,
        "actuator_ids": list(range(8, 50)),
        "clock_atol": CLOCK_ATOL,
        "value_atol": VALUE_ATOL,
        "impulse_atol": IMPULSE_ATOL,
        "phase_windows": [list(x) for x in PHASE_WINDOWS],
        "phase_query_dt": 0.025,
        "peak_prominence": 20.0,
        "phase_bins": 12,
        "phase_stability_min_R": 0.8,
        "phase_stability_max_drift": 0.15,
        "analysis_window": [0.5, 10.0],
        "speed_loss_threshold": 0.5,
        "event_radius_controls": 20,
        "expected_existing_events": 46,
        "pre_offsets_inclusive": [-19, 0],
        "post_offsets_inclusive": [1, 20],
        "support_threshold_bw": 0.02,
        "command_gain": 0.02,
        "world_vertical_axis": 2,
    }
    for key, value in exact.items():
        require(contract.get(key) == value, f"E163 contract differs: {key}")
    require(
        contract.get("group_legs")
        == [leg for leg in range(6) for _ in range((17, 15, 16, 17, 15, 16)[leg])],
        "E163 muscle ownership differs",
    )
    counts = freeze.get("counts", {})
    require(
        counts
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_models": 0,
            "state_store_instances": 0,
            "physics_steps": 0,
            "neural_steps": 0,
            "fit_calls": 0,
            "candidate_selections": 0,
            "F6_exposed": False,
        },
        "E163 execution boundary changed",
    )
    require(
        freeze.get("resource_limits")
        == {
            "internal_s": 55,
            "external_s": 70,
            "supervisor_s": 80,
            "raw_bytes": RAW_LIMIT,
            "free_bytes": FREE_LIMIT,
            "cpu_threads": 1,
            "gpu_count": 0,
        },
        "E163 resources changed",
    )
    require(
        freeze.get("diagnostic_only") is True
        and freeze.get("new_timing_metrics_seen_before_freeze") is False,
        "E163 diagnostic boundary changed",
    )
    for name, count in (
        ("input_sha256", 224),
        ("reused_source_sha256", 31),
        ("preservation_sha256", 3175),
    ):
        require(len(freeze[name]) == count, f"E163 {name} count differs")
    verify_hash_map(frozen_hashes(freeze), "pre-frozen")
    require(
        sha256(source_review_path) == source_review_sha256,
        "Exact E163 source-review SHA required",
    )
    review = json.loads(source_review_path.read_text(encoding="utf-8"))
    require(
        review.get("freeze_sha256") == FREEZE_SHA
        and review.get("blocking_defects") == []
        and review.get("accepts_single_saved_diagnostic") is True
        and review.get("original_processes_allowed") == 1,
        "Parent did not approve one E163 saved diagnostic",
    )
    require(
        review.get("source_sha256") == {name: sha256(ROOT / name) for name in OWNED},
        "E163 source-review source map differs",
    )
    return freeze, review


def quadratic_peaks(times: Any, values: Any, *, prominence: float = 20.0) -> np.ndarray:
    """Three-point refinement copied from the frozen tonic-onset source."""
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    require(
        t.ndim == y.ndim == 1
        and t.shape == y.shape
        and len(t) >= 3
        and np.isfinite(t).all()
        and np.isfinite(y).all(),
        "Finite peak trace required",
    )
    points, _ = find_peaks(y, prominence=prominence)
    refined = []
    for index in points:
        if index == 0 or index + 1 == len(y):
            continue
        before, center, after = y[index - 1 : index + 2]
        denominator = before - 2 * center + after
        offset = 0.5 * (before - after) / denominator if denominator else 0.0
        require(
            np.isfinite(offset) and abs(offset) <= 1,
            "Invalid quadratic E1 peak refinement",
        )
        refined.append(float(t[index] + offset * (t[index + 1] - t[index])))
    return np.asarray(refined, dtype=np.float64)


def phase_trace(times: Any, peaks: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(peaks, dtype=np.float64)
    require(
        t.ndim == p.ndim == 1
        and np.isfinite(t).all()
        and np.isfinite(p).all()
        and np.all(np.diff(p) > 0),
        "Ordered phase inputs required",
    )
    unwrapped = np.full(t.shape, np.nan)
    if len(p) >= 2:
        unwrapped = np.interp(
            t, p, np.arange(len(p), dtype=float), left=np.nan, right=np.nan
        )
    available = np.isfinite(unwrapped)
    phase = np.full(t.shape, np.nan)
    phase[available] = np.mod(unwrapped[available], 1.0)
    return phase, available


def circular_summary(values: Any, harmonic: int = 1) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan, 0
    z = np.mean(np.exp(2j * np.pi * harmonic * array))
    mean = (np.angle(z) / (2 * np.pi * harmonic)) % (1.0 / harmonic)
    return float(mean), float(abs(z)), len(array)


def pair_phase_statistics(
    phase_at_query: np.ndarray, query_times: np.ndarray
) -> dict[str, np.ndarray]:
    phase = np.asarray(phase_at_query, dtype=np.float64)
    require(phase.shape == (len(query_times), 6), "Six-leg query phase required")
    common_bracket = np.isfinite(phase).all(axis=1)
    pairs = np.asarray(
        [(a, b) for a in range(6) for b in range(a + 1, 6)], dtype=np.int64
    )
    means = np.full((15, 3), np.nan)
    resultant = np.full((15, 3), np.nan)
    counts = np.zeros((15, 3), dtype=np.int64)
    for pair_index, (first, second) in enumerate(pairs):
        delta = np.mod(phase[:, first] - phase[:, second], 1.0)
        for window_index, (start, stop) in enumerate(PHASE_WINDOWS):
            selected = (query_times >= start) & (query_times < stop) & common_bracket
            (
                means[pair_index, window_index],
                resultant[pair_index, window_index],
                counts[pair_index, window_index],
            ) = circular_summary(delta[selected])
    drift = np.full(15, np.nan)
    stable = np.zeros(15, dtype=bool)
    for pair_index in range(15):
        if np.isfinite(means[pair_index]).all():
            distances = [
                abs((means[pair_index, a] - means[pair_index, b] + 0.5) % 1 - 0.5)
                for a in range(3)
                for b in range(a + 1, 3)
            ]
            drift[pair_index] = max(distances)
            stable[pair_index] = bool(
                np.all(resultant[pair_index] >= 0.8) and drift[pair_index] <= 0.15
            )
    return {
        "pair_indices": pairs,
        "pair_mean": means,
        "pair_R": resultant,
        "pair_count": counts,
        "pair_common_bracket": common_bracket,
        "pair_max_drift": drift,
        "pair_time_consistent": stable,
    }


def interruption_events(
    speed: Any, times: Any, window: Any | None = None
) -> dict[str, np.ndarray]:
    speed_array = np.asarray(speed, dtype=np.float64)
    time_array = np.asarray(times, dtype=np.float64)
    require(
        speed_array.shape == time_array.shape
        and speed_array.ndim == 1
        and np.isfinite(speed_array).all()
        and np.isfinite(time_array).all(),
        "Finite speed clock required",
    )
    inside = (
        (time_array >= 0.5) & (time_array < 10.0)
        if window is None
        else np.asarray(window)
    )
    require(
        inside.dtype == np.dtype(bool) and inside.shape == speed_array.shape,
        "Boolean interruption window required",
    )
    raw = (
        np.flatnonzero(
            inside[1:]
            & inside[:-1]
            & (speed_array[:-1] > 0.5)
            & (speed_array[1:] <= 0.5)
        )
        + 1
    )
    included = []
    reasons = []
    for index in raw:
        before = index - EVENT_RADIUS
        after = index + EVENT_RADIUS
        reason = "included"
        if before < 0 or not np.all(inside[max(0, before) : index + 1]):
            reason = "pre_outside_window"
        elif after >= len(speed_array) or not np.all(
            inside[index : min(len(speed_array), after + 1)]
        ):
            reason = "post_outside_window"
        else:
            included.append(int(index))
        reasons.append(reason)
    included = np.asarray(included, dtype=np.int64)
    offsets = np.arange(-EVENT_RADIUS, EVENT_RADIUS + 1, dtype=np.int64)
    return {
        "crossing_index": raw.astype(np.int64),
        "crossing_exclusion_reason": np.asarray(reasons, dtype="U32"),
        "event_index": included.astype(np.int64),
        "event_offset": offsets,
        "event_grid": included[:, None] + offsets
        if len(included)
        else np.empty((0, 41), np.int64),
    }


def stack_at_events(values: Any, grid: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return (
        array[grid]
        if len(grid)
        else np.empty((0, grid.shape[1]) + array.shape[1:], dtype=array.dtype)
    )


def harmonic_event_summary(
    event_phase: np.ndarray, event_times: np.ndarray
) -> dict[str, np.ndarray]:
    phase = np.asarray(event_phase, dtype=np.float64)
    require(phase.ndim == 2 and phase.shape[1] == 6, "Event phase shape differs")
    means = np.full((4, 2, 6), np.nan)
    resultant = np.full((4, 2, 6), np.nan)
    counts = np.zeros((4, 2, 6), dtype=np.int64)
    masks = [np.ones(len(event_times), dtype=bool)] + [
        (event_times >= start) & (event_times < stop) for start, stop in PHASE_WINDOWS
    ]
    for window, selected in enumerate(masks):
        for harmonic in (1, 2):
            for leg in range(6):
                (
                    means[window, harmonic - 1, leg],
                    resultant[window, harmonic - 1, leg],
                    counts[window, harmonic - 1, leg],
                ) = circular_summary(phase[selected, leg], harmonic)
    return {
        "event_phase_harmonic_mean": means,
        "event_phase_harmonic_R": resultant,
        "event_phase_harmonic_count": counts,
    }


def native_force_blocks(force_forward: Any, normal_load: Any) -> dict[str, np.ndarray]:
    force = finite(force_forward, (10000, 6), "native forward force")
    load = finite(normal_load, (10000, 6), "native load")
    require(np.all(load >= 0), "Native load must be nonnegative")
    blocks = force.reshape(200, 50, 6)
    loads = load.reshape(200, 50, 6)
    mean = blocks.mean(axis=1)
    positive = np.maximum(blocks, 0).sum(axis=1) * NATIVE_DT
    negative = np.maximum(-blocks, 0).sum(axis=1) * NATIVE_DT
    closure = positive - negative - CONTROL_DT * mean
    return {
        "block_mean_force": mean,
        "block_positive_impulse": positive,
        "block_negative_impulse": negative,
        "block_mean_load": loads.mean(axis=1),
        "block_support_fraction": (loads >= 0.02).mean(axis=1),
        "block_impulse_closure": closure,
    }


def force_event_records(crossings: Any) -> dict[str, np.ndarray]:
    crossing = np.asarray(crossings, dtype=np.int64)
    included, reasons = [], []
    for tick in crossing:
        if tick - 20 < 201:
            reasons.append("pre_outside_native_1_2s")
        elif tick + 20 > 400:
            reasons.append("post_outside_native_1_2s")
        else:
            reasons.append("included")
            included.append(int(tick))
    ticks = np.asarray(included, dtype=np.int64)
    offsets = np.arange(-20, 21, dtype=np.int64)
    grid = (
        ticks[:, None] + offsets[None, :] - 201
        if len(ticks)
        else np.empty((0, 41), np.int64)
    )
    return {
        "force_crossing_tick": crossing,
        "force_event_tick": ticks,
        "force_event_offset": offsets,
        "force_event_block_index": grid,
        "force_event_exclusion_reason": np.asarray(reasons, dtype="U32"),
    }


def force_pre_post(
    positive: np.ndarray, negative: np.ndarray, grid: np.ndarray
) -> dict[str, np.ndarray]:
    pos = stack_at_events(positive, grid)
    neg = stack_at_events(negative, grid)
    # grid offsets are -20..20: pre is positions 1..20 (-19..0), post 21..40 (1..20).
    pre_pos, post_pos = pos[:, 1:21].sum(axis=1), pos[:, 21:41].sum(axis=1)
    pre_neg, post_neg = neg[:, 1:21].sum(axis=1), neg[:, 21:41].sum(axis=1)

    def two_stats(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = value.shape[1:]
        if len(value):
            return value.mean(axis=0), np.median(value, axis=0)
        return np.full(shape, np.nan), np.full(shape, np.nan)

    out = {
        "event_pre_positive": pre_pos,
        "event_post_positive": post_pos,
        "event_pre_negative": pre_neg,
        "event_post_negative": post_neg,
    }
    for name, value in tuple(out.items()):
        out[name + "_mean"], out[name + "_median"] = two_stats(value)
    return out


def phase_bin_statistics(
    phase: Any, values: Any, selected: Any, bins: int = 12
) -> dict[str, np.ndarray]:
    phase_array = np.asarray(phase, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    mask = np.asarray(selected, dtype=bool) & np.isfinite(phase_array)
    require(
        value.shape[0] == len(phase_array) and mask.shape == phase_array.shape,
        "Phase-bin shape differs",
    )
    count = np.zeros(bins, dtype=np.int64)
    mean = np.full((bins,) + value.shape[1:], np.nan)
    index = np.zeros(len(phase_array), dtype=np.int64)
    index[mask] = np.floor(np.mod(phase_array[mask], 1.0) * bins).astype(np.int64)
    for bin_index in range(bins):
        rows = mask & (index == bin_index)
        count[bin_index] = rows.sum()
        if count[bin_index]:
            mean[bin_index] = value[rows].mean(axis=0)
    return {"count": count, "available": count > 0, "mean": mean}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _path(freeze: Mapping[str, Any], key: str) -> Path:
    return safe_path(freeze["paths"][key])


def analyze_loaded(
    data: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    signals, metadata, calibration = (
        data["signals"],
        data["metadata"],
        data["calibration"],
    )
    recruitment, adapter, features, native = (
        data["recruitment"],
        data["adapter"],
        data["features"],
        data["native"],
    )
    trace, groups, old_events, layout = (
        data["trace"],
        data["groups"],
        data["old_events"],
        data["layout"],
    )
    signal_time = finite(signals["time_s"], (2000,), "signal time")
    require(
        np.allclose(
            signal_time, np.arange(1, 2001) * CONTROL_DT, atol=CLOCK_ATOL, rtol=0
        ),
        "Signal clock differs",
    )
    observed_ids = np.asarray(signals["observed_ids"], dtype=np.int64)
    require(
        observed_ids.shape == (446,)
        and len(np.unique(observed_ids)) == 446
        and np.array_equal(observed_ids, metadata["observed_ids"]),
        "Observed IDs differ",
    )
    e1_columns = np.asarray(
        [np.flatnonzero(observed_ids == value)[0] for value in E1_IDS]
    )
    core_ids = np.asarray(calibration["core_ids"], dtype=np.int64)
    core_groups = np.asarray(calibration["core_groups"], dtype=np.int64)
    core_types = np.asarray(calibration["core_cell_types"], dtype=np.int64)
    for leg, cell in enumerate(E1_IDS):
        match = np.flatnonzero(core_ids == cell)
        require(
            match.shape == (1,)
            and core_types[match[0]] == 0
            and core_groups[match[0]] == CORE_GROUPS[leg],
            "E1 core mapping differs",
        )
    require(
        type(groups) is list
        and len(groups) == 96
        and [group.get("leg") for group in groups]
        == [LEGS[x] for x in data["group_legs"]],
        "Static muscle ownership differs",
    )
    membership = np.asarray(adapter["membership"], dtype=np.int64)
    pool = finite(adapter["pool"], (96, 391), "pool")
    require(
        membership.shape == (391,)
        and np.all((membership >= 0) & (membership < 96))
        and np.all(pool >= 0)
        and np.array_equal(np.argmax(pool, axis=0), membership)
        and np.array_equal((pool > 0).sum(axis=0), np.ones(391, dtype=np.int64))
        and np.allclose(pool.sum(axis=1), 1, atol=1e-12, rtol=0),
        "MN pool mapping differs",
    )
    for muscle, group in enumerate(groups):
        require(
            np.array_equal(
                np.asarray(group["columns"], dtype=np.int64),
                np.flatnonzero(membership == muscle),
            ),
            "Static group columns differ",
        )
    motor_ids = np.asarray(adapter["motor_ids"], dtype=np.int64)
    require(
        motor_ids.shape == (391,)
        and len(np.unique(motor_ids)) == 391
        and np.array_equal(motor_ids, metadata["motor_ids"])
        and np.array_equal(motor_ids, calibration["motor_ids"]),
        "Motor IDs differ",
    )
    for key in ("moment", "rest", "pool", "membership", "ltm"):
        require(
            np.array_equal(np.asarray(adapter[key]), np.asarray(calibration[key])),
            f"Adapter/calibration mapping differs: {key}",
        )
    require(
        np.array_equal(np.asarray(adapter["ltm"], dtype=np.int64), LTM_GROUPS),
        "LTM mapping differs",
    )
    source_ids = np.asarray(metadata["source_ids"], dtype=np.int64)
    require(
        np.array_equal(source_ids[SOURCE_COLUMNS], E1_IDS)
        and np.array_equal(e1_columns, SOURCE_COLUMNS),
        "E1 source columns differ",
    )
    motor_columns = np.asarray(
        [np.flatnonzero(observed_ids == value)[0] for value in motor_ids]
    )
    rate = finite(signals["rate"], (2000, 446), "observed rate")
    motor_rate = finite(recruitment["motor_rate"], (2001, 391), "motor rate")
    activation = finite(recruitment["activation"], (2001, 96), "activation")
    signal_activation = finite(
        signals["muscle_activation"], (2000, 96), "signal activation"
    )
    control_tick = np.asarray(recruitment["control_tick"], dtype=np.int64)
    network_tick = np.asarray(recruitment["network_tick"], dtype=np.int64)
    control_times = np.arange(2001) * CONTROL_DT
    physics_time = finite(recruitment["physics_time_s"], (2001,), "physics time")
    require(
        np.array_equal(control_tick, np.arange(2001))
        and np.array_equal(network_tick, control_tick * 20)
        and np.allclose(physics_time, control_times, atol=CLOCK_ATOL, rtol=0),
        "Recruitment clock differs",
    )
    for key in ("complete", "activation_available", "motor_rate_available"):
        flag = np.asarray(recruitment[key])
        require(
            flag.dtype == np.dtype(bool) and flag.shape == (2001,) and flag.all(),
            f"Recruitment availability differs: {key}",
        )
    e1_rate = rate[:, e1_columns]
    peaks = [quadratic_peaks(signal_time, e1_rate[:, leg]) for leg in range(6)]
    phase = np.full((2001, 6), np.nan)
    phase_available = np.zeros((2001, 6), bool)
    for leg in range(6):
        phase[:, leg], phase_available[:, leg] = phase_trace(control_times, peaks[leg])
    query_times = np.arange(0.5, 10.0, 0.025)
    query_phase = np.column_stack([phase_trace(query_times, peak)[0] for peak in peaks])
    pair_stats = pair_phase_statistics(query_phase, query_times)
    feature_time = finite(features["time_s"], (2001,), "feature time")
    require(
        np.allclose(feature_time, control_times, atol=CLOCK_ATOL, rtol=0),
        "Feature clock differs",
    )
    speed = finite(features["speed_mm_s"], (2001,), "speed")
    window = np.asarray(features["window"])
    require(
        window.dtype == np.dtype(bool) and window.shape == (2001,),
        "E152 analysis window differs",
    )
    events = interruption_events(speed, feature_time, window)
    old_event_index = np.asarray(old_events["event_index"], dtype=np.int64)
    old_event_offset = np.asarray(old_events["offset_control"], dtype=np.int64)
    old_event_offset_s = finite(old_events["offset_s"], (41,), "E152 event offset")
    require(
        old_event_index.shape == (46,)
        and old_event_offset.shape == (41,)
        and np.array_equal(events["event_index"], old_event_index)
        and np.array_equal(events["event_offset"], old_event_offset)
        and np.allclose(
            old_event_offset_s,
            events["event_offset"] * CONTROL_DT,
            atol=CLOCK_ATOL,
            rtol=0,
        )
        and len(events["event_index"]) == 46,
        "E152 event reconstruction differs",
    )
    rows = trace.get("samples")
    require(
        trace.get("body_hash") == data["body_hash"]
        and type(rows) is list
        and len(rows) == 2001
        and np.array_equal([row["control_tick"] for row in rows], np.arange(2001)),
        "Trace rows/ticks differ",
    )
    targets = finite([row["joint_targets_rad"] for row in rows], (2001, 42), "targets")
    trace_time = finite([row["time_s"] for row in rows], (2001,), "trace time")
    require(
        np.allclose(trace_time, control_times, atol=CLOCK_ATOL, rtol=0),
        "Trace clock differs",
    )
    rest = finite(adapter["rest"], (42,), "rest")
    moment = finite(adapter["moment"], (42, 96), "moment")
    reconstructed = rest[None, :] + 0.02 * activation[1:] @ moment.T
    target_derivative = np.full_like(targets, np.nan)
    target_derivative[1:] = np.diff(targets, axis=0) / CONTROL_DT
    pooled_rate = motor_rate @ pool.T
    event_grid = events["event_grid"]
    event_phase = phase[events["event_index"]]
    harmonic = harmonic_event_summary(event_phase, feature_time[events["event_index"]])
    arrays: dict[str, np.ndarray] = {
        "signal_time_s": signal_time,
        "e1_columns": e1_columns,
        "e1_rate": e1_rate,
        "control_time_s": control_times,
        "e1_phase": phase,
        "e1_phase_available": phase_available,
        "phase_query_time_s": query_times,
        "phase_query": query_phase,
        **pair_stats,
        **events,
        **harmonic,
        "event_time_s": feature_time[events["event_index"]],
        "event_e1_phase": event_phase,
        "event_e1_phase_available": phase_available[events["event_index"]],
        "event_e1_phase_window": stack_at_events(phase, event_grid),
        "event_e1_phase_available_window": stack_at_events(phase_available, event_grid),
        "event_mn_pool_rate": stack_at_events(pooled_rate, event_grid),
        "event_activation": stack_at_events(activation, event_grid),
        "event_joint_target": stack_at_events(targets, event_grid),
        "event_joint_target_derivative": stack_at_events(target_derivative, event_grid),
        "event_source_e1_rate": stack_at_events(
            np.vstack((np.full((1, 6), np.nan), e1_rate)), event_grid
        ),
        "event_speed_mm_s": stack_at_events(speed, event_grid),
        "event_boundary_load_bw": stack_at_events(
            features["boundary_load_bw"], event_grid
        ),
        "comparison_signal_motor_rate": rate[:, motor_columns],
        "comparison_saved_motor_rate": motor_rate[1:],
        "residual_motor_rate": rate[:, motor_columns] - motor_rate[1:],
        "comparison_signal_activation": signal_activation,
        "comparison_saved_activation": activation[1:],
        "residual_activation": signal_activation - activation[1:],
        "comparison_recorded_target": targets[1:],
        "comparison_reconstructed_target": reconstructed,
        "residual_target": targets[1:] - reconstructed,
    }
    peak_offsets = np.r_[0, np.cumsum([len(value) for value in peaks])]
    arrays["e1_peak_offsets"] = peak_offsets
    arrays["e1_peak_time_s"] = (
        np.concatenate(peaks) if peak_offsets[-1] else np.empty(0)
    )
    native_index = np.asarray(native["native_index"], dtype=np.int64)
    native_time = finite(native["time_s"], (10000,), "native time")
    native_tick = np.asarray(native["control_tick"], dtype=np.int64)
    require(
        np.array_equal(native_index, np.arange(10000))
        and np.allclose(
            native_time, 1.0001 + np.arange(10000) * NATIVE_DT, atol=CLOCK_ATOL, rtol=0
        )
        and np.array_equal(native_tick, 201 + np.arange(10000) // 50),
        "Native clock/index differs",
    )
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
    ):
        receipt = np.asarray(native[key])
        require(
            receipt.dtype == np.dtype(bool)
            and receipt.shape == (10000,)
            and receipt.all(),
            f"Native receipt differs: {key}",
        )
    force_world = finite(native["force_world_bw"], (10000, 6, 3), "world force")
    heading = finite(native["heading_horizontal"], (10000, 3), "heading")
    forward = finite(native["force_forward_bw"], (10000, 6), "forward force")
    projected_forward = np.einsum("nli,ni->nl", force_world, heading)
    force_blocks = native_force_blocks(forward, native["normal_load_bw"])
    require(
        layout.get("actuator_ids") == list(range(8, 50))
        and layout.get("body_hash") == data["body_hash"],
        "Native layout differs",
    )
    native_ctrl = finite(native["ctrl"], (10000, 82), "native ctrl")[:, 8:50]
    arrays.update(force_blocks)
    arrays.update(
        {
            "comparison_recorded_forward_force": forward,
            "comparison_projected_forward_force": projected_forward,
            "residual_force_projection": projected_forward - forward,
            "comparison_native_ctrl": native_ctrl,
            "comparison_target_at_native_tick": targets[native_tick],
            "residual_native_ctrl_target": native_ctrl - targets[native_tick],
            "block_control_tick": np.arange(201, 401, dtype=np.int64),
            "block_endpoint_time_s": np.arange(201, 401) * CONTROL_DT,
        }
    )
    force_events = force_event_records(events["crossing_index"])
    arrays.update(force_events)
    fg = force_events["force_event_block_index"]
    for name in (
        "block_mean_force",
        "block_positive_impulse",
        "block_negative_impulse",
        "block_mean_load",
        "block_support_fraction",
    ):
        arrays["force_event_" + name.removeprefix("block_")] = stack_at_events(
            force_blocks[name], fg
        )
    force_ticks = force_events["force_event_tick"]
    arrays["force_event_activation"] = stack_at_events(activation[201:401], fg)
    arrays["force_event_joint_target"] = stack_at_events(targets[201:401], fg)
    arrays["force_event_e1_phase"] = stack_at_events(phase[201:401], fg)
    arrays["force_event_e1_phase_available"] = stack_at_events(
        phase_available[201:401], fg
    )
    arrays.update(
        force_pre_post(
            force_blocks["block_positive_impulse"],
            force_blocks["block_negative_impulse"],
            fg,
        )
    )
    # Fixed phase bins: neural/command across three windows, force only 1-2 s.
    group_legs = np.asarray(data["group_legs"], dtype=np.int64)
    for window_index, (start, stop) in enumerate(PHASE_WINDOWS):
        selected = (control_times >= start) & (control_times < stop)
        for leg in range(6):
            muscles = group_legs == leg
            for label, values in (
                ("pool_rate", pooled_rate[:, muscles]),
                ("activation", activation[:, muscles]),
                ("target", targets[:, 7 * leg : 7 * leg + 7]),
            ):
                stats = phase_bin_statistics(phase[:, leg], values, selected)
                for key, value in stats.items():
                    arrays[f"phasebin_w{window_index}_{LEGS[leg]}_{label}_{key}"] = (
                        value
                    )
    block_ticks = np.arange(201, 401)
    for leg in range(6):
        phase_endpoint = phase[block_ticks, leg]
        stats = phase_bin_statistics(
            phase_endpoint,
            force_blocks["block_mean_force"][:, leg],
            np.ones(200, dtype=bool),
        )
        for key, value in stats.items():
            arrays[f"forcebin_{LEGS[leg]}_mean_force_{key}"] = value
        available = np.isfinite(phase_endpoint)
        index = np.zeros(200, dtype=np.int64)
        index[available] = np.floor(np.mod(phase_endpoint[available], 1) * 12).astype(
            np.int64
        )
        for label in ("positive", "negative"):
            source = force_blocks[f"block_{label}_impulse"][:, leg]
            sums = np.full(12, np.nan)
            counts = np.zeros(12, np.int64)
            for bin_index in range(12):
                selected = available & (index == bin_index)
                counts[bin_index] = selected.sum()
                if counts[bin_index]:
                    sums[bin_index] = source[selected].sum()
            arrays[f"forcebin_{LEGS[leg]}_{label}_impulse_sum"] = sums
            arrays[f"forcebin_{LEGS[leg]}_{label}_count"] = counts
            arrays[f"forcebin_{LEGS[leg]}_{label}_available"] = counts > 0
    summary = {
        "pair_time_consistent_count": int(pair_stats["pair_time_consistent"].sum()),
        "interruption_event_count": len(events["event_index"]),
        "force_event_count": len(force_ticks),
        "harmonic_mean_semantics": "harmonic h mean phase is modulo 1/h; R is the concentration strength",
        "overlapping_events_are_independent": False,
        "interpretation": "Diagnostic phase/event/force timing only; no causal, biological tripod, F4, gait-quality, or intervention-effect validation.",
    }
    return arrays, summary


def validate_numeric_contract(arrays: Mapping[str, np.ndarray]) -> None:
    """Enforce tolerances only after the caller has persisted raw comparisons."""
    checks = (
        ("residual_motor_rate", VALUE_ATOL, "Signal/recruitment motor rates differ"),
        ("residual_activation", VALUE_ATOL, "Signal activation differs"),
        ("residual_target", VALUE_ATOL, "Target reconstruction differs"),
        (
            "residual_force_projection",
            VALUE_ATOL,
            "World-force heading projection differs",
        ),
        (
            "residual_native_ctrl_target",
            VALUE_ATOL,
            "Native ctrl/target mapping differs",
        ),
        ("block_impulse_closure", IMPULSE_ATOL, "Signed impulse closure differs"),
    )
    for key, tolerance, message in checks:
        residual = np.asarray(arrays[key], dtype=np.float64)
        require(residual.size > 0 and np.isfinite(residual).all(), f"Invalid {key}")
        require(np.max(np.abs(residual)) <= tolerance, message)


def finalize(
    output: Path, result: dict[str, Any], ledger: dict[str, Any], started: float
) -> dict[str, Any]:
    errors: list[dict[str, str]] = list(result.get("publication_errors", []))

    def fail(stage: str, exc: BaseException) -> None:
        errors.append(
            {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}
        )
        result.update(
            {
                "accepted": False,
                "publication_complete": False,
                "diagnostic_valid": False,
                "candidate_adopted": False,
                "whole_goal_complete": False,
                "F6_exposed": False,
                "F1_F8_complete": False,
            }
        )

    def manifest_value() -> dict[str, dict[str, Any]]:
        return {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "output-manifest.json"
        }

    def quarantine(path: Path) -> None:
        if path.exists():
            invalid = path.with_name(
                f"{path.stem}.invalid-{os.getpid()}-{time.time_ns()}{path.suffix}"
            )
            path.replace(invalid)

    def update_accounting() -> None:
        result["wall_s"] = time.perf_counter() - started
        if result["wall_s"] > INTERNAL_S:
            fail("deadline", TimeoutError("E163 internal deadline exceeded"))
        total = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        result["total_output_bytes"] = total
        if total > RAW_LIMIT:
            fail("output_limit", ValueError("E163 total output exceeds 64 MiB"))

    def sync_error_views() -> None:
        result["publication_errors"] = errors
        ledger["publication_errors"] = errors

    pools = threadpool_info()
    result["threadpools"] = pools
    if any(int(pool.get("num_threads", 0)) != 1 for pool in pools):
        fail("thread_limit", ValueError("Numerical threadpool count differs from one"))
    update_accounting()
    sync_error_views()
    try:
        atomic_json(output / "ledger.json", ledger)
        atomic_json(output / "result.json", result)
        atomic_json(output / "output-manifest.json", manifest_value())
    except Exception as exc:  # noqa: BLE001
        fail("initial_publication", exc)

    # Re-evaluate resource limits after all costly work and publish the terminal
    # result before the terminal manifest so every live manifest hash is final.
    update_accounting()
    sync_error_views()
    try:
        atomic_json(output / "ledger.json", ledger)
        atomic_json(output / "result.json", result)
        atomic_json(output / "output-manifest.json", manifest_value())
    except Exception as exc:  # noqa: BLE001
        fail("final_publication", exc)

    # Publication itself consumes time and bytes. A late overrun must enter the
    # same revocation path as an I/O failure, even if both writes succeeded.
    update_accounting()
    if errors:
        # Never leave a live success receipt or a success-era manifest after a
        # late failure.  Renamed files retain partial/stale publication evidence.
        try:
            receipt = output / "result.json"
            if receipt.exists():
                try:
                    persisted = json.loads(receipt.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    persisted = {"accepted": True}
                if persisted.get("accepted") is True:
                    quarantine(receipt)
            quarantine(output / "output-manifest.json")
        except Exception as exc:  # noqa: BLE001
            fail("revoke_success", exc)
        update_accounting()
        sync_error_views()
        for stage, path, value in (
            ("failure_ledger", output / "ledger.json", ledger),
            ("failure_result", output / "result.json", result),
        ):
            try:
                atomic_json(path, value)
            except Exception as exc:  # noqa: BLE001
                fail(stage, exc)
                if path.name == "result.json":
                    try:
                        quarantine(path)
                    except Exception as revoke_exc:  # noqa: BLE001
                        fail("failure_result_revoke", revoke_exc)
        sync_error_views()
        try:
            atomic_json(output / "output-manifest.json", manifest_value())
        except Exception as exc:  # noqa: BLE001
            fail("failure_manifest", exc)
            try:
                quarantine(output / "output-manifest.json")
            except Exception as revoke_exc:  # noqa: BLE001
                fail("failure_manifest_revoke", revoke_exc)
        # Best-effort terminal error synchronization. Persistent failures remain
        # explicit in the returned object and cannot leave accepted=true live.
        update_accounting()
        sync_error_views()
        try:
            atomic_json(output / "ledger.json", ledger)
        except Exception as exc:  # noqa: BLE001
            fail("terminal_failure_ledger", exc)
        try:
            atomic_json(output / "result.json", result)
        except Exception as exc:  # noqa: BLE001
            fail("terminal_failure_result", exc)
            try:
                quarantine(output / "result.json")
            except Exception as revoke_exc:  # noqa: BLE001
                fail("terminal_failure_result_revoke", revoke_exc)
        try:
            atomic_json(output / "output-manifest.json", manifest_value())
        except Exception as exc:  # noqa: BLE001
            fail("terminal_failure_manifest", exc)
            try:
                quarantine(output / "output-manifest.json")
            except Exception as revoke_exc:  # noqa: BLE001
                fail("terminal_failure_manifest_revoke", revoke_exc)
    return result


def run(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    require(not output.exists(), "E163 output exists; no retry/overwrite")
    require(shutil.disk_usage(output.parent).free >= FREE_LIMIT, "Less than 1 GiB free")
    output.mkdir(parents=False)
    result = {
        "schema": RESULT_SCHEMA,
        "accepted": False,
        "publication_complete": False,
        "diagnostic_valid": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "F1_F8_complete": False,
        "active_UI_changed": False,
        "interpretation_limit": "Saved timing statistics only; phase is not causal or biological validation.",
    }
    freeze: dict[str, Any] | None = None
    source_before: dict[str, str] = {}
    owned_before: dict[str, str] = {}
    primary_error = None
    secondary: list[dict[str, str]] = []
    ready = False
    freeze_snapshot_sha: str | None = None
    review_snapshot_sha: str | None = None
    frozen_post_verified = False
    try:
        freeze, review = verified_inputs(
            freeze_path, freeze_sha, review_path, review_sha
        )
        atomic_json(output / "freeze-snapshot.json", freeze)
        atomic_json(output / "source-review-snapshot.json", review)
        freeze_snapshot_sha = sha256(output / "freeze-snapshot.json")
        review_snapshot_sha = sha256(output / "source-review-snapshot.json")
        paths = {key: _path(freeze, key) for key in freeze["paths"]}
        frozen = frozen_hashes(freeze)
        source_before = {
            str(path.relative_to(ROOT)).replace("\\", "/"): frozen[
                str(path.relative_to(ROOT)).replace("\\", "/")
            ]
            for path in paths.values()
        }
        owned_before = dict(review["source_sha256"])
        loaded: dict[str, Any] = {}
        for key in (
            "signals",
            "metadata",
            "calibration",
            "recruitment",
            "adapter",
            "features",
            "old_events",
            "native",
        ):
            with np.load(paths[key], allow_pickle=False) as archive:
                loaded[key] = {name: archive[name] for name in archive.files}
        loaded["trace"] = json.loads(paths["trace"].read_text(encoding="utf-8"))
        state = json.loads(paths["state_json"].read_text(encoding="utf-8"))
        loaded["groups"] = state["adapter"]["groups"]
        loaded["layout"] = json.loads(paths["layout"].read_text(encoding="utf-8"))
        loaded["group_legs"] = freeze["sample_contract"]["group_legs"]
        loaded["body_hash"] = freeze["body_hash"]
        provenance = json.loads(
            paths["projection_provenance"].read_text(encoding="utf-8")
        )
        require(
            type(provenance) is list
            and len(provenance) == 6
            and {row["group"]: row["leg"] for row in provenance}
            == {int(CORE_GROUPS[leg]): LEGS[leg] for leg in range(6)},
            "Projection source-to-leg provenance differs",
        )
        arrays, summary = analyze_loaded(loaded)
        atomic_npz(output / "raw-derived.npz", **arrays)
        validate_numeric_contract(arrays)
        atomic_json(output / "summary.json", _jsonable(summary))
        result.update(summary)
        ready = True
    except Exception as exc:  # noqa: BLE001
        primary_error = {"error_type": type(exc).__name__, "error": str(exc)}
        result["primary_error"] = primary_error
        try:
            (output / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        except Exception as log_exc:  # noqa: BLE001
            secondary.append(
                {
                    "stage": "diagnostic_logging",
                    "error_type": type(log_exc).__name__,
                    "error": str(log_exc),
                }
            )
    if freeze is not None:
        try:
            verify_hash_map(frozen_hashes(freeze), "post-frozen")
            frozen_post_verified = True
            require(sha256(freeze_path) == freeze_sha, "E163 freeze changed")
            require(sha256(review_path) == review_sha, "E163 source review changed")
            require(
                freeze_snapshot_sha == sha256(output / "freeze-snapshot.json")
                and review_snapshot_sha
                == sha256(output / "source-review-snapshot.json"),
                "E163 gate snapshot changed",
            )
            require(
                json.loads(
                    (output / "freeze-snapshot.json").read_text(encoding="utf-8")
                )
                == freeze
                and json.loads(
                    (output / "source-review-snapshot.json").read_text(encoding="utf-8")
                )
                == review,
                "E163 gate snapshot content differs",
            )
        except Exception as exc:  # noqa: BLE001
            secondary.append(
                {
                    "stage": "post_frozen",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    try:
        source_after = dict(source_before) if frozen_post_verified else {}
        owned_after = {name: sha256(ROOT / name) for name in OWNED}
        require(
            not owned_before or owned_after == owned_before,
            "E163 owned source changed during analysis",
        )
        result["source_hashes_pre"] = source_before
        result["source_hashes_post"] = source_after
        result["frozen_hashes_post_verified"] = frozen_post_verified
        result["owned_source_hashes_post"] = owned_after
    except Exception as exc:  # noqa: BLE001
        secondary.append(
            {
                "stage": "post_source",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["secondary_errors"] = secondary
    success = ready and primary_error is None and not secondary
    result.update(
        {
            "accepted": success,
            "publication_complete": success,
            "diagnostic_valid": success,
        }
    )
    ledger_counts = (
        freeze["counts"]
        if freeze
        else {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_models": 0,
            "state_store_instances": 0,
            "physics_steps": 0,
            "neural_steps": 0,
            "fit_calls": 0,
            "candidate_selections": 0,
            "F6_exposed": False,
        }
    )
    ledger = {
        "schema": "flylab.m3.e163.ledger.v1",
        **ledger_counts,
        "primary_error": primary_error,
        "secondary_errors": secondary,
        "candidate_adopted": False,
        "whole_goal_complete": False,
    }
    return finalize(output, result, ledger, started)


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
    print(json.dumps(_jsonable(result), allow_nan=False))
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
