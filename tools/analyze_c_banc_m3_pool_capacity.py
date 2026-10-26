"""E143 CPU-only optimistic bounds for frozen within-muscle MN pooling.

This reads one already recorded E142 trajectory.  It does not construct or step
a neural network or body, fit weights, solve an LP, or produce an adoptable
weight matrix.  The pointwise envelopes are necessary conditions only: their
extrema may use different neurons at different times and joints.
"""

# ruff: noqa: BLE001  # preserve primary failures while retaining partial evidence

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.storage import StateStore
from tools import analyze_c_banc_m3_support_transfer as e122
from tools.fit_c_banc_m3_native_all_leg import own_leg_targets, reference_at_phase
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks

FREEZE_SHA = "b6ba163d4d854a282a26507ec1c253a79ef0cd2204ef8f74842a9c0a93731c24"
DESIGN_SHA = "cb6e130d46a18fd380f58ca1c4529a1766d44dc87c57e11df015e5ceec2c754f"
SCHEMA = "flylab.m3.e143.pool-capacity.v1"
OWNED = (
    "tools/analyze_c_banc_m3_pool_capacity.py",
    "tests/test_c_banc_pool_capacity.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
E1_IDS = (53861, 71439, 152280, 92648, 97727, 91167)
OFFSETS = (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
DEFAULT_FREEZE = ROOT / "verification/banc-natural-gait-m3-20260919/e143-pool-capacity-freeze.json"
DEFAULT_OUTPUT = ROOT / "verification/banc-natural-gait-m3-20260919/e143-pool-capacity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_path(relative: str, expected: str) -> None:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe frozen path: {relative}")
    require(sha256(ROOT / path) == expected, f"Frozen hash mismatch: {relative}")


def save_json_new(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz_new(path: Path, arrays: dict[str, Any]) -> None:
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def add_secondary(primary: BaseException, operation: str, secondary: BaseException) -> None:
    primary.add_note(
        f"Secondary E143 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E143_POOL_CAPACITY_DESIGN_20260921.md",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "data_directory": "verification/banc-natural-gait-m3-20260919/e142-cached-native-body",
        "reference_path": "verification/banc-natural-gait-m3-20260919/e85-space-compatible-cadence-reference/commands.npz",
        "prior_actual_conservative_s": 2156.580485100043,
        "prior_conservative_process_s": 2157,
        "internal_wall_limit_s": 25,
        "external_wall_limit_s": 35,
        "supervisor_limit_s": 40,
        "prospective_cumulative_limit_s": 2197,
        "original_program_cumulative_limit_s": 2271,
        "original_processes": 1,
        "automatic_retries": 0,
        "minimum_free_disk_bytes": 1073741824,
        "maximum_raw_bytes": 33554432,
        "cpu_threads": 1,
        "gpu_count": 0,
        "recorded_EMA_steps": 2000,
        "original_neural_steps": 0,
        "physics_steps": 0,
        "conditional_network_response_calls": 0,
        "fit_calls": 0,
        "LP_calls": 0,
        "leg_order": list(LEGS),
        "E1_ids": list(E1_IDS),
        "reference_offsets": list(OFFSETS),
        "decoder_atol": 1e-10,
        "reference_reconstruction_RMS_limit": 0.002,
        "candidate_adopted": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
    }


def reviewed_inputs(
    freeze_path: Path,
    freeze_sha: str,
    review_path: Path,
    review_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete the source gate before opening any recorded numerical input."""
    require(freeze_sha == FREEZE_SHA, "Exact E143 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E143 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E143 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E143 design changed")
    for mapping_name in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(mapping_name)
        require(isinstance(mapping, dict) and mapping, f"Missing frozen mapping: {mapping_name}")
        for relative, digest in mapping.items():
            checked_path(relative, digest)

    require(sha256(review_path) == review_sha, "E143 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepted_for_single_pool_capacity_analysis") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E143 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    require(array.shape == shape, f"E143 {name} shape mismatch: {array.shape}")
    require(np.issubdtype(array.dtype, np.number), f"E143 {name} is not numeric")
    require(np.isfinite(array).all(), f"E143 {name} contains nonfinite values")
    return array.astype(np.float64, copy=False)


def verify_clock(rows: list[dict[str, Any]], signal_time: Any) -> np.ndarray:
    require(len(rows) == 2001, "E143 requires 2001 control states including t0")
    ticks = np.asarray([row.get("control_tick") for row in rows])
    require(np.array_equal(ticks, np.arange(2001)), "E143 control ticks have a gap or shift")
    times = finite_array([row.get("time_s") for row in rows], (2001,), "control clock")
    require(
        np.allclose(times, np.arange(2001) * 0.005, atol=2e-8, rtol=0),
        "E143 control clock differs from the fixed 5 ms grid",
    )
    endpoint_time = finite_array(signal_time, (2000,), "signal endpoint clock")
    require(
        np.array_equal(endpoint_time, times[1:]),
        "E143 signal endpoints do not map exactly to trace rows[1:]",
    )
    return times


def validate_decoder(
    cal: dict[str, Any],
    meta: dict[str, Any],
    signal: dict[str, Any],
    anatomy: dict[str, Any],
    start: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    values = e122.decoder_calibration(cal, meta, signal, anatomy, start)
    rate = np.asarray(start["network"]["rate"])
    require(rate.shape == (158706,) and np.isfinite(rate).all() and np.all(rate == 0),
            "E143 original network rate is not whole-zero")
    require(start.get("control_tick") == 0 and start["network"].get("tick") == 0,
            "E143 original checkpoint does not start at tick zero")
    return values


def ordered_motor_rate(
    signal: dict[str, Any], motor_ids: np.ndarray, observed_ids: np.ndarray
) -> np.ndarray:
    rates = finite_array(signal["rate"], (2000, 446), "recorded observed rate")
    lookup = {int(cell): index for index, cell in enumerate(observed_ids)}
    require(len(lookup) == 446 and all(int(cell) in lookup for cell in motor_ids),
            "E143 observed cohort does not contain every motor ID exactly once")
    return rates[:, [lookup[int(cell)] for cell in motor_ids]]


def ema_filter(rates: Any, progress: dict[str, Any] | None = None) -> np.ndarray:
    rates = finite_array(rates, (2000, 391), "ordered motor rate")
    alpha = -np.expm1(-0.005 / 0.01)
    result = np.empty_like(rates)
    state = np.zeros(391, dtype=np.float64)
    for index in range(2000):
        state += alpha * (rates[index] - state)
        result[index] = state
        if progress is not None:
            progress["recorded_EMA_updates_completed"] += 1
    return result


def pool_envelopes(
    filtered: Any,
    pool: Any,
    membership: Any,
    moment: Any,
    rest: Any,
    ltm: Any,
) -> dict[str, np.ndarray]:
    """Pointwise optimistic command and derivative envelopes, without fitting."""
    filtered = finite_array(filtered, (2000, 391), "filtered motor rate")
    pool = finite_array(pool, (96, 391), "pool")
    moment = finite_array(moment, (42, 96), "moment")
    rest = finite_array(rest, (42,), "rest")
    membership = np.asarray(membership)
    ltm = np.asarray(ltm)
    require(membership.shape == (391,) and np.issubdtype(membership.dtype, np.integer),
            "E143 membership shape/type mismatch")
    require(np.all((membership >= 0) & (membership < 96)), "E143 membership is out of range")
    require(ltm.shape == (6,) and np.issubdtype(ltm.dtype, np.integer)
            and np.all((ltm >= 0) & (ltm < 96)), "E143 tendon indices invalid")
    require(np.all(pool >= 0) and np.allclose(pool.sum(axis=1), 1, atol=1e-12, rtol=0),
            "E143 pool must be nonnegative and row-normalized")
    require(np.array_equal(np.sum(pool > 0, axis=0), np.ones(391, dtype=int))
            and np.array_equal(np.argmax(pool, axis=0), membership),
            "E143 pool support differs from exact membership")

    activation = filtered @ pool.T
    low = np.empty((2000, 96), dtype=np.float64)
    high = np.empty_like(low)
    derivative = np.diff(filtered, axis=0) / 0.005
    derivative_low = np.empty((1999, 96), dtype=np.float64)
    derivative_high = np.empty_like(derivative_low)
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        require(columns.size > 0, f"E143 empty MN pool: muscle {muscle}")
        low[:, muscle] = np.min(filtered[:, columns], axis=1)
        high[:, muscle] = np.max(filtered[:, columns], axis=1)
        derivative_low[:, muscle] = np.min(derivative[:, columns], axis=1)
        derivative_high[:, muscle] = np.max(derivative[:, columns], axis=1)

    positive = np.maximum(moment, 0)
    negative = np.minimum(moment, 0)
    command = rest + 0.02 * (activation @ moment.T)
    command_low = rest + 0.02 * (low @ positive.T + high @ negative.T)
    command_high = rest + 0.02 * (high @ positive.T + low @ negative.T)
    command_velocity = np.diff(command, axis=0) / 0.005
    velocity_low = 0.02 * (
        derivative_low @ positive.T + derivative_high @ negative.T
    )
    velocity_high = 0.02 * (
        derivative_high @ positive.T + derivative_low @ negative.T
    )
    return {
        "activation": activation,
        "activation_low": low,
        "activation_high": high,
        "command": command,
        "command_low": command_low,
        "command_high": command_high,
        "motor_derivative": derivative,
        "activation_derivative_low": derivative_low,
        "activation_derivative_high": derivative_high,
        "command_velocity": command_velocity,
        "velocity_low": velocity_low,
        "velocity_high": velocity_high,
        "tendon_low": 0.2 * np.tanh(low[:, ltm] / 10),
        "tendon_high": 0.2 * np.tanh(high[:, ltm] / 10),
    }


def target_basis(
    times: Any,
    signal: dict[str, Any],
    observed_ids: Any,
    reference: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    times = finite_array(times, (2001,), "target grid")
    observed = np.asarray(observed_ids)
    rates = finite_array(signal["rate"], (2000, 446), "target source rate")
    lookup = {int(cell): index for index, cell in enumerate(observed)}
    require(all(cell in lookup for cell in E1_IDS), "E143 fixed E1 IDs missing from observed cohort")
    columns = np.asarray([lookup[cell] for cell in E1_IDS], dtype=np.int64)
    source_rate = np.vstack((np.zeros((1, 6)), rates[:, columns]))
    peaks = [quadratic_peaks(times, source_rate[:, column], prominence=20) for column in range(6)]
    require(all(len(item) >= 2 for item in peaks), "E143 E1 peaks do not bracket a phase interval")
    phases = np.column_stack([phase_trace(times, item) for item in peaks])
    last_peaks = np.asarray([item[-1] for item in peaks])
    mask = (times >= 0.5) & (times < np.min(last_peaks)) & np.isfinite(phases).all(axis=1)
    indices = np.flatnonzero(mask)
    require(indices.size > 1, "E143 target mask is empty or has no adjacent pair")
    require(np.all(np.diff(indices) == 1), "E143 target mask contains a non-adjacent gap")

    reference_time = finite_array(reference["time_s"], (2000,), "E85 reference clock")
    reference_command = finite_array(reference["command"], (2000, 42), "E85 reference command")
    reference_phase = finite_array(reference["phase"], (2000, 6), "E85 reference phase")
    require(np.allclose(np.diff(reference_time), 0.005, atol=2e-8, rtol=0),
            "E143 E85 reference clock differs")
    for leg, expected in enumerate(OFFSETS):
        delta = np.mod((reference_phase[:, leg] - reference_phase[:, 0]) / (2 * np.pi), 1)
        error = np.abs(np.mod(delta - expected + 0.5, 1) - 0.5)
        require(np.max(error) <= 1e-10, f"E143 E85 reference offset differs for {LEGS[leg]}")
    train = (reference_time >= 5) & (reference_time < 8)
    holdout = (reference_time >= 8) & (reference_time < 10)
    reconstructed = reference_at_phase(
        reference, reference_phase[holdout, 0] / (2 * np.pi), 0, train
    )
    residual = float(np.sqrt(np.mean(np.square(reconstructed - reference_command[holdout]))))
    target = own_leg_targets(reference, phases, np.asarray(OFFSETS), mask, 0)
    padded_peaks = np.full((6, max(len(item) for item in peaks)), np.nan)
    for leg, values in enumerate(peaks):
        padded_peaks[leg, : len(values)] = values
    return {
        "e1_columns": columns,
        "source_rate": source_rate,
        "peaks_s": padded_peaks,
        "phase_cycles": phases,
        "last_peaks_s": last_peaks,
        "mask": mask,
        "target_indices": indices,
        "target_rad": target,
        "reference_train_mask": train,
        "reference_holdout_mask": holdout,
        "reference_reconstructed_rad": reconstructed,
        "reference_residual_rms_rad": residual,
    }


def outside_distance(target: Any, low: Any, high: Any) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    require(target.shape == low.shape == high.shape, "E143 envelope comparison shape mismatch")
    require(np.isfinite(target).all() and np.isfinite(low).all() and np.isfinite(high).all(),
            "E143 envelope comparison contains nonfinite values")
    require(np.all(low <= high), "E143 lower envelope exceeds upper envelope")
    return np.maximum(np.maximum(low - target, target - high), 0)


def _ratio(floor: float, baseline: float) -> float | None:
    return floor / baseline if baseline > 0 else None


def metric_summary(baseline: Any, target: Any, distance: Any) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    require(baseline.shape == target.shape == distance.shape and baseline.ndim == 2
            and baseline.shape[1] == 42, "E143 metric shape mismatch")

    def one(error: np.ndarray, floor: np.ndarray) -> dict[str, Any]:
        baseline_rms = float(np.sqrt(np.mean(np.square(error))))
        floor_rms = float(np.sqrt(np.mean(np.square(floor))))
        return {
            "baseline_RMS": baseline_rms,
            "optimistic_RMS_floor": floor_rms,
            "floor_to_baseline_ratio": _ratio(floor_rms, baseline_rms),
            "outside_fraction": float(np.count_nonzero(floor > 0) / floor.size),
            "maximum_outside_distance": float(np.max(floor)),
        }

    error = baseline - target
    result = one(error, distance)
    result["by_leg"] = {
        leg: one(error[:, 7 * index : 7 * (index + 1)], distance[:, 7 * index : 7 * (index + 1)])
        for index, leg in enumerate(LEGS)
    }
    result["per_joint"] = [one(error[:, index : index + 1], distance[:, index : index + 1])
                           for index in range(42)]
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def load_inputs(freeze: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / freeze["data_directory"]
    trace = json.loads((source / "trace.json").read_text(encoding="utf-8"))
    return {
        "rows": trace["samples"],
        "signal": _load_npz(source / "recruitment-signals.npz"),
        "meta": _load_npz(source / "recruitment-metadata.npz"),
        "cal": _load_npz(source / "assembled-calibration.npz"),
        "anatomy": json.loads((source / "muscle-provenance.json").read_text(encoding="utf-8")),
        "start": StateStore.load(source / "start"),
        "source_phase": json.loads((source / "source-phase-summary.json").read_text(encoding="utf-8")),
        "reference": _load_npz(ROOT / freeze["reference_path"]),
    }


def calculate(
    data: dict[str, Any],
    freeze: dict[str, Any],
    progress: dict[str, Any],
    partial: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if partial is None:
        partial = {}
    rows = data["rows"]
    signal = data["signal"]
    # Retain the unvalidated basis first.  A bad clock, anatomy row, or NaN then
    # still leaves the exact generated-fixture/original input that caused it.
    partial.update({
        "unvalidated_control_ticks": np.asarray([row.get("control_tick") for row in rows]),
        "unvalidated_grid_time_s": np.asarray([row.get("time_s") for row in rows]),
        "unvalidated_endpoint_time_s": np.asarray(signal.get("time_s", [])),
        "unvalidated_observed_ids": np.asarray(signal.get("observed_ids", [])),
        "unvalidated_recorded_rate": np.asarray(signal.get("rate", [])),
        "unvalidated_recorded_activation": np.asarray(signal.get("muscle_activation", [])),
        "unvalidated_pool": np.asarray(data["cal"].get("pool", [])),
        "unvalidated_membership": np.asarray(data["cal"].get("membership", [])),
        "unvalidated_moment": np.asarray(data["cal"].get("moment", [])),
        "unvalidated_rest": np.asarray(data["cal"].get("rest", [])),
        "unvalidated_ltm": np.asarray(data["cal"].get("ltm", [])),
    })
    times = verify_clock(rows, signal["time_s"])
    partial["grid_time_s"] = times
    recorded_request_full = finite_array(
        [row["joint_targets_rad"] for row in rows], (2001, 42), "stored full-grid request"
    )
    partial["recorded_request_full_grid_rad"] = recorded_request_full
    motor_ids, observed_ids, pool, moment, rest, ltm, muscles, roster = validate_decoder(
        data["cal"], data["meta"], signal, data["anatomy"], data["start"]
    )
    require(tuple(freeze["leg_order"]) == LEGS and tuple(freeze["E1_ids"]) == E1_IDS
            and tuple(freeze["reference_offsets"]) == OFFSETS,
            "E143 frozen phase identity changed")
    phase_ids = tuple(int(value) for value in data["source_phase"].get("E1_ids", []))
    require(set(phase_ids) == set(E1_IDS) and len(phase_ids) == 6,
            "E143 saved source phase E1 roster differs")

    rates = ordered_motor_rate(signal, motor_ids, observed_ids)
    partial.update({"motor_ids": motor_ids, "observed_ids": observed_ids,
                    "recorded_motor_rate": rates})
    progress["recorded_EMA_updates_attempted"] = 2000
    filtered = ema_filter(rates, progress)
    partial["filtered_motor_rate"] = filtered
    envelope = pool_envelopes(
        filtered, pool, data["cal"]["membership"], moment, rest, ltm
    )
    partial.update({key: value for key, value in envelope.items()
                    if key != "motor_derivative"})
    target = target_basis(times, signal, observed_ids, data["reference"])
    partial.update({
        "source_E1_rate_full_grid": target["source_rate"],
        "source_E1_peaks_s": target["peaks_s"],
        "source_phase_cycles": target["phase_cycles"],
        "target_mask": target["mask"],
        "target_indices": target["target_indices"],
        "target_rad": target["target_rad"],
    })
    indices = np.asarray(target["target_indices"])
    endpoint_rows = indices - 1
    target_rad = np.asarray(target["target_rad"])
    command_low = envelope["command_low"][endpoint_rows]
    command_high = envelope["command_high"][endpoint_rows]
    command_distance = outside_distance(target_rad, command_low, command_high)

    pair_start = indices[:-1]
    pair_end = indices[1:]
    adjacent = pair_end == pair_start + 1
    require(np.all(adjacent), "E143 velocity target attempted to bridge a mask gap")
    target_velocity = np.diff(target_rad, axis=0) / 0.005
    velocity_rows = pair_end - 2
    velocity_low = envelope["velocity_low"][velocity_rows]
    velocity_high = envelope["velocity_high"][velocity_rows]
    velocity_distance = outside_distance(target_velocity, velocity_low, velocity_high)

    recorded_activation = finite_array(signal["muscle_activation"], (2000, 96), "stored activation")
    # t0 is the held body action.  Only the 2000 actual decoder endpoints are
    # compared to the rest-plus-moment reconstruction.
    recorded_request = recorded_request_full[1:]
    recorded_tendon = finite_array(
        [[row["tendon_inputs"][f"{leg}_tarsus"] for leg in LEGS] for row in rows[1:]],
        (2000, 6),
        "stored tendon",
    )
    decoder_errors = {
        "activation": float(np.max(np.abs(envelope["activation"] - recorded_activation))),
        "request": float(np.max(np.abs(envelope["command"] - recorded_request))),
        "tendon": float(np.max(np.abs(
            0.2 * np.tanh(envelope["activation"][:, ltm] / 10) - recorded_tendon
        ))),
    }
    raw = {
        "grid_time_s": times,
        "endpoint_time_s": times[1:],
        "leg_order": np.asarray(LEGS),
        "E1_ids": np.asarray(E1_IDS),
        "E1_columns": target["e1_columns"],
        "reference_offsets_cycles": np.asarray(OFFSETS),
        "motor_ids": motor_ids,
        "observed_ids": observed_ids,
        "motor_roster": np.asarray(roster),
        "pool": pool,
        "membership": np.asarray(data["cal"]["membership"]),
        "moment": moment,
        "rest": rest,
        "ltm": ltm,
        "recorded_motor_rate": rates,
        "filtered_motor_rate": filtered,
        "recorded_activation": recorded_activation,
        "activation": envelope["activation"],
        "activation_low": envelope["activation_low"],
        "activation_high": envelope["activation_high"],
        "baseline_endpoint_rad": envelope["command"],
        "recorded_request_full_grid_rad": recorded_request_full,
        "recorded_request_endpoint_rad": recorded_request,
        "command_low_rad": envelope["command_low"],
        "command_high_rad": envelope["command_high"],
        "recorded_tendon": recorded_tendon,
        "tendon_low": envelope["tendon_low"],
        "tendon_high": envelope["tendon_high"],
        "source_E1_rate_full_grid": target["source_rate"],
        "source_E1_peaks_s": target["peaks_s"],
        "source_phase_cycles": target["phase_cycles"],
        "target_mask": target["mask"],
        "target_indices": indices,
        "target_rad": target_rad,
        "selected_baseline_rad": envelope["command"][endpoint_rows],
        "selected_command_low_rad": command_low,
        "selected_command_high_rad": command_high,
        "command_outside_distance_rad": command_distance,
        "velocity_pair_start_indices": pair_start,
        "velocity_pair_end_indices": pair_end,
        "target_velocity_rad_s": target_velocity,
        "baseline_velocity_rad_s": envelope["command_velocity"][velocity_rows],
        "velocity_low_rad_s": velocity_low,
        "velocity_high_rad_s": velocity_high,
        "velocity_outside_distance_rad_s": velocity_distance,
        "reference_train_mask": target["reference_train_mask"],
        "reference_holdout_mask": target["reference_holdout_mask"],
        "reference_reconstructed_rad": target["reference_reconstructed_rad"],
    }
    facts = {
        "decoder_errors": decoder_errors,
        "reference_reconstruction_RMS_rad": target["reference_residual_rms_rad"],
        "command": metric_summary(raw["selected_baseline_rad"], target_rad, command_distance),
        "velocity": metric_summary(raw["baseline_velocity_rad_s"], target_velocity, velocity_distance),
        "target_states": int(indices.size),
        "target_velocity_pairs": int(target_velocity.shape[0]),
        "first_target_grid_index": int(indices[0]),
        "last_target_grid_index": int(indices[-1]),
        "phase_last_peaks_s": np.asarray(target["last_peaks_s"]).tolist(),
        "muscles": len(muscles),
    }
    return raw, facts


def assert_numerics(raw: dict[str, Any], facts: dict[str, Any], freeze: dict[str, Any]) -> None:
    tolerance = float(freeze["decoder_atol"])
    require(max(facts["decoder_errors"].values()) <= tolerance,
            f"E143 recorded decoder mismatch: {facts['decoder_errors']}")
    require(facts["reference_reconstruction_RMS_rad"] <= freeze["reference_reconstruction_RMS_limit"],
            "E143 E85 reference reconstruction exceeds frozen RMS limit")
    require(np.all(raw["activation"] >= raw["activation_low"] - 1e-12)
            and np.all(raw["activation"] <= raw["activation_high"] + 1e-12),
            "E143 baseline activation lies outside its convex envelope")
    require(np.all(raw["baseline_endpoint_rad"] >= raw["command_low_rad"] - 1e-12)
            and np.all(raw["baseline_endpoint_rad"] <= raw["command_high_rad"] + 1e-12),
            "E143 baseline command lies outside its optimistic envelope")
    require(np.all(raw["baseline_velocity_rad_s"] >= raw["velocity_low_rad_s"] - 1e-10)
            and np.all(raw["baseline_velocity_rad_s"] <= raw["velocity_high_rad_s"] + 1e-10),
            "E143 baseline derivative lies outside its optimistic envelope")
    for metric in (facts["command"], facts["velocity"]):
        require(metric["optimistic_RMS_floor"] <= metric["baseline_RMS"] + 1e-12,
                "E143 optimistic floor exceeds baseline RMS")


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e143.pool-capacity-result.v1",
        "experiment": "M3-E143-pool-capacity",
        "status": "failed",
        "validity": "invalid_incomplete",
        "candidate_adopted": False,
        "F6_exposed": False,
        "whole_goal_complete": False,
        "failure": progress.get(
            "primary_failure",
            {"type": type(primary).__name__, "message": str(primary)},
        ),
    }


def run(
    freeze_path: Path,
    freeze_sha: str,
    review_path: Path,
    review_sha: str,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "schema": "flylab.m3.e143.pool-capacity-execution.v1",
        "status": "created",
        "recorded_EMA_updates_attempted": 0,
        "recorded_EMA_updates_completed": 0,
        "original_neural_steps": 0,
        "CUDA_calls": 0,
        "physics_steps": 0,
        "conditional_network_response_calls": 0,
        "fit_calls": 0,
        "LP_calls": 0,
        "original_processes": 1,
        "automatic_retries": 0,
        "cpu_threads_requested": 1,
        "gpu_count": 0,
        "candidate_adopted": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
        "timings_s": {},
        "failed_raw_operations": [],
        "failed_persistence_operations": [],
        "failed_resource_checks": [],
    }
    partial: dict[str, Any] = {}
    freeze: dict[str, Any] | None = None
    primary: BaseException | None = None
    summary: dict[str, Any] = {}
    raw_saved = False

    def capture_failure(error: BaseException, operation: str) -> None:
        nonlocal primary
        if primary is None:
            primary = error
            progress["primary_failure"] = {
                "type": type(error).__name__, "message": str(error)
            }
        else:
            add_secondary(primary, operation, error)

    def check_resources(stage: str) -> None:
        elapsed = time.perf_counter() - started
        progress[f"wall_s_{stage}"] = elapsed
        progress["wall_s"] = elapsed
        if freeze is None:
            return
        try:
            size = output_bytes(output)
        except BaseException as error:
            size = None
            progress["failed_resource_checks"].append(f"{stage}:output_bytes_unavailable")
            capture_failure(error, f"{stage} output byte accounting")
        progress[f"bytes_{stage}"] = size
        progress["raw_bytes"] = size
        checks = (
            (elapsed > freeze["internal_wall_limit_s"], "internal_wall_limit_s",
             TimeoutError(f"E143 exceeded internal wall limit during {stage}")),
            (size is not None and size > freeze["maximum_raw_bytes"], "maximum_raw_bytes",
             RuntimeError(f"E143 exceeded raw byte limit during {stage}")),
        )
        for failed, label, error in checks:
            marker = f"{stage}:{label}"
            if failed and marker not in progress["failed_resource_checks"]:
                progress["failed_resource_checks"].append(marker)
                capture_failure(error, f"{stage} resource check")

    try:
        stage = time.perf_counter()
        freeze, review = reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha)
        progress["timings_s"]["source_gate"] = time.perf_counter() - stage
        progress["source_review_sha256"] = review_sha
        progress["source_sha256"] = review["source_sha256"]
        save_json_new(output / "freeze-snapshot.json", freeze)
        save_json_new(output / "source-review-snapshot.json", review)
        free = shutil.disk_usage(output.parent).free
        progress["starting_free_disk_bytes"] = free
        require(free >= freeze["minimum_free_disk_bytes"], "E143 insufficient starting free disk")
        require(time.perf_counter() - started < freeze["internal_wall_limit_s"],
                "E143 source gate exhausted internal wall limit")

        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(all(int(item.get("num_threads", 1)) <= 1 for item in progress["thread_pools"]),
                    "E143 CPU thread pool exceeds one thread")
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            raw, facts = calculate(data, freeze, progress, partial)
            partial.update(raw)
            progress["timings_s"]["calculation"] = time.perf_counter() - stage
            stage = time.perf_counter()
            try:
                save_npz_new(output / "pool-capacity-raw.npz", raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {"operation": "save pool-capacity-raw.npz", "type": type(error).__name__,
                     "message": str(error)}
                )
                raise
            raw_saved = True
            progress["timings_s"]["raw_persistence"] = time.perf_counter() - stage
            assert_numerics(raw, facts, freeze)

        progress["status"] = "pending_finalization"
        summary = {
            "schema": "flylab.m3.e143.pool-capacity-result.v1",
            "experiment": "M3-E143-pool-capacity",
            "status": "pass",
            "validity": "conditional_saved_E142_capacity_bound",
            "hypothesis_verdict": "reported_as_optimistic_necessary_condition_only",
            "analysis": facts,
            "interpretation": {
                "large_floor": "positive limit on reducing this saved-reference residual with normalized within-muscle pooling",
                "small_floor": "does not exclude a useful static weighting and is not evidence that one exists",
                "scope": "pointwise joint/time extrema are necessary but not sufficient for one static weight vector across time and joints",
                "gait_claim": "not a natural-gait candidate or gait proof",
                "returned_weight_matrix": False,
            },
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
            "required_persistence_included_in_resource_check": True,
        }
    except BaseException as error:
        primary = error
        progress["status"] = "failed"
        progress["primary_failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if partial and not raw_saved:
            try:
                save_npz_new(output / "partial-raw.npz", partial)
                progress["partial_raw_saved"] = True
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {"operation": "save partial-raw.npz", "type": type(error).__name__, "message": str(error)}
                )
                if primary is None:
                    capture_failure(error, "partial raw")
                else:
                    add_secondary(primary, "partial raw", error)
        check_resources("before_required_receipts")
        if primary is not None:
            progress["status"] = "failed"
            if "primary_failure" not in progress:
                progress["primary_failure"] = {
                    "type": type(primary).__name__, "message": str(primary)
                }
            summary = failure_summary(primary, progress)

        # The execution receipt is deliberately pending when calculations pass.
        # Only summary.json is authoritative, after every required receipt and a
        # post-summary wall/byte check.
        try:
            save_json_new(output / "execution-raw.json", progress)
        except BaseException as error:
            progress["failed_persistence_operations"].append(
                {"operation": "save execution-raw.json", "type": type(error).__name__,
                 "message": str(error)}
            )
            capture_failure(error, "save execution-raw.json")
        if progress["failed_raw_operations"] or progress["failed_persistence_operations"]:
            try:
                save_json_new(output / "secondary-errors.json", {
                    "raw": progress["failed_raw_operations"],
                    "persistence": progress["failed_persistence_operations"],
                    "resource": progress["failed_resource_checks"],
                })
            except BaseException as error:
                progress["failed_persistence_operations"].append(
                    {"operation": "save secondary-errors.json", "type": type(error).__name__,
                     "message": str(error)}
                )
                capture_failure(error, "save secondary-errors.json")

        check_resources("before_summary")
        if primary is not None:
            progress["status"] = "failed"
            summary = failure_summary(primary, progress)

        summary_was_success = summary.get("status") == "pass"
        try:
            save_json_new(output / "summary.json", summary)
        except BaseException as error:
            progress["failed_persistence_operations"].append(
                {"operation": "save summary.json", "type": type(error).__name__,
                 "message": str(error)}
            )
            capture_failure(error, "save summary.json")

        check_resources("after_summary")
        if primary is None:
            progress["status"] = "pass"
        elif summary_was_success:
            # A pass file is never left authoritative when its own persistence
            # crosses a resource bound.  Rename preserves it as non-authoritative
            # evidence; a new failure summary occupies the canonical path.
            try:
                if (output / "summary.json").exists():
                    (output / "summary.json").rename(output / "invalidated-success-summary.json")
                save_json_new(output / "summary.json", failure_summary(primary, progress))
                summary = failure_summary(primary, progress)
            except BaseException as error:
                progress["failed_persistence_operations"].append(
                    {"operation": "invalidate authoritative success summary",
                     "type": type(error).__name__, "message": str(error)}
                )
                capture_failure(error, "invalidate authoritative success summary")

        if primary is not None:
            progress["status"] = "failed"
            finalization = {
                "status": "failed",
                "primary_failure": progress.get("primary_failure"),
                "failed_raw_operations": progress["failed_raw_operations"],
                "failed_persistence_operations": progress["failed_persistence_operations"],
                "failed_resource_checks": progress["failed_resource_checks"],
                "wall_s_at_finalization_receipt": time.perf_counter() - started,
                "bytes_last_observed": progress.get("raw_bytes"),
            }
            try:
                save_json_new(output / "finalization-errors.json", finalization)
            except BaseException as error:
                add_secondary(primary, "save finalization-errors.json", error)
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    # This post-write receipt is returned through the CLI and retained in the
    # existing supervisor's stdout, without another unaccounted file write.
    summary["final_resources"] = {
        "wall_s_after_required_persistence": progress["wall_s_after_summary"],
        "bytes_after_required_persistence": progress["bytes_after_summary"],
        "all_required_persistence_and_resource_checks_passed": True,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
