"""E191 frozen, one-solve conditional singleton transfer fit on saved E169 inputs."""

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
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

import clarabel
import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e191.singleton-transfer.v1"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DESIGN = "docs/BANC_M3_E191_SINGLETON_TRANSFER_DESIGN_20260924.md"
OWNED = ("tools/fit_c_banc_m3_singleton_transfer.py", "tests/test_c_banc_singleton_transfer.py")
N_PAIR, N_W, N_X = 1029, 2058, 4117
ATOL, VELOCITY_ATOL = 1e-12, 5e-10
SOLVER_RESIDUAL_ATOL = 1e-7
SOLVER_LINEAR_ATOL = 1e-8
FIT_WALL_LIMIT_S = 20.


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_path(name: str) -> Path:
    require(isinstance(name, str) and name, "empty frozen path")
    relative = Path(name)
    result = (ROOT / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and result.is_relative_to(ROOT), "unsafe frozen path")
    return result


def checked_json(path: Path, digest: str) -> dict[str, Any]:
    require(isinstance(digest, str) and len(digest) == 64 and sha256(path) == digest, "JSON SHA256 differs")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), "JSON object required")
    return result


def validate_authority(freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = checked_json(freeze_path, freeze_sha)
    require(freeze.get("schema") == SCHEMA, "freeze schema differs")
    for field in ("input_npz", "design"):
        item = freeze.get(field)
        require(isinstance(item, dict) and set(item) >= {"path", "sha256"}, "freeze missing " + field)
        require(sha256(frozen_path(item["path"])) == item["sha256"], field + " SHA256 differs")
    require(freeze["design"]["path"] == DESIGN, "design path differs")
    sources = freeze.get("source_files")
    require(isinstance(sources, dict) and set(sources) >= set(OWNED), "source map incomplete")
    require(all(sha256(frozen_path(name)) == digest for name, digest in sources.items()), "source SHA256 differs")
    before = freeze.get("cumulative_before_s")
    cap = freeze.get("stage_cap_s")
    require(
        freeze.get("max_supervisor_s") == 95
        and freeze.get("max_internal_s") == 80
        and freeze.get("max_output_bytes") == 268435456
        and type(before) in (int, float)
        and type(cap) in (int, float)
        and math.isfinite(before)
        and math.isfinite(cap)
        and before == 4655.7939677997165
        and cap == 4994
        and before + freeze["max_supervisor_s"] <= cap,
        "resource freeze differs",
    )
    review = checked_json(review_path, review_sha)
    require(
        review.get("approved") is True
        and review.get("freeze_sha256") == freeze_sha
        and review.get("source_files") == sources,
        "source review does not approve exact freeze/sources",
    )
    return freeze, review


def atom_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atom_npz(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


SPECS = {
    "loaded_recruitment_bank": ((2001, 391, 3), "float64"),
    "loaded_recruitment_activation": ((2001, 96), "float64"),
    "loaded_recruitment_original_activation": ((2001, 96), "float64"),
    "loaded_recruitment_motor_rate": ((2001, 391), "float64"),
    "loaded_recruitment_control_tick": ((2001,), "int64"),
    "loaded_recruitment_network_tick": ((2001,), "int64"),
    "loaded_recruitment_physics_time_s": ((2001,), "float64"),
    "loaded_recruitment_complete": ((2001,), "bool"),
    "loaded_recruitment_bank_available": ((2001,), "bool"),
    "loaded_recruitment_activation_available": ((2001,), "bool"),
    "loaded_recruitment_original_activation_available": ((2001,), "bool"),
    "loaded_recruitment_motor_rate_available": ((2001,), "bool"),
    "loaded_adapter_membership": ((391,), "int64"),
    "loaded_adapter_motor_ids": ((391,), "int64"),
    "loaded_adapter_moment": ((42, 96), "float64"),
    "loaded_adapter_rest": ((42,), "float64"),
    "loaded_adapter_pool": ((96, 391), "float64"),
    "loaded_adapter_ltm": ((6,), "int64"),
    "loaded_parameters_membership": ((391,), "int64"),
    "loaded_parameters_motor_ids": ((391,), "int64"),
    "loaded_parameters_slopes": ((96, 391, 3, 2), "float64"),
    "loaded_parameters_kappa": ((391,), "float64"),
    "loaded_parameters_tau": ((3,), "float64"),
    "loaded_calibration_membership": ((391,), "int64"),
    "loaded_calibration_motor_ids": ((391,), "int64"),
    "loaded_calibration_moment": ((42, 96), "float64"),
    "loaded_calibration_rest": ((42,), "float64"),
    "loaded_calibration_pool": ((96, 391), "float64"),
    "loaded_calibration_ltm": ((6,), "int64"),
    "loaded_state_parameters_fixed_muscles": ((19,), "int64"),
    "loaded_state_parameters_membership": ((391,), "int64"),
    "loaded_state_parameters_slopes": ((96, 391, 3, 2), "float64"),
    "loaded_state_parameters_kappa": ((391,), "float64"),
    "loaded_state_parameters_tau": ((3,), "float64"),
    "loaded_state_initial_bank": ((391, 3), "float64"),
    "loaded_state_initial_activation": ((96,), "float64"),
    "loaded_state_group_columns_json": ((), "str"),
    "loaded_trace_recorded_command": ((2001, 42), "float64"),
    "loaded_trace_tendon": ((2001, 6), "float64"),
    "loaded_trace_control_tick": ((2001,), "int64"),
    "loaded_trace_time_s": ((2001,), "float64"),
    "loaded_trace_motor_rate": ((2001, 391), "float64"),
    "loaded_startup_target_command": ((99, 42), "float64"),
    "loaded_startup_target_tendon": ((99, 6), "float64"),
    "loaded_startup_recorded_command": ((2001, 42), "float64"),
    "loaded_startup_recorded_tendon": ((2000, 6), "float64"),
    "loaded_target_mask_target_indices": ((1824,), "int64"),
    "loaded_target_mask_pair_start": ((1823,), "int64"),
    "loaded_target_mask_pair_end": ((1823,), "int64"),
    "loaded_target_mask_control_time_s": ((2001,), "float64"),
    "loaded_target_mask_mask": ((2001,), "bool"),
    "loaded_phase_diagnostic_control_time_s": ((2001,), "float64"),
    "loaded_phase_diagnostic_e1_phase": ((2001, 6), "float64"),
    "loaded_phase_diagnostic_e1_phase_available": ((2001, 6), "bool"),
    "target": ((1824, 42), "float64"),
    "target_velocity": ((1823, 42), "float64"),
    "target_indices": ((1824,), "int64"),
    "affine_rows_command": ((1824,), "int64"),
    "affine_rows_velocity_start": ((1823,), "int64"),
    "affine_rows_velocity_end": ((1823,), "int64"),
    "affine_rows_startup": ((99,), "int64"),
}


def load_input(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in SPECS if key in archive}
        for key in ("loaded_reference_command",):
            if key in archive:
                data[key] = archive[key]
    return data


def validate_schema(data: dict[str, np.ndarray]) -> None:
    require(set(SPECS) <= set(data), "original input subset missing")
    for key, (shape, dtype) in SPECS.items():
        value = data[key]
        require(value.shape == shape and (value.dtype.kind in "US" if dtype == "str" else value.dtype == np.dtype(dtype)), "input shape/dtype differs: " + key)
        if dtype != "str" and key != "loaded_phase_diagnostic_e1_phase":
            require(np.isfinite(value).all(), "nonfinite input: " + key)


def native_command(data: dict[str, np.ndarray], activation: np.ndarray) -> np.ndarray:
    moment = data["loaded_adapter_moment"]
    rest = data["loaded_adapter_rest"]
    return np.asarray([rest + .02 * (moment @ activation[tick]) for tick in range(activation.shape[0])])


def native_activation(data: dict[str, np.ndarray], slopes: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    bank = data["loaded_recruitment_bank"]
    original = data["loaded_recruitment_original_activation"]
    membership = data["loaded_adapter_membership"]
    kappa = data["loaded_parameters_kappa"]
    result = np.empty_like(original)
    result[:, fixed] = original[:, fixed]
    hinge = np.maximum(bank - kappa[None, :, None], 0)
    for muscle in np.setdiff1d(np.arange(96), fixed):
        motors = np.flatnonzero(membership == muscle)
        low = slopes[muscle, motors, :, 0]
        delta = slopes[muscle, motors, :, 1] - low
        result[:, muscle] = np.sum(low[None] * bank[:, motors], axis=(1, 2)) + np.sum(delta[None] * hinge[:, motors], axis=(1, 2))
    result[0] = data["loaded_recruitment_activation"][0]
    return result


def validate_input(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    validate_schema(data)
    get = data.__getitem__
    membership = get("loaded_adapter_membership")
    require(np.all((membership >= 0) & (membership < 96)), "ownership invalid")
    counts = np.bincount(membership, minlength=96)
    ltm = np.sort(get("loaded_adapter_ltm"))
    fixed19 = np.sort(get("loaded_state_parameters_fixed_muscles"))
    singleton = np.flatnonzero(counts == 1)
    require(np.all((membership >= 0) & (membership < 96)) and np.all(counts > 0), "ownership invalid")
    require(ltm.size == 6 and np.unique(ltm).size == 6 and np.all((ltm >= 0) & (ltm < 96)) and singleton.size == 13 and not np.isin(singleton, ltm).any(), "LTM/singleton counts invalid")
    require(np.array_equal(fixed19, np.sort(np.r_[ltm, singleton])), "E150 fixed ownership differs")
    variable = np.setdiff1d(np.arange(96), ltm)
    motors = np.flatnonzero(np.isin(membership, variable))
    require(variable.size == 90 and motors.size == 343 and np.isin(membership[motors], singleton).sum() == 13, "E191 variable ownership differs")
    groups = json.loads(str(get("loaded_state_group_columns_json")))
    require(isinstance(groups, list) and len(groups) == 96, "group provenance invalid")
    for muscle, columns in enumerate(groups):
        require(columns == np.flatnonzero(membership == muscle).tolist(), "noncontiguous graph-column ownership differs")
    for field in ("membership", "motor_ids", "moment", "rest", "pool", "ltm"):
        reference = get("loaded_adapter_" + field)
        for prefix in ("loaded_parameters_", "loaded_calibration_"):
            key = prefix + field
            if key in data:
                require(np.array_equal(reference, data[key]), "source alias differs: " + key)
    for field in ("membership", "slopes", "kappa", "tau"):
        key = "loaded_state_parameters_" + field
        require(key in data and np.array_equal(data[key], get("loaded_parameters_" + field)), "state parameter alias differs: " + field)
    motor_ids = get("loaded_adapter_motor_ids")
    require(np.unique(motor_ids).size == 391 and np.array_equal(motor_ids, np.sort(motor_ids)) and np.all((motor_ids >= 0) & (motor_ids < 158706)), "motor graph IDs differ")
    require(np.array_equal(get("loaded_recruitment_control_tick"), np.arange(2001)) and np.array_equal(get("loaded_recruitment_network_tick"), np.arange(2001) * 20), "clock ticks differ")
    require(np.array_equal(get("loaded_trace_control_tick"), np.arange(2001)) and np.allclose(get("loaded_trace_time_s"), np.arange(2001) * .005, rtol=0, atol=2e-8), "trace clock differs")
    require(np.allclose(get("loaded_recruitment_physics_time_s"), np.arange(2001) * .005, rtol=0, atol=2e-8), "recruitment clock differs")
    require(np.allclose(get("loaded_target_mask_control_time_s"), np.arange(2001) * .005, rtol=0, atol=2e-8) and np.allclose(get("loaded_phase_diagnostic_control_time_s"), np.arange(2001) * .005, rtol=0, atol=2e-8), "target/phase clock differs")
    require(np.array_equal(np.flatnonzero(get("loaded_target_mask_mask")), get("target_indices")), "target mask differs")
    phase = get("loaded_phase_diagnostic_e1_phase")
    available = get("loaded_phase_diagnostic_e1_phase_available")
    require(np.array_equal(available, np.isfinite(phase)) and available[get("target_indices")].all(), "phase availability differs")
    for field in ("complete", "bank_available", "activation_available", "original_activation_available", "motor_rate_available"):
        require(get("loaded_recruitment_" + field).all(), "incomplete recruitment: " + field)
    require(np.array_equal(get("loaded_trace_motor_rate"), get("loaded_recruitment_motor_rate")), "motor-rate alias differs")
    require(np.array_equal(get("loaded_trace_recorded_command"), get("loaded_startup_recorded_command")), "command alias differs")
    require(np.array_equal(get("target_indices"), np.arange(100, 1924)) and np.array_equal(get("affine_rows_command"), get("target_indices")), "target rows differ")
    require(np.array_equal(get("affine_rows_velocity_start"), np.arange(100, 1923)) and np.array_equal(get("affine_rows_velocity_end"), np.arange(101, 1924)), "velocity rows differ")
    require(np.array_equal(get("affine_rows_startup"), np.arange(1, 100)), "startup rows differ")
    for alias, key in (("loaded_target_mask_target_indices", "target_indices"), ("loaded_target_mask_pair_start", "affine_rows_velocity_start"), ("loaded_target_mask_pair_end", "affine_rows_velocity_end")):
        require(alias in data and np.array_equal(data[alias], get(key)), "target mask alias differs")
    require(np.array_equal(get("loaded_trace_tendon")[1:], get("loaded_startup_recorded_tendon")), "tendon one-tick alias differs")
    slopes = get("loaded_parameters_slopes")
    pool = get("loaded_adapter_pool")
    require(np.all(slopes >= 0) and np.all(slopes <= 2) and np.all(get("loaded_parameters_kappa") >= 0) and np.array_equal(get("loaded_parameters_tau"), np.array([.005, .01, .04])), "transfer inputs invalid")
    require(np.all(pool >= 0) and np.array_equal(pool != 0, (np.arange(96)[:, None] == membership[None, :])) and np.allclose(pool.sum(axis=1), 1, rtol=0, atol=ATOL), "pool support/normalization differs")
    require(np.all(get("loaded_recruitment_bank") >= 0) and np.all(get("loaded_recruitment_motor_rate") >= 0) and np.all(get("loaded_recruitment_activation") >= 0) and np.all(get("loaded_recruitment_original_activation") >= 0), "negative recruitment input")
    require(np.array_equal(get("loaded_recruitment_bank")[0], np.zeros((391, 3))) and np.array_equal(get("loaded_recruitment_activation")[0], np.zeros(96)), "initial zero state differs")
    alpha = -np.expm1(-.005 / get("loaded_parameters_tau"))
    predicted_bank = get("loaded_recruitment_bank")[:-1] + alpha[None, None, :] * (get("loaded_recruitment_motor_rate")[1:, :, None] - get("loaded_recruitment_bank")[:-1])
    require(np.max(np.abs(get("loaded_recruitment_bank")[1:] - predicted_bank)) <= ATOL, "bank recurrence differs")
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    moment = get("loaded_adapter_moment")
    for muscle in range(96):
        outside = np.r_[np.arange(7 * group_legs[muscle]), np.arange(7 * (group_legs[muscle] + 1), 42)]
        require(np.all(moment[outside, muscle] == 0), "cross-leg moment differs")
    support = np.arange(96)[:, None] == membership[None, :]
    require(np.all(slopes[~support] == 0), "slopes outside support")
    canonical = np.zeros_like(slopes)
    canonical[:, :, 1, :] = pool[:, :, None]
    require(np.array_equal(slopes[fixed19], canonical[fixed19]), "E150 fixed slopes differ")
    reconstructed = native_activation(data, slopes, fixed19)
    require(np.max(np.abs(reconstructed[1:] - get("loaded_recruitment_activation")[1:])) <= ATOL, "E150 activation reconstruction differs")
    for muscle in fixed19:
        columns = np.flatnonzero(membership == muscle)
        original_middle = np.sum(get("loaded_recruitment_bank")[:, columns, 1] * pool[muscle, columns][None, :], axis=1)
        require(np.max(np.abs(original_middle - get("loaded_recruitment_original_activation")[:, muscle])) <= ATOL, "fixed E150 original activation differs from middle bank")
    command = native_command(data, get("loaded_recruitment_activation"))
    require(np.max(np.abs(command[1:] - get("loaded_trace_recorded_command")[1:])) <= ATOL, "E150 command reconstruction differs")
    if "loaded_state_initial_bank" in data:
        require(np.array_equal(data["loaded_state_initial_bank"], get("loaded_recruitment_bank")[0]), "initial bank differs")
    if "loaded_state_initial_activation" in data:
        require(np.array_equal(data["loaded_state_initial_activation"], get("loaded_recruitment_activation")[0]), "initial activation differs")
    pair_motors = np.repeat(motors, 3)
    pair_filters = np.tile(np.arange(3), motors.size)
    pair_muscles = membership[pair_motors]
    bank = get("loaded_recruitment_bank")[:, pair_motors, pair_filters]
    kappa = get("loaded_parameters_kappa")[pair_motors]
    basis = np.empty((2001, N_W))
    basis[:, 0::2] = np.minimum(bank, kappa)
    basis[:, 1::2] = np.maximum(bank - kappa, 0)
    w0 = slopes[pair_muscles, pair_motors, pair_filters].reshape(-1)
    require(pair_motors.size == N_PAIR and w0.size == N_W, "pair count differs")
    return {"variable_muscles": variable, "variable_motors": motors, "singleton_muscles": singleton, "pair_motors": pair_motors, "pair_filters": pair_filters, "pair_muscles": pair_muscles, "basis": basis, "w0": w0, "group_legs": group_legs}


def metrics(data: dict[str, np.ndarray], command: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rows = data["target_indices"]
    start, end = data["affine_rows_velocity_start"], data["affine_rows_velocity_end"]
    velocity = (command[end] - command[start]) / .005
    error = command[rows] - data["target"]
    velocity_error = velocity - data["target_velocity"]
    startup_error = command[1:100] - data["loaded_startup_target_command"]
    rms = lambda array: float(np.sqrt(np.mean(array * array)))
    result = {"overall_command_RMS": rms(error), "overall_velocity_RMS": rms(velocity_error), "startup_q_RMS": rms(startup_error), "startup_q_max_abs": float(np.max(np.abs(startup_error))), "by_leg": {}}
    for leg, name in enumerate(LEGS):
        columns = slice(7 * leg, 7 * (leg + 1))
        result["by_leg"][name] = {"command_RMS": rms(error[:, columns]), "velocity_RMS": rms(velocity_error[:, columns])}
    ltm_error = data["loaded_trace_tendon"][1:100] - data["loaded_startup_target_tendon"]
    result["startup_LTM_RMS"] = rms(ltm_error)
    result["startup_LTM_max_abs"] = float(np.max(np.abs(ltm_error)))
    return result, {"command_error": error, "velocity_error": velocity_error, "startup_error": startup_error, "velocity": velocity}


def affine_candidate(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros((2001, 96))
    np.add.at(delta.T, np.repeat(layout["pair_muscles"], 2), (layout["basis"] * (w - layout["w0"])).T)
    return data["loaded_recruitment_activation"] + delta, data["loaded_trace_recorded_command"] + .02 * (delta @ data["loaded_adapter_moment"].T)


def candidate(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], w: np.ndarray) -> dict[str, np.ndarray]:
    slopes = np.array(data["loaded_parameters_slopes"], copy=True)
    slopes[layout["pair_muscles"], layout["pair_motors"], layout["pair_filters"]] = w.reshape(N_PAIR, 2)
    activation = native_activation(data, slopes, data["loaded_adapter_ltm"])
    command = native_command(data, activation)
    command[0] = data["loaded_trace_recorded_command"][0]
    return {"slopes": slopes, "activation": activation, "command": command}


def physical_gates(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], w: np.ndarray) -> dict[str, bool]:
    require(w.shape == (N_W,) and np.isfinite(w).all(), "candidate vector invalid")
    pairs = w.reshape(N_PAIR, 2)
    lower, upper = np.min(pairs, axis=1), np.max(pairs, axis=1)
    muscles = layout["pair_muscles"]
    counts = np.bincount(data["loaded_adapter_membership"], minlength=96)
    floor = 1e-6 / counts[muscles[::3]]
    low_sum = np.bincount(muscles, weights=lower, minlength=96)
    high_sum = np.bincount(muscles, weights=upper, minlength=96)
    variable = layout["variable_muscles"]
    return {"coefficient_exact_bounds": bool(np.all((w >= 0) & (w <= 2))), "mn_floor": bool(np.all(lower.reshape(343, 3).sum(axis=1) >= floor - ATOL)), "muscle_lower": bool(np.all(low_sum[variable] >= .5 - ATOL)), "muscle_upper": bool(np.all(high_sum[variable] <= 2 + ATOL))}


def proposal_gates(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], x: np.ndarray, native: dict[str, np.ndarray], actual: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    gates = physical_gates(data, layout, x[:N_W])
    gates.update(margin_bounds=bool(-1 <= x[-1] <= 1), fixed_LTM_exact=bool(np.array_equal(native["activation"][:, data["loaded_adapter_ltm"]], data["loaded_recruitment_activation"][:, data["loaded_adapter_ltm"]])), support_exact=bool(np.all(native["slopes"][np.arange(96)[:, None] != data["loaded_adapter_membership"][None, :]] == 0)), activation_nonnegative=bool(np.isfinite(native["activation"]).all() and np.all(native["activation"] >= -ATOL)), overall_command_20pct=bool(actual["overall_command_RMS"] <= .8 * baseline["overall_command_RMS"]), overall_velocity_nonincrease=bool(actual["overall_velocity_RMS"] <= baseline["overall_velocity_RMS"]), all_leg_nonincrease=bool(all(actual["by_leg"][leg][kind] <= baseline["by_leg"][leg][kind] for leg in LEGS for kind in ("command_RMS", "velocity_RMS"))), startup_q=bool(actual["startup_q_RMS"] <= .018 and actual["startup_q_max_abs"] <= .04), startup_LTM=bool(actual["startup_LTM_RMS"] <= .009 and actual["startup_LTM_max_abs"] <= .025))
    return gates


def local_affine(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], leg: int, kind: str) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    owners = np.repeat(layout["pair_muscles"], 2)
    active = np.flatnonzero(layout["group_legs"][owners] == leg)
    joints = np.arange(7 * leg, 7 * (leg + 1))
    moment = data["loaded_adapter_moment"][np.ix_(joints, owners[active])]
    if kind == "command":
        rows = data["target_indices"]
        values = layout["basis"][rows][:, active]
        anchor = data["loaded_trace_recorded_command"][np.ix_(rows, joints)] - data["target"][:, joints]
    elif kind == "velocity":
        start, end = data["affine_rows_velocity_start"], data["affine_rows_velocity_end"]
        values = (layout["basis"][end][:, active] - layout["basis"][start][:, active]) / .005
        anchor = (data["loaded_trace_recorded_command"][end][:, joints] - data["loaded_trace_recorded_command"][start][:, joints]) / .005 - data["target_velocity"][:, joints]
    elif kind == "startup":
        rows = np.arange(1, 100)
        values = layout["basis"][rows][:, active]
        anchor = data["loaded_trace_recorded_command"][np.ix_(rows, joints)] - data["loaded_startup_target_command"][:, joints]
    else:
        raise ValueError("unknown block kind")
    local = sparse.csr_matrix((.02 * values[:, None, :] * moment[None, :, :]).reshape(anchor.size, active.size))
    return local, anchor.reshape(-1) - local @ layout["w0"][active], active


def pack_sparse(prefix: str, matrix: sparse.spmatrix, raw: dict[str, np.ndarray]) -> None:
    compressed = matrix.tocsc()
    raw[prefix + "_data"] = compressed.data
    raw[prefix + "_indices"] = compressed.indices
    raw[prefix + "_indptr"] = compressed.indptr
    raw[prefix + "_shape"] = np.asarray(compressed.shape)


def build_problem(data: dict[str, np.ndarray], layout: dict[str, np.ndarray], baseline: dict[str, Any], raw: dict[str, np.ndarray]) -> tuple[sparse.csc_matrix, np.ndarray, sparse.csc_matrix, np.ndarray, list[Any], np.ndarray]:
    blocks: dict[str, tuple[sparse.csr_matrix, np.ndarray]] = {}
    startup_parts = []
    startup_offsets = []
    witnesses = np.stack((layout["w0"], np.zeros(N_W), .5 + .5 * np.cos(np.arange(N_W) * .173)))
    for leg, name in enumerate(LEGS):
        for kind in ("command", "velocity", "startup"):
            direct, offset, active = local_affine(data, layout, leg, kind)
            r = np.linalg.qr(np.column_stack((direct.toarray(), offset)), mode="reduced")[1]
            selector = sparse.csr_matrix((np.ones(active.size), (np.arange(active.size), active)), shape=(active.size, N_W))
            compressed = sparse.csr_matrix(r[:, :-1]) @ selector
            compressed_offset = r[:, -1]
            direct_norms = np.asarray([np.linalg.norm(direct @ w[active] + offset) for w in witnesses])
            compressed_norms = np.asarray([np.linalg.norm(compressed @ w + compressed_offset) for w in witnesses])
            key = kind + "_" + name
            raw["qr_" + key] = r
            raw["qr_" + key + "_active"] = active
            raw["qr_" + key + "_offset"] = offset
            raw["qr_" + key + "_direct_norms"] = direct_norms
            raw["qr_" + key + "_compressed_norms"] = compressed_norms
            require(np.all(np.abs(direct_norms - compressed_norms) <= ATOL + 1e-10 * direct_norms), "QR norm mismatch: " + key)
            blocks[key] = compressed, compressed_offset
            if kind == "startup":
                startup_parts.append(direct @ selector)
                startup_offsets.append(offset)
    require(len(blocks) == 18, "QR block count differs")
    startup_matrix = sparse.vstack(startup_parts, format="csr")
    startup_offset = np.concatenate(startup_offsets)

    criteria: list[tuple[str, sparse.csr_matrix, np.ndarray, float]] = []

    def add(name: str, keys: list[str], limit: float) -> None:
        require(np.isfinite(limit) and limit > 0, "nonpositive SOC limit: " + name)
        criteria.append((name, sparse.vstack([blocks[key][0] for key in keys], format="csr"), np.concatenate([blocks[key][1] for key in keys]), float(limit)))

    add("overall_command", ["command_" + leg for leg in LEGS], math.sqrt(1824 * 42) * .8 * baseline["overall_command_RMS"])
    add("overall_velocity", ["velocity_" + leg for leg in LEGS], math.sqrt(1823 * 42) * baseline["overall_velocity_RMS"])
    for leg in LEGS:
        add(leg + "_command", ["command_" + leg], math.sqrt(1824 * 7) * baseline["by_leg"][leg]["command_RMS"])
        add(leg + "_velocity", ["velocity_" + leg], math.sqrt(1823 * 7) * baseline["by_leg"][leg]["velocity_RMS"])
    add("startup", ["startup_" + leg for leg in LEGS], math.sqrt(99 * 42) * .018)
    require(len(criteria) == 15, "SOC count differs")

    guard = 1e-8
    lower = np.r_[np.full(N_W, guard), np.zeros(2 * N_PAIR), -1.]
    upper = np.r_[np.full(N_W, 2 - guard), np.full(2 * N_PAIR, 2.), 1.]
    eye = sparse.eye(N_X, format="csr")
    rows = [-eye, eye]
    rhs = [-lower, upper]
    p = np.arange(N_PAIR)
    low, high, ell, u = 2 * p, 2 * p + 1, N_W + p, N_W + N_PAIR + p
    erows = np.repeat(np.arange(4 * N_PAIR), 2)
    ecols = np.concatenate([np.column_stack((ell, low)).reshape(-1), np.column_stack((ell, high)).reshape(-1), np.column_stack((low, u)).reshape(-1), np.column_stack((high, u)).reshape(-1)])
    epigraph = sparse.csr_matrix((np.tile([1., -1.], 4 * N_PAIR), (erows, ecols)), shape=(4 * N_PAIR, N_X))
    rows.append(epigraph)
    rhs.append(np.full(4 * N_PAIR, -guard))
    mn_floor = sparse.csr_matrix((-np.ones(N_PAIR), (np.repeat(np.arange(343), 3), ell)), shape=(343, N_X))
    counts = np.bincount(data["loaded_adapter_membership"], minlength=96)
    floor = 1e-6 / counts[layout["pair_muscles"][::3]]
    rows.append(mn_floor)
    rhs.append(-floor - guard)
    muscle_index = np.searchsorted(layout["variable_muscles"], layout["pair_muscles"])
    require(np.array_equal(layout["variable_muscles"][muscle_index], layout["pair_muscles"]), "muscle map differs")
    rows.extend((sparse.csr_matrix((-np.ones(N_PAIR), (muscle_index, ell)), shape=(90, N_X)), sparse.csr_matrix((np.ones(N_PAIR), (muscle_index, u)), shape=(90, N_X))))
    rhs.extend((np.full(90, -.5 - guard), np.full(90, 2 - guard)))
    tcol = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    envelope_zero = sparse.csr_matrix((startup_matrix.shape[0], 2 * N_PAIR))
    rows.extend((sparse.hstack((startup_matrix / .04, envelope_zero, tcol), format="csr"), sparse.hstack((-startup_matrix / .04, envelope_zero, tcol), format="csr")))
    rhs.extend((1 - startup_offset / .04, 1 + startup_offset / .04))
    linear = sparse.vstack(rows, format="csr")
    linear_rhs = np.concatenate(rhs)
    require(linear.shape == (21189, N_X), "first cone dimensions differ")
    parts = [linear]
    right = [linear_rhs]
    cones: list[Any] = [clarabel.NonnegativeConeT(linear.shape[0])]
    names = ["NonnegativeConeT"]
    dims = [linear.shape[0]]
    for name, matrix, offset, limit in criteria:
        head = sparse.csr_matrix(([1.], ([0], [N_X - 1])), shape=(1, N_X))
        tail = sparse.hstack((-matrix / limit, sparse.csr_matrix((offset.size, N_X - N_W))), format="csr")
        soc_a = sparse.vstack((head, tail), format="csr")
        soc_b = np.r_[1., offset / limit]
        parts.append(soc_a)
        right.append(soc_b)
        cones.append(clarabel.SecondOrderConeT(soc_a.shape[0]))
        names.append(name)
        dims.append(soc_a.shape[0])
    a = sparse.vstack(parts, format="csc")
    b = np.concatenate(right)
    q = np.r_[np.zeros(N_X - 1), -1.]
    p_matrix = sparse.csc_matrix((N_X, N_X))
    counts_mn = counts[layout["pair_muscles"][::3]]
    v = .999 * layout["w0"] + np.repeat(.001 / (3 * counts_mn), 6)
    pair = v.reshape(N_PAIR, 2)
    witness = np.r_[v, np.min(pair, axis=1) - guard, np.max(pair, axis=1) + guard, -.251]
    slack = b - a @ witness
    pack_sparse("problem_A", a, raw)
    raw.update(problem_b=b, problem_q=q, cone_dims=np.asarray(dims), cone_kinds=np.asarray(names), lower=lower, upper=upper, baseline_witness=witness, baseline_witness_slack=slack, witness_v=v, startup_matrix_shape=np.asarray(startup_matrix.shape), startup_offset=startup_offset)
    require(np.min(slack[:21189]) >= -ATOL, "fixed baseline witness linear infeasible")
    start = 21189
    for name, dim in zip(names[1:], dims[1:], strict=True):
        section = slack[start:start + dim]
        require(section[0] >= np.linalg.norm(section[1:]) - ATOL, "fixed baseline witness SOC infeasible: " + name)
        start += dim
    return p_matrix, q, a, b, cones, witness


def solver_settings() -> clarabel.DefaultSettings:
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.direct_solve_method = "qdldl"
    settings.max_iter = 100
    settings.time_limit = FIT_WALL_LIMIT_S
    for key in ("tol_gap_abs", "tol_gap_rel", "tol_feas", "tol_infeas_abs", "tol_infeas_rel"):
        setattr(settings, key, 1e-10)
    return settings


def solver_arrays(solution: Any) -> dict[str, np.ndarray]:
    return {"solver_x": np.asarray(solution.x), "solver_s": np.asarray(solution.s), "solver_z": np.asarray(solution.z), "solver_status": np.asarray(str(solution.status)), "solver_iterations": np.asarray(solution.iterations), "solver_solve_time": np.asarray(solution.solve_time), "solver_obj_val": np.asarray(solution.obj_val), "solver_obj_val_dual": np.asarray(solution.obj_val_dual)}


def run(freeze_path: str | Path, freeze_sha256: str, source_review: str | Path, source_review_sha256: str, out_path: str | Path) -> dict[str, Any]:
    began = time.perf_counter()
    freeze_path = Path(freeze_path).resolve()
    source_review = Path(source_review).resolve()
    freeze, _ = validate_authority(freeze_path, freeze_sha256, source_review, source_review_sha256)
    out = Path(out_path).resolve()
    require(out.is_relative_to(ROOT) and not out.exists(), "output must be a new workspace directory")
    require(shutil.disk_usage(out.parent).free >= 2 * 1024**3, "free-space reserve unavailable")
    out.mkdir(parents=True, exist_ok=False)
    ledger: dict[str, Any] = {"schema": SCHEMA + ".ledger", "pid": os.getpid(), "started_perf_counter": began, "stage": "created", "fit_calls": 0, "solver_returns": 0, "original_processes": 1, "gpu_count": 0, "retries": 0, "cpu_count": os.cpu_count(), "python_version": sys.version, "numpy_version": np.__version__, "scipy_version": scipy.__version__, "clarabel_version": clarabel.__version__, "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    try:
        import psutil

        ledger["cpu_affinity"] = psutil.Process().cpu_affinity()
    except (ImportError, AttributeError):
        ledger["cpu_affinity"] = "unavailable"
    result: dict[str, Any] = {"schema": SCHEMA + ".result", "validity": "incomplete", "hypothesis_verdict": "invalid_not_tested", "conditional_proposal": False, "candidate_adopted": False, "whole_goal_complete": False, "F6_exposed": False}
    published_success = False
    input_raw: dict[str, np.ndarray] = {}
    pre_saved = False
    secondary_errors: list[dict[str, str]] = []

    def deadline(stage: str) -> None:
        elapsed = time.perf_counter() - began
        ledger["elapsed_s"] = elapsed
        ledger["stage"] = stage
        atom_json(out / "ledger.json", ledger)
        require(elapsed <= freeze["max_internal_s"], "internal 80-second limit exceeded")

    def check_output() -> None:
        bytes_used = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        require(bytes_used <= freeze["max_output_bytes"], "output byte limit exceeded")
        require(shutil.disk_usage(out).free >= 2 * 1024**3, "terminal free-space reserve unavailable")
        ledger["output_bytes"] = bytes_used

    try:
        with threadpool_limits(limits=1):
            ledger["threadpools_during"] = threadpool_info()
            require(all(item.get("num_threads") == 1 for item in ledger["threadpools_during"]), "CPU BLAS threads exceed one")
            data = load_input(frozen_path(freeze["input_npz"]["path"]))
            input_raw = dict(data)
            atom_npz(out / "pre-solve-raw.npz", input_raw)
            pre_saved = True
            deadline("input_saved")
            layout = validate_input(data)
            input_raw.update({"variable_muscles": layout["variable_muscles"], "variable_motors": layout["variable_motors"], "fixed_muscles": data["loaded_adapter_ltm"], "new_singleton_muscles": layout["singleton_muscles"], "pair_motors": layout["pair_motors"], "pair_filters": layout["pair_filters"], "pair_muscles": layout["pair_muscles"], "branch_basis": layout["basis"], "w0": layout["w0"]})
            recorded = data["loaded_trace_recorded_command"]
            baseline, baseline_arrays = metrics(data, recorded)
            _, affine0 = affine_candidate(data, layout, layout["w0"])
            require(np.array_equal(affine0, recorded), "w0 affine baseline differs")
            native0 = candidate(data, layout, layout["w0"])
            _, native0_arrays = metrics(data, native0["command"])
            input_raw["baseline_native_activation_residual"] = native0["activation"] - data["loaded_recruitment_activation"]
            input_raw["baseline_native_command_residual"] = native0["command"] - recorded
            input_raw["baseline_native_velocity_residual"] = native0_arrays["velocity"] - baseline_arrays["velocity"]
            atom_npz(out / "pre-solve-raw.npz", input_raw)
            pre_saved = True
            require(np.max(np.abs(input_raw["baseline_native_activation_residual"][1:])) <= ATOL and np.max(np.abs(input_raw["baseline_native_command_residual"][1:])) <= ATOL and np.max(np.abs(input_raw["baseline_native_velocity_residual"])) <= VELOCITY_ATOL, "w0 native baseline differs")
            require(all(physical_gates(data, layout, layout["w0"]).values()), "w0 physical gates fail")
            deadline("baseline_checked")
            p, q, a, b, cones, _witness = build_problem(data, layout, baseline, input_raw)
            atom_npz(out / "pre-solve-raw.npz", input_raw)
            pre_saved = True
            check_output()
            deadline("problem_saved")
            settings = solver_settings()
            ledger["fit_calls"] = 1
            deadline("solve_entered")
            solve_began = time.perf_counter()
            solution = clarabel.DefaultSolver(p, q, a, b, cones, settings).solve()
            ledger["fit_wall_s"] = time.perf_counter() - solve_began
            ledger["solver_returns"] = 1
            returned = solver_arrays(solution)
            atom_npz(out / "solver-return-raw.npz", returned)
            deadline("solver_return_saved")
            x = np.asarray(solution.x, dtype=np.float64)
            s = np.asarray(solution.s, dtype=np.float64)
            z = np.asarray(solution.z, dtype=np.float64)
            require(x.shape == (N_X,) and s.shape == b.shape and z.shape == b.shape and np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all(), "solver vector invalid")
            primal = a @ x + s - b
            dual = q + a.T @ z
            derived = {"primal_residual": primal, "dual_residual": dual, "linear_slack": b[:21189] - a[:21189] @ x, "objective_margin": np.asarray(x[-1]), "solver_slack_min": np.asarray(np.min(s)), "solver_dual_min": np.asarray(np.min(z))}
            atom_npz(out / "solver-derived-raw.npz", derived)
            native = candidate(data, layout, x[:N_W])
            affine_activation, affine_command = affine_candidate(data, layout, x[:N_W])
            actual, actual_arrays = metrics(data, native["command"])
            _, affine_arrays = metrics(data, affine_command)
            candidate_raw = {"x": x, "w": x[:N_W], "candidate_slopes": native["slopes"], "native_activation": native["activation"], "native_command": native["command"], "native_velocity": actual_arrays["velocity"], "affine_activation": affine_activation, "affine_command": affine_command, "affine_velocity": affine_arrays["velocity"], "native_affine_activation_residual": native["activation"] - affine_activation, "native_affine_command_residual": native["command"] - affine_command, "native_affine_velocity_residual": actual_arrays["velocity"] - affine_arrays["velocity"], **actual_arrays}
            atom_npz(out / "candidate-raw.npz", candidate_raw)
            deadline("candidate_saved")
            require(ledger["fit_wall_s"] <= FIT_WALL_LIMIT_S, "inclusive 20-second fit limit exceeded")
            require(str(solution.status) in ("Solved", "AlmostSolved"), "solver termination invalid")
            require(np.max(np.abs(primal)) <= SOLVER_RESIDUAL_ATOL and np.max(np.abs(dual)) <= SOLVER_RESIDUAL_ATOL and np.min(derived["linear_slack"]) >= -SOLVER_LINEAR_ATOL, "solver primal/dual residual too large")
            cone_start = 21189
            for cone in cones[1:]:
                section = b[cone_start:cone_start + cone.dim] - a[cone_start:cone_start + cone.dim] @ x
                require(section[0] >= np.linalg.norm(section[1:]) - SOLVER_RESIDUAL_ATOL, "solver SOC primal violation")
                cone_start += cone.dim
            require(np.max(np.abs(candidate_raw["native_affine_activation_residual"][1:])) <= ATOL and np.max(np.abs(candidate_raw["native_affine_command_residual"][1:])) <= ATOL and np.max(np.abs(candidate_raw["native_affine_velocity_residual"])) <= VELOCITY_ATOL, "candidate native/affine mismatch")
            gates = proposal_gates(data, layout, x, native, actual, baseline)
            proposal = all(gates.values())
            result.update(validity="valid", hypothesis_verdict="conditional_proposal" if proposal else "valid_rejection_inconclusive", conditional_proposal=proposal, solver_status=str(solution.status), margin=float(x[-1]), baseline_metrics=baseline, candidate_metrics=actual, gates=gates, numerical_residuals={"primal_max_abs": float(np.max(np.abs(primal))), "dual_max_abs": float(np.max(np.abs(dual))), "native_affine_velocity_max_abs": float(np.max(np.abs(candidate_raw["native_affine_velocity_residual"])))})
            deadline("candidate_checked")
        validate_authority(freeze_path, freeze_sha256, source_review, source_review_sha256)
        check_output()
        deadline("hashes_rechecked")
        manifest = {path.relative_to(out).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in out.rglob("*") if path.is_file() and path.suffix == ".npz"}
        atom_json(out / "output-manifest.json", {"schema": SCHEMA + ".manifest", "files": manifest})
        require(all(sha256(out / name) == receipt["sha256"] and (out / name).stat().st_size == receipt["bytes"] for name, receipt in manifest.items()), "immutable output manifest verification failed")
        check_output()
        deadline("manifest_saved")
        atom_json(out / "result.json", result)
        published_success = True
        require(json.loads((out / "result.json").read_text(encoding="utf-8")) == result, "terminal result readback differs")
        check_output()
        deadline("complete")
    except BaseException as error:
        result.update(validity="incomplete", hypothesis_verdict="invalid_not_tested", conditional_proposal=False, technical_failure=True, primary_error={"type": type(error).__name__, "message": str(error), "traceback": "".join(traceback.format_exception(error))})
        ledger.update(stage="technical_failed", elapsed_s=time.perf_counter() - began)
        if input_raw and not pre_saved:
            try:
                atom_npz(out / "pre-solve-raw.npz", input_raw)
            except Exception as save_error:  # noqa: BLE001
                secondary_errors.append({"stage": "pre_solve_fallback", "error": repr(save_error)})
        elif input_raw and pre_saved:
            try:
                atom_npz(out / "pre-solve-raw.npz", input_raw)
            except Exception as save_error:  # noqa: BLE001
                secondary_errors.append({"stage": "updated_pre_solve_fallback", "error": repr(save_error)})
        result["secondary_errors"] = secondary_errors
        ledger["secondary_errors"] = secondary_errors
        (out / "output-manifest.json").unlink(missing_ok=True)
        if published_success:
            (out / "result.json").unlink(missing_ok=True)
        try:
            atom_json(out / "ledger.json", ledger)
            atom_json(out / "result.json", result)
        except Exception as publication_error:  # noqa: BLE001
            (out / "result.json").unlink(missing_ok=True)
            error.add_note("failure publication also failed: " + repr(publication_error))
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    result = run(arguments.freeze, arguments.freeze_sha256, arguments.source_review, arguments.source_review_sha256, arguments.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
