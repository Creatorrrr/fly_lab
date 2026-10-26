"""E232 saved-only effector command and six-dimensional memory shift audit."""

# Retain raw aligned inputs and every available intermediate on failure.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from tools import fit_c_banc_m3_grouped_response as grouped
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e232-effector-distribution-shift"
FREEZE_SHA = "3678adb09b66be0145098213de7d43cecd08e1c5b4cd53550fc719ceadd06d96"
FREEZE_SCHEMA = "flylab.m3.e232.effector-distribution-shift.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e232.effector-distribution-shift.review.v1"
OWNED = ("tools/analyze_c_banc_m3_effector_distribution_shift.py",
         "tests/test_c_banc_effector_distribution_shift.py")
CONTROL_KEYS = prior.COMMON_FIELDS + prior.EFFECTOR_FIELDS
RATE_PARTS = ("free_dynamics", "quantization", "fixed")
COMMAND_PARTS = (*RATE_PARTS, "initial_activation")


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
        raise ValueError("Repository-relative E232 input required")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E232 freeze and parent source-review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or freeze.get("controls") != {"prefix": [0, 600],
                "suffix": [600, 1663], "seam": 600, "count": 1664,
                "training_count": 2001, "startup": [1, 100],
                "early": [100, 601], "continuation": [601, 1664]}
            or freeze.get("numerics") != {"dt_s": .005,
                "muscle_tau_s": .01, "joint_scale": .02,
                "command_atol": 2e-10, "first_tick_history_atol": 1e-8,
                "first_tick_effector_atol": 1e-5,
                "first_tick_command_atol": 1e-7,
                "projected_coverage_atol": 1e-10,
                "nearest_chunk_max": 32, "startup_rms_max_rad": .018,
                "startup_abs_max_rad": .04}
            or freeze.get("resource_limits") != {"cpu_threads": 1,
                "gpu_count": 0, "original_attempts": 1, "retries": 0,
                "body_constructors": 0, "new_neural_ticks": 0,
                "new_physics_steps": 0, "fit_calls": 0, "solver_calls": 0,
                "internal_s": 25., "finalization_reserve_s": 5.,
                "external_s": 30., "supervisor_s": 35.,
                "resident_bytes": 2147483648, "output_bytes": 134217728,
                "minimum_free_bytes": 1073741824}
            or freeze.get("previous_cumulative_limit_s") != 6104
            or freeze.get("allocation_increase_s") != 10
            or freeze.get("original_cumulative_s") != 6076.219747399653
            or freeze.get("original_cumulative_limit_s") != 6114
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E232 fixed saved-data/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Parent-approved E232 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E232 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E232 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E232 hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E232 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Frozen E232 package version changed: " + package)
    if output != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("Exact new E232 output directory required")
    if shutil.disk_usage(output.parent).free < freeze["resource_limits"][
            "minimum_free_bytes"]:
        raise OSError("E232 output disk reserve unavailable")
    return freeze, review


def load_npz_members(path, members):
    """Read only requested NPZ entries; never materialize E217's full B."""
    with np.load(path, allow_pickle=False) as archive:
        if not set(members).issubset(archive.files):
            raise ValueError("E232 frozen NPZ members missing: " + str(path))
        return {name: archive[name].copy() for name in members}


def seam_merge(prefix, suffix):
    """Control600 is identical on every native and readout field, then omitted."""
    if set(prefix) != set(CONTROL_KEYS) or set(suffix) != set(CONTROL_KEYS):
        raise ValueError("E232 actual control raw key set changed")
    shapes = {"control_tick": (), "network_tick": (), "physics_time_s": (),
        "motor_rate32": (391,), "decoder_input32": (391,),
        "activation": (96,), "command": (42,), "tendon": (6,),
        "free_effector64": (343,), "h_fast": (343, 3), "h_slow": (343, 3)}
    joined = {}
    for key in CONTROL_KEYS:
        left, right = prefix[key], suffix[key]
        dtype = (np.int64 if key in ("control_tick", "network_tick") else
                 np.float32 if key in ("motor_rate32", "decoder_input32") else
                 np.float64)
        if (left.shape != (601, *shapes[key])
                or right.shape != (1064, *shapes[key])
                or left.dtype != dtype or right.dtype != dtype
                or left[-1].tobytes() != right[0].tobytes()):
            raise ValueError("E232 control600 seam or schema differs: " + key)
        joined[key] = np.concatenate((left, right[1:]), axis=0)
    ticks = np.arange(1664)
    if (not np.array_equal(joined["control_tick"], ticks)
            or not np.array_equal(joined["network_tick"], ticks * 20)
            or not np.allclose(joined["physics_time_s"], ticks * .005,
                               atol=2e-8, rtol=0)
            or any(not np.isfinite(joined[key]).all() for key in CONTROL_KEYS
                   if key not in ("control_tick", "network_tick"))):
        raise ValueError("E232 joined actual control clocks/values invalid")
    return joined


def check_ownership(parameters, prototypes, conditional, start, actual,
                    candidate_beta):
    """Verify original ordered MN and canonical anatomy before comparisons."""
    ids = parameters["motor_ids"]
    free = parameters["free_columns"]
    fixed = parameters["fixed_columns"]
    membership = parameters["membership"]
    if (ids.shape != (391,) or free.shape != (343,) or fixed.shape != (48,)
            or membership.shape != (391,) or parameters["pool"].shape != (96, 391)
            or parameters["moment"].shape != (42, 96)
            or parameters["rest"].shape != (42,)
            or parameters["ltm"].shape != (6,)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.array_equal(np.unique(ids), np.sort(ids))
            or np.any((membership < 0) | (membership >= 96))
            or prototypes["centers"].shape != (391, 64, 9)
            or prototypes["sigma2"].shape != (391,)
            or np.any(prototypes["sigma2"] <= 0)
            or not np.isfinite(prototypes["centers"]).all()
            or not np.isfinite(prototypes["sigma2"]).all()):
        raise ValueError("E232 original 391/343/48 anatomical ownership changed")
    patch = start["neural_current_trial_patch"]
    native = start["network"]["neural_current_effector"]
    calibration = start["adapter"]["canonical"]["calibration"]
    for key in ("motor_ids", "free_columns", "fixed_columns", "membership",
                "pool", "moment", "rest", "ltm"):
        exact(parameters[key], patch["parameters"][key], "E232 patch." + key)
    for key in ("motor_ids", "pool", "moment", "rest", "ltm"):
        exact(parameters[key], calibration[key], "E232 calibration." + key)
    for key in ("motor_ids", "free_ids", "fixed_ids", "membership"):
        expected = (ids[free] if key == "free_ids" else ids[fixed]
                    if key == "fixed_ids" else parameters[key])
        exact(expected, native[key], "E232 native." + key)
    for key in ("centers", "sigma2"):
        exact(prototypes[key], patch["prototypes"][key], "E232 prototype." + key)
    exact(conditional["beta"], candidate_beta, "E232 frozen E218 beta")
    exact(candidate_beta, patch["beta"], "E232 actual candidate beta")
    exact(start["adapter"]["canonical"]["activation"], actual[
        "activation"][0], "E232 actual initial activation")
    if (not np.array_equal(actual["decoder_input32"][:, free], actual[
            "free_effector64"].astype(np.float32))
            or not np.array_equal(actual["decoder_input32"][:, fixed],
                                  actual["motor_rate32"][:, fixed])):
        raise ValueError("E232 actual free cast/fixed original readout differs")
    return free, fixed


def ema_decompose(actual, conditional, parameters):
    """Linearly propagate four differences through original 10 ms EMA."""
    free, fixed = parameters["free_columns"], parameters["fixed_columns"]
    pool, moment = parameters["pool"], parameters["moment"]
    rate = np.asarray(conditional["rate"][:1664], np.float64)
    if (rate.shape != (1664, 391)
            or conditional["q_direct"].shape != (1664, 42)
            or conditional["activation"].shape != (1664, 96)
            or not np.isfinite(rate).all()):
        raise ValueError("E232 conditional control readout shape/nonfinite")
    delta = {
        "free_dynamics": actual["free_effector64"] - rate[:, free],
        "quantization": (actual["decoder_input32"][:, free].astype(np.float64)
                         - actual["free_effector64"]),
        "fixed": (actual["decoder_input32"][:, fixed].astype(np.float64)
                  - rate[:, fixed])}
    alpha = -np.expm1(-.005 / .01)
    activation = {}
    command = {}
    for name in RATE_PARTS:
        columns = free if name != "fixed" else fixed
        pooled = delta[name] @ pool[:, columns].T
        active = np.zeros(96, np.float64)
        activation[name] = np.zeros((1664, 96), np.float64)
        command[name] = np.zeros((1664, 42), np.float64)
        for tick in range(1, 1664):
            active += alpha * (pooled[tick] - active)
            activation[name][tick] = active
            command[name][tick] = .02 * (moment @ active)
    initial_delta = (actual["activation"][0] -
                     conditional["activation"][0])
    activation["initial_activation"] = np.empty((1664, 96), np.float64)
    activation["initial_activation"][0] = initial_delta
    command["initial_activation"] = np.zeros((1664, 42), np.float64)
    for tick in range(1, 1664):
        activation["initial_activation"][tick] = (
            (1 - alpha) * activation["initial_activation"][tick - 1])
        command["initial_activation"][tick] = .02 * (moment @
            activation["initial_activation"][tick])
    measured = actual["command"] - conditional["q_direct"]
    composed = sum(command.values())
    residual = measured - composed
    return {"rate_delta_" + key: value for key, value in delta.items()} | {
        "command_component_" + key: value for key, value in command.items()} | {
        "activation_component_" + key: value for key, value in activation.items()} | {
        "actual_minus_conditional_command": measured,
        "reconstruction_residual": residual,
        "initial_activation_delta": initial_delta}


def projected_nearest(memory, centers, sigma2, *, chunk=32, check=None):
    """Exact 6D squared distance with first-index tie semantics; no 9D fill."""
    memory = np.asarray(memory, np.float64)
    centers = np.asarray(centers, np.float64)
    sigma2 = np.asarray(sigma2, np.float64)
    if (memory.ndim != 3 or memory.shape[1:] != (len(sigma2), 6)
            or centers.shape != (len(sigma2), 64, 6)
            or not 1 <= chunk <= 32 or not np.isfinite(memory).all()
            or not np.isfinite(centers).all() or not np.isfinite(sigma2).all()
            or np.any(sigma2 <= 0)):
        raise ValueError("E232 fixed finite 6D/64-center geometry required")
    nearest = np.empty(memory.shape[:2], np.int64)
    distance = np.empty(memory.shape[:2], np.float64)
    for first in range(0, len(memory), chunk):
        if check is not None:
            check()
        stop = min(first + chunk, len(memory))
        delta = memory[first:stop, :, None, :] - centers[None, :, :, :]
        d = np.sum(delta * delta, axis=-1) / (2 * sigma2[None, :, None])
        nearest[first:stop] = np.argmin(d, axis=2)
        distance[first:stop] = np.take_along_axis(d,
            nearest[first:stop, :, None], axis=2)[:, :, 0]
    return distance, nearest


def startup_metrics(command, target, limits):
    """Fixed E197 controls1..99; no perturbation or physical claim."""
    errors = np.asarray(command, np.float64)[1:100] - target[1:100]
    if errors.shape != (99, 42) or not np.isfinite(errors).all():
        raise ValueError("E232 finite 99x42 startup errors required")
    at = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
    rms = float(np.sqrt(np.mean(errors * errors)))
    maximum = float(np.abs(errors[at]))
    return {"rms_rad": rms, "max_abs_rad": maximum,
        "max_abs_control": int(at[0] + 1), "max_abs_joint_column": int(at[1]),
        "passes_original_limits": bool(rms <= limits["startup_rms_max_rad"]
                                       and maximum <= limits["startup_abs_max_rad"])}


def first_exceed(series, threshold):
    locations = np.flatnonzero(np.asarray(series) > threshold)
    return int(locations[0]) if len(locations) else None


def save_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


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
    limits, numerics = freeze["resource_limits"], freeze["numerics"]
    ledger = {"schema": "flylab.m3.e232.effector-distribution-shift.ledger.v1",
        "phase": "gate_passed", "saved_data_analysis_jobs_requested": 1,
        "saved_data_analysis_jobs_completed": 0, "retries": 0,
        "body_constructors": 0, "new_neural_ticks": 0, "new_physics_steps": 0,
        "cuda_calls": 0, "fit_calls": 0, "solver_calls": 0,
        "control_rows_analyzed": 0, "training_rows_analyzed": 0,
        "nearest_chunk_rows_max": 0, "raw_artifacts": [],
        "phase_wall_s": {}, "storage_errors": []}
    result = {"schema": "flylab.m3.e232.effector-distribution-shift.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "saved_signal_decomposition_only": True,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "causal_feedback_attribution": False,
        "nine_dimensional_feature_observed": False,
        "new_full_circuit_or_body_execution": False}
    primary = None
    process = None

    def phase(name):
        ledger["phase"] = name
        ledger["phase_wall_s"][name] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def budget():
        elapsed = time.perf_counter() - began
        if elapsed >= limits["internal_s"] - limits["finalization_reserve_s"]:
            raise TimeoutError("E232 finalization reserve reached")
        if process is not None:
            status = process.memory_info()
            peak = getattr(status, "peak_wset", status.rss)
            ledger["peak_resident_bytes"] = max(
                ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E232 resident memory cap exceeded")

    def retain(name, arrays):
        save_npz(out / name, arrays)
        ledger["raw_artifacts"].append(name)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name,
                                             "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E232 " + name + ": " + repr(error))

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        import psutil
        from threadpoolctl import threadpool_limits
        process = psutil.Process()
        with threadpool_limits(limits=1):
            paths = {key: safe(name) for key, name in freeze["paths"].items()}
            phase("load_pinned_members")
            prefix = load_npz_members(paths["actual_prefix"], CONTROL_KEYS)
            suffix = load_npz_members(paths["actual_suffix"], CONTROL_KEYS)
            parameters = load_npz_members(paths["parameters"], (
                "motor_ids", "free_columns", "fixed_columns", "membership",
                "pool", "moment", "rest", "ltm"))
            prototypes = load_npz_members(paths["prototypes"], (
                "centers", "sigma2", "membership"))
            conditional = load_npz_members(paths["conditional"], (
                "beta", "rate", "q_direct", "activation", "tendon"))
            candidate = load_npz_members(paths["candidate"], ("beta",))
            startup = load_npz_members(paths["startup"], ("recorded_command",))
            basis = load_npz_members(paths["basis"], ("h_fast", "h_slow"))
            start = StateStore.load(paths["actual_start"])
            budget()
            actual = seam_merge(prefix, suffix)
            cond = {key: value[:1664].copy() if value.ndim else value
                    for key, value in conditional.items()}
            if (conditional["rate"].shape != (2001, 391)
                    or conditional["q_direct"].shape != (2001, 42)
                    or conditional["activation"].shape != (2001, 96)
                    or conditional["tendon"].shape != (2001, 6)
                    or startup["recorded_command"].shape != (2001, 42)
                    or basis["h_fast"].shape != (2001, 391, 3)
                    or basis["h_slow"].shape != (2001, 391, 3)
                    or cond["beta"].shape != (96, 64)
                    or not all(np.isfinite(value).all() for value in cond.values())
                    or not np.isfinite(startup["recorded_command"]).all()
                    or not np.isfinite(basis["h_fast"]).all()
                    or not np.isfinite(basis["h_slow"]).all()):
                raise ValueError("E232 frozen conditional/startup/history shape differs")
            free, _fixed = check_ownership(parameters, prototypes, cond,
                                           start, actual, candidate["beta"])
            exact(parameters["membership"], prototypes["membership"],
                  "E232 prototype owner membership")
            ledger["control_rows_analyzed"] = 1664
            phase("aligned_raw")
            retain("aligned-control.npz", actual)
            retain("conditional-aligned.npz", {key: cond[key] for key in (
                "rate", "q_direct", "activation", "tendon")})
            budget()
            phase("canonical_readout")
            groups = start["adapter"]["canonical"]["groups"]
            actual_readout = grouped.canonical_group_readout(
                actual["decoder_input32"].astype(np.float64),
                parameters["pool"], parameters["moment"], parameters["rest"],
                parameters["ltm"], groups, actual["activation"][0],
                actual["command"][0])
            conditional_readout = grouped.canonical_group_readout(
                cond["rate"], parameters["pool"], parameters["moment"],
                parameters["rest"], parameters["ltm"], groups,
                cond["activation"][0], cond["q_direct"][0])
            checks = {}
            for prefix_name, measured, rebuilt in (
                    ("actual", actual, actual_readout),
                    ("conditional", {"command": cond["q_direct"],
                        "activation": cond["activation"],
                        "tendon": cond["tendon"]}, conditional_readout)):
                for key in ("command", "activation", "tendon"):
                    error = float(np.max(np.abs(measured[key] - rebuilt[key])))
                    checks[prefix_name + "_" + key + "_max_abs"] = error
                    if error > numerics["command_atol"]:
                        raise ValueError("E232 original canonical " +
                                         prefix_name + "." + key + " differs")
            if actual["command"][0].tobytes() != cond["q_direct"][0].tobytes():
                raise ValueError("E232 held control0 command changed")
            phase("four_component_raw")
            decomposition = ema_decompose(actual, cond, parameters)
            retain("command-decomposition.npz", decomposition)
            residual_max = float(np.max(np.abs(
                decomposition["reconstruction_residual"][1:])))
            checks["four_component_command_residual_max_abs"] = residual_max
            if residual_max > numerics["command_atol"]:
                raise ValueError("E232 four-component command difference does not close")
            activation_composed = sum(decomposition[
                "activation_component_" + name] for name in COMMAND_PARTS)
            activation_error = float(np.max(np.abs((actual["activation"] -
                cond["activation"]) - activation_composed)))
            checks["four_component_activation_residual_max_abs"] = activation_error
            if activation_error > numerics["command_atol"]:
                raise ValueError("E232 four-component activation difference differs")
            budget()
            phase("memory_shift_raw")
            h_fast_delta = actual["h_fast"] - basis["h_fast"][:1664, free]
            h_slow_delta = actual["h_slow"] - basis["h_slow"][:1664, free]
            h_fast_max = np.max(np.abs(h_fast_delta), axis=(1, 2))
            h_slow_max = np.max(np.abs(h_slow_delta), axis=(1, 2))
            memory_max = np.maximum(h_fast_max, h_slow_max)
            effector_max = np.max(np.abs(decomposition[
                "rate_delta_free_dynamics"]), axis=1)
            q_max = np.max(np.abs(decomposition[
                "actual_minus_conditional_command"]), axis=1)
            retain("memory-shift.npz", {"h_fast_delta": h_fast_delta,
                "h_slow_delta": h_slow_delta, "h_fast_max_abs": h_fast_max,
                "h_slow_max_abs": h_slow_max, "memory_max_abs": memory_max,
                "effector_max_abs": effector_max, "command_max_abs": q_max})
            phase("projected_training_raw")
            center6 = prototypes["centers"][free, :, 3:9]
            sigma2 = prototypes["sigma2"][free]
            train_memory = np.concatenate((basis["h_fast"][:, free],
                                           basis["h_slow"][:, free]), axis=2)
            train_distance, train_nearest = projected_nearest(train_memory,
                center6, sigma2, chunk=numerics["nearest_chunk_max"],
                check=budget)
            train_max = np.max(train_distance, axis=0)
            retain("projected-training.npz", {
                "train_projected_distance": train_distance,
                "train_nearest_index": train_nearest,
                "train_projected_max": train_max})
            ledger["training_rows_analyzed"] = 2001
            ledger["nearest_chunk_rows_max"] = numerics["nearest_chunk_max"]
            budget()
            phase("projected_actual_raw")
            actual_memory = np.concatenate((actual["h_fast"],
                                            actual["h_slow"]), axis=2)
            distance, nearest = projected_nearest(actual_memory, center6,
                sigma2, chunk=numerics["nearest_chunk_max"], check=budget)
            exceed = distance > train_max[None, :] + numerics[
                "projected_coverage_atol"]
            retain("projected-nearest.npz", {
                "train_projected_distance": train_distance,
                "train_nearest_index": train_nearest,
                "train_projected_max": train_max,
                "projected_distance": distance, "nearest_index": nearest,
                "exceeds_training": exceed})
            budget()
            phase("summary")
            target = startup["recorded_command"]
            startup_report = {"actual": startup_metrics(actual["command"],
                target, numerics), "conditional": startup_metrics(
                    cond["q_direct"], target, numerics)}
            for name in COMMAND_PARTS:
                component = decomposition["command_component_" + name]
                standalone = component[1:100]
                location = np.unravel_index(np.argmax(np.abs(standalone)),
                                              standalone.shape)
                startup_report[name] = {"component_rms_rad": float(np.sqrt(
                    np.mean(standalone * standalone))),
                    "component_max_abs_rad": float(np.abs(standalone[location])),
                    "component_max_control": int(location[0] + 1),
                    "component_max_joint_column": int(location[1]),
                    "actual_with_component_removed": startup_metrics(
                        actual["command"] - component, target, numerics)}
            regions = {"row0": (0, 1), "first_5ms": (1, 2),
                "startup": (1, 100), "early": (100, 601),
                "continuation": (601, 1664)}
            region_report = {name: {"controls": [first, stop],
                "h_fast_max_abs": float(np.max(h_fast_max[first:stop])),
                "h_slow_max_abs": float(np.max(h_slow_max[first:stop])),
                "free_effector_max_abs": float(np.max(
                    effector_max[first:stop])),
                "command_max_abs": float(np.max(q_max[first:stop])),
                "projected_exceed_count": int(np.count_nonzero(
                    exceed[first:stop]))} for name, (first, stop) in
                regions.items()}
            first_tick_pass = bool(memory_max[1] <= numerics[
                "first_tick_history_atol"] and effector_max[1] <= numerics[
                "first_tick_effector_atol"] and q_max[1] <= numerics[
                "first_tick_command_atol"])
            threshold_crossings = {
                "memory_control": first_exceed(memory_max, numerics[
                    "first_tick_history_atol"]),
                "free_effector_control": first_exceed(effector_max, numerics[
                    "first_tick_effector_atol"]),
                "command_control": first_exceed(q_max, numerics[
                    "first_tick_command_atol"])}
            precision_removed_pass = startup_report["quantization"][
                "actual_with_component_removed"]["passes_original_limits"]
            later_shift = any(value is not None and value > 1 for value in
                              threshold_crossings.values())
            direction = ("inspect_initial_implementation_or_input_order"
                if not first_tick_pass else "precision_path_next_diagnostic"
                if precision_removed_pass else "capture_actual_stage_inputs_next"
                if later_shift else "unresolved_saved_signal_difference")
            summary = {"schema":
                "flylab.m3.e232.effector-distribution-shift.summary.v1",
                "control_rows": 1664, "conditional_training_rows": 2001,
                "seam_control": 600, "row0_held_command_separate": True,
                "first_tick_within_diagnostic_tolerance": first_tick_pass,
                "first_threshold_exceed_control": threshold_crossings,
                "regions": region_report, "startup": startup_report,
                "canonical_and_component_checks": checks,
                "projected_six_dimensional": {
                    "coordinates": [3, 9], "center_count_per_MN": 64,
                    "nearest_tie_policy": "first_index",
                    "scope": "same per-MN center/sigma metric compared with observed training maximum",
                    "actual_exceeding_cells": int(np.count_nonzero(exceed)),
                    "actual_exceeding_control_count": int(np.count_nonzero(
                        np.any(exceed, axis=1))),
                    "actual_exceeding_MN_count": int(np.count_nonzero(
                        np.any(exceed, axis=0))),
                    "first_exceed_control": (int(np.flatnonzero(np.any(
                        exceed, axis=1))[0]) if np.any(exceed) else None)},
                "diagnostic_direction": direction,
                "limitations": ["same control index is not same gait phase",
                    "native E/I/drive stages unobserved in actual walking",
                    "6D nearest-distance exceedance does not prove support or convex-hull exit",
                    "6D projected distance is not a full 9D probability",
                    "component removal is algebraic, not feedback/body intervention"],
                "candidate_adopted": False, "F6_exposed": False}
            prior.write_receipt(out / "summary.json", summary)
            result.update(validity="valid_saved_signal_decomposition",
                summary=summary)
            ledger["saved_data_analysis_jobs_completed"] = 1
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
        ledger["output_bytes_before_final_receipts"] = sum(
            path.stat().st_size for path in out.rglob("*") if path.is_file())
        if ledger["wall_s"] > limits["internal_s"] and primary is None:
            primary = TimeoutError("E232 internal wall cap exceeded")
            result["validity"] = "resource_limited"
        if ledger["output_bytes_before_final_receipts"] > limits[
                "output_bytes"] and primary is None:
            primary = OSError("E232 output cap exceeded")
            result["validity"] = "technical_storage_failure"
        if primary is not None and result["validity"] == (
                "valid_saved_signal_decomposition"):
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", lambda: prior.write_receipt(
            out / "advance-ledger.json", ledger))
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", manifest(out)))
        if primary is not None and result["validity"] == (
                "valid_saved_signal_decomposition"):
            result.update(validity="technical_storage_failure",
                          execution_status="failed", primary_error=repr(primary))
            attempt("result.json/invalidated", lambda: prior.write_receipt(
                out / "result.json", result))
            attempt("output-manifest.json/invalidated", lambda: prior.write_receipt(
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
