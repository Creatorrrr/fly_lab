"""E236: two bounded CPU replays of frozen effector response stage inputs."""

# Retain the first failure while independently publishing partial raw receipts.
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e236-saved-effector-input-replay"
FREEZE_SHA = "33853d0c54d6ead39185f98f2d8aff41f14830241ba3be8f96b47f858765737f"
FREEZE_SCHEMA = "flylab.m3.e236.saved-effector-input-replay.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e236.saved-effector-input-replay.review.v1"
OWNED = ("tools/replay_c_banc_m3_saved_effector_inputs.py",
         "tests/test_c_banc_saved_effector_inputs.py")
CHANNELS = ("positive_recurrent", "negative_magnitude", "drive")
DT = .00025
TAU_FAST = .016
INTERNAL_S = 75.
RESERVE_S = 10.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    path = Path(relative)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(ROOT):
        raise ValueError("Repository-relative E236 frozen path required")
    return resolved


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E236 freeze and approved review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    native = freeze.get("native", {})
    limits = freeze.get("resource_limits", {})
    tolerance = freeze.get("tolerances", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or sha256(safe(freeze["design"])) != freeze.get("design_sha256")
            or native != {"motors": 391, "free": 343, "fixed": 48,
                "muscles": 96, "centers": 64, "features": 9,
                "control_rows": 100, "stages": 7920,
                "ticks_per_arm": 1980, "arms": ["actual", "training"],
                "dt_s": DT, "control_dt_s": .005,
                "tau_fast_s": TAU_FAST, "ltm": [15, 30, 46, 63, 78, 94]}
            or tolerance != {"rate64": 2e-6, "memory64": 2e-10,
                "command": 1e-7, "tendon": 1e-7, "activation": 5e-5}
            or limits.get("cpu_threads") != 1 or limits.get("gpu_count") != 0
            or limits.get("original_attempts") != 1 or limits.get("retries") != 0
            or limits.get("neural_constructors") != 0
            or limits.get("body_constructors") != 0
            or limits.get("physics_steps") != 0
            or limits.get("fit_calls") != 0 or limits.get("solver_calls") != 0
            or limits.get("internal_s") != INTERNAL_S
            or limits.get("finalization_reserve_s") != RESERVE_S
            or limits.get("external_s") != 85 or limits.get("supervisor_s") != 95
            or limits.get("resident_bytes") != 4 << 30
            or limits.get("output_bytes") != 256 << 20
            or limits.get("minimum_free_bytes") != 2 << 30
            or freeze.get("original_cumulative_s") != 6118.154093699664
            or freeze.get("previous_cumulative_limit_s") != 6154
            or freeze.get("allocation_increase_s") != 60
            or freeze.get("original_cumulative_limit_s") != 6214
            or any(freeze.get(key) is not False for key in
                   ("candidate_adopted", "F2_complete", "F6_exposed",
                    "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("Frozen E236 math/resource/acceptance contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved E236 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Complete E236 frozen inventory required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("E236 frozen input/source changed: " + name)
    for label in ("source_sha256", "additional_reused_source_sha256",
                  "additional_input_sha256"):
        mapping = review.get(label, {})
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E236 hashes must be a mapping")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E236 input/source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E236 frozen package version changed: " + package)
    if output.resolve() != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("New exact E236 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E236 minimum free disk unavailable")
    return freeze, review


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("Required frozen E236 raw keys missing: " + str(path))
        return {key: archive[key].copy() for key in keys}


def validate_model(parameters, prototypes, beta, free_ids, initial):
    n = len(free_ids)
    if (n < 1 or free_ids.dtype != np.int64 or free_ids.ndim != 1
            or len(np.unique(free_ids)) != n
            or parameters["scale"].shape != (n, 3)
            or prototypes["centers"].shape != (n, 64, 9)
            or prototypes["sigma2"].shape != (n,)
            or beta.ndim != 2 or beta.shape[0] < 1 or beta.shape[1] != 64
            or parameters["membership"].shape != (n,)
            or any(parameters[key].shape != (n,) for key in
                   ("tau", "cap", "inverse_adaptation_tau"))
            or initial["rate"].shape != (n,)
            or any(initial[key].shape != (n, 3) for key in ("fast", "slow"))):
        raise ValueError("Owned E217/E218 response shape changed")
    values = (*(value for key, value in parameters.items()
                if key != "membership"), *prototypes.values(), beta,
              *initial.values())
    if (not all(np.asarray(value).dtype == np.float64
                and np.isfinite(value).all() for value in values)
            or parameters["membership"].dtype != np.int64
            or free_ids.min() < 0
            or np.any(parameters["scale"] <= 0)
            or np.any(prototypes["sigma2"] <= 0)
            or np.any(parameters["tau"] <= 0)
            or np.any(parameters["cap"] <= 0)
            or np.any(parameters["inverse_adaptation_tau"] <= 0)
            or np.any((beta < 0) | (beta > 1))
            or np.any((parameters["membership"] < 0)
                      | (parameters["membership"] >= len(beta)))
            or np.any(initial["rate"] < 0)
            or np.any(initial["rate"] > parameters["cap"])):
        raise ValueError("Finite original E217/E218 response parameters required")


def validate_saved_layout(params, basis, conditional, startup, e235, metadata,
                          training_headers):
    """Check full frozen source identities before a numerical replay begins."""
    motor = params["motor_ids"]
    free = params["free_columns"]
    fixed = params["fixed_columns"]
    float64 = ((params, {"scale": (391, 3), "initial_rate": (391,),
                         "tau": (391,), "cap": (391,),
                         "inverse_adaptation_tau": (391,),
                         "pool": (96, 391), "moment": (42, 96),
                         "rest": (42,)}),
               (basis, {"h_fast": (2001, 391, 3),
                        "h_slow": (2001, 391, 3)}),
               (conditional, {"rate": (2001, 391),
                              "activation": (2001, 96),
                              "q_direct": (2001, 42),
                              "tendon": (2001, 6)}),
               (startup, {"loaded_startup_target_command": (99, 42),
                          "loaded_startup_target_tendon": (99, 6)}),
               (e235, {"activation": (100, 96), "command": (100, 42),
                       "tendon": (100, 6),
                       "free_effector64": (100, 343),
                       "h_fast": (100, 343, 3),
                       "h_slow": (100, 343, 3)}))
    for archive, mapping in float64:
        for key, shape in mapping.items():
            array = archive[key]
            if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
                raise ValueError("Frozen E236 source shape/dtype/finiteness changed: " + key)
    if (motor.dtype != np.int64 or motor.shape != (391,)
            or free.dtype != np.int64 or free.shape != (343,)
            or fixed.dtype != np.int64 or fixed.shape != (48,)
            or len(np.unique(motor)) != 391
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.array_equal(np.sort(free), free)
            or not np.array_equal(np.sort(fixed), fixed)
            or e235["motor_rate32"].shape != (100, 391)
            or e235["decoder_input32"].shape != (100, 391)
            or e235["motor_rate32"].dtype != np.float32
            or e235["decoder_input32"].dtype != np.float32
            or not np.isfinite(e235["motor_rate32"]).all()
            or not np.isfinite(e235["decoder_input32"]).all()
            or metadata.get("motor_ids") != motor.tolist()
            or metadata.get("free_motor_ids") != motor[free].tolist()
            or any(array.shape != (160000, 391) or array.dtype != np.float32
                   for array in training_headers.values())
            or not np.array_equal(conditional["rate"][:100, fixed],
                                  e235["motor_rate32"][:, fixed].astype(np.float64))
            or not np.array_equal(conditional["activation"][0],
                                  e235["activation"][0])):
        raise ValueError("E235/E217/E218/E208 frozen ownership or initial layout changed")


def validate_initial_state(initial, params, basis, conditional, free):
    """Use the same recorded zero-memory/initial-rate contract for both arms."""
    if (not np.array_equal(initial["rate"], params["initial_rate"][free])
            or not np.array_equal(initial["rate"], conditional["rate"][0, free])
            or not np.array_equal(initial["fast"], basis["h_fast"][0, free])
            or not np.array_equal(initial["slow"], basis["h_slow"][0, free])
            or not np.all(initial["fast"] == 0)
            or not np.all(initial["slow"] == 0)):
        raise ValueError("Actual/training response initial state changed")


def finalize_arm_receipts(entry, state, save_partial, persist, storage_errors,
                          replay_error=None):
    """Keep a replay failure primary while trying both independent receipts."""
    entry.update(ticks_entered=state["ticks_entered"],
                 ticks_completed=state["ticks_completed"],
                 memory_ticks_completed=state["memory_ticks_completed"],
                 stage_targets_completed=state["stage_targets_completed"])
    primary = replay_error
    for label, callback in (("partial-boundary", save_partial),
                            ("advance-ledger", persist)):
        try:
            callback()
        except BaseException as error:
            storage_errors.append({"artifact": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E236 " + label + ": " + repr(error))
    return primary


def new_workspace(stages, n, initial, sample_every):
    if stages < 4 or stages % (4 * sample_every):
        raise ValueError("Complete RK/control stage count required")
    controls = stages // (4 * sample_every)
    state = {"rate_stage": np.full((stages, n), np.nan, np.float64),
        "fast_stage": np.full((stages, n, 3), np.nan, np.float64),
        "slow_stage": np.full((stages, n, 3), np.nan, np.float64),
        "rate64": np.full((controls + 1, n), np.nan, np.float64),
        "fast64": np.full((controls + 1, n, 3), np.nan, np.float64),
        "slow64": np.full((controls + 1, n, 3), np.nan, np.float64),
        "memory_ticks_completed": 0, "stage_targets_completed": 0,
        "ticks_entered": 0, "ticks_completed": 0,
        "sample_every": sample_every}
    state["rate64"][0] = initial["rate"]
    state["fast64"][0] = initial["fast"]
    state["slow64"][0] = initial["slow"]
    return state


def transform_inputs(inputs, scale):
    arrays = [np.asarray(inputs[key]) for key in CHANNELS]
    if (any(value.dtype != np.float32 or value.ndim != 2
            or value.shape != arrays[0].shape or not np.isfinite(value).all()
            for value in arrays)
            or arrays[0].shape[1] != len(scale)
            or np.any(arrays[0] < 0) or np.any(arrays[1] < 0)):
        raise ValueError("Finite aligned float32 nonnegative signed-current stages required")
    return np.stack((np.log1p(arrays[0].astype(np.float64)) / scale[:, 0],
                     np.log1p(arrays[1].astype(np.float64)) / scale[:, 1],
                     np.arcsinh(arrays[2].astype(np.float64)) / scale[:, 2]),
                    axis=2)


def integrate_memory(x, inverse_slow, state, initial, *, check=None):
    fast, slow = initial["fast"].copy(), initial["slow"].copy()
    controls_per = state["sample_every"]
    for tick in range(len(x) // 4):
        if check is not None and tick % 32 == 0:
            check()
        xf = x[4 * tick:4 * tick + 4]
        kf, ks = [], []
        for stage, multiplier in enumerate((0., DT / 2, DT / 2, DT)):
            if stage == 0:
                f_stage, s_stage = fast, slow
            else:
                f_stage = fast + multiplier * kf[-1]
                s_stage = slow + multiplier * ks[-1]
            state["fast_stage"][4 * tick + stage] = f_stage
            state["slow_stage"][4 * tick + stage] = s_stage
            kf.append((xf[stage] - f_stage) / TAU_FAST)
            ks.append((xf[stage] - s_stage) * inverse_slow[:, None])
        fast += DT / 6 * (kf[0] + 2 * kf[1] + 2 * kf[2] + kf[3])
        slow += DT / 6 * (ks[0] + 2 * ks[1] + 2 * ks[2] + ks[3])
        state["memory_ticks_completed"] = tick + 1
        if (tick + 1) % controls_per == 0:
            row = (tick + 1) // controls_per
            state["fast64"][row] = fast
            state["slow64"][row] = slow


def reaction_targets(x, state, parameters, prototypes, beta, *, chunk=8,
                     check=None):
    n = x.shape[1]
    if type(chunk) is not int or not 1 <= chunk <= 64:
        raise ValueError("Bounded reaction chunk required")
    target = np.full((len(x), n), np.nan, np.float64)
    centers = prototypes["centers"]
    sigma = prototypes["sigma2"]
    owned_beta = beta[parameters["membership"]]
    cap = parameters["cap"]
    for start in range(0, len(x), chunk):
        if check is not None:
            check()
        end = min(start + chunk, len(x))
        descriptor = np.concatenate((x[start:end], state["fast_stage"][start:end],
                                     state["slow_stage"][start:end]), axis=2)
        difference = descriptor[:, :, None, :] - centers[None, :, :, :]
        logits = -np.sum(difference * difference, axis=3) / (2 * sigma[None, :, None])
        logits -= logits.max(axis=2, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=2, keepdims=True)
        weighted = np.sum(probability * owned_beta[None, :, :], axis=2)
        gate_argument = (x[start:end, :, 0] + np.abs(x[start:end, :, 2])
                         + np.maximum(state["fast_stage"][start:end, :, 0], 0)
                         + np.abs(state["fast_stage"][start:end, :, 2]))
        gate = -np.expm1(-8 * gate_argument)
        target[start:end] = cap[None, :] * gate * weighted
        state["stage_targets_completed"] = end
    return target


def integrate_rate(target, tau, state, initial, *, check=None):
    rate = initial["rate"].copy()
    every = state["sample_every"]
    for tick in range(len(target) // 4):
        state["ticks_entered"] = tick + 1
        if check is not None and tick % 32 == 0:
            check()
        base = 4 * tick
        state["rate_stage"][base] = rate
        k1 = (target[base] - rate) / tau
        s2 = rate + DT / 2 * k1
        state["rate_stage"][base + 1] = s2
        k2 = (target[base + 1] - s2) / tau
        s3 = rate + DT / 2 * k2
        state["rate_stage"][base + 2] = s3
        k3 = (target[base + 2] - s3) / tau
        s4 = rate + DT * k3
        state["rate_stage"][base + 3] = s4
        k4 = (target[base + 3] - s4) / tau
        rate += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        state["ticks_completed"] = tick + 1
        if (tick + 1) % every == 0:
            state["rate64"][(tick + 1) // every] = rate


def replay(inputs, parameters, prototypes, beta, initial, *,
           sample_every=20, chunk=8, check=None, state=None):
    """Full two-history RK4 with externally supplied four-stage input per tick."""
    n = len(initial["rate"])
    validate_model(parameters, prototypes, beta, np.arange(n, dtype=np.int64),
                   initial)
    x = transform_inputs(inputs, parameters["scale"])
    if state is None:
        state = new_workspace(len(x), n, initial, sample_every)
    integrate_memory(x, parameters["inverse_adaptation_tau"], state,
                     initial, check=check)
    targets = reaction_targets(x, state, parameters, prototypes, beta,
                               chunk=chunk, check=check)
    integrate_rate(targets, parameters["tau"], state, initial, check=check)
    return state


def readout(rate, pool, moment, rest, ltm, activation0, held_command):
    """Original 10 ms muscle EMA, 0.02 moment, held row zero, six LTM."""
    rates = np.asarray(rate)
    rows = len(rates)
    if (rates.shape != (rows, 391) or rates.dtype not in (np.float32, np.float64)
            or pool.shape != (96, 391) or moment.shape != (42, 96)
            or rest.shape != (42,) or ltm.shape != (6,)
            or activation0.shape != (96,) or held_command.shape != (42,)
            or not np.array_equal(ltm, [15, 30, 46, 63, 78, 94])
            or not np.isfinite(rates).all() or not np.isfinite(pool).all()
            or not np.isfinite(moment).all() or not np.isfinite(rest).all()
            or not np.isfinite(activation0).all()
            or not np.isfinite(held_command).all()):
        raise ValueError("Finite original E150 canonical muscle readout required")
    activation = np.empty((rows, 96), np.float64)
    command = np.empty((rows, 42), np.float64)
    tendon = np.empty((rows, 6), np.float64)
    active = activation0.copy()
    activation[0] = active
    command[0] = held_command
    tendon[0] = .2 * np.tanh(active[ltm] / 10)
    alpha = -np.expm1(-.005 / .01)
    for control in range(1, rows):
        active += alpha * (pool @ rates[control].astype(np.float64) - active)
        activation[control] = active
        command[control] = rest + .02 * (moment @ active)
        tendon[control] = .2 * np.tanh(active[ltm] / 10)
    return {"motor_rate": rates.copy(), "activation": activation,
            "command": command, "tendon": tendon}


def startup_metrics(decoded, target_command, target_tendon):
    if (decoded["command"].shape != (100, 42)
            or decoded["tendon"].shape != (100, 6)
            or target_command.shape != (99, 42)
            or target_tendon.shape != (99, 6)
            or not all(np.isfinite(value).all() for value in
                       (decoded["command"], decoded["tendon"],
                        target_command, target_tendon))):
        raise ValueError("E197 original 99-row startup target contract changed")
    q = decoded["command"][1:] - target_command
    t = decoded["tendon"][1:] - target_tendon
    return {"command_rms_rad": float(np.sqrt(np.mean(q * q))),
            "command_max_abs_rad": float(np.max(np.abs(q))),
            "tendon_rms": float(np.sqrt(np.mean(t * t))),
            "tendon_max_abs": float(np.max(np.abs(t))),
            "command_rms_within_original": bool(np.sqrt(np.mean(q * q)) <= .018),
            "command_max_within_original": bool(np.max(np.abs(q)) <= .04),
            "tendon_rms_within_original": bool(np.sqrt(np.mean(t * t)) <= .009),
            "tendon_max_within_original": bool(np.max(np.abs(t)) <= .025)}


def error_summary(actual, reference):
    left, right = np.asarray(actual), np.asarray(reference)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Aligned finite E236 replay/reference arrays required")
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference * difference))),
            "byte_exact": bool(left.dtype == right.dtype
                               and left.tobytes() == right.tobytes())}


def save_npz(path, **arrays):
    temporary = Path(path).with_name(Path(path).name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False), encoding="utf-8")


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
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e236.saved-effector-input-replay.ledger.v1",
        "status": "gate_passed", "original_processes": 1,
        "neural_constructors": 0, "body_constructors": 0,
        "physics_steps": 0, "gpu_calls": 0, "fit_calls": 0, "solver_calls": 0,
        "retries": 0, "arms": [], "storage_errors": []}
    result = {"schema": "flylab.m3.e236.saved-effector-input-replay.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "F3_F5_10s_evaluated": False}
    primary = None
    thread_limits = process = None
    state = None
    active_arm = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        write_json(out / "advance-ledger.json", ledger)

    def attempt(label, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E236 " + label + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E236 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E236 resident memory cap exceeded")

    def save_partial():
        if state is None or active_arm is None:
            return
        destination = out / (active_arm + "-boundary.npz")
        if destination.exists():
            return
        rows = state["ticks_completed"] // state["sample_every"] + 1
        save_npz(destination, rate64=state["rate64"][:rows],
                 fast64=state["fast64"][:rows], slow64=state["slow64"][:rows],
                 completed_controls=np.array(rows - 1, np.int64),
                 ticks_completed=np.array(state["ticks_completed"], np.int64))
        if state["ticks_completed"] < 1980:
            rate_tick = state["ticks_completed"]
            memory_tick = state["memory_ticks_completed"]
            save_npz(out / (active_arm + "-partial-stages.npz"),
                rate64=state["rate_stage"][4 * max(0, rate_tick - 1):4 * rate_tick],
                fast64=state["fast_stage"][4 * max(0, memory_tick - 1):4 * memory_tick],
                slow64=state["slow_stage"][4 * max(0, memory_tick - 1):4 * memory_tick],
                rate_tick=np.array(rate_tick, np.int64),
                memory_tick=np.array(memory_tick, np.int64),
                stage_targets_completed=np.array(state["stage_targets_completed"], np.int64))

    try:
        write_json(out / "source-review-snapshot.json", review)
        from threadpoolctl import threadpool_limits
        thread_limits = threadpool_limits(limits=1)
        ledger["threading"] = "threadpoolctl limits=1"
        import psutil
        process = psutil.Process()
        paths = freeze["paths"]
        params = load_npz(safe(paths["parameters"]),
            ("motor_ids", "free_columns", "fixed_columns", "membership",
             "scale", "tau", "cap", "inverse_adaptation_tau",
             "initial_rate", "pool", "moment", "rest", "ltm"))
        prototypes = load_npz(safe(paths["prototypes"]),
                              ("centers", "sigma2", "membership"))
        beta = load_npz(safe(paths["candidate"]), ("beta",))["beta"]
        basis = load_npz(safe(paths["basis"]), ("h_fast", "h_slow"))
        conditional = load_npz(safe(paths["conditional"]),
                               ("rate", "activation", "q_direct", "tendon"))
        startup = load_npz(safe(paths["startup_targets"]),
            ("loaded_startup_target_command", "loaded_startup_target_tendon"))
        e235 = load_npz(safe(paths["actual"]) / "control-raw.npz",
            ("motor_rate32", "decoder_input32", "activation", "command",
             "tendon", "free_effector64", "h_fast", "h_slow"))
        motor = params["motor_ids"]
        free = params["free_columns"]
        fixed = params["fixed_columns"]
        if (beta.shape != (96, 64)
                or not np.array_equal(prototypes["membership"],
                                      params["membership"])
                or not np.array_equal(params["ltm"], [15, 30, 46, 63, 78, 94])
                or params["pool"].shape != (96, 391)
                or params["moment"].shape != (42, 96)
                or params["rest"].shape != (42,)):
            raise ValueError("E217 ordered anatomy/ownership changed")
        e208_meta = json.loads(safe(paths["training_metadata"]).read_text(
            encoding="utf-8"))
        if e208_meta.get("motor_ids") != motor.tolist():
            raise ValueError("E208 original 391-MN column order changed")
        e235_meta = json.loads((safe(paths["actual"]) / "observation.json").read_text(
            encoding="utf-8"))
        training_headers = {channel: np.load(safe(paths["training_" + label]),
            mmap_mode="r") for channel, label in zip(
                CHANNELS, ("positive", "negative", "drive"))}
        validate_saved_layout(params, basis, conditional, startup, e235,
                              e235_meta, training_headers)
        initial = {"rate": e235["free_effector64"][0].copy(),
                   "fast": e235["h_fast"][0].copy(),
                   "slow": e235["h_slow"][0].copy()}
        parameter = {"scale": params["scale"][free],
                     "membership": params["membership"][free].copy(),
                     "tau": params["tau"][free], "cap": params["cap"][free],
                     "inverse_adaptation_tau": params["inverse_adaptation_tau"][free]}
        proto = {"centers": prototypes["centers"][free],
                 "sigma2": prototypes["sigma2"][free]}
        validate_initial_state(initial, params, basis, conditional, free)
        validate_model(parameter, proto, beta, motor[free], initial)
        actual_inputs = {channel: np.load(safe(paths["actual"]) / (
            "stage-" + channel + ".npy"), mmap_mode="r")
            for channel in CHANNELS}
        training_inputs = {channel: np.asarray(array[:7920, free], dtype=np.float32)
            for channel, array in training_headers.items()}
        if any(value.shape != (7920, 343) or value.dtype != np.float32
               for value in (*actual_inputs.values(), *training_inputs.values())):
            raise ValueError("Complete aligned E235/E208 stage input arrays required")
        rate32_actual = e235["motor_rate32"]
        rate64_training = conditional["rate"][:100]
        if (rate32_actual.shape != (100, 391)
                or rate32_actual.dtype != np.float32
                or rate64_training.shape != (100, 391)
                or rate64_training.dtype != np.float64
                or not np.array_equal(rate32_actual[:, fixed].astype(np.float64),
                                      rate64_training[:, fixed])):
            raise ValueError("E235/E218 fixed48 original neuronal rate changed")
        ledger["status"] = "response_replay_requested"
        persist()
        replay_states = {}
        raw_readouts = {}
        for arm, inputs in (("actual", actual_inputs), ("training", training_inputs)):
            budget()
            active_arm = arm
            entry = {"arm": arm, "requested": True, "entered": False,
                     "returned": False, "stage_inputs": 7920,
                     "ticks_requested": 1980, "ticks_entered": 0,
                     "ticks_completed": 0, "memory_ticks_completed": 0,
                     "stage_targets_completed": 0}
            ledger["arms"].append(entry)
            persist()
            state = new_workspace(7920, 343, initial, 20)
            entry["entered"] = True
            persist()
            replay_error = None
            try:
                replay(inputs, parameter, proto, beta, initial,
                       sample_every=20, chunk=8, check=budget, state=state)
            except BaseException as error:
                replay_error = error
            finally:
                replay_error = finalize_arm_receipts(entry, state, save_partial,
                    persist, ledger["storage_errors"], replay_error)
            if replay_error is not None:
                raise replay_error
            entry["returned"] = True
            replay_states[arm] = state
            budget()
            if arm == "actual":
                error = {"rate64": np.load(safe(paths["actual"]) /
                    "stage-effector64.npy", mmap_mode="r") - state["rate_stage"],
                    "fast64": np.load(safe(paths["actual"]) /
                    "stage-fast64.npy", mmap_mode="r") - state["fast_stage"],
                    "slow64": np.load(safe(paths["actual"]) /
                    "stage-slow64.npy", mmap_mode="r") - state["slow_stage"]}
                save_npz(out / "actual-stage-errors.npz", **error,
                         observed_minus_replay=np.array(True))
            if arm == "actual":
                assembled = rate32_actual.copy()
                assembled[1:, free] = state["rate64"][1:].astype(np.float32)
                assembled[0] = e235["decoder_input32"][0]
                decoded = readout(assembled, params["pool"], params["moment"],
                    params["rest"], params["ltm"], e235["activation"][0],
                    e235["command"][0])
                save_npz(out / "actual-readout.npz", **decoded)
                raw_readouts[arm] = decoded
            else:
                assembled64 = rate64_training.copy()
                assembled64[:, free] = state["rate64"]
                assembled32 = assembled64.astype(np.float32)
                decoded64 = readout(assembled64, params["pool"], params["moment"],
                    params["rest"], params["ltm"], e235["activation"][0],
                    e235["command"][0])
                decoded32 = readout(assembled32, params["pool"], params["moment"],
                    params["rest"], params["ltm"], e235["activation"][0],
                    e235["command"][0])
                save_npz(out / "training-readout64.npz", **decoded64)
                save_npz(out / "training-readout32.npz", **decoded32)
                raw_readouts["training64"] = decoded64
                raw_readouts["training32"] = decoded32
            state = None
            active_arm = None
        ledger["status"] = "all_raw_saved_before_validation"
        persist()
        with np.load(out / "actual-stage-errors.npz", allow_pickle=False) as saved:
            errors = {key: saved[key].copy() for key in ("rate64", "fast64", "slow64")}
        stage_comparison = {key: error_summary(value, np.zeros_like(value))
                            for key, value in errors.items()}
        actual = replay_states["actual"]
        training = replay_states["training"]
        boundary = {"actual": {
            "rate64": error_summary(actual["rate64"], e235["free_effector64"]),
            "fast64": error_summary(actual["fast64"], e235["h_fast"]),
            "slow64": error_summary(actual["slow64"], e235["h_slow"])},
            "training": {
            "rate64": error_summary(training["rate64"], rate64_training[:, free]),
            "fast64": error_summary(training["fast64"], basis["h_fast"][:100, free]),
            "slow64": error_summary(training["slow64"], basis["h_slow"][:100, free])}}
        readout_comparison = {"actual": {
            "decoder_input32_after_row0": error_summary(
                raw_readouts["actual"]["motor_rate"][1:],
                e235["decoder_input32"][1:]),
            "activation": error_summary(raw_readouts["actual"]["activation"],
                                        e235["activation"]),
            "command": error_summary(raw_readouts["actual"]["command"],
                                     e235["command"]),
            "tendon": error_summary(raw_readouts["actual"]["tendon"],
                                    e235["tendon"])},
            "training64": {
            "activation": error_summary(raw_readouts["training64"]["activation"],
                                        conditional["activation"][:100]),
            "command": error_summary(raw_readouts["training64"]["command"],
                                     conditional["q_direct"][:100]),
            "tendon": error_summary(raw_readouts["training64"]["tendon"],
                                    conditional["tendon"][:100])}}
        actual_minus_training = (raw_readouts["actual"]["command"]
                                 - raw_readouts["training32"]["command"])
        recorded_actual_minus_training = (e235["command"]
                                          - raw_readouts["training32"]["command"])
        difference_closure = error_summary(actual_minus_training,
                                           recorded_actual_minus_training)
        comparison = {"stage_error_convention": "observed_minus_replay",
            "stage": stage_comparison, "boundary": boundary,
            "readout": readout_comparison,
            "same_equation_input_difference_command_closure": difference_closure,
            "control0_decoder_not_consumed": True,
            "actual_assembled_rate0_is_initial_only": True,
            "float32_decoder_byte_exact_reported_separately": True}
        write_json(out / "raw-comparison.json", comparison)
        target_command = startup["loaded_startup_target_command"]
        target_tendon = startup["loaded_startup_target_tendon"]
        startup_report = {name: startup_metrics(decoded, target_command,
                                target_tendon) for name, decoded in raw_readouts.items()}
        write_json(out / "startup-metrics.json", startup_report)
        tol = freeze["tolerances"]
        if (any(item["ticks_completed"] != 1980
                or item["stage_targets_completed"] != 7920
                for item in ledger["arms"])
            or stage_comparison["rate64"]["max_abs"] > tol["rate64"]
            or max(stage_comparison[key]["max_abs"] for key in
                   ("fast64", "slow64")) > tol["memory64"]
            or any(boundary[arm]["rate64"]["max_abs"] > tol["rate64"]
                   or max(boundary[arm][key]["max_abs"] for key in
                          ("fast64", "slow64")) > tol["memory64"]
                   for arm in ("actual", "training"))
            or any(readout_comparison[arm]["command"]["max_abs"] > tol["command"]
                   or readout_comparison[arm]["tendon"]["max_abs"] > tol["tendon"]
                   or readout_comparison[arm]["activation"]["max_abs"]
                   > tol["activation"] for arm in ("actual", "training64"))
            or difference_closure["max_abs"] > tol["command"]):
            raise ValueError("E236 saved-stage independent replay fidelity failed")
        result.update(validity="valid_replay", execution_status="complete",
            actual_input_replay_valid=True, training_input_replay_valid=True,
            startup_metrics_are_original_unpadded=True,
            conclusion="saved conditional input-response relation only")
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_or_fidelity_failure",
                      primary_error=repr(error), traceback=traceback.format_exc())
    finally:
        if state is not None:
            attempt("partial-boundary", save_partial)
        if thread_limits is not None:
            attempt("cpu-thread-restore", thread_limits.restore_original_limits)
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E236 internal wall limit exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E236 wall limit: " + repr(error))
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                      if path.is_file())
        if ledger["output_bytes"] > limits["output_bytes"]:
            error = OSError("E236 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E236 output cap: " + repr(error))
        if primary is not None and result["validity"] == "valid_replay":
            result["pre_storage_validity"] = "valid_replay"
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: write_json(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: write_json(
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
        result = run(args.freeze, args.freeze_sha256,
                     args.review, args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"]}, sort_keys=True))
    return 0 if result["validity"] == "valid_replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
