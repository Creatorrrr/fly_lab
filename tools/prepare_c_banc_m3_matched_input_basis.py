"""E241 conditional grouped basis from E240 saved same-tendon teacher inputs.

No BANC network, body, teacher fit, or physics is constructed here.
"""
# ruff: noqa: BLE001 -- preserve first scientific/GPU exception during cleanup

from __future__ import annotations

import argparse
import gc
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

from tools import banc_m3_grouped_response_basis as grouped
from tools import banc_m3_multiscale_neural_basis as multiscale
from tools import fit_c_banc_m3_grouped_response as fit_group
from tools import observe_c_banc_m3_matched_teacher_inputs as prior
from tools import prepare_c_banc_m3_grouped_response_basis as old_runner

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "c316e39f0d64516a5cf5dc94143e366b748b57addda5bf678849f8fa953a3520"
FREEZE_SCHEMA = "flylab.m3.e241.matched-input-basis.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e241.matched-input-basis.review.v1"
OWNED = ("tools/prepare_c_banc_m3_matched_input_basis.py",
         "tests/test_c_banc_matched_input_basis.py")
OUT = BASE / "e241-matched-input-basis"
CHANNELS = ("positive_recurrent", "negative_magnitude", "drive")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
VALID = "valid_matched_input_grouped_basis"
SPLIT = {"train_control_first": 1, "train_control_last": 1200,
         "validation_control_first": 1201, "validation_control_last": 2000,
         "scale_stage_stop": 96000, "prototype_row_stop": 1201,
         "velocity_previous_control": True, "F6_independent_exposure": False}
LIMITS = {"cpu_threads": 1, "gpu_count": 1, "original_attempts": 1,
          "retries": 0, "history_launches": 1, "basis_launches": 1,
          "cuda_launches": 2, "channel_transfer_count": 6,
          "fit_calls": 0, "solver_calls": 0, "body_constructors": 0,
          "neural_constructors": 0, "scientific_neural_ticks": 0,
          "scientific_physics_steps": 0, "conditional_ticks_per_launch": 40000,
          "internal_s": 110, "finalization_reserve_s": 10,
          "external_s": 130, "supervisor_s": 140,
          "gpu_bytes": 3221225472, "resident_bytes": 6442450944,
          "output_bytes": 1610612736, "minimum_free_bytes": 4294967296}


def safe(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash file required")
    name = Path(relative)
    if name.is_absolute() or any(part in (".", "..") for part in name.parts):
        raise ValueError("Unsafe E241 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E241 frozen file missing or escaped")
    return path


def save_json(path, value):
    old_runner.save_json(path, value)


def save_arrays(path, arrays):
    old_runner.save_arrays(path, arrays)


def attempt(errors, label, callback):
    try:
        return callback()
    except BaseException as error:  # preserve primary
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    if (Path(freeze_path).resolve() != (BASE / "e241-matched-input-basis-freeze.json").resolve()
            or freeze_sha != FREEZE_SHA or prior.sha256(freeze_path) != FREEZE_SHA
            or prior.sha256(review_path) != review_sha):
        raise ValueError("Exact E241 frozen design/review required")
    freeze = json.loads(Path(freeze_path).read_text("utf-8"))
    review = json.loads(Path(review_path).read_text("utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E241_MATCHED_INPUT_BASIS_DESIGN_20260925.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("graph_hash") !=
                "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("split") != SPLIT or freeze.get("resource_limits") != LIMITS
            or freeze.get("shape") != {"motors": 391, "free_motors": 343,
                "fixed_motors": 48, "muscles": 96, "free_muscles": 90,
                "fixed_muscles": 6, "basis_count": 64,
                "descriptor_width": 9, "samples": 2001, "stage_rows": 160000}
            or freeze.get("original_cumulative_s") != 6537.21311039964
            or freeze.get("previous_cumulative_limit_s") != 6605
            or freeze.get("allocation_increase_s") != 75
            or freeze.get("original_cumulative_limit_s") != 6680
            or any(freeze.get(k) is not False for k in
                   ("candidate_adopted", "F2_complete", "F6_exposed",
                    "whole_goal_complete", "active_UI_changed"))
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Reviewed E241 source/split/resource contract required")
    for record, labels in ((freeze, ("input_sha256", "reused_source_sha256",
                                    "preservation_sha256")),
                           (review, ("source_sha256", "additional_reused_source_sha256",
                                     "additional_input_sha256"))):
        for label in labels:
            for name, digest in record.get(label, {}).items():
                if prior.sha256(safe(name)) != digest:
                    raise ValueError("E241 source/input hash changed: " + name)
    if prior.sha256(safe(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E241 frozen design changed")
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E241 frozen package changed: " + name)
    if out.resolve() != OUT.resolve() or out.exists() or not out.parent.is_dir():
        raise ValueError("New exact E241 output directory required")
    if shutil.disk_usage(out.parent).free < LIMITS["minimum_free_bytes"]:
        raise OSError("E241 requires 4 GiB free disk")
    paths = {key: safe(value) for key, value in freeze["paths"].items()
             if (ROOT / value).is_file()}
    for key in ("start",):
        paths[key] = (ROOT / freeze["paths"][key]).resolve()
        if not paths[key].is_dir():
            raise ValueError("E241 saved original start missing")
    proof = json.loads(paths["teacher_proof"].read_text("utf-8"))
    captured = json.loads(paths["teacher_result"].read_text("utf-8"))
    ledger = json.loads(paths["teacher_ledger"].read_text("utf-8"))
    supervisor = json.loads(paths["teacher_supervisor"].read_text("utf-8"))
    arms = ledger.get("arms", [])
    if (proof.get("capture_validity") != "independently_valid"
            or proof.get("hypothesis_verdict") != "supported_for_teacher_reference"
            or proof.get("candidate_adopted") is not False
            or captured.get("validity") != "valid_native_tendon_teacher_capture"
            or captured.get("teacher") != "valid_native_tendon_capture"
            or captured.get("candidate_adopted") is not False
            or len(arms) != 1 or arms[0].get("valid") is not True
            or arms[0].get("controls_completed") != 2000
            or arms[0].get("native_samples") != 100000
            or arms[0].get("storage_errors") or ledger.get("storage_errors")
            or supervisor.get("exit_code") != 0
            or supervisor.get("live") != []
            or supervisor.get("terminated") != []):
        raise ValueError("E241 independently verified complete E240 input required")
    return freeze, review, paths


def training_geometry(positive, negative, drive, descriptor, membership,
                      pool, cap, *, scale_stop=96000, prototype_stop=1201,
                      count=64):
    """Neither validation stages nor target q can enter scale/anchor choice."""
    scale = training_scale(positive, negative, drive, scale_stop=scale_stop)
    prototypes = training_prototypes(descriptor, membership, pool, cap,
                                      prototype_stop=prototype_stop, count=count)
    return scale, prototypes


def training_scale(positive, negative, drive, *, scale_stop=96000):
    if scale_stop < 4 or scale_stop % 4 or scale_stop > len(positive):
        raise ValueError("Complete training-only geometry limits required")
    return multiscale.compute_scales(positive[:scale_stop], negative[:scale_stop],
                                     drive[:scale_stop])


def training_prototypes(descriptor, membership, pool, cap, *,
                        prototype_stop=1201, count=64):
    if prototype_stop < count or prototype_stop > len(descriptor):
        raise ValueError("Complete training-only prototype rows required")
    prototypes = grouped.select_grouped_prototypes(
        descriptor[:prototype_stop], membership, pool, cap, count=count)
    if np.any(prototypes["anchor_rows"] >= prototype_stop):
        raise ValueError("Validation descriptor leaked into E241 anchors")
    return prototypes


def validate_inverse_provenance(value):
    """E217 saved the original float32 inverse after exact float64 promotion."""
    inverse = np.asarray(value)
    if (inverse.shape != (391,) or inverse.dtype != np.float64
            or not np.isfinite(inverse).all()
            or not np.array_equal(inverse,
                                  inverse.astype(np.float32).astype(np.float64))):
        raise ValueError("E241 original float32-derived inverse provenance required")
    return multiscale.validate_inverse_slow(inverse, 391)


def target_alignment(control_tick, times, applied, teacher):
    tick = np.asarray(control_tick)
    time_s = np.asarray(times)
    applied = np.asarray(applied)
    teacher = np.asarray(teacher)
    if (tick.shape != (2001,) or tick.dtype != np.int64
            or not np.array_equal(tick, np.arange(2001))
            or time_s.shape != (2001,) or not np.isfinite(time_s).all()
            or np.max(np.abs(time_s - np.arange(2001) * .005)) > 2e-8
            or applied.shape != (2001, 42) or applied.dtype != np.float64
            or teacher.shape != (2000, 42) or teacher.dtype != np.float64
            or not np.isfinite(applied).all() or not np.isfinite(teacher).all()
            or applied[1:].tobytes() != teacher.tobytes()):
        raise ValueError("E241 same-control E85 applied q alignment changed")
    return {"train_controls": np.arange(1, 1201, dtype=np.int64),
            "validation_controls": np.arange(1201, 2001, dtype=np.int64),
            "train_velocity_start": np.arange(0, 1200, dtype=np.int64),
            "train_velocity_end": np.arange(1, 1201, dtype=np.int64),
            "validation_velocity_start": np.arange(1200, 2000, dtype=np.int64),
            "validation_velocity_end": np.arange(1201, 2001, dtype=np.int64)}


def matched_intercept(H, observed_rates, free, fixed, pool, moment, rest,
                      ltm, groups, initial_activation, held):
    """New H free response plus actual matched-trajectory fixed MN outputs."""
    H = np.asarray(H, np.float64)
    observed = np.asarray(observed_rates)
    free = np.asarray(free, np.int64)
    fixed = np.asarray(fixed, np.int64)
    if (H.ndim != 2 or H.shape[1] != 391 or len(H) < 1
            or observed.shape != H.shape
            or observed.dtype != np.float32
            or free.shape != (343,) or fixed.shape != (48,)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.isfinite(H).all() or not np.isfinite(observed).all()):
        raise ValueError("E241 actual fixed-rate/intercept identity required")
    rate = H.copy()
    rate[:, fixed] = observed[:, fixed].astype(np.float64)
    canonical = fit_group.canonical_group_readout(
        rate, pool, moment, rest, ltm, groups, initial_activation, held)
    return {"rate_intercept": rate, "q_intercept": canonical["command"],
            "activation_intercept": canonical["activation"],
            "tendon_intercept": canonical["tendon"]}


def validate_outputs(history, basis, intercept, affine, canonical, observed,
                     decoded_q, decoded_tendon, tendon_names, cap, free, fixed,
                     initial, tau, tolerance):
    if any(not np.isfinite(value).all() for value in (*history.values(), *basis.values(),
            *intercept.values(), *affine.values(), *canonical.values())):
        raise ValueError("E241 finite raw history/basis/readout required")
    if (not np.array_equal(basis["B"][0], np.zeros((391, 64)))
            or not np.array_equal(basis["H"][0], initial)
            or not np.array_equal(basis["H_factor"][0], np.ones(391))
            or np.min(basis["B"]) < tolerance["B_min"]):
        raise ValueError("E241 grouped initial/envelope basis changed")
    for key in ("h_fast", "h_slow", "descriptor"):
        if not np.allclose(history[key], basis[key], atol=tolerance["history_atol"],
                           rtol=tolerance["history_rtol"]):
            raise ValueError("E241 history and basis memory differ: " + key)
    if np.any(np.sum(basis["B"], axis=2) + basis["H_factor"] >
              1 + tolerance["group_envelope_atol"] +
              tolerance["group_envelope_rtol"]):
        raise ValueError("E241 grouped response mass envelope changed")
    decay_step = .00025 / tau
    decay_tick = 1 - decay_step + decay_step**2/2 - decay_step**3/6 + decay_step**4/24
    decay = decay_tick[None] ** (20 * np.arange(len(basis["H_factor"]))[:, None])
    if (not np.allclose(basis["H_factor"], decay,
                        atol=tolerance["group_envelope_atol"],
                        rtol=tolerance["group_envelope_rtol"])
            or not np.allclose(basis["H"], initial[None] * decay,
                               atol=tolerance["group_envelope_atol"],
                               rtol=tolerance["group_envelope_rtol"])):
        raise ValueError("E241 original-rate RK4 decay changed")
    expected_probe = basis["H"] + cap[None] * np.sum(
        basis["B"] * grouped.PROBE[None, None, :], axis=2)
    if not np.allclose(expected_probe, basis["probe_rate"],
                       atol=tolerance["probe_rate_atol"],
                       rtol=tolerance["probe_rate_rtol"]):
        raise ValueError("E241 recorded probe rate differs")
    if (not np.array_equal(intercept["rate_intercept"][:, fixed],
                           observed[:, fixed].astype(np.float64))
            or not np.allclose(intercept["rate_intercept"][:, free],
                               basis["H"][:, free], atol=2e-12, rtol=2e-12)):
        raise ValueError("E241 new fixed/free rate intercept changed")
    if (not np.allclose(affine["q_probe_affine"], affine["q_probe_direct"],
                        atol=tolerance["probe_q_atol"],
                        rtol=tolerance["probe_q_rtol"])):
        raise ValueError("E241 affine/direct probe q differs")
    if not np.allclose(canonical["command"][1:], decoded_q[1:],
                       atol=2e-10, rtol=2e-12):
        raise ValueError("E241 original canonical decoder q mismatch")
    names = [tendon_names.index(leg + "_tarsus") for leg in LEGS]
    if not np.allclose(canonical["tendon"][1:], decoded_tendon[1:, names],
                       atol=tolerance["tendon_atol"], rtol=0):
        raise ValueError("E241 original canonical tendon mismatch")
    return {"history_basis_max_abs": max(float(np.max(np.abs(history[k] - basis[k])))
                                      for k in ("h_fast", "h_slow", "descriptor")),
            "canonical_q_max_abs": float(np.max(np.abs(
                canonical["command"][1:] - decoded_q[1:]))),
            "probe_q_max_abs": float(np.max(np.abs(
                affine["q_probe_affine"] - affine["q_probe_direct"]))) }


def manifest(out):
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("result.json", "output-manifest.json"):
            files[path.relative_to(out).as_posix()] = {
                "sha256": prior.sha256(path), "bytes": path.stat().st_size}
    receipt = {"schema": "flylab.m3.e241.matched-input-basis.manifest.v1",
               "files": files,
               "total_bytes": sum(item["bytes"] for item in files.values())}
    save_json(out / "output-manifest.json", receipt)
    return receipt


def context_receipt(context):
    return {"requested": context.calls_requested,
            "synchronized": context.calls_completed,
            "transfer_count": context.transfer_count,
            "upload_bytes": context.upload_bytes,
            "parameter_transfer_count": context.parameter_transfer_count,
            "parameter_upload_bytes": context.parameter_upload_bytes,
            "planned_peak_bytes": context.planned_peak_bytes,
            "peak_owned_pool_bytes": context.peak_owned_pool_bytes,
            "peak_device_used_bytes": context.peak_device_used_bytes,
            "baseline_device_used_bytes": context.baseline_device_used_bytes,
            "launch_events": context.launch_events}


def refresh_cuda_totals(ledger, active=None, name=None, *, ticks_per_launch=40000):
    if active is not None:
        ledger[name] = context_receipt(active)
    receipts = [ledger[key] for key in ("history", "basis") if ledger[key] is not None]
    ledger["cuda_requested"] = sum(item["requested"] for item in receipts)
    ledger["cuda_synchronized"] = sum(item["synchronized"] for item in receipts)
    ledger["conditional_ticks_requested"] = ledger["cuda_requested"] * ticks_per_launch
    ledger["conditional_ticks_synchronized"] = (
        ledger["cuda_synchronized"] * ticks_per_launch)


def finish_receipts(out, result, ledger, primary, elapsed, *, writers=None):
    """Late publication and final deadline are part of the validity verdict."""
    if writers is None:
        writers = {
            "preliminary": lambda: save_json(out / "result.json", dict(result,
                validity="incomplete_or_invalid", finalization_pending=True)),
            "manifest": lambda: manifest(out),
            "result": lambda: save_json(out / "result.json", result),
            "ledger": lambda: save_json(out / "ledger.json", ledger),
            "errors": lambda: save_json(out / "publication-failure.json",
                                        ledger["errors"])}

    def record(label, error):
        nonlocal primary
        ledger["errors"].append({"stage": label, "type": type(error).__name__,
                                 "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E241 publication " + label + ": " + repr(error))
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}

    try:
        writers["preliminary"]()
    except BaseException as error:
        record("result-preliminary", error)
    try:
        receipt = writers["manifest"]()
        if receipt is None or receipt["total_bytes"] > LIMITS["output_bytes"]:
            raise RuntimeError("E241 manifest missing or output cap exceeded")
    except BaseException as error:
        record("output-manifest", error)
    if elapsed() > LIMITS["internal_s"]:
        record("final-internal-time", TimeoutError("E241 internal 110 s exceeded"))
    if ledger["errors"] and primary is None:
        record("prior-evidence", RuntimeError("E241 required raw evidence failed"))
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    result["finalization_pending"] = False
    for _ in range(2):
        before = len(ledger["errors"])
        for label in ("ledger", "errors", "manifest", "result"):
            if label == "errors" and not ledger["errors"]:
                continue
            try:
                value = writers[label]()
                if label == "manifest" and (value is None or
                        value["total_bytes"] > LIMITS["output_bytes"]):
                    raise RuntimeError("E241 final manifest missing or output cap exceeded")
            except BaseException as error:
                record(label + "-final", error)
        if elapsed() > LIMITS["internal_s"] and not any(
                item["stage"] == "final-internal-time" for item in ledger["errors"]):
            record("final-internal-time", TimeoutError("E241 internal 110 s exceeded"))
        if len(ledger["errors"]) == before:
            break
    return primary


def release_context(context):
    cp = context.cp
    cp.cuda.get_current_stream().synchronize()
    for key in ("e", "i", "d", "current_output", "partial_host"):
        if hasattr(context, key):
            setattr(context, key, None)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review, paths = gate(freeze_path, freeze_sha, review_path,
                                 review_sha, out)
    out.mkdir(exist_ok=False)
    phase = "source_gate"
    ledger = {"schema": "flylab.m3.e241.matched-input-basis.ledger.v1",
              "phase": phase, "history": None, "basis": None,
              "cuda_requested": 0, "cuda_synchronized": 0,
              "channel_transfer_count": 0, "conditional_ticks_requested": 0,
              "conditional_ticks_synchronized": 0,
              "full_banc_calls": 0, "body_constructors": 0,
              "neural_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "solver_calls": 0, "retries": 0,
              "errors": []}
    result = {"schema": "flylab.m3.e241.matched-input-basis.result.v1",
              "validity": "incomplete_or_invalid", "candidate_adopted": False,
              "F2_complete": False, "F6_exposed": False,
              "whole_goal_complete": False, "fit_calls": 0,
              "full_banc_calls": 0, "new_physics_steps": 0}
    primary = None
    context = None
    history = basis = None

    def elapsed():
        return time.perf_counter() - began

    def budget(label):
        if elapsed() >= LIMITS["internal_s"] - LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E241 finalization reserve before " + label)
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
        if ledger["peak_resident_bytes"] > LIMITS["resident_bytes"]:
            raise MemoryError("E241 resident limit exceeded")

    try:
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"]})
        save_json(out / "threading.json",
                  old_runner.configure_single_thread_runtime())
        phase = "load_saved_inputs"
        positive, negative, drive = (np.load(paths[channel], mmap_mode="r",
            allow_pickle=False) for channel in CHANNELS)
        if any(x.shape != (160000, 391) or x.dtype != np.float32
               for x in (positive, negative, drive)):
            raise ValueError("E241 complete original signed RK stages required")
        with np.load(paths["parameters"], allow_pickle=False) as stored:
            p = {key: stored[key].copy() for key in
                 ("motor_ids", "membership", "pool", "moment", "rest", "ltm",
                  "initial_rate", "tau", "cap", "inverse_adaptation_tau",
                  "free_columns", "fixed_columns")}
        ids = p["motor_ids"]
        member, pool, moment, rest, ltm = (p[key] for key in
            ("membership", "pool", "moment", "rest", "ltm"))
        initial, tau, cap, inverse = (p[key].astype(np.float64) for key in
            ("initial_rate", "tau", "cap", "inverse_adaptation_tau"))
        inverse = validate_inverse_provenance(p["inverse_adaptation_tau"])
        free, fixed = p["free_columns"], p["fixed_columns"]
        if (ids.shape != (391,) or member.shape != (391,)
                or pool.shape != (96, 391) or moment.shape != (42, 96)
                or rest.shape != (42,) or ltm.shape != (6,)
                or free.shape != (343,) or fixed.shape != (48,)
                or initial.shape != (391,) or tau.shape != (391,)
                or cap.shape != (391,) or inverse.shape != (391,)
                or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                or not np.array_equal(fixed,
                                      np.flatnonzero(np.isin(member, ltm)))
                or not all(np.isfinite(x).all() for x in
                           (pool, moment, rest, initial, tau, cap, inverse))
                or np.any(tau < .00025 * 50) or np.any(cap <= 0)
                or np.any(initial < 0) or np.any(initial > cap)):
            raise ValueError("E241 original anatomy/physiology layout changed")
        start = prior.protocol.StateStore.load(paths["start"])
        if not np.array_equal(start["adapter"]["calibration"]["motor_ids"], ids):
            raise ValueError("E241 saved canonical motor order changed")
        for key in ("pool", "moment", "rest", "ltm"):
            prior.exact(start["adapter"]["calibration"][key], p[key],
                        "E241 actual anatomy " + key)
        groups = start["adapter"]["groups"]
        initial_activation = np.asarray(start["adapter"]["activation"], np.float64)
        if len(groups) != 96 or initial_activation.shape != (96,):
            raise ValueError("E241 actual initial activation/groups required")
        with np.load(paths["control"], allow_pickle=False) as data:
            control = {key: data[key].copy() for key in
                       ("control_tick", "network_tick", "physics_time_s",
                        "motor_rate32", "applied_command", "applied_tendon",
                        "decoded_command", "decoded_tendon")}
        with np.load(paths["teacher_command"], allow_pickle=False) as data:
            teacher = data["command"].copy()
        aligned = target_alignment(control["control_tick"],
                                   control["physics_time_s"],
                                   control["applied_command"], teacher)
        if (control["motor_rate32"].shape != (2001, 391)
                or control["motor_rate32"].dtype != np.float32
                or control["decoded_command"].shape != (2001, 42)
                or control["decoded_tendon"].shape != (2001, 8)
                or control["applied_tendon"].shape != (2001, 8)
                or not np.array_equal(control["network_tick"],
                                      control["control_tick"] * 20)
                or not np.array_equal(control["motor_rate32"][0],
                                      initial.astype(np.float32))):
            raise ValueError("E241 matched original control/readout rows changed")
        metadata = json.loads(paths["observation"].read_text("utf-8"))
        names = metadata["tendon_names"]
        if (len(names) != 8 or len(set(names)) != 8
                or not np.array_equal(np.asarray(metadata["motor_ids"]), ids)
                or not np.array_equal(control["applied_tendon"],
                                      control["decoded_tendon"])):
            raise ValueError("E241 actual ordered same-control tendons required")
        save_arrays(out / "targets.npz", {**control,
            "target_command": control["applied_command"], **aligned,
            "train_target": control["applied_command"][1:1201],
            "validation_target": control["applied_command"][1201:2001],
            "control_velocity": np.diff(control["applied_command"], axis=0) / .005})
        save_json(out / "groups.json", {"groups": groups,
            "tendon_names": names, "split": SPLIT})
        budget("training scale")
        phase = "training_scale"
        scale = training_scale(positive, negative, drive)
        p["scale"] = scale
        save_arrays(out / "parameters.npz", p)
        save_json(out / "ledger.json", ledger)
        phase = "history_upload"
        budget("history CUDA")
        context = multiscale.PersistentMultiscaleInputs(
            positive, negative, drive, gpu_limit=LIMITS["gpu_bytes"])
        ledger["channel_transfer_count"] += context.transfer_count
        phase = "history_launch"
        ledger["cuda_requested"] = 1
        ledger["conditional_ticks_requested"] = 40000
        save_json(out / "ledger.json", ledger)
        history, _ = multiscale.run_history_cuda(context, scale, inverse)
        ledger["history"] = context_receipt(context)
        ledger["cuda_synchronized"] = context.calls_completed
        ledger["conditional_ticks_synchronized"] = 40000 * context.calls_completed
        save_arrays(out / "history.npz", history)
        # Explicit ownership boundary: no history device arrays survive basis upload.
        release_context(context)
        context = None
        budget("prototype selection")
        phase = "training_prototypes"
        prototypes = training_prototypes(history["descriptor"], member,
                                         pool, cap)
        save_arrays(out / "prototypes.npz", prototypes)
        phase = "basis_upload"
        budget("basis CUDA")
        context = grouped.PersistentGroupedInputs(
            positive, negative, drive, gpu_limit=LIMITS["gpu_bytes"])
        ledger["channel_transfer_count"] += context.transfer_count
        phase = "basis_launch"
        ledger["cuda_requested"] = 2
        ledger["conditional_ticks_requested"] = 80000
        save_json(out / "ledger.json", ledger)
        basis, _ = grouped.run_grouped_basis_cuda(
            context, scale, inverse, prototypes, initial, tau, cap)
        ledger["basis"] = context_receipt(context)
        ledger["cuda_synchronized"] += context.calls_completed
        ledger["conditional_ticks_synchronized"] += 40000 * context.calls_completed
        save_arrays(out / "basis.npz", basis)
        release_context(context)
        context = None
        phase = "readout"
        budget("matched readout")
        held = control["applied_command"][0]
        intercept = matched_intercept(basis["H"], control["motor_rate32"],
            free, fixed, pool, moment, rest, ltm, groups, initial_activation, held)
        affine = grouped.group_affine_readout(basis, cap, member, pool,
            moment, rest, ltm, groups, free, fixed,
            control["motor_rate32"].astype(np.float64), intercept)
        save_arrays(out / "affine-readout.npz", affine)
        canonical = fit_group.canonical_group_readout(
            control["motor_rate32"].astype(np.float64), pool, moment, rest,
            ltm, groups, initial_activation, held)
        save_arrays(out / "canonical-original.npz", canonical)
        phase = "validate"
        checks = validate_outputs(history, basis, intercept, affine, canonical,
            control["motor_rate32"], control["decoded_command"],
            control["decoded_tendon"], names, cap, free, fixed, initial, tau,
            freeze["tolerance"])
        q_error = canonical["command"] - control["applied_command"]
        velocity_error = np.diff(q_error, axis=0) / .005
        checks["teacher_q_rms_train"] = float(np.sqrt(np.mean(q_error[1:1201] ** 2)))
        checks["teacher_q_rms_validation"] = float(np.sqrt(np.mean(q_error[1201:] ** 2)))
        checks["teacher_velocity_rms_train"] = float(np.sqrt(np.mean(
            velocity_error[:1200] ** 2)))
        checks["teacher_velocity_rms_validation"] = float(np.sqrt(np.mean(
            velocity_error[1200:] ** 2)))
        checks["startup_q_rms_1_to_99"] = float(np.sqrt(np.mean(q_error[1:100] ** 2)))
        checks["startup_q_max_abs_1_to_99"] = float(np.max(np.abs(q_error[1:100])))
        checks["leg_errors"] = {}
        for index, leg in enumerate(LEGS):
            cols = slice(index * 7, (index + 1) * 7)
            checks["leg_errors"][leg] = {
                "q_train_rms_rad": float(np.sqrt(np.mean(q_error[1:1201, cols] ** 2))),
                "q_validation_rms_rad": float(np.sqrt(np.mean(q_error[1201:, cols] ** 2))),
                "velocity_train_rms_rad_s": float(np.sqrt(np.mean(
                    velocity_error[:1200, cols] ** 2))),
                "velocity_validation_rms_rad_s": float(np.sqrt(np.mean(
                    velocity_error[1200:, cols] ** 2)))}
        result.update(validity=VALID, checks=checks,
            conditional_family="64 muscle-shared anchors; no coefficients fitted",
            matched_teacher_target=True)
    except BaseException as error:  # first failure retained
        primary = error
        ledger["errors"].append({"stage": phase, "type": type(error).__name__,
                                 "message": str(error), "traceback": traceback.format_exc()})
    finally:
        for label, arrays in (("history", history), ("basis", basis)):
            if arrays is not None and not (out / (label + ".npz")).exists():
                attempt(ledger["errors"], label + "-raw-retry",
                        lambda label=label, arrays=arrays: save_arrays(
                            out / (label + ".npz"), arrays))
        if context is not None:
            name = "history" if phase.startswith("history") else "basis"
            refresh_cuda_totals(ledger, context, name)
            partial = attempt(ledger["errors"], "partial-readback",
                              context.partial_readback)
            if partial is not None:
                values, failures = partial
                ledger["errors"].extend({"stage": "partial-readback", **item}
                                        for item in failures)
                if values:
                    attempt(ledger["errors"], "partial-raw", lambda: save_arrays(
                        out / (name + "-partial.npz"), values))
            attempt(ledger["errors"], "context-release", lambda: release_context(context))
        refresh_cuda_totals(ledger)
        ledger["phase"] = phase
        ledger["process_wall_s"] = elapsed()
        attempt(ledger["errors"], "ledger", lambda: save_json(out / "ledger.json", ledger))
    if primary is None and ledger["errors"]:
        primary = RuntimeError("E241 partial/readback/storage failure")
    result.update(process_wall_s=elapsed(), cuda_requested=ledger["cuda_requested"],
                  cuda_synchronized=ledger["cuda_synchronized"],
                  channel_transfer_count=ledger["channel_transfer_count"],
                  errors=ledger["errors"], retries=0)
    if (primary is None and (ledger["cuda_requested"] != 2
            or ledger["cuda_synchronized"] != 2
            or ledger["channel_transfer_count"] != 6)):
        primary = RuntimeError("E241 two completed sequential launches and six uploads required")
    if elapsed() > LIMITS["internal_s"] and primary is None:
        primary = TimeoutError("E241 internal time limit exceeded")
    if primary is not None:
        result.update(validity="incomplete_or_invalid",
                      primary_error={"type": type(primary).__name__,
                                     "message": str(primary)})
    primary = finish_receipts(out, result, ledger, primary, elapsed)
    if primary is not None:
        raise primary
    return result


def generated_cuda_smoke(output):
    """Separate small artificial GPU job; caller/root schedules after FIRST."""
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("New generated-only output required")
    out.mkdir(parents=True)
    began = time.perf_counter()
    stages = 256
    phase = np.arange(stages, dtype=np.float64)[:, None]
    columns = np.arange(3, dtype=np.float64)[None]
    positive = (1 + .2 * np.sin(phase / 17 + columns)).astype(np.float32)
    negative = (.4 + .1 * np.cos(phase / 11 + columns)).astype(np.float32)
    drive = (.2 * np.sin(phase / 13 + columns)).astype(np.float32)
    inverse = np.array([4., 5., 6.], np.float64)
    initial = np.array([.1, .2, .15], np.float64)
    tau = np.array([.02, .025, .03], np.float64)
    cap = np.array([8., 9., 10.], np.float64)
    member = np.array([0, 0, 1], np.int64)
    pool = np.array([[.5, .5, 0.], [0., 0., 1.]], np.float64)
    scale = multiscale.compute_scales(positive[:160], negative[:160], drive[:160])
    hctx = None
    bctx = None
    receipt = {"schema": "flylab.m3.e241.generated-cuda-smoke.v1",
               "original_models": 0, "body_constructors": 0,
               "history_requested": 0, "basis_requested": 0}
    primary = None
    try:
        hctx = multiscale.PersistentMultiscaleInputs(
            positive, negative, drive, sample_every=1)
        receipt["history_requested"] = 1
        save_json(out / "receipt.json", receipt)
        history, _ = multiscale.run_history_cuda(hctx, scale, inverse,
                                                 sample_every=1)
        save_arrays(out / "history.npz", history)
        receipt["history"] = context_receipt(hctx)
        release_context(hctx)
        hctx = None
        prototypes = grouped.select_grouped_prototypes(
            history["descriptor"], member, pool, cap)
        save_arrays(out / "prototypes.npz", prototypes)
        bctx = grouped.PersistentGroupedInputs(
            positive, negative, drive, sample_every=1)
        receipt["basis_requested"] = 1
        save_json(out / "receipt.json", receipt)
        basis, _ = grouped.run_grouped_basis_cuda(
            bctx, scale, inverse, prototypes, initial, tau, cap,
            sample_every=1)
        save_arrays(out / "basis.npz", basis)
        receipt["basis"] = context_receipt(bctx)
        release_context(bctx)
        bctx = None
        oracle_history = multiscale.simulate_history_numpy(
            positive, negative, drive, scale, inverse, sample_every=1)
        oracle_basis = grouped.simulate_grouped_basis_numpy(
            positive, negative, drive, scale, inverse,
            prototypes["centers"], prototypes["sigma2"], initial, tau, cap,
            sample_every=1)
        differences = {"history_" + key: float(np.max(np.abs(value -
            oracle_history[key]))) for key, value in history.items()}
        differences.update({"basis_" + key: float(np.max(np.abs(value -
            oracle_basis[key]))) for key, value in basis.items()})
        if not all(np.allclose(value, oracle_history[key], atol=2e-12,
                               rtol=2e-12) for key, value in history.items()):
            raise ValueError("Generated history CUDA/NumPy differs")
        if not all(np.allclose(value, oracle_basis[key], atol=2e-12,
                               rtol=2e-12) for key, value in basis.items()):
            raise ValueError("Generated grouped CUDA/NumPy differs")
        receipt.update(valid=True, differences=differences)
    except BaseException as error:  # retain GPU first error
        primary = error
        receipt.update(valid=False, error=repr(error))
    finally:
        primary = finish_generated_smoke(out, receipt, primary, began,
                                         (("history", hctx), ("basis", bctx)))
    if primary is not None:
        raise primary
    return receipt


def finish_generated_smoke(out, receipt, primary, began, contexts, *,
                           save_receipt=None, release=None, read_partial=None,
                           save_partial=None, clock=None):
    """Keep the first CUDA failure and all independently recoverable raw arrays."""
    save_receipt = save_receipt or (lambda: save_json(out / "receipt.json", receipt))
    release = release or release_context
    read_partial = read_partial or (lambda context: context.partial_readback())
    save_partial = save_partial or (lambda label, values: save_arrays(
        out / (label + "-partial.npz"), values))
    clock = clock or time.perf_counter
    errors = receipt.setdefault("errors", [])

    def record(label, error):
        nonlocal primary
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E241 generated " + label + ": " + repr(error))
        receipt.update(valid=False, error=repr(primary))

    for label, context in contexts:
        if context is None:
            continue
        receipt[label] = context_receipt(context)
        if getattr(context, "current_output", None) is not None:
            try:
                partial = read_partial(context)
                if partial is not None:
                    values, failures = partial
                    errors.extend({"stage": label + "-readback", **item}
                                  for item in failures)
                    if failures and primary is None:
                        record(label + "-readback", RuntimeError(
                            "Generated partial readback failed"))
                    if values:
                        save_partial(label, values)
            except BaseException as error:
                record(label + "-partial", error)
        try:
            release(context)
        except BaseException as error:
            record(label + "-release", error)
    receipt["wall_s"] = clock() - began
    if receipt["wall_s"] > 30:
        record("generated-internal-time", TimeoutError(
            "E241 generated CUDA internal 30 s exceeded"))
    for _ in range(2):
        before = len(errors)
        try:
            save_receipt()
        except BaseException as error:
            record("receipt", error)
        if len(errors) == before:
            break
    receipt["wall_s"] = clock() - began
    if receipt["wall_s"] > 30 and not any(
            item["stage"] == "generated-internal-time" for item in errors):
        record("generated-internal-time", TimeoutError(
            "E241 generated CUDA internal 30 s exceeded"))
        try:
            save_receipt()
        except BaseException as error:
            record("receipt-final", error)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-cuda-smoke", action="store_true")
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    try:
        if args.generated_cuda_smoke:
            if not args.out or any((args.freeze, args.freeze_sha256,
                                    args.review, args.review_sha256)):
                raise ValueError("Generated smoke uses --out only")
            result = generated_cuda_smoke(args.out)
        else:
            if any(value is None for value in (args.freeze, args.freeze_sha256,
                args.review, args.review_sha256, args.out)):
                raise ValueError("E241 original requires freeze/review/out arguments")
            result = run(args.freeze, args.freeze_sha256, args.review,
                         args.review_sha256, args.out)
    except BaseException as error:  # nonzero original or smoke failure
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
