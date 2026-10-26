"""One-shot E209 conditional signed-input basis preparation; no BANC run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_causal_neural_basis import (
    BASIS_COUNT,
    PersistentSignedInputs,
    compute_scales,
    run_basis_cuda,
    run_history_cuda,
    select_prototypes,
)
from tools.fit_c_banc_m3_local_response import direct_metrics, load_targets
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "e6442836c506af5a9fa1c75ff0f5ce35fbf1b365163d3cf1220afee4bd2c097b"
FREEZE_SCHEMA = "flylab.m3.e209.causal-neural-basis.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e209.causal-neural-basis.review.v1"
OWNED_FILES = (
    "tools/banc_m3_causal_neural_basis.py",
    "tools/prepare_c_banc_m3_causal_neural_basis.py",
    "tests/test_c_banc_causal_neural_basis.py",
)
SHAPE = {"motors": 391, "free_motors": 343, "fixed_motors": 48,
         "muscles": 96, "stages": 160000, "conditional_ticks_per_launch": 40000,
         "samples": 2001, "basis_count": 16}
CLOCK = {"neural_dt_s": .00025, "control_dt_s": .005, "ticks_per_control": 20}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "original_attempts": 1,
    "cuda_calls": 2, "optimizer_calls": 0, "retries": 0, "warmup_calls": 0,
    "internal_s": 85, "model_s": 65, "external_s": 95, "supervisor_s": 100,
    "output_bytes": 536870912, "gpu_buffer_bytes": 1610612736,
    "resident_bytes": 6442450944, "minimum_free_bytes": 4294967296}
INPUT_COUNT = 100
REUSED_COUNT = 14


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E209 path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E209 path")
    if file and not path.is_file():
        raise ValueError("E209 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    """Fail before device upload unless every input and owned source is pinned."""
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E209 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E209 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E209_CAUSAL_NEURAL_BASIS_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED_FILES)
            or freeze.get("shape") != SHAPE or freeze.get("clock") != CLOCK
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("new_neural") != 0 or freeze.get("new_physics") != 0
            or freeze.get("new_fit") != 0 or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E209 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E209 design SHA changed")
    for label, count in (("input_sha256", INPUT_COUNT),
                         ("reused_source_sha256", REUSED_COUNT),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E209 frozen input count changed: " + label)
        for relative, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("E209 malformed frozen hash")
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E209 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E209 preflight exhausted model deadline")
    paths = freeze.get("paths", {})
    if not isinstance(paths, dict) or set(paths) != {
            "current", "parameters", "endpoints", "observation", "trace", "native",
            "start", "e201_manifest", "e201_result", "e201_review",
            "positive", "negative", "drive", "signed_result", "signed_manifest",
            "signed_review", "targets", "target_freeze", "mask"}:
        raise ValueError("E209 original path map changed")
    resolved = {key: repository_path(value, file=key != "start")
                for key, value in paths.items()}
    sources = {name: file_sha(repository_path(name)) for name in OWNED_FILES}
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != sources):
        raise ValueError("E209 parent source review does not authorize these sources")
    additional = review.get("additional_reused_source_sha256", {})
    if not isinstance(additional, dict):
        raise TypeError("E209 reviewed extra source hash map invalid")
    for relative, expected in additional.items():
        if file_sha(repository_path(relative)) != expected:
            raise ValueError("E209 reviewed transitive source changed")
    return freeze, review, resolved


def _attempt(errors, name, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve independent raw artifacts
        errors.append({"stage": name, "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
        return None


def save_arrays(path, arrays):
    np.savez(path, **arrays)


def validate_history(history, scale, n, samples):
    if (history["h"].shape != (samples, n, 3)
            or history["descriptor"].shape != (samples, n, 6)
            or scale.shape != (n, 3)):
        raise ValueError("E209 history shape changed")
    if any(value.dtype != np.float64 or not np.isfinite(value).all()
           for value in (*history.values(), scale)):
        raise ValueError("E209 history nonfinite or non-double")
    if not np.array_equal(history["h"][0], np.zeros((n, 3))):
        raise ValueError("E209 initial causal memory changed")
    if np.min(history["h"][:, :, 0]) < -1e-12:
        raise ValueError("E209 completed excitatory memory negative")


def validate_basis(history, basis, initial, cap):
    n = len(initial)
    count = len(history["h"])
    expected = {"h": (count, n, 3), "descriptor": (count, n, 6),
        "B": (count, n, BASIS_COUNT), "H": (count, n),
        "H_factor": (count, n), "probe_rate": (count, n)}
    if set(basis) != set(expected):
        raise ValueError("E209 basis raw keys changed")
    for key, shape in expected.items():
        if (basis[key].shape != shape or basis[key].dtype != np.float64
                or not np.isfinite(basis[key]).all()):
            raise ValueError("E209 basis shape/dtype/finite failure: " + key)
    if (history["h"].tobytes() != basis["h"].tobytes()
            or history["descriptor"].tobytes() != basis["descriptor"].tobytes()):
        raise ValueError("E209 CUDA history replay bytes changed")
    if (np.min(basis["B"]) < -1e-12 or np.min(basis["H_factor"]) < 0
            or np.max(np.sum(basis["B"], axis=2) - (1 - basis["H_factor"])) > 1e-10):
        raise ValueError("E209 positive basis mass bound failed")
    beta = (np.arange(BASIS_COUNT) + 1) / 17
    reconstructed = basis["H"] + cap[None, :] * np.sum(basis["B"] * beta, axis=2)
    if not np.allclose(reconstructed, basis["probe_rate"], atol=2e-10, rtol=1e-11):
        raise ValueError("E209 independent probe does not match affine basis")
    if (np.min(basis["probe_rate"]) < -2e-10
            or np.max(basis["probe_rate"] - cap) > 2e-10):
        raise ValueError("E209 probe exceeds original rate cap")
    if not np.array_equal(basis["H"][0], initial):
        raise ValueError("E209 initial original rate changed")
    return {"max_probe_affine_abs": float(np.max(np.abs(
        reconstructed - basis["probe_rate"]))),
        "min_B": float(np.min(basis["B"])),
        "max_basis_mass_excess": float(np.max(np.sum(basis["B"], axis=2)
                                         - (1 - basis["H_factor"]))) }


def canonical_readout(start, rates):
    """Original anatomical pool and 10 ms activation, direct control loop."""
    a = start["adapter"]
    c = a["calibration"]
    pool, moment, rest, ltm = (np.asarray(c[key]) for key in
        ("pool", "moment", "rest", "ltm"))
    rates = np.asarray(rates, np.float64)
    if (rates.shape[1] != 391 or pool.shape != (96, 391)
            or moment.shape != (42, 96) or rest.shape != (42,)
            or ltm.shape != (6,) or not np.isfinite(rates).all()
            or np.any(rates < -1e-12)):
        raise ValueError("E209 canonical anatomy/rate invalid")
    active = np.asarray(a["original"]["activation"], np.float64).copy()
    if active.shape != (96,):
        raise ValueError("E209 actual original muscle activation missing")
    command = np.empty((len(rates), 42), np.float64)
    activation = np.empty((len(rates), 96), np.float64)
    tendon = np.empty((len(rates), 6), np.float64)
    names = [a["groups"][int(i)]["leg"] + "_tarsus" for i in ltm]
    if set(names) != {leg + "_tarsus" for leg in LEGS}:
        raise ValueError("E209 six actual LTM tendon names changed")
    order = [names.index(leg + "_tarsus") for leg in LEGS]
    command[0] = np.asarray(start["body"]["action"]["joint_angles"], np.float64)
    activation[0] = active
    tendon[0] = .2 * np.tanh(active[ltm][order] / 10)
    alpha = -np.expm1(-.005 / .01)
    for t in range(1, len(rates)):
        active += alpha * (pool @ rates[t] - active)
        activation[t] = active
        command[t] = rest + .02 * (moment @ active)
        tendon[t] = .2 * np.tanh(active[ltm][order] / 10)
    return {"command": command, "activation": activation, "tendon": tendon}


def affine_readout(start, original_rates, basis, cap, free, fixed):
    """Materialize canonical q intercept and exact single-owner EMA columns."""
    a = start["adapter"]
    c = a["calibration"]
    pool = np.asarray(c["pool"], np.float64)
    moment = np.asarray(c["moment"], np.float64)
    membership = np.asarray(a["parameters"]["membership"], np.int64)
    if (membership.shape != (391,) or not np.array_equal(
            np.sort(np.r_[free, fixed]), np.arange(391))):
        raise ValueError("E209 frozen free/fixed ownership changed")
    for m, owner in enumerate(membership):
        nonzero = np.flatnonzero(pool[:, m])
        if len(nonzero) != 1 or nonzero[0] != owner:
            raise ValueError("E209 one original anatomical owner per MN required")
    intercept = np.asarray(basis["H"], np.float64).copy()
    intercept[:, fixed] = original_rates[:, fixed]
    baseline = canonical_readout(start, intercept)
    B = np.asarray(basis["B"], np.float64)
    filtered = np.empty_like(B)
    filtered[0] = 0
    alpha = -np.expm1(-.005 / .01)
    for t in range(1, len(B)):
        filtered[t] = filtered[t - 1] + alpha * (B[t] - filtered[t - 1])
    D = filtered * (cap * pool[membership, np.arange(391)])[None, :, None]
    D[:, fixed] = 0
    beta = (np.arange(BASIS_COUNT) + 1) / 17
    delta_rate = cap[None, :] * np.sum(B * beta, axis=2)
    delta_rate[:, fixed] = 0
    direct_rates = intercept + delta_rate
    direct = canonical_readout(start, direct_rates)
    delta_muscle = np.zeros((len(B), 96), np.float64)
    weighted = np.sum(D * beta, axis=2)
    for m, owner in enumerate(membership):
        delta_muscle[:, owner] += weighted[:, m]
    affine_command = baseline["command"] + .02 * (delta_muscle @ moment.T)
    affine_command[0] = baseline["command"][0]
    return {"rate_intercept": intercept, "D": D,
        "q_intercept": baseline["command"], "q_probe_affine": affine_command,
        "q_probe_direct": direct["command"],
        "tendon_intercept": baseline["tendon"],
        "tendon_probe": direct["tendon"],
        "probe_rate_conditional": direct_rates,
        "activation_intercept": baseline["activation"],
        "activation_probe": direct["activation"]}


def validate_affine_readout(start, affine, canonical_original, fixed):
    """Check persisted affine raw and the three canonical LTM readouts."""
    if any(not np.isfinite(value).all() for value in affine.values()):
        raise ValueError("E209 affine raw nonfinite")
    if not np.allclose(affine["q_probe_affine"], affine["q_probe_direct"],
                       atol=2e-12, rtol=2e-12):
        raise ValueError("E209 canonical affine command mismatches direct decoder")
    if not np.array_equal(affine["D"][:, fixed], np.zeros_like(affine["D"][:, fixed])):
        raise ValueError("E209 fixed MN affine basis must be zero")
    ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
    for key in ("activation_intercept", "activation_probe"):
        if not np.array_equal(affine[key][:, ltm], canonical_original["activation"][:, ltm]):
            raise ValueError("E209 original versus changed canonical LTM activation differs")
    for key in ("tendon_intercept", "tendon_probe"):
        if not np.array_equal(affine[key], canonical_original["tendon"]):
            raise ValueError("E209 original versus changed canonical tendon differs")
    return {"affine_command_max_abs": float(np.max(np.abs(
        affine["q_probe_affine"] - affine["q_probe_direct"]))),
        "fixed_ltm_activation_exact": True, "fixed_ltm_tendon_exact": True}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e209.output-manifest.v1", "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def collect_conditional(out, context, scale, initial, tau, cap,
                        *, sample_every=20, kernel_walls=None,
                        before_basis=None, after_basis=None):
    """Persist each launch raw before validation; first failure blocks second."""
    out = Path(out)
    history, wall = run_history_cuda(context, scale, sample_every=sample_every)
    if kernel_walls is not None:
        kernel_walls["history_kernel_wall_s"] = wall
    save_arrays(out / "history.npz", history)
    samples = context.ticks // sample_every + 1
    validate_history(history, scale, context.n, samples)
    prototypes = select_prototypes(history["descriptor"])
    save_arrays(out / "prototypes.npz", prototypes)
    if before_basis is not None:
        before_basis()
    basis, wall = run_basis_cuda(context, scale, prototypes, initial, tau, cap,
                                 sample_every=sample_every)
    if kernel_walls is not None:
        kernel_walls["basis_kernel_wall_s"] = wall
    save_arrays(out / "basis.npz", basis)
    if after_basis is not None:
        after_basis()
    checks = validate_basis(history, basis, initial, cap)
    return history, prototypes, basis, checks


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E209 output exists: " + str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    freeze = review = paths = context = None
    phase = "gate"
    ledger = {"schema": "flylab.m3.e209.ledger.v1", "phase": phase,
        "cuda_requested": 0, "cuda_synchronized": 0, "warmup_calls": 0,
        "optimizer_calls": 0, "full_banc_calls": 0, "physics_steps": 0,
        "upload_bytes": None, "channel_transfer_count": None,
        "parameter_transfer_count": None, "parameter_upload_bytes": None,
        "planned_output_bytes": None, "planned_parameter_bytes": None,
        "planned_peak_gpu_bytes": None, "peak_owned_cupy_pool_bytes": None,
        "peak_device_used_bytes": None, "baseline_device_used_bytes": None,
        "history_kernel_wall_s": None, "basis_kernel_wall_s": None,
        "model_elapsed_s": None, "launch_events": []}
    result = {"schema": "flylab.m3.e209.causal-neural-basis.result.v1",
        "validity": "incomplete", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "fit_calls": 0, "full_banc_calls": 0, "new_physics_steps": 0}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E209 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        from flylab.c.storage import StateStore

        phase = "input"
        positive = np.load(paths["positive"], mmap_mode="r", allow_pickle=False)
        negative = np.load(paths["negative"], mmap_mode="r", allow_pickle=False)
        drive = np.load(paths["drive"], mmap_mode="r", allow_pickle=False)
        if any(value.shape != (160000, 391) or value.dtype != np.float32
               for value in (positive, negative, drive)):
            raise ValueError("E209 actual native channel shape/dtype changed")
        scale = compute_scales(positive, negative, drive)
        with (np.load(paths["parameters"], allow_pickle=False) as p,
              np.load(paths["endpoints"], allow_pickle=False) as endpoints,
              np.load(paths["mask"], allow_pickle=False) as mask):
            mask_arrays = {key: mask[key].copy() for key in mask.files}
            initial = p["initial_rate"].astype(np.float64)
            tau = p["tau"].astype(np.float64)
            cap = p["cap"].astype(np.float64)
            motor_ids = p["motor_ids"].copy()
            original_rates = endpoints["motor_rate"].astype(np.float64)
            free = mask["free_columns"].astype(np.int64)
            fixed = mask["fixed_columns"].astype(np.int64)
            if (initial.shape != (391,) or tau.shape != (391,) or cap.shape != (391,)
                    or original_rates.shape != (2001, 391)
                    or motor_ids.shape != (391,) or free.shape != (343,)
                    or fixed.shape != (48,) or np.any(tau < .00025 * 50)
                    or np.any(cap <= 0) or np.any(initial < 0)
                    or np.any(initial > cap) or np.any(original_rates < 0)
                    or np.any(original_rates > cap[None, :])
                    or not all(np.isfinite(x).all() for x in
                               (initial, tau, cap, original_rates))):
                raise ValueError("E209 original motor physiology or mask changed")
        start = StateStore.load(paths["start"])
        membership = np.asarray(start["adapter"]["parameters"]["membership"], np.int64)
        ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
        expected_fixed = np.flatnonzero(np.isin(membership, ltm))
        if (membership.shape != (391,) or ltm.shape != (6,)
                or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                or not np.array_equal(free, np.sort(free))
                or not np.array_equal(fixed, expected_fixed)
                or not np.array_equal(mask_arrays["trainable_mask"],
                    np.isin(np.arange(391), free))
                or not np.array_equal(mask_arrays["membership"], membership)
                or not np.array_equal(mask_arrays["ltm"], ltm)
                or not np.array_equal(mask_arrays["motor_ids"], motor_ids)
                or mask_arrays["motor_roster"].tolist() !=
                   start["monotone_recruitment_patch"]["motor_roster"]
                or not np.array_equal(
                    start["adapter"]["calibration"]["motor_ids"], motor_ids)
                or not np.array_equal(
                    start["network"]["motor_timescale"]["ids"], motor_ids)
                or not np.array_equal(original_rates[0], initial.astype(np.float64))):
            raise ValueError("E209 pinned native MN identity/ownership changed")
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        if len(trace["samples"]) != 2001:
            raise ValueError("E209 original 2001 control trace required")
        save_arrays(out / "parameters.npz", {"scale": scale, "initial_rate": initial,
            "tau": tau, "cap": cap, "motor_ids": motor_ids,
            "free_columns": free, "fixed_columns": fixed})
        if elapsed() >= RESOURCE["model_s"]:
            raise TimeoutError("E209 model deadline before CUDA")
        phase = "upload"
        context = PersistentSignedInputs(positive, negative, drive,
            gpu_limit=RESOURCE["gpu_buffer_bytes"])
        ledger["upload_bytes"] = context.upload_bytes
        ledger["channel_transfer_count"] = context.transfer_count
        ledger["planned_output_bytes"] = context.planned_output_bytes
        ledger["planned_parameter_bytes"] = context.planned_parameter_bytes
        ledger["planned_peak_gpu_bytes"] = context.planned_peak_bytes
        phase = "history"
        def before_basis():
            nonlocal phase
            if elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E209 model deadline before basis CUDA")
            phase = "basis"

        def after_basis():
            ledger["model_elapsed_s"] = elapsed()
            if ledger["model_elapsed_s"] >= RESOURCE["model_s"]:
                raise TimeoutError("E209 model deadline after basis CUDA")

        _, _, basis, basis_checks = collect_conditional(
            out, context, scale, initial, tau, cap,
            kernel_walls=ledger, before_basis=before_basis, after_basis=after_basis)
        phase = "basis_complete"
        phase = "readout"
        affine = affine_readout(start, original_rates, basis, cap, free, fixed)
        save_arrays(out / "affine-readout.npz", affine)
        canonical_original = canonical_readout(start, original_rates)
        save_arrays(out / "canonical-original.npz", canonical_original)
        affine_checks = validate_affine_readout(start, affine, canonical_original, fixed)
        target = load_targets(paths["targets"], trace)
        original_metrics = direct_metrics(canonical_original, target)
        trace_command = np.asarray([r["joint_targets_rad"] for r in trace["samples"]])
        trace_tendon = np.asarray([[r["tendon_inputs"][leg + "_tarsus"] for leg in LEGS]
                                  for r in trace["samples"]])
        with np.load(paths["native"], allow_pickle=False) as native:
            native_ltm_activation = native["activation"][:, ltm]
        command_difference = canonical_original["command"][1:] - trace_command[1:]
        readout_change = {"canonical_vs_E150_monotone_command_rms": float(np.sqrt(
            np.mean(command_difference**2))),
            "canonical_vs_E150_monotone_command_max_abs": float(np.max(
                np.abs(command_difference)))}
        fixed_ltm_residual = {"activation_max_abs_vs_native": float(np.max(
            np.abs(canonical_original["activation"][:, ltm] - native_ltm_activation))),
            "tendon_max_abs_vs_trace": float(np.max(np.abs(
                canonical_original["tendon"] - trace_tendon)))}
        if elapsed() >= RESOURCE["model_s"]:
            raise TimeoutError("E209 model deadline before validity verdict")
        result.update(validity="valid_conditional_causal_basis",
            H209_implementation_valid=True, basis_checks=basis_checks,
            affine_checks=affine_checks,
            canonical_original_metrics=original_metrics,
            canonical_vs_monotone_readout_change=readout_change,
            fixed_ltm_native_residual=fixed_ltm_residual,
            affine_q_max_abs=float(np.max(np.abs(
                affine["q_probe_affine"] - affine["q_probe_direct"]))))
    except BaseException as error:  # noqa: BLE001 - raw/error retention is mandatory
        primary = error
        errors.append({"stage": phase, "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    finally:
        if context is not None:
            ledger["cuda_requested"] = context.calls_requested
            ledger["cuda_synchronized"] = context.calls_completed
            ledger["parameter_transfer_count"] = context.parameter_transfer_count
            ledger["parameter_upload_bytes"] = context.parameter_upload_bytes
            ledger["conditional_ticks_requested"] = context.calls_requested * context.ticks
            ledger["conditional_ticks_synchronized"] = context.calls_completed * context.ticks
            ledger["launch_events"] = context.launch_events
            ledger["peak_owned_cupy_pool_bytes"] = context.peak_owned_pool_bytes
            ledger["peak_device_used_bytes"] = context.peak_device_used_bytes
            ledger["baseline_device_used_bytes"] = context.baseline_device_used_bytes
            if context.current_output is not None:
                partial = _attempt(errors, "partial-readback", context.partial_readback)
                if partial is not None:
                    arrays, readback_errors = partial
                    errors.extend({"stage": "partial-readback", **failure}
                                  for failure in readback_errors)
                    _attempt(errors, "partial-raw", lambda: save_arrays(
                        out / (context.current_name + "-partial.npz"), arrays))
        ledger["phase"] = phase
        ledger["process_wall_s"] = elapsed()
        _attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    if primary is None and errors:
        primary = RuntimeError("E209 raw artifact persistence failed")
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0) + elapsed(),
        cuda_requested=ledger["cuda_requested"],
        cuda_synchronized=ledger["cuda_synchronized"], retries=0)
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    _attempt(errors, "execution-errors.json", lambda: save_json(
        out / "execution-errors.json", errors))
    preliminary = dict(result, validity="incomplete_or_invalid", finalization_pending=True)
    _attempt(errors, "result.json-preliminary", lambda: save_json(out / "result.json", preliminary))
    manifest = _attempt(errors, "output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E209 output missing or over 512 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E209 internal 85 s exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors or primary is not None:
        result["validity"] = "incomplete_or_invalid"
    result["finalization_pending"] = False
    _attempt(errors, "result.json-final", lambda: save_json(out / "result.json", result))
    if errors and primary is None:
        primary = RuntimeError("E209 final publication failed")
    if errors:
        _attempt(errors, "publication-failure.json", lambda: save_json(
            out / "publication-failure.json", errors))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", "--freeze-sha", dest="freeze_sha", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", "--review-sha", dest="review_sha", required=True)
    parser.add_argument("--out", "--output", dest="out", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.review, args.review_sha, args.out)


if __name__ == "__main__":
    main()
