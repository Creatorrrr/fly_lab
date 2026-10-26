"""E213 one bounded conditional two-timescale CUDA preparation and readout."""

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

from tools.banc_m3_multiscale_neural_basis import (
    PersistentMultiscaleInputs,
    compute_scales,
    run_basis_cuda,
    run_history_cuda,
    select_prototypes_9d,
    validate_inverse_slow,
)
from tools.fit_c_banc_m3_local_response import direct_metrics, load_targets
from tools.prepare_c_banc_m3_causal_neural_basis import (
    affine_readout,
    canonical_readout,
    save_arrays,
    validate_affine_readout,
)
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "0f3482398bdaf79172dfbde7230d75d4effffa3a1ad10e1baf72f1f10c02dd7f"
FREEZE_SCHEMA = "flylab.m3.e213.multiscale-neural-basis.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e213.multiscale-neural-basis.review.v1"
OWNED = ("tools/banc_m3_multiscale_neural_basis.py",
         "tools/prepare_c_banc_m3_multiscale_neural_basis.py",
         "tests/test_c_banc_multiscale_neural_basis.py")
SHAPE = {"motors": 391, "free_motors": 343, "fixed_motors": 48,
    "stages": 160000, "ticks_each": 40000, "samples": 2001,
    "descriptor_width": 9, "basis_count": 16}
RESPONSE = {"fast_tau": .016,
    "slow_inverse": "E201 actual float32 inverse_adaptation_tau cast to float64",
    "initial_fast_slow": 0.0, "gate": "E209 fast memory only unchanged",
    "dt": .00025, "control_ticks": 20, "beta_bounds": [0.0, 1.0],
    "frozen_scale": "E209 parameters.npz",
    "prototype_rule": "E209 row0-first FPS and strict-positive median sigma2 in 9D"}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "original_attempts": 1,
    "cuda_launches": 2, "optimizer_calls": 0, "retries": 0,
    "model_s": 65, "internal_s": 85, "external_s": 95,
    "supervisor_s": 100, "gpu_bytes": 1610612736,
    "resident_bytes": 6442450944, "output_bytes": 536870912,
    "minimum_free_bytes": 4294967296}


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E213 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E213 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E213 path")
    if file and not path.is_file():
        raise ValueError("E213 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E213 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E213 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E213_MULTISCALE_NEURAL_BASIS_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != SHAPE or freeze.get("response") != RESPONSE
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("new_full_banc") != 0 or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E213 frozen science/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E213 design SHA changed")
    sources = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(relative)) != expected
            for relative, expected in extra.items()):
        raise ValueError("E213 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != sources):
        raise ValueError("E213 source review does not authorize these sources")
    for label, count in (("input_sha256", 120),
                         ("reused_source_sha256", 17),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E213 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E213 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E213 preflight exhausted model deadline")
    paths = freeze.get("paths", {})
    if not isinstance(paths, dict) or set(paths) != {
            "current", "parameters", "endpoints", "observation", "trace",
            "native", "start", "e201_manifest", "e201_result", "e201_review",
            "positive", "negative", "drive", "signed_result",
            "signed_manifest", "signed_review", "targets", "target_freeze",
            "mask", "e209_parameters", "e209_history", "e209_basis",
            "e209_affine"}:
        raise ValueError("E213 frozen original path map changed")
    return freeze, review, {key: repository_path(value, file=key != "start")
                            for key, value in paths.items()}


def _attempt(errors, name, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve independent raw
        errors.append({"stage": name, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def validate_history(history, scale, inverse, n, samples):
    expected = {"h_fast": (samples, n, 3), "h_slow": (samples, n, 3),
                "descriptor": (samples, n, 9)}
    if set(history) != set(expected) or scale.shape != (n, 3):
        raise ValueError("E213 history raw keys or scale shape changed")
    validate_inverse_slow(inverse, n)
    for key, shape in expected.items():
        value = history[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("E213 history shape/dtype/nonfinite: " + key)
    if (not np.array_equal(history["h_fast"][0], np.zeros((n, 3)))
            or not np.array_equal(history["h_slow"][0], np.zeros((n, 3)))
            or not np.array_equal(history["descriptor"][0, :, 3:], np.zeros((n, 6)))):
        raise ValueError("E213 initial two memories changed")


def validate_basis(history, basis, initial, cap):
    n = len(initial)
    count = len(history["h_fast"])
    expected = {"h_fast": (count, n, 3), "h_slow": (count, n, 3),
        "descriptor": (count, n, 9), "B": (count, n, 16),
        "H": (count, n), "H_factor": (count, n),
        "probe_rate": (count, n)}
    if set(basis) != set(expected):
        raise ValueError("E213 basis raw keys changed")
    for key, shape in expected.items():
        value = basis[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("E213 basis shape/dtype/nonfinite: " + key)
    for key in ("h_fast", "h_slow", "descriptor"):
        if history[key].tobytes() != basis[key].tobytes():
            raise ValueError("E213 two CUDA history launches differ: " + key)
    if (np.min(basis["B"]) < -1e-12 or np.min(basis["H_factor"]) < 0
            or np.max(np.sum(basis["B"], axis=2) -
                      (1 - basis["H_factor"])) > 1e-10):
        raise ValueError("E213 positive basis mass bound failed")
    beta = (np.arange(16) + 1) / 17
    reconstructed = basis["H"] + cap[None] * np.sum(basis["B"] * beta, axis=2)
    if not np.allclose(reconstructed, basis["probe_rate"],
                       atol=2e-10, rtol=1e-11):
        raise ValueError("E213 independent probe mismatches affine basis")
    if (np.min(basis["probe_rate"]) < -2e-10
            or np.max(basis["probe_rate"] - cap) > 2e-10
            or not np.array_equal(basis["H"][0], initial)):
        raise ValueError("E213 original initial/cap physiology changed")
    return {"max_probe_affine_abs": float(np.max(np.abs(
        reconstructed - basis["probe_rate"]))),
        "min_B": float(np.min(basis["B"])),
        "max_basis_mass_excess": float(np.max(
            np.sum(basis["B"], axis=2) - (1 - basis["H_factor"])))}


def validate_e209_invariants(history, basis, affine, old_history, old_basis,
                             old_affine):
    for new, old, label in (
            (history["h_fast"], old_history["h"], "history h_fast"),
            (basis["h_fast"], old_basis["h"], "basis h_fast"),
            (basis["H"], old_basis["H"], "homogeneous H"),
            (basis["H_factor"], old_basis["H_factor"], "homogeneous factor"),
            (affine["q_intercept"], old_affine["q_intercept"], "q intercept"),
            (affine["tendon_intercept"], old_affine["tendon_intercept"],
             "fixed tendon intercept")):
        if new.tobytes() != old.tobytes():
            raise ValueError("E213 E209 unchanged bytes differ: " + label)
    if (not np.allclose(np.sum(basis["B"], axis=2),
                        np.sum(old_basis["B"], axis=2),
                        atol=2e-10, rtol=2e-11)
            or not np.allclose(np.sum(affine["D"], axis=2),
                               np.sum(old_affine["D"], axis=2),
                               atol=2e-10, rtol=2e-11)):
        raise ValueError("E213 E209 output envelope changed")
    return {"h_fast_H_H_factor_q_intercept_tendon_exact": True,
        "max_B_sum_abs": float(np.max(np.abs(
            np.sum(basis["B"], axis=2) - np.sum(old_basis["B"], axis=2)))),
        "max_D_sum_abs": float(np.max(np.abs(
            np.sum(affine["D"], axis=2) - np.sum(old_affine["D"], axis=2))))}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e213.output-manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def collect_conditional(out, context, scale, inverse, initial, tau, cap,
                        *, sample_every=20, kernel_walls=None,
                        before_basis=None, after_basis=None):
    """Save both launches before validation; history failure blocks basis."""
    out = Path(out)
    history, wall = run_history_cuda(context, scale, inverse,
                                     sample_every=sample_every)
    if kernel_walls is not None:
        kernel_walls["history_kernel_wall_s"] = wall
    save_arrays(out / "history.npz", history)
    validate_history(history, scale, inverse, context.n,
                     context.ticks // sample_every + 1)
    prototypes = select_prototypes_9d(history["descriptor"])
    save_arrays(out / "prototypes.npz", prototypes)
    if before_basis is not None:
        before_basis()
    basis, wall = run_basis_cuda(context, scale, inverse, prototypes,
                                 initial, tau, cap, sample_every=sample_every)
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
        raise FileExistsError("One-shot E213 output exists: " + str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    freeze = review = paths = context = None
    phase = "gate"
    ledger = {"schema": "flylab.m3.e213.ledger.v1", "phase": phase,
        "cuda_requested": 0, "cuda_synchronized": 0, "warmup_calls": 0,
        "optimizer_calls": 0, "full_banc_calls": 0, "physics_steps": 0,
        "upload_bytes": None, "channel_transfer_count": None,
        "parameter_transfer_count": None, "parameter_upload_bytes": None,
        "planned_output_bytes": None, "planned_parameter_bytes": None,
        "planned_peak_gpu_bytes": None, "peak_owned_cupy_pool_bytes": None,
        "peak_device_used_bytes": None, "baseline_device_used_bytes": None,
        "history_kernel_wall_s": None, "basis_kernel_wall_s": None,
        "model_elapsed_s": None, "launch_events": []}
    result = {"schema": "flylab.m3.e213.multiscale-neural-basis.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "fit_calls": 0, "full_banc_calls": 0, "new_physics_steps": 0}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E213 requires 4 GiB free disk")
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
        if any(v.shape != (160000, 391) or v.dtype != np.float32
               for v in (positive, negative, drive)):
            raise ValueError("E213 native signed channel shape/dtype changed")
        with (np.load(paths["parameters"], allow_pickle=False) as p,
              np.load(paths["endpoints"], allow_pickle=False) as endpoints,
              np.load(paths["mask"], allow_pickle=False) as mask,
              np.load(paths["e209_parameters"], allow_pickle=False) as old):
            initial = p["initial_rate"].astype(np.float64)
            tau = p["tau"].astype(np.float64)
            cap = p["cap"].astype(np.float64)
            inverse_f32 = p["inverse_adaptation_tau"].copy()
            inverse = inverse_f32.astype(np.float64)
            motor_ids = p["motor_ids"].copy()
            original_rates = endpoints["motor_rate"].astype(np.float64)
            free = mask["free_columns"].astype(np.int64)
            fixed = mask["fixed_columns"].astype(np.int64)
            scale = old["scale"].copy()
            if (inverse_f32.dtype != np.float32
                    or not np.array_equal(inverse, inverse_f32.astype(np.float64))
                    or not np.array_equal(initial, old["initial_rate"])
                    or not np.array_equal(tau, old["tau"])
                    or not np.array_equal(cap, old["cap"])
                    or not np.array_equal(motor_ids, old["motor_ids"])
                    or not np.array_equal(free, old["free_columns"])
                    or not np.array_equal(fixed, old["fixed_columns"])
                    or initial.shape != (391,) or tau.shape != (391,)
                    or cap.shape != (391,) or original_rates.shape != (2001, 391)
                    or scale.shape != (391, 3) or np.any(scale <= 0)
                    or np.any(tau < .00025 * 50) or np.any(cap <= 0)
                    or np.any(initial < 0) or np.any(initial > cap)
                    or np.any(original_rates < 0)
                    or np.any(original_rates > cap[None])
                    or not all(np.isfinite(v).all() for v in (
                        initial, tau, cap, inverse, original_rates, scale))):
                raise ValueError("E213 original motor physiology or E209 scale changed")
            mask_arrays = {key: mask[key].copy() for key in mask.files}
        validate_inverse_slow(inverse, 391)
        if compute_scales(positive, negative, drive).tobytes() != scale.tobytes():
            raise ValueError("E213 frozen E209 scale differs from recorded signed inputs")
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
                or not np.array_equal(original_rates[0], initial)):
            raise ValueError("E213 pinned native MN identity/ownership changed")
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        if len(trace["samples"]) != 2001:
            raise ValueError("E213 original 2001 control trace required")
        save_arrays(out / "parameters.npz", {
            "scale": scale, "initial_rate": initial, "tau": tau, "cap": cap,
            "inverse_adaptation_tau": inverse, "motor_ids": motor_ids,
            "free_columns": free, "fixed_columns": fixed})
        if elapsed() >= RESOURCE["model_s"]:
            raise TimeoutError("E213 model deadline before CUDA")
        phase = "upload"
        context = PersistentMultiscaleInputs(
            positive, negative, drive, gpu_limit=RESOURCE["gpu_bytes"])
        ledger["upload_bytes"] = context.upload_bytes
        ledger["channel_transfer_count"] = context.transfer_count
        ledger["planned_output_bytes"] = context.planned_output_bytes
        ledger["planned_parameter_bytes"] = context.planned_parameter_bytes
        ledger["planned_peak_gpu_bytes"] = context.planned_peak_bytes
        phase = "history"

        def before_basis():
            nonlocal phase
            if elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E213 model deadline before basis CUDA")
            phase = "basis"

        def after_basis():
            ledger["model_elapsed_s"] = elapsed()
            if ledger["model_elapsed_s"] >= RESOURCE["model_s"]:
                raise TimeoutError("E213 model deadline after basis CUDA")

        history, _, basis, basis_checks = collect_conditional(
            out, context, scale, inverse, initial, tau, cap,
            kernel_walls=ledger, before_basis=before_basis, after_basis=after_basis)
        phase = "readout"
        affine = affine_readout(start, original_rates, basis, cap, free, fixed)
        save_arrays(out / "affine-readout.npz", affine)
        canonical_original = canonical_readout(start, original_rates)
        save_arrays(out / "canonical-original.npz", canonical_original)
        with (np.load(paths["e209_history"], allow_pickle=False) as old_history,
              np.load(paths["e209_basis"], allow_pickle=False) as old_basis,
              np.load(paths["e209_affine"], allow_pickle=False) as old_affine):
            old_checks = validate_e209_invariants(
                history, basis, affine, old_history, old_basis, old_affine)
        affine_checks = validate_affine_readout(
            start, affine, canonical_original, fixed)
        target = load_targets(paths["targets"], trace)
        original_metrics = direct_metrics(canonical_original, target)
        trace_command = np.asarray(
            [r["joint_targets_rad"] for r in trace["samples"]])
        trace_tendon = np.asarray(
            [[r["tendon_inputs"][leg + "_tarsus"] for leg in LEGS]
             for r in trace["samples"]])
        with np.load(paths["native"], allow_pickle=False) as native:
            native_ltm_activation = native["activation"][:, ltm]
        command_difference = canonical_original["command"][1:] - trace_command[1:]
        readout_change = {"canonical_vs_E150_monotone_command_rms":
            float(np.sqrt(np.mean(command_difference**2))),
            "canonical_vs_E150_monotone_command_max_abs":
            float(np.max(np.abs(command_difference)))}
        fixed_ltm_residual = {"activation_max_abs_vs_native": float(np.max(
            np.abs(canonical_original["activation"][:, ltm] - native_ltm_activation))),
            "tendon_max_abs_vs_trace": float(np.max(np.abs(
                canonical_original["tendon"] - trace_tendon)))}
        if elapsed() >= RESOURCE["model_s"]:
            raise TimeoutError("E213 model deadline before validity verdict")
        result.update(validity="valid_conditional_multiscale_basis",
            H213_implementation_valid=True, basis_checks=basis_checks,
            old_envelope_checks=old_checks, affine_checks=affine_checks,
            canonical_original_metrics=original_metrics,
            canonical_vs_monotone_readout_change=readout_change,
            fixed_ltm_native_residual=fixed_ltm_residual,
            affine_q_max_abs=float(np.max(np.abs(
                affine["q_probe_affine"] - affine["q_probe_direct"]))))
    except BaseException as error:  # noqa: BLE001 - preserve raw/error
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
        primary = RuntimeError("E213 raw artifact persistence failed")
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
    _attempt(errors, "result-preliminary", lambda: save_json(
        out / "result.json", dict(result, validity="incomplete_or_invalid",
                                  finalization_pending=True)))
    manifest = _attempt(errors, "output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E213 output missing or over 512 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E213 internal 85 s exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors or primary is not None:
        result["validity"] = "incomplete_or_invalid"
    result["finalization_pending"] = False
    try:
        save_json(out / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - preserve primary
        errors.append({"stage": "result-final", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if primary is None:
            primary = error
    if errors:
        _attempt(errors, "publication-failure.json",
                 lambda: save_json(out / "publication-failure.json", errors))
        if primary is None:
            primary = RuntimeError("E213 final publication failed")
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review,
        args.review_sha256, args.out)


if __name__ == "__main__":
    main()
