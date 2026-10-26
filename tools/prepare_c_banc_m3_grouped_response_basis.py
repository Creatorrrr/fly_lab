"""E217 one CUDA grouped-basis observation on frozen signed stage inputs."""

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

from tools.banc_m3_causal_neural_basis import compute_scales
from tools.banc_m3_grouped_response_basis import (
    BASIS_COUNT,
    PROBE,
    PersistentGroupedInputs,
    group_affine_readout,
    run_grouped_basis_cuda,
    select_grouped_prototypes,
)
from tools.banc_m3_multiscale_neural_basis import validate_inverse_slow
from tools.fit_c_banc_m3_local_response import direct_metrics, load_targets
from tools.prepare_c_banc_m3_causal_neural_basis import save_arrays
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "d7ffb20ce2207c64d8def8d3a32c6a81c2dd535a8d690e543e49eee167854e78"
FREEZE_SCHEMA = "flylab.m3.e217.grouped-response-basis.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e217.grouped-response-basis.review.v1"
OWNED = ("tools/banc_m3_grouped_response_basis.py",
         "tools/prepare_c_banc_m3_grouped_response_basis.py",
         "tests/test_c_banc_grouped_response_basis.py")
SHAPE = {"motors": 391, "free_motors": 343, "fixed_motors": 48,
    "muscles": 96, "free_muscles": 90, "fixed_muscles": 6,
    "basis_count": 64, "descriptor_width": 9, "samples": 2001,
    "stage_rows": 160000, "future_fit_coefficients": 5760}
REACTION = {"fast_tau_s": .016,
    "slow": "actual E201 inverse_adaptation_tau float32 values cast float64",
    "gate": "unchanged E213 fast-only", "beta_min": 0.0,
    "beta_max": 1.0, "probe": "(j+1)/65",
    "sharing": "same original muscle; own inputs and own memories only at runtime"}
PROTOTYPES = {"count": 64, "first_row": 0,
    "tie": "first unselected row",
    "distance": "sum_m pool_owned_m*cap_m/G_owner * sum_a(delta_descriptor_ma**2)",
    "sigma": "max(1e-4,median strictly positive nearest-other-center squared distances)); all zero fallback1",
    "target_input": False}
TOLERANCE = {"history_atol": 2e-12, "history_rtol": 2e-12,
    "group_envelope_atol": 2e-10, "group_envelope_rtol": 2e-12,
    "probe_rate_atol": 2e-9, "probe_rate_rtol": 2e-11,
    "probe_q_atol": 2e-10, "probe_q_rtol": 2e-12,
    "B_min": -1e-12}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "original_attempts": 1,
    "cuda_launches": 1, "history_launches": 0, "optimizer_calls": 0,
    "retries": 0, "internal_s": 90, "external_s": 110,
    "supervisor_s": 120, "gpu_bytes": 2147483648,
    "resident_bytes": 5368709120, "output_bytes": 805306368,
    "minimum_free_bytes": 4294967296}


def repository_path(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E217 repository-relative slash file required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E217 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E217 frozen file missing or escaped")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E217 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E217 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E217_GROUPED_RESPONSE_BASIS_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != SHAPE or freeze.get("reaction") != REACTION
            or freeze.get("prototypes") != PROTOTYPES
            or freeze.get("tolerance") != TOLERANCE
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("clock") != {"neural_dt_s": .00025,
                "control_dt_s": .005, "ticks_per_control": 20,
                "activation_tau_s": .01}
            or freeze.get("generated_cuda") != {"actual_launches_max": 1,
                "neural_ticks_max": 40, "external_s": 30,
                "original_data": False}
            or freeze.get("new_full_banc") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E217 scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E217 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E217 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E217 parent source review does not authorize files")
    for label, count in (("input_sha256", 20),
                         ("reused_source_sha256", 30),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E217 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E217 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E217 hash gate exceeded internal deadline")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "parameters", "endpoints", "positive", "negative", "drive",
            "mask", "targets", "trace", "history", "old_basis",
            "old_affine", "basis_parameters", "canonical_original"}:
        raise ValueError("E217 frozen original input map changed")
    return freeze, review, {key: repository_path(value)
                            for key, value in paths.items()}


def validate_raw_basis(basis, history, initial, cap, old_basis):
    """Raw arrays are saved by the runner before any of these scientific gates."""
    samples, n = history["descriptor"].shape[:2]
    expected = {"h_fast": (samples, n, 3), "h_slow": (samples, n, 3),
        "descriptor": (samples, n, 9), "B": (samples, n, 64),
        "H": (samples, n), "H_factor": (samples, n),
        "probe_rate": (samples, n)}
    if set(basis) != set(expected):
        raise ValueError("E217 raw basis keys changed")
    for key, shape in expected.items():
        value = basis[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("E217 raw basis shape/dtype/nonfinite: " + key)
    errors = {}
    for key in ("h_fast", "h_slow", "descriptor"):
        reference = history[key]
        errors[key + "_max_abs"] = float(np.max(np.abs(basis[key] - reference)))
        if not np.allclose(basis[key], reference,
                           atol=TOLERANCE["history_atol"],
                           rtol=TOLERANCE["history_rtol"]):
            raise ValueError("E217 reused E213 history differs: " + key)
    for key in ("H", "H_factor"):
        errors[key + "_max_abs"] = float(np.max(np.abs(basis[key] - old_basis[key])))
        if not np.allclose(basis[key], old_basis[key],
                           atol=TOLERANCE["history_atol"],
                           rtol=TOLERANCE["history_rtol"]):
            raise ValueError("E217 unchanged homogeneous state differs: " + key)
    B = basis["B"]
    factor = basis["H_factor"]
    reconstructed = basis["H"] + cap[None] * np.sum(B * PROBE, axis=2)
    errors["probe_rate_max_abs"] = float(np.max(np.abs(
        reconstructed - basis["probe_rate"])))
    errors["B_sum_old_max_abs"] = float(np.max(np.abs(
        np.sum(B, axis=2) - np.sum(old_basis["B"], axis=2))))
    if (np.min(B) < TOLERANCE["B_min"]
            or np.max(np.sum(B, axis=2) - (1 - factor)) > 1e-10
            or np.min(basis["probe_rate"]) < -2e-9
            or np.max(basis["probe_rate"] - cap[None]) > 2e-9
            or not np.array_equal(basis["H"][0], initial)
            or not np.allclose(reconstructed, basis["probe_rate"],
                               atol=TOLERANCE["probe_rate_atol"],
                               rtol=TOLERANCE["probe_rate_rtol"])
            or not np.allclose(np.sum(B, axis=2),
                               np.sum(old_basis["B"], axis=2),
                               atol=TOLERANCE["group_envelope_atol"],
                               rtol=TOLERANCE["group_envelope_rtol"])):
        raise ValueError("E217 positive basis/probe/envelope contract failed")
    return errors


def validate_group_affine(affine, old_affine, membership, fixed, ltm,
                          canonical_original):
    """Compare group envelope and direct readout, keeping fixed LTM exact."""
    D = affine["D_group"]
    samples = len(D)
    if (D.shape != (samples, 96, BASIS_COUNT)
            or old_affine["D"].shape != (samples, 391, 16)
            or np.any(~np.isfinite(D)) or np.min(D) < -1e-12):
        raise ValueError("E217 group pooled basis shape/finite/positive changed")
    old_sum = np.sum(old_affine["D"], axis=2)
    old_group = np.zeros((samples, 96), np.float64)
    for t in range(samples):
        np.add.at(old_group[t], membership, old_sum[t])
    new_sum = np.sum(D, axis=2)
    fixed_groups = np.unique(membership[fixed])
    if len(fixed_groups) != 6 or np.any(D[:, fixed_groups] != 0):
        raise ValueError("E217 fixed six LTM group basis must remain zero")
    activation = affine["activation_probe"]
    old_activation = old_affine["activation_intercept"]
    if (ltm.shape != (6,) or not np.array_equal(
            np.sort(fixed), np.flatnonzero(np.isin(membership, ltm)))):
        raise ValueError("E217 LTM fixed ownership changed")
    if (not np.allclose(new_sum, old_group,
            atol=TOLERANCE["group_envelope_atol"],
            rtol=TOLERANCE["group_envelope_rtol"])
            or not np.array_equal(affine["q_intercept"],
                                  old_affine["q_intercept"])
            or not np.array_equal(affine["tendon_intercept"],
                                  old_affine["tendon_intercept"])
            or not np.allclose(affine["q_probe_affine"],
                               affine["q_probe_direct"],
                               atol=TOLERANCE["probe_q_atol"],
                               rtol=TOLERANCE["probe_q_rtol"])
            or activation.shape != old_activation.shape
            or not np.array_equal(activation[:, ltm],
                                  canonical_original["activation"][:, ltm])
            or not np.array_equal(old_activation[:, ltm],
                                  canonical_original["activation"][:, ltm])
            or not np.array_equal(affine["tendon_probe"],
                                  canonical_original["tendon"])
            or not np.array_equal(affine["tendon_intercept"],
                                  canonical_original["tendon"])):
        raise ValueError("E217 group envelope or canonical probe mismatch")
    return {"group_envelope_max_abs": float(np.max(np.abs(new_sum-old_group))),
        "probe_q_max_abs": float(np.max(np.abs(
            affine["q_probe_affine"] - affine["q_probe_direct"]))),
        "fixed_group_count": 6, "fixed_ltm_activation_exact": True,
        "fixed_ltm_tendon_exact": True}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e217.output-manifest.v1", "files": files,
                "total_bytes": sum(item["bytes"] for item in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def _attempt(errors, stage, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - retain primary failure
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E217 output exists: " + str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    freeze = review = paths = context = None
    phase = "gate"
    ledger = {"schema": "flylab.m3.e217.ledger.v1", "phase": phase,
        "cuda_requested": 0, "cuda_synchronized": 0,
        "history_launches": 0, "warmup_calls": 0,
        "optimizer_calls": 0, "full_banc_calls": 0, "physics_steps": 0,
        "upload_bytes": None, "channel_transfer_count": None,
        "parameter_transfer_count": None, "parameter_upload_bytes": None,
        "planned_output_bytes": None, "planned_parameter_bytes": None,
        "planned_peak_gpu_bytes": None, "peak_owned_cupy_pool_bytes": None,
        "peak_device_used_bytes": None, "baseline_device_used_bytes": None,
        "basis_kernel_wall_s": None, "launch_events": []}
    result = {"schema": "flylab.m3.e217.grouped-response-basis.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "fit_calls": 0, "full_banc_calls": 0, "new_physics_steps": 0}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E217 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        phase = "input"
        positive = np.load(paths["positive"], mmap_mode="r", allow_pickle=False)
        negative = np.load(paths["negative"], mmap_mode="r", allow_pickle=False)
        drive = np.load(paths["drive"], mmap_mode="r", allow_pickle=False)
        if any(v.shape != (160000, 391) or v.dtype != np.float32
               for v in (positive, negative, drive)):
            raise ValueError("E217 actual signed-channel dimensions changed")
        with (np.load(paths["parameters"], allow_pickle=False) as p,
              np.load(paths["endpoints"], allow_pickle=False) as endpoints,
              np.load(paths["mask"], allow_pickle=False) as mask,
              np.load(paths["basis_parameters"], allow_pickle=False) as old_p):
            initial = p["initial_rate"].astype(np.float64)
            tau = p["tau"].astype(np.float64)
            cap = p["cap"].astype(np.float64)
            inverse_f32 = p["inverse_adaptation_tau"].copy()
            inverse = inverse_f32.astype(np.float64)
            motor_ids = p["motor_ids"].copy()
            membership = p["membership"].astype(np.int64)
            pool = p["pool"].astype(np.float64)
            moment = p["moment"].astype(np.float64)
            rest = p["rest"].astype(np.float64)
            ltm = p["ltm"].astype(np.int64)
            groups = json.loads(str(p["muscle_parameters_json"]))
            original_rates = endpoints["motor_rate"].astype(np.float64)
            free = mask["free_columns"].astype(np.int64)
            fixed = mask["fixed_columns"].astype(np.int64)
            scale = old_p["scale"].copy()
            if (inverse_f32.dtype != np.float32
                    or not np.array_equal(inverse, inverse_f32.astype(np.float64))
                    or not np.array_equal(initial, old_p["initial_rate"])
                    or not np.array_equal(tau, old_p["tau"])
                    or not np.array_equal(cap, old_p["cap"])
                    or not np.array_equal(motor_ids, old_p["motor_ids"])
                    or not np.array_equal(free, old_p["free_columns"])
                    or not np.array_equal(fixed, old_p["fixed_columns"])
                    or initial.shape != (391,) or tau.shape != (391,)
                    or cap.shape != (391,) or scale.shape != (391, 3)
                    or membership.shape != (391,) or pool.shape != (96, 391)
                    or moment.shape != (42, 96) or rest.shape != (42,)
                    or ltm.shape != (6,) or len(groups) != 96
                    or original_rates.shape != (2001, 391)
                    or not np.array_equal(original_rates[0], initial)
                    or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or not np.array_equal(fixed,
                        np.flatnonzero(np.isin(membership, ltm)))
                    or not np.array_equal(mask["motor_ids"], motor_ids)
                    or not np.array_equal(mask["membership"], membership)
                    or not np.array_equal(mask["ltm"], ltm)
                    or np.any(scale <= 0) or np.any(tau < .00025 * 50)
                    or np.any(cap <= 0) or np.any(initial < 0)
                    or np.any(initial > cap) or np.any(original_rates < 0)
                    or np.any(original_rates > cap[None])
                    or not all(np.isfinite(v).all() for v in (
                        initial, tau, cap, inverse, scale, pool, moment,
                        rest, original_rates))):
                raise ValueError("E217 frozen motor identity or physiology changed")
        validate_inverse_slow(inverse, 391)
        if compute_scales(positive, negative, drive).tobytes() != scale.tobytes():
            raise ValueError("E217 E213 scale differs from signed stage inputs")
        with np.load(paths["history"], allow_pickle=False) as old_h:
            history = {key: old_h[key].copy() for key in (
                "h_fast", "h_slow", "descriptor")}
        if (history["h_fast"].shape != (2001, 391, 3)
                or history["h_slow"].shape != (2001, 391, 3)
                or history["descriptor"].shape != (2001, 391, 9)
                or not all(np.isfinite(v).all() for v in history.values())):
            raise ValueError("E217 pinned E213 history invalid")
        with np.load(paths["old_basis"], allow_pickle=False) as old_b:
            old_basis = {key: old_b[key].copy() for key in ("B", "H", "H_factor")}
        with np.load(paths["old_affine"], allow_pickle=False) as old_a:
            old_affine = {key: old_a[key].copy() for key in (
                "D", "q_intercept", "tendon_intercept", "rate_intercept",
                "activation_intercept")}
        with np.load(paths["canonical_original"], allow_pickle=False) as old_c:
            canonical_original = {key: old_c[key].copy() for key in (
                "command", "activation", "tendon")}
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        if len(trace["samples"]) != 2001:
            raise ValueError("E217 original full 2001-control trace required")
        target = load_targets(paths["targets"], trace)
        save_arrays(out / "parameters.npz", {"scale": scale,
            "initial_rate": initial, "tau": tau, "cap": cap,
            "inverse_adaptation_tau": inverse, "motor_ids": motor_ids,
            "membership": membership, "free_columns": free,
            "fixed_columns": fixed, "ltm": ltm, "pool": pool,
            "moment": moment, "rest": rest})
        phase = "prototypes"
        prototypes = select_grouped_prototypes(
            history["descriptor"], membership, pool, cap, count=BASIS_COUNT)
        save_arrays(out / "prototypes.npz", prototypes)
        if (prototypes["anchor_rows"].shape != (96, 64)
                or prototypes["centers"].shape != (391, 64, 9)
                or prototypes["sigma2"].shape != (391,)
                or np.any(prototypes["sigma2"] <= 0)
                or any(len(np.unique(rows)) != 64
                       for rows in prototypes["anchor_rows"])):
            raise ValueError("E217 geometry shape/uniqueness failed")
        if elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E217 deadline before GPU upload")
        phase = "upload"
        context = PersistentGroupedInputs(
            positive, negative, drive, gpu_limit=RESOURCE["gpu_bytes"])
        ledger.update(upload_bytes=context.upload_bytes,
            channel_transfer_count=context.transfer_count,
            planned_output_bytes=context.planned_output_bytes,
            planned_parameter_bytes=context.planned_parameter_bytes,
            planned_peak_gpu_bytes=context.planned_peak_bytes)
        if elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E217 deadline before CUDA launch")
        phase = "basis"
        save_json(out / "ledger.json", dict(ledger, cuda_requested=1,
            launch_state="requested_before_kernel"))
        basis, wall = run_grouped_basis_cuda(context, scale, inverse,
            prototypes, initial, tau, cap)
        ledger["basis_kernel_wall_s"] = wall
        save_arrays(out / "basis.npz", basis)
        if elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E217 deadline after basis raw persistence")
        basis_checks = validate_raw_basis(basis, history, initial, cap, old_basis)
        phase = "readout"
        affine = group_affine_readout(basis, cap, membership, pool,
            moment, rest, ltm, groups, free, fixed, original_rates, old_affine)
        save_arrays(out / "affine-readout.npz", affine)
        affine_checks = validate_group_affine(affine, old_affine,
            membership, fixed, ltm, canonical_original)
        original_metrics = direct_metrics(canonical_original, target)
        trace_command = np.asarray(
            [row["joint_targets_rad"] for row in trace["samples"]])
        readout_difference = canonical_original["command"][1:] - trace_command[1:]
        if elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E217 deadline before validity verdict")
        result.update(validity="valid_conditional_grouped_basis",
            H217_implementation_valid=True, basis_checks=basis_checks,
            affine_checks=affine_checks,
            canonical_original_metrics=original_metrics,
            canonical_vs_monotone_readout_change={
                "command_rms": float(np.sqrt(np.mean(readout_difference**2))),
                "command_max_abs": float(np.max(np.abs(readout_difference)))},
            grouped_future_coefficients=5760,
            model_family="64 common anchors and muscle-shared beta; no fit")
    except BaseException as error:  # noqa: BLE001 - raw and primary error survive
        primary = error
        errors.append({"stage": phase, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    finally:
        if context is not None:
            ledger["cuda_requested"] = context.calls_requested
            ledger["cuda_synchronized"] = context.calls_completed
            ledger["conditional_ticks_requested"] = context.calls_requested * context.ticks
            ledger["conditional_ticks_synchronized"] = context.calls_completed * context.ticks
            ledger["parameter_transfer_count"] = context.parameter_transfer_count
            ledger["parameter_upload_bytes"] = context.parameter_upload_bytes
            ledger["launch_events"] = context.launch_events
            ledger["peak_owned_cupy_pool_bytes"] = context.peak_owned_pool_bytes
            ledger["peak_device_used_bytes"] = context.peak_device_used_bytes
            ledger["baseline_device_used_bytes"] = context.baseline_device_used_bytes
            if context.current_output is not None:
                partial = _attempt(errors, "partial-readback", context.partial_readback)
                if partial is not None:
                    arrays, failures = partial
                    errors.extend({"stage": "partial-readback", **item}
                                  for item in failures)
                    _attempt(errors, "partial-raw", lambda: save_arrays(
                        out / "basis-partial.npz", arrays))
        ledger["phase"] = phase
        ledger["process_wall_s"] = elapsed()
        _attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    if primary is None and errors:
        primary = RuntimeError("E217 required raw artifact persistence failed")
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
        result["resource_failure"] = "E217 output missing or over 768 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E217 internal 90 s exceeded"
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
            primary = RuntimeError("E217 final publication failed")
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
    try:
        result = run(args.freeze, args.freeze_sha256,
                     args.review, args.review_sha256, args.out)
    except BaseException as error:  # noqa: BLE001 - explicit nonzero original failure
        print(f"E217 invalid: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
