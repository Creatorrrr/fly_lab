"""E200 saved-command capacity of twelve silent muscles; no neural or body run."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
SILENT = np.array([8, 17, 21, 34, 37, 42, 47, 52, 65, 66, 71, 82], np.int64)
MAX_SECONDS = 15.0
MAX_BYTES = 8 * 1024 * 1024
SAVED_KEYS = ("target_indices", "target", "recorded_command",
              "loaded_adapter_moment", "loaded_adapter_rest")
NATIVE_KEYS = ("control_tick", "network_tick", "physics_time_s", "complete",
               "activation_available", "activation")
ADAPTER_KEYS = ("moment", "rest")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(relative: str) -> Path:
    require(isinstance(relative, str), "path must be text")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "source path escapes repository")
    return path


def array(record, name, shape, dtype=None):
    require(name in record, f"missing {name}")
    value = np.asarray(record[name])
    require(value.shape == shape, f"{name} shape {value.shape}, expected {shape}")
    if dtype is not None:
        require(value.dtype == np.dtype(dtype), f"{name} dtype {value.dtype}")
    require(value.dtype.kind in "biuf" and np.all(np.isfinite(value)), f"{name} nonfinite")
    return value


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def metrics(baseline: np.ndarray, residual: np.ndarray) -> dict:
    base_rms = rms(baseline)
    candidate_rms = rms(residual)
    return {"baseline_rms": base_rms, "candidate_rms": candidate_rms,
            "rms_ratio": None if base_rms == 0 else candidate_rms / base_rms,
            "squared_error_reduction": float(np.sum(baseline**2) - np.sum(residual**2))}


def analyze_arrays(saved: dict, native: dict, adapter: dict,
                   support_result: dict, silent=SILENT) -> tuple[dict, dict]:
    """Validate saved semantics, then solve optimistic per-boundary command corrections."""
    selected = np.asarray(silent)
    require(selected.ndim == 1 and selected.dtype == np.int64 and
            len(selected) > 0 and np.array_equal(selected, np.unique(selected)),
            "silent muscle roster")
    q = len(selected)
    require(support_result.get("validity") == "valid_saved_muscle_support" and
            support_result.get("run_status") == "completed", "E199 result status")
    zero = support_result["windows"]["all"]["activation_always_zero"]
    require(isinstance(zero, list) and
            [item["muscle_index"] for item in zero] == selected.tolist(),
            "E199 zero muscle roster differs")
    require(support_result["windows"]["all"]["ticks"] == [1, 2000],
            "E199 all window")

    indices = array(saved, "target_indices", (1824,), "int64")
    require(np.all((indices >= 1) & (indices <= 2000)) and
            np.all(np.diff(indices) > 0), "target indices order/range")
    target = array(saved, "target", (1824, 42)).astype(np.float64)
    recorded = array(saved, "recorded_command", (2001, 42)).astype(np.float64)
    saved_moment = array(saved, "loaded_adapter_moment", (42, 96))
    saved_rest = array(saved, "loaded_adapter_rest", (42,))
    moment = array(adapter, "moment", (42, 96))
    rest = array(adapter, "rest", (42,))
    require(np.array_equal(saved_moment, moment) and np.array_equal(saved_rest, rest),
            "E197/E150 adapter differs")
    require(np.array_equal(array(native, "control_tick", (2001,), "int64"),
                           np.arange(2001)), "native control ticks")
    require(np.array_equal(array(native, "network_tick", (2001,), "int64"),
                           20 * np.arange(2001)), "native network ticks")
    require(np.max(np.abs(array(native, "physics_time_s", (2001,)) -
                          .005 * np.arange(2001))) <= 1e-9, "native time grid")
    for key in ("complete", "activation_available"):
        require(np.all(array(native, key, (2001,), "bool")), f"native {key}")
    activation = array(native, "activation", (2001, 96)).astype(np.float64)
    require(np.all(activation >= 0) and np.all(activation[:, selected] == 0),
            "silent muscle activation is not exactly zero")
    reconstruction = rest[None, :] + .02 * (activation @ moment.T)
    initial_command_error = float(np.max(np.abs(reconstruction[0] - recorded[0])))
    command_error = float(np.max(np.abs(reconstruction[1:] - recorded[1:])))
    require(command_error <= 1e-12, "recorded command reconstruction")

    b = .02 * moment[:, selected]
    y = target - recorded[indices]
    coefficients = np.empty((len(indices), q), np.float64)
    for i, row in enumerate(y):
        coefficients[i] = nnls(b, row, maxiter=10 * q)[0]
    require(np.all(np.isfinite(coefficients)) and np.all(coefficients >= 0),
            "NNLS activation")
    nonnegative_correction = coefficients @ b.T
    nonnegative_residual = nonnegative_correction - y
    gradient = nonnegative_residual @ b
    gradient_min = float(np.min(gradient))
    complement_max = float(np.max(np.abs(coefficients * gradient)))
    require(gradient_min >= -1e-9 and complement_max <= 1e-9,
            "NNLS KKT residual")

    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    tolerance = max(b.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    signed_correction = (y @ u[:, :rank]) @ u[:, :rank].T
    signed_residual = signed_correction - y
    signed_orthogonality = float(np.max(np.abs(signed_residual @ b)))
    require(signed_orthogonality <= 1e-9, "signed projection orthogonality")

    whole_nnls = metrics(y, nonnegative_residual)
    whole_signed = metrics(y, signed_residual)
    legs = {}
    for i, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
        columns = slice(7 * i, 7 * (i + 1))
        legs[leg] = {"nnls": metrics(y[:, columns], nonnegative_residual[:, columns]),
                     "signed_span": metrics(y[:, columns], signed_residual[:, columns])}
    qvalues = np.quantile(coefficients, (0, .05, .5, .95, 1), axis=0,
                          method="linear").T
    per_muscle = [
        {"muscle_index": int(muscle),
         "activation_quantiles": [float(value) for value in qvalues[j]],
         "activation_exact_zero_fraction": float(np.mean(coefficients[:, j] == 0))}
        for j, muscle in enumerate(selected)
    ]
    ratio_nnls, ratio_signed = whole_nnls["rms_ratio"], whole_signed["rms_ratio"]
    if ratio_nnls is None:
        verdict = "inconclusive_zero_baseline"
    elif ratio_nnls <= .8 - 1e-9:
        verdict = "saved_nonnegative_command_headroom"
    elif ratio_signed is not None and ratio_signed > .8 + 1e-9:
        verdict = "reject_twelve_muscle_twenty_percent_hypothesis"
    else:
        verdict = "signed_relaxation_or_boundary_only"
    result = {"validity": "valid_saved_command_capacity",
              "hypothesis_verdict": verdict,
              "silent_muscle_indices": selected.astype(int).tolist(),
              "target_count": len(indices), "command_reconstruction_max_abs": command_error,
              "command_reconstruction_rows": [1, 2000],
              "initial_command_difference_max_abs": initial_command_error,
              "nnls_kkt_gradient_min": gradient_min,
              "nnls_kkt_complementarity_max_abs": complement_max,
              "nnls_maxiter": 10 * q, "svd_singular_values": singular.tolist(),
              "svd_rank_tolerance": float(tolerance), "svd_rank": rank,
              "signed_orthogonality_max_abs": signed_orthogonality,
              "whole": {"nnls": whole_nnls, "signed_span": whole_signed},
              "legs": legs, "per_muscle": per_muscle,
              "interpretation_limit": "Optimistic offline command-space bound; NNLS weights are not unique physiological activation or gait outcomes"}
    raw = {"B": b, "Y": y, "indices": indices,
           "nnls_activation": coefficients,
           "nnls_correction": nonnegative_correction,
           "nnls_residual": nonnegative_residual,
           "signed_correction": signed_correction,
           "signed_residual": signed_residual,
           "singular_values": singular, "baseline_command": recorded[indices],
           "target": target, "silent_muscle_indices": selected,
           "svd_rank": np.array(rank, np.int64)}
    return result, raw


def run(freeze_path: Path, freeze_sha: str, review_path: Path,
        review_sha: str, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    phase = "hash_gate"
    loaded: list[str] = []
    calculation_requested = False
    calculation_completed = False
    try:
        require(sha256(freeze_path) == freeze_sha, "freeze SHA mismatch")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        require(sha256(review_path) == review_sha, "review SHA mismatch")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        require(freeze.get("schema") == "flylab.m3.e200.silent-muscle-capacity.freeze.v1" and
                review.get("schema") == "flylab.m3.e200.silent-muscle-capacity.review.v1" and
                review.get("approved") is True and
                review.get("freeze_sha256") == freeze_sha and
                review.get("tool_sha256") == sha256(Path(__file__)) and
                review.get("test_sha256") == sha256(
                    ROOT / "tests/test_c_banc_silent_muscle_capacity.py"),
                "source review gate")
        require(freeze.get("owned_files") == [
                    "tools/analyze_c_banc_m3_silent_muscle_capacity.py",
                    "tests/test_c_banc_silent_muscle_capacity.py"], "owned files")
        require(freeze.get("silent_muscles") == SILENT.tolist() and
                freeze.get("shape") == {"samples": 2001, "targets": 1824,
                                        "muscles": 96, "joints": 42} and
                freeze.get("contract") == {"moment_scale": .02,
                                           "command_atol": 1e-12,
                                           "kkt_atol": 1e-9,
                                           "ratio_boundary": .8,
                                           "ratio_guard": 1e-9,
                                           "nnls_maxiter": 120,
                                           "svd_rank": "max_shape_times_eps_times_largest_singular_value",
                                           "reconstruction_rows": [1, 2000]},
                "freeze calculation contract")
        require(freeze.get("revision") == 2 and
                freeze.get("amendment") ==
                    "docs/BANC_M3_E200_INITIAL_COMMAND_RECOVERY_20260924.md" and
                sha256(source_path(freeze["amendment"])) == freeze["amendment_sha256"] and
                sha256(source_path(
                    "verification/banc-natural-gait-m3-20260919/"
                    "e200-silent-muscle-capacity-freeze.json")) ==
                    freeze["superseded_freeze_sha256"], "v2 recovery amendment")
        recovery = freeze["recovery"]
        require(recovery.get("total_E200_original_attempts") == 2 and
                recovery.get("manual_technical_retries") == 1 and
                recovery.get("automatic_retries") == 0 and
                sha256(source_path(
                    "verification/banc-natural-gait-m3-20260919/"
                    "parent-e200-supervisor.json")) == recovery["first_supervisor_sha256"] and
                sha256(source_path(
                    "verification/banc-natural-gait-m3-20260919/"
                    "parent-e200-reconstruction-diagnosis.json")) ==
                    recovery["diagnosis_sha256"], "v2 prior failure provenance")
        limits = freeze["resource_limits"]
        require(limits.get("cpu_threads") == 1 and limits.get("gpu_count") == 0 and
                limits.get("internal_s") == MAX_SECONDS and
                limits.get("output_bytes") == MAX_BYTES and
                limits.get("original_attempts") == 1 and limits.get("retries") == 0,
                "resource contract")
        design = freeze["design"]
        require(design == "docs/BANC_M3_E200_SILENT_MUSCLE_CAPACITY_DESIGN_20260924.md" and
                sha256(source_path(design)) == freeze["design_sha256"], "design SHA")
        for relative, digest in freeze["preservation_sha256"].items():
            require(sha256(source_path(relative)) == digest, "preservation SHA")
        roles = freeze["input_roles"]
        require(set(roles) == {"saved", "saved_manifest", "saved_review", "native",
                               "adapter", "native_manifest", "support_result",
                               "support_review"} and
                set(roles.values()) == set(freeze["input_sha256"]), "input roles")
        for role, relative in roles.items():
            require(sha256(source_path(relative)) == freeze["input_sha256"][relative],
                    f"{role} SHA")
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        phase = "load"
        with np.load(source_path(roles["saved"]), allow_pickle=False) as archive:
            saved = {key: archive[key] for key in SAVED_KEYS}
        loaded.append("saved")
        with np.load(source_path(roles["native"]), allow_pickle=False) as archive:
            native = {key: archive[key] for key in NATIVE_KEYS}
        loaded.append("native")
        with np.load(source_path(roles["adapter"]), allow_pickle=False) as archive:
            adapter = {key: archive[key] for key in ADAPTER_KEYS}
        loaded.append("adapter")
        support = json.loads(source_path(roles["support_result"]).read_text(encoding="utf-8"))
        loaded.append("support_result")
        support_review = json.loads(source_path(roles["support_review"]).read_text(encoding="utf-8"))
        loaded.append("support_review")
        saved_review = json.loads(source_path(roles["saved_review"]).read_text(encoding="utf-8"))
        loaded.append("saved_review")
        phase = "analysis"
        require(support_review.get("validity") == "valid_saved_diagnostic" and
                support_review.get("result_sha256") == freeze["input_sha256"][
                    roles["support_result"]] and
                support_review.get("independent_checks") == 302, "E199 final review")
        require(saved_review.get("validity") == "valid_rejection" and
                saved_review.get("original_attempts") == 1 and
                saved_review.get("retries") == 0, "E197 final review")
        require(support["source_inputs"]["native"]["sha256"] ==
                freeze["input_sha256"][roles["native"]] and
                support["source_inputs"]["adapter"]["sha256"] ==
                freeze["input_sha256"][roles["adapter"]], "E199/E150 source SHA")
        calculation_requested = True
        result, raw = analyze_arrays(saved, native, adapter, support)
        calculation_completed = True
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        phase = "publication"
        np.savez_compressed(output / "projection.npz", **raw)
        result.update({"experiment": "M3-E200", "run_status": "completed",
                       "freeze_sha256": freeze_sha, "review_sha256": review_sha,
                       "tool_sha256": review["tool_sha256"],
                       "test_sha256": review["test_sha256"],
                       "source_inputs": {role: {"path": relative,
                                                "sha256": freeze["input_sha256"][relative]}
                                         for role, relative in roles.items()},
                       "loaded_input_roles": loaded,
                       "calculation_requested": calculation_requested,
                       "calculation_completed": calculation_completed,
                       "original_calculation_count": 1,
                       "E200_total_original_attempts": 2,
                       "manual_technical_retries": 1,
                       "retries": 0, "nnls_solve_count": len(raw["indices"]),
                       "neural_parameter_fit_count": 0,
                       "neural_advance_count": 0, "physics_step_count": 0,
                       "fit_count": 0, "F6_exposed": False,
                       "candidate_adopted": False,
                       "prior_cumulative_s": freeze["prior_cumulative_s"],
                       "reserved_supervisor_s": limits["supervisor_s"],
                       "wall_seconds": float(time.perf_counter() - started),
                       "output_manifest_file": "output-manifest.json"})
        result_payload = (json.dumps(result, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8")
        files = {name: {"sha256": sha256(output / name),
                        "bytes": (output / name).stat().st_size}
                 for name in ("projection.npz",)}
        files["result.json"] = {"sha256": hashlib.sha256(result_payload).hexdigest(),
                                "bytes": len(result_payload)}
        manifest_payload = (json.dumps({"schema": "flylab.m3.e200.output-manifest.v1",
                                        "files": files}, indent=2) + "\n").encode("utf-8")
        require(files["projection.npz"]["bytes"] + len(result_payload) +
                len(manifest_payload) <= MAX_BYTES, "output byte limit")
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        pending_result = output / "result.json.pending"
        pending_manifest = output / "output-manifest.json.pending"
        pending_result.write_bytes(result_payload)
        pending_manifest.write_bytes(manifest_payload)
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        pending_manifest.rename(output / "output-manifest.json")
        pending_result.rename(output / "result.json")
        return result
    except Exception as exc:
        error = {"experiment": "M3-E200", "run_status": "failed",
                 "validity": "inconclusive", "phase": phase,
                 "error_type": type(exc).__name__, "error": str(exc),
                 "partial": any(output.iterdir()),
                 "loaded_input_roles": loaded,
                 "calculation_requested": calculation_requested,
                 "calculation_completed": calculation_completed,
                 "original_calculation_count": int(calculation_requested),
                 "E200_total_original_attempts": 2,
                 "manual_technical_retries": 1,
                 "retries": 0, "neural_advance_count": 0,
                 "physics_step_count": 0, "fit_count": 0,
                 "neural_parameter_fit_count": 0,
                 "freeze_sha256": freeze_sha, "review_sha256": review_sha,
                 "wall_seconds": float(time.perf_counter() - started)}
        (output / "error.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.out)
    print(json.dumps({"run_status": result["run_status"],
                      "validity": result["validity"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
