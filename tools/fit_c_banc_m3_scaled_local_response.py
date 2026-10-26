"""E204 one bounded fit in scaled coordinates of the frozen E203 local model."""

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

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fit_c_banc_m3_local_response import (
    DecoderAdjoint,
    EvaluationLimitReached,
    PersistentLocalResponse,
    direct_metrics,
    feasible,
    load_targets,
    objective_and_command_gradient,
    validate_response,
)
from tools.verify_c_banc_m3_local_response import (
    file_sha,
    load_original,
    repository_path,
    write_json,
)

FREEZE_SCHEMA = "flylab.m3.e204.scaled-local-response-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e204.scaled-local-response-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_scaled_local_response.py",
         "tests/test_c_banc_scaled_local_response.py")
FREEZE_SHA = "183b0efa52ecdc44c500b960b60cf50d2a5891996c0a7eee59023255e0b18f0c"
SCALE = 0.0003115579064523788
BOUND = 1 / SCALE
MAX_EVALUATIONS = 60


def scaled_to_native(variables):
    return SCALE * np.asarray(variables, np.float64)


def native_to_scaled_gradient(gradient):
    return SCALE * np.asarray(gradient, np.float64)


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e204.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E204 SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E204 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design") != "docs/BANC_M3_E204_SCALED_LOCAL_RESPONSE_DESIGN_20260924.md"
            or file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256")
            or file_sha(repository_path(freeze["base_freeze"])) != freeze.get("base_freeze_sha256")
            or freeze.get("base_freeze_sha256") != "a069feb1f6b740acb5eab10828ddaaafe616b3ec0111387e5ed7a0bea8b239c7"
            or freeze.get("solver", {}).get("hard_cuda_evaluations") != 60
            or freeze.get("resource_limits", {}).get("internal_s") != 210
            or freeze.get("solver", {}).get("coordinate_scale") != SCALE
            or freeze.get("solver", {}).get("scaled_bound") != BOUND
            or freeze.get("solver", {}).get("gtol") != SCALE * 1e-6
            or freeze.get("solver", {}).get("maxls") != 30
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E204 frozen contract changed")
    for label, count in (("input_sha256", 106),
                         ("reused_source_sha256", 10),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("Frozen E204 hash count changed")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E204 frozen input/source changed: " + name)
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256")
    if extra is not None and (not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != sha for name, sha in extra.items())):
        raise ValueError("Additional E204 reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != own):
        raise ValueError("E204 parent source review does not authorize these files")
    return freeze, review


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E204 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e204.fit-ledger.v1", "optimizer_calls": 0,
        "cuda_evaluations": 0, "cache_hits": 0, "new_neural_ticks": 0,
        "new_physics_steps": 0, "generated_warmup_calls": 0,
        "original_warmup_calls": 0, "optimizer_requests": [], "errors": []}
    result = {"schema": "flylab.m3.e204.fit-result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    freeze = None
    cache = {}
    evaluations = []
    best = None
    device = None
    parameter = None
    target = None

    def elapsed():
        return time.perf_counter() - began

    def attempt(label, operation):
        nonlocal primary
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - retain partial raw
            ledger["errors"].append({"stage": label, "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc()})
            if primary is None:
                primary = error
            return None

    try:
        freeze, review = verify_gate(freeze_path, freeze_sha,
                                     review_path, review_sha)
        if shutil.disk_usage(out).free < 4 * 1024**3:
            raise OSError("E204 requires 4 GiB free disk")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            paths = {key: repository_path(value, file=key != "start")
                     for key, value in freeze["paths"].items()}
            write_json(out / "provenance.json", {"freeze_sha256": FREEZE_SHA,
                "review_sha256": review_sha, "source_sha256": review["source_sha256"],
                "input_sha256": freeze["input_sha256"]})
            current, parameter, _endpoints, start, trace, _native = load_original(paths)
            with np.load(paths["mask"], allow_pickle=False) as data:
                mask = {key: data[key].copy() for key in data.files}
            free = np.asarray(mask["free_columns"], np.int64)
            fixed = np.asarray(mask["fixed_columns"], np.int64)
            membership = np.asarray(parameter["membership"], np.int64)
            ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
            expected_fixed = np.flatnonzero(np.isin(membership, ltm))
            if (free.shape != (343,) or fixed.shape != (48,)
                    or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or not np.array_equal(free, np.sort(free))
                    or not np.array_equal(fixed, expected_fixed)
                    or not np.array_equal(mask["trainable_mask"],
                                          np.isin(np.arange(391), free))
                    or not np.array_equal(mask["membership"], membership)
                    or not np.array_equal(mask["ltm"], ltm)
                    or not np.array_equal(mask["motor_ids"], parameter["motor_ids"])
                    or mask["motor_roster"].tolist() !=
                       start["monotone_recruitment_patch"]["motor_roster"]):
                raise ValueError("Frozen E204 anatomical free/fixed MN mask changed")
            target = load_targets(paths["targets"], trace)
            decoder = DecoderAdjoint(start)
            p = {key: parameter[key] for key in (
                "initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain",
                "inverse_adaptation_tau")}
            device = PersistentLocalResponse(current, p)
            with np.load(paths["e202_response"], allow_pickle=False) as data:
                e202_response = {key: data[key].copy() for key in (
                    "rate", "adaptation", "rate_sensitivity",
                    "adaptation_sensitivity")}
            with np.load(paths["e202_decoded"], allow_pickle=False) as data:
                e202_decoded = {key: data[key].copy() for key in (
                    "bank", "activation", "command", "tendon")}
            with np.load(paths["e203_initial_raw"], allow_pickle=False) as data:
                initial_gradient_w = data["gradient"].copy()
                if (data["free_weights"].shape != (2401,)
                        or not np.array_equal(data["free_weights"], np.zeros(2401))
                        or initial_gradient_w.shape != (2401,)
                        or not np.isfinite(initial_gradient_w).all()):
                    raise ValueError("E203 initial weight/gradient record invalid")
            initial_note = json.loads(Path(paths["e203_initial_note"]).read_text(
                encoding="utf-8"))
            initial_merit = float(initial_note["merit"])
            if (not np.isfinite(initial_merit)
                    or not np.isclose(1 / np.max(np.abs(initial_gradient_w)), SCALE,
                                      atol=0, rtol=1e-12)):
                raise ValueError("E204 frozen gradient scale is inconsistent")

            def evaluate(scaled_vector):
                nonlocal best
                variables = np.ascontiguousarray(scaled_vector, dtype=np.float64)
                values = np.ascontiguousarray(scaled_to_native(variables))
                if (variables.shape != (2401,) or not np.isfinite(variables).all()
                        or np.any(np.abs(variables) > BOUND)
                        or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
                    raise ValueError("Finite bounded E204 scaled variables required")
                key = values.tobytes()
                request = {"order": len(ledger["optimizer_requests"]),
                    "wall_s": elapsed(), "cache_hit": key in cache,
                    "weight_sha256": hashlib.sha256(key).hexdigest()}
                ledger["optimizer_requests"].append(request)
                if key in cache:
                    ledger["cache_hits"] += 1
                    write_json(out / f"cache-hit-{request['order']:03}.json", request)
                    return cache[key]
                if elapsed() >= 180:
                    raise TimeoutError("E204 fit 180 s deadline exceeded")
                if len(evaluations) >= MAX_EVALUATIONS:
                    raise EvaluationLimitReached("E204 hard 60 CUDA evaluations reached")
                weights = np.zeros((391, 7), np.float64)
                weights[free] = values.reshape(343, 7)
                response = device.evaluate(weights)
                ledger["cuda_evaluations"] = device.stats["launch_synchronized"]
                try:
                    validate_response(response, e202_response["rate"].shape[0], 391)
                except (TypeError, ValueError) as invalid:
                    invalid_index = len(evaluations)
                    try:
                        raw = {key: value for key, value in response.items()
                               if isinstance(value, np.ndarray)}
                        np.savez_compressed(out / f"invalid-response-{invalid_index:03}.npz",
                                            weights=weights, free_weights=values,
                                            scaled_variables=variables, **raw)
                        write_json(out / f"invalid-response-{invalid_index:03}.json",
                            {"validity": "incomplete_or_invalid",
                             "reason": str(invalid), "wall_s": elapsed()})
                    except BaseException as storage_error:  # noqa: BLE001
                        ledger["errors"].append({"stage": "invalid-response-raw",
                            "type": type(storage_error).__name__,
                            "message": str(storage_error)})
                    raise
                decoded = decoder.forward(response["rate"])
                merit, gq, terms = objective_and_command_gradient(
                    decoded, target, values)
                g_rate = decoder.backward(gq, decoded)
                gradient_all = np.einsum("tm,tmk->mk", g_rate,
                                         response["rate_sensitivity"], optimize=True)
                gradient = gradient_all[free].reshape(-1)
                gradient += 2e-4 * values / len(values)
                if not np.isfinite(gradient).all():
                    raise ValueError("Nonfinite E204 analytic gradient")
                scaled_gradient = native_to_scaled_gradient(gradient)
                metrics = direct_metrics(decoded, target)
                accepted = feasible(metrics, target["baseline"])
                serial = len(evaluations)
                record = {"index": serial, "merit": merit, "terms": terms,
                    "metrics": metrics, "feasible": accepted,
                    "wall_s": elapsed(), "cache_hit": False,
                    "weight_sha256": hashlib.sha256(key).hexdigest()}
                np.savez_compressed(out / f"evaluation-{serial:03}.npz",
                    weights=weights, free_weights=values, gradient=gradient,
                    scaled_variables=variables, scaled_gradient=scaled_gradient,
                    command=decoded["command"])
                write_json(out / f"evaluation-{serial:03}.json", record)
                evaluations.append(record)
                cache[key] = (merit, scaled_gradient.copy())
                choose = (best is None
                    or accepted and not best["record"]["feasible"]
                    or accepted and best["record"]["feasible"]
                       and metrics["command_rms"] < best["record"]["metrics"]["command_rms"]
                    or not accepted and not best["record"]["feasible"]
                       and merit < best["record"]["merit"])
                if choose:
                    best = {"record": record, "response": response,
                            "decoded": decoded, "weights": weights,
                            "gradient": gradient.copy(),
                            "scaled_variables": variables.copy()}
                return cache[key]

            zero = np.zeros(2401, np.float64)
            evaluate(zero)
            first = best
            for key, original in e202_response.items():
                actual = first["response"][key]
                if (actual.dtype != original.dtype or actual.shape != original.shape
                        or actual.tobytes() != original.tobytes()):
                    raise ValueError("E204 w0 failed bit-exact E202 response: " + key)
            for key, original in e202_decoded.items():
                np.testing.assert_allclose(first["decoded"][key], original,
                    atol=2e-12, rtol=1e-12,
                    err_msg="E204 w0 decoder mismatch: " + key)
            np.testing.assert_allclose(first["record"]["merit"], initial_merit,
                atol=2e-12, rtol=1e-12, err_msg="E204 w0 merit mismatches E203")
            np.testing.assert_allclose(first["gradient"], initial_gradient_w,
                atol=2e-12, rtol=1e-12, err_msg="E204 w0 gradient mismatches E203")
            ledger["w0_reconstruction_verified"] = True
            ledger["optimizer_calls"] = 1
            try:
                optimized = minimize(evaluate, zero, jac=True,
                    method="L-BFGS-B", bounds=[(-BOUND, BOUND)] * 2401,
                    options={"maxiter": 40, "maxfun": 60, "maxls": 30,
                             "maxcor": 10, "ftol": 1e-12, "gtol": SCALE * 1e-6})
                result["optimizer"] = {"success": bool(optimized.success),
                    "status": int(optimized.status), "message": str(optimized.message),
                    "nit": int(optimized.nit), "nfev": int(optimized.nfev)}
            except EvaluationLimitReached as bounded:
                result["optimizer"] = {"success": False,
                    "status": "hard_cuda_evaluation_limit", "message": str(bounded)}
            if elapsed() >= 180:
                raise TimeoutError("E204 fit 180 s deadline exceeded")
            if best is None:
                raise ValueError("No finite E204 evaluation selected")
            result["validity"] = ("supported_conditional_local_response_proposal"
                if best["record"]["feasible"] else "valid_bounded_search_no_proposal")
            result["selected_evaluation"] = best["record"]["index"]
            result["selected_metrics"] = best["record"]["metrics"]
            result["selected_merit"] = best["record"]["merit"]
    except BaseException as error:  # noqa: BLE001 - raw outlives failure
        primary = error
        ledger["errors"].append({"stage": "run", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    if best is not None:
        attempt("selected-state.npz", lambda: np.savez_compressed(
            out / "selected-state.npz", **best["response"],
            weights=best["weights"], motor_ids=parameter["motor_ids"],
            scaled_variables=best["scaled_variables"]))
        attempt("selected-decoder.npz", lambda: np.savez_compressed(
            out / "selected-decoder.npz", **best["decoded"]))
        attempt("selection.json", lambda: write_json(out / "selection.json", {
            "selected": best["record"], "baseline": target["baseline"],
            "evaluation_count": len(evaluations),
            "status": ("verified_candidate" if primary is None
                       and ledger.get("w0_reconstruction_verified")
                       else "unverified_incomplete_raw"),
            "w0_reconstruction_verified": bool(
                ledger.get("w0_reconstruction_verified", False))}))
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
    if device is not None:
        ledger["cuda"] = device.stats
        ledger["cuda_evaluations"] = device.stats["launch_synchronized"]
    ledger["evaluation_count"] = len(evaluations)
    ledger["process_wall_s"] = elapsed()
    result["process_wall_s"] = elapsed()
    result["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    result["conditional_evaluations"] = ledger["cuda_evaluations"]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    attempt("ledger.json", lambda: write_json(out / "ledger.json", ledger))
    preliminary = {**result, "validity": "incomplete_or_invalid",
                   "finalization_pending": True}
    attempt("result.json-preliminary", lambda: write_json(out / "result.json", preliminary))
    manifest = attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > 256 * 1024**2:
        primary = primary or RuntimeError("E204 output manifest/256 MiB limit failed")
        result["validity"] = "incomplete_or_invalid"
    if elapsed() >= 210:
        primary = primary or TimeoutError("E204 internal 210 s deadline exceeded")
        result["validity"] = "incomplete_or_invalid"
    if primary is None:
        result["finalization_pending"] = False
        attempt("result.json-final", lambda: write_json(out / "result.json", result))
    if primary is not None:
        attempt("publication-failure.json", lambda: write_json(
            out / "publication-failure.json", ledger["errors"]))
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.review, args.review_sha, args.output)


if __name__ == "__main__":
    main()
