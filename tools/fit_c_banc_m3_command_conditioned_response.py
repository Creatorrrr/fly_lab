"""E205 bounded command-conditioned coordinates for the frozen local MN fit."""

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
    _ema,
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

FREEZE_SCHEMA = "flylab.m3.e205.command-conditioned-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e205.command-conditioned-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_command_conditioned_response.py",
         "tests/test_c_banc_command_conditioned_response.py")
FREEZE_SHA = "583b3948a508117f421a7010b33092f83ee8e19d5f795009a057e92b9722561c"
MAX_EVALUATIONS = 45


def actual_limit_objective(decoded, target, free_weights):
    return objective_and_command_gradient(decoded, target, free_weights, inner=1.)


def command_diagonal_scale(start, response, decoded, targets, free_columns):
    """Frozen decoder-local command Jacobian diagonal; no neural replay."""
    a = start["adapter"]
    p, c = a["parameters"], a["calibration"]
    membership = np.asarray(p["membership"], np.int64)
    pool = np.asarray(c["pool"], np.float64)
    moment = np.asarray(c["moment"], np.float64)
    fixed = np.asarray(p["fixed_muscles"], np.int64)
    kappa = np.asarray(p["kappa"], np.float64)
    slopes = np.asarray(p["slopes"], np.float64)
    bank = np.asarray(decoded["bank"], np.float64)
    rate_tangent = np.asarray(response["rate_sensitivity"], np.float64)
    free = np.asarray(free_columns, np.int64)
    indices = np.asarray(targets["indices"], np.int64)
    original_rms = float(targets["baseline"]["command_rms"])
    if (membership.shape != (391,) or pool.shape != (96, 391)
            or moment.shape != (42, 96) or fixed.shape != (19,)
            or kappa.shape != (391,) or slopes.shape != (96, 391, 3, 2)
            or bank.shape != (2001, 391, 3)
            or rate_tangent.shape != (2001, 391, 7)
            or rate_tangent.dtype != np.float64 or free.shape != (343,)
            or indices.shape != (1824,) or original_rms <= 0
            or np.any(membership < 0) or np.any(membership >= 96)
            or not np.isfinite(rate_tangent).all()
            or not np.isfinite(bank).all() or not np.isfinite(moment).all()
            or not np.isfinite(pool).all()):
        raise ValueError("Complete finite E205 saved decoder/tangent contract required")
    owner = np.zeros_like(pool, dtype=bool)
    owner[membership, np.arange(391)] = True
    if np.any(pool[~owner] != 0):
        raise ValueError("E205 decoder pool includes a non-owned MN")
    if np.any(rate_tangent[0] != 0):
        raise ValueError("E205 initial rate tangent is not zero")
    d = np.zeros_like(rate_tangent)
    fixed_motor = np.isin(membership, fixed)
    low = slopes[membership, np.arange(391), :, 0]
    high = slopes[membership, np.arange(391), :, 1]
    tau = np.asarray(p["tau"], np.float64)
    for k, period in enumerate(tau):
        alpha = -np.expm1(-.005 / period)
        filtered = _ema(rate_tangent[1:], alpha, np.zeros((391, 7)))
        slope = np.where(bank[:, :, k] > kappa[None, :],
                         high[None, :, k], low[None, :, k])
        d[:, ~fixed_motor] += slope[:, ~fixed_motor, None] * filtered[:, ~fixed_motor]
    pooled = _ema(rate_tangent[1:], -np.expm1(-.005 / .01), np.zeros((391, 7)))
    d[:, fixed_motor] = (pool[membership[fixed_motor],
                              np.flatnonzero(fixed_motor)][None, :, None]
                         * pooled[:, fixed_motor])
    if not np.isfinite(d).all() or np.any(d[0] != 0):
        raise ValueError("E205 command tangent invalid")
    coefficient = np.sum((.02 * moment[:, membership])**2, axis=0)
    denominator = len(indices) * 42 * (.8 * original_rms)**2
    h_all = (2 * coefficient[:, None] * np.sum(d[indices]**2, axis=0)
             / denominator + 2e-4 / 2401)
    h = h_all[free]
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("E205 command diagonal must be finite positive")
    exponent = np.clip(np.floor(-.5 * np.log2(h)), -30, 0).astype(np.int64)
    scale = np.exp2(exponent.astype(np.float64))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("E205 power-of-two scale invalid")
    return {"D": d, "h": h, "scale": scale,
            "exponent": exponent, "free_columns": free}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e205.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E205 SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E205 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design") != "docs/BANC_M3_E205_COMMAND_CONDITIONED_FIT_DESIGN_20260924.md"
            or file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256")
            or file_sha(repository_path(freeze["base_freeze"])) != freeze.get("base_freeze_sha256")
            or freeze.get("base_freeze_sha256") != "183b0efa52ecdc44c500b960b60cf50d2a5891996c0a7eee59023255e0b18f0c"
            or freeze.get("solver", {}).get("hard_cuda_evaluations") != 45
            or freeze.get("resource_limits", {}).get("internal_s") != 135
            or freeze.get("solver", {}).get("inner_guard") != 1.
            or freeze.get("solver", {}).get("gtol") != "min(scale)*1e-6"
            or freeze.get("solver", {}).get("maxls") != 30
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E205 frozen contract changed")
    for label, count in (("input_sha256", 117),
                         ("reused_source_sha256", 12),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("Frozen E205 hash count changed")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E205 frozen input/source changed: " + name)
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256")
    if extra is not None and (not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != sha for name, sha in extra.items())):
        raise ValueError("Additional E205 reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != own):
        raise ValueError("E205 parent source review does not authorize these files")
    return freeze, review


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E205 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e205.fit-ledger.v1", "optimizer_calls": 0,
        "cuda_evaluations": 0, "cache_hits": 0, "new_neural_ticks": 0,
        "new_physics_steps": 0, "generated_warmup_calls": 0,
        "original_warmup_calls": 0, "optimizer_requests": [], "errors": []}
    result = {"schema": "flylab.m3.e205.fit-result.v1",
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
            raise OSError("E205 requires 4 GiB free disk")
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
                raise ValueError("Frozen E205 anatomical free/fixed MN mask changed")
            target = load_targets(paths["targets"], trace)
            decoder = DecoderAdjoint(start)
            p = {key: parameter[key] for key in (
                "initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain",
                "inverse_adaptation_tau")}
            with np.load(paths["e204_response"], allow_pickle=False) as data:
                e204_response = {key: data[key].copy() for key in (
                    "rate", "adaptation", "rate_sensitivity",
                    "adaptation_sensitivity")}
                selected_weights = data["weights"].copy()
            with np.load(paths["e204_decoded"], allow_pickle=False) as data:
                e204_decoded = {key: data[key].copy() for key in (
                    "bank", "activation", "command", "tendon")}
            with np.load(paths["e204_initial_raw"], allow_pickle=False) as data:
                initial_gradient_w = data["gradient"].copy()
                initial_weights = data["weights"].copy()
                initial_free = data["free_weights"].copy()
                if (initial_free.shape != (2401,)
                        or initial_weights.shape != (391, 7)
                        or initial_gradient_w.shape != (2401,)
                        or not np.isfinite(initial_gradient_w).all()
                        or not np.isfinite(initial_weights).all()
                        or not np.array_equal(selected_weights, initial_weights)
                        or not np.array_equal(initial_weights[free].reshape(-1), initial_free)
                        or np.any(initial_weights[fixed] != 0)
                        or np.any(np.abs(initial_free) > 1)):
                    raise ValueError("E204 selected weight/gradient record invalid")
            initial_note = json.loads(Path(paths["e204_initial_note"]).read_text(
                encoding="utf-8"))
            initial_merit = float(initial_note["merit"])
            if not np.isfinite(initial_merit):
                raise ValueError("E204 selected merit invalid")
            preconditioner = command_diagonal_scale(
                start, e204_response, e204_decoded, target, free)
            scale = preconditioner["scale"].reshape(-1)
            inverse_scale = 1 / scale
            initial_u = initial_free / scale
            if (not np.array_equal(scale * initial_u, initial_free)
                    or np.any(np.abs(initial_u) > inverse_scale)):
                raise ValueError("E205 initial power-of-two coordinate roundtrip failed")
            np.savez_compressed(out / "preconditioner.npz", **preconditioner,
                initial_weights=initial_weights, initial_free_weights=initial_free,
                initial_scaled_variables=initial_u.reshape(343, 7))
            device = PersistentLocalResponse(current, p)

            def evaluate(scaled_vector):
                nonlocal best
                variables = np.ascontiguousarray(scaled_vector, dtype=np.float64)
                values = np.ascontiguousarray(scale * variables)
                if (variables.shape != (2401,) or not np.isfinite(variables).all()
                        or np.any(np.abs(variables) > inverse_scale)
                        or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
                    raise ValueError("Finite bounded E205 conditioned variables required")
                key = values.tobytes()
                request = {"order": len(ledger["optimizer_requests"]),
                    "wall_s": elapsed(), "cache_hit": key in cache,
                    "weight_sha256": hashlib.sha256(key).hexdigest()}
                ledger["optimizer_requests"].append(request)
                if key in cache:
                    ledger["cache_hits"] += 1
                    write_json(out / f"cache-hit-{request['order']:03}.json", request)
                    return cache[key]
                if elapsed() >= 120:
                    raise TimeoutError("E205 fit 120 s deadline exceeded")
                if len(evaluations) >= MAX_EVALUATIONS:
                    raise EvaluationLimitReached("E205 hard 45 CUDA evaluations reached")
                weights = np.zeros((391, 7), np.float64)
                weights[free] = values.reshape(343, 7)
                response = device.evaluate(weights)
                ledger["cuda_evaluations"] = device.stats["launch_synchronized"]
                try:
                    validate_response(response, e204_response["rate"].shape[0], 391)
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
                merit, gq, terms = actual_limit_objective(
                    decoded, target, values)
                g_rate = decoder.backward(gq, decoded)
                gradient_all = np.einsum("tm,tmk->mk", g_rate,
                                         response["rate_sensitivity"], optimize=True)
                gradient = gradient_all[free].reshape(-1)
                gradient += 2e-4 * values / len(values)
                if not np.isfinite(gradient).all():
                    raise ValueError("Nonfinite E205 analytic gradient")
                scaled_gradient = scale * gradient
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

            evaluate(initial_u)
            first = best
            for key, original in e204_response.items():
                actual = first["response"][key]
                if (actual.dtype != original.dtype or actual.shape != original.shape
                        or actual.tobytes() != original.tobytes()):
                    raise ValueError("E205 initial failed bit-exact E204 response: " + key)
            for key, original in e204_decoded.items():
                np.testing.assert_allclose(first["decoded"][key], original,
                    atol=2e-12, rtol=1e-12,
                    err_msg="E205 initial decoder mismatch: " + key)
            old_merit, old_gq, _old_terms = objective_and_command_gradient(
                first["decoded"], target, initial_free)
            old_rate_gradient = decoder.backward(old_gq, first["decoded"])
            old_gradient = np.einsum("tm,tmk->mk", old_rate_gradient,
                first["response"]["rate_sensitivity"], optimize=True)[free].reshape(-1)
            old_gradient += 2e-4 * initial_free / len(initial_free)
            np.testing.assert_allclose(old_merit, initial_merit,
                atol=2e-12, rtol=1e-12, err_msg="E205 initial old merit mismatches E204")
            np.testing.assert_allclose(old_gradient, initial_gradient_w,
                atol=2e-12, rtol=1e-12, err_msg="E205 initial old gradient mismatches E204")
            ledger["initial_reconstruction_verified"] = True
            ledger["optimizer_calls"] = 1
            try:
                optimized = minimize(evaluate, initial_u, jac=True,
                    method="L-BFGS-B", bounds=list(zip(-inverse_scale, inverse_scale)),
                    options={"maxiter": 40, "maxfun": 45, "maxls": 30,
                             "maxcor": 10, "ftol": 1e-12, "gtol": float(np.min(scale)) * 1e-6})
                result["optimizer"] = {"success": bool(optimized.success),
                    "status": int(optimized.status), "message": str(optimized.message),
                    "nit": int(optimized.nit), "nfev": int(optimized.nfev)}
            except EvaluationLimitReached as bounded:
                result["optimizer"] = {"success": False,
                    "status": "hard_cuda_evaluation_limit", "message": str(bounded)}
            if elapsed() >= 120:
                raise TimeoutError("E205 fit 120 s deadline exceeded")
            if best is None:
                raise ValueError("No finite E205 evaluation selected")
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
            scaled_variables=best["scaled_variables"], scale=scale.reshape(343, 7)))
        attempt("selected-decoder.npz", lambda: np.savez_compressed(
            out / "selected-decoder.npz", **best["decoded"]))
        attempt("selection.json", lambda: write_json(out / "selection.json", {
            "selected": best["record"], "baseline": target["baseline"],
            "evaluation_count": len(evaluations),
            "status": ("verified_candidate" if primary is None
                       and ledger.get("initial_reconstruction_verified")
                       else "unverified_incomplete_raw"),
            "initial_reconstruction_verified": bool(
                ledger.get("initial_reconstruction_verified", False))}))
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
        primary = primary or RuntimeError("E205 output manifest/256 MiB limit failed")
        result["validity"] = "incomplete_or_invalid"
    if elapsed() >= 135:
        primary = primary or TimeoutError("E205 internal 135 s deadline exceeded")
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
