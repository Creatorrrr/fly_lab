"""E207 two-call native threshold-initialization diagnostic; no fit or body."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fit_c_banc_m3_command_conditioned_response import command_diagonal_scale
from tools.fit_c_banc_m3_local_response import (
    DecoderAdjoint,
    PersistentLocalResponse,
    direct_metrics,
    feasible,
    load_targets,
    validate_response,
)
from tools.verify_c_banc_m3_local_response import (
    file_sha,
    repository_path,
    write_json,
)

FREEZE_SHA = "ade8d57c279ae1438c83c98af7e3b4e80611c40418dd84d3096dea0e0e019aff"
FREEZE_SCHEMA = "flylab.m3.e207.threshold-initialization.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e207.threshold-initialization.review.v1"
OWNED = ("tools/verify_c_banc_m3_threshold_initialization.py",
         "tests/test_c_banc_threshold_initialization.py")
RESPONSE_KEYS = ("rate", "adaptation", "rate_sensitivity",
                 "adaptation_sensitivity")
DECODED_KEYS = ("bank", "activation", "command", "tendon")


def silent_free_mask(response, weights, free, audit_columns):
    """Intersect exact all-time zeros in all four states with free zero weights."""
    weights = np.asarray(weights)
    free = np.asarray(free, np.int64)
    if weights.shape != (391, 7) or free.shape != (343,):
        raise ValueError("E207 frozen motor/weight mask changed")
    free_mask = np.zeros(391, dtype=bool)
    free_mask[free] = True
    eligible = free_mask & np.all(weights == 0, axis=1)
    for key in RESPONSE_KEYS:
        array = response[key]
        expected = (2001, 391) if key in {"rate", "adaptation"} else (2001, 391, 7)
        if array.shape != expected or array.dtype != np.float64:
            raise ValueError("E207 saved E205 response shape/dtype changed: " + key)
        eligible &= np.all(array == 0, axis=tuple(i for i in range(array.ndim)
                                                  if i != 1))
    columns = np.flatnonzero(eligible)
    if (len(columns) != 178
            or not np.array_equal(columns, np.asarray(audit_columns, np.int64))):
        raise ValueError("E207 zero-gradient eligible columns differ from audit")
    return eligible


def seed_bias(current_stage0, parameters, weights, eligible, *, rho=.01):
    """Invert the original instantaneous rectifier argument without clipping."""
    i0 = np.asarray(current_stage0, np.float64)
    weights = np.asarray(weights, np.float64)
    eligible = np.asarray(eligible, bool)
    if (i0.shape != (391,) or weights.shape != (391, 7)
            or eligible.shape != (391,) or int(np.count_nonzero(eligible)) != 178
            or not np.isfinite(i0).all() or not np.isfinite(weights).all()
            or np.any(weights[eligible] != 0)):
        raise ValueError("E207 seed input/mask invalid")
    c = np.asarray(parameters["coefficient"], np.float64)
    theta = np.asarray(parameters["threshold"], np.float64)
    z = np.asarray(parameters["initial_adaptation"], np.float64)
    cap = np.asarray(parameters["cap"], np.float64)
    r = np.asarray(parameters["initial_rate"], np.float64)
    if any(value.shape != (391,) or not np.isfinite(value).all()
           for value in (c, theta, z, cap, r)):
        raise ValueError("E207 original parameters/initial state invalid")
    s0 = c * (i0 - theta - z)
    chosen_cap = cap[eligible]
    if np.any(chosen_cap <= rho):
        raise ValueError("E207 cap is not above target instantaneous phi")
    v = 2 * (np.arctanh(rho / chosen_cap) - s0[eligible])
    if not np.isfinite(v).all() or np.any(np.abs(v) >= 1):
        raise ValueError("E207 seed inverse exceeds tanh domain")
    bias = np.arctanh(v)
    if not np.isfinite(bias).all() or np.any(np.abs(bias) > 1):
        raise ValueError("E207 seed bias exceeds original weight range")
    seeded = weights.copy()
    seeded[eligible, 0] = bias
    if (not np.array_equal(seeded[~eligible], weights[~eligible])
            or np.any(seeded[eligible, 1:] != 0)
            or not np.array_equal(r, np.asarray(parameters["initial_rate"], np.float64))
            or not np.array_equal(z, np.asarray(parameters["initial_adaptation"], np.float64))):
        raise ValueError("E207 seed changed noneligible weights/initial state")
    phi = chosen_cap * np.maximum(np.tanh(
        s0[eligible] + .5 * np.tanh(bias)), 0.)
    np.testing.assert_allclose(phi, rho, atol=1e-10, rtol=1e-10,
                               err_msg="E207 instantaneous phi inverse mismatch")
    return {"weights": seeded, "eligible": eligible,
        "s0": s0, "bias": bias, "phi0": phi,
        "original_initial_rate": r.copy(),
        "original_initial_adaptation": z.copy()}


def save_raw_response(path, response):
    """Write all available arrays before finite validation, including NaNs."""
    if not isinstance(response, dict):
        raise TypeError("E207 device returned no response mapping")
    np.savez_compressed(path, **{key: value for key, value in response.items()
                                 if isinstance(value, np.ndarray)})


def exact_response_subset(actual, baseline, selected):
    for key in RESPONSE_KEYS:
        a, b = actual[key], baseline[key]
        if (a.dtype != b.dtype or a.shape != b.shape
                or np.ascontiguousarray(a[:, selected]).tobytes()
                   != np.ascontiguousarray(b[:, selected]).tobytes()):
            raise ValueError("E207 unchanged MN response mismatch: " + key)


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E207 SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E207 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design") != "docs/BANC_M3_E207_THRESHOLD_INITIALIZATION_DESIGN_20260924.md"
            or file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256")
            or file_sha(repository_path(freeze["base_freeze"])) != freeze.get("base_freeze_sha256")
            or freeze.get("base_freeze_sha256") != "ebb917e0ab7c0882e3f2c604e29d06b6a45a7179cd7aba4d4dd67d636334434c"
            or freeze.get("initialization", {}).get("eligible_count") != 178
            or freeze.get("initialization", {}).get("target_initial_phi_hz") != .01
            or freeze.get("resource_limits", {}).get("cuda_calls") != 2
            or freeze.get("resource_limits", {}).get("internal_s") != 35
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E207 frozen contract changed")
    for label, count in (("input_sha256", 146),
                         ("reused_source_sha256", 14),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("Frozen E207 hash count changed")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E207 frozen input/source changed: " + name)
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256")
    if extra is not None and (not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != sha for name, sha in extra.items())):
        raise ValueError("Additional E207 reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != own):
        raise ValueError("E207 parent source review does not authorize these files")
    return freeze, review


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e207.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


def late_cell_metrics(seed_response, tangent, eligible, target_indices):
    columns = np.flatnonzero(eligible)
    indices = np.asarray(target_indices, np.int64)
    if len(columns) != 178 or indices.shape != (1824,):
        raise ValueError("E207 late-cell eligibility/target rows invalid")
    rate = seed_response["rate"][indices][:, columns]
    h_data = tangent["h"] - 2e-4 / 2401
    free = tangent["free_columns"]
    positions = np.searchsorted(free, columns)
    if (h_data.shape != (343, 7) or np.any(positions >= len(free))
            or not np.array_equal(free[positions], columns)):
        raise ValueError("E207 seed tangent/free mapping invalid")
    rms = np.sqrt(np.mean(rate * rate, axis=0))
    peak = np.max(rate, axis=0)
    maximum_h = np.max(h_data[positions], axis=1)
    recruited = (rms >= 1e-4) & (maximum_h >= 1e-6)
    if not all(np.isfinite(v).all() for v in (rms, peak, maximum_h)):
        raise ValueError("E207 late-cell metrics nonfinite")
    return {"motor_columns": columns, "late_rate_rms_hz": rms,
        "late_rate_max_hz": peak, "max_h_data": maximum_h,
        "recruited": recruited, "recruited_count": int(np.count_nonzero(recruited))}


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E207 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e207.ledger.v1", "device_calls_requested": 0,
        "device_calls_returned": 0, "new_neural_ticks": 0,
        "new_physics_steps": 0, "optimizer_calls": 0,
        "generated_warmup_calls": 0, "original_warmup_calls": 0,
        "errors": []}
    result = {"schema": "flylab.m3.e207.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "scope": "two_conditional_native_MN_responses_no_fit_or_body"}
    primary = None
    freeze = None
    device = None

    def elapsed():
        return time.perf_counter() - began

    def attempt(label, operation):
        nonlocal primary
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - retain first failure
            ledger["errors"].append({"stage": label,
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
            if primary is None:
                primary = error
            return None

    try:
        freeze, review = verify_gate(freeze_path, freeze_sha,
                                     review_path, review_sha)
        if shutil.disk_usage(out).free < 4 * 1024**3:
            raise OSError("E207 requires 4 GiB free disk")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        from flylab.c.storage import StateStore

        with threadpool_limits(limits=1):
            paths = {key: repository_path(value, file=key != "start")
                     for key, value in freeze["paths"].items()}
            write_json(out / "provenance.json", {"freeze_sha256": FREEZE_SHA,
                "review_sha256": review_sha, "source_sha256": review["source_sha256"],
                "input_sha256": freeze["input_sha256"]})
            current = np.load(paths["current"], mmap_mode="r", allow_pickle=False)
            if current.shape != (160000, 391) or current.dtype != np.float32:
                raise ValueError("E207 native stage current shape/dtype changed")
            with np.load(paths["parameters"], allow_pickle=False) as data:
                parameter = {key: data[key].copy() for key in data.files}
            start = StateStore.load(paths["start"])
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target = load_targets(paths["targets"], trace)
            with np.load(paths["mask"], allow_pickle=False) as data:
                free = data["free_columns"].copy()
                fixed = data["fixed_columns"].copy()
            with np.load(paths["e205_response"], allow_pickle=False) as data:
                stored_response = {key: data[key].copy() for key in RESPONSE_KEYS}
                baseline_weights = data["weights"].copy()
            with np.load(paths["e205_decoded"], allow_pickle=False) as data:
                stored_decoded = {key: data[key].copy() for key in DECODED_KEYS}
            with np.load(paths["e205_selected_raw"], allow_pickle=False) as data:
                selected_weights = data["weights"].copy()
            with np.load(paths["e205_preconditioner"], allow_pickle=False) as data:
                initial_d = data["D"].copy()
                preconditioner_weights = data["initial_weights"].copy()
            audit = json.loads(paths["zero_gradient_audit"].read_text(
                encoding="utf-8"))
            if (free.shape != (343,) or fixed.shape != (48,)
                    or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or not np.array_equal(baseline_weights, selected_weights)
                    or initial_d.shape != (2001, 391, 7)
                    or preconditioner_weights.shape != (391, 7)
                    or audit.get("zero_free_count") != 178):
                raise ValueError("E207 frozen E205 mask/weight/audit mismatch")
            eligible = silent_free_mask(stored_response, baseline_weights, free,
                                        audit["free_motor_columns"])
            if (np.any(initial_d[:, eligible] != 0)
                    or np.any(preconditioner_weights[eligible] != 0)):
                raise ValueError("E207 silent MN was active in saved preconditioner")
            seeded = seed_bias(current[0], parameter, baseline_weights, eligible)
            np.savez_compressed(out / "seed-weights.npz", **seeded,
                baseline_weights=baseline_weights,
                motor_ids=parameter["motor_ids"])
            p = {key: parameter[key] for key in (
                "initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain",
                "inverse_adaptation_tau")}
            original_r = p["initial_rate"].copy()
            original_z = p["initial_adaptation"].copy()
            device = PersistentLocalResponse(current, p)
            ledger["device_calls_requested"] += 1
            baseline = device.evaluate(baseline_weights)
            ledger["device_calls_returned"] += 1
            save_raw_response(out / "baseline-response.npz", baseline)
            validate_response(baseline, 2001, 391)
            for key in RESPONSE_KEYS:
                a, b = baseline[key], stored_response[key]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError("E207 baseline failed bit-exact E205 response: " + key)
            decoder = DecoderAdjoint(start)
            baseline_decoded = decoder.forward(baseline["rate"])
            np.savez_compressed(out / "baseline-decoder.npz", **baseline_decoded)
            for key in DECODED_KEYS:
                np.testing.assert_allclose(baseline_decoded[key], stored_decoded[key],
                    atol=2e-12, rtol=1e-12,
                    err_msg="E207 baseline decoder differs from E205: " + key)
            ledger["baseline_fidelity_verified"] = True
            if elapsed() >= 35:
                raise TimeoutError("E207 internal 35 s deadline before seed call")
            ledger["device_calls_requested"] += 1
            response = device.evaluate(seeded["weights"])
            ledger["device_calls_returned"] += 1
            save_raw_response(out / "seeded-response.npz", response)
            validate_response(response, 2001, 391)
            if (p["initial_rate"].tobytes() != original_r.tobytes()
                    or p["initial_adaptation"].tobytes() != original_z.tobytes()):
                raise ValueError("E207 native initial r/z changed between calls")
            exact_response_subset(response, baseline, ~eligible)
            decoded = decoder.forward(response["rate"])
            np.savez_compressed(out / "seeded-decoder.npz", **decoded)
            ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
            if (np.ascontiguousarray(decoded["activation"][:, ltm]).tobytes()
                    != np.ascontiguousarray(
                        baseline_decoded["activation"][:, ltm]).tobytes()
                    or decoded["tendon"].tobytes()
                    != baseline_decoded["tendon"].tobytes()):
                raise ValueError("E207 fixed LTM/tendon changed")
            tangent = command_diagonal_scale(start, response, decoded, target, free)
            h_data = tangent["h"] - 2e-4 / 2401
            np.savez_compressed(out / "seeded-tangent.npz", **tangent,
                h_data=h_data)
            cells = late_cell_metrics(response, tangent, eligible, target["indices"])
            np.savez_compressed(out / "late-cell-metrics.npz", **cells)
            baseline_metrics = direct_metrics(baseline_decoded, target)
            seeded_metrics = direct_metrics(decoded, target)
            quantiles = {key: np.quantile(cells[key], [0, .25, .5, .75, 1]).tolist()
                for key in ("late_rate_rms_hz", "late_rate_max_hz", "max_h_data")}
            write_json(out / "metrics.json", {"baseline": baseline_metrics,
                "seeded": seeded_metrics,
                "eligible_count": int(np.count_nonzero(eligible)),
                "recruited_count": cells["recruited_count"],
                "late_quantiles": quantiles,
                "existing_candidate_criteria_pass": feasible(
                    seeded_metrics, target["baseline"]),
                "target_phi_hz": .01, "actual_initial_phi_hz": seeded["phi0"].tolist()})
            result["eligible_count"] = int(np.count_nonzero(eligible))
            result["recruited_count"] = cells["recruited_count"]
            result["existing_candidate_criteria_pass"] = feasible(
                seeded_metrics, target["baseline"])
            result["baseline_metrics"] = baseline_metrics
            result["seeded_metrics"] = seeded_metrics
            result["validity"] = ("supported_threshold_initialization_signal"
                if cells["recruited_count"] >= 89
                else "bounded_threshold_initialization_not_supported")
    except BaseException as error:  # noqa: BLE001 - partial raw outlives failure
        primary = error
        ledger["errors"].append({"stage": "run", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    if device is not None:
        ledger["device"] = device.stats
        ledger["cuda_launch_requested"] = device.stats.get("launch_requested")
        ledger["cuda_launch_synchronized"] = device.stats.get("launch_synchronized")
        ledger["cuda_launch_completion_unknown"] = device.stats.get(
            "launch_completion_unknown")
        ledger["conditional_ticks_completed"] = device.stats.get(
            "conditional_ticks_completed")
        ledger["stage_inputs_consumed"] = device.stats.get("stage_inputs_consumed")
    try:
        import psutil
        memory = psutil.Process().memory_info()
        ledger["peak_resident_bytes"] = int(getattr(memory, "peak_wset", memory.rss))
        if ledger["peak_resident_bytes"] > 4 * 1024**3:
            primary = primary or MemoryError("E207 4 GiB resident cap exceeded")
    except ImportError:
        ledger["peak_resident_bytes"] = None
    ledger["process_wall_s"] = elapsed()
    result["process_wall_s"] = elapsed()
    result["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    attempt("ledger.json", lambda: write_json(out / "ledger.json", ledger))
    preliminary = {**result, "validity": "incomplete_or_invalid",
                   "finalization_pending": True}
    attempt("result.json-preliminary", lambda: write_json(out / "result.json", preliminary))
    manifest = attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > 256 * 1024**2:
        primary = primary or RuntimeError("E207 output manifest/256 MiB limit failed")
    if elapsed() >= 35:
        primary = primary or TimeoutError("E207 internal 35 s deadline exceeded")
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
