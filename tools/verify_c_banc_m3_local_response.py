"""One reviewed E202 conditional E150 MN reconstruction; no BANC simulation."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_local_response import simulate_cuda

FREEZE_SCHEMA = "flylab.m3.e202.local-response.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e202.local-response.review.v1"
OWNED = ("tools/banc_m3_local_response.py",
         "tools/verify_c_banc_m3_local_response.py",
         "tests/test_c_banc_local_response.py")
SHAPE = {"motors": 391, "muscles": 96, "controls": 2000, "samples": 2001,
         "stages": 160000, "conditional_ticks": 40000,
         "parameters_per_motor": 7}
CLOCK = {"neural_dt_s": .00025, "control_dt_s": .005,
         "ticks_per_control": 20}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "internal_s": 90,
            "external_s": 105, "supervisor_s": 120,
            "output_bytes": 256 * 1024**2,
            "gpu_buffer_bytes": 768 * 1024**2,
            "minimum_free_bytes": 4 * 1024**3,
            "original_attempts": 1, "retries": 0}
FIDELITY = {"rate_rms": .5, "rate_max": 5.,
            "activation_rms": .25, "activation_max": 2.5,
            "command_rms": .001, "command_max": .01,
            "tendon_rms": .0005, "tendon_max": .005}
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe frozen path: " + relative)
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped frozen path: " + relative)
    if file and not path.is_file():
        raise ValueError("Frozen file required: " + relative)
    return path


def write_json(path, value):
    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item
    Path(path).write_text(json.dumps(convert(value), indent=2,
        ensure_ascii=False, allow_nan=False), encoding="utf-8")


def verify_gate(freeze_path, freeze_sha, review_path, review_sha,
                *, elapsed=None):
    """Check all 88 inputs, six reused sources, M2 pin, and three owned files."""
    if file_sha(freeze_path) != freeze_sha:
        raise ValueError("E202 freeze SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E202 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E202_LOCAL_RESPONSE_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != SHAPE or freeze.get("clock") != CLOCK
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("fidelity") != FIDELITY
            or freeze.get("original_conditional_calls") != 1
            or freeze.get("optimizer_calls") != 0
            or freeze.get("new_neural") != 0 or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E202 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256"):
        raise ValueError("E202 design changed")
    if set(freeze.get("paths", {})) != {"current", "parameters", "endpoints",
            "observation", "trace", "native", "start", "e201_manifest",
            "e201_result", "e201_review"}:
        raise ValueError("Exact E202 path keys required")
    for key, value in freeze["paths"].items():
        repository_path(value, file=key != "start")
    for label, count in (("input_sha256", 88),
                         ("reused_source_sha256", 6),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E202 frozen " + label + " cardinality changed")
        for relative, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Malformed E202 SHA entry")
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E202 input/source changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E202 preflight exhausted internal deadline")
    source_hashes = {name: file_sha(repository_path(name)) for name in OWNED}
    additional = review.get("additional_reused_source_sha256")
    if additional is not None and (not isinstance(additional, dict) or not all(
            file_sha(repository_path(name)) == sha
            for name, sha in additional.items())):
        raise ValueError("Additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != source_hashes):
        raise ValueError("E202 source review does not authorize this runner")
    return freeze, review


def load_original(paths):
    """Read frozen data and checkpoint only; construct no model/body/graph."""
    from flylab.c.storage import StateStore

    observation = json.loads(paths["observation"].read_text(encoding="utf-8"))
    if (observation.get("start_tick") != 0 or observation.get("end_tick") != 40000
            or observation.get("capacity_stages") != 160000
            or observation.get("overflow") != [0]
            or observation.get("armed") != [0]
            or observation.get("counter_requested") != [160000] * 391):
        raise ValueError("Frozen E201 native stage observation invalid")
    current = np.load(paths["current"], mmap_mode="r", allow_pickle=False)
    if current.shape != (160000, 391) or current.dtype != np.float32:
        raise ValueError("Frozen E201 current shape/dtype changed")
    with np.load(paths["parameters"], allow_pickle=False) as data:
        parameter = {key: data[key].copy() for key in data.files}
    with np.load(paths["endpoints"], allow_pickle=False) as data:
        endpoints = {key: data[key].copy() for key in data.files}
    start = StateStore.load(paths["start"])
    if (start.get("control_tick") != 0 or start["network"]["tick"] != 0
            or not np.array_equal(parameter["motor_ids"], endpoints["motor_ids"])):
        raise ValueError("E150/E201 start or MN mapping changed")
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    with np.load(paths["native"], allow_pickle=False) as data:
        native = {key: data[key].copy() for key in data.files}
    e201_result = json.loads(paths["e201_result"].read_text(encoding="utf-8"))
    if e201_result.get("validity") != "valid_exact_native_stage_dataset":
        raise ValueError("E201 exact dataset not approved")
    return current, parameter, endpoints, start, trace, native


def decoder_from_snapshot(start, rates):
    """Apply frozen E150 two-layer decoder state without constructing adapters."""
    adapter = start["adapter"]
    calibration = adapter["calibration"]
    params = adapter["parameters"]
    rates = np.asarray(rates)
    if rates.shape[1] != 391 or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Finite nonnegative E150 local rates required")
    pool = calibration["pool"]
    moment = calibration["moment"]
    rest = calibration["rest"]
    ltm = calibration["ltm"]
    membership = params["membership"]
    kappa = params["kappa"]
    slopes = params["slopes"]
    fixed = params["fixed_muscles"]
    variable = np.setdiff1d(np.arange(96, dtype=np.int64), fixed)
    if (pool.shape != (96, 391) or moment.shape != (42, 96)
            or rest.shape != (42,) or ltm.shape != (6,)
            or membership.shape != (391,) or kappa.shape != (391,)
            or slopes.shape != (96, 391, 3, 2)
            or params["tau"].shape != (3,)
            or adapter["bank"].shape != (391, 3)
            or adapter["original"]["activation"].shape != (96,)):
        raise ValueError("Complete original E150 decoder snapshot required")
    n = len(rates)
    bank_rows = np.empty((n, 391, 3), np.float64)
    activation_rows = np.empty((n, 96), np.float64)
    command_rows = np.empty((n, 42), np.float64)
    tendon_rows = np.empty((n, 6), np.float64)
    bank = adapter["bank"].astype(np.float64, copy=True)
    original_activation = adapter["original"]["activation"].astype(np.float64, copy=True)
    active = adapter["activation"].astype(np.float64, copy=True)
    bank_rows[0], activation_rows[0] = bank, active
    command_rows[0] = np.asarray(start["body"]["action"]["joint_angles"], np.float64)
    alpha = -np.expm1(-.005 / params["tau"])
    original_alpha = -np.expm1(-.005 / .01)
    names = [adapter["groups"][int(index)]["leg"] + "_tarsus" for index in ltm]
    if len(set(names)) != 6 or set(names) != {leg + "_tarsus" for leg in LEGS}:
        raise ValueError("Six original named LTM tendons required")
    tendon_columns = [names.index(leg + "_tarsus") for leg in LEGS]
    tendon_rows[0] = .2 * np.tanh(active[ltm][tendon_columns] / 10)
    for control in range(1, n):
        rate = rates[control]
        original_activation += original_alpha * (pool @ rate - original_activation)
        bank += alpha[None, :] * (rate[:, None] - bank)
        active[fixed] = original_activation[fixed]
        hinge = np.maximum(bank - kappa[:, None], 0.)
        for muscle in variable:
            members = np.flatnonzero(membership == muscle)
            low = slopes[muscle, members, :, 0]
            delta = slopes[muscle, members, :, 1] - low
            active[muscle] = np.sum(low * bank[members]) + np.sum(delta * hinge[members])
        if not np.isfinite(active).all() or np.any(active < 0):
            raise ValueError("E150 decoded activation invalid")
        bank_rows[control], activation_rows[control] = bank, active
        command_rows[control] = rest + .02 * (moment @ active)
        tendon_rows[control] = .2 * np.tanh(active[ltm][tendon_columns] / 10)
    return {"bank": bank_rows, "activation": activation_rows,
            "command": command_rows, "tendon": tendon_rows,
            "tendon_names": np.asarray([leg + "_tarsus" for leg in LEGS])}


def error_metrics(actual, target):
    actual, target = np.asarray(actual), np.asarray(target)
    if actual.shape != target.shape or not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError("Finite aligned E202 comparison required")
    diff = actual.astype(np.float64) - target.astype(np.float64)
    return {"rms": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff)))}


def evaluate_raw(out, response, decoded, endpoints, trace, native):
    if (response["rate"].shape != (2001, 391)
            or response["adaptation"].shape != (2001, 391)
            or response["rate_sensitivity"].shape != (2001, 391, 7)
            or response["adaptation_sensitivity"].shape != (2001, 391, 7)
            or any(not np.isfinite(value).all() for value in response.values())):
        raise ValueError("E202 response/tangent shape or finite failure")
    if len(trace["samples"]) != 2001:
        raise ValueError("E150 full trace required")
    target_command = np.asarray([row["joint_targets_rad"] for row in trace["samples"]], np.float64)
    target_tendon = np.asarray([[row["tendon_inputs"][leg + "_tarsus"] for leg in LEGS]
                                for row in trace["samples"]], np.float64)
    metrics = {
        "rate": error_metrics(response["rate"], endpoints["motor_rate"]),
        "activation": error_metrics(decoded["activation"], native["activation"]),
        "command": error_metrics(decoded["command"][1:], target_command[1:]),
        "tendon": error_metrics(decoded["tendon"][1:], target_tendon[1:]),
    }
    for name, limits in (("rate", ("rate_rms", "rate_max")),
                         ("activation", ("activation_rms", "activation_max")),
                         ("command", ("command_rms", "command_max")),
                         ("tendon", ("tendon_rms", "tendon_max"))):
        metrics[name]["pass"] = (metrics[name]["rms"] <= FIDELITY[limits[0]]
                                 and metrics[name]["max_abs"] <= FIDELITY[limits[1]])
    return metrics


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e202.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E202 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e202.ledger.v1", "conditional_calls_requested": 0,
              "conditional_calls_completed": 0, "optimizer_calls": 0,
              "new_BANC_neural_ticks": 0, "new_physics_steps": 0,
              "generated_warmup_calls": 0, "generated_warmup_ticks": 0,
              "original_conditional_ticks_completed": None,
              "original_stage_inputs_consumed": None,
              "cuda_kernel_compile_s": None, "cuda_main_s": None,
              "cuda_call_wall_s": None, "cuda_phase_timing": {},
              "errors": []}
    result = {"schema": "flylab.m3.e202.local-response.result.v1",
              "validity": "incomplete_or_invalid", "candidate_adopted": False,
              "F6_exposed": False, "whole_goal_complete": False,
              "new_gait_candidate": False, "optimizer_calls": 0}
    primary = None
    freeze = None

    def attempt(label, operation):
        nonlocal primary
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - retain partial raw
            ledger["errors"].append({"stage": label,
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
            if primary is None:
                primary = error
            return None

    try:
        freeze, review = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=lambda: time.perf_counter() - began)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E202 requires 4 GiB free disk")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(item.get("num_threads") != 1 for item in pools
                   if item.get("user_api") in ("blas", "openmp")):
                raise RuntimeError("E202 CPU/BLAS/OpenMP one-thread requirement")
            paths = {key: repository_path(value, file=key != "start")
                     for key, value in freeze["paths"].items()}
            write_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
                "review_sha256": review_sha, "source_sha256": review["source_sha256"],
                "input_sha256": freeze["input_sha256"]})
            current, parameter, endpoints, start, trace, native = load_original(paths)
            p = {key: parameter[key] for key in (
                "initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain",
                "inverse_adaptation_tau")}
            weights = np.zeros((391, 7), np.float64)
            ledger["conditional_calls_requested"] = 1
            call_began = time.perf_counter()
            cuda_timing = ledger["cuda_phase_timing"]
            try:
                response = simulate_cuda(current, p, weights, timing=cuda_timing)
            finally:
                ledger["cuda_call_wall_s"] = time.perf_counter() - call_began
                ledger["cuda_kernel_compile_s"] = cuda_timing.get("compile_s")
                ledger["cuda_main_s"] = cuda_timing.get("kernel_main_s")
                ledger["original_conditional_ticks_completed"] = (
                    cuda_timing.get("conditional_ticks_completed"))
                ledger["original_stage_inputs_consumed"] = (
                    cuda_timing.get("stage_inputs_consumed"))
            ledger["conditional_calls_completed"] = 1
            np.savez_compressed(out / "local-response.npz", **response,
                weights=weights, motor_ids=parameter["motor_ids"],
                boundary_ticks=np.arange(2001, dtype=np.int64) * 20)
            decoded = decoder_from_snapshot(start, response["rate"])
            np.savez_compressed(out / "decoded.npz", **decoded)
            if time.perf_counter() - began >= RESOURCE["internal_s"]:
                raise TimeoutError("E202 internal 90 s deadline exceeded")
            # Only saved raw may be judged. Any fidelity miss leaves both raw files.
            result["metrics"] = evaluate_raw(out, response, decoded,
                endpoints, trace, native)
            if not all(item["pass"] for item in result["metrics"].values()):
                raise ValueError("E202 original E150 conditional reconstruction exceeded fidelity")
            result["validity"] = "valid_fidelity_E150_conditional_reconstruction"
    except BaseException as error:  # noqa: BLE001 - preserve raw and primary
        primary = error
        ledger["errors"].append({"stage": "run", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = time.perf_counter() - began
    result["process_wall_s"] = ledger["process_wall_s"]
    result["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    result["conservative_cumulative_s"] = (
        (freeze or {}).get("prior_cumulative_s", 0) + ledger["process_wall_s"])
    result["conditional_calls"] = ledger["conditional_calls_completed"]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    attempt("ledger.json", lambda: write_json(out / "ledger.json", ledger))
    preliminary = {**result, "validity": "incomplete_or_invalid",
                   "finalization_pending": True}
    attempt("result.json-preliminary", lambda: write_json(out / "result.json", preliminary))
    manifest = attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        primary = primary or RuntimeError("E202 output manifest/256 MiB limit failed")
        result["validity"] = "incomplete_or_invalid"
    if time.perf_counter() - began >= RESOURCE["internal_s"]:
        primary = primary or TimeoutError("E202 internal 90 s deadline exceeded")
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
