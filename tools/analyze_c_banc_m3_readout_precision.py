"""E219 saved-rate decoder and float32 sensitivity audit; no neural execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.storage import StateStore
from tools.fit_c_banc_m3_causal_neural_basis import proxy_verdict
from tools.fit_c_banc_m3_local_response import (
    DecoderAdjoint,
    direct_metrics,
    load_targets,
)
from tools.prepare_c_banc_m3_causal_neural_basis import canonical_readout
from tools.verify_c_banc_m3_local_response import LEGS

FREEZE_SHA = "d04bcf853833a62369193a53b6b92d9e353f0d8d093f9b151ce83ba8b9877297"
FREEZE_SCHEMA = "flylab.m3.e219.readout-precision.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e219.readout-precision.review.v1"
OWNED = ("tools/analyze_c_banc_m3_readout_precision.py",
         "tests/test_c_banc_readout_precision.py")
PROFILES = ("original_monotone", "original_canonical", "new_monotone",
            "new_canonical", "rate32_readout64", "readout32")
SHAPE = {"samples": 2001, "motors": 391, "muscles": 96, "joints": 42, "tendons": 6}
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
            "profile_count": 6, "solver_calls": 0, "cuda_calls": 0,
            "neural_calls": 0, "physics_steps": 0, "retries": 0,
            "internal_s": 20, "external_s": 30, "supervisor_s": 40,
            "resident_bytes": 4294967296, "output_bytes": 67108864,
            "minimum_free_bytes": 1073741824}
DTYPE_PROFILES = {name: ("float32" if name == "readout32" else "float64")
                  for name in PROFILES}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(relative: str, *, directory: bool = False) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "unsafe repository path")
    part = Path(relative)
    require(not part.is_absolute() and not any(p in (".", "..") for p in part.parts),
            "unsafe repository path")
    path = (ROOT / part).resolve()
    require(path.is_relative_to(ROOT) and
            (path.is_dir() if directory else path.is_file()), "missing frozen source")
    return path


def verify_gate(freeze_path: Path, freeze_sha: str,
                review_path: Path, review_sha: str) -> tuple[dict, dict]:
    require(freeze_sha == FREEZE_SHA and sha256(freeze_path) == FREEZE_SHA,
            "E219 freeze SHA mismatch")
    require(sha256(review_path) == review_sha, "E219 review SHA mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    require(freeze.get("schema") == FREEZE_SCHEMA and
            freeze.get("owned_files") == list(OWNED) and
            freeze.get("design") == "docs/BANC_M3_E219_READOUT_PRECISION_DESIGN_20260925.md" and
            sha256(source_path(freeze["design"])) == freeze.get("design_sha256") and
            freeze.get("profiles") == list(PROFILES) and freeze.get("shape") == SHAPE and
            freeze.get("resource_limits") == RESOURCE and
            freeze.get("candidate_adopted") is False and
            freeze.get("F6_exposed") is False and
            freeze.get("whole_goal_complete") is False,
            "E219 frozen contract mismatch")
    tool_sha, test_sha = (sha256(source_path(name)) for name in OWNED)
    require(review.get("schema") == REVIEW_SCHEMA and
            review.get("approved") is True and
            review.get("freeze_sha256") == FREEZE_SHA and
            review.get("tool_sha256") == tool_sha and
            review.get("test_sha256") == test_sha,
            "E219 source review mismatch")
    for field in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        values = freeze.get(field)
        require(isinstance(values, dict) and values, f"E219 {field} missing")
        for relative, expected in values.items():
            require(sha256(source_path(relative)) == expected,
                    f"E219 {field} mismatch: {relative}")
    paths = freeze.get("paths", {})
    require(set(paths) == {"parameters", "endpoints", "targets", "trace", "start",
                           "candidate", "canonical_original"}, "E219 paths")
    for role, relative in paths.items():
        source_path(relative, directory=role == "start")
    return freeze, {"review_sha256": review_sha, "tool_sha256": tool_sha,
                    "test_sha256": test_sha}


def _rates(value: np.ndarray) -> np.ndarray:
    rates = np.asarray(value)
    require(rates.ndim == 2 and rates.shape[1] == 391 and
            rates.dtype.kind == "f" and np.isfinite(rates).all() and
            np.min(rates) >= 0, "finite nonnegative 391-MN rates required")
    return rates


def readout32(start: dict, rates: np.ndarray) -> dict[str, np.ndarray]:
    """Float32 arithmetic at every recurrent activation and decoder step."""
    rates = _rates(rates).astype(np.float32)
    adapter = start["adapter"]
    calibration = adapter["calibration"]
    pool = np.asarray(calibration["pool"], np.float32)
    moment = np.asarray(calibration["moment"], np.float32)
    rest = np.asarray(calibration["rest"], np.float32)
    ltm = np.asarray(calibration["ltm"], np.int64)
    active = np.asarray(adapter["original"]["activation"], np.float32).copy()
    held = np.asarray(start["body"]["action"]["joint_angles"], np.float64)
    require(pool.shape == (96, 391) and moment.shape == (42, 96) and
            rest.shape == (42,) and ltm.shape == (6,) and
            active.shape == (96,) and held.shape == (42,) and
            len(set(ltm.tolist())) == 6 and np.all((ltm >= 0) & (ltm < 96)),
            "E219 float32 original anatomy/initial shape")
    names = [adapter["groups"][int(i)]["leg"] + "_tarsus" for i in ltm]
    require(set(names) == {leg + "_tarsus" for leg in LEGS},
            "E219 six original tendon names")
    order = np.asarray([names.index(leg + "_tarsus") for leg in LEGS])
    alpha = np.float32(-np.expm1(-.005 / .01))
    gain = np.float32(.02)
    tendon_gain = np.float32(.2)
    tendon_scale = np.float32(10)
    command = np.empty((len(rates), 42), np.float64)
    activation = np.empty((len(rates), 96), np.float64)
    tendon = np.empty((len(rates), 6), np.float64)
    command[0], activation[0] = held, active
    tendon[0] = tendon_gain * np.tanh(active[ltm][order] / tendon_scale)
    for t in range(1, len(rates)):
        pooled = pool @ rates[t]
        active += alpha * (pooled - active)
        q = rest + gain * (moment @ active)
        tt = tendon_gain * np.tanh(active[ltm][order] / tendon_scale)
        require(pooled.dtype == active.dtype == q.dtype == tt.dtype == np.float32,
                "E219 float32 arithmetic promoted")
        command[t], activation[t], tendon[t] = q, active, tt
    return {"command": command, "activation": activation, "tendon": tendon}


def compute_profiles(start: dict, original_rates: np.ndarray,
                     candidate_rates: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    original = _rates(original_rates)
    candidate = _rates(candidate_rates)
    require(original.shape == candidate.shape, "unaligned original/candidate rates")
    return {
        "original_monotone": DecoderAdjoint(start).forward(original),
        "original_canonical": canonical_readout(start, original),
        "new_monotone": DecoderAdjoint(start).forward(candidate),
        "new_canonical": canonical_readout(start, candidate),
        "rate32_readout64": canonical_readout(start, candidate.astype(np.float32)),
        "readout32": readout32(start, candidate),
    }


def close(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    require(actual.shape == expected.shape and np.allclose(actual, expected,
            atol=2e-10, rtol=2e-12), f"E219 {label} restoration failed")
    return float(np.max(np.abs(actual - expected)))


def assess(profiles: dict, target: dict, trace: dict, canonical: dict,
           candidate: dict) -> tuple[dict, dict]:
    trace_q = np.asarray([row["joint_targets_rad"] for row in trace["samples"]])
    restoration = {
        "original_monotone_command_max_abs": close(
            profiles["original_monotone"]["command"], trace_q,
            "original monotone command"),
    }
    for key in ("command", "activation", "tendon"):
        restoration[f"original_canonical_{key}_max_abs"] = close(
            profiles["original_canonical"][key], canonical[key],
            "original canonical " + key)
    for key, saved in (("command", "q_direct"), ("activation", "activation"),
                       ("tendon", "tendon")):
        restoration[f"new_canonical_{key}_max_abs"] = close(
            profiles["new_canonical"][key], candidate[saved],
            "new canonical " + key)
    baseline = target["baseline"]
    reports = {}
    for name in PROFILES:
        metrics = direct_metrics(profiles[name], target)
        verdict = proxy_verdict(metrics, baseline)
        margins = {"startup_q_rms": .018 - metrics["startup_q_rms"],
                   "startup_q_max": .04 - metrics["startup_q_max"],
                   "startup_tendon_rms": .009 - metrics["startup_tendon_rms"],
                   "startup_tendon_max": .025 - metrics["startup_tendon_max"]}
        reports[name] = {"metrics": metrics, "strict_proxy": verdict,
                         "startup_margins": margins,
                         "calculation_dtype": DTYPE_PROFILES[name],
                         "stored_dtype": "float64"}
    require(reports["new_canonical"]["strict_proxy"]["passed"],
            "E218 candidate strict proxy no longer passes")
    return reports, restoration


def save_json(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def output_manifest(out: Path) -> dict:
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file() and
             p.name != "output-manifest.json"}
    return {"schema": "flylab.m3.e219.output-manifest.v1", "files": files,
            "total_bytes": sum(v["bytes"] for v in files.values())}


def run(freeze_path: Path, freeze_sha: str, review_path: Path,
        review_sha: str, output: Path) -> dict:
    start_time = time.monotonic()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    phase = "gate"
    loaded_roles: list[str] = []
    profiles_done: list[str] = []
    calculation_requested = False
    review_hashes = None
    try:
        require(shutil.disk_usage(out).free >= RESOURCE["minimum_free_bytes"],
                "E219 free disk")
        freeze, review_hashes = verify_gate(freeze_path, freeze_sha,
                                            review_path, review_sha)
        paths = {name: source_path(relative, directory=name == "start")
                 for name, relative in freeze["paths"].items()}
        phase = "load"
        start = StateStore.load(paths["start"]); loaded_roles.append("start")
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        loaded_roles.append("trace")
        target = load_targets(paths["targets"], trace); loaded_roles.append("targets")
        with np.load(paths["endpoints"], allow_pickle=False) as z:
            original_rates = z["motor_rate"].copy()
            original_ids = z["motor_ids"].copy()
        loaded_roles.append("endpoints")
        with np.load(paths["candidate"], allow_pickle=False) as z:
            candidate = {key: z[key].copy() for key in
                         ("rate", "q_direct", "activation", "tendon")}
        loaded_roles.append("candidate")
        with np.load(paths["canonical_original"], allow_pickle=False) as z:
            canonical = {key: z[key].copy() for key in
                         ("command", "activation", "tendon")}
        loaded_roles.append("canonical_original")
        with np.load(paths["parameters"], allow_pickle=False) as z:
            parameter_ids = z["motor_ids"].copy()
        loaded_roles.append("parameters")
        require(original_rates.shape == candidate["rate"].shape == (2001, 391)
                and np.array_equal(original_ids, parameter_ids)
                and np.array_equal(original_ids,
                    np.asarray(start["adapter"]["parameters"]["motor_ids"])),
                "E219 exact 391 MN identity/dimension")
        require(all(np.isfinite(v).all() for v in candidate.values()) and
                np.isfinite(original_rates).all(), "E219 nonfinite rates/reference")
        require(time.monotonic() - start_time < RESOURCE["internal_s"],
                "E219 internal deadline before calculation")
        phase = "readout"
        calculation_requested = True
        profiles = compute_profiles(start, original_rates, candidate["rate"])
        for name in PROFILES:
            np.savez_compressed(out / f"{name}.npz", **profiles[name])
            profiles_done.append(name)
            require(time.monotonic() - start_time < RESOURCE["internal_s"],
                    "E219 internal readout deadline")
        phase = "assessment"
        reports, restoration = assess(profiles, target, trace, canonical, candidate)
        save_json(out / "metrics.json", {"profiles": reports,
                                          "restoration": restoration,
                                          "baseline": target["baseline"]})
        phase = "publication"
        elapsed = time.monotonic() - start_time
        manifest_before = output_manifest(out)
        require(elapsed < RESOURCE["internal_s"] and
                manifest_before["total_bytes"] < RESOURCE["output_bytes"],
                "E219 deadline/output capacity before result")
        result = {"schema": "flylab.m3.e219.readout-precision.result.v1",
                  "run_status": "completed", "validity": "valid_saved_readout_diagnostic",
                  "profiles": list(PROFILES), "profile_metrics_file": "metrics.json",
                  "raw_files": [f"{name}.npz" for name in PROFILES],
                  "restoration": restoration,
                  "new_canonical_strict_proxy_pass": True,
                  "candidate_adopted": False, "F6_exposed": False,
                  "whole_goal_complete": False,
                  "original_calculation_count": 1,
                  "calculation_requested": True, "calculation_completed": True,
                  "loaded_input_roles": loaded_roles, "retries": 0,
                  "neural_calls": 0, "physics_steps": 0, "solver_calls": 0,
                  "cuda_calls": 0, "elapsed_internal_s": elapsed,
                  "provenance": {"freeze_sha256": freeze_sha,
                     **review_hashes, "input_sha256": freeze["input_sha256"],
                     "reused_source_sha256": freeze["reused_source_sha256"],
                     "preservation_sha256": freeze["preservation_sha256"]},
                  "output_manifest_file": "output-manifest.json"}
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        result_bytes = payload.encode("utf-8")
        manifest_before["files"]["result.json"] = {
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
            "bytes": len(result_bytes)}
        manifest_before["total_bytes"] += len(result_bytes)
        manifest_bytes = (json.dumps(manifest_before, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8")
        require(manifest_before["total_bytes"] + len(manifest_bytes) <
                RESOURCE["output_bytes"] and
                time.monotonic() - start_time < RESOURCE["internal_s"],
                "E219 result capacity/deadline")
        (out / "output-manifest.json").write_bytes(manifest_bytes)
        require(time.monotonic() - start_time < RESOURCE["internal_s"],
                "E219 final publication deadline")
        (out / "result.json").write_bytes(result_bytes)
        return result
    except Exception as exc:  # noqa: BLE001 - preserve any partial raw on failure
        failure = {"schema": "flylab.m3.e219.readout-precision.failure.v1",
                   "run_status": "failed", "validity": "invalid",
                   "phase": phase, "error_type": type(exc).__name__,
                   "error": str(exc), "loaded_input_roles": loaded_roles,
                   "completed_raw_profiles": profiles_done,
                   "calculation_requested": calculation_requested,
                   "calculation_completed": False,
                   "original_calculation_count": int(calculation_requested),
                   "retries": 0, "elapsed_internal_s": time.monotonic() - start_time,
                   "provenance": {"freeze_sha256": freeze_sha,
                                  "review_sha256": review_sha,
                                  **(review_hashes or {})}}
        save_json(out / "failure.json", failure)
        save_json(out / "output-manifest.json", output_manifest(out))
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.out)
    print(json.dumps({"run_status": result["run_status"], "out": str(args.out)}))
    if result["run_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
