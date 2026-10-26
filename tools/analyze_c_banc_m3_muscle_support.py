"""E199 saved E150/E198 muscle support audit; no model or simulation imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"all": slice(0, 2000), "startup": slice(0, 99), "gait": slice(99, 2000)}
INPUT_NAMES = {
    "native": "native-recruitment-state.npz",
    "adapter": "adapter-provenance.npz",
    "muscles": "muscle-provenance.json",
    "operating_range": "per-mn.npz",
    "operating_result": "result.json",
    "operating_review": "parent-e198-final-review.json",
}
MAX_SECONDS = 15.0
MAX_BYTES = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(relative: str) -> Path:
    require(isinstance(relative, str), "source path is not text")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "source path escapes repository")
    return path


def array(record, name, shape, dtype=None):
    require(name in record, f"missing {name}")
    value = np.asarray(record[name])
    require(value.shape == shape, f"{name} shape {value.shape} != {shape}")
    if dtype is not None:
        require(value.dtype == np.dtype(dtype), f"{name} dtype")
    require(value.dtype.kind in "biuf" and np.all(np.isfinite(value)), f"{name} nonfinite")
    return value


def analyze_arrays(operating: dict, adapter: dict, muscles: dict,
                   native: dict, operating_result: dict, operating_review: dict) -> dict:
    """Generated-array-compatible fixed 2000-boundary descriptive analysis."""
    require("motor_ids" in operating, "missing E198 motor IDs")
    n = np.asarray(operating["motor_ids"]).size
    require("pool" in adapter, "missing pool")
    m = np.asarray(adapter["pool"]).shape[0]
    ids = array(operating, "motor_ids", (n,), "int64")
    require(n > 0 and m > 0 and len(np.unique(ids)) == n, "motor roster IDs")
    require(np.array_equal(ids, array(adapter, "motor_ids", (n,), "int64")),
            "E198/adapter motor order differs")
    membership = array(adapter, "membership", (n,), "int64")
    require(np.all((0 <= membership) & (membership < m)), "membership range")
    pool = array(adapter, "pool", (m, n)).astype(np.float64)
    require(np.all(pool >= 0), "negative pool")
    require(np.array_equal(np.count_nonzero(pool > 0, axis=0), np.ones(n, int)) and
            np.array_equal(np.argmax(pool, axis=0), membership) and
            np.all(np.bincount(membership, minlength=m) > 0), "pool ownership")
    moment = array(adapter, "moment", (42, m)).astype(np.float64)
    ltm = array(adapter, "ltm", (6,), "int64")
    require(len(np.unique(ltm)) == 6 and np.all((0 <= ltm) & (ltm < m)), "LTM indices")
    require(isinstance(muscles.get("motor_roster"), list) and
            len(muscles["motor_roster"]) == n and
            all(isinstance(x, str) and x for x in muscles["motor_roster"]) and
            len(set(muscles["motor_roster"])) == n, "root roster")
    groups = muscles.get("muscles")
    require(isinstance(groups, list) and len(groups) == m, "muscle provenance count")
    for muscle, group in enumerate(groups):
        require(isinstance(group, dict) and isinstance(group.get("leg"), str) and
                bool(group["leg"]) and isinstance(group.get("muscle"), str) and
                bool(group["muscle"]) and
                group.get("columns") == np.flatnonzero(membership == muscle).tolist(),
                f"muscle {muscle} ownership")

    require("window_names" in operating and
            np.array_equal(np.asarray(operating["window_names"]),
                           np.array(list(WINDOWS))), "E198 window order")
    r = array(operating, "r_quantiles", (3, n, 5)).astype(np.float64)
    phi = array(operating, "phi_quantiles", (3, n, 5)).astype(np.float64)
    active = array(operating, "active_fraction", (3, n)).astype(np.float64)
    require(np.all(r >= 0) and np.all(phi >= 0) and
            np.all(np.diff(r, axis=2) >= 0) and
            np.all(np.diff(phi, axis=2) >= 0) and
            np.all((active >= 0) & (active <= 1)), "E198 summary values")
    require(operating_result.get("run_status") == "completed" and
            operating_result.get("validity") == "valid_saved_operating_range" and
            operating_result.get("motor_count") == n and
            operating_result.get("quantile_method") == "NumPy linear", "E198 result")
    require(operating_review.get("validity") == "valid_saved_diagnostic" and
            operating_review.get("independent_checks") == 43 and
            operating_review.get("original_attempts") == 1 and
            operating_review.get("retries") == 0, "E198 final review")

    require(np.array_equal(array(native, "control_tick", (2001,), "int64"),
                           np.arange(2001)), "control ticks")
    require(np.array_equal(array(native, "network_tick", (2001,), "int64"),
                           20 * np.arange(2001)), "network ticks")
    require(np.max(np.abs(array(native, "physics_time_s", (2001,)) -
                          .005 * np.arange(2001))) <= 1e-9, "native time grid")
    for key in ("complete", "activation_available", "motor_rate_available"):
        require(np.all(array(native, key, (2001,), "bool")), f"native {key}")
    activation = array(native, "activation", (2001, m)).astype(np.float64)
    require(np.all(activation >= 0), "negative activation")
    require(array(native, "motor_rate", (2001, n)).min() >= 0,
            "negative native motor rate")

    result = {"validity": "valid_saved_muscle_support", "motor_count": int(n),
              "muscle_count": int(m), "windows": {},
              "interpretation_limit": "Saved activation and position-command support; no force or gait causality"}
    for window_index, (name, part) in enumerate(WINDOWS.items()):
        zero_rate = r[window_index, :, -1] == 0
        zero_phi = active[window_index] == 0
        values = activation[part.start + 1:part.stop + 1]
        require(values.shape == (part.stop - part.start, m), "window activation rows")
        rows = []
        for muscle, group in enumerate(groups):
            columns = np.flatnonzero(membership == muscle)
            q = np.quantile(values[:, muscle], (0, .05, .5, .95, 1), method="linear")
            zeros = columns[zero_rate[columns]]
            phi_zeros = columns[zero_phi[columns]]
            rows.append({"muscle_index": int(muscle), "leg": group["leg"],
                         "muscle": group["muscle"], "member_count": len(columns),
                         "member_motor_ids": ids[columns].astype(int).tolist(),
                         "member_root_ids": [muscles["motor_roster"][i] for i in columns],
                         "zero_rate_count": len(zeros),
                         "zero_rate_motor_ids": ids[zeros].astype(int).tolist(),
                         "zero_rate_root_ids": [muscles["motor_roster"][i] for i in zeros],
                         "zero_phi_count": len(phi_zeros),
                         "zero_phi_motor_ids": ids[phi_zeros].astype(int).tolist(),
                         "zero_phi_root_ids": [muscles["motor_roster"][i] for i in phi_zeros],
                         "activation_quantiles": [float(x) for x in q],
                         "activation_mean": float(np.mean(values[:, muscle])),
                         "activation_exact_zero_samples": int(np.count_nonzero(values[:, muscle] == 0)),
                         "activation_p95_minus_p05": float(q[3] - q[1]),
                         "initial_activation": float(activation[0, muscle]),
                         "initial_activation_zero": bool(activation[0, muscle] == 0),
                         "joint_rows": np.flatnonzero(moment[:, muscle]).astype(int).tolist(),
                         "is_ltm": bool(muscle in ltm)})
        all_zero = [row for row in rows if row["zero_rate_count"] == row["member_count"]]
        partial = [row for row in rows if 0 < row["zero_rate_count"] < row["member_count"]]
        no_zero = [row for row in rows if row["zero_rate_count"] == 0]
        inactive = [row for row in rows if row["activation_quantiles"][-1] == 0]
        contradictions = [row for row in all_zero if row["activation_quantiles"][-1] > 0]
        zero_initial_contradictions = [
            row for row in contradictions
            if name in ("all", "startup") and row["initial_activation_zero"]
        ]
        residual_possible = [
            row for row in contradictions
            if name == "gait" or not row["initial_activation_zero"]
        ]
        result["windows"][name] = {
            "ticks": [part.start + 1, part.stop], "muscles": rows,
            "all_member_rate_zero_count": len(all_zero),
            "some_member_rate_zero_count": len(partial),
            "no_member_rate_zero_count": len(no_zero),
            "activation_always_zero": [
                {"muscle_index": row["muscle_index"], "leg": row["leg"],
                 "muscle": row["muscle"]} for row in inactive],
            "all_zero_rate_with_positive_activation": [
                {"muscle_index": row["muscle_index"], "leg": row["leg"],
                 "muscle": row["muscle"]} for row in contradictions],
            "zero_initial_contradictions": [
                {"muscle_index": row["muscle_index"], "leg": row["leg"],
                 "muscle": row["muscle"]} for row in zero_initial_contradictions],
            "filter_residual_possible": bool(residual_possible),
        }
    if result["windows"]["all"]["zero_initial_contradictions"]:
        result["hypothesis_verdict"] = "inconclusive_zero_initial_state_contradiction"
    elif result["windows"]["all"]["all_zero_rate_with_positive_activation"]:
        result["hypothesis_verdict"] = "inconclusive_initial_filter_state"
    elif result["windows"]["all"]["activation_always_zero"]:
        result["hypothesis_verdict"] = "some_muscles_without_saved_activation"
    else:
        result["hypothesis_verdict"] = "all_muscles_have_saved_activation"
    return result


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
        require(freeze.get("schema") == "flylab.m3.e199.muscle-support.freeze.v1" and
                review.get("schema") == "flylab.m3.e199.muscle-support.review.v1" and
                review.get("approved") is True and
                review.get("freeze_sha256") == freeze_sha and
                review.get("tool_sha256") == sha256(Path(__file__)) and
                review.get("test_sha256") == sha256(
                    ROOT / "tests/test_c_banc_muscle_support.py"), "source review gate")
        require(freeze.get("owned_files") == ["tools/analyze_c_banc_m3_muscle_support.py",
                                              "tests/test_c_banc_muscle_support.py"],
                "owned file contract")
        require(freeze.get("shape") == {"samples": 2000, "motors": 391,
                                        "muscles": 96, "joints": 42,
                                        "dt_s": .005, "neural_steps_per_control": 20,
                                        "time_atol": 1e-9} and
                freeze.get("windows") == {"all": [1, 2000], "startup": [1, 99],
                                           "gait": [100, 2000]} and
                freeze.get("quantile_method") == "linear", "freeze analysis contract")
        limits = freeze["resource_limits"]
        require(limits.get("cpu_threads") == 1 and limits.get("gpu_count") == 0 and
                limits.get("internal_s") == MAX_SECONDS and
                limits.get("output_bytes") == MAX_BYTES and
                limits.get("original_attempts") == 1 and limits.get("retries") == 0,
                "freeze resource limits")
        design = freeze["design"]
        require(design == "docs/BANC_M3_E199_MUSCLE_SUPPORT_DESIGN_20260924.md" and
                sha256(source_path(design)) == freeze["design_sha256"], "design SHA")
        for relative, digest in freeze["preservation_sha256"].items():
            require(sha256(source_path(relative)) == digest, "preservation SHA")
        roles = freeze["input_roles"]
        require(set(roles) == set(INPUT_NAMES) and
                set(roles.values()) == set(freeze["input_sha256"]), "input roles")
        for role, filename in INPUT_NAMES.items():
            path = source_path(roles[role])
            require(path.name == filename, f"{role} filename")
            require(sha256(path) == freeze["input_sha256"][roles[role]],
                    f"{role} input SHA")
        manifest = source_path(freeze["source_manifest"])
        require(manifest.name == "output-manifest.json" and
                sha256(manifest) == freeze["source_manifest_sha256"],
                "E150 manifest SHA")
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        phase = "load"
        records = {}
        for role in INPUT_NAMES:
            path = source_path(roles[role])
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    records[role] = {key: archive[key] for key in archive.files}
            else:
                records[role] = json.loads(path.read_text(encoding="utf-8"))
            loaded.append(role)
        phase = "analysis"
        require(np.asarray(records["operating_range"]["motor_ids"]).shape == (391,) and
                np.asarray(records["adapter"]["pool"]).shape == (96, 391) and
                np.asarray(records["native"]["activation"]).shape == (2001, 96),
                "frozen source dimensions")
        require(records["operating_result"]["source_inputs"]["native"]["sha256"] ==
                freeze["input_sha256"][roles["native"]], "E198/E150 native source SHA")
        require(records["operating_review"].get("result_sha256") ==
                freeze["input_sha256"][roles["operating_result"]] and
                records["operating_review"].get("raw_sha256") ==
                freeze["input_sha256"][roles["operating_range"]], "E198 review SHA")
        calculation_requested = True
        result = analyze_arrays(records["operating_range"], records["adapter"],
                                records["muscles"], records["native"],
                                records["operating_result"], records["operating_review"])
        calculation_completed = True
        phase = "publication"
        result.update({"experiment": "M3-E199", "run_status": "completed",
                       "freeze_sha256": freeze_sha, "review_sha256": review_sha,
                       "tool_sha256": review["tool_sha256"],
                       "test_sha256": review["test_sha256"],
                       "source_manifest": {"path": freeze["source_manifest"],
                                           "sha256": freeze["source_manifest_sha256"]},
                       "source_inputs": {role: {"path": relative,
                                                "sha256": freeze["input_sha256"][relative]}
                                         for role, relative in roles.items()},
                       "loaded_input_roles": loaded,
                       "calculation_requested": calculation_requested,
                       "calculation_completed": calculation_completed,
                       "original_calculation_count": 1, "retries": 0,
                       "neural_advance_count": 0, "physics_step_count": 0,
                       "fit_count": 0, "F6_exposed": False,
                       "candidate_adopted": False,
                       "prior_cumulative_s": freeze["prior_cumulative_s"],
                       "reserved_supervisor_s": freeze["resource_limits"]["supervisor_s"],
                       "wall_seconds": float(time.perf_counter() - started),
                       "output_manifest_file": "output-manifest.json"})
        payload = (json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n").encode("utf-8")
        manifest_value = {"schema": "flylab.m3.e199.muscle-support.output-manifest.v1",
                          "files": {"result.json": {"sha256": hashlib.sha256(payload).hexdigest(),
                                                    "bytes": len(payload)}}}
        manifest_payload = (json.dumps(manifest_value, indent=2) + "\n").encode("utf-8")
        require(len(payload) + len(manifest_payload) <= MAX_BYTES, "output byte limit")
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        pending_result = output / "result.json.pending"
        pending_manifest = output / "output-manifest.json.pending"
        pending_result.write_bytes(payload)
        pending_manifest.write_bytes(manifest_payload)
        require(time.perf_counter() - started <= MAX_SECONDS, "internal time limit")
        pending_manifest.rename(output / "output-manifest.json")
        pending_result.rename(output / "result.json")
        return result
    except Exception as exc:
        error = {"experiment": "M3-E199", "run_status": "failed",
                 "validity": "inconclusive", "phase": phase,
                 "error_type": type(exc).__name__, "error": str(exc),
                 "partial": any(output.iterdir()),
                 "loaded_input_roles": loaded,
                 "calculation_requested": calculation_requested,
                 "calculation_completed": calculation_completed,
                 "original_calculation_count": int(calculation_requested),
                 "retries": 0, "neural_advance_count": 0,
                 "physics_step_count": 0, "fit_count": 0,
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
