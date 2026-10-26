"""E214: one global physical-beta fit of the frozen E213 multiscale basis.

Six E210 physical problems are prepared and stored without local solves. Their
first three cones are joined by E211 with one global startup cone. No neural or
body model is constructed here.
"""

from __future__ import annotations

import argparse
import gc
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

from tools.fit_c_banc_m3_causal_neural_basis import (
    build_leg_problem,
    leg_anatomy,
    physical_joint_matrix,
    proxy_verdict,
)
from tools.fit_c_banc_m3_causal_neural_basis import (
    save_problem as save_leg_problem,
)
from tools.fit_c_banc_m3_global_startup import (
    SolverAllowanceUnavailable,
    assemble_global,
    assess_solution,
    ensure_solve_budget,
    load_local_problem,
    physical_limit_diagnostics,
    soft_limit_overrun,
    solve_once,
)
from tools.fit_c_banc_m3_global_startup import (
    save_problem as save_global_problem,
)
from tools.fit_c_banc_m3_local_response import (
    DecoderAdjoint,
    direct_metrics,
    load_targets,
)
from tools.prepare_c_banc_m3_causal_neural_basis import canonical_readout
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "491d2339680da1fb26e916004782ffc34b09c0012244ffc622d3b7d64ec4141c"
FREEZE_SCHEMA = "flylab.m3.e214.multiscale-response-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e214.multiscale-response-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_multiscale_response.py",
         "tests/test_c_banc_multiscale_response_fit.py")
LEG_WIDTHS = (976, 864, 896, 992, 880, 880)
PAD = 1 - 1e-6
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 1, "retries": 0, "internal_s": 240,
    "solver_s": 180, "external_s": 270, "supervisor_s": 300,
    "resident_bytes": 8589934592, "output_bytes": 536870912,
    "minimum_free_bytes": 4294967296}


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E214 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E214 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E214 path")
    if file and not path.is_file():
        raise ValueError("E214 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E214 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E214 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    shape = {"variables": 5488, "motors": 391, "free_motors": 343,
        "fixed_motors": 48, "basis_count": 16, "samples": 2001,
        "startup_rows": 99, "global_startup_values": 4158,
        "leg_variables": list(LEG_WIDTHS), "cones": 19}
    solver = {"name": "clarabel", "max_threads": 1, "max_iter": 150,
        "time_limit_s": 180.0, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8, "coordinate": "physical beta in six leg blocks",
        "objective": "sum unchanged E210 normalized leg objectives",
        "beta_min": 0.0, "beta_max": 1.0, "PAD": PAD, "raw_clip": False,
        "weak_column_pruning": False, "converged_status": "Solved"}
    proxy = {"global_command_ratio": .8, "leg_command_ratio": 1.,
        "leg_velocity_ratio": 1., "global_velocity_ratio": 1.,
        "startup_q_rms": .018, "startup_q_max": .04,
        "startup_tendon_rms": .009, "startup_tendon_max": .025,
        "original_command_rms": .056686385349786386,
        "training_extra": "global startup norm with unchanged PAD; no per-leg startup RMS gate"}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E214_MULTISCALE_FIT_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != shape
            or freeze.get("solver") != solver
            or freeze.get("proxy") != proxy
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("clock") != {"neural_dt_s": .00025,
                                      "control_dt_s": .005, "ticks_per_control": 20}
            or freeze.get("tolerance") != {"affine_q_atol": 2e-12,
                "affine_q_rtol": 2e-12, "gram_relative_frobenius_max": 1e-10,
                "gram_min_eigen_relative": -1e-10}
            or freeze.get("local_solver_calls") != 0
            or freeze.get("generated_solver", {}).get("actual_calls") != 0
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E214 scientific or resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E214 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E214 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E214 source review does not authorize these files")
    for label, count in (("input_sha256", 138),
                         ("reused_source_sha256", 26),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E214 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E214 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E214 hash gate exceeded internal deadline")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("E214 paths missing")
    resolved = {key: repository_path(value, file=key != "start")
                for key, value in paths.items()}
    return freeze, review, resolved


def prepare_local_problems(out, start, D, q0, target, motors_by_leg, affine_sha,
                           *, elapsed=None, builder=build_leg_problem,
                           saver=save_leg_problem):
    """Build, save, and release one dense physical leg problem at a time."""
    out = Path(out)
    records = []
    for index, leg in enumerate(LEGS):
        if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E214 local preparation exceeded internal deadline")
        path = out / ("leg-" + leg)
        path.mkdir()
        begun = time.perf_counter()
        problem = builder(start, D, q0, target, motors_by_leg, index)
        try:
            if problem["leg"] != leg or not np.array_equal(
                    problem["motor_columns"], motors_by_leg[index]):
                raise ValueError("E214 local builder ownership changed")
            saver(path, problem, affine_sha)
            records.append({"leg": leg, "motor_columns":
                np.asarray(problem["motor_columns"], np.int64).tolist(),
                "variables": int(problem["P"].shape[0]),
                "rows": int(problem["A"].shape[0]),
                "P_nnz": int(problem["P"].nnz), "A_nnz": int(problem["A"].nnz),
                "build_and_save_wall_s": time.perf_counter() - begun,
                "local_solver_calls": 0})
        finally:
            del problem
            gc.collect()
        save_json(out / "local-preparation.json", records)
    return records


def verify_global_structure(problem, locals_, motors_by_leg, free):
    """Check frozen ownership and cones after relaxed sparse-rank assembly."""
    free = np.asarray(free, np.int64)
    if len(locals_) != 6 or [v["leg"] for v in locals_] != list(LEGS):
        raise ValueError("E214 six-leg order changed")
    if any(not np.array_equal(v["motor_columns"], motors_by_leg[i])
           for i, v in enumerate(locals_)):
        raise ValueError("E214 per-leg motor ownership changed")
    columns = np.concatenate([v["motor_columns"] for v in locals_])
    dims = np.asarray(problem["cone_dimensions"], np.int64)
    kinds = list(problem["cone_types"])
    expected_dims = np.r_[np.concatenate([v["dims"][:3] for v in locals_]), 4159]
    expected_kinds = [kind for v in locals_ for kind in v["kinds"][:3]]
    expected_kinds.append("SecondOrderConeT")
    if (free.shape != (343,) or len(np.unique(free)) != 343
            or not np.array_equal(np.sort(columns), np.sort(free))
            or not np.array_equal(problem["motor_columns"], columns)
            or sum(v["P"].shape[0] for v in locals_) != 5488
            or problem["P"].shape != (5488, 5488)
            or problem["A"].shape != (int(dims.sum()), 5488)
            or problem["startup_matrix"].shape != (4158, 5488)
            or problem["startup_offset"].shape != (4158,)
            or problem["linear"].shape != (5488,)
            or problem["b"].shape != (int(dims.sum()),)
            or len(problem["cones"]) != 19
            or dims.shape != (19,) or not np.array_equal(dims, expected_dims)
            or kinds != expected_kinds
            or any(v["M"].shape != (693, LEG_WIDTHS[i])
                   for i, v in enumerate(locals_))
            or any(v["kinds"] != ["NonnegativeConeT"]
                   + ["SecondOrderConeT"] * 3 for v in locals_)
            or any(v["P"].shape[0] != LEG_WIDTHS[i]
                   for i, v in enumerate(locals_))):
        raise ValueError("E214 global 5488-variable/19-cone/ownership contract changed")
    if not np.array_equal(problem["startup_offset"],
                          np.concatenate([v["startup_offset"] for v in locals_])):
        raise ValueError("E214 global startup ordering changed")
    return {"variables": 5488, "cones": 19, "startup_values": 4158,
        "A_rows": int(problem["A"].shape[0]), "A_nnz": int(problem["A"].nnz),
        "P_nnz": int(problem["P"].nnz), "free_motors": 343}


def load_frozen_arrays(paths, start):
    with (np.load(paths["affine"], allow_pickle=False) as affine,
          np.load(paths["basis"], allow_pickle=False) as basis,
          np.load(paths["basis_parameters"], allow_pickle=False) as params,
          np.load(paths["endpoints"], allow_pickle=False) as native,
          np.load(paths["mask"], allow_pickle=False) as mask):
        arrays = {"D": affine["D"].copy(),
            "q0": affine["q_intercept"].copy(),
            "H": basis["H"].copy(), "B": basis["B"].copy(),
            "cap": params["cap"].copy(),
            "free": mask["free_columns"].astype(np.int64),
            "fixed": mask["fixed_columns"].astype(np.int64),
            "original_rates": native["motor_rate"].copy()}
        if (arrays["D"].shape != (2001, 391, 16)
                or arrays["q0"].shape != (2001, 42)
                or arrays["H"].shape != (2001, 391)
                or arrays["B"].shape != arrays["D"].shape
                or arrays["cap"].shape != (391,)
                or arrays["free"].shape != (343,)
                or arrays["fixed"].shape != (48,)
                or arrays["original_rates"].shape != arrays["H"].shape
                or not np.array_equal(np.sort(np.r_[arrays["free"], arrays["fixed"]]),
                                      np.arange(391))
                or not np.array_equal(mask["motor_ids"], params["motor_ids"])
                or not np.array_equal(mask["membership"],
                    start["adapter"]["parameters"]["membership"])
                or np.any(arrays["D"][:, arrays["fixed"]] != 0)
                or not all(np.isfinite(value).all() for value in arrays.values())):
            raise ValueError("E214 E213 basis/mask/anatomy changed")
    return arrays


def publish_candidate(out, start, arrays, target, trace, motors_by_leg,
                      membership, moment, locals_, problem, raw, info, report):
    """Save all four readouts before applying scientific acceptance checks."""
    beta = np.zeros((391, 16), np.float64)
    offset = 0
    for motors in motors_by_leg:
        width = len(motors) * 16
        beta[motors] = raw["solver_x"][offset:offset + width].reshape(-1, 16)
        offset += width
    rate = arrays["H"] + arrays["cap"][None, :] * np.sum(
        arrays["B"] * beta[None], axis=2)
    rate[:, arrays["fixed"]] = arrays["original_rates"][:, arrays["fixed"]]
    direct = canonical_readout(start, rate)
    q_affine = arrays["q0"].copy()
    for i, motors in enumerate(motors_by_leg):
        joint = physical_joint_matrix(arrays["D"], membership, moment, motors, i)
        q_affine[:, 7*i:7*i+7] += joint @ beta[motors].reshape(-1)
        del joint
    np.savez(Path(out) / "candidate.npz", beta=beta, rate=rate,
        q_affine=q_affine, q_direct=direct["command"],
        tendon=direct["tendon"], activation=direct["activation"])
    original = canonical_readout(start, arrays["original_rates"])
    decoder = DecoderAdjoint(start)
    old_mono = decoder.forward(arrays["original_rates"])
    new_mono = decoder.forward(rate)
    np.savez(Path(out) / "readout-2x2.npz",
        original_monotone_q=old_mono["command"],
        original_monotone_tendon=old_mono["tendon"],
        new_monotone_q=new_mono["command"],
        new_monotone_tendon=new_mono["tendon"],
        original_canonical_q=original["command"],
        original_canonical_tendon=original["tendon"],
        new_canonical_q=direct["command"],
        new_canonical_tendon=direct["tendon"])
    ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
    affine_exact = bool(np.allclose(q_affine, direct["command"],
        atol=2e-12, rtol=2e-12))
    ltm_exact = bool(np.array_equal(original["activation"][:, ltm],
        direct["activation"][:, ltm]) and np.array_equal(
        original["tendon"], direct["tendon"]))
    metrics = direct_metrics(direct, target)
    indices = np.asarray(target["indices"], np.int64)
    pair_start = np.asarray(target["pair_start"], np.int64)
    pair_end = np.asarray(target["pair_end"], np.int64)
    physical_legs = []
    for i, local in enumerate(locals_):
        j = slice(7*i, 7*(i+1))
        qres = q_affine[indices, j] - target["target"][:, j]
        vres = ((q_affine[pair_end, j] - q_affine[pair_start, j]) / .005
                - target["velocity"][:, j])
        sres = q_affine[1:100, j] - target["startup_command"][:, j]
        physical_legs.append({"leg": local["leg"],
            "q_rms": float(np.sqrt(np.mean(qres * qres))),
            "velocity_rms": float(np.sqrt(np.mean(vres * vres))),
            "startup_q_rms": float(np.sqrt(np.mean(sres * sres))),
            "startup_q_max": float(np.max(np.abs(sres)))})
    physical = physical_limit_diagnostics(physical_legs, locals_)
    physical["startup_raw_affine_match"] = bool(all(np.isclose(
        leg["startup_q_rms"], report["direct"]["legs"][i]["startup_q_rms"],
        atol=2e-12, rtol=2e-12) for i, leg in enumerate(physical_legs)))
    report["direct"]["physical"] = physical
    report["direct_training_pass"] = bool(report["direct_training_pass"]
        and all(physical["leg_q_original_limit_pass"])
        and all(physical["leg_velocity_original_limit_pass"])
        and physical["startup_raw_affine_match"])
    save_json(Path(out) / "direct-metrics.json", report)
    verdict = proxy_verdict(metrics, target["baseline"])
    verdict.update(global_cone_numerical_pass=report["numerical_pass"],
        global_training_pass=report["direct_training_pass"],
        affine_exact=affine_exact, ltm_exact=ltm_exact)
    verdict["passed"] = bool(verdict["passed"] and report["numerical_pass"]
        and report["direct_training_pass"] and affine_exact and ltm_exact)
    native_q = np.asarray([row["joint_targets_rad"] for row in trace["samples"]])
    return {"solver_info": info, "solver_assessment": report,
        "metrics": metrics, "baseline": target["baseline"],
        "proxy_verdict": verdict,
        "original_monotone_native_q_max_abs": float(np.max(np.abs(
            old_mono["command"] - native_q)))}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e214.output-manifest.v1", "files": files,
                "total_bytes": sum(item["bytes"] for item in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def _save_attempt(errors, stage, operation):
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
        raise FileExistsError("One-shot E214 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = None
    primary = None
    errors = []
    ledger = {"schema": "flylab.m3.e214.ledger.v1", "solve_requested": False,
        "constructor_entered": False, "constructor_completed": False,
        "solve_called": False, "solver_returned": False, "solver_calls": 0,
        "local_solver_calls": 0, "local_problems_saved": 0,
        "constructor_wall_s": 0., "solver_wall_s": 0.,
        "full_banc_calls": 0, "cuda_calls": 0, "physics_steps": 0, "retries": 0}
    result = {"schema": "flylab.m3.e214.multiscale-response-fit.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "conditional_proxy_only": True}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E214 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "affine_sha256": freeze["input_sha256"][freeze["paths"]["affine"]]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        from flylab.c.storage import StateStore

        with threadpool_limits(limits=1):
            start = StateStore.load(paths["start"])
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target = load_targets(paths["targets"], trace)
            arrays = load_frozen_arrays(paths, start)
            membership, moment, motors_by_leg = leg_anatomy(start, arrays["free"])
            affine_sha = freeze["input_sha256"][freeze["paths"]["affine"]]
            records = prepare_local_problems(out, start, arrays["D"], arrays["q0"],
                target, motors_by_leg, affine_sha, elapsed=elapsed)
            ledger["local_problems_saved"] = len(records)
            save_json(out / "ledger.json", ledger)
            locals_ = [load_local_problem(out / ("leg-" + leg) / "pre-solve.npz",
                                          leg, affine_sha) for leg in LEGS]
            problem = assemble_global(locals_, strict=False)
            result["structure"] = verify_global_structure(
                problem, locals_, motors_by_leg, arrays["free"])
            save_global_problem(out, problem, affine_sha)
            # The complete E211 allowance is checked after expensive preparation
            # and once more by solve_once after its constructor.
            ensure_solve_budget(elapsed)
            raw, info = solve_once(problem, out, ledger, elapsed=elapsed)
            result["solver_info"] = info
            report = assess_solution(problem, raw, info, locals_)
            save_json(out / "direct-metrics.json", report)
            result["solver_assessment"] = report
            if report["eligible_beta"]:
                result.update(publish_candidate(out, start, arrays, target, trace,
                    motors_by_leg, membership, moment, locals_, problem, raw,
                    info, report))
                result["validity"] = "valid_conditional_fit_rejected_or_supported"
            else:
                result["validity"] = "valid_bounded_unresolved_no_complete_fit"
    except SolverAllowanceUnavailable as error:
        result["validity"] = "valid_bounded_unresolved_no_solver_call"
        result["budget_block"] = str(error)
        ledger["blocked_before_solve"] = True
        ledger["budget_block"] = str(error)
    except BaseException as error:  # noqa: BLE001 - preserve raw and primary
        primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    ledger["local_problems_saved"] = sum(
        (out / ("leg-" + leg) / "pre-solve.npz").is_file() for leg in LEGS)
    _save_attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"], local_solver_calls=0,
        solver_wall_s=ledger["solver_wall_s"],
        solver_soft_limit_overrun_s=ledger.get("solver_soft_limit_overrun_s",
            soft_limit_overrun(ledger["solver_wall_s"])),
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0) + elapsed(),
        retries=0)
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__, "message": str(primary)}
    _save_attempt(errors, "execution-errors.json",
                  lambda: save_json(out / "execution-errors.json", errors))
    _save_attempt(errors, "result-preliminary", lambda: save_json(out / "result.json",
        dict(result, validity="incomplete_or_invalid", finalization_pending=True)))
    manifest = _save_attempt(errors, "output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E214 output manifest missing or over 512 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E214 hard internal deadline exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors:
        result["validity"] = "incomplete_or_invalid"
    result["finalization_pending"] = False
    try:
        save_json(out / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - retain primary
        errors.append({"stage": "result-final", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if primary is None:
            primary = error
    if errors:
        _save_attempt(errors, "publication-failure.json",
                      lambda: save_json(out / "publication-failure.json", errors))
        if primary is None:
            primary = RuntimeError("E214 raw/publication failure")
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
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
