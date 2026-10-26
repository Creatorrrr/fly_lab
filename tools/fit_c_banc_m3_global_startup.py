"""E211 one bounded global-startup Clarabel fit of frozen E210 physical cones.

The saved local problems supply all QP coefficients.  This module neither
simulates the neural network nor changes the E209 response family.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fit_c_banc_m3_causal_neural_basis import (
    leg_anatomy,
    physical_joint_matrix,
    proxy_verdict,
)
from tools.fit_c_banc_m3_constrained_tangent import csc_arrays
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

FREEZE_SHA = "042377087a2068dd4af20f1adeaf1738bc0cc96b7e5b2e43a43bdd557883f450"
FREEZE_SCHEMA = "flylab.m3.e211.global-startup-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e211.global-startup-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_global_startup.py",
         "tests/test_c_banc_global_startup_fit.py")
LEG_WIDTHS = (976, 864, 896, 992, 880, 880)
PAD = 1 - 1e-6
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 1, "retries": 0, "internal_s": 240,
    "solver_s": 180, "external_s": 270, "supervisor_s": 300,
    "resident_bytes": 8589934592, "output_bytes": 536870912,
    "minimum_free_bytes": 4294967296}


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E211 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E211 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E211 path")
    if file and not path.is_file():
        raise ValueError("E211 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E211 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E211 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    expected_shape = {"variables": 5488, "motors": 391, "free_motors": 343,
        "fixed_motors": 48, "basis_count": 16, "samples": 2001,
        "startup_rows": 99, "global_startup_values": 4158,
        "global_rows": 34451, "global_nnz_A": 10767527,
        "global_nnz_P": 617784, "leg_variables": list(LEG_WIDTHS), "cones": 19}
    expected_solver = {"name": "clarabel", "max_threads": 1, "max_iter": 150,
        "time_limit_s": 180.0, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8, "coordinate": "physical beta in six leg blocks",
        "objective": "sum unchanged E210 normalized leg objectives",
        "beta_min": 0.0, "beta_max": 1.0, "PAD": PAD, "raw_clip": False,
        "weak_column_pruning": False, "converged_status": "Solved"}
    expected_proxy = {"global_command_ratio": .8, "leg_command_ratio": 1.,
        "leg_velocity_ratio": 1., "global_velocity_ratio": 1.,
        "startup_q_rms": .018, "startup_q_max": .04,
        "startup_tendon_rms": .009, "startup_tendon_max": .025,
        "original_command_rms": .056686385349786386,
        "training_extra": "global startup norm with unchanged PAD; no per-leg startup RMS gate"}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E211_GLOBAL_STARTUP_FIT_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != expected_shape
            or freeze.get("solver") != expected_solver
            or freeze.get("proxy") != expected_proxy
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("clock") != {"neural_dt_s": .00025,
                                      "control_dt_s": .005, "ticks_per_control": 20}
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E211 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E211 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E211 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E211 parent source review does not authorize these sources")
    for label, count in (("input_sha256", 166),
                         ("reused_source_sha256", 21),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E211 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E211 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E211 preflight exceeded internal deadline")
    expected_local = "verification/banc-natural-gait-m3-20260919/e210-causal-neural-basis-fit"
    if freeze.get("local_problem_directory") != expected_local:
        raise ValueError("E211 frozen local problem directory changed")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("E211 frozen paths missing")
    resolved = {key: repository_path(value, file=key != "start")
                for key, value in paths.items()}
    return freeze, review, resolved


def _restore_csc(saved, prefix):
    matrix = sparse.csc_matrix(
        (np.asarray(saved[prefix + "_data"], np.float64),
         np.asarray(saved[prefix + "_indices"], np.int32),
         np.asarray(saved[prefix + "_indptr"], np.int32)),
        shape=tuple(np.asarray(saved[prefix + "_shape"], np.int64)))
    if not matrix.has_sorted_indices or not np.isfinite(matrix.data).all():
        raise ValueError("E211 saved CSC sorted/finite contract failed: " + prefix)
    return matrix


def load_local_problem(path, expected_leg, affine_sha):
    """Load one frozen physical cone and validate ownership/order/source."""
    if expected_leg not in LEGS:
        raise ValueError("E211 unknown leg")
    with np.load(path, allow_pickle=False) as saved:
        P = _restore_csc(saved, "P")
        A = _restore_csc(saved, "A")
        M = _restore_csc(saved, "startup_matrix")
        dims = np.asarray(saved["cone_dimensions"], np.int64)
        kinds = [str(v) for v in saved["cone_types"]]
        width = LEG_WIDTHS[LEGS.index(expected_leg)]
        cols = np.asarray(saved["motor_columns"], np.int64)
        linear = np.asarray(saved["linear"], np.float64)
        b = np.asarray(saved["b"], np.float64)
        offset = np.asarray(saved["startup_offset"], np.float64)
        constant = float(saved["objective_constant"])
        if (P.shape != (width, width) or A.shape[1] != width
                or M.shape != (693, width) or len(cols) * 16 != width
                or len(np.unique(cols)) != len(cols)
                or np.any((cols < 0) | (cols >= 391))
                or linear.shape != (width,) or b.shape != (A.shape[0],)
                or offset.shape != (693,) or dims.shape != (4,)
                or kinds != ["NonnegativeConeT"] + ["SecondOrderConeT"] * 3
                or dims[0] != 2 * width + 2 * 693
                or dims[3] != 694 or dims.sum() != A.shape[0]
                or not np.isfinite(linear).all() or not np.isfinite(b).all()
                or not np.isfinite(offset).all() or not np.isfinite(constant)
                or str(saved["source_affine_sha256"]) != affine_sha
                or not np.all(P.indices <= np.repeat(
                    np.arange(width), np.diff(P.indptr)))):
            raise ValueError("E211 frozen E210 local problem contract changed: " + expected_leg)
        # Local final SOC is exactly the normalized raw startup rows.
        tail = A[-694:]
        if (tail[0].nnz != 0 or not np.array_equal(tail[1:].toarray(),
                (-M / np.sqrt(693)).toarray())
                or not np.array_equal(b[-694:],
                    np.r_[.018 * PAD, offset / np.sqrt(693)])):
            raise ValueError("E211 local startup cone is not saved raw startup")
        return {"leg": expected_leg, "P": P, "A": A, "M": M,
                "linear": linear.copy(), "b": b.copy(), "dims": dims.copy(),
                "kinds": kinds, "motor_columns": cols.copy(),
                "startup_offset": offset.copy(), "objective_constant": constant,
                "q_offset": np.asarray(saved["q_offset"], np.float64).copy(),
                "velocity_offset": np.asarray(saved["velocity_offset"], np.float64).copy(),
                "normalizer": float(saved["normalizer"]),
                "command_limit": float(saved["command_limit"]),
                "velocity_limit": float(saved["velocity_limit"])}


def global_startup_soc(matrix, offset, limit=.018):
    """One SOC for the physical global RMS of all startup samples."""
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    offset = np.asarray(offset, np.float64).reshape(-1)
    n, width = matrix.shape
    if n == 0 or offset.shape != (n,) or not np.isfinite(matrix.data).all():
        raise ValueError("E211 aligned finite global startup required")
    rows = sparse.vstack(
        (sparse.csc_matrix((1, width)), -matrix / np.sqrt(n)), format="csc")
    rhs = np.r_[limit * PAD, offset / np.sqrt(n)]
    return rows, rhs


def assemble_global(locals_, *, strict=True):
    """Concatenate six immutable local QPs, replacing six startup SOCs by one."""
    if len(locals_) != 6 or [v["leg"] for v in locals_] != list(LEGS):
        raise ValueError("E211 fixed six-leg order required")
    cols = np.concatenate([v["motor_columns"] for v in locals_])
    total_width = sum(v["P"].shape[0] for v in locals_)
    if (len(np.unique(cols)) != len(cols)
            or (strict and (len(cols) != 343 or total_width != 5488))):
        raise ValueError("E211 free motor columns overlap or missing")
    P = sparse.block_diag([v["P"] for v in locals_], format="csc")
    linear = np.concatenate([v["linear"] for v in locals_])
    constant = float(sum(v["objective_constant"] for v in locals_))
    local_A = sparse.block_diag([v["A"][:-int(v["dims"][-1])] for v in locals_],
                                format="csc")
    local_b = np.concatenate([v["b"][:-int(v["dims"][-1])] for v in locals_])
    M = sparse.block_diag([v["M"] for v in locals_], format="csc")
    e = np.concatenate([v["startup_offset"] for v in locals_])
    n = len(e)
    if (M.shape != (n, total_width) or n <= 0
            or (strict and n != 4158)):
        raise ValueError("E211 global startup dimensions changed")
    global_soc, global_rhs = global_startup_soc(M, e)
    A = sparse.vstack((local_A, global_soc), format="csc")
    b = np.r_[local_b, global_rhs]
    dims = np.r_[np.concatenate([v["dims"][:3] for v in locals_]), n + 1]
    kinds = [kind for v in locals_ for kind in v["kinds"][:3]]
    kinds.append("SecondOrderConeT")
    if ((strict and (A.shape != (34451, 5488)
            or A.nnz != 10767527 or P.nnz != 617784))
            or len(dims) != 19 or int(dims.sum()) != len(b)
            or not np.isfinite(A.data).all() or not np.isfinite(P.data).all()
            or not np.isfinite(linear).all() or not np.isfinite(b).all()):
        raise ValueError("E211 frozen global sparse shape/finite contract changed")
    cones = [clarabel.NonnegativeConeT(int(size)) if kind == "NonnegativeConeT"
             else clarabel.SecondOrderConeT(int(size))
             for kind, size in zip(kinds, dims)]
    return {"P": P, "linear": linear, "objective_constant": constant,
            "A": A, "b": b, "cone_dimensions": dims, "cone_types": kinds,
            "cones": cones, "motor_columns": cols, "startup_matrix": M,
            "startup_offset": e, "width": total_width}


def save_problem(out, problem, affine_sha):
    arrays = {"linear": problem["linear"], "b": problem["b"],
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "objective_constant": np.asarray(problem["objective_constant"]),
        "motor_columns": problem["motor_columns"],
        "startup_offset": problem["startup_offset"],
        "source_affine_sha256": np.asarray(affine_sha)}
    for name in ("P", "A", "startup_matrix"):
        arrays.update(csc_arrays(name, problem[name]))
    np.savez(Path(out) / "pre-solve.npz", **arrays)


def solver_settings():
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 150
    settings.time_limit = 180.
    settings.tol_gap_abs = 1e-8
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-8
    return settings


class SolverAllowanceUnavailable(TimeoutError):
    """The frozen full solve allowance no longer fits before a call."""


def ensure_solve_budget(elapsed):
    if elapsed() + RESOURCE["solver_s"] > RESOURCE["internal_s"]:
        raise SolverAllowanceUnavailable("E211 less than 180 s solver allowance remains")


def soft_limit_overrun(wall_s, reported_s=None):
    """A Clarabel time limit is soft and an overrun is diagnostic only."""
    durations = [float(wall_s)]
    if reported_s is not None and np.isfinite(reported_s):
        durations.append(float(reported_s))
    return max(0., max(durations) - RESOURCE["solver_s"])


def validate_raw_shapes(problem, raw):
    """The raw file must be written before a malformed return is rejected."""
    width, rows = problem["A"].shape[1], problem["A"].shape[0]
    for key, shape in (("solver_x", (width,)), ("solver_s", (rows,)),
                       ("solver_z", (rows,))):
        value = raw.get(key)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError("E211 solver raw schema shape mismatch: " + key)


def solve_once(problem, out, ledger, *, elapsed, solver_factory=None):
    """Persist request/call and raw return before assessment; one call only."""
    out = Path(out)
    if ledger.get("solve_called") or ledger.get("solver_calls", 0) != 0:
        raise RuntimeError("E211 one-call solver already called")
    if solver_factory is None:
        solver_factory = clarabel.DefaultSolver
    settings = solver_settings()
    save_json(out / "solver-settings.json", {"configured": {
        "max_threads": 1, "max_iter": 150, "time_limit_s": 180.,
        "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8},
        "all_settings_text": str(settings)})
    ensure_solve_budget(elapsed)
    ledger["solve_requested"] = True
    save_json(out / "ledger.json", ledger)
    ledger["constructor_entered"] = True
    t0 = time.perf_counter()
    constructor_error = None
    try:
        solver = solver_factory(problem["P"], problem["linear"], problem["A"],
                                problem["b"], problem["cones"], settings)
        ledger["constructor_completed"] = True
    except BaseException as error:
        constructor_error = error
        raise
    finally:
        ledger["constructor_wall_s"] = time.perf_counter() - t0
        try:
            save_json(out / "ledger.json", ledger)
        except BaseException:
            if constructor_error is None:
                raise
    ensure_solve_budget(elapsed)
    ledger["solve_called"] = True
    ledger["solver_calls"] = 1
    try:
        save_json(out / "ledger.json", ledger)
    except BaseException:
        ledger["solve_called"] = False
        ledger["solver_calls"] = 0
        raise
    t1 = time.perf_counter()
    solve_error = None
    try:
        solution = solver.solve()
        ledger["solver_returned"] = True
    except BaseException as error:
        solve_error = error
        raise
    finally:
        ledger["solver_wall_s"] = time.perf_counter() - t1
        try:
            save_json(out / "ledger.json", ledger)
        except BaseException:
            if solve_error is None:
                raise
    raw = {"solver_x": np.asarray(solution.x, np.float64),
           "solver_s": np.asarray(solution.s, np.float64),
           "solver_z": np.asarray(solution.z, np.float64)}
    np.savez(out / "solver-raw.npz", **raw)
    info = {"status": str(solution.status),
        "iterations": int(solution.iterations),
        "objective_primal": float(solution.obj_val),
        "objective_dual": float(solution.obj_val_dual),
        "primal_residual": float(solution.r_prim),
        "dual_residual": float(solution.r_dual),
        "solve_time_s": float(solution.solve_time),
        "process_call_wall_s": time.perf_counter() - t1}
    save_json(out / "solver-info.json", info)
    ledger["solver_soft_limit_overrun_s"] = soft_limit_overrun(
        ledger["solver_wall_s"], info["solve_time_s"])
    save_json(out / "ledger.json", ledger)
    validate_raw_shapes(problem, raw)
    return raw, info


def assess_solution(problem, raw, info, locals_):
    """Separate finite bounded coefficients from solver/numerical/direct checks."""
    width, rows = problem["A"].shape[1], problem["A"].shape[0]
    report = {"status": info.get("status"),
              "status_solved": info.get("status") == "Solved"}
    for key, shape in (("solver_x", (width,)), ("solver_s", (rows,)),
                       ("solver_z", (rows,))):
        value = raw.get(key)
        report[key + "_shape_finite"] = bool(isinstance(value, np.ndarray)
            and value.shape == shape and np.isfinite(value).all())
    x = raw.get("solver_x")
    report["box_pass"] = bool(report["solver_x_shape_finite"]
        and np.min(x) >= 0 and np.max(x) <= 1)
    report["eligible_beta"] = bool(report["status_solved"] and report["box_pass"])
    if not report["eligible_beta"]:
        report["numerical_pass"] = False
        report["direct_training_pass"] = False
        return report
    report["diagnostics_finite"] = bool(info.get("iterations", -1) >= 0 and
        all(np.isfinite(info.get(key, np.nan)) for key in (
            "objective_primal", "objective_dual", "primal_residual",
            "dual_residual", "solve_time_s")))
    report["primal_equality_max"] = None
    report["cone_violation_max"] = None
    if report["solver_s_shape_finite"] and report["solver_z_shape_finite"]:
        s = raw["solver_s"]
        equality = problem["A"] @ x + s - problem["b"]
        report["primal_equality_max"] = float(np.max(np.abs(equality)))
        violation = 0.
        index = 0
        for kind, size in zip(problem["cone_types"], problem["cone_dimensions"]):
            cone_s = s[index:index + size]
            if kind == "NonnegativeConeT":
                violation = max(violation, -float(np.min(cone_s)))
            else:
                violation = max(violation, float(np.linalg.norm(cone_s[1:]) - cone_s[0]))
            index += size
        report["cone_violation_max"] = max(0., violation)
    direct = []
    offset = 0
    for local in locals_:
        width_local = local["P"].shape[0]
        beta = x[offset:offset + width_local]
        offset += width_local
        startup = local["M"] @ beta + local["startup_offset"]
        direct.append({"leg": local["leg"], "startup_q_rms":
            float(np.sqrt(np.mean(startup * startup))),
            "startup_q_max": float(np.max(np.abs(startup)))})
    global_startup = problem["startup_matrix"] @ x + problem["startup_offset"]
    report["direct"] = {"startup_q_rms":
        float(np.sqrt(np.mean(global_startup * global_startup))),
        "startup_q_max": float(np.max(np.abs(global_startup))),
        "legs": direct}
    report["padding_residual"] = {
        "startup_q_rms_above_padded":
            max(0., report["direct"]["startup_q_rms"] - .018 * PAD),
        "startup_q_max_above_padded":
            max(0., report["direct"]["startup_q_max"] - .04 * PAD)}
    report["direct_training_pass"] = bool(
        report["direct"]["startup_q_rms"] <= .018
        and report["direct"]["startup_q_max"] <= .04)
    objective = float(.5 * x @ (problem["P"] + problem["P"].T -
        sparse.diags(problem["P"].diagonal())) @ x +
        problem["linear"] @ x + problem["objective_constant"])
    report["objective_direct"] = objective
    report["objective_pass"] = bool(report["diagnostics_finite"] and abs(
        objective - (info["objective_primal"] + problem["objective_constant"])) <= 1e-6)
    report["numerical_pass"] = bool(report["diagnostics_finite"]
        and report["primal_equality_max"] is not None
        and report["primal_equality_max"] <= 1e-6
        and report["cone_violation_max"] <= 1e-6
        and report["objective_pass"])
    return report


def physical_limit_diagnostics(physical_legs, locals_):
    """Report PAD margins separately from unchanged physical nonworsening."""
    q_original = [bool(v["q_rms"] <= local["command_limit"])
                  for v, local in zip(physical_legs, locals_)]
    v_original = [bool(v["velocity_rms"] <= local["velocity_limit"])
                  for v, local in zip(physical_legs, locals_)]
    return {"legs": physical_legs,
        "leg_q_original_limit_pass": q_original,
        "leg_velocity_original_limit_pass": v_original,
        "leg_q_padding_residual": [
            max(0., v["q_rms"] - local["command_limit"] * PAD)
            for v, local in zip(physical_legs, locals_)],
        "leg_velocity_padding_residual": [
            max(0., v["velocity_rms"] - local["velocity_limit"] * PAD)
            for v, local in zip(physical_legs, locals_)]}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e211.output-manifest.v1",
                "files": files,
                "total_bytes": sum(value["bytes"] for value in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def _save_attempt(errors, stage, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve publication failure
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E211 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = review = paths = None
    primary = None
    errors = []
    ledger = {"schema": "flylab.m3.e211.ledger.v1", "solve_requested": False,
        "constructor_entered": False, "constructor_completed": False,
        "solve_called": False, "solver_returned": False, "solver_calls": 0,
        "constructor_wall_s": 0., "solver_wall_s": 0.,
        "full_banc_calls": 0, "cuda_calls": 0, "physics_steps": 0, "retries": 0}
    result = {"schema": "flylab.m3.e211.global-startup-fit.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "conditional_proxy_only": True}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E211 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            affine_sha = freeze["input_sha256"][freeze["paths"]["affine"]]
            local_dir = repository_path(freeze["local_problem_directory"], file=False)
            locals_ = [load_local_problem(local_dir / ("leg-" + leg) / "pre-solve.npz",
                                          leg, affine_sha) for leg in LEGS]
            problem = assemble_global(locals_)
            # Verify motor ownership against the original anatomical snapshot
            # before constructing or calling the solver.
            from flylab.c.storage import StateStore
            start = StateStore.load(paths["start"])
            with np.load(paths["mask"], allow_pickle=False) as mask:
                free = mask["free_columns"].astype(np.int64)
                fixed = mask["fixed_columns"].astype(np.int64)
            membership, moment, motors_by_leg = leg_anatomy(start, free)
            if (not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or any(not np.array_equal(locals_[i]["motor_columns"],
                                              motors_by_leg[i]) for i in range(6))):
                raise ValueError("E211 saved motor ownership differs from E209")
            save_problem(out, problem, affine_sha)
            ensure_solve_budget(elapsed)
            raw, info = solve_once(problem, out, ledger, elapsed=elapsed)
            result["solver_info"] = info
            report = assess_solution(problem, raw, info, locals_)
            save_json(out / "direct-metrics.json", report)
            result["solver_assessment"] = report
            if report["eligible_beta"]:
                trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
                target = load_targets(paths["targets"], trace)
                with (np.load(paths["affine"], allow_pickle=False) as affine,
                      np.load(paths["basis"], allow_pickle=False) as basis,
                      np.load(paths["basis_parameters"], allow_pickle=False) as params,
                      np.load(paths["endpoints"], allow_pickle=False) as native,
                      np.load(paths["mask"], allow_pickle=False) as mask):
                    D = affine["D"].copy()
                    q0 = affine["q_intercept"].copy()
                    H = basis["H"].copy()
                    B = basis["B"].copy()
                    cap = params["cap"].copy()
                    free = mask["free_columns"].astype(np.int64)
                    fixed = mask["fixed_columns"].astype(np.int64)
                    original_rates = native["motor_rate"].copy()
                    if (D.shape != (2001, 391, 16) or q0.shape != (2001, 42)
                            or H.shape != (2001, 391) or B.shape != D.shape
                            or cap.shape != (391,) or free.shape != (343,)
                            or fixed.shape != (48,) or original_rates.shape != H.shape
                            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                            or not np.array_equal(mask["motor_ids"], params["motor_ids"])
                            or not np.array_equal(mask["membership"],
                                start["adapter"]["parameters"]["membership"])
                            or np.any(D[:, fixed] != 0)):
                        raise ValueError("E211 frozen basis/anatomy changed")
                beta = np.zeros((391, 16), np.float64)
                index = 0
                for motors in motors_by_leg:
                    width = len(motors) * 16
                    beta[motors] = raw["solver_x"][index:index + width].reshape(-1, 16)
                    index += width
                rate = H + cap[None, :] * np.sum(B * beta[None], axis=2)
                rate[:, fixed] = original_rates[:, fixed]
                direct = canonical_readout(start, rate)
                q_affine = q0.copy()
                for i, motors in enumerate(motors_by_leg):
                    joint = physical_joint_matrix(D, membership, moment, motors, i)
                    q_affine[:, 7*i:7*i+7] += joint @ beta[motors].reshape(-1)
                np.savez(out / "candidate.npz", beta=beta, rate=rate,
                    q_affine=q_affine, q_direct=direct["command"],
                    tendon=direct["tendon"], activation=direct["activation"])
                canonical_original = canonical_readout(start, original_rates)
                decoder = DecoderAdjoint(start)
                old_mono = decoder.forward(original_rates)
                new_mono = decoder.forward(rate)
                np.savez(out / "readout-2x2.npz",
                    original_monotone_q=old_mono["command"],
                    original_monotone_tendon=old_mono["tendon"],
                    new_monotone_q=new_mono["command"],
                    new_monotone_tendon=new_mono["tendon"],
                    original_canonical_q=canonical_original["command"],
                    original_canonical_tendon=canonical_original["tendon"],
                    new_canonical_q=direct["command"],
                    new_canonical_tendon=direct["tendon"])
                ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
                affine_exact = bool(np.allclose(q_affine, direct["command"],
                    atol=2e-12, rtol=2e-12))
                ltm_exact = bool(np.array_equal(
                    canonical_original["activation"][:, ltm],
                    direct["activation"][:, ltm]) and np.array_equal(
                    canonical_original["tendon"], direct["tendon"]))
                metrics = direct_metrics(direct, target)
                indices = np.asarray(target["indices"], np.int64)
                pair_start = np.asarray(target["pair_start"], np.int64)
                pair_end = np.asarray(target["pair_end"], np.int64)
                physical_legs = []
                for i, local in enumerate(locals_):
                    j = slice(7 * i, 7 * (i + 1))
                    qres = q_affine[indices, j] - target["target"][:, j]
                    vres = ((q_affine[pair_end, j] - q_affine[pair_start, j]) / .005
                            - target["velocity"][:, j])
                    sres = q_affine[1:100, j] - target["startup_command"][:, j]
                    physical_legs.append({"leg": local["leg"],
                        "q_rms": float(np.sqrt(np.mean(qres * qres))),
                        "velocity_rms": float(np.sqrt(np.mean(vres * vres))),
                        "startup_q_rms": float(np.sqrt(np.mean(sres * sres))),
                        "startup_q_max": float(np.max(np.abs(sres)))})
                direct_physical = physical_limit_diagnostics(physical_legs, locals_)
                direct_physical["startup_raw_affine_match"] = bool(all(
                        np.isclose(v["startup_q_rms"],
                                   report["direct"]["legs"][i]["startup_q_rms"],
                                   atol=2e-12, rtol=2e-12)
                        for i, v in enumerate(physical_legs)))
                report["direct"]["physical"] = direct_physical
                report["direct_training_pass"] = bool(report["direct_training_pass"]
                    and all(direct_physical["leg_q_original_limit_pass"])
                    and all(direct_physical["leg_velocity_original_limit_pass"])
                    and direct_physical["startup_raw_affine_match"])
                save_json(out / "direct-metrics.json", report)
                verdict = proxy_verdict(metrics, target["baseline"])
                verdict.update(global_cone_numerical_pass=report["numerical_pass"],
                    global_training_pass=report["direct_training_pass"],
                    affine_exact=affine_exact, ltm_exact=ltm_exact)
                verdict["passed"] = bool(verdict["passed"]
                    and report["numerical_pass"] and report["direct_training_pass"]
                    and affine_exact and ltm_exact)
                result.update(validity="valid_conditional_fit_rejected_or_supported",
                    metrics=metrics, baseline=target["baseline"], proxy_verdict=verdict)
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
    _save_attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"], solver_wall_s=ledger["solver_wall_s"],
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
    _save_attempt(errors, "result-preliminary",
        lambda: save_json(out / "result.json",
            dict(result, validity="incomplete_or_invalid", finalization_pending=True)))
    manifest = _save_attempt(errors, "output-manifest.json",
                             lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E211 output manifest missing or over 512 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E211 hard internal deadline exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors:
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
        _save_attempt(errors, "publication-failure.json",
                      lambda: save_json(out / "publication-failure.json", errors))
        if primary is None:
            primary = RuntimeError("E211 raw/publication failure")
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
