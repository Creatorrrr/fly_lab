"""E210 six bounded leg-local convex fits on fixed E209 conditional bases.

No neural simulation, CUDA work, body work, or model adoption occurs here.
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

from tools.fit_c_banc_m3_constrained_tangent import (
    compress_norm,
    csc_arrays,
    soc_rows,
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

FREEZE_SHA = "b8f747978a7516804232c50c22fe7435c927553ab7117752a7814f09989506c1"
FREEZE_SCHEMA = "flylab.m3.e210.causal-neural-basis-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e210.causal-neural-basis-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_causal_neural_basis.py",
         "tests/test_c_banc_causal_neural_basis_fit.py")
LEG_WIDTHS = (976, 864, 896, 992, 880, 880)
PAD = 1 - 1e-6
RIDGE = 1e-8
SHAPE = {"motors": 391, "free_motors": 343, "fixed_motors": 48,
    "basis_count": 16, "variables": 5488, "samples": 2001,
    "target_rows": 1824, "velocity_pairs": 1823, "startup_rows": 99,
    "leg_variables": list(LEG_WIDTHS)}
CLOCK = {"neural_dt_s": .00025, "control_dt_s": .005,
         "ticks_per_control": 20}
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 6, "retries": 0, "internal_s": 180,
    "solver_total_s": 120, "solver_per_leg_s": 20,
    "external_s": 220, "supervisor_s": 240,
    "resident_bytes": 8589934592, "output_bytes": 805306368,
    "minimum_free_bytes": 4294967296}
SOLVER = {"name": "clarabel", "order": list(LEGS), "max_iter": 150,
    "time_limit_s": 20., "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
    "tol_feas": 1e-8, "coordinate": "physical beta", "beta_min": 0.,
    "beta_max": 1., "ridge_mean_beta2": RIDGE, "PAD": PAD,
    "raw_clip": False, "weak_column_pruning": False,
    "numerical_primal_residual_max": 1e-6, "converged_status": "Solved"}


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E210 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E210 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E210 path")
    if file and not path.is_file():
        raise ValueError("E210 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha,
                *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E210 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E210 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E210_CAUSAL_BASIS_FIT_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != SHAPE or freeze.get("clock") != CLOCK
            or freeze.get("solver") != SOLVER
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E210 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E210 design SHA changed")
    for label, count in (("input_sha256", 112),
                         ("reused_source_sha256", 19),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E210 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E210 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E210 preflight exceeded internal deadline")
    paths = freeze.get("paths", {})
    if not isinstance(paths, dict) or set(paths) != {
            "current", "parameters", "endpoints", "observation", "trace", "native",
            "start", "e201_manifest", "e201_result", "e201_review",
            "positive", "negative", "drive", "signed_result", "signed_manifest",
            "signed_review", "targets", "target_freeze", "mask", "basis", "history",
            "prototypes", "basis_parameters", "affine", "canonical_original",
            "e209_result", "e209_manifest", "e209_ledger", "e209_final_review",
            "e209_supervisor"}:
        raise ValueError("E210 frozen original path map changed")
    resolved = {key: repository_path(value, file=key != "start")
                for key, value in paths.items()}
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E210 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E210 parent source review does not authorize these sources")
    return freeze, review, resolved


def leg_anatomy(start, free):
    a = start["adapter"]
    membership = np.asarray(a["parameters"]["membership"], np.int64)
    moment = np.asarray(a["calibration"]["moment"], np.float64)
    groups = a["groups"]
    if (membership.shape != (391,) or moment.shape != (42, 96)
            or len(groups) != 96 or np.any((membership < 0) | (membership >= 96))
            or not np.array_equal(np.sort(free), free)):
        raise ValueError("E210 native anatomical ownership changed")
    muscle_leg = [group["leg"] for group in groups]
    if any(leg not in LEGS for leg in muscle_leg):
        raise ValueError("E210 muscle without one of six legs")
    for muscle, leg in enumerate(muscle_leg):
        index = LEGS.index(leg)
        other = np.r_[0:7 * index, 7 * (index + 1):42]
        if np.any(moment[other, muscle] != 0):
            raise ValueError("E210 moment crosses anatomical leg")
    motors = []
    for leg in LEGS:
        motors.append(np.asarray([m for m in free
            if muscle_leg[membership[m]] == leg], np.int64))
    if tuple(len(v) * 16 for v in motors) != LEG_WIDTHS:
        raise ValueError("E210 fixed per-leg physical beta widths changed")
    return membership, moment, motors


def physical_joint_matrix(D, membership, moment, motors, leg_index):
    """No beta scaling: one muscle owner and original moment for each column."""
    D = np.asarray(D, np.float64)
    motors = np.asarray(motors, np.int64)
    if D.ndim != 3 or D.shape[1:] != (391, 16):
        raise ValueError("E210 fixed E209 D shape changed")
    gains = .02 * moment[7 * leg_index:7 * leg_index + 7,
                          membership[motors]]
    joint = D[:, motors, :][:, None, :, :] * gains[None, :, :, None]
    return np.ascontiguousarray(joint.reshape(len(D), 7, len(motors) * 16))


def quadratic_objective(matrix, offset, *, ridge=RIDGE):
    """Clarabel 0.5*x'Px+q'x plus separately retained constant."""
    matrix = np.asarray(matrix, np.float64)
    offset = np.asarray(offset, np.float64)
    width = matrix.shape[1]
    if matrix.ndim != 2 or offset.shape != (len(matrix),) or width < 1:
        raise ValueError("E210 aligned normalized command residual required")
    gram = matrix.T @ matrix + ridge / width * np.eye(width)
    return (sparse.triu(sparse.csc_matrix(2 * gram), format="csc"),
            2 * matrix.T @ offset, float(offset @ offset))


def compressed_soc(matrix, offset, physical_limit, width, *, compress=compress_norm):
    normalizer = np.sqrt(len(offset)) * physical_limit
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("E210 positive original norm limit required")
    factor, shifted, note = compress(matrix / normalizer, offset / normalizer)
    rows, rhs = soc_rows(factor, shifted, PAD, width=width)
    return rows, rhs, factor, shifted, {"normalizer": float(normalizer),
        "physical_limit": float(physical_limit), **note}


def box_and_startup(startup, offset, width):
    """Unpadded beta box and padded pointwise startup inequalities."""
    startup = sparse.csr_matrix(startup, dtype=np.float64)
    offset = np.asarray(offset, np.float64).reshape(-1)
    if startup.shape != (len(offset), width):
        raise ValueError("E210 startup rows/physical beta width differ")
    identity = sparse.eye(width, format="csr")
    A = sparse.vstack((-identity, identity, startup, -startup), format="csr")
    b = np.r_[np.zeros(width), np.ones(width),
              .04 * PAD - offset, .04 * PAD + offset]
    return A, b


def build_leg_problem(start, D, q_intercept, target, motors_by_leg, leg_index,
                      *, compress=compress_norm):
    """Construct one original-coordinate physical beta QP/SOC."""
    membership = np.asarray(start["adapter"]["parameters"]["membership"], np.int64)
    moment = np.asarray(start["adapter"]["calibration"]["moment"], np.float64)
    motors = np.asarray(motors_by_leg[leg_index], np.int64)
    leg = LEGS[leg_index]
    width = len(motors) * 16
    q0 = np.asarray(q_intercept, np.float64)
    indices = np.asarray(target["indices"], np.int64)
    starts = np.asarray(target["pair_start"], np.int64)
    ends = np.asarray(target["pair_end"], np.int64)
    if (q0.shape != (2001, 42) or D.shape != (2001, 391, 16)
            or width != LEG_WIDTHS[leg_index] or indices.shape != (1824,)
            or starts.shape != (1823,) or ends.shape != (1823,)
            or not np.array_equal(indices, np.arange(100, 1924))
            or not np.array_equal(starts, indices[:-1])
            or not np.array_equal(ends, indices[1:])
            or np.any(D[0] != 0) or not np.isfinite(D).all()
            or not np.isfinite(q0).all()):
        raise ValueError("E210 original causal affine/target rows changed")
    joint = physical_joint_matrix(D, membership, moment, motors, leg_index)
    j0 = 7 * leg_index
    qres = (q0[indices, j0:j0 + 7] - target["target"][:, j0:j0 + 7]).reshape(-1)
    vres = ((q0[ends, j0:j0 + 7] - q0[starts, j0:j0 + 7]) / .005
        - target["velocity"][:, j0:j0 + 7]).reshape(-1)
    sres = (q0[1:100, j0:j0 + 7]
            - target["startup_command"][:, j0:j0 + 7]).reshape(-1)
    jq = joint[indices].reshape(1824 * 7, width)
    jv = ((joint[ends] - joint[starts]) / .005).reshape(1823 * 7, width)
    js = joint[1:100].reshape(99 * 7, width)
    baseline = target["baseline"]
    command_limit = baseline["leg_command_rms"][leg_index]
    velocity_limit = baseline["leg_velocity_rms"][leg_index]
    normalizer = np.sqrt(len(qres)) * command_limit
    if (normalizer <= 0 or velocity_limit <= 0
            or not np.isfinite(normalizer) or not np.isfinite(velocity_limit)):
        raise ValueError("E210 positive original per-leg baseline limits required")
    C, c = jq / normalizer, qres / normalizer
    P, linear, constant = quadratic_objective(C, c)
    norm_rows = []
    norm_rhs = []
    factors = {}
    notes = {}
    for name, matrix, offset, limit in (("command", jq, qres, command_limit),
                                         ("velocity", jv, vres, velocity_limit)):
        rows, rhs, factor, shifted, note = compressed_soc(
            matrix, offset, limit, width, compress=compress)
        norm_rows.append(rows)
        norm_rhs.append(rhs)
        factors[name + "_factor"] = factor
        factors[name + "_offset"] = shifted
        notes[name] = note
    startup = sparse.csr_matrix(js)
    nonnegative, nn_rhs = box_and_startup(startup, sres, width)
    if not np.isfinite(nn_rhs).all():
        raise ValueError("E210 startup bound nonfinite")
    startup_rows, startup_rhs = soc_rows(startup / np.sqrt(len(sres)),
        sres / np.sqrt(len(sres)), .018 * PAD, width=width)
    A = sparse.vstack((nonnegative, *norm_rows, startup_rows), format="csc")
    b = np.r_[nn_rhs, *norm_rhs, startup_rhs]
    cone_dimensions = np.asarray([len(nn_rhs), *(len(v) for v in norm_rhs),
                                  len(startup_rhs)], np.int64)
    cone_types = ["NonnegativeConeT", "SecondOrderConeT",
                  "SecondOrderConeT", "SecondOrderConeT"]
    cones = [clarabel.NonnegativeConeT(int(cone_dimensions[0])),
             *(clarabel.SecondOrderConeT(int(v)) for v in cone_dimensions[1:])]
    if (A.shape != (len(b), width) or not np.isfinite(A.data).all()
            or not np.isfinite(P.data).all() or not np.isfinite(linear).all()):
        raise ValueError("E210 nonfinite original conic problem")
    return {"leg": leg, "leg_index": leg_index, "motor_columns": motors,
        "width": width, "P": P, "linear": linear, "objective_constant": constant,
        "A": A, "b": b, "cones": cones, "cone_dimensions": cone_dimensions,
        "cone_types": cone_types, "factors": factors, "factor_notes": notes,
        "q_offset": qres, "velocity_offset": vres, "startup_offset": sres,
        "startup_matrix": startup, "q_matrix": jq, "velocity_matrix": jv,
        "normalizer": float(normalizer), "command_limit": float(command_limit),
        "velocity_limit": float(velocity_limit),
        "target_indices": indices, "pair_start": starts, "pair_end": ends}


def save_problem(path, problem, source_sha):
    """All exact solver inputs; dense q/velocity maps reconstruct from pinned D."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"linear": problem["linear"], "b": problem["b"],
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "objective_constant": np.asarray(problem["objective_constant"]),
        "motor_columns": problem["motor_columns"],
        "target_indices": problem["target_indices"],
        "pair_start": problem["pair_start"], "pair_end": problem["pair_end"],
        "q_offset": problem["q_offset"],
        "velocity_offset": problem["velocity_offset"],
        "startup_offset": problem["startup_offset"],
        "normalizer": np.asarray(problem["normalizer"]),
        "command_limit": np.asarray(problem["command_limit"]),
        "velocity_limit": np.asarray(problem["velocity_limit"]),
        "source_affine_sha256": np.asarray(source_sha)}
    arrays.update(csc_arrays("P", problem["P"]))
    arrays.update(csc_arrays("A", problem["A"]))
    arrays.update(csc_arrays("startup_matrix", problem["startup_matrix"]))
    np.savez(path / "pre-solve.npz", **arrays)
    np.savez(path / "norm-factors.npz", **problem["factors"])
    save_json(path / "norm-factors.json", problem["factor_notes"])


def solver_settings():
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 150
    settings.time_limit = 20.
    settings.tol_gap_abs = 1e-8
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-8
    return settings


def ensure_leg_budget(elapsed):
    """A full frozen 20 s solve allowance must remain after preparation."""
    if elapsed() + RESOURCE["solver_per_leg_s"] > RESOURCE["internal_s"]:
        raise TimeoutError("E210 less than one 20 s leg allowance remains")


def solve_leg_once(problem, path, ledger, *, entry=None, remaining_guard=None):
    """One physical-beta Clarabel call, storing raw before any status check."""
    path = Path(path)
    settings = solver_settings()
    if entry is None:
        entry = {}
    save_json(path / "solver-settings.json", {"configured": {
        "max_iter": settings.max_iter, "time_limit": settings.time_limit,
        "max_threads": settings.max_threads,
        "tol_gap_abs": settings.tol_gap_abs,
        "tol_gap_rel": settings.tol_gap_rel, "tol_feas": settings.tol_feas},
        "all_settings_text": str(settings)})
    entry["constructor_entered"] = True
    constructor_started = time.perf_counter()
    constructor_error = None
    try:
        solver = clarabel.DefaultSolver(problem["P"], problem["linear"],
            problem["A"], problem["b"], problem["cones"], settings)
        entry["constructor_completed"] = True
    except BaseException as error:
        constructor_error = error
        raise
    finally:
        duration = time.perf_counter() - constructor_started
        entry["constructor_wall_s"] = duration
        ledger["constructor_wall_s"] = ledger.get("constructor_wall_s", 0.) + duration
        try:
            save_json(path.parent / "ledger.json", ledger)
        except BaseException as storage_error:
            entry["ledger_storage_error"] = str(storage_error)
            if constructor_error is None:
                raise
    if remaining_guard is not None:
        remaining_guard()
    entry["solve_called"] = True
    ledger["solver_calls"] += 1
    try:
        save_json(path.parent / "ledger.json", ledger)
    except BaseException:
        # Persistence precedes the call; a failed write must not invent a solve.
        entry["solve_called"] = False
        ledger["solver_calls"] -= 1
        raise
    before = time.perf_counter()
    solve_error = None
    try:
        solution = solver.solve()
        entry["solver_returned"] = True
    except BaseException as error:
        solve_error = error
        raise
    finally:
        ledger["solver_wall_s"] += time.perf_counter() - before
        try:
            save_json(path.parent / "ledger.json", ledger)
        except BaseException as storage_error:
            entry["ledger_storage_error"] = str(storage_error)
            if solve_error is None:
                raise
    raw = {"solver_x": np.asarray(solution.x, np.float64),
        "solver_s": np.asarray(solution.s, np.float64),
        "solver_z": np.asarray(solution.z, np.float64)}
    np.savez(path / "solver-raw.npz", **raw)
    info = {"status": str(solution.status), "iterations": int(solution.iterations),
        "objective_primal": float(solution.obj_val),
        "objective_dual": float(solution.obj_val_dual),
        "primal_residual": float(solution.r_prim),
        "dual_residual": float(solution.r_dual),
        "solve_time_s": float(solution.solve_time),
        "process_call_wall_s": time.perf_counter() - before,
        "settings": {"max_iter": settings.max_iter,
            "time_limit_s": settings.time_limit,
            "max_threads": settings.max_threads,
            "tol_gap_abs": settings.tol_gap_abs,
            "tol_gap_rel": settings.tol_gap_rel,
            "tol_feas": settings.tol_feas}}
    save_json(path / "solver-info.json", info)
    return raw, info


def validate_solver_return(problem, raw, info):
    """Assess saved raw without discarding finite Solved beta on numeric miss."""
    width, rows = problem["width"], len(problem["b"])
    report = {"status": info.get("status"), "status_solved": info.get("status") == "Solved"}
    for key, shape in (("solver_x", (width,)),
                       ("solver_s", (rows,)), ("solver_z", (rows,))):
        value = raw.get(key)
        report[key + "_shape_finite"] = bool(isinstance(value, np.ndarray)
            and value.shape == shape and value.dtype == np.float64
            and np.isfinite(value).all())
    x = raw["solver_x"]
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
        index = 0
        violation = 0.
        for kind, size in zip(problem["cone_types"], problem["cone_dimensions"]):
            cone_s = s[index:index + size]
            if kind == "NonnegativeConeT":
                violation = max(violation, -float(np.min(cone_s)))
            else:
                violation = max(violation,
                    float(np.linalg.norm(cone_s[1:]) - cone_s[0]))
            index += size
        report["cone_violation_max"] = max(0., violation)
    report["numerical_pass"] = bool(report["diagnostics_finite"]
        and report["primal_equality_max"] is not None
        and report["primal_equality_max"] <= 1e-6
        and report["cone_violation_max"] <= 1e-6)
    q = problem["q_matrix"] @ x + problem["q_offset"]
    v = problem["velocity_matrix"] @ x + problem["velocity_offset"]
    startup = problem["startup_matrix"] @ x + problem["startup_offset"]
    q_rms = float(np.sqrt(np.mean(q * q)))
    v_rms = float(np.sqrt(np.mean(v * v)))
    startup_rms = float(np.sqrt(np.mean(startup * startup)))
    startup_max = float(np.max(np.abs(startup)))
    direct = {"q_rms": q_rms, "velocity_rms": v_rms,
        "startup_q_rms": startup_rms, "startup_q_max": startup_max,
        "q_nonworse": bool(q_rms <= problem["command_limit"]),
        "velocity_nonworse": bool(v_rms <= problem["velocity_limit"]),
        "startup_rms_pass": bool(startup_rms <= .018),
        "startup_max_pass": bool(startup_max <= .04),
        "objective_direct": float((q / problem["normalizer"]) @
            (q / problem["normalizer"]) + RIDGE * np.mean(x * x))}
    report["direct"] = direct
    report["direct_training_pass"] = bool(all(direct[key] for key in (
        "q_nonworse", "velocity_nonworse", "startup_rms_pass", "startup_max_pass")))
    report["objective_pass"] = bool(report["diagnostics_finite"] and abs(
        direct["objective_direct"] -
        (info["objective_primal"] + problem["objective_constant"])) <= 1e-6)
    report["numerical_pass"] = bool(report["numerical_pass"] and report["objective_pass"])
    return report


def proxy_verdict(metrics, baseline):
    components = {"global_command_20_percent": bool(
            metrics["command_rms"] <= .8 * .056686385349786386),
        "global_velocity_nonworse": bool(metrics["velocity_rms"] <= baseline["velocity_rms"]),
        "leg_command_nonworse": [bool(a <= b) for a, b in zip(
            metrics["leg_command_rms"], baseline["leg_command_rms"])],
        "leg_velocity_nonworse": [bool(a <= b) for a, b in zip(
            metrics["leg_velocity_rms"], baseline["leg_velocity_rms"])],
        "startup_q_rms": bool(metrics["startup_q_rms"] <= .018),
        "startup_q_max": bool(metrics["startup_q_max"] <= .04),
        "startup_tendon_rms": bool(metrics["startup_tendon_rms"] <= .009),
        "startup_tendon_max": bool(metrics["startup_tendon_max"] <= .025)}
    passed = all(all(value) if isinstance(value, list) else value
                 for value in components.values())
    return {"passed": bool(passed), "components": components,
        "conditional_proxy_only": True, "candidate_adopted": False}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e210.output-manifest.v1", "files": files,
        "total_bytes": sum(value["bytes"] for value in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E210 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = review = paths = None
    primary = None
    errors = []
    leg_results = {}
    leg_reports = {}
    ledger = {"schema": "flylab.m3.e210.ledger.v1", "solver_calls": 0,
        "solver_wall_s": 0., "constructor_wall_s": 0.,
        "full_banc_calls": 0, "cuda_calls": 0,
        "physics_steps": 0, "retries": 0, "legs": {}}
    result = {"schema": "flylab.m3.e210.causal-neural-basis-fit.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "conditional_proxy_only": True}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E210 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
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
            with (np.load(paths["affine"], allow_pickle=False) as data,
                  np.load(paths["basis"], allow_pickle=False) as response,
                  np.load(paths["basis_parameters"], allow_pickle=False) as params,
                  np.load(paths["endpoints"], allow_pickle=False) as native,
                  np.load(paths["mask"], allow_pickle=False) as mask):
                D = data["D"].copy()
                q0 = data["q_intercept"].copy()
                H = response["H"].copy()
                B = response["B"].copy()
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
                            start["adapter"]["parameters"]["membership"])):
                    raise ValueError("E210 frozen basis/mask anatomy changed")
            membership, moment, motors_by_leg = leg_anatomy(start, free)
            if not np.array_equal(D[:, fixed], np.zeros_like(D[:, fixed])):
                raise ValueError("E210 fixed LTM D must be zero")
            for leg_index, leg in enumerate(LEGS):
                leg_dir = out / ("leg-" + leg)
                leg_dir.mkdir()
                entry = {"status": "not_started", "solve_requested": False,
                    "constructor_entered": False, "constructor_completed": False,
                    "solve_called": False, "solver_returned": False}
                ledger["legs"][leg] = entry
                if (elapsed() + RESOURCE["solver_per_leg_s"] > RESOURCE["internal_s"]
                        or ledger["solver_wall_s"] >= RESOURCE["solver_total_s"]):
                    entry["status"] = "blocked_by_deadline"
                    save_json(out / "ledger.json", ledger)
                    continue
                try:
                    problem = build_leg_problem(start, D, q0, target,
                        motors_by_leg, leg_index)
                    save_problem(leg_dir, problem, freeze["input_sha256"][
                        freeze["paths"]["affine"]])
                    try:
                        ensure_leg_budget(elapsed)
                    except TimeoutError:
                        entry["status"] = "blocked_after_preparation"
                        save_json(out / "ledger.json", ledger)
                        continue
                    entry["status"] = "solver_requested"
                    entry["solve_requested"] = True
                    ledger["requested_leg"] = leg
                    save_json(out / "ledger.json", ledger)
                    raw, info = solve_leg_once(problem, leg_dir, ledger,
                        entry=entry, remaining_guard=lambda: ensure_leg_budget(elapsed))
                    entry["status"] = info["status"]
                    report = validate_solver_return(problem, raw, info)
                    save_json(leg_dir / "direct-metrics.json", report)
                    entry["assessment"] = report
                    leg_reports[leg] = report
                    if report["eligible_beta"]:
                        entry["status"] = "solved_bounded_coefficients"
                        leg_results[leg] = {"motors": motors_by_leg[leg_index],
                                            "beta": raw["solver_x"].copy()}
                    else:
                        entry["status"] = "bounded_unresolved_no_coefficients"
                except TimeoutError as error:
                    entry["status"] = "blocked_after_constructor" if (
                        entry["constructor_completed"] and not entry["solve_called"]
                    ) else "blocked_by_deadline"
                    entry["error"] = {"type": type(error).__name__, "message": str(error)}
                except BaseException as error:  # noqa: BLE001 - keep independent legs
                    entry["status"] = "failed_or_unresolved"
                    entry["error"] = {"type": type(error).__name__, "message": str(error)}
                    errors.append({"stage": leg, "type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc()})
                    if primary is None:
                        primary = error
                finally:
                    _save_attempt(errors, "leg-ledger", lambda: save_json(
                        out / "ledger.json", ledger))
                    if "problem" in locals():
                        del problem
            if len(leg_results) == 6:
                beta = np.zeros((391, 16), np.float64)
                for leg in LEGS:
                    record = leg_results[leg]
                    beta[record["motors"]] = record["beta"].reshape(-1, 16)
                rate = H + cap[None, :] * np.sum(B * beta[None], axis=2)
                rate[:, fixed] = original_rates[:, fixed]
                direct = canonical_readout(start, rate)
                q_affine = q0.copy()
                for leg_index, motors in enumerate(motors_by_leg):
                    joint = physical_joint_matrix(D, membership, moment, motors,
                                                  leg_index)
                    q_affine[:, 7 * leg_index:7 * leg_index + 7] += (
                        joint @ beta[motors].reshape(-1))
                np.savez(out / "candidate.npz", beta=beta, rate=rate,
                    q_affine=q_affine, q_direct=direct["command"],
                    tendon=direct["tendon"], activation=direct["activation"])
                canonical_original = canonical_readout(start, original_rates)
                ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
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
                affine_exact = bool(np.allclose(q_affine, direct["command"],
                                   atol=2e-12, rtol=2e-12))
                ltm_exact = bool(np.array_equal(
                    canonical_original["activation"][:, ltm],
                    direct["activation"][:, ltm]) and np.array_equal(
                    canonical_original["tendon"], direct["tendon"]))
                metrics = direct_metrics(direct, target)
                verdict = proxy_verdict(metrics, target["baseline"])
                leg_constraint_pass = bool(all(
                    leg_reports[leg]["numerical_pass"] and
                    leg_reports[leg]["direct_training_pass"] for leg in LEGS))
                verdict["leg_constraint_pass"] = leg_constraint_pass
                verdict["affine_exact"] = affine_exact
                verdict["ltm_exact"] = ltm_exact
                verdict["passed"] = bool(verdict["passed"] and leg_constraint_pass
                    and affine_exact and ltm_exact)
                native_q = np.asarray([row["joint_targets_rad"] for row in trace["samples"]])
                result.update(validity="valid_conditional_fit_rejected_or_supported",
                    metrics=metrics, baseline=target["baseline"],
                    proxy_verdict=verdict,
                    original_monotone_native_q_max_abs=float(np.max(np.abs(
                        old_mono["command"] - native_q))))
            else:
                result["validity"] = "valid_bounded_unresolved_no_complete_fit"
    except BaseException as error:  # noqa: BLE001 - preserve raw and primary
        if primary is None:
            primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    ledger["completed_leg_count"] = len(leg_results)
    _attempt = lambda name, fn: _save_attempt(errors, name, fn)
    _attempt("ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"], solver_wall_s=ledger["solver_wall_s"],
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0) + elapsed(),
        retries=0)
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    _attempt("execution-errors.json", lambda: save_json(out / "execution-errors.json", errors))
    _attempt("result-preliminary", lambda: save_json(out / "result.json",
        dict(result, validity="incomplete_or_invalid", finalization_pending=True)))
    manifest = _attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E210 output manifest missing or over 768 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"] or ledger["solver_wall_s"] > RESOURCE["solver_total_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E210 internal or total solver deadline exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors or primary is not None:
        result["validity"] = "incomplete_or_invalid"
    result["finalization_pending"] = False
    _attempt("result-final", lambda: save_json(out / "result.json", result))
    if errors and primary is None:
        primary = RuntimeError("E210 raw/publication failure")
    if errors:
        _attempt("publication-failure.json", lambda: save_json(
            out / "publication-failure.json", errors))
    if primary is not None:
        raise primary
    return result


def _save_attempt(errors, name, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve first failure
        errors.append({"stage": name, "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review,
        args.review_sha256, args.out)


if __name__ == "__main__":
    main()
