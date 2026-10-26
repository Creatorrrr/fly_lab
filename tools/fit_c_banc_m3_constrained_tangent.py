"""E206 one CPU Clarabel solve on frozen E205 local command tangents."""

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

from tools.fit_c_banc_m3_command_conditioned_response import command_diagonal_scale
from tools.fit_c_banc_m3_local_response import (
    LEGS,
    direct_metrics,
    feasible,
    load_targets,
)
from tools.verify_c_banc_m3_local_response import file_sha, repository_path, write_json

FREEZE_SHA = "ebb917e0ab7c0882e3f2c604e29d06b6a45a7179cd7aba4d4dd67d636334434c"
FREEZE_SCHEMA = "flylab.m3.e206.constrained-tangent.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e206.constrained-tangent.review.v1"
OWNED = ("tools/fit_c_banc_m3_constrained_tangent.py",
         "tests/test_c_banc_constrained_tangent.py")
PAD = 1 - 1e-6
N = 2401


def compress_norm(matrix, offset):
    """Retain every positive augmented-Gram eigenvalue, including weak columns."""
    matrix = np.asarray(matrix, np.float64)
    offset = np.asarray(offset, np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(offset):
        raise ValueError("E206 norm affine dimensions differ")
    augmented = np.column_stack((matrix, offset))
    gram = augmented.T @ augmented
    values, vectors = np.linalg.eigh(gram)
    max_eigen = float(np.max(values))
    minimum = float(np.min(values))
    if minimum < -1e-10 * max(1., max_eigen):
        raise ValueError("E206 augmented Gram has material negative eigenvalue")
    clipped = np.maximum(values, 0.)
    factor = np.sqrt(clipped)[:, None] * vectors.T
    gram_norm = float(np.linalg.norm(gram))
    residual_norm = float(np.linalg.norm(factor.T @ factor - gram))
    error = residual_norm / gram_norm if gram_norm else (0. if residual_norm == 0 else np.inf)
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError("E206 norm compression changed its Gram")
    return factor[:, :-1], factor[:, -1], {
        "minimum_eigenvalue": minimum, "maximum_eigenvalue": max_eigen,
        "positive_eigenvalues": int(np.count_nonzero(values > 0)),
        "dimension": len(values), "relative_frobenius_error": error}


def soc_rows(matrix, offset, limit, width=N):
    """Encode ||matrix*u+offset||<=limit as A*u+s=b, s in SOC."""
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    offset = np.asarray(offset, np.float64).reshape(-1)
    if (matrix.shape != (len(offset), width) or not np.isfinite(limit)
            or limit <= 0 or not np.isfinite(offset).all()
            or not np.isfinite(matrix.data).all()):
        raise ValueError("E206 SOC affine shape/finite limit invalid")
    rows = sparse.vstack((sparse.csr_matrix((1, width)), -matrix), format="csr")
    return rows, np.r_[limit, offset]


def embed_local(matrix, columns, width=N):
    local = sparse.csr_matrix(matrix, dtype=np.float64)
    selector = sparse.csr_matrix((np.ones(len(columns)),
        (np.arange(len(columns)), columns)), shape=(len(columns), width))
    return local @ selector


def csc_arrays(name, matrix):
    value = matrix.tocsc()
    return {f"{name}_data": value.data, f"{name}_indices": value.indices,
            f"{name}_indptr": value.indptr, f"{name}_shape": np.asarray(value.shape)}


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E206 SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E206 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design") != "docs/BANC_M3_E206_CONSTRAINED_TANGENT_DESIGN_20260924.md"
            or file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256")
            or file_sha(repository_path(freeze["base_freeze"])) != freeze.get("base_freeze_sha256")
            or freeze.get("base_freeze_sha256") != "583b3948a508117f421a7010b33092f83ee8e19d5f795009a057e92b9722561c"
            or freeze.get("resource_limits", {}).get("solver_calls") != 1
            or freeze.get("resource_limits", {}).get("internal_s") != 100
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E206 frozen contract changed")
    for label, count in (("input_sha256", 134),
                         ("reused_source_sha256", 14),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("Frozen E206 hash count changed")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E206 frozen input/source changed: " + name)
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256")
    if extra is not None and (not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != sha for name, sha in extra.items())):
        raise ValueError("Additional E206 reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != own):
        raise ValueError("E206 parent source review does not authorize these files")
    return freeze, review


def build_problem(start, d, scale, baseline_weights, decoded, target, free):
    """Build the one exact local convex problem, retaining leg-local raw maps."""
    d = np.asarray(d, np.float64)
    scale = np.asarray(scale, np.float64)
    free = np.asarray(free, np.int64)
    weights = np.asarray(baseline_weights, np.float64)
    q0 = np.asarray(decoded["command"], np.float64)
    a = start["adapter"]
    membership = np.asarray(a["parameters"]["membership"], np.int64)
    moment = np.asarray(a["calibration"]["moment"], np.float64)
    groups = a["groups"]
    if (d.shape != (2001, 391, 7) or scale.shape != (343, 7)
            or weights.shape != (391, 7) or free.shape != (343,)
            or q0.shape != (2001, 42) or membership.shape != (391,)
            or moment.shape != (42, 96) or not np.isfinite(d).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(q0).all() or np.any(d[0] != 0)
            or not np.array_equal(np.sort(free), free)):
        raise ValueError("E206 frozen tangent/anatomical shape changed")
    muscle_leg = [groups[g]["leg"] for g in range(96)]
    if any(leg not in LEGS for leg in muscle_leg):
        raise ValueError("E206 muscle without a leg")
    for muscle, leg in enumerate(muscle_leg):
        other = np.r_[0:7 * LEGS.index(leg), 7 * (LEGS.index(leg) + 1):42]
        if np.any(moment[other, muscle] != 0):
            raise ValueError("E206 moment crosses an anatomical leg")
    flat_scale = scale.reshape(-1)
    flat_weight = weights[free].reshape(-1)
    lower = (-1 - flat_weight) / flat_scale
    upper = (1 - flat_weight) / flat_scale
    if np.any(lower > 0) or np.any(upper < 0) or np.any(weights[np.setdiff1d(
            np.arange(391), free)] != 0):
        raise ValueError("E206 original coefficient bounds violated")

    indices = np.asarray(target["indices"], np.int64)
    starts = np.asarray(target["pair_start"], np.int64)
    ends = np.asarray(target["pair_end"], np.int64)
    if (indices.shape != (1824,) or starts.shape != (1823,)
            or ends.shape != (1823,) or not np.array_equal(ends, starts + 1)):
        raise ValueError("E206 frozen target/velocity rows changed")
    eq = q0[indices] - target["target"]
    ev = (q0[ends] - q0[starts]) / .005 - target["velocity"]
    es = q0[1:100] - target["startup_command"]
    baseline = target["baseline"]
    denominator = 1824 * 42 * (.8 * baseline["command_rms"])**2
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("E206 original command baseline invalid")
    p_dense = np.eye(N) * 2e-8
    linear = np.zeros(N)
    norm_rows = []
    norm_rhs = []
    norm_records = {}
    leg_columns = {}
    startup_data, startup_rows, startup_columns = [], [], []
    for leg_index, leg in enumerate(LEGS):
        motors = np.array([m for m in free if muscle_leg[membership[m]] == leg],
                          dtype=np.int64)
        positions = np.searchsorted(free, motors)
        columns = (positions[:, None] * 7 + np.arange(7)).reshape(-1)
        leg_columns[leg] = columns
        if not len(columns):
            raise ValueError("E206 empty anatomical leg")
        gains = .02 * np.repeat(moment[7 * leg_index:7 * leg_index + 7,
                                          membership[motors]], 7, axis=1)
        tangent = (d[:, motors] * scale[positions][None]).reshape(2001, -1)
        joint = tangent[:, None, :] * gains[None, :, :]
        jq = joint[indices].reshape(1824 * 7, -1)
        e_leg = eq[:, 7 * leg_index:7 * leg_index + 7].reshape(-1)
        p_dense[np.ix_(columns, columns)] += 2 * (jq.T @ jq) / denominator
        linear[columns] += 2 * (jq.T @ e_leg) / denominator

        for kind, raw_matrix, raw_offset, limit in (
            ("command", jq, e_leg, baseline["leg_command_rms"][leg_index]),
            ("velocity", ((joint[ends] - joint[starts]) / .005).reshape(
                1823 * 7, -1), ev[:, 7 * leg_index:7 * leg_index + 7].reshape(-1),
                baseline["leg_velocity_rms"][leg_index])):
            normalizer = np.sqrt(len(raw_offset)) * limit
            factor, offset, note = compress_norm(
                raw_matrix / normalizer, raw_offset / normalizer)
            embedded = embed_local(factor, columns)
            rows, rhs = soc_rows(embedded, offset, PAD)
            norm_rows.append(rows)
            norm_rhs.append(rhs)
            norm_records[f"{leg}_{kind}"] = {"factor": factor,
                "offset": offset, "columns": columns, "normalizer": normalizer,
                "limit": limit, "gram": note}

        local_start = joint[1:100].reshape(99 * 7, -1)
        nonzero = sparse.coo_matrix(local_start)
        row_map = (np.arange(99)[:, None] * 42
                   + np.arange(7 * leg_index, 7 * leg_index + 7)[None, :]).reshape(-1)
        startup_data.append(nonzero.data)
        startup_rows.append(row_map[nonzero.row])
        startup_columns.append(columns[nonzero.col])
        del joint, jq, tangent

    startup = sparse.coo_matrix((np.concatenate(startup_data),
        (np.concatenate(startup_rows), np.concatenate(startup_columns))),
        shape=(99 * 42, N)).tocsr()
    startup_offset = es.reshape(-1)
    identity = sparse.eye(N, format="csr")
    max_limit = .04 * PAD
    nonnegative = sparse.vstack((identity, -identity, startup, -startup),
                                format="csr")
    nonnegative_rhs = np.r_[upper, -lower, max_limit - startup_offset,
                            max_limit + startup_offset]
    if not np.isfinite(nonnegative_rhs).all():
        raise ValueError("E206 nonnegative cone rhs invalid")
    startup_soc, startup_soc_rhs = soc_rows(
        startup / np.sqrt(99 * 42), startup_offset / np.sqrt(99 * 42),
        .018 * PAD)
    trust_soc, trust_rhs = soc_rows(identity, np.zeros(N), PAD)
    A = sparse.vstack([nonnegative, *norm_rows, startup_soc, trust_soc],
                      format="csc")
    b = np.concatenate([nonnegative_rhs, *norm_rhs, startup_soc_rhs,
                        trust_rhs])
    cone_dimensions = np.array([len(nonnegative_rhs),
        *(len(v) for v in norm_rhs), len(startup_soc_rhs), len(trust_rhs)],
        dtype=np.int64)
    cone_types = ["NonnegativeConeT", *["SecondOrderConeT"] * 14]
    cones = [clarabel.NonnegativeConeT(int(cone_dimensions[0])),
        *[clarabel.SecondOrderConeT(int(v)) for v in cone_dimensions[1:]]]
    p_upper = sparse.triu(sparse.csc_matrix(p_dense), format="csc")
    objective_constant = float(np.sum(eq * eq) / denominator)
    if (len(norm_rhs) != 12 or len(cones) != 15 or A.shape[0] != len(b)
            or not np.isfinite(p_upper.data).all()
            or not np.isfinite(A.data).all() or not np.isfinite(linear).all()):
        raise ValueError("E206 conic matrix contract invalid")
    return {"P": p_upper, "linear": linear, "A": A, "b": b,
        "cones": cones, "cone_types": cone_types,
        "cone_dimensions": cone_dimensions, "objective_constant": objective_constant,
        "D": d, "scale": scale, "baseline_weights": weights,
        "free_columns": free, "leg_columns": leg_columns,
        "target_residual": eq, "velocity_residual": ev,
        "startup_residual": es, "startup_matrix": startup,
        "norm_records": norm_records, "lower": lower, "upper": upper}


def predicted_command(problem, start, baseline_command, u):
    u = np.asarray(u, np.float64)
    delta_free = problem["scale"].reshape(-1) * u
    delta_weights = np.zeros((391, 7), np.float64)
    delta_weights[problem["free_columns"]] = delta_free.reshape(343, 7)
    membership = np.asarray(start["adapter"]["parameters"]["membership"], np.int64)
    moment = np.asarray(start["adapter"]["calibration"]["moment"], np.float64)
    motor_activation = np.sum(problem["D"] * delta_weights[None], axis=2)
    activation = np.zeros((2001, 96), np.float64)
    for muscle in range(96):
        activation[:, muscle] = np.sum(motor_activation[:, membership == muscle],
                                       axis=1)
    predicted = np.asarray(baseline_command, np.float64) + .02 * activation @ moment.T
    if not np.array_equal(predicted[0], baseline_command[0]):
        raise ValueError("E206 initial command changed")
    return predicted, delta_weights


def raw_constraint_report(problem, metrics, u, weights, target):
    baseline = target["baseline"]
    component = {
        "coefficient_bounds": bool(np.all(np.abs(weights) <= 1)),
        "trust_norm": bool(np.linalg.norm(u) <= 1),
        "startup_q_rms": bool(metrics["startup_q_rms"] <= .018),
        "startup_q_max": bool(metrics["startup_q_max"] <= .04),
        "startup_tendon_rms": bool(metrics["startup_tendon_rms"] <= .009),
        "startup_tendon_max": bool(metrics["startup_tendon_max"] <= .025),
        "overall_velocity": bool(metrics["velocity_rms"] <= baseline["velocity_rms"]),
        "leg_command": [bool(v <= b) for v, b in zip(
            metrics["leg_command_rms"], baseline["leg_command_rms"])],
        "leg_velocity": [bool(v <= b) for v, b in zip(
            metrics["leg_velocity_rms"], baseline["leg_velocity_rms"])],
    }
    passed = bool(all(v if isinstance(v, bool) else all(v)
                      for v in component.values()))
    return {"passed": passed, "components": component,
        "command_20_percent_prediction": bool(
            metrics["command_rms"] <= .8 * baseline["command_rms"]),
        "clarabel_padding": PAD,
        "original_limits_used_for_direct_check": True}


def save_problem(out, problem):
    arrays = {"D": problem["D"], "scale": problem["scale"],
        "baseline_weights": problem["baseline_weights"],
        "free_columns": problem["free_columns"],
        "linear": problem["linear"], "b": problem["b"],
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "objective_constant": np.asarray(problem["objective_constant"]),
        "target_residual": problem["target_residual"],
        "velocity_residual": problem["velocity_residual"],
        "startup_residual": problem["startup_residual"],
        "lower": problem["lower"], "upper": problem["upper"]}
    arrays.update(csc_arrays("P", problem["P"]))
    arrays.update(csc_arrays("A", problem["A"]))
    arrays.update(csc_arrays("startup_matrix", problem["startup_matrix"]))
    np.savez_compressed(Path(out) / "pre-solve.npz", **arrays)
    factors = {}
    notes = {}
    for name, record in problem["norm_records"].items():
        factors[name + "_factor"] = record["factor"]
        factors[name + "_offset"] = record["offset"]
        factors[name + "_columns"] = record["columns"]
        notes[name] = {"normalizer": record["normalizer"],
            "limit": record["limit"], **record["gram"]}
    np.savez_compressed(Path(out) / "norm-factors.npz", **factors)
    write_json(Path(out) / "norm-factors.json", notes)
    write_json(Path(out) / "leg-columns.json", {
        leg: columns.tolist() for leg, columns in problem["leg_columns"].items()})


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e206.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


def validate_solver_return(raw, info, constraint_rows, variables=N):
    """Validate every returned primal, slack, dual and diagnostic after raw saving."""
    for name, shape in (("solver_x", (variables,)),
                        ("solver_s", (constraint_rows,)),
                        ("solver_z", (constraint_rows,))):
        value = raw.get(name)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError("E206 invalid saved Clarabel return: " + name)
    if (info.get("status") not in {"Solved", "AlmostSolved", "MaxIterations"}
            or not isinstance(info.get("iterations"), int)
            or info["iterations"] < 0
            or any(not np.isfinite(info.get(key, np.nan)) for key in (
                "objective_primal", "objective_dual", "primal_residual",
                "dual_residual", "solve_time_s"))):
        raise ValueError("E206 invalid saved Clarabel status or diagnostics")


def json_safe_diagnostics(info):
    return {key: (str(value) if isinstance(value, float)
                  and not np.isfinite(value) else value)
            for key, value in info.items()}


def solve_once(problem, settings_contract, out, ledger):
    """Construct and invoke one Clarabel solver, saving its return immediately."""
    settings = clarabel.DefaultSettings()
    for name, value in settings_contract.items():
        if name in {"name", "version", "numpy_version", "scipy_version"}:
            continue
        setattr(settings, name, value)
    history = []
    ledger["solver_constructor_entered"] = True
    solver = clarabel.DefaultSolver(problem["P"], problem["linear"],
        problem["A"], problem["b"], problem["cones"], settings)
    ledger["solver_constructed"] = True
    def callback(info):
        history.append({"iterations": int(info.iterations),
            "res_primal": float(info.res_primal), "res_dual": float(info.res_dual),
            "solve_time": float(info.solve_time), "status": str(info.status)})
        return False
    solver.set_termination_callback(callback)
    ledger["solver_calls"] += 1
    ledger["solver_solve_entered"] = True
    try:
        solution = solver.solve()
    except BaseException:
        try:
            write_json(Path(out) / "solver-history.json", history)
        except BaseException as storage_error:  # noqa: BLE001
            ledger["errors"].append({"stage": "solver-history",
                "type": type(storage_error).__name__, "message": str(storage_error)})
        raise
    ledger["solver_returned"] = True
    raw = {"solver_x": np.asarray(solution.x, np.float64),
        "solver_s": np.asarray(solution.s, np.float64),
        "solver_z": np.asarray(solution.z, np.float64)}
    np.savez_compressed(Path(out) / "solver-raw.npz", **raw)
    info = {"status": str(solution.status), "iterations": int(solution.iterations),
        "objective_primal": float(solution.obj_val),
        "objective_dual": float(solution.obj_val_dual),
        "primal_residual": float(solution.r_prim),
        "dual_residual": float(solution.r_dual),
        "solve_time_s": float(solution.solve_time)}
    write_json(Path(out) / "solver-raw.json", json_safe_diagnostics(info))
    write_json(Path(out) / "solver-history.json", history)
    return raw, info


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E206 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e206.ledger.v1", "solver_calls": 0,
        "solver_constructor_entered": False, "solver_constructed": False,
        "solver_solve_entered": False, "solver_returned": False,
        "new_neural_ticks": 0, "new_physics_steps": 0,
        "new_cuda_evaluations": 0, "errors": []}
    result = {"schema": "flylab.m3.e206.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "scope": "linearized_saved_E205_tangent_only"}
    primary = None
    freeze = None

    def elapsed():
        return time.perf_counter() - began

    def attempt(label, operation):
        nonlocal primary
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - preserve first failure
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
            raise OSError("E206 requires 4 GiB free disk")
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
            start = StateStore.load(paths["start"])
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target = load_targets(paths["targets"], trace)
            with np.load(paths["mask"], allow_pickle=False) as data:
                free = data["free_columns"].copy()
                fixed = data["fixed_columns"].copy()
            with np.load(paths["e205_response"], allow_pickle=False) as data:
                response = {key: data[key].copy() for key in (
                    "rate", "adaptation", "rate_sensitivity",
                    "adaptation_sensitivity")}
                baseline_weights = data["weights"].copy()
            with np.load(paths["e205_decoded"], allow_pickle=False) as data:
                decoded = {key: data[key].copy() for key in (
                    "bank", "activation", "command", "tendon")}
            with np.load(paths["e205_preconditioner"], allow_pickle=False) as data:
                scale = data["scale"].copy()
                preconditioner_free = data["free_columns"].copy()
            with np.load(paths["e205_selected_raw"], allow_pickle=False) as data:
                selected_weights = data["weights"].copy()
                selected_free = data["free_weights"].copy()
            if (free.shape != (343,) or fixed.shape != (48,)
                    or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or not np.array_equal(free, np.sort(free))
                    or not np.array_equal(free, preconditioner_free)
                    or scale.shape != (343, 7)
                    or not np.array_equal(np.exp2(np.log2(scale)), scale)
                    or not np.array_equal(baseline_weights, selected_weights)
                    or not np.array_equal(baseline_weights[free].reshape(-1), selected_free)
                    or np.any(baseline_weights[fixed] != 0)):
                raise ValueError("E206 saved E205 variable/mask/scale mismatch")
            tangent = command_diagonal_scale(start, response, decoded, target, free)
            problem = build_problem(start, tangent["D"], scale,
                                    baseline_weights, decoded, target, free)
            save_problem(out, problem)
            write_json(out / "solver-settings.json", freeze["solver"])
            ledger["pre_solve_saved"] = True
            if elapsed() >= 100:
                raise TimeoutError("E206 internal 100 s deadline before solve")
            raw, info = solve_once(problem, freeze["solver"], out, ledger)
            validate_solver_return(raw, info, len(problem["b"]))
            u = raw["solver_x"]
            command, delta = predicted_command(problem, start, decoded["command"], u)
            weights = baseline_weights + delta
            tendon = np.asarray(decoded["tendon"], np.float64).copy()
            np.savez_compressed(out / "linearized-candidate.npz",
                u=u, delta_weights=delta, candidate_weights=weights,
                predicted_command=command, predicted_tendon=tendon)
            metrics = direct_metrics({"command": command, "tendon": tendon}, target)
            constraints = raw_constraint_report(problem, metrics, u, weights, target)
            write_json(out / "direct-check.json", {"metrics": metrics,
                "constraints": constraints})
            result["solver_termination"] = info["status"]
            result["raw_constraint_pass"] = constraints["passed"]
            result["command_20_percent_prediction"] = constraints[
                "command_20_percent_prediction"]
            result["metrics"] = metrics
            result["validity"] = ("supported_linearized_direction"
                if constraints["passed"] and constraints["command_20_percent_prediction"]
                and feasible(metrics, target["baseline"])
                else "bounded_linearized_search_no_proposal")
    except BaseException as error:  # noqa: BLE001 - raw outlives failure
        primary = error
        ledger["errors"].append({"stage": "run", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    ledger["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    try:
        import psutil
        memory = psutil.Process().memory_info()
        ledger["peak_resident_bytes"] = int(getattr(memory, "peak_wset", memory.rss))
        if ledger["peak_resident_bytes"] > 4 * 1024**3:
            primary = primary or MemoryError("E206 4 GiB resident cap exceeded")
    except ImportError:
        ledger["peak_resident_bytes"] = None
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    result["process_wall_s"] = elapsed()
    result["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    result["solver_calls"] = ledger["solver_calls"]
    attempt("ledger.json", lambda: write_json(out / "ledger.json", ledger))
    preliminary = {**result, "validity": "incomplete_or_invalid",
                   "finalization_pending": True}
    attempt("result.json-preliminary", lambda: write_json(out / "result.json", preliminary))
    manifest = attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > 256 * 1024**2:
        primary = primary or RuntimeError("E206 output manifest/256 MiB limit failed")
    if elapsed() >= 100:
        primary = primary or TimeoutError("E206 internal 100 s deadline exceeded")
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
