"""E218: one bounded six-leg fit of the E217 muscle-shared 64-column basis.

The saved signed input and conditional basis are fixed. This builds original
physical command constraints; it runs no BANC, CUDA, motor, or body model.
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
from importlib.metadata import version
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_c_banc_m3_dynamic_capacity import (
    SolverAllowanceUnavailable,
    ensure_solve_budget,
    solve_once,
)
from tools.fit_c_banc_m3_causal_neural_basis import (
    box_and_startup,
    compressed_soc,
    proxy_verdict,
    quadratic_objective,
)
from tools.fit_c_banc_m3_constrained_tangent import csc_arrays, soc_rows
from tools.fit_c_banc_m3_global_startup import (
    assemble_global,
    assess_solution,
    physical_limit_diagnostics,
)
from tools.fit_c_banc_m3_local_response import direct_metrics, load_targets
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "170bd713a4b01263359bd57dfcc10a8659618dfa10e41115941063b0b3b2fc11"
FREEZE_SCHEMA = "flylab.m3.e218.grouped-response-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e218.grouped-response-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_grouped_response.py",
         "tests/test_c_banc_grouped_response_fit.py")
LEG_GROUPS = (16, 14, 15, 16, 14, 15)
LEG_WIDTHS = tuple(v * 64 for v in LEG_GROUPS)
PAD = 1 - 1e-6
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 1, "retries": 0, "internal_s": 150,
    "solver_s": 90, "external_s": 180, "supervisor_s": 200,
    "resident_bytes": 8589934592, "output_bytes": 805306368,
    "minimum_free_bytes": 4294967296}


def repository_path(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E218 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E218 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E218 frozen file missing or escaped")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E218 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E218 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    shape = {"variables": 5760, "motors": 391, "free_motors": 343,
        "fixed_motors": 48, "muscles": 96, "free_muscles": 90,
        "basis_count": 64, "samples": 2001, "startup_rows": 99,
        "global_startup_values": 4158, "leg_groups": list(LEG_GROUPS),
        "leg_variables": list(LEG_WIDTHS), "cones": 19}
    solver = {"name": "clarabel", "max_threads": 1, "max_iter": 150,
        "time_limit_s": 90.0, "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8, "tol_feas": 1e-8,
        "coordinate": "physical shared group beta in six leg blocks",
        "objective": "sum normalized per-leg command objectives plus 1e-8 mean leg beta squared",
        "beta_min": 0.0, "beta_max": 1.0,
        "ridge_mean_beta2": 1e-8, "PAD": PAD,
        "raw_clip": False, "weak_column_pruning": False,
        "converged_status": "Solved"}
    proxy = {"global_command_ratio": .8, "leg_command_ratio": 1.,
        "leg_velocity_ratio": 1., "global_velocity_ratio": 1.,
        "startup_q_rms": .018, "startup_q_max": .04,
        "startup_tendon_rms": .009, "startup_tendon_max": .025,
        "original_command_rms": .056686385349786386,
        "training_extra": "global startup norm with unchanged PAD; no per-leg startup RMS gate"}
    tolerance = {"affine_q_atol": 2e-10, "affine_q_rtol": 2e-12,
        "gram_relative_frobenius_max": 1e-10,
        "gram_min_eigen_relative": -1e-10,
        "numerical_residual_max": 1e-6, "objective_abs_max": 1e-6}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E218_GROUPED_RESPONSE_FIT_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != shape or freeze.get("solver") != solver
            or freeze.get("proxy") != proxy
            or freeze.get("tolerance") != tolerance
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("clock") != {"neural_dt_s": .00025,
                "control_dt_s": .005, "ticks_per_control": 20,
                "activation_tau_s": .01}
            or freeze.get("local_solver_calls") != 0
            or freeze.get("generated_solver") != {"actual_calls": 0,
                "reuse": "E215 solve_once generated analytic proof pinned above"}
            or any(freeze.get(key) != 0 for key in (
                "new_cuda", "new_neural", "new_physics"))
            or any(freeze.get(key) is not False for key in (
                "F6_exposed", "candidate_adopted", "whole_goal_complete"))):
        raise ValueError("E218 scientific/resource contract changed")
    if freeze.get("packages") != {name: version(name) for name in (
            "numpy", "scipy", "clarabel")}:
        raise ValueError("E218 frozen numerical package versions changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E218 design changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E218 additional source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E218 parent source review invalid")
    for label, count in (("input_sha256", 20),
                         ("reused_source_sha256", 35),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E218 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E218 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E218 hash gate exceeded internal deadline")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "parameters", "endpoints", "mask", "targets", "trace",
            "basis", "affine", "geometry", "prepared_parameters"}:
        raise ValueError("E218 frozen paths changed")
    return freeze, review, {key: repository_path(value)
                            for key, value in paths.items()}


def group_anatomy(membership, pool, moment, ltm, groups, free, fixed):
    """Return free original muscle columns per leg, never MN columns."""
    membership = np.asarray(membership, np.int64)
    pool = np.asarray(pool, np.float64)
    moment = np.asarray(moment, np.float64)
    ltm = np.asarray(ltm, np.int64)
    free = np.asarray(free, np.int64)
    fixed = np.asarray(fixed, np.int64)
    if (membership.shape != (391,) or pool.shape != (96, 391)
            or moment.shape != (42, 96) or ltm.shape != (6,)
            or len(groups) != 96 or free.shape != (343,)
            or fixed.shape != (48,)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.array_equal(fixed,
                np.flatnonzero(np.isin(membership, ltm)))
            or np.any((membership < 0) | (membership >= 96))
            or not np.isfinite(pool).all() or not np.isfinite(moment).all()
            or np.any(pool < 0)):
        raise ValueError("E218 original muscle/MN ownership invalid")
    for m, owner in enumerate(membership):
        nonzero = np.flatnonzero(pool[:, m])
        if len(nonzero) != 1 or nonzero[0] != owner:
            raise ValueError("E218 original single-muscle pool changed")
    muscle_leg = [record["leg"] for record in groups]
    if any(leg not in LEGS for leg in muscle_leg):
        raise ValueError("E218 muscle without anatomical leg")
    for g, leg in enumerate(muscle_leg):
        index = LEGS.index(leg)
        other = np.r_[0:7*index, 7*(index+1):42]
        if np.any(moment[other, g] != 0):
            raise ValueError("E218 cross-leg anatomical moment")
    free_groups = np.unique(membership[free])
    if (len(free_groups) != 90 or not np.array_equal(
            np.unique(membership[fixed]), np.sort(ltm))
            or len(np.unique(ltm)) != 6):
        raise ValueError("E218 free90/fixed6 muscle set changed")
    by_leg = [np.asarray([g for g in free_groups if muscle_leg[g] == leg],
                         np.int64) for leg in LEGS]
    if tuple(len(v) for v in by_leg) != LEG_GROUPS:
        raise ValueError("E218 per-leg group width changed")
    return by_leg


def group_joint_matrix(D_group, moment, group_columns, leg_index):
    """Physical 7-joint Jacobian for 64 coefficients per original muscle."""
    D_group = np.asarray(D_group, np.float64)
    moment = np.asarray(moment, np.float64)
    columns = np.asarray(group_columns, np.int64)
    if (D_group.ndim != 3 or D_group.shape[1:] != (96, 64)
            or moment.shape != (42, 96) or leg_index not in range(6)
            or columns.ndim != 1 or len(np.unique(columns)) != len(columns)
            or np.any((columns < 0) | (columns >= 96))):
        raise ValueError("E218 original grouped Jacobian dimensions invalid")
    gain = .02 * moment[7*leg_index:7*leg_index+7, columns]
    values = D_group[:, None, columns, :] * gain[None, :, :, None]
    return np.ascontiguousarray(values.reshape(len(D_group), 7, len(columns)*64))


def build_group_leg_problem(D_group, q0, target, moment, by_leg, leg_index):
    """Explicit group-64 QP/SOC; no E210 motor/16-shaped builder call."""
    if leg_index not in range(6) or len(by_leg) != 6:
        raise ValueError("E218 six original leg groups required")
    columns = np.asarray(by_leg[leg_index], np.int64)
    width = len(columns)*64
    q0 = np.asarray(q0, np.float64)
    indices = np.asarray(target["indices"], np.int64)
    starts = np.asarray(target["pair_start"], np.int64)
    ends = np.asarray(target["pair_end"], np.int64)
    if (D_group.shape != (2001, 96, 64) or q0.shape != (2001, 42)
            or width != LEG_WIDTHS[leg_index] or indices.shape != (1824,)
            or starts.shape != (1823,) or ends.shape != (1823,)
            or not np.array_equal(indices, np.arange(100, 1924))
            or not np.array_equal(starts, indices[:-1])
            or not np.array_equal(ends, indices[1:])
            or np.any(D_group[0] != 0)
            or not np.isfinite(D_group).all() or not np.isfinite(q0).all()):
        raise ValueError("E218 frozen grouped affine clocks/values changed")
    joint = group_joint_matrix(D_group, moment, columns, leg_index)
    j0 = 7*leg_index
    qres = (q0[indices, j0:j0+7] - target["target"][:, j0:j0+7]).reshape(-1)
    vres = ((q0[ends, j0:j0+7] - q0[starts, j0:j0+7]) / .005
            - target["velocity"][:, j0:j0+7]).reshape(-1)
    sres = (q0[1:100, j0:j0+7] -
            target["startup_command"][:, j0:j0+7]).reshape(-1)
    jq = joint[indices].reshape(1824*7, width)
    jv = ((joint[ends] - joint[starts]) / .005).reshape(1823*7, width)
    js = joint[1:100].reshape(99*7, width)
    del joint
    baseline = target["baseline"]
    qlimit = float(baseline["leg_command_rms"][leg_index])
    vlimit = float(baseline["leg_velocity_rms"][leg_index])
    normalizer = np.sqrt(len(qres))*qlimit
    if not np.isfinite(normalizer) or normalizer <= 0 or vlimit <= 0:
        raise ValueError("E218 positive physical baseline limits required")
    P, linear, constant = quadratic_objective(jq/normalizer, qres/normalizer)
    norm_rows, norm_rhs, factors, notes = [], [], {}, {}
    for name, matrix, offset, limit in (("command", jq, qres, qlimit),
                                        ("velocity", jv, vres, vlimit)):
        rows, rhs, factor, shifted, note = compressed_soc(
            matrix, offset, limit, width)
        norm_rows.append(rows)
        norm_rhs.append(rhs)
        factors[name + "_factor"] = factor
        factors[name + "_offset"] = shifted
        notes[name] = note
        if (note["relative_frobenius_error"] > 1e-10
                or note["minimum_eigenvalue"] < -1e-10 *
                    max(1., note["maximum_eigenvalue"])):
            raise ValueError("E218 augmented Gram compression failed")
    startup = sparse.csr_matrix(js)
    nn_rows, nn_rhs = box_and_startup(startup, sres, width)
    startup_rows, startup_rhs = soc_rows(startup/np.sqrt(len(sres)),
        sres/np.sqrt(len(sres)), .018*PAD, width=width)
    A = sparse.vstack((nn_rows, *norm_rows, startup_rows), format="csc")
    b = np.r_[nn_rhs, *norm_rhs, startup_rhs]
    dims = np.asarray([len(nn_rhs), *(len(v) for v in norm_rhs),
                       len(startup_rhs)], np.int64)
    kinds = ["NonnegativeConeT"] + ["SecondOrderConeT"]*3
    cones = [clarabel.NonnegativeConeT(int(dims[0]))] + [
        clarabel.SecondOrderConeT(int(v)) for v in dims[1:]]
    if (A.shape != (len(b), width) or not np.isfinite(A.data).all()
            or not np.isfinite(P.data).all() or not np.isfinite(linear).all()
            or not np.isfinite(b).all()):
        raise ValueError("E218 nonfinite physical group conic problem")
    return {"leg": LEGS[leg_index], "leg_index": leg_index,
        "group_columns": columns, "motor_columns": columns,
        "width": width, "P": P, "linear": linear,
        "objective_constant": constant, "A": A, "b": b,
        "cones": cones, "cone_dimensions": dims, "cone_types": kinds,
        "factors": factors, "factor_notes": notes,
        "q_offset": qres, "velocity_offset": vres,
        "startup_offset": sres, "startup_matrix": startup,
        "q_matrix": jq, "velocity_matrix": jv,
        "normalizer": float(normalizer), "command_limit": qlimit,
        "velocity_limit": vlimit, "target_indices": indices,
        "pair_start": starts, "pair_end": ends}


def save_group_problem(path, problem, affine_sha):
    """Persist one local raw cone and original-muscle column identity."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"linear": problem["linear"], "b": problem["b"],
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "objective_constant": np.asarray(problem["objective_constant"]),
        "group_columns": problem["group_columns"],
        "motor_columns": problem["group_columns"],
        "axis_semantics": np.asarray("original_muscle_group_not_motor"),
        "target_indices": problem["target_indices"],
        "pair_start": problem["pair_start"],
        "pair_end": problem["pair_end"],
        "q_offset": problem["q_offset"],
        "velocity_offset": problem["velocity_offset"],
        "startup_offset": problem["startup_offset"],
        "normalizer": np.asarray(problem["normalizer"]),
        "command_limit": np.asarray(problem["command_limit"]),
        "velocity_limit": np.asarray(problem["velocity_limit"]),
        "source_affine_sha256": np.asarray(affine_sha)}
    for key in ("P", "A", "startup_matrix"):
        arrays.update(csc_arrays(key, problem[key]))
    np.savez(path / "pre-solve.npz", **arrays)
    np.savez(path / "norm-factors.npz", **problem["factors"])
    save_json(path / "norm-factors.json", problem["factor_notes"])


def _restore_csc(saved, prefix):
    matrix = sparse.csc_matrix((
        np.asarray(saved[prefix + "_data"], np.float64),
        np.asarray(saved[prefix + "_indices"], np.int32),
        np.asarray(saved[prefix + "_indptr"], np.int32)),
        shape=tuple(np.asarray(saved[prefix + "_shape"], np.int64)))
    if not matrix.has_sorted_indices or not np.isfinite(matrix.data).all():
        raise ValueError("E218 saved CSC invalid: " + prefix)
    return matrix


def load_group_problem(path, expected_leg, affine_sha):
    """Reload original-muscle 64-column cone without the E211 motor/16 schema."""
    if expected_leg not in LEGS:
        raise ValueError("E218 unknown local leg")
    index = LEGS.index(expected_leg)
    width = LEG_WIDTHS[index]
    with np.load(path, allow_pickle=False) as saved:
        P = _restore_csc(saved, "P")
        A = _restore_csc(saved, "A")
        M = _restore_csc(saved, "startup_matrix")
        columns = np.asarray(saved["group_columns"], np.int64)
        dims = np.asarray(saved["cone_dimensions"], np.int64)
        kinds = [str(v) for v in saved["cone_types"]]
        linear = np.asarray(saved["linear"], np.float64)
        b = np.asarray(saved["b"], np.float64)
        offset = np.asarray(saved["startup_offset"], np.float64)
        constant = float(saved["objective_constant"])
        if (P.shape != (width, width) or A.shape[1] != width
                or M.shape != (693, width)
                or columns.shape != (LEG_GROUPS[index],)
                or not np.array_equal(np.sort(columns), columns)
                or np.any((columns < 0) | (columns >= 96))
                or not np.array_equal(saved["motor_columns"], columns)
                or str(saved["axis_semantics"]) !=
                    "original_muscle_group_not_motor"
                or linear.shape != (width,) or b.shape != (A.shape[0],)
                or offset.shape != (693,) or dims.shape != (4,)
                or dims[0] != 2*width + 2*693 or dims[3] != 694
                or dims.sum() != A.shape[0]
                or kinds != ["NonnegativeConeT"] + ["SecondOrderConeT"]*3
                or not np.isfinite(linear).all() or not np.isfinite(b).all()
                or not np.isfinite(offset).all() or not np.isfinite(constant)
                or str(saved["source_affine_sha256"]) != affine_sha
                or not np.all(P.indices <= np.repeat(
                    np.arange(width), np.diff(P.indptr)))):
            raise ValueError("E218 saved grouped local contract changed: " + expected_leg)
        tail = A[-694:]
        if (tail[0].nnz != 0 or not np.array_equal(tail[1:].toarray(),
                (-M / np.sqrt(693)).toarray())
                or not np.array_equal(b[-694:],
                    np.r_[.018*PAD, offset/np.sqrt(693)])):
            raise ValueError("E218 local startup cone is not raw startup map")
        return {"leg": expected_leg, "P": P, "A": A, "M": M,
            "linear": linear.copy(), "b": b.copy(), "dims": dims.copy(),
            "kinds": kinds, "group_columns": columns.copy(),
            "motor_columns": columns.copy(), "startup_offset": offset.copy(),
            "objective_constant": constant,
            "q_offset": np.asarray(saved["q_offset"], np.float64).copy(),
            "velocity_offset": np.asarray(saved["velocity_offset"], np.float64).copy(),
            "normalizer": float(saved["normalizer"]),
            "command_limit": float(saved["command_limit"]),
            "velocity_limit": float(saved["velocity_limit"])}


def prepare_local_problems(out, D_group, q0, target, moment, by_leg,
                           affine_sha, *, elapsed=None,
                           builder=build_group_leg_problem,
                           saver=save_group_problem):
    """Sequential build/save/release, with no local solver invocation."""
    out = Path(out)
    records = []
    for index, leg in enumerate(LEGS):
        if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
            raise TimeoutError("E218 local preparation exceeded internal deadline")
        path = out / ("leg-" + leg)
        path.mkdir()
        began = time.perf_counter()
        problem = builder(D_group, q0, target, moment, by_leg, index)
        try:
            if (problem["leg"] != leg or problem["width"] != LEG_WIDTHS[index]
                    or not np.array_equal(problem["group_columns"], by_leg[index])):
                raise ValueError("E218 local group builder changed ownership")
            saver(path, problem, affine_sha)
            records.append({"leg": leg, "group_columns": by_leg[index].tolist(),
                "variables": int(problem["P"].shape[0]),
                "rows": int(problem["A"].shape[0]),
                "P_nnz": int(problem["P"].nnz), "A_nnz": int(problem["A"].nnz),
                "build_and_save_wall_s": time.perf_counter()-began,
                "local_solver_calls": 0})
        finally:
            del problem
            gc.collect()
        save_json(out / "local-preparation.json", records)
    return records


def verify_group_global_structure(problem, locals_, by_leg, free_groups):
    """Restore all grouped ownership/width/cone checks omitted by strict=False."""
    if len(locals_) != 6 or [v["leg"] for v in locals_] != list(LEGS):
        raise ValueError("E218 six leg local order changed")
    columns = np.concatenate([local["group_columns"] for local in locals_])
    dims = np.asarray(problem["cone_dimensions"], np.int64)
    kinds = list(problem["cone_types"])
    expected_dims = np.r_[np.concatenate([v["dims"][:3] for v in locals_]), 4159]
    expected_kinds = [kind for v in locals_ for kind in v["kinds"][:3]]
    expected_kinds.append("SecondOrderConeT")
    if (len(free_groups) != 90 or len(np.unique(columns)) != 90
            or not np.array_equal(np.sort(columns), np.sort(free_groups))
            or not np.array_equal(problem["motor_columns"], columns)
            or any(not np.array_equal(v["group_columns"], by_leg[i])
                   for i, v in enumerate(locals_))
            or any(v["P"].shape != (LEG_WIDTHS[i], LEG_WIDTHS[i])
                   or v["M"].shape != (693, LEG_WIDTHS[i])
                   for i, v in enumerate(locals_))
            or problem["P"].shape != (5760, 5760)
            or problem["A"].shape != (int(dims.sum()), 5760)
            or problem["startup_matrix"].shape != (4158, 5760)
            or problem["startup_offset"].shape != (4158,)
            or problem["linear"].shape != (5760,)
            or problem["b"].shape != (int(dims.sum()),)
            or len(problem["cones"]) != 19 or dims.shape != (19,)
            or not np.array_equal(dims, expected_dims)
            or kinds != expected_kinds
            or not np.array_equal(problem["startup_offset"],
                np.concatenate([v["startup_offset"] for v in locals_]))):
        raise ValueError("E218 grouped 5760/19-cone/global startup structure changed")
    return {"variables": 5760, "free_muscles": 90, "cones": 19,
        "startup_values": 4158, "A_rows": int(problem["A"].shape[0]),
        "A_nnz": int(problem["A"].nnz), "P_nnz": int(problem["P"].nnz)}


def save_group_global_problem(out, problem, affine_sha):
    # The generic assembler internally calls these motor_columns. Persist the
    # same indices with their true original-muscle meaning in the raw cone.
    arrays = {"linear": problem["linear"], "b": problem["b"],
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "objective_constant": np.asarray(problem["objective_constant"]),
        "motor_columns": problem["motor_columns"],
        "group_columns": problem["motor_columns"],
        "axis_semantics": np.asarray("original_muscle_group_not_motor"),
        "startup_offset": problem["startup_offset"],
        "source_affine_sha256": np.asarray(affine_sha)}
    for key in ("P", "A", "startup_matrix"):
        arrays.update(csc_arrays(key, problem[key]))
    np.savez(Path(out) / "pre-solve.npz", **arrays)


def canonical_group_readout(rate, pool, moment, rest, ltm, groups,
                            initial_activation, held_command):
    """Original anatomical pool and 10 ms activation, no StateStore/model."""
    rate = np.asarray(rate, np.float64)
    pool = np.asarray(pool, np.float64)
    moment = np.asarray(moment, np.float64)
    rest = np.asarray(rest, np.float64)
    ltm = np.asarray(ltm, np.int64)
    active = np.asarray(initial_activation, np.float64).copy()
    held = np.asarray(held_command, np.float64)
    frames = len(rate)
    if (rate.ndim != 2 or rate.shape != (frames, 391) or frames < 1
            or pool.shape != (96, 391)
            or moment.shape != (42, 96) or rest.shape != (42,)
            or ltm.shape != (6,) or active.shape != (96,)
            or held.shape != (42,) or not np.isfinite(rate).all()
            or not np.isfinite(active).all()):
        raise ValueError("E218 direct canonical readout dimensions changed")
    names = [groups[int(g)]["leg"] + "_tarsus" for g in ltm]
    if set(names) != {leg + "_tarsus" for leg in LEGS}:
        raise ValueError("E218 six original LTM tendon names changed")
    order = [names.index(leg + "_tarsus") for leg in LEGS]
    command = np.empty((frames, 42), np.float64)
    activation = np.empty((frames, 96), np.float64)
    tendon = np.empty((frames, 6), np.float64)
    command[0] = held
    activation[0] = active
    tendon[0] = .2*np.tanh(active[ltm][order]/10)
    alpha = -np.expm1(-.005/.01)
    for t in range(1, frames):
        active += alpha*(pool @ rate[t] - active)
        activation[t] = active
        command[t] = rest + .02*(moment @ active)
        tendon[t] = .2*np.tanh(active[ltm][order]/10)
    return {"command": command, "activation": activation, "tendon": tendon}


def direct_group_readout(beta, arrays):
    """Use own MN basis to compute rates, and compare pooled group affine q."""
    beta = np.asarray(beta, np.float64)
    D = arrays["D_group"]
    B = arrays["B"]
    H = arrays["H"]
    cap = arrays["cap"]
    membership = arrays["membership"]
    fixed = arrays["fixed"]
    q0 = arrays["q0"]
    frames = len(H)
    if (beta.shape != (96, 64) or D.shape != (frames, 96, 64)
            or B.shape != (frames, 391, 64)
            or H.shape != (frames, 391) or cap.shape != (391,)
            or membership.shape != (391,) or fixed.shape != (48,)
            or q0.shape != (frames, 42)
            or arrays["original_rates"].shape != (frames, 391)
            or np.any(beta < 0) or np.any(beta > 1)
            or not np.isfinite(beta).all()):
        raise ValueError("E218 strict shared physical beta/rate shape changed")
    rate = H + cap[None] * np.sum(B * beta[membership][None], axis=2)
    rate[:, fixed] = arrays["original_rates"][:, fixed]
    direct = canonical_group_readout(rate, arrays["pool"], arrays["moment"],
        arrays["rest"], arrays["ltm"], arrays["groups"],
        arrays["activation0"], q0[0])
    delta = np.sum(D*beta[None], axis=2)
    affine = q0 + .02*(delta @ arrays["moment"].T)
    affine[0] = q0[0]
    return beta, rate, direct, affine


def bounded_infeasible_diagnostic(problem, raw, info, unpadded_b=None):
    """Raw exact-cone dual box separation scoped to this conditional family."""
    z = raw.get("solver_z")
    rows = problem["A"].shape[0]
    report = {"status": info.get("status"), "certificate_verified": False,
        "scope": "E217 conditional muscle-shared 64-basis family only"}
    if not isinstance(z, np.ndarray) or z.shape != (rows,) or not np.isfinite(z).all():
        return dict(report, reason="raw dual absent, malformed, or nonfinite")
    az = np.asarray(problem["A"].T @ z).reshape(-1)
    violations = []
    offset = 0
    for kind, size in zip(problem["cone_types"], problem["cone_dimensions"]):
        piece = z[offset:offset+size]
        value = (-float(np.min(piece)) if kind == "NonnegativeConeT"
                 else float(np.linalg.norm(piece[1:])-piece[0]))
        violations.append(max(0., value))
        offset += size
    if offset != rows:
        return dict(report, reason="cone row partition changed")
    btz = float(problem["b"] @ z)
    box_min = float(np.minimum(az, 0).sum())
    separation = box_min - btz
    report = dict(report, b_dot_z=btz, box_linear_minimum=box_min,
        box_separation_margin=separation,
        dual_cone_violation_max=max(violations),
        dual_residual_reported=info.get("dual_residual"),
        certificate_verified=bool(np.isfinite(separation)
            and separation > 1e-8 and max(violations) == 0
            and np.isfinite(info.get("dual_residual", np.nan))
            and info["dual_residual"] <= 1e-8))
    if unpadded_b is not None:
        physical_b = np.asarray(unpadded_b, np.float64)
        if physical_b.shape != (rows,) or not np.isfinite(physical_b).all():
            raise ValueError("E218 unpadded physical bounds invalid")
        margin = box_min - float(physical_b @ z)
        report["unpadded_b_dot_z"] = float(physical_b @ z)
        report["unpadded_box_separation_margin"] = margin
        report["unpadded_certificate_verified"] = bool(
            np.isfinite(margin) and margin > 1e-8
            and max(violations) == 0
            and np.isfinite(info.get("dual_residual", np.nan))
            and info["dual_residual"] <= 1e-8)
    return report


def load_frozen_arrays(paths):
    with (np.load(paths["parameters"], allow_pickle=False) as p,
          np.load(paths["prepared_parameters"], allow_pickle=False) as prepared,
          np.load(paths["endpoints"], allow_pickle=False) as native,
          np.load(paths["mask"], allow_pickle=False) as mask,
          np.load(paths["basis"], allow_pickle=False) as basis,
          np.load(paths["affine"], allow_pickle=False) as affine,
          np.load(paths["geometry"], allow_pickle=False) as geometry):
        arrays = {"D_group": affine["D_group"].copy(),
            "q0": affine["q_intercept"].copy(),
            "B": basis["B"].copy(), "H": basis["H"].copy(),
            "cap": p["cap"].astype(np.float64),
            "original_rates": native["motor_rate"].astype(np.float64),
            "free": mask["free_columns"].astype(np.int64),
            "fixed": mask["fixed_columns"].astype(np.int64),
            "membership": p["membership"].astype(np.int64),
            "pool": p["pool"].astype(np.float64),
            "moment": p["moment"].astype(np.float64),
            "rest": p["rest"].astype(np.float64),
            "ltm": p["ltm"].astype(np.int64),
            "groups": json.loads(str(p["muscle_parameters_json"])),
            "activation0": affine["activation_intercept"][0].astype(np.float64)}
        if (arrays["D_group"].shape != (2001, 96, 64)
                or arrays["q0"].shape != (2001, 42)
                or arrays["B"].shape != (2001, 391, 64)
                or arrays["H"].shape != (2001, 391)
                or arrays["cap"].shape != (391,)
                or arrays["original_rates"].shape != (2001, 391)
                or arrays["free"].shape != (343,)
                or arrays["fixed"].shape != (48,)
                or arrays["activation0"].shape != (96,)
                or not np.array_equal(np.sort(np.r_[arrays["free"], arrays["fixed"]]),
                                      np.arange(391))
                or not np.array_equal(prepared["motor_ids"], p["motor_ids"])
                or not np.array_equal(mask["motor_ids"], p["motor_ids"])
                or not np.array_equal(prepared["membership"], arrays["membership"])
                or not np.array_equal(mask["membership"], arrays["membership"])
                or not np.array_equal(prepared["free_columns"], arrays["free"])
                or not np.array_equal(prepared["fixed_columns"], arrays["fixed"])
                or not np.array_equal(prepared["pool"], arrays["pool"])
                or not np.array_equal(prepared["moment"], arrays["moment"])
                or not np.array_equal(prepared["rest"], arrays["rest"])
                or not np.array_equal(prepared["ltm"], arrays["ltm"])
                or not np.array_equal(prepared["cap"], arrays["cap"])
                or geometry["membership"].shape != (391,)
                or not np.array_equal(geometry["membership"], arrays["membership"])
                or not np.array_equal(arrays["original_rates"][0],
                                      p["initial_rate"])
                or not np.allclose(affine["rate_intercept"][:, arrays["free"]],
                                   arrays["H"][:, arrays["free"]],
                                   atol=2e-12, rtol=2e-12)
                or not np.array_equal(affine["rate_intercept"][:, arrays["fixed"]],
                                      arrays["original_rates"][:, arrays["fixed"]])
                or np.any(arrays["D_group"][:, arrays["ltm"]] != 0)
                or np.any(arrays["D_group"][0] != 0)
                or np.any(arrays["B"][0] != 0)
                or np.min(arrays["B"]) < -1e-12
                or np.any(arrays["original_rates"] < 0)
                or np.any(arrays["original_rates"] > arrays["cap"][None])
                or not all(np.isfinite(v).all() for v in arrays.values()
                           if isinstance(v, np.ndarray))):
            raise ValueError("E218 pinned E217 grouped basis/anatomy changed")
    return arrays


def unpadded_rhs(problem, locals_):
    """Physical limits without numerical PAD, for diagnostic comparison only."""
    b = np.asarray(problem["b"], np.float64).copy()
    row = 0
    for local in locals_:
        width = local["P"].shape[0]
        nn, qdim, vdim = (int(v) for v in local["dims"][:3])
        b[row+2*width:row+nn] += .04*(1-PAD)
        b[row+nn] += 1-PAD  # normalized q limit is one
        b[row+nn+qdim] += 1-PAD  # normalized velocity limit is one
        row += nn+qdim+vdim
    if row != int(np.sum(problem["cone_dimensions"][:-1])):
        raise ValueError("E218 original cone offset changed")
    b[row] += .018*(1-PAD)
    return b


def publish_reconstruction(out, arrays, target, trace, by_leg,
                           locals_, problem, raw, report, source_hashes):
    """Preserve a complete solved/box raw readout before acceptance verdict."""
    x = raw["solver_x"]
    beta = np.zeros((96, 64), np.float64)
    cursor = 0
    for groups in by_leg:
        width = len(groups)*64
        beta[groups] = x[cursor:cursor+width].reshape(-1, 64)
        cursor += width
    beta, rate, direct, q_affine = direct_group_readout(beta, arrays)
    np.savez(Path(out) / "reconstruction-raw.npz", beta=beta, rate=rate,
        q_affine=q_affine, q_direct=direct["command"],
        tendon=direct["tendon"], activation=direct["activation"])
    if not all(np.isfinite(value).all() for value in (
            beta, rate, q_affine, direct["command"], direct["tendon"],
            direct["activation"])):
        raise ValueError("E218 saved direct reconstruction has nonfinite output")
    original = canonical_group_readout(arrays["original_rates"], arrays["pool"],
        arrays["moment"], arrays["rest"], arrays["ltm"], arrays["groups"],
        arrays["activation0"], arrays["q0"][0])
    np.savez(Path(out) / "original-canonical.npz", **original)
    metrics = direct_metrics(direct, target)
    baseline = target["baseline"]
    affine_exact = bool(np.allclose(q_affine, direct["command"],
        atol=2e-10, rtol=2e-12))
    fixed_exact = bool(np.array_equal(direct["activation"][:, arrays["ltm"]],
            original["activation"][:, arrays["ltm"]]) and
        np.array_equal(direct["tendon"], original["tendon"]))
    legs = []
    for i, leg in enumerate(LEGS):
        j = slice(7*i, 7*(i+1))
        eq = q_affine[target["indices"], j] - target["target"][:, j]
        ev = ((q_affine[target["pair_end"], j]
            - q_affine[target["pair_start"], j])/.005
            - target["velocity"][:, j])
        es = q_affine[1:100, j] - target["startup_command"][:, j]
        legs.append({"leg": leg, "q_rms": float(np.sqrt(np.mean(eq*eq))),
            "velocity_rms": float(np.sqrt(np.mean(ev*ev))),
            "startup_q_rms": float(np.sqrt(np.mean(es*es))),
            "startup_q_max": float(np.max(np.abs(es)))})
    physical = physical_limit_diagnostics(legs, locals_)
    physical["startup_raw_affine_match"] = bool(all(np.isclose(
        leg["startup_q_rms"], report["direct"]["legs"][i]["startup_q_rms"],
        atol=2e-10, rtol=2e-12) for i, leg in enumerate(legs)))
    report["direct"]["physical"] = physical
    report["direct_training_pass"] = bool(report["direct_training_pass"]
        and all(physical["leg_q_original_limit_pass"])
        and all(physical["leg_velocity_original_limit_pass"])
        and physical["startup_raw_affine_match"])
    save_json(Path(out) / "direct-metrics.json", report)
    verdict = proxy_verdict(metrics, baseline)
    verdict.update(global_cone_numerical_pass=report["numerical_pass"],
        global_training_pass=report["direct_training_pass"],
        affine_exact=affine_exact, fixed_ltm_exact=fixed_exact)
    verdict["passed"] = bool(verdict["passed"] and report["numerical_pass"]
        and report["direct_training_pass"] and affine_exact and fixed_exact)
    trace_q = np.asarray([row["joint_targets_rad"]
                          for row in trace["samples"]])
    diagnostics = {"metrics": metrics, "baseline": baseline,
        "proxy_verdict": verdict,
        "canonical_original_vs_E150_monotone_q_max_abs": float(np.max(
            np.abs(original["command"] - trace_q))),
        "affine_q_max_abs": float(np.max(np.abs(
            q_affine - direct["command"])))}
    save_json(Path(out) / "readout-diagnostics.json", diagnostics)
    if verdict["passed"]:
        np.savez(Path(out) / "candidate.npz", beta=beta,
            group_columns=np.concatenate(by_leg),
            source_family=np.asarray("E217 muscle-shared 64-basis"),
            source_geometry_sha256=np.asarray(source_hashes["geometry"]),
            source_scale_slow_sha256=np.asarray(source_hashes["prepared_parameters"]),
            source_basis_sha256=np.asarray(source_hashes["basis"]),
            rate=rate, q_direct=direct["command"],
            q_affine=q_affine, tendon=direct["tendon"],
            activation=direct["activation"])
    return diagnostics


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e218.output-manifest.v1",
        "files": files,
        "total_bytes": sum(value["bytes"] for value in files.values())}
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
        raise FileExistsError("One-shot E218 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = None
    primary = None
    errors = []
    phase = "gate"
    ledger = {"schema": "flylab.m3.e218.ledger.v1", "phase": phase,
        "solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0,
        "local_solver_calls": 0, "local_problems_saved": 0,
        "constructor_wall_s": None, "solver_wall_s": None,
        "full_banc_calls": 0, "cuda_calls": 0,
        "physics_steps": 0, "retries": 0}
    result = {"schema": "flylab.m3.e218.grouped-response-fit.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "conditional_proxy_only": True}

    def elapsed():
        return time.perf_counter()-began

    try:
        freeze, review, paths = verify_gate(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E218 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "family": "E217 muscle-shared 64-basis"})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            phase = "input"
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            if len(trace["samples"]) != 2001:
                raise ValueError("E218 original 2001-control trace changed")
            target = load_targets(paths["targets"], trace)
            arrays = load_frozen_arrays(paths)
            by_leg = group_anatomy(arrays["membership"], arrays["pool"],
                arrays["moment"], arrays["ltm"], arrays["groups"],
                arrays["free"], arrays["fixed"])
            free_groups = np.unique(arrays["membership"][arrays["free"]])
            affine_sha = freeze["input_sha256"][freeze["paths"]["affine"]]
            phase = "local-preparation"
            records = prepare_local_problems(out, arrays["D_group"],
                arrays["q0"], target, arrays["moment"], by_leg,
                affine_sha, elapsed=elapsed)
            ledger["local_problems_saved"] = len(records)
            save_json(out / "ledger.json", ledger)
            locals_ = [load_group_problem(out / ("leg-" + leg) / "pre-solve.npz",
                                          leg, affine_sha) for leg in LEGS]
            problem = assemble_global(locals_, strict=False)
            result["structure"] = verify_group_global_structure(
                problem, locals_, by_leg, free_groups)
            phase = "global-presolve"
            save_group_global_problem(out, problem, affine_sha)
            physical_b = unpadded_rhs(problem, locals_)
            np.savez(out / "unpadded-bounds.npz", b=physical_b,
                original_padded_b=problem["b"])
            ensure_solve_budget(elapsed)
            phase = "solver"
            raw, info = solve_once(problem, out, ledger, elapsed=elapsed)
            result["solver_info"] = info
            report = assess_solution(problem, raw, info, locals_)
            save_json(out / "direct-metrics.json", report)
            result["solver_assessment"] = report
            if report["eligible_beta"]:
                phase = "direct-readout"
                diagnostics = publish_reconstruction(out, arrays, target,
                    trace, by_leg, locals_, problem, raw, report,
                    {key: freeze["input_sha256"][freeze["paths"][key]]
                     for key in ("geometry", "prepared_parameters", "basis")})
                result.update(diagnostics)
                result["validity"] = "valid_conditional_fit_supported" if (
                    diagnostics["proxy_verdict"]["passed"]
                    ) else "valid_conditional_fit_rejected"
            else:
                result["validity"] = "valid_bounded_unresolved_no_complete_fit"
                result["infeasibility_diagnostic"] = bounded_infeasible_diagnostic(
                    problem, raw, info, physical_b)
                save_json(out / "infeasibility-diagnostic.json",
                          result["infeasibility_diagnostic"])
    except SolverAllowanceUnavailable as error:
        result["validity"] = "valid_bounded_unresolved_no_solver_call"
        result["budget_block"] = str(error)
        ledger["blocked_before_solve"] = True
        ledger["budget_block"] = str(error)
    except BaseException as error:  # noqa: BLE001 - preserve raw and primary
        primary = error
        errors.append({"stage": phase, "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    ledger["phase"] = phase
    ledger["process_wall_s"] = elapsed()
    ledger["local_problems_saved"] = sum(
        (out / ("leg-" + leg) / "pre-solve.npz").is_file() for leg in LEGS)
    _save_attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"], local_solver_calls=0,
        solver_wall_s=ledger["solver_wall_s"],
        solver_soft_limit_overrun_s=ledger.get("solver_soft_limit_overrun_s"),
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0)+elapsed(),
        retries=0)
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    _save_attempt(errors, "execution-errors.json",
                  lambda: save_json(out / "execution-errors.json", errors))
    _save_attempt(errors, "result-preliminary", lambda: save_json(out / "result.json",
        dict(result, validity="incomplete_or_invalid", finalization_pending=True)))
    manifest = _save_attempt(errors, "output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E218 output manifest missing or over 768 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E218 hard internal deadline exceeded"
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
            primary = RuntimeError("E218 raw/publication failure")
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
    try:
        result = run(args.freeze, args.freeze_sha256,
                     args.review, args.review_sha256, args.out)
    except BaseException as error:  # noqa: BLE001 - nonzero invalid exit
        print(f"E218 invalid: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
