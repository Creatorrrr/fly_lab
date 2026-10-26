"""E242 one bounded, saved-input, muscle-shared teacher fit.

This program does not instantiate a neural network or a body. The validation
interval is used only after the one frozen training QP has returned.
"""
# ruff: noqa: BLE001 -- retain the first solver/scientific exception on cleanup
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import fit_c_banc_m3_grouped_response as grouped
from tools import prepare_c_banc_m3_matched_input_basis as prepared
from tools.fit_c_banc_m3_causal_neural_basis import quadratic_objective
from tools.fit_c_banc_m3_constrained_tangent import csc_arrays
from tools.verify_c_banc_m3_native_current import configure_single_thread_runtime

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e242-matched-teacher-fit"
FREEZE_SHA = "e397929cdbdb6ff2353e96be35d944b75f9d31bdfdc998710e4a00ad7899a062"
FREEZE_SCHEMA = "flylab.m3.e242.matched-teacher-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e242.matched-teacher-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_matched_teacher.py",
         "tests/test_c_banc_matched_teacher_fit.py")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
WIDTHS = (1024, 896, 960, 1024, 896, 960)
INTERNAL_S = 150.0
RESERVE_S = 10.0
SOLVER_S = 90.0


def save_json(path, value):
    prepared.save_json(Path(path), value)


def save_arrays(path, arrays):
    prepared.save_arrays(Path(path), arrays)


def safe(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E242 repository-relative slash file required")
    name = Path(relative)
    if name.is_absolute() or any(part in (".", "..") for part in name.parts):
        raise ValueError("Unsafe E242 input path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E242 frozen input missing")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    if (Path(freeze_path).resolve() !=
            (BASE / "e242-matched-teacher-fit-freeze.json").resolve()
            or freeze_sha != FREEZE_SHA
            or prepared.prior.sha256(freeze_path) != FREEZE_SHA
            or prepared.prior.sha256(review_path) != review_sha):
        raise ValueError("Exact E242 freeze and source review required")
    freeze = json.loads(Path(freeze_path).read_text("utf-8"))
    review = json.loads(Path(review_path).read_text("utf-8"))
    limits = freeze.get("resource_limits", {})
    shape = freeze.get("shape", {})
    split = freeze.get("split", {})
    objective = freeze.get("objective", {})
    solver = freeze.get("solver", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") !=
                "docs/BANC_M3_E242_MATCHED_TEACHER_FIT_DESIGN_20260925.md"
            or freeze.get("owned_files") != list(OWNED)
            or shape.get("variables") != 5760
            or shape.get("leg_widths") != list(WIDTHS)
            or shape.get("samples") != 2001
            or shape.get("free_motors") != 343
            or shape.get("fixed_motors") != 48
            or shape.get("basis_count") != 64
            or split.get("train_control_first") != 1
            or split.get("train_control_last") != 1200
            or split.get("validation_control_first") != 1201
            or split.get("validation_control_last") != 2000
            or objective.get("ridge_per_leg") != 1e-8
            or objective.get("beta_min") != 0
            or objective.get("beta_max") != 1
            or solver.get("max_threads") != 1
            or solver.get("max_iter") != 150
            or solver.get("time_limit_s") != SOLVER_S
            or solver.get("required_status") != "Solved"
            or limits.get("internal_s") != INTERNAL_S
            or limits.get("finalization_reserve_s") != RESERVE_S
            or limits.get("solver_constructors") != 1
            or limits.get("solver_calls") != 1
            or limits.get("original_attempts") != 1
            or limits.get("retries") != 0
            or limits.get("gpu_count") != 0
            or freeze.get("candidate_adopted") is not False
            or freeze.get("F6_exposed") is not False
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != {
                name: prepared.prior.sha256(safe(name)) for name in OWNED}):
        raise ValueError("E242 reviewed scientific/resource contract changed")
    if prepared.prior.sha256(safe(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E242 design changed")
    for name, expected in freeze["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("E242 pinned package changed: " + name)
    for source, labels in ((freeze, ("input_sha256", "reused_source_sha256",
                                    "preservation_sha256")),
                           (review, ("additional_reused_source_sha256",
                                     "additional_input_sha256"))):
        for label in labels:
            for relative, expected in source.get(label, {}).items():
                if prepared.prior.sha256(safe(relative)) != expected:
                    raise ValueError("E242 frozen byte changed: " + relative)
    if out.resolve() != OUT.resolve() or out.exists() or not out.parent.is_dir():
        raise ValueError("New exact E242 output directory required")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E242 free-disk reserve unavailable")
    paths = {key: safe(value) for key, value in freeze["paths"].items()}
    source = json.loads(paths["source_result"].read_text("utf-8"))
    proof = json.loads(paths["source_proof"].read_text("utf-8"))
    source_ledger = json.loads(paths["source_ledger"].read_text("utf-8"))
    supervisor = json.loads(paths["source_supervisor"].read_text("utf-8"))
    if (source.get("validity") != "valid_matched_input_grouped_basis"
            or proof.get("validity") != "independently_valid"
            or source_ledger.get("cuda_synchronized") != 2
            or source_ledger.get("fit_calls") != 0
            or source_ledger.get("solver_calls") != 0
            or source_ledger.get("errors")
            or supervisor.get("exit_code") != 0):
        raise ValueError("E242 valid independently checked E241 basis required")
    return freeze, paths


def load_npz(path, keys):
    """Read a compressed source once; later leg loops use cached arrays."""
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in keys}


def rms(value):
    value = np.asarray(value, np.float64)
    return float(np.sqrt(np.mean(value * value)))


def training_scales(original_command, target, freeze_normalization):
    """Only original-decoder TRAIN controls contribute q and velocity scales."""
    original = np.asarray(original_command, np.float64)
    target = np.asarray(target, np.float64)
    if (original.shape != (2001, 42) or target.shape != (2001, 42)
            or not np.isfinite(original).all() or not np.isfinite(target).all()):
        raise ValueError("E242 complete finite same-control commands required")
    error = original - target
    result = {}
    for i, leg in enumerate(LEGS):
        part = error[:, 7*i:7*(i+1)]
        q = rms(part[1:1201])
        v = rms(np.diff(part[:1201], axis=0) / .005)
        fixed = freeze_normalization[leg]
        if (not np.isfinite(q) or not np.isfinite(v) or q <= 0 or v <= 0
                or not np.isclose(q, fixed["q"], atol=1e-12, rtol=1e-12)
                or not np.isclose(v, fixed["velocity"], atol=1e-12, rtol=1e-12)):
            raise ValueError("E242 frozen TRAIN-only scale mismatch: " + leg)
        result[leg] = {"q": q, "velocity": v}
    return result


def train_leg_matrix(D_group, q_intercept, target, moment, columns, leg_index,
                     q_scale, velocity_scale):
    """Rows correspond to q[c] and (q[c]-q[c-1])/.005 for c=1..1200."""
    if (not 0 <= leg_index < 6 or q_scale <= 0 or velocity_scale <= 0
            or not np.isfinite([q_scale, velocity_scale]).all()
            or np.shape(q_intercept) != (2001, 42)
            or np.shape(target) != (2001, 42)):
        raise ValueError("E242 aligned leg-training layout required")
    columns = np.asarray(columns, np.int64)
    joint = grouped.group_joint_matrix(D_group[:1201], moment, columns, leg_index)
    if joint.shape != (1201, 7, len(columns)*64):
        raise ValueError("E242 muscle-group column order changed")
    sl = slice(7*leg_index, 7*(leg_index+1))
    residual = np.asarray(q_intercept[:, sl] - target[:, sl], np.float64)
    matrix, offset = normalized_train_rows(joint, residual[:1201],
                                            q_scale, velocity_scale)
    if matrix.shape != (16800, len(columns)*64) or not np.isfinite(matrix).all():
        raise ValueError("E242 nonfinite or incomplete TRAIN objective")
    return matrix, offset


def normalized_train_rows(joint, residual, q_scale, velocity_scale):
    """Actual fit row ordering, also usable for tiny independent algebra tests."""
    joint = np.asarray(joint, np.float64)
    residual = np.asarray(residual, np.float64)
    count = len(joint)-1
    if (joint.ndim != 3 or count < 1 or residual.shape != joint.shape[:2]
            or q_scale <= 0 or velocity_scale <= 0
            or not np.isfinite([q_scale, velocity_scale]).all()
            or not np.isfinite(joint).all() or not np.isfinite(residual).all()):
        raise ValueError("E242 finite adjacent-control TRAIN rows required")
    q_normalizer = np.sqrt(count*joint.shape[1]) * q_scale
    v_normalizer = np.sqrt(count*joint.shape[1]) * velocity_scale
    matrix = np.concatenate((joint[1:].reshape(-1, joint.shape[2])/q_normalizer,
        (np.diff(joint, axis=0)/.005).reshape(-1, joint.shape[2])/v_normalizer), axis=0)
    offset = np.r_[residual[1:].ravel()/q_normalizer,
                   (np.diff(residual, axis=0)/.005).ravel()/v_normalizer]
    return matrix, offset


def build_qp(D_group, q_intercept, target, moment, by_leg, scales):
    """One six-block training QP; this never accesses validation target rows."""
    blocks, linear, constants = [], [], []
    for index, leg in enumerate(LEGS):
        columns = np.asarray(by_leg[index], np.int64)
        if len(columns)*64 != WIDTHS[index]:
            raise ValueError("E242 frozen per-leg group width changed")
        matrix, residual = train_leg_matrix(D_group, q_intercept, target,
            moment, columns, index, scales[leg]["q"], scales[leg]["velocity"])
        P, q, constant = quadratic_objective(matrix, residual, ridge=1e-8)
        blocks.append(P)
        linear.append(q)
        constants.append(constant)
        del matrix, residual
    problem = assemble_box_blocks(blocks, linear, constants, by_leg)
    P, A, b = problem["P"], problem["A"], problem["b"]
    width = P.shape[0]
    if (width != 5760 or P.shape != (width, width)
            or A.shape != (11520, width)
            or not np.all(b[:width] == 0)
            or not np.all(b[width:] == 1)):
        raise ValueError("E242 single 5760-variable box QP malformed")
    return problem


def assemble_box_blocks(blocks, linear, constants, by_leg):
    """Shared small/large assembly; preserve physical group column order."""
    widths = [len(q) for q in linear]
    if (not blocks or len(blocks) != len(linear) or len(blocks) != len(constants)
            or len(blocks) != len(by_leg)
            or any(block.shape != (width, width)
                   for block, width in zip(blocks, widths))):
        raise ValueError("E242 aligned per-leg QP blocks required")
    offsets = np.r_[0, np.cumsum(widths)].astype(np.int64)
    width = int(offsets[-1])
    P = sparse.block_diag(blocks, format="csc")
    identity = sparse.eye(width, format="csc")
    A = sparse.vstack((-identity, identity), format="csc")
    b = np.r_[np.zeros(width), np.ones(width)]
    return {"P": P, "linear": np.concatenate(linear), "A": A, "b": b,
            "objective_constant": float(sum(constants)),
            "leg_objective_constants": np.asarray(constants, np.float64),
            "leg_offsets": offsets,
            "group_columns": np.concatenate(by_leg).astype(np.int64),
            "leg_group_counts": np.asarray([len(v) for v in by_leg], np.int64),
            "cone_dimensions": np.asarray([2*width], np.int64)}


def save_problem(out, problem, scales):
    arrays = {key: problem[key] for key in ("linear", "b", "objective_constant",
        "leg_objective_constants", "leg_offsets", "group_columns",
        "leg_group_counts", "cone_dimensions")}
    arrays["cone_types"] = np.asarray(["NonnegativeConeT"])
    arrays["normalization_q"] = np.asarray([scales[leg]["q"] for leg in LEGS])
    arrays["normalization_velocity"] = np.asarray(
        [scales[leg]["velocity"] for leg in LEGS])
    arrays["axis_semantics"] = np.asarray("original_muscle_group_64")
    for key in ("P", "A"):
        arrays.update(csc_arrays(key, problem[key]))
    save_arrays(Path(out) / "pre-solve.npz", arrays)


def objective_value(problem, x):
    x = np.asarray(x, np.float64)
    return float(.5*x @ symmetric_p_matvec(problem["P"], x) + problem["linear"] @ x +
                 problem["objective_constant"])


def symmetric_p_matvec(upper, x):
    """Clarabel stores only the upper CSC triangle of its symmetric Hessian."""
    return upper @ x + upper.T @ x - upper.diagonal()*x


def solver_settings():
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 150
    settings.time_limit = 90.
    settings.tol_gap_abs = 1e-8
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-8
    return settings


def ensure_solve_budget(elapsed):
    if elapsed() + SOLVER_S + RESERVE_S > INTERNAL_S:
        raise TimeoutError("E242 full solver and finalization allowance unavailable")


def solve_once(problem, out, ledger, elapsed, solver_factory=None):
    """Persist one requested constructor/call and all raw arrays before checks."""
    if ledger["constructor_entered"] or ledger["solver_calls"]:
        raise RuntimeError("E242 second solver construction/call forbidden")
    solver_factory = solver_factory or clarabel.DefaultSolver
    settings = solver_settings()
    save_json(Path(out) / "solver-settings.json", {"configured": {
        "max_threads": 1, "max_iter": 150, "time_limit_s": 90.,
        "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8},
        "all_settings_text": str(settings)})
    ensure_solve_budget(elapsed)
    ledger["constructor_requested"] = True
    ledger["constructor_entered"] = True
    try:
        save_json(Path(out) / "ledger.json", ledger)
    except BaseException:
        ledger["constructor_entered"] = False
        raise
    t0 = time.perf_counter()
    constructor_error = None
    try:
        solver = solver_factory(problem["P"], problem["linear"],
            problem["A"], problem["b"], [clarabel.NonnegativeConeT(11520)],
            settings)
        ledger["constructor_completed"] = True
    except BaseException as error:
        constructor_error = error
        raise
    finally:
        ledger["constructor_wall_s"] = time.perf_counter() - t0
        try:
            save_json(Path(out) / "ledger.json", ledger)
        except BaseException as error:
            ledger.setdefault("secondary_errors", []).append({
                "stage": "constructor-ledger", "message": str(error)})
            if constructor_error is None:
                raise
    ensure_solve_budget(elapsed)
    ledger["solver_calls"] = 1
    ledger["solve_called"] = True
    try:
        save_json(Path(out) / "ledger.json", ledger)
    except BaseException:
        ledger["solver_calls"] = 0
        ledger["solve_called"] = False
        raise
    t1 = time.perf_counter()
    solve_error = None
    returned_ledger_error = None
    try:
        solution = solver.solve()
        ledger["solver_returned"] = True
    except BaseException as error:
        solve_error = error
        raise
    finally:
        ledger["solver_wall_s"] = time.perf_counter() - t1
        try:
            save_json(Path(out) / "ledger.json", ledger)
        except BaseException as error:
            ledger.setdefault("secondary_errors", []).append({
                "stage": "solve-ledger", "message": str(error)})
            if solve_error is None:
                returned_ledger_error = error
    # A returned solution remains evidence even when the post-return receipt
    # fails. Attempt arrays and info independently, then re-raise first error.
    primary = returned_ledger_error
    raw = {}
    for key, attribute in (("solver_x", "x"), ("solver_s", "s"),
                           ("solver_z", "z")):
        try:
            raw[key] = np.asarray(getattr(solution, attribute), np.float64)
        except BaseException as error:
            ledger.setdefault("secondary_errors", []).append({
                "stage": "raw-readback-"+key, "message": str(error)})
            if primary is None:
                primary = error
    if raw:
        try:
            save_arrays(Path(out) / "solver-raw.npz", raw)
        except BaseException as error:
            ledger.setdefault("secondary_errors", []).append({
                "stage": "raw-storage", "message": str(error)})
            if primary is None:
                primary = error
    info = {}
    for key, attribute, caster in (
        ("status", "status", str), ("iterations", "iterations", int),
        ("objective_primal", "obj_val", float),
        ("objective_dual", "obj_val_dual", float),
        ("primal_residual", "r_prim", float),
        ("dual_residual", "r_dual", float),
        ("solve_time_s", "solve_time", float)):
        try:
            info[key] = caster(getattr(solution, attribute))
        except BaseException as error:
            ledger.setdefault("secondary_errors", []).append({
                "stage": "info-readback-"+key, "message": str(error)})
            if primary is None:
                primary = error
    info["process_call_wall_s"] = time.perf_counter()-t1
    try:
        save_json(Path(out) / "solver-info.json", info)
    except BaseException as error:
        ledger.setdefault("secondary_errors", []).append({
            "stage": "info-storage", "message": str(error)})
        if primary is None:
            primary = error
    if primary is not None:
        raise primary
    return raw, info


def assess_solver(problem, raw, info, numerical):
    """Report KKT/box diagnostics without changing the returned raw solution."""
    x, s, z = (raw[key] for key in ("solver_x", "solver_s", "solver_z"))
    if (x.shape != (5760,) or s.shape != (11520,) or z.shape != (11520,)
            or not all(np.isfinite(value).all() for value in (x, s, z))):
        raise ValueError("E242 raw solver vector shape/nonfinite failure")
    primal = float(np.max(np.abs(problem["A"] @ x + s - problem["b"])))
    stationarity = float(np.max(np.abs(symmetric_p_matvec(problem["P"], x) +
        problem["linear"] + problem["A"].T @ z)))
    nonnegative = max(0., -float(min(np.min(s), np.min(z))))
    complementarity = float(np.max(np.abs(s*z)))
    box = max(0., -float(np.min(x)), float(np.max(x))-1.)
    objective_without_constant = objective_value(problem, x) - problem["objective_constant"]
    objective_error = abs(objective_without_constant - info["objective_primal"])
    checks = {"primal_residual_inf": primal, "stationarity_inf": stationarity,
        "nonnegative_violation": nonnegative,
        "complementarity_max_abs": complementarity,
        "raw_box_violation": box,
        "recomputed_objective": objective_without_constant,
        "objective_absolute_error": float(objective_error),
        "objective_with_constant": objective_value(problem, x)}
    finite_info = all(np.isfinite(info[key]) for key in
        ("objective_primal", "objective_dual", "primal_residual",
         "dual_residual", "solve_time_s", "process_call_wall_s"))
    valid = (info["status"] == "Solved" and finite_info
        and box <= numerical["raw_box_tolerance"]
        and primal <= numerical["primal_residual_inf"]
        and stationarity <= numerical["stationarity_inf"]
        and nonnegative <= numerical["nonnegative_tolerance"]
        and complementarity <= numerical["complementarity_max_abs"]
        and objective_error <= numerical["objective_atol"] +
            numerical["objective_rtol"]*abs(info["objective_primal"]))
    return checks, valid


def beta_from_x(x, by_leg):
    x = np.asarray(x, np.float64)
    if x.shape != (5760,) or not np.isfinite(x).all():
        raise ValueError("E242 beta vector invalid")
    beta = np.zeros((96, 64), np.float64)
    cursor = 0
    for columns in by_leg:
        width = len(columns)*64
        beta[columns] = x[cursor:cursor+width].reshape(len(columns), 64)
        cursor += width
    if cursor != 5760:
        raise ValueError("E242 group column coverage changed")
    return beta


def candidate_readout(beta, arrays, groups):
    """Preserve original fixed48 and native float32 cast, plus 64-bit control."""
    B, H = arrays["B"], arrays["H"]
    member, cap = arrays["membership"], arrays["cap"]
    free, fixed = arrays["free_columns"], arrays["fixed_columns"]
    original = arrays["motor_rate32"]
    if (beta.shape != (96, 64) or B.shape != (2001, 391, 64)
            or H.shape != (2001, 391) or original.shape != (2001, 391)
            or original.dtype != np.float32 or member.shape != (391,)
            or cap.shape != (391,) or free.shape != (343,)
            or fixed.shape != (48,)):
        raise ValueError("E242 complete original grouped readout required")
    rate64, rate32 = assemble_rates(B, H, beta, member, cap,
                                    free, fixed, original)
    kwargs = (arrays["pool"], arrays["moment"], arrays["rest"],
              arrays["ltm"], groups, arrays["activation0"], arrays["held0"])
    read64 = grouped.canonical_group_readout(rate64, *kwargs)
    read32 = grouped.canonical_group_readout(rate32.astype(np.float64), *kwargs)
    delta = np.sum(arrays["D_group"]*beta[None], axis=2)
    affine = arrays["q_intercept"] + .02*(delta @ arrays["moment"].T)
    affine[0] = arrays["held0"]
    return {"beta": beta, "rate64": rate64, "decoder_input32": rate32,
            "q64": read64["command"], "q32": read32["command"],
            "activation64": read64["activation"],
            "activation32": read32["activation"],
            "tendon64": read64["tendon"], "tendon32": read32["tendon"],
            "affine_q": affine}


def validate_candidate(candidate, arrays, original_tendon, numerical):
    """Check complete raw decoder paths before any candidate verdict."""
    expected = {"beta": (96, 64), "rate64": (2001, 391),
        "decoder_input32": (2001, 391), "q64": (2001, 42),
        "q32": (2001, 42), "activation64": (2001, 96),
        "activation32": (2001, 96), "tendon64": (2001, 6),
        "tendon32": (2001, 6), "affine_q": (2001, 42)}
    if (set(candidate) != set(expected) or any(
            candidate[key].shape != shape or not np.isfinite(candidate[key]).all()
            for key, shape in expected.items())
            or candidate["decoder_input32"].dtype != np.float32):
        raise ValueError("E242 candidate array shape/dtype/nonfinite failure")
    fixed = arrays["fixed_columns"]
    ltm = arrays["ltm"]
    if (not np.array_equal(candidate["beta"][ltm], np.zeros((6, 64)))
            or not np.array_equal(candidate["rate64"][:, fixed],
                arrays["motor_rate32"][:, fixed].astype(np.float64))
            or not np.array_equal(candidate["decoder_input32"][:, fixed],
                arrays["motor_rate32"][:, fixed])
            or any(not np.array_equal(candidate[key][0], arrays["held0"])
                   for key in ("q64", "q32", "affine_q"))
            or any(not np.array_equal(candidate[key][0], arrays["activation0"])
                   for key in ("activation64", "activation32"))):
        raise ValueError("E242 held row/fixed-MN/LTM identity failure")
    original_tendon = np.asarray(original_tendon, np.float64)
    if original_tendon.shape != (2001, 6):
        raise ValueError("E242 original six-tendon layout changed")
    tendon_errors = {key: float(np.max(np.abs(candidate[key] - original_tendon)))
                     for key in ("tendon64", "tendon32")}
    if any(value > numerical["tendon_atol"] for value in tendon_errors.values()):
        raise ValueError("E242 original fixed tendon changed")
    return tendon_errors


def assemble_rates(B, H, beta, member, cap, free, fixed, original):
    """Compute free64, then the actual free32/fixed-native32 decoder input."""
    if (B.ndim != 3 or H.shape != B.shape[:2] or B.shape[2] != beta.shape[1]
            or member.shape != (B.shape[1],) or cap.shape != member.shape
            or original.shape != H.shape or original.dtype != np.float32
            or not np.array_equal(np.sort(np.r_[free, fixed]),
                                  np.arange(B.shape[1]))):
        raise ValueError("E242 rate ownership or shape invalid")
    rate64 = H.copy()
    for start in range(0, len(H), 32):
        stop = min(start+32, len(H))
        rate64[start:stop, free] += cap[free][None] * np.sum(
            B[start:stop, free] * beta[member[free]][None], axis=2)
    rate64[:, fixed] = original[:, fixed].astype(np.float64)
    rate32 = rate64.astype(np.float32)
    rate32[:, fixed] = original[:, fixed]
    return rate64, rate32


def command_metrics(command, target):
    if np.shape(command) != (2001, 42) or np.shape(target) != (2001, 42):
        raise ValueError("E242 complete candidate and teacher required")
    error = np.asarray(command-target, np.float64)
    velocity = np.diff(error, axis=0)/.005
    metrics = {"global": {}, "legs": {}, "startup": {
        "q_rms_rad": rms(error[1:100]),
        "q_max_abs_rad": float(np.max(np.abs(error[1:100])))}}
    for label, lo, hi in (("train", 1, 1201), ("validation", 1201, 2001)):
        metrics["global"][label] = {
            "q_rms_rad": rms(error[lo:hi]),
            "q_max_abs_rad": float(np.max(np.abs(error[lo:hi]))),
            "velocity_rms_rad_s": rms(velocity[lo-1:hi-1]),
            "velocity_max_abs_rad_s": float(np.max(np.abs(velocity[lo-1:hi-1])))}
    for i, leg in enumerate(LEGS):
        sl = slice(i*7, (i+1)*7)
        metrics["legs"][leg] = {}
        for label, lo, hi in (("train", 1, 1201), ("validation", 1201, 2001)):
            metrics["legs"][leg][label] = {
                "q_rms_rad": rms(error[lo:hi, sl]),
                "q_max_abs_rad": float(np.max(np.abs(error[lo:hi, sl]))),
                "velocity_rms_rad_s": rms(velocity[lo-1:hi-1, sl]),
                "velocity_max_abs_rad_s": float(np.max(np.abs(
                    velocity[lo-1:hi-1, sl])))}
    return metrics


def eligibility(candidate, baseline, limits):
    reasons = []
    for phase in ("train", "validation"):
        for metric, key in (("q_rms_rad", "global_q_rms_fraction_train_and_validation"),
                            ("velocity_rms_rad_s", "global_velocity_rms_fraction_train_and_validation")):
            if candidate["global"][phase][metric] > limits[key]*baseline["global"][phase][metric]:
                reasons.append("global."+phase+"."+metric)
        for leg in LEGS:
            for metric, key in (("q_rms_rad", "each_leg_q_rms_fraction_train_and_validation"),
                                ("velocity_rms_rad_s", "each_leg_velocity_rms_fraction_train_and_validation")):
                if candidate["legs"][leg][phase][metric] > limits[key]*baseline["legs"][leg][phase][metric]:
                    reasons.append(leg+"."+phase+"."+metric)
    if candidate["startup"]["q_rms_rad"] > limits["startup_q_rms_fraction"]*baseline["startup"]["q_rms_rad"]:
        reasons.append("startup.q_rms_rad")
    if candidate["startup"]["q_max_abs_rad"] > limits["startup_q_max_fraction"]*baseline["startup"]["q_max_abs_rad"]:
        reasons.append("startup.q_max_abs_rad")
    return {"eligible": not reasons, "reasons": reasons}


def manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in ("result.json", "output-manifest.json"):
            files[path.relative_to(out).as_posix()] = {
                "sha256": prepared.prior.sha256(path), "bytes": path.stat().st_size}
    value = {"schema": "flylab.m3.e242.matched-teacher-fit.manifest.v1",
             "files": files, "total_bytes": sum(x["bytes"] for x in files.values())}
    if value["total_bytes"] > 402653184:
        raise OSError("E242 frozen 384 MiB output limit exceeded")
    save_json(Path(out)/"output-manifest.json", value)
    return value


def finish_receipts(out, result, ledger, primary, elapsed):
    """Publish ledger -> manifest -> result, settling any late failure again."""
    out = Path(out)
    errors = ledger.setdefault("errors", [])
    if errors and primary is None:
        primary = RuntimeError("E242 prior scientific or storage error")

    def note(label, error):
        nonlocal primary
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error

    def try_write(label, callback):
        try:
            callback()
        except BaseException as error:
            note(label, error)

    def deadline():
        if elapsed() > INTERNAL_S and not any(
                item["stage"] == "publication-deadline" for item in errors):
            note("publication-deadline", TimeoutError(
                "E242 internal deadline including publication"))

    result["finalization_pending"] = True
    # A newly discovered error changes the ledger bytes. Republish from the
    # ledger so the final manifest hashes the latest receipt. This is bounded.
    for _ in range(4):
        before = len(errors)
        deadline()
        ledger["process_wall_s"] = elapsed()
        try_write("ledger", lambda: save_json(out/"ledger.json", ledger))
        try_write("manifest", lambda: manifest(out))
        deadline()
        if primary is not None or errors:
            result["validity"] = "incomplete_or_invalid"
            result["candidate_eligible"] = False
            verdict = result.get("metrics", {}).get("eligibility")
            if isinstance(verdict, dict):
                verdict["eligible"] = False
                if "publication_or_technical_failure" not in verdict["reasons"]:
                    verdict["reasons"].append("publication_or_technical_failure")
        result["errors"] = errors
        result["primary_error"] = (None if primary is None else
            {"type": type(primary).__name__, "message": str(primary)})
        result["process_wall_s"] = elapsed()
        result["finalization_pending"] = False
        try_write("result", lambda: save_json(out/"result.json", result))
        deadline()
        if len(errors) == before:
            break
    else:
        # A persistent storage error may prevent a complete manifest. Never
        # advertise a candidate; retain the first error and a best-effort log.
        result["validity"] = "incomplete_or_invalid"
        result["candidate_eligible"] = False
        result["finalization_pending"] = True
        result["errors"] = errors
        result["primary_error"] = (None if primary is None else
            {"type": type(primary).__name__, "message": str(primary)})
        try:
            save_json(out/"result.json", result)
        except BaseException as error:
            errors.append({"stage": "terminal-result", "type": type(error).__name__,
                           "message": str(error)})
            if primary is None:
                primary = error
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, paths = gate(freeze_path, freeze_sha, review_path, review_sha, out)
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e242.matched-teacher-fit.ledger.v1",
        "phase": "gate", "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0,
        "constructor_wall_s": 0., "solver_wall_s": 0.,
        "full_banc_calls": 0, "cuda_calls": 0, "physics_steps": 0,
        "body_constructors": 0, "retries": 0, "errors": []}
    result = {"schema": "flylab.m3.e242.matched-teacher-fit.result.v1",
        "validity": "incomplete_or_invalid", "candidate_eligible": False,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    def elapsed():
        return time.perf_counter()-began
    def budget(label):
        if elapsed() >= INTERNAL_S-RESERVE_S:
            raise TimeoutError("E242 finalization reserve before "+label)
        import psutil
        mem = psutil.Process().memory_info()
        ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
            getattr(mem, "peak_wset", mem.rss))
        if ledger["peak_resident_bytes"] > freeze["resource_limits"]["resident_bytes"]:
            raise MemoryError("E242 resident cap exceeded")
    try:
        save_json(out/"provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha,
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"]})
        save_json(out/"threading.json", configure_single_thread_runtime())
        budget("input load")
        ledger["phase"] = "load"
        p = load_npz(paths["parameters"], ("motor_ids", "membership", "pool",
            "moment", "rest", "ltm", "initial_rate", "tau", "cap",
            "inverse_adaptation_tau", "free_columns", "fixed_columns"))
        basis = load_npz(paths["basis"], ("B", "H"))
        affine = load_npz(paths["affine"], ("D_group", "q_intercept"))
        targets = load_npz(paths["targets"], ("target_command", "motor_rate32",
            "control_tick", "train_controls", "validation_controls",
            "train_velocity_start", "train_velocity_end",
            "validation_velocity_start", "validation_velocity_end",
            "decoded_tendon"))
        canonical = load_npz(paths["canonical_original"],
            ("command", "activation", "tendon"))
        groups_json = json.loads(paths["groups"].read_text("utf-8"))
        groups = groups_json["groups"]
        if (p["inverse_adaptation_tau"].dtype != np.float64
                or not np.array_equal(p["inverse_adaptation_tau"],
                    p["inverse_adaptation_tau"].astype(np.float32).astype(np.float64))
                or not np.array_equal(targets["control_tick"], np.arange(2001))
                or not np.array_equal(targets["train_controls"], np.arange(1,1201))
                or not np.array_equal(targets["validation_controls"], np.arange(1201,2001))
                or not np.array_equal(targets["train_velocity_start"], np.arange(1200))
                or not np.array_equal(targets["train_velocity_end"], np.arange(1,1201))
                or not np.array_equal(targets["validation_velocity_start"], np.arange(1200,2000))
                or not np.array_equal(targets["validation_velocity_end"], np.arange(1201,2001))
                or basis["B"].shape != (2001,391,64)
                or basis["H"].shape != (2001,391)
                or affine["D_group"].shape != (2001,96,64)
                or affine["q_intercept"].shape != (2001,42)
                or targets["target_command"].shape != (2001,42)
                or targets["motor_rate32"].shape != (2001,391)
                or targets["motor_rate32"].dtype != np.float32
                or canonical["activation"].shape != (2001,96)
                or canonical["command"].shape != (2001,42)
                or not np.array_equal(targets["target_command"][0],
                                      canonical["command"][0])):
            raise ValueError("E242 E241 saved split/provenance/layout changed")
        by_leg = grouped.group_anatomy(p["membership"], p["pool"],
            p["moment"], p["ltm"], groups, p["free_columns"],
            p["fixed_columns"])
        scales = training_scales(canonical["command"],
                                 targets["target_command"], freeze["normalization"])
        save_json(out/"normalization.json", scales)
        budget("QP construction")
        ledger["phase"] = "build_qp"
        problem = build_qp(affine["D_group"], affine["q_intercept"],
            targets["target_command"], p["moment"], by_leg, scales)
        save_problem(out, problem, scales)
        save_json(out/"ledger.json", ledger)
        budget("one solver")
        ledger["phase"] = "solve"
        raw, info = solve_once(problem, out, ledger, elapsed)
        ledger["phase"] = "assess"
        checks, solver_valid = assess_solver(problem, raw, info, freeze["numerical"])
        save_json(out/"solver-checks.json", checks)
        result["solver"] = {"status": info["status"], "checks": checks,
                            "valid_for_candidate": solver_valid}
        if not solver_valid:
            result["validity"] = "bounded_unresolved_or_numerically_invalid"
        else:
            budget("candidate readout")
            raw_x = raw["solver_x"]
            projected_x = np.clip(raw_x, 0., 1.)
            projection = float(np.max(np.abs(raw_x-projected_x)))
            arrays = {**p, **basis, **affine,
                "motor_rate32": targets["motor_rate32"],
                "activation0": canonical["activation"][0],
                "held0": canonical["command"][0]}
            raw_candidate = candidate_readout(beta_from_x(raw_x, by_leg),
                arrays, groups)
            raw_metrics = command_metrics(raw_candidate["q32"],
                targets["target_command"])
            save_arrays(out/"candidate-raw.npz", raw_candidate)
            raw_tendon_errors = validate_candidate(raw_candidate, arrays,
                canonical["tendon"], freeze["numerical"])
            candidate = (raw_candidate if projection == 0 else
                candidate_readout(beta_from_x(projected_x, by_leg), arrays, groups))
            save_arrays(out/"candidate.npz", candidate)
            tendon_errors = validate_candidate(candidate, arrays,
                canonical["tendon"], freeze["numerical"])
            original_metrics = command_metrics(canonical["command"],
                targets["target_command"])
            projected_metrics = command_metrics(candidate["q32"],
                targets["target_command"])
            verdict = eligibility(projected_metrics, original_metrics,
                                   freeze["eligibility"])
            numerical = freeze["numerical"]
            affine_error = float(np.max(np.abs(candidate["affine_q"][1:] -
                                                candidate["q64"][1:])))
            affine_pass = np.allclose(candidate["affine_q"], candidate["q64"],
                atol=numerical["affine_atol"], rtol=numerical["affine_rtol"])
            if not affine_pass:
                verdict["reasons"].append("affine_direct_mismatch")
            verdict["eligible"] = not verdict["reasons"]
            metrics = {"original": original_metrics, "raw": raw_metrics,
                "projected": projected_metrics, "eligibility": verdict,
                "raw_objective": objective_value(problem, raw_x),
                "projected_objective": objective_value(problem, projected_x),
                "projection_max_abs": projection,
                "affine_q_max_abs": affine_error,
                "raw_tendon_max_abs": raw_tendon_errors,
                "tendon_max_abs": tendon_errors,
                "q64_q32_max_abs": float(np.max(np.abs(
                    candidate["q64"] - candidate["q32"])))}
            save_json(out/"metrics.json", metrics)
            result.update(validity="valid_conditional_matched_teacher_fit",
                candidate_eligible=verdict["eligible"], metrics=metrics)
    except BaseException as error:
        primary = error
        ledger["errors"].append({"stage": ledger["phase"],
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    if elapsed() > INTERNAL_S and primary is None:
        primary = TimeoutError("E242 internal deadline")
    primary = finish_receipts(out, result, ledger, primary, elapsed)
    if primary is not None:
        raise primary
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.out)
    print(json.dumps({"validity": result["validity"],
                      "candidate_eligible": result["candidate_eligible"]}))


if __name__ == "__main__":
    main()
