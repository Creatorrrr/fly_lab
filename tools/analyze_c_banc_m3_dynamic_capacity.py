"""E215 one bounded LM dynamic-capacity diagnostic in original muscle units.

The time-varying innovation is deliberately freer than the frozen neural
response. A feasible result is a conditional LM witness, never a gait or
neural-controller candidate.
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

from tools.fit_c_banc_m3_constrained_tangent import csc_arrays
from tools.fit_c_banc_m3_local_response import load_targets
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "70707ebcd027cf4639f7daee99b045a3062907a71ef4a487921323b7d5d290ea"
FREEZE_SCHEMA = "flylab.m3.e215.lm-dynamic-capacity.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e215.lm-dynamic-capacity.review.v1"
OWNED = ("tools/analyze_c_banc_m3_dynamic_capacity.py",
         "tests/test_c_banc_dynamic_capacity.py")
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 1, "retries": 0, "internal_s": 150,
    "solver_s": 90, "external_s": 180, "supervisor_s": 200,
    "resident_bytes": 8589934592, "output_bytes": 268435456,
    "minimum_free_bytes": 4294967296}
EPS = 2e-12
PHYSICAL_ATOL = 1e-8
FORWARD_ATOL = 2e-10
DT = .00025
CONTROL_DT = .005
ALPHA = -np.expm1(-CONTROL_DT / .01)
DECAY = 1 - ALPHA


def repository_path(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E215 repository-relative slash file required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E215 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E215 frozen file missing or escaped")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E215 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E215 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    shape = {"leg": "lm", "leg_index": 1, "motors": 391,
        "free_motors": 343, "leg_free_motors": 54,
        "leg_free_muscles": 14, "variables": 26922,
        "samples": 2001, "selected_samples": 1923,
        "startup_rows": 99, "main_rows": 1824,
        "velocity_rows": 1823, "cones": 3}
    solver = {"name": "clarabel", "max_threads": 1,
        "max_iter": 150, "time_limit_s": 90.0,
        "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8, "PAD": 1.0, "ridge": 0.0,
        "raw_clip": False, "converged_status": "Solved"}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E215_LM_DYNAMIC_CAPACITY_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != shape or freeze.get("solver") != solver
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("clock") != {"neural_dt_s": DT,
                "control_dt_s": CONTROL_DT, "ticks_per_control": 20,
                "activation_tau_s": .01}
            or freeze.get("tolerance") != {
                "innovation_envelope_epsilon": EPS,
                "physical_constraint_atol": PHYSICAL_ATOL,
                "forward_readout_atol": FORWARD_ATOL}
            or freeze.get("generated_solver") != {
                "actual_calls_max": 1, "external_s": 30,
                "original_data": False}
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E215 scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E215 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E215 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E215 parent source review does not authorize files")
    for label, count in (("input_sha256", 13),
                         ("reused_source_sha256", 26),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E215 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E215 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E215 hash gate exceeded internal deadline")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "parameters", "trace", "targets", "mask", "basis", "affine",
            "basis_parameters"}:
        raise ValueError("E215 original input map changed")
    return freeze, review, {key: repository_path(value)
                            for key, value in paths.items()}


def rk4_decay(tau):
    """Twenty exact scalar RK4 homogeneous steps at the frozen neural dt."""
    tau = np.asarray(tau, np.float64)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("E215 positive finite original tau required")
    h = DT / tau
    return (1 - h + h*h/2 - h*h*h/6 + h*h*h*h/24) ** 20


def rk4_stage_input_coefficients(tau):
    """Unscaled positive independent stage-to-step weights, including leak."""
    h = DT / np.asarray(tau, np.float64)
    return DT / 6 * np.stack((1 - h + h*h/2 - h*h*h/4,
        2 - h + h*h/2, 2 - h, np.ones_like(h)), axis=-1)


def derive_lm_inputs(params, mask, basis, affine):
    """Pinned E150 anatomy and actual E213 mass, with no model execution."""
    membership = np.asarray(params["membership"], np.int64)
    pool = np.asarray(params["pool"], np.float64)
    moment = np.asarray(params["moment"], np.float64)
    cap = np.asarray(params["cap"], np.float64)
    tau = np.asarray(params["tau"], np.float64)
    initial = np.asarray(params["initial_rate"], np.float64)
    motor_ids = np.asarray(params["motor_ids"], np.int64)
    free = np.asarray(mask["free_columns"], np.int64)
    fixed = np.asarray(mask["fixed_columns"], np.int64)
    B = np.asarray(basis["B"], np.float64)
    q0 = np.asarray(affine["q_intercept"], np.float64)
    groups = json.loads(str(params["muscle_parameters_json"]))
    if (membership.shape != (391,) or pool.shape != (96, 391)
            or moment.shape != (42, 96) or cap.shape != (391,)
            or tau.shape != (391,) or initial.shape != (391,)
            or motor_ids.shape != (391,) or free.shape != (343,)
            or fixed.shape != (48,) or B.shape != (2001, 391, 16)
            or q0.shape != (2001, 42) or len(groups) != 96
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
            or not np.array_equal(mask["membership"], membership)
            or not np.array_equal(mask["motor_ids"], motor_ids)
            or np.any((membership < 0) | (membership >= 96))
            or np.any(cap <= 0) or np.any(tau <= 0)
            or not all(np.isfinite(v).all() for v in (
                pool, moment, cap, tau, initial, B, q0))):
        raise ValueError("E215 original motor/anatomical identity changed")
    if np.any(pool < 0) or np.any(B[0, free] != 0) or np.any(initial[free] != 0):
        raise ValueError("E215 free initial state or nonnegative pool changed")
    for motor, owner in enumerate(membership):
        if np.any(pool[np.arange(96) != owner, motor] != 0):
            raise ValueError("E215 motor has cross-muscle pool weight")
    muscles = np.asarray([g for g in np.unique(membership[free])
        if groups[int(g)]["leg"] == "lm"], np.int64)
    motors = np.asarray([m for g in muscles for m in free
                         if membership[m] == g], np.int64)
    if len(muscles) != 14 or len(motors) != 54 or not np.array_equal(
            np.unique(membership[motors]), muscles):
        raise ValueError("E215 LM14/54 anatomical ownership changed")
    if any(np.any(moment[np.r_[0:7, 14:42], g] != 0) for g in muscles):
        raise ValueError("E215 LM muscle crosses leg moment rows")
    F = np.empty(14, np.float64)
    G = np.empty(14, np.float64)
    mass = np.sum(B[:1924, motors], axis=2)
    J = np.empty((1923, 54), np.float64)
    S = np.empty((1923, 14), np.float64)
    for j, muscle in enumerate(muscles):
        own = np.flatnonzero(membership[motors] == muscle)
        local_tau = tau[motors[own]]
        if np.any(local_tau != local_tau[0]):
            raise ValueError("E215 unequal tau inside one LM muscle")
        F[j] = float(rk4_decay(local_tau[0]))
        weights = pool[muscle, motors[own]]
        G[j] = float(weights @ cap[motors[own]])
        if not np.isfinite(G[j]) or G[j] <= 0:
            raise ValueError("E215 nonpositive pooled rate capacity")
        J[:, own] = cap[motors[own]][None] * (
            mass[1:, own] - F[j] * mass[:-1, own])
        S[:, j] = (J[:, own] @ weights) / G[j]
    L = .02 * moment[7:14, muscles] * G[None]
    return {"muscles": muscles, "motors": motors, "motor_ids": motor_ids[motors],
        "membership": membership[motors], "pool_owned": pool[membership[motors], motors],
        "cap": cap[motors], "tau": tau[motors], "F": F, "G": G,
        "mass": mass, "J": J, "S": S, "L": L,
        "q0": q0[:, 7:14], "free": free, "fixed": fixed}


def dynamic_operator(F, frames, *, alpha=ALPHA, decay=DECAY):
    """time-major x -> pooled-rate innovation forcing."""
    F = np.asarray(F, np.float64)
    groups = len(F)
    if (groups == 0 or frames < 1 or np.any(~np.isfinite(F))
            or np.any((F < 0) | (F > 1)) or not 0 < alpha < 1
            or not 0 < decay < 1):
        raise ValueError("E215 stable finite scalar filter required")
    n = frames * groups
    diagonals = [np.ones(n)]
    offsets = [0]
    if frames > 1:
        diagonals.append(np.tile(-(decay + F), frames - 1))
        offsets.append(-groups)
    if frames > 2:
        diagonals.append(np.tile(decay * F, frames - 2))
        offsets.append(-2 * groups)
    return sparse.diags(diagonals, offsets, shape=(n, n), format="csc") / alpha


def _frame_rows(frames, groups):
    frames = np.asarray(frames, np.int64)
    return ((frames[:, None] - 1) * groups + np.arange(groups)).reshape(-1)


def build_capacity_problem(data, target, main_frames, startup_frames,
                           pair_start, pair_end, command_limit, velocity_limit,
                           *, alpha=ALPHA, decay=DECAY, epsilon=EPS,
                           startup_limit=.04, control_dt=CONTROL_DT):
    """One sparse QP with NN [-T,T,-I,I,startup,-startup], qSOC, vSOC."""
    F = np.asarray(data["F"], np.float64)
    S = np.asarray(data["S"], np.float64)
    L = np.asarray(data["L"], np.float64)
    q0 = np.asarray(data["q0"], np.float64)
    nframes, ngroups = S.shape
    qdim = L.shape[0]
    main = np.asarray(main_frames, np.int64)
    startup = np.asarray(startup_frames, np.int64)
    starts = np.asarray(pair_start, np.int64)
    ends = np.asarray(pair_end, np.int64)
    for field in ("target", "startup_target", "velocity_target"):
        if field not in target:
            raise ValueError("E215 target field missing: " + field)
    tq = np.asarray(target["target"], np.float64)
    ts = np.asarray(target["startup_target"], np.float64)
    tv = np.asarray(target["velocity_target"], np.float64)
    if (nframes < 1 or ngroups < 1 or F.shape != (ngroups,)
            or L.shape[1] != ngroups or q0.shape != (nframes + 1, qdim)
            or main.ndim != 1 or len(main) < 1 or startup.ndim != 1
            or len(startup) < 1 or starts.shape != ends.shape
            or tq.shape != (len(main), qdim)
            or ts.shape != (len(startup), qdim)
            or tv.shape != (len(starts), qdim)
            or np.any((main < 1) | (main > nframes))
            or np.any((startup < 1) | (startup > nframes))
            or np.any((starts < 1) | (starts > nframes))
            or np.any((ends < 1) | (ends > nframes))
            or np.any(ends <= starts)
            or not np.isfinite(command_limit) or command_limit <= 0
            or not np.isfinite(velocity_limit) or velocity_limit <= 0
            or not np.isfinite(control_dt) or control_dt <= 0
            or not np.isfinite(epsilon) or epsilon < 0
            or not np.isfinite(startup_limit) or startup_limit <= 0
            or not all(np.isfinite(v).all() for v in (S, L, q0, tq, ts, tv))):
        raise ValueError("E215 aligned finite physical sample/target contract changed")
    if np.any(S < -epsilon):
        raise ValueError("E215 innovation envelope materially negative")
    n = nframes * ngroups
    T = dynamic_operator(F, nframes, alpha=alpha, decay=decay)
    Q = sparse.kron(sparse.eye(nframes, format="csc"), sparse.csc_matrix(L),
                    format="csr")
    qmap = Q[_frame_rows(main, qdim)].tocsc()
    smap = Q[_frame_rows(startup, qdim)].tocsc()
    vmap = (Q[_frame_rows(ends, qdim)]
        - Q[_frame_rows(starts, qdim)]).tocsc() / control_dt
    qoffset = (q0[main] - tq).reshape(-1)
    soffset = (q0[startup] - ts).reshape(-1)
    voffset = ((q0[ends] - q0[starts]) / control_dt - tv).reshape(-1)
    identity = sparse.eye(n, format="csc")
    nn = sparse.vstack((-T, T, -identity, identity,
                        smap, -smap), format="csc")
    nn_rhs = np.r_[np.full(n, epsilon), S.reshape(-1) + epsilon,
                   np.zeros(n), np.ones(n),
                   startup_limit - soffset, startup_limit + soffset]
    qnorm = np.sqrt(len(qoffset)) * command_limit
    vnorm = np.sqrt(len(voffset)) * velocity_limit if len(voffset) else 1.
    qscaled = qmap / qnorm
    vscaled = vmap / vnorm
    q_soc = sparse.vstack((sparse.csc_matrix((1, n)), -qscaled), format="csc")
    v_soc = sparse.vstack((sparse.csc_matrix((1, n)), -vscaled), format="csc")
    q_rhs = np.r_[1., qoffset / qnorm]
    v_rhs = np.r_[1., voffset / vnorm]
    A = sparse.vstack((nn, q_soc, v_soc), format="csc")
    b = np.r_[nn_rhs, q_rhs, v_rhs]
    gram = qscaled.T @ qscaled
    P = sparse.triu(2 * gram, format="csc")
    linear = np.asarray(2 * qscaled.T @ (qoffset / qnorm)).reshape(-1)
    constant = float((qoffset / qnorm) @ (qoffset / qnorm))
    dimensions = np.asarray([len(nn_rhs), len(q_rhs), len(v_rhs)], np.int64)
    kinds = ["NonnegativeConeT", "SecondOrderConeT", "SecondOrderConeT"]
    cones = [clarabel.NonnegativeConeT(int(dimensions[0])),
             clarabel.SecondOrderConeT(int(dimensions[1])),
             clarabel.SecondOrderConeT(int(dimensions[2]))]
    if (A.shape != (int(dimensions.sum()), n) or P.shape != (n, n)
            or not np.isfinite(A.data).all() or not np.isfinite(P.data).all()
            or not np.isfinite(b).all() or not np.isfinite(linear).all()
            or not np.isfinite(constant)):
        raise ValueError("E215 sparse physical problem nonfinite or unaligned")
    return {"P": P, "linear": linear, "objective_constant": constant,
        "A": A, "b": b, "cones": cones, "cone_dimensions": dimensions,
        "cone_types": kinds, "forcing_map": T,
        "command_map": qmap, "velocity_map": vmap,
        "startup_map": smap, "command_offset": qoffset,
        "velocity_offset": voffset, "startup_offset": soffset,
        "main_frames": main, "startup_frames": startup,
        "pair_start": starts, "pair_end": ends,
        "command_limit": float(command_limit),
        "velocity_limit": float(velocity_limit),
        "startup_limit": float(startup_limit), "epsilon": float(epsilon),
        "alpha": float(alpha), "decay": float(decay),
        "control_dt": float(control_dt), "groups": ngroups, "frames": nframes}


def generated_analytic_case():
    """Two-frame one-muscle problem; exact optimum x=[.29,.5]."""
    data = {"F": np.array([.5]), "S": np.full((2, 1), 2.),
            "L": np.array([[1.]]), "q0": np.zeros((3, 1))}
    target = {"target": np.array([[.5], [.5]]),
              "startup_target": np.array([[.25]]),
              "velocity_target": np.array([[0.]]),
              "baseline": {"leg_command_rms": [1., 1.]}}
    problem = build_capacity_problem(data, target, [1, 2], [1], [1], [2],
        1., 100., epsilon=0., startup_limit=.04)
    return problem, data, target, np.array([.29, .5])


def save_physical_inputs(out, data):
    keys = ("muscles", "motors", "motor_ids", "membership", "pool_owned",
        "cap", "tau", "F", "G", "mass", "J", "S", "L", "q0",
        "free", "fixed")
    np.savez(Path(out) / "physical-inputs.npz",
             **{key: data[key] for key in keys})


def save_pre_solve(out, problem, source_sha):
    arrays = {"linear": problem["linear"], "b": problem["b"],
        "objective_constant": np.asarray(problem["objective_constant"]),
        "cone_dimensions": problem["cone_dimensions"],
        "cone_types": np.asarray(problem["cone_types"]),
        "command_offset": problem["command_offset"],
        "velocity_offset": problem["velocity_offset"],
        "startup_offset": problem["startup_offset"],
        "main_frames": problem["main_frames"],
        "startup_frames": problem["startup_frames"],
        "pair_start": problem["pair_start"],
        "pair_end": problem["pair_end"],
        "command_limit": np.asarray(problem["command_limit"]),
        "velocity_limit": np.asarray(problem["velocity_limit"]),
        "startup_limit": np.asarray(problem["startup_limit"]),
        "epsilon": np.asarray(problem["epsilon"]),
        "alpha": np.asarray(problem["alpha"]),
        "decay": np.asarray(problem["decay"]),
        "control_dt": np.asarray(problem["control_dt"]),
        "groups": np.asarray(problem["groups"]),
        "frames": np.asarray(problem["frames"]),
        "source_basis_sha256": np.asarray(source_sha)}
    for name in ("P", "A", "forcing_map", "command_map",
                 "velocity_map", "startup_map"):
        arrays.update(csc_arrays(name, problem[name]))
    np.savez(Path(out) / "pre-solve.npz", **arrays)


def ensure_solve_budget(elapsed):
    if elapsed() + RESOURCE["solver_s"] > RESOURCE["internal_s"]:
        raise SolverAllowanceUnavailable("E215 less than full 90 s solve allowance")


class SolverAllowanceUnavailable(TimeoutError):
    """One original solve cannot start within the frozen internal budget."""


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


def solve_once(problem, out, ledger, *, elapsed, solver_factory=None):
    """One call; requested/called/returned and all raw persist before verdict."""
    out = Path(out)
    if ledger.get("solve_called") or ledger.get("solver_calls", 0):
        raise RuntimeError("E215 one solver call already used")
    if solver_factory is None:
        solver_factory = clarabel.DefaultSolver
    settings = solver_settings()
    save_json(out / "solver-settings.json", {"configured": {
        "max_threads": 1, "max_iter": 150, "time_limit_s": 90.,
        "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8},
        "all_settings_text": str(settings)})
    ensure_solve_budget(elapsed)
    ledger["solve_requested"] = True
    save_json(out / "ledger.json", ledger)
    ledger["constructor_entered"] = True
    began = time.perf_counter()
    constructor_error = None
    try:
        solver = solver_factory(problem["P"], problem["linear"], problem["A"],
                                problem["b"], problem["cones"], settings)
        ledger["constructor_completed"] = True
    except BaseException as error:
        constructor_error = error
        raise
    finally:
        ledger["constructor_wall_s"] = time.perf_counter() - began
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
    begun = time.perf_counter()
    solve_error = None
    try:
        solution = solver.solve()
        ledger["solver_returned"] = True
    except BaseException as error:
        solve_error = error
        raise
    finally:
        ledger["solver_wall_s"] = time.perf_counter() - begun
        try:
            save_json(out / "ledger.json", ledger)
        except BaseException:
            if solve_error is None:
                raise
    raw = {"solver_x": np.asarray(solution.x, np.float64),
           "solver_s": np.asarray(solution.s, np.float64),
           "solver_z": np.asarray(solution.z, np.float64)}
    np.savez(out / "solver-raw.npz", **raw)
    info = {"status": str(solution.status), "iterations": int(solution.iterations),
        "objective_primal": float(solution.obj_val),
        "objective_dual": float(solution.obj_val_dual),
        "primal_residual": float(solution.r_prim),
        "dual_residual": float(solution.r_dual),
        "solve_time_s": float(solution.solve_time),
        "process_call_wall_s": time.perf_counter() - begun}
    save_json(out / "solver-info.json", info)
    ledger["solver_soft_limit_overrun_s"] = max(0.,
        max(ledger["solver_wall_s"], info["solve_time_s"]) - RESOURCE["solver_s"])
    save_json(out / "ledger.json", ledger)
    if (raw["solver_x"].shape != (problem["P"].shape[0],)
            or raw["solver_s"].shape != (problem["A"].shape[0],)
            or raw["solver_z"].shape != (problem["A"].shape[0],)):
        raise ValueError("E215 saved solver raw shape changed")
    return raw, info


def forward_reconstruction(problem, data, x):
    """Independently run F rate and d/alpha activation from inferred forcing."""
    frames, groups = problem["frames"], problem["groups"]
    activation = np.asarray(x, np.float64).reshape(frames, groups)
    force = np.asarray(problem["forcing_map"] @ x).reshape(frames, groups)
    pooled = np.zeros_like(force)
    forward = np.zeros_like(force)
    prior_rate = np.zeros(groups)
    prior_activation = np.zeros(groups)
    for frame in range(frames):
        prior_rate = data["F"] * prior_rate + force[frame]
        prior_activation = problem["decay"] * prior_activation + (
            problem["alpha"] * prior_rate)
        pooled[frame] = prior_rate
        forward[frame] = prior_activation
    q_linear = data["q0"][1:frames+1] + activation @ data["L"].T
    q_forward = data["q0"][1:frames+1] + forward @ data["L"].T
    return {"forcing": force, "pooled_rate": pooled,
        "activation": activation, "forward_activation": forward,
        "q_linear": q_linear, "q_forward": q_forward}


def solved_diagnostic(problem, data, target, raw, info, out):
    """Report strict physical residuals; save reconstruction before checks."""
    x = raw["solver_x"]
    if (not np.isfinite(x).all() or not np.isfinite(raw["solver_s"]).all()
            or not np.isfinite(raw["solver_z"]).all()):
        raise ValueError("E215 Solved return nonfinite")
    restored = forward_reconstruction(problem, data, x)
    np.savez(Path(out) / "diagnostic.npz", solver_x=x, **restored,
        command_residual=(problem["command_map"] @ x
                          + problem["command_offset"]),
        velocity_residual=(problem["velocity_map"] @ x
                           + problem["velocity_offset"]),
        startup_residual=(problem["startup_map"] @ x
                          + problem["startup_offset"]))
    force = restored["forcing"]
    qerr = np.asarray(problem["command_map"] @ x
                      + problem["command_offset"])
    verr = np.asarray(problem["velocity_map"] @ x
                      + problem["velocity_offset"])
    serr = np.asarray(problem["startup_map"] @ x
                      + problem["startup_offset"])
    q_rms = float(np.sqrt(np.mean(qerr * qerr)))
    v_rms = float(np.sqrt(np.mean(verr * verr)))
    s_max = float(np.max(np.abs(serr)))
    f_lower = float(np.min(force + problem["epsilon"]))
    f_upper = float(np.max(force - data["S"] - problem["epsilon"]))
    x_lower = float(np.min(x))
    x_upper = float(np.max(x))
    max_forward = float(np.max(np.abs(restored["q_linear"]
                                      - restored["q_forward"])))
    cone_residual = problem["A"] @ x + raw["solver_s"] - problem["b"]
    equality_max = float(np.max(np.abs(cone_residual)))
    q_baseline = float(target["baseline"]["leg_command_rms"][1])
    report = {"status": info["status"], "feasible_lm_only": False,
        "numeric_tolerance": PHYSICAL_ATOL,
        "forward_tolerance": FORWARD_ATOL,
        "x_min": x_lower, "x_max": x_upper,
        "forcing_lower_margin_min": f_lower,
        "forcing_upper_residual_max": f_upper,
        "startup_q_max": s_max,
        "startup_q_rms": float(np.sqrt(np.mean(serr * serr))),
        "main_command_rms": q_rms,
        "main_command_baseline_rms": q_baseline,
        "main_command_improvement_fraction": 1 - q_rms / q_baseline,
        "main_command_20pct_proxy": bool(q_rms <= .8 * q_baseline),
        "main_velocity_rms": v_rms,
        "main_velocity_baseline_rms": problem["velocity_limit"],
        "forward_q_max_abs": max_forward,
        "primal_equality_max": equality_max,
        "scope": "LM conditional free forcing only; no neural or gait candidate"}
    report["feasible_lm_only"] = bool(
        x_lower >= -PHYSICAL_ATOL and x_upper <= 1 + PHYSICAL_ATOL
        and f_lower >= -PHYSICAL_ATOL and f_upper <= PHYSICAL_ATOL
        and s_max <= problem["startup_limit"] + PHYSICAL_ATOL
        and q_rms <= problem["command_limit"] + PHYSICAL_ATOL
        and v_rms <= problem["velocity_limit"] + PHYSICAL_ATOL
        and max_forward <= FORWARD_ATOL and equality_max <= PHYSICAL_ATOL)
    save_json(Path(out) / "diagnostic.json", report)
    return report


def infeasible_diagnostic(problem, raw, info, out):
    """Independent cone and box-separation statistics from the returned dual."""
    z = raw["solver_z"]
    if not np.isfinite(z).all():
        report = {"status": info["status"], "certificate_verified": False,
                  "reason": "nonfinite dual"}
        save_json(Path(out) / "diagnostic.json", report)
        return report
    az = np.asarray(problem["A"].T @ z).reshape(-1)
    cone = []
    offset = 0
    max_dual_cone_violation = 0.
    for kind, size in zip(problem["cone_types"], problem["cone_dimensions"]):
        piece = z[offset:offset+size]
        violation = (-float(np.min(piece)) if kind == "NonnegativeConeT"
                     else float(np.linalg.norm(piece[1:]) - piece[0]))
        max_dual_cone_violation = max(max_dual_cone_violation, violation)
        cone.append({"type": kind, "dimension": int(size),
                     "b_dot_z": float(problem["b"][offset:offset+size] @ piece),
                     "dual_cone_violation": max(0., violation)})
        offset += size
    btz = float(problem["b"] @ z)
    box_lower = float(np.minimum(az, 0).sum())
    separation = box_lower - btz
    report = {"status": info["status"], "b_dot_z": btz,
        "A_transpose_z_max_abs": float(np.max(np.abs(az))),
        "box_linear_minimum": box_lower,
        "box_separation_margin": separation,
        "dual_cone_violation_max": max_dual_cone_violation,
        "primal_residual_reported": info["primal_residual"],
        "dual_residual_reported": info["dual_residual"],
        "cone_contributions": cone,
        "certificate_verified": bool(np.isfinite(separation)
            and separation > PHYSICAL_ATOL
            # A tiny negative component in each NN dual can sum to a false
            # separation. Keep raw z and measured error; only an exact cone
            # member may certify this uncorrected support-bound inequality.
            and max_dual_cone_violation == 0.
            and np.isfinite(info["dual_residual"])
            and info["dual_residual"] <= PHYSICAL_ATOL),
        "scope": "LM conditional free forcing only; no global impossibility claim"}
    save_json(Path(out) / "diagnostic.json", report)
    return report


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e215.output-manifest.v1", "files": files,
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


def load_original_inputs(paths):
    """Load pinned arrays only after the review gate; construct no models."""
    with (np.load(paths["parameters"], allow_pickle=False) as p,
          np.load(paths["mask"], allow_pickle=False) as mask,
          np.load(paths["basis"], allow_pickle=False) as basis,
          np.load(paths["affine"], allow_pickle=False) as affine,
          np.load(paths["basis_parameters"], allow_pickle=False) as bp):
        for key in ("cap", "tau", "initial_rate", "motor_ids"):
            if not np.array_equal(p[key].astype(np.float64),
                                  bp[key].astype(np.float64)):
                raise ValueError("E215 E201/E213 original motor parameter mismatch: " + key)
        if (not np.array_equal(mask["free_columns"], bp["free_columns"])
                or not np.array_equal(mask["fixed_columns"], bp["fixed_columns"])):
            raise ValueError("E215 E203/E213 free/fixed ownership mismatch")
        data = derive_lm_inputs(p, mask, basis, affine)
        if (int(p["motor_ids"][220]) != 92104
                or int(p["membership"][220]) != 19
                or 220 not in data["motors"]
                or not np.isclose(float(p["tau"][220]),
                                  .014046886935830116, atol=1e-15, rtol=0)
                or np.any(data["tau"][data["motors"] != 220]
                          != .019999999552965164)):
            raise ValueError("E215 actual singleton LM tau lost")
        if not np.array_equal(mask["motor_ids"], p["motor_ids"]):
            raise ValueError("E215 E203/E201 motor ID mismatch")
    return data


def original_target(paths, trace):
    target = load_targets(paths["targets"], trace)
    main = np.asarray(target["indices"], np.int64)
    starts = np.asarray(target["pair_start"], np.int64)
    ends = np.asarray(target["pair_end"], np.int64)
    if (not np.array_equal(main, np.arange(100, 1924))
            or not np.array_equal(starts, main[:-1])
            or not np.array_equal(ends, main[1:])
            or target["target"].shape != (1824, 42)
            or target["velocity"].shape != (1823, 42)
            or target["startup_command"].shape != (99, 42)):
        raise ValueError("E215 frozen original LM target rows changed")
    local = {"target": target["target"][:, 7:14],
             "startup_target": target["startup_command"][:, 7:14],
             "velocity_target": target["velocity"][:, 7:14],
             "baseline": target["baseline"]}
    qlimit = float(target["baseline"]["leg_command_rms"][1])
    vlimit = float(target["baseline"]["leg_velocity_rms"][1])
    return local, main, np.arange(1, 100), starts, ends, qlimit, vlimit


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E215 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = None
    primary = None
    errors = []
    ledger = {"schema": "flylab.m3.e215.ledger.v1",
        "solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0,
        "constructor_wall_s": 0., "solver_wall_s": 0.,
        "full_banc_calls": 0, "cuda_calls": 0, "neural_ticks": 0,
        "physics_steps": 0, "retries": 0}
    result = {"schema": "flylab.m3.e215.lm-dynamic-capacity.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "diagnostic_only": True}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E215 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target, main, startup, starts, ends, qlimit, vlimit = original_target(
                paths, trace)
            data = load_original_inputs(paths)
            save_physical_inputs(out, data)
            # S/J/mass are retained before evaluating the innovation envelope.
            local_data = dict(data, q0=data["q0"][:1924])
            problem = build_capacity_problem(local_data, target, main,
                startup, starts, ends, qlimit, vlimit)
            save_pre_solve(out, problem, freeze["input_sha256"][freeze["paths"]["basis"]])
            save_json(out / "problem-summary.json", {
                "variables": int(problem["P"].shape[0]),
                "rows": int(problem["A"].shape[0]),
                "P_nnz": int(problem["P"].nnz),
                "A_nnz": int(problem["A"].nnz),
                "cone_dimensions": problem["cone_dimensions"].tolist(),
                "cone_types": problem["cone_types"],
                "nonnegative_order": ["-forcing", "+forcing", "-x", "+x",
                                      "+startup", "-startup"],
                "main_frames": [int(main[0]), int(main[-1])],
                "startup_frames": [int(startup[0]), int(startup[-1])]})
            if (problem["P"].shape != (26922, 26922)
                    or problem["cone_dimensions"].shape != (3,)
                    or problem["frames"] != 1923 or problem["groups"] != 14
                    or len(problem["startup_offset"]) != 99 * 7
                    or len(problem["command_offset"]) != 1824 * 7
                    or len(problem["velocity_offset"]) != 1823 * 7):
                raise ValueError("E215 frozen LM sparse problem shape changed")
            ensure_solve_budget(elapsed)
            raw, info = solve_once(problem, out, ledger, elapsed=elapsed)
            result["solver_info"] = info
            if info["status"] == "Solved":
                diagnosis = solved_diagnostic(problem, data, target, raw, info, out)
                result["validity"] = ("valid_conditional_LM_dynamic_feasible"
                    if diagnosis["feasible_lm_only"] else "valid_bounded_unresolved")
            elif info["status"] == "PrimalInfeasible":
                diagnosis = infeasible_diagnostic(problem, raw, info, out)
                result["validity"] = ("valid_conditional_LM_dynamic_infeasible"
                    if diagnosis["certificate_verified"] else "valid_bounded_unresolved")
            else:
                diagnosis = {"status": info["status"],
                    "scope": "LM conditional free forcing only",
                    "reason": "non-Solved/non-certified bounded return"}
                save_json(out / "diagnostic.json", diagnosis)
                result["validity"] = "valid_bounded_unresolved"
            result["diagnostic"] = diagnosis
    except SolverAllowanceUnavailable as error:
        result["validity"] = "valid_bounded_unresolved_no_solver_call"
        result["budget_block"] = str(error)
        ledger["blocked_before_solve"] = True
        ledger["budget_block"] = str(error)
    except BaseException as error:  # noqa: BLE001 - retain raw and primary
        primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    _save_attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"],
        solver_wall_s=ledger["solver_wall_s"],
        solver_soft_limit_overrun_s=ledger.get("solver_soft_limit_overrun_s",
            max(0., ledger["solver_wall_s"] - RESOURCE["solver_s"])),
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
        result["resource_failure"] = "E215 output manifest missing or over 256 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E215 hard internal deadline exceeded"
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
            primary = RuntimeError("E215 raw/publication failure")
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
