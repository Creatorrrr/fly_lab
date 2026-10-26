"""E216: five new bounded leg diagnostics plus the frozen E215 LM witness.

Only a complete set of six physical witnesses is combined. The time-varying
forcing is a relaxed capacity diagnostic, not a neural or body candidate.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import analyze_c_banc_m3_dynamic_capacity as e215
from tools.fit_c_banc_m3_local_response import direct_metrics, feasible, load_targets
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "65ff5010cea1696e49ce293600fa183c565f7e39e093307e1e211def18decadf"
FREEZE_SCHEMA = "flylab.m3.e216.six-leg-dynamic-capacity.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e216.six-leg-dynamic-capacity.review.v1"
OWNED = ("tools/analyze_c_banc_m3_six_leg_capacity.py",
         "tests/test_c_banc_six_leg_capacity.py")
NEW_LEGS = ("lf", "lh", "rf", "rm", "rh")
LEG_MOTORS = (61, 54, 56, 62, 55, 55)
LEG_MUSCLES = (16, 14, 15, 16, 14, 15)
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 5, "retries": 0, "internal_s": 150,
    "solver_s": 90, "external_s": 180, "supervisor_s": 200,
    "resident_bytes": 8589934592, "output_bytes": 268435456,
    "minimum_free_bytes": 4294967296}


def repository_path(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E216 repository-relative slash file required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E216 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("E216 frozen file missing or escaped")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E216 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E216 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    shape = {"leg_order": list(LEGS), "new_legs": list(NEW_LEGS),
        "motors": 391, "free_motors": 343, "fixed_motors": 48,
        "free_muscles": 90, "leg_free_motors": list(LEG_MOTORS),
        "leg_free_muscles": list(LEG_MUSCLES),
        "samples": 2001, "selected_samples": 1923,
        "startup_rows": 99, "main_rows": 1824,
        "velocity_rows": 1823, "reused_lm_solver_calls": 0}
    solver = {"name": "clarabel", "max_threads": 1, "max_iter": 150,
        "time_limit_s": 90.0, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8, "PAD": 1.0, "ridge": 0.0,
        "raw_clip": False, "converged_status": "Solved"}
    proxy = {"command_ratio": .8, "velocity_nonworse": True,
        "leg_q_v_nonworse": True, "startup_q_rms": .018,
        "startup_q_max": .04, "startup_tendon_rms": .009,
        "startup_tendon_max": .025,
        "strict_existing_feasible_function": True}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E216_SIX_LEG_DYNAMIC_CAPACITY_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != shape
            or freeze.get("clock") != {"neural_dt_s": .00025,
                "control_dt_s": .005, "ticks_per_control": 20,
                "activation_tau_s": .01}
            or freeze.get("solver") != solver
            or freeze.get("tolerance") != {
                "innovation_envelope_epsilon": 2e-12,
                "physical_constraint_atol": 1e-8,
                "forward_readout_atol": 2e-10}
            or freeze.get("global_proxy") != proxy
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("generated_solver", {}).get("actual_calls") != 0
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E216 scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E216 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E216 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E216 parent review does not authorize these sources")
    for label, count in (("input_sha256", 27),
                         ("reused_source_sha256", 28),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E216 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E216 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E216 hash gate exceeded internal deadline")
    paths = freeze.get("paths")
    expected_keys = {"parameters", "trace", "targets", "mask", "basis",
        "affine", "basis_parameters", "lm_physical", "lm_pre", "lm_raw",
        "lm_info", "lm_arrays", "lm_diagnostic", "lm_result", "lm_manifest",
        "lm_ledger"}
    if not isinstance(paths, dict) or set(paths) != expected_keys:
        raise ValueError("E216 frozen input path map changed")
    return freeze, review, {key: repository_path(value)
                            for key, value in paths.items()}


def validate_lm_reuse(paths):
    """Load the already-solved LM witness and reject any incomplete record."""
    result = json.loads(paths["lm_result"].read_text(encoding="utf-8"))
    info = json.loads(paths["lm_info"].read_text(encoding="utf-8"))
    diagnosis = json.loads(paths["lm_diagnostic"].read_text(encoding="utf-8"))
    ledger = json.loads(paths["lm_ledger"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["lm_manifest"].read_text(encoding="utf-8"))
    with (np.load(paths["lm_raw"], allow_pickle=False) as raw,
          np.load(paths["lm_arrays"], allow_pickle=False) as arrays,
          np.load(paths["lm_physical"], allow_pickle=False) as physical,
          np.load(paths["lm_pre"], allow_pickle=False) as pre):
        x = raw["solver_x"].copy()
        q = arrays["q_linear"].copy()
        q_forward = arrays["q_forward"].copy()
        muscles = physical["muscles"].copy()
        motors = physical["motors"].copy()
        if (result.get("validity") != "valid_conditional_LM_dynamic_feasible"
                or result.get("finalization_pending") is not False
                or info.get("status") != "Solved"
                or diagnosis.get("feasible_lm_only") is not True
                or ledger.get("solver_calls") != 1
                or ledger.get("solver_returned") is not True
                or manifest.get("schema") != "flylab.m3.e215.output-manifest.v1"
                or x.shape != (1923 * 14,)
                or arrays["solver_x"].shape != x.shape
                or not np.array_equal(x, arrays["solver_x"])
                or q.shape != (1923, 7) or q_forward.shape != q.shape
                or muscles.shape != (14,) or motors.shape != (54,)
                or pre["groups"] != 14 or pre["frames"] != 1923
                or not all(np.isfinite(a).all() for a in (x, q, q_forward))
                or np.max(np.abs(q - q_forward)) > e215.FORWARD_ATOL):
            raise ValueError("E216 frozen E215 LM witness incomplete")
    return {"x": x, "q": q, "q_forward": q_forward,
        "muscles": muscles, "motors": motors,
        "diagnostic": diagnosis, "result_validity": result["validity"],
        "original_solver_calls": 1, "new_solver_calls": 0}


def derive_leg_inputs(params, mask, basis, affine, leg):
    """Original anatomical/physiological LM formula generalized to one leg."""
    if leg not in NEW_LEGS:
        raise ValueError("E216 only five new legs may be derived")
    leg_index = LEGS.index(leg)
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
                pool, moment, cap, tau, initial, B, q0))
            or np.any(pool < 0) or np.any(B[0, free] != 0)
            or np.any(initial[free] != 0)):
        raise ValueError("E216 original motor/anatomical identity changed")
    for motor, owner in enumerate(membership):
        if np.any(pool[np.arange(96) != owner, motor] != 0):
            raise ValueError("E216 cross-muscle pool ownership")
    muscles = np.asarray([g for g in np.unique(membership[free])
        if groups[int(g)]["leg"] == leg], np.int64)
    motors = np.asarray([m for g in muscles for m in free
                         if membership[m] == g], np.int64)
    if (len(muscles) != LEG_MUSCLES[leg_index]
            or len(motors) != LEG_MOTORS[leg_index]
            or not np.array_equal(np.unique(membership[motors]), muscles)):
        raise ValueError("E216 frozen leg motor/muscle ownership changed")
    other = np.r_[0:7*leg_index, 7*(leg_index+1):42]
    if any(np.any(moment[other, g] != 0) for g in muscles):
        raise ValueError("E216 muscle moment crosses anatomical leg")
    F = np.empty(len(muscles), np.float64)
    G = np.empty(len(muscles), np.float64)
    mass = np.sum(B[:1924, motors], axis=2)
    J = np.empty((1923, len(motors)), np.float64)
    S = np.empty((1923, len(muscles)), np.float64)
    for j, muscle in enumerate(muscles):
        own = np.flatnonzero(membership[motors] == muscle)
        own_tau = tau[motors[own]]
        if (np.any(own_tau != own_tau[0])
                or np.any(own_tau != .019999999552965164)):
            raise ValueError("E216 non-LM original tau changed")
        F[j] = float(e215.rk4_decay(own_tau[0]))
        weights = pool[muscle, motors[own]]
        G[j] = float(weights @ cap[motors[own]])
        if not np.isfinite(G[j]) or G[j] <= 0:
            raise ValueError("E216 nonpositive muscle rate capacity")
        J[:, own] = cap[motors[own]][None] * (
            mass[1:, own] - F[j] * mass[:-1, own])
        S[:, j] = (J[:, own] @ weights) / G[j]
    L = .02 * moment[7*leg_index:7*(leg_index+1), muscles] * G[None]
    return {"leg": leg, "leg_index": leg_index,
        "muscles": muscles, "motors": motors,
        "motor_ids": motor_ids[motors],
        "membership": membership[motors],
        "pool_owned": pool[membership[motors], motors],
        "cap": cap[motors], "tau": tau[motors],
        "F": F, "G": G, "mass": mass, "J": J, "S": S,
        "L": L, "q0": q0[:, 7*leg_index:7*(leg_index+1)],
        "free": free, "fixed": fixed}


def local_target(target, leg_index):
    sl = slice(7*leg_index, 7*(leg_index+1))
    return {"target": target["target"][:, sl],
        "startup_target": target["startup_command"][:, sl],
        "velocity_target": target["velocity"][:, sl],
        "baseline": target["baseline"]}


def verify_all_ownership(params, mask, lm):
    """Confirm the five new legs and reused LM partition all 343/90 units."""
    membership = np.asarray(params["membership"], np.int64)
    free = np.asarray(mask["free_columns"], np.int64)
    groups = json.loads(str(params["muscle_parameters_json"]))
    by_leg_motors = {}
    by_leg_muscles = {}
    if (len(groups) != 96 or membership.shape != (391,)
            or free.shape != (343,) or not np.array_equal(np.sort(free), free)):
        raise ValueError("E216 frozen six-leg ownership metadata changed")
    for i, leg in enumerate(LEGS):
        muscles = np.asarray([g for g in np.unique(membership[free])
                              if groups[int(g)]["leg"] == leg], np.int64)
        motors = np.asarray([m for g in muscles for m in free
                             if membership[m] == g], np.int64)
        if len(muscles) != LEG_MUSCLES[i] or len(motors) != LEG_MOTORS[i]:
            raise ValueError("E216 six-leg motor/muscle count changed")
        by_leg_motors[leg] = motors
        by_leg_muscles[leg] = muscles
    if (not np.array_equal(by_leg_motors["lm"], lm["motors"])
            or not np.array_equal(by_leg_muscles["lm"], lm["muscles"])
            or not np.array_equal(np.sort(np.concatenate(list(by_leg_motors.values()))),
                                  free)
            or len(np.unique(np.concatenate(list(by_leg_muscles.values())))) != 90):
        raise ValueError("E216 reused LM or total 343/90 ownership differs")
    return {leg: {"motors": by_leg_motors[leg].tolist(),
                  "muscles": by_leg_muscles[leg].tolist()} for leg in LEGS}


def assess_leg_solved(problem, data, target, raw, info, out, leg_index):
    """E215 filter reconstruction with this leg's actual baseline index."""
    x = raw["solver_x"]
    if (not np.isfinite(x).all() or not np.isfinite(raw["solver_s"]).all()
            or not np.isfinite(raw["solver_z"]).all()):
        raise ValueError("E216 Solved return nonfinite")
    restored = e215.forward_reconstruction(problem, data, x)
    qerr = np.asarray(problem["command_map"] @ x + problem["command_offset"])
    verr = np.asarray(problem["velocity_map"] @ x + problem["velocity_offset"])
    serr = np.asarray(problem["startup_map"] @ x + problem["startup_offset"])
    np.savez(Path(out) / "diagnostic.npz", solver_x=x, **restored,
             command_residual=qerr, velocity_residual=verr,
             startup_residual=serr)
    force = restored["forcing"]
    equality = problem["A"] @ x + raw["solver_s"] - problem["b"]
    baseline = float(target["baseline"]["leg_command_rms"][leg_index])
    metrics = {"status": info["status"], "leg": LEGS[leg_index],
        "feasible_leg_only": False,
        "numeric_tolerance": e215.PHYSICAL_ATOL,
        "forward_tolerance": e215.FORWARD_ATOL,
        "x_min": float(np.min(x)), "x_max": float(np.max(x)),
        "forcing_lower_margin_min": float(np.min(force + problem["epsilon"])),
        "forcing_upper_residual_max": float(np.max(
            force - data["S"] - problem["epsilon"])),
        "startup_q_max": float(np.max(np.abs(serr))),
        "startup_q_rms": float(np.sqrt(np.mean(serr * serr))),
        "main_command_rms": float(np.sqrt(np.mean(qerr * qerr))),
        "main_command_baseline_rms": baseline,
        "main_velocity_rms": float(np.sqrt(np.mean(verr * verr))),
        "main_velocity_baseline_rms": problem["velocity_limit"],
        "forward_q_max_abs": float(np.max(np.abs(
            restored["q_linear"] - restored["q_forward"]))),
        "primal_equality_max": float(np.max(np.abs(equality))),
        "scope": "single-leg conditional free forcing only"}
    metrics["main_command_improvement_fraction"] = (
        1 - metrics["main_command_rms"] / baseline)
    metrics["feasible_leg_only"] = bool(
        metrics["x_min"] >= -e215.PHYSICAL_ATOL
        and metrics["x_max"] <= 1 + e215.PHYSICAL_ATOL
        and metrics["forcing_lower_margin_min"] >= -e215.PHYSICAL_ATOL
        and metrics["forcing_upper_residual_max"] <= e215.PHYSICAL_ATOL
        and metrics["startup_q_max"] <= problem["startup_limit"] + e215.PHYSICAL_ATOL
        and metrics["main_command_rms"] <= problem["command_limit"] + e215.PHYSICAL_ATOL
        and metrics["main_velocity_rms"] <= problem["velocity_limit"] + e215.PHYSICAL_ATOL
        and metrics["forward_q_max_abs"] <= e215.FORWARD_ATOL
        and metrics["primal_equality_max"] <= e215.PHYSICAL_ATOL)
    save_json(Path(out) / "diagnostic.json", metrics)
    return metrics, restored["q_linear"]


def combine_six_leg_command(held_row0, per_leg_q):
    """Build exactly held row0 plus 1923 frames, never filling missing legs."""
    if set(per_leg_q) != set(LEGS):
        raise ValueError("E216 all six valid legs required before combination")
    row0 = np.asarray(held_row0, np.float64)
    if row0.shape != (42,) or not np.isfinite(row0).all():
        raise ValueError("E216 actual held row0 command required")
    blocks = []
    for leg in LEGS:
        q = np.asarray(per_leg_q[leg], np.float64)
        if q.shape != (1923, 7) or not np.isfinite(q).all():
            raise ValueError("E216 complete finite per-leg 1923x7 command required")
        blocks.append(q)
    return np.vstack((row0[None], np.concatenate(blocks, axis=1)))


def evaluate_global_proxy(command, tendon, target):
    """Use the unchanged E203 whole-body metrics and strict feasibility."""
    metrics = direct_metrics({"command": command, "tendon": tendon}, target)
    passed = feasible(metrics, target["baseline"])
    return {"metrics": metrics, "baseline": target["baseline"],
        "strict_original_proxy_pass": bool(passed),
        "scope": "six-leg conditional free forcing only; no neural/body candidate"}


def count_new_calls(legs):
    """Never infer an unreturned call as zero or count the reused LM call."""
    if not set(legs).issubset(set(NEW_LEGS)):
        raise ValueError("E216 ledger includes a reused or unknown leg")
    calls = sum(int(row.get("solver_calls", 0)) for row in legs.values())
    if (calls < 0 or calls > 5
            or any(row.get("solver_calls", 0) not in (0, 1)
                   for row in legs.values())):
        raise ValueError("E216 per-leg or aggregate one-call cap exceeded")
    return calls


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e216.output-manifest.v1", "files": files,
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
        raise FileExistsError("One-shot E216 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = None
    primary = None
    errors = []
    ledger = {"schema": "flylab.m3.e216.ledger.v1",
        "new_solver_calls": 0, "reused_lm_new_solver_calls": 0,
        "solver_wall_s": 0., "constructor_wall_s": 0.,
        "current_leg": None, "completed_new_legs": [], "legs": {},
        "full_banc_calls": 0, "cuda_calls": 0, "neural_ticks": 0,
        "physics_steps": 0, "retries": 0}
    result = {"schema": "flylab.m3.e216.six-leg-dynamic-capacity.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "diagnostic_only": True, "global_proxy_evaluated": False}

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E216 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_lm_solver_calls_in_this_process": 0})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            lm = validate_lm_reuse(paths)
            result["reused_lm"] = {"validity": lm["result_validity"],
                "original_solver_calls": 1, "new_solver_calls": 0,
                "motor_count": len(lm["motors"]),
                "muscle_count": len(lm["muscles"])}
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target = load_targets(paths["targets"], trace)
            held = np.asarray(trace["samples"][0]["joint_targets_rad"], np.float64)
            with (np.load(paths["parameters"], allow_pickle=False) as p,
                  np.load(paths["mask"], allow_pickle=False) as m,
                  np.load(paths["basis"], allow_pickle=False) as b,
                  np.load(paths["affine"], allow_pickle=False) as a,
                  np.load(paths["basis_parameters"], allow_pickle=False) as bp):
                for key in ("cap", "tau", "initial_rate", "motor_ids"):
                    if not np.array_equal(p[key], bp[key]):
                        raise ValueError("E216 E201/E213 motor identity differs: " + key)
                if (not np.array_equal(m["free_columns"], bp["free_columns"])
                        or not np.array_equal(m["fixed_columns"], bp["fixed_columns"])):
                    raise ValueError("E216 E203/E213 free ownership differs")
                params = {key: p[key].copy() for key in (
                    "membership", "pool", "moment", "cap", "tau",
                    "initial_rate", "motor_ids", "muscle_parameters_json")}
                mask = {key: m[key].copy() for key in (
                    "free_columns", "fixed_columns", "membership", "motor_ids")}
                basis = {"B": b["B"].copy()}
                affine = {"q_intercept": a["q_intercept"].copy(),
                          "tendon_intercept": a["tendon_intercept"].copy()}
            if (affine["tendon_intercept"].shape != (2001, 6)
                    or len(trace["samples"]) != 2001
                    or held.shape != (42,)):
                raise ValueError("E216 exact original held row/tendon clock changed")
            if (target["indices"].shape != (1824,)
                    or not np.array_equal(target["indices"], np.arange(100, 1924))
                    or not np.array_equal(target["pair_start"],
                                          target["indices"][:-1])
                    or not np.array_equal(target["pair_end"],
                                          target["indices"][1:])):
                raise ValueError("E216 frozen original target rows changed")
            ownership = verify_all_ownership(params, mask, lm)
            save_json(out / "ownership.json", ownership)

            per_leg_q = {"lm": lm["q"]}
            status = None
            for leg in NEW_LEGS:
                index = LEGS.index(leg)
                leg_dir = out / ("leg-" + leg)
                leg_dir.mkdir()
                entry = {"status": "preparing", "requested": False,
                    "called": False, "returned": False, "solver_calls": 0}
                ledger["current_leg"] = leg
                ledger["legs"][leg] = entry
                save_json(out / "ledger.json", ledger)
                local_ledger = {"solve_requested": False,
                    "constructor_entered": False,
                    "constructor_completed": False,
                    "solve_called": False, "solver_returned": False,
                    "solver_calls": 0, "constructor_wall_s": 0.,
                    "solver_wall_s": 0.}
                try:
                    if ledger["new_solver_calls"] >= RESOURCE["solver_calls_max"]:
                        raise RuntimeError("E216 five-call total exhausted")
                    e215.ensure_solve_budget(elapsed)
                    data = derive_leg_inputs(params, mask, basis, affine, leg)
                    e215.save_physical_inputs(leg_dir, data)
                    local = local_target(target, index)
                    problem = e215.build_capacity_problem(
                        dict(data, q0=data["q0"][:1924]), local,
                        target["indices"], np.arange(1, 100),
                        target["pair_start"], target["pair_end"],
                        float(target["baseline"]["leg_command_rms"][index]),
                        float(target["baseline"]["leg_velocity_rms"][index]))
                    e215.save_pre_solve(leg_dir, problem,
                        freeze["input_sha256"][freeze["paths"]["basis"]])
                    entry["pre_solve_saved"] = True
                    save_json(out / "ledger.json", ledger)
                    e215.ensure_solve_budget(elapsed)
                    raw, info = e215.solve_once(problem, leg_dir, local_ledger,
                                                elapsed=elapsed)
                    entry["status"] = info["status"]
                    entry["solver_info"] = info
                    if info["status"] == "Solved":
                        diagnosis, q = assess_leg_solved(problem, data, local,
                            raw, info, leg_dir, index)
                        entry["diagnostic"] = diagnosis
                        if diagnosis["feasible_leg_only"]:
                            per_leg_q[leg] = q
                            entry["status"] = "valid_conditional_leg_feasible"
                            ledger["completed_new_legs"].append(leg)
                        else:
                            status = "valid_bounded_unresolved"
                    elif info["status"] == "PrimalInfeasible":
                        diagnosis = e215.infeasible_diagnostic(
                            problem, raw, info, leg_dir)
                        entry["diagnostic"] = diagnosis
                        status = ("valid_conditional_leg_infeasible"
                            if diagnosis["certificate_verified"]
                            else "valid_bounded_unresolved")
                    else:
                        status = "valid_bounded_unresolved"
                        save_json(leg_dir / "diagnostic.json", {
                            "status": info["status"],
                            "reason": "bounded non-Solved leg return"})
                except e215.SolverAllowanceUnavailable as error:
                    entry["status"] = "blocked_by_shared_deadline"
                    entry["budget_block"] = str(error)
                    status = "valid_bounded_unresolved_no_further_solver_call"
                except BaseException as error:  # noqa: BLE001 - fail closed
                    entry["status"] = "incomplete_or_invalid"
                    entry["error"] = {"type": type(error).__name__,
                                      "message": str(error)}
                    errors.append({"stage": leg, "type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc()})
                    if primary is None:
                        primary = error
                    status = "incomplete_or_invalid"
                finally:
                    entry.update(requested=bool(local_ledger["solve_requested"]),
                        called=bool(local_ledger["solve_called"]),
                        returned=bool(local_ledger["solver_returned"]),
                        solver_calls=int(local_ledger["solver_calls"]),
                        constructor_wall_s=float(local_ledger["constructor_wall_s"]),
                        solver_wall_s=float(local_ledger["solver_wall_s"]))
                    ledger["new_solver_calls"] = count_new_calls(ledger["legs"])
                    ledger["solver_wall_s"] = sum(
                        item.get("solver_wall_s", 0.) for item in ledger["legs"].values())
                    ledger["constructor_wall_s"] = sum(
                        item.get("constructor_wall_s", 0.)
                        for item in ledger["legs"].values())
                    try:
                        save_json(out / "ledger.json", ledger)
                    except BaseException as error:  # noqa: BLE001 - no next solve
                        errors.append({"stage": "aggregate-ledger",
                            "type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()})
                        if primary is None:
                            primary = error
                        status = "incomplete_or_invalid"
                if status is not None:
                    break
            result["leg_status"] = {leg: row["status"]
                for leg, row in ledger["legs"].items()}
            result["completed_new_legs"] = list(ledger["completed_new_legs"])
            if status is not None:
                result["validity"] = status
            elif set(per_leg_q) == set(LEGS):
                command = combine_six_leg_command(held, per_leg_q)
                tendon = affine["tendon_intercept"][:1924].copy()
                np.savez(out / "combined-conditional.npz",
                    frame_indices=np.arange(1924, dtype=np.int64),
                    command=command, tendon=tendon,
                    leg_order=np.asarray(LEGS))
                report = evaluate_global_proxy(command, tendon, target)
                save_json(out / "global-metrics.json", report)
                result.update(validity="valid_conditional_six_leg_dynamic_capacity",
                    global_proxy_evaluated=True,
                    global_proxy_pass=report["strict_original_proxy_pass"],
                    global_metrics=report["metrics"], baseline=target["baseline"])
            else:
                raise ValueError("E216 missing-leg state without bounded stop")
    except BaseException as error:  # noqa: BLE001 - preserve first error
        if primary is None:
            primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    ledger["process_wall_s"] = elapsed()
    _save_attempt(errors, "ledger.json", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        new_solver_calls=ledger["new_solver_calls"],
        reused_lm_new_solver_calls=0,
        solver_wall_s=ledger["solver_wall_s"],
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0) + elapsed(),
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
        result["resource_failure"] = "E216 output manifest missing or over 256 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E216 hard internal deadline exceeded"
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
            primary = RuntimeError("E216 raw/publication failure")
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
