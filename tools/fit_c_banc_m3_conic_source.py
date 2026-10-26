"""E275 all-startup-row/SOC source directions, explicitly injected into E270.

Clarabel predicts directions only. The unchanged nonlinear evaluator, startup
checks, training objective, budgets and final selection judge every candidate.
Import performs no original, CUDA, BANC, body or solver execution.
"""

from __future__ import annotations

# Preserve primary solver/calculation/storage faults, including secondary saves.
# ruff: noqa: BLE001
import argparse
import copy
import functools
import json
import sys
import time
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import fit_c_banc_m3_feasible_source as previous
from tools import fit_c_banc_m3_source_coefficients as base

SCHEMA = "flylab.m3.e275.conic-source-fit.v1"
REVIEW_SCHEMA = "flylab.m3.e275.source-review.v1"
SOURCES = (*previous.SOURCES, "tools/fit_c_banc_m3_conic_source.py",
           "tests/test_c_banc_conic_source_fit.py")
ENGINE_SOURCES = previous.ENGINE_SOURCES
SOLVER_KIND = "actual Clarabel SOCP calls"
CONTRACT = {k: v for k, v in base.CONTRACT.items() if k not in ("maxiter", "ftol")}
CONTRACT.update(solver="clarabel.DefaultSolver:SOCP", clarabel_version=clarabel.__version__,
    max_outer_steps=24, initial_radius=.05, max_radius=.5, min_radius=1e-6,
    linear_slack_fraction=.8, proximal_weight=1e-3,
    conic_max_iter=100, conic_wall_s=2.0, conic_max_threads=1,
    tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8,
    independent_residual_tolerance=1e-8,
    alphas=[1., .5, .25, .125, .0625, .03125], armijo=1e-4,
    generated_e273_min_speedup=1.5, startup_peak_rows="all_99_times",
    startup_soc_count=2)


def validate_freeze(path, digest, review_path, review_digest):
    """Eight live source pins; preserved E270 syntax/inputs and E273 gate."""
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT,
                 "Exact E275 schema/contract and installed Clarabel version required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES),
                 "Eight E275 source pins required")
    for name, sha in sources.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Reviewed E275 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
                 and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest,
                 "Exact E275 source review required")
    inputs = freeze.get("inputs")
    base.require(isinstance(inputs, dict) and inputs, "E275 input SHA map required")
    for spec in inputs.values():
        base.require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "Exact input path/SHA required")
        base.require(base.sha256(base.safe_path(spec["path"])) == spec["sha256"], "Frozen input SHA changed")
    original = previous.hashed_gate(freeze, "e270_original_freeze")
    base.require(original.get("schema") == base.SCHEMA and original.get("contract") == base.CONTRACT,
                 "Preserved original E270 freeze required")
    for name, spec in original["inputs"].items():
        base.require(inputs.get(name) == spec, "Original E270 input changed: "+name)
    for key in ("contexts", "targets", "legs", "identity", "native_validation", "generated_cuda_gate"):
        base.require(freeze[key] == original[key], "Original E270 context/gate changed: "+key)
    for name in ("native_validation", "generated_cuda_gate", "generated_e273_gate"):
        gate = previous.hashed_gate(freeze, name)
        base.require(gate.get("status") == "pass", name+" did not pass")
        if name == "native_validation":
            base.require(all(gate.get(k) is True for k in
                ("counters_valid", "raw_valid", "identity_valid", "decoder_valid")), "Native checks missing")
        if name == "generated_e273_gate":
            speed = gate.get("median_speedup")
            base.require(gate.get("exactness_valid") is True and gate.get("speed_valid") is True,
                         "E273 exactness and speed gate required")
            base.require(isinstance(speed, (float, int)) and not isinstance(speed, bool)
                         and np.isfinite(speed) and speed >= 1.5, "E273 median speedup below 1.5")
            base.require(gate.get("source_sha256") == {k: sources[k] for k in ENGINE_SOURCES},
                         "E273 gate source hashes differ")
    return freeze


def remaining_budget(budget):
    now = budget.clock()
    remaining = {"leg_s": 30-(now-budget.leg_started), "shared_fit_s": 180-(now-budget.started),
                 "original_s": None if budget.deadline is None else 290-(now-budget.deadline.started)}
    limit = min(2.0, *(v for v in remaining.values() if v is not None))
    if limit <= 0:
        budget.check()
        raise base.BudgetExceeded("leg")
    return remaining, float(limit)


def conic_problem(current, fixed_sse, radius):
    """Build C-order f64 bounds/all-peak inequalities plus exactly two SOCs."""
    c = np.asarray(current["coefficients"])
    g = np.asarray(current["gradient"])
    qe, le = np.asarray(current["qerr"]), np.asarray(current["lerr"])
    jq, jl = np.asarray(current["startup_q_jacobian"]), np.asarray(current["startup_ltm_jacobian"])
    fixed = np.asarray(fixed_sse, dtype=np.float64)
    base.require(c.ndim == 2 and c.shape[1] == 3 and 1 <= len(c) <= 1660
                 and c.dtype == np.float64 and base.coefficient_valid(c, interior=True),
                 "Strict feasible float64 coefficient triples required")
    params = c.size
    base.require(g.shape == (params,) and qe.shape == (99, 7) and le.shape == (99, 1)
                 and jq.shape == (99, 7, params) and jl.shape == (99, 1, params)
                 and all(v.dtype == np.float64 for v in (g, qe, le, jq, jl)),
                 "Retained complete float64 startup Jacobians/errors required")
    base.require(all(np.isfinite(v).all() for v in (c, g, qe, le, jq, jl, fixed))
                 and fixed.shape == (2,) and np.all(fixed >= 0)
                 and np.isfinite(radius) and 1e-6 <= radius <= .5, "Finite conic input/fixed SSE/trust radius required")
    scale = float(np.max(np.abs(g)))
    base.require(scale > 0, "Nonzero finite gradient required for SOCP")
    c, g = np.array(c, order="C", copy=True), np.array(g, order="C", copy=True)
    qe, le = np.array(qe, order="C", copy=True), np.array(le, order="C", copy=True)
    jq, jl = np.array(jq, order="C", copy=True), np.array(jl, order="C", copy=True)
    errors = (qe.ravel(), le.ravel())
    jacobians = (jq.reshape(qe.size, params), jl.reshape(le.size, params))
    peaks = np.array([.04, .025])*(1-1e-4)
    available = np.array([99*42*(.018*(1-1e-4))**2, 99*6*(.009*(1-1e-4))**2])-fixed
    sse = np.array([np.sum(e*e) for e in errors])
    base.require(np.all(available >= sse) and all(np.all(np.abs(e) <= peak) for e, peak in zip(errors, peaks, strict=True)),
                 "Current startup is not strictly feasible under fixed global budgets")
    norm2 = sse+.8*(available-sse)
    radii = np.sqrt(norm2)
    bounds = np.column_stack((np.maximum(-radius, -c.ravel()), np.minimum(radius, 8-c.ravel())))
    triple = base.triple_constraint(len(c))
    sums = c.sum(1)
    identity = sparse.eye(params, format="csc", dtype=np.float64)
    linear_a = [identity, -identity, sparse.csc_matrix(triple.A), sparse.csc_matrix(-triple.A)]
    linear_b = [bounds[:, 1], -bounds[:, 0], np.asarray(triple.ub)-sums, sums-np.asarray(triple.lb)]
    for error, jac, peak in zip(errors, jacobians, peaks, strict=True):
        linear_a.extend((sparse.csc_matrix(jac), sparse.csc_matrix(-jac)))
        linear_b.extend((.8*(peak-error), .8*(peak+error)))
    a = sparse.vstack(linear_a, format="csc")
    b = np.concatenate(linear_b)
    linear_rows = len(b)
    soc_a, soc_b = [], []
    for error, jac, norm_radius in zip(errors, jacobians, radii, strict=True):
        soc_a.append(sparse.vstack((sparse.csc_matrix((1, params)), sparse.csc_matrix(-jac)), format="csc"))
        soc_b.append(np.r_[norm_radius, error])
    return {"P": 1e-3*identity, "q": g/scale,
            "A": sparse.vstack((a, *soc_a), format="csc"), "b": np.concatenate((b, *soc_b)),
            "cones": [clarabel.NonnegativeConeT(linear_rows), clarabel.SecondOrderConeT(qe.size+1),
                      clarabel.SecondOrderConeT(le.size+1)], "nonnegative_rows": linear_rows,
            "coefficients": c, "gradient": g, "normalized_gradient": g/scale, "gradient_scale": scale,
            "bounds": bounds, "qerr": qe, "lerr": le, "startup_q_jacobian": jq, "startup_ltm_jacobian": jl,
            "fixed_sse": fixed.copy(), "available_sse": available, "current_sse": sse,
            "norm_squared_limits": norm2, "radii": radii, "peaks": peaks, "radius": float(radius)}


def conic_residuals(problem, direction):
    """Independent coefficient/trust, all-peak and curved-norm adjudication."""
    d = np.asarray(direction)
    base.require(d.dtype == np.float64 and d.shape == problem["gradient"].shape and np.isfinite(d).all(),
                 "Finite float64 conic direction required")
    c = problem["coefficients"].ravel()+d
    sums = c.reshape(-1, 3).sum(1)
    coefficient = np.r_[c, 8-c, sums-.125*(1+1e-6), 8*(1-1e-6)-sums,
                        d-problem["bounds"][:, 0], problem["bounds"][:, 1]-d]
    linear, soc = [], []
    for ek, jk, peak, norm_radius in zip(("qerr", "lerr"), ("startup_q_jacobian", "startup_ltm_jacobian"),
                                       problem["peaks"], problem["radii"], strict=True):
        e = problem[ek].ravel(order="C")
        jd = problem[jk].reshape(e.size, len(d), order="C")@d
        linear.extend((.8*(peak-e)-jd, .8*(peak+e)+jd))
        soc.append(norm_radius-np.linalg.norm(e+jd))
    linear, soc = np.concatenate(linear), np.asarray(soc)
    return {"coefficient": coefficient, "linear": linear, "soc": soc,
            "minimum": float(min(np.min(coefficient), np.min(linear), np.min(soc)))}


def solver_settings(time_limit):
    base.require(np.isfinite(time_limit) and 0 < time_limit <= 2, "Bounded Clarabel time limit required")
    settings = clarabel.DefaultSettings()
    values = {"verbose": False, "max_iter": 100, "tol_gap_abs": 1e-8,
              "tol_gap_rel": 1e-8, "tol_feas": 1e-8, "time_limit": float(time_limit), "max_threads": 1}
    for name, value in values.items():
        base.require(hasattr(settings, name), "Installed Clarabel lacks required setting: "+name)
        setattr(settings, name, value)
        base.require(getattr(settings, name) == value, "Clarabel setting did not apply: "+name)
    return settings, values


def _phase():
    return {"requested": 0, "completed": 0, "failed": 0, "wall_s": 0.0}


def solve_direction(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """One constructor and at most one actual solve, with primary failure data."""
    settings, values = solver_settings(time_limit)
    receipt = {"solver_calls_kind": SOLVER_KIND, "clarabel_version": clarabel.__version__,
               "settings": values, "constructor": _phase(), "solve": _phase(),
               "iterations": 0, "iterations_known": False, "status": "requested"}
    result, solver = None, None
    phase = receipt["constructor"]
    phase["requested"] = 1
    began = wall_clock()
    try:
        try:
            solver = solver_factory(problem["P"], problem["q"], problem["A"], problem["b"], problem["cones"], settings)
            phase["completed"] = 1
        except Exception:
            phase["failed"] = 1
            raise
        finally:
            phase["wall_s"] = wall_clock()-began
        phase = receipt["solve"]
        phase["requested"] = 1
        began = wall_clock()
        try:
            result = solver.solve()
            phase["completed"] = 1
        except Exception:
            phase["failed"] = 1
            raise
        finally:
            phase["wall_s"] = wall_clock()-began
        receipt.update(status=str(result.status), iterations=int(result.iterations), iterations_known=True)
        direction = np.asarray(result.x, dtype=np.float64)
        s, z = np.asarray(result.s, dtype=np.float64), np.asarray(result.z, dtype=np.float64)
        finite = (direction.shape == problem["q"].shape and s.shape == problem["b"].shape
                  and z.shape == problem["b"].shape and all(np.isfinite(v).all() for v in (direction, s, z)))
        residuals = conic_residuals(problem, direction) if finite else None
        info = {k: previous.finite_scalar(getattr(result, k)) for k in
                ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual") if hasattr(result, k)}
        receipt.update(info=info, minimum_residual=None if residuals is None else residuals["minimum"])
        valid = str(result.status) == "Solved" and finite and residuals["minimum"] >= -1e-8
        receipt.update(solution_valid=bool(valid), finite_solution=bool(finite))
        return {"direction": direction, "s": s, "z": z, "valid": bool(valid),
                "residuals": residuals, "receipt": receipt}
    except Exception as error:
        receipt["status"] = "execution_failure"
        error.conic_receipt = copy.deepcopy(receipt)
        error.partial_conic_solution = result
        raise


def _empty_stats():
    return {"solver_calls_kind": SOLVER_KIND, "slsqp_calls": 0, "lp_calls": 0,
            "leg_optimizer_calls": 0, "outer_steps": 0, "socp_constructor_requests": 0,
            "socp_constructor_completed": 0, "socp_constructor_failed": 0,
            "socp_solve_requests": 0, "socp_solve_completed": 0, "socp_solve_failed": 0,
            "socp_iterations_unknown_calls": 0, "clarabel_iterations": 0,
            "socp_constructor_wall_s": 0.0, "socp_solve_wall_s": 0.0,
            "accepted_steps": 0, "rejected_proposals": 0, "legs": []}


def stats_for(budget):
    if not hasattr(budget, "e275_stats"):
        budget.e275_stats = _empty_stats()
    return budget.e275_stats


def _add_solver_cost(receipt, budget, cost, stats):
    budget.solver_calls += receipt["solve"]["requested"]
    budget.iterations += receipt["iterations"]
    for target in (cost, stats):
        for phase in ("constructor", "solve"):
            for key in ("requested", "completed", "failed"):
                label = "requests" if key == "requested" else key
                target["socp_"+phase+"_"+label] += receipt[phase][key]
            target["socp_"+phase+"_wall_s"] += receipt[phase]["wall_s"]
        target["clarabel_iterations"] += receipt["iterations"]
        target["socp_iterations_unknown_calls"] += int(receipt["solve"]["requested"] and not receipt["iterations_known"])


def save_direction(path, problem, solved):
    fields = ("coefficients", "gradient", "normalized_gradient", "gradient_scale", "bounds", "qerr", "lerr",
              "startup_q_jacobian", "startup_ltm_jacobian", "fixed_sse", "available_sse", "current_sse",
              "norm_squared_limits", "radii", "peaks", "radius", "nonnegative_rows")
    arrays = {key: problem[key] for key in fields}
    if solved is not None:
        arrays.update(direction=solved["direction"], solver_s=solved["s"], solver_z=solved["z"])
        if solved["residuals"] is not None:
            arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
    np.savez(path, **arrays)


def fit_conic_leg(evaluator, initial, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """Bounded all-row/SOC directions and unchanged actual nonlinear backtracking."""
    budget, stats = evaluator.budget, stats_for(evaluator.budget)
    stats["leg_optimizer_calls"] += 1
    history = {"outer": [], "proposals": [], "socp": []}
    cost = {k: v for k, v in _empty_stats().items() if k not in ("legs", "leg_optimizer_calls")}
    cost["leg"] = Path(evaluator.output).name
    stats["legs"].append(cost)
    status, budget_scope, primary, radius = "complete", None, None, .05
    fallback = {"coefficients": initial.copy(), "objective": 1., "mse": evaluator.baseline_mse,
                "baseline_mse": evaluator.baseline_mse, "valid": False, "evaluation": None}

    def persist_history():
        base.write_json(evaluator.output/"conic-history.json", history)
        base.write_json(evaluator.output/"conic-cost.json", {**cost, "budget": budget.receipt()})

    try:
        budget.check()
        current = evaluator.evaluate(initial.ravel())
        base.require(previous.entry_finite(current) and previous.eligible(current), "Neutral iterate is not strict feasible/finite")
        for outer in range(24):
            budget.check()
            cost["outer_steps"] += 1
            stats["outer_steps"] += 1
            history["outer"].append({"step": outer, "radius": radius, "evaluation": current["evaluation"]})
            if radius < 1e-6:
                status = "minimum_radius"
                break
            if np.max(np.abs(current["gradient"])) == 0:
                status = "stationary"
                break
            problem = conic_problem(current, evaluator.fixed_sse, radius)
            remaining, limit = remaining_budget(budget)
            index = len(history["socp"])
            row = {"outer_step": outer, "radius": radius, "remaining_budget": remaining,
                   "time_limit_s": limit, "status": "requested"}
            history["socp"].append(row)
            solved = None
            save_direction(evaluator.output/f"socp-{index:03d}.npz", problem, None)
            persist_history()
            # Input persistence is inside the same bounded leg/shared clocks.
            remaining, limit = remaining_budget(budget)
            row.update(remaining_budget=remaining, time_limit_s=limit)
            try:
                solved = solve_direction(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock)
                row.update(solved["receipt"])
                _add_solver_cost(solved["receipt"], budget, cost, stats)
            except Exception as error:
                receipt = getattr(error, "conic_receipt", None)
                if receipt is not None:
                    row.update(receipt)
                    _add_solver_cost(receipt, budget, cost, stats)
                raise
            save_direction(evaluator.output/f"socp-{index:03d}.npz", problem, solved)
            persist_history()
            if not solved["valid"]:
                status = "bounded_socp_failure"
                break
            budget.check()
            direction = solved["direction"]
            if np.max(np.abs(direction)) == 0:
                status = "zero_direction"
                break
            accepted = False
            for alpha in (1., .5, .25, .125, .0625, .03125):
                budget.check()
                candidate = np.clip(current["coefficients"].ravel()+alpha*direction, 0, 8).reshape(initial.shape)
                proposal = {"outer_step": outer, "alpha": alpha, "radius": radius, "evaluation": None}
                history["proposals"].append(proposal)
                if not base.coefficient_valid(candidate, interior=True):
                    proposal["status"] = "rejected_coefficient_before_evaluation"
                else:
                    try:
                        actual = evaluator.evaluate(candidate.ravel())
                    except base.BudgetExceeded as error:
                        saved = base.preserve_primary(error, "E275 interrupted proposal metadata failed",
                            lambda stop=error, entry=proposal, old=current, point=candidate:
                            previous.interrupted_proposal(evaluator, entry, old, point, stop))
                        if saved is not True:
                            error.abort_run = True
                        raise
                    slope, armijo = previous.proposal_numbers(proposal, current, candidate, actual)
                    if previous.eligible(actual) and slope < 0 and actual["objective"] <= armijo:
                        proposal.update(status="accepted", adopted_as_iterate=True)
                        current, accepted = actual, True
                        cost["accepted_steps"] += 1
                        stats["accepted_steps"] += 1
                        if alpha == 1:
                            radius = min(.5, 2*radius)
                        persist_history()
                        break
                    proposal["status"] = "rejected_true_constraint_or_armijo"
                cost["rejected_proposals"] += 1
                stats["rejected_proposals"] += 1
                persist_history()
            if not accepted:
                radius *= .5
        else:
            status = "outer_step_limit"
    except base.BudgetExceeded as error:
        primary, status, budget_scope = error, str(error), error.scope
    except Exception as error:
        primary, status = error, "execution_failure"
    completed = [entry for entry in evaluator.evaluations if previous.eligible(entry)]
    selected = dict(base.choose_candidate(completed, fallback), status=status, budget_scope=budget_scope)

    def persist_all():
        persist_history()
        np.savez(evaluator.output/"selected.npz", **{k: v for k, v in selected.items() if isinstance(v, np.ndarray)})
        base.write_json(evaluator.output/"selection.json", {"status": status, "budget_scope": budget_scope,
            "evaluation": selected["evaluation"], "objective": selected["objective"], "mse": selected["mse"],
            "cost": cost, "budget": budget.receipt(), "solver_calls_kind": SOLVER_KIND, "slsqp_calls": 0})
        return True

    if primary is not None:
        saved = base.preserve_primary(primary, "E275 fit history/selection persistence failed", persist_all)
        if saved is not True:
            primary.abort_run = True
        if (getattr(primary, "abort_run", False) or not isinstance(primary, base.BudgetExceeded)
                or primary.scope == "original"):
            raise primary
    else:
        persist_all()
    return selected


class ConicRunMeter(base.RunMeter):
    def receipt(self):
        return {**super().receipt(), "optimizer": stats_for(self.budget) if self.budget else _empty_stats()}


def original_run(freeze, output, started):
    from tools.banc_m3_source_coefficient_engine import (
        CudaSourceCoefficientEngine,
        prepare_problem,
    )
    meter = ConicRunMeter(output, base.OriginalDeadline(started))
    primary = None
    try:
        return base.execute_original(freeze, output, CudaSourceCoefficientEngine, prepare_problem, meter,
            leg_fit=fit_conic_leg,
            evaluator_factory=functools.partial(base.LegEvaluator, retain_startup_jacobians=True))
    except Exception as error:
        primary = error
        saved = base.preserve_primary(error, "E275 original failure receipt failed",
            lambda stop=error: base.failure_record(stop, None, output, "e275-original", meter.receipt()) or True)
        if saved is not True:
            error.abort_run = True
        raise
    finally:
        if primary is None:
            meter.save()
        else:
            base.preserve_primary(primary, "E275 final run cost persistence failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    base.require(not output.exists(), "Fresh E275 output directory required")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output/"cost.json", {"phase": "freeze_validation", "original_wall_s": 0.,
        "solver_calls_kind": SOLVER_KIND, "slsqp_calls": 0, "optimizer_calls": 0})
    try:
        freeze = validate_freeze(Path(args.freeze), args.freeze_sha256, Path(args.source_review), args.source_review_sha256)
        verdict = original_run(freeze, output, started)
        base.write_json(output/"result.json", {"status": "complete", "original_wall_s": time.perf_counter()-started, **verdict})
        return 0
    except Exception as error:
        record = {"status": "budget_stop" if isinstance(error, base.BudgetExceeded) else "error",
            "error": str(error), "exception": type(error).__name__, "notes": getattr(error, "__notes__", []),
            "original_wall_s": time.perf_counter()-started}
        def persist_terminal():
            cost = json.loads((output/"cost.json").read_text(encoding="utf-8"))
            cost["original_wall_s"] = record["original_wall_s"]
            base.write_json(output/"cost.json", cost)
            base.write_json(output/"result.json", record)
        base.preserve_primary(error, "E275 terminal failure receipt failed", persist_terminal)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
