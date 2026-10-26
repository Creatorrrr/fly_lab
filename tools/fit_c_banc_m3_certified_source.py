"""E277 certified original QP/SOC source directions, explicitly injected into E270.

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_conic_certificate as certificate
from tools import fit_c_banc_m3_conic_source as conic
from tools import fit_c_banc_m3_feasible_source as previous
from tools import fit_c_banc_m3_source_coefficients as base

SCHEMA = "flylab.m3.e277.certified-source-fit.v1"
REVIEW_SCHEMA = "flylab.m3.e277.source-review.v1"
SOURCES = (*conic.SOURCES, "tools/banc_m3_conic_certificate.py",
           "tests/test_c_banc_conic_certificate.py", "tools/fit_c_banc_m3_certified_source.py",
           "tests/test_c_banc_certified_source_fit.py")
ENGINE_SOURCES = conic.ENGINE_SOURCES
SOLVER_KIND = conic.SOLVER_KIND
CONTRACT = copy.deepcopy(conic.CONTRACT)
CONTRACT.update(direction_certificate=copy.deepcopy(certificate.CERTIFICATE_CONTRACT),
                accepted_alpha_radius="alpha<1:max(1e-6,alpha*r);alpha=1:min(.5,2*r)")
conic_problem = conic.conic_problem
conic_residuals = conic.conic_residuals
solve_certified = certificate.solve_certified


def validate_freeze(path, digest, review_path, review_digest):
    """Twelve live source pins; preserved E270 syntax/inputs and E273 gate."""
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT,
                 "Exact E277 schema/contract and installed Clarabel version required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES),
                 "Twelve E277 source pins required")
    for name, sha in sources.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Reviewed E277 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
                 and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest,
                 "Exact E277 source review required")
    inputs = freeze.get("inputs")
    base.require(isinstance(inputs, dict) and inputs, "E277 input SHA map required")
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
    e275 = previous.hashed_gate(freeze, "e275_original_freeze")
    base.require(e275.get("schema") == conic.SCHEMA and e275.get("contract") == conic.CONTRACT,
                 "Preserved original E275 freeze required")
    base.require(e275.get("approved_sources") == {k: sources[k] for k in conic.SOURCES},
                 "Original E275 eight source pins changed")
    for name, spec in e275["inputs"].items():
        base.require(inputs.get(name) == spec, "Original E275 input changed: "+name)
    for key in ("contexts", "targets", "legs", "identity", "native_validation", "generated_cuda_gate",
                "e270_original_freeze", "generated_e273_gate"):
        base.require(freeze[key] == e275[key], "Original E275 context/gate changed: "+key)
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


remaining_budget = conic.remaining_budget


def _empty_stats():
    return {"solver_calls_kind": SOLVER_KIND, "slsqp_calls": 0, "lp_calls": 0,
            "leg_optimizer_calls": 0, "outer_steps": 0, "socp_constructor_requests": 0,
            "socp_constructor_completed": 0, "socp_constructor_failed": 0,
            "socp_solve_requests": 0, "socp_solve_completed": 0, "socp_solve_failed": 0,
            "socp_iterations_unknown_calls": 0, "clarabel_iterations": 0,
            "socp_constructor_wall_s": 0.0, "socp_solve_wall_s": 0.0,
            "certificate_requests": 0, "certificate_completed": 0, "certificate_failed": 0,
            "certificate_wall_s": 0.0, "certificate_wall_unknown_calls": 0,
            "certified_solved": 0, "certified_almost_solved": 0,
            "accepted_steps": 0, "rejected_proposals": 0, "legs": []}


def stats_for(budget):
    if not hasattr(budget, "e277_stats"):
        budget.e277_stats = _empty_stats()
    return budget.e277_stats


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
        phase = receipt.get("certificate", {"requested": 0, "completed": 0, "failed": 0,
                                             "wall_s": 0., "wall_known": True})
        for key in ("requested", "completed", "failed"):
            target["certificate_"+("requests" if key == "requested" else key)] += phase[key]
        if phase["wall_known"] and target["certificate_wall_s"] is not None:
            target["certificate_wall_s"] += phase["wall_s"]
        elif not phase["wall_known"]:
            target["certificate_wall_s"] = None
            target["certificate_wall_unknown_calls"] += 1
        if receipt.get("solution_valid") and phase["completed"]:
            key = "certified_solved" if receipt["status"] == "Solved" else "certified_almost_solved"
            target[key] += 1


def save_direction(path, problem, solved):
    fields = ("coefficients", "gradient", "normalized_gradient", "gradient_scale", "bounds", "qerr", "lerr",
              "startup_q_jacobian", "startup_ltm_jacobian", "fixed_sse", "available_sse", "current_sse",
              "norm_squared_limits", "radii", "peaks", "radius", "nonnegative_rows")
    arrays = {key: problem[key] for key in fields}
    if solved is not None:
        arrays.update(direction=solved["direction"], solver_s=solved["s"], solver_z=solved["z"])
        if solved["residuals"] is not None:
            arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
        if solved.get("primal_slack") is not None:
            arrays["primal_slack"] = solved["primal_slack"]
    np.savez(path, **arrays)


def fit_certified_leg(evaluator, initial, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """Certified original directions and unchanged actual nonlinear backtracking."""
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
                solved = solve_certified(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock)
                row.update(solved["receipt"])
                row.update(original_valid=solved["original_valid"], original_receipt=solved["original_receipt"])
                _add_solver_cost(solved["receipt"], budget, cost, stats)
            except Exception as error:
                receipt = getattr(error, "conic_receipt", None)
                if receipt is not None:
                    row.update(receipt)
                    _add_solver_cost(receipt, budget, cost, stats)
                partial = getattr(error, "partial_certified_solution", None)
                if partial is not None:
                    saved = base.preserve_primary(error, "E277 partial certified solution persistence failed",
                        lambda raw=partial, idx=index, inputs=problem:
                        save_direction(evaluator.output/f"socp-{idx:03d}.npz", inputs, raw) or True)
                    if saved is not True:
                        error.abort_run = True
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
                proposal = {"outer_step": outer, "alpha": alpha, "radius": radius, "evaluation": None,
                            "next_radius": None}
                history["proposals"].append(proposal)
                if not base.coefficient_valid(candidate, interior=True):
                    proposal["status"] = "rejected_coefficient_before_evaluation"
                else:
                    try:
                        actual = evaluator.evaluate(candidate.ravel())
                    except base.BudgetExceeded as error:
                        saved = base.preserve_primary(error, "E277 interrupted proposal metadata failed",
                            lambda stop=error, entry=proposal, old=current, point=candidate:
                            previous.interrupted_proposal(evaluator, entry, old, point, stop))
                        if saved is not True:
                            error.abort_run = True
                        raise
                    slope, armijo = previous.proposal_numbers(proposal, current, candidate, actual)
                    if previous.eligible(actual) and slope < 0 and actual["objective"] <= armijo:
                        next_radius = min(.5, 2*radius) if alpha == 1 else max(1e-6, alpha*radius)
                        proposal.update(status="accepted", adopted_as_iterate=True, next_radius=next_radius)
                        current, accepted = actual, True
                        cost["accepted_steps"] += 1
                        stats["accepted_steps"] += 1
                        radius = next_radius
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
        saved = base.preserve_primary(primary, "E277 fit history/selection persistence failed", persist_all)
        if saved is not True:
            primary.abort_run = True
        if (getattr(primary, "abort_run", False) or not isinstance(primary, base.BudgetExceeded)
                or primary.scope == "original"):
            raise primary
    else:
        persist_all()
    return selected


class CertifiedRunMeter(base.RunMeter):
    def receipt(self):
        return {**super().receipt(), "optimizer": stats_for(self.budget) if self.budget else _empty_stats()}


def original_run(freeze, output, started):
    from tools.banc_m3_source_coefficient_engine import (
        CudaSourceCoefficientEngine,
        prepare_problem,
    )
    meter = CertifiedRunMeter(output, base.OriginalDeadline(started))
    primary = None
    try:
        return base.execute_original(freeze, output, CudaSourceCoefficientEngine, prepare_problem, meter,
            leg_fit=fit_certified_leg,
            evaluator_factory=functools.partial(base.LegEvaluator, retain_startup_jacobians=True))
    except Exception as error:
        primary = error
        saved = base.preserve_primary(error, "E277 original failure receipt failed",
            lambda stop=error: base.failure_record(stop, None, output, "e277-original", meter.receipt()) or True)
        if saved is not True:
            error.abort_run = True
        raise
    finally:
        if primary is None:
            meter.save()
        else:
            base.preserve_primary(primary, "E277 final run cost persistence failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    base.require(not output.exists(), "Fresh E277 output directory required")
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
        base.preserve_primary(error, "E277 terminal failure receipt failed", persist_terminal)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
