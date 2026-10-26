"""E274 startup-feasible source fitting, explicitly injected into E270.

Freeze uses E270 input/context/reference syntax with a distinct schema and
CONTRACT, six current approved_sources, e270_original_freeze:{input:alias}
(hashed preserved JSON), and generated_e273_gate:{input:alias}. The latter
must pass exactness_valid/speed_valid, median_speedup>=1.5 and pin the current
engine/test source_sha256. Every original E270 input and context stays exact.
Review has schema, verdict='approved', approved_sources and freeze_sha256.
Importing this module never runs original data, CUDA, BANC or a body.
"""

from __future__ import annotations

# Persist a primary calculation error without allowing later storage failures to mask it.
# ruff: noqa: BLE001
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import fit_c_banc_m3_source_coefficients as base

SCHEMA = "flylab.m3.e274.feasible-source-fit.v1"
REVIEW_SCHEMA = "flylab.m3.e274.source-review.v1"
SOURCES = (
    *base.SOURCES,
    "tools/fit_c_banc_m3_feasible_source.py",
    "tests/test_c_banc_feasible_source_fit.py",
)
ENGINE_SOURCES = (
    "tools/banc_m3_source_coefficient_engine.py",
    "tests/test_c_banc_source_coefficient_engine.py",
)
CONTRACT = {k: v for k, v in base.CONTRACT.items() if k not in ("maxiter", "ftol")}
CONTRACT.update(
    solver="scipy.optimize.linprog:highs",
    max_outer_steps=24,
    initial_radius=0.05,
    max_radius=0.5,
    min_radius=1e-6,
    linear_slack_fraction=0.8,
    primal_feasibility_tolerance=1e-9,
    dual_feasibility_tolerance=1e-9,
    lp_maxiter=2000,
    lp_wall_s=1.0,
    alphas=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
    armijo=1e-4,
    generated_e273_min_speedup=1.5,
)


def hashed_gate(freeze, name):
    ref = freeze[name]
    base.require(
        isinstance(ref, dict)
        and set(ref) == {"input"}
        and ref["input"] in freeze["inputs"],
        "Explicit hashed JSON gate required: " + name,
    )
    return json.loads(
        base.safe_path(freeze["inputs"][ref["input"]]["path"]).read_text(
            encoding="utf-8"
        )
    )


def validate_freeze(path, digest, review_path, review_digest):
    freeze = base.verified_json(path, digest)
    review = base.verified_json(review_path, review_digest)
    base.require(
        freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT,
        "Exact E274 schema/contract required",
    )
    sources = freeze.get("approved_sources")
    base.require(
        isinstance(sources, dict) and set(sources) == set(SOURCES),
        "Six E274 source pins required",
    )
    for name, sha in sources.items():
        base.require(
            base.sha256(base.safe_path(name)) == sha,
            "Reviewed E274 source changed: " + name,
        )
    base.require(
        review.get("schema") == REVIEW_SCHEMA
        and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources
        and review.get("freeze_sha256") == digest,
        "Exact E274 source review required",
    )
    inputs = freeze.get("inputs")
    base.require(isinstance(inputs, dict) and inputs, "E274 input SHA map required")
    for spec in inputs.values():
        base.require(
            isinstance(spec, dict) and set(spec) == {"path", "sha256"},
            "Exact input path/SHA required",
        )
        base.require(
            base.sha256(base.safe_path(spec["path"])) == spec["sha256"],
            "Frozen input SHA changed",
        )
    original = hashed_gate(freeze, "e270_original_freeze")
    base.require(
        original.get("schema") == base.SCHEMA
        and original.get("contract") == base.CONTRACT,
        "Preserved original E270 freeze required",
    )
    # The preserved original source pin is historical; live pins are the six above.
    for name, spec in original["inputs"].items():
        base.require(inputs.get(name) == spec, "Original E270 input changed: " + name)
    for key in (
        "contexts",
        "targets",
        "legs",
        "identity",
        "native_validation",
        "generated_cuda_gate",
    ):
        base.require(
            freeze[key] == original[key], "Original E270 context/gate changed: " + key
        )
    for name in ("native_validation", "generated_cuda_gate", "generated_e273_gate"):
        gate = hashed_gate(freeze, name)
        base.require(gate.get("status") == "pass", name + " did not pass")
        if name == "native_validation":
            base.require(
                all(
                    gate.get(k) is True
                    for k in (
                        "counters_valid",
                        "raw_valid",
                        "identity_valid",
                        "decoder_valid",
                    )
                ),
                "Native checks missing",
            )
        if name == "generated_e273_gate":
            base.require(
                gate.get("exactness_valid") is True and gate.get("speed_valid") is True,
                "E273 exactness and speed gate required",
            )
            speed = gate.get("median_speedup")
            base.require(
                isinstance(speed, (float, int))
                and not isinstance(speed, bool)
                and np.isfinite(speed)
                and speed >= 1.5,
                "E273 median speedup below 1.5",
            )
            base.require(
                gate.get("source_sha256") == {k: sources[k] for k in ENGINE_SOURCES},
                "E273 gate source hashes differ",
            )
    return freeze


def remaining_budget(budget):
    now = budget.clock()
    remaining = {
        "leg_s": 30 - (now - budget.leg_started),
        "shared_fit_s": 180 - (now - budget.started),
        "original_s": None
        if budget.deadline is None
        else 290 - (now - budget.deadline.started),
    }
    finite = [v for v in remaining.values() if v is not None]
    time_limit = min(1.0, *finite)
    if time_limit <= 0:
        budget.check()
        raise base.BudgetExceeded("leg")
    return remaining, float(time_limit)


def lp_problem(current, radius):
    c = current["coefficients"].ravel()
    g = current["gradient"]
    h, j = current["constraints"], current["constraint_jacobian"]
    scale = float(np.max(np.abs(g)))
    base.require(
        np.isfinite(scale) and scale > 0, "Nonzero finite gradient required for LP"
    )
    triple = base.triple_constraint(len(c) // 3)
    sums = triple.A @ c
    a = np.concatenate((-j, triple.A, -triple.A))
    b = np.concatenate((0.8 * h, triple.ub - sums, sums - triple.lb))
    bounds = np.column_stack((np.maximum(-radius, -c), np.minimum(radius, 8 - c)))
    return {"c": g / scale, "A_ub": a, "b_ub": b, "bounds": bounds}, scale


def lp_residuals(problem, d):
    bounds = problem["bounds"]
    return np.concatenate(
        (problem["b_ub"] - problem["A_ub"] @ d, d - bounds[:, 0], bounds[:, 1] - d)
    )


def entry_finite(entry):
    keys = ("coefficients", "gradient", "constraints", "constraint_jacobian")
    return (
        all(np.isfinite(entry[k]).all() for k in keys)
        and all(np.isfinite(entry[k]) for k in ("objective", "mse", "baseline_mse"))
        and entry["baseline_mse"] > 0
        and entry["mse"] >= 0
    )


def finite_scalar(value):
    return float(value) if np.isfinite(value) else None


def proposal_numbers(proposal, current, candidate, actual):
    slope = float(current["gradient"] @ (candidate - current["coefficients"]).ravel())
    armijo_bound = float(current["objective"] + 1e-4 * slope)
    proposal.update(
        evaluation=actual["evaluation"],
        evaluation_complete=True,
        directional_derivative=finite_scalar(slope),
        objective=finite_scalar(actual["objective"]),
        armijo_bound=finite_scalar(armijo_bound),
        minimum_actual_constraint=finite_scalar(np.min(actual["constraints"])),
        actual_constraints=[finite_scalar(value) for value in actual["constraints"]],
        mse=finite_scalar(actual["mse"]),
        strict_feasible_train_nonworse=eligible(actual),
    )
    return slope, armijo_bound


def interrupted_proposal(evaluator, proposal, current, candidate, error):
    """Inspect completed cache/raw artifacts without requesting another evaluation."""
    proposal.update(
        status="incomplete_budget_stop",
        budget_scope=error.scope,
        evaluation_complete=False,
        adopted_as_iterate=False,
    )
    actual = evaluator.cache.get(candidate.ravel().tobytes())
    if actual is not None:
        proposal_numbers(proposal, current, candidate, actual)
        proposal.update(
            status="completed_budget_stop_before_iterate_adoption",
            completion_source="coefficient_cache",
        )
    else:
        index = len(evaluator.evaluations)
        startup = (evaluator.output / f"eval-{index:03d}-startup.npz").is_file()
        teacher = (evaluator.output / f"eval-{index:03d}-teacher.npz").is_file()
        proposal.update(
            attempt_index=index,
            startup_raw_saved=startup,
            teacher_raw_saved=teacher,
            completion_source="startup_only_raw"
            if startup and not teacher
            else "raw_present_without_completed_cache"
            if startup or teacher
            else "no_completed_raw",
        )
    return True


def eligible(entry):
    return bool(
        entry_finite(entry)
        and base.coefficient_valid(entry["coefficients"], interior=True)
        and np.all(entry["constraints"] >= 0)
        and entry["mse"] <= entry["baseline_mse"]
    )


def stats_for(budget):
    if not hasattr(budget, "e274_stats"):
        budget.e274_stats = {
            "solver_calls_kind": "actual HiGHS LP calls",
            "slsqp_calls": 0,
            "leg_optimizer_calls": 0,
            "outer_steps": 0,
            "lp_calls": 0,
            "lp_completed_calls": 0,
            "lp_iterations_unknown_calls": 0,
            "highs_iterations": 0,
            "lp_wall_s": 0.0,
            "accepted_steps": 0,
            "rejected_proposals": 0,
            "legs": [],
        }
    return budget.e274_stats


def fit_feasible_leg(
    evaluator, initial, *, lp_solver=linprog, wall_clock=time.perf_counter
):
    """One bounded feasible LP/backtracking pass; select all completed valid evaluations."""
    budget = evaluator.budget
    stats = stats_for(budget)
    stats["leg_optimizer_calls"] += 1
    history = {"outer": [], "proposals": [], "lp": []}
    cost = {
        "leg": Path(evaluator.output).name,
        "outer_steps": 0,
        "lp_calls": 0,
        "lp_completed_calls": 0,
        "lp_iterations_unknown_calls": 0,
        "highs_iterations": 0,
        "lp_wall_s": 0.0,
        "accepted_steps": 0,
        "rejected_proposals": 0,
    }
    stats["legs"].append(cost)
    status, budget_scope, primary = "complete", None, None
    radius = 0.05
    fallback = {
        "coefficients": initial.copy(),
        "objective": 1.0,
        "mse": evaluator.baseline_mse,
        "baseline_mse": evaluator.baseline_mse,
        "valid": False,
        "evaluation": None,
    }

    def persist_history():
        base.write_json(evaluator.output / "feasible-history.json", history)
        base.write_json(
            evaluator.output / "feasible-cost.json",
            {
                **cost,
                "budget": budget.receipt(),
                "solver_calls_kind": "actual HiGHS LP calls",
                "slsqp_calls": 0,
            },
        )

    try:
        budget.check()
        current = evaluator.evaluate(initial.ravel())
        base.require(
            entry_finite(current) and eligible(current),
            "Neutral iterate is not strict feasible/finite",
        )
        for outer in range(24):
            budget.check()
            cost["outer_steps"] += 1
            stats["outer_steps"] += 1
            history["outer"].append(
                {"step": outer, "radius": radius, "evaluation": current["evaluation"]}
            )
            if radius < 1e-6:
                status = "minimum_radius"
                break
            if np.max(np.abs(current["gradient"])) == 0:
                status = "stationary"
                break
            problem, scale = lp_problem(current, radius)
            remaining, time_limit = remaining_budget(budget)
            options = {
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "maxiter": 2000,
                "time_limit": time_limit,
            }
            lp_index = len(history["lp"])
            receipt = {
                "outer_step": outer,
                "radius": radius,
                "remaining_budget": remaining,
                "time_limit_s": time_limit,
                "gradient_scale": scale,
                "status": "requested",
                "options": options,
                "requested": True,
                "completed": False,
            }
            history["lp"].append(receipt)
            budget.solver_calls += 1
            cost["lp_calls"] += 1
            stats["lp_calls"] += 1
            began = wall_clock()
            try:
                result = lp_solver(**problem, method="highs", options=options)
                cost["lp_completed_calls"] += 1
                stats["lp_completed_calls"] += 1
                nit = int(getattr(result, "nit", 0))
                budget.iterations += nit
                cost["highs_iterations"] += nit
                stats["highs_iterations"] += nit
                receipt.update(
                    status=int(result.status),
                    message=str(result.message),
                    nit=nit,
                    success=bool(result.success),
                    completed=True,
                )
            except Exception as error:
                cost["lp_iterations_unknown_calls"] += 1
                stats["lp_iterations_unknown_calls"] += 1
                receipt.update(status="execution_failure", error=str(error))
                raise
            finally:
                elapsed = wall_clock() - began
                receipt["wall_s"] = elapsed
                cost["lp_wall_s"] += elapsed
                stats["lp_wall_s"] += elapsed
            direction = getattr(result, "x", None)
            finite = (
                isinstance(direction, np.ndarray)
                and direction.shape == current["gradient"].shape
                and np.isfinite(direction).all()
            )
            residual = (
                lp_residuals(problem, direction) if finite else np.array([], np.float64)
            )
            receipt["minimum_linear_residual"] = (
                float(np.min(residual)) if len(residual) else None
            )
            np.savez(
                evaluator.output / f"lp-{lp_index:03d}.npz",
                coefficients=current["coefficients"],
                direction=direction if finite else np.array([], np.float64),
                linear_residual=residual,
                normalized_gradient=problem["c"],
                bounds=problem["bounds"],
            )
            persist_history()
            if not result.success or not finite or np.min(residual) < -1e-9:
                status = "bounded_lp_failure"
                break
            budget.check()
            if np.max(np.abs(direction)) == 0:
                status = "zero_direction"
                break
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                budget.check()
                candidate = np.clip(
                    current["coefficients"].ravel() + alpha * direction, 0, 8
                ).reshape(initial.shape)
                proposal = {
                    "outer_step": outer,
                    "alpha": alpha,
                    "radius": radius,
                    "evaluation": None,
                }
                history["proposals"].append(proposal)
                if not base.coefficient_valid(candidate, interior=True):
                    proposal["status"] = "rejected_coefficient_before_evaluation"
                else:
                    try:
                        actual = evaluator.evaluate(candidate.ravel())
                    except base.BudgetExceeded as error:
                        recorded = base.preserve_primary(
                            error,
                            "E274 interrupted proposal metadata failed",
                            lambda stop=error, row=proposal, point=current, proposed=candidate: (
                                interrupted_proposal(
                                    evaluator, row, point, proposed, stop
                                )
                            ),
                        )
                        if recorded is not True:
                            error.abort_run = True
                        raise
                    slope, armijo_bound = proposal_numbers(
                        proposal, current, candidate, actual
                    )
                    if (
                        eligible(actual)
                        and slope < 0
                        and actual["objective"] <= armijo_bound
                    ):
                        proposal["status"] = "accepted"
                        proposal["adopted_as_iterate"] = True
                        current = actual
                        accepted = True
                        cost["accepted_steps"] += 1
                        stats["accepted_steps"] += 1
                        if alpha == 1.0:
                            radius = min(0.5, 2 * radius)
                        persist_history()
                        break
                    proposal["status"] = "rejected_true_constraint_or_armijo"
                cost["rejected_proposals"] += 1
                stats["rejected_proposals"] += 1
                persist_history()
            if not accepted:
                radius *= 0.5
        else:
            status = "outer_step_limit"
    except base.BudgetExceeded as error:
        primary = error
        status = str(error)
        budget_scope = error.scope
    except Exception as error:
        primary = error
        status = "execution_failure"
    # A completed evaluation that caused a budget stop is still eligible for selection.
    completed = [e for e in evaluator.evaluations if eligible(e)]
    selected = base.choose_candidate(completed, fallback)
    selected = dict(selected, status=status, budget_scope=budget_scope)

    def persist_all():
        persist_history()
        np.savez(
            evaluator.output / "selected.npz",
            **{k: v for k, v in selected.items() if isinstance(v, np.ndarray)},
        )
        base.write_json(
            evaluator.output / "selection.json",
            {
                "status": status,
                "budget_scope": budget_scope,
                "evaluation": selected["evaluation"],
                "objective": selected["objective"],
                "mse": selected["mse"],
                "cost": cost,
                "budget": budget.receipt(),
                "solver_calls_kind": "actual HiGHS LP calls",
                "slsqp_calls": 0,
            },
        )
        return True

    if primary is not None:
        saved = base.preserve_primary(
            primary, "E274 fit history/selection persistence failed", persist_all
        )
        if saved is not True:
            primary.abort_run = True
        if getattr(primary, "abort_run", False) or not isinstance(
            primary, base.BudgetExceeded
        ):
            raise primary
    else:
        persist_all()
    return selected


class FeasibleRunMeter(base.RunMeter):
    def receipt(self):
        return {
            **super().receipt(),
            "optimizer": stats_for(self.budget)
            if self.budget
            else {
                "solver_calls_kind": "actual HiGHS LP calls",
                "slsqp_calls": 0,
                "leg_optimizer_calls": 0,
                "outer_steps": 0,
                "lp_calls": 0,
                "lp_completed_calls": 0,
                "lp_iterations_unknown_calls": 0,
                "highs_iterations": 0,
                "lp_wall_s": 0.0,
                "legs": [],
            },
        }


def original_run(freeze, output, started):
    from tools.banc_m3_source_coefficient_engine import (
        CudaSourceCoefficientEngine,
        prepare_problem,
    )

    meter = FeasibleRunMeter(output, base.OriginalDeadline(started))
    primary = None
    try:
        return base.execute_original(
            freeze,
            output,
            CudaSourceCoefficientEngine,
            prepare_problem,
            meter,
            leg_fit=fit_feasible_leg,
        )
    except Exception as error:
        primary = error
        base.failure_record(error, None, output, "e274-original", meter.receipt())
        raise
    finally:
        if primary is None:
            meter.save()
        else:
            base.preserve_primary(
                primary, "E274 final run cost persistence failed", meter.save
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "freeze",
        "freeze-sha256",
        "source-review",
        "source-review-sha256",
        "output",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    base.require(not output.exists(), "Fresh E274 output directory required")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(
        output / "cost.json",
        {
            "phase": "freeze_validation",
            "original_wall_s": 0.0,
            "solver_calls_kind": "actual HiGHS LP calls",
            "slsqp_calls": 0,
            "optimizer_calls": 0,
        },
    )
    try:
        freeze = validate_freeze(
            Path(args.freeze),
            args.freeze_sha256,
            Path(args.source_review),
            args.source_review_sha256,
        )
        verdict = original_run(freeze, output, started)
        base.write_json(
            output / "result.json",
            {
                "status": "complete",
                "original_wall_s": time.perf_counter() - started,
                **verdict,
            },
        )
        return 0
    except Exception as error:
        record = {
            "status": "budget_stop"
            if isinstance(error, base.BudgetExceeded)
            else "error",
            "error": str(error),
            "exception": type(error).__name__,
            "notes": getattr(error, "__notes__", []),
            "original_wall_s": time.perf_counter() - started,
        }

        def persist_terminal():
            cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
            cost["original_wall_s"] = record["original_wall_s"]
            base.write_json(output / "cost.json", cost)
            base.write_json(output / "result.json", record)

        base.preserve_primary(
            error,
            "E274 terminal failure receipt failed",
            persist_terminal,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
