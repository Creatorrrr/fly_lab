"""E281 startup-only fixed-quota feasibility, with bounded fresh-J corrections."""
from __future__ import annotations

# Failures must preserve the primary exception and attempted-work receipts.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import banc_m3_feasibility_restoration as restoration

base = restoration.base
CONTRACT = {k: copy.deepcopy(restoration.CONTRACT[k]) for k in
    ("max_corrections", "original_radius", "linear_absolute_margin", "selector_margin", "primal_tolerance",
     "dual_tolerance", "stationarity_tolerance", "gap_lower_relative_tolerance", "gap_upper_relative_tolerance",
     "complementarity_relative_tolerance", "statuses", "hessian", "clip", "clarabel_version", "max_iter",
     "time_limit_s", "max_threads", "tol_gap_abs", "tol_gap_rel", "tol_feas")}
CONTRACT.update(q_peak=.036, ltm_peak=.025, q_global_rms=.0162, ltm_leg_rms=.009,
    q_global_elements=4158, ltm_leg_elements=99, correction_radius=.05,
    reproduction_q_ltm=1e-10, reproduction_jacobian=1e-9,
    quota_allocation="neutral C-order f64 SSE share", f32="one float32 then float64 round-trip; no clip",
    objective_calls=0, teacher_calls=0, armijo=False)
CASE_KEYS = {"neutral_coefficients", "neutral_startup_q", "neutral_startup_ltm", "neutral_q_jacobian",
             "neutral_ltm_jacobian", "target_q", "target_ltm", "q_sse_quota", "ltm_sse_quota"}


def sum_square(value):
    return float(np.sum(np.square(np.array(value, dtype=np.float64, order="C", copy=True)), dtype=np.float64))


def quota_rows(neutral_errors_by_leg):
    base.require(len(neutral_errors_by_leg) == 6, "Six neutral error arrays required")
    errors = [np.asarray(e) for e in neutral_errors_by_leg]
    base.require(all(e.dtype == np.float64 and e.shape == (99, 7) and np.isfinite(e).all() for e in errors),
                 "Finite complete float64 neutral startup errors required")
    sse = np.array([sum_square(e) for e in errors])
    total = float(np.sum(sse, dtype=np.float64))
    base.require(np.isfinite(total) and total > 0 and np.all(sse > 0), "Positive finite fixed q shares required")
    shares = sse/total
    qtotal, lquota = 4158*(.0162*(1-1e-4))**2, 99*(.009*(1-1e-4))**2
    return [{"neutral_sse": float(s), "share": float(w), "q_sse_quota": float(w*qtotal), "ltm_sse_quota": lquota}
            for s, w in zip(sse, shares, strict=True)]


def validate_case(case):
    base.require(set(case) == CASE_KEYS and all(isinstance(v, np.ndarray) and v.dtype == np.float64
        and np.isfinite(v).all() for v in case.values()), "Exact nine finite float64 case keys required")
    c = case["neutral_coefficients"]
    base.require(c.ndim == 2 and c.shape[1] == 3 and 1 <= len(c) <= 1660
        and np.array_equal(c, base.neutral(len(c))), "Exact original neutral coefficient triples required")
    shapes = {"neutral_startup_q": (101, 7), "neutral_startup_ltm": (101, 1), "neutral_q_jacobian": (99, 7, c.size),
              "neutral_ltm_jacobian": (99, 1, c.size), "target_q": (99, 7), "target_ltm": (99, 1),
              "q_sse_quota": (), "ltm_sse_quota": ()}
    base.require(all(case[k].shape == shape for k, shape in shapes.items()), "Complete original startup case axes required")
    q, ltm = float(case["q_sse_quota"]), float(case["ltm_sse_quota"])
    base.require(0 < q <= 4158*(.0162*(1-1e-4))**2 and ltm == 99*(.009*(1-1e-4))**2
        and np.sqrt(q) >= 1e-6 and np.sqrt(ltm) >= 1e-6, "Positive bounded fixed startup quotas required")


def validate_entry(entry, *, derivative):
    c = np.asarray(entry["coefficients"])
    base.require(c.dtype == np.float64 and c.ndim == 2 and c.shape[1] == 3 and np.isfinite(c).all(),
                 "Finite complete evaluated coefficient axes required")
    shapes = {"startup_q": (101, 7), "startup_ltm": (101, 1), "qerr": (99, 7), "lerr": (99, 1)}
    if derivative:
        shapes.update(startup_q_jacobian=(99, 7, c.size), startup_ltm_jacobian=(99, 1, c.size))
    for name, shape in shapes.items():
        v = np.asarray(entry[name])
        base.require(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all(), "Finite complete evaluation required: "+name)


def coefficient_checks(c, origin, correction=None):
    c, origin = np.asarray(c), np.asarray(origin)
    shape = c.dtype == origin.dtype == np.float64 and c.shape == origin.shape
    checks = {"coefficient": bool(shape and base.coefficient_valid(c, interior=True)),
              "neutral_trust": bool(shape and np.isfinite(c).all() and np.all(c >= origin-.05) and np.all(c <= origin+.05))}
    if correction is not None:
        z = np.asarray(correction)
        checks["correction"] = bool(z.dtype == np.float64 and z.size == c.size and np.isfinite(z).all()
                                    and np.all(np.abs(z) <= .05))
    return {"checks": checks, "pass": bool(all(checks.values()))}


def strict_startup(entry, origin, q_quota, ltm_quota):
    validate_entry(entry, derivative=False)
    checks = coefficient_checks(entry["coefficients"], origin)["checks"]
    metrics = {}
    for key, error, peak, quota in (("q", entry["qerr"], .036*(1-1e-4), q_quota),
                                  ("ltm", entry["lerr"], .025*(1-1e-4), ltm_quota)):
        sse, maximum = sum_square(error), float(np.max(np.abs(error)))
        metrics[key] = {"sse": sse, "sse_quota": float(quota), "max_abs": maximum, "peak_limit": peak}
        checks[key+"_peak"], checks[key+"_quota"] = maximum <= peak, sse <= quota
    return {"checks": checks, "metrics": metrics, "pass": bool(all(checks.values()))}


def startup_margin_problem(current, origin, q_quota, ltm_quota):
    """Explicit stronger original QP/SOC; infeasible startup current is allowed."""
    validate_entry(current, derivative=True)
    c, origin = np.asarray(current["coefficients"]), np.asarray(origin)
    base.require(coefficient_checks(c, origin)["pass"] and np.array_equal(origin, base.neutral(len(c))),
                 "Original interior coefficients and neutral trust required")
    quotas = np.array([q_quota, ltm_quota], np.float64)
    base.require(np.isfinite(quotas).all() and 0 < quotas[0] <= 4158*(.0162*(1-1e-4))**2
        and quotas[1] == 99*(.009*(1-1e-4))**2, "Fixed bounded q and LTM quotas required")
    radii, peaks = np.sqrt(quotas)-1e-6, np.array([.036, .025])*(1-1e-4)-1e-6
    base.require(np.all(radii >= 0), "Absolute norm margin exceeds quota")
    n = c.size
    lo = np.maximum.reduce((-c.ravel(), origin.ravel()-.05-c.ravel(), np.full(n, -.05)))
    hi = np.minimum.reduce((8-c.ravel(), origin.ravel()+.05-c.ravel(), np.full(n, .05)))
    triple, identity = base.triple_constraint(len(c)), sparse.eye(n, format="csc", dtype=np.float64)
    blocks = [identity, -identity, sparse.csc_matrix(triple.A), sparse.csc_matrix(-triple.A)]
    rhs = [hi, -lo, np.asarray(triple.ub)-c.sum(1), c.sum(1)-np.asarray(triple.lb)]
    errors = [np.array(current[k], order="C", copy=True) for k in ("qerr", "lerr")]
    js = [np.array(current[k], order="C", copy=True) for k in ("startup_q_jacobian", "startup_ltm_jacobian")]
    for e, j, peak in zip(errors, js, peaks, strict=True):
        blocks.extend((sparse.csc_matrix(j.reshape(e.size, n)), sparse.csc_matrix(-j.reshape(e.size, n))))
        rhs.extend((peak-e.ravel(), peak+e.ravel()))
    rows = sum(len(r) for r in rhs)
    for e, j, radius in zip(errors, js, radii, strict=True):
        blocks.append(sparse.vstack((sparse.csc_matrix((1, n)), sparse.csc_matrix(-j.reshape(e.size, n))), format="csc"))
        rhs.append(np.r_[radius, e.ravel()])
    return {"P": identity, "q": np.zeros(n), "A": sparse.vstack(blocks, format="csc"), "b": np.concatenate(rhs),
        "cones": [clarabel.NonnegativeConeT(rows), clarabel.SecondOrderConeT(694), clarabel.SecondOrderConeT(100)],
        "nonnegative_rows": rows, "coefficients": c.copy(), "before_coefficients": origin.copy(),
        "bounds": np.column_stack((lo, hi)), "qerr": errors[0], "lerr": errors[1],
        "startup_q_jacobian": js[0], "startup_ltm_jacobian": js[1],
        "quotas": quotas, "peaks": peaks, "radii": radii, "radius": .05, "correction_radius": .05}


def validate_problem(problem):
    """Reject stale/mutated sparse rows or stronger-limit metadata before solve."""
    restoration._validate_problem(problem)
    quotas = problem["quotas"]
    base.require(isinstance(quotas, np.ndarray) and quotas.dtype == np.float64 and quotas.shape == (2,), "Original quota axes required")
    current = {k: problem[k] for k in ("coefficients", "qerr", "lerr", "startup_q_jacobian", "startup_ltm_jacobian")}
    # The equation validator needs only finite dummy complete responses, not target replay.
    current.update(startup_q=np.zeros((101, 7)), startup_ltm=np.zeros((101, 1)))
    exact = startup_margin_problem(current, problem["before_coefficients"], *quotas)
    for key in ("P", "A"):
        delta = problem[key]-exact[key]
        delta.eliminate_zeros()
        base.require(delta.nnz == 0, "Strong original sparse equation differs: "+key)
    for key in ("q", "b", "bounds", "peaks", "radii", "quotas"):
        base.require(np.array_equal(problem[key], exact[key]), "Strong original metadata/RHS differs: "+key)
    base.require(problem["radius"] == problem["correction_radius"] == .05, "Exact original .05 limits required")
    return True


def solve_margin(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    validate_problem(problem)
    return restoration.solve_restoration(problem, time_limit, solver_factory=solver_factory, wall_clock=wall_clock)


def problem_arrays(problem):
    arrays = {k: v for k, v in problem.items() if isinstance(v, (np.ndarray, float))}
    for key in ("P", "A"):
        matrix = problem[key]
        arrays.update({key+"_data": matrix.data, key+"_indices": matrix.indices,
                       key+"_indptr": matrix.indptr, key+"_shape": np.array(matrix.shape, np.int64)})
    return arrays


def run_leg(case, evaluate, *, persist, check=lambda: None, remaining=lambda: 2.,
            solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """evaluate(c, derivative=bool) advances startup only, even for the f32 replay."""
    validate_case(case)
    origin, quotas = case["neutral_coefficients"], (float(case["q_sse_quota"]), float(case["ltm_sse_quota"]))
    cost = {k: restoration.phase() for k in ("evaluation", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_jacobians_used=0, derivative_evaluations_requested=0)
    result = {"status": "requested", "run_valid": False, "f64_success": False, "f32_success": False,
              "reproduction": {}, "corrections": [], "cost": cost}
    primary = None
    def save_entry(label, entry):
        persist(label, {k: v for k, v in entry.items() if isinstance(v, np.ndarray)},
                {k: v for k, v in entry.items() if not isinstance(v, np.ndarray)})
    def observe(c, label, derivative):
        check()
        base.require(cost["evaluation"]["requested"] < 4, "At most four startup evaluations per leg")
        cost["derivative_evaluations_requested"] += int(derivative)
        try:
            row = restoration.metered(cost, "evaluation", lambda: evaluate(c, derivative=derivative), wall_clock)
        except Exception as error:
            row = getattr(error, "completed_margin_entry", None)
            if row is not None:
                cost["evaluation"]["completed"] += 1
            else:
                row = getattr(error, "phase_partial_result", None)
            if row is not None:
                base.preserve_primary(error, "Completed margin evaluation persistence failed", lambda: save_entry(label, row))
            raise
        save_entry(label, row)
        validate_entry(row, derivative=derivative)
        base.require(np.array_equal(row["coefficients"], c)
            and np.array_equal(row["qerr"], row["startup_q"][1:100]-case["target_q"])
            and np.array_equal(row["lerr"], row["startup_ltm"][1:100]-case["target_ltm"]), "Actual coefficient/error/target mismatch")
        check()
        return row
    def account(receipt):
        for key in ("constructor", "solve", "certificate"):
            for field in ("requested", "completed", "failed"):
                cost[key][field] += receipt[key][field]
            if not receipt[key]["wall_known"]:
                cost[key].update(wall_s=None, wall_known=False)
            elif cost[key]["wall_s"] is not None:
                cost[key]["wall_s"] += receipt[key]["wall_s"]
        if receipt["iterations_known"]:
            cost["iterations"] += receipt["iterations"]
        elif receipt["solve"]["requested"]:
            cost["iterations_unknown_calls"] += 1
    try:
        current = observe(origin.copy(), "neutral", True)
        for actual, saved, tol in (("startup_q", "neutral_startup_q", 1e-10), ("startup_ltm", "neutral_startup_ltm", 1e-10),
            ("startup_q_jacobian", "neutral_q_jacobian", 1e-9), ("startup_ltm_jacobian", "neutral_ltm_jacobian", 1e-9)):
            result["reproduction"][actual] = base.numerical_gate(current[actual], case[saved], tol)
        persist("reproduction", None, result["reproduction"])
        if not all(gate["pass"] for gate in result["reproduction"].values()):
            raise restoration.ReproductionError("Neutral q/LTM/both startup Jacobian reproduction failed")
        initial = strict_startup(current, origin, *quotas)
        result["neutral_judgement"] = initial
        if initial["pass"]:
            result["f64_success"] = True
        for iteration in range(2):
            if result["f64_success"]:
                break
            check()
            problem = startup_margin_problem(current, origin, *quotas)
            cost["fresh_jacobians_used"] += 1
            prefix = f"correction-{iteration:02d}"
            persist(prefix+"-problem", problem_arrays(problem), {"jacobian_evaluation": current["evaluation"]})
            check()
            limit = min(2., remaining())
            if limit <= 0:
                raise base.BudgetExceeded("original")
            try:
                solved = solve_margin(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock)
                account(solved["receipt"])
            except Exception as error:
                receipt = getattr(error, "restoration_receipt", None)
                if receipt is not None:
                    account(receipt)
                    result["corrections"].append({"iteration": iteration, "receipt": receipt, "status": "execution_failure"})
                    raw = getattr(error, "partial_restoration_solution", None)
                    arrays = {} if raw is None else {key: np.asarray(getattr(raw, attr), np.float64)
                        for key, attr in (("correction", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                    base.preserve_primary(error, "Partial margin solve persistence failed",
                        lambda prefix=prefix, arrays=arrays, receipt=receipt: persist(prefix+"-solve", arrays, receipt))
                raise
            arrays = {k: solved[k] for k in ("correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
            if solved["residuals"] is not None:
                arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
            persist(prefix+"-solve", arrays, solved["receipt"])
            row = {"iteration": iteration, "receipt": solved["receipt"], "actual_evaluated": False}
            result["corrections"].append(row)
            if not solved["valid"]:
                result["status"] = "uncertified_correction"
                break
            c = current["coefficients"]+solved["correction"].reshape(origin.shape)
            bounds = coefficient_checks(c, origin, solved["correction"])
            persist(prefix+"-coefficients", {"coefficients": c}, {"modified_after_solve": False, **bounds})
            if not bounds["pass"]:
                result["status"] = row["status"] = "strict_coefficient_trust_or_correction_rejection"
                break
            current = observe(c, prefix+"-actual", True)
            judgement = strict_startup(current, origin, *quotas)
            row.update(actual_evaluated=True, judgement=judgement, status="f64_success" if judgement["pass"] else "actual_margin_rejection")
            persist(prefix+"-judgement", None, row)
            if judgement["pass"]:
                result["f64_success"] = True
                break
        else:
            result["status"] = "two_corrections_rejected"
        if result["f64_success"]:
            c32 = current["coefficients"].astype(np.float32)
            rounded = c32.astype(np.float64)
            bounds = coefficient_checks(rounded, origin)
            persist("f32-coefficients", {"coefficients32": c32, "coefficients": rounded}, {"modified_after_roundtrip": False, **bounds})
            result["f32_bounds"] = bounds
            if not bounds["pass"]:
                result["status"] = "f32_coefficient_or_trust_rejection_before_evaluation"
            else:
                final = observe(rounded, "f32-actual", False)
                judgement = strict_startup(final, origin, *quotas)
                result.update(f32_success=judgement["pass"], f32_judgement=judgement,
                    status="supported" if judgement["pass"] else "f32_startup_rejection")
                persist("f32-judgement", None, judgement)
        result["run_valid"] = True
    except Exception as error:
        primary = error
        result.update(status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
    if primary is None:
        try:
            persist("leg-result", None, result)
        except Exception as error:
            result.update(status="invalid", run_valid=False, error=str(error), exception=type(error).__name__)
            error.margin_leg_result, error.abort_run = result, True
            raise
    else:
        primary.margin_leg_result = result
        saved = base.preserve_primary(primary, "Margin failure persistence failed", lambda: persist("leg-result", None, result) or True)
        if saved is not True:
            primary.abort_run = True
        raise primary
    return result
