"""E284 RF-only full hard-domain capacity; no import-time model or solver calls."""
from __future__ import annotations

# Retain primary exceptions through secondary timing or persistence failures.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import banc_m3_startup_margin as margin

base, metering = margin.base, margin.restoration
CONTRACT = {k: copy.deepcopy(margin.CONTRACT[k]) for k in
    ("max_corrections", "linear_absolute_margin", "selector_margin", "primal_tolerance", "dual_tolerance",
     "stationarity_tolerance", "gap_lower_relative_tolerance", "gap_upper_relative_tolerance",
     "complementarity_relative_tolerance", "statuses", "hessian", "clip", "clarabel_version", "max_iter",
     "max_threads", "tol_gap_abs", "tol_gap_rel", "tol_feas", "reproduction_q_ltm", "reproduction_jacobian")}
CONTRACT.update(max_corrections=4, time_limit_s=4., coefficient_domain="individual[0,8];triple interior;no neutral or step trust",
    coefficient_endpoints="0 and 8 exactly f32 representable;raw candidate and f32 triple checked exactly;no repair",
    q_peak=.036, q_sse_limit=4158*(.0162*(1-1e-4))**2, ltm_peak=.025, ltm_sse_limit=99*(.009*(1-1e-4))**2,
    scope="RF only;global q SSE budget used as RF upper bound;no global norm or initialization claim",
    sequence="neutral reproduce;max4 fresh-J minnorm corrections;single f32 actual replay if f64 passed",
    f32="one float32 then float64 round-trip;no clip", teacher_calls=0, objective_calls=0, armijo=False)
PROBLEM_KEYS = {"P", "q", "A", "b", "cones", "cone_dimensions", "coefficients", "bounds", "coefficient_bounds",
                "qerr", "lerr", "q_jacobian", "ltm_jacobian", "peaks", "radii"}


def capacity_problem(entry):
    """Explicit RF minimum-norm QP with hard coefficients, peaks and two SOCs."""
    margin.validate_entry(entry, derivative=True)
    c = entry["coefficients"].copy(); n = c.size
    base.require(3 <= n <= 783 and base.coefficient_valid(c, interior=True), "Bounded original hard coefficient triples required")
    identity = sparse.eye(n, format="csc", dtype=np.float64)
    triple = sparse.csc_matrix((np.ones(n), (np.repeat(np.arange(n//3), 3), np.arange(n))), shape=(n//3, n))
    endpoints = np.tile([0., 8.], (n, 1)); bounds = endpoints-c.ravel()[:, None]
    qe, le = (entry[k].ravel(order="C").copy() for k in ("qerr", "lerr"))
    qj, lj = (sparse.csc_matrix(entry[k].reshape(width, n)) for k, width in
              (("startup_q_jacobian", 693), ("startup_ltm_jacobian", 99)))
    peaks = np.array([.036, .025])*(1-1e-4)-1e-6
    radii = np.sqrt(np.array([CONTRACT["q_sse_limit"], CONTRACT["ltm_sse_limit"]]))-1e-6
    blocks = [identity, -identity, triple, -triple, qj, -qj, lj, -lj]
    rhs = [bounds[:, 1], -bounds[:, 0], 8*(1-1e-6)-c.sum(1), c.sum(1)-.125*(1+1e-6),
           peaks[0]-qe, peaks[0]+qe, peaks[1]-le, peaks[1]+le]
    rows = sum(len(v) for v in rhs)
    for error, jacobian, radius in zip((qe, le), (qj, lj), radii, strict=True):
        blocks.append(sparse.vstack((sparse.csc_matrix((1, n)), -jacobian), format="csc"))
        rhs.append(np.r_[radius, error])
    dims = (rows, 694, 100)
    return {"P": identity, "q": np.zeros(n), "A": sparse.vstack(blocks, format="csc"), "b": np.concatenate(rhs),
        "cones": [clarabel.NonnegativeConeT(rows), clarabel.SecondOrderConeT(694), clarabel.SecondOrderConeT(100)],
        "cone_dimensions": dims, "coefficients": c, "bounds": bounds, "coefficient_bounds": endpoints,
        "qerr": qe, "lerr": le, "q_jacobian": qj, "ltm_jacobian": lj, "peaks": peaks, "radii": radii}


def validate_problem(p):
    base.require(set(p) == PROBLEM_KEYS, "Exact full-domain problem;no dormant trust metadata")
    c = p["coefficients"]
    base.require(isinstance(c, np.ndarray) and c.dtype == np.float64 and 1 <= len(c) <= 261
        and base.coefficient_valid(c, interior=True), "Original bounded hard coefficient domain required")
    n, dims = c.size, (2*c.size+2*len(c)+1584, 694, 100)
    for key, shape in (("P", (n, n)), ("A", (sum(dims), n)), ("q_jacobian", (693, n)), ("ltm_jacobian", (99, n))):
        v = p[key]
        base.require(sparse.isspmatrix_csc(v) and v.dtype == np.float64 and v.shape == shape
            and v.has_canonical_format and v.has_sorted_indices and np.isfinite(v.data).all(), "Original canonical finite CSC required: "+key)
    for key, shape in (("q", (n,)), ("b", (sum(dims),)), ("bounds", (n, 2)), ("coefficient_bounds", (n, 2)),
                       ("qerr", (693,)), ("lerr", (99,)), ("peaks", (2,)), ("radii", (2,))):
        v = p[key]
        base.require(isinstance(v, np.ndarray) and v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all(), "Original complete metadata required: "+key)
    base.require(tuple(p["cone_dimensions"]) == dims and len(p["cones"]) == 3
        and isinstance(p["cones"][0], clarabel.NonnegativeConeT)
        and all(isinstance(v, clarabel.SecondOrderConeT) for v in p["cones"][1:])
        and tuple(v.dim for v in p["cones"]) == dims, "Exact nonnegative/two-SOC dimensions required")
    entry = {"coefficients": c, "qerr": p["qerr"].reshape(99, 7), "lerr": p["lerr"].reshape(99, 1),
        "startup_q": np.zeros((101, 7)), "startup_ltm": np.zeros((101, 1)),
        "startup_q_jacobian": p["q_jacobian"].toarray().reshape(99, 7, n),
        "startup_ltm_jacobian": p["ltm_jacobian"].toarray().reshape(99, 1, n)}
    exact = capacity_problem(entry)
    for key in ("P", "A"):
        delta = p[key]-exact[key]; delta.eliminate_zeros()
        base.require(delta.nnz == 0, "RF original sparse equation changed: "+key)
    for key in ("q", "b", "bounds", "coefficient_bounds", "peaks", "radii"):
        base.require(np.array_equal(p[key], exact[key]), "RF original metadata/RHS changed: "+key)
    return dims


def capacity_residuals(p, correction):
    d = np.asarray(correction)
    base.require(d.dtype == np.float64 and d.shape == p["q"].shape and np.isfinite(d).all(), "Finite complete RF raw correction required")
    c = p["coefficients"].ravel()+d; sums = c.reshape(-1, 3).sum(1)
    qp, lp = p["qerr"]+p["q_jacobian"]@d, p["lerr"]+p["ltm_jacobian"]@d
    residuals = {"coefficient": np.r_[c, 8-c, sums-.125*(1+1e-6), 8*(1-1e-6)-sums],
        "bounds": np.r_[d-p["bounds"][:, 0], p["bounds"][:, 1]-d],
        "peak": np.r_[p["peaks"][0]-np.abs(qp), p["peaks"][1]-np.abs(lp)],
        "soc": p["radii"]-np.array([np.linalg.norm(qp), np.linalg.norm(lp)])}
    return {**residuals, "minimum": float(min(np.min(v) for v in residuals.values()))}


def capacity_certificate(p, correction, solver_s, solver_y, status):
    dims = validate_problem(p)
    d, s, y = (np.asarray(v) for v in (correction, solver_s, solver_y))
    checks = {k: False for k in ("input_valid", "status", "residual", "primal_cone", "dual_cone", "finite_math", "stationarity", "gap", "complementarity")}
    checks["status"] = str(status) in ("Solved", "AlmostSolved")
    receipt = {"status": str(status), "valid": False, "checks": checks}
    checks["input_valid"] = all(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all()
        for v, shape in ((d, p["q"].shape), (s, p["b"].shape), (y, p["b"].shape)))
    if not checks["input_valid"]: return {"valid": False, "receipt": receipt, "primal_slack": None, "residuals": None}
    with np.errstate(over="ignore", invalid="ignore"):
        exact, residuals = p["b"]-p["A"]@d, capacity_residuals(p, d)
        cones = [metering.original_certificate._cone_slacks(v, dims) for v in (exact, s, y)]
        w = p["A"].T@y; stationarity = np.max(abs(d+w))
        primal, dual = .5*np.dot(d, d), -.5*np.dot(w, w)-np.dot(p["b"], y)
        gap, scale = primal-dual, 1+max(abs(primal), abs(dual))
        comp, affine = np.dot(exact, y), np.max(abs(exact-s))
    checks.update(finite_math=bool(all(np.isfinite(v) for v in (residuals["minimum"], stationarity, primal, dual, gap, scale, comp, affine,
        *(v for cone in cones for v in cone)))), residual=bool(residuals["minimum"] >= -1e-8),
        primal_cone=bool(all(v >= -1e-8 for v in cones[0])), dual_cone=bool(all(v >= 0 for v in cones[2])),
        stationarity=bool(stationarity <= 1e-8), gap=bool(-1e-10*scale <= gap <= 1e-8*scale),
        complementarity=bool(abs(comp) <= 1e-8*scale))
    number = metering.original_certificate._number
    receipt.update(valid=bool(all(checks.values())), minimum_residual=number(residuals["minimum"]), raw_affine_infinity_norm=number(affine),
        raw_cone_slacks=[number(v) for v in cones[1]], primal_cone_slacks=[number(v) for v in cones[0]], dual_cone_slacks=[number(v) for v in cones[2]],
        stationarity_infinity_norm=number(stationarity), primal_objective=number(primal), dual_lower_bound=number(dual), gap=number(gap),
        objective_scale=number(scale), complementarity=number(comp))
    return {"valid": receipt["valid"], "receipt": receipt, "primal_slack": exact, "residuals": residuals}


def solve_capacity(p, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    validate_problem(p)
    base.require(clarabel.__version__ == "0.11.1" and np.isfinite(time_limit) and 0 < time_limit <= 4, "Frozen Clarabel and bounded RF time required")
    settings = clarabel.DefaultSettings()
    applied = {"verbose": False, "max_iter": 100, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
               "tol_feas": 1e-8, "max_threads": 1, "time_limit": float(time_limit)}
    for name, value in applied.items(): setattr(settings, name, value); base.require(getattr(settings, name) == value, "Frozen setting changed")
    receipt = {"settings": applied, "iterations": None, "iterations_known": False, "status": "requested",
               **{k: metering.phase() for k in ("constructor", "solve", "certificate")}}
    raw = None
    try:
        solver = metering.metered(receipt, "constructor", lambda: solver_factory(p["P"], p["q"], p["A"], p["b"], p["cones"], settings), wall_clock)
        raw = metering.metered(receipt, "solve", solver.solve, wall_clock)
        receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        d, s, y = (np.asarray(v, np.float64) for v in (raw.x, raw.s, raw.z))
        cert = metering.metered(receipt, "certificate", lambda: capacity_certificate(p, d, s, y, raw.status), wall_clock)
        receipt.update(solution_valid=cert["valid"], certificate_receipt=cert["receipt"], solver_info={k: metering.previous.finite_scalar(getattr(raw, k))
            for k in ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual") if hasattr(raw, k)})
        return {**cert, "correction": d, "solver_s": s, "solver_y": y, "receipt": receipt}
    except Exception as error:
        if raw is None and getattr(error, "restoration_phase", None) == "solve":
            raw = getattr(error, "phase_partial_result", None)
            if raw is not None: receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        error.capacity_solve_receipt, error.partial_capacity_solution = copy.deepcopy(receipt), raw
        raise


def strict_rf(entry):
    margin.validate_entry(entry, derivative=False)
    checks, metrics = {"coefficient": base.coefficient_valid(entry["coefficients"], interior=True)}, {}
    for name, error, peak, sse in (("q", entry["qerr"], .036*(1-1e-4), CONTRACT["q_sse_limit"]),
                                   ("ltm", entry["lerr"], .025*(1-1e-4), CONTRACT["ltm_sse_limit"])):
        total, maximum = margin.sum_square(error), float(np.max(abs(error)))
        checks[name+"_peak"], checks[name+"_sse"] = maximum <= peak, total <= sse
        metrics[name] = {"sse": total, "sse_limit": sse, "peak": maximum, "peak_limit": peak}
    return {"pass": bool(all(checks.values())), "checks": checks, "metrics": metrics, "rf_only": True, "global_norm_supported": False}


def problem_arrays(p):
    arrays = {k: v for k, v in p.items() if isinstance(v, np.ndarray)}
    arrays["cone_dimensions"] = np.array(p["cone_dimensions"], np.int64)
    for key in ("P", "A", "q_jacobian", "ltm_jacobian"):
        m = p[key]; arrays.update({key+"_data": m.data, key+"_indices": m.indices, key+"_indptr": m.indptr, key+"_shape": np.array(m.shape, np.int64)})
    return arrays


def run_capacity(case, evaluate, *, persist, check=lambda: None, remaining=lambda: 4.,
                 solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter, solve=solve_capacity):
    margin.validate_case(case)
    cost = {k: metering.phase() for k in ("evaluation", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_jacobians_used=0)
    result = {"status": "requested", "run_valid": False, "f64_success": False, "f32_success": False,
        "rf_capacity_supported": False, "initialization_supported": False, "global_norm_supported": False,
        "reproduction": {}, "trajectory": [], "corrections": [], "cost": cost}
    primary = None
    def save_entry(label, row):
        persist(label, {k: v for k, v in row.items() if isinstance(v, np.ndarray)}, {k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
    def observe(c, label, derivative):
        check(); row = None
        try:
            row = metering.metered(cost, "evaluation", lambda: evaluate(c, derivative=derivative, label=label), wall_clock)
            margin.validate_entry(row, derivative=derivative)
            base.require(np.array_equal(row["coefficients"], c) and np.array_equal(row["qerr"], row["startup_q"][1:100]-case["target_q"])
                and np.array_equal(row["lerr"], row["startup_ltm"][1:100]-case["target_ltm"]), "RF coefficient/error/target mismatch")
            save_entry(label, row); check(); return row
        except Exception as error:
            partial = row if row is not None else getattr(error, "completed_margin_entry", None)
            if partial is None and getattr(error, "restoration_phase", None) == "evaluation": partial = getattr(error, "phase_partial_result", None)
            if isinstance(partial, dict):
                result["partial_evaluation"] = {"label": label, "completed_response_available": True}
                base.preserve_primary(error, "Partial RF actual persistence failed", lambda: save_entry(label+"-partial", partial))
            raise
    def account(receipt):
        for key in ("constructor", "solve", "certificate"):
            for field in ("requested", "completed", "failed"): cost[key][field] += receipt[key][field]
            if not receipt[key]["wall_known"]: cost[key].update(wall_s=None, wall_known=False)
            elif cost[key]["wall_s"] is not None: cost[key]["wall_s"] += receipt[key]["wall_s"]
        if receipt["iterations_known"]: cost["iterations"] += receipt["iterations"]
        elif receipt["solve"]["requested"]: cost["iterations_unknown_calls"] += 1
    try:
        current = observe(case["neutral_coefficients"].copy(), "neutral", True)
        result["reproduction"] = {actual: base.numerical_gate(current[actual], case[saved], tolerance) for actual, saved, tolerance in
            (("startup_q", "neutral_startup_q", 1e-10), ("startup_ltm", "neutral_startup_ltm", 1e-10),
             ("startup_q_jacobian", "neutral_q_jacobian", 1e-9), ("startup_ltm_jacobian", "neutral_ltm_jacobian", 1e-9))}
        persist("reproduction", None, result["reproduction"])
        if not all(g["pass"] for g in result["reproduction"].values()): raise metering.ReproductionError("RF neutral q/LTM/J reproduction failed")
        judged = strict_rf(current); result["trajectory"].append(judged); result["f64_success"] = judged["pass"]
        for iteration in range(4):
            if result["f64_success"]: break
            check(); problem = capacity_problem(current); cost["fresh_jacobians_used"] += 1; prefix = f"correction-{iteration:02d}"
            persist(prefix+"-problem", problem_arrays(problem), {"iteration": iteration, "neutral_trust_applied": False, "step_trust_applied": False})
            check(); limit = min(4., remaining())
            if limit <= 0: raise base.BudgetExceeded("original")
            try:
                solved = solve(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock); account(solved["receipt"])
            except Exception as error:
                receipt = getattr(error, "capacity_solve_receipt", None)
                if receipt is not None:
                    account(receipt); result["corrections"].append({"receipt": receipt, "iteration": iteration, "status": "execution_failure"})
                    raw = getattr(error, "partial_capacity_solution", None)
                    def save_partial(raw=raw, prefix=prefix, receipt=receipt):
                        arrays = {} if raw is None else {k: np.asarray(getattr(raw, attr), np.float64) for k, attr in (("correction", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                        persist(prefix+"-solve", arrays, receipt)
                    base.preserve_primary(error, "Partial RF solve persistence failed", save_partial)
                raise
            arrays = {k: solved[k] for k in ("correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
            if solved["residuals"] is not None: arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
            persist(prefix+"-solve", arrays, solved["receipt"])
            row = {"iteration": iteration, "receipt": solved["receipt"], "actual_evaluated": False}; result["corrections"].append(row)
            if not solved["valid"]: result["status"] = "uncertified_rf_correction"; break
            candidate = current["coefficients"]+solved["correction"].reshape(current["coefficients"].shape)
            coefficient_pass = base.coefficient_valid(candidate, interior=True)
            persist(prefix+"-coefficients", {"coefficients": candidate}, {"pass": coefficient_pass, "hard_domain_only": True})
            if not coefficient_pass: result["status"] = "strict_hard_coefficient_rejection"; break
            current = observe(candidate, prefix+"-actual", True); row["actual_evaluated"] = True
            judged = strict_rf(current); result["trajectory"].append(judged); row["judgement"] = judged
            persist(prefix+"-judgement", None, row); result["f64_success"] = judged["pass"]
        else: result["status"] = "four_rf_corrections_rejected"
        if result["f64_success"]:
            c32 = current["coefficients"].astype(np.float32); rounded = c32.astype(np.float64)
            coefficient_pass = base.coefficient_valid(rounded, interior=True)
            persist("f32-coefficients", {"coefficients32": c32, "coefficients": rounded}, {"pass": coefficient_pass, "hard_domain_only": True})
            if not coefficient_pass: result["status"] = "f32_hard_coefficient_rejection_before_evaluation"
            else:
                final = observe(rounded, "f32-actual", False); judged = strict_rf(final)
                result.update(f32_success=judged["pass"], f32_judgement=judged, status="supported" if judged["pass"] else "f32_rf_startup_rejection")
        result.update(run_valid=True, rf_capacity_supported=bool(result["f64_success"] and result["f32_success"]))
    except Exception as error:
        primary = error; result.update(run_valid=False, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
    if primary is None:
        try: persist("capacity-result", None, result)
        except Exception as error:
            result.update(run_valid=False, rf_capacity_supported=False, status="invalid", error=str(error), exception=type(error).__name__)
            error.capacity_result, error.abort_run = result, True; raise
    else:
        primary.capacity_result = result
        saved = base.preserve_primary(primary, "RF failure persistence failed", lambda: persist("capacity-result", None, result) or True)
        if saved is not True: primary.abort_run = True
        raise primary
    return result
