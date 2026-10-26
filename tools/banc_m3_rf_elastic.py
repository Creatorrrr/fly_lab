"""E285 RF elastic peak restoration with fixed actual line search and fresh J."""
from __future__ import annotations

# Primary errors retain partial work through secondary storage/timing failures.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import banc_m3_rf_capacity as capacity

base, margin, metering = capacity.base, capacity.margin, capacity.metering
ALPHAS = (1., .5, .25, .125)
CONTRACT = {k: copy.deepcopy(capacity.CONTRACT[k]) for k in capacity.CONTRACT if k not in
            ("max_corrections", "coefficient_domain", "sequence", "hessian", "armijo")}
CONTRACT.update(max_outer=6, step_radius=.5, slack_bounds=[0., .01], slack_penalty=100000.,
    hessian="identity including peak slack", coefficient_domain="hard[0,8]/triple interior;scaled local step<=.5;no neutral box",
    alphas=list(ALPHAS), peak_decrease_fraction=1e-4, slack_scope="intermediate linear q peak rows only;no actual criterion slack",
    sequence="neutral reproduce;max6 elastic QPs and fixed4alpha actual trials;fresh J only after accepted nonfinal point;one f32 replay if f64 passed",
    trial_acceptance="predicted peak<current;strict checks except q_peak;actual peak<=current-1e-4*(current-predicted)",
    fresh_reproduction_q_ltm=1e-10)
PROBLEM_KEYS = {*capacity.PROBLEM_KEYS, "step_radius", "slack_bounds", "slack_penalty"}


def elastic_problem(entry):
    margin.validate_entry(entry, derivative=True)
    c = entry["coefficients"].copy(); n = c.size
    base.require(3 <= n <= 783 and base.coefficient_valid(c, interior=True), "Original RF hard coefficient triples required")
    eye = sparse.eye(n, format="csc", dtype=np.float64); zero = sparse.csc_matrix((n, 1))
    triple = sparse.csc_matrix((np.ones(n), (np.repeat(np.arange(n//3), 3), np.arange(n))), shape=(n//3, n))
    endpoints = np.tile([0., 8.], (n, 1))
    step = np.column_stack((np.maximum(-c.ravel(), -.5), np.minimum(8-c.ravel(), .5)))
    bounds = np.vstack((step, [0., .01]))
    qe, le = (entry[k].ravel(order="C").copy() for k in ("qerr", "lerr"))
    qj, lj = (sparse.csc_matrix(entry[k].reshape(size, n)) for k, size in
              (("startup_q_jacobian", 693), ("startup_ltm_jacobian", 99)))
    peaks = np.array([.036, .025])*(1-1e-4)-1e-6
    radii = np.sqrt(np.array([CONTRACT["q_sse_limit"], CONTRACT["ltm_sse_limit"]]))-1e-6
    def padded(m): return sparse.hstack((m, sparse.csc_matrix((m.shape[0], 1))), format="csc")
    negative_slack = sparse.csc_matrix(-np.ones((693, 1)))
    slack_row = sparse.csc_matrix(([1.], ([0], [n])), shape=(1, n+1))
    blocks = [sparse.hstack((eye, zero), format="csc"), sparse.hstack((-eye, zero), format="csc"),
        padded(triple), padded(-triple), sparse.hstack((qj, negative_slack), format="csc"),
        sparse.hstack((-qj, negative_slack), format="csc"), padded(lj), padded(-lj), slack_row, -slack_row]
    rhs = [step[:, 1], -step[:, 0], 8*(1-1e-6)-c.sum(1), c.sum(1)-.125*(1+1e-6),
           peaks[0]-qe, peaks[0]+qe, peaks[1]-le, peaks[1]+le, np.array([.01]), np.array([0.])]
    rows = sum(len(v) for v in rhs)
    for error, jacobian, radius in zip((qe, le), (qj, lj), radii, strict=True):
        blocks.append(sparse.vstack((sparse.csc_matrix((1, n+1)), padded(-jacobian)), format="csc")); rhs.append(np.r_[radius, error])
    dims = (rows, 694, 100)
    return {"P": sparse.eye(n+1, format="csc", dtype=np.float64), "q": np.r_[np.zeros(n), 100000.],
        "A": sparse.vstack(blocks, format="csc"), "b": np.concatenate(rhs),
        "cones": [clarabel.NonnegativeConeT(rows), clarabel.SecondOrderConeT(694), clarabel.SecondOrderConeT(100)],
        "cone_dimensions": dims, "coefficients": c, "bounds": bounds, "coefficient_bounds": endpoints,
        "qerr": qe, "lerr": le, "q_jacobian": qj, "ltm_jacobian": lj, "peaks": peaks, "radii": radii,
        "step_radius": .5, "slack_bounds": np.array([0., .01]), "slack_penalty": 100000.}


def validate_problem(p):
    base.require(set(p) == PROBLEM_KEYS, "Exact elastic RF metadata;no neutral trust")
    c = p["coefficients"]
    base.require(isinstance(c, np.ndarray) and c.dtype == np.float64 and 1 <= len(c) <= 261
        and base.coefficient_valid(c, interior=True), "Original bounded hard coefficients required")
    n, dims = c.size, (2*c.size+2*len(c)+1586, 694, 100)
    for key, shape in (("P", (n+1, n+1)), ("A", (sum(dims), n+1)), ("q_jacobian", (693, n)), ("ltm_jacobian", (99, n))):
        v = p[key]
        base.require(sparse.isspmatrix_csc(v) and v.dtype == np.float64 and v.shape == shape and v.has_canonical_format
            and v.has_sorted_indices and np.isfinite(v.data).all(), "Original canonical finite elastic CSC required: "+key)
    for key, shape in (("q", (n+1,)), ("b", (sum(dims),)), ("bounds", (n+1, 2)), ("coefficient_bounds", (n, 2)),
                       ("qerr", (693,)), ("lerr", (99,)), ("peaks", (2,)), ("radii", (2,)), ("slack_bounds", (2,))):
        v = p[key]
        base.require(isinstance(v, np.ndarray) and v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all(), "Original complete elastic metadata required: "+key)
    base.require(p["step_radius"] == .5 and p["slack_penalty"] == 100000. and tuple(p["cone_dimensions"]) == dims
        and len(p["cones"]) == 3 and isinstance(p["cones"][0], clarabel.NonnegativeConeT)
        and all(isinstance(v, clarabel.SecondOrderConeT) for v in p["cones"][1:]) and tuple(v.dim for v in p["cones"]) == dims,
        "Exact elastic bounds/penalty/cone dimensions required")
    entry = {"coefficients": c, "qerr": p["qerr"].reshape(99, 7), "lerr": p["lerr"].reshape(99, 1),
        "startup_q": np.zeros((101, 7)), "startup_ltm": np.zeros((101, 1)),
        "startup_q_jacobian": p["q_jacobian"].toarray().reshape(99, 7, n), "startup_ltm_jacobian": p["ltm_jacobian"].toarray().reshape(99, 1, n)}
    exact = elastic_problem(entry)
    for key in ("P", "A"):
        delta = p[key]-exact[key]; delta.eliminate_zeros(); base.require(delta.nnz == 0, "Original elastic equation changed: "+key)
    for key in ("q", "b", "bounds", "coefficient_bounds", "peaks", "radii", "slack_bounds"):
        base.require(np.array_equal(p[key], exact[key]), "Original elastic RHS/metadata changed: "+key)
    return dims


def elastic_residuals(p, solution):
    x = np.asarray(solution)
    base.require(x.dtype == np.float64 and x.shape == p["q"].shape and np.isfinite(x).all(), "Finite complete neural correction plus elastic slack required")
    d, slack = x[:-1], x[-1]; c = p["coefficients"].ravel()+d; sums = c.reshape(-1, 3).sum(1)
    qp, lp = p["qerr"]+p["q_jacobian"]@d, p["lerr"]+p["ltm_jacobian"]@d
    residuals = {"coefficient": np.r_[c, 8-c, sums-.125*(1+1e-6), 8*(1-1e-6)-sums],
        "bounds": np.r_[x-p["bounds"][:, 0], p["bounds"][:, 1]-x],
        "peak": np.r_[p["peaks"][0]+slack-np.abs(qp), p["peaks"][1]-np.abs(lp)],
        "soc": p["radii"]-np.array([np.linalg.norm(qp), np.linalg.norm(lp)])}
    return {**residuals, "minimum": float(min(np.min(v) for v in residuals.values()))}


def elastic_certificate(p, solution, solver_s, solver_y, status):
    dims = validate_problem(p); x, s, y = (np.asarray(v) for v in (solution, solver_s, solver_y))
    checks = {k: False for k in ("input_valid", "status", "residual", "primal_cone", "dual_cone", "finite_math", "stationarity", "gap", "complementarity")}
    checks["status"] = str(status) in ("Solved", "AlmostSolved")
    receipt = {"status": str(status), "valid": False, "checks": checks}
    checks["input_valid"] = all(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all()
        for v, shape in ((x, p["q"].shape), (s, p["b"].shape), (y, p["b"].shape)))
    if not checks["input_valid"]: return {"valid": False, "receipt": receipt, "primal_slack": None, "residuals": None}
    with np.errstate(over="ignore", invalid="ignore"):
        exact, residuals = p["b"]-p["A"]@x, elastic_residuals(p, x)
        cones = [metering.original_certificate._cone_slacks(v, dims) for v in (exact, s, y)]
        w = p["q"]+p["A"].T@y; stationarity = np.max(abs(x+w))
        primal, dual = .5*np.dot(x, x)+np.dot(p["q"], x), -.5*np.dot(w, w)-np.dot(p["b"], y)
        gap, scale = primal-dual, 1+max(abs(primal), abs(dual)); comp, affine = np.dot(exact, y), np.max(abs(exact-s))
    checks.update(finite_math=bool(all(np.isfinite(v) for v in (residuals["minimum"], stationarity, primal, dual, gap, scale, comp, affine,
        *(v for cone in cones for v in cone)))), residual=bool(residuals["minimum"] >= -1e-8),
        primal_cone=bool(all(v >= -1e-8 for v in cones[0])), dual_cone=bool(all(v >= 0 for v in cones[2])),
        stationarity=bool(stationarity <= 1e-8), gap=bool(-1e-10*scale <= gap <= 1e-8*scale), complementarity=bool(abs(comp) <= 1e-8*scale))
    number = metering.original_certificate._number
    receipt.update(valid=bool(all(checks.values())), minimum_residual=number(residuals["minimum"]), raw_affine_infinity_norm=number(affine),
        raw_cone_slacks=[number(v) for v in cones[1]], primal_cone_slacks=[number(v) for v in cones[0]], dual_cone_slacks=[number(v) for v in cones[2]],
        stationarity_infinity_norm=number(stationarity), primal_objective=number(primal), dual_lower_bound=number(dual), gap=number(gap),
        objective_scale=number(scale), complementarity=number(comp))
    return {"valid": receipt["valid"], "receipt": receipt, "primal_slack": exact, "residuals": residuals}


def solve_elastic(p, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    validate_problem(p)
    base.require(clarabel.__version__ == "0.11.1" and np.isfinite(time_limit) and 0 < time_limit <= 4, "Frozen Clarabel and bounded elastic time required")
    settings = clarabel.DefaultSettings()
    applied = {"verbose": False, "max_iter": 100, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8, "max_threads": 1, "time_limit": float(time_limit)}
    for name, value in applied.items(): setattr(settings, name, value); base.require(getattr(settings, name) == value, "Frozen elastic setting changed")
    receipt = {"settings": applied, "iterations": None, "iterations_known": False, "status": "requested", **{k: metering.phase() for k in ("constructor", "solve", "certificate")}}
    raw = None
    try:
        solver = metering.metered(receipt, "constructor", lambda: solver_factory(p["P"], p["q"], p["A"], p["b"], p["cones"], settings), wall_clock)
        raw = metering.metered(receipt, "solve", solver.solve, wall_clock)
        receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        x, s, y = (np.asarray(v, np.float64) for v in (raw.x, raw.s, raw.z))
        cert = metering.metered(receipt, "certificate", lambda: elastic_certificate(p, x, s, y, raw.status), wall_clock)
        receipt.update(solution_valid=cert["valid"], certificate_receipt=cert["receipt"], solver_info={k: metering.previous.finite_scalar(getattr(raw, k)) for k in
            ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual") if hasattr(raw, k)})
        return {**cert, "solution": x, "correction": x[:-1], "elastic_slack": float(x[-1]), "solver_s": s, "solver_y": y, "receipt": receipt}
    except Exception as error:
        if raw is None and getattr(error, "restoration_phase", None) == "solve":
            raw = getattr(error, "phase_partial_result", None)
            if raw is not None: receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        error.elastic_solve_receipt, error.partial_elastic_solution = copy.deepcopy(receipt), raw; raise


def trial_judgement(entry, current_peak, predicted_peak):
    judged = capacity.strict_rf(entry)
    required = current_peak-1e-4*(current_peak-predicted_peak)
    checks = {k: v for k, v in judged["checks"].items() if k != "q_peak"}
    checks.update(predicted_decrease=bool(np.isfinite(predicted_peak) and predicted_peak < current_peak),
                  actual_peak_decrease=bool(judged["metrics"]["q"]["peak"] <= required))
    return {"accepted": bool(all(checks.values())), "checks": checks, "strict": judged,
            "current_peak": current_peak, "predicted_peak": predicted_peak, "required_peak": required}


def problem_arrays(p):
    return {**capacity.problem_arrays(p), "step_radius": np.array(p["step_radius"]), "slack_penalty": np.array(p["slack_penalty"])}


def run_elastic(case, evaluate, *, persist, check=lambda: None, remaining=lambda: 4.,
                solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter, solve=solve_elastic):
    margin.validate_case(case)
    cost = {k: metering.phase() for k in ("evaluation", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_jacobians_used=0)
    result = {"status": "requested", "run_valid": False, "f64_success": False, "f32_success": False,
        "rf_capacity_supported": False, "initialization_supported": False, "global_norm_supported": False,
        "reproduction": {}, "trajectory": [], "outers": [], "cost": cost}; primary = None
    def save_entry(label, row):
        persist(label, {k: v for k, v in row.items() if isinstance(v, np.ndarray)}, {k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
    def observe(c, label, derivative):
        check(); row = None
        try:
            row = metering.metered(cost, "evaluation", lambda: evaluate(c, derivative=derivative, label=label), wall_clock)
            margin.validate_entry(row, derivative=derivative)
            base.require(np.array_equal(row["coefficients"], c) and np.array_equal(row["qerr"], row["startup_q"][1:100]-case["target_q"])
                and np.array_equal(row["lerr"], row["startup_ltm"][1:100]-case["target_ltm"]), "Elastic actual coefficient/error/target mismatch")
            save_entry(label, row); check(); return row
        except Exception as error:
            partial = row if row is not None else getattr(error, "completed_margin_entry", None)
            if partial is None and getattr(error, "restoration_phase", None) == "evaluation": partial = getattr(error, "phase_partial_result", None)
            if isinstance(partial, dict):
                result["partial_evaluation"] = {"label": label, "completed_response_available": True}
                base.preserve_primary(error, "Partial elastic actual persistence failed", lambda: save_entry(label+"-partial", partial))
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
        if not all(g["pass"] for g in result["reproduction"].values()): raise metering.ReproductionError("Elastic neutral q/LTM/J reproduction failed")
        judged = capacity.strict_rf(current); result["trajectory"].append(judged); result["f64_success"] = judged["pass"]
        for outer in range(6):
            if result["f64_success"]: break
            prefix = f"outer-{outer:02d}"
            if outer:
                fresh = observe(current["coefficients"].copy(), prefix+"-fresh", True)
                gates = {k: base.numerical_gate(fresh[k], current[k], 1e-10) for k in ("startup_q", "startup_ltm")}
                persist(prefix+"-fresh-reproduction", None, gates)
                if not all(g["pass"] for g in gates.values()): raise metering.ReproductionError("Accepted RF point fresh-J replay failed")
                current = fresh
            check(); problem = elastic_problem(current); cost["fresh_jacobians_used"] += 1
            record = {"outer": outer, "trials": [], "accepted": False}; result["outers"].append(record)
            persist(prefix+"-problem", problem_arrays(problem), {"outer": outer, "slack_scope": "linear q peaks only"})
            check(); limit = min(4., remaining())
            if limit <= 0: raise base.BudgetExceeded("original")
            try: solved = solve(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock); account(solved["receipt"])
            except Exception as error:
                receipt = getattr(error, "elastic_solve_receipt", None)
                if receipt is not None:
                    account(receipt); record.update(receipt=receipt, status="execution_failure")
                    raw = getattr(error, "partial_elastic_solution", None)
                    def save_partial(raw=raw, prefix=prefix, receipt=receipt):
                        arrays = {} if raw is None else {k: np.asarray(getattr(raw, attr), np.float64) for k, attr in (("solution", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                        persist(prefix+"-solve", arrays, receipt)
                    base.preserve_primary(error, "Partial elastic solve persistence failed", save_partial)
                raise
            record.update(receipt=solved["receipt"], elastic_slack=solved["elastic_slack"])
            arrays = {k: solved[k] for k in ("solution", "correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
            if solved["residuals"] is not None: arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
            persist(prefix+"-solve", arrays, solved["receipt"])
            if not solved["valid"]: result["status"] = "uncertified_elastic_direction"; break
            peak = float(np.max(abs(current["qerr"])))
            for index, alpha in enumerate(ALPHAS):
                label = prefix+f"-alpha-{index}"
                d = alpha*solved["correction"]; candidate = current["coefficients"]+d.reshape(current["coefficients"].shape)
                predicted = float(np.max(abs(problem["qerr"]+problem["q_jacobian"]@d)))
                gates = {"coefficient": base.coefficient_valid(candidate, interior=True), "step": bool(np.isfinite(d).all() and np.max(abs(d)) <= .5),
                         "predicted_decrease": bool(np.isfinite(predicted) and predicted < peak)}
                trial = {"alpha": alpha, "label": label, "gates": gates, "predicted_peak": predicted, "actual_evaluated": False, "accepted": False}
                record["trials"].append(trial)
                persist(label+"-coefficients", {"coefficients": candidate, "scaled_correction": d}, trial)
                if all(gates.values()):
                    row = observe(candidate, label+"-actual", False); trial["actual_evaluated"] = True
                    trial["judgement"] = trial_judgement(row, peak, predicted); trial["accepted"] = trial["judgement"]["accepted"]
                    if trial["accepted"]:
                        current = row; record.update(accepted=True, accepted_alpha=alpha)
                        judged = trial["judgement"]["strict"]; result["trajectory"].append(judged); result["f64_success"] = judged["pass"]
                persist(label+"-judgement", None, trial)
                if trial["accepted"]: break
            persist(prefix+"-result", None, record)
            if not record["accepted"]: result["status"] = "fixed_alpha_line_search_rejected"; break
        else: result["status"] = "six_outer_exhausted"
        if result["f64_success"]:
            c32 = current["coefficients"].astype(np.float32); rounded = c32.astype(np.float64)
            coefficient_pass = base.coefficient_valid(rounded, interior=True)
            persist("f32-coefficients", {"coefficients32": c32, "coefficients": rounded}, {"pass": coefficient_pass})
            if not coefficient_pass: result["status"] = "f32_hard_coefficient_rejection_before_evaluation"
            else:
                final = observe(rounded, "f32-actual", False); judged = capacity.strict_rf(final)
                result.update(f32_success=judged["pass"], f32_judgement=judged, status="supported" if judged["pass"] else "f32_rf_startup_rejection")
        result.update(run_valid=True, rf_capacity_supported=bool(result["f64_success"] and result["f32_success"]))
    except Exception as error:
        primary = error; result.update(run_valid=False, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
    if primary is None:
        try: persist("elastic-result", None, result)
        except Exception as error:
            result.update(run_valid=False, rf_capacity_supported=False, status="invalid", error=str(error), exception=type(error).__name__)
            error.elastic_result, error.abort_run = result, True; raise
    else:
        primary.elastic_result = result
        saved = base.preserve_primary(primary, "Elastic failure persistence failed", lambda: persist("elastic-result", None, result) or True)
        if saved is not True: primary.abort_run = True
        raise primary
    return result
