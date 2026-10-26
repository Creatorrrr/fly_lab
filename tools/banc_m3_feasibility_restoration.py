"""E280 bounded minimum-norm startup restoration using fresh conditional J.

No import-time data reads, solver execution, optimization or CUDA work.
"""
from __future__ import annotations

# Preserve primary failures when timing or persistence also fails.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import banc_m3_conic_certificate as original_certificate
from tools import fit_c_banc_m3_conic_source as conic
from tools import fit_c_banc_m3_feasible_source as previous

base = conic.base
CONTRACT = {"max_corrections": 2, "original_radius": .05, "linear_absolute_margin": 1e-6,
    "selector_margin": 1e-4, "primal_tolerance": 1e-8, "dual_tolerance": 0.,
    "stationarity_tolerance": 1e-8, "gap_lower_relative_tolerance": 1e-10,
    "gap_upper_relative_tolerance": 1e-8, "complementarity_relative_tolerance": 1e-8,
    "statuses": ["Solved", "AlmostSolved"], "hessian": "identity", "clip": False,
    "clarabel_version": clarabel.__version__, "max_iter": 100, "time_limit_s": 2., "max_threads": 1,
    "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8,
    "armijo": 1e-4, "reproduction_q_ltm": 1e-10, "reproduction_objective": 1e-10,
    "reproduction_before_gradient": 1e-9, "reproduction_trial_gradient": "not_saved_not_compared"}


def restoration_problem(current, before_coefficients, fixed_sse, radius, correction_radius):
    """Infeasible startup trials are allowed; coefficient/trust axes stay fixed."""
    c, before = np.asarray(current["coefficients"]), np.asarray(before_coefficients)
    base.require(c.dtype == before.dtype == np.float64 and c.shape == before.shape
        and c.ndim == 2 and c.shape[1] == 3 and 1 <= len(c) <= 1660
        and base.coefficient_valid(c, interior=True) and base.coefficient_valid(before, interior=True),
        "Finite original interior coefficient triples required")
    n = c.size
    errors = [np.asarray(current[k]) for k in ("qerr", "lerr")]
    jacobians = [np.asarray(current[k]) for k in ("startup_q_jacobian", "startup_ltm_jacobian")]
    for e, j, shape in zip(errors, jacobians, ((99, 7), (99, 1)), strict=True):
        base.require(e.dtype == j.dtype == np.float64 and e.shape == shape and j.shape == (*shape, n)
            and np.isfinite(e).all() and np.isfinite(j).all(), "Finite fresh complete startup errors/J required")
    fixed = np.asarray(fixed_sse, dtype=np.float64)
    base.require(fixed.shape == (2,) and np.isfinite(fixed).all() and np.all(fixed >= 0)
        and radius == .05 and np.isfinite(correction_radius) and 0 < correction_radius <= radius,
        "Exact original trust radius and finite fixed SSE/correction bound required")
    available = np.array([99*42*(.018*(1-1e-4))**2, 99*6*(.009*(1-1e-4))**2])-fixed
    base.require(np.all(available >= 0), "Negative remaining global norm budget")
    original_radii = np.sqrt(available)
    radii = original_radii-1e-6
    peaks = np.array([.04, .025])*(1-1e-4)-1e-6
    base.require(np.all(radii >= 0), "Absolute norm margin exceeds remaining budget")
    c, before = np.array(c, order="C", copy=True), np.array(before, order="C", copy=True)
    errors = [np.array(e, order="C", copy=True) for e in errors]
    jacobians = [np.array(j, order="C", copy=True) for j in jacobians]
    lo = np.maximum.reduce((-c.ravel(), before.ravel()-radius-c.ravel(), np.full(n, -correction_radius)))
    hi = np.minimum.reduce((8-c.ravel(), before.ravel()+radius-c.ravel(), np.full(n, correction_radius)))
    base.require(np.all(lo <= hi), "Empty original coefficient/trust/correction intersection")
    triple = base.triple_constraint(len(c))
    identity = sparse.eye(n, format="csc", dtype=np.float64)
    blocks = [identity, -identity, sparse.csc_matrix(triple.A), sparse.csc_matrix(-triple.A)]
    rhs = [hi, -lo, np.asarray(triple.ub)-c.sum(1), c.sum(1)-np.asarray(triple.lb)]
    for e, j, peak in zip(errors, jacobians, peaks, strict=True):
        j = j.reshape(e.size, n)
        blocks.extend((sparse.csc_matrix(j), sparse.csc_matrix(-j)))
        rhs.extend((peak-e.ravel(), peak+e.ravel()))
    linear_rows = sum(len(v) for v in rhs)
    for e, j, norm_radius in zip(errors, jacobians, radii, strict=True):
        blocks.append(sparse.vstack((sparse.csc_matrix((1, n)), sparse.csc_matrix(-j.reshape(e.size, n))), format="csc"))
        rhs.append(np.r_[norm_radius, e.ravel()])
    return {"P": identity, "q": np.zeros(n), "A": sparse.vstack(blocks, format="csc"), "b": np.concatenate(rhs),
        "cones": [clarabel.NonnegativeConeT(linear_rows), clarabel.SecondOrderConeT(694), clarabel.SecondOrderConeT(100)],
        "nonnegative_rows": linear_rows, "coefficients": c, "before_coefficients": before,
        "bounds": np.column_stack((lo, hi)), "qerr": errors[0], "lerr": errors[1],
        "startup_q_jacobian": jacobians[0], "startup_ltm_jacobian": jacobians[1],
        "fixed_sse": fixed.copy(), "available_sse": available, "original_radii": original_radii,
        "peaks": peaks, "radii": radii, "radius": radius, "correction_radius": float(correction_radius)}


def restoration_residuals(problem, correction):
    z = np.asarray(correction)
    base.require(z.dtype == np.float64 and z.shape == problem["q"].shape and np.isfinite(z).all(),
                 "Finite complete float64 correction required")
    c = problem["coefficients"].ravel()+z
    sums = c.reshape(-1, 3).sum(1)
    coefficient = np.r_[c, 8-c, sums-.125*(1+1e-6), 8*(1-1e-6)-sums,
        z-problem["bounds"][:, 0], problem["bounds"][:, 1]-z]
    trust = problem["radius"]-np.abs(c-problem["before_coefficients"].ravel())
    change = problem["correction_radius"]-np.abs(z)
    linear, soc = [], []
    for ekey, jkey, peak, radius in zip(("qerr", "lerr"), ("startup_q_jacobian", "startup_ltm_jacobian"),
                                       problem["peaks"], problem["radii"], strict=True):
        predicted = problem[ekey].ravel()+problem[jkey].reshape(problem[ekey].size, z.size)@z
        linear.extend((peak-predicted, peak+predicted))
        soc.append(radius-np.linalg.norm(predicted))
    arrays = {"coefficient": coefficient, "trust": trust, "correction": change,
              "linear": np.concatenate(linear), "soc": np.asarray(soc)}
    return {**arrays, "minimum": float(min(np.min(v) for v in arrays.values()))}


def _validate_problem(problem):
    n = problem["q"].size
    base.require(problem["q"].dtype == np.float64 and problem["q"].shape == (n,)
        and 3 <= n <= 4980 and n % 3 == 0 and np.array_equal(problem["q"], np.zeros(n)), "Original zero linear objective required")
    dims = (2*n+2*(n//3)+1584, 694, 100)
    for k, shape in (("P", (n, n)), ("A", (sum(dims), n))):
        v = problem[k]
        base.require(sparse.isspmatrix_csc(v) and v.dtype == np.float64 and v.shape == shape
            and v.has_canonical_format and v.has_sorted_indices and np.isfinite(v.data).all(), "Canonical finite CSC required: "+k)
    diff = problem["P"]-sparse.eye(n, format="csc")
    diff.eliminate_zeros()
    base.require(diff.nnz == 0, "Restoration Hessian must be exact identity")
    base.require(problem["b"].dtype == np.float64 and problem["b"].shape == (sum(dims),)
        and np.isfinite(problem["b"]).all() and len(problem["cones"]) == 3
        and isinstance(problem["cones"][0], clarabel.NonnegativeConeT)
        and all(isinstance(c, clarabel.SecondOrderConeT) for c in problem["cones"][1:])
        and tuple(c.dim for c in problem["cones"]) == dims and problem["nonnegative_rows"] == dims[0],
        "Exact original RHS/cones required")
    for k, shape in {"coefficients": (n//3, 3), "before_coefficients": (n//3, 3), "bounds": (n, 2),
        "qerr": (99, 7), "lerr": (99, 1), "startup_q_jacobian": (99, 7, n),
        "startup_ltm_jacobian": (99, 1, n), "peaks": (2,), "radii": (2,)}.items():
        v = problem[k]
        base.require(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all(), "Finite original metadata required: "+k)
    base.require(problem["radius"] == .05 and np.isfinite(problem["correction_radius"])
        and 0 < problem["correction_radius"] <= .05, "Original restoration radii required")
    return dims


def restoration_certificate(problem, correction, solver_s, solver_y, status):
    dims = _validate_problem(problem)
    z, s, y = (np.asarray(v) for v in (correction, solver_s, solver_y))
    checks = {k: False for k in ("input_valid", "status", "residual", "primal_cone", "dual_cone",
                                "finite_math", "stationarity", "gap", "complementarity")}
    checks["status"] = str(status) in CONTRACT["statuses"]
    receipt = {"status": str(status), "valid": False, "checks": checks, "minimum_residual": None,
        "raw_affine_infinity_norm": None, "raw_cone_slacks": None, "primal_cone_slacks": None,
        "dual_cone_slacks": None, "stationarity_infinity_norm": None, "primal_objective": None,
        "dual_lower_bound": None, "gap": None, "objective_scale": None, "complementarity": None}
    finite = all(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all()
        for v, shape in ((z, problem["q"].shape), (s, problem["b"].shape), (y, problem["b"].shape)))
    checks["input_valid"] = bool(finite)
    if not finite:
        return {"valid": False, "receipt": receipt, "primal_slack": None, "residuals": None}
    with np.errstate(over="ignore", invalid="ignore"):
        exact = problem["b"]-problem["A"]@z
        res = restoration_residuals(problem, z)
        cones = [original_certificate._cone_slacks(v, dims) for v in (exact, s, y)]
        w = problem["A"].T@y
        stationarity = np.max(np.abs(z+w))
        primal, dual = .5*np.dot(z, z), -.5*np.dot(w, w)-np.dot(problem["b"], y)
        gap, scale = primal-dual, 1+max(abs(primal), abs(dual))
        complementary, affine = np.dot(exact, y), np.max(np.abs(exact-s))
    checks.update(finite_math=bool(all(np.isfinite(v) for v in
        (res["minimum"], stationarity, primal, dual, gap, scale, complementary, affine, *(v for cone in cones for v in cone)))),
        residual=bool(np.isfinite(res["minimum"]) and res["minimum"] >= -1e-8),
        primal_cone=bool(all(np.isfinite(v) and v >= -1e-8 for v in cones[0])),
        dual_cone=bool(all(np.isfinite(v) and v >= 0 for v in cones[2])),
        stationarity=bool(np.isfinite(stationarity) and stationarity <= 1e-8),
        gap=bool(np.isfinite(gap) and np.isfinite(scale) and -1e-10*scale <= gap <= 1e-8*scale),
        complementarity=bool(np.isfinite(complementary) and np.isfinite(scale) and abs(complementary) <= 1e-8*scale))
    number = original_certificate._number
    valid = bool(all(checks.values()))
    receipt.update(valid=valid, minimum_residual=number(res["minimum"]), raw_affine_infinity_norm=number(affine),
        raw_cone_slacks=[number(v) for v in cones[1]], primal_cone_slacks=[number(v) for v in cones[0]],
        dual_cone_slacks=[number(v) for v in cones[2]], stationarity_infinity_norm=number(stationarity),
        primal_objective=number(primal), dual_lower_bound=number(dual), gap=number(gap),
        objective_scale=number(scale), complementarity=number(complementary))
    return {"valid": valid, "receipt": receipt, "primal_slack": exact, "residuals": res}


def phase():
    return {"requested": 0, "completed": 0, "failed": 0, "wall_s": 0., "wall_known": True}


def metered(receipt, key, callback, clock):
    row = receipt[key]
    row["requested"] += 1
    began, primary, result = None, None, None
    try:
        began = clock()
        result = callback()
        row["completed"] += 1
    except Exception as error:
        primary = error
        row["failed"] += 1
    finally:
        try:
            base.require(began is not None, "Phase start timing unavailable")
            elapsed = clock()-began
            if row["wall_s"] is not None:
                row["wall_s"] += elapsed
        except Exception as error:
            row.update(wall_s=None, wall_known=False)
            if primary is None:
                primary = error
                row["failed"] += 1
            else:
                primary.add_note("Restoration timing failed: "+repr(error))
    if primary is not None:
        primary.restoration_phase = key
        primary.phase_partial_result = result
        raise primary
    return result


def solve_restoration(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    _validate_problem(problem)
    settings, applied = conic.solver_settings(time_limit)
    receipt = {"solver_calls_kind": "actual Clarabel restoration SOCP calls", "settings": applied,
        "constructor": phase(), "solve": phase(), "certificate": phase(),
        "iterations": None, "iterations_known": False, "status": "requested"}
    raw = None
    try:
        solver = metered(receipt, "constructor", lambda: solver_factory(problem["P"], problem["q"], problem["A"],
            problem["b"], problem["cones"], settings), wall_clock)
        raw = metered(receipt, "solve", solver.solve, wall_clock)
        receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        z, s, y = (np.asarray(v, dtype=np.float64) for v in (raw.x, raw.s, raw.z))
        certificate = metered(receipt, "certificate", lambda: restoration_certificate(problem, z, s, y, raw.status), wall_clock)
        receipt.update(solution_valid=certificate["valid"], certificate_receipt=certificate["receipt"],
            solver_info={k: previous.finite_scalar(getattr(raw, k)) for k in
                         ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual") if hasattr(raw, k)})
        return {**certificate, "correction": z, "solver_s": s, "solver_y": y, "receipt": receipt}
    except Exception as error:
        if raw is None and getattr(error, "restoration_phase", None) == "solve":
            raw = getattr(error, "phase_partial_result", None)
            if raw is not None:
                receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        error.restoration_receipt = copy.deepcopy(receipt)
        error.partial_restoration_solution = raw
        raise


CASE_KEYS = {"before_coefficients", "trial_coefficients", "comparator_coefficients", "before_startup_q",
    "trial_startup_q", "before_startup_ltm", "trial_startup_ltm", "before_objective", "trial_objective",
    "before_gradient", "fixed_sse", "baseline_mse", "radius", "alpha", "correction_radius", "reference_comparator_objective"}


def validate_case(case):
    base.require(set(case) == CASE_KEYS and all(isinstance(v, np.ndarray) and v.dtype == np.float64
        and np.isfinite(v).all() for v in case.values()), "Exact finite float64 case NPZ keys required")
    c = case["before_coefficients"]
    base.require(c.ndim == 2 and c.shape[1] == 3 and 1 <= len(c) <= 1660, "Case coefficient triples required")
    for name in ("before", "trial", "comparator"):
        v = case[name+"_coefficients"]
        base.require(v.shape == c.shape and base.coefficient_valid(v, interior=True), "Case interior coefficients required")
    for name in ("before", "trial"):
        base.require(case[name+"_startup_q"].shape == (101, 7)
            and case[name+"_startup_ltm"].shape == (101, 1), "Exact full startup case response required")
    base.require(case["before_gradient"].shape == (c.size,) and case["fixed_sse"].shape == (2,)
        and np.all(case["fixed_sse"] >= 0), "Case gradient/fixed SSE axes required")
    for key in ("before_objective", "trial_objective", "baseline_mse", "radius", "alpha", "correction_radius",
                "reference_comparator_objective"):
        base.require(case[key].shape == (), "Scalar case value required: "+key)
    base.require(float(case["radius"]) == .05 and float(case["baseline_mse"]) > 0
        and float(case["alpha"]) > 0 and float(case["alpha"]) <= 1
        and np.array_equal(c, base.neutral(len(c)))
        and float(case["correction_radius"]) == np.max(np.abs(case["trial_coefficients"]-c))
        and 0 < float(case["correction_radius"]) <= .05
        and np.max(np.abs(case["comparator_coefficients"]-c)) <= .05, "Exact before/trial trust/correction semantics required")


class ReproductionError(RuntimeError):
    pass


def restore_case(case, evaluate, *, persist, check=lambda: None, remaining=lambda: 2.,
                 solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """Before/trial reproduction, then at most two fresh-J corrections.

    evaluate(flat_c) uses the original nonlinear evaluator. persist(name,
    arrays_or_None, receipt) is mandatory and its failures cannot be success.
    """
    validate_case(case)
    cost = {k: phase() for k in ("evaluation", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_trial_jacobians_used=0)
    result = {"status": "requested", "run_valid": False, "restored": False, "comparator_improved": False,
              "reproduction": {}, "corrections": [], "cost": cost}
    primary = None
    def observe(c, label):
        check()
        try:
            entry = metered(cost, "evaluation", lambda: evaluate(c.ravel()), wall_clock)
        except Exception as error:
            entry = getattr(error, "completed_restoration_entry", None)
            if entry is None:
                entry = getattr(error, "phase_partial_result", None)
            if entry is not None:
                # A deadline after cache completion does not erase the completed raw.
                if getattr(error, "completed_restoration_entry", None) is not None:
                    cost["evaluation"]["completed"] += 1
                base.preserve_primary(error, "Completed restoration evaluation persistence failed",
                    lambda: persist(label, {k: v for k, v in entry.items() if isinstance(v, np.ndarray)},
                        {k: v for k, v in entry.items() if not isinstance(v, np.ndarray)}))
            raise
        persist(label, {k: v for k, v in entry.items() if isinstance(v, np.ndarray)},
                {k: v for k, v in entry.items() if not isinstance(v, np.ndarray)})
        check()
        return entry
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
        before = observe(case["before_coefficients"], "before")
        trial = observe(case["trial_coefficients"], "trial")
        for name, entry in (("before", before), ("trial", trial)):
            gates = {}
            for key, field, tolerance in (("startup_q", "startup_q", 1e-10), ("startup_ltm", "startup_ltm", 1e-10),
                                         ("objective", "objective", 1e-10)):
                gates[key] = base.numerical_gate(np.asarray(entry[field]), case[name+"_"+key], tolerance)
            if name == "before":
                gates["gradient"] = base.numerical_gate(entry["gradient"], case["before_gradient"], 1e-9)
            result["reproduction"][name] = gates
        reproduced = all(g["pass"] for rows in result["reproduction"].values() for g in rows.values())
        persist("reproduction", None, result["reproduction"])
        if not reproduced:
            raise ReproductionError("Before/trial reproduction failed; no correction solve")
        base.require(previous.eligible(before) and previous.entry_finite(before), "Before is not strict feasible")
        base.require(not previous.eligible(trial), "Frozen rejected trial unexpectedly eligible")
        current = trial
        for iteration in range(2):
            check()
            problem = restoration_problem(current, case["before_coefficients"], case["fixed_sse"],
                float(case["radius"]), float(case["correction_radius"]))
            cost["fresh_trial_jacobians_used"] += 1
            prefix = f"correction-{iteration:02d}"
            arrays = {k: v for k, v in problem.items() if isinstance(v, (np.ndarray, float))}
            persist(prefix+"-problem", arrays, {"iteration": iteration, "jacobian_evaluation": current["evaluation"]})
            check()
            limit = min(2., remaining())
            base.require(limit > 0, "No scientific time remaining")
            try:
                solved = solve_restoration(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock)
                account(solved["receipt"])
            except Exception as error:
                receipt = getattr(error, "restoration_receipt", None)
                if receipt is not None:
                    account(receipt)
                    result["corrections"].append({"iteration": iteration, "receipt": receipt, "status": "execution_failure"})
                    raw = getattr(error, "partial_restoration_solution", None)
                    arrays = {} if raw is None else {key: np.asarray(getattr(raw, attr), dtype=np.float64)
                        for key, attr in (("correction", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                    base.preserve_primary(error, "Partial restoration solve persistence failed",
                        lambda prefix=prefix, arrays=arrays, receipt=receipt: persist(prefix+"-solve", arrays, receipt))
                raise
            arrays = {k: solved[k] for k in ("correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
            if solved["residuals"] is not None:
                arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
            persist(prefix+"-solve", arrays, solved["receipt"])
            row = {"iteration": iteration, "receipt": solved["receipt"], "actual_evaluated": False,
                   "true_startup_verified": False}
            result["corrections"].append(row)
            if not solved["valid"]:
                result.update(status="uncertified_correction", run_valid=True)
                break
            c = current["coefficients"]+solved["correction"].reshape(current["coefficients"].shape)
            persist(prefix+"-coefficients", {"coefficients": c}, {"modified_after_solve": False})
            if not base.coefficient_valid(c, interior=True):
                row["status"] = "strict_coefficient_rejection"
                result.update(status=row["status"], run_valid=True)
                break
            if not (np.all(c >= case["before_coefficients"]-float(case["radius"]))
                and np.all(c <= case["before_coefficients"]+float(case["radius"]))
                and np.all(np.abs(solved["correction"]) <= float(case["correction_radius"]))):
                row["status"] = "strict_trust_or_correction_rejection"
                result.update(status=row["status"], run_valid=True)
                break
            current = observe(c, prefix+"-actual")
            row.update(actual_evaluated=True, true_startup_verified=True)
            slope, armijo = previous.proposal_numbers(row, before, c, current)
            accepted = previous.eligible(current) and slope < 0 and current["objective"] <= armijo
            row.update(strict_accepted=bool(accepted), status="restored" if accepted else "true_constraint_or_armijo_rejection")
            persist(prefix+"-judgement", None, row)
            if accepted:
                result.update(status="restored", run_valid=True, restored=True,
                    comparator_improved=bool(current["objective"] < float(case["reference_comparator_objective"])),
                    corrected_objective=float(current["objective"]), reference_comparator_objective=float(case["reference_comparator_objective"]))
                break
        else:
            result.update(status="two_corrections_rejected", run_valid=True)
    except Exception as error:
        primary = error
        result.update(status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error),
                      exception=type(error).__name__)
    if primary is None:
        try:
            persist("case-result", None, result)
        except Exception as error:
            result.update(status="invalid", run_valid=False, error=str(error), exception=type(error).__name__)
            error.restoration_case_result = result
            error.abort_run = True
            raise
    else:
        primary.restoration_case_result = result
        saved = base.preserve_primary(primary, "Restoration case failure persistence failed",
                                      lambda: persist("case-result", None, result) or True)
        if saved is not True:
            primary.abort_run = True
        raise primary
    return result
