"""E282 sparse joint startup feasibility; no import-time evaluation or solver."""
from __future__ import annotations

# Preserve attempted work and primary errors through timing/storage failures.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import banc_m3_startup_margin as margin

base, metering = margin.base, margin.restoration
CONTRACT = {**copy.deepcopy(margin.CONTRACT), "max_corrections": 4, "time_limit_s": 4.,
    "q_allocation": "one global q SOC; no proportional per-leg quota", "q_row_order": "leg,time,local joint",
    "quota_allocation": "global q; fixed per-leg LTM", "ltm_allocation": "one SOC per leg", "hessian": "identity", "clip": False,
    "coefficient_interval": "[0,8] intersect neutral+/-0.05 intersect current+/-0.05;ceil-f32 lower/floor-f32 upper stored f64;no extra inward ULP",
    "candidate_interval_check": "after current+raw correction, before actual evaluation;exact inward interval;no tolerance/clip/repair/retry"}


def coefficient_interval(current, origin):
    """Original new-value interval and nearest inward f32 endpoints, in f64."""
    c, o = np.asarray(current), np.asarray(origin)
    base.require(c.dtype == o.dtype == np.float64 and c.shape == o.shape and c.ndim == 2 and c.shape[1] == 3
        and np.isfinite(c).all() and np.isfinite(o).all(), "Finite original coefficient triples required")
    c, o = c.ravel(), o.ravel()
    lo = np.maximum.reduce((np.zeros_like(c), o-.05, c-.05))
    hi = np.minimum.reduce((np.full_like(c, 8.), o+.05, c+.05))
    lower, upper = lo.astype(np.float32), hi.astype(np.float32)
    outside = lower.astype(np.float64) < lo
    lower[outside] = np.nextafter(lower[outside], np.float32(np.inf))
    outside = upper.astype(np.float64) > hi
    upper[outside] = np.nextafter(upper[outside], np.float32(-np.inf))
    inward = np.column_stack((lower.astype(np.float64), upper.astype(np.float64)))
    base.require(np.all(inward[:, 0] <= inward[:, 1]), "Nonempty inward f32 coefficient interval required")
    return np.column_stack((lo, hi)), inward


def joint_problem(entries, origins):
    """Small block rosters allowed for generated tests; runner requires all six."""
    base.require(1 <= len(entries) <= 6 and len(entries) == len(origins), "One to six complete blocks required")
    sizes, qblocks, lblocks, cblocks, oblocks, qe, le = [], [], [], [], [], [], []
    for entry, origin in zip(entries, origins, strict=True):
        margin.validate_entry(entry, derivative=True)
        c, origin = np.asarray(entry["coefficients"]), np.asarray(origin)
        base.require(1 <= len(c) <= 1660 and margin.coefficient_checks(c, origin)["pass"]
            and np.array_equal(origin, base.neutral(len(c))), "Original neutral .05 trust/interior required")
        sizes.append(c.size); cblocks.append(c.copy()); oblocks.append(origin.copy())
        qblocks.append(sparse.csc_matrix(entry["startup_q_jacobian"].reshape(693, c.size)))
        lblocks.append(sparse.csc_matrix(entry["startup_ltm_jacobian"].reshape(99, c.size)))
        qe.append(entry["qerr"].ravel(order="C")); le.append(entry["lerr"].ravel(order="C"))
    n, k = sum(sizes), len(sizes)
    base.require(3 <= n <= 4980 and n % 3 == 0, "Bounded joint coefficient triples required")
    c, origin = np.concatenate(cblocks), np.concatenate(oblocks)
    qj, lj = sparse.block_diag(qblocks, format="csc"), sparse.block_diag(lblocks, format="csc")
    qe, le = np.concatenate(qe), np.concatenate(le)
    original_bounds, inward_bounds = coefficient_interval(c, origin)
    lo, hi = (inward_bounds-c.ravel()[:, None]).T
    identity = sparse.eye(n, format="csc", dtype=np.float64)
    triple = sparse.csc_matrix((np.ones(n), (np.repeat(np.arange(n//3), 3), np.arange(n))), shape=(n//3, n))
    peaks = np.array([.036, .025])*(1-1e-4)-1e-6
    radii = np.array([np.sqrt(4158*(.0162*(1-1e-4))**2)-1e-6,
                      np.sqrt(99*(.009*(1-1e-4))**2)-1e-6])
    blocks = [identity, -identity, triple, -triple, qj, -qj, lj, -lj]
    rhs = [hi, -lo, 8*(1-1e-6)-c.sum(1), c.sum(1)-.125*(1+1e-6),
           peaks[0]-qe, peaks[0]+qe, peaks[1]-le, peaks[1]+le]
    rows = sum(len(r) for r in rhs)
    blocks.append(sparse.vstack((sparse.csc_matrix((1, n)), -qj), format="csc"))
    rhs.append(np.r_[radii[0], qe])
    for index in range(k):
        block = lj[index*99:(index+1)*99]
        blocks.append(sparse.vstack((sparse.csc_matrix((1, n)), -block), format="csc"))
        rhs.append(np.r_[radii[1], le[index*99:(index+1)*99]])
    dims = (rows, 1+693*k, *([100]*k))
    return {"P": identity, "q": np.zeros(n), "A": sparse.vstack(blocks, format="csc"), "b": np.concatenate(rhs),
        "cones": [clarabel.NonnegativeConeT(rows), *[clarabel.SecondOrderConeT(d) for d in dims[1:]]],
        "cone_dimensions": dims, "sizes": np.array(sizes, np.int64), "parameter_slices": np.r_[0, np.cumsum(sizes)],
        "coefficients": c, "origins": origin, "bounds": np.column_stack((lo, hi)),
        "original_coefficient_bounds": original_bounds, "coefficient_bounds": inward_bounds,
        "qerr": qe.copy(), "lerr": le.copy(), "q_jacobian": qj, "ltm_jacobian": lj,
        "peaks": peaks, "radii": radii, "radius": .05, "correction_radius": .05}


def validate_problem(problem):
    sizes = problem["sizes"]
    base.require(isinstance(sizes, np.ndarray) and sizes.dtype == np.int64 and sizes.ndim == 1
        and 1 <= len(sizes) <= 6 and np.all(sizes >= 3) and np.all(sizes % 3 == 0) and sum(sizes) <= 4980,
        "Original bounded joint block sizes required")
    n, k = int(sum(sizes)), len(sizes)
    dims = (2*n+2*(n//3)+1584*k, 1+693*k, *([100]*k))
    for key, shape in (("P", (n, n)), ("A", (sum(dims), n)), ("q_jacobian", (693*k, n)), ("ltm_jacobian", (99*k, n))):
        v = problem[key]
        base.require(sparse.isspmatrix_csc(v) and v.dtype == np.float64 and v.shape == shape
            and v.has_canonical_format and v.has_sorted_indices and np.isfinite(v.data).all(), "Canonical finite joint CSC required: "+key)
    for key, shape in (("q", (n,)), ("b", (sum(dims),)), ("coefficients", (n//3, 3)), ("origins", (n//3, 3)),
                       ("bounds", (n, 2)), ("original_coefficient_bounds", (n, 2)), ("coefficient_bounds", (n, 2)),
                       ("qerr", (693*k,)), ("lerr", (99*k,)), ("peaks", (2,)), ("radii", (2,))):
        v = problem[key]
        base.require(isinstance(v, np.ndarray) and v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all(),
                     "Finite original joint metadata required: "+key)
    base.require(tuple(problem["cone_dimensions"]) == dims and len(problem["cones"]) == len(dims)
        and isinstance(problem["cones"][0], clarabel.NonnegativeConeT)
        and all(isinstance(c, clarabel.SecondOrderConeT) for c in problem["cones"][1:])
        and tuple(c.dim for c in problem["cones"]) == dims, "Exact joint cone roster required")
    base.require(np.array_equal(problem["parameter_slices"], np.r_[0, np.cumsum(sizes)])
        and problem["radius"] == problem["correction_radius"] == .05, "Original joint slices/trust required")
    # Reconstruct rows from each sparse local block without a dense global J.
    entries, origins, begin = [], [], 0
    for index, size in enumerate(sizes):
        end = begin+int(size)
        for key, width in (("q_jacobian", 693), ("ltm_jacobian", 99)):
            block = problem[key][index*width:(index+1)*width]
            base.require(block[:, :begin].nnz == block[:, end:].nnz == 0, "Joint J must be block diagonal")
        c = problem["coefficients"].ravel()[begin:end].reshape(-1, 3)
        entries.append({"coefficients": c, "qerr": problem["qerr"][index*693:(index+1)*693].reshape(99, 7),
            "lerr": problem["lerr"][index*99:(index+1)*99].reshape(99, 1), "startup_q": np.zeros((101, 7)),
            "startup_ltm": np.zeros((101, 1)),
            "startup_q_jacobian": problem["q_jacobian"][index*693:(index+1)*693, begin:end].toarray().reshape(99, 7, int(size)),
            "startup_ltm_jacobian": problem["ltm_jacobian"][index*99:(index+1)*99, begin:end].toarray().reshape(99, 1, int(size))})
        origins.append(problem["origins"].ravel()[begin:end].reshape(-1, 3)); begin = end
    exact = joint_problem(entries, origins)
    for key in ("P", "A"):
        delta = problem[key]-exact[key]; delta.eliminate_zeros()
        base.require(delta.nnz == 0, "Joint original sparse equation differs: "+key)
    for key in ("q", "b", "bounds", "original_coefficient_bounds", "coefficient_bounds", "peaks", "radii"):
        base.require(np.array_equal(problem[key], exact[key]), "Joint original metadata/RHS differs: "+key)
    return dims


def joint_residuals(problem, correction):
    z = np.asarray(correction)
    base.require(z.dtype == np.float64 and z.shape == problem["q"].shape and np.isfinite(z).all(), "Finite complete original joint correction required")
    c = problem["coefficients"].ravel()+z; sums = c.reshape(-1, 3).sum(1)
    qp, lp = problem["qerr"]+problem["q_jacobian"]@z, problem["lerr"]+problem["ltm_jacobian"]@z
    arrays = {"coefficient": np.r_[c, 8-c, sums-.125*(1+1e-6), 8*(1-1e-6)-sums],
        "trust": .05-np.abs(c-problem["origins"].ravel()), "correction": .05-np.abs(z),
        "bounds": np.r_[z-problem["bounds"][:, 0], problem["bounds"][:, 1]-z],
        "inward_coefficient_interval": np.r_[c-problem["coefficient_bounds"][:, 0], problem["coefficient_bounds"][:, 1]-c],
        "peak": np.r_[problem["peaks"][0]-np.abs(qp), problem["peaks"][1]-np.abs(lp)],
        "soc": np.r_[problem["radii"][0]-np.linalg.norm(qp),
                     problem["radii"][1]-np.linalg.norm(lp.reshape(-1, 99), axis=1)]}
    return {**arrays, "minimum": float(min(np.min(a) for a in arrays.values()))}


def joint_certificate(problem, correction, solver_s, solver_y, status):
    dims = validate_problem(problem)
    z, s, y = (np.asarray(v) for v in (correction, solver_s, solver_y))
    checks = {k: False for k in ("input_valid", "status", "residual", "primal_cone", "dual_cone", "finite_math", "stationarity", "gap", "complementarity")}
    checks["status"] = str(status) in ("Solved", "AlmostSolved")
    receipt = {"status": str(status), "valid": False, "checks": checks}
    checks["input_valid"] = all(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all()
        for v, shape in ((z, problem["q"].shape), (s, problem["b"].shape), (y, problem["b"].shape)))
    if not checks["input_valid"]:
        return {"valid": False, "receipt": receipt, "primal_slack": None, "residuals": None}
    with np.errstate(over="ignore", invalid="ignore"):
        exact = problem["b"]-problem["A"]@z; res = joint_residuals(problem, z)
        cones = [metering.original_certificate._cone_slacks(v, dims) for v in (exact, s, y)]
        w = problem["A"].T@y; stationarity = np.max(abs(z+w))
        primal, dual = .5*np.dot(z, z), -.5*np.dot(w, w)-np.dot(problem["b"], y)
        gap, scale = primal-dual, 1+max(abs(primal), abs(dual))
        comp, affine = np.dot(exact, y), np.max(abs(exact-s))
    checks.update(finite_math=bool(all(np.isfinite(v) for v in (res["minimum"], stationarity, primal, dual, gap, scale, comp, affine,
        *(v for cone in cones for v in cone)))), residual=bool(res["minimum"] >= -1e-8),
        primal_cone=bool(all(v >= -1e-8 for v in cones[0])), dual_cone=bool(all(v >= 0 for v in cones[2])),
        stationarity=bool(stationarity <= 1e-8), gap=bool(-1e-10*scale <= gap <= 1e-8*scale),
        complementarity=bool(abs(comp) <= 1e-8*scale))
    num = metering.original_certificate._number
    receipt.update(valid=bool(all(checks.values())), minimum_residual=num(res["minimum"]), raw_affine_infinity_norm=num(affine),
        raw_cone_slacks=[num(v) for v in cones[1]], primal_cone_slacks=[num(v) for v in cones[0]], dual_cone_slacks=[num(v) for v in cones[2]],
        stationarity_infinity_norm=num(stationarity), primal_objective=num(primal), dual_lower_bound=num(dual), gap=num(gap),
        objective_scale=num(scale), complementarity=num(comp))
    return {"valid": receipt["valid"], "receipt": receipt, "primal_slack": exact, "residuals": res}


def solve_joint(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    validate_problem(problem)
    base.require(clarabel.__version__ == "0.11.1" and np.isfinite(time_limit) and 0 < time_limit <= 4, "Frozen Clarabel and bounded joint time limit required")
    settings = clarabel.DefaultSettings()
    applied = {"verbose": False, "max_iter": 100, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8,
               "tol_feas": 1e-8, "max_threads": 1, "time_limit": float(time_limit)}
    for name, value in applied.items():
        setattr(settings, name, value); base.require(getattr(settings, name) == value, "Joint solver setting differs")
    receipt = {"settings": applied, "iterations": None, "iterations_known": False, "status": "requested",
               **{k: metering.phase() for k in ("constructor", "solve", "certificate")}}
    raw = None
    try:
        solver = metering.metered(receipt, "constructor", lambda: solver_factory(problem["P"], problem["q"], problem["A"], problem["b"], problem["cones"], settings), wall_clock)
        raw = metering.metered(receipt, "solve", solver.solve, wall_clock)
        receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        z, s, y = (np.asarray(v, np.float64) for v in (raw.x, raw.s, raw.z))
        cert = metering.metered(receipt, "certificate", lambda: joint_certificate(problem, z, s, y, raw.status), wall_clock)
        receipt.update(solution_valid=cert["valid"], certificate_receipt=cert["receipt"], solver_info={k: metering.previous.finite_scalar(getattr(raw, k))
            for k in ("solve_time", "obj_val", "obj_val_dual", "r_prim", "r_dual") if hasattr(raw, k)})
        return {**cert, "correction": z, "solver_s": s, "solver_y": y, "receipt": receipt}
    except Exception as error:
        if raw is None and getattr(error, "restoration_phase", None) == "solve":
            raw = getattr(error, "phase_partial_result", None)
            if raw is not None:
                receipt.update(status=str(raw.status), iterations=int(raw.iterations), iterations_known=True)
        error.joint_solve_receipt, error.partial_joint_solution = copy.deepcopy(receipt), raw
        raise


def strict_joint(entries, origins):
    base.require(1 <= len(entries) <= 6 and len(entries) == len(origins), "Complete joint block roster required")
    checks, per_leg, qe = {}, [], []
    for index, (entry, origin) in enumerate(zip(entries, origins, strict=True)):
        margin.validate_entry(entry, derivative=False)
        coeff = margin.coefficient_checks(entry["coefficients"], origin)
        q, ltm = entry["qerr"], entry["lerr"]
        row = {"q_sse": margin.sum_square(q), "q_peak": float(np.max(abs(q))),
               "ltm_sse": margin.sum_square(ltm), "ltm_peak": float(np.max(abs(ltm)))}
        checks[str(index)] = bool(coeff["pass"] and row["q_peak"] <= .036*(1-1e-4)
            and row["ltm_peak"] <= .025*(1-1e-4) and row["ltm_sse"] <= 99*(.009*(1-1e-4))**2)
        row["coefficient_checks"] = coeff; per_leg.append(row); qe.append(q.ravel(order="C"))
    q_sse = margin.sum_square(np.concatenate(qe))
    checks["global_q_sse"] = q_sse <= 4158*(.0162*(1-1e-4))**2
    return {"pass": bool(all(checks.values())), "checks": checks, "per_leg": per_leg, "q_sse": q_sse,
            "q_sse_limit": 4158*(.0162*(1-1e-4))**2}


def problem_arrays(problem):
    arrays = {k: v for k, v in problem.items() if isinstance(v, (np.ndarray, float))}
    arrays["cone_dimensions"] = np.array(problem["cone_dimensions"], np.int64)
    for key in ("P", "A", "q_jacobian", "ltm_jacobian"):
        m = problem[key]; arrays.update({key+"_data": m.data, key+"_indices": m.indices, key+"_indptr": m.indptr, key+"_shape": np.array(m.shape, np.int64)})
    return arrays


def run_joint(cases, evaluate_round, *, persist, check=lambda: None, remaining=lambda: 4.,
              solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    base.require(1 <= len(cases) <= 6, "Complete joint cases required")
    for case in cases: margin.validate_case(case)
    origins = [case["neutral_coefficients"] for case in cases]
    cost = {k: metering.phase() for k in ("round", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_jacobian_rounds_used=0)
    result = {"status": "requested", "run_valid": False, "f64_success": False, "f32_success": False,
              "initialization_supported": False, "reproduction": [], "trajectory": [], "corrections": [], "cost": cost}
    primary = None
    def observe(cs, label, derivative):
        check()
        rows = None
        try:
            rows = metering.metered(cost, "round", lambda: evaluate_round(cs, derivative=derivative, label=label), wall_clock)
            base.require(len(rows) == len(cases), "Incomplete joint actual evaluation round")
            for entry, c, case in zip(rows, cs, cases, strict=True):
                margin.validate_entry(entry, derivative=derivative)
                base.require(np.array_equal(entry["coefficients"], c)
                    and np.array_equal(entry["qerr"], entry["startup_q"][1:100]-case["target_q"])
                    and np.array_equal(entry["lerr"], entry["startup_ltm"][1:100]-case["target_ltm"]), "Joint coefficient/error/target mismatch")
            check(); return rows
        except Exception as error:
            partial = rows if rows is not None else getattr(error, "partial_joint_round", None)
            if partial is None and getattr(error, "restoration_phase", None) == "round":
                partial = getattr(error, "phase_partial_result", None)
            if isinstance(partial, list):
                result["partial_round"] = {"label": label, "completed_blocks": len(partial), "expected_blocks": len(cases)}
                for index, entry in enumerate(partial):
                    arrays = {k: v for k, v in entry.items() if isinstance(v, np.ndarray)}
                    base.preserve_primary(error, "Partial joint round persistence failed",
                        lambda index=index, arrays=arrays: persist(f"{label}-partial-{index:02d}", arrays, {"derivative": derivative}))
            raise
    def account(receipt):
        for key in ("constructor", "solve", "certificate"):
            for field in ("requested", "completed", "failed"): cost[key][field] += receipt[key][field]
            if not receipt[key]["wall_known"]: cost[key].update(wall_s=None, wall_known=False)
            elif cost[key]["wall_s"] is not None: cost[key]["wall_s"] += receipt[key]["wall_s"]
        if receipt["iterations_known"]: cost["iterations"] += receipt["iterations"]
        elif receipt["solve"]["requested"]: cost["iterations_unknown_calls"] += 1
    try:
        current = observe([o.copy() for o in origins], "neutral", True)
        for row, case in zip(current, cases, strict=True):
            gates = {actual: base.numerical_gate(row[actual], case[saved], tolerance) for actual, saved, tolerance in
                (("startup_q", "neutral_startup_q", 1e-10), ("startup_ltm", "neutral_startup_ltm", 1e-10),
                 ("startup_q_jacobian", "neutral_q_jacobian", 1e-9), ("startup_ltm_jacobian", "neutral_ltm_jacobian", 1e-9))}
            result["reproduction"].append(gates)
        persist("reproduction", None, result["reproduction"])
        if not all(g["pass"] for row in result["reproduction"] for g in row.values()):
            raise metering.ReproductionError("Joint neutral q/LTM/J reproduction failed")
        judgement = strict_joint(current, origins); result["trajectory"].append(judgement)
        result["f64_success"] = judgement["pass"]
        for iteration in range(4):
            if result["f64_success"]: break
            check(); problem = joint_problem(current, origins); cost["fresh_jacobian_rounds_used"] += 1
            prefix = f"joint-{iteration:02d}"
            persist(prefix+"-problem", problem_arrays(problem), {"iteration": iteration})
            check(); limit = min(4., remaining())
            if limit <= 0: raise base.BudgetExceeded("original")
            try:
                solved = solve_joint(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock); account(solved["receipt"])
            except Exception as error:
                receipt = getattr(error, "joint_solve_receipt", None)
                if receipt is not None:
                    account(receipt); result["corrections"].append({"receipt": receipt, "iteration": iteration, "status": "execution_failure"})
                    raw = getattr(error, "partial_joint_solution", None)
                    arrays = {} if raw is None else {k: np.asarray(getattr(raw, attr), np.float64) for k, attr in (("correction", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                    base.preserve_primary(error, "Partial joint solve persistence failed", lambda prefix=prefix, arrays=arrays, receipt=receipt: persist(prefix+"-solve", arrays, receipt))
                raise
            arrays = {k: solved[k] for k in ("correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
            if solved["residuals"] is not None: arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
            persist(prefix+"-solve", arrays, solved["receipt"])
            row = {"iteration": iteration, "receipt": solved["receipt"], "actual_evaluated": False}; result["corrections"].append(row)
            if not solved["valid"]:
                result["status"] = "uncertified_joint_correction"; break
            begin, cs = 0, []
            for entry in current:
                end = begin+entry["coefficients"].size
                cs.append(entry["coefficients"]+solved["correction"][begin:end].reshape(entry["coefficients"].shape)); begin = end
            bounds = margin.coefficient_checks(np.concatenate(cs), np.concatenate(origins), solved["correction"])
            new_values = np.concatenate(cs).ravel()
            inward_pass = bool(np.all(new_values >= problem["coefficient_bounds"][:, 0])
                               and np.all(new_values <= problem["coefficient_bounds"][:, 1]))
            original_pass = bounds["pass"]
            bounds.update(original_coefficient_checks_pass=original_pass, inward_coefficient_interval=inward_pass)
            bounds["pass"] = bool(original_pass and inward_pass)
            persist(prefix+"-coefficients", {"coefficients_leg_major": np.concatenate(cs)}, bounds)
            if not original_pass:
                result["status"] = "strict_coefficient_trust_or_correction_rejection"; break
            if not inward_pass:
                result["status"] = "strict_inward_coefficient_interval_rejection"; break
            current = observe(cs, prefix+"-actual", True); row["actual_evaluated"] = True
            judgement = strict_joint(current, origins); result["trajectory"].append(judgement); row["judgement"] = judgement
            persist(prefix+"-judgement", None, row); result["f64_success"] = judgement["pass"]
        else:
            result["status"] = "four_joint_corrections_rejected"
        if result["f64_success"]:
            cs32 = [entry["coefficients"].astype(np.float32) for entry in current]; cs = [c.astype(np.float64) for c in cs32]
            bounds = margin.coefficient_checks(np.concatenate(cs), np.concatenate(origins))
            persist("f32-coefficients", {"coefficients32_leg_major": np.concatenate(cs32), "coefficients_leg_major": np.concatenate(cs)}, bounds)
            if not bounds["pass"]: result["status"] = "f32_coefficient_or_trust_rejection_before_evaluation"
            else:
                final = observe(cs, "f32-actual", False); judgement = strict_joint(final, origins)
                result.update(f32_success=judgement["pass"], f32_judgement=judgement, status="supported" if judgement["pass"] else "f32_joint_startup_rejection")
        result.update(run_valid=True, initialization_supported=bool(result["f64_success"] and result["f32_success"]))
    except Exception as error:
        primary = error; result.update(run_valid=False, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
    if primary is None:
        try: persist("joint-result", None, result)
        except Exception as error:
            result.update(run_valid=False, initialization_supported=False, status="invalid", error=str(error), exception=type(error).__name__)
            error.joint_result, error.abort_run = result, True; raise
    else:
        primary.joint_result = result
        saved = base.preserve_primary(primary, "Joint failure persistence failed", lambda: persist("joint-result", None, result) or True)
        if saved is not True: primary.abort_run = True
        raise primary
    return result
