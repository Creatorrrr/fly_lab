"""E277 original f64 QP/SOC certificates; no import-time reads or solves."""

from __future__ import annotations

# Preserve the primary calculation/solver failure and its raw evidence.
# ruff: noqa: BLE001
import copy
import time

import clarabel
import numpy as np
from scipy import sparse

from tools import fit_c_banc_m3_conic_source as conic

CERTIFICATE_CONTRACT = {
    "statuses": ["Solved", "AlmostSolved"], "primal_tolerance": 1e-8,
    "dual_cone_tolerance": 0., "stationarity_relative_tolerance": 1e-8,
    "gap_lower_relative_tolerance": 1e-10, "gap_upper_relative_tolerance": 1e-8,
    "complementarity_relative_tolerance": 1e-8, "hessian_diagonal": .001,
    "primal_slack": "b-A*d", "dual_lower_bound": "-.5*dot(q+A.T*z,q+A.T*z)/.001-b.T*z",
}


def _validate_problem(problem):
    require = conic.base.require
    require(isinstance(problem, dict), "Original conic problem mapping required")
    p, a, q, b = (problem[k] for k in ("P", "A", "q", "b"))
    require(isinstance(q, np.ndarray) and q.dtype == np.float64 and q.ndim == 1
            and 3 <= q.size <= 4980 and q.size % 3 == 0 and np.isfinite(q).all(),
            "Finite original float64 QP vector required")
    require(isinstance(b, np.ndarray) and b.dtype == np.float64 and b.ndim == 1
            and np.isfinite(b).all(), "Finite original float64 conic RHS required")
    n = q.size
    for key, value, shape in (("P", p, (n, n)), ("A", a, (len(b), n))):
        require(sparse.isspmatrix_csc(value) and value.dtype == np.float64
                and value.shape == shape and value.has_canonical_format
                and value.has_sorted_indices and np.isfinite(value.data).all(),
                "Canonical finite float64 CSC required: "+key)
    difference = p-.001*sparse.eye(n, format="csc")
    difference.eliminate_zeros()
    require(difference.nnz == 0, "Certificate requires exact P=.001*I")
    linear_rows = 2*n+2*(n//3)+2*(99*7+99)
    dims = (linear_rows, 694, 100)
    cones = problem["cones"]
    require(isinstance(cones, (tuple, list)) and len(cones) == 3
            and isinstance(cones[0], clarabel.NonnegativeConeT)
            and all(isinstance(c, clarabel.SecondOrderConeT) for c in cones[1:])
            and tuple(c.dim for c in cones) == dims
            and problem["nonnegative_rows"] == linear_rows,
            "Exact original nonnegative and two SOC row counts required")
    require(isinstance(b, np.ndarray) and b.dtype == np.float64 and b.shape == (sum(dims),)
            and np.isfinite(b).all(), "Finite original float64 conic RHS required")
    shapes = {"coefficients": (n//3, 3), "gradient": (n,), "bounds": (n, 2),
              "qerr": (99, 7), "lerr": (99, 1), "startup_q_jacobian": (99, 7, n),
              "startup_ltm_jacobian": (99, 1, n), "peaks": (2,), "radii": (2,)}
    for key, shape in shapes.items():
        value = problem[key]
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all(),
                "Finite original constraint metadata required: "+key)
    return dims


def _cone_slacks(value, dims):
    margins = [float(np.min(value[:dims[0]]))]
    offset = dims[0]
    for dim in dims[1:]:
        margins.append(float(value[offset]-np.linalg.norm(value[offset+1:offset+dim])))
        offset += dim
    return margins


def _number(value):
    return float(value) if np.isfinite(value) else None


def conic_certificate(problem, direction, solver_s, solver_z, status):
    """Certify the original convex problem without repairing raw d/s/z.

    Malformed problems raise. Incomplete or nonfinite solutions are bounded
    rejections with a JSON receipt. Raw auxiliary slack is diagnostic only;
    the primal cone check uses the separately returned exact affine slack.
    """
    dims = _validate_problem(problem)
    checks = {k: False for k in ("input_valid", "status", "original_residual", "primal_cone",
              "dual_cone", "finite_math", "stationarity", "gap", "complementarity")}
    checks["status"] = str(status) in CERTIFICATE_CONTRACT["statuses"]
    receipt = {"status": str(status), "valid": False, "original_valid": False,
        "checks": checks, "contract": copy.deepcopy(CERTIFICATE_CONTRACT),
        "original_minimum_residual": None, "raw_affine_infinity_norm": None,
        "raw_cone_slacks": None, "primal_cone_slacks": None, "dual_cone_slacks": None,
        "stationarity_infinity_norm": None, "stationarity_limit": None,
        "primal_objective": None, "dual_lower_bound": None, "gap": None,
        "objective_scale": None, "gap_lower_limit": None, "gap_upper_limit": None,
        "complementarity": None, "complementarity_limit": None}
    d, s, z = (np.asarray(v) for v in (direction, solver_s, solver_z))
    finite = all(v.dtype == np.float64 and v.shape == shape and np.isfinite(v).all()
                 for v, shape in ((d, problem["q"].shape), (s, problem["b"].shape), (z, problem["b"].shape)))
    checks["input_valid"] = bool(finite)
    if not finite:
        receipt["outcome"] = "incomplete_or_nonfinite_solution"
        return {"valid": False, "receipt": receipt, "primal_slack": None}
    with np.errstate(over="ignore", invalid="ignore"):
        exact = problem["b"]-problem["A"]@d
        residuals = conic.conic_residuals(problem, d)
        primal_margins, raw_margins, dual_margins = (_cone_slacks(v, dims) for v in (exact, s, z))
        w = problem["q"]+problem["A"].T@z
        stationarity = problem["P"]@d+w
        stationarity_inf = np.max(np.abs(stationarity))
        stationarity_limit = 1e-8*(1+np.max(np.abs(problem["q"])))
        primal = .5*np.dot(d, problem["P"]@d)+np.dot(problem["q"], d)
        dual = -.5*np.dot(w, w)/.001-np.dot(problem["b"], z)
        gap = primal-dual
        scale = 1+max(abs(primal), abs(dual))
        complementarity = np.dot(exact, z)
        affine_inf = np.max(np.abs(exact-s))
    numbers = (residuals["minimum"], *primal_margins, *dual_margins, stationarity_inf,
               primal, dual, gap, scale, complementarity, affine_inf, *raw_margins)
    checks["finite_math"] = bool(all(np.isfinite(v) for v in numbers)
                                  and np.isfinite(exact).all() and np.isfinite(w).all())
    checks["original_residual"] = bool(np.isfinite(residuals["minimum"]) and residuals["minimum"] >= -1e-8)
    checks["primal_cone"] = bool(all(np.isfinite(v) and v >= -1e-8 for v in primal_margins))
    checks["dual_cone"] = bool(all(np.isfinite(v) and v >= 0 for v in dual_margins))
    checks["stationarity"] = bool(np.isfinite(stationarity_inf) and stationarity_inf <= stationarity_limit)
    checks["gap"] = bool(np.isfinite(gap) and np.isfinite(scale) and -1e-10*scale <= gap <= 1e-8*scale)
    checks["complementarity"] = bool(np.isfinite(complementarity) and np.isfinite(scale)
                                      and abs(complementarity) <= 1e-8*scale)
    receipt.update(original_valid=bool(str(status) == "Solved" and checks["original_residual"]),
        original_minimum_residual=_number(residuals["minimum"]), raw_affine_infinity_norm=_number(affine_inf),
        raw_cone_slacks=[_number(v) for v in raw_margins], primal_cone_slacks=[_number(v) for v in primal_margins],
        dual_cone_slacks=[_number(v) for v in dual_margins], stationarity_infinity_norm=_number(stationarity_inf),
        stationarity_limit=_number(stationarity_limit), primal_objective=_number(primal), dual_lower_bound=_number(dual),
        gap=_number(gap), objective_scale=_number(scale), gap_lower_limit=_number(-1e-10*scale),
        gap_upper_limit=_number(1e-8*scale), complementarity=_number(complementarity),
        complementarity_limit=_number(1e-8*scale))
    valid = bool(all(checks.values()))
    receipt.update(valid=valid, outcome="certified" if valid else "certificate_rejected")
    return {"valid": valid, "receipt": receipt, "primal_slack": exact}


def solve_certified(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter):
    """Exactly one E275 solve; preserve its raw evidence and meter certification."""
    phase = {"requested": 0, "completed": 0, "failed": 0, "wall_s": 0., "wall_known": True}
    try:
        raw = conic.solve_direction(problem, time_limit, solver_factory=solver_factory, wall_clock=wall_clock)
    except Exception as error:
        if hasattr(error, "conic_receipt"):
            error.conic_receipt["certificate"] = copy.deepcopy(phase)
        raise
    receipt = copy.deepcopy(raw["receipt"])
    receipt["certificate"] = phase
    original_receipt = copy.deepcopy(raw["receipt"])
    phase["requested"] = 1
    began, primary, certificate = None, None, None
    try:
        began = wall_clock()
        certificate = conic_certificate(problem, raw["direction"], raw["s"], raw["z"], raw["receipt"]["status"])
        phase["completed"] = 1
    except Exception as error:
        primary = error
        phase["failed"] = 1
    finally:
        try:
            if began is None:
                phase.update(wall_s=None, wall_known=False)
            else:
                phase["wall_s"] = wall_clock()-began
        except Exception as timing_error:
            phase.update(wall_s=None, wall_known=False)
            if primary is None:
                primary = timing_error
                phase["failed"] = 1
            else:
                primary.add_note("Certificate timing failed: "+repr(timing_error))
    if primary is not None:
        receipt["certificate_status"] = "execution_failure"
        primary.conic_receipt = receipt
        primary.partial_conic_solution = raw
        primary.partial_certified_solution = raw
        raise primary
    receipt.update(original_solution_valid=raw["valid"], solution_valid=certificate["valid"],
                   certificate_status=certificate["receipt"]["outcome"], certificate_receipt=certificate["receipt"])
    return {**raw, "valid": certificate["valid"], "original_valid": raw["valid"],
            "original_receipt": original_receipt, "certificate": certificate["receipt"],
            "primal_slack": certificate["primal_slack"], "receipt": receipt}
