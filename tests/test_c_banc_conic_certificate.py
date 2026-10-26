"""Small generated original-QP certificates only; no original data or CUDA."""

import json
import time
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tools import banc_m3_conic_certificate as cert

ACTUAL_COST = []


def problem():
    current = {"coefficients": cert.conic.base.neutral(1), "gradient": np.array([-1., 0., 0.]),
        "qerr": np.zeros((99, 7)), "lerr": np.zeros((99, 1)),
        "startup_q_jacobian": np.zeros((99, 7, 3)), "startup_ltm_jacobian": np.zeros((99, 1, 3))}
    return cert.conic.conic_problem(current, (0., 0.), .05)


def exact_solution(p):
    d = np.array([.05, 0., 0.])
    s, z = p["b"]-p["A"]@d, np.zeros_like(p["b"])
    z[0] = 1-.001*.05
    return d, s, z


def counted_factory(*args):
    started = time.perf_counter()
    row = {"constructor_requested": 1, "constructor_completed": 0, "constructor_wall_s": None,
           "solve_requested": 0, "solve_completed": 0, "solve_wall_s": 0., "iterations": 0}
    ACTUAL_COST.append(row)
    try:
        solver = clarabel.DefaultSolver(*args)
        row["constructor_completed"] = 1
    finally:
        row["constructor_wall_s"] = time.perf_counter()-started
    class Solver:
        def solve(self):
            row["solve_requested"] += 1
            began = time.perf_counter()
            try:
                result = solver.solve()
                row["solve_completed"] += 1
                row["iterations"] += result.iterations
                return result
            finally:
                row["solve_wall_s"] += time.perf_counter()-began
    return Solver()


@pytest.fixture(scope="module", autouse=True)
def actual_cost():
    yield
    print("generated_e277_certificate_actual_cost", {k: sum(row[k] for row in ACTUAL_COST) for k in
        ("constructor_requested", "constructor_completed", "constructor_wall_s", "solve_requested",
         "solve_completed", "solve_wall_s", "iterations")})


@pytest.mark.parametrize("status", ["Solved", "AlmostSolved"])
def test_direct_minimizer_has_verifiable_dual_bound_and_immutable_inputs(status):
    p = problem()
    d, s, z = exact_solution(p)
    before = {k: v.copy() for k, v in p.items() if isinstance(v, np.ndarray) or sparse.issparse(v)}
    raw = [v.copy() for v in (d, s, z)]
    result = cert.conic_certificate(p, d, s, z, status)
    receipt = result["receipt"]
    assert result["valid"] and all(receipt["checks"].values())
    assert receipt["primal_objective"] == pytest.approx(-.05+.5*.001*.05**2)
    assert receipt["dual_lower_bound"] == pytest.approx(receipt["primal_objective"], abs=1e-13)
    assert abs(receipt["gap"]) < 1e-13 and receipt["stationarity_infinity_norm"] < 1e-15
    assert receipt["original_valid"] is (status == "Solved")
    json.dumps(receipt, allow_nan=False)
    for key in ("A", "P"):
        assert (p[key] != before[key]).nnz == 0
    for key, value in p.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
    for a, b in zip((d, s, z), raw, strict=True):
        np.testing.assert_array_equal(a, b)


def test_actual_tiny_clarabel_matches_original_minimizer_and_policy():
    p = problem()
    result = cert.solve_certified(p, 2., solver_factory=counted_factory)
    assert result["valid"] and result["original_valid"]
    optimum = -.05+.5*.001*.05**2
    receipt = result["certificate"]
    # With the deliberately small proximal Hessian, objective certification
    # permits a larger distance in the two nearly flat null coordinates.
    error = receipt["primal_objective"]-optimum
    assert -1e-13 <= error <= receipt["gap"]+1e-13
    assert np.linalg.norm(result["direction"]-np.array([.05, 0., 0.])) <= np.sqrt(2*(receipt["gap"]+1e-13)/.001)
    assert result["receipt"]["certificate"]["completed"] == 1
    assert result["original_receipt"]["status"] == result["receipt"]["status"] == "Solved"
    assert result["original_receipt"] == {k: v for k, v in result["receipt"].items()
        if k not in ("certificate", "original_solution_valid", "certificate_status", "certificate_receipt")}


def test_analytic_curved_soc_minimizer_and_dual_bound():
    inputs = problem()
    inputs["startup_q_jacobian"][:, :, 0] = 1.
    p = cert.conic.conic_problem(inputs, (1.34, 0.), .05)
    length = np.sqrt(693.)
    d = np.array([p["radii"][0]/length, 0., 0.])
    exact = p["b"]-p["A"]@d
    z = np.zeros_like(exact)
    multiplier = (1-.001*d[0])/length
    offset = p["nonnegative_rows"]
    z[offset+1:offset+694] = -multiplier/length
    # Construct a feasible analytic dual in f64, without modifying a solver
    # output or projecting it. Its lower bound remains independently checked.
    z[offset] = np.nextafter(np.linalg.norm(z[offset+1:offset+694]), np.inf)
    result = cert.conic_certificate(p, d, exact, z, "AlmostSolved")
    assert result["valid"] and result["receipt"]["dual_cone_slacks"][1] >= 0
    assert abs(result["receipt"]["gap"]) < 1e-13
    assert result["receipt"]["primal_objective"] == pytest.approx(.0005*d[0]**2-d[0], abs=1e-14)


@pytest.mark.parametrize("status", ["Solved", "AlmostSolved"])
def test_finite_raw_slack_mismatch_is_preserved_and_independently_certified(status):
    p = problem()
    d, s, z = exact_solution(p)
    s[0] = -.1
    result = cert.conic_certificate(p, d, s, z, status)
    assert result["valid"] and result["receipt"]["raw_affine_infinity_norm"] == .1
    assert result["receipt"]["raw_cone_slacks"][0] == -.1 and s[0] == -.1
    assert result["primal_slack"][0] == 0


@pytest.mark.parametrize("status", ["MaxTime", "MaxIterations", "NumericalError", "PrimalInfeasible"])
def test_other_status_rejected_even_with_exact_optimality(status):
    p = problem()
    result = cert.conic_certificate(p, *exact_solution(p), status)
    assert not result["valid"] and not result["receipt"]["checks"]["status"]
    assert all(v for k, v in result["receipt"]["checks"].items() if k != "status")


@pytest.mark.parametrize("status", ["Solved", "AlmostSolved"])
@pytest.mark.parametrize("bad", ["direction", "dual_negative", "dual_soc", "stationarity", "gap"])
def test_status_never_overrides_failed_certificate(status, bad):
    p = problem()
    d, s, z = exact_solution(p)
    if bad == "direction":
        d[0] = .06
    elif bad == "dual_negative":
        z[1] = -1e-14
    elif bad == "dual_soc":
        z[p["nonnegative_rows"]+1] = 1e-14
    elif bad == "stationarity":
        z[0] -= 1e-6
    else:
        # An inactive positive dual leaves stationarity untouched, but makes
        # the original dual bound/gap and complementarity fail.
        z[p["nonnegative_rows"]] = 1e-3
    result = cert.conic_certificate(p, d, s, z, status)
    assert not result["valid"]
    if bad.startswith("dual"):
        assert not result["receipt"]["checks"]["dual_cone"]
    if bad == "gap":
        assert not result["receipt"]["checks"]["gap"]
        assert not result["receipt"]["checks"]["complementarity"]


@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("bad", ["nan", "inf", "shape", "dtype"])
def test_incomplete_or_nonfinite_raw_is_bounded_rejection(which, bad):
    p = problem()
    arrays = list(exact_solution(p))
    if bad in ("nan", "inf"):
        arrays[which][0] = float(bad)
    elif bad == "shape":
        arrays[which] = arrays[which][:-1]
    else:
        arrays[which] = arrays[which].astype(np.float32)
    result = cert.conic_certificate(p, *arrays, "AlmostSolved")
    assert not result["valid"] and result["primal_slack"] is None
    assert result["receipt"]["outcome"] == "incomplete_or_nonfinite_solution"
    json.dumps(result["receipt"], allow_nan=False)


@pytest.mark.parametrize("bad", ["hessian_off_diagonal", "hessian_diagonal", "cone_rows", "A_nan", "P_inf", "noncanonical", "metadata"])
def test_malformed_original_problem_explicitly_fails(bad):
    p = problem()
    if bad.startswith("hessian"):
        p["P"] = sparse.csc_matrix(np.array([[.001, .0001 if bad.endswith("off_diagonal") else 0., 0.],
            [0., .001 if bad.endswith("off_diagonal") else .002, 0.], [0., 0., .001]]))
    elif bad == "cone_rows":
        p["cones"][1] = clarabel.SecondOrderConeT(693)
    elif bad in ("A_nan", "P_inf"):
        p[bad[0]].data[0] = np.nan if bad.endswith("nan") else np.inf
    elif bad == "noncanonical":
        p["A"].has_canonical_format = False
    else:
        p["startup_q_jacobian"] = p["startup_q_jacobian"][:-1]
    with pytest.raises(ValueError):
        cert.conic_certificate(p, *exact_solution(problem()), "Solved")


def fake_factory(p, status="AlmostSolved", fault=None):
    def factory(*args):
        if fault == "constructor":
            raise RuntimeError("generated constructor primary")
        class Solver:
            def solve(self):
                if fault == "solve":
                    raise RuntimeError("generated solve primary")
                d, s, z = exact_solution(p)
                s[0] = -.1
                return SimpleNamespace(x=d, s=s, z=z, status=status, iterations=1)
        return Solver()
    return factory


def test_wrapper_calls_original_once_preserves_raw_and_old_valid():
    p = problem()
    result = cert.solve_certified(p, 2., solver_factory=fake_factory(p))
    assert result["valid"] and not result["original_valid"]
    assert result["original_receipt"]["solution_valid"] is False
    assert result["s"][0] == -.1 and result["primal_slack"][0] == 0
    assert result["receipt"]["constructor"]["requested"] == result["receipt"]["solve"]["requested"] == 1


@pytest.mark.parametrize("fault", ["constructor", "solve"])
def test_wrapper_preserves_primary_solver_failure_without_retry(fault):
    p = problem()
    with pytest.raises(RuntimeError, match="primary") as caught:
        cert.solve_certified(p, 2., solver_factory=fake_factory(p, fault=fault))
    receipt = caught.value.conic_receipt
    assert receipt[fault]["failed"] == 1 and receipt["constructor"]["requested"] == 1
    assert receipt["certificate"]["requested"] == 0
    assert caught.value.partial_conic_solution is None


def test_certificate_failure_preserves_primary_raw_and_unknown_secondary_timing(monkeypatch):
    p = problem()
    primary = RuntimeError("generated certificate primary")
    def fail(*args):
        raise primary
    calls = [0]
    def clock():
        calls[0] += 1
        if calls[0] == 6:
            raise OSError("generated secondary timing")
        return float(calls[0])
    monkeypatch.setattr(cert, "conic_certificate", fail)
    with pytest.raises(RuntimeError) as caught:
        cert.solve_certified(p, 2., solver_factory=fake_factory(p), wall_clock=clock)
    assert caught.value is primary and primary.conic_receipt["solve"]["completed"] == 1
    assert primary.conic_receipt["certificate"] == {"requested": 1, "completed": 0, "failed": 1,
                                                   "wall_s": None, "wall_known": False}
    assert primary.partial_conic_solution["s"][0] == -.1
    assert any("timing failed" in note for note in primary.__notes__)
