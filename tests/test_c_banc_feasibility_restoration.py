"""Tiny generated E280 equations/failure cases; no original or CUDA execution."""
import copy
import time
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest

from tools import banc_m3_feasibility_restoration as fit

ACTUAL_COST = []


def counted_factory(*args):
    row = {"constructor_requested": 1, "constructor_completed": 0, "constructor_wall_s": 0.,
           "solve_requested": 0, "solve_completed": 0, "solve_wall_s": 0., "iterations": 0}
    ACTUAL_COST.append(row)
    began = time.perf_counter()
    try:
        solver = clarabel.DefaultSolver(*args)
        row["constructor_completed"] += 1
    finally:
        row["constructor_wall_s"] += time.perf_counter()-began
    class Solver:
        def solve(self):
            row["solve_requested"] += 1
            began = time.perf_counter()
            try:
                raw = solver.solve()
                row["solve_completed"] += 1
                row["iterations"] += raw.iterations
                return raw
            finally:
                row["solve_wall_s"] += time.perf_counter()-began
    return Solver()


@pytest.fixture(scope="module", autouse=True)
def actual_cost():
    yield
    print("generated_e280_tiny_actual_cost", {k: sum(r[k] for r in ACTUAL_COST) for k in
        ("constructor_requested", "constructor_completed", "constructor_wall_s", "solve_requested",
         "solve_completed", "solve_wall_s", "iterations")})


def entry(c, evaluation=0, *, nonlinear=True):
    c = np.asarray(c, np.float64).reshape(1, 3)
    t = c[0, 0]-1
    qe, le = np.zeros((99, 7)), np.zeros((99, 1))
    jq, jl = np.zeros((99, 7, 3)), np.zeros((99, 1, 3))
    qe[0, 0] = 3*t+(10*t*t if nonlinear else 0)
    jq[0, 0, 0] = 3+(20*t if nonlinear else 0)
    h, cj = fit.base.startup_constraints(qe, le, jq, jl, np.zeros(2))
    q, ltm = np.zeros((101, 7)), np.zeros((101, 1))
    q[1:100], ltm[1:100] = qe, le
    objective = 1-4*t
    return {"coefficients": c.copy(), "evaluation": evaluation, "objective": objective,
        "gradient": np.array([-4., 0., 0.]), "mse": .1*objective, "baseline_mse": .1,
        "qerr": qe, "lerr": le, "constraints": h, "constraint_jacobian": cj,
        "startup_q_jacobian": jq, "startup_ltm_jacobian": jl,
        "startup_q": q, "startup_ltm": ltm, "valid": bool(np.all(h >= 0))}


def case_pack(alpha=.25):
    before, trial = entry([1, 0, 0]), entry([1.02, 0, 0], 1)
    case = {"before_coefficients": before["coefficients"], "trial_coefficients": trial["coefficients"],
        "comparator_coefficients": np.array([[1.01, 0., 0.]]),
        "before_startup_q": before["startup_q"], "trial_startup_q": trial["startup_q"],
        "before_startup_ltm": before["startup_ltm"], "trial_startup_ltm": trial["startup_ltm"],
        "before_gradient": before["gradient"], "fixed_sse": np.zeros(2)}
    case.update({k: np.array(v, dtype=np.float64) for k, v in
        {"before_objective": 1., "trial_objective": trial["objective"], "baseline_mse": .1,
         "radius": .05, "alpha": alpha, "correction_radius": np.max(np.abs(trial["coefficients"]-before["coefficients"])),
         "reference_comparator_objective": .96}.items()})
    return case


def analytic_problem():
    current = entry([1.02, .01, .01])
    current["qerr"][:] = 0
    current["qerr"][0, 0] = .05
    current["startup_q_jacobian"][:] = 0
    current["startup_q_jacobian"][0, 0, 0] = 1
    return fit.restoration_problem(current, fit.base.neutral(1), np.zeros(2), .05, .02)


def analytic_solution(problem):
    correction = np.array([problem["peaks"][0]-.05, 0., 0.])
    slack = problem["b"]-problem["A"]@correction
    dual = np.zeros_like(slack)
    dual[8] = -correction[0]
    return correction, slack, dual


def test_minimum_norm_analytic_and_actual_tiny_solver():
    problem = analytic_problem()
    expected, s, y = analytic_solution(problem)
    assert fit.restoration_certificate(problem, expected, s, y, "AlmostSolved")["valid"]
    actual = fit.solve_restoration(problem, 2., solver_factory=counted_factory)
    assert actual["valid"] and actual["receipt"]["iterations_known"]
    np.testing.assert_allclose(actual["correction"], expected, atol=1e-6, rtol=0)
    assert fit.restoration_residuals(problem, actual["correction"])["minimum"] >= -1e-8
    assert problem["P"].has_canonical_format and problem["P"].nnz == 3


def test_infeasible_start_is_allowed_but_impossible_linear_problem_rejected():
    problem = analytic_problem()
    problem["A"] = type(problem["A"])(problem["A"].shape, dtype=np.float64)
    problem["A"].eliminate_zeros()
    # Peak RHS remains negative; no correction can alter it.
    solved = fit.solve_restoration(problem, 2., solver_factory=counted_factory)
    assert not solved["valid"] and solved["receipt"]["status"] == "PrimalInfeasible"


def test_all_peak_rows_soc_signs_global_fixed_sse_and_absolute_margin():
    problem = analytic_problem()
    d = np.array([-.01, .002, -.001])
    exact = problem["b"]-problem["A"]@d
    nq = problem["nonnegative_rows"]
    np.testing.assert_array_equal(exact[nq+1:nq+694],
        problem["qerr"].ravel()+problem["startup_q_jacobian"].reshape(693, 3)@d)
    assert exact[nq] == np.sqrt(99*42*(.018*.9999)**2)-1e-6
    assert exact[nq+694] == np.sqrt(99*6*(.009*.9999)**2)-1e-6
    assert nq == 2*3+2+2*(693+99)
    current = entry([1.02, .01, .01])
    fixed = np.array([.2, .003])
    p = fit.restoration_problem(current, fit.base.neutral(1), fixed, .05, .02)
    np.testing.assert_allclose(p["radii"], np.sqrt(p["available_sse"])-1e-6, rtol=0, atol=0)
    assert p["peaks"][0] == .04*.9999-1e-6
    with pytest.raises(ValueError, match="Negative remaining"):
        fit.restoration_problem(current, fit.base.neutral(1), np.array([10., 0.]), .05, .02)


@pytest.mark.parametrize("which", ["status", "dual", "stationarity", "primal", "nan", "shape", "dtype"])
def test_forged_certificate_rejected(which):
    p = analytic_problem()
    d, s, y = analytic_solution(p)
    status = "Solved"
    if which == "status":
        status = "MaxIterations"
    elif which == "dual":
        y[0] = -1e-12
    elif which == "stationarity":
        y[8] += 1e-6
    elif which == "primal":
        d[0] += 1e-6
    elif which == "nan":
        s[0] = np.nan
    elif which == "shape":
        s = s[:-1]
    elif which == "dtype":
        d = d.astype(np.float32)
    assert not fit.restoration_certificate(p, d, s, y, status)["valid"]


def test_finite_raw_slack_difference_is_diagnostic_not_a_new_gate():
    p = analytic_problem()
    d, s, y = analytic_solution(p)
    cert = fit.restoration_certificate(p, d, s-.2, y, "Solved")
    assert cert["valid"] and cert["receipt"]["raw_affine_infinity_norm"] > .19
    assert cert["receipt"]["raw_cone_slacks"][0] < 0


def test_dual_gap_and_complementarity_are_independent_requirements():
    p = analytic_problem()
    p["qerr"][:] = 0
    # Use a valid zero primal with a null-J peak dual: stationary yet nonoptimal dual.
    p["startup_q_jacobian"][:] = 0
    p["A"] = fit.restoration_problem(entry([1.01, .01, .01]), fit.base.neutral(1), np.zeros(2), .05, .02)["A"]
    p["A"][8, 0] = 0
    p["A"].eliminate_zeros()
    d = np.zeros(3)
    y = np.zeros_like(p["b"])
    y[9] = 1e-3
    s = p["b"]-p["A"]@d
    cert = fit.restoration_certificate(p, d, s, y, "Solved")
    assert cert["receipt"]["checks"]["stationarity"]
    assert not cert["receipt"]["checks"]["gap"] and not cert["receipt"]["checks"]["complementarity"]


def stub_solver(directions, *, valid=True):
    calls = []
    def solve(p, *args, **kwargs):
        d = np.array(directions[len(calls)], np.float64)
        calls.append({k: copy.deepcopy(v) for k, v in p.items() if k != "cones"})
        receipt = {k: {**fit.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, status="Solved", generated_stub=True)
        return {"correction": d, "solver_s": p["b"]-p["A"]@d, "solver_y": np.zeros_like(p["b"]),
            "primal_slack": p["b"]-p["A"]@d, "residuals": fit.restoration_residuals(p, d), "receipt": receipt, "valid": valid}
    return solve, calls


def run_generated(monkeypatch, directions, *, comparator=.96, persist=None, mutate=None, valid=True):
    case = case_pack()
    case["reference_comparator_objective"] = np.array(comparator)
    evaluated, saved = [], {}
    def evaluate(c):
        row = entry(c, len(evaluated))
        if mutate is not None:
            mutate(row)
        evaluated.append(row)
        return row
    solve, calls = stub_solver(directions, valid=valid)
    monkeypatch.setattr(fit, "solve_restoration", solve)
    def store(name, arrays, receipt):
        saved[name] = (arrays, copy.deepcopy(receipt))
        if persist:
            persist(name, arrays, receipt)
    result = fit.restore_case(case, evaluate, persist=store)
    return result, calls, evaluated, saved


def test_early_success_and_comparator_are_separate(monkeypatch):
    result, calls, evaluated, _ = run_generated(monkeypatch, [[-.01, 0., 0.]], comparator=.95)
    assert result["restored"] and not result["comparator_improved"]
    assert len(calls) == 1 and len(evaluated) == 3 and result["cost"]["fresh_trial_jacobians_used"] == 1


def test_two_actual_failures_no_third_solve_and_fresh_jacobian(monkeypatch):
    result, calls, evaluated, _ = run_generated(monkeypatch, [[-.0001, 0., 0.], [-.0001, 0., 0.]])
    assert result["status"] == "two_corrections_rejected" and len(calls) == 2 and len(evaluated) == 4
    assert not np.array_equal(calls[0]["startup_q_jacobian"], calls[1]["startup_q_jacobian"])
    np.testing.assert_array_equal(calls[1]["startup_q_jacobian"], evaluated[2]["startup_q_jacobian"])
    for p in calls:
        np.testing.assert_array_equal(p["before_coefficients"], fit.base.neutral(1))


def test_actual_tiny_nonlinear_restoration_uses_two_fresh_jacobians():
    evaluated, saved = [], {}
    def evaluate(c):
        row = entry(c, len(evaluated))
        evaluated.append(row)
        return row
    result = fit.restore_case(case_pack(), evaluate, persist=lambda n, a, r: saved.update({n: (a, r)}),
                              solver_factory=counted_factory)
    assert result["run_valid"] and result["restored"] and result["comparator_improved"]
    assert len(evaluated) == 4 and len(result["corrections"]) == 2
    assert result["cost"]["solve"]["requested"] == result["cost"]["solve"]["completed"] == 2
    assert result["cost"]["fresh_trial_jacobians_used"] == 2
    assert all(row["receipt"]["certificate_receipt"]["valid"] for row in result["corrections"])
    assert not result["corrections"][0]["strict_accepted"]
    np.testing.assert_array_equal(saved["correction-01-problem"][0]["startup_q_jacobian"], evaluated[2]["startup_q_jacobian"])


@pytest.mark.parametrize("direction,status", [([-.01, -1e-12, 0], "strict_coefficient_rejection"),
    ([.030000000001, 0, 0], "strict_trust_or_correction_rejection"),
    ([-.020000000001, 0, 0], "strict_trust_or_correction_rejection")])
def test_certified_tolerance_cannot_relax_actual_bounds(monkeypatch, direction, status):
    result, calls, evaluated, saved = run_generated(monkeypatch, [direction])
    assert result["status"] == status and result["run_valid"] and len(evaluated) == 2 and len(calls) == 1
    assert not result["corrections"][0]["actual_evaluated"]
    assert not result["corrections"][0]["true_startup_verified"]
    np.testing.assert_array_equal(saved["correction-00-solve"][0]["correction"], direction)


def test_armijo_and_train_nonworse_remain_strict(monkeypatch):
    def mutate(row):
        if row["evaluation"] >= 2:
            row["objective"] = 1.1
            row["mse"] = .11
    result, _, _, _ = run_generated(monkeypatch, [[-.01, 0., 0.], [0., 0., 0.]], mutate=mutate)
    assert not result["restored"] and result["status"] == "two_corrections_rejected"


def test_reproduction_failure_has_no_solver_calls(monkeypatch):
    def mutate(row):
        row["startup_q"][0, 0] += 1e-8
    with pytest.raises(fit.ReproductionError) as caught:
        run_generated(monkeypatch, [], mutate=mutate)
    assert caught.value.restoration_case_result["cost"]["solve"]["requested"] == 0


def test_final_storage_failure_retains_completed_case_cost(monkeypatch):
    original = OSError("final storage")
    def persist(name, *_):
        if name == "case-result":
            raise original
    with pytest.raises(OSError) as caught:
        run_generated(monkeypatch, [[-.01, 0., 0.]], persist=persist)
    assert caught.value is original and original.abort_run
    assert original.restoration_case_result["cost"]["solve"]["requested"] == 1
    assert original.restoration_case_result["cost"]["evaluation"]["completed"] == 3
    assert not original.restoration_case_result["run_valid"]


@pytest.mark.parametrize("complete", [False, True])
def test_original_deadline_identity_and_partial_evaluation(complete):
    case = case_pack()
    original = fit.base.BudgetExceeded("original")
    original.add_note("first note")
    saved = {}
    def evaluate(c):
        if complete:
            original.completed_restoration_entry = entry(c)
        raise original
    def persist(name, arrays, receipt):
        saved[name] = arrays
        if name == "case-result":
            raise OSError("secondary")
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.restore_case(case, evaluate, persist=persist)
    assert caught.value is original and "first note" in original.__notes__
    assert any("secondary" in note for note in original.__notes__)
    assert original.restoration_case_result["status"] == "incomplete"
    assert original.restoration_case_result["cost"]["evaluation"]["completed"] == int(complete)
    assert ("before" in saved) == complete and original.abort_run


def test_certificate_failure_preserves_actual_completed_raw(monkeypatch):
    original = ArithmeticError("certificate")
    raw = SimpleNamespace(x=[-.01, 0., 0.], s=np.zeros(2386), z=np.zeros(2386), status="Solved", iterations=7)
    class Solver:
        def solve(self):
            return raw
    def certificate(*_):
        raise original
    monkeypatch.setattr(fit, "restoration_certificate", certificate)
    saved = {}
    with pytest.raises(ArithmeticError) as caught:
        fit.restore_case(case_pack(), lambda c: entry(c), persist=lambda n, a, r: saved.update({n: a}),
                         solver_factory=lambda *_: Solver())
    assert caught.value is original
    assert original.partial_restoration_solution is raw
    assert original.restoration_case_result["cost"]["iterations"] == 7
    np.testing.assert_array_equal(saved["correction-00-solve"]["correction"], raw.x)


def test_solve_completion_timing_failure_preserves_raw_and_unknown_wall():
    p = analytic_problem()
    d, s, y = analytic_solution(p)
    raw = SimpleNamespace(x=d, s=s, z=y, status="Solved", iterations=4)
    original = OSError("clock")
    count = [0]
    def clock():
        count[0] += 1
        if count[0] == 4:
            raise original
        return float(count[0])
    solver = SimpleNamespace(solve=lambda: raw)
    with pytest.raises(OSError) as caught:
        fit.solve_restoration(p, 2., solver_factory=lambda *_: solver, wall_clock=clock)
    assert caught.value is original and original.partial_restoration_solution is raw
    assert original.restoration_receipt["solve"]["completed"] == 1
    assert original.restoration_receipt["solve"]["wall_s"] is None
    assert original.restoration_receipt["iterations"] == 4


@pytest.mark.parametrize("mutation", ["trial_gradient", "shape", "float32", "radius", "negative", "nan"])
def test_case_schema_fail_closed(mutation):
    case = case_pack()
    if mutation == "trial_gradient":
        case[mutation] = np.zeros(3)
    elif mutation == "shape":
        case["trial_startup_q"] = np.zeros((99, 7))
    elif mutation == "float32":
        case["before_gradient"] = np.zeros(3, np.float32)
    elif mutation == "radius":
        case["radius"] = np.array(.1)
    elif mutation == "negative":
        case["fixed_sse"][0] = -1
    elif mutation == "nan":
        case["before_gradient"][0] = np.nan
    with pytest.raises(ValueError):
        fit.validate_case(case)
