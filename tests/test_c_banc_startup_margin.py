"""Tiny generated E281 feasibility equations and bounded-loop tests."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from tests.test_c_banc_feasibility_restoration import ACTUAL_COST, counted_factory
from tools import banc_m3_startup_margin as fit


@pytest.fixture(scope="module", autouse=True)
def tiny_cost():
    began = len(ACTUAL_COST)
    yield
    print("generated_e281_actual_tiny_cost", {k: sum(r[k] for r in ACTUAL_COST[began:]) for k in
        ("constructor_requested", "constructor_completed", "constructor_wall_s", "solve_requested",
         "solve_completed", "solve_wall_s", "iterations")})


def entry(c, index=0, *, derivative=True):
    c = np.asarray(c, np.float64).reshape(1, 3)
    t = c[0, 0]-1
    e, le = np.zeros((99, 7)), np.zeros((99, 1))
    e[0, 0] = .04-3*t+10*t*t
    q, ltm = np.zeros((101, 7)), np.zeros((101, 1))
    q[1:100], ltm[1:100] = e, le
    result = {"coefficients": c.copy(), "evaluation": index, "startup_q": q, "startup_ltm": ltm, "qerr": e, "lerr": le}
    if derivative:
        jq, jl = np.zeros((99, 7, 3)), np.zeros((99, 1, 3))
        jq[0, 0, 0] = -3+20*t
        result.update(startup_q_jacobian=jq, startup_ltm_jacobian=jl)
    return result


def case_pack():
    row = entry([1., 0., 0.])
    return {"neutral_coefficients": row["coefficients"], "neutral_startup_q": row["startup_q"],
        "neutral_startup_ltm": row["startup_ltm"], "neutral_q_jacobian": row["startup_q_jacobian"],
        "neutral_ltm_jacobian": row["startup_ltm_jacobian"], "target_q": np.zeros((99, 7)), "target_ltm": np.zeros((99, 1)),
        "q_sse_quota": np.array(4158*(.0162*.9999)**2/6), "ltm_sse_quota": np.array(99*(.009*.9999)**2)}


def problem(continuous=False):
    row = entry([1., 0., 0.])
    row["qerr"][:] = .02 if continuous else 0
    if not continuous:
        row["qerr"][0, 0] = .04
    row["startup_q_jacobian"][:] = 0
    row["startup_q_jacobian"][:, :, 0] = 1 if continuous else 0
    if not continuous:
        row["startup_q_jacobian"][0, 0, 0] = 1
    case = case_pack()
    return fit.startup_margin_problem(row, case["neutral_coefficients"], float(case["q_sse_quota"]), float(case["ltm_sse_quota"]))


def test_quota_allocation_global_and_leg_meaning():
    errors = [np.full((99, 7), .01*(i+1), np.float64) for i in range(6)]
    rows = fit.quota_rows(errors)
    total = 4158*(.0162*.9999)**2
    assert np.isclose(sum(r["q_sse_quota"] for r in rows), total, rtol=0, atol=1e-15)
    assert all(r["ltm_sse_quota"] == 99*(.009*.9999)**2 for r in rows)
    expected = np.array([fit.sum_square(e) for e in errors]); expected /= expected.sum()
    np.testing.assert_array_equal([r["share"] for r in rows], expected)
    assert rows[-1]["q_sse_quota"] > 693*(.0162*.9999)**2


@pytest.mark.parametrize("mutation", ["zero", "nan", "shape", "dtype", "number"])
def test_invalid_quota_input_rejected(mutation):
    errors = [np.full((99, 7), .02) for _ in range(6)]
    if mutation == "zero":
        errors[0][:] = 0
    elif mutation == "nan":
        errors[0][0, 0] = np.nan
    elif mutation == "shape":
        errors[0] = np.zeros((98, 7))
    elif mutation == "dtype":
        errors[0] = errors[0].astype(np.float32)
    else:
        errors.pop()
    with pytest.raises(ValueError):
        fit.quota_rows(errors)


@pytest.mark.parametrize("continuous", [False, True])
def test_actual_tiny_analytic_peak_or_soc_minimum_norm(continuous):
    p = problem(continuous)
    assert fit.validate_problem(p)
    solved = fit.solve_margin(p, 2., solver_factory=counted_factory)
    assert solved["valid"] and solved["receipt"]["solve"]["completed"] == 1
    expected = p["radii"][0]/np.sqrt(693)-.02 if continuous else p["peaks"][0]-.04
    optimum = np.array([expected, 0., 0.])
    exact = p["b"]-p["A"]@optimum
    dual = np.zeros_like(exact)
    if continuous:
        offset = p["nonnegative_rows"]
        multiplier = -expected/np.sqrt(693)
        dual[offset], dual[offset+1:offset+694] = multiplier, -multiplier/np.sqrt(693)
    else:
        dual[8] = -expected
    analytic = fit.restoration.restoration_certificate(p, optimum, exact, dual, "Solved")
    assert analytic["valid"] and abs(analytic["receipt"]["gap"]) < 1e-12
    # Strong convexity bounds distance by the certified objective gap. Zero-bound
    # inactive coordinates need not reach 1e-6 with the frozen 1e-8 gap tolerance.
    gap = solved["receipt"]["certificate_receipt"]["gap"]
    assert .5*np.sum((solved["correction"]-optimum)**2) <= gap+1e-10
    assert abs(solved["correction"][0]-expected) <= 1e-6
    assert solved["receipt"]["certificate_receipt"]["valid"]
    assert p["peaks"][0] == .036*.9999-1e-6
    assert p["radii"][0] == np.sqrt(p["quotas"][0])-1e-6


def test_actual_tiny_infeasible_problem_is_scientific_rejection():
    p = problem()
    # Construct matching metadata/equations with a genuinely unchangeable violating row.
    current = entry([1., 0., 0.])
    current["startup_q_jacobian"][:] = 0
    p = fit.startup_margin_problem(current, fit.base.neutral(1), *p["quotas"])
    solved = fit.solve_margin(p, 2., solver_factory=counted_factory)
    assert not solved["valid"] and solved["receipt"]["status"] == "PrimalInfeasible"


@pytest.mark.parametrize("key", ["b", "A", "P", "peaks", "radii", "quotas", "bounds", "correction_radius"])
def test_mutated_problem_cannot_reach_solver(key):
    p = problem()
    if key in ("P", "A"):
        p[key] = p[key].copy()
        p[key].data[0] += .001
    elif key == "correction_radius":
        p[key] = .049
    else:
        p[key] = p[key].copy()
        p[key].flat[0] += .001
    with pytest.raises(ValueError):
        fit.solve_margin(p, 2., solver_factory=lambda *_: pytest.fail("invalid problem must not construct"))


def test_all_peak_soc_matrix_signs_and_saved_sparse_arrays():
    p = problem(True)
    d = np.array([-.005, 0., 0.])
    s = p["b"]-p["A"]@d; start = p["nonnegative_rows"]
    np.testing.assert_array_equal(s[start+1:start+694], p["qerr"].ravel()+p["startup_q_jacobian"].reshape(693, 3)@d)
    arrays = fit.problem_arrays(p)
    reconstructed = sparse.csc_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=arrays["A_shape"])
    np.testing.assert_array_equal(reconstructed.toarray(), p["A"].toarray())


@pytest.mark.parametrize("kind", ["q_peak", "q_quota", "ltm_peak", "ltm_quota", "coefficient", "trust"])
def test_strict_actual_limits_have_no_certificate_tolerance(kind):
    row = entry([1.01, 0., 0.]); case = case_pack()
    if kind == "q_peak":
        row["qerr"][0, 0] = .036*.9999+1e-12
    elif kind == "q_quota":
        row["qerr"][:] = np.sqrt(float(case["q_sse_quota"])/693)+1e-10
    elif kind == "ltm_peak":
        row["lerr"][0, 0] = .025*.9999+1e-12
    elif kind == "ltm_quota":
        row["lerr"][:] = .009*.9999+1e-12
    elif kind == "coefficient":
        row["coefficients"][0, 1] = -1e-12
    elif kind == "trust":
        row["coefficients"][0, 0] = 1.05+1e-12
    judged = fit.strict_startup(row, case["neutral_coefficients"], float(case["q_sse_quota"]), float(case["ltm_sse_quota"]))
    assert not judged["pass"]
    assert not judged["checks"]["neutral_trust" if kind == "trust" else kind]


def stub(directions, *, valid=True):
    calls = []
    def solve(p, *args, **kwargs):
        d = np.array(directions[len(calls)], np.float64)
        calls.append({k: copy.deepcopy(v) for k, v in p.items() if k != "cones"})
        receipt = {k: {**fit.restoration.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, generated_stub=True)
        exact = p["b"]-p["A"]@d
        return {"valid": valid, "receipt": receipt, "correction": d, "solver_s": exact,
                "solver_y": np.zeros_like(exact), "primal_slack": exact, "residuals": None}
    return solve, calls


def generated_run(monkeypatch, directions, *, mutate=None, store=None, valid=True):
    calls, saved = [], {}
    solve, problems = stub(directions, valid=valid)
    monkeypatch.setattr(fit, "solve_margin", solve)
    def evaluate(c, *, derivative):
        row = entry(c, len(calls), derivative=derivative)
        if mutate:
            mutate(row, derivative)
        row["startup_q"][1:100], row["startup_ltm"][1:100] = row["qerr"], row["lerr"]
        calls.append((derivative, row))
        return row
    def persist(name, arrays, receipt):
        saved[name] = (arrays, copy.deepcopy(receipt))
        if store:
            store(name, arrays, receipt)
    result = fit.run_leg(case_pack(), evaluate, persist=persist)
    return result, calls, problems, saved


def test_infeasible_neutral_early_success_then_f32_no_objective(monkeypatch):
    result, calls, problems, saved = generated_run(monkeypatch, [[.002, 0., 0.]])
    assert result["f64_success"] and result["f32_success"] and result["run_valid"]
    assert not result["neutral_judgement"]["pass"] and len(problems) == 1
    assert [x[0] for x in calls] == [True, True, False]
    assert saved["f32-coefficients"][0]["coefficients32"].dtype == np.float32
    assert "objective" not in calls[0][1] and "gradient" not in calls[0][1]


def test_actual_tiny_two_fresh_j_corrections_and_f32():
    calls, saved = [], {}
    def evaluate(c, *, derivative):
        row = entry(c, len(calls), derivative=derivative)
        calls.append(row)
        return row
    result = fit.run_leg(case_pack(), evaluate, persist=lambda n, a, r: saved.update({n: a}), solver_factory=counted_factory)
    assert result["f64_success"] and result["f32_success"] and result["cost"]["solve"]["requested"] == 2
    assert len(calls) == 4 and result["cost"]["fresh_jacobians_used"] == 2
    np.testing.assert_array_equal(saved["correction-01-problem"]["startup_q_jacobian"], calls[1]["startup_q_jacobian"])
    assert result["corrections"][0]["status"] == "actual_margin_rejection"


def test_two_failures_stop_without_f32_or_third_solver(monkeypatch):
    result, calls, problems, saved = generated_run(monkeypatch, [[0., 0., 0.], [0., 0., 0.]])
    assert result["status"] == "two_corrections_rejected" and result["run_valid"]
    assert len(calls) == 3 and len(problems) == 2 and "f32-coefficients" not in saved


@pytest.mark.parametrize("valid,direction", [(False, [0., 0., 0.]), (True, [.002, -1e-12, 0.]), (True, [.050000000001, 0., 0.])])
def test_uncertified_or_strict_bounds_reject_before_actual_evaluation(monkeypatch, valid, direction):
    result, calls, _, _ = generated_run(monkeypatch, [direction], valid=valid)
    assert result["run_valid"] and not result["f64_success"] and len(calls) == 1
    assert not result["corrections"][0]["actual_evaluated"]


def test_f32_trust_failure_saves_raw_before_evaluation(monkeypatch):
    def mutate(row, derivative):
        if row["evaluation"]:
            row["qerr"][:] = 0
    result, calls, _, saved = generated_run(monkeypatch, [[-.05, 0., 0.]], mutate=mutate)
    assert result["f64_success"] and not result["f32_success"] and len(calls) == 2
    assert result["status"] == "f32_coefficient_or_trust_rejection_before_evaluation"
    assert saved["f32-coefficients"][0]["coefficients"][0, 0] < .95


def test_f32_actual_startup_failure_is_a_separate_scientific_reject(monkeypatch):
    def mutate(row, derivative):
        if not derivative:
            row["qerr"][0, 0] = .036*.9999+1e-12
    result, calls, _, saved = generated_run(monkeypatch, [[.002, 0., 0.]], mutate=mutate)
    assert result["f64_success"] and not result["f32_success"] and len(calls) == 3
    assert result["status"] == "f32_startup_rejection" and "f32-actual" in saved


@pytest.mark.parametrize("key", ["neutral_startup_q", "neutral_startup_ltm", "neutral_q_jacobian", "neutral_ltm_jacobian"])
def test_reproduction_failure_stops_before_solver(monkeypatch, key):
    case = case_pack(); case[key].flat[0] += 1e-6
    with pytest.raises(fit.restoration.ReproductionError) as caught:
        fit.run_leg(case, lambda c, derivative: entry(c, derivative=derivative), persist=lambda *_: None,
                    solver_factory=lambda *_: pytest.fail("no solver after reproduction failure"))
    assert caught.value.margin_leg_result["cost"]["solve"]["requested"] == 0


def test_final_storage_failure_preserves_primary_and_completed_cost(monkeypatch):
    original = OSError("final result")
    def store(name, *_):
        if name == "leg-result":
            raise original
    with pytest.raises(OSError) as caught:
        generated_run(monkeypatch, [[.002, 0., 0.]], store=store)
    assert caught.value is original and original.abort_run
    assert original.margin_leg_result["cost"]["solve"]["requested"] == 1
    assert original.margin_leg_result["cost"]["evaluation"]["completed"] == 3
    assert not original.margin_leg_result["run_valid"]


@pytest.mark.parametrize("complete", [False, True])
def test_deadline_complete_partial_and_secondary_failure_preserve_identity(complete):
    original = fit.base.BudgetExceeded("original"); original.add_note("first")
    saved = {}
    def evaluate(c, *, derivative):
        if complete:
            original.completed_margin_entry = entry(c, derivative=derivative)
        raise original
    def persist(name, arrays, receipt):
        saved[name] = arrays
        if name == "leg-result":
            raise OSError("secondary storage")
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.run_leg(case_pack(), evaluate, persist=persist)
    assert caught.value is original and "first" in original.__notes__
    assert any("secondary storage" in s for s in original.__notes__)
    assert original.margin_leg_result["cost"]["evaluation"]["completed"] == int(complete)
    assert ("neutral" in saved) == complete


def test_partial_raw_solve_is_preserved_when_certificate_fails(monkeypatch):
    error = ArithmeticError("certificate")
    raw = SimpleNamespace(x=[.002, 0., 0.], s=np.zeros(2386), z=np.zeros(2386), status="Solved", iterations=9)
    solver = SimpleNamespace(solve=lambda: raw)
    def fail(*_):
        raise error
    monkeypatch.setattr(fit.restoration, "restoration_certificate", fail)
    saved = {}
    with pytest.raises(ArithmeticError) as caught:
        fit.run_leg(case_pack(), lambda c, derivative: entry(c, derivative=derivative),
                    persist=lambda n, a, r: saved.update({n: a}), solver_factory=lambda *_: solver)
    assert caught.value is error and error.margin_leg_result["cost"]["iterations"] == 9
    np.testing.assert_array_equal(saved["correction-00-solve"]["correction"], raw.x)


@pytest.mark.parametrize("kind", ["extra", "dtype", "shape", "nan", "quota", "ltm_quota", "neutral"])
def test_invalid_case_schema(kind):
    case = case_pack()
    if kind == "extra":
        case["teacher"] = np.zeros(1)
    elif kind == "dtype":
        case["target_q"] = case["target_q"].astype(np.float32)
    elif kind == "shape":
        case["neutral_q_jacobian"] = np.zeros((99, 7, 2))
    elif kind == "nan":
        case["target_q"][0, 0] = np.nan
    elif kind == "quota":
        case["q_sse_quota"] = np.array(0.)
    elif kind == "ltm_quota":
        case["ltm_sse_quota"] = np.array(.009)
    else:
        case["neutral_coefficients"][0, 0] = 1.01
    with pytest.raises(ValueError):
        fit.validate_case(case)
