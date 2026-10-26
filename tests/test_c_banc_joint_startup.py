"""E282 generated n<=6 equations, certificate, strict bounds and failure receipts."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from tests.test_c_banc_feasibility_restoration import ACTUAL_COST
from tests.test_c_banc_feasibility_restoration import (
    counted_factory as recorded_factory,
)
from tests.test_c_banc_startup_margin import case_pack, entry
from tools import banc_m3_joint_startup as fit


def counted_factory(*args):
    assert args[0].shape[0] <= 6 and len(ACTUAL_COST) < 60
    return recorded_factory(*args)


@pytest.fixture(scope="module", autouse=True)
def tiny_cost():
    began = len(ACTUAL_COST)
    yield
    print("generated_e282_actual_tiny_cost", {k: sum(r[k] for r in ACTUAL_COST[began:]) for k in
        ("constructor_requested", "constructor_completed", "constructor_wall_s", "solve_requested",
         "solve_completed", "solve_wall_s", "iterations")})


def linear_problem(*, continuous=False, blocks=1, zero_j=False):
    rows = [entry([1., 0., 0.]) for _ in range(blocks)]
    for row in rows:
        row["qerr"][:] = .032 if continuous else 0
        row["startup_q_jacobian"][:] = 0
        if continuous:
            row["startup_q_jacobian"][:, :, 0] = 1
        else:
            row["qerr"][0, 0] = .04
            row["startup_q_jacobian"][0, 0, 0] = 1
        if zero_j: row["startup_q_jacobian"][:] = 0
    return fit.joint_problem(rows, [fit.base.neutral(1) for _ in rows])


@pytest.mark.parametrize("continuous,blocks", [(False, 1), (True, 2)])
def test_tiny_solver_independent_analytic_primal_dual(continuous, blocks):
    p = linear_problem(continuous=continuous, blocks=blocks)
    expected = p["radii"][0]/np.sqrt(693*blocks)-.032 if continuous else p["peaks"][0]-.04
    optimum = np.tile([expected, 0., 0.], blocks)
    slack, dual = p["b"]-p["A"]@optimum, np.zeros_like(p["b"])
    if continuous:
        start = p["cone_dimensions"][0]
        multiplier = -expected*np.sqrt(blocks/693)
        dual[start], dual[start+1:start+1+693*blocks] = multiplier, -multiplier/np.sqrt(693*blocks)
    else:
        dual[8] = -expected
    analytic = fit.joint_certificate(p, optimum, slack, dual, "AlmostSolved")
    assert analytic["valid"] and abs(analytic["receipt"]["gap"]) < 1e-12
    # Independent original equations, not solver-provided slack or objective.
    np.testing.assert_allclose(optimum+p["A"].T@dual, 0., atol=1e-14)
    assert abs(.5*optimum@optimum+.5*np.linalg.norm(p["A"].T@dual)**2+p["b"]@dual) < 1e-12
    solved = fit.solve_joint(p, 4., solver_factory=counted_factory)
    assert solved["valid"] and solved["receipt"]["solve"]["completed"] == 1
    gap = solved["receipt"]["certificate_receipt"]["gap"]
    assert .5*np.sum((solved["correction"]-optimum)**2) <= gap+1e-10
    np.testing.assert_allclose(solved["correction"][::3], expected, atol=1e-6)
    assert solved["receipt"]["settings"]["time_limit"] == 4.


def test_tiny_linear_infeasible_is_not_an_execution_failure():
    solved = fit.solve_joint(linear_problem(zero_j=True), 4., solver_factory=counted_factory)
    assert not solved["valid"] and solved["receipt"]["status"] == "PrimalInfeasible"
    assert solved["receipt"]["iterations_known"] and solved["receipt"]["solve"]["completed"] == 1


def test_sparse_global_q_one_soc_ltm_per_block_signs_and_serialization():
    p = linear_problem(continuous=True, blocks=2)
    assert p["cone_dimensions"][1:] == (1387, 100, 100)
    assert sparse.isspmatrix_csc(p["q_jacobian"]) and p["q_jacobian"][:693, 3:].nnz == 0
    z = np.array([-.01, 0., 0., -.011, 0., 0.])
    slack, begin = p["b"]-p["A"]@z, p["cone_dimensions"][0]
    np.testing.assert_array_equal(slack[begin+1:begin+1387], p["qerr"]+p["q_jacobian"]@z)
    for index in range(2):
        start = begin+1387+100*index
        assert slack[start] == p["radii"][1]
        np.testing.assert_array_equal(slack[start+1:start+100], p["lerr"][99*index:99*(index+1)])
    arrays = fit.problem_arrays(p)
    restored = sparse.csc_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=arrays["A_shape"])
    assert (restored != p["A"]).nnz == 0


@pytest.mark.parametrize("key", ["A", "P", "b", "bounds", "original_coefficient_bounds", "coefficient_bounds", "peaks", "radii", "q", "radius", "cone", "offblock"])
def test_original_problem_mutations_reject_before_constructor(key):
    p = linear_problem(blocks=2)
    if key in ("P", "A"):
        p[key].data[0] += 1e-3
    elif key == "radius": p[key] = .049
    elif key == "cone": p["cone_dimensions"] = (1,)
    elif key == "offblock":
        p["q_jacobian"] = p["q_jacobian"].tolil(); p["q_jacobian"][0, 3] = 1
        p["q_jacobian"] = p["q_jacobian"].tocsc()
    else: p[key].flat[0] += 1e-3
    with pytest.raises(ValueError):
        fit.solve_joint(p, 4., solver_factory=lambda *_: pytest.fail("no constructor"))


@pytest.mark.parametrize("kind", ["status", "stationarity", "dual", "nan", "shape", "primal", "raw_difference"])
def test_certificate_uses_original_primal_and_dual_not_status_or_raw_slack(kind):
    p = linear_problem(); optimum = np.array([p["peaks"][0]-.04, 0., 0.])
    s, y, status = p["b"]-p["A"]@optimum, np.zeros_like(p["b"]), "Solved"
    y[8] = -optimum[0]
    if kind == "status": status = "MaxTime"
    elif kind == "stationarity": y[8] = 0
    elif kind == "dual": y[0] = -1e-12
    elif kind == "nan": s[0] = np.nan
    elif kind == "shape": s = s[:-1]
    elif kind == "primal": optimum[0] += 1e-6
    else: s[:] += .01
    cert = fit.joint_certificate(p, optimum, s, y, status)
    assert cert["valid"] == (kind == "raw_difference")
    if kind == "raw_difference": assert cert["receipt"]["raw_affine_infinity_norm"] > .009


def test_global_actual_quota_allows_an_individual_legacy_quota_miss():
    rows = [entry([1., 0., 0.]) for _ in range(6)]
    for row in rows: row["qerr"][:] = 0
    rows[0]["qerr"][:] = .02
    judge = fit.strict_joint(rows, [fit.base.neutral(1)]*6)
    assert judge["pass"] and judge["per_leg"][0]["q_sse"] > judge["q_sse_limit"]/6
    assert judge["q_sse"] == fit.margin.sum_square(np.concatenate([r["qerr"].ravel() for r in rows]))


@pytest.mark.parametrize("kind", ["q_peak", "global_q", "ltm_peak", "per_leg_ltm", "coefficient", "trust"])
def test_actual_strict_limits_do_not_inherit_certificate_tolerance(kind):
    rows = [entry([1.01, 0., 0.]) for _ in range(6)]
    for row in rows: row["qerr"][:] = 0
    if kind == "q_peak": rows[0]["qerr"][0, 0] = .036*.9999+1e-12
    elif kind == "global_q":
        for row in rows: row["qerr"][:] = .0162*.9999+1e-12
    elif kind == "ltm_peak": rows[0]["lerr"][0, 0] = .025*.9999+1e-12
    elif kind == "per_leg_ltm": rows[0]["lerr"][:] = .009*.9999+1e-12
    elif kind == "coefficient": rows[0]["coefficients"][0, 1] = -1e-12
    else: rows[0]["coefficients"][0, 0] = 1.05+1e-12
    assert not fit.strict_joint(rows, [fit.base.neutral(1)]*6)["pass"]


def stub(directions, valid=True):
    calls = []
    def solve(p, *args, **kwargs):
        z = np.array(directions[len(calls)], np.float64); calls.append(p)
        receipt = {k: {**fit.metering.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, generated_stub=True)
        s = p["b"]-p["A"]@z
        return {"valid": valid, "receipt": receipt, "correction": z, "solver_s": s,
                "solver_y": np.zeros_like(s), "primal_slack": s, "residuals": None}
    return solve, calls


def generated_run(monkeypatch, directions, *, mutate=None, valid=True, store=None):
    solve, problems = stub(directions, valid)
    monkeypatch.setattr(fit, "solve_joint", solve)
    calls, saved = [], {}
    def evaluate(cs, *, derivative, label):
        rows = [entry(c, len(calls), derivative=derivative) for c in cs]
        if mutate: mutate(rows, derivative, label)
        for row in rows:
            row["startup_q"][1:100], row["startup_ltm"][1:100] = row["qerr"], row["lerr"]
        calls.append((derivative, rows, label)); return rows
    def persist(name, arrays, receipt):
        saved[name] = arrays, copy.deepcopy(receipt)
        if store: store(name, arrays, receipt)
    result = fit.run_joint([case_pack()], evaluate, persist=persist)
    return result, calls, problems, saved


def test_fresh_j_actual_tiny_two_joint_corrections_then_single_f32():
    calls, saved = [], {}
    def evaluate(cs, *, derivative, label):
        rows = [entry(c, len(calls), derivative=derivative) for c in cs]; calls.append((derivative, rows))
        return rows
    result = fit.run_joint([case_pack()], evaluate, persist=lambda n, a, r: saved.update({n: a}), solver_factory=counted_factory)
    assert result["initialization_supported"] and result["cost"]["solve"]["requested"] == 2
    assert [c[0] for c in calls] == [True, True, True, False]
    qj = sparse.csc_matrix((saved["joint-01-problem"]["q_jacobian_data"], saved["joint-01-problem"]["q_jacobian_indices"],
        saved["joint-01-problem"]["q_jacobian_indptr"]), shape=(693, 3))
    np.testing.assert_array_equal(qj.toarray(), calls[1][1][0]["startup_q_jacobian"].reshape(693, 3))


def test_four_corrections_no_fifth_or_f32(monkeypatch):
    result, calls, problems, saved = generated_run(monkeypatch, [[0., 0., 0.]]*4)
    assert result["run_valid"] and result["status"] == "four_joint_corrections_rejected"
    assert len(problems) == 4 and len(calls) == 5 and "f32-coefficients" not in saved


@pytest.mark.parametrize("valid,direction", [(False, [0., 0., 0.]), (True, [.002, -1e-12, 0.]), (True, [.05+1e-12, 0., 0.])])
def test_uncertified_and_exact_coefficient_trust_correction_precheck(monkeypatch, valid, direction):
    result, calls, _, _ = generated_run(monkeypatch, [direction], valid=valid)
    assert result["run_valid"] and len(calls) == 1 and not result["corrections"][0]["actual_evaluated"]


def test_cumulative_trust_remains_original_neutral(monkeypatch):
    result, calls, problems, _ = generated_run(monkeypatch, [[.03, 0., 0.], [.03, 0., 0.]],
        mutate=lambda rows, derivative, label: rows[0]["qerr"].fill(.04) if label != "neutral" else None)
    assert len(calls) == 2 and len(problems) == 2
    assert result["status"] == "strict_coefficient_trust_or_correction_rejection"
    assert problems[1]["bounds"][0, 1] == np.float64(np.float32(1.05))-1.03


def test_f32_actual_startup_rejection_still_required(monkeypatch):
    def mutate(rows, derivative, label):
        if not derivative: rows[0]["qerr"][0, 0] = .036*.9999+1e-12
    result, calls, _, saved = generated_run(monkeypatch, [[.002, 0., 0.]], mutate=mutate)
    assert result["f64_success"] and not result["f32_success"] and result["run_valid"]
    assert len(calls) == 3
    assert saved["f32-coefficients"][0]["coefficients32_leg_major"].dtype == np.float32
    assert result["status"] == "f32_joint_startup_rejection"


def test_nearest_inward_f32_bounds_lower_main_upper_fast_upper_and_zero():
    p = linear_problem(); original, inward = p["original_coefficient_bounds"], p["coefficient_bounds"]
    expected = np.array([[float(np.nextafter(np.float32(.95), np.float32(np.inf))), float(np.float32(1.05))],
                         [0., float(np.nextafter(np.float32(.05), np.float32(-np.inf)))],
                         [0., float(np.nextafter(np.float32(.05), np.float32(-np.inf)))]])
    np.testing.assert_array_equal(original, [[.95, 1.05], [0., .05], [0., .05]])
    np.testing.assert_array_equal(inward, expected)
    # Main upper already rounds inward and zero is exact: no gratuitous ULP.
    assert inward[0, 1] == float(np.float32(1.05)) and inward[1, 0] == 0.
    np.testing.assert_array_equal(p["bounds"], inward-p["coefficients"].ravel()[:, None])
    np.testing.assert_array_equal(p["b"][:3], p["bounds"][:, 1])
    np.testing.assert_array_equal(p["b"][3:6], -p["bounds"][:, 0])
    assert np.all(inward[:, 0] >= original[:, 0]) and np.all(inward[:, 1] <= original[:, 1])


def test_current_intersection_ceil_already_inward_gets_no_extra_ulp():
    c = np.array([[1.04, .049, .02]])
    original, inward = fit.coefficient_interval(c, fit.base.neutral(1))
    assert original[0, 0] == 1.04-.05
    assert np.float64(np.float32(original[0, 0])) > original[0, 0]
    assert inward[0, 0] == np.float64(np.float32(original[0, 0]))
    for value in inward.ravel(): assert np.float64(np.float32(value)) == value
    assert np.all(np.nextafter(inward[:, 0].astype(np.float32), np.float32(-np.inf)).astype(float) < original[:, 0])
    assert np.all(np.nextafter(inward[:, 1].astype(np.float32), np.float32(np.inf)).astype(float) > original[:, 1])


@pytest.mark.parametrize("endpoint", ["lower", "main_upper", "fast_upper"])
def test_inward_endpoint_candidates_keep_actual_f32_replay_without_repair(monkeypatch, endpoint):
    _, interval = fit.coefficient_interval(fit.base.neutral(1), fit.base.neutral(1))
    d = [interval[0, 0]-1, 0., 0.] if endpoint == "lower" else [interval[0, 1]-1, 0., 0.]
    if endpoint == "fast_upper": d = [.002, interval[1, 1], 0.]
    def mutate(rows, derivative, label):
        if label != "neutral": rows[0]["qerr"][:] = 0
    result, calls, _, saved = generated_run(monkeypatch, [d], mutate=mutate)
    assert result["initialization_supported"] and len(calls) == 3 and not calls[-1][0]
    np.testing.assert_array_equal(saved["joint-00-solve"][0]["correction"], d)
    assert saved["joint-00-coefficients"][1]["inward_coefficient_interval"]
    raw64 = saved["joint-00-coefficients"][0]["coefficients_leg_major"]
    np.testing.assert_array_equal(saved["f32-coefficients"][0]["coefficients_leg_major"], raw64.astype(np.float32).astype(np.float64))


def test_certified_tolerance_does_not_relax_inward_candidate_interval(monkeypatch):
    p = linear_problem()
    optimum = p["coefficient_bounds"][0, 1]-1
    current = entry([1., 0., 0.]); current["qerr"][:] = 0
    current["qerr"][0, 0] = p["peaks"][0]+optimum
    current["startup_q_jacobian"][:] = 0; current["startup_q_jacobian"][0, 0, 0] = -1
    p = fit.joint_problem([current], [fit.base.neutral(1)])
    z = np.array([optimum+1e-10, 0., 0.]); s = p["b"]-p["A"]@z; y = np.zeros_like(s); y[8] = optimum
    cert = fit.joint_certificate(p, z, s, y, "Solved")
    assert cert["valid"] and cert["residuals"]["inward_coefficient_interval"].min() < 0
    assert fit.margin.coefficient_checks(np.array([[1+z[0], 0., 0.]]), fit.base.neutral(1))["pass"]
    # A raw correction just outside the inward fast upper still satisfies the
    # original box/trust. A generated solver stub isolates the strict postgate.
    d = [.002, .05-1e-10, 0.]
    result, calls, _, saved = generated_run(monkeypatch, [d])
    assert result["run_valid"] and result["status"] == "strict_inward_coefficient_interval_rejection"
    assert len(calls) == 1 and result["cost"]["solve"]["requested"] == 1
    assert not result["corrections"][0]["actual_evaluated"] and "f32-coefficients" not in saved
    np.testing.assert_array_equal(saved["joint-00-solve"][0]["correction"], d)
    receipt = saved["joint-00-coefficients"][1]
    assert receipt["original_coefficient_checks_pass"] and not receipt["inward_coefficient_interval"] and not receipt["pass"]


@pytest.mark.parametrize("key", ["neutral_startup_q", "neutral_startup_ltm", "neutral_q_jacobian", "neutral_ltm_jacobian"])
def test_reproduction_failure_has_no_solver(key):
    case = case_pack(); case[key].flat[0] += 1e-6
    with pytest.raises(fit.metering.ReproductionError) as caught:
        fit.run_joint([case], lambda cs, derivative, label: [entry(c, derivative=derivative) for c in cs], persist=lambda *_: None,
                      solver_factory=lambda *_: pytest.fail("no solver"))
    assert caught.value.joint_result["cost"]["solve"]["requested"] == 0


def test_terminal_storage_failure_attaches_exact_completed_cost(monkeypatch):
    original = OSError("joint-result")
    def store(name, *_):
        if name == "joint-result": raise original
    with pytest.raises(OSError) as caught: generated_run(monkeypatch, [[.002, 0., 0.]], store=store)
    assert caught.value is original and original.abort_run
    assert original.joint_result["cost"]["solve"]["completed"] == 1
    assert original.joint_result["cost"]["round"]["completed"] == 3 and not original.joint_result["run_valid"]


def test_partial_round_deadline_identity_and_secondary_storage_failure():
    original = fit.base.BudgetExceeded("original"); original.add_note("primary")
    def evaluate(cs, **kwargs):
        original.partial_joint_round = [entry(cs[0])]; raise original
    def persist(*_): raise OSError("secondary storage")
    with pytest.raises(fit.base.BudgetExceeded) as caught: fit.run_joint([case_pack(), case_pack()], evaluate, persist=persist)
    assert caught.value is original and original.joint_result["status"] == "incomplete"
    assert original.joint_result["partial_round"]["completed_blocks"] == 1
    assert original.joint_result["cost"]["round"]["failed"] == 1
    assert "primary" in original.__notes__ and any("secondary storage" in n for n in original.__notes__)


@pytest.mark.parametrize("phase", ["constructor", "solve", "certificate", "solve_clock"])
def test_phase_failure_same_exception_partial_cost_and_raw(monkeypatch, phase):
    p = linear_problem(); original = OSError(phase)
    raw = SimpleNamespace(x=np.zeros(3), s=np.zeros_like(p["b"]), z=np.zeros_like(p["b"]), status="Solved", iterations=7)
    def fail(*_): raise original
    solver = SimpleNamespace(solve=fail if phase == "solve" else lambda: raw)
    factory = fail if phase == "constructor" else lambda *_: solver
    if phase == "certificate": monkeypatch.setattr(fit, "joint_certificate", fail)
    times = [0]
    def clock():
        times[0] += 1
        if phase == "solve_clock" and times[0] == 4: raise original
        return times[0]
    with pytest.raises(OSError) as caught: fit.solve_joint(p, 4., solver_factory=factory, wall_clock=clock)
    assert caught.value is original
    receipt = original.joint_solve_receipt
    assert receipt["constructor"]["requested"] == 1
    assert receipt["solve"]["requested"] == int(phase != "constructor")
    assert receipt["iterations_known"] == (phase in ("certificate", "solve_clock"))
    if phase in ("certificate", "solve_clock"): assert original.partial_joint_solution is raw and receipt["iterations"] == 7
    if phase == "solve_clock": assert not receipt["solve"]["wall_known"] and receipt["solve"]["completed"] == 1
