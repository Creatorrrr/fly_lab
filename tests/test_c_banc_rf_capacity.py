"""Generated E284 n<=6 math/loop/roster/provenance/lifetime tests; original/CUDA0."""
import copy
import json
import time
import weakref
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tests import test_c_banc_startup_continuation as old_tests
from tests import test_c_banc_startup_margin_runner as startup_tests
from tools import banc_m3_rf_capacity as fit
from tools import verify_c_banc_m3_rf_capacity as run

ACTUAL_COST = []


@pytest.fixture(scope="module", autouse=True)
def generated_cost():
    started = time.perf_counter()
    yield
    assert time.perf_counter()-started < 120 and len(ACTUAL_COST) <= 20
    print("E284_TINY_ACTUAL_COST", {k: sum(r[k] for r in ACTUAL_COST) for k in
        ("constructor_requested", "constructor_completed", "constructor_failed", "constructor_wall_s",
         "solve_requested", "solve_completed", "solve_failed", "solve_wall_s", "iterations")})


def counted_factory(*args):
    assert args[0].shape[0] <= 6 and len(ACTUAL_COST) < 20
    row = {"constructor_requested": 1, "constructor_completed": 0, "constructor_failed": 0, "constructor_wall_s": 0.,
           "solve_requested": 0, "solve_completed": 0, "solve_failed": 0, "solve_wall_s": 0., "iterations": 0}
    ACTUAL_COST.append(row); started = time.perf_counter()
    try:
        solver = clarabel.DefaultSolver(*args); row["constructor_completed"] = 1
    except Exception:
        row["constructor_failed"] = 1; raise
    finally: row["constructor_wall_s"] = time.perf_counter()-started
    def solve():
        row["solve_requested"] += 1; started = time.perf_counter()
        try:
            raw = solver.solve(); row["solve_completed"] += 1; row["iterations"] += raw.iterations; return raw
        except Exception:
            row["solve_failed"] += 1; raise
        finally: row["solve_wall_s"] += time.perf_counter()-started
    return SimpleNamespace(solve=solve)


def actual(c, *, derivative=True, nonlinear=False, qvalue=None):
    c = np.asarray(c, np.float64).reshape(-1, 3).copy(); assert c.size <= 6
    x = c[0, 0]-1
    q = .04-.01*x+(.001*x*x if nonlinear else 0) if qvalue is None else qvalue
    row = {"coefficients": c, "startup_q": np.zeros((101, 7)), "startup_ltm": np.zeros((101, 1)),
           "qerr": np.zeros((99, 7)), "lerr": np.zeros((99, 1))}
    row["qerr"][0, 0] = q; row["startup_q"][1:100] = row["qerr"]
    if derivative:
        row["startup_q_jacobian"] = np.zeros((99, 7, c.size)); row["startup_q_jacobian"][0, 0, 0] = -.01+(.002*x if nonlinear else 0)
        row["startup_ltm_jacobian"] = np.zeros((99, 1, c.size))
    return row


def case_pack(nonlinear=False):
    row = actual(fit.base.neutral(1), nonlinear=nonlinear)
    return {"neutral_coefficients": row["coefficients"], "neutral_startup_q": row["startup_q"], "neutral_startup_ltm": row["startup_ltm"],
        "neutral_q_jacobian": row["startup_q_jacobian"], "neutral_ltm_jacobian": row["startup_ltm_jacobian"],
        "target_q": np.zeros((99, 7)), "target_ltm": np.zeros((99, 1)), "q_sse_quota": np.array(.1),
        "ltm_sse_quota": np.array(99*(.009*.9999)**2)}


@pytest.mark.parametrize("kind", ["peak", "continuous_q_peak", "ltm_soc"])
def test_actual_tiny_independent_hard_domain_analytic_certificate(kind):
    entry = actual(fit.base.neutral(1)); entry["startup_q_jacobian"][:] = 0
    if kind == "peak": entry["startup_q_jacobian"][0, 0, 0] = -.01
    elif kind == "continuous_q_peak":
        entry["qerr"][:] = .045; entry["startup_q_jacobian"][:, :, 0] = -.01
    else:
        entry["qerr"][:] = 0; entry["lerr"][:] = .012; entry["startup_ltm_jacobian"][:, :, 0] = -.01
    p = fit.capacity_problem(entry); optimum = np.zeros(3); dual = np.zeros_like(p["b"])
    if kind == "peak": optimum[0] = (.04-p["peaks"][0])/.01; dual[8] = optimum[0]/.01
    elif kind == "continuous_q_peak":
        optimum[0] = (.045-p["peaks"][0])/.01
        dual[8:701] = optimum[0]/(.01*693)
    else:
        size, offset, index, value = 99, p["cone_dimensions"][0]+694, 1, .012
        optimum[0] = (value-p["radii"][index]/np.sqrt(size))/.01
        multiplier = optimum[0]/(.01*np.sqrt(size)); dual[offset] = multiplier; dual[offset+1:offset+1+size] = -multiplier/np.sqrt(size)
    exact = p["b"]-p["A"]@optimum
    analytic = fit.capacity_certificate(p, optimum, exact, dual, "AlmostSolved")
    assert analytic["valid"] and abs(analytic["receipt"]["gap"]) < 1e-10
    np.testing.assert_allclose(optimum+p["A"].T@dual, 0, atol=1e-13)
    assert abs(.5*optimum@optimum+.5*np.linalg.norm(p["A"].T@dual)**2+p["b"]@dual) < 1e-10
    solved = fit.solve_capacity(p, 4., solver_factory=counted_factory)
    assert solved["valid"] and solved["correction"][0] > .05
    assert .5*np.sum((solved["correction"]-optimum)**2) <= solved["receipt"]["certificate_receipt"]["gap"]+1e-10
    assert fit.validate_problem(p) == (1592, 694, 100)


def test_actual_tiny_unchangeable_violation_is_bounded_uncertified_result():
    row = actual(fit.base.neutral(1)); row["startup_q_jacobian"][:] = 0
    result = fit.solve_capacity(fit.capacity_problem(row), 4., solver_factory=counted_factory)
    assert not result["valid"] and result["receipt"]["status"] == "PrimalInfeasible"


def test_sparse_rows_hard_domain_without_neutral_or_step_box():
    row = actual([[2., .25, .5], [1., .1, .2]])
    p = fit.capacity_problem(row); assert p["P"].shape == (6, 6)
    np.testing.assert_array_equal(p["bounds"], np.tile([0., 8.], (6, 1))-row["coefficients"].ravel()[:, None])
    assert not ({"radius", "correction_radius", "origins", "trust"} & set(p))
    d = np.array([.6, .1, 0., -.2, 0., .3]); slack = p["b"]-p["A"]@d; start = p["cone_dimensions"][0]
    np.testing.assert_array_equal(slack[start+1:start+694], p["qerr"]+p["q_jacobian"]@d)
    np.testing.assert_array_equal(slack[start+695:start+794], p["lerr"]+p["ltm_jacobian"]@d)
    arrays = fit.problem_arrays(p)
    recovered = sparse.csc_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=arrays["A_shape"])
    assert (recovered != p["A"]).nnz == 0


@pytest.mark.parametrize("key", ["P", "A", "b", "q", "bounds", "coefficient_bounds", "peaks", "radii", "trust", "cone"])
def test_original_metadata_mutation_rejects_without_solver(key):
    p = fit.capacity_problem(actual(fit.base.neutral(1)))
    if key in ("P", "A"): p[key].data[0] += 1e-4
    elif key == "trust": p["radius"] = .05
    elif key == "cone": p["cone_dimensions"] = (1,)
    else: p[key].flat[0] += 1e-4
    with pytest.raises(ValueError): fit.solve_capacity(p, 4., solver_factory=lambda *_: pytest.fail("no solver"))


@pytest.mark.parametrize("kind", ["status", "dual", "stationarity", "primal", "nan", "shape", "raw_difference"])
def test_independent_original_certificate_rejects_bad_d_y_or_raw_shapes(kind):
    p = fit.capacity_problem(actual(fit.base.neutral(1))); d = np.array([(.04-p["peaks"][0])/.01, 0., 0.])
    s, y, status = p["b"]-p["A"]@d, np.zeros_like(p["b"]), "Solved"; y[8] = d[0]/.01
    if kind == "status": status = "MaxTime"
    elif kind == "dual": y[0] = -1e-12
    elif kind == "stationarity": y[8] = 0
    elif kind == "primal": d[0] -= .001
    elif kind == "nan": s[0] = np.inf
    elif kind == "shape": y = y[:-1]
    else: s[:] += .001
    cert = fit.capacity_certificate(p, d, s, y, status)
    assert cert["valid"] == (kind == "raw_difference")


@pytest.mark.parametrize("kind", ["q_peak", "q_sse", "ltm_peak", "ltm_sse", "negative", "upper", "triple_low", "triple_high"])
def test_actual_strict_domain_and_output_have_zero_tolerance(kind):
    row = actual([2., .2, .3], qvalue=0)
    if kind == "q_peak": row["qerr"][0, 0] = .036*.9999+1e-12
    elif kind == "q_sse": row["qerr"][:] = np.sqrt(fit.CONTRACT["q_sse_limit"]/693)+1e-12
    elif kind == "ltm_peak": row["lerr"][0, 0] = .025*.9999+1e-12
    elif kind == "ltm_sse": row["lerr"][:] = .009*.9999+1e-12
    elif kind == "negative": row["coefficients"][0, 1] = -1e-12
    elif kind == "upper": row["coefficients"][0, 0] = 8+1e-12
    elif kind == "triple_low": row["coefficients"][0] = [.125*(1+1e-6)-1e-12, 0., 0.]
    else: row["coefficients"][0] = [8*(1-1e-6)+1e-12, 0., 0.]
    assert not fit.strict_rf(row)["pass"]
    valid = fit.strict_rf(actual([3., .2, .3], qvalue=0))
    assert valid["pass"] and not valid["global_norm_supported"]


def stub(directions, valid=True):
    calls = []
    def solve(p, *args, **kwargs):
        d = np.array(directions[len(calls)], np.float64); calls.append(p)
        receipt = {k: {**fit.metering.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, generated_stub=True)
        s = p["b"]-p["A"]@d
        return {"valid": valid, "receipt": receipt, "correction": d, "solver_s": s, "solver_y": np.zeros_like(s), "primal_slack": s, "residuals": None}
    return solve, calls


def simulate(directions=None, *, nonlinear=False, values=None, mutate=None, store=None, valid=True):
    calls, saved, problems = [], {}, []
    def evaluate(c, *, derivative, label):
        row = actual(c, derivative=derivative, nonlinear=nonlinear, qvalue=None if values is None else values[len(calls)])
        if mutate: mutate(row, derivative, label)
        calls.append((label, derivative, row)); return row
    def persist(n, a, r):
        saved[n] = copy.deepcopy((a, r))
        if store: store(n, a, r)
    kwargs = {"solver_factory": counted_factory}
    if directions is not None: kwargs["solve"], problems = stub(directions, valid)
    result = fit.run_capacity(case_pack(nonlinear), evaluate, persist=persist, **kwargs)
    return result, calls, saved, problems


def test_actual_tiny_nonlinear_fresh_j_and_one_f32_replay():
    result, calls, saved, _ = simulate(nonlinear=True)
    assert result["rf_capacity_supported"] and result["f64_success"] and result["f32_success"]
    assert not result["initialization_supported"] and not result["global_norm_supported"]
    assert result["cost"]["solve"]["completed"] == 2 and len(calls) == 4
    assert [c[1] for c in calls] == [True]*3+[False]
    for index in range(2):
        arrays = saved[f"correction-{index:02d}-problem"][0]
        jq = sparse.csc_matrix((arrays["q_jacobian_data"], arrays["q_jacobian_indices"], arrays["q_jacobian_indptr"]), shape=arrays["q_jacobian_shape"])
        np.testing.assert_array_equal(jq.toarray(), calls[index][2]["startup_q_jacobian"].reshape(693, 3))
    assert saved["correction-00-solve"][0]["correction"][0] > .05
    np.testing.assert_array_equal(calls[-1][2]["coefficients"], calls[-2][2]["coefficients"].astype(np.float32).astype(np.float64))


def test_four_corrections_no_fifth_and_no_line_search():
    result, calls, saved, problems = simulate([[.2, 0., 0.]]*4, values=[.04]*5)
    assert result["run_valid"] and result["status"] == "four_rf_corrections_rejected" and len(calls) == 5 and len(problems) == 4
    assert "f32-coefficients" not in saved and all(c[1] for c in calls)


@pytest.mark.parametrize("direction,valid,status", [([.5,-1e-12,0.],True,"strict_hard_coefficient_rejection"),
    ([7.,0.,0.],True,"strict_hard_coefficient_rejection"), ([0.,0.,0.],False,"uncertified_rf_correction")])
def test_raw_correction_is_never_clipped_or_retried(direction, valid, status):
    result, calls, saved, _ = simulate([direction], valid=valid)
    assert result["run_valid"] and result["status"] == status and len(calls) == 1
    np.testing.assert_array_equal(saved["correction-00-solve"][0]["correction"], direction)
    assert not result["corrections"][0]["actual_evaluated"]


def test_f32_triple_failure_preserves_raw_and_stops_before_replay():
    direction = [.125*(1+1e-6)-1, 0., 0.]
    result, calls, saved, _ = simulate([direction], values=[.04, 0.])
    assert result["f64_success"] and not result["f32_success"] and len(calls) == 2
    assert result["status"] == "f32_hard_coefficient_rejection_before_evaluation"
    assert saved["f32-coefficients"][0]["coefficients32"].dtype == np.float32


def test_f32_actual_peak_fail_is_scientific_reject_not_invalid():
    result, calls, _, _ = simulate([[.5,0.,0.]], values=[.04,.035,.036*.9999+1e-12])
    assert result["run_valid"] and result["f64_success"] and not result["rf_capacity_supported"]
    assert result["status"] == "f32_rf_startup_rejection" and len(calls) == 3


@pytest.mark.parametrize("key", ["neutral_startup_q", "neutral_startup_ltm", "neutral_q_jacobian", "neutral_ltm_jacobian"])
def test_reproduction_fail_stops_all_solver_work(key):
    case = case_pack(); case[key].flat[0] += 1e-6
    with pytest.raises(fit.metering.ReproductionError) as caught:
        fit.run_capacity(case, lambda c, derivative, label: actual(c, derivative=derivative), persist=lambda *_: None,
                         solver_factory=lambda *_: pytest.fail("no solver"))
    assert caught.value.capacity_result["cost"]["solve"]["requested"] == 0


def test_target_relationship_and_primary_partial_deadline_storage_identity():
    original = fit.base.BudgetExceeded("original"); original.add_note("initial")
    def evaluate(c, **kwargs): original.completed_margin_entry = actual(c); raise original
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.run_capacity(case_pack(), evaluate, persist=lambda *_: (_ for _ in ()).throw(OSError("secondary storage")))
    assert caught.value is original and original.capacity_result["status"] == "incomplete"
    assert original.capacity_result["partial_evaluation"]["completed_response_available"]
    assert any("secondary storage" in n for n in original.__notes__)
    with pytest.raises(ValueError) as caught:
        simulate([], mutate=lambda row, *_: row["qerr"].fill(.2))
    assert caught.value.capacity_result["cost"]["solve"]["requested"] == 0


@pytest.mark.parametrize("phase", ["constructor", "solve", "certificate", "solve_clock"])
def test_phase_failure_preserves_partial_raw_requested_cost_and_exception(monkeypatch, phase):
    p = fit.capacity_problem(actual(fit.base.neutral(1))); original = OSError(phase)
    raw = SimpleNamespace(x=np.zeros(3), s=np.zeros_like(p["b"]), z=np.zeros_like(p["b"]), status="Solved", iterations=11)
    def fail(*_): raise original
    factory = fail if phase == "constructor" else lambda *_: SimpleNamespace(solve=fail if phase == "solve" else lambda: raw)
    if phase == "certificate": monkeypatch.setattr(fit, "capacity_certificate", fail)
    times = [0]
    def clock():
        times[0] += 1
        if phase == "solve_clock" and times[0] == 4: raise original
        return times[0]
    saved = {}
    with pytest.raises(OSError) as caught:
        fit.run_capacity(case_pack(), lambda c, derivative, label: actual(c, derivative=derivative),
            persist=lambda n, a, r: saved.update({n:a}), solver_factory=factory,
            solve=lambda p, t, **kwargs: fit.solve_capacity(p, t, solver_factory=factory, wall_clock=clock))
    assert caught.value is original and original.capacity_result["cost"]["constructor"]["requested"] == 1
    assert original.capacity_result["cost"]["solve"]["requested"] == int(phase != "constructor")
    if phase in ("certificate", "solve_clock"):
        assert original.partial_capacity_solution is raw and original.capacity_result["cost"]["iterations"] == 11
        np.testing.assert_array_equal(saved["correction-00-solve"]["correction"], raw.x)


def test_final_storage_failure_retains_cost_same_primary_and_no_support():
    original = OSError("final storage")
    def store(name, *args):
        if name == "capacity-result": raise original
    with pytest.raises(OSError) as caught: simulate([[.5,0.,0.]], values=[.04,.035,.035], store=store)
    assert caught.value is original and original.abort_run and not original.capacity_result["run_valid"]
    assert not original.capacity_result["rf_capacity_supported"]
    assert original.capacity_result["cost"]["evaluation"]["completed"] == 3 and original.capacity_result["cost"]["iterations"] == 2


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    old, original, save_old, tmp = old_tests.freeze_fixture.__wrapped__(tmp_path, monkeypatch)
    fp, fs, rp, rs = save_old(); owned = {}
    for name in run.SOURCES:
        path = tmp/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("generated E284 source"); owned[name] = fit.base.sha256(path)
    store = old_tests.prior_tests.old_tests.old_tests.store_json
    audit = {"schema":"flylab.m3.e283.root-original-audit.v1", "verdict":"valid_diagnostic_initialization_rejected",
        "reproduction_exact":True,"sparse_equations_exact":True,"independent_primal_dual_checks_pass":True,
        "terminal_status":"stage_0_two_corrections_rejected","stage1_reached":False,"f32_evaluated":False,
        "candidate_created":False,"F6_exposed":False,"engine_requests":18,"startup_evaluations":18,"solver_calls":2}
    dual = tmp/"dual.npz"; np.savez(dual, generated=np.zeros(1))
    proof = {"schema":"flylab.m3.e283.saved-fresh-rf-linear-bound.v1","model_calls":0,"solver_calls":0,"original_calls":0,
        "dual_source":{"path":dual.name,"sha256":fit.base.sha256(dual)}, "records":[{"label": label,"dual_nonnegative":True,
            "final_linear_infeasibility_supported":True,"max_possible_weighted_slack_at_final_peak":-1.,
            "certified_linear_peak_lower_bound":.0368} for label in ("neutral","stage-0-joint-00-actual","stage-0-joint-01-actual")]}
    sensitivity = {"schema":"flylab.m3.e283.rf-radius-sensitivity.v1","model_calls":0,"solver_calls":0,"original_calls":0,
        "records":[{"label": label,"bound_at_radius_005":-1.,"bound_at_radius_010":-.5,
                    "larger_radius_certificate_inconclusive_not_feasible":False} for label in ("neutral","stage-0-joint-01-actual")]}
    freeze = {"schema":run.SCHEMA,"contract":copy.deepcopy(run.CONTRACT),"approved_sources":owned,
        "inherited_sources":{**old["approved_sources"],**old["inherited_sources"]},
        "e283_freeze":{"path":fp.name,"sha256":fs},"e283_source_review":{"path":rp.name,"sha256":rs},
        "e283_root_result":store(tmp/"e283-audit.json",audit),"e283_rf_linear_bound":store(tmp/"rf-bound.json",proof),
        "e283_radius_sensitivity":store(tmp/"radius.json",sensitivity), "rf_index":3,"rf_case":old["cases"][3],
        "rf_engine_roster":old["engine_roster"][3],
        **{k:old[k] for k in ("inputs","cases","engine_roster","packages","e280_resource_qualification")}}
    monkeypatch.setattr(run, "package_versions", lambda: old["packages"])
    roster_calls = []; monkeypatch.setattr(run,"validate_rf_roster",lambda *args: roster_calls.append(args))
    def save():
        fs = store(tmp/"e284-freeze.json",freeze)
        rs = store(tmp/"e284-review.json",{"schema":run.REVIEW_SCHEMA,"verdict":"approved","approved_sources":owned,"freeze_sha256":fs["sha256"]})
        return tmp/fs["path"],fs["sha256"],tmp/rs["path"],rs["sha256"]
    return freeze,original,save,tmp,roster_calls


def test_full_nested_26_source_provenance_preserves_lifetime_qualification(freeze_fixture):
    freeze, original, save, _, calls = freeze_fixture
    actual, inherited = run.validate_freeze(*save())
    assert actual == freeze and inherited == original and len(run.REQUIRED_INHERITED_SOURCES) == 26 and len(calls) == 1


@pytest.mark.parametrize("kind", ["source", "inherited", "contract", "packages", "inputs", "rf_index", "rf_case", "rf_roster", "prior_review", "audit", "proof", "radius", "qualification"])
def test_bad_freeze_is_closed_before_any_original(freeze_fixture, kind):
    freeze, _, save, tmp, calls = freeze_fixture
    if kind == "source": (tmp/run.SOURCES[0]).write_text("changed")
    elif kind == "inherited": (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif kind == "contract": freeze["contract"]["max_corrections"] = 5
    elif kind == "packages": freeze["packages"] = {"changed":True}
    elif kind == "inputs": freeze["inputs"] = {"changed":True}
    elif kind == "rf_index": freeze["rf_index"] = 4
    elif kind == "rf_case": freeze["rf_case"] = freeze["cases"][4]
    elif kind == "rf_roster": freeze["rf_engine_roster"] = freeze["engine_roster"][4]
    elif kind == "prior_review": freeze["e283_source_review"]["sha256"] = "0"*64
    elif kind == "audit": freeze["e283_root_result"]["sha256"] = "0"*64
    elif kind == "proof": freeze["e283_rf_linear_bound"]["sha256"] = "0"*64
    elif kind == "radius": freeze["e283_radius_sensitivity"]["sha256"] = "0"*64
    else: del freeze["e280_resource_qualification"]
    with pytest.raises(ValueError): run.validate_freeze(*save())
    assert not calls


def test_production_rf_exact_schema_and_no_other_leg_acceptance():
    # Full-size shapes only; no problem construction, neural or solver work.
    case = case_pack(); case["neutral_coefficients"] = fit.base.neutral(261)
    case["neutral_q_jacobian"] = np.zeros((99,7,783)); case["neutral_ltm_jacobian"] = np.zeros((99,1,783))
    row = {"leg":"RF","samples":101, **{k+"_indices":list(range(n)) for k,n in
           zip(("motor","muscle","edge","q","ltm"),(70,17,261,7,1),strict=True)}}
    assert run.validate_rf_roster(row,case)["edge_indices"].shape == (261,)
    for name in ("motor","muscle","edge","q","ltm"):
        bad = copy.deepcopy(row); bad[name+"_indices"].pop()
        with pytest.raises(ValueError): run.validate_rf_roster(bad,case)
    row["leg"] = "RM"
    with pytest.raises(ValueError): run.validate_rf_roster(row,case)


def meter(tmp, clock=lambda:0.):
    output = tmp/"output"; output.mkdir()
    m = run.CapacityMeter(output,run.ScienceDeadline(0.,clock=clock)); m.budget=run.CapacityBudget(m.deadline)
    return m


def test_actual_evaluator_hard_domain_beyond_trust_and_one_engine_live(tmp_path):
    case = case_pack(); m = meter(tmp_path); weak, derivatives = [], []
    class Engine(startup_tests.FakeEngine):
        def solve(self,c,*,derivative,host):
            derivatives.append(derivative); row=actual(c,derivative=derivative)
            self.last_receipt={"completed_steps":2000,"derivative_completed_steps":2000*int(derivative)}
            return {"q":row["startup_q"],"ltm":row["startup_ltm"],"receipt":self.last_receipt}
    def factory(mapping):
        assert mapping == "RF" and all(w() is None for w in weak)
        engine=Engine(case); weak.append(weakref.ref(engine)); return engine
    cleanup=m.cleanup
    def check_cleanup():
        assert all(w() is None for w in weak); cleanup()
    m.cleanup=check_cleanup; evaluate=run.make_evaluator("RF",case,factory,m)
    for index in range(6):
        row=evaluate(np.array([[1.7,.2,.3]]),derivative=index<5,label=f"eval-{index}")
        assert row["coefficients"][0,0] == 1.7
    assert all(w() is None for w in weak) and m.engine_requests == m.engine_creation["completed"] == 6
    assert derivatives == [True]*5+[False] and m.active_contexts == 0
    assert m.budget.receipt()["forward_completed_steps"] == 12000 and m.budget.receipt()["derivative_completed_steps"] == 10000


def test_first_constructor_failure_accounts_only_requested_engine(tmp_path):
    m=meter(tmp_path); original=OSError("constructor")
    def fail(*_): raise original
    with pytest.raises(OSError) as caught: run.make_evaluator("RF",case_pack(),fail,m)(fit.base.neutral(1),derivative=True,label="neutral")
    assert caught.value is original and m.engine_requests == 1 and m.engine_creation["failed"] == 1
    assert m.budget.receipt()["evaluations_requested"] == 0 and m.active_contexts == 0


def test_completed_evaluation_cleanup_failure_preserves_raw_response_and_ticks(tmp_path):
    case=case_pack(); m=meter(tmp_path); original=OSError("cleanup after complete")
    def cleanup(): raise original
    m.cleanup=cleanup
    evaluate=run.make_evaluator("RF",case,lambda *_:startup_tests.FakeEngine(case),m)
    saved={}
    with pytest.raises(OSError) as caught:
        fit.run_capacity(case,evaluate,persist=lambda n,a,r:saved.update({n:a}))
    assert caught.value is original and original.capacity_result["partial_evaluation"]["completed_response_available"]
    assert m.budget.counts["evaluations_completed"] == 1 and m.budget.counts["forward_completed_steps"] == 2000
    assert m.active_contexts == 0 and m.engine_requests == 1 and "neutral-partial" in saved
    assert (m.output/"neutral"/"eval-000.npz").is_file()
    assert m.evaluation_events[0]["status"] == "invalid"


def test_completed_evaluation_deadline_same_primary_and_finite_raw(tmp_path):
    case=case_pack(); now=[0.]; m=meter(tmp_path,clock=lambda:now[0]); original=fit.base.BudgetExceeded("original")
    original.add_note("first deadline")
    def check():
        if now[0] > 20: raise original
    m.deadline.check=check
    def factory(_):
        engine=startup_tests.FakeEngine(case); host_copy=engine.host_copy
        def copy(value,*,label): now[0]=20.00001; return host_copy(value,label=label)
        engine.host_copy=copy; return engine
    saved={}
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.run_capacity(case,run.make_evaluator("RF",case,factory,m),persist=lambda n,a,r:saved.update({n:a}))
    assert caught.value is original and original.capacity_result["status"] == "incomplete"
    assert m.budget.counts["evaluations_completed"] == 1 and m.budget.counts["derivative_completed_steps"] == 2000
    assert "neutral-partial" in saved and original.capacity_result["cost"]["solve"]["requested"] == 0
    assert "first deadline" in original.__notes__ and m.engine_requests == 1


def test_only_rf_startup_mapping_contract_no_teacher_or_other_leg(monkeypatch):
    axes=(np.array([2]),np.array([3]),np.array([4]),np.arange(7),np.array([5]))
    roster={"leg":"RF","samples":101,**{k+"_indices":v.tolist() for k,v in zip(("motor","muscle","edge","q","ltm"),axes,strict=True)}}
    original={"contexts":{"startup":{"mapping":{"stage_order":"order","stage_total":"startup"}},"teacher":{"mapping":{"stage_total":"forbidden"}}},"legs":{"q":"qleg"}}
    calls=[]
    def load(_,ref):
        assert ref != "forbidden"; calls.append(ref)
        return np.array(["start","half1","half2","end"]) if ref=="order" else np.zeros(1)
    def contract(mapping,legs,index):
        assert index == 3; calls.append(index); return {"RF_only":True},axes
    monkeypatch.setattr(run.base,"load_ref",load); monkeypatch.setattr(run.base,"validate_leg_ownership",lambda *_:None)
    monkeypatch.setattr(run.base,"contract_leg",contract)
    assert run.load_rf_inputs(original,roster)[2] == {"RF_only":True} and calls == ["order","startup","qleg",3]
    roster["edge_indices"]=[6]
    with pytest.raises(ValueError): run.load_rf_inputs(original,roster)


def test_fixed_evaluation_tick_time_memory_limits(tmp_path):
    now=[0.]; m=meter(tmp_path,clock=lambda:now[0])
    for _ in range(5): m.budget.request(True)
    with pytest.raises(fit.base.BudgetExceeded): m.budget.request(True)
    m.budget.request(False)
    with pytest.raises(fit.base.BudgetExceeded): m.budget.request(False)
    now[0]=20.000001
    with pytest.raises(fit.base.BudgetExceeded) as caught: m.deadline.check()
    assert caught.value.scope == "original" and m.deadline.receipt()["supervisor_wall_limit_s"] == 30.
    m.peak_estimate=4*1024**3+1
    with pytest.raises(ValueError): m.enforce_memory()
    m.peak_estimate=0; engine=startup_tests.FakeEngine(case_pack()); engine.problem["gpu_bytes_estimate"]=2*1024**3+1
    with pytest.raises(ValueError): m.memory(engine,"oversized")
    assert m.active_contexts == 0


@pytest.mark.parametrize("mode", ["terminal", "deadline", "after_aggregation"])
def test_upper_same_exception_partial_cost_is_aggregated_once(tmp_path,monkeypatch,mode):
    m=meter(tmp_path); original=fit.base.BudgetExceeded("original") if mode=="deadline" else OSError("storage")
    original.add_note("first"); solve,_=stub([[.5,0.,0.]])
    p=fit.capacity_problem(actual(fit.base.neutral(1))); cost=solve(p)["receipt"]
    cost.update(iterations_unknown_calls=0)
    partial={"cost":cost,"run_valid":True,"rf_capacity_supported":True}
    monkeypatch.setattr(run,"validate_rf_roster",lambda *_:{})
    def fail(*args,**kwargs):
        if mode=="after_aggregation": return partial
        original.capacity_result=partial; raise original
    monkeypatch.setattr(run.capacity,"run_capacity",fail)
    if mode=="after_aggregation": monkeypatch.setattr(m,"save",lambda:(_ for _ in ()).throw(original))
    freeze={"rf_engine_roster":{},"inherited_sources":{},"approved_sources":{},**{k:{} for k in
        ("e283_freeze","e283_source_review","e283_root_result","e283_rf_linear_bound","e283_radius_sensitivity","e280_resource_qualification")}}
    with pytest.raises(type(original)) as caught: run.execute_capacity(freeze,({}, {}, "RF"),case_pack(),lambda *_:pytest.fail("no engine"),m)
    assert caught.value is original and m.socp_cost["solve"]["requested"] == 1 and m.socp_cost["iterations"] == 2
    assert not original.capacity_result["run_valid"] and not original.capacity_result["rf_capacity_supported"]


def test_invalid_cli_has_original_solver_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"validate_freeze",lambda *_:(_ for _ in ()).throw(ValueError("freeze")))
    monkeypatch.setattr(run,"original_run",lambda *_:pytest.fail("no original"))
    output=tmp_path/"fresh"; args=["--freeze","x","--freeze-sha256","x","--source-review","x","--source-review-sha256","x","--output",str(output)]
    assert run.main(args)==1 and json.loads((output/"cost.json").read_text())["original_calls"] == 0
    assert not json.loads((output/"result.json").read_text())["rf_capacity_supported"]
    with pytest.raises(ValueError): run.main(args)
