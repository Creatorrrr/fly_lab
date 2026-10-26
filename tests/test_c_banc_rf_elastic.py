"""E285 independent tiny math and mocked RF lifecycle tests; original/CUDA0."""
import copy
import json
import time
import weakref
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tests import test_c_banc_rf_capacity as prior_tests
from tests import test_c_banc_startup_margin_runner as startup_tests
from tools import banc_m3_rf_elastic as fit
from tools import verify_c_banc_m3_rf_elastic as run

ACTUAL_COST = []


@pytest.fixture(scope="module", autouse=True)
def generated_cost():
    started = time.perf_counter()
    yield
    assert time.perf_counter()-started < 120 and len(ACTUAL_COST) <= 20
    print("E285_TINY_ACTUAL_COST", {k: sum(r[k] for r in ACTUAL_COST) for k in
        ("constructor_requested", "constructor_completed", "constructor_failed", "constructor_wall_s",
         "solve_requested", "solve_completed", "solve_failed", "solve_wall_s", "iterations")})


def counted_factory(*args):
    assert args[0].shape[0] <= 7 and len(ACTUAL_COST) < 20
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
    t = c[0, 0]-1
    q = .04-.02*t+(.004*t*t if nonlinear else 0) if qvalue is None else qvalue
    row = {"coefficients": c, "startup_q": np.zeros((101, 7)), "startup_ltm": np.zeros((101, 1)),
           "qerr": np.zeros((99, 7)), "lerr": np.zeros((99, 1))}
    row["qerr"][0, 0] = q; row["startup_q"][1:100] = row["qerr"]
    if derivative:
        row["startup_q_jacobian"] = np.zeros((99, 7, c.size)); row["startup_q_jacobian"][0, 0, 0] = -.02+(.008*t if nonlinear else 0)
        row["startup_ltm_jacobian"] = np.zeros((99, 1, c.size))
    return row


def case_pack(nonlinear=False):
    row = actual(fit.base.neutral(1), nonlinear=nonlinear)
    return {"neutral_coefficients": row["coefficients"], "neutral_startup_q": row["startup_q"], "neutral_startup_ltm": row["startup_ltm"],
        "neutral_q_jacobian": row["startup_q_jacobian"], "neutral_ltm_jacobian": row["startup_ltm_jacobian"],
        "target_q": np.zeros((99, 7)), "target_ltm": np.zeros((99, 1)), "q_sse_quota": np.array(.1),
        "ltm_sse_quota": np.array(99*(.009*.9999)**2)}


def analytic(kind="peak"):
    entry = actual(fit.base.neutral(1))
    if kind == "slack": entry["startup_q_jacobian"][:] = 0
    elif kind == "step": entry["qerr"][0, 0] = .048
    p = fit.elastic_problem(entry); x = np.zeros(4); y = np.zeros_like(p["b"])
    if kind == "peak":
        x[0] = (.04-p["peaks"][0])/.02; y[8] = x[0]/.02; y[p["cone_dimensions"][0]-1] = 100000-y[8]
    else:
        x[0] = .5 if kind == "step" else 0.
        x[-1] = entry["qerr"][0, 0]-.02*x[0]-p["peaks"][0]
        y[8] = 100000+x[-1]
        if kind == "step": y[0] = .02*y[8]-x[0]
    return p, x, y


@pytest.mark.parametrize("kind", ["peak", "slack", "step"])
def test_independent_analytic_penalized_primal_dual_and_actual_tiny(kind):
    p, optimum, y = analytic(kind); exact = p["b"]-p["A"]@optimum
    certificate = fit.elastic_certificate(p, optimum, exact, y, "AlmostSolved")
    assert certificate["valid"]
    w = p["q"]+p["A"].T@y
    np.testing.assert_allclose(optimum+w, 0, atol=2e-11)
    primal = .5*optimum@optimum+100000*optimum[-1]
    dual = -.5*w@w-p["b"]@y
    assert abs(primal-dual) < 1e-9 and abs(exact@y) < 1e-9
    solved = fit.solve_elastic(p, 4., solver_factory=counted_factory)
    assert solved["valid"] and solved["solution"].shape == (4,)
    np.testing.assert_array_equal(solved["correction"], solved["solution"][:-1])
    assert abs(solved["elastic_slack"]-optimum[-1]) < 1e-8
    assert .5*np.sum((solved["solution"]-optimum)**2) <= solved["receipt"]["certificate_receipt"]["gap"]+1e-8


def test_sparse_equations_slack_only_on_q_peak_and_explicit_last_rows():
    row = actual([[2., .25, .5], [1., .1, .2]])
    p = fit.elastic_problem(row); n = 6
    I = np.eye(n); triple = np.kron(np.eye(2), np.ones((1, 3)))
    Q = row["startup_q_jacobian"].reshape(693, n); L = row["startup_ltm_jacobian"].reshape(99, n)
    def pad(a): return np.column_stack((a, np.zeros(len(a))))
    nonnegative = np.vstack((pad(I), pad(-I), pad(triple), pad(-triple),
        np.column_stack((Q, -np.ones(693))), np.column_stack((-Q, -np.ones(693))), pad(L), pad(-L),
        np.r_[np.zeros(n), 1.], np.r_[np.zeros(n), -1.]))
    expected = np.vstack((nonnegative, np.zeros((1, n+1)), pad(-Q), np.zeros((1, n+1)), pad(-L)))
    np.testing.assert_array_equal(p["A"].toarray(), expected)
    np.testing.assert_array_equal(p["P"].toarray(), np.eye(7))
    np.testing.assert_array_equal(p["q"], np.r_[np.zeros(6), 100000.])
    np.testing.assert_array_equal(p["bounds"][:-1], np.column_stack((np.maximum(-row["coefficients"].ravel(), -.5),
                                                                                np.minimum(8-row["coefficients"].ravel(), .5))))
    assert fit.validate_problem(p) == (1602, 694, 100)
    assert not ({"radius", "correction_radius", "origins", "neutral_trust"} & set(p))
    np.testing.assert_array_equal(p["b"][1600:1602], [.01, 0.])
    arrays = fit.problem_arrays(p)
    recovered = sparse.csc_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=arrays["A_shape"])
    assert (recovered != p["A"]).nnz == 0 and arrays["slack_penalty"] == 100000


@pytest.mark.parametrize("key", ["P", "A", "q", "b", "bounds", "coefficient_bounds", "peaks", "radii", "step_radius", "slack_bounds", "slack_penalty", "trust", "cone"])
def test_mutated_original_problem_fails_before_solver(key):
    p = fit.elastic_problem(actual(fit.base.neutral(1)))
    if key in ("P", "A"): p[key].data[0] += 1e-4
    elif key in ("step_radius", "slack_penalty"): p[key] += 1e-4
    elif key == "trust": p["radius"] = .05
    elif key == "cone": p["cone_dimensions"] = (1,)
    else: p[key].flat[0] += 1e-4
    with pytest.raises(ValueError): fit.solve_elastic(p, 4., solver_factory=lambda *_: pytest.fail("no solver"))


@pytest.mark.parametrize("kind", ["status", "dual", "stationarity", "primal", "nan", "shape", "raw_difference"])
def test_certificate_independent_of_raw_s_but_rejects_bad_x_y(kind):
    p, x, y = analytic(); s = p["b"]-p["A"]@x; status = "Solved"
    if kind == "status": status = "MaxTime"
    elif kind == "dual": y[0] = -1e-12
    elif kind == "stationarity": y[8] = 0
    elif kind == "primal": x[0] -= .001
    elif kind == "nan": s[0] = np.inf
    elif kind == "shape": y = y[:-1]
    else: s += .001
    certificate = fit.elastic_certificate(p, x, s, y, status)
    assert certificate["valid"] == (kind == "raw_difference")
    if kind == "raw_difference": assert certificate["receipt"]["raw_affine_infinity_norm"] >= .00099999


@pytest.mark.parametrize("kind", ["q_peak", "q_sse", "ltm_peak", "ltm_sse", "negative", "upper", "triple_low", "triple_high"])
def test_actual_trial_slack_never_relaxes_other_checks(kind):
    row = actual([[2., .2, .3]], qvalue=.038)
    if kind == "q_peak": pass
    elif kind == "q_sse": row["qerr"][:] = np.sqrt(fit.CONTRACT["q_sse_limit"]/693)+1e-12
    elif kind == "ltm_peak": row["lerr"][0, 0] = .025*.9999+1e-12
    elif kind == "ltm_sse": row["lerr"][:] = .009*.9999+1e-12
    elif kind == "negative": row["coefficients"][0, 1] = -1e-12
    elif kind == "upper": row["coefficients"][0, 0] = 8+1e-12
    elif kind == "triple_low": row["coefficients"][0] = [.125*(1+1e-6)-1e-12, 0., 0.]
    else: row["coefficients"][0] = [8*(1-1e-6)+1e-12, 0., 0.]
    judged = fit.trial_judgement(row, .04, .035)
    assert judged["accepted"] == (kind == "q_peak") and not judged["strict"]["pass"]
    assert "q_peak" not in judged["checks"] and not judged["strict"]["global_norm_supported"]


def stub(directions, *, valid=True):
    queue = list(directions); problems = []
    def solve(p, *args, **kwargs):
        problems.append({k: copy.deepcopy(v) for k,v in p.items() if k != "cones"}); direction = np.asarray(queue.pop(0), np.float64)
        x = np.r_[direction, .004]
        receipt = {k: {**fit.metering.phase(), "requested": 1, "completed": 1, "wall_s": .01} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, status="Solved", solution_valid=valid)
        return {"valid": valid, "solution": x, "correction": direction, "elastic_slack": x[-1],
            "solver_s": np.zeros_like(p["b"]), "solver_y": np.zeros_like(p["b"]), "primal_slack": np.zeros_like(p["b"]),
            "residuals": None, "receipt": receipt}
    return solve, problems


def simulate(directions, *, values=None, nonlinear=False, valid=True, mutate=None, store=None):
    case = case_pack(nonlinear); calls, saved = [], {}; solve, problems = stub(directions, valid=valid)
    values = iter(values) if values is not None else None
    def evaluate(c, *, derivative, label):
        calls.append((c.copy(), derivative, label)); row = actual(c, derivative=derivative, nonlinear=nonlinear,
                                                                qvalue=None if values is None else next(values))
        if mutate: mutate(row, derivative, label)
        return row
    def persist(name, arrays, receipt):
        saved[name] = (copy.deepcopy(arrays), copy.deepcopy(receipt))
        if store: store(name, arrays, receipt)
    result = fit.run_elastic(case, evaluate, persist=persist, solve=solve)
    return result, calls, saved, problems


def test_actual_nonlinear_tiny_uses_only_fresh_j_then_single_f32():
    calls, saved = [], {}
    def evaluate(c, *, derivative, label):
        row = actual(c, derivative=derivative, nonlinear=True); calls.append((copy.deepcopy(row), derivative, label)); return row
    result = fit.run_elastic(case_pack(True), evaluate, persist=lambda n,a,r: saved.update({n:(copy.deepcopy(a),copy.deepcopy(r))}),
                             solver_factory=counted_factory)
    assert result["run_valid"] and result["f32_success"] and 2 <= len(result["outers"]) <= 6
    assert sum(label == "f32-actual" for _,_,label in calls) == 1 and not calls[-1][1]
    for row, derivative, label in calls:
        if "fresh" in label:
            assert derivative
            t = row["coefficients"][0,0]-1
            assert row["startup_q_jacobian"][0,0,0] == -.02+.008*t
            assert saved[label+"-reproduction"][1]["startup_q"]["pass"]
    for index in range(1,len(result["outers"])):
        stored = saved[f"outer-{index:02d}-problem"][0]
        assert stored["q_jacobian_data"][0] != saved["outer-00-problem"][0]["q_jacobian_data"][0]


def test_actual_flat_j_positive_slack_stops_before_any_trial():
    case = case_pack(); case["neutral_q_jacobian"][:] = 0; calls=[]
    def evaluate(c, *, derivative, label):
        row=actual(c,derivative=derivative)
        if derivative: row["startup_q_jacobian"][:] = 0
        calls.append(label); return row
    result = fit.run_elastic(case,evaluate,persist=lambda *_:None,solver_factory=counted_factory)
    assert result["run_valid"] and result["status"] == "fixed_alpha_line_search_rejected" and calls == ["neutral"]
    assert result["outers"][0]["elastic_slack"] > 0 and all(not r["actual_evaluated"] for r in result["outers"][0]["trials"])


def test_actual_q_soc_infeasible_remains_bounded_uncertified():
    row = actual(fit.base.neutral(1)); row["qerr"][:] = .04; row["startup_q_jacobian"][:] = 0
    result = fit.solve_elastic(fit.elastic_problem(row),4.,solver_factory=counted_factory)
    assert not result["valid"] and result["receipt"]["status"] == "PrimalInfeasible"


def test_fixed_alpha_full_rejection_half_acceptance_fresh_then_f32():
    result,calls,_,_ = simulate([[.4,0,0],[.1,0,0]], values=[.04,.041,.038,.038,.035,.035])
    assert result["rf_capacity_supported"] and result["outers"][0]["accepted_alpha"] == .5
    assert [d for _,d,_ in calls] == [True,False,False,True,False,False]
    assert not result["outers"][0]["trials"][0]["accepted"] and result["outers"][0]["trials"][1]["accepted"]


def test_certified_raw_overstep_is_checked_per_alpha_without_clipping():
    direction=np.array([.5+1e-12,0,0])
    result,calls,saved,_ = simulate([direction],values=[.04,.035,.035])
    trials=result["outers"][0]["trials"]
    assert not trials[0]["gates"]["step"] and not trials[0]["actual_evaluated"] and trials[1]["accepted"]
    np.testing.assert_array_equal(saved["outer-00-solve"][0]["solution"][:-1],direction)
    np.testing.assert_array_equal(calls[1][0],fit.base.neutral(1)+.5*direction)


@pytest.mark.parametrize("direction,valid,status", [([.1,-1e-12,0],True,"fixed_alpha_line_search_rejected"),
    ([.1,0,0],False,"uncertified_elastic_direction"), ([-.1,0,0],True,"fixed_alpha_line_search_rejected")])
def test_invalid_raw_or_unhelpful_prediction_never_evaluated_or_repaired(direction,valid,status):
    result,calls,saved,_=simulate([direction],valid=valid)
    assert result["run_valid"] and result["status"] == status and len(calls)==1
    np.testing.assert_array_equal(saved["outer-00-solve"][0]["correction"],direction)


def test_fixed_six_outer_four_alpha_maximum31_and_derivatives6():
    solve,_=stub([[.4,0,0]]*6); calls=[]; remembered={}; case=case_pack()
    def evaluate(c, *, derivative,label):
        calls.append((label,derivative))
        if label=="neutral": q=.04
        elif derivative: q=remembered[tuple(c.ravel())]
        elif label=="f32-actual": q=.034
        else:
            outer=int(label.split("-")[1]); index=int(label.split("-")[3])
            q=.041 if index<3 else (.04-.0005*(outer+1) if outer<5 else .034)
            if index==3: remembered[tuple(c.ravel())]=q
        return actual(c,derivative=derivative,qvalue=q)
    result=fit.run_elastic(case,evaluate,persist=lambda *_:None,solve=solve)
    assert result["rf_capacity_supported"] and len(calls)==31 and sum(d for _,d in calls)==6
    assert len(result["outers"])==6 and result["cost"]["fresh_jacobians_used"]==6
    assert all(r["accepted_alpha"]==.125 and len(r["trials"])==4 for r in result["outers"])


def test_six_outer_nonfinal_exhaustion_no_seventh_fresh_or_qp():
    values=[.04]
    for outer in range(6):
        if outer: values.append(.04-.0001*outer)
        values.append(.04-.0001*(outer+1))
    result,calls,_,_=simulate([[.01,0,0]]*6,values=values)
    assert result["status"]=="six_outer_exhausted" and result["run_valid"] and not result["f64_success"]
    assert len(calls)==12 and sum(d for _,d,_ in calls)==6 and result["cost"]["solve"]["requested"]==6


def test_f32_bad_triple_is_saved_without_repair_or_actual_replay():
    directions=[[-.5,0,0],[.125*(1+1e-6)-.5,0,0]]
    def mutate(row,derivative,_):
        if derivative: row["startup_q_jacobian"][0,0,0]=.02
    case=case_pack(); case["neutral_q_jacobian"][0,0,0]=.02
    values=iter([.04,.039,.039,0.]); calls=[]; saved={}; solve,_=stub(directions)
    def evaluate(c,*,derivative,label):
        calls.append(label); row=actual(c,derivative=derivative,qvalue=next(values)); mutate(row,derivative,label); return row
    result=fit.run_elastic(case,evaluate,persist=lambda n,a,r:saved.update({n:a}),solve=solve)
    assert result["f64_success"] and not result["f32_success"] and len(calls)==4
    assert result["status"]=="f32_hard_coefficient_rejection_before_evaluation"
    assert saved["f32-coefficients"]["coefficients32"].dtype==np.float32


def test_f32_actual_gate_has_no_linear_slack():
    result,calls,_,_=simulate([[.4,0,0]],values=[.04,.035,.036*.9999+1e-12])
    assert result["run_valid"] and result["f64_success"] and not result["rf_capacity_supported"] and len(calls)==3
    assert result["status"]=="f32_rf_startup_rejection"


@pytest.mark.parametrize("key", ["neutral_startup_q","neutral_startup_ltm","neutral_q_jacobian","neutral_ltm_jacobian"])
def test_neutral_reproduction_stops_solver(key):
    case=case_pack(); case[key].flat[0]+=1e-6
    with pytest.raises(fit.metering.ReproductionError) as caught:
        fit.run_elastic(case,lambda c,derivative,label:actual(c,derivative=derivative),persist=lambda *_:None,
                        solver_factory=lambda *_:pytest.fail("no solver"))
    assert caught.value.elastic_result["cost"]["solve"]["requested"]==0


def test_fresh_reproduction_failure_never_uses_stale_j_or_solves_again():
    original=fit.metering.ReproductionError
    with pytest.raises(original) as caught: simulate([[.1,0,0]],values=[.04,.038,.038+1e-8])
    assert caught.value.elastic_result["cost"]["solve"]["requested"]==1
    assert caught.value.elastic_result["cost"]["evaluation"]["completed"]==3


@pytest.mark.parametrize("phase", ["constructor","solve","certificate","solve_clock"])
def test_failure_phase_primary_raw_iterations_and_metering(monkeypatch,phase):
    p=fit.elastic_problem(actual(fit.base.neutral(1))); original=OSError(phase)
    raw=SimpleNamespace(x=np.zeros(4),s=np.zeros_like(p["b"]),z=np.zeros_like(p["b"]),status="Solved",iterations=11)
    def fail(*_): raise original
    factory=fail if phase=="constructor" else lambda *_:SimpleNamespace(solve=fail if phase=="solve" else lambda:raw)
    if phase=="certificate": monkeypatch.setattr(fit,"elastic_certificate",fail)
    times=[0]
    def clock():
        times[0]+=1
        if phase=="solve_clock" and times[0]==4: raise original
        return times[0]
    saved={}
    with pytest.raises(OSError) as caught:
        fit.run_elastic(case_pack(),lambda c,derivative,label:actual(c,derivative=derivative),persist=lambda n,a,r:saved.update({n:a}),
            solve=lambda p,t,**kwargs:fit.solve_elastic(p,t,solver_factory=factory,wall_clock=clock))
    assert caught.value is original and original.elastic_result["cost"]["constructor"]["requested"]==1
    assert original.elastic_result["cost"]["solve"]["requested"]==int(phase!="constructor")
    if phase in ("certificate","solve_clock"):
        assert original.partial_elastic_solution is raw and original.elastic_result["cost"]["iterations"]==11
        np.testing.assert_array_equal(saved["outer-00-solve"]["solution"],raw.x)


def test_original_deadline_secondary_storage_retains_first_object_and_partial():
    original=fit.base.BudgetExceeded("original"); original.add_note("first deadline")
    def evaluate(c,**kwargs): original.completed_margin_entry=actual(c); raise original
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.run_elastic(case_pack(),evaluate,persist=lambda *_:(_ for _ in ()).throw(OSError("secondary storage")))
    assert caught.value is original and original.elastic_result["status"]=="incomplete"
    assert original.elastic_result["partial_evaluation"]["completed_response_available"] and original.abort_run
    assert original.elastic_result["cost"]["solve"]["requested"]==0 and any("secondary storage" in n for n in original.__notes__)


def test_final_storage_failure_attaches_cost_once_and_revokes_support():
    original=OSError("terminal")
    def store(name,*_):
        if name=="elastic-result": raise original
    with pytest.raises(OSError) as caught: simulate([[.4,0,0]],values=[.04,.035,.035],store=store)
    assert caught.value is original and original.abort_run and not original.elastic_result["run_valid"]
    assert not original.elastic_result["rf_capacity_supported"] and original.elastic_result["cost"]["evaluation"]["completed"]==3
    assert original.elastic_result["cost"]["iterations"]==2


@pytest.fixture
def freeze_fixture(tmp_path,monkeypatch):
    old,original,save_old,tmp,_=prior_tests.freeze_fixture.__wrapped__(tmp_path,monkeypatch)
    fp,fs,rp,rs=save_old(); owned={}
    for name in run.SOURCES:
        path=tmp/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("generated E285 source"); owned[name]=fit.base.sha256(path)
    def store(path,value):
        path.write_text(json.dumps(value)); return {"path":path.name,"sha256":fit.base.sha256(path)}
    raw=tmp/"e284-proof-raw.npz"; np.savez(raw,generated=np.zeros(1)); spec={"path":raw.name,"sha256":fit.base.sha256(raw)}
    audit={"schema":"flylab.m3.e284.root-original-audit.v1","verdict":"valid_diagnostic_rf_capacity_not_supported",
        "reproduction_exact":True,"hard_domain_sparse_equations_exact":True,"no_neutral_or_step_trust":True,
        "engine_requests":1,"startup_evaluations":1,"solver_calls":1,"corrected_actual_evaluations":0,"f32_evaluations":0,
        **dict.fromkeys(("rf_capacity_supported","initialization_supported","global_norm_supported","candidate_created","F6_exposed"),False),
        "infeasibility_checks":[{"label":"full_hard_domain_original","bounded_linear_infeasibility_supported":True,"max_possible_weighted_slack":-1.}]}
    proof={"schema":"flylab.m3.e284.peak-structure-audit.v1","model_calls":0,"solver_calls":0,"original_calls":0,
        "source_problem":spec,"source_dual":spec,"converted_dual_nonnegative":True,"q0_warm_j_change_max":.1,
        "records":[{"label":label,"input":spec,"fixed_linear_infeasibility_supported":index==0,
            "weighted_slack_upper":-1. if index==0 else 1.,"peak_lower_bound":.0361} for index,label in
            enumerate(("neutral","e283_first_middle_actual","e283_last_middle_actual"))]}
    freeze={"schema":run.SCHEMA,"contract":copy.deepcopy(run.CONTRACT),"approved_sources":owned,
        "inherited_sources":{**old["inherited_sources"],**old["approved_sources"]},
        "e284_freeze":{"path":fp.name,"sha256":fs},"e284_source_review":{"path":rp.name,"sha256":rs},
        "e284_root_result":store(tmp/"e284-audit.json",audit),"e284_peak_structure_audit":store(tmp/"peak-structure.json",proof),
        "rf_index":3,"rf_case":old["cases"][3],"rf_engine_roster":old["engine_roster"][3],
        **{k:old[k] for k in ("inputs","cases","engine_roster","packages","e280_resource_qualification")}}
    monkeypatch.setattr(run,"package_versions",lambda:old["packages"])
    calls=[]; monkeypatch.setattr(run,"validate_rf_roster",lambda *args:calls.append(args))
    def save():
        f=store(tmp/"e285-freeze.json",freeze)
        r=store(tmp/"e285-review.json",{"schema":run.REVIEW_SCHEMA,"verdict":"approved","approved_sources":owned,"freeze_sha256":f["sha256"]})
        return tmp/f["path"],f["sha256"],tmp/r["path"],r["sha256"]
    return freeze,original,save,tmp,calls


def test_nested29_sources_startup_only_provenance_and_resource_qualification(freeze_fixture):
    freeze,original,save,_,calls=freeze_fixture
    actual_freeze,inherited=run.validate_freeze(*save())
    assert actual_freeze==freeze and inherited==original and len(run.REQUIRED_INHERITED_SOURCES)==29 and len(calls)==1


@pytest.mark.parametrize("kind", ["source","inherited","contract","packages","inputs","rf_index","rf_case","rf_roster","prior_review","audit","proof","proof_input","qualification"])
def test_bad_freeze_closed_before_original(freeze_fixture,kind):
    freeze,_,save,tmp,calls=freeze_fixture
    if kind=="source": (tmp/run.SOURCES[0]).write_text("changed")
    elif kind=="inherited": (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif kind=="contract": freeze["contract"]["max_outer"]=7
    elif kind=="packages": freeze["packages"]={"changed":True}
    elif kind=="inputs": freeze["inputs"]={"changed":True}
    elif kind=="rf_index": freeze["rf_index"]=4
    elif kind=="rf_case": freeze["rf_case"]=freeze["cases"][4]
    elif kind=="rf_roster": freeze["rf_engine_roster"]=freeze["engine_roster"][4]
    elif kind=="prior_review": freeze["e284_source_review"]["sha256"]="0"*64
    elif kind=="audit": freeze["e284_root_result"]["sha256"]="0"*64
    elif kind=="proof": freeze["e284_peak_structure_audit"]["sha256"]="0"*64
    elif kind=="proof_input": (tmp/"e284-proof-raw.npz").write_bytes(b"tampered")
    else: del freeze["e280_resource_qualification"]
    with pytest.raises((ValueError,KeyError)): run.validate_freeze(*save())
    assert not calls


@pytest.mark.parametrize("kind", ["audit_verdict","audit_engine_count","audit_supported","audit_witness","peak_schema","peak_counter","peak_labels","peak_neutral","peak_warm","peak_change"])
def test_invalid_prior_evidence_semantics_rejected(freeze_fixture,kind):
    freeze,_,save,tmp,_=freeze_fixture
    key="e284_root_result" if kind.startswith("audit") else "e284_peak_structure_audit"
    path=tmp/freeze[key]["path"]; value=json.loads(path.read_text())
    if kind=="audit_verdict": value["verdict"]="accepted"
    elif kind=="audit_engine_count": value["engine_requests"]=2
    elif kind=="audit_supported": value["rf_capacity_supported"]=True
    elif kind=="audit_witness": value["infeasibility_checks"][0]["max_possible_weighted_slack"]=1.
    elif kind=="peak_schema": value["schema"]="changed"
    elif kind=="peak_counter": value["solver_calls"]=1
    elif kind=="peak_labels": value["records"][0]["label"]="changed"
    elif kind=="peak_neutral": value["records"][0]["peak_lower_bound"]=.035
    elif kind=="peak_warm": value["records"][1]["fixed_linear_infeasibility_supported"]=True
    else: value["q0_warm_j_change_max"]=0.
    path.write_text(json.dumps(value)); freeze[key]["sha256"]=fit.base.sha256(path)
    with pytest.raises(ValueError): run.validate_freeze(*save())


def meter(tmp,clock=lambda:0.):
    output=tmp/"output"; output.mkdir()
    m=run.ElasticMeter(output,run.ScienceDeadline(0.,clock=clock)); m.budget=run.ElasticBudget(m.deadline); return m


def test_one_actual_engine_live31_evaluations_and_derivatives6(tmp_path):
    case=case_pack(); m=meter(tmp_path); refs=[]; modes=[]
    class Engine(startup_tests.FakeEngine):
        def solve(self,c,*,derivative,host):
            modes.append(derivative); row=actual(c,derivative=derivative)
            self.last_receipt={"completed_steps":2000,"derivative_completed_steps":2000*int(derivative)}
            return {"q":row["startup_q"],"ltm":row["startup_ltm"],"receipt":self.last_receipt}
    def factory(mapping):
        assert mapping=="RF" and all(r() is None for r in refs)
        engine=Engine(case); refs.append(weakref.ref(engine)); return engine
    cleanup=m.cleanup
    def check_cleanup(): assert all(r() is None for r in refs); cleanup()
    m.cleanup=check_cleanup; evaluate=run.make_evaluator("RF",case,factory,m)
    for index in range(31): evaluate(np.array([[1.7,.2,.3]]),derivative=index<6,label=f"eval-{index}")
    assert all(r() is None for r in refs) and m.engine_requests==m.engine_creation["completed"]==31 and m.active_contexts==0
    assert modes==[True]*6+[False]*25 and m.budget.counts["forward_completed_steps"]==62000
    assert m.budget.counts["derivative_completed_steps"]==12000


def test_first_constructor_failure_requested_only_one_engine_and_no_evaluation(tmp_path):
    m=meter(tmp_path); original=OSError("constructor")
    def fail(*_): raise original
    with pytest.raises(OSError) as caught: run.make_evaluator("RF",case_pack(),fail,m)(fit.base.neutral(1),derivative=True,label="neutral")
    assert caught.value is original and m.engine_requests==m.engine_creation["failed"]==1
    assert m.budget.counts["evaluations_requested"]==0 and m.active_contexts==0


def test_completed_cleanup_failure_preserves_raw_ticks_and_same_error(tmp_path):
    case=case_pack(); m=meter(tmp_path); original=OSError("cleanup")
    def cleanup(): raise original
    m.cleanup=cleanup; saved={}
    with pytest.raises(OSError) as caught:
        fit.run_elastic(case,run.make_evaluator("RF",case,lambda *_:startup_tests.FakeEngine(case),m),persist=lambda n,a,r:saved.update({n:a}))
    assert caught.value is original and original.elastic_result["partial_evaluation"]["completed_response_available"]
    assert m.engine_requests==1 and m.budget.counts["evaluations_completed"]==1 and m.budget.counts["forward_completed_steps"]==2000
    assert m.active_contexts==0 and "neutral-partial" in saved and (m.output/"neutral"/"eval-000.npz").is_file()


def test_completed_deadline_same_primary_and_saved_response(tmp_path):
    case=case_pack(); now=[0.]; m=meter(tmp_path,clock=lambda:now[0]); original=fit.base.BudgetExceeded("original"); original.add_note("initial")
    def check():
        if now[0]>35: raise original
    m.deadline.check=check
    def factory(_):
        engine=startup_tests.FakeEngine(case); host_copy=engine.host_copy
        def copy(value,*,label): now[0]=35.00001; return host_copy(value,label=label)
        engine.host_copy=copy; return engine
    saved={}
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.run_elastic(case,run.make_evaluator("RF",case,factory,m),persist=lambda n,a,r:saved.update({n:a}))
    assert caught.value is original and original.elastic_result["status"]=="incomplete" and "neutral-partial" in saved
    assert m.budget.counts["evaluations_completed"]==1 and m.budget.counts["derivative_completed_steps"]==2000
    assert m.engine_requests==1 and "initial" in original.__notes__


def test_exact_eval_derivative_time_memory_and_qp_bounds(tmp_path):
    now=[0.]; m=meter(tmp_path,clock=lambda:now[0])
    for _ in range(6): m.budget.request(True)
    with pytest.raises(fit.base.BudgetExceeded): m.budget.request(True)
    for _ in range(25): m.budget.request(False)
    with pytest.raises(fit.base.BudgetExceeded): m.budget.request(False)
    now[0]=35.00001
    with pytest.raises(fit.base.BudgetExceeded) as caught: m.deadline.check()
    assert caught.value.scope=="original" and m.deadline.receipt()["supervisor_wall_limit_s"]==40.
    m.peak_estimate=4*1024**3+1
    with pytest.raises(ValueError): m.enforce_memory()
    m.peak_estimate=0; engine=startup_tests.FakeEngine(case_pack()); engine.problem["gpu_bytes_estimate"]=2*1024**3+1
    with pytest.raises(ValueError): m.memory(engine,"oversized")
    solve,_=stub([[.1,0,0]]*7)
    p=fit.elastic_problem(actual(fit.base.neutral(1))); cost=solve(p)["receipt"]; cost["iterations_unknown_calls"]=0
    for _ in range(6): m.account(cost)
    with pytest.raises(ValueError): m.account(cost)


@pytest.mark.parametrize("mode", ["terminal","deadline","after_aggregation"])
def test_upper_cost_aggregates_once_and_keeps_primary(tmp_path,monkeypatch,mode):
    m=meter(tmp_path); original=fit.base.BudgetExceeded("original") if mode=="deadline" else OSError("storage"); original.add_note("first")
    solve,_=stub([[.1,0,0]]); p=fit.elastic_problem(actual(fit.base.neutral(1))); cost=solve(p)["receipt"]; cost["iterations_unknown_calls"]=0
    partial={"cost":cost,"run_valid":True,"rf_capacity_supported":True}
    monkeypatch.setattr(run,"validate_rf_roster",lambda *_:{})
    def fail(*args,**kwargs):
        if mode=="after_aggregation": return partial
        original.elastic_result=partial; raise original
    monkeypatch.setattr(fit,"run_elastic",fail)
    if mode=="after_aggregation": monkeypatch.setattr(m,"save",lambda:(_ for _ in ()).throw(original))
    freeze={"rf_engine_roster":{},"inherited_sources":{},"approved_sources":{},**{k:{} for k in
        ("e284_freeze","e284_source_review","e284_root_result","e284_peak_structure_audit","e280_resource_qualification")}}
    with pytest.raises(type(original)) as caught: run.execute_elastic(freeze,({},{},"RF"),case_pack(),lambda *_:pytest.fail("no engine"),m)
    assert caught.value is original and m.socp_cost["solve"]["requested"]==1 and m.socp_cost["iterations"]==2
    assert not original.elastic_result["run_valid"] and not original.elastic_result["rf_capacity_supported"]


def test_invalid_cli_freeze_original_solver_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"validate_freeze",lambda *_:(_ for _ in ()).throw(ValueError("freeze")))
    monkeypatch.setattr(run,"original_run",lambda *_:pytest.fail("no original"))
    output=tmp_path/"fresh"; args=["--freeze","x","--freeze-sha256","x","--source-review","x","--source-review-sha256","x","--output",str(output)]
    assert run.main(args)==1 and json.loads((output/"cost.json").read_text())["original_calls"]==0
    assert not json.loads((output/"result.json").read_text())["rf_capacity_supported"]
    with pytest.raises(ValueError): run.main(args)
