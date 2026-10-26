"""Generated E283 continuation, strict units, provenance and failure accounting."""
import copy
import json
import time
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tests import test_c_banc_joint_startup_runner as prior_tests
from tools import verify_c_banc_m3_startup_continuation as run

TINY_COST, EVALUATIONS, FAILURE_COST, FAKE_CALLS = [], [], [], []


@pytest.fixture(scope="module", autouse=True)
def generated_cost():
    start = time.perf_counter()
    yield
    wall = time.perf_counter()-start
    assert wall < 120 and len(TINY_COST) <= 20
    print("e283_generated_cost", {"wall_s": wall, "tiny_solve_calls": len(TINY_COST),
        "tiny_iterations": sum(r["iterations"] for r in TINY_COST), "actual_generated_evaluations": len(EVALUATIONS),
        "constructor_s": sum(r["constructor_s"] for r in TINY_COST),
        "solve_s": sum(r["solve_s"] for r in TINY_COST), "injected_phase_failures": FAILURE_COST,
        "fake_round_evaluations": len(FAKE_CALLS), "original": 0, "CUDA": 0, "fullsize": 0})


def actual(c, *, derivative=True, nonlinear=False, qvalue=None):
    c = np.asarray(c, np.float64).reshape(-1, 3).copy()
    assert c.size <= 6
    x = c[0, 0]-1
    q = .04-.1*x-(x*x if nonlinear else 0) if qvalue is None else qvalue
    row = {"coefficients": c, "startup_q": np.zeros((101, 7)), "startup_ltm": np.zeros((101, 1)),
           "qerr": np.zeros((99, 7)), "lerr": np.zeros((99, 1))}
    row["qerr"][0, 0] = q; row["startup_q"][1:100] = row["qerr"]
    if derivative:
        row["startup_q_jacobian"] = np.zeros((99, 7, c.size)); row["startup_q_jacobian"][0, 0, 0] = -.1-(2*x if nonlinear else 0)
        row["startup_ltm_jacobian"] = np.zeros((99, 1, c.size))
    return row


def case_pack(*, nonlinear=False):
    row = actual(run.base.neutral(1), nonlinear=nonlinear)
    return {"neutral_coefficients": row["coefficients"], "neutral_startup_q": row["startup_q"],
        "neutral_startup_ltm": row["startup_ltm"], "neutral_q_jacobian": row["startup_q_jacobian"],
        "neutral_ltm_jacobian": row["startup_ltm_jacobian"], "target_q": np.zeros((99, 7)),
        "target_ltm": np.zeros((99, 1)), "q_sse_quota": np.array(.1),
        "ltm_sse_quota": np.array(99*(.009*(1-1e-4))**2)}


def counted_factory(*args):
    assert args[0].shape[0] <= 6 and len(TINY_COST) < 20
    record = {"iterations": 0, "constructor_s": 0., "solve_s": 0.}; TINY_COST.append(record)
    start = time.perf_counter()
    try: solver = clarabel.DefaultSolver(*args)
    finally: record["constructor_s"] = time.perf_counter()-start
    def solve():
        start = time.perf_counter()
        try:
            result = solver.solve(); record["iterations"] = result.iterations; return result
        finally: record["solve_s"] = time.perf_counter()-start
    return SimpleNamespace(solve=solve)


def stub(directions, *, valid=True):
    problems = []
    def solve(problem, *args, **kwargs):
        z = np.array(directions[len(problems)], np.float64); problems.append(problem)
        receipt = {k: {**run.metering.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
        receipt.update(iterations=2, iterations_known=True, generated_stub=True)
        s = problem["b"]-problem["A"]@z
        return {"valid": valid, "receipt": receipt, "correction": z, "solver_s": s,
                "solver_y": np.zeros_like(s), "primal_slack": s, "residuals": None}
    return solve, problems


def simulate(directions=None, *, values=None, nonlinear=False, mutate=None, store=None, valid=True):
    calls, saved = [], {}
    def evaluate(cs, *, derivative, label):
        rows = [actual(c, derivative=derivative, nonlinear=nonlinear,
                       qvalue=None if values is None else values[len(calls)]) for c in cs]
        if mutate: mutate(rows, derivative, label)
        calls.append((label, derivative, rows))
        if directions is None: EVALUATIONS.extend(rows)
        else: FAKE_CALLS.extend(rows)
        return rows
    def persist(name, arrays, receipt):
        saved[name] = copy.deepcopy((arrays, receipt))
        if store: store(name, arrays, receipt)
    kwargs, problems = {}, []
    if directions is None: kwargs["solver_factory"] = counted_factory
    else: kwargs["solve"], problems = stub(directions, valid=valid)
    result = run.run_continuation([case_pack(nonlinear=nonlinear)], evaluate, persist=persist, **kwargs)
    return result, calls, saved, problems


def test_actual_nonlinear_two_stages_fresh_j_without_transition_evaluation():
    result, calls, saved, _ = simulate(nonlinear=True)
    assert result["initialization_supported"] and len(calls) == 4
    assert [c[0] for c in calls] == ["neutral", "stage-0-joint-00-actual", "stage-1-joint-00-actual", "f32-actual"]
    assert [c[1] for c in calls] == [True, True, True, False]
    assert result["cost"]["solve"]["completed"] == 2
    for stage in (0, 1):
        arrays, metadata = saved[f"stage-{stage}-joint-00-problem"]
        matrix = sparse.csc_matrix((arrays["q_jacobian_data"], arrays["q_jacobian_indices"], arrays["q_jacobian_indptr"]), shape=arrays["q_jacobian_shape"])
        np.testing.assert_array_equal(matrix.toarray(), calls[stage][2][0]["startup_q_jacobian"].reshape(693, 3)*run.STAGES[stage]["q_scale"])
        np.testing.assert_array_equal(arrays["qerr"], calls[stage][2][0]["qerr"].ravel()*run.STAGES[stage]["q_scale"])
        assert metadata["stage"] == stage and metadata["raw_units_preserved"]
        np.testing.assert_array_equal(arrays["origins"], run.base.neutral(1))
    assert calls[1][2][0]["startup_q_jacobian"][0, 0, 0] != calls[0][2][0]["startup_q_jacobian"][0, 0, 0]
    np.testing.assert_array_equal(saved["f32-coefficients"][0]["coefficients_leg_major"], calls[-2][2][0]["coefficients"].astype(np.float32).astype(np.float64))


def test_stage_view_raw_unchanged_and_only_q_error_j_scaled():
    row = actual(run.base.neutral(1)); original = copy.deepcopy(row)
    view = run.stage_view([row], 0)[0]
    for key in original: np.testing.assert_array_equal(row[key], original[key])
    for key in ("qerr", "startup_q_jacobian"): np.testing.assert_array_equal(view[key], original[key]*18/19)
    for key in ("startup_q", "startup_ltm", "lerr", "startup_ltm_jacobian", "coefficients"): np.testing.assert_array_equal(view[key], original[key])
    view["qerr"][:] = 0
    np.testing.assert_array_equal(row["qerr"], original["qerr"])


@pytest.mark.parametrize("kind", ["q_peak", "global_q", "ltm_peak", "ltm_sse", "trust"])
def test_strict_original_and_normalized_limits_zero_tolerance(kind):
    rows = [actual(run.base.neutral(1), qvalue=0) for _ in range(6)]
    if kind == "q_peak": rows[0]["qerr"][0, 0] = run.STAGES[0]["q_peak_limit"]+1e-12
    elif kind == "global_q":
        for row in rows: row["qerr"][:] = .0171*(1-1e-4)+1e-12
    elif kind == "ltm_peak": rows[0]["lerr"][0, 0] = .025*(1-1e-4)+1e-12
    elif kind == "ltm_sse": rows[0]["lerr"][:] = .009*(1-1e-4)+1e-12
    else: rows[0]["coefficients"][0, 0] = 1.05+1e-12
    assert not run.strict_stage(rows, [run.base.neutral(1)]*6, 0)["pass"]
    good = actual(run.base.neutral(1), qvalue=.037)
    assert run.strict_stage([good], [run.base.neutral(1)], 0)["pass"]
    assert not run.strict_stage([good], [run.base.neutral(1)], 1)["pass"]


@pytest.mark.parametrize("values,status,solves", [([.04,.039,.039], "stage_0_two_corrections_rejected", 2),
    ([.04,.039,.037,.037,.037], "stage_1_two_corrections_rejected", 4)])
def test_fixed_two_per_stage_four_total_and_failure_no_f32(values, status, solves):
    result, calls, saved, problems = simulate([[.002,0.,0.]]*solves, values=values)
    assert result["run_valid"] and not result["initialization_supported"] and result["status"] == status
    assert len(problems) == solves and len(calls) == solves+1 and "f32-coefficients" not in saved
    assert all(s["corrections"] == 2 for s in result["stages"])


def test_original_trust_not_reset_at_stage_transition():
    result, calls, _, problems = simulate([[.03,0.,0.],[.03,0.,0.]], values=[.04,.037])
    assert len(calls) == 2 and len(problems) == 2
    assert result["status"] == "strict_coefficient_trust_or_correction_rejection"
    np.testing.assert_array_equal(problems[1]["origins"], run.base.neutral(1))
    assert problems[1]["coefficient_bounds"][0,1] == float(np.float32(1.05))


@pytest.mark.parametrize("direction,valid,status", [([.002,-1e-12,0.],True,"strict_coefficient_trust_or_correction_rejection"),
    ([.002,.05-1e-10,0.],True,"strict_inward_coefficient_interval_rejection"), ([0.,0.,0.],False,"uncertified_joint_correction")])
def test_raw_solution_not_clipped_retried_or_evaluated(direction, valid, status):
    result, calls, saved, _ = simulate([direction], valid=valid)
    assert result["status"] == status and len(calls) == 1 and not result["f64_success"]
    np.testing.assert_array_equal(saved["stage-0-joint-00-solve"][0]["correction"], direction)


def test_final_f64_then_one_actual_f32_strict_rejection():
    result, calls, _, _ = simulate([[.002,0.,0.]], values=[.04,.035,.036*(1-1e-4)+1e-12])
    assert result["f64_success"] and not result["f32_success"] and result["run_valid"]
    assert len(calls) == 3 and not calls[-1][1]


@pytest.mark.parametrize("field", ["qerr", "lerr"])
def test_actual_target_relationship_checked_before_problem(field):
    def mutate(rows, *args): rows[0][field].flat[0] += 1e-6
    with pytest.raises(ValueError) as caught: simulate([], mutate=mutate)
    assert caught.value.joint_result["cost"]["solve"]["requested"] == 0
    assert caught.value.joint_result["partial_round"]["completed_blocks"] == 1


def test_primary_partial_deadline_and_secondary_storage_cost():
    error = run.base.BudgetExceeded("original"); error.add_note("primary")
    def evaluate(cs, **kwargs): error.partial_joint_round = [actual(cs[0])]; raise error
    def persist(*args): raise OSError("secondary storage")
    with pytest.raises(run.base.BudgetExceeded) as caught:
        run.run_continuation([case_pack(),case_pack()], evaluate, persist=persist)
    assert caught.value is error and error.joint_result["status"] == "incomplete"
    assert error.joint_result["cost"]["round"]["failed"] == 1 and error.joint_result["partial_round"]["completed_blocks"] == 1
    assert "primary" in error.__notes__ and any("secondary storage" in n for n in error.__notes__)


def test_completed_solver_primary_storage_and_terminal_cost():
    error = OSError("terminal")
    def store(name, *args):
        if name == "continuation-result": raise error
    with pytest.raises(OSError) as caught: simulate([[.002,0.,0.]], values=[.04,.035,.035], store=store)
    assert caught.value is error and error.abort_run
    assert error.joint_result["cost"]["solve"]["completed"] == 1
    assert error.joint_result["cost"]["round"]["completed"] == 3 and not error.joint_result["initialization_supported"]


def evidence():
    audit = {"schema":"flylab.m3.e282.root-original-audit.v1", "verdict":"valid_diagnostic_initialization_rejected",
        "reproduction_exact":True, "sparse_original_problem_exact":True,"coefficient_evaluations":6,"engine_requests":6,
        "solver_calls":1,"candidate_created":False,"dual_infeasibility_checks":[{"label":"frozen_joint_problem",
        "bounded_linear_infeasibility_supported":True,"max_possible_dual_slack_in_box":-1.}]}
    proof = {"schema":"flylab.m3.e282.peak-only-infeasibility.v1","solver_calls":0,"model_calls":0,"original_calls":0,
        "proofs":[{"label":"RF_q_peaks_and_coefficient_only","dual_nonnegative":True,
        "certified_linear_peak_lower_bound":.036805003120814854,"upper_at_strict_actual_peak":-1.,
        "scope":"Original f64 neutral +/-0.05 and coefficient/triple constraints, no norm or LTM requirements; neutral Jacobian only."}]}
    return audit,proof


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    old, original, save_old, tmp, _ = prior_tests.freeze_fixture.__wrapped__(tmp_path, monkeypatch)
    fp, fs, rp, rs = save_old()
    owned = {}
    for name in run.SOURCES:
        path=tmp/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("generated E283 source")
        owned[name]=run.base.sha256(path)
    audit,proof=evidence()
    store = prior_tests.old_tests.old_tests.store_json
    freeze={"schema":run.SCHEMA,"contract":copy.deepcopy(run.CONTRACT),"approved_sources":owned,
        "inherited_sources":{**old["inherited_sources"],**old["approved_sources"]},
        "e282_freeze":{"path":fp.name,"sha256":fs},"e282_source_review":{"path":rp.name,"sha256":rs},
        "e282_root_result":store(tmp/"audit.json",audit),"e282_rf_peak_proof":store(tmp/"proof.json",proof),
        **{k:old[k] for k in ("inputs","cases","engine_roster","packages","e280_resource_qualification")}}
    def save():
        fs=store(tmp/"e283-freeze.json",freeze)
        rs=store(tmp/"e283-review.json",{"schema":run.REVIEW_SCHEMA,"verdict":"approved","approved_sources":owned,"freeze_sha256":fs["sha256"]})
        return tmp/fs["path"],fs["sha256"],tmp/rs["path"],rs["sha256"]
    return freeze,original,save,tmp


def test_exact_nested_twentyfour_source_freeze(freeze_fixture):
    freeze,original,save,_=freeze_fixture
    actual,prior=run.validate_freeze(*save())
    assert actual==freeze and prior==original and len(run.REQUIRED_INHERITED_SOURCES)==24


@pytest.mark.parametrize("kind", ["source","inherited","contract","inputs","proof","audit","review"])
def test_provenance_fail_closed(freeze_fixture,kind):
    freeze,_,save,tmp=freeze_fixture
    if kind=="source": (tmp/run.SOURCES[0]).write_text("tampered")
    elif kind=="inherited": (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("tampered")
    elif kind=="contract": freeze["contract"]["stages"][0]["max_corrections"]=3
    elif kind=="inputs": freeze["inputs"]={"changed":True}
    elif kind=="proof": freeze["e282_rf_peak_proof"]["sha256"]="0"*64
    elif kind=="audit":
        audit,_=evidence(); audit["candidate_created"]=True
        freeze["e282_root_result"]=prior_tests.old_tests.old_tests.store_json(tmp/"audit.json",audit)
    else: freeze["e282_source_review"]["sha256"]="0"*64
    with pytest.raises(ValueError): run.validate_freeze(*save())


def test_invalid_cli_gates_original_solver_zero_and_fresh_output(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"validate_freeze",lambda *args: (_ for _ in ()).throw(ValueError("freeze")))
    monkeypatch.setattr(run,"original_run",lambda *args: pytest.fail("no original"))
    output=tmp_path/"fresh"
    args=["--freeze","x","--freeze-sha256","x","--source-review","x","--source-review-sha256","x","--output",str(output)]
    assert run.main(args)==1
    assert json.loads((output/"cost.json").read_text())["original_calls"]==0
    assert not json.loads((output/"result.json").read_text())["initialization_supported"]
    with pytest.raises(ValueError): run.main(args)

@pytest.mark.parametrize("phase", ["constructor", "solve"])
def test_real_solver_phase_failure_keeps_primary_and_attempted_cost(phase):
    original = OSError("first "+phase)
    def fail(*args): raise original
    factory = fail if phase == "constructor" else lambda *args: SimpleNamespace(solve=fail)
    def evaluate(cs, *, derivative, label): return [actual(c, derivative=derivative) for c in cs]
    saved = {}
    def store(name, arrays, receipt):
        saved[name] = copy.deepcopy((arrays, receipt))
        if name.endswith("-solve"): raise OSError("secondary publication")
    with pytest.raises(OSError) as caught:
        run.run_continuation([case_pack()], evaluate, persist=store, solver_factory=factory)
    assert caught.value is original and original.joint_result["status"] == "invalid"
    cost = original.joint_result["cost"]
    FAILURE_COST.append({"phase": phase, **{k: copy.deepcopy(cost[k]) for k in ("constructor", "solve", "certificate")},
                         "iterations": cost["iterations"], "iterations_unknown_calls": cost["iterations_unknown_calls"]})
    assert cost["constructor"]["requested"] == 1
    assert cost["constructor"]["failed"] == int(phase == "constructor")
    assert cost["solve"]["requested"] == int(phase == "solve")
    assert cost["solve"]["failed"] == int(phase == "solve")
    assert cost["iterations_unknown_calls"] == int(phase == "solve")
    assert original.joint_result["stages"][0]["corrections"] == 1
    assert saved["stage-0-joint-00-solve"][1][phase]["failed"] == 1
    assert any("secondary publication" in n for n in original.__notes__)


@pytest.mark.parametrize("mode", ["before_account", "after_account", "complete"])
def test_upper_cost_once_failure_and_no_promotion(tmp_path, monkeypatch, mode):
    meter = prior_tests.meter(tmp_path)
    monkeypatch.setattr(run, "validate_roster", lambda *args: {})
    original = OSError("first")
    result = {"cost": prior_tests.cost_row(), "run_valid": True, "initialization_supported": False}
    def driver(*args, **kwargs):
        if mode == "before_account": original.joint_result = result; raise original
        return result
    monkeypatch.setattr(run, "run_continuation", driver)
    if mode == "after_account": monkeypatch.setattr(meter, "save", lambda: (_ for _ in ()).throw(original))
    freeze = {"engine_roster": [], "inherited_sources": {}, "approved_sources": {},
              **{k: {} for k in ("e282_freeze", "e282_source_review", "e282_root_result", "e282_rf_peak_proof")}}
    if mode == "complete":
        value = run.execute_continuation(freeze, ({}, {}, []), [], lambda *args: pytest.fail("no engine"), meter)
        assert not any(value[k] for k in ("candidate_created", "full_fit_started", "body_promoted", "F6_exposed"))
    else:
        with pytest.raises(OSError) as caught: run.execute_continuation(freeze, ({}, {}, []), [], lambda *args: pytest.fail("no engine"), meter)
        assert caught.value is original and not original.joint_result["run_valid"]
    assert meter.socp_cost["solve"]["requested"] == 1 and meter.socp_cost["iterations"] == 7
    assert json.loads((meter.output/"provenance.json").read_text())["e282_valid_rejection_preserved"]


def test_reuses_frozen_engine_budget_meter_and_roster_without_global_patch():
    assert run.make_round_evaluator is run.inherited.make_round_evaluator
    assert run.JointBudget is run.inherited.JointBudget and run.JointMeter is run.inherited.JointMeter
    assert run.ScienceDeadline is run.inherited.ScienceDeadline and run.validate_roster is run.inherited.validate_roster
    assert run.CONTRACT["max_engines"] == 36 and run.CONTRACT["max_solve_calls"] == 4
    assert run.CONTRACT["max_derivative_ticks"] == 60000 and run.CONTRACT["science_wall_s"] == 40.
