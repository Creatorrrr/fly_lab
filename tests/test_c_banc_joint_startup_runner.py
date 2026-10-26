"""Generated E282 provenance, lifecycle, upper failure/cost integration; CUDA0."""
import copy
import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_startup_margin_runner as old_tests
from tests.test_c_banc_joint_startup import case_pack
from tests.test_c_banc_source_coefficient_engine import (
    generated_problem,
    prepare_problem,
)
from tools import verify_c_banc_m3_joint_startup as run


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    old, original, save_old, tmp = old_tests.freeze_fixture.__wrapped__(tmp_path, monkeypatch)
    opath, osha, rpath, rsha = save_old()
    spec = {"path": "e281-freeze.json", "sha256": osha}
    review = {"path": "e281-review.json", "sha256": rsha}
    (tmp/spec["path"]).write_bytes(opath.read_bytes()); (tmp/review["path"]).write_bytes(rpath.read_bytes())
    owned = {}
    for name in run.SOURCES:
        path = tmp/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("generated E282 source")
        owned[name] = run.base.sha256(path)
    freeze = {"schema": run.SCHEMA, "contract": copy.deepcopy(run.CONTRACT), "approved_sources": owned,
        "inherited_sources": {**old["approved_sources"], **old["inherited_sources"]},
        "e281_freeze": spec, "e281_source_review": review,
        "e281_root_result": old_tests.old_tests.store_json(tmp/"e281-root-result.json", {
            "schema": "flylab.m3.e281.root-original-audit.v1", "verdict": "valid_diagnostic_initialization_rejected", "reproduction_exact": True}),
        **{k: old[k] for k in ("inputs", "cases", "engine_roster", "packages", "e280_resource_qualification")}}
    monkeypatch.setattr(run, "package_versions", lambda: old["packages"])
    # This nested provenance fixture uses six tiny inherited cases. Production
    # global-axis/4980 validation is tested independently, not relaxed in code.
    roster_calls = []
    monkeypatch.setattr(run, "validate_roster", lambda rows, cases: roster_calls.append((rows, cases)))
    def save():
        fs = old_tests.old_tests.store_json(tmp/"e282-freeze.json", freeze)
        rs = old_tests.old_tests.store_json(tmp/"e282-review.json", {"schema": run.REVIEW_SCHEMA, "verdict": "approved",
            "approved_sources": owned, "freeze_sha256": fs["sha256"]})
        return tmp/fs["path"], fs["sha256"], tmp/rs["path"], rs["sha256"]
    return freeze, original, save, tmp, roster_calls


def test_complete_nested_freeze_checks_qualification_and_all_twenty_sources(freeze_fixture):
    freeze, original, save, _, calls = freeze_fixture
    actual, inherited = run.validate_freeze(*save())
    assert actual == freeze and inherited == original and len(run.REQUIRED_INHERITED_SOURCES) == 20
    assert len(calls) == 1 and len(calls[0][1]) == 6


@pytest.mark.parametrize("kind", ["source", "inherited_source", "contract", "packages", "inputs", "cases", "roster", "old_review", "root_result", "qualification_missing", "qualification_tampered", "qualification_identity"])
def test_bad_provenance_cannot_reach_roster_or_original(freeze_fixture, kind):
    freeze, _, save, tmp, calls = freeze_fixture
    if kind == "source": (tmp/run.SOURCES[0]).write_text("tampered")
    elif kind == "inherited_source": (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("tampered")
    elif kind == "contract": freeze["contract"]["max_corrections"] = 5
    elif kind == "packages": freeze["packages"] = {"changed": True}
    elif kind == "inputs": freeze["inputs"] = {"changed": True}
    elif kind == "cases": freeze["cases"] = list(reversed(freeze["cases"]))
    elif kind == "roster": freeze["engine_roster"] = list(reversed(freeze["engine_roster"]))
    elif kind == "old_review": freeze["e281_source_review"]["sha256"] = "0"*64
    elif kind == "root_result":
        freeze["e281_root_result"] = old_tests.old_tests.store_json(tmp/"e281-root-result.json", {"verdict": "supported"})
    elif kind == "qualification_missing": del freeze["e280_resource_qualification"]
    elif kind == "qualification_tampered": (tmp/freeze["e280_resource_qualification"]["path"]).write_text("tampered")
    else: freeze["e280_resource_qualification"] = {"path": "other.json", "sha256": "0"*64}
    with pytest.raises(ValueError): run.validate_freeze(*save())
    assert not calls


def production_schema():
    # Shape-only arrays: no conic problem, model evaluation, solver or raw input.
    edges, motors, muscles = [300, 282, 254, 261, 268, 295], [69, 63, 63, 70, 63, 63], [17, 15, 16, 17, 15, 16]
    roster, cases, cursors = [], [], [0, 0, 0]
    for index, leg in enumerate(run.base.LEGS):
        row = {"leg": leg, "samples": 101, "q_indices": list(range(index*7, (index+1)*7)), "ltm_indices": [index]}
        for slot, (name, count) in enumerate(zip(("motor", "muscle", "edge"), (motors[index], muscles[index], edges[index]), strict=True)):
            row[name+"_indices"] = list(range(cursors[slot], cursors[slot]+count)); cursors[slot] += count
        case = case_pack(); case["neutral_coefficients"] = run.base.neutral(edges[index])
        case["neutral_q_jacobian"] = np.zeros((99, 7, 3*edges[index]))
        case["neutral_ltm_jacobian"] = np.zeros((99, 1, 3*edges[index]))
        cases.append(case); roster.append(row)
    return roster, cases


def test_production_exact_global_schema_has_no_missing_or_duplicate_axes():
    roster, cases = production_schema(); axes = run.validate_roster(roster, cases)
    assert axes["parameter_slices"].shape == (6, 2) and axes["parameter_slices"][-1, 1] == 4980
    np.testing.assert_array_equal(axes["leg_major_edge_indices"], np.arange(1660))
    for key in ("motor", "muscle", "edge", "q", "ltm"):
        changed = copy.deepcopy(roster); changed[1][key+"_indices"][0] = changed[0][key+"_indices"][0]
        changed[1][key+"_indices"].sort()
        with pytest.raises(ValueError): run.validate_roster(changed, cases)
    with pytest.raises(ValueError): run.validate_roster(roster[:2], cases[:2])
    tiny = [case_pack() for _ in range(6)]
    with pytest.raises(ValueError): run.validate_roster(roster, tiny)


def meter(tmp, clock=lambda: 0.):
    output = tmp/"output"; output.mkdir()
    m = run.JointMeter(output, run.ScienceDeadline(0., clock=clock)); m.budget = run.JointBudget(m.deadline)
    return m


def test_actual_new_engine_references_vanish_between_all_36_evaluations(tmp_path):
    cases = [case_pack() for _ in range(6)]; m = meter(tmp_path); weak, derivatives = [], []
    class Engine(old_tests.FakeEngine):
        def solve(self, c, *, derivative, host):
            derivatives.append(derivative)
            return super().solve(c, derivative=derivative, host=host)
    def factory(mapping):
        assert all(ref() is None for ref in weak)
        engine = Engine(cases[mapping]); weak.append(weakref.ref(engine))
        return engine
    cleanup = m.cleanup
    def check_cleanup():
        assert all(ref() is None for ref in weak)
        cleanup()
    m.cleanup = check_cleanup
    evaluate = run.make_round_evaluator(({}, {}, list(range(6))), cases, factory, m)
    for index in range(6):
        rows = evaluate([c["neutral_coefficients"] for c in cases], derivative=index < 5, label=f"round-{index}")
        assert len(rows) == 6 and all(isinstance(r["startup_q"], np.ndarray) for r in rows)
    gc.collect(); assert all(ref() is None for ref in weak)
    assert m.engine_requests == m.engine_creation["completed"] == 36 and m.active_contexts == 0
    assert m.budget.receipt()["forward_completed_steps"] == 72000
    assert m.budget.receipt()["derivative_completed_steps"] == 60000
    assert derivatives == [True]*30+[False]*6
    assert len(m.cleanup_events) == 36 and all(e["status"] == "complete" for e in m.round_events)


@pytest.mark.parametrize("failure_index", [0, 2])
def test_failed_constructor_only_counts_actual_requests_and_saves_previous_rows(tmp_path, failure_index):
    cases = [case_pack() for _ in range(6)]; m = meter(tmp_path); original = OSError("constructor")
    def factory(index):
        if index == failure_index: raise original
        return old_tests.FakeEngine(cases[index])
    evaluate = run.make_round_evaluator(({}, {}, list(range(6))), cases, factory, m)
    with pytest.raises(OSError) as caught: evaluate([c["neutral_coefficients"] for c in cases], derivative=True, label="neutral")
    assert caught.value is original and m.engine_requests == failure_index+1
    assert m.engine_creation["completed"] == failure_index and m.engine_creation["failed"] == 1
    assert m.budget.counts["evaluations_requested"] == failure_index
    assert len(original.partial_joint_round) == failure_index and m.active_contexts == 0
    assert (m.output/"cost.json").is_file()


def test_completed_deadline_evaluation_keeps_original_object_and_complete_ticks(tmp_path):
    cases = [case_pack() for _ in range(6)]; now = [0.]; m = meter(tmp_path, clock=lambda: now[0])
    original = run.base.BudgetExceeded("original"); original.add_note("first deadline")
    def check():
        if now[0] > 40: raise original
    m.deadline.check = check
    def factory(index):
        engine = old_tests.FakeEngine(cases[index]); host_copy = engine.host_copy
        def copy(value, *, label):
            now[0] = 40.001; return host_copy(value, label=label)
        engine.host_copy = copy; return engine
    evaluate = run.make_round_evaluator(({}, {}, list(range(6))), cases, factory, m)
    with pytest.raises(run.base.BudgetExceeded) as caught: evaluate([c["neutral_coefficients"] for c in cases], derivative=True, label="neutral")
    assert caught.value is original and len(original.partial_joint_round) == 1
    assert m.engine_requests == 1 and m.budget.counts["evaluations_completed"] == 1
    assert m.budget.counts["forward_completed_steps"] == m.budget.counts["derivative_completed_steps"] == 2000
    assert (m.output/"neutral"/"LF"/"actual.npz").is_file()


def test_schema_integrates_real_prepare_problem_without_constructing_cuda():
    raw = generated_problem(100)
    for key in ("edge_ids", "edge_post", "edge_source", "weights"): raw[key] = raw[key][:1]
    raw["moment"] = np.vstack((raw["moment"], np.zeros((4, 2))))
    raw["rest"] = np.r_[raw["rest"], np.zeros(4)]
    prepared = prepare_problem(raw)
    case = case_pack(); engine = old_tests.FakeEngine(case)
    engine.problem = prepared
    evaluator = run.inherited.StartupEvaluator(engine, case, run.JointBudget(run.ScienceDeadline(0., clock=lambda: 0.)), None)
    assert evaluator.engine.problem["params"] == 3 and "parameters" not in evaluator.engine.problem
    assert prepared["samples"] == 101 and prepared["stages"] == 8000


def cost_row():
    cost = {k: {**run.joint.metering.phase(), "requested": 1, "completed": 1} for k in ("constructor", "solve", "certificate")}
    return {**cost, "iterations": 7, "iterations_unknown_calls": 0}


@pytest.mark.parametrize("mode", ["terminal_storage", "deadline", "secondary_cost", "after_aggregation", "complete"])
def test_upper_loop_aggregates_primary_partial_cost_exactly_once(tmp_path, monkeypatch, mode):
    m = meter(tmp_path); original = run.base.BudgetExceeded("original") if mode == "deadline" else OSError("storage")
    original.add_note("first")
    monkeypatch.setattr(run, "validate_roster", lambda *_: {})
    result = {"cost": cost_row(), "run_valid": mode == "complete", "initialization_supported": False}
    def fake_run(*args, **kwargs):
        if mode not in ("complete", "after_aggregation"): original.joint_result = result; raise original
        return result
    monkeypatch.setattr(run.joint, "run_joint", fake_run)
    if mode == "secondary_cost": monkeypatch.setattr(m, "save", lambda: (_ for _ in ()).throw(OSError("secondary cost")))
    if mode == "after_aggregation": monkeypatch.setattr(m, "save", lambda: (_ for _ in ()).throw(original))
    if mode == "complete":
        actual = run.execute_joint({"engine_roster": []}, ({}, {}, []), [], lambda *_: pytest.fail("no engine"), m)
        assert not actual["candidate_created"] and not actual["full_fit_started"] and not actual["F6_exposed"]
    else:
        with pytest.raises(type(original)) as caught:
            run.execute_joint({"engine_roster": []}, ({}, {}, []), [], lambda *_: pytest.fail("no engine"), m)
        assert caught.value is original and "first" in original.__notes__
        if mode == "secondary_cost": assert any("secondary cost" in note for note in original.__notes__)
    assert m.socp_cost["solve"]["requested"] == 1 and m.socp_cost["iterations"] == 7


@pytest.mark.parametrize("mode", ["constructor", "reproduction"])
def test_real_upper_path_halts_and_preserves_partial_round_before_any_solver(tmp_path, monkeypatch, mode):
    m = meter(tmp_path); cases = [case_pack() for _ in range(6)]
    monkeypatch.setattr(run, "validate_roster", lambda *_: {})
    original = OSError("third engine")
    if mode == "reproduction": cases[0]["neutral_startup_q"][0, 0] = 1e-6
    def factory(index):
        if mode == "constructor" and index == 2: raise original
        return old_tests.FakeEngine(cases[index])
    with pytest.raises(OSError if mode == "constructor" else run.joint.metering.ReproductionError) as caught:
        run.execute_joint({"engine_roster": []}, ({}, {}, list(range(6))), cases, factory, m)
    result = caught.value.joint_result
    assert not result["run_valid"] and m.socp_cost["solve"]["requested"] == 0
    assert m.engine_requests == (3 if mode == "constructor" else 6)
    if mode == "constructor":
        assert caught.value is original and result["partial_round"]["completed_blocks"] == 2
        assert result["cost"]["round"]["failed"] == 1
        assert (m.output/"neutral-partial-01.npz").is_file()
    else:
        assert len(result["reproduction"]) == 6 and result["cost"]["round"]["completed"] == 1


def test_budget_limits_and_science_deadline_are_exact():
    now = [0.]; budget = run.JointBudget(run.ScienceDeadline(0., clock=lambda: now[0]))
    for _ in range(30): budget.request(True)
    with pytest.raises(run.base.BudgetExceeded): budget.request(True)
    for _ in range(6): budget.request(False)
    with pytest.raises(run.base.BudgetExceeded): budget.request(False)
    assert budget.counts["forward_requested_steps"] == 72000 and budget.counts["derivative_requested_steps"] == 60000
    now[0] = 40.000001
    with pytest.raises(run.base.BudgetExceeded) as caught: budget.deadline.check()
    assert caught.value.scope == "original" and budget.deadline.receipt()["supervisor_wall_limit_s"] == 55.


def test_cleanup_primary_storage_secondary_and_no_engine_recreation(tmp_path, monkeypatch):
    m = meter(tmp_path); primary = OSError("pool")
    m.cp = SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: (_ for _ in ()).throw(primary)))
    monkeypatch.setattr(m, "save", lambda: (_ for _ in ()).throw(OSError("secondary save")))
    with pytest.raises(OSError) as caught: m.cleanup()
    assert caught.value is primary and any("secondary save" in n for n in primary.__notes__)
    assert m.engine_requests == 0


def test_invalid_cli_freeze_cost_is_original_solver_zero(tmp_path, monkeypatch):
    error = OSError("invalid freeze")
    monkeypatch.setattr(run, "validate_freeze", lambda *_: (_ for _ in ()).throw(error))
    monkeypatch.setattr(run, "original_run", lambda *_: pytest.fail("no original"))
    output = tmp_path/"fresh"
    assert run.main(["--freeze", "x", "--freeze-sha256", "x", "--source-review", "x", "--source-review-sha256", "x", "--output", str(output)]) == 1
    assert json.loads((output/"cost.json").read_text())["original_calls"] == 0
    assert not json.loads((output/"result.json").read_text())["initialization_supported"]
