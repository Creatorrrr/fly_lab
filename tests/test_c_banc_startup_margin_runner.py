"""Generated E281 freeze/evaluator/cleanup integration; original and CUDA0."""
import copy
import gc
import json
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_feasibility_restoration_runner as old_tests
from tests import test_c_banc_startup_margin as generated
from tools import verify_c_banc_m3_startup_margin as run


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    old, original, save_old, tmp = old_tests.freeze_fixture.__wrapped__(tmp_path, monkeypatch)
    old_path, old_sha, *_ = save_old()
    pins = {**old["approved_sources"], **old["inherited_sources"]}
    owned = {}
    for name in run.SOURCES:
        path = tmp/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("generated E281 source")
        owned[name] = run.base.sha256(path)
    cases, roster = [], []
    case = generated.case_pack()
    quota = run.margin.quota_rows([case["neutral_startup_q"][1:100]-case["target_q"]]*6)
    for leg, qrow in zip(run.base.LEGS, quota, strict=True):
        pack = copy.deepcopy(case)
        pack["q_sse_quota"] = np.array(qrow["q_sse_quota"])
        path = tmp/("e281-"+leg+".npz"); np.savez(path, **pack)
        cases.append({"leg": leg, "path": path.name, "sha256": run.base.sha256(path),
                      "neutral_sse": qrow["neutral_sse"], "share": qrow["share"]})
        roster.append({"leg": leg, "samples": 101, "motor_indices": [0], "muscle_indices": [0],
                       "edge_indices": [0], "q_indices": list(range(7)), "ltm_indices": [0]})
    root_result = {"schema": "flylab.m3.e280.root-original-audit.v1", "verdict": "valid_method_supported_for_next_fit_design",
                   "reproduction_exact": True, "cases": [{"leg": leg, "accepted": True, "beat_comparator": True} for leg in run.base.LEGS]}
    freeze = {"schema": run.SCHEMA, "contract": copy.deepcopy(run.CONTRACT), "approved_sources": owned,
        "inherited_sources": pins, "e280_freeze": {"path": old_path.name, "sha256": old_sha},
        "e280_root_result": old_tests.store_json(tmp/"root-result.json", root_result), "inputs": old["inputs"],
        "packages": old["packages"], "cases": cases, "engine_roster": roster}
    lifetime = {"schema": "flylab.m3.e280.root-retrospective-lifetime.v1",
                "source_sha256": pins["tools/verify_c_banc_m3_feasibility_restoration.py"]}
    qualification = {"schema": "flylab.m3.e280.resource-qualification.v1",
        "verdict": "numerical_result_retained_actual_lifetime_assurance_withdrawn",
        "numeric_evidence_valid": True, "actual_live_engine_bound_verified": False,
        "future_reuse_requires_explicit_lifetime_release": True, "numeric_review": freeze["e280_root_result"],
        "lifetime_review": old_tests.store_json(tmp/"lifetime-review.json", lifetime)}
    freeze["e280_resource_qualification"] = old_tests.store_json(tmp/"qualification.json", qualification)
    monkeypatch.setattr(run, "package_versions", lambda: old["packages"])
    def save():
        spec = old_tests.store_json(tmp/"new-freeze.json", freeze)
        review = {"schema": run.REVIEW_SCHEMA, "verdict": "approved", "approved_sources": owned, "freeze_sha256": spec["sha256"]}
        rs = old_tests.store_json(tmp/"new-review.json", review)
        return tmp/"new-freeze.json", spec["sha256"], tmp/"new-review.json", rs["sha256"]
    return freeze, original, save, tmp


def test_valid_freeze_exact_quota_roster_and_pins(freeze_fixture):
    freeze, original, save, _ = freeze_fixture
    actual, old = run.validate_freeze(*save())
    assert actual == freeze and old == original and len(run.REQUIRED_INHERITED_SOURCES) == 16


@pytest.mark.parametrize("kind", ["source", "old_source", "packages", "case_sha", "case_order", "roster", "quota", "share", "root_result", "contract", "input"])
def test_invalid_freeze_cannot_start_an_engine(freeze_fixture, kind):
    freeze, _, save, tmp = freeze_fixture
    if kind == "source":
        (tmp/run.SOURCES[0]).write_text("changed")
    elif kind == "old_source":
        (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif kind == "packages":
        freeze["packages"] = {"different": True}
    elif kind == "case_sha":
        freeze["cases"][0]["sha256"] = "0"*64
    elif kind == "case_order":
        freeze["cases"].reverse()
    elif kind == "roster":
        freeze["engine_roster"][0]["samples"] = 1201
    elif kind == "quota":
        row = freeze["cases"][0]
        with np.load(tmp/row["path"]) as z: pack = {k: z[k] for k in z.files}
        pack["q_sse_quota"] *= .99
        np.savez(tmp/row["path"], **pack); row["sha256"] = run.base.sha256(tmp/row["path"])
    elif kind == "share":
        freeze["cases"][0]["share"] += 1e-10
    elif kind == "root_result":
        freeze["e280_root_result"] = old_tests.store_json(tmp/"root-result.json", {"verdict": "rejected"})
    elif kind == "contract":
        freeze["contract"]["max_engines"] = 7
    else:
        freeze["inputs"] = {"different": True}
    with pytest.raises(ValueError):
        run.validate_freeze(*save())


@pytest.mark.parametrize("kind", ["missing", "tampered", "schema", "verdict", "numeric_invalid", "live_verified",
    "future_release_false", "numeric_identity", "lifetime_missing", "lifetime_tampered", "lifetime_schema", "lifetime_source"])
def test_e280_qualification_is_required_and_bound_to_numeric_and_frozen_source(freeze_fixture, kind):
    freeze, _, save, tmp = freeze_fixture
    qpath = tmp/freeze["e280_resource_qualification"]["path"]
    qualification = json.loads(qpath.read_text())
    if kind == "missing":
        del freeze["e280_resource_qualification"]
    elif kind == "tampered":
        qpath.write_text("changed after pinning")
    elif kind.startswith("lifetime_"):
        spec = qualification["lifetime_review"]
        lpath = tmp/spec["path"]
        lifetime = json.loads(lpath.read_text())
        if kind == "lifetime_missing":
            del qualification["lifetime_review"]
        elif kind == "lifetime_tampered":
            lpath.write_text("changed after pinning")
        else:
            lifetime["schema" if kind == "lifetime_schema" else "source_sha256"] = "wrong frozen identity"
            qualification["lifetime_review"] = old_tests.store_json(lpath, lifetime)
        freeze["e280_resource_qualification"] = old_tests.store_json(qpath, qualification)
    else:
        values = {"schema": ("schema", "wrong"), "verdict": ("verdict", "approved"),
                  "numeric_invalid": ("numeric_evidence_valid", False), "live_verified": ("actual_live_engine_bound_verified", True),
                  "future_release_false": ("future_reuse_requires_explicit_lifetime_release", False),
                  "numeric_identity": ("numeric_review", {"path": "different.json", "sha256": freeze["e280_root_result"]["sha256"]})}
        key, value = values[kind]
        qualification[key] = value
        freeze["e280_resource_qualification"] = old_tests.store_json(qpath, qualification)
    with pytest.raises(ValueError):
        run.validate_freeze(*save())


def meter(tmp, clock=lambda: 0.):
    output = tmp/"output"; output.mkdir()
    return run.StartupMeter(output, run.ScienceDeadline(0., clock=clock))


def builder(log, *, mismatch=None, deadline_error=None, completed=False):
    @contextmanager
    def build(index, case, folder, budget):
        budget.leg_evaluations = 0; log.append(("open", index))
        count = [0]
        def evaluate(c, *, derivative):
            budget.request(derivative)
            row = generated.entry(c, count[0], derivative=derivative)
            count[0] += 1
            if mismatch == index:
                row["startup_q"][0, 0] = 1e-6
            if deadline_error:
                if completed:
                    deadline_error.completed_margin_entry = row
                raise deadline_error
            budget.counts["evaluations_completed"] += 1
            return row
        try:
            yield evaluate
        finally:
            log.append(("close", index))
    return build


def stub_each(monkeypatch, *, valid=True, directions=None):
    solve, calls = generated.stub(directions or [[.002, 0., 0.]]*12, valid=valid)
    monkeypatch.setattr(run.margin, "solve_margin", solve)
    return calls


def test_six_successful_legs_sequential_and_f32_only_without_derivative(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = stub_each(monkeypatch); m, log = meter(tmp), []
    result = run.execute_legs(freeze, builder(log), m)
    assert result["run_valid"] and result["initialization_supported"] and result["global_margin_gate"]["pass"]
    assert m.budget.counts["evaluations_requested"] == 18 and m.socp_cost["solve"]["requested"] == 6
    assert m.budget.counts["derivative_requested_steps"] == 24000
    assert log == [(phase, i) for i in range(6) for phase in ("open", "close")] and len(calls) == 6
    assert len(m.cleanup_events) == 6 and not result["full_fit_started"] and not result["candidate_created"]


def test_scientific_rejection_continues_fixed_remaining_legs(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = stub_each(monkeypatch, valid=False); m, log = meter(tmp), []
    result = run.execute_legs(freeze, builder(log), m)
    assert result["run_valid"] and not result["initialization_supported"] and len(calls) == 6
    assert m.budget.counts["evaluations_requested"] == 6 and len(log) == 12


def test_reproduction_error_stops_remaining_and_retains_earlier_cost(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = stub_each(monkeypatch); m, log = meter(tmp), []
    with pytest.raises(run.margin.restoration.ReproductionError) as caught:
        run.execute_legs(freeze, builder(log, mismatch=1), m)
    assert len(calls) == 1 and len(caught.value.partial_margin_results) == 2
    assert caught.value.partial_margin_results[0]["f32_success"]
    assert m.socp_cost["solve"]["requested"] == 1 and len(log) == 4


def test_final_storage_primary_and_exact_once_upper_cost(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    stub_each(monkeypatch); m = meter(tmp); original = OSError("leg-result")
    persist = run.persist
    def fail(folder, name, arrays, receipt):
        if name == "leg-result": raise original
        return persist(folder, name, arrays, receipt)
    monkeypatch.setattr(run, "persist", fail)
    with pytest.raises(OSError) as caught:
        run.execute_legs(freeze, builder([]), m)
    assert caught.value is original and original.abort_run
    assert m.socp_cost["solve"]["requested"] == 1 and m.socp_cost["iterations"] == 2
    assert len(original.partial_margin_results) == 1 and not original.partial_margin_results[0]["run_valid"]


@pytest.mark.parametrize("completed", [False, True])
def test_upper_deadline_identity_notes_partial_cost_and_secondary_failure(freeze_fixture, monkeypatch, completed):
    freeze, _, _, tmp = freeze_fixture
    m = meter(tmp); original = run.base.BudgetExceeded("original"); original.add_note("initial")
    monkeypatch.setattr(m, "save", lambda: (_ for _ in ()).throw(OSError("secondary cost")))
    with pytest.raises(run.base.BudgetExceeded) as caught:
        run.execute_legs(freeze, builder([], deadline_error=original, completed=completed), m)
    assert caught.value is original and "initial" in original.__notes__
    assert original.partial_margin_results[0]["cost"]["evaluation"]["completed"] == int(completed)
    assert any("secondary cost" in note for note in original.__notes__)


class FakeEngine:
    def __init__(self, case):
        self.case, self.problem = case, {"samples": 101, "stages": 8000, "params": 3, "gpu_bytes_estimate": 1024}
        self.receipt, self.last_receipt, self.host_copy_events = {}, {}, []
        self.derivatives, self.labels = [], []
    def solve(self, c, *, derivative, host):
        assert host is False
        self.derivatives.append(derivative)
        row = generated.entry(c, derivative=derivative)
        receipt = {"completed_steps": 2000, "derivative_completed_steps": 2000*int(derivative)}
        self.last_receipt = receipt
        return {"q": row["startup_q"], "ltm": row["startup_ltm"], "receipt": receipt}
    def host_copy(self, value, *, label):
        self.labels.append(label)
        return value.copy()
    def command_jacobian(self, result, controls, *, host):
        assert host is True
        np.testing.assert_array_equal(controls, np.arange(1, 100))
        return {"q": self.case["neutral_q_jacobian"], "ltm": self.case["neutral_ltm_jacobian"]}


def test_actual_startup_evaluator_schema_selective_copy_and_no_teacher(tmp_path):
    case = generated.case_pack(); m = meter(tmp_path)
    budget = run.StartupBudget(m.deadline); engine = FakeEngine(case)
    evaluator = run.StartupEvaluator(engine, case, budget, m.output)
    row = evaluator.evaluate(case["neutral_coefficients"], derivative=True)
    final = evaluator.evaluate(case["neutral_coefficients"], derivative=False)
    assert "startup_q_jacobian" in row and "startup_q_jacobian" not in final
    assert engine.derivatives == [True, False] and len(engine.labels) == 4
    assert budget.receipt()["evaluations_completed"] == 2
    assert budget.receipt()["forward_completed_steps"] == 4000 and budget.receipt()["derivative_completed_steps"] == 2000
    assert (m.output/"eval-000.npz").is_file() and (m.output/"eval-001.npz").is_file()


def test_evaluator_completed_deadline_preserves_raw_and_ticks(tmp_path):
    case = generated.case_pack(); now = [0.]; m = meter(tmp_path, lambda: now[0]); engine = FakeEngine(case)
    copy = engine.host_copy
    def host_copy(value, *, label):
        now[0] = 36.
        return copy(value, label=label)
    engine.host_copy = host_copy
    budget = run.StartupBudget(m.deadline); evaluator = run.StartupEvaluator(engine, case, budget, m.output)
    with pytest.raises(run.base.BudgetExceeded) as caught:
        evaluator.evaluate(case["neutral_coefficients"], derivative=True)
    assert caught.value.scope == "original" and hasattr(caught.value, "completed_margin_entry")
    assert budget.counts["evaluations_completed"] == 1 and budget.counts["forward_completed_steps"] == 2000


def test_first_constructor_failure_counts_only_attempted_startup(tmp_path):
    m = meter(tmp_path); original = OSError("constructor")
    def factory(_): raise original
    build = run.make_builder(({}, {}, [{}]), factory, m)
    with pytest.raises(OSError) as caught, build(0, generated.case_pack(), m.output, run.StartupBudget(m.deadline)):
        pytest.fail("no evaluator after constructor failure")
    assert caught.value is original and m.engine_requests == 1 and m.active_contexts == 0


def test_successful_scope_drops_evaluator_before_pool_cleanup(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    stub_each(monkeypatch); m = meter(tmp); weak = []
    @contextmanager
    def build(index, case, folder, budget):
        class Owner:
            def __init__(self): self.counter = 0
            def evaluate(self, c, *, derivative):
                row = generated.entry(c, self.counter, derivative=derivative); self.counter += 1
                return row
        owner = Owner(); weak.append(weakref.ref(owner))
        yield owner.evaluate
    cleanup = m.cleanup
    def check_cleanup():
        gc.collect()
        assert weak[-1]() is None
        cleanup()
    monkeypatch.setattr(m, "cleanup", check_cleanup)
    assert run.execute_legs(freeze, build, m)["initialization_supported"]


def test_oversized_memory_failure_unwinds_meter_without_losing_primary(tmp_path):
    m = meter(tmp_path); engine = FakeEngine(generated.case_pack()); engine.problem["gpu_bytes_estimate"] = 5*1024**3
    with pytest.raises(ValueError): m.memory(engine, "oversized")
    assert m.active_contexts == 0 and m.current_estimate == 0


def test_startup_load_never_requests_teacher_target_or_mapping(monkeypatch):
    axes = (np.array([2]), np.array([3]), np.array([4]), np.arange(7), np.array([5]))
    roster = [{"leg": leg, "samples": 101, **{k+"_indices": a.tolist() for k, a in
              zip(("motor", "muscle", "edge", "q", "ltm"), axes, strict=True)}} for leg in run.base.LEGS]
    original = {"contexts": {"startup": {"mapping": {"stage_order": "order", "stage_total": "startup"}},
                             "teacher": {"mapping": {"stage_total": "forbidden"}}}, "legs": {"q": "qleg"}}
    calls = []
    def load(_, ref):
        assert ref != "forbidden"; calls.append(ref)
        return np.array(["start", "half1", "half2", "end"]) if ref == "order" else np.zeros(1)
    monkeypatch.setattr(run.base, "load_ref", load)
    monkeypatch.setattr(run.base, "validate_leg_ownership", lambda *_: None)
    monkeypatch.setattr(run.base, "contract_leg", lambda *args: ({"startup_only": True}, axes))
    _, _, locals = run.load_startup_inputs(original, roster)
    assert len(locals) == 6 and calls == ["order", "startup", "qleg"]


def test_cleanup_preserves_primary_and_secondary_storage_error(tmp_path, monkeypatch):
    m = meter(tmp_path); original = OSError("pool")
    m.cp = SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: (_ for _ in ()).throw(original)))
    monkeypatch.setattr(m, "save", lambda: (_ for _ in ()).throw(OSError("secondary cleanup")))
    with pytest.raises(OSError) as caught: m.cleanup()
    assert caught.value is original and any("secondary cleanup" in s for s in original.__notes__)


def test_science_and_tick_reservation_limits():
    now = [0.]; budget = run.StartupBudget(run.ScienceDeadline(0., clock=lambda: now[0]))
    for _ in range(4): budget.request(True)
    with pytest.raises(run.base.BudgetExceeded): budget.request(False)
    budget.leg_evaluations = 0; budget.counts["evaluations_requested"] = 24
    with pytest.raises(run.base.BudgetExceeded): budget.request(False)
    now[0] = 35.000001
    with pytest.raises(run.base.BudgetExceeded) as caught: budget.deadline.check()
    assert caught.value.scope == "original"


def test_main_invalid_freeze_has_original_solver_zero(monkeypatch, tmp_path):
    error = OSError("freeze invalid")
    monkeypatch.setattr(run, "validate_freeze", lambda *_: (_ for _ in ()).throw(error))
    monkeypatch.setattr(run, "original_run", lambda *_: pytest.fail("no original after invalid freeze"))
    output = tmp_path/"fresh"
    assert run.main(["--freeze", "x", "--freeze-sha256", "x", "--source-review", "x", "--source-review-sha256", "x", "--output", str(output)]) == 1
    assert json.loads((output/"cost.json").read_text())["original_calls"] == 0
    assert not json.loads((output/"result.json").read_text())["initialization_supported"]
