"""Generated frozen runner contracts and real control-flow mocks; original0."""
import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_feasibility_restoration as generated
from tools import verify_c_banc_m3_feasibility_restoration as run


def store_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    run.base.write_json(path, value)
    return {"path": str(path.name), "sha256": run.base.sha256(path)}


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run.base, "ROOT", tmp_path)
    packages = {"generated": "CPU only"}
    monkeypatch.setattr(run, "package_versions", lambda: packages)
    source_pins = {}
    for name in (*run.SOURCES, *run.REQUIRED_INHERITED_SOURCES, "tools/extra_dependency.py"):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated frozen stand-in", encoding="utf-8")
        source_pins[name] = run.base.sha256(path)
    owned = {k: source_pins[k] for k in run.SOURCES}
    pins = {k: v for k, v in source_pins.items() if k not in owned}
    inputs = {}
    gates = {
        "native_validation": {"status": "pass", **dict.fromkeys(("counters_valid", "raw_valid", "identity_valid", "decoder_valid"), True)},
        "generated_cuda_gate": {"status": "pass"},
        "generated_e273_gate": {"status": "pass", "exactness_valid": True, "speed_valid": True,
                                "median_speedup": 2., "source_sha256": {k: pins[k] for k in run.inherited.ENGINE_SOURCES}}}
    original = {"schema": run.inherited.SCHEMA, "contract": copy.deepcopy(run.inherited.CONTRACT),
        "approved_sources": {k: pins[k] for k in run.inherited.SOURCES}, "inputs": inputs}
    for key, value in gates.items():
        inputs[key] = store_json(tmp_path/(key+".json"), value)
        original[key] = {"input": key}
    case_rows = []
    for row in run.CASES:
        case = generated.case_pack(row["alpha"])
        if row["leg"] == "RM":
            case["comparator_coefficients"] = case["before_coefficients"].copy()
            case["reference_comparator_objective"] = case["before_objective"].copy()
        path = tmp_path/(row["leg"]+".npz")
        np.savez(path, **case)
        case_rows.append({**row, "path": path.name, "sha256": run.base.sha256(path)})
    freeze = {"schema": run.SCHEMA, "contract": copy.deepcopy(run.CONTRACT), "approved_sources": owned,
        "inherited_sources": pins, "e277_freeze": store_json(tmp_path/"e277.json", original),
        "inputs": copy.deepcopy(inputs), "packages": packages, "cases": case_rows}
    def save():
        spec = store_json(tmp_path/"freeze.json", freeze)
        review = {"schema": run.REVIEW_SCHEMA, "verdict": "approved", "approved_sources": owned,
                  "freeze_sha256": spec["sha256"]}
        rs = store_json(tmp_path/"review.json", review)
        return tmp_path/"freeze.json", spec["sha256"], tmp_path/"review.json", rs["sha256"]
    return freeze, original, save, tmp_path


def test_complete_freeze_accepts_extra_pinned_dependencies(freeze_fixture):
    freeze, original, save, _ = freeze_fixture
    actual, old = run.validate_freeze(*save())
    assert actual == freeze and old == original
    assert "trial_gradient" not in run.restoration.CASE_KEYS
    assert len(run.REQUIRED_INHERITED_SOURCES) == 12


@pytest.mark.parametrize("mutation", ["source", "inherited", "missing_dependency", "package", "order", "alpha",
    "input", "e277", "contract", "review", "gate", "case_sha", "rm_comparator"])
def test_freeze_rejection_precedes_engine_creation(freeze_fixture, mutation):
    freeze, original, save, tmp = freeze_fixture
    args = save()
    if mutation == "source":
        (tmp/run.SOURCES[0]).write_text("changed")
    elif mutation == "inherited":
        (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif mutation == "missing_dependency":
        del freeze["inherited_sources"][run.REQUIRED_INHERITED_SOURCES[0]]
    elif mutation == "package":
        freeze["packages"] = {"changed": True}
    elif mutation == "order":
        freeze["cases"].reverse()
    elif mutation == "alpha":
        freeze["cases"][0]["alpha"] = .125
    elif mutation == "input":
        freeze["inputs"]["unexpected"] = freeze["inputs"]["native_validation"]
    elif mutation == "e277":
        original["contract"]["changed"] = True
        freeze["e277_freeze"] = store_json(tmp/"e277.json", original)
    elif mutation == "contract":
        freeze["contract"]["max_evaluations"] = 25
    elif mutation == "review":
        review = json.loads(args[2].read_text())
        review["verdict"] = "rejected"
        args[2].write_text(json.dumps(review))
        args = (*args[:3], run.base.sha256(args[2]))
    elif mutation == "gate":
        original["inputs"]["generated_e273_gate"] = store_json(tmp/"generated_e273_gate.json", {"status": "pass"})
        freeze["inputs"] = copy.deepcopy(original["inputs"])
        freeze["e277_freeze"] = store_json(tmp/"e277.json", original)
    elif mutation == "case_sha":
        freeze["cases"][0]["sha256"] = "0"*64
    elif mutation == "rm_comparator":
        row = freeze["cases"][4]
        with np.load(tmp/row["path"]) as z:
            case = {k: z[k] for k in z.files}
        case["reference_comparator_objective"] = np.array(.99)
        np.savez(tmp/row["path"], **case)
        row["sha256"] = run.base.sha256(tmp/row["path"])
    if mutation not in ("source", "inherited", "review"):
        args = save()
    with pytest.raises(ValueError):
        run.validate_freeze(*args)


def make_meter(tmp, clock=lambda: 0.):
    output = tmp/"output"
    output.mkdir()
    return run.RestorationMeter(output, run.ScienceDeadline(0., clock=clock))


def generated_builder(log, *, wrong_case=None, final_error=None, meter=None):
    @contextmanager
    def build(index, case, folder, budget):
        log.append(("open", index))
        budget.start_leg()
        count = [0]
        def evaluate(c):
            budget.check(new=True)
            budget.unique += 1
            budget.total_unique += 1
            row = generated.entry(c, count[0])
            count[0] += 1
            if index == wrong_case:
                row["startup_q"][0, 0] = 1e-7
            return row
        try:
            yield evaluate
        finally:
            log.append(("close", index))
            if final_error:
                raise final_error
    return build


def per_case_stub(monkeypatch):
    solve, calls = generated.stub_solver([[-.009, 0., 0.]]*12)
    monkeypatch.setattr(run.restoration, "solve_restoration", solve)
    return calls


def test_six_cases_sequential_with_one_early_success_each(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = per_case_stub(monkeypatch)
    meter, log = make_meter(tmp), []
    result = run.execute_cases(freeze, generated_builder(log), meter)
    assert result["run_valid"] and result["status"] == "complete" and result["method_supported"]
    assert len(calls) == 6 and meter.budget.total_unique == 18 and meter.budget.solver_calls == 6
    assert log == [(phase, i) for i in range(6) for phase in ("open", "close")]
    assert len(meter.cleanup_events) == 6
    assert not result["candidate_created"] and not result["F6_exposed"]


def test_later_reproduction_failure_preserves_prior_cases_and_stops(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = per_case_stub(monkeypatch)
    meter, log = make_meter(tmp), []
    with pytest.raises(run.restoration.ReproductionError) as caught:
        run.execute_cases(freeze, generated_builder(log, wrong_case=1), meter)
    assert len(calls) == 1 and meter.budget.total_unique == 5 and meter.budget.solver_calls == 1
    assert len(caught.value.partial_restoration_results) == 2
    assert caught.value.partial_restoration_results[0]["restored"]
    assert caught.value.partial_restoration_results[1]["status"] == "invalid"
    assert log == [("open", 0), ("close", 0), ("open", 1), ("close", 1)]
    assert (meter.output/"LF/correction-00-solve.npz").is_file()


def test_r1_final_storage_error_counted_once_in_upper_loop(freeze_fixture, monkeypatch):
    freeze, _, _, tmp = freeze_fixture
    calls = per_case_stub(monkeypatch)
    original = OSError("case-result write")
    persist = run._persist
    def broken(folder, name, arrays, receipt):
        if name == "case-result":
            raise original
        persist(folder, name, arrays, receipt)
    monkeypatch.setattr(run, "_persist", broken)
    meter, log = make_meter(tmp), []
    with pytest.raises(OSError) as caught:
        run.execute_cases(freeze, generated_builder(log), meter)
    assert caught.value is original and len(calls) == 1
    assert meter.budget.solver_calls == 1 and meter.budget.iterations == 2
    assert meter.budget.total_unique == 3 and len(meter.case_results) == 1
    assert meter.case_results[0]["cost"]["evaluation"]["completed"] == 3
    assert meter.case_results[0]["status"] == "invalid" and log[-1] == ("close", 0)
    receipt = json.loads((meter.output/"cost.json").read_text())
    assert receipt["fit"]["solver_calls"] == 1


@pytest.mark.parametrize("kind", ["coefficient", "trust", "correction"])
def test_r3_strict_bounds_never_request_actual_evaluation(freeze_fixture, monkeypatch, kind):
    freeze, _, _, tmp = freeze_fixture
    directions = {"coefficient": [-.01, -1e-12, 0.], "trust": [.030000000001, 0., 0.],
                  "correction": [-.020000000001, 0., 0.]}
    solve, calls = generated.stub_solver([directions[kind]]*6)
    monkeypatch.setattr(run.restoration, "solve_restoration", solve)
    meter = make_meter(tmp)
    result = run.execute_cases(freeze, generated_builder([]), meter)
    assert result["run_valid"] and not result["method_supported"] and len(calls) == 6
    assert meter.budget.total_unique == 12
    assert all(not row["corrections"][0]["actual_evaluated"] for row in result["cases"])


def test_r2_first_constructor_failure_counts_one_attempt(tmp_path, monkeypatch):
    meter = make_meter(tmp_path)
    case = generated.case_pack()
    idx = (np.arange(1), np.arange(1), np.arange(1), np.arange(7), np.arange(1))
    monkeypatch.setattr(run.base, "contract_leg", lambda *args: ({"generated": True}, idx))
    mappings = {"startup": {}, "teacher": {}}
    indices = {0: idx}
    targets = {"startup_q": np.zeros((99, 7)), "startup_ltm": np.zeros((99, 1)), "teacher_q": np.zeros((1201, 7))}
    error = OSError("first constructor")
    def factory(_):
        raise error
    build = run.make_evaluator_builder((mappings, {}, indices, targets), factory, meter)
    budget = run.RestorationBudget(clock=lambda: 0., deadline=meter.deadline)
    with pytest.raises(OSError) as caught, build(0, case, meter.output, budget):
        pytest.fail("failed first constructor cannot yield")
    assert caught.value is error and meter.engine_requests == 1 and meter.active_contexts == 0


def test_membership_arrays_convert_to_explicit_indices_and_training_only(monkeypatch):
    membership = np.array([1, 0, 1, 0, 2, 3, 4, 5])
    edge_membership = np.repeat(np.arange(6), [300, 282, 254, 261, 268, 295])
    values = {"motor": membership, "muscle": membership, "edge": edge_membership,
              "q": np.repeat(np.arange(6), 7)[::-1], "ltm": np.arange(6)[::-1],
              "teacher_q": np.zeros((3001, 42)), "startup_q": np.zeros((99, 42)), "startup_ltm": np.zeros((99, 6))}
    original = {"contexts": {name: {"mapping": {"stage_order": np.array([1, 2, 3, 4]), "context": name}}
                            for name in ("startup", "teacher")},
                "legs": {k: k for k in ("motor", "muscle", "edge", "q", "ltm")},
                "targets": {k: k for k in ("teacher_q", "startup_q", "startup_ltm")}}
    monkeypatch.setattr(run.base, "load_ref", lambda original, ref: values[ref] if isinstance(ref, str) and ref in values else ref)
    monkeypatch.setattr(run.base, "validate_static_contexts", lambda *_: None)
    monkeypatch.setattr(run.base, "validate_leg_ownership", lambda *_: None)
    monkeypatch.setattr(run.base, "validate_targets", lambda *_: None)
    monkeypatch.setattr(run.base, "training_mapping", lambda mapping: {**mapping, "training_only": True})
    mappings, _, indices, targets = run.load_inputs(original)
    np.testing.assert_array_equal(indices[0][0], [1, 3])
    np.testing.assert_array_equal(indices[0][3], np.arange(35, 42))
    np.testing.assert_array_equal(indices[0][4], [5])
    assert targets["teacher_q"].shape == (1201, 42) and mappings["teacher"]["training_only"]


@pytest.mark.parametrize("complete", [False, True])
def test_upper_original_deadline_identity_and_secondary_cost_failure(freeze_fixture, monkeypatch, complete):
    freeze, _, _, tmp = freeze_fixture
    meter = make_meter(tmp)
    original = run.base.BudgetExceeded("original")
    original.add_note("initial")
    @contextmanager
    def build(*_):
        def evaluate(c):
            if complete:
                original.completed_restoration_entry = generated.entry(c)
            raise original
        yield evaluate
    monkeypatch.setattr(meter, "save", lambda: (_ for _ in ()).throw(OSError("cost secondary")))
    with pytest.raises(run.base.BudgetExceeded) as caught:
        run.execute_cases(freeze, build, meter)
    assert caught.value is original and len(original.partial_restoration_results) == 1
    assert "initial" in original.__notes__ and any("cost secondary" in s for s in original.__notes__)
    assert original.partial_restoration_results[0]["cost"]["evaluation"]["completed"] == int(complete)


def test_cleanup_failure_preserves_primary_and_unknown_time(tmp_path, monkeypatch):
    meter = make_meter(tmp_path)
    original = OSError("pool cleanup")
    pool = SimpleNamespace(free_all_blocks=lambda: (_ for _ in ()).throw(original))
    meter.cp = SimpleNamespace(get_default_memory_pool=lambda: pool)
    monkeypatch.setattr(meter, "save", lambda: (_ for _ in ()).throw(OSError("save cleanup")))
    with pytest.raises(OSError) as caught:
        meter.cleanup()
    assert caught.value is original and meter.cleanup_events[-1]["status"] == "failed"
    assert any("save cleanup" in s for s in original.__notes__)


def test_cleanup_uses_shared_pool_between_cases(tmp_path):
    meter = make_meter(tmp_path)
    calls = []
    meter.cp = SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: calls.append(True)))
    meter.cleanup()
    assert calls == [True]


def test_deadline_and_evaluation_bounds():
    clock = [0.]
    deadline = run.ScienceDeadline(0., clock=lambda: clock[0])
    budget = run.RestorationBudget(clock=lambda: clock[0], deadline=deadline)
    budget.unique = 4
    with pytest.raises(run.base.BudgetExceeded, match="restoration_evaluations"):
        budget.check(new=True)
    budget.unique, budget.total_unique = 0, 24
    with pytest.raises(run.base.BudgetExceeded):
        budget.check(new=True)
    clock[0] = 45.000001
    with pytest.raises(run.base.BudgetExceeded) as caught:
        budget.check()
    assert caught.value.scope == "original"


def test_main_freeze_failure_does_not_open_original(monkeypatch, tmp_path):
    original = OSError("freeze")
    monkeypatch.setattr(run, "validate_freeze", lambda *_: (_ for _ in ()).throw(original))
    monkeypatch.setattr(run, "original_run", lambda *_: pytest.fail("must not call original"))
    output = tmp_path/"new-output"
    assert run.main(["--freeze", "missing", "--freeze-sha256", "x", "--source-review", "missing",
                     "--source-review-sha256", "x", "--output", str(output)]) == 1
    receipt = json.loads((output/"cost.json").read_text())
    assert receipt["original_calls"] == receipt["solver_calls"] == 0
    result = json.loads((output/"result.json").read_text())
    assert result["status"] == "invalid" and not result["method_supported"]
