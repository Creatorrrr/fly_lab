"""E291 generated CPU qualification; fake body only, no solver/model/CUDA."""
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_recurrent_resource_body as probe


def windows_error(code):
    error = PermissionError("injected publication conflict")
    error.winerror = code
    return error


def test_root_publication_transient_preserves_value_and_cost(tmp_path):
    target = tmp_path / "ledger.json"
    value = {"control": 9, "measured": .045, "frozen": [1, 2, 3]}
    before = copy.deepcopy(value)
    calls, waits = [], []
    def writer(path, received):
        calls.append(copy.deepcopy(received))
        if len(calls) <= 2:
            raise windows_error(32)
        probe.receipts.write_receipt(path, received)
    probe.bounded_write_receipt(target, value, writer=writer, sleeper=waits.append)
    assert calls == [before] * 3 and value == before
    assert json.loads(target.read_text("utf8")) == before
    assert waits == [.005, .01]
    events = [json.loads(line) for line in target.with_name(target.name + ".write-conflicts.jsonl").read_text("utf8").splitlines()]
    assert [event["attempt"] for event in events] == [1, 2]
    assert sum(event["delay_s"] for event in events) == .015


def test_root_publication_permanent_bounded_and_first_error_preserved(tmp_path):
    target = tmp_path / "ledger.json"
    errors, waits = [], []
    def writer(_path, _value):
        error = windows_error(5); errors.append(error); raise error
    with pytest.raises(PermissionError) as caught:
        probe.bounded_write_receipt(target, {}, writer=writer, sleeper=waits.append)
    assert caught.value is errors[0] and len(errors) == 6
    assert waits == [.005, .01, .02, .04, .08]
    events = [json.loads(line) for line in target.with_name(target.name + ".write-conflicts.jsonl").read_text("utf8").splitlines()]
    assert len(events) == 6 and not events[-1]["retry"]
    assert not target.exists()


@pytest.mark.parametrize("error", [PermissionError("POSIX permissions"), windows_error(87), OSError("disk unavailable")])
def test_root_publication_other_errors_no_retry(tmp_path, error):
    calls, waits = [], []
    def writer(_path, _value):
        calls.append(1); raise error
    with pytest.raises(type(error)) as caught:
        probe.bounded_write_receipt(tmp_path / "ledger.json", {}, writer=writer, sleeper=waits.append)
    assert caught.value is error and len(calls) == 1 and not waits


@pytest.mark.skipif(probe.os.name != "nt", reason="Actual Windows file-sharing semantics")
def test_root_real_windows_read_handle_conflict_recovers(tmp_path):
    target = tmp_path / "ledger.json"
    target.write_text('{"old": true}', encoding="utf8")
    handle = target.open("rb")
    waits = []
    try:
        with pytest.raises(PermissionError) as caught:
            probe.receipts.write_receipt(target, {"new": True})
        assert caught.value.winerror in (5, 32, 33)
        assert json.loads(target.read_text("utf8")) == {"old": True}
        def release(delay):
            waits.append(delay); handle.close()
        probe.bounded_write_receipt(target, {"new": True}, sleeper=release)
        assert waits == [.005]
        assert json.loads(target.read_text("utf8")) == {"new": True}
    finally:
        handle.close()


def metrics(candidate=False):
    return {
        "health": {
            "minimum_native_upright": 1.0,
            "nonfoot_contact_count": 0,
            **{
                key: True
                for key in (
                    "healthy",
                    "finite",
                    "complete_1p5s",
                    "clock_exact",
                    "observed_clock_valid",
                )
            },
        },
        "forward": {
            "available": True,
            "window_samples": 200,
            "last_included_control": 299,
            "fixed_initial_heading_forward_mm": probe.BASELINE_FORWARD_MM,
        },
        "slip": {
            "available": True,
            "load_weighted_mm_s": 1.0,
            "missing_selected_slip": 0,
            "support_samples": 20,
            "support_load_sum_bw": 1.0,
        },
        "gait": {
            "window": {
                "start_s": 0.5,
                "end_s": 1.5,
                "sampled_duration_s": 1.0,
                "bouts": [{"start_s": 0.5, "end_s": 1.2, "samples": 141}]
                if candidate
                else [],
            },
            "events": {
                leg: {
                    "complete_cycles": int(candidate),
                    "cycles": [{"aep_s": 0.6, "pep_s": 0.8, "next_aep_s": 1.0}]
                    if candidate
                    else [],
                }
                for leg in probe.LEGS
            },
        },
    }


COUNTS = {"fake_sessions": 0, "fake_controls": 0, "fake_neural_ticks": 0,
          "fake_native_samples": 0, "fake_partial_component_snapshots": 0,
          "fake_partial_checkpoint_save_attempts": 0,
          "original_constructors": 0, "CUDA": 0, "physics": 0, "fit": 0}
BEGAN = time.perf_counter()


class FakeHealthStop(RuntimeError):
    pass


class FakeSession:
    def __init__(self, arm, owner):
        COUNTS["fake_sessions"] += 1
        self.arm, self.owner = arm, owner
        self.control_tick, self.closed, self.fault = 0, False, False
        self.network = SimpleNamespace(tick=0)
        self.body = SimpleNamespace(fault=False, physics_time=lambda: self.time_s)
        self.time_s = 0.
        self.close_calls = 0

    def advance(self, controls):
        assert controls == 1
        COUNTS["fake_controls"] += 1
        self.network.tick += 20
        COUNTS["fake_neural_ticks"] += 20
        count = 27 if self.arm == "candidate" and self.owner.mode in ("health", "partial") else 50
        for _ in range(count):
            self.time_s += .0001
            self.native.append((self.time_s, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.))
            COUNTS["fake_native_samples"] += 1
        if self.arm == "candidate" and self.owner.mode in ("health", "partial"):
            raise FakeHealthStop("measured fake health stop")
        if self.arm == "candidate" and self.owner.mode == "advance":
            raise RuntimeError("first fake advance error")
        self.control_tick += 1

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.owner.mode == "close":
            raise RuntimeError("fake close error")


class FakeDeps:
    health_stop = FakeHealthStop

    def __init__(self, mode="pass"):
        self.mode, self.sessions = mode, []
        self.saved_partial = 0
        self.journal_records = []

    def load(self, _freeze):
        return {"ids": np.arange(2, dtype=np.int64), "baseline_short": metrics()}

    def construct(self, arm, data, ledger, persist, on_return):
        if self.mode == "constructor" and arm == "candidate":
            raise RuntimeError("fake constructor error")
        session = FakeSession(arm, self)
        self.sessions.append(session)
        on_return(session)
        if self.mode == "post_return":
            raise OSError("fake post-return receipt error")
        return session

    def initial(self, session, data, arm):
        assert session.control_tick == session.network.tick == 0
        if self.mode == "initial":
            raise ValueError("whole initial mismatch")

    def install(self, session, records):
        session.native = records
        return lambda: None

    def capture(self, session, raw):
        row = {"control_tick": session.control_tick, "time_s": session.control_tick * .005,
               "joint_targets_rad": np.zeros(42), "tendon_inputs": {f"{leg}_tarsus": 0. for leg in probe.LEGS}}
        raw["rows"].append(row)
        if self.mode == "readout" and session.arm == "candidate" and session.control_tick == 2:
            raise RuntimeError("fake readout error")
        raw["controls"].append({"control_tick": session.control_tick})
        raw["recruitment"].append({"complete": True})
        raw["activity"].append({"control_tick": session.control_tick, "complete": True})

    def compare_neutral(self, raw, data):
        if self.mode == "neutral":
            raise ValueError("fake neutral mismatch")
        assert len(raw["rows"]) == 2 and len(raw["native"]) == 50
        return {"exact": True, "whole_end_exact": self.mode != "whole"}

    def startup(self, raw, data):
        assert len(raw["rows"]) == 101
        return {"strict_E141_startup_screen": self.mode != "startup"}

    def save_state(self, path, session):
        if self.mode == "state" and path.name == "final":
            raise OSError("fake state storage error")
        if self.mode in ("health", "partial") and path.name == "final" and session.arm == "candidate":
            raise ValueError("healthy snapshot only")
        self.write(path.with_suffix(".json"), {"control": session.control_tick})

    def save_partial(self, path, session):
        self.saved_partial += 1
        if self.mode == "partial":
            raise OSError("partial checkpoint failure")

    def journal(self, path):
        records = self.journal_records
        return SimpleNamespace(append=lambda value: records.append(copy.deepcopy(value)), close=lambda: None)

    def advance(self, session, arm, index, clock, journal):
        return probe.protocol.advance_control(session, arm, index, clock, journal)

    def readout_receipt(self, journal, arm, index, event, complete, error=None):
        probe.protocol.record_readout(journal, arm, index, event, complete=complete, error=error)

    def write(self, path, value):
        probe.receipts.write_receipt(path, value)
        if self.mode == "publication" and path.name == "output-manifest.json":
            raise OSError("fake final publication error")

    def save_raw(self, folder, raw, data, clock):
        self.write(folder / "fake-raw.json", {"rows": len(raw["rows"]), "native": len(raw["native"])})
        if self.mode == "raw":
            raise OSError("fake raw storage error")
        return []

    def short_metrics(self, raw, folder, clock, data):
        assert len(raw["rows"]) == 301 and len(raw["native"]) == 15000
        return metrics(self.mode != "short")



def execute(tmp_path, mode="pass", budget=None, cleanup=None):
    deps = FakeDeps(mode)
    result, error = probe.execute({}, tmp_path / "out", deps=deps,
                                  budget=budget or (lambda: None), cleanup=cleanup)
    return deps, result, error


def test_continuous_candidate_and_fixed_scientific_cost(tmp_path):
    deps, result, error = execute(tmp_path)
    assert error is None and result["run_valid"] and result["neutral_exact"]
    assert result["actual_startup_pass"] and result["short_gait_pass"]
    assert result["next_separate_10s_eligible"]
    assert [s.control_tick for s in deps.sessions] == [1, 300]
    assert all(s.closed and s.close_calls == 1 for s in deps.sessions)
    assert result["cost"]["scientific_neural_ticks"] == 6020
    assert result["cost"]["scientific_physics_steps"] == 15050
    for key in ("adopted", "candidate_adopted", "natural_gait_complete", "F1_F8_complete", "F2_complete", "F6_exposed"):
        assert result[key] is False
    assert "prediction" not in result["arms"]["candidate"]
    assert not list((tmp_path / "out").rglob("*prediction*"))


@pytest.mark.parametrize("mode", ["neutral", "whole", "initial"])
def test_neutral_prefix_and_whole_endpoint_prevent_candidate(tmp_path, mode):
    deps, result, error = execute(tmp_path, mode)
    assert error is not None and not result["run_valid"]
    assert len(deps.sessions) == 1 and deps.sessions[0].close_calls == 1


def test_startup_rejection_stops_without_new_session(tmp_path):
    deps, result, error = execute(tmp_path, "startup")
    assert error is None and result["run_valid"] and not result["actual_startup_pass"]
    assert deps.sessions[1].control_tick == 100
    assert result["cost"]["scientific_neural_ticks"] == 2020
    assert not result["arms"]["candidate"]["short"]["available"]


@pytest.mark.parametrize("mode", ["constructor", "post_return", "advance", "readout", "raw", "state", "close", "publication", "partial"])
def test_failures_retain_cost_and_close_every_returned_owner(tmp_path, mode):
    deps, result, error = execute(tmp_path, mode)
    assert error is not None and not result["run_valid"]
    assert all(s.closed and s.close_calls == 1 for s in deps.sessions)
    assert (tmp_path / "out/errors.json").exists()
    if mode == "post_return":
        ledger = json.loads((tmp_path / "out/advance-ledger.json").read_text())
        assert ledger["arms"]["neutral"]["owned_session_recovered"]
    if mode == "advance":
        assert "first fake advance error" in result["first_error"]["message"]
        assert result["cost"]["scientific_neural_ticks"] == 40
        assert result["cost"]["scientific_physics_steps"] == 100


def test_native_health_stop_is_rejection_with_partial_cost(tmp_path):
    deps, result, error = execute(tmp_path, "health")
    assert error is None and result["run_valid"] and not result["actual_startup_pass"]
    assert deps.saved_partial == 1 and deps.sessions[1].control_tick == 0
    assert result["cost"]["scientific_physics_steps"] == 77


@pytest.mark.parametrize("error", [MemoryError("memory cap"), TimeoutError("science deadline")])
def test_budget_failure_prevents_body_and_retains_first_error(tmp_path, error):
    def budget():
        raise error
    deps, result, actual = execute(tmp_path, budget=budget)
    assert actual is error and not result["run_valid"] and not deps.sessions


def test_cleanup_failure_keeps_full_cost(tmp_path):
    def cleanup():
        raise OSError("runtime cleanup")
    deps, result, error = execute(tmp_path, cleanup=cleanup)
    assert str(error) == "runtime cleanup" and not result["run_valid"]
    assert result["cost"]["scientific_neural_ticks"] == 6020
    assert all(s.closed for s in deps.sessions)


def activity():
    return {"rate": np.zeros(3, np.float32), "drive": np.zeros(3, np.float32),
            "adaptation": np.zeros(3, np.float32), "synaptic_resource": np.ones(2, np.float64),
            "complete": False, "control_tick": 1}


@pytest.mark.parametrize("bad", [np.ones(3, np.float64), np.ones(2, np.float32),
                                  np.array([1., np.nan]), np.array([1., 1.1]), np.array([-.1, 1.])])
def test_resource_shape_dtype_finite_and_bounds(bad):
    record = activity()
    record["synaptic_resource"] = bad
    with pytest.raises(ValueError):
        probe.validate_activity(record, 3, 2)


def test_partial_activity_preserves_received_arrays_and_invalid_bytes(tmp_path):
    first, second = activity(), activity()
    first.pop("drive")
    second["synaptic_resource"] = np.array([1, 2, 3], np.int16)
    data = {"ids": np.arange(3), "motor_ids": np.array([2]), "source_ids": np.arange(2)}
    path = tmp_path / "activity.npz"
    probe.save_activity(path, [first, second], data)
    with np.load(path, allow_pickle=False) as saved:
        assert not saved["complete"].any() and np.isnan(saved["drive"][0]).all()
        assert np.array_equal(saved["rate"][0], first["rate"])
        assert saved["invalid_synaptic_resource_1"].dtype == np.int16
        assert np.array_equal(saved["invalid_synaptic_resource_1"], second["synaptic_resource"])


@pytest.mark.parametrize("mutation", ["missing", "reversed", "outside", "one_leg", "slip", "bout"])
def test_actual_short_cycles_fail_closed(mutation):
    candidate = metrics(True)
    cycle = candidate["gait"]["events"][probe.LEGS[0]]["cycles"][0]
    if mutation == "missing":
        cycle.pop("pep_s")
    elif mutation == "reversed":
        cycle["pep_s"] = 1.1
    elif mutation == "outside":
        cycle["next_aep_s"] = 1.5
    elif mutation == "one_leg":
        candidate["gait"]["events"][probe.LEGS[1]]["complete_cycles"] = 0
    elif mutation == "slip":
        candidate["slip"]["load_weighted_mm_s"] = 1.8026436968249956
    else:
        candidate["gait"]["window"]["bouts"] = []
    assert not probe.assess_short(metrics(), candidate)["pass"]


def test_exact_grid_bout_duration():
    candidate = metrics(True)
    candidate["gait"]["window"]["bouts"] = [{"start_s": .55, "end_s": .85, "samples": 61}]
    for leg in probe.LEGS:
        candidate["gait"]["events"][leg]["cycles"] = [{"aep_s": .57, "pep_s": .7, "next_aep_s": .83}]
    assert probe.assess_short(metrics(), candidate)["pass"]


def test_gate_sha_precedes_any_dependency(tmp_path):
    path = tmp_path / "freeze.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")



def generated_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    def safe(name):
        from pathlib import Path
        relative = Path(name)
        target = (tmp_path / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(tmp_path):
            raise ValueError("Repository relative frozen input required")
        return target
    monkeypatch.setattr(probe, "safe", safe)
    def put(name, value):
        target = safe(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return {"path": name, "sha256": probe.protocol.sha256(target)}
    sources = {}
    for name in probe.SOURCE_FILES:
        sources[name] = put(name, {"generated_source": name})["sha256"]
    roster = safe(probe.ROSTER_PATH)
    roster.parent.mkdir(parents=True, exist_ok=True)
    np.savez(roster, fake=np.zeros(1))
    monkeypatch.setattr(probe, "ROSTER_SHA", probe.protocol.sha256(roster))
    monkeypatch.setattr(probe, "checked_roster", lambda _value: {})
    class NoModel:
        def load(self, freeze):
            return {}
    monkeypatch.setattr(probe, "NativeDependencies", NoModel)
    paths = {key: "inputs/" + key + ".json" for key in probe.PATHS}
    paths["selection"] = probe.ROSTER_PATH
    inputs = {probe.ROSTER_PATH: probe.ROSTER_SHA}
    for key, name in paths.items():
        if key != "selection":
            inputs[name] = put(name, {"generated_input": key})["sha256"]
    packages = {"numpy": probe.importlib.metadata.version("numpy")}
    old = {"schema": probe.prior.FREEZE_SCHEMA, "paths": paths, "input_sha256": inputs,
           "graph_hash": "graph", "source_network_identity": "original",
           "adapter_identity": "adapter", "packages": packages}
    old_spec = put("receipts/old-freeze.json", old)
    old_review = put("receipts/old-review.json", {"schema": probe.prior.REVIEW_SCHEMA,
        "approved": True, "blocking_defects": [], "freeze_sha256": old_spec["sha256"],
        "source_sha256": {name: sources[name] for name in probe.prior.OWNED}})
    design = put("design.json", {"generated": True})
    preservation = put("preserved.json", {"M2": "preserved"})
    freeze = {"schema": probe.FREEZE_SCHEMA, "owned_files": list(probe.OWNED),
        "paths": paths, "startup_limits": probe.STARTUP_LIMITS, "resource_limits": probe.LIMITS,
        "n": 158706, "edges": 725849, "premotor_count": 7136, "source_count": 7136,
        "motor_count": 391, "utilization": .05, "neutral_utilization": 0., "recovery_tau": .150,
        "candidate_adopted": False, "input_sha256": inputs, "reused_source_sha256": sources,
        "preservation_sha256": {preservation["path"]: preservation["sha256"]},
        "design": design["path"], "design_sha256": design["sha256"],
        "e279_freeze": old_spec, "e279_source_review": old_review,
        "graph_hash": "graph", "source_network_identity": "original", "adapter_identity": "adapter",
        "packages": packages, "output": "output"}
    for alias in ("e289_source_review", "e290_source_review", "tiny_qualification"):
        expected = {name: sources[name] for name in probe.PRODUCERS
                    if alias == "tiny_qualification" or ("synapses" if alias == "e289_source_review" else "trial") in name}
        freeze[alias] = put("receipts/" + alias + ".json", {"approved": True,
                                "blocking_defects": [], "source_sha256": expected})
    def call():
        spec = put("freeze.json", freeze)
        review = put("review.json", {"schema": probe.REVIEW_SCHEMA, "approved": True,
            "blocking_defects": [], "freeze_sha256": spec["sha256"],
            "source_sha256": {name: sources[name] for name in probe.OWNED}})
        return probe.gate(safe(spec["path"]), spec["sha256"], safe(review["path"]),
                          review["sha256"], safe("output"))
    return freeze, call, safe


def test_complete_generated_gate_never_constructs_models(tmp_path, monkeypatch):
    freeze, call, _safe = generated_gate(tmp_path, monkeypatch)
    assert call() == freeze


@pytest.mark.parametrize("mutation", ["input", "source", "roster", "path", "package", "review", "tiny", "output"])
def test_gate_tampering_prevents_execution(tmp_path, monkeypatch, mutation):
    freeze, call, safe = generated_gate(tmp_path, monkeypatch)
    if mutation in ("input", "source", "roster"):
        name = (freeze["paths"]["startup"] if mutation == "input" else
                probe.OWNED[0] if mutation == "source" else probe.ROSTER_PATH)
        safe(name).write_bytes(b"mutated")
    elif mutation == "path":
        freeze["paths"]["startup"] = "../outside.json"
    elif mutation == "package":
        # Both root provenance and current environment must agree with pins.
        freeze["packages"] = {"numpy": "never-installed"}
        old_path = safe(freeze["e279_freeze"]["path"])
        old = json.loads(old_path.read_text())
        old["packages"] = freeze["packages"]
        old_path.write_text(json.dumps(old))
        freeze["e279_freeze"]["sha256"] = probe.protocol.sha256(old_path)
        review_path = safe(freeze["e279_source_review"]["path"])
        review = json.loads(review_path.read_text())
        review["freeze_sha256"] = freeze["e279_freeze"]["sha256"]
        review_path.write_text(json.dumps(review))
        freeze["e279_source_review"]["sha256"] = probe.protocol.sha256(review_path)
    elif mutation in ("review", "tiny"):
        alias = "tiny_qualification" if mutation == "tiny" else "e290_source_review"
        path = safe(freeze[alias]["path"])
        receipt = json.loads(path.read_text())
        receipt["approved"] = False
        path.write_text(json.dumps(receipt))
        freeze[alias]["sha256"] = probe.protocol.sha256(path)
    else:
        safe("output").mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        call()


def test_resource_roster_fixed_parameters_without_original_data():
    n, edges = 7136, 725849
    source = np.arange(n, dtype=np.int64)
    pre = np.resize(source, edges)
    roster = {"premotor_ids": source, "source_ids": source.copy(), "motor_ids": np.arange(n, n + 391, dtype=np.int64),
        "edge_ids": np.arange(edges, dtype=np.int64), "pre_ids": pre, "post_ids": pre.copy(),
        "edge_source_slots": pre.astype(np.int32), "weights32": np.ones(edges, np.float32),
        "utilization": np.full(n, .05), "neutral_utilization": np.zeros(n),
        "recovery_tau": np.full(n, .150), "descriptive_edge_groups": np.zeros(edges, np.int64)}
    probe.checked_roster(roster)
    roster["utilization"][0] = .06
    with pytest.raises(ValueError, match="Fixed"):
        probe.checked_roster(roster)



def test_finalization_deadline_revokes_published_eligibility(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(probe, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    class LatePublication(FakeDeps):
        def write(self, path, value):
            super().write(path, value)
            if path.name == "output-manifest.json":
                now[0] = 31.0
    deps = LatePublication()
    result, error = probe.execute({}, tmp_path / "out", deps=deps, budget=lambda: None)
    assert isinstance(error, TimeoutError) and not result["run_valid"]
    assert not result["next_separate_10s_eligible"]
    assert result["cost"]["scientific_neural_ticks"] == 6020
    assert not (tmp_path / "out/output-manifest.json").exists()
    stored = json.loads((tmp_path / "out/result.json").read_text())
    assert not stored["run_valid"] and all(not a["interpretation_validated"] for a in stored["arms"].values())


def test_primary_science_error_survives_secondary_storage_and_close(tmp_path):
    class MultipleFailures(FakeDeps):
        def save_raw(self, folder, raw, data, clock):
            super().save_raw(folder, raw, data, clock)
            if folder.name == "candidate":
                raise OSError("secondary raw publication")
            return []
        def construct(self, *args):
            session = super().construct(*args)
            original_close = session.close
            def close():
                original_close()
                if session.arm == "candidate":
                    raise OSError("secondary close")
            session.close = close
            return session
    deps = MultipleFailures("advance")
    result, error = probe.execute({}, tmp_path / "out", deps=deps, budget=lambda: None)
    assert str(error) == "first fake advance error"
    assert any("secondary raw publication" in n for n in error.__notes__)
    assert any("secondary close" in n for n in error.__notes__)
    assert all(s.close_calls == 1 for s in deps.sessions)
    assert not result["run_valid"]



def endpoint_state():
    return {"wall_s": 1.2, "control_tick": 1,
            "network": {"tick": 20, "rate": np.array([.25, .5], np.float32)},
            "body": {"physics_time_s": .005, "wall_s": .01},
            "adapter": {"bank": np.zeros((2, 3), np.float64)},
            "held": {"command": [0., 1.]}}


def test_independent_endpoint_wall_costs_accepted_and_preserved():
    actual, baseline = endpoint_state(), endpoint_state()
    actual["wall_s"] += .123
    before_actual, before_baseline = copy.deepcopy(actual), copy.deepcopy(baseline)
    report = probe.compare_neutral_endpoint(actual, baseline, 1.0)
    assert report["whole_end_exact"] and report["excluded_fields"] == ["wall_s"]
    assert report["actual_wall_s"] == 1.323 and report["baseline_wall_s"] == 1.2
    assert report["starting_wall_s"] == 1.0 and report["numeric_tolerance"] == 0
    probe.layer.require_equal(actual, before_actual, "Observed actual snapshot unchanged")
    probe.layer.require_equal(baseline, before_baseline, "Observed baseline snapshot unchanged")


@pytest.mark.parametrize("which", ["actual", "baseline"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -.1, .999, 1, np.float64(1.2)])
def test_invalid_or_regressive_endpoint_wall_rejected(which, value):
    actual, baseline = endpoint_state(), endpoint_state()
    node = actual if which == "actual" else baseline
    if value is None:
        node.pop("wall_s")
    else:
        node["wall_s"] = value
    with pytest.raises(ValueError, match="wall"):
        probe.compare_neutral_endpoint(actual, baseline, 1.0)


@pytest.mark.parametrize("mutation", ["array_value", "array_dtype", "float", "clock", "nested_wall", "field"])
def test_wall_exception_does_not_relax_scientific_endpoint(mutation):
    actual, baseline = endpoint_state(), endpoint_state()
    actual["wall_s"] += .123
    if mutation == "array_value":
        actual["network"]["rate"][0] = np.nextafter(np.float32(.25), np.float32(1.))
    elif mutation == "array_dtype":
        actual["network"]["rate"] = actual["network"]["rate"].astype(np.float64)
    elif mutation == "float":
        actual["held"]["command"][0] = 1e-15
    elif mutation == "clock":
        actual["body"]["physics_time_s"] += 1e-15
    elif mutation == "nested_wall":
        actual["body"]["wall_s"] += .123
    else:
        actual.pop("control_tick")
    with pytest.raises(ValueError):
        probe.compare_neutral_endpoint(actual, baseline, 1.0)


def test_endpoint_only_storage_sequence_scalar_normalization():
    actual, baseline = endpoint_state(), endpoint_state()
    actual["held"]["command"] = (np.float64(0.), 1.)
    report = probe.compare_neutral_endpoint(actual, baseline, 1.0)
    assert report["whole_end_exact"] and report["actual_representation_changes"]
    assert report["excluded_fields"] == ["wall_s"]


def test_generated_cost_receipt():
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 90
    print(json.dumps({"wall_s": elapsed, "fake_counts": COUNTS, "original": 0,
                      "CUDA": 0, "body": 0, "physics": 0, "solver": 0}))
