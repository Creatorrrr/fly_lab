"""Focused generated runtime diagnostics; no real Original or candidate payload."""
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import banc_m3_conserved_resource as r
from tools import banc_m3_contact_learning_admission as admission
from tools import banc_m3_contact_runtime_journal as runtime
from tools import banc_m3_startup_ray_supervisor as s


def pin(name="request.json"):
    return {"path": name, "bytes": 7, "sha256": "1"*64}


def journal_fixture(*, stream=None, guard=None):
    now = [100.0]
    owner = {"call_id": admission.D_RUNTIME_CALLS[0], "request": pin("supervisor-request.json"),
             "entry_counter_s": 100.0, "owner_pid": 17, "owner_creation_ticks": 19, "job_name": "generated-fixture"}
    journal = runtime.RuntimeJournal(100.0, owner, clock=lambda: now[0], stream=io.StringIO() if stream is None else stream, guard=guard)
    journal.bind({"call_id": owner["call_id"], "role": "contact_candidate_body", "source_freeze": pin("source-y.json")}, pin())
    return journal, now


def test_math_runtime_journal_caps_first_error_and_scalar_projection():
    journal, now = journal_fixture()
    error = RuntimeError("\U0001f9ea"*3000)
    journal.error("constructor", error)
    for i in range(300):
        journal.error("secondary-"+str(i), error)
        with journal.phase("candidate-hash"):
            now[0] += .001
    class NoTraversal:
        def __iter__(self):
            raise AssertionError("array/result traversal forbidden")
    journal.acquire_work({"physics_requested": 3, "physics_host_returned": 2, "physics_device_confirmed": 1,
                          "physics_unreturned_unknown": True, "whole_array": NoTraversal()},
                         {"body_constructors_requested": 1, "body_constructors_returned": 0, "unknown_work": True,
                          "unbounded_result": NoTraversal()})
    snapshot = journal.snapshot()
    encoded = journal.encoded()
    assert len(encoded) <= runtime.MAX_JSON_BYTES and json.loads(encoded)["schema"] == runtime.SCHEMA
    assert len(snapshot["events"]) <= runtime.MAX_EVENTS-1 and snapshot["events_dropped"] > 0
    assert snapshot["bytes_truncated"] and snapshot["incomplete"]
    assert len(snapshot["first_error"]["message"]) == runtime.MAX_ERROR_CHARS
    assert snapshot["first_error"]["stage"] == "constructor" and len(snapshot["secondary_errors"]) <= 8
    assert snapshot["secondary_errors_dropped"] > 0 and snapshot["last_phase"]["kind"] == "RETURN"
    assert snapshot["work"]["counters"] == {"physics_requested": 3, "physics_host_returned": 2,
                                           "physics_device_confirmed": 1, "physics_unreturned_unknown": True}
    assert snapshot["work"]["construction"]["unknown_work"] is True
    assert snapshot["run_valid"] is False and snapshot["diagnostic_only"] is True
    fresh, _ = journal_fixture()
    zero_construction = {key: False if key in ("device_confirmed", "unknown_work") else 0 for key in runtime.CONSTRUCTION}
    fresh.acquire_work({"physics_requested": 0, "neural_requested": 0}, zero_construction)
    assert fresh.snapshot()["work"] is None  # preallocated zero/false construction is not zero-work proof
    fresh.acquire_work({}, {"unknown_work": True, "device_confirmed": False})
    assert fresh.snapshot()["work"] == {"construction": {"device_confirmed": False, "unknown_work": True}}
    acquired, _ = journal_fixture()
    acquired.acquire_work({}, {"body_constructors_requested": 1, "body_constructors_returned": 1, "unknown_work": False})
    assert acquired.snapshot()["work"] == {"construction": {"body_constructors_requested": 1,
                                                           "body_constructors_returned": 1, "unknown_work": False}}
    assert "counters" not in acquired.snapshot()["work"]  # returned constructor cannot create a control receipt


def test_math_runtime_journal_phase_and_cold_fence_elapsed():
    journal, now = journal_fixture()
    with journal.phase("body-plan"):
        now[0] = 101.0
        with journal.fence("pre_array"):
            now[0] = 103.0
        now[0] = 104.0
        with pytest.raises(RuntimeError, match="cold-sentinel"), journal.fence("pre_array"):
            now[0] = 105.0
            raise RuntimeError("cold-sentinel")
        now[0] = 106.0
    snapshot = journal.snapshot()
    fence = snapshot["cold_fences_nested_wall"]["pre_array"]
    assert fence == {"attempted": 2, "completed": 1, "failed": 1, "wall_s": 3.0}
    assert snapshot["events"] == [{"phase": "body-plan", "kind": "START", "observed_s": 0.0},
                                  {"phase": "body-plan", "kind": "RETURN", "observed_s": 6.0, "inclusive_wall_s": 6.0}]
    assert snapshot["first_error"]["stage"] == "cold-fence:pre_array"
    start = journal.start("arm-data")
    now[0] = 109.0
    active = journal.snapshot()
    assert active["active"][-1]["state"] == "NOT_RETURNED" and active["active"][-1]["observed_partial_s"] == 3.0
    with journal.fence("publication"):
        now[0] = 110.0
        mid = journal.snapshot()
        assert mid["active_fence"]["label"] == "publication" and mid["active_fence"]["observed_partial_s"] == 1.0
    journal.finish("arm-data", start)
    assert journal.snapshot()["active"] == []


def test_math_runtime_journal_snapshot_replacement_and_reserved_terminal():
    journal, now = journal_fixture()
    result = {"first_error": None, "secondary_errors": [], "work": object()}
    journal.error("construct", RuntimeError("acquired-first"))
    for _ in range(100):
        with journal.phase("candidate-load"):
            now[0] += .001
    result = {"first_error": {"stage": "publication", "type": "ValueError", "message": "later"},
              "secondary_errors": [], "work": object()}
    held_report = {"record": result}
    assert journal.terminal_failure(held_report["record"])
    snapshot = journal.snapshot()
    assert snapshot["first_error"]["message"] == "acquired-first"
    assert snapshot["terminal"]["kind"] == "RETURN" and snapshot["last_phase"]["phase"] == "candidate-load"
    snapshot["first_error"]["message"] = "mutated detached copy"
    assert journal.snapshot()["first_error"]["message"] == "acquired-first"
    with pytest.raises(ValueError, match="once"):
        journal.terminal_failure()
    lines = journal.stream.getvalue().splitlines()
    assert len(lines) == 1 and len(lines[0].encode("ascii")) <= runtime.MAX_JSON_BYTES
    durable = json.loads(lines[0][len(runtime.PREFIX):])
    assert durable["terminal"]["state"] == "NOT_RETURNED" and durable["work"] is None


def test_gate_runtime_fault_phases_preserve_first_error(tmp_path, monkeypatch):
    for failed_phase in ("body-plan", "arm-data", "pre-construct", "construct", "initial-transfer", "control-loop", "posthash", "science-publication"):
        journal, now = journal_fixture()
        callbacks = []
        with pytest.raises(RuntimeError, match="original-sentinel"), journal.phase(failed_phase, opaque=True):
            callbacks.append("entered")
            now[0] += .5
            raise RuntimeError("original-sentinel")
        journal.error("later-publication", OSError("secondary"))
        assert journal.terminal_failure()
        snapshot = journal.snapshot()
        assert callbacks == ["entered"] and snapshot["first_error"]["message"] == "original-sentinel"
        event = next(row for row in snapshot["events"] if row["phase"] == failed_phase and row["kind"] == "FAIL")
        assert event["inclusive_wall_s"] == .5
        assert not any(row["kind"] == "RETURN" and row["phase"] == failed_phase for row in snapshot["events"])
        first_wire = json.loads(journal.stream.getvalue().splitlines()[0][len(runtime.PREFIX):])
        assert first_wire["last_phase"]["phase"] == failed_phase and first_wire["active"][-1]["state"] == "NOT_RETURNED"
        assert first_wire["work"] is None and snapshot["run_valid"] is False

    def exercise_nested_start(failed_start):
        nested, clock = journal_fixture()
        callbacks, writes, guards = [], [], []
        expected_type = OSError if failed_start == "stream" else (RuntimeError if failed_start == "owner" else TimeoutError)
        expected_message = {"stream": "nested-stream-sentinel", "owner": "nested-owner-sentinel",
                            "cross270": "contact same owner ENTRY internal deadline"}[failed_start]
        original_error = expected_type(expected_message)
        class NestedStream:
            def write(self, value):
                writes.append(json.loads(value[len(runtime.PREFIX):]))
                if failed_start == "stream":
                    raise original_error
                clock[0] = 370.0  # successful flush returns within1s, at exact science270
                return len(value)
            def flush(self):
                pass
        def owned():
            guards.append("checked")
            if failed_start == "owner":
                raise original_error
        nested.stream, nested.guard = NestedStream(), owned
        clock[0] = 369.75
        with (
            pytest.raises(expected_type, match=expected_message) as observed,
            nested.phase("body-plan"),
            nested.phase("candidate-header", opaque=True),
        ):
            callbacks.append("forbidden-entry")
        snapshot = nested.snapshot()
        assert type(observed.value) is expected_type and str(observed.value) == expected_message
        if failed_start != "cross270":
            assert observed.value is original_error
        assert callbacks == [] and snapshot["active"] == []
        assert [(row["phase"], row["kind"]) for row in snapshot["events"]] == [
            ("body-plan", "START"), ("candidate-header", "START"),
            ("candidate-header", "FAIL"), ("body-plan", "FAIL")]
        assert snapshot["first_error"]["type"] == expected_type.__name__ and snapshot["first_error"]["message"] == expected_message
        assert all("LIFO" not in row["message"] for row in [snapshot["first_error"], *snapshot["secondary_errors"]])
        assert nested.emit_attempts == 1 and len(writes) == (0 if failed_start == "owner" else 1)
        assert len(guards) == (2 if failed_start == "cross270" else 1)
        if writes:
            assert [row["phase"] for row in writes[0]["active"]] == ["body-plan", "candidate-header"]
            assert all(row["state"] == "NOT_RETURNED" for row in writes[0]["active"])
        assert snapshot["stream"]["last"]["status"] == ("RETURN" if failed_start == "cross270" else "FAIL_OR_NOT_RETURNED")

    for failed_start in ("stream", "owner", "cross270"):
        exercise_nested_start(failed_start)

    from tools import verify_c_banc_m3_candidate_body_handoff as body
    from tools import verify_c_banc_m3_common_descent_body as common
    from tools import verify_c_banc_m3_startup_ray as ray

    def exercise_generated_lifecycle(stopped):
        projected, clock = journal_fixture()
        counters = common.new_counters()
        observed_work, wires = [], []
        called = {"construct": 0, "initial": 0, "advance": 0, "synchronize": 0}
        class GeneratedSession:
            closed, control_tick = False, 0
            network, body, adapter = SimpleNamespace(tick=0), SimpleNamespace(), SimpleNamespace()
            def advance(self, _count, *, capture):
                assert capture is False
                called["advance"] += 1
                assert stopped == "returned-advance"
                self.control_tick, self.network.tick = 1, 20
                counters.update(physics_requested=50, physics_host_returned=50,
                                neural_requested=20, neural_host_returned=20, neural_device_confirmed=20)
        session = GeneratedSession()
        class GeneratedFactory:
            install_contact = None  # inert generated observer never calls this
            def construct(self, _data, construction, timely):
                timely()
                called["construct"] += 1
                construction.update(body_constructors_requested=1, body_constructors_returned=1,
                                    settling_physics_requested=2000, settling_physics_host_returned=2000,
                                    neural_setup_requested=320, neural_setup_host_returned=320,
                                    neural_capture_requested=160, neural_capture_recorded=160,
                                    device_confirmed=True, unknown_work=False)
                return session
            def synchronize(self):
                called["synchronize"] += 1
                assert stopped == "returned-advance", "no new device synchronization after failed next START"
                observed_work.append((projected.snapshot()["work"], len(wires)))
                clock[0] = 370.0
                raise TimeoutError("generated-existing-sync-unreturned")
        class GeneratedObserver:
            def __init__(self, *_args, **_kwargs):
                pass
            def install(self):
                return self
        class BoundaryStream:
            def write(self, value):
                snapshot = json.loads(value[len(runtime.PREFIX):])
                wires.append(snapshot)
                if snapshot["last_phase"]["phase"] == stopped:
                    clock[0] = 370.0
                    raise OSError("generated-next-START-unreturned")
                return len(value)
            def flush(self):
                pass
        def initial(*_args, **_kwargs):
            called["initial"] += 1
            return {"cpg_hash": "generated-inert"}
        def closed(_observer, owned_session, _result, _timer):
            owned_session.closed = True
        timer = SimpleNamespace(runtime_journal=projected, check=lambda _label="science": projected.science(),
                                fence=lambda _label: projected.science(), root=tmp_path, limits={})
        class NoSciencePublication:
            folder = tmp_path
            def json(self, name, _value):
                timer.check("science")
                assert name == "transfer.json", "only the acquired generated transfer before270"
            def arrays(self, _name, _value):
                raise AssertionError("no new array processing after270")
        projected.stream = BoundaryStream()
        with monkeypatch.context() as changed:
            changed.setattr(common, "new_counters", lambda: counters)
            changed.setattr(common, "OriginalFactory", GeneratedFactory)
            changed.setattr(common, "LiveObserver", GeneratedObserver)
            changed.setattr(common, "trace_metadata", lambda _session: {})
            changed.setattr(common, "capture", lambda _session: ({}, {}))
            changed.setattr(body, "initial_body", initial)
            changed.setattr(body, "finish_owned", closed)
            with pytest.raises(TimeoutError, match="same owner ENTRY internal deadline"):
                ray.continuous_arm({}, "candidate", {}, {}, timer, NoSciencePublication(), None)
        snapshot = projected.snapshot()
        assert called["construct"] == 1 and session.closed and snapshot["active"] == []
        assert snapshot["work"]["construction"]["body_constructors_returned"] == 1
        assert snapshot["work"]["construction"]["device_confirmed"] is True
        if stopped == "returned-advance":
            assert called == {"construct": 1, "initial": 1, "advance": 1, "synchronize": 1}
            acquired, write_count = observed_work[0]
            assert write_count == len(wires) == 4  # no extra incremental journal stream writes
            assert acquired["counters"]["controls_requested"] == acquired["counters"]["controls_host_returned"] == 1
            assert acquired["counters"]["controls_completed"] == 0
            assert acquired["counters"]["physics_host_returned"] == 50 and acquired["counters"]["physics_device_confirmed"] == 0
            assert snapshot["work"]["counters"]["physics_device_confirmed"] == 0  # unreturned existing sync cannot confirm
            assert snapshot["first_error"]["message"] == "generated-existing-sync-unreturned"
        else:
            boundary = next(row for row in wires if row["last_phase"]["phase"] == stopped)
            assert boundary["active"][-1]["state"] == "NOT_RETURNED"
            assert boundary["work"]["construction"]["body_constructors_returned"] == 1
            assert boundary["work"]["construction"]["device_confirmed"] is True
            assert called["advance"] == called["synchronize"] == 0
            assert called["initial"] == (1 if stopped == "control-loop" else 0)
            if stopped == "initial-transfer":
                assert "counters" not in boundary["work"]
            else:
                assert boundary["work"]["counters"]["completed_rows"] == 1
                assert boundary["work"]["counters"]["controls_host_returned"] == 0
            assert snapshot["first_error"]["message"] == "generated-next-START-unreturned"

    for stopped in ("initial-transfer", "control-loop", "returned-advance"):
        exercise_generated_lifecycle(stopped)


def test_gate_runtime_deadline270_285_flush_once_and_no_late_science():
    journal, now = journal_fixture()
    with pytest.raises(ValueError, match="no fresh origin"):
        runtime.RuntimeJournal(101.0, journal.owner, clock=lambda: 101.0, stream=io.StringIO())
    now[0] = 370.0  # exact actual ENTRY+270
    timer = object.__new__(admission.Admission)
    timer.entry, timer._deadline_s, timer.runtime_journal = 100.0, 270, journal
    touched = []
    timer.root = Path(".")
    original = admission.time.perf_counter
    admission.time.perf_counter = lambda: now[0]
    try:
        with pytest.raises(TimeoutError):
            timer.read(pin())
        timer._cold_fence = lambda label: touched.append(label)
        with pytest.raises(TimeoutError):
            timer.fence("pre_array")
        for _operation in ("header", "Original-read", "posthash", "construct", "neural", "physics", "Source-history-scan"):
            with pytest.raises(TimeoutError), runtime.phase(timer, "construct"):
                touched.append(_operation)
        assert touched == [] and journal.snapshot()["cold_fences_nested_wall"]["pre_array"]["failed"] == 1
        assert journal.terminal_failure()  # acquired metadata only, still before285
        assert timer._deadline_s == 270 and journal.entry == 100.0
    finally:
        admission.time.perf_counter = original
    for deadline in (384.0, 385.0, 386.0):
        blocked, clock = journal_fixture()
        clock[0] = deadline
        assert blocked.terminal_failure() is False and blocked.stream.getvalue() == ""
        assert blocked.snapshot()["stream"]["last"]["status"] == "REFUSED_NO_FLUSH_TIME"
    late, clock = journal_fixture()
    class SlowStream:
        def write(self, value):
            clock[0] += 1.01
            return len(value)
        def flush(self):
            pass
    late.stream = SlowStream()
    assert late.terminal_failure() is False
    assert late.snapshot()["stream"]["last"]["status"] == "FAIL_OR_NOT_RETURNED"
    assert late.snapshot()["first_error"]["stage"] == "failure-metadata-publication"
    capped, _ = journal_fixture()
    for _ in range(15):
        assert capped.emit()
    assert capped.emit() is False and capped.terminal_failure()
    assert capped.emit_attempts == 16 and capped.stream_bytes <= runtime.MAX_STREAM_BYTES
    class BrokenStream:
        def write(self, _value):
            raise OSError("pipe-closed")
        def flush(self):
            raise AssertionError("must not flush failed write")
    broken, _ = journal_fixture(stream=BrokenStream())
    broken.error("arm-data", RuntimeError("primary"))
    assert broken.terminal_failure() is False
    assert broken.snapshot()["first_error"]["message"] == "primary"
    with pytest.raises(ValueError, match="once"):
        broken.terminal_failure()


def test_gate_runtime_body_worker_failure_acquired_journal(tmp_path, monkeypatch):
    from tools import banc_m3_contact_learning as science
    from tools import verify_c_banc_m3_contact_learning as worker
    from tools import verify_c_banc_m3_live_replay as replay

    journal, now = journal_fixture()
    fake = SimpleNamespace(runtime_journal=journal, role="contact_candidate_body", limits={"output_cap_bytes": 96 << 20},
                           request={"contract": pin("stage-a.json"), "source_freeze": pin("source-y.json"),
                                    "artifacts": {"report": pin("audit.json"), "candidate_arrays": pin("candidate.npz")}},
                           request_spec=pin(), root=tmp_path)
    fake.check = lambda _label="science": journal.science()
    class Publisher:
        folder, admission = tmp_path, fake
        def json(self, _name, _value):
            raise OSError("later-publication")
        def write(self, _name, _raw):
            raise OSError("later-publication")
    class Saved:
        def __init__(self, _gate, _np):
            pass
        def json(self, _spec):
            return {"run_valid": True, "candidate_present": True, "arithmetic_valid": True,
                    "candidate": copy.deepcopy(admission.RETRY_PINS["candidate"])}
        def posthash(self):
            fake.check()
            raise OSError("posthash-secondary")
    monkeypatch.setattr(replay, "publisher_for", lambda _gate: Publisher())
    monkeypatch.setattr(science, "Saved", Saved)
    def body_plan(*_args):
        now[0] = 370.0
        raise TimeoutError("contact same owner ENTRY internal deadline")
    monkeypatch.setattr(science, "body_plan", body_plan)
    with pytest.raises(ValueError, match="technical failure retained"):
        worker.body_worker(fake)
    snapshot = journal.snapshot()
    assert snapshot["first_error"]["stage"] == "body-plan" and snapshot["first_error"]["type"] == "TimeoutError"
    assert snapshot["work"] is None and snapshot["terminal"]["state"] == "RETURNED"
    assert fake.request["artifacts"]["candidate_arrays"] == pin("candidate.npz")
    assert not list(tmp_path.iterdir())  # the fake publication could not create a science result


def test_gate_runtime_owner_drift_refuses_diagnostic_validity(monkeypatch):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    journal, _ = journal_fixture()
    owner = journal.owner
    monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps(owner))
    monkeypatch.setattr(runtime.os, "getpid", lambda: 23)
    monkeypatch.setattr(supervisor, "creation_ticks", lambda pid: 29 if pid == 23 else 19)
    monkeypatch.setattr(supervisor, "in_named_job", lambda _name: True)
    runtime.owned_stream_guard(owner, 23, 29)
    for kind in ("token", "worker", "owner_creation", "job"):
        with monkeypatch.context() as changed:
            if kind == "token":
                changed.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps(dict(owner, call_id="foreign")))
            elif kind == "worker":
                changed.setattr(runtime.os, "getpid", lambda: 99)
            elif kind == "owner_creation":
                changed.setattr(supervisor, "creation_ticks", lambda _pid: 999)
            else:
                changed.setattr(supervisor, "in_named_job", lambda _name: False)
            with pytest.raises(ValueError):
                runtime.owned_stream_guard(owner, 23, 29)
            refused, _ = journal_fixture(guard=lambda: runtime.owned_stream_guard(owner, 23, 29))
            assert refused.terminal_failure() is False and refused.stream.getvalue() == ""
            assert refused.snapshot()["trust"].startswith("UNTRUSTED") and refused.snapshot()["run_valid"] is False
    timer = object.__new__(admission.Admission)
    timer.entry, timer._deadline_s, timer.runtime_journal = 100.0, 270, journal
    timer.role, timer.stage = "contact_candidate_body", "pre_array"
    monkeypatch.setattr(admission.time, "perf_counter", lambda: 101.0)
    def rejected(_label):
        raise ValueError("Source/lease/foreign shared lock drift")
    timer._cold_fence = rejected
    with pytest.raises(ValueError, match="drift"):
        timer.fence("pre_construct")
    assert journal.terminal_failure()
    snapshot = journal.snapshot()
    assert snapshot["trust"].startswith("UNTRUSTED") and snapshot["run_valid"] is False
    assert snapshot["cold_fences_nested_wall"]["pre_construct"]["failed"] == 1


def copy_runtime_budget_metadata(root):
    """Copy only the closed budget/policy/event JSON chain, never payload pins."""
    allowed = ("before_ledger", "initial_policy", "first_transfer", "additional_transfer", "final_transfer",
               "continuation_transfer", "direct_human_auto_authority", "direct_science_policy",
               "direct_science_generated_transfer", "direct_science_generated_additional_transfer",
               "direct_science_generated_final_transfer", "direct_science_generated_continuation_transfer")
    visited = set()
    def copy_pin(spec):
        if spec["path"] in visited:
            return
        visited.add(spec["path"])
        assert Path(spec["path"]).suffix == ".json" and 0 < spec["bytes"] <= 524288
        raw = admission.leaf(spec, admission.ROOT, 524288)
        target = root / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        value = json.loads(raw)
        for key in allowed:
            item = value.get(key)
            if type(item) is dict and set(item) == {"path", "bytes", "sha256"}:
                copy_pin(item)
    for spec in (admission.D_HISTORY_PINS["closed276"], admission.D_HISTORY_PINS["runtime_transfer"]):
        copy_pin(spec)
    anchor = json.loads((root/admission.D_HISTORY_PINS["closed276"]["path"]).read_bytes())
    return dict(anchor, direct_science_generated_runtime_transfer=admission.D_HISTORY_PINS["runtime_transfer"],
                direct_science_generated_limit_s=960)


def copy_runtime_recovery_metadata(root):
    """Immutable failed-y Source/cost/budget JSON only; no Original payload."""
    copy_runtime_budget_metadata(root)
    for spec in admission.RUNTIME_RECOVERY_PINS.values():
        assert Path(spec["path"]).suffix == ".json" and 0 < spec["bytes"] <= 524288
        raw = admission.leaf(spec, admission.ROOT, 524288)
        target = root/spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    closed = json.loads((root/admission.RUNTIME_RECOVERY_PINS["closed277"]["path"]).read_bytes())
    return dict(closed, direct_science_generated_runtime_residual_transfer=admission.RUNTIME_RECOVERY_PINS["residual_transfer"],
                direct_science_generated_limit_s=964)


def test_gate_runtime_fifth_transfer_python_native_catalog_and_negatives(tmp_path):
    from tools import banc_m3_startup_ray_supervisor as supervisor
    valid = copy_runtime_budget_metadata(tmp_path)
    assert supervisor.generated_budget_limit(valid, tmp_path) == 960
    cases = []
    for kind in ("lower", "missing", "duplicate", "stale", "source", "call", "debit", "amount", "grant", "cap"):
        wrong = copy.deepcopy(valid)
        if kind == "lower":
            wrong["direct_science_generated_limit_s"] = 900
        elif kind == "missing":
            del wrong["direct_science_generated_runtime_transfer"]
        elif kind == "duplicate":
            wrong["direct_science_generated_runtime_transfer"]["path"] += ".duplicate.json"
        elif kind == "stale":
            wrong["direct_science_generated_runtime_transfer"] = wrong["direct_science_generated_continuation_transfer"]
        elif kind == "source":
            wrong["direct_science_generated_runtime_transfer"]["sha256"] = "0"*64
        elif kind == "call":
            wrong["direct_science_calls"][admission.C_RETRY_CALLS[0]]["state"] = "admitted"
        elif kind == "debit":
            wrong["debits"][0]["cost_id"] = "mutated"
        elif kind in ("amount", "grant"):
            spec = wrong["direct_science_generated_runtime_transfer"]
            altered = json.loads((tmp_path/spec["path"]).read_bytes())
            altered["transfer_s" if kind == "amount" else "new_resource_grant_s"] += 1
            name = admission.BASE+"/runtime-"+kind+".json"
            raw = json.dumps(altered).encode("utf8")
            (tmp_path/name).write_bytes(raw)
            wrong["direct_science_generated_runtime_transfer"] = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        else:
            wrong["direct_science_generated_limit_s"] = 961
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            supervisor.generated_budget_limit(wrong, tmp_path)
        cases.append(wrong)
    residual = copy_runtime_recovery_metadata(tmp_path)
    assert supervisor.generated_budget_limit(residual, tmp_path) == 964
    assert residual["debits"][276]["cost_id"] == "direct-science-20261007-contact-lint63"
    assert residual["debits"][276]["actual_returned_exit_code"] == 1
    residual_cases = []
    for kind in ("lower", "missing", "call", "debit", "cap", "future_pointer"):
        wrong = copy.deepcopy(residual)
        if kind == "lower":
            wrong["direct_science_generated_limit_s"] = 960
        elif kind == "missing":
            del wrong["direct_science_generated_runtime_residual_transfer"]
        elif kind == "call":
            wrong["direct_science_calls"]["direct-science-20261007-contact-lint63"]["state"] = "admitted"
        elif kind == "debit":
            wrong["debits"][276]["actual_returned_exit_code"] = 0
        elif kind == "cap":
            wrong["direct_science_generated_limit_s"] = 965
        else:
            wrong["direct_science_generated_limit_s"] = 300
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            supervisor.generated_budget_limit(wrong, tmp_path)
        residual_cases.append(wrong)
    assert shutil.which("pwsh"), "Native qualification requires pwsh; no skip"
    source = (admission.ROOT/admission.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    definitions = source[source.index("$ErrorActionPreference="):source.index("$ledger=Get-Content -LiteralPath (Safe $ledgerName)")]
    root_anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert definitions.count(root_anchor) == 1
    definitions = definitions.replace(root_anchor, "$taskRoot=[IO.Path]::GetFullPath('"+tmp_path.as_posix().replace("'", "''")+"')")
    data = tmp_path/"runtime-budget-cases.json"
    data.write_text(json.dumps({"positive": valid, "negative": cases, "residual_positive": residual, "residual_negative": residual_cases}), encoding="utf8")
    script = tmp_path/"runtime-budget.ps1"
    script.write_text(definitions + "\n" + r"""
$c=Get-Content -LiteralPath (Safe 'runtime-budget-cases.json') -Raw|ConvertFrom-Json -AsHashtable
Need ((Generated-Budget $c.positive) -eq 960) 'Native exact effective960'
foreach($l in $c.negative){$refused=$false;try{$null=Generated-Budget $l}catch{$refused=$true};Need $refused 'Native same meaningful strict fifth negatives'}
Need ((Generated-Budget $c.residual_positive) -eq 964) 'Native exact conserved residual964'
foreach($l in $c.residual_negative){$refused=$false;try{$null=Generated-Budget $l}catch{$refused=$true};Need $refused 'Native strict residual negatives'}
'RUNTIME_CATALOG_PASS'
""", encoding="utf8")
    observed = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script)], capture_output=True, timeout=24,
                              check=False, creationflags=0x08000000 if os.name == "nt" else 0)
    assert observed.returncode == 0 and b"RUNTIME_CATALOG_PASS" in observed.stdout, observed.stderr


def test_math_preparation_lifetime_totals_and_global_only_fence():
    journal, now = journal_fixture()
    for _ in range(2):
        with journal.phase("input-original-files"):
            now[0] += 1.0
            with journal.fence("pre_array"):
                now[0] += 2.0
            now[0] += 1.0
        with journal.phase("input-native-assets"):
            now[0] += 1.0
            with journal.fence("pre_array"):
                now[0] += 1.0
            now[0] += 1.0
    with journal.fence("publication"):
        now[0] += 2.0
    snapshot = journal.snapshot()
    original = snapshot["preparation"]["input-original-files"]
    assets = snapshot["preparation"]["input-native-assets"]
    assert (original["attempted"], original["completed"], original["failed"]) == (2, 2, 0)
    assert original["inclusive_wall_s"] == 8.0 and original["nested_fence"]["wall_s"] == 4.0
    assert assets["inclusive_wall_s"] == 6.0 and assets["nested_fence"]["wall_s"] == 2.0
    assert original["nested_fence"] == {"attempted": 2, "completed": 2, "failed": 0, "wall_s": 4.0}
    assert original["state"] == assets["state"] == "RETURNED"
    assert snapshot["cold_fences_nested_wall"]["pre_array"]["wall_s"] == 6.0
    assert snapshot["cold_fences_nested_wall"]["publication"]["wall_s"] == 2.0
    assert sum(row["nested_fence"]["attempted"] for row in snapshot["preparation"].values()) == 4
    assert len(snapshot["preparation"]) == 8 and snapshot["work"] is None
    assert journal.emit_attempts == 0 and journal.stream.getvalue() == ""


def test_math_preparation_partial_detached_snapshots_and_captured_nearest_slot():
    journal, now = journal_fixture()
    with journal.phase("input-original-files"):
        now[0] = 102.0
        with journal.phase("candidate-load"), journal.fence("pre_array"):
            now[0] = 105.0
            totals = copy.deepcopy(journal.preparation)
            first = journal.snapshot()
            now[0] = 110.0
            second = journal.snapshot()
            assert journal.preparation == totals  # snapshot cannot accumulate partial time
            a, b = first["preparation"]["input-original-files"], second["preparation"]["input-original-files"]
            assert a["observed_partial_s"] == 5.0 and b["observed_partial_s"] == 10.0
            assert a["nested_fence_partial_s"] == 3.0 and b["nested_fence_partial_s"] == 8.0
            assert a["inclusive_wall_s"] == a["nested_fence"]["wall_s"] == 0.0
            assert b["state"] == "NOT_RETURNED" and second["active_fence"]["preparation"] == "input-original-files"
            first["preparation"]["input-original-files"]["nested_fence"]["attempted"] = 999
            assert journal.snapshot()["preparation"]["input-original-files"]["nested_fence"]["attempted"] == 1
            with journal.phase("input-native-assets"):
                now[0] = 111.0
            now[0] = 112.0
        now[0] = 115.0
        with journal.phase("input-native-assets"), journal.phase("candidate-hash"), journal.fence("pre_array"):
            now[0] = 116.0
    snapshot = journal.snapshot()
    assert snapshot["preparation"]["input-original-files"]["inclusive_wall_s"] == 16.0
    assert snapshot["preparation"]["input-original-files"]["nested_fence"]["wall_s"] == 10.0
    assert snapshot["preparation"]["input-native-assets"]["nested_fence"]["wall_s"] == 1.0
    assert snapshot["preparation"]["input-whole-metadata"]["state"] == "NOT_STARTED"
    assert snapshot["active"] == [] and snapshot["active_fence"] is None
    assert journal.emit_attempts == 0


def test_gate_preparation_second_stage_fence_failure_identity_and_unstarted_next():
    journal, now = journal_fixture()
    sentinel = RuntimeError("generated-second-stage-fence")
    callbacks = []
    with pytest.raises(RuntimeError) as observed, journal.phase("arm-data", opaque=True):
        with journal.phase("input-original-files"):
            callbacks.append("original")
            now[0] += 1.0
        with journal.phase("input-native-assets"):
            callbacks.append("assets")
            with journal.fence("pre_array"):
                now[0] += 2.0
                raise sentinel
        with journal.phase("input-whole-metadata"):
            callbacks.append("forbidden-next")
    snapshot = journal.snapshot()
    slot = snapshot["preparation"]["input-native-assets"]
    assert observed.value is sentinel and callbacks == ["original", "assets"]
    assert (slot["attempted"], slot["completed"], slot["failed"], slot["active"]) == (1, 0, 1, 0)
    assert slot["state"] == "FAILED" and slot["inclusive_wall_s"] == 2.0
    assert slot["nested_fence"] == {"attempted": 1, "completed": 0, "failed": 1, "wall_s": 2.0}
    assert snapshot["cold_fences_nested_wall"]["pre_array"] == slot["nested_fence"]
    assert snapshot["first_error"]["stage"] == "cold-fence:pre_array"
    assert snapshot["first_error"]["message"] == str(sentinel)
    assert snapshot["preparation"]["input-whole-metadata"]["state"] == "NOT_STARTED"
    assert snapshot["active"] == [] and snapshot["active_fence"] is None and snapshot["work"] is None
    assert journal.emit_attempts == len(journal.stream.getvalue().splitlines()) == 1
    durable = json.loads(journal.stream.getvalue().splitlines()[0][len(runtime.PREFIX):])
    assert all(row["state"] == "NOT_STARTED" for row in durable["preparation"].values())
    assert durable["observed_s"] == 0.0  # as-of observation cannot exclude later entry


@pytest.mark.parametrize("fault", ["clock", "event", "finish", "error-record"])
def test_gate_preparation_payload_exception_survives_nested_diagnostic_unwind(monkeypatch, fault):
    journal, now = journal_fixture()
    primary = RuntimeError("generated-primary-payload")
    diagnostic = OSError("generated-unwind-observation")
    entered = []
    broken = [False]
    original_clock, original_event, original_finish, original_error = journal.clock, journal._event, journal.finish, journal._error_at
    def clock():
        if broken[0] and fault == "clock":
            raise diagnostic
        return original_clock()
    def event(*args, **kwargs):
        if broken[0] and fault == "event":
            raise diagnostic
        return original_event(*args, **kwargs)
    def finish(*args, **kwargs):
        if broken[0] and fault == "finish":
            raise diagnostic
        return original_finish(*args, **kwargs)
    def error_at(*args, **kwargs):
        if broken[0] and fault == "error-record":
            raise diagnostic
        return original_error(*args, **kwargs)
    with monkeypatch.context() as changed:
        changed.setattr(journal, "clock", clock)
        changed.setattr(journal, "_event", event)
        changed.setattr(journal, "finish", finish)
        changed.setattr(journal, "_error_at", error_at)
        with (
            pytest.raises(RuntimeError) as observed,
            journal.phase("body-plan"), journal.phase("arm-data"), journal.phase("input-native-assets"),
            journal.fence("pre_array"),
        ):
            entered.append("once")
            now[0] += 1.0
            broken[0] = True
            raise primary
    snapshot = journal.snapshot()
    assert observed.value is primary and entered == ["once"]
    assert snapshot["first_error"]["message"] == str(primary)
    assert snapshot["first_error"]["stage"] == "cold-fence:pre_array"
    assert snapshot["observation_incomplete"] and snapshot["incomplete"]
    assert snapshot["active"] == [] and snapshot["active_fence"] is None
    slot = snapshot["preparation"]["input-native-assets"]
    assert slot["failed"] == 1 and slot["completed"] == 0 and slot["state"] == "FAILED"
    if fault in ("clock", "event", "finish"):
        assert slot["inclusive_wall_s"] is None and slot["observation_incomplete"]
    if fault == "clock":
        assert slot["nested_fence"]["wall_s"] is None
        assert snapshot["cold_fences_nested_wall"]["pre_array"]["wall_s"] is None
    assert journal.emit_attempts == 0 and snapshot["work"] is None


def test_gate_preparation_normal_refusals_remain_refusals_and_no_callback_retry(monkeypatch):
    journal, now = journal_fixture()
    now[0] = 370.0
    entered = []
    with pytest.raises(TimeoutError), journal.phase("input-graph-load"):
        entered.append("forbidden-late")
    snapshot = journal.snapshot()
    slot = snapshot["preparation"]["input-graph-load"]
    assert slot["attempted"] == slot["entry_refused"] == 1
    assert slot["completed"] == slot["failed"] == 0 and slot["state"] == "NOT_STARTED"
    assert entered == [] and snapshot["active"] == [] and snapshot["work"] is None
    with pytest.raises(ValueError, match="non-opaque"), journal.phase("input-state-load", opaque=True):
        entered.append("forbidden-opaque")
    refused, _clock = journal_fixture()
    clock_error = ValueError("generated-normal-clock-refusal")
    with monkeypatch.context() as changed:
        def failed_clock():
            raise clock_error
        changed.setattr(refused, "clock", failed_clock)
        with pytest.raises(ValueError) as observed, refused.phase("input-state-load"):
            entered.append("forbidden-clock")
    assert observed.value is clock_error and entered == []
    assert refused.snapshot()["preparation"]["input-state-load"]["state"] == "NOT_STARTED"
    owner_error = OSError("generated-owner-refusal")
    owned, _clock = journal_fixture()
    def owner():
        raise owner_error
    owned.guard = owner
    with pytest.raises(OSError) as observed, owned.phase("arm-data", opaque=True):
        entered.append("forbidden-owner")
    assert observed.value is owner_error and entered == [] and owned.snapshot()["active"] == []
    assert owned.emit_attempts == 1 and owned.stream.getvalue() == ""


def test_math_preparation_fixed_retention_under_event_wire_stream_terminal_pressure():
    journal, now = journal_fixture()
    journal.error("generated-first", RuntimeError("retain-primary"))
    for _ in range(100):
        for name in runtime.PREPARATION_PHASES:
            with journal.phase(name):
                now[0] += .001
    for _ in range(20):
        journal.error("generated-secondary", RuntimeError("\U0001f9ea"*3000))
    journal.acquire_work({"physics_requested": 1, "physics_unreturned_unknown": True}, {"unknown_work": True})
    totals = copy.deepcopy(journal.preparation)
    assert journal.emit_attempts == 0 and journal.stream.getvalue() == ""
    with journal.phase("input-state-load"), journal.fence("pre_array"):
        now[0] += 1.0
        before = copy.deepcopy(journal.preparation)
        for cap in (16384, runtime.MAX_JSON_BYTES):
            raw = journal.encoded(cap=cap)
            value = json.loads(raw)
            assert len(raw) <= cap and len(value["events"]) <= 63
            assert value["bytes_truncated"] and value["events_dropped"] > 0
            assert set(value["preparation"]) == set(runtime.PREPARATION_PHASES)
            assert all(value["preparation"][name]["completed"] == 100 for name in runtime.PREPARATION_PHASES)
            assert value["first_error"]["message"] == "retain-primary" and value["work"]["construction"]["unknown_work"]
            assert value["active"][-1]["phase"] == "input-state-load" and value["active_fence"]["label"] == "pre_array"
            assert value["cold_fences_nested_wall"]["pre_array"]["attempted"] == 1
            assert journal.preparation == before
        for _ in range(15):
            assert journal.emit()
        assert journal.emit() is False and journal.terminal_failure()
        assert journal.emit_attempts == 16 and journal.stream_bytes <= 524288
        with pytest.raises(ValueError, match="once"):
            journal.terminal_failure()
        snapshot = journal.snapshot()
        assert snapshot["terminal"]["state"] == "RETURNED" and len(snapshot["preparation"]) == 8
        assert journal.preparation == before
    assert totals["input-state-load"]["attempted"] == 100
    assert journal.preparation["input-state-load"]["attempted"] == 101
    assert len(journal.stream.getvalue().splitlines()) == 16


def generated_preparation_inputs(arm, changed, journal, now, *, fail_at=None):
    """Inert paths and generated arrays only; no Original leaf/header/digest.

    This fixture exercises the production arm_data statements with a separately
    assembled callback-order oracle. File/asset methods never reach the disk.
    """
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    from tools import verify_c_banc_m3_startup_ray as ray

    trace, expected = [], []
    sentinel = RuntimeError("generated-metadata-callback")
    selected = {role: pin(role + ".npz") for role in ("recurrent_mask", "reference", "startup", "audited_candidate")}
    names = [spec["path"] for spec in selected.values()] + ["generated-"+str(i)+( ".npy" if i < 2 else ".bin") for i in range(131)]
    files = {name: pin(name) for name in names}
    inventory = {"selected_archives": selected, "files": files, "graph_bundle": "generated-graph.json",
                 "initial_whole": "generated-state.json", "graph_hash": "generated-graph-hash"}
    excluded = {selected["audited_candidate"]["path"]}
    if arm == "neutral":
        excluded.add(selected["reference"]["path"])
    allowed = {name: spec for name, spec in files.items() if name not in excluded}
    dependencies = {"network_asset_download_permitted": False, "mesh_cache_complete_currently": True,
                    "native_constructor_asset_bytes": {"/generated-only/asset.bin": pin("asset.bin")}}
    def mark(*row):
        trace.append(row)
        now[0] += .001
    class InertPath:
        def __init__(self, name):
            self.name = name
            self.suffix = ".npy" if name.endswith(".npy") else ".bin"
            self.parents = ()
        def stat(self):
            mark("stat", self.name)
            return SimpleNamespace(st_size=7)
        def is_absolute(self):
            mark("absolute", self.name)
            return True
        def is_file(self):
            mark("file", self.name)
            return True
        def is_symlink(self):
            mark("symlink", self.name)
            return False
        def lstat(self):
            mark("lstat", self.name)
            return SimpleNamespace(st_file_attributes=0)
        def open(self, mode):
            assert mode == "rb"
            mark("open-generated", self.name)
            return io.BytesIO(b"generated-inert-header")
    def relative(name, root):
        assert root == "generated-no-disk-root"
        mark("path", name)
        return InertPath(name)
    def absolute(name):
        assert name == "/generated-only/asset.bin"
        mark("asset-path", name)
        return InertPath(name)
    def check(label="science"):
        mark("check", label)
        if journal is not None:
            journal.science()
        return True
    def fence(label):
        mark("fence", label)
        if journal is not None:
            with journal.fence(label):
                check()
        else:
            check()
    class Core:
        def digest(self, path, timely):
            mark("generated-digest", path.name)
            timely()
            return "1"*64
        def strict_npy_header(self, stream):
            mark("generated-header",)
            assert stream.read() == b"generated-inert-header"
        def load_selected(self, root, descriptor, ledger, *, timely, generated):
            assert root == "generated-no-disk-root" and generated is False
            mark("selected", descriptor["path"])
            timely()
            ledger.append({"path": descriptor["path"], "expected_sha256": descriptor["sha256"], "bytes": 7})
            if descriptor["path"] == selected["recurrent_mask"]["path"]:
                groups = np.arange(8547, dtype=np.int64)
                if fail_at == "coverage":
                    groups[-1] = 0
                return {"positions": np.arange(8547, dtype=np.int64), "edge_parameters": groups}
            return {"generated": descriptor["path"]}
    def native():
        mark("native",)
        return Core()
    def leaf(spec, root):
        assert root == "generated-no-disk-root"
        mark("leaf-generated", spec["path"])
        value = dependencies if spec["path"] == "dependencies.json" else inventory
        assert spec["path"] in ("dependencies.json", "inventory.json")
        return json.dumps(value).encode("utf8")
    class Graph:
        @property
        def hash(self):
            mark("graph-hash",)
            return "generated-wrong-hash" if fail_at == "graph" else inventory["graph_hash"]
    graph, saved = Graph(), {"generated": "saved"}
    def graph_load(path):
        mark("graph-load", path.name)
        return graph
    def state_load(path, *, max_files):
        assert max_files == 123
        mark("state-load", path.name)
        return saved
    def checked_checkpoint(value):
        assert value is saved
        mark("checkpoint",)
    api = {"GraphStore": SimpleNamespace(load=graph_load), "StateStore": SimpleNamespace(load=state_load),
           "recovery": SimpleNamespace(checked_checkpoint=checked_checkpoint)}
    def metadata(root, actual):
        assert root == "generated-no-disk-root" and actual is inventory
        mark("whole-metadata",)
        if fail_at == "metadata":
            raise sentinel
    def resolve():
        mark("resolve-api",)
        return api
    inputs = {"body_native_dependencies": pin("dependencies.json"),
              "neutral_registration": {"body_inventory": pin("inventory.json"),
                                       "body_native_dependencies": pin("dependencies.json"), "source_input_roster": allowed}}
    timer = SimpleNamespace(root="generated-no-disk-root", inputs=inputs, fence=fence, check=check)
    if journal is not None:
        timer.runtime_journal = journal
    changed.setattr(ref, "native", native)
    changed.setattr(ray, "path", relative)
    changed.setattr(ray, "Path", absolute)
    changed.setattr(ray, "leaf", leaf)
    raw = {"theta_actual": np.full(8547, 2.0, dtype=np.float64),
           "selected_weights_candidate": np.array([3.0, 4.0], dtype=np.float32)}
    def expected_fence():
        expected.extend([("fence", "pre_array"), ("check", "science")])
    expected.append(("native",))
    if arm == "neutral":
        expected.append(("leaf-generated", "inventory.json"))
    expected_fence()
    for name in allowed:
        expected_fence()
        expected.extend([("path", name), ("stat", name), ("generated-digest", name), ("check", "science")])
        if name.endswith(".npy"):
            expected_fence()
            expected.extend([("open-generated", name), ("generated-header",)])
    expected.append(("leaf-generated", "dependencies.json"))
    for name in dependencies["native_constructor_asset_bytes"]:
        expected_fence()
        expected.extend([("asset-path", name), ("absolute", name), ("file", name), ("symlink", name),
                         ("stat", name), ("symlink", name), ("lstat", name),
                         ("generated-digest", name), ("check", "science")])
    expected_fence()
    expected.append(("whole-metadata",))
    if fail_at != "metadata":
        expected.append(("resolve-api",))
        expected_fence()
        expected.extend([("path", inventory["graph_bundle"]), ("graph-load", inventory["graph_bundle"])])
        expected_fence()
        expected.extend([("path", inventory["initial_whole"]), ("state-load", inventory["initial_whole"]), ("graph-hash",)])
        if fail_at != "graph":
            expected.append(("checkpoint",))
            for role in (("recurrent_mask", "startup") if arm == "neutral" else ("recurrent_mask", "reference", "startup")):
                expected_fence()
                expected.extend([("selected", selected[role]["path"]), ("check", "science")])
    return SimpleNamespace(ray=ray, timer=timer, raw=raw, inventory=inventory, b=SimpleNamespace(whole_metadata=metadata, resolve_original_api=resolve),
                           trace=trace, expected=expected, sentinel=sentinel, graph=graph, saved=saved)


@pytest.mark.parametrize("arm", ["neutral", "candidate"])
@pytest.mark.parametrize("fail_at", [None, "metadata", "graph", "coverage"])
def test_gate_preparation_generated_arm_and_no_journal_callback_order(monkeypatch, arm, fail_at):
    traces = []
    for observed in (False, True):
        journal, now = journal_fixture()
        with monkeypatch.context() as changed:
            generated = generated_preparation_inputs(arm, changed, journal if observed else None, now, fail_at=fail_at)
            if fail_at is None:
                result = generated.ray.arm_data(generated.raw, generated.inventory, generated.timer, arm, b=generated.b)
                assert result["graph"] is generated.graph and result["saved"] is generated.saved
                assert result["theta"].shape == (8547,) and result["positions"].dtype.name == "int64"
                assert (result["theta"] == (1.0 if arm == "neutral" else 2.0)).all()
                assert ("reference" in result) == (arm == "candidate")
                assert ("selected_weights" in result) == (arm == "candidate")
                if arm == "candidate":
                    assert result["selected_weights"] is not generated.raw["selected_weights_candidate"]
            else:
                exception = RuntimeError if fail_at == "metadata" else ValueError
                with pytest.raises(exception) as error:
                    generated.ray.arm_data(generated.raw, generated.inventory, generated.timer, arm, b=generated.b)
                if fail_at == "metadata":
                    assert error.value is generated.sentinel
                else:
                    assert str(error.value) == ("same source graph" if fail_at == "graph" else "actual source group coverage")
            assert generated.trace == generated.expected
            traces.append(generated.trace)
        snapshot = journal.snapshot()
        assert journal.emit_attempts == 0 and journal.stream.getvalue() == "" and snapshot["work"] is None
        slots = snapshot["preparation"]
        if observed:
            names = list(runtime.PREPARATION_PHASES)
            last = {None: 7, "metadata": 2, "graph": 5, "coverage": 7}[fail_at]
            for index, name in enumerate(names):
                assert slots[name]["attempted"] == (1 if index <= last else 0)
                assert slots[name]["state"] == ("NOT_STARTED" if index > last else
                    ("FAILED" if fail_at is not None and index == last else "RETURNED"))
            assert snapshot["active"] == []
            if fail_at is None:
                assert snapshot["first_error"] is None
                assert sum(row["nested_fence"]["attempted"] for row in slots.values()) == snapshot["cold_fences_nested_wall"]["pre_array"]["attempted"]
        else:
            assert all(row["state"] == "NOT_STARTED" and row["attempted"] == 0 for row in slots.values())
            assert snapshot["first_error"] is None
    assert traces[0] == traces[1]  # generated no-journal callback/input/check order equivalence


def test_gate_preparation_existing_error_capture_preserves_pending_exception_only(monkeypatch):
    primary = RuntimeError("generated-worker-callback")
    clock_error = ValueError("generated-error-capture-clock")
    for operation in ("error", "acquire-errors"):
        journal, _ = journal_fixture()
        with monkeypatch.context() as changed:
            def unavailable():
                raise clock_error
            changed.setattr(journal, "clock", unavailable)
            try:
                raise primary
            except RuntimeError as acquired:
                if operation == "error":
                    journal.error("arm-data", acquired)
                else:
                    journal.acquire_errors({"first_error": {"stage": "arm-data", "type": "RuntimeError", "message": str(acquired)},
                                            "secondary_errors": []})
                assert acquired is primary and journal.first_error["message"] == str(primary)
            with pytest.raises(ValueError) as observed:
                if operation == "error":
                    journal.error("normal-capture", RuntimeError("generated-normal"))
                else:
                    journal.acquire_errors({"first_error": {"stage": "normal", "type": "RuntimeError", "message": "generated-normal"}})
            assert observed.value is clock_error
        snapshot = journal.snapshot()
        assert snapshot["first_error"]["message"] == str(primary)
        assert snapshot["first_error"]["observed_s"] is None and snapshot["observation_incomplete"]
        assert snapshot["work"] is None and journal.emit_attempts == 0


def test_gate_preparation_normal_finish_clock_refusal_after_returned_callback(monkeypatch):
    journal, _ = journal_fixture()
    callback, failure = [], ValueError("generated-normal-finish-clock")
    with monkeypatch.context() as changed, pytest.raises(ValueError) as observed, journal.phase("input-result-check"):
        callback.append("returned-once")
        def refused():
            raise failure
        changed.setattr(journal, "clock", refused)
    snapshot = journal.snapshot()
    slot = snapshot["preparation"]["input-result-check"]
    assert observed.value is failure and callback == ["returned-once"]
    assert slot["completed"] == 1 and slot["failed"] == 0 and slot["state"] == "RETURNED"
    assert slot["inclusive_wall_s"] is None and slot["observation_incomplete"]
    assert snapshot["active"] == [] and snapshot["observation_incomplete"] and snapshot["work"] is None
    assert journal.emit_attempts == 0


def test_gate_preparation_fourteenth_existing_frame_rejects_fifteenth_entry():
    journal, _ = journal_fixture()
    starts = [journal.start("body-plan") for _ in range(14)]
    opened = copy.deepcopy(journal.active)
    callbacks = []
    assert runtime.MAX_ACTIVE_PHASES == 14 and len(runtime.PHASES) == 22
    with pytest.raises(ValueError, match="bounded diagnostic nesting"), journal.phase("input-graph-load"):
        callbacks.append("forbidden-fifteenth")
    snapshot = journal.snapshot()
    slot = snapshot["preparation"]["input-graph-load"]
    assert callbacks == [] and journal.active == opened and len(snapshot["active"]) == 14
    assert (slot["attempted"], slot["entry_refused"], slot["completed"], slot["failed"]) == (1, 1, 0, 0)
    assert slot["active"] == 0 and slot["state"] == "NOT_STARTED"
    assert snapshot["preparation"]["input-state-load"]["attempted"] == 0
    assert snapshot["first_error"]["stage"] == "input-graph-load"
    for start in reversed(starts):
        journal.finish("body-plan", start)
    closed = journal.snapshot()
    assert closed["active"] == [] and closed["last_phase"]["kind"] == "RETURN"
    assert sum(row["kind"] == "START" and row["phase"] == "body-plan" for row in closed["events"]) == 14
    assert sum(row["kind"] == "RETURN" and row["phase"] == "body-plan" for row in closed["events"]) == 14
    assert closed["preparation"]["input-graph-load"]["state"] == "NOT_STARTED"
    assert journal.emit_attempts == 0 and journal.stream.getvalue() == "" and closed["work"] is None


def test_gate_preparation_start_event_failure_never_enters_callback_or_strands_outer(monkeypatch):
    journal, _ = journal_fixture()
    sentinel = OSError("generated-preparation-START-observation")
    callbacks, failed_starts, outer_close = [], [], []
    event, finish = journal._event, journal.finish
    def failed_event(name, kind, start=None):
        if name == "input-native-assets" and kind == "START":
            failed_starts.append(name)
            raise sentinel
        return event(name, kind, start)
    def observed_finish(name, start, **kwargs):
        if name == "arm-data":
            outer_close.append(copy.deepcopy(journal.active))
        return finish(name, start, **kwargs)
    with monkeypatch.context() as changed:
        changed.setattr(journal, "_event", failed_event)
        changed.setattr(journal, "finish", observed_finish)
        with pytest.raises(OSError) as acquired, journal.phase("arm-data", opaque=True):
            with journal.phase("input-native-assets"):
                callbacks.append("forbidden-payload")
            with journal.phase("input-whole-metadata"):
                callbacks.append("forbidden-next")
    snapshot = journal.snapshot()
    slot = snapshot["preparation"]["input-native-assets"]
    assert acquired.value is sentinel and callbacks == [] and failed_starts == ["input-native-assets"]
    assert snapshot["first_error"]["stage"] == "input-native-assets"
    assert snapshot["first_error"]["type"] == "OSError" and snapshot["first_error"]["message"] == str(sentinel)
    assert (slot["attempted"], slot["entry_refused"], slot["completed"], slot["failed"]) == (1, 1, 0, 0)
    assert slot["state"] == "NOT_STARTED" and slot["active"] == 0
    assert snapshot["preparation"]["input-whole-metadata"]["state"] == "NOT_STARTED"
    assert snapshot["preparation"]["input-whole-metadata"]["attempted"] == 0
    assert outer_close == [[{"phase": "arm-data", "started_s": 0.0}]]
    assert snapshot["active"] == [] and snapshot["active_fence"] is None and snapshot["work"] is None
    assert [(row["phase"], row["kind"]) for row in snapshot["events"]] == [
        ("arm-data", "START"), ("input-native-assets", "FAIL"), ("arm-data", "FAIL")]
    assert journal.emit_attempts == len(journal.stream.getvalue().splitlines()) == 1


def conserved_put(root, name, value):
    from tools import banc_m3_conserved_resource as resource
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf8")
    return resource.fixture_text_write(root, name, raw)


def conserved_copy(root, spec):
    from tools import banc_m3_conserved_resource as resource
    try:
        raw = resource.source_text(spec, admission.ROOT)
    except (ValueError, OSError):
        from tests.banc_m3_checkpoint_fixture import archive_pair
        try:
            raw = archive_pair(spec)
        except (ValueError, OSError):
            pass
        else:
            return resource.fixture_text_write(root, spec['path'], raw)
        if spec["path"] not in resource.G_REPLACEMENTS:
            raise
        for mapping in (resource.j_archives, resource.i_archives, resource.h_archives, resource.g_archives):
            item = mapping(admission.ROOT).get(spec["path"])
            if item is not None and item["source"] == spec:
                raw = resource.raw_pin(item["archive"], admission.ROOT, 524288)
                break
        else:
            raise
    return resource.fixture_text_write(root, spec["path"], raw)


def conserved_historical_copy(root, spec, mapped):
    from tools import banc_m3_conserved_resource as resource
    item = mapped.get(spec["path"])
    if item is not None:
        resource.exact(spec, item["source"], "fixture exact historical archive mapping")
        raw = resource.raw_pin(item["archive"], admission.ROOT, 524288)
        return resource.fixture_text_write(root, spec["path"], raw)
    return conserved_copy(root, spec)

def conserved_recompute(ledger, policy, grant, generated_cap, original_cap):
    generated = original = 0.0
    for i, debit in enumerate(ledger["debits"]):
        if i >= 118 and debit["mode"] == "generated":
            generated += float(debit["actual_outer_full_return_s"])
        if i >= 165 and debit["mode"] == "original":
            original += float(debit["actual_outer_full_return_s"])
    stage = float(policy["generated_stage_before_s"]) + generated
    ledger.update(direct_science_generated_used_s=generated, generated_stage_used_s=stage,
                  generated_shared_cumulative_used_s=float(policy["generated_shared_before_s"]) + generated,
                  generated_shared_remaining_s=float(generated_cap) - float(policy["generated_shared_baseline_s"]) - stage,
                  original_cumulative_used_s=float(grant["original_cumulative_before_s"]) + original,
                  original_stage_used_s=float(grant["original_stage_before_s"]) + original,
                  startup_ray_original_local_used_s=float(grant["science_local_used_before_s"]) + original,
                  original_remaining_s=float(original_cap) - float(grant["original_cumulative_before_s"]) - original)


def conserved_junit(root, name, rows):
    import xml.etree.ElementTree as ET
    document = ET.Element("testsuites")
    suite = ET.SubElement(document, "testsuite", name="generated-metadata-fixture", tests=str(len(rows)),
                          errors="0", failures="0", skipped="0")
    for row in rows:
        ET.SubElement(suite, "testcase", classname=row["classname"], name=row["name"], time="0.001")
    raw = ET.tostring(document, encoding="utf8")
    from tools import banc_m3_conserved_resource as resource
    return resource.fixture_text_write(root, name, raw)


def conserved_fake_paid(root, ledger, source, call, role, argv, caps, *, name, request=None):
    from tools import banc_m3_conserved_resource as resource
    default_request = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": call,
               "role": role, "source_freeze": source, "generated_behavior": "command", "generated_argv": argv,
               "limits": dict(zip(("outer_s", "child_s", "cleanup_s"), caps, strict=True))}
    request = default_request if request is None else request
    request_spec = conserved_put(root, resource.BASE + "/" + name + "-request.json", request)
    outer_spec = conserved_put(root, resource.BASE + "/" + name + "-outer.json",
                               {"exit_code": 0, "all_owned_ended": True, "reservation_released": True, "stdout_eof": True,
                                "stderr_eof": True, "unknown_owned_identities": {}, "timeout": False, "request": request_spec})
    trace_spec = conserved_put(root, resource.BASE + "/" + name + "-trace.json",
                               {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                                "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125,
                                "exit_code": 0, "source_freeze": source, "request": request_spec})
    cost = {"cost_id": call, "mode": "generated", "request": request_spec, "source_freeze": source,
            "outer": outer_spec, "trace": trace_spec, "actual_outer_full_return_s": .125, "exit_code": 0,
            "charged_once": True, "all_owned_ended": True, "reservation_released": True, "cap_exceeded": False}
    cost_spec = conserved_put(root, resource.BASE + "/" + name + "-cost.json", cost)
    ledger["debits"].append({"cost_id": call, "mode": "generated", "actual_outer_full_return_s": .125,
                             "actual_returned_exit_code": 0, "descriptor": cost_spec})
    ledger["direct_science_calls"][call] = {"state": "charged", "mode": "generated", "request": request_spec, "reserved_s": caps[0]}
    return cost_spec


def conserved_metadata_fixture(root):
    from tools import banc_m3_conserved_resource as resource
    conserved_runtime_recovery_copy(root)  # only exact JSON policy/transfer anchors
    for spec in (resource.HISTORICAL_PIN, resource.POLICY_PIN, resource.J_PIN, resource.HUMAN_PIN,
                 resource.ROOT_PIN, resource.ROSTER_PIN, resource.E_DESCRIPTOR_PIN):
        conserved_copy(root, spec)
    historical = resource.read_pin(resource.HISTORICAL_PIN, root, 524288)
    policy, grant = resource.read_pin(resource.POLICY_PIN, root), resource.read_pin(resource.J_PIN, root)
    frozen = {}
    for name in (*resource.BOOTSTRAP_SOURCES, resource.ROSTER_PIN["path"], resource.E_PATH,
                 "tests/test_c_banc_contact_body_retry.py", "tests/test_c_banc_startup_ray_execution.py"):
        spec = admission.spec_for(name, admission.ROOT)
        conserved_copy(root, spec)
        frozen[name] = spec
    source = conserved_put(root, resource.E_SOURCE_PATH,
                           {"schema": "flylab.m3.contact-learning.source-freeze.v1",
                            "ROOT_review_cycle_complete": True, "sources": frozen})
    before = copy.deepcopy(historical)
    bootstrap_cost = conserved_fake_paid(root, before, source, resource.BOOTSTRAP_CALL, "execution",
                                        ["-m", "pytest", resource.BOOTSTRAP_NODE, "--tb=short", "-q",
                                         "--junitxml=" + resource.BASE + "/parent-conserved-resource-bootstrap-junit.xml"],
                                        (11, 8, 3), name="generated-bootstrap")
    conserved_recompute(before, policy, grant, 1700, 16200)
    xml = conserved_junit(root, resource.BASE + "/parent-conserved-resource-bootstrap-junit.xml",
                         [{"classname": "tests.test_c_banc_contact_runtime_journal", "name": resource.BOOTSTRAP_NODE.split("::")[1]}])
    bootstrap = conserved_put(root, resource.BASE + "/generated-bootstrap-proof.json",
                              {"schema": "flylab.m3.conserved-resource.bootstrap.v1", "cost": bootstrap_cost,
                               "source_freeze": source, "fixture_node": resource.BOOTSTRAP_NODE, "junit": xml})
    before_spec = conserved_put(root, resource.BASE + "/generated-allocation-before.json", before)
    event = {"schema": "flylab.m3.conserved-resource-allocation.v1", "event_id": "conserved-resource-20261008-600",
             "before_ledger": before_spec, "historical_closed282": resource.HISTORICAL_PIN, "initial_policy": resource.POLICY_PIN,
             "transfers": {key: before[key] for key in resource.TRANSFER_KEYS}, "j_grant": resource.J_PIN,
             "direct_human_auto_authority": resource.HUMAN_PIN,
             "root_decision": {"source": resource.ROOT_PIN, "pool_transfer_s": 600, "human_numerical_approval": False},
             "old_caps": resource.OLD_CAPS, "new_caps": resource.NEW_CAPS, "delta_s": resource.DELTA,
             "combined_global_s": 17900, "new_resource_grant_s": 0, "bootstrap": bootstrap}
    event_spec = conserved_put(root, resource.ALLOCATION_PATH, event)
    after = resource.allocated_snapshot(before, event_spec, policy, grant)
    return SimpleNamespace(root=root, historical=historical, before=before, after=after, event=event,
                           event_spec=event_spec, policy=policy, grant=grant, source=source, bootstrap=bootstrap)


def conserved_qualification_fixture(fixture):
    from tools import banc_m3_conserved_resource as resource
    live = copy.deepcopy(fixture.after)
    costs = []
    for role in ("lint", "math", "gate-a", "gate-b"):
        plan = resource.generated_plan(role, fixture.root)
        costs.append(conserved_fake_paid(fixture.root, live, fixture.source, plan["call"], plan["role"],
                                         plan["argv"], plan["caps"], name="generated-E-" + role))
    conserved_recompute(live, fixture.policy, fixture.grant, 2300, 15600)
    roster, suites = resource.complete_roster(fixture.root), []
    for role in ("math", "gate-a", "gate-b"):
        rows = roster[role.replace("-", "_")]
        xml = conserved_junit(fixture.root, resource.BASE + "/parent-contact-learning-e-" + role + "-junit.xml", rows)
        suites.append({"file": xml, "node_ids": [row["node_id"] for row in rows], "names": [row["name"] for row in rows],
                       "classnames": [row["classname"] for row in rows], "count": len(rows)})
    qualified = {"schema": "flylab.m3.direct-science.qualification.v2", "epoch": resource.EPOCH, "old_E360": "UNPASSED",
                 "allocation": fixture.event_spec, "qualification_roster": resource.ROSTER_PIN, "source_freeze": fixture.source,
                 "actual_costs": costs, "tests": suites, "evidence": costs}
    return live, qualified


def conserved_native_definitions(root, *, account_pin=None, contact_pin=None):
    from tests.banc_m3_checkpoint_fixture import stage_legacy
    stage_legacy(root)
    from tools import banc_m3_conserved_resource as resource
    return resource.d19_r1_defs(root,account_pin,contact_pin)
    account_cap = 393216 if account_pin is None else 262144
    account = resource.text_spec(admission.BASE + "/parent-direct-science-account.ps1", admission.ROOT, account_cap) if account_pin is None else account_pin
    source = resource.raw_pin(account, admission.ROOT, account_cap).decode("utf8")
    action_guard = "Need ($descriptor.path -cin @(Conserved-ActiveEventPaths)) 'Exactly two schema-bound prospective routes'"
    assert source.count(action_guard) == 1  # actual action and definitions host share production path helper
    assert source.count("if($descriptor.path -ceq ($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json')){Conserved-F-Unused $ledger}") == 1
    definitions = source[source.index("$ErrorActionPreference="):source.index("$ledger=Get-Content -LiteralPath (Safe $ledgerName)")]
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert definitions.count(anchor) == 1
    contact = resource.text_spec(admission.BASE + "/parent-contact-learning-contract-native.ps1", admission.ROOT, 262144) if contact_pin is None else contact_pin
    contact_definitions = resource.raw_pin(contact, admission.ROOT, 262144)
    assert len(contact_definitions) <= 262144
    return definitions.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + root.as_posix().replace("'", "''") + "')") + "\n" + contact_definitions.decode("utf8")


def conserved_source_negatives(fixture):
    from tools import banc_m3_conserved_resource as resource
    freeze = resource.read_pin(fixture.source, fixture.root)
    cases = []
    for i, name in enumerate(resource.BOOTSTRAP_SOURCES):
        wrong = copy.deepcopy(freeze)
        del wrong["sources"][name]
        cases.append(conserved_put(fixture.root, resource.BASE + "/generated-missing-critical-" + str(i) + ".json", wrong))
    name = resource.HELPER_SOURCE
    for i, kind in enumerate(("Count", "Keys", "bool", "float", "path", "hash")):
        wrong = copy.deepcopy(freeze)
        spec = wrong["sources"][name]
        if kind == "Count":
            spec["Count"] = 3
        elif kind == "Keys":
            spec["Keys"] = ["path", "bytes", "sha256"]
        elif kind == "bool":
            spec["bytes"] = True
        elif kind == "float":
            spec["bytes"] = float(spec["bytes"])
        elif kind == "path":
            spec["path"] = "tools/banc_m3_startup_ray_supervisor.py"
        else:
            spec["sha256"] = "0"*64
        cases.append(conserved_put(fixture.root, resource.BASE + "/generated-critical-descriptor-" + str(i) + ".json", wrong))
    return cases


def conserved_native_window(entry, limits, now):
    """Use the real parent's ENTRY; never grant the Native host a new clock."""
    from tools import banc_m3_conserved_resource as resource
    resource.need(type(entry) is float and type(now) is float and 0 < entry <= now, "actual finite parent ENTRY clock")
    resource.finite(entry, "parent ENTRY")
    resource.finite(now, "current ENTRY clock")
    for key in ("outer_s", "child_s", "cleanup_s"):
        resource.need(type(limits[key]) is int and limits[key] > 0, "typed actual parent limits")
    resource.need(limits["child_s"] + limits["cleanup_s"] <= limits["outer_s"], "parent cleanup reserve kept")
    child_end = entry + min(limits["child_s"], limits["outer_s"] - limits["cleanup_s"])
    native_end, cleanup_end = child_end - 1.0, child_end - .1
    resource.need(now < native_end, "pre-spawn refusal: insufficient actual remaining Native interval")
    return native_end, cleanup_end, child_end, entry + limits["outer_s"]


def conserved_native_parent_call(q, harness, resource):
    return (q["mode"] == "generated" and q["role"] == "execution"
            and (q["call_id"] in (resource.BOOTSTRAP_CALL, resource.QUAL_CALLS[3], resource.F_QUAL_CALLS[3], resource.I_QUAL_CALLS[3], resource.J_QUAL_CALLS[3], "direct-science-20261008-contact-k-bootstrap", "direct-science-20261008-contact-k-gate-b", "direct-science-20261008-contact-l-bootstrap", "direct-science-20261008-contact-l-gate-a", "direct-science-20261008-contact-l-gate-b", "direct-science-20261008-contact-l2-bootstrap", "direct-science-20261008-contact-l2-gate-a", "direct-science-20261008-contact-l2-gate-b", "direct-science-20261008-contact-l3-bootstrap", "direct-science-20261008-contact-l3-gate-a", "direct-science-20261008-contact-l3-gate-b", "direct-science-20261008-contact-l4-bootstrap", "direct-science-20261008-contact-l4-gate-a", "direct-science-20261008-contact-l4-gate-b", "direct-science-20261008-contact-l5-bootstrap", "direct-science-20261008-contact-l5-gate-a", "direct-science-20261008-contact-l5-gate-b", "direct-science-20261008-contact-l6-bootstrap", "direct-science-20261008-contact-l6-gate-a", "direct-science-20261008-contact-l6-gate-b", "direct-science-20261008-contact-l7-bootstrap", "direct-science-20261008-contact-l7-gate-a", "direct-science-20261008-contact-l7-gate-b", "direct-science-20261008-contact-l8-bootstrap", "direct-science-20261008-contact-l8-gate-a", "direct-science-20261008-contact-l8-gate-b", "direct-science-20261008-contact-l9-bootstrap", "direct-science-20261008-contact-l9-gate-a", "direct-science-20261008-contact-l9-gate-b", "direct-science-20261008-contact-l9r2-bootstrap", "direct-science-20261008-contact-l9r2-gate-a", "direct-science-20261008-contact-l9r2-gate-b", "direct-science-20261008-contact-l10-bootstrap", "direct-science-20261008-contact-l10-gate-a", "direct-science-20261008-contact-l10-gate-b", "direct-science-20261008-contact-l11-bootstrap", "direct-science-20261008-contact-l11-gate-a", "direct-science-20261008-contact-l11-gate-b", "direct-science-20261008-contact-l12-bootstrap", "direct-science-20261008-contact-l12-gate-a", "direct-science-20261008-contact-l12-gate-b", "direct-science-20261008-contact-l13-bootstrap", "direct-science-20261008-contact-l13-gate-a", "direct-science-20261008-contact-l13-gate-b", "direct-science-20261008-contact-l13r2-bootstrap", "direct-science-20261008-contact-l13r2-gate-a", "direct-science-20261008-contact-l13r2-gate-b", "direct-science-20261009-contact-s22-bootstrap", "direct-science-20261009-contact-s22-gate-a", "direct-science-20261009-contact-s22-gate-b", "direct-science-20261009-contact-s23-bootstrap", "direct-science-20261009-contact-s23-gate-a", "direct-science-20261009-contact-s23-gate-b", "direct-science-20261009-contact-s24-bootstrap", "direct-science-20261009-contact-s24-gate-a", "direct-science-20261009-contact-s24-gate-b", "direct-science-20261009-contact-s25-bootstrap", "direct-science-20261009-contact-s25-gate-a", "direct-science-20261009-contact-s25-gate-b", "direct-science-20261009-contact-s26-bootstrap", "direct-science-20261009-contact-s26-gate-a", "direct-science-20261009-contact-s26-gate-b", "direct-science-20261009-contact-s27-bootstrap", "direct-science-20261009-contact-s27-gate-a", "direct-science-20261009-contact-s27-gate-b")
                 or q["call_id"] in (harness.BOOTSTRAP_CALL, harness.QUAL_CALLS[2], harness.QUAL_CALLS[3])))


def conserved_run_native(root, argv, *, observation_root=None):
    """Actual nested Job; full Native evidence succeeds or fails under one ENTRY."""
    import threading
    import time

    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    from tools import verify_c_banc_m3_startup_ray as ray

    observation_root = root if observation_root is None else observation_root
    observed = SimpleNamespace(returncode=None, stdout=b"", stderr=b"", first_error=None, secondary_errors=[],
                               timeout=False, pre_spawn_refused=True, all_owned_ended=True, stdout_eof=False,
                               stderr_eof=False, worker_pid=None, worker_creation_ticks=None, identities={},
                               unknown_owned_identities={}, entry=None, limits=None, window=None)
    job, readers, chunks, eof, stream_errors = None, [], [[], []], [False, False], []
    cleanup_end = None
    try:
        resource.need("FLYLAB_DIRECT_SCIENCE_OWNER" in os.environ, "actual owned generated parent required; no standalone test launch")
        token = ray.json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8"))
        resource.exact(token["schema"], supervisor.SCHEMA + ".lock-owner.v1", "actual supervisor-issued parent")
        resource.exact(resource.read_pin(resource.text_spec(admission.SHARED_LOCK, admission.ROOT, 16384), admission.ROOT, 16384), token, "same live parent exclusion witness")
        resource.exact(supervisor.creation_ticks(token["owner_pid"]), token["owner_creation_ticks"], "live actual parent identity")
        resource.need(supervisor.in_named_job(token["job_name"]), "current fixture belongs to the actual parent Job")
        q = resource.read_pin(token["request"], admission.ROOT)
        resource.exact(q["call_id"], token["call_id"], "same actual parent request")
        resource.need(q["mode"] == "generated" and q["role"] == "execution", "actual Generated execution parent")
        from tools import banc_m3_checkpoint_harness_runtime as harness
        resource.need(conserved_native_parent_call(q, harness, resource), "one actual fixed spent-bootstrap fixture or complete E/F/I gate-B parent")
        resource.need(q["call_id"] != resource.BOOTSTRAP_CALL or resource.BOOTSTRAP_DISPATCHABLE,
                      "NOT_DISPATCHABLE: no observed complete old-budget fit; conservative Source policy")
        expected = (11, 8, 3) if q["call_id"] == resource.BOOTSTRAP_CALL else resource.QUAL_CAPS[3]
        from tools import banc_m3_checkpoint_runtime as checkpoint
        if q["call_id"] in (harness.BOOTSTRAP_CALL, harness.QUAL_CALLS[2], harness.QUAL_CALLS[3]):
            checkpoint = harness
        if q['call_id'] in (checkpoint.BOOTSTRAP_CALL, checkpoint.QUAL_CALLS[3]) or checkpoint is harness and q['call_id'] == harness.QUAL_CALLS[2]:
            if checkpoint is harness:
                from tests.banc_m3_checkpoint_harness_fixture import parent
            else:
                from tests.banc_m3_checkpoint_fixture import parent
            expected = parent(q, token)
        if q["call_id"] == resource.I_QUAL_CALLS[3]:
            expected = (360, 340, 20)
            from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
            current = resource.read_pin(resource.text_spec(q["ledger_path"], admission.ROOT, 524288), admission.ROOT, 524288)
            resource.exact(resource.current_caps(current, admission.ROOT, generated_budget_limit), resource.NEW_CAPS, "strict actual I parent ledger")
            recovery, _anchor, _policy, _grant = resource.i_preflight_recovery(current[resource.I_KEY], admission.ROOT, generated_budget_limit)
            resource.exact(resource.i_progress(current, recovery, current[resource.I_KEY], q["source_freeze"], admission.ROOT), (3, 0), "exact three prior successes before executing I gate-b")
            resource.exact(current["direct_science_calls"][q["call_id"]], {"state": "admitted", "mode": "generated", "request": token["request"], "reserved_s": 360}, "actual admitted parent gate-b")
            resource.f_request_identity(q, resource.i_generated_plan("gate-b", admission.ROOT), q["source_freeze"])
        if q["call_id"] == resource.J_QUAL_CALLS[3]:
            expected = (360, 340, 20)
            from tools.banc_m3_startup_ray_supervisor import generated_budget_limit
            current = resource.read_pin(resource.text_spec(q["ledger_path"], admission.ROOT, 524288), admission.ROOT, 524288)
            resource.exact(resource.current_caps(current, admission.ROOT, generated_budget_limit), resource.NEW_CAPS, "strict actual J parent ledger")
            recovery, _anchor, _policy, _grant = resource.j_lint_cleanup(current[resource.J_KEY], admission.ROOT, generated_budget_limit)
            resource.exact(resource.j_progress(current, recovery, current[resource.J_KEY], q["source_freeze"], admission.ROOT), (3, 0), "exact three prior successes before executing J gate-b")
            resource.exact(current["direct_science_calls"][q["call_id"]], {"state": "admitted", "mode": "generated", "request": token["request"], "reserved_s": 360}, "actual admitted parent gate-b")
            resource.f_request_identity(q, resource.j_generated_plan("gate-b", admission.ROOT), q["source_freeze"])
        for key, value in zip(("outer_s", "child_s", "cleanup_s"), expected, strict=True):
            resource.exact(q["limits"][key], value, "same genuine parent envelope")
        if q['call_id'] in (checkpoint.BOOTSTRAP_CALL, checkpoint.QUAL_CALLS[3]) or checkpoint is harness and q['call_id'] == harness.QUAL_CALLS[2]:
            checkpoint.source_fence(q['source_freeze'], admission.ROOT)
        else:
            resource.bootstrap_sources(q["source_freeze"], admission.ROOT)
        observed.entry, observed.limits = token["entry_counter_s"], q["limits"]
        observed.window = conserved_native_window(observed.entry, observed.limits, time.perf_counter())
        native_end, cleanup_end, _child_end, _outer_end = observed.window
        resource.need(type(q["limits"]["host_bytes"]) is int and q["limits"]["host_bytes"] > 0, "same actual bounded host resource")
        job_name = "Local\\flylab-startup-ray-bootstrap-" + str(os.getpid()) + "-" + resource.secrets.token_hex(16)
        job = supervisor.OwnedJob(job_name, q["limits"]["host_bytes"])
        resource.need(time.perf_counter() < native_end, "pre-spawn refusal: parent Native interval expired after Job setup")
        resource.d19_spawn(job,argv,root,dict(os.environ),token)
        observed.worker_pid, observed.worker_creation_ticks = job.pid, job.identities[str(job.pid)]
        observed.pre_spawn_refused = False
        observed.all_owned_ended = False
        def drain(index):
            try:
                total = 0
                buffer = supervisor.ct.create_string_buffer(65536)
                while True:
                    count = supervisor.wt.DWORD()
                    ok = job.k.ReadFile(job.reads[index], buffer, len(buffer), supervisor.ct.byref(count), None)
                    if not ok:
                        if supervisor.ct.get_last_error() == 109:
                            eof[index] = True
                            break
                        supervisor.checked(ok)
                    if count.value == 0:
                        eof[index] = True
                        break
                    total += count.value
                    chunks[index].append(buffer.raw[:count.value])
                    resource.need(total <= 512 << 10, "bounded complete Native stream")
            except Exception as error:  # noqa: BLE001 -- retain every owned stream failure for HOLD.
                stream_errors.append({"type": type(error).__name__, "message": str(error)})
        for index in (0, 1):
            reader = threading.Thread(target=drain, args=(index,), daemon=True)
            reader.start()
            readers.append(reader)
        resource.need(time.perf_counter() < native_end, "pre-resume refusal: parent Native interval expired")
        job.resume()
        while job.active() != 0:
            job.observe()
            resource.need(not stream_errors, "complete Native stream failed")
            if time.perf_counter() >= native_end:
                observed.timeout = True
                raise TimeoutError("same parent ENTRY Native deadline; owned descendant termination required")
            time.sleep(.01)
        job.observe()
        if time.perf_counter() >= native_end:
            observed.timeout = True
            raise TimeoutError("Native completion observed after the same parent ENTRY deadline")
        observed.returncode = job.exit_code()
        resource.need(type(observed.returncode) is int, "actual known Native primary exit")
    except Exception as error:  # noqa: BLE001 -- preserve the first owned Native failure.
        observed.first_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        if job:
            try:
                if job.active() != 0:
                    job.observe()
                    job.terminate()
                while job.active() != 0 and time.perf_counter() < cleanup_end:
                    time.sleep(.01)
                job.observe()
                for reader in readers:
                    reader.join(max(0.0, cleanup_end - time.perf_counter()))
                observed.returncode = job.exit_code()
                observed.identities = dict(job.identities)
                observed.all_owned_ended = (job.active() == 0 and type(observed.returncode) is int
                                            and all(eof) and not stream_errors and all(not r.is_alive() for r in readers))
                if not observed.all_owned_ended:
                    observed.unknown_owned_identities = dict(job.identities)
                if all(not reader.is_alive() for reader in readers):
                    job.close()
                else:
                    observed.retained_job = job  # unknown nested ownership; outer Job still owns descendants
            except Exception as error:  # noqa: BLE001 -- cleanup uncertainty retains owned Job and HOLD.
                observed.all_owned_ended = False
                observed.unknown_owned_identities = dict(job.identities)
                observed.secondary_errors.append({"type": type(error).__name__, "message": str(error)})
                observed.retained_job = job
        observed.stdout, observed.stderr = (b"".join(parts) for parts in chunks)
        observed.stdout_eof, observed.stderr_eof = eof
        observed.secondary_errors.extend(stream_errors)
        observation = {key: getattr(observed, key) for key in
                       ("returncode", "first_error", "secondary_errors", "timeout", "pre_spawn_refused", "all_owned_ended",
                        "stdout_eof", "stderr_eof", "worker_pid", "worker_creation_ticks", "identities",
                        "unknown_owned_identities", "entry", "limits", "window")}
        for label in ("stdout", "stderr"):
            raw = getattr(observed, label)
            target = observation_root/("generated-conserved-native-" + label + ".bin")
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            assert target.read_bytes() == raw
            observation[label] = {"path": target.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        observation["observed_counter_s"] = time.perf_counter()
        conserved_put(observation_root, "generated-conserved-native-observation.json", observation)
    return observed


def conserved_invalid_events(event):
    rows = []
    for kind in ("bool-cap", "extra-Count", "extra-Keys", "nonconserved", "old-policy", "J-reset", "old-Original", "human-number"):
        wrong = copy.deepcopy(event)
        if kind == "bool-cap":
            wrong["new_caps"]["generated_global"] = True
        elif kind == "extra-Count":
            wrong["new_caps"]["Count"] = 3
        elif kind == "extra-Keys":
            wrong["new_caps"]["Keys"] = ["generated_global", "generated_local", "original_global"]
        elif kind == "nonconserved":
            wrong["delta_s"]["original_global"] = -599
        elif kind == "old-policy":
            wrong["initial_policy"]["sha256"] = "0"*64
        elif kind == "J-reset":
            wrong["j_grant"]["sha256"] = "0"*64
        elif kind == "old-Original":
            wrong["new_caps"]["original_global"] = 16200
        else:
            wrong["root_decision"]["human_numerical_approval"] = True
        rows.append(wrong)
    return rows


def test_math_conserved_ordered_remaining_and_exact_six_field_diff(tmp_path):
    from tools import banc_m3_conserved_resource as resource
    fixture = conserved_metadata_fixture(tmp_path)
    after = fixture.after
    changed = {key for key in set(after) | set(fixture.before) if after.get(key) != fixture.before.get(key)}
    assert changed == {resource.ALLOCATION_KEY, *resource.CAP_FIELDS.values(), "generated_shared_remaining_s", "original_remaining_s"}
    assert after["debits"] == fixture.before["debits"] and after["direct_science_calls"] == fixture.before["direct_science_calls"]
    assert after["generated_shared_cumulative_limit_s"] + after["original_cumulative_limit_s"] == 17900
    oracle = copy.deepcopy(after)
    conserved_recompute(oracle, fixture.policy, fixture.grant, 2300, 15600)
    assert all(oracle[key] == after[key] for key in resource.TOTAL_KEYS)
    assert after["original_remaining_s"] == 15600.0 - float(fixture.grant["original_cumulative_before_s"]) - sum(
        (float(row["actual_outer_full_return_s"]) for row in after["debits"][165:] if row["mode"] == "original"), 0.0)
    assert fixture.grant["global_limit_s"] == 16200 and fixture.grant["stage_limit_s"] == 6300 and fixture.grant["science_local_limit_s"] == 6000
    f_fixture = conserved_f_metadata_fixture(tmp_path/"generated-F-math")
    conserved_f_review_negatives(f_fixture)
    conserved_f_route_negatives(f_fixture)
    f_after = f_fixture.after
    assert resource.current_caps(f_after, f_fixture.root, __import__("tools.banc_m3_startup_ray_supervisor", fromlist=["generated_budget_limit"]).generated_budget_limit) == resource.NEW_CAPS
    assert {key for key in set(f_after) | set(f_fixture.before) if f_after.get(key) != f_fixture.before.get(key)} == changed
    assert f_after["debits"] == f_fixture.before["debits"] and len(f_after["debits"]) == 283
    assert len(f_after["direct_science_calls"]) == 165 and f_after["direct_science_calls"] == f_fixture.before["direct_science_calls"]
    assert f_after["debits"][-1]["actual_outer_full_return_s"] == 9.208 and f_after["debits"][-1]["actual_returned_exit_code"] == 124
    f_oracle = copy.deepcopy(f_after)
    conserved_recompute(f_oracle, f_fixture.policy, f_fixture.grant, 2300, 15600)
    assert all(f_after[key] == f_oracle[key] for key in resource.TOTAL_KEYS)
    conserved_g_contract_cases(tmp_path/"generated-G-math")
    conserved_h_contract_cases(tmp_path/"generated-H-math")


def test_math_conserved_complete_parameterized_roster_and_disjoint_shards():
    from tools import banc_m3_conserved_resource as resource
    roster = resource.complete_roster(admission.ROOT)
    sets = {role: {row["node_id"] for row in roster[role]} for role in ("math", "gate_a", "gate_b", "gate")}
    assert len(sets["math"]) == 19 and len(sets["gate"]) == 53 and len(sets["gate_a"]) == 21 and len(sets["gate_b"]) == 32
    assert not sets["gate_a"] & sets["gate_b"] and sets["gate_a"] | sets["gate_b"] == sets["gate"]
    for module in roster["indivisible_groups"]:
        family = {row["node_id"] for row in roster["gate"] if row["module"] == module}
        assert family <= sets["gate_a"] or family <= sets["gate_b"]
    assert len([row for row in roster["gate"] if "generated_arm_and_no_journal_callback_order[" in row["name"]]) == 8
    assert len([row for row in roster["gate"] if "payload_exception_survives_nested_diagnostic_unwind[" in row["name"]]) == 4
    for role in ("math", "gate-a", "gate-b"):
        plan = resource.generated_plan(role, admission.ROOT)
        assert "-k" not in plan["argv"]
        assert plan["argv"][2:-3] == [row["node_id"] for row in roster[role.replace("-", "_")]]
        f_plan = resource.f_generated_plan(role, admission.ROOT)
        assert f_plan["argv"][2:-3] == plan["argv"][2:-3] and "-k" not in f_plan["argv"]
        assert f_plan["limits"]["host_bytes"] == 2147483648 and f_plan["limits"]["output_cap_bytes"] == 8388608
    assert resource.f_plan(admission.ROOT)["epoch"] == resource.F_EPOCH and resource.f_plan(admission.ROOT)["roster_epoch"] == resource.EPOCH
    for role in ("lint", "math", "gate-a", "gate-b"):
        old, current = resource.f_generated_plan(role, admission.ROOT), resource.g_generated_plan(role, admission.ROOT)
        assert current["limits"] == old["limits"] and current["role"] == old["role"]
        assert current["argv"] == old["argv"] if role == "lint" else current["argv"][:-1] == old["argv"][:-1]
        assert current["call"] == "direct-science-20261008-contact-g-" + role and "-k" not in current["argv"]
    assert resource.g_plan(admission.ROOT)["generated_close_s"] == 20


def test_gate_conserved_bootstrap_production_transaction_and_native_parity(tmp_path):
    """One fixed prospective bootstrap. Full evidence retained; outer11 fit UNKNOWN."""
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    fixture = conserved_metadata_fixture(tmp_path)
    assert supervisor.generated_budget_limit(fixture.before, tmp_path) == 964
    assert resource.current_caps(fixture.after, tmp_path, supervisor.generated_budget_limit) == resource.NEW_CAPS
    source_negatives = conserved_source_negatives(fixture)
    for spec in source_negatives:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.bootstrap_sources(spec, tmp_path)
    for name in resource.BOOTSTRAP_SOURCES:
        target = tmp_path/name
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        try:
            with pytest.raises(ValueError):
                resource.allocation_before(fixture.event, tmp_path, supervisor.generated_budget_limit)
        finally:
            target.write_bytes(original)
    invalid_events = conserved_invalid_events(fixture.event)
    for event in invalid_events:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.allocation_before(event, tmp_path, supervisor.generated_budget_limit)
    live, qualification = conserved_qualification_fixture(fixture)
    assert resource.validate_qualification(qualification, live, tmp_path, supervisor.generated_budget_limit) == len(fixture.before["debits"]) + 4
    ledger = tmp_path/admission.LEDGER
    conserved_put(tmp_path, admission.LEDGER, fixture.before)
    before_raw = ledger.read_bytes()
    lock = supervisor.SharedLock(tmp_path/admission.SHARED_LOCK, {"call_id": "generated-bootstrap-exclusion"})
    with pytest.raises(FileExistsError):
        supervisor.SharedLock(tmp_path/admission.SHARED_LOCK, {"call_id": "competing-generated"})
    try:
        resource.atomic_ledger_publish(ledger, before_raw, fixture.after, lock,
                                       validate=lambda value: resource.current_caps(value, tmp_path, supervisor.generated_budget_limit))
        assert json.loads(ledger.read_bytes()) == fixture.after
        assert ledger.read_bytes() != before_raw and (tmp_path/admission.SHARED_LOCK).read_bytes() == lock.raw
    finally:
        lock.release(True)
    assert not (tmp_path/admission.SHARED_LOCK).exists()
    lock = supervisor.SharedLock(tmp_path/admission.SHARED_LOCK, {"call_id": "generated-stale"})
    current = ledger.read_bytes()
    with pytest.raises(ValueError, match="raw-before"):
        resource.atomic_ledger_publish(ledger, before_raw, fixture.after, lock, validate=lambda _: None)
    assert ledger.read_bytes() == current
    lock.release(True)
    assert shutil.which("pwsh"), "Native bootstrap requires pwsh; no skip"
    bootstrap_request = {"schema": supervisor.SCHEMA + ".supervision.v1", "call_id": resource.BOOTSTRAP_CALL,
                         "mode": "generated", "role": "execution", "source_freeze": fixture.source,
                         "generated_behavior": "command",
                         "generated_argv": ["-m", "pytest", resource.BOOTSTRAP_NODE, "--tb=short", "-q",
                                            "--junitxml=" + resource.BASE + "/parent-conserved-resource-bootstrap-junit.xml"],
                         "limits": {"outer_s": 11, "child_s": 8, "cleanup_s": 3,
                                    "host_bytes": 2 << 30, "output_cap_bytes": 8 << 20},
                         "ledger_path": admission.LEDGER, "shared_original_lock_path": admission.SHARED_LOCK,
                         "output": resource.BASE + "/parent-" + resource.BOOTSTRAP_CALL}
    admission_negatives = []
    for key, value in (("call_id", resource.BOOTSTRAP_CALL + "-replay"), ("mode", "original"),
                       ("role", "numerical"), ("generated_behavior", "normal"),
                       ("generated_argv", ["-m", "pytest", "--collect-only"]), ("output", "generated-other-output")):
        wrong = copy.deepcopy(bootstrap_request)
        wrong[key] = value
        admission_negatives.append(wrong)
    for key, value in (("outer_s", True), ("child_s", 9), ("cleanup_s", 2), ("host_bytes", 0), ("output_cap_bytes", 1)):
        wrong = copy.deepcopy(bootstrap_request)
        wrong["limits"][key] = value
        admission_negatives.append(wrong)
    f_fixture = conserved_f_metadata_fixture(tmp_path/"generated-F-native")
    f_cases = conserved_f_cases_file(f_fixture)
    g_cases = conserved_g_native_cases_file(tmp_path/"generated-G-native")
    h_cases = conserved_h_native_cases_file(tmp_path/"generated-H-native")
    i_cases = conserved_i_native_cases_file(tmp_path/"generated-I-native")
    j_native_cases_file(tmp_path/"generated-J-native")
    stage_record(tmp_path)
    assert i_cases["bytes"] < 2097152
    assert h_cases["bytes"] < 2097152
    assert g_cases["bytes"] < 2097152
    assert f_cases["bytes"] < 2097152  # same2MiB producer/Native-consumer cap; full evidence retained
    data = conserved_put(tmp_path, "generated-conserved-cases.json",
                         {"before": fixture.before, "after": fixture.after, "events": invalid_events,
                          "live": live, "qualification": qualification, "source_negatives": source_negatives,
                           "critical_sources": list(resource.BOOTSTRAP_SOURCES),
                           "admission_request": bootstrap_request, "admission_negatives": admission_negatives})
    conserved_put(tmp_path, admission.LEDGER, fixture.before)
    script = tmp_path/"generated-conserved-bootstrap.ps1"
    body = r"""
$c=Conserved-ReadPin (Pin 'generated-conserved-cases.json') 2097152
$historical=Conserved-ReadPin (Conserved-Pins).historical 524288
Conserved-BootstrapAdmission $c.admission_request $historical
foreach($q in $c.admission_negatives){$failed=$false;try{Conserved-BootstrapAdmission $q $historical}catch{$failed=$true};Need $failed 'Exact fresh Native admission rejects alternate role/node/bounds'}
$failed=$false;try{Conserved-BootstrapAdmission $c.admission_request $c.before}catch{$failed=$true};Need $failed 'Already paid call cannot be admitted again'
Conserved-Same (Conserved-Caps $c.after) @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600} 'Native/Python cap parity'
foreach($spec in $c.source_negatives){$failed=$false;try{Conserved-BootstrapSource $spec}catch{$failed=$true};Need $failed 'Every missing/typed critical Source negative rejected'}
foreach($name in $c.critical_sources){
  $p=Safe $name;$raw=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllBytes($p,[byte[]]($raw+10))
  $failed=$false
  try{$null=Conserved-Caps $c.after}catch{$failed=$true}finally{[IO.File]::WriteAllBytes($p,$raw)}
  Need $failed 'Allocation cap check rejects changed current critical Source, including embedded account/fixture'
}
foreach($e in $c.events){$refused=$false;try{$null=Conserved-AllocationBefore $e}catch{$refused=$true};Need $refused 'Every typed event negative refused, Count/Keys cannot shadow'}
Need ((Conserved-Qualification $c.qualification $c.live) -eq $c.before.debits.Length+4) 'All four fees and three full reports; no shard alone'
$contactThreeCosts=@($c.qualification.actual_costs|Select-Object -First 3)
$contactSubsetCosts=@($c.qualification.actual_costs[0],$c.qualification.actual_costs[2])
$contactFifthCosts=@($c.qualification.actual_costs)+@($c.qualification.actual_costs[0])
$contactCostNegatives=@(@{label='three';actual_costs=$contactThreeCosts},@{label='subset';actual_costs=$contactSubsetCosts},@{label='fifth';actual_costs=$contactFifthCosts})
foreach($contactFamilyCall in @('contact-learning-20261008-e-body','contact-learning-20261008-e-body-audit','contact-learning-20261008-f-body','contact-learning-20261008-f-body-audit','contact-learning-20261008-g-body','contact-learning-20261008-g-body-audit','contact-learning-20261008-h-body','contact-learning-20261008-h-body-audit')){
  $contactFamilyScience=@{call_id=$contactFamilyCall}
  $null=_Contact-QualificationCosts $contactFamilyScience $c.qualification
  foreach($contactCostCase in $contactCostNegatives){
    $refused=$false;try{$null=_Contact-QualificationCosts $contactFamilyScience $contactCostCase}catch{$refused=$true}
    Need $refused ('Actual Contact four-cost check rejects '+$contactCostCase.label+' for '+$contactFamilyCall)
  }
}
foreach($contactFamilyCall in @('contact-learning-20261007-b-body','contact-learning-20261007-b-body-audit','contact-learning-20261007-c-body','contact-learning-20261007-c-body-audit','contact-learning-20261007-d-body','contact-learning-20261007-d-body-audit')){
  $contactFamilyScience=@{call_id=$contactFamilyCall}
  $null=_Contact-QualificationCosts $contactFamilyScience @{actual_costs=$contactThreeCosts}
  $refused=$false;try{$null=_Contact-QualificationCosts $contactFamilyScience $c.qualification}catch{$refused=$true}
  Need $refused ('Actual Contact historical three-cost check rejects four for '+$contactFamilyCall)
}
$failed=$false;try{$null=Conserved-Json '{"event":1,"event":1}'}catch{$failed=$true};Need $failed 'Duplicate raw Native JSON event rejected'
$script:ledgerTransaction=$null
$owner=Start-LedgerTransaction 'generated-bootstrap'
Closed-Old $c.before
$failed=$false;try{$null=Start-LedgerTransaction 'competing-generated'}catch{$failed=$true};Need $failed 'Common exclusion competition refuses without foreign token reuse'
$target=Safe $ledgerName;$newer=[byte[]]($owner.before+32);[IO.File]::WriteAllBytes($target,$newer)
$failed=$false;try{$null=Publish-LedgerTransaction $c.after $owner}catch{$failed=$true};Need $failed 'Actual production raw-before rejects stale bytes'
Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -ceq [Convert]::ToBase64String($newer)) 'Stale refusal did not overwrite newer state'
Stop-LedgerTransaction $owner
$owner=Start-LedgerTransaction 'generated-bootstrap-atomic'
$pin=Publish-LedgerTransaction $c.after $owner
Conserved-Same (Conserved-ReadPin $pin 524288) $c.after 'Atomic publication complete exact readback'
Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'Exact locally owned release only'

$previousRoot=$taskRoot
$previousTransaction=$script:ledgerTransaction
try{
  $taskRoot=[IO.Path]::GetFullPath((Join-Path $previousRoot 'generated-F-native'))
  $f=Conserved-ReadPin (Pin 'generated-F-cases.json') 2097152
  $beforeF=Conserved-ReadPin $f.before 524288;$afterF=Conserved-ReadPin $f.after 524288;$liveF=Conserved-ReadPin $f.live 524288
  Conserved-Same @(Conserved-ActiveEventPaths) @('verification/banc-natural-gait-m3-20260919/parent-conserved-resource-allocation-20261008-a.json','verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-20261008-f-fixed.json') 'Production route helper used by actual action: two independent literal members'
  Conserved-Same @(Conserved-F-ForbiddenSources) @('verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json','verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-20261008-f-fixed.json','verification/banc-natural-gait-m3-20260919/parent-contact-learning-source-freeze-20261008-f-fixed.json','verification/banc-natural-gait-m3-20260919/parent-contact-learning-qualified-20261008-f-fixed.json') 'Four exact independent forbidden Source paths'
  Conserved-Same @(Conserved-HistoricalWriterSources) @('verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1','tools/banc_m3_conserved_resource_native_core.ps1','tools/banc_m3_conserved_resource_native_epoch.ps1') 'All three old writer Source names, no comma/addition collapse'
  foreach($route in $f.route_negatives){$refused=$false;try{$null=Conserved-RouteBefore $route.descriptor $route.event}catch{$refused=$true};Need $refused 'Original F, aliases and schema/path cross-pairs rejected'}
  $reviewName='verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json'
  $reviewPath=Safe $reviewName;$reviewRaw=[IO.File]::ReadAllBytes($reviewPath)
  foreach($wrong in $f.review_negatives){
    $refused=$false
    try{[IO.File]::WriteAllBytes($reviewPath,[IO.File]::ReadAllBytes((Safe $wrong.path)));$null=Conserved-F-Review (Pin $reviewName)}catch{$refused=$true}finally{[IO.File]::WriteAllBytes($reviewPath,$reviewRaw)}
    Need $refused 'Complete inherited vector and separate current fixup provenance negatives'
  }
  Conserved-Same (Conserved-Caps $afterF) @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600} 'Strict F cap parity'
  foreach($e in $f.events){$refused=$false;try{$null=Conserved-F-Before $e}catch{$refused=$true};Need $refused 'Every strict F event/type/schema/history negative'}
  foreach($case in $f.cases){
    $state=Conserved-ReadPin $case.state 524288;$accepted=$true
    try{$null=Conserved-F-Dispatch $state $case.q $case.request $case.phase}catch{
      if($case.no_pointer -ceq $true){Need ($_.Exception.Message -ceq 'New/mixed caps require exact event') 'Missing-pointer caps refuse before forbidden science request access'}
      $accepted=$false
    }
    Conserved-Same $accepted $case.accept ('Exact Native/Python F progression '+$case.label)
  }
  Need ((Conserved-F-Qualification $f.qualification $liveF) -eq 287) 'Entire19/21/32 and all4Fcosts'
  foreach($q in $f.qualification_negatives){$refused=$false;try{$null=Conserved-F-Qualification $q $liveF}catch{$refused=$true};Need $refused 'No old epoch/subset/type/reordered/Source qualification releases F'}
  $freezeF=Conserved-ReadPin $f.source
  $contactName=$taskBase+'/parent-contact-learning-contract-native.ps1'
  $contactSource=$freezeF.sources[$contactName]
  Conserved-Same $contactSource.path $contactName 'Exact copied current F Native contact contract'
  Conserved-F-Text $contactSource
  $contract=Conserved-ReadPin $f.science.contract;$null=Contact-F-Binding $f.science $contract
  $bodySpec=(Conserved-F-Pins).body;$bodyPath=Safe $bodySpec.path;$bodyRaw=[IO.File]::ReadAllBytes($bodyPath)
  $bodyDrift=Conserved-ReadPin $bodySpec;$bodyDrift.static_status='ORIGINAL_RELEASED'
  $refused=$false
  try{
    [IO.File]::WriteAllBytes($bodyPath,$utf8.GetBytes(($bodyDrift|ConvertTo-Json -Depth 100)+[char]10))
    $null=Contact-F-Binding $f.science $contract
  }catch{$refused=$true}finally{[IO.File]::WriteAllBytes($bodyPath,$bodyRaw)}
  Need $refused 'Exact F static-scope Source drift refuses; no free Original release flag'
  foreach($name in (Conserved-F-Sources)){
    $path=Safe $name;$raw=[IO.File]::ReadAllBytes($path);[IO.File]::WriteAllBytes($path,[byte[]]($raw+10))
    $refused=$false
    try{$null=Conserved-Caps $afterF}catch{$refused=$true}finally{[IO.File]::WriteAllBytes($path,$raw)}
    Need $refused 'All current critical F Source drift refuses before spawn'
  }
  $proof=@{schema='flylab.m3.conserved-resource.bootstrap.v1';cost=(Conserved-F-Pins).failed.cost;source_freeze=(Conserved-F-Pins).failed.source_freeze;fixture_node='tests/test_c_banc_contact_runtime_journal.py::test_gate_conserved_bootstrap_production_transaction_and_native_parity';junit=(Conserved-F-Pins).failed.junit}
  $proofPin=Write-New 'generated-F-v1-failed-proof.json' $proof
  $refused=$false;try{Conserved-Bootstrap $proofPin $beforeF}catch{$refused=$true};Need $refused 'Actual failure124/9.208/emptyJUnit never v1 success'
  $script:ledgerTransaction=$null;$ownerF=Start-LedgerTransaction 'generated-F-allocation'
  Conserved-Same (Conserved-Json $ownerF.before) $beforeF 'Real closed283 temp anchor, never cap reset clone'
  $newerF=[byte[]]($ownerF.before+32);[IO.File]::WriteAllBytes((Safe $ledgerName),$newerF)
  $refused=$false;try{$null=Publish-LedgerTransaction $afterF $ownerF}catch{$refused=$true};Need $refused 'F stale raw refuses no overwrite'
  Stop-LedgerTransaction $ownerF
  $ownerF=Start-LedgerTransaction 'generated-F-atomic'
  $publishedF=Publish-LedgerTransaction $afterF $ownerF;Conserved-Same (Conserved-ReadPin $publishedF 524288) $afterF 'Actual unchanged Native core full F atomic readback'
  Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'F owns exact exclusion release'
}finally{
  try{
    $currentFTransaction=$script:ledgerTransaction
    if($null -ne $currentFTransaction -and -not [Object]::ReferenceEquals($currentFTransaction,$previousTransaction) -and $currentFTransaction.uncertain -ceq $false){
      Stop-LedgerTransaction $currentFTransaction
    }
  }finally{$taskRoot=$previousRoot;$script:ledgerTransaction=$previousTransaction}
}

$previousRootG=$taskRoot;$previousTransactionG=$script:ledgerTransaction
try{
  $taskRoot=[IO.Path]::GetFullPath((Join-Path $previousRootG 'generated-G-native'))
  $g=Conserved-TextJson (Conserved-TextPin 'generated-G-cases.json' 2097152) 2097152
  Need ($g.fixture_only -ceq $true) 'Generated metadata only; never actual G qualification'
  $beforeG=Conserved-TextJson $g.before 524288;$afterG=Conserved-TextJson $g.after 524288
  $closedRaw=Conserved-TextRaw $g.before_raw 524288
  Conserved-Same (Conserved-G-Registration $beforeG $g.amendment $closedRaw) $afterG 'Sole pointer registration Python/actual-Native parity'
  foreach($bad in $g.negatives){$l=Conserved-TextJson $bad.state 524288;$failed=$false;try{$null=Generated-Budget $l}catch{$failed=$true};Need $failed ('Exact standalone G budget rejects '+$bad.label)}
  foreach($case in $g.states){$l=Conserved-TextJson $case.state 524288;$result=Conserved-G-Dispatch $l $case.q $case.request $case.phase;Conserved-Same $result.epoch 'contact-qualification-repair-20261008-g' 'Full ordered G role parity'}
  $qualified=Conserved-TextJson $g.qualified_state 524288
  Need ((Conserved-G-Qualification $g.qualification $qualified) -eq 288) 'Exactly4 G successes and whole19/21/32 JUnits'
  foreach($wrong in $g.qualification_negatives){$failed=$false;try{$null=Conserved-G-Qualification $wrong $qualified}catch{$failed=$true};Need $failed 'Old epochs/subsets/reordering/typed fields never qualify G'}
  foreach($role in @('input_audit','neutral_settling','saved_startup_transfer','tracking_neutral','live_replay','geometry_capture','contact_candidate_body')){
    $original=@{mode='original';source_freeze=$g.source;role=$role;call_id='startup-ray-legacy-blocked';science_request=@{path='must-not-read-Original.npz';bytes=[long]1;sha256=('0'*64)}}
    $failed=$false;try{$null=Conserved-G-Dispatch $afterG $original @{} 'closed'}catch{Need ($_.Exception.Message -ceq 'Global Original block before every legacy-specific input') 'Block before forbidden input access';$failed=$true};Need $failed 'Every Original family stays globally blocked'
  }
  foreach($name in $g.critical_sources){
    $p=Conserved-TextPath $name;$raw=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllBytes($p,[byte[]]($raw+10))
    try{$null=Conserved-G-Caps $afterG;$failed=$false;try{$null=Conserved-G-CurrentReview (Conserved-TextJson $g.amendment).reviewed_source}catch{$failed=$true};Need $failed 'Live G Source drift blocks execution; historical caps still charge failures'}finally{[IO.File]::WriteAllBytes($p,$raw)}
  }
  $contract=Conserved-TextJson $g.science.contract;$null=Contact-G-Binding $g.science $contract
  $ledgerFile=Safe $ledgerName;[IO.File]::WriteAllBytes($ledgerFile,$closedRaw)
  $account=Safe 'generated-G-account.ps1';$amendmentJSON=$g.amendment|ConvertTo-Json -Depth 10 -Compress
  & pwsh -NoProfile -File $account -Action qualification-amendment -QualificationAmendmentDescriptorJSON $amendmentJSON
  Need ($LASTEXITCODE -eq 0) 'Actual temporary Native dedicated one-pointer action'
  $registered=Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile));Conserved-Same $registered $afterG 'Action changes exactly one pointer and no old field'
  $registeredRaw=[IO.File]::ReadAllBytes($ledgerFile)
  & pwsh -NoProfile -File $account -Action qualification-amendment -QualificationAmendmentDescriptorJSON $amendmentJSON
  Need ($LASTEXITCODE -ne 0) 'Duplicate G registration rejects';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($registeredRaw)) 'Duplicate registration preserves exact raw'
  [IO.File]::WriteAllBytes($ledgerFile,[Convert]::FromBase64String($g.admitted_raw_b64))
  $helper=Conserved-TextPath 'tools/banc_m3_conserved_resource.py';$helperRaw=[IO.File]::ReadAllBytes($helper);[IO.File]::WriteAllBytes($helper,[byte[]]($helperRaw+10))
  try{
    & pwsh -NoProfile -File $account -Action commit -TracePath $g.trace_name
    Need ($LASTEXITCODE -eq 0) 'Actual Native once commit of generated initial failure despite Source drift'
    $failed=Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))
    Need ($failed.debits.Length -eq 285 -and (Conserved-Count $failed.direct_science_calls) -eq 167) 'Exactly one additional generated failure cost'
    Conserved-Same $failed.debits[283] $beforeG.debits[283] 'Actual F1.755/exit1 untouched';Conserved-Same $failed.debits[-1].actual_returned_exit_code ([long]1) 'Failed G never success'
    Conserved-Same $failed.debits[-1].actual_outer_full_return_s ([double]0.125) 'Synthetic fixture clock only, no fake live cost'
    $committedRaw=[IO.File]::ReadAllBytes($ledgerFile)
    & pwsh -NoProfile -File $account -Action commit -TracePath $g.trace_name
    Need ($LASTEXITCODE -ne 0) 'Actual Native duplicate cost rejected';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($committedRaw)) 'Duplicate exact no refund/reset/debit mutation'
  }finally{[IO.File]::WriteAllBytes($helper,$helperRaw)}
  $failed=$false;try{$null=Conserved-G-Dispatch (Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))) $g.q $g.request 'closed'}catch{$failed=$true};Need $failed 'Once-paid G failure blocks every next role/retry'
  Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'Actual metadata owner release; no manufactured cleanup'
  $target=Join-Path $taskRoot 'generated-G-link-target';$null=[IO.Directory]::CreateDirectory($target)
  [IO.File]::WriteAllText((Join-Path $target 'metadata.json'),'{}',$utf8)
  $link=Join-Path $taskRoot 'generated-G-link'
  if([OperatingSystem]::IsWindows()){$null=New-Item -ItemType Junction -Path $link -Target $target}else{$null=[IO.Directory]::CreateSymbolicLink($link,$target)}
  $failed=$false;try{$null=Conserved-TextPin 'generated-G-link/metadata.json'}catch{$failed=$true};Need $failed 'Actual reparse ancestor refused, not missing-as-safe'
  foreach($name in @('../escape.json','C:/escape.json','\\?\C:\escape.json','generated:ads.json','generated\bad.json','generated/NUL.json')){$failed=$false;try{$null=Conserved-TextPath $name}catch{$failed=$true};Need $failed 'Lexical namespace/containment/device refusal before filesystem access'}
  $long='generated-long-text/'+(@(0..7|ForEach-Object{'segment'+$_.ToString('00')+('x'*28)}) -join '/')+'/metadata.json'
  $pin=Conserved-TextPin $long;$null=Conserved-TextRaw $pin
  $wrong=@{path=$long;bytes=$pin.bytes;sha256=('0'*64)};$failed=$false;try{$null=Conserved-TextRaw $wrong}catch{$failed=$true};Need $failed 'Actual long-path wrong hash/readback pin refused'
}finally{
  try{if($null -ne $script:ledgerTransaction -and -not [Object]::ReferenceEquals($script:ledgerTransaction,$previousTransactionG) -and $script:ledgerTransaction.uncertain -ceq $false){Stop-LedgerTransaction $script:ledgerTransaction}}finally{$taskRoot=$previousRootG;$script:ledgerTransaction=$previousTransactionG}
}

$previousRootH=$taskRoot;$previousTransactionH=$script:ledgerTransaction
try{
  $taskRoot=[IO.Path]::GetFullPath((Join-Path $previousRootH 'generated-H-native'))
  $h=Conserved-TextJson (Conserved-TextPin 'generated-H-cases.json' 2097152) 2097152
  Need ($h.fixture_only -ceq $true) 'Generated H metadata only, never real qualification'
  $beforeH=Conserved-TextJson $h.before 524288;$afterH=Conserved-TextJson $h.after 524288;$beforeRaw=Conserved-TextRaw $h.before_raw 524288
  Conserved-Same (Conserved-H-Registration $beforeH $h.repair $beforeRaw) $afterH 'Exact one H pointer Python/Native parity'
  Need ($beforeH.debits.Length -eq 285 -and (Conserved-Count $beforeH.direct_science_calls) -eq 167) 'All actual G history retained'
  Conserved-Same $beforeH.debits[284].actual_outer_full_return_s ([double]11.355) 'Actual G failure never refunded'
  Conserved-Same $beforeH.debits[284].actual_returned_exit_code ([long]1) 'Actual failed G never qualified'
  $failed=$false;try{$null=Conserved-H-Registration $beforeH $h.repair ([byte[]]($beforeRaw+32))}catch{$failed=$true};Need $failed 'H exact raw-before rejects'
  foreach($bad in $h.negatives){$l=Conserved-TextJson $bad.state 524288;$failed=$false;try{$null=Generated-Budget $l}catch{$failed=$true};Need $failed ('H strict history/caps reject '+$bad.label)}
  foreach($case in $h.states){$l=Conserved-TextJson $case.state 524288;$result=Conserved-F-Dispatch $l $case.q $case.request 'closed';Conserved-Same $result.epoch 'contact-qualification-lint-repair-20261008-h' 'Global H first ordered role parity'}
  $qualified=Conserved-TextJson $h.qualified_state 524288
  Need ((Conserved-Qualification $h.qualification $qualified) -eq 289) 'H offset285 four successes end289, whole19/21/32'
  foreach($bad in $h.qualification_negatives){$failed=$false;try{$null=Conserved-Qualification $bad $qualified}catch{$failed=$true};Need $failed 'H old epoch/pointer/subset/order/typed-JUnit reject'}
  foreach($key in (Conserved-OwnerKeys)){
    $bad=Conserved-Json ($afterH|ConvertTo-Json -Depth 100);$bad[$key]=@{foreign=@{state='ownership_unknown'}}
    $null=Conserved-H-Caps $bad;$failed=$false;try{$null=Conserved-H-Dispatch $bad $h.q $h.request 'closed'}catch{$failed=$true};Need $failed 'Unknown/foreign/HOLD prevents H dispatch'
  }
  foreach($role in @('input_audit','neutral_settling','saved_startup_transfer','tracking_neutral','live_replay','geometry_capture','contact_direction','contact_candidate_body','contact_body_audit')){
    $original=@{mode='original';source_freeze=$h.source;role=$role;call_id='startup-ray-legacy-blocked';science_request=@{path='must-not-read-Original.npz';bytes=[long]1;sha256=('0'*64)}}
    $failed=$false;try{$null=Conserved-F-Dispatch $afterH $original @{} 'closed'}catch{Need ($_.Exception.Message -ceq 'Global Original block before every legacy-specific input') 'H lexical block before Original request access';$failed=$true};Need $failed 'All legacy Original families blocked before H qualification'
  }
  foreach($name in $h.critical_sources){
    $p=Conserved-TextPath $name;$raw=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllBytes($p,[byte[]]($raw+10))
    try{$null=Conserved-H-Caps $afterH;$failed=$false;try{$null=Conserved-H-CurrentReview (Conserved-TextJson $h.repair).reviewed_source}catch{$failed=$true};Need $failed 'Current H Source drift blocks science, remains chargeable'}finally{[IO.File]::WriteAllBytes($p,$raw)}
  }
  $ledgerFile=Safe $ledgerName;[IO.File]::WriteAllBytes($ledgerFile,$beforeRaw)
  $account=Safe 'generated-H-account.ps1';$repairJSON=$h.repair|ConvertTo-Json -Depth 10 -Compress
  & pwsh -NoProfile -File $account -Action qualification-lint-repair -QualificationLintRepairDescriptorJSON $repairJSON
  Need ($LASTEXITCODE -eq 0) 'Actual temporary dedicated Native H pointer action'
  Conserved-Same (Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))) $afterH 'Exactly one immutable H pointer, every old field unchanged'
  $registeredRaw=[IO.File]::ReadAllBytes($ledgerFile)
  & pwsh -NoProfile -File $account -Action qualification-lint-repair -QualificationLintRepairDescriptorJSON $repairJSON
  Need ($LASTEXITCODE -ne 0) 'H duplicate registration rejects';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($registeredRaw)) 'Duplicate leaves exact H registered raw'
  [IO.File]::WriteAllBytes($ledgerFile,[Convert]::FromBase64String($h.admitted_raw_b64))
  $helper=Conserved-TextPath 'tools/banc_m3_conserved_resource.py';$helperRaw=[IO.File]::ReadAllBytes($helper);[IO.File]::WriteAllBytes($helper,[byte[]]($helperRaw+10))
  try{
    & pwsh -NoProfile -File $account -Action commit -TracePath $h.trace_name
    Need ($LASTEXITCODE -eq 0) 'Actual Native H failed once settlement with current Source drift'
    $settled=Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile));Need ($settled.debits.Length -eq 286 -and (Conserved-Count $settled.direct_science_calls) -eq 168) 'One added H fixture charge only'
    Conserved-Same @($settled.debits|Select-Object -First 285) $beforeH.debits 'Every actual old285 debit intact'
    foreach($entry in $beforeH.direct_science_calls.GetEnumerator()){Conserved-Same $settled.direct_science_calls[$entry.Key] $entry.Value 'Every old167 once call intact'}
    Conserved-Same $settled.debits[-1].actual_returned_exit_code ([long]1) 'Real H failure not success';Conserved-Same $settled.debits[-1].actual_outer_full_return_s ([double]0.125) 'Synthetic fixture clock only'
    foreach($key in (Conserved-OwnerKeys)){Conserved-Same $settled[$key] @{} 'Actual temporary failed once closure'}
    $saved=[IO.File]::ReadAllBytes($ledgerFile)
    & pwsh -NoProfile -File $account -Action commit -TracePath $h.trace_name
    Need ($LASTEXITCODE -ne 0) 'Duplicate H once cost rejects';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($saved)) 'No retry/refund/reset'
  }finally{[IO.File]::WriteAllBytes($helper,$helperRaw)}
  $failed=$false;try{$null=Conserved-F-Dispatch (Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))) $h.q $h.request 'closed'}catch{$failed=$true};Need $failed 'Paid H failure blocks all later roles/replacements/body'
  Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'Actual temporary H writer release, no unknown closure fabricated'
}finally{
  try{if($null -ne $script:ledgerTransaction -and -not [Object]::ReferenceEquals($script:ledgerTransaction,$previousTransactionH) -and $script:ledgerTransaction.uncertain -ceq $false){Stop-LedgerTransaction $script:ledgerTransaction}}finally{$taskRoot=$previousRootH;$script:ledgerTransaction=$previousTransactionH}
}

$previousRootI=$taskRoot;$previousTransactionI=$script:ledgerTransaction
try{
  $taskRoot=[IO.Path]::GetFullPath((Join-Path $previousRootI 'generated-I-native'))
  $i=Conserved-TextJson (Conserved-TextPin 'generated-I-cases.json' 2097152) 2097152
  Need ($i.fixture_only -ceq $true) 'Generated H metadata only, never real qualification'
  $beforeI=Conserved-TextJson $i.before 524288;$afterI=Conserved-TextJson $i.after 524288;$beforeRaw=Conserved-TextRaw $i.before_raw 524288
  Conserved-Same (Conserved-I-Registration $beforeI $i.recovery $beforeRaw) $afterI 'Exact one H pointer Python/Native parity'
  Need ($beforeI.debits.Length -eq 286 -and (Conserved-Count $beforeI.direct_science_calls) -eq 168) 'All actual G history retained'
  $history=Conserved-I-History;$hd=$beforeI.debits[285];$hc=Conserved-TextJson $hd.descriptor
  $failed=$false;try{$null=_Conserved-ActualCostProof $hc $hd 'generated'}catch{$failed=$true};Need $failed 'Current success proof never accepts H timeout'
  $failed=$false;try{$null=_Conserved-ActualCostProof $hc $hd 'historical-bootstrap'}catch{$failed=$true};Need $failed 'H cannot be historical-bootstrap alias'
  foreach($case in $i.states){
    foreach($bounds in @(@(361,340,20),@(360,339,20),@(360,340,19))){$bad=Conserved-Json ($case.q|ConvertTo-Json -Depth 100);$bad.limits.outer_s=[long]$bounds[0];$bad.limits.child_s=[long]$bounds[1];$bad.limits.cleanup_s=[long]$bounds[2];$failed=$false;try{$null=Conserved-F-Dispatch (Conserved-TextJson $case.state 524288) $bad $case.request 'closed'}catch{$failed=$true};Need $failed 'Only literal I gate-b36034020'}
  }
  Conserved-Same $beforeI.debits[285].actual_outer_full_return_s ([double]13.253) 'Actual G failure never refunded'
  Conserved-Same $beforeI.debits[285].actual_returned_exit_code ([long]124) 'Actual failed G never qualified'
  $failed=$false;try{$null=Conserved-I-Registration $beforeI $i.recovery ([byte[]]($beforeRaw+32))}catch{$failed=$true};Need $failed 'H exact raw-before rejects'
  foreach($bad in $i.negatives){$l=Conserved-TextJson $bad.state 524288;$failed=$false;try{$null=Generated-Budget $l}catch{$failed=$true};Need $failed ('H strict history/caps reject '+$bad.label)}
  foreach($case in $i.states){$l=Conserved-TextJson $case.state 524288;$result=Conserved-F-Dispatch $l $case.q $case.request 'closed';Conserved-Same $result.epoch 'contact-preflight-overhead-recovery-20261008-i' 'Global H first ordered role parity'}
  $qualified=Conserved-TextJson $i.qualified_state 524288
  Need ((Conserved-Qualification $i.qualification $qualified) -eq 290) 'H offset285 four successes end289, whole19/21/32'
  foreach($bad in $i.qualification_negatives){$failed=$false;try{$null=Conserved-Qualification $bad $qualified}catch{$failed=$true};Need $failed 'H old epoch/pointer/subset/order/typed-JUnit reject'}
  foreach($key in (Conserved-OwnerKeys)){
    $bad=Conserved-Json ($afterI|ConvertTo-Json -Depth 100);$bad[$key]=@{foreign=@{state='ownership_unknown'}}
    $null=Conserved-I-Caps $bad;$failed=$false;try{$null=Conserved-I-Dispatch $bad $i.q $i.request 'closed'}catch{$failed=$true};Need $failed 'Unknown/foreign/HOLD prevents H dispatch'
  }
  foreach($role in @('input_audit','neutral_settling','saved_startup_transfer','tracking_neutral','live_replay','geometry_capture','contact_direction','contact_candidate_body','contact_body_audit')){
    $original=@{mode='original';source_freeze=$i.source;role=$role;call_id='startup-ray-legacy-blocked';science_request=@{path='must-not-read-Original.npz';bytes=[long]1;sha256=('0'*64)}}
    $failed=$false;try{$null=Conserved-F-Dispatch $afterI $original @{} 'closed'}catch{Need ($_.Exception.Message -ceq 'Global Original block before every legacy-specific input') 'H lexical block before Original request access';$failed=$true};Need $failed 'All legacy Original families blocked before H qualification'
  }
  foreach($name in $i.critical_sources){
    $p=Conserved-TextPath $name;$raw=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllBytes($p,[byte[]]($raw+10))
    try{$null=Conserved-I-Caps $afterI;$failed=$false;try{$null=Conserved-I-CurrentReview (Conserved-TextJson $i.recovery).reviewed_source}catch{$failed=$true};Need $failed 'Current H Source drift blocks science, remains chargeable'}finally{[IO.File]::WriteAllBytes($p,$raw)}
  }
  foreach($call in @('contact-learning-20261008-i-body','contact-learning-20261008-i-body-audit')){
    $failed=$false;try{_Contact-QualificationCosts @{call_id=$call} @{actual_costs=@(1,2,3)}}catch{$failed=$true};Need $failed 'Production NativeContact blocks three costs for both I calls'
    _Contact-QualificationCosts @{call_id=$call} @{actual_costs=@(1,2,3,4)}
  }
  _Contact-QualificationCosts @{call_id='contact-learning-20261007-c-body'} @{actual_costs=@(1,2,3)}
  $ledgerFile=Safe $ledgerName;[IO.File]::WriteAllBytes($ledgerFile,$beforeRaw)
  $account=Safe 'generated-I-account.ps1';$repairJSON=$i.recovery|ConvertTo-Json -Depth 10 -Compress
  & pwsh -NoProfile -File $account -Action preflight-recovery -PreflightRecoveryDescriptorJSON $repairJSON
  Need ($LASTEXITCODE -eq 0) 'Actual temporary dedicated Native H pointer action'
  Conserved-Same (Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))) $afterI 'Exactly one immutable H pointer, every old field unchanged'
  $registeredRaw=[IO.File]::ReadAllBytes($ledgerFile)
  & pwsh -NoProfile -File $account -Action preflight-recovery -PreflightRecoveryDescriptorJSON $repairJSON
  Need ($LASTEXITCODE -ne 0) 'H duplicate registration rejects';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($registeredRaw)) 'Duplicate leaves exact H registered raw'
  [IO.File]::WriteAllBytes($ledgerFile,[Convert]::FromBase64String($i.admitted_raw_b64))
  $helper=Conserved-TextPath 'tools/banc_m3_conserved_resource.py';$helperRaw=[IO.File]::ReadAllBytes($helper);[IO.File]::WriteAllBytes($helper,[byte[]]($helperRaw+10))
  try{
    & pwsh -NoProfile -File $account -Action commit -TracePath $i.trace_name
    Need ($LASTEXITCODE -eq 0) 'Actual Native H failed once settlement with current Source drift'
    $settled=Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile));Need ($settled.debits.Length -eq 287 -and (Conserved-Count $settled.direct_science_calls) -eq 169) 'One added H fixture charge only'
    Conserved-Same @($settled.debits|Select-Object -First 286) $beforeI.debits 'Every actual old285 debit intact'
    foreach($entry in $beforeI.direct_science_calls.GetEnumerator()){Conserved-Same $settled.direct_science_calls[$entry.Key] $entry.Value 'Every old167 once call intact'}
    Conserved-Same $settled.debits[-1].actual_returned_exit_code ([long]1) 'Real H failure not success';Conserved-Same $settled.debits[-1].actual_outer_full_return_s ([double]0.125) 'Synthetic fixture clock only'
    foreach($key in (Conserved-OwnerKeys)){Conserved-Same $settled[$key] @{} 'Actual temporary failed once closure'}
    $saved=[IO.File]::ReadAllBytes($ledgerFile)
    & pwsh -NoProfile -File $account -Action commit -TracePath $i.trace_name
    Need ($LASTEXITCODE -ne 0) 'Duplicate H once cost rejects';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($ledgerFile)) -ceq [Convert]::ToBase64String($saved)) 'No retry/refund/reset'
  }finally{[IO.File]::WriteAllBytes($helper,$helperRaw)}
  $failed=$false;try{$null=Conserved-F-Dispatch (Conserved-Json ([IO.File]::ReadAllBytes($ledgerFile))) $i.q $i.request 'closed'}catch{$failed=$true};Need $failed 'Paid H failure blocks all later roles/replacements/body'
  Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'Actual temporary H writer release, no unknown closure fabricated'
}finally{
  try{if($null -ne $script:ledgerTransaction -and -not [Object]::ReferenceEquals($script:ledgerTransaction,$previousTransactionI) -and $script:ledgerTransaction.uncertain -ceq $false){Stop-LedgerTransaction $script:ledgerTransaction}}finally{$taskRoot=$previousRootI;$script:ledgerTransaction=$previousTransactionI}
}


$prevJ=$taskRoot;$txJ0=$script:ledgerTransaction
try{
$taskRoot=[IO.Path]::GetFullPath((Join-Path $prevJ 'generated-J-native'))
$rf=Conserved-TextPath (_Conserved-RecordPin).path;$rb=Conserved-TextRaw (_Conserved-RecordPin) 262144;$rs=Conserved-Json $rb
foreach($f in @('g','h','i','j')){
$r=$rs.families[$f];$x=_Conserved-Epoch $f;$ex=Conserved-Json ($r.epoch|ConvertTo-Json -Depth 20);$ex.paths=$r.paths
Conserved-Same $x $ex 'Epoch values';Need ($x.offset -is [long] -and $x.end -is [long] -and $x.envelope -is [long]) 'Long epoch'
Conserved-Same (& $r.epoch.pins) $r.pins 'Pins';Conserved-Same (& ('Conserved-'+$f.ToUpperInvariant()+'-Paths')) $r.paths 'Paths'
}
Conserved-Same (Conserved-H-GPaths) $rs.families.h.predecessor 'G paths';Conserved-Same (Conserved-I-HPaths) $rs.families.i.predecessor 'H paths';Conserved-Same (Conserved-J-IPaths) $rs.families.j.predecessor 'I paths'
foreach($f in @('G','x','')){$failed=$false;try{$null=_Conserved-Epoch $f}catch{$failed=$true};Need $failed 'Literal family'}
foreach($kind in @('schema','family','record','pin','path','epoch','calls','callable','bool','string','float','pinbool','pinstring','pinfloat')){
$bad=Conserved-Json $rb
switch($kind){
schema{$bad.schema='foreign'}
family{$g=$bad.families.g;$bad.families.Remove('g');$bad.families['G']=$g}
record{$bad.families.g['Count']=1}
pin{$bad.families.g.pins.before['Count']=1}
path{$bad.families.g.paths.body=$true}
epoch{$bad.families.j.epoch['Count']=1}
calls{$bad.families.j.epoch.calls=@('foreign','foreign')}
callable{$bad.families.j.epoch.caps='Invoke-Expression'}
bool{$bad.families.j.epoch.offset=$true}
string{$bad.families.j.epoch.end='291'}
float{$bad.families.j.epoch.envelope=[double]575}
pinbool{$bad.families.g.pins.before.bytes=$true}
pinstring{$bad.families.g.pins.before.bytes='12295'}
pinfloat{$bad.families.g.pins.before.bytes=[double]12295}
}
$failed=$false;try{_Conserved-RecordShape $bad}catch{$failed=$true};Need $failed 'Production shape'
}
foreach($kind in @('missing','changed','alias')){
if($kind -ceq 'alias'){[IO.File]::Move($rf,$rf+'.alias')}elseif($kind -ceq 'missing'){[IO.File]::Delete($rf)}else{$bad=[byte[]]$rb.Clone();$bad[0]=32;[IO.File]::WriteAllBytes($rf,$bad)}
try{$failed=$false;try{$null=Conserved-J-Pins}catch{$failed=$true};Need $failed 'Record auth'}finally{if($kind -ceq 'alias'){[IO.File]::Move($rf+'.alias',$rf)}else{[IO.File]::WriteAllBytes($rf,$rb)}}
}
$j=Conserved-TextJson (Conserved-TextPin 'generated-J-cases.json' 2097152) 2097152;Need ($j.fixture_only -ceq $true) 'J fixture'
$raw=Conserved-TextRaw $j.before_raw 524288;$bj=Conserved-Json $raw;$aj=Conserved-TextJson $j.after 524288
Conserved-Same (Conserved-J-Registration $bj $j.cleanup $raw) $aj 'Sole J'
Need ($bj.debits.Length -eq 287 -and (Conserved-Count $bj.direct_science_calls) -eq 169) 'I history'
$history=Conserved-J-History;Conserved-Same $history.anchor $bj 'I failure'
foreach($call in @('contact-learning-20261008-j-body','contact-learning-20261008-j-body-audit')){
$failed=$false;try{_Contact-QualificationCosts @{call_id=$call} @{actual_costs=@(1,2,3)}}catch{$failed=$true};Need $failed 'Four costs'
_Contact-QualificationCosts @{call_id=$call} @{actual_costs=@(1,2,3,4)}
}
_Contact-QualificationCosts @{call_id='contact-learning-20261007-c-body'} @{actual_costs=@(1,2,3)}
$lf=Safe $ledgerName;[IO.File]::WriteAllBytes($lf,$raw);$ac=Safe 'generated-J-account.ps1';$ds=$j.cleanup|ConvertTo-Json -Depth 10 -Compress
& pwsh -NoProfile -File $ac -Action qualification-lint-cleanup -QualificationLintCleanupDescriptorJSON $ds
Need ($LASTEXITCODE -eq 0) 'J registration';Conserved-Same (Conserved-Json ([IO.File]::ReadAllBytes($lf))) $aj 'Sole pointer'
[IO.File]::WriteAllBytes($lf,[Convert]::FromBase64String($j.admitted_raw_b64))
$sf=Conserved-TextPath 'tools/banc_m3_conserved_resource.py';$srJ=[IO.File]::ReadAllBytes($sf);[IO.File]::WriteAllBytes($sf,[byte[]]($srJ+10))
try{
$null=Conserved-J-Caps $aj;$failed=$false;try{$null=Conserved-J-CurrentReview (Conserved-TextJson $j.cleanup).reviewed_source}catch{$failed=$true};Need $failed 'Drift'
$pd=[IO.File]::ReadAllBytes($lf);$bad=[byte[]]$rb.Clone();$bad[0]=32;[IO.File]::WriteAllBytes($rf,$bad)
try{
& pwsh -NoProfile -File $ac -Action commit -TracePath $j.trace_name
Need ($LASTEXITCODE -ne 0) 'Evidence pending'
Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($lf)) -ceq [Convert]::ToBase64String($pd)) 'Pending exact'
}finally{[IO.File]::WriteAllBytes($rf,$rb)}
& pwsh -NoProfile -File $ac -Action commit -TracePath $j.trace_name
Need ($LASTEXITCODE -eq 0) 'Once fee';$st=Conserved-Json ([IO.File]::ReadAllBytes($lf));Need ($st.debits.Length -eq 288) 'Sole fixture fee'
Conserved-Same @($st.debits|Select-Object -First 287) $bj.debits 'Old fees';foreach($key in (Conserved-OwnerKeys)){Conserved-Same $st[$key] @{} 'Closure'}
$saved=[IO.File]::ReadAllBytes($lf)
& pwsh -NoProfile -File $ac -Action commit -TracePath $j.trace_name
Need ($LASTEXITCODE -ne 0) 'Duplicate fee';Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes($lf)) -ceq [Convert]::ToBase64String($saved)) 'Once ledger'
}finally{[IO.File]::WriteAllBytes($sf,$srJ)}
$taskRoot=[IO.Path]::GetFullPath((Join-Path $prevJ 'generated-J-eof'));$ac=Safe 'generated-J-account.ps1'
& pwsh -NoProfile -File $ac -Action commit -TracePath $j.trace_name
$held=Conserved-Json ([IO.File]::ReadAllBytes((Safe $ledgerName)));Need ($held.debits.Length -eq 287 -and $held.direct_science_blocking_holds.Count -eq 1) 'EOF HOLD'
}finally{
try{if($null -ne $script:ledgerTransaction -and -not [Object]::ReferenceEquals($script:ledgerTransaction,$txJ0) -and $script:ledgerTransaction.uncertain -ceq $false){Stop-LedgerTransaction $script:ledgerTransaction}}finally{$taskRoot=$prevJ;$script:ledgerTransaction=$txJ0}
}

$owner=Start-LedgerTransaction 'generated-uncertain'
$owner.uncertain=$true
$failed=$false;try{Stop-LedgerTransaction $owner}catch{$failed=$true};Need $failed 'Uncertain persistence retains hold'
Need ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe ($taskBase+'/parent-e358-original-active.lock')))) -ceq [Convert]::ToBase64String($owner.raw)) 'Unknown retains exact own token; no success'
'CONSERVATION_TRANSACTION_FIXTURE_RETURN'
"""
    assert data["bytes"] < 2 << 20
    script.write_text(conserved_native_definitions(tmp_path) + "\n" + body, encoding="utf8")
    observed = conserved_run_native(tmp_path, [shutil.which("pwsh"), "-NoProfile", "-File", str(script)])
    assert observed.first_error is None and not observed.secondary_errors and not observed.timeout
    assert observed.all_owned_ended and observed.stdout_eof and observed.stderr_eof and not observed.unknown_owned_identities
    assert observed.returncode == 0 and b"CONSERVATION_TRANSACTION_FIXTURE_RETURN" in observed.stdout, observed.stderr
    assert (tmp_path/admission.SHARED_LOCK).exists()  # retained generated hold, never a live lock
    assert fixture.after["debits"] == fixture.before["debits"] and fixture.after["direct_science_calls"] == fixture.before["direct_science_calls"]
    with pytest.raises(ValueError):
        resource.text_spec("generated-G-link/metadata.json", tmp_path/"generated-G-native")
    long_name = "/".join(["generated-long-text"] + ["segment" + str(i).zfill(2) + "x" * 28 for i in range(8)]) + "/metadata.json"
    assert resource.raw_pin(resource.text_spec(long_name, tmp_path/"generated-G-native"), tmp_path/"generated-G-native")


def test_gate_conserved_local_capability_stale_bytes_and_uncertain_hold(tmp_path, monkeypatch):
    import time

    from tools import banc_m3_conserved_resource as resource
    limits = {"outer_s": 11, "child_s": 8, "cleanup_s": 3}
    assert conserved_native_window(100.0, limits, 101.0) == (107.0, 107.9, 108.0, 111.0)
    with pytest.raises(ValueError, match="remaining Native"):
        conserved_native_window(100.0, limits, 107.0)
    with pytest.raises(ValueError, match="typed actual"):
        conserved_native_window(100.0, dict(limits, child_s=True), 101.0)
    with monkeypatch.context() as missing_owner:
        missing_owner.delenv("FLYLAB_DIRECT_SCIENCE_OWNER", raising=False)
        refused = conserved_run_native(tmp_path, ["must-not-be-spawned"])
    assert refused.pre_spawn_refused and refused.worker_pid is None and refused.returncode is None
    assert refused.first_error["type"] == "ValueError" and not refused.unknown_owned_identities
    assert resource.BOOTSTRAP_DISPATCHABLE is True  # an attempt is distinct from a successful result
    fixture = conserved_metadata_fixture(tmp_path/"generated-rejected-bootstrap")
    proof = resource.read_pin(fixture.bootstrap, fixture.root)
    failed_cost = resource.read_pin(proof["cost"], fixture.root)
    failed_cost["exit_code"] = 1
    failed_outer = resource.read_pin(failed_cost["outer"], fixture.root)
    failed_outer["exit_code"] = 1
    failed_cost["outer"] = conserved_put(fixture.root, "generated-rejected-outer.json", failed_outer)
    failed_trace = resource.read_pin(failed_cost["trace"], fixture.root)
    failed_trace["exit_code"] = 1
    failed_cost["trace"] = conserved_put(fixture.root, "generated-rejected-trace.json", failed_trace)
    failed_spec = conserved_put(fixture.root, "generated-rejected-cost.json", failed_cost)
    failed_before = copy.deepcopy(fixture.before)
    failed_before["debits"][-1].update(descriptor=failed_spec, actual_returned_exit_code=1)
    proof["cost"] = failed_spec
    rejected_proof = conserved_put(fixture.root, "generated-rejected-proof.json", proof)
    with pytest.raises(ValueError):
        resource.bootstrap_proof(rejected_proof, failed_before, fixture.root)
    from tools import banc_m3_startup_ray_supervisor as supervisor
    refusal_root = tmp_path/"generated-initial-refusal"
    refusal_request = {"schema": supervisor.SCHEMA + ".supervision.v1", "call_id": resource.BOOTSTRAP_CALL,
                       "mode": "generated", "role": "execution", "generated_behavior": "command",
                       "ledger_path": admission.LEDGER, "shared_original_lock_path": admission.SHARED_LOCK,
                       "output": resource.BASE + "/parent-" + resource.BOOTSTRAP_CALL,
                       "limits": {"outer_s": 11, "child_s": 8, "cleanup_s": 3, "output_cap_bytes": 8 << 20}}
    refusal_spec = conserved_put(refusal_root, "generated-initial-request.json", refusal_request)
    refusal_before = copy.deepcopy(fixture.historical)
    refusal_before["direct_science_calls"][resource.BOOTSTRAP_CALL] = {
        "request": refusal_spec, "state": "admitted", "mode": "generated", "reserved_s": 11}
    refusal_before["direct_science_reservations"] = {
        resource.BOOTSTRAP_CALL: {"request": refusal_spec, "reserved_s": 11}}
    conserved_put(refusal_root, admission.LEDGER, refusal_before)
    before_bytes = (refusal_root/admission.LEDGER).read_bytes()
    refusal = supervisor.bootstrap_pre_spawn_refusal(refusal_root, refusal_spec, refusal_request,
                                                    ValueError("actual generated initial failure"), entry=time.perf_counter())
    assert refusal["pre_spawn_refused"] and refusal["exit_code"] == 1 and not refusal["job_created"]
    assert refusal["worker_pid"] is None and refusal["all_owned_ended"] and refusal["stdout_eof"] and refusal["stderr_eof"]
    assert refusal["first_error"]["message"] == "actual generated initial failure"
    assert (refusal_root/admission.LEDGER).read_bytes() == before_bytes
    with pytest.raises(ValueError, match="before actual output"):
        supervisor.bootstrap_pre_spawn_refusal(refusal_root, refusal_spec, refusal_request,
                                              ValueError("must not overwrite"), entry=time.perf_counter())
    target = tmp_path/"generated-ledger.json"
    target.write_bytes(b'{"value":0}')
    lock = resource.SharedLock(tmp_path/"generated-common.lock", {"owner": "generated"})
    clone = object.__new__(resource.SharedLock)
    clone.__dict__.update(lock.__dict__)
    with pytest.raises(ValueError, match="local owner"):
        clone.authenticate()
    old = target.read_bytes()
    with monkeypatch.context() as changed:
        def uncertain_replace(_temporary, _target):
            raise OSError("generated atomic replacement uncertain")
        changed.setattr(resource.os, "replace", uncertain_replace)
        with pytest.raises(OSError, match="uncertain"):
            resource.atomic_ledger_publish(target, old, {"value": 1}, lock, validate=lambda _: None)
    assert lock.uncertain and target.read_bytes() == old and lock.target.read_bytes() == lock.raw
    with pytest.raises(ValueError, match="retains hold"):
        lock.release(True)
    from tools import banc_m3_startup_ray_supervisor as supervisor
    wrong = resource.SharedLock(tmp_path/"foreign-local.lock", {"request": pin(), "call_id": "generated"})
    with pytest.raises(ValueError, match="common physical"):
        supervisor.transition(tmp_path, "must-not-be-read.json", {}, {}, {}, {}, "begin", lock=wrong)
    wrong.release(True)
    conserved_g_registration_cases(tmp_path/"generated-G-registration")
    conserved_g_text_cases(tmp_path/"generated-G-long-text")
    conserved_h_registration_cases(tmp_path/"generated-H-registration")
    conserved_i_registration_cases(tmp_path/"generated-I-registration")


def test_gate_conserved_epoch_types_events_history_and_active_admission(tmp_path):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    fixture = conserved_metadata_fixture(tmp_path)
    active = copy.deepcopy(fixture.after)
    active["direct_science_workers"] = {"generated-next": {"state": "begun"}}
    assert resource.current_caps(active, tmp_path, supervisor.generated_budget_limit) == resource.NEW_CAPS
    with pytest.raises(ValueError):
        resource.known_closed(active)  # empty owners belongs to allocation, not later admission
    for kind in ("missing-event", "mixed-cap", "float-cap", "bool-usage", "nonfinite", "changed-history", "duplicate-debit", "restored-J-cap", "extra-event"):
        wrong = copy.deepcopy(fixture.after)
        if kind == "missing-event":
            del wrong[resource.ALLOCATION_KEY]
        elif kind == "mixed-cap":
            wrong["generated_shared_cumulative_limit_s"] = 1700
        elif kind == "float-cap":
            wrong["generated_shared_cumulative_limit_s"] = 2300.0
        elif kind == "bool-usage":
            wrong["direct_science_generated_used_s"] = True
        elif kind == "nonfinite":
            wrong["generated_shared_remaining_s"] = float("inf")
        elif kind == "changed-history":
            wrong["debits"][281]["actual_returned_exit_code"] = 0
        elif kind == "duplicate-debit":
            wrong["debits"].append(copy.deepcopy(wrong["debits"][-1]))
        elif kind == "restored-J-cap":
            wrong["original_cumulative_limit_s"] = 16200
        else:
            wrong[resource.ALLOCATION_KEY + "_duplicate"] = wrong[resource.ALLOCATION_KEY]
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.current_caps(wrong, tmp_path, supervisor.generated_budget_limit)
    from tools import verify_c_banc_m3_startup_ray as ray
    with pytest.raises(ValueError, match="duplicate"):
        ray.json_load(b'{"event":1,"event":1}')
    source = (admission.ROOT/admission.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    assert "Legacy grant/rebind cannot restore Original16200" in source
    assert fixture.historical["debits"][276]["actual_returned_exit_code"] == 1
    assert fixture.historical["debits"][279]["actual_returned_exit_code"] == 124
    assert [row["actual_returned_exit_code"] for row in fixture.historical["debits"][280:282]] == [1, 1]
    f_fixture = conserved_f_metadata_fixture(tmp_path/"generated-F-events")
    proof = conserved_put(f_fixture.root, "generated-actual-failed-v1-proof.json", {
        "schema": "flylab.m3.conserved-resource.bootstrap.v1", "cost": resource.F_FAILED_PINS["cost"],
        "source_freeze": resource.F_FAILED_PINS["source_freeze"], "fixture_node": resource.BOOTSTRAP_NODE,
        "junit": resource.F_FAILED_PINS["junit"]})
    with pytest.raises(ValueError):
        resource.bootstrap_proof(proof, f_fixture.before, f_fixture.root)
    for name in resource.F_REQUIRED_SOURCES:
        target = f_fixture.root/name
        raw = target.read_bytes()
        target.write_bytes(raw+b"\n")
        try:
            with pytest.raises(ValueError):
                resource.current_caps(f_fixture.after, f_fixture.root, supervisor.generated_budget_limit)
        finally:
            target.write_bytes(raw)
    for kind in ("mixed", "duplicate", "history", "boolean-usage", "nonfinite", "restored-J", "foreign-event"):
        wrong = copy.deepcopy(f_fixture.after)
        if kind == "mixed":
            wrong["generated_shared_cumulative_limit_s"] = 1700
        elif kind == "duplicate":
            wrong["debits"].append(copy.deepcopy(wrong["debits"][-1]))
        elif kind == "history":
            wrong["debits"][282]["actual_returned_exit_code"] = 0
        elif kind == "boolean-usage":
            wrong["direct_science_generated_used_s"] = True
        elif kind == "nonfinite":
            wrong["original_remaining_s"] = float("nan")
        elif kind == "restored-J":
            wrong["original_cumulative_limit_s"] = 16200
        else:
            wrong[resource.ALLOCATION_KEY]["path"] = resource.ALLOCATION_PATH
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.current_caps(wrong, f_fixture.root, supervisor.generated_budget_limit)
    conserved_g_source_and_scope_cases(tmp_path/"generated-G-current-Source")
    conserved_h_source_and_scope_cases(tmp_path/"generated-H-current-Source")
    conserved_i_source_and_scope_cases(tmp_path/"generated-I-current-Source")
    conserved_i_contract_cases(tmp_path/"generated-I-contract")
    j_contract_cases(tmp_path/"generated-J-contract")


def test_gate_conserved_qualification_four_costs_three_junits_and_fee(tmp_path):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    fixture = conserved_metadata_fixture(tmp_path)
    live, qualified = conserved_qualification_fixture(fixture)
    end = resource.validate_qualification(qualified, live, tmp_path, supervisor.generated_budget_limit)
    assert end == len(fixture.before["debits"]) + 4
    for kind in ("one-shard", "three-costs", "duplicate-cost", "reordered", "changed-source", "missing-node", "wrong-class"):
        wrong = copy.deepcopy(qualified)
        if kind == "one-shard":
            wrong["tests"] = wrong["tests"][:2]
        elif kind == "three-costs":
            wrong["actual_costs"].pop()
        elif kind == "duplicate-cost":
            wrong["actual_costs"][3] = wrong["actual_costs"][2]
        elif kind == "reordered":
            wrong["actual_costs"][1], wrong["actual_costs"][2] = wrong["actual_costs"][2], wrong["actual_costs"][1]
        elif kind == "changed-source":
            wrong["source_freeze"]["sha256"] = "0"*64
        elif kind == "missing-node":
            wrong["tests"][1]["node_ids"].pop()
        else:
            wrong["tests"][2]["classnames"][0] = "generated.substitution"
        with pytest.raises((ValueError, KeyError, IndexError, FileNotFoundError)):
            resource.validate_qualification(wrong, live, tmp_path, supervisor.generated_budget_limit)
    import xml.etree.ElementTree as ET
    spec = qualified["tests"][2]["file"]
    raw = (tmp_path/spec["path"]).read_bytes()
    for kind in ("skip", "failure", "error", "duplicate"):
        document = ET.fromstring(raw)
        suite = document.find("testsuite")
        if kind == "duplicate":
            suite.append(copy.deepcopy(suite[0]))
        else:
            ET.SubElement(suite[0], {"skip": "skipped", "failure": "failure", "error": "error"}[kind])
        bad = ET.tostring(document, encoding="utf8")
        target = tmp_path/spec["path"]
        target.write_bytes(bad)
        wrong = copy.deepcopy(qualified)
        wrong["tests"][2]["file"].update(bytes=len(bad), sha256=hashlib.sha256(bad).hexdigest())
        with pytest.raises(ValueError):
            resource.validate_qualification(wrong, live, tmp_path, supervisor.generated_budget_limit)
    (tmp_path/spec["path"]).write_bytes(raw)
    cost_spec = qualified["actual_costs"][0]
    cost = resource.read_pin(cost_spec, tmp_path)
    trace = resource.read_pin(cost["trace"], tmp_path)
    trace["first_final_ms"] += 1
    cost["trace"] = conserved_put(tmp_path, "generated-wrong-first-FINAL.json", trace)
    wrong_cost = conserved_put(tmp_path, "generated-wrong-fee.json", cost)
    wrong = copy.deepcopy(qualified)
    wrong_live = copy.deepcopy(live)
    wrong["actual_costs"][0] = wrong_cost
    wrong_live["debits"][len(fixture.before["debits"])]["descriptor"] = wrong_cost
    with pytest.raises(ValueError):
        resource.validate_qualification(wrong, wrong_live, tmp_path, supervisor.generated_budget_limit)
    f_fixture = conserved_f_metadata_fixture(tmp_path/"generated-F-qualification")
    f_live, f_qualified = conserved_f_qualification_fixture(f_fixture)
    assert resource.validate_f_qualification(f_qualified, f_live, f_fixture.root, supervisor.generated_budget_limit) == 287
    for wrong in conserved_f_qualification_negatives(f_qualified):
        with pytest.raises((ValueError, KeyError, IndexError, FileNotFoundError)):
            resource.validate_f_qualification(wrong, f_live, f_fixture.root, supervisor.generated_budget_limit)
    conserved_g_order_and_qualification_cases(tmp_path/"generated-G-full-qualification")
    conserved_h_order_and_qualification_cases(tmp_path/"generated-H-full-qualification")
    conserved_i_order_and_qualification_cases(tmp_path/"generated-I-full-qualification")
    j_qualification_cases(tmp_path/"generated-J-full-qualification")


def test_gate_conserved_e_routing_candidate_and_journal_boundaries(tmp_path, monkeypatch, capsys):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    from tools import verify_c_banc_m3_contact_learning as worker
    from tools import verify_c_banc_m3_startup_ray as ray
    fixture = conserved_metadata_fixture(tmp_path)
    descriptor = resource.read_pin(resource.E_DESCRIPTOR_PIN, tmp_path)
    stage = json.loads((admission.ROOT/admission.STAGE_SPEC).read_bytes())
    assert descriptor["candidate"] == admission.RETRY_PINS["candidate"]
    assert stage["limits"]["contact_candidate_body"]["internal_s"] == 270
    assert tuple(stage["limits"]["contact_body_audit"][key] for key in ("outer_s", "child_s", "internal_s", "cleanup_s")) == (120, 100, 90, 20)
    assert resource.QUAL_CAPS == ((15, 10, 5), (25, 20, 5), (120, 110, 10), (120, 110, 10))
    import time
    origin = time.perf_counter()
    owner = {"call_id": resource.E_CALLS[0], "request": pin("generated-owner-request.json"), "entry_counter_s": origin,
             "owner_pid": os.getpid(), "owner_creation_ticks": 19, "job_name": "generated-only"}
    science_spec = pin("generated-E-science.json")
    outer = {"call_id": owner["call_id"], "role": "contact_candidate_body", "source_freeze": fixture.source, "science_request": science_spec}
    monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps(owner))
    monkeypatch.setattr(supervisor, "creation_ticks", lambda _pid: 19)
    monkeypatch.setattr(supervisor, "in_named_job", lambda _name: True)
    monkeypatch.setattr(ray, "leaf", lambda spec, *_args, **_kwargs: json.dumps(outer).encode("utf8"))
    sentinel = RuntimeError("generated-E-admission-before-any-array")
    def refused_admission(*_args, **kwargs):
        assert kwargs["runtime_journal"] is not None and kwargs["entry"] == origin
        raise sentinel
    monkeypatch.setattr(admission, "Admission", refused_admission)
    with pytest.raises(RuntimeError) as observed:
        worker.main(["--request", science_spec["path"]])
    assert observed.value is sentinel
    rows = [json.loads(line[len(runtime.PREFIX):]) for line in capsys.readouterr().err.splitlines() if line.startswith(runtime.PREFIX)]
    assert rows and all(row["identity"]["call_id"] == resource.E_CALLS[0] for row in rows)
    assert rows[-1]["first_error"]["message"] == str(sentinel) and rows[-1]["run_valid"] is False
    assert rows[-1]["work"] is None and runtime.MAX_ACTIVE_PHASES == 14 and runtime.MAX_EVENTS == 64
    assert runtime.MAX_EMITS == 16 and runtime.MAX_JSON_BYTES == 32768 and runtime.MAX_STREAM_BYTES == 524288
    assert runtime.SCIENCE_S == 270 and runtime.WORKER_S == 285 and runtime.FLUSH_ALLOWANCE_S == 1
    owner["call_id"] = resource.F_CALLS[0]
    science_spec = pin("generated-F-science.json")
    f_source = dict(fixture.source, path=resource.F_SOURCE_PATH)
    outer.update(call_id=owner["call_id"], source_freeze=f_source, science_request=science_spec)
    monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps(owner))
    with pytest.raises(RuntimeError) as observed_f:
        worker.main(["--request", science_spec["path"]])
    assert observed_f.value is sentinel
    f_rows = [json.loads(line[len(runtime.PREFIX):]) for line in capsys.readouterr().err.splitlines() if line.startswith(runtime.PREFIX)]
    assert f_rows and all(row["identity"]["call_id"] == resource.F_CALLS[0] for row in f_rows)
    assert f_rows[-1]["first_error"]["message"] == str(sentinel) and f_rows[-1]["run_valid"] is False
    assert f_rows[-1]["work"] is None and runtime.MAX_ACTIVE_PHASES == 14 and runtime.MAX_EVENTS == 64


def conserved_f_metadata_fixture(root):
    """Generated historical F fixture from actual archived Source; no live F replay."""
    from tools import banc_m3_conserved_resource as resource

    conserved_runtime_recovery_copy(root)
    pins = (resource.HISTORICAL_PIN, resource.POLICY_PIN, resource.J_PIN, resource.HUMAN_PIN,
            resource.ROOT_PIN, resource.ROSTER_PIN, resource.E_DESCRIPTOR_PIN, resource.F_CLOSED_PIN,
            resource.F_ROOT_PIN, resource.F_ADDENDUM_PIN, resource.F_ADMIN_PIN, resource.F_BEFORE_SOURCE_PIN,
            *resource.F_ASTRA_PINS.values(), *resource.F_FAILED_PINS.values(), resource.F_ORIGINAL_DESCRIPTOR_PIN,
            resource.F_ORIGINAL_REVIEW_PIN, resource.F_ORIGINAL_EVENT_PIN, resource.F_REFUSAL_PIN,
            resource.F_ADOPTION_PIN, resource.F_FIXUP_DESIGN_PIN, resource.F_FIXUP_BEFORE_PIN,
            *resource.F_FIXUP_ASTRA_PINS.values())
    for spec in pins:
        ledger_pin = spec["path"] in (resource.HISTORICAL_PIN["path"], resource.F_CLOSED_PIN["path"])
        raw = resource.raw_pin(spec, admission.ROOT, cap=524288 if ledger_pin else 1048576, allow_empty=True)
        resource.fixture_text_write(root, spec["path"], raw)
    before_source = resource.read_pin(resource.F_BEFORE_SOURCE_PIN, root)
    for item in before_source["source_snapshots"]:
        conserved_copy(root, {"path": item["archive"], "bytes": item["bytes"], "sha256": item["sha256"]})
    fixup_before = resource.read_pin(resource.F_FIXUP_BEFORE_PIN, root)
    for item in fixup_before["snapshots"]:
        conserved_copy(root, item["archive"])
    actual_review, historical_map = resource.historical_f_reviewed_sources(admission.ROOT)
    sources = copy.deepcopy(actual_review["sources"])
    assert all(name in sources for name in (*resource.F_REQUIRED_SOURCES, admission.STAGE_SPEC))
    for name, spec in sources.items():
        assert Path(name).suffix in (".py", ".ps1", ".js", ".json", ".txt", ".ini", ".md", ".xml")
        if not Path(name).is_absolute():
            conserved_historical_copy(root, spec, historical_map)
    for pin in (*actual_review["root_reviews"], actual_review["fixup_provenance"]["root_fixup"]):
        conserved_copy(root, pin)
    reviewed = conserved_put(root, resource.F_REVIEW_PATH, copy.deepcopy(actual_review))
    before = resource.read_pin(resource.F_CLOSED_PIN, root, 524288)
    policy, grant = resource.read_pin(resource.POLICY_PIN, root), resource.read_pin(resource.J_PIN, root)
    event = resource.read_pin(resource.F_ORIGINAL_EVENT_PIN, root)
    event.update(schema=resource.F_SCHEMA, event_id="conserved-prequalification-20261008-f-fixed-600",
                 reviewed_source=reviewed, superseded_administrative_refusal=resource.F_REFUSAL_PIN)
    event_spec = conserved_put(root, resource.F_EVENT_PATH, event)
    source = conserved_put(root, resource.F_SOURCE_PATH, {"schema": "flylab.m3.contact-learning.source-freeze.v1",
        "ROOT_review_cycle_complete": True, "qualification_epoch": resource.F_EPOCH, "reviewed_source": reviewed,
        "prequalification_event": event_spec, "sources": sources, "body_retry": resource.F_DESCRIPTOR_PIN})
    after = resource.allocated_snapshot(before, event_spec, policy, grant)
    conserved_put(root, admission.LEDGER, before)
    return SimpleNamespace(root=root, before=before, after=after, event=event, event_spec=event_spec,
                           source=source, policy=policy, grant=grant, reviewed=reviewed)


def conserved_f_request(fixture, role):
    from tools import banc_m3_conserved_resource as resource
    plan = resource.f_generated_plan(role, fixture.root)
    return {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"],
            "role": plan["role"], "source_freeze": fixture.source, "shared_original_lock_path": admission.SHARED_LOCK,
            "ledger_path": admission.LEDGER, "limits": plan["limits"], "generated_behavior": "command",
            "generated_argv": plan["argv"], "output": plan["output"]}


def conserved_f_charge(fixture, live, role):
    from tools import banc_m3_conserved_resource as resource
    plan = resource.f_generated_plan(role, fixture.root)
    cost = conserved_fake_paid(fixture.root, live, fixture.source, plan["call"], plan["role"], plan["argv"],
                               resource.QUAL_CAPS[("lint", "math", "gate-a", "gate-b").index(role)],
                               name="generated-F-" + role, request=conserved_f_request(fixture, role))
    conserved_recompute(live, fixture.policy, fixture.grant, 2300, 15600)
    return cost


def conserved_f_qualification_fixture(fixture):
    from tools import banc_m3_conserved_resource as resource
    live, costs = copy.deepcopy(fixture.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        costs.append(conserved_f_charge(fixture, live, role))
    roster, suites = resource.complete_roster(fixture.root), []
    for role in ("math", "gate-a", "gate-b"):
        rows = roster[role.replace("-", "_")]
        xml = conserved_junit(fixture.root, resource.BASE + "/parent-contact-learning-f-" + role + "-junit.xml", rows)
        suites.append({"file": xml, "node_ids": [r["node_id"] for r in rows], "names": [r["name"] for r in rows],
                       "classnames": [r["classname"] for r in rows], "count": len(rows)})
    qualified = {"schema": "flylab.m3.direct-science.qualification.v3", "epoch": resource.F_EPOCH, "old_E360": "UNPASSED",
                 "allocation": fixture.event_spec, "reviewed_source": fixture.reviewed, "qualification_roster": resource.ROSTER_PIN,
                 "source_freeze": fixture.source, "actual_costs": costs, "tests": suites, "evidence": costs}
    spec = conserved_put(fixture.root, resource.F_QUAL_PATH, qualified)
    live["direct_science_qualification"] = spec
    return live, qualified


def conserved_f_dispatch_cases(fixture):
    """Both exact phases, every Original family, generic/alias/order/fifth/HOLD."""
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    states, cases = [], []
    live = copy.deepcopy(fixture.after)
    def add(state, q, phase, expected, label):
        index = len(cases)
        spec = conserved_put(fixture.root, "generated-F-dispatch-request-" + str(index) + ".json", q)
        state_spec = conserved_put(fixture.root, "generated-F-dispatch-state-" + str(index) + ".json", state)
        try:
            result = resource.prequalification_dispatch(state, q, spec, fixture.root, supervisor.generated_budget_limit, phase=phase)
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            assert not expected, label
        else:
            assert expected and result["epoch"] == resource.F_EPOCH, label
        cases.append({"state": state_spec, "q": q, "request": spec, "phase": phase, "accept": expected, "label": label})
        return spec
    for index, role in enumerate(("lint", "math", "gate-a", "gate-b")):
        states.append(copy.deepcopy(live))
        q = conserved_f_request(fixture, role)
        spec = add(live, q, "closed", True, "next exact F " + role)
        active = copy.deepcopy(live)
        active["direct_science_calls"][q["call_id"]] = {"state": "admitted", "mode": "generated", "request": spec, "reserved_s": q["limits"]["outer_s"]}
        active["direct_science_reservations"] = {q["call_id"]: {"request": spec, "reserved_s": q["limits"]["outer_s"]}}
        cases.append({"state": conserved_put(fixture.root, "generated-F-own-admitted-" + role + ".json", active),
                      "q": q, "request": spec, "phase": "admitted", "accept": True, "label": "exact own admitted " + role})
        assert resource.prequalification_dispatch(active, q, spec, fixture.root, supervisor.generated_budget_limit, phase="admitted")
        if index == 0:
            for key, value in (("call_id", resource.F_QUAL_CALLS[1]), ("call_id", q["call_id"]+"-retry"),
                               ("generated_argv", ["-m", "pytest", "--collect-only"]), ("generated_behavior", "normal"),
                               ("output", "generated-alternate"), ("role", "execution")):
                wrong = copy.deepcopy(q)
                wrong[key] = value
                add(live, wrong, "closed", False, key + " alias/generic/order")
            for key, value in (("host_bytes", 1), ("output_cap_bytes", 1), ("outer_s", True)):
                wrong = copy.deepcopy(q)
                wrong["limits"][key] = value
                add(live, wrong, "closed", False, "bounds " + key)
            from tools import verify_c_banc_m3_startup_ray as ray
            forbidden_path = "must-not-read-Original-no-pointer.json"
            forbidden_request = {"path": forbidden_path, "bytes": 1, "sha256": "0"*64}
            for kind in ("all-new", "mixed-shared", "mixed-local", "mixed-original"):
                wrong = copy.deepcopy(live)
                del wrong[resource.ALLOCATION_KEY]
                if kind != "all-new":
                    for cap_key, field in resource.CAP_FIELDS.items():
                        wrong[field] = resource.OLD_CAPS[cap_key]
                    field = {"mixed-shared": "generated_shared_cumulative_limit_s",
                             "mixed-local": "direct_science_generated_limit_s",
                             "mixed-original": "original_cumulative_limit_s"}[kind]
                    wrong[field] = {"mixed-shared": 2300, "mixed-local": 1564, "mixed-original": 15600}[kind]
                original = dict(q, mode="original", role="input_audit", call_id="startup-ray-generated-no-pointer",
                                science_request=forbidden_request, grant=resource.J_PIN)
                request_spec = add(wrong, original, "closed", False, "no-pointer " + kind + " before all Original access")
                cases[-1]["no_pointer"] = True
                visited = []
                actual_leaf = ray.leaf
                def forbidden_read(spec, *_args, visited=visited, **_kwargs):
                    visited.append(spec["path"])
                    raise AssertionError("request/Source read before missing-pointer cap refusal")
                def guarded_leaf(spec, *args, visited=visited, forbidden_path=forbidden_path, actual_leaf=actual_leaf, **kwargs):
                    if spec["path"] == forbidden_path:
                        visited.append(spec["path"])
                        raise AssertionError("forbidden Original science request access")
                    return actual_leaf(spec, *args, **kwargs)
                def forbidden_admission(*_args, visited=visited, **_kwargs):
                    visited.append("Original-specific Admission")
                    raise AssertionError("Original Admission preceded global cap refusal")
                conserved_put(fixture.root, admission.LEDGER, wrong)
                try:
                    with pytest.MonkeyPatch.context() as guarded:
                        guarded.setattr(resource, "read_pin", forbidden_read)
                        guarded.setattr(ray, "leaf", guarded_leaf)
                        guarded.setattr(ray, "Admission", forbidden_admission)
                        guarded.setattr(supervisor, "OwnedJob", forbidden_admission)
                        with pytest.raises(ValueError, match="new/mixed caps"):
                            resource.prequalification_dispatch(wrong, original, request_spec, fixture.root,
                                                               supervisor.generated_budget_limit, phase="closed")
                        with pytest.raises(ValueError, match="new/mixed caps"):
                            supervisor._supervise(request_spec["path"], root=fixture.root)
                    assert visited == [] and not (fixture.root/forbidden_path).exists()
                    assert not (fixture.root/original["output"]).exists()
                finally:
                    conserved_put(fixture.root, admission.LEDGER, fixture.before)
            for phase in ("closed", "admitted"):
                legacy = copy.deepcopy(fixture.before)
                legacy_request = conserved_put(fixture.root, "generated-F-legacy-"+phase+".json", q)
                if phase == "admitted":
                    legacy["direct_science_calls"][q["call_id"]] = {"state": "admitted", "mode": "generated", "request": legacy_request, "reserved_s": 15}
                    legacy["direct_science_reservations"] = {q["call_id"]: {"request": legacy_request, "reserved_s": 15}}
                assert resource.prequalification_dispatch(legacy, q, legacy_request, fixture.root, supervisor.generated_budget_limit, phase=phase) is None
                cases.append({"state": conserved_put(fixture.root, "generated-F-legacy-state-"+phase+".json", legacy),
                              "q": q, "request": legacy_request, "phase": phase, "accept": True,
                              "label": "legitimate old-cap "+phase+" remains outside F restriction; not a dispatch grant"})
            for original_role in ("input_audit", "ray", "candidate_audit", "saved_startup_transfer",
                                  "tracking_neutral", "tracking_candidate", "tracking_diagnosis", "live_replay",
                                  "geometry_capture", "geometry_analyze", *admission.ROLES):
                wrong = dict(q, mode="original", role=original_role, call_id="startup-ray-generated-bypass",
                             science_request={"path": "must-not-read-Original.json", "bytes": 1, "sha256": "0"*64})
                add(live, wrong, "closed", False, "all Original blocked " + original_role)
            for key in resource.OWNER_KEYS:
                wrong = copy.deepcopy(live)
                wrong[key] = {"foreign": {"state": "ownership_unknown"}}
                add(wrong, q, "closed", False, "owner/HOLD " + key)
        conserved_f_charge(fixture, live, role)
    failed = copy.deepcopy(states[1])
    failed_cost = resource.read_pin(failed["debits"][-1]["descriptor"], fixture.root)
    failed_outer = resource.read_pin(failed_cost["outer"], fixture.root)
    failed_trace = resource.read_pin(failed_cost["trace"], fixture.root)
    failed_outer["exit_code"] = failed_trace["exit_code"] = failed_cost["exit_code"] = 1
    failed_cost["outer"] = conserved_put(fixture.root, "generated-F-paid-failure-outer.json", failed_outer)
    failed_cost["trace"] = conserved_put(fixture.root, "generated-F-paid-failure-trace.json", failed_trace)
    failed["debits"][-1].update(descriptor=conserved_put(fixture.root, "generated-F-paid-failure-cost.json", failed_cost), actual_returned_exit_code=1)
    add(failed, conserved_f_request(fixture, "math"), "closed", False, "spent full failure blocks progression, no fee credit")
    qualified_live, qualified = conserved_f_qualification_fixture(fixture)
    assert qualified_live["debits"] == live["debits"]
    add(qualified_live, conserved_f_request(fixture, "gate-b"), "closed", False, "fifth/retry after all four")
    stage = resource.read_pin(resource.read_pin(resource.F_DESCRIPTOR_PIN, fixture.root)["contract"], fixture.root)
    science = {"schema": admission.SCHEMA + ".request.v1", "mode": "original", "role": admission.ROLES[2],
               "call_id": resource.F_CALLS[0], "source_freeze": fixture.source, "body_retry": resource.F_DESCRIPTOR_PIN,
               "contract": admission.spec_for(admission.STAGE_SPEC, fixture.root), "grant": resource.J_PIN,
               "inputs": admission.INPUTS, "roster": admission.ROSTER, "limits": stage["limits"][admission.ROLES[2]],
               "local_debit_offset": 165, "resource_increment_s": 0,
               "output": resource.BASE+"/parent-"+resource.F_CALLS[0]+"/result"}
    def read_binding(spec):
        return resource.read_pin(spec, fixture.root)
    bound = admission.runtime_f_binding(science, stage, read_binding)
    assert bound == read_binding(resource.F_DESCRIPTOR_PIN) and bound["static_status"] == "PREQUALIFICATION_ONLY"
    assert bound["candidate"] == admission.RETRY_PINS["candidate"] and bound["historical_closed283"] == resource.F_CLOSED_PIN
    body_target = fixture.root/resource.F_DESCRIPTOR_PIN["path"]
    body_raw = body_target.read_bytes()
    drift = copy.deepcopy(bound)
    drift["static_status"] = "ORIGINAL_RELEASED"
    try:
        body_target.write_bytes(json.dumps(drift, allow_nan=False).encode("utf8"))
        with pytest.raises(ValueError):
            admission.runtime_f_binding(science, stage, read_binding)
    finally:
        body_target.write_bytes(body_raw)
    def wrong_static_read(spec):
        value = read_binding(spec)
        return dict(value, static_status=True) if spec == resource.F_DESCRIPTOR_PIN else value
    with pytest.raises(ValueError):
        admission.runtime_f_binding(science, stage, wrong_static_read)
    science_spec = conserved_put(fixture.root, "generated-F-scope-only-science.json", science)
    q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "original", "call_id": resource.F_CALLS[0],
         "role": admission.ROLES[2], "source_freeze": fixture.source, "shared_original_lock_path": admission.SHARED_LOCK,
         "ledger_path": admission.LEDGER, "limits": {k: science["limits"][k] for k in
         ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}, "output": science["output"]+"-supervision",
         "science_request": science_spec, "grant": resource.J_PIN}
    body_spec = add(qualified_live, q, "closed", True, "only F candidate body after full qualification")
    active_body = copy.deepcopy(qualified_live)
    active_body["direct_science_calls"][q["call_id"]] = {"state": "admitted", "mode": "original", "request": body_spec, "reserved_s": 300}
    assert resource.prequalification_dispatch(active_body, q, body_spec, fixture.root, supervisor.generated_budget_limit, phase="admitted")
    cases.append({"state": conserved_put(fixture.root, "generated-F-own-admitted-body.json", active_body), "q": q,
                  "request": body_spec, "phase": "admitted", "accept": True, "label": "exact own Original body; no premature Original reservation"})
    wrong = copy.deepcopy(qualified_live)
    wrong["direct_science_qualification"] = fixture.before["direct_science_qualification"]
    add(wrong, q, "closed", False, "legacy registered qualification never releases F")
    for role in ("tracking_neutral", "saved_startup_transfer", "contact_direction", "contact_candidate_audit", "geometry_capture", "live_replay"):
        add(qualified_live, dict(q, role=role), "closed", False, "no generic Original release even after full qualification")
    add(qualified_live, dict(q, call_id=resource.F_CALLS[1], role=admission.ROLES[3]), "closed", False, "audit requires actual sole body predecessor")
    return cases, states, qualified_live, qualified, science


def conserved_f_review_negatives(fixture):
    from tools import banc_m3_conserved_resource as resource
    review = resource.read_pin(fixture.reviewed, fixture.root)
    original = resource.read_pin(resource.F_ORIGINAL_REVIEW_PIN, fixture.root)
    assert len(original["sources"]) == 722 and len(review["sources"]) == 723
    assert set(review["sources"]) == set(original["sources"]) | {resource.F_PATH}
    assert review["sources"][resource.F_ORIGINAL_DESCRIPTOR_PIN["path"]] == resource.F_ORIGINAL_DESCRIPTOR_PIN
    unrelated = next(name for name in original["sources"] if name not in resource.F_REQUIRED_SOURCES
                     and name not in resource.F_FIXUP_SOURCES and name != resource.F_ORIGINAL_DESCRIPTOR_PIN["path"])
    target = fixture.root/resource.F_REVIEW_PATH
    actual = target.read_bytes()
    specs = []
    for kind in ("missing-inherited", "unrelated-pin", "old-static", "extra-Source", "old-schema", "before13", "old-Astra", "root-path"):
        wrong = copy.deepcopy(review)
        if kind == "missing-inherited":
            del wrong["sources"][unrelated]
        elif kind == "unrelated-pin":
            wrong["sources"][unrelated]["bytes"] += 1
        elif kind == "old-static":
            del wrong["sources"][resource.F_ORIGINAL_DESCRIPTOR_PIN["path"]]
        elif kind == "extra-Source":
            wrong["sources"]["generated-undeclared-extra.py"] = {"path": "generated-undeclared-extra.py", "bytes": 1, "sha256": "0"*64}
        elif kind == "old-schema":
            wrong["schema"] = "flylab.m3.conserved-resource-prequalification.reviewed-source.v1"
        elif kind == "before13":
            wrong["fixup_provenance"]["before_source"] = resource.F_BEFORE_SOURCE_PIN
        elif kind == "old-Astra":
            wrong["fixup_provenance"]["astra_review"] = resource.F_ASTRA_PINS
        else:
            wrong["fixup_provenance"]["root_fixup"]["path"] = "docs/generated-not-actual-root-fixup.md"
        spec = conserved_put(fixture.root, "generated-F-review-negative-" + kind + ".json", wrong)
        specs.append(spec)
        try:
            target.write_bytes((fixture.root/spec["path"]).read_bytes())
            with pytest.raises(ValueError):
                resource.f_reviewed_sources(admission.spec_for(resource.F_REVIEW_PATH, fixture.root), fixture.root)
        finally:
            target.write_bytes(actual)
    assert resource.f_reviewed_sources(fixture.reviewed, fixture.root) == review
    return specs


def conserved_f_route_negatives(fixture):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    original = resource.read_pin(resource.F_ORIGINAL_EVENT_PIN, fixture.root)
    assert original["schema"] == "flylab.m3.conserved-resource-prequalification.v1"
    cases = [
        {"descriptor": resource.F_ORIGINAL_EVENT_PIN, "event": original},
        {"descriptor": dict(fixture.event_spec, path=resource.ALLOCATION_PATH), "event": fixture.event},
        {"descriptor": fixture.event_spec, "event": original},
        {"descriptor": dict(fixture.event_spec, path=resource.F_EVENT_PATH + ".alias"), "event": fixture.event},
    ]
    for case in cases:
        with pytest.raises(ValueError):
            resource.allocation_route_before(case["descriptor"], case["event"], fixture.root, supervisor.generated_budget_limit)
    wrong = copy.deepcopy(fixture.after)
    wrong[resource.ALLOCATION_KEY] = resource.F_ORIGINAL_EVENT_PIN
    with pytest.raises(ValueError, match="only strict v1/F event paths"):
        resource.current_caps(wrong, fixture.root, supervisor.generated_budget_limit)
    assert resource.f_administrative_refusal(fixture.root)["scientific_cost_created"] is False
    assert fixture.event["superseded_administrative_refusal"] == resource.F_REFUSAL_PIN
    return cases


def conserved_f_cases_file(fixture):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    cases, _states, live, qualified, science = conserved_f_dispatch_cases(fixture)
    event_negatives = conserved_invalid_events(fixture.event)
    for key, value in (("scope", "ORIGINAL"), ("schema", "flylab.m3.conserved-resource-allocation.v1"),
                       ("event_id", "conserved-resource-20261008-600"), ("before_ledger", resource.HISTORICAL_PIN),
                       ("superseded_administrative_refusal", resource.F_FAILED_PINS["cost"]),
                       ("schema", "flylab.m3.conserved-resource-prequalification.v1")):
        event_negatives.append(dict(fixture.event, **{key: value}))
    for event in event_negatives:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.prequalification_before(event, fixture.root, supervisor.generated_budget_limit)
    data = {"before": resource.F_CLOSED_PIN,
            "after": conserved_put(fixture.root, "generated-F-after.json", fixture.after),
            "live": conserved_put(fixture.root, "generated-F-full-qualified.json", live), "qualification": qualified,
            "events": event_negatives, "cases": cases, "source": fixture.source, "science": science,
            "review_negatives": conserved_f_review_negatives(fixture), "route_negatives": conserved_f_route_negatives(fixture),
            "qualification_negatives": conserved_f_qualification_negatives(qualified)}
    return conserved_put(fixture.root, "generated-F-cases.json", data)


def conserved_f_qualification_negatives(qualified):
    from tools import banc_m3_conserved_resource as resource
    cases = []
    for kind in ("v1", "v2", "E-epoch", "one-shard", "three-costs", "duplicate-cost", "reordered", "wrong-class", "missing-node", "boolean-count", "old-Source"):
        wrong = copy.deepcopy(qualified)
        if kind == "v1":
            wrong["schema"] = "flylab.m3.direct-science.qualification.v1"
        elif kind == "v2":
            wrong["schema"] = "flylab.m3.direct-science.qualification.v2"
        elif kind == "E-epoch":
            wrong["epoch"] = resource.EPOCH
        elif kind == "one-shard":
            wrong["tests"].pop()
        elif kind == "three-costs":
            wrong["actual_costs"].pop()
        elif kind == "duplicate-cost":
            wrong["actual_costs"][3] = wrong["actual_costs"][2]
        elif kind == "reordered":
            wrong["actual_costs"][1], wrong["actual_costs"][2] = wrong["actual_costs"][2], wrong["actual_costs"][1]
        elif kind == "wrong-class":
            wrong["tests"][2]["classnames"][0] = "generated.substitution"
        elif kind == "missing-node":
            wrong["tests"][1]["node_ids"].pop()
        elif kind == "boolean-count":
            wrong["tests"][0]["count"] = True
        else:
            wrong["source_freeze"] = resource.F_FAILED_PINS["source_freeze"]
        cases.append(wrong)
    return cases


def conserved_runtime_recovery_copy(root):
    """Private bounded generated text copy; the preparation prefix stays exact."""
    from tools import banc_m3_conserved_resource as resource
    root.mkdir(parents=True, exist_ok=True)
    stage_record(root)
    allowed = ("before_ledger", "initial_policy", "first_transfer", "additional_transfer", "final_transfer",
               "continuation_transfer", "direct_human_auto_authority", "direct_science_policy",
               "direct_science_generated_transfer", "direct_science_generated_additional_transfer",
               "direct_science_generated_final_transfer", "direct_science_generated_continuation_transfer")
    visited = set()
    def copy_pin(spec):
        if spec["path"] in visited:
            return
        visited.add(spec["path"])
        assert Path(spec["path"]).suffix == ".json" and 0 < spec["bytes"] <= 524288
        raw = resource.raw_pin(spec, admission.ROOT, 524288)
        resource.fixture_text_write(root, spec["path"], raw)
        value = json.loads(raw)
        for key in allowed:
            item = value.get(key)
            if type(item) is dict and set(item) == {"path", "bytes", "sha256"}:
                copy_pin(item)
    for spec in (admission.D_HISTORY_PINS["closed276"], admission.D_HISTORY_PINS["runtime_transfer"]):
        copy_pin(spec)
    for spec in admission.RUNTIME_RECOVERY_PINS.values():
        assert Path(spec["path"]).suffix == ".json" and 0 < spec["bytes"] <= 524288
        resource.fixture_text_write(root, spec["path"], resource.raw_pin(spec, admission.ROOT, 524288))
    closed = resource.read_pin(admission.RUNTIME_RECOVERY_PINS["closed277"], root, 524288)
    return dict(closed, direct_science_generated_runtime_residual_transfer=admission.RUNTIME_RECOVERY_PINS["residual_transfer"],
                direct_science_generated_limit_s=964)


def conserved_g_qualification_negatives(qualified):
    from tools import banc_m3_conserved_resource as resource
    cases = conserved_f_qualification_negatives(qualified)
    cases.append(dict(copy.deepcopy(qualified), schema="flylab.m3.direct-science.qualification.v3"))
    cases.append(dict(copy.deepcopy(qualified), epoch=resource.F_EPOCH))
    return cases


def conserved_g_metadata_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    stage_record(root)
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    root.mkdir(parents=True, exist_ok=True)
    review_spec = resource.text_spec(resource.G_REVIEW_PATH, admission.ROOT, 1048576)
    historical, mapped = resource.historical_f_reviewed_sources(admission.ROOT)
    review = resource.g_review_metadata(review_spec, admission.ROOT, historical, mapped)
    before_h = resource.h_archives(admission.ROOT)
    for name in resource.G_REPLACEMENTS:
        item = before_h[name]
        assert review["sources"][name] == item["source"]
        resource.fixture_text_write(root, name, resource.raw_pin(item["archive"], admission.ROOT, 262144))
    copied = set()
    def walk(value):
        if type(value) is dict:
            if set(value) == {"path", "bytes", "sha256"}:
                name = value["path"]
                if Path(name).is_absolute() or name == admission.LEDGER or name in resource.G_REPLACEMENTS:
                    return
                suffix = Path(name).suffix
                if suffix not in (".json", ".md", ".txt", ".xml", ".ini") and "/before/" not in name:
                    return
                key = (name, value["sha256"])
                if key in copied:
                    return
                copied.add(key)
                raw = resource.raw_pin(value, admission.ROOT, 2097152, allow_empty=True)
                resource.fixture_text_write(root, name, raw)
                if suffix == ".json":
                    walk(json.loads(raw))
            else:
                archive = value.get("archive")
                if type(archive) is str and {"path", "bytes", "sha256"} <= set(value):
                    walk({"path": archive, "bytes": value["bytes"], "sha256": value["sha256"]})
                for key, child in value.items():
                    if key != "sources":  # old Source vectors are history; no live superseded reads
                        walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
    for spec in review["sources"].values():
        if not Path(spec["path"]).is_absolute() and spec["path"] not in resource.G_REPLACEMENTS:
            conserved_copy(root, spec)
    for spec in (resource.G_BEFORE_PIN, resource.G_ROOT_PIN, *resource.G_ASTRA_PINS.values(),
                 *review["root_reviews"], *review["sol_reports"], review_spec):
        walk(spec)
    historical, mapped = resource.historical_f_reviewed_sources(admission.ROOT)
    for item in mapped.values():
        walk(item["archive"])
        if item["source"]["path"] not in resource.G_REPLACEMENTS:
            walk(item["source"])
    walk(historical)
    actual_amendment = resource.read_pin(resource.text_spec(resource.G_AMENDMENT_PATH, admission.ROOT, 1048576), admission.ROOT)
    walk(actual_amendment)
    amendment = conserved_put(root, resource.G_AMENDMENT_PATH, actual_amendment)
    prior = resource.read_pin(mapped[resource.F_SOURCE_PATH]["source"], root)
    freeze = dict(prior, qualification_epoch=resource.G_EPOCH, reviewed_source=review_spec,
                  qualification_amendment=amendment, sources=review["sources"], body_retry=review["sources"][resource.G_PATH])
    source = conserved_put(root, resource.G_SOURCE_PATH, freeze)
    before_raw = resource.raw_pin(resource.G_CLOSED_PIN, root, 524288)
    before = json.loads(before_raw)
    resource.fixture_text_write(root, admission.LEDGER, before_raw)
    after = resource.g_registration(before, amendment, root, supervisor.generated_budget_limit, before_raw=before_raw)
    policy, grant = resource.read_pin(resource.POLICY_PIN, root), resource.read_pin(resource.J_PIN, root)
    return SimpleNamespace(root=root, before=before, before_raw=before_raw, after=after, amendment=amendment,
                           review=review, reviewed=review_spec, source=source, policy=policy, grant=grant)


def conserved_g_request(fixture, role):
    from tools import banc_m3_conserved_resource as resource
    plan = resource.g_generated_plan(role, fixture.root)
    return {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"],
            "role": plan["role"], "source_freeze": fixture.source, "shared_original_lock_path": admission.SHARED_LOCK,
            "ledger_path": admission.LEDGER, "limits": plan["limits"], "generated_behavior": "command",
            "generated_argv": plan["argv"], "output": plan["output"]}


def conserved_g_paid(fixture, live, role):
    q = conserved_g_request(fixture, role)
    caps = tuple(q["limits"][k] for k in ("outer_s", "child_s", "cleanup_s"))
    cost = conserved_fake_paid(fixture.root, live, fixture.source, q["call_id"], q["role"], q["generated_argv"], caps,
                               name="generated-G-" + role, request=q)
    conserved_recompute(live, fixture.policy, fixture.grant, 2300, 15600)
    return cost


def conserved_g_qualified(fixture):
    from tools import banc_m3_conserved_resource as resource
    live, costs = copy.deepcopy(fixture.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        costs.append(conserved_g_paid(fixture, live, role))
    roster, suites = resource.complete_roster(fixture.root), []
    for role in ("math", "gate-a", "gate-b"):
        rows = roster[role.replace("-", "_")]
        spec = conserved_junit(fixture.root, resource.BASE + "/parent-contact-learning-g-" + role + "-junit.xml", rows)
        suites.append({"file": spec, "node_ids": [r["node_id"] for r in rows], "names": [r["name"] for r in rows],
                       "classnames": [r["classname"] for r in rows], "count": len(rows)})
    q = {"schema": "flylab.m3.direct-science.qualification.v4", "epoch": resource.G_EPOCH, "old_E360": "UNPASSED",
         "allocation": live[resource.ALLOCATION_KEY], "qualification_amendment": fixture.amendment,
         "reviewed_source": fixture.reviewed, "qualification_roster": resource.ROSTER_PIN, "source_freeze": fixture.source,
         "actual_costs": costs, "tests": suites, "evidence": costs}
    return live, q


def conserved_g_negative_states(fixture):
    from tools import banc_m3_conserved_resource as resource
    cases = []
    for kind in ("missing-pointer", "extra-pointer", "alias", "caps", "failure-cost", "old-call", "old-debit", "malformed-pointer-legacy-caps", "missing-allocation-legacy-caps", "foreign-owner"):
        bad = copy.deepcopy(fixture.after)
        if kind == "missing-pointer":
            del bad[resource.G_KEY]
        elif kind == "extra-pointer":
            bad[resource.G_KEY + "_alias"] = fixture.amendment
        elif kind == "alias":
            bad[resource.G_KEY]["path"] += ".alias.json"
        elif kind == "caps":
            bad["direct_science_generated_limit_s"] += 1
        elif kind == "failure-cost":
            bad["debits"][-1]["actual_outer_full_return_s"] = 0.0
        elif kind == "old-call":
            bad["direct_science_calls"][resource.F_QUAL_CALLS[0]]["state"] = "admitted"
        elif kind == "old-debit":
            bad["debits"][0]["cost_id"] += "-changed"
        elif kind in ("malformed-pointer-legacy-caps", "missing-allocation-legacy-caps"):
            bad.pop(resource.ALLOCATION_KEY)
            bad.update(generated_shared_cumulative_limit_s=1700, direct_science_generated_limit_s=964,
                       original_cumulative_limit_s=16200)
            if kind == "malformed-pointer-legacy-caps":
                bad[resource.G_KEY] = {}
        else:
            bad = copy.deepcopy(fixture.before)
            bad["direct_science_workers"]["foreign"] = {"request": fixture.amendment}
        cases.append((kind, bad))
    return cases


def conserved_g_contract_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    fixture = conserved_g_metadata_fixture(root)
    assert set(fixture.after) - set(fixture.before) == {resource.G_KEY}
    assert {k: v for k, v in fixture.after.items() if k != resource.G_KEY} == fixture.before
    assert fixture.before["debits"][-1]["actual_outer_full_return_s"] == 1.755
    assert len(fixture.before["debits"]) == 284 and len(fixture.before["direct_science_calls"]) == 166
    for kind, bad in conserved_g_negative_states(fixture):
        if kind == "foreign-owner":
            with pytest.raises(ValueError):
                resource.g_registration(bad, fixture.amendment, root, supervisor.generated_budget_limit, before_raw=fixture.before_raw)
        else:
            with pytest.raises((ValueError, KeyError, FileNotFoundError)):
                resource.prequalification_dispatch(bad, conserved_g_request(fixture, "lint"), {}, root, supervisor.generated_budget_limit, phase="closed")
        if kind in ("malformed-pointer-legacy-caps", "missing-allocation-legacy-caps"):
            with pytest.raises((ValueError, KeyError, FileNotFoundError)):
                supervisor.generated_budget_limit(bad, root)
    with pytest.raises(ValueError, match="raw"):
        resource.g_registration(fixture.before, fixture.amendment, root, supervisor.generated_budget_limit, before_raw=fixture.before_raw + b" ")
    with pytest.raises(ValueError, match="duplicate"):
        resource.g_registration(fixture.after, fixture.amendment, root, supervisor.generated_budget_limit, before_raw=fixture.before_raw)
    return fixture


def conserved_g_registration_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_g_metadata_fixture(root)
    ledger = resource._text_path(root, admission.LEDGER)
    lock = supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-G-registration"})
    with pytest.raises(FileExistsError):
        supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-G-foreign"})
    try:
        resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        assert json.loads(ledger.read_bytes()) == f.after
        with pytest.raises(ValueError, match="raw-before"):
            resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        lock.uncertain = True
        with pytest.raises(ValueError, match="uncertain"):
            lock.release(True)
        assert lock.target.read_bytes() == lock.raw
        lock.uncertain = False
    finally:
        lock.release(True)
    assert not lock.target.exists()


def conserved_g_order_and_qualification_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_g_metadata_fixture(root)
    live = copy.deepcopy(f.after)
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_g_request(f, role)
        request = conserved_put(root, "generated-G-next-" + role + ".json", q)
        assert resource.g_dispatch(live, q, request, root, supervisor.generated_budget_limit, phase="closed")["epoch"] == resource.G_EPOCH
        wrong = dict(q, call_id=q["call_id"] + "-retry")
        bad = conserved_put(root, "generated-G-alias-" + role + ".json", wrong)
        with pytest.raises(ValueError):
            resource.g_dispatch(live, wrong, bad, root, supervisor.generated_budget_limit, phase="closed")
        original = dict(q, mode="original", call_id="startup-ray-legacy-blocked", role="input_audit",
                        science_request={"path": "must-not-read-Original.npz", "bytes": 1, "sha256": "0" * 64})
        original_spec = conserved_put(root, "generated-G-Original-blocked-" + role + ".json", original)
        with pytest.raises(ValueError, match="global Original"):
            resource.g_dispatch(live, original, original_spec, root, supervisor.generated_budget_limit, phase="closed")
        conserved_g_paid(f, live, role)
    complete, qualification = conserved_g_qualified(f)
    assert resource.validate_g_qualification(qualification, complete, root, supervisor.generated_budget_limit) == 288
    for bad in conserved_g_qualification_negatives(qualification):
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.validate_g_qualification(bad, complete, root, supervisor.generated_budget_limit)
    for legacy in (resource.E_QUAL_PATH, resource.F_QUAL_PATH):
        bad = copy.deepcopy(complete)
        bad["direct_science_qualification"] = {"path": legacy, "bytes": 1, "sha256": "0" * 64}
        q = {"mode": "original", "source_freeze": f.source}
        with pytest.raises(ValueError, match="old E/F"):
            resource.g_dispatch(bad, q, {}, root, supervisor.generated_budget_limit, phase="closed")
    assert [len(qualification["tests"][i]["node_ids"]) for i in range(3)] == [19, 21, 32]



def conserved_g_text_cases(root):
    from tools import banc_m3_conserved_resource as resource
    root.mkdir(parents=True, exist_ok=True)
    name = "/".join(["generated-long-text"] + ["segment" + str(i).zfill(2) + "x" * 28 for i in range(8)]) + "/metadata.json"
    raw = b'{"fixture":"UTF8-text-only","value":1}'
    pin = resource.fixture_text_write(root, name, raw)
    assert len(str(root / name)) > 269 and resource.raw_pin(pin, root) == raw
    assert resource.text_spec(name, root) == pin
    before = resource._text_stamp(resource._text_path(root, name).stat())
    target = resource._text_path(root, name)
    import os
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000_000))
    assert resource._text_stamp(target.stat()) != before
    for field, value in (("sha256", "0" * 64), ("bytes", True), ("bytes", len(raw) + 1)):
        wrong = dict(pin, **{field: value})
        with pytest.raises(ValueError):
            resource.raw_pin(wrong, root)
    for bad in ("../escape.json", "/escape.json", "C:/escape.json", "\\\\?\\C:\\escape.json", "\\\\server\\share\\file.json",
                "generated:ads.json", "generated\\file.json", "./metadata.json", "generated/NUL.json", "generated/../file.json"):
        with pytest.raises(ValueError):
            resource.raw_pin(dict(pin, path=bad), root)
    with pytest.raises((ValueError, FileNotFoundError)):
        resource.raw_pin(dict(pin, path="not-present/metadata.json"), root)
    invalid = b"\xff"
    invalid_target = resource._text_path(root, "generated-invalid-utf8.json", write=True)
    invalid_target.write_bytes(invalid)
    with pytest.raises(UnicodeDecodeError):
        resource.raw_pin({"path": "generated-invalid-utf8.json", "bytes": 1, "sha256": hashlib.sha256(invalid).hexdigest()}, root)
    assert resource._text_path(root, name).read_bytes() == raw


def conserved_g_source_and_scope_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_g_metadata_fixture(root)
    good_raw = resource.raw_pin(f.reviewed, root)
    for kind in ("schema", "missing", "extra", "replacement", "external", "own-cycle"):
        bad = copy.deepcopy(f.review)
        if kind == "schema":
            bad["schema"] = "flylab.m3.contact-qualification.reviewed-source.v2"
        elif kind == "missing":
            del bad["sources"][resource.G_REPLACEMENTS[0]]
        elif kind == "extra":
            bad["sources"]["generated-extra.py"] = {"path": "generated-extra.py", "bytes": 1, "sha256": "0" * 64}
        elif kind == "replacement":
            bad["replacements"][resource.G_REPLACEMENTS[0]]["before"]["sha256"] = "0" * 64
        elif kind == "external":
            name = next(n for n in bad["sources"] if Path(n).is_absolute())
            bad["sources"][name]["sha256"] = "0" * 64
        else:
            bad["sources"][resource.G_REVIEW_PATH] = f.reviewed
        try:
            pin = conserved_put(root, resource.G_REVIEW_PATH, bad)
            with pytest.raises((ValueError, KeyError, FileNotFoundError)):
                resource.g_current_review(pin, root)
        finally:
            resource.fixture_text_write(root, resource.G_REVIEW_PATH, good_raw)
    q = conserved_g_request(f, "lint")
    request = conserved_put(root, "generated-G-Source-drift-request.json", q)
    source = resource._text_path(root, resource.HELPER_SOURCE)
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    try:
        assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
        with pytest.raises(ValueError):
            resource.g_dispatch(f.after, q, request, root, supervisor.generated_budget_limit, phase="closed")
    finally:
        source.write_bytes(original)
    body = resource.read_pin(f.review["sources"][resource.G_PATH], root)
    contract = resource.read_pin(body["contract"], root)
    science = {"call_id": resource.G_CALLS[0], "role": admission.ROLES[2], "source_freeze": f.source,
               "body_retry": f.review["sources"][resource.G_PATH], "contract": body["contract"], "grant": body["grant"],
               "inputs": body["inputs"], "roster": body["roster"], "limits": contract["limits"][admission.ROLES[2]]}
    read = lambda p: resource.read_pin(p, root)
    assert admission.retry_binding(science, contract, read) == body
    assert body["candidate"] == admission.RETRY_PINS["candidate"] and body["failed_f_lint"]["path"] == resource.G_F_PATHS["cost"]
    assert body["maximum_body_calls"] == body["maximum_audit_calls"] == 1 and body["automatic_repeats"] == 0
    with pytest.raises(ValueError):
        admission.retry_binding(dict(science, call_id=resource.F_CALLS[0]), contract, read)


def conserved_g_native_cases_file(root):
    """Actual temporary supervisor refusal plus synthetic once-accounting inputs."""
    import time

    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    conserved_g_text_cases(root)
    f = conserved_g_metadata_fixture(root)
    before = conserved_put(root, "generated-G-before.json", f.before)
    after = conserved_put(root, "generated-G-after.json", f.after)
    states = []
    live = copy.deepcopy(f.after)
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_g_request(f, role)
        request = conserved_put(root, "generated-G-ordered-request-" + role + ".json", q)
        state = conserved_put(root, "generated-G-ordered-state-" + role + ".json", live)
        states.append({"state": state, "q": q, "request": request, "phase": "closed", "accept": True})
        conserved_g_paid(f, live, role)
    qualified_live, qualification = conserved_g_qualified(f)
    qualified_state = conserved_put(root, "generated-G-qualified-state.json", qualified_live)
    negatives = [{"label": k, "state": conserved_put(root, "generated-G-negative-" + k + ".json", v)}
                 for k, v in conserved_g_negative_states(f) if k != "foreign-owner"]
    q = conserved_g_request(f, "lint")
    request_name = "generated-G-actual-initial-request.json"
    request = conserved_put(root, request_name, q)
    admitted = copy.deepcopy(f.after)
    admitted["direct_science_calls"][q["call_id"]] = {"request": request, "state": "admitted", "mode": "generated", "reserved_s": 15}
    admitted["direct_science_reservations"] = {q["call_id"]: {"request": request, "reserved_s": 15}}
    conserved_put(root, admission.LEDGER, admitted)
    admitted_raw = resource._text_path(root, admission.LEDGER).read_bytes()
    source = resource._text_path(root, resource.HELPER_SOURCE)
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    try:
        outer = supervisor._supervise(request_name, root=root, entry=time.perf_counter())
        assert outer["refusal_origin"] == "ACTUAL_G_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT"
        assert outer["job_created"] is False and outer["worker_started"] is False and outer["worker_pid"] is None
        assert outer["stdout"] is None and outer["stderr"] is None and outer["stream_state"] == "NO_WORKER_STREAMS_OPENED"
        assert not any(k in outer for k in ("trace", "dispatch_ms", "first_final_ms", "actual_outer_full_return_s"))
        assert outer["initial_ledger"]["sha256"] == hashlib.sha256(admitted_raw).hexdigest()
        assert resource._text_path(root, admission.LEDGER).read_bytes() == admitted_raw
        assert resource.current_caps(admitted, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
    finally:
        source.write_bytes(original)
    outer_pin = resource.text_spec(q["output"] + "/outer.json", root, 1048576)
    trace_name = "generated-G-failed-trace.json"
    trace = conserved_put(root, trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
        "dispatch_ms": 1791420000000, "first_final_ms": 1791420000125, "full_return_s": 0.125, "exit_code": 1,
        "source_freeze": f.source, "request": request, "fixture_only": True})
    account_text = (admission.ROOT / admission.BASE / "parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert account_text.count(anchor) == 1
    account_text = account_text.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + root.as_posix().replace("'", "''") + "')")
    resource.fixture_text_write(root, "generated-G-account.ps1", account_text.encode("utf8"))
    resource.fixture_text_write(root, admission.LEDGER, f.before_raw)
    body = resource.read_pin(f.review["sources"][resource.G_PATH], root)
    contract = resource.read_pin(body["contract"], root)
    science = {"call_id": resource.G_CALLS[0], "role": admission.ROLES[2], "source_freeze": f.source,
               "body_retry": f.review["sources"][resource.G_PATH], "contract": body["contract"], "grant": body["grant"],
               "inputs": body["inputs"], "roster": body["roster"], "limits": contract["limits"][admission.ROLES[2]]}
    return conserved_put(root, "generated-G-cases.json", {"before": before, "before_raw": resource.G_CLOSED_PIN, "after": after,
        "amendment": f.amendment, "source": f.source, "states": states, "negatives": negatives,
        "qualified_state": qualified_state, "qualification": qualification, "qualification_negatives": conserved_g_qualification_negatives(qualification),
        "admitted": conserved_put(root, "generated-G-admitted.json", admitted), "admitted_raw_b64": __import__("base64").b64encode(admitted_raw).decode("ascii"),
        "request": request, "q": q, "outer": outer_pin, "trace": trace, "trace_name": trace_name,
        "science": science, "critical_sources": list(resource.G_REPLACEMENTS), "fixture_only": True})


def conserved_h_metadata_fixture(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor

    conserved_g_metadata_fixture(root)
    mapped = resource.h_archives(admission.ROOT)
    for item in mapped.values():
        resource.fixture_text_write(root, item["archive"]["path"], resource.raw_pin(item["archive"], admission.ROOT, 524288, allow_empty=True))
        if item["source"]["path"] not in resource.G_REPLACEMENTS:
            resource.fixture_text_write(root, item["source"]["path"], resource.raw_pin(item["source"], admission.ROOT, 524288, allow_empty=True))
    copied = set()
    def walk(value):
        if type(value) is dict:
            if set(value) == {"path", "bytes", "sha256"}:
                name = value["path"]
                if Path(name).is_absolute() or name == admission.LEDGER or name in resource.G_REPLACEMENTS:
                    return
                if Path(name).suffix not in (".json", ".md", ".txt", ".xml", ".ini"):
                    return
                key = (name, value["sha256"])
                if key in copied:
                    return
                copied.add(key)
                raw = resource.raw_pin(value, admission.ROOT, 2097152, allow_empty=True)
                resource.fixture_text_write(root, name, raw)
                if Path(name).suffix == ".json":
                    walk(json.loads(raw))
            else:
                for key, child in value.items():
                    if key != "sources":
                        walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
    reviewed = resource.text_spec(resource.H_REVIEW_PATH, admission.ROOT, 1048576)
    _closed, _policy, _grant, historical, _mapped, _pins = resource.h_history(admission.ROOT, supervisor.generated_budget_limit)
    review = resource.h_review_metadata(reviewed, admission.ROOT, historical, mapped)
    for spec in review["sources"].values():
        if not Path(spec["path"]).is_absolute():
            conserved_copy(root, spec)
    for spec in (resource.H_BEFORE_PIN, resource.H_ROOT_PIN, *resource.H_ASTRA_PINS.values(), *review["root_reviews"], *review["sol_reports"], reviewed):
        walk(spec)
    repair = resource.text_spec(resource.H_REPAIR_PATH, admission.ROOT, 1048576)
    walk(repair)
    a = resource.read_pin(repair, root)
    prior = resource.read_pin(mapped[resource.G_SOURCE_PATH]["source"], root)
    source = conserved_put(root, resource.H_SOURCE_PATH, dict(prior, qualification_epoch=resource.H_EPOCH,
        reviewed_source=reviewed, qualification_amendment=a["qualification_amendment"], qualification_lint_repair=repair,
        sources=review["sources"], body_retry=review["sources"][resource.H_PATH]))
    assert source["bytes"] <= 262144
    before_raw = resource.raw_pin(resource.H_CLOSED_PIN, root, 524288)
    before = json.loads(before_raw)
    resource.fixture_text_write(root, admission.LEDGER, before_raw)
    after = resource.h_registration(before, repair, root, supervisor.generated_budget_limit, before_raw=before_raw)
    return SimpleNamespace(root=root, before=before, before_raw=before_raw, after=after, repair=repair, reviewed=reviewed, review=review, source=source,
        policy=resource.read_pin(resource.POLICY_PIN, root), grant=resource.read_pin(resource.J_PIN, root))


def conserved_h_request(fixture, role):
    from tools import banc_m3_conserved_resource as resource
    plan = resource.h_generated_plan(role, fixture.root)
    return {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"], "role": plan["role"],
        "source_freeze": fixture.source, "shared_original_lock_path": admission.SHARED_LOCK, "ledger_path": admission.LEDGER,
        "limits": plan["limits"], "generated_behavior": "command", "generated_argv": plan["argv"], "output": plan["output"]}


def conserved_h_paid(fixture, live, role):
    q = conserved_h_request(fixture, role)
    cost = conserved_fake_paid(fixture.root, live, fixture.source, q["call_id"], q["role"], q["generated_argv"],
        tuple(q["limits"][k] for k in ("outer_s", "child_s", "cleanup_s")), name="generated-H-" + role, request=q)
    conserved_recompute(live, fixture.policy, fixture.grant, 2300, 15600)
    return cost


def conserved_h_qualified(fixture):
    from tools import banc_m3_conserved_resource as resource
    live, costs = copy.deepcopy(fixture.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        costs.append(conserved_h_paid(fixture, live, role))
    roster, suites = resource.complete_roster(fixture.root), []
    for role in ("math", "gate-a", "gate-b"):
        rows = roster[role.replace("-", "_")]
        spec = conserved_junit(fixture.root, resource.BASE + "/parent-contact-learning-h-" + role + "-junit.xml", rows)
        suites.append({"file": spec, "node_ids": [r["node_id"] for r in rows], "names": [r["name"] for r in rows],
                       "classnames": [r["classname"] for r in rows], "count": len(rows)})
    return live, {"schema": "flylab.m3.direct-science.qualification.v5", "epoch": resource.H_EPOCH, "old_E360": "UNPASSED",
        "allocation": live[resource.ALLOCATION_KEY], "qualification_amendment": live[resource.G_KEY], "qualification_lint_repair": fixture.repair,
        "reviewed_source": fixture.reviewed, "qualification_roster": resource.ROSTER_PIN, "source_freeze": fixture.source,
        "actual_costs": costs, "tests": suites, "evidence": costs}


def conserved_h_negative_states(fixture):
    from tools import banc_m3_conserved_resource as resource
    cases = []
    for kind in ("extra-pointer", "alias", "caps", "failed-G-refund", "old-G-call", "old-debit", "missing-G-pointer"):
        bad = copy.deepcopy(fixture.after)
        if kind == "extra-pointer":
            bad[resource.H_KEY + "_alias"] = fixture.repair
        elif kind == "alias":
            bad[resource.H_KEY]["path"] += ".alias.json"
        elif kind == "caps":
            bad["direct_science_generated_limit_s"] += 1
        elif kind == "failed-G-refund":
            bad["debits"][284]["actual_outer_full_return_s"] = 0.0
        elif kind == "old-G-call":
            bad["direct_science_calls"][resource.G_QUAL_CALLS[0]]["state"] = "admitted"
        elif kind == "old-debit":
            bad["debits"][0]["cost_id"] += "-changed"
        else:
            del bad[resource.G_KEY]
        cases.append((kind, bad))
    return cases


def conserved_h_contract_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_h_metadata_fixture(root)
    assert set(f.after) - set(f.before) == {resource.H_KEY} and all(f.after[k] == v for k, v in f.before.items())
    assert len(f.before["debits"]) == 285 and len(f.before["direct_science_calls"]) == 167
    assert f.before["debits"][284]["actual_outer_full_return_s"] == 11.355 and f.before["debits"][284]["actual_returned_exit_code"] == 1
    assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
    for _kind, bad in conserved_h_negative_states(f):
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.current_caps(bad, root, supervisor.generated_budget_limit)
    with pytest.raises(ValueError, match="raw"):
        resource.h_registration(f.before, f.repair, root, supervisor.generated_budget_limit, before_raw=f.before_raw + b" ")
    with pytest.raises(ValueError, match="duplicate"):
        resource.h_registration(f.after, f.repair, root, supervisor.generated_budget_limit, before_raw=f.before_raw)


def conserved_h_registration_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_h_metadata_fixture(root)
    ledger = resource._text_path(root, admission.LEDGER)
    lock = supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-H-registration"})
    with pytest.raises(FileExistsError):
        supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-H-foreign"})
    try:
        resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        published = ledger.read_bytes()
        assert json.loads(published) == f.after
        with pytest.raises(ValueError, match="raw-before"):
            resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        assert ledger.read_bytes() == published
        lock.uncertain = True
        with pytest.raises(ValueError, match="uncertain"):
            lock.release(True)
        assert lock.target.read_bytes() == lock.raw
        lock.uncertain = False
    finally:
        lock.release(True)
    q = conserved_h_request(f, "lint")
    request = conserved_put(root, "generated-H-closed-request.json", q)
    for key in resource.OWNER_KEYS:
        wrong = copy.deepcopy(f.after)
        wrong[key] = {"foreign": {"state": "ownership_unknown"}}
        assert resource.current_caps(wrong, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
        with pytest.raises(ValueError):
            resource.prequalification_dispatch(wrong, q, request, root, supervisor.generated_budget_limit, phase="closed")


def conserved_h_order_and_qualification_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_h_metadata_fixture(root)
    live = copy.deepcopy(f.after)
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_h_request(f, role)
        request = conserved_put(root, "generated-H-next-" + role + ".json", q)
        assert resource.prequalification_dispatch(live, q, request, root, supervisor.generated_budget_limit, phase="closed")["epoch"] == resource.H_EPOCH
        for wrong in (dict(q, call_id=q["call_id"] + "-retry"), dict(q, generated_argv=q["generated_argv"][:2])):
            bad = conserved_put(root, "generated-H-wrong-next.json", wrong)
            with pytest.raises(ValueError):
                resource.h_dispatch(live, wrong, bad, root, supervisor.generated_budget_limit, phase="closed")
        for original_role in ("input_audit", "neutral_settling", "saved_startup_transfer", "tracking_neutral", "live_replay", "geometry_capture", *admission.ROLES):
            original = dict(q, mode="original", call_id="startup-ray-legacy-blocked", role=original_role,
                            science_request={"path": "must-not-read-Original.npz", "bytes": 1, "sha256": "0" * 64})
            bad = conserved_put(root, "generated-H-legacy-blocked.json", original)
            with pytest.raises(ValueError, match="global Original"):
                resource.prequalification_dispatch(live, original, bad, root, supervisor.generated_budget_limit, phase="closed")
        conserved_h_paid(f, live, role)
    complete, qualified = conserved_h_qualified(f)
    assert resource.validate_qualification(qualified, complete, root, supervisor.generated_budget_limit) == 289
    for bad in conserved_g_qualification_negatives(qualified) + [dict(qualified, qualification_lint_repair=f.before[resource.G_KEY])]:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.validate_qualification(bad, complete, root, supervisor.generated_budget_limit)
    q = conserved_h_request(f, "gate-b")
    request = conserved_put(root, "generated-H-no-fifth.json", q)
    with pytest.raises(ValueError):
        resource.h_dispatch(complete, q, request, root, supervisor.generated_budget_limit, phase="closed")


def conserved_h_source_and_scope_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_h_metadata_fixture(root)
    repair = resource.read_pin(f.repair, root)
    for name in resource.G_REPLACEMENTS:
        target = resource._text_path(root, name)
        raw = target.read_bytes()
        target.write_bytes(raw + b"\n")
        try:
            assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
            with pytest.raises(ValueError):
                resource.h_current_review(repair["reviewed_source"], root, supervisor.generated_budget_limit)
        finally:
            target.write_bytes(raw)
    body_spec = f.review["sources"][resource.H_PATH]
    body = resource.read_pin(body_spec, root)
    contract = resource.read_pin(body["contract"], root)
    science = {"call_id": resource.H_CALLS[0], "role": admission.ROLES[2], "source_freeze": f.source, "body_retry": body_spec,
               "contract": body["contract"], "grant": body["grant"], "inputs": body["inputs"], "roster": body["roster"], "limits": contract["limits"][admission.ROLES[2]]}
    def read(spec):
        return resource.read_pin(spec, root)
    assert admission.retry_binding(science, contract, read) == body
    assert body["qualification_amendment_path"] == resource.G_AMENDMENT_PATH and body["allocation_event_path"] == resource.F_EVENT_PATH
    for call in (*resource.G_CALLS, *resource.F_CALLS):
        with pytest.raises(ValueError):
            admission.runtime_h_binding(dict(science, call_id=call), contract, read)


def conserved_h_native_cases_file(root):
    """Synthetic H costs; actual temporary lexical refusal, current owned Native."""
    import time

    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor

    f = conserved_h_metadata_fixture(root)
    live, states = copy.deepcopy(f.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_h_request(f, role)
        request = conserved_put(root, "generated-H-ordered-request-" + role + ".json", q)
        states.append({"state": conserved_put(root, "generated-H-ordered-state-" + role + ".json", live), "q": q, "request": request})
        conserved_h_paid(f, live, role)
    qualified_live, qualified = conserved_h_qualified(f)
    negatives = [{"label": label, "state": conserved_put(root, "generated-H-negative-" + label + ".json", bad)}
                 for label, bad in conserved_h_negative_states(f)]
    q = conserved_h_request(f, "lint")
    request_name = "generated-H-actual-initial-request.json"
    request = conserved_put(root, request_name, q)
    admitted = copy.deepcopy(f.after)
    admitted["direct_science_calls"][q["call_id"]] = {"request": request, "state": "admitted", "mode": "generated", "reserved_s": 15}
    admitted["direct_science_reservations"] = {q["call_id"]: {"request": request, "reserved_s": 15}}
    conserved_put(root, admission.LEDGER, admitted)
    admitted_raw = resource._text_path(root, admission.LEDGER).read_bytes()
    target = resource._text_path(root, resource.HELPER_SOURCE)
    raw = target.read_bytes()
    target.write_bytes(raw + b"\n")
    try:
        outer = supervisor._supervise(request_name, root=root, entry=time.perf_counter())
        assert outer["refusal_origin"] == "ACTUAL_H_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT"
        assert outer["qualification_lint_repair"] == f.repair and outer["qualification_amendment"] == f.before[resource.G_KEY]
        assert outer["first_error"]["type"] == "ValueError" and outer["job_created"] is False and outer["worker_started"] is False
        assert outer["stdout"] is None and outer["stderr"] is None and not outer["stdout_eof_observed"] and not outer["stderr_eof_observed"]
        assert outer["runtime_reservation_state"] == "NEVER_CREATED" and outer["root_admission_reservation_state"] == "STILL_PRESENT_PENDING_NATIVE_COMMIT"
        assert not any(k in outer for k in ("trace", "dispatch_ms", "first_final_ms", "actual_outer_full_return_s"))
        assert resource.raw_pin(outer["initial_ledger"], root, 524288) == admitted_raw
        assert resource.current_caps(admitted, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
    finally:
        target.write_bytes(raw)
    trace_name = "generated-H-failed-trace.json"
    conserved_put(root, trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
        "dispatch_ms": 1791420000000, "first_final_ms": 1791420000125, "full_return_s": 0.125, "exit_code": 1,
        "source_freeze": f.source, "request": request, "fixture_only": True})
    account_text = (admission.ROOT / admission.BASE / "parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert account_text.count(anchor) == 1
    account_text = account_text.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + root.as_posix().replace("'", "''") + "')")
    resource.fixture_text_write(root, "generated-H-account.ps1", account_text.encode("utf8"))
    resource.fixture_text_write(root, admission.LEDGER, f.before_raw)
    return conserved_put(root, "generated-H-cases.json", {"before": conserved_put(root, "generated-H-before.json", f.before),
        "before_raw": resource.H_CLOSED_PIN, "after": conserved_put(root, "generated-H-after.json", f.after), "repair": f.repair, "source": f.source,
        "states": states, "negatives": negatives, "qualified_state": conserved_put(root, "generated-H-qualified.json", qualified_live),
        "qualification": qualified, "qualification_negatives": conserved_g_qualification_negatives(qualified) + [dict(qualified, qualification_lint_repair=f.before[resource.G_KEY])],
        "admitted_raw_b64": __import__("base64").b64encode(admitted_raw).decode("ascii"), "q": q, "request": request,
        "trace_name": trace_name, "critical_sources": list(resource.G_REPLACEMENTS), "fixture_only": True})


def conserved_i_metadata_fixture(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor

    conserved_h_metadata_fixture(root)
    mapped = resource.i_archives(admission.ROOT)
    for item in mapped.values():
        resource.fixture_text_write(root, item["archive"]["path"], resource.raw_pin(item["archive"], admission.ROOT, 524288, allow_empty=True))
        if item["source"]["path"] not in resource.G_REPLACEMENTS:
            resource.fixture_text_write(root, item["source"]["path"], resource.raw_pin(item["source"], admission.ROOT, 524288, allow_empty=True))
    copied = set()
    def walk(value):
        if type(value) is dict:
            if set(value) == {"path", "bytes", "sha256"}:
                name = value["path"]
                if Path(name).is_absolute() or name == admission.LEDGER or name in resource.G_REPLACEMENTS:
                    return
                if Path(name).suffix not in (".json", ".md", ".txt", ".xml", ".ini"):
                    return
                key = (name, value["sha256"])
                if key in copied:
                    return
                copied.add(key)
                raw = resource.raw_pin(value, admission.ROOT, 2097152, allow_empty=True)
                resource.fixture_text_write(root, name, raw)
                if Path(name).suffix == ".json":
                    walk(json.loads(raw))
            else:
                for key, child in value.items():
                    if key != "sources":
                        walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
    reviewed = resource.text_spec(resource.I_REVIEW_PATH, admission.ROOT, 1048576)
    _closed, _policy, _grant, historical, _mapped, _pins = resource.i_history(admission.ROOT, supervisor.generated_budget_limit)
    review = resource.i_review_metadata(reviewed, admission.ROOT, historical, mapped)
    for spec in review["sources"].values():
        if not Path(spec["path"]).is_absolute():
            conserved_copy(root, spec)
    for spec in (resource.I_BEFORE_PIN, resource.I_ROOT_PIN, *resource.I_ASTRA_PINS.values(), *review["root_reviews"], *review["sol_reports"], reviewed):
        walk(spec)
    repair = resource.text_spec(resource.I_RECOVERY_PATH, admission.ROOT, 1048576)
    walk(repair)
    resource.read_pin(repair, root)
    prior = resource.read_pin(mapped[resource.H_SOURCE_PATH]["source"], root)
    source = conserved_put(root, resource.I_SOURCE_PATH, dict(prior, qualification_epoch=resource.I_EPOCH,
        reviewed_source=reviewed, preflight_recovery=repair,
        sources=review["sources"], body_retry=review["sources"][resource.I_PATH]))
    assert source["bytes"] <= 262144
    before_raw = resource.raw_pin(resource.I_CLOSED_PIN, root, 524288)
    before = json.loads(before_raw)
    resource.fixture_text_write(root, admission.LEDGER, before_raw)
    after = resource.i_registration(before, repair, root, supervisor.generated_budget_limit, before_raw=before_raw)
    return SimpleNamespace(root=root, before=before, before_raw=before_raw, after=after, recovery=repair, reviewed=reviewed, review=review, source=source,
        policy=resource.read_pin(resource.POLICY_PIN, root), grant=resource.read_pin(resource.J_PIN, root))


def conserved_i_request(fixture, role):
    from tools import banc_m3_conserved_resource as resource
    plan = resource.i_generated_plan(role, fixture.root)
    return {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "call_id": plan["call"], "role": plan["role"],
        "source_freeze": fixture.source, "shared_original_lock_path": admission.SHARED_LOCK, "ledger_path": admission.LEDGER,
        "limits": plan["limits"], "generated_behavior": "command", "generated_argv": plan["argv"], "output": plan["output"]}


def conserved_i_paid(fixture, live, role):
    q = conserved_i_request(fixture, role)
    cost = conserved_fake_paid(fixture.root, live, fixture.source, q["call_id"], q["role"], q["generated_argv"],
        tuple(q["limits"][k] for k in ("outer_s", "child_s", "cleanup_s")), name="generated-I-" + role, request=q)
    conserved_recompute(live, fixture.policy, fixture.grant, 2300, 15600)
    return cost


def conserved_i_qualified(fixture):
    from tools import banc_m3_conserved_resource as resource
    live, costs = copy.deepcopy(fixture.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        costs.append(conserved_i_paid(fixture, live, role))
    roster, suites = resource.complete_roster(fixture.root), []
    for role in ("math", "gate-a", "gate-b"):
        rows = roster[role.replace("-", "_")]
        spec = conserved_junit(fixture.root, resource.BASE + "/parent-contact-learning-i-" + role + "-junit.xml", rows)
        suites.append({"file": spec, "node_ids": [r["node_id"] for r in rows], "names": [r["name"] for r in rows],
                       "classnames": [r["classname"] for r in rows], "count": len(rows)})
    return live, {"schema": "flylab.m3.direct-science.qualification.v6", "epoch": resource.I_EPOCH, "old_E360": "UNPASSED",
        "allocation": live[resource.ALLOCATION_KEY], "qualification_amendment": live[resource.G_KEY], "qualification_lint_repair": fixture.before[resource.H_KEY], "preflight_recovery": fixture.recovery,
        "reviewed_source": fixture.reviewed, "qualification_roster": resource.ROSTER_PIN, "source_freeze": fixture.source,
        "actual_costs": costs, "tests": suites, "evidence": costs}


def conserved_i_negative_states(fixture):
    from tools import banc_m3_conserved_resource as resource
    cases = []
    for kind in ("extra-pointer", "alias", "caps", "failed-G-refund", "old-G-call", "old-debit", "missing-G-pointer"):
        bad = copy.deepcopy(fixture.after)
        if kind == "extra-pointer":
            bad[resource.I_KEY + "_alias"] = fixture.recovery
        elif kind == "alias":
            bad[resource.I_KEY]["path"] += ".alias.json"
        elif kind == "caps":
            bad["direct_science_generated_limit_s"] += 1
        elif kind == "failed-G-refund":
            bad["debits"][284]["actual_outer_full_return_s"] = 0.0
        elif kind == "old-G-call":
            bad["direct_science_calls"][resource.G_QUAL_CALLS[0]]["state"] = "admitted"
        elif kind == "old-debit":
            bad["debits"][0]["cost_id"] += "-changed"
        else:
            del bad[resource.G_KEY]
        cases.append((kind, bad))
    return cases


def conserved_i_contract_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_i_metadata_fixture(root)
    assert set(f.after) - set(f.before) == {resource.I_KEY} and all(f.after[k] == v for k, v in f.before.items())
    assert len(f.before["debits"]) == 286 and len(f.before["direct_science_calls"]) == 168
    assert f.before["debits"][285]["actual_outer_full_return_s"] == 13.253 and f.before["debits"][285]["actual_returned_exit_code"] == 124
    assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
    assert sum(p["limits"]["outer_s"] for p in resource.i_plan(root)["calls"]) + 20 == 575
    assert 1564.0 - f.before["direct_science_generated_used_s"] >= 575
    assert f.before["generated_shared_remaining_s"] >= 575
    cost = resource.read_pin(resource.i_archives(root)[resource.I_H_PATHS["cost"]]["source"], root)
    debit = f.before["debits"][285]
    with pytest.raises(ValueError):
        admission.cost_proof(cost, debit, lambda p: resource.read_pin(p, root), generated=True)
    for field, value in (("dispatch_ms", True), ("first_final_ms", "1791444067776"), ("dispatch_ms", 1791444054524)):
        trace = dict(resource.read_pin(cost["trace"], root), **{field: value})
        def read(p, _trace=trace):
            return _trace if p == cost["trace"] else resource.read_pin(p, root)
        with pytest.raises(ValueError):
            admission.cost_proof(cost, debit, read, generated=True, expected_timeout=True)
    for field in (resource.I_KEY, resource.H_KEY):
        missing = copy.deepcopy(f.after)
        del missing[field]
        q = conserved_i_request(f, "lint")
        request = conserved_put(root, "generated-I-missing-pointer.json", q)
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.prequalification_dispatch(missing, q, request, root, supervisor.generated_budget_limit, phase="closed")
    for _kind, bad in conserved_i_negative_states(f):
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.current_caps(bad, root, supervisor.generated_budget_limit)
    with pytest.raises(ValueError, match="raw"):
        resource.i_registration(f.before, f.recovery, root, supervisor.generated_budget_limit, before_raw=f.before_raw + b" ")
    with pytest.raises(ValueError, match="duplicate"):
        resource.i_registration(f.after, f.recovery, root, supervisor.generated_budget_limit, before_raw=f.before_raw)


def conserved_i_registration_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_i_metadata_fixture(root)
    ledger = resource._text_path(root, admission.LEDGER)
    lock = supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-I-registration"})
    with pytest.raises(FileExistsError):
        supervisor.SharedLock(resource._text_path(root, admission.SHARED_LOCK, write=True), {"call_id": "generated-I-foreign"})
    try:
        resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        published = ledger.read_bytes()
        assert json.loads(published) == f.after
        with pytest.raises(ValueError, match="raw-before"):
            resource.atomic_ledger_publish(ledger, f.before_raw, f.after, lock, validate=lambda l: resource.current_caps(l, root, supervisor.generated_budget_limit))
        assert ledger.read_bytes() == published
        lock.uncertain = True
        with pytest.raises(ValueError, match="uncertain"):
            lock.release(True)
        assert lock.target.read_bytes() == lock.raw
        lock.uncertain = False
    finally:
        lock.release(True)
    q = conserved_i_request(f, "lint")
    request = conserved_put(root, "generated-I-closed-request.json", q)
    for key in resource.OWNER_KEYS:
        wrong = copy.deepcopy(f.after)
        wrong[key] = {"foreign": {"state": "ownership_unknown"}}
        assert resource.current_caps(wrong, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
        with pytest.raises(ValueError):
            resource.prequalification_dispatch(wrong, q, request, root, supervisor.generated_budget_limit, phase="closed")


def conserved_i_order_and_qualification_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_i_metadata_fixture(root)
    live = copy.deepcopy(f.after)
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_i_request(f, role)
        request = conserved_put(root, "generated-I-next-" + role + ".json", q)
        assert resource.prequalification_dispatch(live, q, request, root, supervisor.generated_budget_limit, phase="closed")["epoch"] == resource.I_EPOCH
        for wrong in (dict(q, call_id=q["call_id"] + "-retry"), dict(q, generated_argv=q["generated_argv"][:2])):
            bad = conserved_put(root, "generated-I-wrong-next.json", wrong)
            with pytest.raises(ValueError):
                resource.i_dispatch(live, wrong, bad, root, supervisor.generated_budget_limit, phase="closed")
        for limits in (dict(q["limits"], outer_s=361), dict(q["limits"], outer_s=360, child_s=339),
                       dict(q["limits"], outer_s=360, cleanup_s=19)):
            wrong = dict(q, limits=limits)
            bad = conserved_put(root, "generated-I-wrong360.json", wrong)
            with pytest.raises(ValueError):
                resource.prequalification_dispatch(live, wrong, bad, root, supervisor.generated_budget_limit, phase="closed")
        if role != "gate-b":
            wrong = dict(q, limits=resource.i_generated_plan("gate-b", root)["limits"])
            bad = conserved_put(root, "generated-I-non-gate360.json", wrong)
            with pytest.raises(ValueError):
                resource.prequalification_dispatch(live, wrong, bad, root, supervisor.generated_budget_limit, phase="closed")
        for original_role in ("input_audit", "neutral_settling", "saved_startup_transfer", "tracking_neutral", "live_replay", "geometry_capture", *admission.ROLES):
            original = dict(q, mode="original", call_id="startup-ray-legacy-blocked", role=original_role,
                            science_request={"path": "must-not-read-Original.npz", "bytes": 1, "sha256": "0" * 64})
            bad = conserved_put(root, "generated-I-legacy-blocked.json", original)
            with pytest.raises(ValueError, match="global Original"):
                resource.prequalification_dispatch(live, original, bad, root, supervisor.generated_budget_limit, phase="closed")
        conserved_i_paid(f, live, role)
    complete, qualified = conserved_i_qualified(f)
    assert resource.validate_qualification(qualified, complete, root, supervisor.generated_budget_limit) == 290
    for bad in conserved_g_qualification_negatives(qualified) + [dict(qualified, preflight_recovery=f.before[resource.G_KEY])]:
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            resource.validate_qualification(bad, complete, root, supervisor.generated_budget_limit)
    q = conserved_i_request(f, "gate-b")
    request = conserved_put(root, "generated-I-no-fifth.json", q)
    with pytest.raises(ValueError):
        resource.i_dispatch(complete, q, request, root, supervisor.generated_budget_limit, phase="closed")


def conserved_i_source_and_scope_cases(root):
    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor
    f = conserved_i_metadata_fixture(root)
    repair = resource.read_pin(f.recovery, root)
    for name in resource.G_REPLACEMENTS:
        target = resource._text_path(root, name)
        raw = target.read_bytes()
        for fault in ("missing", "oversize"):
            try:
                if fault == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"#" * 262145)
                assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
                with pytest.raises((ValueError, FileNotFoundError)):
                    resource.i_current_review(repair["reviewed_source"], root, supervisor.generated_budget_limit)
            finally:
                target.write_bytes(raw)
    for name in resource.G_REPLACEMENTS:
        target = resource._text_path(root, name)
        raw = target.read_bytes()
        target.write_bytes(raw + b"\n")
        try:
            assert resource.current_caps(f.after, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
            with pytest.raises(ValueError):
                resource.i_current_review(repair["reviewed_source"], root, supervisor.generated_budget_limit)
        finally:
            target.write_bytes(raw)
    body_spec = f.review["sources"][resource.I_PATH]
    body = resource.read_pin(body_spec, root)
    contract = resource.read_pin(body["contract"], root)
    science = {"call_id": resource.I_CALLS[0], "role": admission.ROLES[2], "source_freeze": f.source, "body_retry": body_spec,
               "contract": body["contract"], "grant": body["grant"], "inputs": body["inputs"], "roster": body["roster"], "limits": contract["limits"][admission.ROLES[2]]}
    def read(spec):
        return resource.read_pin(spec, root)
    assert admission.retry_binding(science, contract, read) == body
    assert body["qualification_amendment_path"] == resource.G_AMENDMENT_PATH and body["allocation_event_path"] == resource.F_EVENT_PATH
    for call in (*resource.H_CALLS, *resource.G_CALLS, *resource.F_CALLS):
        with pytest.raises(ValueError):
            admission.runtime_h_binding(dict(science, call_id=call), contract, read)


def conserved_i_native_cases_file(root):
    """Synthetic H costs; actual temporary lexical refusal, current owned Native."""
    import time

    from tools import banc_m3_conserved_resource as resource
    from tools import banc_m3_startup_ray_supervisor as supervisor

    f = conserved_i_metadata_fixture(root)
    live, states = copy.deepcopy(f.after), []
    for role in ("lint", "math", "gate-a", "gate-b"):
        q = conserved_i_request(f, role)
        request = conserved_put(root, "generated-I-ordered-request-" + role + ".json", q)
        states.append({"state": conserved_put(root, "generated-I-ordered-state-" + role + ".json", live), "q": q, "request": request})
        conserved_i_paid(f, live, role)
    qualified_live, qualified = conserved_i_qualified(f)
    negatives = [{"label": label, "state": conserved_put(root, "generated-I-negative-" + label + ".json", bad)}
                 for label, bad in conserved_i_negative_states(f)]
    q = conserved_i_request(f, "lint")
    request_name = "generated-I-actual-initial-request.json"
    request = conserved_put(root, request_name, q)
    admitted = copy.deepcopy(f.after)
    admitted["direct_science_calls"][q["call_id"]] = {"request": request, "state": "admitted", "mode": "generated", "reserved_s": 40}
    admitted["direct_science_reservations"] = {q["call_id"]: {"request": request, "reserved_s": 40}}
    conserved_put(root, admission.LEDGER, admitted)
    admitted_raw = resource._text_path(root, admission.LEDGER).read_bytes()
    target = resource._text_path(root, resource.HELPER_SOURCE)
    raw = target.read_bytes()
    target.write_bytes(raw + b"\n")
    try:
        outer = supervisor._supervise(request_name, root=root, entry=time.perf_counter())
        assert outer["refusal_origin"] == "ACTUAL_I_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT"
        assert outer["preflight_recovery"] == f.recovery and outer["qualification_amendment"] == f.before[resource.G_KEY]
        assert outer["first_error"]["type"] == "ValueError" and outer["job_created"] is False and outer["worker_started"] is False
        assert outer["stdout"] is None and outer["stderr"] is None and not outer["stdout_eof_observed"] and not outer["stderr_eof_observed"]
        assert outer["runtime_reservation_state"] == "NEVER_CREATED" and outer["root_admission_reservation_state"] == "STILL_PRESENT_PENDING_NATIVE_COMMIT"
        assert not any(k in outer for k in ("trace", "dispatch_ms", "first_final_ms", "actual_outer_full_return_s"))
        assert resource.raw_pin(outer["initial_ledger"], root, 524288) == admitted_raw
        assert resource.current_caps(admitted, root, supervisor.generated_budget_limit) == resource.NEW_CAPS
    finally:
        target.write_bytes(raw)
    trace_name = "generated-I-failed-trace.json"
    conserved_put(root, trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
        "dispatch_ms": 1791420000000, "first_final_ms": 1791420000125, "full_return_s": 0.125, "exit_code": 1,
        "source_freeze": f.source, "request": request, "fixture_only": True})
    account_text = (admission.ROOT / admission.BASE / "parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert account_text.count(anchor) == 1
    account_text = account_text.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + root.as_posix().replace("'", "''") + "')")
    resource.fixture_text_write(root, "generated-I-account.ps1", account_text.encode("utf8"))
    resource.fixture_text_write(root, admission.LEDGER, f.before_raw)
    return conserved_put(root, "generated-I-cases.json", {"before": conserved_put(root, "generated-I-before.json", f.before),
        "before_raw": resource.I_CLOSED_PIN, "after": conserved_put(root, "generated-I-after.json", f.after), "recovery": f.recovery, "source": f.source,
        "states": states, "negatives": negatives, "qualified_state": conserved_put(root, "generated-I-qualified.json", qualified_live),
        "qualification": qualified, "qualification_negatives": conserved_g_qualification_negatives(qualified) + [dict(qualified, preflight_recovery=f.before[resource.G_KEY])],
        "admitted_raw_b64": __import__("base64").b64encode(admitted_raw).decode("ascii"), "q": q, "request": request,
        "trace_name": trace_name, "critical_sources": list(resource.G_REPLACEMENTS), "fixture_only": True})


def stage_record(p):
    r.fixture_text_write(p,r.IMMUTABLE_RECORD_PIN["path"],r.raw_pin(r.IMMUTABLE_RECORD_PIN,admission.ROOT,262144))

def j_fixture(p):
    conserved_i_metadata_fixture(p)
    m=r.j_archives(admission.ROOT)
    for item in m.values():
        r.fixture_text_write(p,item["archive"]["path"],r.raw_pin(item["archive"],admission.ROOT,524288,allow_empty=True))
        if item["source"]["path"] not in r.G_REPLACEMENTS:
            r.fixture_text_write(p,item["source"]["path"],r.raw_pin(item["source"],admission.ROOT,524288,allow_empty=True))
    rv=r.text_spec(r.J_REVIEW_PATH,admission.ROOT,1048576)
    h=r.j_history(admission.ROOT,s.generated_budget_limit)[3]
    v=r.j_review_metadata(rv,admission.ROOT,h,m)
    for spec in v["sources"].values():
        if not Path(spec["path"]).is_absolute():
            conserved_copy(p,spec)
    cl=r.text_spec(r.J_CLEANUP_PATH,admission.ROOT,1048576)
    for spec in (r.J_BEFORE_PIN,r.J_ROOT_PIN,*r.J_ASTRA_PINS.values(),
                 *v["root_reviews"],*v["sol_reports"],rv,cl):
        conserved_copy(p,spec)
    prior=r.read_pin(m[r.I_SOURCE_PATH]["source"],p)
    source=conserved_put(p,r.J_SOURCE_PATH,dict(prior,qualification_epoch=r.J_EPOCH,
        reviewed_source=rv,qualification_lint_cleanup=cl,sources=v["sources"],body_retry=v["sources"][r.J_PATH]))
    assert source["bytes"]<=262144
    raw0=r.raw_pin(r.J_CLOSED_PIN,p,524288)
    before=json.loads(raw0)
    r.fixture_text_write(p,admission.LEDGER,raw0)
    after=r.j_registration(before,cl,p,s.generated_budget_limit,before_raw=raw0)
    return SimpleNamespace(p=p,before=before,raw0=raw0,after=after,cl=cl,rv=rv,v=v,source=source,
        policy=r.read_pin(r.POLICY_PIN,p),grant=r.read_pin(r.J_PIN,p))

def j_request(f,role):
    plan=r.j_generated_plan(role,f.p)
    return {"schema":"flylab.m3.direct-science.supervision.v1","mode":"generated","call_id":plan["call"],"role":plan["role"],
        "source_freeze":f.source,"shared_original_lock_path":admission.SHARED_LOCK,"ledger_path":admission.LEDGER,
        "limits":plan["limits"],"generated_behavior":"command","generated_argv":plan["argv"],"output":plan["output"]}

def j_paid(f,live,role):
    q=j_request(f,role)
    cost=conserved_fake_paid(f.p,live,f.source,q["call_id"],q["role"],q["generated_argv"],
        tuple(q["limits"][k] for k in ("outer_s","child_s","cleanup_s")),name="generated-J-"+role,request=q)
    conserved_recompute(live,f.policy,f.grant,2300,15600)
    return cost

def j_qualified(f):
    live,costs=copy.deepcopy(f.after),[]
    for role in ("lint","math","gate-a","gate-b"):
        costs.append(j_paid(f,live,role))
    roster,suites=r.complete_roster(f.p),[]
    for role in ("math","gate-a","gate-b"):
        rows=roster[role.replace("-","_")]
        spec=conserved_junit(f.p,r.BASE+"/parent-contact-learning-j-"+role+"-junit.xml",rows)
        suites.append({"file":spec,"node_ids":[r["node_id"] for r in rows],"names":[r["name"] for r in rows],
                       "classnames":[r["classname"] for r in rows],"count":len(rows)})
    return live,{"schema":"flylab.m3.direct-science.qualification.v7","epoch":r.J_EPOCH,"old_E360":"UNPASSED",
        "allocation":live[r.ALLOCATION_KEY],"qualification_amendment":live[r.G_KEY],"qualification_lint_repair":f.before[r.H_KEY],"preflight_recovery":f.before[r.I_KEY],"qualification_lint_cleanup":f.cl,
        "reviewed_source":f.rv,"qualification_roster":r.ROSTER_PIN,"source_freeze":f.source,
        "actual_costs":costs,"tests":suites,"evidence":costs}

def j_negative_states(f):
    cases=[]
    for kind in ("extra-pointer","alias","caps","failed-I-refund","old-I-call","old-debit"):
        bad=copy.deepcopy(f.after)
        if kind=="extra-pointer":
            bad[r.J_KEY+"_alias"]=f.cl
        elif kind=="alias":
            bad[r.J_KEY]["path"]+=".alias.json"
        elif kind=="caps":
            bad["direct_science_generated_limit_s"]+=1
        elif kind=="failed-I-refund":
            bad["debits"][286]["actual_outer_full_return_s"]=0.0
        elif kind=="old-I-call":
            bad["direct_science_calls"][r.I_QUAL_CALLS[0]]["state"]="admitted"
        else:
            bad["debits"][0]["cost_id"]+="-changed"
        cases.append((kind,bad))
    for family in "GHIJ":
        bad=copy.deepcopy(f.after)
        del bad[getattr(r,family+"_KEY")]
        cases.append(("missing-"+family+"-pointer",bad))
    return cases

def j_contract_cases(p):
    f=j_fixture(p)
    assert set(f.after) - set(f.before)=={r.J_KEY} and all(f.after[k]==v for k,v in f.before.items())
    assert (len(f.before["debits"]),len(f.before["direct_science_calls"]))==(287,169)
    assert (f.before["debits"][286]["actual_outer_full_return_s"],f.before["debits"][286]["actual_returned_exit_code"])==(20.284,1)
    assert r.current_caps(f.after,p,s.generated_budget_limit)==r.NEW_CAPS
    assert r._qualification_family("j")["envelope"]==sum(p["limits"]["outer_s"] for p in r.j_plan(p)["calls"])+20==575
    m=r.j_archives(p)
    assert m["tools/banc_m3_conserved_resource_native_epoch.ps1"]["source"]["sha256"]=="eff7346fd295f115794f0f705f7cf10164f216a3d40707623f7e72ecba1602ed"
    assert m[r.BASE+"/parent-direct-science-account.ps1"]["source"]["sha256"]=="a434fc7f521cf52f0cfdee66b4a1b124bf9b6529fef2739403ea18a962f64d84"
    old=r.read_pin(m[r.I_REVIEW_PATH]["source"],p)["sources"]
    assert len(f.v["sources"])==728 and set(f.v["sources"])==set(old) | {r.J_PATH,r.IMMUTABLE_RECORD_PIN["path"]}
    assert f.v["sources"][r.IMMUTABLE_RECORD_PIN["path"]]==r.IMMUTABLE_RECORD_PIN
    assert set(f.v["replacements"])==set(r.G_REPLACEMENTS)
    assert sum(f.v["sources"][k]==v for k,v in old.items() if k not in r.G_REPLACEMENTS)==719
    for _kind,bad in j_negative_states(f):
        with pytest.raises((ValueError,KeyError,FileNotFoundError)):
            r.current_caps(bad,p,s.generated_budget_limit)
    raw,descriptor=r.raw_pin(f.cl,p),r.read_pin(f.cl,p)
    try:
        for field,value in (("refund_s",True),("new_resource_grant_s",1),("failed_i_lint",{})):
            bad=conserved_put(p,r.J_CLEANUP_PATH,dict(descriptor,**{field:value}))
            with pytest.raises((ValueError,KeyError)):
                r.j_lint_cleanup(bad,p,s.generated_budget_limit)
    finally:
        r.fixture_text_write(p,r.J_CLEANUP_PATH,raw)
    cost,debit=r.read_pin(f.before["debits"][286]["descriptor"],p),f.before["debits"][286]
    assert admission.cost_proof(cost,debit,lambda pin:r.read_pin(pin,p),generated=True)["call_id"]==r.I_QUAL_CALLS[0]
    with pytest.raises(ValueError):
        admission.cost_proof(cost,debit,lambda pin:r.read_pin(pin,p),generated=True,expected_timeout=True)

def j_qualification_cases(p):
    f=j_fixture(p)
    live=copy.deepcopy(f.after)
    for role in ("lint","math","gate-a","gate-b"):
        q=j_request(f,role)
        pin=conserved_put(p,"generated-J-next.json",q)
        assert r.prequalification_dispatch(live,q,pin,p,s.generated_budget_limit,phase="closed")["epoch"]==r.J_EPOCH
        for bad in (dict(q,call_id=q["call_id"]+"-retry"),dict(q,generated_argv=q["generated_argv"][:2]),
                    dict(q,limits=dict(q["limits"],outer_s=361)),dict(q,limits=dict(q["limits"],child_s=339)),
                    dict(q,limits=dict(q["limits"],cleanup_s=19))):
            bad_pin=conserved_put(p,"generated-J-wrong-next.json",bad)
            with pytest.raises(ValueError):
                r.prequalification_dispatch(live,bad,bad_pin,p,s.generated_budget_limit,phase="closed")
        for role0 in ("input_audit","neutral_settling","saved_startup_transfer","tracking_neutral","live_replay","geometry_capture",*admission.ROLES):
            legacy=dict(q,mode="original",call_id="startup-ray-legacy-blocked",role=role0,
                science_request={"path":"must-not-read-Original.npz","bytes":1,"sha256":"0" * 64})
            with pytest.raises(ValueError,match="global Original"):
                r.prequalification_dispatch(live,legacy,{},p,s.generated_budget_limit,phase="closed")
        j_paid(f,live,role)
    done,qual=j_qualified(f)
    assert r.validate_qualification(qual,done,p,s.generated_budget_limit)==291
    for bad in conserved_g_qualification_negatives(qual)+[dict(qual,preflight_recovery=f.before[r.G_KEY]),dict(qual,qualification_lint_cleanup=f.before[r.I_KEY])]:
        with pytest.raises((ValueError,KeyError,FileNotFoundError)):
            r.validate_qualification(bad,done,p,s.generated_budget_limit)
    for test in qual["tests"]:
        good=r.raw_pin(test["file"],p)
        target=p/test["file"]["path"]
        target.write_bytes(good+b"invalid")
        try:
            with pytest.raises(ValueError):
                r.validate_qualification(qual,done,p,s.generated_budget_limit)
        finally:
            target.write_bytes(good)
    with pytest.raises(ValueError):
        r.j_dispatch(done,q,pin,p,s.generated_budget_limit,phase="closed")

def j_native_cases_file(p):
    import time
    f=j_fixture(p)
    q=j_request(f,"lint")
    name="generated-J-initial-request.json"
    request=conserved_put(p,name,q)
    admitted=copy.deepcopy(f.after)
    admitted["direct_science_calls"][q["call_id"]]={"request":request,"state":"admitted","mode":"generated","reserved_s":40}
    admitted["direct_science_reservations"]={q["call_id"]:{"request":request,"reserved_s":40}}
    conserved_put(p,admission.LEDGER,admitted)
    raw1=(p/admission.LEDGER).read_bytes()
    target=p/r.HELPER_SOURCE
    raw=target.read_bytes()
    target.write_bytes(raw+b"\n")
    try:
        outer=s._supervise(name,root=p,entry=time.perf_counter())
        assert outer["refusal_origin"]=="ACTUAL_J_INITIAL_GUARD_BEFORE_LOCK_JOB_ATTEMPT" and outer["qualification_lint_cleanup"]==f.cl
        assert outer["job_created"] is False and outer["stdout"] is None and outer["stdout_eof_observed"] is False
        assert r.raw_pin(outer["initial_ledger"],p,524288)==raw1
    finally:
        target.write_bytes(raw)
    trace="generated-J-fixture-only-trace.json"
    conserved_put(p,trace,{"schema":"flylab.m3.direct-science.genuine-return.v1","phase":"FINAL","first_error":None,
        "dispatch_ms":1791420000000,"first_final_ms":1791420000125,"full_return_s":0.125,"exit_code":1,
        "source_freeze":f.source,"request":request,"fixture_only":True})
    text=(admission.ROOT/admission.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor="$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    r.need(text.count(anchor)==1,"one fixture p")
    text=text.replace(anchor,"$taskRoot=[IO.Path]::GetFullPath('"+p.as_posix().replace("'","''")+"')")
    r.fixture_text_write(p,"generated-J-account.ps1",text.encode("utf8"))
    eof=p.parent/"generated-J-eof"
    shutil.copytree(p,eof)
    conserved_put(eof,q["output"]+"/outer.json",dict(outer,stdout_eof=False))
    r.fixture_text_write(eof,"generated-J-account.ps1",text.replace(p.as_posix().replace("'","''"),eof.as_posix().replace("'","''")).encode("utf8"))
    r.fixture_text_write(p,admission.LEDGER,f.raw0)
    return conserved_put(p,"generated-J-cases.json",{"before_raw":r.J_CLOSED_PIN,"after":conserved_put(p,"generated-J-after.json",f.after),
        "cleanup":f.cl,"source":f.source,"q":q,"request":request,"trace_name":trace,
        "admitted_raw_b64":__import__("base64").b64encode(raw1).decode("ascii"),"fixture_only":True})
