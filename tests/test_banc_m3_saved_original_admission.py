"""Generated metadata-only unit cases. R1 SOURCE PROPOSAL: NOT_RUN by Sol.

These fixtures replace grant/roster SHA constants and native identity functions;
they cannot admit Original data, certify Windows closure, or claim science PASS.
ROOT runs only against the promoted canonical helper under its finite test role.
"""
import hashlib
import json
import os
import sys
import time
import types

import pytest

import tools
from tools import banc_m3_saved_original_admission as saved


def put(root, name, value, size=None):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, allow_nan=False).encode("utf8")
    if size is not None:
        assert len(raw) <= size
        raw += b" " * (size - len(raw))
    p.write_bytes(raw)
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Every byte is newly generated test metadata; no saved Original input is read.
    root = tmp_path
    base = saved.BASE
    owner_entry = time.perf_counter()
    auth = {"path": base + "/generated-authority.json", "bytes": 1, "sha256": "a" * 64}
    old_source = {"path": base + "/generated-old-source.json", "bytes": 1, "sha256": "b" * 64}
    old_inputs = {"path": base + "/generated-old-inputs.json", "bytes": 1, "sha256": "c" * 64}
    grant = {"schema": saved.SCHEMA + ".original-grant.v1", "mode": "original",
             "grant_id": "startup-ray-original-20261007-j", "additional_s": 6000,
             "global_limit_s": 16200, "stage_limit_s": 6300, "science_local_limit_s": 6000,
             "worker_max": 1, "automatic_restarts": 0, "reset": False,
             "resource_increment_s": 0, "local_debit_offset": 165,
             "original_cumulative_before_s": 92.0, "original_stage_before_s": 12.0,
             "science_local_used_before_s": 0.0,
             "human_authorization": auth, "source_freeze": old_source, "inputs": old_inputs}
    grant_spec = put(root, saved.GRANT_PATH, grant, 2227)
    monkeypatch.setattr(saved, "GRANT_SHA", grant_spec["sha256"])
    roster_spec = put(root, saved.ROSTER_PATH,
                      {"schema": "flylab.m3.after-startup-ray.saved-roster.v1", "no_new_body_work": True}, 4377)
    monkeypatch.setattr(saved, "ROSTER_SHA", roster_spec["sha256"])
    source_specs = {}
    for name in saved.REQUIRED_SOURCES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")  # Empty package-style Source permitted only in Source fence.
        source_specs[name] = {"path": name, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    freeze_spec = put(root, base + "/generated-freeze.json",
                      {"schema": saved.SCHEMA + ".source-freeze.v1", "ROOT_review_cycle_complete": True,
                       "human_authorization": auth, "sources": source_specs})
    debits = [{"mode": "generated", "cost_id": "generated-prefix-" + str(i)} for i in range(165)]
    calls = {}
    for i, role in enumerate(saved.OLD_ROLES):
        call = "startup-ray-generated-j-" + role
        dummy_request = {"path": base + "/generated-old-request-" + str(i) + ".json",
                         "bytes": 1, "sha256": "d" * 64}
        outer = {"request": dummy_request, "exit_code": 0, "all_owned_ended": True,
                 "reservation_released": True, "stdout_eof": True, "stderr_eof": True,
                 "unknown_owned_identities": {}}
        outer_spec = put(root, base + "/generated-old-outer-" + str(i) + ".json", outer)
        trace_spec = put(root, base + "/generated-old-trace-" + str(i) + ".json",
                         {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL",
                          "first_error": None, "dispatch_ms": 1000, "first_final_ms": 2000,
                          "full_return_s": 1.0, "exit_code": 0, "source_freeze": old_source,
                          "request": dummy_request})
        cost = {"schema": saved.SCHEMA + ".full-return.v1", "cost_id": call, "role": role,
                "charged_once": True, "actual_outer_full_return_s": 1.0, "exit_code": 0,
                "all_owned_ended": True, "reservation_released": True, "outer": outer_spec,
                "trace": trace_spec, "supervisor_request": dummy_request, "grant": grant_spec,
                "source_freeze": old_source, "inputs": old_inputs, "local_debit_offset": 165}
        spec = put(root, base + "/generated-old-cost-" + str(i) + ".json", cost)
        debits.append({"mode": "original", "cost_id": call, "descriptor": spec,
                       "actual_outer_full_return_s": 1.0, "actual_returned_exit_code": 0})
        calls[call] = {"request": dummy_request, "state": "charged"}
    ledger = {"debits": debits, "direct_science_calls": calls,
              "original_cumulative_used_s": 100.0, "original_stage_used_s": 20.0,
              "startup_ray_original_local_used_s": 8.0, "original_remaining_s": 16100.0,
              "original_cumulative_limit_s": 16200, "original_stage_cap_s": 6300,
              "startup_ray_original_local_limit_s": 6000, "original_jobs_running": 0,
              "original_reserved_s": 0,
              "startup_ray_original_grants": {grant["grant_id"]: {
                  "grant": grant_spec, "source_freeze": old_source, "inputs": old_inputs,
                  "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}}}
    for key in ("startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds", "e360_pending_actuals",
                "e360_blocking_holds", "direct_science_pending_actuals", "direct_science_blocking_holds",
                "e359_reservations", "e360_reservations", "direct_science_reservations", "direct_science_workers",
                "startup_ray_original_workers", "startup_ray_original_reservations"):
        ledger[key] = {}
    # A paid generated qualification tail is preserved, never counted as Original.
    ledger["debits"].append({"mode": "generated", "cost_id": "generated-qualified-tail"})
    before_spec = put(root, base + "/generated-before.json", ledger)
    call = "startup-ray-generated-saved-case"
    q = {"schema": saved.SCHEMA + ".saved-diagnosis.request.v1", "mode": "original", "role": saved.ROLE,
         "call_id": call, "limits": dict(saved.LIMITS), "ledger_path": saved.LEDGER,
         "shared_original_lock_path": saved.SHARED_LOCK, "resource_increment_s": 0,
         "local_debit_offset": 165, "grant": grant_spec, "roster": roster_spec,
         "source_freeze": freeze_spec, "before_ledger": before_spec,
         "output": base + "/parent-" + call + "/diagnosis"}
    request_name = base + "/generated-request.json"
    request_spec = put(root, request_name, q)
    outer = {"schema": "flylab.m3.direct-science.supervision.v1", "role": saved.ROLE,
             "mode": "original", "call_id": call, "science_request": request_spec,
             "grant": grant_spec, "source_freeze": freeze_spec, "output": q["output"] + "-supervision",
             "ledger_path": saved.LEDGER, "shared_original_lock_path": saved.SHARED_LOCK,
             "limits": {key: saved.LIMITS[key] for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
    outer_spec = put(root, base + "/generated-outer-request.json", outer)
    ledger["direct_science_calls"][call] = {"request": outer_spec, "state": "admitted", "mode": "original", "reserved_s": 120}
    put(root, saved.LEDGER, ledger)
    pid = os.getpid()
    native = types.SimpleNamespace(creation_ticks=lambda actual: 111 if actual == pid else 222,
                                   in_named_job=lambda name: name == "Local\\flylab-startup-ray-generated-case")
    monkeypatch.setitem(sys.modules, "tools.banc_m3_startup_ray_supervisor", native)
    monkeypatch.setattr(tools, "banc_m3_startup_ray_supervisor", native, raising=False)
    return types.SimpleNamespace(root=root, q=q, ledger=ledger, request_name=request_name,
                                 request_spec=request_spec, outer_spec=outer_spec, entry=owner_entry,
                                 native=native, monkeypatch=monkeypatch)


def admit(case, stage="pre_spawn", entry=None):
    return saved.Admission(case.request_name, stage, root=case.root, entry=case.entry if entry is None else entry)


def own(case):
    call = case.q["call_id"]
    token = {"schema": "flylab.m3.direct-science.lock-owner.v1",
             "call_id": call, "request": case.outer_spec, "owner_pid": os.getpid() + 1,
             "owner_creation_ticks": 222, "entry_counter_s": case.entry,
             "job_name": "Local\\flylab-startup-ray-generated-case"}
    spec = put(case.root, saved.SHARED_LOCK, token)
    case.monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps(token))
    case.ledger["startup_ray_original_workers"] = {call: {
        "request": case.request_spec, "grant": case.q["grant"], "state": "begun", "worker_max": 1,
        "reserved_s": 120, "lock_token": spec, "owner_pid": token["owner_pid"], "owner_creation_ticks": 222,
        "worker_pid": os.getpid(), "worker_creation_ticks": 111}}
    case.ledger["startup_ray_original_reservations"] = {call: {"request": case.request_spec, "grant": case.q["grant"], "reserved_s": 120}}
    case.ledger["original_jobs_running"], case.ledger["original_reserved_s"] = 1, 120
    put(case.root, saved.LEDGER, case.ledger)
    (case.root / (case.q["output"] + "-supervision")).mkdir(parents=True)


def test_fresh_generated_metadata_baseline_and_source_only(case):
    gate = admit(case)
    assert gate.check("again") is True
    assert gate.roster_path == case.root / saved.ROSTER_PATH
    assert len(gate.before["debits"]) == 174


def test_actual_owner_clock_is_deadline_origin(case):
    own(case)
    gate = admit(case, "pre_array", time.perf_counter())
    assert gate.entry == case.entry
    assert gate.caller_entry >= gate.entry
    assert gate.check("selected-member") is True


def test_default_pre_array_entry_infers_same_owner(case):
    own(case)
    gate = saved.Admission(case.request_name, "pre_array", root=case.root)
    assert gate.entry == case.entry


@pytest.mark.parametrize("key", ["direct_science_pending_actuals", "startup_ray_original_blocking_holds", "e360_reservations"])
def test_pending_or_unrelated_ownership_refuses(case, key):
    case.ledger[key]["unclosed"] = {}
    put(case.root, saved.LEDGER, case.ledger)
    with pytest.raises(ValueError):
        admit(case)


def test_every_debit_including_generated_tail_preserved(case):
    case.ledger["debits"][-1]["cost_id"] = "changed"
    put(case.root, saved.LEDGER, case.ledger)
    with pytest.raises(ValueError, match="all debits"):
        admit(case)


def test_usage_drift_refuses_without_rounding(case):
    case.ledger["original_cumulative_used_s"] += 1e-10
    put(case.root, saved.LEDGER, case.ledger)
    with pytest.raises(ValueError, match="pinned usage"):
        admit(case)


def test_source_change_refuses_before_numeric(case):
    gate = admit(case)
    (case.root / saved.REQUIRED_SOURCES[0]).write_bytes(b"# drift\n")
    with pytest.raises(ValueError):
        gate.check("pre_array")


def test_duplicate_output_refuses_before_spawn(case):
    (case.root / case.q["output"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh closed"):
        admit(case)


def test_shared_original_lock_blocks_pre_spawn(case):
    put(case.root, saved.SHARED_LOCK, {"foreign": True})
    with pytest.raises(ValueError, match="fresh closed"):
        admit(case)


@pytest.mark.parametrize("fault", ["ticks", "job", "environment", "pid"])
def test_pre_array_exact_native_lease_refusal(case, fault):
    own(case)
    gate = admit(case, "pre_array")
    if fault == "ticks":
        case.native.creation_ticks = lambda pid: 999
    elif fault == "job":
        case.native.in_named_job = lambda name: False
    elif fault == "environment":
        case.monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", "{}")
    else:
        case.ledger["startup_ray_original_workers"][case.q["call_id"]]["worker_pid"] += 1000
        put(case.root, saved.LEDGER, case.ledger)
    with pytest.raises(ValueError):
        gate.check("pre_array")


def test_owning_entry_deadline_cannot_restart_in_worker(case):
    own(case)
    case.monkeypatch.setattr(saved.time, "perf_counter", lambda: case.entry + 100.0)
    def forbidden_read(*args, **kwargs):
        raise AssertionError("deadline must refuse before request or Source reads")
    case.monkeypatch.setattr(saved, "spec_for", forbidden_read)
    with pytest.raises(TimeoutError):
        admit(case, "pre_array")


def test_hash_callback_reads_no_metadata_or_sources(case):
    gate = admit(case)
    def forbidden_read(*args, **kwargs):
        raise AssertionError("ordinary callback must be deadline-only")
    case.monkeypatch.setattr(saved, "leaf", forbidden_read)
    case.monkeypatch.setattr(saved, "sources", forbidden_read)
    case.monkeypatch.setattr(saved, "spec_for", forbidden_read)
    assert gate.check("hash-block") is True


def test_publication_is_a_fresh_cold_fence(case):
    own(case)
    gate = admit(case, "pre_array")
    (case.root / case.q["output"]).mkdir(parents=True)
    assert gate.check("publication") is True
    (case.root / saved.REQUIRED_SOURCES[0]).write_bytes(b"# changed before publication\n")
    with pytest.raises(ValueError):
        gate.check("publication")


def test_json_duplicate_or_nonfinite_refuses():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e1000}', b'{"x":-1e1000}'):
        with pytest.raises(ValueError):
            saved.json_load(raw)


def test_empty_metadata_is_not_source(case):
    spec = {"path": saved.REQUIRED_SOURCES[0], "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    with pytest.raises(ValueError):
        saved.leaf(spec, case.root)
