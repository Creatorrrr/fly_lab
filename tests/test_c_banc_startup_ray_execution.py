"""Finite generated observations of real gate/supervisor; no Original array entry.

ROOT runs this file through the new paid generated command supervisor. Scratch
metadata is explicitly synthetic and never grants the real project's Original.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import banc_m3_startup_ray_supervisor as supervisor
from tools import verify_c_banc_m3_startup_ray as ray

pytestmark = pytest.mark.skipif(os.name != "nt", reason="actual Windows Job/identity required")


def put(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, allow_nan=False).encode("utf8")
    p.write_bytes(raw)
    return {"path": name, "bytes": len(raw), "sha256": ray.hashlib.sha256(raw).hexdigest()}


def install_source(root, helpers=False):
    sources = {}
    names = [ray.SOURCE, "tools/banc_m3_startup_ray_supervisor.py", "tools/banc_m3_contact_runtime_journal.py",
             "tools/banc_m3_conserved_resource.py"]
    if helpers:
        names += list(ray.HELPER_SOURCES)
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ray.ROOT / name, p)
        sources[name] = ray.spec_for(name, root, cap=2097152)
    return sources


def gate_fixture(root):
    sources = install_source(root, helpers=True)
    proof = put(root, "synthetic-review.json", {"schema": "explicit.generated.synthetic-review.v1"})
    freeze = put(root, "source-freeze.json", {"schema": ray.SCHEMA + ".source-freeze.v1",
                 "ROOT_review_cycle_complete": True, "sources": sources,
                 "actual_ROOT_FIRST": proof, "actual_ROOT_SECOND": proof, "actual_Sol_FIRST": proof,
                 "human_authorization": proof})
    inputs = put(root, "synthetic-inputs.json", {"schema": "explicit.generated.no-original-inputs.v1"})
    before = put(root, "synthetic-before.json", {"debits": [], "original_cumulative_used_s": 10020.115960899504,
                                               "original_stage_used_s": 126.361})
    grant = put(root, "synthetic-grant.json", {"schema": ray.SCHEMA + ".original-grant.v1", "grant_id": "synthetic",
                "mode": "original", "additional_s": 6000, "global_limit_s": 16200, "stage_limit_s": 6300,
                "science_local_limit_s": 6000, "worker_max": 1, "automatic_restarts": 0, "reset": False,
                "source_freeze": freeze, "inputs": inputs, "local_debit_offset": 0, "human_authorization": proof,
                "before_ledger": before, "original_cumulative_before_s": 10020.115960899504,
                "original_stage_before_s": 126.361})
    paths, role_limits = {}, {}
    attempt = "startup-ray-synthetic"
    for i, role in enumerate(ray.ROLES):
        paths[role] = {"request": supervisor.BASE + "/parent-" + attempt + "-" + role + "-request.json",
                       "output": supervisor.BASE + "/parent-" + attempt + "/" + role}
        role_limits[role] = {"outer_s": ray.OUTER[i], "child_s": ray.CHILD[i], "internal_s": ray.INTERNAL[i],
                            "host_bytes": 2147483648, "output_cap_bytes": 16777216,
                            "sample_cap": 15000, "GPU_delta_plus_allowance_cap_bytes": 2147483648}
    ledger_name = supervisor.BASE + "/synthetic-ledger.json"
    manifest = put(root, "synthetic-manifest.json", {"schema": ray.SCHEMA + ".manifest.v1", "attempt_id": attempt,
                "alphas": list(ray.ALPHAS), "roles": list(ray.ROLES), "role_paths": paths, "role_limits": role_limits,
                "grant": grant, "source_freeze": freeze, "inputs": inputs, "local_debit_offset": 0,
                "ledger_path": ledger_name, "shared_original_lock_path": supervisor.SHARED_LOCK})
    q = {"schema": ray.SCHEMA + ".request.v1", "role": "input_audit", "mode": "original",
         "call_id": attempt + "-input_audit", "manifest": manifest, "grant": grant, "source_freeze": freeze,
         "inputs": inputs, "local_debit_offset": 0, "output": paths["input_audit"]["output"],
         "limits": role_limits["input_audit"], "artifacts": {}}
    spec = put(root, paths["input_audit"]["request"], q)
    ledger = {"debits": [], "startup_ray_original_grants": {"synthetic": {"grant": grant, "source_freeze": freeze,
              "inputs": inputs, "local_debit_offset": 0, "worker_max": 1, "automatic_restarts": 0}},
              "startup_ray_original_workers": {}, "startup_ray_original_reservations": {},
              "startup_ray_original_pending_actuals": {}, "startup_ray_original_blocking_holds": {},
              "e360_pending_actuals": {}, "e360_blocking_holds": {}, "e359_reservations": {}, "e360_reservations": {},
              "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {}, "direct_science_calls": {},
              "original_jobs_running": 0, "original_reserved_s": 0, "original_cumulative_used_s": 10020.115960899504,
              "original_stage_used_s": 126.361, "startup_ray_original_local_used_s": 0.0,
              "original_cumulative_limit_s": 16200, "original_stage_cap_s": 6300,
              "startup_ray_original_local_limit_s": 6000, "original_remaining_s": 16200 - 10020.115960899504 - 0.0}
    put(root, ledger_name, ledger)
    return q, spec, ledger, ledger_name


def test_real_gate_accepts_synthetic_metadata_and_rejects_finite_tampering_before_arrays(tmp_path):
    q, spec, _, _ = gate_fixture(tmp_path)
    admitted = ray.Admission(spec["path"], "pre_spawn", root=tmp_path, entry=time.perf_counter())
    assert admitted.check() is True
    for key, value in (("mode", "generated"), ("role", "unknown"), ("local_debit_offset", True),
                       ("local_debit_offset", 1), ("source_freeze", {"path": "../../foreign", "bytes": 1, "sha256": "0" * 64}),
                       ("inputs", None), ("grant", None)):
        bad = copy.deepcopy(q)
        bad[key] = value
        put(tmp_path, spec["path"], bad)
        with pytest.raises((ValueError, TypeError, KeyError)):
            ray.Admission(spec["path"], "pre_spawn", root=tmp_path, entry=time.perf_counter())
    put(tmp_path, spec["path"], q)
    for value in (True, float("inf")):
        with pytest.raises(ValueError):
            ray.finite_float(value, "actual field")
    # No numerical/Original loader was called by these actual production gates.
    assert not (tmp_path / q["output"]).exists()


def test_real_gate_cold_source_and_input_drift_and_unknown_pending(tmp_path):
    q, spec, ledger, ledger_name = gate_fixture(tmp_path)
    gate = ray.Admission(spec["path"], "pre_spawn", root=tmp_path, entry=time.perf_counter())
    source = tmp_path / ray.SOURCE
    original = source.read_bytes()
    source.write_bytes(original + b"\n# drift\n")
    with pytest.raises(ValueError):
        gate.fence()
    source.write_bytes(original)
    inputs = tmp_path / q["inputs"]["path"]
    raw = inputs.read_bytes()
    inputs.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        gate.fence()
    inputs.write_bytes(raw)
    for field in ("startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds"):
        bad = copy.deepcopy(ledger)
        bad[field] = {"unclosed": {"request": spec}}
        put(tmp_path, ledger_name, bad)
        with pytest.raises(ValueError, match="pending/unknown"):
            gate.fence()
    put(tmp_path, ledger_name, ledger)
    assert gate.fence() is True


def test_real_prearray_identity_and_same_entry_cold_check_inside_actual_job(tmp_path):
    q, spec, ledger, ledger_name = gate_fixture(tmp_path)
    token = json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])
    assert supervisor.in_named_job(token["job_name"])
    outer = put(tmp_path, "synthetic-supervisor-request.json", {"science_request": spec})
    token.update(call_id=q["call_id"], request=outer)
    lock = supervisor.SharedLock(tmp_path / supervisor.SHARED_LOCK, token)
    lock_spec = ray.spec_for(supervisor.SHARED_LOCK, tmp_path)
    identity = {"owner_pid": token["owner_pid"], "owner_creation_ticks": token["owner_creation_ticks"],
                "worker_pid": os.getpid(), "worker_creation_ticks": supervisor.creation_ticks(os.getpid())}
    ledger["direct_science_calls"][q["call_id"]] = {"request": outer, "state": "admitted", "mode": "original", "reserved_s": 120}
    ledger["startup_ray_original_workers"][q["call_id"]] = {"request": spec, "grant": q["grant"], "state": "begun",
                "worker_max": 1, "reserved_s": 120, "lock_token": lock_spec, **identity}
    ledger["startup_ray_original_reservations"][q["call_id"]] = {"request": spec, "grant": q["grant"], "reserved_s": 120}
    ledger["original_jobs_running"], ledger["original_reserved_s"] = 1, 120
    put(tmp_path, ledger_name, ledger)
    try:
        gate = ray.Admission(spec["path"], "pre_array", root=tmp_path, entry=token["entry_counter_s"])
        assert gate.fence() is True
        bad = copy.deepcopy(ledger)
        bad["startup_ray_original_workers"][q["call_id"]]["worker_creation_ticks"] += 1
        put(tmp_path, ledger_name, bad)
        with pytest.raises(ValueError, match="creation"):
            gate.fence()
        put(tmp_path, ledger_name, ledger)
        p = tmp_path / q["inputs"]["path"]
        p.write_bytes(p.read_bytes() + b" ")
        with pytest.raises(ValueError):
            gate.fence()
    finally:
        # This synthetic lock has no child to close; it is never the real ROOT lock.
        lock.release(True)


def process_request(root, behavior):
    sources = install_source(root)
    freeze = put(root, "source-freeze.json", {"ROOT_review_cycle_complete": True, "sources": sources})
    call = "generated-" + behavior.replace("-", "")
    q = {"schema": supervisor.SCHEMA + ".supervision.v1", "mode": "generated", "call_id": call, "role": "process",
         "source_freeze": freeze, "shared_original_lock_path": supervisor.SHARED_LOCK,
         "ledger_path": supervisor.BASE + "/synthetic-ledger.json", "output": supervisor.BASE + "/" + call,
         "limits": {"outer_s": 5, "child_s": 2, "cleanup_s": 3, "host_bytes": 268435456, "output_cap_bytes": 262144},
         "generated_behavior": behavior}
    spec = put(root, supervisor.BASE + "/" + call + "-request.json", q)
    ledger = {"direct_science_calls": {call: {"request": spec, "state": "admitted", "mode": "generated", "reserved_s": 5}},
              "direct_science_reservations": {call: {"request": spec, "reserved_s": 5}},
              "direct_science_workers": {}, "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {},
              "direct_science_generated_used_s": 0.0, "direct_science_generated_limit_s": 300,
              "generated_shared_cumulative_used_s": 728.2120156000001, "generated_shared_cumulative_limit_s": 1700,
              "generated_shared_remaining_s": 1700.0 - 728.2120156000001, "debits": []}
    put(root, q["ledger_path"], ledger)
    return q, spec


@pytest.mark.parametrize("behavior,exit_code", [("normal", 0), ("nonzero", 7), ("timeout-descendant", 124)])
def test_actual_supervisor_normal_nonzero_timeout_descendant_and_both_eof(tmp_path, behavior, exit_code):
    q, spec = process_request(tmp_path, behavior)
    actual = supervisor.supervise(spec["path"], root=tmp_path, entry=time.perf_counter())
    assert actual["exit_code"] == exit_code and actual["all_owned_ended"] and actual["reservation_released"]
    assert actual["stdout_eof"] and actual["stderr_eof"] and not actual["unknown_owned_identities"]
    assert type(actual["worker_creation_ticks"]) is int and actual["worker_creation_ticks"] > 10**16
    worker_identity = json.loads((tmp_path / actual["stdout"]["path"]).read_bytes().splitlines()[0])
    assert worker_identity["actual_worker_pid"] == actual["worker_pid"]
    assert Path(worker_identity["python_prefix"]).resolve() == Path(sys.prefix).resolve()
    assert not (tmp_path / supervisor.SHARED_LOCK).exists()
    ledger = ray.json_load((tmp_path / q["ledger_path"]).read_bytes())
    assert q["call_id"] in ledger["direct_science_pending_actuals"]
    assert not ledger["direct_science_reservations"] and not ledger["direct_science_workers"] and ledger["debits"] == []
    if behavior == "timeout-descendant":
        assert actual["timeout"] and actual["first_error"]["type"] == "TimeoutError"
        assert b"descendant_pid" in (tmp_path / actual["stdout"]["path"]).read_bytes()
        from tools.banc_m3_contact_runtime_journal import MAX_JSON_BYTES, PREFIX
        lines = (tmp_path / actual["stderr"]["path"]).read_bytes().splitlines()
        snapshots = [json.loads(line[len(PREFIX):]) for line in lines if line.startswith(PREFIX.encode("ascii"))]
        assert snapshots and all(len(line) <= MAX_JSON_BYTES for line in lines if line.startswith(PREFIX.encode("ascii")))
        last = snapshots[-1]
        assert last["identity"]["call_id"] == q["call_id"] and last["identity"]["source_freeze"] == q["source_freeze"]
        assert last["identity"]["science_request"] == spec
        assert last["last_phase"]["phase"] == "construct" and last["last_phase"]["kind"] == "START"
        assert last["active"][-1]["state"] == "NOT_RETURNED" and last["work"] is None
        assert last["run_valid"] is False and last["diagnostic_only"] is True
    with pytest.raises(ValueError):
        supervisor.supervise(spec["path"], root=tmp_path, entry=time.perf_counter())
    assert ray.json_load((tmp_path / q["ledger_path"]).read_bytes())["debits"] == []


def test_existing_real_shared_lock_competition_preserves_foreign_owner():
    token = json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])
    target = ray.ROOT / supervisor.SHARED_LOCK
    raw = target.read_bytes()
    assert json.loads(raw)["owner_pid"] == token["owner_pid"]
    assert supervisor.creation_ticks(token["owner_pid"]) == token["owner_creation_ticks"]
    with pytest.raises(FileExistsError):
        supervisor.SharedLock(target, {"call_id": "competing-generated-observation"})
    assert target.read_bytes() == raw


def test_scratch_supervisor_lock_contention_has_no_worker_or_debit(tmp_path):
    q, spec = process_request(tmp_path, "normal")
    foreign = supervisor.SharedLock(tmp_path / supervisor.SHARED_LOCK, {"call_id": "existing-other-owner"})
    try:
        observed = supervisor.supervise(spec["path"], root=tmp_path, entry=time.perf_counter())
        assert observed["first_error"]["type"] == "FileExistsError" and observed["worker_pid"] is None
        assert not observed["reservation_released"] and not observed["all_owned_ended"]
        assert foreign.target.read_bytes() == foreign.raw
        assert ray.json_load((tmp_path / q["ledger_path"]).read_bytes())["debits"] == []
    finally:
        foreign.release(True)


def test_actual_native_once_account_rejects_already_charged_genuine_return(tmp_path):
    """Exercise the real writer on an actual earlier charged return, inside this paid job."""
    trace = os.environ["FLYLAB_DIRECT_SCIENCE_DUPLICATE_TRACE"]
    ledger_path = ray.ROOT / supervisor.BASE / "parent-e358-execution-ledger.json"
    before = ledger_path.read_bytes()
    assert len(before) <= 524288
    live = ray.json_load(before)
    foreign_lock_path = ray.ROOT/supervisor.SHARED_LOCK
    foreign_before = foreign_lock_path.read_bytes()
    # The real paid parent still owns its live exclusion. Invoke the actual writer
    # on exact genuine metadata copied to generated scratch, with only its root
    # rebound; it must acquire its own token before load and reach the old duplicate
    # assertion without reusing/removing the live parent's token.
    scratch_ledger = tmp_path/supervisor.BASE/"parent-e358-execution-ledger.json"
    scratch_ledger.parent.mkdir(parents=True, exist_ok=True)
    scratch_ledger.write_bytes(before)
    def copy_metadata(spec, cap=1048576):
        assert spec["path"].endswith(".json")
        raw = ray.leaf(spec, ray.ROOT, cap=cap)
        target = tmp_path/spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        assert target.read_bytes() == raw
        return ray.json_load(raw)
    policy = copy_metadata(live["direct_science_policy"])
    copy_metadata(policy["before_ledger"], 524288)
    genuine_trace = copy_metadata(ray.spec_for(trace, ray.ROOT, cap=2097152), 2097152)
    genuine_request = copy_metadata(genuine_trace["request"])
    assert live["direct_science_calls"][genuine_request["call_id"]]["state"] == "charged"
    assert any(d["cost_id"] == genuine_request["call_id"] for d in live["debits"])
    account_name = supervisor.BASE + "/parent-direct-science-account.ps1"
    account_source = (ray.ROOT/account_name).read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert account_source.count(anchor) == 1
    scratch_account = tmp_path/account_name
    scratch_account.write_text(account_source.replace(
        anchor, "$taskRoot=[IO.Path]::GetFullPath('" + tmp_path.as_posix().replace("'", "''") + "')"), encoding="utf8")
    # Components remain available at their production paths in this generated root.
    for name in ("tools/banc_m3_conserved_resource.py", "tools/banc_m3_conserved_resource_native_core.ps1",
                 "tools/banc_m3_conserved_resource_native_epoch.ps1"):
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ray.ROOT/name, target)
    observation = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(scratch_account),
         "-Action", "commit", "-TracePath", trace],
        cwd=tmp_path, capture_output=True, timeout=10, check=False,
        creationflags=0x08000000,
    )
    assert observation.returncode == 1
    assert b"Fresh once commit; duplicate debit rejected" in observation.stdout + observation.stderr
    assert ledger_path.read_bytes() == before
    assert scratch_ledger.read_bytes() == before and foreign_lock_path.read_bytes() == foreign_before
    assert not (tmp_path/supervisor.SHARED_LOCK).exists()
