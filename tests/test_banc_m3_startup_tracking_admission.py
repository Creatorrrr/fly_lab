"""Generated metadata/admission fixtures; no Original array, neural, or body work."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import banc_m3_startup_tracking_admission as tracking


def history(role="tracking_neutral"):
    values, debits, calls, captured = {}, [], [], []

    def pin(name, value):
        descriptor = {"path": name, "bytes": 1, "sha256": "0" * 64}
        values[name] = value
        return descriptor

    grant = {"source_freeze": {"historical": True}, "inputs": tracking.INPUTS,
             "original_cumulative_before_s": 1000., "original_stage_before_s": 100., "science_local_used_before_s": 371.257}
    for i in range(165):
        debits.append({"cost_id": f"old-{i}", "mode": "generated"})
    for i, name in enumerate(tracking.OLD_ROLES):
        cost = {"schema": "flylab.m3.startup-ray.full-return.v1", "cost_id": f"J-{i}", "mode": "original", "role": name,
                "exit_code": 0, "source_freeze": grant["source_freeze"], "inputs": grant["inputs"],
                "grant": tracking.GRANT, "local_debit_offset": 165, "actual_outer_full_return_s": 1.}
        debits.append({"cost_id": cost["cost_id"], "mode": "original", "descriptor": pin(cost["cost_id"], cost)})
    for i in range(1, 4):
        cost = {"schema": "flylab.m3.startup-ray.saved-diagnosis.full-return.v1", "cost_id": f"startup-ray-20261007-j-saved-transfer-{i:02d}",
                "mode": "original", "exit_code": 0 if i == 3 else 1, "grant": tracking.GRANT, "local_debit_offset": 165,
                "actual_outer_full_return_s": 1., "pre_array_refused": i == 1, "original_payload_accesses": 0 if i == 1 else None}
        debits.append({"cost_id": cost["cost_id"], "mode": "original", "descriptor": pin(cost["cost_id"], cost)})
    q = {"role": role, "grant": tracking.GRANT, "source_freeze": {"current": True}, "prior_captures": {}}
    for i in range(tracking.ROLES.index(role)):
        cost = {"schema": tracking.SCHEMA + ".full-return.v1", "cost_id": tracking.CALLS[i], "mode": "original", "role": tracking.ROLES[i],
                "exit_code": 0, "cap_exceeded": False, "grant": tracking.GRANT, "inputs": tracking.INPUTS, "roster": tracking.ROSTER,
                "source_freeze": q["source_freeze"], "local_debit_offset": 165, "actual_outer_full_return_s": 1., "request": {"role": tracking.ROLES[i]}}
        debits.append({"cost_id": cost["cost_id"], "mode": "original", "descriptor": pin(cost["cost_id"], cost)})
        q["prior_captures"][tracking.ROLES[i]] = pin("capture-" + str(i), {"schema": tracking.SCHEMA + ".capture-manifest.v1", "role": tracking.ROLES[i],
                                                                   "run_valid": True, "capture_complete": True, "request": cost["request"]})
    for i in range(25):
        cost = {"schema": "flylab.m3.direct-science.full-return.v1", "cost_id": f"generated-{i}", "exit_code": 0, "charged_once": True,
                "actual_outer_full_return_s": 1.}
        debits.append({"cost_id": cost["cost_id"], "mode": "generated", "descriptor": pin(cost["cost_id"], cost),
                       "actual_outer_full_return_s": 1., "actual_returned_exit_code": 0})
    total = float(11 + tracking.ROLES.index(role))
    before = {"debits": debits, "original_cumulative_used_s": grant["original_cumulative_before_s"] + total,
              "original_stage_used_s": grant["original_stage_before_s"] + total,
              "startup_ray_original_local_used_s": grant["science_local_used_before_s"] + total,
              "original_remaining_s": 16200. - grant["original_cumulative_before_s"] - total}

    def closed(cost, debit):
        calls.append(cost["cost_id"])
        return cost

    def read(spec):
        captured.append(spec["path"])
        return values[spec["path"]]

    return before, grant, {"original_inputs": tracking.INPUTS}, q, read, closed, values, calls, captured


@pytest.mark.parametrize("role", tracking.ROLES)
def test_history_keeps_J8_both_saved_failures_success03_and_every_generated_tail(role):
    fixture = history(role)
    original = copy.deepcopy(fixture[0])
    total = tracking.validate_history(*fixture[:6])
    assert total == 11 + tracking.ROLES.index(role)
    assert fixture[0] == original
    assert len(fixture[7]) == 11 + tracking.ROLES.index(role)
    assert "startup-ray-20261007-j-saved-transfer-01" in fixture[7]
    assert "startup-ray-20261007-j-saved-transfer-02" in fixture[7]
    assert "startup-ray-20261007-j-saved-transfer-03" in fixture[7]
    assert all(f"generated-{i}" in fixture[8] for i in range(25))


@pytest.mark.parametrize("mutation", ["saved03_lost", "carry_reset", "duplicate_cost", "failed_predecessor", "incomplete_predecessor", "source_rebound"])
def test_history_refuses_rollback_double_charge_failed_or_incomplete_predecessor(mutation):
    f = history("tracking_candidate")
    if mutation == "saved03_lost":
        f[0]["debits"] = [d for d in f[0]["debits"] if d["cost_id"] != "startup-ray-20261007-j-saved-transfer-03"]
    elif mutation == "carry_reset":
        f[0]["startup_ray_original_local_used_s"] = 12.
    elif mutation == "duplicate_cost":
        f[0]["debits"].append(copy.deepcopy(f[0]["debits"][-1]))
    elif mutation == "failed_predecessor":
        f[6][tracking.CALLS[0]]["exit_code"] = 1
    elif mutation == "incomplete_predecessor":
        f[6]["capture-0"]["capture_complete"] = False
    else:
        f[6]["J-0"]["source_freeze"] = {"current": True}
    with pytest.raises(ValueError):
        tracking.validate_history(*f[:6])


def test_deadline_first_prevents_request_Source_and_original_reads(monkeypatch):
    monkeypatch.setenv("FLYLAB_DIRECT_SCIENCE_OWNER", json.dumps({"entry_counter_s": 100.}))
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 381.)
    touched = []
    monkeypatch.setattr(tracking, "spec_for", lambda *args: touched.append(args))
    with pytest.raises(TimeoutError, match="same owner ENTRY"):
        tracking.Admission("never-open-request.json", entry=101.)
    assert touched == []


def test_hot_callbacks_check_deadline_without_rehashing_all_Source(monkeypatch):
    admission = tracking.Admission.__new__(tracking.Admission)
    admission.entry, admission._deadline_s = 100., 80
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 101.)
    labels = []
    admission.fence = lambda label: labels.append(label)
    assert admission.check("science") is True and labels == []
    admission.check("pre_array")
    admission.check("publication")
    assert labels == ["pre_array", "publication"]
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 181.)
    with pytest.raises(TimeoutError):
        admission.check("publication")
    assert labels == ["pre_array", "publication"]  # deadline precedes cold work


def test_source_fence_accepts_JS_and_detects_same_length_Source_mutation(tmp_path):
    sources = {}
    for name in tracking.REQUIRED_SOURCES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = b"# generated text fixture\n"
        p.write_bytes(raw)
        sources[name] = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    freeze = {"schema": tracking.SCHEMA + ".source-freeze.v1", "ROOT_review_cycle_complete": True, "sources": sources}
    checkpoints = []
    tracking.current_sources(freeze, tmp_path, lambda: checkpoints.append(True))
    assert len(checkpoints) == 2 * len(sources)
    collector = tmp_path / (tracking.BASE + "/parent-direct-science-genuine-collector.js")
    collector.write_bytes(b"X" * sources[collector.relative_to(tmp_path).as_posix()]["bytes"])
    with pytest.raises(ValueError, match="hash"):
        tracking.current_sources(freeze, tmp_path, lambda: None)


def test_role_limits_are_separate_without_mutating_old_saved_contract():
    from tools import banc_m3_saved_original_admission as saved
    old = copy.deepcopy(saved.LIMITS)
    arm, diagnosis = tracking.limits_for("tracking_neutral"), tracking.limits_for("tracking_diagnosis")
    assert (arm["outer_s"], arm["child_s"], arm["internal_s"], arm["cleanup_s"]) == (300, 280, 250, 20)
    assert (diagnosis["outer_s"], diagnosis["child_s"], diagnosis["internal_s"], diagnosis["cleanup_s"]) == (120, 100, 80, 20)
    assert diagnosis["host_bytes"] == 2 * 1024**3 and arm["host_bytes"] == 24 * 1024**3
    assert saved.LIMITS == old and saved.ROLE == "saved_startup_transfer"
    with pytest.raises(ValueError):
        tracking.limits_for("neutral_body")


def refusal_fixture(tmp_path, monkeypatch):
    from tools import banc_m3_startup_ray_supervisor as supervisor

    def write(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False), encoding="utf8")
        return {"path": name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    call = tracking.CALLS[0]
    science = write(tracking.BASE + "/synthetic-refused-science.json", {"schema": tracking.SCHEMA + ".request.v1", "mode": "original", "role": tracking.ROLES[0], "limits": {}})
    freeze = write(tracking.BASE + "/synthetic-current-source.json", {"fixture": "generated metadata only; deliberately refused"})
    limits = tracking.limits_for(tracking.ROLES[0])
    outer_request = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "original", "role": tracking.ROLES[0], "call_id": call,
                     "science_request": science, "source_freeze": freeze, "grant": tracking.GRANT,
                     "shared_original_lock_path": tracking.SHARED_LOCK, "ledger_path": tracking.LEDGER,
                     "output": tracking.BASE + "/parent-" + call + "/result-supervision",
                     "limits": {k: limits[k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
    request = write(tracking.BASE + "/synthetic-supervisor-request.json", outer_request)
    before = {"debits": [{"cost_id": f"synthetic-prior-{i}", "mode": "generated"} for i in range(165)]}
    before_spec = write(tracking.BASE + "/synthetic-policy-before.json", before)
    policy = write(tracking.BASE + "/synthetic-policy.json", {"schema": "flylab.m3.direct-science.policy.v1", "generated_local_limit_s": 300,
                                                            "original_authorized": False, "before_ledger": before_spec, "local_debit_offset": 165})
    ledger = {"debits": before["debits"], "direct_science_policy": policy,
              "direct_science_calls": {call: {"request": request, "state": "admitted", "mode": "original", "reserved_s": 300}},
              "startup_ray_original_workers": {}, "startup_ray_original_reservations": {}, "startup_ray_original_pending_actuals": {},
              "startup_ray_original_blocking_holds": {}, "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {},
              "original_jobs_running": 0, "original_reserved_s": 0}
    write(tracking.LEDGER, ledger)
    monkeypatch.setattr(supervisor, "creation_ticks", lambda pid: 999)
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: 101.)
    attempted = []

    def never_job(*args):
        attempted.append(args)
        raise AssertionError("pre-spawn refusal must not create a Job")

    monkeypatch.setattr(supervisor, "OwnedJob", never_job)
    result = supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    return supervisor, write, outer_request, request, ledger, result, attempted


def test_tracking_pre_spawn_refusal_has_actual_empty_streams_no_Job_and_no_duplicate_publication(tmp_path, monkeypatch):
    supervisor, _, q, request, _, outer, attempted = refusal_fixture(tmp_path, monkeypatch)
    assert outer["exit_code"] == 1 and outer["first_error"]["type"] == "ValueError"
    assert outer["pre_spawn_refused"] and outer["worker_started"] is False and outer["job_created"] is False
    assert outer["worker_pid"] is None and outer["worker_exit_code"] is None and outer["original_payload_accesses"] == 0
    assert outer["all_owned_ended"] and outer["reservation_released"] and outer["stdout_eof"] and outer["stderr_eof"]
    assert attempted == []
    output = tmp_path / q["output"]
    assert (output / "stdout.bin").read_bytes() == (output / "stderr.bin").read_bytes() == b""
    snapshot = (output / "outer.json").read_bytes()
    assert len(snapshot) <= 64 << 10 and outer["output_usage"]["combined_bytes_including_outer"] == len(snapshot)
    with pytest.raises(ValueError, match="publish twice"):
        supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    assert (output / "outer.json").read_bytes() == snapshot and attempted == []


def test_native_once_writer_charges_synthetic_refusal_then_rejects_duplicate(tmp_path, monkeypatch):
    # Generated fixture: synthetic request/return/debits under a replaced literal
    # temp root. Only the real small J grant metadata and account Source are read.
    _, write, q, request, ledger, _, attempted = refusal_fixture(tmp_path, monkeypatch)
    assert os.name == "nt" and shutil.which("pwsh"), "Native PowerShell fixture requires the assigned Windows host"
    root = Path(tracking.__file__).resolve().parents[1]
    original_ledger = root / tracking.LEDGER
    original_digest = hashlib.sha256(original_ledger.read_bytes()).hexdigest()
    grant_file = root / tracking.GRANT["path"]
    grant_target = tmp_path / tracking.GRANT["path"]
    grant_target.parent.mkdir(parents=True, exist_ok=True)
    grant_target.write_bytes(grant_file.read_bytes())
    assert hashlib.sha256(grant_target.read_bytes()).hexdigest() == tracking.GRANT["sha256"]
    g = json.loads(grant_target.read_text(encoding="utf8"))
    ledger["startup_ray_original_grants"] = {g["grant_id"]: {"grant": tracking.GRANT, "source_freeze": g["source_freeze"], "inputs": g["inputs"],
                                                           "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}}
    write(tracking.LEDGER, ledger)
    trace_name = tracking.BASE + "/synthetic-genuine-return.json"
    write(trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                       "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125, "exit_code": 1,
                       "request": request, "source_freeze": q["source_freeze"]})
    source = (root / tracking.BASE / "parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    escaped_root = tmp_path.as_posix().replace("'", "''")
    script = tmp_path / "synthetic-account.ps1"
    script.write_text(source.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('" + escaped_root + "')"), encoding="utf8")
    command = [shutil.which("pwsh"), "-NoProfile", "-File", str(script), "-Action", "commit", "-TracePath", trace_name]
    # Native PowerShell can localize error prefixes using the host code page.
    # Check the invariant ASCII rejection in bytes so decoding cannot hide it.
    first = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert first.returncode == 0, first.stderr
    after = json.loads((tmp_path / tracking.LEDGER).read_text(encoding="utf8"))
    assert len(after["debits"]) == 166 and after["debits"][-1]["cost_id"] == tracking.CALLS[0]
    assert after["debits"][-1]["actual_outer_full_return_s"] == .125
    cost = json.loads((tmp_path / after["debits"][-1]["descriptor"]["path"]).read_text(encoding="utf8"))
    assert cost["pre_spawn_refused"] and cost["worker_started"] is False and cost["original_payload_accesses"] == 0
    assert cost["schema"] == tracking.SCHEMA + ".full-return.v1" and cost["charged_once"] is True
    duplicate = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert duplicate.returncode != 0 and b"Fresh once commit" in duplicate.stderr
    assert len(json.loads((tmp_path / tracking.LEDGER).read_text(encoding="utf8"))["debits"]) == 166
    assert hashlib.sha256(original_ledger.read_bytes()).hexdigest() == original_digest and attempted == []
