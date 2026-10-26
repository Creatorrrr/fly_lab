"""Bounded generated headers and metadata-only production retry fixtures.

No large synthetic arrays or Original numeric payloads are opened by this file.
The Native batch runs one process; the real owner9 remains a separate selection.
"""
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import banc_m3_contact_learning as s
from tools import banc_m3_contact_learning_admission as gate
from tools.banc_m3_saved_original_admission import closed_cost


def pin_file(path, name):
    raw = path.read_bytes()
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False), encoding="utf8")
    return pin_file(path, name)


def bounded(spec, root=gate.ROOT, cap=1 << 20):
    assert Path(spec["path"]).suffix == ".json"
    return gate.json_load(gate.leaf(spec, root, cap))


def header(shape, dtype="|u1", fortran=False, version=(1, 0)):
    stream = io.BytesIO()
    value = {"shape": shape, "fortran_order": fortran, "descr": dtype}
    if version == (1, 0):
        np.lib.format.write_array_header_1_0(stream, value)
    else:
        np.lib.format.write_array_header_2_0(stream, value)
    return stream.getvalue()


@pytest.fixture(name="retry_tmp")
def retry_tmp_fixture():
    """Owned short scratch roots preserve archived relative paths on Windows."""
    with tempfile.TemporaryDirectory(prefix="cb-") as folder:
        yield Path(folder)


def whole_spec(size):
    return {"path": "generated-counts.npy", "bytes": size, "sha256": "0" * 64}


def test_math_whole_npy_headers_sizes_and_strict_negatives():
    cap = 160 << 20
    for version in ((1, 0), (2, 0)):
        n = len(header((1,), version=version))
        for size in (92678944, cap):
            assert s.whole_npy_bytes(whole_spec(size), (size-n,), "|u1", version, False, n, size, cap) == size-n
        # Exact bounded4096 header arithmetic, including the version prefix.
        n = (10 if version == (1, 0) else 12) + 4096
        assert s.whole_npy_bytes(whole_spec(n), (0,), "|u1", version, False, n, n, cap) == 0
    positive = [whole_spec(256), (128,), "|u1", (1, 0), False, 128, 256, cap]
    negatives = ((0, whole_spec(cap+1)), (1, (-1,)), (1, (1.5,)), (1, (True,)),
                 (1, (1,)*5), (2, "|O"), (2, "<c8"), (3, (3, 0)), (4, True),
                 (5, 4107), (5, 10), (5, 127), (6, 257))
    for index, value in negatives:
        args = positive.copy()
        args[index] = value
        with pytest.raises(ValueError):
            s.whole_npy_bytes(*args)
    assert s.member_bytes((64 << 20,), "|u1", 64 << 20) == 64 << 20
    with pytest.raises(ValueError, match="member64MiB"):
        s.member_bytes(((64 << 20)+1,), "|u1", 64 << 20)


def test_math_body_plan_134_whole_npy_and_npz_selected_caps(tmp_path):
    stage = bounded(gate.spec_for(gate.STAGE_SPEC, gate.ROOT))
    raw_header = header((92678944-128,))
    assert len(raw_header) == 128
    candidate = tmp_path / "generated-candidate.npz"
    np.savez(candidate, theta_actual=np.zeros(1, np.float64))
    candidate_spec = pin_file(candidate, candidate.name)
    old = {"path": "immutable-old-audited.npz"}
    role = {"role": "original_initial_whole", "producer_pin_inherited_from": "generated-producer.json"}
    files = {"generated-counts.npy": dict(bytes=92678944, sha256="0"*64, **role)}
    files.update({"metadata-"+str(i)+".json": dict(bytes=1, sha256="0"*64, **role) for i in range(130)})
    files[old["path"]] = {"bytes": 1, "sha256": "0"*64, "kind": "npz", "selected_archive_role": "audited_candidate"}
    selected = {"audited_candidate": old}
    for name in ("recurrent_mask", "reference", "startup"):
        archive = tmp_path / ("generated-selected-" + name + ".npz")
        np.savez(archive, x=np.zeros(1, np.float64))
        spec = pin_file(archive, archive.name)
        files[archive.name] = {"bytes": spec["bytes"], "sha256": spec["sha256"], "kind": "npz", "selected_archive_role": name}
        selected[name] = dict(spec, kind="npz", selected_schema={"x": {"shape": [1], "dtype": "<f8", "finite": True}}, producer="generated-producer.json")
    assert len(files) == 135 and sum("role" in v for v in files.values()) == 131
    assert all(set(v) == ({"bytes", "sha256", "role", "producer_pin_inherited_from"} if "role" in v
                          else {"bytes", "sha256", "kind", "selected_archive_role"}) for v in files.values())
    with pytest.raises(ValueError, match="descriptor fields"):
        s.whole_npy_descriptor(dict(files["generated-counts.npy"], path="generated-counts.npy"), 160 << 20)
    admission = SimpleNamespace(root=tmp_path, limits=copy.deepcopy(stage["limits"][gate.ROLES[2]]),
                                inputs={"body_inventory": {"inventory": {"files": files, "selected_archives": selected}}},
                                fence=lambda *args: True, check=lambda *args: True)
    saved = s.Saved(admission, np)
    calls = []
    original_digest = saved.base.digest
    class HeaderPath:
        suffix = ".npy"
        def open(self, mode):
            assert mode == "rb"
            return io.BytesIO(raw_header)
        def stat(self):
            return SimpleNamespace(st_size=files["generated-counts.npy"]["bytes"])
    def digest(spec):
        calls.append(spec["path"])
        if spec["path"] in files and "selected_schema" not in spec:
            assert set(spec) == {"path", "bytes", "sha256"}
            assert spec["bytes"] == files[spec["path"]]["bytes"] and spec["sha256"] == files[spec["path"]]["sha256"]
        if spec["path"] == candidate.name or spec["path"] == "oversized-member.npz" or spec["path"].startswith("generated-selected-"):
            return original_digest(spec)
        if spec["path"].endswith(".npy"):
            return HeaderPath(), 100
        return tmp_path/spec["path"], 100
    saved.base.digest = digest  # header-only stream/stat boundary, not a planning decision
    inventory_before = copy.deepcopy(admission.inputs["body_inventory"]["inventory"])
    plan = s.body_plan(admission, saved, candidate_spec, np)
    assert admission.inputs["body_inventory"]["inventory"] == inventory_before
    assert len(plan["entries"]) == 134 and plan["selected_upper_bytes"] == 92678944-128+130+24+8
    assert sum(v["path"].startswith("generated-selected-") and v["planned_bytes"] == 8 for v in plan["entries"]) == 3
    assert all(v["path"] != old["path"] for v in plan["entries"])
    assert sum(spec["path"] == "generated-counts.npy" for spec, _ in saved.base.ledger) == 1
    saved.posthash()  # actual production posthash includes the planned whole NPY
    for size in ((160 << 20), (160 << 20)+1):
        files["generated-counts.npy"]["bytes"] = size
        raw_header = header((size-128,))
        before = len(calls)
        if size == 160 << 20:
            assert s.body_plan(admission, saved, candidate_spec, np)["entries"][0]["planned_bytes"] == size-128
        else:
            with pytest.raises(ValueError):
                s.body_plan(admission, saved, candidate_spec, np)
            assert len(calls) == before  # size cap precedes digest/header access
    files["generated-counts.npy"]["bytes"] = 92678944
    raw_header = header((92678944-128,))
    for kind in ("fortran", "object", "payload", "version", "header"):
        if kind == "fortran":
            raw_header = header((92678944-128,), fortran=True)
        elif kind == "object":
            raw_header = header((1,), dtype="|O")
        elif kind == "payload":
            raw_header = header((1,))
        elif kind == "version":
            raw_header = b"\x93NUMPY\x03\x00" + b"\x00"*120
        else:
            raw_header = header((1,)*1500)
        with pytest.raises(ValueError):
            s.body_plan(admission, saved, candidate_spec, np)
    raw_header = header((92678944-128,))
    admission.limits["host_bytes"] = 3*plan["selected_upper_bytes"]+(96 << 20)-1
    with pytest.raises(ValueError, match="temporary host"):
        s.body_plan(admission, saved, candidate_spec, np)
    admission.limits = copy.deepcopy(stage["limits"][gate.ROLES[2]])
    for name, value in files.items():
        if name.endswith(".json"):
            value["bytes"] = 4 << 20
    with pytest.raises(ValueError, match="selected body512MiB"):
        s.body_plan(admission, saved, candidate_spec, np)
    for name, value in files.items():
        if name.endswith(".json"):
            value["bytes"] = 1
    import zipfile
    oversized = tmp_path/"oversized-member.npz"
    with zipfile.ZipFile(oversized, "w") as archive:
        archive.writestr("too_big.npy", header(((64 << 20)+1,)))
    spec = pin_file(oversized, oversized.name)
    with pytest.raises(ValueError, match="member64MiB"):
        saved.arrays(spec, plan_only=True)  # unchanged actual NPZ production planner


class MetadataFixture:
    """Real pinned historic JSON and small generated current proofs, never arrays."""
    def __init__(self, root):
        self.root, self.seen = root, {}
        self.contract_spec = gate.spec_for(gate.STAGE_SPEC, gate.ROOT)
        self.contract = self.read(self.contract_spec)
        self.roster, self.grant = self.read(gate.ROSTER), self.read(gate.GRANT)
        r = dict(copy.deepcopy(gate.RETRY_PINS), schema=gate.SCHEMA+".technical-body-retry.v1", retry_id="contact-learning-20261007-c",
                 allowed_roles=list(gate.ROLES[2:]), calls=list(gate.C_RETRY_CALLS), maximum_retries=1, automatic_repeats=0,
                 resource_increment_s=0, contract=self.contract_spec, grant=gate.GRANT, inputs=gate.INPUTS, roster=gate.ROSTER,
                 descriptor_recovery=copy.deepcopy(gate.C_RECOVERY_PINS))
        self.retry = write(root, gate.C_RETRY_PATH, r)
        self.source = gate.spec_for(gate.C_SOURCE_PATH, gate.ROOT)
        self.before = self.read(gate.C_PASSED_W_CLOSED, 524288)
        for spec in gate.RECOVERY_COSTS:
            paid = self.read(spec)
            for key in ("outer", "trace", "request", "source_freeze"):
                self.read(paid[key])
        for i, call in enumerate(gate.C_L_CALLS):
            self.add_generated(call, i)
        self.body = self.science(gate.ROLES[2])
        self.audit_before = copy.deepcopy(self.before)
        self.add_body(self.audit_before)
        self.audit = self.science(gate.ROLES[3])
        self.copy_seen()

    def read(self, spec, cap=1 << 20):
        root = self.root if (self.root/spec["path"]).is_file() else gate.ROOT
        value = bounded(spec, root, cap)
        if root == gate.ROOT:
            self.seen[spec["path"]] = spec
        return value

    def copy_seen(self):
        for name, spec in self.seen.items():
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(gate.leaf(spec, gate.ROOT, 1 << 20))
        for index_pin in (gate.RETRY_PINS["archive_k"], gate.C_RECOVERY_PINS["archive_s"]):
            index = self.read(index_pin)
            for item in index["sources"]:
                spec = item["as_executed"]
                target = self.root/spec["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gate.leaf(spec, gate.ROOT, 1 << 20))
        qualified = self.read(gate.C_RECOVERY_PINS["qualification_s"])
        for test in qualified["tests"]:
            spec = test["file"]
            target = self.root/spec["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gate.leaf(spec, gate.ROOT, 1 << 20))

    def closed(self, cost, debit):
        for key in ("outer", "trace"):
            self.read(cost[key])
        root = self.root if (self.root/cost["outer"]["path"]).exists() else gate.ROOT
        return closed_cost(cost, debit, root)

    @staticmethod
    def archive(item):
        path = Path(item["archive_path"])
        assert path.stat().st_size <= 1 << 20
        raw = path.read_bytes()
        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
        return raw

    def science(self, role):
        r = self.read(self.retry)
        manifest_spec = r["historical_manifests"][1] if role == gate.ROLES[2] else self.body_manifest
        manifest = self.read(manifest_spec)
        return {"schema": gate.SCHEMA+".request.v1", "mode": "original", "role": role,
                "call_id": gate.C_RETRY_CALLS[gate.ROLES[2:].index(role)], "limits": self.contract["limits"][role],
                "contract": self.contract_spec, "body_retry": self.retry, "source_freeze": self.source,
                "grant": gate.GRANT, "inputs": gate.INPUTS, "roster": gate.ROSTER, "local_debit_offset": 165,
                "ledger_path": gate.LEDGER, "shared_original_lock_path": gate.SHARED_LOCK,
                "historical_manifest": self.read(self.roster["previous_geometry_roster"])["historical_manifest"],
                "before_ledger": write(self.root, gate.BASE+"/generated-before-"+role+".json", self.before if role == gate.ROLES[2] else self.audit_before),
                "resource_increment_s": 0, "output": gate.BASE+"/parent-"+gate.C_RETRY_CALLS[gate.ROLES[2:].index(role)]+"/result",
                "artifacts": dict(manifest["artifacts"], manifest=manifest_spec)}

    def proof(self, call, mode, owner_spec, exit_code=0):
        stem = gate.BASE+"/generated-proof-"+call
        outer = write(self.root, stem+"-outer.json", {"request": owner_spec, "exit_code": exit_code, "timeout": False,
                      "all_owned_ended": True, "reservation_released": True, "stdout_eof": True, "stderr_eof": True, "unknown_owned_identities": {}})
        trace = write(self.root, stem+"-trace.json", {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                      "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125, "exit_code": exit_code,
                      "request": owner_spec, "source_freeze": self.source})
        return {"schema": "flylab.m3.direct-science.full-return.v1", "cost_id": call, "mode": mode, "request": owner_spec,
                "source_freeze": self.source, "outer": outer, "trace": trace, "actual_outer_full_return_s": .125,
                "exit_code": exit_code, "charged_once": True, "all_owned_ended": True, "reservation_released": True, "cap_exceeded": False}

    def add_generated(self, call, index):
        caps = ((8, 4, 4), (12, 8, 4), (80, 72, 8))[index]
        q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "role": ("lint", "numerical", "execution")[index],
             "call_id": call, "source_freeze": self.source, "limits": dict(zip(("outer_s", "child_s", "cleanup_s"), caps, strict=True))}
        request = write(self.root, gate.BASE+"/generated-proof-"+call+"-request.json", q)
        cost = self.proof(call, "generated", request)
        spec = write(self.root, gate.BASE+"/generated-proof-"+call+"-cost.json", cost)
        self.before["debits"].append({"cost_id": call, "mode": "generated", "actual_outer_full_return_s": .125, "actual_returned_exit_code": 0, "descriptor": spec})
        self.before["direct_science_calls"][call] = {"request": request, "state": "charged", "mode": "generated", "reserved_s": caps[0]}

    def add_body(self, before):
        q = self.body
        science_spec = write(self.root, gate.BASE+"/generated-proof-b-science.json", q)
        owner = {"call_id": q["call_id"], "role": q["role"], "science_request": science_spec, "grant": gate.GRANT,
                 "source_freeze": self.source, "output": q["output"]+"-supervision",
                 "limits": {k: q["limits"][k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
        owner_spec = write(self.root, gate.BASE+"/generated-proof-b-owner.json", owner)
        cost = self.proof(q["call_id"], "original", owner_spec)
        cost.update(schema=gate.SCHEMA+".full-return.v1", request=science_spec, supervisor_request=owner_spec,
                    **{k: q[k] for k in ("role", "grant", "inputs", "roster", "contract", "local_debit_offset", "resource_increment_s", "body_retry")})
        cost_spec = write(self.root, gate.BASE+"/generated-proof-b-cost.json", cost)
        before["debits"].append({"cost_id": q["call_id"], "mode": "original", "actual_outer_full_return_s": .125, "actual_returned_exit_code": 0, "descriptor": cost_spec})
        before["direct_science_calls"][q["call_id"]] = {"request": owner_spec, "state": "charged", "mode": "original", "reserved_s": 300}
        for key in ("original_cumulative_used_s", "original_stage_used_s", "startup_ray_original_local_used_s"):
            before[key] += .125
        before["original_remaining_s"] -= .125
        report = {"run_valid": True, "first_error": None, "secondary_errors": [], "candidate_present": True,
                  "candidate": self.read(self.retry)["candidate"], "reason": "startup_rejected", "actual_controls": 99}
        self.body_report = write(self.root, gate.BASE+"/generated-proof-b-report.json", report)
        self.body_manifest = write(self.root, gate.BASE+"/generated-proof-b-manifest.json", {"schema": gate.SCHEMA+".manifest.v1", "role": q["role"],
                   "source_freeze": self.source, "request": science_spec, "contract": q["contract"], "run_valid": True,
                   "artifacts": {"report": self.body_report, "control": {"generated": True}, "native": {"generated": True},
                                 "evaluation": {"generated": True}, "trace": {"generated": True}}})


def native_script(root, body):
    source = (gate.ROOT/gate.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    definitions = source[source.index("function Need("):source.index("function Write-New(")]
    helper = (gate.ROOT/gate.BASE/"parent-contact-learning-contract-native.ps1").as_posix().replace("'", "''")
    setup = "$ErrorActionPreference='Stop'\n$taskRoot=[IO.Path]::GetFullPath('"+root.as_posix().replace("'", "''")+"')\n$taskBase='"+gate.BASE+"'\n$utf8=[Text.UTF8Encoding]::new($false,$true)\n"
    target = root/"generated-native-retry-batch.ps1"
    prepare = (gate.ROOT/gate.BASE/"parent-contact-learning-prepare-native.ps1").read_text(encoding="utf8")
    comparator = prepare[prepare.index("function Contact-SameTyped("):prepare.index("function Target(")]
    target.write_text(setup+definitions+"\n"+comparator+"\n. '"+helper+"'\n"+body, encoding="utf8")
    return target


def test_gate_retry_python_native_history21_22_and_strict_negatives(retry_tmp):
    tmp_path = retry_tmp
    fixture = MetadataFixture(tmp_path)
    positives = ((fixture.body, fixture.before), (fixture.audit, fixture.audit_before))
    for q, before in positives:
        assert gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive) > 0
    fixture.copy_seen()
    current = fixture.read(gate.spec_for(gate.conserved.F_SOURCE_PATH, gate.ROOT))
    gate.current_sources(current, gate.ROOT, lambda: True)  # actual fresh F Source; every c/x historical assertion stays
    drift = copy.deepcopy(current)
    drift["sources"]["tools/banc_m3_contact_learning.py"]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        gate.current_sources(drift, gate.ROOT, lambda: True)
    cases = []
    for q, before in positives:
        cases.append({"q": q, "before": write(tmp_path, gate.BASE+"/generated-case-before-"+q["role"]+".json", before), "pass": True})
    for kind in ("missing_descriptor", "wrong_role", "a_with_descriptor", "suffix", "cap", "source", "candidate", "history", "extra_original", "wrong_exit", "timeout", "unknown", "failure_tuple",
                 "b_replay", "s_source", "missing_t", "b_failure_tuple", "s_qualification", "s_archive"):
        q, before = copy.deepcopy(fixture.body), copy.deepcopy(fixture.before)
        if kind == "missing_descriptor":
            del q["body_retry"]
        elif kind == "wrong_role":
            q["role"] = gate.ROLES[0]
        elif kind == "a_with_descriptor":
            q["call_id"] = fixture.contract["calls"][2]
        elif kind == "suffix":
            q["call_id"] += "-c"
        elif kind == "cap":
            q["limits"]["outer_s"] += 1
        elif kind == "source":
            q["source_freeze"] = gate.RETRY_PINS["source_k"]
        elif kind == "b_replay":
            q["call_id"] = gate.RETRY_CALLS[0]
        elif kind == "s_source":
            q["source_freeze"] = gate.C_RECOVERY_PINS["source_s"]
        elif kind == "missing_t":
            omitted = before["debits"].pop()
            del before["direct_science_calls"][omitted["cost_id"]]
        elif kind == "candidate":
            wrong = fixture.read(fixture.retry)
            wrong["candidate"]["alpha"] = .0009
            q["body_retry"] = write(tmp_path, gate.BASE+"/generated-wrong-candidate-descriptor.json", wrong)
        elif kind in ("history", "extra_original", "wrong_exit"):
            if kind == "history":
                before["debits"][0]["cost_id"] = "changed-old-cost"
            elif kind == "extra_original":
                before["debits"].append(copy.deepcopy(before["debits"][-1]))
            else:
                next(d for d in before["debits"] if d["cost_id"] == fixture.contract["calls"][2])["actual_returned_exit_code"] = 0
        else:
            # Changing exact owner/report bytes must fail their actual production pin boundary.
            cost = fixture.read(gate.RETRY_PINS["historical_costs"][2])
            spec = cost["outer"] if kind in ("timeout", "unknown") else fixture.read(gate.RETRY_PINS["historical_manifests"][2])["artifacts"]["report"]
            if kind == "b_failure_tuple":
                spec = gate.C_RECOVERY_PINS["report_b"]
            elif kind in ("s_qualification", "s_archive"):
                spec = gate.C_RECOVERY_PINS["qualification_s" if kind == "s_qualification" else "archive_s"]
            original = (tmp_path/spec["path"]).read_bytes()
            wrong = fixture.read(spec)
            if kind == "timeout":
                wrong["timeout"] = True
            elif kind == "unknown":
                wrong["unknown_owned_identities"] = {"generated": 1}
            elif kind == "s_qualification":
                wrong["source_freeze"]["sha256"] = "0"*64
            elif kind == "s_archive":
                wrong["sources"][0]["as_executed"]["sha256"] = "0"*64
            else:
                wrong["first_error"]["message"] = "scientific rejection"
            write(tmp_path, spec["path"], wrong)
            with pytest.raises((ValueError, KeyError)):
                gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive)
            (tmp_path/spec["path"]).write_bytes(original)
            cases.append({"q": q, "before": cases[0]["before"], "pass": False, "tamper": spec, "replacement": wrong})
            continue
        with pytest.raises((ValueError, KeyError)):
            if "body_retry" not in q:
                gate.retry_binding(q, fixture.contract, fixture.read)
            else:
                gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive)
        spec = write(tmp_path, gate.BASE+"/generated-case-"+kind+"-before.json", before)
        cases.append({"q": q, "before": spec, "pass": False})
    # Exercise the actual Prepare comparator in the same Native process. Reorder
    # every nested object's keys while preserving types and ordered array values.
    def reorder(value):
        if isinstance(value, dict):
            return {k: reorder(value[k]) for k in reversed(list(value))}
        if isinstance(value, list):
            return [reorder(v) for v in value]
        return value
    original_identity = copy.deepcopy(gate.RETRY_PINS["candidate"])
    changed_identities = []
    for key, nested, value in (("palette", "full_native_palette_count", 8353),
                               ("loss", "s_mm", .021), ("source_freeze", "bytes", 64278),
                               ("contract", "sha256", "0"*64)):
        wrong = copy.deepcopy(original_identity)
        wrong[key][nested] = value
        changed_identities.append(wrong)
    wrong = copy.deepcopy(original_identity)
    wrong["index"] = float(wrong["index"])
    changed_identities.append(wrong)  # exact scalar type, not loose numeric equality
    wrong = copy.deepcopy(original_identity)
    del wrong["roster"]
    changed_identities.append(wrong)  # complete key sets
    compare_spec = write(tmp_path, gate.BASE+"/generated-prepare-comparator.json",
                         {"original": original_identity, "reordered": reorder(original_identity),
                          "changed": changed_identities, "array_original": [1, {"x": .5}, "last"],
                          "array_reordered_object": [1, {"x": .5}, "last"],
                          "array_wrong_order": ["last", {"x": .5}, 1]})
    case_spec = write(tmp_path, gate.BASE+"/generated-native-cases.json", cases)
    body = "$cases=Read-Pin "+("('"+json.dumps(case_spec).replace("'", "''")+"'|ConvertFrom-Json -AsHashtable)")+"\n"
    body += """
foreach($case in $cases){
  $backup=$null
  if($case.Contains('tamper')){$p=Safe $case.tamper.path;$backup=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllText($p,($case.replacement|ConvertTo-Json -Depth 100),$utf8)}
  $passed=$false
  try{$q=$case.q;$contract=Read-Pin $q.contract;$r=Contact-RetryBinding $q $contract;Need ($null -ne $r) 'Fixed b descriptor required';$before=Read-Pin $case.before 524288;$g=Read-Pin $q.grant;$null=Contact-RetryHistory $q $before $before $contract $g $r;$passed=$true}
  catch{if($case.pass){throw}}
  finally{if($null -ne $backup){[IO.File]::WriteAllBytes($p,$backup)}}
  Need ($passed -ceq $case.pass) 'Actual Native retry positive/strict negative outcome'
}
"""
    body += "$identity=Read-Pin ('"+json.dumps(compare_spec).replace("'", "''")+"'|ConvertFrom-Json -AsHashtable)\n"
    body += """
Contact-SameTyped $identity.reordered $identity.original 'Reordered entire nested candidate accepted'
Contact-SameTyped $identity.array_reordered_object $identity.array_original 'Ordered arrays retain typed values'
foreach($wrong in $identity.changed){
  $refused=$false;try{Contact-SameTyped $wrong $identity.original 'Changed nested candidate rejected'}catch{$refused=$true}
  Need $refused 'Actual Prepare comparator rejects changed palette/loss/provenance/type/key set'
}
$refused=$false;try{Contact-SameTyped $identity.array_wrong_order $identity.array_original 'Array order exact'}catch{$refused=$true}
Need $refused 'Actual Prepare comparator rejects changed array order'
"""
    script = native_script(tmp_path, body)
    observed = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script)], capture_output=True, timeout=24, check=False)
    assert observed.returncode == 0, observed.stderr


def test_gate_retry_body_audit_candidate_k_execution_l_identity(retry_tmp, monkeypatch):
    tmp_path = retry_tmp
    fixture = MetadataFixture(tmp_path)
    report = fixture.read(fixture.body_report)
    gate.retry_candidate_identity(fixture.audit, report, fixture.read)
    assert report["candidate"]["source_freeze"] == gate.RETRY_PINS["source_k"] != fixture.audit["source_freeze"]
    from tools import analyze_c_banc_m3_contact_learning as audit
    from tools import verify_c_banc_m3_startup_ray as ray
    touched = []
    saved = SimpleNamespace(json=lambda spec: report if spec == fixture.body_report else {},
                            arrays=lambda *args, **kwargs: touched.append(args) or ({"expanded_bytes": 0} if kwargs else {}))
    admission = SimpleNamespace(request=fixture.audit, read=fixture.read, limits={"selected_expanded_cap_bytes": 512 << 20},
                                inputs={"body_inventory": {"inventory": {}}})
    def evaluator(*args, **kwargs):
        return {"screen_accepted": False, "metrics": None, "gates": None}
    monkeypatch.setattr(ray, "audit_body", evaluator)  # only the unchanged downstream evaluator is synthetic
    result = {}
    audit.body_audit(admission, result, SimpleNamespace(folder=tmp_path), saved, np)
    assert result["classification"] == "scientific_rejected" and result["candidate"] == gate.RETRY_PINS["candidate"]
    for key, value in (("source_freeze", fixture.source), ("alpha", .0009), ("index", 3), ("theta_sha256", "0"*64), ("loss", {}), ("palette", {})):
        wrong = copy.deepcopy(report)
        wrong["candidate"][key] = value
        with pytest.raises(ValueError):
            gate.retry_candidate_identity(fixture.audit, wrong, fixture.read)
    wrong = copy.deepcopy(fixture.audit)
    manifest = fixture.read(wrong["artifacts"]["manifest"])
    manifest["source_freeze"] = gate.RETRY_PINS["source_k"]
    wrong["artifacts"]["manifest"] = write(tmp_path, gate.BASE+"/generated-wrong-immediate-manifest.json", manifest)
    with pytest.raises(ValueError):
        gate.retry_candidate_identity(wrong, report, fixture.read)
    legacy = dict(fixture.audit, call_id=fixture.contract["calls"][3])
    del legacy["body_retry"]
    with pytest.raises(ValueError, match="same frozen"):
        gate.retry_candidate_identity(legacy, report, fixture.read)


def test_gate_retry_refusal_temp_once_commit_and_duplicate(retry_tmp, monkeypatch):
    tmp_path = retry_tmp
    from tools import banc_m3_startup_ray_supervisor as supervisor
    fixture = MetadataFixture(tmp_path)
    science = copy.deepcopy(fixture.body)
    refused_freeze = write(tmp_path, fixture.source["path"], {"fixture": "intentional current Source pre-spawn refusal"})
    science["source_freeze"] = refused_freeze
    science_spec = write(tmp_path, gate.BASE+"/generated-refused-b-science.json", science)
    call = gate.C_RETRY_CALLS[0]
    limits = science["limits"]
    q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "original", "role": science["role"], "call_id": call,
         "science_request": science_spec, "source_freeze": refused_freeze, "grant": gate.GRANT, "ledger_path": gate.LEDGER,
         "shared_original_lock_path": gate.SHARED_LOCK, "output": science["output"]+"-supervision",
         "limits": {k: limits[k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
    request = write(tmp_path, gate.BASE+"/generated-refused-b-owner.json", q)
    debits = [{"cost_id": "generated-old-"+str(i), "mode": "generated"} for i in range(165)]
    before = write(tmp_path, gate.BASE+"/generated-refusal-policy-before.json", {"debits": debits})
    policy = write(tmp_path, gate.BASE+"/generated-refusal-policy.json", {"schema": "flylab.m3.direct-science.policy.v1", "generated_local_limit_s": 300,
                   "original_authorized": False, "before_ledger": before, "local_debit_offset": 165})
    ledger = {"debits": debits, "direct_science_policy": policy,
              "direct_science_calls": {call: {"request": request, "state": "admitted", "mode": "original", "reserved_s": 300}},
              "startup_ray_original_workers": {}, "startup_ray_original_reservations": {}, "startup_ray_original_pending_actuals": {}, "startup_ray_original_blocking_holds": {},
              "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {}, "original_jobs_running": 0, "original_reserved_s": 0,
              "startup_ray_original_grants": {fixture.grant["grant_id"]: {"grant": gate.GRANT, "source_freeze": fixture.grant["source_freeze"], "inputs": fixture.grant["inputs"],
                                              "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}}}
    write(tmp_path, gate.LEDGER, ledger)
    monkeypatch.setattr(supervisor, "creation_ticks", lambda pid: 999)
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: 101.)
    def no_job(*args):
        raise AssertionError("No Job in actual pre-spawn refusal")
    monkeypatch.setattr(supervisor, "OwnedJob", no_job)
    outer = supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    assert outer["pre_spawn_refused"] and not outer["job_created"] and not outer["worker_started"] and outer["original_payload_accesses"] == 0
    for label in ("stdout", "stderr"):
        assert (tmp_path/q["output"]/(label+".bin")).read_bytes() == b""
    ledger_path = tmp_path/gate.LEDGER
    admitted_bytes = ledger_path.read_bytes()
    refusal_files = {p.name: p.read_bytes() for p in (tmp_path/q["output"]).iterdir()}
    with pytest.raises(ValueError, match="^refused call cannot publish twice$"):
        supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    assert ledger_path.read_bytes() == admitted_bytes
    assert {p.name: p.read_bytes() for p in (tmp_path/q["output"]).iterdir()} == refusal_files
    assert os.name == "nt" and shutil.which("pwsh")
    trace_name = gate.BASE+"/generated-refused-b-return.json"
    write(tmp_path, trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                               "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125, "exit_code": 1, "request": request, "source_freeze": refused_freeze})
    source = (gate.ROOT/gate.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    target = tmp_path/"generated-refusal-account.ps1"
    target.write_text(source.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('"+tmp_path.as_posix().replace("'", "''")+"')"), encoding="utf8")
    helper = tmp_path/gate.BASE/"parent-contact-learning-contract-native.ps1"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_bytes((gate.ROOT/gate.BASE/"parent-contact-learning-contract-native.ps1").read_bytes())
    command = [shutil.which("pwsh"), "-NoProfile", "-File", str(target), "-Action", "commit", "-TracePath", trace_name]
    first = subprocess.run(command, capture_output=True, timeout=8, check=False)
    assert first.returncode == 0, first.stderr
    after = json.loads((tmp_path/gate.LEDGER).read_text(encoding="utf8"))
    assert len(after["debits"]) == 166 and after["direct_science_calls"][call]["state"] == "charged"
    cost = bounded(after["debits"][-1]["descriptor"], tmp_path)
    assert cost["body_retry"] == fixture.retry and cost["pre_spawn_refused"] and cost["original_payload_accesses"] == 0
    duplicate = subprocess.run(command, capture_output=True, timeout=8, check=False)
    assert duplicate.returncode != 0 and b"Fresh once commit" in duplicate.stderr
    assert len(json.loads((tmp_path/gate.LEDGER).read_text(encoding="utf8"))["debits"]) == 166
    charged_bytes = ledger_path.read_bytes()
    with pytest.raises(ValueError, match=r"^actual current admitted refused request\.state$"):
        supervisor.supervise(request["path"], root=tmp_path, entry=100.)
    assert ledger_path.read_bytes() == charged_bytes
    assert len(json.loads(charged_bytes)["debits"]) == 166
    assert {p.name: p.read_bytes() for p in (tmp_path/q["output"]).iterdir()} == refusal_files


class RuntimeMetadataFixture(MetadataFixture):
    """Hypothetical d metadata in scratch; no Original execution or raw arrays."""
    def __init__(self, root):
        from tests.test_c_banc_contact_runtime_journal import (
            copy_runtime_recovery_metadata,
        )
        self.root, self.seen = root, {}
        self.contract_spec = gate.spec_for(gate.STAGE_SPEC, gate.ROOT)
        self.contract = self.read(self.contract_spec)
        self.roster, self.grant = self.read(gate.ROSTER), self.read(gate.GRANT)
        self.source = gate.spec_for(gate.D_SOURCE_PATH, gate.ROOT)
        self.retry = write(root, gate.D_RUNTIME_PATH, gate.runtime_descriptor(self.contract_spec))
        self.before = copy_runtime_recovery_metadata(root)
        for spec in gate.D_HISTORY_PINS.values():
            if spec["path"].endswith(".json"):
                self.read(spec, 524288)
        for i, call in enumerate(gate.D_L_CALLS):
            self.add_generated(call, i)
        self.recompute(self.before)
        self.qualification = self.make_qualification()
        self.before["direct_science_qualification"] = self.qualification
        self.body = self.science(gate.ROLES[2])
        self.audit_before = copy.deepcopy(self.before)
        self.add_body(self.audit_before)
        self.recompute(self.audit_before)
        self.audit = self.science(gate.ROLES[3])
        self.copy_seen()

    def add_generated(self, call, index):
        caps = ((8, 4, 4), (12, 8, 4), (96, 88, 8))[index]
        q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "role": ("lint", "numerical", "execution")[index],
             "call_id": call, "source_freeze": self.source, "limits": dict(zip(("outer_s", "child_s", "cleanup_s"), caps, strict=True))}
        request = write(self.root, gate.BASE+"/generated-runtime-"+call+"-request.json", q)
        cost = self.proof(call, "generated", request)
        spec = write(self.root, gate.BASE+"/generated-runtime-"+call+"-cost.json", cost)
        self.before["debits"].append({"cost_id": call, "mode": "generated", "actual_outer_full_return_s": .125, "actual_returned_exit_code": 0, "descriptor": spec})
        self.before["direct_science_calls"][call] = {"request": request, "state": "charged", "mode": "generated", "reserved_s": caps[0]}

    def recompute(self, before):
        policy = self.read(before["direct_science_policy"])
        generated, original = 0., 0.
        for debit in before["debits"][118:]:
            if debit["mode"] == "generated":
                generated += float(debit["actual_outer_full_return_s"])
        for debit in before["debits"][165:]:
            if debit["mode"] == "original":
                original += float(debit["actual_outer_full_return_s"])
        before["direct_science_generated_used_s"] = generated
        before["generated_stage_used_s"] = float(policy["generated_stage_before_s"]) + generated
        before["generated_shared_cumulative_used_s"] = float(policy["generated_shared_before_s"]) + generated
        before["generated_shared_remaining_s"] = 1700. - policy["generated_shared_baseline_s"] - before["generated_stage_used_s"]
        for key, carry in (("original_cumulative_used_s", "original_cumulative_before_s"),
                           ("original_stage_used_s", "original_stage_before_s"), ("startup_ray_original_local_used_s", "science_local_used_before_s")):
            before[key] = self.grant[carry] + original
        before["original_remaining_s"] = 16200. - self.grant["original_cumulative_before_s"] - original

    def make_qualification(self):
        import xml.etree.ElementTree as ET
        historical = self.read(gate.D_HISTORY_PINS["qualification_x"])
        names = gate.runtime_test_names(historical)
        items = []
        for i, kind in enumerate(("math", "gate")):
            values = names[kind]
            classes = ([("tests.test_c_banc_contact_learning", 9), ("tests.test_c_banc_contact_body_retry", 2), ("tests.test_c_banc_contact_runtime_journal", 3)]
                       if i == 0 else [("tests.test_c_banc_contact_learning", 9), ("tests.test_c_banc_contact_metadata", 3),
                                      ("tests.test_c_banc_startup_ray_execution", 9), ("tests.test_c_banc_contact_body_retry", 4),
                                      ("tests.test_c_banc_contact_runtime_journal", 5)])
            root = ET.Element("testsuites")
            suite = ET.SubElement(root, "testsuite", tests=str(len(values)), failures="0", errors="0", skipped="0")
            case_classes = [name for name, count in classes for _ in range(count)]
            for name, classname in zip(values, case_classes, strict=True):
                ET.SubElement(suite, "testcase", name=name, classname=classname)
            path = gate.BASE+"/parent-contact-learning-"+kind+("67" if i == 0 else "68")+"-junit.xml"
            target = self.root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(ET.tostring(root))
            items.append({"names": values, "count": len(values), "file": pin_file(target, path)})
        return write(self.root, gate.BASE+"/generated-runtime-qualified-y.json",
                     {"schema": "flylab.m3.direct-science.qualification.v1", "source_freeze": self.source,
                      "actual_costs": [d["descriptor"] for d in self.before["debits"][277:280]], "tests": items,
                      "fixture_only": True, "Original_execution": "NOT_RUN"})

    def science(self, role):
        r = self.read(self.retry)
        manifest_spec = r["historical_manifests"][1] if role == gate.ROLES[2] else self.body_manifest
        manifest = self.read(manifest_spec)
        call = gate.D_RUNTIME_CALLS[gate.ROLES[2:].index(role)]
        before = self.before if role == gate.ROLES[2] else self.audit_before
        return {"schema": gate.SCHEMA+".request.v1", "mode": "original", "role": role, "call_id": call,
                "limits": self.contract["limits"][role], "contract": self.contract_spec, "body_retry": self.retry,
                "source_freeze": self.source, "qualification": self.qualification,
                "grant": gate.GRANT, "inputs": gate.INPUTS, "roster": gate.ROSTER, "local_debit_offset": 165,
                "ledger_path": gate.LEDGER, "shared_original_lock_path": gate.SHARED_LOCK,
                "historical_manifest": self.read(self.roster["previous_geometry_roster"])["historical_manifest"],
                "before_ledger": write(self.root, gate.BASE+"/generated-d-before-"+role+".json", before),
                "resource_increment_s": 0, "output": gate.BASE+"/parent-"+call+"/result",
                "artifacts": dict(manifest["artifacts"], manifest=manifest_spec)}

    def copy_seen(self):
        super().copy_seen()
        x = self.read(gate.D_HISTORY_PINS["archive_x"])
        y = self.read(gate.RUNTIME_RECOVERY_PINS["archive_y"])
        qx = self.read(gate.D_HISTORY_PINS["qualification_x"])
        for spec in [item["as_executed"] for item in [*x["sources"], *y["sources"]]] + [item["file"] for item in qx["tests"]] + [gate.D_HISTORY_PINS["stderr_c"]]:
            target = self.root/spec["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gate.leaf(spec, gate.ROOT, 1 << 20))

    def archive(self, item):
        relative = Path(item["archive_path"]).relative_to(gate.ROOT)
        scratch = self.root/relative
        if scratch.is_file():
            raw = scratch.read_bytes()
            assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
            return raw
        return MetadataFixture.archive(item)


def test_gate_runtime_d_history_single_call_and_strict_negatives(retry_tmp, monkeypatch):
    root = retry_tmp
    fixture = RuntimeMetadataFixture(root)
    positives = ((fixture.body, fixture.before), (fixture.audit, fixture.audit_before))
    cases = []
    for q, before in positives:
        assert gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive) > 0
        assert before["debits"][275]["cost_id"] == gate.C_RETRY_CALLS[0] and before["debits"][275]["actual_returned_exit_code"] == 1
        cases.append({"q": q, "before": write(root, gate.BASE+"/runtime-positive-"+q["role"]+".json", before), "pass": True})
    fixture.copy_seen()
    for kind in ("missing", "suffix", "b_replay", "c_replay", "source_x", "extra_generated", "extra_original", "prefix", "prior_call", "cap", "stale_qualification", "candidate"):
        q, before = copy.deepcopy(fixture.body), copy.deepcopy(fixture.before)
        if kind == "missing":
            del q["body_retry"]
        elif kind == "suffix":
            q["call_id"] += "-retry"
        elif kind == "b_replay":
            q["call_id"] = gate.RETRY_CALLS[0]
        elif kind == "c_replay":
            q["call_id"] = gate.C_RETRY_CALLS[0]
        elif kind == "source_x":
            q["source_freeze"] = gate.D_HISTORY_PINS["source_x"]
        elif kind in ("extra_generated", "extra_original"):
            extra = copy.deepcopy(before["debits"][-1])
            extra["cost_id"] = "extra-current"
            extra["mode"] = "generated" if kind == "extra_generated" else "original"
            before["debits"].append(extra)
        elif kind == "prefix":
            before["debits"][0]["cost_id"] = "changed-old"
        elif kind == "prior_call":
            before["direct_science_calls"][gate.C_RETRY_CALLS[0]]["state"] = "admitted"
        elif kind == "cap":
            q["limits"]["input_file_cap_bytes"] += 1
        elif kind == "stale_qualification":
            q["qualification"] = gate.D_HISTORY_PINS["qualification_x"]
        else:
            wrong = fixture.read(fixture.retry)
            wrong["candidate"]["palette"]["full_native_palette_count"] += 1
            q["body_retry"] = write(root, gate.BASE+"/runtime-wrong-candidate.json", wrong)
        with pytest.raises((ValueError, KeyError)):
            gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive)
        cases.append({"q": q, "before": write(root, gate.BASE+"/runtime-negative-"+kind+".json", before), "pass": False})
    # Future audit needs a genuine successful body; diagnostic failure cannot substitute.
    for kind in ("exit1", "invalid_report", "candidate_relabel", "archive14", "c_first_error", "c_outer"):
        q, before = copy.deepcopy(fixture.audit), copy.deepcopy(fixture.audit_before)
        if kind == "exit1":
            before["debits"][280]["actual_returned_exit_code"] = 1
            with pytest.raises((ValueError, KeyError)):
                gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive)
            cases.append({"q": q, "before": write(root, gate.BASE+"/runtime-negative-exit1.json", before), "pass": False})
            continue
        spec = (fixture.body_report if kind in ("invalid_report", "candidate_relabel")
                else gate.D_HISTORY_PINS[{"archive14": "archive_x", "c_first_error": "actual_c", "c_outer": "outer_c"}[kind]])
        original = (root/spec["path"]).read_bytes()
        wrong = fixture.read(spec)
        if kind == "invalid_report":
            wrong["run_valid"] = False
        elif kind == "candidate_relabel":
            wrong["candidate"]["source_freeze"] = fixture.source
        elif kind == "archive14":
            wrong["sources"].pop()
        elif kind == "c_first_error":
            wrong["terminal_failure"]["first_error"]["message"] = "scientific rejection"
        else:
            wrong["timeout"] = True
        write(root, spec["path"], wrong)
        with pytest.raises((ValueError, KeyError, AssertionError)):
            gate.validate_history(before, fixture.grant, fixture.roster, q, fixture.read, fixture.closed, fixture.archive)
        (root/spec["path"]).write_bytes(original)
        cases.append({"q": q, "before": cases[1]["before"], "pass": False, "tamper": spec, "replacement": wrong})

    # Full actual production Native functions; no archived Python/Native imports.
    source = (gate.ROOT/gate.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
    definitions = source[source.index("$ErrorActionPreference="):source.index("$ledger=Get-Content -LiteralPath (Safe $ledgerName)")]
    anchor = "$taskRoot=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert definitions.count(anchor) == 1
    definitions = definitions.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('"+root.as_posix().replace("'", "''")+"')")
    contract = (gate.ROOT/gate.BASE/"parent-contact-learning-contract-native.ps1").read_text(encoding="utf8")
    case_spec = write(root, gate.BASE+"/runtime-native-cases.json", cases)
    script = root/"runtime-native-history.ps1"
    body = "$cases=Read-Pin ('"+json.dumps(case_spec).replace("'", "''")+"'|ConvertFrom-Json -AsHashtable)\n" + r"""
foreach($case in $cases){
  $backup=$null
  if($case.Contains('tamper')){$p=Safe $case.tamper.path;$backup=[IO.File]::ReadAllBytes($p);[IO.File]::WriteAllText($p,($case.replacement|ConvertTo-Json -Depth 100),$utf8)}
  $passed=$false
  try{$q=$case.q;$contract=Read-Pin $q.contract;$r=Contact-RuntimeBinding $q $contract;$before=Read-Pin $case.before 524288;$g=Read-Pin $q.grant;$null=Contact-RuntimeHistory $q $before $before $contract $g $r;$passed=$true}
  catch{if($case.pass){throw}}
  finally{if($null -ne $backup){[IO.File]::WriteAllBytes($p,$backup)}}
  Need ($passed -ceq $case.pass) 'Native exact d hypothetical positive/strict negative outcome'
}
'RUNTIME_D_HISTORY_PASS'
"""
    script.write_text(definitions+"\n"+contract+"\n"+body, encoding="utf8")
    observed = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script)], capture_output=True, timeout=24,
                              check=False, creationflags=0x08000000 if os.name == "nt" else 0)
    assert observed.returncode == 0 and b"RUNTIME_D_HISTORY_PASS" in observed.stdout, observed.stderr

    # The complete d body pre-CreateProcess refusal/once-account path uses only
    # generated scratch metadata. It cannot admit or execute an Original worker.
    from tools import banc_m3_startup_ray_supervisor as supervisor
    science = copy.deepcopy(fixture.body)
    refused_source = write(root, fixture.source["path"], {"fixture_only": True, "reason": "intentional current Source refusal"})
    science["source_freeze"] = refused_source
    science_spec = write(root, gate.BASE+"/generated-refused-d-science.json", science)
    call, limits = gate.D_RUNTIME_CALLS[0], science["limits"]
    q = {"schema": "flylab.m3.direct-science.supervision.v1", "mode": "original", "role": science["role"], "call_id": call,
         "science_request": science_spec, "source_freeze": refused_source, "grant": gate.GRANT, "ledger_path": gate.LEDGER,
         "shared_original_lock_path": gate.SHARED_LOCK, "output": science["output"]+"-supervision",
         "limits": {k: limits[k] for k in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes")}}
    request = write(root, gate.BASE+"/generated-refused-d-owner.json", q)
    debits = [{"cost_id": "generated-old-"+str(i), "mode": "generated"} for i in range(165)]
    before = write(root, gate.BASE+"/generated-d-refusal-policy-before.json", {"debits": debits})
    policy = write(root, gate.BASE+"/generated-d-refusal-policy.json",
                   {"schema": "flylab.m3.direct-science.policy.v1", "generated_local_limit_s": 300,
                    "original_authorized": False, "before_ledger": before, "local_debit_offset": 165})
    ledger = {"debits": debits, "direct_science_policy": policy,
              "direct_science_calls": {call: {"request": request, "state": "admitted", "mode": "original", "reserved_s": 300}},
              "startup_ray_original_workers": {}, "startup_ray_original_reservations": {}, "startup_ray_original_pending_actuals": {},
              "startup_ray_original_blocking_holds": {}, "direct_science_pending_actuals": {}, "direct_science_blocking_holds": {},
              "original_jobs_running": 0, "original_reserved_s": 0,
              "startup_ray_original_grants": {fixture.grant["grant_id"]:
                  {"grant": gate.GRANT, "source_freeze": fixture.grant["source_freeze"], "inputs": fixture.grant["inputs"],
                   "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}}}
    write(root, gate.LEDGER, ledger)
    ledger_path = root/gate.LEDGER
    admitted_bytes = ledger_path.read_bytes()
    original_admission, acquired_errors = gate.Admission, []
    def acquiring_admission(*args, **kwargs):
        try:
            return original_admission(*args, **kwargs)
        except Exception as error:
            acquired_errors.append(error)
            raise
    def no_job(*_args):
        raise AssertionError("No Job/CreateProcess or numeric worker in exact d refusal")
    with monkeypatch.context() as changed:
        changed.setattr(supervisor, "creation_ticks", lambda _pid: 999)
        changed.setattr(supervisor.time, "perf_counter", lambda: 101.)
        changed.setattr(supervisor, "OwnedJob", no_job)
        changed.setattr(gate, "Admission", acquiring_admission)
        # Every routing drift stops before outer creation or any accounting mutation.
        for kind in ("role", "suffix", "missing", "descriptor", "source_path", "source_pin"):
            wrong, wrong_science = copy.deepcopy(q), copy.deepcopy(science)
            if kind == "role":
                wrong_science["role"] = gate.ROLES[3]
            elif kind == "suffix":
                wrong["call_id"] += "-retry"
            elif kind == "missing":
                del wrong_science["body_retry"]
            elif kind == "descriptor":
                wrong_science["body_retry"]["path"] += ".substitute.json"
            elif kind == "source_path":
                wrong_science["source_freeze"] = gate.D_HISTORY_PINS["source_x"]
            else:
                wrong["source_freeze"]["sha256"] = "0"*64
            wrong["science_request"] = write(root, gate.BASE+"/generated-d-refusal-negative-"+kind+"-science.json", wrong_science)
            wrong_spec = write(root, gate.BASE+"/generated-d-refusal-negative-"+kind+"-owner.json", wrong)
            with pytest.raises((ValueError, KeyError)):
                supervisor.contact_pre_spawn_refusal(root, wrong_spec, wrong, RuntimeError("original-refusal-sentinel"), entry=100.)
            assert not (root/q["output"]).exists() and ledger_path.read_bytes() == admitted_bytes
        outer = supervisor.supervise(request["path"], root=root, entry=100.)
        assert len(acquired_errors) == 1 and outer["first_error"]["type"] == type(acquired_errors[0]).__name__
        assert outer["first_error"]["message"] == str(acquired_errors[0])
        assert outer["pre_spawn_refused"] and not outer["job_created"] and not outer["worker_started"]
        assert outer["worker_pid"] is None and outer["worker_creation_ticks"] is None
        assert outer["all_owned_ended"] and outer["reservation_released"] and outer["stdout_eof"] and outer["stderr_eof"]
        assert outer["unknown_owned_identities"] == {} and outer["original_payload_accesses"] == 0
        assert outer["refusal_binding"]["body_retry"] == fixture.retry and outer["refusal_binding"]["source_freeze"] == refused_source
        assert ledger_path.read_bytes() == admitted_bytes
        for label in ("stdout", "stderr"):
            assert (root/q["output"]/(label+".bin")).read_bytes() == b""
        refusal_files = {path.name: path.read_bytes() for path in (root/q["output"]).iterdir()}
        with pytest.raises(ValueError, match="^refused call cannot publish twice$"):
            supervisor.supervise(request["path"], root=root, entry=100.)
        assert ledger_path.read_bytes() == admitted_bytes
        assert {path.name: path.read_bytes() for path in (root/q["output"]).iterdir()} == refusal_files

        assert os.name == "nt" and shutil.which("pwsh"), "Native exact d once-account requires Windows/pwsh; no skip"
        trace_name = gate.BASE+"/generated-refused-d-return.json"
        write(root, trace_name, {"schema": "flylab.m3.direct-science.genuine-return.v1", "phase": "FINAL", "first_error": None,
                                "dispatch_ms": 1000, "first_final_ms": 1125, "full_return_s": .125, "exit_code": 1,
                                "request": request, "source_freeze": refused_source, "fixture_only": True, "Original_execution": "NOT_RUN"})
        account_source = (gate.ROOT/gate.BASE/"parent-direct-science-account.ps1").read_text(encoding="utf8")
        assert account_source.count(anchor) == 1
        account = root/"generated-d-refusal-account.ps1"
        account.write_text(account_source.replace(anchor, "$taskRoot=[IO.Path]::GetFullPath('"+root.as_posix().replace("'", "''")+"')"), encoding="utf8")
        helper = root/gate.BASE/"parent-contact-learning-contract-native.ps1"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_bytes((gate.ROOT/gate.BASE/"parent-contact-learning-contract-native.ps1").read_bytes())
        command = [shutil.which("pwsh"), "-NoProfile", "-File", str(account), "-Action", "commit", "-TracePath", trace_name]
        first = subprocess.run(command, capture_output=True, timeout=8, check=False, creationflags=0x08000000)
        assert first.returncode == 0, first.stderr
        charged_bytes = ledger_path.read_bytes()
        after = json.loads(charged_bytes)
        assert len(after["debits"]) == 166 and after["debits"][:165] == debits
        assert after["direct_science_calls"][call]["state"] == "charged"
        cost = bounded(after["debits"][-1]["descriptor"], root)
        assert cost["body_retry"] == fixture.retry and cost["pre_spawn_refused"] and cost["exit_code"] == 1
        assert cost["original_payload_accesses"] == 0 and not cost["worker_started"] and not cost["job_created"]
        assert cost["refusal_binding"] == outer["refusal_binding"] and cost["charged_once"]
        duplicate = subprocess.run(command, capture_output=True, timeout=8, check=False, creationflags=0x08000000)
        assert duplicate.returncode != 0 and b"Fresh once commit" in duplicate.stderr
        assert ledger_path.read_bytes() == charged_bytes and len(json.loads(charged_bytes)["debits"]) == 166
        with pytest.raises(ValueError, match=r"^actual current admitted refused request\.state$"):
            supervisor.supervise(request["path"], root=root, entry=100.)
        assert ledger_path.read_bytes() == charged_bytes
        assert {path.name: path.read_bytes() for path in (root/q["output"]).iterdir()} == refusal_files
