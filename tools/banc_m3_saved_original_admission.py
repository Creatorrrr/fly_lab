"""Read-only saved Original CPU admission; no numerical or native science imports.

ROOT publishes the request/freeze/baseline and commits the genuine outer return.
This gate grants no neural replay, body work, capture, or gait acceptance authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
LEDGER = BASE + "/parent-e358-execution-ledger.json"
SHARED_LOCK = BASE + "/parent-e358-original-active.lock"
SCHEMA = "flylab.m3.startup-ray"
ROLE = "saved_startup_transfer"
OLD_ROLES = ("input_audit", "ray", "candidate_audit", "neutral_body", "candidate_body",
             "neutral_audit", "candidate_body_audit", "close")
GRANT_PATH = BASE + "/parent-startup-ray-original-grant-20261007-j.json"
GRANT_SHA = "7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f"
ROSTER_PATH = BASE + "/parent-after-startup-ray-saved-roster.json"
ROSTER_SHA = "07ad6d8d52f6e8b63b7aac016efbc62a0fbc4ad757e6f48781f9dfb9a5e360ef"
REQUIRED_SOURCES = ("tools/banc_m3_saved_original_admission.py",
                    "tools/analyze_c_banc_m3_startup_ray_transfer.py",
                    "tools/banc_m3_startup_ray_supervisor.py",
                    BASE + "/parent-direct-science-account.ps1")
LIMITS = {"outer_s": 120, "child_s": 100, "cleanup_s": 20,
          "host_bytes": 2 * 1024**3, "output_cap_bytes": 16 * 1024**2,
          "input_file_cap_bytes": 96 * 1024**2,
          "selected_expanded_cap_bytes": 64 * 1024**2,
          "member_cap_bytes": 16 * 1024**2, "worker_max": 1}


def require(ok, label):
    if not ok:
        raise ValueError(label)


def exact_value(actual, expected, label):
    require(type(actual) is type(expected), label + " type")
    if type(expected) is dict:
        require(set(actual) == set(expected), label + " fields")
        for key in expected:
            exact_value(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), label + " length")
        for a, b in zip(actual, expected, strict=True):
            exact_value(a, b, label)
    else:
        require(actual == expected, label)


def path(name, root):
    require(type(name) is str and name and "\\" not in name and not Path(name).is_absolute(), "relative path")
    target = root / name
    require(".." not in Path(name).parts and target.resolve().is_relative_to(root), "contained path")
    for part in (target, *target.parents):
        if part == root.parent:
            break
        if part.exists():
            st = part.lstat()
            require(not part.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400), "no reparse")
    return target


def json_load(raw):
    def parse_float(text):
        value = float(text)
        require(math.isfinite(value), "finite JSON float")
        return value

    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw.decode("utf8", errors="strict"), object_pairs_hook=pairs, parse_float=parse_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def descriptor(spec, cap, *, allow_empty=False):
    require(type(spec) is dict and set(spec) == {"path", "bytes", "sha256"}, "descriptor fields")
    require(type(spec["bytes"]) is int and (0 if allow_empty else 1) <= spec["bytes"] <= cap
            and type(spec["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", spec["sha256"]), "descriptor bounds")


def leaf(spec, root, cap=1048576, *, allow_empty=False):
    descriptor(spec, cap, allow_empty=allow_empty)
    p = path(spec["path"], root)
    st = p.stat()
    require(st.st_size == spec["bytes"], "exact metadata size")
    with p.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) == spec["bytes"] and len(raw) <= cap, "bounded metadata read")
    require(raw == p.read_bytes() and p.stat().st_mtime_ns == st.st_mtime_ns
            and hashlib.sha256(raw).hexdigest() == spec["sha256"], "exact hash/timestamp/readback")
    raw.decode("utf8", errors="strict")
    return raw


def spec_for(name, root, cap=1048576):
    p = path(name, root)
    st = p.stat()
    require(0 < st.st_size <= cap, "bounded metadata publication")
    with p.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) == st.st_size and p.stat().st_mtime_ns == st.st_mtime_ns, "stable metadata publication")
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def finite(value, label):
    require(type(value) is float and math.isfinite(value) and value >= 0, "finite double " + label)
    return value


def sources(freeze, root):
    require(freeze["schema"] == SCHEMA + ".source-freeze.v1"
            and freeze["ROOT_review_cycle_complete"] is True, "reviewed fresh narrow Source freeze")
    require(type(freeze["sources"]) is dict and all(name in freeze["sources"] for name in REQUIRED_SOURCES),
            "actual saved reader/admission/owner/account frozen")
    for name, spec in freeze["sources"].items():
        exact_value(name, spec["path"], "Source dictionary path")
        require(Path(name).suffix in (".py", ".ps1", ".js", ".json", ".txt", ".ini", ".md"), "text Source only")
        if Path(name).is_absolute():
            descriptor(spec, 2097152, allow_empty=True)
            p = Path(name)
            for part in (p, *p.parents):
                st = part.lstat()
                require(not part.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400), "Source no reparse")
            before = p.stat()
            require(before.st_size == spec["bytes"], "Source bytes")
            raw = p.read_bytes()
            require(raw == p.read_bytes() and p.stat().st_mtime_ns == before.st_mtime_ns
                    and hashlib.sha256(raw).hexdigest() == spec["sha256"], "Source cold fence")
            raw.decode("utf8", errors="strict")
        else:
            leaf(spec, root, 2097152, allow_empty=True)


def closed_cost(cost, debit, root):
    exact_value(cost["cost_id"], debit["cost_id"], "cost/debit identity")
    exact_value(cost["charged_once"], True, "once charged")
    seconds = finite(cost["actual_outer_full_return_s"], "actual outer return")
    require(seconds > 0, "positive actual return")
    exact_value(seconds, debit["actual_outer_full_return_s"], "ordered debit seconds")
    exact_value(cost["exit_code"], debit["actual_returned_exit_code"], "returned exit")
    require(type(cost["exit_code"]) is int, "typed exit")
    exact_value(cost["all_owned_ended"], True, "prior ownership closed")
    exact_value(cost["reservation_released"], True, "prior reservation released")
    outer = json_load(leaf(cost["outer"], root))
    for key in ("all_owned_ended", "reservation_released", "stdout_eof", "stderr_eof"):
        exact_value(outer[key], True, "actual prior " + key)
    exact_value(outer["unknown_owned_identities"], {}, "prior identities known")
    exact_value(outer["exit_code"], cost["exit_code"], "outer actual exit")
    trace = json_load(leaf(cost["trace"], root, 2097152))
    require(trace["schema"] == "flylab.m3.direct-science.genuine-return.v1"
            and trace["phase"] == "FINAL" and trace["first_error"] is None, "genuine prior FINAL")
    require(type(trace["dispatch_ms"]) is int and type(trace["first_final_ms"]) is int
            and trace["first_final_ms"] > trace["dispatch_ms"], "actual prior outer timestamps")
    exact_value((trace["first_final_ms"] - trace["dispatch_ms"]) / 1000.0, seconds, "actual full-return clock")
    exact_value(trace["full_return_s"], seconds, "exact prior full return")
    exact_value(trace["exit_code"], cost["exit_code"], "genuine prior exit")
    exact_value(trace["source_freeze"], cost["source_freeze"], "prior executed Source")
    exact_value(trace["request"], cost["supervisor_request"], "actual prior supervisor request")
    exact_value(outer["request"], trace["request"], "prior outer request")
    return cost


class Admission:
    """Metadata/Source fence before numerical import and selected Original reads."""
    def __init__(self, request, stage="pre_array", *, root=ROOT, entry=None):
        require(stage in ("pre_spawn", "pre_array"), "admission stage")
        if stage == "pre_array":
            # Infer the actual supervisor origin before any filesystem reads. The
            # worker's newer entry never supplies a restarted deadline.
            owner_entry = json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8"))["entry_counter_s"]
            finite(owner_entry, "actual owner ENTRY")
            require(owner_entry > 0, "positive actual owner ENTRY")
            caller_entry = owner_entry if entry is None else entry
            require(type(caller_entry) is float and math.isfinite(caller_entry)
                    and caller_entry >= owner_entry, "caller ENTRY follows actual owner ENTRY")
        else:
            require(type(entry) is float and math.isfinite(entry) and entry > 0, "caller actual ENTRY required")
            owner_entry = caller_entry = entry
        self.stage, self.entry, self.caller_entry = stage, owner_entry, caller_entry
        self.deadline()  # Before resolving/reading request, Source, or baseline.
        self.root = Path(root).resolve()
        self.request_spec = spec_for(request, self.root)
        self.request = json_load(leaf(self.request_spec, self.root))
        q = self.request
        require(q["schema"] == SCHEMA + ".saved-diagnosis.request.v1" and q["mode"] == "original"
                and q["role"] == ROLE, "saved Original role only")
        require(type(q["call_id"]) is str and re.fullmatch("startup-ray-[a-z0-9_-]{1,100}", q["call_id"]), "finite call identity")
        self.role, self.limits = ROLE, q["limits"]
        exact_value(self.limits, LIMITS, "fixed saved CPU bounds")
        exact_value(q["ledger_path"], LEDGER, "existing ledger")
        exact_value(q["shared_original_lock_path"], SHARED_LOCK, "shared Original lock")
        exact_value(q["resource_increment_s"], 0, "no resource increment")
        exact_value(q["local_debit_offset"], 165, "existing J accounting offset")
        exact_value(q["grant"], {"path": GRANT_PATH, "bytes": 2227, "sha256": GRANT_SHA}, "existing J grant metadata")
        exact_value(q["roster"], {"path": ROSTER_PATH, "bytes": 4377, "sha256": ROSTER_SHA}, "immutable saved roster")
        self.grant = json_load(leaf(q["grant"], self.root))
        self.freeze = json_load(leaf(q["source_freeze"], self.root))
        self.before = json_load(leaf(q["before_ledger"], self.root, 524288))
        self.roster_path = path(q["roster"]["path"], self.root)
        self.roster = json_load(leaf(q["roster"], self.root))
        require(self.roster["schema"] == "flylab.m3.after-startup-ray.saved-roster.v1"
                and self.roster["no_new_body_work"] is True, "saved acquisition roster")
        require(q["output"].startswith(BASE + "/parent-" + q["call_id"] + "/"), "finite call output")
        self.output = path(q["output"], self.root)
        g = self.grant
        for key, expected in (("schema", SCHEMA + ".original-grant.v1"), ("mode", "original"),
                              ("grant_id", "startup-ray-original-20261007-j"), ("local_debit_offset", 165),
                              ("additional_s", 6000), ("global_limit_s", 16200), ("stage_limit_s", 6300),
                              ("science_local_limit_s", 6000), ("worker_max", 1), ("automatic_restarts", 0),
                              ("reset", False), ("resource_increment_s", 0)):
            exact_value(g[key], expected, "preserved J quota " + key)
        exact_value(self.freeze["human_authorization"], g["human_authorization"], "same existing authority")
        require(len(self.before["debits"]) >= 173, "ROOT pinned current baseline after closed eight roles")
        self.check("initial")

    def deadline(self):
        """Every callback checks the same owner origin; no metadata reads here."""
        elapsed = time.perf_counter() - self.entry
        require(math.isfinite(elapsed) and elapsed >= 0, "same actual ENTRY clock")
        if elapsed >= LIMITS["child_s"]:
            raise TimeoutError("saved CPU same owner ENTRY child interval")
        return True

    def check(self, label="science"):
        """Deadline callback; named physical boundaries additionally cold-fence.

        Other labels perform only the deadline check and make no new Source,
        ownership, or admission-validation claim. Ordinary True means only that
        the current deadline check passed; it is not a cached admission PASS.
        """
        self.deadline()  # First action on every callback, including cold boundaries.
        require(type(label) is str, "boundary label")
        if label in ("initial", "pre_array", "publication"):
            self.fence(label)
            self.deadline()
        return True

    def fence(self, label):
        """Full cold metadata/Source/lease fence at explicit boundaries only."""
        self.deadline()
        require(label in ("initial", "pre_array", "publication"), "explicit cold boundary")
        q, g, root = self.request, self.grant, self.root
        for spec, expected, cap in ((self.request_spec, q, 1048576), (q["grant"], g, 1048576),
                                    (q["source_freeze"], self.freeze, 1048576), (q["roster"], self.roster, 1048576),
                                    (q["before_ledger"], self.before, 524288)):
            exact_value(json_load(leaf(spec, root, cap)), expected, "fresh exact metadata")
        sources(self.freeze, root)  # Bind stdlib physical identity helper before its import.
        live = json_load(leaf(spec_for(LEDGER, root, 524288), root, 524288))
        exact_value(live["debits"], self.before["debits"], "all debits unchanged before current once return")
        for key in ("original_cumulative_used_s", "original_stage_used_s", "startup_ray_original_local_used_s",
                    "original_remaining_s"):
            finite(live[key], key)
            exact_value(live[key], self.before[key], "current exact pinned usage")
        for key, value in (("original_cumulative_limit_s", 16200), ("original_stage_cap_s", 6300),
                           ("startup_ray_original_local_limit_s", 6000)):
            exact_value(live[key], value, "same existing live quota")
        for key in ("startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds", "e360_pending_actuals",
                    "e360_blocking_holds", "direct_science_pending_actuals", "direct_science_blocking_holds",
                    "e359_reservations", "e360_reservations", "direct_science_reservations", "direct_science_workers"):
            exact_value(live[key], {}, "no pending/unrelated owner " + key)
        event = {"grant": q["grant"], "source_freeze": g["source_freeze"], "inputs": g["inputs"],
                 "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}
        exact_value(live["startup_ray_original_grants"][g["grant_id"]], event, "old grant event retained")
        prior_original = [d for d in self.before["debits"][165:] if d["mode"] == "original"]
        prior_total = 0.0
        for debit, role in zip(prior_original[:8], OLD_ROLES, strict=True):
            require(debit["mode"] == "original", "eight original prior roles")
            cost = closed_cost(json_load(leaf(debit["descriptor"], root)), debit, root)
            require(cost["schema"] == SCHEMA + ".full-return.v1" and cost["role"] == role
                    and cost["exit_code"] == 0, "closed actual J prefix, no role replay")
            exact_value(cost["grant"], q["grant"], "prior same J grant")
            for key, expected in (("source_freeze", g["source_freeze"]), ("inputs", g["inputs"]), ("local_debit_offset", 165)):
                exact_value(cost[key], expected, "historical cost anchor " + key)
            old_call = live["direct_science_calls"][cost["cost_id"]]
            exact_value(old_call["state"], "charged", "prior role charged")
            exact_value(old_call["request"], cost["supervisor_request"], "prior actual call")
            prior_total += cost["actual_outer_full_return_s"]
        require(len(prior_original) <= 10, "finite earlier saved failures only")
        for index, debit in enumerate(prior_original[8:], start=1):
            cost = closed_cost(json_load(leaf(debit["descriptor"], root)), debit, root)
            exact_value(cost["cost_id"], f"startup-ray-20261007-j-saved-transfer-{index:02d}", "recorded earlier saved call")
            require(cost["schema"] == SCHEMA + ".saved-diagnosis.full-return.v1" and cost["exit_code"] == 1,
                    "genuine closed saved technical failure preserved")
            if index == 1:
                require(cost["pre_array_refused"] is True and cost["original_payload_accesses"] == 0,
                        "first actual pre-array refusal only")
            exact_value(cost["grant"], q["grant"], "saved refusal same grant")
            exact_value(cost["inputs"], q["roster"], "saved refusal same roster")
            exact_value(live["direct_science_calls"][cost["cost_id"]]["state"], "charged", "saved refusal charged")
            prior_total += cost["actual_outer_full_return_s"]
        # Match the Native writer's ordered J subtotal plus preserved J carry.
        cumulative_anchor = finite(g["original_cumulative_before_s"], "J cumulative anchor")
        stage_anchor = finite(g["original_stage_before_s"], "J stage anchor")
        carry = finite(g["science_local_used_before_s"], "J prior scope carry")
        for key, expected in (("original_cumulative_used_s", cumulative_anchor + prior_total),
                              ("original_stage_used_s", stage_anchor + prior_total),
                              ("startup_ray_original_local_used_s", carry + prior_total),
                              ("original_remaining_s", 16200.0 - cumulative_anchor - prior_total)):
            exact_value(self.before[key], expected, "closed J ordered baseline " + key)
        for key, cap in (("original_cumulative_used_s", 16200), ("original_stage_used_s", 6300),
                         ("startup_ray_original_local_used_s", 6000)):
            require(live[key] + LIMITS["outer_s"] <= cap, "CPU full-return reserve " + key)
        call = live["direct_science_calls"][q["call_id"]]
        require(call["state"] == "admitted" and call["mode"] == "original", "ROOT admitted actual saved call")
        exact_value(call["reserved_s"], 120, "actual saved reservation")
        outer = json_load(leaf(call["request"], root))
        for key, value in (("schema", "flylab.m3.direct-science.supervision.v1"), ("role", ROLE), ("mode", "original"),
                           ("call_id", q["call_id"]), ("science_request", self.request_spec), ("grant", q["grant"]),
                           ("source_freeze", q["source_freeze"]), ("output", q["output"] + "-supervision"), ("ledger_path", LEDGER),
                           ("shared_original_lock_path", SHARED_LOCK)):
            exact_value(outer[key], value, "actual outer binding " + key)
        for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
            exact_value(outer["limits"][key], LIMITS[key], "outer saved limit")
        from tools import banc_m3_startup_ray_supervisor as supervisor
        lock_path = path(SHARED_LOCK, root)
        supervision_output = path(outer["output"], root)
        workers, reservations = live["startup_ray_original_workers"], live["startup_ray_original_reservations"]
        if self.stage == "pre_spawn":
            exact_value(workers, {}, "no active Original worker")
            exact_value(reservations, {}, "no active Original reservation")
            require(not lock_path.exists() and not self.output.exists() and not supervision_output.exists(), "fresh closed shared exclusion/output")
        else:
            require(type(workers) is dict and set(workers) == {q["call_id"]}, "one actual saved CPU worker")
            w = workers[q["call_id"]]
            require(set(w) == {"request", "grant", "state", "worker_max", "reserved_s", "lock_token",
                               "worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"}, "complete actual worker lease")
            for key, value in (("request", self.request_spec), ("grant", q["grant"]), ("state", "begun"),
                               ("worker_max", 1), ("reserved_s", 120), ("worker_pid", os.getpid())):
                exact_value(w[key], value, "actual worker lease " + key)
            for key in ("worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"):
                require(type(w[key]) is int and w[key] > 0, "actual positive Native identity")
            require(w["owner_pid"] != w["worker_pid"], "distinct supervisor/worker")
            for prefix in ("owner", "worker"):
                exact_value(supervisor.creation_ticks(w[prefix + "_pid"]), w[prefix + "_creation_ticks"], "actual creation ticks")
            exact_value(w["lock_token"]["path"], SHARED_LOCK, "same shared lock token")
            token = json_load(leaf(w["lock_token"], root))
            exact_value(token["schema"], "flylab.m3.direct-science.lock-owner.v1", "actual lock schema")
            exact_value(json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8")), token, "actual owner environment token")
            for key, expected in (("call_id", q["call_id"]), ("request", call["request"]),
                                  ("owner_pid", w["owner_pid"]), ("owner_creation_ticks", w["owner_creation_ticks"])):
                exact_value(token[key], expected, "exact owner token " + key)
            owner_entry = finite(token["entry_counter_s"], "same owner ENTRY")
            require(self.caller_entry >= owner_entry, "caller ENTRY follows actual owner ENTRY")
            exact_value(owner_entry, self.entry, "same initially inferred supervisor origin")
            require(supervisor.in_named_job(token["job_name"]), "actual worker in same named owned Job")
            exact_value(reservations, {q["call_id"]: {"request": self.request_spec, "grant": q["grant"], "reserved_s": 120}}, "sole saved Original reservation")
            require(supervision_output.is_dir(), "actual supervisor-created outer output")
            if label == "initial":
                require(not self.output.exists(), "fresh reader output; worker alone creates it")
        total_output = 0
        for directory in (self.output, supervision_output):
            if directory.exists():
                require(directory.is_dir(), "output directory")
                for item in directory.rglob("*"):
                    safe_item = path(item.relative_to(root).as_posix(), root)
                    if safe_item.is_file():
                        total_output += safe_item.stat().st_size
        require(total_output <= LIMITS["output_cap_bytes"], "combined bounded reader and opaque supervisor output")
        active = self.stage == "pre_array"
        exact_value(live["original_jobs_running"], 1 if active else 0, "shared Original worker count")
        exact_value(live["original_reserved_s"], 120 if active else 0, "shared Original reserved seconds")
        sources(self.freeze, root)
        self.deadline()
        return True
