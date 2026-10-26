"""Sibling tracking admission. Historical J8/saved admission contracts stay intact."""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

from tools.banc_m3_saved_original_admission import (
    closed_cost,
    descriptor,
    exact_value,
    json_load,
    leaf,
    path,
    require,
    spec_for,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
LEDGER = BASE + "/parent-e358-execution-ledger.json"
SHARED_LOCK = BASE + "/parent-e358-original-active.lock"
SCHEMA = "flylab.m3.startup-tracking"
ROLES = ("tracking_neutral", "tracking_candidate", "tracking_diagnosis")
CALLS = ("tracking-capture-20261007-a-neutral", "tracking-capture-20261007-a-candidate", "tracking-capture-20261007-a-diagnosis")
OLD_ROLES = ("input_audit", "ray", "candidate_audit", "neutral_body", "candidate_body", "neutral_audit", "candidate_body_audit", "close")
GRANT = {"path": BASE + "/parent-startup-ray-original-grant-20261007-j.json", "bytes": 2227,
         "sha256": "7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f"}
INPUTS = {"path": BASE + "/parent-startup-ray-original-inputs.json", "bytes": 210093,
          "sha256": "4bdd075fc9980a275319fc0a26210c15335050ff7e96bc2fd29eb0ec5aeda151"}
ROSTER = {"path": BASE + "/parent-tracking-input-roster-20261007.json", "bytes": 5226,
          "sha256": "c67c1ae9714524ae834858f918dcffcceb6f8a3be0af0aac02bb58dd24dc13c8"}
REQUIRED_SOURCES = ("tools/banc_m3_startup_tracking_admission.py", "tools/observe_c_banc_m3_startup_tracking.py",
                    "tools/analyze_c_banc_m3_startup_tracking.py", "tests/test_c_banc_startup_tracking.py",
                    "tests/test_banc_m3_startup_tracking_admission.py", "tools/banc_m3_saved_original_admission.py",
                    "tools/banc_m3_startup_ray_supervisor.py", "tools/verify_c_banc_m3_startup_ray.py",
                    "tools/verify_c_banc_m3_common_descent_body.py", "tools/verify_c_banc_m3_startup_baseline.py",
                    "tools/verify_c_banc_m3_candidate_body_handoff.py", BASE + "/parent-direct-science-account.ps1",
                    BASE + "/parent-direct-science-genuine-collector.js", BASE + "/parent-tracking-capture-prepare-native.ps1")


def limits_for(role):
    require(role in ROLES, "only three new tracking roles")
    diagnosis = role == "tracking_diagnosis"
    result = {"outer_s": 120 if diagnosis else 300, "child_s": 100 if diagnosis else 280,
              "internal_s": 80 if diagnosis else 250, "cleanup_s": 20,
              "host_bytes": (2 if diagnosis else 24) * 1024**3,
              "output_cap_bytes": (16 if diagnosis else 96) * 1024**2,
              "input_file_cap_bytes": 96 * 1024**2, "selected_expanded_cap_bytes": 208 * 1024**2,
              "member_cap_bytes": 16 * 1024**2, "worker_max": 1}
    if not diagnosis:
        result.update(GPU_delta_plus_allowance_cap_bytes=8 * 1024**3, sample_cap=3000)
    return result


def finite(value, label):
    require(type(value) is float and math.isfinite(value) and value >= 0, "finite double " + label)
    return value


def current_sources(freeze, root, check):
    require(freeze["schema"] == SCHEMA + ".source-freeze.v1" and freeze["ROOT_review_cycle_complete"] is True, "reviewed current capture Source")
    require(type(freeze["sources"]) is dict and all(name in freeze["sources"] for name in REQUIRED_SOURCES), "complete production capture closure")
    for name, spec in freeze["sources"].items():
        check()
        exact_value(name, spec["path"], "Source key/path")
        require(Path(name).suffix in (".py", ".ps1", ".js", ".json", ".txt", ".ini", ".md"), "finite text Source suffix")
        if Path(name).is_absolute():
            descriptor(spec, 2097152, allow_empty=True)
            p = Path(name)
            for part in (p, *p.parents):
                st = part.lstat()
                require(not part.is_symlink() and not getattr(st, "st_file_attributes", 0) & 0x400, "Source no reparse")
            before = p.stat()
            require(before.st_size == spec["bytes"], "Source size")
            raw = p.read_bytes()
            require(raw == p.read_bytes() and p.stat().st_mtime_ns == before.st_mtime_ns
                    and hashlib.sha256(raw).hexdigest() == spec["sha256"], "fresh absolute Source")
            raw.decode("utf8", errors="strict")
        else:
            leaf(spec, root, 2097152, allow_empty=True)
        check()


def validate_history(before, grant, roster, q, read, closed):
    """All earlier genuine Original returns, including both failed saved attempts."""
    require(type(before["debits"]) is list and len(before["debits"]) >= 187, "current baseline preserves paid prefix")
    require(len({d["cost_id"] for d in before["debits"]}) == len(before["debits"]), "unique existing once costs")
    role_index = ROLES.index(q["role"])
    originals = [d for d in before["debits"][165:] if d["mode"] == "original"]
    require(len(originals) == 11 + role_index, "J8 plus all saved03 plus exact tracking predecessors")
    total = 0.0
    for index, debit in enumerate(originals):
        cost = closed(read(debit["descriptor"]), debit)
        exact_value(cost["grant"], q["grant"], "same preserved J grant")
        if index < 8:
            require(cost["schema"] == "flylab.m3.startup-ray.full-return.v1" and cost["role"] == OLD_ROLES[index]
                    and cost["exit_code"] == 0, "genuine successful J8 prefix")
            exact_value(cost["source_freeze"], grant["source_freeze"], "historical J freeze never rebound")
            exact_value(cost["inputs"], grant["inputs"], "historical J inputs preserved")
        elif index < 11:
            attempt = index - 7
            require(cost["schema"] == "flylab.m3.startup-ray.saved-diagnosis.full-return.v1"
                    and cost["cost_id"] == f"startup-ray-20261007-j-saved-transfer-{attempt:02d}"
                    and cost["exit_code"] == (0 if attempt == 3 else 1), "all saved01/02 failures and03 success preserved")
            if attempt == 1:
                require(cost["pre_array_refused"] is True and cost["original_payload_accesses"] == 0, "honest saved01 no-array refusal")
        else:
            predecessor = index - 11
            require(cost["schema"] == SCHEMA + ".full-return.v1" and cost["role"] == ROLES[predecessor]
                    and cost["cost_id"] == CALLS[predecessor] and cost["exit_code"] == 0
                    and cost["cap_exceeded"] is False, "one closed successful predecessor; failed blocks next")
            exact_value(cost["inputs"], INPUTS, "tracking same historical numeric binder")
            exact_value(cost["roster"], ROSTER, "tracking same current acquisition roster")
            exact_value(cost["source_freeze"], q["source_freeze"], "same actually qualified capture Source")
            spec = q["prior_captures"][ROLES[predecessor]]
            manifest = read(spec)
            require(manifest["schema"] == SCHEMA + ".capture-manifest.v1" and manifest["role"] == ROLES[predecessor]
                    and manifest["run_valid"] is True and manifest["capture_complete"] is True, "complete actual predecessor capture")
            exact_value(manifest["request"], cost["request"], "same paid capture request")
        exact_value(cost["local_debit_offset"], 165, "J offset preserved")
        total += finite(cost["actual_outer_full_return_s"], "ordered original full return")
    # Generated tail is untouched by this role. Validate its genuine once metadata too.
    for debit in before["debits"][165:]:
        require(debit["mode"] in ("original", "generated"), "no unknown cost mode")
        if debit["mode"] == "generated":
            cost = read(debit["descriptor"])
            require(cost["schema"] == "flylab.m3.direct-science.full-return.v1" and cost["charged_once"] is True, "all generated tail retained")
            exact_value(cost["cost_id"], debit["cost_id"], "generated cost/debit identity")
            exact_value(cost["actual_outer_full_return_s"], debit["actual_outer_full_return_s"], "generated actual seconds")
            exact_value(cost["exit_code"], debit["actual_returned_exit_code"], "generated actual exit")
    for key, expected in (("original_cumulative_used_s", finite(grant["original_cumulative_before_s"], "J anchor") + total),
                          ("original_stage_used_s", finite(grant["original_stage_before_s"], "stage anchor") + total),
                          ("startup_ray_original_local_used_s", finite(grant["science_local_used_before_s"], "preserved J carry") + total),
                          ("original_remaining_s", 16200.0 - grant["original_cumulative_before_s"] - total)):
        exact_value(before[key], expected, "complete ordered history " + key)
    exact_value(roster["original_inputs"], INPUTS, "roster exact binder")
    return total


class Admission:
    def __init__(self, request_path, stage="pre_array", *, root=ROOT, entry=None):
        require(stage in ("pre_array", "pre_spawn"), "tracking admission stage")
        if stage == "pre_array":
            owner = json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8"))
            origin = finite(owner["entry_counter_s"], "actual owner ENTRY")
            caller = origin if entry is None else entry
            require(type(caller) is float and math.isfinite(caller) and caller >= origin > 0, "same actual supervisor clock")
        else:
            origin = caller = finite(entry, "pre-spawn actual ENTRY")
            require(origin > 0, "positive actual ENTRY")
        self.stage, self.entry, self.caller_entry, self._deadline_s = stage, origin, caller, 280
        self.deadline()  # no path/Source/ledger read before actual owner deadline
        self.root = Path(root).resolve()
        self.request_spec = spec_for(request_path, self.root)
        self.request = self.read(self.request_spec)
        q = self.request
        require(q["schema"] == SCHEMA + ".request.v1" and q["mode"] == "original" and q["role"] in ROLES, "new three Original roles only")
        self.role, self.limits = q["role"], q["limits"]
        exact_value(self.limits, limits_for(self.role), "fixed role bounds")
        self._deadline_s = self.limits["internal_s"]
        self.deadline()
        exact_value(q["call_id"], CALLS[ROLES.index(self.role)], "one predetermined role call")
        exact_value(q["output"], BASE + "/parent-" + q["call_id"] + "/result", "fixed fresh output")
        exact_value(q["ledger_path"], LEDGER, "same actual ledger")
        exact_value(q["shared_original_lock_path"], SHARED_LOCK, "same shared Original exclusion")
        exact_value(q["local_debit_offset"], 165, "preserved J offset")
        exact_value(q["resource_increment_s"], 0, "no grant or reset")
        exact_value(q["grant"], GRANT, "historical grant")
        exact_value(q["inputs"], INPUTS, "historical Original binder")
        exact_value(q["roster"], ROSTER, "fixed current metadata roster")
        self.grant, self.inputs, self.roster = self.read(GRANT), self.read(INPUTS), self.read(ROSTER)
        self.manifest_spec = self.roster["historical_manifest"]
        exact_value(q["historical_manifest"], self.manifest_spec, "historical candidate manifest")
        self.read(self.manifest_spec)
        self.freeze, self.before = self.read(q["source_freeze"]), self.read(q["before_ledger"], 524288)
        self.output = path(q["output"], self.root)
        require(self.roster["schema"] == SCHEMA + ".roster.v1" and self.roster["role_order"] == list(ROLES)
                and self.roster["adoption"] is False and self.roster["resource_increment_s"] == 0, "fixed non-adoption roster")
        exact_value(self.roster["window_starts"], [100, 164, 228, 292], "fixed context windows")
        require(self.roster["fixed_candidate"]["alpha"] == .0001 and self.roster["capture_controls"] == [0, 300], "fixed rejected candidate/window")
        index = ROLES.index(self.role)
        exact_value(set(q["prior_captures"]), set(ROLES[:index]), "finite ordered prior capture manifests")
        if self.role == "tracking_diagnosis":
            exact_value(set(q["artifacts"]), set(ROLES[:2]), "finite diagnosis bundles")
            for prior in ROLES[:2]:
                bundle = q["artifacts"][prior]
                exact_value(set(bundle), {"manifest", "body_report", "control", "native", "trace", "tracking", "context"}, "selected saved raw artifacts")
                exact_value(bundle["manifest"], q["prior_captures"][prior], "same admitted complete manifest")
        else:
            exact_value(q["artifacts"], {}, "no unauthorized original input expansion")
        self.check("initial")

    def deadline(self):
        elapsed = time.perf_counter() - self.entry
        require(math.isfinite(elapsed) and elapsed >= 0, "same actual owner clock")
        if elapsed >= self._deadline_s:
            raise TimeoutError("tracking same owner ENTRY internal deadline")
        return True

    def read(self, spec, cap=1048576):
        self.deadline()
        result = json_load(leaf(spec, self.root, cap))
        self.deadline()
        return result

    def check(self, label="science"):
        self.deadline()
        require(type(label) is str, "callback label")
        if label in ("initial", "pre_array", "publication"):
            self.fence(label)
        self.deadline()
        return True

    def fence(self, label="publication"):
        self.deadline()
        require(label in ("initial", "pre_array", "publication"), "explicit cold tracking boundary")
        q, root = self.request, self.root
        for spec, value, cap in ((self.request_spec, q, 1048576), (q["grant"], self.grant, 1048576),
                                 (q["inputs"], self.inputs, 262144), (q["roster"], self.roster, 1048576),
                                 (q["source_freeze"], self.freeze, 1048576), (q["before_ledger"], self.before, 524288)):
            exact_value(self.read(spec, cap), value, "fresh exact admitted metadata")
        current_sources(self.freeze, root, self.deadline)
        g = self.grant
        for key, expected in (("schema", "flylab.m3.startup-ray.original-grant.v1"), ("mode", "original"),
                              ("grant_id", "startup-ray-original-20261007-j"), ("local_debit_offset", 165),
                              ("global_limit_s", 16200), ("stage_limit_s", 6300), ("science_local_limit_s", 6000),
                              ("worker_max", 1), ("automatic_restarts", 0), ("reset", False), ("resource_increment_s", 0)):
            exact_value(g[key], expected, "same existing J grant " + key)
        exact_value(self.freeze["human_authorization"], g["human_authorization"], "same existing human authority")
        live = self.read(spec_for(LEDGER, root, 524288), 524288)
        exact_value(live["debits"], self.before["debits"], "every historical/generated debit unchanged")
        for key in ("original_cumulative_used_s", "original_stage_used_s", "startup_ray_original_local_used_s", "original_remaining_s"):
            finite(live[key], key)
            exact_value(live[key], self.before[key], "same fresh before usage")
        for key, value in (("original_cumulative_limit_s", 16200), ("original_stage_cap_s", 6300), ("startup_ray_original_local_limit_s", 6000)):
            exact_value(live[key], value, "no quota reset")
        for key in ("startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds", "e360_pending_actuals", "e360_blocking_holds",
                    "direct_science_pending_actuals", "direct_science_blocking_holds", "e359_reservations", "e360_reservations", "direct_science_reservations", "direct_science_workers"):
            exact_value(live[key], {}, "closed unrelated scope " + key)
        event = {"grant": GRANT, "source_freeze": g["source_freeze"], "inputs": g["inputs"], "local_debit_offset": 165, "worker_max": 1, "automatic_restarts": 0}
        exact_value(live["startup_ray_original_grants"][g["grant_id"]], event, "historical registered J event retained")

        def prior_closed(cost, debit):
            self.deadline()
            result = closed_cost(cost, debit, root)
            self.deadline()
            exact_value(live["direct_science_calls"][cost["cost_id"]]["state"], "charged", "every prior actual once charged")
            return result

        validate_history(self.before, g, self.roster, q, self.read, prior_closed)
        for key, cap in (("original_cumulative_used_s", 16200), ("original_stage_used_s", 6300), ("startup_ray_original_local_used_s", 6000)):
            require(live[key] + self.limits["outer_s"] <= cap, "full actual-return budget reserve")
        qualification = self.read(q["qualification"])
        require(qualification["schema"] == "flylab.m3.direct-science.qualification.v1", "actual paid generated qualification")
        exact_value(qualification["source_freeze"], q["source_freeze"], "actually qualified current Source")
        exact_value(live["direct_science_qualification"], q["qualification"], "registered actual qualification")
        require(len(qualification["actual_costs"]) == 3, "lint/science-fixture/actual-Job qualification")
        for spec in qualification["actual_costs"]:
            cost = self.read(spec)
            require(cost["exit_code"] == 0 and cost["charged_once"] is True and cost["cap_exceeded"] is False
                    and cost["all_owned_ended"] is True and cost["reservation_released"] is True
                    and cost["cost_id"] in {d["cost_id"] for d in live["debits"]}, "genuine paid closed qualified role")
            exact_value(cost["source_freeze"], q["source_freeze"], "qualified exact Source")
        call = live["direct_science_calls"][q["call_id"]]
        exact_value(call["state"], "admitted", "fresh nonreplayed actual call")
        exact_value(call["mode"], "original", "Original role")
        exact_value(call["reserved_s"], self.limits["outer_s"], "actual role reservation")
        outer = self.read(call["request"])
        for key, value in (("schema", "flylab.m3.direct-science.supervision.v1"), ("mode", "original"), ("role", self.role),
                           ("call_id", q["call_id"]), ("science_request", self.request_spec), ("source_freeze", q["source_freeze"]),
                           ("grant", q["grant"]), ("output", q["output"] + "-supervision"), ("ledger_path", LEDGER), ("shared_original_lock_path", SHARED_LOCK)):
            exact_value(outer[key], value, "exact owner dispatch " + key)
        for key in ("outer_s", "child_s", "cleanup_s", "host_bytes", "output_cap_bytes"):
            exact_value(outer["limits"][key], self.limits[key], "same outer physical bounds")
        from tools import banc_m3_startup_ray_supervisor as supervisor
        workers, reservations = live["startup_ray_original_workers"], live["startup_ray_original_reservations"]
        supervision = path(outer["output"], root)
        if self.stage == "pre_spawn":
            exact_value(workers, {}, "no existing Original worker")
            exact_value(reservations, {}, "no existing Original reservation")
            require(not path(SHARED_LOCK, root).exists() and not self.output.exists() and not supervision.exists(), "fresh shared exclusion/output")
        else:
            require(type(workers) is dict and set(workers) == {q["call_id"]}, "sole actual worker")
            w = workers[q["call_id"]]
            exact_value(set(w), {"request", "grant", "state", "worker_max", "reserved_s", "lock_token", "worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"}, "complete actual owner lease")
            for key, value in (("request", self.request_spec), ("grant", GRANT), ("state", "begun"), ("worker_max", 1),
                               ("reserved_s", self.limits["outer_s"]), ("worker_pid", os.getpid())):
                exact_value(w[key], value, "actual worker binding")
            for key in ("worker_pid", "worker_creation_ticks", "owner_pid", "owner_creation_ticks"):
                require(type(w[key]) is int and w[key] > 0, "typed positive process identity")
            require(w["worker_pid"] != w["owner_pid"], "distinct actual owner/worker")
            for prefix in ("owner", "worker"):
                exact_value(supervisor.creation_ticks(w[prefix + "_pid"]), w[prefix + "_creation_ticks"], "actual PID creation identity")
            exact_value(w["lock_token"]["path"], SHARED_LOCK, "shared lock only")
            token = self.read(w["lock_token"])
            exact_value(token["schema"], "flylab.m3.direct-science.lock-owner.v1", "actual shared token schema")
            exact_value(json_load(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"].encode("utf8")), token, "actual environment owner token")
            for key, value in (("call_id", q["call_id"]), ("request", call["request"]), ("owner_pid", w["owner_pid"]), ("owner_creation_ticks", w["owner_creation_ticks"]), ("entry_counter_s", self.entry)):
                exact_value(token[key], value, "same actual supervisor origin")
            require(supervisor.in_named_job(token["job_name"]), "actual same named Windows Job")
            exact_value(reservations, {q["call_id"]: {"request": self.request_spec, "grant": GRANT, "reserved_s": self.limits["outer_s"]}}, "sole actual reservation")
            require(supervision.is_dir(), "owner-created supervision output")
            if label == "initial":
                require(not self.output.exists(), "worker-alone fresh output")
        total = 0
        for directory in (self.output, supervision):
            if directory.exists():
                require(directory.is_dir(), "output directory")
                for item in directory.rglob("*"):
                    self.deadline()
                    safe = path(item.relative_to(root).as_posix(), root)
                    if safe.is_file():
                        total += safe.stat().st_size
                        if directory == supervision:
                            cap = (1 << 20) if safe.name in ("stdout.bin", "stderr.bin") else (64 << 10)
                            require(safe.stat().st_size <= cap, "tracking opaque stream/metadata finite cap")
                if directory == self.output:
                    result_bytes = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
                    require(result_bytes <= self.limits["output_cap_bytes"] - (3 << 20), "tracking-only three MiB supervision reserve")
        require(total <= self.limits["output_cap_bytes"], "result plus opaque supervision combined output cap")
        exact_value(live["original_jobs_running"], int(self.stage == "pre_array"), "actual shared jobs")
        exact_value(live["original_reserved_s"], self.limits["outer_s"] if self.stage == "pre_array" else 0, "actual shared reserved seconds")
        self.deadline()
        return True
