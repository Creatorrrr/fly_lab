"""Two local geometry Original roles; historical J/tracking/replay stays intact."""
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
SCHEMA = "flylab.m3.local-foot-geometry"
ROLES = ("geometry_capture", "geometry_analyze")
CALLS = ("local-foot-geometry-20261007-b-capture", "local-foot-geometry-20261007-b-analyze")
OLD_ROLES = ("input_audit", "ray", "candidate_audit", "neutral_body", "candidate_body", "neutral_audit", "candidate_body_audit", "close")
GRANT = {"path": BASE + "/parent-startup-ray-original-grant-20261007-j.json", "bytes": 2227,
         "sha256": "7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f"}
INPUTS = {"path": BASE + "/parent-startup-ray-original-inputs.json", "bytes": 210093,
          "sha256": "4bdd075fc9980a275319fc0a26210c15335050ff7e96bc2fd29eb0ec5aeda151"}
ROSTER = {"path": BASE + "/parent-local-foot-geometry-input-roster-20261007-c.json", "bytes": 10031,
          "sha256": "d05c2c73eeddcb64300f71c00f85f00d0eec22675cca735fb37d583fb550765b"}
REQUIRED_SOURCES = ("tools/banc_m3_local_foot_geometry_admission.py", "tools/observe_c_banc_m3_local_foot_geometry.py",
                    "tools/analyze_c_banc_m3_local_foot_geometry.py", "tools/observe_c_banc_m3_startup_tracking.py",
                    "tools/analyze_c_banc_m3_startup_tracking.py", "tools/banc_m3_live_replay_admission.py",
                    "tools/banc_m3_saved_original_admission.py", "tools/verify_c_banc_m3_startup_ray.py",
                    "tests/test_c_banc_local_foot_geometry.py", "tests/test_c_banc_startup_ray_execution.py",
                    "tools/banc_m3_startup_ray_supervisor.py", BASE + "/parent-direct-science-account.ps1",
                    BASE + "/parent-direct-science-genuine-collector.js", BASE + "/parent-local-foot-geometry-prepare-native.ps1")


def limits_for(role):
    require(role in ROLES, "two fixed Original geometry roles")
    capture = role == ROLES[0]
    limits = {"outer_s": 300 if capture else 120, "child_s": 285 if capture else 100,
              "internal_s": 270 if capture else 90, "cleanup_s": 15 if capture else 20,
              "host_bytes": (24 if capture else 2) * 1024**3, "output_cap_bytes": (160 if capture else 16) * 1024**2,
              "input_file_cap_bytes": 160 * 1024**2, "selected_expanded_cap_bytes": 208 * 1024**2,
              "member_cap_bytes": 16 * 1024**2, "worker_max": 1}
    if capture:
        limits.update(GPU_delta_plus_allowance_cap_bytes=8 * 1024**3, sample_cap=3000)
    return limits

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


def validate_failed_capture(before, originals, roster, read, closed, archive_read):
    failed = roster["failed_geometry_attempt"]
    exact_value(originals[15]["descriptor"], failed["cost"], "exact actual failed geometry-a cost")
    cost = closed(read(failed["cost"]), originals[15])
    require(cost["schema"] == SCHEMA + ".full-return.v1" and cost["role"] == ROLES[0]
            and cost["cost_id"] == failed["call_id"] == "local-foot-geometry-20261007-a-capture"
            and cost["exit_code"] == 1 and cost["cap_exceeded"] is False
            and cost["worker_started"] is True and cost["job_created"] is True, "actual failed-a closed worker preserved")
    exact_value(cost["source_freeze"], failed["source_freeze"], "failed-a actual Source preserved")
    exact_value(cost["outer"], failed["outer"], "failed-a actual owner record preserved")
    report, manifest = read(failed["body_report"]), read(failed["manifest"])
    require(report["run_valid"] is False and report["capture_complete"] is False and report["all_owned_ended"] is True
            and report["first_error"]["message"] == failed["first_error"] == "typed bounded mj_sizeModel before allocation"
            and report["first_error"]["stage"] == "construct" and report["construction"]["unknown_work"] is False
            and report["sampler"]["state"] == "CLOSED", "failed-a actual partial failure/known work")
    for key, value in (("body_constructors_requested", 1), ("body_constructors_returned", 1),
                       ("neural_setup_requested", 320), ("neural_setup_host_returned", 320),
                       ("neural_capture_requested", 160), ("neural_capture_recorded", 160),
                       ("settling_physics_requested", 2000), ("settling_physics_host_returned", 2000)):
        exact_value(report["construction"][key], value, "actual failed-a construction " + key)
    for key in ("controls_requested", "controls_host_returned", "controls_completed", "neural_requested",
                "neural_host_returned", "neural_device_confirmed", "physics_requested", "physics_host_returned", "physics_device_confirmed"):
        exact_value(report["counters"][key], 0, "actual failed-a normal interval " + key)
    require(manifest["run_valid"] is False and manifest["capture_complete"] is False
            and "model" not in manifest["artifacts"] and "geometry" not in manifest["artifacts"], "failed-a has no invented complete geometry")
    exact_value(manifest["artifacts"]["body_report"], failed["body_report"], "actual failed-a manifest/partial report")
    old = read(failed["closed_ledger"])
    require(len(old["debits"]) == 207, "actual closed-after-failed-a207")
    exact_value(before["debits"][:207], old["debits"], "every old failed/qualified debit preserved")
    for call, value in old["direct_science_calls"].items():
        exact_value(before["direct_science_calls"][call], value, "every closed-after-failed-a call preserved")
    freeze, archive = read(failed["source_freeze"]), read(failed["source_archive"])
    require(archive["schema"] == "flylab.m3.local-foot-geometry-source-archive.v1" and len(archive["sources"]) == 8,
            "eight actual executed failed-a sources")
    exact_value(archive["source_freeze"], failed["source_freeze"], "failed-a archive freeze")
    for item in archive["sources"]:
        spec = freeze["sources"][item["original_path"]]
        require(item["bytes"] == spec["bytes"] and item["sha256"] == spec["sha256"], "failed-a archived Source binding")
        raw = archive_read(item)
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "actual failed-a archived bytes")


def validate_history(before, grant, roster, q, read, closed, archive_read):
    """Preserve15 prior Original returns, failed-a, archive8+6+8 and b predecessor."""
    from tools import banc_m3_live_replay_admission as replay
    require(len(before["debits"]) >= 200 and len({d["cost_id"] for d in before["debits"]}) == len(before["debits"]), "all200 plus current qualifications once preserved")
    originals = [d for d in before["debits"][165:] if d["mode"] == "original"]
    prior = ROLES.index(q["role"])
    require(len(originals) == 16 + prior, "exact historical15/failed-a plus sole b predecessor")
    actual = read(roster["actual_replay_result"])
    require(actual["status"] == "ACTUAL_SAVED_OUTPUT_CORRESPONDENCE_COMPLETE" and actual["observations"]["output_correspondence_passed"] is True
            and actual["candidate_adopted"] is False and actual["resources"]["debits"] == 200, "actual completed replay, rejection retained")
    exact_value(actual["resources"]["ledger"], roster["historical_live_ledger_descriptor_as_of_closed200"], "historical as-of live descriptor")
    require(all(roster["historical_closed_200_ledger"][k] == actual["resources"]["ledger"][k] for k in ("bytes", "sha256")), "immutable closed200 identical bytes/hash")
    closed200 = read(roster["historical_closed_200_ledger"])
    require(len(closed200["debits"]) == 200 and len(closed200["direct_science_calls"]) == 82, "actual immutable closed200 metadata")
    exact_value(before["debits"][:200], closed200["debits"], "all200 historical debit rows retained")
    for call, value in closed200["direct_science_calls"].items():
        exact_value(before["direct_science_calls"][call], value, "every historical direct call unchanged")
    replay_cost = closed(read(originals[14]["descriptor"]), originals[14])
    exact_value(originals[14]["descriptor"], roster["replay_costs"][3], "exact actual replay once return")
    require(replay_cost["schema"] == "flylab.m3.live-replay.full-return.v1" and replay_cost["role"] == "live_replay" and replay_cost["exit_code"] == 0, "actual completed replay role")
    exact_value(replay_cost["source_freeze"], roster["replay_source_freeze"], "historical replay Source descriptor")
    # Reuse prior14 validator on a metadata projection; actual debits are untouched.
    historic = dict(before)
    historic["debits"] = [d for d in before["debits"] if d["cost_id"] not in {item["cost_id"] for item in originals[14:]}]
    old_total = 0.0
    for d in originals[:14]:
        old_total += finite(read(d["descriptor"])["actual_outer_full_return_s"], "historical actual return")
    for key, value in (("original_cumulative_used_s", grant["original_cumulative_before_s"] + old_total),
                       ("original_stage_used_s", grant["original_stage_before_s"] + old_total),
                       ("startup_ray_original_local_used_s", grant["science_local_used_before_s"] + old_total),
                       ("original_remaining_s", 16200.0 - grant["original_cumulative_before_s"] - old_total)):
        historic[key] = value
    old_roster = read(roster["previous_replay_roster"])
    replay.validate_history(historic, grant, old_roster, {}, read, closed, archive_read)
    freeze, archive = read(roster["replay_source_freeze"]), read(roster["replay_source_archive"])
    require(archive["schema"] == "flylab.m3.live-replay-source-archive.v1" and len(archive["sources"]) == 6, "actual replay archived six Source")
    for item in archive["sources"]:
        frozen = freeze["sources"][item["original_path"]]
        require(item["bytes"] == frozen["bytes"] and item["sha256"] == frozen["sha256"], "historical replay Source archive binding")
        raw = archive_read(item)
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "actual historical replay archive bytes")
    for spec in roster["replay_costs"][:3]:
        cost = read(spec)
        require(cost["charged_once"] is True and cost["exit_code"] == 0, "three actual replay qualification costs preserved")
        require(any(d["descriptor"] == spec for d in before["debits"][165:] if d["mode"] == "generated"), "exact generated replay cost row retained")
    validate_failed_capture(before, originals, roster, read, closed, archive_read)
    if prior:
        debit = originals[16]
        cost = closed(read(debit["descriptor"]), debit)
        require(cost["schema"] == SCHEMA + ".full-return.v1" and cost["role"] == ROLES[0] and cost["cost_id"] == CALLS[0]
                and cost["exit_code"] == 0 and cost["cap_exceeded"] is False, "only complete charged predecessor permits CPU analysis")
        exact_value(cost["source_freeze"], q["source_freeze"], "same new geometry Source")
        manifest = read(q["artifacts"]["manifest"])
        require(manifest["role"] == ROLES[0] and manifest["run_valid"] is True and manifest["capture_complete"] is True, "complete new neutral geometry manifest")
        for key, value in q["artifacts"].items():
            if key != "manifest":
                exact_value(manifest["artifacts"][key], value, "exact first-observed geometry input")
    else:
        exact_value(q["artifacts"], {}, "capture has no invented geometry inputs")
    total = 0.0
    for d in originals:
        total += finite(read(d["descriptor"])["actual_outer_full_return_s"], "all Original costs")
    for key, value in (("original_cumulative_used_s", grant["original_cumulative_before_s"] + total),
                       ("original_stage_used_s", grant["original_stage_before_s"] + total),
                       ("startup_ray_original_local_used_s", grant["science_local_used_before_s"] + total),
                       ("original_remaining_s", 16200.0 - grant["original_cumulative_before_s"] - total)):
        exact_value(before[key], value, "all Original cost and J carry " + key)
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
        self.stage, self.entry, self.caller_entry, self._deadline_s = stage, origin, caller, 270
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
        require(self.roster["schema"] == SCHEMA + ".input-roster.v1", "corrected replay roster b")
        contract = self.roster["contract"]
        exact_value(contract["boundary_controls"], [100, 101, 164, 165, 228, 229, 292, 293], "eight fixed post-forward boundaries")
        require(contract["arm"] == "neutral" and contract["extra_live_forward"] == contract["extra_live_neural"] == contract["extra_live_physics"] == contract["gradient"] == contract["new_candidate"] == 0, "unchanged neutral-only scope")
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

        def archive_read(item):
            self.deadline()
            archived = Path(item["archive_path"])
            require(archived.is_absolute(), "absolute as-executed archive path")
            relative = archived.relative_to(root).as_posix()
            require(relative.startswith((BASE + "/parent-tracking-as-executed-c/", BASE + "/parent-live-replay-as-executed-a/",
                                         BASE + "/parent-local-foot-geometry-as-executed-c/")), "historical archive only; never current canonical")
            raw = leaf({"path": relative, "bytes": item["bytes"], "sha256": item["sha256"]}, root, 2097152)
            self.deadline()
            return raw

        validate_history(self.before, g, self.roster, q, self.read, prior_closed, archive_read)
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
