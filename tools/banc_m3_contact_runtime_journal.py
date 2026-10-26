"""Bounded acquired contact-body diagnostics; no science or validity authority."""
# ruff: noqa: BLE001 -- diagnostic failures never replace the acquired first error.
from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext

SCHEMA = "flylab.m3.contact-body-runtime.journal.v1"
MAX_EVENTS = 64
MAX_ACTIVE_PHASES = 14  # independent of the expanded fixed phase allowlist
MAX_JSON_BYTES = 32768
MAX_ERROR_CHARS = 1024
MAX_SECONDARY = 8
MAX_EMITS = 16
MAX_STREAM_BYTES = 524288
FLUSH_ALLOWANCE_S = 1.0
SCIENCE_S, WORKER_S = 270.0, 285.0
PREFIX = "FLYLAB_CONTACT_RUNTIME "
PREPARATION_PHASES = ("input-original-files", "input-native-assets", "input-whole-metadata",
                      "input-resolve-api", "input-graph-load", "input-state-load",
                      "input-selected-arrays", "input-result-check")
PHASES = ("admission-entry", "body-entry", "body-plan", "candidate-header",
          "candidate-load", "candidate-hash", "arm-data", "pre-construct",
          "construct", "initial-transfer", "control-loop", "posthash",
          "science-publication", "failure-metadata-publication") + PREPARATION_PHASES
FENCE_LABELS = ("initial", "pre_array", "publication", "pre_construct")
COUNTERS = ("controls_requested", "controls_host_returned", "controls_completed",
            "physics_requested", "physics_host_returned", "physics_device_confirmed",
            "neural_requested", "neural_host_returned", "neural_device_confirmed",
            "completed_rows", "observer_samples", "advance_unreturned_unknown",
            "neural_unreturned_unknown", "physics_unreturned_unknown")
CONSTRUCTION = ("body_constructors_requested", "body_constructors_returned",
                "settling_physics_requested", "settling_physics_host_returned",
                "neural_setup_requested", "neural_setup_host_returned",
                "neural_capture_requested", "neural_capture_recorded",
                "device_confirmed", "unknown_work")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _pin(value):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}, "diagnostic fixed pin")
    need(type(value["path"]) is str and 0 < len(value["path"]) <= 512
         and type(value["bytes"]) is int and 0 < value["bytes"] <= 1048576
         and type(value["sha256"]) is str and len(value["sha256"]) == 64
         and all(c in "0123456789abcdef" for c in value["sha256"]), "diagnostic bounded pin")
    return dict(value)


class RuntimeJournal:
    """Independent lifetime; stream snapshots are explicitly untrusted metadata.

    The one-second allowance is a start/return check, not a hard bound on a
    synchronous pipe write. A blocked write is NOT_RETURNED in the last snapshot;
    the existing Job owner at ENTRY+285 supplies the final hard cutoff.
    """
    def __init__(self, entry, owner, *, clock=time.perf_counter, stream=None, guard=None, role="contact_candidate_body"):
        need(type(entry) is float and math.isfinite(entry) and entry > 0, "actual owner ENTRY origin")
        need(type(owner) is dict and type(owner.get("call_id")) is str
             and 0 < len(owner["call_id"]) <= 128, "acquired exact owner call")
        need(role in ("contact_candidate_body", "process"), "fixed body or generated process diagnostic role")
        need(type(owner.get("entry_counter_s")) is float and owner["entry_counter_s"] == entry, "same acquired owner ENTRY, no fresh origin")
        self.entry, self.clock = entry, clock
        self.owner = dict(owner)
        self.identity = {"call_id": owner["call_id"], "role": role,
                         "source_freeze": None, "science_request": None,
                         "supervisor_request": _pin(owner["request"])}
        self.stream = sys.stderr if stream is None else stream
        self.guard = guard
        self.events, self.active = [], []
        self.last_phase = None
        self.fences = {k: {"attempted": 0, "completed": 0, "failed": 0, "wall_s": 0.0} for k in FENCE_LABELS}
        self.active_fence = None
        # Fixed lifetime totals, never reset by a later arm or snapshot. Unknown
        # wall is sticky None; known zero is observation accounting, not work proof.
        self.preparation = {name: {"attempted": 0, "completed": 0, "failed": 0,
                                 "entry_refused": 0, "active": 0, "state": "NOT_STARTED",
                                 "inclusive_wall_s": 0.0, "observation_incomplete": False,
                                 "nested_fence": {"attempted": 0, "completed": 0,
                                                  "failed": 0, "wall_s": 0.0}}
                            for name in PREPARATION_PHASES}
        self.observation_incomplete = False
        self.first_error, self.secondary_errors = None, []
        self.events_dropped = self.secondary_errors_dropped = 0
        self.bytes_truncated = False
        self.work = None
        self.emit_attempts = self.stream_bytes = 0
        self.last_emission = None
        self._last_emission_error = None  # acquired returned exception, never serialized
        self.terminal_attempted = False
        self.terminal = None
        self._now()

    def _now(self):
        elapsed = self.clock() - self.entry
        need(type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0, "same actual diagnostic clock")
        return elapsed

    def bind(self, request, request_spec):
        from tools import banc_m3_checkpoint_harness_runtime as harness
        from tools import banc_m3_checkpoint_runtime as checkpoint
        if request['call_id'] in harness.CALLS:
            checkpoint = harness
        if request['call_id'] in checkpoint.CALLS:
            need(request['role'] == ('contact_candidate_body', 'contact_body_audit')[checkpoint.CALLS.index(request['call_id'])],
                 'literal K journal body/audit route; original eight preparation slots')
        need(request["call_id"] == self.identity["call_id"]
             and request["role"] == self.identity["role"], "same acquired diagnostic call/role")
        source, science = _pin(request["source_freeze"]), _pin(request_spec)
        for key, value in (("source_freeze", source), ("science_request", science)):
            need(self.identity[key] is None or self.identity[key] == value, "immutable acquired diagnostic " + key)
            self.identity[key] = value

    def science(self):
        if self._now() >= SCIENCE_S:
            raise TimeoutError("contact same owner ENTRY internal deadline")

    def _error_at(self, stage, error, observed):
        # This path uses no clock/owner/science callback. It can preserve an
        # already acquired exception even when the diagnostic clock is refused.
        try:
            message = str(error)[:MAX_ERROR_CHARS]
        except BaseException:
            message = "UNKNOWN: exception message observation failed"
            self.observation_incomplete = True
        record = {"stage": str(stage)[:128], "type": type(error).__name__[:128],
                  "message": message, "observed_s": observed}
        if self.first_error is None:
            self.first_error = record
        elif len(self.secondary_errors) < MAX_SECONDARY:
            self.secondary_errors.append(record)
        else:
            self.secondary_errors_dropped += 1
        return record

    def error(self, stage, error):
        if sys.exc_info()[1] is error:
            # Existing worker/outer error capture is also diagnostic unwind.
            record = self._preserve_error(stage, error)
            try:
                record["observed_s"] = self._now()
            except BaseException as secondary:
                self._observation_failure(stage, secondary)
            return record
        return self._error_at(stage, error, self._now())  # normal refusal propagates

    def _preserve_error(self, stage, error):
        try:
            return self._error_at(stage, error, None)
        except BaseException as secondary:
            # Diagnostic bookkeeping itself may be unavailable during unwind.
            self.observation_incomplete = True
            try:
                message = str(error)[:MAX_ERROR_CHARS]
            except BaseException:
                message = "UNKNOWN: exception message observation failed"
            record = {"stage": stage[:128], "type": type(error).__name__[:128],
                      "message": message, "observed_s": None}
            if self.first_error is None:
                self.first_error = record
            self._observation_failure(stage, secondary)
            return record

    def _observation_failure(self, stage, error, slot=None):
        self.observation_incomplete = True
        if slot is not None:
            slot["observation_incomplete"] = True
        # The caller has already recorded the payload/START error first. A
        # secondary diagnostic error must never become a new primary exception.
        try:
            self._error_at("diagnostic-unwind:" + stage, error, None)
        except BaseException:
            self.secondary_errors_dropped += 1

    @staticmethod
    def _wall(item, key, elapsed):
        if item[key] is not None:
            item[key] += elapsed

    def acquire_errors(self, result):
        """Only fixed scalar error fields; never normalize/traverse the result."""
        if type(result) is not dict:
            return
        pending = sys.exc_info()[1]
        rows = [result.get("first_error")]
        secondary = result.get("secondary_errors")
        if type(secondary) is list:
            rows.extend(secondary[:MAX_SECONDARY])
        for row in rows:
            if type(row) is not dict:
                continue
            record = {key: row.get(key) for key in ("stage", "type", "message")}
            if not all(type(v) is str for v in record.values()):
                continue
            observation_error = None
            try:
                observed = self._now()
            except BaseException as secondary:
                if pending is None:
                    raise
                observed, observation_error = None, secondary
            record = {"stage": record["stage"][:128], "type": record["type"][:128],
                      "message": record["message"][:MAX_ERROR_CHARS], "observed_s": observed}
            comparable = {k: record[k] for k in ("stage", "type", "message")}
            known = [self.first_error, *self.secondary_errors]
            if any(r is not None and {k: r[k] for k in comparable} == comparable for r in known):
                if observation_error is not None:
                    self._observation_failure("error-acquisition", observation_error)
                continue
            if self.first_error is None:
                self.first_error = record
            elif len(self.secondary_errors) < MAX_SECONDARY:
                self.secondary_errors.append(record)
            else:
                self.secondary_errors_dropped += 1
            if observation_error is not None:
                self._observation_failure("error-acquisition", observation_error)

    def acquire_work(self, counters, construction):
        """Project exact builtin scalars from existing receipts, including unknown."""
        projection = {}
        for group, keys, values in (("counters", COUNTERS, counters), ("construction", CONSTRUCTION, construction)):
            if type(values) is not dict:
                continue
            projected = {}
            for key in keys:
                value = values.get(key)
                if type(value) is bool or (type(value) is int and 0 <= value <= 10**12):
                    projected[key] = value
            if projected and any(v is True or (type(v) is int and v > 0) for v in projected.values()):
                projection[group] = projected
        if projection:
            self.work = projection

    def _event(self, phase, kind, start=None):
        row = {"phase": phase, "kind": kind, "observed_s": self._now()}
        if start is not None:
            row["inclusive_wall_s"] = row["observed_s"] - start
        self.last_phase = dict(row)
        if len(self.events) < MAX_EVENTS - 1:  # reserve one terminal slot
            self.events.append(row)
        else:
            self.events_dropped += 1
        return row["observed_s"]

    def start(self, phase, *, opaque=False):
        need(phase in PHASES and phase != "failure-metadata-publication", "fixed science diagnostic phase")
        slot = self.preparation.get(phase)
        need(slot is None or not opaque, "preparation phase is non-opaque")
        if slot is not None:
            slot["attempted"] += 1
        try:
            self.science()
            need(len(self.active) < MAX_ACTIVE_PHASES, "bounded diagnostic nesting")
            start = self._now()
        except BaseException as error:
            if slot is not None:
                slot["entry_refused"] += 1
                if not slot["active"]:
                    slot["state"] = "NOT_STARTED"
                self._preserve_error(phase, error)
            raise  # normal clock/science refusal: never enter/retry the callback
        self.active.append({"phase": phase, "started_s": start})
        if slot is not None:
            slot["active"] += 1
            slot["state"] = "NOT_RETURNED"
        try:
            self._event(phase, "START")
            if opaque:
                returned = self.emit()
                if not returned and self.last_emission is not None and self.last_emission["status"] == "FAIL_OR_NOT_RETURNED":
                    if self._last_emission_error is not None:
                        raise self._last_emission_error
                    raise ValueError("opaque diagnostic stream/owner guard refused")
                self.science()  # a synchronous stream return cannot grant late science
        except BaseException as error:
            self._unwind(phase, start, error, entered=False)
            raise
        return start

    def finish(self, phase, start, *, error=None, entered=True, error_record=None):
        need(self.active and self.active[-1]["phase"] == phase, "diagnostic phase LIFO")
        self.active.pop()
        slot = self.preparation.get(phase)
        if slot is not None:
            slot["active"] -= 1
            if entered:
                slot["failed" if error is not None else "completed"] += 1
                slot["state"] = "FAILED" if error is not None else "RETURNED"
            else:
                slot["entry_refused"] += 1
                slot["state"] = "NOT_STARTED"
            if slot["active"]:
                slot["state"] = "NOT_RETURNED"
        try:
            observed = self._event(phase, "FAIL" if error is not None else "RETURN", start)
        except BaseException:
            self.observation_incomplete = True
            if slot is not None:
                slot["inclusive_wall_s"] = None
                slot["observation_incomplete"] = True
            raise
        if slot is not None:
            self._wall(slot, "inclusive_wall_s", observed-start)
        if error_record is not None:
            error_record["observed_s"] = observed
        elif error is not None:
            self._error_at(phase, error, observed)

    def _unwind(self, phase, start, error, *, entered=True):
        # Record first before any finish clock/bookkeeping can fail. Bare raise
        # in the caller rethrows the same acquired exception object.
        record = self._preserve_error(phase, error)
        try:
            self.finish(phase, start, error=error, entered=entered, error_record=record)
        except BaseException as secondary:
            slot = self.preparation.get(phase)
            # A finish refusal before its pop must not strand an enclosing
            # phase. Remove only this exact matching frame; never retry finish.
            if self.active and self.active[-1] == {"phase": phase, "started_s": start}:
                self.active.pop()
                if slot is not None:
                    slot["active"] -= 1
                    if entered:
                        slot["failed"] += 1
                    else:
                        slot["entry_refused"] += 1
                    slot["state"] = ("NOT_RETURNED" if slot["active"] else
                                     ("FAILED" if entered else "NOT_STARTED"))
            if slot is not None:
                slot["inclusive_wall_s"] = None
            self._observation_failure(phase, secondary, slot)

    @contextmanager
    def phase(self, phase, *, opaque=False):
        start = self.start(phase, opaque=opaque)
        try:
            yield
        except BaseException as error:
            self._unwind(phase, start, error)
            raise
        else:
            self.finish(phase, start)

    def _fence_finish(self, item, nested, start):
        try:
            observed = self._now()
            self._wall(item, "wall_s", observed-start)
            if nested is not None:
                self._wall(nested, "wall_s", observed-start)
            return observed
        except BaseException:
            item["wall_s"] = None
            if nested is not None:
                nested["wall_s"] = None
            self.observation_incomplete = True
            raise
        finally:
            self.active_fence = None

    @contextmanager
    def fence(self, label):
        need(label in FENCE_LABELS and self.active_fence is None, "fixed cold fence label")
        start = self._now()
        item = self.fences[label]
        # Capture nearest active preparation at entry. Unwind never looks up
        # a current/last stage, which could have changed during the callback.
        name = next((row["phase"] for row in reversed(self.active)
                     if row["phase"] in PREPARATION_PHASES), None)
        slot = None if name is None else self.preparation[name]
        nested = None if slot is None else slot["nested_fence"]
        item["attempted"] += 1
        if nested is not None:
            nested["attempted"] += 1
        self.active_fence = {"label": label, "started_s": start, "preparation": name}
        try:
            yield
        except BaseException as error:
            item["failed"] += 1
            if nested is not None:
                nested["failed"] += 1
            record = self._preserve_error("cold-fence:" + label, error)
            try:
                record["observed_s"] = self._fence_finish(item, nested, start)
            except BaseException as secondary:
                self._observation_failure("cold-fence:" + label, secondary, slot)
            raise
        else:
            item["completed"] += 1
            if nested is not None:
                nested["completed"] += 1
            try:
                self._fence_finish(item, nested, start)
            except BaseException:
                if slot is not None:
                    slot["observation_incomplete"] = True
                raise  # normal diagnostic refusal remains a refusal

    def _projection(self):
        now = self._now()
        active = [dict(item, observed_partial_s=now-item["started_s"], state="NOT_RETURNED") for item in self.active]
        fence = None if self.active_fence is None else dict(self.active_fence, observed_partial_s=now-self.active_fence["started_s"])
        preparation = {}
        for name, item in self.preparation.items():
            row = dict(item, nested_fence=dict(item["nested_fence"]))
            row["observed_partial_s"] = sum(now-scope["started_s"] for scope in self.active
                                            if scope["phase"] == name)
            row["nested_fence_partial_s"] = (now-self.active_fence["started_s"]
                if self.active_fence is not None and self.active_fence["preparation"] == name else 0.0)
            preparation[name] = row
        return {"schema": SCHEMA, "identity": dict(self.identity), "owner_entry_counter_s": self.entry,
                "observed_s": now, "diagnostic_only": True, "run_valid": False,
                "trust": "UNTRUSTED_DIAGNOSTIC_NO_FRESH_SOURCE_POSTHASH_LEASE_ATTESTATION",
                "events": [dict(v) for v in self.events], "active": active,
                "last_phase": None if self.last_phase is None else dict(self.last_phase),
                "cold_fences_nested_wall": {k: dict(v) for k, v in self.fences.items()},
                "preparation": preparation, "observation_incomplete": self.observation_incomplete,
                "active_fence": fence, "first_error": None if self.first_error is None else dict(self.first_error),
                "secondary_errors": [dict(v) for v in self.secondary_errors],
                "events_dropped": self.events_dropped, "secondary_errors_dropped": self.secondary_errors_dropped,
                "bytes_truncated": self.bytes_truncated,
                "incomplete": bool(self.observation_incomplete or self.events_dropped or self.bytes_truncated or active or fence),
                "work": self.work, "work_semantics": "existing scalar receipts only; missing/unreturned work UNKNOWN; no invented zero receipt",
                "timing_semantics": "per-journal lifetime totals; partial times are projections only; inclusive and nested fence wall are separate, do not add; null wall UNKNOWN",
                "preparation_semantics": "states only as of observed_s; no later-entry or zero body/neural/physics proof; non-opaque memory needs an acquired snapshot",
                "stream": {"attempts": self.emit_attempts, "bytes": self.stream_bytes,
                           "last": None if self.last_emission is None else dict(self.last_emission),
                           "flush_allowance_s": FLUSH_ALLOWANCE_S, "worker_cutoff_s": WORKER_S},
                "terminal": self.terminal}

    def encoded(self, *, cap=MAX_JSON_BYTES):
        need(type(cap) is int and 16384 <= cap <= MAX_JSON_BYTES, "fixed bounded journal wire cap")
        value = self._projection()
        while True:
            raw = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
            if len(raw) <= cap:
                return raw
            self.bytes_truncated = True
            value["bytes_truncated"], value["incomplete"] = True, True
            # Preserve all eight preparation slots, first error, active phase/fence,
            # work, terminal and global fence totals within the unchanged wire cap.
            if value["events"]:
                value["events"].pop(0)
                value["events_dropped"] += 1
            elif value["secondary_errors"]:
                value["secondary_errors"].pop()
                value["secondary_errors_dropped"] += 1
            else:
                raise ValueError("reserved diagnostic fields exceed bounded wire cap")

    def snapshot(self):
        return json.loads(self.encoded().decode("ascii"))  # independent and bounded valid JSON

    def emit(self, *, terminal=False):
        self._last_emission_error = None
        if terminal:
            need(not self.terminal_attempted, "terminal metadata attempt once")
            self.terminal_attempted = True
        elif self.emit_attempts >= MAX_EMITS - 1:
            return False
        now = self._now()
        if now >= WORKER_S - FLUSH_ALLOWANCE_S or (not terminal and now >= SCIENCE_S):
            self.last_emission = {"kind": "TERMINAL" if terminal else "BOUNDARY", "status": "REFUSED_NO_FLUSH_TIME", "started_s": now}
            return False
        if terminal:
            self.terminal = {"phase": "failure-metadata-publication", "kind": "START", "observed_s": now, "state": "NOT_RETURNED"}
        self.emit_attempts += 1
        self.last_emission = {"kind": "TERMINAL" if terminal else "BOUNDARY", "status": "NOT_RETURNED", "started_s": now}
        try:
            if self.guard is not None:
                self.guard()
            previous_bytes = self.stream_bytes
            ceiling = MAX_STREAM_BYTES if terminal else MAX_STREAM_BYTES-MAX_JSON_BYTES
            for _ in range(8):
                raw = self.encoded(cap=MAX_JSON_BYTES-len(PREFIX)-1)
                wire = PREFIX.encode("ascii") + raw + b"\n"
                projected = previous_bytes + len(wire)
                need(projected <= ceiling, "terminal stream bytes reserved")
                if self.stream_bytes == projected:
                    break
                self.stream_bytes = projected  # attempted bytes, not proof of delivery
            else:
                raise ValueError("bounded stream byte-count closure")
            count = self.stream.write(wire.decode("ascii"))
            self.stream.flush()
            returned = self._now()
            need(count is None or count == len(wire), "complete diagnostic stream write return")
            if self.guard is not None:
                self.guard()
            need(returned < WORKER_S and returned-now <= FLUSH_ALLOWANCE_S, "metadata flush return within owner cutoff/allowance")
            self.last_emission.update(status="RETURN", returned_s=returned)
            if terminal:
                self.terminal.update(kind="RETURN", state="RETURNED", returned_s=returned)
            return True
        except Exception as error:
            self._last_emission_error = error
            stage = "failure-metadata-publication" if terminal else "boundary-metadata-publication"
            record = self._preserve_error(stage, error)
            try:
                record["observed_s"] = self._now()
            except BaseException as secondary:
                self._observation_failure(stage, secondary)
            self.last_emission["status"] = "FAIL_OR_NOT_RETURNED"
            return False

    def terminal_failure(self, result=None):
        if result is not None:
            self.acquire_errors(result)
        return self.emit(terminal=True)


def phase(timer, name, *, opaque=False):
    journal = getattr(timer, "runtime_journal", None)
    return nullcontext() if journal is None else journal.phase(name, opaque=opaque)


def owned_stream_guard(owner, worker_pid, worker_creation):
    """Minimal live process/Job check; never scans Source, history or inputs."""
    from tools import banc_m3_startup_ray_supervisor as supervisor
    need(os.getpid() == worker_pid and json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"]) == owner,
         "acquired diagnostic owner environment/process drift")
    need(supervisor.creation_ticks(worker_pid) == worker_creation
         and supervisor.creation_ticks(owner["owner_pid"]) == owner["owner_creation_ticks"]
         and supervisor.in_named_job(owner["job_name"]), "live diagnostic PID creation/named Job")
