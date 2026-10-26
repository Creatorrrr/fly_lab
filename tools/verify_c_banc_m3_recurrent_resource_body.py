"""E291 fixed resource recurrence: exact neutral, strict startup, continuous body.

Generated qualification uses injected fake sessions only. Original execution
requires pinned source approvals and a separately approved tiny qualification.
"""
from __future__ import annotations

# Preserve first errors and every secondary cleanup/publication failure.
# ruff: noqa: BLE001
import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_recurrent_resource_synapses as neural
from tools import banc_m3_recurrent_resource_trial as layer
from tools import verify_c_banc_m3_source_decoder_body as e288
from tools import verify_c_banc_m3_source_native_transfer as prior

protocol, native, short, gait, receipts = prior.protocol, prior.native, prior.short, prior.gait, prior.receipts
LEGS = prior.LEGS
STARTUP_LIMITS, LIMITS = copy.deepcopy(prior.STARTUP_LIMITS), copy.deepcopy(prior.LIMITS)
FREEZE_SCHEMA = "flylab.m3.e291.recurrent-resource-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e291.recurrent-resource-body.review.v1"
OWNED = ("tools/verify_c_banc_m3_recurrent_resource_body.py", "tests/test_c_banc_recurrent_resource_body.py")
PRODUCERS = ("tools/banc_m3_recurrent_resource_synapses.py", "tests/test_c_banc_recurrent_resource_synapses.py",
             "tools/banc_m3_recurrent_resource_trial.py", "tests/test_c_banc_recurrent_resource_trial.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *PRODUCERS, *prior.SOURCE_FILES,
    "tools/verify_c_banc_m3_source_decoder_body.py", "tools/banc_m3_recurrent_mask.py",
    "tools/probe_c_banc_m3_core_rhythm.py")))
PATHS = (prior.PATHS - {"candidate", "predictions"}) | {"baseline_neutral_final_checkpoint"}
ROSTER_PATH = "verification/banc-natural-gait-m3-20260919/e289-recurrent-resource-roster.npz"
ROSTER_SHA = "1a758419034490ae20070ebe4d82dc8797241c065fb6c87f9b1ba6f6d4c6abd2"
BASELINE_FORWARD_MM = e288.BASELINE_FORWARD_MM
assess_short = e288.assess_short
safe = prior.safe


def checked_roster(roster):
    keys = layer.SELECTION_KEYS | {"neutral_utilization", "descriptive_edge_groups"}
    if set(roster) != keys:
        raise ValueError("Complete fixed resource roster required")
    selection = {key: roster[key] for key in layer.SELECTION_KEYS}
    layer.validate_selection_metadata(selection, 158706, roster["motor_ids"])
    if (roster["premotor_ids"].shape != (7136,) or roster["source_ids"].shape != (7136,)
            or roster["edge_ids"].shape != (725849,) or roster["motor_ids"].shape != (391,)
            or not layer.equal(roster["utilization"], np.full(7136, .05, np.float64))
            or not layer.equal(roster["neutral_utilization"], np.zeros(7136, np.float64))
            or not layer.equal(roster["recovery_tau"], np.full(7136, .150, np.float64))
            or roster["descriptive_edge_groups"].dtype != np.int64
            or roster["descriptive_edge_groups"].shape != (725849,)):
        raise ValueError("Fixed 7136/725849 u=.05 tau=.150 roster required")
    return selection


def verify_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Exact pinned receipt spec required")
    path = safe(spec["path"])
    if protocol.sha256(path) != spec["sha256"]:
        raise ValueError("Pinned receipt SHA differs")
    return json.loads(path.read_text(encoding="utf-8"))


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if protocol.sha256(freeze_path) != freeze_sha or protocol.sha256(review_path) != review_sha:
        raise ValueError("E291 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("paths", {})) != PATHS or freeze.get("startup_limits") != STARTUP_LIMITS
            or freeze.get("resource_limits") != LIMITS or freeze.get("n") != 158706
            or freeze.get("edges") != 725849 or freeze.get("premotor_count") != 7136
            or freeze.get("source_count") != 7136 or freeze.get("motor_count") != 391
            or freeze.get("utilization") != .05 or freeze.get("neutral_utilization") != 0.
            or freeze.get("recovery_tau") != .150 or freeze.get("candidate_adopted") is not False
            or freeze["paths"]["selection"] != ROSTER_PATH
            or freeze["input_sha256"].get(ROSTER_PATH) != ROSTER_SHA):
        raise ValueError("Exact E291 scientific/resource freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E291 source review required")
    pins = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"], review["source_sha256"],
                  freeze["preservation_sha256"], {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E291 hash pin")
            pins[name] = digest
    if not set(SOURCE_FILES) <= set(pins) or not freeze["preservation_sha256"]:
        raise ValueError("Actual inherited sources and preservation pins required")
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or protocol.sha256(safe(name)) != digest:
            raise ValueError("Frozen E291 bytes changed: " + name)
    for alias, name in freeze["paths"].items():
        path = safe(name)
        children = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        if not children or any(p.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for p in children):
            raise ValueError("Every required input byte must be pinned: " + alias)
    old = verify_spec(freeze["e279_freeze"])
    old_review = verify_spec(freeze["e279_source_review"])
    if (old.get("schema") != prior.FREEZE_SCHEMA or old_review.get("schema") != prior.REVIEW_SCHEMA
            or old_review.get("approved") is not True or old_review.get("blocking_defects") != []
            or old_review.get("freeze_sha256") != freeze["e279_freeze"]["sha256"]):
        raise ValueError("Approved original E279 provenance required")
    for name, digest in old_review["source_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("Inherited E279 source identity differs")
    for alias in prior.PATHS - {"candidate", "predictions", "selection"}:
        if freeze["paths"][alias] != old["paths"][alias]:
            raise ValueError("Original E150/E279 input changed: " + alias)
    for alias in prior.PATHS - {"candidate", "predictions", "selection"}:
        path = safe(freeze["paths"][alias])
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        for file in files:
            name = file.relative_to(ROOT).as_posix()
            if freeze["input_sha256"].get(name) != old["input_sha256"].get(name):
                raise ValueError("Original E279 input digest differs: " + name)
    for name, digest in old.get("preservation_sha256", {}).items():
        if pins.get(name) != digest:
            raise ValueError("Original preservation digest differs: " + name)
    for key in ("graph_hash", "source_network_identity", "adapter_identity", "packages"):
        if freeze.get(key) != old[key]:
            raise ValueError("Original provenance changed: " + key)
    if not freeze["packages"]:
        raise ValueError("Pinned native packages required")
    producer_sha = {name: pins[name] for name in PRODUCERS}
    for alias in ("e289_source_review", "e290_source_review", "tiny_qualification"):
        receipt = verify_spec(freeze[alias])
        expected = producer_sha if alias == "tiny_qualification" else {
            name: digest for name, digest in producer_sha.items() if ("synapses" if alias == "e289_source_review" else "trial") in name}
        if (receipt.get("approved") is not True or receipt.get("blocking_defects") != []
                or receipt.get("source_sha256") != expected):
            raise ValueError("Approved current producer/tiny qualification required: " + alias)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen E291 package changed: " + name)
    checked_roster(protocol.load_npz(safe(ROSTER_PATH)))
    # Compute identities from stored metadata only; never instantiate a neural/body model.
    NativeDependencies().load(freeze)
    if output.resolve() != safe(freeze["output"]) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E291 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E291 disk reserve unavailable")
    return freeze


class NativeDependencies(prior.NativeDependencies):
    def write(self, path, value):
        return bounded_write_receipt(path, value)

    def load(self, freeze):
        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore
        paths = freeze["paths"]
        roster = protocol.load_npz(safe(paths["selection"]))
        selection = checked_roster(roster)
        saved = StateStore.load(safe(paths["neutral_checkpoint"]))
        layer.validate_neutral(saved["network"])
        e288.validate_zero_checkpoint(saved)
        graph = GraphStore.load(safe(paths["graph_bundle"]))
        if (graph.hash != freeze["graph_hash"] or saved["network"]["model_hash"] != freeze["source_network_identity"]
                or saved["adapter"]["hash"] != freeze["adapter_identity"]):
            raise ValueError("Original graph/neural/adapter identity differs")
        layer.require_equal(roster["motor_ids"], saved["monotone_recruitment_patch"]["motor_ids"], "Original motor roster")
        for arm, u in (("neutral", roster["neutral_utilization"]), ("candidate", roster["utilization"])):
            metadata = neural.resource_metadata(saved["network"]["model_hash"], {**selection, "utilization": u})
            if neural.resource_identity(metadata) != freeze[arm + "_network_identity"]:
                raise ValueError("Computed resource network identity differs: " + arm)
        baseline_end = StateStore.load(safe(paths["baseline_neutral_final_checkpoint"]))
        if "source_filter_trial_patch" in baseline_end:
            baseline_end = prior.layer.remove_source_layer(baseline_end)
        baseline_short = json.loads(safe(paths["baseline_short_result"]).read_text(encoding="utf-8"))
        if (baseline_short.get("slip", {}).get("load_weighted_mm_s") != 1.8026436968249955
                or baseline_short.get("forward", {}).get("fixed_initial_heading_forward_mm") != BASELINE_FORWARD_MM):
            raise ValueError("Exact E257 short baseline required")
        return {"graph": graph, "saved": saved, "selection": selection, "roster": roster,
                "freeze": freeze, "baseline_end": baseline_end,
                "startup": protocol.load_npz(safe(paths["startup"])),
                "baseline_rows": json.loads(safe(paths["baseline_trace"]).read_text(encoding="utf-8"))["samples"],
                "baseline_native": protocol.load_npz(safe(paths["baseline_native"])),
                "baseline_recruitment": protocol.load_npz(safe(paths["baseline_recruitment"])),
                "baseline_short": baseline_short,
                "ids": np.unique(np.r_[roster["premotor_ids"], roster["motor_ids"]]).astype(np.int64),
                "source_ids": roster["source_ids"], "motor_ids": roster["motor_ids"]}

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            base = layer.MonotoneRecruitmentTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            try:
                s = data["selection"]
                trial = layer.RecurrentResourceTrial(base, s["edge_ids"], s["premotor_ids"],
                    data["roster"]["neutral_utilization"] if arm == "neutral" else s["utilization"], s["recovery_tau"])
                trial._e291_arm = arm
                return trial
            except Exception as error:
                layer.cleanup_owned(base, error)
                raise
        return short.construction_with_body_count(arm, callback, ledger, persist, on_return)

    def initial(self, session, data, arm):
        changes, expected_changes = [], []
        saved = session.snapshot()
        actual = prior.checkpoint_representation(layer.remove_resource_layer(saved), changes)
        expected = prior.checkpoint_representation(data["saved"], expected_changes)
        layer.require_equal(actual, expected, "Exact whole original initial state")
        e288.validate_zero_checkpoint(actual)
        if (session.network.identity != data["freeze"][arm + "_network_identity"]
                or session.adapter.identity != data["freeze"]["adapter_identity"]
                or session.control_tick != 0 or session.network.tick != 0
                or abs(session.body.physics_time()) > 2e-8 or session.network.device != "cuda"
                or session.network.cuda_implementation != "packed" or session.network.interval_steps != 20):
            raise ValueError("Exact original CUDA clocks and identities required")
        layer.require_equal(saved["network"]["synaptic_resource"], np.ones(7136, np.float64), "Initial resource one")
        return {"storage_representation_exact": True, "actual_representation_changes": changes,
                "expected_representation_changes": expected_changes, "array_coercions": 0,
                "numeric_tolerance": 0, "excluded_fields": []}

    def capture(self, session, raw):
        row = protocol.capture_control(session)
        raw["rows"].append(row)
        raw["controls"].append(short.capture_control_raw(session, row))
        prior.monotone.make_observer(raw["recruitment"], original=lambda *_args: {})(session, np.empty(0, np.int64))
        activity = {"control_tick": int(session.control_tick), "complete": False}
        raw["activity"].append(activity)
        # Each received array is published before validating the next readout.
        for key, attr in (("rate", "rate"), ("drive", "drive"), ("adaptation", "adaptation")):
            activity[key] = getattr(session.network, attr)[raw["ids"]].detach().cpu().numpy().copy()
        activity["synaptic_resource"] = session.network.resource.detach().cpu().numpy().copy()
        resource = activity["synaptic_resource"]
        raw["resource_description"] = {"minimum": float(np.min(resource)), "maximum": float(np.max(resource)),
            "maximum_absolute_change_from_one": float(np.max(np.abs(resource - 1))),
            "selection_criterion": False, "physiological_adjudication": False}
        validate_activity(activity, len(raw["ids"]), 7136)
        activity.update(network_tick=int(session.network.tick), physics_time_s=session.body.physics_time())
        if session.network.tick != session.control_tick * 20 or abs(session.body.physics_time() - session.control_tick * .005) > 2e-8:
            raise ValueError("Actual readout neural/physical clock differs")
        activity["complete"] = True
        if session._e291_arm == "neutral" and session.control_tick == 1:
            raw["neutral_end_snapshot"] = session.snapshot()

    def compare_neutral(self, raw, data):
        result = super().compare_neutral(raw, data)
        endpoint = compare_neutral_endpoint(
            layer.remove_resource_layer(raw["neutral_end_snapshot"]),
            data["baseline_end"], data["saved"]["wall_s"],
        )
        for activity in raw["activity"]:
            layer.require_equal(activity["synaptic_resource"], np.ones(7136, np.float64), "Neutral resource one")
        return {**result, **endpoint}

    def save_raw(self, folder, raw, data, clock):
        failures = []
        def attempt(name, callback):
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        attempt("native-contact.npz", lambda: np.savez_compressed(folder / "native-contact.npz", **native.native_arrays(raw["native"])))
        attempt("native-recruitment-state.npz", lambda: prior.monotone.save_native_state(folder / "native-recruitment-state.npz", raw["recruitment"]))
        attempt("control-raw.npz", lambda: short.save_stacked(folder / "control-raw.npz", raw["controls"],
            ("control_tick", "network_tick", "physics_time_s", "motor_rate32", "activation", "command", "tendon")))
        attempt("activity.npz", lambda: save_activity(folder / "activity.npz", raw["activity"], data))
        attempt("resource-description.json", lambda: self.write(folder / "resource-description.json", raw.get("resource_description", {})))
        attempt("trace.json", lambda: self.write(folder / "trace.json", {"schema": "flylab.m3.e291.trace.v1",
            "samples": raw["rows"], "graph_hash": data["graph"].hash,
            "controller": "original E150 with fixed recurrent resource neural layer", **raw.get("trace_metadata", {})}))
        attempt("clock-events.json", lambda: self.write(folder / "clock-events.json", clock.events))
        return failures



def bounded_write_receipt(path, value, *, writer=None, sleeper=None):
    """Retry only Windows publication conflicts, without replaying a control."""
    writer = receipts.write_receipt if writer is None else writer
    sleeper = time.sleep if sleeper is None else sleeper
    path = Path(path)
    first = None
    delays = (.005, .01, .02, .04, .08)
    for attempt in range(6):
        try:
            return writer(path, value)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            first = error if first is None else first
            retry = attempt < len(delays)
            event = {"path": path.as_posix(), "attempt": attempt + 1,
                     "winerror": error.winerror, "error": str(error),
                     "retry": retry, "delay_s": delays[attempt] if retry else 0.,
                     "observed_perf_counter_s": time.perf_counter()}
            try:
                with path.with_name(path.name + ".write-conflicts.jsonl").open(
                        "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            except Exception as evidence_error:
                first.add_note("Write-conflict evidence unavailable: " + repr(evidence_error))
                raise first from evidence_error
            if not retry:
                first.add_note("Atomic receipt still unavailable after six bounded attempts")
                raise first
            sleeper(delays[attempt])


def compare_neutral_endpoint(actual, baseline, starting_wall_s):
    """Independent wall measurements remain intact; all scientific fields exact."""
    values = (actual.get("wall_s"), baseline.get("wall_s"), starting_wall_s)
    if any(type(value) is not float or not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("Native finite nonnegative Python float wall observations required")
    actual_wall, baseline_wall, starting_wall = values
    if actual_wall < starting_wall or baseline_wall < starting_wall:
        raise ValueError("Endpoint observed wall cost precedes original starting wall")
    # New dictionaries exclude exactly one top-level field; nested wall fields,
    # arrays, physical/neural clocks and every other numeric leaf remain exact.
    a_changes, e_changes = [], []
    scientific_actual = prior.checkpoint_representation(
        {key: value for key, value in actual.items() if key != "wall_s"}, a_changes
    )
    scientific_baseline = prior.checkpoint_representation(
        {key: value for key, value in baseline.items() if key != "wall_s"}, e_changes
    )
    layer.require_equal(scientific_actual, scientific_baseline, "Whole neutral .005 scientific endpoint exact")
    return {
        "whole_end_exact": True, "excluded_fields": ["wall_s"],
        "actual_wall_s": actual_wall, "baseline_wall_s": baseline_wall,
        "starting_wall_s": starting_wall,
        "comparison_scope": "Every whole endpoint field except top-level wall_s; all nested fields exact",
        "wall_observations_preserved": True, "numeric_tolerance": 0, "array_coercions": 0,
        "actual_representation_changes": a_changes, "expected_representation_changes": e_changes,
    }


def validate_activity(record, cells, sources):
    for key, shape, dtype in (("rate", (cells,), np.float32), ("drive", (cells,), np.float32),
            ("adaptation", (cells,), np.float32), ("synaptic_resource", (sources,), np.float64)):
        value = record.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Native finite resource activity required: " + key)
    resource = record["synaptic_resource"]
    if np.any(resource < 0) or np.any(resource > 1):
        raise ValueError("Resource bounds [0,1] required")


def save_activity(path, records, data):
    arrays = {"observed_ids": data["ids"], "motor_ids": data["motor_ids"], "source_ids": data["source_ids"],
        "complete": np.asarray([r["complete"] for r in records], bool),
        "control_tick": np.asarray([r["control_tick"] for r in records], np.int64)}
    for key, count, dtype in (("rate", len(data["ids"]), np.float32), ("drive", len(data["ids"]), np.float32),
            ("adaptation", len(data["ids"]), np.float32), ("synaptic_resource", len(data["source_ids"]), np.float64)):
        arrays[key] = np.full((len(records), count), np.nan, dtype)
        for i, record in enumerate(records):
            value = record.get(key)
            if isinstance(value, np.ndarray) and value.shape == (count,) and value.dtype == dtype:
                arrays[key][i] = value
            elif key in record:
                # Invalid shape/dtype must remain inspectable without coercion.
                arrays[f"invalid_{key}_{i}"] = value
    for key, dtype in (("network_tick", np.int64), ("physics_time_s", np.float64)):
        arrays[key] = np.asarray([r.get(key, -1 if key == "network_tick" else np.nan) for r in records], dtype)
    np.savez_compressed(path, **arrays)


def run_arm(arm, data, freeze, out, ledger, deps, budget):
    folder = out / arm
    folder.mkdir()
    raw = {key: [] for key in ("rows", "controls", "recruitment", "activity", "native")}
    raw["ids"] = data["ids"]
    clock, holder = native.AdvanceLedger(), {}
    session = detach = journal = None
    primary, measured_stop, errors = None, None, []
    entry = {"status": "requested", "completed_controls": 0, "readout_endpoints": 0,
             "construction_requested": False, "construction_completed": False, "continuation_entered": False,
             "owned_session_recovered": False, "close_requested": False, "close_returned": False}
    ledger["arms"][arm] = entry
    metrics = {"startup": {"available": False}, "short": {"available": False}}
    def persist():
        deps.write(out / "advance-ledger.json", ledger)
    def fail(stage, error):
        nonlocal primary
        errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E291 " + stage + ": " + repr(error))
    try:
        budget()
        entry["construction_requested"] = True
        persist()
        session = deps.construct(arm, data, ledger, persist, lambda value: holder.setdefault("session", value))
        entry["construction_completed"] = True
        if hasattr(deps, "trace_metadata"):
            raw["trace_metadata"] = deps.trace_metadata(session)
        entry["initial_comparison"] = deps.initial(session, data, arm)
        deps.save_state(folder / "observed-start", session)
        detach = deps.install(session, raw["native"])
        journal = deps.journal(folder / "control-journal.jsonl")
        deps.capture(session, raw)
        entry["readout_endpoints"] = 1
        persist()
        last = 1 if arm == "neutral" else 300
        for control in range(1, last + 1):
            budget()
            before_native = len(raw["native"])
            try:
                event = deps.advance(session, arm, control, clock, journal)
            except deps.health_stop as error:
                measured_stop = {"control": control, "message": str(error), "native_samples": len(raw["native"])}
                break
            entry["completed_controls"] = control
            try:
                deps.capture(session, raw)
                deps.readout_receipt(journal, arm, control, event, True)
            except Exception as error:
                try:
                    deps.readout_receipt(journal, arm, control, event, False, error)
                except Exception as secondary:
                    error.add_note("Secondary E291 readout journal: " + repr(secondary))
                raise
            entry["readout_endpoints"] = control + 1
            if (session.control_tick != control or session.network.tick != control * 20
                    or abs(session.body.physics_time() - control * .005) > 2e-8
                    or len(raw["native"]) != before_native + 50):
                raise ValueError("Incomplete actual control/neural/native interval")
            persist()
            if session.fault or session.body.fault:
                measured_stop = {"control": control, "message": "Measured session/body fault"}
                break
            if arm == "candidate" and control == 100:
                metrics["startup"] = deps.startup(raw, data)
                if metrics["startup"].get("strict_E141_startup_screen") is not True:
                    break
                entry["continuation_entered"] = True
                persist()
        if arm == "neutral":
            if measured_stop is not None:
                raise ValueError("Neutral gate measured health stop")
            metrics["neutral_prefix"] = deps.compare_neutral(raw, data)
            raw.pop("neutral_end_snapshot", None)
        entry["status"] = "measured_native_health_stop" if measured_stop is not None else (
            "strict_startup_failed" if arm == "candidate" and not entry["continuation_entered"] else "science_complete")
    except Exception as error:
        fail("science", error)
        entry["traceback"] = traceback.format_exc()
    finally:
        if session is None and "session" in holder:
            session = holder["session"]
            entry["owned_session_recovered"] = True
        entry.update(native_samples=len(raw["native"]), health_stop=measured_stop,
                     actual_neural_s=clock.neural_s, actual_physical_s=clock.physical_s,
                     clock_events=clock.events, clock_valid=clock.clock_valid)
        for label, callback in (("native detach", detach), ("journal close", journal.close if journal else None)):
            if callback:
                try:
                    callback()
                except Exception as error:
                    fail(label, error)
        try:
            for failure in deps.save_raw(folder, raw, data, clock):
                fail("raw " + failure["artifact"], RuntimeError(failure["error"]))
        except Exception as error:
            fail("raw publication", error)
        if session is not None:
            try:
                deps.save_state(folder / "final", session)
            except Exception as error:
                # A guarded unhealthy snapshot is unavailable, not a new technical failure.
                if not (measured_stop is not None and isinstance(error, ValueError)):
                    fail("whole final state", error)
                entry["whole_final_unavailable"] = repr(error)
                try:
                    deps.save_partial(folder / "partial-final", session)
                except Exception as partial:
                    fail("partial state", partial)
            entry["close_requested"] = True
            try:
                session.close()
                entry["close_returned"] = True
            except Exception as error:
                fail("session close", error)
        if primary is None and arm == "candidate" and entry["completed_controls"] == 300:
            try:
                metrics["short"] = deps.short_metrics(raw, folder, clock, data)
            except Exception as error:
                fail("short analysis", error)
        entry["errors"] = errors
        if primary is not None:
            entry["status"] = "technical_or_resource_failure"
            metrics["validity"] = "invalid_or_incomplete"
            metrics["first_error"] = {"type": type(primary).__name__, "message": str(primary)}
        else:
            metrics["validity"] = "valid_recurrent_resource_body_diagnostic"
        metrics.update(status=entry["status"], completed_controls=entry["completed_controls"], health_stop=measured_stop)
        metrics["interpretation_validated"] = primary is None
        try:
            persist()
            deps.write(folder / "arm-result.json", metrics)
        except Exception as error:
            fail("arm publication", error)
            metrics.update(validity="invalid_or_incomplete", status="technical_or_resource_failure", interpretation_validated=False)
    return metrics, primary


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e291.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {
        "schema": "flylab.m3.e291.result.v1",
        "validity": "incomplete",
        "candidate_adopted": False,
        "diagnostic_id": freeze.get("diagnostic_id", "E291"),
        "run_valid": False,
        "neutral_exact": False,
        "actual_startup_pass": False,
        "short_gait_pass": False,
        "adopted": False,
        "next_separate_10s_eligible": False,
        "natural_gait_complete": False,
        "F1_F8_complete": False,
        "arm_receipt_path": "tools.verify_c_banc_m3_recurrent_resource_body.run_arm",
        "E272_executed": False,
        "F6_exposed": False,
        "F2_complete": False,
        "F3_F8_pass": False,
        "ten_second_executed": False,
        "arms": {},
    }
    errors, primary = [], None

    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E291 science deadline/reserve reached")
        if budget is not None:
            budget()
        elif hasattr(deps, "gpu_free_before"):
            import cupy
            import psutil

            memory = psutil.Process().memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            free, _ = cupy.cuda.runtime.memGetInfo()
            added = max(0, deps.gpu_free_before - free)
            ledger["peak_rss_bytes"] = max(peak, ledger.get("peak_rss_bytes", 0))
            ledger["peak_added_gpu_bytes"] = max(
                added, ledger.get("peak_added_gpu_bytes", 0)
            )
            if (
                peak > LIMITS["host_peak_bytes"]
                or added > LIMITS["additional_gpu_peak_bytes"]
            ):
                raise MemoryError("E291 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E291 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e291.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = run_arm(
            "neutral", data, freeze, out, ledger, deps, check
        )
        result["arms"]["neutral"] = neutral
        result["neutral_exact"] = (
            primary is None
            and neutral.get("neutral_prefix", {}).get("exact") is True
            and neutral.get("neutral_prefix", {}).get("whole_end_exact") is True
        )
        if primary is None and not result["neutral_exact"]:
            raise ValueError("Whole neutral exact comparison required before candidate")
        if result["neutral_exact"]:
            candidate, primary = run_arm(
                "candidate", data, freeze, out, ledger, deps, check
            )
            result["arms"]["candidate"] = candidate
            result["actual_startup_pass"] = (
                primary is None
                and candidate.get("startup", {}).get("strict_E141_startup_screen")
                is True
            )
            if primary is None and candidate.get("completed_controls") == 300:
                result["short_adjudication"] = assess_short(
                    data["baseline_short"], candidate.get("short", {})
                )
                result["short_gait_pass"] = result["short_adjudication"]["pass"]
            result["next_separate_10s_eligible"] = (
                result["actual_startup_pass"] and result["short_gait_pass"]
            )
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_recurrent_resource_body_diagnostic"
            if primary is None
            else "invalid_or_incomplete"
        )
    except Exception as error:
        primary = error if primary is None else primary
        errors.append(
            {
                "stage": "execution",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    if cleanup is not None:
        try:
            cleanup()
        except Exception as error:
            primary = primary or error
            errors.append(
                {
                    "stage": "runtime cleanup",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    if primary is not None:
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append(
            {
                "stage": "primary",
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": getattr(primary, "__notes__", []),
            }
        )
    result["elapsed_s"] = time.perf_counter() - began
    result["cost"] = {
        "scientific_neural_ticks": sum(
            round(v.get("actual_neural_s", 0) / 0.00025)
            for v in ledger["arms"].values()
            if v.get("actual_neural_s") is not None
        ),
        "scientific_physics_steps": sum(
            round(v.get("actual_physical_s", 0) / 0.0001)
            for v in ledger["arms"].values()
            if v.get("actual_physical_s") is not None
        ),
        "unknown_cost_arms": [
            k
            for k, v in ledger["arms"].items()
            if v.get("actual_neural_s") is None or v.get("actual_physical_s") is None
        ],
        "construction_calls": len(ledger["construction"]),
        "known_executed_setup_ticks": sum(
            v.get("known_completed_setup_steps", 0) for v in ledger["construction"]
        ),
        "recorded_capture_setup_ticks": sum(
            v.get("recorded_capture_setup_steps", 0) for v in ledger["construction"]
        ),
        "known_settling_physics_steps": sum(
            v.get("known_completed_settling_steps", 0)
            for v in ledger["body_construction"]
        ),
        "unknown_setup_calls": [
            v.get("label")
            for v in ledger["construction"]
            if v.get("unknown_setup_steps", 0)
            or v.get("unknown_capture_setup_steps", 0)
        ],
    }
    if (
        result["cost"]["scientific_neural_ticks"] > LIMITS["scientific_neural_ticks"]
        or result["cost"]["scientific_physics_steps"]
        > LIMITS["scientific_physics_steps"]
        or len(
            deps.sessions if hasattr(deps, "sessions") else ledger["body_construction"]
        )
        > 2
        or result["cost"]["unknown_cost_arms"]
    ):
        primary = primary or RuntimeError(
            "E291 scientific clocks/caps unavailable or exceeded"
        )
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append({"stage": "cost", "message": str(primary)})
    finalization_began = time.perf_counter()
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e291.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E291 final publication deadline")
        deps.write(out / "result.json", result)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E291 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        errors.append({"stage": "publication", "message": str(error)})
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            deps.write(out / "errors.json", {"errors": errors})
            deps.write(out / "result.json", result)
        except Exception as secondary:
            primary.add_note("Secondary E291 invalid publication: " + repr(secondary))
            (out / "result.json").unlink(missing_ok=True)
    return result, primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, output)
    import cupy
    import torch
    from threadpoolctl import threadpool_limits

    old_threads, old_env = torch.get_num_threads(), {}
    limiter, cleaned = None, False

    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        failures = []
        for callback in (
            (lambda: limiter.restore_original_limits()) if limiter else lambda: None,
            lambda: torch.set_num_threads(old_threads),
        ):
            try:
                callback()
            except Exception as error:
                failures.append(error)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if failures:
            for error in failures[1:]:
                failures[0].add_note("Secondary runtime cleanup: " + repr(error))
            raise failures[0]

    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_env[name] = os.environ.get(name)
            os.environ[name] = "1"
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        deps = NativeDependencies()
        deps.gpu_free_before, _ = cupy.cuda.runtime.memGetInfo()
        result, error = execute(freeze, output, deps=deps, cleanup=cleanup)
        return (
            0
            if error is None
            and result["validity"] == "valid_recurrent_resource_body_diagnostic"
            else 1
        )
    finally:
        cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
