"""E278: one rejected E277 candidate's staged native transfer diagnostic."""
# Broad catches retain the first error, partial clocks/raw and all cleanup costs.
# ruff: noqa: BLE001
from __future__ import annotations

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

from tools import analyze_c_banc_biological_gait as gait
from tools import banc_m3_source_filter_trial as layer
from tools import verify_c_banc_m3_e113_body as native
from tools import verify_c_banc_m3_lh_partition_body as short
from tools import verify_c_banc_m3_native_monotone_body as monotone
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools import verify_c_banc_m3_neural_current_body as receipts

FREEZE_SCHEMA = "flylab.m3.e278.source-native-transfer.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e278.source-native-transfer.review.v1"
OWNED = ("tools/verify_c_banc_m3_source_native_transfer.py", "tests/test_c_banc_source_native_transfer.py")
SOURCE_FILES = (*OWNED, "tools/banc_m3_source_filter_trial.py", "tools/banc_m3_source_filtered_synapses.py",
    "tools/banc_m3_monotone_recruitment.py", "tools/banc_m3_motor_timescale.py",
    "tools/verify_c_banc_m3_lh_partition_body.py", "tools/verify_c_banc_m3_e113_body.py",
    "tools/verify_c_banc_m3_native_monotone_body.py", "tools/verify_c_banc_m3_native_timescale_transfer.py",
    "tools/verify_c_banc_m3_neural_current_body.py", "tools/analyze_c_banc_biological_gait.py",
    "flylab/c/graph.py", "flylab/c/storage.py", "flylab/c/walking_trace.py", "flylab/c/banc_walking.py")
PATHS = {"graph_bundle", "neutral_checkpoint", "baseline_trace", "baseline_native", "baseline_recruitment",
         "baseline_short_result", "baseline_short_trace", "baseline_short_native", "startup", "candidate", "selection", "predictions"}
LEGS = protocol.LEGS
STARTUP_LIMITS = {"q_RMS": .018, "q_max_abs": .04, "LTM_RMS": .009, "LTM_max_abs": .025}
LIMITS = {"internal_wall_s": 115, "science_wall_s": 85, "finalization_reserve_s": 30,
          "host_peak_bytes": 6 << 30, "additional_gpu_peak_bytes": 4 << 30, "output_bytes": 2 << 30,
          "minimum_free_disk_bytes": 4 << 30, "original_attempts": 1, "retries": 0,
          "body_constructors": 2, "scientific_neural_ticks": 6020, "scientific_physics_steps": 15050,
          "cpu_threads": 1}


def safe(name):
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository relative frozen input required")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if protocol.sha256(freeze_path) != freeze_sha or protocol.sha256(review_path) != review_sha:
        raise ValueError("E278 freeze/source review digest mismatch")
    freeze, review = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("paths", {})) != PATHS or freeze.get("startup_limits") != STARTUP_LIMITS
            or freeze.get("resource_limits") != LIMITS
            or freeze.get("n") != 158706 or freeze.get("edges") != 1660
            or freeze.get("source_count") != 51 or freeze.get("motor_count") != 391
            or not isinstance(freeze.get("neutral_network_identity"), str)
            or not isinstance(freeze.get("candidate_network_identity"), str)
            or freeze.get("E277_status") != "rejected" or freeze.get("candidate_adopted") is not False):
        raise ValueError("E278 exact scientific/resource freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E278 approved matching source review required")
    pins = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"], review["source_sha256"],
                  freeze.get("preservation_sha256", {}), {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E278 hash pin")
            pins[name] = digest
    if not set(SOURCE_FILES).issubset(pins):
        raise ValueError("E278 required implementation source pins missing")
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or protocol.sha256(safe(name)) != digest:
            raise ValueError("E278 pinned bytes differ: " + name)
    for alias, value in freeze["paths"].items():
        paths = list(value.values()) if alias == "predictions" else [value]
        if alias == "predictions" and set(value) != set(LEGS):
            raise ValueError("Exactly six fixed E277 startup predictions required")
        for name in paths:
            path = safe(name)
            children = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
            if not children or any(p.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for p in children):
                raise ValueError("Every required input byte must be frozen: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E278 package differs: " + name)
    if output.resolve() != safe(freeze["output"]) or output.exists() or not output.parent.is_dir():
        raise ValueError("E278 fresh fixed output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E278 disk reserve unavailable")
    prediction_columns(freeze)
    return freeze


def prediction_columns(freeze):
    q, ltm = freeze["prediction_joint_columns"], freeze["prediction_ltm_columns"]
    if set(q) != set(LEGS) or set(ltm) != set(LEGS):
        raise ValueError("Six explicit prediction leg column mappings required")
    if (any(len(q[leg]) != 7 or len(ltm[leg]) != 1 for leg in LEGS)
            or sorted(column for leg in LEGS for column in q[leg]) != list(range(42))
            or sorted(column for leg in LEGS for column in ltm[leg]) != list(range(6))):
        raise ValueError("Exact joint/tendon partition required")
    return q, ltm


def checkpoint_representation(value, changes, path="state"):
    """Match StateStore.pack scalars/sequences; preserve every ndarray bit."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: checkpoint_representation(item, changes, path + "." + key)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if isinstance(value, tuple):
            changes.append({"path": path, "from": "tuple", "to": "list"})
        return [checkpoint_representation(item, changes, path + "." + str(i))
                for i, item in enumerate(value)]
    if isinstance(value, np.generic):
        converted = value.item()
        changes.append({"path": path, "from": type(value).__name__, "to": type(converted).__name__})
        return converted
    return value


def phase_metrics(actual_q, actual_ltm, predicted_q, predicted_ltm, freeze):
    """Original controls1..99 only; retain explicit joint ownership and windows."""
    aq, al, pq, pl = (np.asarray(value, np.float64) for value in
                      (actual_q, actual_ltm, predicted_q, predicted_ltm))
    if (aq.shape != (101, 42) or pq.shape != aq.shape or al.shape != (101, 6)
            or pl.shape != al.shape or not all(np.isfinite(value).all() for value in (aq, al, pq, pl))):
        raise ValueError("Complete finite101endpoint predictions/commands required")
    qcols, lcols = prediction_columns(freeze)
    qe, le = aq[1:100] - pq[1:100], al[1:100] - pl[1:100]
    def error(value):
        return {"RMS": float(np.sqrt(np.mean(value ** 2))), "max_abs": float(np.max(np.abs(value)))}
    report = {"q": error(qe), "LTM": error(le), "controls": [1, 99], "legs": {}, "windows": []}
    for leg in LEGS:
        report["legs"][leg] = {"q": error(qe[:, qcols[leg]]), "LTM": error(le[:, lcols[leg]])}
    for block in range(5):
        # Integer control IDs avoid assigning 0.1/0.2 boundary samples by roundoff.
        first, last = max(1, block * 20), min(99, (block + 1) * 20 - 1)
        selected = slice(first - 1, last)
        report["windows"].append({"start_s": block * .1, "end_s": (block + 1) * .1,
            "first_control": first, "last_control": last, "samples": last - first + 1,
            "q": error(qe[selected]), "LTM": error(le[selected])})
    report["material_prediction_mismatch"] = bool(report["q"]["RMS"] > .0018
        or report["q"]["max_abs"] > .004 or report["LTM"]["RMS"] > .0009 or report["LTM"]["max_abs"] > .0025)
    return report, {"q_error": qe, "ltm_error": le, "controls": np.arange(1, 100, dtype=np.int64)}


class NativeDependencies:
    """Actual integrations; generated tests inject their own complete interface."""
    health_stop = native.NativeHealthStop

    def load(self, freeze):
        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore

        paths = freeze["paths"]
        data = {"graph": GraphStore.load(safe(paths["graph_bundle"])),
                "saved": StateStore.load(safe(paths["neutral_checkpoint"])),
                "selection": protocol.load_npz(safe(paths["selection"])),
                "candidate": protocol.load_npz(safe(paths["candidate"])),
                "startup": protocol.load_npz(safe(paths["startup"])),
                "baseline_rows": json.loads(safe(paths["baseline_trace"]).read_text(encoding="utf-8"))["samples"],
                "baseline_native": protocol.load_npz(safe(paths["baseline_native"])),
                "baseline_recruitment": protocol.load_npz(safe(paths["baseline_recruitment"])),
                "baseline_short": json.loads(safe(paths["baseline_short_result"]).read_text(encoding="utf-8"))}
        c = data["candidate"]["coefficients32"]
        if c.shape != (1660, 3) or c.dtype != np.float32 or not np.isfinite(c).all():
            raise ValueError("Fixed finite E277 coefficients32 required")
        data["coefficients"] = c.astype(np.float64)
        selection = data["selection"]
        # E267's persisted source-selection schema names the eligible roster
        # allowed_motor_ids; E267's runtime validator calls it free_motor_ids.
        allowed = selection["allowed_motor_ids"]
        if (selection["edge_ids"].shape != (1660,) or selection["source_ids"].shape != (51,)
                or allowed.shape != (343,) or allowed.dtype != np.int64
                or selection["all_motor_ids"].shape != (391,)
                or not np.isin(selection["post_ids"], allowed).all()):
            raise ValueError("Exact original E267 allowed/source/edge roster required")
        data["allowed_motor_ids"] = allowed.copy()
        layer.validate_neutral(data["saved"]["network"])
        if (data["graph"].hash != freeze["graph_hash"]
                or data["saved"]["network"]["model_hash"] != freeze["source_network_identity"]
                or data["saved"]["adapter"]["hash"] != freeze["adapter_identity"]):
            raise ValueError("Frozen original graph/neural/adapter identity differs")
        qcols, lcols = prediction_columns(freeze)
        pq, pl = np.empty((101, 42)), np.empty((101, 6))
        for leg in LEGS:
            prediction = protocol.load_npz(safe(paths["predictions"][leg]))
            q, ltm = prediction["q"], prediction["ltm"]
            if q.shape != (101, 7) or ltm.shape != (101, 1) or not np.isfinite(q).all() or not np.isfinite(ltm).all():
                raise ValueError("Complete fixed conditional leg prediction required")
            pq[:, qcols[leg]], pl[:, lcols[leg]] = q, ltm
        data.update(predicted_q=pq, predicted_ltm=pl, freeze=freeze)
        data["source_ids"] = data["selection"]["source_ids"]
        data["motor_ids"] = data["saved"]["monotone_recruitment_patch"]["motor_ids"]
        if data["source_ids"].shape != (51,) or data["motor_ids"].shape != (391,):
            raise ValueError("Exactly51source and391motor observers required")
        data["ids"] = np.unique(np.r_[data["source_ids"], data["motor_ids"]]).astype(np.int64)
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            base = layer.MonotoneRecruitmentTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            try:
                coefficients = (np.tile(np.array([1, 0, 0], np.float64), (1660, 1))
                                if arm == "neutral" else data["coefficients"].copy())
                session = layer.SourceFilterTrial(base, data["selection"]["edge_ids"],
                    coefficients, data["allowed_motor_ids"])
                return session
            except Exception as error:
                layer.cleanup_owned(base, error)
                raise
        return short.construction_with_body_count(arm, callback, ledger, persist, on_return)

    def initial(self, session, data, arm):
        saved = session.snapshot()
        actual_changes, expected_changes = [], []
        actual = checkpoint_representation(layer.remove_source_layer(saved), actual_changes)
        expected = checkpoint_representation(data["saved"], expected_changes)
        layer.require_equal(actual, expected, "Whole original neutral stored representation")
        layer.validate_neutral(data["saved"]["network"])
        if session.network.identity != data["freeze"][arm + "_network_identity"]:
            raise ValueError("E278 fixed constructed source filter identity differs")
        if (session.control_tick != 0 or session.network.tick != 0
                or abs(session.body.physics_time()) > 2e-8
                or session.network.device != "cuda" or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20 or np.any(saved["network"]["source_history"] != 0)):
            raise ValueError("E278 actual CUDA zero-clock/history neutral start required")
        return {"storage_representation_exact": True, "actual_representation_changes": actual_changes,
                "expected_representation_changes": expected_changes, "numeric_tolerance": 0,
                "array_coercions": 0, "excluded_fields": []}

    def install(self, session, native_records):
        return native.install_native_observer(session.body, native_records)

    def trace_metadata(self, session):
        return {"body_hash": session.body.model_hash, "model_hash": session.network.identity,
                "adapter_hash": session.adapter.identity, "parameters": session.p.state_dict()}

    def capture(self, session, raw):
        row = protocol.capture_control(session)
        raw["rows"].append(row)
        raw["controls"].append(short.capture_control_raw(session, row))
        monotone.make_observer(raw["recruitment"], original=lambda *_args: {})(session, np.empty(0, np.int64))
        activity = {"control_tick": int(session.control_tick), "complete": False}
        raw["activity"].append(activity)
        activity.update(short.capture_activity(session, raw["ids"]))
        activity["source_history"] = session.network.history.detach().cpu().numpy().copy()
        if activity["source_history"].shape != (51, 2) or not np.isfinite(activity["source_history"]).all():
            raise ValueError("Actual finite51source history required")
        for key in ("rate", "drive", "adaptation"):
            if activity[key].shape != (len(raw["ids"]),) or not np.isfinite(activity[key]).all():
                raise ValueError("Actual selected neural observation invalid")
        activity["complete"] = True

    def compare_neutral(self, raw, data):
        from tools.verify_c_banc_observation import exact

        if len(raw["rows"]) != 2 or len(raw["native"]) != 50:
            raise ValueError("Complete neutral .005 prefix required")
        exact(raw["rows"], data["baseline_rows"][:2], "Original E150 control0..1 trace")
        observed = native.native_arrays(raw["native"])
        for key, value in observed.items():
            layer.require_equal(value, data["baseline_native"][key][:50], "Original native50 " + key)
        records = raw["recruitment"]
        for key in ("bank", "activation", "original_activation", "motor_rate", "control_tick", "network_tick", "complete"):
            actual = np.asarray([record[key] for record in records])
            layer.require_equal(actual, data["baseline_recruitment"][key][:2], "Original recruitment2 " + key)
        layer.require_equal(np.asarray([record["time_s"] for record in records]),
                            data["baseline_recruitment"]["physics_time_s"][:2], "Original recruitment physical clocks")
        return {"exact": True, "control_rows": 2, "native_samples": 50, "recruitment_rows": 2}

    def startup(self, raw, data):
        return protocol.strict_startup(raw["rows"], raw["native"], data["startup"], STARTUP_LIMITS)

    def save_state(self, path, session):
        protocol.StateStore.save(path, session.snapshot())

    def save_partial(self, path, session):
        # An unhealthy whole snapshot may be forbidden: retain components independently.
        from flylab.c.storage import StateStore

        failures, causes = [], []
        for label in ("network", "adapter", "body"):
            try:
                saved = getattr(session, label).snapshot()
                StateStore.save(path / label, saved)
            except Exception as error:
                failures.append({"component": label, "type": type(error).__name__,
                                 "message": str(error), "error": repr(error)})
                causes.append(error)
        if failures:
            aggregate = RuntimeError("Required partial checkpoint components unavailable: "
                                     + ", ".join(row["component"] for row in failures))
            aggregate.component_errors = copy.deepcopy(failures)
            for row in failures:
                aggregate.add_note("Partial " + row["component"] + ": " + row["error"])
            try:
                path.mkdir(parents=True, exist_ok=True)
                receipts.write_receipt(path / "component-errors.json", failures)
            except Exception as journal_error:
                aggregate.add_note("Partial component error journal failed: " + repr(journal_error))
            raise aggregate from causes[0]

    def journal(self, path):
        return receipts.FiniteControlJournal(path)

    def advance(self, session, arm, index, clock, journal):
        return protocol.advance_control(session, arm, index, clock, journal)

    def readout_receipt(self, journal, arm, index, event, complete, error=None):
        protocol.record_readout(journal, arm, index, event, complete=complete, error=error)

    def write(self, path, value):
        receipts.write_receipt(path, value)

    def save_raw(self, folder, raw, data, clock):
        failures = []
        def attempt(name, callback):
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        attempt("native-contact.npz", lambda: np.savez_compressed(folder / "native-contact.npz", **native.native_arrays(raw["native"])))
        attempt("native-recruitment-state.npz", lambda: monotone.save_native_state(folder / "native-recruitment-state.npz", raw["recruitment"]))
        attempt("control-raw.npz", lambda: short.save_stacked(folder / "control-raw.npz", raw["controls"],
            ("control_tick", "network_tick", "physics_time_s", "motor_rate32", "activation", "command", "tendon")))
        def save_activity():
            arrays = {"observed_ids": raw["ids"], "motor_ids": data["motor_ids"], "source_ids": data["source_ids"],
                      "complete": np.asarray([r["complete"] for r in raw["activity"]], bool),
                      "control_tick": np.asarray([r["control_tick"] for r in raw["activity"]], np.int64)}
            for key, shape, dtype in (("rate", (len(raw["ids"]),), np.float32), ("drive", (len(raw["ids"]),), np.float32),
                                     ("adaptation", (len(raw["ids"]),), np.float32), ("source_history", (51, 2), np.float64)):
                arrays[key] = np.full((len(raw["activity"]), *shape), np.nan, dtype)
                for i, record in enumerate(raw["activity"]):
                    if key in record:
                        arrays[key][i] = record[key]
            np.savez_compressed(folder / "activity.npz", **arrays)
        attempt("activity.npz", save_activity)
        attempt("trace.json", lambda: self.write(folder / "trace.json", {"schema": "flylab.m3.e278.trace.v1",
            "samples": raw["rows"], "graph_hash": data["graph"].hash, "controller": "original E150 with source neural layer",
            **raw.get("trace_metadata", {})}))
        attempt("clock-events.json", lambda: self.write(folder / "clock-events.json", clock.events))
        return failures

    def short_metrics(self, raw, folder, clock, data):
        arrays = native.native_arrays(raw["native"])
        complete = len(raw["rows"]) == 301 and len(raw["native"]) == 15000
        result = {"health": short.short_health(raw["rows"], arrays, clock, bool(raw["rows"][-1].get("fault"))),
                  "forward": short.short_forward(raw["rows"], complete=complete),
                  "slip": short.short_window_slip(arrays, complete=complete, minimum_load=.02)}
        if complete:
            result["gait"], _, _ = gait.analyze(gait.load_model(folder / "trace.json"))
        result["baseline"] = data["baseline_short"]
        return result


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
    metrics = {"startup": {"available": False}, "prediction": {"available": False}, "short": {"available": False}}
    def persist():
        deps.write(out / "advance-ledger.json", ledger)
    def fail(stage, error):
        nonlocal primary
        errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E278 " + stage + ": " + repr(error))
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
                    error.add_note("Secondary E278 readout journal: " + repr(secondary))
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
                q = np.asarray([r["joint_targets_rad"] for r in raw["rows"]], np.float64)
                ltm = np.asarray([[r["tendon_inputs"][f"{leg}_tarsus"] for leg in LEGS] for r in raw["rows"]], np.float64)
                metrics["prediction"], effects = phase_metrics(q, ltm, data["predicted_q"], data["predicted_ltm"], freeze)
                np.savez(folder / "prediction-effects.npz", **effects)
                if metrics["startup"].get("strict_E141_startup_screen") is not True:
                    break
                entry["continuation_entered"] = True
                persist()
        if arm == "neutral":
            if measured_stop is not None:
                raise ValueError("Neutral gate measured health stop")
            metrics["neutral_prefix"] = deps.compare_neutral(raw, data)
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
            metrics["validity"] = "valid_transfer_diagnostic"
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
    ledger = {"schema": "flylab.m3.e278.ledger.v1", "arms": {}, "construction": [], "body_construction": [],
              "original_attempts": 1, "retries": 0, "fit_calls": 0, "nonneutral_current_calls": 0}
    result = {"schema": "flylab.m3.e278.result.v1", "validity": "incomplete", "candidate_adopted": False,
              "diagnostic_id": freeze.get("diagnostic_id", "E278"),
              "E277_status": "rejected", "E272_executed": False, "F6_exposed": False,
              "F2_complete": False, "F3_F8_pass": False, "ten_second_executed": False, "arms": {}}
    errors, primary = [], None
    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E278 science deadline/reserve reached")
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
            ledger["peak_added_gpu_bytes"] = max(added, ledger.get("peak_added_gpu_bytes", 0))
            if peak > LIMITS["host_peak_bytes"] or added > LIMITS["additional_gpu_peak_bytes"]:
                raise MemoryError("E278 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E278 output disk reserve reached")
    try:
        deps.write(out / "source-manifest.json", {"schema": "flylab.m3.e278.sources.v1",
            "input_sha256": freeze.get("input_sha256", {}), "source_sha256": freeze.get("reused_source_sha256", {}),
            "owned_sha256": {name: protocol.sha256(ROOT / name) for name in OWNED}})
        data = deps.load(freeze)
        neutral, primary = run_arm("neutral", data, freeze, out, ledger, deps, check)
        result["arms"]["neutral"] = neutral
        if primary is None and neutral.get("neutral_prefix", {}).get("exact") is True:
            candidate, primary = run_arm("candidate", data, freeze, out, ledger, deps, check)
            result["arms"]["candidate"] = candidate
        result["validity"] = "valid_transfer_diagnostic" if primary is None else "invalid_or_incomplete"
    except Exception as error:
        primary = error if primary is None else primary
        errors.append({"stage": "execution", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    if cleanup is not None:
        try:
            cleanup()
        except Exception as error:
            primary = primary or error
            errors.append({"stage": "runtime cleanup", "type": type(error).__name__, "message": str(error)})
    if primary is not None:
        result.update(validity="invalid_or_incomplete", first_error={"type": type(primary).__name__, "message": str(primary)})
        errors.append({"stage": "primary", "type": type(primary).__name__, "message": str(primary),
                       "notes": getattr(primary, "__notes__", [])})
    result["elapsed_s"] = time.perf_counter() - began
    result["cost"] = {"scientific_neural_ticks": sum(round(v.get("actual_neural_s", 0) / .00025)
        for v in ledger["arms"].values() if v.get("actual_neural_s") is not None),
        "scientific_physics_steps": sum(round(v.get("actual_physical_s", 0) / .0001)
        for v in ledger["arms"].values() if v.get("actual_physical_s") is not None),
        "unknown_cost_arms": [k for k, v in ledger["arms"].items() if v.get("actual_neural_s") is None or v.get("actual_physical_s") is None],
        "construction_calls": len(ledger["construction"]),
        "known_executed_setup_ticks": sum(v.get("known_completed_setup_steps", 0) for v in ledger["construction"]),
        "recorded_capture_setup_ticks": sum(v.get("recorded_capture_setup_steps", 0) for v in ledger["construction"]),
        "known_settling_physics_steps": sum(v.get("known_completed_settling_steps", 0) for v in ledger["body_construction"]),
        "unknown_setup_calls": [v.get("label") for v in ledger["construction"]
            if v.get("unknown_setup_steps", 0) or v.get("unknown_capture_setup_steps", 0)]}
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e278.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if time.perf_counter() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E278 final publication deadline")
        deps.write(out / "result.json", result)
        if time.perf_counter() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E278 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(validity="invalid_or_incomplete", first_error={"type": type(primary).__name__, "message": str(primary)})
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        errors.append({"stage": "publication", "message": str(error)})
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            deps.write(out / "errors.json", {"errors": errors})
            deps.write(out / "result.json", result)
        except Exception as secondary:
            primary.add_note("Secondary E278 invalid publication: " + repr(secondary))
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
        for callback in ((lambda: limiter.restore_original_limits()) if limiter else lambda: None,
                         lambda: torch.set_num_threads(old_threads)):
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
        return 0 if error is None and result["validity"] == "valid_transfer_diagnostic" else 1
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
    return run(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
