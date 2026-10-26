"""E307 one neutral SSP body for exactly at most 100 controls; no continuation."""
from __future__ import annotations

# Preserve first scientific exceptions and secondary cleanup/publication failures.
# ruff: noqa: BLE001
import argparse
import copy
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_ssp_recovery_body as e306

runner, e301, e305 = e306.runner, e306.e301, e306.e305
OWNED = ("tools/verify_c_banc_m3_ssp_neutral_body.py", "tests/test_c_banc_ssp_neutral_body.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *e306.SOURCE_FILES)))
PATHS, POLICY, STARTUP_LIMITS = e306.PATHS, e306.POLICY, copy.deepcopy(e306.STARTUP_LIMITS)
BASE_KEYS = tuple(key for key in e306.BASE_KEYS if key != "resource_limits")
LIMITS = {**copy.deepcopy(e306.LIMITS), "internal_wall_s": 55, "science_wall_s": 40, "finalization_reserve_s": 15,
          "body_constructors": 1, "scientific_neural_ticks": 2000, "scientific_physics_steps": 5000}
SUPERVISOR_WALL_S = 65
FREEZE_SCHEMA = "flylab.m3.e307.ssp-neutral-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e307.ssp-neutral-body.review.v1"
RESULT_SCHEMA = "flylab.m3.e307.ssp-neutral-body.result.v1"
DESIGN = "docs/BANC_M3_E307_SSP_NEUTRAL_DESIGN_20260928.md"
DESIGN_SHA = "094ed7849140db8d2015759481de16b832942b3c8b6bc45adab5e80180aa038d"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e307-ssp-neutral-body"
E306_SHAS = {"freeze": "ed943af832cc4d97a845e53aabecb9b13792adfcec1a8fd9ac391cdc767c08a6",
              "review": "f824fae64ff3faa10e142efac1f6054710c9596d1a2245fce7b4b57719baab6e",
              "result": "6be280252bfa8a0a3978b0b63405f68474c5899a7615a6ad287e9c38a882e406",
              "output_manifest": "836c7a7e3fa0d3da01e9e02cf726fdf3965d4a68d52a81c61adc08957d64ccb4",
              "root_audit": "c98fad1ea12db00a642378cf6c81b97e20d16af4cfd4b1c2fa1c6497f02e0f10"}
RECEIPT_PATHS = {"freeze": e306.OUTPUT_PATH + "-freeze.json", "review": e306.OUTPUT_PATH + "-source-review.json",
                 **{key: e306.OUTPUT_PATH + "/" + filename for key, filename in
                    (("result", "result.json"), ("ledger", "advance-ledger.json"), ("source_manifest", "source-manifest.json"),
                     ("output_manifest", "output-manifest.json"), ("result_sha256", "result-sha256.json"))},
                 "root_audit": "verification/banc-natural-gait-m3-20260919/parent-e306-original-raw-audit.json"}
COMPARATOR_PATHS = {"original_trace": "verification/banc-natural-gait-m3-20260919/e150-native-monotone-body/trace.json",
                    "e306_candidate_trace": e306.OUTPUT_PATH + "/candidate/trace.json",
                    "e306_neutral_first_neural": e306.OUTPUT_PATH + "/neutral/first-control-neural"}
safe = e306.safe


def validate_e306(result, ledger, audit):
    expected = {"construction_calls": 2, "known_executed_setup_ticks": 560, "recorded_capture_setup_ticks": 280,
                "known_settling_physics_steps": 4000, "scientific_neural_ticks": 2020, "scientific_physics_steps": 5050,
                "unknown_cost_arms": [], "unknown_setup_calls": []}
    flags = ("candidate_adopted", "adopted", "F1_F8_complete", "F2_complete", "F3_F8_pass", "F6_exposed", "natural_gait_complete",
             "ten_second_executed", "actual_startup_pass", "short_gait_pass", "next_separate_10s_eligible")
    if (result.get("schema") != e306.RESULT_SCHEMA or result.get("diagnostic_id") != "E306" or result.get("run_valid") is not True
            or result.get("publication_status") != "complete" or any(result.get(key) is not False for key in flags)
            or result.get("neutral_prefix_valid") is not True or result.get("ssp_neural_endpoint_exact") is not True
            or audit.get("schema") != "flylab.m3.e306.root-original-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True or audit.get("candidate_rejected") is not True
            or any(audit.get(key) is not False for key in ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "biological_claim", "actual_startup_pass"))):
        raise ValueError("Valid audited unpromoted E306 startup rejection required")
    if result.get("cost") != expected or audit.get("cost") != expected or ledger.get("original_attempts") != 1 or ledger.get("retries") != 0 or ledger.get("fit_calls") != 0:
        raise ValueError("Exact known E306 scientific/setup/body accounting required")
    for arm, controls, ticks, samples in (("neutral", 1, 20, 50), ("candidate", 100, 2000, 5000)):
        measured, entry = audit["arms"][arm], ledger["arms"][arm]
        if (measured.get("controls") != controls or measured.get("neural_ticks") != ticks or measured.get("native_samples") != samples
                or measured.get("first_neural_endpoint_exact") is not True or measured.get("original_whole_start_exact") is not True
                or measured.get("closed") is not True or entry.get("completed_controls") != controls or entry.get("native_samples") != samples
                or entry.get("close_returned") is not True or entry.get("clock_valid") is not True or entry.get("continuation_entered") is not False
                or round(entry.get("actual_neural_s", -1) / .00025) != ticks or round(entry.get("actual_physical_s", -1) / .0001) != samples):
            raise ValueError("Actual E306 exact completed and closed arms required")
    construction, physical = ledger["construction"], ledger["body_construction"]
    if (len(construction) != 2 or len(physical) != 2
            or sum(row["known_completed_setup_steps"] for row in construction) != 560
            or sum(row["recorded_capture_setup_steps"] for row in construction) != 280
            or any(row.get("unknown_setup_steps") or row.get("unknown_capture_setup_steps") for row in construction)
            or sum(row["known_completed_settling_steps"] for row in physical) != 4000
            or any(row.get("unknown_step_entries") or row.get("active_unreturned_frames") for row in physical)):
        raise ValueError("Exact E306 constructor and settling records required")
    if result["arms"]["candidate"]["startup"].get("strict_E141_startup_screen") is not False:
        raise ValueError("Original E306 failed strict startup must remain failed")


def e306_lineage(freeze, pins):
    specs = freeze["e306_lineage"]
    if set(specs) != set(RECEIPT_PATHS) or freeze.get("comparator_paths") != COMPARATOR_PATHS:
        raise ValueError("Exact E306 receipt and comparator paths required")
    for alias, name in RECEIPT_PATHS.items():
        if specs[alias]["path"] != name or alias in E306_SHAS and specs[alias]["sha256"] != E306_SHAS[alias]:
            raise ValueError("Literal frozen actual E306 lineage required: " + alias)
    records = {alias: e305.receipt(spec, pins) for alias, spec in specs.items()}
    old, review, source = records["freeze"], records["review"], records["source_manifest"]
    if (old.get("schema") != e306.FREEZE_SCHEMA or old.get("diagnostic_id") != "E306" or old.get("resource_limits") != e306.LIMITS
            or old.get("startup_limits") != STARTUP_LIMITS or old.get("policy") != POLICY
            or review.get("schema") != e306.REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != E306_SHAS["freeze"] or set(review.get("source_sha256", {})) != set(e306.OWNED)
            or source.get("schema") != "flylab.m3.e306.sources.v1" or source.get("owned_sha256") != review["source_sha256"]
            or source.get("input_sha256") != old["input_sha256"]
            or source.get("executed_body_runner") != "tools.verify_c_banc_m3_ssp_recovery_body.execute"
            or source.get("reused_arm_runner") != "tools.verify_c_banc_m3_recurrent_resource_body.run_arm"):
        raise ValueError("Exact approved and actually executed E306 sources required")
    e306.require_pins(pins, old["input_sha256"], old["reused_source_sha256"], old["preservation_sha256"],
                       review["source_sha256"], source["source_sha256"], {old["design"]: old["design_sha256"]})
    baseline, inherited = e301.base_lineage(old)
    diagnostic = e301.diagnostic_lineage(old, baseline)
    e306.require_pins(pins, inherited, diagnostic, baseline["input_sha256"])
    for key in e306.BASE_KEYS:
        if old.get(key) != baseline[key]:
            raise ValueError("Every original E296 context entry required")
    if old["paths"] != baseline["paths"] or set(old["paths"]) != PATHS:
        raise ValueError("Exact original E296 input path chain required")
    ssp = e306.ssp_lineage(old, pins)
    e306.validate_packages(old["packages"], baseline["packages"], ssp["e305_freeze"]["packages"])
    validate_e306(records["result"], records["ledger"], records["root_audit"])
    manifest, result_pin, audit = records["output_manifest"], records["result_sha256"], records["root_audit"]
    if (manifest.get("schema") != "flylab.m3.e306.output-manifest.v1" or result_pin != {"schema": "flylab.m3.e306.result-sha256.v1", "path": "result.json", "sha256": E306_SHAS["result"]}
            or audit.get("result_sha256") != E306_SHAS["result"] or audit.get("output_manifest_sha256") != E306_SHAS["output_manifest"]):
        raise ValueError("Audited E306 independent result and manifest bindings required")
    for name, entry in manifest["files"].items():
        path = safe(e306.OUTPUT_PATH + "/" + name)
        if pins.get(path.relative_to(ROOT).as_posix()) != entry["sha256"] or runner.protocol.sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Every actual E306 output manifest byte required")
    for folder in (COMPARATOR_PATHS["e306_neutral_first_neural"],):
        files = [path for path in safe(folder).rglob("*") if path.is_file()]
        if not files or any(path.relative_to(ROOT).as_posix() not in pins for path in files):
            raise ValueError("Complete actual E306 neutral endpoint bytes required")
    for key in (*BASE_KEYS, "packages", "paths", "ssp_lineage", "ssp_checkpoints", "neutral_network_identity", "candidate_network_identity"):
        if freeze.get(key) != old[key]:
            raise ValueError("Exact inherited E306 context required: " + key)
    return records


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E307 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E307" or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS or freeze.get("supervisor_wall_s") != SUPERVISOR_WALL_S
            or freeze.get("startup_limits") != STARTUP_LIMITS or freeze.get("policy") != POLICY or freeze.get("candidate_adopted") is not False
            or freeze.get("output") != OUTPUT_PATH or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA):
        raise ValueError("Exact prospective one-body neutral E307 contract required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E307 source review required")
    pins = e301.base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"], review["source_sha256"], {DESIGN: DESIGN_SHA})
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("Every actual E307 execution source pin required")
    e306_lineage(freeze, pins)
    for name in freeze["paths"].values():
        path = safe(name)
        files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else [path]
        if not files or any(item.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for item in files):
            raise ValueError("Every inherited original input byte required")
    for name, version in freeze["packages"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        if actual != version:
            raise ValueError("Pinned actual E307 package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh literal E307 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E307 disk reserve unavailable")
    return freeze


def trace_arrays(rows, *, limit=101):
    if len(rows) < limit:
        raise ValueError("Complete matching half-second trace required")
    prefix = rows[:limit]
    controls = np.asarray([row["control_tick"] for row in prefix])
    times = np.asarray([row["time_s"] for row in prefix])
    if (controls.dtype != np.int64 or not np.array_equal(controls, np.arange(limit, dtype=np.int64))
            or times.dtype != np.float64 or not np.isfinite(times).all() or not np.array_equal(times, np.arange(limit) * .005)):
        raise ValueError("Exact native trace control and nominal time clocks required")
    arrays = {"control_tick": controls, "time_s": times}
    for key, field, width in (("q", "joint_targets_rad", 42), ("joint_angles", "joint_angles_rad", 42)):
        value = np.asarray([row[field] for row in prefix])
        if value.dtype != np.float64 or value.shape != (limit, width) or not np.isfinite(value).all():
            raise ValueError("Finite native float64 aligned trace required: " + key)
        arrays[key] = value
    order = tuple(f"{leg}_tarsus" for leg in runner.LEGS)
    tendon = np.asarray([[row["tendon_inputs"][name] for name in order] for row in prefix])
    if tendon.dtype != np.float64 or tendon.shape != (limit, 6) or not np.isfinite(tendon).all():
        raise ValueError("Finite native float64 aligned LTM trace required")
    arrays["LTM"] = tendon
    return arrays


def compare_traces(actual_rows, reference_rows, label):
    actual, reference = trace_arrays(actual_rows), trace_arrays(reference_rows)
    deltas, summaries = {}, {}
    for key in ("control_tick", "time_s"):
        runner.layer.require_equal(actual[key], reference[key], "Aligned same observation " + key)
    deltas.update(control_tick=actual["control_tick"], time_s=actual["time_s"])
    for key in ("q", "LTM", "joint_angles"):
        delta, summary = e306.numeric_delta(actual[key], reference[key], actual[key].shape, label + " " + key)
        row, index = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        summary.update(max_control_tick=int(actual["control_tick"][row]), max_time_s=float(actual["time_s"][row]), max_index=int(index),
                       signed_delta_at_max=float(delta[row, index]), actual_at_max=float(actual[key][row, index]), reference_at_max=float(reference[key][row, index]),
                       selection_criterion=False)
        deltas[key + "_delta"] = delta
        summaries[key] = summary
    return deltas, summaries


def validate_neutral_utilization(data):
    sources, zero = data["source_ids"], data["roster"]["neutral_utilization"]
    if (not isinstance(sources, np.ndarray) or sources.dtype != np.int64 or sources.ndim != 1 or not len(sources)
            or np.any(sources < 0) or np.any(sources[1:] <= sources[:-1])
            or not isinstance(zero, np.ndarray) or zero.dtype != np.float64 or zero.shape != sources.shape
            or zero.tobytes() != np.zeros_like(zero).tobytes()):
        raise ValueError("Exact native ordered neutral source roster and positive-zero utilization required")
    runner.layer.require_equal(sources, data["roster"]["source_ids"], "Exact neutral source order")


def require_neutral_factors(arrays):
    runner.layer.require_equal(arrays["g32"], np.ones_like(arrays["g32"]), "Every completed neutral endpoint factor one")


class NativeDependencies(e306.NativeDependencies):
    def __init__(self, *, trace_loader=None, **kwargs):
        super().__init__(**kwargs)
        self._trace_loader = trace_loader
        self.e306_first_reference = None

    def load(self, freeze):
        data = super().load(freeze)
        validate_neutral_utilization(data)
        loader = self._checkpoint_loader
        if loader is None:
            from flylab.c.storage import StateStore
            loader = StateStore.load
        self.e306_first_reference = loader(safe(COMPARATOR_PATHS["e306_neutral_first_neural"]))
        runner.layer.require_equal(self.e306_first_reference, self.neural_references["neutral"], "Entire E306/E305 neutral neural endpoint")
        trace_loader = self._trace_loader or (lambda path: json.loads(path.read_text(encoding="utf8"))["samples"])
        data["e306_candidate_rows"] = trace_loader(safe(COMPARATOR_PATHS["e306_candidate_trace"]))
        trace_arrays(data["baseline_rows"])
        trace_arrays(data["e306_candidate_rows"])
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        if arm != "neutral":
            raise ValueError("E307 permits only one neutral session")
        return super().construct(arm, data, ledger, persist, on_return)

    def first_control(self, session, raw):
        super().first_control(session, raw)
        runner.layer.require_equal(raw["ssp_first_control_neural"], self.e306_first_reference, "Entire actual E306 neutral first neural endpoint")
        raw["ssp_first_control_check"]["e306_neutral_endpoint_exact"] = True

    def capture(self, session, raw):
        super().capture(session, raw)
        resource = raw["activity"][-1]["synaptic_resource"]
        runner.layer.require_equal(resource, np.ones_like(resource), "All E307 neutral observed resources one")

    def compare_neutral(self, raw, data):
        comparison = super().compare_neutral(raw, data)
        comparison.update(scope="first .005-second interval only", e306_neutral_endpoint_exact=True, final_half_second_comparison=False)
        return comparison

    def startup(self, raw, data):
        result = super().startup(raw, data)
        arrays = runner.native.native_arrays(raw["native"])
        upright, nonfoot = arrays["upright"], arrays["nonfoot_contact"]
        healthy = bool(len(upright) == 5000 and np.isfinite(upright).all() and np.min(upright) >= .9 and not np.any(nonfoot))
        result.update(native_health_pass=healthy, minimum_native_upright=float(np.min(upright)) if len(upright) else None,
                      nonfoot_contact_count=int(np.count_nonzero(nonfoot)), strict_E141_command_screen=result["strict_E141_startup_screen"])
        result["strict_E141_startup_screen"] = result["strict_E141_startup_screen"] and healthy
        return result

    def comparisons(self, raw, data):
        summaries = {}
        raw["half_second_trace_deltas"] = {}
        for label, rows in (("neutral_minus_original_E150", data["baseline_rows"]), ("neutral_minus_active_E306", data["e306_candidate_rows"])):
            arrays, summary = compare_traces(raw["rows"], rows, label)
            raw["half_second_trace_deltas"][label] = arrays
            summaries[label] = summary
        return {"complete_half_second": True, "controls": 100, "observation_rows": 101, "window_s": [0., .5], "numeric_comparisons": summaries,
                "original_E150_contrast": "Original RK4 versus new SSP u0; solver and closed-loop feedback may differ",
                "active_E306_contrast": "Same SSP/roster/tau; active u versus u0 includes resulting closed-loop feedback",
                "new_acceptance_threshold": False, "causal_mechanism_claim": False}

    def save_endpoints(self, folder, raw, data):
        arrays, summary = e301.endpoint_factors(raw, data, "neutral")
        require_neutral_factors(arrays)
        np.savez_compressed(folder / "recovery-factor-endpoints.npz", **arrays)
        self.write(folder / "recovery-factor-endpoints.json", {**summary, "neutral_all_factors_one": True})

    def save_raw(self, folder, raw, data, clock):
        failures = super().save_raw(folder, raw, data, clock)
        for label, arrays in raw.get("half_second_trace_deltas", {}).items():
            try:
                np.savez_compressed(folder / (label + ".npz"), **arrays)
            except Exception as error:
                failures.append({"artifact": label + ".npz", "error": repr(error)})
        return failures

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e307.sources.v1", "diagnostic_id": "E307",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_ssp_neutral_body.execute",
                "executed_arm_runner": "tools.verify_c_banc_m3_ssp_neutral_body.run_arm",
                "reused_dependencies": "tools.verify_c_banc_m3_ssp_recovery_body.NativeDependencies",
                "candidate_session_constructed": False}
        return runner.NativeDependencies.write(self, path, value)


def run_arm(data, freeze, out, ledger, deps, budget):
    arm = "neutral"
    folder = out / arm
    folder.mkdir()
    raw = {key: [] for key in ("rows", "controls", "recruitment", "activity", "native")}
    raw["ids"] = data["ids"]
    clock, holder = runner.native.AdvanceLedger(), {}
    session = detach = journal = None
    primary, measured_stop, errors = None, None, []
    entry = {"status": "requested", "completed_controls": 0, "readout_endpoints": 0,
             "construction_requested": False, "construction_completed": False, "continuation_entered": False,
             "owned_session_recovered": False, "close_requested": False, "close_returned": False}
    ledger["arms"][arm] = entry
    metrics = {"startup": {"available": False}, "comparisons": {"complete_half_second": False}, "first_control_comparison": {}}
    def persist():
        deps.write(out / "advance-ledger.json", ledger)
    def fail(stage, error):
        nonlocal primary
        errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E307 " + stage + ": " + repr(error))
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
        for control in range(1, 101):
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
                    error.add_note("Secondary E307 readout journal: " + repr(secondary))
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
            if control == 1:
                metrics["first_control_comparison"] = deps.compare_neutral(raw, data)
                raw.pop("neutral_end_snapshot", None)
            if control == 100:
                metrics["startup"] = deps.startup(raw, data)
                metrics["comparisons"] = deps.comparisons(raw, data)
        entry["status"] = "measured_native_health_stop" if measured_stop is not None else (
            "strict_startup_failed" if entry["completed_controls"] == 100
                and metrics["startup"].get("strict_E141_startup_screen") is not True else "science_complete")
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
        entry["errors"] = errors
        if primary is not None:
            entry["status"] = "technical_or_resource_failure"
            metrics["validity"] = "invalid_or_incomplete"
            metrics["first_error"] = {"type": type(primary).__name__, "message": str(primary)}
        else:
            metrics["validity"] = "valid_ssp_neutral_body_diagnostic"
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
    if freeze.get("diagnostic_id") != "E307":
        raise ValueError("Explicit E307 diagnostic identity required")
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e307.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "validity": "incomplete",
        "publication_status": "pending",
        "candidate_adopted": False,
        "diagnostic_id": freeze.get("diagnostic_id", "E307"),
        "run_valid": False,
        "neutral_prefix_valid": False,
        "ssp_neural_endpoint_exact": False,
        "actual_startup_pass": False,
        "short_gait_pass": False,
        "adopted": False,
        "next_separate_10s_eligible": False,
        "natural_gait_complete": False,
        "F1_F8_complete": False,
        "arm_receipt_path": "tools.verify_c_banc_m3_ssp_neutral_body.run_arm",
        "biological_claim": False,
        "half_second_comparison_complete": False,
        "hypothesis_outcome": "not_adjudicated",
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
            raise TimeoutError("E307 science deadline/reserve reached")
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
                raise MemoryError("E307 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E307 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e307.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = run_arm(data, freeze, out, ledger, deps, check)
        result["arms"]["neutral"] = neutral
        result["neutral_prefix_valid"] = neutral.get("first_control_comparison", {}).get("neutral_prefix_valid") is True
        result["ssp_neural_endpoint_exact"] = neutral.get("first_control_comparison", {}).get("ssp_neural_endpoint_exact") is True
        if primary is None and neutral.get("completed_controls") == 100 and neutral.get("health_stop") is None:
            result["actual_startup_pass"] = neutral.get("startup", {}).get("strict_E141_startup_screen") is True
            result["half_second_comparison_complete"] = neutral.get("comparisons", {}).get("complete_half_second") is True
            if result["half_second_comparison_complete"] and neutral.get("startup", {}).get("native_health_pass") is True:
                result["hypothesis_outcome"] = ("neutral_passes_original_startup" if result["actual_startup_pass"]
                                                else "neutral_also_fails_original_startup")
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_ssp_neutral_body_diagnostic"
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
            half_second_comparison_complete=False,
            hypothesis_outcome="not_adjudicated",
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
        > LIMITS["body_constructors"]
        or result["cost"]["unknown_cost_arms"]
        or result["cost"]["unknown_setup_calls"]
        or any(row.get("unknown_step_entries") or row.get("active_unreturned_frames") for row in ledger["body_construction"])
    ):
        primary = primary or RuntimeError(
            "E307 scientific clocks/caps unavailable or exceeded"
        )
        result.update(
            run_valid=False,
            half_second_comparison_complete=False,
            hypothesis_outcome="not_adjudicated",
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
        manifest = runner.short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e307.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E307 final publication deadline")
        result["publication_status"] = "complete"
        deps.write(out / "result.json", result)
        deps.write(out / "result-sha256.json", {"schema": "flylab.m3.e307.result-sha256.v1", "path": "result.json", "sha256": runner.protocol.sha256(out / "result.json")})
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E307 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(
            run_valid=False,
            half_second_comparison_complete=False,
            hypothesis_outcome="not_adjudicated",
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        result["publication_status"] = "invalid"
        errors.append({"stage": "publication", "type": type(error).__name__, "message": str(error)})
        def secondary(stage, failure):
            errors.append({"stage": stage, "type": type(failure).__name__, "message": str(failure)})
            primary.add_note("Secondary E307 " + stage + ": " + repr(failure))
        def clear_receipt(name):
            try:
                (out / name).unlink(missing_ok=True)
            except Exception as failure:
                secondary("invalidate " + name, failure)
        for name in ("output-manifest.json", "result-sha256.json"):
            clear_receipt(name)
        # Independent attempts make an earlier failed error write unable to
        # suppress the best available invalid result/status publication.
        for name, value in (("errors.json", {"errors": errors}), ("result.json", result),
                            ("publication-status.json", {"schema": "flylab.m3.e307.publication-status.v1", "status": "invalid",
                                "run_valid": False, "first_error": result["first_error"], "success_receipts_trusted": False})):
            try:
                deps.write(out / name, value)
            except Exception as failure:
                secondary("invalid publication " + name, failure)
                if name == "result.json":
                    clear_receipt(name)
        # Include secondary invalidation/write failures whenever still writable.
        try:
            deps.write(out / "errors.json", {"errors": errors})
        except Exception as failure:
            secondary("final errors publication", failure)
    return result, primary
def run(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, output)
    import cupy
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads, previous_env, limiter, cleaned = torch.get_num_threads(), {}, None, False
    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        failures = []
        for callback in ((lambda: limiter.restore_original_limits()) if limiter else lambda: None, lambda: torch.set_num_threads(previous_threads)):
            try:
                callback()
            except Exception as error:
                failures.append(error)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if failures:
            for error in failures[1:]:
                failures[0].add_note("Secondary E307 cleanup: " + repr(error))
            raise failures[0]
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            previous_env[name] = os.environ.get(name)
            os.environ[name] = "1"
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        deps = NativeDependencies()
        deps.gpu_free_before, _ = cupy.cuda.runtime.memGetInfo()
        result, error = execute(freeze, output, deps=deps, cleanup=cleanup)
        return 0 if error is None and result["run_valid"] else 1
    finally:
        cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    return run(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
