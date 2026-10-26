"""E225: continue the frozen E224 effector body from 3 to 10 seconds."""

# Raw saving and owned-session cleanup must preserve the primary failure.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_c_banc_biological_gait as gait
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_native_grouped_body as grouped
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.banc_m3_neural_current_trial import NeuralCurrentTrial

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "5e48f7fe176bdcd3a81584dff81715a2bec856d395a9abad2916062d4eae298b"
FREEZE_SCHEMA = "flylab.m3.e225.effector-continuation.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e225.effector-continuation.review.v1"
OWNED = ("tools/verify_c_banc_m3_effector_continuation.py",
         "tests/test_c_banc_effector_continuation.py")
OUT = BASE / "e225-effector-continuation"
INTERNAL_S = 85.
RESERVE_S = 10.
CONTROL_KEYS = prior.COMMON_FIELDS + prior.EFFECTOR_FIELDS
NATIVE_KEYS = ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid",
               "nonfoot_contact", "upright")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    name = Path(relative)
    resolved = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not resolved.is_relative_to(ROOT):
        raise ValueError("Repository-relative E225 frozen path required")
    return resolved


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """Verify the complete frozen experiment before any graph/model creation."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E225 freeze and parent review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    native, limits, criteria = (freeze.get(key, {}) for key in
                                ("native", "resource_limits", "criteria"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or native != {"neurons": 158706, "mode": "effector_canonical",
                "start_control": 600, "end_control": 2000,
                "additional_controls": 1400, "prefix_rows": 601,
                "prefix_native_samples": 30000, "additional_native_samples": 70000,
                "combined_rows": 2001, "combined_native_samples": 100000,
                "neural_dt_s": .00025, "physics_dt_s": .0001,
                "control_dt_s": .005, "neural_steps_per_control": 20,
                "physics_steps_per_control": 50,
                "maximum_new_neural_ticks": 28000,
                "maximum_new_physics_steps": 70000,
                "clone_controls": 0, "cut_controls": 0}
            or limits != {"cpu_threads": 1, "gpu_count": 1,
                "original_attempts": 1, "retries": 0, "fit_calls": 0,
                "solver_calls": 0, "internal_s": INTERNAL_S,
                "external_s": 90., "supervisor_s": 95.,
                "finalization_reserve_s": RESERVE_S,
                "resident_bytes": 6442450944,
                "allocated_gpu_bytes": 2147483648,
                "output_bytes": 536870912,
                "minimum_free_bytes": 2147483648}
            or criteria != {"F3_F5_original_10s_evaluator": True,
                "checkpoint_restore_and_seam": "byte exact excluding wall_s only",
                "future_clone_complete": False,
                "whole_goal_complete": False}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E225 fixed continuation/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved exact E225 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("Frozen E225 hash inventory missing: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E225 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E225 hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E225 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E225 frozen package version changed: " + package)
    if output.exists() or not output.parent.is_dir() or output != OUT.resolve():
        raise FileExistsError("New exact E225 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E225 minimum free output disk unavailable")
    return freeze, review


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def read_prefix(control_path, native_path, trace_path):
    """Validate saved E224 prefix structure, without replaying any dynamics."""
    control, native = load_npz(control_path), load_npz(native_path)
    with Path(trace_path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if set(control) != set(CONTROL_KEYS) or set(native) != set(NATIVE_KEYS):
        raise ValueError("Exact E224 effector raw key sets required")
    control_shapes = {"control_tick": (), "network_tick": (),
        "physics_time_s": (), "motor_rate32": (391,),
        "decoder_input32": (391,), "activation": (96,),
        "command": (42,), "tendon": (6,), "free_effector64": (343,),
        "h_fast": (343, 3), "h_slow": (343, 3)}
    native_shapes = {"time_s": (), "normal_load_bw": (6,),
        "slip_mm_s": (6,), "slip_valid": (6,),
        "nonfoot_contact": (), "upright": ()}
    if (len(rows) != 601 or any(control[k].shape != (601, *shape)
            for k, shape in control_shapes.items())
            or any(native[k].shape != (30000, *shape)
                   for k, shape in native_shapes.items())
            or control["motor_rate32"].dtype != np.float32
            or control["decoder_input32"].dtype != np.float32
            or any(control[k].dtype != np.float64 for k in CONTROL_KEYS
                   if k not in ("control_tick", "network_tick", "motor_rate32",
                                "decoder_input32"))
            or any(control[k].dtype != np.int64 for k in
                   ("control_tick", "network_tick"))
            or any(native[k].dtype != np.float64 for k in
                   ("time_s", "normal_load_bw", "slip_mm_s", "upright"))
            or any(native[k].dtype != np.bool_ for k in
                   ("slip_valid", "nonfoot_contact"))):
        raise ValueError("E224 prefix raw shape/dtype changed")
    if (not np.array_equal(control["control_tick"], np.arange(601))
            or not np.array_equal(control["network_tick"], np.arange(601) * 20)
            or not np.allclose(control["physics_time_s"], np.arange(601) * .005,
                               atol=2e-8, rtol=0)
            or not np.allclose(native["time_s"],
                               np.arange(1, 30001) * .0001, atol=2e-8, rtol=0)
            or not np.isfinite(control["physics_time_s"]).all()
            or any(not np.isfinite(control[k]).all() for k in CONTROL_KEYS)
            or not np.isfinite(native["time_s"]).all()
            or not np.isfinite(native["normal_load_bw"]).all()
            or not np.isfinite(native["upright"]).all()
            or not np.isfinite(native["slip_mm_s"][native["slip_valid"]]).all()
            or np.any(native["nonfoot_contact"])):
        raise ValueError("E224 prefix clock or native health changed")
    if any(row.get("control_tick") != i or not np.isclose(row.get("time_s"),
           i * .005, atol=2e-8, rtol=0) or not np.array_equal(
               np.asarray(row["joint_targets_rad"], np.float64),
               control["command"][i]) or not np.array_equal(
                   np.asarray(row["motor_rates"], np.float32),
                   control["motor_rate32"][i]) or not np.array_equal(
                       np.asarray([row["tendon_inputs"][key]
                                   for key in prior.TENDONS], np.float64),
                       control["tendon"][i]) for i, row in enumerate(rows)):
        raise ValueError("E224 prefix trace control clock changed")
    return control, native, rows


def check_seam(saved, restored, control, rows, session):
    """The restored disk state and live row600 must equal the saved E224 seam."""
    protocol.exact(saved, restored, "E225 complete disk restore except wall_s")
    row, value = prior.capture_state(session, "effector_canonical")
    protocol.exact(rows[-1], protocol.jsonable(row), "E225 trace seam")
    for key in CONTROL_KEYS:
        old = control[key][-1]
        new = value[key]
        if isinstance(new, np.ndarray):
            if (old.dtype != new.dtype or old.shape != new.shape
                    or old.tobytes() != new.tobytes()):
                raise ValueError("E225 control seam bytes changed: " + key)
        elif (np.asarray(old).dtype != np.asarray(new).dtype
              or np.asarray(old).tobytes() != np.asarray(new).tobytes()):
            raise ValueError("E225 control seam changed: " + key)
    if (session.control_tick != 600 or session.network.tick != 12000
            or abs(session.body.physics_time() - 3.) > 2e-8):
        raise ValueError("E225 restored three-second clocks changed")
    return row, value


def native_as_list(data):
    return [tuple(data[key][index] for key in NATIVE_KEYS)
            for index in range(len(data["time_s"]))]


def combine_raw(prefix_control, prefix_native, prefix_rows, suffix_control,
                suffix_native, suffix_rows):
    """Join exact saved prefix and new suffix, discarding only duplicate row600."""
    if (len(suffix_rows) != 1401 or len(suffix_native) != 70000
            or len(prefix_rows) != 601 or set(prefix_control) != set(CONTROL_KEYS)):
        raise ValueError("E225 complete prefix/suffix required before combination")
    protocol.exact(prefix_rows[-1], suffix_rows[0], "E225 duplicate trace seam")
    control = {key: np.concatenate((prefix_control[key],
        np.asarray([row[key] for row in suffix_control[1:]])), axis=0)
        for key in CONTROL_KEYS}
    native_suffix = e113.native_arrays(suffix_native)
    native = {key: np.concatenate((prefix_native[key], native_suffix[key]), axis=0)
              for key in NATIVE_KEYS}
    rows = prefix_rows + suffix_rows[1:]
    if (len(rows) != 2001 or any(value.shape[0] != 2001
            for value in control.values()) or any(value.shape[0] != 100000
                                         for value in native.values())
            or any(control[key].dtype != prefix_control[key].dtype
                   or control[key].shape[1:] != prefix_control[key].shape[1:]
                   for key in CONTROL_KEYS)
            or any(native[key].dtype != prefix_native[key].dtype
                   or native[key].shape[1:] != prefix_native[key].shape[1:]
                   for key in NATIVE_KEYS)
            or not np.array_equal(control["control_tick"], np.arange(2001))
            or not np.array_equal(control["network_tick"], np.arange(2001) * 20)
            or not np.allclose(control["physics_time_s"],
                               np.arange(2001) * .005, atol=2e-8, rtol=0)
            or not np.allclose(native["time_s"],
                               np.arange(1, 100001) * .0001, atol=2e-8, rtol=0)
            or any(row.get("control_tick") != i for i, row in enumerate(rows))
            or any(not np.isclose(row.get("time_s"), i * .005,
                                   atol=2e-8, rtol=0)
                   for i, row in enumerate(rows))
            or any(not np.isfinite(control[key]).all() for key in CONTROL_KEYS)
            or not np.isfinite(native["normal_load_bw"]).all()
            or not np.isfinite(native["slip_mm_s"][native["slip_valid"]]).all()
            or not np.isfinite(native["upright"]).all()):
        raise ValueError("E225 combined 10-second data has a gap, duplicate or fault")
    return control, native, rows


def save_npz(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e225.effector-continuation.ledger.v1",
        "status": "gate_passed", "original_attempts": 0, "retries": 0,
        "fit_calls": 0, "solver_calls": 0, "construction": [],
        "clock_events": [], "storage_errors": [], "clone": "not_executed",
        "motor_cut": "not_executed", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e225.effector-continuation.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "restore_exact": False, "seam_exact": False, "full_10s": False,
        "whole_disk_future_exact": False, "clone": "not_executed",
        "motor_cut": "not_executed", "F2_complete": False,
        "candidate_adopted": False, "F6_exposed": False,
        "whole_goal_complete": False}
    clock = e113.AdvanceLedger()
    session = journal = restore_observer = thread_limits = process = None
    gpu_free_before = None
    suffix_rows, suffix_controls, suffix_native = [], [], []
    primary = measured = None
    trace_durable = -1

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E225 " + name + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E225 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > freeze["resource_limits"]["resident_bytes"]:
                raise MemoryError("E225 resident memory cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            current_free, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before - current_free)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), added)
            if added > freeze["resource_limits"]["allocated_gpu_bytes"]:
                raise MemoryError("E225 additional CUDA allocation cap exceeded")

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        prefix_control, prefix_native, prefix_rows = read_prefix(*(
            safe(freeze["paths"][key]) for key in
            ("prefix_control", "prefix_native", "prefix_trace")))
        target = load_npz(safe(freeze["paths"]["targets"]))
        startup = grouped.startup_screen([
            {"command": prefix_control["command"][i],
             "tendon": prefix_control["tendon"][i]} for i in range(100)],
            target, freeze["startup_limits"])
        prior.write_receipt(out / "startup-diagnostic.json", startup)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threading = one_cpu_thread()
        ledger["threading"] = threading
        import psutil
        process = psutil.Process()
        import cupy as cp
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        budget()
        graph = protocol.GraphStore.load(safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("E225 exact original full graph required")
        saved = protocol.StateStore.load(safe(freeze["paths"]["checkpoint"]))
        ledger["original_attempts"] = 1
        ledger["status"] = "restore_requested"
        persist()
        session = protocol.profiled_construction("effector_continuation_restore",
            lambda: NeuralCurrentTrial.from_checkpoint(graph, saved),
            ledger["construction"], persist)
        if session.mode != "effector_canonical" or session.network.n != 158706:
            raise ValueError("E225 exact E224 effector checkpoint required")
        protocol.exact(saved, session.snapshot(), "E225 exact restored full checkpoint")
        result["restore_exact"] = True
        protocol.StateStore.save(out / "restored", session.snapshot())
        protocol.exact(saved, protocol.StateStore.load(out / "restored"),
                       "E225 disk-serialized restore exact")
        seam_row, seam_control = check_seam(saved, session.snapshot(),
            prefix_control, prefix_rows, session)
        result["seam_exact"] = True
        prior.write_receipt(out / "seam.json", {"restore_exact": True,
            "seam_exact": True, "control_tick": 600, "network_tick": 12000,
            "physics_time_s": 3., "source_control_sha256": sha256(
                safe(freeze["paths"]["prefix_control"])),
            "source_native_sha256": sha256(safe(freeze["paths"]["prefix_native"]))})
        suffix_rows.append(seam_row)
        suffix_controls.append(seam_control)
        restore_observer = e113.install_native_observer(session.body, suffix_native)
        journal = prior.FiniteControlJournal(out / "control-journal.jsonl")
        with (out / "continuation-trace.jsonl").open("x", encoding="utf-8",
                                                     buffering=1) as stream:
            stream.write(json.dumps(protocol.jsonable(seam_row),
                                    allow_nan=False) + "\n")
            trace_durable = 600
            for index in range(601, 2001):
                budget()
                before = (int(session.control_tick), int(session.network.tick),
                          float(session.body.physics_time()), len(suffix_native))
                event = protocol.advance_control(session, "continuation", index,
                                                 clock, journal)
                try:
                    row, value = prior.capture_state(session, "effector_canonical")
                    suffix_rows.append(row)
                    suffix_controls.append(value)
                    protocol.record_readout(journal, "continuation", index, event,
                                            complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, "continuation", index, event,
                                                complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E225 readout journal: "
                                       + repr(secondary))
                    raise
                if session.fault or session.body.fault:
                    raise e113.NativeHealthStop("E225 measured body fault")
                if (session.control_tick != before[0] + 1
                        or session.network.tick != before[1] + 20
                        or abs(session.body.physics_time() - before[2] - .005) > 2e-8
                        or len(suffix_native) != before[3] + 50):
                    raise ValueError("E225 incomplete native control interval")
                stream.write(json.dumps(protocol.jsonable(row),
                                        allow_nan=False) + "\n")
                trace_durable = index
        prior.save_control(out / "continuation-control.npz", suffix_controls,
                           "effector_canonical")
        prior.save_native_contact(out / "continuation-native-contact.npz",
                                  suffix_native)
        control, native, rows = combine_raw(prefix_control, prefix_native,
            prefix_rows, suffix_controls, suffix_native, suffix_rows)
        save_npz(out / "combined-control.npz", control)
        save_npz(out / "combined-native-contact.npz", native)
        if session.fault or session.body.fault or not clock.clock_valid:
            raise e113.NativeHealthStop("E225 complete native health failed")
        protocol.StateStore.save(out / "final", session.snapshot())
        trace = {"schema": "flylab.m3.e225.effector-continuation.trace.v1",
            "graph_hash": graph.hash, "model_hash": session.network.identity,
            "body_hash": session.body.model_hash,
            "adapter_hash": session.adapter.identity,
            "parameters": {"experiment": "E225 frozen effector continuation"},
            "samples": rows}
        prior.write_receipt(out / "combined-trace.json", trace)
        quality, _, _ = gait.analyze(gait.load_model(out / "combined-trace.json"))
        prior.write_receipt(out / "quality-summary.json", quality)
        slip = e113.supported_slip(native, complete=True)
        prior.write_receipt(out / "supported-slip.json", slip)
        health = e113.body_quality(rows, native_as_list(native), quality,
            reason="duration", fault=session.fault or session.body.fault,
            clock_valid=clock.clock_valid, future_exact=False)
        prior.write_receipt(out / "body-quality.json", health)
        if not health["complete_10s_native_health"]:
            raise e113.NativeHealthStop("E225 complete 10 s native health failed")
        comparison = grouped.baseline_comparison(quality, slip,
            safe(freeze["paths"]["baseline"]))
        comparison["candidate_descriptive"] = {key: health.get(key)
            for key in comparison["comparison_fields"]}
        prior.write_receipt(out / "e150-same-window-comparison.json", comparison)
        result.update(validity="valid_full_body_diagnostic", full_10s=True,
            quality=quality, body_quality=health, supported_slip=slip,
            startup=startup, e150_comparison=comparison,
            development_screens=health["development_screens"])
    except e113.NativeHealthStop as error:
        measured = error
        result.update(validity="measured_native_health_stop", measured_error=repr(error))
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        attempt("continuation-control.npz", lambda: None if
            (out / "continuation-control.npz").exists() else prior.save_control(
                out / "continuation-control.npz", suffix_controls,
                "effector_canonical"))
        attempt("continuation-native-contact.npz", lambda: None if
            (out / "continuation-native-contact.npz").exists()
            else prior.save_native_contact(out / "continuation-native-contact.npz",
                                           suffix_native))
        for label, callback in (("native_observer", restore_observer),
                                ("journal", journal.close if journal else None),
                                ("session", session.close if session else None),
                                ("cpu_thread_restore", thread_limits.restore_original_limits
                                 if thread_limits else None)):
            if callback is not None:
                attempt("cleanup_" + label, callback)
        ledger["clock_events"] = clock.events
        ledger["completed_continuation_controls"] = max(0, len(suffix_controls) - 1)
        ledger["continuation_native_samples"] = len(suffix_native)
        ledger["last_durable_trace_control"] = trace_durable
        ledger["journal"] = journal.summary() if journal else None
        ledger["actual_neural_s"] = clock.neural_s
        ledger["actual_physical_s"] = clock.physical_s
        ledger["clock_valid"] = clock.clock_valid
        ledger["unknown_neural_intervals"] = sum(
            event["actual_neural_s"] is None for event in clock.events)
        ledger["unknown_physical_intervals"] = sum(
            event["actual_physical_s"] is None for event in clock.events)
        ledger["confirmed_neural_s"] = sum(event["actual_neural_s"]
            for event in clock.events if event["actual_neural_s"] is not None)
        ledger["confirmed_physical_s"] = sum(event["actual_physical_s"]
            for event in clock.events if event["actual_physical_s"] is not None)
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E225 internal wall limit exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E225 wall limit: " + repr(error))
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                     if path.is_file())
        if ledger["output_bytes"] > freeze["resource_limits"]["output_bytes"]:
            error = OSError("E225 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E225 output cap: " + repr(error))
        if primary is not None and measured is not None:
            result["measured_error"] = repr(measured)
            result["post_health_technical_failure"] = True
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.freeze, args.freeze_sha256, args.review,
                     args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"],
                      "full_10s": result["full_10s"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
