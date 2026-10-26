"""E222 one-shot full-BANC/body assay for the frozen E221 native response."""

# Preserve the primary model error while independently saving partial raw data.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import hashlib
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
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.banc_m3_native_grouped_trial import NativeGroupedTrial

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "0fa1dc5bf35aec50f7f2229a6b3b51a1287a3ab9f48fef239d8eef441998b47b"
FREEZE_SCHEMA = "flylab.m3.e222.native-grouped-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e222.native-grouped-body.review.v1"
OWNED = ("tools/banc_m3_native_grouped_trial.py",
         "tools/verify_c_banc_m3_native_grouped_body.py",
         "tests/test_c_banc_native_grouped_trial.py",
         "tests/test_c_banc_native_grouped_body.py")
TENDONS = tuple(leg + "_tarsus" for leg in e113.LEGS)
INTERNAL_S = 175.
RESERVE_S = 20.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(name):
    relative = Path(name)
    value = (ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not value.is_relative_to(ROOT):
        raise ValueError("Repository-relative E222 input path required")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(protocol.jsonable(value),
        ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """No original model entry before complete frozen input and source review."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E222 freeze and parent review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or freeze.get("candidate_adopted") is not False
            or freeze.get("F6_exposed") is not False
            or freeze.get("whole_goal_complete") is not False
            or freeze.get("active_UI_changed") is not False
            or freeze.get("native", {}).get("main_controls") != 2000
            or freeze.get("native", {}).get("neurons") != 158706
            or freeze.get("resource_limits", {}).get("internal_s") != INTERNAL_S
            or freeze.get("resource_limits", {}).get("finalization_reserve_s") != RESERVE_S
            or freeze.get("resource_limits", {}).get("original_attempts") != 1
            or freeze.get("resource_limits", {}).get("retries") != 0
            or freeze.get("resource_limits", {}).get("fit_calls") != 0
            or freeze.get("resource_limits", {}).get("solver_calls") != 0):
        raise ValueError("E222 fixed experiment/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved exact E222 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        for name, expected in freeze[label].items():
            if len(expected) != 64 or sha256(safe(name)) != expected:
                raise ValueError("Frozen E222 file changed: " + name)
    for name, expected in review["source_sha256"].items():
        if len(expected) != 64 or sha256(safe(name)) != expected:
            raise ValueError("Reviewed E222 source changed: " + name)
    for name, expected in review.get("additional_reused_source_sha256", {}).items():
        if len(expected) != 64 or sha256(safe(name)) != expected:
            raise ValueError("Additional reviewed source changed: " + name)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("E222 output must be new under existing parent")
    if output.resolve() != (BASE / "e222-native-grouped-body").resolve():
        raise ValueError("Exact E222 output directory required")
    if shutil.disk_usage(output.parent).free < freeze["resource_limits"]["minimum_free_bytes"]:
        raise OSError("E222 minimum free output disk unavailable")
    return freeze, review


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_selected_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys) <= set(archive.files):
            raise ValueError("Required E222 frozen NPZ arrays missing")
        return {key: archive[key].copy() for key in keys}


def capture_native_state(session):
    """Control-boundary raw from live, already-updated network/adapter only."""
    net = session.network
    row = protocol.capture_control(session)
    values = {"control_tick": int(session.control_tick),
        "network_tick": int(net.tick),
        "physics_time_s": float(session.body.physics_time()),
        "motor_rate32": np.asarray(session.rate, np.float32).copy(),
        "free_rate64": net._free_rate64.detach().cpu().numpy().copy(),
        "h_fast": net._free_fast64.detach().cpu().numpy().copy(),
        "h_slow": net._free_slow64.detach().cpu().numpy().copy(),
        "activation": session.adapter.activation.copy(),
        "command": np.asarray(row["joint_targets_rad"], np.float64).copy(),
        "tendon": np.asarray([row["tendon_inputs"][key] for key in TENDONS],
                             np.float64)}
    shapes = {"motor_rate32": (391,), "free_rate64": (343,),
        "h_fast": (343, 3), "h_slow": (343, 3),
        "activation": (96,), "command": (42,), "tendon": (6,)}
    if (any(values[key].shape != shape or not np.isfinite(values[key]).all()
            for key, shape in shapes.items())
            or values["motor_rate32"].dtype != np.float32
            or any(values[key].dtype != np.float64 for key in shapes
                   if key != "motor_rate32")):
        raise ValueError("Finite complete E222 live control arrays required")
    return row, values


def save_control(path, values):
    keys = ("control_tick", "network_tick", "physics_time_s", "motor_rate32",
            "free_rate64", "h_fast", "h_slow", "activation", "command", "tendon")
    if values:
        np.savez_compressed(path, **{key: np.asarray([r[key] for r in values])
                                     for key in keys})
    else:
        np.savez_compressed(path, control_tick=np.empty(0, np.int64),
            network_tick=np.empty(0, np.int64), physics_time_s=np.empty(0),
            motor_rate32=np.empty((0, 391), np.float32),
            free_rate64=np.empty((0, 343)), h_fast=np.empty((0, 343, 3)),
            h_slow=np.empty((0, 343, 3)), activation=np.empty((0, 96)),
            command=np.empty((0, 42)), tendon=np.empty((0, 6)))


def startup_screen(values, target, limits):
    if len(values) < 100:
        return {"available": False, "reason": "incomplete original99 prefix"}
    q = np.asarray([r["command"] for r in values[1:100]])
    ltm = np.asarray([r["tendon"] for r in values[1:100]])
    target_q, target_ltm = (target[k] for k in
        ("loaded_startup_target_command", "loaded_startup_target_tendon"))
    if target_q.shape != (99, 42) or target_ltm.shape != (99, 6):
        raise ValueError("Original E197 startup target dimensions changed")
    def metrics(actual, expected):
        error = actual - expected
        return {"RMS": float(np.sqrt(np.mean(error**2))),
                "max_abs": float(np.max(np.abs(error)))}
    command, tendon = metrics(q, target_q), metrics(ltm, target_ltm)
    return {"available": True, "command": command, "LTM": tendon,
            "pass": command["RMS"] <= limits["q_RMS"]
                    and command["max_abs"] <= limits["q_max_abs"]
                    and tendon["RMS"] <= limits["LTM_RMS"]
                    and tendon["max_abs"] <= limits["LTM_max_abs"]}


def exact_scientific(left, right, label):
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    a.pop("wall_s", None)
    b.pop("wall_s", None)
    protocol.exact(a, b, label)


def cut_screen(values, neutral, rows=None):
    if len(values) != 20:
        return {"available": False, "pass": False}
    active = np.asarray([v["activation"] for v in values])
    command = np.asarray([v["command"] for v in values])
    tendon = np.asarray([v["tendon"] for v in values])
    cut = bool(np.array_equal(active[:4], np.zeros((4, 96)))
        and np.array_equal(command[:4], np.broadcast_to(neutral, (4, 42)))
        and np.array_equal(tendon[:4], np.zeros((4, 6))))
    restart = bool(np.any(active[4:] > 0)
                   and np.any(np.abs(command[4:] - neutral) > 0))
    abdomen = bool(rows is not None and len(rows) == 20 and all(
        row.get("tendon_inputs", {}).get(key) == .2
        for row in rows for key in ("abdomen_pitch", "abdomen_yaw")))
    return {"available": True, "cut_zero": cut, "restart_positive": restart,
            "abdomen_holds": abdomen, "pass": cut and restart and abdomen}


def baseline_comparison(quality, slip, baseline_directory):
    """Compare descriptive same-window quantities without moving F3–F5 gates."""
    old_quality = json.loads((baseline_directory / "quality-summary.json")
                             .read_text(encoding="utf-8"))
    old_slip = json.loads((baseline_directory / "supported-slip.json")
                          .read_text(encoding="utf-8"))
    names = ("fixed_initial_heading_net_forward_mm", "net_over_path",
             "minimum_native_upright")
    source = quality.get("physical", {})
    new_values = {"fixed_initial_heading_net_forward_mm": None,
                  "net_over_path": None,
                  "minimum_native_upright": source.get("minimum_upright")}
    # The original native-health evaluator, rather than the biological gait
    # analyzer, supplies heading/path and upright values below.
    return {"baseline_quality": old_quality, "baseline_slip": old_slip,
            "candidate_descriptive": new_values, "comparison_fields": list(names),
            "candidate_slip": slip}


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha, Path(review_path),
                          review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e222.native-grouped-body.ledger.v1",
        "status": "gate_passed", "original_attempts": 0,
        "fit_calls": 0, "solver_calls": 0, "automatic_retries": 0,
        "construction": [], "storage_errors": [], "clock_events": [],
        "candidate_adopted": False, "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e222.native-grouped-body.result.v1",
        "validity": "incomplete", "main_complete": False,
        "candidate_adopted": False, "F6_exposed": False,
        "whole_goal_complete": False}
    primary = None
    cut_started = False
    cut_completed = False
    process = None
    gpu_free_before = None
    thread_limits = None
    session = clone = None
    restore_main = restore_clone = restore_cut = None
    journal = None
    rows, controls, clone_rows, clone_controls, cut_rows, cut_controls = (
        [] for _ in range(6))
    native, clone_native, cut_native = [], [], []
    clock = e113.AdvanceLedger()
    last_saved = -1

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        write_json(out / "advance-ledger.json", ledger)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E222 storage " + name + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E222 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(
                ledger.get("peak_resident_bytes", 0), peak)
            if peak > freeze["resource_limits"]["resident_bytes"]:
                raise MemoryError("E222 resident memory cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            current_free, _ = cp.cuda.runtime.memGetInfo()
            delta = max(0, gpu_free_before - current_free)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), delta)
            if delta > freeze["resource_limits"]["allocated_gpu_bytes"]:
                raise MemoryError("E222 additional CUDA allocation cap exceeded")

    def append(owner, index, target_session, row_list, array_list, samples):
        before = (int(target_session.control_tick), int(target_session.network.tick),
                  float(target_session.body.physics_time()), len(samples))
        event = protocol.advance_control(target_session, owner, index, clock, journal)
        try:
            row, values = capture_native_state(target_session)
            row_list.append(row)
            array_list.append(values)
            protocol.record_readout(journal, owner, index, event, complete=True)
        except BaseException as error:
            try:
                protocol.record_readout(journal, owner, index, event,
                                        complete=False, error=error)
            except BaseException as secondary:
                error.add_note("Secondary E222 readout journal: " + repr(secondary))
            raise
        if (target_session.control_tick != before[0] + 1
                or target_session.network.tick != before[1] + 20
                or abs(target_session.body.physics_time() - before[2] - .005) > 2e-8
                or len(samples) != before[3] + 50 or target_session.fault
                or target_session.body.fault):
            raise ValueError("E222 incomplete or faulty native control interval")
        return row

    try:
        write_json(out / "source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threads = one_cpu_thread()
        ledger["threading"] = threads
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        import cupy as cp
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        graph = protocol.GraphStore.load(safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("Exact E150 full BANC graph required")
        start = protocol.StateStore.load(safe(freeze["paths"]["start"]))
        parameters = load_npz(safe(freeze["paths"]["parameters"]))
        prototypes = load_npz(safe(freeze["paths"]["prototypes"]))
        candidate = load_selected_npz(safe(freeze["paths"]["candidate"]), ("beta",))
        target = load_selected_npz(safe(freeze["paths"]["targets"]),
            ("loaded_startup_target_command", "loaded_startup_target_tendon"))
        ledger["original_attempts"] = 1
        ledger["status"] = "construction_requested"
        persist()
        session = protocol.profiled_construction("native_grouped_start",
            lambda: NativeGroupedTrial.from_e150_checkpoint(graph, start,
                parameters, prototypes, candidate["beta"]), ledger["construction"], persist)
        if (session.network.n != 158706 or session.control_tick != 0
                or session.network.tick != 0):
            raise ValueError("E222 neutral full graph clock changed")
        ledger["model_hash"] = session.network.identity
        ledger["adapter_hash"] = session.adapter.identity
        ledger["body_hash"] = session.body.model_hash
        ledger["rss_after_setup_bytes"] = process.memory_info().rss
        ledger["rss_delta_setup_bytes"] = ledger["rss_after_setup_bytes"] - memory_before
        persist()
        protocol.StateStore.save(out / "start", session.snapshot())
        restore_main = e113.install_native_observer(session.body, native)
        journal = protocol.ControlJournal(out / "control-journal.jsonl")
        row0, value0 = capture_native_state(session)
        rows.append(row0)
        controls.append(value0)
        with (out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(protocol.jsonable(row0), allow_nan=False) + "\n")
            last_saved = 0
            for index in range(1, 2001):
                budget()
                row = append("main", index, session, rows, controls, native)
                stream.write(json.dumps(protocol.jsonable(row), allow_nan=False) + "\n")
                last_saved = index
                if index == 200:
                    protocol.StateStore.save(out / "checkpoint-1s", session.snapshot())
                    saved = protocol.StateStore.load(out / "checkpoint-1s")
                    clone = protocol.profiled_construction("native_grouped_clone",
                        lambda saved=saved: NativeGroupedTrial.from_checkpoint(graph, saved),
                        ledger["construction"], persist)
                    restore_clone = e113.install_native_observer(clone.body,
                                                                  clone_native)
                    clone_row0, clone_value0 = capture_native_state(clone)
                    clone_rows.append(clone_row0)
                    clone_controls.append(clone_value0)
                    for clone_index in range(1, 21):
                        budget()
                        append("clone", clone_index, clone, clone_rows,
                               clone_controls, clone_native)
                    protocol.StateStore.save(out / "clone-1p1", clone.snapshot())
                    if len(clone_native) != 1000:
                        raise ValueError("E222 clone native sample count mismatch")
                if index == 220:
                    protocol.StateStore.save(out / "main-1p1", session.snapshot())
                    exact_scientific(protocol.StateStore.load(out / "main-1p1"),
                        protocol.StateStore.load(out / "clone-1p1"),
                        "E222 disk-clone 1.1 second exact state")
                    result["clone_1p1_exact"] = True
                    if clone is not None:
                        restore_clone()
                        restore_clone = None
                        clone.close()
                        clone = None
        main_complete = len(rows) == 2001 and len(native) == 100000
        result["main_complete"] = main_complete
        if not main_complete:
            raise ValueError("E222 original ten-second native body incomplete")
        protocol.StateStore.save(out / "final", session.snapshot())
        write_json(out / "trace.json", {"schema": "flylab.m3.e222.trace.v1",
            "graph_hash": graph.hash, "model_hash": session.network.identity,
            "body_hash": session.body.model_hash, "adapter_hash": session.adapter.identity,
            "parameters": {"experiment": "E222 native grouped response"},
            "samples": rows})
        save_control(out / "main-control.npz", controls)
        np.savez_compressed(out / "native-contact.npz", **e113.native_arrays(native))
        ledger["main_sealed_before_cut"] = True
        persist()
        quality, _, _ = gait.analyze(gait.load_model(out / "trace.json"))
        write_json(out / "quality-summary.json", quality)
        slip = e113.supported_slip(e113.native_arrays(native), complete=True)
        write_json(out / "supported-slip.json", slip)
        health = e113.body_quality(rows, native, quality, reason="duration",
            fault=session.fault or session.body.fault,
            clock_valid=clock.clock_valid, future_exact=result["clone_1p1_exact"])
        write_json(out / "body-quality.json", health)
        if not health["complete_10s_native_health"]:
            raise e113.NativeHealthStop("E222 complete native health gate failed")
        startup = startup_screen(controls, target, freeze["startup_limits"])
        write_json(out / "startup-diagnostic.json", startup)
        comparison = baseline_comparison(quality, slip,
            safe(freeze["paths"]["baseline"]))
        comparison["candidate_descriptive"] = {key: health.get(key)
            for key in comparison["comparison_fields"]}
        write_json(out / "e150-same-window-comparison.json", comparison)
        result.update(validity="valid_full_body_diagnostic",
                      main_health=health, startup=startup,
                      gait_quality=quality, supported_slip=slip,
                      e150_comparison=comparison)
        write_json(out / "main-result.json", copy.deepcopy(result))
        persist()
        cut_started = True
        restore_main()
        restore_main = None
        restore_cut = e113.install_native_observer(session.body, cut_native)
        neutral = np.asarray(session.body.neutral, np.float64).copy()
        for motor_cut, count, label in ((True, 4, "cut"), (False, 16, "restart")):
            session.set_cuts({"motor": motor_cut})
            for _ in range(count):
                budget()
                append(label, len(cut_rows) + 1, session,
                       cut_rows, cut_controls, cut_native)
        result["cut_restart"] = cut_screen(cut_controls, neutral, cut_rows)
        result["cut_restart_status"] = (
            "passed" if result["cut_restart"]["pass"] else "failed_screen")
        write_json(out / "cut-restart-result.json", result["cut_restart"])
        cut_completed = True
    except BaseException as error:
        primary = error
        if (out / "main-result.json").exists():
            result["cut_restart_status"] = (
                "resource_limited" if isinstance(error, (TimeoutError, MemoryError))
                else "failed_execution")
            result["main_valid_despite_cut_failure"] = cut_started
        elif isinstance(error, e113.NativeHealthStop):
            result["validity"] = "measured_native_health_stop"
        elif isinstance(error, (TimeoutError, MemoryError)):
            result["validity"] = "resource_limited"
        else:
            result["validity"] = "technical_failure"
        result.update(primary_error=repr(error), traceback=traceback.format_exc())
    finally:
        for name, callback in (
            ("main-control.npz", lambda: None if (out / "main-control.npz").exists()
             else save_control(out / "main-control.npz", controls)),
            ("clone-control.npz", lambda: save_control(out / "clone-control.npz", clone_controls)),
            ("cut-control.npz", lambda: save_control(out / "cut-control.npz", cut_controls)),
            ("native-contact.npz", lambda: None if
             (out / "native-contact.npz").exists() else np.savez_compressed(
                out / "native-contact.npz", **e113.native_arrays(native))),
            ("clone-native-contact.npz", lambda: np.savez_compressed(
                out / "clone-native-contact.npz", **e113.native_arrays(clone_native))),
            ("cut-native-contact.npz", lambda: np.savez_compressed(
                out / "cut-native-contact.npz", **e113.native_arrays(cut_native))),
            ("cut-trace.json", lambda: write_json(out / "cut-trace.json", cut_rows))):
            attempt(name, callback)
        attempt("clone-trace.json", lambda: write_json(out / "clone-trace.json",
                                                       clone_rows))
        for name, callback in (("main_observer", restore_main),
                               ("clone_observer", restore_clone),
                               ("cut_observer", restore_cut),
                               ("journal", journal.close if journal else None),
                               ("clone", clone.close if clone else None),
                               ("main", session.close if session else None)):
            if callback is not None:
                attempt("cleanup_" + name, callback)
        if thread_limits is not None:
            attempt("cpu_thread_restore", thread_limits.restore_original_limits)
        ledger["clock_events"] = clock.events
        ledger["actual_neural_s"] = clock.neural_s
        ledger["actual_physical_s"] = clock.physical_s
        ledger["clock_valid"] = clock.clock_valid
        ledger["main_rows"] = len(rows)
        ledger["main_native_samples"] = len(native)
        ledger["clone_rows"] = len(clone_rows)
        ledger["clone_native_samples"] = len(clone_native)
        ledger["cut_rows"] = len(cut_rows)
        ledger["cut_native_samples"] = len(cut_native)
        ledger["last_durable_main_trace_index"] = last_saved
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E222 internal wall limit exceeded")
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E222 wall limit: " + repr(error))
        output_bytes = sum(path.stat().st_size for path in out.rglob("*")
                           if path.is_file())
        ledger["output_bytes"] = output_bytes
        if output_bytes > freeze["resource_limits"]["output_bytes"]:
            error = OSError("E222 output byte cap exceeded")
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E222 output cap: " + repr(error))
        if primary is not None:
            result["execution_status"] = "failed"
            result["primary_error"] = repr(primary)
            if (out / "main-result.json").exists():
                result["main_valid_despite_cut_failure"] = cut_started and not cut_completed
                if cut_started and not cut_completed:
                    result.setdefault("cut_restart_status", "failed_execution")
                elif ledger["storage_errors"]:
                    result["post_main_storage_failure"] = True
            elif ledger["storage_errors"] and result["validity"] in (
                    "incomplete", "technical_failure"):
                result["validity"] = "technical_storage_failure"
            elif result["validity"] == "incomplete":
                result["validity"] = "technical_failure"
        else:
            result["execution_status"] = "complete"
        attempt("result.json", lambda: write_json(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: write_json(
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
        "main_complete": result["main_complete"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
