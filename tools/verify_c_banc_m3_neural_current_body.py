"""E224 bounded original two-arm neural-current effector body diagnostic."""

# Preserve primary scientific/runtime failures across independent raw cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.banc_m3_neural_current_trial import (
    MODES,
    NeuralCurrentTrial,
    unwrap_neural_current_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "96360b7d09678c0355983dbae2aacaea0acaa28a664f7c37018542c57346c53b"
FREEZE_SCHEMA = "flylab.m3.e224.neural-current-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e224.neural-current-body.review.v1"
OWNED = ("tools/banc_m3_neural_current_trial.py",
         "tools/verify_c_banc_m3_neural_current_body.py",
         "tests/test_c_banc_neural_current_trial.py",
         "tests/test_c_banc_neural_current_body.py")
INTERNAL_S = 145.
RESERVE_S = 20.
TENDONS = tuple(leg + "_tarsus" for leg in e113.LEGS)
COMMON_FIELDS = ("control_tick", "network_tick", "physics_time_s",
                 "motor_rate32", "decoder_input32", "activation", "command", "tendon")
EFFECTOR_FIELDS = ("free_effector64", "h_fast", "h_slow")


def finite_receipt(value):
    """Represent unavailable nonfinite JSON scalars as null, with locations."""
    unavailable = []

    def visit(item, path):
        if isinstance(item, dict):
            return {str(key): visit(member, path + "." + str(key))
                    for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(member, path + "[" + str(index) + "]")
                    for index, member in enumerate(item)]
        if isinstance(item, np.ndarray):
            return visit(item.tolist(), path)
        if isinstance(item, np.generic):
            return visit(item.item(), path)
        if isinstance(item, float) and not math.isfinite(item):
            unavailable.append(path)
            return None
        if isinstance(item, Path):
            return str(item)
        return item

    cleaned = visit(value, "$")
    return cleaned, unavailable


def write_receipt(path, value):
    cleaned, unavailable = finite_receipt(value)
    if unavailable and isinstance(cleaned, dict):
        cleaned["unavailable_nonfinite_fields"] = unavailable
    protocol.write_json(path, cleaned)


class FiniteControlJournal(protocol.ControlJournal):
    def append(self, value):
        cleaned, unavailable = finite_receipt(value)
        if unavailable:
            cleaned["unavailable_nonfinite_fields"] = unavailable
        super().append(cleaned)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    path = Path(relative)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(ROOT):
        raise ValueError("Repository-relative E224 frozen path required")
    return resolved


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """Verify all frozen bytes/packages/source review before model construction."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E224 freeze and approved review SHAs required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    native = freeze.get("native", {})
    limits = freeze.get("resource_limits", {})
    criteria = freeze.get("criteria", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or freeze.get("candidate_adopted") is not False
            or freeze.get("F6_exposed") is not False
            or freeze.get("F2_complete") is not False
            or freeze.get("whole_goal_complete") is not False
            or freeze.get("active_UI_changed") is not False
            or native != {"neurons": 158706, "motors": 391, "free": 343,
                "fixed": 48, "muscles": 96, "basis": 64,
                "arms": list(MODES), "controls_per_arm": 600,
                "neural_steps_per_control": 20, "physics_steps_per_control": 50,
                "neural_dt_s": .00025, "physics_dt_s": .0001,
                "control_dt_s": .005, "maximum_neural_ticks": 24000,
                "maximum_physics_steps": 60000,
                "clone_controls": 0, "cut_controls": 0}
            or limits.get("internal_s") != INTERNAL_S
            or limits.get("finalization_reserve_s") != RESERVE_S
            or limits.get("original_attempts") != 1
            or limits.get("arms") != 2 or limits.get("retries") != 0
            or limits.get("fit_calls") != 0 or limits.get("solver_calls") != 0
            or limits.get("cpu_threads") != 1 or limits.get("gpu_count") != 1
            or criteria != {"upright_min": .9, "nonfoot_contact": 0,
                "numeric_fault": False, "complete_3s_required_for_next_phase": True,
                "F3_F5_10s_replaced": False, "whole_goal_complete": False}):
        raise ValueError("E224 two-arm scientific/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved exact E224 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("Frozen E224 hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E224 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E224 source hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E224 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E224 frozen package version changed: " + package)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("New E224 output directory under existing parent required")
    if output.resolve() != (BASE / "e224-neural-current-body").resolve():
        raise ValueError("Exact E224 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E224 minimum free disk unavailable")
    return freeze, review


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_candidate_beta(path):
    with np.load(path, allow_pickle=False) as archive:
        if "beta" not in archive.files:
            raise ValueError("Frozen E218 selected beta missing")
        return archive["beta"].copy()


def capture_state(session, mode):
    """Return original neuronal readout and actual decoder input separately."""
    row = protocol.capture_control(session)
    original = np.asarray(session.rate)
    decoder = (original if mode == "legacy_canonical"
               else session.adapter.last_decoder_input32)
    value = {"control_tick": int(session.control_tick),
             "network_tick": int(session.network.tick),
             "physics_time_s": float(session.body.physics_time()),
             "motor_rate32": original.copy(),
             "decoder_input32": np.asarray(decoder).copy(),
             "activation": session.adapter.activation.copy(),
             "command": np.asarray(row["joint_targets_rad"], np.float64).copy(),
             "tendon": np.asarray([row["tendon_inputs"][key] for key in TENDONS],
                                  np.float64)}
    if mode == "effector_canonical":
        net = session.network
        value.update(free_effector64=net._free_rate64.detach().cpu().numpy().copy(),
                     h_fast=net._free_fast64.detach().cpu().numpy().copy(),
                     h_slow=net._free_slow64.detach().cpu().numpy().copy())
    shapes = {"motor_rate32": (391,), "decoder_input32": (391,),
              "activation": (96,), "command": (42,), "tendon": (6,)}
    if mode == "effector_canonical":
        shapes.update(free_effector64=(343,), h_fast=(343, 3), h_slow=(343, 3))
    if (any(value[key].shape != shape or not np.isfinite(value[key]).all()
            for key, shape in shapes.items())
            or value["motor_rate32"].dtype != np.float32
            or value["decoder_input32"].dtype != np.float32
            or any(value[key].dtype != np.float64 for key in shapes
                   if key not in ("motor_rate32", "decoder_input32"))):
        raise ValueError("Finite, mode-specific E224 live control raw required")
    if (mode == "legacy_canonical"
            and not np.array_equal(value["motor_rate32"],
                                   value["decoder_input32"])):
        raise ValueError("Legacy decoder must consume original motor rate")
    return row, value


def save_control(path, records, mode):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    keys = COMMON_FIELDS + (EFFECTOR_FIELDS if mode == "effector_canonical" else ())
    if records:
        np.savez_compressed(temporary, **{
            key: np.asarray([row[key] for row in records]) for key in keys})
    else:
        arrays = {"control_tick": np.empty(0, np.int64),
                  "network_tick": np.empty(0, np.int64),
                  "physics_time_s": np.empty(0, np.float64),
                  "motor_rate32": np.empty((0, 391), np.float32),
                  "decoder_input32": np.empty((0, 391), np.float32),
                  "activation": np.empty((0, 96), np.float64),
                  "command": np.empty((0, 42), np.float64),
                  "tendon": np.empty((0, 6), np.float64)}
        if mode == "effector_canonical":
            arrays.update(free_effector64=np.empty((0, 343), np.float64),
                          h_fast=np.empty((0, 343, 3), np.float64),
                          h_slow=np.empty((0, 343, 3), np.float64))
        np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_native_contact(path, samples):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **e113.native_arrays(samples))
    temporary.replace(path)


def three_second_health(rows, native, clock, session):
    """3 s diagnostic only; never call or relabel frozen 10 s F3–F5."""
    data = e113.native_arrays(native)
    completed = (len(rows) == 601 and len(native) == 30000
                 and session.control_tick == 600 and session.network.tick == 12000
                 and clock.clock_valid
                 and not session.fault and not session.body.fault)
    row_time = np.asarray([row["time_s"] for row in rows])
    native_time = data["time_s"]
    timing = bool(completed and np.allclose(row_time, np.arange(601) * .005,
        atol=2e-8, rtol=0) and np.allclose(native_time,
        np.arange(1, 30001) * .0001, atol=2e-8, rtol=0))
    finite = bool(np.isfinite(data["normal_load_bw"]).all()
                  and np.isfinite(data["upright"]).all()
                  and all(not row.get("fault") for row in rows))
    min_upright = float(np.min(data["upright"])) if len(native) else None
    nonfoot = int(np.count_nonzero(data["nonfoot_contact"]))
    healthy = bool(timing and finite and nonfoot == 0
                   and min_upright is not None and min_upright >= .9)
    position = np.asarray([row["body"]["position"] for row in rows], np.float64)
    movement = (float(np.linalg.norm(position[-1] - position[0]))
                if len(position) > 1 and np.isfinite(position).all() else None)
    return {"complete_3s": bool(completed), "native_timing_valid": timing,
            "finite": finite, "minimum_native_upright": min_upright,
            "nonfoot_contact_count": nonfoot,
            "first_nonfoot_time_s": (float(native_time[np.flatnonzero(
                data["nonfoot_contact"])[0]]) if nonfoot else None),
            "movement_mm": movement, "healthy_3s": healthy,
            "F3_F5_10s_evaluated": False}


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
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e224.neural-current-body.ledger.v1",
        "status": "gate_passed", "original_processes": 1,
        "arm_order": list(MODES), "arms_requested": 0, "arms_constructed": 0,
        "automatic_retries": 0, "fit_calls": 0, "solver_calls": 0,
        "construction": [], "storage_errors": [], "clock_events": [],
        "clone": "not_executed", "motor_cut": "not_executed",
        "candidate_adopted": False, "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e224.neural-current-body.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "arm_order": list(MODES), "arms": {},
        "clone": "not_executed", "motor_cut": "not_executed",
        "F3_F5_10s_evaluated": False, "candidate_adopted": False,
        "F2_complete": False, "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    thread_limits = process = None
    gpu_free_before = None
    source_start = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        write_receipt(out / "advance-ledger.json", ledger)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name,
                                             "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E224 " + name + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E224 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E224 resident memory cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            now, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before - now)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E224 additional CUDA memory cap exceeded")

    def execute_arm(mode, graph, start, parameters, prototypes, beta):
        nonlocal source_start
        arm = out / mode
        arm.mkdir()
        clock = e113.AdvanceLedger()
        rows, controls, native = [], [], []
        session = journal = restore_observer = None
        arm_primary = None
        measured = None
        durable_trace = -1
        arm_result = {"schema": "flylab.m3.e224.arm-result.v1",
            "mode": mode, "validity": "incomplete", "complete_3s": False,
            "clone": "not_executed", "motor_cut": "not_executed",
            "F3_F5_10s_evaluated": False}

        def arm_attempt(name, callback):
            nonlocal arm_primary
            try:
                callback()
            except BaseException as error:
                ledger["storage_errors"].append({"artifact": mode + "/" + name,
                                                  "error": repr(error)})
                if arm_primary is None:
                    arm_primary = error
                else:
                    arm_primary.add_note("Secondary E224 arm storage: " + repr(error))

        try:
            budget()
            ledger["arms_requested"] += 1
            ledger["status"] = mode + ":construction_requested"
            persist()
            session = protocol.profiled_construction(mode,
                lambda: NeuralCurrentTrial.from_e150_checkpoint(graph, start,
                    parameters, prototypes, beta, mode),
                ledger["construction"], persist)
            ledger["arms_constructed"] += 1
            if (session.network.n != 158706 or session.control_tick != 0
                    or session.network.tick != 0 or session.rate.shape != (391,)
                    or np.any(session.rate != 0)):
                raise ValueError("E224 exact full-network neutral start changed")
            start_snapshot = session.snapshot()
            neutral, _, _ = unwrap_neural_current_checkpoint(start_snapshot)
            if source_start is None:
                source_start = neutral
            else:
                protocol.exact(source_start, neutral,
                               "E224 identical original body/neuron/action/clock start")
            protocol.StateStore.save(arm / "start", start_snapshot)
            restore_observer = e113.install_native_observer(session.body, native)
            journal = FiniteControlJournal(arm / "control-journal.jsonl")
            row0, value0 = capture_state(session, mode)
            rows.append(row0)
            controls.append(value0)
            with (arm / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(finite_receipt(protocol.jsonable(row0))[0],
                                        allow_nan=False) + "\n")
                durable_trace = 0
                for index in range(1, 601):
                    budget()
                    before = (int(session.control_tick), int(session.network.tick),
                              float(session.body.physics_time()), len(native))
                    event = protocol.advance_control(session, mode, index,
                                                     clock, journal)
                    try:
                        row, value = capture_state(session, mode)
                        rows.append(row)
                        controls.append(value)
                        protocol.record_readout(journal, mode, index, event,
                                                complete=True)
                    except BaseException as error:
                        try:
                            protocol.record_readout(journal, mode, index, event,
                                                    complete=False, error=error)
                        except BaseException as secondary:
                            error.add_note("Secondary E224 readout journal: "
                                           + repr(secondary))
                        raise
                    if session.fault or session.body.fault:
                        raise e113.NativeHealthStop("E224 measured body fault")
                    if (session.control_tick != before[0] + 1
                            or session.network.tick != before[1] + 20
                            or abs(session.body.physics_time() - before[2]
                                   - .005) > 2e-8
                            or len(native) != before[3] + 50):
                        raise ValueError("E224 incomplete control/native clock interval")
                    stream.write(json.dumps(finite_receipt(protocol.jsonable(row))[0],
                                            allow_nan=False) + "\n")
                    durable_trace = index
                    if index == 1:
                        protocol.StateStore.save(arm / "checkpoint-5ms",
                                                 session.snapshot())
            save_control(arm / "control-raw.npz", controls, mode)
            save_native_contact(arm / "native-contact.npz", native)
            health = three_second_health(rows, native, clock, session)
            arm_result["health"] = health
            arm_result["complete_3s"] = health["complete_3s"]
            if not health["healthy_3s"]:
                raise e113.NativeHealthStop("E224 measured 3 s health screen failed")
            protocol.StateStore.save(arm / "final", session.snapshot())
            arm_result["validity"] = "healthy_3s_diagnostic"
        except e113.NativeHealthStop as error:
            measured = error
            nonfinite_native = sorted({field for row in native for field, value in (
                ("time_s", row[0]), ("normal_load_bw", row[1]),
                ("upright", row[5])) if not np.isfinite(value).all()})
            nonfinite_clock = sorted({field for event in clock.events
                for field in ("physics_time_before_s", "physics_time_after_s",
                              "actual_physical_s")
                if event[field] is not None and not np.isfinite(event[field])})
            body_numeric = False
            if session is not None and hasattr(session.body, "d"):
                body_numeric = any(not np.isfinite(getattr(session.body.d, key)).all()
                                   for key in ("qpos", "qvel"))
            arm_result.update(validity="measured_native_health_stop",
                              measured_error=repr(error),
                              last_native_time_s=(native[-1][0] if native else None),
                              minimum_native_upright=(float(np.min(
                                  [v[5] for v in native])) if native and
                                  "upright" not in nonfinite_native else float("nan")
                                  if native else None),
                              nonfoot_contact_count=sum(int(v[4]) for v in native),
                              numeric_fault=bool(nonfinite_native or nonfinite_clock
                                                 or body_numeric),
                              nonfinite_native_fields=nonfinite_native,
                              nonfinite_clock_fields=nonfinite_clock,
                              nonfinite_body_state=body_numeric)
        except (TimeoutError, MemoryError) as error:
            arm_primary = error
            arm_result.update(validity="resource_limited", primary_error=repr(error),
                              traceback=traceback.format_exc())
        except BaseException as error:
            arm_primary = error
            arm_result.update(validity="technical_failure", primary_error=repr(error),
                              traceback=traceback.format_exc())
        finally:
            classified_primary = arm_primary is not None
            arm_attempt("control-raw.npz", lambda: None if
                (arm / "control-raw.npz").exists() else save_control(
                    arm / "control-raw.npz", controls, mode))
            arm_attempt("native-contact.npz", lambda: None if
                (arm / "native-contact.npz").exists() else save_native_contact(
                    arm / "native-contact.npz", native))
            for label, callback in (("native_observer", restore_observer),
                                    ("journal", journal.close if journal else None),
                                    ("session", session.close if session else None)):
                if callback is not None:
                    arm_attempt("cleanup_" + label, callback)
            arm_ledger = {"schema": "flylab.m3.e224.arm-ledger.v1",
                "mode": mode, "clock_events": clock.events,
                "actual_neural_s": clock.neural_s,
                "actual_physical_s": clock.physical_s,
                "clock_valid": clock.clock_valid,
                "completed_controls": max(0, len(controls) - 1),
                "native_samples": len(native),
                "last_durable_trace_index": durable_trace,
                "journal": journal.summary() if journal else None,
                "measured_health_stop": measured is not None,
                "technical_error": repr(arm_primary) if arm_primary else None}
            arm_attempt("advance-ledger.json", lambda: write_receipt(
                arm / "advance-ledger.json", arm_ledger))
            ledger["clock_events"].extend(clock.events)
            ledger["status"] = mode + ":finished"
            arm_attempt("global-ledger.json", persist)
            if arm_primary is not None:
                arm_result["execution_status"] = "failed"
                arm_result["primary_error"] = repr(arm_primary)
                if not classified_primary:
                    arm_result["pre_storage_validity"] = arm_result["validity"]
                    arm_result["validity"] = "technical_storage_failure"
                if measured is not None:
                    arm_result["measured_error"] = repr(measured)
                    arm_result["post_health_cleanup_failure"] = True
            else:
                arm_result["execution_status"] = "complete"
            arm_result["control_rows"] = len(controls)
            arm_result["native_samples"] = len(native)
            arm_result, unavailable = finite_receipt(arm_result)
            if unavailable:
                arm_result["unavailable_nonfinite_fields"] = unavailable
            arm_attempt("result.json", lambda: write_receipt(
                arm / "result.json", arm_result))
            if arm_primary is not None and arm_result["execution_status"] == "complete":
                arm_result["pre_storage_validity"] = arm_result["validity"]
                arm_result.update(validity="technical_storage_failure",
                                  execution_status="failed",
                                  primary_error=repr(arm_primary))
            result["arms"][mode] = arm_result
        if arm_primary is not None:
            raise arm_primary
        return arm_result

    try:
        write_receipt(out / "source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threading = one_cpu_thread()
        ledger["threading"] = threading
        import psutil
        process = psutil.Process()
        import cupy as cp
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        graph = protocol.GraphStore.load(safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("E224 exact original full BANC graph required")
        start = protocol.StateStore.load(safe(freeze["paths"]["start"]))
        parameters = load_npz(safe(freeze["paths"]["parameters"]))
        prototypes = load_npz(safe(freeze["paths"]["prototypes"]))
        beta = load_candidate_beta(safe(freeze["paths"]["candidate"]))
        ledger["status"] = "original_two_arm_construction_requested"
        persist()
        for mode in MODES:
            execute_arm(mode, graph, start, parameters, prototypes, beta)
        result["validity"] = "completed_two_arm_diagnostic"
        result["legacy_healthy_3s"] = (
            result["arms"]["legacy_canonical"]["validity"]
            == "healthy_3s_diagnostic")
        result["effector_healthy_3s"] = (
            result["arms"]["effector_canonical"]["validity"]
            == "healthy_3s_diagnostic")
        result["both_healthy_3s"] = (result["legacy_healthy_3s"]
                                      and result["effector_healthy_3s"])
        result["both_failed_health"] = (not result["legacy_healthy_3s"]
                                        and not result["effector_healthy_3s"])
        result["next_phase_condition_met"] = result["effector_healthy_3s"]
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        if thread_limits is not None:
            attempt("cpu_thread_restore", thread_limits.restore_original_limits)
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E224 internal wall limit exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E224 wall limit: " + repr(error))
        ledger["confirmed_neural_s"] = sum(
            event["actual_neural_s"] for event in ledger["clock_events"]
            if event["actual_neural_s"] is not None)
        ledger["confirmed_physical_s"] = sum(
            event["actual_physical_s"] for event in ledger["clock_events"]
            if event["actual_physical_s"] is not None)
        ledger["unknown_neural_intervals"] = sum(
            event["actual_neural_s"] is None for event in ledger["clock_events"])
        ledger["unknown_physical_intervals"] = sum(
            event["actual_physical_s"] is None for event in ledger["clock_events"])
        ledger["actual_neural_s"] = (ledger["confirmed_neural_s"]
            if not ledger["unknown_neural_intervals"] else None)
        ledger["actual_physical_s"] = (ledger["confirmed_physical_s"]
            if not ledger["unknown_physical_intervals"] else None)
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                     if path.is_file())
        if ledger["output_bytes"] > limits["output_bytes"]:
            error = OSError("E224 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E224 output cap: " + repr(error))
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: write_receipt(
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
                      "both_healthy_3s": result["both_healthy_3s"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
