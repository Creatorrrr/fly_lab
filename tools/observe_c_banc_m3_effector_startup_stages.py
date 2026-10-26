"""E235: bounded, passive RK-stage capture of the E224 effector startup."""

# Preserve the first scientific/runtime failure across independent raw cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import gc
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

from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools import verify_c_banc_m3_neural_current_body as e224
from tools.banc_m3_effector_stage_observer import (
    CHANNELS,
    EffectorStageObserverNetwork,
)
from tools.banc_m3_neural_current_trial import NeuralCurrentTrial
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "29c334dc8b31e38fa8bd86f69beba32bd025e5fb648dc48248355a5e45940538"
OLD_FREEZE_SHA = "40e313a01ccf51fcbc2a595aae74de8c72c9529f6427f9072f264fafca677ca0"
AMENDMENT_SHA = "e188197ee22ad891c03f2d4523f7b3d12d4301e86a8442854b7e6394834344c0"
FREEZE_SCHEMA = "flylab.m3.e235.effector-startup-stages.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e235.effector-startup-stages.review.v1"
OWNED = ("tools/observe_c_banc_m3_effector_startup_stages.py",
         "tests/test_c_banc_effector_startup_stages.py")
OUT = BASE / "e235-effector-startup-stages"
E208_METADATA = BASE / "e208-signed-input-channels/observation.json"
INTERNAL_S = 45.
RESERVE_S = 10.
EXPECTED_NATIVE = {"neurons": 158706, "motors": 391, "free_motors": 343,
    "fixed_motors": 48, "muscles": 96, "mode": "effector_canonical",
    "start_control": 0, "end_control": 99, "control_rows": 100,
    "neural_ticks": 1980, "physics_steps": 4950, "RK_stages": 7920,
    "neural_dt_s": .00025, "physics_dt_s": .0001, "control_dt_s": .005}
EXPECTED_CONSTRUCTION = {
    "restore_E224_start": {"captured_interval_entries": 7,
        "executed_interval_entries": 14,
        "known_completed_setup_steps": 280,
        "recorded_capture_setup_steps": 140},
    "construct_E234_observer": {"captured_interval_entries": 1,
        "executed_interval_entries": 2,
        "known_completed_setup_steps": 40,
        "recorded_capture_setup_steps": 20},
}


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
        raise ValueError("Repository-relative frozen E235 path required")
    return resolved


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """Complete byte/package/source/resource gate before GraphStore or model use."""
    if (Path(freeze_path).resolve() != (BASE /
            "e235-effector-startup-stages-freeze-v2.json").resolve()
            or freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E235 freeze and approved review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or freeze.get("native") != EXPECTED_NATIVE
            or limits.get("internal_s") != INTERNAL_S
            or limits.get("finalization_reserve_s") != RESERVE_S
            or limits.get("external_s") != 50
            or limits.get("supervisor_s") != 60
            or limits.get("cpu_threads") != 1 or limits.get("gpu_count") != 1
            or limits.get("original_attempts") != 1 or limits.get("retries") != 0
            or limits.get("body_constructors") != 1
            or limits.get("nominal_neural_constructors") != 8
            or limits.get("fit_calls") != 0 or limits.get("solver_calls") != 0
            or limits.get("resident_bytes") != 6 << 30
            or limits.get("allocated_gpu_bytes") != 3 << 30
            or limits.get("output_bytes") != 512 << 20
            or limits.get("minimum_free_bytes") != 2 << 30
            or freeze.get("candidate_adopted") is not False
            or freeze.get("F2_complete") is not False
            or freeze.get("F6_exposed") is not False
            or freeze.get("whole_goal_complete") is not False
            or freeze.get("active_UI_changed") is not False
            or freeze.get("original_cumulative_s") != 6090.0136165996555
            or freeze.get("previous_cumulative_limit_s") != 6119
            or freeze.get("allocation_increase_s") != 35
            or freeze.get("original_cumulative_limit_s") != 6154
            or freeze.get("expected_construction") != EXPECTED_CONSTRUCTION
            or freeze.get("amendment_sha256") != AMENDMENT_SHA
            or sha256(safe(freeze.get("amendment", ""))) != AMENDMENT_SHA
            or freeze.get("supersedes_freeze", {}).get("sha256") != OLD_FREEZE_SHA
            or sha256(safe(freeze.get("supersedes_freeze", {}).get("path", "")))
               != OLD_FREEZE_SHA):
        raise ValueError("Frozen E235 scientific/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)
            or E208_METADATA.relative_to(ROOT).as_posix()
               not in review.get("additional_input_sha256", {})):
        raise ValueError("Approved exact E235 source and E208 metadata review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Frozen E235 input/source inventory missing: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E235 input/source changed: " + name)
    for label in ("source_sha256", "additional_reused_source_sha256",
                  "additional_input_sha256"):
        mapping = review.get(label, {})
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E235 hash mapping invalid")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E235 source/input changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E235 frozen package version changed: " + package)
    if output.resolve() != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("New exact E235 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E235 minimum free disk unavailable")
    return freeze, review


def replace_network(session, *, network_type=EffectorStageObserverNetwork):
    """Reconstruct one observer network from the live E223 source, then bind it."""
    old = session.network
    saved = old.snapshot()
    patch = session.neural_current_trial_patch
    p, c, beta = patch["parameters"], patch["prototypes"], patch["beta"]
    free, fixed = p["free_columns"], p["fixed_columns"]
    candidate = network_type(source_matrix(old),
        *(getattr(old, key).cpu().numpy().copy()
          for key in ("tau", "a", "threshold", "cap")),
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
        device=old.device, cuda_implementation=old.cuda_implementation,
        capture_steps=old.interval_steps, core_ids=old.core_ids,
        core_gain=old.core_gain, motor_ids=old.motor_ids,
        motor_gains=old.motor_gains, motor_log_tau=old.motor_log_tau,
        free_motor_ids=old.motor_ids[free], fixed_motor_ids=old.motor_ids[fixed],
        membership=p["membership"][free], scale=p["scale"][free],
        centers=c["centers"][free], sigma2=c["sigma2"][free], beta=beta,
        observation_capacity=7920)
    if (candidate.identity != old.identity
            or candidate.source_timescale_identity
               != patch["source_timescale_identity"]):
        raise ValueError("E234 observer changed E223/source network identity")
    candidate.restore(saved)
    exact(saved, candidate.snapshot(), "E235 replaced whole neuronal snapshot")
    session.network = candidate
    session.adapter.network = candidate
    del old
    gc.collect()
    return candidate


def exact_control_prefix(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("E224 control raw schema changed")
    for key, value in actual.items():
        prior = expected[key][:len(value)]
        if (value.dtype != prior.dtype or value.shape != prior.shape
                or value.tobytes() != prior.tobytes()):
            raise ValueError("E235/E224 control raw changed: " + key)


def validate_observation(raw, counters, overflow, armed, metadata,
                         *, completed):
    """Validate after raw persistence; partial samples remain available."""
    if (set(raw) != set(CHANNELS) or counters.dtype != np.int64
            or counters.shape != (343,) or np.any(counters < 0)
            or overflow not in (0, 1) or armed != 0
            or metadata["capacity"] != 7920 or metadata["free_count"] != 343
            or metadata["start_tick"] != 0):
        raise ValueError("E235 observation ownership/counter contract changed")
    if completed and (not np.all(counters == 7920) or overflow != 0
                      or metadata["end_tick"] != 1980):
        raise ValueError("E235 complete stage count/overflow/end tick changed")
    for channel in CHANNELS:
        value = raw[channel]
        vector = channel in ("fast64", "slow64")
        shape = (7920, 343, 3) if vector else (7920, 343)
        dtype = np.float64 if channel.endswith("64") else np.float32
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("E235 stage raw shape/dtype changed: " + channel)
        if completed and not np.isfinite(value).all():
            raise ValueError("E235 completed stage raw nonfinite: " + channel)


def validate_construction(rows, expected=EXPECTED_CONSTRUCTION):
    """Charge synchronized setup/capture separately from 1980 science ticks."""
    if len(rows) != len(expected) or [row.get("label") for row in rows] != list(expected):
        raise ValueError("E235 exact nested construction profile required")
    for row in rows:
        if (not row.get("synchronize_returned")
                or not row.get("ownership_transferred")
                or not row.get("profiler_restored")
                or row.get("unknown_setup_steps") != 0
                or row.get("unknown_capture_setup_steps") != 0
                or row.get("unknown_capture_state_entries") != 0
                or row.get("interval_entries") != row.get("interval_returns")
                or any(row.get(key) != value
                       for key, value in expected[row["label"]].items())):
            raise ValueError("E235 measured setup/capture profile changed: "
                             + row["label"])
    return {"executed_setup_steps": sum(
                row["known_completed_setup_steps"] for row in rows),
            "captured_setup_steps": sum(
                row["recorded_capture_setup_steps"] for row in rows)}


def _channel_difference(observed, training, scale, kind):
    bits = observed.view(np.uint32) != training.view(np.uint32)
    locations = np.argwhere(bits)
    first = int(locations[0, 0]) if locations.size else None
    delta = observed.astype(np.float64) - training.astype(np.float64)
    if kind == "positive_recurrent" or kind == "negative_magnitude":
        transformed_observed = np.log1p(observed.astype(np.float64)) / scale
        transformed_training = np.log1p(training.astype(np.float64)) / scale
    else:
        transformed_observed = np.arcsinh(observed.astype(np.float64)) / scale
        transformed_training = np.arcsinh(training.astype(np.float64)) / scale
    transformed_delta = transformed_observed - transformed_training
    return {"first_80_stages_exact": bool(not bits[:80].any()),
        "first_difference_stage": first,
        "first_difference_neural_tick": first // 4 + 1 if first is not None else None,
        "first_difference_control": first // 80 + 1 if first is not None else None,
        "first_difference_free_column": (int(locations[0, 1])
            if locations.size else None),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "transformed_max_abs": float(np.max(np.abs(transformed_delta))),
        "transformed_rms": float(np.sqrt(np.mean(transformed_delta**2)))}


def compare_training_inputs(raw, training, motor_ids, free_ids, scale):
    """Compare same-index E208 float32 inputs without resampling or feedback."""
    motor = np.asarray(motor_ids)
    free = np.asarray(free_ids)
    if (motor.dtype != np.int64 or motor.shape != (391,)
            or free.dtype != np.int64 or free.shape != (343,)
            or scale.dtype != np.float64 or scale.shape != (343, 3)
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Ordered E208/E217 motor and transform ownership required")
    if (len(np.unique(motor)) != 391 or len(np.unique(free)) != 343
            or not np.array_equal(free, motor[np.isin(motor, free)])):
        raise ValueError("E208 motor roster invalid")
    columns = np.array([np.flatnonzero(motor == neuron)[0] for neuron in free])
    result = {}
    for k, channel in enumerate(CHANNELS[:3]):
        source = training[channel]
        if source.dtype != np.float32 or source.shape != (160000, 391):
            raise ValueError("Frozen E208 native input matrix changed: " + channel)
        current = raw[channel]
        if current.dtype != np.float32 or current.shape != (7920, 343):
            raise ValueError("E235 observed native input matrix changed: " + channel)
        prior = np.asarray(source[:7920, columns], np.float32)
        if not np.isfinite(current).all() or not np.isfinite(prior).all():
            raise ValueError("Finite E235/E208 stage input comparison required")
        result[channel] = _channel_difference(current, prior, scale[:, k], channel)
    result["first_5ms_inputs_exact"] = all(
        result[channel]["first_80_stages_exact"] for channel in CHANNELS[:3])
    result["any_input_difference"] = any(
        result[channel]["first_difference_stage"] is not None
        for channel in CHANNELS[:3])
    result["any_difference_after_first_5ms"] = any(
        result[channel]["first_difference_stage"] is not None
        and result[channel]["first_difference_stage"] >= 80
        for channel in CHANNELS[:3])
    return result


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def save_stage_raw(out, observer):
    """Persist each recoverable channel independently after sealing the epoch."""
    if not getattr(observer, "_observation_ever_armed", False):
        return
    observer.disable_observation()
    metadata = observer.observation_static_metadata()
    metadata["motor_ids"] = observer.native_motor_ids.tolist()
    metadata["free_motor_ids"] = observer.free_motor_ids.tolist()
    errors = []
    if not (out / "observation.json").exists():
        try:
            e224.write_receipt(out / "observation.json", metadata)
        except BaseException as error:
            errors.append(("metadata", error))
    for channel in CHANNELS:
        destination = out / ("stage-" + channel + ".npy")
        if not destination.exists():
            try:
                temporary = out / ("stage-" + channel + ".tmp.npy")
                np.save(temporary, observer.copy_values(channel), allow_pickle=False)
                temporary.replace(destination)
            except BaseException as error:
                errors.append((channel, error))
    if not (out / "observation.npz").exists():
        try:
            temporary = out / "observation.tmp.npz"
            np.savez(temporary, counters=observer.copy_counters(),
                     overflow=np.array(observer.copy_overflow(), np.int32),
                     armed=np.array(observer.copy_armed(), np.int32))
            temporary.replace(out / "observation.npz")
        except BaseException as error:
            errors.append(("counters", error))
    if errors:
        first = errors[0][1]
        for label, error in errors[1:]:
            first.add_note("Secondary E235 stage raw " + label + ": " + repr(error))
        raise first


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e235.effector-startup-stages.ledger.v1",
        "status": "gate_passed", "original_processes": 1,
        "nominal_neural_constructors": {
            "E224_restore_internal": 7, "E234_observer": 1},
        "body_constructors_requested": 0, "body_constructors_returned": 0,
        "observer_constructors_requested": 0, "observer_constructors_returned": 0,
        "controls_requested": 0, "controls_completed": 0,
        "construction": [], "clock_events": [], "storage_errors": [],
        "retries": 0, "fit_calls": 0, "solver_calls": 0,
        "candidate_adopted": False, "F6_exposed": False,
        "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e235.effector-startup-stages.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "F3_F5_10s_evaluated": False}
    session = observer = journal = restore_native = thread_limits = None
    process = None
    gpu_free_before = None
    clock = e113.AdvanceLedger()
    rows, controls, native = [], [], []
    trace_last = -1
    primary = None
    measured = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        e224.write_receipt(out / "advance-ledger.json", ledger)

    def attempt(label, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E235 " + label + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E235 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E235 resident memory cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            available, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before - available)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E235 additional CUDA memory cap exceeded")

    def own_session(value):
        nonlocal session
        session = value

    def own_observer(value):
        nonlocal observer
        observer = value

    try:
        e224.write_receipt(out / "source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        import psutil
        process = psutil.Process()
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        graph = protocol.GraphStore.load(safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("Exact E235 full BANC graph required")
        start = protocol.StateStore.load(safe(freeze["paths"]["start"]))
        ledger["body_constructors_requested"] = 1
        ledger["status"] = "restore_E224_start_requested"
        persist()
        session = protocol.profiled_construction("restore_E224_start",
            lambda: NeuralCurrentTrial.from_checkpoint(graph, start),
            ledger["construction"], persist, on_return=own_session)
        ledger["body_constructors_returned"] = 1
        exact(start, session.snapshot(), "E235 restored E224 start")
        ledger["observer_constructors_requested"] = 1
        ledger["status"] = "construct_E234_observer_requested"
        persist()
        observer = protocol.profiled_construction("construct_E234_observer",
            lambda: replace_network(session), ledger["construction"], persist,
            on_return=own_observer)
        ledger["observer_constructors_returned"] = 1
        ledger["synchronized_setup"] = validate_construction(
            ledger["construction"], freeze["expected_construction"])
        exact(start, session.snapshot(), "E235 whole observer start")
        protocol.StateStore.save(out / "observed-start", session.snapshot())
        restore_native = e113.install_native_observer(session.body, native)
        journal = e224.FiniteControlJournal(out / "control-journal.jsonl")
        row0, value0 = e224.capture_state(session, "effector_canonical")
        rows.append(row0)
        controls.append(value0)
        observer.arm_observation()
        with (out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(e224.finite_receipt(protocol.jsonable(row0))[0],
                                    allow_nan=False) + "\n")
            trace_last = 0
            for tick in range(1, 100):
                budget()
                before = (int(session.control_tick), int(session.network.tick),
                          float(session.body.physics_time()), len(native))
                ledger["controls_requested"] += 1
                event = protocol.advance_control(session, "startup", tick,
                                                 clock, journal)
                try:
                    row, value = e224.capture_state(session, "effector_canonical")
                    rows.append(row)
                    controls.append(value)
                    protocol.record_readout(journal, "startup", tick, event,
                                            complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, "startup", tick, event,
                                                complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E235 readout journal: "
                                       + repr(secondary))
                    raise
                if session.fault or session.body.fault:
                    raise e113.NativeHealthStop("E235 measured body fault")
                if (session.control_tick != before[0] + 1
                        or session.network.tick != before[1] + 20
                        or abs(session.body.physics_time() - before[2] - .005) > 2e-8
                        or len(native) != before[3] + 50):
                    raise ValueError("E235 incomplete control/native interval")
                stream.write(json.dumps(e224.finite_receipt(protocol.jsonable(row))[0],
                                        allow_nan=False) + "\n")
                trace_last = tick
                ledger["controls_completed"] += 1
                if tick == 1:
                    protocol.StateStore.save(out / "checkpoint-5ms",
                                             session.snapshot())
        e224.save_control(out / "control-raw.npz", controls, "effector_canonical")
        e224.save_native_contact(out / "native-contact.npz", native)
        protocol.StateStore.save(out / "final", session.snapshot())
        observer.disable_observation()
        save_stage_raw(out, observer)
        ledger["status"] = "raw_saved_before_validation"
        persist()
        if (len(controls) != 100 or len(native) != 4950
                or session.control_tick != 99 or session.network.tick != 1980
                or not clock.clock_valid or session.fault or session.body.fault):
            raise ValueError("E235 complete 99-control clock/health required")
        exact(start, protocol.StateStore.load(out / "observed-start"),
              "E235 observed start disk snapshot")
        exact(protocol.StateStore.load(safe(freeze["paths"]["checkpoint_5ms"])),
              protocol.StateStore.load(out / "checkpoint-5ms"),
              "E235 whole 5ms checkpoint")
        with (np.load(out / "control-raw.npz", allow_pickle=False) as actual,
              np.load(safe(freeze["paths"]["control"]), allow_pickle=False) as prior):
            exact_control_prefix(dict(actual), dict(prior))
        with (np.load(out / "native-contact.npz", allow_pickle=False) as actual,
              np.load(safe(freeze["paths"]["native"]), allow_pickle=False) as prior):
            exact_control_prefix(dict(actual), dict(prior))
        original_trace = [json.loads(line) for line in
            safe(freeze["paths"]["trace"]).read_text(encoding="utf-8").splitlines()]
        current_trace = [json.loads(line) for line in
            (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(current_trace) != 100 or len(original_trace) < 100:
            raise ValueError("E235/E224 trace length changed")
        exact(original_trace[:100], current_trace, "E235/E224 trace prefix")
        with np.load(out / "observation.npz", allow_pickle=False) as packet:
            counters = packet["counters"].copy()
            overflow = int(packet["overflow"])
            armed = int(packet["armed"])
        metadata = json.loads((out / "observation.json").read_text(encoding="utf-8"))
        stage = {channel: np.load(out / ("stage-" + channel + ".npy"),
                                   mmap_mode="r") for channel in CHANNELS}
        validate_observation(stage, counters, overflow, armed, metadata,
                             completed=True)
        parameters = e224.load_npz(safe(freeze["paths"]["parameters"]))
        e208_meta = json.loads(E208_METADATA.read_text(encoding="utf-8"))
        if (not np.array_equal(np.asarray(e208_meta["motor_ids"], np.int64),
                               parameters["motor_ids"])
                or metadata["motor_ids"] != parameters["motor_ids"].tolist()
                or metadata["free_motor_ids"] != parameters["motor_ids"][
                    parameters["free_columns"]].tolist()):
            raise ValueError("E208/E217/E235 ordered MN ownership changed")
        training = {channel: np.load(safe(freeze["paths"]["training_" + label]),
                                     mmap_mode="r")
            for channel, label in zip(CHANNELS[:3], ("positive", "negative", "drive"))}
        comparison = compare_training_inputs(stage, training,
            parameters["motor_ids"],
            parameters["motor_ids"][parameters["free_columns"]],
            parameters["scale"][parameters["free_columns"]])
        e224.write_receipt(out / "input-comparison.json", comparison)
        if not comparison["first_5ms_inputs_exact"]:
            next_step = "inspect_initial_state_or_stage_collection"
        elif comparison["any_difference_after_first_5ms"]:
            next_step = "compare_actual_stage_inputs_and_sensory_path"
        else:
            next_step = "inspect_memory_response_equation_or_rounding"
        result.update(validity="valid_capture", execution_status="complete",
            control_rows=len(controls), native_samples=len(native),
            recorded_stages=int(counters[0]), first_5ms_inputs_exact=(
                comparison["first_5ms_inputs_exact"]),
            any_input_difference=comparison["any_input_difference"],
            next_step=next_step,
            conclusion="same-index passive input observation only")
    except e113.NativeHealthStop as error:
        measured = error
        result.update(validity="measured_native_health_stop",
                      measured_error=repr(error), numeric_fault=bool(
                          any(not np.isfinite(item[1]).all()
                              or not np.isfinite(item[5]) for item in native)))
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        # Every available raw is attempted independently, including partial work.
        if observer is not None:
            attempt("stage-observer", lambda: save_stage_raw(out, observer))
        if not (out / "control-raw.npz").exists():
            attempt("control-raw.npz", lambda: e224.save_control(
                out / "control-raw.npz", controls, "effector_canonical"))
        if not (out / "native-contact.npz").exists():
            attempt("native-contact.npz", lambda: e224.save_native_contact(
                out / "native-contact.npz", native))
        for label, callback in (("native-observer", restore_native),
                                ("control-journal", journal.close if journal else None),
                                ("session", (lambda: None if session.closed
                                    else session.close()) if session else None),
                                ("cpu-threads", thread_limits.restore_original_limits
                                 if thread_limits else None)):
            if callback is not None:
                attempt("cleanup-" + label, callback)
        ledger["clock_events"] = clock.events
        ledger["actual_neural_s"] = clock.neural_s
        ledger["actual_physical_s"] = clock.physical_s
        ledger["clock_valid"] = clock.clock_valid
        ledger["native_samples"] = len(native)
        ledger["control_rows"] = len(controls)
        ledger["last_durable_trace_index"] = trace_last
        ledger["journal"] = journal.summary() if journal else None
        ledger["measured_health_stop"] = measured is not None
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E235 internal wall limit exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E235 wall limit: " + repr(error))
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                      if path.is_file())
        if ledger["output_bytes"] > limits["output_bytes"]:
            error = OSError("E235 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E235 output cap: " + repr(error))
        if primary is not None and result["validity"] == "valid_capture":
            result["pre_storage_validity"] = result["validity"]
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: e224.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: e224.write_receipt(
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
        result = run(args.freeze, args.freeze_sha256,
                     args.review, args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"],
                      "first_5ms_inputs_exact": result.get("first_5ms_inputs_exact"),
                      "measured_error": result.get("measured_error")},
                     sort_keys=True))
    return 0 if result["validity"] == "valid_capture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
