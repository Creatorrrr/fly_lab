"""One-shot E201 collection of actual E150 native RK4 stage inputs.

The parent invokes this only after independent source review. No model is fit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_c_banc_m3_native_current import (
    AdvanceLedger,
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SCHEMA = "flylab.m3.e201.native-stage-input.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e201.native-stage-input.review.v1"
OWNED_FILES = (
    "tools/banc_m3_timescale_stage_input.py",
    "tools/observe_c_banc_m3_timescale_stage_input.py",
    "tests/test_c_banc_timescale_stage_input.py",
    "tests/test_c_banc_timescale_stage_capture.py",
)
SHAPE = {"neurons": 158706, "graph_edges": 11584852, "motors": 391,
         "muscles": 96, "controls": 2000, "samples": 2001,
         "stages": 160000, "neural_ticks": 40000,
         "physics_steps": 100000}
CLOCK = {"neural_dt_s": .00025, "physics_dt_s": .0001,
         "control_dt_s": .005, "future_control": 220}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "internal_s": 270,
            "model_s": 245, "external_s": 285, "supervisor_s": 300,
            "output_bytes": 2 * 1024**3,
            "observation_bytes": 512 * 1024**2,
            "minimum_free_bytes": 12 * 1024**3,
            "original_attempts": 1, "retries": 0}
GRAPH_HASH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"


def repository_path(relative: str, *, file: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe frozen repository path: " + relative)
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped frozen path: " + relative)
    if file and not path.is_file():
        raise ValueError("Frozen file required: " + relative)
    return path


def verify_contract(freeze_path, freeze_sha, review_path, review_sha,
                    *, elapsed=None):
    """Verify both gates and every frozen byte before graph/model construction."""
    freeze_path, review_path = Path(freeze_path), Path(review_path)
    if len(freeze_sha) != 64 or file_sha(freeze_path) != freeze_sha:
        raise ValueError("E201 freeze SHA changed")
    if len(review_sha) != 64 or file_sha(review_path) != review_sha:
        raise ValueError("E201 review SHA changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E201_NATIVE_STAGE_INPUT_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED_FILES)
            or freeze.get("shape") != SHAPE or freeze.get("clock") != CLOCK
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("graph_hash") != GRAPH_HASH
            or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("new_fit") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E201 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"], file=True)) != freeze.get("design_sha256"):
        raise ValueError("E201 frozen design changed")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "graph", "start", "future", "final", "trace", "native",
            "signals", "metadata", "adapter", "muscles"}:
        raise ValueError("Exact E201 original paths required")
    for key, relative in paths.items():
        repository_path(relative, file=key not in {"graph", "start", "future", "final"})
    for label, count in (("input_sha256", 250),
                         ("reused_source_sha256", 27),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E201 " + label + " cardinality changed")
        for relative, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Malformed E201 frozen SHA")
            if file_sha(repository_path(relative, file=True)) != expected:
                raise ValueError("E201 frozen input changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E201 preflight exhausted model deadline")
    source_hashes = {name: file_sha(repository_path(name, file=True))
                     for name in OWNED_FILES}
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != source_hashes):
        raise ValueError("E201 parent source review does not authorize these sources")
    for relative, expected in review.get("additional_reused_source_sha256", {}).items():
        if file_sha(repository_path(relative, file=True)) != expected:
            raise ValueError("E201 reviewed transitive source changed: " + relative)
    return freeze, review


def _attempt(errors, label, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve siblings on failure
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def save_observation(out, observer, errors):
    """Preserve metadata and full NaN-padded NPY independently on failure."""
    out = Path(out)
    _attempt(errors, "observation-disable", observer.disable_observation)
    metadata = _attempt(errors, "observation-static", observer.observation_static_metadata)
    if metadata is None:
        metadata = {"start_tick": None, "end_tick": None}
    for key, getter in (("counter_requested", observer.copy_counters),
                        ("overflow", observer.copy_overflow),
                        ("armed", observer.copy_armed)):
        metadata[key] = _attempt(errors, key, getter)
    counts = metadata.get("counter_requested")
    capacity = metadata.get("capacity_stages")
    metadata["counter_completed"] = (np.minimum(counts, capacity)
        if counts is not None and capacity is not None else None)
    _attempt(errors, "observation.json", lambda: save_json(out / "observation.json", metadata))
    values = _attempt(errors, "current-hostcopy", observer.copy_values)
    if values is not None:
        _attempt(errors, "motor-total-current.npy", lambda: np.save(
            out / "motor-total-current.npy", values, allow_pickle=False))
    return metadata


def endpoint(session, motor_ids):
    return {"control_tick": int(session.control_tick),
            "network_tick": int(session.network.tick),
            "physics_time_s": float(session.body.physics_time()),
            "motor_rate": np.asarray(session.network.readout(motor_ids),
                                     np.float64).copy(),
            "adaptation": session.network.adaptation[motor_ids].cpu().numpy().copy(),
            "activation": np.asarray(session.adapter.activation, np.float64).copy(),
            "bank": np.asarray(session.adapter.bank, np.float64).copy()}


def save_endpoints(path, records, motor_ids):
    fields = ("control_tick", "network_tick", "physics_time_s", "motor_rate",
              "adaptation", "activation", "bank")
    shapes = {"motor_rate": (391,), "adaptation": (391,),
              "activation": (96,), "bank": (391, 3)}
    arrays = {}
    for key in fields:
        dtype = (np.int64 if key in ("control_tick", "network_tick")
                 else np.float32 if key == "adaptation" else np.float64)
        arrays[key] = np.asarray([r[key] for r in records], dtype=dtype).reshape(
            (len(records), *shapes.get(key, ())))
    arrays["motor_ids"] = np.asarray(motor_ids, dtype=np.int64)
    arrays["recorded_samples"] = np.asarray(len(records), dtype=np.int64)
    np.savez_compressed(path, **arrays)


def save_motor_parameters(path, session, motor_ids, original_paths):
    """Capture actual E150 arrays and anatomical ownership, without defaults."""
    net = session.network
    motor_ids = np.asarray(motor_ids, np.int64)
    membership = np.asarray(session.monotone_recruitment_patch["membership"], np.int64)
    if membership.shape != (391,):
        raise ValueError("Actual E150 391-MN membership required")
    muscles = json.loads(original_paths["muscles"].read_text(encoding="utf-8"))
    with np.load(original_paths["metadata"], allow_pickle=False) as original:
        if not np.array_equal(original["motor_ids"], motor_ids):
            raise ValueError("E150 frozen metadata MN roster changed")
    arrays = {"motor_ids": motor_ids,
              "initial_rate": net.rate[motor_ids].cpu().numpy().copy(),
              "initial_adaptation": net.adaptation[motor_ids].cpu().numpy().copy(),
              "tau": net.tau[motor_ids].cpu().numpy().copy(),
              "a": net.a[motor_ids].cpu().numpy().copy(),
              "threshold": net.threshold[motor_ids].cpu().numpy().copy(),
              "cap": net.cap[motor_ids].cpu().numpy().copy(),
              "coefficient": net.coefficient[motor_ids].cpu().numpy().copy(),
              "adaptation_gain": net._cell_gain[motor_ids].cpu().numpy().copy(),
              "adaptation_tau": net._cell_adaptation_tau[motor_ids].cpu().numpy().copy(),
              "inverse_adaptation_tau": net._cell_inverse_tau[motor_ids].cpu().numpy().copy(),
              "motor_log_tau": net.motor_log_tau.copy(),
              "membership": membership.copy(),
              "pool": session.adapter.calibration["pool"].copy(),
              "moment": session.adapter.calibration["moment"].copy(),
              "rest": session.adapter.calibration["rest"].copy(),
              "ltm": session.adapter.calibration["ltm"].copy(),
              "muscle_parameters_json": np.asarray(json.dumps(
                  muscles["muscles"], sort_keys=True, ensure_ascii=False))}
    for name, value in arrays.items():
        if name != "muscle_parameters_json" and not np.isfinite(value).all():
            raise ValueError("Nonfinite actual E150 parameter: " + name)
    np.savez_compressed(path, **arrays)


def compare_signal_adaptation(path, motor_ids, records):
    """E150 detailed signals start at control 1; native state includes row 0."""
    from tools.verify_c_banc_observation import exact

    with np.load(path, allow_pickle=False) as original:
        observed_ids = original["observed_ids"]
        positions = {int(value): index for index, value in enumerate(observed_ids)}
        if (len(positions) != len(observed_ids)
                or not all(int(value) in positions for value in motor_ids)):
            raise ValueError("Original E150 signal MN mapping changed")
        columns = [positions[int(value)] for value in motor_ids]
        adaptation = original["adaptation"]
        if adaptation.shape != (len(records) - 1, len(observed_ids)):
            raise ValueError("Original E150 adaptation signal shape changed")
        exact(adaptation[:, columns], np.asarray([r["adaptation"] for r in records[1:]],
              dtype=adaptation.dtype), "E201 original adaptation")


def validate_raw(freeze, paths, out, rows, records, metadata, snapshots):
    """Evaluate only after raw files are stored; no permissive comparisons."""
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    n = SHAPE["samples"]
    if len(rows) != n or len(records) != n:
        raise ValueError("Incomplete E201 2001 original boundaries")
    for label, source, observed in (
        ("start", paths["start"], snapshots["start"]),
        ("main-1p1", paths["future"], snapshots["future"]),
        ("final", paths["final"], snapshots["final"]),
    ):
        exact(StateStore.load(source), observed, "E201 exact " + label)
    with paths["trace"].open(encoding="utf-8") as stream:
        original_trace = json.load(stream)
    exact(original_trace["samples"], rows, "E201 full original trace")
    with np.load(paths["native"], allow_pickle=False) as original:
        for key in ("control_tick", "network_tick", "physics_time_s",
                    "motor_rate", "activation", "bank"):
            exact(original[key], np.asarray([r[key] for r in records],
                dtype=original[key].dtype), "E201 original native " + key)
        for key in ("complete", "motor_rate_available", "activation_available",
                    "bank_available"):
            exact(original[key], np.ones(n, bool), "E201 original native " + key)
    compare_signal_adaptation(paths["signals"],
        np.asarray(metadata["motor_ids"], np.int64), records)
    if (metadata.get("start_tick") != 0 or metadata.get("end_tick") != 40000
            or metadata.get("capacity_stages") != 160000):
        raise ValueError("E201 native stage clock/capacity changed")
    counts = np.asarray(metadata["counter_requested"])
    if (counts.shape != (391,) or not np.array_equal(counts,
            np.full(391, 160000, dtype=np.int64))
            or np.asarray(metadata["overflow"]).tolist() != [0]
            or np.asarray(metadata["armed"]).tolist() != [0]):
        raise ValueError("E201 native stage count/overflow/armed mismatch")
    stages = np.load(out / "motor-total-current.npy", mmap_mode="r", allow_pickle=False)
    if (stages.shape != (160000, 391) or stages.dtype != np.float32
            or not np.isfinite(stages).all()):
        raise ValueError("E201 native stage array shape/dtype/finite mismatch")
    return {"snapshot_start_exact": True, "snapshot_future_exact": True,
            "snapshot_final_exact": True, "trace_2001_exact": True,
            "native_rate_activation_bank_exact": True,
            "native_adaptation_exact": True,
            "current_160000x391_finite": True}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"output-manifest.json", "result.json"}:
            relative = path.relative_to(out).as_posix()
            files[relative] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e201.output-manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def run(freeze_path, freeze_sha, review_path, review_sha, out):
    began = time.perf_counter()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("One-shot E201 output exists: " + str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    freeze = review = session = observer = motor_ids = None
    rows, records = [], []
    snapshots = {}
    metadata = None
    ledger = AdvanceLedger()
    setup = {"models_constructed": 0, "models_completed_lower_bound": 0,
             "model_setup_completion": "not_started",
             "warmup_native_ticks": 0, "capture_native_ticks": 0,
             "warmup_completed_lower_bound": 0,
             "capture_completed_lower_bound": 0,
             "actual_main_neural_ticks": None, "actual_main_physics_steps": None,
             "attempted_controls": 0, "synchronized_controls": 0,
             "recorded_control_endpoints": 0}
    result = {"schema": "flylab.m3.e201.native-stage-input.result.v1",
              "validity": "incomplete", "candidate_adopted": False,
              "F6_exposed": False, "whole_goal_complete": False,
              "fit_calls": 0, "new_gait_candidate": False}
    paths = None

    def elapsed():
        return time.perf_counter() - began

    try:
        freeze, review = verify_contract(freeze_path, freeze_sha,
            review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E201 requires 12 GiB free disk")
        paths = {key: repository_path(value, file=key not in {
            "graph", "start", "future", "final"})
            for key, value in freeze["paths"].items()}
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "design_sha256": freeze["design_sha256"],
            "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "additional_reused_source_sha256": review.get("additional_reused_source_sha256", {})})
        threads = configure_single_thread_runtime()
        save_json(out / "threading.json", threads)
        import torch

        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import capture_control
        from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
        from tools.banc_m3_timescale_stage_input import (
            replace_with_timescale_stage_network,
        )
        from tools.verify_c_banc_observation import exact

        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA GPU required; no fallback")
        save_json(out / "environment.json", {"python": sys.version,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "pid": os.getpid(),
            "actual_cuda": True})
        graph = GraphStore.load(paths["graph"])
        if graph.n != SHAPE["neurons"] or len(graph.indices) != SHAPE["graph_edges"] or graph.hash != GRAPH_HASH:
            raise ValueError("Frozen full BANC graph identity changed")
        start = StateStore.load(paths["start"])
        if start["control_tick"] != 0 or start["network"]["tick"] != 0:
            raise ValueError("E150 original start clock changed")
        setup["model_setup_completion"] = "E150_restore_requested_completion_unknown"
        setup["models_constructed"] = None
        setup["warmup_native_ticks"] = None
        setup["capture_native_ticks"] = None
        session = MonotoneRecruitmentTrial.from_checkpoint(graph, start)
        setup["models_constructed"] = 6
        setup["models_completed_lower_bound"] = 6
        setup["warmup_native_ticks"] = 12 * session.network.interval_steps
        setup["capture_native_ticks"] = 6 * session.network.interval_steps
        setup["warmup_completed_lower_bound"] = setup["warmup_native_ticks"]
        setup["capture_completed_lower_bound"] = setup["capture_native_ticks"]
        setup["model_setup_completion"] = "E150_restore_complete"
        exact(start, session.snapshot(), "E201 restored E150 whole start")
        if session.network.device != "cuda" or session.network.cuda_implementation != "packed":
            raise ValueError("Original E150 packed CUDA required")
        setup["model_setup_completion"] = "observer_requested_completion_unknown"
        setup["models_constructed"] = None
        setup["warmup_native_ticks"] = None
        setup["capture_native_ticks"] = None
        observer = replace_with_timescale_stage_network(session)
        setup["models_constructed"] = 7
        setup["models_completed_lower_bound"] = 7
        setup["warmup_native_ticks"] = 14 * observer.interval_steps
        setup["capture_native_ticks"] = 7 * observer.interval_steps
        setup["warmup_completed_lower_bound"] = setup["warmup_native_ticks"]
        setup["capture_completed_lower_bound"] = setup["capture_native_ticks"]
        setup["model_setup_completion"] = "observer_complete"
        exact(start, session.snapshot(), "E201 observed E150 whole start")
        motor_ids = observer.observation_ids.copy()
        save_motor_parameters(out / "motor-parameters.npz", session, motor_ids, paths)
        snapshots["start"] = session.snapshot()
        StateStore.save(out / "observed-start", snapshots["start"])
        observer.arm_observation()
        rows.append(capture_control(session))
        records.append(endpoint(session, motor_ids))
        for control in range(1, SHAPE["controls"] + 1):
            if elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E201 model collection deadline reached")
            ledger.advance(session)
            if (session.control_tick != control or observer.tick != 20 * control
                    or abs(session.body.physics_time() - .005 * control) > 1e-8):
                raise ValueError("E201 original neural/body/control clocks diverged")
            rows.append(capture_control(session))
            records.append(endpoint(session, motor_ids))
            if control == CLOCK["future_control"]:
                snapshots["future"] = session.snapshot()
                StateStore.save(out / "observed-main-1p1", snapshots["future"])
            if session.fault or session.body.fault:
                raise RuntimeError("E201 original native body health fault")
        snapshots["final"] = session.snapshot()
        StateStore.save(out / "observed-final", snapshots["final"])
        torch.cuda.current_stream().synchronize()
    except BaseException as error:  # noqa: BLE001 - raw preservation remains mandatory
        primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    finally:
        totals = ledger.totals()
        setup["attempted_controls"] = totals["requested_controls"]
        setup["synchronized_controls"] = totals["synchronized_controls"]
        setup["recorded_control_endpoints"] = len(records)
        setup["actual_main_neural_ticks"] = (
            None if totals["actual_neural_s"] is None else
            round(totals["actual_neural_s"] / CLOCK["neural_dt_s"]))
        setup["actual_main_physics_steps"] = (
            None if totals["actual_physical_s"] is None else
            round(totals["actual_physical_s"] / CLOCK["physics_dt_s"]))
        if observer is not None:
            metadata = save_observation(out, observer, errors)
        _attempt(errors, "advance-ledger.json", lambda: save_json(
            out / "advance-ledger.json", {"schema": "flylab.m3.e201.ledger.v1",
                "events": ledger.events, "totals": totals,
                "setup": setup, "process_wall_s": elapsed()}))
        _attempt(errors, "trace.json", lambda: save_json(out / "trace.json",
            {"schema": "flylab.m3.e201.trace.v1", "samples": rows}))
        if motor_ids is not None:
            _attempt(errors, "endpoints.npz", lambda: save_endpoints(
                out / "endpoints.npz", records, motor_ids))
        if session is not None:
            _attempt(errors, "session-close", session.close)

    if primary is None and errors:
        primary = RuntimeError("E201 raw artifact or cleanup failed")
    if primary is None:
        _attempt(errors, "evaluate", lambda: result.update(
            comparisons=validate_raw(freeze, paths, out, rows,
                records, metadata, snapshots)))
        if errors:
            primary = RuntimeError("E201 exact baseline evaluation failed")
    if primary is None:
        result["validity"] = "valid_exact_native_stage_dataset"
    else:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                    "message": str(primary)}
    result.update(process_wall_s=elapsed(), setup=setup,
        raw_errors=errors, prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=((freeze or {}).get("prior_cumulative_s", 0)
                                   + elapsed()),
        original_attempts=1, retries=0)
    _attempt(errors, "execution-errors.json", lambda: save_json(
        out / "execution-errors.json", errors))
    preliminary = copy.deepcopy(result)
    preliminary.update(validity="incomplete_or_invalid", finalization_pending=True)
    _attempt(errors, "result.json-preliminary", lambda: save_json(
        out / "result.json", preliminary))
    manifest = _attempt(errors, "output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "output manifest missing or output exceeds 2 GiB"
        _attempt(errors, "result.json-final", lambda: save_json(out / "result.json", result))
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E201 internal 270 s exceeded"
        _attempt(errors, "result.json-deadline", lambda: save_json(out / "result.json", result))
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if not errors:
        result["finalization_pending"] = False
        _attempt(errors, "result.json-final", lambda: save_json(
            out / "result.json", result))
    if errors and primary is None:
        result["validity"] = "incomplete_or_invalid"
        primary = RuntimeError("E201 final raw/result persistence failed")
    if errors:
        _attempt(errors, "publication-failure.json", lambda: save_json(
            out / "publication-failure.json", errors))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review,
        args.review_sha256, args.out)


if __name__ == "__main__":
    main()
