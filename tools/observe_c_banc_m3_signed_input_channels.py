"""One-shot E208 collection of actual E150 signed RK4 stage inputs.

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

FREEZE_SCHEMA = "flylab.m3.e208.signed-input-channels.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e208.signed-input-channels.review.v1"
OWNED_FILES = (
    "tools/banc_m3_signed_input_channels.py",
    "tools/observe_c_banc_m3_signed_input_channels.py",
    "tests/test_c_banc_signed_input_channels.py",
)
SHAPE = {"neurons": 158706, "graph_edges": 11584852, "motors": 391,
         "muscles": 96, "controls": 2000, "samples": 2001,
         "stages": 160000, "neural_ticks": 40000,
         "physics_steps": 100000}
CLOCK = {"neural_dt_s": .00025, "physics_dt_s": .0001,
         "control_dt_s": .005, "future_control": 220}
RESOURCE = {"cpu_threads": 1, "gpu_count": 1, "internal_s": 135,
            "model_s": 115, "external_s": 150, "supervisor_s": 155,
            "output_bytes": 2 * 1024**3,
            "observation_bytes": 1024**3,
            "resident_bytes": 12 * 1024**3,
            "minimum_free_bytes": 12 * 1024**3,
            "original_attempts": 1, "retries": 0}
GRAPH_HASH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
CHANNEL_FILES = {
    "total_current": "motor-total-current.npy",
    "positive_recurrent": "motor-positive-recurrent.npy",
    "negative_magnitude": "motor-negative-magnitude.npy",
    "drive": "motor-drive.npy",
}


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
        raise ValueError("E208 freeze SHA changed")
    if len(review_sha) != 64 or file_sha(review_path) != review_sha:
        raise ValueError("E208 review SHA changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E208_SIGNED_INPUT_CHANNELS_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED_FILES)
            or freeze.get("shape") != SHAPE or freeze.get("clock") != CLOCK
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("graph_hash") != GRAPH_HASH
            or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("new_fit") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E208 frozen scientific/resource contract changed")
    observation = freeze.get("observation", {})
    hypothesis = freeze.get("hypothesis", {})
    if (observation.get("channels") != list(CHANNEL_FILES)
            or observation.get("dtype") != "float32"
            or observation.get("shape") != [160000, 391]
            or observation.get("balance_atol") != 1e-5
            or observation.get("balance_roundoff_multiplier") != 64
            or observation.get("stages_per_control") != 80
            or hypothesis.get("closed_motor_count") != 25
            or hypothesis.get("support_count") != 13
            or hypothesis.get("positive_rms_min") != .001):
        raise ValueError("E208 frozen channel/hypothesis contract changed")
    if file_sha(repository_path(freeze["design"], file=True)) != freeze.get("design_sha256"):
        raise ValueError("E208 frozen design changed")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "graph", "start", "future", "final", "trace", "native",
            "signals", "metadata", "adapter", "muscles", "stage_current",
            "parameters", "targets", "late_input_review"}:
        raise ValueError("Exact E208 original paths required")
    for key, relative in paths.items():
        repository_path(relative, file=key not in {"graph", "start", "future", "final"})
    for label, count in (("input_sha256", 257),
                         ("reused_source_sha256", 31),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E208 " + label + " cardinality changed")
        for relative, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Malformed E208 frozen SHA")
            if file_sha(repository_path(relative, file=True)) != expected:
                raise ValueError("E208 frozen input changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E208 preflight exhausted model deadline")
    source_hashes = {name: file_sha(repository_path(name, file=True))
                     for name in OWNED_FILES}
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != source_hashes):
        raise ValueError("E208 parent source review does not authorize these sources")
    for relative, expected in review.get("additional_reused_source_sha256", {}).items():
        if file_sha(repository_path(relative, file=True)) != expected:
            raise ValueError("E208 reviewed transitive source changed: " + relative)
    return freeze, review


def _attempt(errors, label, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve siblings on failure
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def save_observation(out, observer, errors):
    """Preserve metadata and each full NaN-padded NPY independently."""
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
    for channel, filename in CHANNEL_FILES.items():
        values = _attempt(errors, channel + "-hostcopy",
            lambda channel=channel: observer.copy_values(channel))
        if values is not None:
            _attempt(errors, filename, lambda values=values, filename=filename: np.save(
                out / filename, values, allow_pickle=False))
    return metadata


def save_motor_csr(path, network, motor_ids):
    """Persist actual native CSR rows, including signed weight values."""
    from tools.probe_c_banc_m3_core_rhythm import source_matrix

    full = source_matrix(network)
    rows = full[np.asarray(motor_ids, np.int64), :].tocsr()
    if rows.shape != (391, SHAPE["neurons"]):
        raise ValueError("E208 native motor CSR dimensions changed")
    np.savez_compressed(path, data=rows.data.copy(),
        indices=rows.indices.copy(), indptr=rows.indptr.copy(),
        shape=np.asarray(rows.shape, np.int64),
        motor_ids=np.asarray(motor_ids, np.int64).copy())


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
              dtype=adaptation.dtype), "E208 original adaptation")


def stage_rows_for_controls(control_rows):
    """Endpoint row t covers exactly its preceding 80 RK stages."""
    ticks = np.asarray(control_rows)
    if (ticks.ndim != 1 or ticks.dtype.kind not in "iu"
            or len(ticks) != 1824 or len(np.unique(ticks)) != len(ticks)
            or np.any((ticks < 1) | (ticks > SHAPE["controls"]))):
        raise ValueError("E208 exact 1824 late target controls required")
    return ((ticks[:, None] - 1) * 80 + np.arange(80)[None, :]).reshape(-1)


def balance_block(total, pos, neg, drive):
    """Return maximum actual signed-sum error and bound fraction."""
    total, pos, neg, drive = (np.asarray(value, np.float64) for value in
        (total, pos, neg, drive))
    residual = np.abs(total - (drive + pos - neg))
    bound = 1e-5 + 64 * np.finfo(np.float32).eps * (np.abs(drive) + pos + neg)
    if np.any(residual > bound):
        raise ValueError("E208 signed channel balance exceeds roundoff bound")
    return float(np.max(residual)), float(np.max(residual / bound))


def channel_balance(out, paths):
    """Validate complete channel raw, exact E201 current, and roundoff bound."""
    arrays = {name: np.load(Path(out) / filename, mmap_mode="r", allow_pickle=False)
              for name, filename in CHANNEL_FILES.items()}
    old = np.load(paths["stage_current"], mmap_mode="r", allow_pickle=False)
    for name, array in arrays.items():
        if array.shape != (160000, 391) or array.dtype != np.float32:
            raise ValueError("E208 malformed channel: " + name)
    if old.shape != arrays["total_current"].shape or old.dtype != np.float32:
        raise ValueError("E201 original current dimensions changed")
    max_abs = 0.
    max_fraction = 0.
    for lo in range(0, SHAPE["stages"], 2048):
        hi = min(lo + 2048, SHAPE["stages"])
        blocks = {name: np.asarray(array[lo:hi]) for name, array in arrays.items()}
        if any(not np.isfinite(value).all() for value in blocks.values()):
            raise ValueError("E208 nonfinite observed channel")
        if (np.any(blocks["positive_recurrent"] < 0)
                or np.any(blocks["negative_magnitude"] < 0)):
            raise ValueError("E208 signed magnitude channel negative")
        if blocks["total_current"].tobytes() != np.asarray(old[lo:hi]).tobytes():
            raise ValueError("E208 total current differs from E201 bytes")
        residual, fraction = balance_block(*(blocks[key] for key in CHANNEL_FILES))
        max_abs = max(max_abs, residual)
        max_fraction = max(max_fraction, fraction)
    return arrays, {"max_abs_residual": max_abs,
                    "max_bound_fraction": max_fraction}


def late_channel_metrics(arrays, targets_path, review_path, motor_ids):
    """Compute H208 only on frozen 25 cells and exact target-stage windows."""
    with np.load(targets_path, allow_pickle=False) as target:
        stages = stage_rows_for_controls(target["target_indices"])
    prior = json.loads(Path(review_path).read_text(encoding="utf-8"))
    columns = np.asarray(prior["family_closed_columns"], np.int64)
    ids = np.asarray(prior["family_closed_motor_ids"], np.int64)
    if (columns.shape != (25,) or ids.shape != (25,)
            or len(np.unique(columns)) != 25
            or not np.array_equal(np.asarray(motor_ids)[columns], ids)):
        raise ValueError("E208 frozen 25 closed MN roster changed")
    stats = {}
    n = len(stages)
    for name, array in arrays.items():
        sums = np.zeros(25, np.float64)
        squares = np.zeros(25, np.float64)
        minima = np.full(25, np.inf)
        maxima = np.full(25, -np.inf)
        for lo in range(0, n, 2048):
            block = np.asarray(array[stages[lo:lo + 2048, None], columns], np.float64)
            sums += np.sum(block, axis=0)
            squares += np.sum(block * block, axis=0)
            minima = np.minimum(minima, np.min(block, axis=0))
            maxima = np.maximum(maxima, np.max(block, axis=0))
        mean = sums / n
        rms = np.sqrt(squares / n)
        stats[name] = {"mean": mean.tolist(), "rms": rms.tolist(),
            "std": np.sqrt(np.maximum(0, squares / n - mean * mean)).tolist(),
            "minimum": minima.tolist(), "maximum": maxima.tolist()}
    positive = np.asarray(stats["positive_recurrent"]["rms"])
    negative = np.asarray(stats["negative_magnitude"]["rms"])
    support = (positive >= .001) & (negative > positive)
    return {"control_rows": 1824, "stage_rows": n,
        "motor_columns": columns.tolist(), "motor_ids": ids.tolist(),
        "channels": stats, "support_per_cell": support.tolist(),
        "support_count": int(np.count_nonzero(support)),
        "H208_supported": bool(np.count_nonzero(support) >= 13)}


def validate_raw(freeze, paths, out, rows, records, metadata, snapshots):
    """Evaluate only after raw files are stored; no permissive comparisons."""
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    n = SHAPE["samples"]
    if len(rows) != n or len(records) != n:
        raise ValueError("Incomplete E208 2001 original boundaries")
    for label, source, observed in (
        ("start", paths["start"], snapshots["start"]),
        ("main-1p1", paths["future"], snapshots["future"]),
        ("final", paths["final"], snapshots["final"]),
    ):
        exact(StateStore.load(source), observed, "E208 exact " + label)
    with paths["trace"].open(encoding="utf-8") as stream:
        original_trace = json.load(stream)
    exact(original_trace["samples"], rows, "E208 full original trace")
    with np.load(paths["native"], allow_pickle=False) as original:
        for key in ("control_tick", "network_tick", "physics_time_s",
                    "motor_rate", "activation", "bank"):
            exact(original[key], np.asarray([r[key] for r in records],
                dtype=original[key].dtype), "E208 original native " + key)
        for key in ("complete", "motor_rate_available", "activation_available",
                    "bank_available"):
            exact(original[key], np.ones(n, bool), "E208 original native " + key)
    compare_signal_adaptation(paths["signals"],
        np.asarray(metadata["motor_ids"], np.int64), records)
    if (metadata.get("start_tick") != 0 or metadata.get("end_tick") != 40000
            or metadata.get("capacity_stages") != 160000):
        raise ValueError("E208 native stage clock/capacity changed")
    counts = np.asarray(metadata["counter_requested"])
    if (counts.shape != (391,) or not np.array_equal(counts,
            np.full(391, 160000, dtype=np.int64))
            or np.asarray(metadata["overflow"]).tolist() != [0]
            or np.asarray(metadata["armed"]).tolist() != [0]):
        raise ValueError("E208 native stage count/overflow/armed mismatch")
    stages = np.load(out / "motor-total-current.npy", mmap_mode="r", allow_pickle=False)
    if (stages.shape != (160000, 391) or stages.dtype != np.float32
            or not np.isfinite(stages).all()):
        raise ValueError("E208 native stage array shape/dtype/finite mismatch")
    with (np.load(out / "motor-parameters.npz", allow_pickle=False) as current,
          np.load(paths["parameters"], allow_pickle=False) as original):
        if set(current.files) != set(original.files):
            raise ValueError("E208 E201 motor parameter keys changed")
        for key in current.files:
            exact(current[key], original[key], "E208 E201 parameter " + key)
    arrays, balance = channel_balance(out, paths)
    late = late_channel_metrics(arrays, paths["targets"],
        paths["late_input_review"], np.asarray(metadata["motor_ids"], np.int64))
    save_json(out / "late-signed-input.json", late)
    return {"snapshot_start_exact": True, "snapshot_future_exact": True,
            "snapshot_final_exact": True, "trace_2001_exact": True,
            "native_rate_activation_bank_exact": True,
            "native_adaptation_exact": True,
            "current_160000x391_finite": True,
            "e201_current_exact": True, "e201_parameters_exact": True,
            "signed_channel_balance": balance, "H208": {
                "support_count": late["support_count"],
                "supported": late["H208_supported"]}}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"output-manifest.json", "result.json"}:
            relative = path.relative_to(out).as_posix()
            files[relative] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e208.output-manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def run(freeze_path, freeze_sha, review_path, review_sha, out):
    began = time.perf_counter()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("One-shot E208 output exists: " + str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    freeze = review = session = observer = motor_ids = None
    rows, records = [], []
    snapshots = {}
    metadata = None
    ledger = AdvanceLedger()
    original_requested = False
    setup = {"models_constructed": 0, "models_completed_lower_bound": 0,
             "model_setup_completion": "not_started",
             "warmup_native_ticks": 0, "capture_native_ticks": 0,
             "warmup_completed_lower_bound": 0,
             "capture_completed_lower_bound": 0,
             "actual_main_neural_ticks": None, "actual_main_physics_steps": None,
             "attempted_controls": 0, "synchronized_controls": 0,
             "recorded_control_endpoints": 0}
    result = {"schema": "flylab.m3.e208.signed-input-channels.result.v1",
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
            raise OSError("E208 requires 12 GiB free disk")
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
        from tools.banc_m3_signed_input_channels import (
            replace_with_signed_input_network,
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
        original_requested = True
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
        exact(start, session.snapshot(), "E208 restored E150 whole start")
        if session.network.device != "cuda" or session.network.cuda_implementation != "packed":
            raise ValueError("Original E150 packed CUDA required")
        setup["model_setup_completion"] = "observer_requested_completion_unknown"
        setup["models_constructed"] = None
        setup["warmup_native_ticks"] = None
        setup["capture_native_ticks"] = None
        observer = replace_with_signed_input_network(session)
        setup["models_constructed"] = 7
        setup["models_completed_lower_bound"] = 7
        setup["warmup_native_ticks"] = 14 * observer.interval_steps
        setup["capture_native_ticks"] = 7 * observer.interval_steps
        setup["warmup_completed_lower_bound"] = setup["warmup_native_ticks"]
        setup["capture_completed_lower_bound"] = setup["capture_native_ticks"]
        setup["model_setup_completion"] = "observer_complete"
        exact(start, session.snapshot(), "E208 observed E150 whole start")
        motor_ids = observer.observation_ids.copy()
        save_motor_parameters(out / "motor-parameters.npz", session, motor_ids, paths)
        save_motor_csr(out / "motor-csr.npz", observer, motor_ids)
        snapshots["start"] = session.snapshot()
        StateStore.save(out / "observed-start", snapshots["start"])
        observer.arm_observation()
        rows.append(capture_control(session))
        records.append(endpoint(session, motor_ids))
        for control in range(1, SHAPE["controls"] + 1):
            if elapsed() >= RESOURCE["model_s"]:
                raise TimeoutError("E208 model collection deadline reached")
            ledger.advance(session)
            if (session.control_tick != control or observer.tick != 20 * control
                    or abs(session.body.physics_time() - .005 * control) > 1e-8):
                raise ValueError("E208 original neural/body/control clocks diverged")
            rows.append(capture_control(session))
            records.append(endpoint(session, motor_ids))
            if control == CLOCK["future_control"]:
                snapshots["future"] = session.snapshot()
                StateStore.save(out / "observed-main-1p1", snapshots["future"])
            if session.fault or session.body.fault:
                raise RuntimeError("E208 original native body health fault")
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
            out / "advance-ledger.json", {"schema": "flylab.m3.e208.ledger.v1",
                "events": ledger.events, "totals": totals,
                "setup": setup, "process_wall_s": elapsed()}))
        _attempt(errors, "trace.json", lambda: save_json(out / "trace.json",
            {"schema": "flylab.m3.e208.trace.v1", "samples": rows}))
        if motor_ids is not None:
            _attempt(errors, "endpoints.npz", lambda: save_endpoints(
                out / "endpoints.npz", records, motor_ids))
        if session is not None:
            _attempt(errors, "session-close", session.close)

    if primary is None and errors:
        primary = RuntimeError("E208 raw artifact or cleanup failed")
    if primary is None:
        _attempt(errors, "evaluate", lambda: result.update(
            comparisons=validate_raw(freeze, paths, out, rows,
                records, metadata, snapshots)))
        if errors:
            primary = RuntimeError("E208 exact baseline evaluation failed")
    if primary is None:
        result["validity"] = "valid_exact_signed_input_dataset"
        result["H208"] = result["comparisons"]["H208"]
    else:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                    "message": str(primary)}
    result.update(process_wall_s=elapsed(), setup=setup,
        raw_errors=errors, prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=((freeze or {}).get("prior_cumulative_s", 0)
                                   + elapsed()),
        original_attempts=int(original_requested), retries=0,
        signed_channel_model_tested=False, F1_F8_complete=False)
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
        result["resource_failure"] = "E208 internal 135 s exceeded"
        _attempt(errors, "result.json-deadline", lambda: save_json(out / "result.json", result))
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if not errors:
        result["finalization_pending"] = False
        _attempt(errors, "result.json-final", lambda: save_json(
            out / "result.json", result))
    if errors and primary is None:
        result["validity"] = "incomplete_or_invalid"
        primary = RuntimeError("E208 final raw/result persistence failed")
    if errors:
        _attempt(errors, "publication-failure.json", lambda: save_json(
            out / "publication-failure.json", errors))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha", "--freeze-sha256", dest="freeze_sha", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha", "--review-sha256", dest="review_sha", required=True)
    parser.add_argument("--output", "--out", dest="output", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.review,
        args.review_sha, args.output)


if __name__ == "__main__":
    main()

