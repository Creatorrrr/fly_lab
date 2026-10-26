"""E248: bounded, body-free response of the unchanged full BANC network.

The intervention changes only held sensory drive and selected presynaptic
output-mask entries.  It is a local neural contrast, not a gait controller.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SCHEMA = "flylab.m3.e248.lh-neural-response.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e248.lh-neural-response.review.v1"
FROZEN_SHA256 = "5406d62911ed1d462cb48cc0a9c337d6c6bfd5c72a77c74dafa10bf8337e24dd"
VALIDITY = "valid_local_neural_response"
ARM_NAMES = (
    "intact_low", "intact_high", "bridge_cut_low", "bridge_cut_high",
    "sensory_cut_low", "sensory_cut_high",
)
REFERENCE = "disk_intact_low"
N = 158706
CONTROLS = 40
STEPS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def gate(freeze_path: Path, freeze_hash: str, review_path: Path,
         review_hash: str) -> dict:
    """Check every pinned byte and package before any original array load."""
    if (freeze_hash != FROZEN_SHA256 or sha256(freeze_path) != freeze_hash
            or sha256(review_path) != review_hash):
        raise ValueError("Freeze or source-review SHA256 differs")
    freeze, review = read_json(freeze_path), read_json(review_path)
    if freeze.get("schema") != FREEZE_SCHEMA or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("E248 freeze/review schema differs")
    if (review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_hash):
        raise ValueError("E248 source review did not approve this freeze")
    owned = freeze["owned_files"]
    if set(review.get("source_sha256", {})) != set(owned):
        raise ValueError("Owned source review set differs")
    for name, digest in review["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Owned source SHA256 differs: " + name)
    pins = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"],
                  freeze["preservation_sha256"],
                  review.get("additional_input_sha256", {}),
                  review.get("additional_reused_source_sha256", {}),
                  {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting frozen SHA256 for " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Frozen input/source SHA256 differs: " + name)
    for name, expected in freeze["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Pinned package version differs: " + name)
    if (freeze["n"] != N or freeze["initial_tick"] != 40000
            or freeze["declared_edges"] != 11584852
            or freeze["arms"] != list(ARM_NAMES)):
        raise ValueError("Frozen neural-response contract differs")
    return freeze


def select_mask(baseline_mask: np.ndarray, source_ids: np.ndarray,
                bridge_ids: np.ndarray, cut: str) -> np.ndarray:
    """Preserve every pre-existing zero when installing an outgoing cut."""
    mask = np.asarray(baseline_mask)
    if mask.ndim != 1 or mask.dtype != np.float32 or not np.isin(mask, (0, 1)).all():
        raise ValueError("Binary native float32 output mask required")
    if cut not in ("intact", "bridge", "sensory"):
        raise ValueError("Unknown outgoing cut")
    result = mask.copy()
    ids = source_ids if cut == "sensory" else bridge_ids if cut == "bridge" else ()
    rows = np.asarray(ids)
    if rows.size and (rows.ndim != 1 or rows.dtype.kind not in "iu"
                      or len(np.unique(rows)) != len(rows)
                      or np.any(rows < 0) or np.any(rows >= len(mask))):
        raise ValueError("Invalid cut neuron rows")
    result[rows.astype(np.int64)] = np.float32(0)
    return result


def held_drive(base: np.ndarray, source_ids: np.ndarray, value: float) -> np.ndarray:
    drive = np.asarray(base)
    rows = np.asarray(source_ids)
    if (drive.ndim != 1 or drive.dtype != np.float32
            or not np.isfinite(drive).all() or rows.ndim != 1
            or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0) or np.any(rows >= len(drive))):
        raise ValueError("Native held drive and ordered source rows required")
    result = drive.copy()
    result[rows] = np.float32(value)
    if not np.isfinite(result).all():
        raise ValueError("Float32 sensory input overflow")
    return result


def response_metrics(rates: dict[str, np.ndarray], motor_rows: np.ndarray,
                     *, cap: float = 200., response_min: float = 1e-4,
                     interaction_min: float = 1e-4,
                     fraction_min: float = .2) -> dict:
    """Compute the preregistered 40-control LH contrast, retaining signs."""
    rows = np.asarray(motor_rows)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("Unique nonempty LH motor rows required")
    for name in ARM_NAMES:
        a = np.asarray(rates[name])
        if (a.ndim != 2 or a.shape[0] != CONTROLS + 1
                or a.dtype != np.float32 or not np.isfinite(a).all()
                or np.any(rows < 0) or np.any(rows >= a.shape[1])):
            raise ValueError("Complete finite full-rate arm required: " + name)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Positive motor cap required")
    intact = (rates["intact_high"][1:, rows].astype(np.float64)
              - rates["intact_low"][1:, rows].astype(np.float64)) / cap
    bridge = (rates["bridge_cut_high"][1:, rows].astype(np.float64)
              - rates["bridge_cut_low"][1:, rows].astype(np.float64)) / cap
    interaction = intact - bridge
    r = float(np.sqrt(np.mean(intact ** 2)))
    d = float(np.sqrt(np.mean(interaction ** 2)))
    ratio = d / r if r > 0 else None
    response = bool(r >= response_min)
    dependent = bool(d >= interaction_min and d >= fraction_min * r)
    return {
        "response_R": r, "interaction_D": d,
        "interaction_over_response": ratio,
        "response_present": response,
        "interaction_present": bool(d >= interaction_min),
        "fraction_pass": bool(d >= fraction_min * r),
        "H248_supported": bool(response and dependent),
        "lh_motor_rows": rows.astype(np.int64).tolist(),
        "delta_intact_min": float(np.min(intact)),
        "delta_intact_max": float(np.max(intact)),
        "delta_bridge_min": float(np.min(bridge)),
        "delta_bridge_max": float(np.max(bridge)),
        "interaction_min": float(np.min(interaction)),
        "interaction_max": float(np.max(interaction)),
    }


def cut_validation(low: np.ndarray, high: np.ndarray,
                   source_ids: np.ndarray) -> dict:
    """A cut must suppress every downstream difference, not only MN output."""
    a, b = np.asarray(low), np.asarray(high)
    rows = np.asarray(source_ids, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 2 or a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("Aligned full native rate arrays required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite sensory-cut states required")
    non_source = np.ones(a.shape[1], dtype=bool)
    non_source[rows] = False
    exact = bool(a[:, non_source].tobytes() == b[:, non_source].tobytes())
    delta = b[:, rows].astype(np.float64) - a[:, rows].astype(np.float64)
    return {
        "non_source_bitexact": exact,
        "source_difference_max_abs": float(np.max(np.abs(delta))),
        "source_difference_rms": float(np.sqrt(np.mean(delta ** 2))),
        "source_differing_cell_time_samples": int(np.count_nonzero(
            a[:, rows].view(np.uint32) != b[:, rows].view(np.uint32))),
    }


def new_ledger() -> dict:
    return {
        "schema": "flylab.m3.e248.lh-neural-response.ledger.v1",
        "constructors": [], "arms": {}, "setup_ticks_completed": 0,
        "science_ticks_completed": 0, "science_ticks_requested": 0,
        "body_constructors": 0, "physics_steps": 0, "fit_calls": 0,
        "solver_calls": 0, "retry_count": 0,
    }


def begin_operation(ledger: dict, key: str, *, controls: int = CONTROLS) -> dict:
    if key in ledger["arms"]:
        raise ValueError("An arm may be entered only once")
    entry = {
        "requested": True, "entered": False, "completed": False,
        "requested_controls": controls, "entered_controls": 0,
        "completed_controls": 0, "unknown_partial": False,
        "initial_restored_exact": False,
    }
    ledger["arms"][key] = entry
    ledger["science_ticks_requested"] += controls * STEPS
    return entry


def run_arm(network, saved: dict, arm: str, out: Path, ledger: dict,
            source_ids: np.ndarray, bridge_ids: np.ndarray, low: float,
            high: float, started: float, persist=write_json) -> dict:
    """Run exactly one requested arm; all attempted work is durably journaled."""
    if arm not in (*ARM_NAMES, REFERENCE):
        raise ValueError("Unregistered arm")
    entry = begin_operation(ledger, arm)
    persist(out.parent / "advance-ledger.json", ledger)
    entry["entered"] = True
    persist(out.parent / "advance-ledger.json", ledger)
    network.restore(saved)
    from tools.verify_c_banc_observation import exact
    restored = network.snapshot()
    exact(restored, saved)
    entry["initial_restored_exact"] = True
    entry["initial_snapshot_sha256"] = hashlib.sha256(
        b"".join(np.ascontiguousarray(restored[k]).tobytes()
                 for k in ("rate", "adaptation", "drive", "output_mask"))).hexdigest()
    cut = "sensory" if arm.startswith("sensory") else "bridge" if arm.startswith("bridge") else "intact"
    value = high if arm.endswith("high") else low
    desired_mask = select_mask(saved["output_mask"], source_ids, bridge_ids, cut)
    desired_drive = held_drive(saved["drive"], source_ids, value)
    prior_zeros = np.flatnonzero(saved["output_mask"] == 0)
    new_zeros = source_ids if cut == "sensory" else bridge_ids if cut == "bridge" else np.array([], np.int64)
    network.set_muted(np.unique(np.concatenate((prior_zeros, new_zeros))))
    if not same_bits(network.output_mask.detach().cpu().numpy(), desired_mask):
        raise AssertionError("Actual outgoing mask differs")
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "drive.npy", desired_drive, allow_pickle=False)
    np.save(out / "mask.npy", desired_mask, allow_pickle=False)
    raw = np.lib.format.open_memmap(out / "rate.npy", mode="w+", dtype=np.float32,
                                    shape=(CONTROLS + 1, len(saved["rate"])))
    raw[:] = np.nan
    raw[0] = network.readout()
    raw.flush()
    if not same_bits(raw[0], saved["rate"]):
        raise AssertionError("Initial restored full rate differs")
    persist(out.parent / "advance-ledger.json", ledger)
    try:
        for control in range(1, CONTROLS + 1):
            if time.perf_counter() - started >= 38:
                raise TimeoutError("E248 scientific deadline reached")
            entry["entered_controls"] = control
            entry["unknown_partial"] = True
            persist(out.parent / "advance-ledger.json", ledger)
            network.advance(desired_drive, STEPS)
            raw[control] = network.readout()
            raw.flush()
            entry["completed_controls"] = control
            entry["unknown_partial"] = False
            ledger["science_ticks_completed"] += STEPS
            persist(out.parent / "advance-ledger.json", ledger)
        state = network.snapshot()
        if state["tick"] != saved["tick"] + CONTROLS * STEPS:
            raise AssertionError("Final neural clock differs")
        if (not same_bits(state["drive"], desired_drive)
                or not same_bits(state["output_mask"], desired_mask)
                or not np.isfinite(state["adaptation"]).all()
                or np.any(state["adaptation"] < 0)):
            raise AssertionError("Final drive/mask/adaptation differs or is invalid")
        from flylab.c.storage import StateStore
        StateStore.save(out / "final", state)
        entry["completed"] = True
        persist(out.parent / "advance-ledger.json", ledger)
        return state
    finally:
        raw.flush()
        del raw


def _check_space(freeze: dict, out: Path) -> None:
    import psutil
    import torch
    limits = freeze["resource_limits"]
    if psutil.disk_usage(str(out.parent)).free < limits["minimum_free_bytes"]:
        raise RuntimeError("E248 disk free space below preregistered minimum")
    if torch.cuda.mem_get_info()[0] < limits["minimum_gpu_free_bytes"]:
        raise RuntimeError("E248 CUDA free bytes below preregistered minimum")


def _configure_cpu() -> dict:
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    info = threadpool_info()
    if torch.get_num_threads() != 1 or any(item["num_threads"] != 1 for item in info):
        raise RuntimeError("E248 effective CPU worker count is not one")
    return {"torch_num_threads": torch.get_num_threads(),
            "threadpools": [{"internal_api": item["internal_api"],
                             "num_threads": item["num_threads"]} for item in info]}


def actual_backend(network) -> dict:
    """A model hash alone does not specify CUDA kernel or capture length."""
    receipt = {"device": getattr(network, "device", None),
               "cuda_implementation": getattr(network, "cuda_implementation", None),
               "interval_steps": getattr(network, "interval_steps", None),
               "captured_graph": getattr(network, "graph", None) is not None,
               "n": getattr(network, "n", None)}
    if receipt != {"device": "cuda", "cuda_implementation": "packed",
                   "interval_steps": STEPS, "captured_graph": True, "n": N}:
        raise ValueError("Actual E150 CUDA packed graph/capture contract differs")
    return receipt


def _check_live_resources(freeze: dict, started: float, out: Path) -> dict:
    import psutil
    import torch

    elapsed = time.perf_counter() - started
    limits = freeze["resource_limits"]
    receipt = {
        "elapsed_s": elapsed,
        "rss_bytes": psutil.Process().memory_info().rss,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "free_disk_bytes": psutil.disk_usage(str(out)).free,
        "free_cuda_bytes": torch.cuda.mem_get_info()[0],
    }
    if (elapsed >= limits["internal_s"] - limits["finalization_reserve_s"]
            or receipt["rss_bytes"] > limits["resident_bytes"]
            or receipt["cuda_peak_allocated_bytes"] > limits["additional_gpu_bytes"]
            or receipt["free_disk_bytes"] < limits["minimum_free_bytes"]):
        raise TimeoutError("E248 resource or scientific deadline reached")
    return receipt


def _load_inputs(freeze: dict) -> tuple:
    """Called only after gate; load exact frozen arrays, never construct a body."""
    from scipy.sparse import csr_matrix

    from flylab.c.storage import StateStore
    from tools.banc_m3_intrinsic_recruitment import patch_tau

    path = lambda key: ROOT / freeze["paths"][key]
    with np.load(path("native_csr"), allow_pickle=False) as file:
        indptr, indices, data = (file[k].copy() for k in ("indptr", "indices", "data"))
    if (data.dtype != np.float32 or len(data) != freeze["declared_edges"]
            or indptr.shape != (N + 1,) or indices.shape != data.shape
            or int(indptr[-1]) != len(data)):
        raise ValueError("Exact native CSR layout required")
    weights = csr_matrix((data, indices, indptr), shape=(N, N), dtype=np.float32)
    weights.check_format(full_check=True)
    with np.load(path("configuration"), allow_pickle=False) as file:
        config = {name: file[name].copy() for name in file.files}
    with np.load(path("calibration"), allow_pickle=False) as file:
        calibration = {name: file[name].copy() for name in file.files}
    tau = patch_tau(np.full(N, .02, dtype=np.float32),
                    calibration["tau_ids"], calibration["tau_values"])
    if not same_bits(tau, config["membrane_tau"]):
        raise ValueError("Actual E150 membrane tau differs")
    saved = StateStore.load(path("checkpoint"))["network"]
    if (saved["model_hash"] != freeze["network_identity"]
            or saved["tick"] != freeze["initial_tick"]
            or saved["device"] != "cuda" or saved["dt"] != .00025
            or saved["adaptation_gain"] != 8.
            or saved["adaptation_tau_s"] != .3):
        raise ValueError("Frozen neural checkpoint identity or clock differs")
    identities = read_json(path("identities"))
    if identities["network_identity"] != freeze["network_identity"]:
        raise ValueError("E150 network identity differs")
    groups = read_json(path("groups"))["groups"]
    with np.load(path("parameters"), allow_pickle=False) as file:
        motor_ids = file["motor_ids"].copy()
        free_columns = file["free_columns"].copy()
    lh_columns = sorted({int(column) for group in groups if group["leg"] == "lh"
                         for column in group["columns"] if column in free_columns})
    lh_rows = motor_ids[lh_columns]
    if (len(lh_rows) != 56 or len(np.unique(lh_rows)) != 56
            or not np.array_equal(motor_ids, config["motor_ids"])):
        raise ValueError("Original 56 free LH motor roster differs")
    selection = read_json(path("selection"))["selected"]
    if selection != freeze["selection"]:
        raise ValueError("Frozen E247 structural selection differs")
    return weights, tau, config, saved, lh_rows


def _construct(weights, tau, config, saved, freeze):
    from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
    from tools.verify_c_banc_observation import exact

    ones = np.ones(N, np.float32)
    network = MotorTimescaleNetwork(weights, tau, ones,
        np.full(N, 7.5, np.float32), np.full(N, 200., np.float32),
        adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
        device="cuda", cuda_implementation="packed", capture_steps=20,
        core_ids=saved["core_adaptation"]["ids"],
        core_gain=saved["core_adaptation"]["gain"],
        motor_ids=saved["motor_adaptation"]["ids"],
        motor_gains=saved["motor_adaptation"]["gains"],
        motor_log_tau=saved["motor_timescale"]["log_tau"])
    actual_backend(network)
    if network.identity != freeze["network_identity"]:
        raise AssertionError("Constructed full BANC model identity differs")
    network.restore(saved)
    exact(network.snapshot(), saved)
    if (not same_bits(network._cell_gain.detach().cpu().numpy(), config["cell_gain"])
            or not same_bits(network.tau.detach().cpu().numpy(), config["membrane_tau"])
            or not same_bits(network._cell_adaptation_tau.detach().cpu().numpy(),
                             saved["motor_timescale"]["tau32"])
            or not same_bits(network._cell_inverse_tau.detach().cpu().numpy(),
                             saved["motor_timescale"]["inverse32"])):
        raise AssertionError("Native parameter tensor differs")
    return network


def _manifest(out: Path, limit: int) -> dict:
    files = {}
    total = 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise RuntimeError("E248 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {"sha256": sha256(path), "bytes": size}
    return {"schema": "flylab.m3.e248.output-manifest.v1", "bytes": total,
            "files": files}


def finish_receipts(out: Path, freeze: dict, ledger: dict, result: dict,
                    errors: list, primary: Exception | None, started: float,
                    persist=write_json) -> Exception | None:
    """Settle mutable receipts before hashing them; revoke support on any fault."""
    limit_s = freeze["resource_limits"]["internal_s"]
    result["work"] = {
        "constructors_requested": len(ledger["constructors"]),
        "constructors_completed": sum(bool(x["completed"]) for x in ledger["constructors"]),
        "setup_ticks_completed": ledger["setup_ticks_completed"],
        "science_ticks_requested": ledger["science_ticks_requested"],
        "science_ticks_completed": ledger["science_ticks_completed"],
        "arms_requested": len(ledger["arms"]),
        "arms_completed": sum(bool(x["completed"]) for x in ledger["arms"].values()),
        "arms_unknown_partial": [name for name, entry in ledger["arms"].items()
                                 if entry["unknown_partial"]],
    }

    def failed(error: Exception, label: str) -> None:
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"type": type(error).__name__, "message": f"{label}: {error}"})
        result["validity"] = ("resource_limited" if isinstance(primary, TimeoutError)
                              else "invalid_or_incomplete")
        result["H248_supported"] = False
        result["future_exact"] = False
        result["finalization_pending"] = True
        result["first_error"] = {"type": type(primary).__name__, "message": str(primary)}

    def revoke_metrics() -> None:
        path = out / "metrics.json"
        if not path.exists():
            return
        metrics = read_json(path)
        metrics.setdefault("H248_measured_condition_met", metrics["H248_supported"])
        metrics["H248_supported"] = False
        metrics["interpretation_status"] = "not_tested_invalid_or_incomplete"
        persist(path, metrics)

    for label, action in (
        ("advance-ledger.json", lambda: persist(out / "advance-ledger.json", ledger)),
        ("errors.json", lambda: persist(out / "errors.json", {"errors": errors})),
    ):
        try:
            action()
        except Exception as error:  # noqa: BLE001 - receipt failure is measured
            failed(error, label)
    result["elapsed_s"] = time.perf_counter() - started
    if result["elapsed_s"] >= limit_s:
        failed(TimeoutError("E248 internal deadline including finalization"), "deadline")
    if primary is not None:
        try:
            revoke_metrics()
        except Exception as error:  # noqa: BLE001 - keep original first
            failed(error, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)

    # A failed final-result write may follow a successful manifest. On that
    # path rewrite metrics/errors first, then rebuild or remove the manifest.
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                _manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - started
        if result["elapsed_s"] >= limit_s:
            raise TimeoutError("E248 publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - started >= limit_s:
            raise TimeoutError("E248 result publication passed internal deadline")
    except Exception as error:  # noqa: BLE001 - must fail closed after any write
        failed(error, "publication")
        try:
            revoke_metrics()
        except Exception as secondary:  # noqa: BLE001 - retain first error
            failed(secondary, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)
        try:
            persist(out / "errors.json", {"errors": errors})
        except Exception as secondary:  # noqa: BLE001 - retain first error
            failed(secondary, "errors receipt")
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            persist(out / "result.json", result)
        except Exception as secondary:  # noqa: BLE001 - original result remains incomplete
            failed(secondary, "result retry")
            try:
                persist(out / "errors.json", {"errors": errors})
            except Exception as tertiary:  # noqa: BLE001 - last secondary retained in memory
                failed(tertiary, "errors retry")
        # Hash only the settled ledger, metrics and errors. The result is
        # excluded from the manifest to avoid a publication cycle.
        try:
            persist(out / "output-manifest.json",
                    _manifest(out, freeze["resource_limits"]["output_bytes"]))
        except Exception as secondary:  # noqa: BLE001 - never retain stale manifest
            failed(secondary, "manifest retry")
            (out / "output-manifest.json").unlink(missing_ok=True)
    return primary


def execute(freeze: dict, out: Path) -> None:
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    ledger = new_ledger()
    result = {"schema": "flylab.m3.e248.lh-neural-response.result.v1",
              "validity": "incomplete", "H248_supported": False,
              "candidate_adopted": False, "F6_exposed": False,
              "future_exact": False, "first_error": None,
              "finalization_pending": True,
              "original_attempts": 1}
    errors = []
    write_json(out / "advance-ledger.json", ledger)
    write_json(out / "result.json", result)
    network = None
    reference = None
    primary = None
    try:
        result["cpu_runtime"] = _configure_cpu()
        _check_space(freeze, out)
        _check_live_resources(freeze, start, out)
        weights, tau, config, saved, lh_rows = _load_inputs(freeze)
        selection = freeze["selection"]
        source_ids = np.asarray(selection["source_rows"], dtype=np.int64)
        bridge_ids = np.asarray(selection["bridge_rows"], dtype=np.int64)
        low = float(selection["train_steady_min"])
        high = float(selection["train_steady_max"])
        write_json(out / "selection.json", {"port": selection["port"],
            "source_rows": source_ids.tolist(), "bridge_rows": bridge_ids.tolist(),
            "lh_free_motor_rows": lh_rows.tolist(), "low32": float(np.float32(low)),
            "high32": float(np.float32(high)), "operation": "held neural drive"})
        write_json(out / "provenance.json", {"network_identity": freeze["network_identity"],
            "initial_tick": saved["tick"], "csr_edges": freeze["declared_edges"],
            "source_checkpoint": freeze["paths"]["checkpoint"],
            "future_scope": "new held-input future only", "body_constructors": 0})
        for index in range(2):
            label = "arm_network" if index == 0 else "disk_reference_network"
            request = {"name": label, "requested": True, "entered": False,
                       "completed": False, "setup_ticks_completed": None,
                       "unknown_partial": False}
            ledger["constructors"].append(request)
            write_json(out / "advance-ledger.json", ledger)
            if time.perf_counter() - start >= 38:
                raise TimeoutError("E248 constructor start deadline")
            request["entered"] = True
            request["unknown_partial"] = True
            write_json(out / "advance-ledger.json", ledger)
            candidate = _construct(weights, tau, config, saved, freeze)
            request["actual_backend"] = actual_backend(candidate)
            request["completed"] = True
            request["unknown_partial"] = False
            request["setup_ticks_completed"] = 60
            ledger["setup_ticks_completed"] += 60
            write_json(out / "advance-ledger.json", ledger)
            if index == 0:
                network = candidate
                StateStore.save(out / "neural-start", candidate.snapshot())
                exact(StateStore.load(out / "neural-start"), saved)
                for arm in ARM_NAMES:
                    _check_live_resources(freeze, start, out)
                    run_arm(network, saved, arm, out / arm, ledger,
                            source_ids, bridge_ids, low, high, start)
                network = None
                candidate = None
                gc.collect()
            else:
                reference = candidate
                disk = StateStore.load(out / "neural-start")
                exact(disk, saved)
                _check_live_resources(freeze, start, out)
                run_arm(reference, disk, REFERENCE, out / REFERENCE, ledger,
                        source_ids, bridge_ids, low, high, start)
        rates = {name: np.load(out / name / "rate.npy", mmap_mode="r")
                 for name in (*ARM_NAMES, REFERENCE)}
        metrics = response_metrics(rates, lh_rows,
            response_min=freeze["rate_response_min"],
            interaction_min=freeze["interaction_min"],
            fraction_min=freeze["interaction_fraction_min"])
        cut = cut_validation(rates["sensory_cut_low"],
                             rates["sensory_cut_high"], source_ids)
        disk_exact = same_bits(rates["intact_low"], rates[REFERENCE])
        exact(StateStore.load(out / "intact_low" / "final"),
              StateStore.load(out / REFERENCE / "final"))
        if not disk_exact or not cut["non_source_bitexact"]:
            raise AssertionError("Cut or independent disk future exactness failed")
        for name, values in rates.items():
            if (np.any(values < -1e-4)
                    or np.any(values > 200.001)):
                raise AssertionError("Full rate outside bounded physiology")
            final = StateStore.load(out / name / "final")
            if (not np.isfinite(final["adaptation"]).all()
                    or np.any(final["adaptation"] < 0)):
                raise AssertionError("Final adaptation invalid")
        write_json(out / "metrics.json", {**metrics, **cut,
            "disk_intact_low_full_rate_exact": disk_exact,
            "disk_intact_low_final_state_exact": True})
        result["resources"] = _check_live_resources(freeze, start, out)
        result["validity"] = VALIDITY
        result["H248_supported"] = metrics["H248_supported"]
        result["future_exact"] = True
    except Exception as error:  # noqa: BLE001 - preserve unknown original failure
        primary = error
        result["first_error"] = {"type": type(error).__name__, "message": str(error)}
        result["validity"] = "resource_limited" if isinstance(error, TimeoutError) else "invalid_or_incomplete"
        errors.append({**result["first_error"], "traceback": traceback.format_exc()})
    finally:
        primary = finish_receipts(out, freeze, ledger, result, errors, primary,
                                  start, persist=write_json)
        if primary is not None:
            raise RuntimeError("E248 invalid: " + str(primary)) from primary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out != (ROOT / BASE / "e248-lh-neural-response").resolve():
        raise ValueError("E248 fixed output directory required")
    freeze = gate(args.freeze.resolve(), args.freeze_sha256,
                  args.review.resolve(), args.review_sha256)
    execute(freeze, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
