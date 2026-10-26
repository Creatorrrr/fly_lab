"""E141: one reviewed native motor-timescale transfer diagnostic.

The CLI is the only path that may evaluate the frozen full BANC inputs.  Importing
this module only exposes validation, accounting, and raw-persistence helpers.
"""

# Broad catches preserve the first failure while independently saving raw data,
# receipts, cleanup faults, and secondary errors from the single original run.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from flylab.c.walking_trace import capture_control
from tools import fit_c_banc_m3_joint_adaptive_recruitment as joint_fit
from tools import verify_c_banc_m3_e121_port_interference as e121
from tools.analyze_c_banc_biological_gait import analyze, load_model
from tools.banc_m3_adaptation_timescale import CudaAdaptationTimescaleEngine
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial, patch_tau
from tools.banc_m3_motor_adaptation import MotorAdaptationTrial
from tools.banc_m3_motor_timescale import (
    LOG_TAU_LOWER,
    LOG_TAU_UPPER,
    MotorTimescaleTrial,
)
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_m3_e113_body import (
    AdvanceLedger,
    NativeHealthStop,
    body_quality,
    finite_error,
    install_native_observer,
    native_arrays,
    observation_ids,
    prepare_neutral_checkpoint,
    source_diagnostics,
    supported_slip,
)
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "8930030a5a690b2caf713acad9dc5b4a79f22257ffbdae6125aeb7919d8ecc6e"
DESIGN_SHA = "809f5245da5925e001f392ab14b55bd9ba07755906eafd7588df2f5afc61df47"
ADDENDUM_SHA = "b3913ff1690daabdd13f66095dae5927fb5ed364f16f57bf185a25f63123e7c5"
OWNED = (
    "tools/verify_c_banc_m3_native_timescale_transfer.py",
    "tests/test_c_banc_native_timescale_transfer.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
MUSCLE_COUNT = 96
PREPARE_LIMIT_S = 45.0
BODY_LIMIT_S = 180.0
INTERNAL_LIMIT_S = 260.0
RESERVE_S = 12.0
RAW_LIMIT_BYTES = 4 << 30
MINIMUM_FREE_BYTES = 12 << 30
PRIOR_CONSERVATIVE_S = 1981.0
PROSPECTIVE_LIMIT_S = 2271.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Replace a JSON receipt only after a complete finite serialization."""
    path = Path(path)
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                         sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _safe_path(relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Only frozen repository-relative paths are allowed")
    result = (ROOT / name).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen path escaped the repository")
    return result


def _exact_mapping_hashes(mapping, label):
    if not isinstance(mapping, dict):
        raise TypeError("Frozen " + label + " mapping required")
    for relative, expected in mapping.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid frozen hash for " + relative)
        if sha256(_safe_path(relative)) != expected:
            raise ValueError("Frozen " + label + " changed: " + relative)


def reviewed_inputs(freeze_path, freeze_sha256, source_review_path,
                    source_review_sha256):
    """Validate every frozen byte before graph, CUDA, neural, or body work."""
    if freeze_sha256 != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA:
        raise ValueError("Exact E141 v2 freeze SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    conditional = freeze.get("conditional", {})
    native = freeze.get("native", {})
    if (freeze.get("schema") != "flylab.m3.e141.native-transfer-freeze.v1"
            or freeze.get("revision") != 2
            or freeze.get("design_sha256") != DESIGN_SHA
            or sha256(_safe_path(freeze.get("design_path", ""))) != DESIGN_SHA
            or freeze.get("addendum_sha256") != ADDENDUM_SHA
            or sha256(_safe_path(freeze.get("addendum_path", ""))) != ADDENDUM_SHA
            or len(freeze.get("input_sha256", {})) != 213
            or len(freeze.get("reused_source_sha256", {})) != 79
            or len(freeze.get("preservation_sha256", {})) != 1647
            or tuple(freeze.get("owned_files", ())) != OWNED
            or freeze.get("original_conditional_accepted") is not False
            or freeze.get("original_conditional_failed_legs") != [2, 5]
            or conditional != {"batches": 1, "neurons": 391, "muscles": 96,
                "fit_edges": 1572, "contexts": 3, "ticks_per_context": 40000,
                "context_ticks": 120000, "model_s": 30,
                "decoder_endpoints": 6000, "derivative_banks": 0,
                "compare_fields": ["rate", "adaptation", "activation",
                                   "command", "tendon"],
                "atol": 1e-8, "rtol": 0}
            or native != {"neurons": 158706, "main_controls": 2000,
                "clone_controls": 20, "neural_steps_per_control": 20,
                "physics_steps_per_control": 50, "neural_dt_s": .00025,
                "physics_dt_s": .0001, "control_dt_s": .005,
                "actual_neural_s": 10.1, "actual_physical_s": 10.1,
                "neural_ticks": 40400, "physics_steps": 101000,
                "checkpoint_control": 200, "compare_control": 220,
                "development_window_s": [.5, 10.]}
            or freeze.get("startup_limits") != {"q_RMS": .018,
                "q_max_abs": .04, "LTM_RMS": .009, "LTM_max_abs": .025}
            or freeze.get("historical_E121_native_startup_limits") != {
                "q_RMS": .02, "q_max_abs": .05,
                "LTM_RMS": .01, "LTM_max_abs": .03}
            or freeze.get("prepare_wall_limit_s") != PREPARE_LIMIT_S
            or freeze.get("body_wall_limit_s") != BODY_LIMIT_S
            or freeze.get("internal_wall_limit_s") != INTERNAL_LIMIT_S
            or freeze.get("reserve_s") != RESERVE_S
            or freeze.get("maximum_raw_bytes") != RAW_LIMIT_BYTES
            or freeze.get("minimum_free_disk_bytes") != MINIMUM_FREE_BYTES
            or freeze.get("prior_conservative_process_s") != PRIOR_CONSERVATIVE_S
            or freeze.get("prospective_cumulative_limit_s") != PROSPECTIVE_LIMIT_S
            or freeze.get("original_processes") != 1
            or freeze.get("retries") != 0 or freeze.get("fit_calls") != 0
            or freeze.get("LP_calls") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("active_UI_changed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False
            or freeze.get("cpu_threads") != 1 or freeze.get("gpu_count") != 1):
        raise ValueError("Frozen E141 protocol or resource contract changed")
    if (not isinstance(source_review_sha256, str)
            or len(source_review_sha256) != 64
            or sha256(source_review_path) != source_review_sha256):
        raise ValueError("Exact E141 parent source-review SHA required")
    review = json.loads(Path(source_review_path).read_text(encoding="utf-8"))
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    if (review.get("accepted_for_single_original_native_transfer") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != expected_sources):
        raise ValueError("Accepted exact E141 parent source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        _exact_mapping_hashes(freeze[label], label)
    return freeze, review


def _same_array(left, right):
    return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def exact_arrays(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(label + " key set changed")
    for key in actual:
        if not _same_array(actual[key], expected[key]):
            raise ValueError(label + " changed: " + key)


def assemble_candidate(freeze, fixture, source_calibration, *, loader=load_npz,
                       path_resolver=_safe_path, file_hasher=sha256):
    """Rebuild all six fixed raws without using the parent prepared arrays."""
    edge_count = int(fixture["edge_ids"].size)
    motor_count = int(fixture["motor_ids"].size)
    muscle_count = int(fixture["pool"].shape[0])
    if (fixture["leg_edge_mask"].shape != (6, edge_count)
            or fixture["leg_motor_mask"].shape != (6, motor_count)
            or fixture["leg_muscle_mask"].shape != (6, muscle_count)):
        raise ValueError("Six-leg fixture masks required")
    delta = np.empty(edge_count, np.float64)
    gain = np.empty(muscle_count, np.float64)
    log_tau = np.empty(muscle_count, np.float64)
    expected = {key: np.empty((3, 2001, width), np.float64)
        for key, width in (("rate", motor_count), ("adaptation", motor_count),
                           ("activation", muscle_count), ("command", 42),
                           ("tendon", 6))}
    edge_filled = np.zeros(edge_count, bool)
    muscle_filled = np.zeros(muscle_count, bool)
    provenance = []
    selected = freeze.get("selected_legs", [])
    if [item.get("leg") for item in selected] != list(range(6)):
        raise ValueError("Frozen six-leg source order required")
    for item in selected:
        leg = int(item["leg"])
        source = item.get("source", {})
        path = path_resolver(source.get("raw_path", ""))
        if file_hasher(path) != source.get("raw_sha256"):
            raise ValueError("Selected leg raw changed")
        raw = loader(path)
        edges = np.flatnonzero(fixture["leg_edge_mask"][leg])
        motors = np.flatnonzero(fixture["leg_motor_mask"][leg])
        muscles = np.flatnonzero(fixture["leg_muscle_mask"][leg])
        if edge_filled[edges].any() or muscle_filled[muscles].any():
            raise ValueError("Overlapping six-leg parameter partition")
        checks = (("delta", edges.size, np.float64),
                  ("gain", muscles.size, np.float64),
                  ("log_tau", muscles.size, np.float64))
        for key, length, dtype in checks:
            value = raw.get(key)
            if (not isinstance(value, np.ndarray) or value.shape != (length,)
                    or value.dtype != dtype or not np.isfinite(value).all()):
                raise ValueError("Invalid selected leg parameter: " + key)
        identities = (("edge_ids", fixture["edge_ids"][edges]),
                      ("motor_indices", motors), ("muscle_indices", muscles))
        for key, answer in identities:
            if key not in raw or not np.array_equal(raw[key], answer):
                raise ValueError("Selected leg identity changed: " + key)
        for key, width, columns in (("rate", motor_count, motors),
                                    ("adaptation", motor_count, motors)):
            value = raw.get(key)
            if (not isinstance(value, np.ndarray)
                    or value.shape != (3, 2001, len(columns))
                    or value.dtype != np.float64 or not np.isfinite(value).all()):
                raise ValueError("Invalid selected response: " + key)
            expected[key][:, :, columns] = value
        activation = raw.get("activation")
        command = raw.get("command")
        tendon = raw.get("tendon")
        if (activation.shape != (3, 2001, len(muscles))
                or command.shape != (3, 2001, 42)
                or tendon.shape != (3, 2001, 1)
                or any(value.dtype != np.float64 or not np.isfinite(value).all()
                       for value in (activation, command, tendon))):
            raise ValueError("Invalid selected decoded response")
        delta[edges] = raw["delta"]
        gain[muscles] = raw["gain"]
        log_tau[muscles] = raw["log_tau"]
        expected["activation"][:, :, muscles] = activation
        expected["command"][:, :, 7 * leg:7 * leg + 7] = (
            command[:, :, 7 * leg:7 * leg + 7])
        expected["tendon"][:, :, leg] = tendon[:, :, 0]
        edge_filled[edges] = True
        muscle_filled[muscles] = True
        provenance.append({"leg": leg, "raw_path": str(path),
                           "raw_sha256": source["raw_sha256"],
                           "original_guard_pass": bool(item["original_guard_pass"]),
                           "ratio": float(item["ratio"])})
    if not edge_filled.all() or not muscle_filled.all():
        raise ValueError("Incomplete six-leg parameter coverage")
    bounds = np.asarray(fixture["delta_bounds"])
    if (bounds.shape != (edge_count, 2) or bounds.dtype != np.float64
            or np.any(delta < bounds[:, 0]) or np.any(delta > bounds[:, 1])
            or np.any((gain < 0) | (gain > 8))
            or np.any((log_tau < LOG_TAU_LOWER) | (log_tau > LOG_TAU_UPPER))):
        raise ValueError("Frozen delta/gain/log-tau bounds changed")
    if (source_calibration["values"].dtype != np.float32
            or source_calibration["source_values"].dtype != np.float32
            or source_calibration["individual_edge_ids"].shape != (edge_count,)
            or not np.array_equal(source_calibration["individual_edge_ids"],
                                  fixture["edge_ids"])):
        raise ValueError("E121 selected calibration identity changed")
    positions = {int(edge): index
                 for index, edge in enumerate(source_calibration["edge_ids"])}
    if len(positions) != len(source_calibration["edge_ids"]):
        raise ValueError("Duplicate E121 calibration edge IDs")
    try:
        mapped = np.asarray([positions[int(edge)] for edge in fixture["edge_ids"]],
                            dtype=np.int64)
    except KeyError as error:
        raise ValueError("Selected edge absent from E121 calibration") from error
    base = source_calibration["values"][mapped].astype(np.float64)
    if not np.array_equal(base, fixture["base_fit_weights"]):
        raise ValueError("E121 native selected base weights changed")
    target64 = base * np.exp(delta)
    target32 = target64.astype(np.float32)
    source = source_calibration["source_values"][mapped].astype(np.float64)
    efficacy = target64 / source
    if (not np.isfinite(target64).all() or not np.isfinite(target32).all()
            or not np.isfinite(efficacy).all()
            or np.any(source == 0)
            or not np.array_equal((source * efficacy).astype(np.float32), target32)):
        raise ValueError("Native float32 target/derived efficacy closure failed")
    changed = {key: value.copy() for key, value in source_calibration.items()}
    changed["values"][mapped] = target32
    changed["individual_edge_efficacies"] = efficacy.copy()
    other = np.ones(len(changed["values"]), bool)
    other[mapped] = False
    if (not np.array_equal(changed["source_values"],
                           source_calibration["source_values"])
            or not np.array_equal(changed["values"][other],
                                  source_calibration["values"][other])
            or not np.array_equal(np.signbit(changed["values"]),
                                  np.signbit(source_calibration["values"]))):
        raise ValueError("Native calibration changed a non-target value or sign")
    guard = np.asarray([item["original_guard_pass"] for item in selected], bool)
    parameters = {"delta": delta, "gain": gain, "log_tau": log_tau,
        "muscle_gains32": gain.astype(np.float32),
        "global_edge_ids": fixture["edge_ids"].copy(),
        "motor_ids": fixture["motor_ids"].copy(),
        "membership": fixture["membership"].astype(np.int64, copy=True),
        "native_fit_values32": target32,
        "native_fit_values64": target64,
        "calibration_positions": mapped,
        "individual_edge_efficacies": efficacy,
        "candidate_adopted": np.array(False),
        "original_conditional_guard_pass": guard}
    report = {"legs": provenance, "edge_count": edge_count,
        "muscle_count": muscle_count, "motor_count": motor_count,
        "derived_efficacy_min": float(efficacy.min()),
        "derived_efficacy_max": float(efficacy.max()),
        "failed_guard_legs": np.flatnonzero(~guard).tolist(),
        "clipped": False}
    return parameters, changed, expected, report


def compare_conditional(actual, expected, fields, atol=1e-8):
    report = {}
    accepted = True
    for key in fields:
        left = np.asarray(actual.get(key))
        right = np.asarray(expected.get(key))
        shape = left.shape == right.shape
        finite = bool(shape and np.isfinite(left).all() and np.isfinite(right).all())
        maximum = float(np.max(np.abs(left - right))) if finite and left.size else None
        good = bool(finite and maximum <= atol)
        report[key] = {"shape_exact": shape, "finite": finite,
                       "max_abs": maximum, "accepted": good}
        accepted &= good
    return {"accepted": bool(accepted), "atol": atol, "rtol": 0.,
            "fields": report}


def durable_preentry(row, persist, label, *, request_field=None):
    """Persist intent and request; a receipt failure guarantees no callback."""
    row.update(stage=label + "_intent", requested=False, entered=False,
               returned=False, post_return_error=False, known_zero_work=False)
    if request_field is not None:
        row[request_field] = False
    try:
        persist()
    except Exception as error:
        row.update(stage=label + "_intent_receipt_failed", known_zero_work=True,
                   pre_entry_error=repr(error))
        raise
    row.update(stage=label + "_requested", requested=True)
    if request_field is not None:
        row[request_field] = True
    try:
        persist()
    except Exception as error:
        row.update(stage=label + "_request_receipt_failed", requested=False,
                   entered=False, known_zero_work=True, pre_entry_error=repr(error))
        if request_field is not None:
            row[request_field] = False
        try:
            persist()
        except Exception as repair:
            error.add_note("Secondary pre-entry receipt repair: " + repr(repair))
        raise
    row.update(stage=label + "_request_receipt_persisted")


def accounted_operation(label, callback, rows, persist):
    row = {"label": label, "started_wall_s": time.perf_counter()}
    rows.append(row)
    durable_preentry(row, persist, label)
    try:
        row.update(entered=True, stage=label + "_entered")
        result = callback()
        row.update(returned=True, stage=label + "_returned")
    except Exception as error:
        row.update(stage=label + "_failed", error=repr(error), returned=False)
        try:
            persist()
        except Exception as secondary:
            error.add_note("Secondary operation receipt: " + repr(secondary))
        raise
    try:
        persist()
    except Exception as error:
        row.update(stage=label + "_post_return_receipt_failed",
                   post_return_error=True, error=repr(error))
        raise
    return result


def profiled_construction(label, callback, rows, persist, *,
                          interval_code=None, synchronize=None,
                          capture_probe=None, on_return=None):
    """Count actual construction _interval calls and seal them by synchronization."""
    if interval_code is None:
        interval_code = ResearchRateNetwork._interval.__code__
        if capture_probe is None:
            def capture_probe():
                import torch
                return bool(torch.cuda.is_current_stream_capturing())
    elif capture_probe is None:
        capture_probe = lambda: False
    if synchronize is None:
        def synchronize():
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA synchronization required for E141 setup")
            torch.cuda.synchronize()
    row = {"label": label, "started_wall_s": time.perf_counter(),
           "interval_entries": 0, "interval_returns": 0,
           "entered_steps": 0, "returned_steps": 0,
           "executed_interval_entries": 0, "captured_interval_entries": 0,
           "unknown_capture_state_entries": 0,
           "entered_executed_setup_steps": 0,
           "entered_capture_setup_steps": 0,
           "entered_unknown_capture_state_steps": 0,
           "returned_executed_setup_steps": 0,
           "returned_capture_setup_steps": 0,
           "returned_unknown_capture_state_steps": 0,
           "recorded_capture_setup_steps": 0,
           "unknown_capture_setup_steps": 0,
           "synchronize_requested": False, "synchronize_returned": False,
           "known_completed_setup_steps": 0, "unknown_setup_steps": 0,
           "profiler_restored": False, "ownership_transferred": False,
           "capture_probe_api": ("torch.cuda.is_current_stream_capturing"
               if interval_code is ResearchRateNetwork._interval.__code__
               else "injected_or_generated")}
    rows.append(row)
    durable_preentry(row, persist, label)
    previous = sys.getprofile()
    stack = {}

    def profiler(frame, event, arg):
        del arg
        if frame.f_code is interval_code:
            if event == "call":
                steps = int(getattr(frame.f_locals.get("self"), "interval_steps", 0))
                capture = None
                try:
                    capture = bool(capture_probe())
                except Exception as error:
                    row.setdefault("capture_probe_errors", []).append(repr(error))
                row["interval_entries"] += 1
                row["entered_steps"] += steps
                if capture is True:
                    kind = "capture"
                    row["captured_interval_entries"] += 1
                elif capture is False:
                    kind = "executed"
                    row["executed_interval_entries"] += 1
                else:
                    kind = "unknown_capture_state"
                    row["unknown_capture_state_entries"] += 1
                row["entered_" + kind + "_setup_steps"] += steps
                stack[id(frame)] = (steps, kind)
            elif event == "return":
                row["interval_returns"] += 1
                steps, kind = stack.pop(id(frame), (0, "unknown_capture_state"))
                row["returned_steps"] += steps
                row["returned_" + kind + "_setup_steps"] += steps
        return profiler

    result = None
    primary = None
    callback_succeeded = False
    try:
        sys.setprofile(profiler)
        row.update(entered=True, stage=label + "_entered")
        result = callback()
        callback_succeeded = True
        row.update(returned=True, stage=label + "_callback_returned")
        if on_return is not None:
            on_return(result)
            row["ownership_transferred"] = True
    except Exception as error:
        primary = error
        if callback_succeeded:
            row.update(returned=True, post_return_error=True,
                       stage=label + "_ownership_transfer_failed",
                       error=repr(error))
        else:
            row.update(returned=False, stage=label + "_callback_failed",
                       error=repr(error))
    finally:
        try:
            sys.setprofile(previous)
            row["profiler_restored"] = sys.getprofile() is previous
        except Exception as error:
            row["profiler_restore_error"] = repr(error)
            if primary is None:
                row["stage"] = label + "_profiler_restore_failed"
                primary = error
            else:
                primary.add_note("Secondary profiler restoration: " + repr(error))
        row["synchronize_requested"] = True
        try:
            synchronize()
            row["synchronize_returned"] = True
            if primary is None:
                row["stage"] = label + "_synchronized"
        except Exception as error:
            row["unknown_setup_steps"] = row["entered_steps"]
            row["synchronize_error"] = repr(error)
            if primary is None:
                row["stage"] = label + "_synchronize_failed"
                primary = error
            else:
                primary.add_note("Secondary setup synchronization: " + repr(error))
        if callback_succeeded and row["synchronize_returned"]:
            row["known_completed_setup_steps"] = row[
                "returned_executed_setup_steps"]
            row["recorded_capture_setup_steps"] = row[
                "returned_capture_setup_steps"]
            row["unknown_setup_steps"] = max(0, row[
                "entered_executed_setup_steps"] - row[
                    "returned_executed_setup_steps"]) + row[
                        "entered_unknown_capture_state_steps"]
            row["unknown_capture_setup_steps"] = max(0, row[
                "entered_capture_setup_steps"] - row[
                    "returned_capture_setup_steps"])
        else:
            row["known_completed_setup_steps"] = 0
            row["recorded_capture_setup_steps"] = 0
            row["unknown_setup_steps"] = (row["entered_executed_setup_steps"]
                + row["entered_unknown_capture_state_steps"])
            row["unknown_capture_setup_steps"] = row[
                "entered_capture_setup_steps"]
        row["finished_wall_s"] = time.perf_counter()
        try:
            persist()
        except Exception as error:
            row.update(post_return_error=bool(row.get("returned")),
                       stage=label + "_final_receipt_failed")
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary construction receipt: " + repr(error))
    if primary is not None:
        if result is not None and hasattr(result, "close"):
            try:
                result.close()
            except Exception as error:
                primary.add_note("Secondary constructed-owner close: " + repr(error))
        raise primary
    return result


class ControlJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.rows = 0
        self.by_stage = {}
        self.by_owner_stage = {}

    def append(self, value):
        self.stream.write(json.dumps(jsonable(value), ensure_ascii=False,
                                     sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        self.rows += 1
        stage = str(value.get("stage", "missing"))
        owner = str(value.get("owner", "missing"))
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        label = owner + ":" + stage
        self.by_owner_stage[label] = self.by_owner_stage.get(label, 0) + 1

    def summary(self):
        return {"rows": self.rows, "by_stage": dict(self.by_stage),
                "by_owner_stage": dict(self.by_owner_stage)}

    def close(self):
        self.stream.close()


def advance_control(session, owner, index, clock, journal):
    before = {"control_tick": int(session.control_tick),
              "neural_tick": int(session.network.tick),
              "physics_time_s": float(session.body.physics_time())}
    intent = {"owner": owner, "index": index, "stage": "intent",
              "requested": False, "entered": False, "returned": False,
              "before": before}
    journal.append(intent)
    request = {**intent, "stage": "requested", "requested": True}
    journal.append(request)
    try:
        clock.advance(session, 1, owner)
    except Exception as error:
        event = clock.events[-1] if clock.events else None
        failure = {**request, "stage": "advance_failed", "entered": True,
                   "error": type(error).__name__, "message": str(error),
                   "clock_event": event, "advance_returned": False,
                   "synchronized_readout": False,
                   "unknown_neural_work": True}
        try:
            journal.append(failure)
        except Exception as secondary:
            error.add_note("Secondary failed-control journal: " + repr(secondary))
        raise
    event = clock.events[-1]
    journal.append({**request, "stage": "advance_returned", "entered": True,
                    "returned": True, "advance_returned": True,
                    "synchronized_readout": False,
                    "clock_event": event, "unknown_neural_work": False})
    return event


def record_readout(journal, owner, index, event, *, complete, error=None):
    """Keep successful advance clocks separate from host-visible observations."""
    row = {"owner": owner, "index": index,
           "stage": "readout_returned" if complete else "readout_failed",
           "requested": True, "entered": True, "returned": bool(complete),
           "advance_returned": True, "synchronized_readout": bool(complete),
           "clock_event": event, "unknown_neural_work": False,
           "unknown_readout": not bool(complete)}
    if error is not None:
        row.update(error=type(error).__name__, message=str(error))
    journal.append(row)


def _hash_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _raw_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _close_owned(primary, operations, secondary):
    for label, operation in operations:
        if operation is None:
            continue
        try:
            operation()
        except Exception as error:
            secondary.append({"stage": label, "error": type(error).__name__,
                              "message": str(error)})
            if primary is not None:
                primary.add_note("Secondary " + label + ": " + repr(error))
            else:
                primary = error
    return primary


def _observe(session, detailed, available=None):
    if available is None:
        available = {}
    current = session.network._current(session.network.rate)
    available["current_tensor"] = current[detailed]
    fields = available.setdefault("fields", {})
    fields["rate"] = session.network.readout(detailed)
    fields["recurrent_current"] = available["current_tensor"].cpu().numpy().copy()
    fields["drive"] = session.network.drive[detailed].cpu().numpy().copy()
    fields["adaptation"] = session.network.adaptation[detailed].cpu().numpy().copy()
    fields["muscle_activation"] = session.adapter.activation.copy()
    available["complete"] = True
    return fields


def _signal_arrays(signal_rows, detailed, activation_width=MUSCLE_COUNT):
    """Keep empty health-stop traces two-dimensional for downstream diagnostics."""
    if signal_rows:
        return {key: np.asarray([row[key] for row in signal_rows]) for key in
                ("rate", "recurrent_current", "drive", "adaptation",
                 "muscle_activation")}
    neural_shape = (0, len(detailed))
    return {
        "rate": np.empty(neural_shape, np.float32),
        "recurrent_current": np.empty(neural_shape, np.float32),
        "drive": np.empty(neural_shape, np.float32),
        "adaptation": np.empty(neural_shape, np.float32),
        "muscle_activation": np.empty((0, activation_width), np.float32),
    }


def strict_startup(rows, native, startup, limits):
    observed = max(0, min(len(rows) - 1, 99))
    times = np.asarray([row["time_s"] for row in rows[1:observed + 1]])
    q = np.asarray([row["joint_targets_rad"] for row in rows[1:observed + 1]],
                   dtype=float)
    order = tuple(f"{leg}_tarsus" for leg in LEGS)
    tendon = np.asarray([[row["tendon_inputs"][key] for key in order]
                         for row in rows[1:observed + 1]], dtype=float)
    complete = bool(observed == 99 and len(native) >= 4950
        and np.array_equal(times, np.arange(1, 100) * .005)
        and q.shape == (99, 42) and tendon.shape == (99, 6)
        and startup["target_command"].shape == (99, 42)
        and startup["target_tendon"].shape == (99, 6))
    missing = {"RMS": None, "max_abs": None, "available": False}
    q_error = finite_error(q, startup["target_command"]) if complete else missing.copy()
    tendon_error = (finite_error(tendon, startup["target_tendon"])
                    if complete else missing.copy())
    screen = bool(complete and q_error["available"] and tendon_error["available"]
        and q_error["RMS"] <= limits["q_RMS"]
        and q_error["max_abs"] <= limits["q_max_abs"]
        and tendon_error["RMS"] <= limits["LTM_RMS"]
        and tendon_error["max_abs"] <= limits["LTM_max_abs"])
    return {"complete_original99_prefix": complete, "observed_controls": observed,
            "q": q_error, "LTM": tendon_error, "limits": copy.deepcopy(limits),
            "strict_E141_startup_screen": screen, "tendon_order": list(order)}


def native_outcome(reason, measured_health_stop, summary, future_exact,
                   strict, raw_errors, analysis_errors):
    """Give technical integrity precedence while retaining measured health facts."""
    technical = bool(raw_errors or analysis_errors)
    screens = summary.get("development_screens", {})
    all_numeric = bool(all(screens.get(key) is True for key in
                           ("F3_numeric", "F4_numeric", "F5_numeric")))
    if technical:
        validity, verdict = "incomplete", "invalid_not_tested"
    elif measured_health_stop:
        validity, verdict = "measured_native_health_stop", "refuted_health_stop"
    elif (reason == "duration" and summary.get("complete_10s_native_health") is True
            and future_exact):
        validity = "valid_full_BANC_native_transfer_diagnostic"
        verdict = ("supported_development_diagnostic" if all_numeric
                   and strict.get("strict_E141_startup_screen") is True
                   else "refuted_numeric_or_startup")
    else:
        validity, verdict = "incomplete", "invalid_not_tested"
    return validity, verdict, technical


def _save_body_raw(out, graph, neutral, session, rows, signal_rows, detailed,
                   native, clone_native, clock, spec, partial_observations):
    errors = []

    def attempt(label, callback):
        try:
            callback()
        except Exception as error:
            errors.append({"artifact": label, "error": type(error).__name__,
                           "message": str(error), "traceback": traceback.format_exc()})

    attempt("clock-events.json", lambda: write_json(out / "clock-events.json", clock.events))
    attempt("clock-summary.json", lambda: write_json(out / "clock-summary.json", {
        "actual_neural_s": clock.neural_s, "actual_physical_s": clock.physical_s,
        "clock_valid": clock.clock_valid, "actual_costs_by_owner": clock.by_owner(),
        "main_completed_controls": max(0, len(rows) - 1),
        "main_native_samples": len(native), "clone_native_samples": len(clone_native),
        "incomplete_control_possible": bool(clock.events and (
            clock.events[-1]["advance_error"] is not None
            or clock.events[-1]["clock_read_errors"]))}))
    attempt("native-contact.npz", lambda: np.savez_compressed(
        out / "native-contact.npz", **native_arrays(native)))
    attempt("clone-native-contact.npz", lambda: np.savez_compressed(
        out / "clone-native-contact.npz", **native_arrays(clone_native)))

    def trace():
        model_hash = (session.network.identity if session is not None
                      else neutral["network"]["model_hash"])
        body_hash = session.body.model_hash if session is not None else None
        adapter_hash = session.adapter.identity if session is not None else None
        write_json(out / "trace.json", {"schema": "flylab.m3-e141-native-transfer-trace.v1",
            "controller": "fixed E141 delta/gain/log-tau; full native sensory/body feedback",
            "graph_hash": graph.hash, "model_hash": model_hash,
            "body_hash": body_hash, "adapter_hash": adapter_hash,
            "parameters": {**neutral["parameters"], "diagnostic": spec},
            "samples": rows})
    attempt("trace.json", trace)

    def signals():
        count = len(signal_rows)
        activation_width = (len(np.asarray(session.adapter.activation))
                            if session is not None else MUSCLE_COUNT)
        values = _signal_arrays(signal_rows, detailed, activation_width)
        np.savez_compressed(out / "recruitment-signals.npz",
            time_s=np.asarray([row["time_s"] for row in rows[1:count + 1]]),
            observed_ids=detailed, **values)
    if detailed is not None:
        attempt("recruitment-signals.npz", signals)

    def partial():
        payload = {}
        receipt = []
        for index, item in enumerate(partial_observations):
            fields = item.get("available", {}).get("fields", {})
            for key, value in fields.items():
                payload[f"partial{index}_{key}"] = np.asarray(value)
            receipt.append({"time_s": item.get("time_s"),
                            "control_tick": item.get("control_tick"),
                            "fields": sorted(fields),
                            "complete": bool(item.get("available", {}).get("complete"))})
        np.savez_compressed(out / "partial-observation.npz", **payload)
        write_json(out / "partial-observation.json", receipt)
    if partial_observations:
        attempt("partial-observation", partial)
    return errors


def _save_static_provenance(out, session, candidate_weights, baseline_weights,
                            fixture, parameters, candidate_calibration, anatomy):
    actual = source_matrix(session.network)
    if (not np.array_equal(actual.indptr, candidate_weights.indptr)
            or not np.array_equal(actual.indices, candidate_weights.indices)
            or not np.array_equal(actual.data, candidate_weights.data)):
        raise ValueError("Runtime candidate global CSR changed")
    targets = np.asarray(fixture["edge_ids"], np.int64)
    non_target = np.ones(len(actual.data), bool)
    non_target[targets] = False
    if (not np.array_equal(actual.data[non_target], baseline_weights.data[non_target])
            or not np.array_equal(actual.data[targets], parameters["native_fit_values32"])):
        raise ValueError("Runtime target/non-target CSR proof failed")
    np.savez(out / "actual-global-csr.npz", indptr=actual.indptr,
             indices=actual.indices, data=actual.data,
             selected_edge_ids=targets,
             selected_values=actual.data[targets])
    write_json(out / "csr-provenance.json", {
        "rows": actual.shape[0], "nnz": int(actual.nnz), "selected": len(targets),
        "indptr_sha256": _hash_array(actual.indptr),
        "indices_sha256": _hash_array(actual.indices),
        "data_sha256": _hash_array(actual.data),
        "candidate_non_target_sha256": _hash_array(actual.data[non_target]),
        "baseline_non_target_sha256": _hash_array(baseline_weights.data[non_target]),
        "non_target_exact": True, "selected_float32_exact": True})
    network = session.network
    actual_tau32 = network._cell_adaptation_tau.detach().cpu().numpy().copy()
    actual_inverse32 = network._cell_inverse_tau.detach().cpu().numpy().copy()
    actual_gain32 = network._cell_gain.detach().cpu().numpy().copy()
    if (not np.array_equal(actual_tau32, network._full_tau32)
            or not np.array_equal(actual_inverse32, network._full_inverse32)
            or not np.array_equal(actual_gain32[network.motor_ids],
                                  network.motor_gains)):
        raise ValueError("Actual device motor gain/timescale tensors changed")
    np.savez(out / "motor-timescale-configuration.npz",
        motor_ids=network.motor_ids, motor_gains=network.motor_gains,
        motor_log_tau=network.motor_log_tau,
        muscle_gains=parameters["gain"], muscle_gains32=parameters["muscle_gains32"],
        muscle_log_tau=parameters["log_tau"], membership=parameters["membership"],
        motor_tau64=network._full_tau64[network.motor_ids],
        motor_tau32=network._full_tau32[network.motor_ids],
        motor_inverse32=network._full_inverse32[network.motor_ids],
        actual_motor_gain32=actual_gain32[network.motor_ids],
        actual_motor_tau32=actual_tau32[network.motor_ids],
        actual_motor_inverse32=actual_inverse32[network.motor_ids],
        membrane_tau=network.tau.detach().cpu().numpy().copy(),
        cell_gain=actual_gain32)
    np.savez(out / "adapter-provenance.npz",
        motor_ids=candidate_calibration["motor_ids"],
        moment=candidate_calibration["moment"], rest=candidate_calibration["rest"],
        pool=candidate_calibration["pool"], membership=candidate_calibration["membership"],
        ltm=candidate_calibration["ltm"])
    write_json(out / "muscle-provenance.json", anatomy)
    patch = session.motor_timescale_patch
    motor_patch = session.motor_adaptation_patch
    write_json(out / "network-layer-identities.json", {
        "network_identity": network.identity,
        "network_source_motor_identity": network.source_motor_identity,
        "motor_source_core_identity": network.source_core_identity,
        "adapter_identity": session.adapter.identity,
        "timescale_schema": patch["schema"],
        "timescale_source_adapter_identity": patch["source_adapter_identity"],
        "timescale_source_motor_identity": patch["source_motor_identity"],
        "motor_schema": motor_patch["schema"],
        "motor_source_adapter_identity": motor_patch["source_adapter_identity"],
        "motor_source_neural_identity": motor_patch["source_neural_identity"],
        "motor_source_cuda_implementation": motor_patch[
            "source_cuda_implementation"],
        "motor_roster_sha256": _hash_array(
            np.asarray(patch["motor_roster"], dtype="U")),
        "candidate_adopted": False, "F6_exposed": False})


def _construction_totals(rows):
    return {"attempts": len(rows),
        "returned": sum(item.get("returned") is True for item in rows),
        "interval_entries": sum(item.get("interval_entries", 0) for item in rows),
        "interval_returns": sum(item.get("interval_returns", 0) for item in rows),
        "executed_interval_entries": sum(
            item.get("executed_interval_entries", 0) for item in rows),
        "captured_interval_entries": sum(
            item.get("captured_interval_entries", 0) for item in rows),
        "unknown_capture_state_entries": sum(
            item.get("unknown_capture_state_entries", 0) for item in rows),
        "known_completed_setup_steps": sum(
            item.get("known_completed_setup_steps", 0) for item in rows),
        "recorded_capture_setup_steps": sum(
            item.get("recorded_capture_setup_steps", 0) for item in rows),
        "unknown_setup_steps": sum(item.get("unknown_setup_steps", 0) for item in rows),
        "unknown_capture_setup_steps": sum(
            item.get("unknown_capture_setup_steps", 0) for item in rows),
        "all_synchronized": bool(rows and all(
            item.get("synchronize_returned") is True for item in rows)),
        "all_profilers_restored": bool(rows and all(
            item.get("profiler_restored") is True for item in rows))}


def cleanup_cuda_runtime(entered, ledger, *, torch_module=None, cupy_module=None):
    """Clean an entered CUDA scope; pre-entry rejection imports no CUDA module."""
    receipt = {"entered": bool(entered), "performed": False,
               "torch_sync_returned": False, "cupy_sync_returned": False,
               "torch_cache_released": False, "cupy_pool_released": False,
               "errors": []}
    ledger["cuda_cleanup"] = receipt
    if not entered:
        receipt["stage"] = "skipped_before_cuda_entry"
        return []
    errors = []

    def attempt(stage, callback, returned_key):
        try:
            callback()
            receipt[returned_key] = True
        except Exception as error:
            receipt["errors"].append({"stage": stage,
                "error": type(error).__name__, "message": str(error)})
            errors.append((stage, error))

    try:
        if torch_module is None:
            import torch as torch_module
        torch_available = bool(torch_module.cuda.is_available())
        receipt["torch_cuda_available"] = torch_available
        if torch_available:
            attempt("final_torch_synchronize", torch_module.cuda.synchronize,
                    "torch_sync_returned")
            attempt("final_torch_empty_cache", torch_module.cuda.empty_cache,
                    "torch_cache_released")
    except Exception as error:
        receipt["errors"].append({"stage": "final_torch_import",
            "error": type(error).__name__, "message": str(error)})
        errors.append(("final_torch_import", error))
    try:
        if cupy_module is None:
            import cupy as cupy_module
        attempt("final_cupy_synchronize",
                lambda: cupy_module.cuda.Device().synchronize(),
                "cupy_sync_returned")
        attempt("final_cupy_free_pool",
                cupy_module.get_default_memory_pool().free_all_blocks,
                "cupy_pool_released")
    except Exception as error:
        receipt["errors"].append({"stage": "final_cupy_import",
            "error": type(error).__name__, "message": str(error)})
        errors.append(("final_cupy_import", error))
    try:
        gc.collect()
        receipt["python_gc_returned"] = True
    except Exception as error:
        receipt["errors"].append({"stage": "final_python_gc",
            "error": type(error).__name__, "message": str(error)})
        errors.append(("final_python_gc", error))
    receipt["performed"] = True
    receipt["stage"] = "completed" if not errors else "completed_with_errors"
    return errors


def _persist_manifest(out):
    files = {}
    for path in sorted(item for item in out.rglob("*") if item.is_file()
                       and item.name not in {"output-manifest.json", "result.json"}):
        relative = path.relative_to(out).as_posix()
        files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e141.output-manifest.v1",
                "excludes": ["output-manifest.json", "result.json"],
                "file_count": len(files), "files": files,
                "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(out / "output-manifest.json", manifest)
    return {"path": "output-manifest.json", "sha256": sha256(out / "output-manifest.json"),
            "file_count": len(files), "total_bytes": manifest["total_bytes"]}


def _conditional_forward(engine, parameters, out, ledger, persist):
    row = {"label": "conditional_forward", "solve_requested": False,
           "solve_returned": False, "derivative": False,
           "requested_context_ticks": engine.contexts * engine.ticks,
           "requested_conditional_s": engine.contexts * engine.ticks * .00025,
           "requested_decoder_endpoints": engine.contexts * (engine.samples - 1),
           "requested_derivative_banks": 0}
    ledger["conditional"] = row
    durable_preentry(row, persist, "conditional_forward",
                     request_field="solve_requested")
    device = None
    raw = {}
    try:
        row.update(entered=True, solve_entered=True,
                   stage="conditional_forward_entered")
        device, raw, timing = engine.solve(parameters["delta"], parameters["gain"],
                                           parameters["log_tau"], derivative=False)
        row.update(solve_returned=True, returned=True,
                   stage="conditional_forward_returned", progress=timing["progress"],
                   timing={key: value for key, value in timing.items()
                           if key != "progress"})
        raw.update(delta=parameters["delta"], gain=parameters["gain"],
                   log_tau=parameters["log_tau"],
                   edge_ids=engine.problem["edge_ids"],
                   motor_indices=engine.problem["motor_indices"],
                   muscle_indices=engine.problem["muscle_indices"])
        path = out / "conditional-forward-raw.npz"
        np.savez(path, **raw)
        row.update(raw_saved=True, raw_path=path.name, raw_sha256=sha256(path))
        persist()
        return raw
    except Exception as error:
        row.update(stage="conditional_forward_failed", error=repr(error),
                   post_return_error=bool(row.get("solve_returned")))
        progress = getattr(error, "response_progress", None)
        if progress is not None:
            row["progress"] = progress
        partial = dict(getattr(error, "partial_response", {}) or {})
        partial.update({key: value for key, value in raw.items() if key not in partial})
        if device is not None:
            try:
                rescued, secondary = engine.rescue_device_state(device)
                for key, value in rescued.items():
                    partial.setdefault(key, value)
                row["secondary_rescue_errors"] = secondary
            except Exception as secondary:
                error.add_note("Secondary conditional rescue: " + repr(secondary))
        if partial:
            try:
                path = out / "conditional-forward-partial-raw.npz"
                np.savez(path, **partial)
                row["partial_raw"] = {"path": path.name, "sha256": sha256(path),
                                      "keys": sorted(partial)}
            except Exception as secondary:
                error.add_note("Secondary conditional partial raw: " + repr(secondary))
        try:
            persist()
        except Exception as secondary:
            error.add_note("Secondary conditional ledger: " + repr(secondary))
        raise


def _prepare_candidate(freeze, fixture, parameters, candidate_calibration,
                       source_calibration, graph, original, anatomy, out,
                       ledger, persist):
    encoder = BancTargetEncoder(graph, forward_drive=5000)
    unscaled, _ = rate_weights(graph)
    scaled = encoder.calibrate_weights(unscaled, candidate_calibration["motor_ids"])
    baseline_weights = patch_weights(scaled, source_calibration["edge_ids"],
                                     source_calibration["source_values"],
                                     source_calibration["values"])
    candidate_weights = patch_weights(scaled, candidate_calibration["edge_ids"],
                                      candidate_calibration["source_values"],
                                      candidate_calibration["values"])
    full_tau = patch_tau(np.full(graph.n, .02, np.float32),
                         candidate_calibration["tau_ids"],
                         candidate_calibration["tau_values"])
    intrinsic_out = out / "intrinsic-preparation"
    intrinsic_out.mkdir(exist_ok=False)
    profiled_construction("prepare_intrinsic_neutral",
        lambda: prepare_neutral_checkpoint(graph, original, candidate_calibration,
            anatomy, candidate_weights, full_tau, intrinsic_out),
        ledger["construction_attempts"], persist)
    inner = accounted_operation("load_intrinsic_neutral",
        lambda: StateStore.load(intrinsic_out / "candidate-neutral"),
        ledger["storage_operations"], persist)
    base = motor = trial = disk = None
    try:
        base = profiled_construction("restore_intrinsic_neutral",
            lambda: IntrinsicRecruitmentTrial.from_checkpoint(graph, inner),
            ledger["construction_attempts"], persist)

        def transfer_base(_candidate):
            nonlocal base
            base = None

        motor = profiled_construction("apply_motor_gain",
            lambda: MotorAdaptationTrial(base, parameters["gain"],
                anatomy["muscles"], candidate_calibration["membership"]),
            ledger["construction_attempts"], persist, on_return=transfer_base)

        def transfer_motor(_candidate):
            nonlocal motor
            motor = None

        trial = profiled_construction("apply_motor_timescale",
            lambda: MotorTimescaleTrial(motor, parameters["log_tau"],
                anatomy["muscles"], candidate_calibration["membership"]),
            ledger["construction_attempts"], persist, on_return=transfer_motor)
        neutral = trial.snapshot()
        for key in ("body", "world", "cuts", "navigation", "parameters", "seed",
                    "control_tick", "cpg_hash", "initial_position", "signed_forward_mm"):
            exact(neutral[key], original[key], "E141 original zero-state " + key)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            exact(neutral["network"][key], original["network"][key],
                  "E141 original zero neural " + key)
        if neutral["network"]["tick"] != 0 or neutral["control_tick"] != 0:
            raise ValueError("E141 native candidate must start at whole zero state")
        _save_static_provenance(out, trial, candidate_weights, baseline_weights,
                                fixture, parameters, candidate_calibration, anatomy)
        accounted_operation("save_candidate_neutral",
            lambda: StateStore.save(out / "candidate-neutral", neutral),
            ledger["storage_operations"], persist)
        saved = accounted_operation("load_candidate_neutral",
            lambda: StateStore.load(out / "candidate-neutral"),
            ledger["storage_operations"], persist)
        disk = profiled_construction("restore_candidate_neutral",
            lambda: MotorTimescaleTrial.from_checkpoint(graph, saved),
            ledger["construction_attempts"], persist)
        exact(neutral, disk.snapshot(), "E141 whole outer neutral disk state")
        return neutral, candidate_weights, baseline_weights
    finally:
        active = sys.exc_info()[1]
        cleanup = []
        for label, owner in (("candidate disk clone", disk), ("timescale", trial),
                             ("motor", motor), ("intrinsic", base)):
            if owner is None or getattr(owner, "closed", False):
                continue
            try:
                owner.close()
            except Exception as error:
                cleanup.append((label, error))
        if cleanup and active is not None:
            for label, error in cleanup:
                active.add_note("Secondary " + label + " close: " + repr(error))
        elif cleanup:
            raise cleanup[0][1]


def _native_body(freeze, graph, candidate_calibration, observation,
                 startup, neutral, parameters, anatomy, out, spec,
                 ledger, persist, process_began):
    body_began = time.perf_counter()
    session = clone = None
    restore_main = restore_clone = None
    journal = None
    rows = []
    signal_rows = []
    native = []
    clone_native = []
    partial_observations = []
    clock = AdvanceLedger()
    detailed = None
    reason = "duration"
    runtime_error = None
    raw_errors = []
    future_ready = False
    measured_health_stop = False
    secondary = []
    try:
        saved = accounted_operation("load_native_candidate",
            lambda: StateStore.load(out / "candidate-neutral"),
            ledger["storage_operations"], persist)
        session = profiled_construction("restore_native_main",
            lambda: MotorTimescaleTrial.from_checkpoint(graph, saved),
            ledger["construction_attempts"], persist)
        if session.network.device != "cuda":
            raise ValueError("Original full BANC CUDA body required")
        exact(neutral, session.snapshot(), "E141 native outer zero start")
        _, detailed, metadata_sources, _dns = observation_ids(
            graph, candidate_calibration, observation)
        weights = source_matrix(session.network)
        motor_ell = dict(zip(candidate_calibration["motor_ids"].tolist(),
            session.network.motor_log_tau.tolist(), strict=True))
        observed_motor_ell = np.asarray(
            [motor_ell.get(int(cell), np.nan) for cell in detailed], np.float64)
        np.savez_compressed(out / "recruitment-metadata.npz",
            observed_ids=detailed, motor_ids=candidate_calibration["motor_ids"],
            source_ids=metadata_sources,
            source_to_motor=weights[candidate_calibration["motor_ids"]][
                :, metadata_sources].toarray(),
            gain=session.network._cell_gain.detach().cpu().numpy()[detailed].copy(),
            adaptation_log_tau=np.log(
                session.network._full_tau64[detailed] / np.float64(.3)),
            motor_log_tau=observed_motor_ell,
            motor_log_tau_available=np.isfinite(observed_motor_ell),
            **{key: getattr(session.network, key).detach().cpu().numpy()[detailed].copy()
               for key in ("tau", "a", "threshold", "cap")})
        restore_main = install_native_observer(session.body, native)
        accounted_operation("save_native_start",
            lambda: StateStore.save(out / "start", session.snapshot()),
            ledger["storage_operations"], persist)
        rows.append(capture_control(session))
        available = {}
        try:
            _observe(session, detailed, available)
            np.savez_compressed(out / "start-observation.npz",
                                observed_ids=detailed, **available["fields"])
        except Exception:
            partial_observations.append({"time_s": rows[0]["time_s"],
                "control_tick": 0, "available": available})
            raise
        write_json(out / "advance-start.json", {"experiment": "E141",
            "main_requested_controls": 2000, "clone_requested_controls": 20,
            "neural_steps_per_control": 20, "physics_steps_per_control": 50,
            "body_wall_before_first_advance_s": time.perf_counter() - body_began})
        journal = ControlJournal(out / "advance-journal.jsonl")
        with (out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(jsonable(rows[0]), allow_nan=False) + "\n")
            for tick in range(1, 2001):
                now = time.perf_counter()
                if (now - body_began > BODY_LIMIT_S - RESERVE_S
                        or now - process_began > INTERNAL_LIMIT_S - RESERVE_S):
                    reason = "resource_guard"
                    break
                before_native = len(native)
                event = advance_control(session, "main", tick, clock, journal)
                available = {}
                try:
                    row = capture_control(session)
                    if (len(native) - before_native != 50
                            or session.network.tick != tick * 20
                            or abs(session.body.physics_time() - tick * .005) > 1e-8):
                        raise ValueError(
                            "Complete 5 ms control requires 20 neural/50 physics steps")
                    rows.append(row)
                    stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
                    signal_rows.append(_observe(session, detailed, available))
                    record_readout(journal, "main", tick, event, complete=True)
                except Exception as error:
                    try:
                        partial_time = (rows[-1]["time_s"] if len(rows) > tick
                                        else session.body.physics_time())
                    except Exception as time_error:
                        partial_time = None
                        error.add_note("Secondary partial clock: " + repr(time_error))
                    partial_observations.append({
                        "time_s": partial_time,
                        "control_tick": tick, "available": available})
                    try:
                        record_readout(journal, "main", tick, event,
                                       complete=False, error=error)
                    except Exception as secondary_error:
                        error.add_note("Secondary readout journal: "
                                       + repr(secondary_error))
                    raise
                if session.fault or session.body.fault:
                    reason = "native_health_guard"
                    measured_health_stop = True
                    break
                if tick == 200:
                    checkpoint = accounted_operation("snapshot_checkpoint_1s",
                        session.snapshot, ledger["storage_operations"], persist)
                    accounted_operation("save_checkpoint_1s",
                        lambda value=checkpoint: StateStore.save(
                            out / "checkpoint-1s", value),
                        ledger["storage_operations"], persist)
                    clone_saved = accounted_operation("load_checkpoint_1s",
                        lambda: StateStore.load(out / "checkpoint-1s"),
                        ledger["storage_operations"], persist)
                    clone = profiled_construction("restore_checkpoint_clone",
                        lambda value=clone_saved: MotorTimescaleTrial.from_checkpoint(
                            graph, value),
                        ledger["construction_attempts"], persist)
                    try:
                        restore_clone = install_native_observer(clone.body, clone_native)
                        for clone_tick in range(1, 21):
                            now = time.perf_counter()
                            if (now - body_began > BODY_LIMIT_S - RESERVE_S
                                    or now - process_began > INTERNAL_LIMIT_S - RESERVE_S):
                                reason = "resource_guard"
                                break
                            advance_control(clone, "clone", clone_tick,
                                            clock, journal)
                        if reason != "resource_guard":
                            if (clone.network.tick != 4400 or len(clone_native) != 1000
                                    or abs(clone.body.physics_time() - 1.1) > 1e-8):
                                raise ValueError("Exact 0.1 s clone clocks required")
                            clone_state = accounted_operation("snapshot_clone_1p1",
                                clone.snapshot, ledger["storage_operations"], persist)
                            record_readout(journal, "clone", 20,
                                           clock.events[-1], complete=True)
                            accounted_operation("save_clone_1p1",
                                lambda value=clone_state: StateStore.save(
                                    out / "clone-1p1", value),
                                ledger["storage_operations"], persist)
                    finally:
                        active = sys.exc_info()[1]
                        close_errors = []
                        for label, operation in (("clone observer", restore_clone),
                                ("clone", clone.close if clone else None)):
                            if operation is None:
                                continue
                            try:
                                operation()
                            except Exception as error:
                                close_errors.append(error)
                                if active is not None:
                                    active.add_note("Secondary " + label + ": " + repr(error))
                        restore_clone = None
                        clone = None
                        if active is None and close_errors:
                            raise close_errors[0]
                    if reason == "resource_guard":
                        break
                if tick == 220:
                    main_state = accounted_operation("snapshot_main_1p1",
                        session.snapshot, ledger["storage_operations"], persist)
                    accounted_operation("save_main_1p1",
                        lambda value=main_state: StateStore.save(
                            out / "main-1p1", value),
                        ledger["storage_operations"], persist)
                    future_ready = True
                if tick % 200 == 0:
                    ledger["main_recorded_controls"] = len(rows) - 1
                    ledger["clock"] = {"neural_s": clock.neural_s,
                                       "physical_s": clock.physical_s,
                                       "valid": clock.clock_valid}
                    persist()
        if reason == "duration" and not session.fault and not session.body.fault:
            final_state = accounted_operation("snapshot_native_final",
                session.snapshot, ledger["storage_operations"], persist)
            accounted_operation("save_native_final",
                lambda: StateStore.save(out / "final", final_state),
                ledger["storage_operations"], persist)
    except NativeHealthStop as error:
        reason = "native_health_guard"
        measured_health_stop = True
        ledger["measured_health_stop"] = {"type": type(error).__name__,
                                           "message": str(error)}
    except Exception as error:
        reason = "execution_error"
        runtime_error = error
    finally:
        if journal is not None:
            ledger["scientific_control_receipts"] = journal.summary()
            try:
                journal.close()
                ledger["advance_journal_rows"] = journal.rows
            except Exception as error:
                if runtime_error is None:
                    runtime_error = error
                    reason = "execution_error"
                else:
                    runtime_error.add_note("Secondary journal close: " + repr(error))
        ledger["clock"] = {"neural_s": clock.neural_s,
                           "physical_s": clock.physical_s,
                           "valid": clock.clock_valid,
                           "actual_costs_by_owner": clock.by_owner()}
        try:
            raw_errors.extend(_save_body_raw(out, graph, neutral, session, rows,
                signal_rows, detailed, native, clone_native, clock, spec,
                partial_observations))
        except Exception as error:
            raw_errors.append({"artifact": "body_raw_protocol",
                               "error": type(error).__name__, "message": str(error)})
        runtime_error = _close_owned(runtime_error, (
            ("clone observer", restore_clone),
            ("clone", clone.close if clone else None),
            ("main observer", restore_main),
            ("main", session.close if session else None)), secondary)
        if raw_errors:
            try:
                write_json(out / "raw-persistence-errors.json", raw_errors)
            except Exception as error:
                if runtime_error is not None:
                    runtime_error.add_note("Secondary raw-error receipt: " + repr(error))
                else:
                    runtime_error = error
                    reason = "execution_error"
    if runtime_error is not None:
        try:
            write_json(out / "native-runtime-error.json", {"reason": reason,
                "primary": {"type": type(runtime_error).__name__,
                            "message": str(runtime_error),
                            "notes": list(getattr(runtime_error, "__notes__", []))},
                "secondary": secondary, "raw_errors": raw_errors,
                "clock": {"neural_s": clock.neural_s,
                          "physical_s": clock.physical_s,
                          "valid": clock.clock_valid}})
        except Exception as error:
            runtime_error.add_note("Secondary native error receipt: " + repr(error))
        raise runtime_error

    analysis_errors = []
    def analysis_attempt(label, callback, default=None):
        try:
            return callback()
        except Exception as error:
            analysis_errors.append({"stage": label, "error": type(error).__name__,
                                    "message": str(error),
                                    "traceback": traceback.format_exc()})
            return default

    future_exact = False
    if future_ready and not raw_errors:
        def verify_future():
            exact(StateStore.load(out / "clone-1p1"),
                  StateStore.load(out / "main-1p1"),
                  "E141 whole 1 s checkpoint future at 1.1 s")
            return True
        future_exact = analysis_attempt("whole_disk_future", verify_future, False)
    quality = None
    if len(rows) == 2001 and not raw_errors:
        analyzed = analysis_attempt("original_gait_evaluator",
            lambda: analyze(load_model(out / "trace.json")))
        if analyzed is not None:
            quality = analyzed[0]
            analysis_attempt("quality-analysis.json",
                             lambda: write_json(out / "quality-analysis.json", quality))
    fault = bool(raw_errors or runtime_error or (session is not None and
                 (session.fault or session.body.fault)))
    summary = analysis_attempt("body_quality",
        lambda: body_quality(rows, native, quality, reason=reason, fault=fault,
            clock_valid=clock.clock_valid, future_exact=future_exact),
        {"complete_10s_native_health": False,
         "development_screens": {"F3_numeric": False, "F4_numeric": False,
                                 "F5_numeric": False}})
    analysis_attempt("quality-summary.json",
                     lambda: write_json(out / "quality-summary.json", summary))
    strict = analysis_attempt("strict_startup",
        lambda: strict_startup(rows, native, startup, freeze["startup_limits"]),
        {"strict_E141_startup_screen": False})
    historical = analysis_attempt("historical_E121_startup",
        lambda: e121.original_startup(rows, native, startup),
        {"original_E100_E103_startup_screen": False})
    analysis_attempt("startup-diagnostic.json", lambda: write_json(
        out / "startup-diagnostic.json", {"strict_E141": strict,
                                           "historical_E121_native": historical}))
    analysis_attempt("actual-startup-original99-raw.npz",
                     lambda: e121.save_startup_raw(out, rows, startup))
    slip = analysis_attempt("supported_slip", lambda: supported_slip(
        native_arrays(native), complete=summary["complete_10s_native_health"]),
        {"available": False, "load_weighted_mm_s": None,
         "support_samples": None, "support_load_sum_bw": None})
    analysis_attempt("supported-slip.json",
                     lambda: write_json(out / "supported-slip.json", slip))
    baseline_dir = _safe_path(freeze["baseline_directory"])
    def baseline_comparison():
        baseline = {name: json.loads((baseline_dir / filename).read_text(encoding="utf-8"))
            for name, filename in (("quality", "quality-summary.json"),
                                   ("startup", "startup-diagnostic.json"),
                                   ("supported_slip", "supported-slip.json"),
                                   ("result", "result.json"))}
        value = {"window_s": [.5, 10.], "baseline_E121": baseline,
                 "candidate_E141": {"quality": summary, "strict_startup": strict,
                                    "historical_startup": historical,
                                    "supported_slip": slip},
                 "component_attribution_allowed": False}
        write_json(out / "E121-E141-same-window-comparison.json", value)
        return value
    comparison = analysis_attempt("E121_same_window", baseline_comparison, {})
    if not raw_errors and (out / "recruitment-signals.npz").exists():
        analysis_attempt("source_diagnostics", lambda: source_diagnostics(
            BASE, out, candidate_calibration, neutral,
            summary["complete_10s_native_health"], reason))
    if analysis_errors:
        analysis_attempt("analysis-errors.json",
                         lambda: write_json(out / "analysis-errors.json", analysis_errors))
    validity, verdict, technical = native_outcome(reason, measured_health_stop,
        summary, future_exact, strict, raw_errors, analysis_errors)
    return {"validity": validity, "hypothesis_verdict": verdict,
        "technical_failure": technical,
        "stop_reason": reason, "measured_health_stop": measured_health_stop,
        "rows": len(rows), "main_completed_controls": max(0, len(rows) - 1),
        "main_native_samples": len(native), "clone_native_samples": len(clone_native),
        "actual_neural_s": clock.neural_s, "actual_physical_s": clock.physical_s,
        "clock_valid": clock.clock_valid, "whole_disk_future_exact": future_exact,
        "quality_summary": summary, "strict_startup": strict,
        "historical_E121_startup": historical, "supported_slip": slip,
        "same_window_comparison_saved": bool(comparison),
        "raw_persistence_errors": raw_errors, "analysis_errors": analysis_errors,
        "body_analysis_wall_s": time.perf_counter() - body_began}


def run(freeze_path, freeze_sha256, source_review_path,
        source_review_sha256, output):
    process_began = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve single E141 output: " + str(output))
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"schema": "flylab.m3.e141.execution-ledger.v1", "status": "created",
        "construction_attempts": [], "storage_operations": [], "conditional": None,
        "main_recorded_controls": 0, "clock": {}, "secondary_errors": [],
        "candidate_adopted": False, "F6_exposed": False,
        "fit_calls": 0, "LP_calls": 0, "original_processes": 1}
    primary = None
    result = None
    limits = None
    engine = None
    cuda_runtime_entered = False

    def persist():
        ledger["construction_totals"] = _construction_totals(
            ledger["construction_attempts"])
        ledger["process_wall_s"] = time.perf_counter() - process_began
        write_json(output / "advance-ledger.json", ledger)

    persist()
    try:
        freeze, review = reviewed_inputs(freeze_path, freeze_sha256,
                                         source_review_path, source_review_sha256)
        write_json(output / "source-review-snapshot.json", review)
        if shutil.disk_usage(output).free < MINIMUM_FREE_BYTES:
            raise OSError("E141 requires at least 12 GiB free before original work")
        limits, threads = joint_fit.one_cpu_thread()
        ledger["cpu_threads"] = threads
        e128 = json.loads(_safe_path(freeze["e128_freeze_path"]).read_text(
            encoding="utf-8"))
        fixture = joint_fit.frozen_problem(e128)
        source_calibration = load_npz(_safe_path(freeze["source_calibration_path"]))
        prepared_parameters = load_npz(_safe_path(freeze["parameters_path"]))
        prepared_calibration = load_npz(_safe_path(freeze["calibration_path"]))
        prepared_expected = load_npz(_safe_path(freeze["conditional_expected_path"]))
        parameters, candidate_calibration, expected, assembly = assemble_candidate(
            freeze, fixture, source_calibration)
        exact_arrays(parameters, prepared_parameters, "E141 parent parameter oracle")
        exact_arrays(candidate_calibration, prepared_calibration,
                     "E141 parent calibration oracle")
        exact_arrays(expected, prepared_expected, "E141 parent response oracle")
        if (assembly["failed_guard_legs"] != [2, 5]
                or parameters["candidate_adopted"].item() is not False):
            raise ValueError("Historical LH/RH guard failures must remain explicit")
        np.savez(output / "assembled-parameters.npz", **parameters)
        np.savez(output / "assembled-calibration.npz", **candidate_calibration)
        np.savez(output / "assembled-conditional-expected.npz", **expected)
        write_json(output / "assembly-review.json", assembly)
        ledger["status"] = "assembly_exact"
        persist()
        if time.perf_counter() - process_began >= INTERNAL_LIMIT_S - RESERVE_S:
            raise TimeoutError("E141 reserve exhausted before conditional forward")
        engine_row = {"label": "conditional_engine", "requested": False,
                      "entered": False, "returned": False}
        ledger["conditional_engine"] = engine_row
        durable_preentry(engine_row, persist, "conditional_engine")
        try:
            engine_row.update(entered=True, stage="conditional_engine_entered")
            cuda_runtime_entered = True
            ledger["cuda_runtime_entered"] = True
            engine = CudaAdaptationTimescaleEngine(fixture)
        except Exception as error:
            engine_row.update(stage="conditional_engine_failed", error=repr(error))
            raise
        engine_row.update(returned=True, stage="conditional_engine_returned",
                          metadata=engine.metadata())
        try:
            persist()
        except Exception as error:
            engine_row.update(stage="conditional_engine_post_return_receipt_failed",
                              post_return_error=True, error=repr(error))
            raise
        conditional_raw = _conditional_forward(engine, parameters, output,
                                               ledger, persist)
        comparison = compare_conditional(conditional_raw, expected,
            freeze["conditional"]["compare_fields"], freeze["conditional"]["atol"])
        write_json(output / "conditional-comparison.json", comparison)
        ledger["conditional_comparison"] = comparison
        ledger["status"] = "conditional_raw_compared"
        persist()
        if not comparison["accepted"]:
            result = {"experiment": "M3-E141-native-timescale-transfer",
                "validity": "conditional_numeric_mismatch",
                "hypothesis_verdict": "invalid_not_tested",
                "technical_failure": False, "body_started": False,
                "conditional_replay": comparison,
                "historical_conditional_accepted": False,
                "historical_failed_legs": [2, 5], "candidate_adopted": False,
                "F6_exposed": False, "whole_goal_complete": False}
        else:
            del engine
            engine = None
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as error:
                raise RuntimeError("Conditional CUDA cleanup failed") from error
            graph = GraphStore.load(_safe_path(freeze["graph_path"]))
            if graph.hash != freeze["graph_hash"] or graph.n != 158706:
                raise ValueError("Frozen original full BANC graph required")
            original = accounted_operation("load_original_zero_checkpoint",
                lambda: StateStore.load(_safe_path(freeze["original_checkpoint_path"])),
                ledger["storage_operations"], persist)
            baseline_neutral = accounted_operation("load_E121_neutral",
                lambda: StateStore.load(_safe_path(freeze["baseline_neutral_path"])),
                ledger["storage_operations"], persist)
            for key in ("body", "world", "cuts", "navigation", "parameters", "seed",
                        "control_tick", "cpg_hash", "initial_position", "signed_forward_mm"):
                exact(original[key], baseline_neutral[key], "E121 zero baseline " + key)
            for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
                exact(original["network"][key], baseline_neutral["network"][key],
                      "E121 zero neural " + key)
            anatomy = json.loads((_safe_path(freeze["baseline_directory"])
                / "muscle-provenance.json").read_text(encoding="utf-8"))
            neutral, _, _ = _prepare_candidate(freeze, fixture, parameters,
                candidate_calibration, source_calibration, graph, original,
                anatomy, output, ledger, persist)
            prepare_wall = time.perf_counter() - process_began
            ledger["prepare_wall_s"] = prepare_wall
            if prepare_wall > PREPARE_LIMIT_S:
                raise TimeoutError("E141 preparation exceeded frozen 45 s")
            observation = load_npz(BASE / "parent-e113-expected-observation.npz")
            startup = load_npz(BASE / "e103-startup-preserving-recruitment/startup-inputs.npz")
            spec = {"experiment": "E141", "hypothesis": "fixed six-leg native transfer",
                "main_controls": 2000, "clone_controls": 20,
                "neural_steps_per_control": 20, "physics_steps_per_control": 50,
                "historical_conditional_accepted": False,
                "historical_failed_legs": [2, 5], "candidate_adopted": False,
                "F6_exposed": False, "whole_goal_complete": False,
                "freeze_sha256": FREEZE_SHA, "source_review_sha256": source_review_sha256}
            write_json(output / "spec.json", spec)
            body = _native_body(freeze, graph, candidate_calibration, observation,
                startup, neutral, parameters, anatomy, output, spec, ledger,
                persist, process_began)
            process_wall = time.perf_counter() - process_began
            resources = bool(body["body_analysis_wall_s"] <= BODY_LIMIT_S
                and process_wall <= INTERNAL_LIMIT_S
                and PRIOR_CONSERVATIVE_S + process_wall <= PROSPECTIVE_LIMIT_S
                and _raw_bytes(output) <= RAW_LIMIT_BYTES
                and body["clock_valid"])
            if not resources:
                body.update(validity="resource_limit_incomplete",
                            hypothesis_verdict="invalid_not_tested")
            result = {"experiment": "M3-E141-native-timescale-transfer",
                **body, "body_started": True,
                "conditional_replay": comparison,
                "historical_conditional_accepted": False,
                "historical_failed_legs": [2, 5],
                "resource_limit_met": resources,
                "prepare_wall_s": prepare_wall, "process_wall_s": process_wall,
                "prior_conservative_process_s": PRIOR_CONSERVATIVE_S,
                "conservative_cumulative_s": PRIOR_CONSERVATIVE_S + process_wall,
                "construction_accounting": _construction_totals(
                    ledger["construction_attempts"]),
                "requested_conditional_model_s": 30.,
                "requested_conditional_context_ticks": 120000,
                "requested_scientific_neural_ticks": 40400,
                "requested_scientific_physics_steps": 101000,
                "actual_scientific_neural_ticks": (None if body["actual_neural_s"]
                    is None else round(body["actual_neural_s"] / .00025)),
                "actual_scientific_physics_steps": (None if body[
                    "actual_physical_s"] is None else round(
                        body["actual_physical_s"] / .0001)),
                "fit_calls": 0, "LP_calls": 0, "candidate_adopted": False,
                "F6_exposed": False, "whole_goal_complete": False,
                "M3_complete": False}
        ledger["status"] = "result_ready"
    except Exception as error:
        primary = error
        ledger["status"] = "technical_failed"
        ledger["primary_error"] = {"type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "notes": list(getattr(error, "__notes__", [])),
            "response_progress": getattr(error, "response_progress", None),
            "partial_response_keys": sorted(
                (getattr(error, "partial_response", {}) or {}).keys())}
    finally:
        if engine is not None:
            try:
                ledger["conditional_engine_final"] = engine.metadata()
            except Exception as error:
                ledger["secondary_errors"].append({"stage": "engine_metadata",
                    "error": type(error).__name__, "message": str(error)})
        engine = None
        cleanup_errors = cleanup_cuda_runtime(cuda_runtime_entered, ledger)
        for stage, error in cleanup_errors:
            ledger["secondary_errors"].append({"stage": stage,
                "error": type(error).__name__, "message": str(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary " + stage + ": " + repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:
                ledger["secondary_errors"].append({"stage": "thread_restore",
                    "error": type(error).__name__, "message": str(error)})
                if primary is None:
                    primary = error
        ledger["construction_totals"] = _construction_totals(
            ledger["construction_attempts"])
        ledger["raw_bytes"] = _raw_bytes(output)
        ledger["process_wall_s"] = time.perf_counter() - process_began
        try:
            persist()
        except Exception as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary final ledger: " + repr(error))
        if primary is None and result is None:
            primary = RuntimeError("E141 ended without a result")
        if primary is None and ledger["secondary_errors"]:
            primary = RuntimeError("E141 cleanup or persistence failed")
        if primary is None and time.perf_counter() - process_began > INTERNAL_LIMIT_S:
            primary = TimeoutError("E141 finalization exceeded internal 260 s")
        if primary is None:
            try:
                manifest = _persist_manifest(output)
                result["output_manifest"] = manifest
                result["process_wall_s"] = time.perf_counter() - process_began
                if result["process_wall_s"] > INTERNAL_LIMIT_S:
                    result.update(validity="resource_limit_incomplete",
                                  hypothesis_verdict="invalid_not_tested",
                                  resource_limit_met=False)
                write_json(output / "result.json", result)
            except Exception as error:
                primary = error
        if primary is not None:
            try:
                if result is not None:
                    result.update(validity="resource_limit_incomplete" if isinstance(
                        primary, TimeoutError) else "incomplete",
                        hypothesis_verdict="invalid_not_tested", technical_failure=True,
                        candidate_adopted=False, F6_exposed=False,
                        whole_goal_complete=False)
                    write_json(output / "result.json", result)
                write_json(output / "execution-error.json", {
                    "primary": {"type": type(primary).__name__,
                                "message": str(primary),
                                "notes": list(getattr(primary, "__notes__", []))},
                    "secondary": ledger["secondary_errors"],
                    "construction_accounting": ledger["construction_totals"],
                    "conditional": ledger.get("conditional"),
                    "clock": ledger.get("clock"),
                    "raw_bytes": _raw_bytes(output),
                    "process_wall_s": time.perf_counter() - process_began,
                    "candidate_adopted": False, "F6_exposed": False,
                    "whole_goal_complete": False})
            except Exception as receipt_error:
                primary.add_note("Execution error receipt failed: " + repr(receipt_error))
    if primary is not None:
        raise primary
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.freeze, arguments.freeze_sha256,
                         arguments.source_review, arguments.source_review_sha256,
                         arguments.output), allow_nan=False), flush=True)
