"""E249 saved LH input-group and local-slope diagnostic; no neural advance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.c.storage import StateStore
from tools import analyze_c_banc_m3_signal_discrimination as prior
from tools.analyze_c_banc_m3_saved_current import warp_current

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = (BASE / "e249-lh-input-slope").resolve()
OWNED = ("tools/analyze_c_banc_m3_lh_input_slope.py",
         "tests/test_c_banc_lh_input_slope.py")
STATES = ("baseline", "intact_low", "intact_high", "bridge_cut_low",
          "bridge_cut_high", "sensory_cut_low", "sensory_cut_high",
          "disk_intact_low")
GROUPS = ("source", "bridge", "other")
FREEZE_SHA = "f58ca44dc57f3a32988c5d3970c3fca0918d359a35a35af153a1bec5c15c99d9"
NETWORK_IDENTITY = "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
RESOURCE = {"internal_s": 30, "finalization_reserve_s": 5,
    "external_s": 40, "supervisor_s": 45, "cpu_threads": 1, "gpu_count": 0,
    "original_attempts": 1, "retries": 0, "body_constructors": 0,
    "neural_constructors": 0, "fit_calls": 0, "solver_calls": 0,
    "resident_bytes": 2147483648, "output_bytes": 67108864,
    "minimum_free_bytes": 536870912}


def state_directory(relative):
    path = (ROOT / relative).resolve()
    if (not isinstance(relative, str) or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(ROOT) or not path.is_dir()):
        raise ValueError("E249 unsafe/missing state directory")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or prior.sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E249 freeze SHA changed")
    if prior.sha(review_path) != review_sha:
        raise ValueError("E249 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != "flylab.m3.e249.lh-input-slope.freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != RESOURCE
            or list(freeze.get("states", {})) != list(STATES)
            or freeze.get("network_identity") != NETWORK_IDENTITY
            or freeze.get("n") != 158706
            or freeze.get("declared_edges") != 11584852
            or freeze.get("saturation_relative_slope") != .1
            or freeze.get("saturation_bridge_count") != 3
            or freeze.get("small_direct_normalized_rms") != 1e-4
            or len(freeze.get("selection", {}).get("source_rows", [])) != 23
            or len(freeze["selection"].get("bridge_rows", [])) != 4):
        raise ValueError("E249 frozen scientific/resource contract changed")
    own = {name: prior.sha(prior.pinned_path(name)) for name in OWNED}
    if (review.get("schema") != "flylab.m3.e249.lh-input-slope.review.v1"
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E249 parent source review invalid")
    if freeze.get("packages") != {name: version(name) for name in freeze["packages"]}:
        raise ValueError("E249 package versions changed")
    if prior.sha(prior.pinned_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E249 design changed")
    for hashes in (freeze["input_sha256"], freeze["reused_source_sha256"],
                   freeze["preservation_sha256"],
                   review.get("additional_reused_source_sha256"),
                   review.get("additional_input_sha256")):
        if not isinstance(hashes, dict):
            raise TypeError("E249 required hash map missing")
        for name, expected in hashes.items():
            if prior.sha(prior.pinned_path(name)) != expected:
                raise ValueError("E249 pinned input/source changed: " + name)
    paths = {key: prior.pinned_path(name) for key, name in freeze["paths"].items()}
    if set(paths) != {"native_csr", "configuration", "roster", "parameters",
                      "groups", "e248_freeze", "e248_review", "e248_output_review"}:
        raise ValueError("E249 frozen input roles changed")
    directories = {name: state_directory(freeze["states"][name]) for name in STATES}
    return freeze, review, paths, directories


def load(paths, directories, *, progress=None):
    values = {}
    for role, keys in (("native_csr", ("indptr", "indices", "data")),
                       ("configuration", ("membrane_tau",)),
                       ("parameters", ("motor_ids", "free_columns", "fixed_columns",
                                       "membership"))):
        with np.load(paths[role], allow_pickle=False) as saved:
            values[role] = {key: saved[key].copy() for key in keys}
        if progress:
            progress(role)
    for role in ("roster", "groups", "e248_freeze", "e248_review",
                 "e248_output_review"):
        value = json.loads(paths[role].read_text(encoding="utf-8"))
        values[role] = value["groups"] if role == "groups" else value
        if progress:
            progress(role)
    values["states"] = {}
    for name, directory in directories.items():
        values["states"][name] = StateStore.load(directory)
        if progress:
            progress("state:" + name)
    return values


def validate(inputs, freeze, *, strict_original=True):
    csr = inputs["native_csr"]
    ptr = np.asarray(csr["indptr"])
    indices = np.asarray(csr["indices"])
    weight = np.asarray(csr["data"])
    n = len(ptr)-1
    roster = inputs["roster"]
    params = inputs["parameters"]
    source = np.asarray(freeze["selection"]["source_rows"], np.int64)
    bridge = np.asarray(freeze["selection"]["bridge_rows"], np.int64)
    motor_rows = np.asarray(roster["motor_node_rows"], np.int64)
    free = np.asarray(params["free_columns"], np.int64)
    fixed = np.asarray(params["fixed_columns"], np.int64)
    owner = np.asarray(params["membership"], np.int64)
    tau_all = np.asarray(inputs["configuration"]["membrane_tau"])
    lh_columns = np.asarray([column for column in free
        if inputs["groups"][int(owner[column])]["leg"] == "lh"], np.int64)
    motor = np.sort(motor_rows[lh_columns])
    rows = np.r_[np.sort(source), np.sort(bridge), motor]
    if (ptr.shape != (n+1,) or ptr[0] != 0 or ptr[-1] != len(indices)
            or len(weight) != len(indices) or np.any(np.diff(ptr) < 0)
            or np.any((indices < 0) | (indices >= n))
            or not np.isfinite(weight).all()
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(len(motor_rows)))
            or not np.array_equal(params["motor_ids"], motor_rows)
            or owner.shape != (len(motor_rows),)
            or len(inputs["groups"]) != 96
            or tau_all.shape != (n,)
            or len(np.unique(rows)) != len(rows)
            or np.any((rows < 0) | (rows >= n))
            or not np.all(tau_all[rows] > 0)):
        raise ValueError("E249 CSR/anatomy/tau identity changed")
    if strict_original and (n != 158706 or len(indices) != 11584852
            or weight.dtype != np.float32
            or len(motor_rows) != 391 or len(free) != 343 or len(fixed) != 48
            or len(source) != 23 or len(bridge) != 4 or len(motor) != 56
            or len(rows) != 83):
        raise ValueError("E249 frozen original dimensions changed")
    e248 = inputs["e248_freeze"]
    if (not np.array_equal(e248["selection"]["source_rows"], source)
            or not np.array_equal(e248["selection"]["bridge_rows"], bridge)
            or not np.array_equal(e248["selection"]["reachable_motor_rows"],
                                  freeze["selection"]["reachable_motor_rows"])):
        raise ValueError("E249 original E248 selection changed")
    if strict_original and (inputs["e248_output_review"].get("validity") !=
            "independently_valid" or inputs["e248_review"].get("validity") != "valid"):
        raise ValueError("E249 valid E248 parent source/output reviews required")
    for name in STATES:
        state = inputs["states"][name]
        expected_tick = 40000 if name == "baseline" else 40800
        if (state.get("model_hash") != NETWORK_IDENTITY
                or state.get("tick") != expected_tick
                or any(np.asarray(state[key]).shape != (n,) for key in
                       ("rate", "drive", "adaptation", "output_mask"))
                or (strict_original and (np.asarray(state["rate"]).dtype != np.float32
                    or np.asarray(state["output_mask"]).dtype != np.float32))
                or any(not np.isfinite(state[key]).all() for key in
                       ("rate", "drive", "adaptation", "output_mask"))
                or np.any(state["rate"] < 0) or np.any(state["adaptation"] < 0)
                or not np.isin(state["output_mask"], [0, 1]).all()):
            raise ValueError("E249 saved network state identity/finite changed: " + name)
    node_by_row = {node["node_row"]: node for node in roster["nodes"]}
    if any(int(row) not in node_by_row for row in rows):
        raise ValueError("E249 selected node annotation missing")
    return rows, np.r_[np.zeros(len(source), np.int8),
                       np.ones(len(bridge), np.int8),
                       np.full(len(motor), 2, np.int8)]


def analyze_arrays(csr, configuration, states, source_rows, bridge_rows,
                   motor_rows, *, raw=None, progress=None, strict_original=True):
    """Pure saved-state attribution, with caller-owned completed-state prefix."""
    raw = {} if raw is None else raw
    source = np.sort(np.asarray(source_rows, np.int64))
    bridge = np.sort(np.asarray(bridge_rows, np.int64))
    motor = np.sort(np.asarray(motor_rows, np.int64))
    rows = np.r_[source, bridge, motor]
    if (len(np.unique(rows)) != len(rows)
            or list(states) != list(STATES)
            or (strict_original and (len(source), len(bridge), len(motor)) != (23, 4, 56))):
        raise ValueError("E249 ordered disjoint postsynaptic selection required")
    ptr = np.asarray(csr["indptr"], np.int64)
    indices = np.asarray(csr["indices"], np.int64)
    weights = np.asarray(csr["data"], np.float32)
    tau = np.asarray(configuration["membrane_tau"], np.float64)[rows]
    if np.any(tau <= 0) or not np.isfinite(tau).all():
        raise ValueError("E249 positive finite actual membrane tau required")
    offsets = np.r_[0, np.cumsum(ptr[rows+1] - ptr[rows])].astype(np.int64)
    edge_id = np.concatenate([np.arange(ptr[row], ptr[row+1], dtype=np.int64)
                              for row in rows])
    pre = indices[edge_id]
    post = np.repeat(rows, np.diff(offsets))
    edge_row_index = np.repeat(np.arange(len(rows)), np.diff(offsets))
    group = np.full(len(edge_id), 2, np.int8)
    group[np.isin(pre, source)] = 0
    group[np.isin(pre, bridge)] = 1
    raw.update(state_names=np.asarray(STATES), row_ids=rows,
        row_group=np.r_[np.zeros(len(source), np.int8),
                        np.ones(len(bridge), np.int8),
                        np.full(len(motor), 2, np.int8)],
        group_names=np.asarray(GROUPS), edge_id=edge_id,
        edge_pre=pre, edge_post=post, edge_row_index=edge_row_index,
        edge_native_weight=weights[edge_id].copy(), edge_group=group,
        edge_contribution=np.full((8, len(edge_id)), np.nan),
        signed=np.full((8, len(rows), 3), np.nan),
        positive=np.full((8, len(rows), 3), np.nan),
        negative=np.full((8, len(rows), 3), np.nan),
        gross=np.full((8, len(rows), 3), np.nan),
        state_tick=np.full(8, -1, np.int64),
        completed_states=np.asarray(0, np.int64))
    for field in ("current", "drive", "adaptation", "rate", "margin", "phi",
                  "slope", "dr_dt"):
        raw[field] = np.full((8, len(rows)), np.nan)
    for field in ("active", "rectified", "saturated"):
        raw[field] = np.zeros((8, len(rows)), bool)
    raw["warp_bridge_current32"] = np.full((8, len(bridge)), np.nan, np.float32)
    raw["warp_minus_math"] = np.full((8, len(bridge)), np.nan)
    for state_index, name in enumerate(STATES):
        state = states[name]
        rate = np.asarray(state["rate"], np.float32)
        mask = np.asarray(state["output_mask"], np.float32)
        drive = np.asarray(state["drive"], np.float64)[rows]
        adaptation = np.asarray(state["adaptation"], np.float64)[rows]
        if (rate.shape != (len(ptr)-1,) or mask.shape != rate.shape
                or not np.isfinite(rate).all() or not np.isfinite(mask).all()
                or not np.isfinite(drive).all() or not np.isfinite(adaptation).all()
                or np.any(adaptation < 0) or np.any(rate < 0)
                or not np.isin(mask, [0, 1]).all()):
            raise ValueError("E249 saved state finite/shape/binary mismatch: " + name)
        term = (weights[edge_id].astype(np.float64) *
                (rate[pre].astype(np.float64) * mask[pre].astype(np.float64)))
        current = np.zeros(len(rows), np.float64)
        signed = np.zeros((len(rows), 3), np.float64)
        positive = np.zeros_like(signed)
        negative = np.zeros_like(signed)
        gross = np.zeros_like(signed)
        for row_index in range(len(rows)):
            first, last = offsets[row_index:row_index+2]
            terms = term[first:last]
            current[row_index] = np.sum(terms, dtype=np.float64)
            for category in range(3):
                selected = terms[group[first:last] == category]
                signed[row_index, category] = np.sum(selected, dtype=np.float64)
                positive[row_index, category] = np.sum(selected[selected > 0],
                                                       dtype=np.float64)
                negative[row_index, category] = np.sum(selected[selected < 0],
                                                       dtype=np.float64)
                gross[row_index, category] = np.sum(np.abs(selected), dtype=np.float64)
        if not np.allclose(signed.sum(axis=1), current, rtol=1e-12, atol=1e-8):
            raise ValueError("E249 source/bridge/other partition fails current reconstruction")
        margin = drive + current - 7.5 - adaptation
        activation = np.tanh(margin / 200.)
        phi = np.maximum(200. * activation, 0.)
        active = margin > 0
        slope = np.where(active, 1. - activation**2, 0.)
        dr_dt = (phi - rate[rows].astype(np.float64)) / tau
        raw["state_tick"][state_index] = int(state["tick"])
        raw["edge_contribution"][state_index] = term
        for field, values in (("signed", signed), ("positive", positive),
                              ("negative", negative), ("gross", gross),
                              ("current", current), ("drive", drive),
                              ("adaptation", adaptation),
                              ("rate", rate[rows].astype(np.float64)),
                              ("margin", margin), ("phi", phi),
                              ("slope", slope), ("dr_dt", dr_dt),
                              ("active", active), ("rectified", ~active),
                              ("saturated", active & (slope <= .1))):
            raw[field][state_index] = values
        for bridge_index in range(len(bridge)):
            row_index = len(source) + bridge_index
            first, last = offsets[row_index:row_index+2]
            warp = warp_current(weights[edge_id[first:last]].astype(np.float32),
                rate[pre[first:last]].astype(np.float32),
                mask[pre[first:last]].astype(np.float32), mode="packed")
            raw["warp_bridge_current32"][state_index, bridge_index] = warp
            raw["warp_minus_math"][state_index, bridge_index] = (
                float(warp) - current[row_index])
        raw["completed_states"] = np.asarray(state_index+1, np.int64)
        if progress:
            progress(name, state_index+1)
    bridge_slice = slice(len(source), len(source)+len(bridge))
    contrasts = {field: raw[field][2, bridge_slice] - raw[field][1, bridge_slice]
                 for field in ("current", "drive", "adaptation", "rate",
                               "margin", "phi")}
    contrasts["source_signed"] = (raw["signed"][2, bridge_slice, 0] -
                                   raw["signed"][1, bridge_slice, 0])
    for field, values in contrasts.items():
        raw["contrast_" + field + "_delta"] = values.copy()
    source_edge = np.flatnonzero((group == 0) & np.isin(post, bridge))
    source_edge_delta = (raw["edge_contribution"][2, source_edge] -
                         raw["edge_contribution"][1, source_edge])
    raw["source_bridge_edge_ids"] = edge_id[source_edge]
    raw["source_bridge_edge_delta"] = source_edge_delta
    cancellation_signed = np.zeros(len(bridge), np.float64)
    cancellation_gross = np.zeros(len(bridge), np.float64)
    for j, row in enumerate(bridge):
        selected = source_edge_delta[post[source_edge] == row]
        cancellation_signed[j] = np.sum(selected, dtype=np.float64)
        cancellation_gross[j] = np.sum(np.abs(selected), dtype=np.float64)
    if not np.allclose(cancellation_signed, contrasts["source_signed"],
                       rtol=1e-12, atol=1e-8):
        raise ValueError("E249 source-edge delta sum mismatch")
    zero_gross = cancellation_gross == 0
    cancellation_ratio = np.divide(np.abs(cancellation_signed), cancellation_gross,
                                   out=np.zeros_like(cancellation_gross),
                                   where=~zero_gross)
    raw["cancellation_signed"] = cancellation_signed
    raw["cancellation_gross"] = cancellation_gross
    raw["cancellation_net_to_gross"] = cancellation_ratio
    raw["cancellation_zero_gross"] = zero_gross
    both_saturated = raw["saturated"][1, bridge_slice] & raw["saturated"][2, bridge_slice]
    saturation_count = int(np.count_nonzero(both_saturated))
    source_rms_normalized = float(np.sqrt(np.mean((contrasts["source_signed"]/200.)**2)))
    summary = {"schema": "flylab.m3.e249.lh-input-slope.summary.v1",
        "H249_saturation": bool(saturation_count >= 3),
        "H249_small_direct_input": bool(source_rms_normalized <= 1e-4),
        "same_bridge_saturated_both_intact_count": saturation_count,
        "same_bridge_saturated_both_intact_rows": bridge[both_saturated].tolist(),
        "source_bridge_delta_rms_normalized_cap200": source_rms_normalized,
        "saturation_relative_slope_limit": .1,
        "small_direct_normalized_rms_limit": 1e-4,
        "source_edge_cancellation_ratio_definition":
            "abs(signed_delta_sum)/sum(abs(edge_delta)); zero when gross zero",
        "bridge_rows": bridge.tolist(),
        "bridge": [{"node_row": int(row),
                    "active_low": bool(raw["active"][1, bridge_slice][j]),
                    "active_high": bool(raw["active"][2, bridge_slice][j]),
                    "slope_low": float(raw["slope"][1, bridge_slice][j]),
                    "slope_high": float(raw["slope"][2, bridge_slice][j]),
                    "source_input_delta": float(contrasts["source_signed"][j]),
                    "source_edge_delta_gross": float(cancellation_gross[j]),
                    "source_edge_net_to_gross": float(cancellation_ratio[j]),
                    "source_edge_zero_gross": bool(zero_gross[j]),
                    "observed_rate_delta": float(contrasts["rate"][j]),
                    "static_phi_delta": float(contrasts["phi"][j])}
                   for j, row in enumerate(bridge)],
        "mathematical_saved_state_only": True,
        "stage_current_observed": False,
        "causal_edge_effect_proven": False,
        "natural_gait_success": False}
    return summary, raw


def annotations_for(roster, rows):
    nodes = {int(node["node_row"]): node for node in roster["nodes"]}
    return {"schema": "flylab.m3.e249.lh-input-slope.annotations.v1",
            "source": "E245 frozen validated roster nodes/source_annotations",
            "nodes": [nodes[int(row)] for row in rows]}


def invalidate_support(result, summary):
    result["H249_saturation"] = None
    result["H249_small_direct_input"] = None
    result["verdict_status"] = "not_tested"
    if summary:
        summary["H249_saturation"] = None
        summary["H249_small_direct_input"] = None
        summary["verdict_status"] = "not_tested"


def publish(out, result, summary, raw, annotations, ledger, *, elapsed, deadline):
    if raw:
        prior.write_npz_once(out, "raw.npz", raw)
    if annotations:
        prior.save_json(out / "annotations.json", annotations)
    if summary:
        prior.save_json(out / "summary.json", summary)
    ledger["process_wall_s"] = elapsed()
    prior.save_json(out / "errors.json", ledger["errors"])
    prior.save_json(out / "ledger.json", ledger)
    files = {p.name: {"sha256": prior.sha(p), "bytes": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file()
             and p.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": "flylab.m3.e249.lh-input-slope.manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_bytes = prior.json_bytes(manifest)
    result_bytes = prior.json_bytes(result)
    if manifest["total_bytes"] + len(manifest_bytes) + len(result_bytes) > RESOURCE["output_bytes"]:
        raise OSError("E249 output exceeds 64 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E249 deadline before manifest publication")
    (out / "output-manifest.json").write_bytes(manifest_bytes)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E249 deadline before result publication")
    (out / "result.json").write_bytes(result_bytes)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E249 deadline after result publication")


def settle(out, result, summary, raw, annotations, ledger, *, elapsed, deadline):
    first = None
    started = time.perf_counter()
    for attempt in range(1, 4):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, summary, raw, annotations, ledger,
                    elapsed=elapsed, deadline=deadline)
            result["publication_wall_s"] = time.perf_counter() - started
            result["process_wall_s"] = elapsed()
            prior.save_json(out / "result.json", result)
            if time.perf_counter() >= deadline:
                raise TimeoutError("E249 deadline after final time receipt")
            return first
        except BaseException as error:  # noqa: BLE001 - retain first and later errors
            if first is None:
                first = error
            ledger["errors"].append(prior.record_error(error, "publication",
                                                        attempt=attempt))
            result["validity"] = "incomplete_or_invalid"
            invalidate_support(result, summary)
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            if time.perf_counter() >= deadline:
                deadline = float("inf")
    result["finalization_pending"] = False
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, summary, raw, annotations, ledger,
                elapsed=elapsed, deadline=float("inf"))
        return first
    except BaseException as error:  # noqa: BLE001 - explicit pending receipt
        ledger["errors"].append(prior.record_error(error, "terminal_publication"))
        result["finalization_pending"] = True
        result["errors"] = ledger["errors"]
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
            if summary:
                try:
                    prior.save_json(out / "summary.json", summary)
                except BaseException as summary_error:  # noqa: BLE001 - remove stale verdict
                    (out / "summary.json").unlink(missing_ok=True)
                    ledger["errors"].append(prior.record_error(
                        summary_error, "terminal_summary"))
            prior.save_json(out / "publication-failure.json", ledger["errors"])
            prior.save_json(out / "result.json", result)
        except BaseException as secondary:  # noqa: BLE001 - retain both failures
            ledger["errors"].append(prior.record_error(secondary, "terminal_receipt"))
    return first


def run(freeze_path, freeze_sha, review_path, review_sha, output,
        *, generated_test_output=False):
    began = time.perf_counter()
    out = Path(output).resolve()
    if not generated_test_output and out != OUT:
        raise ValueError("E249 original output must use reserved directory")
    if out.exists():
        raise FileExistsError("E249 one-shot output already exists")
    out.mkdir(parents=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    process = psutil.Process()
    ledger = {"schema": "flylab.m3.e249.lh-input-slope.ledger.v1",
        "phase": "gate", "loaded_input_roles": [], "analysis_requested_count": 0,
        "analysis_entered_count": 0, "analysis_completed_count": 0,
        "original_calculation_count": 0, "completed_states": 0,
        "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
        "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
        "neural_constructors": 0, "process_id": process.pid,
        "child_process_ids_start": [child.pid for child in process.children(recursive=True)],
        "retries": 0, "errors": []}
    result = {"schema": "flylab.m3.e249.lh-input-slope.result.v1",
        "validity": "incomplete_or_invalid", "verdict_status": "not_tested",
        "H249_saturation": None, "H249_small_direct_input": None,
        "candidate_adopted": False, "F2_complete": False, "F6_exposed": False,
        "whole_goal_complete": False, "active_UI_changed": False,
        "finalization_pending": True, "model_or_stage_advanced": False,
        "causal_edge_effect_proven": False, "natural_gait_success": False}
    raw = {}
    summary = {}
    annotations = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter() - began

    def progress(name, count):
        ledger["completed_states"] = count
        if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
            raise TimeoutError("E249 analysis exhausted prepublication deadline")
        if process.memory_info().rss > RESOURCE["resident_bytes"]:
            raise MemoryError("E249 RSS exceeds 2 GiB")

    try:
        freeze, review, paths, directories = gate(
            freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "additional_reused_source_sha256": review["additional_reused_source_sha256"],
            "additional_input_sha256": review["additional_input_sha256"],
            "packages": freeze["packages"],
            "network_identity": freeze["network_identity"]}
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E249 free disk below 512 MiB")
        ledger["phase"] = "input"
        ledger["threadpool_before"] = threadpool_info()
        limiter = None
        work_error = None

        def loaded(role):
            ledger["loaded_input_roles"].append(role)
            if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
                raise TimeoutError("E249 input deadline")

        try:
            limiter = threadpool_limits(limits=1)
            observed = threadpool_info()
            ledger["threadpool_observed"] = observed
            if any(pool.get("num_threads") != 1 for pool in observed):
                raise RuntimeError("E249 imported threadpool exceeds one thread")
            inputs = load(paths, directories, progress=loaded)
            rows, _ = validate(inputs, freeze)
            annotations = annotations_for(inputs["roster"], rows)
            owner = np.asarray(inputs["parameters"]["membership"], np.int64)
            motor_rows = np.asarray(inputs["roster"]["motor_node_rows"], np.int64)
            free = np.asarray(inputs["parameters"]["free_columns"], np.int64)
            lh = np.sort(motor_rows[[column for column in free
                if inputs["groups"][int(owner[column])]["leg"] == "lh"]])
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            summary, raw = analyze_arrays(inputs["native_csr"],
                inputs["configuration"], inputs["states"],
                freeze["selection"]["source_rows"],
                freeze["selection"]["bridge_rows"], lh,
                raw=raw, progress=progress)
            ledger["analysis_completed_count"] = 1
        except BaseException as error:  # noqa: BLE001 - retain analysis and restore separately
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:  # noqa: BLE001 - retain restore separately
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["summary"] = summary
        result["H249_saturation"] = summary["H249_saturation"]
        result["H249_small_direct_input"] = summary["H249_small_direct_input"]
        result["verdict_status"] = "tested_saved_endpoint_only"
        result["validity"] = "valid_saved_endpoint_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:  # noqa: BLE001 - preserve first work failure
        primary = error
        invalidate_support(result, summary)
        ledger["errors"].append(prior.record_error(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append({"stage": "threadpool_restore",
                "type": type(restore_error).__name__, "message": str(restore_error),
                "traceback": "".join(traceback.format_exception(
                    type(restore_error), restore_error, restore_error.__traceback__))})
    if raw:
        ledger["completed_states"] = int(raw["completed_states"])
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
        analysis_requested_count=ledger["analysis_requested_count"],
        analysis_entered_count=ledger["analysis_entered_count"],
        analysis_completed_count=ledger["analysis_completed_count"],
        original_calculation_count=ledger["original_calculation_count"],
        completed_states=ledger["completed_states"],
        fit_calls=0, solver_calls=0, cuda_calls=0, full_banc_calls=0,
        body_constructors=0, physics_steps=0, neural_constructors=0,
        errors=ledger["errors"], process_wall_s=elapsed())
    ledger["child_process_ids_end"] = [child.pid for child in process.children(recursive=True)]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle(out, result, summary, raw, annotations, ledger,
        elapsed=elapsed, deadline=began+RESOURCE["internal_s"])
    if primary is None:
        primary = publication_error
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
