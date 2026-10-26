"""E190: gated, descriptive analysis of saved RM vertical/contact sequences."""
# Failure handlers retain original errors and partial output.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

ENTRY_STARTED = time.perf_counter()
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import psutil
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "docs/BANC_M3_E190_VERTICAL_SEQUENCE_DESIGN_20260924.md"
OWNED = ("tools/analyze_c_banc_m3_vertical_sequence.py", "tests/test_c_banc_vertical_sequence.py")
FREEZE_SCHEMA = "flylab.m3.e190.vertical-sequence.freeze.v2"
REVIEW_SCHEMA = "flylab.m3.e190.source-review.v1"
RESULT_SCHEMA = "flylab.m3.e190.vertical-sequence.result.v1"
EVENTS = (131, 219, 308, 391, 433, 480, 521, 564)
INCLUDED = (219, 308, 391)
RM = 4
SUPPORT = 0.02
LIMIT_S = 25.0
OUTPUT_LIMIT = 33_554_432
FREE_LIMIT = 1_073_741_824
AMENDMENT = "docs/BANC_M3_E190_FORCE_ORDER_AMENDMENT_20260924.md"
INPUT_KEYS = ("native", "derived", "force", "contact", "aligned", "layout", "observed", "force_metadata")
Z_NAMES = ("command", "total", "root", "femur", "tarsus", "own7", "residual")
CONDITIONS = ("all", "negative", "positive")
LAGS = (-2, -1, 0, 1, 2)
SOURCE_FIELDS = {
    "native": ("native_index", "control_tick", "time_s", "rotation_world", "heading_horizontal",
               "force_world_bw", "force_forward_bw", "normal_load_bw", "slip_mm_s", "slip_valid",
               "nonfoot_contact", "upright", "qvel", "step_requested", "integration_complete",
               "state_captured", "state_copy_complete", "shadow_forward_complete", "observer_returned",
               "diagnostics_returned", "nonfoot_returned", "observation_complete", "source_verified"),
    "derived": ("native_load", "native_slip_mm_s", "native_slip_valid", "native_rm_actual_qvel",
                "native_force_forward", "contact_available", "native_z_total", "native_z_root",
                "native_z_own7", "native_z_residual", "native_rm_command_projection_z",
                "trace_control_tick", "trace_time_s", "rm_trace_joint_targets"),
    "force": ("time_s", "available", "finite", "error", "thorax_rotation", "heading_xy",
              "leg_force_bw", "segment_force", "leg_heading_xy_force_bw"),
    "contact": ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid", "nonfoot_contact", "upright"),
    "aligned": ("control_time_s", "joint_target_rad_42"),
    "observed": ("offsets", "native_index", "contact_id", "segment", "leg", "sign", "frame", "wrench"),
}


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def deadline(phase: str) -> None:
    require(time.perf_counter() - ENTRY_STARTED < LIMIT_S, f"E190 internal deadline after {phase}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(relative: str) -> Path:
    require(isinstance(relative, str), "Frozen path must be text")
    part = Path(relative)
    require(not part.is_absolute() and ".." not in part.parts and part.parts, "Unsafe frozen path")
    path = (ROOT / part).resolve()
    require(path.is_relative_to(ROOT), "Frozen path escapes repository")
    return path


def sha_map(mapping: Any) -> dict[str, str]:
    require(isinstance(mapping, dict) and mapping, "Frozen input map missing")
    for name, expected in mapping.items():
        require(isinstance(expected, str) and len(expected) == 64, "Malformed input digest")
        require(digest(safe(name)) == expected, f"Frozen input changed: {name}")
    return mapping


def gate(freeze_path: Path, review_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Finish every source/path/hash check before any np.load call."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(freeze.get("schema") == FREEZE_SCHEMA, "E190 freeze schema")
    require(freeze.get("design") == DESIGN and digest(safe(DESIGN)) == freeze.get("design_sha256"), "E190 design changed")
    require(freeze.get("amendment") == AMENDMENT and digest(safe(AMENDMENT)) == freeze.get("amendment_sha256"),
            "E190 force-order amendment changed")
    require(freeze.get("owned_files") == list(OWNED), "E190 owned source contract")
    limits = freeze.get("resource_limits", {})
    for key, value in {"internal_s": 25, "supervisor_s": 40, "cpu_threads": 1, "gpu": 0,
                       "output_bytes": OUTPUT_LIMIT, "free_bytes": FREE_LIMIT}.items():
        require(limits.get(key) == value, f"E190 resource limit changed: {key}")
    require(freeze.get("prior_cumulative_s") == 4654.9093643997185 and freeze.get("allocation") == 4994,
            "E190 cumulative budget changed")
    require(freeze.get("event_starts") == list(EVENTS) and freeze.get("included_starts") == list(INCLUDED)
            and freeze.get("lag_controls") == list(LAGS), "E190 event/lag contract changed")
    paths = freeze.get("paths")
    require(isinstance(paths, dict) and set(INPUT_KEYS).issubset(paths), "E190 input paths incomplete")
    resolved = {key: safe(paths[key]) for key in INPUT_KEYS}
    frozen = sha_map(freeze.get("inputs_sha256"))
    require(len(frozen) == 18, "E190 frozen input inventory count")
    require(all(paths[key] in frozen for key in INPUT_KEYS), "E190 input digest incomplete")
    freeze_sha = digest(freeze_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    require(review.get("schema") == REVIEW_SCHEMA and review.get("approved") is True,
            "E190 parent source review not approved")
    require(review.get("freeze_sha256") == freeze_sha, "E190 source review freeze mismatch")
    expected_source = {name: digest(safe(name)) for name in OWNED}
    require(review.get("source_sha256") == expected_source, "E190 source review source mismatch")
    return freeze, review, resolved


def array(source: Mapping[str, Any], key: str, shape: tuple[int, ...], *, finite: bool = True) -> np.ndarray:
    value = np.asarray(source[key])
    require(value.shape == shape and value.dtype.kind in "biuf", f"E190 {key} shape/dtype")
    if finite:
        require(np.isfinite(value).all(), f"E190 {key} nonfinite")
    return value


def same(a: Any, b: Any, name: str) -> None:
    require(np.array_equal(a, b, equal_nan=True), f"E190 exact join mismatch: {name}")


def reconstruct_force_orders(observed: Mapping[str, Any], weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay E157 per-contact normalized leg accumulation and E185 raw segment accumulation."""
    require(np.isfinite(weight) and weight > 0, "E190 positive finite body weight")
    offsets = array(observed, "offsets", (10001,))
    require(offsets.dtype.kind in "iu", "E190 integral contact offsets")
    require(offsets[0] == 0 and offsets[-1] == 24836 and np.all(np.diff(offsets) >= 0),
            "E190 observed contact offsets")
    indices = array(observed, "native_index", (24836,))
    same(indices, np.repeat(np.arange(10000), np.diff(offsets)), "E157 observed contact index/offset")
    ids = array(observed, "contact_id", (24836,))
    segments = array(observed, "segment", (24836,))
    legs = array(observed, "leg", (24836,))
    signs = array(observed, "sign", (24836,))
    require(all(values.dtype.kind in "iu" for values in (indices, ids, segments, legs, signs)),
            "E190 integral contact identity")
    frames = array(observed, "frame", (24836, 3, 3))
    wrenches = array(observed, "wrench", (24836, 6))
    require(np.all((segments >= 0) & (segments < 30)) and np.array_equal(legs, segments // 5)
            and np.isin(signs, (-1, 1)).all(), "E190 contact segment/leg/sign")
    for start, stop in pairwise(offsets):
        require(np.all(np.diff(ids[start:stop]) > 0), "E190 observed contact order")
    # A zero normal wrench is still a recorded contact and participates in the original loops.
    e157_legs = np.zeros((10000, 6, 3), dtype=np.float64)
    e185_segments = np.zeros((10000, 30, 3), dtype=np.float64)
    for row in range(24836):
        force = frames[row].T @ wrenches[row, :3]
        native_index, leg, segment, sign = int(indices[row]), int(legs[row]), int(segments[row]), int(signs[row])
        e157_legs[native_index, leg] += sign * force / weight
        e185_segments[native_index, segment] += sign * force
    e185_legs = e185_segments.reshape(10000, 6, 5, 3).sum(axis=2) / weight
    return e157_legs, e185_segments, e185_legs


def validate_join(native: Mapping[str, Any], derived: Mapping[str, Any], force: Mapping[str, Any],
                  contact: Mapping[str, Any], aligned: Mapping[str, Any], layout: Mapping[str, Any],
                  observed: Mapping[str, Any], force_metadata: Mapping[str, Any], body_hash: str) -> dict[str, Any]:
    n = 10000
    same(array(native, "native_index", (n,)), np.arange(n), "E157 index")
    same(array(native, "control_tick", (n,)), 201 + np.arange(n) // 50, "E157 control tick")
    nt = array(native, "time_s", (n,))
    require(np.allclose(nt, 1.0001 + np.arange(n) * 0.0001, atol=2e-8, rtol=0), "E157 native clock")
    ft = array(force, "time_s", (30000,))[10000:20000]
    same(nt, ft, "E157/E185 native time")
    same(nt, array(contact, "time_s", (30000,))[10000:20000], "E157/E185 contact time")
    available = array(force, "available", (30000,))
    finite = array(force, "finite", (30000,))
    errors = np.asarray(force["error"])
    require(available.dtype == finite.dtype == np.bool_ and available[10000:20000].all()
            and finite[10000:20000].all() and errors.shape == (30000,)
            and np.all(errors[10000:20000] == ""), "E185 force availability/finite/error")
    for key in ("step_requested", "integration_complete", "state_captured", "state_copy_complete",
                "shadow_forward_complete", "observer_returned", "diagnostics_returned", "nonfoot_returned",
                "observation_complete", "source_verified"):
        flag = array(native, key, (n,))
        require(flag.dtype == np.bool_ and flag.all(), f"E157 completion flag: {key}")
    inventory = force_metadata.get("inventory", {})
    weight = layout.get("weight0")
    require(type(weight) in (int, float) and np.isfinite(weight) and weight > 0,
            "E157 body weight")
    require(inventory.get("weight0") == weight and layout.get("body_hash") == body_hash
            and inventory.get("body_model_hash") == body_hash, "E157/E185 body/weight identity")
    for first, second in (("rotation_world", "thorax_rotation"),
                          ("heading_horizontal", "heading_xy")):
        same(native[first], force[second][10000:20000], f"E157/E185 {first}/{second}")
    e157_force, e185_segment, e185_force = reconstruct_force_orders(observed, weight)
    same(e157_force, native["force_world_bw"], "E157 contact-normalized force order")
    same(e185_segment, force["segment_force"][10000:20000], "E185 raw segment force order")
    same(e185_force, force["leg_force_bw"][10000:20000], "E185 segment-then-leg force order")
    # Match each original (6,3) @ (3,) call; a batched matrix kernel need not
    # preserve its last-bit reduction order.
    e185_ap = np.stack([segments.reshape(6, 5, 3).sum(axis=1) @ heading
                       for segments, heading in zip(e185_segment, force["heading_xy"][10000:20000])]) / weight
    e157_ap = np.stack([legs @ heading for legs, heading in zip(e157_force, native["heading_horizontal"])])
    same(e185_ap,
         force["leg_heading_xy_force_bw"][10000:20000], "E185 planar AP projection")
    same(e157_ap, native["force_forward_bw"], "E157 planar AP projection")
    for key in ("normal_load_bw", "slip_mm_s", "slip_valid", "nonfoot_contact", "upright"):
        # E185 native-contact records these observables on the full 30000 clock.
        same(native[key], contact[key][10000:20000], f"E157/E185 {key}")
    same(native["normal_load_bw"], derived["native_load"], "E157/E161 load")
    same(native["slip_mm_s"], derived["native_slip_mm_s"], "E157/E161 slip")
    same(native["slip_valid"], derived["native_slip_valid"], "E157/E161 slip validity")
    dofs = np.asarray(layout["leg_dof_addresses"])
    require(dofs.shape == (6, 7) and dofs.dtype.kind in "iu"
            and (dofs >= 0).all() and (dofs < 108).all(), "RM integral layout dimensions")
    same(native["qvel"][:, dofs[RM]],
         derived["native_rm_actual_qvel"], "E157/E161 actual RM qvel")
    same(native["force_forward_bw"], derived["native_force_forward"], "E157/E161 AP force")
    names = layout["leg_joint_names"][RM]
    require(len(names) == 7 and names[4].endswith("rm_trochanterfemur-pitch")
            and names[6].endswith("rm_tarsus1-pitch"), "RM femur/tarsus layout names")
    require(np.asarray(layout["leg_dof_addresses"]).shape == (6, 7), "RM layout dimensions")
    ticks = array(derived, "trace_control_tick", (2001,))[200:401]
    times = array(derived, "trace_time_s", (2001,))[200:401]
    same(ticks, np.arange(200, 401), "E161 trace control ticks")
    require(np.allclose(times, ticks * 0.005, rtol=0, atol=2e-8), "E161 trace time")
    same(array(aligned, "control_time_s", (601,))[200:401], times, "E186 trace time")
    targets = array(derived, "rm_trace_joint_targets", (2001, 7))[200:401]
    require(targets.shape == (201, 7) and np.isfinite(targets).all(), "E161 trace RM target")
    same(targets, array(aligned, "joint_target_rad_42", (601, 42))[200:401, 28:35], "E161/E186 RM target")
    z_ok = np.asarray(derived["contact_available"])
    require(z_ok.shape == (n, 6) and z_ok.dtype == np.bool_, "E161 contact availability")
    for key, shape in (("native_z_total", (n, 6)), ("native_z_root", (n, 6)),
                       ("native_z_own7", (n, 6, 7)), ("native_z_residual", (n, 6)),
                       ("native_rm_command_projection_z", (n, 15))):
        value = np.asarray(derived[key])
        require(value.shape == shape and np.isfinite(value[z_ok] if key != "native_rm_command_projection_z" else value[z_ok[:, RM]]).all(),
                f"E161 {key} shape/available values")
        if key != "native_rm_command_projection_z":
            require(np.isnan(value[~z_ok]).all(), f"E161 {key} missing-contact NaN rule")
    require(np.asarray(derived["native_rm_command_projection_z"])[~z_ok[:, RM]].size == 0 or
            np.isnan(np.asarray(derived["native_rm_command_projection_z"])[~z_ok[:, RM]]).all(),
            "E161 missing command projection NaN rule")
    discrepancy = np.abs(e157_force - e185_force)
    return {"valid": True, "native_rows": n, "native_force_offset": 10000,
            "common_control_ticks": [201, 400], "rm_layout_names": [names[4], names[6]],
            "original_world_force_unequal_elements": int(np.count_nonzero(discrepancy)),
            "original_world_force_max_abs_bw": float(discrepancy.max()),
            "claim_limit": "Compared recorded common observables; E185 does not store full native qpos/qvel."}


def lag_pearson(x: np.ndarray, y: np.ndarray, lag: int) -> dict[str, Any]:
    require(lag in LAGS and x.shape == y.shape == (20,), "E190 lag input")
    if lag >= 0:
        a, b = x[:20-lag], y[lag:]
    else:
        a, b = x[-lag:], y[:20+lag]
    valid = np.isfinite(a) & np.isfinite(b)
    aa, bb = a[valid], b[valid]
    result: dict[str, Any] = {"lag_control": lag, "lag_ms": 5 * lag, "possible_pairs": len(a),
                              "sample_pairs": len(aa), "available": False, "pearson_r": None}
    if len(aa) < 8:
        result["reason"] = "fewer_than_8_pairs"
    elif np.var(aa) <= 0 or np.var(bb) <= 0:
        result["reason"] = "zero_variance"
    else:
        result["available"] = True
        result["pearson_r"] = float(np.corrcoef(aa, bb)[0, 1])
        result["reason"] = None
    return result


def bin_metrics(force: np.ndarray, load: np.ndarray, contact_available: np.ndarray,
                z: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One 50 native sample bin; signs are selected before means."""
    require(force.shape == load.shape == contact_available.shape == (50,), "E190 bin shape")
    require(np.isfinite(force).all() and np.isfinite(load).all() and (load >= 0).all(), "E190 force/load")
    require(contact_available.dtype == np.bool_, "E190 contact mask")
    support = (load >= SUPPORT) & contact_available
    result: dict[str, np.ndarray] = {
        "ap_signed_mean": np.array(force.mean()), "ap_positive_magnitude_mean": np.array(np.maximum(force, 0).mean()),
        "ap_negative_magnitude_mean": np.array(np.maximum(-force, 0).mean()),
        "normal_load_mean": np.array(load.mean()), "ap_zero_count": np.array(np.count_nonzero(force == 0)),
        "support_count": np.array(np.count_nonzero(support)), "contact_available_count": np.array(np.count_nonzero(contact_available)),
    }
    for condition in CONDITIONS:
        mask = support.copy()
        if condition == "negative":
            mask &= force < 0
        elif condition == "positive":
            mask &= force > 0
        count = int(mask.sum())
        denominator = float(load[mask].sum())
        result[f"{condition}_count"] = np.array(count)
        result[f"{condition}_load_denominator"] = np.array(denominator)
        result[f"{condition}_valid"] = np.array(count > 0 and denominator > 0)
        result[f"{condition}_downward_fraction"] = np.array(np.nan)
        for name in Z_NAMES:
            values = np.asarray(z[name])
            require(values.shape == (50,), f"E190 {name} bin shape")
            require(np.isfinite(values[contact_available]).all() and np.isnan(values[~contact_available]).all(),
                    f"E190 {name} availability")
            mean = float(np.dot(load[mask], values[mask]) / denominator) if denominator > 0 else np.nan
            result[f"{condition}_{name}_z"] = np.array(mean)
            if name in ("command", "total", "femur", "tarsus"):
                result[f"{condition}_{name}_downward_fraction"] = np.array(float(np.mean(values[mask] < 0)) if count else np.nan)
        if count:
            result[f"{condition}_downward_fraction"] = np.array(float(np.mean(z["femur"][mask] + z["tarsus"][mask] < 0)))
    return result


def analyze(native: Mapping[str, Any], derived: Mapping[str, Any], force: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    idx = np.arange(10000)
    load = np.asarray(native["normal_load_bw"])[:, RM]
    ap = np.asarray(force["leg_heading_xy_force_bw"])[10000:20000, RM]
    available = np.asarray(derived["contact_available"])[:, RM]
    own = np.asarray(derived["native_z_own7"])[:, RM]
    z = {
        "command": np.asarray(derived["native_rm_command_projection_z"]).sum(axis=1),
        "total": np.asarray(derived["native_z_total"])[:, RM],
        "root": np.asarray(derived["native_z_root"])[:, RM],
        "femur": own[:, 4], "tarsus": own[:, 6],
        "own7": own.sum(axis=1), "residual": np.asarray(derived["native_z_residual"])[:, RM],
    }
    # np.sum propagates NaN from unavailable contact, preserving the native mask.
    raw = {"native_index": idx, "native_time_s": np.asarray(native["time_s"]),
           "native_control_tick": np.asarray(native["control_tick"]), "native_ap_force_bw": ap,
           "native_e157_ap_force_bw": np.asarray(native["force_forward_bw"])[:, RM],
           "native_normal_load_bw": load, "native_contact_available": available,
           "native_support": (load >= SUPPORT) & available,
           "native_slip_mm_s": np.asarray(native["slip_mm_s"])[:, RM],
           "native_slip_valid": np.asarray(native["slip_valid"])[:, RM],
           "native_nonfoot_contact": np.asarray(native["nonfoot_contact"]),
           "native_upright": np.asarray(native["upright"])}
    raw.update({f"native_{name}_z": value for name, value in z.items() if name != "own7"})
    raw["native_own7_sum_z"] = z["own7"]
    raw["native_own7_components_z"] = own
    event_idx = np.asarray([np.arange((c - 10 - 201) * 50, (c + 10 - 201) * 50) for c in INCLUDED], dtype=np.int64)
    require(event_idx.shape == (3, 1000), "E190 fixed event window")
    require(np.array_equal(np.asarray(native["control_tick"])[event_idx[:, 0]], np.asarray(INCLUDED) - 10)
            and np.array_equal(np.asarray(native["control_tick"])[event_idx[:, -1]], np.asarray(INCLUDED) + 9),
            "E190 event boundary shift")
    raw["event_native_index"] = event_idx
    raw["event_start_control_tick"] = np.asarray(INCLUDED)
    rows: list[dict[str, np.ndarray]] = []
    lag_rows: list[dict[str, Any]] = []
    for event, positions in zip(INCLUDED, event_idx):
        bins = []
        for b in range(20):
            p = positions[b * 50:(b + 1) * 50]
            bins.append(bin_metrics(ap[p], load[p], available[p], {key: value[p] for key, value in z.items()}))
        rows.append({key: np.stack([row[key] for row in bins]) for key in bins[0]})
        brake = np.asarray([float(np.maximum(-ap[positions[b*50:(b+1)*50]], 0).mean()) for b in range(20)])
        command = rows[-1]["all_command_z"].astype(float)
        foot = rows[-1]["all_femur_z"].astype(float) + rows[-1]["all_tarsus_z"].astype(float)
        lag_rows.append({"event_control_tick": event, "definition": "x(t) versus braking(t+lag); positive lag means x earlier",
                         "command_z": [lag_pearson(command, brake, lag) for lag in LAGS],
                         "femur_plus_tarsus_z": [lag_pearson(foot, brake, lag) for lag in LAGS]})
    raw.update({"bin_" + key: np.stack([row[key] for row in rows]) for key in rows[0]})
    raw["bin_control_tick"] = np.asarray([np.arange(c-10, c+10) for c in INCLUDED])
    raw["bin_braking_magnitude_mean"] = raw["bin_ap_negative_magnitude_mean"]
    raw["lag_control"] = np.asarray(LAGS)
    for signal in ("command_z", "femur_plus_tarsus_z"):
        raw[f"lag_{signal}_r"] = np.asarray([[np.nan if row["pearson_r"] is None else row["pearson_r"]
                                               for row in event[signal]] for event in lag_rows])
        raw[f"lag_{signal}_available"] = np.asarray([[row["available"] for row in event[signal]] for event in lag_rows])
        raw[f"lag_{signal}_pairs"] = np.asarray([[row["sample_pairs"] for row in event[signal]] for event in lag_rows])
    summary = {"events": list(INCLUDED), "excluded_events": [c for c in EVENTS if c not in INCLUDED],
               "excluded_reason": "pre/post 10 control ticks not wholly inside common 201..400 interval",
               "event_window": "pre c-10..c-1; post c..c+9; 50 native samples per tick",
               "lag_results": lag_rows, "condition_order": list(CONDITIONS),
               "interpretation": "Descriptive saved-array timing only. Contact projection is not individual muscle force; ordering is not causation or candidate success. Events overlap in biological history and are not independent biological samples.",
               "LTM78": "Separate tendon pathway, absent from the 15-muscle position-moment sum."}
    return raw, summary


def load_npz(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        require(set(fields).issubset(source.files), f"E190 missing source fields: {path}")
        return {key: source[key] for key in fields}


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def run(freeze_path: Path, review_path: Path, output: Path) -> dict[str, Any]:
    require(output.is_absolute() and output.resolve().is_relative_to(ROOT), "E190 output must be absolute inside repository")
    require(not output.exists(), "E190 output already exists")
    require(shutil.disk_usage(output.parent).free >= FREE_LIMIT, "E190 free disk below 1 GiB")
    output.mkdir()
    result: dict[str, Any] = {"schema": RESULT_SCHEMA, "valid": False, "publication_complete": False,
                              "candidate_adopted": False, "whole_goal_complete": False, "F6_exposed": False,
                              "comparison_status": "not-tested", "primary_error": None, "secondary_errors": [],
                              "runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                          "threadpools": threadpool_info(),
                                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    paths: dict[str, Path] = {}
    frozen: dict[str, Any] | None = None
    try:
        frozen, review, paths = gate(freeze_path, review_path)
        deadline("source gate")
        sources = {key: load_npz(paths[key], SOURCE_FIELDS[key]) for key in SOURCE_FIELDS}
        layout = json.loads(paths["layout"].read_text(encoding="utf-8"))
        force_metadata = json.loads(paths["force_metadata"].read_text(encoding="utf-8"))
        deadline("input load")
        joined = validate_join(*[sources[k] for k in ("native", "derived", "force", "contact", "aligned")],
                               layout, sources["observed"], force_metadata, frozen["body_hash"])
        result["comparison_status"] = "valid"
        result["exact_comparisons"] = joined
        deadline("exact comparisons")
        arrays, summary = analyze(sources["native"], sources["derived"], sources["force"])
        result.update(summary)
        result["valid"] = True
        deadline("analysis")
        with (output / "vertical-sequence.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        require(output_bytes(output) <= OUTPUT_LIMIT, "E190 output exceeds 32 MiB")
        deadline("NPZ persistence")
        sha_map(frozen["inputs_sha256"])
        require(digest(safe(DESIGN)) == frozen["design_sha256"] and
                digest(safe(AMENDMENT)) == frozen["amendment_sha256"], "E190 postrun design changed")
        require(review["source_sha256"] == {name: digest(safe(name)) for name in OWNED}, "E190 postrun source changed")
        deadline("postrun source check")
    except Exception as exc:
        result["valid"] = False
        result["primary_error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        result["elapsed_s"] = time.perf_counter() - ENTRY_STARTED
        try:
            deadline("pre-result persistence")
            result["publication_complete"] = bool(result["valid"])
            save_json(output / "result.json", result)
            require(output_bytes(output) <= OUTPUT_LIMIT, "E190 final output exceeds 32 MiB")
            deadline("result persistence")
        except Exception as exc:
            result["valid"] = False
            result["publication_complete"] = False
            if result["primary_error"] is None:
                result["primary_error"] = {"type": type(exc).__name__, "message": str(exc),
                                           "traceback": traceback.format_exc()}
            result["secondary_errors"].append({"phase": "result persistence", "type": type(exc).__name__, "message": str(exc)})
            try:
                stale = output / "result.json"
                if stale.exists():
                    stale.unlink()
                pending = output / "result.json.tmp"
                if pending.exists():
                    pending.unlink()
                save_json(stale, result)
            except Exception as fallback_error:
                result["secondary_errors"].append({"phase": "failure persistence", "type": type(fallback_error).__name__,
                                                   "message": str(fallback_error)})
                print(json.dumps({"E190_failure": result["primary_error"],
                                  "secondary_errors": result["secondary_errors"]}, allow_nan=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.review, args.output)
    print(json.dumps({"valid": result["valid"], "comparison_status": result["comparison_status"],
                      "elapsed_s": result["elapsed_s"], "primary_error": result["primary_error"]}, allow_nan=False))
    return 0 if result["publication_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
