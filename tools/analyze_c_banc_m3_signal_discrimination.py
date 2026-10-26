"""E246 saved-state nearest-neighbor signal discrimination diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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
from tools.analyze_c_banc_m3_matched_generalization import anatomy_weights

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = (BASE / "e246-signal-discrimination").resolve()
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
REPS = ("R0", "R1", "R2", "R3", "R4")
FREEZE_SHA = "e5ebcc3f6f570fe429515e2a7411bfe83af2cc0ad30ccaa737307a518c737d1e"
FREEZE_SCHEMA = "flylab.m3.e246.signal-discrimination.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e246.signal-discrimination.review.v1"
OWNED = ("tools/analyze_c_banc_m3_signal_discrimination.py",
         "tests/test_c_banc_signal_discrimination.py")
RESOURCE = {"internal_s": 40, "finalization_reserve_s": 5,
    "external_s": 50, "supervisor_s": 60, "cpu_threads": 1, "gpu_count": 0,
    "original_attempts": 1, "retries": 0, "fit_calls": 0, "solver_calls": 0,
    "body_constructors": 0, "neural_constructors": 0,
    "resident_bytes": 3221225472, "output_bytes": 134217728,
    "minimum_free_bytes": 536870912}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       indent=2) + "\n").encode("utf-8")


def save_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def pinned_path(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E246 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or any(piece in (".", "..") for piece in name.parts):
        raise ValueError("E246 unsafe frozen path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("E246 frozen path missing/outside repository")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E246 freeze SHA changed")
    if sha(review_path) != review_sha:
        raise ValueError("E246 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("representations") != [
                "receptor_features", "injected_filter", "afferent_rates",
                "configured_internal_rates", "native_motor_rates"]
            or freeze.get("train_controls") != [1, 1200]
            or freeze.get("query_controls") != [1, 2000]
            or freeze.get("batch_controls") != 128
            or freeze.get("temporal_exclusion_controls") != 20
            or freeze.get("std_floor") != 1e-4
            or freeze.get("normalization_ddof") != 0
            or freeze.get("tie_policy") != "earliest_reference_control"
            or freeze.get("distance_kind") != "euclidean_not_squared"
            or freeze.get("global_rms_ratio_max") != .8
            or freeze.get("per_leg_rms_ratio_max") != 1.
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E246 scientific/resource freeze changed")
    own = {name: sha(pinned_path(name)) for name in OWNED}
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E246 parent source review invalid")
    if freeze.get("packages") != {name: version(name) for name in freeze["packages"]}:
        raise ValueError("E246 frozen packages changed")
    if sha(pinned_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E246 design changed")
    for hashes in (freeze["input_sha256"], freeze["reused_source_sha256"],
                   freeze["preservation_sha256"],
                   review.get("additional_reused_source_sha256"),
                   review.get("additional_input_sha256")):
        if not isinstance(hashes, dict):
            raise TypeError("E246 freeze/review hash map missing")
        for name, expected in hashes.items():
            if sha(pinned_path(name)) != expected:
                raise ValueError("E246 pinned byte changed: " + name)
    paths = {role: pinned_path(name) for role, name in freeze["paths"].items()}
    if set(paths) != {"signal", "signal_metadata", "control", "roster",
                      "parameters", "targets", "groups", "baseline_raw",
                      "baseline_result", "signal_result", "signal_ledger",
                      "signal_manifest", "signal_proof", "signal_final_review",
                      "previous_supervisor", "baseline_proof"}:
        raise ValueError("E246 frozen input role set changed")
    return freeze, review, paths


def load(paths, *, progress=None):
    arrays = {}
    keys = {
        "signal": ("last_features", "old_filtered", "sensory_drive",
                   "selected_rate", "control_tick", "network_tick"),
        "control": ("control_tick", "network_tick", "motor_rate32",
                    "applied_command", "applied_tendon"),
        "parameters": ("free_columns", "fixed_columns", "membership",
                       "pool", "moment", "cap", "ltm"),
        "targets": ("target_command", "control_tick", "network_tick",
                    "motor_rate32", "applied_command", "applied_tendon"),
        "baseline_raw": ("free_columns", "sensitivity_weight",
                         "nearest_train_control", "nearest_train_distance",
                         "neighbor_teacher_q_difference"),
    }
    for role, selected in keys.items():
        with np.load(paths[role], allow_pickle=False) as saved:
            arrays[role] = {key: saved[key].copy() for key in selected}
        if progress:
            progress(role)
    for role in ("signal_metadata", "roster", "groups", "baseline_result",
                 "signal_result", "signal_ledger", "signal_manifest",
                 "signal_proof", "signal_final_review", "previous_supervisor",
                 "baseline_proof"):
        value = json.loads(paths[role].read_text(encoding="utf-8"))
        arrays[role] = value["groups"] if role == "groups" else value
        if progress:
            progress(role)
    return arrays


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def stats(values):
    return {"q_rms_rad": rms(values),
            "q_max_abs_rad": float(np.max(np.abs(values)))}


def distance_stats(values):
    values = np.asarray(values, np.float64)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def normalize(x, weight):
    """Independent per-column TRAIN mean/std; no tail or target enters."""
    x = np.asarray(x, np.float64)
    weight = np.asarray(weight, np.float64)
    if (x.ndim != 2 or x.shape[0] < 1201 or x.shape[1] != len(weight)
            or not np.isfinite(x).all() or not np.isfinite(weight).all()
            or np.any(weight <= 0) or not np.isclose(weight.sum(), 1,
                atol=2e-12, rtol=0)):
        raise ValueError("E246 finite positive balanced column weights required")
    mean = np.mean(x[1:1201], axis=0)
    std = np.std(x[1:1201], axis=0, ddof=0)
    floor = std < 1e-4
    scale = np.maximum(std, 1e-4)
    feature = (x - mean) / scale * np.sqrt(weight)
    return feature, mean, scale, floor


def geometry(inputs, *, strict_original=True):
    """Actual roster identities and per-port/per-leg physical columns."""
    signal, control, roster = (inputs[key] for key in ("signal", "control", "roster"))
    parameters = inputs["parameters"]
    target = np.asarray(inputs["targets"]["target_command"], np.float64)
    previous = inputs["targets"]
    free = np.asarray(parameters["free_columns"], np.int64)
    selected_rows = np.asarray(roster["selected_node_rows"], np.int64)
    motor_rows = np.asarray(roster["motor_node_rows"], np.int64)
    sensory_rows = np.asarray(roster["sensory_node_rows"], np.int64)
    sensory_columns = np.asarray(roster["sensory_selected_columns"], np.int64)
    motor_columns = np.asarray(roster["motor_selected_columns"], np.int64)
    core_rows = np.asarray(roster["core_adaptation_node_rows"], np.int64)
    ports = roster["ports"]
    n = target.shape[0]
    if (target.shape != (n, 42)
            or (strict_original and n != 2001)
            or (strict_original and (
                signal["last_features"].shape != (2001, 241)
                or signal["old_filtered"].shape != (2001, 241)
                or signal["sensory_drive"].shape != (2001, 3604)
                or signal["selected_rate"].shape != (2001, 4046)
                or control["motor_rate32"].shape != (2001, 391)
                or sensory_rows.shape != (3604,)
                or selected_rows.shape != (4046,)
                or motor_rows.shape != (391,)
                or core_rows.shape != (442,)
                or len(ports) != 241 or free.shape != (343,)
                or parameters["fixed_columns"].shape != (48,)))
            or signal["last_features"].shape != (n, len(ports))
            or signal["old_filtered"].shape != (n, len(ports))
            or signal["sensory_drive"].shape != (n, len(sensory_rows))
            or signal["selected_rate"].shape != (n, len(selected_rows))
            or control["motor_rate32"].shape != (n, len(motor_rows))
            or control["applied_command"].shape != (n, 42)
            or control["applied_tendon"].shape != (n, 8)
            or control["motor_rate32"].dtype != np.float32
            or signal["selected_rate"].dtype != np.float32
            or parameters["fixed_columns"].ndim != 1
            or len(free) + len(parameters["fixed_columns"]) != len(motor_rows)
            or not np.array_equal(np.sort(np.r_[free, parameters["fixed_columns"]]),
                                  np.arange(len(motor_rows)))
            or not np.array_equal(signal["control_tick"], np.arange(n))
            or not np.array_equal(control["control_tick"], np.arange(n))
            or not np.array_equal(signal["network_tick"], 20*np.arange(n))
            or not np.array_equal(control["network_tick"], 20*np.arange(n))
            or not np.array_equal(target, control["applied_command"])
            or not all(np.array_equal(previous[key], control[key]) for key in (
                "control_tick", "network_tick", "motor_rate32",
                "applied_command", "applied_tendon"))
            or not np.array_equal(selected_rows[sensory_columns], sensory_rows)
            or not np.array_equal(selected_rows[motor_columns], motor_rows)
            or not np.array_equal(signal["selected_rate"][:, motor_columns],
                                  control["motor_rate32"])):
        raise ValueError("E246 exact E245/E241 control, motor, or roster alignment changed")
    if not isinstance(inputs["signal_metadata"].get("roster_sha256"), str):
        raise TypeError("E246 signal metadata roster identity missing")
    for key in ("selected_node_rows", "sensory_node_rows", "motor_node_rows",
                "sensory_selected_columns", "motor_selected_columns"):
        if not np.array_equal(inputs["signal_metadata"][key], roster[key]):
            raise ValueError("E246 signal metadata/roster mapping changed")
    if not np.isfinite(target).all() or not all(np.isfinite(signal[key]).all()
            for key in ("last_features", "old_filtered", "sensory_drive", "selected_rate")):
        raise ValueError("E246 nonfinite saved signal/target")
    sensory_ports = np.asarray(roster["sensory_ports"], np.int64)
    if (sensory_ports.shape != (len(sensory_rows),)
            or np.any((sensory_ports < 0) | (sensory_ports >= len(ports)))
            or not np.array_equal(signal["sensory_drive"][1:],
                signal["old_filtered"][1:, sensory_ports].astype(np.float32))):
        raise ValueError("E246 same-control old-filter sensory drive changed")
    if not np.array_equal(inputs["signal_metadata"]["sensory_ports"], sensory_ports):
        raise ValueError("E246 metadata sensory port map changed")
    weights = anatomy_weights(parameters, inputs["groups"])
    baseline = inputs["baseline_raw"]
    old_index = np.asarray(baseline["nearest_train_control"], np.int64)
    old_difference = np.asarray(baseline["neighbor_teacher_q_difference"], np.float64)
    if (not np.array_equal(baseline["free_columns"], free)
            or not np.allclose(baseline["sensitivity_weight"], weights,
                               atol=2e-10, rtol=2e-12)
            or old_index.shape != (2001, 6)
            or baseline["nearest_train_distance"].shape != (2001, 6)
            or old_difference.shape != (2001, 42)
            or not np.isfinite(baseline["nearest_train_distance"]).all()
            or np.any(baseline["nearest_train_distance"] < 0)
            or not np.array_equal(old_index[0], np.full(6, -1))
            or np.any(old_index[1:] < 1) or np.any(old_index[1:] > 1200)
            or np.any(np.abs(old_index[1:]-np.arange(1, 2001)[:, None]) <= 20)):
        raise ValueError("E246 original E243 anatomy/neighbor identity changed")
    baseline_difference = np.zeros((2001, 42), np.float64)
    for leg_index in range(6):
        sl = slice(7*leg_index, 7*(leg_index+1))
        original = target[old_index[1:, leg_index], sl] - target[1:, sl]
        if not np.array_equal(old_difference[1:, sl], original):
            raise ValueError("E246 E243 teacher q provenance changed")
        baseline_difference[1:, sl] = -old_difference[1:, sl]
    row_to_selected = {int(row): position for position, row in enumerate(selected_rows)}
    if len(row_to_selected) != len(selected_rows):
        raise ValueError("E246 selected node rows are not unique")
    X = {}
    ids = {}
    column_weights = {}
    for leg_index, leg in enumerate(LEGS):
        port_ids = np.asarray([int(port["port"]) for port in ports
                               if port["leg_index"] == leg_index], np.int64)
        if not len(port_ids) or len(np.unique(port_ids)) != len(port_ids):
            raise ValueError("E246 empty/duplicate leg port roster: " + leg)
        port_weight = np.full(len(port_ids), 1/len(port_ids))
        for rep, field in (("R0", "last_features"), ("R1", "old_filtered")):
            X[rep, leg] = signal[field][:, port_ids]
            ids[rep, leg] = port_ids
            column_weights[rep, leg] = port_weight
        node_ids = [int(row) for port in ports if port["leg_index"] == leg_index
                    for row in port["node_rows"]]
        afferent_columns = np.asarray([row_to_selected[row] for row in node_ids], np.int64)
        afferent_weights = np.concatenate([np.full(len(port["node_rows"]),
            1/(len(port_ids)*len(port["node_rows"]))) for port in ports
            if port["leg_index"] == leg_index])
        X["R2", leg] = signal["selected_rate"][:, afferent_columns]
        ids["R2", leg] = np.asarray(node_ids, np.int64)
        column_weights["R2", leg] = afferent_weights
        selected_free = np.flatnonzero(weights[leg_index] > 0)
        motors = free[selected_free]
        X["R4", leg] = control["motor_rate32"][:, motors]
        ids["R4", leg] = motor_rows[motors]
        column_weights["R4", leg] = weights[leg_index, selected_free]
    motor_set = set(motor_rows)
    nonmotor = np.asarray([int(row) for row in core_rows if row not in motor_set],
                          np.int64)
    if (strict_original and nonmotor.shape != (51,)) or len(np.unique(nonmotor)) != len(nonmotor):
        raise ValueError("E246 original 51 configured nonmotor cells changed")
    internal_columns = np.asarray([row_to_selected[row] for row in nonmotor], np.int64)
    X["R3", "global"] = signal["selected_rate"][:, internal_columns]
    ids["R3", "global"] = nonmotor
    column_weights["R3", "global"] = np.full(len(nonmotor), 1/len(nonmotor))
    return target, baseline_difference, old_index, X, ids, column_weights


def nearest(feature, *, progress=None):
    """One weighted Euclidean NN search with 20-control exclusion."""
    reference = np.ascontiguousarray(feature[1:1201])
    ref_norm = np.einsum("ij,ij->i", reference, reference)
    index = np.full(2001, -1, np.int64)
    distance = np.full(2001, np.nan, np.float64)
    distance[0] = 0
    rows = 0
    for batches, first in enumerate(range(1, 2001, 128), start=1):
        query = np.arange(first, min(first+128, 2001), dtype=np.int64)
        q = np.ascontiguousarray(feature[query])
        squared = (np.einsum("ij,ij->i", q, q)[:, None]
                   + ref_norm[None] - 2*(q @ reference.T))
        np.maximum(squared, 0, out=squared)
        squared[np.abs(query[:, None] - np.arange(1, 1201)[None]) <= 20] = np.inf
        nearest_column = np.argmin(squared, axis=1)
        selected = squared[np.arange(len(query)), nearest_column]
        if not np.isfinite(selected).all():
            raise ValueError("E246 no admissible reference neighbor")
        index[query] = nearest_column + 1
        distance[query] = np.sqrt(selected)
        rows += len(query)
        if progress:
            progress(batches, rows, index, distance)
    return index, distance


def analyze_arrays(inputs, *, raw=None, normalization=None, progress=None,
                   strict_original=True):
    """Pure generated/saved arrays; caller-owned raw keeps actual completed rows."""
    raw = {} if raw is None else raw
    normalization = {} if normalization is None else normalization
    target, baseline, old_index, X, ids, weights = geometry(
        inputs, strict_original=strict_original)
    raw.update(representation_names=np.asarray(REPS),
        nearest_train_control=np.full((5, 2001, 6), -1, np.int64),
        nearest_train_distance=np.full((5, 2001, 6), np.nan, np.float64),
        teacher_q_difference=np.full((5, 2001, 42), np.nan, np.float64),
        baseline_teacher_q_difference=baseline,
        baseline_original_teacher_q_difference=np.asarray(
            inputs["baseline_raw"]["neighbor_teacher_q_difference"],
            np.float64).copy(),
        baseline_nearest_train_control=old_index,
        baseline_nearest_train_distance=np.asarray(
            inputs["baseline_raw"]["nearest_train_distance"], np.float64).copy(),
        completed_batches=np.zeros((5, 6), np.int64),
        completed_query_rows=np.zeros((5, 6), np.int64),
        completed_searches=np.asarray(0, np.int64))
    raw["nearest_train_distance"][:, 0] = 0
    raw["teacher_q_difference"][:, 0] = 0
    for rep_index, rep in enumerate(REPS):
        scopes = ("global",) if rep == "R3" else LEGS
        for scope in scopes:
            X_part = X[rep, scope]
            feature, mean, scale, floor = normalize(X_part, weights[rep, scope])
            for suffix, value in (("mean", mean), ("scale", scale),
                                  ("floor_mask", floor), ("weight", weights[rep, scope]),
                                  ("column_ids", ids[rep, scope])):
                normalization[rep + "_" + scope
                              + "_" + suffix] = np.asarray(value).copy()
            leg_indices = range(6) if rep == "R3" else (LEGS.index(scope),)
            primary_leg = next(iter(leg_indices))

            def batch_done(batches, rows, current_index, current_distance,
                           *, leg_indices=leg_indices, rep_index=rep_index,
                           rep=rep, primary_leg=primary_leg):
                for leg_index in leg_indices:
                    raw["nearest_train_control"][rep_index, :rows+1, leg_index] = (
                        current_index[:rows+1])
                    raw["nearest_train_distance"][rep_index, :rows+1, leg_index] = (
                        current_distance[:rows+1])
                    raw["completed_batches"][rep_index, leg_index] = batches
                    raw["completed_query_rows"][rep_index, leg_index] = rows
                    sl = slice(7*leg_index, 7*(leg_index+1))
                    raw["teacher_q_difference"][rep_index, 1:rows+1, sl] = (
                        target[1:rows+1, sl] - target[current_index[1:rows+1], sl])
                if progress:
                    progress(rep, primary_leg, batches, rows)

            nearest(feature, progress=batch_done)
            raw["completed_searches"] = np.asarray(
                int(raw["completed_searches"]) + 1, np.int64)
    if (int(raw["completed_searches"]) != 25
            or not np.array_equal(raw["completed_query_rows"],
                                  np.full((5, 6), 2000))
            or not all(np.isfinite(value).all() for value in raw.values()
                       if isinstance(value, np.ndarray) and value.dtype.kind == "f")):
        raise ValueError("E246 incomplete saved NN result")
    if (not np.all(raw["nearest_train_control"][3] ==
                   raw["nearest_train_control"][3][:, [0]])
            or not np.all(raw["nearest_train_distance"][3] ==
                          raw["nearest_train_distance"][3][:, [0]])):
        raise ValueError("E246 global internal NN was not reused across legs")
    differences = {rep: raw["teacher_q_difference"][i] for i, rep in enumerate(REPS)}
    differences["R5"] = baseline
    ranges = {"train": (1, 1201), "startup": (1, 100),
              "steady": (100, 1201), "observed_tail": (1201, 2001)}
    report = {"schema": "flylab.m3.e246.signal-discrimination.analysis.v1",
        "descriptive_only": True, "tail_independent_validation": False,
        "distance_scale_comparable_across_representations": False,
        "segments": {}, "bins": [], "predicates": {}, "distance_distributions": {}}
    for name, (lo, hi) in ranges.items():
        report["segments"][name] = {}
        for rep, difference in differences.items():
            section = {"global": stats(difference[lo:hi]), "legs": {}}
            for i, leg in enumerate(LEGS):
                section["legs"][leg] = stats(difference[lo:hi, 7*i:7*(i+1)])
            report["segments"][name][rep] = section
        report["distance_distributions"][name] = {}
        for rep_index, rep in enumerate((*REPS, "R5")):
            distance = (raw["nearest_train_distance"][rep_index]
                        if rep != "R5" else raw["baseline_nearest_train_distance"])
            report["distance_distributions"][name][rep] = {
                leg: distance_stats(distance[lo:hi, leg_index])
                for leg_index, leg in enumerate(LEGS)}
    for bin_index, lo in enumerate(range(1, 2001, 200)):
        hi = lo + 200
        section = {"bin": bin_index+1, "first_control": lo,
                   "last_control": hi-1}
        for rep, difference in differences.items():
            section[rep] = {"global": stats(difference[lo:hi]), "legs": {}}
            for i, leg in enumerate(LEGS):
                section[rep]["legs"][leg] = stats(difference[lo:hi, 7*i:7*(i+1)])
        report["bins"].append(section)
    for rep in REPS:
        predicates = {}
        for segment in ("steady", "observed_tail"):
            candidate = report["segments"][segment][rep]
            reference = report["segments"][segment]["R5"]
            predicates[segment + "/global"] = (candidate["global"]["q_rms_rad"]
                <= .8*reference["global"]["q_rms_rad"])
            for leg in LEGS:
                predicates[segment + "/" + leg] = (candidate["legs"][leg]["q_rms_rad"]
                    <= reference["legs"][leg]["q_rms_rad"])
        report["predicates"][rep] = {"separates_better_under_fixed_metric": bool(
            all(predicates.values())), "checks": predicates,
            "failed": [name for name, passed in predicates.items() if not passed]}
    return report, raw, normalization


def record_error(error, stage, *, attempt=None):
    receipt = {"stage": stage, "type": type(error).__name__,
               "message": str(error), "traceback": traceback.format_exc()}
    if attempt is not None:
        receipt["attempt"] = attempt
    return receipt


def invalidate_support(result):
    """Revoke every scientific support verdict after a technical failure."""
    result["separates_better_under_fixed_metric"] = None
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        for predicate in analysis.get("predicates", {}).values():
            predicate["separates_better_under_fixed_metric"] = None


def write_npz_once(out, filename, arrays):
    if not arrays:
        return
    destination = out / filename
    if destination.exists():
        return
    temporary = out / (filename + ".part")
    primary = None
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    except BaseException as error:  # noqa: BLE001 - preserve original write failure
        primary = error
    try:
        temporary.unlink(missing_ok=True)
    except BaseException as error:  # noqa: BLE001 - preserve cleanup failure too
        if primary is None:
            primary = error
        else:
            primary.add_note("E246 raw cleanup also failed: " + repr(error))
    if primary is not None:
        raise primary


def publish(out, result, raw, normalization, ledger, *, elapsed, deadline):
    """Write raw independently, then exact-byte terminal receipts."""
    secondary = []
    primary = None
    for filename, data in (("diagnostic-raw.npz", raw),
                           ("normalization.npz", normalization)):
        try:
            write_npz_once(out, filename, data)
        except BaseException as error:  # noqa: BLE001 - save the other raw artifact
            if primary is None:
                primary = error
            else:
                secondary.append(record_error(error, "raw_storage_secondary"))
    if secondary:
        ledger.setdefault("secondary_errors", []).extend(secondary)
    if primary is not None:
        raise primary
    ledger["process_wall_s"] = elapsed()
    save_json(out / "errors.json", ledger["errors"])
    save_json(out / "ledger.json", ledger)
    files = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file()
             and path.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": "flylab.m3.e246.signal-discrimination.manifest.v1",
                "files": files, "total_bytes": sum(entry["bytes"] for entry in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_payload = json_bytes(manifest)
    result_payload = json_bytes(result)
    if manifest["total_bytes"] + len(manifest_payload) + len(result_payload) > RESOURCE["output_bytes"]:
        raise OSError("E246 output exceeds 128 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E246 internal deadline before manifest publication")
    (out / "output-manifest.json").write_bytes(manifest_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E246 internal deadline before result publication")
    (out / "result.json").write_bytes(result_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E246 internal deadline after result publication")


def settle_publication(out, result, raw, normalization, ledger, *, elapsed, deadline):
    """Bounded refresh; publication faults invalidate the root verdict."""
    first = None
    start = time.perf_counter()
    for attempt in range(1, 4):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, normalization, ledger,
                    elapsed=elapsed, deadline=deadline)
            result["publication_wall_s"] = time.perf_counter() - start
            result["process_wall_s"] = elapsed()
            save_json(out / "result.json", result)
            if time.perf_counter() >= deadline:
                raise TimeoutError("E246 internal deadline after final time receipt")
            return first
        except BaseException as error:  # noqa: BLE001 - retain first and subsequent faults
            if first is None:
                first = error
            ledger["errors"].append(record_error(error, "publication", attempt=attempt))
            result["validity"] = "incomplete_or_invalid"
            invalidate_support(result)
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            if time.perf_counter() >= deadline:
                deadline = float("inf")  # settle invalid receipt after work timeout
    result["finalization_pending"] = False
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, raw, normalization, ledger,
                elapsed=elapsed, deadline=float("inf"))
        return first
    except BaseException as error:  # noqa: BLE001 - storage failure remains explicit
        ledger["errors"].append(record_error(error, "terminal_publication"))
        result["finalization_pending"] = True
        result["errors"] = ledger["errors"]
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
            save_json(out / "publication-failure.json", ledger["errors"])
            save_json(out / "result.json", result)
        except BaseException as secondary:  # noqa: BLE001 - retain both faults
            ledger["errors"].append(record_error(secondary, "terminal_receipt"))
    return first


def run(freeze_path, freeze_sha, review_path, review_sha, output,
        *, generated_test_output=False):
    began = time.perf_counter()
    out = Path(output).resolve()
    if not generated_test_output and out != OUT:
        raise ValueError("E246 original output must use reserved directory")
    if out.exists():
        raise FileExistsError("E246 one-shot output exists")
    out.mkdir(parents=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    process = psutil.Process()
    ledger = {"schema": "flylab.m3.e246.signal-discrimination.ledger.v1",
        "phase": "gate", "loaded_input_roles": [], "analysis_requested_count": 0,
        "analysis_entered_count": 0, "analysis_completed_count": 0,
        "original_calculation_count": 0,
        "completed_searches": 0, "completed_batches": [[0]*6 for _ in REPS],
        "completed_query_rows": [[0]*6 for _ in REPS],
        "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
        "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
        "neural_constructors": 0, "process_id": process.pid,
        "child_process_ids_start": [child.pid for child in process.children(recursive=True)],
        "retries": 0, "errors": []}
    result = {"schema": "flylab.m3.e246.signal-discrimination.result.v1",
        "validity": "incomplete_or_invalid",
        "neighbor_distance_kind": "euclidean_not_squared",
        "separates_better_under_fixed_metric": None,
        "descriptive_only": True, "observed_tail_independent_validation": False,
        "teacher_q_difference_sign": "query_minus_selected_reference",
        "baseline_original_difference_sign": "selected_reference_minus_query",
        "baseline_sign_conversion": "exact_negation_after_target_provenance_check",
        "timing": {"R0": "encode_c_raw", "R1": "encode_c_old_filter",
                   "R2": "post_20_neural_ticks_c", "R3": "post_20_neural_ticks_c",
                   "R4": "post_20_neural_ticks_c",
                   "teacher_target": "same_control_c_applied_command"},
        "candidate_adopted": False, "F2_complete": False, "F6_exposed": False,
        "whole_goal_complete": False, "active_UI_changed": False,
        "finalization_pending": True}
    raw = {}
    normalization = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter() - began

    def progress(rep, leg, batches, rows):
        index = REPS.index(rep)
        indices = range(6) if rep == "R3" else (leg,)
        for i in indices:
            ledger["completed_batches"][index][i] = batches
            ledger["completed_query_rows"][index][i] = rows
        if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
            raise TimeoutError("E246 analysis exhausted prepublication deadline")
        if process.memory_info().rss > RESOURCE["resident_bytes"]:
            raise MemoryError("E246 RSS exceeds 3 GiB")

    try:
        freeze, review, paths = gate(freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "additional_reused_source_sha256": review["additional_reused_source_sha256"],
            "additional_input_sha256": review["additional_input_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E246 free disk below 512 MiB")
        ledger["phase"] = "input"
        ledger["threadpool_before"] = threadpool_info()
        limiter = None
        work_error = None

        def loaded(role):
            ledger["loaded_input_roles"].append(role)
            if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
                raise TimeoutError("E246 input deadline")

        try:
            limiter = threadpool_limits(limits=1)
            observed = threadpool_info()
            ledger["threadpool_observed"] = observed
            if any(pool.get("num_threads") != 1 for pool in observed):
                raise RuntimeError("E246 imported threadpool exceeds one thread")
            inputs = load(paths, progress=loaded)
            if sha(paths["roster"]) != inputs["signal_metadata"]["roster_sha256"]:
                raise ValueError("E246 roster SHA differs from capture metadata")
            if (inputs["baseline_result"].get("validity") != "valid_saved_diagnostic"
                    or inputs["baseline_result"].get("neighbor_distance_kind") != "euclidean"
                    or inputs["baseline_proof"].get("validity") != "independently_valid"
                    or inputs["signal_result"].get("validity") != "valid_passive_signal_capture"):
                raise ValueError("E246 independently checked E243/E245 receipts required")
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            report, raw, normalization = analyze_arrays(inputs, raw=raw,
                normalization=normalization, progress=progress)
            ledger["analysis_completed_count"] = 1
            ledger["completed_searches"] = int(raw["completed_searches"])
        except BaseException as error:  # noqa: BLE001 - restore must not mask work failure
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:  # noqa: BLE001 - separately preserve restore
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = report
        result["separates_better_under_fixed_metric"] = {
            rep: report["predicates"][rep]["separates_better_under_fixed_metric"]
            for rep in REPS}
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:  # noqa: BLE001 - preserve original error and partial raw
        primary = error
        invalidate_support(result)
        ledger["errors"].append(record_error(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append({"stage": "threadpool_restore",
                "type": type(restore_error).__name__, "message": str(restore_error),
                "traceback": "".join(traceback.format_exception(
                    type(restore_error), restore_error, restore_error.__traceback__))})
    if raw:
        ledger["completed_searches"] = int(raw["completed_searches"])
        ledger["completed_batches"] = raw["completed_batches"].tolist()
        ledger["completed_query_rows"] = raw["completed_query_rows"].tolist()
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
        analysis_requested_count=ledger["analysis_requested_count"],
        analysis_entered_count=ledger["analysis_entered_count"],
        analysis_completed_count=ledger["analysis_completed_count"],
        original_calculation_count=ledger["original_calculation_count"],
        completed_searches=ledger["completed_searches"],
        completed_batches=ledger["completed_batches"],
        completed_query_rows=ledger["completed_query_rows"],
        fit_calls=0, solver_calls=0, cuda_calls=0, full_banc_calls=0,
        body_constructors=0, physics_steps=0, neural_constructors=0,
        errors=ledger["errors"], process_wall_s=elapsed())
    ledger["child_process_ids_end"] = [child.pid for child in process.children(recursive=True)]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle_publication(out, result, raw, normalization,
        ledger, elapsed=elapsed, deadline=began+RESOURCE["internal_s"])
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
