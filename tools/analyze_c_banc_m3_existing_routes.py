"""E247 saved CSR two-hop route audit; no neural or body advance."""

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
from tools import analyze_c_banc_m3_signal_discrimination as prior

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = (BASE / "e247-existing-routes").resolve()
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
FAMILIES = ("sensory_internal", "internal_motor", "sensory_motor")
OWNED = ("tools/analyze_c_banc_m3_existing_routes.py",
         "tests/test_c_banc_existing_routes.py")
FREEZE_SHA = "23e2b525a79e948059fb0f22b5818946d477ecc30aa66ef8a2a517441f6183f2"
REVIEW_SCHEMA = "flylab.m3.e247.existing-routes.review.v1"
RESOURCE = {"internal_s": 30, "finalization_reserve_s": 5,
    "external_s": 40, "supervisor_s": 45, "cpu_threads": 1, "gpu_count": 0,
    "original_attempts": 1, "retries": 0, "fit_calls": 0, "solver_calls": 0,
    "body_constructors": 0, "neural_constructors": 0,
    "resident_bytes": 2147483648, "output_bytes": 67108864,
    "minimum_free_bytes": 536870912}


def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def pinned_path(relative):
    return prior.pinned_path(relative)


def gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or prior.sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E247 freeze SHA changed")
    if prior.sha(review_path) != review_sha:
        raise ValueError("E247 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != "flylab.m3.e247.existing-routes.freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("network_identity") != "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
            or freeze.get("graph_hash") != "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or [freeze.get(key) for key in ("nodes", "declared_edges", "motor_count",
                    "free_motor_count", "fixed_motor_count", "internal_count",
                    "sensory_count", "port_count")] != [158706, 11584852, 391, 343,
                    48, 51, 3604, 241]
            or freeze.get("train_controls") != [1, 1200]
            or freeze.get("steady_controls") != [100, 1200]
            or freeze.get("dynamic_std_min") != 1e-4
            or freeze.get("normalization_ddof") != 0
            or freeze.get("matrix_orientation") != "W[post,pre]"
            or freeze.get("witness_tie_policy") !=
                "lowest_global_output_edge_then_lowest_global_input_edge"
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E247 frozen scientific/resource contract changed")
    own_hashes = {name: prior.sha(pinned_path(name)) for name in OWNED}
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own_hashes):
        raise ValueError("E247 parent source review invalid")
    if freeze.get("packages") != {key: version(key) for key in freeze["packages"]}:
        raise ValueError("E247 pinned package versions changed")
    if prior.sha(pinned_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E247 design bytes changed")
    for hashes in (freeze["input_sha256"], freeze["reused_source_sha256"],
                   freeze["preservation_sha256"],
                   review.get("additional_reused_source_sha256"),
                   review.get("additional_input_sha256")):
        if not isinstance(hashes, dict):
            raise TypeError("E247 required review/freeze hash map missing")
        for name, expected in hashes.items():
            if prior.sha(pinned_path(name)) != expected:
                raise ValueError("E247 pinned byte changed: " + name)
    paths = {role: pinned_path(name) for role, name in freeze["paths"].items()}
    if set(paths) != {"native_csr", "native_provenance", "native_identities",
            "graph_manifest", "graph_identity", "graph_indptr", "graph_indices",
            "graph_counts", "graph_weights", "roster", "signal", "signal_metadata",
            "start_state", "start_manifest", "start_mask", "final_state",
            "final_manifest", "final_mask", "parameters", "groups",
            "signal_proof", "signal_final_review", "discrimination_normalization",
            "discrimination_result", "discrimination_final_review",
            "previous_supervisor"}:
        raise ValueError("E247 input role set changed")
    return freeze, review, paths


def load(paths, *, progress=None):
    data = {}
    npz_keys = {
        "native_csr": ("indptr", "indices", "data"),
        "signal": ("old_filtered", "selected_rate", "control_tick", "network_tick"),
        "parameters": ("motor_ids", "free_columns", "fixed_columns", "membership", "pool",
                       "moment", "cap"),
        "discrimination_normalization": ("R3_global_column_ids",),
    }
    for role, keys in npz_keys.items():
        with np.load(paths[role], allow_pickle=False) as saved:
            data[role] = {key: saved[key].copy() for key in keys}
        if progress:
            progress(role)
    for role in ("graph_indptr", "graph_indices", "graph_counts", "graph_weights",
                 "start_mask", "final_mask"):
        data[role] = np.load(paths[role], allow_pickle=False)
        if progress:
            progress(role)
    for role in ("native_provenance", "native_identities", "graph_manifest",
                 "graph_identity", "roster", "signal_metadata", "start_state",
                 "start_manifest", "final_state", "final_manifest", "groups",
                 "signal_proof", "signal_final_review", "discrimination_result",
                 "discrimination_final_review", "previous_supervisor"):
        value = json.loads(paths[role].read_text(encoding="utf-8"))
        data[role] = value["groups"] if role == "groups" else value
        if progress:
            progress(role)
    return data


def validate(inputs, freeze, paths, *, strict_original=True):
    native = inputs["native_csr"]
    ptr, ix, data = (np.asarray(native[key]) for key in ("indptr", "indices", "data"))
    graph = {key: np.asarray(inputs["graph_" + key]) for key in
             ("indptr", "indices", "counts", "weights")}
    n = len(ptr) - 1
    edges = len(ix)
    roster = inputs["roster"]
    signal = inputs["signal"]
    params = inputs["parameters"]
    start = inputs["start_mask"]
    final = inputs["final_mask"]
    if (n < 1 or edges != len(data) or ptr.shape != (n+1,)
            or graph["indptr"].shape != ptr.shape
            or graph["indices"].shape != ix.shape
            or graph["counts"].shape != ix.shape
            or graph["weights"].shape != ix.shape
            or start.shape != (n,) or not np.array_equal(start, final)
            or not np.array_equal(ptr, graph["indptr"])
            or not np.array_equal(ix, graph["indices"])
            or ptr[0] != 0 or ptr[-1] != edges or np.any(np.diff(ptr) < 0)
            or np.any((ix < 0) | (ix >= n))
            or np.any(graph["counts"] < 1)
            or not np.isfinite(data).all() or not np.isfinite(graph["weights"]).all()
            or not np.isfinite(start).all() or not np.isin(start, [0, 1]).all()
            or np.any((data != 0) & (
                (graph["weights"] == 0) | (np.signbit(data) != np.signbit(graph["weights"]))))):
        raise ValueError("E247 native/graph CSR or source mask identity changed")
    if strict_original and (n != 158706 or edges != 11584852
            or len(roster["sensory_node_rows"]) != 3604
            or len(roster["motor_node_rows"]) != 391
            or len(roster["ports"]) != 241):
        raise ValueError("E247 frozen original graph dimensions changed")
    if strict_original:
        provenance = inputs["native_provenance"]
        if (provenance["rows"] != n or provenance["nnz"] != edges
                or provenance["indptr_sha256"] != array_sha(ptr)
                or provenance["indices_sha256"] != array_sha(ix)
                or provenance["data_sha256"] != array_sha(data)
                or inputs["native_identities"]["network_identity"] != freeze["network_identity"]
                or inputs["graph_manifest"]["graph_hash"] != freeze["graph_hash"]
                or inputs["graph_manifest"]["matrix_orientation"] != "W[post,pre]"
                or inputs["graph_manifest"]["pair_edge_count"] != edges
                or inputs["graph_manifest"]["anatomical_synapse_count"] !=
                    int(np.sum(graph["counts"], dtype=np.int64))
                or inputs["graph_identity"]["manifest_sha256"] !=
                    prior.sha(paths["graph_manifest"])
                or inputs["start_state"]["network"]["model_hash"] != freeze["network_identity"]
                or inputs["final_state"]["network"]["model_hash"] != freeze["network_identity"]
                or inputs["start_state"]["network"]["output_mask"] !=
                    {"__npy__": paths["start_mask"].name}
                or inputs["final_state"]["network"]["output_mask"] !=
                    {"__npy__": paths["final_mask"].name}
                or inputs["start_manifest"]["files"][paths["start_mask"].name] !=
                    prior.sha(paths["start_mask"])
                or inputs["final_manifest"]["files"][paths["final_mask"].name] !=
                    prior.sha(paths["final_mask"])):
            raise ValueError("E247 native provenance/network identity changed")
    if (signal["old_filtered"].shape != (2001, len(roster["ports"]))
            or signal["selected_rate"].shape != (2001, len(roster["selected_node_rows"]))
            or not np.array_equal(signal["control_tick"], np.arange(2001))
            or not np.array_equal(signal["network_tick"], 20*np.arange(2001))
            or not np.isfinite(signal["old_filtered"]).all()
            or not np.isfinite(signal["selected_rate"]).all()
            or not np.array_equal(inputs["signal_metadata"]["selected_node_rows"],
                                  roster["selected_node_rows"])
            or not np.array_equal(inputs["signal_metadata"]["sensory_ports"],
                                  roster["sensory_ports"])
            or inputs["signal_metadata"].get("roster_sha256") !=
                (prior.sha(paths["roster"]) if strict_original else
                 inputs["signal_metadata"].get("roster_sha256"))):
        raise ValueError("E247 E245 signal/roster alignment changed")
    selected = np.asarray(roster["selected_node_rows"], np.int64)
    motors = np.asarray(roster["motor_node_rows"], np.int64)
    sensory = np.asarray(roster["sensory_node_rows"], np.int64)
    core = np.asarray(roster["core_adaptation_node_rows"], np.int64)
    free = np.asarray(params["free_columns"], np.int64)
    fixed = np.asarray(params["fixed_columns"], np.int64)
    owner = np.asarray(params["membership"], np.int64)
    if (not np.array_equal(selected[np.asarray(roster["sensory_selected_columns"])], sensory)
            or not np.array_equal(selected[np.asarray(roster["motor_selected_columns"])], motors)
            or len(np.unique(selected)) != len(selected)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(len(motors)))
            or not np.array_equal(np.asarray(params["motor_ids"], np.int64), motors)
            or owner.shape != (len(motors),) or np.any((owner < 0) | (owner >= 96))
            or np.any((selected < 0) | (selected >= n))
            or len(inputs["groups"]) != 96):
        raise ValueError("E247 node/anatomy roster changed")
    motor_set = set(motors)
    internal = np.asarray([row for row in core if row not in motor_set], np.int64)
    if ((strict_original and len(internal) != 51) or not np.array_equal(
            inputs["discrimination_normalization"]["R3_global_column_ids"], internal)):
        raise ValueError("E247 R3 original internal51 identity changed")
    if strict_original and (inputs["discrimination_result"].get("validity") !=
            "valid_saved_diagnostic"
            or inputs["discrimination_final_review"].get("validity") !=
                "independently_valid"
            or inputs["signal_proof"].get("validity") != "independently_valid"):
        raise ValueError("E247 independently verified E245/E246 receipts required")
    if len(roster["nodes"]) != len(selected) or any(
            node["column"] != j or node["node_row"] != int(selected[j])
            for j, node in enumerate(roster["nodes"])):
        raise ValueError("E247 selected node annotations changed")
    return graph, internal


def column_stats(values, ids, prefix, target):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[0] != 2001 or values.shape[1] != len(ids):
        raise ValueError("E247 control/column statistics shape changed")
    for label, window in (("train", values[1:1201]),
                          ("steady", values[100:1201])):
        for name, result in (("mean", np.mean(window, axis=0)),
                             ("std", np.std(window, axis=0, ddof=0)),
                             ("min", np.min(window, axis=0)),
                             ("max", np.max(window, axis=0))):
            target[prefix + "_" + label + "_" + name] = np.asarray(result)
    target[prefix + "_ids"] = np.asarray(ids, np.int64)
    target[prefix + "_dynamic"] = target[prefix + "_steady_std"] >= 1e-4
    return target[prefix + "_dynamic"]


def extract_family(name, posts, source_mask, ptr, indices, graph, native,
                   sensory_lookup, internal_lookup, raw, *, progress=None):
    """Store every declared edge for one frozen endpoint family."""
    pre_lookup = internal_lookup if name == "internal_motor" else sensory_lookup
    selected = []
    for post in posts:
        first, last = int(ptr[post]), int(ptr[post+1])
        local = np.flatnonzero(pre_lookup[indices[first:last]])
        if len(local):
            selected.append(local.astype(np.int64) + first)
    edge_id = (np.concatenate(selected) if selected else np.empty(0, np.int64))
    post_rows = np.searchsorted(ptr, edge_id, side="right") - 1
    pre_rows = indices[edge_id].astype(np.int64, copy=False)
    graph_weight = graph["weights"][edge_id].astype(np.float32, copy=False)
    native_weight = native[edge_id].astype(np.float32, copy=False)
    mask = source_mask[pre_rows].astype(np.float32, copy=False)
    effective = (native_weight * mask).astype(np.float32)
    count = graph["counts"][edge_id].astype(np.int64, copy=False)
    for suffix, value in (("edge_id", edge_id), ("pre", pre_rows),
                          ("post", post_rows), ("count", count),
                          ("graph_weight", graph_weight),
                          ("native_weight", native_weight),
                          ("source_mask", mask),
                          ("effective_weight", effective)):
        raw[name + "_" + suffix] = np.asarray(value).copy()
    raw[name + "_eligible"] = np.zeros(len(edge_id), bool)
    if progress:
        progress("family", name, len(edge_id))


def annotations_for(roster, params, groups):
    sensory = np.asarray(roster["sensory_node_rows"], np.int64)
    motors = np.asarray(roster["motor_node_rows"], np.int64)
    sensory_ports = np.asarray(roster["sensory_ports"], np.int64)
    owner = np.asarray(params["membership"], np.int64)
    sensor_map = {int(row): int(sensory_ports[j]) for j, row in enumerate(sensory)}
    motor_map = {int(row): j for j, row in enumerate(motors)}
    records = []
    for node in roster["nodes"]:
        row = int(node["node_row"])
        record = dict(node)
        if row in sensor_map:
            port = sensor_map[row]
            record["sensory_port"] = port
            record["sensory_leg"] = roster["ports"][port]["leg"]
        if row in motor_map:
            column = motor_map[row]
            muscle = int(owner[column])
            record["motor_column"] = column
            record["muscle_index"] = muscle
            record["motor_leg"] = groups[muscle]["leg"]
            record["muscle_name"] = groups[muscle]["muscle"]
        records.append(record)
    return {"schema": "flylab.m3.e247.existing-routes.annotations.v1",
            "source": "E245 validated roster nodes/source_annotations",
            "nodes": records}


def analyze_arrays(native_csr, graph, signal, roster, parameters, groups,
                   output_mask, *, raw=None, stats_raw=None, progress=None,
                   strict_original=True):
    """Pure CSR/statistics join; raw dictionaries retain complete families on failure."""
    raw = {} if raw is None else raw
    stats_raw = {} if stats_raw is None else stats_raw
    ptr = np.asarray(native_csr["indptr"], np.int64)
    indices = np.asarray(native_csr["indices"], np.int64)
    native = np.asarray(native_csr["data"], np.float32)
    mask = np.asarray(output_mask, np.float32)
    sensory = np.asarray(roster["sensory_node_rows"], np.int64)
    motors = np.asarray(roster["motor_node_rows"], np.int64)
    core = np.asarray(roster["core_adaptation_node_rows"], np.int64)
    motor_set = set(motors)
    internal = np.asarray([row for row in core if row not in motor_set], np.int64)
    if (strict_original and (len(sensory) != 3604 or len(internal) != 51
            or len(motors) != 391)):
        raise ValueError("E247 original roster dimensions changed")
    selected_rows = np.asarray(roster["selected_node_rows"], np.int64)
    selected_column = {int(row): j for j, row in enumerate(selected_rows)}
    source = np.asarray(signal["selected_rate"], np.float64)
    port_ids = np.arange(len(roster["ports"]), dtype=np.int64)
    port_dynamic = column_stats(signal["old_filtered"], port_ids, "port", stats_raw)
    if progress:
        progress("statistics", "port", len(port_ids))
    sensory_dynamic = column_stats(source[:, [selected_column[int(row)]
                                          for row in sensory]], sensory, "sensory", stats_raw)
    stats_raw["sensory_port"] = np.asarray(roster["sensory_ports"], np.int64)
    if progress:
        progress("statistics", "sensory", len(sensory))
    internal_dynamic = column_stats(source[:, [selected_column[int(row)]
                                           for row in internal]], internal, "internal", stats_raw)
    if progress:
        progress("statistics", "internal", len(internal))
    n = len(ptr) - 1
    sensory_lookup = np.zeros(n, bool)
    internal_lookup = np.zeros(n, bool)
    sensory_lookup[sensory] = True
    internal_lookup[internal] = True
    for family, posts in (("sensory_internal", internal),
                          ("internal_motor", motors),
                          ("sensory_motor", motors)):
        extract_family(family, posts, mask, ptr, indices, graph, native,
                       sensory_lookup, internal_lookup, raw, progress=progress)
    port_of_row = np.full(n, -1, np.int64)
    port_of_row[sensory] = stats_raw["sensory_port"]
    sensory_dynamic_row = np.zeros(n, bool)
    sensory_dynamic_row[sensory] = sensory_dynamic
    internal_dynamic_row = np.zeros(n, bool)
    internal_dynamic_row[internal] = internal_dynamic
    free = np.asarray(parameters["free_columns"], np.int64)
    motor_free = np.zeros(len(motors), bool)
    motor_free[free] = True
    motor_column_of_row = np.full(n, -1, np.int64)
    motor_column_of_row[motors] = np.arange(len(motors))
    for family in FAMILIES:
        pre = raw[family + "_pre"]
        post = raw[family + "_post"]
        eligible = raw[family + "_effective_weight"] != 0
        if family == "internal_motor":
            eligible &= internal_dynamic_row[pre]
            eligible &= motor_free[motor_column_of_row[post]]
        else:
            eligible &= sensory_dynamic_row[pre]
            eligible &= port_dynamic[port_of_row[pre]]
            if family == "sensory_motor":
                eligible &= motor_free[motor_column_of_row[post]]
        raw[family + "_eligible"] = eligible
    raw["motor_column"] = np.arange(len(motors), dtype=np.int64)
    raw["motor_is_free"] = motor_free
    raw["motor_has_chain"] = np.zeros(len(motors), bool)
    raw["witness_motor_column"] = np.full(len(motors), -1, np.int64)
    raw["witness_input_edge"] = np.full(len(motors), -1, np.int64)
    raw["witness_output_edge"] = np.full(len(motors), -1, np.int64)
    sign_counts = np.zeros((2, 2), np.int64)
    leg_pairs = np.zeros((6, 6), np.int64)
    sensory_port = stats_raw["sensory_port"]
    sensory_index = np.full(n, -1, np.int64)
    sensory_index[sensory] = np.arange(len(sensory))
    port_leg = np.asarray([port["leg_index"] for port in roster["ports"]], np.int64)
    owner = np.asarray(parameters["membership"], np.int64)
    motor_leg = np.asarray([LEGS.index(groups[int(owner[j])]["leg"])
                            for j in range(len(motors))], np.int64)
    incoming = {}
    for j in np.flatnonzero(raw["sensory_internal_eligible"]):
        incoming.setdefault(int(raw["sensory_internal_post"][j]), []).append(j)
    output_eligible = np.flatnonzero(raw["internal_motor_eligible"])
    output_eligible = output_eligible[np.argsort(
        raw["internal_motor_edge_id"][output_eligible], kind="stable")]
    for j in output_eligible:
        internal_row = int(raw["internal_motor_pre"][j])
        inputs = incoming.get(internal_row)
        if not inputs:
            continue
        column = int(motor_column_of_row[raw["internal_motor_post"][j]])
        raw["motor_has_chain"][column] = True
        if raw["witness_output_edge"][column] == -1:
            raw["witness_motor_column"][column] = column
            raw["witness_output_edge"][column] = raw["internal_motor_edge_id"][j]
            raw["witness_input_edge"][column] = min(
                int(raw["sensory_internal_edge_id"][i]) for i in inputs)
        input_sign = raw["sensory_internal_effective_weight"][inputs] > 0
        output_sign = int(raw["internal_motor_effective_weight"][j] > 0)
        sign_counts[0, output_sign] += int(np.count_nonzero(~input_sign))
        sign_counts[1, output_sign] += int(np.count_nonzero(input_sign))
        sensory_rows = raw["sensory_internal_pre"][inputs]
        source_legs = port_leg[sensory_port[sensory_index[sensory_rows]]]
        leg_pairs[:, motor_leg[column]] += np.bincount(source_legs, minlength=6)
    raw["chain_sign_counts"] = sign_counts
    raw["sensory_leg_to_motor_leg_counts"] = leg_pairs
    direct = raw["sensory_motor_eligible"]
    direct_motors = np.zeros(len(motors), bool)
    direct_motors[motor_column_of_row[raw["sensory_motor_post"][direct]]] = True
    raw["motor_has_direct_sensory_edge"] = direct_motors
    direct_sign = np.asarray([
        np.count_nonzero(raw["sensory_motor_effective_weight"][direct] < 0),
        np.count_nonzero(raw["sensory_motor_effective_weight"][direct] > 0)], np.int64)
    direct_leg_pairs = np.zeros((6, 6), np.int64)
    direct_source_legs = port_leg[sensory_port[sensory_index[
        raw["sensory_motor_pre"][direct]]]]
    direct_dest_legs = motor_leg[motor_column_of_row[
        raw["sensory_motor_post"][direct]]]
    np.add.at(direct_leg_pairs, (direct_source_legs, direct_dest_legs), 1)
    raw["direct_sign_counts"] = direct_sign
    raw["direct_sensory_leg_to_motor_leg_counts"] = direct_leg_pairs
    per_leg = {}
    for i, leg in enumerate(LEGS):
        leg_free = np.flatnonzero((motor_leg == i) & motor_free)
        reached = leg_free[raw["motor_has_chain"][leg_free]]
        direct_reached = leg_free[direct_motors[leg_free]]
        muscles = {int(owner[j]) for j in leg_free}
        reached_muscles = {int(owner[j]) for j in reached}
        direct_muscles = {int(owner[j]) for j in direct_reached}
        missing = leg_free[~raw["motor_has_chain"][leg_free]]
        node_by_row = {node["node_row"]: node for node in roster["nodes"]}
        per_leg[leg] = {"free_mn_count": len(leg_free),
            "reachable_free_mn_count": len(reached),
            "existing_dynamic_two_hop_coverage": bool(len(reached)),
            "missing_free_mn_columns": missing.tolist(),
            "missing_free_mn_node_rows": motors[missing].tolist(),
            "missing_free_mn_ids": [node_by_row[int(motors[j])]["id"]
                                    for j in missing],
            "eligible_free_muscle_count": len(muscles),
            "reachable_muscle_count": len(reached_muscles),
            "missing_muscle_indices": sorted(muscles-reached_muscles),
            "missing_muscle_names": [groups[j]["muscle"]
                                     for j in sorted(muscles-reached_muscles)],
            "direct_sensory_motor": {"eligible_edge_count": int(np.count_nonzero(
                direct & (motor_leg[motor_column_of_row[
                    raw["sensory_motor_post"]]] == i))),
                "reachable_free_mn_count": len(direct_reached),
                "reachable_muscle_count": len(direct_muscles)}}
    supported = all(entry["reachable_free_mn_count"] > 0 for entry in per_leg.values())
    report = {"schema": "flylab.m3.e247.existing-routes.analysis.v1",
        "existing_dynamic_two_hop_coverage": supported,
        "per_leg": per_leg,
        "edge_family_counts": {family: {"declared": len(raw[family + "_edge_id"]),
            "eligible": int(np.count_nonzero(raw[family + "_eligible"]))}
            for family in FAMILIES},
        "eligible_distinct_edge_pair_count": int(np.sum(sign_counts, dtype=np.int64)),
        "chain_sign_counts": sign_counts.tolist(),
        "chain_sign_order": ["negative", "positive"],
        "sensory_leg_to_motor_leg_counts": leg_pairs.tolist(),
        "sensory_leg_to_motor_leg_order": list(LEGS),
        "direct_sign_counts": direct_sign.tolist(),
        "direct_sensory_leg_to_motor_leg_counts": direct_leg_pairs.tolist(),
        "same_leg_pair_count": int(np.trace(leg_pairs)),
        "cross_leg_pair_count": int(np.sum(leg_pairs) - np.trace(leg_pairs)),
        "eligibility": {
            "sensory_internal": "effective!=0 and sensory_dynamic and port_dynamic",
            "internal_motor": "effective!=0 and internal_dynamic and post_free",
            "sensory_motor": "effective!=0 and sensory_dynamic and port_dynamic and post_free"},
        "dynamic_std_threshold": 1e-4, "dynamic_window": [100, 1200],
        "train_statistics_window": [1, 1200],
        "narrow_existing_route_only": True,
        "observed_causal_contribution": False, "natural_gait_success": False}
    return report, raw, stats_raw, annotations_for(roster, parameters, groups)


def invalidate_support(result):
    result["existing_dynamic_two_hop_coverage"] = None
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        analysis["existing_dynamic_two_hop_coverage"] = None
        for entry in analysis.get("per_leg", {}).values():
            entry["existing_dynamic_two_hop_coverage"] = None


def publish(out, result, raw, statistics, annotations, ledger, *, elapsed, deadline):
    primary = None
    for filename, arrays in (("routes-raw.npz", raw),
                             ("dynamic-stats.npz", statistics)):
        try:
            prior.write_npz_once(out, filename, arrays)
        except BaseException as error:  # noqa: BLE001 - keep other raw artifact
            if primary is None:
                primary = error
            else:
                ledger.setdefault("secondary_errors", []).append(
                    prior.record_error(error, "raw_storage_secondary"))
    if annotations and not (out / "node-annotations.json").exists():
        try:
            prior.save_json(out / "node-annotations.json", annotations)
        except BaseException as error:  # noqa: BLE001 - keep raw first failure
            if primary is None:
                primary = error
            else:
                ledger.setdefault("secondary_errors", []).append(
                    prior.record_error(error, "annotation_storage_secondary"))
    if primary is not None:
        raise primary
    ledger["process_wall_s"] = elapsed()
    prior.save_json(out / "errors.json", ledger["errors"])
    prior.save_json(out / "ledger.json", ledger)
    files = {p.name: {"sha256": prior.sha(p), "bytes": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file()
             and p.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": "flylab.m3.e247.existing-routes.manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_bytes = prior.json_bytes(manifest)
    result_bytes = prior.json_bytes(result)
    if manifest["total_bytes"] + len(manifest_bytes) + len(result_bytes) > RESOURCE["output_bytes"]:
        raise OSError("E247 output exceeds 64 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E247 deadline before manifest publication")
    (out / "output-manifest.json").write_bytes(manifest_bytes)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E247 deadline before result publication")
    (out / "result.json").write_bytes(result_bytes)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E247 deadline after result publication")


def settle(out, result, raw, statistics, annotations, ledger, *, elapsed, deadline):
    first = None
    started = time.perf_counter()
    for attempt in range(1, 4):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, statistics, annotations, ledger,
                    elapsed=elapsed, deadline=deadline)
            result["publication_wall_s"] = time.perf_counter() - started
            result["process_wall_s"] = elapsed()
            prior.save_json(out / "result.json", result)
            if time.perf_counter() >= deadline:
                raise TimeoutError("E247 deadline after final time receipt")
            return first
        except BaseException as error:  # noqa: BLE001 - retain first and later failures
            if first is None:
                first = error
            ledger["errors"].append(prior.record_error(error, "publication",
                                                        attempt=attempt))
            result["validity"] = "incomplete_or_invalid"
            invalidate_support(result)
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            if time.perf_counter() >= deadline:
                deadline = float("inf")
    result["finalization_pending"] = False
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, raw, statistics, annotations, ledger,
                elapsed=elapsed, deadline=float("inf"))
        return first
    except BaseException as error:  # noqa: BLE001 - explicit pending terminal failure
        ledger["errors"].append(prior.record_error(error, "terminal_publication"))
        result["finalization_pending"] = True
        result["errors"] = ledger["errors"]
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
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
        raise ValueError("E247 original output must use reserved directory")
    if out.exists():
        raise FileExistsError("E247 one-shot output already exists")
    out.mkdir(parents=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    process = psutil.Process()
    ledger = {"schema": "flylab.m3.e247.existing-routes.ledger.v1",
        "phase": "gate", "loaded_input_roles": [], "analysis_requested_count": 0,
        "analysis_entered_count": 0, "analysis_completed_count": 0,
        "original_calculation_count": 0,
        "completed_statistics": [], "completed_edge_families": [],
        "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
        "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
        "neural_constructors": 0, "process_id": process.pid,
        "child_process_ids_start": [child.pid for child in process.children(recursive=True)],
        "retries": 0, "errors": []}
    result = {"schema": "flylab.m3.e247.existing-routes.result.v1",
        "validity": "incomplete_or_invalid", "existing_dynamic_two_hop_coverage": None,
        "candidate_adopted": False, "F2_complete": False, "F6_exposed": False,
        "whole_goal_complete": False, "active_UI_changed": False,
        "finalization_pending": True,
        "causal_contribution_established": False, "gait_success_established": False}
    raw = {}
    statistics = {}
    annotations = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter() - began

    def progress(kind, name, _count):
        if kind == "statistics":
            ledger["completed_statistics"].append(name)
        elif kind == "family":
            ledger["completed_edge_families"].append(name)
        if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
            raise TimeoutError("E247 analysis exhausted prepublication deadline")
        if process.memory_info().rss > RESOURCE["resident_bytes"]:
            raise MemoryError("E247 RSS exceeds 2 GiB")

    try:
        freeze, review, paths = gate(freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "additional_reused_source_sha256": review["additional_reused_source_sha256"],
            "additional_input_sha256": review["additional_input_sha256"],
            "packages": freeze["packages"], "graph_hash": freeze["graph_hash"],
            "network_identity": freeze["network_identity"]}
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E247 free disk below 512 MiB")
        ledger["phase"] = "input"
        ledger["threadpool_before"] = threadpool_info()
        limiter = None
        work_error = None

        def loaded(role):
            ledger["loaded_input_roles"].append(role)
            if elapsed() >= RESOURCE["internal_s"]-RESOURCE["finalization_reserve_s"]:
                raise TimeoutError("E247 input deadline")

        try:
            limiter = threadpool_limits(limits=1)
            observed = threadpool_info()
            ledger["threadpool_observed"] = observed
            if any(pool.get("num_threads") != 1 for pool in observed):
                raise RuntimeError("E247 imported threadpool exceeds one thread")
            inputs = load(paths, progress=loaded)
            graph, _ = validate(inputs, freeze, paths)
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            report, raw, statistics, annotations = analyze_arrays(
                inputs["native_csr"], graph, inputs["signal"], inputs["roster"],
                inputs["parameters"], inputs["groups"], inputs["start_mask"],
                raw=raw, stats_raw=statistics, progress=progress)
            ledger["analysis_completed_count"] = 1
        except BaseException as error:  # noqa: BLE001 - restore cannot mask work failure
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
        result["analysis"] = report
        result["existing_dynamic_two_hop_coverage"] = (
            report["existing_dynamic_two_hop_coverage"])
        result["validity"] = "valid_saved_route_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:  # noqa: BLE001 - preserve original and partial raw
        primary = error
        invalidate_support(result)
        ledger["errors"].append(prior.record_error(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append({"stage": "threadpool_restore",
                "type": type(restore_error).__name__, "message": str(restore_error),
                "traceback": "".join(traceback.format_exception(
                    type(restore_error), restore_error, restore_error.__traceback__))})
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
        analysis_requested_count=ledger["analysis_requested_count"],
        analysis_entered_count=ledger["analysis_entered_count"],
        analysis_completed_count=ledger["analysis_completed_count"],
        original_calculation_count=ledger["original_calculation_count"],
        completed_statistics=ledger["completed_statistics"],
        completed_edge_families=ledger["completed_edge_families"],
        fit_calls=0, solver_calls=0, cuda_calls=0, full_banc_calls=0,
        body_constructors=0, physics_steps=0, neural_constructors=0,
        errors=ledger["errors"], process_wall_s=elapsed())
    ledger["child_process_ids_end"] = [child.pid for child in process.children(recursive=True)]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle(out, result, raw, statistics, annotations, ledger,
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
