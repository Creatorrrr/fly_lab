"""Generated-only E247 CSR direction and route receipts."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_existing_routes as e247


def fixture():
    n = 40
    sensory = np.arange(6)
    internal = np.arange(6, 12)
    motors = np.arange(20, 27)
    # post, pre, graph weight, native weight; CSR order fixes global edge IDs.
    entries = [(6, 0, 1, 1), (6, 1, -2, -2), (7, 0, 1, 1),
               (7, 2, 1, 1), (8, 5, 1, 1), (9, 3, 1, 0),
               (10, 4, 1, 1), (20, 0, 1, 1), (20, 6, 2, 2),
               (20, 7, -3, -3), (21, 7, 1, 1), (22, 8, 1, 1),
               (26, 6, 1, 1)]
    entries.sort()
    ptr = np.zeros(n+1, np.int64)
    for post, *_ in entries:
        ptr[post+1] += 1
    ptr = np.cumsum(ptr)
    indices = np.asarray([pre for _, pre, _, _ in entries], np.int64)
    graph_weight = np.asarray([w for _, _, w, _ in entries], np.float32)
    native_weight = np.asarray([w for _, _, _, w in entries], np.float32)
    native = {"indptr": ptr, "indices": indices, "data": native_weight}
    graph = {"indptr": ptr.copy(), "indices": indices.copy(),
             "counts": np.arange(1, len(entries)+1), "weights": graph_weight}
    selected_rows = np.r_[sensory, internal, motors]
    control = np.arange(2001)
    old = np.zeros((2001, 6))
    rate = np.zeros((2001, len(selected_rows)), np.float32)
    dynamic = (control[100:1201] - 100) / 1000
    for p in (0, 1, 2, 3, 4, 5):
        old[100:1201, p] = dynamic
    for row in (0, 1, 3, 4, 5, 6, 7):
        rate[100:1201, np.flatnonzero(selected_rows == row)[0]] = dynamic
    # Sensory2 stays static even though its port changes; internal8 varies only
    # before steady and in the future tail, so neither can pass steady selection.
    rate[1:100, np.flatnonzero(selected_rows == 8)[0]] = 2
    rate[1201:, np.flatnonzero(selected_rows == 8)[0]] = 3
    nodes = [{"column": j, "node_row": int(row),
              "source_annotations": {"root_888": str(row)},
              "root_id": str(100000000000000000 + int(row)),
              "id": f"generated:{row}"} for j, row in enumerate(selected_rows)]
    roster = {"sensory_node_rows": sensory.tolist(),
        "sensory_ports": list(range(6)), "motor_node_rows": motors.tolist(),
        "core_adaptation_node_rows": np.r_[motors, internal].tolist(),
        "selected_node_rows": selected_rows.tolist(),
        "sensory_selected_columns": list(range(6)),
        "motor_selected_columns": list(range(12, 19)),
        "ports": [{"port": i, "leg_index": i, "leg": e247.LEGS[i],
                   "node_rows": [i]} for i in range(6)], "nodes": nodes}
    parameters = {"free_columns": np.arange(6),
        "fixed_columns": np.array([6]), "motor_ids": motors.copy(),
        "membership": np.arange(7),
        "pool": np.zeros((96, 7)), "moment": np.zeros((42, 96)),
        "cap": np.ones(7)}
    groups = [{"leg": e247.LEGS[min(i, 5)], "muscle": f"generated_{i}"}
              for i in range(96)]
    mask = np.ones(n, np.float32)
    mask[4] = 0
    signal = {"old_filtered": old, "selected_rate": rate,
              "control_tick": control, "network_tick": 20*control}
    return native, graph, signal, roster, parameters, groups, mask


def edge_index(raw, family, post, pre):
    found = np.flatnonzero((raw[family + "_post"] == post) &
                          (raw[family + "_pre"] == pre))
    assert len(found) == 1
    return int(found[0])


def test_csr_global_ids_direction_witness_and_pair_counts():
    native, graph, signal, roster, parameters, groups, mask = fixture()
    report, raw, stats, notes = e247.analyze_arrays(
        native, graph, signal, roster, parameters, groups, mask,
        strict_original=False)
    si = edge_index(raw, "sensory_internal", 6, 0)
    assert raw["sensory_internal_edge_id"][si] == native["indptr"][6]
    assert raw["sensory_internal_count"][si] == graph["counts"][
        raw["sensory_internal_edge_id"][si]]
    assert raw["sensory_internal_eligible"][si]
    assert raw["motor_has_chain"][0]
    assert not raw["motor_has_chain"][6]  # fixed motor retains edges, not H247
    assert raw["witness_motor_column"].tolist() == [0, 1, -1, -1, -1, -1, -1]
    assert raw["witness_output_edge"][0] == raw["internal_motor_edge_id"][
        edge_index(raw, "internal_motor", 20, 6)]
    assert raw["witness_input_edge"][0] == min(raw["sensory_internal_edge_id"][
        edge_index(raw, "sensory_internal", 6, pre)] for pre in (0, 1))
    assert report["eligible_distinct_edge_pair_count"] == 4
    assert report["chain_sign_counts"] == [[0, 1], [1, 2]]
    assert report["sensory_leg_to_motor_leg_counts"][0][0] == 2
    assert report["sensory_leg_to_motor_leg_counts"][1][0] == 1
    assert notes["nodes"][0]["source_annotations"] == {"root_888": "0"}
    assert stats["port_train_std"].shape == (6,)


def test_zero_mask_dynamic_and_direct_declared_edges_preserved():
    native, graph, signal, roster, parameters, groups, mask = fixture()
    report, raw, stats, _ = e247.analyze_arrays(
        native, graph, signal, roster, parameters, groups, mask,
        strict_original=False)
    assert not stats["sensory_dynamic"][2]  # changing port alone is insufficient
    assert not stats["internal_dynamic"][2]  # startup/future-only rate excluded
    assert raw["sensory_internal_native_weight"][edge_index(
        raw, "sensory_internal", 9, 3)] == 0
    assert not raw["sensory_internal_eligible"][edge_index(
        raw, "sensory_internal", 9, 3)]
    masked = edge_index(raw, "sensory_internal", 10, 4)
    assert raw["sensory_internal_source_mask"][masked] == 0
    assert raw["sensory_internal_effective_weight"][masked] == 0
    assert not raw["sensory_internal_eligible"][masked]
    assert raw["sensory_internal_eligible"][edge_index(
        raw, "sensory_internal", 8, 5)]
    assert not raw["internal_motor_eligible"][edge_index(
        raw, "internal_motor", 22, 8)]
    assert not raw["internal_motor_eligible"][edge_index(
        raw, "internal_motor", 26, 6)]
    assert raw["sensory_motor_eligible"][edge_index(
        raw, "sensory_motor", 20, 0)]
    assert report["per_leg"]["lf"]["direct_sensory_motor"]["eligible_edge_count"] == 1


def test_completed_family_survives_generated_failure():
    native, graph, signal, roster, parameters, groups, mask = fixture()
    raw = {}
    stats = {}

    def fail(kind, name, _count):
        if kind == "family" and name == "sensory_internal":
            raise RuntimeError("generated family boundary")

    with pytest.raises(RuntimeError, match="family boundary"):
        e247.analyze_arrays(native, graph, signal, roster, parameters,
                            groups, mask, raw=raw, stats_raw=stats,
                            progress=fail, strict_original=False)
    assert len(raw["sensory_internal_edge_id"]) == 7
    assert "internal_motor_edge_id" not in raw
    assert "port_steady_std" in stats and "internal_steady_std" in stats


def test_generated_csr_sign_and_mask_validation(tmp_path):
    native, graph, signal, roster, parameters, groups, mask = fixture()
    inputs = {"native_csr": native, "graph_indptr": graph["indptr"],
        "graph_indices": graph["indices"], "graph_counts": graph["counts"],
        "graph_weights": graph["weights"], "roster": roster,
        "signal": signal, "signal_metadata": {"selected_node_rows":
            roster["selected_node_rows"], "sensory_ports": roster["sensory_ports"],
            "roster_sha256": "generated"},
        "parameters": parameters, "groups": groups,
        "start_mask": mask, "final_mask": mask.copy(),
        "discrimination_normalization": {"R3_global_column_ids": np.arange(6, 12)}}
    graph_out, internal = e247.validate(inputs, {}, {}, strict_original=False)
    assert len(internal) == 6 and graph_out["weights"].shape == (13,)
    assert inputs["parameters"]["motor_ids"][0] == 20
    assert roster["nodes"][12]["root_id"] == "100000000000000020"
    inputs["parameters"]["motor_ids"] = np.r_[np.array([21, 20]), np.arange(22, 27)]
    with pytest.raises(ValueError, match="roster"):
        e247.validate(inputs, {}, {}, strict_original=False)
    inputs["parameters"]["motor_ids"] = np.asarray([
        int(node["root_id"]) for node in roster["nodes"][12:19]], np.int64)
    with pytest.raises(ValueError, match="roster"):
        e247.validate(inputs, {}, {}, strict_original=False)
    inputs["parameters"]["motor_ids"] = np.arange(20, 27)
    inputs["native_csr"]["data"] = native["data"].copy()
    inputs["native_csr"]["data"][0] *= -1
    with pytest.raises(ValueError, match="CSR"):
        e247.validate(inputs, {}, {}, strict_original=False)
    inputs["native_csr"]["data"][0] *= -1
    inputs["final_mask"] = mask.copy()
    inputs["final_mask"][4] = 1
    with pytest.raises(ValueError, match="CSR"):
        e247.validate(inputs, {}, {}, strict_original=False)


def test_late_and_persistent_publication_revoke_nested_support(tmp_path, monkeypatch):
    result = {"validity": "valid_saved_route_diagnostic",
        "existing_dynamic_two_hop_coverage": True,
        "analysis": {"existing_dynamic_two_hop_coverage": True,
                     "per_leg": {"lf": {"existing_dynamic_two_hop_coverage": True,
                                        "reachable_free_mn_count": 1}}}}
    ledger = {"errors": []}
    real_publish = e247.publish
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("generated late write")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(e247, "publish", fail_once)
    error = e247.settle(tmp_path, result, {"edge": np.array([1])},
        {"std": np.array([.2])}, {"nodes": []}, ledger,
        elapsed=lambda: .1, deadline=e247.time.perf_counter()+10)
    assert isinstance(error, OSError)
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["existing_dynamic_two_hop_coverage"] is None
    assert saved["analysis"]["existing_dynamic_two_hop_coverage"] is None
    assert saved["analysis"]["per_leg"]["lf"]["existing_dynamic_two_hop_coverage"] is None
    assert saved["analysis"]["per_leg"]["lf"]["reachable_free_mn_count"] == 1
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name, receipt in manifest["files"].items():
        assert e247.prior.sha(tmp_path / name) == receipt["sha256"]

    def persistent(*_args, **_kwargs):
        raise OSError("persistent")

    stuck = tmp_path / "stuck"
    stuck.mkdir()
    monkeypatch.setattr(e247, "publish", persistent)
    second = {"validity": "valid_saved_route_diagnostic",
        "existing_dynamic_two_hop_coverage": True,
        "analysis": {"existing_dynamic_two_hop_coverage": True,
                     "per_leg": {"rh": {"existing_dynamic_two_hop_coverage": True}}}}
    e247.settle(stuck, second, {}, {}, {}, {"errors": []},
        elapsed=lambda: .1, deadline=e247.time.perf_counter()+10)
    receipt = json.loads((stuck / "result.json").read_bytes())
    assert receipt["finalization_pending"] is True
    assert receipt["analysis"]["per_leg"]["rh"]["existing_dynamic_two_hop_coverage"] is None
    assert not (stuck / "output-manifest.json").exists()


def test_hash_gate_rejects_before_original_npz_load(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("original arrays opened")

    monkeypatch.setattr(e247, "load", forbidden)
    with pytest.raises(ValueError, match="freeze SHA"):
        e247.run(tmp_path / "absent", "wrong", tmp_path / "review", "wrong",
                 tmp_path / "generated-out", generated_test_output=True)
    receipt = json.loads((tmp_path / "generated-out" / "result.json").read_bytes())
    assert receipt["original_calculation_count"] == 0
    assert receipt["loaded_input_roles"] == []
