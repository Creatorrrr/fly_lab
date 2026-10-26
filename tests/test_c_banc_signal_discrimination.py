"""Generated-only E246 signal discrimination checks."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_signal_discrimination as e246


def generated_inputs():
    n = 2001
    motor_rows = np.arange(100, 112)
    ports = []
    sensory_rows = []
    sensory_ports = []
    for leg in range(6):
        for local in range(2):
            rows = list(range(1000 + len(sensory_rows),
                              1000 + len(sensory_rows) + local + 1))
            port_id = 2 * leg + local
            ports.append({"port": port_id, "leg_index": leg,
                          "node_rows": rows})
            sensory_rows.extend(rows)
            sensory_ports.extend([port_id] * len(rows))
    sensory_rows = np.asarray(sensory_rows)
    selected_rows = np.r_[sensory_rows, motor_rows, [9999]]
    old = np.tile(np.arange(12, dtype=np.float64), (n, 1))
    raw = old + np.arange(n)[:, None] / 5000
    selected = np.zeros((n, len(selected_rows)), np.float32)
    selected[:, :len(sensory_rows)] = np.arange(len(sensory_rows))[None]
    selected[:, len(sensory_rows):len(sensory_rows)+12] = (
        np.arange(n)[:, None] / 100 + np.arange(12)[None]).astype(np.float32)
    target = (np.arange(n)[:, None] / 100 + np.arange(42)[None] / 10)
    old_index = np.full((n, 6), -1, np.int64)
    q = np.arange(1, n)
    old_index[1:] = np.where(q <= 1179, q+21, 1)[:, None]
    old_diff = np.zeros((n, 42))
    for leg in range(6):
        sl = slice(7*leg, 7*(leg+1))
        old_diff[1:, sl] = target[old_index[1:, leg], sl] - target[1:, sl]
    roster = {"ports": ports, "sensory_ports": sensory_ports,
        "selected_node_rows": selected_rows.tolist(),
        "sensory_node_rows": sensory_rows.tolist(),
        "motor_node_rows": motor_rows.tolist(),
        "core_adaptation_node_rows": np.r_[motor_rows, [9999]].tolist(),
        "sensory_selected_columns": list(range(len(sensory_rows))),
        "motor_selected_columns": list(range(len(sensory_rows),
                                               len(sensory_rows)+12))}
    pool = np.zeros((96, 12))
    moment = np.zeros((42, 96))
    for motor in range(12):
        pool[motor, motor] = 1
        moment[7*(motor//2), motor] = 1 if motor % 2 == 0 else 3
    parameters = {"free_columns": np.arange(12),
        "fixed_columns": np.asarray([], np.int64),
        "membership": np.arange(12), "pool": pool, "moment": moment,
        "cap": np.ones(12)}
    groups = [{"leg": e246.LEGS[min(j//2, 5)]} for j in range(96)]
    weights = e246.anatomy_weights(parameters, groups)
    control = {"control_tick": np.arange(n),
        "network_tick": 20*np.arange(n),
        "motor_rate32": selected[:, len(sensory_rows):len(sensory_rows)+12],
        "applied_command": target.copy(),
        "applied_tendon": np.zeros((n, 8))}
    return {"roster": roster,
        "signal_metadata": {**{key: roster[key] for key in (
            "selected_node_rows", "sensory_node_rows", "motor_node_rows",
            "sensory_selected_columns", "motor_selected_columns",
            "sensory_ports")}, "roster_sha256": "generated"},
        "signal": {"last_features": raw, "old_filtered": old,
            "sensory_drive": old[:, sensory_ports].astype(np.float32),
            "selected_rate": selected,
            "control_tick": np.arange(n), "network_tick": 20*np.arange(n)},
        "control": control,
        "targets": {**{key: value.copy() for key, value in control.items()},
                    "target_command": target},
        "parameters": parameters, "groups": groups,
        "baseline_raw": {"free_columns": np.arange(12),
            "sensitivity_weight": weights,
            "nearest_train_control": old_index,
            "nearest_train_distance": np.zeros((n, 6)),
            "neighbor_teacher_q_difference": old_diff}}


def test_port_balance_and_second_control_alignment():
    inputs = generated_inputs()
    target, baseline, _, x, ids, weight = e246.geometry(inputs, strict_original=False)
    assert np.array_equal(target, inputs["control"]["applied_command"])
    assert weight["R0", "lf"].tolist() == [.5, .5]
    assert weight["R2", "lf"].tolist() == [.5, .25, .25]
    assert np.allclose(weight["R4", "lf"], [.25, .75])
    assert ids["R3", "global"].tolist() == [9999]
    assert x["R2", "lf"][2, 0] == inputs["signal"]["selected_rate"][2, 0]
    assert baseline[2, 0] == target[2, 0] - target[23, 0]
    feature, mean, scale, _ = e246.normalize(x["R4", "lf"], weight["R4", "lf"])
    chosen, distances = e246.nearest(feature)
    query = 1234
    manual = np.sqrt(np.sum(weight["R4", "lf"] * (
        ((x["R4", "lf"][query]-mean)/scale)[None] -
        (x["R4", "lf"][1:1201]-mean)/scale)**2, axis=1))
    assert chosen[query] == int(np.argmin(manual)) + 1
    assert distances[query] == pytest.approx(float(np.min(manual)), abs=1e-7)
    inputs["control"]["network_tick"][2] += 1
    with pytest.raises(ValueError, match="alignment"):
        e246.geometry(inputs, strict_original=False)


def test_nearest_weighted_distance_exclusion_and_earliest_tie():
    x = np.full((2001, 2), 1000., np.float64)
    x[:, 0] += np.arange(2001)
    x[500] = [0, 0]
    x[479] = [1, 2]
    x[521] = [-1, -2]
    x[480] = [0, 0]  # exactly 20 away: excluded
    index, distance = e246.nearest(x)
    assert index[0] == -1 and distance[0] == 0
    assert index[500] == 479
    assert distance[500] == pytest.approx(np.sqrt(5))


def test_training_standardization_ignores_future_tail():
    x = np.tile(np.array([2., 5.]), (2001, 1))
    x[1:1201, 0] = np.arange(1200)/1200
    first = e246.normalize(x, np.array([.2, .8]))
    x[1201:, 0] += 9000
    x[1201:, 1] = -500
    second = e246.normalize(x, np.array([.2, .8]))
    for before, after in zip(first[1:], second[1:], strict=True):
        assert np.array_equal(before, after)
    assert first[3].tolist() == [False, True]
    assert np.array_equal(first[0][:1201], second[0][:1201])
    assert not np.array_equal(first[0][1201:], second[0][1201:])


def test_analysis_shared_internal_neighbor_and_partial_progress(monkeypatch):
    inputs = generated_inputs()
    seen = []

    def fake_nearest(feature, *, progress):
        index = np.full(2001, -1, np.int64)
        index[1:] = 1
        distance = np.zeros(2001)
        progress(1, 2000, index, distance)
        seen.append(feature.shape[1])
        return index, distance

    monkeypatch.setattr(e246, "nearest", fake_nearest)
    raw = {}
    report, _, norm = e246.analyze_arrays(inputs, raw=raw, strict_original=False)
    assert len(seen) == 25
    assert int(raw["completed_searches"]) == 25
    assert np.array_equal(raw["nearest_train_control"][3],
                          np.ones((2001, 6), np.int64) - np.r_[2, np.zeros(2000)][:, None])
    assert raw["teacher_q_difference"][0, 201, 0] == pytest.approx(
        inputs["targets"]["target_command"][201, 0] -
        inputs["targets"]["target_command"][1, 0])
    assert "R3_global_mean" in norm
    assert np.array_equal(raw["baseline_teacher_q_difference"],
                          -raw["baseline_original_teacher_q_difference"])
    assert len(report["bins"]) == 10

    def fail_after_batch(_rep, _leg, _batch, _rows):
        raise RuntimeError("generated partial")

    partial = {}
    with pytest.raises(RuntimeError, match="generated partial"):
        e246.analyze_arrays(inputs, raw=partial, strict_original=False,
                            progress=fail_after_batch)
    assert partial["completed_query_rows"][0, 0] == 2000
    assert partial["completed_query_rows"][0, 1] == 0
    assert np.isnan(partial["nearest_train_distance"][0, 1, 1])


def test_publication_late_fault_invalidates_and_manifest_tracks_latest(tmp_path, monkeypatch):
    result = {"validity": "valid_saved_diagnostic",
              "separates_better_under_fixed_metric": {"R0": True},
              "analysis": {"predicates": {"R0": {
                  "separates_better_under_fixed_metric": True,
                  "checks": {"steady/global": True}, "failed": []}},
                  "segments": {"steady": {"R0": {"global": {"q_rms_rad": .1}}}}}}
    ledger = {"errors": [], "completed_query_rows": [[0]*6 for _ in e246.REPS]}
    real_publish = e246.publish
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("generated late fault")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(e246, "publish", fail_once)
    first = e246.settle_publication(tmp_path, result, {"x": np.array([2])},
        {"y": np.array([3])}, ledger, elapsed=lambda: .2,
        deadline=e246.time.perf_counter()+10)
    assert isinstance(first, TimeoutError)
    saved = json.loads((tmp_path / "result.json").read_bytes())
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    assert saved["separates_better_under_fixed_metric"] is None
    assert saved["analysis"]["predicates"]["R0"]["separates_better_under_fixed_metric"] is None
    assert saved["analysis"]["predicates"]["R0"]["checks"] == {"steady/global": True}
    assert saved["analysis"]["segments"]["steady"]["R0"]["global"]["q_rms_rad"] == .1
    assert saved["validity"] == "incomplete_or_invalid"
    assert saved["finalization_pending"] is False
    assert len(json.loads((tmp_path / "errors.json").read_bytes())) == 1
    for name, entry in manifest["files"].items():
        assert e246.sha(tmp_path / name) == entry["sha256"]
        assert (tmp_path / name).stat().st_size == entry["bytes"]


def test_raw_first_fault_keeps_other_raw_and_terminal_pending(tmp_path, monkeypatch):
    original_write = e246.write_npz_once
    calls = 0

    def raw_first_fault(out, filename, arrays):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("generated raw first fault")
        return original_write(out, filename, arrays)

    monkeypatch.setattr(e246, "write_npz_once", raw_first_fault)
    result = {"validity": "valid_saved_diagnostic",
              "separates_better_under_fixed_metric": {"R1": True}}
    ledger = {"errors": []}
    first = e246.settle_publication(tmp_path, result, {"x": np.array([1])},
        {"mean": np.array([2.])}, ledger, elapsed=lambda: .1,
        deadline=e246.time.perf_counter()+10)
    assert isinstance(first, OSError)
    assert (tmp_path / "normalization.npz").exists()
    assert (tmp_path / "diagnostic-raw.npz").exists()
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    assert e246.sha(tmp_path / "errors.json") == manifest["files"]["errors.json"]["sha256"]
    assert e246.sha(tmp_path / "ledger.json") == manifest["files"]["ledger.json"]["sha256"]

    def persistent_fault(*_args, **_kwargs):
        raise OSError("generated persistent publication fault")

    stuck = tmp_path / "stuck"
    stuck.mkdir()
    monkeypatch.setattr(e246, "publish", persistent_fault)
    result2 = {"validity": "valid_saved_diagnostic",
               "separates_better_under_fixed_metric": {"R1": True},
               "analysis": {"predicates": {"R1": {
                   "separates_better_under_fixed_metric": True,
                   "checks": {"observed_tail/rh": False}}}}}
    ledger2 = {"errors": []}
    e246.settle_publication(stuck, result2, {}, {}, ledger2,
        elapsed=lambda: .1, deadline=e246.time.perf_counter()+10)
    saved = json.loads((stuck / "result.json").read_bytes())
    assert saved["finalization_pending"] is True
    assert saved["separates_better_under_fixed_metric"] is None
    assert saved["analysis"]["predicates"]["R1"]["separates_better_under_fixed_metric"] is None
    assert saved["analysis"]["predicates"]["R1"]["checks"] == {
        "observed_tail/rh": False}
    assert len(saved["errors"]) == 4
    assert not (stuck / "output-manifest.json").exists()


def test_gate_rejects_before_loading_original(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("original NPZ load occurred")

    monkeypatch.setattr(e246, "load", forbidden)
    with pytest.raises(ValueError, match="freeze SHA"):
        e246.run(tmp_path / "absent-freeze", "wrong", tmp_path / "review", "wrong",
                 tmp_path / "generated-result", generated_test_output=True)
    saved = json.loads((tmp_path / "generated-result" / "result.json").read_bytes())
    assert saved["original_calculation_count"] == 0
    assert saved["loaded_input_roles"] == []
