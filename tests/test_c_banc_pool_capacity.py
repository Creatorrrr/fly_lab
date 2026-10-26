"""Generated-fixture tests for E143; no original E142 arrays are analyzed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_pool_capacity as e143


def synthetic_pool() -> tuple[np.ndarray, np.ndarray]:
    membership = np.r_[np.arange(96), np.zeros(391 - 96, dtype=int)]
    pool = np.zeros((96, 391), dtype=np.float64)
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        pool[muscle, columns] = 1 / columns.size
    return membership, pool


def synthetic_decoder():
    motor_ids = np.arange(391) + 2000
    observed_ids = np.r_[motor_ids[10:], motor_ids[:10], np.arange(55) + 100]
    membership, pool = synthetic_pool()
    moment = np.zeros((42, 96)); moment[0, 0] = 2; moment[0, 1] = -3
    rest = np.linspace(-0.1, 0.1, 42)
    ltm = np.arange(6) * 16 + 15
    muscles = [
        {
            "leg": e143.LEGS[index // 16],
            "muscle": "long_tendon_muscle" if index % 16 == 15 else f"muscle_{index}",
            "columns": np.flatnonzero(pool[index]).tolist(),
        }
        for index in range(96)
    ]
    cal = {
        "motor_ids": motor_ids,
        "pool": pool,
        "membership": membership,
        "moment": moment,
        "rest": rest,
        "ltm": ltm,
    }
    meta = {"motor_ids": motor_ids.copy(), "observed_ids": observed_ids.copy()}
    signal = {"observed_ids": observed_ids.copy()}
    anatomy = {
        "graph_hash": "fixture-graph",
        "motor_roster": [f"flywire:banc:888:{720575941000000000 + index}" for index in range(391)],
        "muscles": muscles,
    }
    start = {
        "control_tick": 0,
        "graph_hash": "fixture-graph",
        "network": {"tick": 0, "rate": np.zeros(158706)},
        "adapter": {
            "groups": muscles,
            "activation": np.zeros(96),
            "original": {"original": {"filtered": np.zeros(391)}},
            "calibration": {key: np.asarray(cal[key]).copy()
                            for key in ("motor_ids", "moment", "rest", "pool", "ltm")},
        },
    }
    return cal, meta, signal, anatomy, start


def test_signed_convex_and_derivative_envelopes_include_static_pool():
    membership, pool = synthetic_pool()
    filtered = np.zeros((2000, 391), dtype=np.float64)
    ramp = np.arange(2000) * 0.005
    # Muscle 0 has many MNs.  Include positive and negative slopes/rates.
    columns = np.flatnonzero(membership == 0)
    filtered[:, columns[0]] = 2 + 3 * ramp
    filtered[:, columns[1]] = 9 - 2 * ramp
    filtered[:, columns[2:]] = 4
    # Muscle 1 is a single-MN pool and therefore has a zero-width envelope.
    filtered[:, 1] = 1.5 + 0.25 * ramp
    moment = np.zeros((42, 96)); moment[0, 0] = 2; moment[0, 1] = -3
    rest = np.linspace(-0.2, 0.2, 42)
    ltm = np.arange(6) * 16 + 15

    result = e143.pool_envelopes(filtered, pool, membership, moment, rest, ltm)
    assert np.array_equal(result["activation_low"][:, 1], filtered[:, 1])
    assert np.array_equal(result["activation_high"][:, 1], filtered[:, 1])
    expected_low = rest[0] + 0.02 * (
        2 * result["activation_low"][:, 0] - 3 * result["activation_high"][:, 1]
    )
    expected_high = rest[0] + 0.02 * (
        2 * result["activation_high"][:, 0] - 3 * result["activation_low"][:, 1]
    )
    assert np.array_equal(result["command_low"][:, 0], expected_low)
    assert np.array_equal(result["command_high"][:, 0], expected_high)
    assert np.all(result["command"] >= result["command_low"] - 1e-12)
    assert np.all(result["command"] <= result["command_high"] + 1e-12)
    assert np.all(result["command_velocity"] >= result["velocity_low"] - 1e-12)
    assert np.all(result["command_velocity"] <= result["velocity_high"] + 1e-12)
    assert np.max(np.abs(np.diff(result["activation"], axis=0) / 0.005
                         - result["motor_derivative"] @ pool.T)) < 1e-12


def test_outside_distance_and_zero_baseline_ratio_use_strict_json_null():
    target = np.array([[-2.0, 0.5, 4.0]])
    low = np.array([[-1.0, 0.0, 1.0]])
    high = np.array([[1.0, 1.0, 3.0]])
    assert np.array_equal(e143.outside_distance(target, low, high), [[1.0, 0.0, 1.0]])
    baseline = np.zeros((2, 42)); target42 = np.zeros_like(baseline)
    summary = e143.metric_summary(baseline, target42, np.zeros_like(baseline))
    assert summary["baseline_RMS"] == 0
    assert summary["floor_to_baseline_ratio"] is None
    assert all(item["floor_to_baseline_ratio"] is None for item in summary["per_joint"])


def test_zero_pointwise_floor_does_not_imply_one_static_weight():
    # output = w*MN1 + (1-w)*MN2.  The two rows respectively require w=0 and
    # w=1 to hit target 2, despite target 2 lying in both pointwise intervals.
    basis = np.array([[1.0, 2.0], [2.0, 1.5]])
    target = np.array([2.0, 2.0])
    assert np.array_equal(e143.outside_distance(target, basis.min(1), basis.max(1)), [0, 0])
    required_w = np.array([
        (target[index] - basis[index, 1]) / (basis[index, 0] - basis[index, 1])
        for index in range(2)
    ])
    assert np.array_equal(required_w, [0, 1])


def test_ema_counts_exactly_2000_stored_updates():
    rates = np.zeros((2000, 391)); rates[:, 0] = 10
    progress = {"recorded_EMA_updates_completed": 0}
    filtered = e143.ema_filter(rates, progress)
    assert progress["recorded_EMA_updates_completed"] == 2000
    assert filtered.shape == rates.shape
    assert filtered[0, 0] == pytest.approx(10 * (1 - np.exp(-0.5)))


@pytest.mark.parametrize("defect", ["support", "normalization", "nan"])
def test_invalid_pool_and_nonfinite_data_fail_closed(defect):
    membership, pool = synthetic_pool()
    filtered = np.zeros((2000, 391))
    if defect == "support":
        pool[0, 0] = 0; pool[1, 0] = 1
    elif defect == "normalization":
        pool[0] *= 0.5
    else:
        filtered[5, 7] = np.nan
    with pytest.raises(ValueError):
        e143.pool_envelopes(filtered, pool, membership, np.zeros((42, 96)),
                            np.zeros(42), np.arange(6) * 16 + 15)


def test_decoder_anatomy_start_zero_and_observed_order_failures():
    cal, meta, signal, anatomy, start = synthetic_decoder()
    values = e143.validate_decoder(cal, meta, signal, anatomy, start)
    assert np.array_equal(values[0], cal["motor_ids"])
    rate = np.zeros((2000, 446)); rate[:, 0] = 3
    signal["rate"] = rate
    ordered = e143.ordered_motor_rate(signal, values[0], values[1])
    assert ordered.shape == (2000, 391)

    cal["membership"][0] = 1
    with pytest.raises(ValueError, match="declared muscle"):
        e143.validate_decoder(cal, meta, signal, anatomy, start)
    cal["membership"][0] = 0
    start["network"]["rate"][0] = 1
    with pytest.raises(ValueError, match="whole-zero"):
        e143.validate_decoder(cal, meta, signal, anatomy, start)
    start["network"]["rate"][0] = 0
    signal["rate"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        e143.ordered_motor_rate(signal, values[0], values[1])


def test_clock_requires_t0_plus_exact_endpoint_mapping():
    rows = [{"control_tick": index, "time_s": index * 0.005} for index in range(2001)]
    endpoint = np.arange(1, 2001) * 0.005
    assert np.array_equal(e143.verify_clock(rows, endpoint), np.arange(2001) * 0.005)
    rows[10]["control_tick"] = 11
    with pytest.raises(ValueError, match="ticks"):
        e143.verify_clock(rows, endpoint)
    rows[10]["control_tick"] = 10
    shifted = endpoint.copy(); shifted[20] += 1e-6
    with pytest.raises(ValueError, match=r"rows\[1:\]"):
        e143.verify_clock(rows, shifted)


def test_target_phase_has_t0_zero_and_only_adjacent_mask_pairs():
    times = np.arange(2001) * 0.005
    observed = np.r_[np.asarray(e143.E1_IDS), np.arange(440) + 900000]
    rate = np.zeros((2000, 446))
    for leg in range(6):
        rate[:, leg] = 60 + 50 * np.sin(2 * np.pi * (times[1:] / 0.4 + leg / 2))
    signal = {"rate": rate}
    reference = {
        "time_s": times[:-1],
        "phase": np.column_stack([2 * np.pi * (times[:-1] / 0.4 + leg / 2)
                                  for leg in range(6)]),
        "command": np.column_stack([np.sin(2 * np.pi * times[:-1] / 0.4 + joint / 8)
                                    for joint in range(42)]),
    }
    result = e143.target_basis(times, signal, observed, reference)
    indices = result["target_indices"]
    assert np.array_equal(result["source_rate"][0], np.zeros(6))
    assert indices[0] == 100
    assert np.all(np.diff(indices) == 1)
    assert np.isfinite(result["target_rad"]).all()
    assert result["reference_residual_rms_rad"] < 0.002


def minimal_frozen_contract():
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 25,
        "maximum_raw_bytes": 33554432,
    }


def stub_success(monkeypatch, *, maximum_raw_bytes=33554432):
    freeze = minimal_frozen_contract()
    freeze["maximum_raw_bytes"] = maximum_raw_bytes
    review = {"source_sha256": {name: "fixture" for name in e143.OWNED}}
    monkeypatch.setattr(e143, "reviewed_inputs", lambda *args: (freeze, review))
    monkeypatch.setattr(e143, "load_inputs", lambda _freeze: {})

    def fake_calculate(_data, _freeze, progress, partial):
        progress["recorded_EMA_updates_attempted"] = 2000
        progress["recorded_EMA_updates_completed"] = 2000
        partial["basis"] = np.arange(3)
        return {"basis": np.arange(3)}, {"fixture": True}

    monkeypatch.setattr(e143, "calculate", fake_calculate)
    monkeypatch.setattr(e143, "assert_numerics", lambda *args: None)
    return freeze


def test_source_gate_failure_precedes_input_load_and_preserves_failure(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(e143, "reviewed_inputs", lambda *args: (_ for _ in ()).throw(
        ValueError("source gate rejected")))
    monkeypatch.setattr(e143, "load_inputs", lambda freeze: loaded.append(freeze))
    output = tmp_path / "gate-failure"
    with pytest.raises(ValueError, match="source gate rejected"):
        e143.run(Path("freeze"), "bad", Path("review"), "bad", output)
    assert loaded == []
    assert (output / "execution-raw.json").exists()
    assert (output / "summary.json").exists()


def test_numerical_failure_saves_raw_and_exact_ema_counts(tmp_path, monkeypatch):
    freeze = minimal_frozen_contract()
    review = {"source_sha256": {name: "fixture" for name in e143.OWNED}}
    monkeypatch.setattr(e143, "reviewed_inputs", lambda *args: (freeze, review))
    monkeypatch.setattr(e143, "load_inputs", lambda _freeze: {"fixture": True})

    def fake_calculate(data, frozen, progress, partial):
        assert data == {"fixture": True} and frozen is freeze
        progress["recorded_EMA_updates_attempted"] = 2000
        progress["recorded_EMA_updates_completed"] = 2000
        partial["basis"] = np.arange(4)
        return {"basis": np.arange(4), "difference": np.ones(3)}, {"bad": True}

    monkeypatch.setattr(e143, "calculate", fake_calculate)
    monkeypatch.setattr(e143, "assert_numerics", lambda *args: (_ for _ in ()).throw(
        ValueError("numerical assertion")))
    output = tmp_path / "numerical-failure"
    with pytest.raises(ValueError, match="numerical assertion"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review-sha", output)
    assert (output / "pool-capacity-raw.npz").exists()
    execution = json.loads((output / "execution-raw.json").read_text())
    assert execution["recorded_EMA_updates_completed"] == 2000
    assert execution["original_neural_steps"] == execution["physics_steps"] == 0
    assert execution["fit_calls"] == execution["LP_calls"] == 0


def test_raw_save_failure_falls_back_to_partial_and_preserves_primary(tmp_path, monkeypatch):
    freeze = minimal_frozen_contract()
    review = {"source_sha256": {name: "fixture" for name in e143.OWNED}}
    monkeypatch.setattr(e143, "reviewed_inputs", lambda *args: (freeze, review))
    monkeypatch.setattr(e143, "load_inputs", lambda _freeze: {})
    def fake_calculate(_data, _freeze, _progress, partial):
        partial["basis"] = np.arange(3)
        return {"basis": np.arange(3)}, {}

    monkeypatch.setattr(e143, "calculate", fake_calculate)
    original = e143.save_npz_new

    def fail_primary(path, arrays):
        if Path(path).name == "pool-capacity-raw.npz":
            raise OSError("primary raw failure")
        return original(path, arrays)

    monkeypatch.setattr(e143, "save_npz_new", fail_primary)
    output = tmp_path / "raw-failure"
    with pytest.raises(OSError, match="primary raw failure"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review-sha", output)
    assert (output / "partial-raw.npz").exists()
    with np.load(output / "partial-raw.npz") as raw:
        assert np.array_equal(raw["basis"], np.arange(3))


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"; output.mkdir()
    marker = output / "marker"; marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        e143.run(Path("freeze"), "bad", Path("review"), "bad", output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_execution_receipt_failure_invalidates_authoritative_summary(tmp_path, monkeypatch):
    stub_success(monkeypatch)
    original = e143.save_json_new

    def fail_execution(path, value):
        if Path(path).name == "execution-raw.json":
            raise OSError("generated execution receipt failure")
        return original(path, value)

    monkeypatch.setattr(e143, "save_json_new", fail_execution)
    output = tmp_path / "execution-failure"
    with pytest.raises(OSError, match="generated execution receipt failure"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review", output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["validity"] == "invalid_incomplete"
    assert not (output / "invalidated-success-summary.json").exists()
    assert (output / "secondary-errors.json").exists()


def test_wall_limit_after_execution_receipt_invalidates_success(tmp_path, monkeypatch):
    stub_success(monkeypatch)
    clock = {"now": 0.0}
    monkeypatch.setattr(e143.time, "perf_counter", lambda: clock["now"])
    original = e143.save_json_new

    def advance_after_execution(path, value):
        result = original(path, value)
        if Path(path).name == "execution-raw.json":
            clock["now"] = 26.0
        return result

    monkeypatch.setattr(e143, "save_json_new", advance_after_execution)
    output = tmp_path / "late-wall"
    with pytest.raises(TimeoutError, match="during before_summary"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review", output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    finalization = json.loads((output / "finalization-errors.json").read_text())
    assert "before_summary:internal_wall_limit_s" in finalization["failed_resource_checks"]


def test_summary_write_failure_never_leaves_authoritative_pass(tmp_path, monkeypatch):
    stub_success(monkeypatch)
    original = e143.save_json_new

    def fail_summary(path, value):
        if Path(path).name == "summary.json":
            raise OSError("generated summary write failure")
        return original(path, value)

    monkeypatch.setattr(e143, "save_json_new", fail_summary)
    output = tmp_path / "summary-failure"
    with pytest.raises(OSError, match="generated summary write failure"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review", output)
    assert not (output / "summary.json").exists()
    finalization = json.loads((output / "finalization-errors.json").read_text())
    assert finalization["status"] == "failed"
    assert finalization["failed_persistence_operations"][0]["operation"] == "save summary.json"


def test_final_summary_byte_crossing_moves_pass_out_of_authoritative_path(tmp_path, monkeypatch):
    freeze = stub_success(monkeypatch, maximum_raw_bytes=100)
    calls = {"count": 0}

    def generated_size(_output):
        calls["count"] += 1
        return 0 if calls["count"] <= 2 else freeze["maximum_raw_bytes"] + 1

    monkeypatch.setattr(e143, "output_bytes", generated_size)
    output = tmp_path / "late-bytes"
    with pytest.raises(RuntimeError, match="during after_summary"):
        e143.run(Path("freeze"), e143.FREEZE_SHA, Path("review"), "review", output)
    summary = json.loads((output / "summary.json").read_text())
    invalidated = json.loads((output / "invalidated-success-summary.json").read_text())
    assert summary["status"] == "failed"
    assert invalidated["status"] == "pass"
    assert "after_summary:maximum_raw_bytes" in json.loads(
        (output / "finalization-errors.json").read_text()
    )["failed_resource_checks"]


def test_complete_summary_then_close_error_invalidates_success(tmp_path, monkeypatch):
    stub_success(monkeypatch)
    writer = e143.save_json_new

    def close_error(path, value):
        writer(path, value)
        if path.name == "summary.json" and value.get("status") == "pass":
            raise OSError("summary close error")

    monkeypatch.setattr(e143, "save_json_new", close_error)
    output = tmp_path / "close-error"
    with pytest.raises(OSError, match="summary close error"):
        e143.run(Path("freeze"), "fixture", Path("review"), "fixture", output)
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    assert (output / "invalidated-success-summary.json").exists()


@pytest.mark.parametrize("primary_failure", [False, True])
def test_resource_stat_failure_preserves_primary_and_invalidates_pass(tmp_path, monkeypatch, primary_failure):
    stub_success(monkeypatch)
    real_size = e143.output_bytes
    calls = 0

    def size(path):
        nonlocal calls
        calls += 1
        if primary_failure or calls >= 3:
            raise OSError("resource stat error")
        return real_size(path)

    if primary_failure:
        def numerical(*_args):
            raise ValueError("original numerical error")
        monkeypatch.setattr(e143, "assert_numerics", numerical)
    monkeypatch.setattr(e143, "output_bytes", size)
    output = tmp_path / "stat-error"
    expected = ValueError if primary_failure else OSError
    with pytest.raises(expected, match="original numerical error" if primary_failure else "resource stat error"):
        e143.run(Path("freeze"), "fixture", Path("review"), "fixture", output)
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    receipt = json.loads((output / "finalization-errors.json").read_text())
    assert receipt["bytes_last_observed"] is None
    assert any("output_bytes_unavailable" in item for item in receipt["failed_resource_checks"])


def test_returned_resource_receipt_includes_summary_persistence(tmp_path, monkeypatch):
    stub_success(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(e143.time, "perf_counter", lambda: clock[0])
    writer = e143.save_json_new

    def after_summary(path, value):
        writer(path, value)
        if path.name == "summary.json":
            clock[0] = 7.0

    monkeypatch.setattr(e143, "save_json_new", after_summary)
    output = tmp_path / "complete-receipt"
    result = e143.run(Path("freeze"), "fixture", Path("review"), "fixture", output)
    receipt = result["final_resources"]
    assert receipt["wall_s_after_required_persistence"] == 7
    assert receipt["bytes_after_required_persistence"] == e143.output_bytes(output)
    assert receipt["all_required_persistence_and_resource_checks_passed"] is True
