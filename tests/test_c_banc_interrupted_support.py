"""Generated-data E152 tests; no frozen recording is opened."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_interrupted_support as e152


def generated_native(blocks=2):
    rows = blocks * 50
    time_s = np.arange(1, rows + 1, dtype=np.float64) * 0.0001
    load = np.zeros((rows, 6), dtype=np.float64)
    slip = np.full((rows, 6), np.nan, dtype=np.float64)
    return time_s, load, slip


def test_native_exact_50_to_1_endpoints_and_weighted_sums():
    time_s, load, slip = generated_native()
    load[:25, 0] = 0.1
    slip[:25, 0] = 2.0
    load[50:, 0] = 0.2
    slip[50:, 0] = 4.0
    blocks, provenance = e152.native_blocks(time_s, load, slip, None)
    assert provenance == "derived_isfinite_E85"
    assert np.array_equal(blocks["endpoint_time_s"], [0.005, 0.01])
    assert np.allclose(blocks["mean_load_bw"][:, 0], [0.05, 0.2])
    assert np.allclose(blocks["supported_duty"][:, 0], [0.5, 1.0])
    assert np.allclose(blocks["supported_load_sum_bw"][:, 0], [2.5, 10.0])
    assert np.allclose(blocks["supported_load_slip_sum_bw_mm_s"][:, 0], [5.0, 40.0])
    aggregate = e152.aggregate_native(blocks, np.ones(2, dtype=bool))
    assert aggregate["per_leg"][0][
        "supported_load_weighted_slip_mm_s"
    ] == pytest.approx(3.6)
    assert aggregate["overall"]["supported_load_weighted_slip_mm_s"] == pytest.approx(
        3.6
    )


def test_native_rejects_shifted_clock_and_supported_nan_only():
    time_s, load, slip = generated_native()
    with pytest.raises(ValueError, match="clock"):
        e152.native_blocks(time_s + 0.0001, load, slip, None)
    load[0, 0] = 0.02
    with pytest.raises(ValueError, match="Supported native slip"):
        e152.native_blocks(time_s, load, slip, None)
    load[0, 0] = 0.019
    blocks, _ = e152.native_blocks(time_s, load, slip, None)
    assert blocks["supported_load_sum_bw"][0, 0] == 0
    with pytest.raises(ValueError, match="shape mismatch"):
        e152.native_blocks(time_s, load, slip, None, strict_rows=True)


def test_recorded_validity_must_be_boolean_and_match_finiteness():
    time_s, load, slip = generated_native()
    slip[:] = 1.0
    valid = np.ones_like(slip, dtype=bool)
    _, provenance = e152.native_blocks(time_s, load, slip, valid)
    assert provenance == "recorded"
    valid[0, 0] = False
    with pytest.raises(ValueError, match="disagrees"):
        e152.native_blocks(time_s, load, slip, valid)
    with pytest.raises(ValueError, match="Boolean"):
        e152.native_blocks(time_s, load, slip, valid.astype(np.uint8))


def test_rotating_body_coordinates_and_world_projection_are_distinct():
    rows = 9
    position = np.zeros((rows, 3))
    basis = np.repeat(np.eye(3)[None], rows, axis=0)
    # Body forward points along world +Y for every sample.
    basis[:, :, 0] = [0, 1, 0]
    basis[:, :, 1] = [-1, 0, 0]
    foot = np.zeros((rows, 6, 3))
    foot[:, :, 1] = np.arange(rows)[:, None] * e152.CONTROL_DT * 3.0
    foot[:, :, 2] = 2.0
    result = e152.foot_kinematics(position, basis, foot)
    assert np.allclose(result["ego_ap_mm"], foot[:, :, 1])
    assert np.allclose(result["ego_height_mm"], 2.0)
    assert np.allclose(result["ego_ap_velocity_mm_s"], 3.0)
    assert np.allclose(result["world_forward_velocity_mm_s"], 3.0)


def test_empty_group_is_null_and_not_zero():
    time_s, load, slip = generated_native()
    blocks, _ = e152.native_blocks(time_s, load, slip, None)
    rows = 3
    summary = e152.aggregate_group(
        np.zeros(rows, dtype=bool),
        np.arange(rows),
        np.arange(rows),
        np.zeros((rows, 6)),
        np.zeros((rows, 6)),
        np.zeros((rows, 6)),
        np.zeros((rows, 6), dtype=bool),
        blocks,
    )
    assert summary["speed_mm_s"] is None
    assert summary["native"]["per_leg"] is None
    assert summary["boundary_supported_forward_ego_AP_fraction_overall"] is None
    assert summary["boundary_supported_forward_ego_AP_fraction_by_leg"] == [None] * 6


def test_all_64_support_patterns_and_fixed_bit_order():
    support = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(bool)
    patterns = e152.support_patterns(support, np.arange(64), np.ones(64, dtype=bool))
    assert len(patterns) == 64
    assert all(item["samples"] == 1 for item in patterns)
    assert patterns[1]["bits_LF_LM_LH_RF_RM_RH"] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert patterns[32]["bits_LF_LM_LH_RF_RM_RH"] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert patterns[37]["speed_median_mm_s"] == 37


def test_event_windows_exclude_both_edges_and_align_41_rows():
    rows = 100
    window = np.zeros(rows, dtype=bool)
    window[10:90] = True
    speed = np.zeros(rows)
    for crossing in (15, 50, 85):
        speed[crossing - 1] = 1.0
    events = e152.interruption_windows(speed, window)
    assert [item["index"] for item in events["all"]] == [15, 50, 85]
    assert events["included_indices"].tolist() == [50]
    assert events["all"][0]["exclusion_reasons"] == ["pre_outside_window"]
    assert events["all"][2]["exclusion_reasons"] == ["post_outside_window"]
    stacks, summary = e152.stack_events({"value": np.arange(rows)}, events)
    assert stacks["value"].shape == (1, 41)
    assert stacks["value"][0].tolist() == list(range(30, 71))
    assert summary["events"][0]["values"]["value"] == {
        "pre_mean": 39.5,
        "post_mean": 59.5,
    }


def test_source_gate_requires_exact_current_two_file_hashes(tmp_path):
    review = tmp_path / "review.json"
    source_map = {name: e152.sha256(e152.ROOT / name) for name in e152.OWNED}
    payload = {
        "accepts_single_saved_diagnostic": True,
        "source_sha256": source_map,
        "freeze_sha256": e152.FREEZE_SHA,
        "original_processes_allowed": 1,
    }
    review.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    assert e152.validate_source_gate(review, digest) == payload
    for key, wrong, message in (
        ("freeze_sha256", "wrong", "freeze SHA"),
        ("original_processes_allowed", 0, "exactly one original process"),
    ):
        invalid = {**payload, key: wrong}
        review.write_text(json.dumps(invalid), encoding="utf-8")
        digest = hashlib.sha256(review.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match=message):
            e152.validate_source_gate(review, digest)
    payload["source_sha256"][e152.OWNED[0]] = "0" * 64
    review.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source-review SHA map"):
        e152.validate_source_gate(review, digest)


def test_frozen_input_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"frozen")
    monkeypatch.setattr(e152, "ROOT", tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    e152.verify_hash_map({"input.bin": digest})
    with pytest.raises(ValueError, match="Frozen hash mismatch"):
        e152.verify_hash_map({"input.bin": "0" * 64})


def test_direct_cli_help_from_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(e152.ROOT / e152.OWNED[0]), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze" in completed.stdout
    assert "--output" in completed.stdout


def test_preflight_time_is_charged_before_any_recording_read(tmp_path, monkeypatch):
    clock = [100.0]
    freeze = {
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
        "recordings": [{"label": "must-not-read"}],
        "resource_limits": {"free_bytes": 0, "internal_wall_s": 35},
    }
    reads = []

    def slow_freeze(*_args):
        clock[0] += 40.0
        return freeze

    monkeypatch.setattr(e152, "load_freeze", slow_freeze)
    monkeypatch.setattr(e152, "validate_source_gate", lambda *_args: {})
    monkeypatch.setattr(e152, "verify_hash_map", lambda *_args: None)
    monkeypatch.setattr(e152, "threadpool_info", list)
    monkeypatch.setattr(e152.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(e152, "load_model", lambda *_args: reads.append("read"))
    output = tmp_path / "late-preflight"
    with pytest.raises(TimeoutError, match="preflight"):
        e152.run(Path("generated-freeze"), "", Path("generated-gate"), "", output)
    assert reads == []
    assert not output.exists()


def test_raw_features_survive_later_summary_failure(tmp_path, monkeypatch):
    trace = tmp_path / "generated-trace.json"
    trace.write_text("{}", encoding="utf-8")
    native_path = tmp_path / "generated-native.npz"
    np.savez(native_path, generated=np.array([1]))
    time_s, load, slip = generated_native()
    blocks, provenance = e152.native_blocks(time_s, load, slip, None)
    rows = 3
    model = {
        "t": np.arange(rows) * e152.CONTROL_DT,
        "window": np.array([False, True, True]),
        "speed": np.array([0.0, 1.0, 0.0]),
        "yaw_rate": np.zeros(rows),
    }
    physical = {
        "time_s": model["t"].copy(),
        "position": np.zeros((rows, 3)),
        "basis": np.repeat(np.eye(3)[None], rows, axis=0),
        "foot_world": np.zeros((rows, 6, 3)),
        "target": np.zeros((rows, 42)),
        "actual": np.zeros((rows, 42)),
        "boundary_load": np.zeros((rows, 6)),
        "boundary_support": np.zeros((rows, 6), dtype=bool),
        "qpos0": np.zeros(1),
        "qvel0": np.zeros(1),
    }
    freeze = {
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
        "recordings": [
            {
                "label": "generated",
                "role": "generated only",
                "trace": str(trace),
                "native": str(native_path),
                "body_hash": "generated",
            }
        ],
        "resource_limits": {
            "free_bytes": 0,
            "internal_wall_s": 35,
            "external_wall_s": 45,
            "raw_bytes": 1024 * 1024,
            "prior_actual_s": 0,
            "prospective_cumulative_s": 0,
        },
    }
    monkeypatch.setattr(e152, "load_freeze", lambda *_args: freeze)
    monkeypatch.setattr(e152, "validate_source_gate", lambda *_args: {})
    monkeypatch.setattr(e152, "verify_hash_map", lambda *_args: None)
    monkeypatch.setattr(e152, "threadpool_info", list)
    monkeypatch.setattr(e152, "load_model", lambda *_args: model)
    monkeypatch.setattr(
        e152,
        "validate_recording",
        lambda *_args: (blocks, provenance, physical),
    )
    primary = RuntimeError("generated-summary-failure")
    monkeypatch.setattr(
        e152,
        "summarize_recording_features",
        lambda *_args: (_ for _ in ()).throw(primary),
    )
    output = tmp_path / "summary-failure"
    with pytest.raises(RuntimeError) as caught:
        e152.run(Path("generated-freeze"), "", Path("generated-gate"), "", output)
    assert caught.value is primary
    with np.load(output / "generated/features.npz", allow_pickle=False) as raw:
        assert raw["tracking_RMS_rad_by_leg"].shape == (3, 6)
        assert raw["native_endpoint_time_s"].shape == (2,)
    failed = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert failed["accepted"] is False
    assert failed["error"]["message"] == "generated-summary-failure"


def test_final_publication_time_and_failure_are_fail_closed(tmp_path):
    path = tmp_path / "result.json"
    result = {"accepted": True, "resource_limit_met": True}
    calls = []

    def writer(target, value):
        calls.append(dict(value))
        Path(target).write_text(json.dumps(value), encoding="utf-8")

    clock_values = iter((0.5, 2.0))
    with pytest.raises(TimeoutError, match="final result publication"):
        e152.publish_final_result(
            path, result, 0.0, 1.0, writer=writer, clock=lambda: next(clock_values)
        )
    assert calls[-1]["accepted"] is False
    assert calls[-1]["resource_limit_met"] is False
    assert json.loads(path.read_text(encoding="utf-8"))["accepted"] is False

    primary = OSError("publication failed")

    def failing_writer(_target, _value):
        raise primary

    with pytest.raises(OSError) as caught:
        e152.publish_final_result(path, result, 0.0, 1.0, writer=failing_writer)
    assert caught.value is primary

    protected = tmp_path / "protected.json"
    publication_calls = 0

    def fail_final_writer(target, value):
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 2:
            raise primary
        Path(target).write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(OSError) as caught:
        e152.publish_final_result(
            protected,
            {"accepted": True},
            0.0,
            1.0,
            writer=fail_final_writer,
            clock=lambda: 0.5,
        )
    assert caught.value is primary
    assert json.loads(protected.read_text(encoding="utf-8"))["accepted"] is False
