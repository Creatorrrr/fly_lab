"""Synthetic E190 clock, sign, support, lag, and rejection checks."""
import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_vertical_sequence as e190


def test_sign_bins_weighted_denominators_and_missing_contact():
    force = np.zeros(50)
    force[:3] = [-2.0, -1.0, 3.0]
    load = np.zeros(50)
    load[:4] = [0.02, 0.08, 0.04, 0.20]
    available = np.zeros(50, dtype=bool)
    available[:3] = True
    z = {name: np.full(50, np.nan) for name in e190.Z_NAMES}
    for value in z.values():
        value[:3] = [-2.0, -4.0, 5.0]
    result = e190.bin_metrics(force, load, available, z)
    assert result["ap_signed_mean"] == 0
    assert result["ap_negative_magnitude_mean"] == 3 / 50
    assert result["ap_positive_magnitude_mean"] == 3 / 50
    assert result["ap_zero_count"] == 47
    assert result["negative_count"] == 2
    assert result["negative_load_denominator"] == pytest.approx(.10)
    assert result["negative_command_z"] == pytest.approx(-3.6)
    assert result["positive_load_denominator"] == pytest.approx(.04)
    assert result["positive_command_z"] == 5
    assert result["all_count"] == 3  # supported fourth row lacks contact geometry
    assert result["negative_command_downward_fraction"] == 1
    assert result["positive_command_downward_fraction"] == 0
    missing = e190.bin_metrics(np.zeros(50), np.zeros(50), np.zeros(50, dtype=bool),
                               {name: np.full(50, np.nan) for name in e190.Z_NAMES})
    assert not missing["all_valid"]
    assert np.isnan(missing["all_command_z"])
    assert missing["all_load_denominator"] == 0


def test_lag_direction_pair_count_and_variance_rules():
    x = np.array([0, 3, 1, 6, 2, 9, 4, 8, 5, 11, 7, 14, 10, 13, 12, 16, 15, 19, 17, 18], dtype=float)
    y = np.r_[0., 0., x[:-2]]
    rows = [e190.lag_pearson(x, y, lag) for lag in e190.LAGS]
    assert rows[4]["lag_control"] == 2
    assert rows[4]["sample_pairs"] == 18
    assert rows[4]["pearson_r"] == pytest.approx(1)
    assert rows[0]["pearson_r"] < 1
    no_support = x.copy()
    no_support[:13] = np.nan
    short = e190.lag_pearson(no_support, y, 0)
    assert short["sample_pairs"] == 7 and not short["available"]
    constant = e190.lag_pearson(np.ones(20), y, 0)
    assert not constant["available"] and constant["reason"] == "zero_variance"


def synthetic_join():
    n = 10000
    time = np.arange(1, 30001) * .0001
    clock = 201 + np.arange(n) // 50
    native = {"native_index": np.arange(n), "control_tick": clock, "time_s": time[10000:20000].copy(),
              "rotation_world": np.zeros((n, 3, 3)), "heading_horizontal": np.zeros((n, 3)),
              "force_world_bw": np.zeros((n, 6, 3)), "force_forward_bw": np.zeros((n, 6)),
              "normal_load_bw": np.zeros((n, 6)), "slip_mm_s": np.zeros((n, 6)),
              "slip_valid": np.zeros((n, 6), dtype=bool), "nonfoot_contact": np.zeros(n, dtype=bool),
              "upright": np.zeros(n), "qvel": np.zeros((n, 108))}
    for key in ("step_requested", "integration_complete", "state_captured", "state_copy_complete",
                "shadow_forward_complete", "observer_returned", "diagnostics_returned", "nonfoot_returned",
                "observation_complete", "source_verified"):
        native[key] = np.ones(n, dtype=bool)
    contact = {key: np.zeros((30000, *value.shape[1:]), dtype=value.dtype)
               for key, value in native.items() if key in ("normal_load_bw", "slip_mm_s", "slip_valid", "nonfoot_contact", "upright")}
    contact["time_s"] = time.copy()
    force = {"time_s": time, "available": np.ones(30000, dtype=bool),
             "finite": np.ones(30000, dtype=bool), "error": np.full(30000, ""),
             "thorax_rotation": np.zeros((30000, 3, 3)), "heading_xy": np.zeros((30000, 3)),
             "leg_force_bw": np.zeros((30000, 6, 3)), "segment_force": np.zeros((30000, 30, 3)),
             "leg_heading_xy_force_bw": np.zeros((30000, 6))}
    derived = {"native_load": native["normal_load_bw"].copy(), "native_slip_mm_s": native["slip_mm_s"].copy(),
               "native_slip_valid": native["slip_valid"].copy(), "native_rm_actual_qvel": np.zeros((n, 7)),
               "native_force_forward": native["force_forward_bw"].copy(),
               "contact_available": np.ones((n, 6), dtype=bool), "native_z_total": np.zeros((n, 6)),
               "native_z_root": np.zeros((n, 6)), "native_z_own7": np.zeros((n, 6, 7)),
               "native_z_residual": np.zeros((n, 6)), "native_rm_command_projection_z": np.zeros((n, 15)),
               "trace_control_tick": np.arange(2001), "trace_time_s": np.arange(2001) * .005,
               "rm_trace_joint_targets": np.zeros((2001, 7))}
    aligned = {"control_time_s": np.arange(601) * .005, "joint_target_rad_42": np.zeros((601, 42))}
    names = [[f"leg{leg}_joint{j}" for j in range(7)] for leg in range(6)]
    names[4][4] = "flylab/rm_coxa-rm_trochanterfemur-pitch"
    names[4][6] = "flylab/rm_tibia-rm_tarsus1-pitch"
    layout = {"leg_joint_names": names, "leg_dof_addresses": np.arange(42).reshape(6, 7).tolist(),
              "weight0": 3.0, "body_hash": "synthetic"}
    offsets = np.zeros(10001, dtype=np.int64)
    offsets[1:] = 24836
    observed = {"offsets": offsets, "native_index": np.zeros(24836, dtype=np.int64),
                "contact_id": np.arange(24836), "segment": np.zeros(24836, dtype=np.int64),
                "leg": np.zeros(24836, dtype=np.int64), "sign": np.ones(24836, dtype=np.int8),
                "frame": np.broadcast_to(np.eye(3), (24836, 3, 3)).copy(),
                "wrench": np.zeros((24836, 6))}
    metadata = {"inventory": {"weight0": 3.0, "body_model_hash": "synthetic"}}
    return native, derived, force, contact, aligned, layout, observed, metadata


def test_exact_join_rejects_shifted_force_and_trace_target():
    sources = synthetic_join()
    assert e190.validate_join(*sources, "synthetic")["valid"]
    sources[2]["leg_force_bw"][10000, 4, 0] = .000001
    with pytest.raises(ValueError, match="E185 segment-then-leg force order"):
        e190.validate_join(*sources, "synthetic")
    sources[2]["leg_force_bw"][10000, 4, 0] = 0
    sources[1]["rm_trace_joint_targets"][219, 4] = .000001
    with pytest.raises(ValueError, match="RM target"):
        e190.validate_join(*sources, "synthetic")


def test_fixed_event_edges_and_exclusion():
    native, derived, force, *_ = synthetic_join()
    raw, summary = e190.analyze(native, derived, force)
    assert raw["event_native_index"].shape == (3, 1000)
    assert raw["event_native_index"][0, 0] == 400  # tick 209, pre start of 219
    assert raw["event_native_index"][0, -1] == 1399  # tick 228, post end
    assert raw["bin_control_tick"][0].tolist() == list(range(209, 229))
    assert raw["bin_control_tick"][:, 10].tolist() == [219, 308, 391]
    assert summary["excluded_events"] == [131, 433, 480, 521, 564]
    assert raw["lag_command_z_pairs"].shape == (3, 5)
    assert not raw["lag_command_z_available"].any()  # constant synthetic signal


def test_mixed_support_bins_keep_same_metric_schema():
    native, derived, force, *_ = synthetic_join()
    native["normal_load_bw"][400:450, 4] = .03
    raw, _ = e190.analyze(native, derived, force)
    assert raw["bin_all_command_downward_fraction"].shape == (3, 20)
    assert raw["bin_all_valid"][0, 0]
    assert not raw["bin_all_valid"][0, 1]
    assert np.isnan(raw["bin_all_command_downward_fraction"][0, 1])
    assert raw["native_own7_sum_z"].shape == (10000,)
    assert raw["native_own7_components_z"].shape == (10000, 7)


def test_alternate_force_reduction_orders_keep_zero_normal_contacts():
    observed = synthetic_join()[6]
    observed["wrench"][0, 0] = 1e16
    observed["segment"][1] = 1
    observed["wrench"][1, 0] = 1
    observed["wrench"][2, 0] = -1e16
    e157, segment, e185 = e190.reconstruct_force_orders(observed, 3.0)
    assert e157[0, 0, 0] != e185[0, 0, 0]
    assert segment[0, 0, 0] == 0
    assert segment[0, 1, 0] == 1
    assert e185[0, 0, 0] == pytest.approx(1 / 3)
    assert observed["wrench"][3, 0] == 0  # retained rather than filtered


def mocked_run(monkeypatch, tmp_path):
    monkeypatch.setattr(e190, "ROOT", tmp_path)
    paths = {key: tmp_path / key for key in e190.INPUT_KEYS}
    paths["layout"].write_text("{}", encoding="utf-8")
    paths["force_metadata"].write_text("{}", encoding="utf-8")
    frozen = {"inputs_sha256": {f"input-{i}": "hash" for i in range(18)},
              "design_sha256": "hash", "amendment_sha256": "hash", "body_hash": "synthetic"}
    review = {"source_sha256": {name: "hash" for name in e190.OWNED}}
    monkeypatch.setattr(e190, "gate", lambda *_: (frozen, review, paths))
    monkeypatch.setattr(e190, "load_npz", lambda *_: {})
    monkeypatch.setattr(e190, "validate_join", lambda *_: {"valid": True})
    monkeypatch.setattr(e190, "analyze", lambda *_: ({"synthetic": np.arange(3)}, {}))
    monkeypatch.setattr(e190, "digest", lambda *_: "hash")
    return tmp_path / "out"


def test_postrun_hash_failure_preserves_raw_and_invalid_result(monkeypatch, tmp_path):
    output = mocked_run(monkeypatch, tmp_path)
    seen = []
    def changed(mapping):
        seen.append(len(mapping))
        raise ValueError("synthetic changed source")
    monkeypatch.setattr(e190, "sha_map", changed)
    result = e190.run(tmp_path / "freeze", tmp_path / "review", output)
    assert seen == [18]
    assert result["comparison_status"] == "valid"
    assert not result["publication_complete"]
    assert "changed source" in result["primary_error"]["message"]
    assert (output / "vertical-sequence.npz").exists()
    assert not json.loads((output / "result.json").read_text())["valid"]


def test_result_write_failure_keeps_primary_and_raw(monkeypatch, tmp_path):
    output = mocked_run(monkeypatch, tmp_path)
    monkeypatch.setattr(e190, "sha_map", lambda *_: None)
    original = e190.save_json
    calls = []
    def fail_once(path, value):
        calls.append(path.name)
        if len(calls) == 1:
            raise OSError("synthetic result write failure")
        return original(path, value)
    monkeypatch.setattr(e190, "save_json", fail_once)
    result = e190.run(tmp_path / "freeze", tmp_path / "review", output)
    assert calls == ["result.json", "result.json"]
    assert not result["publication_complete"]
    assert "synthetic result write failure" in result["primary_error"]["message"]
    assert (output / "vertical-sequence.npz").exists()
    assert not json.loads((output / "result.json").read_text())["publication_complete"]


def test_late_deadline_rewrites_success_receipt_as_failure(monkeypatch, tmp_path):
    output = mocked_run(monkeypatch, tmp_path)
    monkeypatch.setattr(e190, "sha_map", lambda *_: None)
    def late(phase):
        if phase == "result persistence":
            raise ValueError("synthetic late deadline")
    monkeypatch.setattr(e190, "deadline", late)
    result = e190.run(tmp_path / "freeze", tmp_path / "review", output)
    assert not result["publication_complete"]
    assert "late deadline" in result["primary_error"]["message"]
    assert not json.loads((output / "result.json").read_text())["publication_complete"]
