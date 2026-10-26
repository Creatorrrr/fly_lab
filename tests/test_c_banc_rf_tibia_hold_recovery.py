"""Small generated checks for the E231 E230 technical recovery."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from tools import observe_c_banc_m3_effector_contact_decomposition as e229
from tools import observe_c_banc_m3_effector_force_replay as replay
from tools import probe_c_banc_m3_rf_tibia_hold_recovery as e230


def test_tibia_hold_changes_only_joint26_at_fixed_controls():
    original = np.arange(42, dtype=np.float64) / 10
    anchor = original.copy()
    anchor[26] = -3.25
    for arm in e230.ARMS:
        for tick in (1608, 1619, 1620, 1635, 1636, 1647):
            applied = e230.tibia_command(original, anchor, arm, tick)
            changed = arm == "rf_tibia_hold" and 1620 <= tick <= 1635
            assert applied[26] == (-3.25 if changed else original[26])
            assert np.array_equal(np.delete(applied, 26),
                                  np.delete(original, 26))
            original_again = original.copy()
            original_again[26] = 2.6
            assert original_again[26] == 2.6
    with pytest.raises(ValueError, match="outside joint26"):
        tampered = original.copy()
        tampered[25] += .01
        e230.validate_tibia_command(original, tampered, anchor,
                                    "rf_tibia_hold", 1620)
    for bad in (1607, 1648):
        with pytest.raises(ValueError, match="fixed arm/control"):
            e230.tibia_command(original, anchor, "sham", bad)


def test_com_gradient_native_window_and_original_root_rotation():
    ticks = np.arange(2000, dtype=np.float64)
    time_s = 8.035 + (ticks + 1) * .0001
    # Exactly linear trajectory: gradient is 2 mm/s at every native sample.
    center = np.zeros((2000, 3), dtype=np.float64)
    center[:, 0] = 2 * ticks * .0001
    qvel = np.zeros((2000, 12), dtype=np.float64)
    qvel[:, 3:6] = np.array([.2, -.3, .4])
    raw, summary = e230.com_metrics(
        {"time_s": time_s, "com_position": center}, {"qvel": qvel},
        np.array([1., 0., 0.]), np.arange(6))
    assert raw["time_s"].shape == (2000,)
    assert np.allclose(raw["initial_heading_speed_mm_s"], 2.)
    assert np.array_equal(raw["root_angular_velocity_rad_s"], qvel[:, 3:6])
    assert summary["during_break_native_indices"] == [750, 1150]
    assert summary["during_break_backward_distance_mm"] == 0.
    assert summary["during_break_mean_speed_mm_s"] == pytest.approx(2.)
    assert summary["native_first_to_last_COM_displacement_mm"] == pytest.approx(.3998)
    assert summary["native_first_to_last_interval_s"] == pytest.approx(.1999)


def test_com_native_backward_window_and_clock_drift():
    tick = np.arange(2000, dtype=np.float64)
    time_s = 8.035 + (tick + 1) * .0001 - tick * 3e-12
    speed = np.ones(2000, np.float64)
    speed[750:1150] = -4.
    x = np.cumsum(speed) * .0001
    center = np.column_stack((x, np.zeros(2000), np.zeros(2000)))
    raw, summary = e230.com_metrics(
        {"time_s": time_s, "com_position": center},
        {"qvel": np.zeros((2000, 6))}, np.array([1., 0., 0.]),
        np.arange(6))
    assert summary["during_break_backward_distance_mm"] == pytest.approx(.15975)
    assert summary["during_break_min_speed_mm_s"] == pytest.approx(-4.)
    assert raw["initial_heading_speed_mm_s"][800] == pytest.approx(-4.)
    bad_time = time_s.copy()
    bad_time[100] += 3e-7
    with pytest.raises(ValueError, match="clock"):
        e230.com_metrics({"time_s": bad_time, "com_position": center},
                         {"qvel": np.zeros((2000, 6))},
                         np.array([1., 0., 0.]), np.arange(6))


def _partial_registry():
    registry = e229.new_registry(2, 3, 1)
    for index in range(2):
        e229.append(registry["native_state"], {
            "native_index": index, "time_s": 8.0351 + index * .0001,
            "control_tick": 1608,
            "qpos": np.array([index, 1.], np.float64),
            "qvel": np.array([0., 1., index], np.float64),
            "ctrl": np.array([.2 + index], np.float64),
            "state_captured": True, "shadow_forward_returned": True,
            "observation_returned": True})
        registry["contact_offsets"].append(0)
    return registry


def test_partial_native_difference_survives_without_control_return():
    original = _partial_registry()
    source = {"native_state": e229.arrays(original)[0]}
    current = _partial_registry()
    assert e230.first_state_difference(current, source) is None
    current["native_state"]["qpos"][1][0] = -5.
    assert e230.first_state_difference(current, source) == 1
    # This is a native sample receipt; no completed control row is invented.
    assert len(current["native_state"]["qpos"]) == 2


def test_exact_contact_jac_prefix_checks_and_tamper():
    registry = _partial_registry()
    state, contacts, attempts, decomp = e229.arrays(registry)
    source = {"native_state": state, "contacts": contacts,
              "attempts": attempts, "decomposition": decomp}
    e230.compare_reference_slice(registry, source, 0, 2)
    wrong = {**source, "native_state": dict(state)}
    wrong["native_state"]["qvel"] = state["qvel"].copy()
    wrong["native_state"]["qvel"][1, 2] += 1.
    with pytest.raises(ValueError, match="native_state.qvel"):
        e230.compare_reference_slice(registry, wrong, 0, 2)
    wrong = {**source, "contacts": dict(contacts)}
    wrong["contacts"]["offsets"] = contacts["offsets"].copy()
    wrong["contacts"]["offsets"][2] += 1
    with pytest.raises(ValueError, match="contact offsets"):
        e230.compare_reference_slice(registry, wrong, 0, 2)


def test_partial_storage_attempts_every_raw_artifact(tmp_path, monkeypatch):
    original_save = e229.save_npz
    failed = []

    def one_save_failure(path, payload):
        if path.name == "jacobian-attempts.npz":
            raise OSError("synthetic Jacobian storage failure")
        return original_save(path, payload)

    monkeypatch.setattr(e229, "save_npz", one_save_failure)
    sink = replay.ForceSink(["time_s", "available", "error",
                             *replay.FORCE_SHAPES])
    e230.save_arm_raw(tmp_path, _partial_registry(), [], sink,
                      on_error=lambda name, error: failed.append(
                          (name, str(error))))
    assert failed == [("jacobian-attempts.npz",
                       "synthetic Jacobian storage failure")]
    assert not (tmp_path / "jacobian-attempts.npz").exists()
    for name in ("native-state.npz", "contacts-all.npz",
                 "decomposition.npz", "native-contact.npz",
                 "force-native.npz"):
        assert (tmp_path / name).is_file()


def test_first_difference_and_subsequent_rows_reach_same_stream():
    """Exercise the exact runner helper at control1620 and beyond."""
    receipt = {"first_qpos_difference_control": None,
        "first_qvel_difference_control": None,
        "first_ctrl_difference_control": None,
        "first_physical_difference_control": None}
    stream = io.StringIO()
    source_1620 = {"control_tick": 1620, "qpos": [0.],
                   "qvel": [0.], "ctrl": [0.]}
    row_1620 = {"control_tick": 1620, "qpos": [1.],
                "qvel": [0.], "ctrl": [0.]}
    e230.record_physical_row(stream, row_1620, source_1620, receipt)
    assert receipt == {"first_qpos_difference_control": 1620,
        "first_qvel_difference_control": None,
        "first_ctrl_difference_control": None,
        "first_physical_difference_control": 1620}
    source_1621 = {"control_tick": 1621, "qpos": [0.],
                   "qvel": [0.], "ctrl": [0.]}
    row_1621 = {"control_tick": 1621, "qpos": [2.],
                "qvel": [3.], "ctrl": [4.]}
    e230.record_physical_row(stream, row_1621, source_1621, receipt)
    recorded = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert recorded == [row_1620, row_1621]
    assert receipt["first_qpos_difference_control"] == 1620
    assert receipt["first_qvel_difference_control"] == 1621
    assert receipt["first_ctrl_difference_control"] == 1621
    assert receipt["first_physical_difference_control"] == 1620
