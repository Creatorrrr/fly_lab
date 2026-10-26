"""Generated-only E184 contracts; never reads original raw or runs a model."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_forward_break as e184


def control_rows():
    return [
        {
            "control_tick": i,
            "time_s": i * 0.005,
            "command_interval": None
            if i == 0
            else {
                "start_tick": i - 1,
                "end_tick": i,
                "start_s": (i - 1) * 0.005,
                "end_s": i * 0.005,
            },
        }
        for i in range(601)
    ]


def native(load_value=0.0):
    return {
        "time_s": np.arange(1, 30001) * 0.0001,
        "normal_load_bw": np.full((30000, 6), load_value),
        "slip_mm_s": np.full((30000, 6), np.nan),
        "slip_valid": np.zeros((30000, 6), bool),
    }


def test_control_boundaries_and_gaps():
    rows = control_rows()
    np.testing.assert_allclose(e184.check_control(rows), np.arange(601) * 0.005)
    rows[600]["time_s"] = 3.0 + 1e-9
    e184.check_control(rows)  # 3 s floating-point tolerance
    rows[600]["time_s"] = 3.0 + 2e-8
    with pytest.raises(ValueError, match="clock"):
        e184.check_control(rows)
    rows = control_rows()
    rows[5]["command_interval"]["start_tick"] = 0
    with pytest.raises(ValueError, match="interval"):
        e184.check_control(rows)


def test_native_right_closed_binning_and_missingness():
    data = native()
    # Exactly 50 samples in tick 1; boundary 0.005 is in tick 1, 0.0051 in tick 2.
    data["normal_load_bw"][49, 0] = 0.03
    data["normal_load_bw"][50, 0] = 0.04
    data["slip_mm_s"][49, 0] = 2.0
    data["slip_mm_s"][50, 0] = 3.0
    data["slip_valid"][49:51, 0] = True
    out = e184.native_bins(data)
    assert np.isnan(out["normal_load_bw"][0, 0])
    assert np.isnan(out["supported_slip_mm_s"][0, 0])
    assert np.isnan(out["supported_native_samples"][0, 0])
    assert out["support_fraction"][1, 0] == 1 / 50
    assert out["support_fraction"][2, 0] == 1 / 50
    assert out["supported_slip_mm_s"][1, 0] == 2
    assert out["supported_slip_mm_s"][2, 0] == 3
    assert np.isnan(out["supported_slip_mm_s"][3, 0])
    data["time_s"][50] += 2e-8
    with pytest.raises(ValueError, match="clock"):
        e184.native_bins(data)


def test_supported_invalid_slip_refused_and_unsupported_stays_missing():
    data = native()
    data["normal_load_bw"][50, 1] = 0.03
    with pytest.raises(ValueError, match="supported slip"):
        e184.native_bins(data)
    data["slip_valid"][50, 1] = True
    with pytest.raises(ValueError, match="supported slip"):
        e184.native_bins(data)
    data["slip_mm_s"][50, 1] = 5
    out = e184.native_bins(data)
    assert out["supported_slip_mm_s"][2, 1] == 5
    assert np.isnan(out["supported_slip_mm_s"][2, 0])


def test_native_slip_uses_samplewise_load_weighting():
    data = native()
    data["normal_load_bw"][0:2, 0] = [0.02, 0.08]
    data["slip_valid"][0:2, 0] = True
    data["slip_mm_s"][0:2, 0] = [1, 5]
    out = e184.native_bins(data)
    assert np.isclose(out["supported_slip_mm_s"][1, 0], 4.2)
    assert out["supported_native_samples"][1, 0] == 2


def test_crossing_equality_edge_and_disjoint_windows():
    speed = np.ones(601)
    for index in (105, 120, 130, 161, 580):
        speed[index] = 0.5  # exact equality is a crossing
    events = e184.select_events(speed)
    assert [e["index"] for e in events] == [105, 120, 130, 161, 580]
    assert [e["exclusion_reason"] for e in events] == [
        "edge_window",
        None,
        "overlap_selected_window",
        None,
        "edge_window",
    ]
    assert [e["index"] for e in events if e["selected"]] == [120, 161]
    assert e184.select_events(np.full(601, 0.5)) == []


def test_window_samples_and_no_event_output():
    signals = {"x": np.arange(601, dtype=float), "leg": np.ones((601, 6), dtype=float)}
    events = e184.select_events(np.ones(601))
    curves, mean = e184.align_curves(signals, events)
    assert curves["x"].shape == (0, 41)
    assert mean["leg"].shape == (41, 6)
    assert np.isnan(mean["leg"]).all()
    assert e184.event_windows(signals, events) == []
    events = [{"index": 200, "time_s": 1.0, "selected": True}]
    windows = e184.event_windows(signals, events)[0]["periods"]
    assert windows["pre"]["x"]["valid_samples"] == 10
    assert windows["pre"]["x"]["mean"] == 184.5
    assert windows["lead"]["x"]["mean"] == 194.5
    assert windows["post"]["x"]["mean"] == 204.5
    assert windows["recover"]["x"]["mean"] == 215.0
    assert windows["recover"]["x"]["valid_samples"] == 11


def test_band_slip_pools_uneven_native_loads():
    data = native()
    for tick in range(190, 200):
        row = (tick - 1) * 50
        data["normal_load_bw"][row, 0] = 0.18 if tick == 199 else 0.02
        data["slip_mm_s"][row, 0] = 5.0 if tick == 199 else 1.0
        data["slip_valid"][row, 0] = True
    signals = e184.native_bins(data)
    event = [{"index": 200, "time_s": 1.0, "selected": True}]
    lead = e184.event_windows(signals, event)[0]["periods"]["lead"][
        "supported_slip_mm_s"
    ][0]
    assert lead["mean"] == pytest.approx(3.0)
    assert lead["supported_native_samples"] == 10
    assert lead["native_load_sum_bw"] == pytest.approx(0.36)
    assert lead["valid_samples"] == 10
    assert (
        e184.event_windows(signals, event)[0]["periods"]["lead"]["supported_slip_mm_s"][
            1
        ]["mean"]
        is None
    )
    curves, means = e184.align_curves(signals, event)
    assert curves["supported_slip_mm_s"].shape == (1, 41, 6)
    assert means["supported_slip_mm_s"][19, 0] == 5.0


def test_mean_event_slip_pools_native_weights_across_events():
    data = native()
    for tick, load, slip in ((199, 0.02, 1.0), (299, 0.18, 5.0)):
        row = (tick - 1) * 50
        data["normal_load_bw"][row, 0] = load
        data["slip_mm_s"][row, 0] = slip
        data["slip_valid"][row, 0] = True
    signals = e184.native_bins(data)
    events = [{"index": i, "time_s": i * 0.005, "selected": True} for i in (200, 300)]
    _, means = e184.align_curves(signals, events)
    assert means["supported_slip_mm_s"][19, 0] == pytest.approx(4.6)
    plotted = e184._pooled_leg_slip(
        means["supported_slip_weighted_sum_bw_mm_s"],
        means["supported_slip_load_sum_bw"],
    )
    assert plotted[19] == pytest.approx(4.6)


def test_leg_mapping_requires_one_leg_per_joint_and_seven_rows(tmp_path):
    membership = np.arange(391) % 96
    groups = [
        {"leg": e184.LEGS[i // 16], "columns": np.flatnonzero(membership == i).tolist()}
        for i in range(96)
    ]
    moment = np.zeros((42, 96))
    for leg in range(6):
        moment[leg * 7 : (leg + 1) * 7, leg * 16] = 1
    moment[30] = 0  # named RM joint with no active calibration moment
    start = tmp_path / "start"
    start.mkdir()
    np.save(start / "array_00000.npy", moment)
    (start / "state.json").write_text(
        json.dumps(
            {
                "adapter": {
                    "groups": groups,
                    "calibration": {"moment": {"__npy__": "array_00000.npy"}},
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(tmp_path / "timing-parameters.npz", membership=membership)
    files = {"start/array_00000.npy": "synthetic"}
    frozen = np.repeat(np.arange(6), 7).tolist()
    muscle, mn, joint = e184.leg_maps(tmp_path, files, frozen)
    assert muscle.shape == (96,) and mn.shape == (391,)
    assert np.bincount(joint).tolist() == [7] * 6
    moment[0, 16] = 1
    np.save(start / "array_00000.npy", moment)
    with pytest.raises(ValueError, match="exactly one leg"):
        e184.leg_maps(tmp_path, files, frozen)
    moment[0, 16] = 0
    moment[31] = 0
    np.save(start / "array_00000.npy", moment)
    with pytest.raises(ValueError, match="zero-moment"):
        e184.leg_maps(tmp_path, files, frozen)
    moment[31, 64] = 1
    np.save(start / "array_00000.npy", moment)
    groups[0]["columns"].append(1)
    (start / "state.json").write_text(
        json.dumps(
            {
                "adapter": {
                    "groups": groups,
                    "calibration": {"moment": {"__npy__": "array_00000.npy"}},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate MN"):
        e184.leg_maps(tmp_path, files, frozen)


def test_safe_relative_path_refuses_traversal():
    for name in ("../raw", "/absolute", "C:/raw", "a\\b"):
        with pytest.raises(ValueError, match="Unsafe"):
            e184._safe_rel(name)


def test_production_output_path_is_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(e184, "OUTPUT_DIR", tmp_path / "fixed")
    with pytest.raises(ValueError, match="fixed planned directory"):
        e184.execute(
            tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "alternate"
        )
    assert not (tmp_path / "alternate").exists()


def test_generated_arm_end_to_end_zero_events_and_recruitment_gap(tmp_path):
    t = np.arange(601) * 0.005
    rows = control_rows()
    for i, row in enumerate(rows):
        row["body"] = {
            "position": [2 * t[i], 0, 0],
            "basis": np.eye(3).tolist(),
            "legs": {leg: [[2 * t[i], 0, 0]] * 8 for leg in e184.LEGS},
        }
        row["joint_targets_rad"] = [0.1] * 42
        row["joint_angles_rad"] = [0.09] * 42
    (tmp_path / "trace.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )
    moment = np.zeros((42, 96))
    for leg in range(6):
        moment[leg * 7 : (leg + 1) * 7, leg * 16] = 1
    moment[30] = 0
    start = tmp_path / "start"
    start.mkdir()
    np.save(start / "array_00000.npy", moment)
    (start / "state.json").write_text(
        json.dumps(
            {
                "adapter": {
                    "groups": [
                        {
                            "leg": e184.LEGS[i // 16],
                            "columns": np.flatnonzero(
                                np.arange(391) % 96 == i
                            ).tolist(),
                        }
                        for i in range(96)
                    ],
                    "calibration": {"moment": {"__npy__": "array_00000.npy"}},
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(tmp_path / "timing-parameters.npz", membership=np.arange(391) % 96)
    np.savez(tmp_path / "native-contact.npz", **native())
    state = {
        "control_tick": np.arange(601),
        "network_tick": np.arange(601) * 20,
        "physics_time_s": t,
        "complete": np.ones(601, bool),
        "motor_rate": np.zeros((601, 391)),
        "motor_rate_available": np.ones(601, bool),
        "activation": np.zeros((601, 96)),
        "activation_available": np.ones(601, bool),
    }
    np.savez(tmp_path / "native-recruitment-state.npz", **state)
    (tmp_path / "result.json").write_text('{"objective_mm_s":0.0}', encoding="utf-8")
    (tmp_path / "biological-gait-3s.json").write_text(
        json.dumps({"window": {"longest_forward_bout_s": 2.495}}), encoding="utf-8"
    )
    summary, arrays = e184.read_arm(
        tmp_path,
        {"start/array_00000.npy": "synthetic"},
        np.repeat(np.arange(6), 7).tolist(),
    )
    assert summary["selected_event_count"] == 0
    assert summary["events"] == [] and summary["event_periods"] == []
    assert arrays["curve_speed_mm_s"].shape == (0, 41)
    assert np.isnan(arrays["signal_supported_slip_mm_s"]).all()
    assert np.isnan(arrays["signal_supported_native_samples"][0]).all()
    assert summary["objective_mm_s_recomputed"] == 0.0
    assert summary["longest_forward_bout_s_recomputed"] == 2.495

    x = np.where(t <= 1.0, 2 * t, np.where(t <= 1.2, 2.0, 2.0 + 2 * (t - 1.2)))
    for i, row in enumerate(rows):
        row["body"]["position"] = [float(x[i]), 0, 0]
        row["body"]["legs"] = {leg: [[float(x[i]), 0, 0]] * 8 for leg in e184.LEGS}
    (tmp_path / "trace.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )
    speed = e184.position_forward_speed(
        np.column_stack((x, np.zeros(601), np.zeros(601))),
        np.tile([1.0, 0.0, 0.0], (601, 1)),
        0.005,
    )
    events = [e for e in e184.select_events(speed) if e["selected"]]
    assert len(events) == 1
    index = events[0]["index"]
    contact = native()
    for tick in range(index - 10, index):
        row = (tick - 1) * 50
        contact["normal_load_bw"][row, 0] = 0.18 if tick == index - 1 else 0.02
        contact["slip_mm_s"][row, 0] = 5.0 if tick == index - 1 else 1.0
        contact["slip_valid"][row, 0] = True
    np.savez(tmp_path / "native-contact.npz", **contact)
    window = (t >= 0.5) & (t < 3)
    objective = float(np.maximum(0, 0.5 - speed[window]).mean())
    longest = max(
        ((b - a - 1) * 0.005 for a, b in e184.runs(window & (speed > 0.5))), default=0.0
    )
    (tmp_path / "result.json").write_text(
        json.dumps({"objective_mm_s": objective}), encoding="utf-8"
    )
    (tmp_path / "biological-gait-3s.json").write_text(
        json.dumps({"window": {"longest_forward_bout_s": longest}}), encoding="utf-8"
    )
    summary, arrays = e184.read_arm(
        tmp_path,
        {"start/array_00000.npy": "synthetic"},
        np.repeat(np.arange(6), 7).tolist(),
    )
    assert summary["selected_event_count"] == 1
    assert summary["event_periods"][0]["periods"]["lead"]["supported_slip_mm_s"][0][
        "mean"
    ] == pytest.approx(3.0)
    assert arrays["curve_supported_slip_mm_s"].shape == (1, 41, 6)
    assert np.isnan(arrays["signal_supported_slip_mm_s"][0, 0])
    assert "noncausal" in summary["signal_semantics"]["speed_mm_s"]

    rows[200]["body"]["legs"]["lf"] = [[0, 0, 0]] * 7
    (tmp_path / "trace.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="eight ordered leg body origins"):
        e184.read_arm(
            tmp_path,
            {"start/array_00000.npy": "synthetic"},
            np.repeat(np.arange(6), 7).tolist(),
        )
    rows[200]["body"]["legs"]["lf"] = [[float(x[200]), 0, 0]] * 8
    (tmp_path / "trace.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )

    state["motor_rate_available"][100] = False
    np.savez(tmp_path / "native-recruitment-state.npz", **state)
    with pytest.raises(ValueError, match="recruitment clock or availability"):
        e184.read_arm(
            tmp_path,
            {"start/array_00000.npy": "synthetic"},
            np.repeat(np.arange(6), 7).tolist(),
        )
