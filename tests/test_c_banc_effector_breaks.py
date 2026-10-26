"""Small generated E226 math, interval, and fail-closed source tests."""

from __future__ import annotations

import numpy as np
import pytest

from tools import analyze_c_banc_m3_effector_breaks as e226

ANALYSIS = {"control_dt_s": .005, "window_start_s": .5,
    "window_end_s": 1.5, "main_entry_min_s": .8,
    "preceding_bout_min_s": .3,
    "pre_window_s": [-.08, -.02], "post_window_s": [.02, .08],
    "aligned_half_window_s": .1}


def synthetic_native(n=3, *, drift=0.):
    control_time = np.arange(n) * .005
    sample = np.arange(1, (n - 1) * 50 + 1)
    native_time = sample * .0001 + sample * drift
    loads = np.zeros(((n - 1) * 50, 6))
    slip = np.full_like(loads, np.nan)
    valid = np.zeros_like(loads, bool)
    return control_time, {"time_s": native_time,
        "normal_load_bw": loads, "slip_mm_s": slip,
        "slip_valid": valid,
        "nonfoot_contact": np.zeros(len(sample), bool),
        "upright": np.ones(len(sample))}


def test_native_interval_order_tolerance_and_nullable_unsupported_slip():
    control, native = synthetic_native(drift=1e-12)
    native["normal_load_bw"][:50, 0] = .03
    native["slip_mm_s"][:50, 0] = 2.
    native["slip_valid"][:50, 0] = True
    result = e226.aggregate_native_intervals(control, native)
    assert result["available"].tolist() == [False, True, True]
    assert np.isnan(result["mean_normal_load_bw"][0]).all()
    assert result["support_fraction"][1, 0] == 1.
    assert result["weighted_slip_mm_s"][1, 0] == pytest.approx(2.)
    assert not result["weighted_slip_available"][1, 1]
    assert np.isnan(result["weighted_slip_mm_s"][1, 1])
    assert not result["global_weighted_slip_available"][2]
    assert np.isnan(result["global_weighted_slip_mm_s"][2])
    wrong = {key: value.copy() for key, value in native.items()}
    wrong["time_s"][49] += .0001
    with pytest.raises(ValueError, match="boundary"):
        e226.aggregate_native_intervals(control, wrong)
    missing = {key: value[:-1] for key, value in native.items()}
    with pytest.raises(ValueError, match="Exactly 50"):
        e226.aggregate_native_intervals(control, missing)


def test_bout_gaps_main_set_and_incomplete_right_window():
    t = np.arange(290) * .005
    moving = np.zeros(len(t), bool)
    moving[100:180] = True
    moving[184:275] = True
    moving[280:299] = True
    events = e226.detect_breaks(t, moving, ANALYSIS)
    assert len(events) == 2
    assert events[0]["duration_samples"] == 4
    assert events[0]["duration_s"] == pytest.approx(.02)
    assert events[0]["main_comparison"] is True
    assert events[1]["main_comparison"] is True
    windows = e226.aligned_windows(events[1], len(t), ANALYSIS)
    assert windows["pre"]["available"] is True
    assert windows["post"]["available"] is False
    assert windows["post"]["indices"] is None
    # A right-censored below-threshold tail is not a fully observed gap.
    moving[280:] = False
    assert len(e226.detect_breaks(t, moving, ANALYSIS)) == 1


def test_rotation_projection_is_distinct_from_world_translation():
    n = 6
    t = np.arange(n) * .005
    rows = []
    for i, now in enumerate(t):
        theta = i * np.pi / 10
        c, s = np.cos(theta), np.sin(theta)
        basis = [[c, 0., -s], [0., 1., 0.], [s, 0., c]]
        legs = {leg: [[float(now), 0., 0.]] * 8 for leg in e226.LEGS}
        rows.append({"time_s": float(now), "body": {
            "position": [float(now), 0., 0.], "basis": basis, "legs": legs},
            "joint_angles_rad": [0.] * 42, "joint_targets_rad": [0.] * 42,
            "joint_tracking_error_rad": [0.] * 42,
            "qvel": [0.] * 108, "ctrl": [0.] * 82})
    model = {"t": t, "speed": np.cos(np.arange(n) * np.pi / 10),
             "yaw_rate": np.ones(n)}
    motion = e226.derive_kinematics(rows, model)
    assert np.allclose(motion["world_horizontal_speed_mm_s"], 1.)
    assert np.allclose(motion["initial_forward_velocity_mm_s"], 1.)
    assert motion["instant_body_forward_velocity_mm_s"][-1] < .01
    assert np.allclose(motion["world_velocity_xyz_mm_s"][:, 0],
                       np.ones(n))
    assert motion["forward_basis_xyz"].shape == (n, 3)


def test_canonical_contribution_preserves_held_row0():
    rng = np.random.default_rng(226)
    active = rng.uniform(0, 5, (4, 96))
    moment = rng.normal(size=(42, 96)) * .01
    rest = rng.normal(size=42)
    command = rest[None, :] + .02 * active @ moment.T
    command[0] = 17.  # held neutral, deliberately not this row's response
    result = e226.command_contributions(active, command, moment, rest)
    assert result["canonical_valid"].tolist() == [False, True, True, True]
    assert np.array_equal(result["row0_held_command_rad"], command[0])
    assert np.allclose(result["contribution_rad"][1].sum(axis=1),
                       command[1] - rest, atol=2e-12, rtol=0)
    command[2, 5] += .01
    with pytest.raises(ValueError, match="reconstruction"):
        e226.command_contributions(active, command, moment, rest)


def test_event_alignment_retains_partial_valid_mask():
    event = {"event_index": 0, "entry_index": 34,
             "exit_index": 39, "main_comparison": False}
    n = 45
    motion = {"time_s": np.arange(n) * .005,
        "original_smoothed_forward_speed_mm_s": np.zeros(n),
        "world_horizontal_speed_mm_s": np.ones(n),
        "initial_forward_velocity_mm_s": np.ones(n),
        "yaw_rate_rad_s": np.zeros(n),
        "ego_foot_fore_velocity_mm_s": np.zeros((n, 6)),
        "foot_world_horizontal_speed_mm_s": np.zeros((n, 6)),
        "joint_tracking_error_rad": np.zeros((n, 42)),
        "joint_angles_rad": np.zeros((n, 42)),
        "joint_targets_rad": np.zeros((n, 42)),
        "joint_velocity_rad_s": np.zeros((n, 42)),
        "target_velocity_rad_s": np.zeros((n, 42))}
    slip = np.full((n, 6), np.nan)
    slip[18:30, 0] = 2.
    native = {"mean_normal_load_bw": np.zeros((n, 6)),
        "support_fraction": np.zeros((n, 6)),
        "support_leg_count_mean": np.zeros(n),
        "weighted_slip_mm_s": slip}
    groups = [{"leg": "lf", "muscle": str(i)} for i in range(96)]
    events, aligned = e226.analyze_events([event], motion, native,
        np.zeros((n, 96)), np.zeros((n, 42, 96)), groups, ANALYSIS)
    assert not events[0]["windows"]["post"]["available"]
    assert events[0]["means"]["post"] is None
    observed = events[0]["means"]["pre"]["weighted_slip_mm_s"]
    assert observed["mean"] == [2., None, None, None, None, None]
    assert observed["valid_sample_count"] == [12, 0, 0, 0, 0, 0]
    assert observed["window_sample_count"] == 12
    assert aligned["control_index"].shape == (1, 41)
    assert (aligned["control_index"][0, ~aligned["available"][0]] == -1).all()
    assert aligned["available"].sum() < 41


def test_equal_event_aggregate_keeps_per_leg_counts_and_missing_post():
    def phase(value):
        return {"weighted_slip_mm_s": {"mean": [value, None, None, None,
                                                   None, None]}}
    events = [
        {"main_comparison": True, "means": {"pre": phase(2.),
            "during": phase(3.), "post": phase(5.)}},
        {"main_comparison": True, "means": {"pre": phase(4.),
            "during": phase(7.), "post": None}},
        {"main_comparison": False, "means": {"pre": phase(100.),
            "during": phase(100.), "post": phase(100.)}},
    ]
    aggregate = e226.aggregate_main_event_means(events)
    assert aggregate["main_event_count"] == 2
    pre = aggregate["phases"]["pre"]["weighted_slip_mm_s"]
    post = aggregate["phases"]["post"]["weighted_slip_mm_s"]
    assert pre["mean_of_event_means"] == [3., None, None, None, None, None]
    assert pre["available_event_count"] == [2, 0, 0, 0, 0, 0]
    assert post["mean_of_event_means"] == [5., None, None, None, None, None]
    assert post["available_event_count"] == [1, 0, 0, 0, 0, 0]
    assert post["total_main_events"] == 2


def test_gate_rejects_unapproved_source_before_output(tmp_path, monkeypatch):
    freeze_path = tmp_path / "freeze.json"
    review_path = tmp_path / "review.json"
    freeze_path.write_text("{}")
    review_path.write_text("{}")
    monkeypatch.setattr(e226, "sha256", lambda path: e226.FREEZE_SHA
                        if path == freeze_path else "review")
    with pytest.raises(ValueError, match="contract"):
        e226.gate(freeze_path, e226.FREEZE_SHA, review_path, "review",
                  tmp_path / "output")
    assert not (tmp_path / "output").exists()
