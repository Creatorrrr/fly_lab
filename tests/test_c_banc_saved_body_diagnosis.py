"""Generated CPU checks for the E328 saved-only arithmetic and guards."""
import numpy as np
import pytest

from tools import analyze_c_banc_m3_saved_body_diagnosis as subject


def test_support_decomposition_and_half_open_bins():
    t = np.array([.5, .7499, .75, .9999, 1., 1.2499, 1.25, 1.4999, 1.5])
    load = np.zeros((len(t), 6))
    slip = np.full_like(load, np.nan)
    valid = np.zeros_like(load, dtype=bool)
    load[0, 0], slip[0, 0], valid[0, 0] = 1, 1, True
    load[2, 1], slip[2, 1], valid[2, 1] = 2, 2, True
    load[4, 0], slip[4, 0], valid[4, 0] = 3, 3, True
    load[6, 1], slip[6, 1], valid[6, 1] = 4, 4, True
    load[8, 2], slip[8, 2], valid[8, 2] = 100, 100, True
    result = subject.support_breakdown(t, load, slip, valid)
    assert result["weighted_slip_mm_s"] == 3
    assert result["window_time_samples"] == 8
    assert result["legs"]["lf"]["support_duty"] == 2 / 8
    assert result["legs"]["lh"]["weighted_slip_mm_s"] is None
    assert [v["support_samples"] for v in result["time_bins"].values()] == [1, 1, 1, 1]
    assert sum(v["contribution_mm_s"] for v in result["legs"].values()) == 3
    assert sum(v["contribution_mm_s"] for v in result["time_bins"].values()) == 3


def test_no_support_is_unavailable_not_zero():
    t = np.array([.5, .75, 1., 1.25])
    load = np.zeros((4, 6))
    result = subject.support_breakdown(t, load, np.full_like(load, np.nan),
                                       np.zeros_like(load, dtype=bool))
    assert result["weighted_slip_mm_s"] is None
    assert result["legs"]["lf"]["weighted_slip_mm_s"] is None
    assert result["time_bins"]["[0.5,0.75)"]["contribution_mm_s"] is None


@pytest.mark.parametrize("slip_value,valid_value", [(np.nan, True), (1., False)])
def test_invalid_selected_support_rejected(slip_value, valid_value):
    load = np.zeros((1, 6))
    slip = np.zeros((1, 6))
    valid = np.ones((1, 6), dtype=bool)
    load[0, 0], slip[0, 0], valid[0, 0] = .02, slip_value, valid_value
    with pytest.raises(ValueError, match="selected support"):
        subject.support_breakdown(np.array([.5]), load, slip, valid)


def test_exact_clocks_and_missing_native_sample():
    rows = [{"control_tick": i, "time_s": i * .005} for i in range(301)]
    raw = {"control_tick": np.arange(301), "network_tick": np.arange(301) * 20,
           "physics_time_s": np.arange(301) * .005}
    native = {"time_s": np.arange(1, 15001) * .0001}
    subject.validate_clocks(rows, raw, native)
    native["time_s"] = np.delete(native["time_s"], 5100)
    with pytest.raises(ValueError, match="native physical clock"):
        subject.validate_clocks(rows, raw, native)


def test_joint_shape_and_trace_command_correspondence():
    zero = np.zeros(42).tolist()
    rows = [{"joint_angles_rad": zero, "joint_targets_rad": zero,
             "joint_tracking_error_rad": zero} for _ in range(301)]
    raw = {"command": np.zeros((301, 42))}
    measured, command = subject.validate_joint_arrays(rows, raw)
    assert measured.shape == command.shape == (301, len(subject.LEGS) * 7)
    raw["command"][0, 7] = 1
    with pytest.raises(ValueError, match="order/value mismatch"):
        subject.validate_joint_arrays(rows, raw)


def test_moving_run_endpoint_duration_and_eligibility():
    moving = np.zeros(301, dtype=bool)
    moving[100:161] = True  # 61 samples = .300 s, inclusive endpoints
    moving[200:242] = True  # 42 samples = .205 s
    result = subject.moving_runs({"t": np.arange(301) * .005, "dt": .005,
                                  "moving": moving})
    assert [v["eligible"] for v in result] == [True, False]
    assert [round(v["duration_s"], 3) for v in result] == [.3, .205]
    assert result[0]["start_s"] == .5
    assert result[0]["end_s"] == .8


@pytest.mark.parametrize("actual,expected", [(np.nan, 1.), (np.inf, 1.),
                                            (1., np.nan), (1., -np.inf)])
def test_official_comparison_rejects_nonfinite_on_either_side(actual, expected):
    with pytest.raises(ValueError, match="official forward mismatch"):
        subject._close(actual, expected, "forward")


def test_global_tracking_rms_is_exact_pool_of_unequal_leg_errors():
    measured = np.zeros((2, 42))
    command = np.zeros_like(measured)
    measured[:] = np.repeat(np.arange(1., 7.), 7)
    result = subject.tracking_rms(measured, command)
    per_leg = result["source_inferred_per_leg_rms_rad"]
    assert list(per_leg.values()) == [1., 2., 3., 4., 5., 6.]
    assert result["global_rms_rad"] == pytest.approx(
        np.sqrt(sum(value ** 2 for value in per_leg.values()) / 6))
