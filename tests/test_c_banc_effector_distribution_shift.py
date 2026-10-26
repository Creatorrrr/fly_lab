"""Generated-only E232 seam, linear EMA, and projected-distance checks."""

from __future__ import annotations

import numpy as np
import pytest

from tools import analyze_c_banc_m3_effector_distribution_shift as e232


def _controls(first, count):
    ticks = np.arange(first, first + count, dtype=np.int64)
    row = {"control_tick": ticks, "network_tick": ticks * 20,
        "physics_time_s": ticks.astype(np.float64) * .005,
        "motor_rate32": np.zeros((count, 391), np.float32),
        "decoder_input32": np.zeros((count, 391), np.float32),
        "activation": np.zeros((count, 96), np.float64),
        "command": np.zeros((count, 42), np.float64),
        "tendon": np.zeros((count, 6), np.float64),
        "free_effector64": np.zeros((count, 343), np.float64),
        "h_fast": np.zeros((count, 343, 3), np.float64),
        "h_slow": np.zeros((count, 343, 3), np.float64)}
    return row


def test_all_field_seam_exact_and_no_duplicate_control600():
    prefix, suffix = _controls(0, 601), _controls(600, 1064)
    prefix["h_slow"][-1, 2, 1] = 9.25
    suffix["h_slow"][0, 2, 1] = 9.25
    joined = e232.seam_merge(prefix, suffix)
    assert all(len(value) == 1664 for value in joined.values())
    assert np.array_equal(joined["control_tick"], np.arange(1664))
    assert joined["h_slow"][600, 2, 1] == 9.25
    suffix["h_slow"][0, 2, 1] += .01
    with pytest.raises(ValueError, match="control600 seam"):
        e232.seam_merge(prefix, suffix)


def test_four_component_ema_reconstructs_command_and_held_row0():
    count = 1664
    free, fixed = np.arange(343), np.arange(343, 391)
    pool = np.zeros((96, 391), np.float64)
    pool[0, 0] = 1.
    pool[0, 343] = .5
    moment = np.zeros((42, 96), np.float64)
    moment[0, 0] = 1.
    alpha = -np.expm1(-.5)
    effector = np.zeros((count, 343), np.float64)
    effector[1:, 0] = 1.0000001
    decoder = np.zeros((count, 391), np.float32)
    decoder[1:, 0] = effector[1:, 0].astype(np.float32)
    decoder[1:, 343] = 2.
    conditional_activation = np.zeros((count, 96), np.float64)
    actual_activation = np.zeros((count, 96), np.float64)
    conditional_activation[0, 0] = .2
    actual_activation[0, 0] = .5
    actual_command = np.zeros((count, 42), np.float64)
    conditional_command = np.zeros((count, 42), np.float64)
    # Held control0 command is independent of the initial activation difference.
    actual_command[0, 0] = conditional_command[0, 0] = -.7
    for tick in range(1, count):
        conditional_activation[tick, 0] = ((1 - alpha) *
                                           conditional_activation[tick - 1, 0])
        actual_activation[tick, 0] = (actual_activation[tick - 1, 0] +
            alpha * (float(decoder[tick, 0]) + 1. -
                     actual_activation[tick - 1, 0]))
        actual_command[tick, 0] = .02 * actual_activation[tick, 0]
        conditional_command[tick, 0] = .02 * conditional_activation[tick, 0]
    actual = {"free_effector64": effector, "decoder_input32": decoder,
              "activation": actual_activation, "command": actual_command}
    conditional = {"rate": np.zeros((count, 391), np.float64),
                   "activation": conditional_activation,
                   "q_direct": conditional_command}
    parameters = {"free_columns": free, "fixed_columns": fixed,
                  "pool": pool, "moment": moment}
    report = e232.ema_decompose(actual, conditional, parameters)
    assert np.max(np.abs(report["reconstruction_residual"])) < 1e-14
    assert all(report["command_component_" + key][0, 0] == 0.
               for key in e232.COMMAND_PARTS)
    assert report["initial_activation_delta"][0] == pytest.approx(.3)
    expected_initial = .02 * .3 * (1 - alpha)
    assert report["command_component_initial_activation"][1, 0] == pytest.approx(
        expected_initial)
    assert report["command_component_fixed"][1, 0] == pytest.approx(.02*alpha)
    assert report["rate_delta_quantization"][1, 0] == pytest.approx(
        float(np.float32(1.0000001)) - 1.0000001)


def test_projected_nearest_first_tie_and_nonfinite_rejection():
    centers = np.full((1, 64, 6), 100., np.float64)
    centers[0, 0] = 0.
    centers[0, 1] = 0.  # Exact tie goes to the first center.
    centers[0, 2] = [2., 0., 0., 0., 0., 0.]
    memory = np.zeros((3, 1, 6), np.float64)
    memory[1, 0, 0] = 2.
    memory[2, 0, 0] = 1.
    distance, nearest = e232.projected_nearest(memory, centers,
                                                 np.array([2.]), chunk=2)
    assert nearest[:, 0].tolist() == [0, 2, 0]
    assert distance[:, 0].tolist() == [0., 0., .25]
    memory[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 6D"):
        e232.projected_nearest(memory, centers, np.array([2.]))
    with pytest.raises(ValueError, match="finite 6D"):
        e232.projected_nearest(np.zeros((1, 1, 6)), centers,
                               np.array([2.]), chunk=33)


def test_startup_fixed_rows_and_fail_closed_gate():
    commands = np.zeros((1664, 42), np.float64)
    target = np.zeros((2001, 42), np.float64)
    commands[0, 4] = 10.  # Held row0 does not enter startup score.
    commands[9, 7] = .03
    limits = {"startup_rms_max_rad": .018, "startup_abs_max_rad": .04}
    report = e232.startup_metrics(commands, target, limits)
    assert report["max_abs_control"] == 9
    assert report["max_abs_joint_column"] == 7
    assert report["max_abs_rad"] == pytest.approx(.03)
    assert report["passes_original_limits"]
    with pytest.raises(ValueError, match="Exact E232 freeze"):
        e232.gate("absent", "bad", "absent", "bad", e232.OUT)
