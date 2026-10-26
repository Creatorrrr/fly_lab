"""Generated-only E233 ownership repair and unchanged E232 science checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tools import analyze_c_banc_m3_effector_distribution_shift_recovery as e233


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
    joined = e233.seam_merge(prefix, suffix)
    assert all(len(value) == 1664 for value in joined.values())
    assert np.array_equal(joined["control_tick"], np.arange(1664))
    assert joined["h_slow"][600, 2, 1] == 9.25
    suffix["h_slow"][0, 2, 1] += .01
    with pytest.raises(ValueError, match="control600 seam"):
        e233.seam_merge(prefix, suffix)


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
    report = e233.ema_decompose(actual, conditional, parameters)
    assert np.max(np.abs(report["reconstruction_residual"])) < 1e-14
    assert all(report["command_component_" + key][0, 0] == 0.
               for key in e233.COMMAND_PARTS)
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
    distance, nearest = e233.projected_nearest(memory, centers,
                                                 np.array([2.]), chunk=2)
    assert nearest[:, 0].tolist() == [0, 2, 0]
    assert distance[:, 0].tolist() == [0., 0., .25]
    memory[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 6D"):
        e233.projected_nearest(memory, centers, np.array([2.]))
    with pytest.raises(ValueError, match="finite 6D"):
        e233.projected_nearest(np.zeros((1, 1, 6)), centers,
                               np.array([2.]), chunk=33)


def test_startup_fixed_rows_and_fail_closed_gate():
    commands = np.zeros((1664, 42), np.float64)
    target = np.zeros((2001, 42), np.float64)
    commands[0, 4] = 10.  # Held row0 does not enter startup score.
    commands[9, 7] = .03
    limits = {"startup_rms_max_rad": .018, "startup_abs_max_rad": .04}
    report = e233.startup_metrics(commands, target, limits)
    assert report["max_abs_control"] == 9
    assert report["max_abs_joint_column"] == 7
    assert report["max_abs_rad"] == pytest.approx(.03)
    assert report["passes_original_limits"]
    with pytest.raises(ValueError, match="Exact E233 freeze"):
        e233.gate("absent", "bad", "absent", "bad", e233.OUT)


def _snapshot_ownership_fixture():
    """391 global MN, 343 free MN, and an interleaved 48-fixed subset."""
    ids = np.arange(1000, 1391, dtype=np.int64)
    fixed = np.arange(0, 48 * 8, 8, dtype=np.int64)
    free = np.setdiff1d(np.arange(391, dtype=np.int64), fixed)
    membership = np.arange(391, dtype=np.int64) % 96
    beta = np.zeros((96, 64), np.float64)
    params = {"motor_ids": ids, "free_columns": free,
        "fixed_columns": fixed, "membership": membership,
        "pool": np.zeros((96, 391), np.float64),
        "moment": np.zeros((42, 96), np.float64),
        "rest": np.zeros(42, np.float64),
        "ltm": np.arange(6, dtype=np.int64),
        "scale": np.ones((391, 3), np.float64),
        "tau": np.full(391, .02, np.float64),
        "cap": np.ones(391, np.float64),
        "inverse_adaptation_tau": np.full(391, 50., np.float64)}
    prototypes = {"centers": np.zeros((391, 64, 9), np.float64),
                  "sigma2": np.ones(391, np.float64)}
    native = {"motor_ids": ids.copy(), "free_ids": ids[free].copy(),
        "fixed_ids": ids[fixed].copy(), "membership": membership[free].copy(),
        "scale": params["scale"][free].copy(),
        "centers": prototypes["centers"][free].copy(),
        "sigma2": prototypes["sigma2"][free].copy(),
        "beta": beta.copy(), "tau": params["tau"][free].copy(),
        "cap": params["cap"][free].copy(),
        "inverse_slow": params["inverse_adaptation_tau"][free].copy()}
    start = {"neural_current_trial_patch": {
        "parameters": copy.deepcopy(params),
        "prototypes": {key: value.copy() for key, value in prototypes.items()},
        "beta": beta.copy()},
        "network": {"neural_current_effector": native},
        "adapter": {"canonical": {
            "calibration": {key: params[key].copy() for key in (
                "motor_ids", "pool", "moment", "rest", "ltm")},
            "activation": np.zeros(96, np.float64)}}}
    actual = {"decoder_input32": np.zeros((1664, 391), np.float32),
        "motor_rate32": np.zeros((1664, 391), np.float32),
        "free_effector64": np.zeros((1664, 343), np.float64),
        "activation": np.zeros((1664, 96), np.float64)}
    return params, prototypes, {"beta": beta}, start, actual, free, fixed


def test_free_order_native_membership_exact_and_bad_layouts_rejected():
    params, prototypes, conditional, start, actual, free, fixed = (
        _snapshot_ownership_fixture())
    observed_free, observed_fixed = e233.check_ownership(
        params, prototypes, conditional, start, actual, conditional["beta"])
    assert np.array_equal(observed_free, free)
    assert np.array_equal(observed_fixed, fixed)
    baseline = start["network"]["neural_current_effector"]["membership"].copy()
    assert np.array_equal(baseline, params["membership"][free])
    native = start["network"]["neural_current_effector"]
    for wrong in (params["membership"].copy(), baseline[::-1].copy(),
                  baseline.astype(np.int32), np.r_[baseline[1:], 96]):
        native["membership"] = wrong
        with pytest.raises(AssertionError, match="E233 native.membership"):
            e233.check_ownership(params, prototypes, conditional, start,
                                 actual, conditional["beta"])
    native["membership"] = baseline
    native["centers"] = np.ones_like(native["centers"])
    with pytest.raises(AssertionError, match="E233 native.centers"):
        e233.check_ownership(params, prototypes, conditional, start,
                             actual, conditional["beta"])
