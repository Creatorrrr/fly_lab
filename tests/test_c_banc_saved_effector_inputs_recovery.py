"""Synthetic checks for the frozen E237 CPU replay; no saved-data execution."""

from __future__ import annotations

import numpy as np
import pytest

from tools import replay_c_banc_m3_saved_effector_inputs_recovery as m


def fixture(stages: int = 8):
    n = 3
    rng = np.random.default_rng(236)
    params = {"scale": np.array([[1.2, 1.4, 2.1], [2., 1.1, 1.7],
                                  [1.6, 1.9, 1.3]], np.float64),
              "membership": np.array([0, 1, 0], np.int64),
              "tau": np.array([.018, .021, .017], np.float64),
              "cap": np.array([120., 130., 115.], np.float64),
              "inverse_adaptation_tau": np.array([1.3, 2.1, 1.7], np.float64)}
    proto = {"centers": rng.normal(.12, .3, (n, 64, 9)).astype(np.float64),
             "sigma2": np.array([.3, .5, .4], np.float64)}
    beta = rng.uniform(0, 1, (2, 64)).astype(np.float64)
    initial = {"rate": np.array([.3, .2, .4], np.float64),
               "fast": rng.normal(0, .1, (n, 3)).astype(np.float64),
               "slow": rng.normal(0, .1, (n, 3)).astype(np.float64)}
    inputs = {"positive_recurrent": rng.uniform(0, 3, (stages, n)).astype(np.float32),
              "negative_magnitude": rng.uniform(0, 2, (stages, n)).astype(np.float32),
              "drive": rng.normal(0, 1, (stages, n)).astype(np.float32)}
    return params, proto, beta, initial, inputs


def scalar_reference(params, proto, beta, initial, inputs):
    """Literal neuron-by-neuron RK stages with scalar RBF normalization."""
    n = len(initial["rate"])
    ticks = len(inputs["drive"]) // 4
    fast = initial["fast"].copy()
    slow = initial["slow"].copy()
    rate = initial["rate"].copy()
    r_stages, f_stages, s_stages = [], [], []
    r_bounds, f_bounds, s_bounds = [rate.copy()], [fast.copy()], [slow.copy()]
    for tick in range(ticks):
        kr = [[] for _ in range(n)]
        kf = [[] for _ in range(n)]
        ks = [[] for _ in range(n)]
        for stage in range(4):
            factor = (0., m.DT / 2, m.DT / 2, m.DT)[stage]
            rs = np.empty(n)
            fs = np.empty((n, 3))
            ss = np.empty((n, 3))
            for i in range(n):
                previous = max(0, stage - 1)
                rs[i] = rate[i] + (factor * kr[i][previous]
                                   if stage else 0.)
                fs[i] = fast[i] + (factor * kf[i][previous]
                                   if stage else 0.)
                ss[i] = slow[i] + (factor * ks[i][previous]
                                   if stage else 0.)
                index = tick * 4 + stage
                e = float(inputs["positive_recurrent"][index, i])
                inhibitory = float(inputs["negative_magnitude"][index, i])
                drive = float(inputs["drive"][index, i])
                x = np.array((np.log1p(e) / params["scale"][i, 0],
                              np.log1p(inhibitory) / params["scale"][i, 1],
                              np.arcsinh(drive) / params["scale"][i, 2]))
                d = np.r_[x, fs[i], ss[i]]
                exponent = np.array([-sum((d[j] - proto["centers"][i, c, j]) ** 2
                                          for j in range(9)) /
                                     (2 * proto["sigma2"][i]) for c in range(64)])
                p = np.exp(exponent - max(exponent))
                p /= sum(p)
                gain = -np.expm1(-8 * (x[0] + abs(x[2])
                                         + max(fs[i, 0], 0) + abs(fs[i, 2])))
                target = (params["cap"][i] * gain *
                          sum(p[c] * beta[params["membership"][i], c]
                              for c in range(64)))
                kr[i].append((target - rs[i]) / params["tau"][i])
                kf[i].append((x - fs[i]) / m.TAU_FAST)
                ks[i].append((x - ss[i]) * params["inverse_adaptation_tau"][i])
            r_stages.append(rs)
            f_stages.append(fs)
            s_stages.append(ss)
        for i in range(n):
            rate[i] += m.DT / 6 * (kr[i][0] + 2 * kr[i][1] +
                                   2 * kr[i][2] + kr[i][3])
            fast[i] += m.DT / 6 * (kf[i][0] + 2 * kf[i][1] +
                                   2 * kf[i][2] + kf[i][3])
            slow[i] += m.DT / 6 * (ks[i][0] + 2 * ks[i][1] +
                                   2 * ks[i][2] + ks[i][3])
        r_bounds.append(rate.copy())
        f_bounds.append(fast.copy())
        s_bounds.append(slow.copy())
    return {"rate_stage": np.array(r_stages), "fast_stage": np.array(f_stages),
            "slow_stage": np.array(s_stages), "rate64": np.array(r_bounds),
            "fast64": np.array(f_bounds), "slow64": np.array(s_bounds)}


def test_two_tick_scalar_reference_and_signed_error():
    params, proto, beta, initial, inputs = fixture()
    actual = m.replay(inputs, params, proto, beta, initial, sample_every=1,
                      chunk=2)
    expected = scalar_reference(params, proto, beta, initial, inputs)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=0, atol=3e-13)
    assert actual["ticks_entered"] == actual["ticks_completed"] == 2
    assert actual["memory_ticks_completed"] == 2
    assert actual["stage_targets_completed"] == 8
    delta = actual["rate_stage"] + .1 - expected["rate_stage"]
    assert m.error_summary(delta, np.zeros_like(delta))["max_abs"] == pytest.approx(.1)


def test_partial_tick_and_nonfinite_rejection():
    params, proto, beta, initial, inputs = fixture(12)
    state = m.new_workspace(12, 3, initial, 1)
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:  # memory, first reaction chunk, next reaction chunk
            raise TimeoutError("synthetic cap")

    with pytest.raises(TimeoutError):
        m.replay(inputs, params, proto, beta, initial, sample_every=1,
                 chunk=2, check=interrupt, state=state)
    assert state["memory_ticks_completed"] == 3
    assert state["stage_targets_completed"] == 2
    assert state["ticks_completed"] == 0
    params["membership"] = params["membership"].astype(np.float64)
    with pytest.raises(ValueError, match="Finite original"):
        m.replay(inputs, params, proto, beta, initial, sample_every=1)


def test_canonical_held_row_and_original_startup_reference():
    rate = np.zeros((100, 391), np.float32)
    rate[1:, 0] = 4
    pool = np.zeros((96, 391), np.float64)
    pool[0, 0] = 1
    moment = np.zeros((42, 96), np.float64)
    moment[0, 0] = 2
    rest = np.zeros(42, np.float64)
    activation0 = np.zeros(96, np.float64)
    held = np.ones(42, np.float64) * .3
    ltm = np.array([15, 30, 46, 63, 78, 94], np.int64)
    decoded = m.readout(rate, pool, moment, rest, ltm, activation0, held)
    assert decoded["command"][0, 0] == .3
    assert decoded["command"][1, 0] == pytest.approx(.02 * 2 *
                                                       (-np.expm1(-.5)) * 4)
    assert decoded["activation"][0, 0] == 0
    target_q = decoded["command"][1:].copy()
    target_t = decoded["tendon"][1:].copy()
    assert m.startup_metrics(decoded, target_q, target_t)["command_rms_rad"] == 0
    target_q[0, 0] += .1
    assert m.startup_metrics(decoded, target_q, target_t)["command_max_abs_rad"] == .1


def test_shared_initial_state_checks_full_memory_and_conditional_rate():
    initial = {"rate": np.array([.25, .5], np.float64),
               "fast": np.zeros((2, 3), np.float64),
               "slow": np.zeros((2, 3), np.float64)}
    params = {"initial_rate": np.array([.5, .25, .75], np.float64)}
    basis = {"h_fast": np.zeros((1, 3, 3), np.float64),
             "h_slow": np.zeros((1, 3, 3), np.float64)}
    conditional = {"rate": np.array([[.5, .25, .75]], np.float64)}
    free = np.array([1, 0], np.int64)
    m.validate_initial_state(initial, params, basis, conditional, free)
    initial["fast"][1, 2] = 1e-12
    with pytest.raises(ValueError, match="initial state"):
        m.validate_initial_state(initial, params, basis, conditional, free)
    initial["fast"][1, 2] = 0
    conditional["rate"][0, 0] = .26
    with pytest.raises(ValueError, match="initial state"):
        m.validate_initial_state(initial, params, basis, conditional, free)


def test_replay_failure_survives_independent_receipt_failure():
    state = {"ticks_entered": 19, "ticks_completed": 18,
             "memory_ticks_completed": 20, "stage_targets_completed": 72}
    entry = {}
    storage = []
    calls = []

    def partial():
        calls.append("partial")

    def ledger():
        calls.append("ledger")
        raise OSError("synthetic ledger storage failure")

    original = ArithmeticError("synthetic replay failure")
    returned = m.finalize_arm_receipts(entry, state, partial, ledger,
                                       storage, original)
    assert returned is original
    assert calls == ["partial", "ledger"]
    assert entry["ticks_completed"] == 18
    assert entry["stage_targets_completed"] == 72
    assert storage[0]["artifact"] == "advance-ledger"
    assert "synthetic ledger storage failure" in original.__notes__[0]
    no_replay_error = m.finalize_arm_receipts({}, state, partial, ledger, [], None)
    assert isinstance(no_replay_error, OSError)


def test_layout_accepts_dynamic_excluded_neurons_without_changing_ownership():
    def zeros(shape, dtype=np.float64):
        return np.broadcast_to(np.array(0, dtype=dtype), shape)
    params = {key: zeros(shape) for key, shape in {
        "scale": (391, 3), "initial_rate": (391,), "tau": (391,), "cap": (391,),
        "inverse_adaptation_tau": (391,), "pool": (96, 391), "moment": (42, 96),
        "rest": (42,)}.items()}
    params.update(motor_ids=np.arange(391, dtype=np.int64),
                  fixed_columns=np.arange(48, dtype=np.int64),
                  free_columns=np.arange(48, 391, dtype=np.int64))
    basis = {key: zeros((2001, 391, 3)) for key in ["h_fast", "h_slow"]}
    conditional = {key: zeros(shape) for key, shape in {
        "rate": (2001, 391), "activation": (2001, 96),
        "q_direct": (2001, 42), "tendon": (2001, 6)}.items()}
    startup = {"loaded_startup_target_command": zeros((99, 42)),
               "loaded_startup_target_tendon": zeros((99, 6))}
    actual = {key: zeros(shape) for key, shape in {
        "activation": (100, 96), "command": (100, 42), "tendon": (100, 6),
        "free_effector64": (100, 343), "h_fast": (100, 343, 3),
        "h_slow": (100, 343, 3)}.items()}
    actual["motor_rate32"] = np.zeros((100, 391), np.float32)
    actual["motor_rate32"][6, 0] = .005
    actual["decoder_input32"] = actual["motor_rate32"].copy()
    meta = {"motor_ids": params["motor_ids"].tolist(),
            "free_motor_ids": params["motor_ids"][params["free_columns"]].tolist()}
    headers = {key: zeros((160000, 391), np.float32) for key in m.CHANNELS}
    m.validate_saved_layout(params, basis, conditional, startup, actual, meta, headers)
    report = m.fixed_rate_comparison(actual["motor_rate32"], conditional["rate"][:100],
                                    params["fixed_columns"], params["pool"], params["moment"])
    assert report["first_difference_control"] == 6
    assert report["different_control_count"] == 1
    assert report["max_abs"] == pytest.approx(.005)
    assert report["joint_projection_max_abs"] == 0
    meta["free_motor_ids"][0] = -1
    with pytest.raises(ValueError, match="ownership"):
        m.validate_saved_layout(params, basis, conditional, startup, actual, meta, headers)


def test_matched_fixed_readout_changes_only_excluded_columns():
    training = np.array([[.25, .5, .75, 1.], [2., 3., 4., 5.]])
    actual = np.array([[8., 7., 6., 5.], [11., 12., 13., 14.]], np.float32)
    before = training.copy()
    mixed = m.training_with_actual_fixed(training, actual, np.array([1, 3]))
    assert mixed.dtype == np.float32
    np.testing.assert_array_equal(mixed[:, [0, 2]], training[:, [0, 2]].astype(np.float32))
    np.testing.assert_array_equal(mixed[:, [1, 3]], actual[:, [1, 3]])
    np.testing.assert_array_equal(training, before)
