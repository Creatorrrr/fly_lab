"""Generated-only E170 Stage A numerical contract checks."""

import numpy as np
import pytest

from tools.banc_m3_history_recruitment import (
    DT,
    TAU,
    evaluate_history,
    history_state,
    parameter_gates,
)


def synthetic():
    membership = np.arange(391, dtype=np.int64) % 96
    fixed = np.arange(19, dtype=np.int64)
    counts = np.bincount(membership, minlength=96)
    alpha = np.zeros((96, 391, 3, 2), dtype=np.float64)
    beta = np.zeros_like(alpha)
    for mn, muscle in enumerate(membership):
        alpha[muscle, mn] = 0.8 / (counts[muscle] * 3)
        if muscle not in fixed:
            beta[muscle, mn] = 0.2 / (counts[muscle] * 3)
    return membership, fixed, alpha, beta


def test_history_recurrence_limits_and_independent_mn():
    rate = np.zeros((5, 391), dtype=np.float64)
    kappa = np.ones(391, dtype=np.float64)
    rate[1:, 0] = 1
    rate[1:, 1] = 3
    kappa[2] = 0  # explicit 0/0
    before_rate, before_kappa = rate.copy(), kappa.copy()
    r, f = history_state(rate, kappa)
    assert np.array_equal(r[0], np.ones(391))
    assert np.array_equal(f[0], np.zeros(391))
    assert np.array_equal(r[:, 2:], np.ones((5, 389)))
    for mn, s in ((0, 0.5), (1, 0.75)):
        target = 1 / (1 + s)
        decay = np.exp(-(1 + s) * DT / TAU)
        for tick in range(1, 5):
            assert r[tick, mn] == pytest.approx(
                target + (r[tick - 1, mn] - target) * decay
            )
    assert r[-1, 1] < r[-1, 0] < 1
    assert np.array_equal(f, 2 * (1 - r))
    assert np.all((0.5 <= r) & (r <= 1))
    assert np.all((0 <= f) & (f <= 1))
    assert np.array_equal(rate, before_rate)
    assert np.array_equal(kappa, before_kappa)


def test_history_overflow_ratio_and_normal_path_exact():
    rate = np.zeros((2, 391), dtype=np.float64)
    kappa = np.ones(391, dtype=np.float64)
    rate[1, :3] = [1e308, 1.6e308, 9e307]
    kappa[:3] = [1e308, 9e307, 1.6e308]
    rate[1, 3:6] = [1.0, 3.0, 0.0]
    kappa[3:6] = [1.0, 1.0, 0.0]
    with np.errstate(over="raise", invalid="raise"):
        state, history = history_state(rate, kappa)
    expected_s = [0.5, 1 / (1 + 9 / 16), (9 / 16) / (1 + 9 / 16)]
    for mn, s in enumerate(expected_s):
        target = 1 / (1 + s)
        expected = target + (1 - target) * np.exp(-(1 + s) * DT / TAU)
        assert state[1, mn] == expected
    assert state[1, 0] == pytest.approx(0.966132440449357)
    assert np.array_equal(history, 2 * (1 - state))

    # The ordinary finite-sum path retains the FIRST/E168 division and update bits.
    normal = np.divide(
        rate[:, 3:],
        rate[:, 3:] + kappa[None, 3:],
        out=np.zeros_like(rate[:, 3:]),
        where=(rate[:, 3:] + kappa[None, 3:]) != 0,
    )
    target = 1 / (1 + normal[1])
    expected_normal = target + (1 - target) * np.exp(-(1 + normal[1]) * DT / TAU)
    assert np.array_equal(state[1, 3:], expected_normal)


def test_beta_zero_exact_tick_order_and_fixed_ownership():
    membership, fixed, alpha, beta = synthetic()
    rng = np.random.default_rng(170)
    bank = rng.uniform(0, 2, (3, 391, 3)).astype(np.float64)
    original = rng.uniform(0, 1, (3, 96)).astype(np.float64)
    kappa = rng.uniform(0, 1, 391).astype(np.float64)
    moment = rng.normal(size=(42, 96)).astype(np.float64)
    rest = rng.normal(size=42).astype(np.float64)
    history = np.tile(np.array([0.0, 0.5, 1.0])[:, None], (1, 391))
    inputs = [
        x.copy()
        for x in (
            bank,
            original,
            alpha,
            beta,
            kappa,
            membership,
            fixed,
            moment,
            rest,
            history,
        )
    ]
    activation, command = evaluate_history(
        bank,
        original,
        alpha,
        np.zeros_like(beta),
        kappa,
        membership,
        fixed,
        moment,
        rest,
        history,
    )
    expected_activation = np.empty_like(activation)
    expected_command = np.empty_like(command)
    for tick in range(3):
        expected_activation[tick, fixed] = original[tick, fixed]
        hinge = np.maximum(bank[tick] - kappa[:, None], 0.0)
        for muscle in range(19, 96):
            owned = np.flatnonzero(membership == muscle)
            low = alpha[muscle, owned, :, 0]
            high = alpha[muscle, owned, :, 1]
            expected_activation[tick, muscle] = np.sum(
                low * bank[tick, owned]
            ) + np.sum((high - low) * hinge[owned])
        expected_command[tick] = rest + 0.02 * (moment @ expected_activation[tick])
    assert np.array_equal(activation, expected_activation)
    assert np.array_equal(command, expected_command)
    assert np.array_equal(activation[:, fixed], original[:, fixed])
    for current, saved in zip(
        (bank, original, alpha, beta, kappa, membership, fixed, moment, rest, history),
        inputs,
        strict=True,
    ):
        assert np.array_equal(current, saved)

    # One owner's history changes only its own variable-muscle activation.
    history.fill(0)
    history[:, 20] = 1
    with_beta, _ = evaluate_history(
        bank, original, alpha, beta, kappa, membership, fixed, moment, rest, history
    )
    changed = np.flatnonzero(np.any(with_beta != activation, axis=0))
    assert np.array_equal(changed, np.array([20]))


def test_actual_endpoint_gates_and_negative_diagnostic():
    membership, fixed, alpha, beta = synthetic()
    baseline = alpha.copy()
    result = parameter_gates(alpha, beta, baseline, membership, fixed)
    assert result["pass"]
    np.testing.assert_allclose(result["muscle_minimum"], 0.8)
    expected_max = np.full(96, 1.0)
    expected_max[fixed] = 0.8
    np.testing.assert_allclose(result["muscle_maximum"], expected_max)
    assert np.all(result["motor_minimum"] >= 1e-6 / np.bincount(membership)[membership])

    bad = alpha.copy()
    bad[20, 20, 0, 0] = -0.01
    gate = parameter_gates(bad, beta, baseline, membership, fixed)
    assert not gate["pass"] and not gate["gates"]["box"]
    bank = np.zeros((1, 391, 3), dtype=np.float64)
    bank[0, 20, 0] = 1
    diagnostic_kappa = np.zeros(391)
    diagnostic_kappa[20] = 2
    activation, _ = evaluate_history(
        bank,
        np.zeros((1, 96)),
        bad,
        beta,
        diagnostic_kappa,
        membership,
        fixed,
        np.zeros((42, 96)),
        np.zeros(42),
        np.zeros((1, 391)),
    )
    assert activation[0, 20] < 0  # diagnostic coefficient was not clipped

    bad = alpha.copy()
    bad[20, 19, 0, 0] = 0.001
    assert not parameter_gates(bad, beta, baseline, membership, fixed)["gates"][
        "support"
    ]
    bad = alpha.copy()
    bad[0, 0, 0, 0] += 1e-15
    assert not parameter_gates(bad, beta, baseline, membership, fixed)["gates"]["fixed"]
    bad_beta = beta.copy()
    bad_beta[0, 0, 0, 0] = 1e-15
    assert not parameter_gates(alpha, bad_beta, baseline, membership, fixed)["gates"][
        "fixed"
    ]
    bad = alpha.copy()
    bad[20, membership == 20] = 0
    assert not parameter_gates(bad, beta, baseline, membership, fixed)["gates"][
        "mn_floor"
    ]
    assert not parameter_gates(bad, beta, baseline, membership, fixed)["gates"][
        "muscle_budget"
    ]
    high = beta.copy()
    high[20, membership == 20] = 1
    gate = parameter_gates(alpha, high, baseline, membership, fixed)
    assert not gate["gates"]["muscle_budget"]


@pytest.mark.parametrize(
    ("rate", "kappa"),
    [
        (np.zeros((0, 391)), np.zeros(391)),
        (np.zeros((2, 390)), np.zeros(391)),
        (np.zeros((2, 391), dtype=np.float32), np.zeros(391)),
        (np.zeros((2, 391)), np.zeros(391, dtype=np.int64)),
        (-np.ones((2, 391)), np.zeros(391)),
    ],
)
def test_history_rejects_bad_inputs(rate, kappa):
    with pytest.raises(ValueError):
        history_state(rate, kappa)


def test_evaluation_and_gate_reject_bad_shapes_and_dtypes():
    membership, fixed, alpha, beta = synthetic()
    args = [
        np.zeros((1, 391, 3)),
        np.zeros((1, 96)),
        alpha,
        beta,
        np.zeros(391),
        membership,
        fixed,
        np.zeros((42, 96)),
        np.zeros(42),
        np.zeros((1, 391)),
    ]
    for index, invalid in (
        (0, np.zeros((1, 390, 3))),
        (3, beta.astype(np.float32)),
        (5, membership.astype(np.int32)),
        (8, np.zeros(41)),
        (9, np.ones((1, 391)) * 1.1),
    ):
        changed = args.copy()
        changed[index] = invalid
        with pytest.raises(ValueError):
            evaluate_history(*changed)
    with pytest.raises(ValueError):
        parameter_gates(alpha, beta.astype(np.float32), alpha, membership, fixed)
