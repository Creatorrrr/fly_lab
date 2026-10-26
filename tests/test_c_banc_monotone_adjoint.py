"""Generated E316 anatomy only; CUDA coverage is opt-in for root review."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.banc_m3_monotone_adjoint import MonotoneDecoder
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentAdapter
from tools.banc_m3_rhythmic_recruitment import AnatomicalRecruitmentAdapter


def generated(seed=31601):
    rng = np.random.default_rng(seed)
    counts = np.full(96, 4, dtype=np.int64)
    counts[:10] = 5
    counts[10] = 1
    membership = np.repeat(np.arange(96), counts)[rng.permutation(391)]
    pool = np.zeros((96, 391), dtype=np.float64)
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        pool[muscle, members] = 1.0 / len(members)
    ltm = np.array([15, 30, 46, 63, 78, 94], dtype=np.int64)
    groups = [
        {"leg": ("lf", "lm", "lh", "rf", "rm", "rh")[i // 16 if i < 96 else 5],
         "muscle": f"muscle_{i}", "columns": np.flatnonzero(membership == i).tolist()}
        for i in range(96)
    ]
    moment = rng.normal(0.0, 0.05, (42, 96)).astype(np.float64)
    calibration = {
        "moment": moment,
        "rest": np.linspace(-0.2, 0.2, 42, dtype=np.float64),
        "pool": pool,
        "motor_ids": np.arange(391, dtype=np.int64),
        "ltm": ltm,
    }
    slopes = np.zeros((96, 391, 3, 2), dtype=np.float64)
    fixed = set(ltm.tolist()) | {10}
    for muscle in range(96):
        members = np.flatnonzero(membership == muscle)
        if muscle in fixed:
            slopes[muscle, members, 1, :] = pool[muscle, members, None]
        else:
            low_scale, high_scale = ((0.8, 1.2) if muscle % 2 else (1.2, 0.8))
            for channel, ratio in enumerate((0.15, 0.7, 0.15)):
                slopes[muscle, members, channel, 0] = low_scale * ratio / len(members)
                slopes[muscle, members, channel, 1] = high_scale * ratio / len(members)
    tau = np.array([0.005, 0.01, 0.04], dtype=np.float64)
    kappa = np.linspace(0.4, 3.0, 391, dtype=np.float64)
    neutral = np.linspace(-0.3, 0.3, 42, dtype=np.float64)
    roster = [f"generated:{i}" for i in range(391)]
    return calibration, tau, kappa, slopes, membership, neutral, groups, roster


def old_adapter(parts):
    calibration, tau, kappa, slopes, membership, neutral, groups, roster = parts
    body = SimpleNamespace(neutral=neutral.copy())
    base = SimpleNamespace(motor_ids=calibration["motor_ids"].copy(), identity="generated-base", body=body)
    anatomical = AnatomicalRecruitmentAdapter(base, calibration, groups)
    return MonotoneRecruitmentAdapter(anatomical, tau, kappa, slopes, membership, roster)


def tensor_state(rate, bank, original, *, device="cpu", requires_grad=False):
    return (
        torch.tensor(rate, dtype=torch.float32, device=device, requires_grad=requires_grad),
        torch.tensor(bank, dtype=torch.float64, device=device, requires_grad=requires_grad),
        torch.tensor(original, dtype=torch.float64, device=device, requires_grad=requires_grad),
    )


def numpy_step(parts, rate, bank, original, dt):
    cal, tau, kappa, slopes, membership, _, _, _ = parts
    rate64 = rate.astype(np.float64)
    pooled = cal["pool"] @ rate64
    new_original = original + (-np.expm1(-dt / .01)) * (pooled - original)
    new_bank = bank + (-np.expm1(-dt / tau)) * (rate64[:, None] - bank)
    activation = np.empty(96, dtype=np.float64)
    fixed = set(cal["ltm"].tolist()) | set(np.flatnonzero(np.bincount(membership, minlength=96) == 1))
    for muscle in range(96):
        if muscle in fixed:
            activation[muscle] = new_original[muscle]
        else:
            ids = np.flatnonzero(membership == muscle)
            low = slopes[muscle, ids, :, 0]
            delta = slopes[muscle, ids, :, 1] - low
            activation[muscle] = np.sum(low * new_bank[ids]) + np.sum(
                delta * np.maximum(new_bank[ids] - kappa[ids, None], 0.0)
            )
    return {
        "bank": new_bank,
        "original_activation": new_original,
        "activation": activation,
        "q": cal["rest"] + .02 * (cal["moment"] @ activation),
        "tendon": .2 * np.tanh(activation[cal["ltm"]] / 10),
    }


def assert_forward(actual, expected):
    for key in ("bank", "original_activation", "activation", "q", "tendon"):
        tolerance = 1e-11 if key in ("q", "tendon") else 1e-10
        np.testing.assert_allclose(actual[key].detach().cpu().numpy(), expected[key], rtol=0, atol=tolerance)


def test_two_live_rollouts_match_e150_and_independent_numpy():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    rng = np.random.default_rng(31602)
    for rollout in range(2):
        old = old_adapter(parts)
        bank = rng.uniform(0.01, 4.0, (391, 3))
        original = rng.uniform(0.01, 5.0, 96)
        old.bank[:] = bank
        old.original.activation[:] = original
        dts = ([.003, .005, .011, .05] if rollout == 0 else [.02, .007, .001, .015])
        for dt in dts:
            rate = rng.uniform(0, 8, 391).astype(np.float32)
            inputs = tensor_state(rate, bank, original)
            saved = tuple(value.clone() for value in inputs)
            out = decoder.step(*inputs, dt=dt)
            expected = numpy_step(parts, rate, bank, original, dt)
            assert_forward(out, expected)
            q, _ = old.decode(rate.astype(np.float64), dt=dt)
            np.testing.assert_allclose(out["bank"].numpy(), old.bank, rtol=0, atol=1e-10)
            np.testing.assert_allclose(out["original_activation"].numpy(), old.original.activation, rtol=0, atol=1e-10)
            np.testing.assert_allclose(out["activation"].numpy(), old.activation, rtol=0, atol=1e-10)
            np.testing.assert_allclose(out["q"].numpy(), q, rtol=0, atol=1e-11)
            np.testing.assert_allclose(out["tendon"].numpy(), np.array(list(old.tendon_inputs.values())), rtol=0, atol=1e-11)
            for value, before in zip(inputs, saved):
                torch.testing.assert_close(value, before, rtol=0, atol=0)
            bank = out["bank"].detach().numpy()
            original = out["original_activation"].detach().numpy()


def test_hinge_equality_uses_low_slope_and_both_sides():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    member = int(np.flatnonzero(parts[4] == 12)[0])
    rate = np.full(391, 2.0, dtype=np.float32)
    bank = np.full((391, 3), 2.0, dtype=np.float64)
    original = np.ones(96, dtype=np.float64)
    parts[2][member] = 2.0
    decoder = MonotoneDecoder(*parts[:6])
    for displacement in (-1e-3, 0.0, 1e-3):
        old_bank = bank.copy()
        old_bank[member, 0] += displacement
        r, b, a = tensor_state(rate, old_bank, original, requires_grad=True)
        out = decoder.step(r, b, a)
        out["activation"][12].backward()
        low = parts[3][12, member, 0, 0]
        high = parts[3][12, member, 0, 1]
        alpha = -np.expm1(-.005 / parts[1][0])
        expected = (high if displacement > 0 else low) * (1 - alpha)
        assert b.grad[member, 0].item() == pytest.approx(expected, abs=1e-12)


def test_cut_is_exact_zero_and_restart_is_fresh():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    rng = np.random.default_rng(31603)
    rate, bank, old = tensor_state(
        rng.uniform(0, 5, 391), rng.uniform(0, 5, (391, 3)), rng.uniform(0, 5, 96), requires_grad=True
    )
    cut = decoder.step(rate, bank, old, motor_cut=True)
    for key in ("bank", "original_activation", "activation", "tendon"):
        assert torch.count_nonzero(cut[key]) == 0
    np.testing.assert_array_equal(cut["q"].detach().numpy(), parts[5])
    (cut["q"].sum() + cut["bank"].sum() + cut["tendon"].sum()).backward()
    for value in (rate, bank, old):
        assert torch.count_nonzero(value.grad) == 0
    restarted = decoder.step(rate.detach(), cut["bank"].detach(), cut["original_activation"].detach())
    fresh = decoder.step(rate.detach(), torch.zeros_like(bank), torch.zeros_like(old))
    for key in fresh:
        torch.testing.assert_close(restarted[key], fresh[key], atol=0, rtol=0)


def test_cut_with_huge_finite_states_stays_exact_and_has_zero_vjp():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    rate, bank, old = tensor_state(
        np.ones(391, dtype=np.float32),
        np.full((391, 3), 1e308, dtype=np.float64),
        np.full(96, 1e308, dtype=np.float64),
        requires_grad=True,
    )
    cut = decoder.step(rate, bank, old, motor_cut=True)
    for key in ("bank", "original_activation", "activation", "tendon"):
        assert torch.count_nonzero(cut[key]) == 0
    np.testing.assert_array_equal(cut["q"].detach().numpy(), parts[5])
    sum(value.sum() for value in cut.values()).backward()
    for value in (rate, bank, old):
        assert torch.count_nonzero(value.grad) == 0
        assert torch.isfinite(value.grad).all()


def test_noncut_rejects_finite_inputs_that_overflow_activation():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    rate, bank, old = tensor_state(
        np.ones(391, dtype=np.float32),
        np.full((391, 3), 1.7e308, dtype=np.float64),
        np.zeros(96, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="Finite nonnegative muscle state"):
        decoder.step(rate, bank, old, dt=1e-6)


def test_gradients_match_independent_finite_differences():
    parts = generated()
    decoder = MonotoneDecoder(*parts[:6])
    rng = np.random.default_rng(31604)
    rate = rng.uniform(1, 6, 391).astype(np.float32)
    bank = rng.uniform(.1, 5, (391, 3))
    original = rng.uniform(.1, 5, 96)
    dt = .011
    r, b, a = tensor_state(rate, bank, original, requires_grad=True)
    out = decoder.step(r, b, a, dt=dt)
    loss = out["q"].square().sum() + out["tendon"].square().sum() + .01 * out["activation"].sum()
    loss.backward()

    def objective(rr, bb, aa):
        result = numpy_step(parts, rr, bb, aa, dt)
        return np.square(result["q"]).sum() + np.square(result["tendon"]).sum() + .01 * result["activation"].sum()

    for index in (0, 49, 201, 390):
        h = .001  # Larger than one float32 ulp at these rates.
        plus, minus = rate.copy(), rate.copy()
        plus[index] = np.float32(rate[index] + h)
        minus[index] = np.float32(rate[index] - h)
        derivative = (objective(plus, bank, original) - objective(minus, bank, original)) / float(plus[index] - minus[index])
        assert r.grad[index].item() == pytest.approx(derivative, rel=1e-6, abs=1e-8)
    for index, channel in ((0, 0), (55, 1), (190, 2), (390, 0)):
        h = 1e-5
        plus, minus = bank.copy(), bank.copy()
        plus[index, channel] += h
        minus[index, channel] -= h
        derivative = (objective(rate, plus, original) - objective(rate, minus, original)) / (2 * h)
        assert b.grad[index, channel].item() == pytest.approx(derivative, rel=1e-6, abs=1e-8)
    for index in (0, 10, 15, 95):
        h = 1e-5
        plus, minus = original.copy(), original.copy()
        plus[index] += h
        minus[index] -= h
        derivative = (objective(rate, bank, plus) - objective(rate, bank, minus)) / (2 * h)
        assert a.grad[index].item() == pytest.approx(derivative, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("which", ["moment", "pool", "motor_ids", "ltm", "tau", "kappa", "slopes", "membership", "neutral"])
def test_invalid_constructor_inputs(which):
    parts = list(generated())
    calibration = dict(parts[0])
    parts[0] = calibration
    if which == "moment":
        calibration["moment"] = np.zeros((41, 96), dtype=np.float64)
    elif which == "pool":
        calibration["pool"] = calibration["pool"].copy()
        calibration["pool"][0, 0] = 1.0
    elif which == "motor_ids":
        calibration["motor_ids"] = calibration["motor_ids"].copy()
        calibration["motor_ids"][1] = 0
    elif which == "ltm":
        calibration["ltm"] = np.zeros(6, dtype=np.int64)
    elif which == "tau":
        parts[1] = parts[1].copy()
        parts[1][0] += 1e-5
    elif which == "kappa":
        parts[2] = parts[2].copy()
        parts[2][0] = -1
    elif which == "slopes":
        parts[3] = parts[3].copy()
        parts[3][0, 0, 0, 0] = 1
    elif which == "membership":
        parts[4] = parts[4].copy()
        parts[4][0] = 97
    elif which == "neutral":
        parts[5] = np.full(42, np.nan)
    with pytest.raises(ValueError):
        MonotoneDecoder(*parts[:6])


def test_constructor_owns_arrays_and_step_rejects_invalid_inputs():
    parts = list(generated())
    decoder = MonotoneDecoder(*parts[:6])
    original_rest = decoder.rest.clone()
    parts[0]["rest"][:] = 99
    torch.testing.assert_close(decoder.rest, original_rest, atol=0, rtol=0)
    rate, bank, original = tensor_state(np.ones(391), np.ones((391, 3)), np.ones(96))
    for args in (
        (rate[:390], bank, original),
        (rate.to(torch.float64), bank, original),
        (-rate, bank, original),
        (rate, bank.to(torch.float32), original),
        (rate, bank * float("nan"), original),
        (rate, bank, original * -1),
    ):
        with pytest.raises(ValueError):
            decoder.step(*args)
    for bad in (0, -.001, .051, float("nan"), True):
        with pytest.raises(ValueError):
            decoder.step(rate, bank, original, dt=bad)
    with pytest.raises(ValueError):
        decoder.step(rate, bank, original, motor_cut=1)


@pytest.mark.skipif(os.environ.get("BANC_E316_GENERATED_CUDA") != "1", reason="root-only opt-in CUDA")
def test_generated_cuda_matches_cpu_and_has_gradient(monkeypatch):
    parts = generated()
    cpu = MonotoneDecoder(*parts[:6])
    gpu = MonotoneDecoder(*parts[:6], device="cuda")
    rng = np.random.default_rng(31605)
    rate = rng.uniform(0, 8, 391).astype(np.float32)
    bank = rng.uniform(.1, 5, (391, 3))
    original = rng.uniform(.1, 5, 96)
    a = cpu.step(*tensor_state(rate, bank, original), dt=.007)
    r, b, o = tensor_state(rate, bank, original, device="cuda", requires_grad=True)
    def forbidden_host_value(*_args, **_kwargs):
        raise AssertionError("CUDA hot step extracted a tensor value on host")

    with monkeypatch.context() as patch:
        for name in ("item", "cpu", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden_host_value)
        c = gpu.step(r, b, o, dt=.007)
    for key in a:
        torch.testing.assert_close(c[key].cpu(), a[key], atol=1e-10, rtol=0)
    c["q"].square().sum().backward()
    assert torch.isfinite(r.grad).all() and torch.isfinite(b.grad).all() and torch.isfinite(o.grad).all()
