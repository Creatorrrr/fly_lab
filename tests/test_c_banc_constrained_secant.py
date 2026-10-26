"""Generated-only E326 interval, continuous rollout, and failure provenance."""

from __future__ import annotations

import numpy as np
import pytest

from tools import verify_c_banc_m3_constrained_secant as e326


def _startup(q0=0., q1=.08, t0=0., t1=0.):
    return (np.full((99, 42), q0, np.float64),
            np.full((99, 6), t0, np.float64),
            np.full((99, 42), q1, np.float64),
            np.full((99, 6), t1, np.float64),
            np.zeros((99, 42), np.float64),
            np.zeros((99, 6), np.float64))


def test_interval_exact_scalar_and_midpoint():
    values = list(_startup())
    values[2] = np.zeros((99, 42), np.float64)
    values[2][:, 0] = .08
    answer = e326.feasible_alpha(*values)
    assert answer["alpha_max"] == pytest.approx(.5)
    assert answer["alpha_star"] == pytest.approx(.25)
    assert answer["limiting_constraints"] == ["q_max"]
    assert answer["predicted_metrics"]["q"]["max"] == pytest.approx(.02)
    assert answer["limits"]["q_rms"] > .5


def test_interval_independent_bisection_random():
    rng = np.random.default_rng(326)
    for _ in range(20):
        q0 = rng.normal(0, .004, (99, 42))
        t0 = rng.normal(0, .002, (99, 6))
        q1 = q0 + rng.normal(0, .06, (99, 42))
        t1 = t0 + rng.normal(0, .03, (99, 6))
        zeros_q, zeros_t = np.zeros_like(q0), np.zeros_like(t0)
        answer = e326.feasible_alpha(q0, t0, q1, t1, zeros_q, zeros_t)

        def okay(alpha, q0=q0, q1=q1, t0=t0, t1=t1):
            q = q0 + alpha * (q1 - q0)
            t = t0 + alpha * (t1 - t0)
            return (np.sqrt(np.mean(q * q)) <= .018
                    and np.max(np.abs(q)) <= .04
                    and np.sqrt(np.mean(t * t)) <= .009
                    and np.max(np.abs(t)) <= .025)

        lo, hi = 0., 1.
        for _ in range(55):
            middle = (lo + hi) / 2
            if okay(middle):
                lo = middle
            else:
                hi = middle
        assert answer["alpha_max"] == pytest.approx(lo, abs=2e-14)
        assert okay(answer["alpha_star"])


def test_interval_zero_tiny_invalid_and_strict_neutral():
    values = _startup(q1=0., t1=0.)
    assert e326.feasible_alpha(*values)["alpha_max"] == 1.
    tiny = list(values)
    tiny[2] = np.full((99, 42), 1e-14, np.float64)
    assert e326.feasible_alpha(*tiny)["alpha_max"] == 1.
    bad = list(values)
    bad[0] = np.full((99, 42), .04, np.float64)
    with pytest.raises(ValueError, match="strictly pass"):
        e326.feasible_alpha(*bad)
    bad = list(values)
    bad[3] = bad[3].copy()
    bad[3][0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite matching"):
        e326.feasible_alpha(*bad)


def _generated():
    torch = pytest.importorskip("torch")
    q0 = np.zeros((300, 42), np.float64)
    q1 = q0.copy()
    q1[:99, 0] = .08
    t0 = np.zeros((300, 6), np.float64)
    t1 = t0.copy()
    theta1 = np.array([np.float32(.999), np.float32(1.001)], np.float64)
    data = {"drives": np.ones((300, 4), np.float32),
            "motor_rates": np.zeros((300, 391), np.float64),
            "e323_q": q0, "e323_tendon": t0,
            "e325_q": q1, "e325_tendon": t1,
            "e325_theta_minus": theta1,
            "e323_neutral_shape_loss": .01,
            "startup": {"target_command": np.zeros((99, 42), np.float64),
                        "target_tendon": np.zeros((99, 6), np.float64)},
            "native": {"csr": {"data": np.array([.5, -.3], np.float32)},
                       "mask": {"positions": np.array([0, 1]),
                                "edge_parameters": np.array([0, 1])}}}

    class Receipt:
        status = "host_returned"
        forward_controls_requested = 300
        forward_controls_returned = 300
        forward_neural_ticks_requested = 6000
        forward_neural_ticks_returned = 6000
        recompute_controls_requested = 0
        recompute_controls_returned = 0
        recompute_neural_ticks_requested = 0
        recompute_neural_ticks_returned = 0
        backward_blocks_requested = 0
        backward_blocks_returned = 0
        backward_block_ticks_returned = 0
        confirm_requested = 0
        confirm_returned = 0
        confirmed = None

        def confirm(self):
            self.confirm_requested += 1
            self.confirm_returned += 1
            self.status = "confirmed"
            self.confirmed = {"forward_controls": 300, "forward_neural_ticks": 6000,
                              "recompute_controls": 0, "recompute_neural_ticks": 0,
                              "backward_blocks": 0, "backward_block_ticks": 0}

    class Rollout:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0
            self.fail = False

        def run(self, _r, _z, bank, original, held, theta, _mask):
            self.calls += 1
            assert not torch.is_grad_enabled()
            assert held.shape == (300, 4)
            if self.fail:
                error = RuntimeError("injected neural failure")
                error.receipt = Receipt()
                raise error
            # Generated secant is exactly affine in theta[0].
            fraction = (theta[0].to(torch.float64) - 1) / (theta1[0] - 1)
            q = torch.as_tensor(q0 + fraction.item() * (q1 - q0))
            tendon = torch.as_tensor(t0)
            return {"q": q, "tendon": tendon,
                    "rate": torch.zeros(4, dtype=torch.float32),
                    "adaptation": torch.zeros(4, dtype=torch.float32),
                    "bank": bank, "original_activation": original,
                    "receipt": Receipt()}

    class Objective:
        def evaluate(self, q, tendon):
            q99, t99 = q[:99], tendon[:99]
            return {"shape_loss": torch.tensor(.009, dtype=torch.float64),
                    "per_leg_shape_loss": torch.full((6,), .009, dtype=torch.float64),
                    "startup_q_rms": q99.square().mean().sqrt(),
                    "startup_q_max": q99.abs().max(),
                    "startup_tendon_rms": t99.square().mean().sqrt(),
                    "startup_tendon_max": t99.abs().max(),
                    "startup_pass": torch.tensor(True)}

    class Deps:
        def __init__(self):
            self.rollout = Rollout()

        def build(self, _data, _device):
            return self.rollout, Objective(), (
                torch.zeros(4, dtype=torch.float32),
                torch.zeros(4, dtype=torch.float32),
                torch.zeros((391, 3), dtype=torch.float64),
                torch.zeros(96, dtype=torch.float64),
                torch.as_tensor(data["drives"]),
                torch.ones(4, dtype=torch.float32))

    return data, Deps()


def test_one_continuous_generated_rollout_no_backward(monkeypatch):
    torch = pytest.importorskip("torch")
    data, deps = _generated()
    saved, phases = {}, []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no new gradient")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    result, receipts = e326.compute(
        data, dependencies=deps, device="cpu",
        checkpoint=lambda name, arrays: saved.setdefault(name, arrays),
        on_boundary=lambda name, event: phases.append((name, event)))
    assert deps.rollout.calls == 1
    assert result["alpha"]["alpha_max"] == pytest.approx(.5)
    assert result["alpha"]["alpha_star"] == pytest.approx(.25)
    assert result["actual_work"]["forward_neural_ticks"] == 6000
    assert result["actual_work"]["backward_blocks"] == 0
    assert result["constrained_continuation_supported"] is True
    assert saved["constrained-step"]["alpha_star"] == pytest.approx(.25)
    assert "continuation-constrained" in receipts
    assert ("continuation-constrained_forward", "device_confirmed") in phases
    np.testing.assert_allclose(saved["continuation-constrained"]["q"][:99, 0],
                               .08 * ((saved["constrained-step"]["theta"][0] - 1)
                                      / (data["e325_theta_minus"][0] - 1)))


def test_failure_keeps_primary_error_and_partial_receipt():
    data, deps = _generated()
    deps.rollout.fail = True
    progress, phases = [], []

    def journal(_name, _value):
        raise OSError("secondary journal failure")

    with pytest.raises(RuntimeError, match="injected neural failure") as found:
        e326.compute(data, dependencies=deps, device="cpu",
                     on_receipt=journal, on_progress=progress.append,
                     on_boundary=lambda name, event: phases.append((name, event)))
    assert progress[-1]["continuation_requested"] is True
    assert progress[-1]["continuation_completed"] is False
    assert ("continuation-constrained_forward", "failed") in phases
    assert any("secondary journal failure" in note for note in found.value.__notes__)
