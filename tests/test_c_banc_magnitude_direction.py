"""Generated-only E325 direction, reset, and failure-provenance checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import verify_c_banc_m3_magnitude_direction as e325


def _generated(monkeypatch, *, starts=(100, 104)):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(e325, "N", 4)
    monkeypatch.setattr(e325, "GROUPS", 3)
    controls = np.arange(1, 301, dtype=np.float32)
    drives = np.repeat(controls[:, None], 4, axis=1)
    q_trace = np.zeros((301, 42), dtype=np.float64)
    q_trace[1:] = (0.1 + 0.01 * controls.astype(np.float64))[:, None]
    neutral = np.asarray([np.mean(q_trace[c + 1:c + 5] ** 2)
                          for c in starts], np.float64)
    coefficients = np.asarray([.02, .01, 1e-9], np.float64)
    window_gradients = np.asarray([
        2 * np.mean(q_trace[c + 1:c + 5]) * coefficients / len(starts)
        for c in starts], dtype=np.float64)
    gradient = np.sum(window_gradients, axis=0, dtype=np.float64)
    bank = np.zeros((391, 3), np.float64)
    original = np.zeros(96, np.float64)
    whole = {"adapter": {"bank": bank, "original": {"activation": original}}}
    data = {"drives": drives, "rates": drives.copy(),
            "adaptations": drives.copy(),
            "motor_rates": np.repeat(controls[:, None], 391, axis=1),
            "raw": {"command": q_trace, "tendon": np.zeros((301, 6), np.float64)},
            "whole100": whole, "final": whole,
            "native": {"csr": {"data": np.array([.25, -.5, .125], np.float32)},
                       "mask": {"positions": np.array([0, 1, 2]),
                                "edge_parameters": np.array([0, 1, 2])}},
            "e324_gradient": gradient,
            "e324_window_gradients": window_gradients,
            "e324_neutral_losses": neutral,
            "e324_neutral_shape_loss": float(np.mean(neutral)),
            "e324_audited_result_sha256": "generated-e324-proof"}

    class Receipt:
        def __init__(self, controls_count):
            self.controls_count = controls_count
            self.status = "host_returned"
            self.forward_controls_requested = controls_count
            self.forward_controls_returned = controls_count
            self.forward_neural_ticks_requested = controls_count * 20
            self.forward_neural_ticks_returned = controls_count * 20
            self.recompute_controls_requested = 0
            self.recompute_controls_returned = 0
            self.recompute_neural_ticks_requested = 0
            self.recompute_neural_ticks_returned = 0
            self.backward_blocks_requested = 0
            self.backward_blocks_returned = 0
            self.backward_block_ticks_returned = 0
            self.confirm_requested = 0
            self.confirm_returned = 0
            self.confirmed = None

        def confirm(self):
            self.confirm_requested += 1
            self.confirm_returned += 1
            self.status = "confirmed"
            self.confirmed = {"forward_controls": self.controls_count,
                              "forward_neural_ticks": 20 * self.controls_count,
                              "recompute_controls": 0, "recompute_neural_ticks": 0,
                              "backward_blocks": 0, "backward_block_ticks": 0}

    class Decoder:
        fail_control = None

        def step(self, motor, bank_state, original_state, *, dt):
            if int(motor[0]) == self.fail_control:
                raise RuntimeError("injected decoder failure")
            q = 0.1 + 0.01 * motor[0].to(torch.float64)
            return {"q": q.expand(42), "tendon": torch.zeros(6, dtype=torch.float64),
                    "bank": bank_state, "original_activation": original_state}

    class Rollout:
        device = torch.device("cpu")
        decoder = Decoder()
        fail_start = None
        attach_receipt = False

        def __init__(self):
            self.calls = []

        def run(self, r, z, b, o, held, efficacy, mask):
            assert not torch.is_grad_enabled()
            assert not efficacy.requires_grad
            start = int(held[0, 0].item()) - 1
            self.calls.append(start)
            if start == self.fail_start:
                error = RuntimeError("injected neural forward failure")
                if self.attach_receipt:
                    error.receipt = Receipt(len(held))
                raise error
            base = 0.1 + 0.01 * held[:, 0].to(torch.float64)
            shift = (.02 * (efficacy[0] - 1) + .01 * (efficacy[1] - 1)
                     + 1e-9 * (efficacy[2] - 1))
            q = (base + shift)[:, None].expand(-1, 42)
            return {"q": q, "tendon": torch.zeros((len(held), 6), dtype=torch.float64),
                    "rate": held[-1].clone(), "adaptation": held[-1].clone(),
                    "bank": b, "original_activation": o,
                    "receipt": Receipt(len(held))}

    class Objective:
        curves = torch.zeros((6, 128, 7), dtype=torch.float64)

        def evaluate(self, q, tendon):
            loss = q[100:].square().mean()
            zero = torch.tensor(0., dtype=torch.float64)
            return {"shape_loss": loss, "per_leg_shape_loss": loss.expand(6),
                    "startup_q_rms": zero, "startup_q_max": zero,
                    "startup_tendon_rms": zero, "startup_tendon_max": zero,
                    "startup_pass": torch.tensor(True)}

    class Deps:
        def __init__(self):
            self.rollout = Rollout()
            self.build_calls = 0

        def build(self, _data, _device):
            self.build_calls += 1
            initial = (torch.zeros(4, dtype=torch.float32),
                       torch.zeros(4, dtype=torch.float32),
                       torch.zeros((391, 3), dtype=torch.float64),
                       torch.zeros(96, dtype=torch.float64),
                       torch.as_tensor(drives), torch.ones(4, dtype=torch.float32))
            return self.rollout, Objective(), initial

    def shape(q, _curves):
        loss = q.square().mean()
        return loss, loss.expand(6)

    return data, Deps(), shape


def test_magnitude_shared_direction_without_any_backward(monkeypatch):
    torch = pytest.importorskip("torch")
    assert e325.e324.N == 158706
    assert e325.e324.GROUPS == 8547
    starts = (100, 104)
    data, deps, shape = _generated(monkeypatch, starts=starts)
    saved = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("E325 must not differentiate")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    result, receipts = e325.compute(
        data, dependencies=deps, device="cpu", starts=starts,
        short_objective=shape,
        checkpoint=lambda name, arrays: saved.setdefault(name, arrays))
    assert result["comparison"]["classification"] == "supported"
    assert result["minus_improves_neutral"] is True
    assert result["continuation_requested"] is True
    assert result["continuation_completed"] is True
    assert len(receipts) == 5
    assert deps.rollout.calls == [100, 100, 104, 104, 0]
    assert result["actual_work"] == {
        "forward_controls": 316, "forward_neural_ticks": 6320,
        "recompute_controls": 0, "recompute_neural_ticks": 0,
        "backward_blocks": 0, "backward_block_ticks": 0,
        "decoder_replay_controls": 300, "body_calls": 0,
        "optimizer_steps": 0, "retry_count": 0}
    np.testing.assert_array_equal(saved["direction"]["gradient"],
                                  data["e324_gradient"])
    np.testing.assert_allclose(saved["direction"]["direction"],
                               data["e324_gradient"] / np.max(np.abs(data["e324_gradient"])))
    assert result["direction"]["theta_plus_unchanged_groups"] >= 1
    assert result["direction"]["theta_minus_unchanged_groups"] >= 1
    assert result["e324_audited_result_sha256"] == "generated-e324-proof"


def test_gradient_sum_mismatch_stops_before_model(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    data["e324_gradient"] = data["e324_gradient"].copy()
    data["e324_gradient"][0] += 1e-5
    with pytest.raises(ValueError, match="gradient/neutral evidence"):
        e325.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape)
    assert deps.build_calls == 0


def test_decoder_failure_keeps_partial_work_without_confirmation(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.decoder.fail_control = 6
    partial = []
    with pytest.raises(RuntimeError, match="decoder failure"):
        e325.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_progress=lambda value: partial.append(value))
    assert partial[-1]["decoder_replay_controls"] == 5
    assert partial[-1]["decoder_replay_work"] == {
        "requested_controls": 6, "host_returned_controls": 5,
        "device_confirmed_controls": None}
    assert deps.rollout.calls == []


def test_continuation_intent_and_primary_error_survive_receipt_fault(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.fail_start = 0
    deps.rollout.attach_receipt = True
    partial, phases = [], []

    def receipt_journal(name, _receipt):
        if name == "continuation-minus":
            raise OSError("secondary journal failure")

    with pytest.raises(RuntimeError, match="injected neural forward") as found:
        e325.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_progress=lambda value: partial.append(value),
                     on_boundary=lambda name, event: phases.append((name, event)),
                     on_receipt=receipt_journal)
    assert partial[-1]["continuation_requested"] is True
    assert partial[-1]["continuation_completed"] is False
    assert ("continuation-minus_forward", "requested") in phases
    assert ("continuation-minus_forward", "failed") in phases
    assert any("secondary journal failure" in note for note in found.value.__notes__)


def test_window_failure_retains_requested_boundary_and_neural_error(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.fail_start = 100
    deps.rollout.attach_receipt = True
    partial, phases, receipt_rows = [], [], []

    def receipt_journal(name, receipt):
        receipt_rows.append((name, receipt))
        raise OSError("secondary receipt fault")

    with pytest.raises(RuntimeError, match="injected neural forward") as found:
        e325.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_progress=lambda value: partial.append(value),
                     on_boundary=lambda name, event: phases.append((name, event)),
                     on_receipt=receipt_journal)
    assert ("window-100-plus_forward", "requested") in phases
    assert ("window-100-plus_forward", "failed") in phases
    assert not any(name.endswith("-minus_forward") for name, _ in phases)
    assert receipt_rows[0][0] == "window-100-plus"
    assert receipt_rows[0][1]["confirmed"]["forward_controls"] == 4
    assert partial[-1]["direction"]["normalization"] == "gradient/max_abs_gradient"
    assert any("secondary receipt fault" in note for note in found.value.__notes__)


def test_short_shape_matches_numpy_closed_segments():
    torch = pytest.importorskip("torch")
    from tools import verify_c_banc_m3_reset_window_direction as e324

    rng = np.random.default_rng(325)
    curves = rng.normal(size=(6, 128, 7)).astype(np.float64)
    q = rng.normal(size=(4, 42)).astype(np.float64)
    value, _ = e324.short_shape(torch.tensor(q), torch.tensor(curves))
    edges = np.roll(curves, -1, axis=1) - curves
    lengths = np.sum(edges**2, axis=-1)
    x = q.reshape(4, 6, 7)
    offset = x[:, :, None, :] - curves[None]
    factor = np.clip(np.sum(offset * edges[None], axis=-1)
                     / np.where(lengths > 0, lengths, 1)[None], 0, 1)
    nearest = curves[None] + factor[..., None] * edges[None]
    expected = np.min(np.sum((x[:, :, None] - nearest)**2, axis=-1) / 7,
                      axis=-1).mean()
    np.testing.assert_allclose(value.item(), expected, rtol=1e-14)


def test_manifest_delay_invalidates_result_and_retains_cost(tmp_path, monkeypatch):
    class Clock:
        now = 0.

        def __call__(self):
            self.now += .1
            return self.now

    clock = Clock()
    monkeypatch.setattr(e325, "compute", lambda *args, **kwargs: (
        {"comparison": {"classification": "inconclusive"}}, {}))

    class Deps:
        def load(self, _freeze, *, root):
            return {}

    freeze = {"source_sha256": {}, "input_sha256": {
        e325.PATHS["e324_result"]: "a", e325.PATHS["e324_audit_v2"]: "b",
        e325.E324 + "/direction.npz": "c"}}
    output = tmp_path / "out"

    def writer(path, value):
        if path.name == "output-manifest.json":
            clock.now += 500.
        e325.e324.put_json(path, value)

    meter = lambda: {"host_peak_bytes": 1000, "device_free_bytes": 1000,
                     "device_total_bytes": 2000, "torch_reserved_bytes": 0,
                     "torch_peak_reserved_bytes": 0}
    result = e325.run(tmp_path / "f", "f", tmp_path / "r", "r", output,
                      gatekeeper=lambda *args, **kwargs: freeze,
                      dependencies=Deps(), clock=clock, writer=writer,
                      meter=meter, device="cpu")
    assert result["run_valid"] is False
    assert any(error["stage"] == "finalization_deadline" for error in result["errors"])
    assert result["cost"]["elapsed_s"] >= 500
    assert json.loads((output / "result.json").read_text("utf8"))["run_valid"] is False
