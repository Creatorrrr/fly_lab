"""Generated CPU checks for E324's fixed-window calculation and failure journal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import verify_c_banc_m3_reset_window_direction as e324


def _generated(monkeypatch, *, starts=(100, 104)):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(e324, "N", 4)
    monkeypatch.setattr(e324, "GROUPS", 2)
    controls = np.arange(1, 301, dtype=np.float32)
    drives = np.repeat(controls[:, None], 4, axis=1)
    rates = drives.copy()
    motors = np.repeat(controls[:, None], 391, axis=1)
    q_trace = np.zeros((301, 42), dtype=np.float64)
    q_trace[1:] = (0.1 + 0.01 * controls.astype(np.float64))[:, None]
    raw = {"command": q_trace, "tendon": np.zeros((301, 6), np.float64)}
    bank = np.zeros((391, 3), np.float64)
    original = np.zeros(96, np.float64)
    whole = {"adapter": {"bank": bank, "original": {"activation": original}}}
    data = {"drives": drives, "rates": rates,
            "adaptations": rates.copy(), "motor_rates": motors, "raw": raw,
            "whole100": whole, "final": whole,
            "native": {"csr": {"data": np.array([.25, -.5], np.float32)},
                       "mask": {"positions": np.array([0, 1]),
                                "edge_parameters": np.array([0, 1])}},
            "e323_neutral_shape_loss": float(np.mean([
                np.mean(q_trace[c + 1:c + 5] ** 2) for c in starts]))}

    class Receipt:
        def __init__(self, efficacy, count):
            self.efficacy, self.count = efficacy, count
            self.status = "host_returned"
            self.forward_controls_requested = count
            self.forward_controls_returned = count
            self.forward_neural_ticks_requested = 20 * count
            self.forward_neural_ticks_returned = 20 * count
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
            backward = bool(self.efficacy.requires_grad and self.efficacy.grad is not None)
            self.confirmed = {"forward_controls": self.count,
                              "forward_neural_ticks": 20 * self.count,
                              "recompute_controls": self.count if backward else 0,
                              "recompute_neural_ticks": 20 * self.count if backward else 0,
                              "backward_blocks": self.count if backward else 0,
                              "backward_block_ticks": 20 * self.count if backward else 0}
            self.status = "confirmed"

    class Decoder:
        fail_control = None

        def step(self, motor, bank_state, original_state, *, dt):
            if int(motor[0]) == self.fail_control:
                raise RuntimeError("injected decoder replay failure")
            q = 0.1 + 0.01 * motor[0].to(torch.float64)
            return {"q": q.expand(42), "tendon": torch.zeros(6, dtype=torch.float64),
                    "bank": bank_state, "original_activation": original_state}

    class Rollout:
        device = torch.device("cpu")
        decoder = Decoder()
        fail_start = None
        bias_start = None
        attach_receipt = False

        def run(self, r, z, b, o, held, efficacy, mask):
            start = int(held[0, 0].item()) - 1
            if start == self.fail_start:
                error = RuntimeError("injected neural forward failure")
                if self.attach_receipt:
                    error.receipt = Receipt(efficacy, len(held))
                raise error
            base = 0.1 + 0.01 * held[:, 0].to(torch.float64)
            shift = .02 * (efficacy[0] - 1) + .01 * (efficacy[1] - 1)
            if start == self.bias_start:
                shift = shift + .001
            q = (base + shift)[:, None].expand(-1, 42)
            return {"q": q, "tendon": torch.zeros((len(held), 6), dtype=torch.float64),
                    "rate": held[-1].clone(), "adaptation": held[-1].clone(),
                    "bank": b, "original_activation": o,
                    "receipt": Receipt(efficacy, len(held))}

    class Objective:
        curves = torch.zeros((6, 128, 7), dtype=torch.float64)

        def evaluate(self, q, tendon):
            loss = q[100:].square().mean()
            return {"shape_loss": loss,
                    "per_leg_shape_loss": loss.expand(6),
                    "startup_q_rms": torch.tensor(0., dtype=torch.float64),
                    "startup_q_max": torch.tensor(0., dtype=torch.float64),
                    "startup_tendon_rms": torch.tensor(0., dtype=torch.float64),
                    "startup_tendon_max": torch.tensor(0., dtype=torch.float64),
                    "startup_pass": torch.tensor(True)}

    class Dependencies:
        def __init__(self):
            self.rollout = Rollout()

        def build(self, _data, _device):
            initial = (torch.zeros(4, dtype=torch.float32),
                       torch.zeros(4, dtype=torch.float32),
                       torch.zeros((391, 3), dtype=torch.float64),
                       torch.zeros(96, dtype=torch.float64),
                       torch.as_tensor(drives), torch.ones(4, dtype=torch.float32))
            return self.rollout, Objective(), initial

    def shape(q, _curves):
        loss = q.square().mean()
        return loss, loss.expand(6)

    return data, Dependencies(), shape


def test_shared_gradient_and_same_reset_boundaries(monkeypatch):
    starts = (100, 104)
    data, deps, shape = _generated(monkeypatch, starts=starts)
    saved, events = {}, []
    result, receipts = e324.compute(
        data, dependencies=deps, device="cpu", starts=starts,
        short_objective=shape, checkpoint=lambda name, arrays: saved.setdefault(name, arrays),
        on_boundary=lambda name, event: events.append((name, event)))
    assert result["comparison"]["classification"] == "supported"
    assert result["minus_improves_neutral"] is True
    assert result["continuation_requested"] is True
    assert len(result["windows"]) == 2
    assert len(receipts) == 7
    assert result["actual_work"] == {
        "forward_controls": 324, "forward_neural_ticks": 6480,
        "recompute_controls": 8, "recompute_neural_ticks": 160,
        "backward_blocks": 8, "backward_block_ticks": 160,
        "decoder_replay_controls": 300, "body_calls": 0,
        "optimizer_steps": 0, "retry_count": 0}
    q = data["raw"]["command"]
    expected = sum((np.mean(q[c + 1:c + 5]) * 2 / 2)
                   for c in starts)
    np.testing.assert_allclose(saved["direction"]["gradient"],
                               expected * np.array([.02, .01]), rtol=1e-12)
    assert all((f"window-{c:03d}-neutral_backward", "requested") in events
               for c in starts)
    assert all(row["neutral_reproduction"]["end_rate_adaptation_dtype_bytes_exact"]
               for row in result["runs"].values() if row["label"] == "neutral")


def test_bad_neutral_stops_before_backward_but_keeps_forward(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    data["raw"]["command"][101, 0] += 0.001
    # Decoder replay must catch this before even starting a neural forward.
    saved, events = [], []
    with pytest.raises(ValueError, match="decoder replay reproduction"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     checkpoint=lambda name, _: saved.append(name),
                     on_boundary=lambda name, event: events.append((name, event)))
    assert "decoder-replay" in saved
    assert not any("backward" in name for name, _ in events)


def test_window_mismatch_keeps_raw_receipt_and_stops_before_backward(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.bias_start = 100
    saved, events, receipts = [], [], {}
    with pytest.raises(ValueError, match="neutral window reproduction"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     checkpoint=lambda name, _: saved.append(name),
                     on_boundary=lambda name, event: events.append((name, event)),
                     on_receipt=lambda name, value: receipts.update({name: value}))
    assert "decoder-replay" in saved
    assert "window-100-neutral" in saved
    assert receipts["window-100-neutral"]["confirmed"]["forward_controls"] == 4
    assert ("window-100-neutral_forward", "device_confirmed") in events
    assert not any("backward" in name for name, _ in events)


def test_incomplete_input_and_forward_exception_have_unknown_work(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    data["drives"] = data["drives"][:-1]
    with pytest.raises(ValueError, match="all 300"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape)
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.fail_start = 100
    events = []
    with pytest.raises(RuntimeError, match="injected neural"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_boundary=lambda name, event: events.append((name, event)))
    assert ("window-100-neutral_forward", "requested") in events
    assert ("window-100-neutral_forward", "failed") in events
    assert not any("backward" in name for name, _ in events)


def test_replay_failure_reports_requested_returned_without_confirmation(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.decoder.fail_control = 6
    partial, events = [], []
    with pytest.raises(RuntimeError, match="decoder replay failure"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_progress=lambda value: partial.append(value),
                     on_boundary=lambda name, event: events.append((name, event)))
    assert partial[-1]["decoder_replay_controls"] == 5
    assert partial[-1]["decoder_replay_work"] == {
        "requested_controls": 6, "host_returned_controls": 5,
        "device_confirmed_controls": None}
    assert ("decoder_replay", "failed") in events
    assert not any("window-" in name for name, _ in events)


def test_continuation_failure_retains_requested_intent(monkeypatch):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.fail_start = 0
    partial, events = [], []
    with pytest.raises(RuntimeError, match="neural forward failure"):
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape,
                     on_progress=lambda value: partial.append(value),
                     on_boundary=lambda name, event: events.append((name, event)))
    assert partial[-1]["continuation_requested"] is True
    assert partial[-1]["continuation_completed"] is False
    assert ("continuation-minus_forward", "requested") in events
    assert ("continuation-minus_forward", "failed") in events


@pytest.mark.parametrize("fail_start", [100, 0])
def test_receipt_journal_failure_cannot_mask_neural_exception(monkeypatch, fail_start):
    data, deps, shape = _generated(monkeypatch, starts=(100,))
    deps.rollout.fail_start = fail_start
    deps.rollout.attach_receipt = True

    failed_name = "window-100-neutral" if fail_start == 100 else "continuation-minus"

    def journal_fails(name, _receipt):
        if name == failed_name:
            raise OSError("secondary receipt journal fault")

    with pytest.raises(RuntimeError, match="injected neural forward failure") as found:
        e324.compute(data, dependencies=deps, device="cpu", starts=(100,),
                     short_objective=shape, on_receipt=journal_fails)
    assert any("secondary receipt journal fault" in note
               for note in found.value.__notes__)


def test_short_shape_matches_independent_segment_distance():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(324)
    curves = rng.normal(size=(6, 128, 7)).astype(np.float64)
    q = rng.normal(size=(4, 42)).astype(np.float64)
    value, legs = e324.short_shape(torch.tensor(q), torch.tensor(curves))
    edge = np.roll(curves, -1, axis=1) - curves
    lengths = np.sum(edge**2, axis=-1)
    x = q.reshape(4, 6, 7)
    offsets = x[:, :, None, :] - curves[None, :, :, :]
    t = np.clip(np.sum(offsets * edge[None], axis=-1)
                / np.where(lengths > 0, lengths, 1)[None], 0, 1)
    points = curves[None] + t[..., None] * edge[None]
    expected = np.min(np.sum((x[:, :, None] - points)**2, axis=-1) / 7,
                      axis=-1).mean(axis=0)
    np.testing.assert_allclose(legs.detach().numpy(), expected, rtol=1e-14)
    np.testing.assert_allclose(value.item(), expected.mean(), rtol=1e-14)


def test_passive_gate_rejects_review_without_loading_payload(tmp_path, monkeypatch):
    freeze, review = tmp_path / "freeze.json", tmp_path / "review.json"
    freeze.write_text("{}", "utf8")
    review.write_text("{}", "utf8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Fresh literal"):
        e324.gate(freeze, "bad", review, "bad", output, root=tmp_path)
    assert not output.exists()


def test_terminal_manifest_delay_is_failure_with_nonzero_cost(tmp_path, monkeypatch):
    class Clock:
        now = 0.

        def __call__(self):
            self.now += .1
            return self.now

    clock = Clock()
    monkeypatch.setattr(e324, "compute", lambda *args, **kwargs: (
        {"comparison": {"classification": "inconclusive"}}, {}))
    class Deps:
        def load(self, _freeze, *, root):
            return {}

    output = tmp_path / "out"
    freeze = {"source_sha256": {}, "input_sha256": {
        e324.PATHS["e323_result"]: "a", e324.PATHS["e323_audit"]: "b"}}
    def writer(path: Path, value: dict):
        if path.name == "output-manifest.json":
            clock.now += 500.
        e324.put_json(path, value)

    meter = lambda: {"host_peak_bytes": 1000, "device_free_bytes": 1000,
                     "device_total_bytes": 2000, "torch_reserved_bytes": 0,
                     "torch_peak_reserved_bytes": 0}
    result = e324.run(tmp_path / "f", "f", tmp_path / "r", "r", output,
                      gatekeeper=lambda *args, **kwargs: freeze,
                      dependencies=Deps(), clock=clock, writer=writer,
                      meter=meter, device="cpu")
    assert result["run_valid"] is False
    assert any(row["stage"] == "finalization_deadline" for row in result["errors"])
    assert result["cost"]["elapsed_s"] >= 500
    assert json.loads((output / "result.json").read_text("utf8"))["run_valid"] is False


def test_failed_return_boundary_keeps_raw_and_unknown_count(tmp_path, monkeypatch):
    def interrupted(*_args, **kwargs):
        kwargs["on_boundary"]("window-100-neutral_forward", "requested")
        kwargs["checkpoint"]("window-100-neutral", {
            "q": np.zeros((4, 42), np.float64)})
        raise RuntimeError("injected unreturned neural work")

    monkeypatch.setattr(e324, "compute", interrupted)
    class Deps:
        def load(self, _freeze, *, root):
            return {}

    freeze = {"source_sha256": {}, "input_sha256": {
        e324.PATHS["e323_result"]: "a", e324.PATHS["e323_audit"]: "b"}}
    meter = lambda: {"host_peak_bytes": 1000, "device_free_bytes": 1000,
                     "device_total_bytes": 2000, "torch_reserved_bytes": 0,
                     "torch_peak_reserved_bytes": 0}
    output = tmp_path / "out"
    result = e324.run(tmp_path / "f", "f", tmp_path / "r", "r", output,
                      gatekeeper=lambda *args, **kwargs: freeze,
                      dependencies=Deps(), meter=meter, device="cpu")
    assert result["run_valid"] is False
    assert result["errors"][0]["stage"] == "science"
    assert (output / "window-100-neutral.npz").is_file()
    phase = result["cost"]["phases"]["window-100-neutral_forward"]
    assert phase["requested"] is True
    assert phase["host_returned"] is None
    assert phase["device_confirmed"] is None
    assert result["cost"]["elapsed_s"] > 0
