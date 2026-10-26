"""Generated CPU E332 math/closure/failure tests; no original NPZ payloads."""
from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import analyze_c_banc_m3_smooth_temporal as subject
from tools.banc_m3_smooth_temporal import SmoothTemporalObjective


def sample(curves, phase):
    position = np.remainder(phase, 1) * 128
    whole = np.floor(position)
    index = whole.astype(np.int64) % 128
    first = curves[np.arange(6)[None], index]
    last = curves[np.arange(6)[None], (index + 1) % 128]
    return (first * (1 - (position - whole)[..., None])
            + last * (position - whole)[..., None]).reshape(-1, 42)


def independent(curves, normalization, boundary, q):
    """Independent NumPy moment, pair-free scalar and covariance derivative."""
    phase = np.tile(np.arange(128, dtype=np.float64)[:, None] / 128, (1, 6))
    a_ref = sample(curves, phase + 2.382760007990865 * .005).reshape(128, 6, 7).transpose(1, 0, 2)
    angle = 2 * np.pi * (np.arange(128, dtype=np.float64) / 128)
    basis = np.stack((np.cos(angle), np.sin(angle)), -1)
    tau = 7 * normalization

    def encoding(commands):
        x = commands.reshape(-1, 6, 7)
        distance = np.einsum("tlsj,tlsj->tls", x[:, :, None] - curves, x[:, :, None] - curves)
        logits = -distance / tau[None, :, None]
        exponentials = np.exp(logits - logits.max(-1, keepdims=True))
        weights = exponentials / exponentials.sum(-1, keepdims=True)
        target = np.einsum("tls,lsj->tlj", weights, a_ref)
        z = np.einsum("tls,sk->tlk", weights, basis)
        center = np.einsum("tls,lsj->tlj", weights, curves)
        ja = 2 * np.einsum("tls,tlsa,tlsb->tlab", weights, a_ref[None] - target[:, :, None],
                           curves[None] - center[:, :, None]) / tau[None, :, None, None]
        jz = 2 * np.einsum("tls,tlsk,tlsb->tlkb", weights, basis[None, None] - z[:, :, None],
                           curves[None] - center[:, :, None]) / tau[None, :, None, None]
        return {"weights": weights, "target": target.reshape(-1, 42), "z": z, "ja": ja, "jz": jz}

    previous = np.vstack((boundary, q[:3]))
    p, n = encoding(previous), encoding(q)
    residual = (q - p["target"]).reshape(4, 6, 7)
    leg = np.mean(residual ** 2, axis=(0, 2)) / normalization
    z = n["z"]
    sum_z = z.sum(1)
    coordination = np.mean(1 - (np.sum(sum_z ** 2, -1) - np.sum(z ** 2, (1, 2))) / 30)
    pair = np.stack([1 - np.einsum("tk,tk->t", z[:, a], z[:, b]) for a in range(6) for b in range(a+1, 6)], -1)
    direct = 2 * residual / (4 * 6 * 7 * normalization[None, :, None])
    previous_g = -np.einsum("tla,tlab->tlb", direct, p["ja"])
    g = direct - np.einsum("tlk,tlkb->tlb", (sum_z[:, None] - z) / (4 * 15), n["jz"])
    g[:3] += previous_g[1:]
    return {"terms": np.array([leg.mean() + coordination, leg.mean(), coordination, pair[:, [2, 7, 11]].mean()]),
            "g": g.reshape(4, 42), "previous": p, "next": n, "per_leg": leg, "pair": pair.mean(0)}


@pytest.fixture
def fixture():
    phase = 2 * np.pi * (np.arange(128, dtype=np.float64) / 128)
    curves = np.empty((6, 128, 7), np.float64)
    for leg in range(6):
        for joint in range(7):
            curves[leg, :, joint] = (.11 + .013 * leg + .009 * joint) * (
                np.sin(phase + .33 * joint) + .12 * np.cos(2 * phase + .14 * leg))
    objective = SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    p = np.tile(np.array([127.81, 1.74, 3.16, 5.21])[:, None] / 128, (1, 6))
    q = sample(curves, p)
    q += np.random.default_rng(332).normal(size=q.shape) * .0003
    boundary = sample(curves, np.tile(np.array([125.81])[:, None] / 128, (1, 6)))[0]
    return curves, objective, boundary, q


def test_independent_values_covariance_and_all168_FD(fixture):
    curves, objective, boundary, q = fixture
    expected = independent(curves, objective.canonical_constants["normalization"], boundary, q)
    x = torch.tensor(q, requires_grad=True)
    previous = torch.cat((torch.tensor(boundary[None]), x[:3]))
    values = objective.evaluate_pairs(previous, x)
    actual = np.array([float(values[key].detach()) for key in subject.TERMS])
    np.testing.assert_allclose(actual, expected["terms"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(values["per_leg_successor_loss"].detach(), expected["per_leg"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(values["coordination_pair_loss"].detach(), expected["pair"], rtol=1e-10, atol=1e-10)
    for side in ("previous", "next"):
        for key in ("weights", "target", "z"):
            np.testing.assert_allclose(values[side + "_encoding"][key].detach(), expected[side][key], rtol=1e-10, atol=1e-10)
    g = torch.autograd.grad(values["training_loss"], x, retain_graph=True)[0].numpy()
    scaled = torch.autograd.grad(values["training_loss"] / 50, x)[0].numpy()
    np.testing.assert_allclose(g, expected["g"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(scaled, expected["g"] / 50, rtol=1e-8, atol=2e-10)
    np.testing.assert_allclose(scaled, g / 50, rtol=1e-12, atol=1e-12)
    fd = np.empty_like(q)
    for i in range(q.size):
        plus, minus = q.copy(), q.copy()
        plus.flat[i] += 1e-6
        minus.flat[i] -= 1e-6
        fd.flat[i] = (independent(curves, objective.canonical_constants["normalization"], boundary, plus)["terms"][0]
                      - independent(curves, objective.canonical_constants["normalization"], boundary, minus)["terms"][0]) / 2e-6
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(scaled, fd / 50, rtol=1e-6, atol=2e-9)


def test_live_previous_and_component_coordinate_closures(fixture):
    _, objective, boundary, q = fixture
    dq = np.random.default_rng(333).normal(size=q.shape) * 1e-6
    row, raw = subject.analyze_window(objective, boundary, {"neutral": q, "plus": q+dq, "minus": q-dq})
    assert row["Q"] == pytest.approx(np.sum(raw["neutral_gq"] * raw["delta_q"]), abs=1e-12)
    assert row["term_Q"][0] == pytest.approx(sum(row["term_Q"][1:3]), abs=1e-12)
    assert np.mean(row["successor_leg_Q"]) == pytest.approx(row["term_Q"][1], abs=1e-12)
    assert np.mean(row["coordination_pair_Q"]) == pytest.approx(row["term_Q"][2], abs=1e-12)
    assert sum(row["Q_coordinate_leg"]) == pytest.approx(row["Q"], abs=1e-12)
    x = torch.tensor(q, requires_grad=True)
    detached = objective.evaluate_pairs(torch.tensor(np.vstack((boundary, q[:3]))), x)
    held_g = torch.autograd.grad(detached["training_loss"] / 50, x)[0].numpy()
    assert np.max(np.abs(held_g[:3] - raw["neutral_gq"][:3])) > 1e-5
    np.testing.assert_allclose(held_g[3], raw["neutral_gq"][3], rtol=1e-12, atol=1e-12)
    assert "T" not in raw and "T" not in row


def test_symmetric_center_zero_resultant_stationary_counterexample():
    phase = 2 * np.pi * (np.arange(128, dtype=np.float64) / 128)
    curves = np.zeros((6, 128, 7))
    curves[:, :, 0] = np.cos(phase)
    curves[:, :, 1] = np.sin(phase)
    objective = SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    q = torch.zeros((4, 42), dtype=torch.float64, requires_grad=True)
    result = objective.evaluate_pairs(torch.cat((torch.zeros((1, 42)), q[:3])), q)
    assert result["successor_loss"].item() == pytest.approx(0, abs=1e-25)
    assert result["coordination_loss"].item() == pytest.approx(1, abs=1e-14)
    g = torch.autograd.grad(result["training_loss"], q)[0]
    np.testing.assert_allclose(g.numpy() / 50, np.zeros((4, 42)), rtol=1e-8, atol=2e-10)
    assert not result["next_encoding"]["R"].requires_grad
    assert result["next_encoding"]["R"].max().item() < 1e-13
    assert result["next_encoding"]["entropy"].mean().item() == pytest.approx(np.log(128), abs=1e-12)


def test_canonical_constants_approved_bytes_and_mismatch_before_transfer(fixture):
    curves, objective, _, _ = fixture
    canonical = objective.canonical_constants
    norm = canonical["normalization"].copy()
    # A legal tiny difference is approved input and must remain byte-exact.
    norm[0] = np.nextafter(norm[0], np.inf)
    approved = SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)), normalization=norm)
    assert subject.bytes_sha(approved.successor_normalization.numpy()) == subject.bytes_sha(norm)
    assert subject.bytes_sha(approved.temperature.numpy()) == subject.bytes_sha(7 * norm)
    for name, attribute in (("curves", "curves"), ("normalization", "successor_normalization"),
                            ("temperature", "temperature"), ("future_anchors", "future_anchors"), ("phase_basis", "phase_basis")):
        actual = getattr(approved, attribute).numpy()
        assert actual.flags.c_contiguous and actual.dtype.str == "<f8"
        assert subject.bytes_sha(actual) == approved.constant_metadata[name]["c_order_sha256"]
    with pytest.raises(ValueError, match="normalization differs"):
        SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)), normalization=norm * 1.1)
    for kwargs in ({"frequency": 1}, {"dt": .006}, {"frequency": True}, {"device": "meta"}):
        with pytest.raises(ValueError):
            SmoothTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)), **kwargs)


def test_competing_loops_adjacent_vertex_seam_underflow_finite(fixture):
    curves, _, _, _ = fixture
    # Two lobes share crossings. Discrete diagnostics are never training inputs.
    phi = 2 * np.pi * np.arange(128, dtype=np.float64) / 128
    loops = curves.copy()
    loops[:, :, 0] = .2 * np.sin(phi)
    loops[:, :, 1] = .15 * np.sin(2 * phi)
    objective = SmoothTemporalObjective(loops, np.zeros((99, 42)), np.zeros((99, 6)))
    q = sample(loops, np.tile(np.array([-1e-18, .9999, 1/128-.00001, 1/128+.00001])[:, None], (1, 6)))
    x = torch.tensor(q, requires_grad=True)
    value = objective.evaluate_pairs(torch.cat((x[:1], x[:3])), x)
    assert torch.isfinite(torch.autograd.grad(value["training_loss"], x)[0]).all()
    assert torch.isfinite(value["next_encoding"]["weights"]).all()
    enc = objective.encode(torch.full((1, 42), 10., dtype=torch.float64))
    assert enc["zero_weight_count"].sum().item() > 0
    assert torch.isfinite(enc["entropy"]).all()
    np.testing.assert_allclose(enc["weights"].sum(-1), 1, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e308])
def test_nonfinite_and_finite_input_overflow_rejected(fixture, bad):
    _, objective, boundary, q = fixture
    x = torch.tensor(q)
    x[0, 0] = bad
    with pytest.raises((RuntimeError, ValueError), match="finite"):
        objective.evaluate_pairs(torch.tensor(np.vstack((boundary, q[:3]))), x)


def test_shapes_dtype_and_zero_normalization_rejected(fixture):
    curves, objective, _, _ = fixture
    with pytest.raises(TypeError):
        objective.encode(torch.zeros((4, 42), dtype=torch.float32))
    with pytest.raises(ValueError):
        objective.encode(torch.zeros((4, 41), dtype=torch.float64))
    with pytest.raises(ValueError):
        objective.evaluate_pairs(torch.zeros((3, 42), dtype=torch.float64), torch.zeros((4, 42), dtype=torch.float64))
    with pytest.raises(ValueError):
        SmoothTemporalObjective(curves, np.zeros((98, 42)), np.zeros((99, 6)))
    with pytest.raises(ValueError):
        SmoothTemporalObjective(np.zeros((6, 128, 7)), np.zeros((99, 42)), np.zeros((99, 6)))


def test_startup99_guards_and_steady200(fixture):
    _, objective, _, q = fixture
    commands = torch.zeros((300, 42), dtype=torch.float64)
    commands[99:] = torch.tensor(np.tile(q, (51, 1))[:201])
    tendon = torch.zeros((300, 6), dtype=torch.float64)
    result = objective.evaluate(commands, tendon)
    assert result["startup_pass"].item() and result["successor_terms"].shape == (200, 6)
    expected = objective.evaluate_pairs(commands[99:299], commands[100:300])
    assert result["training_loss"].item() == expected["training_loss"].item()
    commands[98, 0] = .0401
    assert not objective.evaluate(commands, tendon)["startup_pass"].item()
    commands[98, 0] = 0
    tendon[0, 0] = .0251
    assert not objective.evaluate(commands, tendon)["startup_pass"].item()


def test_reference_progress_static_minimum_and_fourrow_identity(fixture):
    _, objective, _, _ = fixture
    summary, raw = subject.reference_cases(objective)
    scores = raw["known_static_training_loss"]
    assert scores.shape == (129,) and summary["known_static_count"] == 129
    assert summary["known_static_minimum"] == np.min(scores)
    assert summary["static_margin"] == pytest.approx(np.min(scores) - raw["forward_terms"][0], abs=1e-12)
    assert summary["known_static_discrimination"] == (summary["static_margin"] > 1e-8)
    assert np.mean(scores) > np.min(scores) + 1e-8
    point = raw["known_static_q"][summary["known_static_minimum_index"]]
    singleton = objective.evaluate_pairs(torch.tensor(point[None]), torch.tensor(point[None]))
    repeated = torch.tensor(np.repeat(point[None], 4, axis=0))
    assert singleton["training_loss"].item() == pytest.approx(objective.evaluate_pairs(repeated, repeated)["training_loss"].item(), abs=1e-12)
    blur = ((raw["forward_previous_target"] - raw["forward_q"]).reshape(128, 6, 7)**2).mean(-1) / objective.canonical_constants["normalization"]
    np.testing.assert_allclose(blur, raw["forward_target_blur_normalized_mse"], rtol=1e-12, atol=1e-12)


def rows_from(o, q):
    return [{"start_control": 100 + 4*i, "O": a, "Q": b, "O_minus_Q": a-b} for i, (a, b) in enumerate(zip(o, q, strict=True))]


def test_signed_and_L1_reject_cancellation_zero_information_and_small_signal():
    cancelled = subject.aggregate(rows_from([1., 1.], [101., -99.]))
    assert cancelled["e_signed"] == 0 and cancelled["e_L1"] == 1
    assert not cancelled["finite_command_qualification"]
    cancelling_zero = subject.aggregate(rows_from([1., -1.], [1., -1.]))
    assert cancelling_zero["e_signed"] is None and cancelling_zero["e_L1"] == 0
    assert cancelling_zero["reason"] == "no_direction_information"
    zero = subject.aggregate(rows_from([0.], [0.]))
    assert zero["e_signed"] is zero["e_L1"] is None and zero["absolute_closure_valid"]
    assert not zero["finite_command_qualification"]
    small = subject.aggregate(rows_from([1e-250], [1e-250]))
    assert small["finite_command_qualification"] and small["D_signed"] == 1e-250
    with pytest.raises(ValueError):
        subject.aggregate(rows_from([np.inf], [0.]))


def test_gate_rejects_before_Torch_import_or_any_payload(tmp_path, monkeypatch):
    source = tmp_path / "generated.py"
    source.write_text("# changed", encoding="utf8")
    contract = {"schema": subject.SCHEMA, "evidence": subject.EVIDENCE,
                "source_sha256": {"generated.py": "0"*64}, "input_sha256": {"fake.npz": "0"*64},
                "limits": subject.LIMITS, "calculation": subject.CALCULATION,
                "window_starts": list(subject.STARTS), "original_calls": 0, "candidate_adopted": False}
    path = tmp_path / "contract.json"
    subject.write_json(path, contract)
    touched = []
    original = builtins.__import__
    def forbidden(name, *args, **kwargs):
        if name == "torch":
            touched.append("torch")
            raise AssertionError("Torch before gate")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbidden)
    monkeypatch.setattr(subject.np, "load", lambda *args, **kwargs: touched.append("payload"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        subject.gate(path, subject.sha256(path), tmp_path / "output", root=tmp_path)
    assert touched == []


def test_key_only_loader_clock_and_unused_object_payload(tmp_path):
    path = tmp_path / "small.npz"
    data = {"command": np.zeros((301, 42)), "control_tick": np.arange(301),
            "network_tick": 20*np.arange(301), "physics_time_s": .005*np.arange(301)}
    np.savez(path, **data, rates=np.array([object()], dtype=object))
    loaded = subject.load_npz(path, tuple(data))
    assert subject._clock_commands(loaded, "generated").shape == (301, 42)
    loaded["control_tick"][0] = 1
    with pytest.raises(ValueError, match="clock"):
        subject._clock_commands(loaded, "generated")


def fake_contract(tmp_path):
    pins = {}
    for role in ("result", "ledger"):
        path = tmp_path / subject.EVIDENCE[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf8")
        pins[subject.EVIDENCE[role]] = subject.sha256(path)
    source = tmp_path / "generated.py"
    source.write_text("# generated", encoding="utf8")
    contract = {"source_sha256": {"generated.py": subject.sha256(source)}, "input_sha256": pins}
    path = tmp_path / "contract.json"
    subject.write_json(path, contract)
    return path, contract


def completed_compute(contract, evidence, rows, save, timely):
    rows.extend(rows_from([1.]*50, [1.]*50))
    return {"reference_progress_pass": True, "known_static_discrimination": True}


def generated_run(tmp_path, compute=completed_compute, **kwargs):
    path, contract = fake_contract(tmp_path)
    return subject.run(path, subject.sha256(path), tmp_path / "output", root=tmp_path,
                       gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute, **kwargs)


def test_analysis_valid_distinct_from_input_qualification(tmp_path):
    def rejected(*args):
        result = completed_compute(*args)
        result["known_static_discrimination"] = False
        return result
    result = generated_run(tmp_path, rejected)
    assert result["analysis_valid"] and not result["input_qualification_pass"]
    assert not result["candidate_adopted"] and result["E330_local_direction"] == "refuted"
    assert subject.read(tmp_path / "output/result.json") == result


def test_partial_raw_first_error_and_publication_failure_preserved(tmp_path):
    def partial(_, __, rows, save, timely):
        save("window-100.npz", {"neutral_gq": np.zeros((4, 42))})
        rows.extend(rows_from([1.], [1.]))
        raise ValueError("first calculation failure")
    def writer(path, value):
        if path.name == "result.json":
            raise OSError("secondary publication failure")
        subject.write_json(path, value)
    result = generated_run(tmp_path, partial, writer=writer)
    assert result["first_error"]["message"] == "first calculation failure" and result["completed_windows"] == 1
    assert not result["analysis_valid"] and not result["input_qualification_pass"]
    assert (tmp_path / "output/window-100.npz").exists()
    assert [error["stage"] for error in result["errors"]] == ["analysis", "publication"]
    assert subject.read(tmp_path / "output/result.json") == result


def test_partial_save_failure_retains_bytes_and_attempt(tmp_path):
    def compute(_, __, rows, save, timely):
        save("window-100.npz", {"neutral_gq": np.zeros((4, 42))})
        rows.extend(rows_from([1.], [1.]))
    def saver(path, **values):
        Path(path).write_bytes(b"partial")
        raise OSError("raw save failed")
    result = generated_run(tmp_path, compute, saver=saver)
    assert result["completed_windows"] == 0 and result["first_error"]["message"] == "raw save failed"
    assert (tmp_path / "output/window-100.npz").read_bytes() == b"partial"
    assert not subject.read(tmp_path / "output/advance-ledger.json")["attempts"][0]["returned"]


def test_posthash_failure_preserves_completed50(tmp_path):
    def compute(*args):
        result = completed_compute(*args)
        (tmp_path / "generated.py").write_text("changed", encoding="utf8")
        return result
    result = generated_run(tmp_path, compute)
    assert not result["analysis_valid"] and "SHA256 mismatch" in result["first_error"]["message"]
    assert result["completed_windows"] == 50 and subject.read(tmp_path / "output/result.json") == result


def test_committed_success_repaired_after_deadline_and_cost_coherent(tmp_path):
    elapsed, committed = [0.0], []
    def writer(path, value):
        subject.write_json(path, value)
        if path.name == "result.json":
            committed.append(value["analysis_valid"])
            elapsed[0] = 111.
    result = generated_run(tmp_path, writer=writer, clock=lambda: elapsed[0])
    assert committed == [True] and not result["analysis_valid"] and not result["input_qualification_pass"]
    disk = subject.read(tmp_path / "output/result.json")
    ledger = subject.read(tmp_path / "output/advance-ledger.json")
    assert disk == result
    for key in ("analysis_valid", "input_qualification_pass", "first_error", "errors", "elapsed_s", "completed_windows"):
        assert ledger[key] == result[key]
    assert result["elapsed_s"] == 111.


def test_first_error_survives_emergency_failure_and_all_sinks(tmp_path, monkeypatch):
    original_writer, elapsed = subject.write_json, [0.]
    def compute(*args):
        elapsed[0] = 2.
        raise ValueError("primary computation failure")
    def writer(path, value):
        if path.name == "result.json":
            elapsed[0] = 3.
            raise OSError("result sink failed")
        original_writer(path, value)
    def emergency(path, value):
        if path.name == "publication-failure.json":
            elapsed[0] = 4.
            raise OSError("emergency sink failed")
        original_writer(path, value)
    monkeypatch.setattr(subject, "write_json", emergency)
    result = generated_run(tmp_path, compute, writer=writer, clock=lambda: elapsed[0])
    assert result["first_error"]["message"] == "primary computation failure"
    assert [error["stage"] for error in result["errors"]] == ["analysis", "publication", "emergency_publication"]
    assert subject.read(tmp_path / "output/result.json") == result and result["elapsed_s"] == 4.


def test_no_output_sink_still_returns_original_first_error(tmp_path, monkeypatch):
    path, contract = fake_contract(tmp_path)
    def compute(*args):
        raise ValueError("primary computation failure")
    def failed(*args):
        raise OSError("all sinks unavailable")
    monkeypatch.setattr(subject, "write_json", failed)
    monkeypatch.setattr(subject, "_replace_terminal_json", failed)
    result = subject.run(path, subject.sha256(path), tmp_path / "output", root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute, writer=failed)
    assert result["first_error"]["message"] == "primary computation failure"
    assert [error["stage"] for error in result["errors"]] == ["analysis", "publication", "emergency_publication",
        "terminal_repair_advance-ledger.json", "terminal_repair_result.json"]
    assert not result["analysis_valid"] and not result["input_qualification_pass"]
