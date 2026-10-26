"""E331 generated CPU qualification only; never opens original saved payloads."""
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import analyze_c_banc_m3_temporal_residual as subject
from tools.banc_m3_phase_free_temporal import PhaseFreeTemporalObjective

torch.set_num_threads(1)


@pytest.fixture
def fixture():
    phase = np.arange(128, dtype=np.float64) * 2 * np.pi / 128
    # General seven-dimensional, nonplanar closed curve with distinct edges.
    base = np.stack((np.cos(phase), np.sin(phase), .17 * np.cos(2 * phase),
                     .13 * np.sin(3 * phase), .09 * np.cos(5 * phase),
                     .07 * np.sin(4 * phase), .05 * np.cos(7 * phase)), axis=-1)
    curves = np.stack([base * (1 + .07 * leg) + .03 * leg for leg in range(6)])
    objective = PhaseFreeTemporalObjective(curves, np.zeros((99, 42)), np.zeros((99, 6)))
    def command(coordinates):
        x = np.asarray(coordinates, np.float64)
        x = np.broadcast_to(x[:, None], (len(x), 6)) + np.arange(6)[None] * .71
        floor = np.floor(x)
        index = floor.astype(np.int64) % 128
        leg = np.arange(6)[None]
        return (curves[leg, index] + (x - floor)[..., None]
                * (curves[leg, (index + 1) % 128] - curves[leg, index])).reshape(-1, 42)
    boundary = command(np.array([19.37]))[0]
    q = command(np.array([20.42, 21.37, 22.44, 23.31]))
    return curves, objective, command, boundary, q


def numpy_loss(curves, norm, boundary, q, fixed=None):
    """Test-owned direct NumPy loop; independent of production projection/sample."""
    def project(commands, fixed_index):
        result = np.empty((4, 6))
        for t in range(4):
            for leg in range(6):
                x = commands[t, leg * 7:(leg + 1) * 7]
                edges = np.roll(curves[leg], -1, axis=0) - curves[leg]
                u = np.clip(np.sum((x - curves[leg]) * edges, axis=1) / np.sum(edges ** 2, axis=1), 0, 1)
                distance = np.sum((x - curves[leg] - u[:, None] * edges) ** 2, axis=1)
                index = int(distance.argmin()) if fixed_index is None else fixed_index[t, leg]
                result[t, leg] = ((index + u[index]) / 128) % 1
        return result
    previous = np.vstack((boundary, q[:3]))
    pp = project(previous, None if fixed is None else fixed[0])
    pn = project(q, None if fixed is None else fixed[1])
    target = np.empty((4, 42))
    for t in range(4):
        for leg in range(6):
            x = ((pp[t, leg] + 2.382760007990865 * .005) % 1) * 128
            floor = np.floor(x)
            index = int(floor) % 128
            target[t, leg * 7:(leg + 1) * 7] = curves[leg, index] + (x - floor) * (curves[leg, (index + 1) % 128] - curves[leg, index])
    successor = np.mean(np.mean((q - target).reshape(4, 6, 7) ** 2, axis=(0, 2)) / norm)
    coordination = np.mean([1 - np.cos(2 * np.pi * (pn[:, a] - pn[:, b])) for a, b in subject.PAIRS])
    return successor + coordination


def test_general_7d_values_fixed_neutral_and_live_gradient(fixture):
    curves, objective, _, boundary, q = fixture
    norm = objective.successor_normalization.numpy()
    independent = subject.numpy_terms(curves, norm, boundary, q)
    fixed = (independent["previous"]["segment"], independent["next"]["segment"])
    arms = {"neutral": q, "plus": q.copy(), "minus": q.copy()}
    row, raw = subject.analyze_window(objective, boundary, arms, np.zeros(3), np.zeros(3))
    expected = numpy_loss(curves, norm, boundary, q)
    assert raw["neutral_terms"][0] == pytest.approx(expected, abs=1e-10)
    assert subject.numpy_terms(curves, norm, boundary, q, fixed)["terms"][0] == pytest.approx(expected, abs=1e-10)
    rng = np.random.default_rng(331)
    direction = rng.normal(size=q.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    live_fd = (numpy_loss(curves, norm, boundary, q + eps * direction)
               - numpy_loss(curves, norm, boundary, q - eps * direction)) / (2 * eps * 50)
    fixed_fd = (numpy_loss(curves, norm, boundary, q + eps * direction, fixed)
                - numpy_loss(curves, norm, boundary, q - eps * direction, fixed)) / (2 * eps * 50)
    dot = np.sum(raw["neutral_gq"] * direction)
    assert dot == pytest.approx(live_fd, abs=2e-8, rel=1e-6)
    assert dot == pytest.approx(fixed_fd, abs=2e-8, rel=1e-6)
    assert np.array_equal(raw["boundary"], boundary)
    assert row["Q"] == row["O"] == row["F"] == row["T"] == 0
    # Detaching the three live previous commands must change this derivative.
    held = torch.tensor(np.vstack((boundary, q[:3])), dtype=torch.float64)
    x = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    detached_gradient = torch.autograd.grad(objective.evaluate_pairs(held, x)["training_loss"] / 50, x)[0].numpy()
    assert np.max(np.abs(detached_gradient - raw["neutral_gq"])) > 1e-4


def test_adjacent_vertex_counterexample_is_not_nonlocal(fixture):
    curves, objective, command, boundary, _ = fixture
    q = command(np.array([31.80, 34.37, 36.44, 39.31]))
    plus, minus = q.copy(), q.copy()
    plus[0] = command(np.array([32.20]))[0]
    minus[0] = command(np.array([31.60]))[0]
    row, _ = subject.analyze_window(objective, boundary, {"neutral": q, "plus": plus, "minus": minus}, np.zeros(3), np.zeros(3))
    assert abs(row["O_minus_F"]) > 1e-7
    assert sum(item["nonlocal_endpoint_change"] for item in row["branch_changes"].values()) == 0
    assert row["branch_changes"]["plus_next"]["adjacent_change"] > 0
    norm = objective.successor_normalization.numpy()
    assert row["O"] == pytest.approx((numpy_loss(curves, norm, boundary, plus) - numpy_loss(curves, norm, boundary, minus)) / 50, abs=1e-10)


def test_wrap_nonlocal_and_strict_exact_clamp_semantics(fixture):
    curves, _, _, _, _ = fixture
    edge = curves[:, 1] - curves[:, 0]
    q = np.stack([(curves[:, 0] + x * edge).reshape(42) for x in (-.1, 0, .5, 1, 1.1)])
    projection = subject.numpy_project(curves, q, np.zeros((5, 6), np.int64))
    geo = subject.geometry(projection)
    assert np.array_equal(geo["clamp_regime"][:, 0], [-2, -1, 0, 1, 2])
    def fabricated(segment, phase, regime=0):
        return {"segment": np.full((1, 6), segment), "phase": np.full((1, 6), phase), "clamp_regime": np.full((1, 6), regime)}
    seam = subject.branch_changes(fabricated(127, .999), fabricated(0, .001))
    assert seam["phase_wrap"].all() and seam["adjacent_change"].all()
    assert not seam["nonlocal_endpoint_change"].any()
    assert subject.branch_changes(fabricated(5, .1), fabricated(65, .6))["nonlocal_endpoint_change"].all()
    assert not subject.branch_changes(fabricated(5, .1), fabricated(65, .101))["nonlocal_endpoint_change"].any()


def test_first_argmin_tie_endpoint_and_target_knot(fixture):
    curves, _, _, _, _ = fixture
    q = curves[:, 1].reshape(1, 42).copy()
    projection = subject.numpy_project(curves, q)
    assert np.array_equal(projection["segment"], np.zeros((1, 6), np.int64))
    assert subject.geometry(projection)["any_tie"].all()
    assert np.array_equal(subject.geometry(projection)["clamp_regime"], np.ones((1, 6), np.int8))
    synthetic = {key: value.copy() for key, value in projection.items()}
    synthetic["phase"][:] = .25 - 2.382760007990865 * .005
    assert subject.geometry(synthetic)["target_knot"].all()


def test_no_nonlocal_competitor_has_null_available_and_zero_shares(fixture):
    curves, _, _, _, _ = fixture
    projection = subject.numpy_project(curves, curves[:, 0].reshape(1, 42).copy())
    projection["all_segment_phase"][:] = projection["phase"][..., None]
    assert np.isinf(subject.geometry(projection)["nonlocal_gap"]).all()
    row = {"start_control": 100, "physics_time_s": .5, **subject.residuals(0, 0, 0, 0)}
    assert all(item["share_of_absolute_total_residual"] is None for item in subject.aggregate([row])["terms"].values())


@pytest.mark.parametrize("field", ["q", "boundary", "gtheta", "delta_theta"])
def test_nonfinite_rejected(fixture, field):
    _, objective, _, boundary, q = fixture
    values = {"q": q.copy(), "boundary": boundary.copy(), "gtheta": np.ones(3), "delta_theta": np.ones(3)}
    values[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        subject.analyze_window(objective, values["boundary"], {label: values["q"] for label in ("neutral", "plus", "minus")}, values["gtheta"], values["delta_theta"])


def test_scaling_and_both_closures_with_nonzero_T(fixture):
    _, objective, _, boundary, q = fixture
    rng = np.random.default_rng(332)
    dq = rng.normal(size=q.shape) * 1e-6
    gradient = np.array([.1, -.2, .3], np.float64)
    delta = np.array([.01, .02, -.01], np.float64)
    row, raw = subject.analyze_window(objective, boundary, {"neutral": q, "plus": q + dq, "minus": q - dq}, gradient, delta)
    assert row["T"] == pytest.approx(np.dot(gradient, delta), abs=1e-15)
    assert row["Q"] == pytest.approx(np.sum(raw["neutral_gq"] * (2 * dq)), abs=1e-12)
    assert sum(row["Q_coordinate_leg"]) == pytest.approx(row["Q"], abs=1e-12)
    assert row["two_term_closure_error"] <= 1e-12
    assert row["three_term_closure_error"] <= 1e-12
    assert not subject.INTERPRETATION["closure_proves_correctness"]
    with pytest.raises(ValueError, match="finite"):
        subject.residuals(np.inf, 0, 0, 0)


def test_SHA_gate_precedes_any_payload_load(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_text("changed", encoding="utf8")
    contract = {"schema": subject.SCHEMA, "evidence": subject.EVIDENCE, "source_sha256": {"source.py": "0" * 64},
                "input_sha256": {"fake.npz": "0" * 64}, "limits": subject.LIMITS,
                "window_starts": list(subject.STARTS), "original_calls": 0, "candidate_adopted": False}
    contract_path = tmp_path / "contract.json"
    subject.write_json(contract_path, contract)
    touched = []
    monkeypatch.setattr(subject.np, "load", lambda *args, **kwargs: touched.append(args))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        subject.gate(contract_path, subject.sha256(contract_path), tmp_path / "output", root=tmp_path)
    assert not touched


def test_key_only_loader_does_not_read_unused_payload(tmp_path):
    path = tmp_path / "small.npz"
    np.savez(path, q=np.zeros((4, 42)), rate=np.array([object()], dtype=object))
    assert set(subject.load_npz(path, ("q",))) == {"q"}


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
    contract_path = tmp_path / "generated-contract.json"
    subject.write_json(contract_path, contract)
    return contract_path, contract


def test_partial_raw_first_error_and_publication_failure_preserved(tmp_path):
    contract_path, contract = fake_contract(tmp_path)
    def compute(_, __, rows, save, timely):
        save("window-100.npz", {"neutral_gq": np.zeros((4, 42))})
        rows.append({"start_control": 100})
        raise ValueError("first calculation failure")
    def writer(path, value):
        if Path(path).name == "result.json":
            raise OSError("secondary publication failure")
        subject.write_json(path, value)
    output = tmp_path / "output"
    result = subject.run(contract_path, subject.sha256(contract_path), output, root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute, writer=writer)
    assert not result["analysis_valid"]
    assert result["first_error"]["message"] == "first calculation failure"
    assert result["completed_windows"] == 1
    assert (output / "window-100.npz").is_file()
    failure = subject.read(output / "publication-failure.json")
    assert failure["first_error"]["message"] == "first calculation failure"
    assert [row["stage"] for row in failure["errors"]] == ["analysis", "publication"]


def test_post_hash_failure_and_finalization_deadline_preserve_invalid_result(tmp_path):
    contract_path, contract = fake_contract(tmp_path)
    def compute(_, __, rows, save, timely):
        for c in subject.STARTS:
            rows.append({"start_control": c, "physics_time_s": c * .005, **subject.residuals(0, 0, 0, 0)})
        (tmp_path / "generated.py").write_text("mutated", encoding="utf8")
    output = tmp_path / "post-failure"
    result = subject.run(contract_path, subject.sha256(contract_path), output, root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute)
    assert not result["analysis_valid"] and "SHA256 mismatch" in result["first_error"]["message"]
    assert subject.read(output / "result.json")["completed_windows"] == 50


def test_partial_save_failure_does_not_invent_completion(tmp_path):
    contract_path, contract = fake_contract(tmp_path)
    def compute(_, __, rows, save, timely):
        save("window-100.npz", {"neutral_gq": np.zeros((4, 42))})
        rows.append({"start_control": 100})
    def saver(path, **values):
        Path(path).write_bytes(b"partial")
        raise OSError("raw save failed")
    output = tmp_path / "save-failure"
    result = subject.run(contract_path, subject.sha256(contract_path), output, root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute, saver=saver)
    assert not result["analysis_valid"] and result["completed_windows"] == 0
    assert result["first_error"]["message"] == "raw save failed"
    assert (output / "window-100.npz").read_bytes() == b"partial"
    assert subject.read(output / "advance-ledger.json")["attempts"][0]["returned"] is False


def test_committed_success_is_repaired_after_terminal_deadline(tmp_path):
    contract_path, contract = fake_contract(tmp_path)
    elapsed = [0.0]
    committed = []
    def compute(_, __, rows, save, timely):
        for c in subject.STARTS:
            rows.append({"start_control": c, "physics_time_s": c * .005, **subject.residuals(0, 0, 0, 0)})
    def writer(path, value):
        subject.write_json(path, value)
        if path.name == "result.json":
            committed.append(subject.read(path)["analysis_valid"])
            elapsed[0] = 111.0
    output = tmp_path / "terminal-deadline"
    result = subject.run(contract_path, subject.sha256(contract_path), output, root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute,
                         writer=writer, clock=lambda: elapsed[0])
    assert committed == [True]
    assert result["analysis_valid"] is False
    disk = subject.read(output / "result.json")
    ledger = subject.read(output / "advance-ledger.json")
    assert disk["analysis_valid"] is ledger["analysis_valid"] is False
    assert disk["errors"] == ledger["errors"] == result["errors"]
    assert disk["first_error"] == ledger["first_error"] == result["first_error"]
    assert disk["elapsed_s"] == ledger["elapsed_s"] == result["elapsed_s"] == 111.0
    assert disk["completed_windows"] == ledger["completed_windows"] == 50
    assert result["first_error"]["stage"] == "publication"


def test_first_error_survives_failed_emergency_journal_and_terminal_cost_is_coherent(tmp_path, monkeypatch):
    contract_path, contract = fake_contract(tmp_path)
    original_writer = subject.write_json
    elapsed = [0.0]
    def compute(*args, **kwargs):
        elapsed[0] = 2.0
        raise ValueError("first calculation failure")
    def writer(path, value):
        if path.name == "result.json":
            elapsed[0] = 3.0
            raise OSError("second result-publication failure")
        original_writer(path, value)
    def emergency(path, value):
        if path.name == "publication-failure.json":
            elapsed[0] = 4.0
            raise OSError("third emergency-disk failure")
        original_writer(path, value)
    monkeypatch.setattr(subject, "write_json", emergency)
    output = tmp_path / "emergency-unavailable"
    result = subject.run(contract_path, subject.sha256(contract_path), output, root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute,
                         writer=writer, clock=lambda: elapsed[0])
    assert result["analysis_valid"] is False
    assert result["first_error"]["message"] == "first calculation failure"
    assert [row["message"] for row in result["errors"]] == [
        "first calculation failure", "second result-publication failure", "third emergency-disk failure"]
    assert [row["stage"] for row in result["errors"]] == ["analysis", "publication", "emergency_publication"]
    disk = subject.read(output / "result.json")
    ledger = subject.read(output / "advance-ledger.json")
    assert disk["first_error"] == ledger["first_error"] == result["first_error"]
    assert disk["errors"] == ledger["errors"] == result["errors"]
    assert disk["elapsed_s"] == ledger["elapsed_s"] == result["elapsed_s"] == 4.0
    assert disk["analysis_valid"] is ledger["analysis_valid"] is False
    assert not (output / "publication-failure.json").exists()


def test_all_recovery_sinks_unavailable_still_returns_first_failure(tmp_path, monkeypatch):
    contract_path, contract = fake_contract(tmp_path)
    def compute(*args, **kwargs):
        raise ValueError("primary computation failure")
    def unavailable(*args, **kwargs):
        raise OSError("all output sinks unavailable")
    monkeypatch.setattr(subject, "write_json", unavailable)
    monkeypatch.setattr(subject, "_replace_terminal_json", unavailable)
    result = subject.run(contract_path, subject.sha256(contract_path), tmp_path / "all-unavailable", root=tmp_path,
                         gate_fn=lambda *args, **kwargs: (contract, {}), compute_fn=compute, writer=unavailable)
    assert result["analysis_valid"] is False
    assert result["first_error"]["message"] == "primary computation failure"
    assert [row["stage"] for row in result["errors"]] == [
        "analysis", "publication", "emergency_publication", "terminal_repair_advance-ledger.json", "terminal_repair_result.json"]
    assert result["completed_windows"] == 0 and result["elapsed_s"] >= 0
