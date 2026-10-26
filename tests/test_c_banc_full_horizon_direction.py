"""Generated CPU checks for E322 orchestration; no native files or model calls."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_full_horizon_direction as e


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    began = time.perf_counter()
    yield
    elapsed = time.perf_counter() - began
    torch.set_num_threads(old)
    print("E322_GENERATED_COST", json.dumps({
        "wall_s": elapsed, "CPU_threads": 1, "original_calls": 0,
        "model_calls": 0, "body_calls": 0, "CUDA_calls": 0, "fit_calls": 0}))
    assert elapsed <= 100


class FakeReceipt:
    def __init__(self, backward=False):
        self.status = "forward_returned"
        self.forward_controls_requested = self.forward_controls_returned = 300
        self.forward_neural_ticks_requested = self.forward_neural_ticks_returned = 6000
        self.recompute_controls_requested = self.recompute_controls_returned = 0
        self.recompute_neural_ticks_requested = self.recompute_neural_ticks_returned = 0
        self.backward_blocks_requested = self.backward_blocks_returned = 0
        self.backward_block_ticks_returned = 0
        self.confirm_requested = self.confirm_returned = 0
        self.confirmed = None
        self.backward_expected = backward

    def backward(self, gradient):
        self.status = "backward_returned"
        self.recompute_controls_requested = self.recompute_controls_returned = 300
        self.recompute_neural_ticks_requested = self.recompute_neural_ticks_returned = 6000
        self.backward_blocks_requested = self.backward_blocks_returned = 300
        self.backward_block_ticks_returned = 6000
        return gradient

    def confirm(self):
        self.confirm_requested += 1
        self.confirmed = {
            "forward_controls": self.forward_controls_returned,
            "forward_neural_ticks": self.forward_neural_ticks_returned,
            "recompute_controls": self.recompute_controls_returned,
            "recompute_neural_ticks": self.recompute_neural_ticks_returned,
            "backward_blocks": self.backward_blocks_returned,
            "backward_block_ticks": self.backward_block_ticks_returned,
        }
        self.confirm_returned += 1
        return self.confirmed


class FakeRollout:
    device = torch.device("cpu")

    def __init__(self, fault_at=None):
        self.calls = []
        self.fault_at = fault_at

    def run(self, r, z, bank, original, drives, efficacy, mask):
        self.calls.append((efficacy.detach().clone(), r.clone(), z.clone(),
                           bank.clone(), original.clone(), drives.clone()))
        receipt = FakeReceipt(efficacy.requires_grad)
        if self.fault_at == len(self.calls):
            receipt.status = "forward_failed"
            receipt.forward_controls_requested = 1
            receipt.forward_controls_returned = 0
            receipt.forward_neural_ticks_requested = 1
            receipt.forward_neural_ticks_returned = 0
            error = RuntimeError("generated direction fault")
            error.receipt = receipt
            raise error
        signal = (efficacy - 1).sum() * .0001
        q = torch.full((300, 42), .1, dtype=torch.float64) + signal
        if efficacy.requires_grad:
            q.register_hook(receipt.backward)
        return {"q": q, "tendon": torch.full((300, 6), .02, dtype=torch.float64),
                "rate": r.clone(), "adaptation": z.clone(), "bank": bank.clone(),
                "original_activation": original.clone(), "receipt": receipt}


class FakeObjective:
    def evaluate(self, q, tendon):
        score = q.mean()
        return {"shape_loss": score, "per_leg_shape_loss": score.expand(6),
                "startup_q_rms": score * 0, "startup_q_max": score * 0,
                "startup_tendon_rms": score * 0, "startup_tendon_max": score * 0,
                "startup_pass": torch.tensor(True)}


class FakeDependencies:
    def __init__(self, fault_at=None):
        self.rollout = FakeRollout(fault_at)
        self.loads = 0

    def load(self, freeze, root):
        self.loads += 1
        return generated_data()

    def build(self, data, device):
        assert device == "cpu"
        return self.rollout, FakeObjective(), (
            torch.zeros(e.N, dtype=torch.float32),
            torch.zeros(e.N, dtype=torch.float32),
            torch.zeros((391, 3), dtype=torch.float64),
            torch.zeros(96, dtype=torch.float64),
            torch.zeros((300, 4), dtype=torch.float32),
            torch.ones(e.N, dtype=torch.float32))


def generated_data():
    rate = np.zeros(e.N, np.float32)
    z = np.zeros(e.N, np.float32)
    bank = np.zeros((391, 3), np.float64)
    original = np.zeros(96, np.float64)
    return {
        "native": {"csr": {"data": np.array([.2, .4], np.float32)},
                   "mask": {"positions": np.array([0, 1]),
                            "edge_parameters": np.array([0, 1])}},
        "raw": {"command": np.full((301, 42), .1, np.float64),
                "tendon": np.full((301, 6), .02, np.float64)},
        "final": {"control_tick": 300, "network": {"tick": 6000, "rate": rate,
                                                  "adaptation": z},
                  "adapter": {"bank": bank, "original": {"activation": original}}},
    }


def meter():
    return {"host_rss_bytes": 12 << 20, "host_peak_bytes": 12 << 20,
            "device_free_bytes": 10 << 30, "device_total_bytes": 16 << 30,
            "torch_reserved_bytes": 0, "torch_peak_reserved_bytes": 0}


def fake_gate(*args, **kwargs):
    return {"source_sha256": {"fake.py": "a" * 64},
            "input_sha256": {"fake.npz": "b" * 64}}


def test_single_direction_f32_and_full_control0_reset(tmp_path):
    deps = FakeDependencies()
    output = tmp_path / "result"
    result = e.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                   output, gatekeeper=fake_gate, dependencies=deps, meter=meter,
                   device="cpu")
    assert result["run_valid"] and result["classification"] == "supported"
    assert result["comparison"]["relative_error"] < 1e-9
    assert len(deps.rollout.calls) == 3
    assert all(torch.equal(a[i], b[i]) for a, b in zip(deps.rollout.calls[:1] * 2,
                                                      deps.rollout.calls[1:])
               for i in range(1, 6))
    direction = np.load(output / "direction.npz")
    assert np.all(direction["theta_plus"] == np.float64(np.float32(1.001)))
    assert np.all(direction["theta_minus"] == np.float64(np.float32(.999)))
    assert result["direction"]["selected_weight_changed_plus_count"] == 2
    assert result["minus_improves_neutral"] and result["minus_startup_pass"]
    assert result["cost"]["receipts"]["neutral"]["confirmed"]["backward_blocks"] == 300
    assert result["cost"]["receipts"]["minus"]["confirmed"]["backward_blocks"] == 0
    for name in ("construct", "neutral_forward", "neutral_backward",
                 "plus_forward", "minus_forward"):
        assert result["cost"]["phases"][name]["requested"]
        assert result["cost"]["phases"][name]["host_returned"] is True
    assert result["cost"]["phases"]["neutral_forward"]["device_confirmed"] is True
    assert (output / "neutral.npz").is_file()
    assert (output / "plus.npz").is_file()
    assert (output / "minus.npz").is_file()


def test_neutral_failure_stops_all_direction_work(tmp_path):
    data = generated_data()
    data["raw"]["command"][4, 0] += 1e-5
    deps = FakeDependencies()
    seen = []
    with pytest.raises(ValueError, match="neutral full-trace"):
        e.compute(data, dependencies=deps, device="cpu",
                  checkpoint=lambda name, arrays: seen.append(name))
    assert seen == ["neutral"]
    assert len(deps.rollout.calls) == 1


def test_failed_plus_preserves_neutral_gradient_receipt_and_stops_minus(tmp_path):
    deps = FakeDependencies(fault_at=2)
    output = tmp_path / "result"
    result = e.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                   output, gatekeeper=fake_gate, dependencies=deps, meter=meter,
                   device="cpu")
    assert not result["run_valid"] and result["first_error"]["stage"] == "science"
    assert len(deps.rollout.calls) == 2
    assert (output / "neutral.npz").is_file()
    assert (output / "gradient.npz").is_file()
    assert (output / "direction.npz").is_file()
    assert not (output / "minus.npz").exists()
    assert result["cost"]["receipts"]["plus"]["forward_controls_requested"] == 1
    assert result["cost"]["receipts"]["plus"]["forward_controls_returned"] == 0
    assert result["cost"]["phases"]["plus_forward"]["requested"]
    assert result["cost"]["phases"]["plus_forward"]["host_returned"] is None
    assert result["cost"]["phases"]["plus_forward"]["device_confirmed"] is None
    assert result["cost"]["elapsed_s"] > 0


@pytest.mark.parametrize("fault", ["output-manifest.json", "terminal-result"])
def test_publication_failure_never_returns_success(tmp_path, fault):
    seen = {"result.json": 0}

    def writer(path, value):
        if path.name == "result.json":
            seen["result.json"] += 1
        if path.name == fault or (fault == "terminal-result"
                                  and path.name == "result.json"
                                  and seen["result.json"] == 2):
            raise OSError("generated publication fault")
        e.put_json(path, value)

    output = tmp_path / "result"
    result = e.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                   output, gatekeeper=fake_gate, dependencies=FakeDependencies(),
                   writer=writer, meter=meter, device="cpu")
    assert not result["run_valid"]
    assert result["first_error"]["stage"] in ("output_manifest", "terminal_publication")
    assert result["cost"]["elapsed_s"] > 0
    assert (output / "neutral.npz").exists()
    if fault == "terminal-result":
        assert (output / "publication-failure.json").exists()


def test_manifest_publication_time_is_bounded(tmp_path):
    seconds = [0.0]

    def writer(path, value):
        e.put_json(path, value)
        if path.name == "output-manifest.json":
            seconds[0] += 500.0

    output = tmp_path / "result"
    result = e.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                   output, gatekeeper=fake_gate, dependencies=FakeDependencies(),
                   writer=writer, clock=lambda: seconds[0], meter=meter,
                   device="cpu")
    assert not result["run_valid"]
    assert result["cost"]["finalization_wall_s"] == 500
    assert result["cost"]["elapsed_s"] == 500
    assert {row["stage"] for row in result["errors"]} >= {
        "finalization_deadline", "internal_deadline"}


def test_failed_backward_retains_forward_raw_and_unreturned_boundary(tmp_path, monkeypatch):
    def fail_backward(self, gradient):
        self.status = "backward_failed"
        raise RuntimeError("generated backward fault")

    monkeypatch.setattr(FakeReceipt, "backward", fail_backward)
    output = tmp_path / "result"
    result = e.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                   output, gatekeeper=fake_gate, dependencies=FakeDependencies(),
                   meter=meter, device="cpu")
    assert not result["run_valid"]
    assert (output / "neutral.npz").exists()
    assert not (output / "gradient.npz").exists()
    assert result["cost"]["phases"]["neutral_forward"]["device_confirmed"] is True
    backward = result["cost"]["phases"]["neutral_backward"]
    assert backward["requested"] and backward["host_returned"] is None
    assert backward["device_confirmed"] is None and backward["failed"]
    assert result["cost"]["receipts"]["neutral"]["forward_controls_returned"] == 300


def test_nonfinite_final_bank_is_never_a_neutral_reproduction():
    data = generated_data()
    deps = FakeDependencies()
    rollout, _, initial = deps.build(data, "cpu")
    output = rollout.run(*initial[:4], initial[4],
                         torch.ones(e.GROUPS, dtype=torch.float64), initial[5])
    output["bank"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Finite exact neutral"):
        e._compare_neutral(output, data)


def test_actual_builder_accepts_persisted_list_neutral(monkeypatch):
    from types import SimpleNamespace

    from tools import banc_m3_checkpointed_rollout as rollout_module
    from tools import banc_m3_grouped_current_adjoint as current_module
    from tools import banc_m3_monotone_adjoint as decoder_module
    from tools import banc_m3_phase_free_shape as objective_module
    from tools import banc_m3_ssp_adjoint as neural_module

    seen = []
    monkeypatch.setattr(e, "N", 4)
    monkeypatch.setattr(current_module, "GroupedCurrent", lambda *args: None)
    monkeypatch.setattr(neural_module, "SSPAdjointModel", lambda *args: None)
    monkeypatch.setattr(decoder_module, "MonotoneDecoder", lambda *args, **kwargs: seen.append(args[-1]))
    monkeypatch.setattr(rollout_module, "CheckpointedControlRollout", lambda *args, **kwargs: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(objective_module, "PhaseFreeShapeObjective", lambda *args, **kwargs: None)
    calibration = {key: np.zeros(1, np.float64) for key in ("moment", "rest", "pool", "motor_ids", "ltm")}
    parameters = {key: np.zeros(1, np.float64) for key in ("tau", "kappa", "slopes", "membership")}
    neutral = np.linspace(-.3, .3, 42, dtype=np.float64)
    data = {"native": {"csr": {"data": np.empty(0, np.float32), "indices": np.empty(0, np.int32), "indptr": np.zeros(5, np.int32)},
                        "mask": {"positions": np.empty(0, np.int64), "edge_parameters": np.empty(0, np.int64)},
                        "physiology": {key: np.zeros(4, np.float32) for key in ("tau", "coefficient", "threshold", "cap", "gain", "inverse")}},
            "start": {"adapter": {"hash": "1fd7b86939a6846e01c4837fb5c2fbe24ce8c776e3331c1f75921a376c53b067",
                                    "calibration": calibration, "parameters": parameters,
                                    "bank": np.zeros((391, 3)), "original": {"activation": np.zeros(96)}},
                      "body": {"action": {"joint_angles": neutral.tolist()}},
                      "network": {"rate": np.zeros(4, np.float32), "adaptation": np.zeros(4, np.float32), "output_mask": np.ones(4, np.float32)}},
            "calibration": calibration, "parameters": parameters,
            "curve": np.zeros((6, 128, 7)), "startup": {"target_command": np.zeros((99, 42)), "target_tendon": np.zeros((99, 6))},
            "drives": np.zeros((300, 4), np.float32)}
    e.Dependencies().build(data, "cpu")
    assert len(seen) == 1 and e.exact(seen[0], neutral)
    assert isinstance(data["start"]["body"]["action"]["joint_angles"], list)


def test_constructor_failure_survives_failed_phase_journal():
    class Broken(FakeDependencies):
        def build(self, data, device):
            raise ValueError("primary constructor failure")

    def boundary(name, event):
        if event == "failed":
            raise OSError("secondary journal failure")

    with pytest.raises(ValueError, match="primary constructor failure") as caught:
        e.compute(generated_data(), dependencies=Broken(), device="cpu", on_boundary=boundary)
    assert any("secondary journal failure" in note for note in caught.value.__notes__)


def test_zero_gradient_stops_before_perturbation(monkeypatch):
    class FlatObjective(FakeObjective):
        def evaluate(self, q, tendon):
            out = super().evaluate(q, tendon)
            out["shape_loss"] = out["shape_loss"] * 0
            return out

    deps = FakeDependencies()
    monkeypatch.setattr(deps, "build", lambda data, device: (
        deps.rollout, FlatObjective(), FakeDependencies().build(data, device)[2]))
    seen = []
    with pytest.raises(ValueError, match="gradient zero"):
        e.compute(generated_data(), dependencies=deps, device="cpu",
                  checkpoint=lambda name, arrays: seen.append(name))
    assert seen == ["neutral", "gradient"]
    assert len(deps.rollout.calls) == 1


@pytest.mark.parametrize("observed,predicted,expected", [
    (1.04, 1., "supported"), (-1., 1., "refuted"),
    (1.06, 1., "refuted"), (1e-10, 1., "inconclusive"),
    (float("nan"), 1., "inconclusive")])
def test_fixed_five_percent_screen(observed, predicted, expected):
    assert e.classify(observed, predicted)["classification"] == expected


def test_passive_gate_rejects_absent_root_approval_before_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "OUTPUT", "result")
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("{}")
    review.write_text("{}")
    deps = FakeDependencies()
    with pytest.raises(ValueError, match="frozen diagnostic"):
        e.run(freeze, e.sha256(freeze), review, e.sha256(review),
              tmp_path / "result", root=tmp_path, dependencies=deps,
              meter=meter, device="cpu")
    assert deps.loads == 0
    assert not (tmp_path / "result").exists()


def test_passive_gate_pins_approved_producer_and_rejects_tamper(tmp_path, monkeypatch):
    """Tiny JSON/file fixture proves the gate precedes all injected heavy work."""
    monkeypatch.setattr(e, "OUTPUT", "result")
    monkeypatch.setattr(e, "E318", "producer")
    test_e317 = "tests/test_c_banc_checkpointed_rollout.py"
    test_e319 = "tests/test_c_banc_phase_free_shape.py"
    monkeypatch.setattr(e, "REQUIRED_SOURCES", ("source.py", test_e317, test_e319))
    paths = {key: key + ".json" for key in e.REQUIRED_PATHS}
    for key in ("e318_result", "e318_source_manifest", "e318_output_manifest",
                "e318_raw", "e318_start", "e318_final", "e318_chunks"):
        paths[key] = "producer/" + key + ("" if key in ("e318_start", "e318_final",
                                                       "e318_chunks") else ".json")
    monkeypatch.setattr(e, "REQUIRED_PATHS", paths)

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")
        return path

    source = tmp_path / "source.py"
    source.write_text("pass\n")
    write(test_e317, {"generated": "E317 test source"})
    write(test_e319, {"generated": "E319 test source"})
    source_hashes = {name: e.sha256(tmp_path / name)
                     for name in ("source.py", test_e317, test_e319)}
    producer_sources = {"source.py": source_hashes["source.py"]}
    for key in ("e318_start", "e318_final", "e318_chunks"):
        write(paths[key] + "/data.json", {"generated": True})
    write(paths["e318_raw"], {"generated": True})
    write(paths["e318_result"], {
        "schema": "flylab.m3.e318.full-drive-capture.result.v1",
        "run_valid": True, "learning_input_approved": True})
    write(paths["e318_freeze"], {
        "schema": "flylab.m3.e318.full-drive-capture.freeze.v1",
        "source_sha256": producer_sources, "input_sha256": {}})
    e318_freeze_sha = e.sha256(tmp_path / paths["e318_freeze"])
    write(paths["e318_review"], {
        "schema": "flylab.m3.e318.full-drive-capture.review.v1",
        "approved": True, "blocking_defects": [],
        "freeze_sha256": e318_freeze_sha, "source_sha256": producer_sources})
    write(paths["e318_source_manifest"], {
        "freeze_sha256": e318_freeze_sha,
        "review_sha256": e.sha256(tmp_path / paths["e318_review"]),
        "source_sha256": producer_sources})
    write(paths["e317_review"], {"schema": "flylab.m3.e317.root-second-review.v1",
                                 "approved": True, "blocking_defects": [],
                                 "source_sha256": {test_e317: source_hashes[test_e317]}})
    write(paths["e319_review"], {"schema": "flylab.m3.e319.root-second-review.v1",
                                 "approved": True, "blocking_defects": [],
                                 "source_sha256": {test_e319: source_hashes[test_e319]}})
    for key in ("e317_cuda", "e319_cuda"):
        write(paths[key], {"passed": True})
    for key, schema, test_source in (
        ("e317_qualification", "flylab.m3.e317.generated-qualification.v1", test_e317),
        ("e319_qualification", "flylab.m3.e319.generated-qualification.v1", test_e319)):
        evidence = key + "-evidence.json"
        write(evidence, {"generated": key})
        write(paths[key], {"schema": schema, "approved": True,
                           "blocking_defects": [], "original_calls": 0,
                           "source_sha256": {test_source: source_hashes[test_source]},
                           "evidence_sha256": {evidence: e.sha256(tmp_path / evidence)}})
    write("e314.json", {"schema": "flylab.m3.e314.native-adjoint-generated-qualification.v1",
                        "approved": True, "evidence_sha256": {}})
    write(paths["e315_freeze"], {
        "schema": "flylab.m3.e315.native-adjoint-probe.freeze.v1",
        "source_sha256": {}, "inputs": {}, "producer_reviews": [],
        "qualification": {"path": "e314.json", "sha256": e.sha256(tmp_path / "e314.json")}})
    for key in ("e85_reference", "e103_startup", "e150_calibration", "e150_parameters"):
        write(paths[key], {"generated": True})
    producer_files = {p.relative_to(tmp_path / "producer").as_posix(): {
        "sha256": e.sha256(p), "bytes": p.stat().st_size}
        for p in (tmp_path / "producer").rglob("*") if p.is_file()
        and p != tmp_path / paths["e318_result"]}
    write(paths["e318_output_manifest"], {
        "schema": "flylab.m3.e318.output-manifest.v1", "files": producer_files})
    write(paths["e318_audit"], {"pass": True, "diagnostic_run_valid": True,
                                "learning_input_approved": True,
                                "result_sha256": e.sha256(tmp_path / paths["e318_result"]),
                                "output_manifest_sha256": e.sha256(
                                    tmp_path / paths["e318_output_manifest"])})
    files = e._file_roster(paths, tmp_path) | {"e314.json",
           "e317_qualification-evidence.json", "e319_qualification-evidence.json"}
    pins = {name: e.sha256(tmp_path / name) for name in files}
    freeze = write("freeze.json", {
        "schema": e.SCHEMA, "paths": paths, "limits": e.LIMITS,
        "output": "result", "candidate_adopted": False,
        "input_sha256": pins, "source_sha256": source_hashes})
    review = write("review.json", {
        "schema": e.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
        "freeze_sha256": e.sha256(freeze), "source_sha256": source_hashes})
    assert e.gate(freeze, e.sha256(freeze), review, e.sha256(review),
                  tmp_path / "result", root=tmp_path)["input_sha256"] == pins
    (tmp_path / paths["e318_raw"]).write_text("tampered")
    with pytest.raises(ValueError, match="Frozen source/input differs"):
        e.gate(freeze, e.sha256(freeze), review, e.sha256(review),
               tmp_path / "result", root=tmp_path)
