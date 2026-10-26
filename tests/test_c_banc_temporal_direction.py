"""Generated E330 runner contracts; E329 loss verification is not repeated."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import verify_c_banc_m3_temporal_direction as subject


@pytest.fixture(autouse=True)
def cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def native(count=4):
    return {"csr": {"data": np.array([.25, -.5, .125, -0.], np.float32),
                    "indices": np.array([0, 1, 2, 3], np.int32),
                    "indptr": np.arange(count + 1, dtype=np.int32)},
            "mask": {"positions": np.array([0, 1, 2], np.int64),
                     "edge_parameters": np.arange(3, dtype=np.int64)},
            "network_identity": "generated-source-network"}


class ToyObjective:
    def __init__(self, *, zero=False, startup=True):
        self.zero, self.startup = zero, startup

    def evaluate_pairs(self, previous, next_q):
        successor = (next_q - previous - .01).square().mean()
        coordination = .03 * next_q.square().mean()
        scale = 0. if self.zero else 1.
        successor, coordination = successor * scale, coordination * scale
        return {"training_loss": successor + coordination, "successor_loss": successor,
                "coordination_loss": coordination, "lr_coordination_loss": coordination * .5,
                "shape_loss": next_q.square().mean(),
                "per_leg_successor_loss": successor.expand(6),
                "per_leg_shape_loss": next_q.square().mean().expand(6)}

    def evaluate(self, q, tendon):
        out = self.evaluate_pairs(q[99:299], q[100:300])
        zero = torch.tensor(0., dtype=torch.float64)
        out.update(startup_q_rms=zero, startup_q_max=torch.tensor(0. if self.startup else .041),
                   startup_tendon_rms=zero, startup_tendon_max=zero,
                   startup_pass=torch.tensor(self.startup))
        return out


class Receipt:
    def __init__(self, count):
        self.count = count
        self.status = "host_returned"
        for key in ("forward_controls_requested", "forward_controls_returned"):
            setattr(self, key, count)
        for key in ("forward_neural_ticks_requested", "forward_neural_ticks_returned"):
            setattr(self, key, count * 20)
        for key in ("recompute_controls_requested", "recompute_controls_returned", "recompute_neural_ticks_requested",
                    "recompute_neural_ticks_returned", "backward_blocks_requested", "backward_blocks_returned",
                    "backward_block_ticks_returned", "confirm_requested", "confirm_returned"):
            setattr(self, key, 0)
        self.confirmed = None

    def back(self, gradient):
        self.recompute_controls_requested = self.recompute_controls_returned = self.count
        self.recompute_neural_ticks_requested = self.recompute_neural_ticks_returned = self.count * 20
        self.backward_blocks_requested = self.backward_blocks_returned = self.count
        self.backward_block_ticks_returned = self.count * 20
        return gradient

    def confirm(self):
        self.confirm_requested += 1
        self.confirm_returned += 1
        self.status = "confirmed"
        self.confirmed = {"forward_controls": self.forward_controls_returned,
            "forward_neural_ticks": self.forward_neural_ticks_returned,
            "recompute_controls": self.recompute_controls_returned,
            "recompute_neural_ticks": self.recompute_neural_ticks_returned,
            "backward_blocks": self.backward_blocks_returned,
            "backward_block_ticks": self.backward_block_ticks_returned}


def fixture(monkeypatch, *, starts=(100, 104), zero=False, startup=True, nonlinear=0):
    monkeypatch.setattr(subject, "N", 4)
    monkeypatch.setattr(subject, "GROUPS", 3)
    ticks = np.arange(1, 301, dtype=np.float32)
    drives = np.repeat(ticks[:, None], 4, axis=1)
    command = np.zeros((301, 42), np.float64)
    command[1:] = .2
    bank0, original0 = np.zeros((391, 3)), np.zeros(96)
    def whole(c):
        return {"adapter": {"bank": bank0 + c / 1000,
                             "original": {"activation": original0 + c / 1000}}}
    objective = ToyObjective(zero=zero, startup=startup)
    neutral = [subject.metrics(objective.evaluate_pairs(
        torch.tensor(command[c:c + 4]), torch.tensor(command[c + 1:c + 5]))) for c in starts]
    data = {"drives": drives, "rates": drives.copy(), "adaptations": drives.copy(),
            "motor_rates": np.repeat(ticks[:, None], 391, axis=1),
            "raw": {"command": command, "tendon": np.zeros((301, 6))},
            "whole100": whole(100), "final": whole(300), "native": native(),
            "e329_neutral_terms": {key: float(np.mean([row[key] for row in neutral])) for key in subject.TERM_NAMES},
            "start": {"adapter": {"hash": "generated-decoder-source"}}}

    class Decoder:
        def step(self, motor, bank, original, *, dt):
            assert dt == .005
            control = motor[0].to(torch.float64)
            return {"q": torch.full((42,), .2, dtype=torch.float64), "tendon": torch.zeros(6, dtype=torch.float64),
                    "bank": torch.zeros_like(bank) + control / 1000,
                    "original_activation": torch.zeros_like(original) + control / 1000}

    class Rollout:
        device = torch.device("cpu")
        decoder = Decoder()
        fail = False
        nonfinite_gradient = False
        mutate = False

        def __init__(self):
            self.calls = []

        def run(self, r, z, bank, original, held, theta, mask):
            start = int(held[0, 0]) - 1
            self.calls.append({"start": start, "rate": r.clone(), "adaptation": z.clone(),
                               "bank": bank.clone(), "original": original.clone(), "theta": theta.detach().clone()})
            receipt = Receipt(len(held))
            if self.fail:
                receipt.forward_controls_returned = 2
                receipt.forward_neural_ticks_returned = 40
                error = RuntimeError("injected forward failure")
                error.receipt = receipt
                raise error
            base = torch.full((len(held),), .2, dtype=torch.float64)
            shift = torch.dot(theta - 1, torch.tensor([.02, .01, 1e-9], dtype=torch.float64))
            shift = shift + nonlinear * (theta[0] - 1).square()
            q = (base + shift)[:, None].expand(-1, 42)
            if q.requires_grad:
                q.register_hook(receipt.back)
                if self.nonfinite_gradient:
                    theta.register_hook(lambda grad: torch.full_like(grad, torch.inf))
            if self.mutate:
                r.fill_(-9)
                z.fill_(-9)
                bank.fill_(-9)
                original.fill_(-9)
            end = held[-1, 0].to(torch.float64)
            return {"q": q, "tendon": torch.zeros((len(held), 6), dtype=torch.float64),
                    "rate": held[-1].clone(), "adaptation": held[-1].clone(),
                    "bank": torch.zeros((391, 3), dtype=torch.float64) + end / 1000,
                    "original_activation": torch.zeros(96, dtype=torch.float64) + end / 1000,
                    "receipt": receipt}

    class Deps:
        def __init__(self):
            self.rollout = Rollout()
            self.load_called = False

        def load(self, _freeze, root):
            self.load_called = True
            return data

        def build(self, _data, _device):
            initial = (torch.zeros(4, dtype=torch.float32), torch.zeros(4, dtype=torch.float32),
                       torch.tensor(bank0), torch.tensor(original0), torch.tensor(drives), torch.ones(4))
            return self.rollout, objective, initial
    return data, Deps(), objective


def test_window_boundary_live_previous_last_coordination_and_one_over50():
    obj = ToyObjective()
    q = torch.tensor(np.arange(168).reshape(4, 42) / 10000 + .2, dtype=torch.float64, requires_grad=True)
    boundary = torch.full((42,), .19, dtype=torch.float64, requires_grad=True)
    terms, loss = subject.window_loss(obj, boundary, q)
    actual, fixed = torch.autograd.grad(loss, (q, boundary), allow_unused=True)
    x = q.detach().numpy()
    previous = np.vstack((np.full(42, .19), x[:3]))
    residual = x - previous - .01
    expected = 2 * residual / 168 + .06 * x / 168
    expected[:3] -= 2 * residual[1:] / 168
    np.testing.assert_allclose(actual.numpy(), expected / 50, rtol=1e-12, atol=1e-12)
    assert fixed is None
    bad = obj.evaluate_pairs(torch.tensor(previous), q)["training_loss"] / 50
    wrong = torch.autograd.grad(bad, q)[0].numpy()
    assert np.max(np.abs(wrong[:3] - actual.numpy()[:3])) > 1e-8
    np.testing.assert_allclose(actual[-1], (2 * residual[-1] + .06 * x[-1]) / (168 * 50))
    assert terms["coordination_loss"] > 0


def test_generated_direction_independent_windows_and_original_control0_continuation(monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    deps.rollout.mutate = True
    saved = {}
    result, receipts = subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104),
        checkpoint=lambda name, value: saved.setdefault(name, value))
    assert result["local_direction"] == "supported" and result["local_descent"]
    assert result["continuation"]["classification"] == "supported"
    assert result["continuation"]["original_control0_used"]
    assert len(receipts) == 7
    assert result["actual_work"]["forward_neural_ticks"] == 6480
    assert result["actual_work"]["backward_block_ticks"] == 160
    assert [row["start"] for row in deps.rollout.calls] == [100, 104, 100, 100, 104, 104, 0]
    for row in deps.rollout.calls[:-1]:
        np.testing.assert_array_equal(row["rate"].numpy(), data["rates"][row["start"] - 1])
        assert float(row["bank"][0, 0]) == row["start"] / 1000
    assert deps.rollout.calls[-1]["rate"].abs().sum() == 0
    assert deps.rollout.calls[-1]["bank"].abs().sum() == 0
    gradient = saved["direction"]["gradient"]
    np.testing.assert_array_equal(gradient, saved["direction"]["window_gradients"].sum(axis=0))
    np.testing.assert_allclose(saved["direction"]["direction"], gradient / np.max(np.abs(gradient)))
    assert saved["direction"]["theta_plus"][2] == 1
    for label in ("plus", "minus"):
        theta = saved["direction"]["theta_" + label]
        np.testing.assert_array_equal(theta, theta.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(saved["selected-weights-" + label]["actual32"],
            np.asarray(data["native"]["csr"]["data"][:3].astype(np.float64) * theta, np.float32))
    handoff = result["e331_handoff"]
    assert handoff["eligible"] and handoff["body_not_executed"]
    assert handoff["candidate_content_identity"] != handoff["source_network_identity"]
    assert handoff["native_eligibility"]["theta64_sha256"] == subject.bytes_sha(saved["direction"]["theta_minus"])


def test_zero_gradient_is_valid_inconclusive_with_only_requested_roster(monkeypatch):
    data, deps, _ = fixture(monkeypatch, zero=True)
    result, receipts = subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104))
    assert result["local_direction"] == "inconclusive"
    assert result["continuation"]["reasons"] == ["finite_exact_zero_gradient"]
    assert result["continuation"]["classification"] == "not_tested"
    assert len(receipts) == 2
    assert result["actual_work"]["forward_neural_ticks"] == 160
    assert subject.make_direction(np.zeros(3)) is None


def test_nonfinite_gradient_is_invalid_and_retains_backward_receipt(monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    deps.rollout.nonfinite_gradient = True
    receipts = {}
    with pytest.raises(ValueError, match="gradient"):
        subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104),
                        on_receipt=lambda name, value: receipts.update({name: value}))
    assert receipts["window-100-neutral"]["confirmed"]["backward_block_ticks"] == 80


def test_palette_failure_records_local_pair_then_skips_continuation(monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    values = np.arange(1, 16410, dtype=np.float32) / 65536
    values[:3] = [.25, -.5, .125]
    data["native"]["csr"] = {"data": values, "indices": np.zeros(len(values), np.int32),
                              "indptr": np.array([0, len(values), len(values), len(values), len(values)], np.int32)}
    result, receipts = subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104))
    assert result["local_direction"] == "supported"
    assert result["native_eligibility"]["minus"]["eligible"] is False
    assert result["continuation"]["classification"] == "not_tested"
    assert "minus_native_csr_ineligible" in result["continuation"]["reasons"]
    assert len(receipts) == 6


def test_no_descent_and_startup_failure_have_separate_results(monkeypatch):
    data, deps, _ = fixture(monkeypatch, nonlinear=100)
    result, _ = subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104))
    assert result["local_direction"] == "supported"
    assert result["local_descent"] is False
    assert result["continuation"]["classification"] == "not_tested"
    data, deps, _ = fixture(monkeypatch, startup=False)
    result, _ = subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104))
    assert result["local_direction"] == "supported" and result["local_descent"]
    assert result["continuation"]["completed"]
    assert result["continuation"]["classification"] == "rejected"
    assert result["e331_handoff"]["eligible"] is False


def test_full_palette_bit_patterns_and_neutral_topology_bytes():
    data = native()
    data["csr"]["data"] = np.r_[data["csr"]["data"], np.float32(0.)]
    data["csr"]["indices"] = np.r_[data["csr"]["indices"], np.int32(3)]
    data["csr"]["indptr"][-1] = 5
    before = copy.deepcopy(data)
    patched, actual, metadata = subject.patch_weights(data, np.ones(3))
    assert metadata["full_csr_bit_palette_count"] == 5
    assert subject.bytes_sha(patched["data"]) == subject.bytes_sha(before["csr"]["data"])
    assert actual.tobytes() == before["csr"]["data"][:3].tobytes()
    for field in ("indices", "indptr", "data"):
        assert data["csr"][field].tobytes() == before["csr"][field].tobytes()
    with pytest.raises(ValueError, match="promoted"):
        subject.patch_weights(data, np.array([1.001, 1., 1.]))
    with pytest.raises(ValueError, match="prediction"):
        subject.classify_direction(1., 1.1, .9, np.ones(3), np.ones(3), np.full(3, 2.))
    with pytest.raises(ValueError, match="losses"):
        subject.classify_direction(np.nan, 1., 1., np.ones(3), np.ones(3), np.ones(3))


def test_frozen_normalizer_compare_before_overwrite_actual_tensor_hash():
    vector = np.arange(1, 7, dtype=np.float64) / 10000
    constants = {"normalization_f64": vector.tolist(), "normalization": {"c_order_sha256": subject.bytes_sha(vector)}}
    objective = SimpleNamespace(device=torch.device("cpu"), successor_normalization=torch.tensor(np.nextafter(vector, np.inf)))
    result = subject.install_normalizer(objective, constants)
    assert result["device_reduction_bytes_claimed_equal"] is False
    assert result["actual_loss_denominator_sha256"] == subject.bytes_sha(vector)
    assert objective.successor_normalization.numpy().tobytes() == vector.tobytes()
    assert not objective.successor_normalization.requires_grad
    bad = SimpleNamespace(device=torch.device("cpu"), successor_normalization=torch.tensor(vector * 2))
    old = bad.successor_normalization
    with pytest.raises(ValueError, match="Device-computed"):
        subject.install_normalizer(bad, constants)
    assert bad.successor_normalization is old


def test_real_passive_hash_rejection_precedes_meter_loader_and_model(tmp_path, monkeypatch):
    source = tmp_path / "owned.py"
    source.write_text("generated source", encoding="utf8")
    sources = {"owned.py": "0" * 64}
    monkeypatch.setattr(subject, "_closure", lambda root: (sources, set()))
    frozen = {"schema": subject.SCHEMA, "paths": subject.PATHS, "limits": subject.LIMITS,
              "output": subject.OUTPUT, "candidate_adopted": False, "original_execution_authorized": True,
              "original_supervisor_required": True, "source_sha256": sources, "input_sha256": {}}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(frozen), encoding="utf8")
    reviewed = {"schema": subject.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
                "freeze_sha256": subject.e323.sha256(freeze), "source_sha256": sources}
    review = tmp_path / "review.json"
    review.write_text(json.dumps(reviewed), encoding="utf8")
    def forbidden(*args, **kwargs):
        pytest.fail("passive hash rejection reached CUDA/loader/model")
    with pytest.raises(ValueError, match="pinned source"):
        subject.run(freeze, subject.e323.sha256(freeze), review, subject.e323.sha256(review),
                    tmp_path / subject.OUTPUT, root=tmp_path, meter=forbidden,
                    dependencies=SimpleNamespace(load=forbidden, build=forbidden))


def test_save_failure_retains_partial_confirmed_receipt(monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    receipts, phases = {}, []
    def save(name, arrays):
        if name == "window-100-neutral":
            raise OSError("injected save failure")
    with pytest.raises(OSError, match="save failure"):
        subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104), checkpoint=save,
            on_receipt=lambda name, value: receipts.update({name: value}),
            on_boundary=lambda name, event: phases.append((name, event)))
    row = receipts["window-100-neutral"]
    assert row["confirmed"]["forward_neural_ticks"] == 80
    assert row["confirmed"]["backward_block_ticks"] == 0
    assert ("window-100-neutral_forward", "failed") in phases


def test_original_failure_run_preserves_first_error_partial_receipt_and_publication(tmp_path, monkeypatch):
    data, deps, _ = fixture(monkeypatch, starts=subject.STARTS)
    deps.rollout.fail = True
    frozen = {"input_sha256": {subject.PATHS["e323_result"]: "generated", subject.PATHS["e323_audit"]: "generated"},
              "source_sha256": {}}
    def meter():
        return {"host_peak_bytes": 0, "device_total_bytes": 1, "device_free_bytes": 1,
                "torch_reserved_bytes": 0, "torch_peak_reserved_bytes": 0}
    result = subject.run(tmp_path / "unused", "f", tmp_path / "unused", "r", tmp_path / "out",
        root=tmp_path, gatekeeper=lambda *args, **kwargs: frozen, dependencies=deps, meter=meter, device="cpu")
    assert result["run_valid"] is False and result["local_direction"] == "invalid"
    assert result["first_error"]["message"] == "injected forward failure"
    receipts = json.loads((tmp_path / "out/receipts.json").read_text(encoding="utf8"))
    assert receipts["receipts"]["window-100-neutral"]["confirmed"]["forward_neural_ticks"] == 40
    assert json.loads((tmp_path / "out/result.json").read_text(encoding="utf8")) == result
    assert (tmp_path / "out/output-manifest.json").is_file()
    assert deps.load_called


def generated_freeze():
    return {"input_sha256": {subject.PATHS["e323_result"]: "generated",
                            subject.PATHS["e323_audit"]: "generated"}, "source_sha256": {}}


def cpu_meter():
    return {"host_peak_bytes": 0, "device_total_bytes": 1, "device_free_bytes": 1,
            "torch_reserved_bytes": 0, "torch_peak_reserved_bytes": 0}


def test_prep_time_preserves_absolute_finalization_reserve(tmp_path, monkeypatch):
    elapsed = [0.]
    class Deps:
        def load(self, freeze, root):
            elapsed[0] = 61.
            return {}
    def compute(data, **callbacks):
        elapsed[0] = 179.
        callbacks["deadline"]()
        elapsed[0] = 180.
        callbacks["deadline"]()
        pytest.fail("science was allowed into reserved finalization time")
    monkeypatch.setattr(subject, "compute", compute)
    result = subject.run(tmp_path / "f", "f", tmp_path / "r", "r", tmp_path / "out", root=tmp_path,
        gatekeeper=lambda *args, **kwargs: generated_freeze(), dependencies=Deps(), clock=lambda: elapsed[0],
        meter=cpu_meter, device="cpu")
    assert result["first_error"]["message"] == "E330 science deadline exceeded"
    assert result["cost"]["science_started_elapsed_s"] == 61.
    assert result["cost"]["science_deadline_elapsed_s"] == 180.
    assert result["cost"]["science_wall_s"] == 119.
    assert result["cost"]["finalization_wall_s"] == 0.
    assert not result["run_valid"]


def test_run_save_failure_pins_unreturned_attempt_and_decoder_partial_work(tmp_path, monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    def fail_save(path, **arrays):
        raise OSError("checkpoint disk unavailable")
    result = subject.run(tmp_path / "f", "f", tmp_path / "r", "r", tmp_path / "out", root=tmp_path,
        gatekeeper=lambda *args, **kwargs: generated_freeze(), dependencies=deps, meter=cpu_meter,
        saver=fail_save, device="cpu")
    assert result["first_error"]["message"] == "checkpoint disk unavailable"
    attempt = result["cost"]["attempts"][0]
    assert attempt["requested"] and attempt["returned"] is None and attempt["failed"]
    assert attempt["stage"] == "save_decoder-replay"
    assert result["cost"]["decoder_replay_work"] == {"requested_controls": 300,
        "host_returned_controls": 300, "device_confirmed_controls": None}
    assert result["cost"]["receipts"] == {} and not deps.rollout.calls
    assert json.loads((tmp_path / "out/result.json").read_text("utf8")) == result


def test_publication_failure_does_not_issue_eligible_handoff(tmp_path, monkeypatch):
    class Deps:
        def load(self, freeze, root):
            return {}
    def compute(data, **kwargs):
        return {"local_direction": "supported", "local_descent": True, "requested_rollouts": [],
                "continuation": {"classification": "supported"}, "e331_handoff": {"eligible": True}}, {}
    def writer(path, value):
        if path.name == "output-manifest.json":
            raise OSError("manifest unavailable")
        subject.put_json(path, value)
    monkeypatch.setattr(subject, "compute", compute)
    result = subject.run(tmp_path / "f", "f", tmp_path / "r", "r", tmp_path / "out", root=tmp_path,
        gatekeeper=lambda *args, **kwargs: generated_freeze(), dependencies=Deps(), writer=writer,
        meter=cpu_meter, device="cpu")
    assert result["local_direction"] == "supported" and not result["run_valid"]
    assert result["first_error"]["message"] == "manifest unavailable"
    assert result["e331_handoff"]["eligible"] is False
    assert json.loads((tmp_path / "out/result.json").read_text("utf8")) == result


def test_wrong_return_dtype_is_invalid_and_retains_device_receipt(monkeypatch):
    data, deps, _ = fixture(monkeypatch)
    original = deps.rollout.run
    def bad_return(*args):
        result = original(*args)
        result["adaptation"] = result["adaptation"].to(torch.float64)
        return result
    deps.rollout.run = bad_return
    receipts = {}
    with pytest.raises(ValueError, match="device/shape/dtype"):
        subject.compute(data, dependencies=deps, device="cpu", starts=(100, 104),
            on_receipt=lambda name, value: receipts.update({name: value}))
    assert receipts["window-100-neutral"]["confirmed"]["forward_neural_ticks"] == 80


def test_finite_scalars_with_overflowed_difference_are_invalid():
    with pytest.raises(ValueError, match="loss differences"):
        subject.classify_direction(1., 1e308, -1e308, np.ones(3), np.ones(3), np.ones(3))
    gain = np.ones(3)
    decision = subject.classify_direction(1., 1., 1., np.ones(3), gain, gain)
    assert decision["local_direction"] == "inconclusive" and decision["relative_error"] is None


def test_future_native_mapping_underflow_and_integer_field_limits(monkeypatch):
    data = native()
    gains = np.array([np.float32(.999), 1., 1.], dtype=np.float64)
    bad = copy.deepcopy(data)
    bad["mask"]["positions"][1] = 0
    with pytest.raises(ValueError, match="positions/groups"):
        subject.patch_weights(bad, gains)
    bad = copy.deepcopy(data)
    bad["csr"]["data"][0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError, match="nonzero signs"):
        subject.patch_weights(bad, np.array([.25, 1., 1.]))
    # A technically too-wide node field is distinct from a malformed CSR.
    large = copy.deepcopy(data)
    large["csr"]["indptr"] = np.r_[np.arange(5, dtype=np.int32),
                                   np.full((2 ** 18) - 3, 4, np.int32)]
    _, _, report = subject.patch_weights(large, gains)
    assert report["node_count"] == 2 ** 18 + 1 and report["eligible"] is False
