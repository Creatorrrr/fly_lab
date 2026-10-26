"""Generated CPU-only contract checks for the bounded E315 native probe."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from tools import verify_c_banc_m3_native_adjoint as probe


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def descriptor(root, name, text="test"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    return {"path": name, "sha256": probe.digest(path)}


def frozen_fixture(root):
    sources = {name: descriptor(root, name)["sha256"] for name in probe.SOURCES}
    evidence = descriptor(root, "evidence/check.json")
    qualification = {
        "schema": probe.QUALIFICATION,
        "approved": True,
        "source_sha256": {name: sources[name] for name in probe.PRODUCERS},
        "evidence_sha256": {evidence["path"]: evidence["sha256"]},
        "checks": {key: True for key in ("cpu_independent_pass", "cuda_neutral_exact", "cuda_adjoint_pass")},
        "original_model_runs": 0,
        "body_runs": 0,
        "training_runs": 0,
    }
    q = descriptor(root, "qualification.json", json.dumps(qualification))
    inputs = {name: descriptor(root, "inputs/" + name) for name in probe.INPUTS}
    for name in probe.STATES:
        path = root / "states" / name
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}", encoding="utf8")
        (path / "state.json").write_text("{}", encoding="utf8")
        inputs[name] = {"path": str(path.relative_to(root)).replace("\\", "/"),
                        "manifest_sha256": probe.digest(path / "manifest.json"),
                        "state_sha256": probe.digest(path / "state.json")}
    freeze = {"schema": probe.SCHEMA, "source_sha256": sources, "qualification": q,
              "inputs": inputs, "producer_reviews": [], "candidate_adopted": False,
              "network_identity": "f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be"}
    freeze_path = root / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf8")
    return freeze_path, freeze


def test_gate_accepts_exact_pins_then_rejects_mutation(tmp_path):
    path, freeze = frozen_fixture(tmp_path)
    assert probe.gate(path, tmp_path / "new", root=tmp_path)[0] == freeze
    (tmp_path / "evidence/check.json").write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="evidence differs"):
        probe.gate(path, tmp_path / "new", root=tmp_path)


def test_gate_requires_source_qualification_and_fresh_output(tmp_path):
    path, freeze = frozen_fixture(tmp_path)
    (tmp_path / "old").mkdir()
    with pytest.raises(FileExistsError):
        probe.gate(path, tmp_path / "old", root=tmp_path)
    freeze["source_sha256"][probe.OWNED[0]] = "0" * 64
    path.write_text(json.dumps(freeze), encoding="utf8")
    with pytest.raises(ValueError, match="Source differs"):
        probe.gate(path, tmp_path / "new", root=tmp_path)


@pytest.mark.parametrize(("observed", "predicted", "label"), [
    (1e-4, 1e-4, "supported"), (-1e-4, 1e-4, "refuted"),
    (1e-9, 1e-4, "inconclusive"), (float("nan"), 1e-4, "inconclusive"),
])
def test_strict_local_screen(observed, predicted, label):
    assert probe.screen(observed, predicted)["classification"] == label


class FakeCurrent:
    def __init__(self, weights, edge_ids, group_ids, group_count, device):
        assert isinstance(weights, csr_matrix)
        assert device == "cpu"
        self.n = weights.shape[0]
        self.group_count = group_count
        self.device = torch.device(device)


class FakeSSP:
    def __init__(self, operator, *_args):
        self.group_count = operator.group_count

    def step(self, rate, adaptation, _drive, efficacy, _mask):
        return rate + .1 * (efficacy.sum() - self.group_count) / self.group_count, adaptation


def fake_data():
    zero = np.zeros(probe.N, np.float32)
    one = np.ones(probe.N, np.float32)
    state = lambda rate, adaptation: {"rate": rate.copy(), "adaptation": adaptation.copy(),
                                      "drive": zero.copy(), "output_mask": one.copy()}
    initial = state(zero, zero)
    first = state(zero, zero)
    final = state(one, zero)
    return {"csr": {"indptr": np.zeros(probe.N + 1, np.int32), "indices": np.array([], np.int32),
                    "data": np.array([], np.float32)},
            "mask": {"positions": np.array([0], np.int64), "edge_parameters": np.array([0], np.int64),
                     "population": np.arange(probe.POPULATION, dtype=np.int64)},
            "physiology": {key: np.ones(probe.N, np.float32) for key in
                           ("tau", "coefficient", "threshold", "cap", "gain", "inverse")},
            "states": {"initial_neural": initial, "first_control_neural": first,
                                           "final_whole": {"network": final}}}


def ledger():
    return {"constructors_completed": 0, "forward_requested": 0, "forward_completed": 0,
            "forward_device_confirmed": 0, "backward_requested": 0, "backward_completed": 0,
            "backward_device_confirmed": 0, "neutral_exact": False,
            "operator_constructor_requested": 0, "operator_constructor_completed": 0,
            "model_constructor_requested": 0, "model_constructor_completed": 0}


def test_generated_native_flow_has_32_forward_four_backward_and_raw():
    raw = {}
    cost = ledger()
    result, complete = probe.compute(fake_data(), cost, device="cpu", operator_type=FakeCurrent,
                                     model_type=FakeSSP, partial=raw)
    assert result["classification"] == "supported"
    assert cost["forward_completed"] == cost["forward_requested"] == 32
    assert cost["backward_completed"] == cost["backward_requested"] == 4
    assert cost["forward_device_confirmed"] == 32 and cost["backward_device_confirmed"] == 4
    assert cost["constructors_completed"] == 1 and cost["neutral_exact"]
    assert complete["gradient"].shape == (probe.GROUPS,)
    assert np.array_equal(raw["theta_plus32"], complete["theta_plus32"])
    for prefix, scalar in (("nominal", "nominal_loss"), ("plus", "plus_loss"), ("minus", "minus_loss")):
        assert complete[prefix + "_rate"].dtype == np.float32
        assert complete[prefix + "_adaptation"].shape == (probe.N,)
        independently = complete[prefix + "_rate"][fake_data()["mask"]["population"]].astype(np.float64).mean()
        assert independently == result[scalar]


def test_prefix_failure_preserves_partial_and_stops_at_20():
    data = fake_data()
    data["states"]["first_control_neural"]["rate"][0] = 1
    raw = {}
    cost = ledger()
    with pytest.raises(ValueError, match="First20"):
        probe.compute(data, cost, device="cpu", operator_type=FakeCurrent,
                      model_type=FakeSSP, partial=raw)
    assert cost["forward_completed"] == 20 and cost["backward_requested"] == 0
    assert cost["forward_device_confirmed"] == 20
    assert "first_rate" in raw and "gradient" not in raw


def test_async_host_returned_steps_remain_unconfirmed_after_later_failure():
    class FailsInPlus(FakeSSP):
        def __init__(self, operator, *_args):
            super().__init__(operator)
            self.steps = 0

        def step(self, *args):
            self.steps += 1
            if self.steps == 26:
                raise RuntimeError("injected later host failure")
            return super().step(*args)

    cost, raw = ledger(), {}
    with pytest.raises(RuntimeError, match="injected later"):
        probe.compute(fake_data(), cost, device="cpu", operator_type=FakeCurrent,
                      model_type=FailsInPlus, partial=raw)
    assert cost["forward_requested"] == 26 and cost["forward_completed"] == 25
    assert cost["forward_device_confirmed"] == 24
    assert cost["backward_completed"] == cost["backward_device_confirmed"] == 4
    assert "nominal_rate" in raw and "gradient" in raw and "plus_rate" not in raw


def test_failed_run_publishes_confirmed_cost_and_partial_neural_arrays(tmp_path):
    class FailsInPlus(FakeSSP):
        def __init__(self, operator, *_args):
            super().__init__(operator)
            self.steps = 0

        def step(self, *args):
            self.steps += 1
            if self.steps == 26:
                raise RuntimeError("injected device continuation failure")
            return super().step(*args)

    source = tmp_path / "freeze.json"
    source.write_text("{}", encoding="utf8")
    freeze = {"source_sha256": {}, "inputs": {}, "qualification": {}}

    def calculate(data, cost, **kwargs):
        return probe.compute(data, cost, device="cpu", operator_type=FakeCurrent,
                             model_type=FailsInPlus, **kwargs)

    result = probe.run(source, tmp_path / "out", loader=lambda *_a, **_k: fake_data(),
                       calculator=calculate, gatekeeper=lambda *_a, **_k: (freeze, {"evidence_sha256": {}}),
                       meter=fake_meter)
    assert not result["run_valid"] and result["first_error"]["message"] == "injected device continuation failure"
    assert result["cost"]["forward_host_returned"] == 25
    assert result["cost"]["forward_device_confirmed"] == result["cost"]["science_forward_ticks"] == 24
    assert result["cost"]["forward_unconfirmed_host_returns"] == 1
    with np.load(tmp_path / "out/raw.npz") as raw:
        assert "nominal_rate" in raw and "gradient" in raw and "plus_rate" not in raw


def test_zero_gradient_is_completed_inconclusive_not_success():
    class ZeroResponse(FakeSSP):
        def step(self, rate, adaptation, _drive, efficacy, _mask):
            return rate + efficacy.sum() * 0., adaptation

    cost = ledger()
    result, raw = probe.compute(fake_data(), cost, device="cpu", operator_type=FakeCurrent,
                                model_type=ZeroResponse)
    assert result["classification"] == "inconclusive" and not result["supported"]
    assert cost["forward_completed"] == 32 and cost["backward_completed"] == 4
    assert not np.any(raw["direction"])


def fake_meter():
    return {"host_rss_bytes": 1, "host_peak_bytes": 1,
            "gpu_reserved_bytes": 0, "gpu_peak_reserved_bytes": 0}


def test_run_records_science_failure_partial_and_first_error(tmp_path):
    freeze = {"source_sha256": {}, "inputs": {}, "qualification": {}}
    fake_gate = lambda *_args, **_kwargs: (freeze, {"evidence_sha256": {}})
    source = tmp_path / "freeze.json"
    source.write_text("{}", encoding="utf8")

    def failing(_data, cost, *, partial, deadline, journal, checkpoint):
        assert deadline > 0
        cost["forward_requested"] = 1
        journal(cost)
        partial["partial"] = np.arange(2)
        checkpoint(partial)
        raise RuntimeError("fixture science failure")

    result = probe.run(source, tmp_path / "out", loader=lambda *_a, **_k: {}, calculator=failing,
                       gatekeeper=fake_gate, meter=fake_meter)
    assert not result["run_valid"] and result["classification"] == "not_tested"
    assert result["first_error"]["message"] == "fixture science failure"
    assert result["cost"]["forward_requested"] == 1
    assert json.loads((tmp_path / "out/advance-ledger.json").read_text("utf8"))["forward_requested"] == 1
    with np.load(tmp_path / "out/raw.npz") as raw:
        assert raw["partial"].tolist() == [0, 1]


def test_run_raw_storage_failure_invalidates_without_losing_science_cost(tmp_path):
    freeze = {"source_sha256": {}, "inputs": {}, "qualification": {}}
    fake_gate = lambda *_args, **_kwargs: (freeze, {"evidence_sha256": {}})
    source = tmp_path / "freeze.json"
    source.write_text("{}", encoding="utf8")

    def complete(_data, cost, *, partial, deadline, journal, checkpoint):
        cost.update(forward_requested=32, forward_completed=32, forward_device_confirmed=32,
                    backward_requested=4, backward_completed=4, backward_device_confirmed=4,
                    neutral_exact=True)
        partial["probe"] = np.ones(1)
        return {"classification": "supported"}, {}

    def failed_save(*_args, **_kwargs):
        raise OSError("raw disk fixture")

    result = probe.run(source, tmp_path / "out", loader=lambda *_a, **_k: {},
                       calculator=complete, saver=failed_save, gatekeeper=fake_gate, meter=fake_meter)
    assert not result["run_valid"] and result["cost"]["forward_completed"] == 32
    assert result["first_error"]["message"] == "raw disk fixture"


def test_run_manifest_failure_rewrites_result_as_invalid(tmp_path):
    freeze = {"source_sha256": {}, "inputs": {}, "qualification": {}}
    source = tmp_path / "freeze.json"
    source.write_text("{}", encoding="utf8")

    def writer(path, value):
        if path.name == "output-manifest.json":
            raise OSError("manifest disk fixture")
        probe.publish(path, value)

    def complete(_data, cost, *, partial, deadline, journal, checkpoint):
        cost.update(forward_requested=32, forward_completed=32, forward_device_confirmed=32,
                    backward_requested=4, backward_completed=4, backward_device_confirmed=4,
                    neutral_exact=True)
        return {"classification": "supported"}, {}
    with pytest.raises(OSError, match="manifest disk fixture"):
        probe.run(source, tmp_path / "out", loader=lambda *_a, **_k: {}, calculator=complete,
                  writer=writer, gatekeeper=lambda *_a, **_k: (freeze, {"evidence_sha256": {}}),
                  meter=fake_meter)
    result = json.loads((tmp_path / "out/result.json").read_text("utf8"))
    assert not result["run_valid"] and result["classification"] == "not_tested"
    assert result["first_error"]["message"] == "manifest disk fixture"


def test_generated_full_schema_loader_resolves_native_coefficient_and_ignores_old_weights(tmp_path):
    n, edges, selected = probe.N, probe.EDGES, probe.SELECTED
    positions = np.arange(selected, dtype=np.int64)
    indices = np.zeros(edges, np.int32)
    data = np.ones(edges, np.float32)
    indptr = np.full(n + 1, edges, np.int32)
    indptr[0] = 0
    population = np.arange(probe.POPULATION, dtype=np.int64)
    motor = np.arange(391, dtype=np.int64)
    csr = {"indptr": indptr, "indices": indices, "data": data,
           "selected_edge_ids": np.array([0], np.int64), "selected_values": np.ones(1, np.float32)}
    config = {"cell_gain": np.ones(n, np.float32), "membrane_tau": np.ones(n, np.float32),
              "motor_ids": motor, "actual_motor_inverse32": np.ones(391, np.float32),
              "actual_motor_gain32": np.ones(391, np.float32)}
    load = {"threshold32": np.full(n, 7.5, np.float32), "tau32": np.ones(n, np.float32)}
    mask = {"population": population, "positions": positions,
            "edge_parameters": np.arange(selected, dtype=np.int64) % probe.GROUPS,
            "source_weights": np.full(selected, 2., np.float32)}
    roster = {"premotor_ids": population.copy(), "edge_ids": positions.copy(),
              "weights32": np.ones(selected, np.float32)}
    base = {"tick": 0, "dt": probe.DT, "device": "cuda", "adaptation_gain": 8.,
            "adaptation_tau_s": .3, "model_hash":
            "f9ea25308cf5e074047bf16afba55de9fac89411ca48127f2c37cc4e6e34c7be",
            "schema": "flylab.research-rate.v1",
            "rate": np.zeros(n, np.float32), "adaptation": np.zeros(n, np.float32),
            "drive": np.zeros(n, np.float32), "output_mask": np.ones(n, np.float32),
            "synaptic_resource": np.ones(1, np.float64),
            "motor_timescale": {"inverse32": np.ones(n, np.float32),
                                "source_motor_identity":
                                "a8310c1cff7e7dfa4b71b506fa3828a29b0450d8e62ddd17b611c22ee59ca0a8"},
            "recurrent_resource": {"schema": "flylab.m3-ssprk3-recovery-resource.v1",
                                   "source_model_hash":
                                   "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"}}
    first = {**base, "tick": 20}
    final = {**base, "tick": 2000}
    arrays = dict(zip(probe.INPUTS, (csr, config, load, mask, roster), strict=True))
    states = {"initial_neural": base, "first_control_neural": first, "final_whole": {"network": final}}
    freeze = {"network_identity": base["model_hash"], "inputs": {name: {"path": name}
              for name in (*probe.INPUTS, *probe.STATES)}}
    result = probe.load_inputs(freeze, root=tmp_path,
                               npz_loader=lambda path: arrays[path.name],
                               state_loader=lambda path: states[path.name])
    assert np.array_equal(result["physiology"]["coefficient"],
                          np.full(n, np.float32(1. / 200.), np.float32))
    assert np.array_equal(result["mask"]["source_weights"], mask["source_weights"])
