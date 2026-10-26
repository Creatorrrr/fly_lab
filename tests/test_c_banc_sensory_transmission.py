"""Synthetic checks only; none constructs the graph, body, or CUDA network."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tools import observe_c_banc_m3_sensory_transmission as observer


def test_temporary_method_restores_absent_and_existing_slots():
    class Owner:
        def method(self):
            return "class"

    owner = Owner()
    undo = observer.temporary_method(owner, "method", lambda: "wrapped")
    assert owner.method() == "wrapped"
    undo()
    assert owner.method() == "class"
    assert "method" not in vars(owner)
    owner.method = lambda: "instance"
    prior = owner.method
    undo = observer.temporary_method(owner, "method", lambda: "wrapped")
    undo()
    assert owner.method is prior


def fake_collector():
    class Body:
        def __init__(self):
            self.leg_calls = 0
            self.knee_calls = 0

        def physics_time(self):
            return 0.0

        def leg_observation(self):
            self.leg_calls += 1
            return {"angles_rad": np.zeros(42), "velocities_rad_s": np.zeros(42),
                    "support_load_bw": np.zeros(6), "non_support_load_bw": np.zeros(6)}

        def knee_kinematics(self, *, include_direction=True):
            self.knee_calls += 1
            assert include_direction is False
            return {"flexion_rad": np.zeros(6), "velocity_rad_s": np.zeros(6)}

    body = Body()
    adapter = SimpleNamespace(body=body, filtered=np.zeros(2),
                              velocity_lowpass=np.zeros(42), last_features=np.zeros(2))
    session = SimpleNamespace(control_tick=0, network=SimpleNamespace(tick=0))
    collector = observer.Collector({})
    collector.adapter, collector.session = adapter, session
    collector.unique_ids = np.array([1, 3])
    monkey_n = observer.N

    def original():
        leg = body.leg_observation()
        knee = body.knee_kinematics(include_direction=False)
        assert leg["angles_rad"].shape == (42,) and knee["flexion_rad"].shape == (6,)
        adapter.last_features[:] = (0.2, 0.7)
        adapter.filtered[:] = (0.1, 0.3)
        drive = np.zeros(monkey_n, np.float32)
        drive[collector.unique_ids] = (0.5, 1.0)
        return drive

    return collector, body, adapter, original


def test_encode_capture_calls_once_returns_same_object_and_restores():
    collector, body, adapter, original = fake_collector()
    result = collector.capture_encode(original)
    assert result.dtype == np.float32 and result.shape == (observer.N,)
    assert body.leg_calls == body.knee_calls == 1
    assert "leg_observation" not in vars(body)
    assert "knee_kinematics" not in vars(body)
    row = collector.encoder[0]
    assert row["complete"] is True
    np.testing.assert_array_equal(row["returned_drive"], [0.5, 1.0])
    np.testing.assert_array_equal(row["old_filtered"], [0, 0])
    np.testing.assert_array_equal(row["new_filtered"], [0.1, 0.3])
    assert adapter.last_features[0] == .2


def test_encode_exception_keeps_primary_and_partial_record():
    collector, body, _, _ = fake_collector()
    expected = RuntimeError("original failure")

    def failure():
        body.leg_observation()
        raise expected

    with pytest.raises(RuntimeError) as caught:
        collector.capture_encode(failure)
    assert caught.value is expected
    assert body.leg_calls == 1 and body.knee_calls == 0
    assert "leg_observation" not in vars(body)
    assert "knee_kinematics" not in vars(body)
    assert collector.encoder[0]["complete"] is False
    assert "leg_observation" in collector.encoder[0]
    encoder, neural = collector.arrays()
    assert encoder["angles_rad_available"].tolist() == [True]
    assert encoder["flexion_rad_available"].tolist() == [False]
    assert neural["complete"].size == 0


def test_float32_repeated_index_expansion():
    ids = np.array([1, 1, 2])
    port_index = np.array([0, 1, 2])
    old = np.array([0.1, 0.2, 0.3], np.float64)
    actual = np.zeros(4, np.float32)
    np.add.at(actual, ids, old[port_index])
    expected = np.zeros(4, np.float32)
    expected[1] = np.float32(np.float32(0.1) + 0.2)
    expected[2] = np.float32(0.3)
    assert actual.tobytes() == expected.tobytes()


def test_filter_and_delayed_drive_synthetic(monkeypatch):
    monkeypatch.setattr(observer, "CONTROLS", 2)
    monkeypatch.setattr(observer, "ROWS", 3)
    monkeypatch.setattr(observer, "N", 8)
    body = SimpleNamespace(neutral=np.zeros(42))
    adapter = SimpleNamespace(
        joints=np.arange(42).reshape(6, 7), body=body,
        coxa_sign=np.ones(6), _receptor_keys=[(0, "load", "x")],
        _port_features=np.array([0]), _sensory_unique=False,
        sensory_gain=18., spec={"load_half_bw": .5, "touch_half_bw": .05})
    collector = observer.Collector({})
    collector.adapter = adapter
    collector.unique_ids = np.array([1])
    collector.provenance = {"ports": [{"port": 0}],
        "port_expansion_indices": [1, 1], "port_expansion_ports": [0, 0]}
    encoder = {"control_tick": np.arange(2), "network_tick": np.array([0, 20]),
               "time_s": np.array([0., .005]), "complete": np.ones(2, bool),
               "angles_rad": np.zeros((2, 42)), "velocities_rad_s": np.zeros((2, 42)),
               "encoded_angles_rad": np.zeros((2, 42)),
               "encoded_velocities_rad_s": np.zeros((2, 42)),
               "flexion_rad": np.zeros((2, 6)), "flexion_velocity_rad_s": np.zeros((2, 6)),
               "support_load_bw": np.ones((2, 6)),
               "non_support_load_bw": np.zeros((2, 6)),
               "old_filtered": np.zeros((2, 1)),
               "new_filtered": np.zeros((2, 1)),
               "old_velocity_lowpass": np.zeros((2, 42)),
               "new_velocity_lowpass": np.zeros((2, 42)),
               "last_features": np.ones((2, 1)) * (1 / 1.5),
               "request": np.ones((2, 1)) * 12.,
               "returned_drive": np.zeros((2, 1), np.float32)}
    alpha = 1-math.exp(-.005/.02)
    encoder["new_filtered"][0, 0] = alpha * 12
    encoder["old_filtered"][1, 0] = alpha * 12
    encoder["new_filtered"][1, 0] = alpha * 12 + alpha * (12-alpha * 12)
    encoder["returned_drive"][1, 0] = np.float32(np.float32(alpha * 12) + alpha * 12)
    neural = {"control_tick": np.arange(3), "network_tick": np.array([0, 20, 40]),
              "time_s": np.array([0., .005, .01]), "complete": np.ones(3, bool),
              "drive": np.array([[0.], [0.], [encoder["returned_drive"][1, 0]]], np.float32)}
    assert observer.validate_signals(collector, encoder, neural)[
        "returned_and_delayed_drive_bit_exact"] is True
    neural["drive"][2, 0] = 0
    with pytest.raises(ValueError, match="delayed sensory"):
        observer.validate_signals(collector, encoder, neural)


def test_stale_parent_receipt_rejected(monkeypatch, tmp_path):
    receipt = {"schema": "flylab.m3.e193.parent-input-validation.v1",
               "validated": True, "freeze_sha256": "f", "source_review_sha256": "r",
               "metadata_sha256": "m", "input_sha256": {"a": "b"},
               "parent": {"pid": os.getppid(), "create_time": 1.},
               "issued_monotonic_s": 1., "input_stamps": {"a": [1, 2, 3, 4]}}
    monkeypatch.setattr(observer, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(observer, "sha256", lambda _: "x" * 64)
    monkeypatch.setattr(observer, "read_json", lambda _: receipt)
    monkeypatch.setattr(observer.time, "perf_counter", lambda: 62.)
    monkeypatch.setattr(observer.os, "getppid", lambda: 123)
    receipt["parent"]["pid"] = 123

    class Process:
        def __init__(self, pid):
            assert pid == 123

        def is_running(self):
            return True

        def create_time(self):
            return 1.

        def cwd(self):
            return str(tmp_path)

        def cmdline(self):
            return ["python", "parent-e193-supervise.py"]

    monkeypatch.setattr("psutil.Process", Process)
    monkeypatch.setattr(observer, "SUPERVISOR", "parent-e193-supervise.py")
    with pytest.raises(ValueError, match="stale parent receipt"):
        observer.validate_parent_inputs({"a": "b"}, "f", "r", "m", "x" * 64)


def test_gate_rejects_output_collision_before_hashing(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="output exists"):
        observer.gate(tmp_path / "missing-freeze", tmp_path / "missing-review", output)


def test_cleanup_failures_fail_success_but_do_not_hide_primary(monkeypatch):
    actual = observer.temporary_method
    restores = []

    def failing_restore(owner, name, wrapper):
        undo = actual(owner, name, wrapper)

        def fail():
            undo()
            restores.append(name)
            raise RuntimeError("cleanup " + name)

        return fail

    monkeypatch.setattr(observer, "temporary_method", failing_restore)
    collector, body, _, original = fake_collector()
    with pytest.raises(RuntimeError, match="cleanup knee_kinematics"):
        collector.capture_encode(original)
    assert restores == ["knee_kinematics", "leg_observation"]
    assert "leg_observation" not in vars(body) and "knee_kinematics" not in vars(body)
    assert collector.encoder[0]["complete"] is False
    encoder, _ = collector.arrays()
    assert "cleanup" in encoder["cleanup_error"][0]
    assert "cleanup" in encoder["error"][0]

    restores.clear()
    collector, body, _, _ = fake_collector()
    primary = LookupError("model primary")

    def failure():
        raise primary

    with pytest.raises(LookupError) as caught:
        collector.capture_encode(failure)
    assert caught.value is primary
    assert restores == ["knee_kinematics", "leg_observation"]
    encoder, _ = collector.arrays()
    assert "model primary" in encoder["error"][0]
    assert "cleanup" in encoder["cleanup_error"][0]


def test_partial_arrays_keep_valid_prefix_and_both_archives(tmp_path):
    collector = observer.Collector({})
    collector.unique_ids = np.array([1])
    collector.provenance = {"ports": [{"port": 0}]}
    collector.encoder = [
        {"control_tick": 0, "complete": True, "network_tick": 0,
         "old_filtered": np.array([.25]),
         "leg_observation": {"angles_rad": np.zeros(42)}},
        {"control_tick": 1, "complete": True, "network_tick": 20,
         "old_filtered": np.array([1., 2.]),
         "leg_observation": {"angles_rad": np.zeros(4)}}]
    collector.neural = [{"control_tick": 0, "complete": True, "network_tick": 0,
                         "rate": np.array([2.], np.float32)}]
    encoder, neural = collector.arrays()
    assert encoder["old_filtered_available"].tolist() == [True, False]
    assert encoder["angles_rad_available"].tolist() == [True, False]
    assert encoder["complete"].tolist() == [True, False]
    assert "shape" in encoder["error"][1]
    assert neural["rate_available"].tolist() == [True]
    np.savez_compressed(tmp_path / "encoder.npz", **encoder)
    np.savez_compressed(tmp_path / "neural.npz", **neural)
    with np.load(tmp_path / "encoder.npz", allow_pickle=False) as saved:
        assert saved["old_filtered"][0, 0] == .25
        assert saved["old_filtered_available"].tolist() == [True, False]
    with np.load(tmp_path / "neural.npz", allow_pickle=False) as saved:
        assert saved["rate"][0, 0] == 2


def test_tensor_network_anchor_next_endpoint_and_factory_restore(monkeypatch):
    import torch

    monkeypatch.setattr(observer, "N", 8)

    class Body:
        model_hash = "body"

        def __init__(self):
            self.neutral = np.zeros(42)
            self.time = 0.
            self.leg_calls = 0
            self.knee_calls = 0

        def physics_time(self):
            return self.time

        def leg_observation(self):
            self.leg_calls += 1
            return {"angles_rad": np.zeros(42), "velocities_rad_s": np.zeros(42),
                    "support_load_bw": np.ones(6), "non_support_load_bw": np.zeros(6)}

        def knee_kinematics(self, *, include_direction=True):
            self.knee_calls += 1
            return {"flexion_rad": np.zeros(6), "velocity_rad_s": np.zeros(6)}

    graph = SimpleNamespace(n=8, hash="graph", nodes=[{"id": str(i)} for i in range(8)])
    body = Body()
    from flylab.c.rate_body import RateBodyAdapter

    adapter = object.__new__(RateBodyAdapter)
    adapter.graph, adapter.body, adapter.identity = graph, body, "adapter"
    adapter.sensory_gain = 18.
    adapter.load_gain_multiplier = 1.
    adapter.front_hair_plate = False
    adapter.ports = [(0, "load", "x", np.array([3])),
                     (1, "load", "x", np.array([1]))]
    adapter._sensory_ids = np.array([3, 1])
    adapter._sensory_ports = np.array([0, 1])
    adapter.joints = np.arange(42).reshape(6, 7)
    adapter.coxa_sign = adapter.knee_sign = np.ones(6)
    adapter.spec = {"rows": [], "load_half_bw": .5, "touch_half_bw": .05}
    adapter.filtered = np.zeros(2)
    adapter.velocity_lowpass = np.zeros(42)
    adapter.last_features = np.zeros(2)

    def encode(*, sensory_cut=False):
        assert sensory_cut is False
        body.leg_observation()
        body.knee_kinematics(include_direction=False)
        drive = np.zeros(8, np.float32)
        drive[[3, 1]] = adapter.filtered
        adapter.filtered[:] = [.25, .5]
        adapter.last_features[:] = [.2, .4]
        return drive

    adapter.encode = encode
    parameter = lambda shift: torch.arange(8, dtype=torch.float32) + shift
    network = SimpleNamespace(n=8, device="cuda", cuda_implementation="packed",
        identity="network", tick=0, rate=torch.zeros(8), drive=torch.zeros(8),
        adaptation=torch.zeros(8), output_mask=torch.ones(8),
        cap=parameter(200), coefficient=parameter(1), threshold=parameter(7),
        tau=parameter(20), adaptation_gain=4., adaptation_tau_s=.15,
        torch=SimpleNamespace(as_tensor=lambda x, device: torch.as_tensor(x), equal=torch.equal),
        _current=lambda rate: torch.zeros_like(rate))
    session = SimpleNamespace(adapter=adapter, graph=graph, body=body, network=network,
                              control_tick=0, descending=np.array([7]), navigation=None)
    metadata = {"expected_graph_hash": "graph", "expected_network_identity": "network",
                "expected_adapter_identity": "adapter", "expected_body_hash": "body"}
    collector = observer.Collector(metadata)
    sentinel = object()
    callback_calls = []

    def original_factory(*args, **kwargs):
        def callback(s, detailed, available):
            callback_calls.append(s.control_tick)
            return sentinel
        return callback

    e150 = SimpleNamespace(make_observer=original_factory)
    restore = observer.install_factory(SimpleNamespace(e150=e150), collector)
    callback = e150.make_observer()
    assert callback(session, [], {}) is sentinel
    assert callback_calls == [0]
    assert collector.provenance["unique_sensory_indices"] == [1, 3]
    assert collector.provenance["network_parameters"]["cap"] == [201., 203.]
    assert collector.provenance["network_parameters"]["threshold"] == [8., 10.]
    assert "encode" in vars(adapter)
    returned = adapter.encode(sensory_cut=False)
    assert body.leg_calls == body.knee_calls == 1
    assert "leg_observation" not in vars(body)
    assert "knee_kinematics" not in vars(body)
    session.control_tick, network.tick, body.time = 1, 20, .005
    network.drive = torch.as_tensor(returned)
    assert callback(session, [], {}) is sentinel
    encoder, neural = collector.arrays()
    assert encoder["complete"].tolist() == [True]
    assert neural["complete"].tolist() == [True, True]
    np.testing.assert_array_equal(neural["drive"][1], returned[[1, 3]])
    network.tick = 19
    with pytest.raises(ValueError, match="neural endpoint clock"):
        callback(session, [], {})
    restore()
    assert e150.make_observer is original_factory
    assert adapter.encode is encode
    assert callback_calls == [0, 1, 1]

    def broken_factory(*args, **kwargs):
        def callback(s, detailed, available):
            raise LookupError("original callback")
        return callback

    e150.make_observer = broken_factory
    other = observer.Collector(metadata)
    restore = observer.install_factory(SimpleNamespace(e150=e150), other)
    with pytest.raises(LookupError, match="original callback"):
        e150.make_observer()(session, [], {})
    assert other.neural == [] and other.encoder == []
    restore()
    assert e150.make_observer is broken_factory


def test_generated_gate_rejects_owned_source_mutation(monkeypatch, tmp_path):
    for key, name in {"DESIGN": "design.md", "METADATA": "metadata.json",
                      "FREEZE": "freeze.json", "REVIEW": "review.json",
                      "SUPERVISOR": "supervisor.py", "OUTPUT": "output"}.items():
        monkeypatch.setattr(observer, key, name)
    monkeypatch.setattr(observer, "OWNED", ("observer.py", "test.py"))
    monkeypatch.setattr(observer, "PARAMETERS", {"cumulative_before_s": 2.,
                                                  "cumulative_cap_s": 10.})
    monkeypatch.setattr(observer, "safe", lambda name: tmp_path / name)
    for name in ("design.md", "supervisor.py", "observer.py", "test.py", "input.bin",
                 "old-freeze.json", "old-review.json"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    digest = lambda name: observer.sha256(tmp_path / name)
    metadata = {"schema": "flylab.m3.e193.sensory-transmission.metadata.v1",
        "design_path": "design.md", "design_sha256": digest("design.md"),
        "owned_files": ["observer.py", "test.py"], "original_native_runs": 0,
        "F6_exposed": 0, "candidate_adopted": False,
        "cumulative_before_s": 2., "cumulative_limit_s": 10.,
        "reused_sources": {}, "inputs": {"input.bin": digest("input.bin")},
        "e181_gate": {"freeze_path": "old-freeze.json",
                      "freeze_sha256": digest("old-freeze.json"),
                      "source_review_path": "old-review.json",
                      "source_review_sha256": digest("old-review.json")},
        **{key: key for key in ("graph_path", "neutral_checkpoint", "startup_path",
                                 "baseline_directory", "expected_graph_hash",
                                 "expected_network_identity", "expected_adapter_identity",
                                 "expected_body_hash")}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    sources = {name: digest(name) for name in ("observer.py", "test.py")}
    freeze = {"schema": "flylab.m3.e193.sensory-transmission.freeze.v1",
        "design_path": "design.md", "design_sha256": digest("design.md"),
        "metadata_path": "metadata.json", "metadata_sha256": digest("metadata.json"),
        "output_path": "output", "parameters": observer.PARAMETERS,
        "parent_supervisor_sha256": digest("supervisor.py"),
        "source_hashes": sources, "input_sha256": metadata["inputs"]}
    (tmp_path / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    review = {"schema": "flylab.m3.e193.sensory-transmission.source-review.v1",
        "approved": True, "blocking_defects": [], "freeze_sha256": digest("freeze.json"),
        "source_hashes": sources, "design_sha256": digest("design.md"),
        "metadata_sha256": digest("metadata.json")}
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    assert observer.gate(tmp_path / "freeze.json", tmp_path / "review.json",
                         tmp_path / "output")[0] == metadata
    (tmp_path / "observer.py").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed: observer.py"):
        observer.gate(tmp_path / "freeze.json", tmp_path / "review.json",
                      tmp_path / "output")
