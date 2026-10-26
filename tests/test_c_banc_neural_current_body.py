"""E224 generated body-protocol tests; all neural/physics work is mocked."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_neural_current_body as body


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_raw_schema_keeps_original_and_effector_distinct(tmp_path, monkeypatch):
    row = {"joint_targets_rad": np.zeros(42),
           "tendon_inputs": {key: 0. for key in body.TENDONS}}
    monkeypatch.setattr(body.protocol, "capture_control", lambda session: row)
    original = np.arange(391, dtype=np.float32)
    adapter = SimpleNamespace(activation=np.zeros(96, np.float64),
                              last_decoder_input32=np.full(391, .5, np.float32))
    network = SimpleNamespace(tick=20,
        _free_rate64=FakeTensor(np.ones(343, np.float64)),
        _free_fast64=FakeTensor(np.zeros((343, 3), np.float64)),
        _free_slow64=FakeTensor(np.zeros((343, 3), np.float64)))
    session = SimpleNamespace(rate=original, adapter=adapter, network=network,
        control_tick=1, body=SimpleNamespace(physics_time=lambda: .005))
    _, legacy = body.capture_state(session, "legacy_canonical")
    _, effector = body.capture_state(session, "effector_canonical")
    np.testing.assert_array_equal(legacy["decoder_input32"], original)
    np.testing.assert_array_equal(effector["motor_rate32"], original)
    np.testing.assert_array_equal(effector["decoder_input32"], .5)
    assert not set(body.EFFECTOR_FIELDS) & set(legacy)
    assert set(body.EFFECTOR_FIELDS) <= set(effector)
    body.save_control(tmp_path / "legacy.npz", [legacy], "legacy_canonical")
    body.save_control(tmp_path / "effector.npz", [effector], "effector_canonical")
    with np.load(tmp_path / "legacy.npz") as raw:
        assert not set(body.EFFECTOR_FIELDS) & set(raw.files)
    with np.load(tmp_path / "effector.npz") as raw:
        assert raw["free_effector64"].shape == (1, 343)


def test_review_gate_rejects_before_output_or_model(tmp_path, monkeypatch):
    freeze_path, review_path = tmp_path / "freeze.json", tmp_path / "review.json"
    output = tmp_path / "missing-output"
    freeze_path.write_text("{}", encoding="utf-8")
    review_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(body, "sha256", lambda path: body.FREEZE_SHA
                        if path == freeze_path else "review")
    with pytest.raises(ValueError, match="contract"):
        body.gate(freeze_path, body.FREEZE_SHA, review_path, "review", output)
    assert not output.exists()


def fake_two_arm_run(tmp_path, monkeypatch, first_failure="health"):
    out = tmp_path / "e224-fake"
    freeze = {"paths": {key: key for key in (
        "graph", "start", "parameters", "prototypes", "candidate")},
        "graph_hash": "original", "resource_limits": {
            "resident_bytes": 1 << 40, "allocated_gpu_bytes": 1 << 40,
            "output_bytes": 1 << 40}}
    monkeypatch.setattr(body, "gate", lambda *args: (freeze, {"approved": True}))
    monkeypatch.setattr(body, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(body.protocol.GraphStore, "load", lambda path:
                        SimpleNamespace(hash="original", n=158706))
    class FakeStore:
        @staticmethod
        def load(path):
            return {"schema": "E150"}

        @staticmethod
        def save(path, value):
            path.mkdir()
            (path / "state.json").write_text(json.dumps(value))

    monkeypatch.setattr(body.protocol, "StateStore", FakeStore)
    monkeypatch.setattr(body, "load_npz", lambda path: {})
    monkeypatch.setattr(body, "load_candidate_beta", lambda path:
                        np.zeros((96, 64), np.float64))
    monkeypatch.setattr(body.protocol, "profiled_construction",
                        lambda label, callback, rows, persist: callback())
    monkeypatch.setattr(body, "unwrap_neural_current_checkpoint", lambda saved:
                        ({"identical_E150_start": True}, {}, None))
    import tools.fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (
        SimpleNamespace(restore_original_limits=lambda: None), {}))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1, peak_wset=1)))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=lambda: (1 << 40, 1 << 40)))))
    starts = []
    sample = (0., np.ones(6), np.zeros(6), np.ones(6, bool), False, 1.)

    def observer(physical, values):
        physical.sink = values
        return lambda: setattr(physical, "sink", None)

    monkeypatch.setattr(body.e113, "install_native_observer", observer)

    class FakeSession:
        def __init__(self, mode):
            starts.append(mode)
            self.mode = mode
            self.control_tick = 0
            self.network = SimpleNamespace(n=158706, tick=0, identity=mode)
            self.body = SimpleNamespace(t=0., fault=False, sink=None,
                                        physics_time=lambda: self.body.t)
            self.adapter = SimpleNamespace(identity="canonical",
                                           activation=np.zeros(96))
            self.rate = np.zeros(391, np.float32)
            self.fault = False

        def snapshot(self):
            return {"mode": self.mode, "control_tick": self.control_tick}

        def advance(self, controls):
            assert controls == 1
            if self.mode == "legacy_canonical" and self.control_tick == 1:
                if first_failure == "health":
                    raise body.e113.NativeHealthStop("generated upright stop")
                if first_failure in ("numeric", "numeric_inf"):
                    bad_value = float("nan" if first_failure == "numeric" else "inf")
                    self.body.t = bad_value
                    self.body.sink.append((bad_value, *sample[1:5], bad_value))
                    raise body.e113.NativeHealthStop("generated numeric stop")
                if first_failure == "resource":
                    raise TimeoutError("generated resource stop")
                if first_failure == "technical":
                    raise RuntimeError("generated technical stop")
            self.control_tick += 1
            self.network.tick += 20
            for _ in range(50):
                self.body.t += .0001
                self.body.sink.append((self.body.t, *sample[1:]))

        def close(self):
            pass

    monkeypatch.setattr(body.NeuralCurrentTrial, "from_e150_checkpoint",
                        lambda graph, saved, parameters, prototypes, beta, mode:
                        FakeSession(mode))

    def fake_capture(session, mode):
        row = {"time_s": session.control_tick * .005,
               "body": {"position": [session.control_tick * .005, 0., 0.]},
               "fault": False}
        value = {"control_tick": session.control_tick,
                 "network_tick": session.network.tick,
                 "physics_time_s": session.body.t,
                 "motor_rate32": np.zeros(391, np.float32),
                 "decoder_input32": np.zeros(391, np.float32),
                 "activation": np.zeros(96), "command": np.zeros(42),
                 "tendon": np.zeros(6)}
        if mode == "effector_canonical":
            value.update(free_effector64=np.zeros(343),
                         h_fast=np.zeros((343, 3)), h_slow=np.zeros((343, 3)))
        return row, value

    monkeypatch.setattr(body, "capture_state", fake_capture)
    result = body.run(tmp_path / "freeze", body.FREEZE_SHA,
                      tmp_path / "review", "review", out)
    return out, result, starts


def test_fake_two_arm_health_stop_continues_and_seals_raw(tmp_path, monkeypatch):
    out, result, starts = fake_two_arm_run(tmp_path, monkeypatch)
    assert starts == list(body.MODES)
    assert result["validity"] == "completed_two_arm_diagnostic"
    assert result["both_healthy_3s"] is False
    assert result["next_phase_condition_met"] is True
    assert result["arms"]["legacy_canonical"]["validity"] == (
        "measured_native_health_stop")
    assert result["arms"]["effector_canonical"]["validity"] == (
        "healthy_3s_diagnostic")
    assert (out / "legacy_canonical/checkpoint-5ms").is_dir()
    assert not (out / "legacy_canonical/final").exists()
    assert (out / "effector_canonical/checkpoint-5ms").is_dir()
    assert (out / "effector_canonical/final").is_dir()
    assert not (out / "clone").exists() and not (out / "cut").exists()
    with np.load(out / "legacy_canonical/control-raw.npz") as raw:
        assert raw["motor_rate32"].shape[0] == 2
        assert "free_effector64" not in raw.files
    with np.load(out / "effector_canonical/control-raw.npz") as raw:
        assert raw["free_effector64"].shape == (601, 343)
    manifest = json.loads((out / "output-manifest.json").read_text())
    assert all(body.sha256(out / name) == item["sha256"]
               for name, item in manifest.items())


@pytest.mark.parametrize("failure", ["numeric", "numeric_inf"])
def test_fake_numeric_health_stop_keeps_raw_and_unknown_clock(tmp_path, monkeypatch, failure):
    out, result, starts = fake_two_arm_run(tmp_path, monkeypatch, failure)
    assert starts == list(body.MODES)
    arm = result["arms"]["legacy_canonical"]
    assert arm["validity"] == "measured_native_health_stop"
    assert arm["numeric_fault"] is True
    assert arm["nonfinite_native_fields"] == ["time_s", "upright"]
    assert arm["last_native_time_s"] is None
    assert arm["minimum_native_upright"] is None
    assert arm["unavailable_nonfinite_fields"] == [
        "$.last_native_time_s", "$.minimum_native_upright"]
    assert result["arms"]["effector_canonical"]["validity"] == (
        "healthy_3s_diagnostic")
    with np.load(out / "legacy_canonical/native-contact.npz") as native:
        is_bad = np.isnan if failure == "numeric" else np.isposinf
        assert is_bad(native["time_s"][-1])
        assert is_bad(native["upright"][-1])
    arm_ledger = json.loads((out / "legacy_canonical/advance-ledger.json").read_text())
    assert arm_ledger["actual_physical_s"] is None
    assert arm_ledger["clock_valid"] is False
    assert "$.clock_events[1].physics_time_after_s" in (
        arm_ledger["unavailable_nonfinite_fields"])
    aggregate = json.loads((out / "advance-ledger.json").read_text())
    assert aggregate["actual_physical_s"] is None
    assert aggregate["unknown_physical_intervals"] == 1
    assert aggregate["confirmed_physical_s"] > 0
    saved = json.loads((out / "legacy_canonical/result.json").read_text())
    assert saved == arm
    manifest = json.loads((out / "output-manifest.json").read_text())
    assert all(body.sha256(out / name) == item["sha256"]
               for name, item in manifest.items())


@pytest.mark.parametrize(("failure", "expected"), [
    ("resource", "resource_limited"), ("technical", "technical_failure")])
def test_fake_non_health_failure_aborts_second_arm(
        tmp_path, monkeypatch, failure, expected):
    with pytest.raises((TimeoutError, RuntimeError), match="generated"):
        fake_two_arm_run(tmp_path, monkeypatch, first_failure=failure)
    out = tmp_path / "e224-fake"
    result = json.loads((out / "result.json").read_text())
    assert result["validity"] == expected
    assert result["arms"]["legacy_canonical"]["validity"] == expected
    assert "effector_canonical" not in result["arms"]
    assert (out / "legacy_canonical/control-raw.npz").exists()
    assert (out / "legacy_canonical/native-contact.npz").exists()
    assert (out / "output-manifest.json").exists()
