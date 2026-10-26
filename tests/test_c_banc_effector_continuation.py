"""Generated CPU-only E225 seam, combination, gate, and failure receipts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_effector_continuation as e225


def generated_prefix():
    control = {
        "control_tick": np.arange(601, dtype=np.int64),
        "network_tick": np.arange(601, dtype=np.int64) * 20,
        "physics_time_s": np.arange(601, dtype=np.float64) * .005,
        "motor_rate32": np.zeros((601, 391), np.float32),
        "decoder_input32": np.zeros((601, 391), np.float32),
        "activation": np.zeros((601, 96), np.float64),
        "command": np.zeros((601, 42), np.float64),
        "tendon": np.zeros((601, 6), np.float64),
        "free_effector64": np.zeros((601, 343), np.float64),
        "h_fast": np.zeros((601, 343, 3), np.float64),
        "h_slow": np.zeros((601, 343, 3), np.float64),
    }
    native = {"time_s": np.arange(1, 30001, dtype=np.float64) * .0001,
              "normal_load_bw": np.ones((30000, 6), np.float64),
              "slip_mm_s": np.zeros((30000, 6), np.float64),
              "slip_valid": np.ones((30000, 6), bool),
              "nonfoot_contact": np.zeros(30000, bool),
              "upright": np.ones(30000, np.float64)}
    rows = [{"control_tick": i, "time_s": float(i * .005),
             "joint_targets_rad": [0.] * 42,
             "motor_rates": [0.] * 391,
             "tendon_inputs": {key: 0. for key in e225.prior.TENDONS}}
            for i in range(601)]
    return control, native, rows


def generated_suffix(control, rows):
    suffix_control = [{key: control[key][-1].copy() for key in e225.CONTROL_KEYS}]
    suffix_rows = [rows[-1].copy()]
    suffix_native = []
    for i in range(601, 2001):
        value = {key: control[key][-1].copy() for key in e225.CONTROL_KEYS}
        value["control_tick"] = np.int64(i)
        value["network_tick"] = np.int64(i * 20)
        value["physics_time_s"] = np.float64(i * .005)
        suffix_control.append(value)
        suffix_rows.append({**rows[-1], "control_tick": i,
                            "time_s": float(i * .005)})
        for step in range(50):
            t = 3. + (i - 601) * .005 + (step + 1) * .0001
            suffix_native.append((t, np.ones(6), np.zeros(6),
                                  np.ones(6, bool), False, 1.))
    return suffix_control, suffix_native, suffix_rows


def test_prefix_and_combined_seam_keep_all_samples(tmp_path):
    control, native, rows = generated_prefix()
    control_path, native_path, trace_path = (
        tmp_path / "control.npz", tmp_path / "native.npz", tmp_path / "trace.jsonl")
    np.savez(control_path, **control)
    np.savez(native_path, **native)
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = e225.read_prefix(control_path, native_path, trace_path)
    suffix = generated_suffix(control, rows)
    merged_control, merged_native, merged_rows = e225.combine_raw(*source, *suffix)
    assert merged_control["control_tick"].shape == (2001,)
    assert merged_control["motor_rate32"].dtype == np.float32
    assert np.array_equal(merged_control["command"][:601], control["command"])
    assert np.array_equal(merged_native["time_s"][:30000], native["time_s"])
    assert merged_native["time_s"][30000] == pytest.approx(3.0001)
    assert merged_native["time_s"].shape == (100000,)
    assert len(merged_rows) == 2001
    bad_suffix = (suffix[0], suffix[1], [dict(suffix[2][0], time_s=2.9),
                                         *suffix[2][1:]])
    with pytest.raises(AssertionError, match="trace seam"):
        e225.combine_raw(*source, *bad_suffix)
    wrong = {key: value.copy() for key, value in native.items()}
    wrong["time_s"][300] += .01
    np.savez(native_path, **wrong)
    with pytest.raises(ValueError, match="prefix clock"):
        e225.read_prefix(control_path, native_path, trace_path)


def test_gate_rejects_source_before_output(tmp_path, monkeypatch):
    freeze_path, review_path = tmp_path / "freeze.json", tmp_path / "review.json"
    freeze_path.write_text("{}")
    review_path.write_text("{}")
    monkeypatch.setattr(e225, "sha256", lambda path: e225.FREEZE_SHA
                        if path == freeze_path else "review")
    with pytest.raises(ValueError, match="contract"):
        e225.gate(freeze_path, e225.FREEZE_SHA, review_path, "review",
                  tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()


def fake_run(tmp_path, monkeypatch, *, failure=None):
    out = tmp_path / "continuation"
    prefix_control, prefix_native, prefix_rows = generated_prefix()
    freeze = {"paths": {key: key for key in (
        "graph", "checkpoint", "prefix_control", "prefix_native",
        "prefix_trace", "targets", "baseline")},
        "graph_hash": "graph", "startup_limits": {},
        "resource_limits": {"resident_bytes": 1 << 40,
            "allocated_gpu_bytes": 1 << 40, "output_bytes": 1 << 40}}
    monkeypatch.setattr(e225, "gate", lambda *args: (freeze, {"approved": True}))
    monkeypatch.setattr(e225, "safe", lambda name: tmp_path / name)
    (tmp_path / "prefix_control").write_bytes(b"generated prefix control")
    (tmp_path / "prefix_native").write_bytes(b"generated prefix native")
    monkeypatch.setattr(e225, "read_prefix", lambda *args: (
        prefix_control, prefix_native, prefix_rows))
    monkeypatch.setattr(e225, "load_npz", lambda path: {})
    monkeypatch.setattr(e225.grouped, "startup_screen", lambda *args: {"pass": True})
    monkeypatch.setattr(e225.protocol.GraphStore, "load", lambda path:
                        SimpleNamespace(hash="graph", n=158706))
    class FakeStore:
        @staticmethod
        def load(path):
            return {"same": True}

        @staticmethod
        def save(path, saved):
            path.mkdir()
            (path / "state.json").write_text(json.dumps(saved))

    monkeypatch.setattr(e225.protocol, "StateStore", FakeStore)
    monkeypatch.setattr(e225.protocol, "profiled_construction",
                        lambda label, callback, rows, persist: callback())
    import tools.fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (
        SimpleNamespace(restore_original_limits=lambda: None), {}))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1, peak_wset=1)))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=lambda: (1 << 40, 1 << 40)))))

    class FakeSession:
        mode = "effector_canonical"
        fault = False

        def __init__(self):
            self.control_tick = 600
            self.network = SimpleNamespace(n=158706, tick=12000, identity="net")
            self.body = SimpleNamespace(t=3., sink=None, fault=False,
                                        model_hash="body",
                                        physics_time=lambda: self.body.t)
            self.adapter = SimpleNamespace(identity="adapter")

        def snapshot(self):
            return {"same": True}

        def advance(self, count):
            assert count == 1
            if failure is not None and self.control_tick == 601:
                if failure == "health":
                    self.body.t = float("nan")
                    self.body.sink.append((float("nan"), np.ones(6), np.zeros(6),
                                           np.ones(6, bool), False, float("nan")))
                    raise e225.e113.NativeHealthStop("generated numeric health")
                if failure == "resource":
                    raise TimeoutError("generated timeout")
                raise RuntimeError("generated technical")
            self.control_tick += 1
            self.network.tick += 20
            for _ in range(50):
                self.body.t += .0001
                self.body.sink.append((self.body.t, np.ones(6), np.zeros(6),
                                       np.ones(6, bool), False, 1.))

        def close(self):
            pass

    monkeypatch.setattr(e225.NeuralCurrentTrial, "from_checkpoint",
                        lambda graph, saved: FakeSession())
    monkeypatch.setattr(e225.e113, "install_native_observer",
                        lambda physical, values: (
                            setattr(physical, "sink", values) or
                            (lambda: setattr(physical, "sink", None))))

    def capture(session, mode):
        row = {**prefix_rows[-1], "control_tick": session.control_tick,
               "time_s": float(session.control_tick * .005)}
        value = {key: prefix_control[key][-1].copy() for key in e225.CONTROL_KEYS}
        value["control_tick"] = np.int64(session.control_tick)
        value["network_tick"] = np.int64(session.network.tick)
        value["physics_time_s"] = np.float64(session.body.t)
        return row, value

    monkeypatch.setattr(e225.prior, "capture_state", capture)
    monkeypatch.setattr(e225.gait, "load_model", lambda path: {})
    monkeypatch.setattr(e225.gait, "analyze", lambda model: ({}, None, None))
    monkeypatch.setattr(e225.e113, "supported_slip", lambda *args, **kw: {})
    monkeypatch.setattr(e225.e113, "body_quality", lambda *args, **kw: {
        "complete_10s_native_health": True,
        "development_screens": {"F3_numeric": False,
                                "F4_numeric": False, "F5_numeric": False}})
    monkeypatch.setattr(e225.grouped, "baseline_comparison", lambda *args: {
        "comparison_fields": [], "candidate_descriptive": {}})
    result = e225.run(tmp_path / "freeze", e225.FREEZE_SHA,
                      tmp_path / "review", "review", out)
    return out, result


def test_fake_complete_continuation_seals_exact_combined_raw(tmp_path, monkeypatch):
    out, result = fake_run(tmp_path, monkeypatch)
    assert result["validity"] == "valid_full_body_diagnostic"
    assert result["full_10s"] is True
    assert result["restore_exact"] and result["seam_exact"]
    assert result["whole_disk_future_exact"] is False
    with np.load(out / "combined-control.npz") as combined:
        assert combined["control_tick"].shape == (2001,)
        assert combined["control_tick"][600] == 600
        assert combined["control_tick"][601] == 601
    with np.load(out / "combined-native-contact.npz") as native:
        assert native["time_s"].shape == (100000,)
    receipt = json.loads((out / "advance-ledger.json").read_text())
    assert receipt["completed_continuation_controls"] == 1400
    assert receipt["continuation_native_samples"] == 70000
    assert receipt["actual_neural_s"] == pytest.approx(7.)
    assert receipt["actual_physical_s"] == pytest.approx(7.)
    manifest = json.loads((out / "output-manifest.json").read_text())
    assert all(e225.sha256(out / name) == item["sha256"]
               for name, item in manifest.items())


@pytest.mark.parametrize(("failure", "expected", "error_type"), [
    ("health", "measured_native_health_stop", None),
    ("resource", "resource_limited", TimeoutError),
    ("technical", "technical_failure", RuntimeError)])
def test_fake_partial_failure_preserves_suffix_and_classification(
        tmp_path, monkeypatch, failure, expected, error_type):
    if error_type is None:
        out, result = fake_run(tmp_path, monkeypatch, failure=failure)
    else:
        with pytest.raises(error_type, match="generated"):
            fake_run(tmp_path, monkeypatch, failure=failure)
        out = tmp_path / "continuation"
        result = json.loads((out / "result.json").read_text())
    assert result["validity"] == expected
    assert result["full_10s"] is False
    assert not (out / "combined-control.npz").exists()
    assert (out / "continuation-control.npz").exists()
    assert (out / "continuation-native-contact.npz").exists()
    receipt = json.loads((out / "advance-ledger.json").read_text())
    if failure == "health":
        assert receipt["actual_physical_s"] is None
        with np.load(out / "continuation-native-contact.npz") as native:
            assert np.isnan(native["time_s"][-1])
    assert (out / "output-manifest.json").exists()
