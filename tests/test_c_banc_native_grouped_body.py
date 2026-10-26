"""E222 raw, screen and pre-model gate checks; no CUDA or body is started."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_native_grouped_body as body


def record(command=None, tendon=None):
    return {"command": np.zeros(42) if command is None else command,
            "tendon": np.zeros(6) if tendon is None else tendon}


def test_startup_exact_99_and_original_limits() -> None:
    values = [record() for _ in range(100)]
    target = {"loaded_startup_target_command": np.zeros((99, 42)),
              "loaded_startup_target_tendon": np.zeros((99, 6))}
    limits = {"q_RMS": .018, "q_max_abs": .04,
              "LTM_RMS": .009, "LTM_max_abs": .025}
    assert body.startup_screen(values, target, limits)["pass"]
    assert not body.startup_screen(values[:99], target, limits)["available"]
    values[1] = record(command=np.full(42, .041))
    assert not body.startup_screen(values, target, limits)["pass"]


def test_cut_restart_distinguishes_neutral_from_positive() -> None:
    neutral = np.linspace(-.2, .2, 42)
    cut = [{"activation": np.zeros(96), "command": neutral.copy(),
            "tendon": np.zeros(6)} for _ in range(4)]
    restart = [{"activation": np.ones(96), "command": neutral + .01,
                "tendon": np.full(6, .02)} for _ in range(16)]
    rows = [{"tendon_inputs": {"abdomen_pitch": .2, "abdomen_yaw": .2}}
            for _ in range(20)]
    assert body.cut_screen(cut + restart, neutral, rows)["pass"]
    cut[0]["tendon"][0] = .001
    assert not body.cut_screen(cut + restart, neutral, rows)["pass"]
    cut[0]["tendon"][0] = 0.
    rows[4]["tendon_inputs"]["abdomen_yaw"] = 0.
    assert not body.cut_screen(cut + restart, neutral, rows)["pass"]


class _FakeTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def test_live_control_raw_shapes_and_dtype(monkeypatch) -> None:
    row = {"joint_targets_rad": np.zeros(42),
           "tendon_inputs": {key: 0. for key in body.TENDONS}}
    monkeypatch.setattr(body.protocol, "capture_control", lambda _: row)
    session = SimpleNamespace(control_tick=1, rate=np.ones(391, np.float32),
        body=SimpleNamespace(physics_time=lambda: .005),
        adapter=SimpleNamespace(activation=np.zeros(96)),
        network=SimpleNamespace(tick=20,
            _free_rate64=_FakeTensor(np.ones(343, np.float64)),
            _free_fast64=_FakeTensor(np.zeros((343, 3), np.float64)),
            _free_slow64=_FakeTensor(np.zeros((343, 3), np.float64))))
    _, values = body.capture_native_state(session)
    assert values["motor_rate32"].dtype == np.float32
    assert values["free_rate64"].dtype == np.float64
    assert values["h_fast"].shape == (343, 3)
    assert values["activation"].shape == (96,)


def test_bad_parent_review_blocks_before_any_input_or_model(tmp_path, monkeypatch) -> None:
    freeze_path, review_path = tmp_path / "freeze.json", tmp_path / "review.json"
    out = tmp_path / "new-output"
    freeze_path.write_text(json.dumps({"schema": body.FREEZE_SCHEMA,
        "owned_files": list(body.OWNED), "design": "docs/design",
        "design_sha256": "design", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "active_UI_changed": False,
        "native": {"main_controls": 2000, "neurons": 158706},
        "resource_limits": {"internal_s": 175., "finalization_reserve_s": 20.,
                            "original_attempts": 1, "retries": 0,
                            "fit_calls": 0, "solver_calls": 0}}))
    review_path.write_text(json.dumps({"schema": body.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["blocked"],
        "freeze_sha256": body.FREEZE_SHA,
        "source_sha256": {key: "x" * 64 for key in body.OWNED}}))
    monkeypatch.setattr(body, "sha256", lambda p: body.FREEZE_SHA
                        if p == freeze_path else ("design" if str(p).endswith("design")
                                                 else "review"))
    with pytest.raises(ValueError, match="review"):
        body.gate(freeze_path, body.FREEZE_SHA, review_path, "review", out)
    assert not out.exists()


def _mock_whole_run(tmp_path, monkeypatch, *, health_stop=False,
                    health_screen_fail=False, resource_stop=False,
                    cut_fail=False):
    """All 2040 controls use fake clocks/contacts, never a neural/body backend."""
    out = tmp_path / "e222-fake"
    freeze = {"paths": {key: key for key in
        ("graph", "start", "parameters", "prototypes", "candidate", "targets",
         "baseline")}, "graph_hash": "graph", "startup_limits": {
            "q_RMS": .018, "q_max_abs": .04, "LTM_RMS": .009,
            "LTM_max_abs": .025}, "resource_limits": {
            "resident_bytes": 1 << 40, "allocated_gpu_bytes": 1 << 40,
            "output_bytes": 1 << 40}}
    monkeypatch.setattr(body, "gate", lambda *args: (freeze, {"approved": True}))
    monkeypatch.setattr(body, "safe", lambda name: tmp_path / name)
    monkeypatch.setattr(body.protocol.GraphStore, "load",
                        lambda path: SimpleNamespace(hash="graph", n=158706))
    monkeypatch.setattr(body, "load_npz", lambda path: {})
    monkeypatch.setattr(body, "load_selected_npz", lambda path, keys: {
        "beta": np.zeros((96, 64))} if "candidate" in str(path) else {
        "loaded_startup_target_command": np.zeros((99, 42)),
        "loaded_startup_target_tendon": np.zeros((99, 6))})
    monkeypatch.setattr(body.protocol, "profiled_construction",
                        lambda label, callback, rows, persist: callback())
    monkeypatch.setattr(body.gait, "load_model", lambda path: {})
    monkeypatch.setattr(body.gait, "analyze", lambda data: ({}, None, None))
    monkeypatch.setattr(body.e113, "supported_slip", lambda *a, **k: {"available": True})
    monkeypatch.setattr(body.e113, "body_quality", lambda *a, **k: {
        "complete_10s_native_health": not health_screen_fail})
    monkeypatch.setattr(body, "baseline_comparison", lambda *a: {
        "comparison_fields": [], "candidate_descriptive": {}})
    import tools.fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (
        SimpleNamespace(restore_original_limits=lambda: None), {}))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1, peak_wset=1)))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=lambda: (1 << 40, 1 << 40)))))
    states = {}

    class FakeStore:
        @staticmethod
        def save(path, value):
            states[str(path)] = copy.deepcopy(value)
            path.mkdir()
            (path / "state.json").write_text(json.dumps(value))

        @staticmethod
        def load(path):
            return copy.deepcopy(states.get(str(path), {"schema": "E150"}))

    monkeypatch.setattr(body.protocol, "StateStore", FakeStore)
    sample = (0., np.ones(6), np.zeros(6), np.ones(6, bool), False, 1.)
    hashes_before_cut = []
    observer_installs = []

    def fake_observer(physical, samples):
        observer_installs.append(samples)
        if len(observer_installs) == 3:
            hashes_before_cut.append(body.sha256(out / "native-contact.npz"))
        physical.sink = samples
        return lambda: setattr(physical, "sink", None)

    monkeypatch.setattr(body.e113, "install_native_observer", fake_observer)

    class FakeSession:
        def __init__(self, saved=None):
            saved = saved or {}
            self.control_tick = saved.get("control_tick", 0)
            self.network = SimpleNamespace(tick=saved.get("network_tick", 0),
                                           n=158706, identity="native")
            self.body = SimpleNamespace(t=saved.get("physics_time_s", 0.),
                physics_time=lambda: self.body.t, model_hash="body",
                neutral=np.zeros(42), fault=False, sink=None)
            self.adapter = SimpleNamespace(identity="canonical")
            self.fault = False
            self.cuts = {"motor": False}

        def snapshot(self):
            return {"control_tick": self.control_tick,
                    "network_tick": self.network.tick,
                    "physics_time_s": self.body.t, "wall_s": 0.}

        def advance(self, controls):
            assert controls == 1
            if health_stop and self.control_tick == 0:
                raise body.e113.NativeHealthStop("generated physical stop")
            if resource_stop and self.control_tick == 0:
                raise TimeoutError("generated resource stop")
            if cut_fail and self.cuts["motor"]:
                raise RuntimeError("generated cut failure")
            self.control_tick += 1
            self.network.tick += 20
            for _ in range(50):
                self.body.t += .0001
                if self.body.sink is not None:
                    self.body.sink.append((self.body.t, *sample[1:]))

        def set_cuts(self, cuts):
            self.cuts.update(cuts)

        def close(self):
            pass

    monkeypatch.setattr(body.NativeGroupedTrial, "from_e150_checkpoint",
                        lambda *a: FakeSession())
    monkeypatch.setattr(body.NativeGroupedTrial, "from_checkpoint",
                        lambda graph, saved: FakeSession(saved))
    motor = np.zeros(391, np.float32)
    free = np.zeros(343)
    history = np.zeros((343, 3))
    activation = np.ones(96)
    command = np.full(42, .01)
    tendon = np.full(6, .002)

    def fake_capture(session):
        cut = session.cuts["motor"]
        row = {"control_tick": session.control_tick,
               "time_s": session.control_tick * .005,
               "joint_targets_rad": np.zeros(42) if cut else command,
               "tendon_inputs": {**{key: (0. if cut else .002)
                                        for key in body.TENDONS},
                                 "abdomen_pitch": .2, "abdomen_yaw": .2}}
        value = {"control_tick": session.control_tick,
                 "network_tick": session.network.tick,
                 "physics_time_s": session.body.t, "motor_rate32": motor,
                 "free_rate64": free, "h_fast": history, "h_slow": history,
                 "activation": np.zeros(96) if cut else activation,
                 "command": np.zeros(42) if cut else command,
                 "tendon": np.zeros(6) if cut else tendon}
        return row, value

    monkeypatch.setattr(body, "capture_native_state", fake_capture)
    try:
        result = body.run(tmp_path / "freeze", body.FREEZE_SHA,
                          tmp_path / "review", "review", out)
    except BaseException:
        if not (health_stop or health_screen_fail or resource_stop or cut_fail):
            raise
        result = json.loads((out / "result.json").read_text())
    return out, result, observer_installs, hashes_before_cut


def test_fake_full_path_distinct_native_and_sealed_manifest(tmp_path, monkeypatch):
    out, result, observers, hashes = _mock_whole_run(tmp_path, monkeypatch)
    assert result["main_complete"] and result["clone_1p1_exact"]
    assert result["cut_restart"]["pass"]
    assert [len(samples) for samples in observers] == [100000, 1000, 1000]
    assert hashes == [body.sha256(out / "native-contact.npz")]
    with np.load(out / "main-control.npz") as raw:
        assert raw["control_tick"].shape == (2001,)
    assert len(json.loads((out / "clone-trace.json").read_text())) == 21
    manifest = json.loads((out / "output-manifest.json").read_text())
    assert all(body.sha256(out / name) == item["sha256"]
               for name, item in manifest.items())


def test_fake_measured_health_stop_keeps_classification(tmp_path, monkeypatch):
    out, result, _, _ = _mock_whole_run(tmp_path, monkeypatch, health_stop=True)
    assert result["validity"] == "measured_native_health_stop"
    assert result["main_complete"] is False
    assert "generated physical stop" in result["primary_error"]
    assert (out / "trace.jsonl").exists()


def test_fake_completed_health_screen_keeps_measured_classification(
        tmp_path, monkeypatch):
    out, result, _, _ = _mock_whole_run(
        tmp_path, monkeypatch, health_screen_fail=True)
    assert result["validity"] == "measured_native_health_stop"
    assert result["main_complete"] is True
    assert (out / "body-quality.json").exists()
    assert not (out / "main-result.json").exists()


def test_fake_resource_stop_is_separate_from_health(tmp_path, monkeypatch):
    _, result, _, _ = _mock_whole_run(
        tmp_path, monkeypatch, resource_stop=True)
    assert result["validity"] == "resource_limited"
    assert "generated resource stop" in result["primary_error"]


def test_fake_cut_failure_preserves_sealed_main(tmp_path, monkeypatch):
    out, result, observers, hashes = _mock_whole_run(
        tmp_path, monkeypatch, cut_fail=True)
    assert result["validity"] == "valid_full_body_diagnostic"
    assert result["main_complete"] and result["main_valid_despite_cut_failure"]
    assert result["cut_restart_status"] == "failed_execution"
    assert result["execution_status"] == "failed"
    assert hashes == [body.sha256(out / "native-contact.npz")]
    assert [len(samples) for samples in observers] == [100000, 1000, 0]
    assert json.loads((out / "main-result.json").read_text())["validity"] == (
        "valid_full_body_diagnostic")
