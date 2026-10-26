"""Native shadow hook and output contract tests using fake physics only."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_native_sensory as e195
from tools.banc_m3_native_sensory import (
    ADAPTER_SCHEMA,
    ALGORITHM,
    PATCH_SCHEMA,
    TRIAL_SCHEMA,
    NativeSensoryTrial,
    candidate_identity,
)
from tools.verify_c_banc_m3_native_sensory import (
    gate,
    install_checkpoint_capture,
    install_factory,
    native_arrays,
    restore_candidate_factory,
    validate_checkpoint_disk,
)
from tools.verify_c_banc_observation import exact


class FakeMj:
    mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

    @staticmethod
    def MjData(model):
        return SimpleNamespace(state=np.zeros(2))

    @staticmethod
    def mj_stateSize(model, kind):
        return 2

    @staticmethod
    def mj_getState(model, data, target, kind):
        target[:] = data.state

    @staticmethod
    def mj_setState(model, data, source, kind):
        data.state[:] = source

    @staticmethod
    def mj_forward(model, data):
        data.forwarded = True


class FakeBody:
    mj = FakeMj
    m = object()

    def __init__(self):
        self.d = SimpleNamespace(state=np.zeros(2))
        self.calls = 0

    def _step_physics(self):
        self.calls += 1
        self.d.state[0] += .0001
        self.d.state[1] += 1

    def physics_time(self):
        return self.d.state[0]

    def knee_kinematics(self, *, include_direction):
        assert include_direction is False
        return {"flexion_rad": np.full(6, self.d.state[0]),
                "velocity_rad_s": np.full(6, self.d.state[1])}


class Recorder:
    def __init__(self):
        self.rows = []
        self.native_rows = self.rows

    def record(self, time_s, q, velocity):
        self.rows.append((time_s, np.array(q), np.array(velocity)))

    def record_failure(self, time_s, error):
        self.rows.append((time_s, repr(error)))


def test_shadow_observer_does_not_touch_actor_and_restores_slot():
    body = FakeBody()
    trial = object.__new__(NativeSensoryTrial)
    trial.body = body
    trial.adapter = Recorder()
    trial._native_restore = None
    assert "_step_physics" not in vars(body)
    trial._install_native_observer()
    body._step_physics()
    assert body.calls == 1
    np.testing.assert_array_equal(body.d.state, [.0001, 1.])
    assert trial.adapter.rows[0][0] == pytest.approx(.0001)
    np.testing.assert_array_equal(trial.adapter.rows[0][2], np.ones(6))
    trial._native_restore()
    assert "_step_physics" not in vars(body)
    body._step_physics()
    assert body.calls == 2 and len(trial.adapter.rows) == 1


def test_partial_native_rows_have_fixed_header():
    candidate = SimpleNamespace(native_ports=np.array([1, 2]), native_rows=[
        {"native_index": 0, "time_s": .0001, "control_tick": 1,
         "substep": 1, "interval_dt_s": .0001,
         "flexion_rad": np.zeros(6), "velocity_rad_s": np.ones(6),
         "velocity_lowpass_rad_s": np.zeros(6),
         "features": np.array([.1, .2]), "complete": True}])
    data = native_arrays(candidate)
    assert data["features"].shape == (1, 2)
    assert data["interval_dt_s"].shape == (1,)
    assert data["control_tick"][0] == 1
    bad = dict(candidate.native_rows[0])
    bad.update(native_index=1, time_s=.0002, substep=2,
               features=np.array([.1]), complete=False, error="synthetic late failure")
    candidate.native_rows.append(bad)
    partial = native_arrays(candidate)
    assert partial["complete"].tolist() == [True, False]
    np.testing.assert_array_equal(partial["features"][0], [.1, .2])
    assert np.isnan(partial["features"][1]).all()
    assert "synthetic late failure" in partial["error"][1]


def test_actual_checkpoint_payload_is_compared_to_disk(tmp_path):
    data = {"start": {"value": np.array([1.])},
            "final": {"value": np.array([2.])}}

    class Store:
        @staticmethod
        def save(path, payload):
            data[path.name] = {"value": payload["value"].copy()}

        @staticmethod
        def load(path):
            return data[path.name]

    e181 = SimpleNamespace(StateStore=Store, protocol=SimpleNamespace(exact=exact))
    captured, undo = install_checkpoint_capture(e181)
    e181.StateStore.save(tmp_path / "start", {"value": np.array([1.])})
    e181.StateStore.save(tmp_path / "final", {"value": np.array([2.])})
    undo()
    assert e181.StateStore is Store
    assert set(captured) == {"start", "final"}
    data["final"]["value"][0] = 99.
    with pytest.raises(Exception, match="final"):
        validate_checkpoint_disk(e181, tmp_path, {}, captured)


def test_candidate_factory_observes_candidate_encode_and_restores_slot():
    calls = []
    candidate = SimpleNamespace()
    candidate.encode = lambda: "drive"

    def make_observer(*args, **kwargs):
        return lambda session, detailed, available: {"observer": True}

    e150 = SimpleNamespace(make_observer=make_observer)
    e181 = SimpleNamespace(e150=e150)

    class Collector:
        def __init__(self):
            self.candidate = candidate

        def capture_neural(self, session):
            calls.append("neural")

        def capture_encode(self, original, *args, **kwargs):
            calls.append("candidate_encode")
            return original(*args, **kwargs)

    restore = install_factory(e181, Collector())
    observe = e181.e150.make_observer([])
    assert observe(SimpleNamespace(control_tick=0), None, {}) == {"observer": True}
    assert candidate.encode() == "drive"
    assert calls == ["neural", "candidate_encode"]
    restore()
    assert e181.e150.make_observer is make_observer
    assert candidate.encode() == "drive"


def test_cleanup_failure_keeps_original_restore_error():
    class Base:
        closed = False

        def snapshot(self):
            return {}

        def close(self):
            raise RuntimeError("cleanup failed")

    def fail_exact(*args):
        raise ValueError("primary exact failure")

    original = lambda graph, saved, theta: Base()
    e181 = SimpleNamespace(restore_arm=original, protocol=SimpleNamespace(exact=fail_exact))
    undo = restore_candidate_factory(e181)
    with pytest.raises(ValueError, match="primary exact failure") as exc:
        e181.restore_arm(None, {}, (0.,) * 6)
    assert any("cleanup failed" in note for note in exc.value.__notes__)
    undo()
    assert e181.restore_arm is original


def test_gate_rejects_wrong_review_path_before_any_native_work(tmp_path):
    with pytest.raises(ValueError, match="gate path"):
        gate(tmp_path / "freeze.json", tmp_path / "review.json", tmp_path / "output")


def test_full_trial_rejects_wrong_boundary_before_model_construction():
    identity = candidate_identity("a" * 64)
    saved = {"schema": TRIAL_SCHEMA, "control_tick": 1,
             "native_sensory_patch": {"schema": PATCH_SCHEMA,
                "source_adapter_identity": "a" * 64,
                "adapter_identity": identity, "algorithm": ALGORITHM,
                "control_dt_s": .005, "native_dt_s": .0001,
                "samples_per_control": 50},
             "adapter": {"schema": ADAPTER_SCHEMA,
                         "source_motor_identity": "a" * 64,
                         "hash": identity, "encode_count": 1,
                         "sample_count": 0}}
    with pytest.raises(ValueError, match="boundary"):
        NativeSensoryTrial.from_checkpoint(None, saved)


def test_gate_rejects_tampered_review_schema(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text(json.dumps({"schema": "flylab.m3.e195.native-sensory.freeze.v1"}))
    review.write_text(json.dumps({"schema": "tampered"}))
    monkeypatch.setattr(e195, "safe", lambda name: freeze if name == e195.FREEZE else review)
    with pytest.raises(ValueError, match="gate schema"):
        gate(freeze, review, tmp_path / "output")


def test_effects_rate_population_is_only_hook_club(monkeypatch, tmp_path):
    source = tmp_path / "source"
    arm = tmp_path / "arm"
    source.mkdir()
    arm.mkdir()
    provenance = {"native_ports": [0, 1], "graph_hash": "graph",
        "network_identity": "network", "body_model_hash": "body",
        "leaf_adapter_identity": "leaf", "port_expansion_indices": [0, 1, 2],
        "port_expansion_ports": [0, 1, 2], "unique_sensory_indices": [0, 1, 2],
        "joints": [[0] * 7] * 6, "network_parameters": {"cap": [1., 1., 1.]},
        "ports": [{"kind": "hook", "cell_type": "SNpp41"},
                  {"kind": "club", "cell_type": "x"},
                  {"kind": "load", "cell_type": "x"}]}
    (source / "runtime-provenance.json").write_text(json.dumps(provenance))
    encoder = {"last_features": np.zeros((600, 3)),
               "new_filtered": np.zeros((600, 3)),
               "flexion_rad": np.ones((600, 6))}
    np.savez(source / "encoder-signals.npz", **encoder)
    baseline_rate = np.zeros((601, 3), np.float32)
    np.savez(source / "sensory-signals.npz", rate=baseline_rate)
    physical = {"longest_forward_bout_s": .3,
        "fixed_initial_heading_net_forward_mm": 1., "net_over_path": .5,
        "native_slip_load_weighted_mm_s": 1.}
    (arm / "result.json").write_text(json.dumps({"physical_3s": physical}))
    neural = {"rate": baseline_rate.copy()}
    neural["rate"][101:601, 2] = 100.  # Unchanged load port is outside intervention.
    metadata = {"baseline_sensory_directory": "source", "baseline_directory": "arm"}
    monkeypatch.setattr(e195, "safe", lambda name: source if name == "source" else arm)
    result = {"physical_3s": physical, "validity": "valid_3s_diagnostic",
              "startup_pass": True, "measured_health_stop": False}
    effects = e195.compare_effects(metadata, result, encoder, neural, provenance)
    assert effects["hook_club_connected_cells"] == 2
    assert effects["connected_rate_delta_rms"] == 0.
    assert effects["hook_club_candidate_exact_zero_cell_fraction"] == 1.
    neural["rate"][101, 0] = 2.
    effects = e195.compare_effects(metadata, result, encoder, neural, provenance)
    assert effects["hook_club_baseline_exact_zero_cells"] == 2
    assert effects["hook_club_candidate_exact_zero_cells"] == 1
    assert effects["hook_club_candidate_exact_zero_cell_fraction"] == .5
    assert effects["hook_club_candidate_exact_zero_sample_fraction"] == .999
    assert effects["connected_rate_delta_rms"] == pytest.approx(np.sqrt(4. / 1000.))
    result["physical_3s"] = {name: np.float64(value) for name, value in physical.items()}
    effects = e195.compare_effects(metadata, result, encoder, neural, provenance)
    assert all(type(value) is bool for value in effects["screen"].values())
    json.dumps(effects, allow_nan=False)
