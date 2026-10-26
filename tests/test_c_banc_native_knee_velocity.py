"""Synthetic E194 alignment, integration and fail-closed tests."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_native_knee_velocity as e194


def synthetic_sources():
    headers = {
        "native_index": ((10000,), "int64"),
        "time_s": ((10000,), "float64"),
        "control_tick": ((10000,), "int64"),
        "qpos": ((10000, 3), "float64"),
        "qvel": ((10000, 3), "float64"),
        "ctrl": ((10000, 2), "float64"),
    }
    native = {
        "native_index": np.arange(10000),
        "time_s": 1.0 + (np.arange(10000) + 1) * 0.0001,
        "control_tick": np.repeat(np.arange(201, 401), 50),
        "qpos": np.zeros((10000, 3)),
        "qvel": np.zeros((10000, 3)),
        "ctrl": np.zeros((10000, 2)),
    }
    native.update({key: np.ones(10000, bool) for key in e194.FLAGS})
    rows = [{"control_tick": i, "time_s": i * 0.005,
             "qpos": [0.0] * 3, "qvel": [0.0] * 3,
             "ctrl": [0.0] * 2} for i in range(601)]
    encoder = {"control_tick": np.arange(600),
               "complete": np.ones(600, bool),
               "flexion_rad_available": np.ones(600, bool),
               "flexion_velocity_rad_s_available": np.ones(600, bool),
               "flexion_rad": np.zeros((600, 6)),
               "flexion_velocity_rad_s": np.zeros((600, 6))}
    metadata = {"headers": {key: {"shape": shape, "dtype": dtype}
                            for key, (shape, dtype) in headers.items()},
                "body_hash": "body"}
    layout = {"nq": 3, "nv": 3, "nu": 2}
    saved = {"control_tick": 200, "body": {"modelHash": "body", "state": [0.0]}}
    return metadata, layout, {"samples": rows}, native, encoder, saved


@pytest.mark.parametrize("tamper,match", [
    (lambda n, e: n["control_tick"].__setitem__(50, 201), "control_tick"),
    (lambda n, e: n["qvel"].__setitem__((49, 0), 1.0), "native/trace qvel"),
    (lambda n, e: n["qpos"].resize((9999, 3), refcheck=False), "native header"),
    (lambda n, e: n["source_verified"].__setitem__(3, False), "incomplete native"),
    (lambda n, e: e["complete"].__setitem__(200, False), "encoder completion"),
])
def test_source_validation_rejects_bad_alignment(tamper, match):
    args = synthetic_sources()
    tamper(args[3], args[4])
    with pytest.raises(ValueError, match=match):
        e194.validate_alignment(*args)


def test_smooth_and_transient_integrals():
    # Smooth motion integrates accurately at either resolution.
    t = 1.0 + np.arange(e194.ROWS) * 0.0001
    angle = np.tile((t - 1.0)[:, None] ** 2, (1, 6))
    velocity = np.tile((2 * (t - 1.0))[:, None], (1, 6))
    arrays, result = e194.summarize(t, angle, velocity)
    np.testing.assert_allclose(arrays["native_trapezoid_rad"], arrays["actual_change_rad"], atol=1e-14)
    np.testing.assert_allclose(arrays["boundary_trapezoid_rad"], arrays["actual_change_rad"], atol=1e-14)
    assert result["supporting_legs"] == 0

    # Each interval has zero velocity at both boundaries and a positive
    # transient inside. The native integral captures its displacement.
    phase = (np.arange(e194.ROWS) % 50) / 50
    angle_one = (np.arange(e194.ROWS) // 50) * 0.005 + 0.005 * (phase - np.sin(2 * np.pi * phase) / (2 * np.pi))
    velocity_one = 1.0 - np.cos(2 * np.pi * phase)
    angle = np.tile(angle_one[:, None], (1, 6))
    velocity = np.tile(velocity_one[:, None], (1, 6))
    arrays, result = e194.summarize(t, angle, velocity)
    assert np.max(np.abs(arrays["native_trapezoid_error_rad"])) < 1e-12
    assert np.max(np.abs(arrays["boundary_trapezoid_rad"])) == 0
    assert result["supporting_legs"] == 6
    assert result["verdict"] == "supports_boundary_information_loss_in_saved_trajectory"


def test_stationary_leg_has_null_relative_error():
    t = 1.0 + np.arange(e194.ROWS) * 0.0001
    f = np.zeros((e194.ROWS, 6))
    v = np.zeros_like(f)
    _, result = e194.summarize(t, f, v)
    for leg in e194.LEGS:
        assert result["legs"][leg]["stationary"] is True
        assert result["legs"][leg]["native_trapezoid_relative_error"] is None


def test_observer_failure_checks_actor_without_masking_original_error():
    class FakeMj:
        mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

        @staticmethod
        def mj_stateSize(_model, _typ):
            return 3

        @staticmethod
        def mj_getState(_model, data, dest, _typ):
            dest[:] = data.state

    body = SimpleNamespace(mj=FakeMj(), m=object(), d=SimpleNamespace(state=np.arange(3.0)))
    with pytest.raises(RuntimeError, match="observer failed") as caught, e194.protect_actor(body):
        body.d.state[0] = 4.0
        raise RuntimeError("observer failed")
    assert "Actor integration state changed" in str(caught.value.__notes__)


def test_gate_rejects_tampered_source(tmp_path, monkeypatch):
    monkeypatch.setattr(e194, "ROOT", tmp_path)
    design = tmp_path / "design.md"
    design.write_text("fixed design")
    source = tmp_path / "source.bin"
    source.write_bytes(b"accepted")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text("{}")
    source_hashes = {"source.bin": e194.sha256(source),
                     "checkpoint/state.json": e194.sha256(checkpoint / "state.json")}
    metadata = {"schema": e194.SCHEMA + ".metadata.v1",
                "design_path": "design.md", "design_sha256": e194.sha256(design),
                "checkpoint": "checkpoint", "original_runs": 0,
                "candidate_adopted": False, "reserved_s": 60,
                "cumulative_before_s": 4758, "cumulative_cap_s": 4994,
                "inputs": source_hashes, "reused_sources": {},
                **{key: "source.bin" for key in
                   ("source_native", "source_layout", "source_result", "source_review",
                    "encoder", "provenance", "trace", "e193_review")}}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    params = {"native_rows": 10000, "output_rows": 10001,
              "control_intervals": 200, "steps_per_control": 50,
              "internal_wall_s": 50, "cpu_threads": 1, "blas_threads": 1,
              "gpu_count": 0, "automatic_retries": 0,
              "maximum_output_bytes": e194.MAX_OUTPUT_BYTES,
              "minimum_free_bytes": e194.MIN_FREE_BYTES}
    freeze = {"schema": e194.SCHEMA + ".freeze.v1",
              "metadata_path": "metadata.json", "metadata_sha256": e194.sha256(metadata_path),
              "design_sha256": e194.sha256(design), "output_path": str(e194.OUTPUT),
              "parameters": params, "source_hashes": source_hashes}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    review = {"schema": e194.SCHEMA + ".source-review.v1", "accepted": True,
              "freeze_sha256": e194.sha256(freeze_path), "source_hashes": source_hashes}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    output = tmp_path / e194.OUTPUT
    output.parent.mkdir(parents=True)
    assert e194.verify_gate(freeze_path, review_path, output) == metadata
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        e194.verify_gate(freeze_path, review_path, output)


class FakeData:
    def __init__(self):
        self.time = 1.0
        self.qpos = np.zeros(109)
        self.qvel = np.zeros(108)
        self.ctrl = np.zeros(82)


class FakeMj:
    mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

    def __init__(self, stats):
        self.stats = stats

    def MjData(self, _model):
        self.stats["shadow_data"] += 1
        return FakeData()

    @staticmethod
    def mj_stateSize(_model, _typ):
        return 300

    @staticmethod
    def mj_getState(_model, data, dest, _typ):
        dest[:] = np.concatenate(([data.time], data.qpos, data.qvel, data.ctrl))

    @staticmethod
    def mj_setState(_model, data, source, _typ):
        data.time = source[0]
        data.qpos[:] = source[1:110]
        data.qvel[:] = source[110:218]
        data.ctrl[:] = source[218:300]

    def mj_forward(self, _model, _data):
        self.stats["forwards"] += 1


class FakeBody:
    def __init__(self, stats, *, fail_at=None, close_fails=False):
        self.stats = stats
        self.fail_at = fail_at
        self.close_fails = close_fails
        self.mj = FakeMj(stats)
        self.m = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.0001, integrator=0),
            actuator_gainprm=np.ones((82, 10)),
            actuator_biasprm=np.zeros((82, 10)),
            actuator_ctrlrange=np.ones((82, 2)),
        )
        self.d = FakeData()
        self.t0 = 0.0
        self.model_hash = "body"

    def _step_physics(self):
        raise AssertionError("actor step must never run")

    def knee_kinematics(self, *, include_direction):
        assert include_direction is False
        assert self.d is not self.stats["actor"].d
        count = self.stats["observations"]
        if count == self.fail_at:
            raise RuntimeError("synthetic observer failure")
        self.stats["observations"] += 1
        return {"flexion_rad": np.full(6, self.d.time - 1.0),
                "velocity_rad_s": np.ones(6)}

    def close(self):
        self.stats["closes"] += 1
        if self.close_fails:
            raise RuntimeError("synthetic cleanup failure")


def fake_run_inputs(tmp_path, monkeypatch, *, fail_at=None, close_fails=False):
    metadata, layout, trace, native, encoder, saved = synthetic_sources()
    native["qpos"] = np.zeros((10000, 109))
    native["qvel"] = np.zeros((10000, 108))
    native["ctrl"] = np.zeros((10000, 82))
    for key in ("qpos", "qvel", "ctrl"):
        metadata["headers"][key]["shape"] = list(native[key].shape)
        for row in trace["samples"]:
            row[key] = [0.0] * native[key].shape[1]
    layout.update(nq=109, nv=108, nu=82,
                  leg_dof_addresses=[list(range(i * 7, i * 7 + 7)) for i in range(6)])
    encoder["flexion_rad"][200:401] = (
        np.arange(200, 401) * 0.005 - 1.0)[:, None]
    encoder["flexion_velocity_rad_s"][200:401] = 1.0
    saved["body"]["state"] = np.concatenate(([1.0], np.zeros(109 + 108 + 82))).tolist()
    saved["parameters"] = {"servo_profile": "tracking_all"}
    metadata["checkpoint"] = "checkpoint"
    metadata["source_native"] = "native.bin"
    monkeypatch.setattr(e194, "ROOT", tmp_path)
    (tmp_path / "native.bin").write_bytes(b"synthetic")
    (tmp_path / "checkpoint").mkdir()
    monkeypatch.setattr(e194, "verify_gate", lambda *_args: metadata)
    monkeypatch.setattr(e194, "load_sources", lambda _metadata:
                        (layout, trace, native, encoder, {}))
    monkeypatch.setattr(e194.StateStore, "load", lambda _path: saved)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    stats = {"shadow_data": 0, "forwards": 0, "observations": 0, "closes": 0}
    body = FakeBody(stats, fail_at=fail_at, close_fails=close_fails)
    stats["actor"] = body
    actor_state = np.concatenate(([body.d.time], body.d.qpos, body.d.qvel, body.d.ctrl))

    def construct(_saved, _observation, counters):
        counters.update(body_constructor_attempts=1, body_constructor_completed=1,
                        settling_physics_steps=2000)
        return body

    monkeypatch.setattr(e194, "construct_body_safely", construct)
    output = tmp_path / "output"
    return output, body, stats, actor_state


def test_synthetic_run_reads_all_rows_and_restores_actor(tmp_path, monkeypatch):
    output, body, stats, actor_state = fake_run_inputs(tmp_path, monkeypatch)
    result = e194.run(tmp_path / "unused-freeze", tmp_path / "unused-review", output)
    assert result["complete"] is True
    assert result["rows_complete"] == 10001
    assert result["counters"]["shadow_mjdata"] == stats["shadow_data"] == 1
    assert stats["forwards"] == 10001  # anchor plus all 10000 native states
    assert stats["observations"] == 10001
    assert stats["closes"] == 1
    assert body._step_physics.__func__ is FakeBody._step_physics
    np.testing.assert_array_equal(
        np.concatenate(([body.d.time], body.d.qpos, body.d.qvel, body.d.ctrl)), actor_state)
    with np.load(output / "native-knee-velocity.npz") as saved:
        np.testing.assert_array_equal(saved["source_index"], np.arange(-1, 10000))
        np.testing.assert_array_equal(saved["complete"], np.ones(10001, bool))
        assert saved["time_s"][0] == 1.0
        assert saved["time_s"][50] == pytest.approx(1.005)
        assert saved["time_s"][-1] == pytest.approx(2.0)
        np.testing.assert_allclose(saved["flexion_rad"][50], 0.005, atol=1e-12)


def test_observer_failure_writes_partial_rows_and_closes(tmp_path, monkeypatch):
    output, body, stats, actor_state = fake_run_inputs(tmp_path, monkeypatch, fail_at=72)
    with pytest.raises(RuntimeError, match="synthetic observer failure"):
        e194.run(tmp_path / "unused-freeze", tmp_path / "unused-review", output)
    result = json.loads((output / "result.json").read_text())
    assert result["complete"] is False
    assert result["rows_complete"] == 72
    assert result["error"]["type"] == "RuntimeError"
    assert stats["closes"] == 1
    assert stats["observations"] == 72
    assert body._step_physics.__func__ is FakeBody._step_physics
    np.testing.assert_array_equal(
        np.concatenate(([body.d.time], body.d.qpos, body.d.qvel, body.d.ctrl)), actor_state)
    with np.load(output / "native-knee-velocity.npz") as saved:
        assert np.all(saved["complete"][:72])
        assert not np.any(saved["complete"][72:])
        assert "synthetic observer failure" in saved["error"][72]


def test_cleanup_only_failure_is_nonzero_outcome(tmp_path, monkeypatch):
    output, body, stats, _ = fake_run_inputs(tmp_path, monkeypatch, close_fails=True)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        e194.run(tmp_path / "unused-freeze", tmp_path / "unused-review", output)
    result = json.loads((output / "result.json").read_text())
    assert result["complete"] is False
    assert result["rows_complete"] == 10001
    assert result["error"]["phase"] == "cleanup"
    assert stats["closes"] == 1
    assert body._step_physics.__func__ is FakeBody._step_physics


def test_observer_error_remains_primary_if_cleanup_also_fails(tmp_path, monkeypatch):
    output, body, stats, _ = fake_run_inputs(
        tmp_path, monkeypatch, fail_at=72, close_fails=True)
    with pytest.raises(RuntimeError, match="synthetic observer failure") as caught:
        e194.run(tmp_path / "unused-freeze", tmp_path / "unused-review", output)
    assert "synthetic cleanup failure" in str(caught.value.__notes__)
    result = json.loads((output / "result.json").read_text())
    assert result["rows_complete"] == 72
    assert "synthetic cleanup failure" in result["cleanup_error"]
    assert stats["closes"] == 1
    assert body._step_physics.__func__ is FakeBody._step_physics
