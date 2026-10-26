"""Small synthetic E227 body-replay and force-accounting checks.

These tests never construct FlyGymBody or a neural network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import observe_c_banc_m3_effector_force_replay as replay


def force_row(time_s=3.0001, *, available=True):
    row = {"time_s": time_s, "available": available, "error": ""}
    if available:
        row.update({key: np.ones(shape, np.float64)
                    for key, shape in replay.FORCE_SHAPES.items()})
        row["segment_force"] = np.ones((30, 3))
    return row


def test_force_sink_keeps_only_frozen_fields_and_nulls_missing():
    fields = ("time_s", "available", "error", *replay.FORCE_SHAPES)
    sink = replay.ForceSink(fields)
    sink.append(force_row())
    sink.append({"time_s": 3.0002, "available": False, "error": "shadow failed"})
    arrays = sink.arrays()
    assert set(arrays) == set(fields)
    assert arrays["all_leg_force_bw"].shape == (2, 6, 3)
    assert arrays["full_fly_heading_xy_force_bw"].shape == (2,)
    assert np.isnan(arrays["all_leg_force_bw"][1]).all()
    assert arrays["error"].tolist() == ["", "shadow failed"]
    with pytest.raises(ValueError, match="Complete"):
        sink.append({"time_s": 3.0003, "available": True, "error": ""})


def native_row(time_s, slip=np.nan):
    loads = np.ones(6, np.float64)
    slip_values = np.full(6, slip, np.float64)
    return (time_s, loads, slip_values, np.isfinite(slip_values), False, 1.)


def test_native_chunk_preserves_nan_bits_and_drift():
    rows = [native_row(3.0001 + index * .0001 + 1e-12)
            for index in range(50)]
    reference = replay.e113.native_arrays(rows)
    replay.compare_native_chunk(rows, reference, 0, 50)
    bad = {key: array.copy() for key, array in reference.items()}
    bad["time_s"][0] += 1e-12
    with pytest.raises(ValueError, match="exact native"):
        replay.compare_native_chunk(rows, bad, 0, 50)
    with pytest.raises(ValueError, match="sample count"):
        replay.compare_native_chunk(rows[:-1], reference, 0, 50)


def test_physical_row_excludes_unreplayed_neural_and_navigation():
    candidate = {"control_tick": 601, "time_s": 3.005}
    candidate.update({key: {"value": 1.} for key in replay.PHYSICAL_KEYS})
    reference = {**candidate, "motor_rates": [77.],
                 "navigation": {"different": True}, "neural_request": None}
    replay.compare_physical_row(candidate, reference)
    broken = {**candidate, "ctrl": {"value": 2.}}
    with pytest.raises(AssertionError, match="ctrl"):
        replay.compare_physical_row(broken, reference)
    with pytest.raises(ValueError, match="fields"):
        replay.compare_physical_row({**candidate, "motor_rates": [77.]}, reference)


def test_heading_uses_original_control_zero_native_xy():
    # UI = S @ native; a pitched native vector must be projected to XY.
    native_axis = np.array([3., 4., 2.]) / np.sqrt(29.)
    ui_basis = np.column_stack((replay.S @ native_axis,
                                replay.S @ np.array([0., 0., 1.]),
                                replay.S @ np.array([4., -3., 0.]) / 5.))
    heading = replay.initial_horizontal_heading({"body": {"basis": ui_basis}})
    np.testing.assert_allclose(heading, [3. / 5., 4. / 5., 0.])


def test_signed_force_is_split_before_mean_and_unsupported_is_nullable():
    count = 53150
    force = {"time_s": 3. + (np.arange(count) + 1) * .0001,
             "available": np.ones(count, bool),
             "error": np.full(count, "", dtype="<U1")}
    for key, shape in replay.FORCE_SHAPES.items():
        force[key] = np.zeros((count, *shape), np.float64)
    force["heading_xy"][:, 0] = 1.
    force["all_leg_heading_xy_force_bw"][:600, 0] = np.tile([2., -3.], 300)
    force["all_leg_force_bw"][:600, 0, 0] = np.tile([2., -3.], 300)
    reference = {"time_s": force["time_s"].copy(),
                 "normal_load_bw": np.zeros((count, 6), np.float64)}
    reference["normal_load_bw"][:600, 0] = .03
    event = {"entry_index": 617, "exit_index": 621}
    frozen = {"pre_ticks": [-16, -4], "post_ticks": [4, 16],
              "support_threshold_bw": .02}
    result = replay.force_windows(force, reference, [event], [0], frozen,
        {"body": {"basis": replay.S.tolist()}})
    pre = result["events"][0]["windows"]["pre"]
    assert pre["native_samples"] == 600
    assert pre["current_horizontal"]["positive_supported_BW_s"][0] == pytest.approx(.06)
    assert pre["current_horizontal"]["negative_supported_BW_s"][0] == pytest.approx(.09)
    assert pre["current_horizontal"]["negative_share"][0] == 1.
    assert pre["current_horizontal"]["mean_supported_BW"][1] is None
    assert pre["supported_time_s"][1] == 0.
    assert result["events"][0]["windows"]["post"]["current_horizontal"][
        "negative_share"] is None


def test_gate_fails_before_body_construction(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.gate(tmp_path / "missing-freeze", "wrong",
                    tmp_path / "missing-review", "wrong", tmp_path / "out")


def test_mock_partial_step_retains_raw_and_never_saves_event_snapshot(
        monkeypatch, tmp_path):
    native = None
    forces = None

    class FakeBody:
        model_hash = "same"

        def __init__(self):
            self.time = 3.
            self.closed = False

        def snapshot(self):
            return {"modelHash": "same"}

        def physics_time(self):
            return self.time

        def step_joint_targets(self, *_args, **_kwargs):
            self.time += .0001
            native.append(native_row(self.time))
            forces.append(force_row(self.time))
            raise RuntimeError("generated partial integration")

        def close(self):
            self.closed = True

    body = FakeBody()

    class FakeStore:
        @staticmethod
        def load(_path):
            return {"body": {"modelHash": "same"}}

        @staticmethod
        def save(path, _value):
            Path(path).mkdir()
            (Path(path) / "snapshot.txt").write_text("mock", encoding="utf-8")

    class ThreadLimit:
        def restore_original_limits(self):
            return None

    def install(_body, samples, sink, _inventory, _contacts, _health_stop):
        nonlocal native, forces
        native, forces = samples, sink
        return lambda: None

    row = {"control_tick": 600, "joint_targets_rad": [0.] * 42,
           "adhesion": [False] * 6, "tendon_inputs": {},
           "neural_request": None}
    freeze = {"resource_limits": {"internal_s": 65.,
              "finalization_reserve_s": 8., "resident_bytes": 6 * 2**30,
              "output_bytes": 128 * 2**20},
              "force_fields": ["time_s", "available", "error",
                               *replay.FORCE_SHAPES]}
    monkeypatch.setattr(replay, "gate", lambda *_args: (freeze, {"approved": True}))
    monkeypatch.setattr(replay, "load_reference", lambda _freeze: (
        {"checkpoint": tmp_path / "unused"}, [row] * 1664, {}, {}, [], {}))
    monkeypatch.setattr(replay, "StateStore", FakeStore)
    monkeypatch.setattr(replay.observation, "new_body", lambda _saved: body)
    monkeypatch.setattr(replay.ground, "fly_mass_inventory", lambda _body: {})
    monkeypatch.setattr(replay.ground, "install_native_observer", install)
    monkeypatch.setattr(replay, "capture_physical_row", lambda *_args: {
        "control_tick": 600, "time_s": 3.})
    monkeypatch.setattr(replay, "compare_physical_row", lambda *_args: None)
    from tools import fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (ThreadLimit(), {}))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="generated partial integration"):
        replay.run(tmp_path / "unused-freeze", "", tmp_path / "unused-review",
                   "", out)
    ledger = json.loads((out / "advance-ledger.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert ledger["scientific_controls_requested"] == 1
    assert ledger["scientific_controls_returned"] == 0
    assert ledger["scientific_steps_requested"] == 50
    assert ledger["scientific_steps_observed"] == 1
    assert ledger["snapshot_controls"] == [600]
    assert result["validity"] == "technical_failure"
    assert body.closed
    assert list(out.glob("event-start-*")) == []
    with np.load(out / "force-native.npz") as raw:
        assert len(raw["time_s"]) == 1
    with np.load(out / "replay-native-contact.npz") as raw:
        assert len(raw["time_s"]) == 1
