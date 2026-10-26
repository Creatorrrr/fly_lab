"""E228 command/window and failure tests using synthetic CPU-only objects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import probe_c_banc_m3_local_command_hold as hold


def test_exact_rf_rh_window_and_unmodified_columns():
    source = np.arange(42, dtype=float)
    anchor = source + 100.
    for arm, columns in (("rf_hold", range(21, 28)),
                         ("rh_hold", range(35, 42))):
        for tick in (1619, 1620, 1635, 1636):
            applied = hold.hold_command(source, anchor, arm, tick)
            expected = source.copy()
            if 1620 <= tick <= 1635:
                expected[list(columns)] = anchor[list(columns)]
            np.testing.assert_array_equal(applied, expected)
    np.testing.assert_array_equal(hold.hold_command(source, anchor, "sham", 1620),
                                  source)
    with pytest.raises(ValueError, match="outside"):
        hold.validate_hold(source, np.zeros(42), anchor, "rf_hold", 1620)


def test_native_global_and_local_boundaries():
    assert hold.source_slice(1608) == slice(50350, 50400)
    assert hold.source_slice(1647) == slice(52300, 52350)
    assert hold.local_slice(1608) == slice(0, 50)
    assert hold.local_slice(1647) == slice(1950, 2000)
    with pytest.raises(ValueError):
        hold.source_slice(1607)


def test_phase_signed_force_and_slip_nullable():
    native = {"time_s": 8.035 + (np.arange(2000) + 1) * .0001,
              "normal_load_bw": np.zeros((2000, 6)),
              "slip_mm_s": np.full((2000, 6), np.nan),
              "slip_valid": np.zeros((2000, 6), bool),
              "nonfoot_contact": np.zeros(2000, bool),
              "upright": np.ones(2000)}
    native["normal_load_bw"][:600, 3] = .03
    native["slip_mm_s"][:600, 3] = 2.
    native["slip_valid"][:600, 3] = True
    force = {"time_s": native["time_s"],
             "available": np.ones(2000, bool),
             "all_leg_heading_xy_force_bw": np.zeros((2000, 6))}
    force["all_leg_heading_xy_force_bw"][:600, 3] = np.tile([2., -3.], 300)
    rows = [{"joint_tracking_error_rad": [0.] * 42} for _ in range(41)]
    metrics = hold.phase_metrics(native, force, rows, {"lead": [1608, 1620]})
    lead = metrics["lead"]
    assert lead["native_sample_count"] == 600
    assert lead["positive_supported_BW_s_by_leg"][3] == pytest.approx(.06)
    assert lead["negative_supported_BW_s_by_leg"][3] == pytest.approx(.09)
    assert lead["net_supported_BW_s_by_leg"][3] == pytest.approx(-.03)
    assert lead["supported_slip_mean_mm_s_by_leg"][3] == 2.
    assert lead["supported_slip_mean_mm_s_by_leg"][0] is None
    assert lead["supported_slip_valid_count_by_leg"][0] == 0


def test_gradient_and_frozen_interior_criteria():
    heading = np.array([1., 0., 0.])
    controls = list(range(1607, 1648))
    # UI native x equals x; body displacement is in millimeters.
    values = np.asarray([(tick - 1607) * .001 for tick in controls])
    rows = [{"control_tick": tick, "body": {"position": [value, 0., 0.],
             "yaw": 0.}} for tick, value in zip(controls, values, strict=True)]
    metrics = hold.trajectory_metrics(rows, heading, list(range(1620, 1647)))
    assert len(metrics["initial_forward_speed_mm_s"]) == 41
    assert metrics["summary_controls"] == list(range(1620, 1647))
    assert metrics["min_initial_forward_speed_mm_s"] == pytest.approx(.2)
    assert metrics["initial_forward_displacement_mm"] == pytest.approx(.04)
    assert metrics["backward_distance_mm"] == 0.


def test_zero_denominators_and_rf_leg_index_never_support_automatically():
    thresholds = {"backward_distance_reduction_fraction": .25,
                  "minimum_forward_speed_gain_mm_s": .5,
                  "maximum_forward_distance_loss_mm": .02,
                  "rf_negative_integral_reduction_fraction": .2}
    base = {"trajectory": {"backward_distance_mm": 0.,
            "min_initial_forward_speed_mm_s": -2.,
            "initial_forward_displacement_mm": .01},
            "phases": {"during_break": {
                "negative_supported_BW_s_by_leg": [0., 0., 0., 0., 0., 0.]}}}
    new = {"trajectory": {"backward_distance_mm": 0.,
            "min_initial_forward_speed_mm_s": -1.,
            "initial_forward_displacement_mm": .02},
           "phases": {"during_break": {
                "negative_supported_BW_s_by_leg": [0., 0., 0., 0., 0., 0.]}}}
    outcome = hold.assess_local(base, new, arm="rf_hold", thresholds=thresholds)
    assert outcome["backward_distance_reduction_fraction"] is None
    assert outcome["RF_negative_force_reduction_fraction"] is None
    assert not outcome["local_body_improvement_supported"]
    assert not outcome["RF_command_mechanical_contribution_supported"]
    assert not hold.assess_local(base, None, arm="rh_hold", thresholds=thresholds)[
        "available"]
    base["trajectory"]["backward_distance_mm"] = 1.
    base["phases"]["during_break"]["negative_supported_BW_s_by_leg"][3] = .1
    new["phases"]["during_break"]["negative_supported_BW_s_by_leg"][3] = .07
    outcome = hold.assess_local(base, new, arm="rf_hold", thresholds=thresholds)
    assert outcome["RF_negative_force_reduction_fraction"] == pytest.approx(.3)
    assert outcome["RF_command_mechanical_contribution_supported"]


def test_gate_fails_before_body(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        hold.gate(tmp_path / "missing", "bad", tmp_path / "missing-review",
                  "bad", tmp_path / "out")


def test_partial_native_difference_helper_uses_actual_first_sample():
    reference = {"time_s": np.array([1., 2.]),
        "normal_load_bw": np.zeros((2, 6)),
        "slip_mm_s": np.full((2, 6), np.nan),
        "slip_valid": np.zeros((2, 6), bool),
        "nonfoot_contact": np.zeros(2, bool),
        "upright": np.ones(2)}
    samples = [(1., np.array([0., 0., 0., 1., 0., 0.]),
                np.full(6, np.nan), np.zeros(6, bool), False, 1.)]
    assert hold.first_native_divergence(samples, reference, source_start=0) == 0


def test_mock_sham_then_partial_rf_health_then_identical_reset_rh(
        monkeypatch, tmp_path):
    native_ref = {"time_s": 3. + (np.arange(53150) + 1) * .0001,
        "normal_load_bw": np.zeros((53150, 6)),
        "slip_mm_s": np.full((53150, 6), np.nan),
        "slip_valid": np.zeros((53150, 6), bool),
        "nonfoot_contact": np.zeros(53150, bool),
        "upright": np.ones(53150)}
    source_row = {"control_tick": 1607, "joint_targets_rad": [0.] * 42,
        "adhesion": [False] * 6, "tendon_inputs": {},
        "qpos": [0.], "ctrl": [0.]}
    reference = [source_row] * 1664
    start = {"modelHash": "same"}

    class FakeBody:
        model_hash = "same"

        def __init__(self):
            self.arm_index = -1
            self.time = 8.035
            self.tick = 1607
            self.starts = []
            self.closed = False
            self.native = None
            self.forces = None

        def restore(self, _state):
            self.arm_index += 1
            self.time = 8.035
            self.tick = 1607
            self.starts.append(self.time)

        def snapshot(self):
            return dict(start)

        def physics_time(self):
            return self.time

        def step_joint_targets(self, *_args, **_kwargs):
            tick = self.tick + 1
            count = 1 if self.arm_index == 1 and tick == 1620 else 50
            for substep in range(count):
                source_index = hold.source_slice(tick).start + substep
                now = float(native_ref["time_s"][source_index])
                load = np.zeros(6)
                if self.arm_index == 1 and tick == 1620:
                    load[3] = 1.
                self.native.append((now, load, np.full(6, np.nan),
                                    np.zeros(6, bool), False, 1.))
                force = {"time_s": now, "available": True, "error": ""}
                force.update({key: np.zeros(shape)
                    for key, shape in hold.replay.FORCE_SHAPES.items()})
                self.forces.append(force)
                self.time = now
            if count == 1:
                raise hold.e113.NativeHealthStop("generated RF partial health")
            self.tick = tick

        def close(self):
            self.closed = True

    body = FakeBody()

    class FakeStore:
        @staticmethod
        def load(path):
            return ({"body": dict(start)} if "base" in str(path)
                    else dict(start))

        @staticmethod
        def save(path, _value):
            Path(path).mkdir()
            (Path(path) / "mock.txt").write_text("mock", encoding="utf-8")

    class ThreadLimit:
        def restore_original_limits(self):
            return None

    def install(target, samples, sink, *_args):
        target.native, target.forces = samples, sink
        return lambda: None

    def physical(_body, tick, _tendons):
        return {"control_tick": tick,
            "body": {"position": [(tick - 1607) * .001, 0., 0.], "yaw": 0.},
            "joint_tracking_error_rad": [0.] * 42,
            "qpos": [0.], "ctrl": [0.]}

    freeze = {"resource_limits": {"internal_s": 25.,
              "finalization_reserve_s": 5., "resident_bytes": 6 * 2**30,
              "output_bytes": 32 * 2**20},
              "force_fields": ["time_s", "available", "error",
                               *hold.replay.FORCE_SHAPES],
              "diagnostic_thresholds": {
                  "backward_distance_reduction_fraction": .25,
                  "minimum_forward_speed_gain_mm_s": .5,
                  "maximum_forward_distance_loss_mm": .02,
                  "rf_negative_integral_reduction_fraction": .2},
              "event": {"velocity_summary_controls": list(range(1620, 1647))},
              "phases": {"lead": [1608, 1620], "hold": [1620, 1636],
                         "recovery": [1636, 1648], "during_break": [1623, 1631]},
              "native": {"support_threshold_bw": .02}}
    monkeypatch.setattr(hold, "gate", lambda *_args: (freeze, {"approved": True}))
    monkeypatch.setattr(hold, "load_reference", lambda _freeze: (
        {"base_checkpoint": tmp_path / "base", "body_checkpoint": tmp_path / "event"},
        reference, native_ref, {}))
    monkeypatch.setattr(hold, "StateStore", FakeStore)
    monkeypatch.setattr(hold.observation, "new_body", lambda _base: body)
    monkeypatch.setattr(hold.replay, "initial_horizontal_heading", lambda _row:
                        np.array([1., 0., 0.]))
    monkeypatch.setattr(hold.replay, "capture_physical_row", physical)
    monkeypatch.setattr(hold.replay, "compare_physical_row", lambda *_args: None)
    monkeypatch.setattr(hold.replay, "compare_native_chunk", lambda *_args: None)
    monkeypatch.setattr(hold, "compare_force_chunk", lambda *_args: None)
    monkeypatch.setattr(hold.ground, "install_native_observer", install)
    from tools import fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (ThreadLimit(), {}))
    out = tmp_path / "out"
    result = hold.run(tmp_path / "freeze", "", tmp_path / "review", "", out)
    assert result["validity"] == "measured_native_health_stop"
    assert list(result["arms"]) == ["sham", "rf_hold", "rh_hold"]
    assert [result["arms"][arm]["status"] for arm in hold.ARMS] == [
        "complete_healthy", "measured_native_health_stop", "complete_healthy"]
    assert result["arms"]["rf_hold"]["first_native_difference_sample"] == 50950
    assert result["arms"]["rf_hold"]["first_qpos_difference_control"] is None
    assert result["arms"]["rf_hold"]["first_ctrl_difference_control"] is None
    assert result["local_comparisons"]["rf_hold"]["available"] is False
    assert body.starts == [8.035] * 3
    assert body.closed
    ledger = json.loads((out / "advance-ledger.json").read_text(encoding="utf-8"))
    assert ledger["scientific_steps_observed"] == 4000 + 601
    with np.load(out / "arms/rf_hold/native.npz") as raw:
        assert len(raw["time_s"]) == 601
    with np.load(out / "arms/rh_hold/native.npz") as raw:
        assert len(raw["time_s"]) == 2000


@pytest.mark.parametrize("health", [True, False])
def test_mock_partial_sham_never_launches_hold(monkeypatch, tmp_path, health):
    native = None
    forces = None
    start = {"modelHash": "same"}

    class FakeBody:
        model_hash = "same"

        def __init__(self):
            self.time = 8.035
            self.closed = False
            self.calls = 0

        def restore(self, _snapshot):
            self.time = 8.035

        def snapshot(self):
            return dict(start)

        def physics_time(self):
            return self.time

        def step_joint_targets(self, *_args, **_kwargs):
            self.calls += 1
            self.time += .0001
            native.append((self.time, np.ones(6), np.full(6, np.nan),
                           np.zeros(6, bool), False, 1.))
            forces.append({"time_s": self.time, "available": False,
                           "error": "generated"})
            if health:
                raise hold.e113.NativeHealthStop("generated health")
            raise RuntimeError("generated technical")

        def close(self):
            self.closed = True

    class FakeStore:
        @staticmethod
        def load(path):
            return ({"body": dict(start)} if "base" in str(path)
                    else dict(start))

        @staticmethod
        def save(path, _value):
            Path(path).mkdir()
            (Path(path) / "mock.txt").write_text("mock", encoding="utf-8")

    class ThreadLimit:
        def restore_original_limits(self):
            return None

    body = FakeBody()

    def install(_body, samples, sink, *_args):
        nonlocal native, forces
        native, forces = samples, sink
        return lambda: None

    freeze = {"resource_limits": {"internal_s": 25.,
              "finalization_reserve_s": 5., "resident_bytes": 6 * 2**30,
              "output_bytes": 32 * 2**20},
              "force_fields": ["time_s", "available", "error",
                               *hold.replay.FORCE_SHAPES],
              "diagnostic_thresholds": {}, "event": {}, "phases": {}}
    row = {"control_tick": 1608, "joint_targets_rad": [0.] * 42,
           "adhesion": [False] * 6, "tendon_inputs": {}}
    monkeypatch.setattr(hold, "gate", lambda *_args: (freeze, {"approved": True}))
    monkeypatch.setattr(hold, "load_reference", lambda _freeze: (
        {"base_checkpoint": tmp_path / "base", "body_checkpoint": tmp_path / "event"},
        [row] * 1664, {}, {}))
    monkeypatch.setattr(hold, "StateStore", FakeStore)
    monkeypatch.setattr(hold.observation, "new_body", lambda _saved: body)
    monkeypatch.setattr(hold.replay, "initial_horizontal_heading", lambda _row:
                        np.array([1., 0., 0.]))
    monkeypatch.setattr(hold.replay, "capture_physical_row", lambda *_args: {
        "control_tick": 1607, "time_s": 8.035})
    monkeypatch.setattr(hold.replay, "compare_physical_row", lambda *_args: None)
    monkeypatch.setattr(hold.ground, "install_native_observer", install)
    from tools import fit_c_banc_m3_joint_adaptive_recruitment as threads
    monkeypatch.setattr(threads, "one_cpu_thread", lambda: (ThreadLimit(), {}))
    out = tmp_path / "out"
    if health:
        result = hold.run(tmp_path / "freeze", "", tmp_path / "review", "", out)
        assert result["validity"] == "measured_native_health_stop"
        assert result["execution_status"] == "complete"
    else:
        with pytest.raises(RuntimeError, match="generated technical"):
            hold.run(tmp_path / "freeze", "", tmp_path / "review", "", out)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert result["validity"] == "technical_failure"
    assert body.calls == 1
    assert body.closed
    assert list((out / "arms").iterdir()) == [out / "arms" / "sham"]
    ledger = json.loads((out / "advance-ledger.json").read_text(encoding="utf-8"))
    assert ledger["scientific_controls_requested"] == 1
    assert ledger["scientific_steps_requested"] == 50
    assert ledger["scientific_steps_observed"] == 1
    with np.load(out / "arms/sham/native.npz") as raw:
        assert len(raw["time_s"]) == 1
