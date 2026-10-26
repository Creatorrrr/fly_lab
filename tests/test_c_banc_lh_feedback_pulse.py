"""Small generated contracts for E261; no graph, body, or CUDA construction."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_feedback_pulse as pulse


class FakeBody:
    def __init__(self):
        self.calls = []
        self.joint_ids = np.arange(42)
        self.act_ids = np.arange(42)
        self.m = SimpleNamespace(jnt_range=np.tile([-1., 1.], (42, 1)),
            actuator_ctrlrange=np.tile([-1., 1.], (42, 1)),
            actuator_ctrllimited=np.ones(42, bool))

    def step_joint_targets(self, target, adhesion, **kwargs):
        self.calls.append((target, adhesion, kwargs))
        return len(self.calls)


def test_pulse_only_same_control_one_joint_and_restores_binding():
    body = FakeBody()
    session = SimpleNamespace(body=body, control_tick=55)
    previous = body.__dict__.get("step_joint_targets")
    events = []
    detach = pulse.attach_pulse(session, events, 1, .01)
    original = np.arange(42, dtype=float) / 100
    tendon = {"lh_tarsus": .2}
    assert body.step_joint_targets(original, False, tendon_inputs=tendon, interpolate=True) == 1
    assert body.calls[0][0][19] == original[19] + .01
    assert np.array_equal(body.calls[0][0][np.arange(42) != 19],
                          original[np.arange(42) != 19])
    assert np.array_equal(original, np.arange(42, dtype=float) / 100)
    assert body.calls[0][1:] == (False, {"tendon_inputs": tendon, "interpolate": True})
    session.control_tick = 56
    assert body.step_joint_targets(original, False, tendon_inputs=tendon) == 2
    assert body.calls[1][0] is original
    assert [row["changed"] for row in events] == [True, False]
    detach()
    assert body.__dict__.get("step_joint_targets") is previous


def test_pulse_original_error_propagates_and_binding_restored():
    class Broken(FakeBody):
        def step_joint_targets(self, target, adhesion, **kwargs):
            raise RuntimeError("physical failure")

    body = Broken()
    session = SimpleNamespace(body=body, control_tick=55)
    events = []
    detach = pulse.attach_pulse(session, events, -1, .01)
    with pytest.raises(RuntimeError, match="physical failure"):
        body.step_joint_targets(np.zeros(42), False, tendon_inputs={})
    assert len(events) == 1 and not events[0]["body_call_returned"]
    detach()
    assert "step_joint_targets" not in body.__dict__


def test_pulse_rejects_actuator_clipping_before_physics():
    body = FakeBody()
    body.m.actuator_ctrlrange[19] = [-.001, .001]
    session = SimpleNamespace(body=body, control_tick=55)
    events = []
    detach = pulse.attach_pulse(session, events, 1, .01)
    with pytest.raises(ValueError, match="clipped by actuator"):
        body.step_joint_targets(np.zeros(42), np.zeros(6, bool))
    assert not body.calls and not events
    detach()


def test_restore_in_place_exact_and_wrong_component_rejected(monkeypatch):
    class Component:
        def __init__(self):
            self.value = 1

        def restore(self, value):
            self.value = value

    body, network, adapter = Component(), Component(), Component()
    network.readout = lambda ids: np.array([2], np.float32)
    adapter.motor_ids = np.array([1])
    adapter.tendon_inputs = {"t": .3}
    session = SimpleNamespace(body=body, network=network, adapter=adapter,
        tendon_inputs={"t": .9}, control_tick=99,
        initial_position=np.array([9]), signed_forward_mm=9., wall_s=9.,
        fault="old", rate=None, navigation=None)
    session.set_cuts = lambda cuts: setattr(session, "cuts", cuts.copy())
    session._clocks = lambda: None
    session.snapshot = lambda: {"schema": "trial", "body": body.value,
        "network": network.value, "adapter": adapter.value,
        "cuts": session.cuts, "control_tick": session.control_tick,
        "initial_position": session.initial_position, "signed_forward_mm": session.signed_forward_mm,
        "wall_s": session.wall_s, "navigation": None}
    monkeypatch.setattr(pulse.e257.LHPartitionTrial, "schema", "trial")
    fork = {"schema": "trial", "body": 7, "network": 8, "adapter": 9,
            "cuts": {"motor": False}, "control_tick": 55,
            "initial_position": np.array([1]), "signed_forward_mm": 1.,
            "wall_s": 2., "navigation": None}
    pulse.restore_in_place(session, fork)
    assert (body.value, network.value, adapter.value, session.control_tick) == (7, 8, 9, 55)
    assert session.tendon_inputs == {"t": .3} and session.fault is None
    wrong = {**fork, "body": 999}
    body.restore = lambda value: None
    with pytest.raises(AssertionError):
        pulse.restore_in_place(session, wrong)


def _rows(delta):
    rows = []
    for tick in range(55, 76):
        q = np.zeros(42)
        if tick == 56:
            q[19] = delta
        rows.append({"control_tick": tick, "joint_angles_rad": q})
    return rows


def _events(sign, delayed):
    events = []
    for tick in range(56, 76):
        original = np.zeros(42)
        if tick >= 58:
            original[19] = delayed
        transmitted = original.copy()
        if tick == 56:
            transmitted[19] += sign * .01
        events.append({"control_tick": tick, "original_command": original,
                       "transmitted_command": transmitted,
                       "changed": tick == 56, "body_call_returned": True})
    return events


@pytest.mark.parametrize(("value", "expected"), [
    (.001, "supported_positive_return"),
    (-.001, "refuted_compensatory_return"),
    (0., "inconclusive_below_resolution")])
def test_return_uses_controls_58_to_75_original_decoder_only(value, expected):
    outcome = pulse.contrast(_rows(.004), _rows(-.004),
                             _events(1, value), _events(-1, 0.))
    assert outcome["classification"] == expected
    assert outcome["physical_actual_plus_minus_q19_rad"] == .008
    assert outcome["H261_supported"] == (expected == "supported_positive_return")


def test_short_native_health_missing_clock_and_fault_not_supported():
    class Clock:
        clock_valid = True
    rows = [{"time_s": i * .005, "fault": None,
             "qpos": np.zeros(109), "qvel": np.zeros(108), "ctrl": np.zeros(82),
             "joint_angles_rad": np.zeros(42), "joint_targets_rad": np.zeros(42),
             "joint_tracking_error_rad": np.zeros(42), "motor_rates": np.zeros(391),
             "body": {"position": np.zeros(3), "basis": np.eye(3)}}
            for i in range(55, 76)]
    native = {"time_s": np.arange(2751, 3751) * .0001,
              "upright": np.ones(1000), "normal_load_bw": np.ones((1000, 6)),
              "nonfoot_contact": np.zeros(1000, bool)}
    assert pulse.diagnostic_health(rows, native, Clock(), 55, 75, False)["healthy"]
    native["upright"][200] = np.nan
    assert not pulse.diagnostic_health(rows, native, Clock(), 55, 75, False)["healthy"]
    native["upright"][200] = 1.
    assert not pulse.diagnostic_health(rows, native, Clock(), 55, 75, True)["healthy"]
    rows[3]["qpos"] = np.array([np.nan, 0.])
    assert not pulse.diagnostic_health(rows, native, Clock(), 55, 75, False)["healthy"]
    rows[3]["qpos"] = np.zeros(109)
    del rows[4]["qpos"]
    assert not pulse.diagnostic_health(rows, native, Clock(), 55, 75, False)["healthy"]


def test_legacy_raw_body_command_is_pulsed_but_decoder_series_is_not():
    target = np.zeros(42, np.float64)
    target[19] = .001
    adapter = SimpleNamespace(calibration={"motor_ids": np.arange(391)},
                              activation=np.zeros(96))
    network = SimpleNamespace(tick=1120,
        readout=lambda ids: np.zeros(391, np.float32))
    body = SimpleNamespace(physics_time=lambda: .28)
    session = SimpleNamespace(adapter=adapter, network=network, body=body,
                              control_tick=56)
    row = {"joint_targets_rad": target,
           "tendon_inputs": {name: .2 for name in pulse.e257.TENDONS}}
    raw = pulse.e257.capture_control_raw(session, row)
    events = _events(1, 0.)
    events[0]["transmitted_command"] = target
    series = pulse.decoded_target_series([{"command": np.zeros(42)}, raw],
                                          events[:1])
    assert raw["command"][19] == .001
    assert series.tolist() == [0., 0.]


def test_late_result_failure_revokes_support_and_preserves_primary(tmp_path):
    out = tmp_path
    ledger = {"segments": {}}
    result = {"validity": "valid_short_closed_loop_diagnostic",
              "H261": {"H261_supported": True,
                       "classification": "supported_positive_return"}}
    errors = []
    primary = RuntimeError("first science error")
    pulse.write_json(out / "contrast.json", {"H261_supported": True,
        "classification": "supported_positive_return",
        "decoder_return_mean_plus_minus_q19_rad": .001})
    np.savez(out / "contrast.npz", decoded_q19_plus=np.array([.001]))
    def writer(path, value):
        if path.name == "result.json":
            raise OSError("late publication")
        pulse.write_json(path, value)
    returned = pulse.finalize(out, ledger, result, errors, primary,
                              pulse.time.perf_counter(), 1_000_000, writer=writer)
    assert returned is primary
    assert not result["H261"]["H261_supported"]
    assert not (out / "result.json").exists()
    assert json.loads((out / "errors.json").read_text())[0]["stage"] == "result.json"
    saved = json.loads((out / "contrast.json").read_text())
    assert saved["H261_supported"] is False
    assert saved["classification"] == "unvalidated_after_failure"
    assert (out / "contrast.npz").exists()


def test_persistent_contrast_rewrite_falls_back_to_unvalidated_file(tmp_path):
    pulse.write_json(tmp_path / "contrast.json", {"H261_supported": True,
        "classification": "supported_positive_return"})
    result = {"validity": "valid_short_closed_loop_diagnostic",
              "H261": {"H261_supported": True}}
    def writer(path, value):
        if path.name == "contrast.json":
            raise OSError("contrast writer failed")
        pulse.write_json(path, value)
    primary = pulse.finalize(tmp_path, {}, result, [], RuntimeError("science"),
                             pulse.time.perf_counter(), 1_000_000, writer=writer)
    assert str(primary) == "science"
    assert not (tmp_path / "contrast.json").exists()
    assert (tmp_path / "contrast-unvalidated.json").exists()
    assert not result["H261"]["H261_supported"]
