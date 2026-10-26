"""Small causal tests; no native model or CUDA is constructed."""

import copy
import math

import numpy as np
import pytest

from tools.banc_m3_directional_sensory import (
    HOOK_DEADBAND_RAD_S,
    HOOK_FUNCTION,
    HOOK_VELOCITY_UNITS,
    NativeSensoryTrial,
    NativeVelocityAdapter,
    native_feature,
    validate_trial_boundary,
)
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_native_sensory import NativeVelocityAdapter as E195Adapter
from tools.banc_m3_native_sensory import native_feature as e195_feature
from tools.verify_c_banc_observation import exact


class FakeLeaf:
    def __init__(self):
        self._receptor_keys = [(0, "hook", "SNpp41"), (0, "hook", "SNpp39"),
                               (0, "club", "x"), (0, "load", "x")]
        self._port_features = np.arange(4)
        self.joints = np.tile(np.arange(7), (6, 1))
        self.velocity_lowpass = np.zeros(42)
        self.filtered = np.array([.1, .2, .3, .4], np.float64)
        self.last_features = np.zeros(4)
        self.sensory_gain = 18.
        self.spec = {"load_half_bw": 1.}
        self.boundary_velocity = 0.
        self.calls = 0

    def encode(self, *, sensory_cut=False, dt=.005):
        self.calls += 1
        old = self.filtered.copy()
        features = np.array([e195_feature(kind, cell, 0., self.boundary_velocity,
                                         self.velocity_lowpass[5], self.spec)
                             for _, kind, cell in self._receptor_keys[:3]] + [.25])
        self.last_features = features
        self.filtered += (1 - math.exp(-dt / .02)) * (18 * features - self.filtered)
        self.velocity_lowpass[5] += (1 - math.exp(-dt / .03)) * (
            self.boundary_velocity - self.velocity_lowpass[5])
        return np.zeros(4, np.float32) if sensory_cut else old.astype(np.float32)

    def snapshot(self):
        return {"filtered": self.filtered.copy(), "velocity_lowpass": self.velocity_lowpass.copy(),
                "last_features": self.last_features.copy(), "calls": self.calls}

    def restore(self, saved):
        self.filtered[:] = saved["filtered"]
        self.velocity_lowpass[:] = saved["velocity_lowpass"]
        self.last_features[:] = saved["last_features"]
        self.calls = saved["calls"]


class FakeMotor:
    identity = "a" * 64

    def __init__(self):
        self.original = FakeLeaf()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def snapshot(self):
        return self.original.snapshot()

    def restore(self, saved):
        self.original.restore(saved)


def sample_window(adapter, velocities):
    for i, velocity in enumerate(velocities, 1):
        q = np.zeros(6)
        v = np.full(6, velocity)
        adapter.record((adapter._encode_count - 1) * .005 + i * .0001, q, v)


def test_native_nonlinearity_and_unchanged_drive_and_other_port():
    adapter = NativeVelocityAdapter(FakeMotor())
    leaf = adapter.original
    start = leaf.filtered.copy()
    np.testing.assert_array_equal(adapter.encode(), start.astype(np.float32))
    sample_window(adapter, [20., -20.] * 25)
    assert adapter.native_rows[0]["control_tick"] == 1
    assert adapter.native_rows[-1]["substep"] == 50
    old = leaf.filtered.copy()
    lp = leaf.velocity_lowpass.copy()
    leaf.boundary_velocity = 0.
    drive = adapter.encode()
    np.testing.assert_array_equal(drive, old.astype(np.float32))
    np.testing.assert_allclose(adapter._last_mean, [0.5, 0.5, .4], atol=1e-12)
    assert native_feature("hook", "SNpp41", 0., 0., 0., leaf.spec) == 0.
    assert native_feature("hook", "SNpp41", 0., HOOK_DEADBAND_RAD_S, 0., leaf.spec) == 0.
    assert native_feature("hook", "SNpp39", 0., -HOOK_DEADBAND_RAD_S, 0., leaf.spec) == 0.
    assert native_feature("hook", "SNpp41", 0., math.nextafter(HOOK_DEADBAND_RAD_S,
        math.inf), 0., leaf.spec) == 1.
    assert native_feature("hook", "SNpp39", 0., math.nextafter(-HOOK_DEADBAND_RAD_S,
        -math.inf), 0., leaf.spec) == 1.
    assert native_feature("hook", "unmapped", 0., 100., 0., leaf.spec) == 0.
    assert HOOK_FUNCTION and HOOK_VELOCITY_UNITS == "rad/s"
    assert native_feature("club", "x", 0., 20., 0., leaf.spec) == e195_feature(
        "club", "x", 0., 20., 0., leaf.spec)
    alpha = 1 - math.exp(-.005 / .02)
    np.testing.assert_allclose(leaf.filtered[:3], old[:3] + alpha * (18 *
        np.array([.5, .5, .4]) - old[:3]), atol=1e-12)
    np.testing.assert_allclose(leaf.filtered[3], old[3] + alpha * (18 * .25 - old[3]))
    np.testing.assert_array_equal(leaf.velocity_lowpass, lp)
    assert leaf.calls == 2
    assert adapter._count == 0


def test_e195_adapter_checkpoint_cannot_restore_into_e196():
    old = E195Adapter(FakeMotor()).snapshot()
    candidate = NativeVelocityAdapter(FakeMotor())
    assert old["hash"] != candidate.identity
    with pytest.raises(ValueError, match="identity"):
        candidate.restore(old)


def test_nonzero_boundary_first_two_drives_match_e195_then_diverge():
    old = E195Adapter(FakeMotor())
    candidate = NativeVelocityAdapter(FakeMotor())
    old.original.boundary_velocity = candidate.original.boundary_velocity = 10.
    first_old, first_candidate = old.encode(), candidate.encode()
    np.testing.assert_array_equal(first_old, first_candidate)
    np.testing.assert_array_equal(old.original.last_features,
                                  candidate.original.last_features)
    assert old.original.last_features[0] == .5  # Original linear hook at tick zero.
    for adapter in (old, candidate):
        sample_window(adapter, [10.] * 50)
    second_old, second_candidate = old.encode(), candidate.encode()
    np.testing.assert_array_equal(second_old, second_candidate)
    assert old.original.filtered[0] < candidate.original.filtered[0]
    for adapter in (old, candidate):
        sample_window(adapter, [10.] * 50)
    third_old, third_candidate = old.encode(), candidate.encode()
    assert third_old[0] < third_candidate[0]
    np.testing.assert_array_equal(third_old[2:], third_candidate[2:])


def test_incomplete_interval_cut_and_pending_roundtrip():
    adapter = NativeVelocityAdapter(FakeMotor())
    adapter.encode()
    sample_window(adapter, [10.] * 49)
    with pytest.raises(ValueError, match="complete preceding"):
        adapter.encode()
    adapter.record(.005, np.zeros(6), np.full(6, 10.))
    saved = copy.deepcopy(adapter.snapshot())
    restored = NativeVelocityAdapter(FakeMotor())
    restored.restore(saved)
    np.testing.assert_array_equal(restored.encode(sensory_cut=True), np.zeros(4, np.float32))
    np.testing.assert_array_equal(adapter.encode(sensory_cut=True), np.zeros(4, np.float32))
    np.testing.assert_array_equal(restored.original.filtered, adapter.original.filtered)
    assert restored._encode_count == 2
    assert saved["sample_count"] == 50
    tampered = copy.deepcopy(saved)
    tampered["sample_count"] = 49
    with pytest.raises(ValueError):
        NativeVelocityAdapter(FakeMotor()).restore(tampered)
    empty = NativeVelocityAdapter(FakeMotor()).snapshot()
    empty["start_s"], empty["end_s"] = 10., 20.
    with pytest.raises(ValueError, match="empty native interval"):
        NativeVelocityAdapter(FakeMotor()).restore(empty)


def test_partial_pending_snapshot_has_identical_future_inputs():
    original = NativeVelocityAdapter(FakeMotor())
    original.encode()
    sample_window(original, [20., -20.] * 12)
    original.record(.0025, np.zeros(6), np.full(6, 20.))
    clone = NativeVelocityAdapter(FakeMotor())
    clone.restore(copy.deepcopy(original.snapshot()))
    for i in range(26, 51):
        value = -20. if i % 2 == 0 else 20.
        original.record(i * .0001, np.zeros(6), np.full(6, value))
        clone.record(i * .0001, np.zeros(6), np.full(6, value))
    np.testing.assert_array_equal(original.encode(), clone.encode())
    np.testing.assert_array_equal(original.original.filtered, clone.original.filtered)
    np.testing.assert_array_equal(original.original.last_features,
                                  clone.original.last_features)


def test_actual_endpoint_float_drift_and_boundary_contract():
    adapter = NativeVelocityAdapter(FakeMotor())
    adapter.encode()
    # The body subtracts a settled 0.2 s origin from its floating MuJoCo time.
    t = .2
    for control in range(1, 31):
        for substep in range(50):
            t += .0001
            adapter.record(t - .2, np.zeros(6), np.full(6, 10. if substep % 2 else -10.))
        assert adapter.native_rows[-1]["control_tick"] == control
        assert adapter._count == 50
        np.testing.assert_allclose(adapter._duration,
                                   t - .2 - adapter._start_s, atol=1e-15, rtol=0)
        adapter.encode()
    assert adapter._encode_count == 31
    validate_trial_boundary(0, 0, 0)
    validate_trial_boundary(30, 30, 50)
    for values in ((0, 1, 0), (1, 1, 0), (1, 2, 50), (0, 0, 50)):
        with pytest.raises(ValueError, match="boundary"):
            validate_trial_boundary(*values)


def test_all_six_leg_port_mapping():
    motor = FakeMotor()
    leaf = motor.original
    leaf._receptor_keys = [(leg, "hook", "SNpp41") for leg in range(6)]
    leaf._port_features = np.arange(6)
    leaf.filtered = np.zeros(6)
    leaf.last_features = np.zeros(6)
    adapter = NativeVelocityAdapter(motor)
    adapter._encode_count = 1  # The fake has no motor call for this mapping-only probe.
    adapter.record(.0001, np.arange(6, dtype=float),
                   np.array([0., 2., 4., 6., 8., 10.]))
    np.testing.assert_allclose(adapter.native_rows[0]["features"],
                               [0., 1., 1., 1., 1., 1.], atol=1e-12)


def test_bad_native_sample_retains_prior_raw_rows():
    adapter = NativeVelocityAdapter(FakeMotor())
    adapter.encode()
    adapter.record(.0001, np.zeros(6), np.ones(6))
    with pytest.raises(ValueError, match="six-knee"):
        adapter.record(.0002, np.zeros(5), np.ones(6))
    assert len(adapter.native_rows) == 2
    assert adapter.native_rows[0]["complete"] is True
    assert adapter.native_rows[1]["complete"] is False
    assert "six-knee" in adapter.native_rows[1]["error"]


def test_whole_trial_unwrap_restore_and_next_encode_are_exact(monkeypatch):
    """Exercise the real candidate factory with synthetic underlying state only."""
    restored_sources = []

    def make_base():
        base = object.__new__(MonotoneRecruitmentTrial)
        base.closed = False
        base.adapter = FakeMotor()
        base.control_tick = 0
        base.network_fixture = np.array([1., 2.], dtype=np.float32)
        base.body_fixture = np.array([3., 4.], dtype=np.float64)
        return base

    def snapshot(owner):
        return {"schema": owner.schema, "control_tick": owner.control_tick,
                "adapter": owner.adapter.snapshot(),
                "network": owner.network_fixture.copy(), "body": owner.body_fixture.copy()}

    def from_checkpoint(cls, graph, saved):
        restored_sources.append(copy.deepcopy(saved))
        assert saved["schema"] == MonotoneRecruitmentTrial.schema
        assert "directional_sensory_patch" not in saved and "motor" not in saved["adapter"]
        base = make_base()
        base.adapter.restore(saved["adapter"])
        base.control_tick = saved["control_tick"]
        base.network_fixture[:] = saved["network"]
        base.body_fixture[:] = saved["body"]
        return base

    monkeypatch.setattr(MonotoneRecruitmentTrial, "snapshot", snapshot)
    monkeypatch.setattr(MonotoneRecruitmentTrial, "from_checkpoint", classmethod(from_checkpoint))
    monkeypatch.setattr(MonotoneRecruitmentTrial, "close", lambda owner: setattr(owner, "closed", True))
    monkeypatch.setattr(NativeSensoryTrial, "_install_native_observer", lambda owner: None)
    base = make_base()
    trial = NativeSensoryTrial(base)
    assert base.closed
    trial.adapter.encode()
    sample_window(trial.adapter, [20., -20.] * 25)
    trial.control_tick = 1
    saved = trial.snapshot()
    clone = NativeSensoryTrial.from_checkpoint(None, copy.deepcopy(saved))
    assert len(restored_sources) == 1
    exact(saved, clone.snapshot(), "synthetic whole candidate roundtrip")
    np.testing.assert_array_equal(trial.adapter.encode(), clone.adapter.encode())
    exact(trial.adapter.snapshot(), clone.adapter.snapshot(), "synthetic next encode state")
    trial.close()
    clone.close()


def test_failed_trial_install_transfers_cleanup_ownership(monkeypatch):
    base = object.__new__(MonotoneRecruitmentTrial)
    base.closed = False
    base.adapter = FakeMotor()
    closed = []

    def fail_install(owner):
        raise ValueError("synthetic observer construction failure")

    def close(owner):
        closed.append(owner)
        owner.closed = True

    monkeypatch.setattr(NativeSensoryTrial, "_install_native_observer", fail_install)
    monkeypatch.setattr(MonotoneRecruitmentTrial, "close", close)
    with pytest.raises(ValueError, match="synthetic observer construction failure"):
        NativeSensoryTrial(base)
    assert base.closed and len(closed) == 1
