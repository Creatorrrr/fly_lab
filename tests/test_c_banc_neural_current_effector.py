"""Pure generated E223 routing/checkpoint checks; no CUDA model is built."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.banc_m3_native_grouped_response import NativeGroupedResponseNetwork
from tools.banc_m3_neural_current_effector import (
    NeuralCurrentEffectorNetwork,
    ordered_effector_readout,
    route_native_destinations,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.array(value, copy=True)
        self.device = SimpleNamespace(type="cuda")

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def data_ptr(self):
        return id(self.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def copy_(self, value):
        self.value[...] = np.asarray(value)

    def zero_(self):
        self.value.fill(0)


def fake_network(monkeypatch):
    net = object.__new__(NeuralCurrentEffectorNetwork)
    net.n = 8
    net.device = "cuda"
    net.identity = "e223"
    net.source_timescale_identity = "e150-timescale"
    net.source_grouped_identity = "e221"
    net._native_fingerprint = "parameters"
    net.native_motor_ids = np.array([1, 6, 3, 5], np.int64)
    net.free_motor_ids = np.array([1, 3, 5], np.int64)
    net._motor_count = 4
    net.fixed_motor_ids = np.array([6], np.int64)
    net.membership = np.array([0, 0, 1], np.int64)
    net.scale = np.ones((3, 3), np.float64)
    net.centers = np.zeros((3, 64, 9), np.float64)
    net.sigma2 = np.ones(3, np.float64)
    net.beta = np.zeros((2, 64), np.float64)
    net.free_tau = np.full(3, .02, np.float64)
    net.free_cap = np.full(3, 10., np.float64)
    net.inverse_slow = np.full(3, 5., np.float64)
    net._free_count = 3
    net._free_rate64 = FakeTensor(np.array([.75, 1.25, 2.5], np.float64))
    net._free_fast64 = FakeTensor(np.zeros((3, 3), np.float64))
    net._free_slow64 = FakeTensor(np.zeros((3, 3), np.float64))
    rate = np.zeros(8, np.float32)
    rate[[1, 6, 3, 5]] = [1., 4., 2., 3.]
    net.rate = FakeTensor(rate)
    net.torch = SimpleNamespace(as_tensor=lambda value, device: np.asarray(value),
                                float32=np.float32, float64=np.float64)
    net._native_pointers = {"_free_rate64": net._free_rate64.data_ptr()}
    net.readout = lambda indices: net.rate.value[np.asarray(indices)].copy()
    net._assert_native_parameters = lambda: None
    net._assert_effector_routing = lambda: None
    baseline = {
        "schema": "flylab.research-rate.v1", "model_hash": "e223",
        "tick": 23, "dt": .00025, "device": "cuda",
        "rate": rate.copy(), "drive": np.zeros(8, np.float32),
        "output_mask": np.ones(8, np.float32),
        "adaptation": np.zeros(8, np.float32),
        "motor_timescale": {"ids": net.native_motor_ids.copy(),
                            "schema": "timescale"},
    }
    calls = []
    monkeypatch.setattr(MotorTimescaleNetwork, "snapshot",
                        lambda self: copy.deepcopy(baseline))
    monkeypatch.setattr(MotorTimescaleNetwork, "restore",
                        lambda self, saved: (calls.append("base"),
                            self.rate.copy_(saved["rate"])))
    return net, baseline, calls


def test_destinations_only_and_frozen_argument_layout(monkeypatch):
    scratch_full, scratch_masked = object(), object()
    stage = tuple(object() for _ in range(27))
    routed = route_native_destinations(
        "native_stage", stage, scratch_full, scratch_masked)
    assert routed[-3] is scratch_full and routed[-1] is scratch_masked
    assert routed[-2] is stage[-2]
    assert all(routed[i] is stage[i] for i in range(24))
    assert stage[-3] is not scratch_full
    finish = tuple(object() for _ in range(22))
    assert route_native_destinations("native_finish", finish,
        scratch_full, scratch_masked)[:-1] == finish[:-1]
    with pytest.raises(ValueError, match="layout"):
        route_native_destinations("native_stage", finish,
                                  scratch_full, scratch_masked)

    seen = []
    monkeypatch.setattr(NativeGroupedResponseNetwork, "_launch_native",
                        lambda self, name, tensors, step: seen.append((name, tensors, step)))
    net = object.__new__(NeuralCurrentEffectorNetwork)
    net._effector_full_scratch = scratch_full
    net._effector_masked_scratch = scratch_masked
    net._assert_effector_routing = lambda: None
    net._launch_native("native_stage", stage, .000125)
    assert seen[0] == ("native_stage", routed, .000125)


def test_ordered_effector_is_distinct_from_neuronal_rate():
    rate = np.array([0., 1., 0., 2., 0., 3., 4., 0.], np.float32)
    original = rate.copy()
    result = ordered_effector_readout(rate,
        np.array([1, 6, 3, 5], np.int64), np.array([1, 3, 5], np.int64),
        np.array([.75, 1.25, 2.5], np.float64))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([.75, 4., 1.25, 2.5], np.float32))
    np.testing.assert_array_equal(rate, original)
    with pytest.raises(ValueError, match="effector"):
        ordered_effector_readout(rate, np.array([1, 6, 3, 5], np.int64),
            np.array([1, 3, 5], np.int64), np.array([np.nan, 1., 2.]))


def test_hot_path_never_reads_full_rate_or_parameter_arrays(monkeypatch):
    net, _, _ = fake_network(monkeypatch)
    selected_calls = []

    def selected_readout(indices):
        selected_calls.append(np.asarray(indices).copy())
        return net.rate.value[np.asarray(indices)].copy()

    net.readout = selected_readout
    net.rate.detach = lambda: pytest.fail("full neuronal rate host copy")
    net._assert_native_parameters = lambda: pytest.fail("full GPU parameter readback")
    net._assert_effector_routing = lambda: None
    before = net.rate.value.copy()
    output = net.effector_readout()
    assert len(selected_calls) == 1
    np.testing.assert_array_equal(selected_calls[0], net.native_motor_ids)
    np.testing.assert_array_equal(output,
                                  np.array([.75, 4., 1.25, 2.5], np.float32))
    np.testing.assert_array_equal(net.rate.value, before)
    net._free_rate64.value = net._free_rate64.value.copy()
    with pytest.raises(ValueError, match="layout"):
        net.effector_readout()


def test_fixed_scratch_pointer_and_reset(monkeypatch):
    net = object.__new__(NeuralCurrentEffectorNetwork)
    net.n = 8
    net.torch = SimpleNamespace(float32=np.float32)
    net.rate = FakeTensor(np.zeros(8, np.float32))
    net._fused_buffers = [FakeTensor(np.zeros(8, np.float32))]
    net._effector_full_scratch = FakeTensor(np.ones(8, np.float32))
    net._effector_masked_scratch = FakeTensor(np.ones(8, np.float32))
    net._scratch_pointers = (
        net._effector_full_scratch.data_ptr(),
        net._effector_masked_scratch.data_ptr())
    net._assert_effector_routing()
    monkeypatch.setattr(NativeGroupedResponseNetwork, "reset", lambda self: None)
    net.reset()
    assert not np.any(net._effector_full_scratch.value)
    assert not np.any(net._effector_masked_scratch.value)
    net._effector_masked_scratch = FakeTensor(np.zeros(8, np.float32))
    with pytest.raises(ValueError, match="pointer"):
        net._assert_effector_routing()


def test_independent_snapshot_restore_and_all_invalid_preflight(monkeypatch):
    net, baseline, calls = fake_network(monkeypatch)
    saved = net.snapshot()
    patch = saved["neural_current_effector"]
    assert "native_grouped_response" not in saved
    assert patch["source_timescale_identity"] == "e150-timescale"
    np.testing.assert_array_equal(saved["rate"], baseline["rate"])
    assert saved["rate"][1] != patch["effector64"][0]
    np.testing.assert_array_equal(net.effector_readout(),
                                  np.array([.75, 4., 1.25, 2.5], np.float32))

    changed = copy.deepcopy(saved)
    changed["rate"][1] = np.float32(.5)
    changed["neural_current_effector"]["effector64"][0] = .875
    net.restore(changed)
    assert calls == ["base"]
    assert net.rate.value[1] == np.float32(.5)
    assert net._free_rate64.value[0] == .875

    for mutation in (
        lambda value: value["neural_current_effector"].update(schema="old"),
        lambda value: value["neural_current_effector"]["beta"].__setitem__((0, 0), .5),
        lambda value: value["neural_current_effector"]["fast64"].__setitem__((0, 0), np.nan),
        lambda value: value["motor_timescale"].update(schema="wrong"),
        lambda value: value["rate"].__setitem__(0, np.nan),
        lambda value: value.update(native_grouped_response={}),
        lambda value: value["neural_current_effector"].update(
            effector64=np.zeros(3, np.float32)),
    ):
        bad = copy.deepcopy(saved)
        mutation(bad)
        before = (net.rate.value.copy(), net._free_rate64.value.copy())
        with pytest.raises(ValueError):
            net.restore(bad)
        assert calls == ["base"]
        np.testing.assert_array_equal(net.rate.value, before[0])
        np.testing.assert_array_equal(net._free_rate64.value, before[1])
