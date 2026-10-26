"""Generated CPU/mock checks for the passive E234 recorder; no CUDA run."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.banc_m3_effector_stage_observer import (
    _RECORD_SOURCE,
    CHANNELS,
    EffectorStageObserverNetwork,
    observation_plan,
    stage_record_sources,
    validate_observation_roster,
)
from tools.banc_m3_neural_current_effector import NeuralCurrentEffectorNetwork


class FakeTensor:
    def __init__(self, shape, dtype, fill=0):
        self.value = np.full(shape, fill, dtype=dtype)
        self.device = SimpleNamespace(type="cuda")

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def data_ptr(self):
        return id(self.value)

    def fill_(self, value):
        self.value.fill(value)

    def zero_(self):
        self.value.fill(0)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def fake_observer(capacity=4, free_count=3):
    net = object.__new__(EffectorStageObserverNetwork)
    plan = observation_plan(capacity, free_count)
    net._observation_plan = plan
    net._observation_values = {
        name: FakeTensor(plan["shapes"][name], plan["dtypes"][name], np.nan)
        for name in CHANNELS
    }
    net._observation_counters = FakeTensor((free_count,), np.int64)
    net._observation_armed = FakeTensor((1,), np.int32)
    net._observation_overflow = FakeTensor((1,), np.int32)
    net._observation_pointer_ids = tuple(
        value.data_ptr() for value in net._observation_values.values()
    ) + (net._observation_counters.data_ptr(),
         net._observation_armed.data_ptr(), net._observation_overflow.data_ptr())
    net._observation_ever_armed = False
    net._observation_active = False
    net._observation_stopped = False
    net._observation_start_tick = None
    net._observation_end_tick = None
    net._observation_stream = None
    net._observation_capture_complete = True
    net._construction_stream = 17
    net._test_stream = 17
    net._current_stream_id = lambda: net._test_stream
    net._assert_hot_path_layout = lambda: None
    net.graph = object()
    net.n = 8
    net.tick = 12
    syncs = []
    net.torch = SimpleNamespace(
        float32=np.float32, float64=np.float64,
        int64=np.int64, int32=np.int32,
        cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            current_stream=lambda: SimpleNamespace(
                synchronize=lambda: syncs.append(net._test_stream))),
    )
    net._test_syncs = syncs
    net._positive = FakeTensor((free_count,), np.float32)
    net._negative = FakeTensor((free_count,), np.float32)
    net.drive = FakeTensor((8,), np.float32)
    net._free_row = FakeTensor((free_count,), np.int32)
    net._free_rate64 = FakeTensor((free_count,), np.float64)
    net._free_fast64 = FakeTensor((free_count, 3), np.float64)
    net._free_slow64 = FakeTensor((free_count, 3), np.float64)
    net._stage_rate64 = FakeTensor((free_count,), np.float64)
    net._stage_fast64 = FakeTensor((free_count, 3), np.float64)
    net._stage_slow64 = FakeTensor((free_count, 3), np.float64)
    net._observation_source_pointers = {
        key: getattr(net, key).data_ptr()
        for key in ("_positive", "_negative", "drive", "_free_row",
                    "_free_rate64", "_free_fast64", "_free_slow64",
                    "_stage_rate64", "_stage_fast64", "_stage_slow64")
    }
    return net


def test_plan_bounds_exact_bytes_and_abi():
    plan = observation_plan(7920, 343)
    assert plan["shapes"]["positive_recurrent"] == (7920, 343)
    assert plan["shapes"]["fast64"] == (7920, 343, 3)
    assert plan["shapes"]["slow64"] == (7920, 343, 3)
    assert plan["planned_bytes"] == 7920 * 343 * 68 + 343 * 8 + 8
    assert plan["planned_bytes"] < 256 * 1024**2
    for capacity, free in ((0, 3), (7921, 3), (4, 0), (4, 344),
                           (True, 3), (4, 3.0)):
        with pytest.raises(ValueError):
            observation_plan(capacity, free)

    stage = tuple(object() for _ in range(27))
    finish = tuple(object() for _ in range(22))
    for name, tensors in (("native_stage", stage), ("native_finish", finish)):
        selected = stage_record_sources(name, tensors)
        assert selected == (tensors[0], tensors[1], tensors[2], tensors[3],
                            tensors[15], tensors[16], tensors[17])
    for name, tensors in (("native_stage", finish), ("native_finish", stage),
                          ("csr_input", stage)):
        with pytest.raises(ValueError, match="layout"):
            stage_record_sources(name, tensors)
    assert "drive[free_ids[j]]" in _RECORD_SOURCE
    assert "fast[3*j+k]" in _RECORD_SOURCE
    assert "armed[0] == 0" in _RECORD_SOURCE
    assert "row >= capacity" in _RECORD_SOURCE
    assert "atomicExch(overflow, 1)" in _RECORD_SOURCE


def test_roster_rejected_before_device_allocation(monkeypatch):
    import torch

    matrix = SimpleNamespace(shape=(8, 8))
    motor = np.array([1, 6, 3, 5], np.int64)
    assert validate_observation_roster(matrix, motor,
                                       np.array([1, 3, 5], np.int64)) == 3
    monkeypatch.setattr(torch, "empty", lambda *a, **kw:
                        pytest.fail("allocated before ownership preflight"))
    for bad_motor, bad_free in (
        (motor, np.array([1, 3, 5], np.float64)),
        (motor, np.array([[1, 3, 5]], np.int64)),
        (motor, np.array([1, 1, 3], np.int64)),
        (motor, np.array([3, 1, 5], np.int64)),
        (motor, np.array([1, 3, 7], np.int64)),
        (np.array([1, 6, 3, 3], np.int64), np.array([1, 3], np.int64)),
        (np.array([1, 6, 3, 8], np.int64), np.array([1, 3], np.int64)),
    ):
        with pytest.raises(ValueError, match="subset"):
            EffectorStageObserverNetwork(matrix, device="cuda",
                cuda_implementation="packed", dt=.00025,
                motor_ids=bad_motor, free_motor_ids=bad_free)
    with pytest.raises(ValueError, match="subset"):
        validate_observation_roster(SimpleNamespace(shape=(8, 7)), motor,
                                    np.array([1, 3, 5], np.int64))


def test_write_only_launch_order_and_original_arguments(monkeypatch):
    net = fake_observer()
    calls = []
    net._observation_kernel = lambda grid, block, args: calls.append(
        ("record", grid, block, args))
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "_launch_native",
                        lambda self, name, tensors, step: calls.append(
                            ("parent", name, tensors, step)))
    stage = [FakeTensor((8,), np.float32) for _ in range(27)]
    stage[0] = FakeTensor((3,), np.float32)
    stage[1] = FakeTensor((3,), np.float32)
    stage[3] = FakeTensor((3,), np.int32)
    stage[15] = FakeTensor((3,), np.float64)
    stage[16] = FakeTensor((3, 3), np.float64)
    stage[17] = FakeTensor((3, 3), np.float64)
    stage = tuple(stage)
    before = tuple(value.data_ptr() for value in stage)
    net._launch_native("native_stage", stage, .000125)
    assert calls[0][0] == "record" and calls[1] == (
        "parent", "native_stage", stage, .000125)
    args = calls[0][3]
    np.testing.assert_array_equal(args[:7], np.array(
        [stage[i].data_ptr() for i in (0, 1, 2, 3, 15, 16, 17)],
        dtype=np.uint64))
    assert tuple(value.data_ptr() for value in stage) == before
    wrong = list(stage)
    wrong[16] = FakeTensor((3, 3), np.float32)
    with pytest.raises(ValueError, match="source layout"):
        net._launch_native("native_stage", tuple(wrong), .000125)
    with pytest.raises(ValueError, match="layout"):
        net._launch_native("native_stage", stage[:-1], .000125)
    assert len(calls) == 2


def test_arm_stop_readback_stream_reset_restore_and_pointer_drift(monkeypatch):
    net = fake_observer()
    with pytest.raises(ValueError, match="Stop"):
        net.copy_counters()
    with pytest.raises(ValueError, match="not been armed"):
        net.disable_observation()
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "_advance",
                        lambda self, *a, **kw: (a, kw))
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "reset",
                        lambda self: None)
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "restore",
                        lambda self, saved: saved)
    net.reset()
    assert net.restore({}) == {}
    net.graph = None  # A direct path may run after construction-time capture.
    net.arm_observation()
    assert net._observation_armed.value[0] == 1
    assert net._test_syncs == [17, 17]
    assert net.observation_static_metadata()["start_tick"] == 12
    assert net.observation_static_metadata()["stream"] == 17
    with pytest.raises(ValueError, match="once"):
        net.arm_observation()
    with pytest.raises(ValueError, match="prohibited"):
        net.reset()
    with pytest.raises(ValueError, match="prohibited"):
        net.restore({})
    with pytest.raises(ValueError, match="Stop"):
        net.copy_values("drive")
    net._test_stream = 18
    with pytest.raises(ValueError, match="stream"):
        net._advance(np.zeros(8), 1)
    with pytest.raises(ValueError, match="stream"):
        net.disable_observation()
    net._test_stream = 17
    net.disable_observation()
    assert net._test_syncs == [17, 17, 17, 17]
    assert net.observation_static_metadata()["end_tick"] == 12
    net.disable_observation()
    assert net.copy_armed() == 0 and net.copy_overflow() == 0
    np.testing.assert_array_equal(net.copy_counters(), np.zeros(3, np.int64))
    assert np.isnan(net.copy_values("fast64")).all()
    with pytest.raises(ValueError, match="Unknown"):
        net.copy_values("missing")
    before = net.copy_values("effector64")
    net._advance(np.zeros(8), 1)
    np.testing.assert_array_equal(net.copy_values("effector64"), before)
    net._observation_values["drive"] = FakeTensor((4, 3), np.float32)
    with pytest.raises(ValueError, match="pointer"):
        net._assert_observation_layout()


def test_post_arm_pointer_drift_blocks_replay_and_readback(monkeypatch):
    net = fake_observer()
    calls = []
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "_advance",
                        lambda self, *a, **kw: calls.append("parent"))
    net.arm_observation()
    old_buffer = net._observation_values["fast64"]
    net._observation_values["fast64"] = FakeTensor((4, 3, 3), np.float64)
    assert old_buffer.data_ptr() != net._observation_values["fast64"].data_ptr()
    with pytest.raises(ValueError, match="pointer"):
        net._advance(np.zeros(8), 1)
    assert calls == []
    with pytest.raises(ValueError, match="pointer"):
        net.disable_observation()
    assert not net._observation_stopped

    net2 = fake_observer()
    net2.arm_observation()
    net2.disable_observation()
    net2._test_stream = 18
    with pytest.raises(ValueError, match="stream"):
        net2.copy_values("drive")
    with pytest.raises(ValueError, match="stream"):
        net2.disable_observation()

    net3 = fake_observer()
    net3.arm_observation()
    net3.drive = FakeTensor((8,), np.float32)
    with pytest.raises(ValueError, match="source pointer"):
        net3._advance(np.zeros(8), 1)


def test_wrong_arm_stream_is_nonmutating_and_stage_pointer_blocks_replay(monkeypatch):
    net = fake_observer()
    net._test_stream = 18
    with pytest.raises(ValueError, match="stream"):
        net.arm_observation()
    assert not net._observation_ever_armed
    assert net._observation_start_tick is None
    assert net._observation_armed.value[0] == 0
    assert net._test_syncs == []
    net._test_stream = 17
    net.arm_observation()
    calls = []
    monkeypatch.setattr(NeuralCurrentEffectorNetwork, "_advance",
                        lambda *a, **kw: calls.append("replay"))
    net._stage_fast64 = FakeTensor((3, 3), np.float64)
    with pytest.raises(ValueError, match="source pointer"):
        net._advance(np.zeros(8), 1)
    assert calls == []
