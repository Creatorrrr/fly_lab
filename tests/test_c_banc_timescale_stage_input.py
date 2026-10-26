"""Small synthetic CUDA E150 timescale-current checks; no original BANC."""

import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source
from tools.banc_m3_timescale_stage_input import (
    TimescaleStageInputNetwork,
    _observed_timescale_source,
)


@pytest.fixture(autouse=True)
def one_thread():
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def meter(request):
    value = {"test": request.node.name, "models_constructed": 0,
             "warmup_native_ticks": 0, "capture_native_ticks": 0,
             "main_native_ticks": 0, "body_steps": 0,
             "setup_wall_s": 0., "main_wall_s": 0.}
    yield value
    print("E201_SYNTHETIC_COUNT", value)


def model(meter, observed=False, capacity=32, weights=None):
    import torch

    assert torch.cuda.is_available(), "Actual small CUDA test required"
    if weights is None:
        weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32),
            np.ones(4, np.float32), np.full(4, 7.5, np.float32),
            np.full(4, 200., np.float32))
    kw = {"adaptation_gain": 8., "adaptation_tau_s": .3,
          "dt": .00025, "device": "cuda", "cuda_implementation": "packed",
          "capture_steps": 2, "core_ids": np.array([0], np.int64),
          "core_gain": 0., "motor_ids": np.array([1, 3], np.int64),
          "motor_gains": np.array([6., 7.], np.float32),
          "motor_log_tau": np.zeros(2, np.float64)}
    began = time.perf_counter()
    if observed:
        result = TimescaleStageInputNetwork(*args,
            observation_ids=np.array([1, 3], np.int64),
            capacity_stages=capacity, **kw)
    else:
        result = MotorTimescaleNetwork(*args, **kw)
    meter["setup_wall_s"] += time.perf_counter() - began
    meter["models_constructed"] += 1
    meter["warmup_native_ticks"] += 4
    meter["capture_native_ticks"] += 2
    return result


def advance(network, drive, steps, meter):
    began = time.perf_counter()
    before = network.tick
    network.advance(np.asarray(drive, np.float32), steps=steps)
    meter["main_wall_s"] += time.perf_counter() - began
    meter["main_native_ticks"] += network.tick - before


def test_source_keeps_original_stage_and_finish_bodies():
    for tanh in (False, True):
        base = _timescale_source(tanh)
        observed = _observed_timescale_source(tanh, 2, 8)
        assert observed.count("value = add_exact(drive[row], value);") == 1
        assert observed.count("observed[position * 2ull + column] = value;") == 1
        assert observed.split('extern "C" __global__ void stage')[1] == base.split(
            'extern "C" __global__ void stage')[1]


def test_timescale_stage_tap_is_bit_exact_and_records_four_distinct_inputs(meter):
    weights = csr_matrix(np.asarray([
        [0., 0., 0., 0.], [.5, 0., 0., 0.],
        [0., 0., 0., 0.], [0., 0., 0., 0.]], np.float32))
    base = model(meter, weights=weights)
    observed = model(meter, observed=True, weights=weights)
    saved = base.snapshot()
    saved["rate"] = np.array([2., 1., 3., 4.], np.float32)
    saved["adaptation"] = np.array([.2, .1, .4, .3], np.float32)
    base.restore(saved)
    observed.restore(saved)
    assert observed.identity == base.identity
    np.testing.assert_equal(base.snapshot(), observed.snapshot())
    observed.arm_observation()
    advance(base, [1., 2., 3., 4.], 3, meter)
    advance(observed, [1., 2., 3., 4.], 3, meter)
    np.testing.assert_equal(base.snapshot(), observed.snapshot())
    observed.disable_observation()
    current = observed.copy_values()
    np.testing.assert_array_equal(observed.copy_counters(), [12, 12])
    assert observed.copy_overflow().tolist() == [0]
    assert observed.copy_armed().tolist() == [0]
    assert observed.observation_static_metadata()["end_tick"] == 3
    assert current.shape == (32, 2)
    assert np.isnan(current[12:]).all()
    f = np.float32
    r1 = f(2.)
    h = f(.00025 * .5)
    full = f(.00025)
    tau = f(.02)
    k1 = f(-r1 / tau)
    r2 = f(r1 + f(h * k1))
    k2 = f(-r2 / tau)
    r3 = f(r1 + f(h * k2))
    k3 = f(-r3 / tau)
    r4 = f(r1 + f(full * k3))
    expected = np.asarray([f(f(2.) + f(f(.5) * r))
                           for r in (r1, r2, r3, r4)], np.float32)
    assert expected[1] != expected[2]
    np.testing.assert_array_equal(current[:4, 0], expected)
    np.testing.assert_array_equal(current[:4, 1], np.full(4, 4., np.float32))
    assert meter["main_native_ticks"] == 6


def test_single_arm_overflow_stream_restore_and_stop(meter):
    import torch

    observed = model(meter, observed=True, capacity=5)
    saved = observed.snapshot()
    other = torch.cuda.Stream()
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match="constructor CUDA stream"):
        observed.arm_observation()
    observed.arm_observation()
    with pytest.raises(RuntimeError, match="Arm once"):
        observed.arm_observation()
    advance(observed, [0., 2., 0., 4.], 2, meter)
    armed_state = observed.snapshot()
    with pytest.raises(RuntimeError, match="Cannot restore"):
        observed.restore(saved)
    np.testing.assert_equal(armed_state, observed.snapshot())
    observed.disable_observation()
    first = observed.copy_values()
    advance(observed, [0., 0., 0., 0.], 2, meter)
    np.testing.assert_array_equal(first, observed.copy_values())
    np.testing.assert_array_equal(observed.copy_counters(), [8, 8])
    assert observed.copy_overflow().tolist() == [1]
    assert observed.copy_armed().tolist() == [0]
    assert observed.observation_static_metadata()["end_tick"] == 2


def test_invalid_roster_and_capacity_rejected_before_cuda_model(meter):
    weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32),
            np.ones(4, np.float32), np.full(4, 7.5, np.float32),
            np.full(4, 200., np.float32))
    kw = {"adaptation_gain": 8., "adaptation_tau_s": .3,
          "dt": .00025, "device": "cuda", "cuda_implementation": "packed",
          "capture_steps": 2, "core_ids": np.array([0], np.int64),
          "core_gain": 0., "motor_ids": np.array([1, 3], np.int64),
          "motor_gains": np.array([6., 7.], np.float32),
          "motor_log_tau": np.zeros(2, np.float64)}
    for ids, capacity in (([1, 1], 8), ([4], 8), ([1], 160001), ([1], 0)):
        with pytest.raises(ValueError):
            TimescaleStageInputNetwork(*args,
                observation_ids=np.asarray(ids, np.int64),
                capacity_stages=capacity, **kw)
    assert meter["models_constructed"] == 0
