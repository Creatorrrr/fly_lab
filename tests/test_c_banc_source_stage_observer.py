"""Generated E268 checks; root must reserve and execute CUDA tests separately."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools.banc_m3_source_filtered_synapses import (
    SourceFilteredSynapseNetwork,
    checked_selection,
    source_filtered_cuda_source,
)
from tools.banc_m3_source_stage_observer import (
    MAX_OBSERVATION_BYTES,
    SourceStageObserverNetwork,
    observation_plan,
    observation_status,
    observed_source_cuda_source,
)


@pytest.fixture(autouse=True)
def meter(request, monkeypatch):
    from flylab.c.adaptive_rate import AdaptiveRateNetwork
    from flylab.c.research_rate import ResearchRateNetwork
    counts = {"test": request.node.name, "model_attempts": 0, "completed_models": 0,
        "cuda_warmup_steps": 0, "cuda_capture_steps": 0, "scientific_steps": 0,
        "scientific_neural_s": 0., "model_setup_wall_s": 0., "scientific_wall_s": 0.}
    init, advance = AdaptiveRateNetwork.__init__, ResearchRateNetwork._advance

    def count_init(self, *args, **kwargs):
        counts["model_attempts"] += 1
        began = time.perf_counter()
        try:
            init(self, *args, **kwargs)
        finally:
            counts["model_setup_wall_s"] += time.perf_counter()-began
        counts["completed_models"] += 1
        if self.device == "cuda":
            counts["cuda_warmup_steps"] += 2*self.interval_steps
            counts["cuda_capture_steps"] += self.interval_steps

    def count_advance(self, drive, steps, **kwargs):
        began, before = time.perf_counter(), self.tick
        try:
            return advance(self, drive, steps, **kwargs)
        finally:
            counts["scientific_steps"] += self.tick-before
            counts["scientific_neural_s"] += (self.tick-before)*self.dt
            counts["scientific_wall_s"] += time.perf_counter()-began

    monkeypatch.setattr(AdaptiveRateNetwork, "__init__", count_init)
    monkeypatch.setattr(ResearchRateNetwork, "_advance", count_advance)
    began = time.perf_counter()
    yield counts
    counts["wall_s"] = time.perf_counter()-began
    print("E268_GENERATED_COST " + json.dumps(counts, sort_keys=True))


def parameters(coefficients=None):
    w = csr_matrix(np.array([[0., 0., 0., 0.], [.5, 0., 0., 0.],
                            [0., 0., 0., 0.], [-.25, 0., .2, 0.]], np.float32))
    args = (w, np.full(4, .02, np.float32), np.ones(4, np.float32),
            np.full(4, 7.5, np.float32), np.full(4, 200., np.float32))
    kwargs = {"adaptation_gain": 8., "adaptation_tau_s": .3, "dt": .00025,
        "device": "cuda", "cuda_implementation": "packed", "capture_steps": 2,
        "core_ids": np.array([0], np.int64), "core_gain": 0.,
        "motor_ids": np.array([3, 1], np.int64), "motor_gains": np.array([6., 7.], np.float32),
        "motor_log_tau": np.zeros(2, np.float64), "edge_ids": np.array([0, 1, 2], np.int64),
        "free_motor_ids": np.array([1, 3], np.int64),
        "coefficients": (np.tile(np.array([1., 0., 0.]), (3, 1))
                         if coefficients is None else coefficients)}
    return args, kwargs


def planned(capacity=32):
    return observation_plan(4, np.array([3, 1], np.int64), np.array([0, 2], np.int64),
        np.array([3, 1], np.int64), np.array([0, 2], np.int64), capacity)


def test_cpu_plan_exact_rosters_order_mapping_and_allocation_accounting(meter):
    p = planned()
    assert p["observation_buffer_bytes"] == 8*4+8*(2+2)+8+32*(4*2+24*2)
    assert p["observation_buffer_bytes"] == sum(p["buffer_bytes"].values())
    assert set(p["buffer_bytes"]) == {"motor_map", "source_map", "motor_counts",
        "source_counts", "armed", "overflow", "total_input", "source_rate", "source_mask",
        "source_history"}
    np.testing.assert_array_equal(p["motor_map"], [-1, 1, -1, 0])
    np.testing.assert_array_equal(p["source_map"], [0, -1, 1, -1])
    assert p["motor_map"].dtype == np.int32
    args, kw = parameters()
    selection = checked_selection(args[0], kw["edge_ids"], kw["coefficients"], kw["free_motor_ids"])
    np.testing.assert_array_equal(selection["source_ids"], p["source_ids"])
    assert meter["model_attempts"] == 0


@pytest.mark.parametrize("which", ["motor_order", "motor_dtype", "motor_duplicates",
    "source_order", "source_dtype", "source_missing", "capacity_bool", "capacity_zero",
    "capacity_over", "capacity_float", "memory_over"])
def test_cpu_plan_rejects_wrong_rosters_and_boundaries(which):
    motor, source, capacity, n = np.array([3, 1], np.int64), np.array([0, 2], np.int64), 32, 4
    native_motor, native_source = motor.copy(), source.copy()
    if which == "motor_order":
        motor = motor[::-1]
    elif which == "motor_dtype":
        motor = motor.astype(np.int32)
    elif which == "motor_duplicates":
        motor[:] = 1
    elif which == "source_order":
        source = source[::-1]
    elif which == "source_dtype":
        source = source.astype(np.int32)
    elif which == "source_missing":
        source = source[:1]
    elif which == "capacity_bool":
        capacity = True
    elif which == "capacity_zero":
        capacity = 0
    elif which == "capacity_over":
        capacity = 160001
    elif which == "capacity_float":
        capacity = 32.
    else:
        n, capacity = 4000, 160000
        motor = native_motor = np.arange(4000, dtype=np.int64)
        source = native_source = np.arange(4000, dtype=np.int64)
        assert capacity*(4*len(motor)+24*len(source)) > MAX_OBSERVATION_BYTES
    with pytest.raises(ValueError):
        observation_plan(n, motor, source, native_motor, native_source, capacity)


def test_cpu_status_complete_and_error_common_prefix():
    done = observation_status(np.array([12, 12], np.int64), np.array([12, 12], np.int64),
                              np.array([0], np.int32), 32, 12)
    assert done["complete"] and done["valid_prefix_stages"] == 12
    partial = observation_status(np.array([8, 7], np.int64), np.array([6, 8], np.int64),
                                 np.array([1], np.int32), 5, 8)
    assert not partial["complete"] and partial["valid_prefix_stages"] == 5
    np.testing.assert_array_equal(partial["motor_valid_counts"], [5, 5])
    failed = observation_status(np.array([4, 3], np.int64), np.array([2, 4], np.int64),
                                np.array([0], np.int32), 32, 0)
    assert not failed["complete"] and failed["valid_prefix_stages"] == 2
    with pytest.raises(ValueError):
        observation_status(np.array([-1], np.int64), np.array([1], np.int64),
                           np.array([0], np.int32), 32, 4)


def test_cpu_shader_changes_only_signature_and_post_drive_stores():
    for tanh in (True, False):
        base = source_filtered_cuda_source(tanh)
        observed = observed_source_cuda_source(tanh, 2, 2, 32)
        assert observed.count("value = add_exact(drive[row], value);") == 1
        assert "source_rate[offset] = rate[row];" in observed
        assert "source_mask[offset] = mask[row];" in observed
        assert "source_history[2*offset] = history[2*column];" in observed
        assert "source_values" not in observed
        assert observed.split('extern "C" __global__ void stage')[1] == base.split(
            'extern "C" __global__ void stage')[1]
        # The original sparse arithmetic is byte-identical before the lane-zero block.
        assert observed.split("int lane = threadIdx.x & 31;")[1].split(
            "value = add_exact(drive[row], value);")[0] == base.split(
            "int lane = threadIdx.x & 31;")[1].split("value = add_exact(drive[row], value);")[0]


def test_cpu_constructor_forbids_cpu_fallback_before_model(meter):
    args, kwargs = parameters()
    kwargs["device"] = "cpu"
    with pytest.raises(ValueError, match="no CPU fallback"):
        SourceStageObserverNetwork(*args, **kwargs, observation_ids=kwargs["motor_ids"],
            source_ids=np.array([0, 2], np.int64), capacity_stages=32)
    assert meter["model_attempts"] == 0


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Actual generated CUDA execution requires root reservation")


def model(observed, capacity=32, coefficients=None):
    args, kw = parameters(coefficients)
    if observed:
        return SourceStageObserverNetwork(*args, **kw,
            observation_ids=kw["motor_ids"].copy(), source_ids=np.array([0, 2], np.int64),
            capacity_stages=capacity)
    return SourceFilteredSynapseNetwork(*args, **kw)


def initialize(network):
    saved = network.snapshot()
    saved["rate"] = np.array([2., 1., 3., 4.], np.float32)
    saved["adaptation"] = np.array([.2, .1, .4, .3], np.float32)
    saved["source_history"] = np.array([[.5, 1.], [1.5, 2.5]], np.float64)
    saved["output_mask"] = np.array([0., 1., 1., 1.], np.float32)
    saved["drive"] = np.array([0., 2., 0., 4.], np.float32)
    network.restore(saved)
    return saved


def exact(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    else:
        assert a == b


@pytest.mark.parametrize("nonneutral", [False, True])
def test_cuda_observer_native_parity_raw_mask_and_actual_four_stage_history(nonneutral):
    require_cuda()
    c = np.array([[.25, .5, .25], [0., 1., 1.], [.5, .25, .25]]) if nonneutral else None
    base, observed = model(False, coefficients=c), model(True, coefficients=c)
    initial = initialize(base)
    observed.restore(initial)
    assert base.identity == observed.identity
    exact(base.snapshot(), observed.snapshot())
    assert all(not value.is_inference() for value in observed._observer_buffers.values())
    assert np.all(observed._observer_buffers["motor_counts"].cpu().numpy() == 0)
    observed.reset()
    observed.restore(initial)
    total0 = np.float32(initial["drive"] + base.source_current_oracle())
    observed.arm_observation()
    for candidate in (base, observed):
        candidate.advance(initial["drive"], steps=3)
    exact(base.snapshot(), observed.snapshot())
    observed.disable_observation()
    exported = observed.export_observation()
    assert exported["complete"] and exported["valid_prefix_stages"] == 12
    np.testing.assert_array_equal(exported["motor_counts"], [12, 12])
    np.testing.assert_array_equal(exported["source_counts"], [12, 12])
    exact(exported["total_input"][0], total0[[3, 1]])
    exact(exported["source_rate"][0], initial["rate"][[0, 2]])
    exact(exported["source_history"][0], initial["source_history"])
    np.testing.assert_array_equal(exported["source_mask"][:12], np.tile([0., 1.], (12, 1)))
    assert (exported["source_rate"][:12, 0] > 0).all()
    assert not np.array_equal(exported["source_rate"][1], exported["source_rate"][2])
    # Source cells have zero target; reconstruct same-stage histories directly
    # from the observed raw stage rates and initial f64 histories, not endpoints.
    h0 = initial["source_history"]
    derivatives = []
    for stage in range(4):
        expected_h = h0 if stage == 0 else h0+(.00025*(.5 if stage < 3 else 1))*derivatives[-1]
        exact(exported["source_history"][stage], expected_h)
        derivatives.append((exported["source_rate"][stage].astype(np.float64)[:, None]
                            - expected_h)/np.array([.005, .04]))
    assert all(np.isnan(exported[key][12:]).all() for key in (
        "total_input", "source_rate", "source_mask", "source_history"))
    assert sum(exported["buffer_bytes"].values()) == observed.observation_buffer_bytes
    interrupted_scope = observed.export_observation(expected_stages=16)
    assert not interrupted_scope["complete"] and interrupted_scope["logical_stages"] == 12


def test_cuda_overflow_partial_stream_pointer_and_single_use_guards():
    require_cuda()
    import torch
    observed = model(True, capacity=5)
    saved = initialize(observed)
    other = torch.cuda.Stream()
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match="constructor CUDA stream"):
        observed.arm_observation()
    observed.arm_observation()
    for call in (observed.arm_observation, observed.reset, lambda: observed.restore(saved)):
        with pytest.raises(RuntimeError):
            call()
    with pytest.raises(RuntimeError, match="Stop"):
        observed.export_observation()
    observed.advance(saved["drive"], steps=2)
    observed.disable_observation()
    observed.disable_observation()
    partial = observed.export_observation()
    assert not partial["complete"] and partial["valid_prefix_stages"] == 5
    np.testing.assert_array_equal(partial["motor_counts"], [8, 8])
    np.testing.assert_array_equal(partial["source_counts"], [8, 8])
    assert partial["overflow"].tolist() == [1]
    assert partial["armed"].tolist() == [0]
    assert np.isfinite(partial["source_history"]).all()
    observed.advance(saved["drive"], steps=2)
    exact(partial, observed.export_observation())
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match="constructor CUDA stream"):
        observed.advance(saved["drive"], steps=1)
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match="constructor CUDA stream"):
        observed.set_muted([0])
    original = observed._observer_buffers["source_rate"]
    observed._observer_buffers["source_rate"] = original.clone()
    with pytest.raises(RuntimeError, match="pointer"):
        observed.advance(saved["drive"], steps=1)
    observed._observer_buffers["source_rate"] = original
    exact(partial, observed.export_observation())
