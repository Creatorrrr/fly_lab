"""Small actual-CUDA E126 stage taps; original BANC/body never run here."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_stage_dataset import StageInputNetwork
from tools.verify_c_banc_m3_stage_dataset import load_fixture, run, save_stage_arrays


@pytest.fixture(autouse=True)
def one_thread():
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def meter(request):
    result = {'test': request.node.name, 'cuda_models': 0,
        'cuda_warmup_ticks': 0, 'cuda_capture_ticks': 0,
        'synthetic_native_ticks': 0, 'synthetic_native_s': 0.,
        'body_steps': 0, 'reference_calls': 0,
        'reference_stage_evaluations': 0, 'wall_s': 0.}
    began = time.perf_counter()
    yield result
    result['wall_s'] = time.perf_counter()-began
    print('E126_SYNTHETIC_COUNT', result)


def model(meter, *, observed, capacity=32, weights=None, motor_ids=(1, 3),
          source_ids=(0, 1)):
    import torch

    assert torch.cuda.is_available(), 'Actual CUDA required; no skip/fallback'
    if weights is None:
        weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32),
            np.ones(4, np.float32), np.full(4, 7.5, np.float32),
            np.full(4, 200., np.float32))
    kw = {'adaptation_gain': 8., 'adaptation_tau_s': .3,
          'dt': .00025, 'device': 'cuda', 'cuda_implementation': 'packed',
          'capture_steps': 2, 'core_ids': np.array([1, 3]),
          'core_gain': 0.}
    if observed:
        result = StageInputNetwork(*args, motor_ids=np.array(motor_ids),
            source_ids=np.array(source_ids), capacity_stages=capacity, **kw)
    else:
        result = CoreAdaptationNetwork(*args, **kw)
    meter['cuda_models'] += 1
    meter['cuda_warmup_ticks'] += 4
    meter['cuda_capture_ticks'] += 2
    return result


def advance(network, drive, ticks, meter):
    before = int(network.tick)
    try:
        network.advance(np.asarray(drive, np.float32), steps=ticks)
    finally:
        delta = int(network.tick)-before
        meter['synthetic_native_ticks'] += delta
        meter['synthetic_native_s'] += delta*.00025


def test_dual_stage_taps_preserve_core_bits_and_middle_stage_inputs(meter):
    weights = csr_matrix(np.asarray([
        [0., 0., 0., 0.], [.5, 0., 0., 0.],
        [0., 0., 0., 0.], [0., 0., 0., 0.]], np.float32))
    base = model(meter, observed=False, weights=weights)
    observed = model(meter, observed=True, weights=weights)
    state = base.snapshot()
    state['rate'] = np.array([2., 1., 3., 4.], np.float32)
    state['adaptation'] = np.array([.2, .1, .4, .3], np.float32)
    state['output_mask'][1] = 0.
    base.restore(state)
    observed.restore(state)
    np.testing.assert_equal(base.snapshot(), observed.snapshot())
    assert np.all(observed._observed_counters.cpu().numpy() == 0)
    assert np.all(observed._source_counters.cpu().numpy() == 0)
    observed.arm_observation()
    drive = [1., 2., 3., 4.]
    advance(base, drive, 3, meter)
    advance(observed, drive, 3, meter)
    np.testing.assert_equal(base.snapshot(), observed.snapshot())
    observed.disable_observation()
    current = observed.copy_current_values()
    source = observed.copy_source_values()
    np.testing.assert_array_equal(observed.copy_current_counters(), [12, 12])
    np.testing.assert_array_equal(observed.copy_source_counters(), [12, 12])
    assert observed.copy_current_overflow().tolist() == [0]
    assert observed.copy_source_overflow().tolist() == [0]
    assert observed.copy_armed().tolist() == [0]
    assert current.shape == (32, 2) and source.shape == (32, 2)
    assert np.isnan(current[12:]).all() and np.isnan(source[12:]).all()
    f = np.float32
    r1 = f(2.)
    h = f(.00025*.5)
    full = f(.00025)
    tau = f(.02)
    k1 = f(-r1/tau)
    r2 = f(r1+f(h*k1))
    k2 = f(-r2/tau)
    r3 = f(r1+f(h*k2))
    k3 = f(-r3/tau)
    r4 = f(r1+f(full*k3))
    expected_source = np.asarray([r1, r2, r3, r4], np.float32)
    expected_current = np.asarray([f(f(2.)+f(f(.5)*r))
                                   for r in expected_source], np.float32)
    meter['reference_calls'] += 1
    meter['reference_stage_evaluations'] += 4
    assert expected_source[1] != expected_source[2]
    np.testing.assert_array_equal(source[:4, 0], expected_source)
    np.testing.assert_array_equal(source[:4, 1], np.zeros(4, np.float32))
    np.testing.assert_array_equal(current[:4, 0], expected_current)
    np.testing.assert_array_equal(current[:4, 1], np.full(4, 4., np.float32))
    assert observed.observation_static_metadata()['end_tick'] == 3
    assert meter['synthetic_native_ticks'] == 6


def test_overflow_lifecycle_stream_and_restore_rejection(meter):
    import torch

    observed = model(meter, observed=True, capacity=5)
    saved = observed.snapshot()
    other = torch.cuda.Stream()
    with torch.cuda.stream(other), pytest.raises(
            RuntimeError, match='constructor CUDA stream'):
        observed.arm_observation()
    observed.arm_observation()
    advance(observed, [0., 2., 0., 4.], 2, meter)
    armed = observed.snapshot()
    with pytest.raises(RuntimeError, match='Cannot restore'):
        observed.restore(saved)
    np.testing.assert_equal(armed, observed.snapshot())
    observed.disable_observation()
    before = observed.copy_current_values()
    before_source = observed.copy_source_values()
    advance(observed, [0., 0., 0., 0.], 2, meter)
    np.testing.assert_array_equal(before, observed.copy_current_values())
    np.testing.assert_array_equal(before_source, observed.copy_source_values())
    assert observed.observation_static_metadata()['end_tick'] == 2
    np.testing.assert_array_equal(observed.copy_current_counters(), [8, 8])
    np.testing.assert_array_equal(observed.copy_source_counters(), [8, 8])
    assert observed.copy_current_overflow().tolist() == [1]
    assert observed.copy_source_overflow().tolist() == [1]
    with pytest.raises(RuntimeError, match='Cannot restore'):
        observed.restore(saved)
    assert meter['synthetic_native_ticks'] == 4


def test_capacity_and_roster_validation_precedes_cuda_allocations(meter):
    weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32),
            np.ones(4, np.float32), np.full(4, 7.5, np.float32),
            np.full(4, 200., np.float32))
    kw = {'adaptation_gain': 8., 'adaptation_tau_s': .3,
          'dt': .00025, 'device': 'cuda', 'cuda_implementation': 'packed',
          'capture_steps': 2, 'core_ids': np.array([1, 3]), 'core_gain': 0.}
    for motors, sources, capacity in (
        ([1, 1], [0], 5), ([1], [0, 0], 5), ([4], [0], 5),
        ([1], [0], 160001), ([1], [0], 0)):
        with pytest.raises(ValueError):
            StageInputNetwork(*args, motor_ids=motors, source_ids=sources,
                              capacity_stages=capacity, **kw)
    giant = SimpleNamespace(shape=(100_000_000, 100_000_000))
    with pytest.raises(ValueError, match='512 MiB'):
        StageInputNetwork(giant, *args[1:], motor_ids=[1], source_ids=[2],
                          capacity_stages=1, **kw)
    with pytest.raises(ValueError, match='packed CUDA'):
        StageInputNetwork(*args, motor_ids=[1], source_ids=[1],
            capacity_stages=5, **{**kw, 'device': 'cpu',
                                   'cuda_implementation': 'reference'})
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_raw_hostcopy_and_file_failures_leave_sibling_and_metadata(tmp_path,
                                                                   monkeypatch,
                                                                   meter):
    class Observer:
        def disable_observation(self):
            pass

        def observation_static_metadata(self):
            return {'start_tick': 0, 'end_tick': 2, 'capacity_stages': 8,
                    'motor_ids': np.array([1, 3]),
                    'source_ids': np.array([0, 1])}

        def copy_current_counters(self):
            return np.array([8, 8], np.int64)

        def copy_source_counters(self):
            return np.array([8, 8], np.int64)

        def copy_current_overflow(self):
            return np.array([0], np.int32)

        def copy_source_overflow(self):
            return np.array([0], np.int32)

        def copy_armed(self):
            return np.array([0], np.int32)

        def copy_current_values(self):
            return np.full((8, 2), 3., np.float32)

        def copy_source_values(self):
            return np.full((8, 2), 2., np.float32)

    observer = Observer()
    first = tmp_path/'file-failure'
    first.mkdir()
    original_save = np.save

    def fail_current(path, *args, **kwargs):
        if Path(path).name == 'motor-total-current.npy':
            raise OSError('injected current save failure')
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(np, 'save', fail_current)
    errors = []
    metadata, results = save_stage_arrays(first, observer, errors)
    assert results == {'current': False, 'source': True}
    assert len(errors) == 1 and errors[0]['stage'] == 'motor-total-current.npy'
    assert metadata['end_tick'] == 2
    assert (first/'stage-metadata.json').exists()
    np.testing.assert_array_equal(np.load(first/'source-masked-rate.npy'),
                                  np.full((8, 2), 2., np.float32))
    np.testing.assert_array_equal(np.load(first/'current-stage-rescue.npy'),
                                  np.full((8, 2), 3., np.float32))

    second = tmp_path/'hostcopy-failure'
    second.mkdir()
    monkeypatch.setattr(np, 'save', original_save)

    def broken_source():
        raise RuntimeError('injected source hostcopy failure')

    observer.copy_source_values = broken_source
    errors = []
    metadata, results = save_stage_arrays(second, observer, errors)
    assert results == {'current': True, 'source': False}
    assert len(errors) == 1 and errors[0]['stage'] == 'source-hostcopy'
    assert metadata['source_counter_requested'].tolist() == [8, 8]
    assert (second/'stage-metadata.json').exists()
    np.testing.assert_array_equal(np.load(second/'motor-total-current.npy'),
                                  np.full((8, 2), 3., np.float32))
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_synthetic_fixture_ownership_and_invalid_mapping(tmp_path, meter):
    motor = np.arange(391, dtype=np.int64)
    source = np.r_[np.int64(83877), np.arange(392, 460, dtype=np.int64)]
    motor_col = np.arange(2444, dtype=np.int64) % 391
    source_col = np.arange(2444, dtype=np.int64) % 69
    fixture = {
        'context_names': np.asarray(['E110', 'E113', 'E121']),
        'motor_ids': motor, 'source_ids': source,
        'edge_ids': np.arange(2444, dtype=np.int64),
        'edge_pre': source[source_col], 'edge_post': motor[motor_col],
        'edge_motor_column': motor_col, 'edge_source_column': source_col,
        'calibration_positions': np.arange(2444, dtype=np.int64),
        'edge_values_by_context': np.zeros((3, 2444), np.float32),
        'fit_mask': np.arange(2444) < 1572,
        'fit_edge_ids': np.arange(1572, dtype=np.int64),
        'initial_motor_rate': np.zeros((3, 391), np.float32),
        'initial_motor_adaptation': np.zeros((3, 391), np.float32),
        'initial_source_masked_rate': np.zeros((3, 69), np.float32),
        'pool': np.zeros((96, 391), np.float64),
        'moment': np.zeros((42, 96), np.float64),
        'rest': np.zeros(42, np.float64),
        'ltm': np.arange(6, dtype=np.int64),
    }
    path = tmp_path/'synthetic-fixture.npz'
    np.savez(path, **fixture)
    loaded = load_fixture(path)
    loaded['motor_ids'][0] = 12345
    assert load_fixture(path)['motor_ids'][0] == 0
    fixture['edge_source_column'][0] = 69
    np.savez(path, **fixture)
    with pytest.raises(ValueError, match='outside original edge_source_column'):
        load_fixture(path)
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_missing_freeze_saves_failure_ledger_without_model(tmp_path, meter):
    out = tmp_path/'failed-original-contract'
    with pytest.raises(FileNotFoundError):
        run(tmp_path/'missing-freeze.json', tmp_path/'missing-review.json',
            'E110', out)
    assert (out/'advance-ledger.json').exists()
    assert (out/'setup-and-cost.json').exists()
    assert (out/'execution-errors.json').exists()
    assert not (out/'result.json').exists()
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0
