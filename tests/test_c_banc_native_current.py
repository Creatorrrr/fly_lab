"""Tiny CUDA CSR observation checks; no original graph/body/checkpoint is run."""

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import verify_c_banc_m3_native_current as runner
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_native_current import ObservedCoreAdaptationNetwork
from tools.verify_c_banc_m3_native_current import (
    AdvanceLedger,
    replace_with_observer,
    run,
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
    counts = {'test': request.node.name, 'cuda_models': 0,
              'cuda_setup_kernel_steps': 0,
              'synthetic_native_ticks': 0,
              'synthetic_native_s': 0., 'actual_cuda': False,
              'body_steps': 0, 'wall_s': 0.}
    began = time.perf_counter()
    yield counts
    counts['wall_s'] = time.perf_counter()-began
    print('E125_SYNTHETIC_COUNT', counts)


def model(meter, *, observed=False, capacity=32, capture=2, weights=None):
    import torch

    assert torch.cuda.is_available(), 'Actual CUDA required; no skip/fallback'
    if weights is None:
        weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32),
            np.ones(4, np.float32), np.full(4, 7.5, np.float32),
            np.full(4, 200., np.float32))
    kwargs = {'adaptation_gain': 8., 'adaptation_tau_s': .3,
              'dt': .00025, 'device': 'cuda', 'cuda_implementation': 'packed',
              'capture_steps': capture, 'core_ids': np.array([1, 3]),
              'core_gain': 0.}
    if observed:
        network = ObservedCoreAdaptationNetwork(*args,
            observation_ids=np.array([1, 3]), capacity_stages=capacity,
            **kwargs)
    else:
        network = CoreAdaptationNetwork(*args, **kwargs)
    meter['cuda_models'] += 1
    meter['cuda_setup_kernel_steps'] += 3*capture
    meter['actual_cuda'] = True
    return network


def advance(network, drive, steps, meter):
    before = int(network.tick)
    try:
        network.advance(drive, steps=steps)
    finally:
        delta = int(network.tick)-before
        meter['synthetic_native_ticks'] += delta
        meter['synthetic_native_s'] += delta*.00025


def test_observer_is_bit_exact_and_captures_drive_after_csr(meter):
    weights = csr_matrix(np.array([
        [0., 0., 0., 0.], [.5, 0., 0., 0.],
        [0., 0., 0., 0.], [0., 0., 0., 0.]], np.float32))
    source = model(meter, weights=weights)
    observer = model(meter, observed=True, weights=weights)
    state = source.snapshot()
    state['rate'] = np.array([2., 1., 3., 4.], np.float32)
    state['adaptation'] = np.array([.2, .1, .4, .3], np.float32)
    source.restore(state)
    observer.restore(state)
    assert source.identity == observer.identity
    np.testing.assert_equal(source.snapshot(), observer.snapshot())
    assert np.all(observer._observed_counters.cpu().numpy() == 0)
    assert np.isnan(observer._observed_values.cpu().numpy()).all()
    observer.arm_observation()
    drive = np.array([1., 2., 3., 4.], np.float32)
    advance(source, drive, 4, meter)
    advance(observer, drive, 4, meter)
    np.testing.assert_equal(source.snapshot(), observer.snapshot())
    raw = observer.observation_raw()
    assert raw['total_current'].dtype == np.float32
    assert raw['counter_requested'].tolist() == [16, 16]
    assert raw['counter_completed'].tolist() == [16, 16]
    assert raw['overflow'].tolist() == [0]
    assert raw['start_tick'] == 0 and raw['end_tick'] == 4
    np.testing.assert_array_equal(raw['total_current'][0],
                                  np.array([3., 4.], np.float32))
    # Independent one-cell RK4: row 0 receives only drive=1 below threshold,
    # so its four stage rates follow dr/dt=-r/tau. Row 1 sees 2+0.5*r0.
    f = np.float32
    initial = f(2.)
    tau = f(.02)
    half_dt = f(.00025*.5)
    full_dt = f(.00025)
    k1 = f(-initial / tau)
    stage2 = f(initial + f(half_dt*k1))
    k2 = f(-stage2 / tau)
    stage3 = f(initial + f(half_dt*k2))
    k3 = f(-stage3 / tau)
    stage4 = f(initial + f(full_dt*k3))
    expected = np.asarray([f(f(2.)+f(f(.5)*rate))
                           for rate in (initial, stage2, stage3, stage4)],
                          dtype=np.float32)
    assert expected[1] != expected[2]
    np.testing.assert_array_equal(raw['total_current'][:4, 0], expected)
    np.testing.assert_array_equal(raw['total_current'][:4, 1],
                                  np.full(4, 4., np.float32))
    assert np.isfinite(raw['total_current'][:16]).all()
    assert np.isnan(raw['total_current'][16:]).all()
    assert meter['cuda_setup_kernel_steps'] == 12
    assert meter['synthetic_native_ticks'] == 8


def test_overflow_retains_requested_counter_and_nan_missing(meter):
    observer = model(meter, observed=True, capacity=5)
    observer.arm_observation()
    advance(observer, np.array([0., 2., 0., 4.], np.float32), 2, meter)
    raw = observer.observation_raw()
    assert raw['counter_requested'].tolist() == [8, 8]
    assert raw['counter_completed'].tolist() == [5, 5]
    assert raw['overflow'].tolist() == [1]
    assert raw['total_current'].shape == (5, 2)
    np.testing.assert_array_equal(raw['total_current'],
                                  np.tile([2., 4.], (5, 1)).astype(np.float32))
    assert meter['synthetic_native_ticks'] == 2


def test_armed_stream_ownership_and_single_arm(meter):
    import torch

    observer = model(meter, observed=True)
    before = observer.snapshot()
    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        with pytest.raises(RuntimeError, match='constructor CUDA stream'):
            observer.arm_observation()
        with pytest.raises(RuntimeError, match='constructor CUDA stream'):
            observer.restore(before)
    np.testing.assert_equal(before, observer.snapshot())
    assert observer._observation_start_tick is None
    observer.arm_observation()
    with pytest.raises(RuntimeError, match='Arm once'):
        observer.arm_observation()
    with torch.cuda.stream(other_stream), pytest.raises(
            RuntimeError, match='constructor CUDA stream'):
        advance(observer, np.zeros(4, np.float32), 2, meter)
    assert observer.tick == 0
    advance(observer, np.zeros(4, np.float32), 2, meter)
    raw = observer.observation_raw()
    assert raw['counter_requested'].tolist() == [8, 8]
    assert meter['synthetic_native_ticks'] == 2


def test_disable_freezes_endpoint_and_restore_never_rewinds_recording(meter):
    observer = model(meter, observed=True)
    zero = observer.snapshot()
    observer.arm_observation()
    advance(observer, np.array([0., 2., 0., 4.], np.float32), 2, meter)
    armed_state = observer.snapshot()
    with pytest.raises(RuntimeError, match='Cannot restore'):
        observer.restore(zero)
    np.testing.assert_equal(armed_state, observer.snapshot())
    observer.disable_observation()
    first = observer.observation_raw()
    assert first['end_tick'] == 2
    assert first['counter_requested'].tolist() == [8, 8]
    advance(observer, np.zeros(4, np.float32), 2, meter)
    second = observer.observation_raw()
    for key in first:
        np.testing.assert_equal(first[key], second[key])
    assert observer.tick == 4 and second['end_tick'] == 2
    later_state = observer.snapshot()
    with pytest.raises(RuntimeError, match='Cannot restore'):
        observer.restore(zero)
    np.testing.assert_equal(later_state, observer.snapshot())
    assert meter['synthetic_native_ticks'] == 4


def test_invalid_observer_contract_fails_without_model(meter):
    weights = csr_matrix((4, 4), dtype=np.float32)
    args = (weights, np.full(4, .02, np.float32), np.ones(4, np.float32),
            np.full(4, 7.5, np.float32), np.full(4, 200., np.float32))
    common = {'adaptation_gain': 8., 'adaptation_tau_s': .3,
              'dt': .00025, 'capture_steps': 2,
              'core_ids': np.array([1, 3]), 'core_gain': 0.}
    for ids, capacity in (([1, 1], 8), ([1, 4], 8), ([1], 0)):
        with pytest.raises(ValueError):
            ObservedCoreAdaptationNetwork(*args, observation_ids=ids,
                capacity_stages=capacity, device='cuda',
                cuda_implementation='packed', **common)
    with pytest.raises(ValueError, match='packed CUDA'):
        ObservedCoreAdaptationNetwork(*args, observation_ids=[1],
            capacity_stages=8, device='cpu',
            cuda_implementation='reference', **common)
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_wrapper_network_replacement_keeps_checkpoint_identity(meter):
    source = model(meter)
    state = source.snapshot()
    state['rate'] = np.array([.1, .2, .3, .4], np.float32)
    state['adaptation'] = np.array([.4, .3, .2, .1], np.float32)
    source.restore(state)
    original = source.snapshot()
    wrapper = SimpleNamespace(network=source)
    observed = replace_with_observer(wrapper, np.array([1, 3]), 16)
    meter['cuda_models'] += 1
    meter['cuda_setup_kernel_steps'] += 6
    assert observed is wrapper.network
    np.testing.assert_equal(original, observed.snapshot())
    assert observed.identity == source.identity
    assert meter['cuda_models'] == 2
    assert meter['synthetic_native_ticks'] == 0


def test_missing_freeze_fails_closed_with_cost_artifacts(tmp_path, meter):
    out = tmp_path/'single-output'
    with pytest.raises(FileNotFoundError):
        run(tmp_path/'missing-freeze.json', tmp_path/'missing-review.json', out)
    assert out.is_dir()
    assert (out/'advance-ledger.json').is_file()
    assert (out/'setup-and-cost.json').is_file()
    assert (out/'execution-errors.json').is_file()
    assert not (out/'result.json').exists()
    with pytest.raises(FileExistsError):
        run(tmp_path/'missing-freeze.json', tmp_path/'missing-review.json', out)
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_advance_exception_keeps_requested_work_unknown(meter):
    class Network:
        tick = 0

    class Body:
        def physics_time(self):
            return 0.

    class Session:
        control_tick = 0
        network = Network()
        body = Body()

        def advance(self, controls):
            self.network.tick = 20
            raise RuntimeError('injected neural completion uncertainty')

    ledger = AdvanceLedger()
    with pytest.raises(RuntimeError, match='injected'):
        ledger.advance(Session())
    event = ledger.events[0]
    assert event['requested_controls'] == 1
    assert event['neural_after'] == 20
    assert event['neural_completed_s'] is None
    assert event['cuda_completion'] == 'unknown_after_error'
    assert ledger.totals()['actual_neural_s'] is None
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_synthetic_frozen_hashes_and_source_review_gate(tmp_path, monkeypatch,
                                                        meter):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)

    def add(relative, data):
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return runner.file_sha(path)

    design_sha = add(runner.DESIGN, b'synthetic E125 design')
    source_sha = {name: add(name, name.encode())
                  for name in runner.SOURCE_FILES}
    maps = {}
    for key, count in (('input_sha256', 70),
                       ('reused_source_sha256', 12),
                       ('preservation_sha256', 581)):
        maps[key] = {f'{key}/{index:04}.bin':
                     add(f'{key}/{index:04}.bin', b'x')
                     for index in range(count)}
    path_names = ('graph', 'start', 'future', 'calibration',
                  'metadata', 'signals', 'trace')
    paths = {name: f'input_sha256/{index:04}.bin'
             for index, name in enumerate(path_names)}
    freeze = {'schema': runner.SCHEMA, 'design_path': runner.DESIGN,
        'design_sha256': design_sha, 'paths': paths, **maps,
        'graph_n': 158706, 'graph_edges': 11584852,
        'graph_hash': '9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff',
        'controls': 240, 'neural_dt_s': .00025, 'control_dt_s': .005,
        'physical_dt_s': .0001, 'capacity_stages': 19200,
        'observed_neurons': 391, 'future_control': 220,
        'actual_neural_s': 1.2, 'actual_physical_s': 1.2,
        'internal_wall_limit_s': 90, 'finalize_reserve_s': 15,
        'external_wall_limit_s': 120, 'supervisor_limit_s': 125,
        'prior_conservative_process_s': 951,
        'shared_process_limit_s': 1076, 'automatic_retries': 0,
        'fit_calls': 0, 'F6_exposed': False, 'candidate_adopted': False}
    freeze_path = tmp_path/'freeze.json'
    freeze_path.write_text(json.dumps(freeze), encoding='utf-8')
    review = {'accepted_for_single_original_native_current': True,
        'blocking_defects': [], 'freeze_sha256': runner.file_sha(freeze_path),
        'source_sha256': source_sha}
    review_path = tmp_path/'review.json'
    review_path.write_text(json.dumps(review), encoding='utf-8')
    checked, _ = runner.verify_contract(freeze_path, review_path,
        freeze_sha=runner.file_sha(freeze_path),
        review_sha=runner.file_sha(review_path))
    assert checked == freeze
    review['source_sha256'][runner.SOURCE_FILES[0]] = '0'*64
    review_path.write_text(json.dumps(review), encoding='utf-8')
    with pytest.raises(ValueError, match='source review'):
        runner.verify_contract(freeze_path, review_path)
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0


def test_runner_enforces_measured_threads_before_graph_import(meter):
    script = (
        'import json,sys; '
        'from tools.verify_c_banc_m3_native_current import configure_single_thread_runtime; '
        'm=configure_single_thread_runtime(); '
        'print(json.dumps({"torch_intraop":m["torch_intraop"], '
        '"torch_interop":m["torch_interop"], '
        '"pool_threads":[p["num_threads"] for p in m["threadpools"] '
        'if p["user_api"] in ("blas","openmp")], '
        '"graph_imported":"flylab.c.graph" in sys.modules}))'
    )
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
        cwd=runner.ROOT, text=True, capture_output=True, timeout=20,
        check=True)
    measured = json.loads(completed.stdout)
    assert measured['torch_intraop'] == measured['torch_interop'] == 1
    assert measured['pool_threads']
    assert set(measured['pool_threads']) == {1}
    assert measured['graph_imported'] is False
    assert meter['cuda_models'] == meter['synthetic_native_ticks'] == 0
