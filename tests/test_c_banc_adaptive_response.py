"""Tiny conditional solvers only; no original graph, neural run, or body."""

import hashlib
import json
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools import banc_m3_adaptive_response as response
from tools import verify_c_banc_m3_adaptive_response as runner


@pytest.fixture(autouse=True)
def cpu_threads():
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def meter(request):
    counts = {'test': request.node.name, 'cpu_calls': 0, 'cuda_calls': 0,
              'cuda_setup_calls': 0, 'dop853_calls': 0,
              'cuda_known_failure_ticks': 0,
              'cuda_unknown_requested_ticks': 0,
              'cpu_context_ticks': 0, 'cuda_context_ticks': 0,
              'cuda_setup_context_ticks': 0,
              'dop853_integrated_s': 0.,
              'dop853_function_evaluations': 0, 'actual_cuda': False,
              'wall_s': 0.}
    began = time.perf_counter()
    yield counts
    counts['wall_s'] = time.perf_counter()-began
    print('E124_SYNTHETIC_COUNT', counts)


def fixture(contexts=2, samples=7):
    neurons, muscles = 4, 3
    membership = np.array([0, 1, 1, 2], np.int64)
    pool = np.zeros((muscles, neurons), np.float64)
    pool[membership, np.arange(neurons)] = [1., .3, .7, 1.]
    t = np.arange(samples, dtype=np.float64)*.005
    current = np.empty((contexts, samples, neurons), np.float64)
    for context in range(contexts):
        for neuron in range(neurons):
            current[context, :, neuron] = (
                10.+context*.7+neuron*.2+2.*np.sin(23.*t+neuron))
    return response.prepare_problem({
        'time_s': t, 'current': current,
        'initial_rate': np.array([[2., 3., 4., 5.]]*contexts),
        'initial_adaptation': np.array([[.4, .7, .2, .9]]*contexts),
        'initial_activation': np.array([[.8, 1.3, .6]]*contexts),
        'tau': np.array([.02, .025, .03, .022]),
        'a': [1., 1.1, .9, 1.2], 'threshold': [7.5]*neurons,
        'cap': [200.]*neurons, 'pool': pool,
        'moment': [[.2, -.1, .3], [.1, .5, -.2]],
        'rest': [.05, -.03], 'ltm': np.array([0, 2]),
        'membership': membership, 'motor_ids': np.array([2, 7, 9, 11])})


def cpu(problem, gains, meter):
    value, timing = response.solve_cpu(problem, gains)
    meter['cpu_calls'] += 1
    meter['cpu_context_ticks'] += timing['context_rk4_ticks']
    assert timing['conditional_model_s'] == (
        problem['current'].shape[0]*(problem['current'].shape[1]-1)*.005)
    return value


def cuda(problem, gains, meter):
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1, 'Actual CUDA required; no skip'
    value, timing = response.solve_cuda(problem, gains)
    meter['actual_cuda'] = True
    meter['cuda_calls'] += 1
    meter['cuda_context_ticks'] += timing['context_rk4_ticks']
    return value


def assert_fields_close(left, right, atol=5e-10):
    for key in ('rate', 'adaptation', 'rate_sensitivity',
                'adaptation_sensitivity', 'activation',
                'activation_sensitivity', 'command', 'tendon'):
        np.testing.assert_allclose(left[key], right[key], rtol=0, atol=atol)


def test_cpu_cuda_nonconstant_current_nonzero_initial_and_batch(meter):
    problem = fixture()
    gain = np.array([.3, 1.2, 2.7])
    setup = response.cuda_warmup()
    meter['cuda_setup_calls'] += 1
    meter['cuda_setup_context_ticks'] += setup['setup_context_rk4_ticks']
    assert setup['setup_context_rk4_ticks'] == 20
    reference = cpu(problem, gain, meter)
    actual = cuda(problem, gain, meter)
    assert_fields_close(reference, actual)
    np.testing.assert_array_equal(actual['rate'][:, 0],
                                  problem['initial_rate'])
    np.testing.assert_array_equal(actual['adaptation'][:, 0],
                                  problem['initial_adaptation'])
    np.testing.assert_array_equal(actual['activation'][:, 0],
                                  problem['initial_activation'])
    assert not np.array_equal(actual['command'][:, 0],
                              np.broadcast_to(problem['rest'], (2, 2)))
    for context in range(2):
        single = dict(problem)
        for key in ('current', 'initial_rate', 'initial_adaptation',
                    'initial_activation'):
            single[key] = problem[key][context:context+1]
        separate = cuda(single, gain, meter)
        for key in reference:
            np.testing.assert_array_equal(actual[key][context:context+1],
                                          separate[key])
    assert meter['cpu_context_ticks'] == 240
    assert meter['cuda_context_ticks'] == 480


def test_zero_current_gain0_and_positive_independent_dop853(meter):
    problem = fixture(contexts=1, samples=5)
    problem['current'][:] = 0.
    for gain in (0., 1.4):
        solved = cpu(problem, np.full(3, gain), meter)
        t = problem['time_s']
        for neuron in range(4):
            initial = [problem['initial_rate'][0, neuron],
                       problem['initial_adaptation'][0, neuron], 0., 0.]

            def derivative(_, y, neuron=neuron, gain=gain):
                r, z, v, w = y
                x = problem['a'][neuron]*(
                    -problem['threshold'][neuron]-z)/problem['cap'][neuron]
                tanh = np.tanh(x)
                phi = max(problem['cap'][neuron]*tanh, 0.)
                prime = problem['a'][neuron]*(1.-tanh*tanh) if tanh > 0 else 0.
                return [(phi-r)/problem['tau'][neuron],
                        (gain*r-z)/.3,
                        (-prime*w-v)/problem['tau'][neuron],
                        (r+gain*v-w)/.3]

            truth = solve_ivp(derivative, (0., t[-1]), initial, method='DOP853',
                              t_eval=t, rtol=1e-12, atol=1e-13,
                              max_step=.000125)
            meter['dop853_function_evaluations'] += truth.nfev
            meter['dop853_calls'] += 1
            meter['dop853_integrated_s'] += t[-1]
            assert truth.success
            for index, field in enumerate(('rate', 'adaptation',
                                           'rate_sensitivity',
                                           'adaptation_sensitivity')):
                np.testing.assert_allclose(solved[field][0, :, neuron],
                                           truth.y[index], rtol=0, atol=2e-9)
        np.testing.assert_allclose(solved['rate'][0, :, 0],
            problem['initial_rate'][0, 0]*np.exp(-t/problem['tau'][0]),
            rtol=0, atol=2e-9)
    assert meter['cpu_context_ticks'] == 160


def test_piecewise_linear_dop853_and_finite_difference(meter):
    problem = fixture(contexts=1, samples=6)
    gain = np.array([1.1, 1.1, 1.1])
    center = cpu(problem, gain, meter)
    plus = cpu(problem, gain+runner.EPS, meter)
    minus = cpu(problem, gain-runner.EPS, meter)
    sensitivity, passed = runner.sensitivity_metrics(problem, center, plus, minus)
    assert passed, sensitivity
    neuron = 2
    t = problem['time_s']
    y0 = [problem['initial_rate'][0, neuron],
          problem['initial_adaptation'][0, neuron], 0., 0.]

    def derivative(now, y):
        r, z, v, w = y
        sample = min(int(now/.005), len(t)-2)
        fraction = min(max((now-t[sample])/.005, 0.), 1.)
        current = (problem['current'][0, sample, neuron]*(1.-fraction)
                   +problem['current'][0, sample+1, neuron]*fraction)
        a = problem['a'][neuron]
        cap = problem['cap'][neuron]
        tanh = np.tanh(a*(current-problem['threshold'][neuron]-z)/cap)
        phi = max(cap*tanh, 0.)
        prime = a*(1.-tanh*tanh) if tanh > 0 else 0.
        return [(phi-r)/problem['tau'][neuron], (1.1*r-z)/.3,
                (-prime*w-v)/problem['tau'][neuron],
                (r+1.1*v-w)/.3]

    truth = solve_ivp(derivative, (0., t[-1]), y0, method='DOP853',
                      t_eval=t, rtol=1e-12, atol=1e-13,
                      max_step=.0001)
    meter['dop853_function_evaluations'] += truth.nfev
    meter['dop853_calls'] += 1
    meter['dop853_integrated_s'] += t[-1]
    assert truth.success
    for index, field in enumerate(('rate', 'adaptation',
                                   'rate_sensitivity', 'adaptation_sensitivity')):
        np.testing.assert_allclose(center[field][0, :, neuron], truth.y[index],
                                   rtol=0, atol=2e-7)
    assert meter['cpu_context_ticks'] == 300


def test_cpu_cuda_gain0_and_finite_difference(meter):
    problem = fixture(contexts=1, samples=5)
    for gain in (0., 1., 1.+runner.EPS, 1.-runner.EPS):
        values = np.full(3, gain)
        expected = cpu(problem, values, meter)
        actual = cuda(problem, values, meter)
        assert_fields_close(expected, actual)
    assert meter['cpu_context_ticks'] == meter['cuda_context_ticks'] == 320


def test_validation_rejects_time_membership_ids_gain_and_nonfinite(meter):
    problem = fixture()
    for key, value in (
            ('time_s', np.array([0., .006])),
            ('membership', np.array([0, 0, 1, 2])),
            ('motor_ids', np.array([2, 7, 7, 11])),
            ('tau', np.array([.02, 0., .03, .02])),
            ('initial_adaptation', -np.ones((2, 4))),
            ('initial_activation', -np.ones((2, 3))),
            ('current', np.full((2, 7, 4), np.nan))):
        damaged = dict(problem)
        damaged[key] = value
        with pytest.raises(ValueError):
            response.prepare_problem(damaged)
    for gain in ([-.01, 0., 0.], [0., 9., 0.], [0., np.nan, 0.],
                 [0., 1.], [0., 1., 2., 3.]):
        with pytest.raises(ValueError):
            response.checked_gain(problem, gain)
    assert meter['cpu_context_ticks'] == meter['cuda_context_ticks'] == 0


def test_prepared_inputs_and_persistent_cuda_upload_are_isolated(meter):
    import cupy as cp

    original = fixture(contexts=1, samples=3)
    copies = {key: value.copy() for key, value in original.items()}
    prepared = response.prepare_problem(original)
    assert all(not np.shares_memory(prepared[key], original[key])
               for key in prepared)
    engine = response.CudaAdaptiveResponseEngine(prepared)
    assert engine.metadata()['fixed_upload_count'] == len(engine.FIXED_KEYS) == 13
    assert engine.metadata()['fixed_upload_bytes'] == sum(
        engine.problem[key].nbytes for key in engine.FIXED_KEYS)
    prepared['current'][:] = 0.
    prepared['pool'][:] = 0.
    original['current'][:] = 0.
    original['pool'][:] = 0.
    reference = cpu(copies, np.ones(3), meter)
    first, first_timing = engine.solve(np.ones(3))
    # A caller changing streams must not race the engine's fixed uploads.
    with cp.cuda.Stream(non_blocking=True):
        second, second_timing = engine.solve(np.ones(3))
    assert_fields_close(first, reference)
    assert_fields_close(first, second)
    assert engine.metadata()['fixed_upload_count'] == 13
    assert engine.metadata()['solve_calls'] == 2
    assert first_timing['decoder_backend'].startswith('cupy_cuda')
    assert second_timing['decoder_endpoint_ticks'] == 2
    meter['cuda_calls'] += 2
    meter['cuda_context_ticks'] += 80
    meter['actual_cuda'] = True


def test_readback_failure_after_sync_keeps_completed_ticks_and_raw(monkeypatch,
                                                                    meter):
    import cupy as cp

    engine = response.CudaAdaptiveResponseEngine(fixture(contexts=1, samples=2))
    native_asnumpy = cp.asnumpy
    calls = {'count': 0}

    def fail_second(value, *args, **kwargs):
        calls['count'] += 1
        if calls['count'] == 2:
            raise RuntimeError('injected second transfer failure')
        return native_asnumpy(value, *args, **kwargs)

    monkeypatch.setattr(cp, 'asnumpy', fail_second)
    with pytest.raises(RuntimeError, match='second transfer failure') as caught:
        engine.solve(np.ones(3))
    assert caught.value.completed_context_rk4_ticks == 20
    assert caught.value.completed_conditional_model_s == .005
    progress = caught.value.response_progress
    assert progress['stage'] == 'decoder_synchronized'
    assert progress['decoder_endpoint_ticks_completed'] == 1
    assert all(np.isfinite(value).all()
               for value in caught.value.partial_response.values())
    assert set(caught.value.partial_response) == {
        'rate', 'adaptation', 'rate_sensitivity', 'adaptation_sensitivity',
        'activation', 'activation_sensitivity', 'command', 'tendon'}
    meter['cuda_known_failure_ticks'] += 20


def test_sync_failure_is_unknown_requested_work(monkeypatch, meter):
    engine = response.CudaAdaptiveResponseEngine(fixture(contexts=1, samples=2))

    def fail_sync():
        raise RuntimeError('injected sync failure')

    monkeypatch.setattr(response, '_synchronize', fail_sync)
    with pytest.raises(RuntimeError, match='sync failure') as caught:
        engine.solve(np.ones(3))
    progress = caught.value.response_progress
    assert progress['stage'] == 'enqueued'
    assert progress['completion_class'] == 'unknown'
    assert caught.value.requested_context_rk4_ticks == 20
    assert not hasattr(caught.value, 'completed_context_rk4_ticks')
    meter['cuda_unknown_requested_ticks'] += 20


def test_cuda_failure_has_no_cpu_fallback(monkeypatch, meter):
    problem = fixture(contexts=1, samples=2)

    def fail_gpu(*_):
        raise RuntimeError('injected CUDA launch failure')

    def fail_cpu(*_):
        raise AssertionError('CPU fallback forbidden')

    engine = response.CudaAdaptiveResponseEngine(problem)
    monkeypatch.setattr(response, '_kernel', lambda: fail_gpu)
    monkeypatch.setattr(response, 'solve_cpu', fail_cpu)
    with pytest.raises(RuntimeError, match='injected CUDA launch failure') as caught:
        engine.solve(np.ones(3))
    assert caught.value.response_progress['completion_class'] == 'not_enqueued'
    assert caught.value.requested_context_rk4_ticks == 20
    assert meter['cpu_context_ticks'] == meter['cuda_context_ticks'] == 0


def test_completed_cuda_integral_is_charged_on_decoder_failure(monkeypatch, meter):
    problem = fixture(contexts=1, samples=2)
    engine = response.CudaAdaptiveResponseEngine(problem)

    def broken_decoder(*_):
        raise ValueError('injected endpoint decoder failure')

    monkeypatch.setattr(engine, '_gpu_decode', broken_decoder)
    with pytest.raises(ValueError, match='endpoint decoder failure') as caught:
        engine.solve(np.ones(3))
    assert caught.value.completed_context_rk4_ticks == 20
    assert caught.value.completed_conditional_model_s == .005
    assert caught.value.response_progress['stage'] == 'decoder_requested'
    assert set(caught.value.partial_response) == {
        'rate', 'adaptation', 'rate_sensitivity',
        'adaptation_sensitivity'}
    assert all(np.isfinite(value).all()
               for value in caught.value.partial_response.values())
    meter['cuda_known_failure_ticks'] += 20


def test_fidelity_reports_original_99_and_excludes_t0(meter):
    shape = (3, 2001)
    actual = {'context_names': np.array(['E110', 'E113', 'E121']),
              'startup_mask': np.isin(np.arange(2001), np.arange(1, 100)),
              'actual_motor_rate': np.ones((*shape, 4)),
              'actual_adaptation': np.ones((*shape, 4)),
              'actual_activation': np.ones((*shape, 3)),
              'actual_command': np.ones((*shape, 2)),
              'actual_tendon': np.ones((*shape, 2))}
    predicted = {'rate': actual['actual_motor_rate'].copy(),
                 'adaptation': actual['actual_adaptation'].copy(),
                 'activation': actual['actual_activation'].copy(),
                 'command': actual['actual_command'].copy(),
                 'tendon': actual['actual_tendon'].copy()}
    predicted['rate'][:, 0] += 1000.
    metrics, passed = runner.fidelity_metrics(predicted, actual)
    assert passed
    assert metrics['rate']['E110']['all_2000']['rms'] == 0.
    assert metrics['rate']['E110']['startup_99']['rms'] == 0.
    predicted['rate'][:, 100, 0] += 6.
    metrics, passed = runner.fidelity_metrics(predicted, actual)
    assert not passed
    assert metrics['rate']['E110']['all_2000']['max_location'] == [0, 100, 0]
    assert metrics['rate']['E110']['startup_99']['max_abs'] == 0.


def test_one_shot_raw_and_no_review_fail_closed(tmp_path, meter):
    problem = fixture(contexts=1, samples=2)
    raw = cpu(problem, np.ones(3), meter)
    path = tmp_path/'gain1-raw.npz'
    runner.save_response(path, np.ones(3), raw)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['rate'], raw['rate'])
    with pytest.raises(FileExistsError):
        runner.save_response(path, np.ones(3), raw)
    with pytest.raises(FileExistsError):
        runner.run(None, None, None, None, tmp_path)
    with pytest.raises(TimeoutError):
        runner.room(100., 'synthetic', now=175.)
    assert meter['cpu_context_ticks'] == 20


def test_review_hash_fail_closed_before_fixture_access(tmp_path):
    freeze = tmp_path/'freeze.json'
    freeze.write_text('{}', encoding='utf-8')
    actual_sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='frozen design SHA'):
        runner.reviewed_inputs(freeze, '0'*64, None, None)
    with pytest.raises(ValueError, match='frozen design contract'):
        runner.reviewed_inputs(freeze, actual_sha, None, None)


def test_observed_one_thread_contract():
    limits, observed = runner.one_cpu_thread()
    try:
        assert observed['torch_intraop'] == 1
        assert observed['torch_interop'] == 1
        assert all(pool['num_threads'] == 1 for pool in observed['threadpools'])
    finally:
        limits.restore_original_limits()


def _stubbed_runner(monkeypatch, tmp_path, *, fail_solve=False,
                    nonfinite=False):
    """Exercise receipts without loading the recorded fixture or a real GPU solve."""
    dummy = tmp_path/'synthetic-fixture-marker'
    dummy.write_bytes(b'synthetic fixture marker, no recorded data')

    class Limits:
        def restore_original_limits(self):
            pass

    monkeypatch.setattr(runner, 'one_cpu_thread',
                        lambda: (Limits(), {'synthetic_threads': 1}))
    monkeypatch.setattr(runner, 'reviewed_inputs',
        lambda *_: ({'design_sha256': 'synthetic'}, dummy,
                    {'source_sha256': {}}))
    monkeypatch.setattr(runner, 'save_review_snapshot', lambda *_: None)
    monkeypatch.setattr(runner, 'original_fixture',
                        lambda *_: (fixture(contexts=1, samples=2), {}))
    monkeypatch.setattr(runner, 'cuda_warmup', lambda: {
        'setup_context_rk4_ticks': 20, 'setup_conditional_model_s': .005,
        'backend': 'mocked', 'decoder_backend': 'mocked', 'wall_s': 0.})
    calls = {'solve': 0}

    class StubEngine:
        fixed_upload_count = 13
        solve_calls = 0

        def __init__(self, _):
            pass

        def metadata(self):
            return {'fixed_upload_count': 13, 'fixed_upload_bytes': 0,
                    'decoder_backend': 'mocked',
                    'solve_calls': self.solve_calls}

        def solve(self, _):
            calls['solve'] += 1
            self.solve_calls += 1
            if fail_solve:
                error = RuntimeError('injected unknown async completion')
                error.response_progress = {'stage': 'enqueued',
                    'completion_class': 'unknown',
                    'decoder_endpoint_ticks_completed': None}
                error.partial_response = {'rate': np.ones((1, 2, 4))}
                raise error
            fields = {'rate': np.ones((1, 2, 4)),
                'adaptation': np.ones((1, 2, 4)),
                'rate_sensitivity': np.zeros((1, 2, 4)),
                'adaptation_sensitivity': np.zeros((1, 2, 4)),
                'activation': np.ones((1, 2, 3)),
                'activation_sensitivity': np.zeros((1, 2, 3)),
                'command': np.ones((1, 2, 2)),
                'tendon': np.ones((1, 2, 2))}
            if nonfinite:
                fields['rate'][0, 1, 0] = np.nan
            return fields, {'context_rk4_ticks': 20,
                'conditional_model_s': .005, 'decoder_endpoint_ticks': 1,
                'progress': {'stage': 'host_transfer_complete',
                             'completion_class': 'known_complete'},
                'wall_s': 0.}

    monkeypatch.setattr(runner, 'CudaAdaptiveResponseEngine', StubEngine)
    return calls


def test_runner_preserves_full_raw_before_postcompute_ledger_failure(
        monkeypatch, tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path)
    native_write = runner.write_json

    def fail_after_compute(path, data):
        if (path.name == 'advance-ledger.json'
                and data.get('context_rk4_ticks', 0) > 0):
            raise OSError('injected post-compute ledger failure')
        native_write(path, data)

    monkeypatch.setattr(runner, 'write_json', fail_after_compute)
    output = tmp_path/'mocked-ledger-failure'
    with pytest.raises(OSError, match='post-compute ledger failure'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 1
    with np.load(output/'gain0-raw.npz', allow_pickle=False) as raw:
        assert raw['rate'].shape == (1, 2, 4)
        assert raw['activation'].shape == (1, 2, 3)
    error = json.loads((output/'execution-error.json').read_text())
    assert error['primary']['error'] == 'OSError'
    assert error['context_rk4_ticks'] == 20
    assert error['actual_neural_s'] == 0.
    environment = json.loads((output/'environment.json').read_text())
    assert environment['fixed_upload']['fixed_upload_count'] == 13
    assert environment['decoder_backend'] == 'cupy_cuda_float64_pool_ema_q_tendon'


def test_runner_unknown_async_work_reserves_requested_and_saves_partial(
        monkeypatch, tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path, fail_solve=True)
    output = tmp_path/'mocked-unknown-completion'
    with pytest.raises(RuntimeError, match='unknown async completion'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 1
    with np.load(output/'gain0-partial-raw.npz', allow_pickle=False) as raw:
        assert raw['rate'].shape == (1, 2, 4)
    ledger = json.loads((output/'advance-ledger.json').read_text())
    assert ledger['requested_context_rk4_ticks'] == 120000
    assert ledger['unknown_requested_context_rk4_ticks'] == 120000
    assert ledger['context_rk4_ticks'] == 0
    assert ledger['batch_attempts'][0]['completion_class'] == 'unknown'


def test_runner_saves_nonfinite_raw_then_stops_before_next_arm(
        monkeypatch, tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path, nonfinite=True)
    output = tmp_path/'mocked-nonfinite-raw'
    with pytest.raises(ValueError, match='Nonfinite CUDA raw rate'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 1
    with np.load(output/'gain0-raw.npz', allow_pickle=False) as raw:
        assert np.isnan(raw['rate'][0, 1, 0])
    assert not (output/'gain1-raw.npz').exists()


def test_runner_setup_failure_retains_requested_unknown_reservation(
        monkeypatch, tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path)

    def fail_setup():
        error = RuntimeError('injected CUDA setup sync failure')
        error.response_progress = {'stage': 'enqueued',
                                   'completion_class': 'unknown'}
        raise error

    monkeypatch.setattr(runner, 'cuda_warmup', fail_setup)
    output = tmp_path/'mocked-setup-failure'
    with pytest.raises(RuntimeError, match='setup sync failure'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 0
    ledger = json.loads((output/'advance-ledger.json').read_text())
    assert ledger['cuda_setup_status'] == 'failed'
    assert ledger['cuda_setup_requested_context_rk4_ticks'] == 20
    assert ledger['cuda_setup_unknown_requested_context_rk4_ticks'] == 20


def test_runner_setup_readback_failure_records_known_twenty_ticks(
        monkeypatch, tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path)

    def fail_setup():
        error = RuntimeError('injected setup readback failure')
        error.response_progress = {'stage': 'decoder_synchronized',
                                   'completion_class': 'known_complete',
                                   'decoder_endpoint_ticks_completed': 1}
        error.completed_context_rk4_ticks = 20
        error.completed_conditional_model_s = .005
        error.partial_response = {'rate': np.zeros((1, 2, 1), np.float64)}
        raise error

    monkeypatch.setattr(runner, 'cuda_warmup', fail_setup)
    output = tmp_path/'mocked-setup-known-failure'
    with pytest.raises(RuntimeError, match='setup readback failure'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 0
    ledger = json.loads((output/'advance-ledger.json').read_text())
    assert ledger['cuda_setup_context_rk4_ticks'] == 20
    assert ledger['cuda_setup_conditional_model_s'] == .005
    assert ledger['cuda_setup_unknown_requested_context_rk4_ticks'] == 0
    assert ledger['cuda_setup_decoder_endpoint_ticks_completed'] == 1
    with np.load(output/'cuda-setup-partial-raw.npz', allow_pickle=False) as raw:
        assert raw['rate'].shape == (1, 2, 1)


def test_decoder_failure_after_ema_preserves_completed_intermediate_outputs(monkeypatch, meter):
    import cupy as cp

    engine = response.CudaAdaptiveResponseEngine(fixture(contexts=1, samples=2))

    def fail_tendon(*_):
        raise RuntimeError('injected failure after EMA and command')

    monkeypatch.setattr(cp, 'tanh', fail_tendon)
    with pytest.raises(RuntimeError, match='after EMA and command') as caught:
        engine.solve(np.ones(3))
    assert caught.value.completed_context_rk4_ticks == 20
    partial = caught.value.partial_response
    assert set(partial) == {'rate', 'adaptation', 'rate_sensitivity',
        'adaptation_sensitivity', 'activation', 'activation_sensitivity', 'command'}
    assert all(np.isfinite(value).all() for value in partial.values())
    assert caught.value.response_progress['decoder_completion_class'] == 'unknown'
    meter['cuda_known_failure_ticks'] += 20


def test_warmup_post_solve_metadata_failure_keeps_completion_and_raw(monkeypatch, meter):
    import cupy as cp

    def fail_metadata(*_):
        raise RuntimeError('injected post-solve setup metadata failure')

    monkeypatch.setattr(cp.cuda.runtime, 'getDeviceProperties', fail_metadata)
    with pytest.raises(RuntimeError, match='post-solve setup metadata') as caught:
        response.cuda_warmup()
    assert caught.value.completed_context_rk4_ticks == 20
    assert caught.value.completed_conditional_model_s == .005
    assert caught.value.response_progress['decoder_endpoint_ticks_completed'] == 1
    assert caught.value.fixed_upload_count_completed == 13
    assert len(caught.value.partial_response) == 8
    assert all(np.isfinite(value).all() for value in caught.value.partial_response.values())
    meter['cuda_setup_calls'] += 1
    meter['cuda_setup_context_ticks'] += 20


def test_raw_shape_and_precision_rejected_before_another_arm():
    problem = fixture(contexts=1, samples=2)
    values = {key: np.ones((1, 2, width), np.float64) for key, width in {
        'rate': 4, 'adaptation': 4, 'rate_sensitivity': 4,
        'adaptation_sensitivity': 4, 'activation': 3,
        'activation_sensitivity': 3, 'command': 2, 'tendon': 2}.items()}
    runner.validate_saved_response(values, problem)
    for malformed in (np.ones((1, 2, 4), np.float32), np.ones((2, 1, 4), np.float64)):
        bad = dict(values)
        bad['rate'] = malformed
        with pytest.raises(ValueError, match='Wrong CUDA raw shape or dtype'):
            runner.validate_saved_response(bad, problem)


def test_synchronize_waits_on_the_actual_current_stream(monkeypatch):
    from types import SimpleNamespace

    import cupy as cp

    calls = []
    monkeypatch.setattr(cp.cuda, 'get_current_stream',
        lambda: SimpleNamespace(synchronize=lambda: calls.append('current')))
    response._synchronize()
    assert calls == ['current']


def test_runner_final_raw_rescue_keeps_original_save_error(monkeypatch,
                                                           tmp_path):
    calls = _stubbed_runner(monkeypatch, tmp_path)
    native_save = runner.save_response

    def fail_first_two(path, gain, result):
        if path.name in ('gain0-raw.npz', 'gain0-emergency-raw.npz'):
            raise OSError('injected primary raw save failure')
        native_save(path, gain, result)

    monkeypatch.setattr(runner, 'save_response', fail_first_two)
    output = tmp_path/'mocked-raw-rescue'
    with pytest.raises(OSError, match='primary raw save failure'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    assert calls['solve'] == 1
    with np.load(output/'gain0-final-rescue-raw.npz', allow_pickle=False) as raw:
        assert raw['rate'].shape == (1, 2, 4)
    error = json.loads((output/'execution-error.json').read_text())
    assert error['primary']['error'] == 'OSError'
    assert error['context_rk4_ticks'] == 20


def test_review_snapshot_copies_all_six_exact_files(tmp_path):
    freeze = tmp_path/'freeze.json'
    review = tmp_path/'source-review.json'
    freeze.write_text('{"synthetic":true}', encoding='utf-8')
    review.write_text('{"synthetic":true}', encoding='utf-8')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {name: digest(runner.ROOT/name) for name in runner.SOURCE_FILES}
    output = tmp_path/'snapshot-output'
    output.mkdir()
    runner.save_review_snapshot(output, freeze, digest(freeze),
                                review, digest(review),
                                {'source_sha256': sources})
    manifest = json.loads((output/'reviewed-source/manifest.json').read_text())
    assert len(manifest['files_sha256']) == 6
    for name, expected in manifest['files_sha256'].items():
        assert digest(output/'reviewed-source'/name) == expected


def test_preflight_failure_does_not_enter_cuda_cleanup(monkeypatch, tmp_path):
    class Limits:
        def restore_original_limits(self):
            pass

    monkeypatch.setattr(runner, 'one_cpu_thread',
                        lambda: (Limits(), {'synthetic_threads': 1}))

    def fail_review(*_):
        raise ValueError('injected preflight review failure')

    def forbidden_cuda(*_):
        raise AssertionError('CUDA must stay untouched before preflight')

    monkeypatch.setattr(runner, 'reviewed_inputs', fail_review)
    monkeypatch.setattr(runner, 'cuda_warmup', forbidden_cuda)
    monkeypatch.setattr(runner.gc, 'collect', forbidden_cuda)
    output = tmp_path/'preflight-failure'
    with pytest.raises(ValueError, match='preflight review failure'):
        runner.run(None, 'synthetic', None, 'synthetic', output)
    error = json.loads((output/'execution-error.json').read_text())
    assert error['primary']['error'] == 'ValueError'
    assert error['secondary'] == []
