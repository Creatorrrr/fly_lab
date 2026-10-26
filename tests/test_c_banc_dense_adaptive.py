"""Small synthetic E127 fixtures only; original E110/E113/E121 are never run."""

import math
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools import banc_m3_dense_adaptive as dense
from tools import verify_c_banc_m3_dense_adaptive as runner


@pytest.fixture(autouse=True)
def cpu_threads():
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        assert all(p['num_threads'] == 1 for p in threadpool_info())
        yield


@pytest.fixture
def meter(request):
    count = {'test': request.node.name, 'synthetic_cpu_calls': 0,
             'synthetic_cuda_calls': 0, 'synthetic_cuda_setup_calls': 0,
             'synthetic_cpu_context_ticks': 0,
             'synthetic_cuda_context_ticks': 0,
             'synthetic_cuda_setup_context_ticks': 0,
             'synthetic_dop853_calls': 0,
             'synthetic_dop853_function_evaluations': 0,
             'synthetic_dop853_integrated_s': 0.,
             'synthetic_cuda_known_failure_ticks': 0,
             'synthetic_cuda_unknown_requested_ticks': 0,
             'actual_cuda': False, 'wall_s': 0.}
    began = time.perf_counter()
    yield count
    count['wall_s'] = time.perf_counter()-began
    print('E127_SYNTHETIC_COUNT', count)


def fixture(contexts=2, controls=3):
    neurons, muscles = 4, 3
    membership = np.array([0, 1, 1, 2], np.int64)
    pool = np.zeros((muscles, neurons), np.float64)
    pool[membership, np.arange(neurons)] = [1., .3, .7, 1.]
    ticks = controls*20
    stage = np.empty((contexts, ticks*4, neurons), np.float32)
    for context in range(contexts):
        for tick in range(ticks):
            for neuron in range(neurons):
                base = (8.2 + context*.7 + neuron*.38
                        + 1.4*math.sin(.21*tick+.8*neuron))
                stage[context, tick*4:(tick+1)*4, neuron] = [
                    base-.7, base+.9, base-.45, base+.3]
    return {'stage_current': stage,
            'time_s': np.arange(controls+1, dtype=np.float64)*.005,
            'initial_rate': np.array([[2., 3., 4., 5.]]*contexts),
            'initial_adaptation': np.array([[.4, .7, .2, .9]]*contexts),
            'initial_activation': np.array([[.8, 1.3, .6]]*contexts),
            'tau': [.02, .025, .03, .022],
            'a': [1., 1.1, .9, 1.2],
            'threshold': [7.5, 8., 8.5, 7.8],
            'cap': [200., 220., 180., 210.],
            'pool': pool,
            'moment': [[.2, -.1, .3], [.1, .5, -.2]],
            'rest': [.05, -.03], 'ltm': np.array([0, 2]),
            'membership': membership, 'motor_ids': np.array([2, 7, 9, 11])}


def cpu_reference(problem, gain, meter):
    """Independent discrete four-stage RK4, separate from CUDA source."""
    p = dense.prepare_problem(problem)
    g = np.asarray(gain, np.float64)[p['membership']]
    c, stages, n = p['stage_current'].shape
    ticks = stages//4
    samples = ticks//20+1
    r = p['initial_rate'].copy()
    z = p['initial_adaptation'].copy()
    v, w = np.zeros_like(r), np.zeros_like(r)
    rate = np.empty((c, samples, n), np.float64)
    adaptation = np.empty_like(rate)
    vr = np.empty_like(rate)
    wz = np.empty_like(rate)
    branch = np.zeros((c, ticks, n), np.uint8)
    rate[:, 0], adaptation[:, 0], vr[:, 0], wz[:, 0] = r, z, v, w

    def rhs(y, current):
        rr, zz, vv, ww = y
        u = p['a']*(current-p['threshold']-zz)/p['cap']
        tanh = np.tanh(u)
        active = tanh > 0
        phi = np.where(active, p['cap']*tanh, 0.)
        prime = np.where(active, p['a']*(1.-tanh*tanh), 0.)
        return ((phi-rr)/p['tau'], (g*rr-zz)/.3,
                (-prime*ww-vv)/p['tau'], (rr+g*vv-ww)/.3), active

    for tick in range(ticks):
        y = (r, z, v, w)
        currents = p['stage_current'][:, tick*4:(tick+1)*4].astype(np.float64)
        k1, b1 = rhs(y, currents[:, 0])
        k2, b2 = rhs(tuple(x+.000125*k for x, k in zip(y, k1)), currents[:, 1])
        k3, b3 = rhs(tuple(x+.000125*k for x, k in zip(y, k2)), currents[:, 2])
        k4, b4 = rhs(tuple(x+.00025*k for x, k in zip(y, k3)), currents[:, 3])
        branch[:, tick] = (b1.astype(np.uint8) | (b2.astype(np.uint8)<<1)
                           | (b3.astype(np.uint8)<<2) | (b4.astype(np.uint8)<<3))
        r, z, v, w = (x+(.00025/6.)*(u1+2*u2+2*u3+u4)
                      for x, u1, u2, u3, u4 in zip(y, k1, k2, k3, k4))
        if (tick+1)%20 == 0:
            index = (tick+1)//20
            rate[:, index], adaptation[:, index], vr[:, index], wz[:, index] = r, z, v, w
    pooled = (rate.reshape(c*samples, n) @ p['pool'].T).reshape(c, samples, -1)
    pooled_v = (vr.reshape(c*samples, n) @ p['pool'].T).reshape(c, samples, -1)
    activation = np.empty((c, samples, p['pool'].shape[0]), np.float64)
    sensitivity = np.zeros_like(activation)
    activation[:, 0] = p['initial_activation']
    fraction = -math.expm1(-.005/.01)
    for index in range(1, samples):
        activation[:, index] = (activation[:, index-1]
                                + fraction*(pooled[:, index]-activation[:, index-1]))
        sensitivity[:, index] = (sensitivity[:, index-1]
                                  + fraction*(pooled_v[:, index]-sensitivity[:, index-1]))
    meter['synthetic_cpu_calls'] += 1
    meter['synthetic_cpu_context_ticks'] += c*ticks
    return {'rate': rate, 'adaptation': adaptation,
            'rate_sensitivity': vr, 'adaptation_sensitivity': wz,
            'activation': activation, 'activation_sensitivity': sensitivity,
            'command': p['rest']+.02*(activation @ p['moment'].T),
            'tendon': .2*np.tanh(activation[:, :, p['ltm']]/10.),
            'branch_bits': branch}


def cuda(problem, gain, meter):
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() >= 1, 'Actual CUDA required; no skip'
    result, timing = dense.solve_cuda(problem, gain)
    meter['actual_cuda'] = True
    meter['synthetic_cuda_calls'] += 1
    meter['synthetic_cuda_context_ticks'] += timing['context_rk4_ticks']
    return result


def assert_fields_close(actual, expected, atol=5e-10):
    for field in dense_result_fields():
        np.testing.assert_allclose(actual[field], expected[field], rtol=0, atol=atol)
    np.testing.assert_array_equal(actual['branch_bits'], expected['branch_bits'])


def dense_result_fields():
    return ('rate', 'adaptation', 'rate_sensitivity',
            'adaptation_sensitivity', 'activation',
            'activation_sensitivity', 'command', 'tendon')


def test_all_fields_branch_and_separate_stage_two_three(meter):
    problem = fixture()
    assert np.any(problem['stage_current'][:, 1::4] != problem['stage_current'][:, 2::4])
    setup = dense.cuda_warmup()
    meter['actual_cuda'] = True
    meter['synthetic_cuda_setup_calls'] += 1
    meter['synthetic_cuda_setup_context_ticks'] += setup['setup_context_rk4_ticks']
    assert setup['setup_context_rk4_ticks'] == 20
    gain = np.array([.3, 1.2, 2.7])
    expected = cpu_reference(problem, gain, meter)
    actual = cuda(problem, gain, meter)
    assert_fields_close(actual, expected)
    assert actual['branch_bits'].dtype == np.uint8
    assert actual['branch_bits'].shape == (2, 60, 4)
    assert np.any(actual['branch_bits'] != 0)
    assert np.any(actual['branch_bits'] != 15)
    runner.validate_saved_response(actual, dense.prepare_problem(problem))
    assert meter['synthetic_cpu_context_ticks'] == 120
    assert meter['synthetic_cuda_context_ticks'] == 120


def test_gain0_and_finite_difference_all_fields(meter):
    problem = fixture(contexts=1, controls=2)
    center_gain = np.array([1.1, .8, 1.7])
    center = cuda(problem, center_gain, meter)
    gain0 = cuda(problem, np.zeros(3), meter)
    expected0 = cpu_reference(problem, np.zeros(3), meter)
    assert_fields_close(gain0, expected0)
    h = 1e-6
    plus = cuda(problem, center_gain+h, meter)
    minus = cuda(problem, center_gain-h, meter)
    report, passed = runner.sensitivity_metrics(dense.prepare_problem(problem),
                                                center, plus, minus, h)
    assert passed, report
    assert set(report['branch_changes']) == {'E110'}
    assert meter['synthetic_cuda_context_ticks'] == 160


def test_constant_stage_continuous_reference(meter):
    problem = fixture(contexts=1, controls=2)
    problem['stage_current'][:] = 13.
    gain = np.array([.5, 1.4, 2.0])
    actual = cuda(problem, gain, meter)
    neuron = 0
    p = dense.prepare_problem(problem)
    own_gain = gain[p['membership'][neuron]]

    def ode(_, y):
        r, z, v, w = y
        tanh = math.tanh(p['a'][neuron]*(13.-p['threshold'][neuron]-z)/p['cap'][neuron])
        active = tanh > 0
        phi = p['cap'][neuron]*tanh if active else 0.
        prime = p['a'][neuron]*(1.-tanh*tanh) if active else 0.
        return [(phi-r)/p['tau'][neuron], (own_gain*r-z)/.3,
                (-prime*w-v)/p['tau'][neuron], (r+own_gain*v-w)/.3]

    t = p['time_s']
    truth = solve_ivp(ode, (0., t[-1]),
                      [p['initial_rate'][0, neuron],
                       p['initial_adaptation'][0, neuron], 0., 0.],
                      method='DOP853', t_eval=t, rtol=1e-12, atol=1e-13,
                      max_step=.000125)
    meter['synthetic_dop853_calls'] += 1
    meter['synthetic_dop853_function_evaluations'] += truth.nfev
    meter['synthetic_dop853_integrated_s'] += float(t[-1])
    assert truth.success
    for index, field in enumerate(dense_result_fields()[:4]):
        np.testing.assert_allclose(actual[field][0, :, neuron], truth.y[index],
                                   rtol=0, atol=2e-9)


def test_validation_ownership_private_stream_and_failure_progress(meter,
                                                                  monkeypatch):
    import cupy as cp
    source = fixture(contexts=1, controls=1)
    frozen = dense.prepare_problem(source)
    assert all(not np.shares_memory(frozen[key], np.asarray(source[key])) for key in frozen)
    engine = dense.CudaDenseAdaptiveEngine(frozen)
    assert engine.metadata()['fixed_upload_count'] == len(engine.FIXED_KEYS) == 13
    assert engine.metadata()['fixed_upload_bytes'] == sum(
        engine.problem[key].nbytes for key in engine.FIXED_KEYS)
    expected = cpu_reference(source, np.ones(3), meter)
    source['stage_current'][:] = 0.
    source['pool'][:] = 0.
    frozen['stage_current'][:] = 0.
    frozen['pool'][:] = 0.
    with cp.cuda.Stream(non_blocking=True):
        first, timing = engine.solve(np.ones(3))
    meter['actual_cuda'] = True
    meter['synthetic_cuda_calls'] += 1
    meter['synthetic_cuda_context_ticks'] += timing['context_rk4_ticks']
    assert_fields_close(first, expected)
    assert engine.metadata()['gain_upload_count_completed'] == 1
    assert engine.metadata()['gain_upload_bytes_completed'] == 4*8
    first['rate'][:] = -999.
    second, second_timing = engine.solve(np.ones(3))
    meter['synthetic_cuda_calls'] += 1
    meter['synthetic_cuda_context_ticks'] += second_timing['context_rk4_ticks']
    assert_fields_close(second, expected)
    assert engine.metadata()['fixed_upload_count'] == 13

    def fail_decoder(_):
        raise RuntimeError('synthetic decoder failure')
    monkeypatch.setattr(engine, '_gpu_decode', fail_decoder)
    with pytest.raises(RuntimeError, match='synthetic decoder failure') as info:
        engine.solve(np.ones(3))
    progress = info.value.response_progress
    assert progress['completion_class'] == 'known_complete'
    assert info.value.completed_context_rk4_ticks == 20
    assert info.value.completed_conditional_model_s == pytest.approx(.005)
    assert progress['decoder_completion_class'] == 'unknown'
    assert 'branch_bits' in info.value.partial_response
    meter['synthetic_cuda_known_failure_ticks'] += 20
    assert engine.metadata()['solve_calls'] == 3


def test_bounded_shape_nonfinite_membership_and_gain_rejections(meter):
    source = fixture()
    for key, value in (
        ('stage_current', np.zeros((4, 80, 1), np.float32)),
        ('stage_current', np.zeros((1, 160080, 1), np.float32)),
        ('stage_current', np.zeros((1, 81, 1), np.float32)),
        ('stage_current', source['stage_current'].astype(np.float64)),
        ('stage_current', np.full_like(source['stage_current'], np.nan)),
        ('time_s', np.array([0., .006])),
        ('membership', np.array([0, 0, 1, 2])),
        ('motor_ids', np.array([2, 7, 7, 11])),
        ('tau', np.array([.02, 0., .03, .02])),
        ('initial_adaptation', -np.ones((2, 4))),
        ('initial_activation', -np.ones((2, 3)))):
        damaged = dict(source)
        damaged[key] = value
        with pytest.raises(ValueError):
            dense.prepare_problem(damaged)
    p = dense.prepare_problem(source)
    for gain in ([-.01, 0., 0.], [0., 9., 0.], [0., np.nan, 0.],
                 [0., 1.], [0., 1., 2., 3.]):
        with pytest.raises(ValueError):
            dense.checked_gain(p, gain)
    assert meter['synthetic_cuda_context_ticks'] == 0


def test_late_decoder_error_preserves_available_partial_arrays(meter, monkeypatch):
    import cupy as cp
    engine = dense.CudaDenseAdaptiveEngine(fixture(contexts=1, controls=1))

    def fail_tendon(_):
        raise RuntimeError('synthetic late tendon failure')

    monkeypatch.setattr(cp, 'tanh', fail_tendon)
    with pytest.raises(RuntimeError, match='synthetic late tendon failure') as info:
        engine.solve(np.ones(3))
    error = info.value
    assert error.completed_context_rk4_ticks == 20
    assert error.response_progress['decoder_completion_class'] == 'unknown'
    assert {'rate', 'adaptation', 'rate_sensitivity',
            'adaptation_sensitivity', 'branch_bits', 'activation',
            'activation_sensitivity', 'command'} <= set(error.partial_response)
    assert 'tendon' not in error.partial_response
    for field in error.partial_response.values():
        assert np.isfinite(field).all()
    meter['actual_cuda'] = True
    meter['synthetic_cuda_known_failure_ticks'] += 20


def test_enqueue_then_raise_charges_unknown_neural_ticks(meter, monkeypatch):
    engine = dense.CudaDenseAdaptiveEngine(fixture(contexts=1, controls=1))
    original = dense._kernel()

    def enqueue_then_raise(*args, **kwargs):
        original(*args, **kwargs)  # Real tiny CUDA kernel is enqueued.
        raise RuntimeError('synthetic post-enqueue wrapper failure')

    monkeypatch.setattr(dense, '_kernel', lambda: enqueue_then_raise)
    with pytest.raises(RuntimeError, match='synthetic post-enqueue') as info:
        engine.solve(np.ones(3))
    error = info.value
    assert error.response_progress['stage'] == 'launch_requested'
    assert error.response_progress['completion_class'] == 'unknown'
    assert not hasattr(error, 'completed_context_rk4_ticks')
    ledger = runner._initial_ledger('synthetic-freeze', 'synthetic-review')
    attempt = {}
    runner.account_failed_response(ledger, attempt, error, 20, .005, 1)
    assert ledger['unknown_requested_context_rk4_ticks'] == 20
    assert ledger['unknown_requested_conditional_model_s'] == .005
    assert ledger['gain_upload_count_synchronized'] == 1
    assert ledger['gain_upload_count_unknown_requested'] == 0
    assert attempt['completion_class'] == 'unknown'
    meter['actual_cuda'] = True
    meter['synthetic_cuda_unknown_requested_ticks'] += 20


def test_constructor_and_gain_upload_failure_accounting(meter, monkeypatch):
    import cupy as cp
    original_asarray = cp.asarray
    calls = 0

    def third_upload_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('synthetic fixed upload failure')
        return original_asarray(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(cp, 'asarray', third_upload_fails)
        with pytest.raises(RuntimeError, match='synthetic fixed upload') as info:
            dense.CudaDenseAdaptiveEngine(fixture(contexts=1, controls=1))
    error = info.value
    assert error.fixed_upload_count_requested == 3
    assert error.fixed_upload_count_enqueued == 2
    assert error.fixed_upload_count_synchronized == 0
    assert error.fixed_upload_count_unknown_requested == 3
    ledger = runner._initial_ledger('synthetic-freeze', 'synthetic-review')
    progress = runner.account_fixed_upload_failure(ledger, error, 'main_fixed_upload')
    assert ledger['main_fixed_upload_status'] == 'failed'
    assert ledger['main_fixed_upload'] == progress
    assert progress['fixed_upload_count_unknown_requested'] == 3
    assert progress['fixed_upload_bytes_requested'] > progress['fixed_upload_bytes_enqueued']

    engine = dense.CudaDenseAdaptiveEngine(fixture(contexts=1, controls=1))
    assert engine.metadata()['fixed_upload_count_requested'] == 13
    assert engine.metadata()['fixed_upload_count_enqueued'] == 13
    assert engine.metadata()['fixed_upload_count_synchronized'] == 13
    assert engine.metadata()['fixed_upload_count_unknown_requested'] == 0

    def gain_upload_fails(*_args, **_kwargs):
        raise RuntimeError('synthetic gain upload failure')

    with monkeypatch.context() as patcher:
        patcher.setattr(cp, 'asarray', gain_upload_fails)
        with pytest.raises(RuntimeError, match='synthetic gain upload') as info:
            engine.solve(np.ones(3))
    error = info.value
    gain_progress = error.gain_upload_progress
    assert gain_progress['gain_upload_count_requested'] == 1
    assert gain_progress['gain_upload_count_enqueued'] == 0
    assert gain_progress['gain_upload_count_synchronized'] == 0
    assert gain_progress['gain_upload_count_unknown_requested'] == 1
    assert error.response_progress['completion_class'] == 'not_enqueued'
    assert engine.metadata()['gain_upload_count_unknown_requested'] == 1
    ledger = runner._initial_ledger('synthetic-freeze', 'synthetic-review')
    runner.account_failed_response(ledger, {}, error, 20, .005, 1)
    assert ledger['gain_upload_count_unknown_requested'] == 1
    assert ledger['unknown_requested_context_rk4_ticks'] == 0
    meter['actual_cuda'] = True


def test_all_context_actual_gain0_center_startup_diagnostics(tmp_path, meter,
                                                              monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    target_q = np.zeros((99, 42), np.float64)
    target_ltm = np.zeros((99, 6), np.float64)
    np.savez(tmp_path/'original.npz', startup_target_command=target_q,
             startup_target_tendon=target_ltm)
    freeze = {'original_fixture_path': 'original.npz', 'contexts': []}
    gain0 = {'command': np.zeros((3, 100, 42), np.float64),
             'tendon': np.zeros((3, 100, 6), np.float64)}
    center = {'command': np.zeros((3, 100, 42), np.float64),
              'tendon': np.zeros((3, 100, 6), np.float64)}
    for index, name in enumerate(runner.CONTEXT_NAMES):
        file_name = name+'.npz'
        actual_q = np.zeros((2000, 42), np.float64)
        actual_tendon = np.zeros((2000, 6), np.float64)
        if name == 'E110':
            actual_q[0, 0] = .06  # Preserve an actual native startup failure.
            gain0['command'][index, 1, 0] = .07
        if name == 'E113':
            center['tendon'][index, 1, 0] = .04
        np.savez(tmp_path/file_name, command=actual_q, tendon=actual_tendon)
        freeze['contexts'].append({'name': name, 'paths': {'endpoints': file_name}})
    report = runner.startup_target_metrics(gain0, center, freeze)
    assert set(report) == set(runner.CONTEXT_NAMES)
    for name in runner.CONTEXT_NAMES:
        assert set(report[name]) == {'actual', 'gain0', 'center'}
        for label in ('actual', 'gain0', 'center'):
            assert report[name][label]['command']['rms_limit'] == .02
            assert report[name][label]['command']['max_limit'] == .05
            assert report[name][label]['tendon']['rms_limit'] == .01
            assert report[name][label]['tendon']['max_limit'] == .03
    assert not report['E110']['actual']['command']['passed']
    assert not report['E110']['gain0']['command']['passed']
    assert not report['E113']['center']['tendon']['passed']
    assert report['E121']['actual']['command']['passed']
    assert meter['synthetic_cuda_context_ticks'] == 0


def test_warmup_post_response_error_preserves_complete_upload_accounting(meter,
                                                                          monkeypatch):
    import cupy as cp

    def fail_device_properties(_device):
        raise RuntimeError('synthetic device-description failure after warmup')

    monkeypatch.setattr(cp.cuda.runtime, 'getDeviceProperties', fail_device_properties)
    with pytest.raises(RuntimeError, match='device-description failure') as info:
        dense.cuda_warmup()
    error = info.value
    assert error.completed_context_rk4_ticks == 20
    assert error.fixed_upload_count_requested == 13
    assert error.fixed_upload_count_enqueued == 13
    assert error.fixed_upload_count_synchronized == 13
    assert error.fixed_upload_count_unknown_requested == 0
    assert error.gain_upload_count_requested == 1
    assert error.gain_upload_count_synchronized == 1
    assert error.gain_upload_count_unknown_requested == 0
    assert error.gain_upload_bytes_synchronized == 8
    assert len(error.partial_response) == 9
    assert error.response_progress['decoder_completion_class'] == 'known_complete'
    ledger = runner._initial_ledger('synthetic', 'synthetic')
    fixed = runner.account_fixed_upload_failure(ledger, error, 'cuda_setup_fixed_upload')
    assert fixed['fixed_upload_count_synchronized'] == 13
    meter['actual_cuda'] = True
    meter['synthetic_cuda_setup_calls'] += 1
    meter['synthetic_cuda_setup_context_ticks'] += 20
