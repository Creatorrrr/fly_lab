"""E134 generated-only mixed-muscle CUDA, analytic, and failure-accounting tests."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tools import banc_m3_adaptation_timescale as candidate
from tools import banc_m3_joint_adaptive_recruitment as old
from tools import verify_c_banc_m3_adaptation_timescale as verifier


def evidence_dir(tmp_path):
    location = os.environ.get('E134_SYNTHETIC_EVIDENCE_DIR')
    out = Path(location) if location else tmp_path
    out.mkdir(parents=True, exist_ok=True)
    return out


def generated_problem():
    # Two contexts; stage two and three differ. Two MNs share muscle zero.
    stages = 160
    x = np.arange(stages)
    current = np.empty((2, stages, 3), dtype=np.float32)
    source = np.empty((2, stages, 2), dtype=np.float32)
    for c in range(2):
        current[c, :, 0] = 7.5+1.8*np.sin(.19*x+.2*c)
        current[c, :, 1] = 8.3+2.1*np.cos(.23*x+.4*c)
        current[c, :, 2] = 7.8+1.4*np.sin(.31*x+.3*c)
        source[c, :, 0] = .6+.2*np.sin(.17*x+.1*c)
        source[c, :, 1] = .5+.2*np.cos(.27*x+.2*c)
    pool = np.array([[1.1, .8, 0.], [0., 0., 1.3]])
    moment = np.zeros((42, 2))
    moment[0, 0] = .13
    moment[7, 1] = -.09
    moment[18] = [.07, .11]
    union = np.array([[.9, -.7, 1.2, -.5, .8],
                      [1.1, -.8, 1.3, -.6, .7]], np.float32)
    return {'stage_current': current, 'source_rate': source,
        'time_s': np.array([0., .005, .01]),
        'initial_rate': np.array([[.2, .4, .1], [.3, .1, .2]]),
        'initial_adaptation': np.array([[.15, .07, .2], [.09, .11, .14]]),
        'initial_activation': np.array([[.3, .1], [.2, .4]]),
        'tau': np.array([.02, .014, .025]),
        'a': np.array([1., 1.2, .9]),
        'threshold': np.array([7.5, 8., 7.8]),
        'cap': np.array([200., 180., 220.]),
        'pool': pool, 'moment': moment,
        'rest': np.linspace(-.1, .2, 42), 'ltm': np.array([0, 1]),
        'membership': np.array([0, 0, 1]),
        'motor_ids': np.array([1001, 1002, 1003]),
        'source_ids': np.array([2001, 2002]),
        'union_edge_ids': np.array([3001, 3002, 3003, 3004, 3005]),
        'union_rows': np.array([0, 0, 1, 2, 2]),
        'union_cols': np.array([0, 1, 1, 0, 1]),
        'union_values': union,
        'fit_union_positions': np.arange(5),
        'edge_ids': np.array([3001, 3002, 3003, 3004, 3005]),
        'edge_rows': np.array([0, 0, 1, 2, 2]),
        'edge_cols': np.array([0, 1, 1, 0, 1]),
        'base_fit_weights': union[-1].astype(np.float64)}


def cpu_reference(mapping, delta, gain, ell):
    """Independent small loop over four source stages and six state equations."""
    p = old.prepare_joint_problem(mapping)
    current, source = p['stage_current'], p['source_rate']
    c_count, stages, n_count = current.shape
    ticks = stages//4
    samples = ticks//20+1
    m_count = len(gain)
    T = .3*np.exp(ell)
    stage_prime = np.empty((c_count, stages, n_count))
    branch = np.empty((c_count, ticks, n_count), np.uint8)
    output = {key: np.empty((c_count, samples, n_count))
              for key in candidate.FORWARD_FIELDS}
    for c in range(c_count):
        for n in range(n_count):
            state = np.array([p['initial_rate'][c, n],
                p['initial_adaptation'][c, n], 0., 0., 0., 0.])
            for index, key in enumerate(candidate.FORWARD_FIELDS):
                output[key][c, 0, n] = state[index]
            muscle = p['membership'][n]
            for tick in range(ticks):
                k = np.empty((4, 6))
                bits = 0
                for stage in range(4):
                    s = tick*4+stage
                    I = float(current[c, s, n])
                    for j in range(p['union_ptr'][n], p['union_ptr'][n+1]):
                        presyn = float(source[c, s, p['union_cols'][j]])
                        I += presyn*p['union_diff'][c, j]
                        if p['union_fit'][j] >= 0:
                            I += (presyn*p['union_base'][j]
                                  *np.expm1(delta[p['union_fit'][j]]))
                    y = state if stage == 0 else state + (
                        .00025 if stage == 3 else .000125)*k[stage-1]
                    r, z, v, w, u, h = y
                    tanh = np.tanh(p['a'][n]*(I-p['threshold'][n]-z)/p['cap'][n])
                    active = tanh > 0
                    phi = p['cap'][n]*tanh if active else 0.
                    prime = p['a'][n]*(1-tanh*tanh) if active else 0.
                    bits |= int(active)<<stage
                    stage_prime[c, s, n] = prime
                    g = gain[muscle]
                    membrane = p['tau'][n]
                    adapt = T[muscle]
                    k[stage] = [(phi-r)/membrane, (g*r-z)/adapt,
                        (-prime*w-v)/membrane, (r+g*v-w)/adapt,
                        (-prime*h-u)/membrane,
                        (g*u-h)/adapt-(g*r-z)/adapt]
                branch[c, tick, n] = bits
                state = state + .00025*(k[0]+2*k[1]+2*k[2]+k[3])/6
                if (tick+1)%20 == 0:
                    for index, key in enumerate(candidate.FORWARD_FIELDS):
                        output[key][c, (tick+1)//20, n] = state[index]
    fraction = -math.expm1(-.005/.01)
    activation = np.empty((c_count, samples, m_count))
    activation_v = np.zeros_like(activation)
    activation_u = np.zeros_like(activation)
    activation[:, 0] = p['initial_activation']
    for sample in range(1, samples):
        for key, result in (('rate', activation),
                            ('rate_sensitivity', activation_v),
                            ('rate_log_tau_sensitivity', activation_u)):
            drive = output[key][:, sample]@p['pool'].T
            result[:, sample] = result[:, sample-1] + fraction*(
                drive-result[:, sample-1])
    output.update(activation=activation,
        activation_sensitivity=activation_v,
        activation_log_tau_sensitivity=activation_u,
        command=p['rest']+.02*activation@p['moment'].T,
        tendon=.2*np.tanh(activation[:, :, p['ltm']]/10),
        branch_bits=branch)
    e_count = len(delta)
    B = np.zeros((c_count, samples, e_count+2*m_count))
    for c in range(c_count):
        for e in range(e_count):
            n = p['edge_rows'][e]
            m = p['edge_muscles'][e]
            factor = p['pool'][m, n]
            weight = p['base_fit_weights'][e]*np.exp(delta[e])
            state = np.zeros(3)  # edge p, q and filtered activation
            for tick in range(ticks):
                k = np.empty((4, 2))
                for stage in range(4):
                    s = tick*4+stage
                    local = state[:2] if stage == 0 else state[:2]+(
                        .00025 if stage == 3 else .000125)*k[stage-1]
                    rp, zq = local
                    drive = weight*float(source[c, s, p['edge_cols'][e]])
                    prime = stage_prime[c, s, n]
                    k[stage] = [(prime*(drive-zq)-rp)/p['tau'][n],
                        (gain[m]*rp-zq)/T[m]]
                state[:2] += .00025*(k[0]+2*k[1]+2*k[2]+k[3])/6
                if (tick+1)%20 == 0:
                    state[2] += fraction*(factor*state[0]-state[2])
                    B[c, (tick+1)//20, e] = state[2]
    B[:, :, e_count:e_count+m_count] = activation_v
    B[:, :, e_count+m_count:] = activation_u
    output['B'] = B
    return output


def require_cuda():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('Actual CUDA unavailable')
    return cp


def logged_call(out, label, engine, delta, gain, ell, *, old_engine=False):
    path = out/'synthetic-invocations.json'
    ledger = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'schema': 'flylab.m3.e134.synthetic.v1', 'generated_only': True,
        'calls': []}
    row = {'label': label, 'status': 'requested',
        'requested_context_ticks': engine.contexts*engine.ticks,
        'requested_conditional_s': engine.contexts*engine.ticks*.00025,
        'requested_bank_edge_stage_ticks': engine.contexts*engine.ticks*4*engine.edges}
    ledger['calls'].append(row)
    path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    began = time.perf_counter()
    raw = None
    try:
        if old_engine:
            device, raw, timing = engine.solve(delta, gain, derivative=True)
        else:
            device, raw, timing = engine.solve(delta, gain, ell, derivative=True)
        raw['B'] = engine.host_copy(device['B'], label='synthetic_full_B')
        raw_path = out/(label+'-raw.npz')
        np.savez(raw_path, **raw)
        row.update(status='raw_saved', raw_path=str(raw_path),
            raw_sha256=verifier.sha256(raw_path), progress=timing['progress'])
        return raw
    except BaseException as error:
        row.update(status='failed', error=repr(error),
            progress=getattr(error, 'response_progress', None),
            partial_keys=sorted(getattr(error, 'partial_response', {}) or {}),
            secondary_errors=getattr(error, 'secondary_copy_errors', None))
        if raw:
            partial_path = out/(label+'-partial.npz')
            np.savez(partial_path, **raw)
            row['partial_path'] = str(partial_path)
        raise
    finally:
        row['wall_s'] = time.perf_counter()-began
        row['engine_metadata'] = engine.metadata()
        path.write_text(json.dumps(verifier.safe(ledger), indent=2), encoding='utf-8')


def test_generated_mixed_mapping_all_sensitivities_and_ell0_parity(tmp_path):
    require_cuda()
    out = evidence_dir(tmp_path)
    mapping = generated_problem()
    old_engine = old.CudaJointAdaptiveEngine(mapping)
    engine = candidate.CudaAdaptationTimescaleEngine(mapping)
    delta = np.array([.12, -.09, .08, -.14, .11])
    gain = np.array([.8, 2.2])
    ell = np.log(np.array([.14, 1.2])/.3)
    old_raw = logged_call(out, 'baseline', old_engine, delta, gain, np.zeros(2),
                          old_engine=True)
    zero_raw = logged_call(out, 'zero_tau', engine, delta, gain, np.zeros(2))
    center = logged_call(out, 'center', engine, delta, gain, ell)
    reference = cpu_reference(mapping, delta, gain, ell)
    comparisons = {'old_zero': {}, 'center_cpu': {}, 'finite_difference': {}}
    for key in (*old.FIELDS, 'B'):
        a = zero_raw[key][:, :, :7] if key == 'B' else zero_raw[key]
        comparisons['old_zero'][key] = float(np.max(np.abs(a-old_raw[key])))
    comparisons['old_zero']['branch_exact'] = bool(np.array_equal(
        zero_raw['branch_bits'], old_raw['branch_bits']))
    for key in (*candidate.FIELDS, 'B'):
        comparisons['center_cpu'][key] = float(np.max(np.abs(center[key]-reference[key])))
    comparisons['center_cpu']['branch_exact'] = bool(np.array_equal(
        center['branch_bits'], reference['branch_bits']))
    h = 1e-6
    for kind in ('tau', 'gain', 'edge'):
        if kind == 'tau':
            plus = cpu_reference(mapping, delta, gain, ell+h)
            minus = cpu_reference(mapping, delta, gain, ell-h)
            analytic = center['activation_log_tau_sensitivity']
        elif kind == 'gain':
            plus = cpu_reference(mapping, delta, gain+h, ell)
            minus = cpu_reference(mapping, delta, gain-h, ell)
            analytic = center['activation_sensitivity']
        else:
            direction = np.array([.3, -.7, .9, -.2, .5])
            plus = cpu_reference(mapping, delta+h*direction, gain, ell)
            minus = cpu_reference(mapping, delta-h*direction, gain, ell)
            analytic = np.zeros_like(center['activation'])
            muscles = engine.problem['edge_muscles']
            for i, muscle in enumerate(muscles):
                analytic[:, :, muscle] += center['B'][:, :, i]*direction[i]
        fd = (plus['activation']-minus['activation'])/(2*h)
        comparisons['finite_difference'][kind] = float(np.max(np.abs(fd-analytic)))
    (out/'synthetic-numeric-comparison.json').write_text(json.dumps(
        verifier.safe(comparisons), indent=2), encoding='utf-8')
    assert np.any(mapping['stage_current'][:, 1::4] != mapping['stage_current'][:, 2::4])
    assert np.any(center['branch_bits'] == 0)
    assert np.any(center['branch_bits'] != 0)
    assert comparisons['old_zero']['branch_exact']
    assert max(v for k, v in comparisons['old_zero'].items() if k != 'branch_exact') < 1e-8
    assert comparisons['center_cpu']['branch_exact']
    assert max(v for k, v in comparisons['center_cpu'].items() if k != 'branch_exact') < 1e-8
    assert max(comparisons['finite_difference'].values()) < 2e-5
    assert np.array_equal(candidate.activation_parameter_map(engine.problem),
        np.r_[engine.problem['edge_muscles'], np.arange(2), np.arange(2)])


def test_input_validation_and_owned_problem(tmp_path):
    require_cuda()
    out = evidence_dir(tmp_path)
    mapping = generated_problem()
    engine = candidate.CudaAdaptationTimescaleEngine(mapping)
    before = engine.problem['stage_current'].copy()
    mapping['stage_current'][:] = 0
    assert np.array_equal(engine.problem['stage_current'], before)
    delta, gain = np.zeros(5), np.ones(2)
    for label, bad in (('shape', np.zeros(1)),
                       ('nan', np.array([0., np.nan])),
                       ('low', np.array([candidate.LOG_TAU_LOWER-1e-6, 0.])),
                       ('high', np.array([0., candidate.LOG_TAU_UPPER+1e-6]))):
        count = engine.solve_calls
        row = {'label': label, 'status': 'requested', 'solve_calls_before': count}
        path = out/f'synthetic-invalid-{label}.json'
        path.write_text(json.dumps(row), encoding='utf-8')
        began = time.perf_counter()
        with pytest.raises(ValueError):
            engine.solve(delta, gain, bad, derivative=True)
        row.update(status='rejected_before_solve', wall_s=time.perf_counter()-began,
                   solve_calls_after=engine.solve_calls)
        path.write_text(json.dumps(row, indent=2), encoding='utf-8')
        assert engine.solve_calls == count
    frozen_bounds = json.loads((verifier.BASE/'e134-adaptation-timescale-freeze.json')
        .read_text(encoding='utf-8'))['log_tau_bounds']
    bounds_path = out/'synthetic-exact-boundary-preflight.json'
    bounds_path.write_text(json.dumps({'status': 'requested',
        'frozen_bounds': frozen_bounds}), encoding='utf-8')
    accepted = []
    rejected = []
    for bound in (candidate.LOG_TAU_LOWER, candidate.LOG_TAU_UPPER):
        vector = np.array([bound, 0.])
        checked = engine._check_timescale_vectors(delta, gain, vector)[2]
        accepted.append(bool(np.array_equal(checked, vector)))
    for outside in (np.nextafter(candidate.LOG_TAU_LOWER, -np.inf),
                    np.nextafter(candidate.LOG_TAU_UPPER, np.inf)):
        with pytest.raises(ValueError):
            engine._check_timescale_vectors(delta, gain, np.array([outside, 0.]))
        rejected.append(True)
    bounds_path.write_text(json.dumps({'status': 'completed',
        'frozen_bounds': frozen_bounds,
        'source_bounds': [candidate.LOG_TAU_LOWER, candidate.LOG_TAU_UPPER],
        'exact_endpoints_accepted': accepted, 'nextafter_outside_rejected': rejected,
        'solve_calls': engine.solve_calls}, indent=2), encoding='utf-8')
    assert frozen_bounds == [candidate.LOG_TAU_LOWER, candidate.LOG_TAU_UPPER]
    assert all(accepted) and all(rejected)


@pytest.mark.parametrize('kernel_name', ['prepare_vectors', 'timescale_forward',
    'timescale_muscle_ema', 'timescale_edge_bank'])
def test_cuda_fault_stage_accounting_and_primary(monkeypatch, tmp_path, kernel_name):
    require_cuda()
    out = evidence_dir(tmp_path)
    engine = candidate.CudaAdaptationTimescaleEngine(generated_problem())
    original = candidate._kernel
    primary = RuntimeError('synthetic '+kernel_name+' primary')
    def factory(name):
        if name == kernel_name:
            def fail(*_args, **_kwargs):
                raise primary
            return fail
        return original(name)
    monkeypatch.setattr(candidate, '_kernel', factory)
    row = {'kernel': kernel_name, 'status': 'requested'}
    path = out/f'synthetic-fault-{kernel_name}.json'
    path.write_text(json.dumps(row), encoding='utf-8')
    began = time.perf_counter()
    with pytest.raises(RuntimeError) as caught:
        engine.solve(np.zeros(5), np.ones(2), np.zeros(2), derivative=True)
    error = caught.value
    row.update(status='expected_failure', wall_s=time.perf_counter()-began,
        primary=repr(error), progress=error.response_progress,
        partial_keys=sorted(error.partial_response),
        secondary=error.secondary_copy_errors, metadata=engine.metadata())
    path.write_text(json.dumps(verifier.safe(row), indent=2), encoding='utf-8')
    assert error is primary
    events = error.response_progress['kernel_events']
    failed = next(event for event in events if event['name'] == {
        'timescale_forward': 'forward',
        'timescale_muscle_ema': 'decoder',
        'timescale_edge_bank': 'bank',
        'prepare_vectors': 'prepare'}[kernel_name])
    assert failed['requested'] and failed['unknown_requested']
    assert not failed['synchronized']


def test_host_and_raw_save_failure_keep_primary_and_partial(monkeypatch, tmp_path):
    require_cuda()
    out = evidence_dir(tmp_path)
    engine = candidate.CudaAdaptationTimescaleEngine(generated_problem())
    original_copy = engine.host_copy
    host_primary = RuntimeError('synthetic host primary')
    def bad_host(value, *, progress=None, label='device_output'):
        if label == 'rate':
            raise host_primary
        if label == 'rescue_rate':
            raise RuntimeError('secondary rescue_rate')
        return original_copy(value, progress=progress, label=label)
    monkeypatch.setattr(engine, 'host_copy', bad_host)
    row = {'stage': 'host', 'status': 'requested'}
    path = out/'synthetic-fault-host.json'
    path.write_text(json.dumps(row), encoding='utf-8')
    began = time.perf_counter()
    with pytest.raises(RuntimeError) as caught:
        engine.solve(np.zeros(5), np.ones(2), np.zeros(2), derivative=False)
    error = caught.value
    row.update(status='expected_failure', wall_s=time.perf_counter()-began,
        primary=repr(error), secondary=error.secondary_copy_errors,
        partial_keys=sorted(error.partial_response), progress=error.response_progress)
    path.write_text(json.dumps(verifier.safe(row), indent=2), encoding='utf-8')
    assert error is host_primary
    assert any(item['buffer'] == 'rate' for item in error.secondary_copy_errors)
    assert 'adaptation_log_tau_sensitivity' in error.partial_response
    monkeypatch.setattr(engine, 'host_copy', original_copy)

    original_savez = verifier.np.savez
    raw_primary = OSError('synthetic raw save primary')
    def bad_savez(path, **arrays):
        if Path(path).name == 'generated-raw.npz':
            raise raw_primary
        return original_savez(path, **arrays)
    monkeypatch.setattr(verifier.np, 'savez', bad_savez)
    batch_row = {'name': 'generated_raw_failure'}
    ledger = {'original_batches': [batch_row]}
    raw_path = out/'generated-raw.npz'
    (out/'synthetic-fault-raw-before.json').write_text(json.dumps(
        {'status': 'requested'}), encoding='utf-8')
    began = time.perf_counter()
    with pytest.raises(OSError) as caught:
        verifier._execute(engine, np.zeros(5), np.ones(2), np.zeros(2),
            True, raw_path, batch_row, ledger, out)
    batch_row['wall_s_external_test'] = time.perf_counter()-began
    (out/'synthetic-fault-raw-after.json').write_text(json.dumps(
        verifier.safe(batch_row), indent=2), encoding='utf-8')
    assert caught.value is raw_primary
    assert batch_row['status'] == 'failed'
    assert batch_row['partial_raw_files']


def test_upload_request_and_forward_sync_uncertainty(monkeypatch, tmp_path):
    cp = require_cuda()
    out = evidence_dir(tmp_path)
    engine = candidate.CudaAdaptationTimescaleEngine(generated_problem())
    original_asarray = cp.asarray
    upload_primary = RuntimeError('synthetic first upload primary')
    def fail_first_upload(*_args, **_kwargs):
        raise upload_primary
    row = {'stage': 'upload', 'status': 'requested'}
    path = out/'synthetic-fault-upload.json'
    path.write_text(json.dumps(row), encoding='utf-8')
    monkeypatch.setattr(cp, 'asarray', fail_first_upload)
    began = time.perf_counter()
    with pytest.raises(RuntimeError) as caught:
        engine.solve(np.zeros(5), np.ones(2), np.zeros(2), derivative=True)
    monkeypatch.setattr(cp, 'asarray', original_asarray)
    error = caught.value
    row.update(status='expected_failure', wall_s=time.perf_counter()-began,
        primary=repr(error), progress=error.response_progress,
        metadata=engine.metadata())
    path.write_text(json.dumps(verifier.safe(row), indent=2), encoding='utf-8')
    assert error is upload_primary
    assert error.response_progress['dynamic_upload_requested']['count'] == 1
    assert error.response_progress['dynamic_upload_synchronized']['count'] == 0
    assert error.response_progress['dynamic_upload_unknown_requested']['count'] == 1
    assert error.response_progress['kernel_events'] == []

    forward_primary = RuntimeError('synthetic forward sync primary')
    actual_stream = engine.stream
    class StreamProxy:
        calls = 0
        def __enter__(self):
            return actual_stream.__enter__()
        def __exit__(self, exc_type, exc, tb):
            return actual_stream.__exit__(exc_type, exc, tb)
        def synchronize(self):
            self.calls += 1
            if self.calls == 3:
                raise forward_primary
            return actual_stream.synchronize()
    proxy = StreamProxy()
    engine.stream = proxy
    row = {'stage': 'forward_sync', 'status': 'requested'}
    path = out/'synthetic-fault-forward-sync.json'
    path.write_text(json.dumps(row), encoding='utf-8')
    began = time.perf_counter()
    with pytest.raises(RuntimeError) as caught:
        engine.solve(np.zeros(5), np.ones(2), np.zeros(2), derivative=True)
    engine.stream = actual_stream
    error = caught.value
    row.update(status='expected_failure', wall_s=time.perf_counter()-began,
        primary=repr(error), progress=error.response_progress,
        partial_keys=sorted(error.partial_response),
        secondary=error.secondary_copy_errors)
    path.write_text(json.dumps(verifier.safe(row), indent=2), encoding='utf-8')
    assert error is forward_primary
    progress = error.response_progress
    forward = next(x for x in progress['kernel_events'] if x['name'] == 'forward')
    assert forward['enqueued'] and forward['unknown_requested']
    assert progress['forward_completion_class'] == 'unknown'
    assert progress['completed_context_rk4_ticks'] is None


def test_verifier_requires_exact_parent_review_without_original_run(tmp_path):
    out = evidence_dir(tmp_path)
    with pytest.raises(ValueError, match='Exact E134 frozen SHA'):
        verifier.reviewed_inputs(out/'absent-freeze.json', 'bad',
            out/'absent-review.json', 'bad')
    report = {'status': 'rejected_before_original', 'requested_original_batches': 0}
    (out/'synthetic-verifier-guard.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')


def test_small_branch_stage_counts_and_pair_accounting(tmp_path):
    out = evidence_dir(tmp_path)
    center = np.zeros((2, 3, 2), np.uint8)
    plus = center.copy()
    plus[0, 0, 0] = 1
    plus[0, 0, 1] = 2
    plus[0, 2, 1] = 5
    plus[1, 1, 0] = 8
    minus = center.copy()
    minus[0, 0, 0] = 1
    minus[1, 1, 1] = 4
    path = out/'synthetic-branch-stage-counts.json'
    path.write_text(json.dumps({'status': 'requested', 'shape': list(center.shape)}),
        encoding='utf-8')
    result = {'plus_vs_center': verifier._branch_change_counts(center, plus),
        'minus_vs_center': verifier._branch_change_counts(center, minus),
        'plus_vs_minus': verifier._branch_change_counts(plus, minus)}
    path.write_text(json.dumps(verifier.safe(result), indent=2), encoding='utf-8')
    assert result['plus_vs_center']['count'] == 4
    assert result['plus_vs_center']['first_flat_index'] == 0
    assert result['plus_vs_center']['per_context'] == [
        {'context_index': 0, 'changed_ticks': 2, 'changed_neuron_ticks': 3,
         'changed_stage_bits': [2, 1, 1, 0]},
        {'context_index': 1, 'changed_ticks': 1, 'changed_neuron_ticks': 1,
         'changed_stage_bits': [0, 0, 0, 1]}]
    assert result['minus_vs_center']['count'] == 2
    assert result['plus_vs_minus']['count'] == 4
    assert result['plus_vs_minus']['per_context'][0]['changed_stage_bits'] == [1, 1, 1, 0]
    assert result['plus_vs_minus']['per_context'][1]['changed_stage_bits'] == [0, 0, 1, 1]


class _ReturnedTinyEngine:
    contexts, ticks, neurons, samples, edges = 1, 20, 1, 2, 1

    def __init__(self):
        self.problem = {'edge_ids': np.array([1]), 'motor_indices': np.array([0]),
            'muscle_indices': np.array([0]), 'pool': np.ones((1, 1)),
            'edge_muscles': np.array([0])}
        self.solve_calls = 0
        self.full_b_error = None
        self.metadata_error = None

    def solve(self, _delta, _gain, _log_tau, *, derivative):
        self.solve_calls += 1
        progress = {'completion_class': 'known_complete',
            'completed_context_rk4_ticks': 20,
            'completed_conditional_model_s': .005,
            'completed_decoder_endpoint_ticks': 1,
            'bank_completion_class': 'known_complete' if derivative else 'not_requested'}
        return ({'B': np.zeros((1, 2, 3))},
                {'rate': np.zeros((1, 2, 1)), 'branch_bits': np.zeros((1, 20, 1), np.uint8)},
                {'progress': progress})

    def host_copy(self, value, *, progress=None, label='device_output'):
        if self.full_b_error is not None:
            raise self.full_b_error
        return value

    def rescue_device_state(self, _device):
        return {}, []

    def metadata(self):
        if self.metadata_error is not None:
            raise self.metadata_error
        return {'fabricated': True, 'solve_calls': self.solve_calls}


def test_verifier_returned_response_failure_ledger_and_secondary(monkeypatch, tmp_path):
    out = evidence_dir(tmp_path)/'fake-returned-response'
    out.mkdir(parents=True)
    cases = {}
    original_savez = verifier.np.savez
    original_save = verifier.save
    for label in ('initial_ledger', 'raw_save', 'full_B', 'final_ledger',
                  'secondary_cascade'):
        case_out = out/label
        case_out.mkdir()
        path = case_out/'batch-raw.npz'
        row = {'name': label}
        ledger = {'original_batches': [row]}
        engine = _ReturnedTinyEngine()
        primary = OSError('injected '+label)
        save_calls = 0

        def bad_save(target, value, case_label=label, primary_error=primary):
            nonlocal save_calls
            save_calls += 1
            fail_at = {'initial_ledger': 1, 'final_ledger': 3,
                       'secondary_cascade': 2}.get(case_label)
            if save_calls == fail_at:
                if case_label == 'secondary_cascade':
                    raise OSError('secondary final ledger')
                raise primary_error
            return original_save(target, value)

        def bad_savez(target, *, case_label=label, primary_error=primary,
                      raw_path=path, **arrays):
            if case_label in ('raw_save', 'secondary_cascade') and Path(target) == raw_path:
                raise primary_error
            if case_label == 'secondary_cascade':
                raise OSError('secondary partial save')
            return original_savez(target, **arrays)

        if label == 'full_B':
            engine.full_b_error = primary
        if label == 'secondary_cascade':
            engine.metadata_error = RuntimeError('secondary metadata')
        (case_out/'invocation.json').write_text(json.dumps({
            'status': 'requested', 'fake_engine': True, 'case': label}),
            encoding='utf-8')
        began = time.perf_counter()
        with monkeypatch.context() as patch:
            patch.setattr(verifier, 'save', bad_save)
            patch.setattr(verifier.np, 'savez', bad_savez)
            with pytest.raises(OSError) as caught:
                verifier._execute(engine, np.zeros(1), np.ones(1), np.zeros(1),
                    label == 'full_B', path, row, ledger, case_out)
        observed = {'case': label, 'elapsed_s': time.perf_counter()-began,
            'primary_identity_preserved': caught.value is primary,
            'error': repr(caught.value), 'notes': getattr(caught.value, '__notes__', []),
            'solve_calls': engine.solve_calls, 'save_calls': save_calls, 'row': row}
        (case_out/'observed.json').write_text(json.dumps(verifier.safe(observed),
            indent=2), encoding='utf-8')
        cases[label] = observed
    assert all(item['primary_identity_preserved'] for item in cases.values())
    initial = cases['initial_ledger']
    assert initial['solve_calls'] == 0
    assert not initial['row']['solve_requested']
    assert not initial['row']['solve_returned']
    for label in ('raw_save', 'full_B', 'final_ledger', 'secondary_cascade'):
        row = cases[label]['row']
        assert cases[label]['solve_calls'] == 1
        assert row['solve_requested'] and row['solve_returned']
        assert row['post_return_error']
        assert row['progress']['completed_context_rk4_ticks'] == 20
    assert cases['final_ledger']['row']['raw_sha256']
    assert cases['secondary_cascade']['row']['secondary_metadata_error']
    assert cases['secondary_cascade']['row']['secondary_final_ledger_error']
    assert any('Secondary partial raw save' in note
        for note in cases['secondary_cascade']['notes'])
    ledger = {'tiny_batches': [initial['row']],
        'original_batches': [cases[label]['row'] for label in (
            'raw_save', 'full_B', 'final_ledger', 'secondary_cascade')]+[
            {'solve_requested': True, 'solve_returned': False,
             'post_return_error': False}]}
    verifier._seal_actual_counters(ledger)
    (out/'synthetic-actual-counters.json').write_text(json.dumps(
        verifier.safe(ledger), indent=2), encoding='utf-8')
    assert (ledger['actual_tiny_solve_requested'],
            ledger['actual_tiny_solve_returned'],
            ledger['actual_tiny_solve_unknown'],
            ledger['actual_tiny_post_return_errors']) == (0, 0, 0, 0)
    assert (ledger['actual_original_solve_requested'],
            ledger['actual_original_solve_returned'],
            ledger['actual_original_solve_unknown'],
            ledger['actual_original_post_return_errors'],
            ledger['original_batches_raw_saved']) == (5, 4, 1, 4, 1)
