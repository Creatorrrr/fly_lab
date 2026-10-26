"""E130 generated CUDA equality and bounded adapter failure accounting."""

from __future__ import annotations

import json
import os
import time
import types
from copy import deepcopy

import numpy as np
import pytest

from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import banc_m3_precomputed_expm1 as candidate
from tools import verify_c_banc_m3_precomputed_expm1 as verify


def generated_problem():
    # Original-generated values only: two RK4 midpoint patterns, two muscles,
    # opposite delta signs and a source that can cross the rectifier boundary.
    stages = 160
    t = np.arange(stages, dtype=np.float64)
    current = np.zeros((2, stages, 2), np.float32)
    current[:, :, 0] = (7.2 + 1.8 * np.sin(t * .27))[None, :]
    current[:, :, 1] = (8.1 + 2.2 * np.cos(t * .31))[None, :]
    current[1] += .4
    source = np.empty((2, stages, 2), np.float32)
    source[:, :, 0] = .3 + .25 * (1 + np.sin(t * .17))[None, :]
    source[:, :, 1] = .5 + .2 * (1 + np.cos(t * .21))[None, :]
    source[1] *= 1.2
    moment = np.zeros((42, 2))
    moment[0, 0], moment[7, 1] = .1, -.08
    return {'stage_current': current, 'source_rate': source,
        'time_s': np.array([0., .005, .01]),
        'initial_rate': np.zeros((2, 2)),
        'initial_adaptation': np.zeros((2, 2)),
        'initial_activation': np.zeros((2, 2)),
        'tau': np.array([.02, .025]), 'a': np.array([1., 1.1]),
        'threshold': np.array([7.5, 8.2]), 'cap': np.array([200., 180.]),
        'pool': np.eye(2), 'moment': moment, 'rest': np.zeros(42),
        'ltm': np.array([0, 1]), 'membership': np.array([0, 1]),
        'motor_ids': np.array([1001, 1002]),
        'source_ids': np.array([2001, 2002]),
        'union_edge_ids': np.array([3001, 3002, 3003]),
        'union_rows': np.array([0, 0, 1]),
        'union_cols': np.array([0, 1, 1]),
        'union_values': np.array([[1., -.75, .875], [1.125, -.625, .75]], np.float32),
        'fit_union_positions': np.array([0, 1, 2]),
        'edge_ids': np.array([3001, 3002, 3003]),
        'edge_rows': np.array([0, 0, 1]),
        'edge_cols': np.array([0, 1, 1]),
        'base_fit_weights': np.array([1.125, -.625, .75], np.float64)}


def evidence_dir(tmp_path):
    path = os.environ.get('E130_SYNTHETIC_EVIDENCE_DIR')
    out = tmp_path if path is None else __import__('pathlib').Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_generated_cuda_all_outputs_native_and_restore(tmp_path):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('Actual CUDA device unavailable')
    out = evidence_dir(tmp_path)
    ledger = {'schema': 'flylab.m3.e130.synthetic-cuda-ledger.v1',
              'generated_only': True, 'calls': []}
    source = generated_problem()
    delta = np.array([.19, -.27, .08], np.float64)
    gain = np.array([.7, 2.3], np.float64)
    original_delta = delta.copy()
    factory = neural._kernel
    baseline = neural.CudaJointAdaptiveEngine(source)
    contender = neural.CudaJointAdaptiveEngine(source)
    outputs = {}
    adapters = {}
    for native in (False, True):
        for name, engine in (('baseline', baseline), ('candidate', contender)):
            label = f'{name}-native-{int(native)}'
            entry = {'label': label, 'requested_calls': 1,
                     'requested_context_ticks': 80,
                     'requested_conditional_s': .02,
                     'requested_banks': 1,
                     'requested_edge_stage_ticks': 960,
                     'status': 'requested'}
            ledger['calls'].append(entry)
            (out / 'synthetic-invocations.json').write_text(json.dumps(ledger, indent=2))
            began = time.perf_counter()
            adapter = None
            try:
                if name == 'candidate':
                    with candidate.precomputed_expm1(engine) as adapter:
                        device, raw, timing = engine.solve(delta, gain,
                            native_mode=native, derivative=True)
                else:
                    device, raw, timing = engine.solve(delta, gain,
                        native_mode=native, derivative=True)
                raw['B'] = engine.host_copy(device['B'], label='synthetic_full_B')
                np.savez(out / f'{label}-raw.npz',
                         synthetic_stage_current=source['stage_current'],
                         synthetic_source_rate=source['source_rate'], **raw)
                entry.update(status='raw_saved', completed_calls=1,
                    progress=timing['progress'], raw_path=str(out / f'{label}-raw.npz'),
                    raw_sha256=verify.profile.sha256(out / f'{label}-raw.npz'),
                    completed_context_ticks=timing['progress']['completed_context_rk4_ticks'],
                    completed_edge_stage_ticks=timing['progress'][
                        'completed_bank_edge_stage_ticks'],
                    adapter=adapter.snapshot() if adapter else None)
                outputs[label] = raw
                adapters[label] = adapter
            except BaseException as error:
                entry.update(status='failed', completed_calls=0,
                    error=repr(error), response_progress=getattr(error,
                    'response_progress', None), adapter=getattr(error,
                    'precomputed_expm1_progress', None))
                partial = getattr(error, 'partial_response', None)
                if partial:
                    np.savez(out / f'{label}-partial-raw.npz', **partial)
                raise
            finally:
                entry['wall_s'] = time.perf_counter() - began
                (out / 'synthetic-invocations.json').write_text(json.dumps(
                    ledger, indent=2, default=str))
    # All four raw files and invocation accounting exist before numerical assertions.
    assert np.array_equal(delta, original_delta)
    assert neural._kernel is factory
    assert 'solve' not in vars(contender)
    assert adapters['candidate-native-1'].snapshot()['counts']['prepare']['requested'] == 0
    assert adapters['candidate-native-0'].snapshot()['counts']['prepare'] == {
        'requested': 1, 'enqueued': 1, 'synchronized': 1,
        'unknown_requested': 0}
    assert adapters['candidate-native-0'].snapshot()['counts']['joint_forward'] == {
        'requested': 1, 'enqueued': 1, 'synchronized': 1,
        'unknown_requested': 0}
    assert adapters['candidate-native-0'].gpu_temp_bytes_peak == 24
    for native in (False, True):
        a = outputs[f'baseline-native-{int(native)}']
        b = outputs[f'candidate-native-{int(native)}']
        for key in (*neural.FIELDS, 'B', 'branch_bits'):
            if key == 'branch_bits':
                assert np.array_equal(a[key], b[key])
            else:
                assert np.max(np.abs(a[key] - b[key])) <= 1e-8
        assert np.any(a['branch_bits'] == 0)
        assert np.any(a['branch_bits'] == 15)
        assert np.any((a['branch_bits'] != 0) & (a['branch_bits'] != 15))
    assert any(np.any(outputs['baseline-native-0'][key] !=
                      outputs['baseline-native-1'][key])
               for key in ('rate', 'branch_bits'))


@pytest.mark.parametrize('stage', ['allocation', 'prepare', 'forward'])
def test_failure_accounting_and_primary_exception(monkeypatch, stage, tmp_path):
    cp = pytest.importorskip('cupy')
    engine = neural.CudaJointAdaptiveEngine.__new__(neural.CudaJointAdaptiveEngine)
    engine.edges = 2
    engine.stream = types.SimpleNamespace(synchronize=lambda: None)
    delta = cp.asarray(np.array([.2, -.3])) if cp.cuda.runtime.getDeviceCount() else None
    if delta is None:
        pytest.skip('Actual CUDA device unavailable')
    primary = RuntimeError(stage + ' primary')
    def fake_kernel(name):
        def launch(*args, **kwargs):
            if name == ('prepare_expm1' if stage == 'prepare' else 'joint_forward'):
                raise primary
        return launch
    monkeypatch.setattr(candidate, '_candidate_kernel', fake_kernel)
    if stage == 'allocation':
        def fail_allocation(value):
            raise primary
        monkeypatch.setattr(cp, 'empty_like', fail_allocation)
    args = [None] * 27
    args[5] = delta
    args[-2] = np.int32(0)
    before = delta.copy()
    original_factory = neural._kernel
    began = time.perf_counter()
    with pytest.raises(RuntimeError) as caught, candidate.precomputed_expm1(
            engine) as adapter:
        neural._kernel('joint_forward')((1,), (32,), tuple(args))
    record = {'stage': stage, 'generated_only': True,
              'requested_E128_solve_calls': 0, 'requested_context_ticks': 0,
              'requested_banks': 0, 'requested_adapter_launches': 1,
              'status': 'expected_failure', 'wall_s': time.perf_counter() - began,
              'primary_error': repr(caught.value),
              'adapter': adapter.snapshot(), 'raw_available': False}
    (evidence_dir(tmp_path) / f'synthetic-fault-{stage}.json').write_text(
        json.dumps(record, indent=2))
    assert neural._kernel is original_factory
    assert 'solve' not in vars(engine)
    assert cp.array_equal(delta, before)
    assert caught.value is primary
    counts = adapter.snapshot()['counts']
    assert counts['prepare']['requested'] == 1
    assert counts['prepare']['synchronized'] == (stage == 'forward')
    assert counts['prepare']['unknown_requested'] == (stage != 'forward')
    assert counts['joint_forward']['requested'] == (stage == 'forward')
    assert counts['joint_forward']['unknown_requested'] == (stage == 'forward')
    assert adapter.gpu_temp_bytes_requested == 16
    assert adapter.gpu_temp_bytes_allocated == (0 if stage == 'allocation' else 16)
    assert adapter.gpu_temp_bytes_peak == (0 if stage == 'allocation' else 16)
    assert caught.value.precomputed_expm1_progress == adapter.snapshot()


def _progress(ticks, decoder, bank, seconds):
    return {'completion_class': 'known_complete',
        'forward_completion_class': 'known_complete',
        'decoder_completion_class': 'known_complete',
        'bank_completion_class': 'known_complete',
        'derivative_bank_complete': True,
        'completed_context_rk4_ticks': ticks,
        'completed_conditional_model_s': seconds,
        'completed_decoder_endpoint_ticks': decoder,
        'completed_bank_edge_stage_ticks': bank,
        'dynamic_upload_requested': {'count': 2},
        'dynamic_upload_enqueued': {'count': 2},
        'dynamic_upload_synchronized': {'count': 2},
        'dynamic_upload_events': [{'synchronized': True}] * 2}


def _metadata(calls):
    return {'solve_calls': calls, 'bank_calls': calls,
        'fixed_upload': {'unknown_requested': {'count': 0}},
        'dynamic_upload': {key: {'count': 2*calls} for key in
            ('requested', 'enqueued', 'synchronized')} |
            {'unknown_requested': {'count': 0}}}


def _adapter(edges):
    return {'events': [
        {'kind': 'prepare', 'bytes': edges*8, 'requested': True,
         'enqueued': True, 'synchronized': True, 'unknown_requested': False,
         'error': None},
        {'kind': 'joint_forward', 'bytes': 0, 'requested': True,
         'enqueued': True, 'synchronized': True, 'unknown_requested': False,
         'error': None}],
        'counts': {kind: {'requested': 1, 'enqueued': 1,
            'synchronized': 1, 'unknown_requested': 0}
            for kind in ('prepare', 'joint_forward')},
        'gpu_temp_bytes_requested': edges*8,
        'gpu_temp_bytes_allocated': edges*8,
        'gpu_temp_bytes_peak': edges*8}


def _gate_fixture():
    rows = []
    cuda = []
    seen = {'baseline': 0, 'candidate': 0}
    for i, name in enumerate(verify.ORDER):
        seen[name] += 1
        phase = f'original_{i + 1}_{name}'
        rows.append({'index': i + 1, 'backend': name, 'phase': phase,
            'status': 'raw_saved', 'raw_sha256': 'generated', 'raw_path': 'generated',
            'evaluate_wall_s': 2. if name == 'baseline' else 1.,
            'requested_context_ticks': 120000,
            'requested_derivative_banks': 1,
            'requested_edge_stage_ticks': 131040000,
            'solve_span_count': 1,
            'solve': {'phase': phase, 'returned': True,
                'response_progress': _progress(120000, 6000, 131040000, 30)},
            'engine_metadata': _metadata(seen[name]),
            'adapter': _adapter(273) if name == 'candidate' else None})
        cuda += [{'phase': phase}] * 7
    tiny = [{'backend': name, 'status': 'completed', 'raw_sha256': 'generated',
        'progress': _progress(20, 1, 80, .005),
        'engine_metadata': _metadata(1),
        'adapter': _adapter(1) if name == 'candidate' else None}
        for name in ('baseline', 'candidate')]
    ledger = {'original_evaluations': rows, 'original_evaluations_requested': 6,
        'original_evaluations_completed': 6, 'synthetic_tiny': tiny,
        'synthetic_tiny_requested': 2, 'synthetic_tiny_completed': 2}
    observation = types.SimpleNamespace(cuda=cuda, secondary=[])
    comparisons = [{'accepted': True, 'old_reference': bool(i % 2)}
                   for i in range(6)]
    freeze = {'min_forward_pair_speedup': 1.25,
              'min_median_eval_speedup': 1.2}
    return freeze, ledger, observation, comparisons


@pytest.mark.parametrize('damage', ['empty_adapter', 'partial_adapter',
    'missing_comparison', 'missing_tiny', 'wrong_phase', 'missing_progress',
    'wrong_ticks', 'missing_metadata', 'missing_evaluation', 'empty_evaluations'])
def test_gate_rejects_absent_or_incomplete_generated_evidence(monkeypatch, damage):
    monkeypatch.setattr(verify, 'complete_span', lambda obs, phase, label:
        (2. if 'baseline' in phase else 1.) if label == 'joint_forward' else 1.)
    freeze, ledger, observation, comparisons = _gate_fixture()
    assert verify.result_from_raw(freeze, ledger, observation,
                                  comparisons)['accepted_local_one_point_performance']
    ledger = deepcopy(ledger)
    if damage == 'empty_adapter':
        ledger['original_evaluations'][1]['adapter'] = {
            'events': [], 'counts': {kind: {state: 0 for state in
                ('requested', 'enqueued', 'synchronized', 'unknown_requested')}
                for kind in ('prepare', 'joint_forward')}}
    elif damage == 'partial_adapter':
        ledger['original_evaluations'][1]['adapter']['events'][0]['synchronized'] = False
    elif damage == 'missing_comparison':
        comparisons.pop()
    elif damage == 'missing_tiny':
        ledger['synthetic_tiny'].pop()
    elif damage == 'wrong_phase':
        ledger['original_evaluations'][1]['solve']['phase'] = 'original_1_baseline'
    elif damage == 'missing_progress':
        ledger['original_evaluations'][1]['solve']['response_progress'] = {}
    elif damage == 'wrong_ticks':
        ledger['original_evaluations'][1]['solve']['response_progress'][
            'completed_context_rk4_ticks'] = 119999
    elif damage == 'missing_metadata':
        ledger['original_evaluations'][1]['engine_metadata'] = None
    elif damage == 'missing_evaluation':
        ledger['original_evaluations'].pop()
    elif damage == 'empty_evaluations':
        ledger['original_evaluations'].clear()
    result = verify.result_from_raw(freeze, ledger, observation, comparisons)
    assert not result['accepted_local_one_point_performance']
    if damage in ('missing_evaluation', 'empty_evaluations'):
        assert result['median_whole_eval_wall_speedup'] is None
        assert result['interpretation'] == 'diagnostic_only_incomplete'
    json.dumps(result, allow_nan=False)


def test_double_cleanup_fault_preserves_primary_and_partial_hash(tmp_path):
    path = tmp_path / 'leg-0-evaluation-001-partial-raw.npz'
    np.savez(path, generated_partial=np.array([1.]))
    primary = RuntimeError('generated primary CUDA failure')
    def fail_metadata():
        raise OSError('generated metadata failure')
    def fail_save():
        raise OSError('generated save failure')
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            verify.safe_finally(('metadata', fail_metadata), ('save', fail_save))
    assert caught.value is primary
    assert len(primary.__notes__) == 2
    assert verify.partial_paths(tmp_path, primary) == [{
        'path': str(path), 'sha256': verify.profile.sha256(path)}]


@pytest.mark.parametrize('fault', ['slow', 'write'])
def test_final_manifest_failure_invalidates_generated_result(monkeypatch, tmp_path, fault):
    verify.save(tmp_path / 'result.json',
        {'accepted_local_one_point_performance': True})
    ledger = {'status': 'completed'}
    original = verify.write_output_manifest
    def fail_or_delay(out):
        if fault == 'write':
            raise OSError('generated manifest failure')
        original(out)
        time.sleep(.02)
    monkeypatch.setattr(verify, 'write_output_manifest', fail_or_delay)
    primary = None
    def secondary(stage, error):
        nonlocal primary
        if primary is None:
            primary = error
        else:
            primary.add_note('Secondary ' + stage + ': ' + repr(error))
        return primary
    got = verify.finalize_artifacts(tmp_path, ledger,
        {'internal_wall_limit_s': .01}, time.perf_counter(), None, secondary)
    assert got is primary
    assert isinstance(got, TimeoutError if fault == 'slow' else OSError)
    assert ledger['status'] == 'failed'
    assert not (tmp_path / 'result.json').exists()
    assert (tmp_path / 'advance-ledger.json').exists()


def test_success_manifest_covers_final_generated_files(tmp_path):
    verify.save(tmp_path / 'result.json',
        {'accepted_local_one_point_performance': True})
    ledger = {'status': 'completed'}
    def secondary(stage, error):
        raise AssertionError(stage) from error
    assert verify.finalize_artifacts(tmp_path, ledger,
        {'internal_wall_limit_s': 1.}, time.perf_counter(), None, secondary) is None
    manifest = json.loads((tmp_path / 'output-manifest.json').read_text())
    assert ledger['post_manifest_wall_s'] > 0
    assert ledger['final_wall_limit_s'] == 1.
    assert set(manifest['files']) == {'advance-ledger.json', 'result.json'}
    for relative, digest in manifest['files'].items():
        assert digest == verify.profile.sha256(tmp_path / relative)
