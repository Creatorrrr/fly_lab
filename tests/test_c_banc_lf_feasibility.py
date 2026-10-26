"""CPU-only generated checks for E131 restoration and bounded accounting."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_lf_feasibility as e131


def generated_data(x, *, q_offset=.0, ltm_offset=.0):
    """Two generated parameters drive only initial command/tendon errors."""
    x = np.asarray(x, np.float64)
    q = np.full((3, 99, 7), q_offset + .05 * (x[0] - .4))
    ltm = np.full((3, 99), ltm_offset + .03 * (x[1] - .3))
    jq = np.zeros((3, 99, 7, 2))
    jltm = np.zeros((3, 99, 2))
    jq[..., 0] = .05
    jltm[..., 1] = .03
    return {'delta': x[:1].copy(), 'gain': x[1:].copy(),
            'losses': np.array([.2, .3, .4]),
            'loss_jac': np.zeros((3, 2)),
            'q_error': q, 'ltm_error': ltm,
            'q_jac': jq, 'ltm_jac': jltm}


@pytest.mark.parametrize('offsets', [(.065, .0), (-.025, .0),
                                     (.0, .025), (.0, -.007)])
def test_exact_signed_and_rms_gradient_in_both_directions(offsets):
    x = np.array([.1, .1])
    data = generated_data(x, q_offset=offsets[0], ltm_offset=offsets[1])
    value, gradient, ratios = e131.restoration_value_jac(data)
    assert value > 0
    assert np.max(ratios) > e131.fit.SELECTOR_CAP
    assert gradient.shape == (2,)
    step = 1e-6
    for axis in range(2):
        direction = np.eye(2)[axis] * step
        plus = e131.restoration_value_jac(generated_data(
            x + direction, q_offset=offsets[0], ltm_offset=offsets[1]))[0]
        minus = e131.restoration_value_jac(generated_data(
            x - direction, q_offset=offsets[0], ltm_offset=offsets[1]))[0]
        assert gradient[axis] == pytest.approx((plus - minus) / (2 * step),
                                               rel=2e-6, abs=2e-8)


def test_epigraph_rows_excluded_and_original_cap_unchanged(monkeypatch):
    data = generated_data([.1, .1])
    baseline = e131.restoration_value_jac(data)
    data['losses'] = np.array([99., -4., 1000.])
    data['loss_jac'][:] = 1e12
    after = e131.restoration_value_jac(data)
    assert after[0] == baseline[0]
    assert np.array_equal(after[1], baseline[1])
    assert np.array_equal(after[2], baseline[2])
    calls = []
    original = e131.fit.constraint_values

    def spy(*args, **kwargs):
        calls.append(kwargs['optimizer_cap'])
        return original(*args, **kwargs)

    monkeypatch.setattr(e131.fit, 'constraint_values', spy)
    e131.restoration_value_jac(data)
    assert calls == [e131.fit.OPTIMIZER_CAP]
    assert e131.fit.OPTIMIZER_CAP < e131.fit.SELECTOR_CAP


def test_generated_solver_stops_at_first_selector_feasible():
    seen = []

    def evaluate(x, index):
        seen.append((index, x.copy()))
        return generated_data(x)

    now = time.perf_counter()
    cache = e131.RestorationCache(evaluate, [0., 0.], [[0., 1.], [0., 1.]],
        max_unique=16, fit_deadline=now + 10, internal_deadline=now + 10,
        reserve=0)
    accounting = {}
    solver, stop = e131.run_optimizer(cache, accounting)
    assert solver is None
    assert stop == 'first_selector_feasible'
    assert accounting['status'] == 'bounded_stop'
    assert accounting['requested'] == accounting['bounded_stop'] == 1
    assert accounting['returned'] == accounting['failed_unknown'] == 0
    assert accounting['cache_started'] == cache.started
    assert accounting['elapsed_wall_s'] >= 0
    assert cache.selected_index == cache.started == cache.completed == len(seen)
    assert seen[0][0] == 1
    assert cache.started <= 16
    assert e131.fit.selector_feasible(e131.restoration_value_jac(
        generated_data(seen[-1][1]))[2])
    before = cache.started
    assert cache.get(seen[0][1])[0] > 0
    assert cache.started == before and cache.hits >= 1


def test_optimizer_request_return_initial_stop_and_error_accounting(monkeypatch):
    now = time.perf_counter()

    def make_cache(start):
        return e131.RestorationCache(lambda x, _: generated_data(x), start,
            [[0., 1.], [0., 1.]], max_unique=16,
            fit_deadline=now + 10, internal_deadline=now + 10, reserve=0)

    calls = []

    def fake_minimize(*_args, **_kwargs):
        calls.append('minimize')
        return SimpleNamespace(success=True)

    monkeypatch.setattr(e131, 'minimize', fake_minimize)
    normal = {}
    solver, stop = e131.run_optimizer(make_cache([0., 0.]), normal)
    assert solver.success and stop == 'scipy_finished'
    assert calls == ['minimize']
    assert (normal['requested'], normal['returned'], normal['bounded_stop'],
            normal['failed_unknown'], normal['status']) == (1, 1, 0, 0, 'returned')

    initial = {}
    solver, stop = e131.run_optimizer(make_cache([.4, .3]), initial)
    assert solver is None and stop == 'first_selector_feasible'
    assert calls == ['minimize']
    assert (initial['requested'], initial['returned'], initial['bounded_stop'],
            initial['failed_unknown'], initial['status']) == (0, 0, 0, 0,
                                                             'not_requested')

    primary = RuntimeError('generated optimizer exception')

    def fail_minimize(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(e131, 'minimize', fail_minimize)
    failed = {}
    cache = make_cache([0., 0.])
    with pytest.raises(RuntimeError) as caught:
        e131.run_optimizer(cache, failed)
    assert caught.value is primary
    assert (failed['requested'], failed['returned'], failed['bounded_stop'],
            failed['failed_unknown'], failed['status']) == (1, 0, 0, 1,
                                                            'failed_unknown')
    assert failed['cache_started'] == failed['cache_completed'] == 1
    assert failed['elapsed_wall_s'] >= 0


def test_cache_bounds_unique_cap_deadlines_and_byte_identity():
    seen = []
    clock = [0.]

    def evaluate(x, index):
        seen.append(index)
        return generated_data([0., 0.])

    cache = e131.RestorationCache(evaluate, [0., 0.], [[-1., 1.], [-1., 1.]],
        max_unique=1, fit_deadline=10., internal_deadline=10., reserve=2.,
        clock=lambda: clock[0])
    cache.get([0., 0.])
    cache.get([0., 0.])
    assert seen == [1] and cache.hits == 1
    with pytest.raises(e131.BoundedStop, match='unique_cap'):
        cache.get([.1, 0.])
    with pytest.raises(ValueError, match='in-bounds'):
        cache.get([2., 0.])
    with pytest.raises(ValueError, match='Finite'):
        cache.get([np.nan, 0.])
    second = e131.RestorationCache(evaluate, [0., 0.], [[-1., 1.], [-1., 1.]],
        max_unique=2, fit_deadline=10., internal_deadline=9., reserve=2.,
        clock=lambda: clock[0])
    clock[0] = 7.
    with pytest.raises(e131.BoundedStop, match='pre_call_wall_reserve'):
        second.get([0., 0.])
    assert second.started == 0
    # IEEE signed zero has a distinct byte key despite numerical equality.
    clock[0] = 0.
    third = e131.RestorationCache(evaluate, [0., 0.], [[-1., 1.], [-1., 1.]],
        max_unique=2, fit_deadline=10., internal_deadline=10., reserve=0,
        clock=lambda: clock[0])
    third.get([0., 0.])
    third.get([-0., 0.])
    assert third.started == 2


def test_evidence_gate_rejects_incomplete_generated_rows(monkeypatch):
    progress = {'completion_class': 'known_complete',
        'forward_completion_class': 'known_complete',
        'decoder_completion_class': 'known_complete',
        'bank_completion_class': 'known_complete',
        'derivative_bank_complete': True,
        'completed_context_rk4_ticks': 120000,
        'completed_conditional_model_s': 30,
        'completed_decoder_endpoint_ticks': 6000,
        'completed_bank_edge_stage_ticks': 131040000,
        'dynamic_upload_requested': {'count': 2},
        'dynamic_upload_enqueued': {'count': 2},
        'dynamic_upload_synchronized': {'count': 2},
        'dynamic_upload_events': [{'synchronized': True}] * 2}
    adapter = {'events': [
        {'kind': kind, 'bytes': count, 'requested': True,
         'enqueued': True, 'synchronized': True,
         'unknown_requested': False, 'error': None}
        for kind, count in [('prepare', 273 * 8), ('joint_forward', 0)]],
        'counts': {kind: {'requested': 1, 'enqueued': 1,
            'synchronized': 1, 'unknown_requested': 0}
            for kind in ('prepare', 'joint_forward')},
        'gpu_temp_bytes_requested': 273 * 8,
        'gpu_temp_bytes_allocated': 273 * 8,
        'gpu_temp_bytes_peak': 273 * 8}
    metadata = {'solve_calls': 1, 'bank_calls': 1,
        'fixed_upload': {'unknown_requested': {'count': 0}},
        'dynamic_upload': {key: {'count': count} for key, count in
            [('requested', 2), ('enqueued', 2), ('synchronized', 2),
             ('unknown_requested', 0)]},
        'geometry_cost': {key: 5473 for key in (
            'forward_rows_requested', 'forward_rows_enqueued',
            'forward_rows_completed', 'backward_rows_requested',
            'backward_rows_enqueued', 'backward_rows_completed')} |
            {'forward_rows_unknown_requested': 0,
             'backward_rows_unknown_requested': 0}}
    tiny_progress = dict(progress)
    tiny_progress.update(completed_context_rk4_ticks=20,
        completed_conditional_model_s=.005,
        completed_decoder_endpoint_ticks=1,
        completed_bank_edge_stage_ticks=80)
    tiny_adapter = json.loads(json.dumps(adapter))
    tiny_adapter['events'][0]['bytes'] = 8
    for key in ('gpu_temp_bytes_requested', 'gpu_temp_bytes_allocated',
                'gpu_temp_bytes_peak'):
        tiny_adapter[key] = 8
    row = {'index': 1, 'phase': 'original_1_candidate',
        'backend': 'candidate', 'status': 'raw_saved',
        'raw_path': 'generated', 'raw_index': 1,
        'requested_context_ticks': 120000,
        'requested_conditional_s': 30,
        'requested_scalar_MN_ticks': 8280000,
        'requested_derivative_banks': 1,
        'requested_edge_stage_ticks': 131040000,
        'requested_decoder_endpoints': 6000,
        'raw_sha256': 'generated', 'max_ratio': .9, 'selector_feasible': True,
        'ratios': [[.9] * 4] * 3, 'restoration_value': 0.,
        'restoration_gradient_norm': 0.,
        'solve_span_count': 1,
        'solve': {'phase': 'original_1_candidate', 'returned': True,
                  'response_progress': progress},
        'adapter': adapter, 'engine_metadata': metadata}
    ledger = {'original_evaluations': [row],
              'original_evaluations_requested': 1,
              'original_evaluations_completed': 1,
              'synthetic_tiny': [{'status': 'completed',
                  'raw_sha256': 'generated', 'progress': tiny_progress,
                  'adapter': tiny_adapter, 'engine_metadata': metadata}],
              'synthetic_tiny_requested': 1, 'synthetic_tiny_completed': 1,
              'cuda_spans': [{'phase': 'original_1_candidate'}] * 7,
              'secondary_errors': [],
              'cache_started': 1, 'cache_completed': 1, 'cache_hits': 0,
              'fit_wall_s': 1., 'fit_calls': 0,
              'optimizer': {'status': 'not_requested', 'requested': 0,
                  'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0},
              'source_review_sha256': 'generated',
              'freeze_sha256': 'generated'}
    monkeypatch.setattr(e131.e130, 'complete_span', lambda *_: 1.)
    cache = SimpleNamespace(selected_index=1, hits=0, started=1,
                            completed=1, cache={b'generated': 0})
    comparisons = [{'accepted': True, 'old_reference': True,
                    'actual_sha256': 'generated', 'reference_sha256': 'old'},
                   {'accepted': True, 'old_reference': False,
                    'actual_sha256': 'generated', 'reference_sha256': 'full'}]
    freeze = {'start_raw_path': 'start', 'full_reference_raw_path': 'full',
              'input_sha256': {'start': 'old', 'full': 'full'},
              'fit_wall_limit_s': 20.}
    result = e131.result_from_raw(freeze, ledger, comparisons,
                                  'first_selector_feasible', cache)
    assert result['restoration_feasible']
    for damage in ('adapter', 'comparison', 'progress', 'unknown', 'count',
                   'geometry', 'tiny', 'timing', 'secondary', 'geometry_unknown',
                   'wrong_phase', 'wrong_kind', 'cache_mismatch', 'fit_overrun',
                   'nonfinite_wall', 'stale_index', 'invalid_index',
                   'malformed_ratio'):
        local = json.loads(json.dumps(ledger))
        checks = json.loads(json.dumps(comparisons))
        selected = cache.selected_index
        if damage == 'adapter':
            local['original_evaluations'][0]['adapter']['events'][0]['synchronized'] = False
        elif damage == 'comparison':
            checks[1]['accepted'] = False
        elif damage == 'progress':
            local['original_evaluations'][0]['solve']['response_progress'] = {}
        elif damage == 'unknown':
            local['original_evaluations'][0]['adapter']['counts']['prepare'][
                'unknown_requested'] = 1
        elif damage == 'geometry_unknown':
            local['original_evaluations'][0]['engine_metadata'][
                'geometry_cost']['forward_rows_unknown_requested'] = 1
        elif damage == 'wrong_phase':
            local['original_evaluations'][0]['phase'] = 'original_1_baseline'
        elif damage == 'wrong_kind':
            checks[0]['old_reference'] = False
        elif damage == 'cache_mismatch':
            local['cache_completed'] = 0
        elif damage == 'fit_overrun':
            local['fit_wall_s'] = 20.1
        elif damage == 'nonfinite_wall':
            local['fit_wall_s'] = 'nan'
        elif damage == 'stale_index':
            selected = 0
        elif damage == 'invalid_index':
            selected = 99
        elif damage == 'malformed_ratio':
            local['original_evaluations'][0]['ratios'] = []
        else:
            if damage == 'count':
                local['original_evaluations_completed'] = 0
            elif damage == 'geometry':
                local['original_evaluations'][0]['engine_metadata'][
                    'geometry_cost']['forward_rows_completed'] = 0
            elif damage == 'tiny':
                local['synthetic_tiny_completed'] = 0
            elif damage == 'timing':
                local['cuda_spans'] = []
            else:
                local['secondary_errors'] = [{'stage': 'generated'}]
        candidate = SimpleNamespace(selected_index=selected, hits=0,
            started=1, completed=1, cache={b'generated': 0})
        diagnostic = e131.result_from_raw(freeze, local, checks,
            'first_selector_feasible', candidate)
        assert diagnostic['status'] != 'completed_bounded_fit'
        assert not diagnostic['restoration_feasible']
        json.dumps(diagnostic, allow_nan=False)

    empty = json.loads(json.dumps(ledger))
    empty['original_evaluations'] = []
    empty['original_evaluations_requested'] = 0
    empty['original_evaluations_completed'] = 0
    empty['cache_started'] = empty['cache_completed'] = 0
    empty_cache = SimpleNamespace(selected_index=None, hits=0, started=0,
                                  completed=0, cache={})
    diagnostic = e131.result_from_raw(freeze, empty, comparisons,
                                      'pre_call_wall_reserve', empty_cache)
    assert diagnostic['status'] == 'invalid_evidence'

    extra = json.loads(json.dumps(ledger))
    second = json.loads(json.dumps(extra['original_evaluations'][0]))
    second['index'] = 2
    second['phase'] = second['solve']['phase'] = 'original_2_candidate'
    second['engine_metadata']['solve_calls'] = 2
    second['engine_metadata']['bank_calls'] = 2
    for key in ('requested', 'enqueued', 'synchronized'):
        second['engine_metadata']['dynamic_upload'][key]['count'] = 4
    for key in ('forward_rows_requested', 'forward_rows_enqueued',
                'forward_rows_completed', 'backward_rows_requested',
                'backward_rows_enqueued', 'backward_rows_completed'):
        second['engine_metadata']['geometry_cost'][key] = 10946
    extra['original_evaluations'].append(second)
    extra['original_evaluations_requested'] = 2
    extra['original_evaluations_completed'] = 2
    extra['cache_started'] = extra['cache_completed'] = 2
    extra['fit_calls'] = 1
    extra['optimizer'] = {'status': 'bounded_stop', 'requested': 1,
        'returned': 0, 'bounded_stop': 1, 'failed_unknown': 0}
    extra['cuda_spans'] += [{'phase': 'original_2_candidate'}] * 7
    extra_cache = SimpleNamespace(selected_index=1, hits=0, started=2,
        completed=2, cache={b'first': 0, b'second': 0})
    diagnostic = e131.result_from_raw(freeze, extra, comparisons,
                                      'first_selector_feasible', extra_cache)
    assert diagnostic['status'] == 'invalid_evidence'
    assert not diagnostic['restoration_feasible']

    terminal = json.loads(json.dumps(extra))
    terminal['original_evaluations'][0].update(
        ratios=[[1.1] * 4] * 3, max_ratio=1.1, selector_feasible=False)
    terminal_cache = SimpleNamespace(selected_index=2, hits=0, started=2,
        completed=2, cache={b'first': 0, b'second': 0})
    result = e131.result_from_raw(freeze, terminal, comparisons,
                                  'first_selector_feasible', terminal_cache)
    assert result['status'] == 'completed_bounded_fit'
    assert result['selected_index'] == 2

    bounded_failure = json.loads(json.dumps(ledger))
    bounded_failure['original_evaluations'][0].update(
        ratios=[[1.1] * 4] * 3, max_ratio=1.1, selector_feasible=False)
    bounded_failure['fit_calls'] = 1
    bounded_failure['optimizer'] = {'status': 'returned', 'requested': 1,
        'returned': 1, 'bounded_stop': 0, 'failed_unknown': 0}
    no_selection = SimpleNamespace(selected_index=None, hits=0,
        started=1, completed=1, cache={b'generated': 0})
    result = e131.result_from_raw(freeze, bounded_failure, comparisons,
                                  'scipy_finished', no_selection)
    assert result['status'] == 'completed_bounded_fit'
    assert result['interpretation'] == 'bounded_failure_inconclusive'
    assert not result['restoration_feasible']


def test_failure_preserves_primary_and_partial_raw(tmp_path):
    path = tmp_path / 'leg-0-evaluation-001-partial-raw.npz'
    np.savez(path, partial=np.array([1.]))
    primary = RuntimeError('generated primary')

    def fail_cleanup():
        raise OSError('generated cleanup')

    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            e131.e130.safe_finally(('cleanup', fail_cleanup))
    assert caught.value is primary
    assert 'generated cleanup' in primary.__notes__[0]
    assert e131.e130.partial_paths(tmp_path, primary)[0]['sha256'] == (
        e131.profile.sha256(path))


def test_preflight_review_rejected_before_original_work(tmp_path):
    bad = tmp_path / 'rejected.json'
    bad.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Exact E131 freeze SHA'):
        e131.reviewed_inputs(bad, 'bad', bad, 'bad')


def test_generated_tiny_constructor_failure_preserves_request(monkeypatch, tmp_path):
    primary = RuntimeError('generated tiny constructor failure')
    primary.fixed_upload_progress = {'unknown_requested': {'count': 1}}

    def fail_constructor(*_):
        raise primary

    monkeypatch.setattr(e131.neural, 'CudaJointAdaptiveEngine', fail_constructor)
    ledger = {'synthetic_tiny': [], 'synthetic_tiny_requested': 0,
              'synthetic_tiny_completed': 0}
    with pytest.raises(RuntimeError) as caught:
        e131._run_tiny(tmp_path, ledger)
    assert caught.value is primary
    assert ledger['synthetic_tiny_requested'] == 1
    assert ledger['synthetic_tiny_completed'] == 0
    assert ledger['synthetic_tiny'][0]['status'] == 'failed'
    assert ledger['synthetic_tiny'][0]['fixed_upload_progress'] == (
        primary.fixed_upload_progress)
    assert json.loads((tmp_path / 'advance-ledger.json').read_text())[
        'synthetic_tiny_requested'] == 1


@pytest.mark.parametrize('fault', ['manifest', 'late'])
def test_finalization_fault_cannot_leave_success_receipt(monkeypatch, tmp_path, fault):
    e131.save(tmp_path / 'result.json', {'restoration_feasible': True})
    ledger = {'status': 'completed_bounded_fit'}
    errors = []

    def secondary(stage, error):
        errors.append(stage)
        return error

    if fault == 'manifest':
        def fail_manifest(_):
            raise OSError('generated manifest fault')

        monkeypatch.setattr(e131, 'write_output_manifest', fail_manifest)
        bound = 1.
    else:
        bound = -1.
    result = e131.finalize_artifacts(tmp_path, ledger,
        {'internal_wall_limit_s': bound}, time.perf_counter(), None, secondary)
    assert isinstance(result, OSError if fault == 'manifest' else TimeoutError)
    assert ledger['status'] == 'failed'
    assert not (tmp_path / 'result.json').exists()
    assert errors


def test_final_manifest_covers_generated_receipt(tmp_path):
    e131.save(tmp_path / 'result.json', {'restoration_feasible': False})
    ledger = {'status': 'completed_bounded_fit'}

    def secondary(stage, error):
        raise AssertionError(stage) from error

    assert e131.finalize_artifacts(tmp_path, ledger,
        {'internal_wall_limit_s': 1.}, time.perf_counter(), None,
        secondary) is None
    manifest = json.loads((tmp_path / 'output-manifest.json').read_text())
    assert manifest['schema'] == 'flylab.m3.e131.output-manifest.v1'
    assert set(manifest['files']) == {'advance-ledger.json', 'result.json'}
    for relative, digest in manifest['files'].items():
        assert digest == e131.profile.sha256(tmp_path / relative)
