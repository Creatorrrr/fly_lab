"""Generated CPU tests for E139; no original model or CUDA execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_cross_leg_timescale_restoration as e139


def generated_data(edges, muscles, shift=0.):
    parameters = edges + 2 * muscles
    q = np.zeros((3, 99, 7), np.float64)
    q[0, 0, 0] = .9 + .2 * shift
    jq = np.zeros((3, 99, 7, parameters), np.float64)
    jq[0, 0, 0, 0] = .2
    ltm = np.zeros((3, 99), np.float64)
    jltm = np.zeros((3, 99, parameters), np.float64)
    loss_jac = np.zeros((3, parameters), np.float64)
    loss_jac[:, 0] = [.4, -.1, .2]
    return {'delta': np.r_[np.array([shift]), np.zeros(edges - 1)],
        'gain': np.zeros(muscles), 'log_tau': np.zeros(muscles),
        'losses': np.array([1.1 + .4 * shift, 1.0 - .1 * shift,
                            1.02 + .2 * shift]),
        'loss_jac': loss_jac, 'q_error': q, 'ltm_error': ltm,
        'q_jac': jq, 'ltm_jac': jltm}


@pytest.mark.parametrize(('neurons', 'muscles', 'edges', 'parameters'), [
    (63, 15, 281, 311), (63, 16, 240, 272), (70, 17, 238, 272),
    (63, 15, 255, 285), (63, 16, 285, 317)])
def test_variable_leg_dimensions_work_and_restoration(
        neurons, muscles, edges, parameters):
    work = e139.work_per_evaluation(neurons, edges)
    assert work['scalar_MN_ticks'] == 120000 * neurons
    assert work['edge_stage_ticks'] == 480000 * edges
    result = e139.restoration_value_jac(
        generated_data(edges, muscles), 1.05, 1.05 / .99,
        edges, muscles)
    assert result['gradient'].shape == (parameters,)
    step = 1e-6
    plus = e139.restoration_value_jac(
        generated_data(edges, muscles, step), 1.05, 1.05 / .99,
        edges, muscles)
    minus = e139.restoration_value_jac(
        generated_data(edges, muscles, -step), 1.05, 1.05 / .99,
        edges, muscles)
    assert result['gradient'][0] == pytest.approx(
        (plus['value'] - minus['value']) / (2 * step), abs=1e-7)


def test_non_lf_targets_ltm_scales_and_global_moment():
    fixture = {'ltm': np.array([1, 7, 10, 13, 17, 21]),
        'own_target': np.arange(3 * 5 * 42).reshape(3, 5, 42),
        'target_feet': np.arange(3 * 5 * 6 * 3).reshape(3, 5, 6, 3),
        'joint_scale': np.arange(42) + 1.,
        'foot_scale': np.arange(6 * 3).reshape(6, 3) + 1.,
        'velocity_scale': np.arange(6 * 3).reshape(6, 3) + 21.,
        'startup_target_command': np.arange(99 * 42).reshape(99, 42),
        'startup_target_tendon': np.arange(99 * 6).reshape(99, 6),
        'moment': np.arange(42 * 30).reshape(42, 30)}
    muscles = np.array([3, 13, 18])
    selected = np.array([1, 3])
    got = e139.leg_observation_inputs(fixture, 3, 2, selected, muscles)
    assert got['joint_start'] == 21 and got['ltm_local'] == 1
    assert np.array_equal(got['q_target'], fixture['own_target'][
        2, selected, 21:28])
    assert np.array_equal(got['foot_target'], fixture['target_feet'][
        2, selected, 3])
    assert np.array_equal(got['startup_ltm_target'],
                          fixture['startup_target_tendon'][:, 3])
    assert np.array_equal(got['moment'], fixture['moment'][21:28][:, muscles])


def test_general_initial_jacobian_mapping_and_loss_contraction():
    rng = np.random.default_rng(13)
    muscles, parameters = 3, 8
    b = rng.normal(size=(3, 99, parameters))
    q_coeff = rng.normal(size=(3, 99, 7, muscles))
    ltm_coeff = rng.normal(size=(3, 99))
    mapping = np.array([2, 0, 1, 2, 1, 0, 2, 1], np.int32)
    q_jac, ltm_jac = e139.initial_jacobians(
        b, q_coeff, ltm_coeff, mapping, 1, muscles)
    for p, muscle in enumerate(mapping):
        assert np.array_equal(q_jac[..., p],
                              b[..., p, None] * q_coeff[..., muscle])
        expected = (b[..., p] * ltm_coeff if muscle == 1 else
                    np.zeros((3, 99)))
        assert np.array_equal(ltm_jac[..., p], expected)
    cotangent = rng.normal(size=(3, 99, muscles))
    answer = e139.saved_b_loss_jac(b, cotangent, mapping, muscles)
    for context in range(3):
        for parameter, muscle in enumerate(mapping):
            assert answer[context, parameter] == pytest.approx(
                b[context, :, parameter] @ cotangent[context, :, muscle])


def make_cache(evaluate, *, on_start=lambda _i, _x: None,
               on_complete=lambda _c, _d, _i: None, maximum=4,
               clock=lambda: 0.):
    start = np.zeros(8)
    return e139.LegCache(evaluate, start, np.tile([-2., 2.], (8, 1)),
        on_start, on_complete, max_unique=maximum,
        leg_deadline=40, block_deadline=210, internal_deadline=230,
        clock=clock)


def test_start_cache_hit_then_first_qualifier_is_terminal():
    calls = []
    completed = []

    def evaluate(_x, index):
        calls.append(index)
        return {'restoration_value': float(index),
            'restoration_gradient': np.zeros(8), 'qualifies': index == 2}

    cache = make_cache(evaluate, on_complete=lambda _c, _d, i:
                       completed.append(i))
    start = np.zeros(8)
    assert cache.get(start)[0] == 1
    assert cache.get(start.copy())[0] == 1
    assert cache.hits == 1
    next_point = start.copy()
    next_point[0] = .2
    with pytest.raises(e139.BoundedStop,
                       match='first_guarded_selector_feasible'):
        cache.get(next_point)
    for point in (next_point, np.full(8, .1)):
        with pytest.raises(e139.BoundedStop,
                           match='first_guarded_selector_feasible'):
            cache.get(point)
    assert calls == completed == [1, 2]
    assert cache.selected_index == 2


def test_completion_failure_removes_cache_and_latches_primary():
    calls = []

    def evaluate(_x, index):
        calls.append(index)
        return {'restoration_value': 1.,
            'restoration_gradient': np.zeros(8), 'qualifies': False}

    cache = make_cache(evaluate, on_complete=lambda *_:
        (_ for _ in ()).throw(OSError('receipt failure')))
    with pytest.raises(OSError, match='receipt failure') as failure:
        cache.get(np.zeros(8))
    assert cache.completed == 0 and cache.cache == {}
    with pytest.raises(OSError, match='receipt failure') as second:
        cache.get(np.ones(8) * .1)
    assert second.value is failure.value and calls == [1]


def test_optimizer_request_after_cached_start_only():
    events = []
    cache = make_cache(lambda _x, index: {'restoration_value': float(index),
        'restoration_gradient': np.zeros(8), 'qualifies': False})
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}

    def persist():
        events.append((leg['optimizer']['status'],
                       leg['optimizer']['requested']))

    def solver(fun, x, **kwargs):
        assert cache.started == 1 and cache.completed == 1
        assert events[-1] == ('in_flight_unknown', 1)
        assert kwargs['method'] == 'L-BFGS-B'
        fun(x)
        return SimpleNamespace(success=True, status=0, message='ok',
            nit=1, nfev=1, fun=1., x=x)

    answer, reason = e139.run_leg_optimizer(cache, leg, persist,
                                             solver=solver, clock=lambda: 1.)
    assert answer.success and reason == 'scipy_finished'
    assert cache.hits == 1
    assert events == [('intent', 0), ('in_flight_unknown', 1),
                      ('returned', 1)]
    terminal = cache.terminal_error
    assert isinstance(terminal, e139.BoundedStop)
    for point in (np.zeros(8), np.full(8, .1)):
        with pytest.raises(e139.BoundedStop) as stopped:
            cache.get(point)
        assert stopped.value is terminal
    with pytest.raises(e139.BoundedStop) as repeated:
        e139.run_leg_optimizer(cache, leg, persist,
            solver=lambda *_args, **_kwargs: pytest.fail('second solver'))
    assert repeated.value is terminal
    assert leg['optimizer']['requested'] == leg['optimizer']['returned'] == 1


def test_optimizer_receipt_failure_prevents_solver():
    cache = make_cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(8), 'qualifies': False})
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}

    def persist():
        if leg['optimizer']['status'] == 'in_flight_unknown':
            raise OSError('request receipt')

    with pytest.raises(OSError, match='request receipt') as failure:
        e139.run_leg_optimizer(cache, leg, persist,
            solver=lambda *_args, **_kwargs: pytest.fail('solver entered'))
    assert leg['optimizer']['requested'] == 0
    for point in (np.zeros(8), np.full(8, .1)):
        with pytest.raises(OSError, match='request receipt') as stopped:
            cache.get(point)
        assert stopped.value is failure.value
    with pytest.raises(OSError, match='request receipt') as repeated:
        e139.run_leg_optimizer(cache, leg, persist,
            solver=lambda *_args, **_kwargs: pytest.fail('second solver'))
    assert repeated.value is failure.value


def test_optimizer_intent_persistence_failure_closes_all_paths():
    cache = make_cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(8), 'qualifies': False})
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}
    error = OSError('intent receipt')

    def persist():
        raise error

    with pytest.raises(OSError, match='intent receipt') as failure:
        e139.run_leg_optimizer(cache, leg, persist,
            solver=lambda *_args, **_kwargs: pytest.fail('solver entered'))
    assert failure.value is error
    assert leg['optimizer']['requested'] == 0
    for operation in (
            lambda: cache.get(np.zeros(8)),
            lambda: cache.get(np.full(8, .1)),
            lambda: e139.run_leg_optimizer(cache, leg, persist,
                solver=lambda *_args, **_kwargs: pytest.fail('second solver'))):
        with pytest.raises(OSError, match='intent receipt') as stopped:
            operation()
        assert stopped.value is error


def test_optimizer_solver_error_closes_all_paths_and_preserves_identity():
    cache = make_cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(8), 'qualifies': False})
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}
    error = ArithmeticError('solver primary')
    entries = []

    def solver(*_args, **_kwargs):
        entries.append(1)
        raise error

    with pytest.raises(ArithmeticError, match='solver primary') as failure:
        e139.run_leg_optimizer(cache, leg, lambda: None, solver=solver)
    assert failure.value is error
    assert leg['optimizer']['requested'] == 1
    assert leg['optimizer']['returned'] == 0
    assert leg['optimizer']['failed_unknown'] == 1
    for operation in (
            lambda: cache.get(np.zeros(8)),
            lambda: cache.get(np.full(8, .1)),
            lambda: e139.run_leg_optimizer(cache, leg, lambda: None,
                                            solver=solver)):
        with pytest.raises(ArithmeticError, match='solver primary') as stopped:
            operation()
        assert stopped.value is error
    assert entries == [1]


def test_optimizer_return_persistence_failure_closes_with_known_return():
    cache = make_cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(8), 'qualifies': False})
    leg = {'optimizer': {'status': 'not_requested', 'intent_persisted': False,
        'requested': 0, 'returned': 0, 'bounded_stop': 0,
        'failed_unknown': 0}}
    error = OSError('return receipt')
    solver_entries = []

    def persist():
        if leg['optimizer']['status'] == 'returned':
            raise error

    def solver(_fun, x, **_kwargs):
        solver_entries.append(1)
        return SimpleNamespace(success=True, status=0, message='ok',
            nit=1, nfev=1, fun=1., x=x)

    with pytest.raises(OSError, match='return receipt') as failure:
        e139.run_leg_optimizer(cache, leg, persist, solver=solver)
    assert failure.value is error
    assert leg['optimizer']['requested'] == leg['optimizer']['returned'] == 1
    assert leg['optimizer']['failed_unknown'] == 0
    for operation in (
            lambda: cache.get(np.zeros(8)),
            lambda: cache.get(np.full(8, .1)),
            lambda: e139.run_leg_optimizer(cache, leg, persist, solver=solver)):
        with pytest.raises(OSError, match='return receipt') as stopped:
            operation()
        assert stopped.value is error
    assert solver_entries == [1]


def test_dimension_aware_requested_known_and_unknown_work():
    maximum = e139.work_per_evaluation(70, 238)
    progress = {'forward_completion_class': 'known_complete',
        'decoder_completion_class': 'known_complete',
        'bank_completion_class': 'unknown_enqueued',
        'derivative_bank_complete': False,
        'completed_context_rk4_ticks': 120000,
        'completed_conditional_model_s': 30,
        'completed_decoder_endpoint_ticks': 6000,
        'kernel_events': [{'name': key, 'requested': True}
                          for key in ('forward', 'decoder', 'bank')]}
    row = {'solve_requested': True, 'solve_returned': False,
        'raw_saved': False, 'post_return_error': False,
        'intent_persisted': True, 'neurons': 70,
        'work_per_evaluation': maximum, 'response_progress': progress}
    ledger = {'original_evaluations': [row], 'legs': []}
    e139.refresh_counts(ledger)
    assert ledger['original_scalar_MN_ticks_known_complete'] == 8400000
    assert ledger['original_edge_stage_ticks_requested'] == 114240000
    assert ledger['original_edge_stage_ticks_known_complete'] == 0
    assert ledger['original_edge_stage_ticks_unresolved_upper_bound'] == 114240000


def test_replay_prefix_and_local_tendon_comparison(tmp_path):
    edges, muscles = 3, 2
    common = {'delta': np.arange(edges, dtype=np.float64),
        'gain': np.arange(muscles, dtype=np.float64),
        'edge_ids': np.arange(edges, dtype=np.int64),
        'motor_indices': np.arange(4, dtype=np.int64),
        'muscle_indices': np.arange(muscles, dtype=np.int64),
        'parameter_muscle_map': np.array([0, 1, 0, 0, 1], np.int32),
        'B_initial99': np.ones((3, 99, edges + muscles)),
        'loss_jac': np.ones((3, edges + muscles))}
    common.update({key: np.ones((1,), np.float64) for key in e139.OLD_METRIC
                   if key != 'loss_jac'})
    reference = tmp_path / 'start.npz'
    np.savez(reference, **common)
    actual_values = dict(common)
    actual_values['log_tau'] = np.zeros(muscles)
    actual_values['parameter_muscle_map'] = np.r_[
        common['parameter_muscle_map'], np.arange(muscles, dtype=np.int32)]
    actual_values['B_initial99'] = np.concatenate((
        common['B_initial99'], np.full((3, 99, muscles), 7.)), axis=2)
    actual_values['loss_jac'] = np.c_[common['loss_jac'],
                                      np.full((3, muscles), 8.)]
    actual = tmp_path / 'actual.npz'
    np.savez(actual, **actual_values)
    assert e139.compare_replay(actual, reference, baseline=False,
        edges=edges, muscles=muscles, leg=4)['accepted'] is True


def test_baseline_replay_uses_global_command_and_local_tendon(tmp_path):
    edges, muscles, leg = 2, 3, 5
    reference_values = {'delta': np.zeros(edges), 'gain': np.zeros(muscles),
        'edge_ids': np.arange(edges, dtype=np.int64),
        'motor_indices': np.arange(4, dtype=np.int64),
        'muscle_indices': np.arange(muscles, dtype=np.int64),
        'parameter_muscle_map': np.array([0, 1, 0, 1, 2], np.int32),
        'branch_bits': np.array([1], np.uint8),
        'B_initial99': np.ones((3, 99, edges + muscles)),
        'loss_jac': np.ones((3, edges + muscles)),
        'command': np.arange(42, dtype=np.float64)[None, None, :],
        'tendon': np.ones((1, 1, 1), np.float64)}
    for key in e139.fit.FIELDS:
        reference_values.setdefault(key, np.ones((1,), np.float64))
    for key in e139.OLD_METRIC:
        reference_values.setdefault(key, np.ones((1,), np.float64))
    actual_values = dict(reference_values)
    actual_values['parameter_muscle_map'] = np.r_[
        reference_values['parameter_muscle_map'],
        np.arange(muscles, dtype=np.int32)]
    actual_values['B_initial99'] = np.concatenate((
        reference_values['B_initial99'],
        np.ones((3, 99, muscles))), axis=2)
    actual_values['loss_jac'] = np.c_[reference_values['loss_jac'],
                                      np.ones((3, muscles))]
    actual_values['log_tau'] = np.zeros(muscles)
    reference = tmp_path / 'baseline.npz'
    actual = tmp_path / 'actual.npz'
    np.savez(reference, **reference_values)
    np.savez(actual, **actual_values)
    report = e139.compare_replay(actual, reference, baseline=True,
        edges=edges, muscles=muscles, leg=leg)
    assert report['accepted'] is True
    assert report['fields']['command']['accepted'] is True
    assert report['fields']['tendon']['accepted'] is True


def test_leg_result_requires_terminal_first_qualifier_with_hashed_raw(tmp_path):
    baseline = 1.2
    rows = []
    for local_index, (objective, selector, qualifies) in enumerate((
            (baseline, True, False), (1.17, False, False),
            (1.16, True, True)), 1):
        losses = np.array([objective - .01, objective - .02,
                           objective - .03])
        ratios = np.zeros((3, 4)) if selector else np.ones((3, 4))
        parts = np.zeros((3, 4))
        path = tmp_path / f'raw-{local_index}.npz'
        np.savez(path, losses=losses, ratios=ratios, parts=parts)
        rows.append({'index': local_index, 'local_index': local_index,
            'leg': 1, 'status': 'raw_saved', 'raw_saved': True,
            'solve_requested': True, 'solve_returned': True,
            'post_return_error': False, 'raw_path': str(path),
            'raw_sha256': e139.fit.sha256(path), 'losses': losses.tolist(),
            'ratios': ratios.tolist(), 'parts': parts.tolist(),
            'prior': .01, 'true_worst_objective': objective,
            'selector_feasible': selector, 'restoration_value': .1,
            'gradient_norm': .01, 'guard3': [0., 0., 0.],
            'guard_ratios': [1., 1., 1.],
            'objective_over_baseline': objective / baseline,
            'qualifies': qualifies})
    spec = {'leg': 1, 'name': 'lm', 'baseline_objective': baseline,
            'start_objective': 1.17}
    cache = SimpleNamespace(started=2, completed=2,
        cache={1: None, 2: None}, hits=1, selected_index=2)
    comparisons = {label: {'accepted': True,
        'actual_sha256': rows[index]['raw_sha256']}
        for label, index in (('baseline', 0), ('start', 1))}
    leg = {'optimizer': {'status': 'bounded_stop', 'requested': 1,
        'returned': 0, 'bounded_stop': 1, 'failed_unknown': 0},
        'comparison': comparisons, 'stop_reason':
            'first_guarded_selector_feasible', 'wall_s': 2.}
    result = e139.leg_result(spec, rows, cache, leg)
    assert result['hypothesis_verdict'] == 'supported'
    assert result['selected_local_index'] == 3
    extra = dict(rows[1])
    extra.update(index=4, local_index=4)
    rows.append(extra)
    cache.started = cache.completed = 3
    cache.cache[3] = None
    assert e139.leg_result(spec, rows, cache, leg)[
        'hypothesis_verdict'] == 'not_tested'
    rows.pop()
    cache.started = cache.completed = 2
    cache.cache.pop(3)
    rows[-1]['qualifies'] = False
    cache.selected_index = None
    assert e139.leg_result(spec, rows, cache, leg)[
        'hypothesis_verdict'] == 'not_tested'


def test_preflight_rejects_without_reading_missing_file(tmp_path):
    with pytest.raises(ValueError, match='Exact E139 freeze SHA'):
        e139.reviewed_inputs(tmp_path / 'missing', '0' * 64,
                             tmp_path / 'review', '0' * 64)


def test_finalization_preserves_primary_and_revokes_success(tmp_path,
                                                            monkeypatch):
    (tmp_path / 'result.json').write_text(json.dumps({'status': 'completed'}))
    ledger = {'status': 'completed', 'original_evaluations': [], 'legs': [],
        'primary_error': {'traceback': 'original',
                          'response_progress': {'context': 2}}}
    original = e139.write_output_manifest

    def fail_once(out):
        monkeypatch.setattr(e139, 'write_output_manifest', original)
        raise OSError('manifest')

    monkeypatch.setattr(e139, 'write_output_manifest', fail_once)
    error = RuntimeError('primary')
    assert e139.finalize(tmp_path, ledger, None, 0., error) is error
    assert not (tmp_path / 'result.json').exists()
    saved = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert saved['primary_error']['traceback'] == 'original'
    assert saved['primary_error']['response_progress'] == {'context': 2}
