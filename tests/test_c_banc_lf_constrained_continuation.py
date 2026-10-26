"""Generated CPU checks for E132 warm start, selection and bounded evidence."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_lf_constrained_continuation as e132


def generated_data(x):
    x = np.asarray(x, np.float64)
    losses = np.array([.2 + (x[0] - .1)**2,
                       .19 + (x[1] - .1)**2, .18])
    loss_jac = np.array([[2 * (x[0] - .1), 0.],
                         [0., 2 * (x[1] - .1)], [0., 0.]])
    return {'delta': x[:1].copy(), 'gain': x[1:].copy(),
            'losses': losses, 'loss_jac': loss_jac,
            'q_error': np.zeros((3, 99, 7)),
            'ltm_error': np.zeros((3, 99)),
            'q_jac': np.zeros((3, 99, 7, 2)),
            'ltm_jac': np.zeros((3, 99, 2)),
            'ratios': np.zeros((3, 4)),
            'parts': np.zeros((3, 4)),
            'prior': 0., 'true_worst_objective': float(max(losses))}


def make_cache(evaluate=None, *, max_calls=40, deadline=None):
    now = time.perf_counter()
    cache = e132.fit.EvaluationCache(evaluate or (
        lambda x, _: generated_data(x)), 2, lambda *_: None,
        max_calls=max_calls, deadline=deadline or now + 30,
        total_deadline=now + 30)
    cache.bounds = np.array([[0., 1.], [0., 1.]])
    return cache


def test_feasible_warm_start_uses_original_slsqp_and_continues():
    seen = []

    def evaluate(x, index):
        seen.append((index, x.copy()))
        return generated_data(x)

    cache = make_cache(evaluate)
    accounting = {}
    start = np.array([.5, .5])
    solver, stop = e132.run_optimizer(cache, start, accounting)
    assert stop == 'scipy_finished'
    assert solver.success
    assert (accounting['requested'], accounting['returned'],
            accounting['bounded_stop'], accounting['failed_unknown']) == (1, 1, 0, 0)
    assert np.array_equal(seen[0][1], start)
    assert len(seen) > 1  # Feasible start does not stop the original objective.
    assert cache.started == cache.completed == len(seen)
    assert solver.fun < max(generated_data(start)['losses'])
    assert accounting['cache_hits'] > 0


def test_epigraph_t_is_excluded_from_cache_and_original_math():
    cache = make_cache()
    x = np.array([.4, .3])
    a, t1 = cache.get(np.r_[x, .4])
    b, t2 = cache.get(np.r_[x, .5])
    assert a is b and (t1, t2) == (.4, .5)
    assert cache.started == cache.completed == 1 and cache.hits == 1
    value1, jac1, prior = e132.fit.objective_and_jac(a, t1, 1)
    value2, jac2, _ = e132.fit.objective_and_jac(b, t2, 1)
    assert value2 - value1 == pytest.approx(.1)
    assert prior >= 0 and np.array_equal(jac1, jac2)
    c, j, _ = e132.fit.constraint_values(a['losses'], a['loss_jac'],
        a['q_error'], a['ltm_error'], a['q_jac'], a['ltm_jac'], t1)
    assert c.shape == (4761,) and j.shape == (4761, 3)
    assert np.all(j[:3, -1] == 1)


def test_initial_replay_mismatch_requests_no_optimizer(monkeypatch):
    cache = make_cache(lambda x, _: generated_data(x) |
        {'ratios': np.full((3, 4), 1.1)})
    called = []
    monkeypatch.setattr(e132.fit, 'solve_one_leg',
                        lambda *_: called.append(True))
    accounting = {}
    with pytest.raises(ValueError, match='selector-feasible'):
        e132.run_optimizer(cache, np.array([.4, .3]), accounting)
    assert called == [] and accounting['requested'] == 0
    assert accounting['cache_started'] == accounting['cache_completed'] == 1
    assert accounting['elapsed_wall_s'] >= 0


def test_optimizer_bounded_stop_and_failure_accounting(monkeypatch):
    cache = make_cache(max_calls=1)
    accounting = {}
    solver, stop = e132.run_optimizer(cache, np.array([.4, .3]), accounting)
    assert solver is None and stop == 'per_leg_unique_or_wall_limit'
    assert (accounting['requested'], accounting['returned'],
            accounting['bounded_stop'], accounting['failed_unknown']) == (1, 0, 1, 0)
    assert cache.started == cache.completed == 1

    primary = RuntimeError('generated optimizer failure')

    def fail(*_):
        raise primary

    monkeypatch.setattr(e132.fit, 'solve_one_leg', fail)
    failed = {}
    with pytest.raises(RuntimeError) as caught:
        e132.run_optimizer(make_cache(), np.array([.4, .3]), failed)
    assert caught.value is primary
    assert (failed['requested'], failed['returned'], failed['bounded_stop'],
            failed['failed_unknown'], failed['status']) == (
                1, 0, 0, 1, 'failed_unknown')
    assert failed['cache_started'] == failed['cache_completed'] == 1
    assert failed['elapsed_wall_s'] >= 0


def test_unique_and_wall_limits_preserve_initial_progress():
    cache = make_cache(max_calls=1)
    cache.get([.4, .3, .4])
    with pytest.raises(e132.fit.FitLimit):
        cache.get([.3, .3, .4])
    assert cache.started == cache.completed == 1
    soon = make_cache(deadline=time.perf_counter() + 1)
    with pytest.raises(e132.fit.FitLimit):
        soon.get([.4, .3, .4])
    assert soon.started == soon.completed == 0


def synthetic_row(index, objective, *, feasible=True):
    ratios = [[.5] * 4] * 3 if feasible else [[1.1] * 4] * 3
    return {'index': index, 'phase': f'original_{index}_candidate',
        'backend': 'candidate', 'status': 'raw_saved',
        'raw_path': 'generated', 'raw_sha256': f'sha-{index}',
        'ratios': ratios, 'max_ratio': .5 if feasible else 1.1,
        'selector_feasible': feasible, 'losses': [objective] * 3,
        'parts': [[0.] * 4] * 3, 'prior': 0.,
        'true_worst_objective': objective, 'evaluate_wall_s': 1.}


def synthetic_ledger(rows, *, bounded=False):
    count = len(rows)
    return {'original_evaluations': rows,
        'original_evaluations_requested': count,
        'original_evaluations_completed': count,
        'synthetic_tiny': [{'status': 'completed', 'raw_sha256': 'tiny'}],
        'synthetic_tiny_requested': 1, 'synthetic_tiny_completed': 1,
        'cuda_spans': [{'phase': row['phase']} for row in rows for _ in range(7)],
        'cache_started': count, 'cache_completed': count, 'cache_hits': 3,
        'fit_calls': 1, 'fit_wall_s': 10., 'secondary_errors': [],
        'optimizer': {'status': 'bounded_stop' if bounded else 'returned',
            'requested': 1, 'returned': 0 if bounded else 1,
            'bounded_stop': 1 if bounded else 0, 'failed_unknown': 0}}


def synthetic_result(monkeypatch, rows, *, bounded=False, ledger=None,
                     comparison_valid=True):
    monkeypatch.setattr(e132.e131, '_complete_evaluation', lambda _: True)
    monkeypatch.setattr(e132.e130, 'known_progress', lambda *_a, **_k: True)
    monkeypatch.setattr(e132.e130, 'known_adapter', lambda *_: True)
    monkeypatch.setattr(e132.e130, 'known_metadata', lambda *_: True)
    monkeypatch.setattr(e132.e130, 'complete_span', lambda *_: 1.)
    monkeypatch.setattr(e132, '_parts', lambda _: [[0.] * 4] * 3)
    ledger = ledger or synthetic_ledger(rows, bounded=bounded)
    cache = SimpleNamespace(started=len(rows), completed=len(rows), hits=3,
                            cache={bytes([i]): None for i in range(len(rows))})
    comparison = {'accepted': comparison_valid, 'reference_sha256': 'reference',
                  'actual_sha256': rows[0]['raw_sha256']}
    freeze = {'input_sha256': {'start': 'reference'}, 'start_raw_path': 'start',
              'zero_raw_path': 'zero', 'proposal_objective_ratio_limit': .99,
              'fit_wall_limit_s': 55}
    return e132.result_from_raw(freeze, ledger, comparison,
        'per_leg_unique_or_wall_limit' if bounded else 'scipy_finished', cache)


def test_best_feasible_selection_earliest_tie_and_one_percent(monkeypatch):
    rows = [synthetic_row(1, 1.), synthetic_row(2, .8, feasible=False),
            synthetic_row(3, .985), synthetic_row(4, .985),
            synthetic_row(5, .995)]
    result = synthetic_result(monkeypatch, rows)
    assert result['status'] == 'completed_bounded_fit'
    assert result['selected_index'] == 3
    assert result['selected_objective'] == .985
    assert result['best_any_index'] == 2
    assert result['new_lf_proposal_accepted']
    assert not result['whole_candidate_adopted']
    below = synthetic_result(monkeypatch, [synthetic_row(1, 1.),
        synthetic_row(2, .995)], bounded=True)
    assert below['selected_index'] == 2
    assert not below['new_lf_proposal_accepted']
    start = synthetic_result(monkeypatch, [synthetic_row(1, 1.),
        synthetic_row(2, 1.1)])
    assert start['selected_index'] == 1
    assert not start['new_lf_proposal_accepted']


@pytest.mark.parametrize('damage', ['partial', 'unknown', 'overrun',
    'cache', 'optimizer', 'comparison', 'first_infeasible'])
def test_invalid_evidence_cannot_accept_proposal(monkeypatch, damage):
    rows = [synthetic_row(1, 1.), synthetic_row(2, .8)]
    ledger = synthetic_ledger(rows)
    if damage == 'partial':
        ledger['original_evaluations_completed'] = 1
    elif damage == 'unknown':
        ledger['secondary_errors'] = [{'stage': 'generated unknown'}]
    elif damage == 'overrun':
        ledger['fit_wall_s'] = 55.1
    elif damage == 'cache':
        ledger['cache_completed'] = 1
    elif damage == 'optimizer':
        ledger['optimizer']['requested'] = 0
    else:
        if damage == 'first_infeasible':
            rows[0].update(ratios=[[1.1] * 4] * 3, max_ratio=1.1,
                           selector_feasible=False)
    result = synthetic_result(monkeypatch, rows, ledger=ledger,
        comparison_valid=damage != 'comparison')
    assert not result['new_lf_proposal_accepted']
    assert result['status'] == ('invalid_budget' if damage == 'overrun'
                                else 'invalid_evidence')
    json.dumps(result, allow_nan=False)


def test_full_raw_comparison_excludes_only_execution_provenance(tmp_path):
    reference = tmp_path / 'reference.npz'
    actual = tmp_path / 'actual.npz'
    payload = {'delta': np.array([.1], np.float64),
        'branch_bits': np.array([1, 0], np.uint8),
        'losses': np.array([1.], np.float64),
        'rate': np.array([.2], np.float64),
        'transfer_cost_json': np.array('old')}
    np.savez(reference, **payload)
    np.savez(actual, **(payload | {'losses': np.array([1. + 5e-11]),
        'rate': np.array([.2 + 5e-9]),
        'transfer_cost_json': np.array('new')}))
    assert e132.compare_full_raw(actual, reference)['accepted']
    np.savez(actual, **(payload | {'branch_bits': np.array([0, 0], np.uint8)}))
    assert not e132.compare_full_raw(actual, reference)['accepted']


def test_preflight_rejected_before_original_work(tmp_path):
    bad = tmp_path / 'rejected.json'
    bad.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Exact E132 freeze SHA'):
        e132.reviewed_inputs(bad, 'bad', bad, 'bad')


def test_finalization_failure_removes_success_receipt(monkeypatch, tmp_path):
    e132.save(tmp_path / 'result.json', {'new_lf_proposal_accepted': True})
    ledger = {'status': 'completed_bounded_fit'}
    monkeypatch.setattr(e132, 'write_output_manifest',
                        lambda _: (_ for _ in ()).throw(OSError('generated failure')))
    errors = []

    def secondary(stage, error):
        errors.append(stage)
        return error

    result = e132.finalize_artifacts(tmp_path, ledger,
        {'internal_wall_limit_s': 64}, time.perf_counter(), None, secondary)
    assert isinstance(result, OSError)
    assert not (tmp_path / 'result.json').exists()
    assert errors
