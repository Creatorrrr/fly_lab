"""Generated CPU checks for E137; no frozen original array is evaluated."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import fit_c_banc_m3_timescale_restoration as e137


def generated_data(shift=0.):
    q = np.zeros((3, 99, 7), np.float64)
    q[0, 0, 0] = .9 + .3 * shift
    jq = np.zeros((3, 99, 7, 307), np.float64)
    jq[0, 0, 0, 0] = .3
    ltm = np.zeros((3, 99), np.float64)
    ltm[1, 0] = .2 - .1 * shift
    jltm = np.zeros((3, 99, 307), np.float64)
    jltm[1, 0, 0] = -.1
    loss_jac = np.zeros((3, 307), np.float64)
    loss_jac[:, 0] = [.5, -.2, .1]
    return {'delta': np.r_[np.array([shift]), np.zeros(272)],
        'gain': np.zeros(17), 'log_tau': np.zeros(17),
        'losses': np.array([1.1 + .5 * shift, 1.0 - .2 * shift,
                            1.02 + .1 * shift]),
        'loss_jac': loss_jac, 'q_error': q, 'ltm_error': ltm,
        'q_jac': jq, 'ltm_jac': jltm}


def cache(make_record, *, maximum=3, clock=lambda: 0., on_start=None,
          on_complete=None):
    bounds = np.tile([-2., 2.], (307, 1))
    return e137.RestorationCache(make_record, np.zeros(307), bounds,
        on_start or (lambda _i, _x: None),
        on_complete or (lambda _c, _d, _i: None),
        max_unique=maximum, fit_deadline=100., internal_deadline=100.,
        clock=clock)


def test_restoration_guard_chain_rule_and_original_denominator():
    baseline, guard = 1.2, 1.188
    result = e137.restoration_value_jac(generated_data(), guard, baseline)
    step = 1e-6
    plus = e137.restoration_value_jac(generated_data(step), guard, baseline)
    minus = e137.restoration_value_jac(generated_data(-step), guard, baseline)
    numeric = (plus['value'] - minus['value']) / (2 * step)
    assert result['value'] > 0
    assert result['gradient'][0] == pytest.approx(numeric, abs=1e-8)
    assert np.array_equal(result['gradient'][290:], np.zeros(17))
    assert result['guard3'][0] == pytest.approx(
        (guard - 1.1) / guard)
    assert result['objective_over_baseline'] == pytest.approx(1.1 / baseline)
    assert result['qualifies'] is False


def test_guard_derivative_includes_prior_and_tau_column(monkeypatch):
    data = generated_data()
    data['delta'][0] = 2.
    data['loss_jac'][:, 290] = [.8, -.4, .2]
    seen = {}
    original = fit.constraint_values

    def intercept(*args):
        c, jac, ratios = original(*args)
        seen['c'] = c.copy()
        seen['jac'] = jac.copy()
        return c, jac, ratios

    monkeypatch.setattr(fit, 'constraint_values', intercept)
    result = e137.restoration_value_jac(data, 1.0, 1 / .99)
    prior = 6e-5 * 4 / 1572
    assert result['guard3'][0] == pytest.approx(1. - 1.1 - prior)
    assert result['gradient'].shape == (307,)
    assert seen['c'].shape == (4761,)
    assert seen['jac'].shape == (4761, 308)
    assert result['gradient'][290] == pytest.approx(
        -(min(result['guard3'][0], 0) * .8 +
          min(result['guard3'][1], 0) * -.4 +
          min(result['guard3'][2], 0) * .2))


def test_startup_only_feasible_continues_then_guarded_first_stops():
    completed = []
    evaluated = []

    def evaluate(x, index):
        evaluated.append(index)
        return {'restoration_value': float(index),
            'restoration_gradient': np.zeros(307),
            'qualifies': index == 3}

    subject = cache(evaluate, on_complete=lambda _c, _d, i:
                    completed.append(i))
    for index in (1, 2):
        x = np.zeros(307)
        x[0] = index - 1
        subject.get(x)
    assert completed == [1, 2]
    assert subject.selected_index is None
    x = np.zeros(307)
    x[0] = 1.5
    with pytest.raises(e137.BoundedStop,
                       match='first_guarded_selector_feasible'):
        subject.get(x)
    assert completed == [1, 2, 3]
    assert subject.selected_index == 3
    assert subject.completed == 3
    for vector in (x.copy(), np.full(307, .2)):
        with pytest.raises(e137.BoundedStop,
                           match='first_guarded_selector_feasible'):
            subject.get(vector)
    assert evaluated == [1, 2, 3]


def test_byte_cache_and_cap_and_deadline():
    calls = []
    subject = cache(lambda x, i: (calls.append(i) or {
        'restoration_value': 1., 'restoration_gradient': np.zeros(307),
        'qualifies': False}), maximum=1)
    a = np.zeros(307)
    assert subject.get(a)[0] == 1.
    assert subject.get(a.copy())[0] == 1.
    assert calls == [1] and subject.hits == 1
    b = a.copy()
    b[0] = .1
    with pytest.raises(e137.BoundedStop, match='unique_cap'):
        subject.get(b)
    wall = cache(lambda *_: pytest.fail('must not evaluate'),
                 clock=lambda: 97.)
    with pytest.raises(e137.BoundedStop, match='pre_call_wall_reserve'):
        wall.get(a)


def test_intent_failure_does_not_evaluate():
    def fail_intent(_i, _x):
        raise OSError('intent disk failure')

    subject = cache(lambda *_: pytest.fail('no evaluator call'),
                    on_start=fail_intent)
    with pytest.raises(OSError, match='intent disk failure'):
        subject.get(np.zeros(307))
    assert subject.started == subject.completed == 0


def test_complete_persistence_failure_cannot_support():
    evaluated = []

    def evaluate(_x, index):
        evaluated.append(index)
        return {'restoration_value': 0.,
            'restoration_gradient': np.zeros(307), 'qualifies': index == 2}

    def complete(_cache, _data, index):
        if index == 2:
            raise OSError('row disk')

    subject = cache(evaluate, on_complete=complete)
    x = np.zeros(307)
    subject.get(x)
    x[0] = 1.
    with pytest.raises(OSError, match='row disk'):
        subject.get(x)
    assert subject.completed == 1
    assert subject.selected_index is None
    for vector in (x.copy(), np.full(307, .2)):
        with pytest.raises(OSError, match='row disk'):
            subject.get(vector)
    assert evaluated == [1, 2]


def test_actual_completion_receipt_failure_preserves_raw_and_latches(
        tmp_path):
    raw_path = tmp_path / 'generated-raw.npz'
    losses = np.array([1.19, 1.18, 1.17])
    ratios = np.zeros((3, 4))
    parts = np.zeros((3, 4))
    np.savez(raw_path, losses=losses, ratios=ratios, parts=parts)
    row = {'index': 1, 'status': 'raw_saved', 'raw_saved': True,
        'solve_requested': True, 'solve_returned': True,
        'raw_path': str(raw_path), 'raw_sha256': fit.sha256(raw_path),
        'losses': losses.tolist(), 'ratios': ratios.tolist(),
        'parts': parts.tolist(), 'prior': .01,
        'true_worst_objective': 1.2, 'selector_feasible': True}
    ledger = {'original_evaluations': [row]}
    attempted = []

    def persist():
        attempted.append(row['status'])
        if len(attempted) == 1:
            raise OSError('completion disk failure')
        e137.save(tmp_path / 'advance-ledger.json', ledger)

    evaluated = []

    def evaluate(_x, index):
        evaluated.append(index)
        return {'restoration_value': .1,
            'restoration_gradient': np.zeros(307), 'qualifies': False}

    subject = cache(evaluate, on_complete=lambda c, _d, i:
        e137.complete_restoration_receipt(ledger, c, i, persist))
    with pytest.raises(OSError, match='completion disk failure'):
        subject.get(np.zeros(307))
    assert attempted == ['raw_saved', 'failed']
    assert subject.completed == 0 and subject.cache == {}
    assert row['post_return_error'] is True
    assert row['raw_saved'] is True and row['solve_returned'] is True
    assert row['raw_sha256'] == fit.sha256(raw_path)
    e137.e136.refresh_counts(ledger)
    assert ledger['original_evaluations_requested'] == 1
    assert ledger['original_evaluations_returned'] == 1
    assert ledger['original_evaluations_completed'] == 1
    assert ledger['original_evaluations_post_return_error'] == 1
    persisted = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert persisted['original_evaluations'][0]['status'] == 'failed'
    with pytest.raises(OSError, match='completion disk failure'):
        subject.get(np.full(307, .1))
    assert evaluated == [1]


def test_secondary_receipt_write_error_does_not_replace_primary(tmp_path):
    raw_path = tmp_path / 'generated-raw.npz'
    np.savez(raw_path, losses=np.ones(3))
    row = {'index': 1, 'status': 'raw_saved', 'raw_saved': True,
        'solve_requested': True, 'solve_returned': True,
        'raw_path': str(raw_path), 'raw_sha256': fit.sha256(raw_path),
        'losses': [1., 1., 1.], 'parts': np.zeros((3, 4)).tolist(),
        'ratios': np.zeros((3, 4)).tolist(), 'prior': 0.,
        'true_worst_objective': 1., 'selector_feasible': True}
    ledger = {'original_evaluations': [row]}
    attempts = 0

    def persist():
        nonlocal attempts
        attempts += 1
        raise OSError('primary receipt' if attempts == 1 else
                      'secondary receipt')

    subject = cache(lambda *_: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(307), 'qualifies': False},
        on_complete=lambda c, _d, i:
            e137.complete_restoration_receipt(ledger, c, i, persist))
    with pytest.raises(OSError, match='primary receipt') as failure:
        subject.get(np.zeros(307))
    assert attempts == 2
    assert row['status'] == 'failed' and row['post_return_error'] is True
    assert any('secondary receipt' in note for note in failure.value.__notes__)
    with pytest.raises(OSError, match='primary receipt'):
        subject.get(np.full(307, .1))


def test_optimizer_request_receipt_is_durable_before_entry():
    records = []
    subject = cache(lambda _x, i: {'restoration_value': float(i),
        'restoration_gradient': np.zeros(307), 'qualifies': False})
    ledger = {'optimizer': {'status': 'not_requested', 'requested': 0,
        'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0},
        'fit_calls': 0}

    def persist():
        records.append((ledger['optimizer']['status'],
                        ledger['optimizer']['requested']))

    def solver(fun, x, **kwargs):
        assert kwargs['method'] == 'L-BFGS-B'
        assert kwargs['options'] == e137.OPTIONS
        assert records[-1] == ('in_flight_unknown', 1)
        fun(x)
        return SimpleNamespace(success=True, status=0, message='ok',
            nit=1, nfev=1, fun=1., x=x)

    answer, reason = e137.run_optimizer(subject, ledger, persist,
                                         solver=solver, clock=lambda: 1.)
    assert answer.success and reason == 'scipy_finished'
    assert records == [('intent', 0), ('in_flight_unknown', 1),
                       ('returned', 1)]
    assert ledger['fit_calls'] == 1


def test_optimizer_request_persistence_failure_prevents_entry():
    subject = cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(307), 'qualifies': False})
    ledger = {'optimizer': {'status': 'not_requested', 'requested': 0,
        'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0},
        'fit_calls': 0}

    def persist():
        if ledger['optimizer']['status'] == 'in_flight_unknown':
            raise OSError('receipt write failed')

    with pytest.raises(OSError, match='receipt write failed'):
        e137.run_optimizer(subject, ledger, persist,
            solver=lambda *_args, **_kwargs: pytest.fail('solver entry'))
    assert ledger['optimizer']['requested'] == 0
    assert ledger['fit_calls'] == 0


def test_optimizer_invalid_return_is_known_but_not_success():
    subject = cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(307), 'qualifies': False})
    ledger = {'optimizer': {'status': 'not_requested', 'requested': 0,
        'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0},
        'fit_calls': 0}

    def solver(_fun, x, **_kwargs):
        return SimpleNamespace(success=False, status=2, message='numerical',
            nit=1, nfev=1, fun=1., x=x)

    with pytest.raises(ValueError, match='invalid numerical state'):
        e137.run_optimizer(subject, ledger, lambda: None, solver=solver)
    assert ledger['optimizer']['returned'] == 1
    assert ledger['optimizer']['failed_unknown'] == 0


def test_optimizer_unique_cap_is_known_bounded_stop():
    subject = cache(lambda _x, _i: {'restoration_value': 1.,
        'restoration_gradient': np.zeros(307), 'qualifies': False},
        maximum=1)
    ledger = {'optimizer': {'status': 'not_requested', 'requested': 0,
        'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0},
        'fit_calls': 0}

    def solver(fun, x, **_kwargs):
        changed = x.copy()
        changed[0] = .01
        fun(changed)
        pytest.fail('bound should interrupt solver')

    answer, reason = e137.run_optimizer(subject, ledger, lambda: None,
                                         solver=solver)
    assert answer is None and reason == 'unique_cap'
    assert ledger['optimizer']['requested'] == 1
    assert ledger['optimizer']['bounded_stop'] == 1
    assert ledger['optimizer']['failed_unknown'] == 0


def test_generated_first_raw_parity(tmp_path):
    values = {key: np.array([1.], np.float64) for key in
              (*e137.RESPONSE_RAW, *e137.METRIC_RAW)}
    values.update({key: np.array([2], np.int64) for key in e137.EXACT_RAW})
    first = tmp_path / 'first.npz'
    copy = tmp_path / 'copy.npz'
    np.savez(first, **values)
    np.savez(copy, **values)
    assert e137.compare_start_raw(first, copy)['accepted'] is True
    values['branch_bits'] = np.array([3], np.int64)
    np.savez(copy, **values)
    report = e137.compare_start_raw(first, copy)
    assert report['accepted'] is False
    assert report['fields']['branch_bits']['accepted'] is False


def test_preflight_exact_freeze_rejects_before_input_read(tmp_path):
    with pytest.raises(ValueError, match='Exact E137 freeze SHA'):
        e137.reviewed_inputs(tmp_path / 'absent.json', '0' * 64,
                             tmp_path / 'absent-review.json', '0' * 64)


def test_result_selects_first_jointly_qualifying_hashed_raw(tmp_path):
    baseline = 1.2
    records = []
    for index, (objective, selector, qualifies) in enumerate((
            (1.16, False, False), (1.19, True, False),
            (1.17, True, True)), 1):
        ratios = np.zeros((3, 4)) if selector else np.ones((3, 4))
        losses = np.array([objective - .01, objective - .02,
                           objective - .03])
        parts = np.zeros((3, 4))
        raw_path = tmp_path / f'generated-{index}.npz'
        np.savez(raw_path, losses=losses, parts=parts, ratios=ratios)
        records.append({'index': index, 'status': 'raw_saved',
            'raw_path': str(raw_path), 'raw_sha256': fit.sha256(raw_path),
            'raw_saved': True, 'solve_requested': True,
            'solve_returned': True, 'losses': losses.tolist(),
            'parts': parts.tolist(), 'ratios': ratios.tolist(),
            'prior': .01,
            'restoration_value': .1 / index, 'gradient_norm': .01,
            'guard3': [0., 0., 0.], 'guard_ratios': [1., 1., 1.],
            'startup_negative_count': 1 if index == 1 else 0,
            'selector_feasible': selector,
            'true_worst_objective': objective,
            'objective_over_baseline': objective / baseline,
            'qualifies': qualifies})
    ledger = {'original_evaluations': records,
        'original_evaluations_requested': 3,
        'original_evaluations_returned': 3,
        'original_evaluations_completed': 3,
        'original_evaluations_failed_unknown': 0,
        'original_evaluations_post_return_error': 0,
        'optimizer': {'status': 'bounded_stop', 'requested': 1,
            'returned': 0, 'bounded_stop': 1, 'failed_unknown': 0},
        'fit_calls': 1, 'fit_wall_s': 2., 'raw_bytes': 100,
        'synthetic_tiny': {'status': 'completed', 'raw_saved': True,
                           'solve_returned': True},
        'freeze_sha256': 'generated', 'source_review_sha256': 'generated'}
    local_cache = SimpleNamespace(started=3, completed=3,
        cache={1: None, 2: None, 3: None}, hits=0, selected_index=3)
    freeze = {'max_unique_evaluations': 18, 'fit_wall_limit_s': 40,
        'maximum_raw_bytes': 1073741824, 'start_objective': 1.16,
        'original_objective_guard': .99 * baseline}
    comparison = {'accepted': True,
                  'actual_sha256': records[0]['raw_sha256']}
    evidence = {'baseline_objective': baseline}
    result = e137.result_from_records(freeze, ledger, local_cache,
        comparison, 'first_guarded_selector_feasible', evidence)
    assert result['status'] == 'completed_bounded_restoration'
    assert result['hypothesis_verdict'] == 'supported'
    assert result['selected_index'] == 3
    assert result['selected_objective_over_baseline'] == pytest.approx(
        1.17 / baseline)
    assert result['new_lf_proposal_accepted'] is True
    extra = dict(records[1])
    extra.update(index=4, raw_path=str(tmp_path / 'extra.npz'))
    np.savez(extra['raw_path'], losses=np.array(extra['losses']),
             parts=np.array(extra['parts']), ratios=np.array(extra['ratios']))
    extra['raw_sha256'] = fit.sha256(extra['raw_path'])
    records.append(extra)
    for key in ('original_evaluations_requested',
                'original_evaluations_returned',
                'original_evaluations_completed'):
        ledger[key] = 4
    local_cache.started = local_cache.completed = 4
    local_cache.cache[4] = None
    assert e137.result_from_records(freeze, ledger, local_cache,
        comparison, 'first_guarded_selector_feasible', evidence)[
            'hypothesis_verdict'] == 'not_tested'
    records.pop()
    for key in ('original_evaluations_requested',
                'original_evaluations_returned',
                'original_evaluations_completed'):
        ledger[key] = 3
    local_cache.started = local_cache.completed = 3
    local_cache.cache.pop(4)
    records[2]['qualifies'] = False
    records[2]['true_worst_objective'] = 1.19
    records[2]['objective_over_baseline'] = 1.19 / baseline
    records[2]['losses'] = [1.18, 1.17, 1.16]
    np.savez(records[2]['raw_path'], losses=np.array(records[2]['losses']),
             parts=np.array(records[2]['parts']),
             ratios=np.array(records[2]['ratios']))
    records[2]['raw_sha256'] = fit.sha256(records[2]['raw_path'])
    local_cache.selected_index = None
    result = e137.result_from_records(freeze, ledger, local_cache,
        comparison, 'scipy_finished', evidence)
    assert result['hypothesis_verdict'] == 'refuted'
    assert result['selected_index'] is None
    assert e137.result_from_records(freeze, ledger, local_cache,
        comparison, 'first_guarded_selector_feasible', evidence)[
            'hypothesis_verdict'] == 'not_tested'
    ledger['original_evaluations_post_return_error'] = 1
    assert e137.result_from_records(freeze, ledger, local_cache,
        comparison, 'scipy_finished', evidence)['hypothesis_verdict'] == 'not_tested'


def test_finalization_preserves_primary_and_revokes_late_result(
        tmp_path, monkeypatch):
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'status': 'completed_bounded_restoration'}))
    ledger = {'status': 'completed_bounded_restoration',
        'primary_error': {'type': 'RuntimeError', 'message': 'primary',
            'traceback': 'original traceback',
            'response_progress': {'context': 1}}}
    original = e137.write_output_manifest

    def fail_once(out):
        monkeypatch.setattr(e137, 'write_output_manifest', original)
        raise OSError('manifest disk')

    monkeypatch.setattr(e137, 'write_output_manifest', fail_once)
    primary = RuntimeError('primary')
    returned = e137.finalize(tmp_path, ledger, None, 0., primary)
    assert returned is primary
    assert not result.exists()
    stored = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert stored['primary_error']['traceback'] == 'original traceback'
    assert stored['primary_error']['response_progress'] == {'context': 1}
    assert stored['status'] == 'failed'
    assert stored['finalization_errors'][0]['stage'] == 'final_manifest'


def test_bound_shape_and_guard_contract():
    with pytest.raises(ValueError, match='guard dimensions changed'):
        e137.restoration_value_jac(generated_data(), 1., 1.)
    assert math.isclose(e137.OPTIONS['ftol'], 1e-12)
