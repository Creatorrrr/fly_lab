"""Generated CPU-only E136 chain rule, orchestration, and bounded-stop checks."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_timescale_constrained as e136


def generated_data(losses=None, ratio=0.):
    return {'delta': np.zeros(273), 'gain': np.zeros(17),
        'log_tau': np.zeros(17),
        'losses': np.array([.4, .3, .2] if losses is None else losses),
        'parts': np.zeros((3, 4)), 'loss_jac': np.zeros((3, 307)),
        'q_error': np.zeros((3, 99, 7)),
        'ltm_error': np.zeros((3, 99)),
        'q_jac': np.zeros((3, 99, 7, 307)),
        'ltm_jac': np.zeros((3, 99, 307)),
        'ratios': np.full((3, 4), ratio)}


def generated_ledger():
    return {'original_evaluations': [], 'raw_bytes': 0,
        'optimizer': {'status': 'not_requested', 'intent_persisted': False,
            'requested': 0, 'returned': 0, 'bounded_stop': 0,
            'failed_unknown': 0}, 'fit_calls': 0,
        'synthetic_tiny': {'status': 'completed', 'raw_saved': True,
            'solve_returned': True}, 'freeze_sha256': 'freeze',
        'source_review_sha256': 'review', 'fit_wall_s': 1.}


def test_nontrivial_local_muscle_mapping_tau_columns_and_chain_rule():
    rng = np.random.default_rng(136)
    mapping = np.r_[np.arange(273) % 17, np.arange(17), np.arange(17)]
    mapping[:4] = [9, 2, 13, 0]
    b = rng.normal(size=(3, 99, 307))
    q_coeff = rng.normal(size=(3, 99, 7, 17))
    ltm_coeff = rng.normal(size=(3, 99))
    q, ltm = e136.initial_jacobians(b, q_coeff, ltm_coeff,
                                     mapping, ltm_local=9)
    assert q.shape == (3, 99, 7, 307)
    assert ltm.shape == (3, 99, 307)
    for parameter in (0, 1, 2, 273, 289, 290, 306):
        local = mapping[parameter]
        np.testing.assert_allclose(q[:, :, :, parameter],
            b[:, :, parameter, None] * q_coeff[:, :, :, local])
        expected = (b[:, :, parameter] * ltm_coeff if local == 9
                    else np.zeros((3, 99)))
        np.testing.assert_allclose(ltm[:, :, parameter], expected)
    full_b = rng.normal(size=(3, 11, 307))
    cotangent = rng.normal(size=(3, 11, 17))
    loss = e136.saved_full_b_loss_jac(full_b, cotangent, mapping)
    for parameter in (0, 1, 273, 290, 306):
        np.testing.assert_allclose(loss[:, parameter],
            np.sum(full_b[:, :, parameter] *
                   cotangent[:, :, mapping[parameter]], axis=1))


def test_original_bounds_objective_and_exact_ell_prior_zero():
    fixture = {'delta_bounds': np.tile([[-2., 3.]], (300, 1))}
    bounds = e136.original_bounds(fixture, np.arange(273))
    assert bounds.shape == (307, 2)
    np.testing.assert_allclose(bounds[290:, 0], -math.log(3))
    np.testing.assert_allclose(bounds[290:, 1], math.log(10))
    data = generated_data()
    data['delta'][0] = 2.
    data['gain'][4] = 3.
    value, jac, prior = e136.objective_and_jac(data, .5)
    assert value == pytest.approx(.5 + prior)
    assert prior == pytest.approx(6e-5 * 4 / 1572 + 1e-5 * 9 / 96)
    assert jac.shape == (308,) and jac[-1] == 1
    np.testing.assert_array_equal(jac[290:307], 0.)
    assert jac[0] == pytest.approx(1.2e-4 * 2 / 1572)


def test_slsqp_closures_use_307_columns_and_original_4761_rows():
    data = generated_data()

    class Cache:
        def get(self, x):
            return data, float(x[-1])

    def fake_solver(objective, x0, **kwargs):
        assert kwargs['method'] == 'SLSQP'
        assert kwargs['options'] == {'maxiter': 100, 'ftol': 1e-9,
                                     'disp': False}
        assert len(kwargs['bounds']) == 308
        assert kwargs['bounds'][-1] == (None, None)
        assert objective(x0) == pytest.approx(.41)
        assert kwargs['jac'](x0).shape == (308,)
        constraint = kwargs['constraints'][0]
        assert constraint['fun'](x0).shape == (4761,)
        assert constraint['jac'](x0).shape == (4761, 308)
        return 'generated solver return'

    bounds = np.tile([[-1., 1.]], (307, 1))
    returned = e136.solve_one_fit(Cache(), bounds,
        np.r_[np.zeros(307), .41], solver=fake_solver)
    assert returned == 'generated solver return'


def test_start_vector_exact_float64_proposal_and_first_raw_parity(
        monkeypatch, tmp_path):
    monkeypatch.setattr(e136, 'ROOT', tmp_path)
    raw_path = tmp_path / 'old.npz'
    proposal_path = tmp_path / 'proposal.npz'
    base_path = tmp_path / 'base.npz'
    response = {key: np.zeros((3, 2, 1)) for key in e136.fit.FIELDS}
    response['command'] = np.zeros((3, 2, 42))
    response['tendon'] = np.zeros((3, 2, 1))
    old = response | {'delta': np.zeros(273), 'gain': np.zeros(17),
        'edge_ids': np.arange(273), 'motor_indices': np.arange(69),
        'muscle_indices': np.arange(17),
        'parameter_muscle_map': np.arange(290) % 17,
        'branch_bits': np.zeros((3, 4, 69), np.uint8),
        'B_initial99': np.zeros((3, 1, 290)),
        'losses': np.ones(3), 'parts': np.zeros((3, 4)),
        'ratios': np.zeros((3, 4)), 'loss_jac': np.zeros((3, 290)),
        'q_error': np.zeros((3, 99, 7)),
        'ltm_error': np.zeros((3, 99)),
        'q_coefficient': np.zeros((3, 99, 7, 17)),
        'ltm_coefficient': np.zeros((3, 99))}
    np.savez(raw_path, **old)
    digest = e136.fit.sha256(raw_path)
    np.savez(proposal_path, delta=old['delta'], gain=old['gain'],
        original_evaluation_index=np.array(39),
        source_raw_sha256=np.array(digest))
    np.savez(base_path, tau_B=np.zeros((3, 2, 17)),
             tau_loss_jac=np.zeros((3, 17)))
    freeze = {'start_raw_path': 'old.npz', 'start_proposal_path': 'proposal.npz',
        'input_sha256': {'old.npz': digest}}
    np.testing.assert_array_equal(e136.start_vector(freeze), np.zeros(307))
    actual = old | {'log_tau': np.zeros(17),
        'parameter_muscle_map': np.r_[old['parameter_muscle_map'],
                                      np.arange(17)],
        'B': np.zeros((3, 2, 307)),
        'B_initial99': np.zeros((3, 1, 307)),
        'loss_jac': np.zeros((3, 307))}
    actual_path = tmp_path / 'actual.npz'
    np.savez(actual_path, **actual)
    assert e136.compare_first_raw(actual_path, raw_path,
                                  base_path)['accepted']
    actual['loss_jac'][0, 300] = 1e-8
    np.savez(actual_path, **actual)
    assert not e136.compare_first_raw(actual_path, raw_path,
                                      base_path)['accepted']
    np.savez(proposal_path, delta=np.ones(273), gain=old['gain'],
        original_evaluation_index=np.array(39),
        source_raw_sha256=np.array(digest))
    with pytest.raises(ValueError, match='selected39/proposal'):
        e136.start_vector(freeze)


def test_preflight_rejects_bad_exact_freeze_before_original_data(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Exact E136 freeze SHA'):
        e136.reviewed_inputs(bad, 'bad', bad, 'bad')


def test_durable_intent_return_and_raw_are_separate(tmp_path):
    ledger = generated_ledger()
    path = tmp_path / 'advance-ledger.json'
    stages = []

    def persist():
        e136.refresh_counts(ledger)
        e136.save(path, ledger)

    row = e136.register_intent(ledger, 1, np.zeros(307), persist)

    def evaluator(_vector, _index, directory, **callbacks):
        durable = json.loads(path.read_text(encoding='utf-8'))
        assert durable['original_evaluations_intended'] == 1
        assert durable['original_evaluations_requested'] == 0
        callbacks['on_solve_entry']()
        durable = json.loads(path.read_text(encoding='utf-8'))
        assert durable['original_evaluations_requested'] == 1
        assert durable['original_evaluations'][0]['request_receipt_persisted']
        assert durable['original_evaluations'][0]['in_flight']
        stages.append('solve_entry')
        callbacks['on_solve_return']({'progress': {'stage': 'known_complete'},
                                      'wall_s': .2}, {'rate': np.zeros(1)})
        durable = json.loads(path.read_text(encoding='utf-8'))
        assert durable['original_evaluations_returned'] == 1
        assert durable['original_evaluations_completed'] == 0
        stages.append('returned')
        raw = directory / 'leg-0-evaluation-001-raw.npz'
        np.savez(raw, data=np.array([1.]))
        callbacks['on_raw_saved'](raw, e136.fit.sha256(raw))
        durable = json.loads(path.read_text(encoding='utf-8'))
        assert durable['original_evaluations_completed'] == 1
        stages.append('raw')
        data = generated_data()
        data.update(raw_path=str(raw), raw_sha256=e136.fit.sha256(raw),
            prior=0., true_worst_objective=.4, evaluate_wall_s=.3)
        return data

    record, _ = e136.evaluate_with_receipts(ledger, row, np.zeros(307),
        tmp_path / 'evaluation-001', evaluator, persist)
    assert stages == ['solve_entry', 'returned', 'raw']
    assert row['status'] == 'raw_saved' and row['raw_saved']
    assert record['true_worst_objective'] == .4


def test_intent_persistence_failure_prevents_original_evaluator(tmp_path):
    ledger = generated_ledger()
    with pytest.raises(OSError, match='generated intent failure'):
        e136.register_intent(ledger, 1, np.zeros(307),
            lambda: (_ for _ in ()).throw(OSError('generated intent failure')))
    row = ledger['original_evaluations'][0]
    assert not row['intent_persisted'] and not row['solve_requested']
    e136.refresh_counts(ledger)
    assert ledger['original_evaluations_requested'] == 0


def test_pre_entry_request_write_failure_prevents_original_solve(tmp_path):
    ledger = generated_ledger()
    row = e136.register_intent(ledger, 1, np.zeros(307), lambda: None)
    calls = []

    def persist():
        if row['solve_requested']:
            raise OSError('generated original request receipt failure')

    def evaluator(_vector, _index, _directory, **callbacks):
        callbacks['on_solve_entry']()
        calls.append('expensive solve entered')

    with pytest.raises(OSError, match='request receipt failure'):
        e136.evaluate_with_receipts(ledger, row, np.zeros(307),
            tmp_path / 'evaluation-001', evaluator, persist)
    e136.refresh_counts(ledger)
    assert calls == []
    assert ledger['original_evaluations_requested'] == 0
    assert not row['request_receipt_persisted']
    assert row['status'] == 'failed'


def test_tiny_and_optimizer_request_receipts_at_expensive_entry(
        monkeypatch, tmp_path):
    ledger = generated_ledger()
    ledger_path = tmp_path / 'advance-ledger.json'
    tiny_calls = []

    class TinyEngine:
        def __init__(self, _mapping):
            pass

        def metadata(self):
            return {'geometry_cost': {}}

        def solve(self, *_args, **_kwargs):
            durable = json.loads(ledger_path.read_text(encoding='utf-8'))
            row = durable['synthetic_tiny']
            assert row['solve_requested'] and row['request_receipt_persisted']
            assert row['in_flight'] and not row['solve_returned']
            tiny_calls.append(1)
            raise RuntimeError('generated tiny solve interruption')

    monkeypatch.setattr(e136.tau_module, 'CudaAdaptationTimescaleEngine',
                        TinyEngine)
    with pytest.raises(RuntimeError, match='tiny solve interruption'):
        e136.run_tiny_setup(tmp_path, ledger)
    assert tiny_calls == [1]

    class Cache:
        started = 1
        completed = 1
        hits = 0

        def get(self, _x):
            return generated_data(), 0.

    ledger['optimizer'] = generated_ledger()['optimizer']

    def persist():
        e136.save(ledger_path, ledger)

    def solver(*_args):
        durable = json.loads(ledger_path.read_text(encoding='utf-8'))
        assert durable['optimizer']['requested'] == 1
        assert durable['optimizer']['request_receipt_persisted']
        assert durable['optimizer']['in_flight']
        assert durable['fit_calls'] == 1
        raise e136.fit.FitLimit('generated bounded limit')

    _, reason = e136.bounded_optimizer(Cache(), np.zeros(307),
        np.tile([[-1., 1.]], (307, 1)), ledger, persist, solver=solver)
    assert reason == 'generated bounded limit'
    assert ledger['optimizer']['bounded_stop'] == 1

    ledger['optimizer'] = generated_ledger()['optimizer']
    ledger['fit_calls'] = 0
    solver_calls = []

    def fail_request():
        if ledger['optimizer'].get('requested') == 1:
            raise OSError('generated optimizer request receipt failure')

    with pytest.raises(OSError, match='optimizer request receipt failure'):
        e136.bounded_optimizer(Cache(), np.zeros(307),
            np.tile([[-1., 1.]], (307, 1)), ledger, fail_request,
            solver=lambda *_a: solver_calls.append(1))
    assert solver_calls == [] and ledger['fit_calls'] == 0
    assert ledger['optimizer']['requested'] == 0


def test_tiny_request_receipt_failure_prevents_tiny_solve(monkeypatch, tmp_path):
    ledger = generated_ledger()
    calls = []

    class TinyEngine:
        def __init__(self, _mapping):
            pass

        def metadata(self):
            return {'geometry_cost': {}}

        def solve(self, *_args, **_kwargs):
            calls.append(1)
            raise AssertionError('must not enter')

    monkeypatch.setattr(e136.tau_module, 'CudaAdaptationTimescaleEngine',
                        TinyEngine)
    original_save = e136.save

    def fail_request(path, value):
        if value.get('synthetic_tiny', {}).get('solve_requested'):
            raise OSError('generated tiny request receipt failure')
        original_save(path, value)

    monkeypatch.setattr(e136, 'save', fail_request)
    with pytest.raises(OSError, match='tiny request receipt failure'):
        e136.run_tiny_setup(tmp_path, ledger)
    assert calls == []
    assert not ledger['synthetic_tiny']['solve_requested']


def test_neural_progress_and_geometry_deltas_are_stage_specific(tmp_path):
    ledger = generated_ledger()
    row = e136.register_intent(ledger, 1, np.zeros(307), lambda: None)
    before = {'geometry_cost': {key: 0 for key in e136.GEOMETRY_FIELDS}}
    after = {'geometry_cost': before['geometry_cost'].copy()}

    def evaluator(_vector, _index, _directory, **callbacks):
        callbacks['on_solve_entry']()
        callbacks['on_solve_return']({'progress': {
            'forward_completion_class': 'known_complete',
            'completed_context_rk4_ticks': 120000,
            'completed_conditional_model_s': 30,
            'decoder_completion_class': 'known_complete',
            'completed_decoder_endpoint_ticks': 6000,
            'bank_completion_class': 'known_complete',
            'derivative_bank_complete': True,
            'completed_bank_edge_stage_ticks': 131040000},
            'wall_s': .1}, {'rate': np.zeros(1)})
        raise RuntimeError('generated geometry failure before any row')

    metadata = lambda: before if row.get('solve_returned') is not True else after
    with pytest.raises(RuntimeError, match='geometry failure'):
        e136.evaluate_with_receipts(ledger, row, np.zeros(307),
            tmp_path / 'evaluation-001', evaluator, lambda: None,
            metadata=metadata)
    e136.refresh_counts(ledger)
    assert ledger['original_context_ticks_known_complete'] == 120000
    assert ledger['original_derivative_banks_known_complete'] == 1
    assert ledger['original_geometry_forward_rows_known_complete'] == 0
    assert ledger['original_geometry_backward_rows_requested'] == 0
    assert row['geometry_delta']['forward_rows_completed'] == 0

    # The next synthetic failure has a partially known neural response and
    # already synchronized forward rows, without backward work.
    row2 = e136.register_intent(ledger, 2, np.ones(307), lambda: None)
    row2.update(solve_requested=True, solve_returned=False,
        response_progress={'forward_completion_class': 'known_complete',
            'completed_context_rk4_ticks': 120000,
            'completed_conditional_model_s': 30,
            'decoder_completion_class': 'unknown',
            'bank_completion_class': 'unknown',
            'derivative_bank_complete': False},
        geometry_delta={'forward_rows_requested': 100,
            'forward_rows_enqueued': 100,
            'forward_rows_completed': 40,
            'backward_rows_requested': 0,
            'backward_rows_enqueued': 0,
            'backward_rows_completed': 0})
    e136.refresh_counts(ledger)
    assert ledger['original_context_ticks_known_complete'] == 240000
    assert ledger['original_decoder_endpoints_known_complete'] == 6000
    assert ledger['original_decoder_endpoints_unknown_requested'] == 6000
    assert ledger['original_edge_stage_ticks_unknown_requested'] == 131040000
    assert ledger['original_geometry_forward_rows_requested'] == 100
    assert ledger['original_geometry_forward_rows_known_complete'] == 40
    assert ledger['original_geometry_forward_rows_unknown_enqueued'] == 60
    assert e136.geometry_delta(before, {
        'geometry_cost': {'forward_rows_requested': 100,
            'forward_rows_enqueued': 100, 'forward_rows_completed': 40,
            'backward_rows_requested': 0,
            'backward_rows_enqueued': 0,
            'backward_rows_completed': 0}})['forward_rows_completed'] == 40
    row3 = e136.register_intent(ledger, 3, np.ones(307) * 2,
                                lambda: None)
    row3.update(solve_requested=True, response_progress={
        'kernel_events': [{'name': 'prepare', 'requested': True}],
        'forward_completion_class': 'not_enqueued',
        'decoder_completion_class': 'not_enqueued',
        'bank_completion_class': 'not_enqueued'})
    e136.refresh_counts(ledger)
    assert ledger['original_context_ticks_requested'] == 240000
    assert ledger['original_context_ticks_known_complete'] == 240000
    assert ledger['original_context_ticks_unresolved_upper_bound'] == 120000
    assert ledger['original_geometry_forward_rows_known_complete'] == 40
    assert ledger['original_geometry_forward_rows_unresolved_upper_bound'] == (
        5473 + 60)


def test_cumulative_geometry_counters_are_differenced_per_evaluation():
    first = {'geometry_cost': {'forward_rows_requested': 120,
        'forward_rows_enqueued': 120, 'forward_rows_completed': 120,
        'backward_rows_requested': 0, 'backward_rows_enqueued': 0,
        'backward_rows_completed': 0}}
    second = {'geometry_cost': {'forward_rows_requested': 190,
        'forward_rows_enqueued': 190, 'forward_rows_completed': 160,
        'backward_rows_requested': 20, 'backward_rows_enqueued': 20,
        'backward_rows_completed': 10}}
    zero = {'geometry_cost': {key: 0 for key in e136.GEOMETRY_FIELDS}}
    a = e136.geometry_delta(zero, first)
    b = e136.geometry_delta(first, second)
    assert a['forward_rows_completed'] == 120
    assert b['forward_rows_requested'] == 70
    assert b['forward_rows_completed'] == 40
    assert b['forward_rows_unknown_enqueued'] == 30
    assert b['backward_rows_completed'] == 10
    ledger = generated_ledger()
    ledger['original_evaluations'] = [
        {'solve_requested': True, 'geometry_delta': a},
        {'solve_requested': True, 'geometry_delta': b}]
    e136.refresh_counts(ledger)
    assert ledger['original_geometry_forward_rows_requested'] == 190
    assert ledger['original_geometry_forward_rows_known_complete'] == 160


def test_evaluator_records_actual_geometry_backend_and_device_before_solve(
        tmp_path):
    engine = SimpleNamespace(problem={}, edges=1, muscles=1)
    geometry = SimpleNamespace(device='cuda:3')
    with pytest.raises(ValueError, match='cardinality'):
        e136.evaluate_timescale_leg(engine, {}, geometry, np.zeros(307),
            1, tmp_path)
    assert engine.geometry_backend == 'SimpleNamespace'
    assert engine.geometry_device == 'cuda:3'


def test_raw_cap_stops_after_known_return_and_keeps_saved_bytes(tmp_path):
    ledger = generated_ledger()
    row = e136.register_intent(ledger, 1, np.zeros(307), lambda: None)
    calls = []

    def evaluator(_vector, _index, directory, **callbacks):
        callbacks['on_solve_entry']()
        callbacks['on_solve_return']({'progress': {'stage': 'complete'},
                                      'wall_s': .1}, {'rate': np.zeros(1)})
        calls.append('returned')
        raw = directory / 'leg-0-evaluation-001-raw.npz'
        np.savez(raw, data=np.arange(20))
        callbacks['on_raw_saved'](raw, e136.fit.sha256(raw))
        calls.append('after_raw')
        return generated_data()

    with pytest.raises(ValueError, match='4 GiB original raw cap'):
        e136.evaluate_with_receipts(ledger, row, np.zeros(307),
            tmp_path / 'evaluation-001', evaluator, lambda: None,
            maximum_raw_bytes=1)
    e136.refresh_counts(ledger)
    assert calls == ['returned']
    assert ledger['original_evaluations_returned'] == 1
    assert ledger['original_evaluations_completed'] == 1
    assert ledger['raw_bytes'] > 1
    assert row['status'] == 'failed' and row['post_return_error']


@pytest.mark.parametrize('failure', ['return', 'raw', 'secondary'])
def test_post_return_failures_preserve_known_work_and_primary(
        tmp_path, failure):
    ledger = generated_ledger()
    calls = []
    row = e136.register_intent(ledger, 1, np.zeros(307), lambda: None)

    def persist():
        calls.append(row['status'])
        if failure == 'return' and row['status'] == 'returned_before_raw':
            raise OSError('generated return persistence')
        if failure == 'secondary' and row['status'] == 'failed':
            raise OSError('generated secondary ledger')

    def evaluator(_vector, _index, directory, **callbacks):
        callbacks['on_solve_entry']()
        callbacks['on_solve_return']({'progress': {'stage': 'complete'},
                                      'wall_s': .1}, {'rate': np.zeros(1)})
        if failure == 'secondary':
            raise RuntimeError('generated original primary')
        raw = directory / 'leg-0-evaluation-001-raw.npz'
        np.savez(raw, data=np.array([1.]))
        callbacks['on_raw_saved'](raw, e136.fit.sha256(raw))
        if failure == 'raw':
            raise OSError('generated raw post-return')
        return generated_data()

    expected = 'generated original primary' if failure == 'secondary' else (
        'generated return persistence' if failure == 'return' else
        'generated raw post-return')
    with pytest.raises((OSError, RuntimeError), match=expected) as caught:
        e136.evaluate_with_receipts(ledger, row, np.zeros(307),
            tmp_path / 'evaluation-001', evaluator, persist)
    e136.refresh_counts(ledger)
    assert ledger['original_evaluations_requested'] == 1
    assert ledger['original_evaluations_returned'] == 1
    assert row['post_return_error'] and row['status'] == 'failed'
    if failure == 'return':
        assert not row['raw_saved']
    if failure == 'raw':
        assert row['raw_saved']
    if failure == 'secondary':
        assert any('Secondary E136 evaluation ledger' in note
                   for note in caught.value.__notes__)


def test_optimizer_start_failure_means_zero_fit_calls_and_bounded_stop():
    class Cache:
        started = 1
        completed = 1
        hits = 0

        def get(self, _x):
            return generated_data(ratio=1.), 0.

    ledger = generated_ledger()
    calls = []
    with pytest.raises(ValueError, match='not selector-feasible'):
        e136.bounded_optimizer(Cache(), np.zeros(307),
            np.tile([[-1., 1.]], (307, 1)), ledger, lambda: None,
            solver=lambda *_a: calls.append(1))
    assert calls == [] and ledger['fit_calls'] == 0
    assert ledger['optimizer']['requested'] == 0

    class Parity(Cache):
        def get(self, _x):
            raise ValueError('generated first replay parity')

    with pytest.raises(ValueError, match='first replay parity'):
        e136.bounded_optimizer(Parity(), np.zeros(307),
            np.tile([[-1., 1.]], (307, 1)), ledger, lambda: None,
            solver=lambda *_a: calls.append(1))
    assert calls == [] and ledger['fit_calls'] == 0

    class Feasible(Cache):
        def get(self, _x):
            return generated_data(), 0.

    def bounded(*_args):
        raise e136.fit.FitLimit('per_leg_unique_or_wall_limit')

    _, reason = e136.bounded_optimizer(Feasible(), np.zeros(307),
        np.tile([[-1., 1.]], (307, 1)), ledger, lambda: None,
        solver=bounded)
    assert reason == 'per_leg_unique_or_wall_limit'
    assert ledger['fit_calls'] == 1
    assert ledger['optimizer']['bounded_stop'] == 1


def test_generic_cache_ignores_epigraph_and_stops_before_second_unique():
    calls = []

    def evaluate(vector, index):
        calls.append((index, vector.copy()))
        return generated_data()

    cache = e136.fit.EvaluationCache(evaluate, 307,
        lambda *_: None, max_calls=1,
        deadline=time.perf_counter() + 10,
        total_deadline=time.perf_counter() + 10)
    first, t0 = cache.get(np.r_[np.zeros(307), .41])
    same, t1 = cache.get(np.r_[np.zeros(307), 42.])
    assert first is same and (t0, t1) == (.41, 42.)
    assert len(calls) == 1 and cache.hits == 1
    with pytest.raises(e136.fit.FitLimit):
        cache.get(np.r_[np.ones(307), .41])
    assert cache.started == cache.completed == 1


def test_best_feasible_saved_objective_earliest_tie_and_one_percent(tmp_path):
    ledger = generated_ledger()
    losses = ([1., .8, .7], [.989, .8, .7], [.98, .8, .7],
              [.989, .8, .7])
    for index, values in enumerate(losses, 1):
        raw = tmp_path / f'raw-{index}.npz'
        np.savez(raw, x=np.array([index]))
        ratios = np.full((3, 4), 1. if index == 3 else 0.)
        ledger['original_evaluations'].append({'index': index,
            'status': 'raw_saved', 'raw_saved': True,
            'solve_requested': True, 'solve_returned': True,
            'post_return_error': False, 'losses': np.array(values),
            'parts': np.zeros((3, 4)), 'ratios': ratios, 'prior': 0.,
            'true_worst_objective': values[0],
            'selector_feasible': e136.fit.selector_feasible(ratios),
            'raw_path': str(raw), 'raw_sha256': e136.fit.sha256(raw)})
    e136.refresh_counts(ledger)
    ledger['optimizer'].update(status='returned', intent_persisted=True,
        requested=1, returned=1)
    ledger['fit_calls'] = 1
    cache = SimpleNamespace(started=4, completed=4, hits=1,
                            cache={i: i for i in range(4)})
    freeze = {'max_unique_evaluations': 64, 'fit_wall_limit_s': 95,
        'maximum_raw_bytes': 4294967296,
        'proposal_objective_ratio_limit': .99}
    comparison = {'accepted': True,
        'actual_sha256': ledger['original_evaluations'][0]['raw_sha256']}
    result = e136.result_from_records(freeze, ledger, cache, comparison,
                                       'scipy_finished')
    assert result['status'] == 'completed_bounded_fit'
    assert result['selected_index'] == 2
    assert result['best_any_index'] == 3
    assert result['new_lf_proposal_accepted']
    assert result['hypothesis_verdict'] == 'supported'
    ledger['original_evaluations'][1]['true_worst_objective'] = .995
    ledger['original_evaluations'][1]['losses'][0] = .995
    ledger['original_evaluations'][3]['true_worst_objective'] = .995
    ledger['original_evaluations'][3]['losses'][0] = .995
    result = e136.result_from_records(freeze, ledger, cache, comparison,
                                       'scipy_finished')
    assert result['selected_index'] == 2
    assert not result['new_lf_proposal_accepted']
    assert result['hypothesis_verdict'] == 'refuted'


def test_late_manifest_failure_revokes_success_and_preserves_primary(
        monkeypatch, tmp_path):
    e136.save(tmp_path / 'result.json', {'status': 'completed_bounded_fit',
        'new_lf_proposal_accepted': True})
    monkeypatch.setattr(e136, 'write_output_manifest',
        lambda *_: (_ for _ in ()).throw(OSError('generated late manifest')))
    primary = e136.finalize(tmp_path, {'status': 'completed_bounded_fit'},
        {'internal_wall_limit_s': 108}, time.perf_counter(), None)
    assert isinstance(primary, OSError)
    assert not (tmp_path / 'result.json').exists()
    saved = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert saved['status'] == 'failed'

    e136.save(tmp_path / 'result.json', {'status': 'completed_bounded_fit'})
    primary = RuntimeError('generated original primary')
    original_unlink = Path.unlink
    original_save = e136.save

    def bad_unlink(path, *args, **kwargs):
        if path.name == 'result.json':
            raise OSError('generated unlink failure')
        return original_unlink(path, *args, **kwargs)

    def bad_fallback(path, value):
        if Path(path).name == 'result.json':
            raise OSError('generated fallback failure')
        original_save(path, value)

    monkeypatch.setattr(Path, 'unlink', bad_unlink)
    monkeypatch.setattr(e136, 'save', bad_fallback)
    assert e136.finalize(tmp_path, {'status': 'failed'},
        {'internal_wall_limit_s': 108}, time.perf_counter(), primary) is primary
    assert any('result_removal' in x for x in primary.__notes__)
    assert any('result_invalidation_fallback' in x for x in primary.__notes__)


def test_finalize_keeps_original_traceback_and_neural_progress_after_late_error(
        monkeypatch, tmp_path):
    original = RuntimeError('generated neural primary')
    original.response_progress = {'stage': 'bank_requested',
        'completed_context_rk4_ticks': 120000}
    ledger = {'status': 'failed', 'original_evaluations': [],
        'primary_error': {'type': 'RuntimeError',
            'message': 'generated neural primary',
            'traceback': 'ORIGINAL NEURAL TRACEBACK SENTINEL',
            'response_progress': original.response_progress,
            'notes': ['original note']}}
    monkeypatch.setattr(e136, 'write_output_manifest',
        lambda *_: (_ for _ in ()).throw(OSError('generated late manifest')))
    returned = e136.finalize(tmp_path, ledger,
        {'internal_wall_limit_s': 108}, time.perf_counter(), original)
    assert returned is original
    durable = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert durable['primary_error']['traceback'] == (
        'ORIGINAL NEURAL TRACEBACK SENTINEL')
    assert durable['primary_error']['response_progress'] == (
        original.response_progress)
    assert 'original note' in durable['primary_error']['notes']
    assert any('final_manifest' in item for item in
               durable['primary_error']['notes'])
    assert any(item['stage'] == 'failed_manifest' for item in
               durable['finalization_errors'])
