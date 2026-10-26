"""Generated CPU-only E135 saved-derivative integration and LP failure checks."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_timescale_headroom as e135


def tiny_chain():
    # Nontrivial global LF muscle order [4, 1], local LTM muscle 1.
    global_muscles = np.array([4, 1])
    full_b = np.zeros((3, 4, 14))
    full_b[:, :, 8 + global_muscles] = np.array([[[1., 2.], [3., 4.],
        [5., 6.], [7., 8.]]])
    tau_b = full_b[:, :, 8 + global_muscles]
    cotangent = np.broadcast_to(np.array([[[.1, .2], [.3, .4],
        [.5, .6], [.7, .8]]]), (3, 4, 2)).copy()
    q_coeff = np.broadcast_to(np.array([[[[2., 3.]], [[4., 5.]]]]),
                              (3, 2, 1, 2)).copy()
    ltm_coeff = np.full((3, 2), .25)
    return tau_b, cotangent, q_coeff, ltm_coeff


def generated_constraint_case():
    tau_b = np.broadcast_to(np.arange(100, dtype=np.float64)[None, :, None]
        * np.array([.001, .002])[None, None, :], (3, 100, 2)).copy()
    cotangent = np.full((3, 100, 2), .1)
    q_coeff = np.full((3, 99, 7, 2), .02)
    ltm_coeff = np.full((3, 99), .01)
    raw = {'losses': np.array([.3, .2, .1]),
        'loss_jac': np.zeros((3, 2)),
        'q_error': np.full((3, 99, 7), .001),
        'ltm_error': np.zeros((3, 99)),
        'cotangent': cotangent, 'q_coefficient': q_coeff,
        'ltm_coefficient': ltm_coeff}
    old = {'q_jac': np.zeros((3, 99, 7, 2)),
        'ltm_jac': np.zeros((3, 99, 2)),
        'gradient': np.array([.001, .002, 1.]), 't0': .3}
    return old, raw, tau_b


def generated_lp_base():
    gradient = np.zeros(308)
    gradient[290] = -1.
    gradient[-1] = 1.
    a = np.zeros((1, 308))
    a[0, -1] = -1.
    bounds = np.vstack((np.tile([[-1., 1.]], (290, 1)),
        np.tile([[-math.log(3), math.log(10)]], (17, 1))))
    return {'vector': np.zeros(307), 'bounds': bounds,
        'gradient': gradient, 'a_ub': a, 'b_ub': np.array([0.]),
        'c': np.array([0.]), 'ratios': np.zeros((3, 4)),
        'true_worst_objective': 1.}


def test_global_local_tau_order_and_explicit_chain_rule():
    tau_b, cotangent, q_coeff, ltm_coeff = tiny_chain()
    loss, q, ltm = e135.tau_columns(tau_b, cotangent, q_coeff,
                                    ltm_coeff, local_ltm=1)
    assert loss.shape == (3, 2)
    assert loss[0, 0] == pytest.approx(1*.1 + 3*.3 + 5*.5 + 7*.7)
    assert loss[0, 1] == pytest.approx(2*.2 + 4*.4 + 6*.6 + 8*.8)
    assert q[0, 0, 0].tolist() == pytest.approx([3*2, 4*3])
    assert q[0, 1, 0].tolist() == pytest.approx([5*4, 6*5])
    assert np.all(ltm[:, :, 0] == 0)
    assert ltm[0, 0, 1] == pytest.approx(4*.25)
    full_b = np.zeros((3, 4, 6))
    full_b[:, :, 2] = tau_b[:, :, 1]
    full_b[:, :, 5] = tau_b[:, :, 0]
    old = e135.old_loss_from_full_b(full_b, cotangent,
        np.array([5, 2]), np.array([0, 1]))
    assert old == pytest.approx(loss)


def test_original_constraint_columns_and_tau_derivative_finite_difference():
    old, raw, tau_b = generated_constraint_case()
    extended = e135.extend_constraints(old, raw, tau_b, local_ltm=1)
    assert extended['jac'].shape == (4761, 5)
    assert np.array_equal(extended['gradient'], [.001, .002, 0., 0., 1.])
    assert np.array_equal(extended['jac'][:, :2],
        e135.fit.constraint_values(raw['losses'], raw['loss_jac'],
            raw['q_error'], raw['ltm_error'], old['q_jac'], old['ltm_jac'],
            .3)[1][:, :2])
    assert np.array_equal(extended['jac'][:, -1],
        e135.fit.constraint_values(raw['losses'], raw['loss_jac'],
            raw['q_error'], raw['ltm_error'], old['q_jac'], old['ltm_jac'],
            .3)[1][:, -1])
    eps = 1e-6
    direction = np.array([.3, -.2])
    adjusted = []
    for sign in (1., -1.):
        step = sign * eps * direction
        c, _, _ = e135.fit.constraint_values(
            raw['losses'] + extended['tau_loss_jac'] @ step,
            np.concatenate((raw['loss_jac'], extended['tau_loss_jac']), axis=1),
            raw['q_error'] + np.einsum('ctjm,m->ctj',
                extended['tau_q_jac'], step),
            raw['ltm_error'] + np.einsum('ctm,m->ct',
                extended['tau_ltm_jac'], step),
            np.concatenate((old['q_jac'], extended['tau_q_jac']), axis=3),
            np.concatenate((old['ltm_jac'], extended['tau_ltm_jac']), axis=2),
            .3)
        adjusted.append(c)
    numeric = (adjusted[0] - adjusted[1]) / (2 * eps)
    analytic = extended['jac'][:, 2:4] @ direction
    assert np.max(np.abs(numeric - analytic)) < 1e-8


def test_pure_saved_matrix_lp_radii_bounds_and_duals(tmp_path):
    base = generated_lp_base()
    reductions = []
    for index, rho in enumerate((.005, .02, .05), 1):
        row = e135.solve_radius(base, rho, e135.OPTIONS, tmp_path, index)
        assert row['classification'] == 'optimal_valid'
        assert row['solver_requested'] and row['solver_returned']
        assert row['residual']['max_linear_violation'] <= 1e-7
        with np.load(row['raw_path'], allow_pickle=False) as saved:
            assert saved['x'].shape == (308,)
            assert saved['box'][290, 0] == pytest.approx(-rho *
                (math.log(10) + math.log(3)))
            assert saved['ineqlin_marginals'].shape == (1,)
            assert saved['lower_marginals'].shape == (308,)
            assert saved['upper_marginals'].shape == (308,)
        reductions.append(row['predicted_decrease_fraction'])
    assert reductions == sorted(reductions)
    assert reductions[-1] > .01
    assert base['bounds'][290].tolist() == pytest.approx(
        [-math.log(3), math.log(10)])


def test_caps_restoration_verdict_and_residual_boundaries():
    base = generated_lp_base()
    base['c'] = np.r_[np.zeros(3), np.full(8, -9e-5), np.ones(4750)]
    assert np.count_nonzero(base['c'] < 0) == 8
    base['ratios'][0, 0] = e135.fit.SELECTOR_CAP - 1e-5
    assert e135.fit.selector_feasible(base['ratios'])
    freeze = {'primary_radius': .05, 'predicted_reduction_fraction': .01}
    rows = [{'rho': rho, 'status': 'returned', 'classification': 'optimal_valid',
        'predicted_decrease_fraction': fraction,
        'predicted_decrease': fraction} for rho, fraction in
        zip((.005, .02, .05), (.1, .02, .005), strict=True)]
    result = e135.verdict(base, rows, freeze,
        {'primary_predicted_decrease_fraction': .00033})
    assert result['hypothesis_verdict'] == 'refuted'
    assert result['base_optimizer_cap_violated_rows'] == 8
    assert result['E133_primary_predicted_decrease_fraction'] == .00033
    rows[-1]['predicted_decrease_fraction'] = .02
    assert e135.verdict(base, rows, freeze)['hypothesis_verdict'] == 'supported'
    rows[-1]['classification'] = 'unresolved'
    assert e135.verdict(base, rows, freeze)['hypothesis_verdict'] == 'inconclusive'
    box = e135.radius_bounds(base, .02)
    assert e135.validate_solution(base, box, np.zeros(308), 0.)['valid']
    assert not e135.validate_solution(base, box, np.zeros(307), 0.)['valid']


def test_preflight_rejects_before_original_data(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Exact E135 freeze SHA'):
        e135.reviewed_inputs(bad, 'bad', bad, 'bad')


def test_solver_before_call_error_post_return_rawsave_and_counts(
        monkeypatch, tmp_path):
    base = generated_lp_base()
    invalid = generated_lp_base()
    invalid['vector'][0] = 10.
    pre = {}
    with pytest.raises(ValueError, match='trust box'):
        e135.solve_radius(invalid, .02, e135.OPTIONS, tmp_path, 1,
            progress=pre, solver=lambda *_a, **_k: None)
    assert not pre['solver_requested']
    failed = {}
    with pytest.raises(RuntimeError, match='generated solver'):
        e135.solve_radius(base, .02, e135.OPTIONS, tmp_path, 1,
            progress=failed, solver=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError('generated solver')))
    assert failed['solver_requested'] and failed['solver_failed_unknown']
    answer = SimpleNamespace(status=0, success=True, message='generated',
        nit=1, fun=0., x=np.zeros(308), slack=np.zeros(1))
    monkeypatch.setattr(e135.np, 'savez', lambda *_a, **_k: (_ for _ in ()).throw(
        OSError('generated raw save')))
    returned = {}
    with pytest.raises(OSError, match='generated raw save'):
        e135.solve_radius(base, .02, e135.OPTIONS, tmp_path, 2,
            progress=returned, solver=lambda *_a, **_k: answer)
    assert returned['solver_requested'] and returned['solver_returned']
    assert returned['post_return_error'] and not returned['solver_failed_unknown']
    assert len(returned['partial_answer']['x']) == 308
    ledger = {'lp_rows': [pre, failed, returned]}
    e135.refresh_lp_counts(ledger)
    assert (ledger['lp_requested'], ledger['lp_returned'],
        ledger['lp_failed_unknown'], ledger['lp_post_return_error']) == (2, 1, 1, 1)


def test_durable_intent_and_return_precede_solver_and_raw_save(
        monkeypatch, tmp_path):
    base = generated_lp_base()
    ledger_path = tmp_path / 'advance-ledger.json'
    row = {}
    ledger = {'lp_rows': [row]}
    stages = []

    def persist(current, stage):
        assert current is row
        stages.append(stage)
        e135.refresh_lp_counts(ledger)
        e135.save(ledger_path, ledger)

    def solver(*_args, **_kwargs):
        durable = json.loads(ledger_path.read_text(encoding='utf-8'))
        saved = durable['lp_rows'][0]
        assert saved['solver_request_intent']
        assert saved['solver_intent_persisted']
        assert saved['solver_in_flight']
        assert not saved['solver_requested']
        assert not saved['solver_returned']
        assert durable['lp_request_intents'] == durable['lp_in_flight_unknown'] == 1
        assert durable['lp_requested'] == durable['lp_returned'] == 0
        return SimpleNamespace(status=0, success=True, message='generated',
            nit=1, fun=0., x=np.zeros(308), slack=np.zeros(1))

    original_savez = e135.np.savez

    def save_raw(*args, **kwargs):
        durable = json.loads(ledger_path.read_text(encoding='utf-8'))
        saved = durable['lp_rows'][0]
        assert saved['solver_returned'] and saved['solver_return_persisted']
        assert not saved['solver_in_flight']
        assert len(saved['partial_answer']['x']) == 308
        assert durable['lp_requested'] == durable['lp_returned'] == 1
        assert durable['lp_return_persisted'] == 1
        return original_savez(*args, **kwargs)

    monkeypatch.setattr(e135.np, 'savez', save_raw)
    e135.solve_radius(base, .02, e135.OPTIONS, tmp_path, 1,
        solver=solver, progress=row, persist_transition=persist)
    assert stages == ['request_intent', 'solver_returned']
    assert row['classification'] == 'optimal_valid'


@pytest.mark.parametrize('failed_stage', ['request_intent', 'solver_returned'])
def test_transition_persistence_failure_preserves_call_truth_and_partial(
        tmp_path, failed_stage):
    calls = []
    row = {}

    def persist(_row, stage):
        if stage == failed_stage:
            raise OSError('generated ' + stage + ' persistence')

    def solver(*_args, **_kwargs):
        calls.append('entered')
        return SimpleNamespace(status=0, success=True, message='generated',
            nit=1, fun=0., x=np.zeros(308), slack=np.zeros(1))

    with pytest.raises(OSError, match='generated ' + failed_stage):
        e135.solve_radius(generated_lp_base(), .02, e135.OPTIONS,
            tmp_path, 1, solver=solver, progress=row,
            persist_transition=persist)
    if failed_stage == 'request_intent':
        assert calls == []
        assert not row['solver_requested'] and not row['solver_returned']
        assert not row['solver_request_intent']
    else:
        assert calls == ['entered']
        assert row['solver_requested'] and row['solver_returned']
        assert row['post_return_error'] and not row['solver_return_persisted']
        assert len(row['partial_answer']['x']) == 308
        assert not list(tmp_path.glob('lp-radius-*-raw.npz'))


def _fake_run_setup(monkeypatch, tmp_path, solve):
    paths = {'raw_path': 'raw.npz', 'proposal_path': 'proposal.npz',
        'fixture_path': 'fixture.npz', 'E134_zero_path': 'zero.npz',
        'E134_parameters_path': 'parameters.npz', 'E133_base_path': 'base.npz',
        'E133_result_path': 'old-result.json',
        'E133_final_review_path': 'old-review.json',
        'E134_final_review_path': 'tau-review.json'}
    for key in ('raw_path', 'proposal_path', 'fixture_path',
                'E134_zero_path', 'E134_parameters_path', 'E133_base_path'):
        name = paths[key]
        if key == 'proposal_path':
            values = {'source_raw_sha256': np.array('sha')}
        elif key == 'fixture_path':
            values = {field: np.array([1]) for field in
                ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask',
                 'edge_ids', 'edge_rows', 'membership', 'moment', 'ltm',
                 'delta_bounds', 'motor_ids')}
        elif key == 'E133_base_path':
            values = {field: np.array([1.]) for field in
                ('vector', 'bounds', 'c', 'jac', 'gradient', 'ratios')}
        else:
            values = {'placeholder': np.array([1])}
        np.savez(tmp_path / name, **values)
    for name, value in [('old-result.json',
        {'status': 'completed_known', 'base_true_worst_objective': 1.}),
        ('old-review.json', {'validity': 'valid'}),
        ('tau-review.json', {'validity': 'valid'})]:
        (tmp_path / name).write_text(json.dumps(value), encoding='utf-8')
    freeze = paths | {'input_sha256': {'raw.npz': 'sha'},
        'radii': [.005, .02, .05], 'options': e135.OPTIONS,
        'internal_wall_limit_s': 8, 'pre_call_reserve_s': 2,
        'primary_radius': .05, 'predicted_reduction_fraction': .01}
    base = generated_lp_base()
    base['c'] = np.r_[np.full(8, -1e-4), np.ones(4753)]
    monkeypatch.setattr(e135, 'ROOT', tmp_path)
    monkeypatch.setattr(e135, 'reviewed_inputs', lambda *_: (freeze, {}))
    monkeypatch.setattr(e135, 'reconstruct', lambda *_: base)
    monkeypatch.setattr(e135, 'threadpool_info', list)
    monkeypatch.setattr(e135, 'threadpool_limits', lambda **_: nullcontext())
    monkeypatch.setattr(e135, 'solve_radius', solve)
    return freeze


def test_postreturn_ledger_failure_and_primary_secondary_preservation(
        monkeypatch, tmp_path):
    state = {'called': False, 'injected': 0}

    def fake_solve(_base, rho, _options, _out, index, *, progress,
                   persist_transition):
        progress.update(index=index, rho=rho, status='returned',
            classification='optimal_valid', solver_requested=True,
            solver_returned=True, solver_failed_unknown=False,
            predicted_decrease_fraction=.02, predicted_decrease=.02)
        state['called'] = True
        return progress

    _fake_run_setup(monkeypatch, tmp_path, fake_solve)
    original_save = e135.save

    def fail_once(path, value):
        if (Path(path).name == 'advance-ledger.json' and state['called']
                and state['injected'] == 0):
            state['injected'] += 1
            raise OSError('generated transient ledger')
        original_save(path, value)

    monkeypatch.setattr(e135, 'save', fail_once)
    with pytest.raises(OSError, match='generated transient ledger'):
        e135.run('freeze', 'sha', 'review', 'sha', tmp_path / 'output')
    ledger = json.loads((tmp_path / 'output/advance-ledger.json').read_text())
    assert ledger['lp_requested'] == ledger['lp_returned'] == 1
    assert ledger['lp_post_return_error'] == 1
    assert ledger['lp_skipped'] == 2
    assert [row['reason'] for row in ledger['lp_rows'][1:]] == ['prior_exception'] * 2
    assert ledger['status'] == 'failed'
    assert not (tmp_path / 'output/result.json').exists()


def test_solver_exception_records_unknown_and_unrequested_radii(monkeypatch, tmp_path):
    actual_solve = e135.solve_radius
    calls = []

    def failing_solver(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError('generated solver interruption')

    def wrapped(base, rho, options, out, index, *, progress, persist_transition):
        return actual_solve(base, rho, options, out, index, progress=progress,
            persist_transition=persist_transition, solver=failing_solver)

    _fake_run_setup(monkeypatch, tmp_path, wrapped)
    output = tmp_path / 'output'
    with pytest.raises(RuntimeError, match='generated solver interruption'):
        e135.run('freeze', 'sha', 'review', 'sha', output)
    ledger = json.loads((output / 'advance-ledger.json').read_text())
    assert len(calls) == ledger['lp_requested'] == ledger['lp_failed_unknown'] == 1
    assert ledger['lp_returned'] == ledger['lp_post_return_error'] == 0
    assert ledger['lp_skipped'] == 2
    assert [row['reason'] for row in ledger['lp_rows'][1:]] == ['prior_exception'] * 2
    assert ledger['hypothesis_verdict'] == 'inconclusive'
    assert 'generated solver interruption' in ledger['primary_error']['message']
    assert not (output / 'result.json').exists()


@pytest.mark.parametrize('classification,solver_status,expected_calls', [
    ('unresolved', 1, 1), ('invalid_numerical', 0, 1),
    ('linearized_infeasible', 2, 3)])
def test_run_stops_after_technical_result_but_not_mathematical_infeasibility(
        monkeypatch, tmp_path, classification, solver_status, expected_calls):
    calls = []

    def fake_solve(_base, rho, _options, _out, index, *, progress,
                   persist_transition):
        progress.update(index=index, rho=rho, status='in_flight_unknown',
            solver_request_intent=True, solver_intent_persisted=True,
            solver_in_flight=True, solver_requested=False,
            solver_returned=False)
        persist_transition(progress, 'request_intent')
        calls.append(index)
        progress.update(status='returned_before_raw', solver_in_flight=False,
            solver_requested=True, solver_returned=True,
            solver_return_persisted=True, solver_status=solver_status,
            partial_answer={'status': solver_status,
                            'message': 'generated returned answer'})
        persist_transition(progress, 'solver_returned')
        progress.update(status='returned', classification=classification,
                        message='generated returned answer')
        return progress

    _fake_run_setup(monkeypatch, tmp_path, fake_solve)
    output = tmp_path / 'output'
    if classification == 'linearized_infeasible':
        result_path = e135.run('freeze', 'sha', 'review', 'sha', output)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['status'] == 'completed_known'
        assert result['hypothesis_verdict'] == 'inconclusive'
    else:
        with pytest.raises(RuntimeError, match='technical LP result unresolved'):
            e135.run('freeze', 'sha', 'review', 'sha', output)
        assert not (output / 'result.json').exists()
    ledger = json.loads((output / 'advance-ledger.json').read_text(
        encoding='utf-8'))
    assert len(calls) == expected_calls
    assert ledger['lp_requested'] == ledger['lp_returned'] == expected_calls
    assert ledger['lp_skipped'] == 3 - expected_calls
    assert ledger['lp_failed_unknown'] == 0
    if expected_calls == 1:
        assert ledger['status'] == 'failed'
        assert ledger['hypothesis_verdict'] == 'inconclusive'
        assert [x['reason'] for x in ledger['lp_rows'][1:]] == [
            'prior_technical_lp_result'] * 2


def test_finalization_secondary_failures_cannot_replace_primary(
        monkeypatch, tmp_path):
    e135.save(tmp_path / 'result.json', {'hypothesis_verdict': 'supported'})
    primary = RuntimeError('generated original primary')
    original_unlink = Path.unlink
    original_save = e135.save

    def bad_unlink(path, *args, **kwargs):
        if path.name == 'result.json':
            raise OSError('generated unlink failure')
        return original_unlink(path, *args, **kwargs)

    def bad_fallback(path, value):
        if Path(path).name == 'result.json':
            raise OSError('generated fallback failure')
        original_save(path, value)

    monkeypatch.setattr(Path, 'unlink', bad_unlink)
    monkeypatch.setattr(e135, 'save', bad_fallback)
    assert e135.finalize(tmp_path, {'status': 'failed'},
        {'internal_wall_limit_s': 8}, time.perf_counter(), primary) is primary
    assert any('result_removal' in x for x in primary.__notes__)
    assert any('result_invalidation_fallback' in x for x in primary.__notes__)


def test_late_manifest_failure_revokes_generated_success(monkeypatch, tmp_path):
    e135.save(tmp_path / 'result.json', {'hypothesis_verdict': 'supported'})
    monkeypatch.setattr(e135, 'write_manifest',
        lambda *_: (_ for _ in ()).throw(OSError('generated late manifest')))
    primary = e135.finalize(tmp_path, {'status': 'completed_known'},
        {'internal_wall_limit_s': 8}, time.perf_counter(), None)
    assert isinstance(primary, OSError)
    assert not (tmp_path / 'result.json').exists()
    saved = json.loads((tmp_path / 'advance-ledger.json').read_text())
    assert saved['status'] == 'failed'
