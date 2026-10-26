"""CPU-generated E133 reconstruction, LP and evidence boundary checks."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_lf_linearized_headroom as e133


def generated_fixture_raw():
    fixture = {'leg_motor_mask': np.ones((1, 69), bool),
        'leg_muscle_mask': np.ones((1, 17), bool),
        'leg_edge_mask': np.ones((1, 273), bool),
        'edge_ids': np.arange(100, 373),
        'edge_rows': np.arange(273) % 69,
        'membership': (3 * np.arange(69) + 5) % 17,
        'moment': np.arange(42 * 17, dtype=np.float64).reshape(42, 17) / 1000,
        'ltm': np.array([5]),
        'delta_bounds': np.tile([[-1., 1.]], (273, 1))}
    mapping = np.r_[fixture['membership'][fixture['edge_rows']],
                    np.arange(17)]
    bank = np.broadcast_to(np.arange(290, dtype=np.float64)[None, None, :] /
        10000, (3, 99, 290)).copy()
    raw = {'delta': np.zeros(273), 'gain': np.ones(17),
        'losses': np.array([.5, .4, .3]),
        'loss_jac': np.zeros((3, 290)),
        'q_error': np.zeros((3, 99, 7)),
        'ltm_error': np.zeros((3, 99)),
        'B_initial99': bank,
        'q_coefficient': np.broadcast_to(
            (.02 * fixture['moment'][:7])[None, None, :, :],
            (3, 99, 7, 17)).copy(),
        'ltm_coefficient': np.full((3, 99), .02),
        'ratios': np.zeros((3, 4)),
        'parameter_muscle_map': mapping,
        'edge_ids': fixture['edge_ids'].copy(),
        'motor_indices': np.arange(69, dtype=np.int32),
        'muscle_indices': np.arange(17, dtype=np.int32)}
    proposal = {'delta': raw['delta'].copy(), 'gain': raw['gain'].copy(),
                'original_evaluation_index': np.array(39)}
    return fixture, raw, proposal


def generated_lp_base():
    gradient = np.zeros(291)
    gradient[0] = -1.
    gradient[-1] = 1.
    a = np.zeros((1, 291))
    a[0, -1] = -1.  # t step >= 0; a bounded first parameter gives a finite LP.
    return {'vector': np.zeros(290),
        'bounds': np.tile([[-1., 1.]], (290, 1)),
        'gradient': gradient, 'a_ub': a,
        'b_ub': np.array([0.]), 'c': np.array([0.]),
        'ratios': np.zeros((3, 4)),
        'true_worst_objective': 1.}


def test_reconstruct_nontrivial_local_muscle_permutation():
    fixture, raw, proposal = generated_fixture_raw()
    base = e133.reconstruct(raw, fixture, proposal)
    assert base['jac'].shape == (4761, 291)
    assert base['c'].shape == (4761,)
    assert e133.fit.selector_feasible(base['ratios'])
    selected = [0, 1, 2, 100, 272, 273, 289]
    for parameter in selected:
        local = raw['parameter_muscle_map'][parameter]
        assert base['q_jac'][0, 0, 3, parameter] == pytest.approx(
            raw['B_initial99'][0, 0, parameter] *
            raw['q_coefficient'][0, 0, 3, local])
        expected_ltm = (raw['B_initial99'][0, 0, parameter] * .02
                        if local == 5 else 0.)
        assert base['ltm_jac'][0, 0, parameter] == pytest.approx(expected_ltm)
    assert base['true_worst_objective'] > max(raw['losses'])


@pytest.mark.parametrize('damage', ['mapping', 'coefficient', 'proposal',
    'ratio', 'selector', 'bounds', 'id', 'nonfinite'])
def test_reconstruction_rejects_changed_saved_contract(damage):
    fixture, raw, proposal = generated_fixture_raw()
    if damage == 'mapping':
        raw['parameter_muscle_map'][0] = (raw['parameter_muscle_map'][0] + 1) % 17
    elif damage == 'coefficient':
        raw['q_coefficient'][0, 0, 0, 0] += .01
    elif damage == 'proposal':
        proposal['delta'][0] = .1
    elif damage == 'ratio':
        raw['ratios'][0, 0] = .1
    elif damage == 'selector':
        raw['q_error'][:] = 1.
        _, _, raw['ratios'] = e133.fit.constraint_values(
            raw['losses'], raw['loss_jac'], raw['q_error'],
            raw['ltm_error'], np.zeros((3, 99, 7, 290)),
            np.zeros((3, 99, 290)), max(raw['losses']))
    elif damage == 'bounds':
        raw['delta'][0] = 10.
        proposal['delta'][0] = 10.
    elif damage == 'id':
        raw['edge_ids'][0] = -1
    else:
        raw['B_initial99'][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        e133.reconstruct(raw, fixture, proposal)


def test_generated_cpu_lp_radius_order_and_dual_evidence(tmp_path):
    base = generated_lp_base()
    decreases = []
    for index, rho in enumerate((.005, .02, .05), 1):
        row = e133.solve_radius(base, rho, e133.OPTIONS, tmp_path, index)
        assert row['classification'] == 'optimal_valid'
        assert row['residual']['max_linear_violation'] <= 1e-7
        assert row['residual']['max_box_violation'] <= 1e-7
        assert row['residual']['objective_error'] <= 1e-9
        with np.load(row['raw_path'], allow_pickle=False) as saved:
            assert saved['x'].shape == (291,)
            assert saved['slack'].shape == (1,)
            assert saved['solver_status'] == 0
            for key in ('ineqlin_residual', 'ineqlin_marginals',
                        'lower_residual', 'lower_marginals',
                        'upper_residual', 'upper_marginals'):
                assert key in saved
        decreases.append(row['predicted_decrease_fraction'])
    assert decreases == pytest.approx([.01, .04, .1], abs=1e-8)
    assert decreases == sorted(decreases)


def test_independent_residual_and_fun_checks_reject_bad_solver_answer():
    base = generated_lp_base()
    box = e133.radius_bounds(base, .02)
    y = np.zeros(291)
    assert e133.validate_solution(base, box, y, 0.)['valid']
    assert not e133.validate_solution(base, box, y, 1.)['valid']
    y[0] = .05
    assert not e133.validate_solution(base, box, y, -.05)['valid']
    y[:] = 0.
    y[-1] = -1.
    assert not e133.validate_solution(base, box, y, -1.)['valid']


def test_original_box_intersection_and_two_second_reserve():
    base = generated_lp_base()
    base['vector'][0] = .99
    box = e133.radius_bounds(base, .05)
    assert box[0, 0] == pytest.approx(-.1)
    assert box[0, 1] == pytest.approx(.01)
    assert np.isneginf(box[-1, 0]) and np.isposinf(box[-1, 1])
    freeze = {'internal_wall_limit_s': 8, 'pre_call_reserve_s': 2}
    assert e133.may_start_lp(5.999, 0., freeze)
    assert not e133.may_start_lp(6., 0., freeze)


def test_infeasible_timeout_and_error_remain_unresolved(tmp_path):
    base = generated_lp_base()
    answers = [SimpleNamespace(status=2, success=False, message='infeasible',
        nit=0, fun=None, x=None, slack=None),
        SimpleNamespace(status=1, success=False, message='time limit',
        nit=3, fun=None, x=None, slack=None)]
    for index, answer in enumerate(answers, 1):
        row = e133.solve_radius(base, .02, e133.OPTIONS, tmp_path, index,
                                solver=lambda *_a, value=answer, **_k: value)
        assert row['classification'] == (
            'linearized_infeasible' if index == 1 else 'unresolved')
        assert 'predicted_decrease_fraction' not in row
        assert row['wall_s'] >= 0
    with pytest.raises(RuntimeError, match='generated solver'):
        e133.solve_radius(base, .05, e133.OPTIONS, tmp_path, 3,
            solver=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError('generated solver')))


def test_primary_radius_verdict_is_fixed_and_local():
    base = generated_lp_base()
    freeze = {'primary_radius': .05, 'predicted_reduction_fraction': .01}
    rows = [{'rho': rho, 'status': 'returned', 'classification': 'optimal_valid',
        'predicted_decrease_fraction': fraction,
        'predicted_decrease': fraction} for rho, fraction in
        zip((.005, .02, .05), (.4, .3, .005), strict=True)]
    result = e133.verdict(base, rows, freeze)
    assert result['hypothesis_verdict'] == 'refuted'
    assert result['interpretation'] == 'local_linearized_prediction_only'
    rows[-1]['predicted_decrease_fraction'] = .02
    assert e133.verdict(base, rows, freeze)['hypothesis_verdict'] == 'supported'
    rows[-1]['classification'] = 'linearized_infeasible'
    assert e133.verdict(base, rows, freeze)['hypothesis_verdict'] == 'inconclusive'


def test_preflight_rejects_without_original_npz(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Exact E133 freeze SHA'):
        e133.reviewed_inputs(bad, 'wrong', bad, 'wrong')


def test_final_manifest_and_late_failure_invalidate_receipt(monkeypatch, tmp_path):
    e133.save(tmp_path / 'result.json', {'hypothesis_verdict': 'supported'})
    ledger = {'status': 'completed'}
    primary = e133.finalize(tmp_path, ledger, {'internal_wall_limit_s': 8},
                            time.perf_counter(), None)
    assert primary is None
    manifest = json.loads((tmp_path / 'output-manifest.json').read_text())
    assert manifest['files']['result.json'] == e133.fit.sha256(tmp_path / 'result.json')
    monkeypatch.setattr(e133, 'write_manifest',
                        lambda _: (_ for _ in ()).throw(OSError('generated manifest')))
    primary = e133.finalize(tmp_path, ledger, {'internal_wall_limit_s': 8},
                            time.perf_counter(), None)
    assert isinstance(primary, OSError)
    assert not (tmp_path / 'result.json').exists()


def test_solver_entry_return_and_post_return_raw_failure(monkeypatch, tmp_path):
    base = generated_lp_base()
    pre = generated_lp_base()
    pre['vector'][0] = 10.
    row = {}
    calls = []
    with pytest.raises(ValueError, match='trust box'):
        e133.solve_radius(pre, .02, e133.OPTIONS, tmp_path, 1,
            progress=row, solver=lambda *_a, **_k: calls.append(1))
    assert not calls
    assert (row['solver_requested'], row['solver_returned'],
            row['solver_failed_unknown']) == (False, False, False)
    failed = {}
    with pytest.raises(RuntimeError, match='generated solver'):
        e133.solve_radius(base, .02, e133.OPTIONS, tmp_path, 1,
            progress=failed, solver=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError('generated solver')))
    assert (failed['solver_requested'], failed['solver_returned'],
            failed['solver_failed_unknown']) == (True, False, True)
    answer = SimpleNamespace(status=0, success=True, message='generated', nit=1,
        fun=0., x=np.zeros(291), slack=np.zeros(1))
    monkeypatch.setattr(e133.np, 'savez', lambda *_a, **_k: (_ for _ in ()).throw(
        OSError('generated raw persistence')))
    returned = {}
    with pytest.raises(OSError, match='generated raw persistence'):
        e133.solve_radius(base, .02, e133.OPTIONS, tmp_path, 2,
                          progress=returned, solver=lambda *_a, **_k: answer)
    assert (returned['solver_requested'], returned['solver_returned'],
            returned['solver_failed_unknown'], returned['post_return_error']) == (
                True, True, False, True)
    assert len(returned['partial_answer']['x']) == 291
    assert len(returned['partial_answer']['slack']) == 1
    assert returned['wall_s'] >= 0
    ledger = {'lp_rows': [row, failed, returned]}
    e133.refresh_lp_counts(ledger)
    assert (ledger['lp_requested'], ledger['lp_returned'],
            ledger['lp_failed_unknown'], ledger['lp_post_return_error'],
            ledger['lp_error']) == (2, 1, 1, 1, 3)


def _generated_run_setup(monkeypatch, tmp_path, solve):
    for name, body in [('result.json', {'new_lf_proposal_accepted': True,
        'selected_index': 39, 'selected_objective': 1.}),
        ('review.json', {'validity': 'valid', 'selected_raw_sha256': 'sha'})]:
        (tmp_path / name).write_text(json.dumps(body), encoding='utf-8')
    fixture, raw, proposal = generated_fixture_raw()
    proposal['source_raw_sha256'] = np.array('sha')
    for name, values in [('raw.npz', raw), ('proposal.npz', proposal),
                         ('fixture.npz', fixture)]:
        np.savez(tmp_path / name, **values)
    freeze = {'raw_path': 'raw.npz', 'proposal_path': 'proposal.npz',
        'fixture_path': 'fixture.npz', 'E132_result_path': 'result.json',
        'E132_final_review_path': 'review.json',
        'input_sha256': {'raw.npz': 'sha'},
        'radii': [.005, .02, .05], 'options': e133.OPTIONS,
        'internal_wall_limit_s': 8, 'pre_call_reserve_s': 2,
        'primary_radius': .05, 'predicted_reduction_fraction': .01}
    monkeypatch.setattr(e133, 'ROOT', tmp_path)
    monkeypatch.setattr(e133, 'reviewed_inputs', lambda *_: (freeze, {}))
    monkeypatch.setattr(e133, 'reconstruct', lambda *_: generated_lp_base())
    monkeypatch.setattr(e133, 'threadpool_info', list)
    monkeypatch.setattr(e133, 'threadpool_limits', lambda **_: nullcontext())
    monkeypatch.setattr(e133, 'solve_radius', solve)
    return freeze


def test_transient_post_return_ledger_failure_invalidates_success(
        monkeypatch, tmp_path):
    state = {'returned': False, 'injected': 0}

    def returned_solve(_base, rho, _options, _out, index, *, progress):
        progress.update(index=index, rho=rho, status='returned',
            classification='optimal_valid', solver_requested=True,
            solver_returned=True, solver_failed_unknown=False,
            predicted_decrease_fraction=.02, predicted_decrease=.02)
        state['returned'] = True
        return progress

    _generated_run_setup(monkeypatch, tmp_path, returned_solve)
    original_save = e133.save

    def flaky_save(path, value):
        if (Path(path).name == 'advance-ledger.json' and state['returned']
                and state['injected'] == 0):
            state['injected'] += 1
            raise OSError('generated transient ledger failure')
        original_save(path, value)

    monkeypatch.setattr(e133, 'save', flaky_save)
    with pytest.raises(OSError, match='generated transient ledger failure'):
        e133.run('freeze', 'sha', 'review', 'sha', tmp_path / 'output')
    ledger = json.loads((tmp_path / 'output/advance-ledger.json').read_text())
    assert ledger['lp_requested'] == ledger['lp_returned'] == 1
    assert ledger['lp_post_return_error'] == ledger['lp_error'] == 1
    assert ledger['status'] == 'failed'
    assert not (tmp_path / 'output/result.json').exists()


def test_primary_solver_failure_survives_secondary_ledger_failure(
        monkeypatch, tmp_path):
    primary = RuntimeError('generated primary solver failure')
    state = {'failed': False, 'injected': 0}

    def failed_solve(_base, rho, _options, _out, index, *, progress):
        progress.update(index=index, rho=rho, status='error',
            classification='unresolved', solver_requested=True,
            solver_returned=False, solver_failed_unknown=True)
        state['failed'] = True
        raise primary

    _generated_run_setup(monkeypatch, tmp_path, failed_solve)
    original_save = e133.save

    def flaky_save(path, value):
        if (Path(path).name == 'advance-ledger.json' and state['failed']
                and state['injected'] == 0):
            state['injected'] += 1
            raise OSError('generated secondary ledger failure')
        original_save(path, value)

    monkeypatch.setattr(e133, 'save', flaky_save)
    with pytest.raises(RuntimeError) as caught:
        e133.run('freeze', 'sha', 'review', 'sha', tmp_path / 'output')
    assert caught.value is primary
    assert any('Secondary LP ledger save' in note for note in primary.__notes__)
    ledger = json.loads((tmp_path / 'output/advance-ledger.json').read_text())
    assert ledger['lp_requested'] == ledger['lp_failed_unknown'] == 1
    assert ledger['lp_returned'] == 0


def test_unlink_and_fallback_failure_preserve_primary(monkeypatch, tmp_path):
    e133.save(tmp_path / 'result.json', {'hypothesis_verdict': 'supported'})
    primary = RuntimeError('generated original primary')
    original_unlink = Path.unlink
    original_save = e133.save

    def failed_unlink(path, *args, **kwargs):
        if path.name == 'result.json':
            raise OSError('generated unlink failure')
        return original_unlink(path, *args, **kwargs)

    def failed_fallback(path, value):
        if Path(path).name == 'result.json':
            raise OSError('generated invalidation failure')
        original_save(path, value)

    monkeypatch.setattr(Path, 'unlink', failed_unlink)
    monkeypatch.setattr(e133, 'save', failed_fallback)
    returned = e133.finalize(tmp_path, {'status': 'failed'},
        {'internal_wall_limit_s': 8}, time.perf_counter(), primary)
    assert returned is primary
    assert any('result_removal' in note for note in primary.__notes__)
    assert any('result_invalidation_fallback' in note for note in primary.__notes__)
