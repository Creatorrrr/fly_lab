"""E133 CPU-only LP of saved LF derivatives; no neural or body execution."""

# ruff: noqa: BLE001  # preserve primary persistence/solver error and partial evidence

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '59a5b906c30c78ecbbb4e91a5834b5d8d07e1a76e7d9cc441822209d97e19c12'
SOURCE_FILES = ('tools/analyze_c_banc_m3_lf_linearized_headroom.py',
                'tests/test_c_banc_lf_linearized_headroom.py')
OPTIONS = {'presolve': True, 'primal_feasibility_tolerance': 1e-9,
           'dual_feasibility_tolerance': 1e-9, 'time_limit': 1.0, 'threads': 1}


def save(path, value):
    fit.write_json(path, fit.jsonable(value))


def finite_evidence(value):
    """Keep partial solver data JSON-safe even when an answer is nonfinite."""
    if isinstance(value, np.ndarray):
        return finite_evidence(value.tolist())
    if isinstance(value, np.generic):
        return finite_evidence(value.item())
    if isinstance(value, dict):
        return {key: finite_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_evidence(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def reviewed_inputs(freeze_path, freeze_sha, source_review, review_sha):
    if freeze_sha != FREEZE_SHA or fit.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E133 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    expected = {'schema': 'flylab.m3.e133.lf-linearized-headroom-freeze.v1',
        'leg': 0, 'selected_original_index': 39,
        'optimizer_cap': fit.OPTIMIZER_CAP, 'selector_cap': fit.SELECTOR_CAP,
        'radii': [.005, .02, .05], 'primary_radius': .05,
        'solver': 'highs-ds', 'options': OPTIONS, 'lp_limit': 3,
        'linear_residual_limit': 1e-7, 'objective_consistency_limit': 1e-9,
        'predicted_reduction_fraction': .01,
        'internal_wall_limit_s': 8, 'external_wall_limit_s': 10,
        'supervisor_limit_s': 12, 'pre_call_reserve_s': 2,
        'prior_conservative_process_s': 1514,
        'allocated_conservative_end_s': 1526, 'shared_process_limit_s': 1530,
        'original_processes': 1, 'retries': 0,
        'original_neural_evaluations': 0, 'geometry_calls': 0,
        'body_calls': 0, 'F6_exposed': False, 'candidate_adopted': False}
    dimensions = {'contexts': 3, 'initial_samples': 99, 'joints': 7,
                  'muscles': 17, 'edges': 273, 'parameters': 290,
                  'constraints': 4761}
    paths = ('raw_path', 'proposal_path', 'E132_result_path',
             'E132_final_review_path', 'fixture_path')
    if (any(freeze.get(k) != v for k, v in expected.items())
            or freeze.get('dimensions') != dimensions
            or len(freeze.get('input_sha256', {})) != 8
            or len(freeze.get('reused_source_sha256', {})) != 2
            or any(freeze.get(k) not in freeze['input_sha256'] for k in paths)
            or freeze.get('design_sha256') != fit.sha256(
                ROOT / freeze['design_path'])):
        raise ValueError('E133 frozen contract changed')
    for mapping in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[mapping].items():
            fit.checked_relative(relative, digest)
    if fit.sha256(source_review) != review_sha:
        raise ValueError('E133 source review SHA mismatch')
    review = json.loads(Path(source_review).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_lf_linearized_headroom')
            is not True or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E133 two-source review required')
    for relative, digest in review['source_sha256'].items():
        fit.checked_relative(relative, digest)
    return freeze, review


def reconstruct(raw, fixture, proposal):
    """Rebuild E128 startup Jacobians using local muscle axes and saved B."""
    required = {'delta': (273,), 'gain': (17,), 'losses': (3,),
        'loss_jac': (3, 290), 'q_error': (3, 99, 7),
        'ltm_error': (3, 99), 'B_initial99': (3, 99, 290),
        'q_coefficient': (3, 99, 7, 17),
        'ltm_coefficient': (3, 99), 'ratios': (3, 4),
        'parameter_muscle_map': (290,), 'edge_ids': (273,),
        'motor_indices': (69,), 'muscle_indices': (17,)}
    for key, shape in required.items():
        if key not in raw or np.shape(raw[key]) != shape:
            raise ValueError('E133 saved raw shape changed: ' + key)
    for key in ('delta', 'gain', 'B_initial99', 'q_coefficient',
                'ltm_coefficient', 'losses', 'loss_jac', 'q_error',
                'ltm_error', 'ratios'):
        if np.asarray(raw[key]).dtype != np.float64 or not np.isfinite(raw[key]).all():
            raise ValueError('E133 finite float64 raw required: ' + key)
    mapping = np.asarray(raw['parameter_muscle_map'])
    if mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or np.any(mapping >= 17):
        raise ValueError('E133 invalid parameter muscle map')
    motors = np.flatnonzero(fixture['leg_motor_mask'][0])
    muscles = np.flatnonzero(fixture['leg_muscle_mask'][0])
    edges = np.flatnonzero(fixture['leg_edge_mask'][0])
    if ((len(motors), len(muscles), len(edges)) != (69, 17, 273)
            or not np.array_equal(raw['motor_indices'], motors)
            or not np.array_equal(raw['muscle_indices'], muscles)
            or not np.array_equal(raw['edge_ids'], fixture['edge_ids'][edges])
            or np.any(~np.isin(fixture['edge_rows'][edges], motors))):
        raise ValueError('E133 LF ID/mask mismatch')
    expected_local = np.searchsorted(muscles,
        fixture['membership'][fixture['edge_rows'][edges]])
    if (np.any(expected_local >= 17)
            or not np.array_equal(muscles[expected_local],
                fixture['membership'][fixture['edge_rows'][edges]])
            or not np.array_equal(mapping,
                np.r_[expected_local, np.arange(17)])):
        raise ValueError('E133 parameter-to-local-muscle map mismatch')
    moment = .02 * np.asarray(fixture['moment'][:7, muscles], np.float64)
    expected_coeff = np.broadcast_to(moment[None, None, :, :], (3, 99, 7, 17))
    if not np.array_equal(raw['q_coefficient'], expected_coeff):
        raise ValueError('E133 local q coefficient differs from fixture moment')
    ltm = int(fixture['ltm'][0])
    location = np.flatnonzero(muscles == ltm)
    if location.size != 1:
        raise ValueError('E133 LF LTM muscle mapping changed')
    local_ltm = int(location[0])
    vector = np.r_[raw['delta'], raw['gain']]
    if (vector.shape != (290,) or any(np.asarray(proposal[k]).dtype != np.float64
        or not np.array_equal(proposal[k], raw[k]) for k in ('delta', 'gain'))
        or int(proposal['original_evaluation_index']) != 39):
        raise ValueError('E133 selected proposal/vector mismatch')
    bounds = np.vstack((np.asarray(fixture['delta_bounds'][edges], np.float64),
        np.tile(np.array([[0., 8.]]), (17, 1))))
    if (bounds.shape != (290, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])
            or np.any(vector < bounds[:, 0] - 1e-12)
            or np.any(vector > bounds[:, 1] + 1e-12)):
        raise ValueError('E133 selected vector outside original bounds')
    bank = raw['B_initial99']
    q_jac = bank[:, :, None, :] * np.take(raw['q_coefficient'], mapping, axis=3)
    ltm_jac = bank * raw['ltm_coefficient'][:, :, None] * (
        mapping == local_ltm)[None, None, :]
    t0 = float(max(raw['losses']))
    c, jac, ratios = fit.constraint_values(raw['losses'], raw['loss_jac'],
        raw['q_error'], raw['ltm_error'], q_jac, ltm_jac, t0)
    if (c.shape != (4761,) or jac.shape != (4761, 291)
            or not np.isfinite(c).all() or not np.isfinite(jac).all()
            or not np.allclose(ratios, raw['ratios'], rtol=0, atol=1e-12)
            or not fit.selector_feasible(ratios)):
        raise ValueError('E133 saved startup ratio/selector mismatch')
    _, gradient, prior = fit.objective_and_jac(raw, t0, 273)
    if gradient.shape != (291,) or not np.isfinite(gradient).all():
        raise ValueError('E133 objective gradient invalid')
    return {'vector': vector, 'bounds': bounds, 't0': t0, 'c': c,
            'jac': jac, 'a_ub': -jac, 'b_ub': c, 'gradient': gradient,
            'ratios': ratios, 'prior': prior,
            'true_worst_objective': t0 + prior,
            'q_jac': q_jac, 'ltm_jac': ltm_jac}


def radius_bounds(base, rho):
    original = np.asarray(base['bounds'], np.float64)
    vector = np.asarray(base['vector'], np.float64)
    span = original[:, 1] - original[:, 0]
    lower = np.maximum(original[:, 0] - vector, -rho * span)
    upper = np.minimum(original[:, 1] - vector, rho * span)
    if np.any(lower > upper + 1e-12):
        raise ValueError('E133 trust box intersection empty')
    return np.vstack((np.c_[lower, upper], [[-np.inf, np.inf]]))


def validate_solution(base, box, solution, fun, *, residual_limit=1e-7,
                      objective_limit=1e-9):
    y = np.asarray(solution, np.float64)
    if y.shape != (291,) or not np.isfinite(y).all() or not np.isfinite(fun):
        return {'valid': False, 'reason': 'missing_or_nonfinite_solution'}
    linear_violation = float(max(0., np.max(base['a_ub'] @ y - base['b_ub'])))
    box_violation = float(max(0., np.max(box[:, 0] - y),
                              np.max(y - box[:, 1])))
    independent_fun = float(base['gradient'] @ y)
    fun_error = abs(independent_fun - float(fun))
    return {'valid': bool(linear_violation <= residual_limit
        and box_violation <= residual_limit and fun_error <= objective_limit),
        'max_linear_violation': linear_violation,
        'max_box_violation': box_violation,
        'objective_error': fun_error, 'independent_fun': independent_fun}


def solve_radius(base, rho, options, out, index, *, solver=linprog,
                 progress=None):
    row = progress if progress is not None else {}
    row.update(index=index, rho=rho, status='preparing',
        solver='highs-ds', options=dict(options),
        solver_requested=False, solver_returned=False,
        solver_failed_unknown=False)
    started = time.perf_counter()
    try:
        box = radius_bounds(base, rho)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            row['solver_requested'] = True
            answer = solver(base['gradient'], A_ub=base['a_ub'],
                b_ub=base['b_ub'], bounds=[tuple(x) for x in box],
                method='highs-ds', options=dict(options))
            row['solver_returned'] = True
        row['warnings'] = [str(x.message) for x in captured]
        reported_fun = (float(answer.fun) if answer.fun is not None else None)
        if reported_fun is not None and not np.isfinite(reported_fun):
            reported_fun = None
        row.update(status='returned', solver_status=int(answer.status),
            message=str(answer.message), success=bool(answer.success),
            nit=int(answer.nit), fun=reported_fun)
        x = (np.asarray(answer.x, np.float64) if answer.x is not None
             else np.empty(0, np.float64))
        slack = (np.asarray(answer.slack, np.float64) if answer.slack is not None
                 else np.empty(0, np.float64))
        path = out / f'lp-radius-{index:03d}-raw.npz'
        saved = {'x': x, 'slack': slack, 'box': box,
            'solver_status': np.array(answer.status),
            'solver_fun': np.array(row['fun'] if row['fun'] is not None else np.nan)}
        for section in ('ineqlin', 'lower', 'upper'):
            reported = getattr(answer, section, None)
            if reported is not None:
                for field in ('residual', 'marginals'):
                    value = reported.get(field)
                    if value is not None:
                        saved[f'{section}_{field}'] = np.asarray(value, np.float64)
        np.savez(path, **saved)
        row.update(raw_path=str(path), raw_sha256=fit.sha256(path))
        if answer.status == 0:
            row['residual'] = validate_solution(base, box, x,
                row['fun'] if row['fun'] is not None else np.nan)
            row['classification'] = ('optimal_valid' if answer.success
                                     and row['residual']['valid']
                                     else 'invalid_numerical')
            if row['residual']['valid']:
                row['predicted_decrease'] = float(-row['residual']['independent_fun'])
                row['predicted_decrease_fraction'] = float(
                    -row['residual']['independent_fun'] /
                    base['true_worst_objective'])
        elif answer.status == 2:
            row['classification'] = 'linearized_infeasible'
        else:
            row['classification'] = 'unresolved'
        return row
    except BaseException as error:
        row.update(status='error', classification='unresolved', error=repr(error),
                   traceback=traceback.format_exc())
        row['solver_failed_unknown'] = bool(
            row['solver_requested'] and not row['solver_returned'])
        row['post_return_error'] = bool(row['solver_returned'])
        if 'captured' in locals():
            row['warnings'] = [str(x.message) for x in captured]
        if row['solver_returned'] and 'answer' in locals():
            partial = {'status': getattr(answer, 'status', None),
                'message': str(getattr(answer, 'message', '')),
                'nit': getattr(answer, 'nit', None),
                'fun': getattr(answer, 'fun', None)}
            for key in ('x', 'slack'):
                value = getattr(answer, key, None)
                if value is not None:
                    partial[key] = np.asarray(value, np.float64)
            for section in ('ineqlin', 'lower', 'upper'):
                reported = getattr(answer, section, None)
                if reported is not None:
                    partial[section] = {field: reported.get(field) for field in
                        ('residual', 'marginals') if reported.get(field) is not None}
            row['partial_answer'] = finite_evidence(partial)
        raise
    finally:
        row['wall_s'] = time.perf_counter() - started


def verdict(base, rows, freeze):
    primary = rows[-1] if len(rows) == 3 else None
    valid = bool(primary and primary.get('rho') == freeze['primary_radius']
        and primary.get('classification') == 'optimal_valid')
    decrease = primary.get('predicted_decrease_fraction') if valid else None
    if valid and decrease >= freeze['predicted_reduction_fraction']:
        hypothesis = 'supported'
    elif valid:
        hypothesis = 'refuted'
    else:
        hypothesis = 'inconclusive'
    original = base['bounds']
    x = base['vector']
    slack = np.minimum(x - original[:, 0], original[:, 1] - x)
    c = base['c']
    return {'schema': 'flylab.m3.e133.lf-linearized-headroom-result.v1',
        'status': ('completed_known' if len(rows) == 3 and all(
            row.get('status') == 'returned' and row.get('classification') in
            ('optimal_valid', 'linearized_infeasible') for row in rows)
            else 'partial_unresolved'),
        'hypothesis_verdict': hypothesis,
        'interpretation': 'local_linearized_prediction_only',
        'primary_radius': freeze['primary_radius'],
        'primary_predicted_decrease_fraction': decrease,
        'primary_predicted_decrease': (primary.get('predicted_decrease')
                                       if valid else None),
        'base_true_worst_objective': base['true_worst_objective'],
        'base_max_ratio': float(np.max(base['ratios'])),
        'base_selector_feasible': fit.selector_feasible(base['ratios']),
        'base_optimizer_cap_feasible': bool(np.all(c >= -1e-12)),
        'base_optimizer_cap_violated_rows': int(np.count_nonzero(c < -1e-12)),
        'base_min_inequality': float(np.min(c)),
        'lp_classifications': [row.get('classification') for row in rows],
        'active_parameter_bounds_1e_8': int(np.count_nonzero(slack <= 1e-8)),
        'near_active_startup_constraints_1e_6': int(
            np.count_nonzero(c[3:] <= 1e-6)),
        'original_neural_evaluations': 0, 'geometry_calls': 0,
        'body_calls': 0, 'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False, 'whole_goal_complete': False}


def write_manifest(out):
    save(out / 'output-manifest.json', {'schema': 'flylab.m3.e133.output-manifest.v1',
        'files': {str(p.relative_to(out)).replace('\\', '/'): fit.sha256(p)
            for p in sorted(out.rglob('*')) if p.is_file()
            and p.name != 'output-manifest.json'}})


def may_start_lp(now, began, freeze):
    return now < (began + freeze['internal_wall_limit_s'] -
                  freeze['pre_call_reserve_s'])


def refresh_lp_counts(ledger):
    rows = ledger['lp_rows']
    ledger['lp_requested'] = sum(x.get('solver_requested') is True for x in rows)
    ledger['lp_returned'] = sum(x.get('solver_returned') is True for x in rows)
    ledger['lp_failed_unknown'] = sum(x.get('solver_failed_unknown') is True
                                       for x in rows)
    ledger['lp_post_return_error'] = sum(x.get('post_return_error') is True
                                         for x in rows)
    ledger['lp_skipped'] = sum(x.get('status') == 'not_run' for x in rows)
    ledger['lp_error'] = sum(x.get('status') == 'error' for x in rows)


def finalize(out, ledger, freeze, started, primary):
    """Save final accounting and manifest before returning; revoke late success."""
    def note(stage, error):
        nonlocal primary
        if primary is None:
            primary = error
        else:
            primary.add_note('Secondary ' + stage + ': ' + repr(error))

    for _ in range(2):
        ledger['process_wall_s'] = time.perf_counter() - started
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('final_ledger', error)
        try:
            write_manifest(out)
        except BaseException as error:
            note('final_manifest', error)
        if (freeze is not None and time.perf_counter() - started >
                freeze['internal_wall_limit_s']):
            note('post_manifest_timeout', TimeoutError(
                'E133 internal wall limit exceeded after manifest'))
        if primary is not None:
            break
        ledger['post_manifest_wall_s'] = time.perf_counter() - started
    if primary is not None:
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(primary).__name__,
            'message': str(primary), 'notes': list(getattr(primary, '__notes__', []))}
        path = out / 'result.json'
        if path.exists():
            try:
                path.unlink()
            except BaseException as error:
                note('result_removal', error)
                try:
                    save(path, {'status': 'invalid_finalization',
                                'hypothesis_verdict': 'inconclusive'})
                except BaseException as fallback:
                    note('result_invalidation_fallback', fallback)
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('failed_ledger', error)
        try:
            write_manifest(out)
        except BaseException as error:
            note('failed_manifest', error)
    return primary


def run(freeze_path, freeze_sha, source_review, source_review_sha, out):
    started = time.perf_counter()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('One-shot E133 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e133.lf-linearized-headroom-ledger.v1',
        'pid': os.getpid(), 'status': 'preflight',
        'freeze_sha256': freeze_sha, 'source_review_sha256': source_review_sha,
        'lp_requested': 0, 'lp_returned': 0, 'lp_skipped': 0, 'lp_error': 0,
        'lp_failed_unknown': 0, 'lp_post_return_error': 0,
        'lp_rows': [], 'original_neural_evaluations': 0,
        'geometry_calls': 0, 'body_calls': 0, 'F6_exposed': False,
        'candidate_adopted': False}
    save(out / 'advance-ledger.json', ledger)
    freeze = None
    primary = None
    try:
        freeze, review = reviewed_inputs(
            freeze_path, freeze_sha, source_review, source_review_sha)
        save(out / 'source-review-snapshot.json', review)
        with (np.load(ROOT / freeze['raw_path'], allow_pickle=False) as r,
              np.load(ROOT / freeze['proposal_path'], allow_pickle=False) as p,
              np.load(ROOT / freeze['fixture_path'], allow_pickle=False) as f):
            raw = {k: r[k] for k in r.files}
            proposal = {k: p[k] for k in p.files}
            fixture = {k: f[k] for k in ('leg_motor_mask', 'leg_muscle_mask',
                'leg_edge_mask', 'edge_ids', 'edge_rows', 'membership',
                'moment', 'ltm', 'delta_bounds')}
        e132_result = json.loads((ROOT / freeze['E132_result_path']).read_text(
            encoding='utf-8'))
        e132_review = json.loads((ROOT / freeze['E132_final_review_path']).read_text(
            encoding='utf-8'))
        if (e132_result.get('new_lf_proposal_accepted') is not True
                or e132_result.get('selected_index') != 39
                or e132_review.get('validity') != 'valid'
                or e132_review.get('selected_raw_sha256') !=
                    freeze['input_sha256'][freeze['raw_path']]
                or str(proposal['source_raw_sha256']) !=
                    freeze['input_sha256'][freeze['raw_path']]):
            raise ValueError('E133 selected E132 receipt mismatch')
        base = reconstruct(raw, fixture, proposal)
        if not np.isclose(base['true_worst_objective'],
                          e132_result['selected_objective'], rtol=0, atol=1e-10):
            raise ValueError('E133 selected original objective mismatch')
        base_path = out / 'linearized-base.npz'
        np.savez(base_path, **base)
        ledger['base_raw_path'] = str(base_path)
        ledger['base_raw_sha256'] = fit.sha256(base_path)
        ledger['threadpool_before'] = threadpool_info()
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        with threadpool_limits(limits=1):
            ledger['threadpool_during'] = threadpool_info()
            if any(item.get('num_threads') != 1 for item in
                   ledger['threadpool_during']):
                raise RuntimeError('E133 CPU/BLAS single-thread limit unavailable')
            for index, rho in enumerate(freeze['radii'], 1):
                if not may_start_lp(time.perf_counter(), started, freeze):
                    row = {'index': index, 'rho': rho, 'status': 'not_run',
                        'classification': 'unresolved',
                        'reason': 'pre_call_wall_reserve', 'wall_s': 0.}
                    ledger['lp_rows'].append(row)
                    refresh_lp_counts(ledger)
                    save(out / 'advance-ledger.json', ledger)
                    continue
                row = {'index': index, 'rho': rho, 'status': 'pending',
                    'solver_requested': False, 'solver_returned': False,
                    'solver_failed_unknown': False}
                ledger['lp_rows'].append(row)
                save(out / 'advance-ledger.json', ledger)
                call_started = time.perf_counter()
                case_primary = None
                try:
                    solve_radius(base, rho, freeze['options'], out, index,
                                 progress=row)
                except BaseException as error:
                    case_primary = error
                    row.update(status='error', classification='unresolved',
                        error=repr(error), traceback=traceback.format_exc())
                finally:
                    row['wall_s'] = time.perf_counter() - call_started
                    refresh_lp_counts(ledger)
                    try:
                        save(out / 'advance-ledger.json', ledger)
                    except BaseException as persistence_error:
                        if case_primary is None:
                            case_primary = persistence_error
                            row.update(status='error', classification='unresolved',
                                error=repr(persistence_error),
                                post_return_error=bool(row['solver_returned']))
                            refresh_lp_counts(ledger)
                            try:
                                save(out / 'advance-ledger.json', ledger)
                            except BaseException as retry_error:
                                case_primary.add_note('Secondary LP ledger retry: ' +
                                                      repr(retry_error))
                        else:
                            case_primary.add_note('Secondary LP ledger save: ' +
                                                  repr(persistence_error))
                if case_primary is not None:
                    raise case_primary.with_traceback(case_primary.__traceback__)
        ledger['threadpool_after'] = threadpool_info()
        result = verdict(base, ledger['lp_rows'], freeze)
        result.update(lp_requested=ledger['lp_requested'],
            lp_returned=ledger['lp_returned'], lp_skipped=ledger['lp_skipped'],
            lp_error=ledger['lp_error'],
            lp_failed_unknown=ledger['lp_failed_unknown'],
            lp_post_return_error=ledger['lp_post_return_error'],
            freeze_sha256=freeze_sha, source_review_sha256=source_review_sha)
        if time.perf_counter() >= started + freeze['internal_wall_limit_s']:
            raise TimeoutError('E133 internal result reserve exhausted')
        save(out / 'result.json', result)
        ledger['status'] = result['status']
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'notes': list(getattr(error, '__notes__', []))}
    finally:
        primary = finalize(out, ledger, freeze, started, primary)
    if primary is not None:
        raise primary
    return out / 'result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha', 'source-review',
                 'source-review-sha', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.source_review,
        args.source_review_sha, args.out)


if __name__ == '__main__':
    main()
