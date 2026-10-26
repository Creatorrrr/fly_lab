"""E135 CPU-only LP of saved LF edge/gain and adaptation timescale derivatives."""

# ruff: noqa: BLE001  # preserve primary persistence/solver error and partial evidence

from __future__ import annotations

import argparse
import json
import math
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
from tools import analyze_c_banc_m3_lf_linearized_headroom as e133
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '4ee3396a119ad0abc446cad3aa40e3eec9f37908508287fac0cd9229889d0940'
SOURCE_FILES = ('tools/analyze_c_banc_m3_timescale_headroom.py',
                'tests/test_c_banc_timescale_headroom.py')
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
        raise ValueError('Exact E135 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    expected = {'schema': 'flylab.m3.e135.timescale-headroom-freeze.v1',
        'leg': 0, 'selected_original_index': 39,
        'log_tau_bounds': [-math.log(3), math.log(10)],
        'optimizer_cap': fit.OPTIMIZER_CAP, 'selector_cap': fit.SELECTOR_CAP,
        'response_atol': 1e-8, 'old_loss_jac_atol': 1e-10,
        'old_matrix_atol': 1e-12,
        'radii': [.005, .02, .05], 'primary_radius': .05,
        'solver': 'highs-ds', 'options': OPTIONS, 'lp_limit': 3,
        'linear_residual_limit': 1e-7, 'objective_consistency_limit': 1e-9,
        'predicted_reduction_fraction': .01,
        'internal_wall_limit_s': 8, 'external_wall_limit_s': 10,
        'supervisor_limit_s': 12, 'pre_call_reserve_s': 2,
        'prior_conservative_process_s': 1574,
        'prior_actual_conservative_s': 1573.5490344999707,
        'allocated_conservative_end_s': 1586,
        'prospective_cumulative_limit_s': 1742,
        'prior_closed_allocation_s': 1530,
        'original_processes': 1, 'retries': 0,
        'original_neural_evaluations': 0, 'CUDA_calls': 0,
        'geometry_calls': 0, 'body_calls': 0, 'fit_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'actual_model': 'gpt-5.6-sol', 'actual_effort': 'high',
        'intended_agent': '/root/m3_sol_e132_lf_constrained'}
    dimensions = {'contexts': 3, 'initial_samples': 99, 'samples': 2001,
        'joints': 7, 'muscles': 17, 'neurons': 69, 'edges': 273,
        'old_parameters': 290, 'parameters': 307, 'LP_variables': 308,
        'constraints': 4761}
    global_dimensions = {'neurons': 391, 'muscles': 96, 'edges': 1572,
        'old_parameters': 1668, 'parameters': 1764}
    paths = ('raw_path', 'proposal_path', 'fixture_path', 'E133_base_path',
        'E133_result_path', 'E133_final_review_path', 'E134_zero_path',
        'E134_parameters_path', 'E134_final_review_path',
        'E134_cuda_review_path', 'E134_reference_result_path',
        'E134_output_manifest_path')
    if (any(freeze.get(k) != v for k, v in expected.items())
            or freeze.get('dimensions') != dimensions
            or freeze.get('global_dimensions') != global_dimensions
            or len(freeze.get('input_sha256', {})) != 12
            or len(freeze.get('reused_source_sha256', {})) != 4
            or any(freeze.get(k) not in freeze['input_sha256'] for k in paths)
            or freeze.get('design_sha256') != fit.sha256(
                ROOT / freeze['design_path'])):
        raise ValueError('E135 frozen contract changed')
    for mapping in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[mapping].items():
            fit.checked_relative(relative, digest)
    if fit.sha256(source_review) != review_sha:
        raise ValueError('E135 source review SHA mismatch')
    review = json.loads(Path(source_review).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_timescale_headroom')
            is not True or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E135 two-source review required')
    for relative, digest in review['source_sha256'].items():
        fit.checked_relative(relative, digest)
    return freeze, review


def require_array(data, key, shape, dtype=np.float64):
    if (key not in data or np.shape(data[key]) != shape
            or np.asarray(data[key]).dtype != dtype):
        raise ValueError('E135 shape/dtype mismatch: ' + key)
    value = np.asarray(data[key])
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError('E135 nonfinite array: ' + key)
    return value


def tau_columns(tau_b, cotangent, q_coefficient, ltm_coefficient, local_ltm):
    """Pure saved-array chain rule with no fixed global dimensions."""
    c, samples, muscles = tau_b.shape
    startup = q_coefficient.shape[1]
    if (cotangent.shape != (c, samples, muscles)
            or q_coefficient.shape[0] != c
            or q_coefficient.shape[3] != muscles
            or ltm_coefficient.shape != (c, startup)
            or samples < startup + 1 or not 0 <= local_ltm < muscles):
        raise ValueError('E135 tau chain dimensions disagree')
    if any(not np.isfinite(x).all() for x in
           (tau_b, cotangent, q_coefficient, ltm_coefficient)):
        raise ValueError('E135 nonfinite tau chain input')
    loss = np.sum(cotangent * tau_b, axis=1)
    q = tau_b[:, 1:startup + 1, None, :] * q_coefficient
    ltm = (tau_b[:, 1:startup + 1, :] * ltm_coefficient[:, :, None] *
           (np.arange(muscles) == local_ltm)[None, None, :])
    return loss, q, ltm


def old_loss_from_full_b(full_b, cotangent, columns, local_map):
    selected = full_b[:, :, columns]
    if (selected.shape != cotangent[:, :, local_map].shape
            or not np.isfinite(selected).all()):
        raise ValueError('E135 old full-time B shape/finite mismatch')
    return np.sum(selected * cotangent[:, :, local_map], axis=1)


def extend_constraints(old, raw, tau_b, local_ltm):
    """Append tau derivatives before the unchanged epigraph column."""
    tau_loss, tau_q, tau_ltm = tau_columns(tau_b, raw['cotangent'],
        raw['q_coefficient'], raw['ltm_coefficient'], local_ltm)
    loss_jac = np.concatenate((raw['loss_jac'], tau_loss), axis=1)
    q_jac = np.concatenate((old['q_jac'], tau_q), axis=3)
    ltm_jac = np.concatenate((old['ltm_jac'], tau_ltm), axis=2)
    c, jac, ratios = fit.constraint_values(raw['losses'], loss_jac,
        raw['q_error'], raw['ltm_error'], q_jac, ltm_jac, old['t0'])
    gradient = np.r_[old['gradient'][:-1], np.zeros(tau_b.shape[-1]),
                     old['gradient'][-1]]
    return {'c': c, 'jac': jac, 'ratios': ratios, 'gradient': gradient,
            'tau_B': tau_b, 'tau_loss_jac': tau_loss,
            'tau_q_jac': tau_q, 'tau_ltm_jac': tau_ltm}


def reconstruct(raw, fixture, proposal, zero, parameters, saved_old, freeze):
    """Validate the identical LF point and append E134 logtau columns."""
    old = e133.reconstruct(raw, fixture, proposal)
    atol = freeze['response_atol']
    matrix_atol = freeze['old_matrix_atol']
    old_atol = freeze['old_loss_jac_atol']
    for key in ('vector', 'bounds', 'c', 'gradient', 'ratios'):
        reference = require_array(saved_old, key, np.shape(old[key]))
        if not np.allclose(old[key], reference, rtol=0, atol=matrix_atol):
            raise ValueError('E135 E133 saved base mismatch: ' + key)
    old_jac = require_array(saved_old, 'jac', (4761, 291))
    if not np.allclose(old['jac'], old_jac, rtol=0, atol=matrix_atol):
        raise ValueError('E135 E133 old Jacobian mismatch')
    motors = np.flatnonzero(fixture['leg_motor_mask'][0])
    muscles = np.flatnonzero(fixture['leg_muscle_mask'][0])
    edges = np.flatnonzero(fixture['leg_edge_mask'][0])
    for key, shape in (('base_delta', (1572,)), ('base_gain', (96,)),
                       ('zero_log_tau', (96,))):
        require_array(parameters, key, shape)
    for key, shape in (('lf_muscle_indices', (17,)),
                       ('lf_edge_indices', (273,)),
                       ('edge_ids', (1572,)), ('motor_ids', (391,)),
                       ('muscle_indices', (96,))):
        if (key not in parameters or np.shape(parameters[key]) != shape
                or np.asarray(parameters[key]).dtype.kind not in 'iu'):
            raise ValueError('E135 E134 parameter ID shape/dtype: ' + key)
    if ((len(motors), len(muscles), len(edges)) != (69, 17, 273)
            or not np.array_equal(parameters['lf_muscle_indices'], muscles)
            or not np.array_equal(parameters['lf_edge_indices'], edges)
            or not np.array_equal(parameters['edge_ids'], fixture['edge_ids'])
            or not np.array_equal(parameters['motor_ids'], fixture['motor_ids'])
            or not np.array_equal(parameters['muscle_indices'], np.arange(96))
            or not np.array_equal(parameters['base_delta'][edges], raw['delta'])
            or not np.array_equal(parameters['base_gain'][muscles], raw['gain'])
            or not np.array_equal(parameters['zero_log_tau'], np.zeros(96))):
        raise ValueError('E135 E134 global/local parameter identity mismatch')
    expected_map = np.r_[fixture['membership'][fixture['edge_rows']],
        np.arange(96), np.arange(96)]
    for key, expected in (('delta', parameters['base_delta']),
        ('gain', parameters['base_gain']),
        ('log_tau', parameters['zero_log_tau']),
        ('edge_ids', fixture['edge_ids']),
        ('motor_indices', np.arange(391)),
        ('muscle_indices', np.arange(96)),
        ('parameter_muscle_map', expected_map)):
        if key not in zero or not np.array_equal(zero[key], expected):
            raise ValueError('E135 E134 zero identity mismatch: ' + key)
    full_b = require_array(zero, 'B', (3, 2001, 1764))
    tau_full = require_array(zero, 'activation_log_tau_sensitivity',
                             (3, 2001, 96))
    require_array(zero, 'rate_log_tau_sensitivity', (3, 2001, 391))
    require_array(zero, 'adaptation_log_tau_sensitivity', (3, 2001, 391))
    tau_b = full_b[:, :, 1668 + muscles]
    if not np.allclose(tau_b, tau_full[:, :, muscles], rtol=0, atol=atol):
        raise ValueError('E135 tau B differs from activation sensitivity')
    response = [('rate', motors), ('adaptation', motors),
        ('rate_sensitivity', motors), ('adaptation_sensitivity', motors),
        ('activation', muscles), ('activation_sensitivity', muscles)]
    for key, indices in response:
        local = require_array(raw, key, (3, 2001, len(indices)))
        global_value = require_array(zero, key,
            (3, 2001, 391 if len(indices) == 69 else 96))
        if not np.allclose(local, global_value[:, :, indices], rtol=0, atol=atol):
            raise ValueError('E135 E134/E132 LF response mismatch: ' + key)
    old_command = require_array(raw, 'command', (3, 2001, 42))
    new_command = require_array(zero, 'command', (3, 2001, 42))
    old_tendon = require_array(raw, 'tendon', (3, 2001, 1))
    new_tendon = require_array(zero, 'tendon', (3, 2001, 6))
    old_branch = require_array(raw, 'branch_bits', (3, 40000, 69), np.uint8)
    new_branch = require_array(zero, 'branch_bits', (3, 40000, 391), np.uint8)
    if (not np.allclose(old_command[:, :, :7], new_command[:, :, :7],
                        rtol=0, atol=atol)
            or not np.allclose(old_tendon[:, :, 0], new_tendon[:, :, 0],
                               rtol=0, atol=atol)
            or not np.array_equal(old_branch, new_branch[:, :, motors])):
        raise ValueError('E135 E134/E132 LF command/tendon/branch mismatch')
    old_columns = np.r_[edges, 1572 + muscles]
    old_initial_b = require_array(raw, 'B_initial99', (3, 99, 290))
    if not np.allclose(old_initial_b, full_b[:, 1:100, old_columns],
                       rtol=0, atol=atol):
        raise ValueError('E135 E134/E132 initial B mismatch')
    cotangent = require_array(raw, 'cotangent', (3, 2001, 17))
    old_loss = old_loss_from_full_b(full_b, cotangent, old_columns,
                                     raw['parameter_muscle_map'])
    if not np.allclose(old_loss, raw['loss_jac'], rtol=0, atol=old_atol):
        raise ValueError('E135 old full-time B loss Jacobian mismatch')
    local_ltm = int(np.flatnonzero(muscles == fixture['ltm'][0])[0])
    extended = extend_constraints(old, raw, tau_b, local_ltm)
    c, jac, ratios = (extended[key] for key in ('c', 'jac', 'ratios'))
    if (c.shape != (4761,) or jac.shape != (4761, 308)
            or not np.isfinite(jac).all()
            or not np.allclose(c, old['c'], rtol=0, atol=matrix_atol)
            or not np.allclose(jac[:, :290], old_jac[:, :290],
                rtol=0, atol=matrix_atol)
            or not np.allclose(jac[:, -1], old_jac[:, -1],
                rtol=0, atol=matrix_atol)
            or not np.allclose(ratios, old['ratios'], rtol=0, atol=matrix_atol)):
        raise ValueError('E135 old constraint columns/c/ratios changed')
    tau_bounds = np.tile(np.array([[-math.log(3), math.log(10)]]), (17, 1))
    bounds = np.vstack((old['bounds'], tau_bounds))
    vector = np.r_[old['vector'], np.zeros(17)]
    gradient = extended['gradient']
    return {'vector': vector, 'bounds': bounds, 't0': old['t0'],
        'c': c, 'jac': jac, 'a_ub': -jac, 'b_ub': c,
        'gradient': gradient, 'ratios': ratios, 'prior': old['prior'],
        'true_worst_objective': old['true_worst_objective'],
        'tau_B': tau_b, 'tau_loss_jac': extended['tau_loss_jac'],
        'tau_q_jac': extended['tau_q_jac'],
        'tau_ltm_jac': extended['tau_ltm_jac']}


def radius_bounds(base, rho):
    original = np.asarray(base['bounds'], np.float64)
    vector = np.asarray(base['vector'], np.float64)
    span = original[:, 1] - original[:, 0]
    lower = np.maximum(original[:, 0] - vector, -rho * span)
    upper = np.minimum(original[:, 1] - vector, rho * span)
    if np.any(lower > upper + 1e-12):
        raise ValueError('E135 trust box intersection empty')
    return np.vstack((np.c_[lower, upper], [[-np.inf, np.inf]]))


def validate_solution(base, box, solution, fun, *, residual_limit=1e-7,
                      objective_limit=1e-9):
    y = np.asarray(solution, np.float64)
    if (y.shape != base['gradient'].shape or y.shape != (308,)
            or not np.isfinite(y).all() or not np.isfinite(fun)):
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


def answer_evidence(answer):
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
    return finite_evidence(partial)


def solve_radius(base, rho, options, out, index, *, solver=linprog,
                 progress=None, persist_transition=None):
    row = progress if progress is not None else {}
    row.update(index=index, rho=rho, status='preparing',
        solver='highs-ds', options=dict(options),
        solver_requested=False, solver_returned=False,
        solver_failed_unknown=False, solver_request_intent=False,
        solver_intent_persisted=False, solver_in_flight=False,
        solver_return_persisted=False)
    started = time.perf_counter()
    try:
        box = radius_bounds(base, rho)
        row.update(status='in_flight_unknown', solver_request_intent=True,
                   solver_intent_persisted=True, solver_in_flight=True)
        try:
            if persist_transition is not None:
                persist_transition(row, 'request_intent')
        except BaseException:
            row.update(solver_request_intent=False,
                       solver_intent_persisted=False, solver_in_flight=False)
            raise
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            row['solver_requested'] = True
            answer = solver(base['gradient'], A_ub=base['a_ub'],
                b_ub=base['b_ub'], bounds=[tuple(x) for x in box],
                method='highs-ds', options=dict(options))
            row['solver_returned'] = True
            row['solver_in_flight'] = False
            row['status'] = 'returned_before_raw'
            row['partial_answer'] = answer_evidence(answer)
            row['warnings'] = [str(x.message) for x in captured]
            row['solver_return_persisted'] = True
            try:
                if persist_transition is not None:
                    persist_transition(row, 'solver_returned')
            except BaseException:
                row['solver_return_persisted'] = False
                raise
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
        row['solver_in_flight'] = False
        if 'captured' in locals():
            row['warnings'] = [str(x.message) for x in captured]
        if row['solver_returned'] and 'answer' in locals() and 'partial_answer' not in row:
            try:
                row['partial_answer'] = answer_evidence(answer)
            except BaseException as evidence_error:
                error.add_note('Secondary partial-answer capture: ' +
                               repr(evidence_error))
        raise
    finally:
        row['wall_s'] = time.perf_counter() - started


def verdict(base, rows, freeze, e133_result=None):
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
    return {'schema': 'flylab.m3.e135.timescale-headroom-result.v1',
        'status': ('completed_known' if len(rows) == 3 and all(
            row.get('status') == 'returned' and row.get('classification') in
            ('optimal_valid', 'linearized_infeasible') for row in rows)
            else 'partial_unresolved'),
        'hypothesis_verdict': hypothesis,
        'interpretation': 'local_linearized_timescale_prediction_only',
        'primary_radius': freeze['primary_radius'],
        'primary_predicted_decrease_fraction': decrease,
        'primary_predicted_decrease': (primary.get('predicted_decrease')
                                       if valid else None),
        'E133_primary_predicted_decrease_fraction': (
            e133_result.get('primary_predicted_decrease_fraction')
            if e133_result is not None else None),
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
        'original_neural_evaluations': 0, 'CUDA_calls': 0,
        'geometry_calls': 0, 'body_calls': 0, 'fit_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False, 'whole_goal_complete': False}


def write_manifest(out):
    save(out / 'output-manifest.json', {'schema': 'flylab.m3.e135.output-manifest.v1',
        'files': {str(p.relative_to(out)).replace('\\', '/'): fit.sha256(p)
            for p in sorted(out.rglob('*')) if p.is_file()
            and p.name != 'output-manifest.json'}})


def may_start_lp(now, began, freeze):
    return now < (began + freeze['internal_wall_limit_s'] -
                  freeze['pre_call_reserve_s'])


def refresh_lp_counts(ledger):
    rows = ledger['lp_rows']
    ledger['lp_request_intents'] = sum(
        x.get('solver_request_intent') is True for x in rows)
    ledger['lp_in_flight_unknown'] = sum(
        x.get('solver_in_flight') is True and
        x.get('solver_returned') is not True for x in rows)
    ledger['lp_return_persisted'] = sum(
        x.get('solver_return_persisted') is True for x in rows)
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
                'E135 internal wall limit exceeded after manifest'))
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
        raise FileExistsError('One-shot E135 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e135.timescale-headroom-ledger.v1',
        'pid': os.getpid(), 'status': 'preflight',
        'freeze_sha256': freeze_sha, 'source_review_sha256': source_review_sha,
        'lp_requested': 0, 'lp_returned': 0, 'lp_skipped': 0, 'lp_error': 0,
        'lp_failed_unknown': 0, 'lp_post_return_error': 0,
        'lp_request_intents': 0, 'lp_in_flight_unknown': 0,
        'lp_return_persisted': 0,
        'lp_rows': [], 'original_neural_evaluations': 0, 'CUDA_calls': 0,
        'geometry_calls': 0, 'body_calls': 0, 'fit_calls': 0, 'F6_exposed': False,
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
              np.load(ROOT / freeze['fixture_path'], allow_pickle=False) as f,
              np.load(ROOT / freeze['E134_zero_path'], allow_pickle=False) as z,
              np.load(ROOT / freeze['E134_parameters_path'],
                      allow_pickle=False) as par,
              np.load(ROOT / freeze['E133_base_path'], allow_pickle=False) as b):
            raw = {k: r[k] for k in r.files}
            proposal = {k: p[k] for k in p.files}
            fixture = {k: f[k] for k in ('leg_motor_mask', 'leg_muscle_mask',
                'leg_edge_mask', 'edge_ids', 'edge_rows', 'membership',
                'moment', 'ltm', 'delta_bounds', 'motor_ids')}
            zero = {k: z[k] for k in z.files}
            parameters = {k: par[k] for k in par.files}
            saved_old = {k: b[k] for k in ('vector', 'bounds', 'c', 'jac',
                'gradient', 'ratios')}
        e133_result = json.loads((ROOT / freeze['E133_result_path']).read_text(
            encoding='utf-8'))
        e133_review = json.loads((ROOT / freeze['E133_final_review_path']).read_text(
            encoding='utf-8'))
        e134_review = json.loads((ROOT / freeze['E134_final_review_path']).read_text(
            encoding='utf-8'))
        if (e133_result.get('status') != 'completed_known'
                or e133_review.get('validity') != 'valid'
                or e134_review.get('validity') != 'valid'
                or str(proposal['source_raw_sha256']) !=
                    freeze['input_sha256'][freeze['raw_path']]):
            raise ValueError('E135 frozen predecessor receipts inconsistent')
        base = reconstruct(raw, fixture, proposal, zero, parameters,
                           saved_old, freeze)
        if not np.isclose(base['true_worst_objective'],
                          e133_result['base_true_worst_objective'],
                          rtol=0, atol=1e-12):
            raise ValueError('E135 selected original objective mismatch')
        if np.count_nonzero(base['c'] < -1e-12) != 8:
            raise ValueError('E135 frozen initial optimizer violations changed')
        base_path = out / 'timescale-base.npz'
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
                raise RuntimeError('E135 CPU/BLAS single-thread limit unavailable')
            technical_stop = None

            def persist_transition(_row, _stage):
                refresh_lp_counts(ledger)
                save(out / 'advance-ledger.json', ledger)

            for index, rho in enumerate(freeze['radii'], 1):
                if technical_stop is not None:
                    row = {'index': index, 'rho': rho, 'status': 'not_run',
                        'classification': 'unresolved',
                        'reason': 'prior_technical_lp_result', 'wall_s': 0.}
                    ledger['lp_rows'].append(row)
                    refresh_lp_counts(ledger)
                    save(out / 'advance-ledger.json', ledger)
                    continue
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
                                 progress=row,
                                 persist_transition=persist_transition)
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
                if row.get('classification') in ('unresolved', 'invalid_numerical'):
                    technical_stop = {'index': index, 'rho': rho,
                        'classification': row['classification'],
                        'solver_status': row.get('solver_status'),
                        'message': row.get('message')}
                    ledger['technical_stop'] = technical_stop
                    save(out / 'advance-ledger.json', ledger)
        ledger['threadpool_after'] = threadpool_info()
        if technical_stop is not None:
            ledger['hypothesis_verdict'] = 'inconclusive'
            save(out / 'advance-ledger.json', ledger)
            raise RuntimeError('E135 technical LP result unresolved at radius ' +
                               str(technical_stop['rho']))
        result = verdict(base, ledger['lp_rows'], freeze, e133_result)
        result.update(lp_requested=ledger['lp_requested'],
            lp_returned=ledger['lp_returned'], lp_skipped=ledger['lp_skipped'],
            lp_error=ledger['lp_error'],
            lp_failed_unknown=ledger['lp_failed_unknown'],
            lp_post_return_error=ledger['lp_post_return_error'],
            lp_request_intents=ledger['lp_request_intents'],
            lp_in_flight_unknown=ledger['lp_in_flight_unknown'],
            lp_return_persisted=ledger['lp_return_persisted'],
            freeze_sha256=freeze_sha, source_review_sha256=source_review_sha)
        if time.perf_counter() >= started + freeze['internal_wall_limit_s']:
            raise TimeoutError('E135 internal result reserve exhausted')
        save(out / 'result.json', result)
        ledger['status'] = result['status']
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['hypothesis_verdict'] = 'inconclusive'
        if freeze is not None:
            represented = {row['index'] for row in ledger['lp_rows']}
            for index, rho in enumerate(freeze['radii'], 1):
                if index not in represented:
                    ledger['lp_rows'].append({'index': index, 'rho': rho,
                        'status': 'not_run', 'classification': 'unresolved',
                        'reason': 'prior_exception', 'wall_s': 0.})
            refresh_lp_counts(ledger)
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
    for name in ('freeze', 'freeze-sha256', 'source-review',
                 'source-review-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.source_review,
        args.source_review_sha256, args.output)


if __name__ == '__main__':
    main()
