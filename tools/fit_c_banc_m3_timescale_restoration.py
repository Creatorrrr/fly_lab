"""E137 bounded LF startup restoration with a fixed original-objective guard.

The parent dispatches the one original CUDA process only after source review.
"""

# ruff: noqa: BLE001  # preserve primary CUDA and persistence exceptions

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_adaptation_timescale as tau_module
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import fit_c_banc_m3_timescale_constrained as e136

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '92588cc6114a2fdf1a4954a68f229d44481b2e6333df2e73afcede3ce4c652fc'
SOURCE_FILES = ('tools/fit_c_banc_m3_timescale_restoration.py',
                'tests/test_c_banc_timescale_restoration.py')
OPTIONS = {'maxiter': 100, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10}
EXACT_RAW = ('delta', 'gain', 'log_tau', 'edge_ids', 'motor_indices',
    'muscle_indices', 'parameter_muscle_map', 'branch_bits')
RESPONSE_RAW = (*tau_module.FIELDS, 'B', 'B_initial99',
    'context0_fit_feet', 'context1_fit_feet', 'context2_fit_feet')
METRIC_RAW = ('losses', 'parts', 'ratios', 'loss_jac', 'q_error',
    'ltm_error', 'q_coefficient', 'ltm_coefficient', 'cotangent',
    'q_jac', 'ltm_jac')


def save(path, value):
    e136.save(path, value)


def checked_path(relative, digest):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if (name.is_absolute() or '..' in name.parts or
            not path.is_relative_to(ROOT.resolve()) or
            not isinstance(digest, str) or len(digest) != 64 or
            fit.sha256(path) != digest):
        raise ValueError('E137 frozen path/SHA mismatch: ' + str(relative))
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or fit.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E137 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    fixed = {'schema': 'flylab.m3.e137.timescale-restoration-freeze.v1',
        'leg': 0, 'start_evaluation_index': 45,
        'baseline_evaluation_index': 1,
        'baseline_objective': 1.1371950039105376,
        'start_objective': 1.1078584269431968,
        'original_objective_guard': 1.1258230538714322,
        'proposal_objective_ratio_limit': .99,
        'solver': 'L-BFGS-B', 'solver_options': OPTIONS,
        'optimizer_cap': fit.OPTIMIZER_CAP,
        'selector_cap': fit.SELECTOR_CAP,
        'max_unique_evaluations': 18, 'original_processes': 1,
        'fit_calls': 1, 'retries': 0, 'fit_wall_limit_s': 40,
        'leg_reserve_s': 3, 'process_reserve_s': 5,
        'internal_wall_limit_s': 48, 'external_wall_limit_s': 52,
        'supervisor_limit_s': 56, 'maximum_raw_bytes': 1073741824,
        'full391_batches': 0, 'full_BANC_calls': 0, 'body_calls': 0,
        'F6_exposed': False, 'whole_candidate_adoption': False,
        'active_UI_changed': False}
    dimensions = {'contexts': 3, 'neurons': 69, 'muscles': 17,
        'edges': 273, 'parameters': 307, 'constraints': 4761,
        'startup_constraints': 4758, 'guard_constraints': 3,
        'stages': 160000, 'ticks': 40000, 'samples': 2001}
    maximum = {key: 18 * value for key, value in
               e136.WORK_PER_EVALUATION.items()}
    if (any(freeze.get(key) != value for key, value in fixed.items()) or
            freeze.get('dimensions') != dimensions or
            freeze.get('maximum_original_work') != maximum or
            freeze.get('log_tau_bounds') != [-math.log(3), math.log(10)] or
            freeze.get('tiny') != {'batches': 1, 'context_ticks': 20,
                'conditional_s': .005, 'derivative_banks': 1} or
            len(freeze.get('input_sha256', {})) != 19 or
            len(freeze.get('reused_source_sha256', {})) != 5 or
            freeze.get('design_sha256') != fit.sha256(
                ROOT / freeze.get('design_path', '')) or
            not np.isclose(freeze['original_objective_guard'],
                .99 * freeze['baseline_objective'], rtol=0, atol=1e-15)):
        raise ValueError('E137 fixed restoration contract changed')
    for mapping in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[mapping].items():
            checked_path(relative, digest)
    for name in ('e128_freeze_path', 'fixture_path', 'geometry_path',
                 'start_raw_path', 'baseline_raw_path', 'E136_result_path',
                 'E136_final_review_path'):
        if freeze.get(name) not in freeze['input_sha256']:
            raise ValueError('E137 required input is not frozen: ' + name)
    if fit.sha256(review_path) != review_sha:
        raise ValueError('E137 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_timescale_restoration')
            is not True or review.get('blocking_defects') != [] or
            review.get('freeze_sha256') != freeze_sha or
            set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E137 two-source review required')
    for relative, digest in review['source_sha256'].items():
        checked_path(relative, digest)
    e128 = json.loads((ROOT / freeze['e128_freeze_path']).read_text(
        encoding='utf-8'))
    if (e128.get('fixture_path') != freeze['fixture_path'] or
            e128.get('geometry_path') != freeze['geometry_path'] or
            e128.get('geometry_hash') != freeze['geometry_hash'] or
            [(x['name'], x['current'], x['source']) for x in
                e128.get('contexts', [])] !=
                [(x['name'], x['current'], x['source']) for x in
                    freeze['contexts']]):
        raise ValueError('E137/E128 fixture or context mismatch')
    previous = json.loads((ROOT / freeze['E136_final_review_path']).read_text(
        encoding='utf-8'))
    result = json.loads((ROOT / freeze['E136_result_path']).read_text(
        encoding='utf-8'))
    native = json.loads((ROOT / freeze['E136_native_review_path']).read_text(
        encoding='utf-8'))
    source = json.loads((ROOT / freeze['E136_source_review_path']).read_text(
        encoding='utf-8'))
    if (previous.get('validity') != 'valid' or
            previous.get('hypothesis_verdict') != 'refuted' or
            result.get('status') != 'completed_bounded_fit' or
            result.get('best_any_index') != 45 or
            result.get('selected_index') != 1 or
            not np.isclose(result.get('start_objective', np.nan),
                freeze['baseline_objective'], rtol=0, atol=1e-12) or
            native.get('accepted_as_valid_timescale_constrained_fit')
                is not True or
            source.get('accepted_for_single_original_timescale_constrained_fit')
                is not True or source.get('blocking_defects') != []):
        raise ValueError('E137 predecessor receipts inconsistent')
    return freeze, e128, review


def original_objective(data):
    prior = e136.objective_and_jac(data, 0.)[2]
    return float(np.max(data['losses']) + prior), float(prior)


def start_and_baseline(freeze):
    with (np.load(ROOT / freeze['start_raw_path'], allow_pickle=False) as start,
          np.load(ROOT / freeze['baseline_raw_path'], allow_pickle=False) as baseline):
        vector = np.r_[start['delta'], start['gain'], start['log_tau']]
        start_obj, start_prior = original_objective(start)
        baseline_obj, baseline_prior = original_objective(baseline)
        if (any(start[key].dtype != np.float64 for key in
                ('delta', 'gain', 'log_tau')) or
                vector.shape != (307,) or not np.isfinite(vector).all() or
                not np.isclose(start_obj, freeze['start_objective'],
                    rtol=0, atol=1e-10) or
                not np.isclose(baseline_obj, freeze['baseline_objective'],
                    rtol=0, atol=1e-10)):
            raise ValueError('E137 frozen E13645/E1361 objectives or vector changed')
        start_hash = fit.sha256(ROOT / freeze['start_raw_path'])
        baseline_hash = fit.sha256(ROOT / freeze['baseline_raw_path'])
    return vector, {'start_objective': start_obj,
        'start_prior': start_prior, 'start_raw_sha256': start_hash,
        'baseline_objective': baseline_obj,
        'baseline_prior': baseline_prior,
        'baseline_raw_sha256': baseline_hash}


def compare_start_raw(actual_path, reference_path):
    """Compare the first saved response to E13645, never re-simulate it."""
    report = {'accepted': False, 'fields': {},
        'actual_sha256': fit.sha256(actual_path),
        'reference_sha256': fit.sha256(reference_path)}
    with (np.load(actual_path, allow_pickle=False) as actual,
          np.load(reference_path, allow_pickle=False) as reference):
        for key in (*EXACT_RAW, *RESPONSE_RAW, *METRIC_RAW):
            if key not in actual or key not in reference:
                report['fields'][key] = {'accepted': False, 'reason': 'missing'}
                continue
            a, b = actual[key], reference[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                report['fields'][key] = {'accepted': False,
                    'reason': 'shape_or_dtype', 'actual_shape': a.shape,
                    'reference_shape': b.shape}
                continue
            limit = (0 if key in EXACT_RAW else
                1e-8 if key in RESPONSE_RAW else 1e-10)
            difference = (float(np.max(np.abs(a - b))) if a.size and limit
                          else 0.)
            accepted = bool(np.array_equal(a, b) if limit == 0 else
                np.isfinite(a).all() and np.isfinite(b).all() and
                difference <= limit)
            report['fields'][key] = {'accepted': accepted,
                'max_abs_difference': difference, 'limit': limit}
        report['accepted'] = all(
            item['accepted'] for item in report['fields'].values())
    return report


def restoration_value_jac(data, guard, baseline):
    """The original 4758 startup inequalities plus three fixed loss guards."""
    losses = np.asarray(data['losses'], np.float64)
    _, objective_gradient, prior = e136.objective_and_jac(data, 0.)
    prior_gradient = objective_gradient[:-1]
    c, jac, ratios = fit.constraint_values(losses, data['loss_jac'],
        data['q_error'], data['ltm_error'], data['q_jac'],
        data['ltm_jac'], float(max(losses) + .01))
    if (c.shape != (4761,) or jac.shape != (4761, 308) or
            not np.isclose(guard, .99 * baseline, rtol=0, atol=1e-12)):
        raise ValueError('E137 original startup/guard dimensions changed')
    startup = c[3:]
    guard3 = (guard - losses - prior) / guard
    guard_jac = -(np.asarray(data['loss_jac']) + prior_gradient[None, :]) / guard
    augmented = np.r_[startup, guard3]
    augmented_jac = np.vstack((jac[3:, :-1], guard_jac))
    negative = np.minimum(augmented, 0.)
    value = float(.5 * (negative @ negative))
    gradient = np.asarray(augmented_jac.T @ negative, np.float64)
    original = float(np.max(losses) + prior)
    objective_ratio = original / baseline
    guard_ratios = (losses + prior) / guard
    qualifies = bool(fit.selector_feasible(ratios) and
                     objective_ratio <= .99)
    if (augmented.shape != (4761,) or
            augmented_jac.shape != (4761, 307) or
            not np.isfinite(augmented).all() or
            not np.isfinite(augmented_jac).all() or
            not np.isfinite(gradient).all() or
            not np.isfinite([value, original, objective_ratio]).all()):
        raise ValueError('Nonfinite E137 restoration/guard evidence')
    return {'value': value, 'gradient': gradient,
        'gradient_norm': float(np.linalg.norm(gradient)),
        'startup_negative_count': int(np.count_nonzero(startup < 0)),
        'guard3': guard3, 'guard_ratios': guard_ratios,
        'ratios': ratios, 'prior': prior,
        'original_objective': original,
        'objective_over_baseline': float(objective_ratio),
        'qualifies': qualifies,
        'minimum_startup_residual': float(np.min(startup)),
        'minimum_guard_residual': float(np.min(guard3))}


class BoundedStop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class RestorationCache:
    """Float64 byte-keyed, completed-record-only, one-call restoration cache."""

    def __init__(self, evaluate, start, bounds, on_start, on_complete,
                 *, max_unique, fit_deadline, internal_deadline,
                 clock=time.perf_counter):
        self.evaluate = evaluate
        self.start = np.asarray(start, np.float64)
        self.bounds = np.asarray(bounds, np.float64)
        self.on_start = on_start
        self.on_complete = on_complete
        self.max_unique = max_unique
        self.fit_deadline = fit_deadline
        self.internal_deadline = internal_deadline
        self.clock = clock
        self.cache = {}
        self.started = 0
        self.completed = 0
        self.hits = 0
        self.selected_index = None
        self.terminal_error = None

    def get(self, vector):
        if self.terminal_error is not None:
            raise self.terminal_error
        x = np.asarray(vector, np.float64)
        if (x.shape != (307,) or not np.isfinite(x).all() or
                np.any(x < self.bounds[:, 0] - 1e-12) or
                np.any(x > self.bounds[:, 1] + 1e-12)):
            raise ValueError('E137 finite in-bounds LF307 vector required')
        key = np.ascontiguousarray(x).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.started >= self.max_unique:
            self.terminal_error = BoundedStop('unique_cap')
            raise self.terminal_error
        now = self.clock()
        if (now >= self.fit_deadline - 3 or
                now >= self.internal_deadline - 5):
            self.terminal_error = BoundedStop('pre_call_wall_reserve')
            raise self.terminal_error
        index = self.started + 1
        try:
            self.on_start(index, x)
            self.started = index
            data = self.evaluate(x, index)
            result = (float(data['restoration_value']),
                      np.asarray(data['restoration_gradient'], np.float64))
            self.cache[key] = result
            self.completed += 1
            if data['qualifies'] and index != 1:
                self.selected_index = index
            try:
                self.on_complete(self, data, index)
            except BaseException:
                self.cache.pop(key, None)
                self.completed -= 1
                if self.selected_index == index:
                    self.selected_index = None
                raise
        except BaseException as error:
            self.terminal_error = error
            raise
        if index == 1 and data['qualifies']:
            self.terminal_error = ValueError(
                'E137 frozen infeasible start unexpectedly qualifies')
            raise self.terminal_error
        if data['qualifies']:
            self.terminal_error = BoundedStop(
                'first_guarded_selector_feasible')
            raise self.terminal_error
        return result


def run_optimizer(cache, ledger, persist, *, solver=minimize,
                  clock=time.perf_counter):
    """Explicit E13645 replay, then one bounded L-BFGS-B request."""
    accounting = ledger['optimizer']
    began = clock()
    result = None
    reason = 'scipy_finished'
    try:
        cache.get(cache.start)
        accounting.update(status='intent', intent_persisted=True)
        try:
            persist()
        except BaseException:
            accounting.update(status='not_requested', intent_persisted=False)
            raise
        accounting.update(status='in_flight_unknown', requested=1,
            request_receipt_persisted=True, in_flight=True)
        ledger['fit_calls'] = 1
        try:
            persist()
        except BaseException:
            accounting.update(status='request_receipt_failed', requested=0,
                request_receipt_persisted=False, in_flight=False)
            ledger['fit_calls'] = 0
            raise
        try:
            result = solver(cache.get, cache.start.copy(), method='L-BFGS-B',
                jac=True, bounds=[tuple(row) for row in cache.bounds],
                options=OPTIONS.copy())
        except BoundedStop as error:
            accounting.update(status='bounded_stop', bounded_stop=1,
                              in_flight=False, stop_reason=error.reason)
            reason = error.reason
            persist()
        except BaseException as error:
            accounting.update(status='failed_unknown', failed_unknown=1,
                              in_flight=False, error=repr(error))
            raise
        else:
            accounting.update(status='returned', returned=1, in_flight=False,
                solver={'success': bool(result.success),
                    'status': int(result.status), 'message': str(result.message),
                    'nit': int(result.nit), 'nfev': int(result.nfev),
                    'fun': float(result.fun)})
            persist()
            if (not np.isfinite(result.fun) or
                    np.asarray(result.x).shape != (307,) or
                    not np.isfinite(result.x).all() or
                    (not result.success and int(result.status) != 1)):
                raise ValueError('E137 optimizer returned invalid numerical state')
            if not result.success:
                reason = 'scipy_iteration_or_function_limit'
    finally:
        accounting['elapsed_wall_s'] = clock() - began
        ledger['fit_calls'] = accounting['requested']
        ledger['cache_started'] = cache.started
        ledger['cache_completed'] = cache.completed
        ledger['cache_hits'] = cache.hits
    return result, reason


def result_from_records(freeze, ledger, cache, comparison, stop_reason,
                        baseline_evidence):
    rows = ledger['original_evaluations']
    baseline = baseline_evidence['baseline_objective']
    try:
        complete = bool(1 <= len(rows) <= freeze['max_unique_evaluations'] and
            all(row['index'] == index and e136._complete_row(row) and
                np.isfinite([row['restoration_value'], row['gradient_norm'],
                    row['objective_over_baseline']]).all() and
                len(row['guard3']) == 3 and len(row['guard_ratios']) == 3 and
                np.isfinite(row['guard3']).all() and
                np.isfinite(row['guard_ratios']).all() and
                row['startup_negative_count'] >= 0 and
                row['qualifies'] is bool(row['selector_feasible'] and
                    row['true_worst_objective'] / baseline <= .99) and
                np.isclose(row['objective_over_baseline'],
                    row['true_worst_objective'] / baseline, rtol=0, atol=1e-12)
                for index, row in enumerate(rows, 1)) and
            ledger['original_evaluations_requested'] ==
                ledger['original_evaluations_returned'] ==
                ledger['original_evaluations_completed'] == len(rows) and
            ledger['original_evaluations_failed_unknown'] == 0 and
            ledger['original_evaluations_post_return_error'] == 0 and
            cache.started == cache.completed == len(cache.cache) == len(rows))
    except (KeyError, TypeError, ValueError):
        complete = False
    optimizer = ledger['optimizer']
    optimizer_ok = bool(ledger.get('fit_calls') == 1 and
        optimizer.get('requested') == 1 and
        ((optimizer.get('returned') == 1 and
          optimizer.get('bounded_stop') == 0 and
          optimizer.get('failed_unknown') == 0 and
          optimizer.get('status') == 'returned') or
         (optimizer.get('returned') == 0 and
          optimizer.get('bounded_stop') == 1 and
          optimizer.get('failed_unknown') == 0 and
          optimizer.get('status') == 'bounded_stop')))
    budget_ok = bool(0 <= ledger['fit_wall_s'] <= freeze['fit_wall_limit_s'] and
        ledger['raw_bytes'] <= freeze['maximum_raw_bytes'])
    first_ok = bool(comparison and comparison.get('accepted') is True and
        rows and comparison.get('actual_sha256') == rows[0]['raw_sha256'] and
        rows[0]['selector_feasible'] is False and
        np.isclose(rows[0]['true_worst_objective'],
            freeze['start_objective'], rtol=0, atol=1e-10))
    tiny = ledger.get('synthetic_tiny', {})
    tiny_ok = bool(tiny.get('status') == 'completed' and
        tiny.get('raw_saved') is True and tiny.get('solve_returned') is True)
    valid = complete and optimizer_ok and budget_ok and first_ok and tiny_ok
    candidates = [row for row in rows if row.get('qualifies') is True] if complete else []
    first = candidates[0] if candidates else None
    first_index = first['index'] if first else None
    selection_ok = bool(
        (not candidates and cache.selected_index is None and
         stop_reason != 'first_guarded_selector_feasible') or
        (first is not None and len(candidates) == 1 and
         first_index == len(rows) and
         cache.selected_index == first_index and
         stop_reason == 'first_guarded_selector_feasible'))
    valid = valid and selection_ok
    accepted = bool(valid and first is not None)
    return {'schema': 'flylab.m3.e137.timescale-restoration-result.v1',
        'status': 'completed_bounded_restoration' if valid else 'invalid_evidence',
        'hypothesis_verdict': ('supported' if accepted else
            'refuted' if valid else 'not_tested'),
        'interpretation': 'local_LF_guarded_startup_restoration_only',
        'new_lf_proposal_accepted': accepted,
        'baseline_objective': baseline,
        'start_objective': rows[0]['true_worst_objective'] if complete else None,
        'selected_index': first_index if valid else None,
        'selected_objective': first['true_worst_objective'] if accepted else None,
        'selected_objective_over_baseline': (
            first['true_worst_objective'] / baseline if accepted else None),
        'best_any_index': (min(rows, key=lambda x: x['true_worst_objective'])[
            'index'] if complete else None),
        'first_reference_comparison': first_ok,
        'guard_objective': freeze['original_objective_guard'],
        'stop_reason': stop_reason,
        'original_completion_accounting_complete': complete,
        'optimizer_accounting_complete': optimizer_ok,
        'fit_budget_complete': budget_ok,
        'selection_consistent': selection_ok,
        'synthetic_tiny_complete': tiny_ok,
        'original_evaluations_requested': ledger['original_evaluations_requested'],
        'original_evaluations_returned': ledger['original_evaluations_returned'],
        'original_evaluations_completed': ledger['original_evaluations_completed'],
        'cache_started': cache.started, 'cache_completed': cache.completed,
        'cache_hits': cache.hits, 'fit_calls': ledger['fit_calls'],
        'fit_wall_s': ledger['fit_wall_s'], 'raw_bytes': ledger['raw_bytes'],
        'original_neural_evaluations': ledger['original_evaluations_requested'],
        'body_calls': 0, 'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False, 'whole_goal_complete': False,
        'freeze_sha256': ledger['freeze_sha256'],
        'source_review_sha256': ledger['source_review_sha256']}


def complete_restoration_receipt(ledger, local_cache, index, persist):
    """Persist completed raw/cache metadata or durably mark post-return error."""
    row = ledger['original_evaluations'][index - 1]
    try:
        if not e136._complete_row(row):
            raise ValueError('E137 completed raw receipt invalid')
        ledger.update(cache_started=local_cache.started,
            cache_completed=local_cache.completed,
            cache_hits=local_cache.hits,
            selected_index=local_cache.selected_index)
        persist()
    except BaseException as error:
        row.update(status='failed', error=repr(error),
            traceback=traceback.format_exc(),
            post_return_error=bool(row.get('solve_returned')),
            response_progress=getattr(error, 'response_progress',
                row.get('response_progress')))
        ledger.update(cache_started=local_cache.started,
            cache_completed=local_cache.completed - 1,
            selected_index=None)
        try:
            persist()
        except BaseException as secondary:
            error.add_note('Secondary E137 completion ledger: ' +
                           repr(secondary))
        raise


def write_output_manifest(out):
    save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e137.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): fit.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file() and
            path.name != 'output-manifest.json'}})


def finalize(out, ledger, freeze, began, primary):
    """Preserve original primary evidence and revoke late false success."""
    def note(stage, error):
        nonlocal primary
        ledger.setdefault('finalization_errors', []).append({
            'stage': stage, 'error': repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note('Secondary ' + stage + ': ' + repr(error))

    def keep_primary():
        previous = ledger.get('primary_error')
        evidence = dict(previous) if isinstance(previous, dict) else {}
        evidence.setdefault('type', type(primary).__name__)
        evidence.setdefault('message', str(primary))
        evidence.setdefault('traceback', ''.join(
            traceback.format_exception(primary)))
        evidence.setdefault('response_progress',
                            getattr(primary, 'response_progress', None))
        notes = list(evidence.get('notes') or [])
        for item in getattr(primary, '__notes__', []):
            if item not in notes:
                notes.append(item)
        evidence['notes'] = notes
        ledger['primary_error'] = evidence

    for _ in range(2):
        ledger['process_wall_s'] = time.perf_counter() - began
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('final_ledger', error)
        try:
            write_output_manifest(out)
        except BaseException as error:
            note('final_manifest', error)
        if (freeze is not None and time.perf_counter() - began >
                freeze['internal_wall_limit_s']):
            note('post_manifest_timeout', TimeoutError(
                'E137 internal wall limit exceeded after manifest'))
        if primary is not None:
            break
        ledger['post_manifest_wall_s'] = time.perf_counter() - began
    if primary is not None:
        ledger['status'] = 'failed'
        keep_primary()
        result_path = out / 'result.json'
        if result_path.exists():
            try:
                result_path.unlink()
            except BaseException as error:
                note('result_removal', error)
                try:
                    save(result_path, {'status': 'invalid_finalization',
                        'hypothesis_verdict': 'not_tested',
                        'new_lf_proposal_accepted': False})
                except BaseException as fallback:
                    note('result_invalidation_fallback', fallback)
        keep_primary()
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('failed_ledger', error)
        try:
            write_output_manifest(out)
        except BaseException as error:
            note('failed_manifest', error)
        if ledger.get('finalization_errors'):
            keep_primary()
            try:
                save(out / 'advance-ledger.json', ledger)
            except BaseException as error:
                note('failed_ledger_update', error)
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    """Run the sole reviewed original process, with complete raw before verdict."""
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('One-shot E137 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e137.timescale-restoration-ledger.v1',
        'pid': os.getpid(), 'status': 'preflight',
        'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
        'original_evaluations': [], 'original_evaluations_intended': 0,
        'original_evaluations_requested': 0,
        'original_evaluations_returned': 0,
        'original_evaluations_completed': 0,
        'original_evaluations_failed_unknown': 0,
        'original_evaluations_post_return_error': 0,
        'original_evaluations_unstarted': 0,
        'optimizer': {'status': 'not_requested', 'intent_persisted': False,
            'requested': 0, 'returned': 0, 'bounded_stop': 0,
            'failed_unknown': 0},
        'fit_calls': 0, 'raw_bytes': 0, 'body_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False}
    save(out / 'advance-ledger.json', ledger)
    freeze = limits = engine = cache = None
    primary = comparison = None
    fit_began = None
    try:
        freeze, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, review_path, review_sha)
        save(out / 'source-review-snapshot.json', review)
        start, baseline_evidence = start_and_baseline(freeze)
        ledger['baseline_evidence'] = baseline_evidence
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version,
            'numpy': np.__version__, 'cupy': cp.__version__,
            'torch': torch.__version__,
            'gpu': gpu.decode() if isinstance(gpu, bytes) else str(gpu)}
        e136.run_tiny_setup(out, ledger)
        fixture = fit.frozen_problem(e128)
        geometry = TorchFootGeometry.load(
            ROOT / freeze['geometry_path'], device='cuda')
        torch.cuda.synchronize(geometry.device)
        if geometry.geometry_hash != freeze['geometry_hash']:
            raise ValueError('E137 frozen geometry hash mismatch')
        ledger['geometry_backend'] = type(geometry).__name__
        ledger['geometry_device'] = str(geometry.device)
        ledger['geometry_fixed_upload_bytes'] = sum(
            int(geometry.arrays[key].nbytes) for key in
            ('axes', 'anchors', 'tips', 'reference_angles'))
        motors, muscles, edges = (np.flatnonzero(fixture[key][0]) for key in
            ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask'))
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('E137 LF identity cardinality changed')
        bounds = e136.original_bounds(fixture, edges)
        if (bounds.shape != (307, 2) or
                np.any(start < bounds[:, 0]) or
                np.any(start > bounds[:, 1])):
            raise ValueError('E137 E13645 start outside original bounds')
        try:
            engine = tau_module.CudaAdaptationTimescaleEngine(
                fixture, motors, muscles, edges)
        except BaseException as error:
            ledger['constructor_failure'] = {
                'error': repr(error),
                'fixed_upload_progress': getattr(error,
                    'fixed_upload_progress', None),
                'fixed_upload_by_key': getattr(error,
                    'fixed_upload_by_key', None)}
            try:
                save(out / 'advance-ledger.json', ledger)
            except BaseException as secondary:
                error.add_note('Secondary E137 constructor ledger: ' +
                               repr(secondary))
            raise
        ledger['engine_before'] = e136.snapshot(engine.metadata())
        with np.load(ROOT / freeze['start_raw_path'],
                     allow_pickle=False) as reference:
            for key, expected in (
                    ('motor_indices', engine.problem['motor_indices']),
                    ('muscle_indices', engine.problem['muscle_indices']),
                    ('edge_ids', engine.problem['edge_ids']),
                    ('parameter_muscle_map',
                        tau_module.activation_parameter_map(engine.problem))):
                if not np.array_equal(reference[key], expected):
                    raise ValueError('E137 LF engine identity changed: ' + key)
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        fit_began = time.perf_counter()
        fit_deadline = fit_began + freeze['fit_wall_limit_s']
        internal_deadline = began + freeze['internal_wall_limit_s']

        def persist():
            e136.refresh_counts(ledger)
            save(out / 'advance-ledger.json', ledger)

        def on_start(index, vector):
            e136.register_intent(ledger, index, vector, persist)

        def evaluate(vector, index):
            nonlocal comparison
            row = ledger['original_evaluations'][index - 1]
            directory = out / f'evaluation-{index:03d}'

            def evaluator(x, row_index, path, **callbacks):
                return e136.evaluate_timescale_leg(engine, fixture, geometry,
                    x, row_index, path, **callbacks)

            try:
                record, _ = e136.evaluate_with_receipts(ledger, row, vector,
                    directory, evaluator, persist, metadata=engine.metadata,
                    maximum_raw_bytes=freeze['maximum_raw_bytes'])
                with np.load(record['raw_path'], allow_pickle=False) as raw:
                    restoration = restoration_value_jac(raw,
                        freeze['original_objective_guard'],
                        baseline_evidence['baseline_objective'])
                if (not np.allclose(restoration['ratios'], record['ratios'],
                        rtol=0, atol=1e-12) or
                        not np.isclose(restoration['original_objective'],
                            record['true_worst_objective'], rtol=0, atol=1e-10)):
                    raise ValueError('E137 saved restoration metric mismatch')
                row.update(restoration_value=restoration['value'],
                    gradient_norm=restoration['gradient_norm'],
                    startup_negative_count=restoration['startup_negative_count'],
                    guard3=restoration['guard3'].tolist(),
                    guard_ratios=restoration['guard_ratios'].tolist(),
                    minimum_startup_residual=(
                        restoration['minimum_startup_residual']),
                    minimum_guard_residual=(
                        restoration['minimum_guard_residual']),
                    objective_over_baseline=(
                        restoration['objective_over_baseline']),
                    qualifies=restoration['qualifies'])
                persist()
                if index == 1:
                    comparison = compare_start_raw(record['raw_path'],
                        ROOT / freeze['start_raw_path'])
                    save(out / 'first-reference-comparison.json', comparison)
                    ledger['first_reference_comparison'] = comparison['accepted']
                    persist()
                    if not comparison['accepted']:
                        raise ValueError('E137 E13645 first replay parity failed')
                    if row['selector_feasible']:
                        raise ValueError('E137 E13645 unexpectedly selector feasible')
                record.update(restoration_value=restoration['value'],
                    restoration_gradient=restoration['gradient'],
                    qualifies=restoration['qualifies'])
                return record
            except BaseException as error:
                row.update(status='failed', error=repr(error),
                    traceback=traceback.format_exc(),
                    post_return_error=bool(row.get('solve_returned')),
                    response_progress=getattr(error, 'response_progress',
                        row.get('response_progress')))
                try:
                    persist()
                except BaseException as secondary:
                    error.add_note('Secondary E137 evaluation ledger: ' +
                                   repr(secondary))
                raise

        def on_complete(local_cache, data, index):
            complete_restoration_receipt(ledger, local_cache, index, persist)

        cache = RestorationCache(evaluate, start, bounds, on_start,
            on_complete, max_unique=freeze['max_unique_evaluations'],
            fit_deadline=fit_deadline, internal_deadline=internal_deadline)
        try:
            solver, stop_reason = run_optimizer(cache, ledger, persist)
        finally:
            active = sys.exc_info()[1]
            ledger['fit_wall_s'] = time.perf_counter() - fit_began
            e136.refresh_counts(ledger)
            try:
                persist()
            except BaseException as secondary:
                if active is not None:
                    active.add_note('Secondary E137 fit ledger: ' +
                                    repr(secondary))
                else:
                    raise
        ledger['solver_returned'] = solver is not None
        ledger['stop_reason'] = stop_reason
        if ledger['fit_wall_s'] > freeze['fit_wall_limit_s']:
            raise TimeoutError('E137 fit wall limit exceeded')
        if time.perf_counter() >= internal_deadline - 1:
            raise TimeoutError('E137 result reserve exhausted')
        result = result_from_records(freeze, ledger, cache, comparison,
            stop_reason, baseline_evidence)
        if result['status'] != 'completed_bounded_restoration':
            raise ValueError('E137 original evidence incomplete')
        if result['new_lf_proposal_accepted']:
            selected = ledger['original_evaluations'][
                result['selected_index'] - 1]
            with np.load(selected['raw_path'], allow_pickle=False) as raw:
                np.savez(out / 'selected-lf-proposal.npz',
                    delta=raw['delta'], gain=raw['gain'],
                    log_tau=raw['log_tau'],
                    original_evaluation_index=np.array(result['selected_index']),
                    source_raw_sha256=np.array(selected['raw_sha256']))
            ledger['selected_proposal_sha256'] = fit.sha256(
                out / 'selected-lf-proposal.npz')
        save(out / 'result.json', result)
        ledger['status'] = result['status']
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'response_progress': getattr(error, 'response_progress', None),
            'notes': list(getattr(error, '__notes__', []))}
    finally:
        e136.refresh_counts(ledger)
        if engine is not None:
            try:
                ledger['engine_after'] = e136.snapshot(engine.metadata())
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E137 engine metadata: ' +
                                     repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E137 threadpool restore: ' +
                                     repr(error))
        primary = finalize(out, ledger, freeze, began, primary)
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
