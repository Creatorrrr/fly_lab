"""E140 sparse residual trust-region restoration for the remaining three legs."""

# ruff: noqa: BLE001, B023  # preserve primary/secondary evidence; sync loop callbacks

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_adaptation_timescale as tau_module
from tools import fit_c_banc_m3_cross_leg_timescale_restoration as e139
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import fit_c_banc_m3_timescale_constrained as e136

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '6e7cd764b00a5fb2418fcb56334ba0009a8ac5ed8d2b0d70a182a423980bccaf'
SOURCE_FILES = ('tools/fit_c_banc_m3_residual_trust_restoration.py',
                'tests/test_c_banc_residual_trust_restoration.py')
OPTIONS = {'method': 'trf', 'tr_solver': 'lsmr', 'x_scale': 'jac',
    'loss': 'linear', 'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12,
    'max_nfev': 18, 'verbose': 0,
    'tr_options': {'regularize': True, 'atol': 1e-12, 'btol': 1e-12,
                   'conlim': 1e12, 'maxiter': 300}}
LEG_DIMENSIONS = ((2, 63, 16, 240, 272, 25, 18),
                  (4, 63, 15, 255, 285, 47, 11),
                  (5, 63, 16, 285, 317, 56, 9))


class CompactWriter:
    """Exactly one jsonable conversion per compact durable write."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.attempts = 0
        self.jsonable_calls = 0
        self.conversions_completed = 0
        self.calls = 0
        self.failures = 0
        self.wall_s = 0.
        self.bytes = 0

    def save(self, path, value):
        began = self.clock()
        self.attempts += 1
        self.jsonable_calls += 1
        try:
            converted = fit.jsonable(value)
            self.conversions_completed += 1
            text = json.dumps(converted, ensure_ascii=False, sort_keys=True,
                              allow_nan=False, separators=(',', ':')) + '\n'
            path = Path(path)
            with path.open('w', encoding='utf-8', newline='') as stream:
                stream.write(text)
            size = path.stat().st_size
        except BaseException:
            self.failures += 1
            raise
        else:
            self.calls += 1
            self.bytes += size
        finally:
            self.wall_s += self.clock() - began

    def metadata(self):
        return {'attempts': self.attempts, 'calls': self.calls,
                'failures': self.failures, 'wall_s': self.wall_s,
                'bytes': self.bytes, 'jsonable_calls': self.jsonable_calls,
                'conversions_completed': self.conversions_completed,
                'format': 'compact_sorted_utf8'}


def checked_path(relative, digest):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if (name.is_absolute() or '..' in name.parts or
            not path.is_relative_to(ROOT.resolve()) or
            not isinstance(digest, str) or len(digest) != 64 or
            fit.sha256(path) != digest):
        raise ValueError('E140 frozen path/SHA mismatch: ' + str(relative))
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or fit.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E140 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    fixed = {'schema': 'flylab.m3.e140.residual-trust-freeze.v1',
        'owned_files': list(SOURCE_FILES), 'leg_order': [2, 4, 5],
        'original_processes': 1, 'retries': 0,
        'original_optimizer_calls_max': 3, 'scipy_version': '1.18.1',
        'solver': 'least_squares', 'solver_options': OPTIONS,
        'startup_rows': 4758, 'guard_rows': 3, 'residual_rows': 4761,
        'active_jacobian_rule': 'c_aug < 0; zero rows at c_aug == 0',
        'proposal_objective_ratio_limit': .99,
        'optimizer_cap': fit.OPTIMIZER_CAP,
        'selector_cap': fit.SELECTOR_CAP,
        'max_unique_per_leg': 18, 'max_unique_total': 54,
        'start_replay_per_leg': 1, 'leg_wall_limit_s': 55,
        'leg_reserve_s': 3, 'fit_block_wall_limit_s': 165,
        'internal_wall_limit_s': 180, 'process_reserve_s': 10,
        'external_wall_limit_s': 190, 'supervisor_limit_s': 210,
        'maximum_raw_bytes': 4294967296,
        'minimum_free_disk_bytes': 12884901888,
        'cpu_threads': 1, 'gpu_count': 1, 'full391_batches': 0,
        'full_BANC_calls': 0, 'body_calls': 0, 'LP_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'active_UI_changed': False}
    if (any(freeze.get(key) != value for key, value in fixed.items()) or
            scipy.__version__ != freeze['scipy_version'] or
            freeze.get('log_tau_bounds') != [-math.log(3), math.log(10)] or
            freeze.get('tiny') != {'batches': 1, 'context_ticks': 20,
                'conditional_s': .005, 'derivative_banks': 1} or
            freeze.get('design_sha256') != fit.sha256(
                ROOT / freeze.get('design_path', '')) or
            len(freeze.get('input_sha256', {})) != 71 or
            len(freeze.get('reused_source_sha256', {})) != 54 or
            len(freeze.get('preservation_sha256', {})) != 1536 or
            freeze.get('prepared_input_sha256') !=
                freeze['input_sha256'].get(freeze.get('prepared_input_path'))):
        raise ValueError('E140 fixed freeze contract changed')
    for spec, expected in zip(freeze['legs'], LEG_DIMENSIONS, strict=True):
        fields = ('leg', 'neurons', 'muscles', 'edges', 'parameters',
                  'start_global_index', 'start_local_index')
        if (tuple(spec.get(key) for key in fields) != expected or
                spec.get('max_unique_evaluations') != 18 or
                spec.get('work_per_evaluation') != e139.work_per_evaluation(
                    spec['neurons'], spec['edges']) or
                not np.isclose(spec['original_objective_guard'],
                    .99 * spec['baseline_objective'], rtol=0, atol=1e-15) or
                freeze['input_sha256'].get(spec['start_raw_path']) !=
                    spec['start_raw_sha256'] or
                freeze['input_sha256'].get(spec['baseline_raw_path']) !=
                    spec['baseline_raw_sha256']):
            raise ValueError('E140 frozen leg/start/work changed')
        with np.load(ROOT / spec['baseline_raw_path'],
                     allow_pickle=False) as baseline:
            prior, _ = e139.prior_and_gradient(
                baseline, spec['edges'], spec['muscles'])
            objective = float(np.max(baseline['losses']) + prior)
        if not np.isclose(objective, spec['baseline_objective'],
                          rtol=0, atol=1e-12):
            raise ValueError('E140 frozen E128 baseline denominator changed')
    expected_max = {key: sum(18 * spec['work_per_evaluation'][key]
        for spec in freeze['legs']) for key in e139.WORK_KEYS}
    if freeze.get('max_original_work') != expected_max:
        raise ValueError('E140 maximum original work changed')
    for mapping in ('input_sha256', 'reused_source_sha256',
                    'preservation_sha256'):
        for relative, digest in freeze[mapping].items():
            checked_path(relative, digest)
    if fit.sha256(review_path) != review_sha:
        raise ValueError('E140 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_residual_trust') is not True or
            review.get('blocking_defects') != [] or
            review.get('freeze_sha256') != FREEZE_SHA or
            set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E140 two-source review required')
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
        raise ValueError('E140/E128 frozen fixture/context mismatch')
    return freeze, e128, review


def augmented_residual_jac(data, guard, baseline, edges, muscles):
    """Original c_aug/J_aug, negative residual, and strict-active CSR."""
    losses = np.asarray(data['losses'], np.float64)
    prior, prior_gradient = e139.prior_and_gradient(data, edges, muscles)
    c, jac, ratios = fit.constraint_values(losses, data['loss_jac'],
        data['q_error'], data['ltm_error'], data['q_jac'],
        data['ltm_jac'], float(max(losses) + .01))
    parameters = edges + 2 * muscles
    guard3 = (guard - losses - prior) / guard
    guard_jac = -(np.asarray(data['loss_jac']) +
                  prior_gradient[None, :]) / guard
    c_aug = np.r_[c[3:], guard3]
    j_aug = np.vstack((jac[3:, :-1], guard_jac))
    if (c_aug.shape != (4761,) or j_aug.shape != (4761, parameters) or
            not np.isclose(guard, .99 * baseline, rtol=0, atol=1e-12) or
            not np.isfinite(c_aug).all() or not np.isfinite(j_aug).all()):
        raise ValueError('E140 augmented residual/Jacobian contract changed')
    active = c_aug < 0
    residual = np.minimum(c_aug, 0.)
    active_jac = csr_matrix(j_aug * active[:, None])
    gradient = np.asarray(active_jac.T @ residual).reshape(-1)
    value = float(.5 * residual @ residual)
    objective = float(max(losses) + prior)
    reference = e139.restoration_value_jac(data, guard, baseline,
                                           edges, muscles)
    # Both gradients are the same real-valued dot products.  Dense BLAS sums
    # all 4,761 rows while CSR sums only active rows, so their float64 order
    # can differ.  Bound the difference by the standard dot-product forward
    # error gamma(2n) * sum(abs(a_i*b_i)) for each accumulation.
    unit_roundoff = np.finfo(np.float64).eps / 2
    dense_terms = 2 * len(residual)
    sparse_terms = 2 * int(active.sum())
    dense_gamma = (dense_terms * unit_roundoff /
                   (1 - dense_terms * unit_roundoff))
    sparse_gamma = (sparse_terms * unit_roundoff /
                    (1 - sparse_terms * unit_roundoff))
    absolute_products = np.asarray(
        np.abs(active_jac).T @ np.abs(residual)).reshape(-1)
    gradient_error_bound = ((dense_gamma + sparse_gamma) *
                            absolute_products)
    if (not np.isclose(value, reference['value'], rtol=0, atol=1e-15) or
            np.any(np.abs(gradient - reference['gradient']) >
                   gradient_error_bound) or
            not np.array_equal(ratios, reference['ratios'])):
        raise ValueError('E140 residual representation changed E139 math')
    return {'c_aug': c_aug, 'residual': residual, 'jacobian': active_jac,
        'gradient': gradient, 'value': value,
        'gradient_norm': float(np.linalg.norm(gradient)),
        'active_count': int(active.sum()),
        'startup_negative_count': int(np.count_nonzero(c[3:] < 0)),
        'guard3': guard3, 'guard_ratios': (losses + prior) / guard,
        'ratios': ratios, 'prior': prior,
        'original_objective': objective,
        'objective_over_baseline': objective / baseline,
        'qualifies': bool(fit.selector_feasible(ratios) and objective <= guard),
        'minimum_startup_residual': float(np.min(c[3:])),
        'minimum_guard_residual': float(np.min(guard3))}


class BoundedStop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class ResidualCache:
    """Completed payload cache with defensive callback copies and terminal latch."""

    def __init__(self, evaluate, start, bounds, on_start, on_complete, *,
                 max_unique, leg_deadline, block_deadline, internal_deadline,
                 clock=time.perf_counter):
        self.evaluate = evaluate
        self.start = np.asarray(start, np.float64)
        self.bounds = np.asarray(bounds, np.float64)
        self.on_start = on_start
        self.on_complete = on_complete
        self.max_unique = max_unique
        self.leg_deadline = leg_deadline
        self.block_deadline = block_deadline
        self.internal_deadline = internal_deadline
        self.clock = clock
        self.cache = {}
        self.started = self.completed = self.hits = 0
        self.fun_calls = self.jac_calls = 0
        self.selected_index = None
        self.terminal_error = None
        self.optimizer_invoked = False

    def payload(self, vector):
        if self.terminal_error is not None:
            raise self.terminal_error
        x = np.asarray(vector, np.float64)
        if (x.shape != (len(self.bounds),) or not np.isfinite(x).all() or
                np.any(x < self.bounds[:, 0]) or
                np.any(x > self.bounds[:, 1])):
            self.terminal_error = ValueError('E140 finite in-bounds vector required')
            raise self.terminal_error
        key = np.ascontiguousarray(x).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.started >= self.max_unique:
            self.terminal_error = BoundedStop('leg_unique_cap')
            raise self.terminal_error
        now = self.clock()
        if (now >= self.leg_deadline - 3 or now >= self.block_deadline or
                now >= self.internal_deadline - 10):
            self.terminal_error = BoundedStop('pre_call_wall_reserve')
            raise self.terminal_error
        index = self.started + 1
        try:
            self.on_start(index, x)
            self.started = index
            data = self.evaluate(x, index)
            stored = {'residual': np.array(data['residual'], copy=True),
                'jacobian': data['jacobian'].copy(),
                'qualifies': bool(data['qualifies'])}
            self.cache[key] = stored
            self.completed += 1
            if stored['qualifies'] and index >= 2:
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
        if index == 1 and stored['qualifies']:
            self.terminal_error = ValueError(
                'E140 frozen start unexpectedly jointly qualifies')
            raise self.terminal_error
        if stored['qualifies']:
            self.terminal_error = BoundedStop('first_guarded_selector_feasible')
            raise self.terminal_error
        return stored

    def fun(self, vector):
        self.fun_calls += 1
        return self.payload(vector)['residual'].copy()

    def jac(self, vector):
        self.jac_calls += 1
        return self.payload(vector)['jacobian'].copy()


def run_optimizer(cache, leg_ledger, persist, *, solver=least_squares,
                  clock=time.perf_counter):
    accounting = leg_ledger['optimizer']
    if cache.optimizer_invoked:
        if cache.terminal_error is None:
            cache.terminal_error = RuntimeError(
                'E140 optimizer invocation already in progress')
        raise cache.terminal_error
    cache.optimizer_invoked = True
    began = clock()
    result = None
    reason = 'scipy_finished'
    try:
        cache.payload(cache.start)  # exact replay before solver request
        accounting.update(status='intent', intent_persisted=True)
        try:
            persist()
        except BaseException:
            accounting.update(status='not_requested', intent_persisted=False)
            raise
        accounting.update(status='in_flight_unknown', requested=1,
            request_receipt_persisted=True, in_flight=True)
        try:
            persist()
        except BaseException:
            accounting.update(status='request_receipt_failed', requested=0,
                request_receipt_persisted=False, in_flight=False)
            raise
        try:
            result = solver(cache.fun, cache.start.copy(), jac=cache.jac,
                bounds=(cache.bounds[:, 0], cache.bounds[:, 1]), **OPTIONS)
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
                    'nfev': int(result.nfev), 'njev': (
                        None if result.njev is None else int(result.njev)),
                    'cost': float(result.cost),
                    'optimality': float(result.optimality)})
            persist()
            if (np.asarray(result.x).shape != cache.start.shape or
                    not np.isfinite(result.x).all() or
                    not np.isfinite([result.cost, result.optimality]).all() or
                    int(result.status) < 0):
                raise ValueError('E140 least_squares returned invalid state')
            if int(result.status) == 0:
                reason = 'scipy_max_nfev'
    except BaseException as error:
        cache.terminal_error = error
        raise
    else:
        if cache.terminal_error is None:
            cache.terminal_error = BoundedStop('optimizer_terminal_' + reason)
    finally:
        accounting['elapsed_wall_s'] = clock() - began
        leg_ledger.update(cache_started=cache.started,
            cache_completed=cache.completed, cache_hits=cache.hits,
            fun_calls=cache.fun_calls, jac_calls=cache.jac_calls)
    return result, reason


def save_residual_raw(path, payload):
    jac = payload['jacobian']
    np.savez(path, c_aug=payload['c_aug'], residual=payload['residual'],
        csr_data=jac.data, csr_indices=jac.indices,
        csr_indptr=jac.indptr, csr_shape=np.array(jac.shape, np.int64),
        gradient=payload['gradient'], value=np.array(payload['value']),
        active_count=np.array(payload['active_count']))
    return fit.sha256(path), path.stat().st_size


def account_residual_raw(ledger, row, path, payload, maximum_raw_bytes):
    digest, size = save_residual_raw(path, payload)
    ledger['raw_bytes'] += size
    row.update(residual_raw_saved=True, residual_raw_path=str(path),
        residual_raw_sha256=digest, residual_raw_bytes=size)
    if ledger['raw_bytes'] > maximum_raw_bytes:
        raise ValueError('E140 frozen dual-raw cap exceeded')
    return digest, size


def prepared_inputs(path, fixture, freeze):
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    for key in ('motor_ids', 'membership', 'global_edge_ids'):
        expected = fixture['edge_ids' if key == 'global_edge_ids' else key]
        if not np.array_equal(values[key], expected):
            raise ValueError('E140 prepared global identity changed: ' + key)
    for spec in freeze['legs']:
        prefix = f"leg{spec['leg']}_"
        motors = np.flatnonzero(fixture['leg_motor_mask'][spec['leg']])
        muscles = np.flatnonzero(fixture['leg_muscle_mask'][spec['leg']])
        positions = np.flatnonzero(fixture['leg_edge_mask'][spec['leg']])
        expected = {'motor_indices': motors, 'muscle_indices': muscles,
            'edge_positions': positions,
            'edge_ids': fixture['edge_ids'][positions]}
        for name, answer in expected.items():
            if not np.array_equal(values[prefix + name], answer):
                raise ValueError('E140 prepared leg identity changed: ' + name)
        for name, length in (('delta', spec['edges']),
                             ('gain', spec['muscles']),
                             ('log_tau', spec['muscles'])):
            array = values[prefix + name]
            if (array.shape != (length,) or array.dtype != np.float64 or
                    not np.isfinite(array).all()):
                raise ValueError('E140 prepared start changed: ' + name)
        bounds = e139.original_bounds(fixture, positions, muscles)
        if not np.array_equal(values[prefix + 'bounds'], bounds):
            raise ValueError('E140 prepared bounds changed')
        mapping = values[prefix + 'parameter_muscle_map']
        if (mapping.shape != (spec['parameters'],) or
                mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or
                np.any(mapping >= spec['muscles'])):
            raise ValueError('E140 prepared parameter map changed')
        for name, frozen in (('baseline_objective', 'baseline_objective'),
                             ('start_objective', 'start_objective'),
                             ('objective_guard', 'original_objective_guard')):
            if (values[prefix + name].shape != () or
                    values[prefix + name].dtype != np.float64 or
                    float(values[prefix + name]) != spec[frozen]):
                raise ValueError('E140 prepared objective changed')
    for retained in freeze['retained_legs']:
        prefix = f"retained_leg{retained['leg']}_"
        leg = retained['leg']
        positions = np.flatnonzero(fixture['leg_edge_mask'][leg])
        muscles = np.flatnonzero(fixture['leg_muscle_mask'][leg])
        motors = np.flatnonzero(fixture['leg_motor_mask'][leg])
        for name, answer in (('edge_positions', positions),
                ('edge_ids', fixture['edge_ids'][positions]),
                ('muscle_indices', muscles), ('motor_indices', motors)):
            if not np.array_equal(values[prefix + name], answer):
                raise ValueError('E140 retained identity changed')
        for name, length in (('delta', len(positions)),
                             ('gain', len(muscles)),
                             ('log_tau', len(muscles))):
            if (values[prefix + name].shape != (length,) or
                    values[prefix + name].dtype != np.float64 or
                    not np.isfinite(values[prefix + name]).all()):
                raise ValueError('E140 retained vector changed')
    return values


def compare_start(actual_path, reference_path):
    exact = ('delta', 'gain', 'log_tau', 'edge_ids', 'motor_indices',
             'muscle_indices', 'parameter_muscle_map', 'branch_bits')
    response = (*tau_module.FIELDS, 'B', 'B_initial99')
    metric = ('losses', 'parts', 'cotangent', 'q_error', 'ltm_error',
        'q_coefficient', 'ltm_coefficient', 'q_jac', 'ltm_jac',
        'loss_jac', 'ratios', 'context0_fit_feet', 'context1_fit_feet',
        'context2_fit_feet')
    report = {'accepted': False, 'actual_sha256': fit.sha256(actual_path),
        'reference_sha256': fit.sha256(reference_path), 'fields': {}}
    with (np.load(actual_path, allow_pickle=False) as actual,
          np.load(reference_path, allow_pickle=False) as reference):
        for key in (*exact, *response, *metric):
            if key not in actual or key not in reference:
                report['fields'][key] = {'accepted': False, 'reason': 'missing'}
                continue
            a, b = actual[key], reference[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                report['fields'][key] = {'accepted': False,
                    'reason': 'shape_or_dtype', 'actual_shape': a.shape,
                    'reference_shape': b.shape}
                continue
            limit = 0 if key in exact else 1e-8 if key in response else 1e-10
            difference = (float(np.max(np.abs(a - b))) if a.size and limit
                          else 0.)
            accepted = bool(np.array_equal(a, b) if not limit else
                np.isfinite(a).all() and np.isfinite(b).all() and
                difference <= limit)
            report['fields'][key] = {'accepted': accepted,
                'max_abs_difference': difference, 'limit': limit}
        report['accepted'] = all(
            item['accepted'] for item in report['fields'].values())
    return report


def register_intent(ledger, spec, local_index, vector, persist):
    index = len(ledger['original_evaluations']) + 1
    if index > 54:
        raise BoundedStop('global_unique_cap')
    row = {'index': index, 'leg': spec['leg'], 'local_index': local_index,
        'phase': 'start_replay' if local_index == 1 else 'trust_candidate',
        'status': 'in_flight_unknown', 'backend': 'timescale_cuda',
        'raw_index': 1, 'intent_persisted': True,
        'solve_requested': False, 'request_receipt_persisted': False,
        'in_flight': False, 'solve_returned': False, 'raw_saved': False,
        'residual_raw_saved': False, 'post_return_error': False,
        'neurons': spec['neurons'], 'muscles': spec['muscles'],
        'edges': spec['edges'],
        'work_per_evaluation': spec['work_per_evaluation'],
        'vector_sha256': hashlib.sha256(
            np.asarray(vector, np.float64).tobytes()).hexdigest()}
    ledger['original_evaluations'].append(row)
    try:
        persist()
    except BaseException:
        row.update(status='intent_persistence_failed', intent_persisted=False)
        raise
    return row


def complete_receipt(ledger, leg_ledger, cache, row, persist):
    try:
        if (not e136._complete_row(row) or
                row.get('residual_raw_saved') is not True or
                row.get('residual_raw_sha256') !=
                    fit.sha256(row['residual_raw_path'])):
            raise ValueError('E140 complete dual-raw receipt invalid')
        leg_ledger.update(cache_started=cache.started,
            cache_completed=cache.completed, cache_hits=cache.hits,
            selected_cache_index=cache.selected_index,
            fun_calls=cache.fun_calls, jac_calls=cache.jac_calls)
        persist()
    except BaseException as error:
        row.update(status='failed', error=repr(error),
            traceback=traceback.format_exc(),
            post_return_error=bool(row.get('solve_returned')),
            response_progress=getattr(error, 'response_progress',
                row.get('response_progress')))
        leg_ledger.update(cache_completed=cache.completed - 1,
                          selected_cache_index=None)
        try:
            persist()
        except BaseException as secondary:
            error.add_note('Secondary E140 completion ledger: ' +
                           repr(secondary))
        raise


def run_tiny_setup(out, ledger, writer):
    tiny = fit.synthetic_tiny_setup_problem()
    row = {'status': 'intent', 'synthetic_only': True,
        'requested_context_ticks': 20, 'requested_conditional_s': .005,
        'requested_derivative_banks': 1, 'solve_requested': False,
        'request_receipt_persisted': False, 'in_flight': False,
        'solve_returned': False, 'raw_saved': False}
    ledger['synthetic_tiny'] = row
    writer.save(out / 'advance-ledger.json', ledger)
    engine = None
    try:
        engine = tau_module.CudaAdaptationTimescaleEngine(tiny)
        row['engine_before'] = e136.snapshot(engine.metadata())
        row.update(status='solve_in_flight_unknown', solve_requested=True,
                   request_receipt_persisted=True, in_flight=True)
        try:
            writer.save(out / 'advance-ledger.json', ledger)
        except BaseException:
            row.update(status='request_receipt_failed', solve_requested=False,
                       request_receipt_persisted=False, in_flight=False)
            raise
        d, h, timing = engine.solve(np.zeros(1), np.zeros(1), np.zeros(1),
                                    derivative=True, host=True)
        row.update(solve_returned=True, in_flight=False,
            response_progress=timing['progress'], solve_wall_s=timing['wall_s'])
        writer.save(out / 'advance-ledger.json', ledger)
        h['B'] = engine.host_copy(d['B'], label='tiny_full_B')
        path = out / 'cuda-tiny-setup-raw.npz'
        np.savez(path, synthetic_stage_current=tiny['stage_current'],
                 synthetic_source_rate=tiny['source_rate'], **h)
        row.update(raw_saved=True, raw_path=str(path),
            raw_sha256=fit.sha256(path),
            engine_after=e136.snapshot(engine.metadata()))
        writer.save(out / 'advance-ledger.json', ledger)
        if (timing['context_rk4_ticks'] != 20 or
                not timing['progress']['derivative_bank_complete'] or
                any(not np.isfinite(h[key]).all() for key in
                    (*tau_module.FIELDS, 'B'))):
            raise ValueError('E140 synthetic tiny CUDA setup incomplete')
        row['status'] = 'completed'
        writer.save(out / 'advance-ledger.json', ledger)
    except BaseException as error:
        row.update(status='failed', error=repr(error), in_flight=False,
            response_progress=getattr(error, 'response_progress',
                                      row.get('response_progress')))
        if engine is not None:
            try:
                row['engine_after'] = e136.snapshot(engine.metadata())
            except BaseException as secondary:
                row['engine_after_error'] = repr(secondary)
                error.add_note('Secondary E140 tiny engine metadata: ' +
                               repr(secondary))
        try:
            writer.save(out / 'advance-ledger.json', ledger)
        except BaseException as secondary:
            error.add_note('Secondary E140 tiny ledger: ' + repr(secondary))
        raise


def leg_result(spec, rows, cache, leg_ledger):
    baseline = spec['baseline_objective']
    try:
        complete = bool(1 <= len(rows) <= 18 and
            all(row['leg'] == spec['leg'] and row['local_index'] == index and
                e136._complete_row(row) and
                row['residual_raw_saved'] is True and
                row['residual_raw_sha256'] ==
                    fit.sha256(row['residual_raw_path']) and
                np.isfinite([row['residual_value'], row['gradient_norm'],
                    row['objective_over_baseline']]).all() and
                row['qualifies'] is bool(row['selector_feasible'] and
                    row['true_worst_objective'] <=
                        spec['original_objective_guard'])
                for index, row in enumerate(rows, 1)) and
            cache.started == cache.completed == len(cache.cache) == len(rows) and
            all(row['solve_requested'] is True and
                row['solve_returned'] is True and
                row['post_return_error'] is False for row in rows))
    except (KeyError, TypeError, ValueError, OSError):
        complete = False
    optimizer = leg_ledger['optimizer']
    optimizer_ok = bool(optimizer.get('requested') == 1 and
        optimizer.get('failed_unknown') == 0 and
        ((optimizer.get('returned') == 1 and
          optimizer.get('bounded_stop') == 0 and
          optimizer.get('status') == 'returned') or
         (optimizer.get('returned') == 0 and
          optimizer.get('bounded_stop') == 1 and
          optimizer.get('status') == 'bounded_stop')))
    comparison = leg_ledger.get('comparison', {})
    first_ok = bool(rows and comparison.get('accepted') is True and
        comparison.get('actual_sha256') == rows[0]['raw_sha256'] and
        np.isclose(rows[0]['true_worst_objective'], spec['start_objective'],
                   rtol=0, atol=1e-10) and rows[0]['qualifies'] is False)
    candidates = [row for row in rows[1:] if row.get('qualifies') is True]
    selected = candidates[0] if candidates else None
    stop = leg_ledger.get('stop_reason')
    selection_ok = bool(
        (not candidates and cache.selected_index is None and
         stop != 'first_guarded_selector_feasible') or
        (len(candidates) == 1 and selected is rows[-1] and
         cache.selected_index == selected['local_index'] and
         stop == 'first_guarded_selector_feasible'))
    valid = bool(complete and optimizer_ok and first_ok and selection_ok and
                 0 <= leg_ledger['wall_s'] <= 55)
    accepted = bool(valid and selected is not None)
    return {'leg': spec['leg'], 'name': spec['name'],
        'status': 'completed_bounded_residual_trust' if valid else
                  'invalid_evidence',
        'hypothesis_verdict': ('supported' if accepted else
            'refuted' if valid else 'not_tested'),
        'baseline_objective': baseline,
        'start_objective': rows[0]['true_worst_objective'] if complete else None,
        'selected_index': selected['index'] if accepted else None,
        'selected_local_index': selected['local_index'] if accepted else None,
        'selected_objective': selected['true_worst_objective'] if accepted else None,
        'selected_objective_over_baseline': (
            selected['true_worst_objective'] / baseline if accepted else None),
        'new_leg_proposal_accepted': accepted,
        'original_evaluations': len(rows), 'cache_started': cache.started,
        'cache_completed': cache.completed, 'cache_hits': cache.hits,
        'fun_calls': cache.fun_calls, 'jac_calls': cache.jac_calls,
        'fit_calls': optimizer.get('requested', 0), 'wall_s': leg_ledger['wall_s'],
        'stop_reason': stop, 'raw_complete': complete,
        'optimizer_accounting_complete': optimizer_ok,
        'first_replay_match': first_ok, 'selection_consistent': selection_ok}


def write_package(out, fixture, prepared, ledger, leg_results):
    if not all(item['new_leg_proposal_accepted'] for item in leg_results):
        raise ValueError('E140 incomplete six-leg package')
    delta = np.empty(1572, np.float64)
    gain = np.empty(96, np.float64)
    log_tau = np.empty(96, np.float64)
    edges_filled = np.zeros(1572, bool)
    muscles_filled = np.zeros(96, bool)

    def place(prefix, values):
        positions = prepared[prefix + 'edge_positions']
        muscles = prepared[prefix + 'muscle_indices']
        if np.any(edges_filled[positions]) or np.any(muscles_filled[muscles]):
            raise ValueError('E140 overlapping parameter positions')
        delta[positions] = values['delta']
        gain[muscles] = values['gain']
        log_tau[muscles] = values['log_tau']
        edges_filled[positions] = True
        muscles_filled[muscles] = True

    for retained in (0, 1, 3):
        prefix = f'retained_leg{retained}_'
        place(prefix, {key: prepared[prefix + key]
                       for key in ('delta', 'gain', 'log_tau')})
    for result in leg_results:
        prefix = f"leg{result['leg']}_"
        row = ledger['original_evaluations'][result['selected_index'] - 1]
        with np.load(row['raw_path'], allow_pickle=False) as raw:
            if (not np.array_equal(raw['edge_ids'], prepared[prefix + 'edge_ids'])
                    or not np.array_equal(raw['muscle_indices'],
                                          prepared[prefix + 'muscle_indices'])):
                raise ValueError('E140 selected identity changed')
            place(prefix, raw)
    if not edges_filled.all() or not muscles_filled.all():
        raise ValueError('E140 package coverage incomplete')
    path = out / 'conditional-six-leg-parameter-package.npz'
    np.savez(path, delta=delta, gain=gain, log_tau=log_tau,
        global_edge_ids=fixture['edge_ids'], motor_ids=fixture['motor_ids'],
        membership=fixture['membership'], candidate_adopted=np.array(False))
    return {'path': str(path), 'sha256': fit.sha256(path),
        'edges': 1572, 'muscles': 96, 'motors': 391,
        'candidate_adopted': False}


def write_output_manifest(out, writer):
    writer.save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e140.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): fit.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file() and
            path.name != 'output-manifest.json'}})


def finalize(out, ledger, freeze, began, primary, writer):
    def note(stage, error):
        nonlocal primary
        ledger.setdefault('finalization_errors', []).append(
            {'stage': stage, 'error': repr(error)})
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
        evidence['notes'] = list(dict.fromkeys([
            *(evidence.get('notes') or []),
            *getattr(primary, '__notes__', [])]))
        ledger['primary_error'] = evidence

    ledger['process_wall_s'] = time.perf_counter() - began
    ledger['writer'] = writer.metadata()
    try:
        writer.save(out / 'advance-ledger.json', ledger)
    except BaseException as error:
        note('final_ledger', error)
    try:
        write_output_manifest(out, writer)
    except BaseException as error:
        note('final_manifest', error)
    if (freeze is not None and time.perf_counter() - began >
            freeze['internal_wall_limit_s']):
        note('post_manifest_timeout', TimeoutError(
            'E140 internal wall limit exceeded after manifest'))
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
                    writer.save(result_path, {'status': 'invalid_finalization',
                        'hypothesis_verdict': 'not_tested',
                        'new_parameter_package_accepted': False})
                except BaseException as fallback:
                    note('result_invalidation_fallback', fallback)
        keep_primary()
        ledger['writer'] = writer.metadata()
        try:
            writer.save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('failed_ledger', error)
        try:
            write_output_manifest(out, writer)
        except BaseException as error:
            note('failed_manifest', error)
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('One-shot E140 output exists: ' + str(out))
    out.mkdir(parents=True)
    writer = CompactWriter()
    ledger = {'schema': 'flylab.m3.e140.residual-trust-ledger.v1',
        'pid': os.getpid(), 'status': 'preflight',
        'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
        'original_evaluations': [], 'legs': [], 'fit_calls': 0,
        'raw_bytes': 0, 'body_calls': 0, 'LP_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False}
    writer.save(out / 'advance-ledger.json', ledger)
    freeze = limits = engine = None
    primary = None
    block_began = None
    try:
        freeze, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, review_path, review_sha)
        writer.save(out / 'source-review-snapshot.json', review)
        free = shutil.disk_usage(out).free
        ledger['free_disk_before_original_bytes'] = free
        if free < freeze['minimum_free_disk_bytes']:
            raise OSError('E140 frozen 12 GiB disk reserve unavailable')
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'cupy': cp.__version__, 'torch': torch.__version__,
            'gpu': gpu.decode() if isinstance(gpu, bytes) else str(gpu)}
        run_tiny_setup(out, ledger, writer)
        fixture = fit.frozen_problem(e128)
        prepared = prepared_inputs(ROOT / freeze['prepared_input_path'],
                                   fixture, freeze)
        geometry = TorchFootGeometry.load(
            ROOT / freeze['geometry_path'], device='cuda')
        torch.cuda.synchronize(geometry.device)
        if geometry.geometry_hash != freeze['geometry_hash']:
            raise ValueError('E140 frozen geometry hash mismatch')
        ledger['geometry_backend'] = type(geometry).__name__
        ledger['geometry_device'] = str(geometry.device)
        ledger['status'] = 'ready'
        writer.save(out / 'advance-ledger.json', ledger)
        block_began = time.perf_counter()
        block_deadline = block_began + freeze['fit_block_wall_limit_s']
        internal_deadline = began + freeze['internal_wall_limit_s']
        leg_results = []

        def persist():
            e139.refresh_counts(ledger)
            ledger['writer'] = writer.metadata()
            writer.save(out / 'advance-ledger.json', ledger)

        for spec in freeze['legs']:
            leg = spec['leg']
            leg_began = time.perf_counter()
            leg_deadline = leg_began + freeze['leg_wall_limit_s']
            prefix = f'leg{leg}_'
            leg_ledger = {'leg': leg, 'name': spec['name'],
                'status': 'preparing', 'original_indices': [],
                'comparison': None,
                'optimizer': {'status': 'not_requested',
                    'intent_persisted': False, 'requested': 0,
                    'returned': 0, 'bounded_stop': 0,
                    'failed_unknown': 0}}
            ledger['legs'].append(leg_ledger)
            persist()
            motors = prepared[prefix + 'motor_indices']
            muscles = prepared[prefix + 'muscle_indices']
            edges = prepared[prefix + 'edge_positions']
            bounds = e139.original_bounds(fixture, edges, muscles)
            engine = tau_module.CudaAdaptationTimescaleEngine(
                fixture, motors, muscles, edges)
            leg_ledger['engine_before'] = e136.snapshot(engine.metadata())
            for key, answer in (('motor_indices', motors),
                    ('muscle_indices', muscles),
                    ('edge_ids', prepared[prefix + 'edge_ids']),
                    ('parameter_muscle_map',
                        prepared[prefix + 'parameter_muscle_map'])):
                actual = (tau_module.activation_parameter_map(engine.problem)
                    if key == 'parameter_muscle_map' else engine.problem[key])
                if not np.array_equal(actual, answer):
                    raise ValueError('E140 engine identity changed: ' + key)
            start = np.r_[prepared[prefix + 'delta'],
                prepared[prefix + 'gain'], prepared[prefix + 'log_tau']]
            if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
                raise ValueError('E140 frozen start outside bounds')
            leg_ledger['status'] = 'ready'
            persist()

            def evaluate(vector, local_index):
                row = ledger['original_evaluations'][-1]
                directory = out / f"evaluation-{row['index']:03d}"

                def evaluator(x, outer_index, path, **callbacks):
                    return e139.evaluate_timescale_leg(engine, fixture,
                        geometry, leg, x, outer_index, path, **callbacks)

                try:
                    record, _ = e136.evaluate_with_receipts(ledger, row,
                        vector, directory, evaluator, persist,
                        metadata=engine.metadata,
                        maximum_raw_bytes=freeze['maximum_raw_bytes'])
                    with np.load(record['raw_path'],
                                 allow_pickle=False) as raw:
                        payload = augmented_residual_jac(raw,
                            spec['original_objective_guard'],
                            spec['baseline_objective'], spec['edges'],
                            spec['muscles'])
                    residual_path = directory / 'residual-jacobian-raw.npz'
                    account_residual_raw(ledger, row, residual_path, payload,
                                         freeze['maximum_raw_bytes'])
                    row.update(residual_value=payload['value'],
                        gradient_norm=payload['gradient_norm'],
                        active_count=payload['active_count'],
                        startup_negative_count=payload['startup_negative_count'],
                        guard3=payload['guard3'].tolist(),
                        guard_ratios=payload['guard_ratios'].tolist(),
                        minimum_startup_residual=(
                            payload['minimum_startup_residual']),
                        minimum_guard_residual=(
                            payload['minimum_guard_residual']),
                        objective_over_baseline=(
                            payload['objective_over_baseline']),
                        qualifies=payload['qualifies'])
                    persist()
                    if local_index == 1:
                        comparison = compare_start(record['raw_path'],
                            ROOT / spec['start_raw_path'])
                        writer.save(out / f'leg-{leg}-start-comparison.json',
                                    comparison)
                        leg_ledger['comparison'] = comparison
                        persist()
                        if not comparison['accepted']:
                            raise ValueError('E140 start replay parity failed')
                        if (not np.isclose(record['true_worst_objective'],
                                spec['start_objective'], rtol=0, atol=1e-10) or
                                payload['qualifies']):
                            raise ValueError('E140 fixed start status changed')
                    record.update(residual=payload['residual'],
                        jacobian=payload['jacobian'],
                        qualifies=payload['qualifies'])
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
                        error.add_note('Secondary E140 evaluation ledger: ' +
                                       repr(secondary))
                    raise

            def on_start(local_index, vector):
                row = register_intent(ledger, spec, local_index,
                                      vector, persist)
                leg_ledger['original_indices'].append(row['index'])
                persist()

            def on_complete(cache, _data, local_index):
                row = ledger['original_evaluations'][
                    leg_ledger['original_indices'][local_index - 1] - 1]
                complete_receipt(ledger, leg_ledger, cache, row, persist)

            cache = ResidualCache(evaluate, start, bounds, on_start,
                on_complete, max_unique=18, leg_deadline=leg_deadline,
                block_deadline=block_deadline,
                internal_deadline=internal_deadline)
            try:
                solver_result, stop_reason = run_optimizer(
                    cache, leg_ledger, persist)
            finally:
                active = sys.exc_info()[1]
                leg_ledger['wall_s'] = time.perf_counter() - leg_began
                try:
                    leg_ledger['engine_after'] = e136.snapshot(engine.metadata())
                    persist()
                except BaseException as secondary:
                    if active is not None:
                        active.add_note('Secondary E140 leg ledger: ' +
                                        repr(secondary))
                    else:
                        raise
            leg_ledger.update(stop_reason=stop_reason,
                solver_returned=solver_result is not None)
            rows = [ledger['original_evaluations'][index - 1]
                    for index in leg_ledger['original_indices']]
            result = leg_result(spec, rows, cache, leg_ledger)
            writer.save(out / f'leg-{leg}-result.json', result)
            if result['status'] != 'completed_bounded_residual_trust':
                raise ValueError('E140 incomplete leg evidence: ' + str(leg))
            leg_results.append(result)
            leg_ledger.update(status=result['status'], result=result)
            persist()
            engine = None
        ledger['fit_block_wall_s'] = time.perf_counter() - block_began
        e139.refresh_counts(ledger)
        if (ledger['fit_block_wall_s'] > freeze['fit_block_wall_limit_s'] or
                time.perf_counter() >= internal_deadline - 1 or
                len(ledger['original_evaluations']) > 54 or
                ledger['fit_calls'] != 3 or
                ledger['original_evaluations_requested'] !=
                    ledger['original_evaluations_returned'] or
                ledger['original_evaluations_requested'] !=
                    ledger['original_evaluations_completed'] or
                ledger['original_evaluations_failed_unknown'] or
                ledger['original_evaluations_post_return_error'] or
                ledger['raw_bytes'] > freeze['maximum_raw_bytes'] or
                ledger.get('synthetic_tiny', {}).get('status') != 'completed' or
                ledger.get('synthetic_tiny', {}).get('raw_saved') is not True or
                ledger.get('synthetic_tiny', {}).get('solve_returned') is not True or
                not all(row['residual_raw_saved'] is True
                        for row in ledger['original_evaluations'])):
            raise ValueError('E140 whole original accounting incomplete')
        all_supported = all(x['new_leg_proposal_accepted'] for x in leg_results)
        package = (write_package(out, fixture, prepared, ledger, leg_results)
                   if all_supported else None)
        result = {'schema': 'flylab.m3.e140.residual-trust-result.v1',
            'status': 'completed_bounded_residual_trust',
            'hypothesis_verdict': 'supported' if all_supported else 'refuted',
            'interpretation': 'three_leg_conditional_residual_trust_only',
            'legs': leg_results,
            'new_parameter_package_accepted': all_supported,
            'parameter_package': package,
            'original_evaluations_requested':
                ledger['original_evaluations_requested'],
            'original_evaluations_returned':
                ledger['original_evaluations_returned'],
            'original_evaluations_completed':
                ledger['original_evaluations_completed'],
            'fit_calls': ledger['fit_calls'],
            'fit_block_wall_s': ledger['fit_block_wall_s'],
            'raw_bytes': ledger['raw_bytes'], 'writer': writer.metadata(),
            'body_calls': 0, 'LP_calls': 0, 'F6_exposed': False,
            'candidate_adopted': False, 'whole_candidate_adopted': False,
            'whole_goal_complete': False, 'freeze_sha256': freeze_sha,
            'source_review_sha256': review_sha}
        writer.save(out / 'result.json', result)
        ledger['status'] = result['status']
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'response_progress': getattr(error, 'response_progress', None),
            'notes': list(getattr(error, '__notes__', []))}
    finally:
        e139.refresh_counts(ledger)
        if engine is not None:
            try:
                ledger['last_engine_after'] = e136.snapshot(engine.metadata())
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E140 engine metadata: ' +
                                     repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E140 thread restore: ' +
                                     repr(error))
        primary = finalize(out, ledger, freeze, began, primary, writer)
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
