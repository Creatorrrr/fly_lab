"""E139 one-process, five-leg bounded timescale startup restoration.

The parent supplies a sealed source review before the one original execution.
This module never performs whole-BANC, body, F6, or UI work.
"""

# ruff: noqa: BLE001, B023  # preserve failures; loop callbacks finish synchronously

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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_adaptation_timescale as tau_module
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import fit_c_banc_m3_timescale_constrained as e136

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = 'ba3a70bb15f3b4cfb492cca8ca18ddd2caf44c304076713418475dd0223787d2'
SOURCE_FILES = ('tools/fit_c_banc_m3_cross_leg_timescale_restoration.py',
                'tests/test_c_banc_cross_leg_timescale_restoration.py')
OPTIONS = {'maxiter': 100, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10}
LEG_DIMENSIONS = ((1, 63, 15, 281, 311, 15),
                  (2, 63, 16, 240, 272, 16),
                  (3, 70, 17, 238, 272, 16),
                  (4, 63, 15, 255, 285, 17),
                  (5, 63, 16, 285, 317, 15))
OLD_EXACT = ('delta', 'gain', 'edge_ids', 'motor_indices',
             'muscle_indices')
OLD_METRIC = ('losses', 'parts', 'ratios', 'q_error', 'ltm_error',
              'q_coefficient', 'ltm_coefficient', 'cotangent')
WORK_KEYS = ('conditional_s', 'context_ticks', 'scalar_MN_ticks',
    'edge_stage_ticks', 'derivative_banks', 'decoder_endpoints',
    'geometry_forward_rows', 'geometry_backward_rows')


def save(path, value):
    e136.save(path, value)


def checked_path(relative, digest):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if (name.is_absolute() or '..' in name.parts or
            not path.is_relative_to(ROOT.resolve()) or
            not isinstance(digest, str) or len(digest) != 64 or
            fit.sha256(path) != digest):
        raise ValueError('E139 frozen path/SHA mismatch: ' + str(relative))
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or fit.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E139 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    fixed = {'schema': 'flylab.m3.e139.cross-leg-timescale-restoration-freeze.v1',
        'leg_order': [1, 2, 3, 4, 5], 'original_processes': 1,
        'original_optimizer_calls_max': 5, 'retries': 0,
        'original_lf_fit_calls': 0, 'solver': 'L-BFGS-B',
        'solver_options': OPTIONS, 'proposal_objective_ratio_limit': .99,
        'optimizer_cap': fit.OPTIMIZER_CAP,
        'selector_cap': fit.SELECTOR_CAP,
        'max_unique_per_leg': 24, 'max_unique_total': 120,
        'baseline_replay_per_leg': 1, 'start_replay_per_leg': 1,
        'leg_wall_limit_s': 40, 'leg_reserve_s': 3,
        'fit_block_wall_limit_s': 210, 'process_reserve_s': 10,
        'internal_wall_limit_s': 230, 'external_wall_limit_s': 240,
        'supervisor_limit_s': 260, 'maximum_raw_bytes': 8589934592,
        'minimum_free_disk_bytes': 12884901888,
        'cpu_threads': 1, 'gpu_count': 1, 'full391_batches': 0,
        'full_BANC_calls': 0, 'body_calls': 0, 'LP_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'active_UI_changed': False}
    if (any(freeze.get(key) != value for key, value in fixed.items()) or
            freeze.get('owned_files') != list(SOURCE_FILES) or
            freeze.get('log_tau_bounds') != [-math.log(3), math.log(10)] or
            freeze.get('tiny') != {'batches': 1, 'context_ticks': 20,
                'conditional_s': .005, 'derivative_banks': 1} or
            freeze.get('design_sha256') != fit.sha256(
                ROOT / freeze.get('design_path', '')) or
            len(freeze.get('input_sha256', {})) != 57 or
            len(freeze.get('reused_source_sha256', {})) != 48 or
            len(freeze.get('preservation_sha256', {})) != 1460 or
            len(freeze.get('legs', [])) != 5 or
            freeze.get('prepared_input_sha256') !=
                freeze['input_sha256'].get(freeze.get('prepared_input_path'))):
        raise ValueError('E139 fixed freeze contract changed')
    for spec, expected in zip(freeze['legs'], LEG_DIMENSIONS, strict=True):
        _, neurons, _, edges, _, _ = expected
        if (tuple(spec.get(key) for key in ('leg', 'neurons', 'muscles',
                'edges', 'parameters', 'start_index')) != expected or
                spec.get('max_unique_evaluations') != 24 or
                spec.get('baseline_index') != 1 or
                not np.isclose(spec.get('original_objective_guard'),
                    .99 * spec.get('baseline_objective'), rtol=0,
                    atol=1e-15) or
                spec.get('work_per_evaluation') != work_per_evaluation(
                    neurons, edges)):
            raise ValueError('E139 frozen leg dimension/start/work changed')
        for label in ('start', 'baseline'):
            relative = spec[label + '_raw_path']
            if freeze['input_sha256'].get(relative) != spec[label + '_raw_sha256']:
                raise ValueError('E139 predecessor raw not frozen')
    expected_max = {key: sum(24 * spec['work_per_evaluation'][key]
        for spec in freeze['legs']) for key in WORK_KEYS}
    if freeze.get('max_original_work') != expected_max:
        raise ValueError('E139 maximum original work changed')
    for mapping in ('input_sha256', 'reused_source_sha256',
                    'preservation_sha256'):
        for relative, digest in freeze[mapping].items():
            checked_path(relative, digest)
    if fit.sha256(review_path) != review_sha:
        raise ValueError('E139 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_cross_leg_restoration')
            is not True or review.get('blocking_defects') != [] or
            review.get('freeze_sha256') != FREEZE_SHA or
            set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E139 two-source review required')
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
        raise ValueError('E139/E128 frozen fixture/context mismatch')
    return freeze, e128, review


def work_per_evaluation(neurons, edges):
    return {'conditional_s': 30, 'context_ticks': 120000,
        'scalar_MN_ticks': 120000 * neurons,
        'edge_stage_ticks': 480000 * edges, 'derivative_banks': 1,
        'decoder_endpoints': 6000, 'geometry_forward_rows': 5473,
        'geometry_backward_rows': 5473}


def prior_and_gradient(data, edges, muscles):
    delta = np.asarray(data['delta'], np.float64)
    gain = np.asarray(data['gain'], np.float64)
    if delta.shape != (edges,) or gain.shape != (muscles,):
        raise ValueError('E139 leg prior dimension changed')
    prior = 6e-5 * np.sum(delta * delta) / 1572 + 1e-5 * np.sum(gain * gain) / 96
    gradient = np.r_[1.2e-4 * delta / 1572,
        2e-5 * gain / 96, np.zeros(muscles)]
    return float(prior), gradient


def initial_jacobians(b_initial, q_coefficient, ltm_coefficient,
                      mapping, ltm_local, muscles):
    b = np.asarray(b_initial, np.float64)
    q = np.asarray(q_coefficient, np.float64)
    ltm = np.asarray(ltm_coefficient, np.float64)
    mapping = np.asarray(mapping)
    if (b.ndim != 3 or b.shape[:2] != (3, 99) or
            q.shape != (3, 99, 7, muscles) or
            ltm.shape != (3, 99) or
            mapping.shape != (b.shape[-1],) or
            mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or
            np.any(mapping >= muscles) or not 0 <= ltm_local < muscles):
        raise ValueError('E139 initial Jacobian muscle map mismatch')
    return (b[:, :, None, :] * q[:, :, :, mapping],
        b * ltm[:, :, None] * (mapping == ltm_local)[None, None, :])


def saved_b_loss_jac(b, cotangent, mapping, muscles):
    b = np.asarray(b, np.float64)
    cotangent = np.asarray(cotangent, np.float64)
    mapping = np.asarray(mapping)
    if (b.ndim != 3 or cotangent.shape != (*b.shape[:2], muscles) or
            mapping.shape != (b.shape[-1],) or
            mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or
            np.any(mapping >= muscles)):
        raise ValueError('E139 full B/cotangent mapping mismatch')
    return np.einsum('ctp,ctp->cp', b, cotangent[:, :, mapping])


def restoration_value_jac(data, guard, baseline, edges, muscles):
    losses = np.asarray(data['losses'], np.float64)
    prior, prior_gradient = prior_and_gradient(data, edges, muscles)
    c, jac, ratios = fit.constraint_values(losses, data['loss_jac'],
        data['q_error'], data['ltm_error'], data['q_jac'],
        data['ltm_jac'], float(max(losses) + .01))
    parameters = edges + 2 * muscles
    if (c.shape != (4761,) or jac.shape != (4761, parameters + 1) or
            not np.isclose(guard, .99 * baseline, rtol=0, atol=1e-12)):
        raise ValueError('E139 original startup/guard dimensions changed')
    guard3 = (guard - losses - prior) / guard
    guard_jac = -(np.asarray(data['loss_jac']) +
                  prior_gradient[None, :]) / guard
    residuals = np.r_[c[3:], guard3]
    augmented_jac = np.vstack((jac[3:, :-1], guard_jac))
    negative = np.minimum(residuals, 0.)
    value = float(.5 * negative @ negative)
    gradient = np.asarray(augmented_jac.T @ negative, np.float64)
    objective = float(max(losses) + prior)
    ratio = objective / baseline
    if (residuals.shape != (4761,) or
            augmented_jac.shape != (4761, parameters) or
            not np.isfinite(residuals).all() or
            not np.isfinite(augmented_jac).all() or
            not np.isfinite(gradient).all() or
            not np.isfinite([value, objective, ratio]).all()):
        raise ValueError('Nonfinite E139 restoration evidence')
    return {'value': value, 'gradient': gradient,
        'gradient_norm': float(np.linalg.norm(gradient)),
        'startup_negative_count': int(np.count_nonzero(c[3:] < 0)),
        'guard3': guard3, 'guard_ratios': (losses + prior) / guard,
        'ratios': ratios, 'prior': prior,
        'original_objective': objective,
        'objective_over_baseline': float(ratio),
        'qualifies': bool(fit.selector_feasible(ratios) and ratio <= .99),
        'minimum_startup_residual': float(np.min(c[3:])),
        'minimum_guard_residual': float(np.min(guard3))}


def original_bounds(fixture, edges, muscles):
    bounds = np.vstack((np.asarray(fixture['delta_bounds'][edges], np.float64),
        np.tile(np.array([[0., 8.]]), (len(muscles), 1)),
        np.tile(np.array([[-math.log(3), math.log(10)]]),
                (len(muscles), 1))))
    if (bounds.shape != (len(edges) + 2 * len(muscles), 2) or
            not np.isfinite(bounds).all() or
            np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError('E139 original bounds dimension changed')
    return bounds


def leg_observation_inputs(fixture, leg, ci, selected, muscles):
    """One leg's global-to-local E128 targets, scales, and LTM identity."""
    start = 7 * leg
    ltm_matches = np.flatnonzero(muscles == fixture['ltm'][leg])
    if len(ltm_matches) != 1:
        raise ValueError('E139 one local leg LTM muscle required')
    return {'joint_start': start, 'ltm_local': int(ltm_matches[0]),
        'q_target': fixture['own_target'][ci, selected, start:start + 7],
        'foot_target': fixture['target_feet'][ci, selected, leg],
        'joint_scale': fixture['joint_scale'][start:start + 7],
        'foot_scale': fixture['foot_scale'][leg],
        'velocity_scale': fixture['velocity_scale'][leg],
        'startup_q_target': fixture['startup_target_command'][:,
            start:start + 7],
        'startup_ltm_target': fixture['startup_target_tendon'][:, leg],
        'moment': fixture['moment'][start:start + 7][:, muscles]}


def compare_replay(actual_path, reference_path, *, baseline, edges,
                   muscles, leg):
    """E1281 has eight response fields; later E128 rows do not."""
    report = {'accepted': False, 'kind': 'baseline' if baseline else 'start',
        'actual_sha256': fit.sha256(actual_path),
        'reference_sha256': fit.sha256(reference_path), 'fields': {}}
    exact = [*OLD_EXACT, 'parameter_muscle_map']
    if baseline:
        exact.append('branch_bits')
    response = list(fit.FIELDS) if baseline else []
    metric = [*OLD_METRIC, 'loss_jac']
    old_parameters = edges + muscles
    with (np.load(actual_path, allow_pickle=False) as actual,
          np.load(reference_path, allow_pickle=False) as reference):
        for key in (*exact, *response, 'B_initial99', *metric):
            if key not in actual or key not in reference:
                report['fields'][key] = {'accepted': False, 'reason': 'missing'}
                continue
            a, b = actual[key], reference[key]
            if key in ('parameter_muscle_map', 'loss_jac', 'B_initial99'):
                a = a[..., :old_parameters]
            if (key == 'command' and a.shape[-1] == 7 and
                    b.shape[-1] == 42):
                b = b[..., 7 * leg:7 * leg + 7]
            if (key == 'tendon' and a.shape[-1] == 1 and
                    b.shape[-1] == 6):
                b = b[..., leg:leg + 1]
            if a.shape != b.shape or a.dtype != b.dtype:
                report['fields'][key] = {'accepted': False,
                    'reason': 'shape_or_dtype', 'actual_shape': a.shape,
                    'reference_shape': b.shape}
                continue
            limit = (0 if key in exact else
                1e-8 if key in response or key == 'B_initial99' else 1e-10)
            difference = (float(np.max(np.abs(a - b))) if a.size and limit
                          else 0.)
            accepted = bool(np.array_equal(a, b) if not limit else
                np.isfinite(a).all() and np.isfinite(b).all() and
                difference <= limit)
            report['fields'][key] = {'accepted': accepted,
                'max_abs_difference': difference, 'limit': limit}
        zero_tau = actual.get('log_tau', None)
        report['fields']['log_tau_zero'] = {'accepted': bool(
            zero_tau is not None and zero_tau.shape == (muscles,) and
            zero_tau.dtype == np.float64 and
            np.array_equal(zero_tau, np.zeros(muscles))), 'limit': 0}
        report['accepted'] = all(
            item['accepted'] for item in report['fields'].values())
    return report


class BoundedStop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class LegCache:
    """Per-leg byte cache; baseline is outside it, all terminal states latch."""

    def __init__(self, evaluate, start, bounds, on_start, on_complete,
                 *, max_unique, leg_deadline, block_deadline,
                 internal_deadline, clock=time.perf_counter):
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
        self.started = 0
        self.completed = 0
        self.hits = 0
        self.selected_index = None
        self.terminal_error = None
        self.optimizer_invoked = False

    def get(self, vector):
        if self.terminal_error is not None:
            raise self.terminal_error
        x = np.asarray(vector, np.float64)
        if (x.shape != (len(self.bounds),) or not np.isfinite(x).all() or
                np.any(x < self.bounds[:, 0] - 1e-12) or
                np.any(x > self.bounds[:, 1] + 1e-12)):
            self.terminal_error = ValueError('E139 finite in-bounds vector required')
            raise self.terminal_error
        key = np.ascontiguousarray(x).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.started >= self.max_unique:
            self.terminal_error = BoundedStop('leg_unique_cap')
            raise self.terminal_error
        now = self.clock()
        if (now >= self.leg_deadline - 3 or
                now >= self.block_deadline or
                now >= self.internal_deadline - 10):
            self.terminal_error = BoundedStop('pre_call_wall_reserve')
            raise self.terminal_error
        index = self.started + 1  # index 1 is the start, after baseline replay
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
                'E139 frozen infeasible start unexpectedly qualifies')
            raise self.terminal_error
        if data['qualifies']:
            self.terminal_error = BoundedStop('first_guarded_selector_feasible')
            raise self.terminal_error
        return result


def run_leg_optimizer(cache, leg_ledger, persist, *, solver=minimize,
                      clock=time.perf_counter):
    accounting = leg_ledger['optimizer']
    if cache.optimizer_invoked:
        if cache.terminal_error is None:
            cache.terminal_error = RuntimeError(
                'E139 optimizer invocation already in progress')
        raise cache.terminal_error
    cache.optimizer_invoked = True
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
        try:
            persist()
        except BaseException:
            accounting.update(status='request_receipt_failed', requested=0,
                request_receipt_persisted=False, in_flight=False)
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
                    np.asarray(result.x).shape != cache.start.shape or
                    not np.isfinite(result.x).all() or
                    (not result.success and int(result.status) != 1)):
                raise ValueError('E139 optimizer returned invalid numerical state')
            if not result.success:
                reason = 'scipy_iteration_or_function_limit'
    except BaseException as error:
        cache.terminal_error = error
        raise
    else:
        if cache.terminal_error is None:
            cache.terminal_error = BoundedStop(
                'optimizer_terminal_' + reason)
    finally:
        accounting['elapsed_wall_s'] = clock() - began
        leg_ledger.update(cache_started=cache.started,
            cache_completed=cache.completed, cache_hits=cache.hits)
    return result, reason


def cuda_cotangent_product(engine, device_result, cotangent):
    """Use the unchanged E134 B, indexed by each column's local muscle."""
    import cupy as cp
    import torch

    expected = (engine.contexts, engine.samples, engine.muscles)
    if (not isinstance(cotangent, torch.Tensor) or
            cotangent.shape != expected or cotangent.dtype != torch.float64 or
            cotangent.device.type != 'cuda' or
            device_result['B'].shape !=
                (engine.contexts, engine.samples,
                 engine.edges + 2 * engine.muscles)):
        raise ValueError('E139 CUDA B/cotangent shape or dtype mismatch')
    tensor = cotangent.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        engine.torch_contiguous_copy_bytes += int(tensor.numel() * 8)
    torch.cuda.synchronize(tensor.device)
    with engine.device, engine.stream:
        d_cotangent = cp.from_dlpack(tensor)
        engine.torch_cupy_shared_bytes += int(tensor.numel() * 8)
        mapping = engine.device_indices(
            tau_module.activation_parameter_map(engine.problem),
            'timescale_parameter_muscles')
        gradient = cp.sum(device_result['B'] *
            d_cotangent[:, :, mapping], axis=1)
        engine.stream.synchronize()
        return engine.host_copy(gradient, label='timescale_loss_gradient')


def evaluate_timescale_leg(engine, fixture, geometry, leg, vector, index, out,
                           *, on_solve_entry=None, on_solve_return=None,
                           on_raw_saved=None):
    """One actual variable-dimension E134 solve and original E128 losses."""
    import torch

    engine.geometry_backend = type(geometry).__name__
    engine.geometry_device = str(getattr(geometry, 'device', 'synthetic_cuda'))
    p = engine.problem
    edge_count, muscle_count = engine.edges, engine.muscles
    x = np.asarray(vector, np.float64)
    if x.shape != (edge_count + 2 * muscle_count,) or not np.isfinite(x).all():
        raise ValueError('E139 finite edge/gain/logtau vector required')
    delta = np.ascontiguousarray(x[:edge_count])
    gain = np.ascontiguousarray(x[edge_count:edge_count + muscle_count])
    log_tau = np.ascontiguousarray(x[edge_count + muscle_count:])
    mapping = tau_module.activation_parameter_map(p)
    muscles = p['muscle_indices']
    starts = np.flatnonzero(fixture['startup_mask'])
    if (mapping.shape != (edge_count + 2 * muscle_count,) or
            len(starts) != 99 or
            not np.array_equal(starts, np.arange(1, 100))):
        raise ValueError('E139 startup/map contract changed')
    raw = {'delta': delta, 'gain': gain, 'log_tau': log_tau,
        'edge_ids': p['edge_ids'], 'motor_indices': p['motor_indices'],
        'muscle_indices': muscles, 'parameter_muscle_map': mapping,
        'leg': np.array(leg), 'outer_evaluation_index': np.array(index)}
    d, h, timing = {}, {}, None
    started = time.perf_counter()
    try:
        if on_solve_entry is not None:
            on_solve_entry()
        d, h, timing = engine.solve(delta, gain, log_tau, derivative=True,
                                    host=True)
        if on_solve_return is not None:
            on_solve_return(timing, h)
        raw.update(h)
        raw['B'] = engine.host_copy(d['B'], label='full_B')
        raw['B_initial99'] = raw['B'][:, starts, :].copy()
        raw['timing_context_ticks'] = np.array(timing['context_rk4_ticks'])
        raw['timing_bank_complete'] = np.array(
            timing['progress']['derivative_bank_complete'])
        engine.stream.synchronize()
        activation_all = torch.from_dlpack(d['activation']).detach(
            ).requires_grad_(True)
        engine.torch_activation_shared_bytes += int(d['activation'].nbytes)
        start_indices = fit.torch_indices(engine, starts,
                                          activation_all.device,
                                          'startup_indices')
        losses = np.empty(3, np.float64)
        parts = np.empty((3, 4), np.float64)
        q_error = np.empty((3, 99, 7), np.float64)
        ltm_error = np.empty((3, 99), np.float64)
        q_coefficient = np.empty((3, 99, 7, muscle_count), np.float64)
        ltm_coefficient = np.empty((3, 99), np.float64)
        ltm_matches = np.flatnonzero(muscles == fixture['ltm'][leg])
        if len(ltm_matches) != 1:
            raise ValueError('E139 one local leg LTM muscle required')
        ltm_local = int(ltm_matches[0])
        loss_tensors = []
        joint_start = 7 * leg
        moment = .02 * fixture['moment'][
            joint_start:joint_start + 7, muscles]
        for ci in range(3):
            selected = np.flatnonzero(fixture['fit_time_mask'][ci])
            if len(selected) < 2 or not np.all(np.diff(selected) == 1):
                raise ValueError('E139 contiguous fit mask required')
            activation = activation_all[ci]
            selected_indices = fit.torch_indices(engine, selected,
                activation.device, 'fit_indices')
            command, tendon = fit.torch_leg_loss(engine, activation,
                fixture, leg, geometry, muscles)
            engine.geometry_cost['forward_rows_requested'] += len(selected)
            foot = geometry.leg(command[selected_indices], leg)
            engine.geometry_cost['forward_rows_enqueued'] += len(selected)
            raw[f'context{ci}_fit_feet'] = fit.torch_host_array(engine,
                foot, f'context{ci}_fit_feet')
            engine.geometry_cost['forward_rows_completed'] += len(selected)
            velocity = (foot[1:] - foot[:-1]) / .005
            device = activation.device

            def to(value, label, target_device=device):
                return fit.torch_input_array(engine, value, target_device,
                                             label)

            q_target = to(fixture['own_target'][ci, selected,
                joint_start:joint_start + 7], 'q_target')
            foot_target = to(fixture['target_feet'][ci, selected, leg],
                             'foot_target')
            v_target = (foot_target[1:] - foot_target[:-1]) / .005
            q_loss = ((command[selected_indices] - q_target) /
                to(fixture['joint_scale'][joint_start:joint_start + 7],
                   'joint_scale')).square().mean()
            foot_loss = ((foot - foot_target) /
                to(fixture['foot_scale'][leg], 'foot_scale')).square().mean()
            vel_loss = ((velocity - v_target) /
                to(fixture['velocity_scale'][leg],
                   'velocity_scale')).square().mean()
            ltm_loss = ((tendon[selected_indices] - .2) / .2).square().mean()
            values = torch.stack((q_loss, foot_loss, vel_loss, ltm_loss))
            loss_tensors.append(values.sum())
            parts[ci] = fit.torch_host_array(engine, values, 'loss_parts')
            losses[ci] = float(fit.torch_host_array(engine, values.sum(),
                                                    'loss_scalar'))
            q_error[ci] = fit.torch_host_array(engine,
                command[start_indices] - to(
                    fixture['startup_target_command'][:,
                        joint_start:joint_start + 7],
                    'startup_target_command'), 'initial_q_error')
            ltm_error[ci] = fit.torch_host_array(engine,
                tendon[start_indices] - to(
                    fixture['startup_target_tendon'][:, leg],
                    'startup_target_tendon'), 'initial_ltm_error')
            q_coefficient[ci] = np.broadcast_to(moment[None],
                                                (99, 7, muscle_count))
            initial_activation = fit.torch_host_array(engine,
                activation[start_indices, ltm_local],
                'initial_ltm_activation')
            ltm_coefficient[ci] = .02 * (1 - np.tanh(
                initial_activation / 10) ** 2)
        geometry_rows = int(fixture['fit_time_mask'].sum())
        engine.geometry_cost['backward_rows_requested'] += geometry_rows
        cotangent_t = torch.autograd.grad(sum(loss_tensors),
                                           activation_all)[0]
        engine.geometry_cost['backward_rows_enqueued'] += geometry_rows
        torch.cuda.synchronize(activation_all.device)
        engine.geometry_cost['backward_rows_completed'] += geometry_rows
        loss_jac = cuda_cotangent_product(engine, d, cotangent_t)
        cotangent = fit.torch_host_array(engine, cotangent_t,
                                         'loss_cotangent_raw')
        q_jac, ltm_jac = initial_jacobians(raw['B_initial99'],
            q_coefficient, ltm_coefficient, mapping, ltm_local, muscle_count)
        _, _, ratios = fit.constraint_values(losses, loss_jac, q_error,
            ltm_error, q_jac, ltm_jac, float(max(losses) + .01))
        raw.update(losses=losses, parts=parts, loss_jac=loss_jac,
            q_error=q_error, ltm_error=ltm_error,
            q_coefficient=q_coefficient, ltm_coefficient=ltm_coefficient,
            q_jac=q_jac, ltm_jac=ltm_jac, cotangent=cotangent,
            ratios=ratios, transfer_cost_json=np.array(json.dumps(
                e136.snapshot({'host_transfer': engine.host_transfer,
                'host_transfer_by_label': engine.host_transfer_by_label,
                'torch_host_transfer': engine.torch_host_transfer,
                'torch_host_transfer_by_label':
                    engine.torch_host_transfer_by_label,
                'cotangent_upload': engine.cotangent_upload,
                'index_upload': engine.index_upload,
                'index_upload_by_label': engine.index_upload_by_label,
                'torch_input_transfer': engine.torch_input_transfer,
                'torch_input_transfer_by_label':
                    engine.torch_input_transfer_by_label,
                'torch_cupy_shared_bytes': engine.torch_cupy_shared_bytes,
                'torch_contiguous_copy_bytes': engine.torch_contiguous_copy_bytes,
                'geometry_cost': engine.metadata()['geometry_cost']}))))
        path = out / f'leg-{leg}-evaluation-001-raw.npz'
        np.savez(path, **raw)
        digest = fit.sha256(path)
        if on_raw_saved is not None:
            on_raw_saved(path, digest)
        if any(not np.isfinite(value).all() for value in
               (raw['B'], losses, parts, loss_jac, q_error, ltm_error,
                q_jac, ltm_jac, ratios, cotangent)):
            raise ValueError('Nonfinite E139 saved original raw: ' + str(path))
        prior, _ = prior_and_gradient(raw, edge_count, muscle_count)
        return {'index': index, 'leg': leg, 'delta': delta, 'gain': gain,
            'log_tau': log_tau, 'losses': losses, 'parts': parts,
            'loss_jac': loss_jac, 'q_error': q_error,
            'ltm_error': ltm_error, 'q_jac': q_jac, 'ltm_jac': ltm_jac,
            'ratios': ratios, 'prior': prior,
            'true_worst_objective': float(max(losses) + prior),
            'raw_path': str(path), 'raw_sha256': digest,
            'solve_wall_s': timing['wall_s'],
            'evaluate_wall_s': time.perf_counter() - started}
    except BaseException as error:
        if timing is not None:
            error.response_progress = timing['progress']
        rescued = dict(h)
        if d:
            try:
                more, failures = engine.rescue_device_state(d, starts)
                for key, value in more.items():
                    rescued.setdefault(key, value)
                for failure in failures:
                    error.add_note('Secondary E139 response rescue: ' +
                                   str(failure))
            except BaseException as rescue_error:
                error.add_note('Secondary E139 rescue protocol: ' +
                               repr(rescue_error))
        rescued.update({key: value for key, value in raw.items()
                        if isinstance(value, np.ndarray)})
        rescued.update(getattr(error, 'partial_response', {}))
        try:
            rescued['failure_type'] = np.array(type(error).__name__)
            rescued['failure_message'] = np.array(str(error))
            partial = out / f'leg-{leg}-evaluation-001-partial-raw.npz'
            np.savez(partial, **rescued)
            error.partial_raw_path = str(partial)
        except BaseException as save_error:
            error.add_note('Secondary E139 partial raw save: ' +
                           repr(save_error))
        raise


def requested_work(row):
    if row.get('solve_requested') is not True:
        return {}
    progress = row.get('response_progress')
    if not isinstance(progress, dict):
        return {}
    events = progress.get('kernel_events')
    if isinstance(events, list):
        names = {event.get('name') for event in events if
                 isinstance(event, dict) and event.get('requested') is True}
        forward, decoder, bank = (key in names for key in
                                  ('forward', 'decoder', 'bank'))
    else:
        forward, decoder, bank = (progress.get(key) not in
            (None, 'not_enqueued') for key in
            ('forward_completion_class', 'decoder_completion_class',
             'bank_completion_class'))
    maximum = row['work_per_evaluation']
    work = {}
    if forward:
        for key in ('context_ticks', 'conditional_s', 'scalar_MN_ticks'):
            work[key] = maximum[key]
    if decoder:
        work['decoder_endpoints'] = maximum['decoder_endpoints']
    if bank:
        work['edge_stage_ticks'] = maximum['edge_stage_ticks']
        work['derivative_banks'] = 1
    return work


def known_work(row):
    if row.get('solve_requested') is not True:
        return {}
    progress = row.get('response_progress')
    if not isinstance(progress, dict):
        return {}
    maximum = row['work_per_evaluation']
    forward = progress.get('forward_completion_class') == 'known_complete'
    decoder = progress.get('decoder_completion_class') == 'known_complete'
    bank = (progress.get('bank_completion_class') == 'known_complete' and
            progress.get('derivative_bank_complete') is True)
    specs = {'context_ticks': ('completed_context_rk4_ticks', forward),
        'conditional_s': ('completed_conditional_model_s', forward),
        'decoder_endpoints': ('completed_decoder_endpoint_ticks', decoder),
        'edge_stage_ticks': ('completed_bank_edge_stage_ticks', bank)}
    known = {}
    for key, (field, complete) in specs.items():
        value = progress.get(field)
        if (complete and isinstance(value, (int, float)) and
                np.isfinite(value) and 0 <= value <= maximum[key]):
            known[key] = value
    if 'context_ticks' in known:
        known['scalar_MN_ticks'] = known['context_ticks'] * row['neurons']
    if bank:
        known['derivative_banks'] = 1
    return known


def refresh_counts(ledger):
    """Dimension-aware original work; unknown never becomes known complete."""
    rows = ledger['original_evaluations']
    ledger['original_evaluations_intended'] = sum(
        row.get('intent_persisted') is True for row in rows)
    ledger['original_evaluations_requested'] = sum(
        row.get('solve_requested') is True for row in rows)
    ledger['original_evaluations_returned'] = sum(
        row.get('solve_returned') is True for row in rows)
    ledger['original_evaluations_completed'] = sum(
        row.get('raw_saved') is True for row in rows)
    ledger['original_evaluations_failed_unknown'] = sum(
        row.get('solve_requested') is True and
        row.get('solve_returned') is not True for row in rows)
    ledger['original_evaluations_post_return_error'] = sum(
        row.get('post_return_error') is True for row in rows)
    ledger['original_evaluations_unstarted'] = sum(
        row.get('solve_requested') is not True for row in rows)
    for key in WORK_KEYS:
        if key.startswith('geometry_'):
            direction = key.split('_')[1]
            counter = f'{direction}_rows'
            requested = sum(row.get('geometry_delta', {}).get(
                counter + '_requested', 0) for row in rows)
            enqueued = sum(row.get('geometry_delta', {}).get(
                counter + '_enqueued', 0) for row in rows)
            completed = sum(row.get('geometry_delta', {}).get(
                counter + '_completed', 0) for row in rows)
            missing = sum(row.get('work_per_evaluation', {}).get(key, 0)
                for row in rows if row.get('solve_requested') is True and
                'geometry_delta' not in row)
            ledger['original_' + key + '_requested'] = requested
            ledger['original_' + key + '_enqueued'] = enqueued
            ledger['original_' + key + '_known_complete'] = completed
            ledger['original_' + key + '_unknown_enqueued'] = max(
                0, enqueued - completed)
            ledger['original_' + key + '_not_enqueued'] = max(
                0, requested - enqueued)
            ledger['original_' + key + '_unresolved_upper_bound'] = (
                missing + max(0, enqueued - completed))
        else:
            requested = sum(requested_work(row).get(key, 0) for row in rows)
            known = sum(known_work(row).get(key, 0) for row in rows)
            ledger['original_' + key + '_requested'] = requested
            ledger['original_' + key + '_known_complete'] = known
            ledger['original_' + key + '_unknown_requested'] = max(
                0, requested - known)
            ledger['original_' + key + '_unresolved_upper_bound'] = max(
                0, sum(row['work_per_evaluation'][key] for row in rows
                       if row.get('solve_requested') is True) - known)
    ledger['fit_calls'] = sum(
        leg['optimizer']['requested'] for leg in ledger['legs'])


def register_intent(ledger, leg_spec, local_index, vector, persist):
    index = len(ledger['original_evaluations']) + 1
    if index > 120:
        raise BoundedStop('global_unique_cap')
    row = {'index': index, 'leg': leg_spec['leg'],
        'local_index': local_index,
        'phase': ('baseline' if local_index == 1 else
                  'start' if local_index == 2 else 'restoration_candidate'),
        'status': 'in_flight_unknown', 'backend': 'timescale_cuda',
        'raw_index': 1, 'intent_persisted': True,
        'solve_requested': False, 'request_receipt_persisted': False,
        'in_flight': False, 'solve_returned': False,
        'raw_saved': False, 'post_return_error': False,
        'neurons': leg_spec['neurons'], 'muscles': leg_spec['muscles'],
        'edges': leg_spec['edges'],
        'work_per_evaluation': leg_spec['work_per_evaluation'],
        'vector_sha256': hashlib.sha256(
            np.asarray(vector, np.float64).tobytes()).hexdigest()}
    ledger['original_evaluations'].append(row)
    try:
        persist()
    except BaseException:
        row.update(status='intent_persistence_failed',
                   intent_persisted=False)
        raise
    return row


def complete_receipt(ledger, leg_ledger, cache, row, persist):
    try:
        if not e136._complete_row(row):
            raise ValueError('E139 completed raw receipt invalid')
        leg_ledger.update(cache_started=cache.started,
            cache_completed=cache.completed, cache_hits=cache.hits,
            selected_cache_index=cache.selected_index)
        persist()
    except BaseException as error:
        row.update(status='failed', error=repr(error),
            traceback=traceback.format_exc(),
            post_return_error=bool(row.get('solve_returned')),
            response_progress=getattr(error, 'response_progress',
                row.get('response_progress')))
        leg_ledger.update(cache_started=cache.started,
            cache_completed=cache.completed - 1,
            selected_cache_index=None)
        try:
            persist()
        except BaseException as secondary:
            error.add_note('Secondary E139 completion ledger: ' +
                           repr(secondary))
        raise


def prepared_inputs(path, fixture, freeze):
    """Validate the sealed five starts and E137 LF package coordinates."""
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    for key in ('motor_ids', 'membership', 'global_edge_ids'):
        if not np.array_equal(values[key], fixture[
                'edge_ids' if key == 'global_edge_ids' else key]):
            raise ValueError('E139 prepared global identity changed: ' + key)
    for spec in freeze['legs']:
        leg, edges, muscles, neurons = (spec[key] for key in
            ('leg', 'edges', 'muscles', 'neurons'))
        prefix = f'leg{leg}_'
        expected = {
            'motor_indices': np.flatnonzero(fixture['leg_motor_mask'][leg]),
            'muscle_indices': np.flatnonzero(fixture['leg_muscle_mask'][leg]),
            'edge_positions': np.flatnonzero(fixture['leg_edge_mask'][leg])}
        expected['edge_ids'] = fixture['edge_ids'][expected['edge_positions']]
        for name, answer in expected.items():
            if (values[prefix + name].shape != answer.shape or
                    not np.array_equal(values[prefix + name], answer)):
                raise ValueError('E139 prepared leg identity changed: ' +
                                 prefix + name)
        if (len(expected['motor_indices']) != neurons or
                len(expected['muscle_indices']) != muscles or
                len(expected['edge_positions']) != edges):
            raise ValueError('E139 prepared leg cardinality changed')
        vector_names = ('delta', 'gain', 'log_tau', 'baseline_delta',
                        'baseline_gain', 'baseline_log_tau')
        for name in vector_names:
            array = values[prefix + name]
            length = edges if name.endswith('delta') else muscles
            if (array.shape != (length,) or array.dtype != np.float64 or
                    not np.isfinite(array).all()):
                raise ValueError('E139 prepared vector changed: ' +
                                 prefix + name)
        for name in ('log_tau', 'baseline_log_tau'):
            if not np.array_equal(values[prefix + name], np.zeros(muscles)):
                raise ValueError('E139 frozen old-timescale start changed')
        bounds = original_bounds(fixture, expected['edge_positions'],
                                 expected['muscle_indices'])
        if (values[prefix + 'bounds'].dtype != np.float64 or
                not np.array_equal(values[prefix + 'bounds'], bounds)):
            raise ValueError('E139 prepared original bounds changed')
        for name, frozen in (('baseline_objective', 'baseline_objective'),
                             ('start_objective', 'start_objective'),
                             ('objective_guard', 'original_objective_guard')):
            raw = values[prefix + name]
            if (raw.shape != () or raw.dtype != np.float64 or
                    float(raw) != spec[frozen]):
                raise ValueError('E139 prepared objective/guard changed')
        for name, length in (('old_parameter_muscle_map', edges + muscles),
                             ('parameter_muscle_map', edges + 2 * muscles)):
            array = values[prefix + name]
            if (array.shape != (length,) or array.dtype.kind not in 'iu' or
                    np.any(array < 0) or np.any(array >= muscles)):
                raise ValueError('E139 prepared parameter map changed')
    lf_expected = {'lf_edge_positions': np.flatnonzero(
        fixture['leg_edge_mask'][0]),
        'lf_muscle_indices': np.flatnonzero(fixture['leg_muscle_mask'][0]),
        'lf_motor_indices': np.flatnonzero(fixture['leg_motor_mask'][0])}
    lf_expected['lf_edge_ids'] = fixture['edge_ids'][
        lf_expected['lf_edge_positions']]
    for key, answer in lf_expected.items():
        if not np.array_equal(values[key], answer):
            raise ValueError('E139 prepared LF identity changed: ' + key)
    for key, length in (('lf_delta', 273), ('lf_gain', 17),
                        ('lf_log_tau', 17)):
        if (values[key].shape != (length,) or
                values[key].dtype != np.float64 or
                not np.isfinite(values[key]).all()):
            raise ValueError('E139 prepared LF vector changed: ' + key)
    return values


def leg_result(spec, rows, cache, leg_ledger):
    """A refuted leg is valid only after two parity replays and one bounded fit."""
    baseline = spec['baseline_objective']
    try:
        complete = bool(2 <= len(rows) <= 24 and
            all(row['leg'] == spec['leg'] and
                row['local_index'] == index and e136._complete_row(row) and
                np.isfinite([row['restoration_value'], row['gradient_norm'],
                    row['objective_over_baseline']]).all() and
                len(row['guard3']) == len(row['guard_ratios']) == 3 and
                np.isfinite(row['guard3']).all() and
                np.isfinite(row['guard_ratios']).all() and
                row['qualifies'] is bool(row['selector_feasible'] and
                    row['true_worst_objective'] / baseline <= .99) and
                np.isclose(row['objective_over_baseline'],
                    row['true_worst_objective'] / baseline, rtol=0, atol=1e-12)
                for index, row in enumerate(rows, 1)) and
            cache.started == cache.completed == len(cache.cache) ==
                len(rows) - 1 and
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
    comparisons = leg_ledger.get('comparison', {})
    first_ok = bool(all(comparisons.get(key, {}).get('accepted') is True and
        comparisons[key].get('actual_sha256') == rows[index]['raw_sha256']
        for key, index in (('baseline', 0), ('start', 1)))) if len(rows) >= 2 else False
    fixed_ok = bool(len(rows) >= 2 and
        np.isclose(rows[0]['true_worst_objective'],
            spec['baseline_objective'], rtol=0, atol=1e-10) and
        np.isclose(rows[1]['true_worst_objective'],
            spec['start_objective'], rtol=0, atol=1e-10) and
        rows[1]['selector_feasible'] is False and
        rows[0]['qualifies'] is False and rows[1]['qualifies'] is False)
    candidates = [row for row in rows[2:] if row.get('qualifies') is True]
    selected = candidates[0] if candidates else None
    stop = leg_ledger.get('stop_reason')
    selection_ok = bool(
        (not candidates and cache.selected_index is None and
         stop != 'first_guarded_selector_feasible') or
        (len(candidates) == 1 and selected is rows[-1] and
         cache.selected_index == selected['local_index'] - 1 and
         stop == 'first_guarded_selector_feasible'))
    budget_ok = bool(0 <= leg_ledger['wall_s'] <= 40)
    valid = (complete and optimizer_ok and first_ok and fixed_ok and
             selection_ok and budget_ok)
    accepted = bool(valid and selected is not None)
    return {'leg': spec['leg'], 'name': spec['name'],
        'status': 'completed_bounded_restoration' if valid else 'invalid_evidence',
        'hypothesis_verdict': ('supported' if accepted else
            'refuted' if valid else 'not_tested'),
        'baseline_objective': baseline,
        'start_objective': (rows[1]['true_worst_objective'] if complete else None),
        'selected_index': selected['index'] if accepted else None,
        'selected_local_index': (selected['local_index'] if accepted else None),
        'selected_objective': (selected['true_worst_objective'] if accepted
                               else None),
        'selected_objective_over_baseline': (
            selected['true_worst_objective'] / baseline if accepted else None),
        'new_leg_proposal_accepted': accepted,
        'original_evaluations': len(rows), 'cache_started': cache.started,
        'cache_completed': cache.completed, 'cache_hits': cache.hits,
        'fit_calls': optimizer.get('requested', 0),
        'wall_s': leg_ledger['wall_s'], 'stop_reason': stop,
        'raw_complete': complete, 'optimizer_accounting_complete': optimizer_ok,
        'first_replays_match': first_ok, 'fixed_start_match': fixed_ok,
        'selection_consistent': selection_ok, 'leg_wall_complete': budget_ok}


def write_package(out, fixture, prepared, ledger, leg_results):
    """Only all five supported legs can join sealed LF into research vectors."""
    if not all(item['new_leg_proposal_accepted'] for item in leg_results):
        raise ValueError('E139 incomplete five-leg parameter package')
    delta = np.empty(1572, np.float64)
    gain = np.empty(96, np.float64)
    log_tau = np.empty(96, np.float64)
    edges_filled = np.zeros(1572, bool)
    muscles_filled = np.zeros(96, bool)

    def place(positions, muscle_indices, values):
        if (np.any(edges_filled[positions]) or
                np.any(muscles_filled[muscle_indices])):
            raise ValueError('E139 overlapping leg parameter positions')
        delta[positions] = values['delta']
        gain[muscle_indices] = values['gain']
        log_tau[muscle_indices] = values['log_tau']
        edges_filled[positions] = True
        muscles_filled[muscle_indices] = True

    with np.load(ROOT / ledger['lf_source_raw_path'],
                 allow_pickle=False) as lf:
        for key in ('delta', 'gain', 'log_tau'):
            if not np.array_equal(lf[key], prepared['lf_' + key]):
                raise ValueError('E139 preserved LF proposal changed')
        place(prepared['lf_edge_positions'],
              prepared['lf_muscle_indices'], lf)
    for item in leg_results:
        leg = item['leg']
        prefix = f'leg{leg}_'
        row = ledger['original_evaluations'][item['selected_index'] - 1]
        with np.load(row['raw_path'], allow_pickle=False) as raw:
            if (not np.array_equal(raw['edge_ids'],
                    prepared[prefix + 'edge_ids']) or
                    not np.array_equal(raw['muscle_indices'],
                        prepared[prefix + 'muscle_indices'])):
                raise ValueError('E139 selected package identity changed')
            place(prepared[prefix + 'edge_positions'],
                  prepared[prefix + 'muscle_indices'], raw)
    if (not edges_filled.all() or not muscles_filled.all() or
            not np.isfinite(delta).all() or not np.isfinite(gain).all() or
            not np.isfinite(log_tau).all()):
        raise ValueError('E139 whole parameter coverage incomplete')
    path = out / 'conditional-six-leg-parameter-package.npz'
    np.savez(path, delta=delta, gain=gain, log_tau=log_tau,
        global_edge_ids=fixture['edge_ids'], motor_ids=fixture['motor_ids'],
        membership=fixture['membership'],
        selected_evaluation_indices=np.array([8] +
            [item['selected_index'] for item in leg_results]),
        candidate_adopted=np.array(False))
    return {'path': str(path), 'sha256': fit.sha256(path),
        'edges': 1572, 'muscles': 96, 'motors': 391,
        'candidate_adopted': False}


def write_output_manifest(out):
    save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e139.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): fit.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file() and
            path.name != 'output-manifest.json'}})


def finalize(out, ledger, freeze, began, primary):
    """Revoke late success; keep original traceback and response progress."""
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
                'E139 internal wall limit exceeded after manifest'))
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
                        'new_parameter_package_accepted': False})
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
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('One-shot E139 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e139.cross-leg-restoration-ledger.v1',
        'pid': os.getpid(), 'status': 'preflight',
        'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
        'original_evaluations': [], 'legs': [], 'fit_calls': 0,
        'raw_bytes': 0, 'body_calls': 0, 'LP_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'whole_candidate_adopted': False}
    save(out / 'advance-ledger.json', ledger)
    freeze = limits = engine = None
    primary = None
    block_began = None
    try:
        freeze, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, review_path, review_sha)
        save(out / 'source-review-snapshot.json', review)
        free = shutil.disk_usage(out).free
        ledger['free_disk_before_original_bytes'] = free
        if free < freeze['minimum_free_disk_bytes']:
            raise OSError('E139 frozen 12 GiB original disk reserve unavailable')
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
        prepared = prepared_inputs(ROOT / freeze['prepared_input_path'],
                                   fixture, freeze)
        geometry = TorchFootGeometry.load(
            ROOT / freeze['geometry_path'], device='cuda')
        torch.cuda.synchronize(geometry.device)
        if geometry.geometry_hash != freeze['geometry_hash']:
            raise ValueError('E139 frozen geometry hash mismatch')
        ledger['geometry_backend'] = type(geometry).__name__
        ledger['geometry_device'] = str(geometry.device)
        ledger['geometry_fixed_upload_bytes'] = sum(
            int(geometry.arrays[key].nbytes) for key in
            ('axes', 'anchors', 'tips', 'reference_angles'))
        ledger['lf_source_raw_path'] = str(ROOT / freeze['lf_source_raw_path'])
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        block_began = time.perf_counter()
        block_deadline = block_began + freeze['fit_block_wall_limit_s']
        internal_deadline = began + freeze['internal_wall_limit_s']
        leg_results = []

        def persist():
            refresh_counts(ledger)
            save(out / 'advance-ledger.json', ledger)

        for spec in freeze['legs']:
            leg = spec['leg']
            leg_began = time.perf_counter()
            leg_deadline = leg_began + freeze['leg_wall_limit_s']
            prefix = f'leg{leg}_'
            leg_ledger = {'leg': leg, 'name': spec['name'],
                'status': 'preparing', 'original_indices': [],
                'comparison': {},
                'optimizer': {'status': 'not_requested',
                    'intent_persisted': False, 'requested': 0,
                    'returned': 0, 'bounded_stop': 0,
                    'failed_unknown': 0}}
            ledger['legs'].append(leg_ledger)
            persist()
            motors = prepared[prefix + 'motor_indices']
            muscles = prepared[prefix + 'muscle_indices']
            edges = prepared[prefix + 'edge_positions']
            bounds = original_bounds(fixture, edges, muscles)
            if not np.array_equal(bounds, prepared[prefix + 'bounds']):
                raise ValueError('E139 prepared bounds changed before solve')
            try:
                engine = tau_module.CudaAdaptationTimescaleEngine(
                    fixture, motors, muscles, edges)
            except BaseException as error:
                leg_ledger['constructor_failure'] = {
                    'error': repr(error),
                    'fixed_upload_progress': getattr(error,
                        'fixed_upload_progress', None),
                    'fixed_upload_by_key': getattr(error,
                        'fixed_upload_by_key', None)}
                try:
                    persist()
                except BaseException as secondary:
                    error.add_note('Secondary E139 constructor ledger: ' +
                                   repr(secondary))
                raise
            leg_ledger['engine_before'] = e136.snapshot(engine.metadata())
            for key, answer in (('motor_indices', motors),
                    ('muscle_indices', muscles),
                    ('edge_ids', prepared[prefix + 'edge_ids']),
                    ('parameter_muscle_map',
                        prepared[prefix + 'parameter_muscle_map'])):
                actual = (tau_module.activation_parameter_map(engine.problem)
                    if key == 'parameter_muscle_map' else engine.problem[key])
                if not np.array_equal(actual, answer):
                    raise ValueError('E139 engine/prepared identity changed: ' +
                                     prefix + key)
            if not np.array_equal(
                    prepared[prefix + 'old_parameter_muscle_map'],
                    prepared[prefix + 'parameter_muscle_map'][
                        :spec['edges'] + spec['muscles']]):
                raise ValueError('E139 old parameter map prefix changed')
            leg_ledger['status'] = 'ready'
            persist()
            baseline = np.r_[prepared[prefix + 'baseline_delta'],
                prepared[prefix + 'baseline_gain'],
                prepared[prefix + 'baseline_log_tau']]
            start = np.r_[prepared[prefix + 'delta'],
                prepared[prefix + 'gain'], prepared[prefix + 'log_tau']]
            if (np.any(baseline < bounds[:, 0]) or
                    np.any(baseline > bounds[:, 1]) or
                    np.any(start < bounds[:, 0]) or
                    np.any(start > bounds[:, 1])):
                raise ValueError('E139 frozen baseline/start outside bounds')

            def evaluate_row(vector, local_index):
                row = ledger['original_evaluations'][-1]
                if (row['leg'] != leg or row['local_index'] != local_index or
                        time.perf_counter() >= internal_deadline - 10):
                    raise TimeoutError('E139 row identity or process reserve')
                index = row['index']
                directory = out / f'evaluation-{index:03d}'

                def evaluator(x, row_index, path, **callbacks):
                    return evaluate_timescale_leg(engine, fixture, geometry,
                        leg, x, row_index, path, **callbacks)

                try:
                    record, _ = e136.evaluate_with_receipts(ledger, row,
                        vector, directory, evaluator, persist,
                        metadata=engine.metadata,
                        maximum_raw_bytes=freeze['maximum_raw_bytes'])
                    with np.load(record['raw_path'],
                                 allow_pickle=False) as raw:
                        restoration = restoration_value_jac(raw,
                            spec['original_objective_guard'],
                            spec['baseline_objective'],
                            spec['edges'], spec['muscles'])
                    if (not np.allclose(restoration['ratios'],
                            record['ratios'], rtol=0, atol=1e-12) or
                            not np.isclose(restoration['original_objective'],
                                record['true_worst_objective'],
                                rtol=0, atol=1e-10)):
                        raise ValueError('E139 saved restoration metric mismatch')
                    row.update(restoration_value=restoration['value'],
                        gradient_norm=restoration['gradient_norm'],
                        startup_negative_count=(
                            restoration['startup_negative_count']),
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
                    if local_index in (1, 2):
                        label = 'baseline' if local_index == 1 else 'start'
                        comparison = compare_replay(record['raw_path'],
                            ROOT / spec[label + '_raw_path'],
                            baseline=local_index == 1,
                            edges=spec['edges'], muscles=spec['muscles'],
                            leg=leg)
                        save(out / f'leg-{leg}-{label}-comparison.json',
                             comparison)
                        leg_ledger['comparison'][label] = comparison
                        persist()
                        if not comparison['accepted']:
                            raise ValueError('E139 ' + label +
                                ' first replay parity failed')
                        if not np.isclose(row['true_worst_objective'],
                                spec[label + '_objective'], rtol=0,
                                atol=1e-10):
                            raise ValueError('E139 ' + label +
                                ' replay original objective changed')
                        if local_index == 2 and row['selector_feasible']:
                            raise ValueError('E139 start unexpectedly selector feasible')
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
                        error.add_note('Secondary E139 evaluation ledger: ' +
                                       repr(secondary))
                    raise

            def start_intent(local_index, vector):
                row = register_intent(ledger, spec, local_index + 1,
                                      vector, persist)
                leg_ledger['original_indices'].append(row['index'])
                persist()

            if (time.perf_counter() >= leg_deadline - 3 or
                    time.perf_counter() >= block_deadline or
                    time.perf_counter() >= internal_deadline - 10):
                raise TimeoutError('E139 baseline replay reserve exhausted')
            baseline_row = register_intent(ledger, spec, 1,
                                           baseline, persist)
            leg_ledger['original_indices'].append(baseline_row['index'])
            persist()
            evaluate_row(baseline, 1)
            cache = LegCache(
                lambda vector, cache_index: evaluate_row(
                    vector, cache_index + 1),
                start, bounds, start_intent,
                lambda c, _data, cache_index: complete_receipt(
                    ledger, leg_ledger, c,
                    ledger['original_evaluations'][
                        leg_ledger['original_indices'][cache_index] - 1],
                    persist),
                max_unique=spec['max_unique_evaluations'] - 1,
                leg_deadline=leg_deadline,
                block_deadline=block_deadline,
                internal_deadline=internal_deadline)
            try:
                solver, stop_reason = run_leg_optimizer(cache, leg_ledger,
                                                        persist)
            finally:
                active = sys.exc_info()[1]
                leg_ledger['wall_s'] = time.perf_counter() - leg_began
                try:
                    leg_ledger['engine_after'] = e136.snapshot(
                        engine.metadata())
                    persist()
                except BaseException as secondary:
                    if active is not None:
                        active.add_note('Secondary E139 leg ledger: ' +
                                        repr(secondary))
                    else:
                        raise
            leg_ledger.update(stop_reason=stop_reason,
                solver_returned=solver is not None)
            rows = [ledger['original_evaluations'][index - 1]
                for index in leg_ledger['original_indices']]
            leg_result_value = leg_result(spec, rows, cache, leg_ledger)
            save(out / f'leg-{leg}-result.json', leg_result_value)
            if leg_result_value['status'] != 'completed_bounded_restoration':
                raise ValueError('E139 incomplete leg original evidence: ' +
                                 str(leg))
            leg_results.append(leg_result_value)
            leg_ledger['status'] = leg_result_value['status']
            leg_ledger['result'] = leg_result_value
            persist()
            engine = None
        ledger['fit_block_wall_s'] = time.perf_counter() - block_began
        refresh_counts(ledger)
        if (ledger['fit_block_wall_s'] > freeze['fit_block_wall_limit_s'] or
                time.perf_counter() >= internal_deadline - 1 or
                ledger['raw_bytes'] > freeze['maximum_raw_bytes'] or
                len(ledger['original_evaluations']) > 120 or
                ledger['fit_calls'] != 5 or
                ledger['original_evaluations_requested'] !=
                    ledger['original_evaluations_returned'] or
                ledger['original_evaluations_requested'] !=
                    ledger['original_evaluations_completed'] or
                ledger['original_evaluations_failed_unknown'] or
                ledger['original_evaluations_post_return_error'] or
                ledger['synthetic_tiny'].get('status') != 'completed' or
                ledger['synthetic_tiny'].get('raw_saved') is not True or
                ledger['synthetic_tiny'].get('solve_returned') is not True):
            raise ValueError('E139 whole original work/budget incomplete')
        all_supported = all(item['new_leg_proposal_accepted']
                            for item in leg_results)
        package = (write_package(out, fixture, prepared, ledger, leg_results)
                   if all_supported else None)
        result = {'schema': 'flylab.m3.e139.cross-leg-restoration-result.v1',
            'status': 'completed_bounded_five_leg_restoration',
            'hypothesis_verdict': 'supported' if all_supported else 'refuted',
            'interpretation': 'cross_leg_conditional_restoration_only',
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
            'raw_bytes': ledger['raw_bytes'],
            'body_calls': 0, 'LP_calls': 0, 'F6_exposed': False,
            'candidate_adopted': False, 'whole_candidate_adopted': False,
            'whole_goal_complete': False,
            'freeze_sha256': freeze_sha,
            'source_review_sha256': review_sha}
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
        refresh_counts(ledger)
        if engine is not None:
            try:
                ledger['last_engine_after'] = e136.snapshot(engine.metadata())
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E139 engine metadata: ' +
                                     repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E139 threadpool restore: ' +
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
