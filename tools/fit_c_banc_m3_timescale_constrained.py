"""E136 one bounded, original LF timescale constrained fit.

Only the parent dispatches the frozen original CUDA process after source review.
"""

# ruff: noqa: BLE001  # preserve primary CUDA, solver, and persistence failures

from __future__ import annotations

import argparse
import copy
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = 'a5d7cd43c0bbb7c79a15c638c8850866efba29c6ee77f90ff17f77f74ce9cfd3'
SOURCE_FILES = ('tools/fit_c_banc_m3_timescale_constrained.py',
                'tests/test_c_banc_timescale_constrained.py')
WORK_PER_EVALUATION = {'conditional_s': 30, 'context_ticks': 120000,
    'scalar_MN_ticks': 8280000, 'edge_stage_ticks': 131040000,
    'derivative_banks': 1, 'decoder_endpoints': 6000,
    'geometry_forward_rows': 5473, 'geometry_backward_rows': 5473}


def save(path, value):
    fit.write_json(path, snapshot(value))


def snapshot(value):
    return copy.deepcopy(fit.jsonable(value))


def checked_path(relative, digest):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if (name.is_absolute() or '..' in name.parts or
            not path.is_relative_to(ROOT.resolve()) or
            not isinstance(digest, str) or len(digest) != 64 or
            fit.sha256(path) != digest):
        raise ValueError('E136 frozen path/SHA mismatch: ' + str(relative))
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or fit.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E136 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    expected = {'schema': 'flylab.m3.e136.timescale-constrained-fit-freeze.v1',
        'leg': 0, 'start_evaluation_index': 39, 'solver': 'SLSQP',
        'solver_options': {'maxiter': 100, 'ftol': 1e-9, 'disp': False},
        'max_unique_evaluations': 64, 'fit_wall_limit_s': 95,
        'leg_reserve_s': 3, 'process_reserve_s': 5,
        'internal_wall_limit_s': 108, 'external_wall_limit_s': 116,
        'supervisor_limit_s': 120, 'original_processes': 1, 'fit_calls': 1,
        'retries': 0, 'proposal_objective_ratio_limit': .99,
        'optimizer_cap': fit.OPTIMIZER_CAP,
        'selector_cap': fit.SELECTOR_CAP, 'maximum_raw_bytes': 4294967296,
        'full391_batches': 0, 'full_BANC_calls': 0, 'body_calls': 0,
        'F6_exposed': False, 'whole_candidate_adoption': False,
        'active_UI_changed': False}
    dimensions = {'contexts': 3, 'neurons': 69, 'muscles': 17,
        'edges': 273, 'parameters': 307, 'LP_variables': 308,
        'constraints': 4761, 'stages': 160000, 'ticks': 40000,
        'samples': 2001}
    maximum = {key: 64 * value for key, value in WORK_PER_EVALUATION.items()}
    if (any(freeze.get(key) != value for key, value in expected.items()) or
            freeze.get('dimensions') != dimensions or
            freeze.get('maximum_original_work') != maximum or
            freeze.get('log_tau_bounds') != [-math.log(3), math.log(10)] or
            freeze.get('tiny') != {'batches': 1, 'context_ticks': 20,
                'conditional_s': .005, 'derivative_banks': 1} or
            len(freeze.get('input_sha256', {})) != 18 or
            len(freeze.get('reused_source_sha256', {})) != 4 or
            freeze.get('design_sha256') !=
                fit.sha256(ROOT / freeze.get('design_path', ''))):
        raise ValueError('E136 fixed fit contract changed')
    for key in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[key].items():
            checked_path(relative, digest)
    for key in ('e128_freeze_path', 'fixture_path', 'geometry_path',
                'start_raw_path', 'start_proposal_path', 'E135_base_path'):
        if freeze.get(key) not in freeze['input_sha256']:
            raise ValueError('E136 required original input is not frozen: ' + key)
    if fit.sha256(review_path) != review_sha:
        raise ValueError('E136 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_timescale_constrained_fit')
            is not True or review.get('blocking_defects') != [] or
            review.get('freeze_sha256') != freeze_sha or
            set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E136 two-source review required')
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
        raise ValueError('E136/E128 frozen fixture or context mismatch')
    for key in ('E132_final_review_path', 'E134_final_review_path',
                'E135_final_review_path'):
        prior = json.loads((ROOT / freeze[key]).read_text(encoding='utf-8'))
        if prior.get('validity') != 'valid' or prior.get('run_status') != 'completed':
            raise ValueError('E136 predecessor review is not valid: ' + key)
    e134_source = json.loads((ROOT / freeze['E134_source_review_path']).read_text(
        encoding='utf-8'))
    if (e134_source.get(
            'accepted_for_single_original_adaptation_timescale_verification')
            is not True or e134_source.get('blocking_defects') != []):
        raise ValueError('E136 E134 calculator source review invalid')
    return freeze, e128, review


def start_vector(freeze):
    with (np.load(ROOT / freeze['start_raw_path'], allow_pickle=False) as raw,
          np.load(ROOT / freeze['start_proposal_path'], allow_pickle=False) as proposal):
        if (int(proposal['original_evaluation_index']) != 39 or
                str(proposal['source_raw_sha256']) !=
                    freeze['input_sha256'][freeze['start_raw_path']] or
                any(raw[key].dtype != np.float64 or
                    proposal[key].dtype != np.float64 or
                    not np.array_equal(raw[key], proposal[key])
                    for key in ('delta', 'gain'))):
            raise ValueError('E136 E132 selected39/proposal mismatch')
        vector = np.r_[proposal['delta'], proposal['gain'], np.zeros(17)]
    if vector.shape != (307,) or not np.isfinite(vector).all():
        raise ValueError('E136 warm start must be finite LF307')
    return vector


def original_bounds(fixture, edges):
    old = np.vstack((np.asarray(fixture['delta_bounds'][edges], np.float64),
        np.tile(np.array([[0., 8.]]), (17, 1))))
    if (old.shape != (290, 2) or not np.isfinite(old).all() or
            np.any(old[:, 0] > old[:, 1])):
        raise ValueError('E136 frozen edge/gain bounds changed')
    bounds = np.vstack((old, np.tile(np.array([[-math.log(3),
        math.log(10)]]), (17, 1))))
    if bounds.shape != (307, 2):
        raise ValueError('E136 original bounds dimensions changed')
    return bounds


def initial_jacobians(b_initial, q_coefficient, ltm_coefficient,
                      parameter_muscle_map, ltm_local):
    """The B column belongs to one local muscle, including each tau column."""
    b = np.asarray(b_initial, np.float64)
    q_coeff = np.asarray(q_coefficient, np.float64)
    ltm_coeff = np.asarray(ltm_coefficient, np.float64)
    mapping = np.asarray(parameter_muscle_map)
    if (b.ndim != 3 or b.shape[:2] != (3, 99) or
            q_coeff.shape != (3, 99, 7, 17) or
            ltm_coeff.shape != (3, 99) or
            mapping.shape != (b.shape[-1],) or
            mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or
            np.any(mapping >= 17) or not 0 <= ltm_local < 17):
        raise ValueError('E136 initial Jacobian shape/muscle mapping changed')
    q_jac = b[:, :, None, :] * q_coeff[:, :, :, mapping]
    ltm_jac = b * ltm_coeff[:, :, None] * (mapping == ltm_local)[None, None, :]
    return q_jac, ltm_jac


def saved_full_b_loss_jac(b, cotangent, mapping):
    """Independent CPU identity for generated tests and first-replay checks."""
    b = np.asarray(b, np.float64)
    cotangent = np.asarray(cotangent, np.float64)
    mapping = np.asarray(mapping)
    if (b.ndim != 3 or cotangent.shape != (*b.shape[:2], 17) or
            mapping.shape != (b.shape[-1],) or
            mapping.dtype.kind not in 'iu' or np.any(mapping < 0) or
            np.any(mapping >= 17)):
        raise ValueError('E136 full B/cotangent mapping mismatch')
    return np.einsum('ctp,ctp->cp', b, cotangent[:, :, mapping])


def objective_and_jac(data, t):
    delta = np.asarray(data['delta'], np.float64)
    gain = np.asarray(data['gain'], np.float64)
    if delta.shape != (273,) or gain.shape != (17,):
        raise ValueError('E136 objective LF dimensions changed')
    prior = 6e-5 * np.sum(delta * delta) / 1572 + 1e-5 * np.sum(gain * gain) / 96
    jac = np.r_[1.2e-4 * delta / 1572, 2e-5 * gain / 96,
                np.zeros(17), 1.]
    return float(t + prior), jac, float(prior)


def solve_one_fit(cache, bounds, x0, *, solver=minimize):
    def objective(x):
        data, t = cache.get(x)
        return objective_and_jac(data, t)[0]

    def objective_jac(x):
        data, t = cache.get(x)
        return objective_and_jac(data, t)[1]

    def constraints(x):
        data, t = cache.get(x)
        return fit.constraint_values(data['losses'], data['loss_jac'],
            data['q_error'], data['ltm_error'], data['q_jac'],
            data['ltm_jac'], t)[0]

    def constraint_jac(x):
        data, t = cache.get(x)
        return fit.constraint_values(data['losses'], data['loss_jac'],
            data['q_error'], data['ltm_error'], data['q_jac'],
            data['ltm_jac'], t)[1]

    return solver(objective, x0, method='SLSQP', jac=objective_jac,
        bounds=[*map(tuple, bounds), (None, None)],
        constraints=[{'type': 'ineq', 'fun': constraints, 'jac': constraint_jac}],
        options={'maxiter': 100, 'ftol': 1e-9, 'disp': False})


def compare_first_raw(actual_path, old_path, timescale_base_path):
    """Compare the actual zero-tau replay after its full raw is saved."""
    report = {'accepted': False, 'fields': {},
        'actual_sha256': fit.sha256(actual_path),
        'old_sha256': fit.sha256(old_path),
        'timescale_base_sha256': fit.sha256(timescale_base_path)}
    with (np.load(actual_path, allow_pickle=False) as actual,
          np.load(old_path, allow_pickle=False) as old,
          np.load(timescale_base_path, allow_pickle=False) as base):
        checks = [(key, key, 'exact' if key in ('delta', 'gain', 'edge_ids',
            'motor_indices', 'muscle_indices', 'branch_bits') else 'response')
            for key in (*fit.FIELDS, 'branch_bits', 'delta', 'gain',
                        'edge_ids', 'motor_indices', 'muscle_indices')]
        checks += [('parameter_muscle_map', 'parameter_muscle_map', 'map'),
            ('B_initial99', 'B_initial99', 'old_B'),
            ('losses', 'losses', 'metric'), ('parts', 'parts', 'metric'),
            ('ratios', 'ratios', 'metric'),
            ('loss_jac', 'loss_jac', 'old_loss'),
            ('q_error', 'q_error', 'metric'),
            ('ltm_error', 'ltm_error', 'metric'),
            ('q_coefficient', 'q_coefficient', 'metric'),
            ('ltm_coefficient', 'ltm_coefficient', 'metric')]
        for actual_key, old_key, kind in checks:
            if actual_key not in actual or old_key not in old:
                report['fields'][actual_key] = {'accepted': False, 'reason': 'missing'}
                continue
            a, b = actual[actual_key], old[old_key]
            if kind == 'map':
                a = a[:290]
            elif kind == 'old_B':
                a = a[:, :, :290]
            elif kind == 'old_loss':
                a = a[:, :290]
            elif actual_key == 'command':
                a = a[:, :, :7]
                b = b[:, :, :7]
            elif actual_key == 'tendon':
                a = a[:, :, :1]
                b = b[:, :, :1]
            if a.shape != b.shape or a.dtype != b.dtype:
                report['fields'][actual_key] = {'accepted': False,
                    'reason': 'shape_or_dtype', 'actual_shape': a.shape,
                    'reference_shape': b.shape}
                continue
            limit = (0 if kind in ('exact', 'map') else
                1e-8 if kind in ('response', 'old_B') else 1e-10)
            diff = (float(np.max(np.abs(a - b))) if a.size and limit
                    else 0.)
            ok = (np.array_equal(a, b) if limit == 0 else
                  np.isfinite(a).all() and np.isfinite(b).all() and diff <= limit)
            report['fields'][actual_key] = {'accepted': bool(ok),
                'max_abs_difference': diff, 'limit': limit}
        for name, a, b in (
                ('log_tau_zero', actual['log_tau'], np.zeros(17)),
                ('tau_B_full', actual['B'][:, :, 290:], base['tau_B']),
                ('tau_B_initial99', actual['B_initial99'][:, :, 290:],
                 base['tau_B'][:, 1:100]),
                ('tau_loss_jac', actual['loss_jac'][:, 290:],
                 base['tau_loss_jac'])):
            limit = 0 if name == 'log_tau_zero' else 1e-10
            diff = float(np.max(np.abs(a - b))) if a.size and a.shape == b.shape else None
            ok = bool(a.shape == b.shape and a.dtype == b.dtype and
                np.isfinite(a).all() and np.isfinite(b).all() and
                diff is not None and diff <= limit)
            report['fields'][name] = {'accepted': ok,
                'max_abs_difference': diff, 'limit': limit}
        report['accepted'] = all(x['accepted'] for x in report['fields'].values())
    return report


def cuda_cotangent_product(engine, device_result, cotangent):
    """Contract all E134 B columns with the matching local muscle on CUDA."""
    import cupy as cp
    import torch

    expected = (engine.contexts, engine.samples, engine.muscles)
    if (not isinstance(cotangent, torch.Tensor) or
            cotangent.shape != expected or cotangent.dtype != torch.float64 or
            cotangent.device.type != 'cuda' or
            device_result['B'].shape !=
                (engine.contexts, engine.samples, engine.edges + 2 * engine.muscles)):
        raise ValueError('E136 CUDA B/cotangent shape or dtype mismatch')
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


def evaluate_timescale_leg(engine, fixture, geometry, vector, index, out,
                           *, on_solve_entry=None, on_solve_return=None,
                           on_raw_saved=None):
    """One actual LF solve, all raw fields, then the original nonlinear losses."""
    import torch

    engine.geometry_backend = type(geometry).__name__
    engine.geometry_device = str(getattr(geometry, 'device',
                                         'synthetic_cuda'))
    p = engine.problem
    edge_count, muscle_count = engine.edges, engine.muscles
    if (edge_count, muscle_count) != (273, 17):
        raise ValueError('E136 LF engine cardinality changed')
    x = np.asarray(vector, np.float64)
    if x.shape != (307,) or not np.isfinite(x).all():
        raise ValueError('E136 finite delta/gain/logtau vector required')
    delta = np.ascontiguousarray(x[:edge_count])
    gain = np.ascontiguousarray(x[edge_count:edge_count + muscle_count])
    log_tau = np.ascontiguousarray(x[edge_count + muscle_count:])
    mapping = tau_module.activation_parameter_map(p)
    muscles = p['muscle_indices']
    starts = np.flatnonzero(fixture['startup_mask'])
    if (mapping.shape != (307,) or len(starts) != 99 or
            not np.array_equal(starts, np.arange(1, 100))):
        raise ValueError('E136 startup/map contract changed')
    raw = {'delta': delta, 'gain': gain, 'log_tau': log_tau,
        'edge_ids': p['edge_ids'], 'motor_indices': p['motor_indices'],
        'muscle_indices': muscles, 'parameter_muscle_map': mapping}
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
        start_indices = fit.torch_indices(engine, starts, activation_all.device,
                                          'startup_indices')
        losses = np.empty(3, np.float64)
        parts = np.empty((3, 4), np.float64)
        q_error = np.empty((3, 99, 7), np.float64)
        ltm_error = np.empty((3, 99), np.float64)
        q_coefficient = np.empty((3, 99, 7, 17), np.float64)
        ltm_coefficient = np.empty((3, 99), np.float64)
        ltm_local = int(np.flatnonzero(muscles == fixture['ltm'][0])[0])
        loss_tensors = []
        moment = .02 * fixture['moment'][:7, muscles]
        for ci in range(3):
            selected = np.flatnonzero(fixture['fit_time_mask'][ci])
            if len(selected) < 2 or not np.all(np.diff(selected) == 1):
                raise ValueError('E136 contiguous fit mask required')
            activation = activation_all[ci]
            selected_indices = fit.torch_indices(engine, selected,
                activation.device, 'fit_indices')
            command, tendon = fit.torch_leg_loss(engine, activation,
                fixture, 0, geometry, muscles)
            engine.geometry_cost['forward_rows_requested'] += len(selected)
            foot = geometry.leg(command[selected_indices], 0)
            engine.geometry_cost['forward_rows_enqueued'] += len(selected)
            raw[f'context{ci}_fit_feet'] = fit.torch_host_array(engine, foot,
                f'context{ci}_fit_feet')
            # The host materialization synchronized this context's foot rows.
            engine.geometry_cost['forward_rows_completed'] += len(selected)
            velocity = (foot[1:] - foot[:-1]) / .005
            device = activation.device

            def to(value, label, target_device=device):
                return fit.torch_input_array(engine, value, target_device,
                                             label)

            q_target = to(fixture['own_target'][ci, selected, :7], 'q_target')
            foot_target = to(fixture['target_feet'][ci, selected, 0],
                             'foot_target')
            v_target = (foot_target[1:] - foot_target[:-1]) / .005
            q_loss = ((command[selected_indices] - q_target) /
                to(fixture['joint_scale'][:7], 'joint_scale')).square().mean()
            foot_loss = ((foot - foot_target) /
                to(fixture['foot_scale'][0], 'foot_scale')).square().mean()
            vel_loss = ((velocity - v_target) /
                to(fixture['velocity_scale'][0], 'velocity_scale')).square().mean()
            ltm_loss = ((tendon[selected_indices] - .2) / .2).square().mean()
            values = torch.stack((q_loss, foot_loss, vel_loss, ltm_loss))
            loss_tensors.append(values.sum())
            parts[ci] = fit.torch_host_array(engine, values, 'loss_parts')
            losses[ci] = float(fit.torch_host_array(engine, values.sum(),
                                                    'loss_scalar'))
            q_error[ci] = fit.torch_host_array(engine,
                command[start_indices] - to(
                    fixture['startup_target_command'][:, :7],
                    'startup_target_command'), 'initial_q_error')
            ltm_error[ci] = fit.torch_host_array(engine,
                tendon[start_indices] - to(
                    fixture['startup_target_tendon'][:, 0],
                    'startup_target_tendon'), 'initial_ltm_error')
            q_coefficient[ci] = np.broadcast_to(moment[None], (99, 7, 17))
            initial_activation = fit.torch_host_array(engine,
                activation[start_indices, ltm_local], 'initial_ltm_activation')
            ltm_coefficient[ci] = .02 * (1 - np.tanh(
                initial_activation / 10) ** 2)
        engine.geometry_cost['backward_rows_requested'] += int(
            fixture['fit_time_mask'].sum())
        cotangent_t = torch.autograd.grad(sum(loss_tensors),
                                           activation_all)[0]
        engine.geometry_cost['backward_rows_enqueued'] += int(
            fixture['fit_time_mask'].sum())
        torch.cuda.synchronize(activation_all.device)
        engine.geometry_cost['backward_rows_completed'] += int(
            fixture['fit_time_mask'].sum())
        loss_jac = cuda_cotangent_product(engine, d, cotangent_t)
        cotangent = fit.torch_host_array(engine, cotangent_t,
                                         'loss_cotangent_raw')
        q_jac, ltm_jac = initial_jacobians(raw['B_initial99'], q_coefficient,
            ltm_coefficient, mapping, ltm_local)
        _, _, ratios = fit.constraint_values(losses, loss_jac, q_error,
            ltm_error, q_jac, ltm_jac, float(max(losses) + .01))
        raw.update(losses=losses, parts=parts, loss_jac=loss_jac,
            q_error=q_error, ltm_error=ltm_error, q_coefficient=q_coefficient,
            ltm_coefficient=ltm_coefficient, q_jac=q_jac, ltm_jac=ltm_jac,
            cotangent=cotangent, ratios=ratios,
            transfer_cost_json=np.array(json.dumps(snapshot({
                'host_transfer': engine.host_transfer,
                'host_transfer_by_label': engine.host_transfer_by_label,
                'torch_host_transfer': engine.torch_host_transfer,
                'torch_host_transfer_by_label': engine.torch_host_transfer_by_label,
                'cotangent_upload': engine.cotangent_upload,
                'index_upload': engine.index_upload,
                'index_upload_by_label': engine.index_upload_by_label,
                'torch_input_transfer': engine.torch_input_transfer,
                'torch_input_transfer_by_label': engine.torch_input_transfer_by_label,
                'torch_cupy_shared_bytes': engine.torch_cupy_shared_bytes,
                'torch_contiguous_copy_bytes': engine.torch_contiguous_copy_bytes,
                'geometry_cost': engine.metadata()['geometry_cost']}))))
        path = out / 'leg-0-evaluation-001-raw.npz'
        np.savez(path, **raw)
        digest = fit.sha256(path)
        if on_raw_saved is not None:
            on_raw_saved(path, digest)
        if any(not np.isfinite(value).all() for value in
               (raw['B'], losses, parts, loss_jac, q_error, ltm_error,
                q_jac, ltm_jac, ratios, cotangent)):
            raise ValueError('Nonfinite E136 saved original raw: ' + str(path))
        _, _, prior = objective_and_jac(raw, max(losses))
        return {'index': index, 'delta': delta, 'gain': gain,
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
                    error.add_note('Secondary E136 response rescue: ' +
                                   str(failure))
            except BaseException as rescue_error:
                error.add_note('Secondary E136 rescue protocol: ' +
                               repr(rescue_error))
        rescued.update({key: value for key, value in raw.items()
                        if isinstance(value, np.ndarray)})
        rescued.update(getattr(error, 'partial_response', {}))
        try:
            rescued['failure_type'] = np.array(type(error).__name__)
            rescued['failure_message'] = np.array(str(error))
            partial = out / 'leg-0-evaluation-001-partial-raw.npz'
            np.savez(partial, **rescued)
            error.partial_raw_path = str(partial)
        except BaseException as save_error:
            error.add_note('Secondary E136 partial raw save: ' +
                           repr(save_error))
        raise


def run_tiny_setup(out, ledger):
    """One unrelated 1-MN, 20-tick E134 setup; no recorded-context bytes."""
    tiny = fit.synthetic_tiny_setup_problem()
    row = {'status': 'intent', 'synthetic_only': True,
        'requested_context_ticks': 20, 'requested_conditional_s': .005,
        'requested_derivative_banks': 1, 'solve_requested': False,
        'request_receipt_persisted': False, 'in_flight': False,
        'solve_returned': False, 'raw_saved': False}
    ledger['synthetic_tiny'] = row
    save(out / 'advance-ledger.json', ledger)
    engine = None
    try:
        engine = tau_module.CudaAdaptationTimescaleEngine(tiny)
        row['engine_before'] = snapshot(engine.metadata())
        row.update(status='solve_in_flight_unknown', solve_requested=True,
                   request_receipt_persisted=True, in_flight=True)
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException:
            row.update(status='request_receipt_failed', solve_requested=False,
                       request_receipt_persisted=False, in_flight=False)
            raise
        d, h, timing = engine.solve(np.zeros(1), np.zeros(1),
            np.zeros(1), derivative=True, host=True)
        row.update(solve_returned=True, in_flight=False,
                   response_progress=timing['progress'],
                   solve_wall_s=timing['wall_s'])
        save(out / 'advance-ledger.json', ledger)
        h['B'] = engine.host_copy(d['B'], label='tiny_full_B')
        path = out / 'cuda-tiny-setup-raw.npz'
        np.savez(path, synthetic_stage_current=tiny['stage_current'],
                 synthetic_source_rate=tiny['source_rate'], **h)
        row.update(raw_saved=True, raw_path=str(path),
                   raw_sha256=fit.sha256(path),
                   engine_after=snapshot(engine.metadata()))
        save(out / 'advance-ledger.json', ledger)
        if (timing['context_rk4_ticks'] != 20 or
                not timing['progress']['derivative_bank_complete'] or
                any(not np.isfinite(h[key]).all() for key in
                    (*tau_module.FIELDS, 'B'))):
            raise ValueError('E136 synthetic tiny CUDA setup incomplete')
        row['status'] = 'completed'
        save(out / 'advance-ledger.json', ledger)
        return row
    except BaseException as error:
        row.update(status='failed', error=repr(error), in_flight=False,
            response_progress=getattr(error, 'response_progress',
                                      row.get('response_progress')))
        if engine is not None:
            try:
                row['engine_after'] = snapshot(engine.metadata())
            except BaseException as secondary:
                error.add_note('Secondary E136 tiny metadata: ' + repr(secondary))
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as secondary:
            error.add_note('Secondary E136 tiny ledger: ' + repr(secondary))
        raise


GEOMETRY_FIELDS = tuple(f'{direction}_rows_{state}'
    for direction in ('forward', 'backward')
    for state in ('requested', 'enqueued', 'completed'))


def geometry_delta(before, after):
    """Per-evaluation difference of one engine's cumulative geometry counters."""
    old = before.get('geometry_cost', {})
    new = after.get('geometry_cost', {})
    result = {}
    for key in GEOMETRY_FIELDS:
        a, b = old.get(key, 0), new.get(key, 0)
        if (type(a) is not int or type(b) is not int or a < 0 or b < a):
            raise ValueError('E136 cumulative geometry counter regressed: ' + key)
        result[key] = b - a
    for direction in ('forward', 'backward'):
        requested = result[f'{direction}_rows_requested']
        enqueued = result[f'{direction}_rows_enqueued']
        completed = result[f'{direction}_rows_completed']
        if not 0 <= completed <= enqueued <= requested:
            raise ValueError('E136 geometry counter order invalid: ' + direction)
        result[f'{direction}_rows_not_enqueued'] = requested - enqueued
        result[f'{direction}_rows_unknown_enqueued'] = enqueued - completed
    return result


def _known_neural_work(row):
    """Only synchronized E134 progress is known work, including failed solves."""
    if row.get('solve_requested') is not True:
        return {}
    progress = row.get('response_progress') or {}
    if not isinstance(progress, dict):
        return {}
    known = {}
    forward = progress.get('forward_completion_class') == 'known_complete'
    decoder = progress.get('decoder_completion_class') == 'known_complete'
    bank = (progress.get('bank_completion_class') == 'known_complete' and
            progress.get('derivative_bank_complete') is True)
    fields = {'context_ticks': ('completed_context_rk4_ticks', forward),
        'conditional_s': ('completed_conditional_model_s', forward),
        'decoder_endpoints': ('completed_decoder_endpoint_ticks', decoder),
        'edge_stage_ticks': ('completed_bank_edge_stage_ticks', bank)}
    for key, (field, complete) in fields.items():
        value = progress.get(field)
        limit = WORK_PER_EVALUATION[key]
        if (complete and isinstance(value, (int, float)) and
                np.isfinite(value) and 0 <= value <= limit):
            known[key] = value
    if 'context_ticks' in known:
        known['scalar_MN_ticks'] = known['context_ticks'] * 69
    if bank:
        known['derivative_banks'] = 1
    return known


def _requested_neural_work(row):
    """Count stages entered by E134, rather than all stages in the plan."""
    if row.get('solve_requested') is not True:
        return {}
    progress = row.get('response_progress')
    if not isinstance(progress, dict):
        return {}
    events = progress.get('kernel_events')
    if isinstance(events, list):
        names = {event.get('name') for event in events
                 if isinstance(event, dict) and event.get('requested') is True}
        forward, decoder, bank = (name in names for name in
                                  ('forward', 'decoder', 'bank'))
    else:
        forward = progress.get('forward_completion_class') not in (
            None, 'not_enqueued')
        decoder = progress.get('decoder_completion_class') not in (
            None, 'not_enqueued')
        bank = progress.get('bank_completion_class') not in (
            None, 'not_enqueued')
    requested = {}
    if forward:
        for key in ('context_ticks', 'conditional_s', 'scalar_MN_ticks'):
            requested[key] = WORK_PER_EVALUATION[key]
    if decoder:
        requested['decoder_endpoints'] = WORK_PER_EVALUATION[
            'decoder_endpoints']
    if bank:
        requested['edge_stage_ticks'] = WORK_PER_EVALUATION[
            'edge_stage_ticks']
        requested['derivative_banks'] = 1
    return requested


def refresh_counts(ledger):
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
    for key, maximum in WORK_PER_EVALUATION.items():
        if key.startswith('geometry_'):
            direction = key.split('_')[1]
            counter = f'{direction}_rows'
            requested = sum(row.get('geometry_delta', {}).get(
                counter + '_requested', 0) for row in rows)
            enqueued = sum(row.get('geometry_delta', {}).get(
                counter + '_enqueued', 0) for row in rows)
            completed = sum(row.get('geometry_delta', {}).get(
                counter + '_completed', 0) for row in rows)
            ledger['original_' + key + '_requested'] = requested
            ledger['original_' + key + '_enqueued'] = enqueued
            ledger['original_' + key + '_known_complete'] = completed
            ledger['original_' + key + '_unknown_enqueued'] = max(
                0, enqueued - completed)
            ledger['original_' + key + '_not_enqueued'] = max(
                0, requested - enqueued)
            missing = sum(row.get('solve_requested') is True and
                'geometry_delta' not in row for row in rows)
            ledger['original_' + key + '_unresolved_upper_bound'] = (
                missing * maximum + max(0, enqueued - completed))
        else:
            requested = sum(_requested_neural_work(row).get(key, 0)
                            for row in rows)
            known = sum(_known_neural_work(row).get(key, 0)
                        for row in rows)
            ledger['original_' + key + '_requested'] = requested
            ledger['original_' + key + '_known_complete'] = known
            ledger['original_' + key + '_unknown_requested'] = max(
                0, requested - known)
            ledger['original_' + key + '_unresolved_upper_bound'] = max(
                0, ledger['original_evaluations_requested'] * maximum - known)


def register_intent(ledger, index, vector, persist):
    """Persist an uncertain invocation before any original solve is entered."""
    row = {'index': index, 'phase': f'original_{index}_candidate',
        'status': 'in_flight_unknown', 'backend': 'timescale_cuda',
        'raw_index': 1, 'intent_persisted': True,
        'solve_requested': False, 'request_receipt_persisted': False,
        'in_flight': False, 'solve_returned': False,
        'raw_saved': False, 'post_return_error': False,
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


def evaluate_with_receipts(ledger, row, vector, directory, evaluator,
                           persist, *, first_compare=None, metadata=None,
                           maximum_raw_bytes=4294967296):
    """Own durable intent/return/raw transitions around an injected evaluator."""
    started = time.perf_counter()
    before_metadata = None
    try:
        if metadata is not None:
            before_metadata = snapshot(metadata())
            row['engine_before'] = before_metadata
        directory.mkdir()

        def on_solve_entry():
            row.update(status='solve_in_flight_unknown',
                solve_requested=True, request_receipt_persisted=True,
                in_flight=True)
            try:
                persist()
            except BaseException:
                row.update(status='request_receipt_failed',
                    solve_requested=False, request_receipt_persisted=False,
                    in_flight=False)
                raise

        def on_solve_return(timing, response):
            row.update(status='returned_before_raw', solve_returned=True,
                in_flight=False,
                response_progress=timing['progress'],
                solve_wall_s=timing['wall_s'],
                returned_response_keys=sorted(response))
            persist()

        def on_raw_saved(path, digest):
            row.update(status='raw_saved_before_metrics', raw_saved=True,
                raw_path=str(path), raw_sha256=digest,
                raw_bytes=path.stat().st_size)
            ledger['raw_bytes'] += row['raw_bytes']
            persist()
            if ledger['raw_bytes'] > maximum_raw_bytes:
                raise ValueError('E136 frozen 4 GiB original raw cap exceeded')

        record = evaluator(vector, row['index'], directory,
            on_solve_entry=on_solve_entry,
            on_solve_return=on_solve_return,
            on_raw_saved=on_raw_saved)
        row.update(status='raw_saved', losses=record['losses'],
            parts=record['parts'], ratios=record['ratios'],
            prior=record['prior'],
            true_worst_objective=record['true_worst_objective'],
            max_ratio=float(np.max(record['ratios'])),
            selector_feasible=fit.selector_feasible(record['ratios']),
            evaluate_wall_s=record['evaluate_wall_s'])
        persist()
        comparison = None
        if first_compare is not None:
            comparison = first_compare(record)
            ledger['first_reference_comparison'] = comparison['accepted']
            persist()
            if not comparison['accepted']:
                raise ValueError('E136 first replay E132/E135 parity failed')
            if not row['selector_feasible']:
                raise ValueError('E136 initial selector feasibility failed')
        return record, comparison
    except BaseException as error:
        row.update(status='failed', error=repr(error), in_flight=False,
            traceback=traceback.format_exc(),
            post_return_error=bool(row.get('solve_returned')),
            response_progress=getattr(error, 'response_progress',
                row.get('response_progress')),
            partial_raw_path=getattr(error, 'partial_raw_path', None))
        raise
    finally:
        active = sys.exc_info()[1]
        row['evaluate_wall_s'] = time.perf_counter() - started
        if metadata is not None:
            try:
                after_metadata = snapshot(metadata())
                row['engine_metadata'] = after_metadata
                if before_metadata is not None:
                    row['geometry_delta'] = geometry_delta(
                        before_metadata, after_metadata)
            except BaseException as secondary:
                if active is not None:
                    active.add_note('Secondary E136 engine metadata: ' +
                                    repr(secondary))
                    row['geometry_delta_error'] = repr(secondary)
                else:
                    raise
        try:
            persist()
        except BaseException as secondary:
            if active is not None:
                active.add_note('Secondary E136 evaluation ledger: ' +
                                repr(secondary))
            else:
                raise


def bounded_optimizer(cache, start, bounds, ledger, persist,
                      *, solver=solve_one_fit, clock=time.perf_counter):
    """Replay exactly once, then request a single SLSQP after parity succeeds."""
    accounting = ledger['optimizer']
    began = clock()
    result = None
    reason = 'scipy_finished'
    try:
        initial, _ = cache.get(np.r_[start, 0.])
        if not fit.selector_feasible(initial['ratios']):
            raise ValueError('E136 first original replay is not selector-feasible')
        x0 = np.r_[start, float(np.max(initial['losses']) + .01)]
        accounting.update(status='intent', intent_persisted=True)
        try:
            persist()
        except BaseException:
            accounting.update(status='not_requested', intent_persisted=False)
            raise
        accounting.update(status='in_flight_unknown', requested=1)
        accounting['request_receipt_persisted'] = True
        accounting['in_flight'] = True
        ledger['fit_calls'] = 1
        try:
            persist()
        except BaseException:
            accounting.update(status='request_receipt_failed', requested=0,
                request_receipt_persisted=False, in_flight=False)
            ledger['fit_calls'] = 0
            raise
        try:
            result = solver(cache, bounds, x0)
        except fit.FitLimit as error:
            accounting.update(status='bounded_stop', bounded_stop=1,
                in_flight=False,
                stop_reason=str(error))
            reason = str(error)
            persist()
        except BaseException as error:
            accounting.update(status='failed_unknown', failed_unknown=1,
                in_flight=False,
                error=repr(error))
            raise
        else:
            accounting.update(status='returned', returned=1,
                in_flight=False,
                solver={'success': bool(result.success),
                    'status': int(result.status), 'message': str(result.message),
                    'nit': int(result.nit), 'nfev': int(result.nfev),
                    'fun': float(result.fun)})
            persist()
    finally:
        accounting['elapsed_wall_s'] = clock() - began
        ledger['fit_calls'] = accounting['requested']
        ledger['cache_started'] = cache.started
        ledger['cache_completed'] = cache.completed
        ledger['cache_hits'] = cache.hits
    return result, reason


def _complete_row(row):
    try:
        losses = np.asarray(row['losses'], np.float64)
        ratios = np.asarray(row['ratios'], np.float64)
        parts = np.asarray(row['parts'], np.float64)
        return bool(row['status'] == 'raw_saved' and row['raw_saved'] is True
            and row['solve_requested'] is True and row['solve_returned'] is True
            and row['index'] >= 1 and losses.shape == (3,) and
            ratios.shape == (3, 4) and parts.shape == (3, 4) and
            np.isfinite(losses).all() and np.isfinite(ratios).all() and
            np.isfinite(parts).all() and
            np.isfinite(row['prior']) and row['prior'] >= 0 and
            np.isclose(row['true_worst_objective'],
                float(max(losses) + row['prior']), rtol=0, atol=1e-10) and
            row['selector_feasible'] is fit.selector_feasible(ratios) and
            row['raw_sha256'] == fit.sha256(row['raw_path']))
    except (KeyError, TypeError, ValueError, OSError):
        return False


def result_from_records(freeze, ledger, cache, comparison, stop_reason):
    rows = ledger['original_evaluations']
    complete = bool(1 <= len(rows) <= freeze['max_unique_evaluations'] and
        all(row.get('index') == index and _complete_row(row)
            for index, row in enumerate(rows, 1)) and
        ledger['original_evaluations_requested'] ==
            ledger['original_evaluations_returned'] ==
            ledger['original_evaluations_completed'] == len(rows) and
        ledger['original_evaluations_failed_unknown'] == 0 and
        ledger['original_evaluations_post_return_error'] == 0 and
        cache.started == cache.completed == len(cache.cache) == len(rows))
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
        comparison.get('actual_sha256') == rows[0]['raw_sha256'] and
        rows[0]['selector_feasible'] is True) if rows else False
    tiny = ledger.get('synthetic_tiny', {})
    tiny_ok = bool(tiny.get('status') == 'completed' and
        tiny.get('raw_saved') is True and tiny.get('solve_returned') is True)
    valid = complete and optimizer_ok and budget_ok and first_ok and tiny_ok
    best_any, best = fit.choose_records(rows) if complete else (None, None)
    selected_index = best['index'] if best else None
    selected_objective = best['true_worst_objective'] if best else None
    start_objective = rows[0]['true_worst_objective'] if complete else None
    ratio = (selected_objective / start_objective if start_objective and
        selected_objective is not None else None)
    accepted = bool(valid and selected_index != 1 and ratio is not None and
        ratio <= freeze['proposal_objective_ratio_limit'])
    return {'schema': 'flylab.m3.e136.timescale-constrained-fit-result.v1',
        'status': 'completed_bounded_fit' if valid else 'invalid_evidence',
        'hypothesis_verdict': ('supported' if accepted else
            'refuted' if valid else 'inconclusive'),
        'interpretation': 'local_LF_actual_nonlinear_fit_only',
        'new_lf_proposal_accepted': accepted,
        'start_objective': start_objective,
        'selected_index': selected_index,
        'selected_objective': selected_objective,
        'selected_objective_over_start': ratio,
        'best_any_index': best_any['index'] if best_any else None,
        'best_any_objective': best_any['true_worst_objective'] if best_any else None,
        'stop_reason': stop_reason, 'first_reference_comparison': first_ok,
        'original_completion_accounting_complete': complete,
        'optimizer_accounting_complete': optimizer_ok,
        'fit_budget_complete': budget_ok, 'synthetic_tiny_complete': tiny_ok,
        'original_evaluations_requested': ledger['original_evaluations_requested'],
        'original_evaluations_returned': ledger['original_evaluations_returned'],
        'original_evaluations_completed': ledger['original_evaluations_completed'],
        'cache_started': cache.started, 'cache_completed': cache.completed,
        'cache_hits': cache.hits, 'fit_calls': ledger['fit_calls'],
        'fit_wall_s': ledger['fit_wall_s'], 'raw_bytes': ledger['raw_bytes'],
        'original_neural_evaluations': ledger['original_evaluations_requested'],
        'body_calls': 0, 'F6_exposed': False,
        'candidate_adopted': False, 'whole_candidate_adopted': False,
        'whole_goal_complete': False,
        'freeze_sha256': ledger['freeze_sha256'],
        'source_review_sha256': ledger['source_review_sha256']}


def write_output_manifest(out):
    save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e136.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): fit.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file() and
            path.name != 'output-manifest.json'}})


def finalize(out, ledger, freeze, began, primary):
    """Never leave an apparently successful receipt after late I/O failure."""
    def note(stage, error):
        nonlocal primary
        ledger.setdefault('finalization_errors', []).append({
            'stage': stage, 'error': repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note('Secondary ' + stage + ': ' + repr(error))

    def keep_primary_evidence():
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
                'E136 internal wall limit exceeded after manifest'))
        if primary is not None:
            break
        ledger['post_manifest_wall_s'] = time.perf_counter() - began
    if primary is not None:
        ledger['status'] = 'failed'
        keep_primary_evidence()
        result_path = out / 'result.json'
        if result_path.exists():
            try:
                result_path.unlink()
            except BaseException as error:
                note('result_removal', error)
                try:
                    save(result_path, {'status': 'invalid_finalization',
                        'hypothesis_verdict': 'inconclusive',
                        'new_lf_proposal_accepted': False})
                except BaseException as fallback:
                    note('result_invalidation_fallback', fallback)
        keep_primary_evidence()
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            note('failed_ledger', error)
        try:
            write_output_manifest(out)
        except BaseException as error:
            note('failed_manifest', error)
        if ledger.get('finalization_errors'):
            # A late manifest error must also appear in the durable failed ledger.
            keep_primary_evidence()
            try:
                save(out / 'advance-ledger.json', ledger)
            except BaseException as error:
                note('failed_ledger_update', error)
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('One-shot E136 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e136.timescale-constrained-fit-ledger.v1',
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
        start = start_vector(freeze)
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
        run_tiny_setup(out, ledger)
        fixture = fit.frozen_problem(e128)
        geometry = TorchFootGeometry.load(
            ROOT / freeze['geometry_path'], device='cuda')
        torch.cuda.synchronize(geometry.device)
        if geometry.geometry_hash != freeze['geometry_hash']:
            raise ValueError('E136 frozen geometry hash mismatch')
        ledger['geometry_fixed_upload_bytes'] = sum(
            int(geometry.arrays[key].nbytes) for key in
            ('axes', 'anchors', 'tips', 'reference_angles'))
        motors, muscles, edges = (np.flatnonzero(fixture[key][0]) for key in
            ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask'))
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('E136 LF identity cardinality changed')
        bounds = original_bounds(fixture, edges)
        if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
            raise ValueError('E136 warm start outside original bounds')
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
                error.add_note('Secondary E136 constructor ledger: ' +
                               repr(secondary))
            raise
        ledger['engine_before'] = snapshot(engine.metadata())
        with np.load(ROOT / freeze['start_raw_path'], allow_pickle=False) as ref:
            for key, expected in (
                    ('motor_indices', engine.problem['motor_indices']),
                    ('muscle_indices', engine.problem['muscle_indices']),
                    ('edge_ids', engine.problem['edge_ids'])):
                if not np.array_equal(ref[key], expected):
                    raise ValueError('E136 LF engine identity changed: ' + key)
            if not np.array_equal(ref['parameter_muscle_map'],
                tau_module.activation_parameter_map(engine.problem)[:290]):
                raise ValueError('E136 LF old parameter map changed')
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        fit_began = time.perf_counter()
        fit_deadline = fit_began + freeze['fit_wall_limit_s']
        internal_deadline = began + freeze['internal_wall_limit_s']

        def persist():
            refresh_counts(ledger)
            save(out / 'advance-ledger.json', ledger)

        def on_start(index, vector):
            register_intent(ledger, index, vector, persist)

        def evaluate(vector, index):
            nonlocal comparison
            row = ledger['original_evaluations'][index - 1]
            directory = out / f'evaluation-{index:03d}'

            def first_compare(record):
                report = compare_first_raw(record['raw_path'],
                    ROOT / freeze['start_raw_path'],
                    ROOT / freeze['E135_base_path'])
                save(out / 'first-reference-comparison.json', report)
                return report

            def evaluator(x, row_index, path, **callbacks):
                return evaluate_timescale_leg(engine, fixture, geometry,
                    x, row_index, path, **callbacks)

            record, compared = evaluate_with_receipts(ledger, row, vector,
                directory, evaluator, persist,
                first_compare=first_compare if index == 1 else None,
                metadata=engine.metadata,
                maximum_raw_bytes=freeze['maximum_raw_bytes'])
            if compared is not None:
                comparison = compared
            return record

        cache = fit.EvaluationCache(evaluate, 307, on_start,
            max_calls=freeze['max_unique_evaluations'],
            deadline=fit_deadline, total_deadline=internal_deadline)
        try:
            solver, stop_reason = bounded_optimizer(cache, start, bounds,
                ledger, persist)
        finally:
            active = sys.exc_info()[1]
            ledger['fit_wall_s'] = time.perf_counter() - fit_began
            refresh_counts(ledger)
            try:
                persist()
            except BaseException as secondary:
                if active is not None:
                    active.add_note('Secondary E136 fit ledger: ' +
                                    repr(secondary))
                else:
                    raise
        ledger['solver_returned'] = solver is not None
        ledger['stop_reason'] = stop_reason
        if ledger['fit_wall_s'] > freeze['fit_wall_limit_s']:
            raise TimeoutError('E136 fit wall limit exceeded')
        if time.perf_counter() >= internal_deadline - 1:
            raise TimeoutError('E136 result reserve exhausted')
        result = result_from_records(freeze, ledger, cache, comparison,
                                     stop_reason)
        if result['status'] != 'completed_bounded_fit':
            ledger['hypothesis_verdict'] = 'inconclusive'
            raise ValueError('E136 original evidence incomplete')
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
        refresh_counts(ledger)
        if engine is not None:
            try:
                ledger['engine_after'] = snapshot(engine.metadata())
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E136 engine metadata: ' + repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note('Secondary E136 threadpool restore: ' + repr(error))
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
