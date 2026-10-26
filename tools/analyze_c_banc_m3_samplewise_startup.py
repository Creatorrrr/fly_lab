"""E118 saved-response startup derivative bank; no new forward response."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.fit_c_banc_m3_context_trust import INNER, LEGS, ORIGINAL, SELECTOR_CAP
from tools.verify_c_banc_m3_e117_nonlinear_steps import (
    VARIANTS,
    load_npz,
    peak_decomposition,
    require,
    rescue_raw,
    save_json,
    sha256,
)
from tools.verify_c_banc_m3_e117_nonlinear_steps import (
    reviewed_inputs as e117_inputs,
)

DESIGN_SHA = 'd56c28f76e82f2b141a34628e3bac1ad45b764e733b30e0b867675c82d6c2142'
FREEZE_SHA = 'ce2bccd086a02e1c4d6851679073b2dc02b2160a4a2aefccf32ab03c5b06c7c6'
PREFIX_ROWS = 100
STARTUP_ROWS = 99
EDGE_COUNT = 1572
INTERNAL_WALL_S = 15.
PRIOR_PROCESS_S = 710.
SHARED_PROCESS_LIMIT_S = 735.
BANK_FIELDS = ('current_derivative', 'forcing_derivative',
               'motor_derivative', 'activation_derivative')


def reviewed_inputs(root):
    """Verify pinned inputs and stored responses without any filter/model call."""
    root = Path(root)
    require(sha256(ROOT/'docs/BANC_M3_E118_SAMPLEWISE_STARTUP_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(root/'e118-frozen-design.json') == FREEZE_SHA,
            'E118 frozen design changed')
    freeze = json.loads((root/'e118-frozen-design.json').read_text(
        encoding='utf-8'))
    require(freeze['actual_execution_authorized'] is False and
            freeze['CUDA_derivative_bank_executions'] == 1 and
            freeze['linear_filter_calls'] == freeze['endpoint_filter_calls'] == 1
            and freeze['original_conditional_responses'] == 0 and
            freeze['prefix_rows'] == PREFIX_ROWS and
            freeze['startup_rows'] == STARTUP_ROWS and
            freeze['edge_count'] == EDGE_COUNT and
            freeze['derivative_context_count'] == 2 and
            freeze['derivative_channel_count'] == 2*EDGE_COUNT and
            freeze['internal_wall_s'] == INTERNAL_WALL_S and
            freeze['prior_conservative_process_s'] == PRIOR_PROCESS_S and
            freeze['shared_process_limit_s'] == SHARED_PROCESS_LIMIT_S and
            freeze['startup_gradient_parity_max_abs'] == 1e-8 and
            freeze['startup_ratio_parity_max_abs'] == 1e-8 and
            freeze['gradient_tie_policy'] ==
                'average exact squared-error maxima only' and
            freeze['variant_names'] == list(VARIANTS),
            'E118 fixed axis, math or resource contract changed')
    for relative, expected in freeze['input_sha256'].items():
        require(sha256(root/relative) == expected,
                f'E118 frozen input changed: {relative}')
    for relative, expected in freeze['source_sha256'].items():
        require(sha256(ROOT/relative) == expected,
                f'E118 reused source changed: {relative}')
    parent = json.loads((root/'e117-nonlinear-steps/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    structure = json.loads((root/'parent-e118-structure-review.json').read_text(
        encoding='utf-8'))
    require(parent['accepted_as_valid_failed_nonlinear_diagnostic'] is True and
            parent['blocking_defects'] == [] and
            structure['passed'] is True and len(structure['checks']) == 19 and
            all(structure['checks'].values()) and
            structure['derivative_filter_executions'] == 0,
            'Parent E117/structure acceptance changed')
    data = e117_inputs(root)
    f, step = data['fixture'], data['step']
    response = load_npz(root/'e117-nonlinear-steps/response-raw.npz')
    metrics = load_npz(root/'e117-nonlinear-steps/metrics-raw.npz')
    require(response['current'].shape == (6, 2001, 391) and
            response['muscle_activation'].shape == (6, 2001, 96) and
            response['command'].shape == (6, 2001, 42) and
            response['tendon'].shape == (6, 2001, 6) and
            response['weights'].shape == (3, 391, 49) and
            metrics['startup_ratios'].shape == (3, 6, 2, 4) and
            np.array_equal(response['weights'], step['weights']) and
            np.array_equal(response['pair_context_indices'],
                           step['pair_context_indices']) and
            np.array_equal(response['pair_weight_indices'],
                           step['pair_weight_indices']) and
            np.array_equal(response['time_s'], f['time_s']) and
            np.array_equal(response['variant_names'], step['variant_names']) and
            np.array_equal(metrics['variant_names'], step['variant_names']) and
            np.array_equal(f['startup_mask'][:PREFIX_ROWS],
                           np.arange(PREFIX_ROWS) > 0) and
            not np.any(f['startup_mask'][PREFIX_ROWS:]) and
            np.array_equal(f['time_s'][:PREFIX_ROWS],
                           np.arange(PREFIX_ROWS)*.005) and
            np.isfinite(response['current'][:2, :PREFIX_ROWS]).all() and
            np.isfinite(response['muscle_activation'][:2, :PREFIX_ROWS]).all(),
            'E117 selected_solver saved axes or finite prefix changed')
    rows, cols = f['edge_rows'], f['edge_cols']
    direct_q = f['moment'] @ f['pool']
    mask = step['leg_edge_mask']
    require(mask.shape == (6, EDGE_COUNT) and np.all(mask.sum(axis=0) == 1),
            'Each selected edge must belong to exactly one leg')
    for leg in range(6):
        positions = np.flatnonzero(mask[leg])
        other_joints = np.r_[0:7*leg, 7*leg+7:42]
        other_ltm = np.delete(f['ltm'], leg)
        require(np.count_nonzero(direct_q[np.ix_(other_joints,
                                                  rows[positions])]) == 0 and
                np.count_nonzero(f['pool'][np.ix_(other_ltm,
                                                  rows[positions])]) == 0,
                f'E118 cross-leg direct derivative changed: {LEGS[leg]}')
    selected_weights = step['weights'][0, rows, cols]
    require(np.array_equal(selected_weights,
                           load_npz(root/'e115-context-trust-fit/'
                                    'selected-delta.npz')['selected_solver_W']),
            'E118 selected edge W changed')
    data.update(freeze=freeze, response=response, metrics=metrics,
                selected_weights=selected_weights, root=root)
    return data


def stage_bank_inputs(data, device, synchronize=lambda: None):
    """Transfer all actual saved operands before any bank accounting."""
    f, response = data['fixture'], data['response']
    weights = data['step']['weights'][0]
    values = {
        'source': f['source_rate'][:, :PREFIX_ROWS],
        'current': response['current'][:2, :PREFIX_ROWS],
        'weights': weights,
        'rows': f['edge_rows'], 'cols': f['edge_cols'],
        'tau': f['tau'], 'a': f['a'], 'threshold': f['threshold'],
        'cap': f['cap'],
    }
    staged = {name: torch.as_tensor(value, dtype=torch.long if name in
              ('rows', 'cols') else torch.float64, device=device)
              for name, value in values.items()}
    synchronize()
    return staged


def derivative_bank(staged, *, record=lambda _key, _value: None,
                    on_filter=lambda _name, _state: None):
    """Analytic edgewise chain, with exactly one call to each causal filter."""
    source, current, weights = (staged[key] for key in
                                ('source', 'current', 'weights'))
    rows, cols = staged['rows'], staged['cols']
    tau, a, threshold, cap = (staged[key] for key in
                               ('tau', 'a', 'threshold', 'cap'))
    if any(x.device != source.device for x in staged.values()) or \
            any(x.dtype != torch.float64 for key, x in staged.items()
                if key not in ('rows', 'cols')) or \
            any(staged[key].dtype != torch.long for key in ('rows', 'cols')) or \
            source.ndim != 3 or source.shape[0] != 2 or \
            current.shape != (2, source.shape[1], weights.shape[0]) or \
            rows.ndim != 1 or cols.shape != rows.shape or \
            weights.shape[1] != source.shape[2] or \
            any(x.shape != (weights.shape[0],) for x in
                (tau, a, threshold, cap)) or \
            source.shape[1] < 2 or rows.numel() == 0 or \
            bool(torch.any(rows < 0)) or bool(torch.any(rows >= len(tau))) or \
            bool(torch.any(cols < 0)) or bool(torch.any(cols >= source.shape[2])) \
            or any(not bool(torch.isfinite(x).all()) for key, x in staged.items()
                   if key not in ('rows', 'cols')) or \
            bool(torch.any(tau <= 0)) or bool(torch.any(cap <= 0)):
        raise ValueError('Finite aligned two-context derivative operands required')
    n_time, n_edge = source.shape[1], rows.numel()
    selected_w = weights[rows, cols]
    current_derivative = source[:, :, cols]*selected_w
    record('current_derivative', current_derivative)
    u = a[rows]*(current[:, :, rows]-threshold[rows])/cap[rows]
    tanh_u = torch.tanh(u)
    forcing_derivative = (tanh_u > 0).to(source.dtype)*a[rows]*(1-tanh_u**2)*\
        current_derivative
    record('forcing_derivative', forcing_derivative)
    forcing_flat = forcing_derivative.permute(1, 0, 2).reshape(n_time,
                                                               2*n_edge)
    on_filter('linear', 'started')
    motor_flat = linear_response(forcing_flat, .005, tau[rows].repeat(2))
    on_filter('linear', 'returned')
    motor_derivative = motor_flat.reshape(n_time, 2, n_edge).permute(1, 0, 2)
    record('motor_derivative', motor_derivative)
    on_filter('endpoint', 'started')
    activation_flat = endpoint_response(motor_flat, .005, .01)
    on_filter('endpoint', 'returned')
    activation_derivative = activation_flat.reshape(n_time, 2, n_edge).permute(
        1, 0, 2)
    record('activation_derivative', activation_derivative)
    return {'current_derivative': current_derivative,
        'forcing_derivative': forcing_derivative,
        'motor_derivative': motor_derivative,
        'activation_derivative': activation_derivative}


def bank_metadata(data):
    f = data['fixture']
    return {'edge_ids': f['edge_ids'].copy(),
        'edge_rows': f['edge_rows'].copy(),
        'edge_cols': f['edge_cols'].copy(),
        'selected_weights': data['selected_weights'].copy(),
        'context_names': f['context_names'].copy(),
        'time_s': f['time_s'][:PREFIX_ROWS].copy(),
        'source_rate_prefix': f['source_rate'][:, :PREFIX_ROWS].copy(),
        'selected_current_prefix': data['response']['current'][
            :2, :PREFIX_ROWS].copy(),
        'selected_activation_prefix': data['response']['muscle_activation'][
            :2, :PREFIX_ROWS].copy()}


def detach_available(tensors, raw, primary=None):
    for key, value in tensors.items():
        if key in raw:
            continue
        try:
            raw[key] = value.detach().cpu().numpy().copy()
        except Exception as secondary:
            if primary is None:
                raise
            primary.add_note(f'E118 secondary {key} materialization: {secondary}')


def execute_bank(staged, metadata, out, ledger, persist, deadline,
                 synchronize=lambda: None, bank_fn=derivative_bank):
    """Durable one-bank receipt, raw before assertions, and failure rescue."""
    raw = {key: value.copy() for key, value in metadata.items()}
    tensors = {}
    deadline()
    ledger.update(stage='bank', bank_started=1,
        derivative_bank_reservation_requested=1)
    try:
        persist()
    except BaseException as primary:
        ledger.update(bank_started=0, bank_start_receipt_failed=True)
        rescue_raw(out/'error-bank-input', raw, primary, 'bank input')
        raise

    def record(key, tensor):
        tensors[key] = tensor

    def on_filter(name, state):
        ledger[f'{name}_filter_{state}'] += 1

    try:
        ledger['bank_call_entered'] = True
        returned = bank_fn(staged, record=record, on_filter=on_filter)
        tensors.update(returned)
        detach_available(tensors, raw)
        synchronize()
        ledger['linear_filter_completed'] = ledger['linear_filter_returned']
        ledger['endpoint_filter_completed'] = ledger['endpoint_filter_returned']
        ledger.update(stage='bank_materialized', bank_materialized=1)
        persist()
        np.savez_compressed(out/'derivative-bank-raw.npz', **raw)
        ledger.update(stage='bank_raw_saved', bank_raw_saved=1)
        persist()
        require(all(key in raw and raw[key].shape ==
                    (2, PREFIX_ROWS, EDGE_COUNT) for key in BANK_FIELDS),
                'E118 derivative bank output axes changed')
        return raw
    except BaseException as primary:
        detach_available(tensors, raw, primary)
        rescue_raw(out/'error-bank-raw', raw, primary, 'bank')
        raise


def startup_gradient(q_error, tendon_error, q_jacobian, tendon_jacobian,
                     inner=INNER):
    """Eight E115 startup ratio gradients; exact amax ties are averaged."""
    q_error = np.asarray(q_error, dtype=np.float64)
    tendon_error = np.asarray(tendon_error, dtype=np.float64)
    q_jacobian = np.asarray(q_jacobian, dtype=np.float64)
    tendon_jacobian = np.asarray(tendon_jacobian, dtype=np.float64)
    if q_error.ndim != 3 or q_error.shape[0] != 2 or \
            q_jacobian.shape[:3] != q_error.shape or \
            tendon_error.shape != q_error.shape[:2] or \
            tendon_jacobian.shape != (*tendon_error.shape,
                                      q_jacobian.shape[-1]) or \
            not all(np.isfinite(x).all() for x in
                    (q_error, tendon_error, q_jacobian, tendon_jacobian)):
        raise ValueError('Aligned finite startup error/Jacobian axes required')
    inner = np.asarray(inner, dtype=np.float64)
    if inner.shape != (4,) or np.any(inner <= 0):
        raise ValueError('Four positive startup inner caps required')
    result = np.empty((8, q_jacobian.shape[-1]), dtype=np.float64)
    exact_ties = np.empty((2, 2), dtype=np.int64)
    explanatory_ties = np.empty_like(exact_ties)
    for context in range(2):
        q = q_error[context]
        t = tendon_error[context]
        jq = q_jacobian[context]
        jt = tendon_jacobian[context]
        row = 4*context
        result[row] = 2*np.einsum('ij,ije->e', q, jq)/(
            q.size*inner[0]**2)
        # Match E115's error.square().amax()/cap**2 operation order. Scaling
        # before selecting an exact maximum can merge adjacent float64 values.
        q_unscaled_squared = q**2
        q_ties = q_unscaled_squared == q_unscaled_squared.max()
        result[row+1] = (2*q[q_ties][:, None]*jq[q_ties]/inner[1]**2).mean(
            axis=0)
        result[row+2] = 2*np.einsum('i,ie->e', t, jt)/(
            t.size*inner[2]**2)
        t_unscaled_squared = t**2
        t_ties = t_unscaled_squared == t_unscaled_squared.max()
        result[row+3] = (2*t[t_ties][:, None]*jt[t_ties]/inner[3]**2).mean(
            axis=0)
        exact_ties[context] = [np.count_nonzero(q_ties),
                               np.count_nonzero(t_ties)]
        q_squared = (q/inner[1])**2
        q_max = q_squared.max()
        t_squared = (t/inner[3])**2
        t_max = t_squared.max()
        explanatory_ties[context] = [np.count_nonzero(np.abs(q_squared-q_max)
            <= 1e-10*max(1., abs(q_max))),
            np.count_nonzero(np.abs(t_squared-t_max)
            <= 1e-10*max(1., abs(t_max)))]
    return result, exact_ties, explanatory_ties


def leg_jacobian(bank, data, leg, record=lambda _values: None):
    """Combine edgewise filtered response with the original fixed anatomy."""
    f, response = data['fixture'], data['response']
    positions = np.flatnonzero(data['step']['leg_edge_mask'][leg])
    rows = f['edge_rows'][positions]
    record({'edge_positions': positions, 'edge_ids': f['edge_ids'][positions],
            'edge_rows': rows, 'edge_cols': f['edge_cols'][positions]})
    z = bank['activation_derivative'][:, 1:PREFIX_ROWS, positions]
    direct_q = f['moment'] @ f['pool']
    q_coefficient = .02*direct_q[7*leg:7*leg+7][:, rows]
    q_jacobian = z[:, :, None, :]*q_coefficient[None, None]
    ltm_muscle = f['ltm'][leg]
    activation = response['muscle_activation'][:2, 1:PREFIX_ROWS,
                                               ltm_muscle]
    tendon_coefficient = .02*(1-np.tanh(activation/10)**2)[:, :, None]*\
        f['pool'][ltm_muscle, rows][None, None]
    tendon_jacobian = tendon_coefficient*z
    q_error = response['command'][:2, 1:PREFIX_ROWS,
                                  7*leg:7*leg+7]-f[
                                      'startup_target_command'][None, :,
                                                                7*leg:7*leg+7]
    tendon_error = response['tendon'][:2, 1:PREFIX_ROWS, leg]-f[
        'startup_target_tendon'][None, :, leg]
    record({'q_jacobian': q_jacobian,
            'tendon_jacobian': tendon_jacobian,
            'q_error': q_error, 'tendon_error': tendon_error})
    gradient, exact_ties, explanatory_ties = startup_gradient(
        q_error, tendon_error, q_jacobian, tendon_jacobian)
    reference = data['baseline'][LEGS[leg]]['jacobian'][2:]
    return {'q_jacobian': q_jacobian, 'tendon_jacobian': tendon_jacobian,
        'edge_positions': positions, 'edge_ids': f['edge_ids'][positions],
        'edge_rows': rows, 'edge_cols': f['edge_cols'][positions],
        'q_error': q_error, 'tendon_error': tendon_error,
        'startup_jacobian': gradient, 'reference_jacobian': reference.copy(),
        'gradient_difference': gradient-reference,
        'max_exact_ties': exact_ties,
        'max_explanatory_ties': explanatory_ties}


def signed_predictions(jacobians, data):
    """Apply the two already-saved h vectors to every signed startup sample."""
    step = data['step']
    q = np.empty((3, 2, STARTUP_ROWS, 42), dtype=np.float64)
    tendon = np.empty((3, 2, STARTUP_ROWS, 6), dtype=np.float64)
    for leg, name in enumerate(LEGS):
        item = jacobians[name]
        positions = item['edge_positions']
        for variant in range(3):
            h = step['increments'][variant, positions]
            q[variant, :, :, 7*leg:7*leg+7] = item['q_error'] + \
                np.einsum('ctje,e->ctj', item['q_jacobian'], h)
            tendon[variant, :, :, leg] = item['tendon_error'] + \
                np.einsum('cte,e->ct', item['tendon_jacobian'], h)
    values = np.empty((3, 6, 2, 4), dtype=np.float64)
    for variant in range(3):
        for leg in range(6):
            for context in range(2):
                q_leg = q[variant, context, :, 7*leg:7*leg+7]
                t_leg = tendon[variant, context, :, leg]
                values[variant, leg, context] = [
                    np.sqrt(np.mean(q_leg**2)), np.max(np.abs(q_leg)),
                    np.sqrt(np.mean(t_leg**2)), np.max(np.abs(t_leg))]
    return {'q_error_predicted': q, 'tendon_error_predicted': tendon,
        'startup_values': values, 'startup_ratios': (values/INNER)**2,
        'variant_names': np.asarray(VARIANTS),
        'context_names': data['fixture']['context_names'].copy(),
        'time_s': data['fixture']['time_s'][1:PREFIX_ROWS].copy()}


def comparison_and_peaks(prediction, data):
    """Keep affine, signed-linear and actual nonlinear screens separate."""
    actual = data['metrics']['startup_ratios']
    affine = data['metrics']['affine_startup_ratios']
    linear = prediction['startup_ratios']
    comparison = {'affine_startup_ratios': affine,
        'signed_linear_startup_ratios': linear,
        'actual_nonlinear_startup_ratios': actual,
        'linear_minus_affine': linear-affine,
        'actual_minus_linear': actual-linear,
        'signed_linear_inner_pass': np.all(linear <= SELECTOR_CAP,
                                           axis=(2, 3)),
        'signed_linear_original_pass': np.all(prediction['startup_values'] <=
                                               ORIGINAL, axis=(2, 3)),
        'actual_inner_pass': np.all(actual <= SELECTOR_CAP, axis=(2, 3)),
        'actual_original_pass': np.all(data['metrics']['startup_values'] <=
                                      ORIGINAL, axis=(2, 3))}
    f, response = data['fixture'], data['response']
    indices = np.flatnonzero(f['startup_mask'])
    peaks = {}
    for variant, variant_name in enumerate(VARIANTS):
        peaks[variant_name] = {}
        for leg, leg_name in enumerate(LEGS):
            peaks[variant_name][leg_name] = {}
            for context, context_name in enumerate(f['context_names']):
                q0 = prediction['q_error_predicted'][0, context, :,
                                                      7*leg:7*leg+7]
                t0 = prediction['tendon_error_predicted'][0, context, :, leg]
                q_linear = prediction['q_error_predicted'][variant, context,
                                                            :, 7*leg:7*leg+7]
                t_linear = prediction['tendon_error_predicted'][variant,
                                                                 context, :, leg]
                q_actual = response['command'][2*variant+context,
                    1:PREFIX_ROWS, 7*leg:7*leg+7]-f[
                        'startup_target_command'][:, 7*leg:7*leg+7]
                t_actual = response['tendon'][2*variant+context,
                    1:PREFIX_ROWS, leg]-f['startup_target_tendon'][:, leg]
                peaks[variant_name][leg_name][str(context_name)] = {
                    'qmax': {'signed_linear': peak_decomposition(q0,
                        q_linear, affine[variant, leg, context, 1],
                        threshold=INNER[1], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='qmax'),
                        'actual_nonlinear': peak_decomposition(q0, q_actual,
                        affine[variant, leg, context, 1],
                        threshold=INNER[1], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='qmax')},
                    'LTMmax': {'signed_linear': peak_decomposition(t0,
                        t_linear, affine[variant, leg, context, 3],
                        threshold=INNER[3], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='LTMmax'),
                        'actual_nonlinear': peak_decomposition(t0, t_actual,
                        affine[variant, leg, context, 3],
                        threshold=INNER[3], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='LTMmax')}}
    return comparison, peaks


def run_once(root, out):
    """One later-authorized bank; all numerical raw precedes assertions."""
    root, out = Path(root), Path(out)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'experiment': 'E118-samplewise-startup', 'stage': 'created',
        'prior_conservative_process_s': PRIOR_PROCESS_S,
        'shared_process_limit_s': SHARED_PROCESS_LIMIT_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'bank_started': 0, 'bank_materialized': 0, 'bank_raw_saved': 0,
        'derivative_bank_reservation_requested': 0,
        'bank_call_entered': False, 'linear_filter_started': 0,
        'linear_filter_returned': 0, 'linear_filter_completed': 0,
        'endpoint_filter_started': 0, 'endpoint_filter_returned': 0,
        'endpoint_filter_completed': 0, 'jacobians_materialized': 0,
        'jacobians_raw_saved': 0, 'prediction_materialized': False,
        'prediction_raw_saved': False, 'derivative_contexts': 2,
        'derivative_channels': 2*EDGE_COUNT, 'prefix_rows': PREFIX_ROWS,
        'derivative_horizon_s': .495, 'conditional_response_calls': 0,
        'actual_neural_s': 0., 'actual_physical_s': 0., 'LP_calls': 0,
        'fit_calls': 0, 'calibration_outputs': 0, 'body_calls': 0,
        'secondary_errors': []}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E118 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
        except Exception as secondary:
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E118 secondary clock failure: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadline():
        elapsed = time.monotonic()-started
        if elapsed > INTERNAL_WALL_S or \
                PRIOR_PROCESS_S+elapsed > SHARED_PROCESS_LIMIT_S:
            raise TimeoutError('E118 internal/shared wall limit exceeded')

    def failure_save(filename, value, primary):
        try:
            save_json(out/filename, value)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E118 secondary {filename} save: {secondary}')

    primary = None
    data = None
    metadata = None
    bank = None
    pending_jacobian = None
    jacobians = {}
    prediction = None
    comparison = None
    peaks = None
    try:
        for original, name in (
            (ROOT/'tools/analyze_c_banc_m3_samplewise_startup.py', 'runner.py'),
            (ROOT/'tests/test_c_banc_samplewise_startup.py',
             'test_c_banc_samplewise_startup.py'),
            (ROOT/'docs/BANC_M3_E118_SAMPLEWISE_STARTUP_DESIGN_20260920.md',
             'frozen-design.md'),
            (root/'e118-frozen-design.json', 'e118-frozen-design.json'),
            (root/'parent-e118-structure-review.json',
             'parent-e118-structure-review.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        data = reviewed_inputs(root)
        deadline()
        require(torch.cuda.is_available() and
                'RTX 4080' in torch.cuda.get_device_name(0) and
                all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
                'Reviewed CUDA RTX4080/single-thread environment required')
        torch.set_num_threads(1)
        require(torch.get_num_threads() == 1, 'Torch CPU thread count changed')
        ledger['environment'] = {'cuda_device': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__, 'torch_cuda_version':
            torch.version.cuda, 'torch_cpu_threads': torch.get_num_threads(),
            'omp': os.environ['OMP_NUM_THREADS'],
            'mkl': os.environ['MKL_NUM_THREADS'],
            'openblas': os.environ['OPENBLAS_NUM_THREADS']}
        metadata = bank_metadata(data)
        staged = stage_bank_inputs(data, 'cuda:0', torch.cuda.synchronize)
        deadline()
        bank = execute_bank(staged, metadata, out, ledger, persist, deadline,
                            torch.cuda.synchronize)
        deadline()
        for leg, name in enumerate(LEGS):
            pending_jacobian = {}
            completed = leg_jacobian(bank, data, leg,
                                     record=pending_jacobian.update)
            pending_jacobian = completed
            ledger['jacobians_materialized'] += 1
            ledger['stage'] = f'jacobian_{name}_materialized'
            persist()
            np.savez_compressed(out/f'jacobian-{name}.npz',
                                **pending_jacobian)
            ledger['jacobians_raw_saved'] += 1
            ledger['stage'] = f'jacobian_{name}_raw_saved'
            persist()
            jacobians[name] = pending_jacobian
            pending_jacobian = None
            deadline()
        prediction = signed_predictions(jacobians, data)
        ledger['prediction_materialized'] = True
        persist()
        np.savez_compressed(out/'signed-prediction-raw.npz', **prediction)
        ledger['prediction_raw_saved'] = True
        ledger['stage'] = 'prediction_raw_saved'
        persist()
        comparison, peaks = comparison_and_peaks(prediction, data)
        save_json(out/'signed-prediction-comparison.json', comparison)
        save_json(out/'signed-prediction-peaks.json', peaks)
        save_json(out/'provenance.json', {'design_sha256': DESIGN_SHA,
            'freeze_sha256': FREEZE_SHA,
            'input_sha256': data['freeze']['input_sha256'],
            'source_sha256': data['freeze']['source_sha256'],
            'runner_sha256': sha256(ROOT/
                'tools/analyze_c_banc_m3_samplewise_startup.py'),
            'test_sha256': sha256(ROOT/
                'tests/test_c_banc_samplewise_startup.py'),
            'derivative_semantics': 'fixed selected_solver saved current, activation, source and W',
            'activation_derivative_semantics': 'per-edge endpoint-filtered motor derivative before pool',
            'no_new_forward_response': True})
        gradient_max = max(float(np.max(np.abs(item['gradient_difference'])))
                           for item in jacobians.values())
        ratio_max = float(np.max(np.abs(prediction['startup_ratios'][0]-
            data['metrics']['startup_ratios'][0])))
        require(gradient_max <= 1e-8 and ratio_max <= 1e-8 and
                all(np.isfinite(bank[key]).all() for key in BANK_FIELDS) and
                np.isfinite(prediction['startup_ratios']).all() and
                ledger['linear_filter_started'] ==
                    ledger['linear_filter_completed'] == 1 and
                ledger['endpoint_filter_started'] ==
                    ledger['endpoint_filter_completed'] == 1 and
                ledger['jacobians_raw_saved'] == 6,
                'E118 saved E115 gradient/ratio or resource parity failed')
        deadline()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_saved_response_samplewise_derivative',
            'candidate_adopted': False, 'native_gait_tested': False,
            'new_forward_responses': 0, 'gradient_max_abs': gradient_max,
            'baseline_ratio_max_abs': ratio_max,
            'signed_linear_inner_pass': comparison[
                'signed_linear_inner_pass'],
            'actual_inner_pass': comparison['actual_inner_pass'],
            'no_biological_capacity_conclusion': True, 'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', ledger['primary_error'], error)
        if metadata is not None and not ledger['bank_started']:
            rescue_raw(out/'error-available/input', metadata,
                       error, 'input')
        if bank is not None and not ledger['bank_raw_saved']:
            rescue_raw(out/'error-available/bank', bank, error, 'bank')
        if pending_jacobian:
            rescue_raw(out/'error-available/jacobian', pending_jacobian,
                       error, 'jacobian')
        if prediction is not None and not ledger['prediction_raw_saved']:
            rescue_raw(out/'error-available/prediction', prediction,
                       error, 'prediction')
        if comparison is not None:
            failure_save('error-available-comparison.json', comparison, error)
        if peaks is not None:
            failure_save('error-available-peaks.json', peaks, error)
        raise
    finally:
        if primary is not None:
            refresh_clock(primary)
            failure_save('execution-ledger.json', ledger, primary)
            failure_save('error.json', {'primary_error': ledger['primary_error'],
                'process_wall_s': ledger['process_wall_s'],
                'process_wall_status': ledger['process_wall_status']}, primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919')
    parser.add_argument('--out', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919/e118-samplewise-startup')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
