"""E117 one-batch nonlinear checks of saved E116 directions; no native gait."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_context_response import ContextResponse
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
from tools.fit_c_banc_m3_context_trust import (
    GEOMETRY_HASH,
    INNER,
    LEGS,
    ORIGINAL,
    SELECTOR_CAP,
    frozen_output_metrics,
)
from tools.fit_c_banc_m3_context_trust import reviewed_inputs as e115_inputs

DESIGN_SHA = '8aa0fc1d3c84925a8606cc3d74bc4291ae31fbcee7a9c1addd3bf0462c9eb64b'
FREEZE_SHA = '3047916172654d023482dd4eb77c1562d3ce854e773cd00d9f25b77ac8797b73'
VARIANTS = ('selected_solver', 'shape_box', 'startup_box')
PAIR_CONTEXT = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
PAIR_WEIGHT = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
RESPONSE_FIELDS = ('current', 'forcing', 'motor_rate',
                   'muscle_activation', 'command', 'tendon')
PROCESS_LIMIT_S = 20.
PRIOR_PROCESS_S = 701.
SHARED_PROCESS_LIMIT_S = 731.
TIE_RELATIVE_TOLERANCE = 1e-10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    allow_nan=False, indent=2), encoding='utf-8')


def compose_solver_weights(base, rows, cols, deltas):
    """Extend actual E113 float32 W to float64, modifying selected edges only."""
    base = np.asarray(base, dtype=np.float64)
    rows, cols = np.asarray(rows), np.asarray(cols)
    deltas = np.asarray(deltas, dtype=np.float64)
    if base.ndim != 2 or rows.ndim != 1 or rows.shape != cols.shape or \
            deltas.ndim != 2 or deltas.shape[1] != len(rows) or \
            not np.issubdtype(rows.dtype, np.integer) or \
            not np.issubdtype(cols.dtype, np.integer) or \
            not np.isfinite(base).all() or not np.isfinite(deltas).all() or \
            np.any(rows < 0) or np.any(rows >= base.shape[0]) or \
            np.any(cols < 0) or np.any(cols >= base.shape[1]) or \
            len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError('Aligned finite distinct selected W entries required')
    weights = np.repeat(base[None], len(deltas), axis=0)
    weights[:, rows, cols] = base[rows, cols][None]*np.exp(deltas)
    return weights


def reviewed_inputs(root):
    """Read-only frozen source/input/step checks; no geometry or response."""
    root = Path(root)
    require(sha256(ROOT/'docs/BANC_M3_E117_NONLINEAR_STEPS_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(root/'e117-frozen-design.json') == FREEZE_SHA,
            'E117 frozen design changed')
    freeze = json.loads((root/'e117-frozen-design.json').read_text(
        encoding='utf-8'))
    require(freeze['actual_execution_authorized'] is False and
            freeze['pair_context_indices'] == PAIR_CONTEXT.tolist() and
            freeze['pair_weight_indices'] == PAIR_WEIGHT.tolist() and
            freeze['variant_names'] == list(VARIANTS) and
            freeze['CUDA_batches'] == 1 and freeze['conditional_responses'] == 6
            and freeze['conditional_seconds'] == 60 and
            freeze['internal_wall_s'] == PROCESS_LIMIT_S and
            freeze['prior_conservative_process_s'] == PRIOR_PROCESS_S and
            freeze['shared_process_limit_s'] == SHARED_PROCESS_LIMIT_S and
            freeze['baseline_loss_parts_tolerance'] == 1e-8 and
            freeze['baseline_startup_ratios_tolerance'] == 1e-8 and
            freeze['max_sample_tie_relative_tolerance'] == TIE_RELATIVE_TOLERANCE,
            'E117 fixed scope/resource/pair order changed')
    for relative, expected in freeze['input_sha256'].items():
        require(sha256(root/relative) == expected,
                f'E117 frozen input changed: {relative}')
    for relative, expected in freeze['source_sha256'].items():
        require(sha256(ROOT/relative) == expected,
                f'E117 reused source changed: {relative}')
    review = json.loads((root/'e116-startup-tangent/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    step_report = json.loads((root/'parent-e117-step-inputs.json').read_text(
        encoding='utf-8'))
    require(review['accepted_as_valid_affine_diagnostic'] is True and
            review['blocking_defects'] == [] and
            step_report['passed'] is True and
            all(step_report['checks'].values()) and
            step_report['output_sha256'] == freeze['input_sha256'][
                'parent-e117-step-inputs.npz'],
            'Parent E116/step acceptance changed')
    data = e115_inputs(root)
    fixture = data['fixture']
    step = load_npz(root/'parent-e117-step-inputs.npz')
    selected = load_npz(root/'e115-context-trust-fit/selected-delta.npz')
    mask = data['independent']['leg_edge_mask']
    rows, cols = fixture['edge_rows'], fixture['edge_cols']
    require(step['variant_names'].tolist() == list(VARIANTS) and
            step['deltas'].shape == step['increments'].shape == (3, 1572)
            and step['weights'].shape == (3, 391, 49) and
            step['lower'].shape == step['upper'].shape == (1572,) and
            step['leg_edge_mask'].shape == (6, 1572) and
            np.array_equal(step['pair_context_indices'], PAIR_CONTEXT) and
            np.array_equal(step['pair_weight_indices'], PAIR_WEIGHT) and
            np.array_equal(step['selected_edge_ids'], fixture['edge_ids']) and
            np.array_equal(step['edge_rows'], rows) and
            np.array_equal(step['edge_cols'], cols) and
            np.array_equal(step['leg_edge_mask'], mask) and
            np.array_equal(step['lower'], data['bounds'][:, 0]) and
            np.array_equal(step['upper'], data['bounds'][:, 1]) and
            np.array_equal(step['deltas'][0], selected['delta']) and
            np.array_equal(step['increments'][0], np.zeros(1572)) and
            all(np.isfinite(step[key]).all() for key in
                ('deltas', 'increments', 'weights', 'lower', 'upper')) and
            np.all(step['deltas'] >= step['lower'][None]) and
            np.all(step['deltas'] <= step['upper'][None]) and
            np.all(np.abs(step['increments']) <= .01),
            'Frozen E117 three-delta/edge/bounds data changed')
    for variant in (1, 2):
        require(np.array_equal(step['deltas'][variant],
                               selected['delta']+step['increments'][variant]),
                'E116 h was projected or altered')
    affine_losses = np.zeros((3, 6, 2), dtype=np.float64)
    affine_ratios = np.zeros((3, 6, 2, 4), dtype=np.float64)
    baseline = {}
    for leg_index, leg in enumerate(LEGS):
        selection = json.loads((root/f'e115-context-trust-fit/selection-{leg}.json')
                               .read_text(encoding='utf-8'))
        index = selection['best_feasible']['index']
        raw = load_npz(root/f'e115-context-trust-fit/eval-{leg}-{index:04d}.npz')
        edges = np.flatnonzero(mask[leg_index])
        require(np.array_equal(step['deltas'][0, edges], raw['delta']) and
                np.array_equal(step['deltas'][0, edges],
                               np.asarray(selection['best_feasible']['delta'])),
                f'E115 selected_solver origin changed: {leg}')
        baseline[leg] = raw
        affine_losses[0, leg_index] = raw['losses']
        affine_ratios[0, leg_index] = raw['ratios']
        for variant, label in enumerate(VARIANTS[1:], start=1):
            result = load_npz(root/f'e116-startup-tangent/lp-{leg}-{label}-returned.npz')
            h = result['x'][:-1]
            diagnostic = json.loads((root/f'e116-startup-tangent/'
                f'lp-{leg}-{label}-diagnostic.json').read_text(encoding='utf-8'))
            require(h.shape == (len(edges),) and
                np.array_equal(step['increments'][variant, edges], h),
                f'E116 saved h changed: {leg}/{label}')
            # All six disjoint leg increments contribute to each variant.
            require(np.allclose(np.asarray(diagnostic['predicted_losses']),
                                raw['losses']+raw['jacobian'][:2]@h,
                                rtol=0, atol=1e-12) and
                    np.allclose(np.asarray(diagnostic['predicted_startup_ratios']),
                                (raw['ratios'].ravel()+raw['jacobian'][2:]@h)
                                .reshape(2, 4), rtol=0, atol=1e-12),
                    f'E116 affine prediction changed: {leg}/{label}')
            affine_losses[variant, leg_index] = diagnostic['predicted_losses']
            affine_ratios[variant, leg_index] = diagnostic[
                'predicted_startup_ratios']
    for variant, label in enumerate(VARIANTS[1:], start=1):
        expected_h = np.zeros(1572)
        for leg_index, leg in enumerate(LEGS):
            edges = np.flatnonzero(mask[leg_index])
            returned = load_npz(root/f'e116-startup-tangent/'
                f'lp-{leg}-{label}-returned.npz')
            expected_h[edges] = returned['x'][:-1]
        require(np.array_equal(step['increments'][variant], expected_h),
                f'E116 composite h changed: {label}')
    expected_weights = compose_solver_weights(fixture['weights'][1], rows,
                                              cols, step['deltas'])
    require(np.array_equal(step['weights'], expected_weights),
            'E117 weights differ from actual E113 W times exp(delta)')
    data.update(root=root, step=step, freeze=freeze,
                baseline=baseline, affine_losses=affine_losses,
                affine_ratios=affine_ratios)
    return data


def stage_response_inputs(fixture, step, device, synchronize=lambda: None):
    """Finish all five device transfers before any response charge."""
    source = torch.as_tensor(fixture['source_rate'], dtype=torch.float64,
                             device=device)
    background = torch.as_tensor(fixture['background'], dtype=torch.float64,
                                 device=device)
    weights = torch.as_tensor(step['weights'], dtype=torch.float64,
                              device=device)
    contexts = torch.as_tensor(step['pair_context_indices'], dtype=torch.long,
                               device=device)
    indices = torch.as_tensor(step['pair_weight_indices'], dtype=torch.long,
                              device=device)
    synchronize()
    return source, background, weights, contexts, indices


def detach_available(output, raw, primary=None):
    if output is None:
        return
    for key, value in output.items():
        if key in raw:
            continue
        try:
            raw[key] = value.detach().cpu().numpy().copy()
        except Exception as secondary:
            if primary is None:
                raise
            primary.add_note(f'E117 secondary {key} materialization: {secondary}')


def rescue_raw(directory, values, primary, label):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as secondary:  # noqa: BLE001
        primary.add_note(f'E117 secondary {label} directory: {secondary}')
        return
    for key, value in values.items():
        try:
            np.save(directory/f'{key}.npy', value, allow_pickle=False)
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E117 secondary {label}.{key} save: {secondary}')


def materialize_feet(tensor, selected_count):
    """Copy the one geometry result to host; this never reevaluates geometry."""
    return tensor.detach().cpu().numpy().copy().reshape(
        6, selected_count, 6, 3)


def rescue_available_feet(tensor, feet_fixed, selected_count, out, primary):
    if feet_fixed is None and tensor is not None:
        try:
            feet_fixed = materialize_feet(tensor, selected_count)
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E117 secondary geometry materialization: {secondary}')
    if feet_fixed is not None:
        rescue_raw(out/'error-available/geometry', {'feet_fixed': feet_fixed},
                   primary, 'geometry')


def execute_response_batch(stage, inputs, fixture, step, out, ledger, persist,
                           deadline, synchronize=lambda: None):
    """One charged batch; save complete raw before any geometry or gate."""
    output = None
    # These inputs are already known before the charged response. Keep them in
    # every rescue, including failures in materialization or its receipt.
    raw = {'pair_context_indices': step['pair_context_indices'].copy(),
        'pair_weight_indices': step['pair_weight_indices'].copy(),
        'variant_names': step['variant_names'].copy(),
        'weights': step['weights'].copy(), 'time_s': fixture['time_s'].copy()}
    deadline()
    ledger.update(stage='response', batch_started=1, pairs_started=6,
        conditional_model_s_started=60., batch_reservation_requested=6)
    try:
        persist()
    except BaseException as primary:
        ledger.update(batch_started=0, pairs_started=0,
            conditional_model_s_started=0., batch_start_receipt_failed=True)
        rescue_raw(out/'error-response-input', {
            'source_rate': fixture['source_rate'],
            'background': fixture['background'], **raw}, primary,
            'response input')
        raise
    try:
        ledger['response_call_entered'] = True
        output = stage.evaluate(*inputs)
        detach_available(output, raw)
        synchronize()
        ledger.update(stage='response_materialized', batch_materialized=1,
            pairs_materialized=6, conditional_model_s_materialized=60.)
        persist()
        np.savez_compressed(out/'response-raw.npz', **raw)
        ledger.update(stage='response_raw_saved', batch_raw_saved=1,
            pairs_raw_saved=6, conditional_model_s_raw_saved=60.)
        persist()
        return raw
    except BaseException as primary:
        detach_available(output, raw, primary)
        rescue_raw(out/'error-response-raw', raw, primary, 'response')
        raise


def leg_metrics(raw, feet_fixed, data):
    """E115 normalized four-part loss and original99 four startup statistics."""
    f = data['fixture']
    target_feet = data['independent']['common_target_feet']
    selected = np.flatnonzero(f['fixed_mask'])
    common = np.flatnonzero(f['common_mask'])
    startup = np.flatnonzero(f['startup_mask'])
    require(raw['command'].shape == (6, 2001, 42) and
            raw['tendon'].shape == (6, 2001, 6) and
            feet_fixed.shape == (6, len(selected), 6, 3) and
            target_feet.shape == (2, len(common), 6, 3) and
            np.array_equal(startup, np.arange(1, 100)) and
            np.array_equal(common, np.arange(100, 1923)),
            'E117 response, geometry or frozen time axes changed')
    feet_common = feet_fixed[:, f['common_mask'][selected]]
    parts = np.empty((3, 6, 2, 4), dtype=np.float64)
    ratios = np.empty_like(parts)
    startup_values = np.empty_like(parts)
    for variant in range(3):
        for context in range(2):
            pair = 2*variant+context
            command, tendon = raw['command'][pair], raw['tendon'][pair]
            for leg in range(6):
                joints = slice(7*leg, 7*leg+7)
                dq = command[common, joints]-f['own_target'][
                    context, common, joints]
                df = feet_common[pair, :, leg]-target_feet[context, :, leg]
                dv = np.diff(df, axis=0)/.005
                ltm = tendon[common, leg]-.2
                parts[variant, leg, context] = [
                    np.mean((dq/f['joint_scale'][joints])**2),
                    np.mean((df/f['foot_scale'][leg])**2),
                    np.mean((dv/f['velocity_scale'][leg])**2),
                    np.mean((ltm/.2)**2)]
                start_q = command[startup, joints]-f[
                    'startup_target_command'][:, joints]
                start_ltm = tendon[startup, leg]-f[
                    'startup_target_tendon'][:, leg]
                values = np.array([np.sqrt(np.mean(start_q**2)),
                    np.max(np.abs(start_q)), np.sqrt(np.mean(start_ltm**2)),
                    np.max(np.abs(start_ltm))])
                startup_values[variant, leg, context] = values
                ratios[variant, leg, context] = (values/INNER)**2
    return {'loss_parts': parts, 'losses': parts.sum(axis=-1),
        'startup_values': startup_values, 'startup_ratios': ratios,
        'variant_names': np.asarray(VARIANTS),
        'pair_context_indices': PAIR_CONTEXT.copy(),
        'pair_weight_indices': PAIR_WEIGHT.copy()}


def baseline_parity(metrics, data):
    loss_error = np.empty((6, 2, 4), dtype=np.float64)
    ratio_error = np.empty((6, 2, 4), dtype=np.float64)
    for leg_index, leg in enumerate(LEGS):
        origin = data['baseline'][leg]
        loss_error[leg_index] = metrics['loss_parts'][0, leg_index]-origin['parts']
        ratio_error[leg_index] = metrics['startup_ratios'][0, leg_index]-origin[
            'ratios']
    passed = bool(np.isfinite(loss_error).all() and
        np.isfinite(ratio_error).all() and
        np.max(np.abs(loss_error)) <= 1e-8 and
        np.max(np.abs(ratio_error)) <= 1e-8)
    return {'loss_parts_difference': loss_error,
        'startup_ratios_difference': ratio_error,
        'loss_parts_max_abs': float(np.max(np.abs(loss_error))),
        'startup_ratios_max_abs': float(np.max(np.abs(ratio_error))),
        'limit': 1e-8, 'passed': passed}


def peak_decomposition(baseline_error, trial_error, affine_ratio, *,
                       threshold, startup_indices, time_s, leg, metric):
    """Exact max-switch plus fixed-baseline-sample affine-gap identity."""
    baseline_error = np.asarray(baseline_error, dtype=np.float64)
    trial_error = np.asarray(trial_error, dtype=np.float64)
    indices = np.asarray(startup_indices, dtype=np.int64)
    if baseline_error.shape != trial_error.shape or \
            baseline_error.shape[0] != len(indices) or \
            baseline_error.ndim not in (1, 2) or \
            not all(np.isfinite(x).all() for x in
                    (baseline_error, trial_error)) or \
            not np.isfinite(affine_ratio) or threshold <= 0:
        raise ValueError('Finite aligned startup peak operands required')
    baseline_sq = (baseline_error/threshold)**2
    trial_sq = (trial_error/threshold)**2
    baseline_flat = int(np.argmax(baseline_sq))
    trial_flat = int(np.argmax(trial_sq))
    base_index = np.unravel_index(baseline_flat, baseline_sq.shape)
    trial_index = np.unravel_index(trial_flat, trial_sq.shape)
    base_max = float(baseline_sq[base_index])
    trial_max = float(trial_sq[trial_index])
    r_same = float(trial_sq[base_index])
    r_affine = float(affine_ratio)
    switch = trial_max-r_same
    same_affine = r_same-r_affine
    total = trial_max-r_affine
    def sample(index):
        row = int(index[0])
        position = {'startup_row': row, 'full_time_index': int(indices[row]),
            'time_s': float(time_s[indices[row]]), 'leg': LEGS[leg]}
        if metric == 'qmax':
            position['local_joint'] = int(index[1])
            position['global_joint'] = 7*leg+int(index[1])
        return position
    return {'metric': metric, 'baseline_max_sample': sample(base_index),
        'trial_max_sample': sample(trial_index),
        'baseline_max_signed_error': float(baseline_error[base_index]),
        'trial_same_sample_signed_error': float(trial_error[base_index]),
        'trial_max_signed_error': float(trial_error[trial_index]),
        'baseline_max_squared_ratio': base_max,
        'trial_same_squared_ratio': r_same,
        'trial_max_squared_ratio': trial_max,
        'affine_predicted_squared_ratio': r_affine,
        'max_sample_switch_term': switch,
        'same_sample_affine_gap_term': same_affine,
        'total_nonlinear_minus_affine': total,
        'identity_residual': total-(switch+same_affine),
        'baseline_tie_count': int(np.count_nonzero(np.abs(
            baseline_sq-base_max) <= TIE_RELATIVE_TOLERANCE*max(1., abs(base_max)))),
        'trial_tie_count': int(np.count_nonzero(np.abs(
            trial_sq-trial_max) <= TIE_RELATIVE_TOLERANCE*max(1., abs(trial_max)))),
        'argmax_changed': bool(base_index != trial_index),
        'baseline_tie_limits_curvature_interpretation': bool(np.count_nonzero(
            np.abs(baseline_sq-base_max) <= TIE_RELATIVE_TOLERANCE*
            max(1., abs(base_max))) > 1)}


def all_peak_decompositions(raw, metrics, data):
    f = data['fixture']
    indices = np.flatnonzero(f['startup_mask'])
    reports = {}
    for variant in (1, 2):
        label = VARIANTS[variant]
        reports[label] = {}
        for leg, name in enumerate(LEGS):
            reports[label][name] = {}
            joints = slice(7*leg, 7*leg+7)
            for context in range(2):
                baseline_pair = context
                trial_pair = 2*variant+context
                baseline_q = raw['command'][baseline_pair, indices, joints]-f[
                    'startup_target_command'][:, joints]
                trial_q = raw['command'][trial_pair, indices, joints]-f[
                    'startup_target_command'][:, joints]
                baseline_ltm = raw['tendon'][baseline_pair, indices, leg]-f[
                    'startup_target_tendon'][:, leg]
                trial_ltm = raw['tendon'][trial_pair, indices, leg]-f[
                    'startup_target_tendon'][:, leg]
                reports[label][name][str(f['context_names'][context])] = {
                    'qmax': peak_decomposition(baseline_q, trial_q,
                        data['affine_ratios'][variant, leg, context, 1],
                        threshold=INNER[1], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='qmax'),
                    'LTMmax': peak_decomposition(baseline_ltm, trial_ltm,
                        data['affine_ratios'][variant, leg, context, 3],
                        threshold=INNER[3], startup_indices=indices,
                        time_s=f['time_s'], leg=leg, metric='LTMmax')}
    return reports


def validate_peak_records(peaks, metrics):
    """Check every saved decomposition against nonlinear squared-max ratios."""
    checked = 0
    for variant in (1, 2):
        label = VARIANTS[variant]
        for leg, name in enumerate(LEGS):
            for context, context_name in enumerate(('E110', 'E113')):
                rows = peaks[label][name][context_name]
                for metric, component in (('qmax', 1), ('LTMmax', 3)):
                    row = rows[metric]
                    actual = metrics['startup_ratios'][
                        variant, leg, context, component]
                    baseline = metrics['startup_ratios'][
                        0, leg, context, component]
                    require(abs(row['trial_max_squared_ratio']-actual) <=
                        1e-12*(1+abs(actual)) and
                        abs(row['baseline_max_squared_ratio']-baseline) <=
                        1e-12*(1+abs(baseline)) and
                        abs(row['identity_residual']) <= 1e-12*(1+abs(
                            row['total_nonlinear_minus_affine'])) and
                        row['max_sample_switch_term'] >= -1e-12,
                        f'E117 peak ratio/decomposition failed: '
                        f'{label}/{name}/{context_name}/{metric}')
                    checked += 1
    return checked


def affine_comparison(metrics, data):
    return {'actual_losses': metrics['losses'],
        'affine_predicted_losses': data['affine_losses'],
        'actual_minus_affine_losses': metrics['losses']-data['affine_losses'],
        'actual_startup_ratios': metrics['startup_ratios'],
        'affine_predicted_startup_ratios': data['affine_ratios'],
        'actual_minus_affine_startup_ratios':
            metrics['startup_ratios']-data['affine_ratios'],
        'actual_startup_inner_pass': np.all(
            metrics['startup_ratios'] <= SELECTOR_CAP, axis=(2, 3)),
        'actual_startup_original_pass': np.all(
            metrics['startup_values'] <= ORIGINAL, axis=(2, 3))}


def run_once(root, out):
    """One separately authorized CUDA batch; preserve raw before all gates."""
    root, out = Path(root), Path(out)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'experiment': 'E117-nonlinear-steps', 'stage': 'created',
        'prior_conservative_process_s': PRIOR_PROCESS_S,
        'shared_process_limit_s': SHARED_PROCESS_LIMIT_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'batch_started': 0, 'batch_materialized': 0, 'batch_raw_saved': 0,
        'pairs_started': 0, 'pairs_materialized': 0, 'pairs_raw_saved': 0,
        'conditional_model_s_started': 0.,
        'conditional_model_s_materialized': 0.,
        'conditional_model_s_raw_saved': 0.,
        'response_call_entered': False, 'batch_reservation_requested': 0,
        'geometry_call_entered': False, 'geometry_materialized': False,
        'geometry_raw_saved': False, 'metrics_raw_saved': False,
        'LP_calls': 0, 'fit_calls': 0, 'calibration_outputs': 0,
        'native_network_calls': 0, 'body_calls': 0,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'secondary_errors': []}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E117 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
        except Exception as secondary:
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E117 secondary clock failure: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadline():
        elapsed = time.monotonic()-started
        if elapsed > PROCESS_LIMIT_S or \
                PRIOR_PROCESS_S+elapsed > SHARED_PROCESS_LIMIT_S:
            raise TimeoutError('E117 internal/shared wall limit exceeded')

    def failure_save(filename, value, primary):
        try:
            save_json(out/filename, value)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E117 secondary {filename} save: {secondary}')

    primary = None
    raw = {}
    feet_tensor = None
    selected_count = None
    feet_fixed = None
    metrics = None
    peaks = None
    try:
        for original, name in (
            (ROOT/'tools/verify_c_banc_m3_e117_nonlinear_steps.py', 'runner.py'),
            (ROOT/'tests/test_c_banc_nonlinear_steps.py',
             'test_c_banc_nonlinear_steps.py'),
            (ROOT/'docs/BANC_M3_E117_NONLINEAR_STEPS_DESIGN_20260920.md',
             'frozen-design.md'),
            (root/'e117-frozen-design.json', 'e117-frozen-design.json'),
            (root/'parent-e117-step-inputs.json', 'parent-e117-step-inputs.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        data = reviewed_inputs(root)
        f, step = data['fixture'], data['step']
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
        geometry = TorchFootGeometry.load(
            root/'e111-cuda-foot-geometry/geometry.npz', device='cuda:0')
        require(geometry.geometry_hash == GEOMETRY_HASH,
                'Original E32 fixed geometry hash changed')
        stage = ContextResponse(time_s=f['time_s'], tau=f['tau'], a=f['a'],
            threshold=f['threshold'], cap=f['cap'], pool=f['pool'],
            moment=f['moment'], rest=f['rest'], ltm=f['ltm'], device='cuda:0')
        inputs = stage_response_inputs(f, step, 'cuda:0', torch.cuda.synchronize)
        deadline()
        raw = execute_response_batch(stage, inputs, f, step, out,
            ledger, persist, deadline, torch.cuda.synchronize)
        deadline()
        selected = np.flatnonzero(f['fixed_mask'])
        selected_count = len(selected)
        command = torch.as_tensor(raw['command'][:, selected].reshape(-1, 42),
                                  dtype=torch.float64, device='cuda:0')
        torch.cuda.synchronize()
        deadline()
        ledger['geometry_call_entered'] = True
        feet_tensor = geometry.full(command)
        feet_fixed = materialize_feet(feet_tensor, selected_count)
        torch.cuda.synchronize()
        ledger['geometry_materialized'] = True
        np.savez_compressed(out/'foot-proxy-raw.npz', feet_fixed=feet_fixed,
            fixed_mask=f['fixed_mask'], common_mask=f['common_mask'],
            own_mask=f['own_mask'], pair_context_indices=PAIR_CONTEXT,
            pair_weight_indices=PAIR_WEIGHT)
        ledger['geometry_raw_saved'] = True
        ledger['stage'] = 'geometry_raw_saved'
        persist()
        deadline()
        metrics = leg_metrics(raw, feet_fixed, data)
        np.savez_compressed(out/'metrics-raw.npz', **metrics,
            affine_losses=data['affine_losses'],
            affine_startup_ratios=data['affine_ratios'])
        ledger['metrics_raw_saved'] = True
        ledger['stage'] = 'metrics_raw_saved'
        persist()
        # Baseline parity and peak identities are asserted only after all raw.
        parity = baseline_parity(metrics, data)
        comparison = affine_comparison(metrics, data)
        peaks = all_peak_decompositions(raw, metrics, data)
        save_json(out/'baseline-parity.json', parity)
        save_json(out/'affine-comparison.json', comparison)
        save_json(out/'peak-decomposition.json', peaks)
        screens = {}
        for variant, name in enumerate(VARIANTS):
            pair_raw = {key: raw[key][2*variant:2*variant+2]
                        for key in RESPONSE_FIELDS}
            screens[name] = frozen_output_metrics(pair_raw,
                feet_fixed[2*variant:2*variant+2], data)
        save_json(out/'screens.json', screens)
        save_json(out/'provenance.json', {
            'design_sha256': DESIGN_SHA, 'freeze_sha256': FREEZE_SHA,
            'input_sha256': data['freeze']['input_sha256'],
            'source_sha256': data['freeze']['source_sha256'],
            'runner_sha256': sha256(ROOT/
                'tools/verify_c_banc_m3_e117_nonlinear_steps.py'),
            'test_sha256': sha256(ROOT/'tests/test_c_banc_nonlinear_steps.py'),
            'geometry_hash': geometry.geometry_hash,
            'weights_semantics': 'float64 E113 actual-W times exp(saved solver delta)',
            'background_semantics': 'recorded own-W residual held fixed',
            'foot_semantics': 'fixed E32 command geometry proxy'})
        require(parity['passed'], 'E117 selected_solver/E115 saved eval mismatch')
        require(validate_peak_records(peaks, metrics) == 48,
                'E117 peak record count changed')
        require(np.isfinite(metrics['loss_parts']).all() and
                np.isfinite(metrics['startup_ratios']).all(),
                'Nonfinite E117 nonlinear loss/startup metrics')
        deadline()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_nonlinear_saved_step_diagnostic',
            'candidate_adopted': False, 'native_gait_tested': False,
            'calibration_output': False, 'selected_solver_parity': parity['passed'],
            'variant_screens': {key: value['robust_local_supported']
                                for key, value in screens.items()},
            'no_biological_capacity_conclusion': True,
            'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', ledger['primary_error'], error)
        # Never reevaluate. Save every in-memory result independently.
        available = {key: value for key, value in raw.items()
                     if isinstance(value, np.ndarray)}
        if available and ledger['batch_raw_saved'] == 0:
            rescue_raw(out/'error-available/response', available,
                       error, 'response')
        if (feet_tensor is not None or feet_fixed is not None) and not ledger[
                'geometry_raw_saved']:
            rescue_available_feet(feet_tensor, feet_fixed, selected_count,
                                  out, error)
        if metrics is not None and not ledger['metrics_raw_saved']:
            rescue_raw(out/'error-available/metrics', metrics,
                       error, 'metrics')
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
        'verification/banc-natural-gait-m3-20260919/e117-nonlinear-steps')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
