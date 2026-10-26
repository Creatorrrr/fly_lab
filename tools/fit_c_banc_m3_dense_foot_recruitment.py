"""E112b: audited preflight recovery of the single E112 conditional fit.

This never creates a FlyBody, advances a neural/physical clock, or installs a
controller. The parent-reviewed single CUDA run requires separate approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
from tools.fit_c_banc_m3_block_constrained_recruitment import (
    absolute_shape_screen,
    leg_partition,
)
from tools.fit_c_banc_m3_constrained_startup import native_shape_screen, startup_screen
from tools.verify_c_banc_observation import exact

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
MOTOR_COUNTS = (69, 63, 63, 70, 63, 63)
EDGE_COUNTS = (273, 281, 240, 238, 255, 285)
INNER = np.array([.018, .04, .009, .025], dtype=np.float64)
ORIGINAL = np.array([.02, .05, .01, .03], dtype=np.float64)
OPTIMIZER_CAP = (1-2e-4)**2
SELECTOR_CAP = (1-1e-4)**2
LEG_LIMIT_S = 24.
LEG_MAX_CALLS = 1000
FIT_LIMIT_S = 150.
PROCESS_LIMIT_S = 180.
PRIOR_S = 326.
SHARED_LIMIT_S = 506.
EXPECTED_INPUT_SHA = '47ab37cb492afa2b6d5ca20dc5fbf0a8429167cea401c20c43aa17f7e89f999c'
EXPECTED_DESIGN_SHA = '7a55c6a6c0e5b8a206a7033a71ac4d049c4f59e05310e05d20b077bfb4488eaf'
EXPECTED_GEOMETRY_HASH = '36a47d2f0f0d4e83d5ff8ecf1af95b9ce63269d5158d98642b4f89e0bc230953'
EXPECTED_RECOVERY_SHA = '724ce524271b0d4212507d8652f7443bae81e5d3ab5f0f4805156446ea64bf57'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2,
        ensure_ascii=False, allow_nan=False), encoding='utf-8')


def try_save(path, value):
    try:
        save_json(path, value)
        return None
    except Exception as error:  # noqa: BLE001 - retain earlier experimental error
        return {'type': type(error).__name__, 'message': str(error)}


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(jsonable(value), ensure_ascii=False,
                                allow_nan=False)+'\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def audit_background_roundoff(current, source_rate, weights, background, out):
    """Check recomputed algebra across BLAS reductions, never change stored data.

    Both dot/subtract evaluations have error <= gamma_(n+1) times the sum
    of absolute terms. Use eps (twice unit roundoff), bound both reductions,
    and inflate for rounding while computing that sum. This applies only to
    finite normal-range float64 inputs/products; exact archive hashes and the
    stored-background byte comparison remain separate mandatory checks.
    """
    values = (current, source_rate, weights, background)
    if any(not isinstance(x, np.ndarray) or x.dtype != np.float64 or x.ndim != 2
           for x in values) or current.shape != background.shape or \
            current.shape != (len(source_rate), len(weights)) or \
            source_rate.shape[1] != weights.shape[1] or not source_rate.shape[1]:
        raise ValueError('Compatible nonempty float64 background operands required')
    info = np.finfo(np.float64)
    k = source_rate.shape[1]+1
    if k*info.eps >= 1:
        raise ValueError('Background dot product exceeds roundoff bound domain')
    gamma = k*info.eps/(1-k*info.eps)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        recomputed = current-source_rate@weights.T
        sum_abs = np.abs(current)+np.abs(source_rate)@np.abs(weights).T
        bound = np.nextafter((2*gamma/(1-gamma))*sum_abs, np.inf)
        difference = np.abs(recomputed-background)
        minima = [float(np.min(np.abs(x[x != 0]))) if np.any(x != 0)
                  else 1. for x in values]
        normal_range = all(x >= info.tiny for x in minima) and \
            minima[1]*minima[2] >= info.tiny
    # Persist evidence before evaluating the numerical acceptance conditions.
    np.savez_compressed(out/'background-roundoff-raw.npz',
        recomputed_background=recomputed, stored_background=background,
        absolute_difference=difference, absolute_term_sum=sum_abs, bound=bound)
    finite = all(np.isfinite(x).all() for x in (*values, recomputed, sum_abs,
                                               bound, difference))
    passed = bool(finite and normal_range and np.all(difference <= bound))
    report = {'passed': passed, 'finite': finite, 'normal_range': bool(normal_range),
        'dot_terms': k-1, 'eps': info.eps, 'gamma': gamma,
        'formula': 'nextafter(2*gamma/(1-gamma)*(abs(current)+abs(source)@abs(W).T),+inf)',
        'max_absolute_difference': float(np.max(difference)),
        'bit_unequal_elements': int(np.count_nonzero(recomputed != background)),
        'max_bound': float(np.max(bound)),
        'bound_exceedances': int(np.count_nonzero(difference > bound)),
        'stored_background_replaced': False,
        'source': 'https://nhigham.com/wp-content/uploads/2023/10/high99n.pdf#page=4'}
    save_json(out/'background-roundoff.json', report)
    if not passed:
        raise ValueError('Fixed E110 background algebra exceeds float64 roundoff bound')
    return report


def adjacent_velocity(points, selected, times, *, dt=.005):
    """All adjacent endpoint displacements; no gaps, wrapping, or row skipping."""
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or \
            points.shape[1] != 3 or points.dtype != torch.float64 or \
            points.shape[0] < 2:
        raise ValueError('Finite float64 [T>=2,3] foot positions required')
    selected = np.asarray(selected)
    times = np.asarray(times)
    if selected.shape != (len(points),) or selected.dtype.kind not in 'iu' or \
            times.ndim != 1 or np.any(selected < 0) or \
            np.any(selected >= len(times)) or \
            not np.array_equal(np.diff(selected), np.ones(len(points)-1, dtype=int)) or \
            not np.isfinite(times).all() or not math.isfinite(dt) or dt <= 0 or \
            not np.allclose(np.diff(times[selected]), dt, rtol=0, atol=1e-12):
        raise ValueError('Complete contiguous 5ms selected endpoint grid required')
    if not bool(torch.isfinite(points).all()):
        raise ValueError('Nonfinite foot positions')
    return (points[1:]-points[:-1])/dt


def startup_ratios(command, tendon, target_q, target_ltm, limits=INNER):
    """Four squared original99 single-leg budgets, without mean-max dilution."""
    q = command-target_q
    ltm = tendon-target_ltm
    if q.shape != (99, 7) or ltm.shape != (99,) or \
            not torch.isfinite(q).all() or not torch.isfinite(ltm).all():
        raise ValueError('Finite original99 single-leg outputs required')
    return torch.stack((q.square().mean()/limits[0]**2,
        q.square().max()/limits[1]**2,
        ltm.square().mean()/limits[2]**2,
        ltm.square().max()/limits[3]**2))


def selector_feasible(ratios):
    values = np.asarray(ratios, dtype=np.float64)
    return bool(values.shape == (4,) and np.isfinite(values).all()
                and np.all(values <= SELECTOR_CAP))


def choose_records(records):
    finite = [record for record in records
              if math.isfinite(float(record['objective']))]
    best_any = min(finite, key=lambda record: record['objective']) if finite else None
    feasible = [record for record in finite if selector_feasible(record['ratios'])]
    best_feasible = min(feasible, key=lambda record: record['objective']) if feasible else None
    return best_any, best_feasible


class FitLimit(Exception):
    pass


class EvaluationCache:
    """Count one bit-exact finite log vector once, before starting evaluation."""

    def __init__(self, evaluate, size, on_start, *, max_calls, deadline,
                 total_deadline):
        self.evaluate = evaluate
        self.size = size
        self.on_start = on_start
        self.max_calls = max_calls
        self.deadline = deadline
        self.total_deadline = total_deadline
        self.cache = {}
        self.started = 0
        self.completed = 0
        self.hits = 0

    def get(self, vector):
        x = np.asarray(vector, dtype=np.float64)
        if x.shape != (self.size,) or not np.isfinite(x).all():
            raise ValueError('Finite full log-efficacy vector required')
        key = x.tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        now = time.perf_counter()
        if self.started >= self.max_calls or now >= self.deadline-.5:
            raise FitLimit('per_leg_call_or_wall_limit')
        if now >= self.total_deadline-.5:
            raise FitLimit('whole_fit_wall_limit')
        self.started += 1
        self.on_start(self.started, x)
        result = self.evaluate(x, self.started)
        self.cache[key] = result
        self.completed += 1
        return result


def optimizer_record(result, stop):
    if result is None:
        return {'stop': stop, 'success': None, 'status': None, 'message': None,
                'nit': None, 'nfev': None, 'njev': None, 'objective': None,
                'gradient_max_abs': None, 'log_theta': None}
    grad = np.asarray(result.get('jac', []), dtype=np.float64)
    fun = float(result.get('fun', np.nan))
    return {'stop': stop if math.isfinite(fun) and np.isfinite(grad).all()
                    else 'scipy_nonfinite_result',
            'success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'nit': int(result.nit),
            'nfev': int(result.nfev), 'njev': int(result.njev),
            'objective': fun, 'nonfinite_objective_text': None if math.isfinite(fun)
                else str(fun),
            'gradient_nonfinite_counts': {'nan': int(np.isnan(grad).sum()),
                'positive_infinity': int(np.isposinf(grad).sum()),
                'negative_infinity': int(np.isneginf(grad).sum())},
            'gradient_max_abs': float(np.max(np.abs(grad)))
                if grad.size and np.isfinite(grad).all() else None,
            'gradient': grad.tolist(), 'log_theta': np.asarray(result.x).tolist()}


def optimize_block(cache, initial):
    """One SLSQP start; objective and four exact autograd constraints share cache."""
    result = minimize(lambda x: cache.get(x)['objective'], initial,
        method='SLSQP', jac=lambda x: cache.get(x)['jac'],
        bounds=[(math.log(.001), math.log(10.))]*len(initial),
        constraints=[{'type': 'ineq',
            'fun': lambda x: OPTIMIZER_CAP-cache.get(x)['ratios'],
            'jac': lambda x: -cache.get(x)['ratio_jac']}],
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    return result, optimizer_record(result, 'scipy_finished')


def scalar_metrics(command, feet, velocity, tendon, inputs):
    selected = inputs['selected']
    target = inputs['target']
    target_feet = inputs['target_feet']
    target_velocity = inputs['target_velocity']
    dq = command[selected]-target
    df = feet-target_feet
    dv = velocity-target_velocity
    dl = tendon[selected]-.2
    result = {'q_rms': float(np.sqrt(np.mean(dq**2))),
        'foot_rms': float(np.sqrt(np.mean(df**2))),
        'velocity_rms': float(np.sqrt(np.mean(dv**2))),
        'LF_row6_rms': float(np.sqrt(np.mean(dq[:, 6]**2))),
        'per_leg_q_rms': [float(np.sqrt(np.mean(dq[:, 7*i:7*i+7]**2)))
                          for i in range(6)],
        'per_leg_foot_rms': [float(np.sqrt(np.mean(df[:, i]**2)))
                             for i in range(6)],
        'per_leg_velocity_rms': [float(np.sqrt(np.mean(dv[:, i]**2)))
                                 for i in range(6)],
        'joint_normalized': float(np.mean((dq/inputs['joint_scale'])**2)),
        'foot_normalized': float(np.mean((df/inputs['foot_scale'])**2)),
        'velocity_normalized': float(np.mean((dv/inputs['velocity_scale'])**2)),
        'tendon_normalized': float(np.mean((dl/.2)**2))}
    result['combined'] = sum(result[key] for key in
        ('joint_normalized', 'foot_normalized', 'velocity_normalized',
         'tendon_normalized'))
    return result


def dense_shape_screen(final, actual):
    ratios = {'foot_ratio': final['foot_rms']/actual['foot_rms'],
        'velocity_ratio': final['velocity_rms']/actual['velocity_rms'],
        'q_ratio': final['q_rms']/actual['q_rms'],
        'per_leg_q_ratios': (np.asarray(final['per_leg_q_rms'])
            /np.asarray(actual['per_leg_q_rms'])).tolist()}
    ratios['supported'] = bool(ratios['foot_ratio'] <= .8
        and ratios['velocity_ratio'] <= .8 and ratios['q_ratio'] <= 1.05
        and all(value <= 1.05 for value in ratios['per_leg_q_ratios']))
    return ratios


def reviewed_inputs(root, out):
    """Freeze E110 observation, E111 geometry and earlier immutable anatomy."""
    design_path = ROOT/'docs/BANC_M3_E112_DENSE_FOOT_FIT_DESIGN_20260920.md'
    frozen = json.loads((root/'e112-frozen-design.json').read_text(encoding='utf-8'))
    if sha256(design_path) != EXPECTED_DESIGN_SHA or \
            frozen.get('design_sha256') != EXPECTED_DESIGN_SHA or \
            frozen.get('inputs_sha256') != EXPECTED_INPUT_SHA or \
            frozen.get('actual_fit_authorized') is not False:
        raise ValueError('Frozen E112 design and separate actual-fit approval required')
    # Parent changes actual_fit_authorized only through a separate message;
    # frozen metadata intentionally remains false for provenance.
    input_path = root/'parent-e112-training-inputs.npz'
    if sha256(input_path) != EXPECTED_INPUT_SHA:
        raise ValueError('Frozen E112 training input hash changed')
    recovery = json.loads((root/'e112b-frozen-recovery.json').read_text(encoding='utf-8'))
    failed = root/'e112-dense-foot-recruitment'
    failure_review = json.loads((failed/'parent-failed-preflight-review.json').read_text(
        encoding='utf-8'))
    if sha256(ROOT/'docs/BANC_M3_E112B_PREFLIGHT_RECOVERY_20260920.md') != \
            EXPECTED_RECOVERY_SHA or recovery['design_sha256'] != EXPECTED_RECOVERY_SHA or \
            sha256(failed/'parent-failed-preflight-review.json') != \
            recovery['failed_preflight_review_sha256'] or \
            sha256(failed/'process-wall.json') != recovery['failed_process_receipt_sha256'] or \
            recovery['input_sha256'] != EXPECTED_INPUT_SHA or \
            (recovery['prior_s'], recovery['process_s'], recovery['shared_s']) != \
            (PRIOR_S, PROCESS_LIMIT_S, SHARED_LIMIT_S) or \
            failure_review['recovery_prior_ceiling_s'] != PRIOR_S or \
            failure_review['fit_evaluations'] != 0 or \
            failure_review['conditional_response_calls'] != 0 or \
            failure_review['blocking_audit_defects'] != []:
        raise ValueError('Audited E112 failed preflight and reduced E112b budget required')
    report = json.loads((root/'parent-e112-training-inputs.json').read_text(
        encoding='utf-8'))
    hashes = {}
    for label, relative in report['source_paths'].items():
        path = root/relative
        digest = sha256(path)
        if digest != report['source_sha256'][label]:
            raise ValueError('Frozen E112 source changed: '+label)
        hashes[label] = digest
    if report.get('training_inputs_sha256') != EXPECTED_INPUT_SHA or \
            report.get('geometry_hash') != EXPECTED_GEOMETRY_HASH:
        raise ValueError('Independent E112 training/geometry identity changed')
    e110 = json.loads((root/report['source_paths']['E110_parent_review']).read_text(
        encoding='utf-8'))
    e111 = json.loads((root/report['source_paths']['E111_parent_review']).read_text(
        encoding='utf-8'))
    if (e110.get('accepted_as_valid_actual_diagnostic') is not True or
            e110.get('blocking_defects') != [] or
            e111.get('accepted_as_valid_static_geometry_diagnostic') is not True or
            e111.get('blocking_defects') != []):
        raise ValueError('E110 and E111 final parent acceptance required')
    train = load_npz(input_path)
    old = load_npz(root/report['source_paths']['E108_anatomy_startup_inputs'])
    cal = load_npz(root/report['source_paths']['E110_calibration'])
    parent_baseline = load_npz(root/report['source_paths']['parent_conditional_baseline'])
    geometry = TorchFootGeometry.load(root/report['source_paths']['E111_geometry'])
    trace = json.loads((root/report['source_paths']['E110_trace']).read_text(
        encoding='utf-8'))
    if geometry.geometry_hash != EXPECTED_GEOMETRY_HASH or \
            geometry.metadata['model_hash'] != trace['body_hash']:
        raise ValueError('E111 geometry identity changed')
    for leg in range(6):
        if not np.array_equal(geometry.arrays['leg_frontend_indices'][leg],
                              np.arange(7*leg, 7*leg+7)):
            raise ValueError('Explicit E111 leg frontend order differs from 7-block anatomy')
    for key in ('edge_ids', 'edge_rows', 'edge_cols',
                'original_signed_edge_weights', 'original_w', 'leg_motor_mask',
                'leg_edge_mask', 'startup_mask', 'startup_target_command',
                'startup_target_tendon', 'pool', 'moment', 'rest', 'ltm',
                'membership', 'source_ids', 'motor_ids', 'tau', 'a',
                'threshold', 'cap'):
        exact(train[key], old[key], 'Unchanged E108 anatomy/startup '+key)
    for key in ('edge_ids', 'individual_edge_efficacies', 'motor_ids',
                'moment', 'pool', 'rest', 'ltm', 'membership'):
        source_key = {'edge_ids': 'individual_edge_ids',
            'individual_edge_efficacies': 'individual_edge_efficacies'}.get(key, key)
        target_key = 'initial_efficacy' if key == 'individual_edge_efficacies' else key
        exact(train[target_key], cal[source_key], 'E110 calibration '+key)
    if len(cal['tau_ids']) != 38 or cal['theta'].shape != (96, 48) or \
            not np.array_equal(train['time_s'], np.arange(2001)*.005) or \
            train['source_rate'].shape != (2001, 49) or \
            train['background'].shape != (2001, 391) or \
            train['current'].shape != (2001, 391) or \
            train['current_w'].shape != (391, 49) or \
            train['target'].shape != (1828, 42) or \
            train['target_feet'].shape != (1828, 6, 3) or \
            train['target_velocity'].shape != (1827, 6, 3) or \
            train['actual_command'].shape != (2001, 42) or \
            train['actual_tendon'].shape != (2001, 6):
        raise ValueError('Complete original E110 5ms input and 38tau required')
    selected = np.arange(100, 1928, dtype=np.int64)
    if not np.array_equal(train['selected'], selected) or \
            not np.array_equal(np.flatnonzero(train['train_mask']), selected) or \
            not np.array_equal(np.flatnonzero(train['startup_mask']), np.arange(1, 100)):
        raise ValueError('Original contiguous1828 training and original99 startup required')
    for name, position in (('target_velocity', 'target_feet'),
                           ('actual_velocity', 'actual_feet'),
                           ('predicted_velocity', 'predicted_feet')):
        if not np.array_equal(train[name], np.diff(train[position], axis=0)/.005):
            raise ValueError('Frozen consecutive 5ms velocity changed: '+name)
    if (not np.array_equal(train['joint_scale'],
                           np.maximum(train['target'].std(axis=0), .025)) or
            not np.array_equal(train['foot_scale'],
                               np.maximum(train['target_feet'].std(axis=0), .05)) or
            not np.array_equal(train['velocity_scale'],
                               np.maximum(train['target_velocity'].std(axis=0), .5))):
        raise ValueError('Target-only scales changed')
    audit_background_roundoff(train['current'], train['source_rate'],
                              train['current_w'], train['background'], out)
    dense_fit = np.zeros((391, 49), dtype=bool)
    dense_fit[train['edge_rows'], train['edge_cols']] = True
    if int(dense_fit.sum()) != 1572:
        raise ValueError('Repeated fitted source-MN coordinate')
    exact(train['current_w'][~dense_fit], train['original_w'][~dense_fit],
          'E110 all unfitted dense source-MN weights')
    exact(train['current_w'][train['edge_rows'], train['edge_cols']].astype(np.float32),
          train['actual_signed_edge_weights'], 'E110 actual selected float32 W')
    if not all(np.isfinite(value).all() for key, value in train.items()
               if value.dtype.kind == 'f' and key != 'phase') or \
            not np.isfinite(train['phase'][selected]).all():
        raise ValueError('Nonfinite frozen E112 input')
    motor, edge = leg_partition(train['moment'], train['pool'], train['ltm'],
        train['edge_rows'], expected_motor=train['leg_motor_mask'],
        expected_edge=train['leg_edge_mask'])
    if tuple(motor.sum(axis=1)) != MOTOR_COUNTS or \
            tuple(edge.sum(axis=1)) != EDGE_COUNTS or \
            not np.array_equal(train['source_rate'][0], np.zeros(49)) or \
            not np.array_equal(train['current'][0], np.zeros(391)):
        raise ValueError('Original E110 zero and six independent partitions required')
    for key, parent_key in (('time_s', 'time_s'), ('source_rate', 'source_rate'),
                            ('background', 'background'), ('current_w', 'actual_weights'),
                            ('tau', 'tau'), ('a', 'a'), ('threshold', 'threshold'),
                            ('cap', 'cap'), ('actual_command', 'actual_command'),
                            ('actual_tendon', 'actual_tendon')):
        exact(train[key], parent_baseline[parent_key], 'Parent E110 baseline '+key)
    save_json(out/'input-provenance.json', {'source_sha256': hashes,
        'training_inputs_sha256': EXPECTED_INPUT_SHA,
        'design_sha256': EXPECTED_DESIGN_SHA,
        'recovery_design_sha256': EXPECTED_RECOVERY_SHA,
        'recovery_metadata': recovery,
        'geometry_hash': geometry.geometry_hash,
        'E110_review': True, 'E111_review': True,
        'selected_start_end': [int(selected[0]), int(selected[-1])],
        'selected_count': len(selected), 'velocity_intervals': len(selected)-1})
    return train, cal, parent_baseline, geometry


def selected_actual_float32(train, cal, efficacy):
    """Map unsorted calibration edge IDs explicitly and preserve every other field."""
    efficacy = np.asarray(efficacy, dtype=np.float64)
    if efficacy.shape != (1572,) or not np.isfinite(efficacy).all() or \
            np.any(efficacy < .001) or np.any(efficacy > 10.):
        raise ValueError('Finite selected1572 efficacy inside original bounds required')
    positions = {int(edge): i for i, edge in enumerate(cal['edge_ids'])}
    if len(positions) != len(cal['edge_ids']) or \
            any(int(edge) not in positions for edge in train['edge_ids']):
        raise ValueError('Fitted existing edge absent or duplicated in calibration')
    values = cal['values'].copy()
    for index, edge in enumerate(train['edge_ids']):
        values[positions[int(edge)]] = np.float32(
            train['original_signed_edge_weights'][index]*efficacy[index])
    changed_mask = np.zeros(len(values), dtype=bool)
    changed_mask[[positions[int(edge)] for edge in train['edge_ids']]] = True
    exact(values[~changed_mask], cal['values'][~changed_mask],
          'Final all unchanged calibration values')
    original_sign = np.signbit(train['original_signed_edge_weights'])
    if not np.array_equal(np.signbit(np.float32(
            train['original_signed_edge_weights']*efficacy)), original_sign):
        raise ValueError('Original selected edge sign changed by float32 rounding')
    dense = train['current_w'].copy()
    rows, cols = train['edge_rows'], train['edge_cols']
    if len(np.unique(np.stack((rows, cols), axis=1), axis=0)) != 1572 or \
            not np.array_equal(train['original_w'][rows, cols].astype(np.float32),
                               train['original_signed_edge_weights']):
        raise ValueError('Original signed selected dense row/column map changed')
    dense[rows, cols] = np.float32(
        train['original_signed_edge_weights']*efficacy).astype(np.float64)
    dense_mask = np.zeros((391, 49), dtype=bool)
    dense_mask[rows, cols] = True
    exact(dense[~dense_mask], train['current_w'][~dense_mask],
          'Final all unselected dense source-MN weights')
    updated = {key: value.copy() for key, value in cal.items()}
    updated['values'] = values
    updated['individual_edge_efficacies'] = efficacy.copy()
    for key in cal:
        if key not in ('values', 'individual_edge_efficacies'):
            exact(updated[key], cal[key], 'Unchanged E110 calibration '+key)
    exact(updated['tau_ids'], cal['tau_ids'], 'Original38 tau IDs')
    exact(updated['tau_values'], cal['tau_values'], 'Original38 tau values')
    return dense, updated


class StagedResponse:
    """Stage immutable E110 inputs once; all conditional operations stay on GPU."""

    def __init__(self, train, geometry, *, device='cuda'):
        self.train = train
        self.device = torch.device(device)
        self.geometry = geometry.to(self.device)
        self.device = self.geometry.device
        self.tensor = lambda value: torch.as_tensor(value, dtype=torch.float64,
            device=self.device)
        for key in ('source_rate', 'background', 'tau', 'a', 'threshold',
                    'cap', 'pool', 'moment', 'rest', 'original_w', 'joint_scale',
                    'foot_scale', 'velocity_scale', 'target', 'target_feet',
                    'target_velocity', 'startup_target_command',
                    'startup_target_tendon'):
            setattr(self, key, self.tensor(train[key]))
        self.selected = train['selected']
        self.startup = np.flatnonzero(train['startup_mask'])
        self.ltm = train['ltm']

    def full(self, actual_float32_w):
        weight = self.tensor(actual_float32_w)
        current = self.source_rate@weight.T+self.background
        z = self.cap*torch.tanh(self.a*(current-self.threshold)/self.cap)
        rate = linear_response(torch.relu(z), .005, self.tau)
        activation = endpoint_response(rate@self.pool.T, .005, .01)
        command = self.rest+.02*activation@self.moment.T
        tendon = .2*torch.tanh(activation[:, self.ltm]/10)
        feet = self.geometry.full(command[self.selected])
        velocity = (feet[1:]-feet[:-1])/.005
        return {'current': current, 'motor_rate': rate,
            'muscle_activation': activation, 'command': command,
            'tendon': tendon, 'feet': feet, 'velocity': velocity}

    def block(self, leg, log_theta):
        train = self.train
        motor = np.flatnonzero(train['leg_motor_mask'][leg])
        edges = np.flatnonzero(train['leg_edge_mask'][leg])
        local_rows = np.searchsorted(motor, train['edge_rows'][edges])
        if not np.array_equal(motor[local_rows], train['edge_rows'][edges]):
            raise ValueError('Selected edge outside its only MN leg block')
        local_cols = train['edge_cols'][edges]
        rows_t = torch.as_tensor(local_rows, dtype=torch.long, device=self.device)
        cols_t = torch.as_tensor(local_cols, dtype=torch.long, device=self.device)
        start, stop = 7*leg, 7*(leg+1)
        return {'leg': leg, 'motor': motor, 'edges': edges,
            'rows': rows_t, 'cols': cols_t,
            'signed': self.tensor(train['original_signed_edge_weights'][edges]),
            'original_w': self.original_w[motor],
            'background': self.background[:, motor],
            'tau': self.tau[motor], 'a': self.a[motor],
            'threshold': self.threshold[motor], 'cap': self.cap[motor],
            'pool': self.pool[:, motor],
            'moment': self.moment[start:stop], 'rest': self.rest[start:stop],
            'ltm': int(self.ltm[leg]),
            'target_q': self.target[:, start:stop],
            'target_foot': self.target_feet[:, leg],
            'target_velocity': self.target_velocity[:, leg],
            'q_scale': self.joint_scale[start:stop],
            'foot_scale': self.foot_scale[leg],
            'velocity_scale': self.velocity_scale[leg],
            'startup_q': self.startup_target_command[:, start:stop],
            'startup_tendon': self.startup_target_tendon[:, leg],
            'initial': np.asarray(log_theta, dtype=np.float64)}


def block_tensors(staged, block, parameter):
    """Four equally weighted dense losses and exact original99 inner ratios."""
    efficacy = parameter.exp()
    weights = block['original_w'].clone()
    weights[block['rows'], block['cols']] = block['signed']*efficacy
    current = staged.source_rate@weights.T+block['background']
    z = block['cap']*torch.tanh(block['a']*(current-block['threshold'])/block['cap'])
    rate = linear_response(torch.relu(z), .005, block['tau'])
    activation = endpoint_response(rate@block['pool'].T, .005, .01)
    command = block['rest']+.02*activation@block['moment'].T
    tendon = .2*torch.tanh(activation[:, block['ltm']]/10)
    foot = staged.geometry.leg(command[staged.selected], block['leg'])
    velocity = adjacent_velocity(foot, staged.selected, staged.train['time_s'])
    q_loss = ((command[staged.selected]-block['target_q'])/
              block['q_scale']).square().mean()
    foot_loss = ((foot-block['target_foot'])/block['foot_scale']).square().mean()
    velocity_loss = ((velocity-block['target_velocity'])/
                     block['velocity_scale']).square().mean()
    tendon_loss = ((tendon[staged.selected]-.2)/.2).square().mean()
    prior = (6e-5*len(parameter)/1572)*(efficacy-1).square().mean()
    objective = q_loss+foot_loss+velocity_loss+tendon_loss+prior
    ratios = startup_ratios(command[staged.startup],
        tendon[staged.startup], block['startup_q'], block['startup_tendon'])
    return {'objective': objective, 'ratios': ratios,
        'parts': torch.stack((q_loss, foot_loss, velocity_loss,
                              tendon_loss, prior)),
        'current': current, 'motor_rate': rate,
        'muscle_activation': activation, 'command': command,
        'tendon': tendon, 'feet': foot, 'velocity': velocity}


def detach_np(value):
    return value.detach().cpu().numpy()


def save_technical(out, name, index, x, output, *, gradient=None, ratios_jac=None):
    path = out/f'technical-{name}-{index}.npz'
    payload = {'log_theta': x}
    payload.update({key: detach_np(value) for key, value in output.items()})
    if gradient is not None:
        payload['objective_gradient'] = gradient
    if ratios_jac is not None:
        payload['ratio_jacobian'] = ratios_jac
    try:
        np.savez_compressed(path, **payload)
        return None
    except Exception as error:  # noqa: BLE001 - retain numerical failure
        return {'type': type(error).__name__, 'message': str(error)}


def evaluate_block(staged, block, x, index, *, out, ledger, leg_started,
                   fit_started, records):
    name = LEGS[block['leg']]
    parameter = torch.tensor(x, dtype=torch.float64, device=staged.device,
                             requires_grad=True)
    output = block_tensors(staged, block, parameter)
    if any(not bool(torch.isfinite(value).all()) for value in output.values()):
        persistence_error = save_technical(out, name, index, x, output)
        error = ValueError('Nonfinite E112 block output '+name)
        if persistence_error is not None:
            error.add_note('Technical raw save failed: '+str(persistence_error))
            try_save(out/f'technical-save-error-{name}-{index}.json',
                     {'original_error': str(error), 'persistence_error': persistence_error})
        raise error
    gradient = detach_np(torch.autograd.grad(output['objective'], parameter,
                                              retain_graph=True)[0])
    ratio_jac = detach_np(torch.stack([
        torch.autograd.grad(output['ratios'][i], parameter,
                            retain_graph=i < 3)[0] for i in range(4)]))
    if not np.isfinite(gradient).all() or not np.isfinite(ratio_jac).all():
        persistence_error = save_technical(out, name, index, x, output,
                                           gradient=gradient, ratios_jac=ratio_jac)
        error = ValueError('Nonfinite E112 block gradient '+name)
        if persistence_error is not None:
            error.add_note('Technical raw save failed: '+str(persistence_error))
            try_save(out/f'technical-save-error-{name}-{index}.json',
                     {'original_error': str(error), 'persistence_error': persistence_error})
        raise error
    objective = float(output['objective'].detach())
    ratios = detach_np(output['ratios'])
    record = {'index': index, 'leg': name, 'objective': objective,
        'parts': detach_np(output['parts']).tolist(),
        'ratios': ratios.tolist(),
        'optimizer_inequalities': (OPTIMIZER_CAP-ratios).tolist(),
        'selector_inequalities': (SELECTOR_CAP-ratios).tolist(),
        'strict_startup_feasible': selector_feasible(ratios),
        'gradient': gradient.tolist(),
        'ratio_jacobian': ratio_jac.tolist(),
        'log_theta': x.tolist(),
        'leg_wall_s': time.perf_counter()-leg_started,
        'fit_wall_s': time.perf_counter()-fit_started}
    try:
        append_jsonl(out/f'history-{name}.jsonl', record)
        records.append(record)
        ledger['leg_completed_evaluations'][name] = index
        save_json(out/'execution-ledger.json', ledger)
    except Exception as error:  # preserve first history/ledger failure
        persistence_error = save_technical(out, name, index, x, output,
                                           gradient=gradient, ratios_jac=ratio_jac)
        if persistence_error is not None:
            error.add_note('Technical raw save also failed: '+str(persistence_error))
        raise
    return {'objective': objective, 'jac': gradient, 'ratios': ratios,
            'ratio_jac': ratio_jac, 'record': record}


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def full_validation_response(staged, weights, out, ledger, *, phase, payload):
    """Charge an actual full response before forward and persist its raw output."""
    if phase not in ('baseline', 'final'):
        raise ValueError('Unknown E112 full validation phase')
    ledger[phase+'_validation_calls'] += 1
    save_json(out/'execution-ledger.json', ledger)
    with torch.no_grad():
        raw = staged.full(weights)
    # detach_np transfers CUDA output to host, so completion is charged only
    # after every response tensor has successfully materialized there.
    arrays = {key: detach_np(value) for key, value in raw.items()}
    ledger[phase+'_response_completed'] += 1
    path = out/('baseline-cuda-raw.npz' if phase == 'baseline' else 'prediction.npz')
    # Preserve materialized responses before a bookkeeping write can fail.
    try:
        np.savez_compressed(path, **arrays, **payload)
    except Exception as error:  # keep raw failure primary if the ledger also fails
        ledger_error = try_save(out/'execution-ledger.json', ledger)
        if ledger_error is not None:
            error.add_note('E112 validation ledger persistence also failed: '+
                           str(ledger_error))
        raise
    ledger[phase+'_raw_saved'] += 1
    save_json(out/'execution-ledger.json', ledger)
    return arrays


def baseline_validation(staged, train, parent, out, ledger):
    """Persist full CUDA raw baseline before comparing to frozen NumPy response."""
    arrays = full_validation_response(staged, train['current_w'], out, ledger,
        phase='baseline', payload={
            'actual_command': train['actual_command'],
            'actual_tendon': train['actual_tendon'], 'target': train['target'],
            'target_feet': train['target_feet'],
            'target_velocity': train['target_velocity']})
    parent_names = {'current': 'full_current', 'motor_rate': 'motor_rate',
        'muscle_activation': 'muscle_activation', 'command': 'command',
        'tendon': 'tendon'}
    limits = {'current': 1e-9, 'command': 1e-9, 'tendon': 1e-9,
              'motor_rate': 1e-8, 'muscle_activation': 1e-8}
    errors = {key: float(np.max(np.abs(arrays[key]-parent[parent_key])))
              for key, parent_key in parent_names.items()}
    errors.update({key+'_vs_frozen_prediction': float(np.max(np.abs(
        arrays[key]-train[parent_key]))) for key, parent_key in (
        ('command', 'predicted_command'), ('tendon', 'predicted_tendon'),
        ('motor_rate', 'predicted_motor_rate'),
        ('muscle_activation', 'predicted_activation'),
        ('feet', 'predicted_feet'), ('velocity', 'predicted_velocity'))})
    first99 = arrays['command'][1:100]-train['actual_command'][1:100]
    q_proxy = rms(arrays['command'][train['selected']]
                  -train['actual_command'][train['selected']])
    q_target = rms(train['actual_command'][train['selected']]-train['target'])
    v_proxy = rms(arrays['velocity']-train['actual_velocity'])
    v_target = rms(train['actual_velocity']-train['target_velocity'])
    fidelity = {'first99_q_rms': rms(first99),
        'first99_q_max': float(np.max(np.abs(first99))),
        'training_q_proxy_rms': q_proxy, 'training_q_target_rms': q_target,
        'velocity_proxy_rms': v_proxy, 'velocity_target_rms': v_target}
    checks = {key: math.isfinite(value) and value <= limits[key]
              for key, value in errors.items() if key in limits}
    frozen_limits = {'command': 1e-9, 'tendon': 1e-9,
        'motor_rate': 1e-8, 'muscle_activation': 1e-8,
        'feet': 1e-9, 'velocity': 1e-8}
    checks.update({key+'_vs_frozen_prediction': math.isfinite(
        errors[key+'_vs_frozen_prediction']) and
        errors[key+'_vs_frozen_prediction'] <= bound
        for key, bound in frozen_limits.items()})
    checks.update({'first99_q_rms': fidelity['first99_q_rms'] <= .002,
        'first99_q_max': fidelity['first99_q_max'] <= .01,
        'training_q_proxy_ratio': q_target > 0 and q_proxy/q_target <= .1,
        'velocity_proxy_ratio': v_target > 0 and v_proxy/v_target <= .1})
    save_json(out/'baseline-validation.json', {'errors': errors,
        'fidelity': fidelity, 'checks': checks,
        'raw_saved_before_assertions': True})
    if not all(checks.values()) or \
            not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('E112 parent baseline/actual E110 fidelity failed')
    return arrays


def startup_details(command, tendon, train, limits):
    q = command[train['startup_mask']]-train['startup_target_command']
    l = tendon[train['startup_mask']]-train['startup_target_tendon']
    measured = [rms(q), float(np.max(np.abs(q))),
                rms(l), float(np.max(np.abs(l)))]
    per_leg = []
    for i in range(6):
        values = np.array([rms(q[:, 7*i:7*i+7]),
            float(np.max(np.abs(q[:, 7*i:7*i+7]))),
            rms(l[:, i]), float(np.max(np.abs(l[:, i])))])
        ratios = (values/limits)**2
        per_leg.append({'q_RMS_rad': values[0], 'q_max_abs_rad': values[1],
            'LTM_RMS': values[2], 'LTM_max_abs': values[3],
            'ratios': ratios.tolist(),
            'supported': bool(np.isfinite(values).all() and np.all(values <= limits)),
            'selector_margin_supported': selector_feasible(ratios)})
    return {'q_RMS_rad': measured[0], 'q_max_abs_rad': measured[1],
        'LTM_RMS': measured[2], 'LTM_max_abs': measured[3],
        'limits': list(map(float, limits)),
        'ratios': (np.asarray(measured)/limits).tolist(),
        'supported': bool(np.isfinite(measured).all()
                          and np.all(np.asarray(measured) <= limits)),
        'per_leg': per_leg,
        'selector_margin_supported': all(item['selector_margin_supported']
                                         for item in per_leg)}


def resource_status(ledger):
    whole = ledger.get('process_wall_s')
    fit = ledger.get('fit_wall_s')
    leg_wall = ledger['leg_wall_s']
    started = ledger['leg_started_evaluations']
    if whole is None or not math.isfinite(whole) or whole < 0 or \
            (ledger['fit_started'] and (fit is None or not math.isfinite(fit))) or \
            any(count and leg_wall[name] is None for name, count in started.items()):
        return 'limit_exceeded_or_unverifiable'
    if (whole > PROCESS_LIMIT_S or (fit is not None and fit > FIT_LIMIT_S)
            or any(value is not None and value > LEG_LIMIT_S
                   for value in leg_wall.values())
            or any(value > LEG_MAX_CALLS for value in started.values())
            or sum(started.values()) > 6000 or PRIOR_S+whole > SHARED_LIMIT_S):
        return 'limit_exceeded'
    return 'within_internal_limits_external_receipt_required'


def run(root, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError('One E112 output already exists: '+str(out))
    out.mkdir(parents=True)
    ledger = {'experiment': 'M3-E112b-dense-foot-recruitment',
        'prior_conservative_wall_s': PRIOR_S, 'process_wall_s': None,
        'fit_started': False, 'fit_wall_s': None,
        'active_leg': None, 'active_log_theta': None,
        'leg_started_evaluations': {leg: 0 for leg in LEGS},
        'leg_completed_evaluations': {leg: 0 for leg in LEGS},
        'leg_cache_hits': {leg: 0 for leg in LEGS},
        'leg_wall_s': {leg: None for leg in LEGS},
        'baseline_validation_calls': 0, 'baseline_response_completed': 0,
        'baseline_raw_saved': 0, 'final_validation_calls': 0,
        'final_response_completed': 0, 'final_raw_saved': 0,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'fit_calls': 0}
    result = None
    primary = None
    fit_started = None
    active_leg_started = None
    try:
        save_json(out/'spec.json', {'experiment': ledger['experiment'],
            'conditional_only': True, 'body_validated': False,
            'source_E110': 'actual E110 endpoints, not E105',
            'geometry_hash': EXPECTED_GEOMETRY_HASH,
            'objective': 'equal mean normalized q, foot position, adjacent velocity, LTM plus fixed small efficacy prior',
            'inner_startup': INNER.tolist(), 'original_startup': ORIGINAL.tolist(),
            'optimizer_cap': OPTIMIZER_CAP, 'selector_cap': SELECTOR_CAP,
            'limits': {'leg_s': LEG_LIMIT_S, 'fit_s': FIT_LIMIT_S,
                'process_s': PROCESS_LIMIT_S, 'prior_s': PRIOR_S,
                'shared_s': SHARED_LIMIT_S, 'leg_vectors': LEG_MAX_CALLS,
                'total_vectors': 6000, 'maxiter': 1000},
            'tool_sha256': sha256(__file__)})
        (out/'runner.py').write_bytes(Path(__file__).read_bytes())
        train, cal, parent, geometry = reviewed_inputs(root, out)
        dense_initial, unchanged_cal = selected_actual_float32(
            train, cal, train['initial_efficacy'])
        exact(dense_initial, train['current_w'], 'E110 initial actual float32 W')
        exact(unchanged_cal['values'], cal['values'], 'E110 initial calibration values')
        staged = StagedResponse(train, geometry, device='cuda')
        baseline = baseline_validation(staged, train, parent, out, ledger)
        actual_metrics = scalar_metrics(train['actual_command'],
            train['actual_feet'], train['actual_velocity'], train['actual_tendon'], train)
        baseline_metrics = scalar_metrics(baseline['command'], baseline['feet'],
            baseline['velocity'], baseline['tendon'], train)
        baseline_inner = startup_details(baseline['command'], baseline['tendon'],
                                         train, INNER)
        baseline_original = startup_screen(baseline['command'], baseline['tendon'],
            train['startup_target_command'], train['startup_target_tendon'],
            train['time_s'], train['startup_mask'])
        save_json(out/'baseline-summary.json', {'actual_metrics': actual_metrics,
            'conditional_metrics': baseline_metrics, 'inner_startup': baseline_inner,
            'original_startup': baseline_original,
            'E110_start_is_infeasible': not baseline_inner['supported']})
        if baseline_inner['supported'] or baseline_original['supported']:
            raise ValueError('Known E110 infeasible original99 initializer unexpectedly passed')
        fit_started = time.perf_counter()
        ledger['fit_started'] = True
        fit_deadline = min(fit_started+FIT_LIMIT_S, started+PROCESS_LIMIT_S)
        save_json(out/'execution-ledger.json', ledger)
        chosen = {}
        summaries = {}
        for leg, name in enumerate(LEGS):
            leg_started = time.perf_counter()
            if leg_started >= fit_deadline-.5:
                summaries[name] = {'stop': 'whole_fit_wall_limit_before_block',
                    'best_any': None, 'best_feasible': None,
                    'unique_evaluations': 0, 'wall_s': 0.}
                save_json(out/f'block-{name}-summary.json', summaries[name])
                continue
            edges = np.flatnonzero(train['leg_edge_mask'][leg])
            x0 = np.log(train['initial_efficacy'][edges])
            if len(edges) != EDGE_COUNTS[leg] or not np.isfinite(x0).all():
                raise ValueError('Original E110 leg efficacy block changed')
            block = staged.block(leg, x0)
            records = []
            ledger['active_leg'] = name
            active_leg_started = leg_started
            save_json(out/'execution-ledger.json', ledger)

            def on_start(index, x, *, name=name, leg_started=leg_started):
                ledger['leg_started_evaluations'][name] = index
                ledger['active_log_theta'] = x.tolist()
                event = {'leg': name, 'index': index, 'log_theta': x.tolist(),
                    'fit_wall_s': time.perf_counter()-fit_started,
                    'leg_wall_s': time.perf_counter()-leg_started}
                append_jsonl(out/f'started-{name}.jsonl', event)
                save_json(out/'execution-ledger.json', ledger)

            def evaluate(x, index, *, block=block, leg_started=leg_started,
                         records=records):
                return evaluate_block(staged, block, x, index, out=out,
                    ledger=ledger, leg_started=leg_started,
                    fit_started=fit_started, records=records)

            cache = EvaluationCache(evaluate, len(edges), on_start,
                max_calls=LEG_MAX_CALLS,
                deadline=leg_started+LEG_LIMIT_S, total_deadline=fit_deadline)
            ledger['fit_calls'] += 1
            save_json(out/'execution-ledger.json', ledger)
            try:
                optimizer, termination = optimize_block(cache, x0)
                final_key = np.asarray(optimizer.x, dtype=np.float64).tobytes()
                if final_key in cache.cache:
                    terminal = cache.cache[final_key]
                    termination['ratios'] = terminal['ratios'].tolist()
                    termination['ratio_jacobian'] = terminal['ratio_jac'].tolist()
                else:
                    termination['ratios'] = None
                    termination['ratio_jacobian'] = None
            except FitLimit as error:
                termination = optimizer_record(None, str(error))
            save_json(out/f'optimizer-{name}.json', termination)
            if termination['stop'] == 'scipy_nonfinite_result':
                raise ValueError('Nonfinite original SLSQP result preserved: '+name)
            best_any, best_feasible = choose_records(records)
            for label, record in (('best-any', best_any),
                                  ('best-feasible', best_feasible)):
                if record is not None:
                    np.savez_compressed(out/f'{name}-{label}-coefficients.npz',
                        log_theta=np.asarray(record['log_theta']),
                        efficacy=np.exp(record['log_theta']),
                        edge_ids=train['edge_ids'][edges],
                        objective=record['objective'], ratios=record['ratios'])
            wall = time.perf_counter()-leg_started
            stop = ('per_leg_wall_limit_exceeded' if wall > LEG_LIMIT_S else
                    termination['stop'])
            summaries[name] = {'stop': stop,
                'unique_evaluations': cache.started, 'completed_evaluations': cache.completed,
                'cache_hits': cache.hits, 'wall_s': wall,
                'optimizer': termination,
                'best_any': None if best_any is None else {key: best_any[key]
                    for key in ('index', 'objective', 'ratios')},
                'best_feasible': None if best_feasible is None else {key: best_feasible[key]
                    for key in ('index', 'objective', 'ratios')}}
            ledger['leg_cache_hits'][name] = cache.hits
            ledger['leg_wall_s'][name] = wall
            save_json(out/f'block-{name}-summary.json', summaries[name])
            save_json(out/'all-block-summaries.json', summaries)
            save_json(out/'execution-ledger.json', ledger)
            # Include coefficients, summaries and ledger persistence in the leg.
            wall = time.perf_counter()-leg_started
            summaries[name]['wall_s'] = wall
            ledger['leg_wall_s'][name] = wall
            if wall > LEG_LIMIT_S:
                summaries[name]['stop'] = 'per_leg_wall_limit_exceeded'
            save_json(out/f'block-{name}-summary.json', summaries[name])
            save_json(out/'all-block-summaries.json', summaries)
            ledger['active_leg'] = None
            ledger['active_log_theta'] = None
            save_json(out/'execution-ledger.json', ledger)
            active_leg_started = None
            if best_feasible is not None and wall <= LEG_LIMIT_S:
                chosen[name] = np.exp(best_feasible['log_theta'])
        ledger['fit_wall_s'] = time.perf_counter()-fit_started
        save_json(out/'execution-ledger.json', ledger)
        if len(chosen) != 6:
            result = {'experiment': ledger['experiment'],
                'validity': 'valid_completed_budget_limited_fit',
                'hypothesis_verdict': 'not_supported_no_six_feasible_blocks',
                'local_supported': False, 'candidate': False,
                'block_summaries': summaries, 'actual_baseline_metrics': actual_metrics,
                'conditional_baseline_metrics': baseline_metrics,
                'body_validated': False, 'adoption': 'not_applicable',
                'F1_F8': 'not_assessed', 'fit_calls': ledger['fit_calls'],
                'actual_neural_s': 0., 'actual_physical_s': 0.}
        else:
            efficacy = np.empty(1572, dtype=np.float64)
            for leg, name in enumerate(LEGS):
                efficacy[train['leg_edge_mask'][leg]] = chosen[name]
            if np.any(efficacy < .001*(1-8*np.finfo(float).eps)) or \
                    np.any(efficacy > 10*(1+8*np.finfo(float).eps)):
                raise ValueError('Chosen efficacy outside original bounds')
            efficacy = np.clip(efficacy, .001, 10.)
            dense, updated = selected_actual_float32(train, cal, efficacy)
            np.savez_compressed(out/'selected-coefficients.npz', efficacy=efficacy,
                edge_ids=train['edge_ids'], edge_rows=train['edge_rows'],
                edge_cols=train['edge_cols'], original_signed_weights=
                train['original_signed_edge_weights'])
            final = full_validation_response(staged, dense, out, ledger,
                phase='final', payload={
                    'actual_float32_source_weights': dense,
                    'efficacy': efficacy, 'target': train['target'],
                    'target_feet': train['target_feet'],
                    'target_velocity': train['target_velocity']})
            if not all(np.isfinite(value).all() for value in final.values()):
                raise ValueError('Nonfinite final actual-float32 full response')
            np.savez_compressed(out/'calibration.npz', **updated)
            final_metrics = scalar_metrics(final['command'], final['feet'],
                final['velocity'], final['tendon'], train)
            dense_screen = dense_shape_screen(final_metrics, actual_metrics)
            relative = native_shape_screen(final_metrics, actual_metrics)
            absolute = absolute_shape_screen(final_metrics)
            original_startup = startup_screen(final['command'], final['tendon'],
                train['startup_target_command'], train['startup_target_tendon'],
                train['time_s'], train['startup_mask'])
            inner = startup_details(final['command'], final['tendon'], train, INNER)
            dense_screen['supported'] = bool(dense_screen['supported']
                and inner['supported'] and inner['selector_margin_supported'])
            local_supported = bool(dense_screen['supported'] and relative['supported']
                and absolute['supported'] and original_startup['supported']
                and inner['supported'] and inner['selector_margin_supported'])
            result = {'experiment': ledger['experiment'],
                'validity': 'valid_conditional_fit',
                'hypothesis_verdict': 'dense_foot_and_startup_supported'
                    if dense_screen['supported'] else 'not_supported',
                'local_supported': local_supported, 'candidate': True,
                'conditional_only': True, 'runtime_reference_phase': False,
                'body_validated': False, 'adoption': 'not_applicable',
                'F1_F8': 'not_assessed', 'fit_calls': ledger['fit_calls'],
                'actual_neural_s': 0., 'actual_physical_s': 0.,
                'actual_baseline_metrics': actual_metrics,
                'conditional_baseline_metrics': baseline_metrics,
                'final_metrics': final_metrics,
                'dense_foot_screen': dense_screen,
                'relative_shape_screen': relative,
                'absolute_E101_shape_screen': absolute,
                'original99_startup_screen': original_startup,
                'inner99_startup_screen': inner,
                'block_summaries': summaries,
                'all_38_tau_exact': True, 'all_nonfit_calibration_values_exact': True}
            save_json(out/'selection-and-screens.json', result)
    except Exception as error:  # noqa: BLE001 - preserve first experimental failure
        primary = error
        failure = {'error_type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(),
            'active_leg': ledger['active_leg'],
            'active_log_theta': ledger['active_log_theta'],
            'started': ledger['leg_started_evaluations'],
            'completed': ledger['leg_completed_evaluations'],
            'baseline_calls': ledger['baseline_validation_calls'],
            'baseline_response_completed': ledger['baseline_response_completed'],
            'baseline_raw_saved': ledger['baseline_raw_saved'],
            'final_calls': ledger['final_validation_calls'],
            'final_response_completed': ledger['final_response_completed'],
            'final_raw_saved': ledger['final_raw_saved'],
            'actual_neural_s': 0., 'actual_physical_s': 0.}
        persistence = try_save(out/'execution-error.json', failure)
        if persistence is not None:
            error.add_note('E112 execution-error persistence failed: '+str(persistence))
    finally:
        if ledger['active_leg'] is not None and active_leg_started is not None:
            ledger['leg_wall_s'][ledger['active_leg']] = (
                time.perf_counter()-active_leg_started)
        ledger['process_wall_s'] = time.perf_counter()-started
        if fit_started is not None and ledger['fit_wall_s'] is None:
            ledger['fit_wall_s'] = time.perf_counter()-fit_started
        ledger['conservative_shared_wall_s'] = PRIOR_S+ledger['process_wall_s']
        ledger['resource_status'] = resource_status(ledger)
        if result is not None:
            result['validation_accounting'] = {
                key: ledger[key] for key in (
                    'baseline_validation_calls', 'baseline_response_completed',
                    'baseline_raw_saved', 'final_validation_calls',
                    'final_response_completed', 'final_raw_saved')}
            result['process_wall_s_internal'] = ledger['process_wall_s']
            result['fit_wall_s'] = ledger['fit_wall_s']
            result['resource_status'] = ledger['resource_status']
            result['prior_conservative_wall_s'] = PRIOR_S
            result['conservative_shared_wall_s_internal'] = (
                ledger['conservative_shared_wall_s'])
            if ledger['resource_status'] != 'within_internal_limits_external_receipt_required':
                result['validity'] = 'resource_limit_incomplete'
                result['local_supported'] = False
            save_error = try_save(out/'result.json', result)
            if save_error is not None and primary is None:
                primary = RuntimeError('E112 result persistence failed: '+str(save_error))
        ledger_error = try_save(out/'execution-ledger.json', ledger)
        if ledger_error is not None and primary is None:
            primary = RuntimeError('E112 final ledger persistence failed: '+str(ledger_error))
    if primary is not None:
        raise primary
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(jsonable(run(args.root, args.out)), allow_nan=False,
                     ensure_ascii=False), flush=True)
