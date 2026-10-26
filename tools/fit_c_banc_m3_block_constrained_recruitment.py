# ruff: noqa: B023
"""E108: one bounded six-block native conditional recruitment fit.

This tool never advances a neural network or a physical body. Its only body is
a shadow model used for static forward kinematics and its Jacobian.
"""

import argparse
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.fit_c_banc_m3_constrained_startup import (
    checked_initial_state,
    current_edge_start,
    frozen_tripod_offsets,
    independent_edge_order,
    native_shape_screen,
    observed_roster,
    startup_grid,
    startup_screen,
)
from tools.fit_c_banc_m3_native_all_leg import own_leg_targets, reference_at_phase
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks
from tools.verify_c_banc_observation import exact

BASE = ROOT / 'verification/banc-natural-gait-m3-20260919'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
MN_COUNTS = (69, 63, 63, 70, 63, 63)
EDGE_COUNTS = (273, 281, 240, 238, 255, 285)
STARTUP_LIMITS = np.array([.02, .05, .01, .03], dtype=np.float64)
SAFETY = 1 - 1e-4


def json_safe(value):
    """Persist abnormal diagnostics as JSON null, never NaN or invented zero."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False),
                          encoding='utf-8')


def accepted_review(review, key):
    if not isinstance(review, dict) or review.get(key) is not True or review.get('blocking_defects') != []:
        raise ValueError('Explicit defect-free accepted review required: ' + key)


def verify_metadata_weights(current_w, metadata_w):
    """Compare exact values after lossless native-float32 to float64 widening."""
    actual = np.asarray(current_w)
    metadata = np.asarray(metadata_w)
    if (actual.shape != (391, 49) or actual.dtype != np.float64
            or metadata.shape != (391, 49) or metadata.dtype != np.float32
            or not np.isfinite(actual).all() or not np.isfinite(metadata).all()
            or not np.array_equal(actual, actual.astype(np.float32).astype(np.float64))):
        raise ValueError('Actual float32-provenance source weights and metadata required')
    widened = metadata.astype(np.float64)
    if not np.array_equal(metadata, widened.astype(np.float32)):
        raise ValueError('Metadata float32 widening must be lossless')
    exact(actual, widened, 'E105 metadata actual selected W')
    return True


def leg_partition(moment, pool, ltm, edge_rows, *, expected_motor=None, expected_edge=None):
    """Each motor, hence each signed fitted edge, must reach one leg only."""
    moment = np.asarray(moment)
    pool = np.asarray(pool)
    ltm = np.asarray(ltm)
    edge_rows = np.asarray(edge_rows)
    if (moment.shape != (42, 96) or pool.shape != (96, 391) or ltm.shape != (6,)
            or edge_rows.shape != (1572,) or not np.isfinite(moment).all()
            or not np.isfinite(pool).all() or np.any(edge_rows < 0)
            or np.any(edge_rows >= 391) or np.any(ltm < 0) or np.any(ltm >= 96)):
        raise ValueError('Complete finite fixed moment/pool/edge partition required')
    product = moment @ pool
    motor = np.array([np.any(product[7*i:7*i+7] != 0, axis=0)
                      | (pool[ltm[i]] != 0) for i in range(6)])
    edge = motor[:, edge_rows]
    if (not np.all(motor.sum(axis=0) == 1) or not np.all(edge.sum(axis=0) == 1)
            or tuple(motor.sum(axis=1)) != MN_COUNTS
            or tuple(edge.sum(axis=1)) != EDGE_COUNTS):
        raise ValueError('Unexpected motor or fitted-edge sharing across legs')
    if expected_motor is not None and not np.array_equal(motor, expected_motor):
        raise ValueError('Independent parent MN partition mismatch')
    if expected_edge is not None and not np.array_equal(edge, expected_edge):
        raise ValueError('Independent parent fitted-edge partition mismatch')
    return motor, edge


def check_static_cross_jacobian(jacobian, tolerance=1e-10):
    jacobian = np.asarray(jacobian)
    if jacobian.ndim != 4 or jacobian.shape[1:] != (6, 3, 42) or not np.isfinite(jacobian).all():
        raise ValueError('Finite static 6-leg x 42-joint foot Jacobian required')
    maximum = 0.
    for leg in range(6):
        other = np.r_[0:7*leg, 7*(leg+1):42]
        maximum = max(maximum, float(np.max(np.abs(jacobian[:, leg, :, other]))))
    if maximum > tolerance:
        raise ValueError('Static foot Jacobian has cross-leg influence')
    return maximum


def startup_ratios(command, tendon, target_q, target_tendon):
    q = np.asarray(command) - np.asarray(target_q)
    t = np.asarray(tendon) - np.asarray(target_tendon)
    if q.ndim != 2 or q.shape[1] != 7 or t.shape != (len(q),) or len(q) != 99:
        raise ValueError('Complete single-leg original99 startup outputs required')
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('Finite native startup outputs required')
    return np.array([np.mean(q*q)/.02**2, np.max(q*q)/.05**2,
                     np.mean(t*t)/.01**2, np.max(t*t)/.03**2])


def strict_feasible(ratios):
    ratios = np.asarray(ratios)
    return bool(ratios.shape == (4,) and np.isfinite(ratios).all()
                and np.all(ratios <= SAFETY**2))


def choose_records(records):
    """Only native objective and original-startup feasible records can win."""
    finite = [r for r in records if math.isfinite(r['objective'])]
    best_any = min(finite, key=lambda r: r['objective']) if finite else None
    feasible = [r for r in finite if strict_feasible(r['ratios'])]
    best_feasible = min(feasible, key=lambda r: r['objective']) if feasible else None
    return best_any, best_feasible


def optimizer_record(result, stop):
    if result is None:
        return {'success': None, 'status': None, 'message': None, 'nit': None,
                'nfev': None, 'njev': None, 'objective': None,
                'gradient_max_abs': None, 'stop': stop}
    jac = np.asarray(result.get('jac', []), dtype=np.float64)
    objective = float(result.get('fun', np.nan))
    finite = math.isfinite(objective) and np.isfinite(jac).all()
    return {'success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'nit': int(result.nit),
            'nfev': int(result.nfev), 'njev': int(result.njev),
            'objective': objective if math.isfinite(objective) else None,
            'gradient_max_abs': float(np.max(np.abs(jac))) if jac.size and np.isfinite(jac).all() else None,
            'nonfinite_objective_text': None if math.isfinite(objective) else str(objective),
            'gradient_nonfinite_counts': {'nan': int(np.isnan(jac).sum()),
                'positive_infinity': int(np.isposinf(jac).sum()),
                'negative_infinity': int(np.isneginf(jac).sum())},
            'stop': stop if finite else 'scipy_nonfinite_result'}


def resource_screen(ledger):
    """Charge a prior preflight and reject unknown cost for started fit work."""
    prior = ledger.get('prior_shared_wall_s')
    whole = ledger.get('whole_wall_s')
    fit = ledger.get('fit_wall_s')
    started = ledger.get('leg_started_evaluations', {})
    leg_wall = ledger.get('leg_wall_s', {})
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
               for v in (prior, whole)):
        return 'limit_exceeded_or_unverifiable'
    if (ledger.get('fit_start_elapsed_s') is not None or any(started.values())) and (
            fit is None or not math.isfinite(fit) or fit < 0):
        return 'limit_exceeded_or_unverifiable'
    if fit is not None and (not math.isfinite(fit) or fit < 0):
        return 'limit_exceeded_or_unverifiable'
    if any(count > 0 and (leg_wall.get(leg) is None
            or not math.isfinite(leg_wall[leg]) or leg_wall[leg] < 0)
           for leg, count in started.items()):
        return 'limit_exceeded_or_unverifiable'
    if (prior+whole > 225 or (fit is not None and fit > 150)
            or any(v is not None and v > 24 for v in leg_wall.values())
            or any(v > 200 for v in started.values())
            or sum(started.values()) > 1200):
        return 'limit_exceeded'
    return 'within_limits'


def solve_block(cache, x0, size, path):
    """Persist the actual SciPy return before examining nonfinite fields."""
    optimization = minimize(
        lambda x: cache.get(x)['objective'], x0, method='SLSQP',
        jac=lambda x: cache.get(x)['jac'],
        constraints=[{'type': 'ineq',
            'fun': lambda x: 1-cache.get(x)['ratios'],
            'jac': lambda x: -cache.get(x)['ratio_jac']}],
        bounds=[(math.log(.001), math.log(10.))]*size,
        options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
    record = optimizer_record(optimization,
        'scipy_success' if optimization.success else 'scipy_failure')
    save_json(path, record)
    if record['stop'] == 'scipy_nonfinite_result':
        raise ValueError('Nonfinite SciPy optimizer return; preserved record at '+str(path))
    return optimization, record


def close_shadow_body(body, before, out, *, pending_error=None, exact_check=exact):
    """Report static validation and always attempt close, including on errors."""
    validation_error = None
    close_error = None
    exact_state = None
    closed = None
    try:
        try:
            if before is None:
                raise ValueError('Initial shadow snapshot unavailable')
            exact_check(before, body.snapshot(), 'Static shadow body and clock remain byte-exact')
            exact_state = True
        except Exception as exc:  # noqa: BLE001 - preserve exact-check failure while closing
            validation_error = exc
            exact_state = None if before is None else False
    finally:
        try:
            body.close()
            closed = True
        except Exception as exc:  # noqa: BLE001 - persist close failure with prior error
            close_error = exc
            closed = False
    save_json(out/'shadow-body-final.json', {
        'snapshot_exact': exact_state, 'closed': closed,
        'validation_error': None if validation_error is None else {
            'type': type(validation_error).__name__, 'message': str(validation_error)},
        'close_error': None if close_error is None else {
            'type': type(close_error).__name__, 'message': str(close_error)}})
    for error in (validation_error, close_error):
        if error is not None and pending_error is not None:
            pending_error.add_note('Shadow cleanup '+type(error).__name__+': '+str(error))
    if pending_error is None:
        if validation_error is not None:
            if close_error is not None:
                validation_error.add_note('Shadow close also failed: '+str(close_error))
            raise validation_error
        if close_error is not None:
            raise close_error


@contextmanager
def shadow_body_lifetime(original, out, *, body_factory=new_body):
    """Close a constructed shadow even if capturing its initial state fails."""
    body = body_factory(original)
    before = None
    try:
        before = body.snapshot()
        yield body, before
    finally:
        close_shadow_body(body, before, out, pending_error=sys.exc_info()[1])


class BlockLimit(Exception):
    pass


class NativeCache:
    """One count per new bit-exact float64 vector, shared by all SciPy callbacks."""
    def __init__(self, evaluate, size, *, max_calls, deadline, total_deadline, on_start):
        self.evaluate = evaluate
        self.size = size
        self.max_calls = max_calls
        self.deadline = deadline
        self.total_deadline = total_deadline
        self.on_start = on_start
        self.cache = {}
        self.calls = 0
        self.hits = 0

    def get(self, x):
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.size,) or not np.isfinite(v).all():
            raise ValueError('Finite full block parameter vector required')
        key = v.tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.calls >= self.max_calls or time.perf_counter() >= self.deadline:
            raise BlockLimit('per_leg_call_or_wall_limit')
        if time.perf_counter() >= self.total_deadline:
            raise BlockLimit('whole_fit_wall_limit')
        self.calls += 1
        self.on_start(self.calls, v)
        result = self.evaluate(v, self.calls)
        self.cache[key] = result
        return result


def metrics(command, feet, tendon, mask, target, target_feet, joint_scale, foot_scale):
    joint = (command[mask]-target)/joint_scale
    foot = (feet-target_feet)/foot_scale
    tendon_error = (tendon[mask]-.2)/.2
    return {'combined': float(np.mean(joint**2)+np.mean(foot**2)+np.mean(tendon_error**2)),
            'q_rms': float(np.sqrt(np.mean((command[mask]-target)**2))),
            'foot_rms': float(np.sqrt(np.mean((feet-target_feet)**2))),
            'LF_row6_rms': float(np.sqrt(np.mean((command[mask, 6]-target[:, 6])**2))),
            'per_leg_q_rms': [float(np.sqrt(np.mean((command[mask, 7*i:7*i+7]
                -target[:, 7*i:7*i+7])**2))) for i in range(6)],
            'joint_normalized': float(np.mean(joint**2)),
            'foot_normalized': float(np.mean(foot**2)),
            'tendon_normalized': float(np.mean(tendon_error**2))}


def absolute_shape_screen(final):
    old = json.loads((BASE/'e101-individual-edge-native-recruitment/baseline-output.json').read_text(encoding='utf-8'))
    caps = {'q_rms': .07076026970527503, 'foot_rms': .08708141476210751,
            'LF_row6_rms': .07850842813388147}
    comparison = {key: {'actual': final[key], 'cap': cap, 'passed': final[key] <= cap}
                  for key, cap in caps.items()}
    leg_caps = (1.05*np.asarray(old['per_leg_q_rms'])).tolist()
    comparison['per_leg_q'] = [{'actual': actual, 'cap': cap, 'passed': actual <= cap}
        for actual, cap in zip(final['per_leg_q_rms'], leg_caps)]
    comparison['supported'] = bool(all(v['passed'] for k, v in comparison.items() if k != 'per_leg_q')
        and all(v['passed'] for v in comparison['per_leg_q']))
    return comparison


def main(root, graph_path, out, *, prior_shared_wall_s):
    if not isinstance(prior_shared_wall_s, (int, float)) or not math.isfinite(prior_shared_wall_s) or prior_shared_wall_s < 0:
        raise ValueError('Explicit finite nonnegative prior shared wall cost required')
    if float(prior_shared_wall_s) != 8.:
        raise ValueError('E108 v1 preserved preflight requires exactly 8s shared charge')
    if out.exists():
        raise FileExistsError('Preserve existing E108 output: ' + str(out))
    started = time.perf_counter()
    out.mkdir(parents=True)
    ledger = {'experiment': 'M3-E108-block-constrained-recruitment',
              'prior_shared_wall_s': float(prior_shared_wall_s),
              'shared_limit_s': 225.,
              'preflight_wall_s': None, 'fit_wall_s': None, 'whole_wall_s': None,
              'fit_start_elapsed_s': None,
              'active_leg': None, 'active_leg_start_elapsed_s': None,
              'leg_started_evaluations': {leg: 0 for leg in LEGS},
              'leg_completed_evaluations': {leg: 0 for leg in LEGS},
              'leg_cache_hits': {leg: 0 for leg in LEGS},
              'leg_wall_s': {leg: None for leg in LEGS},
              'baseline_validation_calls': 0, 'final_validation_calls': 0,
              'actual_neural_s': 0., 'actual_physical_s': 0.,
              'resource_screen': None}
    result = None
    error = None
    try:
        result = _run(root, graph_path, out, started, ledger)
    except Exception as exc:  # noqa: BLE001 - persist all technical failures
        error = exc
    finally:
        if ledger['fit_wall_s'] is None and ledger['fit_start_elapsed_s'] is not None:
            ledger['fit_wall_s'] = time.perf_counter()-started-ledger['fit_start_elapsed_s']
        active = ledger['active_leg']
        if (active is not None and ledger['leg_wall_s'][active] is None
                and ledger['active_leg_start_elapsed_s'] is not None):
            ledger['leg_wall_s'][active] = (time.perf_counter()-started
                -ledger['active_leg_start_elapsed_s'])
        # _run owns its shadow body and closes it in its own finally.
        ledger['whole_wall_s'] = time.perf_counter()-started
        ledger['shared_wall_s'] = prior_shared_wall_s+ledger['whole_wall_s']
        ledger['resource_screen'] = resource_screen(ledger)
        if result is not None:
            result['resource_screen'] = ledger['resource_screen']
            result['prior_shared_wall_s'] = prior_shared_wall_s
            result['shared_wall_s'] = ledger['shared_wall_s']
            result['whole_wall_s'] = ledger['whole_wall_s']
            if ledger['resource_screen'] != 'within_limits':
                result['validity'] = 'resource_limit_incomplete'
                result['local_supported'] = False
            save_json(out/'result.json', result)
        if error is not None:
            save_json(out/'execution-error.json', {'validity': 'incomplete',
                'error_type': type(error).__name__, 'message': str(error),
                'traceback': ''.join(traceback.format_exception(error)),
                'whole_wall_s_at_error': ledger['whole_wall_s'],
                'fit_wall_s': ledger['fit_wall_s'],
                'prior_shared_wall_s': prior_shared_wall_s,
                'shared_wall_s': ledger['shared_wall_s'],
                'resource_screen': ledger['resource_screen'],
                'leg_started_evaluations': ledger['leg_started_evaluations'],
                'leg_completed_evaluations': ledger['leg_completed_evaluations'],
                'leg_wall_s': ledger['leg_wall_s'],
                'actual_neural_s': 0., 'actual_physical_s': 0.})
        # Include result/error persistence and shadow close in the final wall.
        ledger['whole_wall_s'] = time.perf_counter()-started
        ledger['shared_wall_s'] = prior_shared_wall_s+ledger['whole_wall_s']
        ledger['resource_screen'] = resource_screen(ledger)
        if result is not None:
            result['resource_screen'] = ledger['resource_screen']
            result['shared_wall_s'] = ledger['shared_wall_s']
            result['whole_wall_s'] = ledger['whole_wall_s']
            if ledger['resource_screen'] != 'within_limits':
                result['validity'] = 'resource_limit_incomplete'
                result['local_supported'] = False
            save_json(out/'result.json', result)
        save_json(out/'execution-ledger.json', ledger)
    if error is not None:
        raise error
    return result


def _run(root, graph_path, out, started, ledger):
    e99 = root/'e99-lf-tarsal-source-preparation'
    e103 = root/'e103-startup-preserving-recruitment'
    e104 = root/'e104-constrained-startup-recruitment'
    e105 = root/'e105-startup-native-body-diagnostic-v2'
    e107 = root/'e107-within-leg-shape-factor-diagnostic'
    v1 = root/'e108-block-constrained-recruitment'
    source = e105/'body'
    paths = {'E105_review': source/'parent-second-review.json',
        'E107_review': e107/'parent-first-review.json',
        'E105_signals': source/'recruitment-signals.npz',
        'E105_metadata': source/'recruitment-metadata.npz',
        'E105_trace': source/'trace.json',
        'E105_calibration': e105/'calibration.npz',
        'E105_start_manifest': source/'start/manifest.json',
        'E104_prediction': e104/'prediction.npz',
        'E104_coefficients': e104/'fit-coefficients.npz',
        'E103_startup': e103/'startup-inputs.npz',
        'E107_frozen': e107/'frozen-both-inputs.npz',
        'E99_calibration': e99/'calibration.npz',
        'E85_reference': root/'e85-space-compatible-cadence-reference/commands.npz',
        'E101_baseline': root/'e101-individual-edge-native-recruitment/baseline-output.json',
        'E32_start_manifest': root/'e32-core-adaptation/candidate/start/manifest.json',
        'parent_expected': root/'parent-e108-expected-inputs.npz',
        'E108_v1_first_review': v1/'parent-first-review.json',
        'causal_helper': ROOT/'tools/banc_m3_causal_recruitment.py',
        'geometry_helper': ROOT/'tools/banc_m3_kinematic_fit.py',
        'runner': Path(__file__),
        'unit_test': ROOT/'tests/test_c_banc_block_constrained_recruitment.py'}
    paths.update({f'graph_{p.name}': p for p in graph_path.iterdir() if p.is_file()})
    spec = {'experiment': 'M3-E108-block-constrained-recruitment',
        'scope': 'One latest-E105-observation conditional native six-block SLSQP fit; no full neural or body advancement',
        'source_sha256': {key: sha256(path) for key, path in paths.items()},
        'limits': {'shared_wall_s': 225, 'prior_shared_wall_s': ledger['prior_shared_wall_s'],
            'current_remaining_wall_s': 225-ledger['prior_shared_wall_s'], 'fit_wall_s': 150,
            'per_leg_wall_s': 24, 'per_leg_unique_native_evaluations': 200,
            'total_unique_native_evaluations': 1200, 'slsqp_maxiter': 200},
        'order': list(LEGS), 'bounds_efficacy': [.001, 10.],
        'objective': 'Per-leg mean normalized seven q squared + mean normalized 3D foot squared + mean own LTM(.2) squared + 6e-5*n_edges/1572*mean((efficacy-1)^2)',
        'startup_constraints': 'Four original99 per-leg inequalities 1-ratio >=0, with 1e-4 relative safety margin for record selection',
        'conditional_only': True, 'runtime_reference_phase': False}
    save_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'unit-test.py').write_bytes(paths['unit_test'].read_bytes())
    accepted_review(json.loads(paths['E105_review'].read_text(encoding='utf-8')),
                    'accepted_as_valid_failed_body_diagnostic')
    accepted_review(json.loads(paths['E107_review'].read_text(encoding='utf-8')),
                    'accepted_as_valid_supported_external_body_diagnostic')
    v1_review = json.loads(paths['E108_v1_first_review'].read_text(encoding='utf-8'))
    for relative, original_hash in v1_review['original_artifacts_sha256'].items():
        if sha256(v1/relative) != original_hash:
            raise ValueError('Preserved E108 v1 original artifact changed: '+relative)
    cal = {k: v.copy() for k, v in np.load(paths['E105_calibration']).items()}
    original_cal = {k: v.copy() for k, v in np.load(paths['E99_calibration']).items()}
    if any(k not in cal for k in original_cal):
        raise ValueError('Original calibration parameter missing')
    for key, value in original_cal.items():
        if key != 'values':
            exact(value, cal[key], 'E105 unchanged original calibration '+key)
    if len(cal['tau_ids']) != 38 or cal['theta'].shape != (96, 48):
        raise ValueError('Original 38tau and fixed legacy parameter dimensions required')
    graph = GraphStore.load(graph_path)
    original = StateStore.load(root/'e32-core-adaptation/candidate/start')
    prepared = StateStore.load(source/'start')
    initial_activation = checked_initial_state(prepared)
    exact(original['body'], prepared['body'], 'Original whole-zero body/world')
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        exact(original['network'][key], prepared['network'][key], 'Original zero neural '+key)
    signal = np.load(paths['E105_signals'])
    metadata = np.load(paths['E105_metadata'])
    expected = np.load(paths['parent_expected'])
    times = np.arange(2001)*.005
    exact(signal['time_s'], times[1:], 'E105 exact 2000 endpoints')
    sources, motor, source_cols, motor_cols = observed_roster(cal, signal['observed_ids'],
        metadata['observed_ids'], metadata['source_ids'])
    exact(motor, metadata['motor_ids'], 'Metadata original motor order')
    raw_rate = signal['rate'].astype(float)
    raw_current = (signal['recurrent_current']+signal['drive']-signal['adaptation']).astype(float)
    source_rate = np.vstack((np.zeros((1, 49)), raw_rate[:, source_cols]))
    full_current = np.vstack((np.zeros((1, 391)), raw_current[:, motor_cols]))
    if (not np.isfinite(source_rate).all() or not np.isfinite(full_current).all()
            or np.any(signal['adaptation'][:, motor_cols] != 0)
            or np.any(metadata['_cell_gain'][motor_cols] != 0)):
        raise ValueError('Complete finite latest source/current and zero MN adaptation/gain required')
    trace = json.loads(paths['E105_trace'].read_text(encoding='utf-8'))
    rows = trace['samples']
    if len(rows) != 2001 or any(row['time_s'] != times[i] for i, row in enumerate(rows)):
        raise ValueError('Complete E105 actual 2001 trace required')
    if (trace['body_hash'] != prepared['body']['modelHash']
            or trace['model_hash'] != prepared['network']['model_hash']
            or trace['adapter_hash'] != prepared['adapter']['hash']
            or trace['graph_hash'] != prepared['graph_hash']
            or any(trace['parameters'][k] != v for k, v in prepared['parameters'].items())):
        raise ValueError('Original E105 body/network/adapter/parameter provenance mismatch')
    actual_activation = np.vstack((initial_activation[None],
                                   signal['muscle_activation'].astype(float)))
    decoded = cal['rest']+.02*actual_activation@cal['moment'].T
    recorded = np.asarray([row['joint_targets_rad'] for row in rows], dtype=float)
    np.testing.assert_allclose(decoded[1:], recorded[1:], rtol=0, atol=1e-6)
    actual_tendon = .2*np.tanh(actual_activation[:, cal['ltm']]/10)
    names = tuple(f'{leg}_tarsus' for leg in LEGS)
    recorded_tendon = np.asarray([[row['tendon_inputs'][name] for name in names]
                                  for row in rows[1:]])
    np.testing.assert_allclose(actual_tendon[1:], recorded_tendon, rtol=0, atol=1e-6)
    startup = np.load(paths['E103_startup'])
    startup_mask = startup['mask'].copy()
    startup_q = startup['target_command'].copy()
    startup_tendon = startup['target_tendon'].copy()
    startup_grid(times, startup_mask, startup_q, startup_tendon)
    np.savez_compressed(out/'startup-inputs.npz', time_s=times, mask=startup_mask,
        target_command=startup_q, target_tendon=startup_tendon)
    reference = np.load(paths['E85_reference'])
    offsets = frozen_tripod_offsets(reference['phase'])
    group_by_leg = {'rh': 0, 'rf': 1, 'lf': 2, 'rm': 3, 'lh': 4, 'lm': 5}
    phases = np.empty((2001, 6))
    peak_last = []
    source_e1_ids = []
    for leg, name in enumerate(LEGS):
        ids = cal['core_ids'][(cal['core_groups'] == group_by_leg[name])
                              & (cal['core_cell_types'] == 0)]
        if len(ids) != 1:
            raise ValueError('Exactly one unchanged E1 per leg required')
        source_e1_ids.append(int(ids[0]))
        col = int(np.flatnonzero(sources == ids[0])[0])
        peaks = quadratic_peaks(times, source_rate[:, col])
        if len(peaks) < 20 or peaks[0] > .5 or peaks[-1] < 9.5:
            raise ValueError('Complete actual latest E1 peaks required')
        phases[:, leg] = phase_trace(times, peaks)
        peak_last.append(float(peaks[-1]))
    mask = (times >= .5) & (times < min(peak_last)) & np.isfinite(phases).all(axis=1)
    if int(mask.sum()) != 1827 or times[np.flatnonzero(mask)[-1]] != 9.63:
        raise ValueError('Exact latest E105 all-six 1827 training samples required')
    target = own_leg_targets(reference, phases, offsets, mask, 0)
    train = (reference['time_s'] >= 5) & (reference['time_s'] < 8)
    holdout = (reference['time_s'] >= 8) & (reference['time_s'] < 10)
    held = reference_at_phase(reference, reference['phase'][holdout, 0]/(2*np.pi), 0, train)
    reference_rms = float(np.sqrt(np.mean((held-reference['command'][holdout])**2)))
    if reference_rms > .002:
        raise ValueError('Original E85 5..8 to 8..10 reconstruction failed')
    frozen = np.load(paths['E107_frozen'])
    exact(frozen['source_time_s'], times, 'E107 original source time')
    exact(frozen['E1_ids'], np.asarray(source_e1_ids), 'E107 frozen actual E1 IDs')
    common = (times[mask] >= 3) & (times[mask] <= 9.5)
    frozen_time = frozen['control_time_s'][frozen['control_time_s'] > 2.5]
    indices = np.searchsorted(frozen_time, times[mask][common])
    if (np.any(indices >= len(frozen_time)) or not np.array_equal(frozen_time[indices], times[mask][common])
            or np.max(np.abs(target[common]-frozen['shape_only_reference_target_rad'][indices])) > 1e-12):
        raise ValueError('E108 target differs from frozen E107 shape_only target at 3..9.5s')
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, motor)
    current = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    original_w = base[motor][:, sources].toarray().astype(float)
    current_w = current[motor][:, sources].toarray().astype(float)
    edge_rows, edge_cols, edge_ids = independent_edge_order(graph, motor, sources, original_w)
    edge_original = base.data[edge_ids].astype(np.float32)
    edge_current = current.data[edge_ids].astype(np.float32)
    exact(edge_original, original_w[edge_rows, edge_cols].astype(np.float32), 'Original signed edges')
    exact(edge_current, current_w[edge_rows, edge_cols].astype(np.float32), 'Actual E105 signed edges')
    e104 = np.load(paths['E104_prediction'])
    exact(current_w, e104['actual_float32_source_weights'], 'E105 actual selected W equals E104 final')
    metadata_w = metadata['source_to_motor'][:,
        [int(np.flatnonzero(metadata['source_ids'] == s)[0]) for s in sources]]
    verify_metadata_weights(current_w, metadata_w)
    coeff = np.load(paths['E104_coefficients'])
    exact(edge_ids, coeff['edge_ids'], 'E104 selected edge order')
    exact(sources, coeff['source_basis_ids'], 'E104 source basis')
    raw_start, initial_theta, projection = current_edge_start(edge_original, edge_current)
    reconstruct = original_w.copy()
    reconstruct[edge_rows, edge_cols] = edge_current.astype(float)
    exact(reconstruct, current_w, 'E105 actual float32 signed dense W')
    motor_mask, edge_mask = leg_partition(cal['moment'], cal['pool'], cal['ltm'], edge_rows,
        expected_motor=expected['leg_motor_mask'], expected_edge=expected['leg_edge_mask'])
    background = full_current-source_rate@current_w.T
    tau = metadata['tau'][motor_cols].astype(float)
    a = metadata['a'][motor_cols].astype(float)
    threshold = metadata['threshold'][motor_cols].astype(float)
    cap = metadata['cap'][motor_cols].astype(float)
    if (not np.allclose(a, 1) or not np.allclose(threshold, 7.5)
            or not np.allclose(cap, 200) or not np.all(np.isin(np.round(tau, 7),
                np.round([.02, .014046887330582595], 7)))):
        raise ValueError('Actual E105 fixed MN physiology required')
    selected = np.flatnonzero(mask)[::20]
    joint_scale = np.maximum(target.std(axis=0), .025)
    input_arrays = {'time_s': times, 'source_ids': sources, 'motor_ids': motor,
        'source_rate': source_rate, 'current': full_current, 'background': background,
        'current_w': current_w, 'phase': phases, 'offsets': offsets,
        'train_mask': mask, 'target': target, 'selected': selected,
        'joint_scale': joint_scale, 'actual_activation': actual_activation,
        'actual_command': decoded, 'recorded_command': recorded,
        'leg_motor_mask': motor_mask, 'leg_edge_mask': edge_mask,
        'startup_mask': startup_mask, 'startup_target_command': startup_q,
        'startup_target_tendon': startup_tendon,
        'edge_ids': edge_ids, 'edge_rows': edge_rows, 'edge_cols': edge_cols,
        'original_signed_edge_weights': edge_original,
        'tau': tau, 'a': a, 'threshold': threshold, 'cap': cap}
    for key, value in input_arrays.items():
        exact(value, expected[key], 'Independent parent expected E108 '+key)
    np.savez_compressed(out/'training-inputs.npz', **input_arrays,
        original_w=original_w, source_e1_ids=np.asarray(source_e1_ids),
        last_peak_s=np.asarray(peak_last), pool=cal['pool'], moment=cal['moment'],
        rest=cal['rest'], ltm=cal['ltm'], membership=cal['membership'],
        current_signed_edge_weights=edge_current, initial_efficacy=initial_theta,
        raw_initial_efficacy=raw_start, original99_recorded_tendon=recorded_tendon,
        reference_reconstruction_rms=reference_rms)
    save_json(out/'input-validation.json', {'E105_E1_ids': source_e1_ids,
        'last_peak_s': peak_last, 'mask_rows': int(mask.sum()),
        'last_mask_s': float(times[np.flatnonzero(mask)[-1]]),
        'reference_reconstruction_rms_rad': reference_rms,
        'E107_common_target_max_abs_rad': float(np.max(np.abs(target[common]
            -frozen['shape_only_reference_target_rad'][indices]))),
        'startup_target_unchanged': True, 'parent_independent_inputs_exact': True,
        'leg_motor_counts': motor_mask.sum(axis=1).tolist(),
        'leg_edge_counts': edge_mask.sum(axis=1).tolist(),
        'initial_roundoff_projection_max_abs': projection})
    if ledger['prior_shared_wall_s']+time.perf_counter()-started >= 225:
        raise TimeoutError('E108 preflight consumed shared wall allowance')
    with shadow_body_lifetime(original, out) as (body, body_before):
        qpos = np.repeat(body.d.qpos[None, :], len(selected), axis=0)
        geometry = StaticFootGeometry(body, qpos)
        target_feet, jac = geometry.evaluate(target[np.searchsorted(np.flatnonzero(mask), selected)], jacobian=True)
        cross = check_static_cross_jacobian(jac)
        foot_scale = np.maximum(target_feet.std(axis=0), .05)
        np.savez_compressed(out/'static-geometry.npz', qpos=qpos,
            target_feet=target_feet, target_jacobian=jac, selected=selected,
            foot_scale=foot_scale)
        save_json(out/'static-geometry-check.json', {'cross_leg_max_abs': cross,
            'frozen_neutral_contexts': len(qpos), 'body_clock_before': body_before.get('clock')})
        return _fit(out, started, ledger, body, body_before, geometry, cal,
            source_rate, background, current_w, original_w, edge_rows, edge_cols,
            edge_ids, edge_original, initial_theta, motor_mask, edge_mask,
            tau, a, threshold, cap, times, mask, target, selected, target_feet,
            joint_scale, foot_scale, startup_mask, startup_q, startup_tendon,
            decoded, actual_tendon, full_current, actual_activation,
            base, current, reference_rms)


def _fit(out, started, ledger, body, body_before, geometry, cal, source_rate,
         background, current_w, original_w, edge_rows, edge_cols, edge_ids,
         edge_original, initial_theta, motor_mask, edge_mask, tau, a, threshold,
         cap, times, mask, target, selected, target_feet, joint_scale, foot_scale,
         startup_mask, startup_q, startup_tendon, decoded, actual_tendon,
         full_current, actual_activation, base, current, reference_rms):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('E108 CUDA native autograd required')
    torch.set_num_threads(1)
    device = 'cuda'
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    source_t = tensor(source_rate)
    background_t = tensor(background)
    pool_t = tensor(cal['pool'])
    moment_t = tensor(cal['moment'])
    rest_t = tensor(cal['rest'])
    tau_t, a_t, threshold_t, cap_t = map(tensor, (tau, a, threshold, cap))
    ltm_t = torch.as_tensor(cal['ltm'], device=device)
    mask_t = torch.as_tensor(mask, device=device)
    startup_t = torch.as_tensor(startup_mask, device=device)
    selected_t = torch.as_tensor(selected, device=device)

    def full_predict(weights):
        current_t = source_t@weights.T+background_t
        z = cap_t*torch.tanh(a_t*(current_t-threshold_t)/cap_t)
        rate = linear_response(torch.relu(z), .005, tau_t)
        activation = endpoint_response(rate@pool_t.T, .005, .01)
        command = rest_t+.02*activation@moment_t.T
        tendon = .2*torch.tanh(activation[:, ltm_t]/10)
        return (item.detach().cpu().numpy() for item in
                (command, tendon, rate, activation, current_t))

    with torch.no_grad():
        bc, bt, br, ba, bi = full_predict(tensor(current_w))
    ledger['baseline_validation_calls'] = 1
    # Preserve the actual and reconstructed raw arrays before any fidelity gate.
    np.savez_compressed(out/'baseline-prediction.npz', command=bc, tendon=bt,
        motor_rate=br, muscle_activation=ba, current=bi,
        measured_current=full_current, actual_command=decoded,
        actual_tendon=actual_tendon, actual_activation=actual_activation,
        target=target, startup_target_command=startup_q,
        startup_target_tendon=startup_tendon)
    if not all(np.isfinite(item).all() for item in (bc, bt, br, ba, bi)):
        raise ValueError('Nonfinite latest-data baseline native response')
    transfer = float(np.sqrt(np.mean((bc[mask]-decoded[mask])**2)))
    target_error = float(np.sqrt(np.mean((decoded[mask]-target)**2)))
    transfer_ratio = transfer/target_error if target_error > 0 else float('nan')
    initial_q_error = bc[startup_mask]-decoded[startup_mask]
    startup_fidelity = {'q_RMS_rad': float(np.sqrt(np.mean(initial_q_error**2))),
                        'q_max_abs_rad': float(np.max(np.abs(initial_q_error)))}
    original_startup = startup_screen(decoded, actual_tendon, startup_q,
        startup_tendon, times, startup_mask)
    fidelity = {'latest_steady_transfer_rms_rad': transfer,
        'latest_actual_to_target_rms_rad': target_error,
        'latest_transfer_ratio': transfer_ratio,
        'initial99_latest_actual_reconstruction': startup_fidelity,
        'E105_actual_vs_original_E100_E103_startup_targets': original_startup,
        'required_ratio_at_most': .1,
        'required_initial99_q_RMS_at_most_rad': .002,
        'required_initial99_q_max_at_most_rad': .01}
    save_json(out/'baseline-fidelity.json', fidelity)
    if (not math.isfinite(transfer_ratio) or transfer_ratio > .1
            or startup_fidelity['q_RMS_rad'] > .002
            or startup_fidelity['q_max_abs_rad'] > .01):
        return {'experiment': 'M3-E108-block-constrained-recruitment',
            'validity': 'method_not_tested_fidelity', 'local_supported': False,
            'candidate': False, 'fidelity': fidelity, 'fit_wall_s': None,
            'actual_neural_s': 0., 'actual_physical_s': 0.}

    baseline_feet = geometry.evaluate(decoded[selected])
    baseline_metrics = metrics(decoded, baseline_feet, actual_tendon, mask,
        target, target_feet, joint_scale, foot_scale)
    np.savez_compressed(out/'baseline-static-feet.npz', feet=baseline_feet,
                        target_feet=target_feet, selected=selected)
    save_json(out/'baseline-output.json', baseline_metrics)
    ledger['preflight_wall_s'] = time.perf_counter()-started
    save_json(out/'execution-ledger.json', ledger)
    if ledger['preflight_wall_s'] >= 75 or not np.isfinite(initial_theta).all():
        raise TimeoutError('E108 preparation exhausted reserved experiment budget')

    fit_started = time.perf_counter()
    ledger['fit_start_elapsed_s'] = fit_started-started
    fit_deadline = min(fit_started+150,
                       started+225-ledger['prior_shared_wall_s']-5)
    histories = {}
    selected_by_leg = {}
    block_summaries = {}

    class LegFoot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, commands, leg):
            values = commands.detach().cpu().numpy()
            points = np.empty((len(values), 3))
            derivative = np.empty((len(values), 3, 7))
            jac_world = np.zeros((3, body.m.nv))
            for row, q in enumerate(values):
                geometry.data.qpos[:] = geometry.qpos[row]
                geometry.data.qpos[geometry.qpos_ids[7*leg:7*(leg+1)]] = q
                geometry.mj.mj_kinematics(body.m, geometry.data)
                geometry.mj.mj_comPos(body.m, geometry.data)
                rotation = geometry.data.xmat[geometry.thorax].reshape(3, 3)
                origin = geometry.data.xpos[geometry.thorax]
                points[row] = (geometry.data.xpos[geometry.tips[leg]]-origin)@rotation
                geometry.mj.mj_jacBody(body.m, geometry.data, jac_world, None,
                                        int(geometry.tips[leg]))
                derivative[row] = (rotation.T@jac_world[:,
                    geometry.qvel_ids[7*leg:7*(leg+1)]])
            ctx.save_for_backward(torch.as_tensor(derivative, dtype=commands.dtype,
                                                   device=commands.device))
            return torch.as_tensor(points, dtype=commands.dtype, device=commands.device)

        @staticmethod
        def backward(ctx, gradient):
            derivative, = ctx.saved_tensors
            return torch.einsum('tx,txj->tj', gradient, derivative), None

    for leg, name in enumerate(LEGS):
        if time.perf_counter() >= fit_deadline:
            block_summaries[name] = {'stop': 'whole_fit_wall_limit_before_block',
                                     'unique_native_evaluations': 0,
                                     'best_any': None, 'best_feasible': None}
            continue
        block_started = time.perf_counter()
        ledger['active_leg'] = name
        ledger['active_leg_start_elapsed_s'] = block_started-started
        save_json(out/'execution-ledger.json', ledger)
        block_deadline = min(block_started+24, fit_deadline)
        rows = np.flatnonzero(motor_mask[leg])
        edges = np.flatnonzero(edge_mask[leg])
        local_rows = np.searchsorted(rows, edge_rows[edges])
        if not np.array_equal(rows[local_rows], edge_rows[edges]):
            raise ValueError('Fitted edge does not map into its one leg block')
        local_cols = edge_cols[edges]
        n = len(edges)
        x0 = np.log(initial_theta[edges])
        signed = tensor(edge_original[edges].astype(float))
        original_local = tensor(original_w[rows])
        source_local = source_t
        background_local = background_t[:, rows]
        tau_local, a_local, threshold_local, cap_local = (
            item[rows] for item in (tau_t, a_t, threshold_t, cap_t))
        pool_local = pool_t[:, rows]
        moment_local = moment_t[7*leg:7*(leg+1)]
        rest_local = rest_t[7*leg:7*(leg+1)]
        target_local = tensor(target[:, 7*leg:7*(leg+1)])
        target_foot = tensor(target_feet[:, leg])
        q_scale = tensor(joint_scale[7*leg:7*(leg+1)])
        f_scale = tensor(foot_scale[leg])
        startup_q_local = tensor(startup_q[:, 7*leg:7*(leg+1)])
        startup_tendon_local = tensor(startup_tendon[:, leg])
        edge_row_t = torch.as_tensor(local_rows, device=device)
        edge_col_t = torch.as_tensor(local_cols, device=device)

        def evaluate(x, index):
            parameter = torch.tensor(x, dtype=torch.float64, device=device,
                                     requires_grad=True)
            efficacy = parameter.exp()
            weights = original_local.clone()
            weights[edge_row_t, edge_col_t] = signed*efficacy
            local_current = source_local@weights.T+background_local
            z = cap_local*torch.tanh(a_local*(local_current-threshold_local)/cap_local)
            rate = linear_response(torch.relu(z), .005, tau_local)
            activation = endpoint_response(rate@pool_local.T, .005, .01)
            command = rest_local+.02*activation@moment_local.T
            tendon = .2*torch.tanh(activation[:, ltm_t[leg]]/10)
            feet = LegFoot.apply(command[selected_t], leg)
            joint_loss = ((command[mask_t]-target_local)/q_scale).square().mean()
            foot_loss = ((feet-target_foot)/f_scale).square().mean()
            tendon_loss = ((tendon[mask_t]-.2)/.2).square().mean()
            prior = (6e-5*n/1572)*(efficacy-1).square().mean()
            objective = joint_loss+foot_loss+tendon_loss+prior
            dq = command[startup_t]-startup_q_local
            dl = tendon[startup_t]-startup_tendon_local
            ratios = torch.stack((dq.square().mean()/.02**2,
                dq.square().max()/.05**2,
                dl.square().mean()/.01**2,
                dl.square().max()/.03**2))
            parts = torch.stack((joint_loss, foot_loss, tendon_loss, prior))
            output_arrays = (command, tendon, rate, activation, feet, local_current)
            scalar_arrays = (objective, ratios, parts)
            if any(not torch.isfinite(item).all().item() for item in (*output_arrays, *scalar_arrays)):
                np.savez_compressed(out/f'technical-nonfinite-{name}-{index}.npz',
                    log_theta=x, command=command.detach().cpu().numpy(),
                    tendon=tendon.detach().cpu().numpy(), rate=rate.detach().cpu().numpy(),
                    activation=activation.detach().cpu().numpy(), feet=feet.detach().cpu().numpy(),
                    current=local_current.detach().cpu().numpy(),
                    objective=float(objective.detach()), ratios=ratios.detach().cpu().numpy())
                raise ValueError('Nonfinite native block output or objective: '+name)
            grad_obj = torch.autograd.grad(objective, parameter, retain_graph=True)[0]
            grad_ratios = torch.stack([torch.autograd.grad(ratios[i], parameter,
                retain_graph=i < 3)[0] for i in range(4)])
            g_obj = grad_obj.detach().cpu().numpy()
            g_ratios = grad_ratios.detach().cpu().numpy()
            if not np.isfinite(g_obj).all() or not np.isfinite(g_ratios).all():
                np.savez_compressed(out/f'technical-nonfinite-gradient-{name}-{index}.npz',
                    log_theta=x, objective=float(objective.detach()),
                    ratios=ratios.detach().cpu().numpy(), objective_gradient=g_obj,
                    ratio_jacobian=g_ratios)
                raise ValueError('Nonfinite native autograd gradient: '+name)
            record = {'index': index, 'leg': name, 'objective': float(objective.detach()),
                'parts': parts.detach().cpu().numpy().tolist(),
                'ratios': ratios.detach().cpu().numpy().tolist(),
                'inequalities': (1-ratios.detach().cpu().numpy()).tolist(),
                'strict_startup_feasible': strict_feasible(ratios.detach().cpu().numpy()),
                'gradient_max_abs': float(np.max(np.abs(g_obj))),
                'gradient_l2': float(np.linalg.norm(g_obj)),
                'constraint_jacobian_max_abs': float(np.max(np.abs(g_ratios))),
                'constraint_jacobian_row_l2': np.linalg.norm(g_ratios, axis=1).tolist(),
                'log_theta': x.tolist(), 'block_wall_s': time.perf_counter()-block_started,
                'fit_wall_s': time.perf_counter()-fit_started}
            with (out/f'history-{name}.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(json_safe(record), allow_nan=False)+'\n')
            ledger['leg_completed_evaluations'][name] = index
            save_json(out/'execution-ledger.json', ledger)
            histories[name].append(record)
            return {'objective': record['objective'], 'jac': g_obj,
                    'ratios': np.asarray(record['ratios']), 'ratio_jac': g_ratios,
                    'record': record}

        histories[name] = []
        def on_start(index, _x):
            ledger['leg_started_evaluations'][name] = index
            save_json(out/'execution-ledger.json', ledger)

        cache = NativeCache(evaluate, n, max_calls=200,
            deadline=block_deadline, total_deadline=fit_deadline,
            on_start=on_start)
        stop = 'scipy_finished'
        try:
            _, termination = solve_block(cache, x0, n,
                out/f'optimizer-{name}.json')
            stop = termination['stop']
        except BlockLimit as exc:
            stop = str(exc)
            termination = optimizer_record(None, stop)
            save_json(out/f'optimizer-{name}.json', termination)
        # This check covers the solver's dense-QP time after its last callback.
        block_wall = time.perf_counter()-block_started
        if block_wall > 24:
            stop = 'per_leg_wall_limit_exceeded'
        ledger['leg_cache_hits'][name] = cache.hits
        best_any, best_feasible = choose_records(histories[name])
        for label, record in (('best-any', best_any), ('best-feasible', best_feasible)):
            if record is not None:
                np.savez_compressed(out/f'{name}-{label}-coefficients.npz',
                    log_theta=np.asarray(record['log_theta']),
                    efficacy=np.exp(record['log_theta']), edge_ids=edge_ids[edges],
                    motor_ids=cal['motor_ids'][edge_rows[edges]],
                    source_ids=np.unique(np.r_[cal['input_ids'],
                        cal['intrinsic_tonic_ids'], cal['coupling_ids']])[edge_cols[edges]],
                    objective=record['objective'], ratios=record['ratios'])
        block_summaries[name] = {'stop': stop, 'unique_native_evaluations': cache.calls,
            'cache_hits': cache.hits, 'wall_s': time.perf_counter()-block_started,
            'optimizer': termination,
            'best_any': {k: best_any[k] for k in ('index', 'objective', 'ratios', 'gradient_max_abs')} if best_any else None,
            'best_feasible': {k: best_feasible[k] for k in ('index', 'objective', 'ratios', 'gradient_max_abs')} if best_feasible else None}
        ledger['leg_wall_s'][name] = block_summaries[name]['wall_s']
        ledger['active_leg'] = None
        ledger['active_leg_start_elapsed_s'] = None
        save_json(out/f'block-{name}-summary.json', block_summaries[name])
        save_json(out/'all-block-summaries.json', block_summaries)
        if best_feasible is not None:
            selected_by_leg[name] = np.exp(best_feasible['log_theta'])
    ledger['fit_wall_s'] = time.perf_counter()-fit_started
    save_json(out/'execution-ledger.json', ledger)
    if ledger['fit_wall_s'] > 150:
        raise TimeoutError('E108 fit exceeded whole fit wall bound')
    if any(name not in selected_by_leg for name in LEGS):
        return {'experiment': 'M3-E108-block-constrained-recruitment',
            'validity': 'valid_completed_budget_limited_fit',
            'hypothesis_verdict': 'not_supported_no_six_feasible_blocks',
            'local_supported': False, 'candidate': False,
            'block_summaries': block_summaries, 'baseline_output': baseline_metrics,
            'fit_wall_s': ledger['fit_wall_s'], 'actual_neural_s': 0.,
            'actual_physical_s': 0.}

    efficacy = np.empty(1572, dtype=float)
    for leg, name in enumerate(LEGS):
        efficacy[edge_mask[leg]] = selected_by_leg[name]
    if (not np.isfinite(efficacy).all() or np.any(efficacy < .001*(1-8*np.finfo(float).eps))
            or np.any(efficacy > 10*(1+8*np.finfo(float).eps))):
        raise ValueError('Selected positive efficacy outside original bounds')
    efficacy = np.clip(efficacy, .001, 10.)
    np.savez_compressed(out/'selected-coefficients.npz', efficacy=efficacy,
        edge_ids=edge_ids, edge_rows=edge_rows, edge_cols=edge_cols,
        original_signed_weights=edge_original)
    positions = {int(e): i for i, e in enumerate(cal['edge_ids'])}
    values = cal['values'].copy()
    for i, edge in enumerate(edge_ids):
        if int(edge) not in positions:
            raise ValueError('Fitted original edge absent from E105 calibration')
        values[positions[int(edge)]] = np.float32(base.data[edge]*efficacy[i])
    changed = patch_weights(base, cal['edge_ids'], cal['source_values'], values)
    exact(changed.indptr, current.indptr, 'Original CSR row topology')
    exact(changed.indices, current.indices, 'Original CSR column topology')
    other = np.ones(len(changed.data), dtype=bool)
    other[edge_ids] = False
    exact(changed.data[other], current.data[other], 'All nonfit weights including incoming sources')
    exact(np.sign(changed.data[edge_ids]), np.sign(base.data[edge_ids]),
          'Original signed source-to-MN edges')
    actual_w = changed[cal['motor_ids']][:, np.unique(np.r_[cal['input_ids'],
        cal['intrinsic_tonic_ids'], cal['coupling_ids']])].toarray().astype(float)
    with torch.no_grad():
        fc, ft, fr, fa, fi = full_predict(tensor(actual_w))
    ledger['final_validation_calls'] = 1
    final_feet = geometry.evaluate(fc[selected]) if np.isfinite(fc).all() else np.full_like(target_feet, np.nan)
    np.savez_compressed(out/'prediction.npz', command=fc, tendon=ft,
        motor_rate=fr, muscle_activation=fa, current=fi,
        feet=final_feet, actual_float32_source_weights=actual_w,
        target=target, target_feet=target_feet, startup_mask=startup_mask,
        startup_target_command=startup_q, startup_target_tendon=startup_tendon)
    if not all(np.isfinite(v).all() for v in (fc, ft, fr, fa, fi, final_feet)):
        raise ValueError('Nonfinite final actual-float32 native output')
    final_metrics = metrics(fc, final_feet, ft, mask, target, target_feet,
        joint_scale, foot_scale)
    relative = native_shape_screen(final_metrics, baseline_metrics)
    absolute = absolute_shape_screen(final_metrics)
    global_startup = startup_screen(fc, ft, startup_q, startup_tendon,
                                    times, startup_mask)
    supported = bool(relative['supported'] and absolute['supported']
                     and global_startup['supported'])
    outcome = {'experiment': 'M3-E108-block-constrained-recruitment',
        'validity': 'valid_conditional_fit' if time.perf_counter()-started <= 225 else 'wall_limit_incomplete',
        'hypothesis_verdict': 'conditional_shape_and_startup_supported' if supported else 'not_supported',
        'local_supported': supported, 'candidate': True,
        'conditional_only': True, 'runtime_reference_phase': False,
        'body_validated': False, 'adoption': 'not_applicable',
        'causal_caveat': 'Actual E105 source and conditional background include closed-loop neural and body feedback; offline fit does not prove candidate closed-loop causality.',
        'F1_F8': 'not_assessed', 'baseline_output': baseline_metrics,
        'final_output': final_metrics, 'relative_shape_screen': relative,
        'absolute_E101_shape_screen': absolute,
        'global_original99_startup_screen': global_startup,
        'block_summaries': block_summaries,
        'reference_reconstruction_rms_rad': reference_rms,
        'fit_wall_s': ledger['fit_wall_s'],
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'all_nonfit_weights_exact': True, 'CSR_topology_exact': True,
        'original_edge_signs_exact': True, 'all_38_tau_exact': True}
    save_json(out/'native-selection-and-screens.json', outcome)
    final_cal = {k: v.copy() for k, v in cal.items()}
    final_cal['values'] = values
    final_cal['individual_edge_efficacies'] = efficacy
    final_cal['conditional_only'] = np.array(True)
    final_cal['runtime_reference_phase'] = np.array(False)
    for key, prior in cal.items():
        if key not in ('values', 'individual_edge_efficacies'):
            exact(final_cal[key], prior, 'E108 nonfit calibration '+key)
    np.savez_compressed(out/'calibration.npz', **final_cal)
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--graph', type=Path,
        default=ROOT/'data/acquisitions/banc888-windows-20260910/bundle')
    parser.add_argument('--out', type=Path, default=BASE/'e108-block-constrained-recruitment-v2')
    parser.add_argument('--prior-shared-wall-s', type=float, required=True)
    args = parser.parse_args()
    main(args.root, args.graph, args.out,
         prior_shared_wall_s=args.prior_shared_wall_s)
