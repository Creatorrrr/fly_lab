"""E116 saved-gradient startup tangent LP diagnostic; no model or calibration run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DESIGN_SHA = 'bb687fcc0381a140286623f589e101ee9c16f76d2b11750d0ef9aeb6c938c7a8'
FREEZE_SHA = 'c782ed3c9f60dcb8cfc13e044293b31073cc62a928415d50e9af2fd17a54367a'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
VARIANTS = ('shape_box', 'startup_box')
STARTUP_METRICS = ('q_RMS', 'q_max', 'LTM_RMS', 'LTM_max')
LOCAL_RADIUS = .01
SELECTOR_CAP = (1-1e-4)**2
LP_LIMIT_S = 1.
PROCESS_LIMIT_S = 20.
PRIOR_PROCESS_S = 700.
SHARED_PROCESS_LIMIT_S = 730.
VERIFICATION_TOLERANCE = 1e-8
SOLVER_OPTIONS = {'presolve': True, 'time_limit': LP_LIMIT_S,
    'primal_feasibility_tolerance': 1e-9,
    'dual_feasibility_tolerance': 1e-9, 'threads': 1}


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


def reviewed_inputs(root):
    """Hash and algebra preflight only; never solves an original-gradient LP."""
    root = Path(root)
    freeze_path = root/'e116-frozen-design.json'
    require(sha256(ROOT/'docs/BANC_M3_E116_STARTUP_TANGENT_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(freeze_path) == FREEZE_SHA,
            'E116 frozen design changed')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    require(freeze['actual_execution_authorized'] is False and
            freeze['linear_program_calls'] == 12 and
            freeze['LP_time_limit_s'] == LP_LIMIT_S and
            freeze['local_delta_radius'] == LOCAL_RADIUS and
            freeze['selector_cap'] == SELECTOR_CAP and
            freeze['method'] == 'highs-ds' and
            freeze['prior_conservative_process_s'] == PRIOR_PROCESS_S and
            freeze['shared_process_limit_s'] == SHARED_PROCESS_LIMIT_S,
            'E116 frozen resource/algebra contract changed')
    for relative, expected in freeze['input_sha256'].items():
        require(sha256(root/relative) == expected,
                f'E116 frozen input hash changed: {relative}')
    review = json.loads((root/'e115-context-trust-fit/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    require(review['accepted_as_valid_failed_conditional_fit'] is True and
            review['robust_local_supported'] is False,
            'E115 accepted failed-fit review changed')
    manifest = json.loads((root/'e115-context-trust-fit/original-output-manifest.json')
                          .read_text(encoding='utf-8'))
    original_hashes = {row['path']: row['sha256']
                       for row in manifest['original_files']}
    require(len(original_hashes) == manifest['original_file_count'] == 3043,
            'E115 original-output manifest changed')
    for relative, expected in freeze['input_sha256'].items():
        prefix = 'e115-context-trust-fit/'
        if relative.startswith(prefix) and relative != prefix+'parent-final-review.json' \
                and relative != prefix+'original-output-manifest.json':
            require(original_hashes.get(relative[len(prefix):]) == expected,
                    f'E115 original artifact not pinned by manifest: {relative}')
    parent = load_npz(root/'parent-e115-data-review-raw.npz')
    selected = load_npz(root/'e115-context-trust-fit/selected-delta.npz')
    mask = parent['leg_edge_mask']
    delta = selected['delta']
    lower, upper = parent['delta_lower'], parent['delta_upper']
    require(mask.shape == (6, 1572) and mask.dtype == np.bool_ and
            np.array_equal(mask.sum(axis=0), np.ones(1572)) and
            delta.shape == lower.shape == upper.shape == (1572,) and
            all(np.isfinite(x).all() for x in (delta, lower, upper)) and
            np.all(lower <= delta) and np.all(delta <= upper),
            'Selected E115 solver delta/partition/bounds changed')
    records = {}
    for leg_index, leg in enumerate(LEGS):
        choice = json.loads((root/f'e115-context-trust-fit/selection-{leg}.json')
                            .read_text(encoding='utf-8'))
        record = choice['best_feasible']
        index = freeze['selected_indices'][leg]
        require(choice['leg'] == leg and isinstance(record, dict) and
                record['index'] == index and
                choice['unique_deltas'] >= index and
                index >= 1,
                f'E115 selected record/index changed: {leg}')
        raw = load_npz(root/f'e115-context-trust-fit/eval-{leg}-{index:04d}.npz')
        edges = np.flatnonzero(mask[leg_index])
        n = len(edges)
        require(raw['delta'].shape == (n,) and raw['parts'].shape == (2, 4)
                and raw['losses'].shape == (2,) and raw['ratios'].shape == (2, 4)
                and raw['jacobian'].shape == (10, n) and
                all(np.isfinite(raw[key]).all() for key in
                    ('delta', 'parts', 'losses', 'ratios', 'jacobian')) and
                np.array_equal(raw['delta'], delta[edges]) and
                np.array_equal(raw['delta'], np.asarray(record['delta'])) and
                np.array_equal(raw['parts'], np.asarray(record['parts'])) and
                np.array_equal(raw['losses'], np.asarray(record['losses'])) and
                np.array_equal(raw['ratios'], np.asarray(record['ratios'])) and
                np.max(np.abs(raw['parts'].sum(axis=1)-raw['losses'])) <= 1e-12 and
                np.all(raw['ratios'] <= SELECTOR_CAP),
                f'E115 selected solver origin/gradient changed: {leg}')
        records[leg] = {'index': index, 'edges': edges, 'delta': raw['delta'],
            'losses': raw['losses'], 'ratios': raw['ratios'],
            'parts': raw['parts'], 'jacobian': raw['jacobian'],
            'bounds': np.stack((lower[edges], upper[edges]), axis=1),
            'source_sha256': freeze['input_sha256'][
                f'e115-context-trust-fit/eval-{leg}-{index:04d}.npz']}
    return {'freeze': freeze, 'records': records,
            'manifest_sha256': freeze['input_sha256'][
                'e115-context-trust-fit/original-output-manifest.json']}


def build_lp(origin, variant):
    """x=(h,t), with shape rows first and eight ordered startup rows second."""
    if variant not in VARIANTS:
        raise ValueError('Unknown E116 LP variant')
    delta = np.asarray(origin['delta'], dtype=np.float64)
    losses = np.asarray(origin['losses'], dtype=np.float64)
    ratios = np.asarray(origin['ratios'], dtype=np.float64)
    jac = np.asarray(origin['jacobian'], dtype=np.float64)
    bounds = np.asarray(origin['bounds'], dtype=np.float64)
    n = len(delta)
    if n == 0 or losses.shape != (2,) or ratios.shape != (2, 4) or \
            jac.shape != (10, n) or bounds.shape != (n, 2) or \
            not all(np.isfinite(x).all() for x in
                    (delta, losses, ratios, jac, bounds)) or \
            np.any(delta < bounds[:, 0]) or np.any(delta > bounds[:, 1]):
        raise ValueError('Finite selected E115 solver origin required')
    h_lower = np.maximum(bounds[:, 0]-delta, -LOCAL_RADIUS)
    h_upper = np.minimum(bounds[:, 1]-delta, LOCAL_RADIUS)
    if np.any(h_lower > h_upper) or np.any(h_lower > 0) or np.any(h_upper < 0):
        raise ValueError('Empty local E116 h bound intersection')
    prior = 6e-5*float(np.sum(delta*delta))/1572.
    prior_gradient = (1.2e-4/1572.)*delta
    c = np.r_[prior_gradient, 1.]
    shape_a = np.column_stack((jac[:2], -np.ones(2)))
    shape_b = -losses
    labels = ['loss_E110', 'loss_E113']
    if variant == 'startup_box':
        startup_a = np.column_stack((jac[2:], np.zeros(8)))
        a_ub = np.vstack((shape_a, startup_a))
        b_ub = np.r_[shape_b, SELECTOR_CAP-ratios.ravel()]
        labels.extend(f'startup_{context}_{metric}' for context in
                      ('E110', 'E113') for metric in STARTUP_METRICS)
    else:
        a_ub, b_ub = shape_a, shape_b
    lower = np.r_[h_lower, 0.]
    upper = np.r_[h_upper, np.inf]
    return {'leg': origin.get('leg'), 'variant': variant, 'c': c,
        'A_ub': a_ub, 'b_ub': b_ub, 'lower': lower, 'upper': upper,
        'row_labels': labels, 'prior': prior,
        'origin_worst_objective': float(np.max(losses)+prior),
        'origin': origin}


def returned_arrays(result):
    """Materialize every available raw HiGHS field before assessing status."""
    def array(obj, name):
        value = getattr(obj, name, None) if obj is not None else None
        return np.asarray([], dtype=np.float64) if value is None else \
            np.asarray(value, dtype=np.float64).copy()
    return {'x': array(result, 'x'), 'slack': array(result, 'slack'),
        'ineqlin_residual': array(getattr(result, 'ineqlin', None), 'residual'),
        'ineqlin_marginal': array(getattr(result, 'ineqlin', None), 'marginals'),
        'lower_residual': array(getattr(result, 'lower', None), 'residual'),
        'lower_marginal': array(getattr(result, 'lower', None), 'marginals'),
        'upper_residual': array(getattr(result, 'upper', None), 'residual'),
        'upper_marginal': array(getattr(result, 'upper', None), 'marginals')}


def scaled_limit(*values):
    finite = [np.abs(np.asarray(value, dtype=np.float64)).ravel()
              for value in values]
    magnitudes = [float(np.max(item[np.isfinite(item)])) for item in finite
                  if np.isfinite(item).any()]
    return VERIFICATION_TOLERANCE*(1.+max(magnitudes, default=0.))


def validate_solution(problem, result, raw, warnings_seen=()):
    """Check primal/dual/KKT with SciPy's marginal sign convention."""
    c, a, b = problem['c'], problem['A_ub'], problem['b_ub']
    lower, upper = problem['lower'], problem['upper']
    n, m = len(c), len(b)
    require(int(result.status) == 0 and bool(result.success),
            f'E116 LP did not finish optimally: {result.status}: {result.message}')
    require(not [row for row in warnings_seen if not row['expected_threads_forwarding']],
            'Unexpected E116 SciPy/HiGHS warning')
    require(all(raw[key].shape == shape for key, shape in (
        ('x', (n,)), ('ineqlin_residual', (m,)),
        ('ineqlin_marginal', (m,)), ('lower_residual', (n,)),
        ('lower_marginal', (n,)), ('upper_residual', (n,)),
        ('upper_marginal', (n,)))) and
        all(np.isfinite(raw[key]).all() for key in
            ('x', 'ineqlin_residual', 'ineqlin_marginal',
             'lower_residual', 'lower_marginal', 'upper_marginal')) and
        np.isfinite(raw['upper_residual'][:-1]).all() and
        np.isinf(raw['upper_residual'][-1]) and
        math.isfinite(float(result.fun)),
        'Incomplete/nonfinite E116 LP primal or dual result')
    x = raw['x']
    inequality = b-a@x
    lower_gap = x-lower
    upper_gap = upper[:-1]-x[:-1]
    mi, ml, mu = (raw[key] for key in
                  ('ineqlin_marginal', 'lower_marginal', 'upper_marginal'))
    stationarity = c-a.T@mi-ml-mu
    dual = float(b@mi+lower@ml+upper[:-1]@mu[:-1])
    primal = float(c@x)
    checks = {
        'objective': abs(float(result.fun)-primal) <=
            scaled_limit(result.fun, primal),
        'inequality_residual': np.max(np.abs(raw['ineqlin_residual']-inequality))
            <= scaled_limit(raw['ineqlin_residual'], inequality),
        'lower_residual': np.max(np.abs(raw['lower_residual']-lower_gap))
            <= scaled_limit(raw['lower_residual'], lower_gap),
        'upper_residual': np.max(np.abs(raw['upper_residual'][:-1]-upper_gap))
            <= scaled_limit(raw['upper_residual'][:-1], upper_gap),
        'primal_feasible': np.min(np.r_[inequality, lower_gap, upper_gap]) >=
            -scaled_limit(b, a@x, lower, upper[:-1], x),
        'dual_sign': np.max(mi) <= scaled_limit(mi) and
            np.min(ml) >= -scaled_limit(ml) and
            np.max(mu) <= scaled_limit(mu),
        'unbounded_t_upper_marginal_zero': abs(mu[-1]) <= scaled_limit(mu[-1]),
        'stationarity': np.max(np.abs(stationarity)) <=
            scaled_limit(c, a.T@mi, ml, mu),
        'complementarity': max(
            np.max(np.abs(inequality*mi)),
            np.max(np.abs(lower_gap*ml)),
            np.max(np.abs(upper_gap*mu[:-1]))) <=
            scaled_limit(primal, dual),
        'primal_dual_gap': abs(primal-dual) <= scaled_limit(primal, dual),
    }
    require(all(checks.values()), f'E116 LP numerical/KKT validation failed: {checks}')
    origin = problem['origin']
    h = x[:-1]
    loss_prediction = origin['losses']+origin['jacobian'][:2]@h
    ratio_prediction = origin['ratios'].ravel()+origin['jacobian'][2:]@h
    active_tolerance = scaled_limit(x, lower, upper[:-1])
    return {'checks': checks, 'primal': primal, 'dual': dual,
        'primal_dual_gap': primal-dual,
        'objective_with_original_prior': primal+problem['prior'],
        'origin_worst_objective': problem['origin_worst_objective'],
        'linearized_reduction': problem['origin_worst_objective']-
            (primal+problem['prior']),
        'predicted_losses': loss_prediction,
        'predicted_startup_ratios': ratio_prediction.reshape(2, 4),
        'startup_prediction_feasible': bool(np.all(
            ratio_prediction <= SELECTOR_CAP+scaled_limit(ratio_prediction))),
        'active_lower_h_count': int(np.count_nonzero(
            np.abs(h-lower[:-1]) <= active_tolerance)),
        'active_upper_h_count': int(np.count_nonzero(
            np.abs(h-upper[:-1]) <= active_tolerance)),
        'row_labels': problem['row_labels'],
        'row_slack': inequality, 'row_marginal': mi,
        'stationarity_max_abs': float(np.max(np.abs(stationarity))),
        'complementarity_max_abs': float(max(
            np.max(np.abs(inequality*mi)),
            np.max(np.abs(lower_gap*ml)),
            np.max(np.abs(upper_gap*mu[:-1]))))}


def warning_rows(caught):
    rows = []
    for item in caught:
        message = str(item.message)
        rows.append({'category': item.category.__name__, 'message': message,
            'expected_threads_forwarding': 'Unrecognized options' in message and
                "'threads': 1" in message and 'HiGHS' in message})
    return rows


def result_metadata(result, warnings_seen):
    """Preserve every available scalar even if its normal JSON cannot be saved."""
    return {'status': getattr(result, 'status', None),
        'success': getattr(result, 'success', None),
        'message': str(getattr(result, 'message', 'unavailable')),
        'nit': getattr(result, 'nit', None),
        'fun': getattr(result, 'fun', None),
        'warnings': warnings_seen}


def rescue_arrays(directory, raw, primary, label):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as secondary:  # noqa: BLE001
        primary.add_note(f'E116 secondary {label} rescue directory: {secondary}')
        return
    for key, value in raw.items():
        try:
            np.save(directory/f'{key}.npy', np.asarray(value), allow_pickle=False)
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E116 secondary {label}.{key} rescue: {secondary}')


def solve_recorded_lp(problem, out, ledger, persist, deadlines, solver=None):
    """One LP: durable input/start, then raw returned result before assertions."""
    solver = linprog if solver is None else solver
    leg, variant = problem['leg'], problem['variant']
    stem = f'lp-{leg}-{variant}'
    input_raw = {'c': problem['c'], 'A_ub': problem['A_ub'],
        'b_ub': problem['b_ub'], 'lower': problem['lower'],
        'upper': problem['upper'],
        'row_labels': np.asarray(problem['row_labels']),
        'origin_delta': problem['origin']['delta'],
        'origin_losses': problem['origin']['losses'],
        'origin_ratios': problem['origin']['ratios'],
        'origin_jacobian': problem['origin']['jacobian'],
        'origin_npz_sha256': np.asarray(problem['origin']['source_sha256'])}
    result = None
    returned_raw = {}
    seen = []
    try:
        deadlines()
        np.savez_compressed(out/f'{stem}-input.npz', **input_raw)
        ledger['lp_input_raw_saved'] += 1
        ledger['active_lp'] = stem
        persist()
        ledger['lp_started'] += 1
        try:
            persist()  # A failed mandatory start receipt prevents the call.
        except BaseException:
            ledger['lp_started'] -= 1
            ledger['start_receipt_failed'] = stem
            raise
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = solver(problem['c'], A_ub=problem['A_ub'],
                b_ub=problem['b_ub'],
                bounds=list(zip(problem['lower'][:-1],
                                problem['upper'][:-1]))+[(0., None)],
                method='highs-ds', options=SOLVER_OPTIONS.copy())
        seen = warning_rows(caught)
        ledger['lp_returned'] += 1
        persist()
        returned_raw = returned_arrays(result)
        np.savez_compressed(out/f'{stem}-returned.npz', **returned_raw)
        save_json(out/f'{stem}-returned.json', dict(
            result_metadata(result, seen), raw_npz_saved_before_assertions=True))
        ledger['lp_raw_saved'] += 1
        persist()
        deadlines()
        diagnostic = validate_solution(problem, result, returned_raw, seen)
        save_json(out/f'{stem}-diagnostic.json', diagnostic)
        ledger['active_lp'] = None
        persist()
        return diagnostic
    except BaseException as primary:
        if result is not None and not returned_raw:
            try:
                returned_raw = returned_arrays(result)
            except Exception as secondary:  # noqa: BLE001
                primary.add_note(f'E116 secondary returned materialization: {secondary}')
        destination = out/f'error-raw-{stem}'
        rescue_arrays(destination/'input', input_raw, primary, 'LP input')
        if returned_raw:
            rescue_arrays(destination/'returned', returned_raw, primary,
                          'LP returned')
        try:
            save_json(destination/'error-metadata.json', dict(
                result_metadata(result, seen),
                primary_type=type(primary).__name__,
                primary_message=str(primary)))
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E116 secondary LP metadata rescue: {secondary}')
        raise


def run_once(root, out):
    """The separately authorized twelve-LP process; no model response."""
    root, out = Path(root), Path(out)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'experiment': 'E116-startup-tangent', 'stage': 'created',
        'prior_conservative_process_s': PRIOR_PROCESS_S,
        'shared_process_limit_s': SHARED_PROCESS_LIMIT_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'lp_input_raw_saved': 0, 'lp_started': 0, 'lp_returned': 0,
        'lp_raw_saved': 0, 'active_lp': None,
        'conditional_responses': 0, 'original_data_fit_calls': 0,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'calibration_outputs': 0, 'secondary_errors': []}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E116 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
        except Exception as secondary:
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E116 secondary wall clock: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadlines():
        elapsed = time.monotonic()-started
        if elapsed > PROCESS_LIMIT_S or \
                PRIOR_PROCESS_S+elapsed > SHARED_PROCESS_LIMIT_S:
            raise TimeoutError('E116 internal/shared process wall limit')

    def failure_save(filename, payload, primary):
        try:
            save_json(out/filename, payload)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E116 secondary {filename} save: {secondary}')

    primary = None
    try:
        for original, name in (
            (ROOT/'tools/analyze_c_banc_m3_startup_tangent.py', 'runner.py'),
            (ROOT/'tests/test_c_banc_startup_tangent.py',
             'test_c_banc_startup_tangent.py'),
            (ROOT/'docs/BANC_M3_E116_STARTUP_TANGENT_DESIGN_20260920.md',
             'frozen-design.md'),
            (root/'e116-frozen-design.json', 'e116-frozen-design.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        data = reviewed_inputs(root)
        require(scipy.__version__ == '1.18.1' and all(
            os.environ.get(key) == '1' for key in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Reviewed SciPy1.18.1/one-thread environment required')
        ledger['environment'] = {'scipy': scipy.__version__,
            'omp': os.environ['OMP_NUM_THREADS'],
            'mkl': os.environ['MKL_NUM_THREADS'],
            'openblas': os.environ['OPENBLAS_NUM_THREADS']}
        save_json(out/'input-provenance.json', {'design_sha256': DESIGN_SHA,
            'freeze_sha256': FREEZE_SHA,
            'input_sha256': data['freeze']['input_sha256'],
            'manifest_sha256': data['manifest_sha256']})
        deadlines()
        ledger['stage'] = 'lp'
        persist()
        reports = {}
        for leg in LEGS:
            reports[leg] = {}
            for variant in VARIANTS:
                deadlines()
                origin = dict(data['records'][leg], leg=leg)
                problem = build_lp(origin, variant)
                reports[leg][variant] = solve_recorded_lp(
                    problem, out, ledger, persist, deadlines)
            shape = reports[leg]['shape_box']
            startup = reports[leg]['startup_box']
            reports[leg]['startup_constraint_cost'] = \
                startup['objective_with_original_prior']-\
                shape['objective_with_original_prior']
            save_json(out/f'comparison-{leg}.json', reports[leg])
        require(ledger['lp_started'] == ledger['lp_returned'] ==
                ledger['lp_raw_saved'] == 12,
                'E116 did not complete exactly twelve original-gradient LPs')
        deadlines()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_saved_gradient_diagnostic',
            'diagnostic_only': True, 'calibration_output': False,
            'no_new_response_or_native_run': True,
            'leg_reports': reports, 'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', ledger['primary_error'], error)
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
        'verification/banc-natural-gait-m3-20260919/e116-startup-tangent')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
