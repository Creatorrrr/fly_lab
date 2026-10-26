"""E120 finite-box local feasibility from saved E119 gradients only."""

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
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
INDICES = {'lf': 100, 'lm': 99, 'lh': 100, 'rf': 16, 'rm': 100, 'rh': 99}
DESIGN_SHA = '4d5dd59c40c8eb7754d165e424d90761a0229606f7a1b0ff6760a622b9cb26eb'
FREEZE_SHA = 'bd4cc72f1dc1fbf381e73d79bd742b84dde4b25c26b948cd844b24a874f439e3'
CAP = (1-1e-4)**2
INNER_Q_RMS, INNER_Q_MAX = .018, .04
INNER_Z_RMS, INNER_Z_MAX = .009, .025
RADIUS = .01
MAX_STARTUP_LP = 8
MAX_TOTAL_LP = 54
MAX_CUTS = 32
RAY_STEPS = 60
PROCESS_LIMIT_S = 80.
PRIOR_S = 771.
SHARED_LIMIT_S = 871.
LP_OPTIONS = {'presolve': True, 'time_limit': 1.,
    'primal_feasibility_tolerance': 1e-9,
    'dual_feasibility_tolerance': 1e-9, 'threads': 1}
EXPECTED_THREAD_WARNING = (
    "Unrecognized options detected: {'threads': 1}. "
    'These will be passed to HiGHS verbatim.')
RMS_ORDER = (('q', 0), ('q', 1), ('z', 0), ('z', 1))


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
    """Read frozen saved E119 records; construct no solver, model or CUDA state."""
    root = Path(root)
    require(sha256(ROOT/'docs/BANC_M3_E120_LOCAL_FEASIBILITY_DESIGN_20260920.md')
            == DESIGN_SHA and sha256(root/'e120-frozen-design.json') ==
            FREEZE_SHA, 'E120 design or frozen input list changed')
    freeze = json.loads((root/'e120-frozen-design.json').read_text(
        encoding='utf-8'))
    require(freeze['actual_execution_authorized'] is False and
            freeze['selected_indices'] == INDICES and
            freeze['local_delta_radius'] == RADIUS and
            freeze['selector_cap'] == CAP and
            freeze['method'] == 'highs-ds' and
            freeze['LP_time_limit_s'] == LP_OPTIONS['time_limit'] and
            freeze['max_startup_LP_per_leg'] == MAX_STARTUP_LP and
            freeze['max_original_LP_calls'] == MAX_TOTAL_LP and
            freeze['max_RMS_cuts_per_leg'] == MAX_CUTS and
            freeze['ray_bisection_iterations'] == RAY_STEPS and
            freeze['internal_limit_s'] == PROCESS_LIMIT_S and
            freeze['prior_conservative_process_s'] == PRIOR_S and
            freeze['shared_process_limit_s'] == SHARED_LIMIT_S and
            freeze['cut_padding_coefficient'] == 1e-12 and
            freeze['dual_allowance_coefficient'] == 1e-9 and
            freeze['KKT_tolerance'] == 1e-8 and
            freeze['gap_coefficient'] == 1e-6 and
            freeze['descent_coefficient'] == 1e-4 and
            freeze['original_responses'] == freeze['derivative_executions'] ==
            freeze['fit_calls'] == freeze['actual_neural_s'] ==
            freeze['actual_physical_s'] == 0 and
            freeze['F6_exposed'] is False and
            len(freeze['input_sha256']) == 25 and
            len(freeze['source_sha256']) == 18,
            'E120 frozen mathematical/resource contract changed')
    for relative, expected in freeze['input_sha256'].items():
        require(sha256(root/relative) == expected,
                f'E120 frozen input changed: {relative}')
    for relative, expected in freeze['source_sha256'].items():
        require(sha256(ROOT/relative) == expected,
                f'E120 reused source changed: {relative}')
    original = json.loads((root/'e119-samplewise-fit/'
        'original-output-manifest.json').read_text(encoding='utf-8'))
    parent = json.loads((root/'e119-samplewise-fit/'
        'parent-final-review.json').read_text(encoding='utf-8'))
    require(original['original_file_count'] == 643 and
            len(original['original_files']) == 643 and
            parent['accepted_as_valid_conditional_fit'] is True and
            parent['blocking_defects'] == [] and
            parent['robust_local_supported'] is False and
            parent['actual_neural_s'] == parent['actual_physical_s'] == 0,
            'E119 parent review/original manifest changed')
    original_map = {row['path']: row['sha256'] for row in
                    original['original_files']}
    selected = load_npz(root/'e119-samplewise-fit/selected-delta.npz')
    bounds = load_npz(root/'parent-e115-data-review-raw.npz')
    require(selected['delta'].shape == (1572,) and
            bounds['delta_lower'].shape == bounds['delta_upper'].shape ==
            (1572,) and bounds['leg_edge_mask'].shape == (6, 1572) and
            np.all(bounds['leg_edge_mask'].sum(axis=0) == 1),
            'E119 selected delta/E115 exact bounds partition changed')
    records = {}
    for leg_index, leg in enumerate(LEGS):
        index = INDICES[leg]
        relative = f'eval-{leg}-{index:03d}.npz'
        require(original_map[relative] == freeze['input_sha256'][
            f'e119-samplewise-fit/{relative}'],
            f'E119 selected raw not in original manifest: {leg}')
        raw = load_npz(root/'e119-samplewise-fit'/relative)
        choice = json.loads((root/'e119-samplewise-fit'/
            f'selection-{leg}.json').read_text(encoding='utf-8'))
        positions = np.flatnonzero(bounds['leg_edge_mask'][leg_index])
        local_bounds = np.stack((bounds['delta_lower'][positions],
                                 bounds['delta_upper'][positions]), axis=1)
        require(choice['best_feasible'] is not None and
                choice['best_feasible']['index'] == index and
                np.array_equal(raw['edge_positions'], positions) and
                np.array_equal(raw['edge_ids'], selected['edge_ids'][positions])
                and np.array_equal(raw['delta'], selected['delta'][positions])
                and np.array_equal(raw['bounds'], local_bounds) and
                np.array_equal(np.asarray(choice['best_feasible']['delta']),
                               raw['delta']) and
                np.array_equal(np.asarray(choice['best_feasible']['losses']),
                               raw['losses']) and
                np.array_equal(np.asarray(choice['best_feasible']['ratios']),
                               raw['ratios']) and
                np.all(raw['ratios'] <= CAP),
                f'E119 selected record/bounds mismatch: {leg}')
        raw['leg'] = leg
        raw['selected_index'] = index
        raw['source_sha256'] = freeze['input_sha256'][
            f'e119-samplewise-fit/{relative}']
        records[leg] = raw
    return {'freeze': freeze, 'records': records,
            'manifest_sha256': freeze['input_sha256'][
                'e119-samplewise-fit/original-output-manifest.json']}


def base_problem(origin):
    """Build fixed local q/LTM Jacobians and finite (h,t) box once per leg."""
    delta = np.asarray(origin['delta'], dtype=np.float64)
    bounds = np.asarray(origin['bounds'], dtype=np.float64)
    losses = np.asarray(origin['losses'], dtype=np.float64)
    gradient = np.asarray(origin['loss_gradient'], dtype=np.float64)
    q = np.asarray(origin['q_error'], dtype=np.float64)
    z = np.asarray(origin['tendon_error'], dtype=np.float64)
    v = np.asarray(origin['v'], dtype=np.float64)
    qco = np.asarray(origin['q_coefficient'], dtype=np.float64)
    zco = np.asarray(origin['ltm_coefficient'], dtype=np.float64)
    gain = np.asarray(origin['ltm_gain'], dtype=np.float64)
    n = len(delta)
    require(delta.ndim == 1 and n > 0 and bounds.shape == (n, 2) and
            losses.shape == (2,) and gradient.shape == (2, n) and
            q.shape == (2, 99, 7) and z.shape == (2, 99) and
            v.shape == (2, 99, n) and qco.shape == (7, n) and
            zco.shape == (n,) and gain.shape == (2, 99) and
            all(np.isfinite(x).all() for x in
                (delta, bounds, losses, gradient, q, z, v, qco, zco, gain))
            and np.all(bounds[:, 0] <= delta) and
            np.all(delta <= bounds[:, 1]),
            'Finite exact E119 selected compact factors/bounds required')
    lower_h = np.maximum(bounds[:, 0]-delta, -RADIUS)
    upper_h = np.minimum(bounds[:, 1]-delta, RADIUS)
    require(np.all(lower_h <= 0) and np.all(upper_h >= 0) and
            np.all(lower_h <= upper_h), 'E120 local box does not contain h=0')
    high_loss = losses+np.maximum(gradient*lower_h[None],
                                  gradient*upper_h[None]).sum(axis=1)
    upper_t = 1.+max(0., float(np.max(high_loss)))
    require(math.isfinite(upper_t) and upper_t > 0,
            'Finite redundant local epigraph upper bound required')
    jq = v[:, :, None, :]*qco[None, None]
    jz = v*gain[:, :, None]*zco[None, None]
    p = 1.2e-4*delta/1572.
    p0 = 6e-5*float(np.sum(delta*delta))/1572.
    result = {'leg': origin.get('leg'), 'source_sha256':
        origin.get('source_sha256'), 'delta': delta, 'original_bounds': bounds,
        'losses': losses, 'loss_gradient': gradient, 'q_error': q,
        'tendon_error': z, 'jq': jq, 'jz': jz, 'h_lower': lower_h,
        'h_upper': upper_h, 'lower': np.r_[lower_h, 0.],
        'upper': np.r_[upper_h, upper_t], 'p': p, 'p0': p0,
        'origin_objective': float(np.max(losses)+p0)}
    require(strict_feasible(result, np.zeros(n), 'startup_box')['feasible'],
            'E119 selected origin not strictly local startup feasible')
    return result


def rms_state(base, metric, context, h):
    if metric == 'q':
        error, jac, scale = (base['q_error'][context], base['jq'][context],
                             INNER_Q_RMS)
    elif metric == 'z':
        error, jac, scale = (base['tendon_error'][context],
                             base['jz'][context], INNER_Z_RMS)
    else:
        raise ValueError('Unknown E120 RMS metric')
    residual = error+np.einsum('...n,n->...', jac, h, optimize=False)
    flat = residual.reshape(-1)
    matrix = jac.reshape(len(flat), -1)
    g = float(np.dot(flat, flat)/(len(flat)*scale**2))
    grad = 2*(flat@matrix)/(len(flat)*scale**2)
    return g, grad, residual


def rms_cut(base, metric, context, center, iteration):
    center = np.asarray(center, dtype=np.float64)
    g, gradient, _ = rms_state(base, metric, context, center)
    radius = np.maximum(np.abs(base['h_lower']), np.abs(base['h_upper']))
    padding = 1e-12*(1+abs(g)+float(np.sum(np.abs(gradient)*
                                              (np.abs(center)+radius))))
    rhs = CAP-g+float(gradient@center)+padding
    return {'metric': metric, 'context': context, 'iteration': iteration,
        'center': center.copy(), 'g': g, 'gradient': gradient,
        'padding': padding, 'rhs': rhs}


def build_lp(base, variant, cuts=()):
    """Shape and ordered signed-sample/RMS-cut finite-box linear programs."""
    require(variant in ('shape_box', 'startup_box'),
            'Unknown E120 LP variant')
    n = len(base['delta'])
    gradient = base['loss_gradient']
    pieces = [np.column_stack((gradient, -np.ones(2)))]
    rhs = [-base['losses']]
    labels = ['shape_E110', 'shape_E113']
    if variant == 'startup_box':
        jq = base['jq'].reshape(1386, n, order='C')/INNER_Q_MAX
        jz = base['jz'].reshape(198, n, order='C')/INNER_Z_MAX
        q = base['q_error'].ravel(order='C')/INNER_Q_MAX
        z = base['tendon_error'].ravel(order='C')/INNER_Z_MAX
        for name, jac, error, sign in (
                ('plus_q', jq, q, 1), ('minus_q', jq, q, -1),
                ('plus_z', jz, z, 1), ('minus_z', jz, z, -1)):
            pieces.append(np.column_stack((sign*jac,
                                           np.zeros(len(error)))))
            rhs.append(math.sqrt(CAP)-sign*error)
            labels.extend(f'{name}_{i}' for i in range(len(error)))
        require(4 <= len(cuts) <= MAX_CUTS and
                [(row['metric'], row['context']) for row in cuts[:4]] ==
                list(RMS_ORDER), 'E120 initial four ordered RMS cuts missing')
        for index, cut in enumerate(cuts):
            pieces.append(np.r_[cut['gradient'], 0.].reshape(1, n+1))
            rhs.append(np.array([cut['rhs']]))
            labels.append(f'rms_{index:02d}_{cut["metric"]}_E'
                          f'{110 if cut["context"] == 0 else 113}')
    elif cuts:
        raise ValueError('Shape-only LP cannot contain startup cuts')
    a = np.vstack(pieces)
    b = np.concatenate(rhs)
    require(a.shape == (len(b), n+1) and len(labels) == len(b) and
            np.isfinite(a).all() and np.isfinite(b).all(),
            'Finite E120 LP row matrix required')
    return {'leg': base['leg'], 'variant': variant, 'c': np.r_[base['p'], 1.],
        'A_ub': a, 'b_ub': b, 'lower': base['lower'].copy(),
        'upper': base['upper'].copy(), 'row_labels': labels,
        'cuts': list(cuts), 'base': base}


def strict_feasible(base, h, variant):
    """Original local box and CAP, with no KKT tolerance or cut padding."""
    h = np.asarray(h, dtype=np.float64)
    require(variant in ('shape_box', 'startup_box') and
            h.shape == base['h_lower'].shape and np.isfinite(h).all(),
            'Finite aligned E120 h required')
    box = bool(np.all(h >= base['h_lower']) and
               np.all(h <= base['h_upper']))
    q = base['q_error']+np.einsum('btjn,n->btj', base['jq'], h,
                                  optimize=False)
    z = base['tendon_error']+np.einsum('btn,n->bt', base['jz'], h,
                                       optimize=False)
    ratios = np.stack((np.mean(q*q, axis=(1, 2))/INNER_Q_RMS**2,
        np.max(q*q, axis=(1, 2))/INNER_Q_MAX**2,
        np.mean(z*z, axis=1)/INNER_Z_RMS**2,
        np.max(z*z, axis=1)/INNER_Z_MAX**2), axis=1)
    startup = bool(np.all(ratios <= CAP))
    return {'feasible': box and (variant == 'shape_box' or startup),
        'box_feasible': box, 'startup_feasible': startup,
        'ratios': ratios, 'q_residual': q, 'tendon_residual': z,
        'lower_margin': h-base['h_lower'],
        'upper_margin': base['h_upper']-h}


def local_objective(base, h):
    h = np.asarray(h, dtype=np.float64)
    t = max(0., float(np.max(base['losses']+base['loss_gradient']@h)))
    return {'t': t, 'linearized_prior': float(base['p0']+base['p']@h),
        'objective': float(t+base['p0']+base['p']@h),
        'predicted_losses': base['losses']+base['loss_gradient']@h}


def ray_repair(base, h, variant):
    """Exactly sixty checks along the origin ray when LP h is not strict."""
    h = np.asarray(h, dtype=np.float64)
    require(strict_feasible(base, np.zeros_like(h), variant)['feasible'],
            'E120 h=0 must be strictly feasible')
    original = strict_feasible(base, h, variant)
    if original['feasible']:
        alpha, repaired, steps = 1., h.copy(), 0
    else:
        lo, hi = 0., 1.
        repaired = np.zeros_like(h)
        for _ in range(RAY_STEPS):
            middle = (lo+hi)/2
            candidate = middle*h
            if strict_feasible(base, candidate, variant)['feasible']:
                lo, repaired = middle, candidate.copy()
            else:
                hi = middle
        alpha, steps = lo, RAY_STEPS
    status = strict_feasible(base, repaired, variant)
    require(status['feasible'], 'E120 repaired h not strictly feasible')
    return {'raw_h': h.copy(), 'raw_feasibility': original,
        'repaired_h': repaired, 'alpha': alpha, 'bisection_steps': steps,
        'repaired_feasibility': status,
        'repaired_objective': local_objective(base, repaired)}


def returned_arrays(result, raw=None, primary=None):
    """Collect all available HiGHS arrays, including on failure rescue."""
    raw = {} if raw is None else raw
    fields = {'x': (result, 'x'), 'slack': (result, 'slack'),
        'ineqlin_residual': (getattr(result, 'ineqlin', None), 'residual'),
        'ineqlin_marginal': (getattr(result, 'ineqlin', None), 'marginals'),
        'lower_residual': (getattr(result, 'lower', None), 'residual'),
        'lower_marginal': (getattr(result, 'lower', None), 'marginals'),
        'upper_residual': (getattr(result, 'upper', None), 'residual'),
        'upper_marginal': (getattr(result, 'upper', None), 'marginals')}
    for key, (container, field) in fields.items():
        if key in raw:
            continue
        try:
            value = getattr(container, field, None)
            raw[key] = (np.asarray([], dtype=np.float64) if value is None else
                        np.asarray(value, dtype=np.float64).copy())
        except Exception as secondary:
            if primary is None:
                raise
            primary.add_note(f'E120 secondary returned {key}: {secondary}')
    return raw


def warning_rows(caught):
    return [{'category': item.category.__name__,
        'message': str(item.message),
        'expected_threads_forwarding': item.category.__name__ ==
            'OptimizeWarning' and str(item.message) == EXPECTED_THREAD_WARNING}
        for item in caught]


def result_metadata(result, warnings_seen):
    return {'status': getattr(result, 'status', None),
        'success': getattr(result, 'success', None),
        'message': str(getattr(result, 'message', 'unavailable')),
        'nit': getattr(result, 'nit', None),
        'fun': getattr(result, 'fun', None), 'warnings': warnings_seen}


def scaled_limit(*values):
    finite = [np.asarray(value, dtype=np.float64).ravel() for value in values]
    maximum = max((float(np.max(np.abs(row[np.isfinite(row)])))
                   for row in finite if np.isfinite(row).any()), default=0.)
    return 1e-8*(1.+maximum)


def dual_certificate(problem, raw):
    """Conservative finite-box lower bound for any projected y<=0."""
    a, b, c = problem['A_ub'], problem['b_ub'], problem['c']
    lower, upper = problem['lower'], problem['upper']
    y_raw = np.asarray(raw['ineqlin_marginal'], dtype=np.float64)
    require(y_raw.shape == b.shape and np.isfinite(y_raw).all() and
            np.isfinite(lower).all() and np.isfinite(upper).all(),
            'Finite E120 box and inequality marginals required')
    y = np.minimum(y_raw, 0.)
    d = c-a.T@y
    corner = np.minimum(d*lower, d*upper)
    dual = float(b@y+np.sum(corner)+problem['base']['p0'])
    radius = np.maximum(np.abs(lower), np.abs(upper))
    allowance = 1e-9*(1+float(np.abs(b)@np.abs(y))+
        float(np.sum((np.abs(c)+np.abs(a).T@np.abs(y))*radius)))
    lower_bound = float(np.nextafter(dual-allowance, -np.inf))
    require(all(math.isfinite(x) for x in (dual, allowance, lower_bound)),
            'Nonfinite E120 projected finite-box dual')
    return {'marginal_raw': y_raw.copy(), 'marginal_projected': y,
        'stationarity_remainder': d, 'box_corner_terms': corner,
        'finite_box_dual': dual, 'numerical_allowance': allowance,
        'lower_bound': lower_bound,
        'certificate_semantics': 'conservative float64 numerical lower bound'}


def validate_solution(problem, result, raw, warnings_seen):
    """Independently check returned optimum/KKT; strict h is a separate test."""
    c, a, b = problem['c'], problem['A_ub'], problem['b_ub']
    lower, upper = problem['lower'], problem['upper']
    n, m = len(c), len(b)
    require(int(result.status) == 0 and bool(result.success),
            f'E120 LP did not finish optimally: {result.status}: {result.message}')
    require(all(row['expected_threads_forwarding'] for row in warnings_seen),
            'Unexpected E120 SciPy/HiGHS warning')
    shapes = {'x': (n,), 'slack': (m,), 'ineqlin_residual': (m,),
        'ineqlin_marginal': (m,), 'lower_residual': (n,),
        'lower_marginal': (n,), 'upper_residual': (n,),
        'upper_marginal': (n,)}
    require(all(raw[key].shape == shape for key, shape in shapes.items()) and
            all(np.isfinite(raw[key]).all() for key in shapes) and
            math.isfinite(float(result.fun)),
            'Incomplete/nonfinite E120 HiGHS returned arrays')
    x = raw['x']
    mi, ml, mu = (raw[key] for key in
                  ('ineqlin_marginal', 'lower_marginal', 'upper_marginal'))
    gap_i, gap_l, gap_u = b-a@x, x-lower, upper-x
    stationarity = c-a.T@mi-ml-mu
    primal = float(c@x)
    dual = float(b@mi+lower@ml+upper@mu)
    complementarity = max(float(np.max(np.abs(gap_i*mi))),
        float(np.max(np.abs(gap_l*ml))),
        float(np.max(np.abs(gap_u*mu))))
    checks = {
        'objective': abs(float(result.fun)-primal) <=
            scaled_limit(result.fun, primal),
        'slack': np.max(np.abs(raw['slack']-gap_i)) <=
            scaled_limit(raw['slack'], gap_i),
        'inequality_residual': np.max(np.abs(raw['ineqlin_residual']-gap_i)) <=
            scaled_limit(raw['ineqlin_residual'], gap_i),
        'lower_residual': np.max(np.abs(raw['lower_residual']-gap_l)) <=
            scaled_limit(raw['lower_residual'], gap_l),
        'upper_residual': np.max(np.abs(raw['upper_residual']-gap_u)) <=
            scaled_limit(raw['upper_residual'], gap_u),
        'primal_sign': min(np.min(gap_i), np.min(gap_l), np.min(gap_u)) >=
            -scaled_limit(b, a@x, lower, upper, x),
        'dual_sign': np.max(mi) <= scaled_limit(mi) and
            np.min(ml) >= -scaled_limit(ml) and
            np.max(mu) <= scaled_limit(mu),
        'stationarity': np.max(np.abs(stationarity)) <=
            scaled_limit(c, a.T@mi, ml, mu),
        'complementarity': complementarity <=
            scaled_limit(primal, dual),
        'primal_dual_gap': abs(primal-dual) <=
            scaled_limit(primal, dual),
    }
    certificate = dual_certificate(problem, raw)
    checks['finite_box_dual_objective'] = (
        abs(certificate['finite_box_dual']-(dual+problem['base']['p0'])) <=
        scaled_limit(certificate['finite_box_dual'], dual,
                     certificate['numerical_allowance']))
    require(all(checks.values()), f'E120 LP numerical/KKT check failed: {checks}')
    return {'checks': checks, 'primal': primal+problem['base']['p0'],
        'dual': dual+problem['base']['p0'],
        'primal_dual_gap': primal-dual, 'row_slack': gap_i,
        'stationarity': stationarity,
        'complementarity_max_abs': complementarity,
        'certificate': certificate}


def rescue_arrays(directory, raw, primary, label):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as secondary:  # noqa: BLE001
        primary.add_note(f'E120 secondary {label} directory: {secondary}')
        return
    for key, value in raw.items():
        try:
            np.save(directory/f'{key}.npy', np.asarray(value),
                    allow_pickle=False)
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E120 secondary {label}.{key}: {secondary}')


def cut_arrays(cuts, n):
    if not cuts:
        return {'cut_center': np.empty((0, n)),
            'cut_gradient': np.empty((0, n)), 'cut_g': np.empty(0),
            'cut_padding': np.empty(0), 'cut_rhs': np.empty(0),
            'cut_metric': np.empty(0, dtype='<U1'),
            'cut_context': np.empty(0, dtype=np.int64),
            'cut_iteration': np.empty(0, dtype=np.int64)}
    return {'cut_center': np.stack([x['center'] for x in cuts]),
        'cut_gradient': np.stack([x['gradient'] for x in cuts]),
        'cut_g': np.array([x['g'] for x in cuts]),
        'cut_padding': np.array([x['padding'] for x in cuts]),
        'cut_rhs': np.array([x['rhs'] for x in cuts]),
        'cut_metric': np.array([x['metric'] for x in cuts]),
        'cut_context': np.array([x['context'] for x in cuts]),
        'cut_iteration': np.array([x['iteration'] for x in cuts])}


def solve_recorded_lp(problem, stem, out, ledger, persist, deadlines,
                      solver=None):
    """Durable LP input/start then returned/materialized/raw before assertions."""
    solver = linprog if solver is None else solver
    n = len(problem['c'])-1
    input_raw = {'c': problem['c'], 'A_ub': problem['A_ub'],
        'b_ub': problem['b_ub'], 'lower': problem['lower'],
        'upper': problem['upper'],
        'row_labels': np.asarray(problem['row_labels']),
        'origin_delta': problem['base']['delta'],
        'origin_losses': problem['base']['losses'],
        'origin_loss_gradient': problem['base']['loss_gradient'],
        'origin_q_error': problem['base']['q_error'],
        'origin_tendon_error': problem['base']['tendon_error'],
        'origin_npz_sha256': np.asarray(problem['base']['source_sha256'])}
    input_raw.update(cut_arrays(problem['cuts'], n))
    result = None
    raw = {}
    seen = []
    caught = []
    try:
        deadlines()
        np.savez_compressed(out/f'{stem}-input.npz', **input_raw)
        ledger['lp_input_raw_saved'] += 1
        ledger['active_lp'] = stem
        persist()
        if ledger['lp_started'] >= MAX_TOTAL_LP:
            raise TimeoutError('E120 maximum54 original LP calls')
        ledger['lp_started'] += 1
        try:
            persist()
        except BaseException:
            ledger['lp_started'] -= 1
            ledger['start_receipt_failed'] = stem
            raise
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = solver(problem['c'], A_ub=problem['A_ub'],
                b_ub=problem['b_ub'],
                bounds=list(zip(problem['lower'], problem['upper'])),
                method='highs-ds', options=LP_OPTIONS.copy())
        ledger['lp_returned'] += 1
        seen = warning_rows(caught)
        persist()
        returned_arrays(result, raw)
        ledger['lp_materialized'] += 1
        persist()
        np.savez_compressed(out/f'{stem}-returned.npz', **raw)
        ledger['lp_raw_saved'] += 1
        save_json(out/f'{stem}-returned.json', dict(
            result_metadata(result, seen),
            raw_npz_saved_before_assertions=True))
        persist()
        deadlines()
        diagnostic = validate_solution(problem, result, raw, seen)
        save_json(out/f'{stem}-diagnostic.json', diagnostic)
        ledger['active_lp'] = None
        persist()
        return {'raw': raw, 'metadata': result_metadata(result, seen),
                'diagnostic': diagnostic}
    except BaseException as primary:
        if not seen and caught:
            try:
                seen = warning_rows(caught)
            except Exception as secondary:  # noqa: BLE001
                primary.add_note(f'E120 secondary warnings rescue: {secondary}')
        if result is not None:
            try:
                returned_arrays(result, raw, primary)
            except Exception as secondary:  # noqa: BLE001
                primary.add_note(f'E120 secondary returned rescue: {secondary}')
        destination = out/f'error-raw-{stem}'
        try:
            rescue_arrays(destination/'input', input_raw, primary, 'LP input')
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E120 secondary LP input rescue: {secondary}')
        if raw:
            try:
                rescue_arrays(destination/'returned', raw, primary,
                              'LP returned')
            except Exception as secondary:  # noqa: BLE001
                primary.add_note(f'E120 secondary LP returned rescue: {secondary}')
        try:
            save_json(destination/'error-metadata.json', dict(
                result_metadata(result, seen),
                primary_type=type(primary).__name__,
                primary_message=str(primary)))
        except Exception as secondary:  # noqa: BLE001
            primary.add_note(f'E120 secondary LP metadata rescue: {secondary}')
        raise


def analyze_leg(base, out, ledger, persist, deadlines, solver=None):
    """One shape LP and at most eight ordered startup outer-approximation LPs."""
    leg = base['leg']
    shape_problem = build_lp(base, 'shape_box')
    shape_returned = solve_recorded_lp(shape_problem, f'lp-{leg}-shape_box',
                                       out, ledger, persist, deadlines, solver)
    shape_repair = ray_repair(base, shape_returned['raw']['x'][:-1],
                              'shape_box')
    shape_upper = min(base['origin_objective'],
                      shape_repair['repaired_objective']['objective'])
    shape_lower = shape_returned['diagnostic']['certificate']['lower_bound']
    shape_gap = shape_upper-shape_lower
    require(shape_gap >= -1e-6*(1+abs(base['origin_objective'])),
            f'E120 inconsistent shape local bounds: {leg}')
    shape = {'lower_bound': shape_lower, 'upper_bound': shape_upper,
        'gap': shape_gap, 'lp_repair': shape_repair,
        'certificate': shape_returned['diagnostic']['certificate']}
    save_json(out/f'shape-{leg}.json', shape)
    cuts = [rms_cut(base, metric, context, np.zeros(len(base['delta'])), 0)
            for metric, context in RMS_ORDER]
    best_upper = base['origin_objective']
    best_h = np.zeros(len(base['delta']), dtype=np.float64)
    lower = -math.inf
    iterations = []
    stop_reason = None
    for iteration in range(1, MAX_STARTUP_LP+1):
        deadlines()
        problem = build_lp(base, 'startup_box', cuts)
        returned = solve_recorded_lp(problem,
            f'lp-{leg}-startup_box-{iteration:02d}', out, ledger,
            persist, deadlines, solver)
        raw_h = returned['raw']['x'][:-1]
        repair = ray_repair(base, raw_h, 'startup_box')
        candidate = repair['repaired_objective']['objective']
        if candidate < best_upper:
            best_upper, best_h = candidate, repair['repaired_h'].copy()
        lower = max(lower, returned['diagnostic']['certificate']['lower_bound'])
        gap = best_upper-lower
        tolerance = 1e-6*(1+abs(base['origin_objective']))
        rms_at_raw = [rms_state(base, metric, context, raw_h)[0]
                      for metric, context in RMS_ORDER]
        record = {'iteration': iteration, 'cut_count': len(cuts),
            'raw_x': returned['raw']['x'], 'repair': repair,
            'RMS_at_raw': rms_at_raw,
            'lower_bound': lower, 'upper_bound': best_upper,
            'best_feasible_h': best_h.copy(), 'gap': gap,
            'gap_tolerance': tolerance,
            'certificate': returned['diagnostic']['certificate']}
        iterations.append(record)
        save_json(out/f'iteration-{leg}-{iteration:02d}.json', record)
        require(gap >= -tolerance,
                f'E120 inconsistent startup local bounds: {leg}')
        if gap <= tolerance:
            stop_reason = 'gap_closed'
            break
        violated = [(metric, context) for (metric, context), g in
                    zip(RMS_ORDER, rms_at_raw) if g > CAP]
        if not violated:
            stop_reason = 'numerical_gap_unresolved'
            break
        if iteration == MAX_STARTUP_LP:
            stop_reason = 'cut_limit_unresolved'
            break
        for metric, context in violated:
            cuts.append(rms_cut(base, metric, context, raw_h, iteration))
        require(len(cuts) <= MAX_CUTS, 'E120 RMS cut limit exceeded')
        save_json(out/f'cut-history-{leg}.json', cuts)
    reduction = base['origin_objective']-best_upper
    limit = 1e-4*(1+abs(base['origin_objective']))
    if reduction >= limit:
        classification = 'feasible_descent_demonstrated'
    elif base['origin_objective']-lower <= limit and \
            best_upper-lower <= 1e-6*(1+abs(base['origin_objective'])):
        classification = 'limited_local_descent'
    else:
        classification = 'unresolved'
    report = {'leg': leg, 'origin_objective': base['origin_objective'],
        'shape_box': shape, 'startup_box': {
            'lower_bound': lower, 'upper_bound': best_upper,
            'gap': best_upper-lower, 'best_feasible_h': best_h,
            'iterations': iterations, 'stop_reason': stop_reason,
            'classification': classification,
            'descent_threshold': limit},
        'shape_reduction_interval': [base['origin_objective']-shape_upper,
                                      base['origin_objective']-shape_lower],
        'startup_reduction_interval': [reduction,
                                        base['origin_objective']-lower],
        'startup_constraint_cost_interval': [lower-shape_upper,
                                              best_upper-shape_lower],
        'diagnostic_only': True, 'candidate_adopted': False,
        'native_gait_tested': False, 'F6_exposed': False}
    save_json(out/f'comparison-{leg}.json', report)
    return report


def run_once(root, out):
    """One later-authorized original-gradient LP process; no model execution."""
    root, out = Path(root), Path(out)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'experiment': 'E120-local-feasibility', 'stage': 'created',
        'prior_conservative_process_s': PRIOR_S,
        'shared_process_limit_s': SHARED_LIMIT_S,
        'process_wall_s': None, 'process_wall_status': 'not_yet_read',
        'lp_input_raw_saved': 0, 'lp_started': 0, 'lp_returned': 0,
        'lp_materialized': 0, 'lp_raw_saved': 0, 'active_lp': None,
        'conditional_responses': 0, 'derivative_bank_calls': 0,
        'fit_calls': 0, 'native_network_calls': 0, 'body_calls': 0,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'calibration_outputs': 0, 'candidate_adopted': False,
        'F6_exposed': False, 'secondary_errors': []}

    def refresh_clock(primary=None):
        try:
            elapsed = time.monotonic()-started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Unverifiable E120 monotonic wall time')
            ledger['process_wall_s'] = elapsed
            ledger['process_wall_status'] = 'available'
        except Exception as secondary:
            ledger['process_wall_s'] = None
            ledger['process_wall_status'] = 'unavailable'
            if primary is None:
                raise
            ledger['secondary_errors'].append({'at': 'clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E120 secondary wall clock: {secondary}')

    def persist():
        refresh_clock()
        save_json(out/'execution-ledger.json', ledger)

    def deadlines():
        elapsed = time.monotonic()-started
        if elapsed > PROCESS_LIMIT_S or PRIOR_S+elapsed > SHARED_LIMIT_S:
            raise TimeoutError('E120 internal/shared process wall limit')

    def failure_save(path, value, primary):
        try:
            save_json(out/path, value)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': path,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E120 secondary {path} save: {secondary}')

    primary = None
    try:
        for original, name in (
            (ROOT/'tools/analyze_c_banc_m3_local_feasibility.py', 'runner.py'),
            (ROOT/'tests/test_c_banc_local_feasibility.py',
             'test_c_banc_local_feasibility.py'),
            (ROOT/'docs/BANC_M3_E120_LOCAL_FEASIBILITY_DESIGN_20260920.md',
             'frozen-design.md'),
            (root/'e120-frozen-design.json', 'e120-frozen-design.json')):
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
            'source_sha256': data['freeze']['source_sha256'],
            'manifest_sha256': data['manifest_sha256'],
            'runner_sha256': sha256(ROOT/
                'tools/analyze_c_banc_m3_local_feasibility.py'),
            'test_sha256': sha256(ROOT/
                'tests/test_c_banc_local_feasibility.py')})
        deadlines()
        ledger['stage'] = 'lp'
        persist()
        reports = {}
        for leg in LEGS:
            deadlines()
            base = base_problem(data['records'][leg])
            save_json(out/f'origin-{leg}.json', {
                'leg': leg, 'selected_index': INDICES[leg],
                'source_npz_sha256': base['source_sha256'],
                'origin_objective': base['origin_objective'],
                'h_lower': base['h_lower'], 'h_upper': base['h_upper'],
                't_upper': base['upper'][-1],
                'strict_origin_feasible': True})
            reports[leg] = analyze_leg(base, out, ledger, persist, deadlines)
        require(12 <= ledger['lp_started'] == ledger['lp_returned'] ==
                ledger['lp_materialized'] == ledger['lp_raw_saved'] <=
                MAX_TOTAL_LP,
                'E120 LP accounting incomplete or over original cap')
        deadlines()
        ledger['stage'] = 'complete'
        persist()
        result = {'status': 'completed_local_feasibility_diagnostic',
            'diagnostic_only': True, 'candidate_adopted': False,
            'native_gait_tested': False, 'F6_exposed': False,
            'no_new_response_or_native_run': True,
            'leg_reports': reports, 'ledger': ledger}
        save_json(out/'result.json', result)
        deadlines()
        persist()
        deadlines()
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
        'verification/banc-natural-gait-m3-20260919/e120-local-feasibility')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
