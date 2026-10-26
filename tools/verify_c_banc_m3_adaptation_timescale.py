"""E134 frozen single-process CUDA verifier. Run only after parent source review."""

# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import banc_m3_joint_adaptive_recruitment as old
from tools import banc_m3_precomputed_expm1 as e130
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools.banc_m3_adaptation_timescale import (
    FIELDS,
    CudaAdaptationTimescaleEngine,
    activation_parameter_map,
)

FREEZE_SHA = '87147051760eec2ec09add779c6e7ce15183e765b31cbfa00ba640f4484a7ea6'
SOURCE_FILES = ('tools/banc_m3_adaptation_timescale.py',
                'tools/verify_c_banc_m3_adaptation_timescale.py',
                'tests/test_c_banc_adaptation_timescale.py')
ORDER = ('baseline', 'zero_tau', 'center',
    'tau_plus_1e-5', 'tau_minus_1e-5',
    'tau_plus_1e-6', 'tau_minus_1e-6',
    'tau_plus_1e-7', 'tau_minus_1e-7',
    'tau_plus_1e-8', 'tau_minus_1e-8',
    'gain_plus', 'gain_minus', 'edge_plus', 'edge_minus')
BASE = ROOT/'verification/banc-natural-gait-m3-20260919'


def sha256(path):
    return fit.sha256(path)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2,
        ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def checked_relative(relative, digest):
    return fit.checked_relative(relative, digest)


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E134 frozen SHA required')
    frozen = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if (frozen.get('schema') != 'flylab.m3.e134.adaptation-timescale-freeze.v1'
            or frozen.get('design_sha256') != sha256(ROOT/frozen['design_path'])
            or frozen.get('original_batch_order') != list(ORDER)
            or frozen.get('derivative_batch_indices') != [1, 2, 3]
            or frozen.get('log_tau_bounds') != [-math.log(3), math.log(10)]
            or frozen.get('h_values') != [1e-5, 1e-6, 1e-7, 1e-8]
            or frozen.get('final_h') != 1e-8
            or (frozen.get('sensitivity_atol'), frozen.get('sensitivity_rtol'),
                frozen.get('parity_atol'), frozen.get('numpy_reference_atol')) !=
                (2e-5, 2e-4, 1e-8, 1e-8)
            or frozen.get('dimensions') != {
                'contexts': 3, 'neurons': 391, 'muscles': 96, 'sources': 69,
                'union_edges': 2444, 'fit_edges': 1572,
                'old_parameters': 1668, 'parameters': 1764,
                'stages': 160000, 'ticks': 40000, 'samples': 2001}
            or (frozen.get('internal_wall_limit_s'),
                frozen.get('external_wall_limit_s'),
                frozen.get('supervisor_limit_s')) != (120, 145, 150)
            or frozen.get('original_processes') != 1
            or frozen.get('retries') != 0
            or frozen.get('tiny') != {'batches': 2, 'context_ticks': 40,
                'conditional_s': .01, 'derivative_banks': 2}
            or frozen.get('full_BANC_calls') != 0
            or frozen.get('body_calls') != 0
            or frozen.get('fit_calls') != 0):
        raise ValueError('E134 frozen numeric/resource contract changed')
    for mapping, count in (('input_sha256', 15), ('reused_source_sha256', 4)):
        if len(frozen.get(mapping, {})) != count:
            raise ValueError('E134 frozen source cardinality changed: '+mapping)
        for relative, digest in frozen[mapping].items():
            checked_relative(relative, digest)
    if sha256(review_path) != review_sha:
        raise ValueError('Exact E134 parent source review SHA required')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_adaptation_timescale_verification') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != FREEZE_SHA
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted defect-free E134 three-source review required')
    for relative, digest in review['source_sha256'].items():
        checked_relative(relative, digest)
    e128 = json.loads((ROOT/frozen['e128_freeze_path']).read_text(encoding='utf-8'))
    if (e128.get('schema') != 'flylab.m3.e128.joint-adaptive-fit-freeze.v1'
            or e128.get('fixture_path') != 'verification/banc-natural-gait-m3-20260919/e128-inputs.npz'
            or [x['name'] for x in e128['contexts']] != ['E110', 'E113', 'E121']):
        raise ValueError('Original E128 context freeze changed')
    for item in e128['contexts']:
        for key in ('current', 'source'):
            relative = item[key]
            if relative not in frozen['input_sha256']:
                raise ValueError('E134 missing frozen stage input: '+relative)
    return frozen, e128, review


def parameters(path, fixture):
    with np.load(path, allow_pickle=False) as data:
        p = {key: data[key].copy() for key in data.files}
    expected = {'base_delta': (1572,), 'base_gain': (96,),
        'zero_log_tau': (96,), 'center_delta': (1572,),
        'center_gain': (96,), 'center_log_tau': (96,),
        'edge_direction': (1572,), 'edge_ids': (1572,),
        'muscle_indices': (96,), 'motor_ids': (391,),
        'lf_edge_indices': (273,), 'lf_muscle_indices': (17,)}
    if (set(p) != set(expected) or any(p[k].shape != shape for k, shape in expected.items())
            or any(p[k].dtype != np.float64 for k in expected if k not in
                   ('edge_ids', 'muscle_indices', 'motor_ids',
                    'lf_edge_indices', 'lf_muscle_indices'))
            or not all(np.isfinite(p[k]).all() for k in expected if p[k].dtype.kind == 'f')):
        raise ValueError('Exact finite E134 parameter shape/dtype required')
    for key, original in (('edge_ids', fixture['edge_ids']),
                          ('motor_ids', fixture['motor_ids']),
                          ('muscle_indices', np.arange(96))):
        if not np.array_equal(p[key], original):
            raise ValueError('E134 fixed parameter ID order changed: '+key)
    j = np.arange(96)
    i = np.arange(1572)
    for key, expected_value in (
        ('zero_log_tau', np.zeros(96)),
        ('center_delta', p['base_delta']),
        ('center_gain', 1+.25*np.sin(j)),
        ('center_log_tau', .25*np.sin(j+.5)),
        ('edge_direction', .5+.5*np.cos(.31*i))):
        if not np.array_equal(p[key], expected_value):
            raise ValueError('E134 immutable parameter formula changed: '+key)
    return p


def batch_vector(name, p):
    delta = p['base_delta'].copy()
    gain = p['base_gain'].copy()
    log_tau = p['zero_log_tau'].copy()
    if name not in ('baseline', 'zero_tau'):
        delta = p['center_delta'].copy()
        gain = p['center_gain'].copy()
        log_tau = p['center_log_tau'].copy()
    if name.startswith('tau_'):
        pieces = name.split('_')
        log_tau += (1 if pieces[1] == 'plus' else -1)*float(pieces[2])
    elif name.startswith('gain_'):
        gain += (1 if name.endswith('plus') else -1)*1e-8
    elif name.startswith('edge_'):
        delta += (1 if name.endswith('plus') else -1)*1e-8*p['edge_direction']
    return delta, gain, log_tau


def _raw_name(index, name):
    return f'batch-{index:03d}-{name}-raw.npz'


def _partial(out, prefix, error, raw=None):
    partial = dict(getattr(error, 'partial_response', {}) or {})
    if raw:
        for key, value in raw.items():
            partial.setdefault(key, value)
    paths = []
    if partial:
        path = out/(prefix+'-partial-raw.npz')
        try:
            np.savez(path, **partial)
            paths.append({'path': str(path), 'sha256': sha256(path),
                          'keys': sorted(partial)})
        except BaseException as secondary:
            error.add_note('Secondary partial raw save: '+repr(secondary))
            if path.exists():
                try:
                    paths.append({'path': str(path), 'sha256': sha256(path),
                                  'status': 'possibly_truncated'})
                except BaseException as digest_error:
                    error.add_note('Secondary partial raw digest: '+repr(digest_error))
    return paths


def _execute(engine, delta, gain, log_tau, derivative, path, row, ledger, out,
             *, baseline=False):
    """Persist requested work, complete raw, and partial failures in that order."""
    row.update(status='requested', requested_context_ticks=engine.contexts*engine.ticks,
        requested_conditional_s=engine.contexts*engine.ticks*.00025,
        requested_scalar_MN_ticks=engine.contexts*engine.ticks*engine.neurons,
        requested_decoder_endpoints=engine.contexts*(engine.samples-1),
        requested_derivative_banks=int(derivative),
        requested_edge_stage_ticks=(engine.contexts*engine.ticks*4*engine.edges
            if derivative else 0), solve_requested=False, solve_returned=False,
        post_return_error=False)
    began = time.perf_counter()
    device = None
    raw = {}
    adapter = None
    try:
        save(out/'advance-ledger.json', ledger)
        if baseline:
            with e130.precomputed_expm1(engine) as adapter:
                row['solve_requested'] = True
                device, raw, timing = engine.solve(delta, gain, derivative=derivative)
                row['solve_returned'] = True
                row['progress'] = timing['progress']
        else:
            row['solve_requested'] = True
            device, raw, timing = engine.solve(delta, gain, log_tau,
                                               derivative=derivative)
            row['solve_returned'] = True
            row['progress'] = timing['progress']
        if derivative:
            raw['B'] = engine.host_copy(device['B'],
                progress=timing['progress'], label='full_B')
        raw.update(delta=delta, gain=gain, log_tau=log_tau,
            edge_ids=engine.problem['edge_ids'],
            motor_indices=engine.problem['motor_indices'],
            muscle_indices=engine.problem['muscle_indices'],
            parameter_muscle_map=(old.activation_parameter_map(engine.problem)
                if baseline else activation_parameter_map(engine.problem)))
        np.savez(path, **raw)  # intentionally uncompressed
        row.update(status='raw_saved', raw_path=str(path), raw_sha256=sha256(path),
                   response_fields=list(old.FIELDS if baseline else FIELDS),
                   branch_saved=True, B_saved=bool(derivative))
        if adapter is not None:
            row['baseline_adapter'] = adapter.snapshot()
        save(out/'advance-ledger.json', ledger)
    except BaseException as error:
        response_progress = getattr(error, 'response_progress', None)
        row.update(status='failed', error_type=type(error).__name__,
            error=repr(error), post_return_error=bool(row['solve_returned']),
            primary_notes=list(getattr(error, '__notes__', [])),
            partial_response_keys=sorted(getattr(error, 'partial_response', {}) or {}),
            secondary_copy_errors=getattr(error, 'secondary_copy_errors', None))
        if response_progress is not None:
            row['progress'] = response_progress
        if adapter is not None:
            try:
                row['baseline_adapter'] = adapter.snapshot()
            except BaseException as secondary:
                row['secondary_adapter_error'] = repr(secondary)
                error.add_note('Secondary baseline adapter snapshot: '+repr(secondary))
        if device is not None:
            try:
                rescued, secondary = engine.rescue_device_state(device)
                for key, value in rescued.items():
                    raw.setdefault(key, value)
                row['secondary_rescue_errors'] = secondary
            except BaseException as secondary:
                error.add_note('Secondary verifier rescue: '+repr(secondary))
        try:
            row['partial_raw_files'] = _partial(out, path.stem.replace('-raw', ''),
                                                error, raw)
        except BaseException as secondary:
            row['secondary_partial_error'] = repr(secondary)
            error.add_note('Secondary partial raw preservation: '+repr(secondary))
        if path.exists() and 'raw_sha256' not in row:
            try:
                row['raw_file_after_error'] = {'path': str(path),
                    'sha256': sha256(path), 'status': 'unknown_or_partial'}
            except BaseException as secondary:
                row['secondary_raw_digest_error'] = repr(secondary)
                error.add_note('Secondary raw digest: '+repr(secondary))
        row['primary_notes'] = list(getattr(error, '__notes__', []))
        raise
    finally:
        active = sys.exc_info()[1]
        final_error = active
        row['wall_s'] = time.perf_counter()-began
        try:
            row['engine_metadata'] = engine.metadata()
        except BaseException as secondary:
            row['secondary_metadata_error'] = repr(secondary)
            if final_error is not None:
                final_error.add_note('Secondary engine metadata: '+repr(secondary))
            else:
                final_error = secondary
                row.update(status='failed', error_type=type(secondary).__name__,
                    error=repr(secondary),
                    post_return_error=bool(row['solve_returned']))
        try:
            save(out/'advance-ledger.json', ledger)
        except BaseException as secondary:
            row['secondary_final_ledger_error'] = repr(secondary)
            if final_error is not None:
                final_error.add_note('Secondary ledger save: '+repr(secondary))
            else:
                final_error = secondary
                row.update(status='failed', error_type=type(secondary).__name__,
                    error=repr(secondary),
                    post_return_error=bool(row['solve_returned']))
        row['wall_s'] = time.perf_counter()-began
        if active is None and final_error is not None:
            raise final_error


def _max_comparison(analytic, finite_difference, limit_atol, limit_rtol):
    if analytic.shape != finite_difference.shape:
        return {'status': 'shape_mismatch', 'accepted': False}
    finite = bool(np.isfinite(analytic).all() and np.isfinite(finite_difference).all())
    diff = np.abs(analytic-finite_difference)
    where = np.unravel_index(int(np.argmax(diff)), diff.shape)
    maximum = float(diff[where])
    fd_max = float(np.max(np.abs(finite_difference)))
    bound = limit_atol+limit_rtol*fd_max
    return {'max_abs_error': maximum, 'argmax': list(map(int, where)),
        'max_abs_FD': fd_max, 'bound': bound, 'finite': finite,
        'accepted': bool(finite and maximum <= bound)}


def _branch_change_counts(reference, candidate):
    """Count changed neuron-ticks and individual RK4 stage bits by context."""
    if reference.shape != candidate.shape or reference.ndim != 3:
        return {'shape_mismatch': True, 'count': None,
                'first_flat_index': None, 'per_context': []}
    per_context = []
    total = 0
    first = None
    stride = reference.shape[1]*reference.shape[2]
    for context in range(reference.shape[0]):
        changed_bits = np.bitwise_xor(reference[context], candidate[context])
        changed_cells = changed_bits != 0
        count = int(np.count_nonzero(changed_cells))
        if count and first is None:
            first = context*stride+int(np.argmax(changed_cells))
        total += count
        per_context.append({'context_index': context,
            'changed_ticks': int(np.count_nonzero(np.any(changed_cells, axis=1))),
            'changed_neuron_ticks': count,
            'changed_stage_bits': [int(np.count_nonzero(changed_bits & (1 << stage)))
                                   for stage in range(4)]})
    return {'count': total, 'first_flat_index': first,
            'per_context': per_context}


def _directional_outputs(center, kind, direction, mapping, moment, ltm):
    if kind == 'tau':
        rate = center['rate_log_tau_sensitivity']
        adaptation = center['adaptation_log_tau_sensitivity']
        activation = center['activation_log_tau_sensitivity']
    elif kind == 'gain':
        rate = center['rate_sensitivity']
        adaptation = center['adaptation_sensitivity']
        activation = center['activation_sensitivity']
    else:
        B = center['B'][:, :, :len(direction)]
        activation = np.zeros_like(center['activation'])
        for i, muscle in enumerate(mapping):
            activation[:, :, muscle] += B[:, :, i]*direction[i]
        rate = adaptation = None
    command = .02*activation@moment.T
    a = center['activation'][:, :, ltm]
    tendon = .02*(1-np.tanh(a/10)**2)*activation[:, :, ltm]
    return {'rate': rate, 'adaptation': adaptation, 'activation': activation,
            'command': command, 'tendon': tendon}


def _compare_all(out, frozen, fixture, ledger):
    """Called only after all 15 complete raw files exist."""
    paths = {name: out/_raw_name(i, name) for i, name in enumerate(ORDER, 1)}
    if any(not path.exists() for path in paths.values()):
        raise ValueError('All 15 complete original raw responses required before comparison')
    results = {'parity': {}, 'finite_layout': {}, 'finite_difference': {},
               'branch_changes': {}, 'accepted': False}
    center = np.load(paths['center'], allow_pickle=False)
    baseline = np.load(paths['baseline'], allow_pickle=False)
    zero = np.load(paths['zero_tau'], allow_pickle=False)
    original_fields = (*old.FIELDS, 'branch_bits')
    required_new = (*FIELDS, 'branch_bits')
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as raw:
            keys = (*original_fields, 'B') if name == 'baseline' else (
                (*required_new, 'B') if name in ('zero_tau', 'center') else required_new)
            n, m, e, c, samples, ticks = 391, 96, 1572, 3, 2001, 40000
            shapes = {'rate': (c, samples, n), 'adaptation': (c, samples, n),
                'rate_sensitivity': (c, samples, n),
                'adaptation_sensitivity': (c, samples, n),
                'rate_log_tau_sensitivity': (c, samples, n),
                'adaptation_log_tau_sensitivity': (c, samples, n),
                'activation': (c, samples, m),
                'activation_sensitivity': (c, samples, m),
                'activation_log_tau_sensitivity': (c, samples, m),
                'command': (c, samples, 42), 'tendon': (c, samples, 6),
                'branch_bits': (c, ticks, n),
                'B': (c, samples, e+m if name == 'baseline' else e+2*m)}
            good = (all(k in raw and raw[k].shape == shapes[k] and
                        raw[k].dtype == (np.uint8 if k == 'branch_bits' else np.float64)
                        for k in keys)
                    and all(np.isfinite(raw[k]).all() for k in keys if k != 'branch_bits')
                    and np.array_equal(raw['edge_ids'], fixture['edge_ids'])
                    and np.array_equal(raw['motor_indices'], np.arange(n))
                    and np.array_equal(raw['muscle_indices'], np.arange(m)))
            expected_map = np.r_[fixture['membership'][fixture['edge_rows']],
                np.arange(m)]
            if name != 'baseline':
                expected_map = np.r_[expected_map, np.arange(m)]
            good = bool(good and np.array_equal(raw['parameter_muscle_map'], expected_map))
            results['finite_layout'][name] = {'accepted': good,
                'raw_sha256': sha256(path), 'field_count': len(keys)}
            if name not in ('baseline', 'zero_tau', 'center'):
                results['branch_changes'][name] = _branch_change_counts(
                    center['branch_bits'], raw['branch_bits'])
    for field in old.FIELDS:
        diff = float(np.max(np.abs(zero[field]-baseline[field])))
        results['parity'][field] = {'max_abs': diff,
            'accepted': bool(np.isfinite(diff) and diff <= frozen['parity_atol'])}
    same = np.array_equal(zero['branch_bits'], baseline['branch_bits'])
    results['parity']['branch_bits'] = {'exact': bool(same), 'accepted': bool(same)}
    B_diff = float(np.max(np.abs(zero['B'][:, :, :1668]-baseline['B'])))
    results['parity']['B_first_1668'] = {'max_abs': B_diff,
        'accepted': bool(np.isfinite(B_diff) and B_diff <= frozen['parity_atol'])}
    mapping = fixture['membership'][fixture['edge_rows']]
    for kind, h_values in (('tau', frozen['h_values']), ('gain', [1e-8]),
                           ('edge', [1e-8])):
        direction = None
        if kind == 'edge':
            with np.load(BASE/'e134-parameters.npz', allow_pickle=False) as p:
                direction = p['edge_direction'].copy()
        analytic = _directional_outputs(center, kind, direction, mapping,
            fixture['moment'], fixture['ltm'])
        for h in h_values:
            plus_name = (f'tau_plus_{h}' if kind == 'tau' else kind+'_plus')
            minus_name = (f'tau_minus_{h}' if kind == 'tau' else kind+'_minus')
            # Frozen names use scientific notation without the Python 1e-05 spelling.
            if kind == 'tau':
                suffix = {1e-5: '1e-5', 1e-6: '1e-6',
                          1e-7: '1e-7', 1e-8: '1e-8'}[h]
                plus_name, minus_name = 'tau_plus_'+suffix, 'tau_minus_'+suffix
            with (np.load(paths[plus_name], allow_pickle=False) as plus,
                  np.load(paths[minus_name], allow_pickle=False) as minus):
                fields = ('rate', 'adaptation', 'activation', 'command', 'tendon')
                if kind == 'edge':
                    fields = ('activation', 'command', 'tendon')
                row = {}
                for field in fields:
                    fd = (plus[field]-minus[field])/(2*h)
                    row[field] = _max_comparison(analytic[field], fd,
                        frozen['sensitivity_atol'], frozen['sensitivity_rtol'])
                row['branch_changes'] = {
                    'plus_vs_center': results['branch_changes'][plus_name],
                    'minus_vs_center': results['branch_changes'][minus_name],
                    'plus_vs_minus': _branch_change_counts(
                        plus['branch_bits'], minus['branch_bits'])}
                row['accepted'] = all(row[field]['accepted'] for field in fields)
                results['finite_difference'][kind+'_'+str(h)] = row
    final_keys = ('tau_1e-08', 'gain_1e-08', 'edge_1e-08')
    results['accepted'] = bool(
        all(v['accepted'] for v in results['finite_layout'].values())
        and all(v['accepted'] for v in results['parity'].values())
        and all(results['finite_difference'][k]['accepted'] for k in final_keys))
    save(out/'numeric-comparison.json', results)
    return results


def _seal_actual_counters(ledger):
    """Count dispatched and returned solves from rows, including failed saves."""
    for kind in ('tiny', 'original'):
        rows = ledger[kind+'_batches']
        ledger['actual_'+kind+'_solve_requested'] = sum(
            bool(row.get('solve_requested')) for row in rows)
        ledger['actual_'+kind+'_solve_returned'] = sum(
            bool(row.get('solve_returned')) for row in rows)
        ledger['actual_'+kind+'_solve_unknown'] = sum(
            bool(row.get('solve_requested')) and not row.get('solve_returned')
            for row in rows)
        ledger['actual_'+kind+'_post_return_errors'] = sum(
            bool(row.get('post_return_error')) for row in rows)
        ledger[kind+'_batches_raw_saved'] = sum(
            'raw_sha256' in row for row in rows)


def run(freeze_path, freeze_sha, source_review, source_review_sha, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve existing E134 original output: '+str(out))
    out.mkdir(parents=True)
    began = time.perf_counter()
    ledger = {'schema': 'flylab.m3.e134.adaptation-timescale-ledger.v1',
        'status': 'preflight_requested', 'original_batches': [],
        'original_batches_requested': 0, 'original_batches_raw_saved': 0,
        'tiny_batches': [], 'tiny_batches_requested': 0,
        'tiny_batches_raw_saved': 0, 'retries': 0,
        'actual_full_BANC_calls': 0, 'actual_body_calls': 0,
        'actual_geometry_calls': 0, 'actual_fit_calls': 0,
        'internal_wall_s': None, 'secondary_errors': []}
    save(out/'advance-ledger.json', ledger)
    frozen = None
    limits = None
    engines = {}
    primary = None
    result = None
    try:
        frozen, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, source_review, source_review_sha)
        save(out/'source-review-snapshot.json', review)
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        fixture = fit.frozen_problem(e128)
        param = parameters(ROOT/frozen['parameter_path'], fixture)
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Actual CUDA GPU required')
        ledger['environment'] = {'python': sys.version, 'cupy': cp.__version__,
            'gpu': str(cp.cuda.runtime.getDeviceProperties(0)['name'])}
        tiny = fit.synthetic_tiny_setup_problem()
        for label in ('baseline', 'candidate'):
            row = {'index': len(ledger['tiny_batches'])+1,
                   'backend': label, 'synthetic_only': True,
                   'solve_requested': False, 'solve_returned': False,
                   'post_return_error': False}
            ledger['tiny_batches'].append(row)
            ledger['tiny_batches_requested'] += 1
            save(out/'advance-ledger.json', ledger)
            engine = (old.CudaJointAdaptiveEngine(tiny) if label == 'baseline'
                      else CudaAdaptationTimescaleEngine(tiny))
            path = out/f'tiny-{label}-raw.npz'
            _execute(engine, np.zeros(1), np.zeros(1), np.zeros(1), True,
                path, row, ledger, out, baseline=label == 'baseline')
            ledger['tiny_batches_raw_saved'] += 1
            save(out/'advance-ledger.json', ledger)
        for label, constructor in (('baseline', old.CudaJointAdaptiveEngine),
                                   ('candidate', CudaAdaptationTimescaleEngine)):
            engines[label] = constructor(fixture)
            ledger[label+'_engine_fixed_upload'] = engines[label].metadata()
            save(out/'advance-ledger.json', ledger)
        ledger['status'] = 'ready'
        save(out/'advance-ledger.json', ledger)
        if time.perf_counter()-began >= frozen['internal_wall_limit_s']-5:
            raise TimeoutError('E134 preflight exhausted original time reserve')
        for index, name in enumerate(ORDER, 1):
            if time.perf_counter()-began >= frozen['internal_wall_limit_s']-1:
                raise TimeoutError('E134 time reserve before next original batch')
            delta, gain, log_tau = batch_vector(name, param)
            engine = engines['baseline' if name == 'baseline' else 'candidate']
            row = {'index': index, 'name': name,
                   'backend': 'baseline' if name == 'baseline' else 'candidate',
                   'solve_requested': False, 'solve_returned': False,
                   'post_return_error': False}
            ledger['original_batches'].append(row)
            ledger['original_batches_requested'] += 1
            path = out/_raw_name(index, name)
            _execute(engine, delta, gain, log_tau, index <= 3,
                     path, row, ledger, out, baseline=name == 'baseline')
            ledger['original_batches_raw_saved'] += 1
            save(out/'advance-ledger.json', ledger)
        ledger['status'] = 'all_original_raw_saved'
        save(out/'advance-ledger.json', ledger)
        comparison = _compare_all(out, frozen, fixture, ledger)
        result = {'experiment': 'M3-E134-adaptation-timescale',
            'validity': 'complete_original_cuda_raw',
            'hypothesis_verdict': 'pending_parent_numpy_reference' if comparison['accepted']
                else 'numeric_rejected',
            'cuda_numeric_screen_supported': comparison['accepted'],
            'calculator_supported': False,
            'original_batches': len(ORDER), 'tiny_batches': 2,
            'original_context_responses': 45,
            'original_conditional_s': 450.,
            'actual_full_BANC_calls': 0, 'actual_body_calls': 0,
            'actual_geometry_calls': 0, 'actual_fit_calls': 0,
            'parent_numpy_reference': 'pending_independent_parent_review',
            'conditional_only': True}
        ledger['status'] = 'numeric_compared'
    except BaseException as error:
        primary = error
        ledger['status'] = 'technical_failed'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'response_progress': getattr(error, 'response_progress', None),
            'fixed_upload_progress': getattr(error, 'fixed_upload_progress', None),
            'fixed_upload_by_key': getattr(error, 'fixed_upload_by_key', None),
            'partial_response_keys': sorted(getattr(error, 'partial_response', {}) or {}),
            'secondary_copy_errors': getattr(error, 'secondary_copy_errors', None),
            'notes': list(getattr(error, '__notes__', []))}
    finally:
        _seal_actual_counters(ledger)
        for label, engine in engines.items():
            try:
                ledger[label+'_engine_final'] = engine.metadata()
            except BaseException as secondary:
                ledger['secondary_errors'].append({'stage': label+'_metadata',
                                                   'error': repr(secondary)})
                if primary is not None:
                    primary.add_note('Secondary metadata: '+repr(secondary))
                else:
                    primary = secondary
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as secondary:
                ledger['secondary_errors'].append({'stage': 'threadpool_restore',
                                                   'error': repr(secondary)})
                if primary is not None:
                    primary.add_note('Secondary threadpool restore: '+repr(secondary))
                else:
                    primary = secondary
        ledger['internal_wall_s'] = time.perf_counter()-began
        ledger['internal_wall_limit_s'] = 120
        ledger['resource_screen'] = ('within_limits' if ledger['internal_wall_s'] <= 120
            else 'limit_exceeded')
        if result is not None:
            if ledger['resource_screen'] != 'within_limits':
                result['validity'] = 'resource_limit_incomplete'
                result['calculator_supported'] = False
            result['internal_wall_s'] = ledger['internal_wall_s']
            result['resource_screen'] = ledger['resource_screen']
            try:
                save(out/'result.json', result)
            except BaseException as secondary:
                ledger['secondary_errors'].append({'stage': 'result_save',
                                                   'error': repr(secondary)})
                if primary is not None:
                    primary.add_note('Secondary result save: '+repr(secondary))
                else:
                    primary = secondary
        if primary is not None:
            ledger['status'] = 'technical_failed'
            ledger['primary_error'] = {'type': type(primary).__name__,
                'message': str(primary),
                'traceback': ''.join(traceback.format_exception(primary)),
                'response_progress': getattr(primary, 'response_progress', None),
                'fixed_upload_progress': getattr(primary, 'fixed_upload_progress', None),
                'fixed_upload_by_key': getattr(primary, 'fixed_upload_by_key', None),
                'notes': list(getattr(primary, '__notes__', []))}
        try:
            save(out/'advance-ledger.json', ledger)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note('Secondary final ledger save: '+repr(secondary))
            else:
                primary = secondary
        if primary is not None:
            if 'primary_error' not in ledger:
                ledger['primary_error'] = {'type': type(primary).__name__,
                    'message': str(primary),
                    'traceback': ''.join(traceback.format_exception(primary)),
                    'notes': list(getattr(primary, '__notes__', []))}
            try:
                save(out/'execution-error.json', ledger['primary_error'])
            except BaseException as secondary:
                primary.add_note('Secondary execution error save: '+repr(secondary))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256,
        args.source_review, args.source_review_sha256, args.output)


if __name__ == '__main__':
    main()
