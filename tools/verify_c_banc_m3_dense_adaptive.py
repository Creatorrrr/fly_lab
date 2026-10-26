"""One-shot E127 CUDA dense-stage response on reviewed, frozen inputs.

This process never constructs or advances a native network or body.  The
parent supervisor owns the sole original-data invocation after source review.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.banc_m3_dense_adaptive import (
    DT,
    TICKS_PER_SAMPLE,
    CudaDenseAdaptiveEngine,
    cuda_warmup,
    directional_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'docs/BANC_M3_E127_DENSE_ADAPTIVE_DESIGN_20260921.md'
SOURCE_FILES = ('tools/banc_m3_dense_adaptive.py',
                'tools/verify_c_banc_m3_dense_adaptive.py',
                'tests/test_c_banc_dense_adaptive.py')
CONTEXT_NAMES = ('E110', 'E113', 'E121')
FIELDS = ('rate', 'adaptation', 'rate_sensitivity',
          'adaptation_sensitivity', 'activation',
          'activation_sensitivity', 'command', 'tendon')
INTERNAL_LIMIT_S = 90.
RESERVE_S = 15.
FIDELITY_LIMITS = {'rate': (.5, 5.), 'activation': (.25, 2.5),
                   'command': (.001, .01), 'tendon': (.0005, .005)}
FREEZE_FIXED = {
    'schema': 'flylab.m3.e127.dense-adaptive-freeze.v1',
    'design_path': 'docs/BANC_M3_E127_DENSE_ADAPTIVE_DESIGN_20260921.md',
    'fixture_path': 'verification/banc-natural-gait-m3-20260919/e127-inputs.npz',
    'original_fixture_path': 'verification/banc-natural-gait-m3-20260919/e124-inputs.npz',
    'reference_directory': 'verification/banc-natural-gait-m3-20260919/e126-dense-parent',
    'input_count': 31, 'reused_source_count': 31, 'preservation_count': 1432,
    'context_count': 3, 'motor_count': 391, 'muscle_count': 96,
    'neural_dt_s': .00025, 'control_dt_s': .005,
    'adaptation_tau_s': .3, 'muscle_tau_s': .01,
    'controls': 2000, 'stage_count': 160000,
    'native_ticks_per_context': 40000,
    'stage_current_dtype': 'float32', 'response_dtype': 'float64',
    'branch_dtype': 'uint8', 'branch_bits_per_tick': 4,
    'gain_min': 0., 'gain_max': 8.,
    'center_gain_rule': '1+.25*sin(j), j=0..95; frozen float64 array',
    'epsilons': [1e-5, 1e-6, 1e-7, 1e-8],
    'adjudication_epsilon': 1e-8,
    'batch_order': ['gain0', 'center', 'plus-1e-5', 'minus-1e-5',
                    'plus-1e-6', 'minus-1e-6', 'plus-1e-7', 'minus-1e-7',
                    'plus-1e-8', 'minus-1e-8'],
    'batch_count': 10, 'conditional_context_responses': 30,
    'conditional_model_s': 300., 'context_rk4_ticks': 1200000,
    'decoder_context_updates': 60000,
    'sensitivity_atol': 2e-5, 'sensitivity_rtol': 2e-4,
    'dense_reference_atol': 1e-8,
    'internal_wall_limit_s': 90., 'finalize_reserve_s': 15.,
    'external_wall_limit_s': 120., 'supervisor_limit_s': 125.,
    'parent_reference_internal_s': 35.,
    'parent_reference_external_s': 45.,
    'parent_reference_supervisor_s': 50.,
    'prior_actual_conservative_s': 1186.1228132999968,
    'prior_conservative_process_s': 1187.,
    'shared_process_limit_s': 1530.,
    'allocated_conservative_end_s': 1362.,
    'actual_neural_s': 0., 'actual_physical_s': 0.,
    'fit_calls': 0, 'automatic_retries': 0,
    'F6_exposed': False, 'candidate_adopted': False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True,
                                     allow_nan=False)+'\n', encoding='utf-8')


def _verified_relative(relative, expected):
    name = Path(relative)
    if (name.is_absolute() or '..' in name.parts
            or not isinstance(expected, str) or len(expected) != 64):
        raise ValueError('Invalid frozen relative path/hash: '+str(relative))
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or sha256(path) != expected:
        raise ValueError('Frozen file SHA mismatch: '+str(relative))
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if sha256(freeze_path) != freeze_sha:
        raise ValueError('E127 frozen design SHA mismatch')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    for key, expected in FREEZE_FIXED.items():
        if freeze.get(key) != expected:
            raise ValueError('E127 frozen constant changed: '+key)
    if freeze.get('design_sha256') != sha256(DESIGN):
        raise ValueError('E127 parent design changed')
    contexts = freeze.get('contexts')
    if (not isinstance(contexts, list) or len(contexts) != 3
            or [item.get('name') for item in contexts] != list(CONTEXT_NAMES)):
        raise ValueError('E127 original context order changed')
    for context in contexts:
        if set(context) != {'name', 'paths', 'parent_review', 'original_manifest'}:
            raise ValueError('E127 original context manifest changed')
        if set(context['paths']) != {'current', 'initial', 'endpoints',
                                    'metadata', 'result'}:
            raise ValueError('E127 original context path set changed')
    mappings = ('input_sha256', 'reused_source_sha256', 'preservation_sha256')
    for key, count in zip(mappings, (31, 31, 1432)):
        entries = freeze.get(key)
        if not isinstance(entries, dict) or len(entries) != count:
            raise ValueError('E127 frozen '+key+' cardinality changed')
        for relative, expected in entries.items():
            _verified_relative(relative, expected)
    inputs = freeze['input_sha256']
    if freeze['fixture_path'] not in inputs or freeze['original_fixture_path'] not in inputs:
        raise ValueError('E127 frozen fixture paths are not input entries')
    for context in contexts:
        for relative in (*context['paths'].values(), context['parent_review'],
                         context['original_manifest']):
            if relative not in inputs:
                raise ValueError('E127 context path is not a frozen input: '+relative)
    if sha256(review_path) != review_sha:
        raise ValueError('E127 source-review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_dense_adaptive') is not True
            or review.get('freeze_sha256') != freeze_sha
            or review.get('blocking_defects') != []):
        raise ValueError('Accepted exact E127 source review required')
    source_hashes = review.get('source_sha256')
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FILES):
        raise ValueError('Exact three-source E127 review map required')
    for relative, expected in source_hashes.items():
        _verified_relative(relative, expected)
    return freeze, review


def frozen_problem(freeze):
    with np.load(ROOT/freeze['fixture_path'], allow_pickle=False) as archive:
        expected = {'time_s', 'tau', 'a', 'threshold', 'cap', 'pool', 'moment',
                    'rest', 'ltm', 'membership', 'motor_ids', 'initial_rate',
                    'initial_adaptation', 'initial_activation', 'context_names',
                    'center_gain', 'epsilons'}
        if set(archive.files) != expected:
            raise ValueError('E127 exact fixture fields changed')
        fixture = {key: archive[key].copy() for key in archive.files}
    if (fixture['context_names'].tolist() != list(CONTEXT_NAMES)
            or fixture['time_s'].shape != (2001,)
            or fixture['time_s'].dtype != np.float64
            or fixture['initial_rate'].shape != (3, 391)
            or fixture['initial_adaptation'].shape != (3, 391)
            or fixture['initial_activation'].shape != (3, 96)
            or fixture['pool'].shape != (96, 391)
            or fixture['moment'].shape != (42, 96)
            or fixture['rest'].shape != (42,)
            or fixture['ltm'].shape != (6,)
            or fixture['center_gain'].shape != (96,)
            or fixture['center_gain'].dtype != np.float64
            or fixture['epsilons'].shape != (4,)
            or fixture['epsilons'].dtype != np.float64
            or not np.array_equal(fixture['epsilons'], np.array(FREEZE_FIXED['epsilons']))
            or not np.isfinite(fixture['center_gain']).all()
            or np.any((fixture['center_gain'] < 0) | (fixture['center_gain'] > 8))):
        raise ValueError('E127 exact fixture dimensions or gain grid changed')
    arrays = []
    for context in freeze['contexts']:
        array = np.load(ROOT/context['paths']['current'], mmap_mode='r',
                        allow_pickle=False)
        if array.shape != (160000, 391) or array.dtype != np.float32:
            raise ValueError('E127 exact native stage shape/dtype changed')
        arrays.append(array)
    fixture['stage_current'] = np.stack(arrays, axis=0)
    return fixture


def one_cpu_thread():
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    limits = threadpool_limits(limits=1)
    observed = threadpool_info()
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or any(pool['num_threads'] != 1 for pool in observed)):
        limits.restore_original_limits()
        raise RuntimeError('CPU/BLAS/Torch one-thread contract violated')
    return limits, {'torch_intraop': torch.get_num_threads(),
                    'torch_interop': torch.get_num_interop_threads(),
                    'threadpools': observed,
                    'environment': {key: os.environ.get(key) for key in
                                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                     'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def room(began, label, now=None):
    elapsed = (time.perf_counter() if now is None else now)-began
    if elapsed+RESERVE_S >= INTERNAL_LIMIT_S:
        raise TimeoutError(label+' violates E127 15 s persistence reserve')


def batch_gains(fixture):
    center = fixture['center_gain']
    values = [('gain0', np.zeros_like(center)), ('center', center.copy())]
    for h in fixture['epsilons']:
        suffix = format(float(h), '.0e').replace('e-0', 'e-')
        values.append(('plus-'+suffix, center+float(h)))
        values.append(('minus-'+suffix, center-float(h)))
    if [name for name, _ in values] != FREEZE_FIXED['batch_order']:
        raise ValueError('Frozen E127 gain batch order changed')
    return values


def save_response(path, gain, result):
    if Path(path).exists():
        raise FileExistsError('One-shot E127 raw response exists: '+str(path))
    np.savez(path, gain=np.asarray(gain, np.float64), **result)


def validate_saved_response(result, problem):
    required = set(FIELDS) | {'branch_bits'}
    if set(result) != required:
        raise ValueError('Complete E127 CUDA response fields required')
    c, stages, n = problem['stage_current'].shape
    s = stages//(4*TICKS_PER_SAMPLE)+1
    m = problem['pool'].shape[0]
    expected = {key: (c, s, n) for key in FIELDS[:4]}
    expected.update(activation=(c, s, m),
                    activation_sensitivity=(c, s, m),
                    command=(c, s, len(problem['rest'])),
                    tendon=(c, s, len(problem['ltm'])),
                    branch_bits=(c, stages//4, n))
    for key, value in result.items():
        dtype = np.uint8 if key == 'branch_bits' else np.float64
        if (not isinstance(value, np.ndarray) or value.dtype != dtype
                or value.shape != expected[key]):
            raise ValueError('Wrong E127 CUDA raw shape/dtype: '+key)
        if key == 'branch_bits':
            if np.any(value > 15):
                raise ValueError('E127 branch bits exceed four stages')
        elif not np.isfinite(value).all():
            raise ValueError('Nonfinite E127 CUDA raw '+key)
        elif key in ('rate', 'adaptation', 'activation') and np.any(value < 0):
            raise ValueError('Negative E127 physiological raw '+key)


def _metric(predicted, actual, start, stop):
    error = np.asarray(predicted[start:stop], np.float64)-actual[start:stop]
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite E127 numerical residual')
    magnitude = np.abs(error)
    local = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    return {'rms': float(np.sqrt(np.mean(error*error))),
            'max_abs': float(magnitude[local]),
            'max_location': [int(local[0])+start, int(local[1])]}


def fidelity_metrics(gain0, freeze, fixture):
    names = {'rate': 'motor_rate', 'adaptation': 'motor_adaptation',
             'activation': 'muscle_activation',
             'command': 'command', 'tendon': 'tendon'}
    report, passed = {}, True
    for context_index, context in enumerate(freeze['contexts']):
        with np.load(ROOT/context['paths']['endpoints'], allow_pickle=False) as archive:
            actual = {field: archive[name].copy() for field, name in names.items()}
        report[context['name']] = {}
        for field in names:
            expected = (2000, gain0[field].shape[-1])
            if actual[field].shape != expected or not np.isfinite(actual[field]).all():
                raise ValueError('E126 actual endpoint shape/finite mismatch: '+field)
            predicted = gain0[field][context_index, 1:]
            all_rows = _metric(predicted, actual[field], 0, 2000)
            first99 = _metric(predicted, actual[field], 0, 99)
            entry = {'all_2000': all_rows, 'startup_99': first99}
            if field in FIDELITY_LIMITS:
                rms_limit, max_limit = FIDELITY_LIMITS[field]
                valid = (all_rows['rms'] <= rms_limit
                         and all_rows['max_abs'] <= max_limit)
                entry.update(rms_limit=rms_limit, max_limit=max_limit,
                             passed=bool(valid))
                passed &= valid
            report[context['name']][field] = entry
    return report, bool(passed)


def dense_reference_metrics(gain0, freeze):
    directory = ROOT/freeze['reference_directory']
    report, passed = {}, True
    with np.load(directory/'dense-response-raw.npz', allow_pickle=False) as neural, \
         np.load(directory/'dense-decode-comparison-raw.npz', allow_pickle=False) as decoded:
        for field, source in (('rate', neural), ('adaptation', neural),
                              ('activation', decoded), ('command', decoded),
                              ('tendon', decoded)):
            reference = source[field]
            if reference.shape != gain0[field].shape:
                raise ValueError('E126 dense reference shape mismatch: '+field)
            difference = np.abs(gain0[field]-reference)
            if not np.isfinite(difference).all():
                raise ValueError('Nonfinite E126 dense reference difference')
            index = np.unravel_index(int(np.argmax(difference)), difference.shape)
            maximum = float(difference[index])
            valid = maximum <= FREEZE_FIXED['dense_reference_atol']
            report[field] = {'max_abs': maximum,
                             'max_location': [int(i) for i in index],
                             'limit': FREEZE_FIXED['dense_reference_atol'],
                             'passed': bool(valid)}
            passed &= valid
    return report, bool(passed)


def startup_target_metrics(gain0, center, freeze):
    with np.load(ROOT/freeze['original_fixture_path'], allow_pickle=False) as archive:
        target_q = archive['startup_target_command'].copy()
        target_ltm = archive['startup_target_tendon'].copy()
    if target_q.shape != (99, 42) or target_ltm.shape != (99, 6):
        raise ValueError('Original startup target shape changed')
    targets = {'command': (target_q, (.02, .05)),
               'tendon': (target_ltm, (.01, .03))}
    result = {}
    for context_index, context in enumerate(freeze['contexts']):
        with np.load(ROOT/context['paths']['endpoints'], allow_pickle=False) as archive:
            actual = {name: archive[name][:99].copy() for name in targets}
        result[context['name']] = {}
        for label, source in (('actual', actual),
                              ('gain0', {name: gain0[name][context_index, 1:100]
                                         for name in targets}),
                              ('center', {name: center[name][context_index, 1:100]
                                          for name in targets})):
            result[context['name']][label] = {}
            for name, (target, limits) in targets.items():
                predicted = source[name]
                if predicted.shape != target.shape or not np.isfinite(predicted).all():
                    raise ValueError('Original99 '+context['name']+' '+label+' '+name+
                                     ' shape/finite mismatch')
                metric = _metric(predicted, target, 0, 99)
                metric.update(rms_limit=limits[0], max_limit=limits[1],
                              passed=bool(metric['rms'] <= limits[0]
                                          and metric['max_abs'] <= limits[1]))
                result[context['name']][label][name] = metric
    return result


def branch_change_metrics(plus, minus):
    if plus.shape != minus.shape or plus.dtype != np.uint8 or minus.dtype != np.uint8:
        raise ValueError('E127 branch comparison shape/dtype mismatch')
    report = {}
    for context_index, name in enumerate(CONTEXT_NAMES[:plus.shape[0]]):
        change = np.bitwise_xor(plus[context_index], minus[context_index])
        report[name] = {'changed_ticks': int(np.count_nonzero(np.any(change != 0, axis=1))),
                        'changed_neuron_ticks': int(np.count_nonzero(change)),
                        'changed_bits_by_stage': [int(np.count_nonzero(change & (1 << bit)))
                                                  for bit in range(4)]}
    return report


def sensitivity_metrics(problem, center, plus, minus, h):
    analytic = directional_sensitivity(problem, center)
    report, passed = {}, True
    for field, derivative in analytic.items():
        finite_difference = (plus[field]-minus[field])/(2.*h)
        residual = np.abs(derivative-finite_difference)
        if not np.isfinite(residual).all():
            raise ValueError('Nonfinite E127 '+field+' sensitivity residual')
        limit = (FREEZE_FIXED['sensitivity_atol']
                 +FREEZE_FIXED['sensitivity_rtol']*
                 float(np.max(np.abs(finite_difference))))
        location = np.unravel_index(int(np.argmax(residual)), residual.shape)
        maximum = float(residual[location])
        valid = maximum <= limit
        passed &= valid
        report[field] = {'max_abs_error': maximum,
                         'max_location': [int(i) for i in location],
                         'limit': limit, 'passed': bool(valid)}
    report['branch_changes'] = branch_change_metrics(plus['branch_bits'],
                                                      minus['branch_bits'])
    return report, bool(passed)


def save_review_snapshot(output, freeze_path, freeze_sha,
                         review_path, review_sha, review):
    target = output/'reviewed-source'
    target.mkdir(exist_ok=False)
    sources = {'freeze.json': (freeze_path, freeze_sha),
               'source-review.json': (review_path, review_sha),
               'design.md': (DESIGN, sha256(DESIGN))}
    sources.update({Path(relative).name: (ROOT/relative, expected)
                    for relative, expected in review['source_sha256'].items()})
    for name, (source, expected) in sources.items():
        shutil.copyfile(source, target/name)
        if sha256(target/name) != expected:
            raise ValueError('Reviewed source snapshot changed: '+name)
    write_json(target/'manifest.json', {
        'schema': 'flylab.m3.e127.review-snapshot.v1',
        'files_sha256': {name: expected for name, (_, expected) in sources.items()}})


def _initial_ledger(freeze_sha, review_sha):
    return {'schema': 'flylab.m3.e127.advance-ledger.v1',
            'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
            'batch_attempts': [], 'completed_batches': 0,
            'completed_context_responses': 0,
            'context_rk4_ticks': 0, 'conditional_model_s': 0.,
            'decoder_endpoint_ticks': 0,
            'requested_context_rk4_ticks': 0,
            'requested_conditional_model_s': 0.,
            'requested_decoder_endpoint_ticks': 0,
            'unknown_requested_context_rk4_ticks': 0,
            'unknown_requested_conditional_model_s': 0.,
            'unknown_requested_decoder_endpoint_ticks': 0,
            'cuda_setup_context_rk4_ticks': 0,
            'cuda_setup_conditional_model_s': 0.,
            'cuda_setup_requested_context_rk4_ticks': 0,
            'cuda_setup_unknown_requested_context_rk4_ticks': 0,
            'main_fixed_upload_status': 'not_requested',
            'main_fixed_upload': None,
            'gain_upload_count_requested': 0,
            'gain_upload_bytes_requested': 0,
            'gain_upload_count_enqueued': 0,
            'gain_upload_bytes_enqueued': 0,
            'gain_upload_count_synchronized': 0,
            'gain_upload_bytes_synchronized': 0,
            'gain_upload_count_unknown_requested': 0,
            'gain_upload_bytes_unknown_requested': 0,
            'native_banc_ticks': 0, 'physical_steps': 0,
            'fit_runs': 0, 'lp_runs': 0}


def account_fixed_upload_failure(ledger, error, label):
    """Preserve constructor uploads even when no engine object was returned."""
    progress = {}
    for unit in ('count', 'bytes'):
        for state in ('requested', 'enqueued', 'synchronized', 'unknown_requested'):
            name = 'fixed_upload_'+unit+'_'+state
            progress[name] = getattr(error, name, None)
        progress['fixed_upload_'+unit+'_completed'] = getattr(
            error, 'fixed_upload_'+unit+'_completed', None)
    ledger[label+'_status'] = 'failed'
    ledger[label] = progress
    return progress


def account_gain_upload(ledger, progress):
    """Charge one attempted gain transfer exactly once, including uncertainty."""
    for unit in ('count', 'bytes'):
        for state in ('requested', 'enqueued', 'synchronized', 'unknown_requested'):
            name = 'gain_upload_'+unit+'_'+state
            ledger[name] += int(progress.get(name, 0))


def account_failed_response(ledger, attempt, error, requested_ticks,
                            requested_s, requested_decoder):
    """Charge a failed CUDA batch from its last reliable stream state."""
    attempt['status'] = 'failed'
    attempt['error'] = type(error).__name__+': '+str(error)
    progress = getattr(error, 'response_progress',
                       {'stage': 'unknown', 'completion_class': 'unknown'})
    attempt['progress'] = progress
    attempt['completion_class'] = progress.get('completion_class', 'unknown')
    account_gain_upload(ledger, progress)
    if progress.get('decoder_completion_class') == 'unknown':
        ledger['unknown_requested_decoder_endpoint_ticks'] += requested_decoder
    completed = getattr(error, 'completed_context_rk4_ticks', None)
    if completed is None:
        if progress.get('completion_class') != 'not_enqueued':
            ledger['unknown_requested_context_rk4_ticks'] += requested_ticks
            ledger['unknown_requested_conditional_model_s'] += requested_s
    else:
        attempt['context_rk4_ticks'] = int(completed)
        attempt['conditional_model_s'] = float(
            getattr(error, 'completed_conditional_model_s', int(completed)*DT))
        ledger['context_rk4_ticks'] += int(completed)
        ledger['conditional_model_s'] += attempt['conditional_model_s']
        ledger['completed_context_responses'] += 3
        decoder_done = progress.get('decoder_endpoint_ticks_completed')
        if decoder_done is not None:
            ledger['decoder_endpoint_ticks'] += decoder_done
    return progress


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError('E127 original output already exists: '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    ledger = _initial_ledger(freeze_sha, review_sha)
    primary, secondary, result_receipt = None, [], None
    limits = engine = None
    gpu_initialized = False
    active_name = active_gain = active_response = active_partial = None
    active_raw_saved = False
    setup_partial = None
    try:
        limits, ledger['cpu_threads'] = one_cpu_thread()
        freeze, review = reviewed_inputs(freeze_path, freeze_sha,
                                         review_path, review_sha)
        save_review_snapshot(output, freeze_path, freeze_sha,
                             review_path, review_sha, review)
        room(began, 'fixture load')
        fixture = frozen_problem(freeze)
        gains = batch_gains(fixture)
        write_json(output/'spec.json', {
            'schema': 'flylab.m3.e127.conditional-dense-adaptation-response.v1',
            'design_sha256': freeze['design_sha256'],
            'fixture_sha256': sha256(ROOT/freeze['fixture_path']),
            'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
            'fixed_native_stage_current': True, 'body_closed_loop': False,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False, 'internal_limit_s': INTERNAL_LIMIT_S,
            'reserve_s': RESERVE_S})
        room(began, 'CUDA compile/warmup')
        ledger['cuda_setup_status'] = 'requested'
        ledger['cuda_setup_requested_context_rk4_ticks'] = TICKS_PER_SAMPLE
        write_json(output/'advance-ledger.json', ledger)
        gpu_initialized = True
        try:
            setup = cuda_warmup()
        except Exception as error:
            ledger['cuda_setup_status'] = 'failed'
            progress = getattr(error, 'response_progress',
                               {'completion_class': 'unknown', 'stage': 'setup_failed'})
            ledger['cuda_setup_progress'] = progress
            account_fixed_upload_failure(ledger, error, 'cuda_setup_fixed_upload')
            ledger['cuda_setup_gain_upload'] = {
                name: getattr(error, name, progress.get(name))
                for unit in ('count', 'bytes')
                for state in ('requested', 'enqueued', 'synchronized', 'unknown_requested')
                for name in ('gain_upload_'+unit+'_'+state,)}
            completed = getattr(error, 'completed_context_rk4_ticks', None)
            if completed is None:
                if progress.get('completion_class') != 'not_enqueued':
                    ledger['cuda_setup_unknown_requested_context_rk4_ticks'] = TICKS_PER_SAMPLE
            else:
                ledger['cuda_setup_context_rk4_ticks'] = int(completed)
                ledger['cuda_setup_conditional_model_s'] = float(
                    getattr(error, 'completed_conditional_model_s', int(completed)*DT))
            setup_partial = getattr(error, 'partial_response', None)
            raise
        ledger['cuda_setup'] = setup
        ledger['cuda_setup_status'] = 'completed'
        ledger['cuda_setup_context_rk4_ticks'] = setup['setup_context_rk4_ticks']
        ledger['cuda_setup_conditional_model_s'] = setup['setup_conditional_model_s']
        ledger['cuda_setup_fixed_upload_status'] = 'synchronized'
        ledger['cuda_setup_gain_upload'] = {
            name.removeprefix('setup_'): value for name, value in setup.items()
            if name.startswith('setup_gain_upload_')}
        ledger['cuda_setup_fixed_upload'] = {
            name: setup['setup_'+name] for name in (
                'fixed_upload_count_requested', 'fixed_upload_bytes_requested',
                'fixed_upload_count_enqueued', 'fixed_upload_bytes_enqueued',
                'fixed_upload_count_synchronized', 'fixed_upload_bytes_synchronized',
                'fixed_upload_count_unknown_requested',
                'fixed_upload_bytes_unknown_requested')}
        write_json(output/'advance-ledger.json', ledger)
        room(began, 'one fixed CUDA input upload')
        ledger['main_fixed_upload_status'] = 'requested'
        write_json(output/'advance-ledger.json', ledger)
        try:
            engine = CudaDenseAdaptiveEngine(fixture)
        except Exception as error:
            account_fixed_upload_failure(ledger, error, 'main_fixed_upload')
            raise
        ledger['main_fixed_upload_status'] = 'synchronized'
        ledger['main_fixed_upload'] = {name: engine.metadata()[name] for name in (
            'fixed_upload_count_requested', 'fixed_upload_bytes_requested',
            'fixed_upload_count_enqueued', 'fixed_upload_bytes_enqueued',
            'fixed_upload_count_synchronized', 'fixed_upload_bytes_synchronized',
            'fixed_upload_count_unknown_requested',
            'fixed_upload_bytes_unknown_requested')}
        del fixture['stage_current']
        ledger['cuda_engine'] = engine.metadata()
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        write_json(output/'environment.json', {
            'python': sys.version, 'numpy': np.__version__,
            'cupy': cp.__version__,
            'gpu': gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name),
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'cuda_driver': cp.cuda.runtime.driverGetVersion(),
            'cpu_threads': ledger['cpu_threads'], 'fixed_upload': engine.metadata(),
            'setup': setup})
        for name, gain in gains:
            room(began, name+' response request')
            requested_ticks = 3*40000
            requested_s = 30.
            requested_decoder = 3*2000
            attempt = {'name': name, 'status': 'requested',
                       'requested_wall_s': time.perf_counter()-began,
                       'requested_context_rk4_ticks': requested_ticks,
                       'requested_conditional_model_s': requested_s,
                       'requested_decoder_endpoint_ticks': requested_decoder,
                       'completion_class': 'requested',
                       'context_rk4_ticks': None,
                       'conditional_model_s': None}
            ledger['batch_attempts'].append(attempt)
            ledger['requested_context_rk4_ticks'] += requested_ticks
            ledger['requested_conditional_model_s'] += requested_s
            ledger['requested_decoder_endpoint_ticks'] += requested_decoder
            write_json(output/'advance-ledger.json', ledger)
            active_name, active_gain = name, gain
            active_response, active_partial, active_raw_saved = None, None, False
            try:
                response, timing = engine.solve(gain)
            except Exception as error:
                account_failed_response(ledger, attempt, error, requested_ticks,
                                        requested_s, requested_decoder)
                active_partial = getattr(error, 'partial_response', None)
                raise
            active_response = response
            attempt.update(status='completed_unsaved',
                           completed_wall_s=time.perf_counter()-began,
                           completion_class='known_complete',
                           progress=timing['progress'],
                           context_rk4_ticks=timing['context_rk4_ticks'],
                           conditional_model_s=timing['conditional_model_s'],
                           response_wall_s=timing['wall_s'])
            ledger['context_rk4_ticks'] += timing['context_rk4_ticks']
            ledger['conditional_model_s'] += timing['conditional_model_s']
            ledger['completed_context_responses'] += 3
            ledger['decoder_endpoint_ticks'] += timing['decoder_endpoint_ticks']
            account_gain_upload(ledger, timing['progress'])
            raw_path = output/(name+'-raw.npz')
            try:
                save_response(raw_path, gain, response)
            except Exception:
                attempt['status'] = 'raw_save_failed'
                try:
                    save_response(output/(name+'-emergency-raw.npz'), gain, response)
                    attempt['emergency_raw_saved'] = True
                except Exception as save_error:  # noqa: BLE001 - retain secondary persistence failure
                    secondary.append({'stage': name+' emergency raw save',
                                      'error': repr(save_error)})
                raise
            active_raw_saved = True
            attempt['status'] = 'completed_raw_saved'
            attempt['raw_sha256'] = sha256(raw_path)
            ledger['completed_batches'] += 1
            validate_saved_response(response, engine.problem)
            active_name = active_gain = active_response = active_partial = None
            write_json(output/'advance-ledger.json', ledger)
        ledger['cuda_engine'] = engine.metadata()
        if (ledger['completed_batches'] != 10
                or ledger['completed_context_responses'] != 30
                or ledger['context_rk4_ticks'] != 1200000
                or ledger['conditional_model_s'] != 300.
                or ledger['decoder_endpoint_ticks'] != 60000
                or ledger['requested_decoder_endpoint_ticks'] != 60000
                or engine.fixed_upload_count != 13 or engine.solve_calls != 10
                or ledger['native_banc_ticks'] != 0
                or ledger['physical_steps'] != 0):
            raise ValueError('E127 exact batch/context/tick contract changed')
        # All ten full raw responses precede numerical adjudication.
        def load_raw(name):
            with np.load(output/(name+'-raw.npz'), allow_pickle=False) as archive:
                return {field: archive[field].copy() for field in
                        (*FIELDS, 'branch_bits')}
        gain0 = load_raw('gain0')
        center = load_raw('center')
        fidelity, fidelity_pass = fidelity_metrics(gain0, freeze, engine.problem)
        dense_reference, dense_pass = dense_reference_metrics(gain0, freeze)
        startup = startup_target_metrics(gain0, center, freeze)
        sensitivity, sensitivity_pass = {}, True
        for h in freeze['epsilons']:
            suffix = format(float(h), '.0e').replace('e-0', 'e-')
            plus, minus = load_raw('plus-'+suffix), load_raw('minus-'+suffix)
            values, valid = sensitivity_metrics(engine.problem, center, plus, minus,
                                                float(h))
            sensitivity[suffix] = {'fields': values, 'passed': valid}
            if h == freeze['adjudication_epsilon']:
                sensitivity_pass = valid
            del plus, minus
        metrics = {'gain0_fidelity': fidelity, 'gain0_dense_reference': dense_reference,
                   'original99_startup_target': startup,
                   'sensitivity_fd': sensitivity,
                   'fidelity_pass': fidelity_pass,
                   'dense_reference_pass': dense_pass,
                   'sensitivity_pass': sensitivity_pass,
                   'adjudication_epsilon': freeze['adjudication_epsilon']}
        write_json(output/'metrics.json', metrics)
        result_receipt = {
            'status': 'valid_recorded_dense_adaptive_diagnostic',
            'run_valid': True, 'fidelity_pass': fidelity_pass,
            'dense_reference_pass': dense_pass,
            'sensitivity_pass': sensitivity_pass,
            'numerical_pass': bool(fidelity_pass and dense_pass and sensitivity_pass),
            'completed_batches': 10, 'context_rk4_ticks': 1200000,
            'conditional_model_s': 300., 'decoder_endpoint_ticks': 60000,
            'actual_neural_s': 0., 'actual_physical_s': 0., 'fit_calls': 0,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False, 'wall_s': None}
    except Exception as error:  # noqa: BLE001 - re-raise after preserving raw and accounting
        primary = error
    finally:
        if setup_partial:
            try:
                np.savez(output/'cuda-setup-partial-raw.npz', **setup_partial)
            except Exception as error:  # noqa: BLE001 - retain secondary persistence failure
                secondary.append({'stage': 'setup partial raw rescue', 'error': repr(error)})
        if active_name is not None and active_gain is not None:
            if active_response is not None and not active_raw_saved:
                for suffix in ('-raw.npz', '-emergency-raw.npz', '-final-rescue-raw.npz'):
                    try:
                        save_response(output/(active_name+suffix), active_gain,
                                      active_response)
                        active_raw_saved = True
                        break
                    except Exception as error:  # noqa: BLE001 - retain secondary persistence failure
                        secondary.append({'stage': active_name+suffix+' rescue',
                                          'error': repr(error)})
            if active_partial:
                try:
                    np.savez(output/(active_name+'-partial-raw.npz'),
                             gain=active_gain, **active_partial)
                except Exception as error:  # noqa: BLE001 - retain secondary persistence failure
                    secondary.append({'stage': active_name+' partial raw rescue',
                                      'error': repr(error)})
        if gpu_initialized:
            try:
                import cupy as cp
                if engine is not None:
                    ledger['cuda_engine'] = engine.metadata()
                cp.cuda.runtime.deviceSynchronize()
                engine = None
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            except Exception as error:  # noqa: BLE001 - preserve cleanup failure alongside primary
                secondary.append({'stage': 'CUDA cleanup', 'error': repr(error)})
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:  # noqa: BLE001 - preserve cleanup failure alongside primary
                secondary.append({'stage': 'CPU thread restore', 'error': repr(error)})
        ledger['wall_s_before_final_receipts'] = time.perf_counter()-began
        try:
            write_json(output/'advance-ledger.json', ledger)
        except Exception as error:  # noqa: BLE001 - retain secondary persistence failure
            secondary.append({'stage': 'ledger save', 'error': repr(error)})
        if primary is None and secondary:
            primary = RuntimeError('E127 cleanup/ledger persistence failed')
        if primary is None and time.perf_counter()-began >= INTERNAL_LIMIT_S:
            primary = TimeoutError('E127 internal90 s limit exceeded')
        if primary is None:
            result_receipt['wall_s'] = time.perf_counter()-began
            try:
                write_json(output/'result.json', result_receipt)
            except Exception as error:  # noqa: BLE001 - result save becomes the primary failure
                primary = error
            if primary is None and time.perf_counter()-began >= INTERNAL_LIMIT_S:
                primary = TimeoutError('E127 final result save exceeded90 s')
        if primary is not None:
            if (output/'result.json').exists():
                result_receipt['run_valid'] = False
                result_receipt['status'] = 'incomplete'
                result_receipt['wall_s'] = time.perf_counter()-began
                try:
                    write_json(output/'result.json', result_receipt)
                except Exception as error:  # noqa: BLE001 - retain secondary persistence failure
                    secondary.append({'stage': 'result invalidation',
                                      'error': repr(error)})
            try:
                write_json(output/'execution-error.json', {
                    'primary': {'error': type(primary).__name__,
                                'message': str(primary)},
                    'secondary': secondary,
                    'context_rk4_ticks': ledger['context_rk4_ticks'],
                    'conditional_model_s': ledger['conditional_model_s'],
                    'requested_context_rk4_ticks': ledger['requested_context_rk4_ticks'],
                    'unknown_requested_context_rk4_ticks':
                        ledger['unknown_requested_context_rk4_ticks'],
                    'actual_neural_s': 0., 'actual_physical_s': 0.,
                    'wall_s': time.perf_counter()-began,
                    'candidate_adopted': False, 'F6_exposed': False,
                    'whole_goal_complete': False})
            except Exception as error:  # noqa: BLE001 - keep the primary error if its receipt fails
                primary.add_note('Error receipt save failed: '+repr(error))
    if primary is not None:
        for item in secondary:
            primary.add_note('Secondary '+item['stage']+': '+item['error'])
        raise primary
    return result_receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha,
                 args.source_review, args.source_review_sha, args.out)
    if not result['numerical_pass']:
        sys.exit(1)
