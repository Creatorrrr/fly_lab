"""One-shot E124 recorded-current response and conditional sensitivity diagnostic.

The parent supervisor owns the sole original invocation after source review.
This runner never constructs GraphStore, the native BANC network, or a body.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.banc_m3_adaptive_response import (
    SUBSTEPS,
    CudaAdaptiveResponseEngine,
    cuda_warmup,
    directional_sensitivity,
    prepare_problem,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'docs/BANC_M3_E124_ADAPTIVE_RESPONSE_DESIGN_20260921.md'
INTERNAL_LIMIT_S = 90.
RESERVE_S = 15.
EPS = 1e-4
GAIN_ORDER = (('gain0', 0.), ('gain1', 1.),
              ('gain1_plus', 1.+EPS), ('gain1_minus', 1.-EPS))
SOURCE_FILES = ('tools/banc_m3_adaptive_response.py',
                'tools/verify_c_banc_m3_adaptive_response.py',
                'tests/test_c_banc_adaptive_response.py')


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
    if (name.is_absolute() or '..' in name.parts or not isinstance(expected, str)
            or len(expected) != 64 or sha256(ROOT/name) != expected):
        raise ValueError('Frozen file SHA mismatch: '+str(relative))
    return ROOT/name


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if sha256(freeze_path) != freeze_sha:
        raise ValueError('E124 frozen design SHA mismatch')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    fixed = {'schema': 'flylab.m3.e124.adaptive-response-freeze.v1',
        'design_path': 'docs/BANC_M3_E124_ADAPTIVE_RESPONSE_DESIGN_20260921.md',
        'fixture_path': 'verification/banc-natural-gait-m3-20260919/e124-inputs.npz',
        'context_names': ['E110', 'E113', 'E121'],
        'gain_values': [0., 1., 1.+EPS, 1.-EPS], 'batch_count': 4,
        'conditional_responses': 12, 'conditional_model_s': 120.,
        'context_rk4_ticks': 480000, 'dt': .005, 'substep_s': .00025,
        'adaptation_tau_s': .3, 'activation_tau_s': .01,
        'gain_min': 0., 'gain_max': 8., 'finite_difference_epsilon': EPS,
        'fidelity_rms_max': {'motor_rate': [.5, 5.],
                             'muscle_activation': [.25, 2.5],
                             'command': [.001, .01], 'tendon': [.0005, .005]},
        'sensitivity_atol': 2e-5, 'sensitivity_rtol': 2e-4,
        'internal_wall_limit_s': INTERNAL_LIMIT_S,
        'finalize_reserve_s': RESERVE_S, 'external_wall_limit_s': 120.,
        'supervisor_limit_s': 125., 'prior_conservative_process_s': 940.,
        'prior_actual_conservative_s': 939.0747094999533,
        'shared_process_limit_s': 1065., 'automatic_retries': 0,
        'actual_neural_s': 0, 'actual_physical_s': 0,
        'fit_calls': 0, 'LP_calls': 0, 'F6_exposed': False,
        'candidate_adopted': False}
    if any(freeze.get(key) != value for key, value in fixed.items()):
        raise ValueError('E124 frozen design contract changed')
    if freeze.get('design_sha256') != sha256(DESIGN):
        raise ValueError('E124 parent design changed')
    for key, count_key in (('input_sha256', 'input_count'),
                           ('reused_source_sha256', 'reused_source_count')):
        mapping = freeze.get(key)
        if (not isinstance(mapping, dict) or not isinstance(freeze.get(count_key), int)
                or len(mapping) != freeze[count_key]):
            raise ValueError('Frozen '+key+' count mismatch')
        for relative, expected in mapping.items():
            _verified_relative(relative, expected)
    preserved = freeze.get('preservation_sha256')
    if (not isinstance(preserved, dict)
            or {key: len(value) for key, value in preserved.items()}
            != {'E121': 211, 'E122': 21, 'E123': 62, 'M2': 263}):
        raise ValueError('Full E121/E122/E123/M2 preservation map required')
    for files in preserved.values():
        for relative, expected in files.items():
            _verified_relative(relative, expected)
    fixture_names = [relative for relative in freeze['input_sha256']
                     if Path(relative).name == 'e124-inputs.npz']
    if fixture_names != [freeze['fixture_path']]:
        raise ValueError('Exactly one frozen E124 fixture required')
    fixture = ROOT/fixture_names[0]
    if sha256(review_path) != review_sha:
        raise ValueError('E124 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_response_probe') is not True
            or review.get('freeze_sha256') != freeze_sha
            or review.get('blocking_defects') != []):
        raise ValueError('Accepted exact E124 source review required')
    source_hashes = review.get('source_sha256')
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FILES):
        raise ValueError('Exact three-source E124 review map required')
    for relative, expected in source_hashes.items():
        _verified_relative(relative, expected)
    return freeze, fixture, review


def original_fixture(path):
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    if set(source) != {
            'time_s', 'context_names', 'current', 'initial_rate',
            'initial_adaptation', 'initial_activation', 'actual_motor_rate',
            'actual_adaptation', 'actual_activation', 'actual_command',
            'actual_tendon', 'tau', 'a', 'threshold', 'cap', 'pool', 'moment',
            'rest', 'ltm', 'membership', 'motor_ids', 'startup_mask',
            'startup_target_command', 'startup_target_tendon'}:
        raise ValueError('Exact E124 recorded-input fixture fields required')
    problem = prepare_problem(source)
    if (source['context_names'].tolist() != ['E110', 'E113', 'E121']
            or problem['current'].shape != (3, 2001, 391)
            or problem['initial_activation'].shape != (3, 96)
            or problem['moment'].shape != (42, 96)
            or problem['rest'].shape != (42,)
            or problem['ltm'].shape != (6,)
            or source['actual_motor_rate'].shape != (3, 2001, 391)
            or source['actual_adaptation'].shape != (3, 2001, 391)
            or source['actual_activation'].shape != (3, 2001, 96)
            or source['actual_command'].shape != (3, 2001, 42)
            or source['actual_tendon'].shape != (3, 2001, 6)
            or source['startup_mask'].shape != (2001,)
            or source['startup_mask'].dtype != bool
            or source['startup_target_command'].shape != (99, 42)
            or source['startup_target_tendon'].shape != (99, 6)
            or not np.array_equal(np.flatnonzero(source['startup_mask']),
                                  np.arange(1, 100))
            or problem['time_s'][-1] != 10.):
        raise ValueError('Exact three-context 10 s original fixture required')
    for key in ('actual_motor_rate', 'actual_adaptation', 'actual_activation',
                'actual_command', 'actual_tendon',
                'startup_target_command', 'startup_target_tendon'):
        if not np.isfinite(source[key]).all():
            raise ValueError('Nonfinite recorded '+key)
    for key, initial in (('actual_motor_rate', 'initial_rate'),
                         ('actual_adaptation', 'initial_adaptation'),
                         ('actual_activation', 'initial_activation')):
        if not np.array_equal(source[key][:, 0], source[initial]):
            raise ValueError('Actual initial '+key+' differs from recorded t0')
    return problem, source


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
                    'threadpools': observed}


def room(began, label, now=None):
    elapsed = (time.perf_counter() if now is None else now)-began
    if elapsed+RESERVE_S >= INTERNAL_LIMIT_S:
        raise TimeoutError(label+' violates E124 15 s persistence reserve')


def _metric(predicted, actual, mask):
    error = np.asarray(predicted[:, mask], np.float64)-actual[:, mask]
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite E124 response residual')
    abs_error = np.abs(error)
    flat = int(np.argmax(abs_error))
    local = tuple(int(i) for i in np.unravel_index(flat, error.shape))
    actual_location = [local[0], int(mask[local[1]]), *local[2:]]
    return {'rms': float(np.sqrt(np.mean(error*error))),
            'max_abs': float(abs_error[local]),
            'max_location': actual_location}


def fidelity_metrics(result, fixture):
    all_mask = np.arange(1, 2001)
    startup = np.flatnonzero(fixture['startup_mask'])
    limits = {'rate': (.5, 5.), 'activation': (.25, 2.5),
              'command': (.001, .01), 'tendon': (.0005, .005)}
    fields = {'rate': 'actual_motor_rate', 'activation': 'actual_activation',
              'command': 'actual_command', 'tendon': 'actual_tendon'}
    values = {}
    passed = True
    for field, actual_key in fields.items():
        values[field] = {}
        for context, name in enumerate(fixture['context_names'].tolist()):
            predicted = result[field][context:context+1]
            actual = np.asarray(fixture[actual_key][context:context+1],
                                np.float64)
            whole = _metric(predicted, actual, all_mask)
            early = _metric(predicted, actual, startup)
            limits_rms, limits_max = limits[field]
            valid = whole['rms'] <= limits_rms and whole['max_abs'] <= limits_max
            passed &= valid
            values[field][name] = {'all_2000': whole, 'startup_99': early,
                                   'rms_limit': limits_rms,
                                   'max_limit': limits_max, 'passed': bool(valid)}
    # Adaptation is measured at both scopes but is not an E124 fidelity gate.
    values['adaptation'] = {}
    for i, name in enumerate(fixture['context_names'].tolist()):
        predicted = result['adaptation'][i:i+1]
        actual = np.asarray(fixture['actual_adaptation'][i:i+1], np.float64)
        values['adaptation'][name] = {
            'all_2000': _metric(predicted, actual, all_mask),
            'startup_99': _metric(predicted, actual, startup)}
    return values, bool(passed)


def precision_and_sampling_descriptors(problem, result):
    """Keep measurable rounding and sampled-current limits distinct.

    Neither quantity is a causal decomposition of the recorded native error:
    the true current between 5 ms samples and cumulative float32 integration
    roundoff were not recorded.
    """
    rounding = {}
    for key in ('rate', 'adaptation', 'activation', 'command', 'tendon'):
        field = result[key][:, 1:]
        difference = field-field.astype(np.float32).astype(np.float64)
        rounding[key] = {'rms': float(np.sqrt(np.mean(difference*difference))),
                         'max_abs': float(np.max(np.abs(difference)))}
    chord = np.diff(problem['current'], axis=1)
    return {'float32_endpoint_rounding_only': rounding,
            'recorded_current_5ms_endpoint_chord': {
                'rms': float(np.sqrt(np.mean(chord*chord))),
                'max_abs': float(np.max(np.abs(chord))),
                'sampling_interval_s': .005},
            'unobserved_between_sample_current_error': 'not identifiable',
            'unobserved_cumulative_native_float32_roundoff': 'not identifiable'}


def sensitivity_metrics(problem, center, plus, minus):
    analytic = directional_sensitivity(problem, center)
    values = {}
    passed = True
    for key, derivative in analytic.items():
        difference = (plus[key]-minus[key])/(2.*EPS)
        residual = np.abs(derivative-difference)
        if not np.isfinite(residual).all():
            raise ValueError('Nonfinite '+key+' sensitivity residual')
        threshold = 2e-5+2e-4*float(np.max(np.abs(difference)))
        flat = int(np.argmax(residual))
        location = tuple(int(i) for i in np.unravel_index(flat, residual.shape))
        valid = float(residual[location]) <= threshold
        passed &= valid
        values[key] = {'max_abs_error': float(residual[location]),
                       'max_location': list(location), 'limit': threshold,
                       'passed': bool(valid)}
    return values, bool(passed)


def effect_metrics(fixture, baseline, changed):
    startup = np.flatnonzero(fixture['startup_mask'])
    later = np.arange(100, 2001)
    result = {}
    for key in ('rate', 'adaptation', 'activation', 'command', 'tendon'):
        result[key] = {}
        for context, name in enumerate(fixture['context_names'].tolist()):
            delta = changed[key][context]-baseline[key][context]
            result[key][name] = {
                'startup_99_rms': float(np.sqrt(np.mean(delta[startup]**2))),
                'startup_99_max_abs': float(np.max(np.abs(delta[startup]))),
                'later_0p5_to_10_rms': float(np.sqrt(np.mean(delta[later]**2))),
                'later_0p5_to_10_max_abs': float(np.max(np.abs(delta[later])))}
    return result


def save_response(path, gain, result):
    if Path(path).exists():
        raise FileExistsError('One-shot raw response already exists')
    np.savez(path, gain=np.asarray(gain, np.float64), **result)


def validate_saved_response(result, problem):
    required = {'rate', 'adaptation', 'rate_sensitivity',
                'adaptation_sensitivity', 'activation',
                'activation_sensitivity', 'command', 'tendon'}
    if set(result) != required:
        raise ValueError('Complete CUDA response fields required')
    c, t, n = problem['current'].shape
    m = problem['pool'].shape[0]
    expected = {key: (c, t, n) for key in ('rate', 'adaptation',
                 'rate_sensitivity', 'adaptation_sensitivity')}
    expected.update(activation=(c, t, m), activation_sensitivity=(c, t, m),
                    command=(c, t, len(problem['rest'])),
                    tendon=(c, t, len(problem['ltm'])))
    for key, value in result.items():
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != expected[key]):
            raise ValueError('Wrong CUDA raw shape or dtype: '+key)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite CUDA raw '+key)
        if key in ('rate', 'adaptation', 'activation') and np.any(value < 0):
            raise ValueError('Negative CUDA physiological raw '+key)


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
            raise ValueError('Reviewed source copy changed: '+name)
    write_json(target/'manifest.json', {
        'schema': 'flylab.m3-e124-review-snapshot.v1',
        'files_sha256': {name: expected for name, (_, expected) in sources.items()}})


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError('E124 original output already exists: '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    ledger = {'batch_attempts': [], 'completed_batches': 0,
              'completed_context_responses': 0, 'context_rk4_ticks': 0,
              'conditional_model_s': 0., 'native_banc_ticks': 0,
              'physical_steps': 0, 'fit_runs': 0, 'lp_runs': 0,
              'requested_context_rk4_ticks': 0,
              'requested_conditional_model_s': 0.,
              'unknown_requested_context_rk4_ticks': 0,
              'unknown_requested_conditional_model_s': 0.,
              'decoder_endpoint_ticks': 0,
              'requested_decoder_endpoint_ticks': 0,
              'unknown_requested_decoder_endpoint_ticks': 0,
              'cuda_setup_context_rk4_ticks': 0,
              'cuda_setup_conditional_model_s': 0.,
              'cuda_setup_requested_context_rk4_ticks': 0,
              'cuda_setup_unknown_requested_context_rk4_ticks': 0,
              'source_review_sha256': review_sha, 'freeze_sha256': freeze_sha}
    primary = None
    secondary = []
    result_receipt = None
    limits = None
    responses = {}
    gpu_initialized = False
    active_name = None
    active_gain = None
    active_response = None
    active_raw_saved = False
    active_partial = None
    setup_partial = None
    engine = None
    try:
        limits, ledger['cpu_threads'] = one_cpu_thread()
        freeze, fixture_path, review = reviewed_inputs(
            freeze_path, freeze_sha, review_path, review_sha)
        save_review_snapshot(output, freeze_path, freeze_sha,
                             review_path, review_sha, review)
        room(began, 'fixture load')
        problem, fixture = original_fixture(fixture_path)
        write_json(output/'spec.json', {
            'schema': 'flylab.m3-e124-conditional-adaptation-response.v1',
            'design_sha256': freeze['design_sha256'],
            'fixture_sha256': sha256(fixture_path),
            'freeze_sha256': freeze_sha, 'source_review_sha256': review_sha,
            'fixed_current': True, 'body_closed_loop': False,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False, 'internal_limit_s': INTERNAL_LIMIT_S,
            'reserve_s': RESERVE_S})
        room(began, 'CUDA compile/warmup')
        ledger['cuda_setup_status'] = 'requested'
        ledger['cuda_setup_requested_context_rk4_ticks'] = SUBSTEPS
        write_json(output/'advance-ledger.json', ledger)
        gpu_initialized = True
        try:
            setup = cuda_warmup()
        except Exception as error:
            ledger['cuda_setup_status'] = 'failed'
            ledger['cuda_setup_progress'] = getattr(error, 'response_progress',
                {'completion_class': 'unknown', 'stage': 'setup_failed'})
            ledger['cuda_setup_fixed_upload_count_completed'] = getattr(
                error, 'fixed_upload_count_completed', None)
            ledger['cuda_setup_fixed_upload_bytes_completed'] = getattr(
                error, 'fixed_upload_bytes_completed', None)
            completed = getattr(error, 'completed_context_rk4_ticks', None)
            if completed is None:
                if ledger['cuda_setup_progress'].get('completion_class') != 'not_enqueued':
                    ledger['cuda_setup_unknown_requested_context_rk4_ticks'] = SUBSTEPS
            else:
                ledger['cuda_setup_context_rk4_ticks'] = int(completed)
                ledger['cuda_setup_conditional_model_s'] = float(
                    getattr(error, 'completed_conditional_model_s',
                            int(completed)*.00025))
            ledger['cuda_setup_decoder_endpoint_ticks_completed'] = (
                ledger['cuda_setup_progress'].get('decoder_endpoint_ticks_completed'))
            setup_partial = getattr(error, 'partial_response', None)
            raise
        ledger['cuda_setup'] = setup
        ledger['cuda_setup_status'] = 'completed'
        ledger['cuda_setup_context_rk4_ticks'] = setup['setup_context_rk4_ticks']
        ledger['cuda_setup_conditional_model_s'] = setup['setup_conditional_model_s']
        write_json(output/'advance-ledger.json', ledger)
        room(began, 'one fixed CUDA input upload')
        engine = CudaAdaptiveResponseEngine(problem)
        ledger['cuda_engine'] = engine.metadata()
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        write_json(output/'environment.json', {
            'python': sys.version, 'numpy': np.__version__,
            'cupy': cp.__version__, 'gpu': gpu_name.decode() if isinstance(
                gpu_name, bytes) else str(gpu_name),
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'cuda_driver': cp.cuda.runtime.driverGetVersion(),
            'cpu_threads': ledger['cpu_threads'],
            'fixed_upload': engine.metadata(),
            'setup': setup,
            'decoder_backend': 'cupy_cuda_float64_pool_ema_q_tendon'})
        for name, value in GAIN_ORDER:
            room(began, name+' response request')
            gain = np.full(96, value, np.float64)
            requested_ticks = 3*2000*SUBSTEPS
            requested_s = 3*2000*.005
            attempt = {'name': name, 'gain': value, 'status': 'requested',
                       'requested_wall_s': time.perf_counter()-began,
                       'requested_context_rk4_ticks': requested_ticks,
                       'requested_conditional_model_s': requested_s,
                       'requested_decoder_endpoint_ticks': 6000,
                       'completion_class': 'requested',
                       'context_rk4_ticks': None,
                       'conditional_model_s': None}
            ledger['batch_attempts'].append(attempt)
            ledger['requested_context_rk4_ticks'] += requested_ticks
            ledger['requested_conditional_model_s'] += requested_s
            ledger['requested_decoder_endpoint_ticks'] += 6000
            write_json(output/'advance-ledger.json', ledger)
            active_name, active_gain = name, gain
            active_response, active_partial, active_raw_saved = None, None, False
            try:
                response, timing = engine.solve(gain)
            except Exception as error:
                attempt['status'] = 'failed'
                attempt['error'] = type(error).__name__+': '+str(error)
                progress = getattr(error, 'response_progress',
                    {'stage': 'unknown', 'completion_class': 'unknown'})
                attempt['progress'] = progress
                attempt['completion_class'] = progress.get('completion_class',
                                                           'unknown')
                if progress.get('decoder_completion_class') == 'unknown':
                    ledger['unknown_requested_decoder_endpoint_ticks'] += 6000
                completed_ticks = getattr(error,
                    'completed_context_rk4_ticks', None)
                if completed_ticks is None:
                    if progress.get('completion_class') != 'not_enqueued':
                        ledger['unknown_requested_context_rk4_ticks'] += requested_ticks
                        ledger['unknown_requested_conditional_model_s'] += requested_s
                else:
                    attempt['context_rk4_ticks'] = int(completed_ticks)
                    attempt['conditional_model_s'] = float(
                        getattr(error, 'completed_conditional_model_s',
                                int(completed_ticks)*.00025))
                    ledger['context_rk4_ticks'] += int(completed_ticks)
                    ledger['conditional_model_s'] += float(
                        attempt['conditional_model_s'])
                    ledger['completed_context_responses'] += 3
                    decoder_ticks = progress.get('decoder_endpoint_ticks_completed')
                    if decoder_ticks is not None:
                        ledger['decoder_endpoint_ticks'] += decoder_ticks
                active_partial = getattr(error, 'partial_response', None)
                raise
            active_response = response  # Retain before any ledger I/O.
            responses[name] = response
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
            try:
                save_response(output/(name+'-raw.npz'), gain, response)
            except Exception:
                attempt['status'] = 'raw_save_failed'
                try:
                    save_response(output/(name+'-emergency-raw.npz'),
                                  gain, response)
                    attempt['emergency_raw_saved'] = True
                except Exception as save_error:  # noqa: BLE001
                    secondary.append({'stage': name+' emergency raw save',
                                      'error': repr(save_error)})
                raise
            active_raw_saved = True
            attempt['status'] = 'completed_raw_saved'
            ledger['completed_batches'] += 1
            validate_saved_response(response, problem)  # Raw precedes all assertions.
            active_name = active_gain = active_response = active_partial = None
            write_json(output/'advance-ledger.json', ledger)
        ledger['cuda_engine'] = engine.metadata()
        fidelity, fidelity_pass = fidelity_metrics(responses['gain0'], fixture)
        sensitivity, sensitivity_pass = sensitivity_metrics(problem,
            responses['gain1'], responses['gain1_plus'], responses['gain1_minus'])
        effects = effect_metrics(fixture, responses['gain0'], responses['gain1'])
        metrics = {'gain0_fidelity': fidelity, 'sensitivity_fd': sensitivity,
                   'gain1_minus_gain0_recorded_current': effects,
                   'precision_and_sampling': precision_and_sampling_descriptors(
                       problem, responses['gain0']),
                   'fidelity_pass': fidelity_pass,
                   'sensitivity_pass': sensitivity_pass}
        write_json(output/'metrics.json', metrics)
        if (ledger['completed_batches'] != 4
                or ledger['completed_context_responses'] != 12
                or ledger['context_rk4_ticks'] != 480000
                or ledger['conditional_model_s'] != 120.
                or ledger['decoder_endpoint_ticks'] != 24000
                or ledger['requested_decoder_endpoint_ticks'] != 24000
                or engine.fixed_upload_count != 13
                or engine.solve_calls != 4
                or ledger['native_banc_ticks'] != 0
                or ledger['physical_steps'] != 0):
            raise ValueError('E124 exact batch/context/tick contract changed')
        if not fidelity_pass or not sensitivity_pass:
            raise ValueError('E124 recorded-current fidelity or FD gate failed')
        result_receipt = {'validity': 'valid_conditional_recorded_current_diagnostic',
            'fidelity_pass': True, 'sensitivity_pass': True,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False, 'conditional_model_s': 120.,
            'context_rk4_ticks': 480000, 'actual_neural_s': 0.,
            'actual_physical_s': 0., 'wall_s': None}
    except Exception as error:  # noqa: BLE001 - preserve first failure
        primary = error
    finally:
        if setup_partial:
            try:
                np.savez(output/'cuda-setup-partial-raw.npz', **setup_partial)
            except Exception as error:  # noqa: BLE001 - preserve setup primary
                secondary.append({'stage': 'setup partial raw rescue',
                                  'error': repr(error)})
        if active_name is not None and active_gain is not None:
            if active_response is not None and not active_raw_saved:
                for suffix in ('-raw.npz', '-emergency-raw.npz',
                               '-final-rescue-raw.npz'):
                    try:
                        save_response(output/(active_name+suffix),
                                      active_gain, active_response)
                        active_raw_saved = True
                        break
                    except Exception as error:  # noqa: BLE001 - preserve primary
                        secondary.append({'stage': active_name+suffix+' rescue',
                                          'error': repr(error)})
            if active_partial:
                try:
                    np.savez(output/(active_name+'-partial-raw.npz'),
                             gain=active_gain, **active_partial)
                except Exception as error:  # noqa: BLE001 - preserve primary
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
            except Exception as error:  # noqa: BLE001 - retain primary
                secondary.append({'stage': 'CUDA cleanup', 'error': repr(error)})
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:  # noqa: BLE001
                secondary.append({'stage': 'CPU thread restore', 'error': repr(error)})
        ledger['wall_s_before_final_receipts'] = time.perf_counter()-began
        try:
            write_json(output/'advance-ledger.json', ledger)
        except Exception as error:  # noqa: BLE001
            secondary.append({'stage': 'ledger save', 'error': repr(error)})
        if primary is None and secondary:
            primary = RuntimeError('E124 cleanup/ledger persistence failed')
        if primary is None and time.perf_counter()-began >= INTERNAL_LIMIT_S:
            primary = TimeoutError('E124 internal90 s limit exceeded')
        if primary is None:
            result_receipt['wall_s'] = time.perf_counter()-began
            try:
                write_json(output/'result.json', result_receipt)
            except Exception as error:  # noqa: BLE001
                primary = error
            if primary is None and time.perf_counter()-began >= INTERNAL_LIMIT_S:
                primary = TimeoutError('E124 final result save exceeded90 s')
        if primary is not None:
            if (output/'result.json').exists():
                result_receipt['validity'] = 'incomplete'
                result_receipt['wall_s'] = time.perf_counter()-began
                try:
                    write_json(output/'result.json', result_receipt)
                except Exception as error:  # noqa: BLE001
                    secondary.append({'stage': 'result invalidation',
                                      'error': repr(error)})
            try:
                write_json(output/'execution-error.json', {
                    'primary': {'error': type(primary).__name__,
                                'message': str(primary)},
                    'secondary': secondary, 'context_rk4_ticks':
                    ledger['context_rk4_ticks'], 'conditional_model_s':
                    ledger['conditional_model_s'], 'actual_neural_s': 0.,
                    'requested_context_rk4_ticks':
                    ledger['requested_context_rk4_ticks'],
                    'unknown_requested_context_rk4_ticks':
                    ledger['unknown_requested_context_rk4_ticks'],
                    'actual_physical_s': 0., 'wall_s': time.perf_counter()-began,
                    'candidate_adopted': False, 'F6_exposed': False,
                    'whole_goal_complete': False})
            except Exception as error:  # noqa: BLE001
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
    parser.add_argument('--out', '--output', dest='output', type=Path,
                        required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.source_review,
        args.source_review_sha, args.output)
