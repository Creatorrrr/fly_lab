"""E130 frozen paired LF comparison; parent dispatches this once after review."""

# ruff: noqa: BLE001  # primary errors and partial CUDA evidence must survive

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import banc_m3_precomputed_expm1 as candidate
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import profile_c_banc_m3_joint_fit_cost as profile

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = 'e520d479523eee39f61b96ba7605e396a0e076a4e44c7c06e425ca038d7ad5d1'
SOURCE_FILES = ('tools/banc_m3_precomputed_expm1.py',
                'tools/verify_c_banc_m3_precomputed_expm1.py',
                'tests/test_c_banc_precomputed_expm1.py')
ORDER = ('baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate')
EXACT = profile.EXACT
LOOSE = profile.NUMERIC_LOOSE
TIGHT = profile.NUMERIC_TIGHT


def save(path, value):
    fit.write_json(path, profile.snapshot(value))


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or profile.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E130 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if (freeze.get('schema') != 'flylab.m3.e130.precomputed-expm1-freeze.v1'
            or freeze.get('design_sha256') != profile.sha256(ROOT / freeze['design_path'])
            or freeze.get('E129_freeze_path') not in freeze.get('input_sha256', {})
            or freeze.get('reference_raw_path') not in freeze.get('input_sha256', {})
            or len(freeze.get('input_sha256', {})) != 7
            or len(freeze.get('reused_source_sha256', {})) != 5
            or freeze.get('evaluation_order') != list(ORDER)
            or freeze.get('original_evaluation_limit') != 6
            or (freeze.get('reference_evaluation_index'),
                freeze.get('reference_leg')) != (13, 0)
            or freeze.get('original_fit_calls') != 0
            or (freeze.get('leg_MN'), freeze.get('leg_muscles'),
                freeze.get('leg_edges')) != (69, 17, 273)
            or (freeze.get('original_context_ticks'),
                freeze.get('original_conditional_s'),
                freeze.get('original_scalar_MN_ticks'),
                freeze.get('original_edge_stage_ticks'),
                freeze.get('original_derivative_banks'),
                freeze.get('original_decoder_endpoints')) !=
                (720000, 180, 49680000, 786240000, 6, 36000)
            or (freeze.get('geometry_forward_rows'),
                freeze.get('geometry_backward_rows')) != (32838, 32838)
            or (freeze.get('synthetic_tiny_batches'),
                freeze.get('synthetic_tiny_context_ticks')) != (2, 40)
            or (freeze.get('internal_wall_limit_s'),
                freeze.get('external_wall_limit_s'),
                freeze.get('supervisor_limit_s')) != (25, 30, 35)
            or (freeze.get('prior_conservative_process_s'),
                freeze.get('allocated_conservative_end_s'),
                freeze.get('shared_process_limit_s')) != (1405, 1440, 1530)
            or (freeze.get('min_forward_pair_speedup'),
                freeze.get('min_median_eval_speedup')) != (1.25, 1.2)
            or (freeze.get('loose_numeric_abs_bound'),
                freeze.get('tight_numeric_abs_bound')) != (1e-8, 1e-10)):
        raise ValueError('E130 frozen contract changed')
    for key in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[key].items():
            profile.checked_path(relative, digest)
    e129_path = ROOT / freeze['E129_freeze_path']
    e129_review = ROOT / 'verification/banc-natural-gait-m3-20260919/parent-e129-final-source-review.json'
    e129, e128, _ = profile.reviewed_inputs(
        e129_path, profile.sha256(e129_path), e129_review,
        profile.sha256(e129_review))
    manifest_rel = ('verification/banc-natural-gait-m3-20260919/'
                    'e129-fixed-lf-profile-supervision/original-output-manifest.json')
    manifest = json.loads((ROOT / manifest_rel).read_text(encoding='utf-8'))
    if manifest.get('original_file_count') != 6 or len(manifest.get('original_files', {})) != 6:
        raise ValueError('E129 original output manifest count changed')
    original_dir = ROOT / 'verification/banc-natural-gait-m3-20260919/e129-fixed-lf-profile'
    for name, digest in manifest['original_files'].items():
        if name != Path(name).name or profile.sha256(original_dir / name) != digest:
            raise ValueError('E129 original output SHA mismatch: ' + name)
    if profile.sha256(review_path) != review_sha:
        raise ValueError('E130 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_precomputed_expm1_comparison') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E130 three-source review required')
    for relative, digest in review['source_sha256'].items():
        profile.checked_path(relative, digest)
    return freeze, e129, e128, review


def reference_vector(path):
    with np.load(path, allow_pickle=False) as z:
        delta, gain = z['delta'].copy(), z['gain'].copy()
        if (delta.shape != (273,) or gain.shape != (17,)
                or delta.dtype != np.float64 or gain.dtype != np.float64
                or not np.isfinite(delta).all() or not np.isfinite(gain).all()
                or np.array_equal(delta, np.zeros_like(delta))
                or np.array_equal(gain, np.zeros_like(gain))):
            raise ValueError('Frozen evaluation 013 must have finite nonzero delta/gain')
        return np.r_[delta, gain]


def compare_raw(actual_path, reference_path, *, old_reference=False):
    """Compare all shared fields; old raw13 may omit full response arrays."""
    report = {'actual_sha256': profile.sha256(actual_path),
              'reference_sha256': profile.sha256(reference_path),
              'old_reference': old_reference, 'fields': {}, 'accepted': False}
    with np.load(actual_path, allow_pickle=False) as a, np.load(
            reference_path, allow_pickle=False) as b:
        for key in (*EXACT, *LOOSE, *TIGHT):
            if key not in a or key not in b:
                if old_reference and key not in b:
                    report['fields'][key] = {'status': 'absent_in_old_reference'}
                    continue
                report['fields'][key] = {'status': 'missing'}
                continue
            left, right = a[key], b[key]
            if old_reference and key.startswith('context') and right.size == 0:
                report['fields'][key] = {'status': 'empty_in_old_reference'}
                continue
            if left.shape != right.shape or left.dtype != right.dtype:
                report['fields'][key] = {'status': 'shape_or_dtype_mismatch',
                                         'actual_shape': left.shape,
                                         'reference_shape': right.shape}
                continue
            if key in EXACT:
                same = bool(np.array_equal(left, right))
                report['fields'][key] = {'status': 'exact' if same else 'mismatch'}
            else:
                finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
                difference = float(np.max(np.abs(left - right))) if left.size else 0.0
                bound = 1e-10 if key in TIGHT else 1e-8
                report['fields'][key] = {'status': 'within_bound' if finite and difference <= bound
                                         else 'mismatch', 'finite': finite,
                                         'max_abs_difference': difference, 'bound': bound}
        report['accepted'] = all(x['status'] in ('exact', 'within_bound',
            'absent_in_old_reference', 'empty_in_old_reference')
            for x in report['fields'].values())
    return report


def complete_span(observation, phase, label):
    spans = [x for x in observation.cuda if x['phase'] == phase and
             x.get('kernel', x.get('label')) == label]
    if len(spans) != (3 if label == 'geometry.leg' else 1):
        return None
    if not all(x['returned'] and x['enqueued'] and x['end_recorded']
               and x['synchronized'] and x['elapsed_ms'] is not None
               and np.isfinite(x['elapsed_ms']) and x['elapsed_ms'] >= 0
               for x in spans):
        return None
    return sum(x['elapsed_ms'] for x in spans) / 1000


def known_progress(progress, *, ticks, conditional_s, decoder, bank):
    return (isinstance(progress, dict)
        and progress.get('completion_class') == 'known_complete'
        and progress.get('forward_completion_class') == 'known_complete'
        and progress.get('decoder_completion_class') == 'known_complete'
        and progress.get('bank_completion_class') == 'known_complete'
        and progress.get('derivative_bank_complete') is True
        and progress.get('completed_context_rk4_ticks') == ticks
        and progress.get('completed_conditional_model_s') == conditional_s
        and progress.get('completed_decoder_endpoint_ticks') == decoder
        and progress.get('completed_bank_edge_stage_ticks') == bank
        and progress.get('dynamic_upload_requested', {}).get('count') == 2
        and progress.get('dynamic_upload_enqueued', {}).get('count') == 2
        and progress.get('dynamic_upload_synchronized', {}).get('count') == 2
        and all(x.get('synchronized') is True for x in
                progress.get('dynamic_upload_events', []))
        and len(progress.get('dynamic_upload_events', [])) == 2)


def known_adapter(adapter, edges):
    if not isinstance(adapter, dict):
        return False
    events = adapter.get('events', [])
    if len(events) != 2 or [e.get('kind') for e in events] != [
            'prepare', 'joint_forward']:
        return False
    for event in events:
        if not (event.get('requested') is True and event.get('enqueued') is True
                and event.get('synchronized') is True
                and event.get('unknown_requested') is False
                and event.get('error') is None):
            return False
    if events[0].get('bytes') != edges * 8:
        return False
    return (adapter.get('gpu_temp_bytes_requested') == edges * 8
        and adapter.get('gpu_temp_bytes_allocated') == edges * 8
        and adapter.get('gpu_temp_bytes_peak') == edges * 8
        and all(adapter.get('counts', {}).get(kind) == {
            'requested': 1, 'enqueued': 1, 'synchronized': 1,
            'unknown_requested': 0} for kind in ('prepare', 'joint_forward')))


def known_metadata(metadata, calls):
    if not isinstance(metadata, dict):
        return False
    return (metadata.get('solve_calls') == calls
        and metadata.get('bank_calls') == calls
        and metadata.get('dynamic_upload', {}).get('requested', {}).get('count') == 2*calls
        and metadata.get('dynamic_upload', {}).get('enqueued', {}).get('count') == 2*calls
        and metadata.get('dynamic_upload', {}).get('synchronized', {}).get('count') == 2*calls
        and metadata.get('dynamic_upload', {}).get('unknown_requested', {}).get('count') == 0
        and metadata.get('fixed_upload', {}).get('unknown_requested', {}).get('count') == 0)


def result_from_raw(freeze, ledger, observation, comparisons):
    rows = ledger.get('original_evaluations', [])
    layout_ok = (len(rows) == 6
        and ledger.get('original_evaluations_requested') == 6
        and ledger.get('original_evaluations_completed') == 6
        and [(r.get('index'), r.get('backend'), r.get('phase')) for r in rows] == [
            (i + 1, name, f'original_{i + 1}_{name}')
            for i, name in enumerate(ORDER)]
        and all(r.get('status') == 'raw_saved' and r.get('raw_sha256')
                and r.get('raw_path') for r in rows))
    tiny = ledger.get('synthetic_tiny', [])
    tiny_ok = (len(tiny) == 2
        and ledger.get('synthetic_tiny_requested') == 2
        and ledger.get('synthetic_tiny_completed') == 2
        and [x.get('backend') for x in tiny] == ['baseline', 'candidate']
        and all(x.get('status') == 'completed' and x.get('raw_sha256')
            and known_progress(x.get('progress'), ticks=20, conditional_s=.005,
                decoder=1, bank=80)
            and known_metadata(x.get('engine_metadata'), 1) for x in tiny)
        and known_adapter(tiny[1].get('adapter'), 1))
    pairs = []
    for pair in range(3 if layout_ok else 0):
        pair_rows = rows[pair * 2:pair * 2 + 2]
        by_name = {row['backend']: row for row in pair_rows}
        baseline, contender = by_name['baseline'], by_name['candidate']
        b = complete_span(observation, baseline['phase'], 'joint_forward')
        c = complete_span(observation, contender['phase'], 'joint_forward')
        pair_wall = (baseline['evaluate_wall_s'] / contender['evaluate_wall_s']
                     if baseline.get('evaluate_wall_s') and contender.get('evaluate_wall_s')
                     else None)
        pairs.append({'pair': pair + 1, 'baseline_forward_cuda_s': b,
                      'candidate_forward_cuda_s': c,
                      'forward_speedup': b / c if b is not None and c else None,
                      'whole_eval_wall_speedup': pair_wall})
    forward_ok = all(p['forward_speedup'] is not None and
                     p['forward_speedup'] >= freeze['min_forward_pair_speedup']
                     for p in pairs)
    median = (float(np.median([p['whole_eval_wall_speedup'] for p in pairs]))
              if pairs and all(p['whole_eval_wall_speedup'] is not None
                               for p in pairs) else None)
    timings_complete = (layout_ok and all(
        complete_span(observation, row['phase'], label) is not None
        for row in rows for label in
        ('joint_forward', 'muscle_ema', 'edge_bank', 'geometry.leg',
         'torch.autograd.grad'))
        and all(sum(x['phase'] == row['phase'] for x in observation.cuda) == 7
                for row in rows))
    numerical = (len(comparisons) == 6
        and all(comparisons[i].get('accepted') is True and
                comparisons[i].get('old_reference') is bool(i % 2)
                for i in range(6)))
    adapter_ok = layout_ok and all(known_adapter(row.get('adapter'), 273)
        for row in rows if row['backend'] == 'candidate')
    progress_ok = layout_ok and all(
        isinstance(row.get('solve'), dict)
        and row['solve'].get('phase') == row['phase']
        and row.get('solve_span_count') == 1
        and row['solve'].get('returned') is True
        and known_progress(row['solve'].get('response_progress'),
            ticks=120000, conditional_s=30, decoder=6000, bank=131040000)
        and row.get('requested_context_ticks') == 120000
        and row.get('requested_derivative_banks') == 1
        and row.get('requested_edge_stage_ticks') == 131040000
        and known_metadata(row.get('engine_metadata'),
            sum(prior['backend'] == row['backend'] for prior in rows[:i + 1]))
        for i, row in enumerate(rows))
    accepted = (layout_ok and tiny_ok and numerical and timings_complete
                and adapter_ok and progress_ok
                and not observation.secondary
                and forward_ok and median is not None
                and median >= freeze['min_median_eval_speedup'])
    return {'schema': 'flylab.m3.e130.precomputed-expm1-result.v1',
            'numerical_comparison_accepted': numerical,
            'exact_six_evaluation_layout_complete': layout_ok,
            'two_tiny_setups_complete': tiny_ok,
            'timing_complete': timings_complete, 'candidate_accounting_complete': adapter_ok,
            'original_completion_accounting_complete': progress_ok,
            'pairs': pairs, 'median_whole_eval_wall_speedup': median,
            'accepted_local_one_point_performance': accepted,
            'interpretation': ('local_performance_adopt' if accepted else
                'diagnostic_only_incomplete' if not layout_ok or not tiny_ok or
                not numerical or not timings_complete or not adapter_ok or
                not progress_ok or observation.secondary
                else 'local_performance_reject'),
            'fit_calls': 0, 'body_calls': 0, 'F6': False}


def safe_finally(*actions):
    """Try all receipt actions without replacing an active primary failure."""
    active = sys.exc_info()[1]
    failures = []
    for label, action in actions:
        try:
            action()
        except BaseException as error:
            if active is not None:
                active.add_note('Secondary ' + label + ': ' + repr(error))
            else:
                failures.append((label, error))
    if failures:
        first = failures[0][1]
        for label, error in failures[1:]:
            first.add_note('Secondary ' + label + ': ' + repr(error))
        raise first


def partial_paths(directory, error, pattern='*partial-raw.npz'):
    files = []
    for path in sorted(directory.glob(pattern)):
        item = {'path': str(path)}
        try:
            item['sha256'] = profile.sha256(path)
        except BaseException as secondary:
            error.add_note('Secondary partial raw hash: ' + repr(secondary))
            item['hash_error'] = repr(secondary)
        files.append(item)
    return files


def write_output_manifest(out):
    manifest = {'schema': 'flylab.m3.e130.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): profile.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file()
            and path.name != 'output-manifest.json'}}
    save(out / 'output-manifest.json', manifest)


def finalize_artifacts(out, ledger, freeze, began, primary, secondary):
    """Manifest and final wall check precede any authoritative success return."""
    bound = freeze['internal_wall_limit_s'] if freeze is not None else None
    ledger['final_wall_limit_s'] = bound
    for pass_index in range(2):
        ledger['process_wall_s'] = time.perf_counter() - began
        try:
            save(out / 'advance-ledger.json', ledger)
        except BaseException as error:
            primary = secondary('final_ledger', error)
        try:
            write_output_manifest(out)
        except BaseException as error:
            primary = secondary('output_manifest', error)
        elapsed = time.perf_counter() - began
        if bound is not None and elapsed > bound:
            primary = secondary('post_manifest_timeout',
                TimeoutError('E130 internal wall limit exceeded after manifest'))
        if primary is not None:
            break
        ledger['post_manifest_wall_s'] = elapsed
        if pass_index == 1:
            return None
    # A failed or late manifest cannot leave a valid adoption receipt behind.
    ledger['status'] = 'failed'
    ledger['primary_error'] = {'type': type(primary).__name__,
        'message': str(primary), 'notes': list(getattr(primary, '__notes__', [])),
        'earlier_traceback': ledger.get('primary_error', {}).get('traceback'),
        'response_progress': getattr(primary, 'response_progress', None)}
    result = out / 'result.json'
    if result.exists():
        try:
            result.unlink()
        except BaseException as error:
            primary = secondary('invalid_result_removal', error)
            try:
                save(result, {'accepted_local_one_point_performance': False,
                              'status': 'invalid_finalization'})
            except BaseException as fallback:
                primary = secondary('invalid_result_overwrite', fallback)
    ledger['process_wall_s'] = time.perf_counter() - began
    try:
        save(out / 'advance-ledger.json', ledger)
    except BaseException as error:
        primary = secondary('failed_ledger', error)
    try:
        write_output_manifest(out)
    except BaseException as error:
        primary = secondary('failed_output_manifest', error)
    return primary


def run(freeze_path, freeze_sha, source_review, source_review_sha, out):
    began = time.perf_counter()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('One-shot E130 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e130.precomputed-expm1-ledger.v1',
              'pid': os.getpid(), 'freeze_sha256': freeze_sha,
              'source_review_sha256': source_review_sha,
              'status': 'preflight', 'original_evaluations': [],
              'original_evaluations_requested': 0,
              'original_evaluations_completed': 0,
              'synthetic_tiny_requested': 0, 'synthetic_tiny_completed': 0,
              'fit_calls': 0, 'body_calls': 0}
    save(out / 'advance-ledger.json', ledger)
    obs = profile.Observation()
    engines = {}
    limits = None
    freeze = None
    primary = None
    comparisons = []
    try:
        freeze, _, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, source_review, source_review_sha)
        save(out / 'source-review-snapshot.json', review)
        vector = reference_vector(ROOT / freeze['reference_raw_path'])
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version, 'cupy': cp.__version__,
            'torch': torch.__version__, 'gpu': gpu.decode() if isinstance(gpu, bytes)
            else str(gpu)}
        for name in ('baseline', 'candidate'):
            entry = {'backend': name, 'requested_context_ticks': 20,
                     'requested_conditional_s': .005, 'requested_bank_edge_stage_ticks': 80,
                     'status': 'requested'}
            ledger.setdefault('synthetic_tiny', []).append(entry)
            ledger['synthetic_tiny_requested'] += 1
            save(out / 'advance-ledger.json', ledger)
            tiny_start = time.perf_counter()
            tiny_engine = None
            try:
                tiny = fit.synthetic_tiny_setup_problem()
                tiny_engine = neural.CudaJointAdaptiveEngine(tiny)
                adapter_scope = (candidate.precomputed_expm1(tiny_engine) if name == 'candidate'
                                 else contextlib.nullcontext(None))
                with adapter_scope as adapter:
                    _, raw, timing = tiny_engine.solve(np.zeros(1), np.zeros(1),
                                                        derivative=True)
                np.savez(out / f'{name}-cuda-tiny-setup-raw.npz',
                         synthetic_stage_current=tiny['stage_current'],
                         synthetic_source_rate=tiny['source_rate'], **raw)
                entry.update(status='raw_saved', raw_sha256=profile.sha256(
                    out / f'{name}-cuda-tiny-setup-raw.npz'), progress=timing['progress'],
                    engine_metadata=profile.snapshot(tiny_engine.metadata()),
                    adapter=adapter.snapshot() if adapter else None)
                save(out / 'advance-ledger.json', ledger)
                if not all(np.isfinite(raw[key]).all() for key in neural.FIELDS):
                    raise ValueError('E130 tiny setup nonfinite')
                entry['status'] = 'completed'
                ledger['synthetic_tiny_completed'] += 1
            except BaseException as error:
                entry.update(status='failed', error=repr(error),
                             progress=getattr(error, 'response_progress', None),
                             adapter=getattr(error, 'precomputed_expm1_progress', None),
                             fixed_upload_progress=getattr(error, 'fixed_upload_progress', None))
                if tiny_engine is not None:
                    try:
                        entry['engine_metadata'] = profile.snapshot(tiny_engine.metadata())
                    except BaseException as secondary:
                        error.add_note('Secondary tiny metadata: ' + repr(secondary))
                partial = getattr(error, 'partial_response', None)
                if partial:
                    try:
                        np.savez(out / f'{name}-cuda-tiny-setup-partial-raw.npz', **partial)
                    except Exception as secondary:
                        error.add_note('Secondary tiny partial save: ' + repr(secondary))
                entry['partial_raw_files'] = partial_paths(
                    out, error, f'{name}-cuda-tiny-setup-partial-raw.npz')
                raise
            finally:
                entry['wall_s'] = time.perf_counter() - tiny_start
                safe_finally(('tiny_ledger', lambda: save(
                    out / 'advance-ledger.json', ledger)))
        fixture = obs.timed('frozen_fixture_load', fit.frozen_problem)(e128)
        def load_geometry():
            geometry = TorchFootGeometry.load(ROOT / e128['geometry_path'], device='cuda')
            torch.cuda.synchronize(geometry.device)
            return geometry
        geometry = obs.timed('geometry_fixed_upload', load_geometry)()
        if geometry.geometry_hash != e128['geometry_hash']:
            raise ValueError('Frozen E128 geometry hash changed')
        ledger['geometry_fixed_upload_bytes'] = sum(int(geometry.arrays[key].nbytes)
            for key in ('axes', 'anchors', 'tips', 'reference_angles'))
        motors, muscles, edges = (np.flatnonzero(fixture[key][0]) for key in
            ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask'))
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('Frozen LF subset cardinality changed')
        for name in ('baseline', 'candidate'):
            try:
                engine = obs.timed(name + '_engine_fixed_upload',
                                   neural.CudaJointAdaptiveEngine)(
                                       fixture, motors, muscles, edges)
            except BaseException as error:
                failure = {'error': repr(error)}
                ledger[name + '_constructor_failure'] = failure
                for key in ('fixed_upload_progress', 'fixed_upload_by_key'):
                    try:
                        failure[key] = profile.snapshot(getattr(error, key, None))
                    except BaseException as secondary:
                        error.add_note('Secondary constructor ' + key + ': ' +
                                       repr(secondary))
                safe_finally(('constructor_ledger', lambda: save(
                    out / 'advance-ledger.json', ledger)))
                raise
            engines[name] = engine
            ledger[name + '_engine_before'] = profile.snapshot(engine.metadata())
            with np.load(ROOT / freeze['reference_raw_path'], allow_pickle=False) as ref:
                for key, expected in (('motor_indices', engine.problem['motor_indices']),
                    ('muscle_indices', engine.problem['muscle_indices']),
                    ('edge_ids', engine.problem['edge_ids']),
                    ('parameter_muscle_map', neural.activation_parameter_map(engine.problem))):
                    if not np.array_equal(ref[key], expected):
                        raise ValueError('Frozen LF ID/mapping changed: ' + key)
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        if time.perf_counter() - began >= freeze['internal_wall_limit_s'] - 5:
            raise TimeoutError('E130 preflight exhausted internal reserve')
        for index, name in enumerate(ORDER):
            engine = engines[name]
            phase = f'original_{index + 1}_{name}'
            row = {'index': index + 1, 'backend': name, 'phase': phase,
                   'status': 'requested', 'requested_context_ticks': 120000,
                   'requested_conditional_s': 30,
                   'requested_scalar_MN_ticks': 8280000,
                   'requested_edge_stage_ticks': 131040000,
                   'requested_derivative_banks': 1,
                   'requested_decoder_endpoints': 6000}
            ledger['original_evaluations'].append(row)
            ledger['original_evaluations_requested'] += 1
            save(out / 'advance-ledger.json', ledger)
            directory = out / f'original-{index + 1:02d}-{name}'
            directory.mkdir()
            obs.phase = phase
            call_start = time.perf_counter()
            adapter = None
            try:
                adapter_scope = (candidate.precomputed_expm1(engine) if name == 'candidate'
                                 else contextlib.nullcontext(None))
                with adapter_scope as adapter, profile.observed_callables(obs, engine, geometry):
                    record = obs.timed('evaluate_leg', fit.evaluate_leg)(
                        engine, fixture, geometry, 0, vector, 1, directory)
                row.update(status='raw_saved', raw_path=record['raw_path'],
                           raw_sha256=record['raw_sha256'], solve_wall_s=record['wall_s'],
                           adapter=adapter.snapshot() if adapter else None)
                ledger['original_evaluations_completed'] += 1
            except BaseException as error:
                row.update(status='failed', error=repr(error),
                           response_progress=getattr(error, 'response_progress', None),
                           adapter=(adapter.snapshot() if adapter else
                                    getattr(error, 'precomputed_expm1_progress', None)),
                           partial_response_keys=list(getattr(error, 'partial_response', {})),
                           partial_raw_files=partial_paths(directory, error))
                raise
            finally:
                row['evaluate_wall_s'] = time.perf_counter() - call_start
                matches = [x for x in obs.solve if x['phase'] == phase]
                row['solve_span_count'] = len(matches)
                safe_finally(
                    ('engine_metadata', lambda row=row, engine=engine: row.update(engine_metadata=
                        profile.snapshot(engine.metadata()))),
                    ('current_phase_solve', lambda row=row, matches=matches: row.update(solve=
                        profile.snapshot(matches[0] if len(matches) == 1 else None))),
                    ('evaluation_ledger', lambda: save(out / 'advance-ledger.json', ledger)))
            if time.perf_counter() - began > freeze['internal_wall_limit_s'] - 2:
                raise TimeoutError('E130 internal reserve exhausted after raw save')
        ledger['status'] = 'raw_saved'
        save(out / 'advance-ledger.json', ledger)
        for pair in range(3):
            rows = ledger['original_evaluations'][2 * pair:2 * pair + 2]
            actual = next(x for x in rows if x['backend'] == 'candidate')
            baseline = next(x for x in rows if x['backend'] == 'baseline')
            comparison = compare_raw(actual['raw_path'], baseline['raw_path'])
            comparisons.append(comparison)
            save(out / f'pair-{pair + 1}-raw-comparison.json', comparison)
            old = compare_raw(baseline['raw_path'], ROOT / freeze['reference_raw_path'],
                              old_reference=True)
            comparisons.append(old)
            save(out / f'pair-{pair + 1}-old-reference-comparison.json', old)
        ledger['status'] = 'compared'
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(),
            'response_progress': getattr(error, 'response_progress', None),
            'precomputed_expm1_progress': getattr(error, 'precomputed_expm1_progress', None),
            'notes': list(getattr(error, '__notes__', []))}
    finally:
        def secondary(stage, error):
            nonlocal primary
            obs.secondary.append({'stage': stage, 'error': repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note('Secondary ' + stage + ': ' + repr(error))
            return primary
        try:
            obs.finish_cuda()
        except Exception as error:
            secondary('finish_cuda', error)
        ledger['host_spans'] = obs.host
        ledger['cuda_spans'] = obs.cuda
        ledger['solve_spans'] = obs.solve
        ledger['secondary_errors'] = obs.secondary
        for name, engine in engines.items():
            try:
                ledger[name + '_engine_after'] = profile.snapshot(engine.metadata())
            except Exception as error:
                secondary(name + '_metadata', error)
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:
                secondary('threadpool_restore', error)
        if primary is None:
            try:
                result = result_from_raw(freeze, ledger, obs, comparisons)
                if time.perf_counter() - began >= freeze['internal_wall_limit_s'] - 1:
                    raise TimeoutError('E130 result reserve exhausted')
                save(out / 'result.json', result)
                ledger['status'] = ('completed' if result['accepted_local_one_point_performance']
                                    else 'comparison_completed_performance_rejected')
            except BaseException as error:
                secondary('result_finalization', error)
        primary = finalize_artifacts(out, ledger, freeze, began, primary, secondary)
    if primary is not None:
        raise primary
    return out / 'result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha', 'source-review', 'source-review-sha', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.source_review, args.source_review_sha,
        args.out)


if __name__ == '__main__':
    main()
