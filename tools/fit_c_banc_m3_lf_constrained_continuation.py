"""E132 bounded LF constrained continuation from the E131 feasible vector.

The parent dispatches the single original process only after its source review.
"""

# ruff: noqa: BLE001  # retain primary CUDA/persistence failures and partial evidence

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import banc_m3_precomputed_expm1 as adapter_module
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import fit_c_banc_m3_lf_feasibility as e131
from tools import profile_c_banc_m3_joint_fit_cost as profile
from tools import verify_c_banc_m3_precomputed_expm1 as e130

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = 'c72514c370261a1a9ed91677b7709972abf4f1692b0ee2780e54aed2900f0725'
SOURCE_FILES = ('tools/fit_c_banc_m3_lf_constrained_continuation.py',
                'tests/test_c_banc_lf_constrained_continuation.py')
EXACT_RAW = frozenset(('delta', 'gain', 'edge_ids', 'motor_indices',
    'muscle_indices', 'parameter_muscle_map', 'branch_bits'))
# This is execution accounting, accumulated over different numbers of calls.
PROVENANCE_RAW = frozenset(('transfer_cost_json',))


def save(path, value):
    fit.write_json(path, profile.snapshot(value))


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or profile.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E132 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    required = {'schema': 'flylab.m3.e132.lf-constrained-continuation-freeze.v1',
        'leg': 0, 'start_evaluation_index': 11,
        'objective': 'E128 max(losses)+delta/gain prior, optimized with original epigraph SLSQP',
        'constraint_count': 4761, 'solver': 'SLSQP',
        'solver_options': {'maxiter': 100, 'ftol': 1e-9, 'disp': False},
        'optimizer_cap': fit.OPTIMIZER_CAP, 'selector_cap': fit.SELECTOR_CAP,
        'proposal_objective_ratio_limit': .99, 'max_unique_evaluations': 40,
        'fit_wall_limit_s': 55, 'leg_reserve_s': fit.LEG_RESERVE_S,
        'process_reserve_s': fit.PROCESS_RESERVE_S,
        'internal_wall_limit_s': 64, 'external_wall_limit_s': 69,
        'supervisor_limit_s': 73, 'prior_conservative_process_s': 1450,
        'allocated_conservative_end_s': 1523, 'shared_process_limit_s': 1530,
        'synthetic_tiny_batches': 1, 'synthetic_tiny_context_ticks': 20,
        'stop_at_first_selector_feasible': False, 'full_BANC_calls': 0,
        'body_calls': 0, 'F6_exposed': False,
        'whole_candidate_adoption': False, 'original_processes': 1,
        'retries': 0}
    work = {'context_ticks': 4800000, 'conditional_s': 1200,
        'scalar_MN_ticks': 331200000, 'edge_stage_ticks': 5241600000,
        'derivative_banks': 40, 'decoder_endpoints': 240000,
        'geometry_forward_rows': 218920, 'geometry_backward_rows': 218920}
    if (any(freeze.get(k) != v for k, v in required.items())
            or freeze.get('dimensions') != {'contexts': 3, 'MN': 69,
                'muscles': 17, 'edges': 273, 'parameters': 290}
            or freeze.get('maximum_original_work') != work
            or len(freeze.get('input_sha256', {})) != 9
            or len(freeze.get('reused_source_sha256', {})) != 2
            or freeze.get('design_sha256') != profile.sha256(ROOT / freeze['design_path'])
            or any(freeze.get(k) not in freeze['input_sha256'] for k in
                ('E131_freeze_path', 'E131_source_review_path', 'start_raw_path',
                 'start_proposal_path', 'zero_raw_path'))):
        raise ValueError('E132 frozen contract changed')
    for key in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[key].items():
            profile.checked_path(relative, digest)
    _, e128, _ = e131.reviewed_inputs(
        ROOT / freeze['E131_freeze_path'],
        freeze['input_sha256'][freeze['E131_freeze_path']],
        ROOT / freeze['E131_source_review_path'],
        freeze['input_sha256'][freeze['E131_source_review_path']])
    manifest_path = ROOT / ('verification/banc-natural-gait-m3-20260919/'
        'e131-lf-feasibility-supervision/original-output-manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest.get('original_files', {})
    if manifest.get('original_file_count') != 19 or len(files) != 19:
        raise ValueError('E131 original manifest count changed')
    original_dir = ROOT / 'verification/banc-natural-gait-m3-20260919/e131-lf-feasibility'
    for relative, digest in files.items():
        name = Path(relative)
        if (name.is_absolute() or '..' in name.parts
                or profile.sha256(original_dir / name) != digest):
            raise ValueError('E131 original output SHA mismatch: ' + relative)
    if profile.sha256(review_path) != review_sha:
        raise ValueError('E132 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_lf_constrained_continuation') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E132 two-source review required')
    for relative, digest in review['source_sha256'].items():
        profile.checked_path(relative, digest)
    return freeze, e128, review


def start_vector(freeze):
    """Require the E131 selected float64 proposal to equal its selected raw."""
    with (np.load(ROOT / freeze['start_raw_path'], allow_pickle=False) as raw,
          np.load(ROOT / freeze['start_proposal_path'], allow_pickle=False) as proposal):
        if (int(proposal['original_evaluation_index']) != 11
                or str(proposal['source_raw_sha256']) !=
                    freeze['input_sha256'][freeze['start_raw_path']]
                or any(raw[k].dtype != np.float64 or proposal[k].dtype != np.float64
                    or not np.array_equal(raw[k], proposal[k])
                    for k in ('delta', 'gain'))):
            raise ValueError('E131 selected proposal differs from frozen raw11')
        vector = np.r_[proposal['delta'], proposal['gain']]
    if vector.shape != (290,) or not np.isfinite(vector).all():
        raise ValueError('E132 start must be finite float64 LF290')
    return vector


def compare_full_raw(actual_path, reference_path):
    """Compare every physical saved field; execution transfer totals are separate."""
    report = {'actual_sha256': profile.sha256(actual_path),
              'reference_sha256': profile.sha256(reference_path),
              'fields': {}, 'excluded_provenance': sorted(PROVENANCE_RAW),
              'accepted': False}
    with np.load(actual_path, allow_pickle=False) as actual, np.load(
            reference_path, allow_pickle=False) as reference:
        names = set(actual.files) | set(reference.files)
        report['provenance_present'] = all(
            key in actual and key in reference for key in PROVENANCE_RAW)
        for key in sorted(names - PROVENANCE_RAW):
            if key not in actual or key not in reference:
                report['fields'][key] = {'status': 'missing'}
                continue
            a, b = actual[key], reference[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                report['fields'][key] = {'status': 'shape_or_dtype_mismatch'}
            elif key in EXACT_RAW or not np.issubdtype(a.dtype, np.floating):
                report['fields'][key] = {'status': 'exact' if np.array_equal(a, b)
                                         else 'mismatch'}
            else:
                bound = 1e-10 if key in ('losses', 'parts', 'ratios') else 1e-8
                difference = float(np.max(np.abs(a - b))) if a.size else 0.
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                report['fields'][key] = {'status': 'within_bound' if finite and
                    difference <= bound else 'mismatch',
                    'max_abs_difference': difference, 'bound': bound}
        report['accepted'] = bool(report['provenance_present']
            and names - PROVENANCE_RAW and all(
            item['status'] in ('exact', 'within_bound')
            for item in report['fields'].values()))
    return report


def original_bounds(fixture, edge_indices):
    return e131.original_bounds(fixture, edge_indices)


def run_optimizer(cache, start, accounting=None, *, clock=time.perf_counter):
    """Replay the feasible start before requesting unchanged E128 SLSQP."""
    if accounting is None:
        accounting = {}
    accounting.update(status='not_requested', requested=0, returned=0,
                      bounded_stop=0, failed_unknown=0)
    solver = None
    stop = 'scipy_finished'
    began = clock()
    try:
        initial, _ = cache.get(np.r_[start, 0.])
        if not fit.selector_feasible(initial['ratios']):
            raise ValueError('E132 initial replay is not selector-feasible')
        x0 = np.r_[start, float(max(initial['losses']) + .01)]
        accounting.update(status='requested', requested=1)
        try:
            solver = fit.solve_one_leg(cache, [tuple(x) for x in cache.bounds], x0)
        except fit.FitLimit as error:
            accounting.update(status='bounded_stop', bounded_stop=1)
            stop = str(error)
        except BaseException as error:
            accounting.update(status='failed_unknown', failed_unknown=1,
                              error=repr(error))
            raise
        else:
            accounting.update(status='returned', returned=1)
    except fit.FitLimit as error:
        stop = str(error)
    except BaseException as error:
        if accounting['requested'] == 0:
            accounting['initial_replay_error'] = repr(error)
        raise
    finally:
        accounting.update(elapsed_wall_s=clock() - began,
                          cache_started=cache.started,
                          cache_completed=cache.completed,
                          cache_hits=cache.hits)
    return solver, stop


def _row_valid(row):
    if not e131._complete_evaluation(row):
        return False
    try:
        ratios = np.asarray(row['ratios'], np.float64)
        losses = np.asarray(row['losses'], np.float64)
        parts = np.asarray(row['parts'], np.float64)
        prior = float(row['prior'])
        objective = float(row['true_worst_objective'])
        wall = float(row['evaluate_wall_s'])
        return (ratios.shape == (3, 4) and losses.shape == (3,)
            and parts.shape == (3, 4) and np.isfinite(ratios).all()
            and np.isfinite(losses).all() and np.isfinite(parts).all()
            and np.isfinite([prior, objective, wall]).all() and wall >= 0
            and np.all(ratios >= 0) and prior >= 0
            and np.isclose(row['max_ratio'], np.max(ratios), rtol=0, atol=1e-12)
            and np.isclose(objective, max(losses) + prior, rtol=0, atol=1e-10)
            and row.get('selector_feasible') is fit.selector_feasible(ratios))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _parts(path):
    with np.load(path, allow_pickle=False) as raw:
        return np.asarray(raw['parts'], np.float64).tolist()


def result_from_raw(freeze, ledger, comparison, stop_reason, cache):
    """Fail closed on a partial call, unknown work or inconsistent selection."""
    rows = ledger.get('original_evaluations', [])
    if not isinstance(rows, list):
        rows = []
    complete = (1 <= len(rows) <= 40
        and type(ledger.get('original_evaluations_requested')) is int
        and type(ledger.get('original_evaluations_completed')) is int
        and len(rows) == ledger['original_evaluations_requested']
        == ledger['original_evaluations_completed']
        and all(isinstance(row, dict) and type(row.get('index')) is int
            and row['index'] == i and row.get('phase') == f'original_{i}_candidate'
            and row.get('backend') == 'candidate' and _row_valid(row)
            for i, row in enumerate(rows, 1)))
    compared = bool(complete and isinstance(comparison, dict)
        and comparison.get('accepted') is True
        and comparison.get('reference_sha256') == freeze.get(
            'input_sha256', {}).get(freeze.get('start_raw_path'))
        and comparison.get('actual_sha256') == rows[0]['raw_sha256'])
    tiny = ledger.get('synthetic_tiny', [])
    try:
        tiny_ok = (len(tiny) == ledger.get('synthetic_tiny_requested')
            == ledger.get('synthetic_tiny_completed') == 1
            and tiny[0].get('status') == 'completed' and tiny[0].get('raw_sha256')
            and e130.known_progress(tiny[0].get('progress'), ticks=20,
                conditional_s=.005, decoder=1, bank=80)
            and e130.known_adapter(tiny[0].get('adapter'), 1)
            and e130.known_metadata(tiny[0].get('engine_metadata'), 1))
    except (TypeError, KeyError, ValueError, OverflowError):
        tiny_ok = False
    spans = ledger.get('cuda_spans', [])
    try:
        timing_ok = bool(complete and all(isinstance(x, dict) for x in spans)
            and all(e130.complete_span(
                type('Observation', (), {'cuda': spans})(), row['phase'], label)
                is not None for row in rows for label in
                ('joint_forward', 'muscle_ema', 'edge_bank', 'geometry.leg',
                 'torch.autograd.grad'))
            and all(sum(x.get('phase') == row['phase'] for x in spans) == 7
                    for row in rows))
    except (TypeError, KeyError, ValueError, OverflowError):
        timing_ok = False
    cache_ok = bool(cache is not None and type(cache.started) is int
        and type(cache.completed) is int and type(cache.hits) is int
        and cache.hits >= 0 and isinstance(cache.cache, dict)
        and cache.started == cache.completed == len(rows) == len(cache.cache)
        and ledger.get('cache_started') == cache.started
        and ledger.get('cache_completed') == cache.completed
        and ledger.get('cache_hits') == cache.hits)
    optimizer = ledger.get('optimizer', {})
    if not isinstance(optimizer, dict):
        optimizer = {}
    counts = tuple(optimizer.get(k) for k in
        ('requested', 'returned', 'bounded_stop', 'failed_unknown'))
    optimizer_ok = bool(ledger.get('fit_calls') == 1 and counts in
        ((1, 1, 0, 0), (1, 0, 1, 0)) and
        ((counts == (1, 1, 0, 0) and optimizer.get('status') == 'returned'
          and stop_reason == 'scipy_finished') or
         (counts == (1, 0, 1, 0) and optimizer.get('status') == 'bounded_stop'
          and stop_reason in ('per_leg_unique_or_wall_limit',
                              'whole_fit_wall_limit'))))
    try:
        wall = float(ledger['fit_wall_s'])
        budget_ok = bool(np.isfinite(wall) and 0 <= wall <=
                         freeze['fit_wall_limit_s'])
        if not np.isfinite(wall):
            wall = None
    except (KeyError, ValueError, TypeError, OverflowError):
        wall, budget_ok = None, False
    start_ok = bool(complete and rows[0]['selector_feasible'] is True)
    any_row, selected = fit.choose_records(rows) if complete else (None, None)
    best_any_index = any_row['index'] if any_row else None
    selected_index = selected['index'] if selected else None
    selection_ok = bool(start_ok and selected is not None and
        selected_index == min((r['index'] for r in rows if
            r['selector_feasible'] and r['true_worst_objective'] ==
            selected['true_worst_objective']), default=None))
    evidence = bool(complete and compared and tiny_ok and timing_ok and cache_ok
        and optimizer_ok and budget_ok and selection_ok
        and not ledger.get('secondary_errors'))
    start_objective = rows[0]['true_worst_objective'] if complete else None
    chosen_objective = selected['true_worst_objective'] if selected else None
    ratio = (chosen_objective / start_objective if start_objective and
        chosen_objective is not None else None)
    accepted = bool(evidence and ratio is not None and
        ratio <= freeze['proposal_objective_ratio_limit']
        and selected_index != 1)
    if not evidence:
        status = 'invalid_budget' if not budget_ok else 'invalid_evidence'
        interpretation = 'invalid_or_incomplete'
    else:
        status = 'completed_bounded_fit'
        interpretation = ('local_objective_improvement_proposal' if accepted else
                          'bounded_local_improvement_below_one_percent')
    result = {'schema': 'flylab.m3.e132.lf-constrained-continuation-result.v1',
        'status': status, 'interpretation': interpretation,
        'new_lf_proposal_accepted': accepted,
        'start_objective': start_objective, 'selected_index': selected_index,
        'selected_objective': chosen_objective,
        'selected_objective_over_start': ratio,
        'best_any_index': best_any_index,
        'best_any_objective': (any_row['true_worst_objective'] if any_row else None),
        'stop_reason': stop_reason,
        'original_evaluations_requested': ledger.get('original_evaluations_requested'),
        'original_evaluations_completed': ledger.get('original_evaluations_completed'),
        'cache_hits': getattr(cache, 'hits', None),
        'fit_calls': ledger.get('fit_calls'), 'fit_wall_s': wall,
        'body_calls': 0, 'F6': False, 'F6_exposed': False,
        'candidate_adopted': False, 'whole_candidate_adopted': False,
        'whole_goal_complete': False, 'first_reference_comparison': compared,
        'synthetic_tiny_complete': bool(tiny_ok), 'timing_complete': timing_ok,
        'original_completion_accounting_complete': complete,
        'cache_accounting_complete': cache_ok,
        'optimizer_accounting_complete': optimizer_ok,
        'selection_consistent': selection_ok, 'fit_budget_complete': budget_ok,
        'source_review_sha256': ledger.get('source_review_sha256'),
        'freeze_sha256': ledger.get('freeze_sha256')}
    if evidence:
        zero_parts = _parts(ROOT / freeze['zero_raw_path'])
        start_parts = _parts(ROOT / freeze['start_raw_path'])
        selected_parts = _parts(selected['raw_path'])
        result['normalized_loss_parts'] = {'zero': zero_parts, 'start': start_parts,
            'selected': selected_parts,
            'selected_minus_zero': (np.asarray(selected_parts) - zero_parts).tolist(),
            'selected_minus_start': (np.asarray(selected_parts) - start_parts).tolist()}
    return result


def write_output_manifest(out):
    save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e132.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): profile.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file()
            and path.name != 'output-manifest.json'}})


def finalize_artifacts(out, ledger, freeze, began, primary, secondary):
    """A late manifest or timeout must never leave a success receipt."""
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
            primary = secondary('post_manifest_timeout', TimeoutError(
                'E132 internal wall limit exceeded after manifest'))
        if primary is not None:
            break
        ledger['post_manifest_wall_s'] = elapsed
        if pass_index == 1:
            return None
    ledger['status'] = 'failed'
    ledger['primary_error'] = {'type': type(primary).__name__,
        'message': str(primary), 'notes': list(getattr(primary, '__notes__', [])),
        'earlier_traceback': ledger.get('primary_error', {}).get('traceback')}
    result_path = out / 'result.json'
    if result_path.exists():
        try:
            result_path.unlink()
        except BaseException as error:
            primary = secondary('invalid_result_removal', error)
            try:
                save(result_path, {'status': 'invalid_finalization',
                                   'new_lf_proposal_accepted': False})
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
        raise FileExistsError('One-shot E132 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e132.lf-constrained-continuation-ledger.v1',
        'pid': os.getpid(), 'freeze_sha256': freeze_sha,
        'source_review_sha256': source_review_sha, 'status': 'preflight',
        'original_evaluations': [], 'original_evaluations_requested': 0,
        'original_evaluations_completed': 0, 'synthetic_tiny': [],
        'synthetic_tiny_requested': 0, 'synthetic_tiny_completed': 0,
        'fit_calls': 0, 'body_calls': 0, 'F6': False, 'F6_exposed': False,
        'candidate_adopted': False, 'whole_candidate_adopted': False}
    save(out / 'advance-ledger.json', ledger)
    obs = profile.Observation()
    freeze = limits = engine = cache = None
    primary = solver = stop_reason = comparison = None
    try:
        freeze, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, source_review, source_review_sha)
        save(out / 'source-review-snapshot.json', review)
        vector = start_vector(freeze)
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version, 'numpy': np.__version__,
            'cupy': cp.__version__, 'torch': torch.__version__,
            'gpu': gpu.decode() if isinstance(gpu, bytes) else str(gpu)}
        e131._run_tiny(out, ledger)
        fixture = obs.timed('frozen_fixture_load', fit.frozen_problem)(e128)

        def load_geometry():
            value = TorchFootGeometry.load(ROOT / e128['geometry_path'], device='cuda')
            torch.cuda.synchronize(value.device)
            return value

        geometry = obs.timed('geometry_fixed_upload', load_geometry)()
        if geometry.geometry_hash != e128['geometry_hash']:
            raise ValueError('Frozen E128 geometry hash changed')
        ledger['geometry_fixed_upload_bytes'] = sum(int(geometry.arrays[k].nbytes)
            for k in ('axes', 'anchors', 'tips', 'reference_angles'))
        motors, muscles, edges = (np.flatnonzero(fixture[k][0]) for k in
            ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask'))
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('Frozen LF cardinality changed')
        bounds = original_bounds(fixture, edges)
        if np.any(vector < bounds[:, 0]) or np.any(vector > bounds[:, 1]):
            raise ValueError('Frozen E131 start outside E128 bounds')
        try:
            engine = obs.timed('lf_engine_fixed_upload',
                neural.CudaJointAdaptiveEngine)(fixture, motors, muscles, edges)
        except BaseException as error:
            ledger['constructor_failure'] = {'error': repr(error),
                'fixed_upload_progress': getattr(error, 'fixed_upload_progress', None),
                'fixed_upload_by_key': getattr(error, 'fixed_upload_by_key', None)}
            e130.safe_finally(('constructor_ledger', lambda: save(
                out / 'advance-ledger.json', ledger)))
            raise
        ledger['engine_before'] = profile.snapshot(engine.metadata())
        with np.load(ROOT / freeze['start_raw_path'], allow_pickle=False) as ref:
            for key, expected in (('motor_indices', engine.problem['motor_indices']),
                    ('muscle_indices', engine.problem['muscle_indices']),
                    ('edge_ids', engine.problem['edge_ids']),
                    ('parameter_muscle_map',
                     neural.activation_parameter_map(engine.problem))):
                if not np.array_equal(ref[key], expected):
                    raise ValueError('Frozen LF ID/mapping changed: ' + key)
        ledger['status'] = 'ready'
        save(out / 'advance-ledger.json', ledger)
        fit_began = time.perf_counter()
        fit_deadline = fit_began + freeze['fit_wall_limit_s']
        internal_deadline = began + freeze['internal_wall_limit_s']
        ledger['optimizer'] = {'status': 'not_requested', 'requested': 0,
            'returned': 0, 'bounded_stop': 0, 'failed_unknown': 0}
        ledger['fit_wall_s'] = 0.
        save(out / 'advance-ledger.json', ledger)

        def evaluate(x, index):
            phase = f'original_{index}_candidate'
            row = {'index': index, 'phase': phase, 'backend': 'candidate',
                'raw_index': 1, 'status': 'requested',
                'requested_context_ticks': 120000,
                'requested_conditional_s': 30,
                'requested_scalar_MN_ticks': 8280000,
                'requested_edge_stage_ticks': 131040000,
                'requested_derivative_banks': 1,
                'requested_decoder_endpoints': 6000}
            ledger['original_evaluations'].append(row)
            ledger['original_evaluations_requested'] += 1
            save(out / 'advance-ledger.json', ledger)
            directory = out / f'evaluation-{index:03d}'
            directory.mkdir()
            obs.phase = phase
            call_start = time.perf_counter()
            adapter = None
            try:
                with (adapter_module.precomputed_expm1(engine) as adapter,
                      profile.observed_callables(obs, engine, geometry)):
                    record = obs.timed('evaluate_leg', fit.evaluate_leg)(
                        engine, fixture, geometry, 0, x, 1, directory)
                row.update(status='raw_saved', raw_path=record['raw_path'],
                    raw_sha256=record['raw_sha256'], solve_wall_s=record['wall_s'],
                    adapter=adapter.snapshot())
                ledger['original_evaluations_completed'] += 1
                save(out / 'advance-ledger.json', ledger)
                if index == 1:
                    nonlocal comparison
                    comparison = compare_full_raw(
                        record['raw_path'], ROOT / freeze['start_raw_path'])
                    save(out / 'first-reference-comparison.json', comparison)
                    ledger['first_reference_comparison'] = comparison['accepted']
                    save(out / 'advance-ledger.json', ledger)
                    if not comparison['accepted']:
                        raise ValueError('E132 initial raw failed E131 full reference')
                row.update(losses=record['losses'], parts=record['parts'],
                    prior=record['prior'],
                    true_worst_objective=record['true_worst_objective'],
                    max_ratio=float(np.max(record['ratios'])),
                    ratios=record['ratios'],
                    selector_feasible=fit.selector_feasible(record['ratios']))
                save(out / 'advance-ledger.json', ledger)
                if index == 1 and not row['selector_feasible']:
                    raise ValueError('E132 initial replay violates selector cap')
                return record
            except BaseException as error:
                row.update(status='failed', error=repr(error),
                    response_progress=getattr(error, 'response_progress', None),
                    adapter=(adapter.snapshot() if adapter else
                        getattr(error, 'precomputed_expm1_progress', None)),
                    partial_response_keys=list(getattr(error, 'partial_response', {})),
                    partial_raw_files=e130.partial_paths(directory, error))
                raise
            finally:
                row['evaluate_wall_s'] = time.perf_counter() - call_start
                matches = [item for item in obs.solve if item['phase'] == phase]
                row['solve_span_count'] = len(matches)
                e130.safe_finally(
                    ('engine_metadata', lambda: row.update(
                        engine_metadata=profile.snapshot(engine.metadata()))),
                    ('current_phase_solve', lambda: row.update(solve=profile.snapshot(
                        matches[0] if len(matches) == 1 else None))),
                    ('evaluation_ledger', lambda: save(out / 'advance-ledger.json', ledger)))

        cache = fit.EvaluationCache(evaluate, 290, lambda *_: None,
            max_calls=freeze['max_unique_evaluations'], deadline=fit_deadline,
            total_deadline=internal_deadline)
        cache.bounds = bounds
        try:
            solver, stop_reason = run_optimizer(cache, vector, ledger['optimizer'])
        finally:
            ledger['fit_wall_s'] = time.perf_counter() - fit_began
            ledger['fit_calls'] = ledger['optimizer']['requested']
            ledger['cache_started'] = cache.started
            ledger['cache_completed'] = cache.completed
            ledger['cache_hits'] = cache.hits
            e130.safe_finally(('fit_progress_ledger', lambda: save(
                out / 'advance-ledger.json', ledger)))
        ledger['solver'] = None if solver is None else {
            'success': bool(solver.success), 'status': int(solver.status),
            'message': str(solver.message), 'nit': int(solver.nit),
            'nfev': int(solver.nfev), 'fun': float(solver.fun)}
        if ledger['fit_wall_s'] > freeze['fit_wall_limit_s']:
            stop_reason = 'fit_wall_exceeded'
        if time.perf_counter() >= internal_deadline - 1:
            raise TimeoutError('E132 result reserve exhausted')
        ledger['status'] = 'raw_saved'
        save(out / 'advance-ledger.json', ledger)
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'response_progress': getattr(error, 'response_progress', None),
            'precomputed_expm1_progress': getattr(error,
                'precomputed_expm1_progress', None),
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
        except BaseException as error:
            secondary('finish_cuda', error)
        ledger['host_spans'] = obs.host
        ledger['cuda_spans'] = obs.cuda
        ledger['solve_spans'] = obs.solve
        ledger['secondary_errors'] = obs.secondary
        if engine is not None:
            try:
                ledger['engine_after'] = profile.snapshot(engine.metadata())
            except BaseException as error:
                secondary('engine_metadata', error)
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                secondary('threadpool_restore', error)
        if primary is None:
            try:
                result = result_from_raw(freeze, ledger, comparison,
                                         stop_reason, cache)
                if time.perf_counter() >= began + freeze['internal_wall_limit_s'] - 1:
                    raise TimeoutError('E132 result reserve exhausted')
                if result['new_lf_proposal_accepted']:
                    selected = ledger['original_evaluations'][result['selected_index'] - 1]
                    with np.load(selected['raw_path'], allow_pickle=False) as raw:
                        np.savez(out / 'selected-lf-proposal.npz', delta=raw['delta'],
                            gain=raw['gain'], original_evaluation_index=np.array(
                                result['selected_index']), source_raw_sha256=np.array(
                                    selected['raw_sha256']))
                    ledger['selected_proposal_sha256'] = profile.sha256(
                        out / 'selected-lf-proposal.npz')
                save(out / 'result.json', result)
                ledger['status'] = result['status']
            except BaseException as error:
                secondary('result_finalization', error)
        primary = finalize_artifacts(out, ledger, freeze, began, primary, secondary)
    if primary is not None:
        raise primary
    return out / 'result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha', 'source-review',
                 'source-review-sha', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.source_review,
        args.source_review_sha, args.out)


if __name__ == '__main__':
    main()
