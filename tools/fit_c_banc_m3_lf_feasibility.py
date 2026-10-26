"""E131 bounded LF initial-constraint restoration, dispatched once by the parent."""

# ruff: noqa: BLE001  # preserve a primary CUDA/persistence failure and its evidence

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import banc_m3_precomputed_expm1 as adapter_module
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import profile_c_banc_m3_joint_fit_cost as profile
from tools import verify_c_banc_m3_precomputed_expm1 as e130

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '5be3e7ae7a376f51b1ffb5d6419201cf0c37aa738bbaa0c4b795c94f137914fb'
SOURCE_FILES = ('tools/fit_c_banc_m3_lf_feasibility.py',
                'tests/test_c_banc_lf_feasibility.py')
SOLVER_OPTIONS = {'maxiter': 100, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10}


class BoundedStop(Exception):
    """Normal bounded termination, including the first selector-feasible raw."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def save(path, value):
    fit.write_json(path, profile.snapshot(value))


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or profile.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E131 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    required = {'schema': 'flylab.m3.e131.lf-feasibility-freeze.v1',
        'leg': 0, 'start_evaluation_index': 13, 'objective':
        '0.5 * sum(min(c[3:], 0)**2)', 'constraint_count': 4758,
        'solver': 'L-BFGS-B', 'solver_options': SOLVER_OPTIONS,
        'optimizer_cap': fit.OPTIMIZER_CAP,
        'selector_cap': fit.SELECTOR_CAP, 'max_unique_evaluations': 16,
        'fit_wall_limit_s': 20, 'call_reserve_s': 2,
        'internal_wall_limit_s': 27, 'external_wall_limit_s': 32,
        'supervisor_limit_s': 35, 'prior_conservative_process_s': 1428,
        'allocated_conservative_end_s': 1463, 'shared_process_limit_s': 1530,
        'synthetic_tiny_batches': 1, 'synthetic_tiny_context_ticks': 20,
        'stop_at_first_selector_feasible': True, 'full_BANC_calls': 0,
        'body_calls': 0, 'F6_exposed': False, 'whole_candidate_adoption': False,
        'original_processes': 1, 'retries': 0}
    if (any(freeze.get(k) != v for k, v in required.items())
            or freeze.get('dimensions') != {'contexts': 3, 'MN': 69,
                'muscles': 17, 'edges': 273, 'parameters': 290}
            or freeze.get('maximum_original_work') != {
                'context_ticks': 1920000, 'conditional_s': 480,
                'scalar_MN_ticks': 132480000,
                'edge_stage_ticks': 2096640000, 'derivative_banks': 16,
                'decoder_endpoints': 96000, 'geometry_forward_rows': 87568,
                'geometry_backward_rows': 87568}
            or len(freeze.get('input_sha256', {})) != 9
            or len(freeze.get('reused_source_sha256', {})) != 3
            or freeze.get('design_sha256') != profile.sha256(ROOT / freeze['design_path'])
            or any(freeze.get(key) not in freeze['input_sha256'] for key in
                   ('E130_freeze_path', 'E130_source_review_path',
                    'start_raw_path', 'full_reference_raw_path'))):
        raise ValueError('E131 frozen contract changed')
    for key in ('input_sha256', 'reused_source_sha256'):
        for relative, digest in freeze[key].items():
            profile.checked_path(relative, digest)
    e130_freeze_path = ROOT / freeze['E130_freeze_path']
    e130_review_path = ROOT / freeze['E130_source_review_path']
    _, _, e128, _ = e130.reviewed_inputs(
        e130_freeze_path, freeze['input_sha256'][freeze['E130_freeze_path']],
        e130_review_path, freeze['input_sha256'][freeze['E130_source_review_path']])
    manifest_path = ROOT / ('verification/banc-natural-gait-m3-20260919/'
                            'e130-precomputed-expm1-supervision/original-output-manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest.get('original_files', {})
    if manifest.get('original_file_count') != 18 or len(files) != 18:
        raise ValueError('E130 original manifest count changed')
    original_dir = ROOT / 'verification/banc-natural-gait-m3-20260919/e130-precomputed-expm1'
    for relative, digest in files.items():
        name = Path(relative)
        if (name.is_absolute() or '..' in name.parts or
                profile.sha256(original_dir / name) != digest):
            raise ValueError('E130 original output SHA mismatch: ' + relative)
    if profile.sha256(review_path) != review_sha:
        raise ValueError('E131 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_lf_feasibility') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E131 two-source review required')
    for relative, digest in review['source_sha256'].items():
        profile.checked_path(relative, digest)
    return freeze, e128, review


def restoration_value_jac(data):
    """Use E128's exact 4761 inequalities; omit only three loss epigraph rows."""
    t = float(np.max(data['losses']) + .01)
    c, jac, ratios = fit.constraint_values(
        data['losses'], data['loss_jac'], data['q_error'],
        data['ltm_error'], data['q_jac'], data['ltm_jac'], t,
        optimizer_cap=fit.OPTIMIZER_CAP)
    p = len(data['delta']) + len(data['gain'])
    if c.shape != (4761,) or jac.shape != (4761, p + 1):
        raise ValueError('E128 constraint or Jacobian shape changed')
    v = np.minimum(c[3:], 0.)
    value = float(.5 * (v @ v))
    gradient = np.asarray(jac[3:, :-1].T @ v, np.float64)
    if (not np.isfinite(value) or not np.isfinite(gradient).all()
            or not np.isfinite(ratios).all()):
        raise ValueError('Nonfinite restoration objective/Jacobian/ratios')
    return value, gradient, ratios


class RestorationCache:
    """A float64 delta/gain byte key counts one physical E128 evaluation."""

    def __init__(self, evaluate, start, bounds, *, max_unique, fit_deadline,
                 internal_deadline, reserve, clock=time.perf_counter):
        self.evaluate = evaluate
        self.start = np.asarray(start, np.float64)
        self.bounds = np.asarray(bounds, np.float64)
        self.max_unique = max_unique
        self.fit_deadline = fit_deadline
        self.internal_deadline = internal_deadline
        self.reserve = reserve
        self.clock = clock
        self.cache = {}
        self.started = 0
        self.completed = 0
        self.hits = 0
        self.selected_index = None

    def get(self, vector):
        x = np.asarray(vector, np.float64)
        if (x.shape != self.start.shape or not np.isfinite(x).all()
                or np.any(x < self.bounds[:, 0] - 1e-12)
                or np.any(x > self.bounds[:, 1] + 1e-12)):
            raise ValueError('Finite in-bounds delta/gain vector required')
        key = np.ascontiguousarray(x).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.started >= self.max_unique:
            raise BoundedStop('unique_cap')
        if self.clock() >= min(self.fit_deadline, self.internal_deadline) - self.reserve:
            raise BoundedStop('pre_call_wall_reserve')
        self.started += 1
        data = self.evaluate(x, self.started)
        self.completed += 1
        value, jac, ratios = restoration_value_jac(data)
        result = (value, jac)
        self.cache[key] = result
        if fit.selector_feasible(ratios):
            self.selected_index = self.started
            raise BoundedStop('first_selector_feasible')
        return result


def original_bounds(fixture, edge_indices):
    bounds = np.vstack((np.asarray(fixture['delta_bounds'][edge_indices], np.float64),
                        np.tile(np.array([[0., 8.]]), (17, 1))))
    if bounds.shape != (290, 2) or not np.isfinite(bounds).all() or np.any(
            bounds[:, 0] > bounds[:, 1]):
        raise ValueError('Frozen LF bounds changed')
    return bounds


def run_optimizer(cache, accounting=None, *, clock=time.perf_counter):
    """The first explicit evaluation is included; the solver reuses its cache."""
    if accounting is None:
        accounting = {}
    accounting.update(status='not_requested', requested=0, returned=0,
                      bounded_stop=0, failed_unknown=0)
    stop = 'scipy_finished'
    solver = None
    began = clock()
    try:
        cache.get(cache.start)
        accounting.update(status='requested', requested=1)
        try:
            solver = minimize(cache.get, cache.start.copy(), method='L-BFGS-B',
                              jac=True, bounds=[tuple(row) for row in cache.bounds],
                              options=SOLVER_OPTIONS.copy())
        except BoundedStop:
            accounting.update(status='bounded_stop', bounded_stop=1)
            raise
        except BaseException as error:
            accounting.update(status='failed_unknown', failed_unknown=1,
                              error=repr(error))
            raise
        accounting.update(status='returned', returned=1)
    except BoundedStop as error:
        stop = error.reason
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


def _complete_evaluation(row):
    if not isinstance(row, dict):
        return False
    solve = row.get('solve')
    metadata = row.get('engine_metadata')
    geometry = metadata.get('geometry_cost', {}) if isinstance(metadata, dict) else {}
    if not isinstance(geometry, dict) or not isinstance(solve, dict):
        return False
    try:
        return (row.get('status') == 'raw_saved'
        and row.get('raw_sha256') and row.get('raw_path')
        and row.get('raw_index') == 1
        and row.get('requested_context_ticks') == 120000
        and row.get('requested_conditional_s') == 30
        and row.get('requested_scalar_MN_ticks') == 8280000
        and row.get('requested_derivative_banks') == 1
        and row.get('requested_edge_stage_ticks') == 131040000
        and row.get('requested_decoder_endpoints') == 6000
        and row.get('solve_span_count') == 1
        and solve.get('phase') == row.get('phase')
        and solve.get('returned') is True
        and e130.known_progress(solve.get('response_progress'),
            ticks=120000, conditional_s=30, decoder=6000, bank=131040000)
        and e130.known_adapter(row.get('adapter'), 273)
        and e130.known_metadata(metadata, row['index'])
        and all(geometry.get(key) == 5473 * row['index'] for key in
            ('forward_rows_requested', 'forward_rows_enqueued',
             'forward_rows_completed', 'backward_rows_requested',
             'backward_rows_enqueued', 'backward_rows_completed'))
        and geometry.get('forward_rows_unknown_requested') == 0
        and geometry.get('backward_rows_unknown_requested') == 0)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False


def _row_ratio_valid(row):
    try:
        ratios = np.asarray(row.get('ratios'), np.float64)
        maximum = float(row.get('max_ratio'))
        value = float(row.get('restoration_value'))
        gradient = float(row.get('restoration_gradient_norm'))
        return (ratios.shape == (3, 4) and np.isfinite(ratios).all()
            and np.all(ratios >= 0)
            and np.isfinite([maximum, value, gradient]).all()
            and value >= 0 and gradient >= 0
            and np.isclose(maximum, np.max(ratios), rtol=0, atol=1e-12)
            and row.get('selector_feasible') is fit.selector_feasible(ratios))
    except (TypeError, ValueError, OverflowError):
        return False


def result_from_raw(freeze, ledger, comparisons, stop_reason, cache):
    rows = ledger.get('original_evaluations', [])
    if not isinstance(rows, list):
        rows = []
    complete = (1 <= len(rows) <= 16
        and type(ledger.get('original_evaluations_requested')) is int
        and type(ledger.get('original_evaluations_completed')) is int
        and len(rows) == ledger.get('original_evaluations_requested')
        == ledger.get('original_evaluations_completed')
        and all(isinstance(row, dict) and type(row.get('index')) is int
                and row.get('index') == i
                and row.get('phase') == f'original_{i}_candidate'
                and row.get('backend') == 'candidate'
                and _complete_evaluation(row) and _row_ratio_valid(row)
                for i, row in enumerate(rows, 1)))
    inputs = freeze.get('input_sha256', {})
    if not isinstance(inputs, dict):
        inputs = {}
    old_sha = inputs.get(freeze.get('start_raw_path'))
    full_sha = inputs.get(freeze.get('full_reference_raw_path'))
    compared = (complete and isinstance(comparisons, list)
        and len(comparisons) == 2 and old_sha is not None and full_sha is not None
        and all(isinstance(item, dict) for item in comparisons)
        and comparisons[0].get('old_reference') is True
        and comparisons[1].get('old_reference') is False
        and comparisons[0].get('reference_sha256') == old_sha
        and comparisons[1].get('reference_sha256') == full_sha
        and all(item.get('accepted') is True
                and item.get('actual_sha256') == rows[0]['raw_sha256']
                for item in comparisons))
    tiny = ledger.get('synthetic_tiny', [])
    if not isinstance(tiny, list):
        tiny = []
    try:
        tiny_ok = (len(tiny) == ledger.get('synthetic_tiny_requested')
            == ledger.get('synthetic_tiny_completed') == 1
            and isinstance(tiny[0], dict)
            and tiny[0].get('status') == 'completed' and tiny[0].get('raw_sha256')
            and e130.known_progress(tiny[0].get('progress'),
                ticks=20, conditional_s=.005, decoder=1, bank=80)
            and e130.known_adapter(tiny[0].get('adapter'), 1)
            and e130.known_metadata(tiny[0].get('engine_metadata'), 1))
    except (TypeError, ValueError, KeyError, OverflowError):
        tiny_ok = False
    spans = ledger.get('cuda_spans', [])
    if not isinstance(spans, list):
        spans = []
    try:
        timing_ok = (complete and all(isinstance(x, dict) for x in spans)
            and all(e130.complete_span(
                type('Observation', (), {'cuda': spans})(),
                row['phase'], label) is not None
                for row in rows for label in ('joint_forward', 'muscle_ema',
                    'edge_bank', 'geometry.leg', 'torch.autograd.grad'))
            and all(sum(x.get('phase') == row['phase'] for x in spans) == 7
                    for row in rows))
    except (TypeError, ValueError, KeyError, OverflowError):
        timing_ok = False
    started = getattr(cache, 'started', None)
    completed = getattr(cache, 'completed', None)
    hits = getattr(cache, 'hits', None)
    cached = getattr(cache, 'cache', None)
    cache_ok = (type(started) is int and type(completed) is int
        and type(hits) is int and hits >= 0 and isinstance(cached, dict)
        and started == completed == len(rows) == len(cached)
        and ledger.get('cache_started') == started
        and ledger.get('cache_completed') == completed
        and ledger.get('cache_hits') == hits)
    selected = getattr(cache, 'selected_index', None)
    selected_valid = (type(selected) is int and 1 <= selected <= len(rows))
    flags = [row.get('selector_feasible') if isinstance(row, dict) else None
             for row in rows]
    selection_ok = (selected_valid and stop_reason == 'first_selector_feasible'
        and selected == len(rows) and flags[-1] is True
        and all(flag is False for flag in flags[:-1]))
    no_selection_ok = (selected is None and all(flag is False for flag in flags)
        and stop_reason in ('scipy_finished', 'unique_cap',
                            'pre_call_wall_reserve'))
    optimizer = ledger.get('optimizer', {})
    if not isinstance(optimizer, dict):
        optimizer = {}
    fit_calls = ledger.get('fit_calls')
    if type(fit_calls) is not int:
        fit_calls = None
    if selection_ok and selected == 1:
        optimizer_ok = (fit_calls == 0 and optimizer.get('requested') == 0
            and optimizer.get('returned') == 0
            and optimizer.get('bounded_stop') == 0
            and optimizer.get('failed_unknown') == 0
            and optimizer.get('status') == 'not_requested')
    elif selection_ok or (no_selection_ok and stop_reason != 'scipy_finished'):
        optimizer_ok = (fit_calls == 1 and optimizer.get('requested') == 1
            and optimizer.get('returned') == 0
            and optimizer.get('bounded_stop') == 1
            and optimizer.get('failed_unknown') == 0
            and optimizer.get('status') == 'bounded_stop')
    elif no_selection_ok and stop_reason == 'scipy_finished':
        optimizer_ok = (fit_calls == 1 and optimizer.get('requested') == 1
            and optimizer.get('returned') == 1
            and optimizer.get('bounded_stop') == 0
            and optimizer.get('failed_unknown') == 0
            and optimizer.get('status') == 'returned')
    else:
        optimizer_ok = False
    wall = ledger.get('fit_wall_s')
    try:
        wall = float(wall)
        budget_ok = bool(np.isfinite(wall) and 0 <= wall <=
                         freeze['fit_wall_limit_s'])
        if not np.isfinite(wall):
            wall = None
    except (TypeError, ValueError, KeyError, OverflowError):
        budget_ok = False
        wall = None
    evidence = bool(complete and compared and tiny_ok and timing_ok and cache_ok
        and optimizer_ok and (selection_ok or no_selection_ok)
        and not ledger.get('secondary_errors') and budget_ok)
    feasible = bool(evidence and selection_ok)
    ratios = []
    for row in rows:
        if isinstance(row, dict):
            try:
                ratio = float(row.get('max_ratio'))
                if np.isfinite(ratio):
                    ratios.append(ratio)
            except (TypeError, ValueError, OverflowError):
                pass
    best = min(ratios) if ratios else None
    return {'schema': 'flylab.m3.e131.lf-feasibility-result.v1',
        'status': ('completed_bounded_fit' if evidence else
                   'invalid_budget' if not budget_ok else 'invalid_evidence'),
        'restoration_feasible': bool(feasible),
        'interpretation': ('limited_LF_restoration_proposal' if feasible else
                           'bounded_failure_inconclusive' if evidence else
                           'invalid_or_incomplete'),
        'selected_index': selected if feasible else None,
        'stop_reason': stop_reason, 'best_max_ratio': best,
        'original_evaluations_requested': ledger.get('original_evaluations_requested'),
        'original_evaluations_completed': ledger.get('original_evaluations_completed'),
        'cache_hits': hits, 'fit_calls': fit_calls, 'body_calls': 0,
        'F6': False, 'F6_exposed': False, 'candidate_adopted': False,
        'whole_goal_complete': False, 'first_reference_comparisons': compared,
        'synthetic_tiny_complete': tiny_ok, 'timing_complete': timing_ok,
        'original_completion_accounting_complete': complete,
        'cache_accounting_complete': cache_ok,
        'optimizer_accounting_complete': optimizer_ok,
        'selection_consistent': selection_ok or no_selection_ok,
        'fit_budget_complete': budget_ok, 'fit_wall_s': wall,
        'source_review_sha256': ledger.get('source_review_sha256'),
        'freeze_sha256': ledger.get('freeze_sha256')}


def _run_tiny(out, ledger):
    row = {'status': 'requested', 'synthetic_only': True,
           'requested_context_ticks': 20, 'requested_conditional_s': .005,
           'requested_edge_stage_ticks': 80, 'requested_derivative_banks': 1}
    ledger['synthetic_tiny'].append(row)
    ledger['synthetic_tiny_requested'] += 1
    save(out / 'advance-ledger.json', ledger)
    engine = None
    adapter = None
    began = time.perf_counter()
    try:
        tiny = fit.synthetic_tiny_setup_problem()
        engine = neural.CudaJointAdaptiveEngine(tiny)
        with adapter_module.precomputed_expm1(engine) as adapter:
            _, raw, timing = engine.solve(np.zeros(1), np.zeros(1), derivative=True)
        path = out / 'synthetic-cuda-tiny-setup-raw.npz'
        np.savez(path, synthetic_stage_current=tiny['stage_current'],
                 synthetic_source_rate=tiny['source_rate'], **raw)
        row.update(status='raw_saved', raw_path=str(path),
                   raw_sha256=profile.sha256(path), progress=timing['progress'],
                   adapter=adapter.snapshot(),
                   engine_metadata=profile.snapshot(engine.metadata()))
        save(out / 'advance-ledger.json', ledger)
        if not all(np.isfinite(raw[key]).all() for key in neural.FIELDS):
            raise ValueError('E131 tiny raw nonfinite')
        row['status'] = 'completed'
        ledger['synthetic_tiny_completed'] += 1
    except BaseException as error:
        row.update(status='failed', error=repr(error),
                   progress=getattr(error, 'response_progress', None),
                   adapter=(adapter.snapshot() if adapter else
                            getattr(error, 'precomputed_expm1_progress', None)),
                   fixed_upload_progress=getattr(error, 'fixed_upload_progress', None))
        if engine is not None:
            e130.safe_finally(('tiny_metadata', lambda: row.update(
                engine_metadata=profile.snapshot(engine.metadata()))))
        partial = getattr(error, 'partial_response', None)
        if partial:
            try:
                np.savez(out / 'synthetic-cuda-tiny-setup-partial-raw.npz', **partial)
            except BaseException as secondary:
                error.add_note('Secondary tiny partial save: ' + repr(secondary))
        row['partial_raw_files'] = e130.partial_paths(
            out, error, 'synthetic-cuda-tiny-setup-partial-raw.npz')
        raise
    finally:
        row['wall_s'] = time.perf_counter() - began
        e130.safe_finally(('tiny_ledger', lambda: save(out / 'advance-ledger.json', ledger)))


def write_output_manifest(out):
    save(out / 'output-manifest.json', {
        'schema': 'flylab.m3.e131.output-manifest.v1',
        'files': {str(path.relative_to(out)).replace('\\', '/'): profile.sha256(path)
            for path in sorted(out.rglob('*')) if path.is_file()
            and path.name != 'output-manifest.json'}})


def finalize_artifacts(out, ledger, freeze, began, primary, secondary):
    """A late or failed receipt cannot leave a success result in place."""
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
                'E131 internal wall limit exceeded after manifest'))
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
                                   'restoration_feasible': False})
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
        raise FileExistsError('One-shot E131 output exists: ' + str(out))
    out.mkdir(parents=True)
    ledger = {'schema': 'flylab.m3.e131.lf-feasibility-ledger.v1',
              'pid': os.getpid(), 'freeze_sha256': freeze_sha,
              'source_review_sha256': source_review_sha, 'status': 'preflight',
              'original_evaluations': [], 'original_evaluations_requested': 0,
              'original_evaluations_completed': 0, 'synthetic_tiny': [],
              'synthetic_tiny_requested': 0, 'synthetic_tiny_completed': 0,
              'fit_calls': 0, 'body_calls': 0, 'F6': False, 'F6_exposed': False,
              'candidate_adopted': False}
    save(out / 'advance-ledger.json', ledger)
    obs = profile.Observation()
    freeze = limits = engine = None
    primary = None
    cache = None
    comparisons = []
    stop_reason = None
    solver = None
    try:
        freeze, e128, review = reviewed_inputs(
            freeze_path, freeze_sha, source_review, source_review_sha)
        save(out / 'source-review-snapshot.json', review)
        vector = e130.reference_vector(ROOT / freeze['start_raw_path'])
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version, 'numpy': np.__version__,
            'cupy': cp.__version__, 'torch': torch.__version__,
            'gpu': gpu.decode() if isinstance(gpu, bytes) else str(gpu)}
        _run_tiny(out, ledger)
        fixture = obs.timed('frozen_fixture_load', fit.frozen_problem)(e128)

        def load_geometry():
            value = TorchFootGeometry.load(ROOT / e128['geometry_path'], device='cuda')
            torch.cuda.synchronize(value.device)
            return value

        geometry = obs.timed('geometry_fixed_upload', load_geometry)()
        if geometry.geometry_hash != e128['geometry_hash']:
            raise ValueError('Frozen E128 geometry hash changed')
        ledger['geometry_fixed_upload_bytes'] = sum(int(geometry.arrays[key].nbytes)
            for key in ('axes', 'anchors', 'tips', 'reference_angles'))
        motors, muscles, edges = (np.flatnonzero(fixture[key][0]) for key in
            ('leg_motor_mask', 'leg_muscle_mask', 'leg_edge_mask'))
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('Frozen LF cardinality changed')
        bounds = original_bounds(fixture, edges)
        if np.any(vector < bounds[:, 0]) or np.any(vector > bounds[:, 1]):
            raise ValueError('Frozen record13 vector is out of E128 bounds')
        try:
            engine = obs.timed('lf_engine_fixed_upload',
                               neural.CudaJointAdaptiveEngine)(
                                   fixture, motors, muscles, edges)
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
                               'returned': 0, 'bounded_stop': 0,
                               'failed_unknown': 0}
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
                           raw_sha256=record['raw_sha256'],
                           solve_wall_s=record['wall_s'], adapter=adapter.snapshot())
                ledger['original_evaluations_completed'] += 1
                save(out / 'advance-ledger.json', ledger)
                if index == 1:
                    references = ((ROOT / freeze['start_raw_path'], True, 'old'),
                                  (ROOT / freeze['full_reference_raw_path'], False, 'full'))
                    for reference, old, label in references:
                        comparison = e130.compare_raw(record['raw_path'], reference,
                                                       old_reference=old)
                        comparisons.append(comparison)
                        save(out / f'first-reference-{label}-comparison.json', comparison)
                    ledger['first_reference_comparisons'] = [
                        x['accepted'] for x in comparisons]
                    save(out / 'advance-ledger.json', ledger)
                    if not all(x['accepted'] for x in comparisons):
                        raise ValueError('First new LF raw failed frozen references')
                value, jac, ratios = restoration_value_jac(record)
                row.update(restoration_value=value, restoration_gradient_norm=float(
                    np.linalg.norm(jac)), max_ratio=float(np.max(ratios)),
                    ratios=ratios, selector_feasible=fit.selector_feasible(ratios))
                save(out / 'advance-ledger.json', ledger)
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
                matches = [x for x in obs.solve if x['phase'] == phase]
                row['solve_span_count'] = len(matches)
                e130.safe_finally(
                    ('engine_metadata', lambda: row.update(
                        engine_metadata=profile.snapshot(engine.metadata()))),
                    ('current_phase_solve', lambda: row.update(solve=profile.snapshot(
                        matches[0] if len(matches) == 1 else None))),
                    ('evaluation_ledger', lambda: save(out / 'advance-ledger.json', ledger)))

        cache = RestorationCache(evaluate, vector, bounds,
            max_unique=freeze['max_unique_evaluations'],
            fit_deadline=fit_deadline, internal_deadline=internal_deadline,
            reserve=freeze['call_reserve_s'])
        try:
            solver, stop_reason = run_optimizer(cache, ledger['optimizer'])
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
            raise TimeoutError('E131 result reserve exhausted')
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
                result = result_from_raw(freeze, ledger, comparisons,
                                         stop_reason, cache)
                if time.perf_counter() >= began + freeze['internal_wall_limit_s'] - 1:
                    raise TimeoutError('E131 result reserve exhausted')
                if result['restoration_feasible']:
                    selected = ledger['original_evaluations'][cache.selected_index - 1]
                    with np.load(selected['raw_path'], allow_pickle=False) as raw:
                        np.savez(out / 'selected-lf-proposal.npz', delta=raw['delta'],
                            gain=raw['gain'], original_evaluation_index=np.array(
                                cache.selected_index), source_raw_sha256=np.array(
                                    selected['raw_sha256']))
                    ledger['selected_proposal_sha256'] = profile.sha256(
                        out / 'selected-lf-proposal.npz')
                save(out / 'result.json', result)
                ledger['status'] = result['status']
            except BaseException as error:
                secondary('result_finalization', error)
        primary = finalize_artifacts(out, ledger, freeze, began,
                                     primary, secondary)
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
