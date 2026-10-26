"""E129: one reviewed, fixed LF E128 evaluation with observation-only timing.

The parent owns original dispatch. This program never invokes a fit or a full
response and deliberately has no retry path.
"""

# ruff: noqa: BLE001  # retain primary failures and secondary timing receipts

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '842b1e83a8e282bb27bc8b64a76c1b84ddff637c82391a05149b04ecb3fe82ba'
SOURCE_FILES = ('tools/profile_c_banc_m3_joint_fit_cost.py',
                'tests/test_c_banc_joint_fit_cost_profile.py')
KERNELS = ('joint_forward', 'muscle_ema', 'edge_bank')
NUMERIC_TIGHT = ('losses', 'parts', 'ratios')
NUMERIC_LOOSE = (*neural.FIELDS, 'B_initial99', 'q_error', 'ltm_error',
                 'loss_jac', 'q_coefficient', 'ltm_coefficient', 'cotangent',
                 'context0_command', 'context0_tendon', 'context1_command',
                 'context1_tendon', 'context2_command', 'context2_tendon')
EXACT = ('delta', 'gain', 'edge_ids', 'motor_indices', 'muscle_indices',
         'parameter_muscle_map', 'branch_bits', 'timing_context_ticks',
         'timing_bank_complete')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    fit.write_json(path, obj)


def snapshot(value):
    """Detach nested engine counters from subsequent in-place mutation."""
    return copy.deepcopy(fit.jsonable(value))


def checked_path(relative, digest):
    return fit.checked_relative(relative, digest)


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('E129 exact freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if (freeze.get('schema') != 'flylab.m3.e129.fixed-lf-profile-freeze.v1'
            or freeze.get('design_sha256') != sha256(ROOT / freeze['design_path'])
            or freeze.get('original_evaluation_limit') != 1
            or freeze.get('original_fit_calls') != 0
            or (freeze.get('leg'), freeze.get('leg_MN'),
                freeze.get('leg_muscles'), freeze.get('leg_edges'))
            != (0, 69, 17, 273)
            or freeze.get('original_context_ticks') != 120000
            or freeze.get('original_conditional_s') != 30
            or freeze.get('internal_wall_limit_s') != 30
            or freeze.get('external_wall_limit_s') != 40
            or freeze.get('supervisor_limit_s') != 45
            or freeze.get('allocated_conservative_end_s') != 1440
            or freeze.get('shared_process_limit_s') != 1530):
        raise ValueError('E129 fixed profile contract changed')
    for key, count in (('input_sha256', 7), ('reused_source_sha256', 3)):
        if len(freeze.get(key, {})) != count:
            raise ValueError('E129 frozen map cardinality changed: ' + key)
        for relative, digest in freeze[key].items():
            checked_path(relative, digest)
    e128_path = checked_path(freeze['E128_freeze_path'],
                            freeze['input_sha256'][freeze['E128_freeze_path']])
    e128 = json.loads(e128_path.read_text(encoding='utf-8'))
    # This call verifies every nested E128 input, source and preservation hash.
    e128_review = ROOT / 'verification/banc-natural-gait-m3-20260919/parent-e128-final-source-review.json'
    fit.reviewed_inputs(e128_path, sha256(e128_path), e128_review,
                        sha256(e128_review))
    manifest_path = checked_path(freeze['original_output_manifest_path'],
                                 freeze['input_sha256'][freeze['original_output_manifest_path']])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    original_dir = ROOT / 'verification/banc-natural-gait-m3-20260919/e128-joint-adaptive-fit'
    if len(manifest.get('original_files', {})) != 112:
        raise ValueError('E128 original output count changed')
    for relative, digest in manifest['original_files'].items():
        if '/' in relative or '\\' in relative:
            raise ValueError('Unexpected original output path')
        if sha256(original_dir / relative) != digest:
            raise ValueError('Original E128 output SHA mismatch: ' + relative)
    if sha256(review_path) != review_sha:
        raise ValueError('E129 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_fixed_lf_profile') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != freeze_sha
            or set(review.get('source_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E129 two-source review required')
    for relative, digest in review['source_sha256'].items():
        checked_path(relative, digest)
    if (e128.get('fixture_path') not in e128['input_sha256']
            or freeze['reference_raw_path'] not in freeze['input_sha256']
            or freeze['reference_leg_summary_path'] not in freeze['input_sha256']):
        raise ValueError('Reference or fixture is not frozen')
    return freeze, e128, review


def reference_vector(reference_path):
    with np.load(reference_path, allow_pickle=False) as z:
        if (z['delta'].shape != (273,) or z['gain'].shape != (17,)
                or not np.array_equal(z['delta'], np.zeros(273))
                or not np.array_equal(z['gain'], np.zeros(17))):
            raise ValueError('LF reference first vector is not zero')
        return np.r_[z['delta'], z['gain']]


def compare_raw(actual_path, reference_path):
    """Read only after E128 has saved actual raw; never create a response."""
    result = {'actual_sha256': sha256(actual_path),
              'reference_sha256': sha256(reference_path), 'fields': {},
              'accepted': False}
    with np.load(actual_path, allow_pickle=False) as actual, np.load(
            reference_path, allow_pickle=False) as reference:
        for key in (*EXACT, *NUMERIC_LOOSE, *NUMERIC_TIGHT):
            if key not in actual or key not in reference:
                result['fields'][key] = {'status': 'missing'}
                continue
            a, b = actual[key], reference[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                result['fields'][key] = {'status': 'shape_or_dtype_mismatch',
                                         'actual_shape': a.shape,
                                         'reference_shape': b.shape}
                continue
            if key in EXACT:
                ok = bool(np.array_equal(a, b))
                result['fields'][key] = {'status': 'exact' if ok else 'mismatch'}
            else:
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                difference = float(np.max(np.abs(a - b))) if a.size else 0.0
                bound = 1e-10 if key in NUMERIC_TIGHT else 1e-8
                result['fields'][key] = {
                    'status': 'within_bound' if finite and difference <= bound else 'mismatch',
                    'finite': finite, 'max_abs_difference': difference, 'bound': bound}
        result['accepted'] = all(x['status'] in ('exact', 'within_bound')
                                 for x in result['fields'].values())
    return result


class Observation:
    """Temporary callable wrappers; CUDA events are recorded on engine.stream."""

    def __init__(self):
        self.phase = 'setup'
        self.engine = None
        self.host = []
        self.cuda = []
        self.secondary = []
        self.solve = []

    def timed(self, label, function):
        def call(*args, **kwargs):
            record = {'phase': self.phase, 'label': label, 'requested': True,
                      'returned': False, 'wall_s': None}
            self.host.append(record)
            began = time.perf_counter()
            try:
                value = function(*args, **kwargs)
                record['returned'] = True
                return value
            except BaseException as error:
                record['error'] = repr(error)
                raise
            finally:
                record['wall_s'] = time.perf_counter() - began
        return call

    def solve_timed(self, function):
        host_call = self.timed('engine.solve', function)

        def call(*args, **kwargs):
            entry = {'phase': self.phase, 'requested': True,
                     'returned': False, 'timing': None,
                     'response_progress': None}
            self.solve.append(entry)
            try:
                value = host_call(*args, **kwargs)
            except BaseException as error:
                entry['error'] = repr(error)
                try:
                    entry['response_progress'] = snapshot(
                        getattr(error, 'response_progress', None))
                except Exception as secondary:
                    self._timing_error(entry, 'solve_failure_snapshot', secondary)
                raise
            entry['returned'] = True
            try:
                entry['timing'] = snapshot(value[2])
                entry['response_progress'] = snapshot(value[2]['progress'])
            except Exception as error:
                self._timing_error(entry, 'solve_success_snapshot', error)
            return value
        return call

    def _timing_error(self, record, stage, error):
        item = {'phase': record['phase'], 'label': record.get('kernel',
                record.get('label')), 'stage': stage, 'error': repr(error)}
        self.secondary.append(item)
        record.setdefault('timing_errors', []).append(item)

    def torch_timed(self, label, function):
        """Host return is independent of Torch current-stream event completion."""
        import torch

        def call(*args, **kwargs):
            record = {'phase': self.phase, 'label': label, 'requested': True,
                      'returned': False, 'enqueued': False,
                      'unknown_requested': False,
                      'start_recorded': False, 'end_recorded': False,
                      'synchronized': False, 'elapsed_ms': None}
            self.cuda.append(record)
            start = end = stream = None
            try:
                tensor = args[0]
                stream = torch.cuda.current_stream(device=tensor.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                record['start_recorded'] = True
            except Exception as error:
                self._timing_error(record, 'torch_start', error)
            host_call = self.timed(label, function)
            try:
                value = host_call(*args, **kwargs)
                record['returned'] = True
                record['enqueued'] = True
                return value
            except BaseException as error:
                record['error'] = repr(error)
                record['unknown_requested'] = True
                raise
            finally:
                if record['start_recorded']:
                    try:
                        end.record(stream)
                        record['end_recorded'] = True
                        record['_events'] = (start, end)
                    except Exception as error:
                        self._timing_error(record, 'torch_end', error)
        return call

    def kernel_factory(self, original):
        def factory(name):
            kernel = original(name)
            if name not in KERNELS:
                return kernel

            def launch(*args, **kwargs):
                record = {'phase': self.phase, 'kernel': name,
                          'requested': True, 'returned': False,
                          'enqueued': False, 'unknown_requested': False,
                          'start_recorded': False, 'end_recorded': False,
                          'synchronized': False, 'elapsed_ms': None}
                self.cuda.append(record)
                engine = self.engine
                if engine is None:
                    raise RuntimeError('Observed CUDA kernel has no bound engine')
                start = end = None
                try:
                    import cupy as cp
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record(engine.stream)
                    record['start_recorded'] = True
                except Exception as error:
                    self._timing_error(record, 'cupy_start', error)
                try:
                    value = kernel(*args, **kwargs)
                except BaseException as error:
                    record['error'] = repr(error)
                    record['unknown_requested'] = True
                    raise
                record['returned'] = True
                record['enqueued'] = True  # RawKernel callable returned after launch.
                if record['start_recorded']:
                    try:
                        end.record(engine.stream)
                        record['end_recorded'] = True
                        record['_events'] = (start, end)
                    except Exception as error:
                        self._timing_error(record, 'cupy_end', error)
                return value
            return launch
        return factory

    def finish_cuda(self):
        for record in self.cuda:
            events = record.pop('_events', None)
            if events is None:
                continue
            try:
                events[1].synchronize()
                record['synchronized'] = True
                if 'kernel' in record:
                    import cupy as cp
                    elapsed = cp.cuda.get_elapsed_time(*events)
                else:
                    elapsed = events[0].elapsed_time(events[1])
                record['elapsed_ms'] = float(elapsed)
            except Exception as error:
                self._timing_error(record, 'cuda_event_resolution', error)


@contextlib.contextmanager
def observed_callables(observation, engine, geometry):
    """Restore exactly the prior callables, even after a partial failure."""
    import torch
    originals = [(neural, '_kernel', neural._kernel),
                 (engine, 'solve', engine.solve),
                 (geometry, 'leg', geometry.leg),
                 (torch.autograd, 'grad', torch.autograd.grad),
                 (fit, 'activation_cotangent_product', fit.activation_cotangent_product),
                 (fit, 'constraint_values', fit.constraint_values),
                 (np, 'savez', np.savez)]
    observation.engine = engine
    labels = ('kernel', 'engine.solve', 'geometry.leg', 'torch.autograd.grad',
              'activation_cotangent_product', 'constraint_values', 'np.savez')
    direct = [(owner, key, key in vars(owner), vars(owner).get(key))
              for owner, key, _ in originals]
    try:
        for (owner, key, original), label in zip(originals, labels):
            if label == 'kernel':
                replacement = observation.kernel_factory(original)
            elif label == 'engine.solve':
                replacement = observation.solve_timed(original)
            elif label in ('geometry.leg', 'torch.autograd.grad'):
                replacement = observation.torch_timed(label, original)
            else:
                replacement = observation.timed(label, original)
            setattr(owner, key, replacement)
        yield
    finally:
        active = sys.exc_info()[1]
        restore_errors = []
        for owner, key, had_direct, direct_value in reversed(direct):
            try:
                if had_direct:
                    setattr(owner, key, direct_value)
                else:
                    delattr(owner, key)
            except Exception as error:
                restore_errors.append((key, error))
                observation.secondary.append({'stage': 'wrapper_restore',
                    'label': key, 'error': repr(error)})
        observation.engine = None
        if restore_errors:
            if active is not None:
                for key, error in restore_errors:
                    active.add_note('Secondary wrapper restore ' + key + ': ' + repr(error))
            else:
                raise restore_errors[0][1]


def one_evaluation(evaluate, engine, fixture, geometry, vector, out, ledger):
    if ledger['original_evaluations_requested'] != 0:
        raise RuntimeError('Original evaluation already requested')
    ledger['original_evaluations_requested'] = 1
    ledger['original_context_ticks_requested'] = 120000
    ledger['original_scalar_mn_ticks_requested'] = 8280000
    ledger['original_edge_stage_ticks_requested'] = 131040000
    ledger['derivative_banks_requested'] = 1
    save_json(out / 'advance-ledger.json', ledger)
    return evaluate(engine, fixture, geometry, 0, vector, 1, out)


def build_result(freeze, observation, comparison):
    if not comparison['accepted']:
        raise ValueError('Raw agreement required before profile interpretation')
    evals = [x for x in observation.host if x['phase'] == 'original'
             and x['label'] == 'evaluate_leg']
    if len(evals) != 1 or not evals[0]['returned'] or not evals[0]['wall_s'] > 0:
        raise ValueError('Exactly one completed original evaluation wall span required')
    eval_wall = evals[0]['wall_s']
    cuda = [x for x in observation.cuda if x['phase'] == 'original']
    expected = {'joint_forward': 1, 'muscle_ema': 1, 'edge_bank': 1,
                'geometry.leg': 3, 'torch.autograd.grad': 1}
    complete = (all(sum(x.get('kernel', x.get('label')) == key for x in cuda) == n
                    for key, n in expected.items())
                and len(cuda) == sum(expected.values())
                and all(x['returned'] and x['enqueued'] and x['end_recorded']
                        and x['synchronized'] and x['elapsed_ms'] is not None
                        and np.isfinite(x['elapsed_ms']) and x['elapsed_ms'] >= 0
                        for x in cuda))
    forward_bank = [x for x in cuda if x.get('kernel') in
                    ('joint_forward', 'edge_bank')]
    measured = (sum(x['elapsed_ms'] for x in forward_bank) / 1000
                if complete else None)
    ratio = eval_wall / freeze['lf_average_wall_s']
    return {
        'schema': 'flylab.m3.e129.fixed-lf-cost-profile.v1',
        'raw_comparison_accepted': True, 'timing_complete': complete,
        'evaluate_leg_wall_s': eval_wall,
        'frozen_lf_unique_average_s': freeze['lf_average_wall_s'],
        'pipeline_to_average_ratio': ratio,
        'pipeline_interpretation': (
            'timing_incomplete_diagnostic_only' if not complete else
            'substantial_one_point_evidence' if ratio >= freeze['ratio_high'] else
            'pipeline_alone_insufficient' if ratio <= freeze['ratio_low'] else
            'inconclusive'),
        'forward_plus_edge_bank_cuda_elapsed_s': measured,
        'forward_bank_to_eval_ratio': measured / eval_wall if measured is not None else None,
        'neural_cuda_improvement_motivated': (
            measured / eval_wall >= freeze['neural_kernel_fraction_threshold']
            if measured is not None else None),
        'cuda_event_scope': ('CuPy private-stream and Torch current-stream elapsed '
                             'spans; not CUPTI complete kernel traffic'),
        'unmeasured_work': ('decoder GEMM/elementwise, library scalar checks, event '
                            'overhead, and other host/device work'),
        'unmeasured_remainder_attribution': 'unknown',
        'fit_calls': 0, 'body_calls': 0, 'F6': False}


def run(freeze_path, freeze_sha, source_review, source_review_sha, out):
    began = time.perf_counter()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('One-shot E129 output exists: ' + str(out))
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'schema': 'flylab.m3.e129.fixed-lf-profile-ledger.v1',
              'pid': os.getpid(), 'freeze_sha256': freeze_sha,
              'source_review_sha256': source_review_sha,
              'original_evaluations_requested': 0,
              'original_fit_calls': 0, 'full_response_batches_requested': 0,
              'synthetic_setup_requested': 0, 'status': 'preflight'}
    save_json(out / 'advance-ledger.json', ledger)
    observation = Observation()
    engine = None
    limits = None
    primary = None
    freeze = None
    comparison = None
    try:
        freeze, e128, review = reviewed_inputs(freeze_path, freeze_sha,
                                                source_review, source_review_sha)
        save_json(out / 'source-review-snapshot.json', review)
        vector = reference_vector(ROOT / freeze['reference_raw_path'])
        limits, threads = fit.one_cpu_thread()
        ledger['cpu_threads'] = threads
        import cupy as cp
        import torch

        from tools.banc_m3_torch_foot_geometry import TorchFootGeometry
        gpu = cp.cuda.runtime.getDeviceProperties(0)['name']
        ledger['environment'] = {'python': sys.version, 'cupy': cp.__version__,
                                 'torch': torch.__version__,
                                 'gpu': gpu.decode() if isinstance(gpu, bytes) else str(gpu)}
        observation.phase = 'setup'
        ledger['synthetic_setup_requested'] = 1
        save_json(out / 'advance-ledger.json', ledger)
        setup_start = time.perf_counter()
        # The explicit E128 tiny setup is the only warmup. It is synthetic.
        original_kernel = neural._kernel
        neural._kernel = observation.kernel_factory(original_kernel)
        tiny_engine = None
        try:
            tiny = fit.synthetic_tiny_setup_problem()
            tiny_engine = observation.timed('tiny_engine_construct', neural.CudaJointAdaptiveEngine)(tiny)
            observation.engine = tiny_engine
            _, tiny_raw, tiny_timing = observation.timed('tiny_engine.solve', tiny_engine.solve)(
                np.zeros(1), np.zeros(1), derivative=True)
            observation.timed('np.savez', np.savez)(
                out / 'cuda-tiny-setup-raw.npz',
                synthetic_stage_current=tiny['stage_current'],
                synthetic_source_rate=tiny['source_rate'], **tiny_raw)
            if not all(np.isfinite(tiny_raw[key]).all() for key in neural.FIELDS):
                raise ValueError('Synthetic tiny setup is nonfinite')
            ledger['tiny_response_progress'] = tiny_timing['progress']
            ledger['tiny_engine_metadata'] = snapshot(tiny_engine.metadata())
            ledger['tiny_raw_sha256'] = sha256(out / 'cuda-tiny-setup-raw.npz')
        except BaseException as error:
            ledger['tiny_response_progress'] = getattr(error, 'response_progress', None)
            if tiny_engine is not None:
                try:
                    ledger['tiny_engine_metadata'] = snapshot(tiny_engine.metadata())
                except Exception as secondary:
                    error.add_note('Secondary tiny engine metadata: ' + repr(secondary))
            else:
                ledger['tiny_fixed_upload_progress'] = getattr(
                    error, 'fixed_upload_progress', None)
            partial = getattr(error, 'partial_response', None)
            if partial:
                try:
                    observation.timed('np.savez', np.savez)(
                        out / 'cuda-tiny-setup-partial-raw.npz', **partial)
                    ledger['tiny_partial_raw_sha256'] = sha256(
                        out / 'cuda-tiny-setup-partial-raw.npz')
                except Exception as secondary:
                    error.add_note('Secondary tiny partial raw save: ' + repr(secondary))
            raise
        finally:
            neural._kernel = original_kernel
            observation.engine = None
        ledger['synthetic_setup_wall_s'] = time.perf_counter() - setup_start
        fixture = observation.timed('frozen_fixture_load', fit.frozen_problem)(e128)
        def load_geometry():
            geometry = TorchFootGeometry.load(ROOT / e128['geometry_path'], device='cuda')
            torch.cuda.synchronize(geometry.device)
            return geometry

        geometry = observation.timed('geometry_fixed_upload', load_geometry)()
        if geometry.geometry_hash != e128['geometry_hash']:
            raise ValueError('Frozen E128 geometry hash changed')
        ledger['geometry_fixed_upload_bytes'] = sum(int(geometry.arrays[key].nbytes)
            for key in ('axes', 'anchors', 'tips', 'reference_angles'))
        motors = np.flatnonzero(fixture['leg_motor_mask'][0])
        muscles = np.flatnonzero(fixture['leg_muscle_mask'][0])
        edges = np.flatnonzero(fixture['leg_edge_mask'][0])
        if (len(motors), len(muscles), len(edges)) != (69, 17, 273):
            raise ValueError('LF subset cardinality changed')
        try:
            engine = observation.timed('leg_engine_fixed_upload', neural.CudaJointAdaptiveEngine)(
                fixture, motors, muscles, edges)
        except BaseException as error:
            ledger['leg_constructor_fixed_upload_progress'] = snapshot(
                getattr(error, 'fixed_upload_progress', None))
            ledger['leg_constructor_fixed_upload_by_key'] = snapshot(
                getattr(error, 'fixed_upload_by_key', None))
            raise
        with np.load(ROOT / freeze['reference_raw_path'], allow_pickle=False) as ref:
            for key, expected in (('motor_indices', engine.problem['motor_indices']),
                                  ('muscle_indices', engine.problem['muscle_indices']),
                                  ('edge_ids', engine.problem['edge_ids']),
                                  ('parameter_muscle_map',
                                   neural.activation_parameter_map(engine.problem))):
                if not np.array_equal(ref[key], expected):
                    raise ValueError('LF explicit ID/mapping changed: ' + key)
        ledger['leg_engine_before'] = snapshot(engine.metadata())
        ledger['status'] = 'ready'
        save_json(out / 'advance-ledger.json', ledger)
        if time.perf_counter() - began >= freeze['internal_wall_limit_s'] - 5:
            raise TimeoutError('E129 preflight exhausted internal reserve')
        observation.phase = 'original'
        with observed_callables(observation, engine, geometry):
            record = one_evaluation(
                observation.timed('evaluate_leg', fit.evaluate_leg),
                engine, fixture, geometry, vector, out, ledger)
        ledger['original_evaluations_completed'] = 1
        ledger['original_solve'] = snapshot(observation.solve)
        ledger['evaluate_record'] = {k: record[k] for k in
            ('index', 'raw_path', 'raw_sha256', 'wall_s')}
        raw_path = Path(record['raw_path'])
        ledger['raw_sha256'] = sha256(raw_path)
        ledger['status'] = 'raw_saved'
        save_json(out / 'advance-ledger.json', ledger)
        # E128 has saved raw before any E129 numeric agreement assertion.
        comparison = compare_raw(raw_path, ROOT / freeze['reference_raw_path'])
        save_json(out / 'raw-comparison.json', comparison)
        if not comparison['accepted']:
            raise ValueError('E129 raw/reference mismatch; timing is diagnostic only')
        if time.perf_counter() - began > freeze['internal_wall_limit_s']:
            raise TimeoutError('E129 internal wall limit exceeded after raw save')
        ledger['status'] = 'compared'
    except BaseException as error:
        primary = error
        ledger['status'] = 'failed'
        ledger['primary_error'] = {'type': type(error).__name__, 'message': str(error),
                                   'traceback': traceback.format_exc(),
                                   'response_progress': getattr(error, 'response_progress', None),
                                   'partial_response_keys': list(getattr(error, 'partial_response', {})),
                                   'notes': list(getattr(error, '__notes__', []))}
        ledger['original_solve'] = snapshot(observation.solve)
        ledger['original_partial_raw'] = []
        for path in out.glob('*partial-raw.npz'):
            try:
                ledger['original_partial_raw'].append(
                    {'path': str(path), 'sha256': sha256(path)})
            except Exception as secondary:
                error.add_note('Secondary partial raw hash: ' + repr(secondary))
    finally:
        def secondary(stage, error):
            nonlocal primary
            item = {'stage': stage, 'error': repr(error)}
            observation.secondary.append(item)
            if primary is not None:
                primary.add_note('Secondary ' + stage + ': ' + repr(error))
            else:
                primary = error

        try:
            observation.finish_cuda()
        except Exception as error:
            secondary('finish_cuda', error)
        ledger['host_spans'] = observation.host
        ledger['cuda_spans'] = observation.cuda
        try:
            ledger['original_solve'] = snapshot(observation.solve)
        except Exception as error:
            secondary('solve_snapshot', error)
        ledger['secondary_errors'] = observation.secondary
        if engine is not None:
            try:
                ledger['leg_engine_after'] = snapshot(engine.metadata())
            except Exception as error:
                secondary('engine_metadata', error)
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:
                secondary('threadpool_restore', error)
        result_path = out / 'result.json'
        if primary is None:
            try:
                payload = build_result(freeze, observation, comparison)
                if time.perf_counter() - began >= freeze['internal_wall_limit_s'] - 1:
                    raise TimeoutError('E129 finalization reserve exhausted')
                save_json(result_path, payload)
                if time.perf_counter() - began > freeze['internal_wall_limit_s']:
                    raise TimeoutError('E129 internal wall limit exceeded in finalization')
                ledger['status'] = ('completed' if payload['timing_complete']
                                    else 'timing_incomplete_diagnostic_only')
            except BaseException as error:
                secondary('result_finalization', error)
                if result_path.exists():
                    try:
                        result_path.unlink()
                    except Exception as removal:
                        secondary('invalid_result_removal', removal)
        if primary is not None:
            ledger['status'] = 'failed'
            ledger['primary_error'] = {
                'type': type(primary).__name__, 'message': str(primary),
                'notes': list(getattr(primary, '__notes__', [])),
                'earlier_traceback': ledger.get('primary_error', {}).get('traceback'),
                'response_progress': getattr(primary, 'response_progress', None)}
        ledger['process_wall_s'] = time.perf_counter() - began
        try:
            save_json(out / 'advance-ledger.json', ledger)
        except Exception as error:
            secondary('final_ledger_save', error)
        if primary is not None and result_path.exists():
            try:
                result_path.unlink()
            except Exception as error:
                secondary('failed_result_removal', error)
        if freeze is not None and time.perf_counter() - began > freeze['internal_wall_limit_s']:
            if primary is None:
                primary = TimeoutError('E129 total internal wall limit exceeded')
            ledger['status'] = 'failed'
            ledger['post_finalization_wall_s'] = time.perf_counter() - began
            if result_path.exists():
                try:
                    result_path.unlink()
                except Exception as error:
                    secondary('timed_out_result_removal', error)
            try:
                save_json(out / 'advance-ledger.json', ledger)
            except Exception as error:
                secondary('timed_out_ledger_save', error)
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
