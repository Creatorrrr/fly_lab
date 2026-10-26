"""Generated-data guards for E129 instrumentation; no original response."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from tools import banc_m3_joint_adaptive_recruitment as neural
from tools import fit_c_banc_m3_joint_adaptive_recruitment as fit
from tools import profile_c_banc_m3_joint_fit_cost as profile


class FakeEngine:
    def solve(self, *args, **kwargs):
        return ('same', args, {'progress': {'stage': 'synthetic'},
                               'kwargs': kwargs})


class FakeGeometry:
    def leg(self, value):
        return value


def test_observation_restores_wrappers_on_exception(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    engine, geometry, obs = FakeEngine(), FakeGeometry(), profile.Observation()
    originals = (neural._kernel, torch.autograd.grad,
                 fit.activation_cotangent_product, fit.constraint_values,
                 np.savez)
    had_solve = 'solve' in vars(engine)
    had_leg = 'leg' in vars(geometry)
    with pytest.raises(ZeroDivisionError), profile.observed_callables(
            obs, engine, geometry):
        assert engine.solve(1, x=2) == ('same', (1,),
                                        {'progress': {'stage': 'synthetic'},
                                         'kwargs': {'x': 2}})
        np.savez(tmp_path / 'synthetic.npz', value=np.array([1]))
        raise ZeroDivisionError('synthetic')
    assert (neural._kernel, torch.autograd.grad,
            fit.activation_cotangent_product, fit.constraint_values,
            np.savez) == originals
    assert ('solve' in vars(engine)) == had_solve
    assert ('leg' in vars(geometry)) == had_leg
    assert obs.engine is None
    assert [x['label'] for x in obs.host] == ['engine.solve', 'np.savez']
    assert all(x['phase'] == 'setup' and x['returned'] for x in obs.host)


def test_cuda_event_phase_success_failure_and_unknown(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            assert stream == 'private'

        def synchronize(self):
            pass

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        Event=FakeEvent, get_elapsed_time=lambda start, end: 12.5))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    obs = profile.Observation()
    obs.engine = types.SimpleNamespace(stream='private')
    observed = obs.kernel_factory(lambda name: lambda *args: args)
    assert observed('joint_forward')(42) == (42,)
    obs.phase = 'original'
    assert observed('edge_bank')(7) == (7,)
    obs.finish_cuda()
    assert [(x['phase'], x['returned'], x['elapsed_ms']) for x in obs.cuda] == [
        ('setup', True, 12.5), ('original', True, 12.5)]
    failing = obs.kernel_factory(lambda name: lambda *args: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        failing('muscle_ema')(3)
    obs.finish_cuda()
    assert obs.cuda[-1]['requested'] and not obs.cuda[-1]['returned']
    assert obs.cuda[-1]['elapsed_ms'] is None


@pytest.mark.parametrize('fault', ['create', 'start', 'end', 'sync', 'callable'])
def test_cupy_event_failures_do_not_change_callable_result(monkeypatch, fault):
    made = []

    class FakeEvent:
        def __init__(self):
            made.append(self)
            if fault == 'create' and len(made) == 1:
                raise RuntimeError('event create')

        def record(self, stream):
            if fault == 'start' and self is made[0]:
                raise RuntimeError('event start')
            if fault == 'end' and self is made[1]:
                raise RuntimeError('event end')

        def synchronize(self):
            if fault == 'sync':
                raise RuntimeError('event synchronize')

    cuda = types.SimpleNamespace(Event=FakeEvent,
        get_elapsed_time=lambda a, b: 2.0)
    monkeypatch.setitem(sys.modules, 'cupy', types.SimpleNamespace(cuda=cuda))
    obs = profile.Observation()
    obs.engine = types.SimpleNamespace(stream='private')

    def callable_kernel(*args):
        if fault == 'callable':
            raise ValueError('original kernel failure')
        return 'original result'

    launch = obs.kernel_factory(lambda name: callable_kernel)('joint_forward')
    if fault == 'callable':
        with pytest.raises(ValueError, match='original kernel failure'):
            launch(1)
    else:
        assert launch(1) == 'original result'
    obs.finish_cuda()
    span = obs.cuda[0]
    assert span['elapsed_ms'] is None
    if fault == 'callable':
        assert span['unknown_requested'] and not span['returned']
    else:
        assert span['returned'] and span['enqueued']
        assert obs.secondary


@pytest.mark.parametrize('fault', [None, 'end', 'sync'])
def test_torch_current_stream_events_and_host_scope(monkeypatch, fault):
    torch = pytest.importorskip('torch')
    made = []

    class FakeEvent:
        def __init__(self, **kwargs):
            assert kwargs == {'enable_timing': True}
            made.append(self)

        def record(self, stream):
            assert stream == 'torch-current'
            if fault == 'end' and self is made[1]:
                raise RuntimeError('end event')

        def synchronize(self):
            if fault == 'sync':
                raise RuntimeError('sync event')

        def elapsed_time(self, end):
            return 4.0

    monkeypatch.setattr(torch.cuda, 'current_stream',
                        lambda device: 'torch-current')
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    obs = profile.Observation()
    tensor = types.SimpleNamespace(device='cuda:0')
    geometry = obs.torch_timed('geometry.leg', lambda value: value)
    assert geometry(tensor) is tensor
    obs.phase = 'original'
    made.clear()
    grad = obs.torch_timed('torch.autograd.grad', lambda value: value)
    assert grad(tensor) is tensor
    obs.finish_cuda()
    assert [x['phase'] for x in obs.cuda] == ['setup', 'original']
    assert [x['label'] for x in obs.host] == ['geometry.leg', 'torch.autograd.grad']
    assert all(x['returned'] and x['wall_s'] is not None for x in obs.host)
    assert obs.cuda[0]['elapsed_ms'] == (4.0 if fault is None else None)
    assert obs.cuda[1]['elapsed_ms'] == (4.0 if fault is None else None)


def test_metadata_snapshot_and_solve_progress_are_detached():
    nested = {'upload': {'requested': {'bytes': 3}}}
    before = profile.snapshot(nested)
    nested['upload']['requested']['bytes'] = 8
    assert before['upload']['requested']['bytes'] == 3
    obs = profile.Observation()
    returned = obs.solve_timed(lambda: (None, None, {
        'progress': {'stage': 'complete'}, 'wall_s': 1.25}))()
    returned[2]['progress']['stage'] = 'later mutated'
    assert obs.solve[0]['response_progress']['stage'] == 'complete'
    error = RuntimeError('partial')
    error.response_progress = {'stage': 'unknown', 'bank_completion_class': 'unknown'}

    def fail():
        raise error

    with pytest.raises(RuntimeError, match='partial'):
        obs.solve_timed(fail)()
    assert obs.solve[1]['response_progress']['bank_completion_class'] == 'unknown'


@pytest.mark.parametrize('end_fails', [False, True])
def test_torch_partial_callable_failure_preserves_primary(monkeypatch, end_fails):
    torch = pytest.importorskip('torch')
    made = []

    class Event:
        def __init__(self, **kwargs):
            made.append(self)

        def record(self, stream):
            if end_fails and self is made[1]:
                raise OSError('secondary end event failure')

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: 'fake')
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    primary = ValueError('original partial Torch failure')

    def fail(tensor):
        raise primary

    obs = profile.Observation()
    with pytest.raises(ValueError) as caught:
        obs.torch_timed('torch.autograd.grad', fail)(
            types.SimpleNamespace(device='cuda:0'))
    assert caught.value is primary
    obs.finish_cuda()
    span = obs.cuda[0]
    assert span['unknown_requested'] and not span['returned']
    assert span['error'] == repr(primary)
    assert span['end_recorded'] is (not end_fails)
    assert span['elapsed_ms'] == (None if end_fails else 1.0)
    assert bool(obs.secondary) is end_fails


def test_incomplete_timing_has_no_negative_bottleneck_verdict():
    obs = profile.Observation()
    obs.host.append({'phase': 'original', 'label': 'evaluate_leg',
                     'returned': True, 'wall_s': 2.0})
    frozen = {'lf_average_wall_s': 2.0, 'ratio_high': .65,
              'ratio_low': .35, 'neural_kernel_fraction_threshold': .5}
    result = profile.build_result(frozen, obs, {'accepted': True})
    assert not result['timing_complete']
    assert result['pipeline_interpretation'] == 'timing_incomplete_diagnostic_only'
    assert result['neural_cuda_improvement_motivated'] is None
    assert result['forward_plus_edge_bank_cuda_elapsed_s'] is None


def test_one_evaluation_ledger_before_call_and_no_retry(tmp_path):
    ledger = {'original_evaluations_requested': 0}
    seen = []

    def evaluator(*args):
        seen.append(json.loads((tmp_path / 'advance-ledger.json').read_text()))
        return {'done': True}

    assert profile.one_evaluation(evaluator, None, None, None,
                                  np.zeros(290), tmp_path, ledger) == {'done': True}
    assert seen[0]['original_evaluations_requested'] == 1
    assert seen[0]['original_context_ticks_requested'] == 120000
    assert seen[0]['original_scalar_mn_ticks_requested'] == 8280000
    assert seen[0]['original_edge_stage_ticks_requested'] == 131040000
    assert seen[0]['derivative_banks_requested'] == 1
    with pytest.raises(RuntimeError, match='already requested'):
        profile.one_evaluation(evaluator, None, None, None,
                               np.zeros(290), tmp_path, ledger)
    assert len(seen) == 1


def test_raw_exists_before_numeric_mismatch_is_reported(tmp_path):
    ledger = {'original_evaluations_requested': 0}
    actual = tmp_path / 'leg-0-evaluation-001-raw.npz'
    reference = tmp_path / 'reference.npz'
    np.savez(reference, losses=np.array([0.0]))

    def evaluator(*args):
        np.savez(actual, losses=np.array([1.0]))
        return {'raw_path': str(actual)}

    profile.one_evaluation(evaluator, None, None, None,
                           np.zeros(290), tmp_path, ledger)
    assert actual.is_file()
    assert not profile.compare_raw(actual, reference)['accepted']
    assert actual.is_file()


def test_reference_vector_and_raw_comparison(tmp_path):
    reference = tmp_path / 'reference.npz'
    actual = tmp_path / 'actual.npz'
    values = {'delta': np.zeros(273), 'gain': np.zeros(17),
              'edge_ids': np.arange(273, dtype=np.int64),
              'motor_indices': np.arange(69, dtype=np.int32),
              'muscle_indices': np.arange(17, dtype=np.int32),
              'parameter_muscle_map': np.arange(290, dtype=np.int32) % 17,
              'branch_bits': np.zeros((1, 2, 3), dtype=np.uint8),
              'timing_context_ticks': np.array(120000),
              'timing_bank_complete': np.array(True)}
    values.update({key: np.array([1.0]) for key in
                   (*profile.NUMERIC_LOOSE, *profile.NUMERIC_TIGHT)})
    np.savez(reference, **values)
    np.savez(actual, **values)
    assert profile.reference_vector(reference).shape == (290,)
    assert profile.compare_raw(actual, reference)['accepted']
    changed = dict(values)
    changed['losses'] = np.array([1.0 + 2e-10])
    np.savez(actual, **changed)
    assert not profile.compare_raw(actual, reference)['accepted']
    changed['losses'] = values['losses']
    changed['branch_bits'] = np.ones((1, 2, 3), dtype=np.uint8)
    np.savez(actual, **changed)
    assert not profile.compare_raw(actual, reference)['accepted']
    changed['branch_bits'] = values['branch_bits']
    changed['gain'] = np.ones(17)
    np.savez(reference, **changed)
    with pytest.raises(ValueError, match='not zero'):
        profile.reference_vector(reference)


def test_exact_freeze_and_review_guards_do_not_load_original(tmp_path, monkeypatch):
    freeze = tmp_path / 'freeze.json'
    review = tmp_path / 'review.json'
    freeze.write_text('{}', encoding='utf-8')
    review.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='exact freeze SHA'):
        profile.reviewed_inputs(freeze, '0' * 64, review, '0' * 64)
    monkeypatch.setattr(profile, 'FREEZE_SHA', profile.sha256(freeze))
    with pytest.raises(ValueError, match='fixed profile contract'):
        profile.reviewed_inputs(freeze, profile.sha256(freeze),
                                review, profile.sha256(review))


def test_review_requires_accepted_exact_two_sources(tmp_path, monkeypatch):
    digest = 'a' * 64
    freeze_path = tmp_path / 'freeze.json'
    review_path = tmp_path / 'review.json'
    e128_path = tmp_path / 'e128.json'
    manifest_path = tmp_path / 'manifest.json'
    e128_path.write_text(json.dumps({'fixture_path': 'fixture',
                                     'input_sha256': {'fixture': digest}}))
    manifest_path.write_text(json.dumps({'original_files': {
        f'file-{i:03d}': digest for i in range(112)}}))
    frozen = {
        'schema': 'flylab.m3.e129.fixed-lf-profile-freeze.v1',
        'design_path': 'design', 'design_sha256': digest,
        'original_evaluation_limit': 1, 'original_fit_calls': 0,
        'leg': 0, 'leg_MN': 69, 'leg_muscles': 17, 'leg_edges': 273,
        'original_context_ticks': 120000, 'original_conditional_s': 30,
        'internal_wall_limit_s': 30, 'external_wall_limit_s': 40,
        'supervisor_limit_s': 45, 'allocated_conservative_end_s': 1440,
        'shared_process_limit_s': 1530,
        'E128_freeze_path': 'e128.json',
        'original_output_manifest_path': 'manifest.json',
        'reference_raw_path': 'reference',
        'reference_leg_summary_path': 'summary',
        'input_sha256': {key: digest for key in
            ('e128.json', 'manifest.json', 'reference', 'summary',
             'other1', 'other2', 'other3')},
        'reused_source_sha256': {key: digest for key in
            ('source1', 'source2', 'source3')},
    }
    freeze_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(profile, 'FREEZE_SHA', digest)
    monkeypatch.setattr(profile, 'sha256', lambda path: digest)
    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    monkeypatch.setattr(profile, 'checked_path',
                        lambda relative, expected: tmp_path / relative)
    monkeypatch.setattr(fit, 'reviewed_inputs', lambda *args: (None, None))
    candidate = {'accepted_for_single_original_fixed_lf_profile': False,
                 'blocking_defects': [], 'freeze_sha256': digest,
                 'source_sha256': dict.fromkeys(profile.SOURCE_FILES, digest)}
    review_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='two-source review'):
        profile.reviewed_inputs(freeze_path, digest, review_path, digest)
    candidate['accepted_for_single_original_fixed_lf_profile'] = True
    candidate['source_sha256'].pop(profile.SOURCE_FILES[1])
    review_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='two-source review'):
        profile.reviewed_inputs(freeze_path, digest, review_path, digest)
    candidate['source_sha256'][profile.SOURCE_FILES[1]] = digest
    review_path.write_text(json.dumps(candidate))
    assert profile.reviewed_inputs(freeze_path, digest, review_path, digest)[2] == candidate


def test_generated_run_restores_threads_when_result_receipt_fails(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    fixture = {'leg_motor_mask': np.ones((1, 69), dtype=bool),
               'leg_muscle_mask': np.ones((1, 17), dtype=bool),
               'leg_edge_mask': np.ones((1, 273), dtype=bool)}
    reference = tmp_path / 'reference.npz'
    np.savez(reference, motor_indices=np.arange(69, dtype=np.int32),
             muscle_indices=np.arange(17, dtype=np.int32),
             edge_ids=np.arange(273, dtype=np.int64),
             parameter_muscle_map=np.arange(290, dtype=np.int32))
    frozen = {'reference_raw_path': 'reference.npz', 'internal_wall_limit_s': 30,
              'lf_average_wall_s': 2.0, 'ratio_high': .65, 'ratio_low': .35,
              'neural_kernel_fraction_threshold': .5}
    restored = []

    class Limits:
        def restore_original_limits(self):
            restored.append(True)

    class Engine:
        def __init__(self, mapping, *subset):
            self.problem = {'motor_indices': np.arange(69, dtype=np.int32),
                            'muscle_indices': np.arange(17, dtype=np.int32),
                            'edge_ids': np.arange(273, dtype=np.int64)}
            self.stream = 'synthetic'
            self.counters = {'requested': {'bytes': 1}}

        def solve(self, *args, **kwargs):
            return {}, {key: np.ones(1) for key in neural.FIELDS}, {
                'progress': {'stage': 'synthetic_complete'}}

        def metadata(self):
            return self.counters

    class Geometry:
        geometry_hash = 'synthetic_hash'
        device = 'synthetic_cuda'
        arrays = dict.fromkeys(('axes', 'anchors', 'tips', 'reference_angles'),
                               np.ones(1))

        @staticmethod
        def load(*args, **kwargs):
            return Geometry()

        def leg(self, value):
            return value

    from tools import banc_m3_torch_foot_geometry as foot

    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    monkeypatch.setattr(profile, 'reviewed_inputs',
                        lambda *args: (frozen,
                            {'geometry_path': 'geometry',
                             'geometry_hash': 'synthetic_hash'}, {}))
    monkeypatch.setattr(profile, 'reference_vector', lambda path: np.zeros(290))
    monkeypatch.setattr(profile, 'compare_raw',
                        lambda *args: {'accepted': True})
    monkeypatch.setattr(fit, 'one_cpu_thread',
                        lambda: (Limits(), {'synthetic': True}))
    monkeypatch.setattr(fit, 'synthetic_tiny_setup_problem',
                        lambda: {'stage_current': np.zeros((1, 80, 1)),
                                 'source_rate': np.zeros((1, 80, 1))})
    monkeypatch.setattr(fit, 'frozen_problem', lambda arg: fixture)
    monkeypatch.setattr(fit, 'evaluate_leg',
                        lambda *args: _fake_saved_eval(args[0], args[-1]))
    monkeypatch.setattr(neural, 'CudaJointAdaptiveEngine', Engine)
    monkeypatch.setattr(neural, 'activation_parameter_map',
                        lambda problem: np.arange(290, dtype=np.int32))
    monkeypatch.setattr(foot, 'TorchFootGeometry', Geometry)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    gpu = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDeviceProperties=lambda index: {'name': b'Synthetic'}))
    monkeypatch.setitem(sys.modules, 'cupy',
                        types.SimpleNamespace(cuda=gpu, __version__='fake'))
    real_save = profile.save_json

    def fail_result(path, obj):
        if path.name == 'result.json':
            raise OSError('synthetic result receipt failure')
        return real_save(path, obj)

    monkeypatch.setattr(profile, 'save_json', fail_result)
    out = tmp_path / 'out'
    with pytest.raises(OSError, match='synthetic result receipt failure'):
        profile.run(tmp_path / 'freeze', 'sha', tmp_path / 'review', 'sha', out)
    assert restored == [True]
    assert not (out / 'result.json').exists()
    ledger = json.loads((out / 'advance-ledger.json').read_text())
    assert ledger['status'] == 'failed'
    assert ledger['original_evaluations_requested'] == 1
    assert ledger['original_evaluations_completed'] == 1
    assert ledger['original_solve'][0]['response_progress']['stage'] == 'synthetic_complete'
    assert (out / 'leg-0-evaluation-001-raw.npz').exists()


def _fake_saved_eval(engine, out):
    engine.solve(np.zeros(273), np.zeros(17), derivative=True)
    path = out / 'leg-0-evaluation-001-raw.npz'
    np.savez(path, synthetic=np.ones(1))
    return {'index': 1, 'raw_path': str(path),
            'raw_sha256': profile.sha256(path), 'wall_s': .1}
