"""E113: one reviewed E112b efficacy transfer to the original whole BANC/body.

This is a native diagnostic, not an E112b conditional-shape pass or M3 adoption.
Run only after the parent first/second source reviews authorize the single trial.
"""

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from flylab.c.walking_trace import capture_control, contact_diagnostics
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial, patch_tau
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.prepare_c_banc_m3_e99_lf_tarsal import source_screen as lf_source_screen
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks
from tools.probe_c_banc_m3_tonic_phase import screens as source_phase_screens
from tools.verify_c_banc_observation import exact

EXPERIMENT = 'M3-E113-dense-foot-native-body'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PAIRS = ('lf-rf', 'lm-rm', 'lh-rh')
PRIOR_WALL_S = 450.0
PREPARE_LIMIT_S = 40.0
BODY_LIMIT_S = 140.0
PROCESS_LIMIT_S = 180.0
SHARED_LIMIT_S = 630.0
CONTROL_DT_S = .005
NEURAL_DT_S = .00025
PHYSICAL_DT_S = .0001


def jsonable(value):
    """Preserve missing/nonfinite as JSON null, never an implicit passing zero."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def save_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def checked_ids(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if (a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or a.shape != b.shape or not np.array_equal(a, b)):
        raise ValueError(label)


def reviewed_inputs(root):
    e112 = root/'e112-dense-foot-recruitment-v2'
    design = Path(__file__).resolve().parents[1]/'docs/BANC_M3_E113_NATIVE_BODY_DESIGN_20260920.md'
    if sha256(design) != 'f44f1cc0fa2929f51639d3eecea38bc4fd6584a842b9bdddf0dfb0b5f31582d8':
        raise ValueError('Frozen E113 native design changed')
    if sha256(root/'parent-e113-expected-inputs.json') != (
            '96789069a39aaf4192becc0b4491c465a871691ca89bfd0de00509c04026984a'):
        raise ValueError('Frozen E113 parent input manifest changed')
    parent = json.loads((root/'parent-e113-expected-inputs.json').read_text(encoding='utf-8'))
    for relative, digest in parent['source_files_sha256'].items():
        if sha256(root/relative) != digest:
            raise ValueError('Parent E113 input changed: '+relative)
    for relative, digest in parent['fixture_sha256'].items():
        if sha256(root/relative) != digest:
            raise ValueError('Parent E113 fixture changed: '+relative)
    review = json.loads((e112/'parent-final-review.json').read_text(encoding='utf-8'))
    manifest = json.loads((e112/'parent-output-manifest.json').read_text(encoding='utf-8'))
    result = json.loads((e112/'result.json').read_text(encoding='utf-8'))
    hashes = manifest.get('original_fit_artifact_sha256')
    if (review.get('accepted_as_valid_conditional_fit') is not True
            or review.get('blocking_defects') != []
            or manifest.get('status') != 'accepted_original_fit_output_manifest'
            or manifest.get('original_count') != 59
            or manifest.get('mismatches') != []
            or manifest.get('parent_final_review_sha256') != sha256(
                e112/'parent-final-review.json')
            or not isinstance(hashes, dict) or len(hashes) != 59):
        raise ValueError('Accepted E112b review and original 59 fit artifacts required')
    for relative, digest in hashes.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or '..' in Path(relative).parts or sha256(e112/relative) != digest):
            raise ValueError('Original E112b artifact changed: '+str(relative))
    if (result.get('validity') != 'valid_conditional_fit'
            or result.get('local_supported') is not False
            or result.get('conditional_only') is not True
            or result.get('body_validated') is not False
            or result.get('original99_startup_screen', {}).get('supported') is not True
            or result.get('inner99_startup_screen', {}).get('supported') is not True
            or result.get('dense_foot_screen', {}).get('supported') is not False
            or result.get('relative_shape_screen', {}).get('supported') is not False
            or result.get('absolute_E101_shape_screen', {}).get('supported') is not False):
        raise ValueError('E112b original99-pass/shape-fail conditional result required')
    baseline = load_npz(root/'e110-e109-native-body-diagnostic/calibration.npz')
    fitted = load_npz(e112/'calibration.npz')
    selected = load_npz(e112/'selected-coefficients.npz')
    prediction = load_npz(e112/'prediction.npz')
    train = load_npz(root/'parent-e112-training-inputs.npz')
    parent_cal = load_npz(root/'parent-e113-expected-calibration.npz')
    parent_obs = load_npz(root/'parent-e113-expected-observation.npz')
    startup = load_npz(root/'e103-startup-preserving-recruitment/startup-inputs.npz')
    for key, value in (('startup_target_command', 'target_command'),
                       ('startup_target_tendon', 'target_tendon')):
        exact(train[key], startup[value], 'Original E100/E103 startup target '+key)
    if (set(baseline) != set(fitted) or set(fitted) != set(parent_cal)
            or fitted['theta'].shape != (96, 48)
            or len(fitted['tau_ids']) != 38 or len(fitted['motor_ids']) != 391
            or len(fitted['individual_edge_ids']) != 1572
            or len(fitted['individual_source_basis_ids']) != 49):
        raise ValueError('Original E110/E112b calibration dimensions changed')
    for key in baseline:
        if key not in ('values', 'individual_edge_efficacies'):
            exact(baseline[key], fitted[key], 'E112b unchanged E110 calibration '+key)
        exact(fitted[key], parent_cal[key], 'Independent E113 calibration '+key)
    checked_ids(selected['edge_ids'], fitted['individual_edge_ids'],
                'E112b selected unsorted edge order')
    checked_ids(train['edge_ids'], selected['edge_ids'], 'E112b training edge order')
    checked_ids(selected['edge_rows'], train['edge_rows'], 'E112b motor row order')
    checked_ids(selected['edge_cols'], train['edge_cols'], 'E112b source column order')
    exact(selected['efficacy'], fitted['individual_edge_efficacies'],
          'E112b selected individual efficacies')
    exact(prediction['efficacy'], selected['efficacy'], 'E112b prediction efficacy')
    exact(selected['original_signed_weights'], train['original_signed_edge_weights'],
          'Original E110 signed selected edge weights')
    if (not np.isfinite(selected['efficacy']).all()
            or np.any((selected['efficacy'] < .001) | (selected['efficacy'] > 10))):
        raise ValueError('Original E112b selected efficacy bounds')
    positions = {int(edge): index for index, edge in enumerate(fitted['edge_ids'])}
    if len(positions) != len(fitted['edge_ids']):
        raise ValueError('Duplicate calibration edge IDs')
    mapped = np.asarray([positions[int(edge)] for edge in selected['edge_ids']])
    exact(fitted['source_values'][mapped], selected['original_signed_weights'],
          'E112b selected original source weights by explicit edge ID map')
    exact(fitted['values'][mapped], np.asarray(
        selected['original_signed_weights'].astype(float)*selected['efficacy'],
        dtype=np.float32), 'E112b selected actual float32 signed weights')
    exact(fitted['tau_ids'], parent_obs['original_tau_ids'], 'Original38 tau IDs')
    exact(fitted['tau_values'], parent_obs['original_tau_values'], 'Original38 tau values')
    checked_ids(fitted['individual_source_basis_ids'], parent_obs['original_source_basis'],
                'Original49 source basis')
    checked_ids(fitted['motor_ids'], parent_obs['motor_ids'], 'Original391 motor order')
    exact(fitted['tau_values'], baseline['tau_values'], 'E110/E112b native tau values')
    if (prediction['command'].shape != (2001, 42)
            or prediction['tendon'].shape != (2001, 6)
            or prediction['actual_float32_source_weights'].shape != (391, 49)):
        raise ValueError('Full E112b conditional prediction required')
    return fitted, baseline, selected, prediction, parent_obs, parent


def prepare(root, graph_path, out, began):
    cal, baseline, selected, prediction, parent_obs, parent = reviewed_inputs(root)
    graph = GraphStore.load(graph_path)
    if graph.n != 158706 or graph.indptr[-1] != len(graph.indices):
        raise ValueError('Original full BANC graph required')
    original = StateStore.load(root/'e32-core-adaptation/candidate/start')
    e105 = root/'e105-startup-native-body-diagnostic-v2'
    e105_start = StateStore.load(e105/'candidate-neutral')
    exact(original['body'], e105_start['body'], 'Original E32 body/world')
    exact(original['world'], e105_start['world'], 'Original E32 world')
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        exact(original['network'][key], e105_start['network'][key], 'E32/E105 '+key)
    if original['network']['tick'] != 0 or original['control_tick'] != 0:
        raise ValueError('Original whole-network zero start required')
    full_tau = patch_tau(np.full(graph.n, .02, np.float32), cal['tau_ids'], cal['tau_values'])
    baseline_tau = patch_tau(np.full(graph.n, .02, np.float32),
                             baseline['tau_ids'], baseline['tau_values'])
    exact(full_tau, baseline_tau, 'Actual full158706 E110/E112b tau equivalence')
    exact(full_tau[cal['motor_ids']], parent_obs['native_motor_tau'],
          'Independent original MN tau')

    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base,
                                                                        cal['motor_ids'])
    before = patch_weights(base, baseline['edge_ids'], baseline['source_values'],
                           baseline['values'])
    after = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    edges = cal['individual_edge_ids']
    checked_ids(np.searchsorted(graph.indptr, edges, side='right')-1,
                cal['individual_edge_motor_ids'], 'Original selected receiving MN rows')
    checked_ids(graph.indices[edges], cal['individual_edge_source_ids'],
                'Original selected49 source columns')
    checked_ids(cal['individual_edge_motor_ids'], cal['motor_ids'][selected['edge_rows']],
                'E112b selected motor row order')
    checked_ids(cal['individual_edge_source_ids'],
                cal['individual_source_basis_ids'][selected['edge_cols']],
                'E112b selected source column order')
    exact(after.indptr, before.indptr, 'Original CSR row topology')
    exact(after.indices, before.indices, 'Original CSR column topology')
    other = np.ones(len(after.data), dtype=bool)
    other[edges] = False
    exact(after.data[other], before.data[other], 'All nonfit/source receiving weights')
    exact(np.signbit(after.data[edges]), np.signbit(base.data[edges]),
          'Original selected synapse signs')
    exact(base.data[edges], selected['original_signed_weights'],
          'Original source-to-MN signed float32 edges')
    exact(after.data[edges], np.asarray(base.data[edges].astype(float)
        * selected['efficacy'], dtype=np.float32), 'Selected actual signed float32 edges')
    dense = after[cal['motor_ids']][:, cal['individual_source_basis_ids']].toarray().astype(float)
    exact(dense, prediction['actual_float32_source_weights'], 'E112b actual dense W')
    exact(dense, parent_obs['actual_float32_source_weights'], 'Independent E113 dense W')

    np.savez_compressed(out/'calibration.npz', **cal)
    anatomy = json.loads((root/'e110-e109-native-body-diagnostic/muscle-provenance.json')
        .read_text(encoding='utf-8'))
    save_json(out/'muscle-provenance.json', anatomy)
    prepare_neutral_checkpoint(graph, original, cal, anatomy, after, full_tau, out)
    elapsed = time.perf_counter()-began
    preparation = {'validity': 'valid_prepared_E112b_body_diagnostic'
        if elapsed <= PREPARE_LIMIT_S else 'resource_limit_incomplete',
        'status': 'conditional_shape_failed', 'E112b_local_supported': False,
        'only_changed_calibration_fields': ['values', 'individual_edge_efficacies'],
        'full_158706_tau_unchanged': True, 'all_nonfit_weights_exact': True,
        'CSR_and_sign_exact': True, 'neutral_disk_exact': True,
        'actual_neural_s': 0.0, 'actual_physical_s': 0.0,
        'fit_calls': 0, 'wall_s': elapsed,
        'resource_limit_met': elapsed <= PREPARE_LIMIT_S,
        'parent_fixture_sha256': parent['fixture_sha256']}
    save_json(out/'preparation-result.json', preparation)
    return graph, cal, prediction, parent_obs, preparation


def prepare_neutral_checkpoint(graph, original, cal, anatomy, expected_weights,
                               full_tau, out):
    """Own and close every preparation trial on restore/persist failures."""
    session = None
    clone = None
    base_trial = None
    try:
        base_trial = EfficacyTrial(graph, original, cal['edge_ids'], cal['source_values'],
            cal['values'], zero_adaptation_ids=cal['zero_adaptation_ids'])
        recruited = RecruitmentTrial(base_trial, cal, anatomy['muscles'])
        session = IntrinsicRecruitmentTrial(recruited, cal['tau_ids'], cal['tau_values'])
        base_trial = None
        if session.network.device != 'cuda':
            raise ValueError('Actual CUDA full BANC required')
        actual = source_matrix(session.network)
        exact(actual.indptr, expected_weights.indptr, 'Runtime CSR row topology')
        exact(actual.indices, expected_weights.indices, 'Runtime CSR column topology')
        exact(actual.data, expected_weights.data, 'Runtime all graph weights')
        exact(session.network.tau.cpu().numpy(), full_tau, 'Runtime actual full158706 tau')
        neutral = session.snapshot()
        exact(neutral['body'], original['body'], 'E113 original body/world neutral')
        exact(neutral['world'], original['world'], 'E113 original world')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(neutral['network'][key], original['network'][key],
                  'E113 original neural '+key)
        StateStore.save(out/'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out/'candidate-neutral'))
        exact(neutral, clone.snapshot(), 'Whole neutral disk exact')
    finally:
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        for label, owned in (('clone', clone), ('main', session), ('base', base_trial)):
            if owned is None:
                continue
            try:
                owned.close()
            except Exception as error:  # noqa: BLE001 - close the other owned trials
                cleanup_errors.append((label, error))
        if cleanup_errors:
            description = '; '.join(f'{label}: {type(error).__name__}: {error}'
                                    for label, error in cleanup_errors)
            if original_error is not None:
                original_error.add_note('E113 preparation cleanup: '+description)
            try:
                save_json(out/'preparation-cleanup-errors.json', [
                    {'owner': label, 'error': type(error).__name__, 'message': str(error)}
                    for label, error in cleanup_errors])
            except Exception as persistence_error:  # noqa: BLE001 - retain first failure
                if original_error is not None:
                    original_error.add_note('Cleanup report failed: '+str(persistence_error))
                else:
                    cleanup_errors[0][1].add_note(
                        'Cleanup report failed: '+str(persistence_error))
            if original_error is None:
                raise cleanup_errors[0][1]


class NativeHealthStop(RuntimeError):
    """A measured 100 us fault; the current control is deliberately incomplete."""


def observation_ids(graph, cal, expected):
    source = np.r_[cal['core_ids'], cal['intrinsic_tonic_ids'], cal['coupling_ids']]
    extra = np.setdiff1d(cal['input_ids'], source)
    basic = np.r_[source, extra, cal['motor_ids']]
    dns = np.asarray(graph.resolve(['flywire:banc:888:720575941500851362',
                                    'flywire:banc:888:720575941626500746']))
    basic = np.r_[basic, dns]
    relay_roots = ('720575941526736740', '720575941518807468',
        '720575941390358788', '720575941504444354',
        '720575941626267786', '720575941613448668',
        '720575941595919217', '720575941511186508')
    relay = np.asarray(graph.resolve(['flywire:banc:888:'+root for root in relay_roots]))
    detailed = np.r_[basic, np.setdiff1d(relay, basic)]
    metadata_sources = np.r_[basic[:-393], dns, np.setdiff1d(relay, basic)]
    for actual, key, length in ((basic, 'basic_ids', 442),
                                (detailed, 'detailed_ids', 446),
                                (metadata_sources, 'metadata_source_ids', 55)):
        if len(actual) != length or len(np.unique(actual)) != length:
            raise ValueError('E113 exact observation cardinality: '+key)
        checked_ids(actual, expected[key], 'E113 exact observation order '+key)
    checked_ids(cal['motor_ids'], expected['motor_ids'], 'E113 original391 motor order')
    return basic, detailed, metadata_sources, dns


def install_native_observer(body, samples):
    """Read contact state on a separate MjData; raise on the first bad step."""
    native_step = body._step_physics
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    integration_type = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, integration_type))

    def step():
        native_step()
        body.mj.mj_getState(body.m, body.d, state, integration_type)
        body.mj.mj_setState(body.m, shadow.d, state, integration_type)
        body.mj.mj_forward(body.m, shadow.d)
        feet = contact_diagnostics(shadow)
        _, rotation, _ = shadow.pose()
        loads = np.asarray(feet['normal_load_bw'], dtype=float)
        slip = np.asarray([np.nan if value is None else value
                           for value in feet['slip_mm_s']], dtype=float)
        nonfoot = bool(shadow.nonfoot_contact())
        upright = float(rotation[2, 2])
        samples.append((float(body.physics_time()), loads.copy(), slip.copy(),
                        np.isfinite(slip), nonfoot, upright))
        if (not np.isfinite(loads).all() or not math.isfinite(upright)
                or not np.isfinite(body.d.qpos).all()
                or not np.isfinite(body.d.qvel).all()
                or nonfoot or upright < .9 or body.fault):
            raise NativeHealthStop('Native 100us health guard at '+str(body.physics_time()))

    body._step_physics = step
    return lambda: setattr(body, '_step_physics', native_step)


def native_arrays(samples):
    if not samples:
        return {'time_s': np.empty(0), 'normal_load_bw': np.empty((0, 6)),
            'slip_mm_s': np.empty((0, 6)),
            'slip_valid': np.empty((0, 6), dtype=bool),
            'nonfoot_contact': np.empty(0, dtype=bool), 'upright': np.empty(0)}
    return {'time_s': np.asarray([r[0] for r in samples]),
        'normal_load_bw': np.asarray([r[1] for r in samples]),
        'slip_mm_s': np.asarray([r[2] for r in samples]),
        'slip_valid': np.asarray([r[3] for r in samples]),
        'nonfoot_contact': np.asarray([r[4] for r in samples]),
        'upright': np.asarray([r[5] for r in samples])}


class AdvanceLedger:
    """Charge real tick/physics deltas, including a failed partial control."""

    def __init__(self):
        self.events = []
        self.neural_s = 0.0
        self.physical_s = 0.0
        self.clock_valid = True

    def advance(self, session, controls, owner):
        before_tick = int(session.network.tick)
        before_physics = float(session.body.physics_time())
        event = {'owner': owner, 'requested_controls': controls, 'started': True,
            'control_tick_before': int(session.control_tick),
            'control_tick_after': None, 'neural_tick_before': before_tick,
            'neural_tick_after': None, 'physics_time_before_s': before_physics,
            'physics_time_after_s': None, 'actual_neural_s': None,
            'actual_physical_s': None, 'clock_valid': False,
            'advance_error': None, 'clock_read_errors': []}
        self.events.append(event)
        clock_error = None
        try:
            session.advance(controls)
        except Exception as exc:
            event['advance_error'] = type(exc).__name__
            raise
        finally:
            after_tick = None
            after_physics = None
            try:
                after_tick = int(session.network.tick)
            except Exception as error:  # noqa: BLE001 - independent clock receipt
                clock_error = error
                event['clock_read_errors'].append('neural: '+str(error))
            try:
                after_physics = float(session.body.physics_time())
            except Exception as error:  # noqa: BLE001 - independent clock receipt
                clock_error = clock_error or error
                event['clock_read_errors'].append('physics: '+str(error))
            try:
                event['control_tick_after'] = int(session.control_tick)
            except Exception as error:  # noqa: BLE001 - retain available physical cost
                clock_error = clock_error or error
                event['clock_read_errors'].append('control: '+str(error))
            neural_valid = after_tick is not None and after_tick >= before_tick
            physical_valid = (after_physics is not None
                and math.isfinite(after_physics-before_physics)
                and after_physics-before_physics >= -1e-9)
            self.clock_valid &= bool(neural_valid and physical_valid)
            if neural_valid and self.neural_s is not None:
                event['actual_neural_s'] = (after_tick-before_tick)*NEURAL_DT_S
                self.neural_s += event['actual_neural_s']
            else:
                self.neural_s = None
            if physical_valid and self.physical_s is not None:
                event['actual_physical_s'] = after_physics-before_physics
                self.physical_s += max(0.0, event['actual_physical_s'])
            else:
                self.physical_s = None
            event['neural_tick_after'] = after_tick
            event['physics_time_after_s'] = after_physics
            event['clock_valid'] = bool(neural_valid and physical_valid)
            if clock_error is not None and event['advance_error'] is None:
                raise clock_error

    def by_owner(self):
        result = {}
        for owner in ('main', 'clone'):
            subset = [event for event in self.events if event['owner'] == owner]
            result[owner] = {'actual_neural_s': sum(
                event['actual_neural_s'] for event in subset)
                if all(event['actual_neural_s'] is not None for event in subset) else None,
                'actual_physical_s': sum(event['actual_physical_s'] for event in subset)
                if all(event['actual_physical_s'] is not None for event in subset) else None,
                'advance_calls': len(subset),
                'incomplete_or_failed_calls': sum(
                    event['advance_error'] is not None for event in subset)}
        return result


def save_raw(out, graph, neutral, session, rows, basic, detailed, dns,
             metadata_sources, signal_rows, native, clone_native, ledger, spec):
    """Persist observed prefixes before any analytic assertion or gait screen."""
    errors = []

    def attempt(label, operation):
        try:
            operation()
        except Exception as error:  # noqa: BLE001 - try every independent raw artifact
            errors.append({'artifact': label, 'error': type(error).__name__,
                           'message': str(error), 'traceback': traceback.format_exc()})

    attempt('clock-events.json', lambda: save_json(out/'clock-events.json', ledger.events))
    attempt('clock-summary.json', lambda: save_json(out/'clock-summary.json', {
        'actual_neural_s': ledger.neural_s,
        'actual_physical_s': ledger.physical_s, 'clock_valid': ledger.clock_valid,
        'actual_costs_by_owner': ledger.by_owner(),
        'main_completed_controls': len(rows)-1,
        'main_native_samples': len(native), 'clone_native_samples': len(clone_native),
        'incomplete_control_possible': bool(ledger.events and (
            ledger.events[-1]['advance_error'] is not None
            or ledger.events[-1]['clock_read_errors']))}))
    attempt('native-contact.npz', lambda: np.savez_compressed(
        out/'native-contact.npz', **native_arrays(native)))
    attempt('clone-native-contact.npz', lambda: np.savez_compressed(
        out/'clone-native-contact.npz', **native_arrays(clone_native)))

    def trace():
        save_json(out/'trace.json', {'schema': 'flylab.m3-e113-native-body-trace.v1',
            'controller': 'E112b fixed efficacy, full BANC native sensory/body feedback',
            'graph_hash': graph.hash, 'model_hash': session.network.identity,
            'body_hash': session.body.model_hash, 'adapter_hash': session.adapter.identity,
            'parameters': {**neutral['parameters'], 'diagnostic': spec}, 'samples': rows})

    attempt('trace.json', trace)
    if basic is None or detailed is None or dns is None:
        return errors

    def details():
        n = len(signal_rows)
        times = np.asarray([row['time_s'] for row in rows[1:n+1]])
        values = {key: np.asarray([row[key] for row in signal_rows]) for key in
                  ('rate', 'recurrent_current', 'drive', 'adaptation', 'muscle_activation')}
        return n, times, values

    def basic_signals():
        n, times, values = details()
        if n:
            columns = np.asarray([int(np.flatnonzero(detailed == cell)[0])
                                  for cell in basic])
            np.savez_compressed(out/'signals.npz', time_s=times, observed_ids=basic,
                rate=values['rate'][:, columns], DN_ids=dns,
                DN_drive=values['drive'][:, columns[-2:]],
                DN_recurrent_current=values['recurrent_current'][:, columns[-2:]],
                DN_adaptation=values['adaptation'][:, columns[-2:]])
        else:
            np.savez_compressed(out/'signals.npz', time_s=np.empty(0), observed_ids=basic,
                rate=np.empty((0, 442), np.float32), DN_ids=dns,
                DN_drive=np.empty((0, 2), np.float32),
                DN_recurrent_current=np.empty((0, 2), np.float32),
                DN_adaptation=np.empty((0, 2), np.float32))

    def detailed_signals():
        n, times, values = details()
        if n:
            np.savez_compressed(out/'recruitment-signals.npz', time_s=times,
                observed_ids=detailed, **values)
        else:
            np.savez_compressed(out/'recruitment-signals.npz', time_s=np.empty(0),
                observed_ids=detailed, rate=np.empty((0, 446), np.float32),
                recurrent_current=np.empty((0, 446), np.float32),
                drive=np.empty((0, 446), np.float32),
                adaptation=np.empty((0, 446), np.float32),
                muscle_activation=np.empty((0, 96)))

    attempt('signals.npz', basic_signals)
    attempt('recruitment-signals.npz', detailed_signals)
    return errors


def finite_error(actual, target):
    a, b = np.asarray(actual, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'RMS': None, 'max_abs': None, 'available': False}
    with np.errstate(over='ignore', invalid='ignore'):
        difference = a-b
        rms = float(np.sqrt(np.mean(difference*difference)))
        maximum = float(np.max(np.abs(difference)))
    if not np.isfinite((rms, maximum)).all():
        return {'RMS': None, 'max_abs': None, 'available': False}
    return {'RMS': rms, 'max_abs': maximum, 'available': True}


def startup_diagnostic(rows, native, prediction, startup):
    """A missing/overflowed original99 prefix is unavailable, never zero error."""
    observed = max(0, min(len(rows)-1, 99))
    times = np.asarray([row['time_s'] for row in rows[1:observed+1]])
    q = np.asarray([row['joint_targets_rad'] for row in rows[1:observed+1]], dtype=float)
    tendon_order = tuple(f'{leg}_tarsus' for leg in LEGS)
    ltm = np.asarray([[row['tendon_inputs'][key] for key in tendon_order]
                      for row in rows[1:observed+1]], dtype=float)
    q = q.reshape((observed, 42)) if q.size == observed*42 else q
    ltm = ltm.reshape((observed, 6)) if ltm.size == observed*6 else ltm
    complete = bool(observed == 99 and len(native) >= 4950
        and np.array_equal(times, np.arange(1, 100)*.005)
        and q.shape == (99, 42) and ltm.shape == (99, 6)
        and prediction['command'].shape == (2001, 42)
        and prediction['tendon'].shape == (2001, 6)
        and startup['target_command'].shape == (99, 42)
        and startup['target_tendon'].shape == (99, 6))
    transferred_q = finite_error(q, prediction['command'][1:observed+1]) if complete else {
        'RMS': None, 'max_abs': None, 'available': False}
    transferred_ltm = finite_error(ltm, prediction['tendon'][1:observed+1]) if complete else {
        'RMS': None, 'max_abs': None, 'available': False}
    original_q = finite_error(q, startup['target_command']) if complete else {
        'RMS': None, 'max_abs': None, 'available': False}
    original_ltm = finite_error(ltm, startup['target_tendon']) if complete else {
        'RMS': None, 'max_abs': None, 'available': False}
    transfer = bool(complete and all((m['available'] and m['RMS'] <= .002
        and m['max_abs'] <= .01) for m in (transferred_q, transferred_ltm)))
    original_pass = bool(complete and original_q['available'] and original_ltm['available']
        and original_q['RMS'] <= .02 and original_q['max_abs'] <= .05
        and original_ltm['RMS'] <= .01 and original_ltm['max_abs'] <= .03)
    return {'complete_original99_prefix': complete, 'observed_controls': observed,
        'E112b_conditional_q': transferred_q, 'E112b_conditional_LTM': transferred_ltm,
        'E112b_conditional_transfer_screen': transfer,
        'original_E100_E103_q': original_q, 'original_E100_E103_LTM': original_ltm,
        'original_E100_E103_startup_screen': original_pass,
        'tendon_order': list(tendon_order)}


def supported_slip(native, *, complete):
    if not complete:
        return {'load_weighted_mm_s': None, 'support_samples': None,
                'support_load_sum_bw': None, 'available': False}
    times, loads, slip, valid = (native[key] for key in
        ('time_s', 'normal_load_bw', 'slip_mm_s', 'slip_valid'))
    if (times.shape != (100000,) or loads.shape != (100000, 6)
            or slip.shape != (100000, 6) or valid.shape != (100000, 6)
            or not np.isfinite(times).all() or not np.isfinite(loads).all()):
        return {'load_weighted_mm_s': None, 'support_samples': None,
                'support_load_sum_bw': None, 'available': False}
    supported = ((times >= .5) & (times < 10))[:, None] & (loads >= .02)
    selected = supported & valid
    if (not selected.any() or np.any(supported & ~valid)
            or not np.isfinite(slip[selected]).all()):
        return {'load_weighted_mm_s': None, 'support_samples': None,
                'support_load_sum_bw': None, 'available': False}
    weight = float(np.sum(loads[selected]))
    value = float(np.sum(loads[selected]*slip[selected])/weight) if weight > 0 else math.nan
    if not math.isfinite(value):
        return {'load_weighted_mm_s': None, 'support_samples': None,
                'support_load_sum_bw': None, 'available': False}
    return {'load_weighted_mm_s': value, 'support_samples': int(selected.sum()),
            'support_load_sum_bw': weight, 'available': True}


def body_quality(rows, native, quality, *, reason, fault, clock_valid, future_exact):
    """Frozen F3-F5 use the original evaluator, full window and native health."""
    complete = bool(reason == 'duration' and len(rows) == 2001
        and len(native) == 100000 and clock_valid
        and not fault and np.array_equal(
            np.asarray([row['time_s'] for row in rows]), np.arange(2001)*.005))
    native_data = native_arrays(native)
    health = bool(complete
        and np.allclose(native_data['time_s'], np.arange(1, 100001)*.0001,
                        rtol=0, atol=2e-8)
        and np.isfinite(native_data['normal_load_bw']).all()
        and not native_data['nonfoot_contact'].any()
        and np.isfinite(native_data['upright']).all()
        and float(np.min(native_data['upright'])) >= .9
        and all(not row.get('fault') for row in rows))
    empty = {'fixed_initial_heading_net_forward_mm': None,
        'net_displacement_mm': None, 'planar_path_mm': None, 'net_over_path': None,
        'forward_eligible_complete_cycles': {leg: None for leg in LEGS},
        'physical_LR': {pair: {'mean_cycles': None, 'resultant_length': None,
                              'samples': None} for pair in PAIRS},
        'knee_width_deg': {leg: None for leg in LEGS},
        'period_cv': {leg: None for leg in LEGS}}
    if complete and isinstance(quality, dict):
        times = np.asarray([row['time_s'] for row in rows])
        position = np.asarray([row['body']['position'] for row in rows], dtype=float)
        forward = np.asarray(rows[0]['body']['basis'], dtype=float)[[0, 2], 0]
        window = (times >= .5) & (times < 10)
        if (position.shape == (2001, 3) and int(window.sum()) == 1900
                and np.isfinite(position).all() and np.isfinite(forward).all()
                and np.linalg.norm(forward) > 0):
            forward = forward/np.linalg.norm(forward)
            planar = position[window][:, [0, 2]]
            delta = planar[-1]-planar[0]
            path = float(np.linalg.norm(np.diff(planar, axis=0), axis=1).sum())
            net = float(np.linalg.norm(delta))
            direct = float(delta@forward)
            ratio = net/path if path > 0 else None
            if np.isfinite((direct, net, path)).all() and (
                    ratio is None or math.isfinite(ratio)):
                empty.update(fixed_initial_heading_net_forward_mm=direct,
                    net_displacement_mm=net, planar_path_mm=path, net_over_path=ratio)
        events = quality.get('events', {})
        pairs = quality.get('pair_phase', {})
        knees = quality.get('knee_interior_deg', {})
        for leg in LEGS:
            item = events.get(leg, {})
            count = item.get('complete_cycles')
            cv = item.get('period_cv')
            empty['forward_eligible_complete_cycles'][leg] = count if (
                isinstance(count, int) and count >= 0) else None
            empty['period_cv'][leg] = cv if (
                cv is not None and math.isfinite(cv)) else None
            knee = knees.get(leg)
            if knee and knee.get('p95') is not None and knee.get('p05') is not None:
                width = knee['p95']-knee['p05']
                empty['knee_width_deg'][leg] = width if math.isfinite(width) else None
        for pair in PAIRS:
            item = pairs.get(pair, {})
            empty['physical_LR'][pair] = {key: value if (
                value is not None and np.isfinite(value)) else None for key, value in (
                (key, item.get(key)) for key in ('mean_cycles', 'resultant_length', 'samples'))}
    cycles = empty['forward_eligible_complete_cycles']
    phase = empty['physical_LR']
    widths = empty['knee_width_deg']
    cvs = empty['period_cv']
    direct = empty['fixed_initial_heading_net_forward_mm']
    ratio = empty['net_over_path']
    f3 = bool(health and direct is not None and ratio is not None
              and direct >= 10 and ratio >= .5)
    f4 = bool(health and all(cycles[leg] is not None and cycles[leg] >= 10 for leg in LEGS)
        and all(phase[pair]['mean_cycles'] is not None
            and phase[pair]['resultant_length'] is not None
            and abs((phase[pair]['mean_cycles']-.5+.5)%1-.5) <= .15
            and phase[pair]['resultant_length'] >= .5 for pair in PAIRS))
    f5 = bool(health and all(widths[leg] is not None and widths[leg] >= limit
        for leg, limit in {'lf': 25, 'rf': 30, 'lh': 40, 'rh': 35}.items())
        and all(cvs[leg] is not None and cvs[leg] <= .3 for leg in LEGS))
    return {**empty, 'complete_10s_native_health': health,
        'whole_disk_future_exact_separate': future_exact,
        'native_nonfoot_samples': int(native_data['nonfoot_contact'].sum()),
        'minimum_native_upright': float(native_data['upright'].min())
            if len(native) and np.isfinite(native_data['upright']).all() else None,
        'development_screens': {'F3_numeric': f3, 'F4_numeric': f4, 'F5_numeric': f5},
        'F6_unused_poses_tested': False, 'final_adoption': False}


def source_diagnostics(root, out, cal, neutral, full_health, reason):
    signals = load_npz(out/'recruitment-signals.npz')
    metadata = load_npz(out/'recruitment-metadata.npz')
    rates = signals['rate']
    ids = signals['observed_ids']
    full_time = np.r_[0., signals['time_s']]
    rates0 = np.vstack((np.zeros((1, len(ids))), rates))
    source_summary = {'complete_10s_neural_trace': False, 'source_screen': False,
        'partial_window_screen': False}
    try:
        e1 = np.asarray([int(cal['core_ids'][(cal['core_groups'] == group)
            & (cal['core_cell_types'] == 0)][0]) for group in range(6)])
        columns = np.asarray([int(np.flatnonzero(ids == cell)[0]) for cell in e1])
        complete = bool(reason == 'duration' and len(signals['time_s']) == 2000
            and rates.shape == (2000, 446)
            and np.array_equal(signals['time_s'], np.arange(1, 2001)*.005)
            and np.isfinite(rates).all())
        period = float(load_npz(root/'e85-space-compatible-cadence-reference/commands.npz')
                       ['period_s'])
        rhythms, pairs = source_phase_screens(full_time, rates0, columns, period)
        partial = bool(all(item['screen'] for item in rhythms)
            and all(item['screen'] for item in pairs.values()))
        source_summary = {'E1_ids': e1.tolist(), 'rhythms': rhythms,
            'LR_by_window': pairs, 'complete_10s_neural_trace': complete,
            'partial_window_screen': partial,
            'source_screen': bool(complete and partial),
            'sample_dt_s': .005, 'native_neural_dt_s': .00025,
            'interpretation': 'Actual whole-network source screen; separate from physical F3-F5'}
    except (ValueError, KeyError, IndexError, TypeError, FileNotFoundError) as error:
        source_summary.update(error=type(error).__name__, message=str(error))
    save_json(out/'source-phase-summary.json', source_summary)

    lf_summary = {'source_preparation_screen': False,
                  'complete_full_source_and_native_health': False}
    try:
        warm = load_npz(root/'e89-own-leg-native-recruitment/warm-start.npz')
        anatomy = json.loads((out/'muscle-provenance.json').read_text(encoding='utf-8'))
        if (len(ids) != 446 or len(metadata['source_ids']) != 55
                or warm['warm_muscles'].shape != (84, 96)
                or anatomy['muscles'][14]['leg'] != 'lf'
                or anatomy['muscles'][14]['muscle'] != 'tarsus_depressor_muscle'
                or anatomy['muscles'][16]['leg'] != 'lf'
                or anatomy['muscles'][16]['muscle'] != 'tarsus_levator_muscle'):
            raise ValueError('Frozen original LF source/anatomy metadata required')
        lookup = {cell: int(np.flatnonzero(ids == cell)[0])
                  for cell in (60038, 11088, 29608, 105382, 53861)}
        for cell in lookup:
            if neutral['network']['rate'][cell] != 0:
                raise ValueError('Original LF source t0 must be zero')
        peaks = quadratic_peaks(full_time, rates0[:, lookup[53861]])
        phase = phase_trace(full_time, peaks)
        target = np.full((len(full_time), 2), np.nan)
        valid = np.isfinite(phase)
        bins = warm['reference_phase_bins']
        for col, muscle in enumerate((14, 16)):
            inverse = warm['warm_muscles'][:, muscle]
            target[valid, col] = np.interp(np.mod(phase[valid], 1),
                np.r_[bins, 1], np.r_[inverse, inverse[0]])
        actual = rates0[:, [lookup[60038], lookup[11088]]]
        presyn = rates0[:, [lookup[29608], lookup[105382]]]
        np.savez_compressed(out/'actual-LF-source-target-raw.npz', time_s=full_time,
            source_ids=np.array([60038, 11088]), source_rate=actual,
            presynaptic_ids=np.array([29608, 105382]), presynaptic_rate=presyn,
            LF_E1_id=53861, LF_E1_peak_times_s=peaks, LF_phase=phase,
            inverse_target_muscle_ids=np.array([14, 16]), inverse_target=target)
        complete = bool(full_health and source_summary['source_screen']
            and source_summary['complete_10s_neural_trace'] and len(full_time) == 2001)
        if not complete:
            raise ValueError('Incomplete whole-network/native source trajectory')
        period = float(load_npz(root/'e85-space-compatible-cadence-reference/commands.npz')
                       ['period_s'])
        source_checks = {
            'depressor_60038': lf_source_screen(full_time, actual[:, 0],
                presyn[:, 0], target[:, 0], period, sign=1),
            'levator_11088': lf_source_screen(full_time, actual[:, 1],
                presyn[:, 1], target[:, 1], period, sign=-1)}
        physiology = {}
        for cell in (60038, 11088):
            col = lookup[cell]
            gain = float(metadata['_cell_gain'][col])
            tau = float(metadata['tau'][col])
            no_drive = bool(np.all(signals['drive'][:, col] == 0))
            physiology[str(cell)] = {'gain': gain, 'tau_s': tau,
                'direct_drive_zero': no_drive,
                'screen': bool(gain == 0 and tau == float(np.float32(.02))
                               and no_drive)}
        lf_summary = {'complete_full_source_and_native_health': True,
            'screens': source_checks, 'native_physiology': physiology,
            'source_preparation_screen': bool(all(x['screen'] for x in source_checks.values())
                and all(x['screen'] for x in physiology.values())),
            'physical_F_separate': True,
            'target_alignment': 'Own actual LF E1 peaks to frozen E89 84-bin inverse',
            'interpretation': 'Actual source screen only; not gait or adoption'}
    except (ValueError, KeyError, IndexError, TypeError, FileNotFoundError) as error:
        lf_summary.update(error=type(error).__name__, message=str(error))
    save_json(out/'actual-LF-source-summary.json', lf_summary)
    return source_summary, lf_summary


def save_startup_raw(out, rows, prediction, startup):
    observed = max(0, min(len(rows)-1, 99))
    time_s = np.asarray([row['time_s'] for row in rows[1:observed+1]])
    q = np.asarray([row['joint_targets_rad'] for row in rows[1:observed+1]], dtype=float)
    tendon_order = tuple(f'{leg}_tarsus' for leg in LEGS)
    ltm = np.asarray([[row['tendon_inputs'][key] for key in tendon_order]
                      for row in rows[1:observed+1]], dtype=float)
    q = q.reshape((observed, 42)) if q.size == observed*42 else np.empty((0, 42))
    ltm = ltm.reshape((observed, 6)) if ltm.size == observed*6 else np.empty((0, 6))
    np.savez_compressed(out/'actual-startup-vs-E112b-raw.npz', time_s=time_s,
        actual_command=q, actual_LTM=ltm,
        E112b_conditional_command=prediction['command'][1:observed+1],
        E112b_conditional_LTM=prediction['tendon'][1:observed+1],
        original_E100_E103_command=startup['target_command'],
        original_E100_E103_LTM=startup['target_tendon'])


def resource_screen(preparation, body_wall, process_wall, ledger):
    return bool(preparation.get('resource_limit_met') is True
        and np.isfinite((body_wall, process_wall)).all()
        and body_wall <= BODY_LIMIT_S and process_wall <= PROCESS_LIMIT_S
        and PRIOR_WALL_S+process_wall <= SHARED_LIMIT_S
        and ledger.clock_valid and ledger.neural_s is not None
        and ledger.physical_s is not None
        and ledger.neural_s <= 10.1000001
        and ledger.physical_s <= 10.1000001)


def run_body(root, out, graph, cal, prediction, parent_obs, preparation,
             spec, process_began):
    body_began = time.perf_counter()
    neutral = StateStore.load(out/'candidate-neutral')
    session = None
    clone = None
    restore_main = None
    restore_clone = None
    ledger = AdvanceLedger()
    rows, signal_rows, native, clone_native = [], [], [], []
    basic = detailed = metadata_sources = dns = None
    reason = 'duration'
    runtime_error = None
    raw_errors = []
    checkpoint_1p1_ready = False
    startup = load_npz(root/'e103-startup-preserving-recruitment/startup-inputs.npz')
    try:
        session = IntrinsicRecruitmentTrial.from_checkpoint(graph, neutral)
        if session.network.device != 'cuda':
            raise ValueError('E113 body must run full BANC on CUDA')
        exact(neutral, session.snapshot(), 'E113 body original disk neutral')
        basic, detailed, metadata_sources, dns = observation_ids(graph, cal, parent_obs)
        weights = source_matrix(session.network)
        np.savez_compressed(out/'recruitment-metadata.npz', observed_ids=detailed,
            motor_ids=cal['motor_ids'], source_ids=metadata_sources,
            source_to_motor=weights[cal['motor_ids']][:, metadata_sources].toarray(),
            **{key: getattr(session.network, key).cpu().numpy()[detailed].copy()
               for key in ('tau', 'a', 'threshold', 'cap', '_cell_gain')})
        restore_main = install_native_observer(session.body, native)
        StateStore.save(out/'start', session.snapshot())
        rows.append(capture_control(session))
        with (out/'trace.jsonl').open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(jsonable(rows[0]), allow_nan=False)+'\n')
            for tick in range(1, 2001):
                if (time.perf_counter()-body_began > BODY_LIMIT_S-8
                        or time.perf_counter()-process_began > PROCESS_LIMIT_S-8):
                    reason = 'resource_guard'
                    break
                previous_native = len(native)
                ledger.advance(session, 1, 'main')
                row = capture_control(session)
                rows.append(row)
                stream.write(json.dumps(jsonable(row), allow_nan=False)+'\n')
                current = session.network._current(session.network.rate)
                signal_rows.append({'rate': session.network.readout(detailed),
                    'recurrent_current': current[detailed].cpu().numpy().copy(),
                    'drive': session.network.drive[detailed].cpu().numpy().copy(),
                    'adaptation': session.network.adaptation[detailed].cpu().numpy().copy(),
                    'muscle_activation': session.adapter.activation.copy()})
                if (len(native)-previous_native != 50
                        or session.network.tick != tick*20
                        or abs(session.body.physics_time()-tick*.005) > 1e-8):
                    raise ValueError('One complete 5ms control requires 20neural/50physics steps')
                if session.fault or session.body.fault:
                    reason = 'native_health_guard'
                    break
                if tick == 200:
                    StateStore.save(out/'checkpoint-1s', session.snapshot())
                    clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
                        StateStore.load(out/'checkpoint-1s'))
                    try:
                        restore_clone = install_native_observer(clone.body, clone_native)
                        ledger.advance(clone, 20, 'clone')
                        if (clone.network.tick != 220*20 or len(clone_native) != 1000
                                or abs(clone.body.physics_time()-1.1) > 1e-8):
                            raise ValueError('Exact .1s clone clocks required')
                        StateStore.save(out/'clone-1p1', clone.snapshot())
                    finally:
                        if restore_clone is not None:
                            restore_clone()
                            restore_clone = None
                        clone.close()
                        clone = None
                if tick == 220:
                    StateStore.save(out/'main-1p1', session.snapshot())
                    checkpoint_1p1_ready = True
                if tick % 200 == 0:
                    print(json.dumps({'E113_main_time_s': tick*.005,
                        'body_wall_s': time.perf_counter()-body_began,
                        'actual_neural_s': ledger.neural_s,
                        'actual_physical_s': ledger.physical_s}), flush=True)
        if (reason == 'duration' and not session.fault and not session.body.fault):
            StateStore.save(out/'final', session.snapshot())
    except NativeHealthStop as error:
        reason = 'native_health_guard'
        runtime_error = error
    except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
        reason = 'execution_error'
        runtime_error = error
    finally:
        if session is not None:
            try:
                raw_errors.extend(save_raw(out, graph, neutral, session, rows, basic,
                    detailed, dns, metadata_sources, signal_rows, native, clone_native,
                    ledger, spec))
            except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
                raw_errors.append({'error': type(error).__name__, 'message': str(error),
                    'traceback': traceback.format_exc()})
            finally:
                if restore_clone is not None:
                    try:
                        restore_clone()
                    except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
                        raw_errors.append({'error': type(error).__name__,
                                           'message': 'Clone observer restore: '+str(error)})
                if clone is not None:
                    try:
                        clone.close()
                    except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
                        raw_errors.append({'error': type(error).__name__,
                                           'message': 'Clone close: '+str(error)})
                if restore_main is not None:
                    try:
                        restore_main()
                    except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
                        raw_errors.append({'error': type(error).__name__,
                                           'message': 'Main observer restore: '+str(error)})
                try:
                    session.close()
                except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
                    raw_errors.append({'error': type(error).__name__,
                                       'message': 'Main close: '+str(error)})
        if runtime_error is not None:
            save_json(out/'runtime-error.json', {'stop_reason': reason,
                'error': type(runtime_error).__name__, 'message': str(runtime_error),
                'actual_neural_s': ledger.neural_s,
                'actual_physical_s': ledger.physical_s,
                'last_clock_event': ledger.events[-1] if ledger.events else None})
        if raw_errors:
            save_json(out/'raw-persistence-errors.json', raw_errors)

    # The raw trace, native contacts, recruitment and actual-clock ledger now exist.
    # No failed control was added as a regular 5 ms endpoint.
    analysis_errors = []
    future_exact = False
    if checkpoint_1p1_ready and not raw_errors:
        try:
            exact(StateStore.load(out/'clone-1p1'), StateStore.load(out/'main-1p1'),
                  'E113 whole checkpoint1s future at1.1s')
            future_exact = True
        except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
            analysis_errors.append({'stage': 'checkpoint_future',
                'error': type(error).__name__, 'message': str(error)})
    quality = None
    if len(rows) == 2001 and not raw_errors:
        try:
            quality, _, _ = analyze(load_model(out/'trace.json'))
            save_json(out/'quality-analysis.json', quality)
        except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
            analysis_errors.append({'stage': 'original_gait_evaluator',
                'error': type(error).__name__, 'message': str(error)})
    fault = bool(runtime_error or raw_errors or (session is not None and
        (session.fault or session.body.fault)))
    summary = body_quality(rows, native, quality, reason=reason, fault=fault,
        clock_valid=ledger.clock_valid, future_exact=future_exact)
    save_json(out/'quality-summary.json', summary)
    save_startup_raw(out, rows, prediction, startup)
    startup_summary = startup_diagnostic(rows, native, prediction, startup)
    save_json(out/'startup-diagnostic.json', startup_summary)
    native_data = native_arrays(native)
    slip = supported_slip(native_data,
                          complete=summary['complete_10s_native_health'])
    save_json(out/'supported-slip.json', slip)
    comparator = json.loads((root/'parent-e113-expected-inputs.json')
        .read_text(encoding='utf-8'))
    baseline = json.loads((root/'e110-e109-native-body-diagnostic/quality-summary.json')
        .read_text(encoding='utf-8'))
    exact(baseline, comparator['E110_original_comparator'],
          'Frozen actual E110 physical comparator')
    baseline_slip = json.loads((root/'e110-e109-native-body-diagnostic/supported-slip.json')
        .read_text(encoding='utf-8'))
    exact(baseline_slip['load_weighted_mm_s'],
          comparator['E110_supported_loadweighted_slip_mm_s'],
          'Frozen actual E110 supported slip')
    comparison = {'E110': baseline, 'E113': summary,
        'E110_supported_loadweighted_slip_mm_s':
            comparator['E110_supported_loadweighted_slip_mm_s'],
        'E113_supported_loadweighted_slip_mm_s': slip['load_weighted_mm_s'],
        'same_window_s': [.5, 10.0]}
    save_json(out/'E110-E113-comparison.json', comparison)
    source_summary = {'source_screen': False}
    lf_summary = {'source_preparation_screen': False}
    if not raw_errors and (out/'recruitment-signals.npz').exists():
        try:
            source_summary, lf_summary = source_diagnostics(root, out, cal, neutral,
                summary['complete_10s_native_health'], reason)
        except Exception as error:  # noqa: BLE001 - preserve original failure and raw diagnostics
            analysis_errors.append({'stage': 'source_diagnostics',
                'error': type(error).__name__, 'message': str(error)})
    if analysis_errors:
        save_json(out/'analysis-errors.json', analysis_errors)
    body_wall = time.perf_counter()-body_began
    process_wall = time.perf_counter()-process_began
    preparation_wall = preparation['wall_s']
    resources = resource_screen(preparation, body_wall, process_wall, ledger)
    validity = ('valid_full_BANC_physical_feedback_diagnosis'
        if (reason == 'duration' and summary['complete_10s_native_health']
            and future_exact and resources and not raw_errors and not analysis_errors)
        else 'resource_limit_incomplete' if not resources else 'incomplete')
    result = {'experiment': EXPERIMENT, 'validity': validity,
        'preparation_status': preparation['validity'],
        'conditional_shape_failed': True, 'E112b_local_supported': False,
        'adoption': 'not_applicable', 'F1_F8': 'not_assessed',
        'F6_unused_poses_tested': False, 'fit_calls': 0,
        'stop_reason': reason, 'main_completed_controls': len(rows)-1,
        'main_native_samples': len(native), 'clone_native_samples': len(clone_native),
        'actual_neural_s': ledger.neural_s, 'actual_physical_s': ledger.physical_s,
        'actual_costs_by_owner': ledger.by_owner(),
        'clock_valid': ledger.clock_valid, 'whole_disk_future_exact': future_exact,
        'device': neutral['network']['device'],
        'quality_summary': summary, 'startup_diagnostic': startup_summary,
        'source_phase_summary': source_summary,
        'actual_LF_source_summary': lf_summary,
        'supported_slip': slip, 'resource_limit_met': resources,
        'preparation_wall_s': preparation_wall, 'body_wall_s': body_wall,
        'process_wall_s': process_wall,
        'prior_E112b_conservative_wall_s': PRIOR_WALL_S,
        'conservative_shared_process_wall_s': PRIOR_WALL_S+process_wall,
        'hypothesis_verdict': 'physical_F3_F5_all_numeric' if validity.startswith('valid_')
            and all(summary['development_screens'].values()) else 'not_supported'}
    save_json(out/'result.json', result)
    if runtime_error is not None and reason == 'execution_error':
        raise runtime_error
    if raw_errors:
        raise RuntimeError('E113 raw persistence failed; see raw-persistence-errors.json')
    return result


def main(root, graph_path, out):
    process_began = time.perf_counter()
    if out.exists():
        raise FileExistsError('Preserve single E113 output: '+str(out))
    out.mkdir(parents=True)
    spec = {'experiment': EXPERIMENT,
        'question': 'Does reviewed E112b efficacy improve actual F3-F5 from the same E32 neutral?',
        'status': 'E112b conditional shape/local failed, direct physical diagnostic only',
        'intervention': 'Only 1572 existing signed source49-to-MN391 float32 efficacies; original 38 tau and all other E110/E32 state. Full BANC native sensory and body, no runtime phase/teacher/ramp/root/pose/reference.',
        'controls': 2000, 'neural_steps_per_control': 20,
        'physics_steps_per_control': 50, 'checkpoint_s': 1.0,
        'clone_future_s': .1, 'F6_exposed': False,
        'limits': {'prepare_wall_s': PREPARE_LIMIT_S,
                   'body_wall_s': BODY_LIMIT_S,
                   'process_wall_s': PROCESS_LIMIT_S,
                   'prior_conservative_wall_s': PRIOR_WALL_S,
                   'shared_wall_s': SHARED_LIMIT_S,
                   'neural_s': 10.1, 'physical_s': 10.1},
        'tool_sha256': sha256(__file__),
        'original_evaluator_sha256': sha256(
            Path(__file__).with_name('analyze_c_banc_biological_gait.py')),
        'E112b_parent_review_sha256': sha256(root/'e112-dense-foot-recruitment-v2/parent-final-review.json'),
        'E112b_original59_manifest_sha256': sha256(
            root/'e112-dense-foot-recruitment-v2/parent-output-manifest.json'),
        'E32_start_manifest_sha256': sha256(root/'e32-core-adaptation/candidate/start/manifest.json'),
        'E110_calibration_sha256': sha256(root/'e110-e109-native-body-diagnostic/calibration.npz'),
        'E112b_calibration_sha256': sha256(root/'e112-dense-foot-recruitment-v2/calibration.npz'),
        'E112b_prediction_sha256': sha256(root/'e112-dense-foot-recruitment-v2/prediction.npz'),
        'parent_fixture_sha256': sha256(root/'parent-e113-expected-calibration.npz'),
        'parent_observation_sha256': sha256(root/'parent-e113-expected-observation.npz'),
        'parent_input_manifest_sha256': sha256(root/'parent-e113-expected-inputs.json'),
        'E113_design_sha256': sha256(
            Path(__file__).resolve().parents[1]/'docs/BANC_M3_E113_NATIVE_BODY_DESIGN_20260920.md')}
    save_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    try:
        graph, cal, prediction, parent_obs, preparation = prepare(
            root, graph_path, out, process_began)
        if not preparation['resource_limit_met']:
            raise TimeoutError('E113 preparation exceeded 40 wall seconds')
        return run_body(root, out, graph, cal, prediction, parent_obs,
                        preparation, spec, process_began)
    except Exception as error:
        try:
            save_json(out/'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error),
                'process_wall_s': time.perf_counter()-process_began,
                'actual_neural_s': load_clock_value(out, 'actual_neural_s'),
                'actual_physical_s': load_clock_value(out, 'actual_physical_s'),
                'fit_calls': 0})
        except Exception as receipt_error:  # noqa: BLE001 - keep original failure
            error.add_note('E113 execution receipt failed: '
                           +type(receipt_error).__name__+': '+str(receipt_error))
        raise


def load_clock_value(out, key):
    path = out/'clock-summary.json'
    return json.loads(path.read_text(encoding='utf-8')).get(key) if path.exists() else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = main(args.root, args.graph, args.out)
    print(json.dumps(jsonable({key: value for key, value in result.items()
        if key not in ('quality_summary', 'startup_diagnostic',
                       'source_phase_summary', 'actual_LF_source_summary')}),
        ensure_ascii=False, allow_nan=False), flush=True)
