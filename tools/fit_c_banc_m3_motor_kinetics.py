# ruff: noqa: B023
"""E109: bounded CUDA conditional fit of existing motor edges and native MN taus.

The only body constructed here is a static shadow for original-neutral foot
kinematics. There is no neural trial, physical step, or runtime phase input.
"""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256
from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.banc_m3_intrinsic_recruitment import patch_tau
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.fit_c_banc_m3_block_constrained_recruitment import (
    BlockLimit,
    NativeCache,
    absolute_shape_screen,
    accepted_review,
    check_static_cross_jacobian,
    choose_records,
    json_safe,
    leg_partition,
    metrics,
    optimizer_record,
    save_json,
    shadow_body_lifetime,
    verify_metadata_weights,
)
from tools.fit_c_banc_m3_constrained_startup import (
    checked_initial_state,
    current_edge_start,
    native_shape_screen,
    startup_screen,
)
from tools.verify_c_banc_observation import exact

BASE = ROOT / 'verification/banc-natural-gait-m3-20260919'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
SOURCE_HASH_PATHS = {
    'E105_review': 'e105-startup-native-body-diagnostic-v2/body/parent-second-review.json',
    'E107_review': 'e107-within-leg-shape-factor-diagnostic/parent-first-review.json',
    'E105_signals': 'e105-startup-native-body-diagnostic-v2/body/recruitment-signals.npz',
    'E105_metadata': 'e105-startup-native-body-diagnostic-v2/body/recruitment-metadata.npz',
    'E105_trace': 'e105-startup-native-body-diagnostic-v2/body/trace.json',
    'E105_calibration': 'e105-startup-native-body-diagnostic-v2/calibration.npz',
    'E105_start_manifest': 'e105-startup-native-body-diagnostic-v2/body/start/manifest.json',
    'E104_prediction': 'e104-constrained-startup-recruitment/prediction.npz',
    'E104_coefficients': 'e104-constrained-startup-recruitment/fit-coefficients.npz',
    'E103_startup': 'e103-startup-preserving-recruitment/startup-inputs.npz',
    'E107_frozen': 'e107-within-leg-shape-factor-diagnostic/frozen-both-inputs.npz',
    'E99_calibration': 'e99-lf-tarsal-source-preparation/calibration.npz',
    'E85_reference': 'e85-space-compatible-cadence-reference/commands.npz',
    'E101_baseline': 'e101-individual-edge-native-recruitment/baseline-output.json',
    'E32_start_manifest': 'e32-core-adaptation/candidate/start/manifest.json',
    'parent_expected': 'parent-e108-expected-inputs.npz',
    'E108_v1_first_review': 'e108-block-constrained-recruitment/parent-first-review.json',
}
PRIOR_SHARED_WALL_S = 40.
WHOLE_LIMIT_S = 185.
FIT_LIMIT_S = 150.
LEG_LIMIT_S = 24.
LEG_MAX_CALLS = 1000
TOTAL_MAX_CALLS = 6000
OPTIMIZER_RESERVE = (1 - 2e-4)**2
SELECTOR_RESERVE = (1 - 1e-4)**2


def native_tau_patch(cal, motor_ids, original_motor_tau, motor_mask, scale):
    """Keep old float64 calibration entries and actual float32 native values."""
    motor_ids = np.asarray(motor_ids)
    tau = np.asarray(original_motor_tau)
    mask = np.asarray(motor_mask)
    scale = np.asarray(scale, dtype=np.float64)
    old_ids = np.asarray(cal['tau_ids'])
    old_values = np.asarray(cal['tau_values'])
    if (motor_ids.shape != (391,) or tau.shape != (391,)
            or mask.shape != (6, 391) or scale.shape != (6,)
            or old_ids.shape != (38,) or old_values.shape != (38,)
            or old_values.dtype != np.float64 or len(np.unique(motor_ids)) != 391
            or len(np.unique(old_ids)) != 38 or np.any(mask.sum(axis=0) != 1)
            or tuple(mask.sum(axis=1)) != (69, 63, 63, 70, 63, 63)
            or np.any(scale < 1) or np.any(scale > 5)
            or not np.isfinite(scale).all() or not np.isfinite(tau).all()
            or not np.array_equal(tau, tau.astype(np.float32).astype(float))):
        raise ValueError('Original native 391 MN/38 tau membership and bounded six scales required')
    special = motor_ids[tau != float(np.float32(.02))]
    if (not np.array_equal(special, np.array([92104]))
            or tau[motor_ids == 92104][0] != float(np.float32(.014046886935830116))
            or len(np.intersect1d(old_ids, motor_ids)) != 1
            or int(np.intersect1d(old_ids, motor_ids)[0]) != 92104):
        raise ValueError('Actual original 390 standard MN and sole MN92104 tau required')
    leg_scale = mask.T @ scale
    actual_motor = np.asarray(tau * leg_scale, dtype=np.float32)
    if np.any(actual_motor < np.float32(.0125)):
        raise ValueError('Motor tau below native minimum')
    ids = np.union1d(old_ids, motor_ids)
    old_positions = np.searchsorted(ids, old_ids)
    motor_positions = np.searchsorted(ids, motor_ids)
    values = np.empty(len(ids), dtype=old_values.dtype)
    values[old_positions] = old_values
    values[motor_positions] = actual_motor.astype(np.float64)
    native_values = values.astype(np.float32)
    nonmotor = ~np.isin(old_ids, motor_ids)
    exact(values[old_positions[nonmotor]], old_values[nonmotor],
          '37 original non-MN calibration taus')
    exact(native_values[old_positions[nonmotor]], old_values[nonmotor].astype(np.float32),
          '37 original non-MN native taus')
    if len(ids) != 428 or len(np.unique(ids)) != 428:
        raise ValueError('Exactly 428 unique patched native tau IDs required')
    mapped = native_values[motor_positions]
    exact(mapped, actual_motor, 'All actual float32 motor taus in patch')
    return ids, values, native_values, actual_motor


def resource_screen(ledger):
    prior, whole, fit = (ledger.get(k) for k in
                         ('prior_shared_wall_s', 'whole_wall_s', 'fit_wall_s'))
    started = ledger.get('leg_started_evaluations', {})
    wall = ledger.get('leg_wall_s', {})
    if (prior != PRIOR_SHARED_WALL_S or whole is None
            or not math.isfinite(whole) or whole < 0):
        return 'limit_exceeded_or_unverifiable'
    if ((ledger.get('fit_start_elapsed_s') is not None or any(started.values()))
            and (fit is None or not math.isfinite(fit) or fit < 0)):
        return 'limit_exceeded_or_unverifiable'
    if any(v > 0 and (wall.get(k) is None or not math.isfinite(wall[k]) or wall[k] < 0)
           for k, v in started.items()):
        return 'limit_exceeded_or_unverifiable'
    if (whole > WHOLE_LIMIT_S or prior + whole > 225
            or (fit is not None and fit > FIT_LIMIT_S)
            or any(v is not None and v > LEG_LIMIT_S for v in wall.values())
            or any(v > LEG_MAX_CALLS for v in started.values())
            or sum(started.values()) > TOTAL_MAX_CALLS):
        return 'limit_exceeded'
    return 'within_limits'


def solve_block(cache, x0, n, path):
    """Use the inner reserve; persist the unaltered SciPy outcome first."""
    result = minimize(lambda x: cache.get(x)['objective'], x0, method='SLSQP',
        jac=lambda x: cache.get(x)['jac'],
        constraints=[{'type': 'ineq',
            'fun': lambda x: OPTIMIZER_RESERVE-cache.get(x)['ratios'],
            'jac': lambda x: -cache.get(x)['ratio_jac']}],
        bounds=[(math.log(.001), math.log(10.))]*n + [(0., math.log(5.))],
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    termination = optimizer_record(result,
        'scipy_success' if result.success else 'scipy_failure')
    save_json(path, termination)
    if termination['stop'] == 'scipy_nonfinite_result':
        raise ValueError('Nonfinite SciPy return; preserved '+str(path))
    return result, termination


def main(root, graph_path, out, *, prior_shared_wall_s):
    if prior_shared_wall_s != PRIOR_SHARED_WALL_S:
        raise ValueError('E108 conservative v1+v2 process cost requires exactly 40s prior charge')
    if out.exists():
        raise FileExistsError('Preserve existing E109 output: '+str(out))
    started = time.perf_counter()
    out.mkdir(parents=True)
    ledger = {'experiment': 'M3-E109-native-motor-kinetics',
        'prior_shared_wall_s': PRIOR_SHARED_WALL_S,
        'whole_limit_s': WHOLE_LIMIT_S, 'fit_limit_s': FIT_LIMIT_S,
        'leg_limit_s': LEG_LIMIT_S, 'shared_limit_s': 225.,
        'preflight_wall_s': None, 'fit_wall_s': None, 'whole_wall_s': None,
        'fit_start_elapsed_s': None, 'active_leg': None,
        'active_leg_start_elapsed_s': None,
        'leg_started_evaluations': {name: 0 for name in LEGS},
        'leg_completed_evaluations': {name: 0 for name in LEGS},
        'leg_cache_hits': {name: 0 for name in LEGS},
        'leg_wall_s': {name: None for name in LEGS},
        'baseline_validation_calls': 0, 'initializer_validation_calls': 0,
        'final_validation_calls': 0, 'actual_neural_s': 0.,
        'actual_physical_s': 0., 'resource_screen': None}
    result = None
    error = None
    try:
        result = _run(root, graph_path, out, started, ledger)
    except Exception as exc:  # noqa: BLE001 - persist every technical failure and started cost
        error = exc
    finally:
        now = time.perf_counter()
        if ledger['fit_start_elapsed_s'] is not None and ledger['fit_wall_s'] is None:
            ledger['fit_wall_s'] = now-started-ledger['fit_start_elapsed_s']
        active = ledger['active_leg']
        if active is not None and ledger['active_leg_start_elapsed_s'] is not None:
            inclusive_wall = now-started-ledger['active_leg_start_elapsed_s']
            previous_wall = ledger['leg_wall_s'][active]
            ledger['leg_wall_s'][active] = (inclusive_wall if previous_wall is None
                else max(previous_wall, inclusive_wall))
        ledger['whole_wall_s'] = now-started
        ledger['shared_wall_s'] = PRIOR_SHARED_WALL_S+ledger['whole_wall_s']
        ledger['resource_screen'] = resource_screen(ledger)
        if result is not None:
            result.update(resource_screen=ledger['resource_screen'],
                whole_wall_s=ledger['whole_wall_s'],
                shared_wall_s=ledger['shared_wall_s'])
            if ledger['resource_screen'] != 'within_limits':
                result['validity'] = 'resource_limit_incomplete'
                result['local_supported'] = False
            save_json(out/'result.json', result)
        if error is not None:
            save_json(out/'execution-error.json', {
                'validity': 'incomplete', 'error_type': type(error).__name__,
                'message': str(error),
                'traceback': ''.join(traceback.format_exception(error)),
                'whole_wall_s_at_error': ledger['whole_wall_s'],
                'fit_wall_s': ledger['fit_wall_s'],
                'shared_wall_s': ledger['shared_wall_s'],
                'resource_screen': ledger['resource_screen'],
                'leg_started_evaluations': ledger['leg_started_evaluations'],
                'leg_completed_evaluations': ledger['leg_completed_evaluations'],
                'leg_wall_s': ledger['leg_wall_s'],
                'actual_neural_s': 0., 'actual_physical_s': 0.})
        # Charge persistence and cleanup to the same experiment.
        ledger['whole_wall_s'] = time.perf_counter()-started
        ledger['shared_wall_s'] = PRIOR_SHARED_WALL_S+ledger['whole_wall_s']
        ledger['resource_screen'] = resource_screen(ledger)
        if result is not None:
            result['whole_wall_s'] = ledger['whole_wall_s']
            result['shared_wall_s'] = ledger['shared_wall_s']
            result['resource_screen'] = ledger['resource_screen']
            if ledger['resource_screen'] != 'within_limits':
                result['validity'] = 'resource_limit_incomplete'
                result['local_supported'] = False
            save_json(out/'result.json', result)
        save_json(out/'execution-ledger.json', ledger)
    if error is not None:
        raise error
    return result


def _run(root, graph_path, out, started, ledger):
    e108 = root/'e108-block-constrained-recruitment-v2'
    review_path = e108/'parent-final-review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    accepted_review(review, 'accepted_as_valid_failed_conditional_fit')
    if review['actual_process_wall_s'] != 31.967537 or review['prior_v1_conservative_charge_s'] != 8:
        raise ValueError('Explicit E108 conservative 40s prior evidence changed')
    for relative, digest in review['original_artifact_sha256'].items():
        if sha256(e108/relative) != digest:
            raise ValueError('Reviewed E108 artifact changed: '+relative)
    old_spec = json.loads((e108/'spec.json').read_text(encoding='utf-8'))
    for key, relative in SOURCE_HASH_PATHS.items():
        if sha256(root/relative) != old_spec['source_sha256'][key]:
            raise ValueError('E108 immutable original source changed: '+key)
    for name in ('causal_helper', 'geometry_helper'):
        filename = {'causal_helper': 'banc_m3_causal_recruitment.py',
                    'geometry_helper': 'banc_m3_kinematic_fit.py'}[name]
        if sha256(ROOT/'tools'/filename) != old_spec['source_sha256'][name]:
            raise ValueError('Reviewed native helper changed: '+name)
    graph_files = sorted(p for p in graph_path.iterdir() if p.is_file())
    for path in graph_files:
        if sha256(path) != old_spec['source_sha256']['graph_'+path.name]:
            raise ValueError('Original graph changed: '+path.name)
    spec = {'experiment': 'M3-E109-native-motor-kinetics',
        'source_sha256': {'E108_parent_final_review': sha256(review_path),
            'E108_reviewed_artifacts': review['original_artifact_sha256'],
            'E108_original_sources': old_spec['source_sha256'],
            'parent_E109_tau_fixture': sha256(root/'parent-e109-tau-mapping-fixture.npz'),
            'parent_E109_tau_fixture_report': sha256(root/'parent-e109-tau-mapping-fixture.json'),
            'E109_runner': sha256(Path(__file__)),
            'E109_unit_test': sha256(ROOT/'tests/test_c_banc_motor_kinetics.py')},
        'prior_shared_wall_s': PRIOR_SHARED_WALL_S,
        'limits': {'whole_wall_s': WHOLE_LIMIT_S, 'fit_wall_s': FIT_LIMIT_S,
            'per_leg_wall_s': LEG_LIMIT_S,
            'per_leg_unique_native_evaluations': LEG_MAX_CALLS,
            'total_unique_native_evaluations': TOTAL_MAX_CALLS,
            'slsqp_maxiter': 1000, 'ftol': 1e-9,
            'callback_wall_reserve_s': .5},
        'leg_order': LEGS, 'efficacy_bounds': [.001, 10.],
        'tau_scale_bounds': [1., 5.],
        'optimizer_ratio_cap': OPTIMIZER_RESERVE,
        'selector_ratio_cap': SELECTOR_RESERVE,
        'original_global_startup_limits': [.02, .05, .01, .03],
        'conditional_only': True, 'runtime_reference_phase': False,
        'native_tau_hypothesis': True}
    save_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'unit-test.py').write_bytes((ROOT/'tests/test_c_banc_motor_kinetics.py').read_bytes())

    train = np.load(e108/'training-inputs.npz')
    expected = np.load(root/'parent-e108-expected-inputs.npz')
    for key in expected.files:
        exact(train[key], expected[key], 'Parent independent E108/E109 '+key)
    static = np.load(e108/'static-geometry.npz')
    previous_baseline = np.load(e108/'baseline-prediction.npz')
    previous_prediction = np.load(e108/'prediction.npz')
    e108_cal = {key: val.copy() for key, val in np.load(e108/'calibration.npz').items()}
    e105_cal = {key: val.copy() for key, val in
        np.load(root/SOURCE_HASH_PATHS['E105_calibration']).items()}
    old_result = json.loads((e108/'result.json').read_text(encoding='utf-8'))
    if old_result['local_supported'] is not False:
        raise ValueError('Preserved E108 conditional failure required')
    for key, value in e105_cal.items():
        if key not in ('values', 'individual_edge_efficacies'):
            exact(value, e108_cal[key], 'E108 unchanged E105 calibration '+key)
    # Save raw E108 input arrays before any later neural/geometry fidelity gate.
    np.savez_compressed(out/'immutable-inputs.npz',
        **{key: train[key] for key in train.files})

    original = StateStore.load(root/'e32-core-adaptation/candidate/start')
    prepared = StateStore.load(root/'e105-startup-native-body-diagnostic-v2/body/start')
    checked_initial_state(prepared)
    exact(original['body'], prepared['body'], 'Original whole-zero body/world')
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        exact(original['network'][key], prepared['network'][key],
              'Original neural startup '+key)
    graph = GraphStore.load(graph_path)
    motor = train['motor_ids']
    sources = train['source_ids']
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, motor)
    current = patch_weights(base, e105_cal['edge_ids'],
                            e105_cal['source_values'], e105_cal['values'])
    selected_network = patch_weights(base, e108_cal['edge_ids'],
                                     e108_cal['source_values'], e108_cal['values'])
    original_w = base[motor][:, sources].toarray().astype(float)
    current_w = current[motor][:, sources].toarray().astype(float)
    selected_w = selected_network[motor][:, sources].toarray().astype(float)
    exact(original_w, train['original_w'], 'Original signed W')
    exact(current_w, train['current_w'], 'E105 actual W for immutable background')
    exact(selected_w, previous_prediction['actual_float32_source_weights'],
          'E108 selected actual float32 W')
    verify_metadata_weights(selected_w, selected_w.astype(np.float32))
    edge_ids = train['edge_ids']
    edge_rows = train['edge_rows']
    edge_cols = train['edge_cols']
    edge_original = train['original_signed_edge_weights']
    exact(edge_original, base.data[edge_ids].astype(np.float32), 'Original edge weights')
    exact(selected_network.indptr, current.indptr, 'Original CSR row topology')
    exact(selected_network.indices, current.indices, 'Original CSR columns')
    nonfit = np.ones(len(base.data), dtype=bool)
    nonfit[edge_ids] = False
    exact(selected_network.data[nonfit], current.data[nonfit], 'E108 all nonfit weights')
    raw_eff, init_eff, projection = current_edge_start(
        edge_original, selected_network.data[edge_ids].astype(np.float32))
    if not np.array_equal(edge_ids, e108_cal['individual_edge_ids']):
        raise ValueError('E108 selected edge order changed')
    motor_mask, edge_mask = leg_partition(train['moment'], train['pool'],
        train['ltm'], edge_rows, expected_motor=train['leg_motor_mask'],
        expected_edge=train['leg_edge_mask'])
    tau = train['tau']
    metadata = np.load(root/SOURCE_HASH_PATHS['E105_metadata'])
    lookup = {int(v): i for i, v in enumerate(metadata['observed_ids'])}
    motor_cols = np.array([lookup[int(v)] for v in motor])
    exact(tau, metadata['tau'][motor_cols].astype(float), 'Actual E105 native MN tau')
    tau_ids, tau_values, native_tau_values, init_motor_tau = native_tau_patch(
        e105_cal, motor, tau, motor_mask, np.ones(6))
    fixture = np.load(root/'parent-e109-tau-mapping-fixture.npz')
    exact(motor, fixture['motor_ids'], 'Parent independent motor IDs')
    exact(tau, fixture['original_motor_tau'], 'Parent independent native MN tau')
    exact(e105_cal['tau_ids'], fixture['original_patch_ids'],
          'Parent independent original tau IDs')
    exact(e105_cal['tau_values'], fixture['original_patch_tau'],
          'Parent independent original float64 tau values')
    exact(tau_ids, fixture['combined_ids'], 'Parent independent 428 native tau IDs')
    exact(init_motor_tau, tau.astype(np.float32), 'Scale-one native tau exact')
    if np.intersect1d(motor, sources).size != 0:
        raise ValueError('Motor/source ID overlap changes native source definition')
    fixture_mask = np.zeros_like(motor_mask)
    fixture_mask[fixture['motor_leg_group'], np.arange(391)] = True
    exact(motor_mask, fixture_mask, 'Parent independent six MN leg groups')
    sentinel_ids, sentinel_cal, sentinel_native, sentinel_motor = native_tau_patch(
        e105_cal, motor, tau, motor_mask, fixture['fixture_scales'])
    exact(sentinel_ids, fixture['combined_ids'], 'Parent sentinel tau patch IDs')
    exact(sentinel_cal, fixture['fixture_calibration_patch_tau'],
          'Parent sentinel float64 calibration tau')
    exact(sentinel_native, fixture['fixture_actual_native_float32_patch_tau'],
          'Parent sentinel float32 native patch tau')
    exact(sentinel_motor, fixture['fixture_actual_float32_motor_tau'],
          'Parent sentinel float32 MN tau')
    # Verify that patch_tau's native float32 contract yields exactly these values.
    full_original_tau = np.ones(int(max(tau_ids.max(), motor.max()))+1,
                                dtype=np.float32)
    full_original_tau[motor] = tau.astype(np.float32)
    patched = patch_tau(full_original_tau, tau_ids, tau_values)
    exact(patched[tau_ids], native_tau_values, '428 native patch values')
    exact(patched[motor], init_motor_tau, '391 actual native MN taus')
    np.savez_compressed(out/'initializer-parameters.npz',
        actual_e108_w=selected_w, original_signed_edge_weights=edge_original,
        edge_ids=edge_ids, edge_rows=edge_rows, edge_cols=edge_cols,
        raw_efficacy=raw_eff, initial_efficacy=init_eff,
        initial_tau_scales=np.ones(6), motor_ids=motor,
        original_motor_tau=tau, native_motor_tau=init_motor_tau,
        tau_patch_ids=tau_ids, tau_patch_calibration_values=tau_values,
        tau_patch_actual_native_values=native_tau_values)
    save_json(out/'input-validation.json', {'E108_review_accepted': True,
        'E108_local_supported': False, 'parent_expected_exact': True,
        'E105_background_reuse': True,
        'E108_actual_float32_initializer_exact': True,
        'initial_efficacy_roundoff_projection_max_abs': projection,
        'mn_counts': motor_mask.sum(axis=1).tolist(),
        'edge_counts': edge_mask.sum(axis=1).tolist(),
        'native_tau_special_motor_id': 92104,
        'tau_patch_ids': len(tau_ids), 'unchanged_nonmotor_tau_ids': 37})
    if time.perf_counter()-started >= WHOLE_LIMIT_S-1:
        raise TimeoutError('Preflight exhausted E109 whole wall')

    with shadow_body_lifetime(original, out) as (body, body_before):
        qpos = np.repeat(body.d.qpos[None, :], len(train['selected']), axis=0)
        exact(qpos, static['qpos'], 'Unchanged original neutral qpos contexts')
        geometry = StaticFootGeometry(body, qpos)
        target_rows = np.searchsorted(np.flatnonzero(train['train_mask']),
                                      train['selected'])
        target_feet, jac = geometry.evaluate(train['target'][target_rows],
                                             jacobian=True)
        np.savez_compressed(out/'static-geometry-raw.npz', qpos=qpos,
            target_feet=target_feet, target_jacobian=jac,
            selected=train['selected'])
        exact(target_feet, static['target_feet'], 'E108 original neutral target feet')
        exact(jac, static['target_jacobian'], 'E108 original neutral Jacobian')
        cross = check_static_cross_jacobian(jac)
        foot_scale = np.maximum(target_feet.std(axis=0), .05)
        exact(foot_scale, static['foot_scale'], 'E108 original foot scale')
        np.savez_compressed(out/'static-geometry.npz', qpos=qpos,
            target_feet=target_feet, target_jacobian=jac,
            selected=train['selected'], foot_scale=foot_scale)
        save_json(out/'static-geometry-check.json', {
            'cross_leg_max_abs': cross, 'frozen_neutral_contexts': len(qpos),
            'body_clock_before': body_before.get('clock'),
            'E108_target_and_jacobian_exact': True})
        return _fit(out, started, ledger, body, geometry, train, static,
            previous_baseline, previous_prediction, e105_cal, e108_cal,
            base, current, selected_network, selected_w, original_w,
            edge_original, init_eff, motor_mask, edge_mask, tau)


def _fit(out, started, ledger, body, geometry, train, static,
         previous_baseline, previous_prediction, e105_cal, e108_cal,
         base, current, selected_network, selected_w, original_w,
         edge_original, init_eff, motor_mask, edge_mask, tau):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('E109 CUDA native autograd required')
    torch.set_num_threads(1)
    device = 'cuda'
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    source_t = tensor(train['source_rate'])
    background_t = tensor(train['background'])
    pool_t = tensor(train['pool'])
    moment_t = tensor(train['moment'])
    rest_t = tensor(train['rest'])
    tau_t = tensor(tau)
    a_t = tensor(train['a'])
    threshold_t = tensor(train['threshold'])
    cap_t = tensor(train['cap'])
    ltm_t = torch.as_tensor(train['ltm'], device=device)
    mask_t = torch.as_tensor(train['train_mask'], device=device)
    selected_t = torch.as_tensor(train['selected'], device=device)
    startup_t = torch.as_tensor(train['startup_mask'], device=device)
    source_rate = train['source_rate']
    background = train['background']
    current_w = train['current_w']
    exact(background, train['current']-source_rate@current_w.T,
          'Actual E105 W defines immutable latest background')

    def full_predict(weights, native_tau):
        current_t = source_t@weights.T+background_t
        z = cap_t*torch.tanh(a_t*(current_t-threshold_t)/cap_t)
        rate = linear_response(torch.relu(z), .005, native_tau)
        activation = endpoint_response(rate@pool_t.T, .005, .01)
        command = rest_t+.02*activation@moment_t.T
        tendon = .2*torch.tanh(activation[:, ltm_t]/10)
        return tuple(item.detach().cpu().numpy() for item in
                     (command, tendon, rate, activation, current_t))

    with torch.no_grad():
        bc, bt, br, ba, bi = full_predict(tensor(current_w), tau_t)
    ledger['baseline_validation_calls'] = 1
    np.savez_compressed(out/'baseline-prediction.npz', command=bc, tendon=bt,
        motor_rate=br, muscle_activation=ba, current=bi,
        measured_current=train['current'], actual_command=train['actual_command'],
        actual_activation=train['actual_activation'])
    baseline_errors = {}
    for key, actual in [('command', bc), ('tendon', bt), ('motor_rate', br),
                        ('muscle_activation', ba), ('current', bi)]:
        prior = previous_baseline[key]
        baseline_errors[key] = float(np.max(np.abs(actual-prior)))
    save_json(out/'baseline-reproduction.json', baseline_errors)
    if not all(np.isfinite(v).all() for v in (bc, bt, br, ba, bi)):
        raise ValueError('Nonfinite E105 original native baseline')
    if any(v > 1e-9 for v in baseline_errors.values()):
        raise ValueError('E105 original native baseline differs from E108')
    actual_command = train['actual_command']
    mask = train['train_mask']
    transfer = float(np.sqrt(np.mean((bc[mask]-actual_command[mask])**2)))
    target_error = float(np.sqrt(np.mean((actual_command[mask]-train['target'])**2)))
    startup_mask = train['startup_mask']
    q_error = bc[startup_mask]-actual_command[startup_mask]
    actual_tendon = .2*np.tanh(train['actual_activation'][:, train['ltm']]/10)
    original_startup = startup_screen(actual_command, actual_tendon,
        train['startup_target_command'], train['startup_target_tendon'],
        train['time_s'], startup_mask)
    fidelity = {'latest_steady_transfer_rms_rad': transfer,
        'latest_actual_to_target_rms_rad': target_error,
        'latest_transfer_ratio': transfer/target_error,
        'initial99_latest_actual_reconstruction': {
            'q_RMS_rad': float(np.sqrt(np.mean(q_error*q_error))),
            'q_max_abs_rad': float(np.max(np.abs(q_error)))},
        'E105_actual_vs_original_E100_E103_startup_targets': original_startup,
        'required_ratio_at_most': .1,
        'required_initial99_q_RMS_at_most_rad': .002,
        'required_initial99_q_max_at_most_rad': .01}
    save_json(out/'baseline-fidelity.json', fidelity)
    previous_fidelity = json.loads((BASE/'e108-block-constrained-recruitment-v2/baseline-fidelity.json').read_text(encoding='utf-8'))
    for key in ('latest_steady_transfer_rms_rad', 'latest_actual_to_target_rms_rad',
                'latest_transfer_ratio'):
        np.testing.assert_allclose(fidelity[key], previous_fidelity[key], rtol=0, atol=1e-12)
    for key, value in fidelity['initial99_latest_actual_reconstruction'].items():
        np.testing.assert_allclose(value,
            previous_fidelity['initial99_latest_actual_reconstruction'][key],
            rtol=0, atol=1e-12)
    if not original_startup['supported']:
        raise ValueError('Original E105 startup gate no longer holds')
    if (not math.isfinite(fidelity['latest_transfer_ratio'])
            or fidelity['latest_transfer_ratio'] > .1
            or fidelity['initial99_latest_actual_reconstruction']['q_RMS_rad'] > .002
            or fidelity['initial99_latest_actual_reconstruction']['q_max_abs_rad'] > .01):
        return {'experiment': 'M3-E109-native-motor-kinetics',
            'validity': 'method_not_tested_fidelity', 'local_supported': False,
            'candidate': False, 'fidelity': fidelity,
            'actual_neural_s': 0., 'actual_physical_s': 0.}
    baseline_feet = geometry.evaluate(actual_command[train['selected']])
    baseline_metrics = metrics(actual_command, baseline_feet, actual_tendon,
        mask, train['target'],
        static['target_feet'], train['joint_scale'], static['foot_scale'])
    np.savez_compressed(out/'baseline-static-feet.npz', feet=baseline_feet,
                        selected=train['selected'])
    prior_baseline_metrics = json.loads((BASE/'e108-block-constrained-recruitment-v2/baseline-output.json').read_text(encoding='utf-8'))
    for key in baseline_metrics:
        np.testing.assert_allclose(baseline_metrics[key], prior_baseline_metrics[key],
                                   rtol=0, atol=1e-9)
    save_json(out/'baseline-output.json', baseline_metrics)

    with torch.no_grad():
        ic, it, ir, ia, ii = full_predict(tensor(selected_w), tau_t)
    ledger['initializer_validation_calls'] = 1
    init_feet = geometry.evaluate(ic[train['selected']])
    np.savez_compressed(out/'initializer-prediction.npz', command=ic, tendon=it,
        motor_rate=ir, muscle_activation=ia, current=ii, feet=init_feet,
        actual_float32_source_weights=selected_w, actual_float32_motor_tau=tau.astype(np.float32))
    init_errors = {}
    for key, actual in [('command', ic), ('tendon', it), ('motor_rate', ir),
                        ('muscle_activation', ia), ('current', ii), ('feet', init_feet)]:
        init_errors[key] = float(np.max(np.abs(actual-previous_prediction[key])))
    save_json(out/'initializer-reproduction.json', init_errors)
    if any(not math.isfinite(v) or v > 1e-9 for v in init_errors.values()):
        raise ValueError('E108 actual-float32 initializer native response mismatch')
    ledger['preflight_wall_s'] = time.perf_counter()-started
    save_json(out/'execution-ledger.json', ledger)
    if ledger['preflight_wall_s'] >= WHOLE_LIMIT_S-FIT_LIMIT_S:
        raise TimeoutError('E109 preflight consumed whole-fit reserved wall')

    fit_started = time.perf_counter()
    ledger['fit_start_elapsed_s'] = fit_started-started
    # Reserve 0.5 s for persistence before each wall bound.
    fit_deadline = min(fit_started+FIT_LIMIT_S-.5, started+WHOLE_LIMIT_S-.5)
    histories = {}
    chosen = {}
    summaries = {}
    edge_ids = train['edge_ids']
    edge_rows = train['edge_rows']
    edge_cols = train['edge_cols']

    class LegFoot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, commands, leg):
            values = commands.detach().cpu().numpy()
            points = np.empty((len(values), 3))
            derivative = np.empty((len(values), 3, 7))
            jac_world = np.zeros((3, body.m.nv))
            for row, q in enumerate(values):
                geometry.data.qpos[:] = geometry.qpos[row]
                geometry.data.qpos[geometry.qpos_ids[7*leg:7*(leg+1)]] = q
                geometry.mj.mj_kinematics(body.m, geometry.data)
                geometry.mj.mj_comPos(body.m, geometry.data)
                rotation = geometry.data.xmat[geometry.thorax].reshape(3, 3)
                origin = geometry.data.xpos[geometry.thorax]
                points[row] = (geometry.data.xpos[geometry.tips[leg]]-origin)@rotation
                geometry.mj.mj_jacBody(body.m, geometry.data, jac_world, None,
                                       int(geometry.tips[leg]))
                derivative[row] = rotation.T@jac_world[:,
                    geometry.qvel_ids[7*leg:7*(leg+1)]]
            ctx.save_for_backward(torch.as_tensor(derivative,
                dtype=commands.dtype, device=commands.device))
            return torch.as_tensor(points, dtype=commands.dtype, device=commands.device)

        @staticmethod
        def backward(ctx, gradient):
            derivative, = ctx.saved_tensors
            return torch.einsum('tx,txj->tj', gradient, derivative), None

    for leg, name in enumerate(LEGS):
        if time.perf_counter() >= fit_deadline:
            summaries[name] = {'stop': 'whole_fit_wall_limit_before_block',
                'unique_native_evaluations': 0, 'best_any': None,
                'best_feasible': None}
            save_json(out/f'block-{name}-summary.json', summaries[name])
            continue
        block_started = time.perf_counter()
        ledger['active_leg'] = name
        ledger['active_leg_start_elapsed_s'] = block_started-started
        save_json(out/'execution-ledger.json', ledger)
        block_deadline = min(block_started+LEG_LIMIT_S-.5, fit_deadline)
        rows = np.flatnonzero(motor_mask[leg])
        edges = np.flatnonzero(edge_mask[leg])
        local_rows = np.searchsorted(rows, edge_rows[edges])
        if not np.array_equal(rows[local_rows], edge_rows[edges]):
            raise ValueError('Fitted edge outside its one leg MN block')
        local_cols = edge_cols[edges]
        n = len(edges)
        x0 = np.r_[np.log(init_eff[edges]), 0.]
        signed = tensor(edge_original[edges].astype(float))
        original_local = tensor(original_w[rows])
        background_local = background_t[:, rows]
        tau_local, a_local, threshold_local, cap_local = (
            arr[rows] for arr in (tau_t, a_t, threshold_t, cap_t))
        pool_local = pool_t[:, rows]
        moment_local = moment_t[7*leg:7*(leg+1)]
        rest_local = rest_t[7*leg:7*(leg+1)]
        target_local = tensor(train['target'][:, 7*leg:7*(leg+1)])
        target_foot = tensor(static['target_feet'][:, leg])
        q_scale = tensor(train['joint_scale'][7*leg:7*(leg+1)])
        f_scale = tensor(static['foot_scale'][leg])
        startup_q_local = tensor(train['startup_target_command'][:, 7*leg:7*(leg+1)])
        startup_tendon_local = tensor(train['startup_target_tendon'][:, leg])
        edge_row_t = torch.as_tensor(local_rows, device=device)
        edge_col_t = torch.as_tensor(local_cols, device=device)

        def evaluate(x, index):
            parameter = torch.tensor(x, dtype=torch.float64, device=device,
                                     requires_grad=True)
            efficacy = parameter[:-1].exp()
            scale = parameter[-1].exp()
            weights = original_local.clone()
            weights[edge_row_t, edge_col_t] = signed*efficacy
            local_current = source_t@weights.T+background_local
            z = cap_local*torch.tanh(a_local*(local_current-threshold_local)/cap_local)
            rate = linear_response(torch.relu(z), .005, tau_local*scale)
            activation = endpoint_response(rate@pool_local.T, .005, .01)
            command = rest_local+.02*activation@moment_local.T
            tendon = .2*torch.tanh(activation[:, ltm_t[leg]]/10)
            feet = LegFoot.apply(command[selected_t], leg)
            q_loss = ((command[mask_t]-target_local)/q_scale).square().mean()
            foot_loss = ((feet-target_foot)/f_scale).square().mean()
            tendon_loss = ((tendon[mask_t]-.2)/.2).square().mean()
            edge_prior = (6e-5*n/1572)*(efficacy-1).square().mean()
            tau_prior = 1e-5*(scale-1).square()
            objective = q_loss+foot_loss+tendon_loss+edge_prior+tau_prior
            dq = command[startup_t]-startup_q_local
            dl = tendon[startup_t]-startup_tendon_local
            ratios = torch.stack((dq.square().mean()/.02**2,
                dq.square().max()/.05**2, dl.square().mean()/.01**2,
                dl.square().max()/.03**2))
            parts = torch.stack((q_loss, foot_loss, tendon_loss,
                                 edge_prior, tau_prior))
            arrays = (command, tendon, rate, activation, feet, local_current,
                      objective, ratios, parts)
            if any(not torch.isfinite(item).all().item() for item in arrays):
                np.savez_compressed(out/f'technical-nonfinite-{name}-{index}.npz',
                    log_parameters=x, command=command.detach().cpu().numpy(),
                    tendon=tendon.detach().cpu().numpy(),
                    rate=rate.detach().cpu().numpy(),
                    activation=activation.detach().cpu().numpy(),
                    feet=feet.detach().cpu().numpy(),
                    current=local_current.detach().cpu().numpy(),
                    objective=float(objective.detach()),
                    ratios=ratios.detach().cpu().numpy())
                raise ValueError('Nonfinite native block output: '+name)
            g_obj = torch.autograd.grad(objective, parameter,
                                        retain_graph=True)[0].detach().cpu().numpy()
            g_ratio = torch.stack([torch.autograd.grad(ratios[i], parameter,
                retain_graph=i < 3)[0] for i in range(4)]).detach().cpu().numpy()
            if not np.isfinite(g_obj).all() or not np.isfinite(g_ratio).all():
                np.savez_compressed(out/f'technical-nonfinite-gradient-{name}-{index}.npz',
                    log_parameters=x, objective=float(objective.detach()),
                    ratios=ratios.detach().cpu().numpy(),
                    objective_gradient=g_obj, ratio_jacobian=g_ratio)
                raise ValueError('Nonfinite native autograd gradient: '+name)
            ratio_np = ratios.detach().cpu().numpy()
            record = {'index': index, 'leg': name,
                'objective': float(objective.detach()),
                'parts': parts.detach().cpu().numpy().tolist(),
                'ratios': ratio_np.tolist(),
                'optimizer_inequalities': (OPTIMIZER_RESERVE-ratio_np).tolist(),
                'selector_inequalities': (SELECTOR_RESERVE-ratio_np).tolist(),
                'strict_startup_feasible': bool(np.all(ratio_np <= SELECTOR_RESERVE)),
                'gradient_max_abs': float(np.max(np.abs(g_obj))),
                'gradient_l2': float(np.linalg.norm(g_obj)),
                'constraint_jacobian_max_abs': float(np.max(np.abs(g_ratio))),
                'constraint_jacobian_row_l2': np.linalg.norm(g_ratio, axis=1).tolist(),
                'log_theta': x[:-1].tolist(), 'log_tau_scale': float(x[-1]),
                'tau_scale': float(scale.detach()),
                'block_wall_s': time.perf_counter()-block_started,
                'fit_wall_s': time.perf_counter()-fit_started}
            with (out/f'history-{name}.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(json_safe(record), allow_nan=False)+'\n')
            ledger['leg_completed_evaluations'][name] = index
            save_json(out/'execution-ledger.json', ledger)
            histories[name].append(record)
            return {'objective': record['objective'], 'jac': g_obj,
                    'ratios': ratio_np, 'ratio_jac': g_ratio,
                    'record': record}

        histories[name] = []
        def on_start(index, x):
            ledger['leg_started_evaluations'][name] = index
            ledger['active_parameters'] = x.tolist()
            save_json(out/'execution-ledger.json', ledger)

        cache = NativeCache(evaluate, n+1, max_calls=LEG_MAX_CALLS,
            deadline=block_deadline, total_deadline=fit_deadline,
            on_start=on_start)
        try:
            _, termination = solve_block(cache, x0, n,
                                         out/f'optimizer-{name}.json')
            stop = termination['stop']
        except BlockLimit as exc:
            stop = str(exc)
            termination = optimizer_record(None, stop)
            save_json(out/f'optimizer-{name}.json', termination)
        ledger['leg_cache_hits'][name] = cache.hits
        best_any, best_feasible = choose_records(histories[name])
        for label, record in (('best-any', best_any),
                              ('best-feasible', best_feasible)):
            if record is not None:
                np.savez_compressed(out/f'{name}-{label}-coefficients.npz',
                    log_theta=np.asarray(record['log_theta']),
                    efficacy=np.exp(record['log_theta']),
                    log_tau_scale=record['log_tau_scale'],
                    tau_scale=record['tau_scale'], edge_ids=edge_ids[edges],
                    motor_ids=train['motor_ids'][edge_rows[edges]],
                    source_ids=train['source_ids'][edge_cols[edges]],
                    objective=record['objective'], ratios=record['ratios'])
        wall = time.perf_counter()-block_started
        if wall > LEG_LIMIT_S:
            stop = 'per_leg_wall_limit_exceeded'
        summaries[name] = {'stop': stop,
            'unique_native_evaluations': cache.calls,
            'cache_hits': cache.hits, 'wall_s': wall,
            'optimizer': termination,
            'best_any': {k: best_any[k] for k in
                ('index', 'objective', 'ratios', 'tau_scale', 'gradient_max_abs')}
                if best_any else None,
            'best_feasible': {k: best_feasible[k] for k in
                ('index', 'objective', 'ratios', 'tau_scale', 'gradient_max_abs')}
                if best_feasible else None}
        ledger['leg_wall_s'][name] = wall
        save_json(out/f'block-{name}-summary.json', summaries[name])
        save_json(out/'all-block-summaries.json', summaries)
        # Include block-summary persistence in its wall accounting.
        wall = time.perf_counter()-block_started
        if wall > LEG_LIMIT_S:
            summaries[name]['stop'] = 'per_leg_wall_limit_exceeded'
        summaries[name]['wall_s'] = wall
        ledger['leg_wall_s'][name] = wall
        save_json(out/f'block-{name}-summary.json', summaries[name])
        save_json(out/'all-block-summaries.json', summaries)
        save_json(out/'execution-ledger.json', ledger)
        if best_feasible is not None:
            chosen[name] = best_feasible
        ledger['active_leg'] = None
        ledger['active_leg_start_elapsed_s'] = None
        ledger.pop('active_parameters', None)
    ledger['fit_wall_s'] = time.perf_counter()-fit_started
    save_json(out/'execution-ledger.json', ledger)
    if any(name not in chosen for name in LEGS):
        return {'experiment': 'M3-E109-native-motor-kinetics',
            'validity': 'valid_completed_budget_limited_fit',
            'hypothesis_verdict': 'not_supported_no_six_feasible_blocks',
            'local_supported': False, 'candidate': False,
            'conditional_only': True, 'native_tau_hypothesis': True,
            'block_summaries': summaries, 'baseline_output': baseline_metrics,
            'fit_wall_s': ledger['fit_wall_s'], 'actual_neural_s': 0.,
            'actual_physical_s': 0.}

    efficacy = np.empty(1572)
    scales = np.empty(6)
    for leg, name in enumerate(LEGS):
        efficacy[edge_mask[leg]] = np.exp(chosen[name]['log_theta'])
        scales[leg] = chosen[name]['tau_scale']
    if (not np.isfinite(efficacy).all() or np.any(efficacy < .001*(1-8*np.finfo(float).eps))
            or np.any(efficacy > 10*(1+8*np.finfo(float).eps))):
        raise ValueError('Selected native efficacy outside original bounds')
    efficacy = np.clip(efficacy, .001, 10.)
    tau_ids, tau_values, native_tau_values, motor_tau = native_tau_patch(
        e105_cal, train['motor_ids'], tau, motor_mask, scales)
    np.savez_compressed(out/'selected-coefficients.npz',
        efficacy=efficacy, tau_scales=scales, motor_tau_actual_float32=motor_tau,
        tau_patch_ids=tau_ids, tau_patch_calibration_values=tau_values,
        tau_patch_actual_native_values=native_tau_values,
        edge_ids=edge_ids, edge_rows=edge_rows, edge_cols=edge_cols,
        original_signed_weights=edge_original)
    positions = {int(e): i for i, e in enumerate(e105_cal['edge_ids'])}
    values = e105_cal['values'].copy()
    for i, edge in enumerate(edge_ids):
        values[positions[int(edge)]] = np.float32(base.data[edge]*efficacy[i])
    changed = patch_weights(base, e105_cal['edge_ids'],
                            e105_cal['source_values'], values)
    exact(changed.indptr, current.indptr, 'Final original CSR row topology')
    exact(changed.indices, current.indices, 'Final original CSR columns')
    other = np.ones(len(base.data), dtype=bool)
    other[edge_ids] = False
    exact(changed.data[other], current.data[other], 'All final nonfit weights')
    exact(np.sign(changed.data[edge_ids]), np.sign(base.data[edge_ids]),
          'Original edge signs')
    actual_w = changed[train['motor_ids']][:, train['source_ids']].toarray().astype(float)
    with torch.no_grad():
        fc, ft, fr, fa, fi = full_predict(tensor(actual_w), tensor(motor_tau))
    ledger['final_validation_calls'] = 1
    final_feet = geometry.evaluate(fc[train['selected']]) if np.isfinite(fc).all() else np.full_like(static['target_feet'], np.nan)
    np.savez_compressed(out/'prediction.npz', command=fc, tendon=ft,
        motor_rate=fr, muscle_activation=fa, current=fi, feet=final_feet,
        actual_float32_source_weights=actual_w,
        motor_tau_actual_float32=motor_tau, tau_scales=scales,
        target=train['target'], target_feet=static['target_feet'],
        startup_mask=train['startup_mask'],
        startup_target_command=train['startup_target_command'],
        startup_target_tendon=train['startup_target_tendon'])
    if not all(np.isfinite(v).all() for v in (fc, ft, fr, fa, fi, final_feet)):
        raise ValueError('Nonfinite final actual-float32 native output')
    final_metrics = metrics(fc, final_feet, ft, mask, train['target'],
        static['target_feet'], train['joint_scale'], static['foot_scale'])
    relative = native_shape_screen(final_metrics, baseline_metrics)
    absolute = absolute_shape_screen(final_metrics)
    startup = startup_screen(fc, ft, train['startup_target_command'],
        train['startup_target_tendon'], train['time_s'], train['startup_mask'])
    supported = bool(relative['supported'] and absolute['supported'] and startup['supported'])
    outcome = {'experiment': 'M3-E109-native-motor-kinetics',
        'validity': 'valid_conditional_fit',
        'hypothesis_verdict': 'conditional_shape_and_startup_supported' if supported else 'not_supported',
        'local_supported': supported, 'candidate': True,
        'conditional_only': True, 'native_tau_hypothesis': True,
        'runtime_reference_phase': False, 'body_validated': False,
        'adoption': 'not_applicable', 'F1_F8': 'not_assessed',
        'causal_caveat': 'Actual E105 source/background include closed-loop feedback; offline fit does not prove candidate closed-loop causality.',
        'baseline_output': baseline_metrics, 'final_output': final_metrics,
        'relative_shape_screen': relative,
        'absolute_E101_shape_screen': absolute,
        'global_original99_startup_screen': startup,
        'block_summaries': summaries, 'tau_scales': scales.tolist(),
        'fit_wall_s': ledger['fit_wall_s'], 'actual_neural_s': 0.,
        'actual_physical_s': 0., 'all_nonfit_weights_exact': True,
        'CSR_topology_exact': True, 'original_edge_signs_exact': True,
        'all_37_nonmotor_tau_calibration_and_native_exact': True}
    save_json(out/'native-selection-and-screens.json', outcome)
    final_cal = {k: v.copy() for k, v in e105_cal.items()}
    final_cal['values'] = values
    final_cal['individual_edge_efficacies'] = efficacy
    final_cal['tau_ids'] = tau_ids
    final_cal['tau_values'] = tau_values
    final_cal['motor_tau_actual_float32'] = motor_tau
    final_cal['actual_native_tau_values'] = native_tau_values
    final_cal['native_tau_scales'] = scales
    final_cal['native_tau_hypothesis'] = np.array(True)
    final_cal['conditional_only'] = np.array(True)
    final_cal['runtime_reference_phase'] = np.array(False)
    for key, prior in e105_cal.items():
        if key not in ('values', 'individual_edge_efficacies', 'tau_ids', 'tau_values'):
            exact(final_cal[key], prior, 'E109 unchanged calibration '+key)
    np.savez_compressed(out/'calibration.npz', **final_cal)
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--graph', type=Path,
        default=ROOT/'data/acquisitions/banc888-windows-20260910/bundle')
    parser.add_argument('--out', type=Path,
        default=BASE/'e109-native-motor-kinetics')
    parser.add_argument('--prior-shared-wall-s', type=float, required=True)
    args = parser.parse_args()
    main(args.root, args.graph, args.out,
         prior_shared_wall_s=args.prior_shared_wall_s)
