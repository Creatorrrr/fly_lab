"""Fit one existing-edge, all-leg recruitment hypothesis to E86 native sources."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.verify_c_banc_observation import exact


def reference_at_phase(reference, phase, lf_column, reference_mask=None):
    """Interpolate actual late E85 LF phase with all 42 columns in one clock."""
    late = reference['time_s'] >= 5 if reference_mask is None else reference_mask
    coordinates = np.mod(reference['phase'][late, lf_column] / (2 * np.pi), 1)
    values = reference['command'][late]
    order = np.argsort(coordinates)
    x, y = coordinates[order], values[order]
    extended_x = np.r_[x[-1] - 1, x, x[0] + 1]
    extended_y = np.vstack((y[-1], y, y[0]))
    query = np.mod(phase, 1)
    return np.column_stack([np.interp(query, extended_x, extended_y[:, j])
                            for j in range(42)])


def own_leg_targets(reference, phases, offsets, mask, lf_column):
    """Assemble simultaneous six-leg labels without moving any source sample."""
    target = np.empty((int(mask.sum()), 42))
    for leg in range(6):
        reference_rows = reference_at_phase(reference,
            phases[mask, leg] - offsets[leg], lf_column)
        target[:, 7*leg:7*(leg+1)] = reference_rows[:, 7*leg:7*(leg+1)]
    return target


def own_leg_warm_target(bins, warm_muscles, phases, offsets, mask, muscle_legs):
    """Use each muscle's verified leg phase with the same 84-bin inverse."""
    return np.column_stack([np.interp(np.mod(phases[mask, muscle_legs[group]]
        - offsets[muscle_legs[group]], 1), np.r_[bins, 1],
        np.r_[warm_muscles[:, group], warm_muscles[0, group]])
        for group in range(warm_muscles.shape[1])])


def project_efficacies(values, present):
    """Remove only exponential roundoff, then enforce the original closed bounds."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Finite fitted efficacy values required')
    selected = values[present]
    tolerance = 8 * np.finfo(np.float64).eps
    if np.any(selected < .001 * (1 - tolerance)) or np.any(selected > 10 * (1 + tolerance)):
        raise ValueError('Fitted efficacy exceeds numerical boundary roundoff')
    projected = values.copy()
    projected[present] = np.clip(selected, .001, 10)
    return projected


def run(root, graph_path, out, *, reconstruct_from=None, own_leg=False):
    leased_bodies = []
    try:
        return _run(root, graph_path, out, leased_bodies,
            reconstruct_from=reconstruct_from, own_leg=own_leg)
    finally:
        for body in leased_bodies:
            body.close()


def _run(root, graph_path, out, leased_bodies, *, reconstruct_from=None, own_leg=False):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    previous_wall_s = 10.0 + 7.370871799997985
    if reconstruct_from is not None:
        # v3 ended after fitting but before persisting its coefficients. Its
        # timestamps establish about15s; charge a conservative20s for recovery.
        previous_wall_s += 20.
    remaining_wall_s = 180 - previous_wall_s
    if own_leg:
        if reconstruct_from is not None:
            raise ValueError('E89 is a new fit; E87 reconstruction is a separate mode')
        previous_wall_s = 0.
        remaining_wall_s = 180.
    e86 = root / 'e86-native-source-cadence'
    e85 = root / 'e85-space-compatible-cadence-reference'
    spec = {
        'experiment': 'M3-E87-native-all-leg-recruitment',
        'hypothesis': 'Actual simultaneous slow native source activity can conditionally recruit all 42 joints, six 3D foot paths and six long-tendon outputs through existing motor edges.',
        'basis': 'E86 all six E1 cadence screens passed; only LF-RF phase passed. Preserve actual middle/hind phase drift in one common LF clock.',
        'labels': 'Interpolate all 42 late E85 recorded command columns against their common LF phase at actual E86 LF peak-interpolated phase. Train only bracketed [.5,last LF peak). Labels are offline and never enter the candidate runtime.',
        'intervention': 'Fit positive original-source efficacies [.001,10] shared by anatomical muscle and 47 existing source nodes on existing source-to-MN edges. Preserve other edges including navigation, all nonmotor edges, 32 E86 taus, fixed moment/rest/pool/neutral body and full initial state.',
        'surrogate': 'Actual E86 endpoint current minus selected source contribution is retained as conditional background. Zero-state causal piecewise-linear 5ms first-order MN response uses actual MN tau; 10ms muscle uses endpoint hold. Compare baseline prediction to recorded E86 commands before fitting; require transfer RMS/reference discrepancy RMS<=.25.',
        'objective': 'Mean normalized 42 joint and 6x3 static-foot residual, six tendon near .2, plus 1e-5 mean squared efficacy displacement. All-time joint/tendon; shadow-FK stride20. Adam log efficacy lr.03 <=750 updates and <=150 fit wall seconds; one candidate, no phase shifts or grids.',
        'limits': 'No neural/physical step beyond candidate constructor. Total <=180 wall seconds; one CUDA owner. No F3-F8/adoption claim.',
        'e86_signals_sha256': sha256(e86 / 'recruitment-signals.npz'),
        'e86_calibration_sha256': sha256(e86 / 'calibration.npz'),
        'e85_reference_sha256': sha256(e85 / 'commands.npz'),
        'helper_sha256': sha256(Path(__file__).with_name('banc_m3_causal_recruitment.py')),
        'tool_sha256': sha256(__file__),
        'preflight_correction': 'Earlier unoptimized attempts rejected signed inhibitory edges, then calculated baseline from original W instead of E86 patched W. Both outputs remain preserved as invalid implementation diagnostics; this run retains signed original W times positive efficacy.',
        'earlier_execution_wall_s': {'v1_conservative_estimate': 10.0,
            'v2_measured': 7.370871799997985},
        'cumulative_wall_limit_s': 180,
    }
    if own_leg:
        spec.update(experiment='M3-E89-own-leg-native-recruitment',
            hypothesis='Own-leg E1 phase alignment of offline E85 labels improves within-leg conditional shape while preserving simultaneous E86 source input and interleg drift.',
            basis='E88 healthy original10s but F3/F4/F5 failed; E86 native source middle/hind LR failed. E87 common-LF conditional fit is the fixed comparator.',
            labels='Each leg seven E85 command columns sampled at its actual own E1 peak-interpolated phase minus its independently verified constant E85 offset to LF. All six phases bracket the common train mask; actual source and background rows remain in original order and time.',
            objective='Identical E87 42joint+six3Dfoot+sixLTM normalized objective, causal filters, original W times575 positive efficacy parameters, Adam .03 max750. Only label and aligned warm target change.',
            limits='One CUDA conditional fit <=180s total/<=150s optimizer,0 neural/physics advance. Candidate is not actual body or F3-F8 validation.',
            e87_frozen_result_sha256=sha256(root/'e87-native-all-leg-recruitment-recovered/result.json'))
        spec.pop('preflight_correction')
        spec.pop('earlier_execution_wall_s')
        spec.pop('cumulative_wall_limit_s')
    if reconstruct_from is not None:
        previous_spec = json.loads((reconstruct_from / 'spec.json').read_text())
        if (previous_spec['experiment'] != spec['experiment']
                or previous_spec['helper_sha256'] != spec['helper_sha256']):
            raise ValueError('Same E87 experiment and causal numerical helper required for reconstruction')
        spec['recovery_amendment'] = {
            'parent_review_round': 2, 'correction_owner': 'parent session',
            'source': str(reconstruct_from),
            'reason': 'v3 finished750updates but exponential endpoint roundoff failed a strict bound before besttheta was saved. Preserve the incomplete v3 and reconstruct that same numerical fit, not a new candidate search.',
            'unchanged': 'All training arrays, warm start, objective,750updates,optimizer settings and causal numerical helper. Require each recorded loss component and the entire unprojected prediction to match v3 exactly before candidate construction.',
            'changes': 'Save best coefficients before validation, project only exponential roundoff into original closed[.001,10] bounds, and report actual recorded E86 output as comparator.',
            'replayed_optimization_counts_as_work': True,
            'v3_prior_wall_conservative_estimate_s': 20.,
            'scientific_candidate_search_count': 1,
        }
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'causal-helper.py').write_bytes(Path(__file__).with_name('banc_m3_causal_recruitment.py').read_bytes())
    cal = {key: value.copy() for key, value in np.load(e86 / 'calibration.npz').items()}
    original_cal = {key: value.copy() for key, value in cal.items()}
    data = np.load(e86 / 'recruitment-signals.npz')
    meta = np.load(e86 / 'recruitment-metadata.npz')
    ref = np.load(e85 / 'commands.npz')
    from flygym_demo.complex_terrain import FlyBodyPreprogrammedSteps

    reference_leg_order = tuple(FlyBodyPreprogrammedSteps().legs)
    if reference_leg_order != ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'):
        raise ValueError('Unexpected E85 FlyBody reference leg order')
    lf_column = reference_leg_order.index('lf')
    graph = GraphStore.load(graph_path)
    original = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    sources = np.unique(np.r_[cal['input_ids'], cal['intrinsic_tonic_ids'], cal['coupling_ids']])
    motor = cal['motor_ids']
    observed = data['observed_ids']
    lookup = {int(value): i for i, value in enumerate(observed)}
    source_cols = np.array([lookup[int(i)] for i in sources])
    motor_cols = np.array([lookup[int(i)] for i in motor])
    if (len(sources) != 47 or len(motor) != 391 or len(np.unique(observed)) != 440
            or not np.array_equal(meta['observed_ids'], observed)
            or not np.array_equal(meta['source_ids'], sources)):
        raise ValueError('Exact 47-source and 391-MN E86 roster required')
    times = data['time_s']
    np.testing.assert_allclose(times, np.arange(2001) * .005, atol=1e-12)
    e1 = int(cal['core_ids'][(cal['core_groups'] == 2) & (cal['core_cell_types'] == 0)][0])
    peaks, _ = find_peaks(data['rate'][:, lookup[e1]], prominence=20)
    peak_times = times[peaks]
    if len(peak_times) < 20 or peak_times[0] > .5 or peak_times[-1] < 9.5:
        raise ValueError('Observed LF E1 complete reference peaks required')
    phase = np.interp(times, peak_times, np.arange(len(peaks)), left=np.nan, right=np.nan)
    phases = None
    offsets = None
    if own_leg:
        phases = np.empty((len(times), 6))
        offsets = np.empty(6)
        group_by_leg = {'rh': 0, 'rf': 1, 'lf': 2, 'rm': 3, 'lh': 4, 'lm': 5}
        last_peaks = []
        for leg, name in enumerate(reference_leg_order):
            group = group_by_leg[name]
            ids = cal['core_ids'][(cal['core_groups'] == group)
                & (cal['core_cell_types'] == 0)]
            if len(ids) != 1:
                raise ValueError('One existing E1 source per leg required')
            found, _ = find_peaks(data['rate'][:, lookup[int(ids[0])]], prominence=20)
            pts = times[found]
            if len(pts) < 20 or pts[0] > .5 or pts[-1] < 9.5:
                raise ValueError('All own-leg E1 phases must bracket training data')
            phases[:, leg] = np.interp(times, pts, np.arange(len(pts)),
                left=np.nan, right=np.nan)
            last_peaks.append(pts[-1])
            delta = np.mod((ref['phase'][:, leg] - ref['phase'][:, lf_column])
                / (2*np.pi), 1)
            reference_offset = float(np.median(delta))
            if (min(abs(reference_offset),abs(reference_offset-.5),
                    abs(reference_offset-1)) > 1e-10
                    or np.max(np.abs(np.mod(delta-reference_offset+.5,1)-.5)) > 1e-10):
                raise ValueError('E85 own-leg tripod offset is not constant 0 or half cycle')
            offsets[leg] = 0. if min(abs(reference_offset),abs(reference_offset-1))<1e-10 else .5
        np.testing.assert_array_equal(phases[:, lf_column], phase)
        mask = (times >= .5) & (times < min(last_peaks))
        mask &= np.isfinite(phases).all(axis=1)
        target = own_leg_targets(ref, phases, offsets, mask, lf_column)
        np.savez_compressed(out/'own-leg-phase.npz',time_s=times,
            leg_order=np.asarray(reference_leg_order),phase=phases,offsets=offsets,
            train_mask=mask,source_group_by_leg=np.array([group_by_leg[name]
                for name in reference_leg_order]),last_peaks=last_peaks)
    else:
        mask = (times >= .5) & (times < peak_times[-1]) & np.isfinite(phase)
        target = reference_at_phase(ref, phase[mask], lf_column)
    ref_train = (ref['time_s'] >= 5) & (ref['time_s'] < 8)
    ref_holdout = (ref['time_s'] >= 8) & (ref['time_s'] < 10)
    reconstructed = reference_at_phase(ref,
        ref['phase'][ref_holdout, lf_column] / (2 * np.pi), lf_column, ref_train)
    reference_residual = float(np.sqrt(np.mean((reconstructed - ref['command'][ref_holdout]) ** 2)))
    np.savez_compressed(out / 'reference-extraction-validation.npz',
        leg_order=np.asarray(reference_leg_order), lf_column=lf_column,
        heldout_time_s=ref['time_s'][ref_holdout],
        heldout_command=ref['command'][ref_holdout], reconstructed=reconstructed,
        residual_rms=reference_residual)
    if reference_residual > .002:
        raise ValueError('E85 held-out common-phase interpolation exceeds .002 rad RMS')
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, motor)
    changed_matrix = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    current_w = changed_matrix[motor][:, sources].toarray().astype(float)
    original_w = base[motor][:, sources].toarray().astype(float)
    measured_sources = data['rate'][:, source_cols].astype(float)
    full_current = (data['recurrent_current'][:, motor_cols]
                    + data['drive'][:, motor_cols] - data['adaptation'][:, motor_cols]).astype(float)
    background = full_current - measured_sources @ current_w.T
    if np.any(data['adaptation'][:, motor_cols] != 0) or np.any(meta['_cell_gain'][motor_cols] != 0):
        raise ValueError('Original MN adaptation must remain zero')
    tau = meta['tau'][motor_cols].astype(float)
    a = meta['a'][motor_cols].astype(float)
    threshold = meta['threshold'][motor_cols].astype(float)
    cap = meta['cap'][motor_cols].astype(float)
    if (not np.allclose(a, 1) or not np.allclose(threshold, 7.5)
            or not np.allclose(cap, 200)
            or not np.all(np.isin(np.round(tau, 7), np.round([.02, .014046887330582595], 7)))):
        raise ValueError('Expected measured MN response physiology required')
    pool, moment, rest = cal['pool'], cal['moment'], cal['rest']
    membership, ltm = cal['membership'], cal['ltm']
    present = (pool @ (original_w != 0).astype(float)) > 0
    if present.shape != (96, 47) or len(ltm) != 6:
        raise ValueError('Original existing source-to-motor anatomy required')
    selected = np.flatnonzero(mask)[::20]
    body = new_body(original)
    leased_bodies.append(body)
    body_before = body.snapshot()
    qpos = np.repeat(body.d.qpos[None, :], len(selected), axis=0)
    geometry = StaticFootGeometry(body, qpos)
    target_feet = geometry.evaluate(target[np.searchsorted(np.flatnonzero(mask),selected)])
    foot_scale = np.maximum(target_feet.std(axis=0), .05)
    joint_scale = np.maximum(target.std(axis=0), .025)
    np.savez_compressed(out / 'training-inputs.npz', time_s=times, phase=phase,
        train_mask=mask, geometry_indices=selected, source_ids=sources, motor_ids=motor,
        source_rate=measured_sources, actual_motor_current=full_current,
        retained_background=background, original_source_weights=original_w,
        current_source_weights=current_w, target_command=target,
        target_feet=target_feet, joint_scale=joint_scale, foot_scale=foot_scale,
        motor_tau=tau, motor_a=a, motor_threshold=threshold, motor_cap=cap,
        present=present, membership=membership, pool=pool, moment=moment, rest=rest,
        reference_extraction_rms=reference_residual,
        **({'own_leg_phase':phases,'own_leg_offsets':offsets} if own_leg else {}))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA fit required')
    torch.set_num_threads(1)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device='cuda')
    source_t, background_t, original_t = map(tensor, (measured_sources, background, original_w))
    pool_t, moment_t, rest_t = map(tensor, (pool, moment, rest))
    tau_t, a_t, threshold_t, cap_t = map(tensor, (tau, a, threshold, cap))
    target_t, target_foot_t = map(tensor, (target, target_feet))
    joint_scale_t, foot_scale_t = map(tensor, (joint_scale, foot_scale))
    membership_t = torch.as_tensor(membership, device='cuda')
    mask_t = torch.as_tensor(mask, device='cuda')
    selected_t = torch.as_tensor(selected, device='cuda')
    ltm_t = torch.as_tensor(ltm, device='cuda')
    present_t = torch.as_tensor(present, device='cuda')

    def predict(theta, *, with_feet=True, direct_weights=None):
        weights = direct_weights if direct_weights is not None else original_t * theta[membership_t]
        current = source_t @ weights.T + background_t
        forcing = torch.relu(cap_t * torch.tanh(a_t * (current - threshold_t) / cap_t))
        rate = linear_response(forcing, .005, tau_t)
        activation = endpoint_response(rate @ pool_t.T, .005, .01)
        command = rest_t + .02 * activation @ moment_t.T
        tendon = .2 * torch.tanh(activation[:, ltm_t] / 10)
        feet = geometry.tensor(command[selected_t]) if with_feet else None
        return command, tendon, feet, rate, activation

    with torch.no_grad():
        baseline_command, baseline_tendon, _, baseline_rate, baseline_activation = predict(
            None, with_feet=False, direct_weights=tensor(current_w))
    baseline_command = baseline_command.cpu().numpy()
    baseline_transfer_rms = float(np.sqrt(np.mean((baseline_command[mask]
        - data['decoded_command'][mask]) ** 2)))
    baseline_reference_rms = float(np.sqrt(np.mean((data['decoded_command'][mask] - target) ** 2)))
    fidelity = baseline_transfer_rms / baseline_reference_rms
    np.savez_compressed(out / 'baseline-prediction.npz', command=baseline_command,
        tendon=baseline_tendon.cpu().numpy(), motor_rate=baseline_rate.cpu().numpy(),
        muscle_activation=baseline_activation.cpu().numpy(), actual_command=data['decoded_command'],
        transfer_rms=baseline_transfer_rms, reference_rms=baseline_reference_rms,
        transfer_ratio=fidelity)
    if not np.isfinite(fidelity) or fidelity > .25:
        write_json(out / 'result.json', {'experiment': spec['experiment'],
            'validity': 'surrogate_method_failure', 'baseline_transfer_ratio': fidelity,
            'baseline_transfer_rms_rad': baseline_transfer_rms,
            'reference_command_rms_rad': baseline_reference_rms,
            'reference_extraction_rms_rad': reference_residual,
            'actual_neural_s': 0, 'actual_physical_s': 0,
            'wall_s': time.perf_counter() - began})
        return
    baseline_tendon = baseline_tendon.cpu().numpy()
    baseline_feet = geometry.evaluate(data['decoded_command'][selected])
    actual_baseline_tendon = .2 * np.tanh(data['muscle_activation'][:, ltm] / 10)

    def output_metrics(commands, feet, tendons):
        joint_error = (commands[mask] - target) / joint_scale
        foot_error = (feet - target_feet) / foot_scale
        tendon_error = (tendons[mask] - .2) / .2
        return {'joint': float(np.mean(joint_error ** 2)),
            'foot': float(np.mean(foot_error ** 2)),
            'tendon': float(np.mean(tendon_error ** 2)),
            'combined': float(np.mean(joint_error ** 2) + np.mean(foot_error ** 2)
                + np.mean(tendon_error ** 2)),
            'per_leg_joint_rms_rad': [float(np.sqrt(np.mean(
                (commands[mask, 7*i:7*i+7] - target[:, 7*i:7*i+7]) ** 2)))
                for i in range(6)],
            'per_leg_foot_xyz_rms_mm': np.sqrt(np.mean((feet - target_feet) ** 2,
                axis=0)).tolist(),
            'per_leg_tendon_rms': np.sqrt(np.mean((tendons[mask] - .2) ** 2,
                axis=0)).tolist()}

    baseline_metrics = output_metrics(data['decoded_command'], baseline_feet, actual_baseline_tendon)
    np.savez_compressed(out / 'baseline-output.npz', feet=baseline_feet,
        baseline_combined_loss=baseline_metrics['combined'])
    previous_metrics = None
    previous_command = None
    if own_leg:
        previous = np.load(root/'e87-native-all-leg-recruitment-recovered/prediction.npz')
        previous_command = previous['command']
        previous_tendon = previous['tendon']
        if previous_command.shape != (len(times),42) or previous_tendon.shape != (len(times),6):
            raise ValueError('Frozen E87 full timeline prediction required')
        previous_feet = geometry.evaluate(previous_command[selected])
        previous_metrics = output_metrics(previous_command,previous_feet,previous_tendon)
        np.savez_compressed(out/'frozen-e87-comparison.npz',command=previous_command,
            tendon=previous_tendon,feet=previous_feet)
    # Nonnegative anatomical inverse at 84 common phases, then conditional
    # current inversion at all real training instants for each muscle.
    bins = np.linspace(0, 1, 84, endpoint=False)
    reference_bins = reference_at_phase(ref, bins, lf_column)
    inverse = np.vstack((.02 * moment, .0001 * np.eye(96)))
    warm_muscles = []
    for row in reference_bins:
        solution = lsq_linear(inverse, np.r_[row - rest, np.zeros(96)],
            bounds=(0, 199), method='bvls', max_iter=300)
        if not solution.success:
            raise ValueError('Bounded anatomical inverse failed')
        warm_muscles.append(solution.x)
    warm_muscles = np.asarray(warm_muscles)
    warm_muscles[:, ltm] = np.maximum(warm_muscles[:, ltm], 30)
    if own_leg:
        anatomy_source=json.loads((e86/'muscle-provenance.json').read_text())
        groups=anatomy_source['muscles']
        if len(groups)!=96 or any(group['leg'] not in reference_leg_order
            for group in groups):
            raise ValueError('96 original muscles with verified leg membership required')
        muscle_legs=np.array([reference_leg_order.index(group['leg']) for group in groups])
        if np.any(np.bincount(muscle_legs,minlength=6)==0):
            raise ValueError('Each original leg requires its anatomical muscle groups')
        for group,info in enumerate(groups):
            cols=np.asarray(info['columns'],dtype=int)
            if not np.array_equal(np.flatnonzero(pool[group]),cols) and not np.array_equal(
                    np.sort(np.flatnonzero(pool[group])),np.sort(cols)):
                raise ValueError('Anatomical muscle motor membership changed')
        warm_target=own_leg_warm_target(bins,warm_muscles,phases,offsets,
            mask,muscle_legs)
    else:
        warm_target = np.column_stack([np.interp(np.mod(phase[mask], 1),
            np.r_[bins, 1], np.r_[warm_muscles[:, g], warm_muscles[0, g]])
            for g in range(96)])
    initial_theta = np.ones((96, 47))
    source_train = measured_sources[mask]
    for group in range(96):
        columns = np.flatnonzero(present[group])
        if not len(columns):
            continue
        mean_w = pool[group] @ original_w
        design = source_train[:, columns] * mean_w[columns]
        desired = 7.5 + 200 * np.arctanh(np.clip(warm_target[:, group] / 200, 0, .995))
        desired -= background[mask] @ pool[group]
        ridge = .01 * np.eye(len(columns))
        fit = lsq_linear(np.vstack((design / 200, ridge)),
            np.r_[desired / 200, .01 * np.ones(len(columns))],
            bounds=(.001, 10), method='bvls', max_iter=300)
        if not fit.success:
            raise ValueError('Bounded source-current inverse failed')
        initial_theta[group, columns] = fit.x
    np.savez_compressed(out / 'warm-start.npz', reference_phase_bins=bins,
        warm_muscles=warm_muscles, warm_target=warm_target,
        initial_theta=initial_theta, present=present,
        **({'muscle_leg_indices':muscle_legs} if own_leg else {}))
    replay_history = None
    if reconstruct_from is not None:
        for filename in ('training-inputs.npz', 'warm-start.npz', 'reference-extraction-validation.npz'):
            prior = np.load(reconstruct_from / filename)
            current = np.load(out / filename)
            if set(prior.files) != set(current.files):
                raise ValueError('Reconstruction input fields differ: ' + filename)
            for key in prior.files:
                np.testing.assert_array_equal(prior[key], current[key], err_msg=filename + ' ' + key)
        replay_history = json.loads((reconstruct_from / 'history.json').read_text())
        if replay_history[-1]['update'] != 750:
            raise ValueError('Original completed750update trajectory required')
    if time.perf_counter() - began > remaining_wall_s - 30:
        raise TimeoutError('E87 preparation left insufficient total wall time for fit')
    log_theta = torch.nn.Parameter(tensor(np.log(initial_theta)))
    optimizer = torch.optim.Adam([log_theta], lr=.03)
    history, best_loss, best = [], math.inf, None
    fit_began = time.perf_counter()
    prefit_wall_s = fit_began - began
    stop = 'max_updates'
    for update in range(751):
        optimizer.zero_grad(set_to_none=True)
        theta = log_theta.exp()
        command, tendon, feet, _, _ = predict(theta)
        loss_joint = ((command[mask_t] - target_t) / joint_scale_t).square().mean()
        loss_foot = ((feet - target_foot_t) / foot_scale_t).square().mean()
        loss_tendon = ((tendon[mask_t] - .2) / .2).square().mean()
        loss_prior = 1e-5 * (theta[present_t] - 1).square().mean()
        loss = loss_joint + loss_foot + loss_tendon + loss_prior
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError('Nonfinite conditional objective')
        if value < best_loss:
            best_loss, best = value, log_theta.detach().clone()
        if update % 25 == 0:
            item = {'update': update, 'loss': value, 'joint': float(loss_joint.detach()),
                'foot': float(loss_foot.detach()), 'tendon': float(loss_tendon.detach()),
                'prior': float(loss_prior.detach()), 'fit_wall_s': time.perf_counter() - fit_began}
            history.append(item)
            write_json(out / 'history.json', history)
            np.savez_compressed(out / 'best-fit-checkpoint.npz',
                best_log_theta=best.cpu().numpy(), current_log_theta=log_theta.detach().cpu().numpy(),
                best_loss=best_loss, observed_update=update, source_ids=sources, present=present)
            print(json.dumps(item), flush=True)
            if replay_history is not None:
                expected = replay_history[update // 25]
                for key in ('update', 'loss', 'joint', 'foot', 'tendon', 'prior'):
                    if item[key] != expected[key]:
                        raise ValueError('Reconstruction differs at update' + str(update) + ' ' + key)
        if time.perf_counter() - fit_began > 150 or time.perf_counter() - began > remaining_wall_s - 30:
            stop = 'fit_wall_limit' if time.perf_counter() - fit_began > 150 else 'total_wall_reserve'
            break
        if update == 750:
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_theta.clamp_(math.log(.001), math.log(10))
    optimizer_wall_s = time.perf_counter() - fit_began
    with torch.no_grad():
        raw_fitted = best.exp()
        command, tendon, feet, rate, activation = predict(raw_fitted)
    raw_theta = raw_fitted.cpu().numpy()
    np.savez_compressed(out / 'raw-fit-coefficients.npz', theta=raw_theta,
        log_theta=best.cpu().numpy(), source_ids=sources, present=present,
        best_loss=best_loss, update=update)
    raw_prediction = {'command': command.cpu().numpy(), 'tendon': tendon.cpu().numpy(),
        'feet': feet.cpu().numpy(), 'motor_rate': rate.cpu().numpy(),
        'muscle_activation': activation.cpu().numpy(),
        'target_command': target, 'target_feet': target_feet}
    np.savez_compressed(out / 'unprojected-prediction.npz', **raw_prediction)
    if reconstruct_from is not None:
        previous_prediction = np.load(reconstruct_from / 'prediction.npz')
        for key in previous_prediction.files:
            np.testing.assert_array_equal(previous_prediction[key], raw_prediction[key],
                err_msg='Reconstructed v3 prediction ' + key)
        write_json(out / 'reconstruction-exact.json', {
            'all_training_and_warm_arrays_exact': True,
            'all31logged_loss_components_exact': True, 'full_unprojected_prediction_exact': True,
            'optimization_execution_count_including_reconstruction': 2,
            'scientific_candidate_search_count': 1, 'actual_neural_s': 0, 'actual_physical_s': 0})
    fitted_theta = project_efficacies(raw_theta, present)
    np.savez_compressed(out / 'fit-coefficients.npz', theta=fitted_theta,
        raw_theta=raw_theta, source_ids=sources, present=present,
        projection_max_abs=float(np.max(abs(fitted_theta - raw_theta))))
    with torch.no_grad():
        fitted = tensor(fitted_theta)
        command, tendon, feet, rate, activation = predict(fitted)
    final_command = command.cpu().numpy()
    final_feet = feet.cpu().numpy()
    final_tendon = tendon.cpu().numpy()
    final_metrics = output_metrics(final_command, final_feet, final_tendon)
    np.savez_compressed(out / 'prediction.npz', command=final_command,
        tendon=final_tendon, feet=final_feet,
        motor_rate=rate.cpu().numpy(), muscle_activation=activation.cpu().numpy(),
        target_command=target, target_feet=target_feet)
    old_edges = {int(edge): k for k, edge in enumerate(cal['edge_ids'])}
    values = cal['values'].copy()
    fitted_edge_ids = []
    for row, motor_id in enumerate(motor):
        start, stop_row = base.indptr[motor_id: motor_id + 2]
        edge_range = np.arange(start, stop_row)
        for edge in edge_range:
            pre = int(base.indices[edge])
            col = np.searchsorted(sources, pre)
            if col >= len(sources) or sources[col] != pre or base.data[edge] == 0:
                continue
            if edge not in old_edges:
                raise ValueError('E86 motor source edge absent from original calibration patch')
            values[old_edges[edge]] = np.float32(base.data[edge] * fitted_theta[membership[row], col])
            fitted_edge_ids.append(edge)
    if len(fitted_edge_ids) != int(np.count_nonzero(original_w)):
        raise ValueError('Every existing signed source-MN edge must be fitted exactly once')
    updated = patch_weights(base, cal['edge_ids'], cal['source_values'], values)
    np.testing.assert_array_equal(updated.indptr, changed_matrix.indptr)
    np.testing.assert_array_equal(updated.indices, changed_matrix.indices)
    if (not np.isfinite(fitted_theta).all() or np.any(fitted_theta[present] < .001)
            or np.any(fitted_theta[present] > 10)):
        raise ValueError('Fitted positive efficacy bounds violated')
    np.testing.assert_array_equal(np.sign(updated.data[fitted_edge_ids]),
        np.sign(base.data[fitted_edge_ids]))
    unchanged = changed_matrix.copy()
    unchanged.data[fitted_edge_ids] = updated.data[fitted_edge_ids]
    np.testing.assert_array_equal(unchanged.data, updated.data)
    legacy_theta = cal.pop('theta')
    cal['values'] = values
    cal['theta'] = fitted_theta
    cal['theta_source_ids'] = sources
    cal['legacy_theta_e83'] = legacy_theta
    for key, original_value in original_cal.items():
        if key not in ('theta', 'values'):
            np.testing.assert_array_equal(cal[key], original_value)
    np.testing.assert_array_equal(cal['tau_ids'][-6:], original_cal['tau_ids'][-6:])
    np.testing.assert_array_equal(cal['tau_values'][:-6], original_cal['tau_values'][:-6])
    np.savez_compressed(out / 'calibration.npz', **cal)
    anatomy = json.loads((e86 / 'muscle-provenance.json').read_text())
    anatomy['native_all_leg_recruitment'] = {
        'source_ids': sources.tolist(), 'parameter_shape': list(fitted_theta.shape),
        'fitted_original_edge_count': len(fitted_edge_ids),
        'conditional_e86_source_only': True, 'runtime_phase_used': False}
    if own_leg:
        anatomy['own_leg_native_recruitment']={
            'phase_alignment':'offline actual E1 per leg minus fixed E85 offset',
            'source_groups_by_leg':[2,5,4,1,3,0],
            'offsets_cycles':offsets.tolist(),
            'all_runtime_phase_used':False,
            'comparison':'same recorded E86 timeline and fixed E87 prediction'}
    write_json(out / 'muscle-provenance.json', anatomy)
    base_trial = EfficacyTrial(graph, original, cal['edge_ids'], cal['source_values'], values,
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial, cal, anatomy['muscles']),
        cal['tau_ids'], cal['tau_values'])
    try:
        neutral = session.snapshot()
        exact(original['body'], neutral['body'], 'Original neutral body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(original['network'][key], neutral['network'][key], 'Original neutral ' + key)
        exact(body_before, body.snapshot(), 'Static FK leaves original body unchanged')
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out / 'candidate-neutral'))
        try:
            exact(neutral, clone.snapshot(), 'E87 neutral disk exact')
        finally:
            clone.close()
    finally:
        session.close()
    total_wall_s = time.perf_counter() - began
    shape_screen = None
    amplitudes = None
    if own_leg:
        new_q = np.asarray(final_metrics['per_leg_joint_rms_rad'])
        old_q = np.asarray(previous_metrics['per_leg_joint_rms_rad'])
        aggregate_q_ratio = float(np.sqrt(np.mean((final_command[mask]-target)**2))
            / np.sqrt(np.mean((previous_command[mask]-target)**2)))
        shape_screen={'aggregate_q_rms_ratio_E89_over_E87':aggregate_q_ratio,
            'aggregate_ratio_at_most_0_7':bool(aggregate_q_ratio<=.7),
            'each_leg_no_worse':bool(np.all(new_q<=old_q)),
            'per_leg_no_worse':(new_q<=old_q).tolist(),
            'development_screen':bool(aggregate_q_ratio<=.7 and np.all(new_q<=old_q))}
        widths=lambda x:(np.percentile(x[mask],95,axis=0)-np.percentile(x[mask],5,axis=0)).tolist()
        amplitudes={'target_joint_p95_p05_rad':(
            np.percentile(target,95,axis=0)-np.percentile(target,5,axis=0)).tolist(),
            'E89_joint_p95_p05_rad':widths(final_command),
            'E87_joint_p95_p05_rad':widths(previous_command),
            'E89_tendon_p95_p05':widths(final_tendon),
            'knee_3D_width':'unmeasured; foot-tip static geometry does not provide knee landmarks',
            'tibia_command_width_proxy_rad':{
                leg:float(np.percentile(final_command[mask,7*i+5],95)
                    -np.percentile(final_command[mask,7*i+5],5))
                for i,leg in enumerate(reference_leg_order) if leg in ('lf','rf','lh','rh')}}
    result = {'experiment': spec['experiment'],
        'validity': 'valid_conditional_fit' if total_wall_s <= remaining_wall_s else 'wall_limit_incomplete',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'hypothesis_verdict': 'conditional_improvement' if final_metrics['combined'] < baseline_metrics['combined']
            else 'not_supported', 'stop': stop, 'updates': update,
        'best_loss': best_loss, 'initial_loss': history[0]['loss'],
        'warm_start_optimization_improved': best_loss < history[0]['loss'],
        'roundoff_projection_max_abs': float(np.max(abs(fitted_theta - raw_theta))),
        'reconstruction_exact': reconstruct_from is not None,
        'baseline_output': baseline_metrics, 'fitted_output': final_metrics,
        'baseline_transfer_ratio': fidelity,
        'baseline_transfer_rms_rad': baseline_transfer_rms,
        'reference_command_rms_rad': baseline_reference_rms,
        'reference_extraction_rms_rad': reference_residual,
        'fit_parameters': int(present.sum()), 'fitted_edges': len(fitted_edge_ids),
        'prefit_wall_s': prefit_wall_s, 'fit_wall_s': optimizer_wall_s,
        'wall_s': total_wall_s,
        'earlier_execution_wall_s': previous_wall_s,
        'cumulative_execution_wall_s': previous_wall_s + total_wall_s,
        'actual_neural_s': 0, 'actual_physical_s': 0,
        'device': str(fitted.device), 'gpu': torch.cuda.get_device_name(),
        'source_lr_repaired': False, 'body_validated': False,
        'nonfit_calibration_exact': True, 'nonfit_weight_data_exact': True,
        'csr_topology_exact': True, 'source_edge_signs_preserved': True,
        'neutral_disk_exact': True}
    if own_leg:
        result.update(hypothesis_verdict='own_leg_shape_supported' if shape_screen['development_screen']
            else 'not_supported',frozen_E87_output=previous_metrics,
            shape_screen=shape_screen,amplitudes=amplitudes,
            body_validated=False,source_lr_repaired=False,
            intervention_scope='offline own-leg labels and warm target only')
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'graph', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--reconstruct-from', type=Path)
    parser.add_argument('--own-leg-target', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the existing experiment output: ' + str(args.out))
    entry_began = time.perf_counter()
    try:
        run(args.root, args.graph, args.out, reconstruct_from=args.reconstruct_from,
            own_leg=args.own_leg_target)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'hypothesis_verdict': 'not_tested', 'error': type(error).__name__,
                'message': str(error), 'execution_wall_s': time.perf_counter() - entry_began,
                'actual_neural_s': 0, 'actual_physical_s': 0})
        raise
