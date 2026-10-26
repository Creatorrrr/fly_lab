"""Source-linked tonic excitation for complementary LM motor recruitment."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.fit_c_banc_m3_rhythmic_recruitment import periodic_filter, phase_average
from tools.probe_c_banc_m3_premotor_relay import harmonic_signal
from tools.verify_c_banc_observation import exact


def harmonic_inverse(target, mean_rate=50.):
    """Finite positive warm labels through the two declared periodic filters."""
    phase = 2 * np.pi * np.arange(30) / 30
    basis = np.column_stack((np.ones(30), np.cos(phase), np.sin(phase)))
    coefficient = np.linalg.lstsq(basis, target, rcond=None)[0]
    a_complex = -(coefficient[1] - 1j * coefficient[2]) / .02
    def transfer(count, dt, tau):
        alpha = -math.expm1(-dt / tau)
        return alpha / (1 - (1 - alpha) * np.exp(-2j * np.pi / count))
    u_complex = a_complex / (transfer(600, .00025, .02) * transfer(30, .005, .01))
    activation = mean_rate + np.real(u_complex * np.exp(2j * np.pi * np.arange(600) / 600))
    if np.any((activation <= 0) | (activation >= 200)):
        raise ValueError('Declared harmonic warm current must be positive and below source cap')
    expected = mean_rate + np.real(a_complex * np.exp(1j * phase))
    actual = periodic_filter(periodic_filter(torch.tensor(activation), .00025, .02)[::20], .005, .01)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-11, rtol=1e-12)
    return activation, coefficient


def run(root, graph_path, out, *, fixed_rest=False):
    out.mkdir(parents=True, exist_ok=False)
    prior = root / 'e45-LM-existing-inhibitory-recruitment'
    actual_source = root / 'e46-LM-relay-full-body'
    reference = root / 'e42-source-aligned-reference'
    spec = {'experiment': 'M3-E48-source-tonic-LM-recruitment',
        'hypothesis': 'Existing descending-linked tonic excitation supplies the mean current needed for a correlated excitatory/inhibitory pair to recruit the LMpleural MN at a complementary phase without silencing it.',
        'path': ['720575941500851362', '720575941612203996', '720575941553503460'],
        'intervention': 'Original DNg100->IN01A015 weight+2.61 preserved; other IN01A015 incoming edgesx.001, its adaptationgain0, other physiologyunchanged. Add its existing +1.02 output to the two existing E1/IN16B016->LM MN efficacy parameters[.001,10]. Retain source topology/signs and all E45 other weights/hind delivery/core phases. Two LMroll/pitch positive moment gains remain[.1,4], rest calibrated to reference means.',
        'tonic_model': 'Use observed E46 final descending rate with original DNweight/half-tanh as a prospective steady nonadapting IN01A015 rate; remaining weak incoming field0 is conditional only. Subtract original weak IN01A015->MN contribution using its saved final rate. The omitted variation is bounded by.001*1.02*200=.204current units; actual fullnetwork validation still required.',
        'warm_start': 'Fit the reference LMroll first harmonic. Set a training-only50rate mean and invert the exact native20ms pluscontrol10ms periodic linear filters. Require all desired activation strictly between0and200; fit three same-sign source current coefficients by bounded leastsquares. This warm label is never supplied at runtime.',
        'fit': 'Same E47 normalized LMroll/pitch and static LMfootxyz objective, plus1e-5efficacy regularization. One CUDA Adam optimization in log positive parameters,lr.03/max1000updates; preserve best finite loss. Four old bounds unchanged, only one new original excitatory edge is learned. No phase/waveform/bounds grid.',
        'decision': 'A valid candidate requires actual unforced MN/newtonic relay/fullBANC/body verification before any gait claim. If tonic transmission is absent, diagnose that path; if it exists but LM still fails, preserve the recruitment failure. External24coreteacher must eventually be removed; F1-F8unchanged.',
        'limits': 'OneGPU,0actualneural/physical advance in fit;<=180wallsec/fit150stop. Shadowgeometry leaves sourcebodyexact; F6unexposed/M2UIpreserved.',
        'tool_sha256': sha256(__file__), 'prior_sha256': sha256(prior / 'calibration.npz'),
        'actual_sha256': sha256(actual_source / 'signals.npz'),
        'tonic_provenance_sha256': sha256(root / 'e47-measured-LM-recruitment-fit/tonic-source-provenance.json')}
    if fixed_rest:
        spec.update(experiment='M3-E49-fixed-rest-tonic-recruitment',
            discovered_limit='E48 conditional fit uses LMmeanactivation197.788 and affine rest15.994rad, outside the existing10rad actuator-call contract from zero activation. It is not an executable neutral-start gait candidate.',
            rest_constraint='Restore the original E43 single-sign reference-extreme LMroll/pitch rest, already physically used inE44. Keep it fixed while fitting. All existing final gates/bounds/source paths remain unchanged. No neural preloading or startupclock.',
            warm_start='Use median observed inhibitory rate as a rectification operating point. Scale its positive below-median waveform by the required mean LMroll activation at the fixed rest, using the declared two periodic filters. Derive tonic efficacy from that threshold and observed background. This is one deterministic source/target-based initialization, not a gain grid.',
            fit='Same normalized angle/foot objective and log positive5parameter CUDAAdam lr.03/max1000, but fixed LMrest is included in predictions; no learned affine mean cancellation. Original harmonic inverse is retained only as a finite numerical reference.')
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    cal = dict(np.load(prior / 'calibration.npz'))
    original_interface = dict(np.load(root / 'e43-rhythmic-recruitment/calibration.npz'))
    observed = dict(np.load(actual_source / 'signals.npz'))
    signal = dict(np.load(root / 'e47-measured-LM-recruitment-fit/prediction.npz'))
    target = signal['target']
    anatomy = json.loads((prior / 'muscle-provenance.json').read_text())
    dn, tonic, mn = graph.resolve(['flywire:banc:888:' + r for r in spec['path']])
    e1, relay, expected_mn = graph.resolve(['flywire:banc:888:' + r for r in anatomy['added_path']])
    if expected_mn != mn or graph.nodes[tonic]['cell_type'] != 'IN01A015' or graph.nodes[tonic]['super_class'] != 'ventral_nerve_cord_intrinsic':
        raise ValueError('Declared intrinsic excitatory source path changed')
    matrix, _ = rate_weights(graph)
    matrix = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(matrix, cal['motor_ids'])
    if not np.isclose(matrix[tonic, dn], 2.61) or not np.isclose(matrix[mn, tonic], 1.02):
        raise ValueError('Original source tonic path weights differ')
    final = StateStore.load(actual_source / 'final-teaching-state')['whole_neural']
    dn_rate = float(final['rate'][dn])
    if dn_rate < 190 or float(final['drive'][dn]) != 5000:
        raise ValueError('Observed strong descending forward arousal required')
    tonic_rate = 200 * math.tanh((float(matrix[tonic, dn]) * dn_rate - 7.5) / 200)
    sources = np.column_stack((signal['native_input'], np.full(600, tonic_rate)))
    original = np.array([float(matrix[mn, i]) for i in (e1, relay, tonic)])
    background = signal['native_background'] - .001 * original[2] * float(final['rate'][tonic])
    warm_u, harmonic = harmonic_inverse(target[:, 8])
    warm_current = 7.5 + 200 * np.arctanh(warm_u / 200)
    x = sources * original
    fit = lsq_linear(np.vstack((x / 200, .001 * np.eye(3))),
        np.r_[(warm_current - background) / 200, .001 * np.ones(3)],
        bounds=(.001, 10), method='bvls', tol=1e-10, max_iter=300)
    if not fit.success:
        raise ValueError('Positive harmonic-current warm fit failed')
    initial_parameters = np.r_[fit.x, 1., .1]
    if fixed_rest:
        center = float(np.median(sources[:, 1]))
        unit_u = np.maximum(center - sources[:, 1], 0) * abs(original[1])
        unit_active = periodic_filter(periodic_filter(torch.tensor(unit_u), .00025, .02)[::20], .005, .01).numpy()
        desired_mean = float((original_interface['rest'][8] - target[:, 8].mean()) / .02)
        inhibitory = desired_mean / unit_active.mean()
        excitatory = .001
        tonic_factor = (7.5 - background.mean() + abs(original[1]) * inhibitory * center
            - original[0] * excitatory * sources[:, 0].mean()) / (original[2] * tonic_rate)
        initial_parameters[:3] = np.clip([excitatory, inhibitory, tonic_factor], .001, 10)
    np.savez_compressed(out / 'training-inputs.npz', sources=sources, background=background,
        original_weights=original, warm_u=warm_u, warm_current=warm_current,
        warm_efficacy=fit.x, LM_reference_harmonic=harmonic,
        optimizer_initial_parameters=initial_parameters, fixed_rest=fixed_rest)
    active = phase_average(observed['time_s'], observed['muscle_activation'])
    base_command = cal['rest'] + .02 * active @ cal['moment'].T
    body = new_body(StateStore.load(root / 'e32-core-adaptation/candidate/start'))
    before = body.snapshot()
    rows = json.loads((reference / 'trace.json').read_text())['samples'][300:330]
    geometry = StaticFootGeometry(body, np.array([r['qpos'] for r in rows]))
    feet_reference = geometry.evaluate(target)
    length = load_model(reference / 'trace-analysis-v2.json')['lengths'][1]
    qscale = np.maximum(target[:, [8, 9]].std(axis=0), .05)
    fscale = np.maximum(feet_reference[:, 1].std(axis=0), .05 * length)
    if not torch.cuda.is_available():
        body.close(); raise RuntimeError('Actual CUDA required')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')
    source_t, weight_t, background_t = map(tensor, (sources, original, background))
    base_t, target_t, feet_t = map(tensor, (base_command, target, feet_reference))
    qscale_t, fscale_t = map(tensor, (qscale, fscale))
    parameter = torch.nn.Parameter(tensor(np.log(initial_parameters)))
    optimizer = torch.optim.Adam([parameter], lr=.03)
    def predict():
        p = parameter.exp()
        current = source_t @ (weight_t * p[:3]) + background_t
        rate = periodic_filter(torch.relu(200 * torch.tanh((current - 7.5) / 200)), .00025, .02)
        activation = periodic_filter(rate[::20], .005, .01)
        command = base_t.clone()
        for k, row in enumerate((8, 9)):
            command[:, row] = (float(original_interface['rest'][row]) - .02 * p[k + 3] * activation
                if fixed_rest else target_t[:, row].mean() - .02 * p[k + 3] * (activation - activation.mean()))
        foot = geometry.tensor(command)
        loss = .5 * (((command[:, [8, 9]] - target_t[:, [8, 9]]) / qscale_t).square().mean()
            + ((foot[:, 1] - feet_t[:, 1]) / fscale_t).square().mean())
        loss += 1e-5 * (p[:3] - 1).square().mean()
        return loss, command, foot, rate, activation
    best, best_loss, history = None, math.inf, []
    try:
        for step in range(1001):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _, _ = predict()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Nonfinite tonic recruitment loss')
            if value < best_loss:
                best_loss = value; best = parameter.detach().clone()
            if step % 100 == 0:
                history.append({'step': step, 'loss': value,
                    'parameters': parameter.exp().detach().cpu().tolist(), 'wall_s': time.perf_counter() - began})
            if step % 500 == 0:
                print(json.dumps(history[-1]), flush=True)
            if step == 1000 or time.perf_counter() - began > 150:
                break
            loss.backward(); optimizer.step()
            with torch.no_grad():
                parameter[:3].clamp_(math.log(.001), math.log(10))
                parameter[3:].clamp_(math.log(.1), math.log(4))
        with torch.no_grad():
            parameter.copy_(best)
            loss, command, foot, rate, activation = predict()
        parameters = parameter.exp().detach().cpu().numpy()
        exact(before, body.snapshot(), 'Tonic fit source physical state')
    finally:
        body.close()
    changed = cal['values'].copy()
    for pre, factor in zip((e1, relay, tonic), parameters[:3]):
        edges = np.arange(matrix.indptr[mn], matrix.indptr[mn + 1])
        edge = int(edges[matrix.indices[edges] == pre][0])
        where = np.flatnonzero(cal['edge_ids'] == edge)
        if len(where) != 1:
            raise ValueError('Existing calibrated motor input required')
        changed[where[0]] = matrix.data[edge] * factor
    new_edges = np.arange(matrix.indptr[tonic], matrix.indptr[tonic + 1])
    new_edges = new_edges[matrix.data[new_edges] != 0]
    new_values = matrix.data[new_edges] * np.float32(.001)
    selected_dn = matrix.indices[new_edges] == dn
    new_values[selected_dn] = matrix.data[new_edges[selected_dn]]
    edge_ids = np.r_[cal['edge_ids'], new_edges]
    source_values = np.r_[cal['source_values'], matrix.data[new_edges]]
    values = np.r_[changed, new_values]
    patch_weights(matrix, edge_ids, source_values, values)
    for k, row in enumerate((8, 9)):
        cal['moment'][row] = original_interface['moment'][row] * parameters[k + 3]
        cal['rest'][row] = (original_interface['rest'][row] if fixed_rest
            else target[:, row].mean() + .02 * parameters[k + 3] * float(activation.mean()))
        cal['joint_delivery_gain'][row] = parameters[k + 3]
    cal.update(edge_ids=edge_ids, source_values=source_values, values=values,
        zero_adaptation_ids=np.union1d(cal['zero_adaptation_ids'], [tonic]),
        input_ids=np.r_[cal['input_ids'], tonic], validation_observed_ids=np.array([dn]),
        LM_tonic_parameters=parameters)
    np.savez_compressed(out / 'calibration.npz', **cal)
    command_np, foot_np = command.detach().cpu().numpy(), foot.detach().cpu().numpy()
    np.savez_compressed(out / 'prediction.npz', command=command_np, target=target,
        foot=foot_np, reference_foot=feet_reference, LM_motor_rate=rate.detach().cpu().numpy(),
        LM_activation=activation.detach().cpu().numpy())
    write_json(out / 'muscle-provenance.json', {**anatomy,
        'input_nodes': anatomy['input_nodes'] + [graph.nodes[tonic]],
        'tonic_path': spec['path'], 'tonic_source_weights': [float(matrix[tonic, dn]), float(matrix[mn, tonic])],
        'tonic_incoming_edges': len(new_edges), 'actual_final_DN_rate': dn_rate,
        'actual_final_original_tonic_rate': float(final['rate'][tonic]), 'predicted_tonic_rate': tonic_rate})
    times = np.arange(30) * .005
    phases = [harmonic_signal(times, foot_np[:, i, 0])['phase_cycle'] for i in (1, 4)]
    result = {'experiment': spec['experiment'], 'validity': 'valid_conditional_source_tonic_fit',
        'hypothesis_verdict': 'inconclusive_until_actual_full_network', 'adoption': 'not_applicable',
        'milestone_outcome': 'pending', 'parameters': parameters.tolist(), 'loss': float(loss),
        'iterations': step, 'warm_current_rms': float(np.sqrt(np.mean((x @ fit.x + background - warm_current) ** 2))),
        'fixed_rest': fixed_rest, 'zero_activation_max_abs_target_rad': float(abs(cal['rest']).max()),
        'LM_rate_p95_minus_p05': float(np.ptp(np.percentile(rate.detach().cpu().numpy(), [5, 95]))),
        'LM_roll_pitch_rms_rad': np.sqrt(np.mean((command_np[:, [8, 9]] - target[:, [8, 9]]) ** 2, axis=0)).tolist(),
        'static_LM_AP_rms_mm': float(np.sqrt(np.mean((foot_np[:, 1, 0] - feet_reference[:, 1, 0]) ** 2))),
        'static_middle_LR_harmonic_antiphase_distance': float(abs((phases[0] - phases[1]) % 1 - .5)),
        'device': str(parameter.device), 'source_body_exact': True,
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0,
        'wall_s': time.perf_counter() - began, 'history': history}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fixed-rest', action='store_true')
    args = p.parse_args()
    try:
        run(args.root, args.graph, args.out, fixed_rest=args.fixed_rest)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
