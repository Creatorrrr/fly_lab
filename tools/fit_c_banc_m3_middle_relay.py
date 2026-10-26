"""Fit an existing complementary LM pathway without moving other core phases."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.fit_c_banc_m3_rhythmic_recruitment import (
    native_period,
    periodic_filter,
    phase_average,
)
from tools.probe_c_banc_m3_premotor_relay import harmonic_signal
from tools.verify_c_banc_observation import exact


def run(source, relay_source, reference, initial, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    root = source.parent
    fit_source = root / 'e43-rhythmic-recruitment'
    spec = {'experiment': 'M3-E45-LM-existing-inhibitory-recruitment',
        'evidence': 'E44 LMroll command follows its E43 prediction but has wrong phase; servo RMS.000816rad. LMroll/pitch share one source pleural MN with only positive LM E1 among30selectedsources. Static replacement of LMroll alone reduces measured LMfoot AP RMS.38094->.04999mm. Other LMjoint phases already track reference.',
        'hypothesis': 'A delayed inhibitory source pathway can recruit the single LMpleural MN at a complementary phase while preserving all existing core phases and other LMmotor paths.',
        'path': ['720575941527002649', '720575941590291084', '720575941553503460'],
        'intervention': 'Existing LM E1->IN16B016 input +2.46 set to frozen E40 effective E1 weight1.1707944; its other incoming edgesx.001 and adaptationgain0. Existing relay->LMpleural edge-4.08 enters fit. All E43 motor/relay calibration otherwise retained. No new anatomical edges or core phase change.',
        'fit': 'Four parameters: existing LM E1->pleural and IN16B016->pleural positive efficacies[.001,10], plus positive LMroll/pitch moment gains[.1,4]. Fixed reference mean rest on these two rows. One projected CUDA Adam lr.05/500updates. Equal weighting of reference-scaled LMroll/pitch error and static LMfootxyz error plus1e-5 efficacy regularization.',
        'relay_approximation': 'Prospective IN16B016 periodic rate uses the frozen E1 effective input, source half-tanh and20ms nonadapting filter; its other weak input field is0 for this conditional fit. Existing weak relay contribution is removed from E41-recorded motor background. Actual newrelay transmission and actualbody must be checked afterward; this approximation is not a measurement of IN16B016.',
        'hind_delivery': 'Only LH/RH knee moment gains are fixed once from E42actual p95-p05 / E44actual p95-p05, with constant rest centered on E42training command means. Positive signs preserved; no other rows change. These are engineering moment proxies, not measured muscle moment arms.',
        'geometry': 'Static shadow MuJoCo kinematics/Jacobians in30already-exposed E42 qpos contexts. No physical integration or runtime trajectory. Sourcebody remains exact. Reference-normalized shape loss is training only and cannot change finalF criteria.',
        'decision': 'A finite source-valid fitted candidate receives actual full-BANC/body validation with unforced MN and unchanged E37coreteacher. Preserve failures; no gain/phase/bound grid. New relay is an engineering operating-point hypothesis, not biological reconstruction.',
        'limits': 'OneGPU,0actual neural/physical advance,180wallsec withfitstop150. F1-F8pending/F6unexposed/M2UIpreserved.',
        'tool_sha256': sha256(__file__), 'source_sha256': sha256(source / 'signals.npz'),
        'original_fit_sha256': sha256(fit_source / 'calibration.npz'),
        'reference_sha256': sha256(reference / 'trace-analysis-v2.json')}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    calibration = dict(np.load(fit_source / 'calibration.npz'))
    inputs = dict(np.load(fit_source / 'training-inputs.npz'))
    actual = dict(np.load(source / 'signals.npz'))
    relay_data = dict(np.load(relay_source / 'observations.npz'))
    anatomy = json.loads((fit_source / 'muscle-provenance.json').read_text())
    e1, relay, mn = graph.resolve(['flywire:banc:888:' + r for r in spec['path']])
    if graph.nodes[relay]['cell_type'] != 'IN16B016':
        raise ValueError('Source IN16B016 identity changed')
    matrix, _ = rate_weights(graph)
    matrix = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(matrix, calibration['motor_ids'])
    if not np.isclose(matrix[relay, e1], 2.46) or not np.isclose(matrix[mn, relay], -4.08):
        raise ValueError('Declared original source path differs')
    muscle = next(i for i, g in enumerate(anatomy['muscles'])
        if g['leg'] == 'lm' and g['muscle'] == 'pleural_remotor_and_abductor_muscle')
    motor_col = int(np.flatnonzero(calibration['motor_ids'] == mn)[0])
    if anatomy['muscles'][muscle]['columns'] != [motor_col]:
        raise ValueError('One source LM pleural motor cell required')
    for row in (8, 9):
        np.testing.assert_array_equal(np.flatnonzero(calibration['moment'][row]), [muscle])
        if calibration['moment'][row, muscle] != -1:
            raise ValueError('Declared LM moment sign changed')
    e1_col = int(np.flatnonzero(calibration['input_ids'] == e1)[0])
    native_e1 = inputs['source_native'][:, e1_col]
    relay_col = int(np.searchsorted(relay_data['observed_ids'], relay))
    if relay_data['observed_ids'][relay_col] != relay:
        raise ValueError('Original relay contribution must be observed')
    old_relay = native_period(phase_average(relay_data['time_s'], relay_data['rate'][:, relay_col]))
    background = inputs['background_native'][:, motor_col] - .001 * float(matrix[mn, relay]) * old_relay
    original_relay_patch = dict(np.load(relay_source / 'calibration.npz'))
    lm_relay = original_relay_patch['zero_adaptation_ids'][np.flatnonzero(original_relay_patch['E1_ids'] == e1)[0]]
    relay_edge = np.arange(matrix.indptr[lm_relay], matrix.indptr[lm_relay + 1])
    e1_edge = int(relay_edge[matrix.indices[relay_edge] == e1][0])
    weight = original_relay_patch['values'][original_relay_patch['edge_ids'] == e1_edge]
    if weight.shape != (1,) or weight[0] <= 0:
        raise ValueError('Frozen E40/E41 effective E1 weight required')
    target = inputs['target']
    base_active = phase_average(actual['time_s'], actual['muscle_activation'])
    base_command = calibration['rest'] + .02 * base_active @ calibration['moment'].T
    source_result = json.loads((source / 'result.json').read_text())
    ref_result = json.loads((reference / 'result.json').read_text())
    source_width = source_result['four_knee_width_deg']
    ref_width = ref_result['knee_width_deg']
    gains = np.ones(42)
    for leg, row in (('lh', 19), ('rh', 40)):
        gains[row] = ref_width[leg] / source_width[leg]
        if not 1 < gains[row] <= 4:
            raise ValueError('Source-based positive hind delivery within original4x bound required')
    body = new_body(StateStore.load(initial / 'start'))
    before_body = body.snapshot()
    rows = json.loads((reference / 'trace.json').read_text())['samples'][300:330]
    geometry = StaticFootGeometry(body, np.array([r['qpos'] for r in rows]))
    reference_foot = geometry.evaluate(target)
    lengths = load_model(reference / 'trace-analysis-v2.json')['lengths']
    qscale = np.maximum(target[:, [8, 9]].std(axis=0), .05)
    fscale = np.maximum(reference_foot[:, 1].std(axis=0), .05 * lengths[1])
    if not torch.cuda.is_available():
        body.close(); raise RuntimeError('Actual CUDA required')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')
    e1_t, background_t = map(tensor, (native_e1, background))
    relay_t = periodic_filter(torch.relu(200 * torch.tanh((float(weight[0]) * e1_t - 7.5) / 200)), .00025, .02)
    base_t, target_t, reference_t = map(tensor, (base_command, target, reference_foot))
    qscale_t, fscale_t = map(tensor, (qscale, fscale))
    parameter = torch.nn.Parameter(tensor([calibration['theta'][muscle, e1_col], .001, 1., 1.]))
    optimizer = torch.optim.Adam([parameter], lr=.05)

    def predict():
        current = float(matrix[mn, e1]) * parameter[0] * e1_t + float(matrix[mn, relay]) * parameter[1] * relay_t + background_t
        rate = periodic_filter(torch.relu(200 * torch.tanh((current - 7.5) / 200)), .00025, .02)
        activation = periodic_filter(rate[::20], .005, .01)
        command = base_t.clone()
        for j, row in enumerate((8, 9)):
            command[:, row] = target_t[:, row].mean() - .02 * parameter[j + 2] * (activation - activation.mean())
        for row in (19, 40):
            command[:, row] = target_t[:, row].mean() + gains[row] * (base_t[:, row] - base_t[:, row].mean())
        foot = geometry.tensor(command)
        qloss = ((command[:, [8, 9]] - target_t[:, [8, 9]]) / qscale_t).square().mean()
        floss = ((foot[:, 1] - reference_t[:, 1]) / fscale_t).square().mean()
        loss = .5 * (qloss + floss) + 1e-5 * (parameter[:2] - 1).square().mean()
        return loss, command, foot, rate, activation

    best, best_value, history = None, math.inf, []
    try:
        for step in range(501):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _, _ = predict()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Nonfinite LM recruitment objective')
            if value < best_value:
                best_value = value; best = parameter.detach().clone()
            if step % 50 == 0:
                history.append({'step': step, 'loss': value, 'parameters': parameter.detach().cpu().tolist(),
                    'wall_s': time.perf_counter() - began})
            if step % 250 == 0:
                print(json.dumps(history[-1]), flush=True)
            if step == 500 or time.perf_counter() - began > 150:
                break
            loss.backward(); optimizer.step()
            with torch.no_grad():
                parameter[:2].clamp_(.001, 10); parameter[2:].clamp_(.1, 4)
        with torch.no_grad():
            parameter.copy_(best)
            loss, command, feet, rates, activation = predict()
        result_parameter = parameter.detach().cpu().numpy()
        command_np, foot_np = command.detach().cpu().numpy(), feet.detach().cpu().numpy()
        exact(before_body, body.snapshot(), 'Offline geometry preserved source body')
    finally:
        body.close()
    # Reuse all old efficacy values except the two identified source motor
    # inputs, then append the disjoint existing new relay incoming row.
    changed = calibration['values'].copy()
    for pre, factor in ((e1, result_parameter[0]), (relay, result_parameter[1])):
        ids = np.arange(matrix.indptr[mn], matrix.indptr[mn + 1])
        edge = int(ids[matrix.indices[ids] == pre][0])
        where = np.flatnonzero(calibration['edge_ids'] == edge)
        if len(where) != 1:
            raise ValueError('Selected motor input must already have source calibration provenance')
        changed[where[0]] = np.float32(matrix.data[edge] * factor)
    new_edges = np.arange(matrix.indptr[relay], matrix.indptr[relay + 1])
    new_edges = new_edges[matrix.data[new_edges] != 0]
    new_values = matrix.data[new_edges] * np.float32(.001)
    new_values[matrix.indices[new_edges] == e1] = weight[0]
    ids = np.r_[calibration['edge_ids'], new_edges]
    original = np.r_[calibration['source_values'], matrix.data[new_edges]]
    values = np.r_[changed, new_values]
    patch_weights(matrix, ids, original, values)
    gains[[8, 9]] = result_parameter[2:]
    moment, rest = calibration['moment'] * gains[:, None], calibration['rest'].copy()
    for row in (8, 9):
        rest[row] = target[:, row].mean() + .02 * gains[row] * float(activation.mean())
    for row in (19, 40):
        rest[row] = target[:, row].mean() - .02 * base_active.mean(axis=0) @ moment[row]
    calibration.update(edge_ids=ids, source_values=original, values=values,
        zero_adaptation_ids=np.union1d(calibration['zero_adaptation_ids'], [relay]),
        input_ids=np.r_[calibration['input_ids'], relay], moment=moment, rest=rest,
        LM_parameters=result_parameter, joint_delivery_gain=gains)
    np.savez_compressed(out / 'calibration.npz', **calibration)
    np.savez_compressed(out / 'prediction.npz', command=command_np, target=target,
        foot=foot_np, reference_foot=reference_foot, prospective_relay=relay_t.detach().cpu().numpy(),
        LM_motor_rate=rates.detach().cpu().numpy(), LM_activation=activation.detach().cpu().numpy())
    write_json(out / 'muscle-provenance.json', {**anatomy,
        'input_nodes': anatomy['input_nodes'] + [graph.nodes[relay]], 'added_path': spec['path'],
        'source_path_weights': [float(matrix[relay, e1]), float(matrix[mn, relay])],
        'effective_E1_input': float(weight[0]), 'new_relay_incoming_edges': len(new_edges)})
    times = np.arange(30) * .005
    phase_l = harmonic_signal(times, foot_np[:, 1, 0])['phase_cycle']
    phase_r = harmonic_signal(times, foot_np[:, 4, 0])['phase_cycle']
    phase_distance = abs((phase_l - phase_r) % 1 - .5)
    report = {'experiment': spec['experiment'], 'validity': 'valid_conditional_new_relay_fit',
        'hypothesis_verdict': 'inconclusive_until_actual_relay_and_body', 'adoption': 'not_applicable',
        'milestone_outcome': 'pending', 'parameters': result_parameter.tolist(),
        'joint_delivery_gains': {str(i): float(gains[i]) for i in (8, 9, 19, 40)},
        'loss': float(loss), 'iterations': step, 'static_middle_LR_harmonic_antiphase_distance': phase_distance,
        'static_LM_AP_rms_mm': float(np.sqrt(np.mean((foot_np[:, 1, 0] - reference_foot[:, 1, 0]) ** 2))),
        'LM_roll_pitch_rms_rad': np.sqrt(np.mean((command_np[:, [8, 9]] - target[:, [8, 9]]) ** 2, axis=0)).tolist(),
        'device': str(parameter.device), 'source_body_exact': True,
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0,
        'wall_s': time.perf_counter() - began, 'history': history}
    write_json(out / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'relay-source', 'reference', 'initial', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.source, args.relay_source, args.reference, args.initial, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
