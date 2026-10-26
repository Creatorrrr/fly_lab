"""Bounded nongradient LM recruitment fit using actual E46 relay observations."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

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


def filtered(values, dt, tau):
    alpha = -math.expm1(-dt / tau)
    transfer = alpha / (1 - (1 - alpha) * np.exp(-2j * np.pi * np.arange(len(values) // 2 + 1) / len(values)))
    return np.fft.irfft(np.fft.rfft(values) * transfer, n=len(values))


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e46-LM-relay-full-body'
    calibration_source = root / 'e45-LM-existing-inhibitory-recruitment'
    reference = root / 'e42-source-aligned-reference'
    spec = {'experiment': 'M3-E47-measured-LM-recruitment-fit',
        'evidence': 'E45 projected Adam entered a zero-LM-rate flat ReLU region. E46 confirms LMmotor0 and healthy forward9.8699mm; all four knee amplitudes pass but middle LRmean.68984 fails. Actual new IN16B016 follows E1: late harmonic.96807,width67.4596,lag.11361. Therefore observed relay delivery is available, but the previous optimizer did not produce complementary recruitment.',
        'intervention': 'Keep E45 source graph/relay physiology/hind delivery/all other motor weights/corewave unchanged. Refit only the same two existing positive source efficacies and two positive LMroll/pitch moment gains within original bounds. LMrest remains a constant calibrated to reference means.',
        'training_data': 'Use actual E46 late[1.5,2.2) E1/IN16B016/current/muscle activation phase means. Remove the two source contributions from actual motor current to obtain remaining fixed development background. Interpolate600native bins. Actual full feedback remains required afterward; this is no held-out evidence.',
        'optimizer': 'Deterministic scipy differential_evolution in logarithms of original positive bounds: two efficacies.001..10, two moment gains.1..4; seed47,popsize8,maxiter80,tol1e-6,polishFalse. Include E45 candidate as x0. This bounded nonlinear fitting algorithm addresses an observed zero-gradient region; no waveform/phase/bounds grid.',
        'objective': 'Same E45 equal LMroll/pitch reference-scaled error and static LMfootxyz error plus1e-5efficacy regularization. Do not add success thresholds or reward finalF labels. Preserve optimizer convergence status and full finite candidate.',
        'limits': 'CPU small four-parameter fit with static shadow geometry,0actual neural/physical advances,<=90wallsec optimization/120total; final full neural remains CUDA. F1-F8pending/F6unexposed/M2UIpreserved.',
        'source_sha256': sha256(source / 'signals.npz'),
        'calibration_sha256': sha256(calibration_source / 'calibration.npz'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    cal = dict(np.load(calibration_source / 'calibration.npz'))
    base = dict(np.load(root / 'e43-rhythmic-recruitment/calibration.npz'))
    data = dict(np.load(source / 'signals.npz'))
    target = np.load(root / 'e43-rhythmic-recruitment/training-inputs.npz')['target']
    anatomy = json.loads((calibration_source / 'muscle-provenance.json').read_text())
    e1, relay, mn = graph.resolve(['flywire:banc:888:' + r for r in anatomy['added_path']])
    motor_col = int(np.flatnonzero(cal['motor_ids'] == mn)[0])
    source_cols = [int(np.flatnonzero(data['observed_ids'] == i)[0]) for i in (e1, relay)]
    matrix, _ = rate_weights(graph)
    matrix = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(matrix, cal['motor_ids'])
    original = np.array([float(matrix[mn, i]) for i in (e1, relay)])
    measured = data['rate'][:, source_cols]
    background = data['motor_current'][:, motor_col] - measured @ (original * cal['LM_parameters'][:2])
    native_input = native_period(phase_average(data['time_s'], measured))
    native_background = native_period(phase_average(data['time_s'], background))
    active = phase_average(data['time_s'], data['muscle_activation'])
    base_command = cal['rest'] + .02 * active @ cal['moment'].T
    body = new_body(StateStore.load(root / 'e32-core-adaptation/candidate/start'))
    before = body.snapshot()
    rows = json.loads((reference / 'trace.json').read_text())['samples'][300:330]
    geometry = StaticFootGeometry(body, np.array([r['qpos'] for r in rows]))
    target_foot = geometry.evaluate(target)
    length = load_model(reference / 'trace-analysis-v2.json')['lengths'][1]
    qscale = np.maximum(target[:, [8, 9]].std(axis=0), .05)
    fscale = np.maximum(target_foot[:, 1].std(axis=0), .05 * length)
    import torch
    check = np.linspace(-3, 4, 600) ** 2
    np.testing.assert_allclose(filtered(check, .00025, .02),
        periodic_filter(torch.tensor(check), .00025, .02).numpy(), atol=1e-12, rtol=1e-12)
    calls, best, best_loss = 0, None, math.inf

    def predict(parameters):
        current = native_input @ (original * parameters[:2]) + native_background
        rate = filtered(np.maximum(200 * np.tanh((current - 7.5) / 200), 0), .00025, .02)
        activation = filtered(rate[::20], .005, .01)
        command = base_command.copy()
        for k, row in enumerate((8, 9)):
            command[:, row] = target[:, row].mean() - .02 * parameters[k + 2] * (activation - activation.mean())
        feet = geometry.evaluate(command)
        loss = .5 * (np.mean(((command[:, [8, 9]] - target[:, [8, 9]]) / qscale) ** 2)
            + np.mean(((feet[:, 1] - target_foot[:, 1]) / fscale) ** 2))
        loss += 1e-5 * np.mean((parameters[:2] - 1) ** 2)
        return loss, command, feet, rate, activation

    def objective(log_parameters):
        nonlocal calls, best, best_loss
        if time.perf_counter() - began > 90:
            raise TimeoutError('Bounded LM fitting wall limit')
        parameters = np.exp(log_parameters)
        loss = predict(parameters)[0]
        if not math.isfinite(loss):
            raise ValueError('Nonfinite measured LM objective')
        calls += 1
        if loss < best_loss:
            best_loss, best = loss, parameters.copy()
        return loss

    try:
        fit = differential_evolution(objective,
            [(math.log(.001), math.log(10))] * 2 + [(math.log(.1), math.log(4))] * 2,
            x0=np.log(cal['LM_parameters']), seed=47, popsize=8, maxiter=80,
            tol=1e-6, polish=False, workers=1)
        loss, command, feet, rate, activation = predict(best)
        exact(before, body.snapshot(), 'Measured fitting preserves source body')
    finally:
        body.close()
    changed = cal['values'].copy()
    for pre, factor in zip((e1, relay), best[:2]):
        edges = np.arange(matrix.indptr[mn], matrix.indptr[mn + 1])
        edge = int(edges[matrix.indices[edges] == pre][0])
        where = np.flatnonzero(cal['edge_ids'] == edge)
        if len(where) != 1:
            raise ValueError('Unique source-calibrated motor edge required')
        changed[where[0]] = matrix.data[edge] * factor
    patch_weights(matrix, cal['edge_ids'], cal['source_values'], changed)
    for k, row in enumerate((8, 9)):
        cal['moment'][row] = base['moment'][row] * best[k + 2]
        cal['rest'][row] = target[:, row].mean() + .02 * best[k + 2] * activation.mean()
        cal['joint_delivery_gain'][row] = best[k + 2]
    cal.update(values=changed, LM_parameters=best)
    np.savez_compressed(out / 'calibration.npz', **cal)
    np.savez_compressed(out / 'prediction.npz', command=command, target=target,
        foot=feet, reference_foot=target_foot, LM_motor_rate=rate, LM_activation=activation,
        native_input=native_input, native_background=native_background)
    write_json(out / 'muscle-provenance.json', anatomy)
    times = np.arange(30) * .005
    phase = [harmonic_signal(times, feet[:, i, 0])['phase_cycle'] for i in (1, 4)]
    result = {'experiment': spec['experiment'], 'validity': 'valid_measured_conditional_fit',
        'hypothesis_verdict': 'inconclusive_until_full_body', 'adoption': 'not_applicable',
        'milestone_outcome': 'pending', 'parameters': best.tolist(), 'loss': loss,
        'optimizer_success': bool(fit.success), 'optimizer_message': str(fit.message),
        'evaluations': calls, 'generations': int(fit.nit),
        'LM_rate_width': float(np.ptp(np.percentile(rate, [5, 95]))),
        'LM_roll_pitch_rms_rad': np.sqrt(np.mean((command[:, [8, 9]] - target[:, [8, 9]]) ** 2, axis=0)).tolist(),
        'static_LM_AP_rms_mm': float(np.sqrt(np.mean((feet[:, 1, 0] - target_foot[:, 1, 0]) ** 2))),
        'static_middle_LR_harmonic_antiphase_distance': float(abs((phase[0] - phase[1]) % 1 - .5)),
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0, 'source_body_exact': True,
        'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.root, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
