"""Conditional recruitment fitting on existing core/relay-to-motor synapses.

The periodic inputs below are training labels, never a runtime controller.
Actual whole-network/body validation is required after this finite fit.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_synaptic_efficacy import patch_weights


def periodic_filter(values, dt, tau):
    """Exact periodic solution of a held-input first-order discrete recurrence."""
    import torch

    count = values.shape[0]
    alpha = -math.expm1(-dt / tau)
    omega = 2 * math.pi * torch.arange(count // 2 + 1,
        device=values.device, dtype=values.dtype) / count
    transfer = alpha / (1 - (1 - alpha) * torch.exp(-1j * omega))
    shape = (len(transfer),) + (1,) * (values.ndim - 1)
    return torch.fft.irfft(torch.fft.rfft(values, dim=0) * transfer.reshape(shape),
        n=count, dim=0)


def phase_average(times, values):
    """Average only already exposed late development observations into 30 bins."""
    active = (times >= 1.5) & (times < 2.2)
    bins = np.rint(times[active] / .005).astype(int) % 30
    rows = []
    for phase in range(30):
        selected = values[active][bins == phase]
        if not len(selected):
            raise ValueError('Every exposed periodic training bin needs observations')
        rows.append(selected.mean(axis=0))
    return np.asarray(rows)


def native_period(control):
    values = np.concatenate((control, control[:1]), axis=0)
    return CubicSpline(np.arange(31) / 30, values, bc_type='periodic')(np.arange(600) / 600)


def run(source, reference, identification, teacher, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E43-rhythmic-muscle-recruitment',
        'question': 'Can existing24core+6calibratedrelay projections recruit anatomical muscle means that produce useful E42 commands?',
        'intervention': 'Preserve E41 sixrelay input patch. For existing30source->391motor edges, fit positive[.001,10] multipliers shared by postsynaptic anatomical muscle and source neuron. Multiply all remaining incoming motor edges by.001 except20navigation ports, whose original M2x64 weights remain unchanged. Set391motor adaptationgain0, retaincore24/relay6gain0 and remaining source physiology. This strong background attenuation and nonadaptation are engineering hypotheses, not measured biology.',
        'decoder': 'Frozen source96muscle mean-pool anatomical moment signs, .02rad/rate unit,10ms muscle activation, constant E42-calibrated rest; no tanh joint saturation. Single-sign rows rest at appropriate reference extreme; unsupported rows at reference mean. Six leg tendons use.2*tanh(filtered sourceLTMmean/10), no static leg preload. Abdomen unchanged. No clock/phase in runtime decoder.',
        'conditional_fit': 'Source24 exact E37 native wave from1.5s plus six E41 late phase-averaged relay rates. Remaining motor currents use E41 late phase-average times.001. Half-tanh activation,20ms periodic held-input native filter,30control samples/cycle and10ms periodic muscle activation. This is an approximation; actual full CUDA with unforced MN and physical senses is required.',
        'objective': '42joint MSE +1e-5mean(theta-1)^2 on present parameters +.01mean(relu(20-pooledLTM)/20)^2. Deterministic nonnegative muscle inverse/current warm start, then one projected CUDA Adam fit lr.05/max1000updates, no bounds/waveform grid. Save best finite observed objective. No surrogate RMS threshold substitutes for body trial.',
        'decision': 'A finite source-valid candidate receives one preregistered full-neural/body diagnostic with the existing24coreteacher even if command reconstruction is imperfect. Actual gait failure determines next change. This stage cannot satisfy F1-F8; F6unexposed/UI/M2preserved.',
        'limits': 'OneGPU,0actual neural/physics seconds,180wallseconds total; optimization stops by150wallseconds. No live service changes.',
        'source_sha256': sha256(source / 'observations.npz'),
        'reference_sha256': sha256(reference / 'commands.npz'),
        'identification_sha256': sha256(identification / 'signals.npz'),
        'teacher_sha256': sha256(teacher / 'teaching-table.npz'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    data = dict(np.load(source / 'observations.npz'))
    relay_patch = dict(np.load(source / 'calibration.npz'))
    wave = dict(np.load(teacher / 'teaching-table.npz'))
    ref = dict(np.load(reference / 'commands.npz'))
    anatomy = dict(np.load(identification / 'signals.npz'))
    collection = json.loads((identification / 'collection-result.json').read_text())
    groups = collection['muscles']
    motor = data['motor_ids']
    if graph.hash != collection['graph_hash'] or len(motor) != 391 or len(groups) != 96:
        raise ValueError('Original391motor/96muscle anatomy and source graph required')
    if [graph.nodes[i]['id'] for i in motor] != collection['motor_roster']:
        raise ValueError('Source motor order differs from anatomical moment columns')
    inputs = np.r_[wave['ids'], data['relay_ids']].astype(np.int64)
    if len(np.unique(inputs)) != 30 or len(np.intersect1d(inputs, motor)):
        raise ValueError('Thirty distinct nonmotor core/relay sources required')
    weights, _ = rate_weights(graph)
    encoder = BancTargetEncoder(graph, forward_drive=5000)
    weights = encoder.calibrate_weights(weights, motor)
    original = weights[motor][:, inputs].toarray().astype(np.float64)
    moment = np.column_stack([anatomy['moment'][:, g['columns'][0]] for g in groups])
    pool = np.zeros((96, 391), dtype=float)
    membership = np.full(391, -1, dtype=int)
    for k, group in enumerate(groups):
        columns = group['columns']
        if np.any(membership[columns] != -1):
            raise ValueError('Each motor neuron must have one original anatomical muscle')
        pool[k, columns] = 1 / len(columns)
        membership[columns] = k
    if np.any(membership < 0):
        raise ValueError('Every original motor neuron must retain anatomical membership')
    target = phase_average(ref['time_s'], ref['command'])
    rest = anatomy['neutral'].copy()
    for j, row in enumerate(moment):
        nonzero = row[row != 0]
        if not len(nonzero):
            rest[j] = target[:, j].mean()
        elif np.all(nonzero < 0):
            rest[j] = target[:, j].max()
        elif np.all(nonzero > 0):
            rest[j] = target[:, j].min()
    source_columns = np.searchsorted(data['observed_ids'], inputs)
    np.testing.assert_array_equal(data['observed_ids'][source_columns], inputs)
    measured_sources = data['rate'][:, source_columns]
    background = data['motor_current'].astype(float) - measured_sources @ original.T
    source_native = np.column_stack((wave['rate'][12000 + 2 * np.arange(600)],
        np.clip(native_period(phase_average(data['time_s'], measured_sources[:, 24:])), 0, 200)))
    background_native = .001 * native_period(phase_average(data['time_s'], background))
    # Navigation is already silent in this exposed forward diagnostic. Its
    # preserved x64 edges are excluded from the motor-background attenuation.
    nav_ids = np.asarray(encoder.ports, dtype=int)
    nav_columns = np.searchsorted(data['observed_ids'], nav_ids)
    np.testing.assert_array_equal(data['observed_ids'][nav_columns], nav_ids)
    if np.any(data['rate'][:, nav_columns] != 0):
        raise ValueError('Conditional diagnostic requires verified silent navigation ports')
    present = (pool @ (original != 0).astype(float)) > 0
    ltm = np.array([i for i, g in enumerate(groups) if g['muscle'] == 'long_tendon_muscle'])
    if len(ltm) != 6:
        raise ValueError('Six source long-tendon muscle groups required')
    warm = []
    inverse = np.vstack((.02 * moment, .0001 * np.eye(96)))
    for row in target:
        fit = lsq_linear(inverse, np.r_[row - rest, np.zeros(96)],
            bounds=(0, 199), method='bvls', tol=1e-8, max_iter=300)
        if not fit.success:
            raise ValueError('Finite anatomical warm-start solver failed')
        warm.append(fit.x)
    warm = np.asarray(warm)
    warm[:, ltm] = np.maximum(warm[:, ltm], 20)
    initial_theta = np.ones((96, 30))
    source_control = source_native[::20]
    for g in range(96):
        selected = np.flatnonzero(present[g])
        if not len(selected):
            continue
        mean_weights = pool[g] @ original
        x = source_control[:, selected] * mean_weights[selected]
        y = 7.5 + 200 * np.arctanh(np.clip(warm[:, g] / 200, 0, .995))
        y -= background_native[::20] @ pool[g]
        fitted = lsq_linear(np.vstack((x / 200, .01 * np.eye(len(selected)))),
            np.r_[y / 200, .01 * np.ones(len(selected))], bounds=(.001, 10),
            method='bvls', tol=1e-8, max_iter=300)
        if not fitted.success:
            raise ValueError('Finite positive-efficacy warm-start solver failed')
        initial_theta[g, selected] = fitted.x
    np.savez_compressed(out / 'training-inputs.npz', input_ids=inputs, motor_ids=motor,
        source_native=source_native, background_native=background_native, target=target,
        moment=moment, rest=rest, pool=pool, membership=membership, present=present,
        original=original, initial_theta=initial_theta, warm_muscles=warm, ltm=ltm)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required for this calibration')
    torch.set_num_threads(1)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')
    source_t, background_t, original_t = map(tensor, (source_native, background_native, original))
    moment_t, rest_t, pool_t, target_t = map(tensor, (moment, rest, pool, target))
    membership_t = torch.as_tensor(membership, device='cuda')
    present_t = torch.as_tensor(present, device='cuda')
    ltm_t = torch.as_tensor(ltm, device='cuda')
    theta = torch.nn.Parameter(tensor(initial_theta))
    optimizer = torch.optim.Adam([theta], lr=.05)

    def predict():
        current = source_t @ (original_t * theta[membership_t]).T + background_t
        activation = torch.relu(200 * torch.tanh((current - 7.5) / 200))
        rate = periodic_filter(activation, .00025, .02)
        pooled = rate[::20] @ pool_t.T
        active = periodic_filter(pooled, .005, .01)
        command = rest_t + .02 * active @ moment_t.T
        loss_q = (command - target_t).square().mean()
        loss = loss_q + 1e-5 * (theta[present_t] - 1).square().mean()
        loss = loss + .01 * (torch.relu(20 - pooled[:, ltm_t]) / 20).square().mean()
        return loss, loss_q, command, rate, pooled

    history, best, best_loss = [], None, math.inf
    stop = 'updates'
    for step in range(1001):
        optimizer.zero_grad(set_to_none=True)
        loss, loss_q, _, _, _ = predict()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError('Nonfinite conditional recruitment objective')
        if value < best_loss:
            best_loss = value
            best = theta.detach().clone()
        if step % 50 == 0:
            history.append({'step': step, 'loss': value,
                'command_rms_rad': math.sqrt(float(loss_q.detach())),
                'wall_s': time.perf_counter() - began})
        if step % 250 == 0:
            print(json.dumps(history[-1]), flush=True)
        if time.perf_counter() - began > 150:
            stop = 'wall_limit'; break
        if step == 1000:
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.clamp_(.001, 10)
    with torch.no_grad():
        theta.copy_(best)
        loss, loss_q, command, rate, pooled = predict()
    fitted_theta = theta.detach().cpu().numpy()
    motor_edges, changed, source_values = [], [], []
    for row, motor_id in enumerate(motor):
        edges = np.arange(weights.indptr[motor_id], weights.indptr[motor_id + 1])
        edges = edges[weights.data[edges] != 0]
        pre = weights.indices[edges]
        factors = np.full(len(edges), .001)
        factors[np.isin(pre, nav_ids)] = 1
        for col, input_id in enumerate(inputs):
            factors[pre == input_id] = fitted_theta[membership[row], col]
        motor_edges.extend(edges); source_values.extend(weights.data[edges])
        changed.extend((weights.data[edges] * factors).astype(np.float32))
    edge_ids = np.r_[relay_patch['edge_ids'], np.array(motor_edges, dtype=np.int64)]
    source_values = np.r_[relay_patch['source_values'], np.array(source_values, dtype=np.float32)]
    changed = np.r_[relay_patch['values'], np.array(changed, dtype=np.float32)]
    checked = patch_weights(weights, edge_ids, source_values, changed)
    np.testing.assert_array_equal(checked[motor][:, nav_ids].data, weights[motor][:, nav_ids].data)
    np.savez_compressed(out / 'calibration.npz', edge_ids=edge_ids, source_values=source_values,
        values=changed, zero_adaptation_ids=np.union1d(data['relay_ids'], motor),
        theta=fitted_theta, input_ids=inputs, motor_ids=motor, moment=moment, rest=rest,
        pool=pool, membership=membership, ltm=ltm)
    np.savez_compressed(out / 'prediction.npz', command=command.detach().cpu().numpy(),
        motor_rate=rate.detach().cpu().numpy(), pooled=pooled.detach().cpu().numpy(), target=target)
    write_json(out / 'muscle-provenance.json', {'graph_hash': graph.hash,
        'motor_roster': collection['motor_roster'], 'muscles': groups,
        'input_nodes': [graph.nodes[i] for i in inputs]})
    report = {'experiment': spec['experiment'], 'validity': 'valid_conditional_fit',
        'hypothesis_verdict': 'inconclusive_until_actual_body', 'adoption': 'not_applicable',
        'milestone_outcome': 'pending', 'stop': stop, 'iterations': step,
        'loss': float(loss), 'conditional_command_rms_rad': math.sqrt(float(loss_q)),
        'per_joint_rms_rad': np.sqrt(np.mean((command.detach().cpu().numpy() - target) ** 2, axis=0)).tolist(),
        'fit_parameters': int(present.sum()), 'source_nonzero_motor_edges': len(motor_edges),
        'source_to_motor_edges': int(np.count_nonzero(original)), 'anatomical_rank': int(np.linalg.matrix_rank(moment)),
        'motor_cells_without_thirty_source_input': int(np.sum(np.all(original == 0, axis=1))),
        'device': str(theta.device), 'gpu': torch.cuda.get_device_name(),
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0,
        'wall_s': time.perf_counter() - began, 'history': history}
    write_json(out / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'per_joint_rms_rad')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'identification', 'teacher', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.source, args.reference, args.identification, args.teacher, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
