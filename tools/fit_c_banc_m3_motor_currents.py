"""Existing-sign/type-pair motor current identification from teaching data.

Conditional replay still reads a forced teacher network. It cannot establish
autonomous activity; only explicit existing-edge efficacy changes are exported.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.train_c_banc_m3_recurrent import command_metrics


def cell_key(node):
    return (node.get('cell_type') or node['id'],
            node.get('source_annotations', {}).get('body_part_effector') or '')


def grouped_design(d, graph):
    rows, parameter_keys, pairs, edge_parameters = [], [], {}, []
    # Load compressed arrays once; repeated npz indexing decompresses whole traces.
    d = {k:d[k] for k in d.files} if isinstance(d, np.lib.npyio.NpzFile) else d
    pre_ids, motor_ids = d['observed_ids'], d['motor_ids']
    regrouping_error = 0.
    source_reference = np.empty_like(d['source_current'], dtype=np.float64)
    for m, post in enumerate(motor_ids):
        sl = slice(d['incoming_indptr'][m], d['incoming_indptr'][m+1])
        sources = d['incoming_indices'][sl]
        weights = d['incoming_data'][sl]
        group, columns = [], {}
        for pre in sources:
            relationship = 'same' if graph.nodes[post].get('soma_side') == graph.nodes[pre].get('soma_side') else 'different'
            key = (cell_key(graph.nodes[post]), cell_key(graph.nodes[pre]), relationship)
            if key not in pairs:
                pairs[key] = len(parameter_keys)
                parameter_keys.append(key)
            parameter = pairs[key]
            edge_parameters.append(parameter)
            if parameter not in columns:
                columns[parameter] = len(columns)
            group.append(columns[parameter])
        x = np.zeros((len(columns), len(d['time_s'])), np.float64)
        values = d['presynaptic'][:, np.searchsorted(pre_ids, sources)].astype(np.float64) * weights
        np.add.at(x, np.asarray(group), values.T)
        source_reference[:, m] = values.sum(axis=1)
        regrouping_error = max(regrouping_error, float(np.max(np.abs(x.sum(axis=0)-source_reference[:, m]))))
        rows.append((x.T, list(columns)))
    width = max(x.shape[1] for x, _ in rows)
    design = np.zeros((len(rows), len(d['time_s']), width), np.float32)
    indices = np.zeros((len(rows), width), np.int64)
    for m, (x, ids) in enumerate(rows):
        design[m, :, :x.shape[1]] = x
        indices[m, :len(ids)] = ids
    return design, indices, np.asarray(edge_parameters), parameter_keys, source_reference, regrouping_error


def conditional_commands(currents, drives, d):
    """Native-step rate equation with interpolated *teacher* presynaptic current."""
    r, z = np.zeros(len(d['motor_ids'])), np.zeros(len(d['motor_ids']))
    activation = np.zeros(d['moment'].shape[1])
    rates, command = [], []
    previous = np.zeros_like(r)
    groups = d['motor_group']
    counts = np.bincount(groups, minlength=len(activation))
    def rhs(rr, zz, current, drive):
        u = np.maximum(200*np.tanh((current+drive-7.5-zz)/200), 0)
        return (u-rr)/.02, (8*rr-zz)/.3
    for t, (next_current, drive) in enumerate(zip(currents, drives)):
        for step in range(10):
            c0 = previous+(next_current-previous)*(step/10)
            cm = previous+(next_current-previous)*((step+.5)/10)
            c1 = previous+(next_current-previous)*((step+1)/10)
            dt = .00025
            k1, a1 = rhs(r, z, c0, drive)
            k2, a2 = rhs(r+dt*.5*k1, z+dt*.5*a1, cm, drive)
            k3, a3 = rhs(r+dt*.5*k2, z+dt*.5*a2, cm, drive)
            k4, a4 = rhs(r+dt*k3, z+dt*a3, c1, drive)
            r += dt/6*(k1+2*k2+2*k3+k4)
            z += dt/6*(a1+2*a2+2*a3+a4)
        previous = next_current
        rates.append(r.copy())
        if t % 2 == 1:
            pooled = np.bincount(groups, weights=r, minlength=len(activation))/counts
            activation += (1-math.exp(-.5))*(pooled-activation)
            command.append(d['rest']+.02*d['moment']@activation)
    return np.asarray(command), np.asarray(rates)


def run(source, graph_path, out, preconditioned=False):
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        'experiment': 'M3-E13c-conditioned-current-fit' if preconditioned else 'M3-E13b-motor-current-fit',
        'hypothesis': 'Motor teaching provides sufficiently rich source presynaptic activity to identify existing motor input efficacies that reproduce the feasible motor trajectories',
        'intervention': 'Positive[.1,10] efficacy multiplier shared by postsynaptic type/effector, presynaptic type/effector and same/different soma side on ALL existing nonzero motor incoming edges. Other graph edges and all cell dynamics unchanged. This explicitly expands E09 internal-only mask; no edges/signs/node identities added or removed.',
        'fit': 'Convex box-constrained current MSE normalized by cap200, tiny-u<=1e-6 upper hinge; mean squared efficacy deviation penalty1e-4. L-BFGS-B max500 iterations; whole prefix0..1.2s training only. Parameters selected solely by training objective.',
        'validation': '1.5..2.2s is a development diagnostic, already exposed by inverse labels. Report current/tanh-target errors and continuous conditional motor ODE replay with actual student adaptation, native.25ms and linearly interpolated teacher presynaptic currents sampled2.5ms.',
        'screen': 'Conditional command RMS<=.05rad and all6 tibia joint command R2>=.5; these are command angles, not observed geometric knees or autonomous ability. Any passing candidate still needs full BANC with all teacher/clamp/boundary inputs removed and neutral plus taught-state starts. F1-F8 unchanged.',
        'limits': 'One CUDA fit<=300wall sec; no physical or full-neural rollout here',
        'source_sha256': sha256(source/'signals.npz'), 'tool_sha256': sha256(__file__),
        'measurement_revision': 'Original E13 stopped before fitting because float32 grouping vs original serial CSR cancellation exceeded tolerance at22 samples. E13b validates the grouping against direct float64 sums of the exact source weights/rates, max discrepancy<=1e-8; original float32 disagreement remains reported. Fit hypothesis/bounds/screen unchanged.',
    }
    if preconditioned:
        spec['numerical_optimizer_revision'] = 'E13b projected gradient max4.98e-4 exceeds1e-6 and Hessian diagonal spans1e-8..1.035. Same convex objective/bounds/data/screens; rescale variables by training Hessian diagonal square root and allow5000 iterations within the SAME300wall limit. No biological parameter-bound search.'
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    with np.load(source/'signals.npz') as loaded:
        d = {k:loaded[k] for k in loaded.files}
    graph = GraphStore.load(graph_path)
    topology = json.loads((source/'topology.json').read_text())
    if graph.hash != topology['graph_hash']:
        raise ValueError('Source graph mismatch')
    began = time.perf_counter()
    design, indices, edge_parameters, keys, source_reference, regrouping_error = grouped_design(d, graph)
    write_json(out/'reconstruction.json', {
        'float64_regrouping_max_error':regrouping_error,
        'original_serial_float32_max_error':float(np.max(abs(source_reference-d['source_current']))),
        'grouped_float32_max_error':float(np.max(abs(source_reference-design.sum(axis=2,dtype=np.float64).T))),
        'design_shape':list(design.shape), 'parameters':len(keys)})
    if regrouping_error > 1e-8:
        raise ValueError('Grouped source current reconstruction mismatch')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA fit required')
    torch.backends.cuda.matmul.allow_tf32 = False
    train = d['time_s'] <= 1.2+1e-9
    validation = (d['time_s'] >= 1.5-1e-9) & (d['time_s'] <= 2.2+1e-9)
    x = torch.as_tensor(design[:,train,:]/200, device='cuda')
    index = torch.as_tensor(indices, device='cuda')
    target = torch.as_tensor(d['required_current'][train].T/200, dtype=torch.float32, device='cuda')
    upper = torch.as_tensor((d['target_activation'][train] <= 1e-6).T, device='cuda')
    if preconditioned:
        diagonal = torch.zeros(len(keys), device='cuda')
        diagonal.scatter_add_(0, index.flatten(), (2*x.square().sum(dim=1)/target.numel()).flatten())
        diagonal.add_(2e-4/len(keys))
        scaling = np.sqrt(diagonal.cpu().numpy().astype(float))
    else:
        scaling = np.ones(len(keys))
    history = []
    best = {'objective': float('inf'), 'parameters': np.ones(len(keys)), 'gradient_max': None}
    def objective(values):
        if time.perf_counter()-began > 285:
            raise TimeoutError('E13 fit wall limit')
        parameters = values/scaling
        theta = torch.tensor(parameters, dtype=torch.float32, device='cuda', requires_grad=True)
        prediction = torch.bmm(x, theta[index].unsqueeze(2)).squeeze(2)
        residual = prediction-target
        residual = torch.where(upper, residual.clamp_min(0), residual)
        loss = residual.square().mean()+1e-4*(theta-1).square().mean()
        loss.backward()
        value = float(loss.detach().cpu())
        gradient = theta.grad.detach().cpu().numpy().astype(float)/scaling
        if value < best['objective']:
            best.update(objective=value, parameters=parameters.copy(), gradient_max=float(np.max(np.abs(gradient))))
        history.append({'evaluation': len(history), 'objective': value, 'wall_s': time.perf_counter()-began})
        if len(history) % 100 == 0:
            print(json.dumps(history[-1]), flush=True)
        return value, gradient
    try:
        optimization = minimize(objective, scaling.copy(), jac=True, method='L-BFGS-B',
            bounds=list(zip(.1*scaling,10*scaling)), options={'maxiter':5000 if preconditioned else 500,'maxls':30,'ftol':1e-10,'gtol':1e-6,'maxcor':10})
    except TimeoutError as exc:
        optimization = SimpleNamespace(success=False, message=str(exc), nit=-1)
    theta = best['parameters']
    currents = np.einsum('mtg,mg->tm', design, theta[indices], optimize=True)
    np.savez_compressed(out/'candidate.npz', efficacy=theta, edge_parameters=edge_parameters,
        incoming_indptr=d['incoming_indptr'], incoming_indices=d['incoming_indices'],
        incoming_source_data=d['incoming_data'], incoming_data=d['incoming_data']*theta[edge_parameters],
        motor_ids=d['motor_ids'], observed_ids=d['observed_ids'], currents=currents,
        moment=d['moment'], rest=d['rest'], motor_group=d['motor_group'])
    write_json(out/'parameter-groups.json', keys)
    write_json(out/'history.json', history)
    command, rates = conditional_commands(currents, d['external_drive'], d)
    np.savez_compressed(out/'conditional-replay.npz', command=command, rate=rates)
    times = d['time_s'][1::2]
    joints = np.array([[0,2,1,4,3,5,6],[7,9,8,11,10,12,13],[14,16,15,18,17,19,20],
        [21,23,22,25,24,26,27],[28,30,29,32,31,33,34],[35,37,36,39,38,40,41]])
    q_metrics = command_metrics(command, d['command'], (times>=1.5-1e-9)&(times<=2.2+1e-9), joints)
    u_pred = np.maximum(200*np.tanh((currents+d['external_drive']-7.5-d['adaptation'])/200),0)
    target_error = u_pred-d['target_activation']
    passed = q_metrics['rms_rad'] <= .05 and all(v is not None and v >= .5 for v in q_metrics['knee_r2'])
    result = {'experiment':spec['experiment'], 'run_status':'completed',
        'validity':'valid_conditional_development_diagnostic',
        'hypothesis_verdict':'conditional_screen_supported' if passed else 'not_supported_in_bounded_fit',
        'adoption':'defer_full_student' if passed else 'reject', 'milestone_outcome':'pending',
        'device':torch.cuda.get_device_name(0), 'wall_s':time.perf_counter()-began,
        'solver_success':bool(optimization.success), 'solver_message':str(optimization.message),
        'iterations':int(optimization.nit), 'evaluations':len(history),
        'initial_objective':history[0]['objective'], 'best_objective':best['objective'],
        'parameters':len(keys), 'existing_motor_incoming_edges':len(edge_parameters),
        'nonzero_source_signs_preserved':bool(np.array_equal(np.sign(d['incoming_data']),np.sign(d['incoming_data']*theta[edge_parameters]))),
        'efficacy_range':[float(theta.min()),float(theta.max())],
        'target_activation_train_rms':float(np.sqrt(np.mean(target_error[train]**2))),
        'target_activation_validation_rms':float(np.sqrt(np.mean(target_error[validation]**2))),
        'conditional_validation':q_metrics, 'screen_passed':passed,
        'candidate_sha256':sha256(out/'candidate.npz')}
    write_json(out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','graph','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preconditioned',action='store_true')
    a=p.parse_args()
    run(a.source,a.graph,a.out,a.preconditioned)
