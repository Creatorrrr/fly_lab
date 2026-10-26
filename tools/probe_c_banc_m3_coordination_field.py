"""Observe actual source coordination paths during an unchanged E31 replay."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_core_adaptation import CORE_TYPES
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_m3_knee_trial import KneeTrial
from tools.verify_c_banc_observation import exact


def coordination_paths(graph, weights, core):
    direct = weights[core][:, core].copy(); direct.eliminate_zeros()
    n, labels = connected_components(direct, directed=False)
    if n != 6 or not np.all(np.bincount(labels) == 4):
        raise ValueError('Six source core groups required')
    incoming = weights[:, core].tocsr(); incoming.eliminate_zeros()
    outgoing = weights[core].tocsc(); outgoing.eliminate_zeros()
    possibilities = np.intersect1d(incoming.nonzero()[0], outgoing.nonzero()[1])
    paths = []
    for relay in possibilities:
        if relay in core or graph.nodes[relay]['super_class'] != 'ventral_nerve_cord_intrinsic':
            continue
        a, b = incoming.indptr[relay:relay+2]; c, d = outgoing.indptr[relay:relay+2]
        for i in range(a, b):
            for j in range(c, d):
                pre = incoming.indices[i]; post = outgoing.indices[j]
                if labels[pre] != labels[post]:
                    paths.append((int(core[pre]), int(relay), int(core[post]),
                        float(incoming.data[i]), float(outgoing.data[j]), int(labels[pre]), int(labels[post])))
    return np.unique([row[1] for row in paths]), paths, labels


def run(source, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E34-coordination-field',
        'question': 'Which source inputs place existing inter-core relay and target cells outside a responsive range during actual E31 movement?',
        'intervention': 'None. Restore original E31 initial complete state and observe24core+164source bridgingVNC+2DNg100. All728two-edge cross-group paths. Runtime input remains actual senses/unchangedtonic; recording does not inject currents.',
        'observations': 'NativeCUDA recurrent current and rate/z/drive at5ms. Decompose current into24core,164bridges,2DNg100 and residual allothercells using exact source float32weights but float64smallmatmul; residual is CUDA full current minus those components. Report local slope-mediated pathgain without claiming phase transfer, causality or global stability.',
        'validity': 'All440physical/control/MN states must equal E31 trace; finalfullr/z must equal E32baseline2.2s. Sources, routing overlap and raw arrays retained before summaries.',
        'decision': 'Use actual receptor/relay/input-range evidence to choose a bounded physiological calibration. No scope expansion, arbitrarygain grid, teacher reconstruction gate or adoption from this observation.',
        'limits': 'OneGPU,2.2modelsec,<=120wallsec; no source/liveUI/F6changes',
        'source_trace_sha256': sha256(source / 'trace.json'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec); (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter(); graph = GraphStore.load(graph_path)
    session = KneeTrial.from_checkpoint(graph, StateStore.load(source / 'start'))
    rows = [capture_control(session)]; observation = []
    selected = np.array([], dtype=np.int64); local = np.empty((0, 0))
    try:
        matrix = source_matrix(session.network)
        core = np.array([i for i, n in enumerate(graph.nodes) if n.get('cell_type') in CORE_TYPES])
        bridges, paths, groups = coordination_paths(graph, matrix, core)
        if len(bridges) != 164 or len(paths) != 728:
            raise ValueError('Frozen path audit differs')
        selected = np.r_[core, bridges, session.descending]
        if len(np.unique(selected)) != 190:
            raise ValueError('Source observation groups overlap')
        lookup = {int(neuron): i for i, neuron in enumerate(selected)}
        local = matrix[selected][:, selected].toarray().astype(float)
        provenance = {'indices': selected.tolist(), 'nodes': [graph.nodes[i] for i in selected],
            'core_groups': groups.tolist(), 'paths': paths,
            'navigation_port_overlap': np.intersect1d(bridges, session.navigation.encoder.ports).tolist(),
            'sensory_port_overlap': np.intersect1d(bridges, session.adapter._sensory_ids).tolist(),
            'columns': ['rate', 'adaptation', 'CUDA_recurrent_J', 'core_J', 'bridge_J', 'DNg100_J', 'other_J', 'direct_drive']}
        write_json(out / 'provenance.json', provenance)
        def observe():
            net = session.network
            rate = net.rate[selected].cpu().numpy().copy()
            z = net.adaptation[selected].cpu().numpy().copy()
            total = net._current(net.rate)[selected].cpu().numpy().copy()
            pieces = [local[:, a:b] @ rate[a:b] for a, b in ((0, 24), (24, 188), (188, 190))]
            drive = net.drive[selected].cpu().numpy().copy()
            observation.append(np.stack((rate, z, total, *pieces, total - sum(pieces), drive), axis=1))
        observe()
        expected = json.loads((source / 'trace.json').read_text())['samples'][:441]
        for tick in range(1, 441):
            if time.perf_counter() - began > 115:
                raise TimeoutError('Bounded E34 observation exceeded wall allowance')
            session.advance(1); row = capture_control(session); rows.append(row); observe()
            for key in ('qpos', 'qvel', 'ctrl', 'joint_targets_rad', 'adhesion', 'tendon_inputs', 'motor_rates'):
                exact(expected[tick][key], row[key], 'E34 source ' + key)
        final = session.snapshot(); StateStore.save(out / 'final', final)
        reference = StateStore.load(Path('verification/banc-natural-gait-m3-20260919/e32-core-adaptation/baseline/final'))
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(reference['network'][key], final['network'][key], 'E34 complete final neural ' + key)
    finally:
        np.savez_compressed(out / 'observations.npz', values=np.asarray(observation), indices=selected,
            local_weights=local)
        write_json(out / 'trace.json', {'samples': rows, 'model_hash': session.network.identity})
        session.close()
    obs = np.asarray(observation); t = np.arange(len(obs)) * .005; mask = (t >= .5) & (t < 2.2)
    margin = obs[:, :, 2] + obs[:, :, 7] - obs[:, :, 1] - 7.5
    slope = np.where(margin > 0, 1 - np.tanh(margin / 200) ** 2, 0.)
    structural = np.array([abs(row[3] * row[4]) for row in paths])
    relays = np.array([lookup[row[1]] for row in paths]); posts = np.array([lookup[row[2]] for row in paths])
    pathgain = slope[:, relays] * slope[:, posts] * structural
    sections = {}
    for name, a, b in (('core', 0, 24), ('bridges', 24, 188)):
        current = obs[mask, a:b]
        sections[name] = {'rectified_cell_time_fraction': float((margin[mask, a:b] <= 0).mean()),
            'half_responsive_cell_time_fraction': float((slope[mask, a:b] >= .5).mean()),
            'median_abs_J_by_source': dict(zip(('core', 'bridges', 'DNg100', 'other', 'direct'),
                np.median(abs(current[:, :, [3, 4, 5, 6, 7]]), axis=(0, 1)).tolist()))}
    write_json(out / 'result.json', {'experiment': spec['experiment'], 'validity': 'valid_exact_observation',
        'hypothesis_verdict': 'descriptive_local_operating_range', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'matched_controls': 440, 'final_full_neural_exact': True, 'sections': sections,
        'path_gain_ratio_quantiles': np.percentile(pathgain[mask].sum(axis=1) / structural.sum(), [5, 50, 95]).tolist(),
        'neural_physics_model_s': 2.2, 'wall_s': time.perf_counter()-began})
    np.savez_compressed(out / 'local-transfer.npz', time_s=t, margin=margin, slope=slope,
        path_gain=pathgain, structural_path_gain=structural)
    print(json.dumps({'sections': sections, 'matched_controls': 440, 'wall_s': time.perf_counter()-began}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory must be new')
    try:
        run(args.source, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'error': type(error).__name__, 'message': str(error),
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested'})
        raise
