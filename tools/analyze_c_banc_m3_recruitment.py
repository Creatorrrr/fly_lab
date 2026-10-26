"""Source attribution of an exposed diagnostic trajectory; no neural advance."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import build_spec
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json


def run(source, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        'experiment': 'M3-E38-exposed-motor-recruitment-attribution',
        'question': 'Which existing incoming populations account for the total current of silent or nearly silent muscles in the exposed E37 core-forcing diagnostic?',
        'method': 'Reconstruct original M2 float32 signed count weights and its original navigation calibration. Attribute every incoming edge to each of391leg motor units at the saved final state. Compare float64 source sums against stored CUDA currents. Separately pool actual rates/core and noncore currents over the entire exposed trajectory by anatomical muscle.',
        'interpretation': 'A single forced transient and arithmetic current decomposition, not a causal lesion, a normal unforced-state diagnosis, or a synaptic efficacy estimate. Current magnitude is not a causal contribution percentage. No candidate selected by this report.',
        'scope': 'CPU read-only source analysis; zero additional neural or physical steps; F1-F8 remain pending; F6 unexposed',
        'source_trace_sha256': sha256(source / 'trace.json'),
        'source_observations_sha256': sha256(source / 'observations.npz'),
        'tool_sha256': sha256(__file__),
    }
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    trace = json.loads((source / 'trace.json').read_text())
    saved = StateStore.load(source / 'teaching-state')
    state = saved['whole_neural']
    obs = np.load(source / 'observations.npz')
    anatomy = build_spec(graph, 2)
    motor = graph.resolve(sorted({n for leg in anatomy['rows'] for g in leg['muscles'] for n in g['ids']}))
    motor = np.sort(motor)
    if graph.hash != trace['graph_hash'] or len(motor) != 391:
        raise ValueError('Source graph or anatomical motor roster differs')
    rates = np.array([r['motor_rates'] for r in trace['samples'][1:]])
    np.testing.assert_array_equal(rates[-1], state['rate'][motor])
    np.testing.assert_array_equal(state['output_mask'][motor], np.ones(391))
    np.testing.assert_array_equal(state['drive'][motor], np.zeros(391))
    encoder = BancTargetEncoder(graph, forward_drive=5000)
    weights, _ = rate_weights(graph)
    weights = encoder.calibrate_weights(weights, motor)
    np.testing.assert_array_equal(weights.indptr, graph.indptr)
    np.testing.assert_array_equal(weights.indices, graph.indices)
    row_ids = np.repeat(np.arange(391), np.diff(graph.indptr)[motor])
    edge_ids = np.concatenate([np.arange(graph.indptr[i], graph.indptr[i + 1]) for i in motor])
    pre = graph.indices[edge_ids]
    w = weights.data[edge_ids].astype(float)
    contribution = w * state['rate'][pre]
    category = np.where(np.isin(pre, saved['forced_core_ids']), 0,
                        np.where(np.isin(pre, encoder.ports), 2, 1))
    category = category * 2 + (w < 0)
    sums = np.bincount(row_ids * 6 + category, weights=contribution,
                       minlength=391 * 6).reshape(391, 6)
    cuda = obs['motor_total_current'][-1].astype(float)
    error = abs(sums.sum(axis=1) - cuda)
    np.savez_compressed(out / 'source-currents.npz', motor_ids=motor,
        incoming_edge_ids=edge_ids, motor_row=row_ids, presynaptic_ids=pre,
        source_weights=w, presynaptic_rate=state['rate'][pre], contribution=contribution,
        category=category, sums=sums, CUDA_total_current=cuda,
        source_sum_error=error, final_rate=state['rate'][motor],
        final_adaptation=state['adaptation'][motor], final_drive=state['drive'][motor])
    # Independent float64 summation differs from the original float32 CUDA CSR
    # reduction. This tolerance checks arithmetic identity, not neural futures.
    np.testing.assert_allclose(sums.sum(axis=1), cuda, rtol=2e-5, atol=.02)
    np.testing.assert_allclose(sums[:, :2].sum(axis=1),
                               obs['motor_core_current'][-1], rtol=1e-12, atol=1e-10)
    muscle_results = []
    for leg in anatomy['rows']:
        for group in leg['muscles']:
            ids = graph.resolve(group['ids'])
            cols = np.searchsorted(motor, ids)
            r = rates[:, cols]
            record = {'leg': leg['leg'], 'muscle': group['muscle'],
                'motor_roots': [graph.nodes[i]['root_id'] for i in ids],
                'units': len(ids), 'entirely_silent_units': int(np.sum(np.max(r, axis=0) == 0)),
                'rate_p0_p50_p95_p100': np.percentile(r, [0, 50, 95, 100]).tolist(),
                'mean_rate_final': float(np.mean(r[-1])),
                'total_current_range': [float(np.min(obs['motor_total_current'][:, cols])), float(np.max(obs['motor_total_current'][:, cols]))],
                'direct_core_current_range': [float(np.min(obs['motor_core_current'][:, cols])), float(np.max(obs['motor_core_current'][:, cols]))],
                'final_mean_current_categories': np.mean(sums[cols], axis=0).tolist()}
            muscle_results.append(record)
    details = []
    for m in muscle_results:
        if m['entirely_silent_units'] == m['units'] or (m['leg'] == 'rm' and m['muscle'] == 'trochanter_flexor_muscle'):
            for root in m['motor_roots']:
                i = next(k for k, n in enumerate(motor) if graph.nodes[n]['root_id'] == root)
                edges = np.flatnonzero(row_ids == i)
                negative = edges[np.argsort(contribution[edges])[:5]]
                positive = edges[np.argsort(contribution[edges])[-5:][::-1]]
                def sources(selected):
                    return [{'root_id': graph.nodes[pre[e]]['root_id'],
                        'cell_type': graph.nodes[pre[e]]['cell_type'],
                        'super_class': graph.nodes[pre[e]]['super_class'],
                        'source_weight': float(w[e]), 'rate': float(state['rate'][pre[e]]),
                        'current': float(contribution[e])} for e in selected]
                details.append({'leg': m['leg'], 'muscle': m['muscle'], 'motor_root': root,
                    'total_current': float(sums[i].sum()), 'rate': float(state['rate'][motor[i]]),
                    'adaptation': float(state['adaptation'][motor[i]]),
                    'largest_negative_sources': sources(negative),
                    'largest_positive_sources': sources(positive)})
    result = {'experiment': spec['experiment'], 'validity': 'valid_exposed_source_attribution',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'graph_hash': graph.hash, 'source_model_hash': saved['source_model_hash'],
        'source_final_time_s': trace['samples'][-1]['time_s'],
        'category_order': ['core_positive', 'core_negative', 'other_positive', 'other_negative', 'navigation_positive', 'navigation_negative'],
        'maximum_current_reconstruction_error': float(error.max()),
        'navigation_port_rate': state['rate'][encoder.ports].tolist(),
        'navigation_port_drive': state['drive'][encoder.ports].tolist(),
        'muscles': muscle_results, 'silent_muscle_source_details': details,
        'additional_model_s': 0, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('muscles', 'silent_muscle_source_details')}))
    print(json.dumps({'silent_muscles': [(m['leg'], m['muscle'], m['units']) for m in muscle_results if m['entirely_silent_units'] == m['units']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.graph, args.out)
