"""Prepare one existing RM E2-to-IN03A010 premotor source row."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact


def selected_row_patch(graph, original, prior_edge_ids, source_id, relay_id):
    """Patch only existing nonzero source edges in the declared receiver row."""
    edges = np.arange(graph.indptr[relay_id], graph.indptr[relay_id + 1])
    edges = edges[original.data[edges] != 0]
    if len(edges) == 0 or np.intersect1d(edges, prior_edge_ids).size:
        raise ValueError('Nonempty unowned existing receiver row required')
    chosen = edges[original.indices[edges] == source_id]
    if len(chosen) != 1 or original.data[chosen[0]] <= 0:
        raise ValueError('Exactly one existing positive RM E2-to-relay edge required')
    values = original.data[edges] * np.float32(.001)
    values[edges == chosen[0]] = np.float32(1.1707944)
    efficacy = float(values[edges == chosen[0]][0] / original.data[chosen[0]])
    if not .001 <= efficacy <= 10:
        raise ValueError('Selected original-source positive efficacy outside bounds')
    return edges, values, int(chosen[0]), efficacy


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    source = root / 'e93-native-coupling-phase-v2'
    e94 = root / 'e94-native-coupling-phase-body-v3'
    spec = {'experiment': 'M3-E95-rm-e2-premotor-source-preparation',
        'hypothesis': 'One existing RM E2-to-IN03A010 positive input and reduced other incoming original edges can make this source represent RM sternal timing while retaining E93 source coordination.',
        'intervention': 'Existing RM E2 index43618 to IN03A010 index54014 effective1.1707944; other existing nonzero receiver edges original*.001; gain8to0; native tau20ms unchanged. No motor edge, other source row, body, initial state, time/phase input, or fit changes.',
        'limits': '<=90wallseconds, zero experimental neural/physical advance; one future body trial <=285wallseconds and <=10.1 whole neural/physical seconds. F1-F8/M2/UI unchanged.',
        'E93_calibration_sha256': sha256(source / 'calibration.npz'),
        'E93_parent_second_review_sha256': sha256(source / 'parent-second-review.json'),
        'E94_result_sha256': sha256(e94 / 'result.json'),
        'E94_parent_second_review_sha256': sha256(e94 / 'parent-second-review.json'),
        'E94_RM_diagnostic_sha256': sha256(e94 / 'parent-rm-native-source-diagnostic.json'),
        'E94_own_leg_targets_sha256': sha256(e94 / 'parent-own-leg-targets.npz'),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    graph = GraphStore.load(graph_path)
    cal = {key: value.copy() for key, value in np.load(source / 'calibration.npz').items()}
    old = {key: value.copy() for key, value in cal.items()}
    relay_id = int(graph.resolve(['flywire:banc:888:720575941504444354'])[0])
    source_id = int(graph.resolve(['flywire:banc:888:720575941487873488'])[0])
    if (relay_id != 54014 or source_id != 43618
            or graph.nodes[relay_id]['cell_type'] != 'IN03A010'
            or source_id not in cal['core_ids']
            or relay_id in np.r_[cal['input_ids'], cal['zero_adaptation_ids'],
                cal['motor_ids'], cal['tau_ids']]):
        raise ValueError('Exact unused original RM E2 and IN03A010 identities required')
    encoder = BancTargetEncoder(graph, forward_drive=5000)
    if relay_id in encoder.ports:
        raise ValueError('Selected premotor cell overlaps navigation port')
    original, _ = rate_weights(graph)
    original = encoder.calibrate_weights(original, cal['motor_ids'])
    current = patch_weights(original, cal['edge_ids'], cal['source_values'], cal['values'])
    edges, values, chosen, efficacy = selected_row_patch(graph, original,
        cal['edge_ids'], source_id, relay_id)
    if (not np.isclose(float(original.data[chosen]), .18, rtol=1e-6, atol=0)
            or not np.isclose(float(current[144018, relay_id]), .00069,
                rtol=1e-6, atol=0)):
        raise ValueError('Frozen E94 original source and RM motor paths required')
    changed = current.copy()
    changed.data[edges] = values
    other = np.ones(len(changed.data), dtype=bool)
    other[edges] = False
    exact(changed.data[other], current.data[other], 'All nonselected E93 edge data')
    exact(changed.indptr, current.indptr, 'Original CSR row pointers')
    exact(changed.indices, current.indices, 'Original CSR indices')
    for field in ('data', 'indices', 'indptr'):
        exact(getattr(changed[cal['motor_ids']], field),
            getattr(current[cal['motor_ids']], field),
            'All391 motor incoming rows ' + field)
    cal['edge_ids'] = np.r_[cal['edge_ids'], edges]
    cal['source_values'] = np.r_[cal['source_values'], original.data[edges]]
    cal['values'] = np.r_[cal['values'], values]
    cal['input_ids'] = np.r_[cal['input_ids'], relay_id]
    cal['zero_adaptation_ids'] = np.union1d(cal['zero_adaptation_ids'], [relay_id])
    rebuilt = patch_weights(original, cal['edge_ids'], cal['source_values'], cal['values'])
    exact(rebuilt.data, changed.data, 'Exactly one original receiver row changed')
    for key, value in old.items():
        if key not in ('edge_ids', 'source_values', 'values', 'input_ids',
                'zero_adaptation_ids'):
            exact(cal[key], value, 'Unchanged E93 calibration ' + key)
    anatomy = json.loads((source / 'muscle-provenance.json').read_text())
    anatomy['input_nodes'] = anatomy['input_nodes'] + [graph.nodes[relay_id]]
    hypothesis = {'source_id': source_id, 'relay_id': relay_id,
        'selected_edge_id': chosen, 'selected_original_weight': float(original.data[chosen]),
        'selected_effective_weight': float(values[edges == chosen][0]),
        'selected_original_efficacy': efficacy, 'incoming_existing_edges': len(edges),
        'weakened_other_edges': len(edges)-1,
        'unchanged_motor_rows_exact': True, 'unchanged_38_tau_values_exact': True,
        'legacy_theta_47_sources_unchanged': True,
        'expected_native_tau_s': float(np.float32(.02)),
        'expected_direct_drive': 0., 'new_adaptation_gain': 0.,
        'model_advance_s': 0., 'conditional_only': True}
    anatomy['RM_e2_premotor_source_hypothesis'] = hypothesis
    np.savez_compressed(out / 'calibration.npz', **cal)
    write_json(out / 'muscle-provenance.json', anatomy)
    write_json(out / 'source-hypothesis.json', hypothesis)
    initial = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base = EfficacyTrial(graph, initial, cal['edge_ids'], cal['source_values'],
        cal['values'], zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base, cal,
        anatomy['muscles']), cal['tau_ids'], cal['tau_values'])
    try:
        if session.network.device != 'cuda':
            raise ValueError('Actual CUDA constructor required')
        exact(source_matrix(session.network).data, changed.data,
            'Prepared runtime original-CSR current weights')
        if (session.network._cell_gain[relay_id].item() != 0
                or session.network.drive[relay_id].item() != 0
                or session.network.tau[relay_id].item() != float(np.float32(.02))):
            raise ValueError('Prepared RM source gain/drive/native tau required')
        neutral = session.snapshot()
        exact(initial['body'], neutral['body'], 'Original E95 neutral body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(initial['network'][key], neutral['network'][key],
                'Original E95 neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
    finally:
        session.close()
    clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
        StateStore.load(out / 'candidate-neutral'))
    try:
        exact(neutral, clone.snapshot(), 'E95 entire neutral disk state')
    finally:
        clone.close()
    if time.perf_counter() - began > 90:
        raise TimeoutError('E95 preparation exceeded 90s wall limit')
    result = {'experiment': spec['experiment'],
        'validity': 'valid_prepared_existing_RM_source',
        'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable',
        'selected_source_count': len(np.unique(np.r_[cal['input_ids'],
            cal['intrinsic_tonic_ids'], cal['coupling_ids']])),
        'candidate': hypothesis, 'neutral_disk_exact': True,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: ' + str(args.out))
    started = time.perf_counter()
    try:
        run(args.root, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested',
                'error': type(error).__name__, 'message': str(error),
                'execution_wall_s': time.perf_counter() - started,
                'actual_neural_s': 0., 'actual_physical_s': 0.})
        raise
