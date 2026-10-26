"""Prepare the frozen E97 readout as an explicit failed-shape diagnostic."""

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


def fitted_values(graph, base, cal, sources, theta, present, membership):
    """Change only original existing source-to-motor edges and preserve signs."""
    values = cal['values'].copy()
    positions = {int(edge): i for i, edge in enumerate(cal['edge_ids'])}
    edges = []
    for row, motor_id in enumerate(cal['motor_ids']):
        for edge in range(graph.indptr[motor_id], graph.indptr[motor_id + 1]):
            pre = int(graph.indices[edge])
            col = int(np.searchsorted(sources, pre))
            if col == len(sources) or sources[col] != pre or base.data[edge] == 0:
                continue
            if edge not in positions or not present[membership[row], col]:
                raise ValueError('Missing existing signed source-to-MN edge or parameter')
            values[positions[edge]] = np.float32(base.data[edge] * theta[membership[row], col])
            edges.append(edge)
    if len(edges) != 1536 or len(set(edges)) != 1536:
        raise ValueError('Exactly 1536 original signed MN edges required')
    return values, np.asarray(edges, dtype=np.int64)


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    e95 = root / 'e95-rm-e2-premotor-source'
    e97 = root / 'e97-softplus-training-native-recruitment'
    prior = json.loads((e97 / 'result.json').read_text())
    review = json.loads((e97 / 'parent-first-review.json').read_text())
    if (prior.get('validity') != 'valid_conditional_fit'
            or prior.get('shape_screen', {}).get('supported') is not False
            or prior.get('fit_parameters') != 590 or prior.get('fitted_edges') != 1536
            or not review):
        raise ValueError('Reviewed E97 failed conditional shape and 590/1536 fit required')
    spec = {'experiment': 'M3-E98-native-readout-body-discriminator-preparation',
        'status': 'diagnostic_from_failed_conditional_shape',
        'E97_shape_supported': False,
        'intervention': 'E95 original neutral plus E97 fixed positive efficacy on 1536 existing signed source-to-MN edges only; no fit, softplus, runtime phase or preload.',
        'limits': 'Preparation <=90 wall seconds; no neural/physical advance.',
        'e95_calibration_sha256': sha256(e95 / 'calibration.npz'),
        'e95_neutral_manifest_sha256': sha256(e95 / 'candidate-neutral/manifest.json'),
        'e97_coefficients_sha256': sha256(e97 / 'fit-coefficients.npz'),
        'e97_result_sha256': sha256(e97 / 'result.json'),
        'e97_parent_review_sha256': sha256(e97 / 'parent-first-review.json'),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    graph = GraphStore.load(graph_path)
    cal = {key: value.copy() for key, value in np.load(e95 / 'calibration.npz').items()}
    old = {key: value.copy() for key, value in cal.items()}
    coefficients = np.load(e97 / 'fit-coefficients.npz')
    sources = np.unique(np.r_[cal['input_ids'], cal['intrinsic_tonic_ids'], cal['coupling_ids']])
    theta = coefficients['theta']
    present = coefficients['present']
    if (len(sources) != 48 or theta.shape != (96, 48) or present.shape != theta.shape
            or int(present.sum()) != 590 or not np.array_equal(sources, coefficients['source_ids'])
            or cal['theta'].shape != (96, 47) or len(cal['theta_source_ids']) != 47
            or len(cal['tau_ids']) != 38 or not np.isfinite(theta).all()
            or np.any(theta[present] < .001) or np.any(theta[present] > 10)):
        raise ValueError('Exact E95/E97 source, tau and positive efficacy provenance required')
    anatomy = json.loads((e95 / 'muscle-provenance.json').read_text())
    membership = np.full(391, -1, dtype=np.int64)
    for group, muscle in enumerate(anatomy['muscles']):
        membership[muscle['columns']] = group
    if (len(anatomy['muscles']) != 96 or np.any(membership < 0)
            or sorted(np.concatenate([m['columns'] for m in anatomy['muscles']]).tolist()) != list(range(391))):
        raise ValueError('Exact 96 muscle/391 motor anatomical membership required')
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, cal['motor_ids'])
    current = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    values, edges = fitted_values(graph, base, cal, sources, theta, present, membership)
    cal['values'] = values
    cal['legacy_theta_e89'] = cal['theta'].copy()
    cal['legacy_theta_source_ids_e89'] = cal['theta_source_ids'].copy()
    cal['theta'] = theta.copy()
    cal['theta_source_ids'] = sources.copy()
    changed = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    exact(current.indptr, changed.indptr, 'Original CSR indptr')
    exact(current.indices, changed.indices, 'Original CSR indices')
    other = np.ones(len(current.data), dtype=bool)
    other[edges] = False
    exact(current.data[other], changed.data[other], 'All nonfitted edges')
    exact(np.sign(changed.data[edges]), np.sign(base.data[edges]), 'Original signed edges')
    for key, value in old.items():
        if key not in ('values', 'theta', 'theta_source_ids'):
            exact(cal[key], value, 'Original E95 calibration ' + key)
    anatomy['E98_fixed_E97_readout'] = {'status': spec['status'],
        'source_ids': sources.tolist(), 'fitted_signed_edges': len(edges),
        'positive_efficacy_parameters': int(present.sum()),
        'runtime_softplus': False, 'E97_shape_supported': False}
    np.savez_compressed(out / 'calibration.npz', **cal)
    write_json(out / 'muscle-provenance.json', anatomy)
    initial = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base_trial = EfficacyTrial(graph, initial, cal['edge_ids'], cal['source_values'],
        cal['values'], zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial, cal,
        anatomy['muscles']), cal['tau_ids'], cal['tau_values'])
    try:
        if session.network.device != 'cuda':
            raise ValueError('Actual CUDA constructor required')
        actual = source_matrix(session.network)
        exact(actual.data, changed.data, 'Actual runtime original CSR weights')
        exact(actual.indptr, changed.indptr, 'Runtime CSR indptr')
        exact(actual.indices, changed.indices, 'Runtime CSR indices')
        neutral = session.snapshot()
        exact(initial['body'], neutral['body'], 'Original neutral body/world')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(initial['network'][key], neutral['network'][key], 'Original neutral ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
    finally:
        session.close()
    clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
        StateStore.load(out / 'candidate-neutral'))
    try:
        exact(neutral, clone.snapshot(), 'Whole neutral disk exact')
    finally:
        clone.close()
    wall = time.perf_counter() - began
    result = {'experiment': spec['experiment'], 'validity': 'valid_prepared_diagnostic',
        'status': spec['status'], 'E97_shape_supported': False,
        'fitted_edges': len(edges), 'fit_parameters': int(present.sum()),
        'selected_sources': len(sources), 'tau_values_exact': True,
        'nonfit_weight_exact': True, 'neutral_disk_exact': True,
        'actual_neural_s': 0., 'actual_physical_s': 0.,
        'wall_s': wall, 'resource_limit_met': wall <= 90}
    write_json(out / 'result.json', result)
    if wall > 90:
        raise TimeoutError('E98 preparation exceeded 90 wall seconds')
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
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error),
                'execution_wall_s': time.perf_counter() - started,
                'actual_neural_s': 0., 'actual_physical_s': 0.})
        raise
