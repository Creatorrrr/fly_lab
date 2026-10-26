"""Re-evaluate E99's two-cell conditional screen using stored float32 weights."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.prepare_c_banc_m3_e99_lf_tarsal import source_screen
from tools.probe_c_banc_m3_tonic_onset import reduced_rk4


def run(root, graph_path, out):
    if not (out / 'result.json').is_file() or not (out / 'parent-first-review.json').is_file():
        raise ValueError('Frozen reviewed E99 original output required')
    for name in ('corrected-conditional-raw.npz', 'corrected-conditional-screen.json',
            'parent-first-review-amendment.json', 'amendment-runner.py'):
        if (out / name).exists():
            raise FileExistsError('Preserve existing E99 correction: ' + name)
    began = time.perf_counter()
    original_hashes = {name: sha256(out / name) for name in
        ('spec.json', 'runner.py', 'result.json', 'conditional-raw.npz',
         'conditional-screen.json', 'analytical-map.json')}
    (out / 'amendment-runner.py').write_bytes(Path(__file__).read_bytes())
    graph = GraphStore.load(graph_path)
    cal = np.load(out / 'calibration.npz')
    raw = np.load(out / 'conditional-raw.npz')
    actual = np.load(root / 'e98-native-readout-body-discriminator/body/recruitment-signals.npz')
    reference = np.load(root / 'e85-space-compatible-cadence-reference/commands.npz')
    map_record = json.loads((out / 'analytical-map.json').read_text())
    selected_edges = list(map(int, map_record['requested_edge_ids'][-3:]))
    expected_presyn = [105382, 106670, 51845]
    if [int(graph.indices[edge]) for edge in selected_edges] != expected_presyn:
        raise ValueError('Original three selected 11088 presynaptic identities required')
    for edge in selected_edges:
        if not graph.indptr[11088] <= edge < graph.indptr[11089]:
            raise ValueError('All three float32 edges must receive at11088')
    positions = {int(edge): index for index, edge in enumerate(cal['edge_ids'])}
    weights = np.array([cal['values'][positions[edge]] for edge in selected_edges])
    if weights.dtype != np.float32 or not np.isfinite(weights).all():
        raise ValueError('Stored three selected float32 weights required')
    ids = actual['observed_ids']
    source_rates = np.column_stack([np.r_[0., actual['rate'][:, np.flatnonzero(ids == pre)[0]]]
        for pre in expected_presyn])
    inputs = raw['proposed_input_current'].copy()
    inputs[:, 1] = raw['retained_background'][:, 1] + source_rates @ weights.astype(float)
    if not np.array_equal(inputs[:, 0], raw['proposed_input_current'][:, 0]):
        raise ValueError('E99 proposed60038 current must remain unchanged')
    predicted = reduced_rk4(np.zeros((2, 2)), inputs,
        np.full(2, float(np.float32(.02))), 2000)
    times = raw['time_s']
    target = raw['own_LF_inverse_target']
    period = float(reference['period_s'])
    screens = {
        'depressor_60038': source_screen(times, predicted[:, 0], raw['E2_rate'],
            target[:, 0], period, sign=1),
        'levator_11088': source_screen(times, predicted[:, 1], raw['I2_rate'],
            target[:, 1], period, sign=-1)}
    np.savez_compressed(out / 'corrected-conditional-raw.npz', time_s=times,
        source_ids=raw['source_ids'], proposed_input_current=inputs,
        predicted_rate=predicted, retained_background=raw['retained_background'],
        own_LF_inverse_target=target, selected_11088_edge_ids=selected_edges,
        selected_11088_presynaptic_ids=expected_presyn,
        selected_11088_stored_float32_weights=weights)
    write_json(out / 'corrected-conditional-screen.json', screens)
    wall = time.perf_counter() - began
    prior = json.loads((out / 'result.json').read_text())['wall_s']
    original_screens = json.loads((out / 'conditional-screen.json').read_text())
    amendment = {'review_corrections': ['E99-R1', 'E99-R2'],
        'original_outputs_unchanged_sha256': original_hashes,
        'corrected_from_actual_prepared_float32_calibration': True,
        'three_11088_presynaptic_ids': expected_presyn,
        'three_11088_edge_ids': selected_edges,
        'three_11088_stored_float32_weights': weights.tolist(),
        'max_input_difference': float(np.max(abs(inputs - raw['proposed_input_current']))),
        'max_rate_difference': float(np.max(abs(predicted - raw['predicted_rate']))),
        'original_screen': {key: value['screen'] for key, value in original_screens.items()},
        'corrected_screen': {key: value['screen'] for key, value in screens.items()},
        'fixed_window_samples': 1400,
        'fixed_window_all_target_finite': bool(np.isfinite(target[(times >= 2.5) & (times < 9.5)]).all()),
        'conditional_native_ode_calls_in_correction': 1,
        'full_network_neural_s': 0., 'physical_s': 0., 'new_cuda_constructors': 0,
        'prior_preparation_wall_s': prior,
        'correction_wall_s': wall,
        'shared_wall_s': prior + wall,
        'shared_90s_limit_met': prior + wall <= 90,
        'body_trial': 'not_run'}
    write_json(out / 'parent-first-review-amendment.json', amendment)
    print(json.dumps(amendment), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.graph, args.out)
