"""Inspect measured source currents and anatomical muscle activation, without advancing a model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from tools.analyze_c_banc_biological_gait import sha256, write_json


def run(path, graph_path, calibration):
    graph = GraphStore.load(graph_path)
    signals = np.load(path / 'recruitment-signals.npz')
    meta = np.load(path / 'recruitment-metadata.npz')
    cal = np.load(calibration / 'calibration.npz')
    groups = json.loads((calibration / 'muscle-provenance.json').read_text())['muscles']
    index = {int(value): k for k, value in enumerate(signals['observed_ids'])}
    active = (signals['time_s'] >= 3.5) & (signals['time_s'] < 5)
    source = signals['rate'][:, [index[int(value)] for value in meta['source_ids']]]
    background = (signals['recurrent_current'][:, [index[int(value)] for value in meta['motor_ids']]]
        - source @ meta['source_to_motor'].T)
    result = {'window_s': [3.5, 5.0], 'current_sampling': '200Hz endpoint; not a native-time average',
        'tool_sha256': sha256(__file__), 'signals_sha256': sha256(path / 'recruitment-signals.npz'),
        'targets': [], 'actual_neural_s': 0, 'actual_physical_s': 0}
    for root in ('720575941562065920', '720575941503422187', '720575941526736740',
            '720575941518807468', '720575941390358788'):
        cell = int(graph.resolve(['flywire:banc:888:' + root])[0])
        col = index[cell]
        margin = (signals['recurrent_current'][:, col] + signals['drive'][:, col]
            - signals['adaptation'][:, col] - meta['threshold'][col])
        row = {'root_id': root, 'index': cell, 'cell_type': graph.nodes[cell]['cell_type'],
            'tau': float(meta['tau'][col]), 'adaptation_gain': float(meta['_cell_gain'][col]),
            'a': float(meta['a'][col])}
        for key in ('rate', 'recurrent_current', 'drive', 'adaptation'):
            values = signals[key][active, col]
            row[key] = {'min': float(values.min()), 'mean': float(values.mean()), 'max': float(values.max())}
        row['net_threshold_margin'] = {'min': float(margin[active].min()),
            'mean': float(margin[active].mean()), 'max': float(margin[active].max()),
            'positive_fraction': float(np.mean(margin[active] > 0))}
        hit = np.flatnonzero(meta['motor_ids'] == cell)
        if len(hit):
            values = background[active, int(hit[0])]
            row['unobserved_background_range'] = [float(values.min()), float(values.mean()), float(values.max())]
        result['targets'].append(row)
    trace = json.loads((path / 'trace.json').read_text())['samples'][1:]
    command = cal['rest'] + .02 * signals['muscle_activation'] @ cal['moment'].T
    targets = np.array([row['joint_targets_rad'] for row in trace])
    result['command_reconstruction_max_abs_rad'] = float(np.max(abs(command - targets)))
    np.testing.assert_allclose(command, targets, atol=2e-14, rtol=0)
    result['silent_muscle_pools'] = [{'index': k, **groups[k]} for k in range(96)
        if signals['muscle_activation'][active, k].max() == 0]
    result['physical_exact'] = json.loads((path / 'exact-prefix.json').read_text())
    result['validity'] = 'valid_recorded_endpoint_current_analysis'
    result['hypothesis_verdict'] = ('supported_in_observed_samples' if all(
        row['rate']['max'] == 0 and row['net_threshold_margin']['max'] <= 0
        for row in result['targets'][:2]) else 'inconclusive')
    result['adoption'] = 'not_applicable'
    result['milestone_outcome'] = 'pending'
    write_json(path / 'recruitment-analysis.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'silent_muscle_pools'}, indent=2))
    print('Silent pools:', [(row['index'], row['leg'], row['muscle']) for row in result['silent_muscle_pools']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('path', 'graph', 'calibration'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.path, args.graph, args.calibration)
