"""Observe existing inhibitory relay transfer under an explicit core teacher.

This extends an exposed neural state with its sensory drive held constant.
There is no additional body trajectory and no claim of autonomous walking.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_core_rhythm import CoreTrial, source_matrix
from tools.probe_c_banc_m3_core_teacher import CoreWaveTeacher
from tools.verify_c_banc_observation import exact


def harmonic_signal(times, values, period=.15):
    values = np.asarray(values, dtype=float)
    angle = 2 * np.pi * np.asarray(times) / period
    design = np.column_stack((np.ones(len(times)), np.cos(angle), np.sin(angle)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    predicted = design @ coefficients
    variance = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    r2 = np.zeros_like(variance)
    np.divide(np.sum((predicted - values.mean(axis=0)) ** 2, axis=0),
              variance, out=r2, where=variance > 1e-16)
    return {'explained_fraction': r2, 'peak_to_peak_first_harmonic': 2 * np.hypot(coefficients[1], coefficients[2]),
            'phase_cycle': np.arctan2(coefficients[2], coefficients[1]) / (2 * np.pi)}


def run(source, initial, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E39-existing-premotor-relay-transfer',
        'question': 'Do existing E1-to-inhibitory-premotor-to-RF tibia-flexor pathways convey a repeatable E1-related signal, and is their operating point responsive under the exposed core teacher?',
        'intervention': 'Continue the exact E37 saved neural state at .445s to2.2s. Hold its last complete sensory/tonic drive constant. Continue the SAME frozen24core wave at every nativeRKstage; remaining158682cells includingall391MN unforced. OriginalW/cellparameters retained. Body never advances; no body root correction.',
        'record': 'All direct presynaptic rates of391legMN at5ms, actual MN currents, and r/z/current/core-current of every source inhibitory VNC relay on an RF E1->relay->RF tibiaflexor path. Source edge memberships/weights and native core endpoints preserved.',
        'analysis': 'On [.7,2.2), fit intercept+sin/cos at the prescribed.15s period to relay rates; report explained fraction, harmonic amplitude, lag relative actual E1 and threshold/relative-slope occupation. These measure response to a known external teacher, not endogenous oscillations or causal contribution percentages.',
        'decision': 'A relay with explained_fraction>=.5, ratep95-p05>=20 and circular phase lag within.15cycles of E1 is a candidate for a subsequent source-efficacy study; no automatic selection/adoption. If no relays meet this descriptive screen, change relay input/physiology assumptions before fitting motor output weights. Do not assume static two-edge paths are functional complementary transmission.',
        'limits': 'OneGPU,1.755additional neural seconds,zero additional physicalseconds,<=120wallseconds; no finalF6exposure; F1-F8pending',
        'tool_sha256': sha256(__file__), 'source_observations_sha256': sha256(source / 'observations.npz'),
        'source_teaching_table_sha256': sha256(source / 'teaching-table.npz')}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    teaching = StateStore.load(source / 'teaching-state')
    session = CoreTrial.from_checkpoint(graph, StateStore.load(initial / 'start'))
    records = []
    try:
        n = session.network
        if n.identity != teaching['source_model_hash'] or n.device != 'cuda':
            raise ValueError('Original CUDA source core model required')
        saved = copy.deepcopy(teaching['whole_neural'])
        saved['model_hash'] = n.identity
        n.restore(saved)
        exact(n.snapshot(), saved, 'Exact E37 full neural starting state')
        matrix = source_matrix(n)
        motor = session.adapter.motor_ids
        incoming = matrix[motor].copy()
        incoming.eliminate_zeros()
        observed_ids = np.union1d(np.unique(incoming.indices), motor)
        rf_e1 = graph.resolve(['flywire:banc:888:720575941554038555'])[0]
        rf_flexor = np.array([i for i in motor if graph.nodes[i]['soma_side'] == 'right'
            and graph.nodes[i]['source_annotations']['body_part_effector'] == 'front_leg'
            and graph.nodes[i]['source_annotations']['peripheral_target_type'] == 'tibia_flexor_muscle'])
        exc_from_e1 = matrix[:, rf_e1].toarray().ravel() > 0
        inhibit_rf = (matrix[rf_flexor].toarray() < 0).any(axis=0)
        relay_ids = np.array([i for i in np.flatnonzero(exc_from_e1 & inhibit_rf)
            if graph.nodes[i]['super_class'] == 'ventral_nerve_cord_intrinsic'
            and i not in teaching['forced_core_ids']])
        observed_ids = np.union1d(observed_ids, np.r_[rf_e1, relay_ids])
        if len(rf_flexor) != 5 or len(relay_ids) != 53:
            raise ValueError('Expected independently audited RF source pathways')
        table = np.load(source / 'teaching-table.npz')
        ids, wave = table['ids'], table['rate']
        observed_ids = np.union1d(observed_ids, ids)
        cursor = int(teaching['teacher_half_step'])
        if cursor != n.tick * 2 or n.tick != 1780:
            raise ValueError('Exposed .445s source clock differs')
        np.testing.assert_array_equal(saved['rate'][ids], wave[cursor])
        n.identity = hashlib.sha256((n.identity + spec['experiment'] + sha256(out / 'spec.json')).encode()).hexdigest()
        teacher = CoreWaveTeacher(n, ids, wave)
        teacher.capture()
        teacher.cursor.fill_(cursor)
        fixed_drive = saved['drive'].copy()
        relay_core = matrix[relay_ids][:, ids].toarray().astype(float)
        paths = []
        for relay in relay_ids:
            for post in rf_flexor:
                if matrix[post, relay] < 0:
                    paths.append({'pre': graph.nodes[rf_e1]['root_id'],
                        'relay': graph.nodes[relay]['root_id'], 'post': graph.nodes[post]['root_id'],
                        'first_weight': float(matrix[relay, rf_e1]), 'second_weight': float(matrix[post, relay])})
        write_json(out / 'provenance.json', {'graph_hash': graph.hash,
            'source_model_hash': teaching['source_model_hash'], 'relay_ids': relay_ids.tolist(),
            'relay_nodes': [graph.nodes[i] for i in relay_ids], 'paths': paths,
            'rf_e1': int(rf_e1), 'rf_flexor': rf_flexor.tolist(),
            'relay_current_columns': ['rate', 'adaptation', 'CUDA_current', 'direct_core_current', 'fixed_drive']})
        def observe():
            values = n.readout(observed_ids)
            core = values[np.searchsorted(observed_ids, ids)]
            currents = n._current(n.rate)
            relay = values[np.searchsorted(observed_ids, relay_ids)]
            z = n.adaptation[relay_ids].cpu().numpy().copy()
            j = currents[relay_ids].cpu().numpy().copy()
            records.append({'time_s': n.tick * n.dt, 'presynaptic_rates': values,
                'motor_current': currents[motor].cpu().numpy().copy(),
                'relay': np.stack((relay, z, j, relay_core @ core, fixed_drive[relay_ids]), axis=1)})
        observe()
        with n.torch.inference_mode():
            for _ in range(351):
                if time.perf_counter() - began > 115:
                    raise TimeoutError('Relay diagnostic wall limit')
                n.advance_readout(fixed_drive, observed_ids, steps=20)
                observe()
                np.testing.assert_array_equal(n.readout(ids), wave[int(teacher.cursor)])
                if int(teacher.cursor) != 2 * n.tick:
                    raise ValueError('Native core clock mismatch')
                if not np.isfinite(records[-1]['presynaptic_rates']).all():
                    raise ValueError('Nonfinite neural output')
        StateStore.save(out / 'final-teaching-state', {'schema': 'flylab.training.premotor-relay.v1',
            'autonomous_controller': False, 'whole_neural': n.snapshot(),
            'source_model_hash': teaching['source_model_hash'], 'teacher_half_step': int(teacher.cursor)})
    finally:
        if records:
            np.savez_compressed(out / 'observations.npz',
                **{k: np.array([r[k] for r in records]) for k in records[0]},
                observed_ids=observed_ids, motor_ids=motor, relay_ids=relay_ids,
                incoming_indptr=incoming.indptr, incoming_indices=incoming.indices, incoming_data=incoming.data)
        session.close()
    times = np.array([r['time_s'] for r in records])
    active = (times >= .7) & (times < 2.2)
    rate = np.array([r['presynaptic_rates'] for r in records])
    relay = np.array([r['relay'] for r in records])
    rel = harmonic_signal(times[active], relay[active, :, 0])
    e1 = harmonic_signal(times[active], rate[active, np.searchsorted(observed_ids, rf_e1)])
    lag = (rel['phase_cycle'] - e1['phase_cycle'] + .5) % 1 - .5
    span = np.percentile(relay[active, :, 0], 95, axis=0) - np.percentile(relay[active, :, 0], 5, axis=0)
    margin = relay[active, :, 2] + relay[active, :, 4] - 7.5 - relay[active, :, 1]
    slope = np.where(margin > 0, 1 - np.tanh(margin / 200) ** 2, 0)
    screen = (rel['explained_fraction'] >= .5) & (span >= 20) & (abs(lag) <= .15)
    rows = [{'root_id': graph.nodes[i]['root_id'], 'cell_type': graph.nodes[i]['cell_type'],
        'harmonic_explained_fraction': float(rel['explained_fraction'][k]),
        'first_harmonic_peak_to_peak': float(rel['peak_to_peak_first_harmonic'][k]),
        'rate_p95_minus_p05': float(span[k]), 'phase_lag_cycles_vs_E1': float(lag[k]),
        'rectified_fraction': float(np.mean(margin[:, k] <= 0)),
        'half_responsive_fraction': float(np.mean(slope[:, k] >= .5)),
        'descriptive_transfer_screen': bool(screen[k])} for k, i in enumerate(relay_ids)]
    result = {'experiment': spec['experiment'], 'validity': 'valid_forced_core_fixed_drive_neural_diagnostic',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'path_count': len(paths),
        'relay_count': len(relay_ids), 'transfer_screen_count': int(screen.sum()),
        'relay_results': rows, 'additional_neural_model_s': (len(records) - 1) * .005,
        'additional_physical_model_s': 0, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'relay_results'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.source, args.initial, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
