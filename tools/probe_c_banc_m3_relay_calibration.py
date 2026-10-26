"""One source-defined premotor relay efficacy/physiology transmission trial."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.probe_c_banc_m3_core_teacher import CoreWaveTeacher
from tools.probe_c_banc_m3_premotor_relay import harmonic_signal
from tools.verify_c_banc_observation import exact


def run(source, teacher_source, initial, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E40-single-relay-transmission-calibration',
        'hypothesis': 'The source RF E1->IN13A001 inhibitory premotor path can transmit a complementary signal when its existing input efficacies and a stationary nonadapting relay operating point are calibrated.',
        'source_relay': '720575941629416428', 'source_E1': '720575941554038555',
        'intervention': 'Only all existing nonzero incoming edges of the predesignated IN13A001 are multiplied by three positive efficacies: sourceRF E1, other24core, and all remaining sources. Its adaptation gain8 becomes0, a source-author nonadapting rate-model hypothesis; originaltau/a/threshold/cap and all otherneurons/edges stay unchanged. No direct motor intervention.',
        'fit': 'Use E39b [.7,1.5) actual currents. Fit cap200 inverse-half-tanh current for desired stationary relay target activation equal to actual E1 rate: 7.5+200atanh(E1/200). Positive efficacy bounds[.001,10], current/200 least squares plus1e-4 mean-square deviation from1. Three-parameter convex fit; no bound/grid search. This fit excludes adaptation because its candidate gain0 has stationaryz0.',
        'comparison': 'Continue same E37 .445s r/z/drive/mask/tick with the SAME24core wave and held final sensorydrive to2.2s. Preserve the old relay adaptation state at the change: it decays naturally withtau.3, never reset. Candidate parameters change once before stepping. Other158681noncore/unselectedcells including391MN are unforced. Compare late [1.5,2.2) to E39b for the same interval, report earlier [.7,2.2) separately.',
        'decision': 'For this source relay only, require first-harmonic explainedfraction>=.5, ratep95-p05>=20, E1phase-lag distance<=.15. Passing permits a subsequent source-provenance joint recruitment study; it is not gait or autonomousoscillation. Failure preserves current target/counterevidence and does not trigger a gain grid. ActualMN currents/rates remain measured, not trained targets.',
        'validity': 'SourceCSR/sign/unselectedvalues preserved; original rate/adaptation/drives not reset; actualCUDA. Candidate neutral checkpoint disk clone next5ms full neural/body future exact before diagnostic teacher is attached; both short states are identification checks only.',
        'limits': 'OneGPU,1.755neural continuation +2x.005full check =1.765neural/ .01physical modelsec,<=180wallsec. Corewaveexternal diagnostic only; noF6/adoption/UI changes.',
        'tool_sha256': sha256(__file__), 'efficacy_tool_sha256': sha256(Path('tools/banc_m3_synaptic_efficacy.py')),
        'source_observations_sha256': sha256(source / 'observations.npz'),
        'teacher_wave_sha256': sha256(teacher_source / 'teaching-table.npz')}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'efficacy-model.py').write_bytes(Path('tools/banc_m3_synaptic_efficacy.py').read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    metadata = json.loads((source / 'provenance.json').read_text())
    data = dict(np.load(source / 'observations.npz'))
    teaching = StateStore.load(teacher_source / 'teaching-state')
    start = StateStore.load(initial / 'start')
    relay = graph.resolve(['flywire:banc:888:' + spec['source_relay']])[0]
    e1 = graph.resolve(['flywire:banc:888:' + spec['source_E1']])[0]
    core = teaching['forced_core_ids']
    motor = data['motor_ids']
    matrix, _ = rate_weights(graph)
    matrix = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(matrix, motor)
    if graph.hash != metadata['graph_hash'] or not np.isin(relay, data['relay_ids']):
        raise ValueError('Predesignated relay/source identity changed')
    position = int(np.flatnonzero(data['relay_ids'] == relay)[0])
    e1rate = data['presynaptic_rates'][:, np.searchsorted(data['observed_ids'], e1)]
    relay_record = data['relay'][:, position]
    direct = float(matrix[relay, e1]) * e1rate.astype(float)
    x = np.column_stack((direct, relay_record[:, 3] - direct, relay_record[:, 2] - relay_record[:, 3]))
    target = 7.5 + 200 * np.arctanh(e1rate.astype(float) / 200)
    train = (data['time_s'] >= .7) & (data['time_s'] < 1.5)
    valid = (data['time_s'] >= 1.5) & (data['time_s'] < 2.2)
    # Match mean-normalized data and mean-normalized three-parameter penalty.
    a = np.vstack((x[train] / (200 * np.sqrt(train.sum())), np.sqrt(1e-4 / 3) * np.eye(3)))
    b = np.r_[target[train] / (200 * np.sqrt(train.sum())), np.sqrt(1e-4 / 3) * np.ones(3)]
    fit = lsq_linear(a, b, bounds=(.001, 10), tol=1e-12, max_iter=100)
    edge_ids = np.arange(matrix.indptr[relay], matrix.indptr[relay + 1])
    edge_ids = edge_ids[matrix.data[edge_ids] != 0]
    pre = matrix.indices[edge_ids]
    groups = np.where(pre == e1, 0, np.where(np.isin(pre, core), 1, 2))
    values = (matrix.data[edge_ids] * fit.x[groups]).astype(np.float32)
    np.savez_compressed(out / 'calibration.npz', edge_ids=edge_ids,
        source_values=matrix.data[edge_ids], values=values, efficacy=fit.x,
        edge_groups=groups, source_current_design=x, target_current=target,
        conditional_fitted_current=x @ fit.x, time_s=data['time_s'])
    report = {'success': bool(fit.success), 'status': int(fit.status), 'message': str(fit.message),
        'optimality': float(fit.optimality), 'efficacy': fit.x.tolist(),
        'incoming_edges': len(edge_ids), 'group_edge_counts': np.bincount(groups, minlength=3).tolist(),
        'train_current_rms': float(np.sqrt(np.mean((x[train] @ fit.x - target[train]) ** 2))),
        'development_current_rms': float(np.sqrt(np.mean((x[valid] @ fit.x - target[valid]) ** 2))),
        'physiology_change': 'Only designated relay gain8->0; existing oldz state retained'}
    write_json(out / 'fit-result.json', report)
    if not fit.success:
        raise ValueError('Three-parameter fit did not complete')
    s = EfficacyTrial(graph, start, edge_ids, matrix.data[edge_ids], values, zero_adaptation_ids=np.array([relay]))
    records = []
    try:
        StateStore.save(out / 'candidate-neutral', s.snapshot())
        clone = EfficacyTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        try:
            clone.advance(1)
            expected = clone.snapshot()
        finally:
            clone.close()
        s.advance(1)
        exact(expected, s.snapshot(), 'New source efficacy candidate disk future')
        n = s.network
        source_state = copy.deepcopy(teaching['whole_neural'])
        source_state['model_hash'] = n.identity
        source_state['core_adaptation'] = n.snapshot()['core_adaptation']
        n.restore(source_state)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(teaching['whole_neural'][key], n.snapshot()[key], 'Preserved E37 initial ' + key)
        candidate_identity = n.identity
        n.identity = hashlib.sha256((candidate_identity + spec['experiment'] + 'diagnostic-core-wave').encode()).hexdigest()
        wave = np.load(teacher_source / 'teaching-table.npz')
        teacher = CoreWaveTeacher(n, wave['ids'], wave['rate'])
        teacher.capture()
        teacher.cursor.fill_(int(teaching['teacher_half_step']))
        observed = np.union1d(motor, np.r_[core, relay])
        fixed_drive = source_state['drive'].copy()
        def observe():
            rates = n.readout(observed)
            currents = n._current(n.rate)
            records.append({'time_s': n.tick * n.dt, 'rate': rates,
                'relay_adaptation': float(n.adaptation[relay]),
                'relay_current': float(currents[relay]),
                'motor_current': currents[motor].cpu().numpy().copy()})
        observe()
        with n.torch.inference_mode():
            for _ in range(351):
                if time.perf_counter() - began > 175:
                    raise TimeoutError('Single relay trial wall limit')
                n.advance_readout(fixed_drive, observed, steps=20)
                observe()
                np.testing.assert_array_equal(n.readout(wave['ids']), wave['rate'][int(teacher.cursor)])
                if int(teacher.cursor) != n.tick * 2 or not np.isfinite(records[-1]['rate']).all():
                    raise ValueError('Native clock or finite neural state check failed')
        StateStore.save(out / 'final-teaching-state', {'schema': 'flylab.training.calibrated-relay.v1',
            'autonomous_controller': False, 'whole_neural': n.snapshot(),
            'unforced_candidate_identity': candidate_identity})
    finally:
        if records:
            np.savez_compressed(out / 'observations.npz',
                **{k: np.array([r[k] for r in records]) for k in records[0]},
                observed_ids=observed, motor_ids=motor)
        s.close()
    result = {'experiment': spec['experiment'], 'validity': 'valid_forced_core_relay_calibration',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'fit': report,
        'disk_future_exact': True, 'additional_neural_model_s': 1.765,
        'additional_physical_model_s': .01, 'wall_s': time.perf_counter() - began}
    times = np.array([r['time_s'] for r in records])
    rates = np.array([r['rate'] for r in records])
    for name, window in (('late', (1.5, 2.2)), ('including_adaptation_transient', (.7, 2.2))):
        active = (times >= window[0]) & (times < window[1])
        arms = {}
        for label, rr in (('baseline', data['relay'][:, position, 0]), ('candidate', rates[:, np.searchsorted(observed, relay)])):
            h = harmonic_signal(times[active], rr[active])
            eh = harmonic_signal(times[active], e1rate[active])
            lag = float((h['phase_cycle'] - eh['phase_cycle'] + .5) % 1 - .5)
            width = float(np.percentile(rr[active], 95) - np.percentile(rr[active], 5))
            explained = float(h['explained_fraction'])
            arms[label] = {'explained_fraction': explained, 'rate_p95_minus_p05': width,
                'phase_lag_cycles_vs_E1': lag, 'screen_pass': explained >= .5 and width >= 20 and abs(lag) <= .15}
        result[name] = arms
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'teacher-source', 'initial', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.source, args.teacher_source, args.initial, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
