"""Transfer a frozen relay operating model to six source-defined homologues."""

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
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.probe_c_banc_m3_core_teacher import CoreWaveTeacher
from tools.probe_c_banc_m3_premotor_relay import harmonic_signal
from tools.verify_c_banc_observation import exact


def homologous_patch(graph, matrix, core, frozen, provenance):
    effective_e1 = frozen['values'][frozen['edge_groups'] == 0]
    if effective_e1.shape != (1,) or effective_e1[0] <= 0:
        raise ValueError('Exactly one frozen positive source E1 weight required')
    if len(provenance['rows']) != 6 or provenance['graph_hash'] != graph.hash:
        raise ValueError('Six source homologues required')
    edge_ids, source, values, relays, e1s = [], [], [], [], []
    for row in provenance['rows']:
        relay, e1 = row['relay_id'], row['E1_id']
        if (graph.nodes[relay]['root_id'] != row['relay_root']
                or graph.nodes[relay]['cell_type'] != 'IN13A001'
                or graph.nodes[e1]['cell_type'] != 'IN17A001'
                or float(matrix[relay, e1]) != row['E1_source_weight']):
            raise ValueError('Source homologue identity or incoming E1 weight changed')
        ids = np.arange(matrix.indptr[relay], matrix.indptr[relay + 1])
        ids = ids[matrix.data[ids] != 0]
        pre = matrix.indices[ids]
        group = np.where(pre == e1, 0, np.where(np.isin(pre, core), 1, 2))
        factor = frozen['efficacy'][group].copy()
        factor[group == 0] = effective_e1[0] / matrix.data[ids[group == 0]]
        if np.any((factor < .001 - 1e-8) | (factor > 10)):
            raise ValueError('Frozen positive source-efficacy limits exceeded')
        changed = (matrix.data[ids] * factor).astype(np.float32)
        # Assignment, rather than round-trip division/multiplication, preserves
        # the frozen effective E1 float32 value exactly across homologues.
        changed[group == 0] = effective_e1[0]
        edge_ids.extend(ids); source.extend(matrix.data[ids]); values.extend(changed)
        relays.append(relay); e1s.append(e1)
    if len(set(relays)) != 6 or len(set(e1s)) != 6:
        raise ValueError('Six distinct source core/relay pairs required')
    return (np.array(edge_ids, dtype=np.int64), np.array(source, dtype=np.float32),
            np.array(values, dtype=np.float32), np.array(relays), np.array(e1s))


def run(calibration, teacher_source, baseline_source, initial, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E41-six-source-homologue-transfer',
        'question': 'Does the frozen E40 relay operating model transfer to all six source IN13A001 homologues and provide a usable inhibitory phase signal for complementary muscle recruitment?',
        'intervention': 'For each source homologue, set its one existing E1 edge to the exact E40 fitted effective weight. Multiply othercore and remaining incoming edges by the same frozen E40 factors; retain all source edge signs and topology. Only sixhomologue adaptation gains become0, originalcore gain0 retained. No refit/new waveform/bounds search.',
        'comparison': 'Same exposed E37 full neural state .445s, original held sensory/tonic drive and same24core teaching table until2.2s. All rate/adaptation/mask/drive states are retained, including decaying previous adaptation. All391motor cells unforced. Compare same-window E39b original rates where recorded.',
        'screen': 'All six relay rates on[1.5,2.2) must have first-harmonic explainedfraction>=.5,p95-p05>=20,and circular lag from their own E1<=.15cycles. Also report [.7,2.2) without erasing its transients. This is a forced-signal transmission screen, not autonomousoscillation/gait.',
        'next_decision': 'If all six pass, calibrate existing core/relay-to-muscle recruitment with source support and actual-body validation. If any fail, preserve the same-parameter transfer failure; do not call sixhomologues interchangeable or repeat a gain grid.',
        'limits': 'OneGPU,1.755neural continuation+.01neutral diskfuture check; .01physical checking only;<=180wallseconds. F1-F8pending/F6unexposed/liveM2UIpreserved.',
        'tool_sha256': sha256(__file__), 'calibration_sha256': sha256(calibration / 'calibration.npz'),
        'provenance_sha256': sha256(calibration / 'homologous-provenance.json'),
        'teaching_wave_sha256': sha256(teacher_source / 'teaching-table.npz')}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    teaching = StateStore.load(teacher_source / 'teaching-state')
    start = StateStore.load(initial / 'start')
    frozen = dict(np.load(calibration / 'calibration.npz'))
    provenance = json.loads((calibration / 'homologous-provenance.json').read_text())
    baseline = dict(np.load(baseline_source / 'observations.npz'))
    core, motor = teaching['forced_core_ids'], baseline['motor_ids']
    matrix, _ = rate_weights(graph)
    matrix = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(matrix, motor)
    edge_ids, original, changed, relays, e1s = homologous_patch(graph, matrix, core, frozen, provenance)
    np.savez_compressed(out / 'calibration.npz', edge_ids=edge_ids, source_values=original,
        values=changed, zero_adaptation_ids=relays, E1_ids=e1s)
    write_json(out / 'provenance.json', provenance)
    s = EfficacyTrial(graph, start, edge_ids, original, changed, zero_adaptation_ids=relays)
    records = []
    try:
        StateStore.save(out / 'candidate-neutral', s.snapshot())
        clone = EfficacyTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        try:
            clone.advance(1); expected = clone.snapshot()
        finally:
            clone.close()
        s.advance(1)
        exact(expected, s.snapshot(), 'Six-homologue neutral disk future')
        n = s.network
        saved = copy.deepcopy(teaching['whole_neural'])
        saved['model_hash'] = n.identity
        saved['core_adaptation'] = n.snapshot()['core_adaptation']
        n.restore(saved)
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(teaching['whole_neural'][key], n.snapshot()[key], 'Preserved original ' + key)
        candidate_identity = n.identity
        n.identity = hashlib.sha256((n.identity + spec['experiment'] + 'external-teacher').encode()).hexdigest()
        wave = dict(np.load(teacher_source / 'teaching-table.npz'))
        teacher = CoreWaveTeacher(n, wave['ids'], wave['rate'])
        teacher.capture(); teacher.cursor.fill_(int(teaching['teacher_half_step']))
        observed = np.union1d(baseline['observed_ids'], np.r_[core, relays])
        fixed_drive = saved['drive'].copy()
        def observe():
            current = n._current(n.rate)
            records.append({'time_s': n.tick * n.dt, 'rate': n.readout(observed),
                'relay_adaptation': n.adaptation[relays].cpu().numpy().copy(),
                'relay_current': current[relays].cpu().numpy().copy(),
                'motor_current': current[motor].cpu().numpy().copy()})
        observe()
        with n.torch.inference_mode():
            for _ in range(351):
                if time.perf_counter() - began > 175:
                    raise TimeoutError('Six-homologue trial wall limit')
                n.advance_readout(fixed_drive, observed, steps=20); observe()
                np.testing.assert_array_equal(n.readout(core), wave['rate'][int(teacher.cursor)])
                if int(teacher.cursor) != n.tick * 2 or not np.isfinite(records[-1]['rate']).all():
                    raise ValueError('Finite-state/native-clock check failed')
        StateStore.save(out / 'final-teaching-state', {'schema': 'flylab.training.six-relays.v1',
            'autonomous_controller': False, 'whole_neural': n.snapshot(),
            'unforced_candidate_identity': candidate_identity})
    finally:
        if records:
            np.savez_compressed(out / 'observations.npz',
                **{k: np.array([r[k] for r in records]) for k in records[0]},
                observed_ids=observed, motor_ids=motor, relay_ids=relays, E1_ids=e1s)
        s.close()
    times = np.array([r['time_s'] for r in records])
    np.testing.assert_array_equal(times, baseline['time_s'])
    rate = np.array([r['rate'] for r in records])
    results = {}
    for name, window in (('late', (1.5, 2.2)), ('including_adaptation_transient', (.7, 2.2))):
        active = (times >= window[0]) & (times < window[1])
        rows = []
        for relay, e1, source in zip(relays, e1s, provenance['rows']):
            e = harmonic_signal(times[active], rate[active, np.searchsorted(observed, e1)])
            arms = {}
            for arm, values in (('candidate', rate[:, np.searchsorted(observed, relay)]),
                                ('baseline', baseline['presynaptic_rates'][:, np.searchsorted(baseline['observed_ids'], relay)])):
                h = harmonic_signal(times[active], values[active])
                lag = float((h['phase_cycle'] - e['phase_cycle'] + .5) % 1 - .5)
                width = float(np.ptp(np.percentile(values[active], [5, 95])))
                explained = float(h['explained_fraction'])
                arms[arm] = {'explained_fraction': explained, 'rate_p95_minus_p05': width,
                    'phase_lag_cycles_vs_E1': lag, 'screen_pass': explained >= .5 and width >= 20 and abs(lag) <= .15}
            rows.append({'leg': source['source_direct_leg_identity'], 'relay_root': source['relay_root'], **arms})
        results[name] = rows
    report = {'experiment': spec['experiment'], 'validity': 'valid_forced_core_homologue_transfer',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'changed_edges': len(edge_ids),
        'disk_future_exact': True, 'late_all_six_screen': all(r['candidate']['screen_pass'] for r in results['late']),
        'results': results, 'additional_neural_model_s': 1.765, 'additional_physical_model_s': .01,
        'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('calibration', 'teacher-source', 'baseline-source', 'initial', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.calibration, args.teacher_source, args.baseline_source, args.initial, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
