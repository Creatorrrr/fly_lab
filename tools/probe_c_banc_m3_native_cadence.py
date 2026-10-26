"""Measure a constant native source-timescale change in the whole BANC network."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.probe_c_banc_m3_pair_coupling import phase_summary
from tools.verify_c_banc_observation import exact


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e83-existing-premotor-source-basis'
    observed_path = root / 'e84-native-premotor-basis-body/recruitment-signals.npz'
    reference_path = root / 'e85-space-compatible-cadence-reference/commands.npz'
    cal = dict(np.load(source / 'calibration.npz'))
    old_cal = {key: value.copy() for key, value in cal.items()}
    previous = np.load(observed_path)
    lookup = {int(value): i for i, value in enumerate(previous['observed_ids'])}
    e1 = np.array([cal['core_ids'][(cal['core_groups'] == group)
        & (cal['core_cell_types'] == 0)][0] for group in range(6)])
    periods = []
    for cell in e1:
        peaks, _ = find_peaks(previous['rate'][:, lookup[int(cell)]], prominence=20)
        ts = previous['time_s'][peaks]
        ts = ts[(ts >= 3.5) & (ts < 5)]
        if len(ts) < 7:
            raise ValueError('At least six already observed source cycles required')
        periods.append(float(np.median(np.diff(ts))))
    baseline_period = float(np.median(periods))
    target_period = float(np.load(reference_path)['period_s'])
    factor = target_period / baseline_period
    spec = {
        'experiment': 'M3-E86-native-source-cadence',
        'hypothesis': 'Scaling the existing nonadapting24core and6coupling relay taus together can retain autonomous source rhythms near the slower E85 feasible physical cadence.',
        'basis': 'E85 originalworld external reference10s passed numericF3-F5 at the slower derived cadence. E84 provided actual47source inputs but still used the fast E72motor interface. This test changes source kinetics, not runtime timing or world clearance.',
        'intervention': 'E83 exact weights, moment/rest, source identities, neural/body zero-state and physiology retained. Multiply only24core and6coupling relay taus by one observed/reference period ratio. Preserve2LM transfer taus and all other neurons. No injected core drive, waveform, phase variable, parameter grid or initial prewarming.',
        'comparator': 'Six E84 source periods from complete peaks in already exposed[3.5,5), plus E85 physical reference period. Moving-body feedback in E84 differs from this held neutral-sensory assay and is not an identical physiological comparator.',
        'competing_explanation': 'Uniform kinetics only preserves the isolated nonadapting30cell dynamics under equal inputs. Unscaled tonic startup, residual full-network feedback and physical feedback can change period, convergence and motor transfer.',
        'measurement': 'One full158706 nativeCUDA10s assay with exact original neutral body held, native250us neural integration,5ms observed source/MN rates, recurrent currents, drive, adaptation. Decode muscle activation locally for measurement only without changing adapter or applying any command to the body.',
        'screen': 'All six E1 in[5,10): width>=20,>=8complete prominence20cycles,CV<=.3,meanperiod within10percent of E85period. Report all3 LR peak phase distancefrom.5<=.15/R>=.8 separately. Source screening is not F3-F6 or biological verification.',
        'validity': 'Exactly30distinct nonmotor/non-tonic gain0 cells changedtau; original26patch plus6newcoupling entries=32, twoLMvalues exact; all other calibration arrays and whole original neutral state exact. Full neutral disk exact and held-neural-only .1s diskfuture exact. Body and its control/physics clocks do not advance.',
        'decision': 'Preserve all actual source/current data. Supported rhythm permits source-constrained joint all-leg calibration on recorded native signals. Coordination failure remains a concern to test and does not become a final gait pass. Failure does not authorize a tau grid or easier final gates.',
        'limits': 'OneCUDA job,10main+.1clone neural seconds,0physical advance,<=180wallseconds. One exposed neutral developmental state; F6unused. Preserve M2/UI/world and all F1-F8.',
        'baseline_periods_s': periods, 'baseline_period_s': baseline_period,
        'target_period_s': target_period, 'timescale_factor': factor,
        'source_sha256': sha256(source / 'calibration.npz'),
        'observed_sha256': sha256(observed_path),
        'reference_sha256': sha256(reference_path), 'tool_sha256': sha256(__file__),
    }
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    core, coupling = cal['core_ids'], cal['coupling_ids']
    changed_ids = np.r_[core, coupling]
    if (len(changed_ids) != 30 or len(np.unique(changed_ids)) != 30
            or np.intersect1d(changed_ids, np.r_[cal['motor_ids'], cal['intrinsic_tonic_ids']]).size
            or np.intersect1d(coupling, cal['tau_ids']).size):
        raise ValueError('Exactly30 distinct source cells and six new tau entries required')
    mask = np.isin(cal['tau_ids'], core)
    if (mask.sum() != 24 or len(core) != 24 or len(coupling) != 6
            or len(cal['tau_ids']) != 26 or len(np.unique(cal['tau_ids'])) != 26):
        raise ValueError('Original 24 core, 6 coupling and 2 LM kinetics required')
    tau_values = cal['tau_values'].copy()
    tau_values[mask] *= factor
    cal['tau_ids'] = np.r_[cal['tau_ids'], coupling]
    cal['tau_values'] = np.r_[tau_values, np.full(6, .02 * factor)]
    if len(cal['tau_ids']) != 32:
        raise ValueError('24core+6coupling+2unchanged LM tau entries required')
    np.testing.assert_allclose(old_cal['tau_values'][mask], .04659108557229384,
        rtol=1e-6, atol=0)
    np.testing.assert_allclose(old_cal['tau_values'][~mask], .014046887330582595,
        rtol=1e-6, atol=0)
    np.testing.assert_allclose(cal['tau_values'][-6:], .02 * factor,
        rtol=1e-6, atol=0)
    for key, value in old_cal.items():
        if key not in ('tau_ids', 'tau_values'):
            np.testing.assert_array_equal(value, cal[key])
    np.testing.assert_array_equal(tau_values[~mask], old_cal['tau_values'][~mask])
    np.savez_compressed(out / 'calibration.npz', **cal)
    anatomy = json.loads((source / 'muscle-provenance.json').read_text())
    anatomy['cadence_hypothesis'] = {
        'cell_ids': changed_ids.tolist(), 'factor': factor,
        'old_period_s': baseline_period, 'target_period_s': target_period,
        'validation': 'Engineered constant native neuronal kinetics; physiological values not measured.'}
    write_json(out / 'muscle-provenance.json', anatomy)
    graph = GraphStore.load(graph_path)
    initial = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base = EfficacyTrial(graph, initial, cal['edge_ids'], cal['source_values'], cal['values'],
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base, cal, anatomy['muscles']),
        cal['tau_ids'], cal['tau_values'])
    sources = np.unique(np.r_[cal['input_ids'], cal['intrinsic_tonic_ids'], coupling])
    dns = np.array(graph.resolve(['flywire:banc:888:720575941500851362',
        'flywire:banc:888:720575941626500746']))
    observed = np.r_[sources, dns, cal['motor_ids']]
    if len(sources) != 47 or len(np.unique(observed)) != len(observed):
        session.close()
        raise ValueError('47distinct source+2DN+391motor observations required')
    rows, future = [], None
    future_exact = False
    neural_s = 0.
    try:
        n = session.network
        if n.device != 'cuda' or len(n.rate) != 158706:
            raise ValueError('Actual whole BANC CUDA network required')
        np.testing.assert_array_equal(n._cell_gain[changed_ids].cpu().numpy(), np.zeros(30))
        np.testing.assert_allclose(n.tau[core].cpu().numpy(),
            old_cal['tau_values'][mask] * factor, rtol=1e-6, atol=0)
        np.testing.assert_allclose(n.tau[coupling].cpu().numpy(), .02 * factor,
            rtol=1e-6, atol=0)
        neutral = session.snapshot()
        exact(initial['body'], neutral['body'], 'Original cadence neutralbody')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(initial['network'][key], neutral['network'][key], 'Original neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out / 'candidate-neutral'))
        try:
            exact(neutral, clone.snapshot(), 'Full cadence neutral disk state')
        finally:
            clone.close()
        weights = source_matrix(n)
        np.savez_compressed(out / 'recruitment-metadata.npz', observed_ids=observed,
            source_ids=sources, motor_ids=cal['motor_ids'],
            source_to_motor=weights[cal['motor_ids']][:, sources].toarray(),
            **{key: getattr(n, key).cpu().numpy()[observed].copy()
               for key in ('tau', 'a', 'threshold', 'cap', '_cell_gain')})
        position, rotation, _ = session.body.pose()
        drive = session.adapter.encode(sensory_cut=session.cuts['sensory'])
        drive[session.descending] += session.p.descending_drive
        drive += session.navigation.drive(position, rotation)
        if np.any(drive[sources] != 0) or np.any(drive[dns] != 5000):
            raise ValueError('Existing forward DNdrive and no injected source current required')
        np.savez_compressed(out / 'held-input.npz', drive=drive)
        body_saved = session.body.snapshot()
        activation = session.adapter.activation.copy()
        alpha = -np.expm1(-.005 / .01)
        for control in range(2001):
            if time.perf_counter() - began > 175:
                raise TimeoutError('E86 bounded CUDA neural assay exceeded wall allowance')
            if control:
                n.advance_readout(drive, observed, 20)
                neural_s += .005
            rate = n.readout(observed)
            if control:
                activation += alpha * (cal['pool'] @ rate[-391:] - activation)
            current = n._current(n.rate)
            rows.append({'rate': rate, 'recurrent_current': current[observed].cpu().numpy().copy(),
                'drive': n.drive[observed].cpu().numpy().copy(),
                'adaptation': n.adaptation[observed].cpu().numpy().copy(),
                'muscle_activation': activation.copy(),
                'decoded_command': cal['rest'] + .02 * cal['moment'] @ activation})
            if n.tick != 20 * control or any(not np.isfinite(v).all() for v in rows[-1].values()):
                raise ValueError('Finite observed native state and exact tick required')
            if control == 1000:
                diagnostic = {'schema': 'flylab.diagnostic.held-cadence-neural.v1',
                    'whole_neural': n.snapshot(), 'body': body_saved,
                    'body_control_tick': session.control_tick,
                    'native_ticks': n.tick, 'held_drive': drive,
                    'autonomous_walking_validated': False}
                StateStore.save(out / 'checkpoint-neural-5s', diagnostic)
                restored = StateStore.load(out / 'checkpoint-neural-5s')
                clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
                    StateStore.load(out / 'candidate-neutral'))
                try:
                    clone.network.restore(restored['whole_neural'])
                    exact(diagnostic['whole_neural'], clone.network.snapshot(), 'Held neural disk state')
                    clone.network.advance_readout(restored['held_drive'], observed, 400)
                    neural_s += .1
                    future = clone.network.snapshot()
                    exact(body_saved, clone.body.snapshot(), 'Held clone body unchanged')
                finally:
                    clone.close()
            if control == 1020:
                exact(future, n.snapshot(), 'Held neural disk .1s future')
                future_exact = True
            if control and control % 200 == 0:
                print(json.dumps({'neural_s': control * .005,
                    'physical_s': 0., 'wall_s': time.perf_counter() - began}), flush=True)
        exact(body_saved, session.body.snapshot(), 'Held original body unchanged')
        final = n.snapshot()
        for key in ('rate', 'adaptation', 'drive'):
            if not np.isfinite(final[key]).all():
                raise ValueError('Finite entire158706 final state required')
        StateStore.save(out / 'final-neural-only', {
            'schema': 'flylab.diagnostic.held-cadence-neural.v1', 'whole_neural': final,
            'body': body_saved, 'body_control_tick': session.control_tick,
            'native_ticks': n.tick, 'held_drive': drive, 'autonomous_walking_validated': False})
    finally:
        if rows:
            np.savez_compressed(out / 'recruitment-signals.npz',
                time_s=np.arange(len(rows)) * .005, observed_ids=observed,
                **{key: np.array([row[key] for row in rows]) for key in rows[0]})
        session.close()
    data = np.load(out / 'recruitment-signals.npz')
    lookup = {int(value): i for i, value in enumerate(observed)}
    time_s = data['time_s']
    late = (time_s >= 5) & (time_s < 10)
    rhythms = []
    for group, cell in enumerate(e1):
        values = data['rate'][:, lookup[int(cell)]]
        peaks, _ = find_peaks(values, prominence=20)
        pts = time_s[peaks]
        periods = np.diff(pts[(pts >= 5) & (pts < 10)])
        width = float(np.percentile(values[late], 95) - np.percentile(values[late], 5))
        mean = float(periods.mean()) if len(periods) else None
        cv = float(periods.std() / periods.mean()) if len(periods) else None
        rhythms.append({'group': group, 'E1_id': int(cell), 'complete_cycles': len(periods),
            'width': width, 'mean_period_s': mean, 'CV': cv,
            'screen': bool(width >= 20 and len(periods) >= 8 and cv <= .3
                and abs(mean / target_period - 1) <= .1)})
    pairs = {}
    for name, left, right in [('LF-RF', 2, 1), ('LM-RM', 5, 3), ('LH-RH', 4, 0)]:
        values = np.zeros((len(time_s), 5))
        values[:, 0] = data['rate'][:, lookup[int(e1[left])]]
        values[:, 4] = data['rate'][:, lookup[int(e1[right])]]
        pairs[name] = phase_summary(time_s, values, start=5., stop=10.)
    result = {'experiment': spec['experiment'], 'validity': 'valid_full_BANC_held_sensory_assay',
        'hypothesis_verdict': 'supported_source_cadence' if all(r['screen'] for r in rhythms)
            else 'not_supported', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'timescale_factor': factor, 'source_rhythms': rhythms, 'neural_LR': pairs,
        'all_LR_screen': all(row['screen'] for row in pairs.values()),
        'body_unchanged': True, 'neutral_disk_exact': True, 'held_neural_future_exact': future_exact,
        'actual_neural_s': round(neural_s, 9), 'actual_physical_s': 0.,
        'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'hypothesis_verdict': 'not_tested', 'error': type(error).__name__, 'message': str(error)})
        raise
