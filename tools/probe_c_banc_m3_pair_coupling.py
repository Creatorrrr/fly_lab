"""Native reduced assay of a source-existing reciprocal inhibitory leg path."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json


def rate_step(rate, weights, current, tau, dt=.00025):
    def derivative(value):
        return (np.maximum(200 * np.tanh((current + weights @ value - 7.5) / 200), 0) - value) / tau
    k1 = derivative(rate); k2 = derivative(rate + dt/2*k1)
    k3 = derivative(rate + dt/2*k2); k4 = derivative(rate + dt*k3)
    return rate + dt/6 * (k1+2*k2+2*k3+k4)


def phase_summary(times, rates, start=3.5, stop=5.):
    mask = (times >= start) & (times < stop)
    phases, rows = [], []
    for column in (0, 4):
        peaks, _ = find_peaks(rates[:, column], prominence=20)
        periods = np.diff(times[peaks[(times[peaks] >= start) & (times[peaks] < stop)]])
        phase = np.interp(times, times[peaks], np.arange(len(peaks))) if len(peaks) else np.full(len(times), np.nan)
        valid = (times > times[peaks[0]]) & (times < times[peaks[-1]]) if len(peaks) else np.zeros(len(times), bool)
        phases.append(phase); mask &= valid
        values = rates[(times >= start) & (times < stop), column]
        cv = float(periods.std()/periods.mean()) if len(periods) else None
        width = float(np.percentile(values, 95)-np.percentile(values, 5))
        rows.append({'width': width, 'complete_cycles': len(periods),
            'mean_period_s': float(periods.mean()) if len(periods) else None, 'CV': cv,
            'rhythm_screen': bool(width >= 20 and len(periods) >= 8 and cv <= .3)})
    z = np.mean(np.exp(2j*np.pi*(phases[0][mask]-phases[1][mask]))) if mask.any() else None
    mean = float((np.angle(z)/(2*np.pi)) % 1) if z is not None else None
    distance = abs((mean-.5+.5) % 1-.5) if mean is not None else None
    return {'cells': rows, 'phase_mean': mean, 'phase_R': float(abs(z)) if z is not None else None,
        'mean_antiphase_distance': distance,
        'screen': bool(all(r['rhythm_screen'] for r in rows) and z is not None and abs(z) >= .8 and distance <= .15)}


def run(root, out, *, excitatory=False, coupling_fit=None):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E54-source-LF-RF-coupling',
        'question': 'Does one source-existing reciprocal inhibitory E1 pathway converge toward half-cycle phase separation without a phase controller?',
        'paths': [['720575941504247575', '720575941430124247', '720575941554038555'],
                  ['720575941554038555', '720575941572720015', '720575941504247575']],
        'intervention': 'E52original LF/RF4core weights/tau/constantfield. Both IN14B013 relay input efficacies set to the already E40 measured effective1.1707944 E1-to-relay weight; their rate tau20ms/adaptation0/otherfixedfield0. Reciprocal inhibitory output normalized to the stronger of the two original source magnitudes(.24), factors2and1. All fourefficacies .001..10, no new edges. No waveform or phase variable used by dynamics.',
        'arms': ['uncoupled_zero', 'coupled_zero', 'coupled_exposed_quarter'],
        'initial_state': 'Zero r for first2arms; third uses exposed E52 LF cycle peak and RF quarter-cycle-ahead core states with relay0. No subsequent intervention/reset. This neural diagnostic is not F6bodypose exposure.',
        'screen': 'In[3.5,5): both E1width>=20,>=8complete prominence20cycles,CV<=.3; LF/RF peak-interpolated phase mean circular distance from.5<=.15,R>=.8 in both coupled arms. Their final mean phase must agree within.1cycle. Uncoupled arm is a comparison only. Do not infer wholeBANC or biological coupling from reduced pass.',
        'decision': 'If bothstarts pass, test actual LF/RF source bridge transfer and extend bounded source motif. If not, examine phase interaction/timing before broadercoupling, not a blindstrength/taugrid.',
        'validity': 'Original pathnodes/weights/typechecked, adaptationzero throughout, source missingLF edge preserved. Uncoupled native initial path must match E52rates before claims. Save all native state and final parameters first.',
        'limits': 'CPU only3x5reducedneural seconds,<=60wallsec,0fullBANC/body; no finalF/adoption/M2UIchange.',
        'e52_sha256': sha256(root / 'e52-homologue-oscillators/parameters.npz'),
        'path_sha256': sha256(root / 'e34-coordination-field/provenance.json'),
        'tool_sha256': sha256(__file__)}
    if excitatory:
        spec.update(experiment='M3-E55-source-positive-LF-RF-coupling',
            question='E54 reciprocal inhibition synchronized E1 near phase0. Do existing positive paths with matched transmission reverse that phase interaction and establish half-cycle separation?',
            paths=[['720575941504247575', '720575941563370038', '720575941554038555'],
                   ['720575941554038555', '720575941439734290', '720575941504247575']],
            intervention='Replace the two diagnostic IN14B013 paths with existing ACH IN10B012 paths. Match the same effectiveE1input1.1707944, relaytau20ms/gain0/otherfield0, and outputmagnitude.24, now positive through actual positive source edges. Original weights are +.24/+1.77 and+.18/+2.79; no source signflip, newedge, strength search or retuning core/field. Other incoming/outgoing effects remain outside this reduced assay.',
            prediction='In a weak-coupling approximation phase drift depends linearly on signed input. E54 attraction toward equal phase suggests positive matched input may repel equal phase. Global half-cycle stability and speed of convergence are unverified; both same5s starts and unchanged screen remain decisive.',
            decision='Pass warrants whole-neural positivepath transmission verification. Failure keeps the5s coordination claim rejected and motivates measured phase-response/convergence analysis before any strength change. Do not extend duration or relax startup acceptance after inspecting results.')
    fitted_output = None
    if coupling_fit is not None:
        fitted = json.loads(coupling_fit.read_text())
        fitted_output = fitted.get('selected_output_weight')
        if (not excitatory or fitted.get('hypothesis_verdict') != 'candidate_for_native_test'
                or fitted_output is None or not np.isfinite(fitted_output) or not .24 < fitted_output <= 1.77):
            raise ValueError('A finite bounded predeclared positive-coupling candidate required')
        spec.update(experiment='M3-E57-native-fitted-source-coupling',
            question='Does the single E56phase-drift-derived stronger source-positive coupling converge from both original starts in the actual native neuronal ODE?',
            intervention='Same E55two existing positive IN10B012 paths, effectiveinput1.1707944,relaytau20ms/gain0/otherfield0 and E52sourcecores. Only both positive source outputweights change together to the frozen E56measured-phase-fit value. All source signs/efficacybounds and five-second3arm assay remain unchanged.',
            prediction='The fitted phase approximation selected an output from original E55observations. Native dynamics may invalidate its weak-coupling extrapolation or startup anchor; the unchanged actual two-start coordination screen remains decisive.',
            selected_output_weight=fitted_output, coupling_fit_sha256=sha256(coupling_fit),
            decision='Pass permits actual whole-BANC source-pair transfer before broader six-core coupling. Failure rejects the same native convergence hypothesis, with no further strength grid or relaxed screen.')
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    source = np.load(root / 'e52-homologue-oscillators/parameters.npz')
    history = np.load(root / 'e52-homologue-oscillators/native-state.npz')
    observed = np.load(root / 'e34-coordination-field/observations.npz')
    meta = json.loads((root / 'e34-coordination-field/provenance.json').read_text())
    ids = np.array([int(np.flatnonzero((source['groups'] == group) & (source['cell_type_index'] == kind))[0])
        for group in (2, 1) for kind in range(4)])
    relay_local = [next(i for i, n in enumerate(meta['nodes']) if n['root_id'] == path[1]) for path in spec['paths']]
    weights = np.zeros((10, 10))
    weights[:8, :8] = source['weights'][np.ix_(ids, ids)]
    local = observed['local_weights']
    original_paths = np.array([[local[relay_local[0], ids[0]], local[ids[4], relay_local[0]]],
                               [local[relay_local[1], ids[4]], local[ids[0], relay_local[1]]]])
    expected_paths = [[.24, 1.77], [.18, 2.79]] if excitatory else [[.39, -.12], [.12, -.24]]
    np.testing.assert_allclose(original_paths, expected_paths, atol=2e-8, rtol=1e-7)
    effective_input = 1.1707944
    output = .23999999463558197 if excitatory else -float(abs(original_paths[:, 1]).max())
    if fitted_output is not None:
        output = fitted_output
    changed_paths = np.tile([effective_input, output], (2, 1))
    efficacy = changed_paths / original_paths
    if np.any((efficacy < .001) | (efficacy > 10)):
        raise ValueError('Four existing source efficacy bounds required')
    weights[8, 0] = weights[9, 4] = effective_input
    weights[4, 8] = weights[0, 9] = output
    current = np.r_[source['current'][ids], 0., 0.]
    tau = np.r_[source['tau'][ids], .02, .02]
    np.savez_compressed(out / 'parameters.npz', weights=weights, current=current, tau=tau,
        source_core_indices=ids, core_ids=source['ids'][ids], relay_ids=observed['indices'][relay_local],
        source_path_weights=original_paths, path_weights=changed_paths, efficacy=efficacy)
    times = np.arange(20001) * .00025
    results = {}
    for name in spec['arms']:
        matrix = weights.copy(); r = np.zeros(10)
        if name == 'uncoupled_zero':
            matrix[:8, 8:] = 0
        if name == 'coupled_exposed_quarter':
            for offset, phase in ((0, 0.), (4, .25)):
                peaks, _ = find_peaks(history['rate'][:, ids[offset]], prominence=20)
                a, b = peaks[-2:]
                index = a + round((b-a)*phase)
                r[offset:offset+4] = history['rate'][index, ids[offset:offset+4]]
        rates = [r.copy()]
        try:
            for step in range(1, 20001):
                if step % 20 == 0 and time.perf_counter()-began > 55:
                    raise TimeoutError('E54 bounded source coupling assay exceeded wall allowance')
                r = rate_step(r, matrix, current, tau)
                rates.append(r.copy())
                if name == 'uncoupled_zero' and step <= 800:
                    np.testing.assert_allclose(r[:8], history['rate'][step, ids], atol=1e-10, rtol=1e-11)
            if not np.isfinite(rates).all():
                raise ValueError('Finite native pair dynamics required')
        finally:
            np.savez_compressed(out / (name+'.npz'), time_s=times[:len(rates)], rate=rates)
        results[name] = phase_summary(times, np.asarray(rates))
        print(json.dumps({'arm': name, **results[name]}), flush=True)
    a, b = (results[n]['phase_mean'] for n in spec['arms'][1:])
    start_agreement = abs((a-b+.5) % 1-.5) if a is not None and b is not None else None
    passed = all(results[n]['screen'] for n in spec['arms'][1:]) and start_agreement <= .1
    result = {'experiment': spec['experiment'], 'validity': 'valid_reduced_source_coupling_assay',
        'hypothesis_verdict': 'supported' if passed else 'not_supported', 'screen': passed,
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'results': results,
        'coupled_start_phase_agreement': start_agreement, 'reduced_neural_model_s': 15.,
        'actual_whole_neural_model_s': 0, 'actual_physical_model_s': 0, 'wall_s': time.perf_counter()-began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--excitatory', action='store_true')
    parser.add_argument('--coupling-fit', type=Path)
    args = parser.parse_args()
    try:
        run(args.root, args.out, excitatory=args.excitatory, coupling_fit=args.coupling_fit)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
