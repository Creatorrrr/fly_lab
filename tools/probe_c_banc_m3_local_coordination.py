"""Autonomous source-core/relay dynamics under explicit constant neural fields.

This is a reduced diagnostic, never a substitute for full BANC/body validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_boundary import rk4


def advance(r, z, weights, current, gain, dt):
    def derivative(rate, adaptation):
        target = np.maximum(200 * np.tanh((current + weights @ rate - 7.5 - adaptation) / 200), 0)
        return (target - rate) / .02, (gain * rate - adaptation) / .3
    k1, a1 = derivative(r, z)
    k2, a2 = derivative(r + dt/2*k1, z + dt/2*a1)
    k3, a3 = derivative(r + dt/2*k2, z + dt/2*a2)
    k4, a4 = derivative(r + dt*k3, z + dt*a3)
    return r + dt/6*(k1+2*k2+2*k3+k4), z + dt/6*(a1+2*a2+2*a3+a4)


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E35-local-coordination',
        'question': 'Can the existing188core/relay recurrent subgraph generate repeated rhythms under constant measured fields, and is DC background inhibition a limiting factor?',
        'scope': 'Reduced188neuron diagnostic, not wholeBANC or body/controller. Outside dynamical feedback is absent in ALL arms. Source188x188weights and signs preserved; DNg100 contribution is its measured .5..2.2 mean, constant fromstart. No oscillatory input, phase, reset or learnedoutput.',
        'arms': ['constant_measured_other_field_with_original8adaptation', 'same_constant_field_core_gain0', 'core_gain0_and_constant_other_field_centered_to0'],
        'interpretation': 'Last arm is an optimistic input-operating-point hypothesis, equivalent to canceling constant other-field bias for188cells. It is not a validated cell physiology change. Failure to obtain repeated core rhythms with these unchanged recurrent weights argues against simple DC correction sufficiency; success only permits a full-feedback candidate design.',
        'screen': 'Descriptive E1 amplitude95-5>=20modelunits, >=3complete peak cycles after.5, CV<=.3 inall6source groups. This is notF4/F5, and no anatomical legphase labels inferred from an SCC. No signal-parameter grid if screenfails.',
        'integration': 'float64source188sparse RK4 .25ms, zero initialr/z,2.2s each; gain8 matches prior independent RK helper on fixed vectors before integration.',
        'limits': 'CPU only,3x2.2reduced-neural-modelsec,<=90wallsec,0physics/wholeBANCsteps, noadoption/F6exposure',
        'observations_sha256': sha256(source / 'observations.npz'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec); (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter(); data = np.load(source / 'observations.npz')
    meta = json.loads((source / 'provenance.json').read_text())
    weights = csr_matrix(data['local_weights'][:188, :188]); observations = data['values']
    times = np.arange(len(observations))*.005; mask = (times >= .5) & (times < 2.2)
    other = observations[mask, :188, 6].mean(axis=0)
    descending = observations[mask, :188, 5].mean(axis=0)
    if np.any(observations[:, :188, 7] != 0):
        raise ValueError('Unexpected direct source core/relay input')
    e1 = np.array([i for i, n in enumerate(meta['nodes'][:24]) if n['cell_type'] == 'IN17A001'])
    if len(e1) != 6:
        raise ValueError('Six original E1 neurons required')
    r = np.linspace(0, 180, 188); z = np.linspace(300, 0, 188)
    actual = advance(r, z, weights, other+descending, np.full(188, 8.), .00025)
    expected = rk4(r, z, other+descending, weights, .00025)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    results = {}
    for name, core_gain, background in (('original', 8., other), ('core_zero', 0., other), ('centered', 0., np.zeros(188))):
        r = np.zeros(188); z = np.zeros(188); gain = np.full(188, 8.); gain[:24] = core_gain
        rates = [r.copy()]; adaptations = [z.copy()]
        for control in range(440):
            if time.perf_counter()-began > 85:
                np.savez_compressed(out / (name+'-incomplete.npz'), rate=rates, adaptation=adaptations)
                raise TimeoutError('E35 bounded CPU assay exceeded wall allowance')
            for _ in range(20):
                r, z = advance(r, z, weights, descending+background, gain, .00025)
            rates.append(r.copy()); adaptations.append(z.copy())
        rates = np.asarray(rates); adaptations = np.asarray(adaptations)
        np.savez_compressed(out / (name+'.npz'), time_s=times, rate=rates, adaptation=adaptations,
            constant_descending_current=descending, constant_other_current=background, core_ids=e1)
        rows = []
        for i in e1:
            values = rates[mask, i]; peaks, _ = find_peaks(values, prominence=20)
            period = np.diff(peaks)*.005
            width = float(np.percentile(values,95)-np.percentile(values,5))
            cv = float(period.std()/period.mean()) if len(period) else None
            rows.append({'root': meta['nodes'][i]['root_id'], 'core_group': meta['core_groups'][i],
                'width': width, 'cycles': len(period), 'period_s': period.tolist(), 'CV': cv,
                'screen': bool(width>=20 and len(period)>=3 and cv<=.3)})
        results[name] = {'core_E1': rows, 'screen': all(row['screen'] for row in rows),
            'whole_rate_finite': bool(np.isfinite(rates).all()), 'constant_field_only': True}
        print(json.dumps({'arm': name, **results[name], 'wall_s': time.perf_counter()-began}), flush=True)
    write_json(out / 'result.json', {'experiment': spec['experiment'], 'validity': 'valid_reduced_autonomous_field_assay',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'results': results,
        'reduced_neural_model_s': 6.6, 'wall_s': time.perf_counter()-began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory must be new')
    try:
        run(args.source, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'error': type(error).__name__, 'message': str(error),
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested'})
        raise
