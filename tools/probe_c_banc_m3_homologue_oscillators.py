"""One measured timescale calibration of six source homologous core circuits."""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_core_adaptation import CORE_TYPES
from tools.probe_c_banc_m3_local_coordination import advance


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E52-source-homologue-oscillators',
        'question': 'Do six existing four-type circuits sustain the already successful RF intrinsic waveform after one source-sign-preserving homologue efficacy normalization and measured timescale adjustment?',
        'intervention': 'RF original nonzero4x4 values by celltype are the target efficacy of existing edges in each other core. Keep missing LF I1->I2 and LM I2->I1 absent. All94efficacies must remain within existing.001..10 bounds. All24core adaptation0 and rate tau=.02*.15/E51 isolated-zero mean period, one derivation with no parameter search. Same E51 constant4 input for each group, diagnostic only.',
        'comparison': 'Six zero-state uncoupled cores; no phase offset or startup envelope. Exact time rescaling of nonadapting constant-field4state dynamics is verified against the original integrator using correspondingly rescaled timestep. Shape is compared after per-cycle phase normalization to E37 exposed RF source mean; no fitting on gait labels.',
        'screen': 'For each E1 native [.5,2.2): p95-p05>=20, >=8complete prominence20 cycles, CV<=.3, mean period within10percent of.15s. Shape error is descriptive only; no claim of phase coordination. Missing-edge failures must not be repaired by adding edges.',
        'decision': 'If all six pass, next inspect source-existing phase coupling and realize constant excitation through actual descending/relay inputs. Fixed fields here never qualify as autonomous fullBANC or F2. No body/F6 exposure or final adoption.',
        'limits': 'CPU only2.2seconds24cell diagnostic,<=60wallsec,0wholeBANC/physics, preserve M2/liveUI.',
        'e51_sha256': sha256(root / 'e51-isolated-source-core/result.json'),
        'parameters_sha256': sha256(root / 'e51-isolated-source-core/source-parameters.npz'),
        'source_sha256': sha256(root / 'e34-coordination-field/observations.npz'),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    e51 = json.loads((root / 'e51-isolated-source-core/result.json').read_text())
    if not all(v['screen'] for k, v in e51['results'].items() if k.startswith('isolated')):
        raise ValueError('Both preceding isolated RF starts must pass')
    period = e51['results']['isolated_RF_from_zero']['mean_period_s']
    tau = .02 * .15 / period
    fixed = np.load(root / 'e51-isolated-source-core/source-parameters.npz')
    obs = np.load(root / 'e34-coordination-field/observations.npz')
    meta = json.loads((root / 'e34-coordination-field/provenance.json').read_text())
    source = obs['local_weights'][:24, :24]
    groups = np.asarray(meta['core_groups'])
    kind = np.array([CORE_TYPES.index(n['cell_type']) for n in meta['nodes'][:24]])
    desired = fixed['weights4'][np.ix_(kind, kind)]
    weight = np.where(source != 0, desired, 0)
    values = weight[source != 0] / source[source != 0]
    if len(values) != 94 or np.any((values < .001) | (values > 10)):
        raise ValueError('Exactly94 existing same-sign bounded source efficacies required')
    if np.any(weight[groups[:, None] != groups[None, :]] != 0):
        raise ValueError('Source core cross-group edges must remain absent')
    current = fixed['constant4'][kind]
    matrix = csr_matrix(weight)
    # For core gain0,z0, rescaling dt and all rate taus leaves each RK stage identical.
    r = np.linspace(1, 180, 24); z = np.zeros(24)
    transformed = advance(r, z, matrix, current, np.zeros(24), .00025 * .02 / tau)
    def scaled_derivative(rate):
        return (np.maximum(200 * np.tanh((current + matrix @ rate - 7.5) / 200), 0) - rate) / tau
    k1 = scaled_derivative(r); k2 = scaled_derivative(r + .000125 * k1)
    k3 = scaled_derivative(r + .000125 * k2); k4 = scaled_derivative(r + .00025 * k3)
    independent = r + .00025 / 6 * (k1 + 2*k2 + 2*k3 + k4)
    np.testing.assert_allclose(transformed[0], independent, atol=3e-14, rtol=2e-16)
    np.testing.assert_array_equal(transformed[1], np.zeros(24))
    np.savez_compressed(out / 'parameters.npz', ids=obs['indices'][:24], weights=weight,
        source_weights=source, tau=np.full(24, tau), current=current, cell_type_index=kind,
        groups=groups, efficacy=values)
    r = np.zeros(24); z = np.zeros(24); rates = [r.copy()]
    times = np.arange(8801) * .00025
    try:
        for step in range(1, 8801):
            if step % 20 == 0 and time.perf_counter() - began > 55:
                raise TimeoutError('E52 bounded CPU assay exceeded wall limit')
            r, z = advance(r, z, matrix, current, np.zeros(24), .00025 * .02 / tau)
            rates.append(r.copy())
        if not np.isfinite(rates).all() or np.any(z != 0):
            raise ValueError('Finite nonadapting core dynamics required')
    finally:
        np.savez_compressed(out / 'native-state.npz', time_s=times[:len(rates)], rate=rates)
    rates = np.asarray(rates)
    window = (times >= .5) & (times < 2.2)
    reference = np.array(json.loads((root / 'e37-core-teacher/teaching-waveform.json').read_text())['mean_cycle'])
    rows, mean_cycles = [], []
    for g in range(6):
        ids = np.array([int(np.flatnonzero((groups == g) & (kind == i))[0]) for i in range(4)])
        selected = np.flatnonzero(window)
        peaks, _ = find_peaks(rates[selected, ids[0]], prominence=20)
        peaks = selected[peaks]
        periods = np.diff(times[peaks])
        width = float(np.percentile(rates[window, ids[0]], 95) - np.percentile(rates[window, ids[0]], 5))
        cv = float(periods.std() / periods.mean()) if len(periods) else None
        cycles = [np.stack([np.interp(np.arange(128)/128, np.linspace(0, 1, b-a+1), rates[a:b+1, i])
            for i in ids], axis=1) for a, b in itertools.pairwise(peaks)]
        cycle = np.mean(cycles, axis=0) if cycles else np.full((128, 4), np.nan)
        mean_cycles.append(cycle)
        row = {'group': g, 'E1_root': meta['nodes'][ids[0]]['root_id'],
            'width': width, 'complete_cycles': len(periods),
            'mean_period_s': float(periods.mean()) if len(periods) else None, 'CV': cv,
            'mean_phase_normalized_RMS_rate': float(np.sqrt(np.mean((cycle-reference)**2))) if cycles else None,
            'screen': bool(width >= 20 and len(periods) >= 8 and cv <= .3 and abs(periods.mean() / .15 - 1) <= .1)}
        rows.append(row)
    np.savez_compressed(out / 'phase-normalized.npz', rate=mean_cycles, reference=reference)
    result = {'experiment': spec['experiment'], 'validity': 'valid_reduced_timescale_assay',
        'hypothesis_verdict': 'supported_under_constant_field' if all(r['screen'] for r in rows) else 'not_supported_under_constant_field',
        'screen': all(r['screen'] for r in rows), 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'tau_s': tau, 'source_RF_period_s': period, 'efficacy_range': [float(values.min()), float(values.max())],
        'core': rows, 'reduced_neural_model_s': 2.2, 'actual_whole_neural_model_s': 0,
        'actual_physical_model_s': 0, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
