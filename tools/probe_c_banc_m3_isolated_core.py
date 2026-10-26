"""Separate original RF four-cell recurrence from its dynamic source bridges."""

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
from tools.probe_c_banc_m3_local_coordination import advance


def rhythmic_summary(times, rates):
    selected = (times >= .5) & (times < 2.2)
    values = rates[selected, 0]
    peaks, _ = find_peaks(values, prominence=20)
    periods = np.diff(times[selected][peaks])
    width = float(np.percentile(values, 95) - np.percentile(values, 5))
    cv = float(periods.std() / periods.mean()) if len(periods) else None
    return {'E1_width': width, 'complete_cycles': len(periods), 'period_s': periods.tolist(),
        'mean_period_s': float(periods.mean()) if len(periods) else None, 'CV': cv,
        'screen': bool(width >= 20 and len(periods) >= 3 and cv <= .3)}


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    e34, e35 = root / 'e34-coordination-field', root / 'e35-local-coordination'
    spec = {'experiment': 'M3-E51-isolated-source-RF-core',
        'question': 'Does the original RF four-cell recurrence sustain rhythm with constant measured outside current, or does the E35 rhythm require time-varying bridge input under that exposed operating point?',
        'arms': ['isolated_RF_from_zero', 'isolated_RF_from_exposed_E35_t0p5', 'dynamic188_from_same_exposed_E35_t0p5'],
        'intervention': 'Original source RF4x4/signs/tau20ms/threshold7.5/cap200; core adaptation0. Outside RF4 is replaced only in isolated arms by E35[.5,2.2) mean bridge current plus original constant DNg100 and other current. Dynamic188 keeps original W and bridge adaptation8/.3. No external waveform/phase/reset during each arm.',
        'comparison': 'The isolated-exposed and dynamic188 arms share exactly the same RF starting r/z. No direct other-core input exists. Difference implicates dynamic bridge input only under this frozen mean-field condition; isolated success is not fullBANC gait or biological proof.',
        'validity': 'RF identities/types/group and absent other-core inputs checked. Dynamic188 first1.7s at5ms must reproduce saved E35 t.5..2.2 exactly. Record native RF state before summarizing and preserve raw arrays on error.',
        'screen': 'E1 p95-p05>=20, >=3complete prominence20 cycles, periodCV<=.3 in local [.5,2.2). Native.25ms samples. If isolated arms pass, derive one tau scaling from their measured period to.15s in a separate preregistered assay; never assume E35 was4node.',
        'limits': 'CPU only3x2.2reduced-neural seconds,<=90wallsec;0wholeBANC/physics, noF6/bodypose exposure, noadoption/M2/UI changes.',
        'source_e34_sha256': sha256(e34 / 'observations.npz'),
        'source_e35_sha256': sha256(e35 / 'core_zero.npz'),
        'integrator_sha256': sha256(Path('tools/probe_c_banc_m3_local_coordination.py')),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    observed = np.load(e34 / 'observations.npz')
    baseline = np.load(e35 / 'core_zero.npz')
    meta = json.loads((e34 / 'provenance.json').read_text())
    types = ('IN17A001', 'INXXX466', 'IN16B036', 'IN19A007')
    rf = np.array([i for cell_type in types for i, node in enumerate(meta['nodes'][:24])
        if meta['core_groups'][i] == 1 and node['cell_type'] == cell_type])
    if len(rf) != 4 or meta['nodes'][rf[0]]['root_id'] != '720575941554038555':
        raise ValueError('Frozen source RF identities changed')
    original = observed['local_weights'][:188, :188]
    other_core = np.setdiff1d(np.arange(24), rf)
    if np.any(original[np.ix_(rf, other_core)] != 0):
        raise ValueError('Unexpected direct cross-core edge')
    selected = (baseline['time_s'] >= .5) & (baseline['time_s'] < 2.2)
    outside = np.setdiff1d(np.arange(188), rf)
    current188 = baseline['constant_descending_current'] + baseline['constant_other_current']
    mean_bridge = original[np.ix_(rf, outside)] @ baseline['rate'][selected][:, outside].mean(axis=0)
    current4 = current188[rf] + mean_bridge
    weight4 = csr_matrix(original[np.ix_(rf, rf)])
    np.savez_compressed(out / 'source-parameters.npz', RF_ids=observed['indices'][rf],
        RF_local_ids=rf, weights4=weight4.toarray(), constant4=current4,
        constant188=current188, mean_bridge_current=mean_bridge,
        initial4=baseline['rate'][100, rf], initial4_adaptation=baseline['adaptation'][100, rf])
    write_json(out / 'provenance.json', {'RF_nodes': [meta['nodes'][i] for i in rf],
        'types': types, 'RF_local_ids': rf.tolist(), 'outside_field_is_diagnostic': True})
    times = np.arange(8801) * .00025
    results, model_s = {}, 0.
    for name in spec['arms']:
        dynamic = name.startswith('dynamic188')
        if dynamic:
            weights, current = csr_matrix(original), current188
            r, z = baseline['rate'][100].copy(), baseline['adaptation'][100].copy()
            gain = np.full(188, 8.); gain[:24] = 0
            readout = rf
        else:
            weights, current = weight4, current4
            exposed = 'exposed' in name
            r = baseline['rate'][100, rf].copy() if exposed else np.zeros(4)
            z = baseline['adaptation'][100, rf].copy() if exposed else np.zeros(4)
            gain = np.zeros(4); readout = np.arange(4)
        rates, adaptations, control_r, control_z = [r[readout].copy()], [z[readout].copy()], [r.copy()], [z.copy()]
        try:
            for step in range(1, 8801):
                if step % 20 == 0 and time.perf_counter() - began > 85:
                    raise TimeoutError('E51 CPU assay exceeded frozen wall allowance')
                r, z = advance(r, z, weights, current, gain, .00025)
                model_s += .00025
                rates.append(r[readout].copy()); adaptations.append(z[readout].copy())
                if step % 20 == 0:
                    control_r.append(r.copy()); control_z.append(z.copy())
                    if dynamic and step <= 6800:
                        np.testing.assert_array_equal(r, baseline['rate'][100 + step // 20])
                        np.testing.assert_array_equal(z, baseline['adaptation'][100 + step // 20])
            if not np.isfinite(rates).all() or not np.isfinite(adaptations).all():
                raise ValueError('Nonfinite reduced dynamics')
        finally:
            np.savez_compressed(out / (name + '.npz'), time_s=times[:len(rates)], rate=rates,
                adaptation=adaptations, control_rate=control_r, control_adaptation=control_z)
        results[name] = rhythmic_summary(times, np.asarray(rates))
        results[name]['reproduced_prior_1p7s_exact'] = True if dynamic else None
        print(json.dumps({'arm': name, **results[name]}), flush=True)
    result = {'experiment': spec['experiment'], 'validity': 'valid_reduced_constant_field_assay',
        'hypothesis_verdict': 'supported_under_frozen_field' if all(results[n]['screen'] for n in spec['arms'][:2]) else 'not_supported_under_frozen_field',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'results': results,
        'reduced_neural_model_s': model_s, 'actual_whole_neural_model_s': 0,
        'actual_physical_model_s': 0, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}), flush=True)


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
