"""Local frozen-field Jacobians on exposed E32 observations, not Lyapunov tests."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E33-core-operating-range',
        'question': 'Does the source core operate in its responsive recurrent regime during actual E32, or does varying external current frequently rectify/saturate it?',
        'method': 'Read fixed E32 CUDA recurrent current/rate/z, full checkpoint and original4-cell source matrices. Differentiate existing half-tanh analytically; compute instantaneous4r or8r/z Jacobian eigenvalues while all outside neural states are held fixed. No integration, training or physical replay.',
        'interpretation': 'Local slope<=.5 is at most half maximum transfer responsiveness; slope0 rectified. Eigenvalues are instantaneous frozen-field descriptors, not global Lyapunov exponents, asymptotic stability, an autonomous limit cycle or inter-leg coordination proof. Two-point and instantaneous observations cannot isolate causality.',
        'parameter_contract': 'OriginalM2 full-neural base a91e46336499c88b112eeffc5b6fdac2526b578046fe41791016095dfa677bcf:tau.02,a1,threshold7.5,cap200; core8/.3 or0/.3. Core receives no direct input under existing sensory/tonic routing. Both saved start/final drive vectors must be0 there.',
        'limits': 'CPU only,<=60wallsec,0neural/physicsmodelsec; no candidate/adoption/F6 exposure',
        'source_result_sha256': sha256(source / 'result.json'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec); (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter(); results = {}
    for arm in ('baseline', 'candidate'):
        data = np.load(source / arm / 'neural-observation.npz')
        core = data['core_ids']; observations = data['core_state']
        provenance = json.loads((source / arm / 'core-provenance.json').read_text())
        groups = np.asarray(provenance['groups']); weights = np.asarray(provenance['core_weights'])
        for state_name in ('start', 'final'):
            state = StateStore.load(source / arm / state_name)['network']
            if state['core_adaptation']['base_model_hash'] != 'a91e46336499c88b112eeffc5b6fdac2526b578046fe41791016095dfa677bcf':
                raise ValueError('Original scalar physiology identity differs')
            if np.any(state['drive'][core] != 0):
                raise ValueError('Core direct external input not accounted for')
        t = np.arange(len(observations)) * .005; select = (t >= .5) & (t < 2.2)
        obs = observations[select]; margin = obs[:, :, 2] - 7.5 - obs[:, :, 1]
        slope = np.where(margin > 0, 1 - np.tanh(margin / 200) ** 2, 0.)
        eigenvalues = []
        for group in range(6):
            indices = np.flatnonzero(groups == group); w = weights[np.ix_(indices, indices)]
            local = []
            for gain in slope[:, indices]:
                rr = (gain[:, None] * w - np.eye(4)) / .02
                if arm == 'baseline':
                    jacobian = np.block([[rr, -np.diag(gain) / .02],
                        [8 / .3 * np.eye(4), -np.eye(4) / .3]])
                else:
                    jacobian = rr
                local.append(np.linalg.eigvals(jacobian))
            eigenvalues.append(np.asarray(local))
        summary = {'core_ids': core.tolist(), 'cell_types': [n['cell_type'] for n in provenance['nodes']],
            'median_transfer_slope': np.median(slope, axis=0).tolist(),
            'rectified_fraction': np.mean(margin <= 0, axis=0).tolist(),
            'at_least_half_slope_fraction': np.mean(slope >= .5, axis=0).tolist(),
            'groups': [{'group': i, 'fraction_any_positive_real_local_eigenvalue': float((e.real.max(axis=1) > 0).mean()),
                'max_real_eigenvalue_quantiles': np.percentile(e.real.max(axis=1), [5, 50, 95]).tolist(),
                'fraction_complex_local_eigenvalue': float((abs(e.imag).max(axis=1) > 1e-8).mean()),
                'median_largest_imag_frequency_hz': float(np.median(abs(e.imag).max(axis=1) / (2*np.pi)))}
                for i, e in enumerate(eigenvalues)]}
        np.savez_compressed(out / (arm + '-local.npz'), time_s=t[select], margin=margin,
            transfer_slope=slope, local_eigenvalues=np.asarray(eigenvalues))
        results[arm] = summary
        print(json.dumps({'arm': arm, 'mean_rectified_fraction': float((margin <= 0).mean()),
            'mean_half_slope_fraction': float((slope >= .5).mean()), 'groups': summary['groups']}), flush=True)
    write_json(out / 'result.json', {'experiment': spec['experiment'], 'validity': 'valid_local_derivative_analysis',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'results': results,
        'model_s': 0., 'wall_s': time.perf_counter() - began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
