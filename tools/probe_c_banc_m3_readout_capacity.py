"""Optimistic linear anatomical readout bound on exposed E20 native rates."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_closed_boundary import JOINTS


def run(reference, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        'experiment': 'M3-E28-exposed-anatomical-readout-bound',
        'hypothesis': 'E20 trained native motor activity, unlike the old E03 source, may already carry teacher timing hidden by uniform unit pooling',
        'intervention': 'CPU saved-rate analysis only. Apply the existing 10ms unit lowpass from zero at every 5ms control. Independent unrestricted perjoint coefficients on only anatomically supported motor units plus a free intercept give an optimistic superset of positive group-shared fixed unit capacities.',
        'data': 'All exposed development samples .5<=t<2.2. No heldout or body-pose exposure; no neural or physics execution. E03 used different source neural dynamics and static unfiltered rates.',
        'decision': 'If even this unconstrained least-squares superset has all-joint RMS>.05rad or any6tibiaR2<.5, simple linear fixed anatomical unit pooling cannot reconstruct this teacher on this recorded window. This is not a proof of nonlinear impossibility or autonomous gait failure. A passing optimistic bound would still need physical/positive shared weights, generalization and closed-loop tests.',
        'validity': 'Native E21 motor roster, finite inputs, lowpass then uniform pooling reconstructs original decoded command within1e-5rad; unconstrained least-squares residual orthogonal to design within1e-7 relative. Preserve arrays before interpretation.',
        'limits': 'CPU only, one fit perjoint, <=60wallsec. No candidate export or product change. F1-F8 pending.',
        'source_sha256': sha256(reference / 'signals.npz'),
        'tool_sha256': sha256(__file__),
    }
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    data = dict(np.load(reference / 'signals.npz'))
    rates = data['rate'][1::2].astype(float)
    times = data['time_s'][1::2]
    mask = (times >= .5 - 1e-9) & (times < 2.2 - 1e-9)
    state = np.zeros(rates.shape[1]); filtered = []
    for row in rates:
        state += (1 - np.exp(-.5)) * (row - state)
        filtered.append(state.copy())
    filtered = np.asarray(filtered)
    group = data['motor_group']; counts = np.bincount(group)
    mapping = .02 * data['moment'][:, group] / counts[group]
    reconstructed = data['rest'] + filtered @ mapping.T
    parity = float(np.max(abs(reconstructed - data['decoded_command'])))
    predictions = np.empty_like(data['command'], dtype=float)
    coefficients = np.zeros((42, rates.shape[1])); intercept = np.empty(42)
    metrics = []; normal_errors = []
    for joint in range(42):
        columns = np.flatnonzero(mapping[joint])
        design = np.column_stack((np.ones(len(rates)), filtered[:, columns] / 200))
        x = design[mask]; y = data['command'][mask, joint]
        fitted, _, rank, singular = np.linalg.lstsq(x, y, rcond=1e-12)
        predictions[:, joint] = design @ fitted
        coefficients[joint, columns] = fitted[1:] / 200
        intercept[joint] = fitted[0]
        residual = predictions[mask, joint] - y
        normal = np.linalg.norm(x.T @ residual) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
        normal_errors.append(float(normal))
        mse = float(np.mean(residual ** 2)); variance = float(np.var(y))
        metrics.append({'joint': joint, 'columns': len(columns), 'rank': int(rank),
            'rms_rad': float(np.sqrt(mse)), 'r2': 1 - mse / variance if variance > 1e-12 else None,
            'singular_values': singular.tolist(), 'normal_equation_relative_residual': float(normal)})
    np.savez_compressed(out / 'readout.npz', times=times, mask=mask, filtered_motor=filtered,
        prediction=predictions, target=data['command'], coefficients=coefficients,
        intercept=intercept, uniform_reconstructed=reconstructed)
    rms = float(np.sqrt(np.mean((predictions[mask] - data['command'][mask]) ** 2)))
    knee_r2 = [metrics[i]['r2'] for i in JOINTS[:, 5]]
    valid = parity <= 1e-5 and max(normal_errors) <= 1e-7 and np.isfinite(predictions).all()
    passed = rms <= .05 and all(v is not None and v >= .5 for v in knee_r2)
    result = {'experiment': spec['experiment'], 'run_status': 'completed',
        'validity': 'valid_saved_activity_bound' if valid else 'invalid',
        'hypothesis_verdict': ('optimistic_bound_passes' if passed else 'fixed_linear_readout_insufficient_for_this_teacher') if valid else 'not_tested',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'uniform_pooling_max_error_rad': parity, 'optimistic_rms_rad': rms,
        'knee_r2': knee_r2, 'optimistic_screen_passed': bool(valid and passed),
        'joint_metrics': metrics, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'joint_metrics'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.out)
