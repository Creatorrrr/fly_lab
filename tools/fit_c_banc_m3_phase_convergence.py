"""Estimate a source coupling magnitude from already observed phase drift."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json


def event_phase(data):
    t, rates = data['time_s'], data['rate']
    peaks = []
    for i in (0, 4):
        indices, _ = find_peaks(rates[:, i], prominence=20)
        y0, y1, y2 = (rates[indices + k, i] for k in (-1, 0, 1))
        correction = .5 * (y0-y2) / (y0-2*y1+y2)
        peaks.append(t[indices] + correction * (t[1]-t[0]))
    valid = (peaks[0] > max(peaks[1][0], .5)) & (peaks[0] < min(peaks[1][-1], 5.))
    times = peaks[0][valid]
    phase = np.arange(len(peaks[0]))[valid] - np.interp(times, peaks[1], np.arange(len(peaks[1])))
    phase = np.unwrap(2*np.pi*((phase+.5) % 1-.5))/(2*np.pi)
    return times, phase


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e55-source-positive-pair'
    spec = {'experiment': 'M3-E56-measured-phase-convergence',
        'question': 'Can observed positive coupling phase drift explain the slow zero-start separation and identify one bounded faster candidate without a strength grid?',
        'data': 'Use E55uncoupled_zero and bothcoupled traces only. Quadratic native peak timestamps, LFpeak-index minus interpolated RFpeak-index, unwrap. Eventdifferences provide average phasevelocity; original uncoupled frequency mismatch is fixed.',
        'model': 'Fit three odd Fourier coefficients b1..b3 to phidot=measured_delta_frequency+sum b_k*sin(2pi*k*phi). Positive/negative reciprocity and same operatingpoint motivate a phase-only approximation, not an exact neural mechanism. This phase variable is training analysis only, never a runtime controller.',
        'validity': 'The fitted scale1 scalarODE must reproduce both original observed phase traces with max circular phaseerror<=.05cycles from their first LFpeak after.5. If not, no strength candidate is produced.',
        'candidate': 'Single brent root for minimal uniform outputscale>=1 whose fitted zero-start phase reaches the unchanged halfcycle band(distance<=.15) by absolute modelt2.0s. Upper scale limits output to smaller original positive output1.77, below the original overall efficacy10 bound. If not bracketed, report no candidate. Preserve existing inputstrength/taus/core/field; do not extend E57 five-second validation or change its screen.',
        'limits': 'CPU<=30wallseconds,0actualneural/physics. Scalar phaseODE numerical predictions are conditional design calculations, not source-neural runs/Fverification. F6/M2UIpreserved.',
        'source_sha256': {name: sha256(source / (name+'.npz')) for name in ('uncoupled_zero', 'coupled_zero', 'coupled_exposed_quarter')},
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    names = ('uncoupled_zero', 'coupled_zero', 'coupled_exposed_quarter')
    events = {name: event_phase(np.load(source / (name+'.npz'))) for name in names}
    t, phi = events[names[0]]
    delta = float(np.polyfit(t, phi, 1)[0])
    midpoints, velocities = [], []
    for name in names[1:]:
        t, phi = events[name]
        midpoints.extend((phi[:-1]+phi[1:])/2)
        velocities.extend(np.diff(phi)/np.diff(t))
    midpoints, velocities = np.array(midpoints), np.array(velocities)
    harmonics = np.arange(1, 4)
    design = np.sin(2*np.pi*midpoints[:, None]*harmonics)
    coefficient = np.linalg.lstsq(design, velocities-delta, rcond=None)[0]
    def predict(scale, anchor, times):
        if time.perf_counter()-began > 25:
            raise TimeoutError('E56 bounded phase-model fit exceeded wall limit')
        def derivative(t, value):
            return [delta+scale*float(np.sin(2*np.pi*harmonics*value[0])@coefficient)]
        solution = solve_ivp(derivative, (anchor[0], float(times[-1])), [anchor[1]],
            t_eval=times, rtol=1e-9, atol=1e-11, max_step=.01)
        if not solution.success or not np.isfinite(solution.y).all():
            raise ValueError('Finite phase approximation solution required')
        return solution.y[0]
    errors, predictions = {}, {}
    for name in names[1:]:
        t, phi = events[name]
        predicted = predict(1., (t[0], phi[0]), t)
        errors[name] = float(np.max(np.abs((predicted-phi+.5) % 1-.5)))
        predictions[name] = predicted
    np.savez_compressed(out / 'training-data.npz', phase_midpoint=midpoints, phase_velocity=velocities,
        coefficient=coefficient, delta_frequency=delta,
        **{name+'_time_s': events[name][0] for name in names},
        **{name+'_phase': events[name][1] for name in names},
        **{name+'_predicted': predictions[name] for name in names[1:]})
    parameter = np.load(source / 'parameters.npz')
    baseline_output = float(parameter['path_weights'][0, 1])
    max_output = float(parameter['source_path_weights'][:, 1].min())
    maximum_scale = max_output / baseline_output
    selected, predicted_final = None, None
    anchor_time, anchor_phase = (events['coupled_zero'][0][0], events['coupled_zero'][1][0])
    def distance(scale):
        phi = float(predict(scale, (anchor_time, anchor_phase), np.array([2.]))[0])
        return abs((phi % 1-.5+.5) % 1-.5)
    error_valid = all(value <= .05 for value in errors.values())
    interval = [distance(1.), distance(maximum_scale)] if error_valid else None
    if interval is not None and interval[0] > .15 and interval[1] <= .15:
        selected = float(brentq(lambda value: distance(value)-.15, 1., maximum_scale, xtol=1e-7, maxiter=30))
        predicted_final = distance(selected)
    result = {'experiment': spec['experiment'], 'validity': 'valid_conditional_phase_fit',
        'hypothesis_verdict': 'candidate_for_native_test' if selected is not None else 'no_candidate',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'delta_frequency_cycles_s': delta, 'odd_fourier_coefficients': coefficient.tolist(),
        'phase_velocity_fit_RMS': float(np.sqrt(np.mean((delta+design@coefficient-velocities)**2))),
        'observed_phase_max_prediction_error_cycles': errors,
        'baseline_output': baseline_output, 'maximum_output': max_output,
        'predicted_boundary_distances_at_scale_endpoints': interval,
        'selected_scale': selected, 'selected_output_weight': baseline_output*selected if selected is not None else None,
        'predicted_halfcycle_distance_at_t2': predicted_final,
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0, 'wall_s': time.perf_counter()-began}
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
