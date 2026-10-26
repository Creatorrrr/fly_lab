"""Estimate feedforward convergence and its frequency cost from recorded E59 cycles."""

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
from tools.probe_c_banc_m3_feedforward_coordination import PATHS


def peak_times(times, values):
    indices, _ = find_peaks(values, prominence=20)
    y0, y1, y2 = (values[indices + k] for k in (-1, 0, 1))
    return times[indices] + .5 * (y0-y2)/(y0-2*y1+y2) * (times[1]-times[0])


def fourier(phi):
    angle = 2*np.pi*np.asarray(phi)[..., None]*np.arange(1, 4)
    return np.concatenate((np.ones_like(angle[..., :1]), np.sin(angle), np.cos(angle)), axis=-1)


def circular_distance(a, b):
    return np.abs((np.asarray(a)-b+.5) % 1-.5)


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root/'e59-six-source-feedforward'
    names = ('zero', 'exposed_six_offsets')
    spec = {
        'experiment': 'M3-E60b-recorded-chain-frequency-and-convergence',
        'measurement_revision': 'E60 original phase comparison inadvertently included times after the last observed peak: np.interp clamped the endpoint. Original artifact retained; E60b restricts all comparisons and candidate-design means to the common measured phase interval. Native E59 verdict and next native screen remain unchanged.',
        'question': 'Does a measured common phase response reproduce E59 chain delay, and can one bounded downstream coupling change accelerate it with its mean-frequency shift compensated by core tau?',
        'data': 'Only E52 unforced and E59 two already exposed native trajectories. Quadratically interpolated E1 peaks; post-cycle reciprocal period minus measured unforced frequency, regressed on source-minus-post phase at cycle midpoint. Pool both arms/six paths for a constant plus sin/cos harmonics1..3. This is explanatory design fitting, not independent validation.',
        'validity': 'Six-variable phase ODE at unchanged weights/tau must reproduce both E59 traces from t.5 to4.9, max circular error across every source-path phase <=.05. No candidate if not. Phase ODE is analysis only, never executed in the walking network.',
        'candidate': 'Keep front pair unchanged. One common multiplier lambda for four downstream positive relay outputs, original efficacy <=10. Downstream tau multiplier=(omega+lambda*H(.5))/(omega+H(.5)) offsets the predicted added mean frequency at half phase. Single brent root for maximum of three LR start-to-start mean differences in[3.5,4.9] to .1; no grid. No candidate if unbracketed. Native next assay retains original[3.5,5) and E59 screens.',
        'limits': 'CPU <=60wallseconds, zero native neural/physical time; all original failures/F1-F8/M2/UI/F6 preserved.',
        'source_sha256': {n: sha256(source/(n+'.npz')) for n in names},
        'tool_sha256': sha256(__file__)}
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    p = np.load(root/'e52-homologue-oscillators/parameters.npz')
    e1 = np.array([np.flatnonzero((p['groups']==g)&(p['cell_type_index']==0))[0] for g in range(6)])
    unforced = np.load(root/'e52-homologue-oscillators/native-state.npz')
    omega = []
    for col in e1:
        peaks = peak_times(unforced['time_s'], unforced['rate'][:, col])
        periods = np.diff(peaks[peaks > .5])
        omega.append(1/periods.mean())
    omega = np.asarray(omega)
    pre = np.empty(6, int)
    for a,b,_ in PATHS:
        pre[b] = a
    events, x, y = {}, [], []
    for name in names:
        history = np.load(source/(name+'.npz'))
        events[name] = [peak_times(history['time_s'], history['rate'][:, col]) for col in e1]
        for g in range(6):
            own, upstream = events[name][g], events[name][pre[g]]
            mid = (own[1:]+own[:-1])/2
            valid = (mid >= .5)&(mid < 4.9)&(mid > upstream[0])&(mid < upstream[-1])
            phi = np.interp(mid[valid], upstream, np.arange(len(upstream))) - (np.arange(len(mid))[valid]+.5)
            x.extend(phi)
            y.extend(1/np.diff(own)[valid]-omega[g])
    x,y = np.asarray(x),np.asarray(y)
    design = fourier(x)
    coefficient = np.linalg.lstsq(design,y,rcond=None)[0]
    h_half = float(fourier(.5)@coefficient)
    common_end = min(peaks[-1] for name in names for peaks in events[name])
    times = np.arange(.5,min(4.901,common_end),.005)
    actual = {name: np.array([np.interp(times, peaks, np.arange(len(peaks))) for peaks in events[name]]) for name in names}
    downstream = np.isin(np.arange(6), [0,3,4,5])
    def predict(name, multiplier):
        if time.perf_counter()-began > 55:
            raise TimeoutError('E60 bounded phase fit exceeded wall limit')
        strength = np.where(downstream, multiplier, 1.)
        tau_ratio = (omega+strength*h_half)/(omega+h_half)
        if np.any(tau_ratio <= 0):
            raise ValueError('Positive frequency compensation required')
        def derivative(t, phase):
            return (omega+strength*(fourier(phase[pre]-phase)@coefficient))/tau_ratio
        sol = solve_ivp(derivative,(times[0],times[-1]),actual[name][:,0], t_eval=times,
            rtol=1e-9,atol=1e-10,max_step=.01)
        if not sol.success or not np.isfinite(sol.y).all():
            raise ValueError('Finite conditional phase integration required')
        return sol.y, tau_ratio
    errors, prediction = {}, {}
    for name in names:
        predicted,_ = predict(name,1.)
        prediction[name] = predicted
        errors[name] = circular_distance(predicted[pre]-predicted, actual[name][pre]-actual[name]).max(axis=1).tolist()
    params = np.load(source/'parameters.npz')
    base_output = params['path_weights'][0,1]
    maximum = float(np.min(10*params['source_path_weights'][2:,1]/params['path_weights'][2:,1]))
    late = times >= 3.5
    def agreement(multiplier):
        pair_means = []
        for name in names:
            phase,_ = predict(name,multiplier)
            pair_means.append([np.angle(np.exp(2j*np.pi*(phase[a,late]-phase[b,late])).mean())/(2*np.pi)
                for a,b in ((2,1),(5,3),(4,0))])
        return circular_distance(pair_means[0],pair_means[1])
    valid = max(max(v) for v in errors.values()) <= .05
    endpoints = [agreement(v).tolist() for v in (1.,maximum)] if valid else None
    selected = None
    if endpoints is not None and max(endpoints[0]) > .1 and max(endpoints[1]) <= .1:
        selected = float(brentq(lambda value: float(agreement(value).max())-.1,1.,maximum,xtol=1e-7))
    selected_tau = predict(names[0],selected)[1] if selected is not None else None
    np.savez_compressed(out/'analysis.npz',phase=x,frequency_shift=y,coefficient=coefficient,
        omega=omega,time_s=times, **{n+'_actual': actual[n] for n in names},
        **{n+'_predicted': prediction[n] for n in names})
    result = {'experiment':spec['experiment'],'validity':'valid_recorded_phase_model_analysis',
        'approximation_supported':valid,'hypothesis_verdict':'candidate_for_native_test' if selected is not None else 'no_candidate',
        'adoption':'not_applicable','milestone_outcome':'pending',
        'unforced_frequency':omega.tolist(),'fourier_coefficients':coefficient.tolist(),
        'halfphase_frequency_shift':h_half,'frequency_fit_RMS':float(np.sqrt(np.mean((design@coefficient-y)**2))),
        'source_path_phase_max_error':errors,'maximum_downstream_multiplier':maximum,
        'actual_comparison_interval_s':[float(times[0]),float(times[-1])],
        'predicted_start_agreement_endpoints':endpoints,'selected_downstream_multiplier':selected,
        'selected_downstream_output':float(base_output*selected) if selected is not None else None,
        'selected_core_tau_ratio_by_group':selected_tau.tolist() if selected_tau is not None else None,
        'actual_neural_model_s':0,'actual_physical_model_s':0,'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.out)
