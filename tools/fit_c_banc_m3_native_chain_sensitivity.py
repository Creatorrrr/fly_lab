"""Differentiate the actual native source-cell ODE to propose one chain correction."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_feedforward_coordination import PATHS


def tangent_step(state, weights, current, tau, weight_direction, tau_direction, dt=.00025):
    """RK4 rates and exact forward sensitivities to log output and log tau."""
    def rhs(value):
        rate, sensitivity = value[:,0],value[:,1:]
        activation = np.tanh((current+weights@rate-7.5)/200)
        derivative = (np.maximum(200*activation,0)-rate)/tau
        slope = np.where(activation>0,1-activation**2,0)
        field_sensitivity = weights@sensitivity
        field_sensitivity[:,0] += weight_direction@rate
        response = (slope[:,None]*field_sensitivity-sensitivity)/tau[:,None]
        response[:,1] -= derivative*tau_direction
        return np.column_stack((derivative,response))
    k1=rhs(state); k2=rhs(state+dt/2*k1)
    k3=rhs(state+dt/2*k2); k4=rhs(state+dt*k3)
    return state+dt/6*(k1+2*k2+2*k3+k4)


def phase_features(times, states, columns, start=3.5, stop=5.):
    phases, gradients, periods, period_gradients, masks = [],[],[],[],[]
    for col in columns:
        rate, sensitivity = states[:,col,0],states[:,col,1:]
        indices,_ = find_peaks(rate,prominence=20)
        y0,y1,y2 = (rate[indices+k] for k in (-1,0,1))
        s0,s1,s2 = (sensitivity[indices+k] for k in (-1,0,1))
        num,den = y0-y2,y0-2*y1+y2
        peak_times = times[indices]+.5*num/den*(times[1]-times[0])
        peak_grad = .5*((s0-s2)*den[:,None]-num[:,None]*(s0-2*s1+s2))/den[:,None]**2*(times[1]-times[0])
        valid = (peak_times >= start)&(peak_times < stop)
        periods.append(float(np.diff(peak_times[valid]).mean()))
        period_gradients.append(np.diff(peak_grad[valid],axis=0).mean(axis=0))
        bins = np.clip(np.searchsorted(peak_times,times,side='right')-1,0,len(peak_times)-2)
        width = peak_times[bins+1]-peak_times[bins]
        fraction = (times-peak_times[bins])/width
        phases.append(bins+fraction)
        gradients.append(-((1-fraction[:,None])*peak_grad[bins]+fraction[:,None]*peak_grad[bins+1])/width[:,None])
        masks.append((times > peak_times[0])&(times < peak_times[-1])&(times >= start)&(times < stop))
    return np.asarray(phases),np.asarray(gradients),np.asarray(periods),np.asarray(period_gradients),np.asarray(masks)


def pair_feature(features,a,b):
    phase,gradient,_,_,masks = features
    valid = masks[a]&masks[b]
    z = np.exp(2j*np.pi*(phase[a,valid]-phase[b,valid]))
    dz = (2j*np.pi*z[:,None]*(gradient[a,valid]-gradient[b,valid])).mean(axis=0)
    return float(np.angle(z.mean())/(2*np.pi) % 1),np.imag(dz/z.mean())/(2*np.pi)


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'e59-six-source-feedforward'
    names=('zero','exposed_six_offsets')
    spec={'experiment':'M3-E61-native-chain-forward-sensitivity',
        'question':'Can an exact local derivative of the native30cell equations identify one downstream output/tau change that reduces the E59 hind-start disagreement without changing average downstream period?',
        'basis':'E60b phase approximation failed maximum path error. Use the actual half-tanh30cell nativeRK4 tangent equations from the same two E59 initialstates, not a learned phase runtime. Compare the entire baseline rate trajectory to E59 numerically.',
        'parameters':'Two log parameters: multiplier on four downstream positive relay outputs and common tau multiplier on sixteen downstream corecells. Root LF/RF8core and both IN10 relays unchanged; all6relaytau/current/94coreweights unchanged.',
        'candidate':'One Newton linear solve: hind LR start-mean difference target .08 cycles (margin inside unchanged .1 native gate); pooled mean downstream period change zero. Analytic quadratic-peak timing derivatives; no candidate if output multiplier not in[1,sourceefficacy10bound] or tau multiplier outside [.9,1.1]. No nonlinear candidate search or gain/tau grid.',
        'validity':'Baseline rate reconstruction atol1e-9/rtol1e-10, finite tangent and nonsingular objective Jacobian. Derivatives and first-order target are design predictions only; next actual native two-start5s run must pass every original E59screen.',
        'limits':'CPU10reducedneural seconds with two tangentcolumns,<=90wallsec,zero fullBANC/physical seconds. F6/M2/UI preserved.',
        'source_sha256':{n:sha256(source/(n+'.npz')) for n in names},'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    p=np.load(source/'parameters.npz')
    homologues=np.load(root/'e52-homologue-oscillators/parameters.npz')
    e1=np.array([np.flatnonzero((homologues['groups']==g)&(homologues['cell_type_index']==0))[0] for g in range(6)])
    dweight=np.zeros((30,30))
    for j,(_,b,_) in enumerate(PATHS):
        if j>=2:
            dweight[e1[b],24+j]=p['weights'][e1[b],24+j]
    dtau=np.r_[np.isin(p['core_groups'],[0,3,4,5]),np.zeros(6)]
    features,checks={},{}
    for name in names:
        observed=np.load(source/(name+'.npz'))
        state=np.zeros((30,3));state[:,0]=observed['rate'][0]
        states=[state.copy()]
        for step in range(1,len(observed['time_s'])):
            if step%100==0 and time.perf_counter()-began>85:
                raise TimeoutError('E61 bounded native sensitivity exceeded wall limit')
            state=tangent_step(state,p['weights'],p['current'],p['tau'],dweight,dtau)
            states.append(state.copy())
        states=np.asarray(states)
        np.savez_compressed(out/(name+'.npz'),time_s=observed['time_s'],state=states)
        checks[name]=float(np.max(abs(states[:,:,0]-observed['rate'])))
        np.testing.assert_allclose(states[:,:,0],observed['rate'],atol=1e-9,rtol=1e-10)
        if not np.isfinite(states).all():
            raise ValueError('Finite native sensitivity required')
        features[name]=phase_features(observed['time_s'],states,e1)
    a,da=pair_feature(features[names[0]],4,0)
    b,db=pair_feature(features[names[1]],4,0)
    difference=float((a-b+.5)%1-.5)
    period_gradient=np.mean([features[n][3][[0,3,4,5]].mean(axis=0) for n in names],axis=0)
    jacobian=np.vstack((da-db,period_gradient))
    change=np.linalg.solve(jacobian,np.array([.08-difference,0.]))
    multiplier=np.exp(change)
    maximum=float(np.min(10*p['source_path_weights'][2:,1]/p['path_weights'][2:,1]))
    valid=bool(np.isfinite(multiplier).all() and 1<=multiplier[0]<=maximum and .9<=multiplier[1]<=1.1)
    result={'experiment':spec['experiment'],'validity':'valid_native_source_sensitivity',
        'hypothesis_verdict':'candidate_for_native_test' if valid else 'no_bounded_candidate',
        'adoption':'not_applicable','milestone_outcome':'pending',
        'baseline_hind_start_difference':difference,'objective_jacobian':jacobian.tolist(),
        'log_parameter_newton_step':change.tolist(),'proposed_output_multiplier':float(multiplier[0]),
        'proposed_downstream_tau_multiplier':float(multiplier[1]),'maximum_output_multiplier':maximum,
        'selected_downstream_multiplier':float(multiplier[0]) if valid else None,
        'selected_downstream_output':float(p['path_weights'][0,1]*multiplier[0]) if valid else None,
        'selected_core_tau_ratio_by_group':np.where(np.isin(np.arange(6),[0,3,4,5]),multiplier[1],1.).tolist() if valid else None,
        'baseline_rate_max_error':checks,'reduced_neural_model_s':10.,'actual_full_neural_model_s':0,
        'actual_physical_model_s':0,'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.out)
