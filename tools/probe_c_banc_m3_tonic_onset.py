"""One bounded conditional native tonic-onset hypothesis for existing BANC cells."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.verify_c_banc_observation import exact


@njit(cache=True)
def _derivative(rate, weights, background, tau):
    n = len(rate)
    result = np.empty(n)
    for post in range(n):
        current = background[post]
        for pre in range(n):
            current += weights[post, pre] * rate[pre]
        activation = max(200 * math.tanh((current - 7.5) / 200), 0.)
        result[post] = (activation - rate[post]) / tau[post]
    return result


@njit(cache=True)
def reduced_rk4(weights, background, tau, controls, dt=.00025):
    """250us RK4; return 5ms control-boundary rates, not native-step traces."""
    rate = np.zeros(len(tau))
    result = np.zeros((controls + 1, len(tau)))
    for control in range(controls):
        left = background[control]
        delta = background[control + 1] - left
        for substep in range(20):
            fraction = substep / 20.
            b1 = left + fraction * delta
            bm = left + (fraction + .025) * delta
            b4 = left + (fraction + .05) * delta
            k1 = _derivative(rate, weights, b1, tau)
            k2 = _derivative(rate + dt/2*k1, weights, bm, tau)
            k3 = _derivative(rate + dt/2*k2, weights, bm, tau)
            k4 = _derivative(rate + dt*k3, weights, b4, tau)
            rate += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        result[control + 1] = rate
    return result


def quadratic_peaks(times, values, *, prominence=20):
    """Refine peaks from saved 5ms rate endpoints; native states are not retained."""
    points, _ = find_peaks(values, prominence=prominence)
    refined = []
    for index in points:
        if index == 0 or index + 1 == len(values):
            continue
        before, center, after = values[index-1:index+2]
        denominator = before - 2*center + after
        offset = .5 * (before - after) / denominator if denominator else 0.
        if not np.isfinite(offset) or abs(offset) > 1:
            raise ValueError('Invalid quadratic E1 peak refinement')
        refined.append(float(times[index] + offset * (times[index+1]-times[index])))
    return np.asarray(refined)


def phase_trace(times, peaks):
    return np.interp(times, peaks, np.arange(len(peaks)), left=np.nan, right=np.nan)


def screening(times, rates, e1_columns, target_period, *, start=.5, stop=10):
    selected = (times >= start) & (times < stop)
    peaks = [quadratic_peaks(times, rates[:, column]) for column in e1_columns]
    rhythm=[]
    for index, column in enumerate(e1_columns):
        within = peaks[index][(peaks[index] >= start) & (peaks[index] < stop)]
        intervals=np.diff(within)
        width=float(np.percentile(rates[selected,column],95)
            -np.percentile(rates[selected,column],5))
        period=float(intervals.mean()) if len(intervals) else None
        cv=float(intervals.std()/intervals.mean()) if len(intervals) else None
        rhythm.append({'E1_column':int(column),'peak_count':len(within),
            'complete_cycles':len(intervals),'width':width,'period_s':period,
            'CV':cv,'screen':bool(width>=20 and len(intervals)>=8 and cv<=.3
                and abs(period/target_period-1)<=.1)})
    pairs={}
    for label,lead,lag in [('LF-RF',2,1),('LM-RM',5,3),('LH-RH',4,0)]:
        first,second=phase_trace(times,peaks[lead]),phase_trace(times,peaks[lag])
        mask=selected & np.isfinite(first) & np.isfinite(second)
        z=np.mean(np.exp(2j*np.pi*(first[mask]-second[mask]))) if mask.any() else None
        mean=float((np.angle(z)/(2*np.pi))%1) if z is not None else None
        distance=abs((mean-.5+.5)%1-.5) if mean is not None else None
        pairs[label]={'left_minus_right_phase':mean,
            'R':float(abs(z)) if z is not None else None,
            'antiphase_distance':distance,
            'screen':bool(z is not None and abs(z)>=.8 and distance<=.15)}
    return peaks,rhythm,pairs


def run(root, graph_path, out):
    out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    source=root/'e89-own-leg-native-recruitment'
    observed=root/'e90-own-leg-native-recruitment-body'
    reference=root/'e85-space-compatible-cadence-reference/commands.npz'
    target_period=float(np.load(reference)['period_s'])
    if not np.isclose(target_period,.4196813764904431,atol=1e-12,rtol=0):
        raise ValueError('Frozen E85 measured period required')
    spec={'experiment':'M3-E91-native-tonic-onset',
        'hypothesis':'Only three existing LM/RF/RH tonic INXXX464 native taus may delay the first E1 peaks by half the E85 reference period from their leading tripod.',
        'intervention':'Original E89 signed W, source weights, all 24core+6coupling+6tonic gain0, zero state and other 33taus held. Single bounded least-squares in three log taus [.02,E85 period], no external phase/teacher/preload/current injection.',
        'method':'Separate36x36 local W from actual E90 5ms currents; retain all other actual recurrent/drive/adaptation contribution as conditional background and linearly interpolate it at 250us native RK4 stages. Store only5ms control-boundary rates and estimate peak times by quadratic interpolation of those endpoints, not by native-rate sampling. Baseline original10s fidelity gates precede fit. One2.5s fit max60 actual ODE calls, then one selected10s check only if startup screen passes.',
        'limits':'0 full-network/physical advance; <=170 reduced neural seconds,<=180wallseconds/<=150fitwallseconds. No grids, no extra starts, F1-F8 unchanged.',
        'source_calibration_sha256':sha256(source/'calibration.npz'),
        'source_candidate_manifest_sha256':sha256(source/'candidate-neutral/manifest.json'),
        'source_candidate_state_sha256':sha256(source/'candidate-neutral/state.json'),
        'reference_commands_sha256':sha256(reference),
        'reference_period_s':target_period,
        'observed_signal_sha256':sha256(observed/'recruitment-signals.npz'),
        'observed_metadata_sha256':sha256(observed/'recruitment-metadata.npz'),
        'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    cal={key:value.copy() for key,value in np.load(source/'calibration.npz').items()}
    data=np.load(observed/'recruitment-signals.npz')
    meta=np.load(observed/'recruitment-metadata.npz')
    graph=GraphStore.load(graph_path)
    initial=StateStore.load(source/'candidate-neutral')
    local_ids=np.r_[cal['core_ids'],cal['coupling_ids'],cal['intrinsic_tonic_ids']]
    if len(local_ids)!=36 or len(np.unique(local_ids))!=36:
        raise ValueError('Unique original24core+6coupling+6tonic local cells required')
    lookup={int(value):i for i,value in enumerate(data['observed_ids'])}
    columns=np.array([lookup[int(value)] for value in local_ids])
    np.testing.assert_array_equal(meta['observed_ids'],data['observed_ids'])
    gains=meta['_cell_gain'][columns]
    if (np.any(gains!=0) or not np.allclose(meta['a'][columns],1)
            or not np.allclose(meta['threshold'][columns],7.5)
            or not np.allclose(meta['cap'][columns],200)):
        raise ValueError('Actual local gain0 and transfer physiology required')
    old_tau=meta['tau'][columns].astype(float)
    if not np.allclose(old_tau[-6:],.02):
        raise ValueError('Original tonic20ms tau required')
    tonic_expected={'rh':157097,'rf':125974,'lf':106670,
        'rm':100866,'lh':20156,'lm':40537}
    for group,leg in enumerate(('rh','rf','lf','rm','lh','lm')):
        if int(cal['intrinsic_tonic_ids'][group])!=tonic_expected[leg]:
            raise ValueError('Original tonic identity and source group order required')
    np.testing.assert_array_equal(initial['network']['rate'][local_ids],np.zeros(36))
    np.testing.assert_array_equal(initial['network']['adaptation'][local_ids],np.zeros(36))
    if initial['network']['tick']!=0:
        raise ValueError('Original zero-state neural tick required')
    actual=np.vstack((np.zeros(36),data['rate'][:,columns].astype(float)))
    drive=data['drive'][:,columns].astype(float)
    if np.any(drive!=0):
        raise ValueError('No direct local source drive allowed')
    current=np.vstack((np.zeros(36),(data['recurrent_current'][:,columns]
        +data['drive'][:,columns]-data['adaptation'][:,columns]).astype(float)))
    base,_=rate_weights(graph)
    base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,cal['motor_ids'])
    patched=patch_weights(base,cal['edge_ids'],cal['source_values'],cal['values'])
    weights=patched[local_ids][:,local_ids].toarray().astype(float)
    background=current-actual@weights.T
    times=np.arange(len(actual))*.005
    np.savez_compressed(out/'reduced-inputs.npz',time_s=times,local_ids=local_ids,
        observed_columns=columns,actual_rate=actual,actual_current=current,
        background=background,local_weights=weights,old_tau=old_tau,
        tonic_ids=cal['intrinsic_tonic_ids'])
    if time.perf_counter()-began>150:
        raise TimeoutError('Preparation exceeded budget before baseline')
    e1=np.array([int(cal['core_ids'][(cal['core_groups']==group)
        & (cal['core_cell_types']==0)][0]) for group in range(6)])
    e1_columns=np.array([np.flatnonzero(local_ids==value)[0] for value in e1])
    baseline=reduced_rk4(weights,background,old_tau,2000)
    reduced_s=10.
    np.savez_compressed(out/'baseline-reduced.npz',time_s=times,rate=baseline,
        actual_rate=actual,E1_ids=e1,E1_rate=baseline[:,e1_columns])
    width=np.percentile(actual,95,axis=0)-np.percentile(actual,5,axis=0)
    normalized=np.sqrt(np.mean((baseline-actual)**2,axis=0))/np.maximum(width,20)
    measured_peaks=[quadratic_peaks(times,actual[:,col]) for col in e1_columns]
    baseline_peaks=[quadratic_peaks(times,baseline[:,col]) for col in e1_columns]
    first3=[]
    for measured,predicted in zip(measured_peaks,baseline_peaks):
        first3.append(float(np.max(abs(measured[:3]-predicted[:3])))
            if len(measured)>=3 and len(predicted)>=3 else None)
    common=(times>=.5)&(times<9.5)
    predicted_phase=[phase_trace(times,peaks) for peaks in baseline_peaks]
    measured_phase=[phase_trace(times,peaks) for peaks in measured_peaks]
    common &= np.isfinite(predicted_phase).all(axis=0) & np.isfinite(measured_phase).all(axis=0)
    phase_errors=np.abs((np.array(predicted_phase)[:,common]
        -np.array(measured_phase)[:,common]+.5)%1-.5)
    max_phase_error=float(phase_errors.max()) if common.any() else None
    baseline_valid=bool(np.all(normalized<=.05) and max_phase_error is not None
        and max_phase_error<=.05 and all(value is not None and value<=.01 for value in first3))
    fidelity={'each_cell_normalized_RMS':normalized.tolist(),
        'maximum_normalized_RMS':float(normalized.max()),
        'E1_first_three_peak_max_time_error_s':first3,
        'common_phase_samples':int(common.sum()),
        'maximum_E1_phase_error_cycles':max_phase_error,
        'screen':baseline_valid,'reduced_neural_s':reduced_s,
        'wall_s':time.perf_counter()-began}
    write_json(out/'baseline-fidelity.json',fidelity)
    if not baseline_valid:
        result={'experiment':spec['experiment'],'validity':'conditional_surrogate_invalid',
            'hypothesis_verdict':'not_tested','adoption':'not_applicable',
            'baseline_fidelity':fidelity,'actual_full_neural_s':0,
            'actual_physical_s':0,'reduced_neural_s':reduced_s,
            'wall_s':time.perf_counter()-began}
        write_json(out/'result.json',result)
        print(json.dumps(result),flush=True)
        return
    # Only the three second-tripod tonic taus can change.
    selected=np.array([30+5,30+1,30+0])
    leading=np.array([3,2,4])
    lagging=np.array([5,1,0])
    history=[]
    calls=0
    fit_began=time.perf_counter()
    best=None

    def objective(log_values):
        nonlocal calls,reduced_s,best
        if calls>=60 or reduced_s+2.5>170 or time.perf_counter()-fit_began>150 or time.perf_counter()-began>150:
            raise TimeoutError('Bounded E91 fit call or wall allowance reached')
        tau=old_tau.copy()
        tau[selected]=np.exp(log_values)
        trajectory=reduced_rk4(weights,background,tau,500)
        calls+=1
        reduced_s+=2.5
        first=[]
        peaks_by_group=[]
        for column in e1_columns:
            points=quadratic_peaks(times[:501],trajectory[:,column])
            if not len(points):
                np.savez_compressed(out/'solver-failed-call.npz',tau=tau,rate=trajectory,
                    call=calls,E1_ids=e1)
                raise ValueError('Missing actual first E1 peak in reduced solver call')
            first.append(float(points[0]));peaks_by_group.append(points)
        residual=np.array([first[lag]-first[lead]-target_period/2
            for lag,lead in zip(lagging,leading)])
        entry={'call':calls,'tau':tau[selected].tolist(),
            'first_peaks_s':first,'peak_counts':[len(p) for p in peaks_by_group],
            'residual_s':residual.tolist(),'wall_s':time.perf_counter()-began}
        history.append(entry)
        write_json(out/'solver-history.json',history)
        np.savez_compressed(out/f'solver-call-{calls:02d}.npz',tau=tau,
            rate=trajectory,E1_rate=trajectory[:,e1_columns],first_peaks=first,
            residual=residual)
        score=float(np.linalg.norm(residual))
        if best is None or score<best[0]:
            best=(score,tau.copy(),trajectory.copy(),residual.copy(),peaks_by_group)
        return residual

    try:
        fitted=least_squares(objective,np.log(np.full(3,.02)),
            bounds=(np.log(np.full(3,.02)),np.log(np.full(3,target_period))),
            diff_step=1e-3,max_nfev=60,xtol=1e-8,ftol=1e-8,gtol=1e-8)
        solver_status={'success':bool(fitted.success),'message':fitted.message,
            'nfev':int(fitted.nfev),'cost':float(fitted.cost)}
    except (TimeoutError,ValueError) as error:
        solver_status={'success':False,'message':str(error),'error':type(error).__name__}
    write_json(out/'solver-status.json',solver_status)
    if best is None:
        raise ValueError('No finite reduced candidate observation')
    _,tau,trajectory,residual,peaks_by_group=best
    widths=np.percentile(trajectory[:501,e1_columns],95,axis=0)-np.percentile(
        trajectory[:501,e1_columns],5,axis=0)
    startup=bool(np.all(abs(residual)<=.05*target_period)
        and all(len(p)>=5 for p in peaks_by_group) and np.all(widths>=20)
        and np.isfinite(trajectory).all())
    np.savez_compressed(out/'selected-startup.npz',tau=tau,rate=trajectory,
        E1_rate=trajectory[:,e1_columns],residual=residual,width=widths)
    selected_validation=None
    if startup and reduced_s+10<=170 and time.perf_counter()-began<155:
        selected_rate=reduced_rk4(weights,background,tau,2000)
        reduced_s+=10
        np.savez_compressed(out/'selected-10s.npz',time_s=times,rate=selected_rate,
            E1_ids=e1,E1_rate=selected_rate[:,e1_columns])
        _,rhythms,pairs=screening(times,selected_rate,e1_columns,target_period)
        selected_validation={'rhythms':rhythms,'LR':pairs,
            'screen':all(row['screen'] for row in rhythms)
                and all(row['screen'] for row in pairs.values())}
        write_json(out/'selected-10s-screen.json',selected_validation)
    result={'experiment':spec['experiment'],'validity':'valid_conditional_diagnostic',
        'hypothesis_verdict':'conditional_startup_supported' if startup else 'not_supported',
        'adoption':'not_applicable','startup_screen':startup,
        'startup_residual_s':residual.tolist(),'startup_width':widths.tolist(),
        'solver':solver_status,'calls':calls,'selected_tau_s':tau[selected].tolist(),
        'selected_10s':selected_validation,'reduced_neural_s':reduced_s,
        'actual_full_neural_s':0,'actual_physical_s':0,
        'fit_wall_s':time.perf_counter()-fit_began,
        'wall_s':time.perf_counter()-began,
        'stored_rate_sampling_s':.005,'native_integration_dt_s':.00025,
        'peak_method':'quadratic interpolation of 5ms control-boundary rates',
        'interpretation_limit':'One local bounded first-peak solver failure does not prove all admissible tau values or later-cycle coordination cannot work.'}
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)
    if not startup or selected_validation is None or not selected_validation['screen']:
        return
    if time.perf_counter()-began>170:
        raise TimeoutError('Insufficient wall allowance for neutral candidate check')
    for cell,value in zip(local_ids[selected],tau[selected]):
        if cell in cal['tau_ids']:
            raise ValueError('Selected tonic tau already patched')
        cal['tau_ids']=np.r_[cal['tau_ids'],cell]
        cal['tau_values']=np.r_[cal['tau_values'],value]
    if len(cal['tau_ids'])!=35:
        raise ValueError('E89 32 tau entries plus three tonic patches required')
    np.savez_compressed(out/'calibration.npz',**cal)
    anatomy=json.loads((source/'muscle-provenance.json').read_text())
    anatomy['tonic_onset_hypothesis']={'cell_ids':local_ids[selected].tolist(),
        'tau_s':tau[selected].tolist(),'conditional_only':True}
    write_json(out/'muscle-provenance.json',anatomy)
    original=StateStore.load(root/'e32-core-adaptation/candidate/start')
    base_trial=EfficacyTrial(graph,original,cal['edge_ids'],cal['source_values'],
        cal['values'],zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial,cal,anatomy['muscles']),
        cal['tau_ids'],cal['tau_values'])
    try:
        neutral=session.snapshot()
        exact(original['body'],neutral['body'],'E91 original neutral body')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(original['network'][key],neutral['network'][key],
                'E91 original neutral neural '+key)
        StateStore.save(out/'candidate-neutral',neutral)
        clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out/'candidate-neutral'))
        try:
            exact(neutral,clone.snapshot(),'E91 neutral disk exact')
        finally:
            clone.close()
    finally:
        session.close()
    result['neutral_disk_exact']=True
    result['wall_s']=time.perf_counter()-began
    write_json(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','graph','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: '+str(args.out))
    try:
        run(args.root,args.graph,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete',
                'hypothesis_verdict':'not_tested','error':type(error).__name__,
                'message':str(error)})
        raise
