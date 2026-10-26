"""Audit all-leg waveform residuals and conditional transfer fidelity on recorded native currents."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_m3_RH_trial import relative_feet
from tools.fit_c_banc_m3_RH_recruitment import cycle_mean, periodic_sample
from tools.fit_c_banc_m3_rhythmic_recruitment import periodic_filter, phase_average


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'e77-native-recruitment-currents'
    reference=root/'e42-source-aligned-reference'
    cal_path=root/'e72-front-knee-delivery/calibration.npz'
    spec={'experiment':'M3-E82-native-all-leg-waveform-audit',
        'question':'Do already measured native currents support a faithful conditional motor transfer model, and where do all-seven-joint/3Dfoot waveforms differ from the reference?',
        'basis':'E79/E81pairedRHcoxa recruitment improved localactualcoxa/foot errors but failed original direction/10s. Geometryfit can make two joints compensate errors of the otherfive. E73original10s/F4/F5 remains developmentbaseline; inspect the complete within-leg residual before another local patch.',
        'data':'Read only unchangedE77/E73first5s exact437source/physical/native match; all-six completeownE1cycles in[3.5,5),at least6perleg. Align each E42command and measuredfoot reference to that leg E37E1peak offline. No model advances, fitting, new source weights, F6exposure or runtimephase use.',
        'transfer':'Mean recorded native endpoint J+drive-adaptation per sourcephase at native .00025grid; existing half-tanh threshold/a/cap, individualMNtau held-input periodicfilter, 5ms pooledreadout and10ms musclefilter, original anatomical moment/rest. Compare predicted wholecommand to actual cyclemean. This is a conditional approximation of recorded input, not fresh BANC dynamics.',
        'decision':'Report actual command/angle/foot residuals and conditional-transfer RMS separately. If aggregate same-leg conditional command RMS is<=25% of its reference discrepancy, the existing recorded-current surrogate is a reasonable next fitting tool; otherwise improve temporal/background modeling before fitting. This screen selects a method and does not pass gait or establish which joint causes yaw.',
        'limits':'CPU recordanalysis,0neural/physicaladvance,<=60wallseconds; preserve all rawfailed candidates/world/F1-F8/M2/UI.',
        'data_sha256':sha256(source/'recruitment-signals.npz'),
        'calibration_sha256':sha256(cal_path),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    data=np.load(source/'recruitment-signals.npz')
    meta=np.load(source/'recruitment-metadata.npz')
    cal=np.load(cal_path)
    wave=np.load(root/'e37-core-teacher/teaching-table.npz')
    rows=json.loads((source/'trace.json').read_text())['samples'][1:]
    refrows=json.loads((reference/'trace.json').read_text())['samples'][1:]
    times=data['time_s'];ref_times=np.array([row['time_s'] for row in refrows])
    q=np.array([row['joint_targets_rad'] for row in rows])
    angles=np.array([row['joint_angles_rad'] for row in rows])
    feet=relative_feet(rows)
    ref_q=phase_average(ref_times,np.array([row['joint_targets_rad'] for row in refrows]))
    ref_angles=phase_average(ref_times,np.array([row['joint_angles_rad'] for row in refrows]))
    ref_feet=phase_average(ref_times,relative_feet(refrows))
    index={int(value):i for i,value in enumerate(data['observed_ids'])}
    motor_columns=np.array([index[int(i)] for i in cal['motor_ids']])
    np.testing.assert_array_equal(data['observed_ids'],meta['observed_ids'])
    if np.any(data['adaptation'][:,motor_columns]!=0):
        raise ValueError('Declared nonadapting motor transfer required')
    outputs=[];saved={};source_order=('rh','rf','lf','rm','lh','lm')
    torch.set_num_threads(1)
    tau=meta['tau'][motor_columns]
    rms=lambda x:np.sqrt(np.mean(np.square(x),axis=0))
    for leg_index,leg in enumerate(LEGS):
        group=source_order.index(leg)
        ids=cal['core_ids'][(cal['core_groups']==group)&(cal['core_cell_types']==0)]
        if len(ids)!=1:
            raise ValueError('Unique original E1per anatomicalleg required')
        e1=int(ids[0]);col=index[e1]
        peaks,_=find_peaks(data['rate'][:,col],prominence=20,distance=20)
        peaks=peaks[(times[peaks]>=3.5)&(times[peaks]<5)]
        if len(peaks)<7:
            raise ValueError('At least6exposed complete sourcecycles perleg required')
        count=round(float(np.median(np.diff(times[peaks])))/.005)
        if not 26<=count<=32:
            raise ValueError('Unexpected sourcecycleperiod')
        native_count=count*20
        teacher_col=int(np.flatnonzero(wave['ids']==e1)[0])
        ref_peak=float(np.argmax(wave['rate'][12000:13200,teacher_col])/1200)
        phases=np.arange(count)/count+ref_peak
        target=periodic_sample(ref_q,phases)
        target_angles=periodic_sample(ref_angles,phases)
        target_feet=periodic_sample(ref_feet,phases)
        actual=cycle_mean(times,q,peaks,count)
        actual_angles=cycle_mean(times,angles,peaks,count)
        actual_feet=cycle_mean(times,feet,peaks,count)
        current=cycle_mean(times,(data['recurrent_current']+data['drive']-data['adaptation'])[:,motor_columns],peaks,native_count)
        response=np.maximum(meta['cap'][motor_columns]*np.tanh(meta['a'][motor_columns]
            *(current-meta['threshold'][motor_columns])/meta['cap'][motor_columns]),0)
        rate=np.empty_like(response)
        for value in np.unique(tau):
            selected=tau==value
            rate[:,selected]=periodic_filter(torch.tensor(response[:,selected]),.00025,float(value)).numpy()
        pooled=rate[::20]@cal['pool'].T
        active=periodic_filter(torch.tensor(pooled),.005,.01).numpy()
        predicted=cal['rest']+.02*active@cal['moment'].T
        own=np.arange(leg_index*7,(leg_index+1)*7)
        command_error=rms(actual[:,own]-target[:,own])
        transfer_error=rms(predicted[:,own]-actual[:,own])
        reference_rms=float(np.sqrt(np.mean((actual[:,own]-target[:,own])**2)))
        transfer_rms=float(np.sqrt(np.mean((predicted[:,own]-actual[:,own])**2)))
        outputs.append({'leg':leg,'source_E1':e1,'source_cycles':len(peaks)-1,
            'period_s':count*.005,'teacher_E1_peak_phase':ref_peak,
            'command_RMS_rad_by_joint':command_error.tolist(),
            'angle_RMS_rad_by_joint':rms(actual_angles[:,own]-target_angles[:,own]).tolist(),
            'tracking_RMS_rad_by_joint':rms(actual_angles[:,own]-actual[:,own]).tolist(),
            'physical_foot_RMS_xyz_mm':rms(actual_feet[:,leg_index]-target_feet[:,leg_index]).tolist(),
            'conditional_transfer_RMS_rad_by_joint':transfer_error.tolist(),
            'reference_command_RMS_rad':reference_rms,'conditional_transfer_RMS_rad':transfer_rms,
            'transfer_to_reference_RMS_ratio':transfer_rms/reference_rms})
        for key,value in {'command':actual,'target':target,'foot':actual_feet,
                'reference_foot':target_feet,'predicted':predicted,'motor_current':current,
                'source_rate':cycle_mean(times,data['rate'],peaks,native_count),
                'actual_muscle_activation':cycle_mean(times,data['muscle_activation'],peaks,count)}.items():
            saved[leg+'_'+key]=value
        if time.perf_counter()-began>55:
            raise TimeoutError('Bounded recordedwaveform audit exceeded walllimit')
    numerator=np.mean([row['conditional_transfer_RMS_rad']**2 for row in outputs])
    denominator=np.mean([row['reference_command_RMS_rad']**2 for row in outputs])
    ratio=float(np.sqrt(numerator/denominator))
    np.savez_compressed(out/'waveforms.npz',**saved,observed_ids=data['observed_ids'],
        motor_ids=cal['motor_ids'],calibration_moment=cal['moment'])
    result={'experiment':spec['experiment'],'validity':'valid_recorded_native_waveform_analysis',
        'hypothesis_verdict':'surrogate_sufficient_for_exploratory_fit' if ratio<=.25 else 'surrogate_needs_revision',
        'adoption':'not_applicable','milestone_outcome':'pending','legs':outputs,
        'aggregate_transfer_to_reference_RMS_ratio':ratio,
        'unsupported_moment_rows':np.flatnonzero(np.all(cal['moment']==0,axis=1)).tolist(),
        'anatomical_moment_rank':int(np.linalg.matrix_rank(cal['moment'])),
        'actual_neural_s':0,'actual_physical_s':0,'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.root,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete_analysis',
                'hypothesis_verdict':'not_tested','error':type(error).__name__,'message':str(error)})
        raise
