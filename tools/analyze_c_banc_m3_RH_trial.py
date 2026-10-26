"""Compare exposed paired-RH recruitment and physical outcomes without model advances."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.fit_c_banc_m3_RH_recruitment import cycle_mean, periodic_sample
from tools.fit_c_banc_m3_rhythmic_recruitment import phase_average


def relative_feet(rows):
    origin = np.array([row['body']['position'] for row in rows])
    basis = np.array([row['body']['basis'] for row in rows])
    tips = np.array([[row['body']['legs'][leg][-1] for leg in LEGS] for row in rows])
    return np.einsum('tlx,txj->tlj', tips-origin[:,None,:], basis)


def measured(path, reference):
    data = np.load(path/'recruitment-signals.npz')
    rows = json.loads((path/'trace.json').read_text())['samples'][1:]
    times = data['time_s']
    column = int(np.flatnonzero(data['observed_ids']==91167)[0])
    peaks,_ = find_peaks(data['rate'][:,column],prominence=20,distance=20)
    peaks = peaks[(times[peaks]>=3.5)&(times[peaks]<5)]
    if len(peaks)<7 or times[-1]<5:
        return {'validity':'insufficient_late_observed_cycles'}
    count = round(float(np.median(np.diff(times[peaks])))/.005)
    phases = np.arange(count)/count+.5
    command = cycle_mean(times,np.array([row['joint_targets_rad'] for row in rows]),peaks,count)
    foot = cycle_mean(times,relative_feet(rows),peaks,count)
    ref_rows = reference['samples'][1:]
    ref_times = np.array([row['time_s'] for row in ref_rows])
    ref_command = periodic_sample(phase_average(ref_times,
        np.array([row['joint_targets_rad'] for row in ref_rows])),phases)
    ref_foot = periodic_sample(phase_average(ref_times,relative_feet(ref_rows)),phases)
    selected = (times>=3.5)&(times<5)
    width = np.diff(np.percentile(data['muscle_activation'][selected][:,[82,83]],[5,95],axis=0),axis=0)[0]
    rms = lambda x:np.sqrt(np.mean(x*x,axis=0)).tolist()
    return {'validity':'valid_exposed_cycle_average','cycles':len(peaks)-1,
        'window_s':[3.5,5],'period_s':count*.005,'RH_muscle_p95_minus_p05':width.tolist(),
        'RH_coxa_command_RMS_rad':rms(command[:,[36,37]]-ref_command[:,[36,37]]),
        'RH_physical_foot_RMS_xyz_mm':rms(foot[:,5]-ref_foot[:,5]),
        'foot_definition':'Actual tarsus5 relative to thorax in body axes; sourceE1-peak phase alignment only for offline comparison.'}


def yaw_change(path):
    rows = json.loads((path/'trace.json').read_text())['samples']
    times = np.array([row['time_s'] for row in rows])
    if times[-1]<2.2:
        return None
    yaw = np.unwrap([row['body']['yaw'] for row in rows])[(times>=.5)&(times<2.2)]
    return float(yaw[-1]-yaw[0])


def run(root, trial):
    baseline = root/'e77-native-recruitment-currents'
    reference = json.loads((root/'e42-source-aligned-reference/trace.json').read_text())
    a,b = measured(baseline,reference),measured(trial,reference)
    yaw_a,yaw_b = yaw_change(baseline),yaw_change(trial)
    raw = json.loads((trial/'result.json').read_text())
    physical,quality = raw['physical'],raw['quality']
    ratio = abs(yaw_b)/abs(yaw_a) if yaw_b is not None else None
    local = (b['validity']=='valid_exposed_cycle_average'
        and min(b['RH_muscle_p95_minus_p05'])>=1
        and all(x<y for x,y in zip(b['RH_coxa_command_RMS_rad'],a['RH_coxa_command_RMS_rad']))
        and all(x<y for x,y in zip(b['RH_physical_foot_RMS_xyz_mm'],a['RH_physical_foot_RMS_xyz_mm'])))
    result = {'experiment':raw['experiment'],'tool_sha256':sha256(__file__),
        'baseline':a,'candidate':b,'early_yaw_window_s':[.5,2.2],
        'baseline_yaw_rad':yaw_a,'candidate_yaw_rad':yaw_b,'absolute_yaw_ratio':ratio,
        'local_recruitment_screen':local,'direction_screen':ratio is not None and ratio<=.7 and physical['stop_reason']=='duration',
        'physical':physical,'LR_observed_prefix':{key:quality['pair_phase'][key]
            for key in ('lf-rf','lm-rm','lh-rh')},
        'knee_width_deg':{key:value['p95']-value['p05'] for key,value in quality['knee_interior_deg'].items()},
        'validity':'valid_actual_full_BANC_body_trial_analysis',
        'hypothesis_verdict':'supported_for_local_recruitment_only' if local else 'not_supported',
        'adoption':'reject' if physical['stop_reason']!='duration' else 'defer',
        'milestone_outcome':'pending','actual_neural_s':0,'actual_physical_s':0,
        'interpretation':'Late local comparisons use exposed same neural-phase cycles. Prefix gait statistics are not full10s or independentF6 evidence; finalF1-F8 unchanged.'}
    write_json(trial/'recruitment-quality-summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.trial)
