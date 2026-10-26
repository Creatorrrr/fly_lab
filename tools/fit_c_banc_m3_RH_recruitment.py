"""Fit paired RH coxa recruitment through existing source paths on exposed native data."""

import argparse
import json
import math
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.fit_c_banc_m3_rhythmic_recruitment import periodic_filter, phase_average
from tools.verify_c_banc_observation import exact


def cycle_mean(times, values, peaks, count):
    values = np.asarray(values)
    flat = values.reshape(len(times), -1)
    phases = np.arange(count) / count
    cycles = []
    for start, end in pairwise(peaks):
        query = times[start] + phases * (times[end] - times[start])
        cycles.append(np.column_stack([np.interp(query, times, v) for v in flat.T]))
    return np.mean(cycles, axis=0).reshape((count,) + values.shape[1:])


def periodic_sample(values, phases):
    values = np.asarray(values)
    flat = values.reshape(len(values), -1)
    grid = np.arange(len(values) + 1) / len(values)
    return np.column_stack([np.interp(phases % 1, grid, np.r_[v, v[0]])
        for v in flat.T]).reshape((len(phases),) + values.shape[1:])


def run(root, graph_path, out, *, positive_relay=False):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e72-front-knee-delivery'
    measured = root / 'e77-native-recruitment-currents'
    reference = root / 'e42-source-aligned-reference'
    spec = {'experiment': 'M3-E78-RH-paired-source-recruitment',
        'hypothesis': 'Existing RHtonic and E1-inhibitory relay paths can recruit the two silent RHcoxa muscles jointly and improve the within-leg foot path. E77actual total MNcurrent never reaches threshold in the inspected late window; unrelated muscles, phase coordination and actual direction may still limit walking.',
        'paths': [['720575941626500746','720575941526736740','720575941562065920'],
            ['720575941552245822','720575941518807468','720575941562065920']],
        'intervention': 'Existing IN01A015 retains its original positiveDNinput1.62; existingIN16B016 uses E1input1.1707944, previously used for the LMrelay. Other incoming nonzero edges of only these two cells attenuatedx.001; both adaptation8to0. All tau/core/otherphysiology preserved. Learn only existing8sameRHsource-to-twoMN positiveefficacies[.001,10] and two positive moment-row36/37 gains[.1,4]. No newedges, source signs flipped, rest/body/world/start/input changes or navigation-port intervention.',
        'conditional_data': 'Original unchanged E77 completeRH E1cycles wholly in[3.5,5); at least6cycles. Align reference E42command labels with E37RH E1peak only offline. Median observed cycle sets trainingperiod; average rate/current/background in neural phase. Newrelay field=originalmeasurednondesignatedcurrentx.001 plus designatedinput; ignore changed unobserved feedback conditionally. Half-tanh/20ms held-input periodicfilter and10ms musclefilter approximate the native RK dynamics. No model advances during fitting.',
        'objective': 'Equal normalized RHroll/pitch command error and RHbody-relative footxyz error using frozen measured kinematic contexts, plus1e-5 original-efficacy prior. Other40commands stay at measured neural-cycle averages. Fixedrest and anatomical muscle ratios preserved. Joint/geometry loss is conditional and does not certify traction or yaw.',
        'warm_start': 'Positive two-muscle leastsquares inversion of reference coxa commands, then bounded source-current leastsquares using half-tanh inverse. This avoids an all-silent zero-gradient initialization. One logpositive CUDAAdam lr.03 max750updates, stopfit by120wallseconds; no phase/gain/bounds grid.',
        'prediction': 'Native wholeBANC test should produce each RHcoxa muscle activation p95-p05>=1 rateunit and lower matched RHcoxa/foot reference error; directional relevance separately requires>=30%less absolute yaw in same[.5,2.2) thanE73 with healthy completion. Original10s F3-F5 remain the actual gait criteria; preserving or meeting a local screen is not final adoption.',
        'validity': 'Original graph paths/signs/no nav20 overlap; exactunchanged allzero neural and originalneutral body; existing-source patch bounds; finite filter/fit values; shadowgeometry leaves bodyexact; wholecandidate diskexact. PreserveE73andallfailedruns/F6unexposed/M2UI.',
        'limits': 'OneGPU,<=180wallseconds fit+preparation,0experimentalneural/physicaladvance. Followup one nativefullBANC/body10s+.1clone<=285wallseconds; no F6exposure.',
        'source_sha256': sha256(source/'calibration.npz'),
        'observation_sha256': sha256(measured/'recruitment-signals.npz'),
        'reference_sha256': sha256(reference/'commands.npz'), 'tool_sha256': sha256(__file__)}
    if positive_relay:
        spec.update(experiment='M3-E80-RH-positive-source-recruitment',
            hypothesis='The reference pleural movement needs an E1-following positive component. E78E1-following inhibitoryoutput was minimized and E79pleuralactivationwidth.315 failed while other local errors improved. The existing excitatory IN01A016 path supplies a missing sign of the same rhythm; actual direction may remain limited by other legs.',
            paths=[spec['paths'][0],['720575941552245822','720575941390358788','720575941562065920']],
            intervention='Start again from unchanged E72/E77, preserving E78/E79failures. Replace the proposed IN16B016 intermediary with existing positiveIN01A016. Its original E1edge+2.28 is set to same effective1.1707944; its other nonzero incomingedgesx.001 and adaptation8to0, tauunchanged20ms. Same IN01A015 tonic preparation. Learn six existing sameRHsource-to-twoMN positiveefficacies and same two moment gains with exactly the E78bounds/objective/window/optimizer. No IN16B016 patch retained, no newedge, rest/time/phase/body/world change.',
            prior_failed_trial_sha256=sha256(root/'e79-RH-paired-recruitment-body/recruitment-quality-summary.json'),
            source_expression_sha256=sha256(root/'e78-RH-paired-source-recruitment/phase-expression-analysis.json'))
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    cal = dict(np.load(source/'calibration.npz'))
    anatomy = json.loads((source/'muscle-provenance.json').read_text())
    data = np.load(measured/'recruitment-signals.npz')
    indices = {int(value): i for i,value in enumerate(data['observed_ids'])}
    dn, tonic, pleural = map(int,graph.resolve(['flywire:banc:888:'+rid for rid in spec['paths'][0]]))
    e1, rhythmic, check_mn = map(int,graph.resolve(['flywire:banc:888:'+rid for rid in spec['paths'][1]]))
    sternal = int(graph.resolve(['flywire:banc:888:720575941503422187'])[0])
    if check_mn != pleural:
        raise ValueError('Two source paths must terminate on original RHpleural')
    encoder = BancTargetEncoder(graph,forward_drive=5000)
    if np.intersect1d([tonic,rhythmic],np.r_[cal['input_ids'],encoder.ports]).size:
        raise ValueError('Only previously unmodified nonnavigation source relays allowed')
    matrix,_ = rate_weights(graph)
    matrix = encoder.calibrate_weights(matrix,cal['motor_ids'])
    expected = (1.62,1.02,2.28,.18) if positive_relay else (1.62,1.02,.57,-1.41)
    actual = (matrix[tonic,dn],matrix[pleural,tonic],matrix[rhythmic,e1],matrix[pleural,rhythmic])
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=0)
    inputs = np.r_[cal['core_ids'][cal['core_groups']==0],cal['intrinsic_tonic_ids'][0],
        cal['coupling_ids'][5],tonic,rhythmic].astype(np.int64)
    motors = np.array([pleural,sternal],dtype=np.int64)
    original = matrix[motors][:,inputs].toarray().astype(float)
    present = original != 0
    if present.sum() != (6 if positive_relay else 7):
        raise ValueError('Declared existing RHsource motor edge count differs')
    current_matrix = patch_weights(matrix,cal['edge_ids'],cal['source_values'],cal['values'])
    current_weights = current_matrix[motors][:,inputs].toarray().astype(float)
    times = data['time_s']
    peaks,_ = find_peaks(data['rate'][:,indices[e1]],prominence=20,distance=20)
    peaks = peaks[(times[peaks]>=3.5)&(times[peaks]<5)]
    if len(peaks)<7:
        raise ValueError('At least6 complete exposedRH cycles required')
    period = float(np.median(np.diff(times[peaks])))
    ncontrol = round(period/.005)
    if not 26<=ncontrol<=32:
        raise ValueError('Unexpected sourcecycle period')
    native_count = ncontrol*20
    phase = np.arange(ncontrol)/ncontrol
    take = lambda key,cells: cycle_mean(times,data[key][:,[indices[int(i)] for i in cells]],peaks,native_count)
    measured_sources = take('rate',inputs)
    observed_current = take('recurrent_current',motors)
    background = observed_current - measured_sources@current_weights.T
    # Conditional relay prediction keeps the observed nondesignated field explicit.
    dn_rate = take('rate',[dn])[:,0]
    e1_rate = take('rate',[e1])[:,0]
    tonic_current = .001*(take('recurrent_current',[tonic])[:,0]-float(matrix[tonic,dn])*dn_rate)
    tonic_current += float(matrix[tonic,dn])*dn_rate
    rhythmic_current = .001*(take('recurrent_current',[rhythmic])[:,0]-float(matrix[rhythmic,e1])*e1_rate)
    rhythmic_current += 1.1707944*e1_rate
    predicted_relays = []
    for values in (tonic_current,rhythmic_current):
        target = np.maximum(200*np.tanh((values-7.5)/200),0)
        predicted_relays.append(periodic_filter(torch.tensor(target),.00025,.02).numpy())
    sources = measured_sources.copy()
    sources[:,-2:] = np.column_stack(predicted_relays)
    ref = np.load(reference/'commands.npz')
    wave = np.load(root/'e37-core-teacher/teaching-table.npz')
    e1column = int(np.flatnonzero(wave['ids']==e1)[0])
    reference_peak = float(np.argmax(wave['rate'][12000:13200,e1column])/1200)
    target = periodic_sample(phase_average(ref['time_s'],ref['command']),phase+reference_peak)
    trace = json.loads((measured/'trace.json').read_text())['samples'][1:]
    actual_commands = np.array([row['joint_targets_rad'] for row in trace])
    base = cycle_mean(times,actual_commands,peaks,ncontrol)
    active = cycle_mean(times,data['muscle_activation'],peaks,ncontrol)
    if np.max(abs(active[:,[82,83]]))!=0:
        raise ValueError('Source RHpair must be actually silent in this fitwindow')
    source_phases = times[peaks[0]]+phase*(times[peaks[1]]-times[peaks[0]])
    qpos = np.array([trace[int(np.argmin(abs(times-t)))]['qpos'] for t in source_phases])
    saved = StateStore.load(root/'e32-core-adaptation/candidate/start')
    body = new_body(saved)
    before = body.snapshot()
    geometry = StaticFootGeometry(body,qpos)
    feet_reference = geometry.evaluate(target)
    qscale = np.maximum(target[:,[36,37]].std(axis=0),.025)
    fscale = np.maximum(feet_reference[:,5].std(axis=0),.05)
    moment = cal['moment'][[36,37]][:,[82,83]]
    warm = np.array([lsq_linear(.02*moment,row-cal['rest'][[36,37]],
        bounds=(0,199),method='bvls',tol=1e-10).x for row in target[:,[36,37]]])
    warm_current = 7.5+200*np.arctanh(warm/200)
    initial = np.full_like(original,.001)
    for row in range(2):
        selected = np.flatnonzero(present[row])
        design = sources[::20,selected]*original[row,selected]
        fit = lsq_linear(np.vstack((design/200,.001*np.eye(len(selected)))),
            np.r_[(warm_current[:,row]-background[::20,row])/200,.001*np.ones(len(selected))],
            bounds=(.001,10),method='bvls',tol=1e-10,max_iter=300)
        if not fit.success:
            body.close();raise ValueError('Current-space warm fit failed')
        initial[row,selected] = fit.x
    np.savez_compressed(out/'training-inputs.npz',input_ids=inputs,motor_ids=motors,
        sources=sources,measured_sources=measured_sources,background=background,
        original_weights=original,present=present,initial=initial,warm_muscles=warm,
        target=target,base_command=base,qpos=qpos,reference_peak_phase=reference_peak,
        reference_foot=feet_reference,period_s=ncontrol*.005,peaks_s=times[peaks],
        predicted_relays=np.column_stack(predicted_relays))
    if not torch.cuda.is_available():
        body.close();raise RuntimeError('Actual CUDA fitting required')
    torch.set_num_threads(1)
    tensor = lambda value:torch.as_tensor(value,dtype=torch.float64,device='cuda')
    source_t,background_t,weights_t = map(tensor,(sources,background,original))
    target_t,base_t,feet_t = map(tensor,(target,base,feet_reference))
    qscale_t,fscale_t = map(tensor,(qscale,fscale))
    moment_t,rest_t = map(tensor,(moment,cal['rest'][[36,37]]))
    parameter = torch.nn.Parameter(tensor(np.log(np.r_[initial[present],[1.,1.]])))
    mask = torch.as_tensor(present,device='cuda')
    optimizer = torch.optim.Adam([parameter],lr=.03)
    def predict():
        values = parameter.exp()
        factors = torch.zeros_like(weights_t).masked_scatter(mask,values[:-2])
        current = source_t@(weights_t*factors).T+background_t
        rate = periodic_filter(torch.relu(200*torch.tanh((current-7.5)/200)),.00025,.02)
        activation = periodic_filter(rate[::20],.005,.01)
        command = base_t.clone()
        command[:,[36,37]] = rest_t+.02*(activation@moment_t.T)*values[-2:]
        feet = geometry.tensor(command)
        loss = .5*(((command[:,[36,37]]-target_t[:,[36,37]])/qscale_t).square().mean()
            +((feet[:,5]-feet_t[:,5])/fscale_t).square().mean())
        loss += 1e-5*(values[:-2]-1).square().mean()
        return loss,command,feet,rate,activation
    best,best_loss,history = None,math.inf,[]
    try:
        for step in range(751):
            optimizer.zero_grad(set_to_none=True)
            loss,*_ = predict()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Finite RHpaired recruitment loss required')
            if value<best_loss:
                best_loss=value;best=parameter.detach().clone()
            if step%100==0 or step==750:
                history.append({'step':step,'loss':value,'wall_s':time.perf_counter()-began})
                print(json.dumps(history[-1]),flush=True)
            if step==750 or time.perf_counter()-began>120:
                break
            loss.backward();optimizer.step()
            with torch.no_grad():
                parameter[:-2].clamp_(math.log(.001),math.log(10))
                parameter[-2:].clamp_(math.log(.1),math.log(4))
        with torch.no_grad():
            parameter.copy_(best)
            loss,command,feet,rate,activation=predict()
        parameters=parameter.exp().detach().cpu().numpy()
        exact(before,body.snapshot(),'Read-only RHshadowgeometry')
    finally:
        body.close()
    factors=np.zeros_like(original);factors[present]=parameters[:-2]
    changed=cal['values'].copy()
    for row,post in enumerate(motors):
        edges=np.arange(matrix.indptr[post],matrix.indptr[post+1])
        for col in np.flatnonzero(present[row]):
            edge=int(edges[matrix.indices[edges]==inputs[col]][0])
            where=np.flatnonzero(cal['edge_ids']==edge)
            if len(where)!=1:
                raise ValueError('Existing calibrated RHmotor input required')
            changed[where[0]]=np.float32(matrix.data[edge]*factors[row,col])
    appended=[];new_values=[]
    for relay,pre,effective in ((tonic,dn,float(matrix[tonic,dn])),(rhythmic,e1,1.1707944)):
        edges=np.arange(matrix.indptr[relay],matrix.indptr[relay+1])
        edges=edges[matrix.data[edges]!=0]
        if np.intersect1d(edges,cal['edge_ids']).size:
            raise ValueError('New source relay must not overlap existing patch ownership')
        values=matrix.data[edges]*np.float32(.001)
        values[matrix.indices[edges]==pre]=np.float32(effective)
        appended.extend(edges);new_values.extend(values)
    new_edges=np.asarray(appended,dtype=np.int64)
    cal.update(edge_ids=np.r_[cal['edge_ids'],new_edges],
        source_values=np.r_[cal['source_values'],matrix.data[new_edges]],
        values=np.r_[changed,np.asarray(new_values,dtype=np.float32)],
        zero_adaptation_ids=np.union1d(cal['zero_adaptation_ids'],[tonic,rhythmic]),
        input_ids=np.r_[cal['input_ids'],tonic,rhythmic],RH_recruitment_parameters=parameters)
    patch_weights(matrix,cal['edge_ids'],cal['source_values'],cal['values'])
    for row,gain in zip((36,37),parameters[-2:]):
        cal['moment'][row]*=gain
    hypothesis={'path_nodes':[[graph.nodes[int(i)] for i in graph.resolve(['flywire:banc:888:'+r for r in path])] for path in spec['paths']],
        'original_path_weights':list(map(float,actual)),'input_ids':inputs.tolist(),
        'original_motor_weights':original.tolist(),'efficacies':factors.tolist(),
        'RH_moment_row_gains':parameters[-2:].tolist(),'rest_unchanged':True,
        'same_source_graph':True,'runtime_teacher':False,'biological_physiology':'unvalidated engineering hypothesis'}
    anatomy['RH_paired_recruitment_hypothesis']=hypothesis
    anatomy['input_nodes'] += [graph.nodes[tonic],graph.nodes[rhythmic]]
    write_json(out/'muscle-provenance.json',anatomy)
    write_json(out/'recruitment-hypothesis.json',hypothesis)
    np.savez_compressed(out/'calibration.npz',**cal)
    command_np,feet_np=command.cpu().numpy(),feet.cpu().numpy()
    np.savez_compressed(out/'prediction.npz',command=command_np,target=target,
        foot=feet_np,reference_foot=feet_reference,motor_rate=rate.cpu().numpy(),
        muscle_activation=activation.cpu().numpy())
    base_trial=EfficacyTrial(graph,saved,cal['edge_ids'],cal['source_values'],cal['values'],
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial,cal,anatomy['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        neutral=session.snapshot()
        exact(saved['body'],neutral['body'],'Original RHcandidate neutralbody')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(saved['network'][key],neutral['network'][key],'Original neural '+key)
        StateStore.save(out/'candidate-neutral',neutral)
    finally:
        session.close()
    clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,StateStore.load(out/'candidate-neutral'))
    try:
        exact(neutral,clone.snapshot(),'Entire RHcandidate diskstate')
    finally:
        clone.close()
    result={'experiment':spec['experiment'],'validity':'valid_conditional_fit_and_neutral_candidate',
        'hypothesis_verdict':'inconclusive_until_native_body','adoption':'not_applicable','milestone_outcome':'pending',
        'conditional_loss':float(loss),'source_complete_cycles':len(peaks)-1,'period_s':ncontrol*.005,
        'parameters':parameters.tolist(),'conditional_coxa_RMS_rad':np.sqrt(np.mean((command_np[:,[36,37]]-target[:,[36,37]])**2,axis=0)).tolist(),
        'conditional_RH_foot_RMS_xyz_mm':np.sqrt(np.mean((feet_np[:,5]-feet_reference[:,5])**2,axis=0)).tolist(),
        'baseline_coxa_RMS_rad':np.sqrt(np.mean((base[:,[36,37]]-target[:,[36,37]])**2,axis=0)).tolist(),
        'source_body_exact':True,'neutral_disk_exact':True,'device':str(parameter.device),
        'actual_neural_s':0,'actual_physical_s':0,'wall_s':time.perf_counter()-began,'history':history}
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--positive-relay',action='store_true')
    args=parser.parse_args()
    try:
        run(args.root,args.graph,args.out,positive_relay=args.positive_relay)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete_preparation',
                'hypothesis_verdict':'not_tested','error':type(error).__name__,'message':str(error)})
        raise
