"""One bounded robustness direction with native complete-state branch checks."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_recurrent_mask import recurrent_mask
from tools.probe_c_banc_m3_closed_boundary import JOINTS
from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, FullCircuit
from tools.train_c_banc_m3_recurrent import command_metrics
from tools.verify_c_banc_m3_full_trial import CalibratedTrial, calibrated_weights


def run(source,candidate,reference,fidelity,perturbation,graph_path,out,*,recurrent=False):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E27-native-robust-feedback-direction',
        'hypothesis':'A direction from tracking plus finite-perturbation sensitivity can improve fixed-perturbation robustness without worsening clean motor tracking',
        'intervention':'Same existing motor input efficacy mask/W0/signs/physiology/full native BANC. Every theta runs its OWN0..0.5s prefix; then clean and fixed E26 +/- .01rate perturbations in fixed IDs/directions evolve continuously to2.2s. No body or motor teacher reset. Body-derived external input remains an explicit teacher.',
        'bounded_noise':'Experimental injection alone is clamped to[0,200] if candidate state reaches a boundary. Record clipping and actual rate delta; reject robustness interpretation if either initial perturb RMS is below99percent of baseline. No clipping of neural integration, commands or outputs.',
        'loss':'Window after injection and before2.2s: mean of3branch tracking MSE plus D=0.5*sum_plus_minus mean(((qperturbed-qclean)/SD)^2), both normalized by the SAME clean target perjoint SD floor.1. Add prior1e-4mean(theta-1)^2. No detached branch in coupled D.',
        'gradient':'Approximate .1s credit only; allbranch state values continue and common theta stays fixed. Shared operator prepare is idempotent for identical theta and rejects actual pending weight changes. Prefix gradients are omitted explicitly. Gradient supplies a direction, not an exact long-horizon derivative.',
        'search':'ONE initial direction g/(abs(g)+1e-8), at most3 projected trials with step.03,.01,.003333333333, each from ORIGINAL theta. Stop first full robustness screen; no new gradient or parameter-bound/gait-threshold search.',
        'screen':'Robust objective decreases and clean tracking RMS does not increase; both perturbation RMS inputs>=99percent baseline; clean6tibia command95-5 ranges>=90percent baseline; each +/- command difference versusclean decreases>=10percent at BOTH .7 and1.7s. Report LR command correlations separately. These are development diagnostics; original reconstruction .05/allkneeR2>=.5 and F1-F8 still required for actual gait.',
        'limits':'OneGPU,initial5.6+atmost3*5.6=22.4neuralmodelsec,<=285wallsec. Preserve incomplete raw output. No model adoption or live UI change.',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'perturbation_sha256':sha256(perturbation/'anchor.npz'),
        'fidelity_sha256':sha256(fidelity/'result.json'),'tool_sha256':sha256(__file__),
        'core_tool_sha256':sha256(Path('tools/train_c_banc_m3_full_circuit.py'))}
    if recurrent:
        spec.update(experiment='M3-E29-premotor-recurrent-robust-direction',
            hypothesis='Existing premotor VNC recurrence efficacy, with E20 motor input fixed, offers a stabilization direction missing from the rejected E27 motor-only mask',
            intervention='Same E20 motor-input weights FIXED. P is every source nonzero direct motor presynaptic VNC intrinsic cell, selected by metadata/topology only. Train only existing nonzero P-to-P edges with positive efficacy[.1,10], initially1. Full158706 native states participate. Body teacher, physiology, mechanics and E26 perturbations unchanged.',
            parameterization='Share efficacy by (post.cell_type or post.id, actual source edge sign, same/different/unknown soma relation). Same/different only if BOTH soma sides are known left/right. P7136, eligible725849, groups8547 independently counted before exposure. No new edges, signs, zeros, cell identities or external state generators.',
            grouping_limit='This is an engineering postsynaptic gain hypothesis, coarser than presynaptic-type-pair specificity. Original signs remain fixed but are model assumptions, not measured receptor physiology. Unknown soma relationships remain unknown.',
            fit_difference='E27 failed motor-only three directions. E09 trained a small core against a stored boundary. This run changes the trainable locus and differentiates full native recurrence. Only new P-P efficacy is regularized; the constant frozen E20 motor prior is omitted. Same screening and one-direction steps.',
            mask_tool_sha256=sha256(Path('tools/banc_m3_recurrent_mask.py')))
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'full-circuit.py').write_bytes(Path('tools/train_c_banc_m3_full_circuit.py').read_bytes())
    if recurrent:(out/'recurrent-mask.py').write_bytes(Path('tools/banc_m3_recurrent_mask.py').read_bytes())
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    cal=dict(np.load(candidate/'candidate.npz'));ref=dict(np.load(reference/'signals.npz'))
    inputs=dict(np.load(fidelity/'sensory-input.npz'));noise=dict(np.load(perturbation/'anchor.npz'))
    check=json.loads((fidelity/'result.json').read_text())
    if not check['solver_screen_passed'] or check['candidate_sha256']!=spec['candidate_sha256']:raise ValueError('Matching source fidelity required')
    s=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,s,cal,learned=False);positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Source weights changed')
        parameters={k:getattr(s.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        initial=s.network.snapshot();pool=s.adapter.pool.copy()
        if not np.all(initial['output_mask']==1):raise ValueError('Full outgoing mask required')
        if recurrent:learned_base,_=calibrated_weights(g,s,cal,learned=True)
    finally:s.close()
    fit_cal=dict(cal);fit_groups=cal['edge_parameters']
    if recurrent:
        population,positions,fit_groups,keys=recurrent_mask(g,base,cal['motor_ids'])
        if not np.array_equal(base.data[positions],learned_base.data[positions]):raise ValueError('Recurrent mask overlaps learned motor inputs')
        base=learned_base
        if (len(population),len(positions),len(keys))!=(7136,725849,8547):raise ValueError('Premotor mask differs from independently frozen topology')
        fit_cal['efficacy']=np.ones(len(keys),np.float32)
        np.savez_compressed(out/'recurrent-mask.npz',population=population,positions=positions,
            edge_parameters=fit_groups,source_weights=base.data[positions])
        write_json(out/'recurrent-parameters.json',{'keys':keys,'graph_hash':g.hash,
            'population_count':len(population),'edge_count':len(positions),'parameter_count':len(keys),
            'source_sign_counts':{str(sign):int(np.count_nonzero(np.sign(base.data[positions])==sign)) for sign in (-1,1)}})
    operator=AnatomicalOperator(base,np.asarray(positions),fit_groups,len(fit_cal['efficacy']),'cuda')
    model=FullCircuit(operator,fit_cal,pool,inputs['drive_ids'],initial,parameters)
    drive=torch.as_tensor(inputs['drive'],device='cuda');target=torch.as_tensor(ref['command'],dtype=torch.float32,device='cuda')
    times=ref['time_s'][1::2];mask=(times>.5+1e-9)&(times<2.2-1e-9)
    scale=torch.as_tensor(np.maximum(np.std(ref['command'][mask],axis=0),.1),dtype=torch.float32,device='cuda')
    ids=noise['perturbed_ids'];direction=noise['direction'];executed_controls=0;results={};accepted=None
    def rollout(theta,tag,gradient=False):
        nonlocal executed_controls
        arm=out/tag;arm.mkdir()
        with torch.no_grad():model.efficacy.copy_(torch.as_tensor(theta,device='cuda'))
        model.zero_grad(set_to_none=True)
        with torch.no_grad():_q,_r,anchor=model(drive[:100],dt=.00025,return_state=True)
        executed_controls+=100
        if gradient:
            for value,name in zip(anchor,('rate','adaptation','activation')):
                if not np.array_equal(value.cpu().numpy(),noise[name]):raise ValueError('Baseline full prefix differs from E26 anchor')
        states=[tuple(v.detach().clone() for v in anchor) for _ in range(3)]
        input_metrics=[]
        for branch,sign in ((1,-1),(2,1)):
            before=states[branch][0].clone()
            values=before[ids]+torch.as_tensor(sign*.01*direction,device='cuda')
            clipped=int(torch.count_nonzero((values<0)|(values>200)).cpu())
            states[branch][0][ids]=values.clamp(0,200)
            delta=(states[branch][0]-before).cpu().numpy()
            input_metrics.append({'clipped_neurons':clipped,'rate_rms':float(np.sqrt(np.mean(delta[ids].astype(float)**2)))})
        commands=[[] for _ in states];rates=[[] for _ in states];tracking=0.;sensitivity=0.
        try:
            for start in range(100,len(drive),20):
                if time.perf_counter()-began>280:raise TimeoutError('Robust probe wall limit')
                end=min(start+20,len(drive));qs=[];next_states=[]
                with torch.set_grad_enabled(gradient):
                    for branch,state in enumerate(states):
                        q,r,new=model(drive[start:end],dt=.00025,initial_state=state,return_state=True)
                        executed_controls+=end-start;qs.append(q);next_states.append(new)
                        commands[branch].append(q.detach().cpu().numpy());rates[branch].append(r.detach().cpu().numpy())
                    selected=torch.as_tensor(mask[start:end,None],device='cuda');denominator=int(mask.sum())*42
                    track=sum(((((q-target[start:end])/scale).square()*selected).sum()) for q in qs)/(3*denominator)
                    sensitive=sum(((((q-qs[0])/scale).square()*selected).sum()) for q in qs[1:])/(2*denominator)
                    loss=track+sensitive
                    tracking+=float(track.detach().cpu());sensitivity+=float(sensitive.detach().cpu())
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite robust loss')
                    if gradient:
                        loss.backward()
                        if not bool(torch.isfinite(model.efficacy.grad).all()):raise FloatingPointError('Nonfinite robust gradient')
                states=[tuple(v.detach() for v in state) for state in next_states]
                if not all(bool(torch.isfinite(v).all()) for state in states for v in state):raise FloatingPointError('Nonfinite full neural branch state')
                del qs,next_states,loss,q,r,new,track,sensitive
                if start%100==0:print(json.dumps({'arm':tag,'control':end,'wall_s':time.perf_counter()-began}),flush=True)
            penalty=1e-4*(model.efficacy-1).square().mean()
            data_gradient=model.efficacy.grad.detach().cpu().numpy().copy() if gradient else None
            if gradient:penalty.backward()
            qs=np.stack([np.concatenate(v) for v in commands]);rs=np.stack([np.concatenate(v) for v in rates])
            np.savez_compressed(arm/'signals.npz',commands=qs,rates=rs,efficacy=theta,
                gradient=model.efficacy.grad.detach().cpu().numpy() if gradient else np.array([]),
                data_gradient=data_gradient if gradient else np.array([]))
            if gradient:
                for index,name in ((1,'negative'),(2,'positive')):
                    old=dict(np.load(perturbation/(name+'.npz')))
                    if not np.array_equal(rs[index],old['rate']):raise ValueError('Shared-gradient branches differ from native E26')
            clean_metrics=command_metrics(qs[0],ref['command'][100:],mask[100:],JOINTS)
            knee=qs[0][:,JOINTS[:,5]];ranges=np.percentile(knee[mask[100:]],95,axis=0)-np.percentile(knee[mask[100:]],5,axis=0)
            lr=[float(np.corrcoef(knee[:,i],knee[:,i+3])[0,1]) if np.std(knee[:,i])*np.std(knee[:,i+3])>1e-12 else None for i in range(3)]
            differences=[]
            for branch in (1,2):
                horizons={}
                for seconds in (.1,.3,.7,1.7):
                    hi=round(seconds/.005);sl=slice(max(0,hi-20),hi)
                    horizons[str(seconds)]=float(np.sqrt(np.mean((qs[branch,sl].astype(float)-qs[0,sl])**2)))
                differences.append(horizons)
            result={'tracking_loss':tracking,'sensitivity_loss':sensitivity,'prior':float(penalty.detach().cpu()),
                'objective':tracking+sensitivity+float(penalty.detach().cpu()),'clean':clean_metrics,
                'clean_knee_command_range_rad':ranges.tolist(),'clean_lr_command_correlation':lr,
                'input_perturbation':input_metrics,'command_difference_rms_by_horizon':differences,
                'wall_s':time.perf_counter()-began}
            write_json(arm/'result.json',result);print(json.dumps({'arm':tag,**result}),flush=True)
            return result,model.efficacy.grad.detach().cpu().numpy().copy() if gradient else None
        except Exception as exc:
            np.savez_compressed(arm/'partial-signals.npz',efficacy=theta,
                **{f'command_{i}':np.concatenate(v) if v else np.empty((0,42)) for i,v in enumerate(commands)})
            write_json(arm/'execution-error.json',{'type':type(exc).__name__,'message':str(exc),'adoption':'not_applicable'})
            raise
    stop='all3trials_rejected'
    try:
        baseline,gradient=rollout(fit_cal['efficacy'],'baseline-gradient',True);results['baseline']=baseline
        direction_step=gradient/(np.abs(gradient)+1e-8)
        for index,step in enumerate((.03,.01,.003333333333)):
            theta=np.clip(fit_cal['efficacy']-step*direction_step,.1,10).astype(np.float32)
            result,_=rollout(theta,f'trial-{index}')
            checks={'objective_decreased':result['objective']<baseline['objective'],
                'clean_tracking_not_worse':result['clean']['rms_rad']<=baseline['clean']['rms_rad'],
                'perturbation_retained':all(a['rate_rms']>=.99*b['rate_rms'] for a,b in zip(result['input_perturbation'],baseline['input_perturbation'])),
                'clean_amplitude_retained':bool(np.all(np.asarray(result['clean_knee_command_range_rad'])>=.9*np.asarray(baseline['clean_knee_command_range_rad']))),
                'both_perturbations_improved':all(result['command_difference_rms_by_horizon'][b][t]<=.9*baseline['command_difference_rms_by_horizon'][b][t] for b in (0,1) for t in ('0.7','1.7'))}
            result.update(step_size=step,screen=checks,screen_passed=all(checks.values()))
            write_json(out/f'trial-{index}/screen.json',result);results[f'trial-{index}']=result
            if all(checks.values()):
                exported=dict(cal)
                if recurrent:
                    exported.update(recurrent_positions=positions,recurrent_edge_parameters=fit_groups,
                        recurrent_source_weights=base.data[positions],recurrent_efficacy=theta,
                        recurrent_population=population)
                else:
                    exported['efficacy']=theta;exported['incoming_data']=cal['incoming_source_data']*theta[cal['edge_parameters']]
                np.savez_compressed(out/'candidate.npz',**exported);accepted=index;stop='first_robustness_screen_pass';break
    except (TimeoutError,FloatingPointError,ValueError,RuntimeError) as exc:stop=type(exc).__name__+': '+str(exc)
    final={'experiment':spec['experiment'],'run_status':'completed' if accepted is not None or len(results)==4 else 'bounded_incomplete',
        'adoption':'defer_full_student' if accepted is not None else 'reject','milestone_outcome':'pending','stop':stop,
        'accepted_trial':accepted,'neural_model_s':executed_controls*.005,'wall_s':time.perf_counter()-began,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'results':results}
    write_json(out/'result.json',final);print(json.dumps(final),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','reference','fidelity','perturbation','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recurrent',action='store_true')
    a=p.parse_args();run(a.source,a.candidate,a.reference,a.fidelity,a.perturbation,a.graph,a.out,recurrent=a.recurrent)
