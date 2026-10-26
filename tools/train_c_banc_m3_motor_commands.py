"""Learn anatomical pooled commands with conditional neural-state inputs.

No neuron is assigned a new output role and no teacher rate enters the motor
state update directly. Recorded presynaptic currents remain a teaching boundary;
an explicitly reported recurrence test replaces motor inputs by student rates.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.fit_c_banc_m3_motor_currents import grouped_design
from tools.train_c_banc_m3_recurrent import command_metrics


class MotorCommandNetwork(torch.nn.Module):
    def __init__(self, efficacy, indices, internal_weights, internal_groups, pool, moment, rest, device='cpu'):
        super().__init__()
        self.efficacy=torch.nn.Parameter(torch.as_tensor(efficacy,dtype=torch.float32,device=device).clone())
        for name,value in (('indices',indices),('internal_weights',internal_weights),
                ('internal_groups',internal_groups),('pool',pool),('moment',moment),('rest',rest)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.long if 'groups' in name or name=='indices' else torch.float32,device=device))

    def forward(self,design,drives,*,student_motor=False,native=False,retain_current_gradient=False,truncate_controls=0):
        current=torch.bmm(design,self.efficacy[self.indices].unsqueeze(2)).squeeze(2).T
        current=current+drives
        if retain_current_gradient:current.retain_grad()
        w=self.internal_weights*self.efficacy[self.internal_groups] if student_motor else None
        r=torch.zeros(self.pool.shape[1],device=current.device)
        z=torch.zeros_like(r);activation=torch.zeros(self.pool.shape[0],device=current.device)
        previous=torch.zeros_like(r)
        commands,rates,targets=[],[],[]
        def rhs(rr,zz,external):
            total=external if w is None else external+w@rr
            target=torch.tanh(total-7.5/200-zz).clamp_min(0)
            return (target-rr)/.02,(8*rr-zz)/.3,target
        for tick in range(current.shape[0]//2):
            half,end=current[2*tick],current[2*tick+1]
            if native:
                stages=[]
                for left,right in ((previous,half),(half,end)):
                    for j in range(10):
                        stages.append((left+(right-left)*(j/10),left+(right-left)*((j+.5)/10),left+(right-left)*((j+1)/10),.00025))
            else:
                stages=[(previous,half,end,.005)]
            for b0,bm,b1,dt in stages:
                k1,a1,_=rhs(r,z,b0)
                k2,a2,_=rhs(r+dt*.5*k1,z+dt*.5*a1,bm)
                k3,a3,_=rhs(r+dt*.5*k2,z+dt*.5*a2,bm)
                k4,a4,u=rhs(r+dt*k3,z+dt*a3,b1)
                r=r+dt/6*(k1+2*k2+2*k3+k4)
                z=z+dt/6*(a1+2*a2+2*a3+a4)
            activation=activation+(1-math.exp(-.5))*(self.pool@r-activation)
            commands.append(self.rest+4*self.moment@activation)
            rates.append(r);targets.append(u);previous=end
            if truncate_controls and (tick+1)%truncate_controls==0:
                # Training credit only. Values, recurrent feedback and clocks
                # remain continuous; there is no teacher-state replacement.
                r=r.detach();z=z.detach();activation=activation.detach()
        return torch.stack(commands),torch.stack(rates),torch.stack(targets),current


def run(source,initial,graph_path,out,student_motor=False,whole_development=False,optimizer_name='adam',truncate_controls=0,self_generated_boundary=False):
    out.mkdir(parents=True,exist_ok=False)
    spec={
        'experiment':'M3-E16-pooled-command-training' if not student_motor else 'M3-E17-motor-recurrent-command-training',
        'hypothesis':'Allowing anatomical motor-unit recruitment to differ, while learning the actual pooled42joint command, resolves unnecessary identical-rate constraints of E13',
        'intervention':'Same existing112471motor input edges and positive type-pair efficacy[.1,10] as E13; tau20ms/cap200/threshold7.5/adaptation8,.3 and96muscle pool/moment/rest/10ms response fixed. All391student motor r/z start zero and evolve continuously. No target/phase/frame index enters forward. Teacher presynaptic inputs remain conditional.',
        'recurrence': 'Existing motor-to-motor edges consume student rates during training' if student_motor else 'Conditional training uses all actual teacher presynaptic states; the SAME selected parameters are then also evaluated with motor-to-motor inputs replaced by student rates',
        'fit':'Initialize E13c efficacy, Adam.03/max200epochs/gradient norm clip1; command MSE normalized by perjoint train SD floor.1rad plus1e-4 mean(efficacy-1)^2. State evolves fromt0; loss[.5,1.2). Best training objective selects parameters, no validation selection. No tau/gain search.',
        'diagnostic':'Already exposed1.5..2.2s command RMS<=.05rad and all6 tibia command R2>=.5. Report silent/saturated motor targets and gradient reach. Validate learned5ms solver against native.25ms under same interpolated2.5ms boundary. Final full BANC teacher-free neutral/taught actualbody/F1-F8 remain required.',
        'limits':'One CUDA job; fit<=180wall sec, complete diagnostic<=300wall sec; no physical or full158706neural rollout',
        'source_sha256':sha256(source/'signals.npz'),'initial_sha256':sha256(initial/'candidate.npz'),
        'tool_sha256':sha256(__file__),
    }
    if whole_development:
        spec.update(experiment='M3-E17-development-motor-recurrence',
            hypothesis='Identification over all already exposed development states with actual motor feedback can reproduce the recorded gait; the former time split and conditional motor inputs omit this operating range and feedback',
            fit='Use the supplied existing-edge candidate, unchanged physiological values/sign/mask/efficacy bounds/pool. Continuous state starts zero; command loss covers[.5,2.2). The earlier1.5..2.2 block is now explicitly training data, not validation. No independent generalization is claimed; original full teacher-free F1-F8 remain mandatory.',
            diagnostic='Report reconstruction on[.5,2.2) and the already exposed1.5..2.2 subset. Native.25ms command RMS<=.05rad and all6 tibia command R2>=.5 remain the same reconstruction screen. This combined change does not isolate contributions of data coverage and motor recurrence.')
    if optimizer_name=='lbfgs':
        spec['optimizer']='Diagonal current-design preconditioning, box-constrained L-BFGS-B max500 objective iterations within the same180fit/300total wall limit. Same physical parameter bounds and command objective; no optimizer success is claimed on time/iteration limit.'
    if truncate_controls:
        if optimizer_name!='adam':raise ValueError('Truncated gradient is not the full objective derivative for line search')
        spec.update(experiment='M3-E17b-continuous-state-truncated-gradient',
            gradient_revision=f'E17 full-horizon recurrent gradient became nonfinite. Detach ONLY derivatives every{truncate_controls}controls ({truncate_controls*.005}s), retaining exact state values and all feedback. Forward must be byte exact with untruncated computation. Adam.03/clip1/max200epochs/180fit sec; same data/parameter bounds/dynamics. Primary reconstruction is native student-motor recurrence; conditional input comparison is secondary.',
            optimizer='Projected Adam with truncated derivative credit; no quasi-Newton convergence claim')
    if self_generated_boundary:
        collection=json.loads((source.parent/'spec.json').read_text())
        if not student_motor or not whole_development or not truncate_controls or source.name!='neutral' or collection['experiment']!='M3-E19-closed-neural-body-teacher' or collection['candidate_sha256']!=sha256(initial/'candidate.npz'):
            raise ValueError('Self-generated collection and initialization provenance mismatch')
        spec.update(experiment='M3-E20-self-generated-boundary-refit',
            hypothesis='One refit to actual nonmotor activities of the deployed full learned network reduces the E19 teacher-field mismatch',
            intervention='Existing W0 source weights times absolute positive type efficacy[.1,10], initialized at the SAME E17b candidate that generated E19. No multiplication onto already calibrated weights. All391 own motor states/feedback continuous from0; nonmotor input from full BANC with only BODY teaching, no motor clamp.',
            fit='One bounded projected Adam refit, .03/max200epochs/180fit seconds, truncate derivative credit every20controls while preserving all forward values. All exposed[.5,2.2) used for training; no independent validation.',
            decision='A conditional fit is not self-consistency. The frozen new theta must be rerun in the full network under the same body teaching, then without all teaching, before any claim of walking improvement. Original reconstruction/F1-F8 thresholds unchanged.',
            collection_spec_sha256=sha256(source.parent/'spec.json'))
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    d=dict(np.load(source/'signals.npz'));old=dict(np.load(initial/'candidate.npz'))
    for name in ('incoming_indptr','incoming_indices','motor_ids'):
        if not np.array_equal(d[name],old[name]):raise ValueError('Source edge or neuron identity changed')
    if not np.array_equal(d['incoming_data'],old['incoming_source_data']):raise ValueError('Refit must use original W0, not compound learned weights')
    g=GraphStore.load(graph_path)
    began=time.perf_counter()
    design,index,edge_groups,_keys,_,error=grouped_design(d,g)
    if error>1e-8 or not np.array_equal(old['edge_parameters'],edge_groups):raise ValueError('Source groups changed')
    m=len(d['motor_ids']);groups=d['motor_group'];pool=np.zeros((d['moment'].shape[1],m),np.float32)
    for k in range(len(pool)):
        ids=np.flatnonzero(groups==k);pool[k,ids]=1/len(ids)
    internal=np.zeros((m,m),np.float32);internal_groups=np.zeros((m,m),np.int64)
    local={int(v):i for i,v in enumerate(d['motor_ids'])}
    for i in range(m):
        for edge in range(d['incoming_indptr'][i],d['incoming_indptr'][i+1]):
            j=local.get(int(d['incoming_indices'][edge]))
            if j is not None:internal[i,j]=d['incoming_data'][edge];internal_groups[i,j]=edge_groups[edge]
    outside_data=dict(d);outside_data['presynaptic']=d['presynaptic'].copy()
    outside_data['presynaptic'][:,np.isin(d['observed_ids'],d['motor_ids'])]=0
    outside,check_index,check_edges,_,_,_=grouped_design(outside_data,g)
    if not np.array_equal(index,check_index) or not np.array_equal(edge_groups,check_edges):raise ValueError('Boundary split changed parameter identity')
    del outside_data
    if not torch.cuda.is_available():raise ValueError('CUDA training required')
    torch.backends.cuda.matmul.allow_tf32=False
    model=MotorCommandNetwork(old['efficacy'],index,internal,internal_groups,pool,d['moment'],d['rest'],'cuda')
    times=d['time_s'][1::2];end_time=2.2 if whole_development else 1.2
    prefix=int(np.count_nonzero(times<end_time+1e-9))
    train_window=(times[:prefix]>=.5-1e-9)&(times[:prefix]<end_time-1e-9)
    selected=outside if student_motor else design
    x=torch.as_tensor(selected[:,:prefix*2,:]/200,device='cuda')
    drive=torch.as_tensor(d['external_drive'][:prefix*2]/200,dtype=torch.float32,device='cuda')
    target=torch.as_tensor(d['command'][:prefix],dtype=torch.float32,device='cuda')
    loss_mask=torch.as_tensor(train_window,device='cuda')
    scale=torch.as_tensor(np.maximum(np.std(d['command'][:prefix][train_window],axis=0),.1),dtype=torch.float32,device='cuda')
    best=float('inf');best_state=None;history=[];gradient_diagnostic=None;best_epoch=-1
    fit_began=time.perf_counter()
    def objective_step(epoch,training_design=x):
        nonlocal best,best_state,best_epoch,gradient_diagnostic
        if time.perf_counter()-fit_began>180:raise TimeoutError('E17 fit wall limit')
        model.zero_grad(set_to_none=True)
        command,_rate,u,current=model(training_design,drive,student_motor=student_motor,retain_current_gradient=epoch==0,truncate_controls=truncate_controls)
        loss=((command[loss_mask]-target[loss_mask])/scale).square().mean()+1e-4*(model.efficacy-1).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite command objective')
        value=float(loss.detach().cpu())
        if value<best:best=value;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        loss.backward()
        if not bool(torch.isfinite(model.efficacy.grad).all()):
            raw=model.efficacy.grad.detach().cpu().numpy()
            np.savez_compressed(out/f'gradient-failure-{epoch}.npz',gradient=raw,
                efficacy=model.efficacy.detach().cpu().numpy(),command=command.detach().cpu().numpy())
            write_json(out/'training-error.json',{'run_status':'nonfinite_gradient','epoch':epoch,'loss':value,
                'nan_gradient_entries':int(np.isnan(raw).sum()),'inf_gradient_entries':int(np.isinf(raw).sum()),
                'wall_s':time.perf_counter()-began,'validity':'incomplete_optimization','adoption':'not_applicable','milestone_outcome':'pending'})
            raise FloatingPointError('Nonfinite gradient preserved before optimizer update/export')
        if epoch==0:
            reached=current.grad.detach().abs().max(dim=0).values.cpu().numpy()>1e-10
            gradient_diagnostic={'motor_rows_with_nonzero_current_gradient':int(reached.sum()),
                'muscles_with_any_current_gradient':int(sum(np.any(reached[groups==k]) for k in range(len(pool)))),
                'initial_silent_target_fraction':float((u<=1e-8).float().mean().detach().cpu()),
                'initial_saturated_target_fraction':float((u>=.995).float().mean().detach().cpu())}
        norm=float(torch.linalg.vector_norm(model.efficacy.grad.double()).cpu())
        history.append({'epoch':epoch,'loss':value,'gradient_norm_before_clip':norm,'wall_s':time.perf_counter()-began})
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(history[-1],allow_nan=False)+'\n')
        if epoch%25==0:print(json.dumps(history[-1]),flush=True)
        return value
    solver_status={'method':optimizer_name}
    if optimizer_name=='adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=.03)
        try:
            for epoch in range(200):
                objective_step(epoch)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                with torch.no_grad():model.efficacy.clamp_(.1,10)
        except TimeoutError:
            solver_status['stop']='fit_wall_limit'
        del optimizer
    else:
        from scipy.optimize import minimize
        diagonal=torch.zeros(len(model.efficacy),device='cuda')
        diagonal.scatter_add_(0,model.indices.flatten(),(2*x.square().sum(dim=1)/(x.shape[0]*x.shape[1])).flatten())
        conditioning=np.sqrt(diagonal.cpu().numpy().astype(float)+2e-4/len(model.efficacy))
        def solve(values):
            with torch.no_grad():model.efficacy.copy_(torch.as_tensor(values/conditioning,dtype=torch.float32,device='cuda'))
            value=objective_step(len(history))
            gradient=model.efficacy.grad.detach().cpu().numpy().astype(float)/conditioning
            return value,gradient
        try:
            fitted=minimize(solve,model.efficacy.detach().cpu().numpy()*conditioning,jac=True,method='L-BFGS-B',
                bounds=list(zip(.1*conditioning,10*conditioning)),options={'maxiter':500,'maxls':20,'maxcor':10,'ftol':1e-10,'gtol':1e-6})
            solver_status.update(success=bool(fitted.success),message=str(fitted.message),iterations=int(fitted.nit))
        except TimeoutError:
            solver_status.update(success=False,message='fit_wall_limit',iterations=None)
    model.load_state_dict(best_state)
    theta=model.efficacy.detach().cpu().numpy()
    np.savez_compressed(out/'candidate.npz',efficacy=theta,edge_parameters=edge_groups,
        incoming_indptr=d['incoming_indptr'],incoming_indices=d['incoming_indices'],
        incoming_source_data=d['incoming_data'],incoming_data=d['incoming_data']*theta[edge_groups],
        motor_ids=d['motor_ids'],moment=d['moment'],rest=d['rest'],motor_group=groups)
    write_json(out/'history.json',history);write_json(out/'gradient-diagnostic.json',gradient_diagnostic)
    x=drive=target=None
    del best_state
    metrics={};predictions={}
    joints=np.array([[0,2,1,4,3,5,6],[7,9,8,11,10,12,13],[14,16,15,18,17,19,20],
        [21,23,22,25,24,26,27],[28,30,29,32,31,33,34],[35,37,36,39,38,40,41]])
    validation=(times>=1.5-1e-9)&(times<=2.2+1e-9)
    drive=torch.as_tensor(d['external_drive']/200,dtype=torch.float32,device='cuda')
    with torch.inference_mode():
        for mode in ('conditional','student_motor'):
            x=torch.as_tensor((design if mode=='conditional' else outside)/200,device='cuda')
            fast,_rate,u,_=model(x,drive,student_motor=mode=='student_motor')
            native,_,_,_=model(x,drive,student_motor=mode=='student_motor',native=True)
            predictions[mode]=fast.cpu().numpy();predictions[mode+'_native']=native.cpu().numpy()
            metrics[mode]=command_metrics(native.cpu().numpy(),d['command'],validation,joints)
            if whole_development:
                metrics[mode]['whole_training_reconstruction']=command_metrics(native.cpu().numpy(),d['command'],(times>=.5-1e-9)&(times<2.2-1e-9),joints)
            metrics[mode].update(native_fast_rms_rad=float(torch.sqrt((native-fast).square().mean()).cpu()),
                silent_target_fraction=float((u<=1e-8).float().mean().cpu()),
                saturated_target_fraction=float((u>=.995).float().mean().cpu()))
            del x
    np.savez_compressed(out/'predictions.npz',**predictions)
    passed={k:v['rms_rad']<=.05 and all(r is not None and r>=.5 for r in v['knee_r2']) for k,v in metrics.items()}
    result={'experiment':spec['experiment'],'run_status':'completed','validity':'valid_conditional_development_diagnostic',
        'hypothesis_verdict':'reconstruction_screen_supported' if passed['student_motor' if student_motor else 'conditional'] else 'not_supported_in_bounded_fit',
        'adoption':'defer_full_student' if (passed['student_motor'] if student_motor else all(passed.values())) else 'reject','milestone_outcome':'pending',
        'epochs':len(history),'best_epoch':best_epoch,'initial_loss':history[0]['loss'],'best_loss':best,
        'solver':solver_status,'later_block_used_for_training':whole_development,
        'wall_s':time.perf_counter()-began,'device':torch.cuda.get_device_name(0),
        'gradient':gradient_diagnostic,'metrics':metrics,'screen_passed':passed,
        'candidate_sha256':sha256(out/'candidate.npz')}
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initial','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--student-motor',action='store_true')
    p.add_argument('--whole-development',action='store_true')
    p.add_argument('--optimizer',choices=('adam','lbfgs'),default='adam')
    p.add_argument('--truncate-controls',type=int,default=0)
    p.add_argument('--self-generated-boundary',action='store_true')
    a=p.parse_args();run(a.source,a.initial,a.graph,a.out,a.student_motor,a.whole_development,a.optimizer,a.truncate_controls,a.self_generated_boundary)
