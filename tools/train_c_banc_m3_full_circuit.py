"""Full-state differentiable BANC diagnostic with a fixed anatomical fit mask.

The complete nonmotor recurrence participates in the derivative. Only existing
motor incoming positive efficacy parameters are trainable. This tool never
exports a sensory recording, clock or teacher as a walking controller.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.adaptive_rate import _CSR_CURRENT
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.target_sense import TargetObservation
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.verify_c_banc_m3_full_trial import CalibratedTrial, calibrated_weights


class SparseCurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rate,efficacy,operator):
        ctx.operator=operator;ctx.generation=operator.generation
        ctx.save_for_backward(rate)
        return operator.multiply(rate)

    @staticmethod
    def backward(ctx,gradient):
        operator=ctx.operator
        if ctx.generation!=operator.generation:raise ValueError('Weights changed before saved-state backward')
        (rate,)=ctx.saved_tensors
        gradient=gradient.contiguous()
        rate_gradient=operator.multiply(gradient,transpose=True) if ctx.needs_input_grad[0] else None
        edge_gradient=gradient[operator.post]*operator.source_edge*rate[operator.pre]
        efficacy_gradient=torch.zeros(operator.parameter_count,device=gradient.device,dtype=gradient.dtype)
        efficacy_gradient.scatter_add_(0,operator.groups,edge_gradient)
        return rate_gradient,efficacy_gradient,None


class AnatomicalOperator:
    def __init__(self,source,edge_positions,groups,parameter_count,device='cpu',dtype=torch.float32):
        self.shape=source.shape;self.n=source.shape[0];self.device=device;self.parameter_count=parameter_count
        if source.shape[1]!=self.n:raise ValueError('Full square recurrence required')
        source=source.copy();source.check_format(full_check=True)
        self.source=source
        self.positions=torch.as_tensor(edge_positions,dtype=torch.long,device=device)
        posts=np.repeat(np.arange(self.n),np.diff(source.indptr))[edge_positions]
        self.post=torch.as_tensor(posts,dtype=torch.long,device=device)
        self.pre=torch.as_tensor(source.indices[edge_positions],dtype=torch.long,device=device)
        self.groups=torch.as_tensor(groups,dtype=torch.long,device=device)
        self.source_edge=torch.as_tensor(source.data[edge_positions],dtype=dtype,device=device)
        self.base=torch.as_tensor(source.data,dtype=dtype,device=device)
        self.values=self.base.clone()
        # Transpose carries edge ordinals, retaining even explicit zero entries.
        marker=csr_matrix((np.arange(source.nnz,dtype=np.int64),source.indices,source.indptr),shape=source.shape).T.tocsr()
        self.transpose_order=torch.as_tensor(marker.data,dtype=torch.long,device=device)
        self.transpose_values=self.values[self.transpose_order].clone()
        self.rows=[torch.as_tensor(p,dtype=torch.long,device=device) for p in (source.indptr,marker.indptr)]
        self.columns=[torch.as_tensor(p,dtype=torch.long,device=device) for p in (source.indices,marker.indices)]
        self.mask=torch.ones(self.n,dtype=dtype,device=device);self.generation=0;self.prepared_efficacy=None
        if device=='cuda':
            if dtype!=torch.float32:raise ValueError('Native CUDA sparse kernel requires float32')
            import cupy as cp
            self.cp=cp;self.kernel=cp.RawKernel(_CSR_CURRENT,'csr_current',options=('--fmad=false',));self.kernel.compile()

    def prepare(self,efficacy):
        if efficacy.shape!=(self.parameter_count,) or not bool(torch.isfinite(efficacy).all()):raise ValueError('Invalid efficacy vector')
        # Concurrent branches may share one immutable W. Only a value change
        # invalidates saved backward contexts, not a repeated identical call.
        if self.prepared_efficacy is not None and torch.equal(self.prepared_efficacy,efficacy.detach()):return
        with torch.no_grad():
            self.values.copy_(self.base)
            self.values[self.positions]=self.source_edge*efficacy[self.groups]
            self.transpose_values.copy_(self.values[self.transpose_order])
            self.prepared_efficacy=efficacy.detach().clone()
        self.generation+=1

    def multiply(self,rate,transpose=False):
        index=int(transpose);values=self.transpose_values if transpose else self.values
        if self.device=='cuda':
            result=torch.empty_like(rate)
            with self.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                self.kernel(((self.n+7)//8,),(256,),tuple(np.uint64(t.data_ptr()) for t in
                    (self.rows[index],self.columns[index],values,rate,self.mask,result))+(np.int32(self.n),))
            return result
        matrix=torch.sparse_csr_tensor(self.rows[index],self.columns[index],values,size=self.shape)
        return torch.sparse.mm(matrix,rate[:,None]).squeeze(1)


class FullCircuit(torch.nn.Module):
    def __init__(self,operator,calibration,pool,drive_ids,initial,parameters):
        super().__init__();self.operator=operator
        self.efficacy=torch.nn.Parameter(torch.as_tensor(calibration['efficacy'],device=operator.device).clone())
        for name,value in (('motor_ids',calibration['motor_ids']),('drive_ids',drive_ids),
                ('pool',pool),('moment',calibration['moment']),('rest',calibration['rest']),
                ('initial_rate',initial['rate']),('initial_adaptation',initial['adaptation']),
                ('tau',parameters['tau']),('slope',parameters['a']),('threshold',parameters['threshold']),('cap',parameters['cap'])):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.long if name.endswith('_ids') else torch.float32,device=operator.device))

    def forward(self,compact_drive,*,dt=.005,return_final=False,initial_state=None,return_state=False):
        steps=round(.005/dt)
        if steps<1 or abs(steps*dt-.005)>1e-12:raise ValueError('Control must have integral RK steps')
        self.operator.prepare(self.efficacy)
        if initial_state is None:
            r=self.initial_rate.clone();z=self.initial_adaptation.clone();a=torch.zeros(self.pool.shape[0],device=r.device)
        else:r,z,a=initial_state
        commands=[];rates=[]
        def rhs(rr,zz,drive):
            current=SparseCurrent.apply(rr,self.efficacy,self.operator)
            target=(self.cap*torch.tanh(self.slope/self.cap*(current+drive-self.threshold-zz))).clamp_min(0)
            return (target-rr)/self.tau,(8*rr-zz)/.3
        for compact in compact_drive:
            drive=torch.zeros_like(r);drive[self.drive_ids]=compact
            for _ in range(steps):
                k1,a1=rhs(r,z,drive)
                k2,a2=rhs(r+dt*.5*k1,z+dt*.5*a1,drive)
                k3,a3=rhs(r+dt*.5*k2,z+dt*.5*a2,drive)
                k4,a4=rhs(r+dt*k3,z+dt*a3,drive)
                r=r+dt/6*(k1+2*k2+2*k3+k4);z=z+dt/6*(a1+2*a2+2*a3+a4)
            a=a+(1-math.exp(-.5))*(self.pool@r[self.motor_ids]-a)
            commands.append(self.rest+.02*self.moment@a);rates.append(r[self.motor_ids])
        output=(torch.stack(commands),torch.stack(rates))
        if return_state:return (*output,(r,z,a))
        return (*output,r,z) if return_final else output


def checkpointed_gradient(model,drive,target,mask,scale,*,dt=.00025,block_controls=20,observer=None):
    """Exact discrete adjoint via recomputation; anchors are self-generated.

    All parameters stay fixed. Native forward contains no resets. Reverse-mode
    recomputation restores only this same trajectory's anchors, never a teacher.
    """
    state=(model.initial_rate.clone(),model.initial_adaptation.clone(),
           torch.zeros(model.pool.shape[0],device=model.efficacy.device))
    anchors=[tuple(v.detach().clone() for v in state)];commands=[];objective=0.
    blocks=list(range(0,len(drive),block_controls));denominator=int(mask.sum())*target.shape[1]
    with torch.no_grad():
        for start in blocks:
            end=min(start+block_controls,len(drive))
            command,_rate,state=model(drive[start:end],dt=dt,initial_state=state,return_state=True)
            anchors.append(tuple(v.detach().clone() for v in state));commands.append(command.detach())
            objective+=float(((((command-target[start:end])/scale).square()*mask[start:end,None]).sum()/denominator).cpu())
            if observer:observer('forward',start,state,None)
    model.zero_grad(set_to_none=True);adjoint=tuple(torch.zeros_like(v) for v in state)
    for index in range(len(blocks)-1,-1,-1):
        start=blocks[index];end=min(start+block_controls,len(drive))
        leaf=tuple(v.detach().clone().requires_grad_(True) for v in anchors[index])
        command,_rate,recomputed=model(drive[start:end],dt=dt,initial_state=leaf,return_state=True)
        if any(not torch.equal(a,b) for a,b in zip(recomputed,anchors[index+1])):raise ValueError('Adjoint recomputation changed native forward state')
        loss=((((command-target[start:end])/scale).square()*mask[start:end,None]).sum()/denominator)
        torch.autograd.backward((loss,*recomputed),(torch.ones_like(loss),*adjoint))
        adjoint=tuple(v.grad.detach().clone() for v in leaf)
        if observer:observer('backward',start,adjoint,model.efficacy.grad)
        if not all(bool(torch.isfinite(v).all()) for v in (*adjoint,model.efficacy.grad)):raise FloatingPointError('Nonfinite long-horizon adjoint')
        del leaf,recomputed,loss,command,_rate
    penalty=1e-4*(model.efficacy-1).square().mean();objective+=float(penalty.detach().cpu());penalty.backward()
    return objective,torch.cat(commands),model.efficacy.grad.detach().clone(),anchors[-1]


def run(source,candidate,reference,labels,graph_path,out,dt):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E22b-native-full-circuit-fidelity',
        'hypothesis':'A differentiable full158706-state BANC recurrence can match native execution sufficiently to train through the previously omitted nonmotor feedback',
        'intervention':'No fitted parameter change. Full source neuron/edge roster, same fixed E20 efficacy and body-derived sensory/tonic input. Same physiological equations and RAW native rate/adaptation arithmetic, eliminating E22 normalized-coordinate float-rounding differences. Existing zero/sign/mask retained. Native .25ms (or explicitly recorded diagnostic dt).',
        'prior_failure':'E22 normalized-coordinate5ms full solver failed badly; no training used it. This native raw-coordinate comparison verifies the full gradient scaffold before any training. It changes both arithmetic coordinates and step size, so it does not isolate those two numerical effects.',
        'derivative_contract':'d_r=W(theta)^T v; d_theta=sum_existing_group(v_post W0_edge r_pre). W remains unchanged throughout a saved forward/backward. CUDA gradient group accumulation uses float32 scatter atomics; not a claim of training bitwise determinism.',
        'measurement':'Body input replay all440steps byte exact; complete native reference E21 motor rates and decoded command. Source-forward sparse current uses same deterministic warp kernel. Derivative implementation separately checked on independent small dense equations.',
        'gate':'Full2.2s command RMS<=.02rad, motor rate RMS<=1 and absolute99percentile<=5. Final direct-presynaptic r and z/8 each RMS<=1/p99<=5, and final outside motor current RMS<=5units (2.5percent cap). Finite full state. No training on this solver if failed. Any later fitted parameter requires another native check.',
        'limits':'One GPU diagnostic,2.2 neural modelsec plus2.2 body-replaysec/300wallsec, no fitting/adoption; existing F1-F8 pending',
        'dt_s':dt,'candidate_sha256':sha256(candidate/'candidate.npz'),
        'reference_sha256':sha256(reference/'signals.npz'),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    cal=dict(np.load(candidate/'candidate.npz'));ref=dict(np.load(reference/'signals.npz'))
    reference_result=json.loads((reference/'result.json').read_text())
    if reference_result['candidate_sha256']!=spec['candidate_sha256']:raise ValueError('Reference candidate mismatch')
    session=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,session,cal,learned=False)
        positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Wrong absolute source-edge weights')
        parameters={k:getattr(session.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        if session.network.adaptation_gain!=8 or session.network.adaptation_tau_s!=.3:raise ValueError('Adaptation contract changed')
        pool=session.adapter.pool.copy();initial=session.network.snapshot()
        if not np.all(initial['output_mask']==1):raise ValueError('Full derivative diagnostic requires all outgoing masks1')
        drive_ids=np.union1d(session.adapter._sensory_ids,np.r_[session.descending,session.navigation.encoder.ports])
        tonic=session.navigation.encoder.drive(TargetObservation(0,20,True));drives=[]
        if np.count_nonzero(tonic)!=22:raise ValueError('Unexpected fixed tonic')
        rows=json.loads((labels/'trace.json').read_text())['samples'][1:441]
        for tick,row in enumerate(rows):
            drive=session.adapter.encode()+tonic
            if np.count_nonzero(drive)!=np.count_nonzero(drive[drive_ids]):raise ValueError('Omitted external input')
            drives.append(drive[drive_ids].copy())
            session.body.step_joint_targets(np.asarray(row['joint_targets_rad']),np.asarray(row['adhesion'],bool),tendon_inputs=row['tendon_inputs'])
            for key in ('qpos','qvel','ctrl'):
                if getattr(session.body.d,key).tobytes()!=np.asarray(row[key]).tobytes():raise AssertionError(f'Body replay mismatch {tick}/{key}')
        np.savez_compressed(out/'sensory-input.npz',drive=np.asarray(drives),drive_ids=drive_ids,initial_rate=initial['rate'],initial_adaptation=initial['adaptation'])
        # The neural state has not advanced while collecting the body-only input.
        if session.network.tick!=origin['network']['tick']:raise ValueError('Unrecorded neural advance')
    finally:session.close()
    operator=AnatomicalOperator(base,np.asarray(positions),cal['edge_parameters'],len(cal['efficacy']),'cuda')
    model=FullCircuit(operator,cal,pool,drive_ids,initial,parameters)
    with torch.inference_mode():
        command,rate,final_rate,final_adaptation=model(torch.as_tensor(np.asarray(drives),device='cuda'),dt=dt,return_final=True)
    arrays={k:v.cpu().numpy() for k,v in (('command',command),('rate',rate),('final_rate',final_rate),('final_adaptation',final_adaptation))}
    np.savez_compressed(out/'replay.npz',**arrays)
    error=arrays['rate']-ref['rate'][1::2];qerror=arrays['command']-ref['decoded_command']
    rate_rms=float(np.sqrt(np.mean(error.astype(float)**2)));rate_p99=float(np.percentile(abs(error),99))
    q_rms=float(np.sqrt(np.mean(qerror**2)));finite=all(np.isfinite(v).all() for v in arrays.values())
    reference_final=StateStore.load(reference/'final-diagnostic')['network']
    end_rate_error=arrays['final_rate']-reference_final['rate']
    end_adaptation_error=(arrays['final_adaptation']-reference_final['adaptation'])/8
    ids=ref['observed_ids'];end_metrics={}
    for name,error in (('rate',end_rate_error[ids]),('adaptation_div8',end_adaptation_error[ids])):
        end_metrics[name]={'rms':float(np.sqrt(np.mean(error.astype(float)**2))),'abs_p99':float(np.percentile(abs(error),99))}
    outside=csr_matrix((cal['incoming_data'],cal['incoming_indices'],cal['incoming_indptr']),shape=(391,g.n))
    outside.data[np.isin(outside.indices,cal['motor_ids'])]=0
    current_error=outside@end_rate_error
    end_metrics['outside_current_rms']=float(np.sqrt(np.mean(current_error.astype(float)**2)))
    passed=finite and rate_rms<=1 and rate_p99<=5 and q_rms<=.02 and end_metrics['outside_current_rms']<=5 and all(end_metrics[k]['rms']<=1 and end_metrics[k]['abs_p99']<=5 for k in ('rate','adaptation_div8'))
    result={'run_status':'completed','validity':'valid_numerical_diagnostic','hypothesis_verdict':'supported_at_frozen_parameters' if passed else 'solver_fidelity_rejected',
        'adoption':'not_applicable','milestone_outcome':'pending','solver_screen_passed':passed,
        'neural_model_s':2.2,'body_replay_model_s':2.2,'wall_s':time.perf_counter()-began,'dt_s':dt,
        'finite':finite,'command_rms_rad':q_rms,'motor_rate_rms':rate_rms,'motor_rate_abs_p99':rate_p99,
        'final_presynaptic_metrics':end_metrics,
        'neurons':g.n,'edges':base.nnz,'candidate_sha256':spec['candidate_sha256'],
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','reference','labels','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dt',type=float,default=.005)
    a=p.parse_args();run(a.source,a.candidate,a.reference,a.labels,a.graph,a.out,a.dt)
