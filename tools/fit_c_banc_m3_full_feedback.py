"""Bounded native full-feedback BANC efficacy learning probe.

Blockwise backward limits memory, while all state values and parameters remain
continuous throughout each complete trajectory. Only derivative credit is cut.
"""

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
from tools.probe_c_banc_m3_closed_boundary import JOINTS
from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, FullCircuit
from tools.train_c_banc_m3_recurrent import command_metrics
from tools.verify_c_banc_m3_full_trial import CalibratedTrial, calibrated_weights


def run(source,candidate,reference,fidelity,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((fidelity/'result.json').read_text())
    if not prior['solver_screen_passed'] or prior['dt_s']!=.00025 or prior['candidate_sha256']!=sha256(candidate/'candidate.npz'):
        raise ValueError('Matching native full-circuit fidelity required')
    spec={'experiment':'M3-E23-native-full-feedback-learning-probe',
        'hypothesis':'Including all nonmotor recurrent feedback in the gradient can reduce the full-network error that persists after E20 conditional refit',
        'intervention':'Full158706-neuron native.25ms recurrence and existing source edges/signs. Train the SAME42,120 positive type-pair motor incoming efficacies, absolute[.1,10] on W0; other weights and all cell physiology fixed. Body-derived input is held for5ms, no motor forcing, no stored nonmotor boundary or per-block teacher reset.',
        'derivative':'Native raw-rate forward, full W(theta)^T rate gradient and eligible-only weight gradient. Every20control/.1s block does backward and accumulates gradient, then detaches only r/z/A derivatives. All state values remain continuous; theta updates ONLY after a whole2.2s trajectory. Long-range credit beyond.1s is omitted.',
        'fit':'One bounded3epoch probe. Adam.03; float64 gradient norm for clip1; theta projected[.1,10]. All exposed[.5,2.2) command error normalized by perjoint SD floor.1 plus1e-4 mean(theta-1)^2. Select lowest COMPLETE pre-update full-trajectory objective only. Native initial trajectory must match E21 motor rates/final fullr,z exactly and command within1e-6rad.',
        'gate':'Report native reconstruction .05rad/all6knee R2>=.5 and feedback gradient reach. No gait adoption from body teaching; frozen best candidate still requires independent actual runtime and original F1-F8.',
        'limits':'One CUDA job, max3complete trajectories=6.6neuralmodelsec; <=240fit/295total wallsec. Preserve partial epoch on timeout/nonfinite failure; never select it as complete.',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'fidelity_sha256':sha256(fidelity/'result.json'),
        'sensory_input_sha256':sha256(fidelity/'sensory-input.npz'),'tool_sha256':sha256(__file__),
        'core_tool_sha256':sha256(Path('tools/train_c_banc_m3_full_circuit.py'))}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'full-circuit.py').write_bytes(Path('tools/train_c_banc_m3_full_circuit.py').read_bytes())
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    cal=dict(np.load(candidate/'candidate.npz'));ref=dict(np.load(reference/'signals.npz'))
    inputs=dict(np.load(fidelity/'sensory-input.npz'))
    session=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,session,cal,learned=False);positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Wrong absolute source weights')
        parameters={k:getattr(session.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        initial=session.network.snapshot();pool=session.adapter.pool.copy()
        if not np.all(initial['output_mask']==1):raise ValueError('Full outgoing mask required')
    finally:session.close()
    operator=AnatomicalOperator(base,np.asarray(positions),cal['edge_parameters'],len(cal['efficacy']),'cuda')
    model=FullCircuit(operator,cal,pool,inputs['drive_ids'],initial,parameters)
    drive=torch.as_tensor(inputs['drive'],device='cuda');target=torch.as_tensor(ref['command'],dtype=torch.float32,device='cuda')
    mask=(ref['time_s'][1::2]>=.5-1e-9)&(ref['time_s'][1::2]<2.2-1e-9)
    scale=torch.as_tensor(np.maximum(np.std(ref['command'][mask],axis=0),.1),dtype=torch.float32,device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    reference_final=StateStore.load(reference/'final-diagnostic')['network']
    best=float('inf');best_theta=None;best_commands=None;history=[];epoch=-1;block=-1;completed_controls=0
    fit_began=time.perf_counter();stop='maximum3epochs';initial_parity=None;latest_forward_state=None
    try:
        for epoch in range(3):
            optimizer.zero_grad(set_to_none=True);state=None;outputs=[];rate_outputs=[];objective=0.
            for block in range(0,len(drive),20):
                if time.perf_counter()-fit_began>240 or time.perf_counter()-began>290:raise TimeoutError('Native learning wall limit')
                end=min(block+20,len(drive))
                command,rate,next_state=model(drive[block:end],dt=.00025,initial_state=state,return_state=True)
                completed_controls+=end-block
                latest_forward_state=tuple(v.detach() for v in next_state)
                selected=torch.as_tensor(mask[block:end],device='cuda')
                loss=(((command-target[block:end])/scale).square()*selected[:,None]).sum()/(int(mask.sum())*42)
                if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(v).all()) for v in next_state):raise FloatingPointError('Nonfinite forward block')
                objective+=float(loss.detach().cpu());loss.backward()
                if not bool(torch.isfinite(model.efficacy.grad).all()):raise FloatingPointError('Nonfinite full-feedback gradient')
                outputs.append(command.detach().cpu().numpy());rate_outputs.append(rate.detach().cpu().numpy())
                state=tuple(v.detach() for v in next_state)
                del command,rate,next_state,loss
                if (block//20)%5==0:print(json.dumps({'epoch':epoch,'control':end,'wall_s':time.perf_counter()-began}),flush=True)
            penalty=1e-4*(model.efficacy-1).square().mean();objective+=float(penalty.detach().cpu());penalty.backward()
            command=np.concatenate(outputs);rate=np.concatenate(rate_outputs)
            if epoch==0:
                initial_parity={'motor_max_abs':float(np.max(abs(rate-ref['rate'][1::2]))),
                    'command_max_abs':float(np.max(abs(command-ref['decoded_command']))),
                    'final_rate_exact':np.array_equal(state[0].cpu().numpy(),reference_final['rate']),
                    'final_adaptation_exact':np.array_equal(state[1].cpu().numpy(),reference_final['adaptation'])}
                write_json(out/'initial-forward-parity.json',initial_parity)
                if initial_parity['motor_max_abs']!=0 or initial_parity['command_max_abs']>1e-6 or not initial_parity['final_rate_exact'] or not initial_parity['final_adaptation_exact']:
                    raise ValueError('Block derivative processing changed native forward trajectory')
            norm=float(torch.linalg.vector_norm(model.efficacy.grad.double()).cpu())
            entry={'epoch':epoch,'objective':objective,'gradient_norm':norm,
                'nonzero_parameter_gradients':int(torch.count_nonzero(model.efficacy.grad).cpu()),
                'command_metrics':command_metrics(command,ref['command'],mask,JOINTS),'wall_s':time.perf_counter()-began}
            history.append(entry)
            with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
            if objective<best:
                best=objective;best_theta=model.efficacy.detach().cpu().numpy().copy();best_commands=command.copy()
            np.savez_compressed(out/f'epoch-{epoch}.npz',efficacy=model.efficacy.detach().cpu().numpy(),
                gradient=model.efficacy.grad.detach().cpu().numpy(),command=command,rate=rate)
            print(json.dumps(entry),flush=True)
            with torch.no_grad():model.efficacy.grad.mul_(min(1.,1/max(norm,1e-30)))
            optimizer.step()
            with torch.no_grad():model.efficacy.clamp_(.1,10)
    except (TimeoutError,FloatingPointError,ValueError) as exc:
        stop=type(exc).__name__+': '+str(exc)
        partial={'efficacy':model.efficacy.detach().cpu().numpy(),'gradient':model.efficacy.grad.detach().cpu().numpy() if model.efficacy.grad is not None else np.array([])}
        if state is not None:partial.update(rate=state[0].cpu().numpy(),adaptation=state[1].cpu().numpy(),activation=state[2].cpu().numpy())
        if latest_forward_state is not None:partial.update(latest_forward_rate=latest_forward_state[0].cpu().numpy(),
            latest_forward_adaptation=latest_forward_state[1].cpu().numpy(),latest_forward_activation=latest_forward_state[2].cpu().numpy())
        np.savez_compressed(out/'incomplete-epoch.npz',**partial)
        write_json(out/'incomplete-epoch.json',{'stop':stop,'epoch':epoch,'block_start_control':block,'complete_history_entries':len(history),'adoption':'not_applicable'})
    if best_theta is not None:
        exported=dict(cal);exported['efficacy']=best_theta;exported['incoming_data']=cal['incoming_source_data']*best_theta[cal['edge_parameters']]
        np.savez_compressed(out/'candidate.npz',**exported);np.savez_compressed(out/'best-command.npz',command=best_commands)
    metrics=command_metrics(best_commands,ref['command'],mask,JOINTS) if best_commands is not None else None
    passed=metrics is not None and metrics['rms_rad']<=.05 and all(v is not None and v>=.5 for v in metrics['knee_r2'])
    result={'experiment':spec['experiment'],'run_status':'completed' if len(history)==3 else 'bounded_incomplete',
        'validity':'native_full_feedback_learning_diagnostic','adoption':'defer_full_student' if passed else 'reject','milestone_outcome':'pending',
        'solver_screen_passed':passed,'stop':stop,'completed_epochs':len(history),'neural_model_s':completed_controls*.005,
        'wall_s':time.perf_counter()-began,'best_objective':best if best_theta is not None else None,
        'metrics':metrics,'initial_parity':initial_parity,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'candidate_sha256':sha256(out/'candidate.npz') if best_theta is not None else None}
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','reference','fidelity','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.candidate,a.reference,a.fidelity,a.graph,a.out)
