"""M3-E09: sign/mask constrained recurrent calibration feasibility screen.

This learner consumes recorded external-neuron currents. It is never a walking
runtime. Only its explicit existing-edge efficacies and cell parameters may be
exported, and a full-network, teacher-free physical experiment is still required.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json


def group_parameters(nodes, weights):
    def cell_key(n):
        return (n.get("cell_type") or n["id"],
            n.get("source_annotations", {}).get("body_part_effector") or "")

    cell_keys = [cell_key(n) for n in nodes]
    types = {key: i for i, key in enumerate(sorted(set(cell_keys)))}
    cells = np.array([types[k] for k in cell_keys])
    pairs, index = {}, np.zeros(weights.shape, dtype=np.int64)
    for post, pre in zip(*np.nonzero(weights)):
        # Preserve laterality-dependent pathways without fitting each individual cell.
        relationship = "same" if nodes[post].get("soma_side") == nodes[pre].get("soma_side") else "different"
        key = (cell_keys[post], cell_keys[pre], relationship)
        if key not in pairs:
            pairs[key] = len(pairs)
        index[post, pre] = pairs[key]
    return cells, index, types, pairs


class CalibrationNetwork(torch.nn.Module):
    def __init__(self, weights, cell_groups, pair_groups, pool, moment, rest,
                 motor_rows, *, recurrent, device="cpu"):
        super().__init__()
        w = np.asarray(weights, np.float32)
        selected = np.ones(len(w), bool) if recurrent else np.asarray(motor_rows, bool)
        eligible = (w != 0) & selected[:, None]
        cell_eligible = np.zeros(int(max(cell_groups))+1, dtype=np.float32)
        cell_eligible[np.asarray(cell_groups)[selected]] = 1
        for name, value in (("base_weights",w),("pair_groups",np.asarray(pair_groups,np.int64)),
            ("cell_groups",np.asarray(cell_groups,np.int64)),("edge_eligible",eligible.astype(np.float32)),
            ("cell_eligible",cell_eligible),("pool",pool),("moment",moment),("rest",rest)):
            value = np.asarray(value)
            self.register_buffer(name, torch.as_tensor(value, device=device,
                dtype=torch.long if value.dtype.kind in "iu" else torch.float32))
        self.log_efficacy = torch.nn.Parameter(torch.zeros(int(np.max(pair_groups))+1, device=device))
        self.log_tau_ratio = torch.nn.Parameter(torch.zeros(len(cell_eligible), device=device))
        self.log_threshold_ratio = torch.nn.Parameter(torch.zeros(len(cell_eligible), device=device))

    def physical_parameters(self):
        w=self.base_weights*torch.exp(self.log_efficacy[self.pair_groups]*self.edge_eligible)
        tau=.02*torch.exp((self.log_tau_ratio*self.cell_eligible)[self.cell_groups])
        threshold=7.5*torch.exp((self.log_threshold_ratio*self.cell_eligible)[self.cell_groups])
        return w,tau,threshold

    def project_parameters(self):
        with torch.no_grad():
            self.log_efficacy.clamp_(math.log(.1),math.log(10))
            self.log_tau_ratio.clamp_(math.log(.0125/.02),math.log(.05/.02))
            self.log_threshold_ratio.clamp_(math.log(1/7.5),math.log(15/7.5))

    def forward(self, stage_current, initial_rate, initial_adaptation, *, return_states=False):
        # Rates/adaptation are normalized by cap200 only for numeric conditioning.
        w,tau,threshold=self.physical_parameters()
        threshold=threshold/200
        r,z=initial_rate/200,initial_adaptation/200
        activation=torch.zeros(self.pool.shape[0],device=r.device)
        output=[]
        recorded_rate,recorded_adaptation=[],[]

        def f(r,z,b):
            target=torch.clamp_min(torch.tanh(w@r+b/200-threshold-z),0)
            return (target-r)/tau,(8*r-z)/.3

        for b in stage_current:
            k1,a1=f(r,z,b[0])
            k2,a2=f(r+.0025*k1,z+.0025*a1,b[1])
            k3,a3=f(r+.0025*k2,z+.0025*a2,b[1])
            k4,a4=f(r+.005*k3,z+.005*a3,b[2])
            r=r+.005/6*(k1+2*k2+2*k3+k4)
            z=z+.005/6*(a1+2*a2+2*a3+a4)
            activation=activation+(1-math.exp(-.5))*(self.pool@r-activation)
            output.append(self.rest+4*self.moment@activation)
            if return_states:
                recorded_rate.append(r)
                recorded_adaptation.append(z)
        if return_states:
            return torch.stack(output),torch.stack(recorded_rate),torch.stack(recorded_adaptation)
        return torch.stack(output),r,z


def command_metrics(command,target,mask,joints):
    error=command[mask]-target[mask]
    knee=[]
    for j in joints[:,5]:
        variance=np.var(target[mask,j])
        knee.append(float(1-np.mean(error[:,j]**2)/variance) if variance>1e-12 else None)
    return {"rms_rad":float(np.sqrt(np.mean(error**2))),"knee_r2":knee}


def run(source, identification, fidelity, out, mode, epochs, seconds):
    out.mkdir(parents=True,exist_ok=False)
    check=json.loads((fidelity/'result.json').read_text())
    if not check['screen_passed'] or not check['whole_control_solver_screen_passed']:
        raise ValueError('Reduced integration has not passed numerical fidelity')
    d=np.load(source/'signals.npz')
    nodes=json.loads((source/'topology.json').read_text())['core_nodes']
    collection=json.loads((identification/'collection-result.json').read_text())
    cells,pairs,types,pair_keys=group_parameters(nodes,d['internal_weights'])
    core=d['core'];pool=np.zeros((len(collection['muscles']),len(core)),np.float32)
    for k,g in enumerate(collection['muscles']):
        ids=d['motor_ids'][g['columns']]
        local=np.searchsorted(core,ids)
        if not np.array_equal(core[local],ids):
            raise ValueError('Anatomical muscle member missing')
        pool[k,local]=1/len(local)
    motor_rows=np.isin(core,d['motor_ids'])
    spec={
        'experiment':'M3-E09-'+mode,
        'hypothesis':'Existing '+mode+' efficacies and cell dynamics can generate the projected six-leg motor commands under a recorded sensory-driven boundary',
        'competing_explanations':['Selected signals/topology lack useful dynamical information','Parameter bounds or optimization fail','Teacher boundary supplies a pattern that will fail in the full loop'],
        'intervention':'Source anatomical mask/sign/count fixed; positive type-pair efficacy ratios[.1,10] only on explicitly selected internal edges; typed tau[.0125,.05]s, threshold[1,15]; cap200/adaptation8/.3 fixed. Pool/moment/rest fixed. '+mode+' means '+('only motor postsynaptic rows/cells' if mode=='motor' else 'all selected premotor and motor rows/cells'),
        'student_inputs':'Only recorded W0[core,outside] currents and actual core external drive at RK stage times; no phase, time, clip index, teacher motor or target input',
        'state_contract':'One original initial state per whole trajectory; rate/adaptation/muscle activation evolve continuously. No per-frame/period/chunk teacher reset.',
        'solver':'5ms RK4 with causal 1ms boundary samples at 0/2/4ms relative to each control; checked against native .25ms integration in E08b; changed parameters must later be rechecked',
        'train_loss_window_s':[0,1.2], 'reported_train_window_s':[.5,1.2],
        'validation_window_s':[1.5,2.2],
        'optimizer':{'name':'Adam','learning_rate':.01,'max_epochs':epochs,'seed':42,'gradient_norm_clip':1,'mean_log_parameter_l2':1e-4},
        'selection':'Lowest training objective only; validation read once after training, no validation-based checkpoint/early stopping',
        'screen':'Validation projected-command RMS<=.05rad and R2>=.5 for every knee; finite output and unchanged topology/signs. This remains a reduced-model diagnostic, not M3 adoption.',
        'limits':{'device':'one CUDA GPU','wall_s':seconds,'full_neural_physics_steps':0},
        'signal_sha256':sha256(source/'signals.npz'), 'tool_sha256':sha256(__file__),
        'fidelity_sha256':sha256(fidelity/'result.json'),
        'free_groups':{'efficacy':len(pair_keys),'cells':len(types)},
    }
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    if not torch.cuda.is_available():
        raise ValueError('Actual CUDA required')
    torch.manual_seed(42);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    model=CalibrationNetwork(d['internal_weights'],cells,pairs,pool,d['moment'],d['rest'],
        motor_rows,recurrent=mode=='recurrent',device='cuda')
    boundary=d['outside_current'][:-1]+d['drive']
    if len(boundary)!=5*len(d['target']):
        raise ValueError('Expected 1ms boundary and5ms target')
    stages=boundary.reshape(-1,5,len(core))[:,[0,2,4]]
    b=torch.tensor(stages,device='cuda')
    r0=torch.tensor(d['rate'][0],device='cuda');z0=torch.tensor(d['adaptation'][0],device='cuda')
    target=d['target'];times=np.arange(1,len(target)+1)*.005
    nt=int(np.count_nonzero(times<1.2))
    truth=torch.tensor(target[:nt],device='cuda',dtype=torch.float32)
    scale=torch.tensor(np.maximum(np.std(target[(times>=.5)&(times<1.2)],axis=0),.1),device='cuda',dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    began=time.perf_counter();best=None;best_value=math.inf;best_epoch=None
    with (out/'training.jsonl').open('w') as stream:
        for epoch in range(epochs):
            if time.perf_counter()-began>seconds:
                break
            optimizer.zero_grad(set_to_none=True)
            command,_,_=model(b[:nt],r0,z0)
            mse=torch.mean(((command-truth)/scale)**2)
            prior=sum(torch.mean(p**2) for p in model.parameters())
            loss=mse+1e-4*prior
            value=float(loss.detach())
            if not np.isfinite(value):
                raise ValueError('Nonfinite calibration objective')
            # Save the parameters that actually produced this objective.
            if value<best_value:
                best_value=value;best_epoch=epoch
                best={k:v.detach().clone() for k,v in model.named_parameters()}
            loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not np.isfinite(gradient):
                raise ValueError('Nonfinite calibration gradient')
            optimizer.step();model.project_parameters()
            row={'epoch':epoch,'objective':value,'gradient_norm':gradient,
                'wall_s':time.perf_counter()-began}
            stream.write(json.dumps(row)+'\n');stream.flush()
            if epoch%20==0:
                print(json.dumps(row),flush=True)
    if best is None:
        raise ValueError('No completed calibration epoch')
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.copy_(best[name])
        command,final_r,final_z=model(b,r0,z0)
        command=command.cpu().numpy()
        w,tau,threshold=[a.cpu().numpy() for a in model.physical_parameters()]
    if not np.isfinite(command).all() or not np.array_equal(np.sign(w),np.sign(d['internal_weights'])):
        raise ValueError('Output or source sign/mask invariant failed')
    np.savez_compressed(out/'candidate.npz',core=core,weights=w,tau=tau,threshold=threshold,
        base_weights=d['internal_weights'],selected_edge_indices=d['selected_edge_indices'],
        pool=pool,moment=d['moment'],rest=d['rest'],pair_groups=pairs,cell_groups=cells,
        command=command,final_normalized_rate=final_r.cpu().numpy(),
        final_normalized_adaptation=final_z.cpu().numpy(),
        **{k:v.cpu().numpy() for k,v in best.items()})
    train=command_metrics(command,target,(times>=.5)&(times<1.2),d['joints'])
    valid=command_metrics(command,target,(times>=1.5)&(times<2.2),d['joints'])
    passed=valid['rms_rad']<=.05 and all(v is not None and v>=.5 for v in valid['knee_r2'])
    report={'experiment':spec['experiment'],'run_status':'completed' if epoch+1==epochs else 'wall_limit',
        'validity':'valid_reduced_boundary_diagnostic','hypothesis_verdict':'supported_for_reduced_screen' if passed else 'not_supported_in_bounded_fit',
        'screen_passed':passed,'training':train,'validation':valid,
        'best_epoch':best_epoch,'best_training_objective':best_value,'last_epoch':epoch,
        'wall_s':time.perf_counter()-began,'device':'cuda',
        'source_sign_and_mask_preserved':True,'adoption':'not_applicable','milestone_outcome':'pending',
        'candidate_sha256':sha256(out/'candidate.npz')}
    write_json(out/'result.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--identification',type=Path,required=True)
    p.add_argument('--fidelity',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('motor','recurrent'),required=True)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--seconds',type=int,default=300)
    a=p.parse_args();run(a.source,a.identification,a.fidelity,a.out,a.mode,a.epochs,a.seconds)
