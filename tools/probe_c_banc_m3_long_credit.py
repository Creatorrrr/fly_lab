"""Native full-trajectory discrete-adjoint diagnostic; no parameter update."""

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
from tools.train_c_banc_m3_full_circuit import (
    AnatomicalOperator,
    FullCircuit,
    checkpointed_gradient,
)
from tools.verify_c_banc_m3_full_trial import CalibratedTrial, calibrated_weights


def run(source,candidate,reference,fidelity,truncated,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E24-native-long-credit-adjoint',
        'hypothesis':'Truncating derivative credit at.1s changes the training direction enough to help explain E23 nonimprovement',
        'intervention':'NO parameter update. Same E20 absolute efficacy, all full native .25ms equations and body-only input. One continuous2.2s forward saves its OWN r/z/A anchors. Reverse-mode recomputation of.1s blocks carries all future r/z/A adjoints; every recomputed state must be byte exact. No teacher-state clamp or truncated credit.',
        'measurement':'Compare full2.2s gradient to E23 epoch0 truncated gradient at identicaltheta/data/objective. Report gradient norm/cosine/finite status, subtract identical parameter prior for data-gradient comparison. This diagnoses temporal credit; it does not isolate particular anatomical pathways.',
        'validity':'Small-system checkpointed gradient matches direct whole-trajectory autograd. Actual forward command/finalfullr,z match E21; each reverse block matches its own anchor. Nonfinite gradients preserved immediately, no optimizer/export.',
        'limits':'One GPU,2.2s forward plus2.2s recomputation=4.4neuralmodelsec,<=285wallsec. F1-F8 pending; no candidate adoption',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'truncated_sha256':sha256(truncated/'epoch-0.npz'),
        'fidelity_sha256':sha256(fidelity/'result.json'),'tool_sha256':sha256(__file__),
        'core_tool_sha256':sha256(Path('tools/train_c_banc_m3_full_circuit.py'))}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'full-circuit.py').write_bytes(Path('tools/train_c_banc_m3_full_circuit.py').read_bytes())
    prior=json.loads((fidelity/'result.json').read_text())
    if not prior['solver_screen_passed'] or prior['candidate_sha256']!=spec['candidate_sha256']:raise ValueError('Matching native fidelity required')
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    cal=dict(np.load(candidate/'candidate.npz'));ref=dict(np.load(reference/'signals.npz'))
    inputs=dict(np.load(fidelity/'sensory-input.npz'));old=dict(np.load(truncated/'epoch-0.npz'))
    if not np.array_equal(old['efficacy'],cal['efficacy']):raise ValueError('Different truncated-gradient candidate')
    s=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,s,cal,learned=False);positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Wrong source weights')
        parameters={k:getattr(s.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        initial=s.network.snapshot();pool=s.adapter.pool.copy()
        if not np.all(initial['output_mask']==1):raise ValueError('Full mask required')
    finally:s.close()
    operator=AnatomicalOperator(base,np.asarray(positions),cal['edge_parameters'],len(cal['efficacy']),'cuda')
    model=FullCircuit(operator,cal,pool,inputs['drive_ids'],initial,parameters)
    drive=torch.as_tensor(inputs['drive'],device='cuda');target=torch.as_tensor(ref['command'],dtype=torch.float32,device='cuda')
    mask=(ref['time_s'][1::2]>=.5-1e-9)&(ref['time_s'][1::2]<2.2-1e-9)
    scale=torch.as_tensor(np.maximum(np.std(ref['command'][mask],axis=0),.1),dtype=torch.float32,device='cuda')
    observations=[]
    def observe(phase,start,state,gradient):
        entry={'phase':phase,'start_control':start,'wall_s':time.perf_counter()-began}
        if phase=='backward':
            arrays=[v.cpu().numpy() for v in state];raw=gradient.detach().cpu().numpy()
            finite=all(np.isfinite(v).all() for v in (*arrays,raw))
            entry.update(finite=finite,adjoint_norms=[float(np.linalg.norm(v.astype(float))) if np.isfinite(v).all() else None for v in arrays],
                gradient_norm=float(np.linalg.norm(raw.astype(float))) if np.isfinite(raw).all() else None)
            if not finite:np.savez_compressed(out/f'nonfinite-{start}.npz',rate_adjoint=arrays[0],adaptation_adjoint=arrays[1],activation_adjoint=arrays[2],gradient=raw)
        observations.append(entry)
        with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        if start%100==0:print(json.dumps(entry),flush=True)
        if entry['wall_s']>280:raise TimeoutError('Long-credit wall limit')
    result={'experiment':spec['experiment'],'adoption':'not_applicable','milestone_outcome':'pending'}
    try:
        objective,command,gradient,final=checkpointed_gradient(model,drive,target,torch.as_tensor(mask,device='cuda'),scale,observer=observe)
        command=command.cpu().numpy();full_gradient=gradient.cpu().numpy()
        np.savez_compressed(out/'gradient.npz',gradient=full_gradient,command=command,efficacy=cal['efficacy'])
        final_reference=StateStore.load(reference/'final-diagnostic')['network']
        parity={'command_max_abs':float(np.max(abs(command-ref['decoded_command']))),
            'final_rate_exact':np.array_equal(final[0].cpu().numpy(),final_reference['rate']),
            'final_adaptation_exact':np.array_equal(final[1].cpu().numpy(),final_reference['adaptation'])}
        if parity['command_max_abs']>1e-6 or not parity['final_rate_exact'] or not parity['final_adaptation_exact']:raise ValueError('Native forward parity changed')
        regularizer=2e-4*(cal['efficacy'].astype(float)-1)/len(cal['efficacy'])
        full=full_gradient.astype(float)-regularizer;short=old['gradient'].astype(float)-regularizer
        norm=np.linalg.norm(full);short_norm=np.linalg.norm(short)
        result.update(run_status='completed',validity='valid_full_discrete_adjoint',objective=objective,parity=parity,
            full_data_gradient_norm=float(norm),truncated_data_gradient_norm=float(short_norm),
            data_gradient_cosine=float(np.dot(full,short)/(norm*short_norm)) if norm*short_norm>0 else None,
            full_data_nonzero=int(np.count_nonzero(full)),truncated_data_nonzero=int(np.count_nonzero(short)))
    except (FloatingPointError,TimeoutError,ValueError) as exc:
        result.update(run_status='bounded_diagnostic_failure',validity='incomplete_full_adjoint',stop=type(exc).__name__+': '+str(exc))
    result.update(wall_s=time.perf_counter()-began,neural_model_s=len(observations)*.1,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','reference','fidelity','truncated','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.candidate,a.reference,a.fidelity,a.truncated,a.graph,a.out)
