"""Finite neural-state perturbations with unchanged native BANC parameters."""

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
from tools.train_c_banc_m3_full_circuit import AnatomicalOperator, FullCircuit
from tools.verify_c_banc_m3_full_trial import CalibratedTrial, calibrated_weights


def run(source,candidate,fidelity,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E26-finite-neural-state-sensitivity',
        'hypothesis':'Small finite VNC nonmotor rate perturbations propagate into material motor-command differences even though E25 single-group weight ULP changes did not',
        'intervention':'Same frozen E20 W0*theta/full158706/native.25ms and saved body-only sensory input. Generate own first.5s state, then exactly once add antithetic fixed Rademacher +/-.01 raw rate units to VNC intrinsic nonmotor cells whose OWN rate is in(.1,199.9). No z/A change, direct motor perturbation or clipping. Seed20260919 and graph index order fixed before exposure.',
        'interpretation':'This is a finite neural-state perturbation diagnostic under body teaching. No physical pose is changed, no F6 confirmation pose is exposed, no trained candidate or biological-chaos claim.',
        'measurement':'Initial actual perturbation magnitude, both branches versus the unchanged E22b trajectory; last100ms command and motor differences at.1/.3/.7/1.7s after injection. Material screen qRMS>.02rad or motorRMS>1; no threshold search.',
        'limits':'One GPU,.5s prefix+2*1.7s=3.9neuralmodelsec/285wallsec; no optimization/adoption',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'fidelity_sha256':sha256(fidelity/'result.json'),
        'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'full-circuit.py').write_bytes(Path('tools/train_c_banc_m3_full_circuit.py').read_bytes())
    check=json.loads((fidelity/'result.json').read_text())
    if not check['solver_screen_passed'] or check['candidate_sha256']!=spec['candidate_sha256']:raise ValueError('Matching native source fidelity required')
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    cal=dict(np.load(candidate/'candidate.npz'));inputs=dict(np.load(fidelity/'sensory-input.npz'));native=dict(np.load(fidelity/'replay.npz'))
    s=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,s,cal,learned=False);positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Wrong absolute source weights')
        parameters={k:getattr(s.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        initial=s.network.snapshot();pool=s.adapter.pool.copy()
        if not np.all(initial['output_mask']==1):raise ValueError('Full outgoing mask required')
    finally:s.close()
    operator=AnatomicalOperator(base,np.asarray(positions),cal['edge_parameters'],len(cal['efficacy']),'cuda')
    model=FullCircuit(operator,cal,pool,inputs['drive_ids'],initial,parameters);drive=torch.as_tensor(inputs['drive'],device='cuda')
    with torch.inference_mode():_prefix,rate,state=model(drive[:100],dt=.00025,return_state=True)
    if not np.array_equal(rate.cpu().numpy(),native['rate'][:100]):raise ValueError('Own prefix differs from native reference')
    raw_rate=state[0].cpu().numpy()
    population=np.array([n.get('super_class')=='ventral_nerve_cord_intrinsic' for n in g.nodes])
    population[cal['motor_ids']]=False
    ids=np.flatnonzero(population&(raw_rate>.1)&(raw_rate<199.9))
    if not len(ids):raise ValueError('No active interior VNC nonmotor population')
    direction=np.random.default_rng(20260919).choice(np.array([-1.,1.],np.float32),len(ids))
    np.savez_compressed(out/'anchor.npz',rate=raw_rate,adaptation=state[1].cpu().numpy(),activation=state[2].cpu().numpy(),
        perturbed_ids=ids,direction=direction)
    results={}
    for name,sign in (('negative',-1),('positive',1)):
        with torch.inference_mode():
            branch=tuple(v.clone() for v in state)
            branch[0][ids]+=torch.as_tensor(sign*.01*direction,device='cuda')
            if bool(torch.any((branch[0]<0)|(branch[0]>200))):raise ValueError('Out-of-range rate perturbation')
            actual_delta=(branch[0]-state[0]).cpu().numpy()
            q,r,final_r,final_z=model(drive[100:],dt=.00025,initial_state=branch,return_final=True)
        arrays={k:v.cpu().numpy() for k,v in (('command',q),('rate',r),('final_rate',final_r),('final_adaptation',final_z))}
        np.savez_compressed(out/(name+'.npz'),**arrays,initial_rate_delta=actual_delta)
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite neural branch')
        qdiff=arrays['command']-native['command'][100:];rdiff=arrays['rate']-native['rate'][100:]
        horizons={}
        for seconds in (.1,.3,.7,1.7):
            hi=round(seconds/.005);window=slice(max(0,hi-20),hi)
            qrms=float(np.sqrt(np.mean(qdiff[window].astype(float)**2)));rrms=float(np.sqrt(np.mean(rdiff[window].astype(float)**2)))
            horizons[str(seconds)]={'command_rms_rad':qrms,'motor_rate_rms':rrms,'material_sensitivity':qrms>.02 or rrms>1}
        result={'perturbed_neurons':len(ids),'actual_initial_rate_rms_on_population':float(np.sqrt(np.mean(actual_delta[ids].astype(float)**2))),
            'initial_motor_delta_max':float(np.max(abs(actual_delta[cal['motor_ids']]))),'horizons_after_injection':horizons,
            'wall_s':time.perf_counter()-began}
        write_json(out/(name+'.json'),result);results[name]=result;print(json.dumps({name:result}),flush=True)
        if time.perf_counter()-began>280:raise TimeoutError('State sensitivity wall limit')
    write_json(out/'result.json',{'experiment':spec['experiment'],'run_status':'completed','validity':'valid_finite_neural_perturbation',
        'adoption':'not_applicable','milestone_outcome':'pending','neural_model_s':3.9,'wall_s':time.perf_counter()-began,'arms':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','fidelity','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.candidate,a.fidelity,a.graph,a.out)
