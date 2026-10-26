"""One-ULP parameter sensitivity under the same complete neural trajectory."""

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


def run(source,candidate,reference,fidelity,adjoint,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    cal=dict(np.load(candidate/'candidate.npz'));grad=dict(np.load(adjoint/'gradient.npz'))
    if not np.array_equal(cal['efficacy'],grad['efficacy']):raise ValueError('Adjoint/candidate identity differs')
    eligible=(cal['efficacy']>.1)&(cal['efficacy']<10)&np.isfinite(grad['gradient'])
    if not np.any(eligible):raise ValueError('No interior parameter')
    selected=int(np.argmax(np.where(eligible,abs(grad['gradient']),-1)))
    original=cal['efficacy'][selected]
    values=[np.nextafter(original,np.float32(-np.inf)),np.nextafter(original,np.float32(np.inf))]
    spec={'experiment':'M3-E25-native-weight-resolution',
        'hypothesis':'The full-trajectory sensitivity is large enough that a minimum representable efficacy change materially alters the native neural trajectory',
        'selection':'Highest absolute E24 full gradient among efficacy parameters strictly inside [.1,10], before either perturbation is exposed',
        'intervention':'Exactly one float32 ULP down and up in that one shared efficacy; no parameter search or adoption. Original W0 times theta, original edge/sign mask. Same full native .25ms BANC and body-derived input/initial state. No motor clamp or external neural phase.',
        'measurement':'Both actual source-weight bit changes and resulting motor/command differences; predicted differential objective versus actual objective. A large response is finite-horizon numerical sensitivity, not a validated Lyapunov exponent or biological chaos claim.',
        'gate':'Flag material sensitivity if command difference RMS>.02rad or motor rate difference RMS>1, the existing numerical-fidelity limits; no walking/adoption inference.',
        'limits':'One GPU,2arms*2.2s=4.4neuralmodelsec/285wallsec, no optimization',
        'selected_parameter':selected,'original':float(original),'probes':[float(v) for v in values],
        'gradient':float(grad['gradient'][selected]),'candidate_sha256':sha256(candidate/'candidate.npz'),
        'adjoint_sha256':sha256(adjoint/'gradient.npz'),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'full-circuit.py').write_bytes(Path('tools/train_c_banc_m3_full_circuit.py').read_bytes())
    began=time.perf_counter();g=GraphStore.load(graph_path);origin=StateStore.load(source/'start')
    ref=dict(np.load(reference/'signals.npz'));inputs=dict(np.load(fidelity/'sensory-input.npz'))
    native=dict(np.load(fidelity/'replay.npz'))
    s=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
    try:
        base,_=calibrated_weights(g,s,cal,learned=False);positions=[]
        for post in cal['motor_ids']:
            ix=np.arange(base.indptr[post],base.indptr[post+1]);positions.extend(ix[base.data[ix]!=0].tolist())
        if not np.array_equal(base.data[positions],cal['incoming_source_data']):raise ValueError('Wrong original source weights')
        parameters={k:getattr(s.network,k).cpu().numpy().copy() for k in ('tau','a','threshold','cap')}
        initial=s.network.snapshot();pool=s.adapter.pool.copy()
        if not np.all(initial['output_mask']==1):raise ValueError('Full outgoing mask required')
    finally:s.close()
    operator=AnatomicalOperator(base,np.asarray(positions),cal['edge_parameters'],len(cal['efficacy']),'cuda')
    model=FullCircuit(operator,cal,pool,inputs['drive_ids'],initial,parameters)
    drive=torch.as_tensor(inputs['drive'],device='cuda')
    mask=(ref['time_s'][1::2]>=.5-1e-9)&(ref['time_s'][1::2]<2.2-1e-9)
    target=ref['command'].astype(np.float32);scale=np.maximum(np.std(ref['command'][mask],axis=0),.1).astype(np.float32)
    def loss(q,theta):
        return float(np.mean(((q[mask]-target[mask])/scale)**2,dtype=np.float64)+1e-4*np.mean((theta-1)**2,dtype=np.float64))
    baseline_loss=loss(native['command'],cal['efficacy']);results={}
    for name,value in zip(('minus_one_ulp','plus_one_ulp'),values):
        theta=cal['efficacy'].copy();theta[selected]=value
        with torch.no_grad():model.efficacy.copy_(torch.as_tensor(theta,device='cuda'))
        with torch.inference_mode():q,r,final_r,final_z=model(drive,dt=.00025,return_final=True)
        arrays={k:v.cpu().numpy() for k,v in (('command',q),('rate',r),('final_rate',final_r),('final_adaptation',final_z))}
        np.savez_compressed(out/(name+'.npz'),**arrays,efficacy=theta)
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite native perturbation')
        changed=cal['incoming_source_data']*theta[cal['edge_parameters']]
        delta=float(value)-float(original);actual_loss=loss(arrays['command'],theta)
        qerror=arrays['command']-native['command'];rerror=arrays['rate']-native['rate']
        qrms=float(np.sqrt(np.mean(qerror.astype(float)**2)));rrms=float(np.sqrt(np.mean(rerror.astype(float)**2)))
        per_control=np.sqrt(np.mean(qerror.astype(float)**2,axis=1));times=ref['time_s'][1::2]
        crossing=np.flatnonzero(per_control>.02)
        result={'efficacy_delta':delta,'changed_stored_source_weights':int(np.count_nonzero(changed!=cal['incoming_data'])),
            'actual_objective':actual_loss,'actual_objective_delta':actual_loss-baseline_loss,
            'first_order_predicted_objective_delta':float(grad['gradient'][selected])*delta,
            'command_difference_rms_rad':qrms,'motor_rate_difference_rms':rrms,
            'first_command_difference_over_02rad_s':float(times[crossing[0]]) if len(crossing) else None,
            'material_sensitivity':qrms>.02 or rrms>1,'wall_s':time.perf_counter()-began}
        write_json(out/(name+'.json'),result);results[name]=result;print(json.dumps({name:result}),flush=True)
        if time.perf_counter()-began>280:raise TimeoutError('Weight-resolution wall limit')
    write_json(out/'result.json',{'experiment':spec['experiment'],'run_status':'completed','validity':'valid_native_weight_sensitivity',
        'adoption':'not_applicable','milestone_outcome':'pending','neural_model_s':4.4,'wall_s':time.perf_counter()-began,
        'baseline_objective':baseline_loss,'selected_parameter':selected,'arms':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','reference','fidelity','adjoint','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.candidate,a.reference,a.fidelity,a.adjoint,a.graph,a.out)
