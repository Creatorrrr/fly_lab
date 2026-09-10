#!/usr/bin/env python3
"""Matched mechanical teacher / BANC feedback / cut-feedback force responses."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flylab.c.joint_teacher import JointTeacher
from flylab.c.muscles import MuscleRig
from flylab.c.single_joint import JointParameters,SingleJointLoop,build_profile
from flylab.c.integrity import write_json


def run(out,graph=None,backend='auto',seconds=.3):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=dict(schema='flylab.same-body-teacher-comparison.v1',seconds=seconds,dt=.001,
        initial_angles=[1.4,2.0],torques=[0.,.01,-.01],torque_interval_s=[.05,.08],
        recovery_interval_s=[.08,seconds],teacher=dict(kp_s2=900.,kd_s=60.),
        neural_gain_mV=18.,neural_rate_half_Hz=100.,neural_dt_s=.0001,physics_dt_s=.00001,
        feedback_iae_reduction=.20,max_deviation_ratio=1.05,
        interpretation='Teacher feasibility is separate from connectome feedback; default model unchanged')
    write_json(out/'spec.json',spec)
    modes=['passive','teacher']+(['neural','neural_cut'] if graph else [])
    traces={};result=dict(schema=spec['schema'],cases=[],feedback_status='NOT_RUN',biological_validation=False)
    for angle in spec['initial_angles']:
        for mode in modes:
            for torque in spec['torques']:
                name=f'{mode}-q{angle:g}-torque{torque:g}'
                body=MuscleRig()
                body.data.qpos[0]=angle;body.mj.mj_forward(body.model,body.data)
                loop=(SingleJointLoop(graph,build_profile(graph,body,JointParameters(initial_q_rad=angle)),backend,body=body)
                      if mode.startswith('neural') else JointTeacher(body))
                rows=[];fault=None;start=time.perf_counter()
                try:
                    for tick in range(round(seconds/.001)):
                        force=torque if .05<=tick*.001<.08 else 0.
                        frame=(loop.step(torque=force,feedback=mode!='neural_cut') if mode.startswith('neural')
                               else loop.step(angle,torque=force,enabled=mode=='teacher'))
                        rows.append(frame)
                        if frame.get('fault'):
                            fault=frame['fault'];break
                except RuntimeError as exc:fault=str(exc)
                write_json(out/(name+'.json'),rows)
                traces[(angle,mode,torque)]=rows
                result['cases'].append(dict(name=name,angle=angle,mode=mode,torque=torque,
                    controls=len(rows),fault=fault,wall_s=time.perf_counter()-start,
                    simulated_nodes=graph.n if mode.startswith('neural') else 0))
                print(name,'fault=',fault,flush=True)
                write_json(out/'report.json',result)
    comparisons=[]
    for angle in spec['initial_angles']:
        for torque in (.01,-.01):
            response={}
            for mode in modes:
                forced,free=traces[(angle,mode,torque)],traces[(angle,mode,0.)]
                if len(forced)!=round(seconds/.001) or len(free)!=len(forced):
                    response[mode]=dict(status='INCOMPLETE');continue
                difference=np.asarray([r['physics']['q_rad'] for r in forced])-np.asarray([r['physics']['q_rad'] for r in free])
                recovery=difference[80:]
                response[mode]=dict(status='COMPLETE',iae=float(np.abs(recovery).sum()*.001),
                                    peak=float(np.max(np.abs(recovery))))
            comparisons.append(dict(angle=angle,torque=torque,responses=response))
    result['comparisons']=comparisons
    if graph:
        checks=[]
        for row in comparisons:
            on,off=row['responses']['neural'],row['responses']['neural_cut']
            checks.append(on['status']==off['status']=='COMPLETE' and off['iae']>0 and
                          on['iae']<=off['iae']*.8 and on['peak']<=off['peak']*1.05)
        result['feedback_status']='PASS_RESPONSE_CRITERIA' if all(checks) else 'FAIL_RESPONSE_CRITERIA'
        result['simulated_nodes']=graph.n;result['graph_hash']=graph.hash
    result['status']='COMPLETE' if all(r['fault'] is None and r['controls']==round(seconds/.001) for r in result['cases']) else 'FAILED_OR_INCOMPLETE_CASES'
    result['default_changed']=False
    write_json(out/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--graph',type=Path)
    p.add_argument('--backend',default='auto',choices=('auto','exp_lif_cuda','exp_lif_mps','exp_lif_cpu_reference'))
    a=p.parse_args()
    from flylab.c.graph import GraphStore
    report=run(a.out,GraphStore.load(a.graph) if a.graph else None,a.backend)
    print(report['status'],report['feedback_status'])
    raise SystemExit(0 if report['status']=='COMPLETE' else 1)
