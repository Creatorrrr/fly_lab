#!/usr/bin/env python3
"""Full declared BANC calculation plus selected tibia motor/actuator experiment."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.neural import BACKEND_CHOICES
from flylab.c.backend_selection import resolve_backend
from flylab.c.cns import tibia_bindings,TibiaDecoder,physical_joint_step
from flylab.c.neural import create_backend
from flylab.c.storage import StateStore
from flylab.c.integrity import write_json
from flylab.body import FlyGymBody
from flylab.engine import config_values
from flylab.sensors import default_world

def run(graph_path,out,seconds=.2,backend='auto'):
    backend=resolve_backend(backend)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);g=GraphStore.load(graph_path);spec=tibia_bindings(g)
    write_json(out/'tibia_bindings.json',spec);results=[]
    for role in ('control','flexor','extensor','motor_disconnected','descending_DNp09'):
        world=default_world();world['sources']=[];world['obstacles']=[]
        body=FlyGymBody(42,world,config_values());n=create_backend(g,backend=backend);decoder=TibiaDecoder(g,spec,body)
        if hasattr(n,'set_readout_cohort'):n.set_readout_cohort(decoder.indices)
        trace=[];initial=body.frame()[1]['jointAngles']
        try:
            drive=np.zeros(g.n,np.float32)
            if role=='descending_DNp09':
                ids=[node['id'] for node in g.nodes if node.get('cell_type')=='DNp09' and node.get('super_class')=='descending']
                if len(ids)!=2:raise ValueError('Reviewed BANC DNp09 pair required')
                drive[g.resolve(ids)]=20.
            elif role!='control':drive[g.resolve(spec['rows'][0]['groups']['flexor' if role=='motor_disconnected' else role])]=20.
            def step():
                targets,diagnostics=decoder.decode(n,disconnected=role=='motor_disconnected')
                if hasattr(n,'begin_advance'):
                    n.begin_advance(drive,50,decoder.indices)
                    try:physical_joint_step(body,targets)
                    finally:n.finish_advance()
                else:
                    n.advance(drive,50,decoder.indices);physical_joint_step(body,targets)
                return diagnostics
            for _ in range(round(seconds/.005)):
                diagnostics=step()
                b,phy=body.frame();trace.append(dict(tick=n.tick,body=b,diagnostics=diagnostics,joint_angles=phy['jointAngles']))
                if body.fault:break
            state=dict(neural=n.snapshot(),body=body.snapshot(),actuator=decoder.snapshot())
            StateStore.save(out/(role+'-final'),state)
            for _ in range(4):step()
            expected=dict(neural=n.snapshot(),body=body.snapshot(),actuator=decoder.snapshot())
            n.restore(state['neural']);body.restore(state['body']);decoder.restore(state['actuator'])
            for _ in range(4):step()
            restore_equal=all(np.array_equal(expected['neural'][k],n.snapshot()[k]) for k in ('v','h','queue','spike_count'))
            restore_equal &= np.array_equal(expected['body']['state'],body.snapshot()['state'])
            restore_equal &= np.array_equal(expected['actuator']['offset'],decoder.snapshot()['offset'])
            points=np.asarray(trace[-1]['body']['legs']['lf']);a=points[1]-points[2];b=points[3]-points[2]
            knee=float(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1.,1.)))
            result=dict(role=role,trace=trace,initial_angles=initial,fault=body.fault,simulated_nodes=g.n,
                        backend=n.backend,neural_tick=state['neural']['tick'],physicalExecuted=True,restore_equal=bool(restore_equal),
                        left_front_knee_interior_rad=knee,scope='Six tibia DOFs; remaining joints neutral; no proprioceptive feedback',
                        biological_validation=False,autonomous_walking_validation=False)
            write_json(out/(role+'.json'),result);results.append({k:v for k,v in result.items() if k!='trace'})
            print(role,n.tick,decoder.offset.tolist(),flush=True)
        finally:body.close()
    values={r['role']:r for r in results}
    direction=values['flexor']['left_front_knee_interior_rad']<values['control']['left_front_knee_interior_rad']<values['extensor']['left_front_knee_interior_rad']
    disconnected=values['motor_disconnected']['left_front_knee_interior_rad']==values['control']['left_front_knee_interior_rad']
    report=dict(status='PASS' if direction and disconnected and all(r['restore_equal'] and not r['fault'] for r in results) else 'FAIL',
               graph=g.manifest,binding=spec,cases=results,actual_flexion_extension=bool(direction),motor_disconnection_equal=bool(disconnected),
               autonomous_walk='NOT_VALIDATED',proprioception='BLOCKED_UNVALIDATED_MAPPING')
    write_json(out/'report.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',required=True);p.add_argument('--out',required=True);p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto');a=p.parse_args()
    raise SystemExit(0 if run(a.graph,a.out,backend=a.backend)['status']=='PASS' else 1)
