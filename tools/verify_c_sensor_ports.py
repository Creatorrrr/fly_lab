#!/usr/bin/env python3
"""Actual geometry to coarse sensory ports, with delayed/disabled controls."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.body import FlyGymBody
from flylab.engine import config_values
from flylab.sensors import default_world
from flylab.c.sensors import CSensorAdapter
from flylab.c.ports import PortBindings,SensoryEncoder
from flylab.c.graph import GraphStore
from flylab.c.integrity import read_json,write_json

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load('data/fafb783/bundle');b=PortBindings(g,read_json('data/fafb783/bindings-visual-head-contact-research-v1.json'))
    results=[]
    for side,z in [('left',-2.),('right',2.)]:
        sensor=CSensorAdapter(42,b.spec['sensor_model']);packets=[]
        for distance in (10.,4.):
            w=default_world();w['sources']=[];w['obstacles']=[dict(id='obstacle',p=[distance,1.,z],r=1.)]
            body=FlyGymBody(42,w,config_values())
            try:
                body.step(dict(forwardSpeed=0.,yawRate=0.,verticalSpeed=0.),.005)
                packet=sensor.observe(body,w,.005,config_values())
                packets.append(dict(packet=packet,diagnostics=copy.deepcopy(sensor.diagnostics),physical_time=body.physics_time()))
            finally:body.close()
        features=packets[-1]['diagnostics']['features'];on=SensoryEncoder(b,42);off=SensoryEncoder(b,42)
        values=[]
        for _ in range(2):
            drive,_,ports=on.encode(packets[-1]['packet'],.005,.0001,supplemental=features)
            disabled,_,_=off.encode(packets[-1]['packet'],.005,.0001,disabled=['*'],supplemental=features)
            values.append(ports)
        expected='loom_'+side;other='loom_'+('right' if side=='left' else 'left')
        delayed=all(p['value']==0 for p in values[0] if p['name'].startswith('loom'))
        active=[p for p in values[1] if p['name']=='loom_LC4_'+side][0]['value']>0
        passed=features[expected]>features[other] and delayed and active and not np.any(disabled)
        results.append(dict(side=side,status='PASS' if passed else 'FAIL',packets=packets,ports=values,
                            one_control_delay=delayed,disabled_input_zero=not bool(np.any(disabled))))
    r=dict(status='PASS' if all(x['status']=='PASS' for x in results) else 'FAIL',graph_hash=g.hash,binding_hash=b.hash,cases=results,
           physicalExecuted=True,neuralPropagation=False,head_contact='no-contact baseline only; force/area calibration not claimed',
           interpretation='Independent actual MuJoCo geometries test left/right silhouette expansion; no obstacle avoidance or retinal reconstruction approval.')
    write_json(out/'report.json',r);print(r['status']);return r
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    raise SystemExit(0 if run(p.parse_args().out)['status']=='PASS' else 1)
