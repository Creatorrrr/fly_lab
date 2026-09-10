#!/usr/bin/env python3
"""Prepare direct-neuron output diagnostics, separate from natural behavior."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.campaign import validate_spec
from flylab.c.integrity import read_json,write_json


def build(bindings):
    if bindings.spec.get('neuromuscular'):raise ValueError('These assays target a descending-to-CPG adapter')
    motor=bindings.spec['motor'];cases=[]
    for role,task,seconds in (('forward','walking',10.),('backward','backward',2.),('yaw_left','yaw_left',2.),('yaw_right','yaw_right',2.),('forward','stop_resume',6.)):
        if not motor[role]['ids']:raise ValueError('Empty output group: '+role)
        controls=round(seconds/.005)
        events=[dict(kind='sensor_off',channels=['*'],duration_controls=controls),
                dict(kind='stimulate',ids=motor[role]['ids'],amplitude_mV=20.,duration_controls=controls)]
        if role=='forward':
            events.append(dict(kind='suppress_spiking',ids=motor['yaw_left']['ids']+motor['yaw_right']['ids'],duration_controls=controls))
        case=dict(name='direct-'+task,mode='C_STRICT',scene='baseline',task=task,seed=42,seconds=seconds,interventions=events)
        if task=='walking':
            # A 10 s forward-drive diagnostic from the center reaches a wall
            # after about 4 s. Use the arena diagonal for actuator validation;
            # keep the original task duration, progress and stuck thresholds.
            case['initial_pose']=dict(position=[-21.5,.8,-15.5],yaw_rad=math.atan2(31.,43.))
        if task=='stop_resume':
            # Test whether removing and restoring ongoing DN spikes controls
            # a moving body. This does not identify a natural halting circuit.
            events.append(dict(kind='suppress_spiking',ids=motor['forward']['ids'],at_tick=20000,duration_controls=300))
            case['task_parameters']=dict(stop_onset_s=2.,stop_release_s=3.5)
        cases.append(case)
    return validate_spec(dict(schema='flylab.campaign.v1',cases=cases))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--bindings',default='data/fafb783/bindings-walking-poisson-reset-current-v3.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new assay specification path')
    g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings));spec=build(b)
    a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,spec)
    print(len(spec['cases']),'direct neural diagnostics prepared; natural_behavior=False')
