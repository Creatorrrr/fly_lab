#!/usr/bin/env python3
"""Predeclare repaired-profile assays; direct stimulation is always explicit."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.campaign import scene_world,validate_spec
from flylab.c.integrity import read_json,write_json


def build(bindings,kind):
    motor=bindings.spec['motor'];forward=motor['forward']['ids']
    if bindings.spec.get('neuromuscular'):raise ValueError('Descending-adapter assay required')
    def case(name,scene,seconds,task='diagnostic'):
        return dict(name=name,scene=scene,mode='C_STRICT',seconds=seconds,seed=42,task=task)
    if kind=='halting':
        stop=motor['stop']['ids']
        fg=next((c['ids'] for c,_ in bindings.research_cohorts if c['name'].startswith('Foxglove')),None)
        if not stop or not fg:raise ValueError('Reviewed BB and FG identities required')
        cases=[]
        for name,ids,muted in (('walking-control',[],False),('bluebell',stop,False),('foxglove',fg,False),('bluebell-synapses-muted',stop,True)):
            c=case(name,'baseline',6.,'stop_resume' if ids else 'diagnostic')
            c['initial_pose']=dict(position=[-15.,.8,0.],yaw_rad=0.)
            c['interventions']=[dict(kind='sensor_off',channels=['*'],duration_controls=1200),
                dict(kind='stimulate',ids=forward,amplitude_mV=20.,duration_controls=1200),
                dict(kind='suppress_spiking',ids=motor['yaw_left']['ids']+motor['yaw_right']['ids'],duration_controls=1200)]
            if ids:
                c['task_parameters']=dict(stop_onset_s=2.,stop_release_s=3.5)
                c['interventions'].append(dict(kind='stimulate',ids=ids,at_tick=20000,amplitude_mV=20.,duration_controls=300))
            if muted:c['interventions'].append(dict(kind='mute_outgoing',ids=ids,at_tick=20000,duration_controls=300))
            cases.append(c)
    elif kind=='hazard':
        base=case('evoked-hazard','hazard_right',10.,'hazard')
        world=scene_world('baseline');world['sources']=[dict(id='hazard',kind='hazard',p=[8.,.7,0.],strength=1.2)]
        base['world']=world
        base['interventions']=[dict(kind='stimulate',ids=forward,amplitude_mV=20.,duration_controls=2000)]
        free=copy.deepcopy(base);free['name']+='-hazard-free';free['world']['sources']=[]
        off=copy.deepcopy(base);off['name']+='-sensory-off'
        off['interventions'].append(dict(kind='sensor_off',channels=['danger'],duration_controls=2000))
        cases=[base,free,off]
    elif kind=='food':
        cases=[]
        for side in ('left','right'):
            c=case('food-'+side,'food_'+side,30.,'food');cases.append(c)
            off=copy.deepcopy(c);off['name']+='-sensory-off'
            off['interventions']=[dict(kind='sensor_off',channels=['odor_left','odor_right'],duration_controls=6000)]
            cases.append(off)
    elif kind=='obstacle':
        c=case('evoked-obstacle','front_obstacle',10.,'obstacle')
        c['interventions']=[dict(kind='stimulate',ids=forward,amplitude_mV=20.,duration_controls=2000)]
        free=copy.deepcopy(c);free.update(name='evoked-obstacle-free',scene='baseline',task='diagnostic')
        off=copy.deepcopy(c);off['name']='evoked-obstacle-sensory-off'
        off['interventions'].append(dict(kind='sensor_off',channels=['loom_left','loom_right','head_contact'],duration_controls=2000))
        cases=[c,free,off]
    else:raise ValueError('Unknown assay kind')
    return validate_spec(dict(schema='flylab.campaign.v1',cases=cases))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--bindings',default='data/fafb783/bindings-walking-visual-contact-walk-off-v6.json')
    p.add_argument('--kind',choices=('halting','hazard','food','obstacle'),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new fixed specification path')
    g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings))
    write_json(a.out,build(b,a.kind))
