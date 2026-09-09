#!/usr/bin/env python3
"""Prescribed overlap tests real head/world collision detection, not avoidance."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.body import FlyGymBody
from flylab.engine import config_values
from flylab.sensors import default_world
from flylab.c.sensors import CSensorAdapter
from flylab.c.integrity import write_json

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Evidence output must be new')
    world=default_world();world['sources']=[];world['obstacles']=[]
    base=FlyGymBody(42,world,config_values())
    try:
        saved=base.snapshot();head=base.frame()[0]['segments']['c_head'];clear=base.head_contact()
    finally:base.close()
    world['obstacles']=[dict(id='prescribed-head-overlap',p=head,r=head[1])]
    touched=FlyGymBody(42,world,config_values())
    try:
        # Restore the same integration state into recompiled collision geometry.
        # This deliberately starts overlapped and does not qualify a walking task.
        touched.restore(saved)
        sensor=CSensorAdapter(42,dict(kind='compressive-odor-v2',half_concentration=1.,extended_observations=True))
        sensor.observe(touched,world,.005,config_values())
        hit=touched.head_contact();feature=sensor.diagnostics['features']['head_contact']
        result=dict(status='PASS' if not clear and hit and feature==1. else 'FAIL',initial_clear=not clear,
                    prescribed_overlap_contact=bool(hit),head_contact_feature=feature,
                    collision_engine='actual MuJoCo mj_forward',locomotion_executed=False,
                    biologicalValidation=False,scope='Contact detection in a prescribed overlap; no bristle mechanics, force calibration or avoidance approval')
        write_json(out,result);print(result);return result
    finally:touched.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    raise SystemExit(0 if run(p.parse_args().out)['status']=='PASS' else 1)
