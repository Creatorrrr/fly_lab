#!/usr/bin/env python3
"""Compare servo gains on disjoint recorded-motion windows without changing defaults."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.body import FlyGymBody
from flylab.engine import config_values
from flylab.sensors import default_world
from flylab.locomotion_experiments import replay_kinematics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--clip',type=Path);p.add_argument('--seconds',type=float,default=.1)
    p.add_argument('--train-start',type=float,default=0.);p.add_argument('--validation-start',type=float,default=.5)
    p.add_argument('--gains',type=float,nargs='+',default=[.5,1.,2.]);a=p.parse_args()
    if not a.gains or any(not np.isfinite(g) or not .1<=g<=5 for g in a.gains):p.error('Gain multipliers must be in .1..5')
    if abs(a.train_start-a.validation_start)<a.seconds:p.error('Calibration and validation windows must not overlap')
    a.out.mkdir(parents=True,exist_ok=False);reports=[]
    def run(gain,start,label):
        b=FlyGymBody(42,default_world(),config_values(None))
        try:
            # Position actuator gain and matching bias must change together.
            b.m.actuator_gainprm[b.act_ids,0]*=gain;b.m.actuator_biasprm[b.act_ids,1]*=gain
            report,trace=replay_kinematics(b,a.seconds,start_s=start,path=a.clip)
            np.savez_compressed(a.out/(label+'.npz'),**trace)
            return dict(multiplier=gain,**report)
        finally:b.close()
    for i,gain in enumerate(a.gains):reports.append(run(gain,a.train_start,'calibration-'+str(i)))
    candidates=[r for r in reports if r['completed']]
    if not candidates:raise RuntimeError('No completed calibration candidate')
    best=min(candidates,key=lambda r:r['q_rmse_rad'])['multiplier']
    validation=[run(g,a.validation_start,'validation-'+str(i)) for i,g in enumerate(dict.fromkeys([1.,best]))]
    output=dict(calibration=reports,selected_multiplier=best,validation=validation,
        selection_metric='calibration joint-angle RMSE, completed runs only',
        validation_scope='disjoint temporal window of the same clip; not an independent held-out animal',
        applied_to_default=False,biological_validation=False)
    (a.out/'report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected_multiplier=best,validation=[{k:r[k] for k in ('multiplier','q_rmse_rad','completed')} for r in validation],applied_to_default=False)))


if __name__=='__main__':main()
