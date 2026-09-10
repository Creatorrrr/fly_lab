#!/usr/bin/env python3
"""Run real recorded-motion, controller, terrain, FlyBody and LF muscle assays."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.sensors import default_world
from flylab.engine import config_values
from flylab.locomotion_experiments import run_controller,replay_kinematics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('replay','controllers','muscle'),required=True)
    p.add_argument('--seconds',type=float,default=.1);p.add_argument('--start',type=float,default=0.)
    p.add_argument('--clip',type=Path);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',choices=('neuromechfly','flybody'),default='neuromechfly')
    p.add_argument('--terrain',choices=('flat','gaps','blocks','mixed','slope'),default='flat')
    p.add_argument('--controllers',nargs='+',choices=('hybrid','cpg','rule'),default=['hybrid','cpg','rule'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not np.isfinite(a.seconds) or not .002<=a.seconds<=60:p.error('seconds must be in .002..60')
    a.out.mkdir(parents=True,exist_ok=False);reports=[];started=time.perf_counter()
    if a.task=='muscle':
        from flylab.c.muscle_imitation import MuscleImitation
        for kind in ('passive','teacher'):
            rig=MuscleImitation(a.seed);samples=[]
            try:
                for _ in range(round(a.seconds/.002)):
                    samples.append(rig.step(None if kind=='teacher' else np.zeros(15),teacher=kind=='teacher'))
                    if rig.done:break
                arrays={k:np.asarray([s[k] for s in samples]) for k in ('time_s','activation','q','target_q','qdot','target_qdot','forces','reward')}
                report=dict(kind=kind,model='official tethered LF 15-MTU model',model_hash=rig.identity,
                    seconds_executed=samples[-1]['time_s'],q_rmse_rad=float(np.sqrt(np.mean((arrays['q']-arrays['target_q'])**2))),
                    velocity_rmse_rad_s=float(np.sqrt(np.mean((arrays['qdot']-arrays['target_qdot'])**2))),
                    mean_reward=float(arrays['reward'].mean()),neural_control=False,biological_validation=False,
                    clip_use='bundled calibration clip; not held-out biological evidence')
                np.savez_compressed(a.out/(kind+'.npz'),**arrays);reports.append(report)
            finally:rig.close()
    else:
        options=BodyOptions(model=a.model,terrain=a.terrain,terrain_seed=a.seed)
        for kind in (['replay'] if a.task=='replay' else a.controllers):
            body=FlyGymBody(a.seed,default_world(),config_values(None),body_options=options)
            try:
                report,arrays=(replay_kinematics(body,a.seconds,start_s=a.start,path=a.clip) if kind=='replay' else run_controller(body,kind,a.seconds,a.seed))
                report.update(body_options=asdict(options),body_model_hash=body.model_hash,physics=body.physics_identity(),seed=a.seed)
                np.savez_compressed(a.out/(kind+'.npz'),**arrays);reports.append(report)
                if body.renderer is None:
                    from PIL import Image
                    Image.fromarray(body.preview()).save(a.out/(kind+'.png'))
            finally:body.close()
    output=dict(schema='flylab.flygym-experiments.v1',reports=reports,wall_s=time.perf_counter()-started,
        scope='mechanical reference experiments; does not promote BANC behavior or Warp physics',biological_validation=False)
    (a.out/'report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
