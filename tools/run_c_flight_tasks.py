#!/usr/bin/env python3
"""Run the official aerodynamic flight tasks with an explicit prototype pilot."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.flybody_flight import make_flight_env
from flylab.c.integrity import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=.3);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tasks',nargs='+',choices=('flight','takeoff','landing'),default=['flight','takeoff','landing'])
    p.add_argument('--wing-pattern',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(cases=[],biological_validation=False)
    for task in a.tasks:
        env=make_flight_env(a.source,task=task,seed=a.seed,seconds=a.seconds,wing_pattern=a.wing_pattern)
        rows=[];fault=None;began=time.perf_counter()
        try:
            env.reset()
            for _ in range(round(a.seconds/env.control_timestep())):
                t=env.step(np.zeros(env.action_spec().shape))
                rows.append(dict(env.task.last_metrics,time_s=env.physics.time(),reward=float(t.reward),
                    wing_angles=env.physics.bind(env.task._wing_joints).qpos.tolist()))
                if t.last():break
        except Exception as exc:fault=str(exc)
        finally:
            write_json(a.out/(task+'-trace.json'),rows)
            report['cases'].append(dict(task=task,provenance=env.flylab_provenance,fault=fault,steps=len(rows),
                seconds=rows[-1]['time_s'] if rows else 0.,wall_s=time.perf_counter()-began,
                complete=bool(rows and rows[-1]['time_s']>=a.seconds-1e-9),
                task_success=bool(rows and rows[-1]['success']),controller='upstream prototype wingbeat; zero residual policy'))
            env.close();write_json(a.out/'report.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
