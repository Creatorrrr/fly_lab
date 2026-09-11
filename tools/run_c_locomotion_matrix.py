"""Longer independent controller/terrain checks on both official body models."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import write_json


def run_case(spec):
    import numpy as np
    from flylab.body import FlyGymBody
    from flylab.body_options import BodyOptions
    from flylab.engine import config_values
    from flylab.sensors import default_world
    from flylab.locomotion_experiments import run_controller
    path=Path(spec['out']);path.mkdir(parents=True,exist_ok=False)
    body=None;start=time.perf_counter()
    try:
        body=FlyGymBody(spec['seed'],default_world(),config_values(),body_options=BodyOptions(
            model=spec['model'],terrain=spec['terrain'],terrain_seed=spec['seed'],render_camera=False))
        result,trace=run_controller(body,spec['controller'],spec['seconds'],spec['seed'])
        np.savez_compressed(path/'trace.npz',**trace)
        result.update(model_hash=body.model_hash,physics=body.physics_identity())
    except Exception as exc:result=dict(completed=False,fault=f'{type(exc).__name__}: {exc}')
    finally:
        if body:body.close()
    result.update(spec,wall_s=time.perf_counter()-start)
    write_json(path/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=3.);p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--terrains',nargs='+',choices=['flat','gaps','blocks','mixed','slope'],default=['flat','gaps','blocks','mixed','slope'])
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not .005<=a.seconds<=60 or abs(a.seconds/.005-round(a.seconds/.005))>1e-8 or not 1<=a.workers<=8:
        p.error('Use integral 5 ms periods and 1..8 workers')
    if not 1<=len(a.seeds)<=5 or len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds):p.error('Use 1..5 distinct uint32 seeds')
    if len(set(a.terrains))!=len(a.terrains):p.error('Use distinct terrains')
    a.out.mkdir(parents=True,exist_ok=False);jobs=[];start=time.perf_counter()
    for model,terrain,controller,seed in product(('neuromechfly','flybody'),a.terrains,('hybrid','cpg','rule'),a.seeds):
        name=f'{model}-{terrain}-{controller}-{seed}'
        jobs.append(dict(model=model,terrain=terrain,controller=controller,seed=seed,seconds=a.seconds,out=str(a.out/name)))
    report=dict(cases=[],planned=len(jobs),scope='mechanical controller comparisons; no connectome simulation or biological validation',biological_validation=False)
    with ProcessPoolExecutor(max_workers=a.workers) as workers:
        pending=[workers.submit(run_case,j) for j in jobs]
        for completed in as_completed(pending):
            result=completed.result();report['cases'].append(result)
            write_json(a.out/'report.json',report)
            print(Path(result['out']).name,result['completed'],result.get('fault'),flush=True)
    report.update(wall_s=time.perf_counter()-start,completed_cases=sum(c['completed'] for c in report['cases']),
        faulted_cases=sum(bool(c.get('fault')) for c in report['cases']),execution_complete=len(report['cases'])==len(jobs))
    write_json(a.out/'report.json',report)


if __name__=='__main__':main()
