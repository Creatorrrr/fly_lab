#!/usr/bin/env python3
"""Native CPU/options/Warp comparison; failed adoption never changes defaults."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=.02)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if not .005 <= args.seconds <= 1 or abs(args.seconds/.005-round(args.seconds/.005)) > 1e-8:
        parser.error('Seconds must be a 5 ms multiple in .005..1')
    if not 1 <= args.repeats <= 10 or len(args.seeds)>5:
        parser.error('Use 1..10 repeats and at most five seeds')
    args.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from flylab.physics import PhysicsProfile,body_factory
    from flylab.sensors import default_world
    from flylab.engine import config_values
    from flylab.c.storage import StateStore
    # Declared before the run. Diagnostic short-horizon tolerances, not a
    # statement that altered contact mechanics preserve behavioral outcomes.
    tolerances=dict(qpos_abs=1e-3,qvel_abs=.1,foot_force_bw_abs=.15)
    profiles=dict(A=PhysicsProfile(),
                  B=PhysicsProfile(noslip_iterations=0,multiccd=False),
                  C=PhysicsProfile(backend='warp'))
    result=dict(schema='flylab.warp-verification.v1',tolerances=tolerances,
        seconds=args.seconds,seeds=args.seeds,repeats=args.repeats,
        comparison='A current CPU; B CPU noslip=0, MULTICCD off; C matching Warp',
        cases=[],checks={},default_changed=False,biological_validation=False)
    traces={}
    try:
        for seed in args.seeds:
            for name,profile in profiles.items():
                began=time.perf_counter()
                body=body_factory(profile)(seed,default_world(),config_values(None))
                constructed=time.perf_counter()-began
                try:
                    source=body.snapshot()
                    timings=[]
                    for repeat in range(args.repeats):
                        if repeat:body.restore(source)
                        q,v,forces,tips=[],[],[],[]
                        started=time.perf_counter()
                        for i in range(round(args.seconds/.005)):
                            if i == 1:body.perturb(.25,.005)
                            body.step(dict(forwardSpeed=1.,yawRate=.1,verticalSpeed=0.),.005)
                            if body.fault:raise RuntimeError(body.fault)
                            q.append(body.d.qpos.copy());v.append(body.d.qvel.copy())
                            forces.append(body._contact_forces('feet')/body.weight0)
                            tips.append(body.d.xpos[body._tip_ids].copy())
                        timings.append(time.perf_counter()-started)
                        trace=dict(qpos=np.asarray(q),qvel=np.asarray(v),feet_bw=np.asarray(forces),tip_mm=np.asarray(tips))
                        np.savez_compressed(args.out/f'{name}-seed{seed}-repeat{repeat}.npz',**trace)
                        if repeat==0:traces[(seed,name)]=trace
                    result['cases'].append(dict(seed=seed,condition=name,build_s=constructed,
                        wall_s=timings,median_wall_s=float(np.median(timings)),physics=body.physics_identity()))
                    if name=='C':
                        # Disk serialization, forward continuation, restore, repeat.
                        checkpoint=args.out/f'warp-state-seed{seed}'
                        StateStore.save(checkpoint,body.snapshot())
                        saved=StateStore.load(checkpoint)
                        command=dict(forwardSpeed=.5,yawRate=-.1,verticalSpeed=0.)
                        body.step(command,.005)
                        expected=body.snapshot()
                        body.restore(saved)
                        body.step(command,.005)
                        actual=body.snapshot()
                        same=all(np.array_equal(actual['warp']['arrays'][key],expected['warp']['arrays'][key])
                                 for key in ('qpos','qvel','act','qacc_warmstart'))
                        result['checks'][f'restore_exact_seed{seed}']=bool(same)
                        result['checks'][f'clock_seed{seed}']=abs(body.physics_time()-(args.seconds+.005))<1e-10
                        # Cross-check the optimized collector against FlyGym's API.
                        segments=[body.BodySegment(f'{leg}_{link}') for leg in ('lf','lm','lh','rf','rm','rh')
                                  for link in ('tarsus1','tarsus2','tarsus3','tarsus4','tarsus5')]
                        official=body.sim.get_bodysegment_contact_forces(body.fly.name,segments,ground_only=True)
                        result['checks'][f'contact_api_seed{seed}']=bool(np.allclose(official,body._contact_forces('feet'),rtol=0,atol=1e-10))
                        cfg=config_values(None);cfg['gravity']=.9;cfg['friction']=.8
                        body.set_config(cfg)
                        body.step(command,.005)
                        result['checks'][f'config_step_seed{seed}']=not body.fault
                        bad=copy.deepcopy(saved);bad['warp']['arrays']['qpos']=bad['warp']['arrays']['qpos'][:,:-1]
                        before=body.d.qpos.copy()
                        try:body.restore(bad)
                        except ValueError:
                            result['checks'][f'bad_restore_atomic_seed{seed}']=bool(np.array_equal(before,body.d.qpos))
                        else:result['checks'][f'bad_restore_atomic_seed{seed}']=False
                finally:body.close()
        result['comparisons']=[]
        for seed in args.seeds:
            for left,right in (('A','B'),('B','C')):
                x,y=traces[(seed,left)],traces[(seed,right)]
                delta={key:float(np.max(np.abs(x[key]-y[key]))) for key in x}
                result['comparisons'].append(dict(seed=seed,left=left,right=right,max_abs_delta=delta))
                if left=='B':
                    result['checks'][f'numerical_seed{seed}']=(delta['qpos']<=tolerances['qpos_abs'] and
                        delta['qvel']<=tolerances['qvel_abs'] and delta['feet_bw']<=tolerances['foot_force_bw_abs'])
        gpu=np.median([row['median_wall_s'] for row in result['cases'] if row['condition']=='C'])
        cpu=np.median([row['median_wall_s'] for row in result['cases'] if row['condition']=='B'])
        result['warp_vs_matched_cpu_speedup']=float(cpu/gpu)
        result['correctness_pass']=all(result['checks'].values())
        result['single_world_adoption']=bool(result['correctness_pass'] and gpu<=cpu*.9)
        result['status']='PASS' if result['correctness_pass'] else 'FAIL_CORRECTNESS'
    except Exception as exc:
        result.update(status='FAIL_EXECUTION',error=str(exc),traceback=traceback.format_exc(),single_world_adoption=False)
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
