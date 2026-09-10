#!/usr/bin/env python3
"""Evaluate same-model Warp physics batches with recorded, held actuator input.

This measures physical worlds only. It is not a batched connectome campaign
or a closed-loop walking benchmark and never selects a production backend.
"""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[1,8,16,32])
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if any(n not in (1,2,8,16,32) for n in a.sizes) or not 1<=a.steps<=1000 or not 1<=a.repeats<=10:
        p.error('Use sizes 1,2,8,16,32; 1..1000 steps and 1..10 repeats')
    a.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import warp as wp
    import mujoco_warp as mjw
    import cupy as cp
    from flylab.body import FlyGymBody
    from flylab.physics import PhysicsProfile
    from flylab.sensors import default_world
    from flylab.engine import config_values
    b=FlyGymBody(42,default_world(),config_values(None),
        physics_profile=PhysicsProfile(noslip_iterations=0,multiccd=False))
    result=dict(schema='flylab.warp-batch-evaluation.v1',scope='physics only, held neutral actuator input',
        model_hash=b.model_hash,physics=b.physics_identity(),cases=[],default_changed=False,
        full_neural_campaign=False,biological_validation=False,
        isolation_check=dict(qpos_tolerance=1e-5,test='World 1 external force; compare untouched worlds in the same batch'))
    try:
        with wp.ScopedDevice('cuda:0'):
            model=mjw.put_model(b.m)
            for size in a.sizes:
                row=dict(worlds=size,status='RUNNING')
                try:
                    free_before,_=cp.cuda.runtime.memGetInfo()
                    d=mjw.put_data(b.m,b.d,nworld=size,njmax=4096,nconmax=1024)
                    start=time.perf_counter()
                    mjw.step(model,d);wp.synchronize()
                    warmup_s=time.perf_counter()-start
                    with wp.ScopedCapture() as capture:
                        for _ in range(a.steps):mjw.step(model,d)
                    # First graph launch can instantiate driver resources. It
                    # is a separate warm-up horizon, not a timed sample.
                    start=time.perf_counter()
                    wp.capture_launch(capture.graph);wp.synchronize()
                    graph_warmup_s=time.perf_counter()-start
                    times=[]
                    for repeat in range(a.repeats):
                        start=time.perf_counter()
                        wp.capture_launch(capture.graph);wp.synchronize()
                        times.append(time.perf_counter()-start)
                        if (d.nacon.numpy()[0]>d.naconmax or d.ncollision.numpy()[0]>d.naconmax or
                            d.nefc.numpy().max()>d.njmax):
                            raise RuntimeError('Batch capacity overflow; samples rejected')
                        if not np.isfinite(d.qpos.numpy()).all() or not np.isfinite(d.qvel.numpy()).all():
                            raise RuntimeError('Non-finite batch state')
                    free_after,_=cp.cuda.runtime.memGetInfo()
                    row.update(status='COMPLETE',warmup_s=warmup_s,
                        graph_warmup_s=graph_warmup_s,warmup_steps=1+a.steps,
                        sample_protocol='Consecutive horizons after separate kernel and graph warm-up',wall_s=times,
                        world_steps_per_s=float(size*a.steps/np.median(times)),
                        memory_allocated_delta_bytes=int(max(0,free_before-free_after)),
                        memory_note='Observed driver allocation delta, not peak VRAM; other processes can affect it',
                        contacts=int(d.nacon.numpy()[0]),max_constraints=int(d.nefc.numpy().max()))
                    np.savez_compressed(a.out/f'worlds-{size}.npz',qpos=d.qpos.numpy(),qvel=d.qvel.numpy(),time_s=d.time.numpy())
                    if size>=8:
                        probe=mjw.put_data(b.m,b.d,nworld=size,njmax=4096,nconmax=1024)
                        force=np.zeros((size,b.m.nbody,6),np.float32)
                        force[1,b.thorax,0]=b.weight0
                        probe.xfrc_applied.assign(force)
                        mjw.step(model,probe);wp.synchronize()
                        q=probe.qpos.numpy()
                        unchanged=np.max(np.abs(q[2:]-q[0]))
                        changed=np.max(np.abs(q[1]-q[0]))
                        row['isolation']=dict(untouched_max_abs=float(unchanged),perturbed_max_abs=float(changed),
                            passed=bool(unchanged<=1e-5 and changed>1e-7))
                        del probe
                    del capture,d
                except Exception as exc:
                    row.update(status='FAILED',error=str(exc))
                    result['cases'].append(row)
                    # Ascending batches: preserve smaller completed sizes and
                    # stop increasing allocation after a failed size.
                    break
                result['cases'].append(row)
                (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
                print(row,flush=True)
        result['status']='COMPLETE' if len(result['cases'])==len(a.sizes) and all(r['status']=='COMPLETE' for r in result['cases']) else 'PARTIAL_OR_FAILED'
    finally:
        b.close()
        (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
