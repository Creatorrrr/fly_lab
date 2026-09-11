#!/usr/bin/env python3
"""Isolate solver configuration and contact order without changing defaults."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import warp as wp
    from flylab.body_warp_resident import ResidentWarpBody
    from flylab.body import FlyGymBody
    from flylab.physics import PhysicsProfile
    from flylab.sensors import default_world
    from flylab.engine import config_values
    from flylab.warp_contacts import OrderedContacts
    from flylab.c.integrity import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--compare-effective',action='store_true',help='Compare CPU nominal and actual GPU solver options over 20 ms')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);report=dict(cases=[],default_changed=False)
    for ordered in (False,True):
        b=ResidentWarpBody(42,default_world(),config_values(None),physics_profile=PhysicsProfile(backend='warp',control_backend='cuda'))
        try:
            initial=b.snapshot()
            if ordered:
                b.ordering=OrderedContacts(b.gpu_model,b.gpu_data,b.stream)
                b.gpu_model.callback.contactfilter=b.ordering
                with wp.ScopedStream(b.stream):b.ordering(b.gpu_model,b.gpu_data)
                b.period_graphs.clear()
            b.step(dict(forwardSpeed=1.,yawRate=.1),.005)
            saved=b.snapshot();traces=[]
            for repeat in range(3):
                b.restore(saved);b.step(dict(forwardSpeed=1.,yawRate=.1),.005)
                traces.append(b.snapshot()['warp']['arrays'])
            differences={key:float(max(np.max(np.abs(t[key]-traces[0][key])) for t in traces[1:])) for key in ('qpos','qvel','qacc_warmstart')}
            report['cases'].append(dict(ordered_contacts=ordered,differences=differences,
                exact=all(value==0 for value in differences.values()),
                tolerance=float(b.gpu_model.opt.tolerance.numpy()[0]),disableflags=int(b.gpu_model.opt.disableflags)))
            if not ordered and a.compare_effective:
                def trace(body):
                    values={key:[] for key in ('qpos','qvel','feet_bw')}
                    for _ in range(4):
                        body.step(dict(forwardSpeed=1.,yawRate=.1),.005)
                        if body.fault:raise RuntimeError(body.fault)
                        values['qpos'].append(body.d.qpos.copy());values['qvel'].append(body.d.qvel.copy())
                        values['feet_bw'].append(body._contact_forces('feet')/body.weight0)
                    return {k:np.asarray(v) for k,v in values.items()}
                b.restore(initial);gpu=trace(b)
                np.savez_compressed(a.out/'gpu-trace.npz',**gpu)
                report['effective_solver_comparisons']=[]
                report['existing_bounds']=dict(qpos=.001,qvel=.1,feet_bw=.15)
                for match_effective in (False,True):
                    cpu=FlyGymBody(42,default_world(),config_values(None),physics_profile=PhysicsProfile(noslip_iterations=0,multiccd=False))
                    try:
                        if match_effective:
                            cpu.m.opt.tolerance=float(b.gpu_model.opt.tolerance.numpy()[0])
                            cpu.m.opt.disableflags |= int(cpu.mj.mjtDisableBit.mjDSBL_ISLAND)
                        cpu._forward_physics();reference=trace(cpu)
                        np.savez_compressed(a.out/f'cpu-effective-{match_effective}.npz',**reference)
                        difference={key:float(np.max(np.abs(reference[key]-gpu[key]))) for key in gpu}
                        report['effective_solver_comparisons'].append(dict(match_effective=match_effective,
                            cpu=cpu.physics_identity(),gpu=b.physics_identity(),max_abs=difference,
                            within_existing_bounds=all(difference[k]<=v for k,v in report['existing_bounds'].items()),
                            note='CPU options overridden only in this diagnostic after common neutral settling; no default or checkpoint is changed'))
                    finally:cpu.close()
            write_json(a.out/'report.json',report);print(report['cases'][-1],flush=True)
        finally:b.close()


if __name__=='__main__':main()
