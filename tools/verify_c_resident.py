#!/usr/bin/env python3
"""Compare resident control without relaxing the existing Warp adoption bounds."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from flylab.physics import PhysicsProfile,body_factory
    from flylab.sensors import default_world
    from flylab.engine import config_values
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]);a=p.parse_args()
    if not 1<=len(a.seeds)<=5:p.error('Use 1..5 seeds')
    a.out.mkdir(parents=True,exist_ok=False)
    profiles=dict(cpu=PhysicsProfile(),matching_cpu=PhysicsProfile(noslip_iterations=0,multiccd=False),
        warp_host=PhysicsProfile(backend='warp'),resident=PhysicsProfile(backend='warp',control_backend='cuda'))
    output=dict(tolerances=dict(qpos_abs=.001,qvel_abs=.1,foot_force_bw_abs=.15),
        controller_tolerances=dict(action_abs=3e-6,state_abs=1e-10),cases=[],comparisons=[],
        timing_scope='body plus controller, .02 s windows, 3 repeats after graph warmup; no neural/sensors/recording',
        biological_validation=False,default_changed=False)
    command=dict(forwardSpeed=1.,yawRate=.1);traces={}
    for seed in a.seeds:
        for name,profile in profiles.items():
            b=body_factory(profile)(seed,default_world(),config_values(None))
            try:
                initial=b.snapshot()
                b.step(command,.005);b.restore(initial)  # compilation is outside timing
                parity=[]
                if name=='resident':
                    for cmd in (command,dict(forwardSpeed=-1.,yawRate=-.4),dict(forwardSpeed=0.,yawRate=0.)):
                        before=b.snapshot();drive=b.motor.map(cmd)
                        expected=b.Action(joint_angles=b.neutral.copy(),adhesion_onoff=np.full(6,b.controller.enable_adhesion)) if np.max(np.abs(drive))<1e-7 else b._stepper.step(drive,b._observation())
                        c=b.controller;network=c.cpg_network
                        cpu_state=np.concatenate([network.curr_phases,network.curr_magnitudes,c.retraction_correction,c.stumbling_correction,c.retraction_persistence_counter])
                        b.resident.set_inputs([cmd])
                        with b.wp.ScopedStream(b.stream):b.resident.enqueue()
                        ctrl=b.resident.views['ctrl'].get(stream=b.resident.stream)[0]
                        gpu_state=b.resident.state.get(stream=b.resident.stream)[0]
                        parity.append(dict(command=cmd,action_abs=float(np.max(np.abs(ctrl[b.act_ids]-expected.joint_angles))),
                            state_abs=float(np.max(np.abs(gpu_state-cpu_state))),adhesion_equal=bool(np.array_equal(ctrl[b.sim._intern_adhesionactuatorids_by_fly[b.fly.name]],expected.adhesion_onoff))))
                        b.restore(before)
                latency=[]
                for repeat in range(3):
                    b.restore(initial);q=[];v=[];f=[];began=time.perf_counter()
                    for i in range(4):
                        if i==1:b.perturb(.25,.005)
                        b.step(command,.005)
                        if b.fault:raise RuntimeError(b.fault)
                        q.append(b.d.qpos.copy());v.append(b.d.qvel.copy());f.append(b._contact_forces('feet')/b.weight0)
                    latency.append(time.perf_counter()-began)
                    trace=dict(qpos=np.asarray(q),qvel=np.asarray(v),feet_bw=np.asarray(f))
                    np.savez_compressed(a.out/f'{name}-{seed}-{repeat}.npz',**trace)
                    if repeat==0:traces[(seed,name)]=trace
                saved=b.snapshot();b.step(command);expected=b.d.qpos.copy();b.restore(saved);b.step(command)
                restore=dict(exact=bool(np.array_equal(expected,b.d.qpos)),qpos_abs=float(np.max(np.abs(expected-b.d.qpos))))
                atomic=None
                if name=='resident':
                    bad=copy.deepcopy(saved);bad['resident_controller']['ticks']+=1
                    before=b.snapshot()
                    try:b.restore(bad)
                    except ValueError:atomic=all(np.array_equal(v,b.resident.snapshot()[k]) for k,v in before['resident_controller'].items() if isinstance(v,np.ndarray)) and np.array_equal(before['warp']['arrays']['qpos'],b.snapshot()['warp']['arrays']['qpos'])
                    else:atomic=False
                output['cases'].append(dict(seed=seed,backend=name,wall_s=latency,median_wall_s=float(np.median(latency)),
                    controller_parity=parity,restore=restore,bad_clock_rejected_atomically=atomic))
            finally:b.close()
        reference=traces[(seed,'matching_cpu')]
        candidate=traces[(seed,'resident')]
        diff={k:float(np.max(np.abs(candidate[k]-reference[k]))) for k in reference}
        output['comparisons'].append(dict(seed=seed,matching_cpu_vs_resident=diff,
            within_existing_bounds=diff['qpos']<=.001 and diff['qvel']<=.1 and diff['feet_bw']<=.15))
    output['controller_parity_pass']=all(r['action_abs']<=3e-6 and r['state_abs']<=1e-10 and r['adhesion_equal'] for c in output['cases'] for r in c['controller_parity'])
    output['adoption_pass']=output['controller_parity_pass'] and all(c['within_existing_bounds'] for c in output['comparisons']) and all(c['restore']['exact'] for c in output['cases'])
    (a.out/'report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output))


if __name__=='__main__':main()
