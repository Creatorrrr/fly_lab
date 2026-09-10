#!/usr/bin/env python3
"""Save the original reversal counterexample and a matched guarded run."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import write_json
from flylab.c.muscles import MuscleRig
from flylab.c.muscle_calibration import direction_status
from flylab.c.storage import StateStore


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'spec.json',dict(excitation=[0.,.35],maximum_seconds=.1,physics_dt=.00001,
        sample_dt=.001,cases=['unguarded','guarded'],expected='Detect the original moment-arm reversal and stop the guarded run on the first offending physical step',
        interpretation='Adversarial engineering test; an expected stop does not make the original muscle routing valid'))
    traces={};results=[]
    for guarded in (False,True):
        name='guarded' if guarded else 'unguarded';directory=out/name;directory.mkdir()
        rig=MuscleRig();rows=[rig.frame()];fault=None
        for _ in range(100):
            try: rig.step((0.,.35),steps=100,direction_guard=guarded)
            except RuntimeError as exc: fault=str(exc)
            rows.append(rig.frame())
            if fault: break
        StateStore.save(directory/'final',rig.snapshot());write_json(directory/'trace.json',rows)
        reversals=[r for r in rows if direction_status(r)['status']!='CONSISTENT']
        result=dict(name=name,physicalExecuted=True,seconds=rig.tick*rig.model.opt.timestep,
            fault=fault,offending_samples=len(reversals),last_direction=direction_status(rows[-1]),
            last_angle_rad=rows[-1]['q_rad'],model_hash=rig.identity)
        write_json(directory/'result.json',result);traces[name]=rows;results.append(result)
    reference={r['tick']:r for r in traces['unguarded']}
    common=[(reference[r['tick']],r) for r in traces['guarded'] if r['tick'] in reference]
    error=max(abs(a['q_rad']-b['q_rad']) for a,b in common)
    checks=dict(reproduces_original_reversal=results[0]['offending_samples']>0,
        stops_on_first_detected_reversal=results[1]['fault']=='FAULT_MUSCLE_DIRECTION' and results[1]['offending_samples']==1,
        stops_before_end=results[1]['seconds']<.1,guard_does_not_change_prior_trajectory=error==0.)
    report=dict(status='PASS_GUARD_DETECTION' if all(checks.values()) else 'FAIL',checks=checks,cases=results,
        compared_boundary_count=len(common),max_prefix_angle_error_rad=error,physicalExecuted=True,
        biological_validation=False,physiological_status='ORIGINAL_ROUTING_STILL_REQUIRES_CALIBRATION')
    write_json(out/'report.json',report);print(report['status'],results[1]);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    a=p.parse_args();r=run(a.out);raise SystemExit(0 if r['status']=='PASS_GUARD_DETECTION' else 1)
