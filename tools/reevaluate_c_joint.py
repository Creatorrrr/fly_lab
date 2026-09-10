#!/usr/bin/env python3
"""Reassess saved joint traces/checkpoint hashes without rerunning physics."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.integrity import digest,file_hash,read_json,write_json
from flylab.c.muscle_calibration import direction_status
from flylab.c.storage import StateStore
from tools.verify_c_single_joint import evaluate


def run(source,out):
    source,out=Path(source),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    report=read_json(source/'report.json');spec=read_json(source/'spec.json')
    results=copy.deepcopy(report['cases']);traces={};files={};checks={}
    if len(results)!=len(spec['cases']): raise ValueError('Incomplete case roster')
    for result in results:
        name=result['name'];directory=source/name
        rows=read_json(directory/'trace.json');state=StateStore.load(directory/'final')
        profile=read_json(directory/'profile.json')
        for file in ('trace.json','profile.json','result.json','final/manifest.json'):
            files[name+'/'+file]=file_hash(directory/file)
        result['physical_state_hash']=hashlib.sha256(state['body']['state'].tobytes()).hexdigest()
        complete=(len(rows)==round(spec['seconds']/spec['control_dt'])+1 and
                  all(abs(row['seconds']-i*spec['control_dt'])<1e-10 for i,row in enumerate(rows)))
        checks[name+'_sampling_complete']=complete
        checks[name+'_profile_hash']=state['profile_hash']==result['profile_hash']==digest(profile)
        checks[name+'_checkpoint_clocks']=(state['neural']['tick']==rows[-1]['neural_tick'] and
                                         state['body']['tick']==rows[-1]['physics']['tick'])
        for row in rows:
            f=row['physics']
            projected=float(np.dot(f['moment_arm_mm'],f['muscle_force']))
            rhs=f['actuator_torque']+f['passive_joint_torque']+f['external_torque']+f['constraint_torque']-f['bias_torque']
            f['torque_projection_error']=abs(projected-f['actuator_torque'])
            f['dynamics_balance_error']=abs(f['effective_inertia']*f['qacc_rad_s2']-rhs)
            row['direction']=direction_status(f)
        traces[name]=rows
    assessment=evaluate(traces,results,spec)
    output=dict(schema='flylab.joint-recorded-audit.v1',physicalExecuted=False,
        status='PASS_RECORDED_EVIDENCE_AUDIT' if all(checks.values()) and assessment['technical_status']=='PASS' else 'FAIL',
        record_checks=checks,assessment=assessment,files=files,
        source_spec_sha256=file_hash(source/'spec.json'),source_report_sha256=file_hash(source/'report.json'),
        continuation_scope='Original run reports exact continuation; this audit reads and hashes checkpoints, without new continuation steps')
    write_json(out/'report.json',output);print(output['status'],assessment['sensorimotor_status'])
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=run(a.source,a.out);raise SystemExit(0 if r['status']=='PASS_RECORDED_EVIDENCE_AUDIT' else 1)
