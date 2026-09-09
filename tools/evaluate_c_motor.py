#!/usr/bin/env python3
"""Score actual adapter traces, including stop-settle and release windows."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import read_json,write_json,digest
from flylab.c.behavior import behavior_metrics
from flylab.sensors import default_world

CRITERIA=dict(schema='flylab.motor-adapter-criteria.v1',backward_s=2.,backward_mm=-1.,
              yaw_test_s=2.,yaw_min_rad=.25,stop_settle_s=.5,stop_observe_s=1.,stop_path_max_mm=.5,
              prior_net_min_mm=1.,release_s=2.,release_net_min_mm=1.,release_signed_min_mm=1.)

def evaluate(directory):
    directory=Path(directory);rows={name:read_json(directory/(name+'.json')) for name in ('forward','backward','yaw_left','yaw_right','stop_release')}
    results={}
    for name,row in rows.items():
        trace=row['trace'];gaps=any(abs(b['simTime']-a['simTime']-.005)>1e-8 for a,b in zip(trace,trace[1:]))
        valid=not row['fault'] and not gaps
        if name=='forward':passed=row['phases'][0]['metrics']['signed_forward_mm']>=1
        elif name=='backward':passed=row['phases'][0]['metrics']['signed_forward_mm']<=-1
        elif name.startswith('yaw'):passed=row['phases'][0]['yaw_change']*(-1 if name=='yaw_left' else 1)>=.25
        else:
            onset=row['phases'][0]['seconds'];release=onset+row['phases'][1]['seconds'];t0=trace[0]['simTime']
            def window(a,b):return [r for r in trace if a-1e-8<=r['simTime']-t0<=b+1e-8]
            before=window(0,onset);stopped=window(onset+.5,onset+1.5);after=window(release,release+2.)
            def net(r):return math.hypot(r[-1]['position'][0]-r[0]['position'][0],r[-1]['position'][2]-r[0]['position'][2])
            measured=dict(prior_net_mm=net(before),stop_path_mm=behavior_metrics(stopped,default_world(),motion_expected=False)['horizontal_path_mm'],
                          release_net_mm=net(after),release_signed_mm=behavior_metrics(after,default_world())['signed_forward_mm'])
            passed=measured['prior_net_mm']>=1 and measured['stop_path_mm']<=.5 and measured['release_net_mm']>=1 and measured['release_signed_mm']>=1
            results['stop_measurements']=measured
        results[name]='PASS' if valid and passed else 'FAIL'
    return dict(status='PASS' if all(results[k]=='PASS' for k in rows) else 'FAIL',results=results,criteria=CRITERIA,
                criteria_hash=digest(CRITERIA),physicalExecuted=True,neural_control=False,biologicalValidation=False,
                interpretation='Engineered body adapter only; yaw threshold fixed after body characterization, before independent neural task evaluation.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose new evidence path')
    r=evaluate(a.input);write_json(a.out,r);print(r)
    raise SystemExit(0 if r['status']=='PASS' else 1)
