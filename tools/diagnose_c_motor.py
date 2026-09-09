#!/usr/bin/env python3
"""Physical response to isolated engineered body commands, independent of brain."""
from pathlib import Path
import argparse
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.body import FlyGymBody
from flylab.engine import config_values
from flylab.sensors import default_world, SensorAdapter
from flylab.c.behavior import behavior_metrics
from flylab.c.integrity import write_json

def run(out,drive=1.):
    if not .1<=drive<=3.3:raise ValueError('Diagnostic drive must be .1..3.3')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);reports=[]
    cases={'forward':[(3.,drive,0.)],'backward':[(2.,-drive,0.)],
           'yaw_left':[(2.,0.,-3.)],'yaw_right':[(2.,0.,3.)],
           'stop_release':[(.5,drive,0.),(1.5,0.,0.),(2.,drive,0.)]}
    for name,segments in cases.items():
        world=default_world();world['sources']=[];world['obstacles']=[]
        body=FlyGymBody(42,world,config_values());sensors=SensorAdapter(42);trace=[];phases=[]
        try:
            def sample(speed):
                b,physics=body.frame();p=sensors.observe(body,world,.005,config_values())
                return dict(simTime=physics['physicsTime'],tick=round(physics['physicsTime']/.0001),position=b['position'],
                            yaw=b['yaw'],forward_axis=[r[0] for r in b['basis']],contact=bool(body.nonfoot_contact()),
                            avoidance=False,motion_enabled=abs(speed)>0,fault=body.fault)
            trace.append(sample(segments[0][1]))
            for seconds,speed,yaw in segments:
                start=len(trace)-1
                for _ in range(round(seconds/.005)):
                    body.step(dict(forwardSpeed=speed,yawRate=yaw,verticalSpeed=0.),.005);trace.append(sample(speed))
                    if body.fault:break
                portion=trace[start:];metrics=behavior_metrics(portion,world,motion_expected=bool(speed or yaw))
                net=math.hypot(portion[-1]['position'][0]-portion[0]['position'][0],portion[-1]['position'][2]-portion[0]['position'][2])
                phases.append(dict(seconds=seconds,command=dict(forwardSpeed=speed,yawRate=yaw),net_mm=net,
                                   yaw_change=sum(math.atan2(math.sin(b['yaw']-a['yaw']),math.cos(b['yaw']-a['yaw'])) for a,b in zip(portion,portion[1:])),metrics=metrics))
                if body.fault:break
            result=dict(name=name,physicalExecuted=True,phases=phases,fault=body.fault,trace=trace,
                        interpretation='Body adapter response only; no neural or natural-behavior approval')
            write_json(out/(name+'.json'),result);reports.append({k:v for k,v in result.items() if k!='trace'})
            print(name,[(p['metrics']['signed_forward_mm'],p['yaw_change']) for p in phases],flush=True)
        finally:body.close()
    write_json(out/'report.json',dict(status='COMPLETE',cases=reports,physicalExecuted=True,drive=drive,
               meaning='Drive controls CPG amplitude; it is not a measured speed servo. Read each fault field.'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--drive',type=float,default=1.);a=p.parse_args();run(a.out,a.drive)
