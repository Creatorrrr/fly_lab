#!/usr/bin/env python3
"""Matched obstacle opportunity and effect, using immutable campaign records."""
import argparse
import copy
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.sensorimotor_campaign import load_case
from flylab.c.behavior import behavior_metrics
from flylab.c.tasks import evaluate
from flylab.c.integrity import digest,write_json

CRITERIA=dict(schema='flylab.matched-obstacle.v1',minimum_seconds=10.,
    control_surface_clearance_mm=.75,minimum_clearance_improvement_mm=.5,
    clearance_model='horizontal thorax-to-sphere-surface distance; engineering opportunity proxy')
CHANNELS=['loom_left','loom_right','head_contact']


def assess(active,free,off,*,assay='natural'):
    if assay not in ('natural','evoked'):raise ValueError('Unknown assay')
    runs=(active,free,off)
    result=dict(criteria=CRITERIA,criteria_hash=digest(CRITERIA),task_status='INCOMPLETE',
        physicalExecuted=all(r['provenance'].get('physical') is True for r in runs),
        assay=assay,natural_behavior=assay=='natural',biological_validation=False,reasons=[])
    def fail(reason,status='INCOMPLETE'):
        result.update(task_status=status,reasons=[reason]);return result
    keys=('graph_hash','binding_hash','model_parameter_hash','mode','seed','body_model_hash','config_hash',
          'initial_pose','neural_backend','versions','app_version','motor_execution')
    if any(any(k not in r['provenance'] or r['provenance'][k]!=active['provenance'].get(k) for k in keys) for r in runs):
        return fail('Control identity mismatch')
    if active['provenance']['mode']!='C_STRICT':return fail('C_STRICT required')
    shared=active.get('interventions',[]);events=off.get('interventions',[])
    if assay=='natural' and (shared or free.get('interventions')):return fail('Natural assay excludes direct stimulation')
    if assay=='evoked':
        if not shared or shared!=free.get('interventions') or events[:len(shared)]!=shared:return fail('Matched stimulation required')
        if any(e.get('kind')!='stimulate' or not e.get('ids') or e.get('at_tick',0)!=0 or e.get('amplitude_mV',0)<=0 for e in shared):return fail('Positive background stimulation from time zero required')
        events=events[len(shared):]
    if len(events)!=1 or events[0].get('kind')!='sensor_off' or events[0].get('channels')!=CHANNELS or events[0].get('at_tick',0)!=0:
        return fail('Exact visual/head-contact off arm required')
    expected=copy.deepcopy(active['world']);expected['obstacles']=[]
    if len(active['world']['obstacles'])!=1 or free['world']!=expected or off['world']!=active['world']:
        return fail('One obstacle and otherwise identical worlds required')
    traces=[r['trace'] for r in runs]
    if any(len(t)<2 for t in traces):return fail('Missing observations')
    if any(t[0]['position']!=traces[0][0]['position'] or t[0]['yaw']!=traces[0][0]['yaw'] for t in traces):return fail('Initial poses differ')
    durations=[t[-1]['simTime']-t[0]['simTime'] for t in traces]
    if min(durations)<10.-1e-8 or max(durations)-min(durations)>1e-8:return fail('Incomplete or unequal duration')
    if any(e.get('duration_controls',0)*.005+1e-8<durations[0] for e in events+shared):return fail('Intervention must cover the assay')
    if any(not r['motion_enabled'] for t in traces for r in t[1:]):return fail('Motor connection must stay enabled')
    names={p['name'] for p in active['provenance']['bindings']['sensory'] if p['channel'] in CHANNELS}
    if len(names)!=3:return fail('Three reviewed sensory ports required')
    for row in traces[2][1:]:
        ports=[p for p in row['sensory_ports'] if p['name'] in names]
        if len(ports)!=3 or any(p['enabled'] or p['value']!=0 for p in ports):return fail('Unverified sensory-off packets')
    metrics=[behavior_metrics(t,r['world']) for t,r in zip(traces,runs)]
    if any(m['faults'] or m['observation_gaps'] for m in metrics):return fail('Fault or observation gap','FAIL')
    obj=active['world']['obstacles'][0]
    closest=[min(math.hypot(row['position'][0]-obj['p'][0],row['position'][2]-obj['p'][2])-obj['r'] for row in t) for t in traces]
    geometry=evaluate(traces[0],active['world'],'obstacle',required_seconds=durations[0])
    result['measurements']=dict(closest_surface_mm=closest,active_contact_s=metrics[0]['contact_duration_s'],
                                active_trajectory_status=geometry['trajectory_status'])
    if max(closest[1:])>CRITERIA['control_surface_clearance_mm']:
        return fail('Both controls must encounter the obstacle location; an unchanged curved path is not sensory avoidance','NOT_APPLICABLE')
    passed=(geometry['trajectory_status']=='PASS' and metrics[0]['contact_duration_s']==0 and
            closest[0]>=max(closest[1:])+CRITERIA['minimum_clearance_improvement_mm'])
    result['task_status']='PASS' if passed else 'FAIL';return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for role in ('active','free','off'):p.add_argument('--'+role,nargs=2,metavar=('CAMPAIGN','CASE'),required=True)
    p.add_argument('--assay',choices=('natural','evoked'),default='natural');p.add_argument('--out',required=True)
    a=p.parse_args()
    if Path(a.out).exists():raise SystemExit('Choose a new output path')
    result=assess(*(load_case(*getattr(a,role)) for role in ('active','free','off')),assay=a.assay)
    write_json(a.out,result);print(result['task_status'])
