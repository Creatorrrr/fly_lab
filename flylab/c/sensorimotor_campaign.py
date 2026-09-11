"""Fixed sensorimotor cases and hazard controls; no behavior-based retuning."""
import copy
import json
import math
from pathlib import Path
from .campaign import scene_world, verify_result_files
from .behavior import behavior_metrics
from .integrity import digest, read_json, file_hash, checked_name

# Ten predeclared placements/frictions. Seeds alone do not vary a deterministic
# LIF's initial pose, so pose and substrate are explicit experimental factors.
INITIAL_CONDITIONS = tuple(dict(seed=seed, initial_pose=dict(position=[x,.8,z],yaw_rad=yaw),
                                config=dict(friction=friction))
    for seed,x,z,yaw,friction in ((7,0,0,0,1.),(19,.5,0,.15,1.),(42,-.5,0,-.15,1.),
        (61,0,.5,.3,1.),(83,0,-.5,-.3,1.),(101,.5,.5,.45,.7),(127,-.5,-.5,-.45,.7),
        (149,.5,-.5,.6,1.3),(173,-.5,.5,-.6,1.3),(197,0,0,math.pi/2,1.)))
HAZARD_CRITERIA = dict(schema='flylab.matched-hazard.v1', minimum_seconds=10.,
    opportunity_distance_mm=3., minimum_active_net_mm=5., minimum_path_mm=5.,
    minimum_clearance_improvement_mm=.5, maximum_hazard_dwell_fraction=.5)


def fixed_spec(seconds=30., conditions=INITIAL_CONDITIONS):
    cases=[]
    for i,condition in enumerate(conditions):
        for scene in ('baseline','food_left','food_right','front_obstacle','hazard_left','hazard_right'):
            world=scene_world(scene)
            base=dict(name=f'{i:02}-{scene}',mode='C_STRICT',scene=scene,seconds=seconds,
                      world=world,**copy.deepcopy(condition))
            cases.append(base)
            if scene.startswith('hazard'):
                off=copy.deepcopy(base);off['name']+='-sensory-off'
                off['intervention']=dict(kind='sensor_off',channels=['danger'],duration_controls=round(seconds/.005))
                cases.append(off)
                free=copy.deepcopy(base);free['name']+='-hazard-free';free['world']['sources']=[]
                cases.append(free)
            elif scene.startswith('food'):
                off=copy.deepcopy(base);off['name']+='-sensory-off'
                off['intervention']=dict(kind='sensor_off',channels=['odor_left','odor_right'],duration_controls=round(seconds/.005))
                cases.append(off)
    return dict(schema='flylab.campaign.v1',cases=cases)


def evaluate_hazard(active, hazard_free, sensory_off, *, assay='natural'):
    """Inputs contain trace, world, provenance, and declared intervention list."""
    runs=(active,hazard_free,sensory_off)
    if assay not in ('natural','evoked'):raise ValueError('Unknown hazard assay')
    result=dict(criteria=HAZARD_CRITERIA,criteria_hash=digest(HAZARD_CRITERIA),assay=assay,natural_behavior=assay=='natural',
                task_status='INCOMPLETE',biological_validation=False,reasons=[],
                physicalExecuted=all(r['provenance'].get('physical') is True for r in runs))
    keys=('graph_hash','binding_hash','model_parameter_hash','mode','seed','body_model_hash','config_hash','initial_pose','neural_backend','versions','app_version','motor_execution')
    if any(any(k not in r['provenance'] for k in keys) for r in runs):
        result['reasons']=['Missing control identity'];return result
    if any(any(r['provenance'].get(k)!=active['provenance'].get(k) for k in keys) for r in runs):
        result['reasons']=['Control identity mismatch'];return result
    if active['provenance'].get('mode')!='C_STRICT':
        result['reasons']=['Natural hazard assay requires C_STRICT'];return result
    shared=active.get('interventions',[])
    events=sensory_off.get('interventions',[])
    if assay=='natural':
        if shared or hazard_free.get('interventions'):
            result['reasons']=['Active and hazard-free arms must have no direct intervention'];return result
    else:
        if not shared or shared!=hazard_free.get('interventions') or events[:len(shared)]!=shared:
            result['reasons']=['Evoked assay requires identical background stimulation in all three arms'];return result
        if any(e.get('kind')!='stimulate' or not e.get('ids') or e.get('amplitude_mV',0)<=0 or e.get('at_tick',0)!=0 for e in shared):
            result['reasons']=['Evoked background must be a declared positive neuronal stimulation from time zero'];return result
        events=events[len(shared):]
        result['shared_stimulation']=shared
    if len(events)!=1 or events[0].get('kind')!='sensor_off' or events[0].get('channels')!=['danger'] or events[0].get('at_tick',0)!=0:
        result['reasons']=['Require a hazard-only sensory-off arm'];return result
    hazards=[s for s in active['world']['sources'] if s['kind']=='hazard' and s['strength']>0]
    expected=copy.deepcopy(active['world']);expected['sources']=[s for s in expected['sources'] if s['kind']!='hazard']
    if len(hazards)!=1 or sensory_off['world']!=active['world'] or hazard_free['world']!=expected:
        result['reasons']=['Require one hazard and otherwise identical matched worlds'];return result
    traces=[r['trace'] for r in runs]
    if any(len(t)<2 for t in traces):return result
    for t in traces[1:]:
        if any(abs(a-b)>1e-6 for a,b in zip(t[0]['position'],traces[0][0]['position'])) or abs(t[0]['yaw']-traces[0][0]['yaw'])>1e-6:
            result['reasons']=['Initial physical poses differ'];return result
    durations=[t[-1]['simTime']-t[0]['simTime'] for t in traces]
    if min(durations)<10.-1e-8 or max(durations)-min(durations)>1e-8:
        result['reasons']=['Incomplete or unequal observation duration'];return result
    if events[0].get('duration_controls',0)*.005+1e-8<durations[2]:
        result['reasons']=['Sensory-off interval does not cover the assay'];return result
    if assay=='evoked' and any(e.get('duration_controls',0)*.005+1e-8<durations[0] for e in shared):
        result['reasons']=['Background stimulation must cover the evoked assay'];return result
    if any(not row['motion_enabled'] for t in traces for row in t[1:]):
        result['reasons']=['Motor connection must remain enabled'];return result
    # Verify suppression in raw input packets, not just a case label.
    danger_names={p['name'] for p in active['provenance'].get('bindings',{}).get('sensory',[]) if p['channel']=='danger'}
    danger_rows=[[p for p in t['sensory_ports'] if p['name'] in danger_names] for t in traces[2][1:]]
    if any(not ports or any(p['enabled'] or p['value']!=0 for p in ports) for ports in danger_rows):
        result['reasons']=['Hazard sensory-off control is unverified'];return result
    metrics=[behavior_metrics(t,r['world']) for t,r in zip(traces,runs)]
    if any(m['faults'] or m['observation_gaps'] for m in metrics):
        result['task_status']='FAIL';result['reasons']=['Fault or observation gap'];return result
    target=hazards[0]['p']
    closest=[min(math.hypot(x['position'][0]-target[0],x['position'][2]-target[2]) for x in t) for t in traces]
    net=math.hypot(traces[0][-1]['position'][0]-traces[0][0]['position'][0],traces[0][-1]['position'][2]-traces[0][0]['position'][2])
    result['measurements']=dict(closest_mm=closest,hazard_dwell_s=[m['hazard_dwell_s'] for m in metrics],active_net_mm=net,active_path_mm=metrics[0]['horizontal_path_mm'])
    if closest[1]>3. or closest[2]>3.:
        result.update(task_status='NOT_APPLICABLE',reasons=['Both controls must approach the hazard location']);return result
    passed=(net>=5. and metrics[0]['horizontal_path_mm']>=5. and metrics[0]['stuck_status']=='NOT_DETECTED'
            and closest[0]>=max(closest[1],closest[2])+.5
            and metrics[0]['hazard_dwell_s']<=.5*metrics[2]['hazard_dwell_s'])
    result['task_status']='PASS' if passed else 'FAIL'
    return result


def load_case(root, name):
    """Load only committed, hash-checked campaign evidence; never orphan chunks."""
    root=Path(root).resolve();name=checked_name(name)
    manifest=read_json(root/'campaign.json');verify_result_files(manifest,root)
    spec=read_json(root/'spec.json')
    if digest(spec)!=manifest['identity']['spec_hash']:raise ValueError('Campaign spec hash mismatch')
    summary=next((r for r in manifest['cases'] if r['name']==name),None)
    if not summary or summary['execution_status']!='COMPLETE':raise ValueError('Case is incomplete: '+name)
    result=read_json(root/summary['result_file'])
    case=next(c for c in spec['cases'] if c['name']==name)
    if result['case']!=case:raise ValueError('Case specification mismatch')
    for key in ('graph_hash','binding_hash'):
        if result['provenance'][key]!=manifest['identity'][key]:raise ValueError('Case identity mismatch')
    directory=root/name;progress=read_json(directory/'progress.json');trace=[]
    for chunk in progress['chunks']:
        path=(directory/chunk['file']).resolve();path.relative_to(directory)
        if file_hash(path)!=chunk['sha256']:raise ValueError('Trace chunk hash mismatch')
        trace.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    if not trace or trace[0]['tick']!=0 or trace[-1]['tick']!=progress['control_tick']*50:
        raise ValueError('Committed trace range mismatch')
    if abs(trace[-1]['simTime']-case['seconds'])>1e-8:raise ValueError('Case trace duration mismatch')
    world=case.get('world',scene_world(case['scene']))
    if digest(world)!=result['provenance']['environment_hash']:raise ValueError('World hash mismatch')
    return dict(trace=trace,world=world,provenance=result['provenance'],
                interventions=case.get('interventions',[case['intervention']] if case.get('intervention') else []))


def assess_hazards(root, *, assay='natural'):
    root=Path(root);spec=read_json(root/'spec.json');names={c['name'] for c in spec['cases']};rows=[]
    for case in spec['cases']:
        name=case['name']
        if not case['scene'].startswith('hazard') or name.endswith(('-sensory-off','-hazard-free')):continue
        trio=(name,name+'-hazard-free',name+'-sensory-off')
        if not all(n in names for n in trio):
            rows.append(dict(name=name,task_status='INCOMPLETE',reasons=['Missing matched controls']));continue
        try:result=evaluate_hazard(*(load_case(root,n) for n in trio),assay=assay)
        except (ValueError,KeyError,OSError) as exc:result=dict(task_status='INCOMPLETE',reasons=[str(exc)])
        rows.append(dict(name=name,**result))
    return dict(schema='flylab.hazard-assessment.v1',criteria=HAZARD_CRITERIA,cases=rows,assay=assay,natural_behavior=assay=='natural',
                task_status='PASS' if rows and all(r['task_status']=='PASS' for r in rows) else 'NOT_VALIDATED',
                physicalExecuted=bool(rows) and all(r.get('physicalExecuted') is True for r in rows),biological_validation=False)
