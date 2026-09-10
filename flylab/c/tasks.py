"""Frozen engineering task criteria, independent of model/controller tuning."""
import math
from .behavior import behavior_metrics
from .integrity import digest, finite

CRITERIA = dict(schema='flylab.task-criteria.v2', walking_s=10., walking_net_mm=5.,
                walking_signed_forward_mm=5., backward_s=2., backward_signed_mm=-1.,
                food_deadline_s=30., food_radius_mm=2., stop_settle_s=.5, stop_measure_s=1.,
                stop_path_mm=.5, resume_s=2., resume_net_mm=1., resume_forward_mm=1.,
                obstacle_requires_matched_controls=True)

CONTROL_CRITERIA = dict(schema='flylab.control-criteria.v1', yaw_s=2., yaw_min_rad=.25,
    prior_net_mm=1., stop_settle_s=.5, stop_measure_s=1., stop_path_mm=.5,
    resume_s=2., resume_net_mm=1., resume_forward_mm=1.)


def control_parameters(task, parameters=None):
    p = parameters or {}
    if task == 'stop_resume':
        if set(p) != {'stop_onset_s','stop_release_s'}: raise ValueError('Stop onset and release required')
        onset = finite(p['stop_onset_s'], 'stop onset', .5, 3596.)
        release = finite(p['stop_release_s'], 'stop release', onset+1.5, 3598.)
        if any(abs(x/.005-round(x/.005))>1e-8 for x in (onset,release)): raise ValueError('Stop times must be control boundaries')
    elif p:
        raise ValueError('This task has no task parameters')
    return p


def evaluate_control(trace, world, task, parameters=None):
    p = control_parameters(task, parameters)
    elapsed = trace[-1]['simTime']-trace[0]['simTime'] if trace else 0.
    needed = p['stop_release_s']+2. if task=='stop_resume' else 2.
    metrics = behavior_metrics(trace,world,motion_expected=task!='stop_resume')
    technical = 'FAIL' if metrics['faults'] else 'INCOMPLETE' if elapsed+1e-8<needed or metrics['observation_gaps'] else 'PASS'
    result = dict(technical_status=technical,task_status=technical,task=task,elapsed_s=elapsed,
                  criteria=CONTROL_CRITERIA,criteria_hash=digest(CONTROL_CRITERIA),metrics=metrics,
                  task_parameters=p,biological_validation=False)
    if technical!='PASS': return result
    def window(start,end):
        t0=trace[0]['simTime']
        return [r for r in trace if start-1e-8<=r['simTime']-t0<=end+1e-8]
    def net(rows):
        return math.hypot(rows[-1]['position'][0]-rows[0]['position'][0],rows[-1]['position'][2]-rows[0]['position'][2])
    if task.startswith('yaw_'):
        rows=window(0,2.)
        angle=sum(math.atan2(math.sin(b['yaw']-a['yaw']),math.cos(b['yaw']-a['yaw'])) for a,b in zip(rows,rows[1:]))
        measured=dict(signed_yaw_rad=angle,horizontal_net_mm=net(rows))
        passed=angle*(-1 if task=='yaw_left' else 1)>=CONTROL_CRITERIA['yaw_min_rad']
    else:
        before=window(0,p['stop_onset_s'])
        stopped=window(p['stop_onset_s']+.5,p['stop_onset_s']+1.5)
        after=window(p['stop_release_s'],p['stop_release_s']+2.)
        measured=dict(prior_net_mm=net(before),stop_path_mm=behavior_metrics(stopped,world,motion_expected=False)['horizontal_path_mm'],
                      resume_net_mm=net(after),resume_forward_mm=behavior_metrics(after,world)['signed_forward_mm'])
        passed=measured['prior_net_mm']>=1 and measured['stop_path_mm']<=.5 and measured['resume_net_mm']>=1 and measured['resume_forward_mm']>=1
    result.update(task_status='PASS' if passed else 'FAIL',measurements=measured)
    return result


def evaluate(trace, world, task='walking', *, required_seconds=None, motion_expected=True, task_parameters=None):
    if task in ('yaw_left','yaw_right','stop_resume'):
        return evaluate_control(trace,world,task,task_parameters)
    control_parameters(task,task_parameters)
    metrics=behavior_metrics(trace,world,motion_expected=motion_expected)
    elapsed=trace[-1]['simTime']-trace[0]['simTime'] if trace else 0.
    net=math.hypot(trace[-1]['position'][0]-trace[0]['position'][0],trace[-1]['position'][2]-trace[0]['position'][2]) if trace else 0.
    needed=required_seconds if required_seconds is not None else CRITERIA.get(task+'_s',30.)
    technical='FAIL' if metrics['faults'] else 'INCOMPLETE' if elapsed+1e-8<needed or metrics['observation_gaps'] else 'PASS'
    status='INCOMPLETE';reasons=[];trajectory_status=None
    if technical=='PASS':
        if task=='walking':
            status='PASS' if elapsed>=10-1e-8 and net>=5 and metrics['signed_forward_mm']>=5 and metrics['stuck_status']=='NOT_DETECTED' else 'FAIL'
            if elapsed<10-1e-8: status='INCOMPLETE'
        elif task=='backward':
            status='PASS' if elapsed>=2-1e-8 and metrics['signed_forward_mm']<=-1 else 'FAIL'
            if elapsed<2-1e-8: status='INCOMPLETE'
        elif task=='food':
            sources=[o for o in world['sources'] if o['kind']=='food' and o['strength']>0 and world['foodOn']]
            if not sources or any(math.hypot(trace[0]['position'][0]-o['p'][0],trace[0]['position'][2]-o['p'][2])<=2 for o in sources):
                status='NOT_APPLICABLE';reasons.append('No valid approach opportunity')
            elif any(r['simTime']-trace[0]['simTime']<=CRITERIA['food_deadline_s']+1e-8 and
                     any(math.hypot(r['position'][0]-o['p'][0],r['position'][2]-o['p'][2])<=CRITERIA['food_radius_mm'] for o in sources) for r in trace): status='PASS'
            else: status='FAIL' if elapsed>=30-1e-8 else 'INCOMPLETE'
        elif task=='obstacle':
            releases=metrics['avoidance_releases']
            if not any(r.get('obstacle_engaged') for r in trace):
                status='NOT_APPLICABLE';reasons.append('No measured obstacle approach')
            elif metrics['stuck_status']=='DETECTED' or metrics['contact_duration_s']>0: status='FAIL'
            elif any(r['status']=='PASS' for r in releases): status='PASS'
            else: status='INCOMPLETE' if any(r['status']=='CENSORED_END_OF_RECORD' for r in releases) else 'FAIL'
            trajectory_status=status
            if status=='PASS':
                status='NOT_EVALUATED'
                reasons.append('Geometric progress observed; matched obstacle-free and sensory-off controls must establish avoidance opportunity and effect')
        elif task=='hazard':
            status='NOT_EVALUATED';reasons.append('Requires matched hazard-free and sensory-off controls; stationary avoidance is not a pass')
        elif task=='diagnostic': status='NOT_EVALUATED'
        else: raise ValueError('Unknown task contract')
    if technical=='FAIL':status='FAIL'
    result=dict(technical_status=technical,task_status=status,task=task,reasons=reasons,
                criteria=CRITERIA,criteria_hash=digest(CRITERIA),elapsed_s=elapsed,horizontal_net_mm=net,
                metrics=metrics,biological_validation=False)
    if task=='obstacle':result['trajectory_status']=trajectory_status
    return result
