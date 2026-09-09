"""Frozen engineering task criteria, independent of model/controller tuning."""
import math
from .behavior import behavior_metrics
from .integrity import digest

CRITERIA = dict(schema='flylab.task-criteria.v1', walking_s=10., walking_net_mm=5.,
                walking_signed_forward_mm=5., backward_s=2., backward_signed_mm=-1.,
                food_deadline_s=30., food_radius_mm=2., stop_settle_s=.5, stop_measure_s=1.,
                stop_path_mm=.5, resume_s=2., resume_net_mm=1., resume_forward_mm=1.)

def evaluate(trace, world, task='walking', *, required_seconds=None, motion_expected=True):
    metrics=behavior_metrics(trace,world,motion_expected=motion_expected)
    elapsed=trace[-1]['simTime']-trace[0]['simTime'] if trace else 0.
    net=math.hypot(trace[-1]['position'][0]-trace[0]['position'][0],trace[-1]['position'][2]-trace[0]['position'][2]) if trace else 0.
    needed=required_seconds if required_seconds is not None else CRITERIA.get(task+'_s',30.)
    technical='FAIL' if metrics['faults'] else 'INCOMPLETE' if elapsed+1e-8<needed or metrics['observation_gaps'] else 'PASS'
    status='INCOMPLETE';reasons=[]
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
        elif task=='hazard':
            status='NOT_EVALUATED';reasons.append('Requires matched hazard-free and sensory-off controls; stationary avoidance is not a pass')
        elif task=='diagnostic': status='NOT_EVALUATED'
        else: raise ValueError('Unknown task contract')
    if technical=='FAIL':status='FAIL'
    return dict(technical_status=technical,task_status=status,task=task,reasons=reasons,
                criteria=CRITERIA,criteria_hash=digest(CRITERIA),elapsed_s=elapsed,horizontal_net_mm=net,
                metrics=metrics,biological_validation=False)
