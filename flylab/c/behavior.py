"""Physical displacement metrics, separate from task success or neural commands."""
import math


CRITERIA = dict(schema='flylab.behavior_criteria.v1', sample_dt_s=.005,
                odor_region_radius_mm=2., stuck_window_s=3., stuck_extent_mm=1.,
                resume_window_s=2., resume_net_mm=1., resume_forward_mm=1.,
                task_approval='NOT_EVALUATED: diagnostic metrics, no validated task thresholds')


def behavior_metrics(trace, world, *, motion_expected=True):
    forward = backward = path = rotation = contact = food = hazard = 0.
    gaps = 0; stuck = []; releases = []; food_entries = 0
    in_food = False
    for i, (a, b) in enumerate(zip(trace, trace[1:])):
        dt = b['simTime']-a['simTime']
        if dt <= 0 or dt > .005+1e-8: gaps += 1
        dx, dz = b['position'][0]-a['position'][0], b['position'][2]-a['position'][2]
        # Body's actual local forward axis before this physical interval.
        axis = a['forward_axis']
        signed = dx*axis[0]+dz*axis[2]
        forward += max(0., signed); backward += max(0., -signed)
        path += math.hypot(dx, dz)
        rotation += abs(math.atan2(math.sin(b['yaw']-a['yaw']), math.cos(b['yaw']-a['yaw'])))
        contact += dt*bool(b['contact'])
        nearby = {kind: any(o['kind']==kind and o['strength']>0 and
                  math.hypot(b['position'][0]-o['p'][0],b['position'][2]-o['p'][2])<=2
                  for o in world['sources']) for kind in ('food','hazard')}
        nearby['food'] &= world['foodOn']
        food += dt*nearby['food']; hazard += dt*nearby['hazard']
        if nearby['food'] and not in_food: food_entries += 1
        in_food = nearby['food']
        if a['avoidance'] and not b['avoidance']: releases.append(i+1)
        if b['simTime']-trace[0]['simTime'] >= 3:
            window = trace[max(0, i+1-600):i+2]
            if len(window)>=601 and motion_expected and all(r['motion_enabled'] for r in window):
                extent = math.hypot(max(r['position'][0] for r in window)-min(r['position'][0] for r in window),
                                    max(r['position'][2] for r in window)-min(r['position'][2] for r in window))
                if extent<1: stuck.append(b['tick'])
    resumed = []
    for index in releases:
        start = trace[index]; cumulative = 0.; verdict = 'INCOMPLETE'
        for a,b in zip(trace[index:],trace[index+1:]):
            if b['simTime']>start['simTime']+2+1e-8: verdict='FAILED'; break
            if b['avoidance'] or not b['motion_enabled']: verdict='INTERRUPTED'; break
            cumulative += ((b['position'][0]-a['position'][0])*a['forward_axis'][0]+
                           (b['position'][2]-a['position'][2])*a['forward_axis'][2])
            net = math.hypot(b['position'][0]-start['position'][0],b['position'][2]-start['position'][2])
            if net>=1 and cumulative>=1: verdict='RESUMED'; break
            if b['simTime']>=start['simTime']+2-1e-8: verdict='FAILED'
        resumed.append(dict(release_tick=start['tick'], status=verdict))
    return dict(signed_forward_mm=forward-backward, forward_mm=forward, backward_mm=backward,
                horizontal_path_mm=path, absolute_rotation_rad=rotation,
                contact_duration_s=contact, food_dwell_s=food, hazard_dwell_s=hazard, food_entries=food_entries,
                stuck_status='DETECTED' if stuck else 'NOT_DETECTED' if trace and trace[-1]['simTime']-trace[0]['simTime']>=3 else 'INCOMPLETE_WINDOW',
                stuck_window_count=len(stuck), first_stuck_tick=stuck[0] if stuck else None,
                avoidance_releases=resumed, observation_gaps=gaps,
                faults=[r['fault'] for r in trace if r['fault']], task_status='NOT_EVALUATED')


def trace_sample(frame):
    b = frame['body']; basis = b['basis']; c = frame['command']
    return dict(simTime=frame['simTime'], tick=frame['tick'], position=b['position'], yaw=b['yaw'],
                forward_axis=[row[0] for row in basis], contact=bool(frame['sensors']['contact']),
                avoidance=bool(c.get('assist_reason') or (frame.get('legacy') or {}).get('avoidanceActive') or
                               (frame.get('legacy') or {}).get('recoveryActive')),
                motion_enabled=bool(c.get('motor_coupled')) and not frame.get('stopped'),
                command=c, sensors=frame['sensors'], sensory_ports=frame['sensory_ports'],
                motor_rates_Hz=frame['motor_rates_Hz'], fault=frame['fault'])
