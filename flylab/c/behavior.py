"""Physical displacement metrics, separate from task success or neural commands."""
import math
from ..navigation_checks import NavigationMonitor, NavigationPolicy


CRITERIA = dict(schema='flylab.behavior_criteria.v1', sample_dt_s=.005,
                odor_region_radius_mm=2., stuck_window_s=3., stuck_extent_mm=1.,
                resume_window_s=2., resume_net_mm=1., resume_forward_mm=1.,
                task_approval='NOT_EVALUATED: diagnostic metrics, no validated task thresholds')


def behavior_metrics(trace, world, *, motion_expected=True):
    forward = backward = path = rotation = contact = food = hazard = 0.
    gaps = 0; stuck = []; releases = []; food_entries = 0
    in_food = False
    monitor=NavigationMonitor(NavigationPolicy(max_sample_gap_seconds=.00500001))
    for row in trace:
        monitor.add(dict(t=row['simTime'],p=row['position'],yaw=row['yaw'],forwardAxis=row['forward_axis'],
                         avoiding=row.get('obstacle_engaged',False),recovering=False,
                         motorCoupled=bool(row['motion_enabled']),motionExpected=bool(motion_expected)))
    audit=monitor.result()
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
    resumed = audit['resumptionEpisodes']
    stuck = audit['stationaryWindows']
    return dict(signed_forward_mm=forward-backward, forward_mm=forward, backward_mm=backward,
                horizontal_path_mm=path, absolute_rotation_rad=rotation,
                contact_duration_s=contact, food_dwell_s=food, hazard_dwell_s=hazard, food_entries=food_entries,
                stuck_status={'FAIL':'DETECTED','PASS':'NOT_DETECTED','INCOMPLETE':'INCOMPLETE_SAMPLING'}.get(audit['status'],'INCOMPLETE_WINDOW'),
                stuck_window_count=len(stuck), first_stuck_tick=round(stuck[0]['endTime']/.0001) if stuck else None,
                navigation=audit,
                avoidance_releases=resumed, observation_gaps=gaps,
                faults=[r['fault'] for r in trace if r['fault']], task_status='NOT_EVALUATED')


def trace_sample(frame):
    b = frame['body']; basis = b['basis']; c = frame['command']
    return dict(simTime=frame['simTime'], tick=frame['tick'], position=b['position'], yaw=b['yaw'],
                forward_axis=[row[0] for row in basis], contact=bool(frame['sensors']['contact']),
                avoidance=bool(c.get('assist_reason') or (frame.get('legacy') or {}).get('avoidanceActive') or
                               (frame.get('legacy') or {}).get('recoveryActive')),
                motion_enabled=bool(c.get('motor_coupled')) and not frame.get('stopped'),
                obstacle_engaged=bool(frame['sensors']['contact'] or min(frame['sensors']['nearRanges'][3:6])<3.5),
                sensor_diagnostics=frame.get('sensor_diagnostics',{}),motor_diagnostics=frame.get('motor_diagnostics',{}),
                command=c, sensors=frame['sensors'], sensory_ports=frame['sensory_ports'],
                motor_rates_Hz=frame['motor_rates_Hz'], fault=frame['fault'])
