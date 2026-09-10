"""Recorded trajectory calibration and independent reference controllers."""
import numpy as np
from . import PHYSICS_DT,CONTROL_DT
from .kinematics import spotlight_trace


def reference_controller(body,kind,seed):
    from flygym_demo.complex_terrain import CPGController,RuleBasedController
    if kind=='hybrid':return body.controller
    if kind=='cpg':return CPGController(body.controller.cpg_network,body.steps,body.order)
    if kind=='rule':return RuleBasedController(timestep=PHYSICS_DT,preprogrammed_steps=body.steps,output_dof_order=body.order,seed=seed)
    raise ValueError('Unknown reference controller')


def run_controller(body,kind,seconds,seed=42):
    if not np.isfinite(seconds) or not .005<=seconds<=60 or abs(seconds/CONTROL_DT-round(seconds/CONTROL_DT))>1e-8:
        raise ValueError('Use an integral number of control periods in .005..60 s')
    controller=reference_controller(body,kind,seed)
    command=dict(forwardSpeed=3.3,yawRate=0.,verticalSpeed=0.)
    body._stepper.set_drive(body.motor.map(command))
    start=body.d.xpos[body.thorax].copy();heading=body.d.xmat[body.thorax].reshape(3,3)[:,0].copy()
    samples=[]
    for tick in range(round(seconds/CONTROL_DT)):
        if kind=='hybrid':body.step(command)
        else:
            for _ in range(round(CONTROL_DT/PHYSICS_DT)):
                action=controller.step()
                if not body.controller.enable_adhesion:action.adhesion_onoff[:]=False
                body.last_action=action;body.apply(body.sim,body.fly.name,action);body._step_physics()
                if not np.isfinite(body.d.qpos).all() or not np.isfinite(body.d.qvel).all():raise RuntimeError('Nonfinite comparison physics')
            body._forward_physics()
            p,R,_=body.pose()
            if R[2,2]<.15 or p[2]<0 or np.linalg.norm(p[:2])>80:body.fault='Comparison body fell or left domain'
        probe=body.contact_probe()
        samples.append(dict(time_s=body.physics_time(),q=body.d.qpos[body.qpos_ids].copy(),
            phase=(controller.leg_phases if kind=='rule' else body.controller.cpg_network.curr_phases).copy(),
            position=body.d.xpos[body.thorax].copy(),slip=np.asarray(probe['slip_speed_mm_s']),
            support=np.asarray(probe['floor_normal_bw']),normal=np.asarray(probe['surface_normals']),
            adhesion=np.asarray(body.last_action.adhesion_onoff).copy()))
        if body.fault:break
    displacement=body.d.xpos[body.thorax]-start
    arrays={key:np.asarray([s[key] for s in samples]) for key in samples[0]}
    summary=dict(controller=kind,seconds_executed=len(samples)*CONTROL_DT,requested_seconds=seconds,
        displacement_mm=float(np.linalg.norm(displacement)),forward_mm=float(displacement@heading),
        slip_integral_mm=(arrays['slip'].sum(axis=0)*CONTROL_DT).tolist(),
        support_fraction=(arrays['support']>1e-6).mean(axis=0).tolist(),fault=body.fault,
        completed=body.fault is None and len(samples)==round(seconds/CONTROL_DT),
        biological_validation=False,neural_control=False,
        input_semantics='native forward rule coordinator' if kind=='rule' else 'held symmetric descending drive [1,1]',
        metric_sampling_s=CONTROL_DT)
    return summary,arrays


def replay_kinematics(body,seconds=.1,*,start_s=0.,path=None,dt=.001):
    from .c.receptors import JointReceptors,ReceptorParameters,CHANNELS
    trace=spotlight_trace(body,dt,path)
    if not np.isfinite(seconds) or not np.isfinite(start_s) or start_s<0 or seconds<=0:
        raise ValueError('Invalid motion replay interval')
    begin=round(start_s/dt);count=round(seconds/dt)
    if abs(begin*dt-start_s)>1e-9 or abs(count*dt-seconds)>1e-9 or count<1 or begin+count>=len(trace['q_rad']):
        raise ValueError('Replay window must fit in the recorded clip on its sample grid')
    q=trace['q_rad'][begin:begin+count+1];v=trace['qdot_rad_s'][begin:begin+count+1]
    ranges=body.m.jnt_range[body.joint_ids].copy()
    for j,(lo,hi) in enumerate(ranges):
        if not body.m.jnt_limited[body.joint_ids[j]] or hi<=lo:
            ranges[j]=[float(trace['q_rad'][:,j].min())-.01,float(trace['q_rad'][:,j].max())+.01]
    receptors=[JointReceptors(ReceptorParameters(float(lo),float(hi),dt=dt)) for lo,hi in ranges]
    # Initialization only. Subsequent qpos changes come solely from integration.
    body.d.qpos[body.qpos_ids]=q[0];body.d.qvel[body.qvel_ids]=v[0]
    body._forward_physics()
    reference=body.mj.MjData(body.m);samples=[]
    for k in range(count):
        sensed=[r.step(float(q[k,j]),float(v[k,j])) for j,r in enumerate(receptors)]
        body.step_joint_targets(q[k],np.zeros(6,dtype=bool),dt)
        # Conditional FK reference shares the actual root pose. It measures
        # joint tracking, not agreement with unrecorded root translation.
        reference.qpos[:]=body.d.qpos;reference.qpos[body.qpos_ids]=q[k+1]
        body.mj.mj_forward(body.m,reference)
        samples.append(dict(time_s=(begin+k+1)*dt,target_q=q[k+1],target_v=v[k+1],
            q=body.d.qpos[body.qpos_ids].copy(),v=body.d.qvel[body.qvel_ids].copy(),
            target_tip=reference.xpos[body._tip_ids].copy(),tip=body.d.xpos[body._tip_ids].copy(),
            force=body.d.actuator_force[body.act_ids].copy(),
            receptors=np.asarray([[item['output'][name] for name in CHANNELS] for item in sensed])))
        if body.fault:break
    arrays={key:np.asarray([s[key] for s in samples]) for key in samples[0]}
    summary=dict(metadata=trace['metadata'],start_s=start_s,seconds_executed=len(samples)*dt,
        q_rmse_rad=float(np.sqrt(np.mean((arrays['q']-arrays['target_q'])**2))),
        velocity_rmse_rad_s=float(np.sqrt(np.mean((arrays['v']-arrays['target_v'])**2))),
        tip_rmse_mm=float(np.sqrt(np.mean((arrays['tip']-arrays['target_tip'])**2))),
        joint_rmse_rad=np.sqrt(np.mean((arrays['q']-arrays['target_q'])**2,axis=0)).tolist(),
        max_abs_actuator_force=float(np.max(np.abs(arrays['force']))),fault=body.fault,
        completed=body.fault is None and len(samples)==count,neural_control=False,biological_validation=False,
        receptor_channels=list(CHANNELS),receptor_ranges_rad=ranges.tolist(),
        receptor_range_source='joint limits where present; recorded calibration clip extrema otherwise',
        reference_tip='conditional FK at actual root; not measured whole-body translation')
    return summary,arrays
