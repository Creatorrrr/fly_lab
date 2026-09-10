"""Mechanical tracking control on the same two-MTU rig as the BANC assay.

This controller knows its target. Its success is a mechanical control, never
evidence of connectome feedback or a physiologically calibrated controller.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .integrity import boolean, digest, finite
from .muscles import MuscleRig, moment_matrix
from .muscle_calibration import direction_status
from .receptors import JointReceptors, ReceptorParameters


@dataclass(frozen=True)
class TeacherParameters:
    dt: float = .001
    kp_s2: float = 900.
    kd_s: float = 60.
    max_excitation: float = 1.

    def __post_init__(self):
        finite(self.dt,'teacher dt',.0001,.005)
        finite(self.kp_s2,'teacher kp',0.,1e5)
        finite(self.kd_s,'teacher kd',0.,1e4)
        finite(self.max_excitation,'maximum excitation',.001,1.)


class JointTeacher:
    def __init__(self, body=None, parameters=None):
        self.body=body or MuscleRig()
        self.p=parameters or TeacherParameters()
        self.steps=round(self.p.dt/self.body.model.opt.timestep)
        if abs(self.steps*self.body.model.opt.timestep-self.p.dt)>1e-12:
            raise ValueError('Teacher period must contain integral physical ticks')
        lo,hi=self.body.model.jnt_range[0]
        self.receptors=JointReceptors(ReceptorParameters(float(lo),float(hi),dt=self.p.dt))
        self.identity=digest(dict(body=self.body.identity,parameters=asdict(self.p),
                                 controller='inverse-dynamics-bounded-antagonist-v1'))
        self.tick=0

    def command(self,target_q,target_velocity=0.):
        b=self.body;m,d,mj=b.model,b.data,b.mj
        finite(target_q,'teacher target',float(m.jnt_range[0,0]),float(m.jnt_range[0,1]))
        finite(target_velocity,'target velocity',-1000.,1000.)
        observation=b.frame()
        if direction_status(observation)['status']!='CONSISTENT':
            raise RuntimeError('FAULT_MUSCLE_DIRECTION')
        desired_acc=self.p.kp_s2*(target_q-d.qpos[0])+self.p.kd_s*(target_velocity-d.qvel[0])
        required=(observation['effective_inertia']*desired_acc+observation['bias_torque']-
                  observation['passive_joint_torque']-
                  float(np.dot(observation['moment_arm_mm'],observation['passive_muscle_force'])))
        gains=np.array([mj.mju_muscleGain(d.actuator_length[i],d.actuator_velocity[i],
            m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9]) for i in b.act_ids])
        coefficients=moment_matrix(mj,m,d)[b.act_ids,0]*gains
        excitation=np.zeros(2)
        candidates=np.flatnonzero((coefficients*required>0)&(np.abs(coefficients)>1e-12))
        if len(candidates):
            best=int(candidates[np.argmax(np.abs(coefficients[candidates]))])
            excitation[best]=np.clip(required/coefficients[best],0.,self.p.max_excitation)
        return excitation,dict(target_q_rad=target_q,target_velocity_rad_s=target_velocity,
            required_active_torque=float(required),torque_per_activation=coefficients.tolist(),
            saturated=bool(np.any(excitation>=self.p.max_excitation)))

    def step(self,target_q,target_velocity=0.,*,torque=0.,enabled=True):
        boolean(enabled,'teacher enabled')
        excitation,control=self.command(target_q,target_velocity)
        sensed=self.receptors.step(float(self.body.data.qpos[0]),float(self.body.data.qvel[0]))
        frame=self.body.step(excitation if enabled else np.zeros(2),steps=self.steps,
                             torque=torque,direction_guard=True)
        self.tick+=1
        return dict(control_tick=self.tick,physics=frame,receptors=sensed,teacher=control,
                    teacher_enabled=enabled,neural_control=False,biological_validation=False)

    def snapshot(self):
        return dict(schema='flylab.joint-teacher-state.v1',identity=self.identity,tick=self.tick,
                    body=self.body.snapshot(),receptors=self.receptors.snapshot())

    def restore(self,saved):
        if saved.get('schema')!='flylab.joint-teacher-state.v1' or saved.get('identity')!=self.identity:
            raise ValueError('Teacher checkpoint identity mismatch')
        tick=saved.get('tick')
        if type(tick) is not int or tick<0 or saved['body']['tick']!=tick*self.steps or saved['receptors']['tick']!=tick:
            raise ValueError('Teacher checkpoint clock mismatch')
        candidate=JointTeacher(MuscleRig(self.body.source),self.p)
        candidate.body.restore(saved['body']);candidate.receptors.restore(saved['receptors'])
        if direction_status(candidate.body.frame())['status']!='CONSISTENT':
            raise ValueError('Teacher checkpoint has invalid muscle geometry')
        candidate.tick=tick
        self.__dict__.update(candidate.__dict__)
