"""Inspectable 15-MTU LF-leg imitation experiment on the official model."""
import copy
import numpy as np
from .integrity import digest


class MuscleImitation:
    def __init__(self,seed=42):
        import mujoco as mj
        from flygym_demo.muscle_imitation import make_imitation_env,ImitationConfig
        self.mj=mj;self.env=make_imitation_env(config=ImitationConfig(test=True,init_noise_scale=0.))
        self.env.reset(seed=seed);self.m=self.env.sim.mj_model;self.d=self.env.sim.mj_data
        self.inverse=mj.MjData(self.m);self.ids=self.env._muscle_actuator_ids
        self.qids=self.env._tracked_qposadrs;self.vids=self.env._tracked_qveladrs
        if len(self.ids)!=15:raise ValueError('Expected official LF model with 15 MTUs')
        self.identity=digest(dict(muscles=self.env.muscle_names,joints=self.env.tracked_joint_names,
            qpos=self.env._clip.qpos.tolist(),dt=self.m.opt.timestep,config=vars(self.env.config)))
        self.steps=0;self.done=False

    def teacher(self,kp=900.,kd=60.):
        from scipy.optimize import lsq_linear
        from .muscles import moment_matrix
        e,m,d,mj=self.env,self.m,self.d,self.mj
        k=min(e._mocap_idx+1,e._clip.n_frames-1)
        mj.mj_copyData(self.inverse,m,d)
        self.inverse.qacc[:]=0.
        self.inverse.qacc[self.vids]=kp*(e._clip.qpos[k]-d.qpos[self.qids])+kd*(e._clip.qvel[k]-d.qvel[self.vids])
        mj.mj_inverse(m,self.inverse)
        moment=moment_matrix(mj,m,d)[self.ids][:,self.vids].T
        gains=np.array([mj.mju_muscleGain(d.actuator_length[i],d.actuator_velocity[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9]) for i in self.ids])
        bias=np.array([mj.mju_muscleBias(d.actuator_length[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_biasprm[i,:9]) for i in self.ids])
        target=self.inverse.qfrc_inverse[self.vids]-moment@bias
        matrix=moment*gains
        scale=max(float(np.linalg.norm(matrix)),1e-12)
        solution=lsq_linear(np.vstack([matrix/scale,.001*np.eye(15)]),np.concatenate([target/scale,np.zeros(15)]),bounds=(0.,1.))
        return solution.x.astype(np.float32)

    def step(self,action=None,*,teacher=False):
        if self.done:raise ValueError('Muscle clip completed')
        if teacher:
            if action is not None:raise ValueError('Select teacher or explicit activation')
            action=self.teacher()
        value=np.asarray(action,dtype=np.float32)
        if value.shape!=(15,) or not np.isfinite(value).all() or np.any(value<0) or np.any(value>1):
            raise ValueError('Fifteen bounded finite muscle activations required')
        obs,reward,terminated,truncated,info=self.env.step(value)
        self.mj.mj_forward(self.m,self.d)
        if not np.isfinite(obs).all():raise RuntimeError('Nonfinite muscle experiment')
        self.steps+=1;self.done=terminated or truncated
        k=min(self.env._mocap_idx,self.env._clip.n_frames-1)
        return dict(time_s=self.steps*self.env.config.control_timestep,activation=value.tolist(),
            q=self.d.qpos[self.qids].tolist(),target_q=self.env._clip.qpos[k].tolist(),
            qdot=self.d.qvel[self.vids].tolist(),target_qdot=self.env._clip.qvel[k].tolist(),
            forces=self.d.actuator_force[self.ids].tolist(),reward=float(reward),done=self.done,
            controller='bounded inverse-dynamics teacher' if teacher else 'explicit activation',
            neural_control=False,biological_validation=False)

    def snapshot(self):
        typ=self.mj.mjtState.mjSTATE_INTEGRATION
        value=np.empty(self.mj.mj_stateSize(self.m,typ));self.mj.mj_getState(self.m,self.d,value,typ)
        return dict(identity=self.identity,state=value,steps=self.steps,done=self.done,index=self.env._mocap_idx,
            reward=self.env._last_reward,rng=copy.deepcopy(self.env._np_random.bit_generator.state))

    def restore(self,state):
        value=np.asarray(state.get('state'));typ=self.mj.mjtState.mjSTATE_INTEGRATION
        if state.get('identity')!=self.identity or value.shape!=(self.mj.mj_stateSize(self.m,typ),) or not np.isfinite(value).all():raise ValueError('Invalid muscle checkpoint')
        k=state.get('index');steps=state.get('steps')
        if type(k) is not int or k<0 or k>=self.env._clip.n_frames or k!=steps or type(state.get('done')) is not bool:raise ValueError('Invalid muscle clock')
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['rng'])
        if abs(value[0]-steps*self.env.config.control_timestep)>1e-8:raise ValueError('Muscle physics clock mismatch')
        self.mj.mj_setState(self.m,self.d,value,typ);self.mj.mj_forward(self.m,self.d)
        self.steps=steps;self.done=state['done'];self.env._mocap_idx=k;self.env._last_reward=state['reward'];self.env._np_random=rng

    def close(self):self.env.close()
