"""Official FlyBody aerodynamics with explicit flight/takeoff/landing tasks.

Tasks provide rewards and contact-aware success measures for policy development.
The default action is only the upstream prototype wingbeat, not a trained pilot.
Body translation is integrated by MuJoCo; targets never move the free root.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import numpy as np

SOURCE_COMMIT='d015e9bfe441bd90ae431bac24c55cb74bdbce26'


def validate_source(path):
    path=Path(path).resolve()
    revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    changed=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=SOURCE_COMMIT or changed:raise ValueError('A clean pinned official FlyBody source checkout is required')
    return path


def make_flight_env(source,*,task='flight',seed=42,seconds=.3,wing_pattern=None):
    if task not in ('flight','takeoff','landing'):raise ValueError('Unknown flight task')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid flight seed')
    if type(seconds) not in (int,float) or not np.isfinite(seconds) or not .002<=seconds<=10:
        raise ValueError('Flight duration must be in .002..10 seconds')
    if version('mujoco')!='3.9.0' or version('dm-control')!='1.0.39':
        raise RuntimeError('Install requirements-flight.txt for the matching MuJoCo ABI')
    source=validate_source(source)
    sys.path.insert(0,str(source))
    import flybody
    if not Path(flybody.__file__).resolve().is_relative_to(source):raise ValueError('Another FlyBody package is already loaded')
    from dm_control import composer
    from dm_control.locomotion.arenas.floors import Floor
    from flybody.fruitfly.fruitfly import FruitFly
    from flybody.tasks.base import Flying
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator

    class FlightTask(Flying):
        def __init__(self):
            super().__init__(walker=FruitFly,arena=Floor(),time_limit=seconds,joint_filter=0.,
                disable_legs=False,floor_contacts=True,num_user_actions=1,body_pitch_angle=0. if task=='takeoff' else 47.5)
            self.pattern=WingBeatPatternGenerator(base_pattern_path=None if wing_pattern is None else str(wing_pattern))
            self.walker.mjcf_model.size.njmax=None;self.walker.mjcf_model.size.nconmax=None
            self.root_entity.mjcf_model.size.memory='128M'
            self.wing_indices=self.walker._action_indices['wings'];self.frequency_index=self.walker._action_indices['user'][0]
            self.target_height_cm=.5 if task!='landing' else .13
            self.last_metrics={}

        def initialize_episode(self,physics,random_state):
            super().initialize_episode(physics,random_state);self._should_terminate=False
            # Keep the official model's joint reference pose when the walker
            # does not provide a qpos vector (WalkerPose.qpos defaults to None).
            pose=self.walker.upright_pose
            if pose.qpos is not None:physics.bind(self._non_root_joints).qpos=pose.qpos
            height=.13 if task=='takeoff' else .5
            angle=np.deg2rad(0. if task=='takeoff' else -47.5)
            self.walker.set_pose(physics,[0.,0.,height],[np.cos(angle/2),0.,np.sin(angle/2),0.])
            self.walker.set_velocity(physics,np.zeros(3),np.zeros(3))
            q,v=self.pattern.reset(initial_phase=random_state.uniform(),return_qvel=True)
            physics.bind(self._wing_joints).qpos=q;physics.bind(self._wing_joints).qvel=v
            self.initial_height_cm=height;self.ground_hold_s=0.;self.air_hold_s=0.;self.metric_time=0.

        def before_step(self,physics,action,random_state):
            action=np.asarray(action).copy()
            spec=self.walker.get_action_spec(physics)
            if action.shape!=spec.shape or not np.isfinite(action).all() or np.any(action<spec.minimum) or np.any(action>spec.maximum):
                raise ValueError('Finite flight actions within the official actuator bounds required')
            frequency=self.pattern.base_beat_freq*(1+self.pattern.rel_freq_range*action[self.frequency_index])
            target=self.pattern.step(ctrl_freq=frequency)
            action[self.wing_indices]+=target-physics.bind(self._wing_joints).qpos
            super().before_step(physics,action,random_state)

        def get_reward_factors(self,physics):
            pos,quat=self.walker.get_pose(physics)
            velocity=physics.bind(self._root_joint).qvel[:3]
            grounded=False
            ground_ids=set(int(physics.bind(g).element_id) for g in self._arena.ground_geoms)
            for c in physics.data.contact:
                if c.dist<=0 and (int(c.geom1) in ground_ids or int(c.geom2) in ground_ids):grounded=True;break
            elapsed=max(0.,physics.time()-self.metric_time);self.metric_time=physics.time()
            self.ground_hold_s=self.ground_hold_s+elapsed if grounded else 0.
            self.air_hold_s=self.air_hold_s+elapsed if not grounded else 0.
            upright=float(1-2*(quat[1]**2+quat[2]**2))
            speed=float(np.linalg.norm(velocity))
            success=(self.air_hold_s>=.05 and pos[2]>=.4 and upright>.3 if task=='takeoff' else
                     self.ground_hold_s>=.05 and speed<.5 and upright>.5 if task=='landing' else
                     self.air_hold_s>=.1 and abs(pos[2]-.5)<.1 and speed<1. and upright>.3)
            self.last_metrics=dict(task=task,position_mm=(np.asarray(pos)*10).tolist(),velocity_cm_s=velocity.tolist(),
                ground_contact=grounded,ground_hold_s=self.ground_hold_s,air_hold_s=self.air_hold_s,success=bool(success),
                upright=upright,neural_control=False,biological_validation=False)
            return [float(np.exp(-((pos[2]-self.target_height_cm)/.3)**2)),float(np.exp(-speed**2/25.)),max(0.,upright)]

        def check_termination(self,physics):
            return not np.isfinite(physics.data.qpos).all() or not np.isfinite(physics.data.qvel).all() or super().check_termination(physics)

    flight_task=FlightTask()
    period=flight_task.control_timestep
    if abs(seconds/period-round(seconds/period))>1e-8:raise ValueError('Flight duration must contain integral control periods')
    # Composer compares accumulated floating-point physics time. Place the
    # timeout between the last two integral control boundaries, so N controls
    # always yield LAST at N*dt even when that float rounds just below seconds.
    environment=composer.Environment(task=flight_task,time_limit=seconds-period/2,random_state=np.random.RandomState(seed),
        strip_singleton_obs_buffer_dim=True)
    environment.flylab_provenance=dict(source_commit=SOURCE_COMMIT,task=task,physics='MuJoCo CPU with official ellipsoid wing forces',
        policy_device='controller supplied by caller; Torch policies may use CUDA',
        wing_pattern='upstream prototype' if wing_pattern is None else hashlib.sha256(Path(wing_pattern).read_bytes()).hexdigest(),
        measured_wingbeat_supplied=wing_pattern is not None,ground_contact=True,active_legs=True,
        duration_seconds=seconds,control_period_s=period,control_steps=round(seconds/period),
        biological_validation=False,trained_policy=False)
    return environment
