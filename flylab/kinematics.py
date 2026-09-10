"""Recorded FlyGym kinematics with one anatomical conversion and explicit time."""
from pathlib import Path
import hashlib
import numpy as np


def spotlight_trace(body, dt=.001, path=None):
    from flygym_demo.spotlight_data.preprocessing import MotionSnippet
    from importlib.resources import files
    if not np.isfinite(dt) or not .0001<=dt<=.02:raise ValueError('Invalid kinematics sampling period')
    if getattr(getattr(body,'body_options',None),'model','neuromechfly')!='neuromechfly':
        raise ValueError('Spotlight anatomical angles require the NeuroMechFly body; FlyBody uses its own trajectories')
    path=Path(path) if path is not None else Path(str(files('flygym_demo.spotlight_data')))/'assets/spotlight_behavior_clip.npz'
    clip=MotionSnippet(path,angles_global2anatomical=True)
    angles=clip.get_joint_angles(dt,body.order)
    if angles.ndim!=2 or angles.shape[0]<2 or angles.shape[1]!=42 or not np.isfinite(angles).all():
        raise ValueError('Invalid recorded 42-DOF kinematics')
    velocities=np.gradient(angles,dt,axis=0)
    return dict(time_s=np.arange(len(angles))*dt,q_rad=angles,qdot_rad_s=velocities,
        metadata=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            trial=clip.experiment_trial,frame_range=clip.framerange_in_raw_recording,
            source_fps=clip.data_fps,dt=dt,joint_names=body.joint_names,
            coordinates='anatomical; right roll/yaw converted once by MotionSnippet',
            use='calibration/training clip; not held-out evidence',biological_validation=False))
