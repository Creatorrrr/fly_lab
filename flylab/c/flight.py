"""Flight research rig with articulated wings and native ellipsoid fluid forces.

Uses the pinned, separately acquired TuragaLab flybody MJCF. The engineering
wingbeat controller is explicit. No root translation or vertical-speed servo
is used during integration. This rig is not a validated neural flight policy.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .integrity import digest, file_hash, finite, bounded_int

class FlightRig:
    def __init__(self,assets,*,fluid=True):
        import mujoco as mj
        self.mj=mj;self.assets=Path(assets).resolve()
        tree=ET.parse(self.assets/'fruitfly.xml');root=tree.getroot()
        # Location is not part of physical model identity. Hash the resolved
        # asset bytes and a canonical directory before compiling locally.
        root.find('compiler').set('meshdir','<content-addressed-assets>')
        option=root.find('option');option.set('timestep','0.00005')
        option.set('density','0.00128' if fluid else '0');option.set('viscosity','0.000185' if fluid else '0')
        for default in root.iter('default'):
            if default.get('class')=='wing':default.find('joint').set('damping','0.007769230')
            if default.get('class') in ('yaw','roll','pitch') and default.find('general') is not None:
                default.find('general').set('gainprm','18')
            if default.get('class')=='wing-fluid':
                default.find('geom').set('fluidshape','ellipsoid');default.find('geom').set('fluidcoef','1.0 0.5 1.5 1.7 1.0')
        # Flight benchmark is an air-only experiment, with self-collisions off.
        # Landing/ground contact is not covered by this configuration.
        for geom in root.iter('geom'):geom.set('contype','0');geom.set('conaffinity','0')
        canonical_xml=ET.tostring(root,encoding='unicode')
        self.identity=digest(dict(schema='flylab.flight-model.v2',xml=canonical_xml,
                                  assets={p.name:file_hash(p) for p in sorted(self.assets.iterdir()) if p.suffix=='.obj'}))
        root.find('compiler').set('meshdir',str(self.assets))
        xml=ET.tostring(root,encoding='unicode')
        self.model=mj.MjModel.from_xml_string(xml);self.data=mj.MjData(self.model);self.tick=0
        self.wing_joints=[self.model.joint('wing_'+axis+'_'+side).id for side in ('left','right') for axis in ('yaw','roll','pitch')]
        self.qpos_indices=self.model.jnt_qposadr[self.wing_joints]
        self.actuators=[self.model.actuator('wing_'+axis+'_'+side).id for side in ('left','right') for axis in ('yaw','roll','pitch')]
        self.data.qpos[:]=self.model.qpos_spring
        angle=-math.radians(47.5);self.data.qpos[:3]=[0.,0.,.5];self.data.qpos[3:7]=[math.cos(angle/2),0.,math.sin(angle/2),0.]
        self.data.qpos[self.qpos_indices]=self.wing_pattern(0.)
        mj.mj_forward(self.model,self.data)

    @staticmethod
    def wing_pattern(phase,amplitude=1.,steering=0.):
        # Same published project's prototype sinusoidal approximation; real
        # measured wingbeat trajectories can replace it as a separate profile.
        yaw=1.1*math.sin(phase-math.pi/2)+.3
        roll=.25*math.sin(2*phase)-.1
        pitch=1.35*math.sin(phase)+.8
        return np.array([amplitude*yaw+steering,roll,pitch,amplitude*yaw-steering,roll,pitch])

    def step(self,steps=4,frequency=218.,amplitude=1.,steering=0.):
        bounded_int(steps,'flight steps',1,10000);finite(frequency,'wing Hz',0.,300.)
        finite(amplitude,'wing amplitude',0.,1.2);finite(steering,'wing asymmetry',-.2,.2)
        for _ in range(steps):
            target=self.wing_pattern(self.data.time*frequency*math.tau,amplitude,steering)
            self.data.ctrl[:]=0.
            self.data.ctrl[self.actuators]=np.clip(target-self.data.qpos[self.qpos_indices],-1.,1.)
            self.mj.mj_step(self.model,self.data);self.tick+=1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError('Flight physics became nonfinite')
        self.mj.mj_forward(self.model,self.data)
        return self.frame()

    def frame(self):
        return dict(tick=self.tick,seconds=self.tick*self.model.opt.timestep,
                    position_mm=(self.data.qpos[:3]*10.).tolist(),quaternion=self.data.qpos[3:7].tolist(),
                    wing_angles_rad=self.data.qpos[self.qpos_indices].tolist(),
                    actuator_forces=self.data.actuator_force[self.actuators].tolist(),
                    model_hash=self.identity,units='upstream cm, g, s; displayed position converted to mm',
                    command_source='explicit prototype wingbeat controller',neural_control=False,
                    fluid_model='MuJoCo ellipsoid lift/drag',flight_stability_validated=False)

    def snapshot(self):
        mask=self.mj.mjtState.mjSTATE_INTEGRATION
        a=np.empty(self.mj.mj_stateSize(self.model,mask));self.mj.mj_getState(self.model,self.data,a,mask)
        return dict(schema='flylab.flight-state.v1',model_hash=self.identity,tick=self.tick,state=a)

    def restore(self,state):
        if state.get('schema')!='flylab.flight-state.v1' or state.get('model_hash')!=self.identity:raise ValueError('Flight model mismatch')
        mask=self.mj.mjtState.mjSTATE_INTEGRATION;a=state['state'];tick=bounded_int(state['tick'],'flight tick')
        if not isinstance(a,np.ndarray) or a.shape!=(self.mj.mj_stateSize(self.model,mask),) or not np.isfinite(a).all():raise ValueError('Invalid flight state')
        candidate=self.mj.MjData(self.model);self.mj.mj_setState(self.model,candidate,a,mask)
        if abs(candidate.time-tick*self.model.opt.timestep)>1e-9:raise ValueError('Flight clock mismatch')
        self.mj.mj_forward(self.model,candidate);self.data=candidate;self.tick=tick
