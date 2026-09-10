"""Physical NeuroMechFly adapter. No kinematic fallback is allowed here.

FlyGym 2.1.0 APIs are pinned. Position and attitude are read, NEVER overwritten
by the locomotion loop. MuJoCo resolves contact, gravity and joint actuators.
"""
from __future__ import annotations
import math, hashlib, json
from importlib.metadata import version
import numpy as np
from . import PHYSICS_DT, CONTROL_DT, FLYGYM_COMMIT
from .common import S, to_ui, to_physics, clamp, number, clone

LEGS=('lf','lm','lh','rf','rm','rh')
LINKS=('coxa','trochanterfemur','tibia','tarsus1','tarsus2','tarsus3','tarsus4','tarsus5')


from .dependencies import dependency_report


class MotorAdapter:
    """Engineered speed/yaw -> left/right CPG amplitude, NOT a speed servo.
    Native tutorial [1.2,.4] turns right; positive UI yaw is native clockwise.
    """
    def map(self,command):
        speed=number(command['forwardSpeed'],'forwardSpeed',-10,10)
        turn=number(command['yawRate'],'yawRate',-5,5)
        if abs(speed)<1e-7 and abs(turn)<1e-7: return np.zeros(2)
        base=clamp(speed/3.3,-1.0,1.0)
        delta=clamp(turn/3.0,-1,1)*(.8 if abs(speed)<.1 else .38)
        return np.clip([base+delta,base-delta],-1.3,1.3)


class FlyGymBody:
    backend='flygym-2.1.0-neuromechfly'
    test_double=False
    def __init__(self,seed,world,config,*,optimized=True,initial_pose=None):
        report=dependency_report()
        if not report['ready']:
            raise RuntimeError('물리 엔진 의존성이 없거나 버전이 다릅니다. '+report['hint']+' '+str(report['versions']))
        import mujoco as mj
        from flygym.simulation import Simulation
        from flygym.compose import FlatGroundWorld, ActuatorType
        from flygym.anatomy import BodySegment, ContactBodiesPreset
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (make_locomotion_fly, PreprogrammedSteps,
            HybridTurningController, HybridControllerObservation, LocomotionAction, apply_locomotion_action)
        self.mj=mj; self.ActuatorType=ActuatorType; self.BodySegment=BodySegment
        self.Observation=HybridControllerObservation; self.Action=LocomotionAction; self.apply=apply_locomotion_action
        self.optimized=optimized
        self.motor=MotorAdapter(); self.world_spec=clone(world); self.config=clone(config)
        self.fly=make_locomotion_fly(name='flylab',add_adhesion=True,colorize=False)
        self.native_world=FlatGroundWorld(name='flylab_arena',half_size=100)
        w=self.native_world
        # Group 0 is physical world; all fly geoms are excluded from sensor rays.
        w.ground_geom.group=0
        bx,_,bz=world['bounds']
        for k,(p,size) in enumerate([((bx+.5,0,2),(.5,bz+1,2)),((-bx-.5,0,2),(.5,bz+1,2)),((0,bz+.5,2),(bx+1,.5,2)),((0,-bz-.5,2),(bx+1,.5,2))]):
            geom=w.mjcf_root.worldbody.add_geom(name=f'wall_{k}',type=mj.mjtGeom.mjGEOM_BOX,pos=p,size=size,rgba=(.16,.21,.26,1),contype=0,conaffinity=0,group=0)
            w.ground_geoms.append(geom)
        # Bright wall stripe: a geometric visual landmark, not a yaw oracle.
        marker=w.mjcf_root.worldbody.add_geom(name='visual_landmark',type=mj.mjtGeom.mjGEOM_BOX,
            pos=(bx-.12,0,3),size=(.08,2.8,3),rgba=(.1,.9,.8,1),contype=0,conaffinity=0,group=0)
        w.ground_geoms.append(marker)
        # Preallocated static slots avoid rebuilding the body when adding obstacles.
        for i in range(12):
            obj=world['obstacles'][i] if i<len(world['obstacles']) else None
            pos=to_physics(obj['p']) if obj else (0,0,-100-i)
            radius=obj['r'] if obj else 1.
            geom=w.mjcf_root.worldbody.add_geom(name=f'obstacle_{i}',type=mj.mjtGeom.mjGEOM_SPHERE,pos=pos,size=(radius,0,0),rgba=(.35,.40,.46,1),contype=0,conaffinity=0,group=0)
            w.ground_geoms.append(geom)
        spawn, quaternion = [0,0,.8], [1,0,0,0]
        if initial_pose is not None:
            if not isinstance(initial_pose,dict) or set(initial_pose)!={'position','yaw_rad'}:
                raise ValueError('Initial position and yaw_rad required')
            p = initial_pose['position']
            if not isinstance(p,list) or len(p)!=3: raise ValueError('Initial UI position requires three values')
            number(p[0],'initial x',-bx+2,bx-2);number(p[1],'initial height',.3,3.)
            number(p[2],'initial z',-bz+2,bz-2)
            yaw=number(initial_pose['yaw_rad'],'initial yaw',-math.pi,math.pi)
            spawn=to_physics(p);quaternion=[math.cos(yaw/2),0.,0.,-math.sin(yaw/2)]
        w.add_fly(self.fly,spawn,Rotation3D('quat',quaternion),
                  bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                  add_ground_contact_sensors=False)
        # Keep fixed segment frames addressable (in particular c_head).
        # Otherwise upstream body lookup returns -1 for fused segments.
        w.mjcf_root.compiler.fusestatic=False
        self.sim=Simulation(w,timestep=PHYSICS_DT)
        self.sim.reset()  # official neutral-keyframe initialization before settling
        self.m=self.sim.mj_model; self.d=self.sim.mj_data
        self.order=self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        if len(self.order)!=42: raise RuntimeError(f'Expected 42 active DOFs, found {len(self.order)}')
        self.steps=PreprogrammedSteps()
        self.controller=HybridTurningController(timestep=PHYSICS_DT,preprogrammed_steps=self.steps,output_dof_order=self.order)
        self.controller.reset(seed=int(seed))
        self.neutral=self.steps.default_pose_by_dof_order(self.order)
        self.last_action=LocomotionAction(joint_angles=self.neutral.copy(),adhesion_onoff=np.ones(6,dtype=bool))
        self.apply(self.sim,self.fly.name,self.last_action)
        self.body_order=self.fly.get_bodysegs_order()
        self.body_indices={b.name:i for i,b in enumerate(self.body_order)}
        self.body_ids=self.sim._internal_bodyids_by_fly[self.fly.name]
        if (self.body_ids<0).any():
            raise RuntimeError('Missing compiled body segment IDs; refusing invalid telemetry')
        self.thorax=int(self.body_ids[self.body_indices['c_thorax']])
        self.act_ids=np.asarray(self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][self.fly.name])
        self.joint_ids=self.m.actuator_trnid[self.act_ids,0]
        self.qpos_ids=self.m.jnt_qposadr[self.joint_ids]
        self.qvel_ids=self.m.jnt_dofadr[self.joint_ids]
        self.joint_names=[mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_JOINT,int(i)) for i in self.joint_ids]
        self.mass=float(self.m.body_mass[self.body_ids].sum())
        self.gravity0=self.m.opt.gravity.copy(); self.weight0=self.mass*float(np.linalg.norm(self.gravity0))
        self.pair_friction0=self.m.pair_friction.copy(); self.geom_friction0=self.m.geom_friction.copy()
        self.obstacle_ids=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,f'obstacle_{i}') for i in range(12)]
        self.world_geom_ids=set(int(i) for i in self.sim._internal_ground_geom_ids)
        self.landmark_id=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'visual_landmark')
        self.floor_id=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,w.ground_geom.name)
        self.ray_mask=np.array([1,0,0,0,0,0],dtype=np.uint8)
        self.m.geom_group[:]=2
        for gid in self.world_geom_ids: self.m.geom_group[gid]=0
        self._prepare_execution()
        # Eye-independent camera preview (not used by neural controller).
        self.renderer=None; self.camera_error=None
        self.descending=np.zeros(2); self.travel=0.; self.collisions=0; self.prev_contact=False; self.push_left=0.; self.push_force=np.zeros(3)
        self.fault=None; self.walk_ticks=0
        self.set_world(world); self.set_config(config)
        # Neutral-pose physical settling; model clocks start AFTER these 0.2 s.
        for _ in range(2000): self.sim.step()
        mj.mj_forward(self.m,self.d)
        self.t0=float(self.d.time); self.last_position=self.d.xpos[self.thorax].copy()
        identity=dict(commit=FLYGYM_COMMIT,adapter='B-0.2.1',mujoco=version('mujoco'),numpy=version('numpy'),joints=self.joint_names,nq=self.m.nq,nv=self.m.nv,nu=self.m.nu,body_names=[b.name for b in self.body_order])
        # Spawn transforms affect the compiled model. Keep the default identity
        # byte-compatible with B/C checkpoints, but distinguish explicit poses.
        if initial_pose is not None: identity['initial_pose']=clone(initial_pose)
        self.model_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if not np.isfinite(self.d.qpos).all(): raise RuntimeError('Non-finite state after neutral settling')

    def _prepare_execution(self):
        from .locomotion import CachedHybridStepper
        self._stepper=CachedHybridStepper(self.controller,compiled_splines=self.optimized)
        self._tip_ids=np.array([self.body_ids[self.body_indices[f'{leg}_tarsus5']] for leg in LEGS])
        geom_by_segment=self.sim._internal_geomid_by_bodyseg_by_fly[self.fly.name]
        self._ground_mask=np.zeros(self.m.ngeom,dtype=bool)
        self._ground_mask[list(self.world_geom_ids)]=True
        self._force_maps={}
        for key,links in [('stumble',('tibia','tarsus1','tarsus2')),('feet',('tarsus1','tarsus2','tarsus3','tarsus4','tarsus5')),('legs',LINKS)]:
            indices=np.full(self.m.ngeom,-1,dtype=np.int32)
            for i,name in enumerate(f'{leg}_{link}' for leg in LEGS for link in links):
                indices[geom_by_segment[self.BodySegment(name)]]=i
            self._force_maps[key]=(indices,6*len(links))
        self._fly_body_mask=np.zeros(self.m.nbody,dtype=bool); self._fly_body_mask[self.body_ids]=True
        self._core_mask=np.zeros(self.m.nbody,dtype=bool)
        for name,i in self.body_indices.items():
            if name in ('c_thorax','c_head') or name.startswith('c_abdomen'):self._core_mask[self.body_ids[i]]=True
        self._panorama_directions=np.array([[math.cos(k/64*math.tau-math.pi),-math.sin(k/64*math.tau-math.pi),0.] for k in range(64)])
        self._near_directions=np.array([[math.cos((k-4)*math.pi/8),-math.sin((k-4)*math.pi/8),0.] for k in range(9)])

    def _contact_forces(self,key,*,floor_only=False):
        mapping,size=self._force_maps[key]
        forces=np.zeros((size,3))
        contacts=self.d.contact
        n=self.d.ncon
        g1=contacts.geom1[:n];g2=contacts.geom2[:n]
        i1=mapping[g1];i2=mapping[g2]
        active=(((i1>=0)&self._ground_mask[g2])|((i2>=0)&self._ground_mask[g1]))&(contacts.exclude[:n]==0)
        if floor_only: active &= (g1==self.floor_id)|(g2==self.floor_id)
        wrench=np.zeros(6)
        for j in np.flatnonzero(active):
            self.mj.mj_contactForce(self.m,self.d,int(j),wrench)
            force=contacts.frame[j].reshape(3,3).T@wrench[:3]
            if i1[j]>=0:forces[i1[j]]-=force
            if i2[j]>=0:forces[i2[j]]+=force
        return forces

    def _observation(self):
        return self.Observation(thorax_z=float(self.d.xpos[self.thorax,2]),
            tarsus5_z=self.d.xpos[self._tip_ids,2],
            stumbling_contact_forces=self._contact_forces('stumble').reshape(6,3,3),
            fly_heading=self.d.xmat[self.thorax].reshape(3,3)[:,0].copy())

    def physics_time(self):
        return float(self.d.time-self.t0)

    def leg_observation(self):
        """Local joint receptors and physical leg contact, without world pose."""
        forces = self._contact_forces('legs').reshape(6, len(LINKS), 3)
        floor = self._contact_forces('legs',floor_only=True).reshape(6,len(LINKS),3)
        adhesion_ids=self.sim._intern_adhesionactuatorids_by_fly[self.fly.name]
        adhesion=np.maximum(0.,self.d.actuator_force[adhesion_ids])/max(self.weight0,1e-30)
        # The flat floor's normal is +z. Adhesion presses the pad into this
        # surface and adds a reaction force; report that contribution separately.
        # This is an engineering support-load estimate, not receptor strain.
        normal=np.maximum(0.,floor.sum(axis=1)[:,2])/max(self.weight0,1e-30)
        return dict(angles_rad=self.d.qpos[self.qpos_ids].copy(),
                    velocities_rad_s=self.d.qvel[self.qvel_ids].copy(),
                    load_bw=np.linalg.norm(forces, axis=2).sum(axis=1)/max(self.weight0, 1e-30),
                    support_load_bw=np.maximum(0.,normal-adhesion),
                    adhesion_force_bw=adhesion,
                    floor_normal_load_bw=normal,
                    non_support_load_bw=np.linalg.norm(forces-floor,axis=2).sum(axis=1)/max(self.weight0,1e-30),
                    segment_contact_bw=np.linalg.norm(forces,axis=2)/max(self.weight0,1e-30))

    def joint_observation(self):
        """Optional force instrumentation; does not alter the walking controller."""
        return dict(schema='flylab.joint-observation.v1', time_s=self.physics_time(),
                    names=list(self.joint_names), angles_rad=self.d.qpos[self.qpos_ids].copy(),
                    velocities_rad_s=self.d.qvel[self.qvel_ids].copy(),
                    positive_axes_world=self.d.xaxis[self.joint_ids].copy(),
                    range_rad=self.m.jnt_range[self.joint_ids].copy(),
                    actuator_control=self.d.ctrl[self.act_ids].copy(),
                    actuator_force=self.d.actuator_force[self.act_ids].copy(),
                    actuator_torque=self.d.qfrc_actuator[self.qvel_ids].copy(),
                    passive_torque=self.d.qfrc_passive[self.qvel_ids].copy(),
                    constraint_torque=self.d.qfrc_constraint[self.qvel_ids].copy(),
                    applied_generalized_torque=self.d.qfrc_applied[self.qvel_ids].copy(),
                    torque_unit='MuJoCo model units; SI conversion not calibrated',
                    actuator_mode='position; force is servo force, not reconstructed muscle force')

    def step_joint_targets(self, targets, adhesion, dt=CONTROL_DT):
        """Direct neural muscle adapter. Does not advance the predefined CPG."""
        targets, adhesion = np.asarray(targets), np.asarray(adhesion)
        if targets.shape != (42,) or not np.isfinite(targets).all() or np.max(np.abs(targets)) > 10:
            raise ValueError('Invalid neural joint targets')
        if adhesion.shape != (6,) or adhesion.dtype != np.dtype(bool):
            raise ValueError('Six boolean adhesion states required')
        number(dt, 'joint control dt', PHYSICS_DT, .05)
        n = round(dt/PHYSICS_DT)
        if abs(n*PHYSICS_DT-dt) > 1e-10: raise ValueError('Integral physical steps required')
        if self.fault: raise RuntimeError(self.fault)
        self.descending.fill(0.)
        self.last_action = self.Action(joint_angles=targets.copy(), adhesion_onoff=adhesion.copy())
        for _ in range(n):
            self.apply(self.sim, self.fly.name, self.last_action)
            self.d.xfrc_applied[:] = 0
            if self.push_left > 0:
                self.d.xfrc_applied[self.thorax, :3] = self.push_force
                self.push_left = max(0., self.push_left-PHYSICS_DT)
            self.sim.step()
            if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
                self.fault = 'non-finite physics state'
                raise RuntimeError(self.fault)
        self.mj.mj_forward(self.m, self.d)
        p, R, _ = self.pose()
        self.travel += float(np.linalg.norm(p-self.last_position))
        self.last_position = p
        collision = self.nonfoot_contact()
        if collision and not self.prev_contact: self.collisions += 1
        self.prev_contact = collision
        self.walk_ticks += 1
        if R[2, 2] < .15 or p[2] < 0 or np.linalg.norm(p[:2]) > 80:
            self.fault = 'Neural muscle body fell or left physical domain; no automatic pose correction'

    def set_config(self,config):
        self.config=clone(config)
        self.m.opt.gravity[:]=self.gravity0*config.get('gravity',1.)
        self.m.pair_friction[:]=self.pair_friction0
        self.m.pair_friction[:,:2]*=config.get('friction',1.)
        self.m.geom_friction[:]=self.geom_friction0
        self.m.geom_friction[:,0]*=config.get('friction',1.)

    def set_world(self,world):
        if world['obstacles']!=self.world_spec['obstacles']:
            raise ValueError('Static collision geometry changes require rebuilding the model')
        self.world_spec=clone(world)
        # A dark visual landmark remains a wall: only the sensor brightness changes.
        self.m.geom_rgba[self.landmark_id]=[.1,.9,.8,1] if world['cueOn'] else [.16,.21,.26,1]

    def pose(self):
        R=self.d.xmat[self.thorax].reshape(3,3).copy(); p=self.d.xpos[self.thorax].copy()
        vel=np.zeros(6); self.mj.mj_objectVelocity(self.m,self.d,self.mj.mjtObj.mjOBJ_BODY,self.thorax,vel,0)
        return p,R,vel

    def ray_hit(self,origin,direction):
        hit=np.array([-1],dtype=np.int32)
        r=self.mj.mj_ray(self.m,self.d,np.asarray(origin,dtype=float),np.asarray(direction,dtype=float),self.ray_mask,1,-1,hit)
        return float(r),int(hit[0])

    def ray(self,origin,direction,limit=20.):
        r,_=self.ray_hit(origin,direction)
        return min(limit,r) if r>=0 else limit

    def visual_panorama(self,origin,R):
        # 64 horizontal ray receptors. Occlusion is resolved by MuJoCo geometry.
        # Brightness is a categorical synthetic material, not retinal photometry.
        values=[]
        for k in range(64):
            a=k/64*math.tau-math.pi
            _,gid=self.ray_hit(origin,R@np.array([math.cos(a),-math.sin(a),0.]))
            values.append(.92 if gid==self.landmark_id and self.world_spec['cueOn'] else 0.)
        return values

    def sensor_rays(self,origin,R):
        if not self.optimized:
            ranges=[self.ray(origin,R@direction,10) for direction in self._near_directions]
            return ranges,self.visual_panorama(origin,R)
        # Keep the same per-ray rotation arithmetic and exact MuJoCo geometry;
        # submit all rays from this receptor origin in one native call.
        directions=np.array([R@direction for direction in np.concatenate((self._near_directions,self._panorama_directions))])
        ids=np.empty(73,dtype=np.int32);distances=np.empty(73)
        self.mj.mj_multiRay(self.m,self.d,np.asarray(origin,dtype=float),directions.ravel(),
            self.ray_mask,1,-1,ids,distances,None,73,np.inf)
        ranges=np.where(distances[:9]<0,10,np.minimum(distances[:9],10)).tolist()
        panorama=np.where((ids[9:]==self.landmark_id)&self.world_spec['cueOn'],.92,0.).tolist()
        return ranges,panorama

    def obstacle_silhouette(self,origin,R):
        """Coarse horizontal obstacle silhouette; an explicit visual proxy."""
        directions=np.array([R@direction for direction in self._panorama_directions])
        ids=np.empty(64,dtype=np.int32);distances=np.empty(64)
        self.mj.mj_multiRay(self.m,self.d,np.asarray(origin,dtype=float),directions.ravel(),
            self.ray_mask,1,-1,ids,distances,None,64,np.inf)
        visible=np.isin(ids,self.obstacle_ids).astype(float)
        return visible.tolist()

    def head_contact(self):
        head_id=int(self.body_ids[self.body_indices['c_head']])
        for contact in self.d.contact[:self.d.ncon]:
            if contact.exclude:continue
            if ((self.m.geom_bodyid[contact.geom1]==head_id and int(contact.geom2) in self.world_geom_ids) or
                (self.m.geom_bodyid[contact.geom2]==head_id and int(contact.geom1) in self.world_geom_ids)):
                return True
        return False

    def perturb(self,bw=.5,duration=.05):
        number(bw,'pushBW',-2,2); number(duration,'duration',.005,.2)
        _,R,_=self.pose(); self.push_force=-R[:,1]*self.weight0*bw; self.push_left=duration

    def step(self,command,dt=CONTROL_DT):
        if self.fault: raise RuntimeError('물리 중단: '+self.fault+' · 초기화가 필요합니다.')
        n=round(dt/PHYSICS_DT)
        if abs(n*PHYSICS_DT-dt)>1e-10: raise ValueError('Control dt must be integer physical steps')
        self.descending=self.motor.map(command)
        parked=np.max(np.abs(self.descending))<1e-7
        if self.optimized and not parked:
            self._stepper.set_drive(self.descending)
        for _ in range(n):
            if parked:
                # Park at neutral with position actuators; no hidden root braking.
                self.last_action=self.Action(joint_angles=self.neutral.copy(),adhesion_onoff=np.ones(6,dtype=bool))
            else:
                obs=self._observation() if self.optimized else self.Observation.from_sim(self.sim,self.fly.name)
                self.last_action=self._stepper.step(self.descending,obs,drive_prepared=True) if self.optimized else self.controller.step(self.descending,obs)
            self.apply(self.sim,self.fly.name,self.last_action)
            self.d.xfrc_applied[:]=0
            if self.push_left>0:
                self.d.xfrc_applied[self.thorax,:3]=self.push_force; self.push_left=max(0,self.push_left-PHYSICS_DT)
            self.sim.step()
            if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
                self.fault='non-finite physics state'; raise RuntimeError(self.fault)
        # mj_step leaves some derived fields at the previous integration point.
        # Recompute for timestamp-consistent displayed poses/contact force telemetry.
        self.mj.mj_forward(self.m,self.d)
        p,R,_=self.pose(); self.travel+=float(np.linalg.norm(p-self.last_position)); self.last_position=p
        collision=self.nonfoot_contact()
        if collision and not self.prev_contact: self.collisions+=1
        self.prev_contact=collision; self.walk_ticks+=1
        if R[2,2]<0.15 or p[2]<0 or np.linalg.norm(p[:2])>80:
            self.fault='넘어짐 또는 유효 영역 이탈. 자동으로 위치를 보정하지 않습니다.'

    def nonfoot_contact(self):
        # Floor support is normal; any leg/body contact with walls or obstacles
        # is an avoidance event. Keep the name for the sensor adapter contract.
        if self.optimized:
            c=self.d.contact;n=self.d.ncon
            g1=c.geom1[:n];g2=c.geom2[:n];b1=self.m.geom_bodyid[g1];b2=self.m.geom_bodyid[g2]
            w1=self._ground_mask[g1];w2=self._ground_mask[g2]
            hits=(w1&(g1!=self.floor_id)&self._fly_body_mask[b2])|(w2&(g2!=self.floor_id)&self._fly_body_mask[b1])|(self._core_mask[b1]&w2)|(self._core_mask[b2]&w1)
            return bool(np.any(hits&(c.exclude[:n]==0)))
        target={int(self.body_ids[i]) for n,i in self.body_indices.items() if n in ('c_thorax','c_head') or n.startswith('c_abdomen')}
        for c in self.d.contact[:self.d.ncon]:
            if c.exclude: continue
            b1=int(self.m.geom_bodyid[c.geom1]); b2=int(self.m.geom_bodyid[c.geom2])
            if ((int(c.geom1) in self.world_geom_ids and int(c.geom1)!=self.floor_id and b2 in self.body_ids) or
                (int(c.geom2) in self.world_geom_ids and int(c.geom2)!=self.floor_id and b1 in self.body_ids)): return True
            if (b1 in target and int(c.geom2) in self.world_geom_ids) or (b2 in target and int(c.geom1) in self.world_geom_ids): return True
        return False

    def frame(self):
        p,R,vel=self.pose(); heading=R[:,0]; yaw=math.atan2(-heading[1],heading[0]); ui_R=S@R
        positions=self.sim.get_body_positions(self.fly.name)
        legs={leg:[to_ui(positions[self.body_indices[f'{leg}_{link}']]) for link in LINKS if f'{leg}_{link}' in self.body_indices] for leg in LEGS}
        foot_forces=(self._contact_forces('feet') if self.optimized else self.sim.get_bodysegment_contact_forces(self.fly.name,[self.BodySegment(f'{leg}_{link}') for leg in LEGS for link in ('tarsus1','tarsus2','tarsus3','tarsus4','tarsus5')],ground_only=True)).reshape(6,5,3).sum(axis=1)
        contacts=np.linalg.norm(foot_forces,axis=1)/max(self.weight0,1e-30)
        segments={name:to_ui(positions[i]) for name,i in self.body_indices.items() if name in ('c_thorax','c_head') or name.startswith('c_abdomen')}
        if 'c_abdomen12' in segments: segments['c_abdomen']=segments['c_abdomen12']
        body=dict(position=to_ui(p),yaw=yaw,yawRate=-float(vel[2]),speed=float(vel[3:]@heading),verticalSpeed=float(vel[5]),roll=0.,pitch=math.asin(clamp(heading[2],-1,1)),basis=ui_R.tolist(),phase=0.,contact=int(self.prev_contact),collisions=self.collisions,travel=self.travel,legs=legs,segments=segments,contactsBW=contacts.tolist())
        physics=dict(backend=self.backend,testDouble=False,physicalDt=PHYSICS_DT,controlDt=CONTROL_DT,
                     physicsTime=float(self.d.time-self.t0),settlingTime=self.t0,substeps=round(CONTROL_DT/PHYSICS_DT),
                     jointNames=self.joint_names,jointAngles=self.d.qpos[self.qpos_ids].tolist(),jointVelocities=self.d.qvel[self.qvel_ids].tolist(),
                     jointTargets=np.asarray(self.last_action.joint_angles).tolist(),actuatorForces=self.sim.get_actuator_forces(self.fly.name,self.ActuatorType.POSITION).tolist(),
                     torqueUnit='MuJoCo model units (not SI calibrated)',contactsBW=contacts.tolist(),contactVectorsBW=[to_ui(f/max(self.weight0,1e-30)) for f in foot_forces],
                     adhesion=np.asarray(self.last_action.adhesion_onoff,dtype=int).tolist(),cpgPhases=self.controller.cpg_network.curr_phases.tolist(),
                     descending=self.descending.tolist(),fault=self.fault,gravity=self.config['gravity'],friction=self.config['friction'],bodyModelHash=self.model_hash)
        return body,physics

    def snapshot(self):
        typ=self.mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(self.mj.mj_stateSize(self.m,typ)); self.mj.mj_getState(self.m,self.d,state,typ)
        cp=self.controller.cpg_network; rs=cp.random_state.get_state()
        return dict(backend=self.backend,modelHash=self.model_hash,state=state.tolist(),t0=self.t0,
             cpg={k:np.asarray(getattr(cp,k)).tolist() for k in ('curr_phases','curr_magnitudes','intrinsic_freqs','intrinsic_amps')},
             rng=[rs[0],rs[1].tolist(),int(rs[2]),int(rs[3]),float(rs[4])],
             controller={k:np.asarray(getattr(self.controller,k)).tolist() for k in ('retraction_correction','stumbling_correction','retraction_persistence_counter')},
             action=dict(joint_angles=np.asarray(self.last_action.joint_angles).tolist(),adhesion_onoff=np.asarray(self.last_action.adhesion_onoff,dtype=int).tolist()),
             descending=self.descending.tolist(),travel=self.travel,collisions=self.collisions,prev_contact=self.prev_contact,
             push_left=self.push_left,push_force=self.push_force.tolist(),last_position=self.last_position.tolist(),fault=self.fault,walk_ticks=self.walk_ticks)

    def restore(self,s):
        if s['backend']!=self.backend or s['modelHash']!=self.model_hash: raise ValueError('다른 물리 모델의 체크포인트입니다.')
        typ=self.mj.mjtState.mjSTATE_INTEGRATION; a=np.asarray(s['state'],dtype=float)
        if a.shape!=(self.mj.mj_stateSize(self.m,typ),) or not np.isfinite(a).all() or np.abs(a).max()>1e9: raise ValueError('Invalid MuJoCo integration state')
        for k in ('curr_phases','curr_magnitudes','intrinsic_freqs','intrinsic_amps'):
            x=np.asarray(s['cpg'][k],dtype=float)
            if x.shape!=(6,) or not np.isfinite(x).all(): raise ValueError('Invalid CPG state')
            getattr(self.controller.cpg_network,k)[:]=x
        for k in ('retraction_correction','stumbling_correction','retraction_persistence_counter'):
            x=np.asarray(s['controller'][k]); dest=getattr(self.controller,k)
            if x.shape!=dest.shape or not np.isfinite(x).all(): raise ValueError('Invalid reflex state')
            dest[:]=x
        rng=s['rng']; self.controller.cpg_network.random_state.set_state((rng[0],np.asarray(rng[1],dtype=np.uint32),rng[2],rng[3],rng[4]))
        angles=np.asarray(s['action']['joint_angles']); adhesion=np.asarray(s['action']['adhesion_onoff'],dtype=bool)
        if angles.shape!=(42,) or adhesion.shape!=(6,) or not np.isfinite(angles).all(): raise ValueError('Invalid action')
        self.last_action=self.Action(joint_angles=angles,adhesion_onoff=adhesion)
        self.mj.mj_setState(self.m,self.d,a,typ); self.mj.mj_forward(self.m,self.d)
        for k in ('t0','travel','collisions','prev_contact','push_left','fault','walk_ticks'): setattr(self,k,clone(s[k]))
        for k in ('descending','push_force','last_position'): setattr(self,k,np.asarray(s[k],dtype=float))

    def preview(self):
        """Native MuJoCo camera; only a visual diagnostic, not neural input."""
        if self.camera_error: raise RuntimeError(self.camera_error)
        try:
            if self.renderer is None: self.renderer=self.mj.Renderer(self.m,height=240,width=400)
            cam=self.mj.MjvCamera(); cam.type=self.mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:]=self.d.xpos[self.thorax]; cam.distance=9.; cam.azimuth=135; cam.elevation=-28
            self.renderer.update_scene(self.d,cam)
            return self.renderer.render().copy()
        except Exception as e:
            self.camera_error='MuJoCo offscreen renderer: '+str(e)
            raise RuntimeError(self.camera_error) from e
    def close(self):
        if self.renderer is not None: self.renderer.close(); self.renderer=None
