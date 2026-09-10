"""CUDA-resident hybrid controller sharing MuJoCo-Warp buffers and stream."""
from pathlib import Path
import hashlib
import numpy as np


class ResidentHybrid:
    def __init__(self,body,model,data,stream,bodies=None):
        import cupy as cp
        import warp as wp
        import mujoco_warp as mjw
        self.cp,self.wp,self.mjw=cp,wp,mjw
        self.body,self.model,self.data=body,model,data
        self.bodies=bodies or [body];self.n=len(self.bodies)
        self.wp_stream=stream;self.stream=cp.cuda.ExternalStream(stream.cuda_stream)
        code=Path(__file__).with_name('hybrid_cuda.cu').read_text(encoding='utf-8')
        self.source_hash=hashlib.sha256(code.encode()).hexdigest()
        self.kernel=cp.RawKernel(code,'hybrid_control',options=('--std=c++17','--fmad=false'))
        self.audit_kernel=cp.RawKernel(code,'audit_physics',options=('--std=c++17','--fmad=false'))
        c=body.controller;stepper=body._stepper;splines=stepper.splines
        if splines is None:raise ValueError('Resident control requires the pinned cubic spline coefficients')
        with self.stream,wp.ScopedStream(stream):
            self.views={key:cp.asarray(getattr(data,key)) for key in ('ctrl','xfrc_applied','xpos','xmat','time','nacon','qpos','qvel','nefc','ncollision')}
            self.health=cp.zeros(4,cp.int32)
            self.contact_world=cp.asarray(data.contact.worldid);self.geom=cp.asarray(data.contact.geom)
            self.force=wp.zeros(data.naconmax,dtype=wp.spatial_vector,device=stream.device)
            self.force_view=cp.asarray(self.force)
            self.contact_ids=wp.array(np.arange(data.naconmax,dtype=np.int32),device=stream.device)
            self.state=cp.asarray(np.array([np.concatenate([getattr(b.controller.cpg_network,k) for k in ('curr_phases','curr_magnitudes')]+[b.controller.retraction_correction,b.controller.stumbling_correction,b.controller.retraction_persistence_counter]) for b in self.bodies]),dtype=cp.float64)
            self.drive=cp.zeros((self.n,2),cp.float64);self.mode=cp.zeros(self.n,cp.int32)
            self.push_left=cp.zeros(self.n,cp.float64);self.push=cp.zeros((self.n,3),cp.float64)
            self.ticks=cp.zeros(self.n,cp.uint64)
            self.constants=[cp.asarray(x,dtype=cp.float64) for x in (
                stepper.neutral,splines.x,splines.coefficients,stepper.vectors,
                stepper.swing+np.array([0.,c.swing_extension]),stepper.knots,
                np.concatenate([c._base_intrinsic_freqs,c.cpg_network.convergence_coefs,c.cpg_network.coupling_weights.ravel(),c.cpg_network.phase_biases.ravel()]),
                [c.retraction_height_threshold,c.stumbling_force_threshold,*c.retraction_rates,*c.stumbling_rates,c.max_correction,c.retraction_persistence_initiation_threshold,c.retraction_persistence_steps,c.enable_adhesion])]
            self.maps=[cp.asarray(x,dtype=cp.int32) for x in (stepper.order,body.act_ids,
                body.sim._intern_adhesionactuatorids_by_fly[body.fly.name],body._tip_ids,body._force_maps['stumble'][0])]
            self.ground=cp.asarray(body._ground_mask,dtype=cp.uint8)
        self.intervals=len(splines.x)-1
        self.stream.synchronize()

    def audit(self):
        with self.stream:
            self.audit_kernel(((self.n+31)//32,),(32,),(*[self.views[k] for k in ('qpos','qvel','nacon','ncollision','nefc')],self.health,
                *map(np.int32,(self.n,self.body.m.nq,self.body.m.nv,self.data.naconmax,self.data.njmax))))

    def check_health(self):
        values=self.health.get(stream=self.stream)
        if values[0]:raise RuntimeError('GPU physics buffer overflow or nonfinite state; interval rejected')
        return dict(contacts=int(values[1]),broadphase=int(values[2]),constraints=int(values[3]))

    def enqueue(self):
        """No allocation, host read or synchronization; suitable for capture."""
        with self.stream,self.wp.ScopedStream(self.wp_stream):
            self.mjw.contact_force(self.model,self.data,self.contact_ids,True,self.force)
            args=(self.state,self.drive,*self.constants,*self.maps,self.ground,self.views['xpos'],self.views['xmat'],
                self.views['nacon'],self.contact_world,self.geom,self.force_view,self.views['ctrl'],self.views['xfrc_applied'],
                self.push_left,self.push,self.mode,self.ticks,self.views['time'],
                *map(np.int32,(self.n,self.body.m.nbody,self.body.m.nu,self.body.thorax,self.intervals,self.data.naconmax)),np.float64(self.body.t0))
            self.kernel(((self.n+31)//32,),(32,),args)

    def set_inputs(self,commands,*,joint_targets=None,adhesion=None):
        if len(commands)!=self.n:raise ValueError('One command per world required')
        drive=np.asarray([b.motor.map(command) for b,command in zip(self.bodies,commands)])
        with self.stream:
            self.drive.set(drive,stream=self.stream)
            self.push_left.set(np.array([b.push_left for b in self.bodies]),stream=self.stream)
            self.push.set(np.asarray([b.push_force for b in self.bodies]),stream=self.stream)
            self.mode.fill(0 if joint_targets is None else 1)
            if joint_targets is not None:
                targets=np.asarray(joint_targets);pads=np.asarray(adhesion)
                if targets.shape!=(self.n,42) or not np.isfinite(targets).all() or np.max(np.abs(targets))>10 or pads.shape!=(self.n,6) or pads.dtype!=np.bool_:
                    raise ValueError('Invalid batched joint targets')
                self.views['ctrl'][:,self.maps[1]]=self.cp.asarray(targets,dtype=self.cp.float32)
                self.views['ctrl'][:,self.maps[2]]=self.cp.asarray(pads,dtype=self.cp.float32)
        for b,d in zip(self.bodies,drive):b.descending=d.copy()

    def sync_controllers(self):
        states=self.state.get(stream=self.stream);push=self.push_left.get(stream=self.stream)
        for i,(body,row) in enumerate(zip(self.bodies,states)):
            cp=body.controller.cpg_network
            cp.curr_phases[:]=row[:6];cp.curr_magnitudes[:]=row[6:12]
            body.controller.retraction_correction[:]=row[12:18]
            body.controller.stumbling_correction[:]=row[18:24]
            body.controller.retraction_persistence_counter[:]=row[24:30].astype(int)
            body.push_left=float(push[i])
            body.last_action=body.Action(joint_angles=body.d.ctrl[body.act_ids].copy(),
                adhesion_onoff=body.d.ctrl[body.sim._intern_adhesionactuatorids_by_fly[body.fly.name]].astype(bool))

    def snapshot(self):
        return dict(shader=self.source_hash,**{k:getattr(self,k).get(stream=self.stream) for k in ('state','drive','mode','ticks','push_left','push')})

    def validate_state(self,saved):
        if saved.get('shader')!=self.source_hash:raise ValueError('GPU controller shader mismatch')
        for key in ('state','drive','mode','ticks','push_left','push'):
            dest=getattr(self,key);value=saved.get(key)
            if not isinstance(value,np.ndarray) or value.shape!=dest.shape or value.dtype!=dest.dtype or not np.isfinite(value).all():
                raise ValueError('Invalid GPU controller state: '+key)
        if np.any((saved['mode']!=0)&(saved['mode']!=1)) or np.any(saved['push_left']<0):
            raise ValueError('Invalid GPU controller mode or push clock')

    def restore(self,saved):
        self.validate_state(saved)
        with self.stream:
            for key in ('state','drive','mode','ticks','push_left','push'):getattr(self,key).set(saved[key],stream=self.stream)
