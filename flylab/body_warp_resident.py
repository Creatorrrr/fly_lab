"""Warp physics and hybrid/reflex control at 10 kHz, observed at 200 Hz."""
import hashlib
import numpy as np
from .body_warp import WarpBody
from . import PHYSICS_DT,CONTROL_DT


class ResidentWarpBody(WarpBody):
    backend='flygym-2.1.0-warp-resident-hybrid-v1'

    def __init__(self,*args,**kwargs):
        self.resident=None;self.period_graphs={}
        super().__init__(*args,**kwargs)
        from .warp_control import ResidentHybrid
        self.resident=ResidentHybrid(self,self.gpu_model,self.gpu_data,self.stream)
        self.model_hash=hashlib.sha256((self.model_hash+self.resident.source_hash).encode()).hexdigest()
        saved=self.snapshot()
        # Compile on a state that is restored before any experiment runs.
        with self.wp.ScopedStream(self.stream):
            self.resident.enqueue();self.resident.audit()
        self.restore(saved)

    def _advance_period(self,command,dt,targets=None,adhesion=None):
        if self.fault:raise RuntimeError(self.fault)
        n=round(dt/PHYSICS_DT)
        if not 1<=n<=500 or abs(n*PHYSICS_DT-dt)>1e-10:raise ValueError('Integral physics interval required')
        self.resident.set_inputs([command],joint_targets=None if targets is None else np.asarray(targets)[None],
            adhesion=None if adhesion is None else np.asarray(adhesion)[None])
        with self.wp.ScopedStream(self.stream):
            if self.physics_profile.cuda_graph:
                if n not in self.period_graphs:
                    with self.wp.ScopedCapture() as capture:
                        for _ in range(n):
                            self.resident.enqueue();self.mjw.step(self.gpu_model,self.gpu_data);self.resident.audit()
                        self.mjw.forward(self.gpu_model,self.gpu_data)
                    self.period_graphs[n]=capture.graph
                self.wp.capture_launch(self.period_graphs[n])
            else:
                for _ in range(n):
                    self.resident.enqueue();self.mjw.step(self.gpu_model,self.gpu_data);self.resident.audit()
                self.mjw.forward(self.gpu_model,self.gpu_data)
        try:
            peaks=self.resident.check_health()
            for key,value in peaks.items():self.capacity_peak[key]=max(self.capacity_peak[key],value)
            self.gpu_ticks+=n;self._sync_observations();self.resident.sync_controllers()
            p,R,_=self.pose();self.travel+=float(np.linalg.norm(p-self.last_position));self.last_position=p
            hit=self.nonfoot_contact()
            if hit and not self.prev_contact:self.collisions+=1
            self.prev_contact=hit;self.walk_ticks+=1
            if R[2,2]<.15 or p[2]<0 or np.linalg.norm(p[:2])>80:self.fault='Resident Warp body fell or left domain'
        except Exception as exc:self.fault=str(exc);raise

    def step(self,command,dt=CONTROL_DT):self._advance_period(command,dt)

    def step_joint_targets(self,targets,adhesion,dt=CONTROL_DT):
        self._advance_period(dict(forwardSpeed=0.,yawRate=0.),dt,targets,adhesion)

    def set_config(self,config):
        super().set_config(config)
        if self.resident is not None:self.resident.model=self.gpu_model;self.period_graphs.clear()

    def physics_identity(self):
        return dict(super().physics_identity(),controller_backend='CUDA hybrid/reflex/contact at 10 kHz',
            observation_cadence='control boundary, normally 200 Hz',experimental=True,
            buffer_contract='CuPy zero-copy views; same underlying CUDA stream; owners retained')

    def snapshot(self):
        result=super().snapshot()
        if self.resident is not None:result['resident_controller']=self.resident.snapshot()
        return result

    def restore(self,state):
        if self.resident is not None:
            self.resident.validate_state(state.get('resident_controller',{}))
            if np.any(state['resident_controller']['ticks']!=state['warp']['ticks']):raise ValueError('Resident clock mismatch')
        super().restore(state)
        if self.resident is not None:self.resident.restore(state['resident_controller'])
