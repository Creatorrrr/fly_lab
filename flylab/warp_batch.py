"""Shared-model GPU physics with device-resident controllers and explicit views."""
from dataclasses import fields,is_dataclass,dataclass
from importlib.metadata import version
import numpy as np
from .body import FlyGymBody
from .physics import PhysicsBodyFactory,warp_option_metadata
from .warp_control import ResidentHybrid
from . import CONTROL_DT,PHYSICS_DT


class BatchBodyView(FlyGymBody):
    backend='flygym-warp-batch-view-v1'

    def __init__(self,*args,**kwargs):
        self._settling=True
        self.gpu_options=None
        super().__init__(*args,**kwargs)
        self._settling=False

    def _step_physics(self):
        if self._settling:return super()._step_physics()
        raise RuntimeError('Batched bodies advance only through their GPU batch owner')

    def physics_identity(self):
        identity=super().physics_identity()
        if self.gpu_options is not None:
            identity['cpu_mirror_options']={k:identity[k] for k in self.gpu_options}
            identity.update(self.gpu_options)
        return dict(identity,precision='float32',experimental=True,
            observation_source='shared MuJoCo-Warp worlds, synchronized at control boundary',
            controller_backend='CUDA resident hybrid',checkpoint_scope='whole batch; CPU view alone cannot resume physics')


@dataclass(frozen=True)
class BatchViewFactory(PhysicsBodyFactory):
    def __call__(self,seed,world,config,**kwargs):
        return BatchBodyView(seed,world,config,physics_profile=self.profile,**kwargs)


class WarpWorldBatch:
    def __init__(self,bodies):
        import warp as wp
        import mujoco_warp as mjw
        import cupy as cp
        if not bodies:raise ValueError('Nonempty physics batch required')
        if version('warp-lang')!='1.14.0' or version('mujoco-warp')!='3.9.0':raise ValueError('Pinned Warp packages required')
        self.bodies=list(bodies);first=bodies[0];self.n=len(bodies)
        if any(b.model_hash!=first.model_hash or b.world_spec!=first.world_spec or b.config!=first.config for b in bodies):
            raise ValueError('Batch worlds must share body, static geometry and physics configuration')
        self.wp,self.mjw,self.cp=wp,mjw,cp;self.m=first.m;self.profile=first.physics_profile
        self.stream=wp.Stream(wp.get_device('cuda:0'));self.graph=None;self.renderer=None
        self.runtime=dict(warp=version('warp-lang'),mujoco_warp=version('mujoco-warp'),device=wp.get_device('cuda:0').name)
        with wp.ScopedStream(self.stream):
            self.model=mjw.put_model(self.m)
            options=warp_option_metadata(self.model)
            for body in self.bodies:body.gpu_options=dict(options)
            self.data=self._put_data()
            mjw.step(self.model,self.data);mjw.forward(self.model,self.data);wp.synchronize_stream(self.stream)
            self.data=self._put_data()
        self.control=ResidentHybrid(first,self.model,self.data,self.stream,bodies)
        self.control.ticks.set(np.array([round(b.physics_time()/PHYSICS_DT) for b in bodies],np.uint64),stream=self.control.stream)
        saved=self.snapshot()
        with wp.ScopedStream(self.stream):self.control.enqueue();self.control.audit()
        self.restore(saved)

    def _put_data(self):
        data=self.mjw.put_data(self.m,self.bodies[0].d,nworld=self.n,
            njmax=self.profile.max_constraints,nconmax=self.profile.max_contacts)
        for name in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','history'):
            dest=getattr(data,name)
            value=np.asarray([getattr(b.d,name) for b in self.bodies])
            if value.size:dest.assign(value)
        data.time.assign(np.array([b.d.time for b in self.bodies],np.float32))
        self.mjw.forward(self.model,data)
        return data

    def advance(self,commands,targets=None,adhesion=None):
        self.control.set_inputs(commands,joint_targets=targets,adhesion=adhesion)
        with self.wp.ScopedStream(self.stream):
            if self.graph is None:
                with self.wp.ScopedCapture() as captured:
                    for _ in range(round(CONTROL_DT/PHYSICS_DT)):
                        self.control.enqueue();self.mjw.step(self.model,self.data);self.control.audit()
                    self.mjw.forward(self.model,self.data)
                self.graph=captured.graph
            self.wp.capture_launch(self.graph)

    def sync(self):
        self.capacity_peak=self.control.check_health()
        ticks=self.control.ticks.get(stream=self.control.stream)
        with self.wp.ScopedStream(self.stream):
            for i,b in enumerate(self.bodies):
                self.mjw.get_data_into(b.d,b.m,self.data,world_id=i)
                n=b.d.ncon;b.d.contact.geom1[:n]=b.d.contact.geom[:n,0];b.d.contact.geom2[:n]=b.d.contact.geom[:n,1]
                b.d.contact.exclude[:n]=(b.d.contact.efc_address[:n]<0).astype(np.int32)
                b.d.time=b.t0+int(ticks[i])*PHYSICS_DT
        self.control.sync_controllers()
        for b in self.bodies:
            p,R,_=b.pose();b.travel+=float(np.linalg.norm(p-b.last_position));b.last_position=p
            hit=b.nonfoot_contact()
            if hit and not b.prev_contact:b.collisions+=1
            b.prev_contact=hit;b.walk_ticks=round(b.physics_time()/CONTROL_DT)
            if b.outside_physical_domain(p,R):b.fault='Batched body fell or left domain'

    def arrays(self):
        result={}
        def collect(obj,prefix=''):
            for f in fields(obj):
                value=getattr(obj,f.name);key=prefix+f.name
                if isinstance(value,self.wp.array):result[key]=value
                elif is_dataclass(value):collect(value,key+'.')
        collect(self.data)
        return result

    def snapshot(self):
        self.wp.synchronize_stream(self.stream)
        return dict(schema='flylab.warp-batch.v1',runtime=self.runtime,model_hash=self.bodies[0].model_hash,
            worlds=self.n,arrays={k:v.numpy().copy() for k,v in self.arrays().items()},controller=self.control.snapshot())

    def restore(self,saved):
        arrays=self.arrays()
        if saved.get('schema')!='flylab.warp-batch.v1' or saved.get('runtime')!=self.runtime or saved.get('worlds')!=self.n or saved.get('model_hash')!=self.bodies[0].model_hash or set(saved.get('arrays',{}))!=set(arrays):
            raise ValueError('Batch physics checkpoint identity mismatch')
        for key,target in arrays.items():
            value=saved['arrays'][key];template=target.numpy()
            if not isinstance(value,np.ndarray) or value.shape!=template.shape or value.dtype!=template.dtype:raise ValueError('Invalid batch physics array: '+key)
            if key in ('qpos','qvel','act','ctrl','qacc_warmstart') and not np.isfinite(value).all():raise ValueError('Nonfinite batch state')
        self.control.validate_state(saved['controller'])
        expected=np.asarray([b.t0 for b in self.bodies])+saved['controller']['ticks']*PHYSICS_DT
        if not np.allclose(saved['arrays']['time'],expected,rtol=1e-6,atol=1e-6):raise ValueError('Batch physics/controller clock mismatch')
        self.control.restore(saved['controller'])
        with self.wp.ScopedStream(self.stream):
            for key,value in arrays.items():value.assign(saved['arrays'][key])
        self.wp.synchronize_stream(self.stream)

    def restore_regrouped(self,saved,rows):
        """Select integration/controller rows and rebuild contact workspaces.

        Contact and constraint arrays contain shared indexing and cannot be
        sliced as independent worlds. A regroup is a new physics epoch.
        """
        if saved.get('schema')!='flylab.warp-batch.v1' or saved.get('runtime')!=self.runtime or saved.get('model_hash')!=self.bodies[0].model_hash:
            raise ValueError('Regrouped physics checkpoint identity mismatch')
        worlds=saved.get('worlds')
        if type(worlds) is not int or not 1<=worlds<=32 or len(rows)!=self.n or len(set(rows))!=len(rows) or any(type(i) is not int or not 0<=i<worlds for i in rows):
            raise ValueError('Invalid physics regroup rows')
        arrays=self.arrays();selected={}
        # Only MuJoCo integration state has independent world rows. Derived
        # contacts, solver workspaces and collision indexes are rebuilt below.
        for key in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','history','time'):
            target=arrays[key].numpy();value=saved.get('arrays',{}).get(key)
            if not isinstance(value,np.ndarray) or value.shape!=(worlds,*target.shape[1:]) or value.dtype!=target.dtype or not np.isfinite(value).all():
                raise ValueError('Invalid regrouped integration state: '+key)
            selected[key]=value[rows].copy()
        source=saved.get('controller',{});controller=dict(shader=source.get('shader'))
        for key in ('state','drive','mode','ticks','push_left','push'):
            value=source.get(key)
            if not isinstance(value,np.ndarray) or value.ndim<1 or value.shape[0]!=worlds:
                raise ValueError('Invalid regrouped controller state: '+key)
            controller[key]=value[rows].copy()
        self.control.validate_state(controller)
        expected=np.asarray([b.t0 for b in self.bodies])+controller['ticks']*PHYSICS_DT
        if not np.allclose(selected['time'],expected,rtol=1e-6,atol=1e-6):raise ValueError('Regrouped physics/controller clock mismatch')
        with self.wp.ScopedStream(self.stream):
            for key,value in selected.items():arrays[key].assign(value)
            self.mjw.forward(self.model,self.data)
        self.control.restore(controller)
        self.wp.synchronize_stream(self.stream)

    def render(self,world=0):
        """Render a selected world on GPU; cameras use the compiled model."""
        if type(world) is not int or not 0<=world<self.n:raise ValueError('Unknown render world')
        from flygym.warp.rendering import WarpGPUBatchRenderer
        if self.m.ncam==0:raise ValueError('Batch model must include a tracking camera')
        with self.wp.ScopedStream(self.stream):
            if self.renderer is None or self.renderer.world_ids!=[world]:
                if self.renderer is not None:self.renderer.close()
                camera=self.bodies[0].mj.mj_id2name(self.m,self.bodies[0].mj.mjtObj.mjOBJ_CAMERA,0)
                self.renderer=WarpGPUBatchRenderer(self.m,camera,n_worlds_total=self.n,worlds=[world],camera_res=(240,400),buffer_frames=False)
            pixels=self.renderer._render_impl(self.data).numpy()[0,0]
        return np.clip(pixels*255,0,255).astype(np.uint8)

    def close(self):
        self.wp.synchronize_stream(self.stream)
        if self.renderer is not None:self.renderer.close();self.renderer=None
