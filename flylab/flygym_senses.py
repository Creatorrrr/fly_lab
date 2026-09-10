"""FlyGym retinal observation and explicit four-site analytic odor fields.

Rendering owns a separate model/data copy. It cannot advance or modify the
live physics, and its observations are not implicitly mapped to neural IDs.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np


class CompoundEyeObserver:
    def __init__(self,body):
        import yaml
        import mujoco as mj
        from flygym import assets_dir
        from flygym.vision.retina import Retina
        self.body,self.mj=body,mj
        path=assets_dir/'model/neuromechfly/vision.yaml'
        config=yaml.safe_load(path.read_text(encoding='utf-8'))
        spec=body.native_world.mjcf_root.copy()
        camera_names=[]
        for side,info in config['sensors'].items():
            parent_id=int(body.body_ids[body.body_indices[info['parent']]])
            parent=mj.mj_id2name(body.m,mj.mjtObj.mjOBJ_BODY,parent_id)
            name='flylab_retina_'+side
            spec.body(parent).add_camera(name=name,pos=info['rel_pos'],euler=info['orientation'],
                                         fovy=config['fovy_per_eye'])
            camera_names.append(name)
        self.model=spec.compile()
        if (self.model.nq,self.model.nv,self.model.ngeom)!=(body.m.nq,body.m.nv,body.m.ngeom):
            raise ValueError('Eye observation model does not match the live body')
        if not np.array_equal(self.model.body_mass,body.m.body_mass):
            raise ValueError('Eye cameras changed model inertia')
        self.data=mj.MjData(self.model)
        # Preserve FlyGym's selective self-occlusion, independently of ray masks.
        self.model.geom_group[:]=body.m.geom_group
        for segment,index in body.body_indices.items():
            body_id=int(body.body_ids[index])
            self.model.geom_group[self.model.geom_bodyid==body_id]=2 if segment in config['hidden_segments'] else 0
        self.options=mj.MjvOption()
        self.options.geomgroup[1]=self.options.geomgroup[2]=0
        self.retina=Retina()
        self.renderer=mj.Renderer(self.model,height=self.retina.nrows,width=self.retina.ncols)
        self.camera_ids=[self.model.camera(name).id for name in camera_names]
        self.metadata=dict(schema='flylab.compound-eyes.v1',body_model_hash=body.model_hash,
            vision_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source='https://neuromechfly.org/',eyes=['left','right'],
            channels=['yellow','pale'],ommatidia_per_eye=len(self.retina.pale_type_mask),
            neural_mapping=None,observation_only=True,biological_validation=False)

    def observe(self):
        b,mj=self.body,self.mj
        mask=mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mj.mj_stateSize(b.m,mask))
        mj.mj_getState(b.m,b.d,state,mask)
        if len(state)!=mj.mj_stateSize(self.model,mask):
            raise ValueError('Eye observation state layout differs')
        # Only the private observation model is forwarded/rendered.
        for name in ('geom_rgba','geom_friction','pair_friction'):
            getattr(self.model,name)[:]=getattr(b.m,name)
        self.model.opt.gravity[:]=b.m.opt.gravity
        mj.mj_setState(self.model,self.data,state,mask)
        mj.mj_forward(self.model,self.data)
        images=[]
        for camera in self.camera_ids:
            self.renderer.update_scene(self.data,camera,scene_option=self.options)
            images.append(self.retina.correct_fisheye(self.renderer.render()))
        raw=np.asarray(images)
        ommatidia=np.asarray([self.retina.raw_image_to_hex_pxls(im) for im in raw],dtype=np.float32)
        if not np.isfinite(ommatidia).all():
            raise RuntimeError('Non-finite retinal observation')
        return dict(metadata=copy.deepcopy(self.metadata),time_s=b.physics_time(),
                    raw_rgb=raw,ommatidia=ommatidia,pale_type_mask=self.retina.pale_type_mask.copy())

    def close(self):
        self.renderer.close()


class FourSiteOdor:
    """Antenna centers plus declared rostrum-relative palp sampling sites.

    The model has no separate palp segments. Palp offsets are therefore an
    explicit engineering placement, pending anatomical calibration.
    """
    sites=(('antenna_left','l_funiculus',(0.,0.,0.)),
           ('antenna_right','r_funiculus',(0.,0.,0.)),
           ('palp_left','c_rostrum',(0.,.06,-.04)),
           ('palp_right','c_rostrum',(0.,-.06,-.04)))

    def __init__(self,*,field='gaussian',half_concentration=1.,sigma_mm=30.**.5,core_radius_mm=.1):
        if field not in ('gaussian','inverse-square'):
            raise ValueError('Unknown odor field')
        for name,value in (('half_concentration',half_concentration),('sigma_mm',sigma_mm),('core_radius_mm',core_radius_mm)):
            if type(value) not in (int,float) or not np.isfinite(value) or not 0<value<=100:
                raise ValueError('Invalid odor parameter: '+name)
        self.model=dict(kind='four-site-odor-v1',field=field,half_concentration=half_concentration,
                        sigma_mm=sigma_mm,core_radius_mm=core_radius_mm)

    def observe(self,body,world):
        from .common import to_physics
        points=[]
        for _,segment,offset in self.sites:
            if segment not in body.body_indices:
                raise ValueError('Olfactory body segment unavailable: '+segment)
            index=int(body.body_ids[body.body_indices[segment]])
            points.append(body.d.xpos[index]+body.d.xmat[index].reshape(3,3)@np.asarray(offset))
        points=np.asarray(points)
        raw=np.zeros((4,2))
        for source in world['sources']:
            if source['kind']=='food' and not world['foodOn']:continue
            component=('food','hazard').index(source['kind'])
            dist2=np.sum((points-np.asarray(to_physics(source['p'])))**2,axis=1)
            if self.model['field']=='gaussian':
                field=np.exp(-dist2/(2*self.model['sigma_mm']**2))
            else:
                field=1./(dist2+self.model['core_radius_mm']**2)
            raw[:,component]+=source['strength']*field
        transduced=raw/(raw+self.model['half_concentration'])
        return dict(schema='flylab.four-site-odor.v1',time_s=body.physics_time(),
            model=copy.deepcopy(self.model),sites=[r[0] for r in self.sites],components=['food','hazard'],
            positions_native_mm=points.tolist(),concentration=raw.tolist(),response=transduced.tolist(),
            concentration_unit='analytic field units; not calibrated molar concentration',
            placement='antenna body origins; palp sites are rostrum-relative engineering offsets',
            neural_mapping='explicit sensor profile only',biological_validation=False)
