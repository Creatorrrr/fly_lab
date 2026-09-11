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
        model_name=getattr(getattr(body,'body_options',None),'model','neuromechfly')
        path=assets_dir/f'model/{model_name}/vision.yaml'
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
        # Multisample resolve on the Windows OpenGL driver changes a handful
        # of edge pixels by one uint8 level even for an unchanged scene. A
        # single sample makes optical input replayable on this render backend.
        self.model.vis.quality.offsamples=0
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
        from .retina_compute import RetinaCompute
        self.compute=RetinaCompute(self.retina)
        self.renderer=mj.Renderer(self.model,height=self.retina.nrows,width=self.retina.ncols)
        self.camera_ids=[self.model.camera(name).id for name in camera_names]
        self.metadata=dict(schema='flylab.compound-eyes.v2',body_model_hash=body.model_hash,
            vision_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source='https://neuromechfly.org/',eyes=['left','right'],
            channels=['yellow','pale'],ommatidia_per_eye=len(self.retina.pale_type_mask),
            neural_mapping=None,observation_only=True,biological_validation=False,
            retinal_compute_backend=self.compute.backend,render_samples=0,
            optics_contract='official-fisheye-single-sample-v2')

    def observe(self,*,include_rgb=True):
        b,mj=self.body,self.mj
        mask=mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mj.mj_stateSize(b.m,mask))
        mj.mj_getState(b.m,b.d,state,mask)
        if len(state)!=mj.mj_stateSize(self.model,mask):
            raise ValueError('Eye observation state layout differs')
        # Only the private observation model is forwarded/rendered.
        for name in ('geom_rgba','geom_friction','pair_friction','geom_pos','geom_quat'):
            getattr(self.model,name)[:]=getattr(b.m,name)
        self.model.opt.gravity[:]=b.m.opt.gravity
        mj.mj_setState(self.model,self.data,state,mask)
        mj.mj_forward(self.model,self.data)
        images=[]
        for camera in self.camera_ids:
            self.renderer.update_scene(self.data,camera,scene_option=self.options)
            images.append(self.renderer.render())
        raw,ommatidia=self.compute.process(images,include_rgb=include_rgb)
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

    def __init__(self,*,field='gaussian',half_concentration=1.,sigma_mm=30.**.5,core_radius_mm=.1,
                 wind_mm_s=(2.,0.,0.),pulse_hz=2.,site_calibration=None):
        if field not in ('gaussian','inverse-square','plume'):
            raise ValueError('Unknown odor field')
        for name,value in (('half_concentration',half_concentration),('sigma_mm',sigma_mm),('core_radius_mm',core_radius_mm)):
            if type(value) not in (int,float) or not np.isfinite(value) or not 0<value<=100:
                raise ValueError('Invalid odor parameter: '+name)
        self.model=dict(kind='four-site-odor-v1',field=field,half_concentration=half_concentration,
                        sigma_mm=sigma_mm,core_radius_mm=core_radius_mm)
        if field=='plume':
            wind=np.asarray(wind_mm_s,dtype=float)
            if wind.shape!=(3,) or not np.isfinite(wind).all() or not 0<np.linalg.norm(wind)<=100:
                raise ValueError('Plume requires a finite nonzero wind vector in mm/s')
            if type(pulse_hz) not in (int,float) or not 0<pulse_hz<=100:
                raise ValueError('Invalid plume frequency')
            self.model.update(wind_mm_s=wind.tolist(),pulse_hz=pulse_hz,
                plume_model='advected periodic Gaussian packets; engineering field, not CFD')
        self.placement='antenna body origins; palp sites are rostrum-relative engineering offsets'
        if site_calibration is not None:
            if not isinstance(site_calibration,dict) or set(site_calibration)!={'sites','evidence','uncertainty'}:
                raise ValueError('Site calibration requires sites, evidence and uncertainty')
            if not site_calibration['evidence'] or not site_calibration['uncertainty']:
                raise ValueError('Site calibration provenance required')
            rows=site_calibration['sites']
            if not isinstance(rows,list) or len(rows)!=4:raise ValueError('Four calibrated sites required')
            sites=[]
            for expected,row in zip(self.sites,rows):
                offset=np.asarray(row.get('offset_mm'),dtype=float)
                if row.get('name')!=expected[0] or not isinstance(row.get('body'),str) or offset.shape!=(3,) or not np.isfinite(offset).all() or np.max(np.abs(offset))>5:
                    raise ValueError('Invalid calibrated olfactory site')
                sites.append((row['name'],row['body'],tuple(offset)))
            self.sites=tuple(sites)
            self.model['site_calibration']=copy.deepcopy(site_calibration)
            self.placement='external site calibration; see evidence and uncertainty'

    def observe(self,body,world):
        from .common import to_physics
        sites=self.sites;placement=self.placement
        flybody=getattr(getattr(body,'body_options',None),'model',None)=='flybody'
        if flybody and 'site_calibration' not in self.model:
            sites=tuple((name,{'l_funiculus':'l_antenna','r_funiculus':'r_antenna'}.get(segment,segment),offset) for name,segment,offset in sites)
            placement='FlyBody antenna origins; palp engineering offsets from rostrum origin in thorax frame; not anatomically calibrated'
        points=[]
        for name,segment,offset in sites:
            if segment not in body.body_indices:
                raise ValueError('Olfactory body segment unavailable: '+segment)
            index=int(body.body_ids[body.body_indices[segment]])
            basis=body.d.xmat[index].reshape(3,3)
            if flybody and name.startswith('palp') and 'site_calibration' not in self.model:
                basis=body.d.xmat[body.thorax].reshape(3,3)
            points.append(body.d.xpos[index]+basis@np.asarray(offset))
        points=np.asarray(points)
        raw=np.zeros((4,2))
        for source in world['sources']:
            if source['kind']=='food' and not world['foodOn']:continue
            component=('food','hazard').index(source['kind'])
            dist2=np.sum((points-np.asarray(to_physics(source['p'])))**2,axis=1)
            if self.model['field']=='gaussian':
                field=np.exp(-dist2/(2*self.model['sigma_mm']**2))
            elif self.model['field']=='inverse-square':
                field=1./(dist2+self.model['core_radius_mm']**2)
            else:
                wind=np.asarray(self.model['wind_mm_s']);speed=np.linalg.norm(wind)
                relative=points-np.asarray(to_physics(source['p']))
                downstream=relative@(wind/speed)
                cross2=np.maximum(0.,dist2-downstream**2)
                age=body.physics_time()-downstream/speed
                pulse=(.5+.5*np.cos(2*np.pi*self.model['pulse_hz']*age))**4
                field=np.exp(-cross2/(2*self.model['sigma_mm']**2))*pulse*np.exp(-np.maximum(0.,downstream)/(10*self.model['sigma_mm']))
                field=np.where(downstream>=0,field,0.)
            raw[:,component]+=source['strength']*field
        transduced=raw/(raw+self.model['half_concentration'])
        return dict(schema='flylab.four-site-odor.v1',time_s=body.physics_time(),
            model=copy.deepcopy(self.model),sites=[r[0] for r in self.sites],components=['food','hazard'],
            positions_native_mm=points.tolist(),concentration=raw.tolist(),response=transduced.tolist(),
            concentration_unit='analytic field units; not calibrated molar concentration',
            placement=placement,site_bodies=[r[1] for r in sites],
            neural_mapping='explicit sensor profile only',biological_validation=False)
