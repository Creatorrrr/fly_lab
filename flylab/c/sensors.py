"""Versioned concentration transduction, retaining B's observation boundary."""
import copy
import math
import numpy as np
from ..common import to_ui
from ..sensors import SensorAdapter
from .integrity import finite


class CSensorAdapter(SensorAdapter):
    def __init__(self, seed, model=None):
        super().__init__(seed)
        self.model = copy.deepcopy(model or {'kind': 'legacy-clipped-v1'})
        if self.model.get('kind') not in ('legacy-clipped-v1', 'compressive-odor-v2','four-site-odor-v1','flygym-multimodal-v2'):
            raise ValueError('Unknown sensor transduction model')
        if self.model['kind'] == 'compressive-odor-v2':
            finite(self.model.get('half_concentration'), 'half concentration', .001, 100.)
        self.four_site_odor=None
        if self.model['kind'] in ('four-site-odor-v1','flygym-multimodal-v2'):
            from ..flygym_senses import FourSiteOdor
            self.four_site_odor=FourSiteOdor(field=self.model.get('field','gaussian'),
                half_concentration=self.model.get('half_concentration',1.),
                sigma_mm=self.model.get('sigma_mm',30.**.5),core_radius_mm=self.model.get('core_radius_mm',.1),
                wind_mm_s=self.model.get('wind_mm_s',(2.,0.,0.)),pulse_hz=self.model.get('pulse_hz',2.),
                site_calibration=self.model.get('site_calibration'))
        self.retina=None
        if self.model.get('retina') is not None:
            from ..retinal_input import RetinalInput
            self.retina=RetinalInput(self.model['retina'])
        self.odor_adaptation=np.zeros((4,2))
        self.adaptation_tick=None
        self.adaptation_tau=finite(self.model.get('adaptation_tau_s',0.),'odor adaptation tau',0.,100.)
        self.adaptation_strength=finite(self.model.get('adaptation_strength',.5),'odor adaptation strength',0.,1.)
        self.diagnostics = {}
        self.previous_silhouette=None

    def observe(self, body, world, dt, config):
        previous = self.previousOdor
        packet = super().observe(body, world, dt, config)
        p, R, _ = body.pose()
        head = p + R @ np.array([.65, 0., .12])
        points = [to_ui(head + R @ np.array([0., y, 0.])) for y in (.3, -.3)]
        def raw(point, kind):
            if kind == 'food' and not world['foodOn']: return 0.
            return sum(o['strength'] * math.exp(-float(np.sum((np.asarray(point)-o['p'])**2))/60.)
                       for o in world['sources'] if o['kind'] == kind)
        food = [raw(p, 'food') for p in points]
        hazard = raw(to_ui(head), 'hazard')
        if self.model['kind'] == 'compressive-odor-v2':
            k = self.model['half_concentration']
            packet['odor'] = [x/(x+k) for x in food]
            packet['danger'] = hazard/(hazard+k)
            mean = sum(packet['odor'])/2
            packet['odorChange'] = float(np.clip((mean-previous)/dt, -2., 2.))
            self.previousOdor = mean
        self.diagnostics = dict(model=self.model, concentration_unit='engineering Gaussian field units',
                                food_raw=food, hazard_raw=hazard, food_transduced=packet['odor'],
                                food_would_clip_legacy=[x>=1 for x in food], hazard_transduced=packet['danger'])
        if self.four_site_odor is not None:
            odor=self.four_site_odor.observe(body,world)
            response=np.asarray(odor['response'])
            if self.adaptation_tau:
                now=round(body.physics_time()*10000)
                if self.adaptation_tick is not None and now<self.adaptation_tick:raise ValueError('Olfactory clock moved backwards')
                elapsed=0. if self.adaptation_tick is None else (now-self.adaptation_tick)/10000
                self.odor_adaptation+=-math.expm1(-elapsed/self.adaptation_tau)*(response-self.odor_adaptation)
                response=np.clip(response-self.adaptation_strength*self.odor_adaptation,0.,1.)
                self.adaptation_tick=now
            odor['adapted_response']=response.tolist()
            packet['odor']=response[:2,0].tolist()
            packet['danger']=float(response[:2,1].mean())
            mean=sum(packet['odor'])/2.
            packet['odorChange']=float(np.clip((mean-previous)/dt,-2.,2.))
            self.previousOdor=mean
            self.diagnostics.update(four_site_odor=odor,food_raw=[r[0] for r in odor['concentration'][:2]],
                hazard_raw=float(np.asarray(odor['concentration'])[:2,1].mean()),
                food_transduced=packet['odor'],hazard_transduced=packet['danger'],
                concentration_unit=odor['concentration_unit'],
                palp_neural_mapping='explicit versioned ports; no implicit mapping')
            self.diagnostics['features']={f'odor_{site}_{component}':float(response[i,j])
                for i,site in enumerate(odor['sites']) for j,component in enumerate(odor['components'])}
        if self.model.get('extended_observations'):
            visible=body.obstacle_silhouette(head,R) if hasattr(body,'obstacle_silhouette') else [0.]*64
            coverage=[sum(visible[:32])/32.,sum(visible[32:])/32.]
            previous=self.previous_silhouette if self.previous_silhouette is not None else coverage
            self.diagnostics.setdefault('features',{}).update(loom_left=max(0.,(coverage[0]-previous[0])/dt),
                loom_right=max(0.,(coverage[1]-previous[1])/dt),head_contact=float(body.head_contact()) if hasattr(body,'head_contact') else 0.)
            self.diagnostics['silhouette']=visible
            self.previous_silhouette=coverage
        if self.retina is not None:
            observation=self.retina.observe(body)
            self.diagnostics.setdefault('features',{}).update(observation.pop('features'))
            self.diagnostics['retina']=observation
        self.last = copy.deepcopy(packet)
        return packet

    def snapshot(self):
        state=dict(super().snapshot(), transduction=self.model, diagnostics=copy.deepcopy(self.diagnostics),
                    previous_silhouette=copy.deepcopy(self.previous_silhouette))
        if self.retina is not None:state['retina']=self.retina.snapshot()
        if self.adaptation_tau:state['odor_adaptation']=dict(tick=self.adaptation_tick,values=self.odor_adaptation.tolist())
        return state

    def restore(self, state):
        if state.get('transduction', {'kind':'legacy-clipped-v1'}) != self.model:
            raise ValueError('Sensor transduction checkpoint mismatch')
        coverage=state.get('previous_silhouette')
        if coverage is not None:
            if not isinstance(coverage,list) or len(coverage)!=2:raise ValueError('Invalid optical history')
            for x in coverage:finite(x,'silhouette coverage',0.,1.)
        candidate=SensorAdapter(1)
        candidate.restore(state)
        retinal=None
        if self.retina is not None:
            from ..retinal_input import RetinalInput
            retinal=RetinalInput(self.model['retina']);retinal.restore(state.get('retina',{}))
        adaptation=np.zeros((4,2));adaptation_tick=None
        if self.adaptation_tau:
            saved=state.get('odor_adaptation',{})
            adaptation=np.asarray(saved.get('values'),dtype=float);adaptation_tick=saved.get('tick')
            if adaptation.shape!=(4,2) or not np.isfinite(adaptation).all() or np.any(adaptation<0) or np.any(adaptation>1) or (adaptation_tick is not None and (type(adaptation_tick) is not int or adaptation_tick<0)):
                raise ValueError('Invalid odor adaptation state')
        self.rng,self.previousOdor,self.last=candidate.rng,candidate.previousOdor,candidate.last
        self.previous_silhouette=copy.deepcopy(coverage)
        self.diagnostics = copy.deepcopy(state.get('diagnostics', {}))
        self.retina=retinal;self.odor_adaptation=adaptation.copy();self.adaptation_tick=adaptation_tick
