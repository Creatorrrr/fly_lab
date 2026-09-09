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
        if self.model.get('kind') not in ('legacy-clipped-v1', 'compressive-odor-v2'):
            raise ValueError('Unknown sensor transduction model')
        if self.model['kind'] == 'compressive-odor-v2':
            finite(self.model.get('half_concentration'), 'half concentration', .001, 100.)
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
        if self.model.get('extended_observations'):
            visible=body.obstacle_silhouette(head,R) if hasattr(body,'obstacle_silhouette') else [0.]*64
            coverage=[sum(visible[:32])/32.,sum(visible[32:])/32.]
            previous=self.previous_silhouette if self.previous_silhouette is not None else coverage
            self.diagnostics['features']=dict(loom_left=max(0.,(coverage[0]-previous[0])/dt),
                loom_right=max(0.,(coverage[1]-previous[1])/dt),head_contact=float(body.head_contact()) if hasattr(body,'head_contact') else 0.)
            self.diagnostics['silhouette']=visible
            self.previous_silhouette=coverage
        self.last = copy.deepcopy(packet)
        return packet

    def snapshot(self):
        return dict(super().snapshot(), transduction=self.model, diagnostics=copy.deepcopy(self.diagnostics),
                    previous_silhouette=copy.deepcopy(self.previous_silhouette))

    def restore(self, state):
        if state.get('transduction', {'kind':'legacy-clipped-v1'}) != self.model:
            raise ValueError('Sensor transduction checkpoint mismatch')
        coverage=state.get('previous_silhouette')
        if coverage is not None:
            if not isinstance(coverage,list) or len(coverage)!=2:raise ValueError('Invalid optical history')
            for x in coverage:finite(x,'silhouette coverage',0.,1.)
        candidate=SensorAdapter(1)
        candidate.restore(state)
        self.rng,self.previousOdor,self.last=candidate.rng,candidate.previousOdor,candidate.last
        self.previous_silhouette=copy.deepcopy(coverage)
        self.diagnostics = copy.deepcopy(state.get('diagnostics', {}))
