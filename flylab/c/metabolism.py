"""Optional, explicit engineering model of ingestion/energy; no learned steering."""
from dataclasses import dataclass, asdict
import copy
import math
from .integrity import finite, digest

@dataclass(frozen=True)
class MetabolicParameters:
    capacity: float=1.
    initial_energy: float=.5
    basal_per_s: float=.001
    locomotion_per_mm: float=.0002
    intake_per_s: float=.05
    mouth_radius_mm: float=.5
    max_feeding_speed_mm_s: float=.5
    def __post_init__(self):
        for key,value in asdict(self).items():finite(value,key,0.,100.)
        if self.capacity<=0 or self.initial_energy>self.capacity or self.mouth_radius_mm<=0:raise ValueError('Invalid energy capacity/feeding radius')

class Metabolism:
    def __init__(self, parameters=None):
        self.p=MetabolicParameters(**(parameters or {}));self.energy=self.p.initial_energy
        self.consumed={};self.tick=0;self.intake=0.;self.expenditure=0.;self.feeding=False

    @property
    def hunger(self):return 1.-self.energy/self.p.capacity

    def step(self, dt, mouth_position, speed, world):
        finite(dt,'metabolic dt',.0001,.1);speed=finite(speed,'metabolic speed',-1000.,1000.)
        if len(mouth_position)!=3 or not all(math.isfinite(x) for x in mouth_position):raise ValueError('Finite mouth pose required')
        spent=min(self.energy,dt*(self.p.basal_per_s+self.p.locomotion_per_mm*abs(speed)))
        self.energy-=spent;self.expenditure+=spent;self.feeding=False;amount=0.
        if world['foodOn'] and abs(speed)<=self.p.max_feeding_speed_mm_s:
            for source in world['sources']:
                if source['kind']!='food' or math.dist(source['p'],mouth_position)>self.p.mouth_radius_mm:continue
                used=self.consumed.get(source['id'],0.)
                available=max(0.,source['strength']-used)
                eaten=min(available,self.p.capacity-self.energy,self.p.intake_per_s*dt-amount)
                if eaten<=0:continue
                self.consumed[source['id']]=used+eaten;self.energy+=eaten;amount+=eaten
                self.feeding=True
        self.intake+=amount;self.tick+=1
        return amount

    def snapshot(self):
        return dict(schema='flylab.metabolism.v1',parameters=asdict(self.p),parameter_hash=digest(asdict(self.p)),
                    energy=self.energy,consumed=copy.deepcopy(self.consumed),tick=self.tick,
                    intake=self.intake,expenditure=self.expenditure,feeding=self.feeding)

    def restore(self,state):
        from .integrity import bounded_int, boolean
        if state.get('schema')!='flylab.metabolism.v1' or state.get('parameter_hash')!=digest(asdict(self.p)) or state.get('parameters')!=asdict(self.p):raise ValueError('Metabolic model mismatch')
        energy=finite(state['energy'],'energy',0.,self.p.capacity)
        intake=finite(state['intake'],'intake',0.,1e9);spent=finite(state['expenditure'],'expenditure',0.,1e9)
        if abs(energy-(self.p.initial_energy+intake-spent))>1e-7:raise ValueError('Energy conservation mismatch')
        consumed=state['consumed']
        if not isinstance(consumed,dict) or len(consumed)>100000 or any(not isinstance(k,str) for k in consumed):raise ValueError('Invalid intake ledger')
        values={k:finite(v,'consumed quantity',0.,1e9) for k,v in consumed.items()}
        if abs(sum(values.values())-intake)>1e-7:raise ValueError('Intake ledger mismatch')
        tick=bounded_int(state['tick'],'metabolic tick');feeding=boolean(state['feeding'],'feeding')
        self.energy,self.intake,self.expenditure,self.consumed,self.tick,self.feeding=energy,intake,spent,copy.deepcopy(values),tick,feeding

    def summary(self):
        return dict(enabled=True,energy=self.energy,hunger=self.hunger,intake=self.intake,feeding=self.feeding,
                    unit='engineering energy units',neural_modulation=False,
                    scope='Proximity and slow movement permit virtual intake; no proboscis/muscle or metabolic calibration')
