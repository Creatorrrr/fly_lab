"""The only upper-level module allowed to read world pose.
Ray receptors use MuJoCo collision geometry (incl. occlusion). Odor is an
analytic Gaussian field. Body angular velocity is an idealized proprioceptor.
None of these sensors is claimed to reproduce compound-eye physiology.
"""
import math
import numpy as np
from .common import RNG,TAU,clamp,number,integer,to_ui,clone

def default_world():
    return dict(bounds=[24,16,18],cueOn=True,cueAngle=0.,foodOn=True,
        sources=[dict(id='food-1',kind='food',p=[8,.7,-4],strength=1.2),dict(id='food-2',kind='food',p=[-11,.7,8],strength=1.),dict(id='hazard-1',kind='hazard',p=[5,.7,9],strength=1.1)],
        obstacles=[dict(id='rock-1',p=[7,1.2,-3],r=1.2),dict(id='rock-2',p=[-10,1.,2],r=1.)])

def validate_world(w):
    if not isinstance(w,dict) or w.get('bounds')!=[24,16,18]: raise ValueError('B arena bounds are fixed at [24,16,18] mm')
    if type(w.get('cueOn')) is not bool or type(w.get('foodOn')) is not bool or w.get('cueAngle')!=0: raise ValueError('Invalid world switches/landmark angle')
    if not isinstance(w.get('sources'),list) or len(w['sources'])>24 or not isinstance(w.get('obstacles'),list) or len(w['obstacles'])>12: raise ValueError('Environment object limit')
    ids=set()
    for o in w['sources']+w['obstacles']:
        if not isinstance(o,dict) or not isinstance(o.get('id'),str) or not 1<=len(o['id'])<=120 or o['id'] in ids: raise ValueError('Invalid/duplicate world ID')
        ids.add(o['id']); p=o.get('p')
        if not isinstance(p,list) or len(p)!=3: raise ValueError('Invalid position')
        number(p[0],'x',-23,23);number(p[1],'height',0,14);number(p[2],'z',-17,17)
        if o in w['sources']:
            if o.get('kind') not in ('food','hazard'): raise ValueError('Invalid source kind')
            number(o.get('strength'),'strength',0,5)
        else: number(o.get('r'),'radius',.2,3)
    return w

def smell(p,world,kind):
    if kind=='food' and not world['foodOn']: return 0.
    return clamp(sum(o['strength']*math.exp(-float(np.sum((np.asarray(p)-o['p'])**2))/60) for o in world['sources'] if o['kind']==kind))

class SensorAdapter:
    def __init__(self,seed): self.rng=RNG(seed^0x51a29);self.previousOdor=0.; self.last=None
    def observe(self,body,world,dt,config):
        p,R,vel=body.pose(); head=p+R@np.array([.65,0,.12]); left=head+R@np.array([0,.3,0]);right=head+R@np.array([0,-.3,0])
        odor=[smell(to_ui(left),world,'food'),smell(to_ui(right),world,'food')]; mean=sum(odor)/2
        derivative=clamp((mean-self.previousOdor)/dt,-2,2); self.previousOdor=mean
        ranges=[body.ray(head,R@np.array([math.cos((k-4)*math.pi/8),-math.sin((k-4)*math.pi/8),0.]),10) for k in range(9)]
        panorama=body.visual_panorama(head,R)
        # Same RNG draw count with the cue on and off for paired comparisons.
        panorama=[clamp(v+self.rng.signed()*.004) if world['cueOn'] else (self.rng.signed()*0.) for v in panorama]
        self.last=dict(schema='flylab.sensors.v2',panorama=panorama,nearRanges=ranges,odor=odor,odorChange=derivative,
            danger=smell(to_ui(head),world,'hazard'),angularVelocity=-float(vel[2])+config['sensorNoise']*self.rng.signed(),
            forwardSpeed=float(vel[3:]@R[:,0]),clearanceDown=body.ray(p,[0,0,-1],16),clearanceUp=body.ray(p,[0,0,1],16),contact=int(body.nonfoot_contact()))
        return clone(self.last)
    def snapshot(self): return dict(rng=self.rng.state,previousOdor=self.previousOdor,last=clone(self.last))
    def restore(self,s):
        self.rng.state=integer(s['rng'],'sensor RNG',1,2**32-1); self.previousOdor=number(s['previousOdor'],'odor',0,1);self.last=clone(s['last'])
