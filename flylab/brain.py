"""Python port of A's disclosed hybrid rate controller, not a brain emulation.
The connectome is not changed by adding a physical body. The 72 engineered
nodes are still engineered; no actual PFL3 identifiers are invented.
"""
from __future__ import annotations
import math, re
import numpy as np
from .common import TAU, RNG, clamp, wrap, clone, integer, number

def validate_graph(g):
    if not isinstance(g,dict) or g.get('schema')!='flylab.connectome.v1': raise ValueError('Unsupported connectome schema')
    ns,es=g.get('nodes'),g.get('edges')
    if not isinstance(ns,list) or not 1<=len(ns)<=440 or not isinstance(es,list) or len(es)>100000: raise ValueError('B hybrid backend: 1..440 real IDs, <=100000 edges')
    ids=set()
    for n in ns:
        if not isinstance(n,dict): raise ValueError('Invalid node')
        id=n.get('id'); bid=n.get('bodyId')
        if not isinstance(id,str) or not re.fullmatch(r'hemibrain:\d+',id) or not isinstance(bid,str) or id!='hemibrain:'+bid or id in ids: raise ValueError('Invalid/duplicate ID')
        if n.get('type') not in ('EPG','PENa','PENb','PEG') or n.get('origin')!='anatomical' or n.get('hemisphere') not in ('L','R'): raise ValueError('Unsupported node metadata')
        if not isinstance(n.get('name'),str) or len(n['name'])>200: raise ValueError('Invalid name')
        integer(n.get('indexFix'),'indexFix',1,9); integer(n.get('pbIndex'),'pbIndex',1,9); ids.add(id)
    if not any(n['type']=='EPG' for n in ns): raise ValueError('EPG required by hybrid controller')
    seen=set()
    for e in es:
        if not isinstance(e,dict) or e.get('pre') not in ids or e.get('post') not in ids: raise ValueError('Unknown edge endpoint')
        number(e.get('count'),'count',1e-9,1e9); k=(e['pre'],e['post'])
        if k in seen: raise ValueError('Aggregate ROI rows before import')
        seen.add(k)
    return g

class Circuit:
    def __init__(self,g):
        self.data=clone(validate_graph(g)); self.nodes=clone(g['nodes']); self.real_count=len(self.nodes)
        for n in self.nodes:
            n['angle']=((n['indexFix']-1)%8)*TAU/8
            n['displayName']=f"{n['type']} {n['hemisphere']}{n['pbIndex']} · {n['bodyId'][-4:]}"
        for typ,cnt in [('HD',16),('GOAL',16),('PFL3-L',16),('PFL3-R',16),('DN',2),('SENSE',4),('ALT',2)]:
            for k in range(cnt):
                self.nodes.append(dict(id=f'model:{typ}:{k}',name=f'{typ} {k+1:02d}',displayName=f'{typ} {k+1:02d}',type=typ,origin='model',angle=k/cnt*TAU))
        self.index={n['id']:i for i,n in enumerate(self.nodes)}; self.N=len(self.nodes); self.pop={}
        for i,n in enumerate(self.nodes): self.pop.setdefault(n['type'],[]).append(i)
        # Sparse COO accumulation (no NxN dense matrix); C can replace this backend.
        self.pre=np.array([self.index[e['pre']] for e in g['edges']],dtype=np.int32)
        self.post=np.array([self.index[e['post']] for e in g['edges']],dtype=np.int32)
        weights=np.sqrt([e['count'] for e in g['edges']]); sums=np.bincount(self.post,weights=weights,minlength=self.N)
        self.weight=weights/np.maximum(sums[self.post],1e-30)
    def propagate(self,a,gain=1.): return np.bincount(self.post,weights=self.weight*a[self.pre]*gain,minlength=self.N)
    def catalog(self): return [dict(n,index=i,signal=dict(kind='activation',unit='1',range=[0,1])) for i,n in enumerate(self.nodes)]

class NeuralController:
    def __init__(self,circuit,seed):
        self.circuit=circuit; self.rng=RNG(seed^0x7cab1); self.a=np.full(circuit.N,.05); self.adapt=np.zeros(circuit.N)
        self.lesions=set(); self.stim={}; self.phaseEstimate=0.; self.heading=0.; self.confidence=0.; self.goal=1.
        self.clock=0.; self.nextGoal=9.; self.avoidSide=1; self.prevContact=0; self.odorMode='explore'; self.turnMemory=0.; self.contactTime=0.; self.recoveryTime=0.
        self.output=dict(yawRate=0.,forwardSpeed=0.,verticalSpeed=0.); self.lastCue=None; self.error=0.; self.recNorm=0.; self.headingBins=[0.]*8
    def val(self,typ,k=0): return float(self.a[self.circuit.pop[typ][k]])
    def mean(self,typ): return float(np.mean(self.a[self.circuit.pop[typ]]))
    def target(self,i,value,dt,tau=.13):
        if i in self.lesions: self.a[i]=0; return
        extra=self.stim.get(i,{}).get('amplitude',0)
        self.a[i]+=(clamp(value+extra)-self.a[i])*(1-math.exp(-dt/tau))
    def step(self,s,dt,config):
        # No world, XYZ or ground-truth yaw argument crosses this boundary.
        self.clock+=dt; c=self.circuit; old=self.a.copy(); cx=cy=cs=0.
        for i,x in enumerate(s['panorama']):
            p=i/64*TAU-math.pi; cx+=x*math.cos(p); cy+=x*math.sin(p); cs+=x
        visible=cs>.8; self.lastCue=wrap(-math.atan2(cy,cx)) if visible else None
        self.phaseEstimate=wrap(self.phaseEstimate+s['angularVelocity']*dt)
        if visible: self.phaseEstimate=wrap(self.phaseEstimate+wrap(self.lastCue-self.phaseEstimate)*(1-math.exp(-dt/.24)))
        for i in c.pop['HD']: self.target(i,.05+.90*math.exp(3.4*(math.cos(c.nodes[i]['angle']-self.phaseEstimate)-1)),dt,.07)
        hx=hy=0.
        for i in c.pop['HD']:
            v=max(0,self.a[i]-.04); hx+=v*math.cos(c.nodes[i]['angle']); hy+=v*math.sin(c.nodes[i]['angle'])
        scaffold=math.atan2(hy,hx); rec=c.propagate(old,config['graphGain']); self.recNorm=float(rec.sum()/c.real_count)
        for i in range(c.real_count):
            n=c.nodes[i]; bump=math.exp(3*(math.cos(n['angle']-scaffold)-1)); omega=clamp(s['angularVelocity']/2,-1,1)
            target=.055+.77*bump
            if n['type'] in ('PENa','PENb'): target=.07+.58*bump+.18*max(0,omega*(1 if n['hemisphere']=='R' else -1))
            if n['type']=='PEG': target=.07+.66*bump
            target+=.30*rec[i]-.045; self.adapt[i]+=(old[i]-self.adapt[i])*dt/2; target-=.025*self.adapt[i]
            self.target(i,target,dt,.28 if n['type']=='PEG' else .11)
        bins=np.zeros(8); counts=np.zeros(8)
        for i in c.pop['EPG']:
            k=(c.nodes[i]['indexFix']-1)%8; bins[k]+=max(0,self.a[i]-.025); counts[k]+=1
        bins/=np.maximum(1,counts); angles=np.arange(8)*TAU/8
        ex=float(bins@np.cos(angles)); ey=float(bins@np.sin(angles)); total=float(bins.sum())
        self.confidence=math.hypot(ex,ey)/total if total>.01 else 0.
        if total>.01: self.heading=math.atan2(ey,ex)
        self.headingBins=bins.tolist()
        if not visible and self.confidence>.1: self.phaseEstimate=wrap(self.phaseEstimate+wrap(self.heading-self.phaseEstimate)*dt*.08)
        ranges=s['nearRanges']; dl=max(math.exp(-r/2.2) for r in ranges[:4]); dr=max(math.exp(-r/2.2) for r in ranges[5:]); front=math.exp(-ranges[4]/2.2)
        # Engineered avoidance memory: preserve a turn until proprioceptive
        # rotation and frontal clearance agree that it is safe to walk again.
        # No world coordinates, true yaw, or direct motor bypass enter this path.
        clearance=min(ranges[3:6])
        if self.turnMemory:
            direction=1 if self.turnMemory>0 else -1
            remaining=clamp(abs(self.turnMemory)-direction*s['angularVelocity']*dt,.001,math.tau)
            self.turnMemory=direction*remaining
            if remaining<=.05 and clearance>4.5 and not s['contact']: self.turnMemory=0.
        elif clearance<3.5 or s['contact']:
            if abs(dl-dr)>.08: self.avoidSide=1 if dl>dr else -1
            self.turnMemory=self.avoidSide*1.3
        self.prevContact=s['contact']
        avoiding=bool(self.turnMemory)
        self.recoveryTime=max(0.,self.recoveryTime-dt)
        self.contactTime=clamp(self.contactTime+(dt if s['contact'] and avoiding else -dt*.5),0.,1.)
        if avoiding and not self.recoveryTime and self.contactTime>.3:
            self.recoveryTime=.4
            self.contactTime=0.

        for i,v in enumerate([dl,dr,*s['odor']]): self.target(c.pop['SENSE'][i],v,dt,.08 if i<2 else .15)
        if config['task']=='heading': self.goal=config['goalAngle']; self.odorMode='heading'
        else:
            if self.clock>self.nextGoal:
                self.goal=wrap(self.goal+(.5+self.rng.next())*self.rng.signed()*2); self.nextGoal=self.clock+7+self.rng.next()*7
            if max(s['odor'])>.10:
                self.goal=wrap(self.heading+clamp((self.val('SENSE',3)-self.val('SENSE',2))*9,-1,1)+(.50 if s['odorChange']<-.02 else 0)); self.odorMode='chemotaxis'
            else: self.odorMode='explore'
            if max(s['odor'])>.88: self.goal=wrap(self.heading+.7); self.odorMode='dwell'
        for i in c.pop['GOAL']: self.target(i,.02+.94*math.exp(3.3*(math.cos(c.nodes[i]['angle']-self.goal)-1)),dt,.2)
        gx=sum(self.a[i]*math.cos(c.nodes[i]['angle']) for i in c.pop['GOAL']); gy=sum(self.a[i]*math.sin(c.nodes[i]['angle']) for i in c.pop['GOAL'])
        self.error=wrap(math.atan2(gy,gx)-self.heading)
        avoid=clamp(self.val('SENSE',0)-self.val('SENSE',1)+max(0,s['danger']-.25)*self.avoidSide,-1,1)
        if avoiding: avoid=.85*(1 if self.turnMemory>0 else -1)
        for k in range(16):
            att=1. if avoiding else .30+.70*self.a[c.pop['GOAL'][k]]
            left=.045+att*(.68*(not avoiding)*max(0,-math.sin(self.error))+.42*max(0,-avoid)*3+.22*s['contact']*(self.avoidSide<0))
            right=.045+att*(.68*(not avoiding)*max(0,math.sin(self.error))+.42*max(0,avoid)*3+.22*s['contact']*(self.avoidSide>0))
            self.target(c.pop['PFL3-L'][k],left,dt,.10); self.target(c.pop['PFL3-R'][k],right,dt,.10)
        self.target(c.pop['DN'][0],clamp(self.mean('PFL3-L')*2.6),dt,.09); self.target(c.pop['DN'][1],clamp(self.mean('PFL3-R')*2.6),dt,.09)
        he=config['altitude']-s['clearanceDown']
        self.target(c.pop['ALT'][0],clamp(.5+he*.2),dt,.25); self.target(c.pop['ALT'][1],clamp(.5-he*.2),dt,.25)
        speed=(3.3 if config['mode']=='walk' else 5.3)*clamp(self.mean('EPG')*4)*(1-clamp(front*.72,0,.8))
        if self.odorMode=='dwell': speed*=.4
        if s['danger']>.3: speed*=1.15
        if avoiding: speed=-2.5 if self.recoveryTime else 0.  # back away, then pivot via CPG
        coupled=config['motorCoupled']
        self.output=dict(yawRate=clamp(3.5*(self.val('DN',1)-self.val('DN',0)),-3,3) if coupled else 0.,forwardSpeed=float(speed) if coupled else 0.,verticalSpeed=2.7*(self.val('ALT',0)-self.val('ALT',1)) if coupled and config['mode']=='flight' else 0.)
        for i in list(self.stim):
            self.stim[i]['remaining']-=dt
            if self.stim[i]['remaining']<=1e-10: del self.stim[i]
        return dict(self.output)
    def intervene(self,ids,kind,amplitude=.65,duration=2):
        for id in ids:
            i=self.circuit.index[id]
            if kind=='stimulate': self.stim[i]=dict(amplitude=amplitude,remaining=duration)
            elif kind=='suppress': self.lesions.add(i); self.a[i]=0
            elif kind=='release': self.lesions.discard(i)
    def snapshot(self):
        scalar=['phaseEstimate','heading','confidence','goal','clock','nextGoal','avoidSide','prevContact','odorMode','turnMemory','contactTime','recoveryTime','output','lastCue','error','recNorm','headingBins']
        return dict(a=self.a.tolist(),adapt=self.adapt.tolist(),lesions=sorted(self.lesions),stim=[[i,clone(x)] for i,x in self.stim.items()],rng=self.rng.state,**{k:clone(getattr(self,k)) for k in scalar})
    def restore(self,s):
        """Validate the entire state before committing it (atomic on rejection).

        fix_2 changes validation only, not the forward dynamics or checkpoint
        schema. Valid v0.2.1 snapshots, including active recovery, remain valid.
        """
        if not isinstance(s,dict) or set(s)!=set(self.snapshot()):
            raise ValueError('Unknown neural state fields')
        if s['odorMode'] not in ('explore','heading','chemotaxis','dwell'):
            raise ValueError('Invalid neural mode')
        if not isinstance(s['output'],dict) or set(s['output'])!={'yawRate','forwardSpeed','verticalSpeed'}:
            raise ValueError('Invalid motor command state')
        for v in s['output'].values(): number(v,'motor',-20,20)
        if s['lastCue'] is not None: number(s['lastCue'],'lastCue',-math.pi,math.pi)
        if not isinstance(s['headingBins'],list) or len(s['headingBins'])!=8:
            raise ValueError('Invalid heading bins')
        for v in s['headingBins']: number(v,'headingBin',0,1)
        arrays={}
        for k in ('a','adapt'):
            if not isinstance(s[k],list): raise ValueError('Neural array must be a list')
            for v in s[k]: number(v,k,0,1)
            arr=np.asarray(s[k],dtype=float)
            if arr.shape!=(self.circuit.N,) or not np.isfinite(arr).all() or (arr<0).any() or (arr>1).any():
                raise ValueError('Invalid neural state')
            arrays[k]=arr.copy()
        if not isinstance(s['lesions'],list) or not isinstance(s['stim'],list):
            raise ValueError('Invalid intervention state')
        lesions=[]
        for i in s['lesions']: lesions.append(integer(i,'lesion',0,self.circuit.N-1))
        if len(set(lesions))!=len(lesions): raise ValueError('Duplicate lesion')
        stimuli={}
        for pair in s['stim']:
            if not isinstance(pair,list) or len(pair)!=2: raise ValueError('Invalid stimulus pair')
            i,v=pair; integer(i,'stim',0,self.circuit.N-1)
            if i in stimuli: raise ValueError('Duplicate stimulus')
            if not isinstance(v,dict) or set(v)!={'amplitude','remaining'}:
                raise ValueError('Invalid stimulus state')
            number(v['amplitude'],'amplitude',-1,1)
            number(v['remaining'],'remaining',1e-12,60)
            stimuli[i]=clone(v)
        rng=integer(s['rng'],'rng',1,2**32-1)
        # State domains follow step(); a tiny time tolerance accommodates JSON
        # floating-point round trips, not seconds-long extra recovery periods.
        for k,lo,hi in (
            ('phaseEstimate',-math.pi,math.pi),('heading',-math.pi,math.pi),
            ('confidence',0,1+1e-12),('goal',-math.pi,math.pi),
            ('clock',0,1e12),('nextGoal',0,1e12),
            ('turnMemory',-TAU,TAU),('contactTime',0,1),
            ('recoveryTime',0,.4+1e-12),('error',-math.pi,math.pi),('recNorm',0,2+1e-12)):
            number(s[k],k,lo,hi)
        number(s['avoidSide'],'avoidSide',-1,1)
        if s['avoidSide'] not in (-1,1): raise ValueError('avoidSide must be -1 or 1')
        number(s['prevContact'],'prevContact',0,1)
        if s['prevContact'] not in (0,1): raise ValueError('prevContact must be 0 or 1')
        # No self mutation above this point, including RNG and activity buffers.
        self.a=arrays['a']; self.adapt=arrays['adapt']; self.lesions=set(lesions)
        self.stim=stimuli; self.rng.state=rng
        for k,v in s.items():
            if k not in ('a','adapt','lesions','stim','rng'): setattr(self,k,clone(v))
