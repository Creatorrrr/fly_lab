"""Opt-in heterogeneous LIF experiments on CPU/MPS, independent of baseline C.

Explicit cell parameters, edge delays/receptor effects, electrical coupling and
reward-gated STDP. Parameters require a declared evidence/assumption record.
This is a model laboratory, not a claim that missing physiology was recovered.
"""
import copy
import math
import numpy as np
from .neural import LIFParameters
from .integrity import digest, finite, bounded_int
from .inputs import validate_input
from .neural_state import validate_neural_ranges

class ResearchLIF:
    def __init__(self,graph,spec,device='cpu'):
        import torch
        if device not in ('cpu','mps'):raise ValueError('Research device must be cpu or mps')
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('BLOCKED_MPS')
        if spec.get('schema')!='flylab.physiology.v1' or spec.get('graph_hash')!=graph.hash:raise ValueError('Physiology graph identity mismatch')
        if not spec.get('evidence') or not spec.get('uncertainty'):raise ValueError('Physiology evidence and uncertainty required')
        if set(spec)-{'schema','graph_hash','evidence','uncertainty','parameters','cells','synapses','gap_junctions','plasticity','name'}:raise ValueError('Unknown physiology field')
        self.torch=torch;self.device=device;self.graph=graph;self.spec=copy.deepcopy(spec);self.hash=digest(spec)
        self.p=LIFParameters(**spec.get('parameters',{}));self.tick=0;self.backend='research_lif_'+device;self.n=graph.n
        if self.p.dtype!='float32':raise ValueError('Research CPU/MPS parity requires float32')
        params={k:np.full(graph.n,getattr(self.p,k),np.float64) for k in ('rest_mV','reset_mV','threshold_mV','tau_m_s','tau_syn_s')}
        for row in spec.get('cells',[]):
            ids=graph.resolve(row['ids'],maximum=10000)
            if not row.get('evidence'):raise ValueError('Cell override evidence required')
            for key,value in row['parameters'].items():
                if key not in params:raise ValueError('Unsupported heterogeneous parameter')
                params[key][ids]=finite(value,key,1e-5 if key.endswith('_s') else -200.,1. if key.endswith('_s') else 200.)
        if np.any(params['threshold_mV']<=np.maximum(params['rest_mV'],params['reset_mV'])):raise ValueError('Threshold must exceed rest/reset')
        pre=graph.indices.copy();post=np.repeat(np.arange(graph.n),np.diff(graph.indptr));weights=graph.weights.copy()
        delays=np.full(len(weights),round(self.p.delay_s/self.p.dt),np.int32)
        def edge_index(row):
            a,b=graph.resolve([row['pre'],row['post']]);start,end=graph.indptr[b:b+2]
            offset=np.searchsorted(graph.indices[start:end],a)
            if offset>=end-start or graph.indices[start+offset]!=a:raise ValueError('Override must reference an existing chemical edge')
            return int(start+offset)
        for row in spec.get('synapses',[]):
            edge=edge_index(row)
            if not row.get('evidence') or not row.get('receptor'):raise ValueError('Explicit receptor/effect evidence required')
            weights[edge]=finite(row['weight_mV'],'synaptic effect',-1000.,1000.)
            delay=finite(row.get('delay_s',self.p.delay_s),'edge delay',self.p.dt,.05)/self.p.dt
            if abs(delay-round(delay))>1e-8:raise ValueError('Integral edge delay required')
            delays[edge]=round(delay)
        self.slots=int(delays.max(initial=1))+1
        self.refractory_ticks=round(self.p.refractory_s/self.p.dt)
        self.max_tick=min(2**31-10000,2**31-1-max(self.refractory_ticks,self.slots))
        if self.slots*graph.n*4>2*1024**3:raise ValueError('Research delay queue exceeds 2 GiB budget')
        def tensor(a,dtype=torch.float32):return torch.as_tensor(np.asarray(a).copy(),dtype=dtype,device=device)
        self.tensor=tensor
        self.params={k:tensor(v) for k,v in params.items()}
        self.v=self.params['rest_mV'].clone();self.h=torch.zeros_like(self.v);self.rate=torch.zeros_like(self.v)
        self.queue=torch.zeros((self.slots,graph.n),dtype=torch.float32,device=device)
        # MPS int32 indices/clocks are explicit; supported experiments <2^31 ticks.
        self.pre=tensor(pre,torch.int32);self.post=tensor(post,torch.int32);self.weights=tensor(weights)
        self.delays=tensor(delays,torch.int32);self.refractory=tensor(np.zeros(graph.n),torch.int32)
        self.spike_count=tensor(np.zeros(graph.n),torch.int32)
        self.suppress=tensor(np.zeros(graph.n,bool),torch.bool);self.mute=self.suppress.clone()
        self.edge_enabled=tensor(np.ones(len(weights),bool),torch.bool);self.edge_mutes=[]
        tm,ts=params['tau_m_s'],params['tau_syn_s']
        em=np.exp(-self.p.dt/tm);es=np.exp(-self.p.dt/ts)
        # Evaluate once in float64. expm1 avoids cancellation near tau_m=tau_syn.
        gap=np.abs(self.p.dt*(1/tm-1/ts));ratio=np.ones_like(gap)
        np.divide(-np.expm1(-gap),gap,out=ratio,where=gap!=0)
        self.em=tensor(em);self.es=tensor(es);self.factor=tensor(self.p.dt/tm*np.maximum(em,es)*ratio)
        self.er=math.exp(-self.p.dt/self.p.rate_tau_s)
        gaps=spec.get('gap_junctions',[]);ga=[];gb=[];gc=[]
        for row in gaps:
            if not row.get('evidence'):raise ValueError('Electrical coupling evidence required')
            a,b=graph.resolve([row['a'],row['b']]);ga.append(a);gb.append(b)
            gc.append(finite(row['coupling'],'dimensionless gap coupling',0.,1.))
        self.ga=tensor(ga,torch.int32);self.gb=tensor(gb,torch.int32);self.gc=tensor(gc)
        pl=spec.get('plasticity');self.plasticity=pl
        if pl:
            if pl.get('rule')!='reward_stdp_v1' or not pl.get('evidence'):raise ValueError('Explicit plasticity rule/evidence required')
            self.eta=finite(pl['eta'],'STDP eta',0.,.1);self.tau=finite(pl['tau_s'],'STDP trace tau',self.p.dt,1.)
            selected=[edge_index(row) for row in pl['edges']]
            if len(set(selected))!=len(selected) or not 1<=len(selected)<=100000:raise ValueError('1..100000 unique plastic edges required')
            self.pe=tensor(selected,torch.int32);self.pp=self.pre[self.pe];self.pq=self.post[self.pe]
            self.original_sign=torch.sign(self.weights[self.pe]);self.maximum=finite(pl['max_abs_mV'],'plastic weight bound',0.,1000.)
            if any(weights[i]==0 or abs(weights[i])>self.maximum for i in selected):raise ValueError('Plastic edge sign and bound must already be defined')
            self.pre_trace=torch.zeros(len(selected),device=device);self.post_trace=torch.zeros(len(selected),device=device)
        self.last_events=[]

    def set_interventions(self,suppress=(),mute=(),edges=()):
        suppress,mute,edges=list(suppress),list(mute),sorted(set(edges))
        for ids in (suppress,mute):
            if any(type(i) is not int or not 0<=i<self.n for i in ids):raise ValueError('Invalid intervention neuron index')
        if any(type(i) is not int or not 0<=i<len(self.graph.weights) for i in edges):raise ValueError('Invalid intervention edge index')
        masks=[]
        for ids in (suppress,mute):
            mask=np.zeros(self.n,bool);mask[ids]=True;masks.append(self.tensor(mask,self.torch.bool))
        enabled=np.ones(len(self.graph.weights),bool);enabled[edges]=False
        edge_enabled=self.tensor(enabled,self.torch.bool)
        self.suppress,self.mute=masks;self.edge_enabled=edge_enabled;self.edge_mutes=edges

    def advance(self,drive,steps,capture=(),pulses=None,*,reward=0.):
        drive,capture=validate_input(self.graph.n,drive,steps,capture,pulses)
        reward=finite(reward,'external scalar reward',-1.,1.)
        bounded_int(self.tick+steps,'research clock',0,self.max_tick)
        t=self.torch;x=self.tensor(drive);selected=self.tensor(capture,t.int32);events=[];start=self.tick
        voltage_events=self.p.integration=='exact-exponential-voltage-events-v1'
        reset_current=self.p.integration in ('exact-exponential-reset-current-v1','exact-exponential-voltage-events-v1')
        if pulses is not None:pi,pv=self.tensor(pulses[0],t.int32),self.tensor(pulses[1])
        for k in range(steps):
            self.h+=self.queue[self.tick%self.slots];self.queue[self.tick%self.slots].zero_()
            if pulses is not None:(self.v if voltage_events else self.h).index_add_(0,pi,pv[k])
            electrical=t.zeros_like(self.v)
            if len(self.ga):
                current=self.gc*(self.v[self.gb]-self.v[self.ga])
                electrical.index_add_(0,self.ga,current);electrical.index_add_(0,self.gb,-current)
            eligible=self.tick>=self.refractory
            if voltage_events and pulses is not None:eligible[pi]=True
            rest=self.params['rest_mV'];reset=self.params['reset_mV'];held=x+electrical
            v=rest+held+(self.v-rest-held)*self.em+self.h*self.factor
            self.h=t.where(eligible,self.h*self.es,self.h) if voltage_events else self.h*self.es
            self.v=t.where(eligible,v,reset)
            crossing=self.v>self.params['threshold_mV'] if voltage_events else self.v>=self.params['threshold_mV']
            spike=eligible&crossing&~self.suppress;self.tick+=1
            targets=((self.tick+self.delays)%self.slots)*self.graph.n+self.post
            emitted=spike&~self.mute
            self.queue.view(-1).index_add_(0,targets,self.weights*emitted[self.pre]*self.edge_enabled)
            if self.plasticity:
                self.pre_trace*=math.exp(-self.p.dt/self.tau);self.post_trace*=math.exp(-self.p.dt/self.tau)
                # Earlier pre + current post potentiates; reversed order depresses.
                delta=self.eta*reward*(self.pre_trace*spike[self.pq]-self.post_trace*spike[self.pp])
                magnitude=t.clamp(t.abs(self.weights[self.pe])+delta,0.,self.maximum)
                self.weights[self.pe]=self.original_sign*magnitude
                self.pre_trace+=spike[self.pp];self.post_trace+=spike[self.pq]
            self.v=t.where(spike,reset,self.v)
            if reset_current:self.h=t.where(spike,0.,self.h)
            self.refractory=t.where(spike,self.tick+self.refractory_ticks,self.refractory)
            self.rate*=self.er;self.rate+=spike*((1-self.er)/self.p.dt);self.spike_count+=spike
            if len(capture):events.append(spike[selected].clone())
        if not all(bool(t.isfinite(a).all().cpu()) for a in (self.v,self.h,self.queue)):raise RuntimeError('FAULT_RESEARCH_NEURAL_NONFINITE')
        self.last_events=[]
        if events:
            times,cells=np.nonzero(t.stack(events).cpu().numpy())
            self.last_events=[dict(tick=start+int(k)+1,index=int(capture[j])) for k,j in zip(times,cells)]

    def readout(self,indices):
        i=self.tensor(indices,self.torch.int32)
        return {k:getattr(self,field)[i].detach().cpu().numpy().copy() for k,field in
                (('voltage_mV','v'),('rate_Hz','rate'),('spike_count','spike_count'))}

    def snapshot(self):
        fields=['v','h','rate','queue','refractory','spike_count','weights','suppress','mute']
        if self.plasticity:fields+=['pre_trace','post_trace']
        return dict(schema='flylab.research-neural-state.v2',graph_hash=self.graph.hash,profile_hash=self.hash,tick=self.tick,
                    parameter_hash=self.p.hash,edge_mutes=list(self.edge_mutes),
                    arrays={k:getattr(self,k).detach().cpu().numpy().copy() for k in fields})

    def restore(self,state):
        if state.get('schema')=='flylab.research-neural-state.v1':
            raise ValueError('Legacy research v1 requires its archived runtime; integration semantics were corrected in v2')
        if state.get('schema')!='flylab.research-neural-state.v2' or state.get('graph_hash')!=self.graph.hash or state.get('profile_hash')!=self.hash or state.get('parameter_hash')!=self.p.hash:raise ValueError('Research checkpoint identity mismatch')
        tick=bounded_int(state['tick'],'research tick',0,self.max_tick);current=self.snapshot()['arrays'];arrays=state['arrays']
        if set(arrays)!=set(current):raise ValueError('Research state field mismatch')
        for k,v in arrays.items():
            if not isinstance(v,np.ndarray) or v.shape!=current[k].shape or v.dtype!=current[k].dtype or not np.isfinite(v).all():raise ValueError('Research array mismatch: '+k)
        validate_neural_ranges(tick,self.p,**{k:arrays[k] for k in ('v','h','queue','rate','spike_count')},refractory_until=arrays['refractory'])
        edges=state.get('edge_mutes')
        if not isinstance(edges,list) or any(type(i) is not int or not 0<=i<len(self.graph.weights) for i in edges) or edges!=sorted(set(edges)):
            raise ValueError('Invalid research edge mask')
        if not self.plasticity and not np.array_equal(arrays['weights'],current['weights']):raise ValueError('Nonplastic weights changed')
        if self.plasticity:
            indices=self.pe.cpu().numpy();others=np.ones(len(current['weights']),bool);others[indices]=False
            if not np.array_equal(arrays['weights'][others],current['weights'][others]) or np.any(np.abs(arrays['weights'][indices])>self.maximum):raise ValueError('Plastic weight bounds/selection mismatch')
            signs=self.original_sign.cpu().numpy()
            if np.any(arrays['weights'][indices]*signs<0):raise ValueError('Plasticity cannot flip transmitter sign')
            bound=-math.expm1(-tick*self.p.dt/self.tau)/-math.expm1(-self.p.dt/self.tau)
            for key in ('pre_trace','post_trace'):
                a=arrays[key]
                if np.any(a<0) or (tick==0 and np.any(a!=0)) or np.any(a>bound+max(1e-5,bound*1e-3)):
                    raise ValueError('Plastic trace outside reachable range: '+key)
        # Allocation and host->device copies finish before any live state changes.
        staged={k:self.tensor(v,getattr(self,k).dtype) for k,v in arrays.items()}
        enabled=np.ones(len(self.graph.weights),bool);enabled[edges]=False
        edge_enabled=self.tensor(enabled,self.torch.bool)
        if self.device=='mps':self.torch.mps.synchronize()
        for k,v in staged.items():setattr(self,k,v)
        self.edge_enabled=edge_enabled;self.edge_mutes=list(edges);self.tick=tick;self.last_events=[]

    def summary(self):
        return dict(tick=self.tick,simulated_node_count=self.n,
                    cumulative_spikes=int(self.spike_count.detach().cpu().numpy().sum(dtype=np.int64)),
                    mean_rate_Hz=float(self.rate.mean().cpu()),min_voltage_mV=float(self.v.min().cpu()),
                    max_voltage_mV=float(self.v.max().cpu()))

    def region_summary(self,groups):
        return [dict(region=name,neurons=len(ids),mean_rate_Hz=float(self.rate[self.tensor(ids,self.torch.int32)].mean().cpu()))
                for name,ids in groups.items()]
