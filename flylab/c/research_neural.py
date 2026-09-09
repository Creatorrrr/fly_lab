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

class ResearchLIF:
    def __init__(self,graph,spec,device='cpu'):
        import torch
        if device not in ('cpu','mps'):raise ValueError('Research device must be cpu or mps')
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('BLOCKED_MPS')
        if spec.get('schema')!='flylab.physiology.v1' or spec.get('graph_hash')!=graph.hash:raise ValueError('Physiology graph identity mismatch')
        if not spec.get('evidence') or not spec.get('uncertainty'):raise ValueError('Physiology evidence and uncertainty required')
        if set(spec)-{'schema','graph_hash','evidence','uncertainty','parameters','cells','synapses','gap_junctions','plasticity','name'}:raise ValueError('Unknown physiology field')
        self.torch=torch;self.device=device;self.graph=graph;self.spec=copy.deepcopy(spec);self.hash=digest(spec)
        self.p=LIFParameters(**spec.get('parameters',{}));self.tick=0;self.backend='research_lif_'+device
        if self.p.dtype!='float32':raise ValueError('Research CPU/MPS parity requires float32')
        params={k:np.full(graph.n,getattr(self.p,k),np.float32) for k in ('rest_mV','reset_mV','threshold_mV','tau_m_s','tau_syn_s')}
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
        tm,ts=self.params['tau_m_s'],self.params['tau_syn_s']
        self.em=torch.exp(-self.p.dt/tm);self.es=torch.exp(-self.p.dt/ts)
        self.factor=torch.where(tm==ts,self.p.dt/tm*self.em,ts/torch.where(tm==ts,torch.ones_like(tm),tm-ts)*(self.em-self.es))
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

    def advance(self,drive,steps,capture=(),reward=0.):
        drive,capture=validate_input(self.graph.n,drive,steps,capture)
        reward=finite(reward,'external scalar reward',-1.,1.)
        bounded_int(self.tick+steps,'research clock',0,2**31-10000)
        t=self.torch;x=self.tensor(drive);selected=self.tensor(capture,t.int32);events=[];start=self.tick
        for _ in range(steps):
            self.h+=self.queue[self.tick%self.slots];self.queue[self.tick%self.slots].zero_()
            electrical=t.zeros_like(self.v)
            if len(self.ga):
                current=self.gc*(self.v[self.gb]-self.v[self.ga])
                electrical.index_add_(0,self.ga,current);electrical.index_add_(0,self.gb,-current)
            eligible=self.tick>=self.refractory
            rest=self.params['rest_mV'];reset=self.params['reset_mV'];held=x+electrical
            v=rest+held+(self.v-rest-held)*self.em+self.h*self.factor
            self.h*=self.es;self.v=t.where(eligible,v,reset)
            spike=eligible&(self.v>=self.params['threshold_mV']);self.tick+=1
            targets=((self.tick+self.delays)%self.slots)*self.graph.n+self.post
            self.queue.view(-1).index_add_(0,targets,self.weights*spike[self.pre])
            if self.plasticity:
                self.pre_trace*=math.exp(-self.p.dt/self.tau);self.post_trace*=math.exp(-self.p.dt/self.tau)
                # Earlier pre + current post potentiates; reversed order depresses.
                delta=self.eta*reward*(self.pre_trace*spike[self.pq]-self.post_trace*spike[self.pp])
                magnitude=t.clamp(t.abs(self.weights[self.pe])+delta,0.,self.maximum)
                self.weights[self.pe]=self.original_sign*magnitude
                self.pre_trace+=spike[self.pp];self.post_trace+=spike[self.pq]
            self.v=t.where(spike,reset,self.v);self.refractory=t.where(spike,self.tick+round(self.p.refractory_s/self.p.dt),self.refractory)
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
        fields=['v','h','rate','queue','refractory','spike_count','weights']
        if self.plasticity:fields+=['pre_trace','post_trace']
        return dict(schema='flylab.research-neural-state.v1',graph_hash=self.graph.hash,profile_hash=self.hash,tick=self.tick,
                    arrays={k:getattr(self,k).detach().cpu().numpy().copy() for k in fields})

    def restore(self,state):
        if state.get('schema')!='flylab.research-neural-state.v1' or state.get('graph_hash')!=self.graph.hash or state.get('profile_hash')!=self.hash:raise ValueError('Research checkpoint identity mismatch')
        tick=bounded_int(state['tick'],'research tick',0,2**31-10000);current=self.snapshot()['arrays'];arrays=state['arrays']
        if set(arrays)!=set(current):raise ValueError('Research state field mismatch')
        for k,v in arrays.items():
            if not isinstance(v,np.ndarray) or v.shape!=current[k].shape or v.dtype!=current[k].dtype or not np.isfinite(v).all():raise ValueError('Research array mismatch: '+k)
        if np.any(arrays['spike_count']<0) or np.any(arrays['rate']<0):raise ValueError('Negative spike state')
        if not self.plasticity and not np.array_equal(arrays['weights'],current['weights']):raise ValueError('Nonplastic weights changed')
        if self.plasticity:
            indices=self.pe.cpu().numpy();others=np.ones(len(current['weights']),bool);others[indices]=False
            if not np.array_equal(arrays['weights'][others],current['weights'][others]) or np.any(np.abs(arrays['weights'][indices])>self.maximum):raise ValueError('Plastic weight bounds/selection mismatch')
            signs=self.original_sign.cpu().numpy()
            if np.any(arrays['weights'][indices]*signs<0):raise ValueError('Plasticity cannot flip transmitter sign')
        for k,v in arrays.items():getattr(self,k).copy_(self.tensor(v,getattr(self,k).dtype))
        self.tick=tick;self.last_events=[]
