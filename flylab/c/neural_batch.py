"""Vectorized worlds using the same ordered CUDA LIF equations.

CSR topology is shared. Weights, interventions, delay queues and all dynamic
state remain owned by each world. Regular CudaLIF objects expose per-world
views for their existing readout/checkpoint contracts.
"""
from pathlib import Path
import hashlib
from types import SimpleNamespace
import numpy as np
from .inputs import validate_input


def batch_source():
    source=Path(__file__).with_name('lif.cu').read_text(encoding='utf-8')
    integrate,rest=source.split('extern "C" __global__ void propagate(',1)
    propagate='extern "C" __global__ void propagate('+rest.split('extern "C" __global__ void read_state(',1)[0]
    integrate=integrate.replace('int integration, double rate_jump)', 'int integration, double rate_jump, int total_steps)')
    integrate=integrate.replace('if (i >= n) return;', '''if (i >= n) return;
    long long world=blockIdx.y, offset=world*n;
    v+=offset; h+=offset; rate+=offset; refractory+=offset; counts+=offset;
    queue+=world*slots*n; suppress+=offset; mute+=offset; emitted+=offset;
    x+=offset; active_rows+=offset; clock+=world; health+=world;
    events+=world*total_steps*channels; pulse_ptr+=world*(n+1);
    pulses+=world*total_steps*pulse_width;''')
    propagate=propagate.replace('int n, int slots, int k)', 'int n, int slots, int k, int edges)')
    propagate=propagate.replace('if (i >= n) return;', '''if (i >= n) return;
    long long world=blockIdx.y;
    weight+=world*edges; emitted+=world*n; active_rows+=world*n;
    queue+=world*slots*n; clock+=world;''')
    return integrate+propagate


class CudaLIFBatch:
    def __init__(self,neurals):
        import cupy as cp
        if not neurals or any(n.backend!='exp_lif_cuda' for n in neurals):raise ValueError('CUDA LIF worlds required')
        first=neurals[0]
        if any(n.graph.hash!=first.graph.hash or n.p!=first.p or n._pending_advance is not None for n in neurals):
            raise ValueError('Batch requires identical topology and neural parameters at a control boundary')
        self.cp=cp;self.neurals=list(neurals);self.worlds=len(neurals);self.n=first.n;self.slots=first.slots
        self.stream=cp.cuda.Stream(non_blocking=True);self.arrays={}
        code=batch_source();self.shader=hashlib.sha256(code.encode()).hexdigest()
        self.module=cp.RawModule(code=code,options=('--std=c++17','--fmad=false'))
        self.integrate=self.module.get_function('integrate');self.propagate=self.module.get_function('propagate')
        for n in neurals:n.stream.synchronize()
        with self.stream:
            for key in ('v','h','rate','refractory_until','queue','spike_count','suppress','mute','emitted','active_rows'):
                self.arrays[key]=cp.stack([getattr(n,key) for n in neurals])
            self.weights=cp.stack([n.W.data for n in neurals])
        self.stream.synchronize()
        for i,n in enumerate(neurals):
            for key,values in self.arrays.items():setattr(n,key,values[i])
            n.W=SimpleNamespace(data=self.weights[i])
            n.indptr=first.indptr;n.indices=first.indices;n.out_ptr=first.out_ptr;n.out_indices=first.out_indices
            n.stream=self.stream;n._graph=None;n._observed_tick=None
        self.shared_topology_bytes=sum(getattr(first,k).nbytes for k in ('indptr','indices','out_ptr','out_indices'))
        self.pending=None;self.layout=None;self.cuda_graph=None

    def begin(self,drives,steps,captures,pulses):
        if self.pending is not None:raise RuntimeError('Previous batch period pending')
        if not all(len(x)==self.worlds for x in (drives,captures,pulses)):raise ValueError('One neural input per world required')
        prepared=[validate_input(self.n,x,steps,c,p) for x,c,p in zip(drives,captures,pulses)]
        unique=np.unique(np.concatenate([item[1] for item in prepared])).astype(np.int32)
        width=max([0]+[len(p[0]) for p in pulses if p is not None])
        ptr=np.zeros((self.worlds,self.n+1),np.int32)
        values=np.zeros((self.worlds,steps,width),np.float32)
        for i,p in enumerate(pulses):
            if p is None:continue
            ids=np.asarray(p[0]);order=np.argsort(ids,kind='stable')
            ptr[i,1:]=np.cumsum(np.bincount(ids.astype(np.int64),minlength=self.n))
            values[i,:,:len(ids)]=np.asarray(p[1],np.float32)[:,order]
        layout=(steps,tuple(unique),width)
        cp=self.cp;first=self.neurals[0];a=self.arrays
        with self.stream:
            if layout!=self.layout:
                self.layout=layout;self.cuda_graph=None
                self.drive=cp.empty((self.worlds,self.n),cp.float32)
                self.clock=cp.empty(self.worlds,cp.uint64);self.health=cp.zeros(self.worlds,cp.int32)
                self.events=cp.empty((self.worlds,steps,max(1,len(unique))),cp.uint8)
                self.pulse_ptr=cp.empty(ptr.shape,cp.int32);self.pulse_values=cp.empty(values.shape,cp.float32)
                mapping=np.full(self.n,-1,np.int32);mapping[unique]=np.arange(len(unique),dtype=np.int32)
                self.capture_map=cp.asarray(mapping)
            self.drive.set(np.asarray([x[0] for x in prepared],np.float32),stream=self.stream)
            self.clock.set(np.array([n.tick for n in self.neurals],np.uint64),stream=self.stream)
            self.pulse_ptr.set(ptr,stream=self.stream);self.pulse_values.set(values,stream=self.stream);self.health.fill(0)
            kind={'exact-exponential-held-drive-v1':0,'exact-exponential-reset-current-v1':1,'exact-exponential-voltage-events-v1':2}[first.p.integration]
            def enqueue():
                for k in range(steps):
                    self.integrate(((self.n+127)//128,self.worlds),(128,),
                        (a['v'],a['h'],a['rate'],a['refractory_until'],a['spike_count'],a['queue'],a['suppress'],a['mute'],a['emitted'],
                         self.drive,first.constants,self.capture_map,self.events,self.pulse_ptr,self.pulse_values,self.health,
                         first.out_ptr,first.out_indices,a['active_rows'],self.clock,
                         *map(np.int32,(self.n,self.slots,first.refractory_ticks,k,len(unique),width,kind)),
                         np.float64((1-first.er)/first.p.dt),np.int32(steps)))
                    self.propagate(((self.n+127)//128,self.worlds),(128,),
                        (first.indptr,first.indices,self.weights,a['emitted'],a['active_rows'],a['queue'],self.clock,
                         *map(np.int32,(self.n,self.slots,k,len(first.graph.weights)))))
            if self.cuda_graph is None:
                self.integrate.compile();self.propagate.compile()
                self.stream.begin_capture()
                try:enqueue()
                finally:self.cuda_graph=self.stream.end_capture()
            self.cuda_graph.launch(self.stream)
        self.pending=(steps,prepared,unique,[n.tick for n in self.neurals])

    def finish(self):
        if self.pending is None:raise RuntimeError('No pending neural batch')
        health=self.health.get(stream=self.stream)
        events=self.events.get(stream=self.stream)
        steps,prepared,unique,starts=self.pending;self.pending=None
        if np.any(health):raise RuntimeError('Nonfinite neural worlds: '+str(np.flatnonzero(health).tolist()))
        for w,n in enumerate(self.neurals):
            capture=prepared[w][1];n.tick+=steps;n._observed_tick=None;n.last_events=[]
            if len(capture):
                times,cells=np.nonzero(events[w,:,:len(unique)][:,np.searchsorted(unique,capture)])
                n.last_events=[dict(tick=starts[w]+int(t)+1,index=int(capture[j])) for t,j in zip(times,cells)]

    def advance(self,drives,steps,captures,pulses):self.begin(drives,steps,captures,pulses);self.finish()
