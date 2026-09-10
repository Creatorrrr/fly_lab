"""Persistent CUDA LIF with ordered CSR sums and replayable control periods.

One CUDA graph launches a whole neural period on an independent stream. The
CPU body may run while it completes; health, events and observations return
in one pinned transfer. No floating-point atomics or fast-math approximation.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from .neural import ExpLIF, LIFParameters
from .inputs import validate_input
from .neural_mps import MetalLIF


class CudaLIF(ExpLIF):
    def __init__(self, graph, parameters=None, *, use_graphs=True):
        p=parameters or LIFParameters()
        if p.dtype != 'float32': raise ValueError('Optimized CUDA kernels require float32')
        if graph.n >= 2**31 or len(graph.weights) >= 2**31:
            raise ValueError('CUDA CSR indices exceed int32 capacity')
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1: raise RuntimeError('No NVIDIA GPU')
            source=Path(__file__).with_name('lif.cu').read_text(encoding='utf-8')
            module=cp.RawModule(code=source,options=('--std=c++17','--fmad=false'))
            kernels={name:module.get_function(name) for name in ('integrate','propagate','read_state','summarize')}
        except Exception as exc:
            raise RuntimeError('BLOCKED_CUDA: CUDA runtime/kernel initialization failed: '+str(exc)) from exc
        super().__init__(graph,p)
        outgoing=self.W.tocsc()
        self.backend='exp_lif_cuda'; self.xp=cp; self.stream=cp.cuda.Stream(non_blocking=True)
        self.use_graphs=bool(use_graphs); self.graph_builds=0; self.graph_replays=0
        self.runtime_identity=dict(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            shader_sha256=hashlib.sha256(source.encode()).hexdigest(),
            numerical_policy='float32-serial-csr-no-fma-cpu-rate-v1')
        self._kernels=kernels; self._module=module; self._pending_advance=None
        self._graph=None; self._layout=None; self._observed_tick=None
        self._intervention_key=None; self.additional_readout=np.empty(0,np.int32)
        self._pulse_indices=None; self._buffers={}
        with self.stream:
            for key in ('v','h','rate','refractory_until','queue','spike_count','suppress','mute'):
                setattr(self,key,cp.asarray(getattr(self,key)))
            self.W=SimpleNamespace(data=cp.asarray(graph.weights,np.float32))
            self.indptr=cp.asarray(graph.indptr,np.int32); self.indices=cp.asarray(graph.indices,np.int32)
            self.out_ptr=cp.asarray(outgoing.indptr,np.int32); self.out_indices=cp.asarray(outgoing.indices,np.int32)
            self.active_rows=cp.zeros(self.n,cp.uint32); self.emitted=cp.zeros(self.n,cp.uint8)
            self.capture_map=cp.full(self.n,-1,cp.int32); self.pulse_ptr=cp.zeros(self.n+1,cp.int32)
            self.constants=cp.asarray([p.rest_mV,p.reset_mV,p.threshold_mV,self.em,self.es,self.er,self.syn_factor],np.float32)
        self.stream.synchronize()

    def host(self,array):
        return array.get(stream=self.stream)

    def _buffer(self,name,shape,dtype):
        dtype=np.dtype(dtype); cached=self._buffers.get(name)
        if cached is None or cached[0].shape != shape or cached[0].dtype != dtype:
            memory=self.xp.cuda.alloc_pinned_memory(max(1,int(np.prod(shape))*dtype.itemsize))
            host=np.ndarray(shape,dtype=dtype,buffer=memory)
            device=self.xp.empty(shape,dtype=dtype)
            cached=(host,device); self._buffers[name]=cached; self._graph=None
        return cached

    def _upload(self,name,value):
        value=np.asarray(value)
        host,device=self._buffer(name,value.shape,value.dtype)
        np.copyto(host,value)
        device.set(host,stream=self.stream)
        return device

    def set_interventions(self,suppress=(),mute=(),edges=()):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before changing interventions')
        suppress,mute,edges=list(suppress),list(mute),sorted(set(edges))
        for ids in (suppress,mute):
            if any(type(i) is not int or not 0<=i<self.n for i in ids): raise ValueError('Invalid intervention neuron index')
        if any(type(i) is not int or not 0<=i<len(self.graph.weights) for i in edges): raise ValueError('Invalid intervention edge index')
        key=(tuple(suppress),tuple(mute),tuple(edges))
        if key != self._intervention_key:
            with self.stream: super().set_interventions(suppress,mute,edges)
            self._intervention_key=key

    set_readout_cohort=MetalLIF.set_readout_cohort
    _unpack_readout=staticmethod(MetalLIF._unpack_readout)
    _unpack_summary=MetalLIF._unpack_summary

    def advance(self,drive,steps,capture=(),pulses=None):
        self.begin_advance(drive,steps,capture,pulses)
        self.finish_advance()

    def begin_advance(self,drive,steps,capture=(),pulses=None):
        if self._pending_advance is not None: raise RuntimeError('Previous neural period still pending')
        drive,capture=validate_input(self.n,drive,steps,capture,pulses)
        unique=np.union1d(capture,self.additional_readout).astype(np.int32)
        inverse=np.searchsorted(unique,capture)
        pi=np.empty(0,np.int32) if pulses is None else np.asarray(pulses[0])
        values=np.empty((steps,0),np.float32) if pulses is None else np.asarray(pulses[1],np.float32)
        layout=(steps,tuple(unique),tuple(pi))
        with self.stream:
            if layout != self._layout:
                self._graph=None
                mapping=np.full(self.n,-1,np.int32); mapping[unique]=np.arange(len(unique),dtype=np.int32)
                self.capture_map.set(mapping,stream=self.stream)
                self._capture_lookup={int(j):i for i,j in enumerate(unique)}
                order=np.argsort(pi,kind='stable')
                ptr=np.zeros(self.n+1,np.int32)
                ptr[1:]=np.cumsum(np.bincount(pi.astype(np.int64),minlength=self.n))
                self.pulse_ptr.set(ptr,stream=self.stream); self._pulse_order=order
                self._capture_device=self.xp.asarray(unique)
                self._layout=layout
            x=self._upload('drive',np.asarray(drive,np.float32))
            pv=self._upload('pulses',np.ascontiguousarray(values[:,self._pulse_order]))
            clock=self._upload('clock',np.array([self.tick],np.uint64))
            event_bytes=max(1,steps*len(unique)); read_start=(8+event_bytes+7)//8*8
            stats_start=read_start+16*len(unique); blocks=(self.n+255)//256
            raw,collected=self._buffer('collected',(stats_start+24*blocks,),np.uint8)
            health=collected[:4].view(self.xp.int32); health.fill(0)
            events=collected[8:8+event_bytes]
            grid=((self.n+127)//128,); threads=(128,)
            kind={'exact-exponential-held-drive-v1':0,'exact-exponential-reset-current-v1':1,
                  'exact-exponential-voltage-events-v1':2}[self.p.integration]
            def enqueue():
                for k in range(steps):
                    args=(self.v,self.h,self.rate,self.refractory_until,self.spike_count,self.queue,
                        self.suppress,self.mute,self.emitted,x,self.constants,self.capture_map,events,
                        self.pulse_ptr,pv,health,self.out_ptr,self.out_indices,self.active_rows,clock,
                        *map(np.int32,(self.n,self.slots,self.refractory_ticks,k,len(unique),len(pi),kind)),
                        np.float64((1-self.er)/self.p.dt))
                    self._kernels['integrate'](grid,threads,args)
                    self._kernels['propagate'](grid,threads,(self.indptr,self.indices,self.W.data,
                        self.emitted,self.active_rows,self.queue,clock,*map(np.int32,(self.n,self.slots,k))))
                if len(unique):
                    self._kernels['read_state'](((len(unique)+127)//128,),threads,
                        (self.v,self.rate,self.spike_count,self._capture_device,collected[read_start:stats_start],np.int32(len(unique))))
                self._kernels['summarize'](((blocks+127)//128,),threads,
                    (self.v,self.rate,self.spike_count,collected[stats_start:],np.int32(self.n)))
            if self.use_graphs:
                if self._graph is None:
                    self.stream.begin_capture()
                    try:
                        enqueue()
                    finally:
                        captured=self.stream.end_capture()
                    self._graph=captured; self.graph_builds+=1
                self._graph.launch(self.stream); self.graph_replays+=1
            else: enqueue()
            collected.get(out=raw,stream=self.stream,blocking=False)
        self._pending_advance=(raw,read_start,stats_start,self.tick,steps,capture.copy(),unique,inverse)
        self.tick+=steps; self._observed_tick=None

    def finish_advance(self):
        if self._pending_advance is None: raise RuntimeError('No pending neural period')
        self.stream.synchronize()
        raw,read_start,stats_start,start,steps,capture,unique,inverse=self._pending_advance
        self._pending_advance=None
        if int(raw[:4].view(np.int32)[0]): raise RuntimeError('FAULT_NEURAL_NONFINITE')
        self.last_events=[]
        if steps and len(capture):
            samples=raw[8:8+steps*len(unique)].reshape(steps,len(unique))[:,inverse]
            times,cells=np.nonzero(samples)
            self.last_events=[dict(tick=start+int(t)+1,index=int(capture[j])) for t,j in zip(times,cells)]
        self._observed_values=self._unpack_readout(raw[read_start:stats_start],len(unique))
        self._observed_summary=self._unpack_summary(raw[stats_start:]); self._observed_tick=self.tick

    def readout(self,indices):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before observing it')
        indices=np.asarray(indices,dtype=np.int32)
        if indices.ndim!=1 or (len(indices) and (indices.min()<0 or indices.max()>=self.n)):
            raise ValueError('Invalid readout indices')
        if self._observed_tick==self.tick and all(int(j) in self._capture_lookup for j in indices):
            order=np.array([self._capture_lookup[int(j)] for j in indices],np.int32)
            return {k:v[order].copy() for k,v in self._observed_values.items()}
        with self.stream: return super().readout(indices)

    def summary(self):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before observing it')
        if self._observed_tick==self.tick: return dict(self._observed_summary)
        with self.stream: return super().summary()

    region_summary=MetalLIF.region_summary

    def snapshot(self):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before checkpointing')
        state=super().snapshot(); state['backend_runtime']=dict(self.runtime_identity)
        return state

    def restore(self,state):
        if self._pending_advance is not None: raise RuntimeError('Finish neural period before restoring')
        if state.get('backend_runtime') != self.runtime_identity:
            raise ValueError('CUDA runtime or shader differs; explicit backend transfer required')
        self._intervention_key=None
        with self.stream: result=super().restore(state)
        self._observed_tick=None
        return result
