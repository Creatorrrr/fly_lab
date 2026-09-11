"""Optional deterministic contact ordering for Warp reproducibility experiments.

Uses the supported contactfilter callback, after narrowphase and before rows
are assembled. This orders complete contact records, never changes forces.
"""
from dataclasses import fields
import numpy as np


class OrderedContacts:
    def __init__(self,model,data,stream):
        import cupy as cp
        import warp as wp
        self.cp,self.wp=cp,wp;self.stream=cp.cuda.ExternalStream(stream.cuda_stream)
        self.wp_stream=stream;self.n=data.naconmax;self.ngeom=model.ngeom
        self.key=wp.zeros(self.n*2,dtype=wp.int64,device=stream.device)
        self.order=wp.zeros(self.n*2,dtype=wp.int32,device=stream.device)
        self.keys,self.indices=cp.asarray(self.key),cp.asarray(self.order)
        self.views={field.name:cp.asarray(getattr(data.contact,field.name)) for field in fields(data.contact)}
        self.nacon=cp.asarray(data.nacon)
        self.scratch={name:cp.empty_like(value) for name,value in self.views.items()}
        self.key_kernel=cp.RawKernel(r'''
        extern "C" __global__ void keys(const int *ncon,const int *world,const int *geom,const int *local,
          long long *key,int *order,int n,int ng) {
          int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;order[i]=i;
          key[i]=i<ncon[0]?(((long long)world[i]*(ng+1)+geom[2*i]+1)*(ng+1)+geom[2*i+1]+1)*65536+local[i]:0x7fffffffffffffffLL;
        }''','keys')
        self.gather=cp.RawKernel(r'''
        extern "C" __global__ void gather(const unsigned char *src,unsigned char *dst,const int *order,int n,int width) {
          int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n*width)return;
          dst[i]=src[order[i/width]*width+i%width];
        }''','gather')

    def __call__(self,model,data):
        with self.stream,self.wp.ScopedStream(self.wp_stream):
            self.key_kernel(((self.n+127)//128,),(128,),
                (self.nacon,self.views['worldid'],self.views['geom'],self.views['geomcollisionid'],self.keys,self.indices,np.int32(self.n),np.int32(self.ngeom)))
            self.wp.utils.radix_sort_pairs(self.key,self.order,self.n)
            for name,src in self.views.items():
                width=src.nbytes//self.n;dest=self.scratch[name]
                self.gather(((src.nbytes+127)//128,),(128,),(src,dest,self.indices,np.int32(self.n),np.int32(width)))
                self.cp.copyto(src,dest)
