"""E303 SSPRK(3,3) for the unchanged recovery-resource neural ODE."""
from __future__ import annotations

import hashlib

import numpy as np

from tools import banc_m3_recovery_resource_synapses as prior
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork

SCHEMA="flylab.m3-ssprk3-recovery-resource.v1"
PRECISION=("SSPRK33: E(y0,h); .75*y0+.25*E(y1,h); (1/3)*y0+(2/3)*E(y2,h); "
    "neural-f32 q_mem=f32(h)/tau32,q_adapt=f32(h)*canonical_inverse32; "
    "rFE=(1-q_mem)*r+q_mem*T,zFE=(1-q_adapt)*z+q_adapt*(gain32*r); "
    "inverseR64=1/tau64; RFE=(1-h*(inverseR64+u64*f64(r32)))*R64+h*inverseR64; u0R=1; "
    "source factor f64 R*(1+(tau*u)*f64(r)),min1->f32; "
    "same-stage resource before neural overwrite; no-FMA/noFTZ pointwise,original CSR FTZ/warp order")
SELECTION_KEYS=prior.SELECTION_KEYS
checked_selection=prior.checked_selection
resource_factors=prior.resource_factors
resource_derivative=prior.resource_derivative
resource_edge_value=prior.resource_edge_value
packed_current_oracle=prior.packed_current_oracle
FAULT_RESOURCE,FAULT_NEUTRAL_RESOURCE,FAULT_ACTIVE_RATE,FAULT_FACTOR,FAULT_FINAL_RESOURCE=1,2,4,8,16
FAULT_NATIVE_RANGE=32


def resource_metadata(source_model_hash,selection):
    return {**prior.resource_metadata(source_model_hash,selection),"schema":SCHEMA,"precision":PRECISION}


def checked_resource_metadata(metadata):
    if not isinstance(metadata,dict) or metadata.get("schema")!=SCHEMA or metadata.get("precision")!=PRECISION:
        raise ValueError("Explicit SSPRK33 resource metadata required")
    result=prior.checked_resource_metadata({**metadata,"schema":prior.SCHEMA,"precision":prior.PRECISION})
    return {**result,"schema":SCHEMA,"precision":PRECISION}


def resource_identity(metadata):
    m=checked_resource_metadata(metadata)
    h=hashlib.sha256((m["source_model_hash"]+SCHEMA+PRECISION).encode())
    for key in sorted(SELECTION_KEYS):
        value=m[key];h.update(key.encode());h.update(value.dtype.str.encode())
        h.update(np.asarray(value.shape,np.int64).tobytes());h.update(value.tobytes())
    return h.hexdigest()


_SSP_SOURCE=r'''
extern "C" __global__ void ssp_resource(const long long* sources,const float* rate,
    const double* resource,const double* base,const double* u,const double* tau,
    double* next,int* fault,double h,double base_weight,double work_weight,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    double R=resource[i],b=base[i];
    // Valid u0 is exactly one; invalid u0 remains invalid, never repaired.
    double nr=R;
    if (u[i]!=0.0) {
        double inverse=1.0/tau[i];
        double q=h*(inverse+u[i]*recovery_f32_to_f64(rate[sources[i]]));
        double fe=(1.0-q)*R+h*inverse;
        nr=base_weight*b+work_weight*fe;
    }
    next[i]=nr;
    int flags=0;
    if (!isfinite(nr)||nr<0.0||nr>1.0) flags|=16;
    if (u[i]==0.0 && R!=1.0) flags|=2;
    if (flags) atomicOr(fault,flags);
}
extern "C" __global__ void ssp_neuron(const float* activation,const float* cap,
    const float* rate,const float* adaptation,const float* tau,
    const float* gain,const float* inverse,const float* base_rate,const float* base_adaptation,
    const float* mask,float* next_rate,float* next_adaptation,float* next_masked,
    int* fault,float h,float base_weight,float work_weight,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    float r=rate[i],z=adaptation[i];
    float target=clamp_exact(mul_exact(cap[i],SSP_ACTIVATION));
    float qr=div_exact(h,tau[i]),qz=mul_exact(h,inverse[i]);
    float rfe=add_exact(mul_exact(sub_exact(1.0f,qr),r),mul_exact(qr,target));
    float zfe=add_exact(mul_exact(sub_exact(1.0f,qz),z),mul_exact(qz,mul_exact(gain[i],r)));
    float nr=add_exact(mul_exact(base_weight,base_rate[i]),mul_exact(work_weight,rfe));
    float nz=add_exact(mul_exact(base_weight,base_adaptation[i]),mul_exact(work_weight,zfe));
    if (recovery_bad_rate(r)||recovery_bad_rate(z)||recovery_bad_rate(nr)||recovery_bad_rate(nz)) atomicOr(fault,32);
    next_rate[i]=nr;next_adaptation[i]=nz;next_masked[i]=mul_exact(nr,mask[i]);
}
'''


def resource_cuda_source(fuse_tanh=True):
    activation="matching_tanh(activation[i])" if fuse_tanh else "activation[i]"
    return prior.resource_cuda_source(fuse_tanh)+_SSP_SOURCE.replace("SSP_ACTIVATION",activation)


class SSPRecoveryResourceNetwork(prior.RecoveryResourceNetwork):
    """Three simultaneous Euler/convex stages for both neutral and active source."""
    def __init__(self,*args,edge_ids,premotor_ids,utilization,recovery_tau,**kwargs):
        tau,cap=np.asarray(args[1]),np.asarray(args[4]);h=kwargs.get("dt",.0001)
        gains=(kwargs.get("adaptation_gain",8.),kwargs.get("core_gain"))
        if (tau.ndim!=1 or not np.isfinite(tau).all() or np.any(tau<=0)
                or cap.shape!=tau.shape or not np.isfinite(cap).all() or np.any(cap<=0)
                or type(h) not in (int,float) or not np.isfinite(h) or h<=0 or h>float(tau.min())/2
                or any(type(g) not in (int,float) or not np.isfinite(g) or g<0 for g in gains)):
            raise ValueError("Resolved positive membrane/cap and nonnegative native gains required")
        super().__init__(*args,edge_ids=edge_ids,premotor_ids=premotor_ids,utilization=utilization,recovery_tau=recovery_tau,**kwargs)
        self._metadata=resource_metadata(self.source_timescale_identity,self._selection)
        self.identity=resource_identity(self._metadata);self._inherited_metadata["model_hash"]=self.identity
        self._ssp_pointers=tuple(value.data_ptr() for value in self._ssp_buffers)

    def _init_resource(self):
        super()._init_resource()
        if not hasattr(self,"_ssp_buffers"):
            with self.torch.inference_mode(False):
                self._ssp_buffers=[self.torch.empty_like(self.rate),self.torch.empty_like(self.adaptation)]

    def _assert_pointers(self):
        super()._assert_pointers()
        if hasattr(self,"_ssp_pointers") and tuple(value.data_ptr() for value in self._ssp_buffers)!=self._ssp_pointers:
            raise ValueError("Captured SSP scratch buffer pointer changed")

    def _init_fusion(self):
        self._init_resource()
        MotorTimescaleNetwork._init_fusion(self)
        source=resource_cuda_source(self.fuse_tanh)
        for name in ("csr_input","recovery_factor","ssp_resource","ssp_neuron"):
            kernel=self._cupy.RawKernel(source,name,options=("--fmad=false",));kernel.compile()
            self._fused_kernels[name]=kernel

    def _ssp_resource_launch(self,r,R,out,base_weight,work_weight):
        count=len(self._selection["source_ids"])
        tensors=(self._resource_sources,r,R,self.resource,self._resource_u,self._resource_tau,out,self._resource_fault)
        self._fused_kernels["ssp_resource"](((count+255)//256,),(256,),
            tuple(np.uint64(value.data_ptr()) for value in tensors)+(np.float64(self.dt),np.float64(base_weight),np.float64(work_weight),np.int32(count)))

    def _step_fused(self):
        if not hasattr(self,"_fused_kernels"):self._init_fusion()
        masked,_,_,inputs,activation,*_=self._fused_buffers
        wr,wz=self._ssp_buffers;wR=self._resource_buffers[0]
        r,z,R=self.rate,self.adaptation,self.resource
        self.torch.mul(r,self.output_mask,out=masked)
        with self._cupy.cuda.ExternalStream(self.torch.cuda.current_stream().cuda_stream):
            for stage,(b,w) in enumerate(((0.,1.),(.75,.25),(1/3,2/3))):
                self._recovery_launch("recovery_factor",(self._resource_sources,r,R,self._resource_u,self._resource_tau,
                    self.resource_factor,self._resource_fault))
                self._launch_fused("csr_input",(self._rowptr,self._columns,self._weight_values,masked,self.drive,self.threshold,z,self.coefficient,
                    inputs,self._resource_edge_source,self.resource_factor))
                if self.fuse_tanh:activation=inputs
                else:self.torch.tanh(inputs,out=activation)
                nr,nz,nR=(self.rate,self.adaptation,self.resource) if stage==2 else (wr,wz,wR)
                # R consumes the same old r before any neural pointer is overwritten.
                self._ssp_resource_launch(r,R,nR,b,w)
                tensors=(activation,self.cap,r,z,self.tau,self._cell_gain,self._cell_inverse_tau,self.rate,self.adaptation,
                         self.output_mask,nr,nz,masked,self._resource_fault)
                self._fused_kernels["ssp_neuron"](((self.n+255)//256,),(256,),
                    tuple(np.uint64(value.data_ptr()) for value in tensors)+(np.float32(self.dt),np.float32(b),np.float32(w),np.int32(self.n)))
                r,z,R=nr,nz,nR

    def _native_guard(self,r,z):
        if (not self.torch.isfinite(r).all() or not self.torch.isfinite(z).all()
                or self.torch.any(r<0) or self.torch.any(z<0)):
            active=r[self._resource_sources[self._resource_u>0]]
            flags=FAULT_NATIVE_RANGE
            if not self.torch.isfinite(active).all() or self.torch.any(active<0):flags|=FAULT_ACTIVE_RATE
            self._resource_fault.bitwise_or_(flags)
            raise prior._fault_error(int(self._resource_fault.numpy()[0]))

    def _step(self):
        if self.device=="cuda":return self._step_fused()
        t=self.torch;r0,z0,R0=self.rate,self.adaptation,self.resource
        r,z,R=r0,z0,R0;wr,wz=self._ssp_buffers;wR=self._resource_buffers[0]
        for stage,(b,w) in enumerate(((0.,1.),(.75,.25),(1/3,2/3))):
            self._native_guard(r,z)
            factor=self._cpu_factors(r,R);self.resource_factor.copy_(t.as_tensor(factor))
            current=t.as_tensor(packed_current_oracle(self._source_weights,r.numpy(),self.output_mask.numpy(),self._selection,R.numpy(),factors=factor))
            inputs=self.drive+current;inputs=inputs-self.threshold;inputs=inputs-z;inputs=self.coefficient*inputs
            target=t.clamp_min(self.cap*t.tanh(inputs),0.)
            qr=self.dt/self.tau;qz=self.dt*self._cell_inverse_tau
            rfe=(1.-qr)*r+qr*target;zfe=(1.-qz)*z+qz*(self._cell_gain*r)
            inverse=1./self._resource_tau
            Rfe=(1.-self.dt*(inverse+self._resource_u*r[self._resource_sources].to(t.float64)))*R+self.dt*inverse
            nR=b*R0+w*Rfe;nR[self._resource_u==0]=1.
            nr,nz=b*r0+w*rfe,b*z0+w*zfe
            self._native_guard(nr,nz)
            flags=prior._stage_flags(nr.numpy(),nR.numpy(),self._selection["source_ids"],self._selection["utilization"])
            if flags:
                if flags&FAULT_RESOURCE:flags=(flags&~FAULT_RESOURCE)|FAULT_FINAL_RESOURCE
                self._resource_fault.bitwise_or_(flags);raise prior._fault_error(int(self._resource_fault.numpy()[0]))
            if stage==2:r0.copy_(nr);z0.copy_(nz);R0.copy_(nR)
            else:wr.copy_(nr);wz.copy_(nz);wR.copy_(nR);r,z,R=wr,wz,wR

    def _record_final_faults(self):
        super()._record_final_faults()
        for value in (self.rate,self.adaptation):
            bits=value.view(self.torch.int32);magnitude=bits&0x7fffffff
            bad=((magnitude>=0x7f800000)|((bits<0)&(magnitude!=0))).any()
            self._resource_fault.bitwise_or_((bad.to(self.torch.int32)*FAULT_NATIVE_RANGE).reshape(1))

    def _prevalidate(self,saved):
        if not isinstance(saved,dict):raise TypeError("Complete SSP checkpoint required")
        for key in ("rate","adaptation"):
            value=saved.get(key)
            if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=(self.n,) or not np.isfinite(value).all() or np.any(value<0):
                raise ValueError("Nonnegative finite all-neuron SSP "+key+" required")
        return super()._prevalidate(saved)
