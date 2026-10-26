"""Bounded generated E303 checks. Actual CUDA is reserved by root only."""
import copy
import json
import os
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.storage import StateStore
from tests.test_c_banc_recovery_resource_synapses import (
    parameters,
    peak_rss_bytes,
    seeded,
)
from tools import banc_m3_ssp_recovery_synapses as ssp

CUDA_ENABLED=os.environ.get("RUN_E303_CUDA")=="1"
DT=.00025
COST=dict.fromkeys(("constructor_requested","constructor_completed","constructor_failed","actual_constructor_requested",
    "actual_constructor_completed","actual_constructor_failed","step_requested","step_completed","step_failed",
    "setup_steps","capture_requested","capture_completed","capture_failed","advance_requested","advance_completed",
    "advance_failed","logical_ticks","oracle_ticks","CUDA_calls","original_calls","body_calls","solver_calls","fit_calls"),0)
COST.update(constructor_wall_s=0.,actual_constructor_wall_s=0.,step_wall_s=0.,advance_wall_s=0.)


@pytest.fixture(scope="module",autouse=True)
def cost_receipt():
    import torch
    prior=torch.get_num_threads();torch.set_num_threads(1);wall,cpu=time.perf_counter(),time.process_time()
    yield
    COST.update(wall_s=time.perf_counter()-wall,process_s=time.process_time()-cpu,peak_rss_bytes=peak_rss_bytes())
    assert torch.get_num_threads()==1;torch.set_num_threads(prior)
    print("E303_GENERATED_COST "+json.dumps(COST,sort_keys=True))
    assert COST["wall_s"]<=60 and COST["peak_rss_bytes"]<=(2 if CUDA_ENABLED else 1.5)*1024**3
    assert COST["logical_ticks"]+COST["oracle_ticks"]<=(3000 if CUDA_ENABLED else 10000)


@pytest.fixture(autouse=True)
def instrument(monkeypatch):
    ctor,actual,step,advance=ssp.SSPRecoveryResourceNetwork.__init__,AdaptiveRateNetwork.__init__,ssp.SSPRecoveryResourceNetwork._step,ssp.SSPRecoveryResourceNetwork._advance
    def metered_ctor(self,*args,**kw):
        assert args[0].shape[0]<=64 and args[0].nnz<=256
        COST["constructor_requested"]+=1;began=time.perf_counter();self._test_setup=True
        if kw["device"]=="cuda":
            assert CUDA_ENABLED;COST["CUDA_calls"]+=1;COST["capture_requested"]+=1
        try:
            result=ctor(self,*args,**kw);COST["constructor_completed"]+=1
            if kw["device"]=="cuda":COST["capture_completed"]+=1
            return result
        except Exception:
            COST["constructor_failed"]+=1
            if kw["device"]=="cuda":COST["capture_failed"]+=1
            raise
        finally:COST["constructor_wall_s"]+=time.perf_counter()-began;self._test_setup=False
    def metered_actual(self,*args,**kw):
        COST["actual_constructor_requested"]+=1;began=time.perf_counter()
        try:result=actual(self,*args,**kw);COST["actual_constructor_completed"]+=1;return result
        except Exception:COST["actual_constructor_failed"]+=1;raise
        finally:COST["actual_constructor_wall_s"]+=time.perf_counter()-began
    def metered_step(self):
        COST["step_requested"]+=1;began=time.perf_counter()
        if getattr(self,"_test_setup",False):COST["setup_steps"]+=1
        try:result=step(self);COST["step_completed"]+=1;return result
        except Exception:COST["step_failed"]+=1;raise
        finally:COST["step_wall_s"]+=time.perf_counter()-began
    def metered_advance(self,*args,**kw):
        COST["advance_requested"]+=1;began=time.perf_counter();tick=self.tick
        try:result=advance(self,*args,**kw);COST["advance_completed"]+=1;return result
        except Exception:COST["advance_failed"]+=1;raise
        finally:COST["advance_wall_s"]+=time.perf_counter()-began;COST["logical_ticks"]+=self.tick-tick
    monkeypatch.setattr(ssp.SSPRecoveryResourceNetwork,"__init__",metered_ctor)
    monkeypatch.setattr(AdaptiveRateNetwork,"__init__",metered_actual)
    monkeypatch.setattr(ssp.SSPRecoveryResourceNetwork,"_step",metered_step)
    monkeypatch.setattr(ssp.SSPRecoveryResourceNetwork,"_advance",metered_advance)


def model(device="cpu",u=None):
    args,kw=parameters(device)
    return ssp.SSPRecoveryResourceNetwork(*args,**kw,edge_ids=np.arange(4,dtype=np.int64),premotor_ids=np.arange(3,dtype=np.int64),
        utilization=np.array([.05,.1,.2]) if u is None else u,recovery_tau=np.array([.15,.12,.2]))


def equal(a,b):
    assert ssp.prior.original._equal(a,b)


def independent_current(w,r,mask,s,R):
    def ftz(x):
        x=np.float32(x)
        return np.copysign(np.float32(0),x) if abs(x)<np.finfo(np.float32).tiny else x
    edges=dict(zip(s["edge_ids"],s["edge_source_slots"],strict=True))
    factors=np.ones(len(R),np.float32)
    for i,pre in enumerate(s["source_ids"]):
        if s["utilization"][i]!=0:factors[i]=np.float32(min(1.,float(R[i])*(1.+float(s["recovery_tau"][i]*s["utilization"][i])*float(r[pre]))))
    result=np.zeros(len(r),np.float32)
    for row in range(len(r)):
        lane=np.zeros(32,np.float32)
        for j in range(32):
            for edge in range(w.indptr[row]+j,w.indptr[row+1],32):
                pre=w.indices[edge];value=np.float32(r[pre]*mask[pre])
                if edge in edges and factors[edges[edge]]!=1:value=ftz(ftz(value)*ftz(factors[edges[edge]]))
                lane[j]=ftz(ftz(lane[j])+ftz(ftz(w.data[edge])*ftz(value)))
        for offset in (16,8,4,2,1):
            old=lane.copy()
            for j in range(32-offset):lane[j]=ftz(ftz(old[j])+ftz(old[j+offset]))
        result[row]=lane[0]
    return result


def oracle_tick(n,state):
    w,tau,a,threshold,cap=parameters()[0];s=n.selection
    gain=np.array([0,0,8,1,2,8],np.float32)
    ztau=np.array([.3,.3,.3,.3,.6,.3],np.float64);inverse=(1./ztau).astype(np.float32)
    r0,z0,R0=(state[k].copy() for k in ("rate","adaptation","synaptic_resource"));r,z,R=r0,z0,R0;stages=[]
    for b,v in ((0.,1.),(.75,.25),(1/3,2/3)):
        stages.append((r.copy(),z.copy(),R.copy()))
        c=independent_current(w,r,state["output_mask"],s,R)
        arg=np.float32(a/cap)*(((state["drive"]+c)-threshold)-z)
        target=np.maximum(cap*np.tanh(arg),np.float32(0))
        qr=np.float32(n.dt)/tau;qz=np.float32(n.dt)*inverse
        nr=np.float32(b)*r0+np.float32(v)*((np.float32(1)-qr)*r+qr*target)
        nz=np.float32(b)*z0+np.float32(v)*((np.float32(1)-qz)*z+qz*(gain*r))
        nextR=np.ones(len(R),np.float64)
        for i,pre in enumerate(s["source_ids"]):
            if s["utilization"][i]!=0:
                inverseR=1./float(s["recovery_tau"][i]);q=n.dt*(inverseR+s["utilization"][i]*float(r[pre]))
                nextR[i]=b*R0[i]+v*((1.-q)*R[i]+n.dt*inverseR)
        r,z,R=nr,nz,nextR
    result=copy.deepcopy(state);result.update(rate=r,adaptation=z,synaptic_resource=R,tick=state["tick"]+1)
    return result,stages


@pytest.mark.parametrize("u",[np.zeros(3),np.array([.05,0,.2])])
def test_three_simultaneous_stages_independent_oracle_mask_and_no_neutral_bypass(u,monkeypatch):
    n=model(u=u);state=seeded(n);n.set_muted([1]);state=n.snapshot();expected,stages=oracle_tick(n,state)
    observed=[];current=[0];factor=n._cpu_factors;actual=ssp.packed_current_oracle
    def observe(r,R):observed.append((r.numpy().copy(),n._ssp_buffers[1].numpy().copy() if observed else n.adaptation.numpy().copy(),R.numpy().copy()));return factor(r,R)
    def count(*args,**kwargs):current[0]+=1;return actual(*args,**kwargs)
    monkeypatch.setattr(n,"_cpu_factors",observe);monkeypatch.setattr(ssp,"packed_current_oracle",count)
    n.advance(state["drive"],steps=1)
    assert len(observed)==3 and current[0]==3
    for got,want in zip(observed,stages,strict=True):
        for a,b in zip(got,want,strict=True):np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
    final=n.snapshot()
    for key in ("rate","adaptation"):np.testing.assert_allclose(final[key],expected[key],rtol=0,atol=1e-6)
    np.testing.assert_allclose(final["synaptic_resource"],expected["synaptic_resource"],rtol=0,atol=1e-13)
    assert np.all(final["synaptic_resource"][u==0]==1)


def smooth_derivative(y,u):
    r,z,R=y[:6],y[6:12],y[12:];w=parameters()[0][0].toarray().astype(np.float64)
    # Independent dense mathematical current, no engine current/derivative helpers.
    rawfactor=R*(1.+np.array([.15,.12,.2])*u*r[:3])
    assert np.all(rawfactor>=1.), "Smooth fixture must stay on fixed unity factor branch"
    factor=np.minimum(1.,rawfactor)
    for edge in ((0,2),(1,0),(2,0),(2,1)):w[edge]*=factor[edge[1]]
    argument=(np.full(6,20.)+w@r-1.5-z)/40
    assert np.all(argument>0), "Smooth fixture must not cross rectification"
    T=40*np.tanh(argument)
    gain=np.array([0,0,8,1,2,8]);tau=np.array([.3,.3,.3,.3,.6,.3])
    return np.r_[(T-r)/.03,(gain*r-z)/tau,(1.-R)/np.array([.15,.12,.2])-u*r[:3]*R]


def integrate64(y,h,u,kind):
    for _ in range(round(.032/h)):
        COST["oracle_ticks"]+=1
        if kind=="ssp":
            a=y+h*smooth_derivative(y,u);b=.75*y+.25*(a+h*smooth_derivative(a,u))
            y=(1/3)*y+(2/3)*(b+h*smooth_derivative(b,u))
        else:
            k1=smooth_derivative(y,u);k2=smooth_derivative(y+h/2*k1,u)
            k3=smooth_derivative(y+h/2*k2,u);k4=smooth_derivative(y+h*k3,u)
            y=y+h/6*(k1+2*k2+2*k3+k4)
    return y


@pytest.mark.parametrize("u",[np.zeros(3),np.array([.05,.1,.2])])
def test_float64_independent_smooth_global_third_order_not_production_f32_claim(u):
    y=np.r_[np.array([4,3,2,5,1,6.]),np.array([.1,.2,.15,.3,.2,.1]),np.ones(3)]
    reference=integrate64(y.copy(),.00003125,u,"rk4")
    errors=[np.linalg.norm(integrate64(y.copy(),h,u,"ssp")-reference) for h in (.004,.002,.001)]
    assert 6.5<errors[0]/errors[1]<10 and 6.5<errors[1]/errors[2]<10


def test_threshold_case_old_RK_stage_negative_all_SSP_stage_valid(monkeypatch):
    w=csr_matrix(np.array([[0,-1,0],[.000001,0,0],[0,0,0]],np.float32))
    n=ssp.SSPRecoveryResourceNetwork(w,np.full(3,.02,np.float32),np.ones(3,np.float32),np.full(3,7.5,np.float32),np.full(3,200,np.float32),
        adaptation_gain=8.,adaptation_tau_s=.3,dt=DT,device="cpu",cuda_implementation="packed",capture_steps=5,
        core_ids=np.array([0,1],np.int64),core_gain=0.,motor_ids=np.array([2],np.int64),motor_gains=np.ones(1,np.float32),motor_log_tau=np.zeros(1,np.float64),
        edge_ids=np.arange(2,dtype=np.int64),premotor_ids=np.array([0,1],np.int64),utilization=np.full(2,.05),recovery_tau=np.full(2,.15))
    def f(r):return (np.maximum(200*np.tanh((np.array([8.,10000.])+w.toarray()[:2,:2]@r-7.5)/200),0)-r)/.02
    r0=np.zeros(2);r1=r0+DT/2*f(r0);r2=r0+DT/2*f(r1);r3=r0+DT*f(r2)
    assert r2[0]<0 or r3[0]<0
    observed=[];method=n._cpu_factors
    def record(r,R):observed.append(r.numpy().copy());return method(r,R)
    monkeypatch.setattr(n,"_cpu_factors",record);n.advance(np.array([8,10000,0],np.float32),steps=1)
    assert len(observed)==3 and all(np.all(r>=0) for r in observed) and np.all(n.snapshot()["rate"]>=0)


def test_metadata_new_method_identity_neutral_not_old_RK_and_no_input_mutation():
    n=model(u=np.zeros(3));s=n.selection;m=ssp.resource_metadata(n.source_timescale_identity,s)
    assert m["schema"]==ssp.SCHEMA and ssp.resource_identity(m)==n.identity
    assert n.identity!=ssp.prior.resource_identity(ssp.prior.resource_metadata(n.source_timescale_identity,s))
    s["utilization"][0]=.2;assert n.selection["utilization"][0]==0
    for old_schema in (ssp.prior.SCHEMA,"flylab.m3-recurrent-resource-synapses.v1"):
        with pytest.raises(ValueError):ssp.checked_resource_metadata({**m,"schema":old_schema})


def test_explicit_unselected_zero_storage_and_neutral_factor_R_exact():
    args,kw=parameters();args[0].data[4]=np.float32(-0.);before=args[0].data.tobytes()
    n=ssp.SSPRecoveryResourceNetwork(*args,**kw,edge_ids=np.arange(4,dtype=np.int64),premotor_ids=np.arange(3,dtype=np.int64),utilization=np.zeros(3),recovery_tau=np.full(3,.15))
    n.advance(np.full(6,3,np.float32),steps=30)
    assert n.resource.numpy().tobytes()==np.ones(3,np.float64).tobytes()
    assert n.resource_factor.numpy().tobytes()==np.ones(3,np.float32).tobytes()
    assert args[0].data.tobytes()==before and n._source_weights.data.tobytes()==before


def test_disk_future_restore_atomic_corruption_and_old_checkpoint_reject(tmp_path):
    n=model();seeded(n);n.advance(np.full(6,3,np.float32),steps=7);saved=n.snapshot()
    StateStore.save(tmp_path/"cut",saved);n.advance(saved["drive"],steps=9);expected=n.snapshot()
    n.reset();n.restore(StateStore.load(tmp_path/"cut"));n.advance(saved["drive"],steps=9);equal(n.snapshot(),expected)
    before=n.snapshot()
    for key in ("rate","adaptation","synaptic_resource"):
        bad=copy.deepcopy(before);bad[key][0]=-.001
        with pytest.raises(ValueError):n.restore(bad)
        equal(n.snapshot(),before)
    bad=copy.deepcopy(before);bad["recurrent_resource"]["schema"]=ssp.prior.SCHEMA
    with pytest.raises(ValueError):n.restore(bad)
    equal(n.snapshot(),before)


@pytest.mark.parametrize("name,index",[("_ssp_buffers",0),("_ssp_buffers",1),*(("_resource_buffers",i) for i in range(4)),
                                      ("rate",None),("adaptation",None),("resource",None)])
def test_scratch_pointer_rejected_before_hostcopy_step(name,index,monkeypatch):
    n=model();values=getattr(n,name)
    if index is None:setattr(n,name,values.clone())
    else:values[index]=values[index].clone()
    tick=n.tick
    monkeypatch.setattr(n.torch.Tensor,"cpu",lambda *a,**k:pytest.fail("host copy after pointer mutation"))
    with pytest.raises(ValueError):n.advance(np.zeros(6,np.float32),steps=1)
    with pytest.raises(ValueError):n.readout([3])
    assert n.tick==tick


@pytest.mark.parametrize("key,value",[("dt",.02),("dt",0.),("core_gain",-1.),("adaptation_gain",-1.)])
def test_constructor_bound_rejection_before_native_construction(key,value):
    args,kw=parameters();kw[key]=value
    with pytest.raises(ValueError):ssp.SSPRecoveryResourceNetwork(*args,**kw,edge_ids=np.arange(4,dtype=np.int64),premotor_ids=np.arange(3,dtype=np.int64),utilization=np.full(3,.05),recovery_tau=np.full(3,.15))


@pytest.mark.parametrize("key",["rate","adaptation"])
@pytest.mark.parametrize("value",[-1e-6,float(-np.nextafter(np.float32(0),np.float32(1))),np.nan,np.inf])
def test_all_neuron_range_sticky_including_non_source_and_subnormal(key,value):
    n=model();state=seeded(n);getattr(n,key)[5]=value
    with pytest.raises(RuntimeError) as error:n.readout([3])
    assert error.value.resource_fault_flags&32
    getattr(n,key)[5]=1.
    with pytest.raises(RuntimeError):n.advance(state["drive"],steps=1)
    n.restore(state);assert n.resource_fault_flags==0
    n.advance(state["drive"],steps=1)


def test_signed_zero_positive_subnormal_state_and_single_packet(monkeypatch):
    n=model();state=seeded(n);state["rate"][0]=-0.;state["rate"][1]=np.nextafter(np.float32(0),np.float32(1));n.restore(state)
    copies=[0];original=n.torch.Tensor.cpu
    def host(value,*args,**kw):copies[0]+=1;return original(value,*args,**kw)
    monkeypatch.setattr(n.torch.Tensor,"cpu",host)
    n.advance_readout(state["drive"],[3,4],steps=1)
    assert copies[0]==1


def test_parameter_mutation_restore_import_and_reset_guards():
    n=model();n._resource_tau[0]=.2
    with pytest.raises(ValueError):n.advance(np.zeros(6,np.float32),steps=1)
    n=model();saved=n.snapshot();base={k:copy.deepcopy(v) for k,v in saved.items() if k not in ("recurrent_resource","synaptic_resource")}
    base["model_hash"]=n.source_timescale_identity;n.import_base_snapshot(base);assert n.resource_fault_flags==0
    base["rate"][0]=1.
    with pytest.raises(ValueError):n.import_base_snapshot(base)


def test_kernel_source_three_maps_no_old_RK4_bypass_and_precision_guard():
    source=ssp.resource_cuda_source()
    assert "rfe=add_exact" in source and "zfe=add_exact" in source and "ssp_neuron" in source
    assert "double fe=(1.0-q)*R+h*inverse" in source and "atomicOr(fault,32)" in source
    assert "double nr=R" in source and "inverseR64=1/tau64" in ssp.PRECISION
    assert "recovery_f32_to_f64" in source and "matching_tanh(activation[i])" in source
    assert "pointwise" in ssp.PRECISION and "canonical_inverse32" in ssp.PRECISION


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required RUN_E303_CUDA=1")
@pytest.mark.parametrize("u",[np.zeros(3),np.array([.05,0,.2])])
def test_cuda_fixed_tolerance_three_stage_capture_disk_future(tmp_path,u):
    cpu,gpu=model(u=u),model("cuda",u=u)
    for n in (cpu,gpu):seeded(n);n.set_muted([1]);n.advance(np.full(6,3,np.float32),steps=100)
    for key in ("rate","adaptation"):np.testing.assert_allclose(gpu.snapshot()[key],cpu.snapshot()[key],rtol=0,atol=2e-5)
    np.testing.assert_allclose(gpu.snapshot()["synaptic_resource"],cpu.snapshot()["synaptic_resource"],rtol=0,atol=2e-8)
    assert np.all(gpu.resource.cpu().numpy()[u==0]==1) and gpu.resource_fault_flags==0
    saved=gpu.snapshot();StateStore.save(tmp_path/"cut",saved)
    gpu.advance(saved["drive"],steps=10);future=gpu.snapshot();gpu.restore(StateStore.load(tmp_path/"cut"));gpu.graph=None
    gpu.advance(saved["drive"],steps=10);equal(gpu.snapshot(),future)
    gpu._ssp_buffers[0]=gpu._ssp_buffers[0].clone()
    with pytest.raises(ValueError):gpu.advance(saved["drive"],steps=1)


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required RUN_E303_CUDA=1")
def test_cuda_native32_negative_subnormal_and_final_buffer_guards():
    gpu=model("cuda");before=gpu.snapshot()
    gpu.adaptation[5]=float(-np.nextafter(np.float32(0),np.float32(1)))
    with pytest.raises(RuntimeError) as error:gpu.readout([3])
    assert error.value.resource_fault_flags&32
    gpu.restore(before);gpu.rate=gpu.rate.clone()
    with pytest.raises(ValueError):gpu.advance(np.ones(6,np.float32),steps=1)
