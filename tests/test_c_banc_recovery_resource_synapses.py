"""E299 bounded generated CPU checks; CUDA requires a root reservation."""
from __future__ import annotations

import copy
import ctypes
import json
import os
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.storage import StateStore
from tools import banc_m3_recovery_resource_synapses as rr
from tools import banc_m3_recurrent_resource_synapses as old
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork

CUDA_ENABLED = os.environ.get("RUN_E299_CUDA") == "1"
CUDA_RATE_ATOL, CUDA_RESOURCE_ATOL, CUDA_TICKS, DT = 2e-5, 2e-8, 100, .00025
COST = {"constructor_requested": 0, "constructor_completed": 0, "constructor_failed": 0,
        "constructor_wall_s": 0., "actual_constructor_requested": 0, "actual_constructor_completed": 0,
        "actual_constructor_failed": 0, "actual_constructor_wall_s": 0.,
        "step_requested": 0, "step_completed": 0, "step_failed": 0, "step_wall_s": 0.,
        "setup_step_requested": 0, "setup_step_completed": 0, "setup_step_failed": 0,
        "advance_requested": 0, "advance_completed": 0, "advance_failed": 0, "advance_wall_s": 0.,
        "logical_ticks": 0, "scalar_resource_ticks": 0, "capture_requested": 0,
        "capture_completed": 0, "capture_failed": 0, "solver_calls": 0,
        "body_calls": 0, "original_calls": 0, "CUDA_calls": 0,
        "direct_cuda_kernel_requested": 0, "direct_cuda_kernel_completed": 0,
        "direct_cuda_kernel_failed": 0, "direct_cuda_kernel_wall_s": 0.,
        "conversion_kernel_setup_requested": 0, "conversion_kernel_setup_completed": 0,
        "conversion_kernel_setup_failed": 0, "conversion_kernel_setup_wall_s": 0.}


def peak_rss_bytes():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    class Counters(ctypes.Structure):
        _fields_ = [("cb",ctypes.c_ulong),("faults",ctypes.c_ulong),
                    *[(name,ctypes.c_size_t) for name in ("peak","working","quota_peak_paged",
                        "quota_paged","quota_peak_nonpaged","quota_nonpaged","pagefile","peak_pagefile")]]
    counters= Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    api=ctypes.WinDLL("psapi",use_last_error=True).GetProcessMemoryInfo
    api.argtypes=(ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong)
    if not api(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.peak)


def checked_cuda_launch(network, callback):
    """Explicit diagnostic synchronization; never part of a control readout."""
    assert CUDA_ENABLED and network.device == "cuda"
    COST["direct_cuda_kernel_requested"] += 1
    began=time.perf_counter()
    try:
        with network._cupy.cuda.ExternalStream(network.torch.cuda.current_stream().cuda_stream):
            callback()
        network.torch.cuda.current_stream().synchronize()
        COST["direct_cuda_kernel_completed"] += 1
    except Exception:
        COST["direct_cuda_kernel_failed"] += 1
        raise
    finally:COST["direct_cuda_kernel_wall_s"] += time.perf_counter()-began


@pytest.fixture(scope="module", autouse=True)
def aggregate_cost():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    wall, process = time.perf_counter(), time.process_time()
    yield
    COST.update(wall_s=time.perf_counter()-wall, process_s=time.process_time()-process)
    COST.update(peak_rss_bytes=peak_rss_bytes(),torch_threads=torch.get_num_threads())
    torch.set_num_threads(previous)
    print("E299_GENERATED_COST "+json.dumps(COST, sort_keys=True))
    assert COST["wall_s"] <= 120
    assert COST["logical_ticks"]+COST["scalar_resource_ticks"] <= (2000 if CUDA_ENABLED else 10000)
    assert COST["peak_rss_bytes"] <= (2 if CUDA_ENABLED else 1)*1024**3 and COST["torch_threads"] == 1


@pytest.fixture(autouse=True)
def instrument(monkeypatch):
    init, base_init, adaptive_init = rr.RecoveryResourceNetwork.__init__, MotorTimescaleNetwork.__init__, AdaptiveRateNetwork.__init__
    step, base_step, advance = rr.RecoveryResourceNetwork._step, AdaptiveRateNetwork._step, rr.RecoveryResourceNetwork._advance
    base_advance = MotorTimescaleNetwork._advance
    def init_cost(method, self, args, kw):
        assert args[0].shape[0] <= 64 and args[0].nnz <= 256
        COST["constructor_requested"] += 1
        began = time.perf_counter()
        self._e299_setup = True
        if kw["device"] == "cuda":
            assert CUDA_ENABLED
            COST["CUDA_calls"] += 1
            COST["capture_requested"] += 1
        try:
            result = method(self, *args, **kw)
            COST["constructor_completed"] += 1
            if kw["device"] == "cuda": COST["capture_completed"] += 1
            return result
        except Exception:
            COST["constructor_failed"] += 1
            if kw["device"] == "cuda": COST["capture_failed"] += 1
            raise
        finally:
            COST["constructor_wall_s"] += time.perf_counter()-began
            self._e299_setup = False
    def new_init(self, *args, **kw): return init_cost(init, self, args, kw)
    def original_init(self, *args, **kw):
        return init_cost(base_init, self, args, kw) if type(self) is MotorTimescaleNetwork else base_init(self, *args, **kw)
    def actual_init(self, *args, **kw):
        COST["actual_constructor_requested"] += 1
        began = time.perf_counter()
        try:
            result = adaptive_init(self, *args, **kw)
            COST["actual_constructor_completed"] += 1
            return result
        except Exception:
            COST["actual_constructor_failed"] += 1
            raise
        finally: COST["actual_constructor_wall_s"] += time.perf_counter()-began
    def step_cost(method, self):
        COST["step_requested"] += 1
        setup = getattr(self, "_e299_setup", False)
        if setup: COST["setup_step_requested"] += 1
        began = time.perf_counter()
        try:
            result = method(self)
            COST["step_completed"] += 1
            if setup: COST["setup_step_completed"] += 1
            return result
        except Exception:
            COST["step_failed"] += 1
            if setup: COST["setup_step_failed"] += 1
            raise
        finally: COST["step_wall_s"] += time.perf_counter()-began
    def new_step(self): return step_cost(step, self)
    def original_step(self):
        return step_cost(base_step, self) if type(self) is MotorTimescaleNetwork else base_step(self)
    def advance_cost(method, self, *args, **kw):
        COST["advance_requested"] += 1
        began, tick = time.perf_counter(), self.tick
        try:
            result = method(self, *args, **kw)
            COST["advance_completed"] += 1
            return result
        except Exception:
            COST["advance_failed"] += 1
            raise
        finally:
            COST["advance_wall_s"] += time.perf_counter()-began
            COST["logical_ticks"] += self.tick-tick
    def new_advance(self, *args, **kw): return advance_cost(advance, self, *args, **kw)
    def original_advance(self, *args, **kw):
        return advance_cost(base_advance, self, *args, **kw) if type(self) is MotorTimescaleNetwork else base_advance(self, *args, **kw)
    monkeypatch.setattr(rr.RecoveryResourceNetwork, "__init__", new_init)
    monkeypatch.setattr(MotorTimescaleNetwork, "__init__", original_init)
    monkeypatch.setattr(AdaptiveRateNetwork, "__init__", actual_init)
    monkeypatch.setattr(rr.RecoveryResourceNetwork, "_step", new_step)
    monkeypatch.setattr(AdaptiveRateNetwork, "_step", original_step)
    monkeypatch.setattr(rr.RecoveryResourceNetwork, "_advance", new_advance)
    monkeypatch.setattr(MotorTimescaleNetwork, "_advance", original_advance)


def parameters(device="cpu"):
    w = np.zeros((6, 6), np.float32)
    w[0, 2], w[1, 0], w[2, 0], w[2, 1] = .07, -.08, .04, .12
    w[3, 0], w[3, 1], w[4, 0], w[4, 2], w[5, 3] = .15, -.1, .09, -.04, -.02
    args = (csr_matrix(w), np.full(6, .03, np.float32), np.ones(6, np.float32),
            np.full(6, 1.5, np.float32), np.full(6, 40., np.float32))
    kw = {"adaptation_gain": 8., "adaptation_tau_s": .3, "dt": DT, "device": device,
          "capture_steps": 5, "cuda_implementation": "packed", "core_ids": np.array([0, 1], np.int64),
          "core_gain": 0., "motor_ids": np.array([3, 4], np.int64), "motor_gains": np.array([1., 2.], np.float32),
          "motor_log_tau": np.array([0., np.log(2.)], np.float64)}
    return args, kw


def model(device="cpu", u=None):
    args, kw = parameters(device)
    return rr.RecoveryResourceNetwork(*args, **kw, edge_ids=np.arange(4, dtype=np.int64),
        premotor_ids=np.arange(3, dtype=np.int64), utilization=np.array([.05, .1, .2]) if u is None else u,
        recovery_tau=np.array([.15, .12, .2]))


def same(a, b):
    assert old._equal(a, b)


def seeded(n):
    state = n.snapshot()
    state.update(tick=17, rate=np.array([4., 3., 2., 5., 1., 6.], np.float32),
        adaptation=np.array([.1, .2, .15, .3, .2, .1], np.float32),
        drive=np.array([2.8, 3., 2.5, 3.1, 3., 2.7], np.float32))
    state["synaptic_resource"] = np.array([.5, .7, .9])
    state["synaptic_resource"][n.selection["utilization"] == 0] = 1.
    n.restore(state)
    return state


def independent_current(w, r, mask, selection, R):
    def ftz(v):
        v = np.float32(v)
        return np.copysign(np.float32(0), v) if abs(v) < np.finfo(np.float32).tiny else v
    chosen = dict(zip(selection["edge_ids"], selection["edge_source_slots"], strict=True))
    factors = []
    for i, pre in enumerate(selection["source_ids"]):
        a = float(R[i])*(1.+(float(selection["recovery_tau"][i])*float(selection["utilization"][i]))*float(r[pre]))
        factors.append(np.float32(1. if selection["utilization"][i] == 0 else min(1., a)))
    result = np.zeros(w.shape[0], np.float32)
    for row in range(len(result)):
        lane = np.zeros(32, np.float32)
        for j in range(32):
            for edge in range(w.indptr[row]+j, w.indptr[row+1], 32):
                pre = w.indices[edge]
                value = np.float32(r[pre]*mask[pre])
                if edge in chosen and factors[chosen[edge]] != np.float32(1):
                    value = ftz(ftz(value)*ftz(factors[chosen[edge]]))
                value = ftz(ftz(w.data[edge])*ftz(value))
                lane[j] = ftz(ftz(lane[j])+ftz(value))
        for offset in (16, 8, 4, 2, 1):
            before = lane.copy()
            for j in range(32-offset): lane[j] = ftz(ftz(before[j])+ftz(before[j+offset]))
        result[row] = lane[0]
    return result


def independent_tick(n, state):
    r0, z0, R0 = (state[k].copy() for k in ("rate", "adaptation", "synaptic_resource"))
    r, z, R = r0, z0, R0
    kr, kz, kR, stages = [], [], [], []
    w, tau, a, threshold, cap = parameters()[0]
    coefficient = a/cap
    gain, ztau = np.array([0., 0., 8., 1., 2., 8.], np.float32), np.array([.3, .3, .3, .3, .6, .3], np.float32)
    selection = n.selection
    for stage in range(4):
        stages.append((r.copy(), R.copy()))
        current = independent_current(w, r, state["output_mask"], selection, R)
        target = np.maximum(cap*np.tanh(coefficient*(state["drive"]+current-threshold-z)), np.float32(0))
        dr, dz = (target-r)/tau, (gain*r-z)/ztau
        dR = np.zeros(R.shape, np.float64)
        for i, pre in enumerate(selection["source_ids"]):
            if selection["utilization"][i] > 0:
                dR[i] = (1.-R[i])/selection["recovery_tau"][i]-(selection["utilization"][i]*np.float64(r[pre]))*R[i]
        kr.append(dr); kz.append(dz); kR.append(dR)
        if stage < 3:
            step = n.dt*(.5 if stage < 2 else 1)
            r, z, R = r0+np.float32(step)*dr, z0+np.float32(step)*dz, R0+step*dR
    result = copy.deepcopy(state)
    for key, base, values in (("rate", r0, kr), ("adaptation", z0, kz), ("synaptic_resource", R0, kR)):
        step = n.dt/6 if key == "synaptic_resource" else np.float32(n.dt/6)
        result[key] = base+step*((values[0]+2*values[1])+2*values[2]+values[3])
    result["tick"] += 1
    return result, stages


def test_identity_selection_owned_and_old_checkpoint_rejected():
    args, kw = parameters()
    edges, pop, u, tau = np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64), np.array([.05, .1, .2]), np.array([.15, .12, .2])
    selection = rr.checked_selection(args[0], edges, pop, u, tau, kw["motor_ids"])
    metadata = rr.resource_metadata("a"*64, selection)
    same(rr.checked_resource_metadata(metadata), metadata)
    assert rr.resource_identity(metadata) != old.resource_identity(old.resource_metadata("a"*64, selection))
    with pytest.raises(ValueError): rr.checked_resource_metadata(old.resource_metadata("a"*64, selection))
    n = rr.RecoveryResourceNetwork(*args, **kw, edge_ids=edges, premotor_ids=pop, utilization=u, recovery_tau=tau)
    before = n.snapshot()
    assert rr.resource_identity(before["recurrent_resource"]) == n.identity
    assert before["recurrent_resource"]["source_model_hash"] == n.source_timescale_identity
    args[0].data[:] = 99; args[1][:] = 1; edges[:] = 0; pop[:] = 0; u[:] = .5; tau[:] = .5
    n.selection["utilization"][:] = 0
    same(n.snapshot(), before)
    bad = copy.deepcopy(before)
    bad["recurrent_resource"] = old.resource_metadata(n.source_timescale_identity, selection)
    bad["model_hash"] = old.resource_identity(bad["recurrent_resource"])
    with pytest.raises(ValueError): n.restore(bad)
    same(n.snapshot(), before)


def test_constant_rate_unity_recovery_analytic_and_incremental_gain():
    rate = np.array([100., 0., 4.], np.float32)
    sources = np.arange(3, dtype=np.int64)
    u, tau, R = np.array([.05, .05, 0.]), np.full(3, .15), np.ones(3)
    equilibrium = 1/(1+(tau*u)*rate.astype(np.float64))
    for _ in range(100):
        same(rr.resource_factors(rate, R, sources, u, tau), np.ones(3, np.float32))
        def f(value): return rr.resource_derivative(rate, value, sources, u, tau)
        k1 = f(R); k2 = f(R+DT/2*k1); k3 = f(R+DT/2*k2); k4 = f(R+DT*k3)
        R += DT/6*((k1+2*k2)+2*k3+k4)
        COST["scalar_resource_ticks"] += 1
    expected = equilibrium+(1-equilibrium)*np.exp(-(1/tau+u*rate)*100*DT)
    np.testing.assert_allclose(R, expected, atol=2e-13, rtol=0)
    factor = rr.resource_factors(rate, np.array([.5, .5, 1.]), sources, u, tau)
    assert factor[0] == np.float32(.875) and factor[1] == np.float32(.5) and factor[2] == 1
    assert .5*(1+2*.15*.05*100) == 1.25


@pytest.mark.parametrize("weight,rate,mask,R,u",[(.4,20.,1.,.3,.1),(-.4,20.,1.,.3,.1),
    (-.5,20.,0.,.2,.1),(1.,-0.,1.,1.,.1),(1.,0.,1.,1.,0.),
    (1.,np.nextafter(np.float32(0),np.float32(1)),1.,1.,.1)])
def test_literal_scalar_ftz_mask_signed_zero_and_unity(weight, rate, mask, R, u):
    def ftz(x):
        x = np.float32(x)
        return np.copysign(np.float32(0), x) if abs(x) < np.finfo(np.float32).tiny else x
    factor = np.float32(1 if u == 0 else min(1.,np.float64(R)*(1.+(np.float64(.15)*np.float64(u))*np.float64(np.float32(rate)))))
    value = np.float32(np.float32(rate)*np.float32(mask))
    if factor != 1: value = ftz(ftz(value)*ftz(factor))
    expected = ftz(ftz(np.float32(weight))*ftz(value))
    actual = rr.resource_edge_value(weight,rate,mask,R,u,.15)
    assert actual.tobytes() == expected.tobytes()


def test_packed_current_shared_sources_mixed_neutral_and_mask_evolution():
    n = model(u=np.array([0., .1, .2]))
    state = seeded(n)
    n.set_muted([1]); state = n.snapshot()
    same(n.resource_current_oracle(), independent_current(parameters()[0][0], state["rate"], state["output_mask"], n.selection, state["synaptic_resource"]))
    altered = state["synaptic_resource"].copy(); altered[1] = .2
    alternate = n.resource_current_oracle(resource=altered)
    current = n.resource_current_oracle()
    same(alternate[3:], current[3:])
    n.advance(state["drive"], steps=1)
    assert n.resource.numpy()[0] == 1 and n.resource.numpy()[1] != state["synaptic_resource"][1]


@pytest.mark.parametrize("kind",["rate_shape","rate_dtype","rate_nan","R_shape","R_dtype","R_nan"])
def test_current_oracle_rejects_malformed_inputs_without_state_mutation(kind):
    n=model();before=seeded(n);r=before["rate"].copy();R=before["synaptic_resource"].copy()
    if kind=="rate_shape":r=r[:3]
    if kind=="rate_dtype":r=r.astype(np.float64)
    if kind=="rate_nan":r[5]=np.nan
    if kind=="R_shape":R=R[:2]
    if kind=="R_dtype":R=R.astype(np.float32)
    if kind=="R_nan":R[1]=np.nan
    with pytest.raises(ValueError):n.resource_current_oracle(rate=r,resource=R)
    same(n.snapshot(),before)


def test_independent_coupled_stage_rk4_and_disk_future(tmp_path):
    n = model(); state = seeded(n)
    expected, stages = independent_tick(n, state)
    assert not np.array_equal(stages[0][0], stages[1][0])
    n.advance(state["drive"], steps=1)
    actual = n.snapshot()
    for key in ("rate", "adaptation"): np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual["synaptic_resource"], expected["synaptic_resource"], atol=2e-15, rtol=0)
    StateStore.save(tmp_path/"state", actual)
    other = model(); other.restore(StateStore.load(tmp_path/"state"))
    same(other.snapshot(), actual)
    pointers = dict(n._recovery_pointers)
    for network in (n, other): network.advance_readout(state["drive"], [0, 3, 4], steps=25)
    same(n.snapshot(), other.snapshot())
    assert all(getattr(n, key).data_ptr() == value for key, value in pointers.items())
    n.reset(); assert n.tick == 0 and np.all(n.resource.numpy() == 1) and n.resource_fault_flags == 0


def test_explicit_unselected_zero_neutral_bitexact_and_base_import():
    args, kw = parameters(); args[0].data[4] = np.float32(0.); args[0].data[8] = np.float32(-0.)
    before = tuple(v.copy() for v in (args[0].indptr,args[0].indices,args[0].data))
    base = MotorTimescaleNetwork(*args, **kw)
    n = rr.RecoveryResourceNetwork(*args, **kw, edge_ids=np.arange(4,dtype=np.int64),
        premotor_ids=np.arange(3,dtype=np.int64), utilization=np.zeros(3), recovery_tau=np.full(3,.15))
    n.import_base_snapshot(base.snapshot())
    assert n.source_timescale_identity == base.identity
    for control in range(3):
        for network in (base,n):
            network.set_muted([0,3] if control == 1 else [])
            network.advance(np.full(6,3.+.1*control,np.float32), steps=10)
        for key in ("rate","adaptation","drive","output_mask","tick"): same(base.snapshot()[key],n.snapshot()[key])
        assert n.resource_fault_flags == 0 and np.all(n.resource.numpy() == 1)
    for actual, expected in zip((args[0].indptr,args[0].indices,args[0].data),before,strict=True): same(actual,expected)
    with pytest.raises(ValueError): n.import_base_snapshot(base.snapshot())
    base.reset(); broken=base.snapshot(); broken["rate"][0]=1
    with pytest.raises(ValueError): n.import_base_snapshot(broken)
    args[0].data[0]=0
    with pytest.raises(ValueError): rr.checked_selection(args[0],np.arange(4,dtype=np.int64),np.arange(3,dtype=np.int64),np.zeros(3),np.full(3,.15),kw["motor_ids"])


@pytest.mark.parametrize("kind",["resource_dtype","resource_shape","nan","range","u0","tick0","active_negative","inactive_too_negative","adaptation","mask","identity","metadata","precision","source","extra"])
def test_invalid_restore_atomic_and_does_not_clear_fault(kind):
    n=model(u=np.array([0.,.1,.2])); before=seeded(n); bad=copy.deepcopy(before)
    if kind=="resource_dtype":bad["synaptic_resource"]=bad["synaptic_resource"].astype(np.float32)
    if kind=="resource_shape":bad["synaptic_resource"]=bad["synaptic_resource"][:2]
    if kind=="nan":bad["synaptic_resource"][1]=np.nan
    if kind=="range":bad["synaptic_resource"][1]=1.000001
    if kind=="u0":bad["synaptic_resource"][0]=.9
    if kind=="tick0":bad["tick"]=0
    if kind=="active_negative":bad["rate"][1]=-1e-6
    if kind=="inactive_too_negative":bad["rate"][0]=-1e-3
    if kind=="adaptation":bad["adaptation"][1]=-1
    if kind=="mask":bad["output_mask"][1]=.5
    if kind=="identity":bad["model_hash"]="0"*64
    if kind=="metadata":bad["recurrent_resource"]["weights32"][0]*=-1
    if kind=="precision":bad["recurrent_resource"]["precision"]+="wrong"
    if kind=="source":bad["recurrent_resource"]["source_model_hash"]="0"*64
    if kind=="extra":bad["extra"]=1
    n._resource_fault.fill_(rr.FAULT_ACTIVE_RATE)
    with pytest.raises(ValueError):n.restore(bad)
    assert n.resource_fault_flags == rr.FAULT_ACTIVE_RATE
    n.restore(before);same(n.snapshot(),before)


def test_restore_stages_all_arrays_before_writing_and_inactive_native_negative(monkeypatch):
    n=model(u=np.array([0.,.1,.2]));before=seeded(n);bad=copy.deepcopy(before);bad["rate"]+=1
    original=n.torch.as_tensor;calls=[0]
    def allocation(*args,**kw):
        calls[0]+=1
        if calls[0]==5:raise MemoryError("last staging allocation")
        return original(*args,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(n.torch,"as_tensor",allocation)
        with pytest.raises(MemoryError):n.restore(bad)
    same(n.snapshot(),before)
    before["rate"][0]=np.float32(-1e-5);n.restore(before)
    same(n.snapshot()["rate"],before["rate"])


@pytest.mark.parametrize("key",["resource","resource_factor","_resource_fault","_resource_u","_resource_tau","rate","drive","_cell_inverse_tau"])
def test_pointer_changes_rejected_before_readout_or_advance(key):
    n=model();setattr(n,key,getattr(n,key).clone())
    with pytest.raises(ValueError):n.snapshot()
    with pytest.raises(ValueError):n.advance(np.zeros(6,np.float32),steps=1)


@pytest.mark.parametrize("index",range(4))
def test_resource_scratch_pointer_change_rejected_before_host_copy_and_step(index,monkeypatch):
    n=model();n._resource_buffers[index]=n._resource_buffers[index].clone();tick=n.tick
    def forbidden(*args,**kw):pytest.fail("Unexpected host copy after scratch replacement")
    monkeypatch.setattr(n.torch.Tensor,"cpu",forbidden)
    for call in (lambda:n.advance(np.zeros(6,np.float32),steps=1),
                 lambda:n.advance_readout(np.zeros(6,np.float32),[3,4],steps=1),
                 lambda:n.readout([3,4])):
        with pytest.raises(ValueError,match="scratch buffer"):call()
    assert n.tick==tick


def test_parameter_bytes_and_original_source_bytes_rejected():
    n=model();n._resource_u[0]=.1
    with pytest.raises(ValueError):n.advance(np.zeros(6,np.float32),steps=1)
    n=model();n._source_weights.data[-1]*=2
    with pytest.raises(ValueError):n.snapshot()
    n=model();n.weights.values()[-1]*=2
    with pytest.raises(ValueError):n.advance(np.zeros(6,np.float32),steps=1)
    n=model();n.weights=n.weights.clone()
    with pytest.raises(ValueError):n.snapshot()


@pytest.mark.parametrize("kind,flags",[("negative",rr.FAULT_ACTIVE_RATE),("negative_subnormal",rr.FAULT_ACTIVE_RATE),("nan",rr.FAULT_ACTIVE_RATE),
    ("Rnegative",rr.FAULT_RESOURCE),("Rlarge",rr.FAULT_RESOURCE),("Rnan",rr.FAULT_RESOURCE),
    ("u0R",rr.FAULT_NEUTRAL_RESOURCE)])
def test_stage_fault_sticky_blocks_advance_readout_snapshot_until_valid_reset(kind,flags):
    n=model(u=np.array([0.,.1,.2]));before=seeded(n)
    if kind=="negative":n.rate[1]=-1e-6
    if kind=="negative_subnormal":n.rate[1]=float(-np.nextafter(np.float32(0),np.float32(1)))
    if kind=="nan":n.rate[1]=np.nan
    if kind=="Rnegative":n.resource[1]=-.1
    if kind=="Rlarge":n.resource[1]=1.1
    if kind=="Rnan":n.resource[1]=np.nan
    if kind=="u0R":n.resource[0]=.9
    with pytest.raises(RuntimeError) as caught:n.advance_readout(before["drive"],[3,4],steps=1)
    assert caught.value.resource_fault_flags & flags
    n.rate[1]=3;n.resource.copy_(n.torch.as_tensor(before["synaptic_resource"]))
    for call in (lambda:n.readout([3,4]),n.snapshot,lambda:n.advance(before["drive"],steps=1)):
        with pytest.raises(RuntimeError):call()
    n.reset();assert n.resource_fault_flags==0
    n.advance(np.full(6,3.,np.float32),steps=1)


def test_internal_rk_rate_fault_detected_even_when_final_outputs_unobserved(monkeypatch):
    n=model();state=seeded(n);calls=[0];actual=rr.packed_current_oracle
    def current(*args,**kw):
        value=actual(*args,**kw);calls[0]+=1
        if calls[0]==1:value[0]=np.nan
        return value
    monkeypatch.setattr(rr,"packed_current_oracle",current)
    with pytest.raises(RuntimeError) as error:n.advance_readout(state["drive"],[3,4],steps=1)
    assert error.value.resource_fault_flags & rr.FAULT_ACTIVE_RATE
    assert calls[0]==1 and n.tick==17


def test_final_resource_fault_detected_and_never_saved(monkeypatch):
    n=model();state=seeded(n);actual=n._cpu_factors;calls=[0]
    def factors(r,R):
        result=actual(r,R);calls[0]+=1
        if calls[0]==4:n.resource[0]=1.2
        return result
    monkeypatch.setattr(n,"_cpu_factors",factors)
    with pytest.raises(RuntimeError) as error:n.advance_readout(state["drive"],[3,4],steps=1)
    assert error.value.resource_fault_flags & rr.FAULT_FINAL_RESOURCE
    with pytest.raises(RuntimeError):n.snapshot()
    n.restore(state);assert n.resource_fault_flags==0


def test_complete_native_validation_and_one_packet_host_publication(monkeypatch):
    n=model();state=seeded(n)
    original=n.torch.Tensor.cpu;copies=[0]
    def host(value,*args,**kw):
        copies[0]+=1
        return original(value,*args,**kw)
    monkeypatch.setattr(n.torch.Tensor,"cpu",host)
    result=n.advance_readout(state["drive"],[3,4],steps=1)
    assert result.shape==(2,) and result.dtype==np.float32
    assert copies[0]==1
    with monkeypatch.context() as patch:
        patch.setattr(n,"_step",lambda:None)
        n.rate[5]=np.nan
        with pytest.raises(RuntimeError,match="complete-network"):n.advance_readout(state["drive"],[3,4],steps=1)
    assert copies[0]==2


def test_kernel_factor_once_per_stage_unity_order_flags_and_guards():
    source=rr.resource_cuda_source()
    assert "resource[i]*(1.0+(tau[i]*u[i])*recovery_f32_to_f64(r))" in source
    assert 'asm("cvt.f64.f32 %0, %1;"' in source and 'asm("cvt.rn.f32.f64 %0, %1;"' in source
    assert "magnitude >= 0x7f800000u" in source and "magnitude != 0u" in source
    assert "factor[s] == 1.0f" in source and "masked[item & 0x3ffff] * factor[s]" in source
    assert "atomicOr(fault, flags)" in source and "flags |= 16" in source
    assert "cell_inverse_tau[i]" in source and "const double*" in source
    args,kw=parameters();kw["dt"]=.0011
    with pytest.raises(ValueError):rr.RecoveryResourceNetwork(*args,**kw,edge_ids=np.arange(4,dtype=np.int64),
        premotor_ids=np.arange(3,dtype=np.int64),utilization=np.full(3,.5),recovery_tau=np.full(3,.02))


def test_positive_subnormal_input_and_subnormal_factor_literal_cpu_semantics():
    smallest=np.nextafter(np.float32(0),np.float32(1))
    rate=np.array([smallest,0.,-0.],np.float32);sources=np.arange(3,dtype=np.int64)
    u,tau,R=np.array([.05,.1,.2]),np.array([.15,.12,.2]),np.array([1.,float(smallest),1.])
    factor=rr.resource_factors(rate,R,sources,u,tau)
    same(factor,np.array([1.,smallest,1.],np.float32))
    derivative=rr.resource_derivative(rate,R,sources,u,tau)
    assert derivative[0].tobytes()==np.float64(-(.05*np.float64(smallest))).tobytes()
    with pytest.raises(RuntimeError):rr.resource_factors(-rate,R,sources,u,tau)


def test_readout_endpoint_rejects_negative_subnormal_without_step():
    n=model();tick=n.tick
    n.rate[0]=float(-np.nextafter(np.float32(0),np.float32(1)))
    with pytest.raises(RuntimeError) as error:n.readout([3,4])
    assert error.value.resource_fault_flags & rr.FAULT_ACTIVE_RATE
    assert n.tick==tick


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required: RUN_E299_CUDA=1")
def test_cuda_fixed_coupled_oracle_cut_disk_capture_and_single_packet(tmp_path,monkeypatch):
    cpu,gpu=model(),model("cuda")
    for n in (cpu,gpu):seeded(n);n.set_muted([1])
    drive=np.full(6,3.,np.float32)
    for n in (cpu,gpu):n.advance(drive,steps=CUDA_TICKS)
    for key in ("rate","adaptation"):np.testing.assert_allclose(gpu.snapshot()[key],cpu.snapshot()[key],atol=CUDA_RATE_ATOL,rtol=0)
    np.testing.assert_allclose(gpu.snapshot()["synaptic_resource"],cpu.snapshot()["synaptic_resource"],atol=CUDA_RESOURCE_ATOL,rtol=0)
    saved=gpu.snapshot();StateStore.save(tmp_path/"gpu",saved);gpu.reset();gpu.restore(StateStore.load(tmp_path/"gpu"))
    other=model("cuda");other.restore(saved);other.graph=None
    for n in (gpu,other):n.advance(drive,steps=10)
    same(gpu.snapshot(),other.snapshot())
    original=gpu.torch.Tensor.cpu;copies=[0]
    def host(value,*args,**kw):copies[0]+=1;return original(value,*args,**kw)
    monkeypatch.setattr(gpu.torch.Tensor,"cpu",host)
    gpu.advance_readout(drive,[3,4],steps=5)
    assert copies[0]==1
    gpu.rate[0]=-1e-6
    with pytest.raises(RuntimeError):gpu.advance_readout(drive,[3,4],steps=5)
    assert copies[0]==2 and gpu.resource_fault_flags & rr.FAULT_ACTIVE_RATE


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required: RUN_E299_CUDA=1")
def test_cuda_neutral_captured_original_bitexact_and_final_fault():
    args,kw=parameters("cuda");args[0].data[4]=np.float32(0.);args[0].data[8]=np.float32(-0.)
    base=MotorTimescaleNetwork(*args,**kw)
    n=rr.RecoveryResourceNetwork(*args,**kw,edge_ids=np.arange(4,dtype=np.int64),
        premotor_ids=np.arange(3,dtype=np.int64),utilization=np.zeros(3),recovery_tau=np.full(3,.15))
    n.import_base_snapshot(base.snapshot())
    for control in range(3):
        for network in (base,n):network.set_muted([0] if control==1 else []);network.advance(np.full(6,3.,np.float32),steps=10)
        for key in ("rate","adaptation","drive","output_mask","tick"):same(base.snapshot()[key],n.snapshot()[key])
    n.resource[0]=1.1
    with pytest.raises(RuntimeError) as error:n.readout([3,4])
    assert error.value.resource_fault_flags & rr.FAULT_FINAL_RESOURCE
    n.reset();assert n.resource_fault_flags==0
    # Directly exercise the actual device finish guard, even with valid stage R.
    stage_R=n.resource.clone()
    n.resource[0]=1.2
    k1,k2,k3=n._resource_buffers[1:]
    for k in (k1,k2,k3):k.zero_()
    checked_cuda_launch(n,lambda:n._recovery_launch("recovery_finish",(n._resource_sources,n.rate,stage_R,
        n._resource_u,n._resource_tau,k1,k2,k3,n.resource,n._resource_fault),n.dt/6))
    with pytest.raises(RuntimeError) as error:n.readout([3,4])
    assert error.value.resource_fault_flags & rr.FAULT_FINAL_RESOURCE


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required: RUN_E299_CUDA=1")
def test_cuda_scratch_pointer_changes_rejected_without_host_copy(monkeypatch):
    n=model("cuda");tick=n.tick
    def forbidden(*args,**kw):pytest.fail("Host copy after captured scratch replacement")
    monkeypatch.setattr(n.torch.Tensor,"cpu",forbidden)
    for name,index in (("_resource_buffers",0),("_fused_buffers",0),("_fused_buffers",5),("_fused_buffers",10)):
        values=getattr(n,name);original=values[index];values[index]=original.clone()
        for call in (lambda:n.advance(np.zeros(6,np.float32),steps=1),lambda:n.readout([3,4])):
            with pytest.raises(ValueError,match="scratch buffer"):call()
        values[index]=original
        n._assert_pointers()
    assert n.tick==tick


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required: RUN_E299_CUDA=1")
def test_cuda_actual_factor_negative_subnormal_positive_conversion_and_subnormal_output():
    n=model("cuda");smallest=np.nextafter(np.float32(0),np.float32(1))
    n.rate.copy_(n.torch.as_tensor(np.array([-smallest,0.,0.,0.,0.,0.],np.float32),device="cuda"))
    def factor():n._recovery_launch("recovery_factor",(n._resource_sources,n.rate,n.resource,
        n._resource_u,n._resource_tau,n.resource_factor,n._resource_fault))
    checked_cuda_launch(n,factor)
    with pytest.raises(RuntimeError) as error:n.readout([3,4])
    assert error.value.resource_fault_flags & rr.FAULT_ACTIVE_RATE
    n.reset()
    r=np.array([smallest,0.,-0.,0.,0.,0.],np.float32)
    R=np.array([1.,float(smallest),1.],np.float64)
    n.rate.copy_(n.torch.as_tensor(r,device="cuda"));n.resource.copy_(n.torch.as_tensor(R,device="cuda"))
    checked_cuda_launch(n,factor)
    same(n.resource_factor.cpu().numpy(),rr.resource_factors(r,R,n.selection["source_ids"],
        n.selection["utilization"],n.selection["recovery_tau"]))
    k,nextR=n._resource_buffers[1],n._resource_buffers[0]
    checked_cuda_launch(n,lambda:n._recovery_launch("recovery_stage",(n._resource_sources,n.rate,
        n.resource,n.resource,n._resource_u,n._resource_tau,k,nextR),n.dt/2))
    expected=rr.resource_derivative(r,R,n.selection["source_ids"],n.selection["utilization"],n.selection["recovery_tau"])
    assert k.cpu().numpy()[0].tobytes()==expected[0].tobytes() and expected[0]<0
    assert n.resource_fault_flags==0


@pytest.mark.skipif(not CUDA_ENABLED,reason="Root reservation required: RUN_E299_CUDA=1")
def test_cuda_explicit_nonftz_float_conversions_are_byteexact():
    n=model("cuda");cp=n._cupy
    source=rr.resource_cuda_source()+r'''
extern "C" __global__ void recovery_conversion_test(const float* src32,const double* src64,
    double* dst64,float* dst32,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {dst64[i]=recovery_f32_to_f64(src32[i]);dst32[i]=recovery_f64_to_f32(src64[i]);}
}
'''
    COST["conversion_kernel_setup_requested"]+=1;began=time.perf_counter()
    try:
        kernel=cp.RawKernel(source,"recovery_conversion_test",options=("--fmad=false","--ftz=true"))
        kernel.compile();COST["conversion_kernel_setup_completed"]+=1
    except Exception:
        COST["conversion_kernel_setup_failed"]+=1
        raise
    finally:COST["conversion_kernel_setup_wall_s"]+=time.perf_counter()-began
    src32=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0x7f7fffff],np.uint32).view(np.float32)
    src64=np.array([0.,-0.,2.**-149,-2.**-149,2.**-150,-2.**-150,float(src32[4]),float(src32[5])],np.float64)
    in32,in64=cp.asarray(src32),cp.asarray(src64);out64,out32=cp.empty(8,cp.float64),cp.empty(8,cp.float32)
    checked_cuda_launch(n,lambda:kernel((1,),(32,),(in32,in64,out64,out32,np.int32(8))))
    same(cp.asnumpy(out64),src32.astype(np.float64))
    same(cp.asnumpy(out32),src64.astype(np.float32))
