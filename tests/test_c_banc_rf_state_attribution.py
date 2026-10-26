"""Generated E286 decoder/state axes, ownership and fail-closed accounting."""
import copy
import json
import time
import weakref

import numpy as np
import pytest

from tests import test_c_banc_rf_elastic as prior_tests
from tools import verify_c_banc_m3_rf_state_attribution as run

COST = {"constructors_requested":0,"constructors_completed":0,"forward_requested":0,"forward_completed":0,
        "copy_requested":0,"copy_completed":0,"forward_ticks":0,"derivative_ticks":0,"solver_calls":0}


@pytest.fixture(scope="module",autouse=True)
def cost():
    start=time.perf_counter()
    yield
    elapsed=time.perf_counter()-start
    assert elapsed<60
    print("e286_generated_cost", {**COST,"wall_s":elapsed,"original":0,"CUDA":0,"fullsize_engine":0})


def mapping():
    n,m=6,2
    ids=np.array(run.Q0_IDS,np.int64)
    member=np.array([0,1,1,1,0,1],np.int64)
    p={"motor_ids":ids,"source_ids":np.array([13],np.int64),"edge_ids":np.array([7],np.int64),
       "edge_post":np.array([0],np.int64),"edge_source":np.array([0],np.int64),"membership":member,
       "pool":np.array([[.5,0,0,0,.5,0],[0,.25,.25,.25,0,.25]]),
       "low":np.full((n,3),.3),"high":np.full((n,3),30.),"kappa":np.ones(n),
       "fixed":np.array([False,False]),"moment":np.zeros((7,m)),"rest":np.full(7,.05),"ltm":np.array([1],np.int64),
       "bank0":np.zeros((n,3)),"activation0":np.zeros(m),"original_activation0":np.zeros(m),
       "initial_rate":np.zeros(n,np.float32),"initial_adaptation":np.zeros(n,np.float32),
       "samples":101,"stages":8000,"params":3,"gpu_bytes_estimate":1000}
    p["moment"][0]=[1.,-1.]
    return p


def raw_response(p):
    n,m=len(p["motor_ids"]),len(p["fixed"])
    rate=np.full((101,n),1e-9); rate[0]=p["initial_rate"]
    bank=np.zeros((101,n,3)); original=np.zeros((101,m))
    bank[0],original[0]=p["bank0"],p["original_activation0"]
    for t in range(1,101):
        bank[t]=bank[t-1]+run.response.FILTER_ALPHA*(rate[t,:,None]-bank[t-1])
        original[t]=original[t-1]+run.response.FIXED_ALPHA*(p["pool"]@rate[t]-original[t-1])
    channels=p["low"]*bank+(p["high"]-p["low"])*np.maximum(bank-p["kappa"][None,:,None],0)
    activation=np.zeros((101,m))
    for i,member in enumerate(p["membership"]): activation[:,member]+=channels.sum(2)[:,i]
    activation[:,p["fixed"]]=original[:,p["fixed"]]
    current=np.repeat(np.arange(8000,dtype=float)[:,None],n,axis=1)
    return {"rate":rate,"adaptation":np.zeros((101,n)),"bank":bank,"original_activation":original,
        "activation":activation,"q":p["rest"]+.02*(activation@p["moment"].T),"ltm":.2*np.tanh(activation[:,p["ltm"]]/10),
        "current":current,"phi":np.ones((8000,n)),"branch":np.zeros((8000,n),np.uint8)}


def reference(p):
    raw=raw_response(p)
    return {"coefficients":run.base.neutral(1),"startup_q":raw["q"],"startup_ltm":raw["ltm"],
        "target_q":np.zeros((99,7)),"target_ltm":np.zeros((99,1)),"qerr":raw["q"][1:100],"lerr":raw["ltm"][1:100]}


class Engine:
    def __init__(self,p,*,copy_error=None,solve_error=None):
        COST["constructors_requested"]+=1; COST["constructors_completed"]+=1
        self.problem=copy.deepcopy(p); self.copy_error=copy_error; self.solve_error=solve_error
        self.receipt={}; self.last_receipt={}; self.host_copy_events=[]; self.device=None
    def solve(self,c,*,derivative,host):
        assert derivative is host is False
        COST["forward_requested"]+=1
        self.device=raw_response(self.problem)
        self.last_receipt={"completed_steps":2000,"derivative_completed_steps":0,"completed_samples":101,"status":"complete"}
        self.device["receipt"]=self.last_receipt
        if self.solve_error:
            self.solve_error.partial_response=self.device.copy(); self.solve_error.response_receipt=self.last_receipt.copy()
            raise self.solve_error
        COST["forward_completed"]+=1; COST["forward_ticks"]+=2000
        return self.device
    def host_copy(self,v,*,label):
        assert any(v is a for a in self.device.values())
        COST["copy_requested"]+=1; self.host_copy_events.append({"label":label,"requested":True,"completed":False})
        if self.copy_error and label.endswith(":"+self.copy_error[0]): raise self.copy_error[1]
        result=v.copy(); COST["copy_completed"]+=1; self.host_copy_events[-1]["completed"]=True
        return result


def meter(tmp,clock=lambda:0.):
    output=tmp/"out"; output.mkdir()
    m=run.StateMeter(output,run.ScienceDeadline(0.,clock=clock));m.budget=run.StateBudget(m.deadline)
    return m


def evaluate(p,m,factory,label="neutral",ref=None):
    return run.evaluate_state(p,run.base.neutral(1),reference(p) if ref is None else ref,factory,m,label)


def test_independent_decoder_fixed_original_and_all_q_ltm():
    p=mapping();p["fixed"][1]=True
    raw=raw_response(p); before=copy.deepcopy(raw)
    arrays,gate=run.reconstruct_decoder(p,raw)
    assert gate["pass"]
    np.testing.assert_array_equal(arrays["reconstructed_activation"][:,1],raw["original_activation"][:,1])
    for key in raw: np.testing.assert_array_equal(raw[key],before[key])
    raw["activation"][83,0]+=.001
    assert not run.reconstruct_decoder(p,raw)[1]["pass"]


def test_exact_control_endpoint_and_last_twenty_tick_four_stage_slicing():
    p=mapping();raw=raw_response(p)
    arrays,report=run.attribution(p,raw,np.zeros((99,7)))
    np.testing.assert_array_equal(arrays["window_stage_ranges"],[[4800,4880],[6560,6640],[7840,7920]])
    assert arrays["selected_current"].shape==(3,20,4,6)
    np.testing.assert_array_equal(arrays["selected_current"][0,:, :,0].ravel(),np.arange(4800,4880))
    np.testing.assert_array_equal(arrays["selected_endpoint_adaptation"],raw["adaptation"][[61,83,99]][:,arrays["q0_motor_indices"]])
    np.testing.assert_allclose(arrays["q0_muscle_contribution"].sum(1)+p["rest"][0],raw["q"][:,0],atol=1e-16)
    assert report["h286"]=="supported" and all(all(row) for row in report["rates_strictly_positive"])
    assert not any(any(row) for row in report["rates_exact_zero"])


@pytest.mark.parametrize("kind",["duplicate","missing","moment","positive","branch","nan","dtype"])
def test_identity_and_raw_changes_fail_closed(kind):
    p=mapping();raw=raw_response(p)
    if kind=="duplicate":p["motor_ids"][1]=p["motor_ids"][0]
    elif kind=="missing":p["motor_ids"][0]=594
    elif kind=="moment":p["moment"][0]=0;raw=raw_response(p)
    elif kind=="positive":p["moment"][0]*=-1;raw=raw_response(p)
    elif kind=="branch":raw["branch"][0,0]=255
    elif kind=="nan":raw["rate"][0,0]=np.nan
    else:raw["phi"]=raw["phi"].astype(np.float32)
    with pytest.raises((ValueError,IndexError)):run.attribution(p,raw,np.zeros((99,7)))


def test_bank_at_kappa_refutes_strict_below_even_hinge_equal_inactive():
    p=mapping();p["kappa"][0]=raw_response(p)["bank"][99,0,0]
    raw=raw_response(p);_,report=run.attribution(p,raw,np.zeros((99,7)))
    assert report["h286"]=="refuted"


def test_response_and_engine_weakrefs_dead_before_cleanup_and_next_constructor(tmp_path):
    p=mapping();m=meter(tmp_path);engines,arrays=[],[]
    def factory(mapping):
        assert all(r() is None for r in engines+arrays)
        e=Engine(mapping);engines.append(weakref.ref(e));original=e.solve
        def solve(*args,**kwargs):
            result=original(*args,**kwargs);arrays.extend(weakref.ref(a) for a in result.values() if isinstance(a,np.ndarray));return result
        e.solve=solve;return e
    cleanup=m.cleanup
    def clean():assert all(r() is None for r in engines+arrays);cleanup()
    m.cleanup=clean
    for label in ("neutral","plateau"):assert evaluate(p,m,factory,label)["h286"]=="supported"
    assert m.engine_requests==m.engine_creation["completed"]==2 and m.peak_contexts==1 and m.active_contexts==0
    assert m.budget.counts["forward_completed_steps"]==4000 and m.budget.counts["derivative_completed_steps"]==0
    with pytest.raises(run.base.BudgetExceeded):m.budget.request(False)
    with pytest.raises(ValueError):m.budget.request(True)


@pytest.mark.parametrize("kind",["copy","forward","reproduction","storage","cleanup"])
def test_primary_partial_raw_cost_and_ownership_failure(tmp_path,monkeypatch,kind):
    p=mapping();m=meter(tmp_path);original=OSError("first "+kind);refs=[]
    def factory(mapping):
        e=Engine(mapping,copy_error=("rate",original) if kind=="copy" else None,solve_error=original if kind=="forward" else None)
        refs.append(weakref.ref(e));return e
    cleanup=m.cleanup
    def clean():
        assert all(r() is None for r in refs)
        if kind=="cleanup":raise original
        cleanup()
    m.cleanup=clean
    ref=reference(p)
    if kind=="reproduction":ref["startup_q"]=ref["startup_q"].copy();ref["startup_q"][0,0]+=.001
    if kind=="storage":
        prior=run.persist
        def persist(output,name,arrays,receipt):
            if name=="neutral":raise original
            return prior(output,name,arrays,receipt)
        monkeypatch.setattr(run,"persist",persist)
    with pytest.raises(ValueError if kind=="reproduction" else OSError) as caught:evaluate(p,m,factory,ref=ref)
    if kind!="reproduction":assert caught.value is original
    assert all(r() is None for r in refs) and m.active_contexts==0
    assert m.budget.counts["forward_completed_steps"]==2000
    if kind=="copy":
        with np.load(m.output/"neutral-partial.npz") as z:assert "rate" not in z and len(z.files)==9
        event=m.evaluation_events[0]
        assert event["copy"]["requested"]==10 and event["copy"]["completed"]==9 and event["copy"]["failed"]==1
        assert event["copy_errors"][0]["key"]=="rate"
    assert (m.output/"cost.json").is_file()


def test_partial_copy_publication_error_keeps_original_copy_error(tmp_path,monkeypatch):
    p=mapping();m=meter(tmp_path);original=OSError("first copy")
    prior=run.persist
    def persist(output,name,arrays,receipt):
        if name.endswith("partial"):raise OSError("secondary storage")
        return prior(output,name,arrays,receipt)
    monkeypatch.setattr(run,"persist",persist)
    with pytest.raises(OSError) as caught:evaluate(p,m,lambda mapping:Engine(mapping,copy_error=("bank",original)))
    assert caught.value is original and any("secondary storage" in n for n in original.__notes__)
    assert m.evaluation_events[0]["copy"]["requested"]==10


def test_post_return_constructor_clock_error_releases_new_engine(tmp_path):
    returned=[False];failed=[False];original=OSError("constructor clock");refs=[]
    def clock():
        if returned[0] and not failed[0]:
            failed[0]=True;raise original
        return 0.
    p=mapping();m=meter(tmp_path,clock)
    def factory(mapping):
        e=Engine(mapping);refs.append(weakref.ref(e));returned[0]=True;return e
    cleanup=m.cleanup
    def clean():assert all(r() is None for r in refs);cleanup()
    m.cleanup=clean
    with pytest.raises(OSError) as caught:evaluate(p,m,factory)
    assert caught.value is original and all(r() is None for r in refs)
    assert m.engine_creation["completed"]==1 and not m.engine_creation["wall_known"]
    assert m.active_contexts==0 and m.budget.counts["evaluations_requested"]==0


def freeze_minimal():
    return {"rf_engine_roster":{},**{k:{} for k in ("e285_freeze","e285_source_review","e285_root_result","e285_plateau_audit",
        "plateau_actual","inherited_sources","approved_sources")}}


@pytest.mark.parametrize("kind",["complete","state_failure","terminal_storage"])
def test_two_state_driver_and_invalid_publication_h_none(tmp_path,monkeypatch,kind):
    p=mapping();m=meter(tmp_path);ref=reference(p)
    case={"neutral_coefficients":ref["coefficients"],"neutral_startup_q":ref["startup_q"],"neutral_startup_ltm":ref["startup_ltm"],
        "target_q":ref["target_q"],"target_ltm":ref["target_ltm"]}
    monkeypatch.setattr(run,"validate_rf_roster",lambda *args:None)
    error=OSError("state failure")
    def factory(mapping):return Engine(mapping,copy_error=("current",error) if kind=="state_failure" else None)
    if kind=="terminal_storage":
        prior=run.persist
        def persist(output,name,arrays,receipt):
            if name=="attribution-result":raise error
            return prior(output,name,arrays,receipt)
        monkeypatch.setattr(run,"persist",persist)
    if kind=="complete":
        result=run.execute_attribution(freeze_minimal(),({}, {}, p),case,ref,factory,m)
        assert result["run_valid"] and result["H286"]=="supported" and len(result["states"])==2
        assert not any(result[k] for k in ("candidate_created","initialization_supported","F6_exposed","body_promoted"))
    else:
        with pytest.raises(OSError) as caught:run.execute_attribution(freeze_minimal(),({}, {}, p),case,ref,factory,m)
        assert caught.value is error and not error.attribution_result["run_valid"] and error.attribution_result["H286"] is None
    assert m.engine_requests==(1 if kind=="state_failure" else 2)


def test_invalid_cli_original_solver_zero_fresh_output(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"validate_freeze",lambda *args:(_ for _ in ()).throw(ValueError("bad freeze")))
    monkeypatch.setattr(run,"original_run",lambda *args:pytest.fail("no original"))
    output=tmp_path/"fresh"
    args=["--freeze","x","--freeze-sha256","x","--source-review","x","--source-review-sha256","x","--output",str(output)]
    assert run.main(args)==1 and json.loads((output/"cost.json").read_text())["original_calls"]==0
    with pytest.raises(ValueError):run.main(args)

@pytest.fixture
def freeze_fixture(tmp_path,monkeypatch):
    old,original,save_old,tmp,_=prior_tests.freeze_fixture.__wrapped__(tmp_path,monkeypatch)
    fp,fs,rp,rs=save_old();owned={}
    for name in run.SOURCES:
        path=tmp/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("generated E286 source");owned[name]=run.base.sha256(path)
    def store(path,value):
        path.write_text(json.dumps(value));return {"path":path.name,"sha256":run.base.sha256(path)}
    with np.load(tmp/old["rf_case"]["path"]) as z:case={k:z[k] for k in z.files}
    q=case["neutral_startup_q"].copy();ltm=case["neutral_startup_ltm"].copy()
    q[1:100]=case["target_q"];q[83,0]=case["target_q"][82,0]+.03693922689093462
    plateau={"coefficients":run.base.neutral(261),"startup_q":q,"startup_ltm":ltm,
        "qerr":q[1:100]-case["target_q"],"lerr":ltm[1:100]-case["target_ltm"]}
    path=tmp/"outer-05-alpha-0-actual.npz";np.savez(path,**plateau);ps={"path":path.name,"sha256":run.base.sha256(path)}
    raw=tmp/"plateau-proof-raw.npz";np.savez(raw,generated=np.zeros(1));rawspec={"path":raw.name,"sha256":run.base.sha256(raw)}
    audit={"schema":"flylab.m3.e285.root-original-audit.v1","verdict":"valid_diagnostic_rf_capacity_not_supported",
        "terminal_status":"six_outer_exhausted","neutral_exact":True,"sparse_equations_exact":True,"actual_decisions_independently_verified":True,
        **dict.fromkeys(("rf_capacity_supported","f32_evaluated","initialization_supported","candidate_created","F6_exposed"),False),
        "final_q_peak":.03693922689093462}
    proof={"schema":"flylab.m3.e285.plateau-audit.v1","original_calls":0,"model_calls":0,"solver_calls":0,
        "final":ps,"final_peak_control":83,"final_peak_local_joint":0,"final_peak_signed_error":audit["final_q_peak"],
        "records":[{"outer":i,"entry":rawspec,"solution":rawspec,"full_hard_domain_fixed_J_infeasible":True,
                    "peak_lower_bound":.036936296347271205} for i in range(6)]}
    freeze={"schema":run.SCHEMA,"contract":copy.deepcopy(run.CONTRACT),"approved_sources":owned,
        "inherited_sources":{**old["inherited_sources"],**old["approved_sources"]},
        "e285_freeze":{"path":fp.name,"sha256":fs},"e285_source_review":{"path":rp.name,"sha256":rs},
        "e285_root_result":store(tmp/"e285-final-audit.json",audit),"e285_plateau_audit":store(tmp/"e285-plateau.json",proof),"plateau_actual":ps,
        **{k:old[k] for k in ("inputs","cases","engine_roster","packages","e280_resource_qualification","rf_index","rf_case","rf_engine_roster")}}
    def save():
        f=store(tmp/"e286-freeze.json",freeze)
        r=store(tmp/"e286-review.json",{"schema":run.REVIEW_SCHEMA,"verdict":"approved","approved_sources":owned,"freeze_sha256":f["sha256"]})
        return tmp/f["path"],f["sha256"],tmp/r["path"],r["sha256"]
    return freeze,original,save,tmp


def test_nested_exact32_source_input_case_roster_and_plateau_gate(freeze_fixture):
    freeze,original,save,_=freeze_fixture
    value,prior=run.validate_freeze(*save())
    assert value==freeze and prior==original and len(run.REQUIRED_INHERITED_SOURCES)==32


@pytest.mark.parametrize("kind",["source","inherited","contract","rf_index","rf_case","inputs","old_review","audit","proof","plateau","qualification"])
def test_frozen_provenance_changes_rejected(freeze_fixture,kind):
    freeze,_,save,tmp=freeze_fixture
    if kind=="source":(tmp/run.SOURCES[0]).write_text("changed")
    elif kind=="inherited":(tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif kind=="contract":freeze["contract"]["max_engines"]=3
    elif kind=="rf_index":freeze["rf_index"]=4
    elif kind=="rf_case":freeze["rf_case"]=freeze["cases"][4]
    elif kind=="inputs":freeze["inputs"]={"changed":True}
    elif kind=="old_review":freeze["e285_source_review"]["sha256"]="0"*64
    elif kind=="audit":
        p=tmp/freeze["e285_root_result"]["path"];value=json.loads(p.read_text());value["candidate_created"]=True;p.write_text(json.dumps(value));freeze["e285_root_result"]["sha256"]=run.base.sha256(p)
    elif kind=="proof":freeze["e285_plateau_audit"]["sha256"]="0"*64
    elif kind=="plateau":(tmp/freeze["plateau_actual"]["path"]).write_bytes(b"changed")
    else:del freeze["e280_resource_qualification"]
    with pytest.raises((ValueError,KeyError)):run.validate_freeze(*save())


def test_clock_budget_and_no_third_or_repeat_construction(tmp_path):
    now=[0.];m=meter(tmp_path,clock=lambda:now[0]);p=mapping()
    evaluate(p,m,lambda mapping:Engine(mapping))
    with pytest.raises(ValueError):evaluate(p,m,lambda mapping:pytest.fail("no repeat constructor"))
    evaluate(p,m,lambda mapping:Engine(mapping),"plateau")
    with pytest.raises(ValueError):evaluate(p,m,lambda mapping:pytest.fail("no third constructor"),"plateau")
    assert m.engine_requests==2
    now[0]=10.000001
    with pytest.raises(run.base.BudgetExceeded):m.deadline.check()
    assert m.deadline.receipt()["supervisor_wall_limit_s"]==15.

@pytest.mark.parametrize("phase",["forward","copy"])
def test_completed_response_or_copy_end_clock_failure_keeps_array_cost(tmp_path,phase):
    p=mapping();ready=[False];fired=[False];original=OSError(phase+" end clock");refs=[]
    def clock():
        if ready[0] and not fired[0]:fired[0]=True;raise original
        return 0.
    m=meter(tmp_path,clock)
    class TimedEngine(Engine):
        def solve(self,*args,**kwargs):
            value=super().solve(*args,**kwargs)
            if phase=="forward":ready[0]=True
            return value
        def host_copy(self,value,*,label):
            copied=super().host_copy(value,label=label)
            if phase=="copy" and label.endswith(":current"):ready[0]=True
            return copied
    def factory(mapping):
        e=TimedEngine(mapping);refs.append(weakref.ref(e));return e
    cleanup=m.cleanup
    def clean():assert all(r() is None for r in refs);cleanup()
    m.cleanup=clean
    with pytest.raises(OSError) as caught:evaluate(p,m,factory)
    assert caught.value is original and all(r() is None for r in refs)
    event=m.evaluation_events[0]
    assert event[phase]["completed"]>=1 and event[phase]["failed"]==1 and not event[phase]["wall_known"]
    assert m.budget.counts["evaluations_completed"]==1 and m.budget.counts["forward_completed_steps"]==2000
    with np.load(m.output/"neutral-partial.npz") as z:assert set(z.files)==set(run.COPY_KEYS)
