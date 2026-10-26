"""E287 tiny/generated CPU contracts; no original or CUDA model evaluation."""
import copy
import json
import time
import weakref
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tools import banc_m3_source_decoder_fit as fit
from tools import fit_c_banc_m3_source_decoder as run

ACTUAL_COST=[]


@pytest.fixture(scope="module",autouse=True)
def generated_cost():
    started=time.perf_counter()
    yield
    assert time.perf_counter()-started<120 and len(ACTUAL_COST)<=20
    print("E287_TINY_ACTUAL_COST",{k:sum(r[k] for r in ACTUAL_COST) for k in
        ("constructor_requested","constructor_completed","constructor_failed","constructor_wall_s","solve_requested","solve_completed","solve_failed","solve_wall_s","iterations")})


def counted_factory(*args):
    assert args[0].shape[0]<=60 and len(ACTUAL_COST)<20
    row={"constructor_requested":1,"constructor_completed":0,"constructor_failed":0,"constructor_wall_s":0.,
         "solve_requested":0,"solve_completed":0,"solve_failed":0,"solve_wall_s":0.,"iterations":0}
    ACTUAL_COST.append(row); started=time.perf_counter()
    try: solver=clarabel.DefaultSolver(*args); row["constructor_completed"]=1
    except Exception: row["constructor_failed"]=1; raise
    finally: row["constructor_wall_s"]=time.perf_counter()-started
    def solve():
        row["solve_requested"]+=1; started=time.perf_counter()
        try: raw=solver.solve(); row["solve_completed"]+=1; row["iterations"]+=raw.iterations; return raw
        except Exception: row["solve_failed"]+=1; raise
        finally: row["solve_wall_s"]+=time.perf_counter()-started
    return SimpleNamespace(solve=solve)


def fixture():
    membership=np.array([0,0,1,1,2],np.int64); pool=np.zeros((3,5)); pool[membership,np.arange(5)]=[.5,.5,.5,.5,1.]
    low=np.zeros((5,3)); low[:,1]=pool[membership,np.arange(5)]
    d={"motor_ids":np.array([20,3,91,7,110],np.int64),"membership":membership,"pool":pool,"low":low,"high":low.copy(),
       "kappa":np.ones(5),"fixed":np.array([False,False,True]),"moment":np.diag([1.,-.7,.2]),"rest":np.array([.1,.2,.3]),
       "ltm":np.array([2],np.int64),"bank0":np.zeros((5,3)),"activation0":np.zeros(3),"original_activation0":np.zeros(3),
       "source_ids":np.array([5],np.int64),"edge_ids":np.array([8],np.int64)}
    legs={"motor":np.array([0,0,1,1,2],np.int64),"muscle":np.arange(3,dtype=np.int64),"q":np.arange(3,dtype=np.int64),"ltm":np.array([2],np.int64)}
    return d,legs


def response(mapping,samples,extent):
    bank=np.linspace(0,extent,samples)[:,None,None]*np.array([1.,1.3,.7,1.1,.9])[None,:,None]*np.array([.8,1.,1.2])[None,None,:]
    original=np.zeros((samples,3)); original[:,2]=np.linspace(0,extent,samples)*.4
    return {"bank":bank,"original_activation":original}


def problem_fixture(*,bad_startup=False):
    mapping,legs=fixture(); training=response(mapping,9,4.); startup=response(mapping,5,.1)
    low,high=mapping["low"]*1.4,mapping["high"]*1.4
    low[-1],high[-1]=mapping["low"][-1],mapping["high"][-1]
    train=fit.decode_candidate(mapping,low,high,**training); start=fit.decode_candidate(mapping,low,high,**startup)
    targets={"train_q":train["q"],"startup_q":start["q"],"startup_ltm":start["ltm"]}
    if bad_startup: targets["startup_q"][0,0]+=.1
    return fit.decoder_problem(mapping,legs,training,startup,targets),mapping,legs,training,startup,targets


def test_affine_piecewise_basis_and_fixed_original_no_extra_ema():
    mapping,legs=fixture(); layout=fit.decoder_layout(mapping,legs); data=response(mapping,8,3.)
    D,offset=fit.affine_q(mapping,layout,**data); x=layout["x0"].copy(); x[:24]*=1.3
    low,high=fit.candidate_slopes(layout,x); decoded=fit.decode_candidate(mapping,low,high,**data)
    literal=np.zeros((8,3))
    for s in range(8):
        for motor,muscle in enumerate(mapping["membership"]):
            if muscle==2: continue
            for k in range(3):
                b=data["bank"][s,motor,k]
                literal[s,muscle]+=low[motor,k]*min(b,mapping["kappa"][motor])+high[motor,k]*max(b-mapping["kappa"][motor],0)
    literal[:,2]=data["original_activation"][:,2]
    np.testing.assert_allclose(decoded["activation"],literal,rtol=0,atol=2e-15)
    np.testing.assert_allclose((D@x[:24]).reshape(8,3)+offset,decoded["q"],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(decoded["activation"][:,2],data["original_activation"][:,2])
    np.testing.assert_array_equal(low[-1],mapping["low"][-1])
    assert layout["variables"]==48 and layout["free_motor"].tolist()==[0,1,2,3]
    assert np.array_equal(layout["x0"][24:36],np.minimum(mapping["low"][:4],mapping["high"][:4]).ravel())


def test_original_dense_objective_and_all_envelopes_peak_soc_rows():
    p,*_=problem_fixture(); layout=p["layout"]; D=p["train_design"].toarray(); r=p["train_error_offset"]; n=48
    expected=2*fit.REGULARIZATION*np.eye(n); expected[:24,:24]+=2*D.T@D/len(r)
    linear=-2*fit.REGULARIZATION*layout["x0"].copy(); linear[:24]+=2*D.T@r/len(r)
    np.testing.assert_allclose(p["P"].toarray(),expected,atol=1e-17,rtol=1e-14)
    np.testing.assert_allclose(p["q"],linear,atol=1e-17,rtol=1e-14)
    rng=np.random.default_rng(823); x=rng.random(48)*.4
    direct=np.mean((D@x[:24]+r)**2)+1e-8*np.sum((x-layout["x0"])**2)
    assert abs(direct-(.5*x@(p["P"]@x)+p["q"]@x+p["objective_constant"]))<1e-16
    nn=p["cone_dimensions"][0]; slack=p["b"]-p["A"]@layout["x0"]
    assert nn==6*12+2*2+4+2*15 and p["cone_dimensions"][1]==16
    for pair in range(12):
        lo,hi,lower,upper=2*pair,2*pair+1,24+pair,36+pair
        np.testing.assert_array_equal(p["A"].getrow(6*pair+2).toarray()[0,[lower,lo]],[1.,-1.])
        np.testing.assert_array_equal(p["A"].getrow(6*pair+5).toarray()[0,[hi,upper]],[1.,-1.])
    np.testing.assert_allclose(slack[nn+1:],p["startup_design"]@layout["x0"][:24]+p["startup_error_offset"],atol=1e-16)
    assert slack[nn]==np.sqrt(15)*fit.Q_RMS and fit.Q_PEAK==.03838315744033522
    arrays=fit.problem_arrays(p)
    assert (sparse.csc_matrix((arrays["P_data"],arrays["P_indices"],arrays["P_indptr"]),shape=arrays["P_shape"])!=p["P"]).nnz==0


@pytest.mark.parametrize("infeasible",[False,True])
def test_actual_tiny_single_solve_original_primal_and_no_optimality_claim(infeasible):
    p,*_=problem_fixture(bad_startup=infeasible); solved=fit.solve_decoder(p,2.,solver_factory=counted_factory)
    assert solved["valid"] != infeasible and solved["receipt"]["constructor"]["completed"]==1
    if not infeasible:
        x=solved["x"]; slack=p["b"]-p["A"]@x; dims=p["cone_dimensions"]
        minimum=min(np.min(slack[:dims[0]]),slack[dims[0]]-np.linalg.norm(slack[dims[0]+1:]))
        assert minimum>=-1e-8 and not solved["receipt"]["certificate_receipt"]["optimality_claim"]
        assert abs(minimum-solved["receipt"]["certificate_receipt"]["minimum_residual"])<1e-15
    else: assert solved["receipt"]["status"]=="PrimalInfeasible"


@pytest.mark.parametrize("key",["P","q","A","b","cone","layout","constant","train","startup"])
def test_mutated_original_math_closed_before_solver(key):
    p,*_=problem_fixture()
    if key in ("P","A"): p[key].data[0]+=.001
    elif key in ("q","b"): p[key][0]+=.001
    elif key=="cone": p["cone_dimensions"]=(1,)
    elif key=="layout": p["layout"]["x0"][0]+=.1
    elif key=="constant": p["objective_constant"]+=.1
    else: p[key+"_error_offset"][0]+=.1
    with pytest.raises(ValueError): fit.solve_decoder(p,2.,solver_factory=lambda *_:pytest.fail("no solver"))


@pytest.mark.parametrize("kind",["status","shape","nan","primal","finite_raw_s_difference"])
def test_primal_cert_status_raw_shape_and_exact_affine_instead_of_solver_s(kind):
    p,*_=problem_fixture(); x=p["layout"]["x0"].copy(); s=p["b"]-p["A"]@x; y=np.zeros_like(s); status="AlmostSolved"
    if kind=="status": status="MaxTime"
    elif kind=="shape": x=x[:-1]
    elif kind=="nan": y[0]=np.inf
    elif kind=="primal": x[24]=-.01
    else: s+=.1
    cert=fit.primal_certificate(p,x,s,y,status)
    assert cert["valid"]==(kind=="finite_raw_s_difference")


@pytest.mark.parametrize("kind",["initial","fixed","support","negative","cross_leg","motor_id","kappa"])
def test_decoder_identity_support_and_zero_initial_contract(kind):
    mapping,legs=fixture()
    if kind=="initial": mapping["bank0"][0,0]=1e-12
    elif kind=="fixed": mapping["low"][-1,0]=1e-12
    elif kind=="support": mapping["membership"][0]=1
    elif kind=="negative": mapping["low"][0,0]=-1e-12
    elif kind=="cross_leg": mapping["moment"][0,1]=1e-12
    elif kind=="motor_id": mapping["motor_ids"][0]=mapping["motor_ids"][1]
    else: mapping["kappa"][0]=-1e-12
    with pytest.raises(ValueError): fit.decoder_layout(mapping,legs)


@pytest.mark.parametrize("kind",["development_key","full_teacher","long_startup","target"])
def test_optimizer_objects_never_include_development(kind):
    _,mapping,legs,training,startup,targets=problem_fixture()
    if kind=="development_key": targets["development_q"]=np.zeros((800,3))
    elif kind=="full_teacher": training=response(mapping,2001,3.)
    elif kind=="long_startup": startup=response(mapping,100,.1)
    else: targets["train_q"]=targets["train_q"].astype(np.float32)
    with pytest.raises(ValueError): fit.decoder_problem(mapping,legs,training,startup,targets)


def test_generic_fixed_ltm_leg_can_be_derived_and_no_anatomy_reorder():
    mapping,legs=fixture(); del legs["ltm"]
    layout=fit.decoder_layout(mapping,legs)
    assert layout["legs"]["ltm"].tolist()==[2] and layout["mapping"]["motor_ids"].tolist()==[20,3,91,7,110]


def meter(tmp,clock=lambda:0.):
    output=tmp/"output"; output.mkdir()
    return run.DecoderMeter(output,run.ScienceDeadline(0.,clock=clock))


def fit_inputs():
    mapping,legs=fixture(); responses={"startup":response(mapping,101,.1),"teacher":response(mapping,2001,4.)}
    low,high=mapping["low"]*1.4,mapping["high"]*1.4; low[-1],high[-1]=mapping["low"][-1],mapping["high"][-1]
    wanted={k:fit.decode_candidate(mapping,low,high,**v) for k,v in responses.items()}
    targets={"startup_q":wanted["startup"]["q"][1:100].copy(),"startup_ltm":wanted["startup"]["ltm"][1:100].copy(),"teacher_q":wanted["teacher"]["q"].copy()}
    baseline=fit.decode_candidate(mapping,mapping["low"],mapping["high"],**responses["teacher"])["q"]
    return mapping,legs,responses,targets,baseline,np.array([[1.,.1,.2]],np.float32)


def stub_solution(p,*,valid=True,negative=False):
    x=p["layout"]["x0"].copy(); x[:24]*=1.4; x[24:36]=np.minimum(x[:24].reshape(-1,2)[:,0],x[:24].reshape(-1,2)[:,1]); x[36:]=np.maximum(x[:24].reshape(-1,2)[:,0],x[:24].reshape(-1,2)[:,1])
    if negative: x[0]=-1e-12
    receipt={k:{**fit.metering.phase(),"requested":1,"completed":1,"wall_s":.01} for k in ("constructor","solve","certificate")}
    receipt.update(iterations=3,iterations_known=True,status="Solved")
    return {"valid":valid,"x":x,"solver_s":p["b"]-p["A"]@x,"solver_y":np.zeros_like(p["b"]),"primal_slack":p["b"]-p["A"]@x,"receipt":receipt}


def test_actual_tiny_fit_training_copy_and_sha_before_single_development(tmp_path,monkeypatch):
    args=fit_inputs(); m=meter(tmp_path); original_assess=fit.assess_candidate; assessments=[]
    def assess(*args):
        meta=json.loads((m.output/"candidate.json").read_text()); assert meta["candidate_sha256"]==fit.base.sha256(m.output/"candidate.npz")
        assert set(meta)=={"candidate_sha256","decoder_identity"}; assessments.append(meta); return original_assess(*args)
    monkeypatch.setattr(fit,"assess_candidate",assess)
    def solve(p,t,**kwargs):
        assert p["train_design"].shape==(3600,24) and p["startup_design"].shape==(297,24)
        return fit.solve_decoder(p,t,**kwargs)
    result=run.fit_saved(*args,m,solver_factory=counted_factory,solve=solve)
    assert result["run_valid"] and result["selection_pass"] and result["schema"]==run.SCHEMA and len(assessments)==1
    assert m.development["requested"]==m.development["completed"]==1 and m.solver_cost["solve"]["requested"]==1
    with np.load(m.output/"candidate.npz") as z:
        assert z["slopes"].shape==(3,5,3,2) and np.all(z["bank0"]==0)
        np.testing.assert_array_equal(z["coefficients64"],args[-1].astype(np.float64))
        np.testing.assert_array_equal(z["low"][-1],args[0]["low"][-1])
    with np.load(m.output/"candidate-startup.npz") as z: assert set(z.files)=={"q","ltm"} and z["q"].shape==(101,3)
    assert result["candidate_sha256"]==fit.base.sha256(m.output/"candidate.npz")


@pytest.mark.parametrize("kind",["fixed_ltm","uncertified","strict_negative","development_fail"])
def test_scientific_rejections_have_no_refit_or_repair(tmp_path,kind):
    args=list(fit_inputs()); m=meter(tmp_path); calls=[]
    if kind=="fixed_ltm": args[3]["startup_ltm"]+=.03
    elif kind=="development_fail": args[3]["teacher_q"][1201:]=args[4][1201:]
    def solve(p,*args,**kwargs): calls.append(p); return stub_solution(p,valid=kind!="uncertified",negative=kind=="strict_negative")
    result=run.fit_saved(*args,m,solve=solve)
    assert result["run_valid"] and not result["selection_pass"] and len(calls)==int(kind!="fixed_ltm")
    assert m.development["requested"]==int(kind=="development_fail")
    if kind=="strict_negative":
        with np.load(m.output/"rejected-decoder.npz") as z: assert z["x"][0]==-1e-12 and z["low"][0,0]==-1e-12
        assert not (m.output/"candidate.npz").exists()
    if kind=="development_fail": assert result["checks"]["train"] and not result["checks"]["development"]


@pytest.mark.parametrize("kind",["q_peak","q_rms","ltm_peak","ltm_rms","train","leg_train","development","leg_development","slope"])
def test_final_gates_unchanged_and_empirical_startup_margin_exact(kind):
    mapping,legs,responses,targets,baseline,_=fit_inputs(); low,high=mapping["low"]*1.4,mapping["high"]*1.4
    low[-1],high[-1]=mapping["low"][-1],mapping["high"][-1]
    predicted={k:fit.decode_candidate(mapping,low,high,**v) for k,v in responses.items()}
    if kind=="q_peak": predicted["startup"]["q"][1,0]+=fit.Q_PEAK+1e-12
    elif kind=="q_rms": predicted["startup"]["q"][1:100]+=fit.Q_RMS+1e-12
    elif kind=="ltm_peak": predicted["startup"]["ltm"][1,0]+=.025+1e-12
    elif kind=="ltm_rms": predicted["startup"]["ltm"][1:100]+=.009+1e-12
    elif kind=="train": predicted["teacher"]["q"][1:1201]=baseline[1:1201]
    elif kind=="leg_train": predicted["teacher"]["q"][1:1201,0]=targets["teacher_q"][1:1201,0]+1.06*(baseline[1:1201,0]-targets["teacher_q"][1:1201,0])
    elif kind=="development": predicted["teacher"]["q"][1201:]=targets["teacher_q"][1201:]+1.01*(baseline[1201:]-targets["teacher_q"][1201:])
    elif kind=="leg_development": predicted["teacher"]["q"][1201:,0]=targets["teacher_q"][1201:,0]+1.06*(baseline[1201:,0]-targets["teacher_q"][1201:,0])
    else: low[0,0]=-1e-12
    result=fit.assess_candidate(mapping,legs,low,high,predicted["startup"],predicted["teacher"],targets,baseline)
    assert not result["selection_pass"]
    if kind in ("q_peak","q_rms"): assert result["checks"]["startup"] and not result["checks"]["internal_startup"]
    if kind=="leg_train": assert not result["checks"]["LF_train"]
    if kind=="leg_development": assert not result["checks"]["LF_development"]


@pytest.mark.parametrize("phase",["constructor","solve","certificate","solve_clock"])
def test_solver_failure_same_primary_partial_raw_cost_and_no_retry(tmp_path,monkeypatch,phase):
    args=fit_inputs(); m=meter(tmp_path); error=OSError(phase); raw=SimpleNamespace(x=np.zeros(48),s=np.zeros(1),z=np.zeros(1),status="Solved",iterations=11)
    def fail(*_): raise error
    factory=fail if phase=="constructor" else lambda *_:SimpleNamespace(solve=fail if phase=="solve" else lambda:raw)
    if phase=="certificate": monkeypatch.setattr(fit,"primal_certificate",fail)
    count=[0]
    def clock():
        count[0]+=1
        if phase=="solve_clock" and count[0]==4: raise error
        return count[0]
    def solve(p,t,**kwargs): return fit.solve_decoder(p,t,solver_factory=factory,wall_clock=clock)
    with pytest.raises(OSError) as caught: run.fit_saved(*args,m,solve=solve)
    assert caught.value is error and error.decoder_fit_result["run_valid"] is False
    assert m.solver_cost["constructor"]["requested"]==1 and m.solver_cost["solve"]["requested"]==int(phase!="constructor")
    if phase in ("certificate","solve_clock"):
        assert error.partial_decoder_solution is raw and m.solver_cost["iterations"]==11
        with np.load(m.output/"decoder-solve.npz") as z: np.testing.assert_array_equal(z["x"],raw.x)


@pytest.mark.parametrize("phase",["raw","candidate","selection","cost","deadline"])
def test_storage_deadline_failure_keeps_completed_solver_accounted_once(tmp_path,monkeypatch,phase):
    args=fit_inputs(); m=meter(tmp_path); error=fit.base.BudgetExceeded("original") if phase=="deadline" else OSError(phase); error.add_note("first")
    original_persist=run.persist; original_save=m.save; solve_calls=[]
    def solve(p,*a,**kw): solve_calls.append(p); return stub_solution(p)
    def persist(output,name,*a):
        if (phase=="raw" and name=="decoder-solve") or (phase=="selection" and name=="selection-result"): raise error
        original_persist(output,name,*a)
    monkeypatch.setattr(run,"persist",persist)
    if phase=="candidate":
        original_npz=np.savez
        def save(path,*a,**kw):
            if str(path).endswith("candidate.npz"): raise error
            original_npz(path,*a,**kw)
        monkeypatch.setattr(run.np,"savez",save)
    elif phase=="cost":
        def save():
            if m.solver_cost["solve"]["completed"]: raise error
            original_save()
        monkeypatch.setattr(m,"save",save)
    elif phase=="deadline":
        def check():
            if m.solver_cost["solve"]["completed"]: raise error
        m.deadline.check=check
    with pytest.raises(type(error)) as caught: run.fit_saved(*args,m,solve=solve)
    assert caught.value is error and len(solve_calls)==1 and m.solver_cost["solve"]["completed"]==1 and m.solver_cost["iterations"]==3
    assert not error.decoder_fit_result["run_valid"] and not error.decoder_fit_result["selection_pass"] and "first" in error.__notes__


class FakeEngine:
    def __init__(self,mapping):
        self.problem={"samples":mapping["samples"],"gpu_bytes_estimate":1000}; self.mapping=mapping
        self.receipt={}; self.last_receipt={}; self.host_copy_events=[]
    def solve(self,c,*,derivative,host):
        assert derivative is host is False; self.last_receipt={"completed_steps":(self.problem["samples"]-1)*20,"derivative_completed_steps":0}
        data=response(self.mapping,self.problem["samples"],.1); decoded=fit.decode_candidate(self.mapping,self.mapping["low"],self.mapping["high"],**data)
        return {**data,**decoded,"rate":np.zeros((self.problem["samples"],5)),"receipt":self.last_receipt}
    def host_copy(self,value,*,label): self.host_copy_events.append(label); return value.copy()


def test_two_contexts_one_actual_live_release_before_cleanup_constructor(tmp_path):
    mapping,_=fixture(); m=meter(tmp_path); refs=[]
    def factory(mapping):
        assert all(ref() is None for ref in refs)
        engine=FakeEngine(mapping); refs.append(weakref.ref(engine)); return engine
    cleanup=m.cleanup
    def clean(): assert all(ref() is None for ref in refs); cleanup()
    m.cleanup=clean
    for name,size in (("startup",101),("teacher",2001)):
        mapping["samples"]=size; host=run.acquire_context(name,mapping,np.array([[1.,0.,0.]]),factory,m)
        assert set(host)==set(run.CONTRACT["host_copy_keys"])
    assert m.engine_requests==m.engine_creation["completed"]==2 and m.forward_requested==m.forward_completed==42000
    assert m.active_contexts==0 and all(ref() is None for ref in refs)
    assert sum(len(e["host_copy_events"]) for e in m.engine_events)==10


@pytest.mark.parametrize("kind",["constructor","forward","copy","cleanup"])
def test_source_failure_requested_partial_counts_without_second_context(tmp_path,kind):
    mapping,_=fixture(); mapping["samples"]=101; m=meter(tmp_path); error=OSError(kind); copied=[]
    class Engine(FakeEngine):
        def solve(self,*a,**kw):
            if kind=="forward":
                error.response_receipt={"completed_steps":1000}; error.partial_response={"rate":np.zeros((51,5))}; raise error
            return super().solve(*a,**kw)
        def host_copy(self,value,*,label):
            copied.append(label)
            if kind=="copy" and label.endswith(":rate"): raise error
            return super().host_copy(value,label=label)
    def factory(mapping):
        if kind=="constructor": raise error
        return Engine(mapping)
    if kind=="cleanup": monkey_cleanup=lambda:(_ for _ in ()).throw(error); m.cleanup=monkey_cleanup
    with pytest.raises(OSError) as caught: run.acquire_context("startup",mapping,np.array([[1.,0.,0.]]),factory,m)
    assert caught.value is error and m.engine_requests==1 and m.active_contexts==0
    assert m.forward_requested==(0 if kind=="constructor" else 2000)
    assert m.forward_completed==(0 if kind=="constructor" else 1000 if kind=="forward" else 2000)
    if kind=="copy":
        assert copied==["startup:"+key for key in run.CONTRACT["host_copy_keys"]]
        with np.load(m.output/"source-startup-partial.npz") as z: assert set(z.files)==set(run.CONTRACT["host_copy_keys"])-{"rate"}


@pytest.mark.parametrize("fault",["constructor_clock","forward_clock","copy_rate","copy_twice"])
@pytest.mark.parametrize("save_failure",[False,True])
def test_source_failure_actual_ownership_host_rescue_and_primary(tmp_path,monkeypatch,fault,save_failure):
    mapping,_=fixture(); mapping["samples"]=101
    error=OSError(fault); error.add_note("original ownership fault")
    pending=[False]; engines=[]; devices=[]; copies=[]; cleanup_alive=[]
    def clock():
        if pending[0]: pending[0]=False; raise error
        return 0.
    m=meter(tmp_path,clock=clock)
    class Device(dict):
        pass
    class Engine(FakeEngine):
        def solve(self,*args,**kwargs):
            device=Device(super().solve(*args,**kwargs)); devices.append(weakref.ref(device))
            if fault=="forward_clock": pending[0]=True
            return device
        def host_copy(self,value,*,label):
            copies.append(label)
            if fault in ("copy_rate","copy_twice") and label.endswith(":rate"): raise error
            if fault=="copy_twice" and label.endswith(":q"): raise OSError("secondary q copy")
            return super().host_copy(value,label=label)
    def factory(mapping):
        engine=Engine(mapping); engines.append(weakref.ref(engine))
        if fault=="constructor_clock": pending[0]=True
        return engine
    original_cleanup=m.cleanup
    def cleanup():
        alive=(sum(ref() is not None for ref in engines),sum(ref() is not None for ref in devices))
        cleanup_alive.append(alive)
        assert alive==(0,0)
        original_cleanup()
    m.cleanup=cleanup
    original_persist=run.persist; publications=[]
    def persist(output,name,arrays,receipt):
        if name.endswith("-partial"):
            publications.append((set(arrays),dict(receipt)))
            if save_failure: raise OSError("secondary partial storage")
        return original_persist(output,name,arrays,receipt)
    monkeypatch.setattr(run,"persist",persist)
    with pytest.raises(OSError) as caught: run.acquire_context("startup",mapping,np.array([[1.,0.,0.]]),factory,m)
    assert caught.value is error and "original ownership fault" in error.__notes__
    assert cleanup_alive==[(0,0)] and all(ref() is None for ref in (*engines,*devices))
    assert not hasattr(error,"phase_partial_result") and m.engine_requests==m.engine_creation["completed"]==1
    assert m.active_contexts==0 and len(m.engine_events)==1
    if fault=="constructor_clock":
        assert m.engine_creation["failed"]==1 and m.evaluation["requested"]==0
        assert m.forward_requested==m.forward_completed==0 and copies==publications==[]
        assert error.partial_response=={}
    else:
        assert m.forward_requested==m.forward_completed==2000 and m.evaluation["completed"]==1
        assert m.evaluation["failed"]==(1 if fault=="forward_clock" else 0)
        assert copies==["startup:"+key for key in run.CONTRACT["host_copy_keys"]]
        expected=set(run.CONTRACT["host_copy_keys"])
        if fault in ("copy_rate","copy_twice"): expected.remove("rate")
        if fault=="copy_twice": expected.remove("q")
        assert set(error.partial_response)==set(error.partial_host_response)==expected
        assert publications==[(expected,{"completed_steps":2000,"derivative_completed_steps":0})]
        for key in expected: assert np.isfinite(error.partial_host_response[key]).all()
        if not save_failure:
            with np.load(m.output/"source-startup-partial.npz") as saved: assert set(saved.files)==expected
        if save_failure: assert any("secondary partial storage" in note for note in error.__notes__)
        if fault=="copy_twice": assert any("secondary q copy" in note for note in error.__notes__)


def test_exact_science_solver_memory_development_limits(tmp_path):
    now=[0.]; m=meter(tmp_path,clock=lambda:now[0]); p,*_=problem_fixture(); receipt=stub_solution(p)["receipt"]
    m.account_solver(receipt)
    with pytest.raises(ValueError): m.account_solver(receipt)
    now[0]=50.0001
    with pytest.raises(fit.base.BudgetExceeded) as caught: m.deadline.check()
    assert caught.value.scope=="original" and m.deadline.receipt()["supervisor_wall_limit_s"]==60
    m.peak_estimate=6*1024**3+1
    with pytest.raises(ValueError): m.enforce_memory()
    m.peak_estimate=0; m.memory_events.append({"live_gpu_estimate":4*1024**3+1})
    with pytest.raises(ValueError): m.enforce_memory()
    for limit in (0.,20.00001):
        with pytest.raises(ValueError): fit.solve_decoder(p,limit,solver_factory=lambda *_:pytest.fail("no solver"))


def test_candidate_payload_hash_covers_fixed_source_and_full_slopes():
    p,mapping,*_=problem_fixture(); arrays=run.candidate_arrays(mapping,p["layout"],p["layout"]["x0"],np.array([[1.,.1,.2]],np.float32))
    digest=fit.decoder_identity(arrays)
    for key in ("coefficients32","kappa","membership","slopes","motor_ids","bank0"):
        bad={k:v.copy() for k,v in arrays.items()}; bad[key].flat[0]+=1
        assert fit.decoder_identity(bad)!=digest
    for motor,muscle in enumerate(mapping["membership"]):
        np.testing.assert_array_equal(arrays["slopes"][muscle,motor,:,0],arrays["low"][motor])
        assert np.all(arrays["slopes"][np.arange(3)!=muscle,motor]==0)


def audit_fixture():
    e270={"schema":"flylab.m3.e270.original-final-review.v1","run_valid":True,"verdict":"rejected_under_frozen_conditional_gates","conditional_support":False,"F6_exposed":False}
    e277={"schema":"flylab.m3.e277.root-original-final-review.v1","run_valid":True,"verdict":"valid_reject","conditional_candidate_supported":False,
          "unchanged_final_checks":{"train":False,"LM_development":False},"F6_exposed":False}
    e279={"schema":"flylab.m3.e279.root-original-raw-audit.v1","supervisor_success":True,"E277_reject_preserved":True,"integrity":"verified_native_diagnostic","F6_exposed":False,
        "arms":{"candidate":{"prediction":{"q":{"RMS":.00013667128182327153,"max_abs":.0015168425596647772}},"strict_startup":False}}}
    e286={"schema":"flylab.m3.e286.root-original-audit.v1","verdict":"valid_state_attribution","H286":"supported",
          "independent_decoder_and_stage_identity_checks":True,"original_reproduction_verified":True,"candidate_created":False,"F6_exposed":False}
    return [e270,e277,e279,e286]


@pytest.mark.parametrize("kind",["good","e270","e277_train","e277_lm","transfer_peak","transfer_rms","e286","f6"])
def test_prior_rejections_observed_margin_and_attribution_provenance(kind):
    gates=audit_fixture()
    if kind=="e270": gates[0]["conditional_support"]=True
    elif kind=="e277_train": gates[1]["unchanged_final_checks"]["train"]=True
    elif kind=="e277_lm": gates[1]["unchanged_final_checks"]["LM_development"]=True
    elif kind.startswith("transfer"):
        key="max_abs" if kind=="transfer_peak" else "RMS"; gates[2]["arms"]["candidate"]["prediction"]["q"][key]+=1e-12
    elif kind=="e286": gates[3]["H286"]="unsupported"
    elif kind=="f6": gates[0]["F6_exposed"]=True
    if kind=="good": run.validate_prior_evidence(*gates)
    else:
        with pytest.raises(ValueError): run.validate_prior_evidence(*gates)


def test_cli_invalid_freeze_has_model_solver_zero_and_schema(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"validate_freeze",lambda *_:(_ for _ in ()).throw(ValueError("freeze")))
    monkeypatch.setattr(run,"execute_original",lambda *_:pytest.fail("no original"))
    output=tmp_path/"fresh"; args=["--freeze","x","--freeze-sha256","x","--source-review","x","--source-review-sha256","x","--output",str(output)]
    assert run.main(args)==1
    cost=json.loads((output/"cost.json").read_text()); result=json.loads((output/"result.json").read_text())
    assert cost["original_calls"]==cost["solver_calls"]==0 and result["schema"]==run.SCHEMA and not result["run_valid"]
    with pytest.raises(ValueError): run.main(args)


def test_raw_finite_overflow_is_json_safe_bounded_rejection():
    p,*_=problem_fixture(); x=np.full(48,1e308)
    cert=fit.primal_certificate(p,x,np.zeros_like(p["b"]),np.zeros_like(p["b"]),"Solved")
    assert not cert["valid"]
    json.dumps(cert["receipt"],allow_nan=False)
    p["development_bank"]=np.zeros(1)
    with pytest.raises(ValueError): fit.validate_problem(p)


def test_sparse_buffer_bytes_and_predevelopment_candidate_output_limit(tmp_path,monkeypatch):
    p,*_=problem_fixture(); matrix=p["A"]
    expected=fit.base.array_bytes((matrix.data,matrix.indices,matrix.indptr))
    assert run.measured_bytes({"first":matrix,"duplicate":matrix})==expected>0
    args=fit_inputs(); m=meter(tmp_path); calls=[]; error=OSError("candidate output limit")
    original=run.check_output
    def output_check(output):
        if (output/"candidate.json").exists(): raise error
        original(output)
    monkeypatch.setattr(run,"check_output",output_check)
    def solve(p,*a,**kw): calls.append(p); return stub_solution(p)
    with pytest.raises(OSError) as caught: run.fit_saved(*args,m,solve=solve)
    assert caught.value is error and len(calls)==1 and m.development["requested"]==0
    assert m.solver_cost["solve"]["completed"]==1 and not error.decoder_fit_result["run_valid"]


@pytest.fixture
def freeze_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(run.base,"ROOT",tmp_path)
    def store(name,value):
        path=tmp_path/name; path.write_text(json.dumps(value)); return {"path":name,"sha256":run.base.sha256(path)}
    pins={}
    for name in (*run.REQUIRED_INHERITED_SOURCES,*run.SOURCES):
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("generated frozen source"); pins[name]=run.base.sha256(path)
    owned={k:pins[k] for k in run.SOURCES}; inherited={k:pins[k] for k in run.REQUIRED_INHERITED_SOURCES}
    old={"schema":run.base.SCHEMA,"contract":run.base.CONTRACT,"approved_sources":{k:pins[k] for k in run.base.SOURCES}}
    of=store("e270.json",old); oreview=store("e270-review.json",{"schema":run.base.REVIEW_SCHEMA,"verdict":"approved","approved_sources":old["approved_sources"],"freeze_sha256":of["sha256"]})
    original={"inputs":{"e270":of},"e270_original_freeze":{"input":"e270"},"contexts":{"startup":{},"teacher":{}},
        "targets":{"original":True},"legs":{"original":True},"identity":{"original":True},"approved_sources":{k:pins[k] for k in run.certified.SOURCES}}
    ef=store("e277.json",original); er=store("e277-review.json",{"schema":run.certified.REVIEW_SCHEMA,"verdict":"approved"})
    state={"approved_sources":{k:pins[k] for k in run.attribution.SOURCES},"inherited_sources":{k:pins[k] for k in run.attribution.REQUIRED_INHERITED_SOURCES},"packages":{"generated":True}}
    sf=store("e286.json",state); sr=store("e286-review.json",{"schema":run.attribution.REVIEW_SCHEMA,"verdict":"approved"})
    dependency_calls=[]
    def e277(*args): dependency_calls.append("E277"); return original
    def e286(*args): dependency_calls.append("E286"); return state,old
    monkeypatch.setattr(run.certified,"validate_freeze",e277); monkeypatch.setattr(run.attribution,"validate_freeze",e286)
    monkeypatch.setattr(run,"package_versions",lambda:state["packages"])
    source=tmp_path/"fixed-source.npz"; c=np.zeros((1660,3),np.float32); c[:,0]=1
    np.savez(source,coefficients32=c,coefficients64=c.astype(np.float64))
    artifact=store("leg-artifact.json",{"generated":True})
    freeze={"schema":run.SCHEMA,"contract":copy.deepcopy(run.CONTRACT),"approved_sources":owned,"inherited_sources":inherited,
        "e270_freeze":of,"e270_source_review":oreview,"e277_freeze":ef,"e277_source_review":er,"e286_freeze":sf,"e286_source_review":sr,
        "source_candidate":{"path":source.name,"sha256":run.base.sha256(source)},"packages":state["packages"],
        **{k:original[k] for k in ("inputs","contexts","targets","legs","identity")},
        "leg_results":[{"leg":leg,**{k:artifact for k in ("final_startup","final_teacher","neutral_startup","neutral_teacher")}} for leg in run.base.LEGS]}
    for key,value in zip(("e270_root_result","e277_root_result","e279_transfer_audit","e286_final_audit"),audit_fixture(),strict=True): freeze[key]=store(key+".json",value)
    def save():
        f=store("freeze.json",freeze); r=store("review.json",{"schema":run.REVIEW_SCHEMA,"verdict":"approved","approved_sources":owned,"freeze_sha256":f["sha256"]})
        return tmp_path/f["path"],f["sha256"],tmp_path/r["path"],r["sha256"]
    return freeze,original,save,tmp_path,dependency_calls


def test_frozen_source34_and_two_review_chains_preserved(freeze_fixture):
    freeze,original,save,_,calls=freeze_fixture
    actual,old=run.validate_freeze(*save())
    assert actual==freeze and old==original and calls==["E277","E286"] and len(run.REQUIRED_INHERITED_SOURCES)==34


@pytest.mark.parametrize("kind",["owned","inherited","contract","package","original_input","e270_review","e270_identity","e277_sha","e286_sha","audit_sha","source_sha","leg_order","leg_sha"])
def test_missing_tampered_provenance_rejected_before_execution(freeze_fixture,kind):
    freeze,_,save,tmp,_=freeze_fixture
    if kind=="owned": (tmp/run.SOURCES[0]).write_text("changed")
    elif kind=="inherited": (tmp/run.REQUIRED_INHERITED_SOURCES[0]).write_text("changed")
    elif kind=="contract": freeze["contract"]["max_solver_calls"]=2
    elif kind=="package": freeze["packages"]={"changed":True}
    elif kind=="original_input": freeze["inputs"]={}
    elif kind=="e270_review": freeze["e270_source_review"]["sha256"]="0"*64
    elif kind=="e270_identity": freeze["e270_freeze"]=freeze["e286_freeze"]
    elif kind=="e277_sha": freeze["e277_freeze"]["sha256"]="0"*64
    elif kind=="e286_sha": freeze["e286_source_review"]["sha256"]="0"*64
    elif kind=="audit_sha": freeze["e279_transfer_audit"]["sha256"]="0"*64
    elif kind=="source_sha": freeze["source_candidate"]["sha256"]="0"*64
    elif kind=="leg_order": freeze["leg_results"]=freeze["leg_results"][::-1]
    else: freeze["leg_results"][0]=copy.deepcopy(freeze["leg_results"][0]); freeze["leg_results"][0]["final_startup"]["sha256"]="0"*64
    with pytest.raises((ValueError,KeyError)): run.validate_freeze(*save())


def full_static_fixture():
    muscle_leg=np.repeat(np.arange(6,dtype=np.int64),16); counts=np.full(96,4); fixed=np.zeros(96,bool); ltms=np.arange(0,96,16,dtype=np.int64)
    counts[ltms]=8; fixed[ltms]=True
    singleton=np.array([1,2,3,17,18,33,34,49,50,65,66,81,82]); counts[singleton]=1; fixed[singleton]=True
    free=np.flatnonzero(~fixed); counts[free[:22]]+=1
    member=np.repeat(np.arange(96,dtype=np.int64),counts); assert len(member)==391
    motor_leg=muscle_leg[member]; pool=np.zeros((96,391)); pool[member,np.arange(391)]=1/counts[member]
    low=np.zeros((391,3)); low[:,1]=pool[member,np.arange(391)]
    qleg=np.repeat(np.arange(6,dtype=np.int64),7); moment=np.zeros((42,96))
    for leg in range(6): moment[np.flatnonzero(qleg==leg),np.flatnonzero(muscle_leg==leg)[1]]=1
    edgeleg=np.repeat(np.arange(6,dtype=np.int64),[300,282,254,261,268,295]); post=np.array([np.flatnonzero(motor_leg==leg)[0] for leg in edgeleg],np.int64)
    mapping={"motor_ids":np.arange(391,dtype=np.int64)+10,"source_ids":np.arange(51,dtype=np.int64)+500,
        "edge_ids":np.arange(1660,dtype=np.int64)+1000,"edge_post":post,"edge_source":np.zeros(1660,np.int64),"weights":np.ones(1660,np.float32),
        "membership":member,"pool":pool,"low":low,"high":low.copy(),"kappa":np.ones(391),"fixed":fixed,"moment":moment,"rest":np.zeros(42),"ltm":ltms,
        "bank0":np.zeros((391,3)),"activation0":np.zeros(96),"original_activation0":np.zeros(96),"stage_order":np.array(["start","half1","half2","end"])}
    for key in ("tau","coefficient","threshold","cap","cell_gain","cell_inverse_adaptation_tau"): mapping[key]=np.ones(391,np.float32)
    return mapping,{"motor":motor_leg,"muscle":muscle_leg,"edge":edgeleg,"q":qleg,"ltm":muscle_leg[ltms]}


def test_full_original_loader_schema330_not_source_allowed_roster(monkeypatch):
    # Full static axes only. Stage arrays are shape sentinels: no waveforms,
    # model preparation/integration or full-size solver is executed.
    mapping,legs=full_static_fixture(); values={**mapping,**{"leg_"+k:v for k,v in legs.items()},
        "startup_stage":SimpleNamespace(shape=(8000,391)),"teacher_stage":SimpleNamespace(shape=(160000,391)),
        "startup_q":np.zeros((99,42)),"startup_ltm":np.zeros((99,6)),"teacher_q":np.zeros((2001,42))}
    contexts={name:{"mapping":{**{k:k for k in mapping},"stage_total":name+"_stage"}} for name in ("startup","teacher")}
    original={"contexts":contexts,"legs":{k:"leg_"+k for k in legs},"targets":{k:k for k in ("startup_q","startup_ltm","teacher_q")}}
    source={"coefficients32":np.column_stack((np.ones(1660,np.float32),np.zeros((1660,2),np.float32))),"coefficients64":np.zeros((1660,3))}
    def load(spec):
        if spec=="source": return source
        size=101 if "startup" in spec else 2001
        return {"q":np.full((size,7),int(spec[0])+1.,np.float64),"ltm":np.full((size,1),int(spec[0])+.1,np.float64)}
    freeze={"source_candidate":"source","leg_results":[{"leg":leg,**{k:f"{index}-{k}" for k in
        ("final_startup","final_teacher","neutral_startup","neutral_teacher")}} for index,leg in enumerate(run.base.LEGS)]}
    monkeypatch.setattr(run.base,"load_ref",lambda _,ref:values[ref]); monkeypatch.setattr(run,"load_npz",load)
    maps,actual_legs,targets,c32,c64,references,baselines=run.load_inputs(freeze,original)
    assert len(fit.decoder_layout(maps["startup"],actual_legs)["free_motor"])==330
    np.testing.assert_array_equal(c64,c32.astype(np.float64)); assert np.all(c64[:,0]==1)
    for index in range(6):
        assert np.all(references["teacher"]["q"][:,legs["q"]==index]==index+1)
        assert np.all(references["startup"]["ltm"][:,legs["ltm"]==index]==index+.1)
    assert targets["teacher_q"].shape==baselines["teacher"].shape==(2001,42)
