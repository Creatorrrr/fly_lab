"""E302 generated fake-only guards; no actual neural engine or CUDA entry."""
import copy
import gc
import json
import time
import weakref
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import probe_c_banc_m3_recovery_first_fault as probe

COST={"fake_constructor_requested":0,"fake_constructor_completed":0,"fake_constructor_failed":0,
      "fake_constructor_wall_s":0.,"fake_advance_requested":0,"fake_advance_completed":0,"fake_advance_failed":0,
      "fake_advance_wall_s":0.,"fake_factor_requested":0,"fake_factor_completed":0,"fake_factor_failed":0,
      "fake_finish_requested":0,"fake_finish_completed":0,"fake_finish_failed":0,"fake_kernel_wall_s":0.,
      "fake_logical_ticks":0,"fake_cleanup_requested":0,"fake_cleanup_completed":0,"fake_cleanup_failed":0,
      "setup_ticks":0,"capture_calls":0,"actual_neural_constructors":0,"original_calls":0,
      "CUDA_calls":0,"body_calls":0,"solver_calls":0,"fit_calls":0}


@pytest.fixture(scope="module",autouse=True)
def bounded_generated_cost():
    wall,cpu=time.perf_counter(),time.process_time()
    with threadpool_limits(limits=1):yield
    memory=psutil.Process().memory_info()
    COST.update(wall_s=time.perf_counter()-wall,process_s=time.process_time()-cpu,
                peak_rss_bytes=max(memory.rss,getattr(memory,"peak_wset",0)))
    print("E302_GENERATED_COST "+json.dumps(COST,sort_keys=True))
    assert COST["wall_s"]<=45 and COST["peak_rss_bytes"]<=1024**3
    assert sum(COST[key] for key in ("actual_neural_constructors","original_calls","CUDA_calls","body_calls","solver_calls","fit_calls"))==0


class FakeNetwork:
    def __init__(self,*,bad_stage=None,bad_tick=1,kind="negative",finish_bad=False,factor_mismatch=False,kernel_error=None):
        began=time.perf_counter();COST["fake_constructor_requested"]+=1
        self.rate=np.zeros(4,np.float32);self.adaptation=np.zeros(4,np.float32)
        self.drive=np.zeros(4,np.float32);self.output_mask=np.ones(4,np.float32)
        self.resource=np.ones(2,np.float64);self.resource_factor=np.ones(2,np.float32)
        self._resource_fault=np.zeros(1,np.int32);self._fused_buffers=[np.zeros(4,np.float32) for _ in range(11)]
        self.selection={"source_ids":np.array([0,2],np.int64),"utilization":np.array([.05,0.],np.float64),
                        "recovery_tau":np.full(2,.15,np.float64)}
        self.tick=0;self.graph=object();self.events=[];self.bad_stage=bad_stage;self.bad_tick=bad_tick;self.kind=kind
        self.finish_bad=finish_bad;self.factor_mismatch=factor_mismatch;self.kernel_error=kernel_error
        self.stage=0
        COST["fake_constructor_completed"]+=1;COST["fake_constructor_wall_s"]+=time.perf_counter()-began

    def _recovery_launch(self,name,tensors,step=None):
        began=time.perf_counter();kind="factor" if name=="recovery_factor" else "finish" if name=="recovery_finish" else None
        if kind:COST["fake_"+kind+"_requested"]+=1
        try:
            self.events.append((name,self.tick+1,self.stage))
            if self.kernel_error is not None and name=="recovery_factor":raise self.kernel_error
            if name=="recovery_factor":
                r,R=tensors[1][self.selection["source_ids"]],tensors[2]
                u=self.selection["utilization"];tau=self.selection["recovery_tau"]
                flags=np.zeros(2,np.int32)
                flags[(~np.isfinite(R))|(R<0)|(R>1)]|=1
                flags[(u==0)&(R!=1)]|=2
                flags[(u>0)&((~np.isfinite(r))|(r<0))]|=4
                with np.errstate(invalid="ignore"):
                    a=np.ones(2,np.float64);a[0]=R[0]*(1+tau[0]*u[0]*float(r[0]))
                    flags[(~np.isfinite(a))|(a<0)]|=8
                    self.resource_factor[:]=np.where(a<1,a,1).astype(np.float32)
                self.resource_factor[flags!=0]=np.array(0x7fffffff,np.uint32).view(np.float32)
                self._resource_fault[0]|=np.bitwise_or.reduce(flags)
                if self.factor_mismatch:self.resource_factor[1]=np.float32(.5)
            if name=="recovery_finish" and self.finish_bad:
                tensors[8][0]=1.01;self._resource_fault[0]|=16
            if kind:COST["fake_"+kind+"_completed"]+=1
        except Exception:
            if kind:COST["fake_"+kind+"_failed"]+=1
            raise
        finally:COST["fake_kernel_wall_s"]+=time.perf_counter()-began

    def advance(self,drive,steps):
        began=time.perf_counter();COST["fake_advance_requested"]+=1
        try:
            assert steps==1 and self.graph is None
            self.drive[:]=drive
            for self.stage in range(4):
                r=self.rate.copy();R=self.resource.copy();self._fused_buffers[2][:]=self.stage+.125
                if self.bad_stage==self.stage and self.bad_tick==self.tick+1:
                    if self.kind=="negative":r[0]=-2e-5
                    elif self.kind=="negative_subnormal":r[0]=-np.nextafter(np.float32(0),np.float32(1))
                    elif self.kind=="rate_nan":r[0]=np.nan
                    elif self.kind=="Rnan":R[0]=np.nan
                    elif self.kind=="Rnegative":R[0]=-.1
                    elif self.kind=="Rlarge":R[0]=1.01
                    elif self.kind=="u0R":R[1]=.99
                args=(self.selection["source_ids"],r,R,self.selection["utilization"],self.selection["recovery_tau"],self.resource_factor,self._resource_fault)
                self._recovery_launch("recovery_factor",args)
                self.events.append(("CSR",self.tick+1,self.stage))
                if self.stage<3:self._recovery_launch("recovery_stage",args,.000125)
                else:
                    final_args=(*args[:5],np.zeros(2),np.zeros(2),np.zeros(2),self.resource,self._resource_fault)
                    self._recovery_launch("recovery_finish",final_args,.00025/6)
            self.events.append(("neural_finish",self.tick+1,4));self.tick+=1
            COST["fake_logical_ticks"]+=1;COST["fake_advance_completed"]+=1
        except Exception:COST["fake_advance_failed"]+=1;raise
        finally:COST["fake_advance_wall_s"]+=time.perf_counter()-began


def copied(value):return value.copy()


def observe(net,**kwargs):return probe.StageObserver(net,host_copy=copied,**kwargs)


@pytest.mark.parametrize("stage",range(4))
def test_first_bad_stage_executes_original_factor_once_then_stops_before_CSR(stage):
    n=FakeNetwork(bad_stage=stage);graph=n.graph;observer=observe(n)
    counts=probe.probe(n,np.arange(4,dtype=np.float32),observer=observer)
    assert counts["completed_ticks"]==0 and counts["partial_ticks"]==1 and n.tick==0
    assert observer.first["stage_index"]==stage and observer.first["actual_flags"]==4
    assert observer.first["negative_source_ids"]==[0] and observer.first["source_count"]==2 and observer.first["active_count"]==1
    assert len([item for item in n.events if item[0]=="recovery_factor"])==stage+1
    assert len([item for item in n.events if item[0]=="CSR"])==stage
    assert n.graph is graph and "_recovery_launch" not in vars(n)
    assert len(observer.ring)==min(stage,2)
    assert observer.bad["rate"].dtype==np.float32 and observer.bad["synaptic_resource"].dtype==np.float64
    assert observer.bad["adaptation"][0]==(0 if stage==0 else stage+.125)
    assert not observer.first["valid"] and observer.first["completed_tick"]==0


@pytest.mark.parametrize("kind,flags",[("negative_subnormal",4),("rate_nan",12),("Rnan",9),("Rnegative",9),("Rlarge",1),("u0R",2)])
def test_original_guard_classes_and_nonfinite_json(kind,flags):
    n=FakeNetwork(bad_stage=2,kind=kind);observer=observe(n)
    probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert observer.first["actual_flags"]==flags
    encoded=json.dumps(observer.first,allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert np.isnan(observer.bad["actual_factor"]).any()


def test_ring_last_two_valid_across_tick_boundary_and_original_override_restored():
    n=FakeNetwork(bad_stage=0,bad_tick=2);original=n._recovery_launch;n._recovery_launch=original
    observer=observe(n);counts=probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert counts["completed_ticks"]==1 and counts["partial_ticks"]==1
    assert [row["stage_index"] for row,_ in observer.ring]==[2,3]
    assert observer.first["attempted_tick"]==2 and observer.first["completed_tick"]==1
    assert n._recovery_launch is original


def test_no_fault_at_maximum20_and_owned_ring_raw():
    n=FakeNetwork();observer=observe(n);counts=probe.probe(n,np.ones(4,np.float32),observer=observer)
    assert counts["completed_ticks"]==20 and counts["partial_ticks"]==0 and observer.first is None
    assert len(observer.rows)==80 and observer.cost["factor_completed"]==80
    assert len(observer.ring)==2 and all(row["factor_nonunity_count"]==0 for row in observer.rows)
    raw=observer.ring[-1][1];n.drive.fill(8);assert np.all(raw["drive"]==1)


def test_finish_guard_stops_before_neural_finish_with_stage3_rate_and_Rbase():
    n=FakeNetwork(finish_bad=True);observer=observe(n)
    counts=probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert counts["completed_ticks"]==0 and n.tick==0
    assert observer.first["phase"]=="after_resource_finish_before_neural_finish" and observer.first["stage_index"]==4
    assert observer.first["actual_flags"]==16 and observer.bad["synaptic_resource"][0]==1.01
    assert observer.bad["adaptation"][0]==3.125
    assert not any(row[0]=="neural_finish" for row in n.events)


def test_valid_actual_factor_mismatch_is_invalid_execution_and_preserves_pending():
    n=FakeNetwork(factor_mismatch=True);observer=observe(n)
    with pytest.raises(ValueError,match="factor32") as error:probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert error.value.e302_probe_counts["advance_failed"]==1 and observer.pending is not None
    assert observer.cost["factor_completed"]==1 and observer.cost["factor_failed"]==1 and observer.first is None
    assert not any(row[0]=="CSR" for row in n.events)


def test_kernel_failure_preserves_first_error_and_counts_restore():
    primary=OSError("factor kernel");n=FakeNetwork(kernel_error=primary);observer=observe(n);graph=n.graph
    with pytest.raises(OSError) as error:probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert error.value is primary and observer.cost["factor_requested"]==1 and observer.cost["factor_completed"]==0
    assert observer.cost["factor_failed"]==1 and observer.pending is not None and n.graph is graph


def test_negative_zero_positive_subnormal_and_exact_formula_bytes():
    smallest=np.nextafter(np.float32(0),np.float32(1));r=np.array([smallest,-0.,0.],np.float32)
    sources=np.arange(3,dtype=np.int64);R=np.array([1.,1.,float(smallest)],np.float64)
    u=np.array([.05,.05,.05]);tau=np.full(3,.15)
    actual,flags=probe.independent_factor(r,R,sources,u,tau)
    assert flags==0 and actual.tobytes()==np.array([1,1,smallest],np.float32).tobytes()


@pytest.mark.parametrize("ticks",[0,21,-1,True,1.5])
def test_prefix_limit_cannot_be_relaxed(ticks):
    n=FakeNetwork();observer=observe(n)
    with pytest.raises(ValueError):probe.probe(n,np.zeros(4,np.float32),observer=observer,max_ticks=ticks)
    assert not observer.installed


def test_stage_deadline_has_no_kernel_entry():
    n=FakeNetwork();observer=observe(n,clock=lambda:1.,deadline=1.)
    with pytest.raises(TimeoutError) as error:probe.probe(n,np.zeros(4,np.float32),observer=observer)
    assert not n.events and error.value.e302_probe_counts["partial_ticks"]==1
    assert observer.cost["factor_requested"]==0 and not observer.installed


class FakeDependencies:
    def __init__(self,*,fault=None,load_error=None,construction_error=None,post_return_error=None,validation_error=None,
                 copy_error=None,cleanup_error=None,resource_error=None):
        self.fault=fault or {};self.load_error=load_error;self.construction_error=construction_error
        self.post_return_error=post_return_error;self.validation_error=validation_error;self.copy_error=copy_error
        self.cleanup_error=cleanup_error;self.resource_error=resource_error;self.refs=[];self.cleanup_live=None

    def load(self,freeze):
        if self.load_error:raise self.load_error
        return {"drive":np.zeros(4,np.float32)}

    def construct(self,data,rows,persist,on_return):
        if self.construction_error:raise self.construction_error
        n=FakeNetwork(**self.fault);self.refs.append(weakref.ref(n));rows.append({"known_completed_setup_steps":0,"recorded_capture_setup_steps":0,"generated_fake":True})
        on_return(n)
        if self.post_return_error:raise self.post_return_error
        return n

    def validate_initial(self,network,data):
        if self.validation_error:raise self.validation_error
        assert network.tick==0

    def host_copy(self,value):
        if self.copy_error:raise self.copy_error
        return value.copy()

    def cleanup(self):
        COST["fake_cleanup_requested"]+=1
        try:
            gc.collect();self.cleanup_live=sum(ref() is not None for ref in self.refs)
            assert self.cleanup_live==0
            if self.cleanup_error:raise self.cleanup_error
            COST["fake_cleanup_completed"]+=1
        except Exception:COST["fake_cleanup_failed"]+=1;raise

    def resource_check(self,output):
        if self.resource_error and self.cleanup_live is not None:raise self.resource_error
        return {"generated_fake":True,"bytes":sum(file.stat().st_size for file in output.rglob("*") if file.is_file())}


def test_execute_valid_first_fault_and_disk_raw_partial_scope(tmp_path):
    deps=FakeDependencies(fault={"bad_stage":2});result,error=probe.execute({},tmp_path/"out",dependencies=deps)
    assert error is None and result["run_valid"] and result["first_fault_observed"]
    assert result["all_valid_factors_unity_before_first_fault"] and deps.cleanup_live==0
    assert result["candidate_failed_drive_directly_measured"] is False and result["e301_cumulative_flags"]==29
    for flag in ("candidate_adopted","F1_F8_complete","F2_complete","F6_exposed","biological_claim"):assert result[flag] is False
    with np.load(tmp_path/"out"/"first-invalid.npz") as raw:
        assert set(raw.files)=={"rate","adaptation","drive","output_mask","synaptic_resource","source_ids","utilization","recovery_tau","actual_factor"}
        assert raw["rate"][0]<0 and raw["actual_factor"].dtype==np.float32
    summary=json.loads((tmp_path/"out"/"stage-summary.json").read_text())
    assert len(summary["last_valid_rows"])==2 and "NOT" in summary["raw_kind"]
    manifest=json.loads((tmp_path/"out"/"output-manifest.json").read_text())
    assert "result.json" not in manifest["files"] and "first-invalid.npz" in manifest["files"]
    assert "source-manifest.json" in manifest["files"]
    assert any(row["phase"]=="after_constructor_while_owned" for row in result["cost"]["resource_samples"])


def test_execute_valid_no_fault_and_final_guard(tmp_path):
    for name,kwargs,found in (("normal",{},False),("final",{"finish_bad":True},True)):
        deps=FakeDependencies(fault=kwargs);result,error=probe.execute({},tmp_path/name,dependencies=deps)
        assert error is None and result["run_valid"] and result["first_fault_observed"] is found
        assert deps.cleanup_live==0


@pytest.mark.parametrize("phase",["load","construction","post_return","validation","copy","cleanup","resource"])
def test_actual_owner_release_on_each_error_and_first_exception_identity(tmp_path,phase):
    primary=OSError(phase);key={"load":"load_error","construction":"construction_error","post_return":"post_return_error",
        "validation":"validation_error","copy":"copy_error","cleanup":"cleanup_error","resource":"resource_error"}[phase]
    deps=FakeDependencies(fault={"bad_stage":0},**{key:primary})
    result,error=probe.execute({},tmp_path/phase,dependencies=deps)
    assert error is primary and not result["run_valid"] and deps.cleanup_live==0
    assert error.e302_partial_result is result and error.e302_ledger is not None
    assert result["cost"]["construction_requested"]==(0 if phase=="load" else 1)
    assert result["cost"]["construction_completed"]==(0 if phase in ("load","construction") else 1)


class FailingWriter(probe.Writer):
    def __init__(self,output,*,name,error):super().__init__(output);self.name=name;self.error=error

    def json(self,name,value):
        if name==self.name:raise self.error
        return super().json(name,value)

    def raw(self,name,value):
        if name==self.name:raise self.error
        return super().raw(name,value)


@pytest.mark.parametrize("name",["ledger.json","stage-summary.json","first-invalid.npz","result.json","source-manifest.json","output-manifest.json"])
def test_publication_failure_keeps_partial_and_cleanup_without_retry(tmp_path,name):
    out=tmp_path/"out";primary=OSError(name);writer=FailingWriter(out,name=name,error=primary)
    deps=FakeDependencies(fault={"bad_stage":2})
    result,error=probe.execute({},out,dependencies=deps,writer=writer)
    assert error is not None and not result["run_valid"] and deps.cleanup_live==0
    assert result["cost"]["construction_requested"]==(0 if name in ("ledger.json","source-manifest.json") else 1)
    assert len(deps.refs)<=1
    if name=="result.json":assert error is primary and result["cost"]["science"]["partial_ticks"]==1


def test_first_kernel_error_survives_secondary_storage_cleanup_and_clock(tmp_path):
    primary=RuntimeError("first kernel");secondary=OSError("cleanup")
    deps=FakeDependencies(fault={"kernel_error":primary},cleanup_error=secondary)
    writer=FailingWriter(tmp_path/"out",name="stage-summary.json",error=OSError("store"))
    result,error=probe.execute({},tmp_path/"out",dependencies=deps,writer=writer)
    assert error is primary and not result["run_valid"] and len(error.__notes__)>=2
    assert result["cost"]["observer"]["factor_failed"]==1 and deps.cleanup_live==0
    record={"wall":0.}
    def broken_clock():raise OSError("clock")
    try:raise primary
    except RuntimeError:
        probe.charge_wall(record,"wall",0.,broken_clock)
        assert record["wall_errors"] and "clock" in primary.__notes__[-1]


def evidence():
    old={"schema":"flylab.m3.e301.recovery-resource-body.freeze.v1","diagnostic_id":"E301","source_network_identity":probe.SOURCE_IDENTITY}
    result={"diagnostic_id":"E301","run_valid":False,"neutral_exact":True,"candidate_adopted":False,"F1_F8_complete":False,"F6_exposed":False,
        "cost":{"scientific_neural_ticks":40,"scientific_physics_steps":50},"first_error":{"message":"Sticky recovery resource fault: 29"},
        "arms":{"candidate":{"completed_controls":0}}}
    audit={**copy.deepcopy(result),"schema":"flylab.m3.e301.root-original-raw-audit.v1","pass":True,
        "arms":{"candidate":{"controls":0,"native_samples":0,"actual_neural_ticks":20,"whole_start_exact":True,"whole_final_available":False,
            "resource_faults":[{"flags":29,"actual_control_tick":0}]},"neutral":{"whole_scientific_endpoint_exact":True}}}
    ledger={"arms":{"candidate":{"clock_events":[{"neural_tick_before":0,"neural_tick_after":20,"physics_time_after_s":0.,"control_tick_after":0}]}}}
    return old,result,ledger,audit,{"files":{}},{"diagnostic_id":"E301"}


def test_e301_evidence_valid_failure_is_not_valid_trajectory():
    probe.validate_e301_evidence(*evidence())


@pytest.mark.parametrize("section,key,value",[(1,"run_valid",True),(1,"neutral_exact",False),(1,"candidate_adopted",True),
    (3,"pass",False),(3,"run_valid",True),(3,"F1_F8_complete",True),(0,"source_network_identity","bad")])
def test_e301_evidence_rejects_altered_meaning(section,key,value):
    values=list(evidence());values[section][key]=value
    with pytest.raises(ValueError):probe.validate_e301_evidence(*values)


@pytest.mark.parametrize("changed",["physics","tick","fault","cost"])
def test_e301_boundary_and_cost_not_interchangeable(changed):
    values=list(evidence())
    if changed=="physics":values[2]["arms"]["candidate"]["clock_events"][0]["physics_time_after_s"]=.005
    if changed=="tick":values[3]["arms"]["candidate"]["actual_neural_ticks"]=0
    if changed=="fault":values[3]["arms"]["candidate"]["resource_faults"][0]["flags"]=4
    if changed=="cost":values[1]["cost"]["scientific_neural_ticks"]=20
    with pytest.raises(ValueError):probe.validate_e301_evidence(*values)


@pytest.fixture
def frozen_fixture(tmp_path):
    from tools import verify_c_banc_m3_recovery_resource_body as e301
    source_names={*probe.OWNED,*e301.SOURCE_FILES};sources={}
    def put(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value,encoding="utf-8");return probe.sha256(path)
    for name in source_names:sources[name]=put(name,"generated source pin "+name)
    design=tmp_path/probe.DESIGN;design.parent.mkdir(parents=True,exist_ok=True);design.write_bytes((probe.ROOT/probe.DESIGN).read_bytes())
    inputs={key:"generated/"+key for key in probe.INPUTS}
    inputs.update(candidate_start="verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/candidate/observed-start",
        neutral_start="verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/neutral/observed-start",
        neutral_final="verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/neutral/final")
    input_pins={name:put(name,"generated input "+name) for name in inputs.values()}
    old,result,ledger,audit,manifest,source=evidence()
    old.update(paths={"neutral_checkpoint":inputs["original_start"],"protected_selection":inputs["protected_roster"]},
        candidate_network_identity="b"*64,reused_source_sha256=sources,input_sha256={},preservation_sha256={})
    source["source_sha256"]=sources
    preserve={"generated/preservation.json":put("generated/preservation.json","preserved")}
    receipts={}
    values={"e301_freeze":old,"e301_result":result,"e301_ledger":ledger,"e301_root_audit":audit,
            "e301_output_manifest":manifest,"e301_source_manifest":source}
    for alias,value in values.items():
        name="verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/"+alias+".json"
        digest=put(name,json.dumps(value));receipts[alias]={"path":name,"sha256":digest};input_pins[name]=digest
    oldreview={"schema":"flylab.m3.e301.recovery-resource-body.review.v1","approved":True,"blocking_defects":[],
               "freeze_sha256":receipts["e301_freeze"]["sha256"],"source_sha256":{name:sources[name] for name in e301.OWNED}}
    name="generated/e301-review.json";digest=put(name,json.dumps(oldreview));receipts["e301_review"]={"path":name,"sha256":digest};input_pins[name]=digest
    freeze={"schema":probe.FREEZE_SCHEMA,"diagnostic_id":"E302","owned_files":list(probe.OWNED),"limits":probe.LIMITS,
        "inputs":inputs,"output":probe.OUTPUT_PATH,"source_network_identity":probe.SOURCE_IDENTITY,"candidate_network_identity":"b"*64,
        "drive_inference":probe.INFERENCE,"candidate_adopted":False,"design":probe.DESIGN,"design_sha256":probe.DESIGN_SHA,
        "source_sha256":sources,"input_sha256":input_pins,"preservation_sha256":preserve,**receipts,
        "packages":dict.fromkeys(("python","numpy","scipy","torch","cupy-cuda13x","psutil","threadpoolctl"),"fake")}
    path=tmp_path/"freeze.json";reviewpath=tmp_path/"review.json";output=tmp_path/probe.OUTPUT_PATH;output.parent.mkdir(parents=True,exist_ok=True)
    def seal():
        path.write_text(json.dumps(freeze),encoding="utf-8");fsha=probe.sha256(path)
        review={"schema":probe.REVIEW_SCHEMA,"approved":True,"blocking_defects":[],"freeze_sha256":fsha,
                "source_sha256":{name:freeze["source_sha256"][name] for name in probe.OWNED}}
        reviewpath.write_text(json.dumps(review),encoding="utf-8")
        return path,fsha,reviewpath,probe.sha256(reviewpath),output
    return freeze,seal,tmp_path


def test_strict_freeze_provenance_accepts_generated_pins(frozen_fixture):
    freeze,seal,root=frozen_fixture
    actual=probe.gate(*seal(),root=root,version_reader=lambda _:"fake")
    assert actual==freeze


@pytest.mark.parametrize("field",["inputs","limits","drive_inference","source_network_identity","candidate_adopted"])
def test_strict_contract_fields_missing_or_altered(frozen_fixture,field):
    freeze,seal,root=frozen_fixture;freeze.pop(field)
    with pytest.raises((ValueError,KeyError)):probe.gate(*seal(),root=root,version_reader=lambda _:"fake")


@pytest.mark.parametrize("target",["input","source","receipt","package","output","chain","original_path"])
def test_tampered_bytes_receipts_packages_chain_output_rejected(frozen_fixture,target):
    freeze,seal,root=frozen_fixture
    if target=="input":(root/freeze["inputs"]["calibration"]).write_bytes(b"tampered")
    if target=="source":(root/probe.OWNED[0]).write_bytes(b"tampered")
    if target=="receipt":freeze["e301_result"]["sha256"]="0"*64
    if target=="package":freeze["packages"]["numpy"]="notfake"
    if target=="chain":freeze["source_sha256"].pop("tools/banc_m3_recovery_resource_synapses.py")
    if target=="original_path":freeze["inputs"]["original_start"]=freeze["inputs"]["neutral_start"]
    args=seal()
    if target=="output":args[-1].mkdir()
    with pytest.raises((ValueError,KeyError)):probe.gate(*args,root=root,version_reader=lambda _:"fake")


def test_literal_output_no_escape_and_json_nonfinite(tmp_path):
    for name in ("../escape","C:/escape","/escape"):
        with pytest.raises(ValueError):probe.safe(name,tmp_path)
    output=tmp_path/"out";output.mkdir();writer=probe.Writer(output)
    writer.json("raw.json",{"a":np.nan,"b":np.inf})
    assert json.loads((output/"raw.json").read_text())=={"a":None,"b":None}


def test_constructor_injection_preserves_native_args_without_real_engine():
    calls=[];owner=[]
    data={"network":{"core_adaptation":{"ids":np.array([0]),"gain":0.},"motor_adaptation":{"ids":np.array([3]),"gains":np.ones(1,np.float32)},
           "motor_timescale":{"log_tau":np.zeros(1,np.float64)}},"selection":{"edge_ids":np.array([1]),"premotor_ids":np.array([0,1,2]),
           "utilization":np.array([.05]),"recovery_tau":np.array([.15])},"configuration":{"membrane_tau":np.full(4,.02,np.float32)},"weights":"tiny-csr"}
    def factory(*args,**kwargs):calls.append((args,kwargs));return object()
    def profiler(label,callback,rows,persist,on_return):
        rows.append({"label":label,"generated_fake":True});persist();n=callback();on_return(n);return n
    deps=probe.NativeDependencies(engine=SimpleNamespace(RecoveryResourceNetwork=factory),profiler=profiler)
    deps.construct(data,[],lambda:None,owner.append)
    args,kwargs=calls[0]
    assert args[0]=="tiny-csr" and len(calls)==1 and len(owner)==1
    assert kwargs["device"]=="cuda" and kwargs["cuda_implementation"]=="packed" and kwargs["capture_steps"]==20 and kwargs["dt"]==.00025
    assert kwargs["adaptation_gain"]==8 and kwargs["adaptation_tau_s"]==.3
    assert np.all(args[2]==1) and np.all(args[3]==7.5) and np.all(args[4]==200)
