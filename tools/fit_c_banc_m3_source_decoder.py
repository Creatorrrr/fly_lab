"""E287 frozen source + one global monotone decoder refit; no body promotion."""
from __future__ import annotations

# Preserve primary errors when cleanup or storage also fails.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import banc_m3_source_decoder_fit as fit
from tools import fit_c_banc_m3_certified_source as certified
from tools import verify_c_banc_m3_feasibility_restoration as restoration
from tools import verify_c_banc_m3_rf_state_attribution as attribution
from tools import verify_c_banc_m3_startup_margin as startup

base=fit.base
SCHEMA,REVIEW_SCHEMA="flylab.m3.e287.source-decoder-fit.v1","flylab.m3.e287.source-review.v1"
SOURCES=("tools/banc_m3_source_decoder_fit.py","tools/fit_c_banc_m3_source_decoder.py","tests/test_c_banc_source_decoder_fit.py")
REQUIRED_INHERITED_SOURCES=(*attribution.REQUIRED_INHERITED_SOURCES,*attribution.SOURCES)
CONTRACT={**copy.deepcopy(fit.CONTRACT),"identity":{"motors":391,"sources":51,"edges":1660,"muscles":96,"joints":42,"ltm":6},
    "decoder_free_motors":330,"decoder_variable_muscles":77,"decoder_variables":3960,
    "engine_samples":{"startup":101,"teacher":2001},"max_engines":2,"max_contexts_live":1,"max_forward_ticks":42000,
    "max_derivative_ticks":0,"host_copy_keys":["bank","rate","original_activation","q","ltm"],
    "science_wall_s":50.,"cleanup_reserve_s":10.,"supervisor_wall_s":60.,"ram_bytes":6*1024**3,
    "gpu_additional_bytes":4*1024**3,"output_bytes":512*1024**2,"free_disk_bytes":2*1024**3,
    "generated_cpu_wall_s":120.,"generated_tiny_solves":20,"generated_max_variables":60,
    "development_evaluations":1,"development_order":"candidate NPZ + SHA immutable save before one held-out evaluation",
    "whole_BANC_calls":0,"physics_steps":0,"source_fit_calls":0,"candidate_adopted":False,"body_promoted":False}
package_versions=startup.package_versions
verify_spec=attribution.verify_spec


def load_npz(spec):
    with np.load(verify_spec(spec),allow_pickle=False) as z: return {k:z[k] for k in z.files}


def validate_prior_evidence(e270,e277,e279,e286):
    base.require(e270.get("schema")=="flylab.m3.e270.original-final-review.v1" and e270.get("run_valid") is True
        and e270.get("verdict")=="rejected_under_frozen_conditional_gates" and e270.get("conditional_support") is False,
        "Original E270 valid rejection required")
    base.require(e277.get("schema")=="flylab.m3.e277.root-original-final-review.v1" and e277.get("run_valid") is True
        and e277.get("verdict")=="valid_reject" and e277.get("conditional_candidate_supported") is False
        and e277.get("unchanged_final_checks",{}).get("train") is False
        and e277.get("unchanged_final_checks",{}).get("LM_development") is False,"E277 train/LM rejection must remain visible")
    base.require(e279.get("schema")=="flylab.m3.e279.root-original-raw-audit.v1" and e279.get("supervisor_success") is True
        and e279.get("E277_reject_preserved") is True and e279.get("integrity")=="verified_native_diagnostic",
        "Observed E279 native transfer audit required")
    candidate=e279.get("arms",{}).get("candidate",{}); prediction=candidate.get("prediction",{}).get("q",{})
    base.require(prediction.get("RMS")==.00013667128182327153 and prediction.get("max_abs")==.0015168425596647772
        and candidate.get("strict_startup") is False,"Exact observed empirical margin required;not a guaranteed transfer bound")
    base.require(e286.get("schema")=="flylab.m3.e286.root-original-audit.v1" and e286.get("verdict")=="valid_state_attribution"
        and e286.get("H286")=="supported" and e286.get("independent_decoder_and_stage_identity_checks") is True
        and e286.get("original_reproduction_verified") is True and e286.get("candidate_created") is False,
        "E286 valid state attribution must be preserved")
    base.require(all(g.get("F6_exposed") is False for g in (e270,e277,e279,e286)),"Unexposed F6 must remain unused")


def validate_freeze(path,digest,review_path,review_digest):
    freeze,review=base.verified_json(path,digest),base.verified_json(review_path,review_digest)
    base.require(freeze.get("schema")==SCHEMA and freeze.get("contract")==CONTRACT,"Exact E287 prospective contract required")
    sources=freeze.get("approved_sources"); pins=freeze.get("inherited_sources")
    base.require(isinstance(sources,dict) and set(sources)==set(SOURCES),"Exactly three reviewed E287 sources required")
    for name,sha in sources.items(): base.require(base.sha256(base.safe_path(name))==sha,"Reviewed E287 source changed: "+name)
    base.require(review.get("schema")==REVIEW_SCHEMA and review.get("verdict")=="approved" and review.get("approved_sources")==sources
        and review.get("freeze_sha256")==digest,"Exact E287 source review required")
    base.require(isinstance(pins,dict) and set(REQUIRED_INHERITED_SOURCES)<=set(pins),"All34 inherited source pins required")
    for name,sha in pins.items(): base.require(base.sha256(base.safe_path(name))==sha,"Inherited source changed: "+name)
    ef,er=freeze["e277_freeze"],freeze["e277_source_review"]
    original=certified.validate_freeze(verify_spec(ef),ef["sha256"],verify_spec(er),er["sha256"])
    for name,sha in original["approved_sources"].items(): base.require(pins.get(name)==sha,"E277 source identity differs")
    of,oreview=freeze["e270_freeze"],freeze["e270_source_review"]
    old=json.loads(verify_spec(of).read_text(encoding="utf-8")); prior_review=json.loads(verify_spec(oreview).read_text(encoding="utf-8"))
    base.require(old.get("schema")==base.SCHEMA and old.get("contract")==base.CONTRACT
        and prior_review.get("schema")==base.REVIEW_SCHEMA and prior_review.get("verdict")=="approved"
        and prior_review.get("approved_sources")==old["approved_sources"] and prior_review.get("freeze_sha256")==of["sha256"],"Preserved original E270 review required")
    original_e270=original["inputs"][original["e270_original_freeze"]["input"]]
    base.require(of==original_e270,"Exact E277 inherited E270 freeze required")
    for key in ("inputs","contexts","targets","legs","identity"):
        base.require(freeze.get(key)==original[key],"Original E277 input/calibration/target/leg provenance changed: "+key)
    for spec in freeze["inputs"].values(): verify_spec(spec)
    sf,sr=freeze["e286_freeze"],freeze["e286_source_review"]
    state,_=attribution.validate_freeze(verify_spec(sf),sf["sha256"],verify_spec(sr),sr["sha256"])
    base.require(all(pins.get(k)==v for k,v in {**state["approved_sources"],**state["inherited_sources"]}.items()),"Exact E286 source chain required")
    base.require(freeze.get("packages")==package_versions()==state["packages"],"Frozen installed packages required")
    gates=[json.loads(verify_spec(freeze[k]).read_text(encoding="utf-8")) for k in
        ("e270_root_result","e277_root_result","e279_transfer_audit","e286_final_audit")]
    validate_prior_evidence(*gates)
    candidate=load_npz(freeze["source_candidate"])
    base.require(set(candidate)=={"coefficients32","coefficients64"} and candidate["coefficients32"].dtype==np.float32
        and candidate["coefficients32"].shape==(1660,3) and base.coefficient_valid(candidate["coefficients32"].astype(np.float64)),"Fixed E277 native source candidate required")
    rows=freeze.get("leg_results")
    base.require(isinstance(rows,list) and len(rows)==6 and [r.get("leg") for r in rows]==list(base.LEGS),"Six original saved leg results required")
    for row in rows:
        base.require(set(row)=={"leg","final_startup","final_teacher","neutral_startup","neutral_teacher"},"Exact saved leg result schema")
        for key in ("final_startup","final_teacher","neutral_startup","neutral_teacher"): verify_spec(row[key])
    return freeze,original


class ScienceDeadline(restoration.ScienceDeadline):
    def remaining(self): return 50-(self.clock()-self.started)
    def receipt(self): return {"original_wall_s":self.clock()-self.started,"science_wall_limit_s":50.,"supervisor_wall_limit_s":60.}


class DecoderMeter(restoration.RestorationMeter):
    def __init__(self,*args):
        self.engine_creation=fit.metering.phase(); self.evaluation=fit.metering.phase(); self.development=fit.metering.phase()
        self.solver_cost={k:fit.metering.phase() for k in ("constructor","solve","certificate")}
        self.solver_cost.update(iterations=0,iterations_unknown_calls=0)
        self.forward_requested=0; self.forward_completed=0; self.selection=None
        super().__init__(*args)
    def receipt(self):
        return {**super().receipt(),"engine_creation":self.engine_creation,"evaluation":self.evaluation,"solver":self.solver_cost,
            "forward_requested_steps":self.forward_requested,"forward_completed_steps":self.forward_completed,"derivative_steps":0,
            "development":self.development,"selection":self.selection,"source_fit_calls":0,
            "fit_calls":self.solver_cost["solve"]["requested"],"decoder_fit_calls":self.solver_cost["solve"]["requested"]}
    def enforce_memory(self):
        try:
            import psutil

            usage=psutil.Process().memory_info()
            self.peak_rss=max(self.peak_rss or 0,usage.rss,getattr(usage,"peak_wset",0))
        except (ImportError,OSError):
            pass
        base.require(self.peak_estimate<=CONTRACT["ram_bytes"] and (self.peak_rss is None or self.peak_rss<=CONTRACT["ram_bytes"]),"E287 six GiB RAM bound exceeded")
        rows=[v for v in self.memory_events if "live_gpu_estimate" in v]
        if rows: base.require(rows[-1]["live_gpu_estimate"]<=CONTRACT["gpu_additional_bytes"],"E287 four GiB additional GPU bound exceeded")
    def memory(self,engine,phase):
        estimate=super().memory(engine,phase)
        try:
            base.require(self.active_contexts<=1,"One actual source engine at a time")
            self.memory_events[-1]["live_gpu_estimate"]=engine.problem["gpu_bytes_estimate"]; self.enforce_memory()
        except Exception as error:
            base.preserve_primary(error,"Rejected source engine memory release failed",lambda:self.release(engine,phase,estimate)); raise
        return estimate
    def account_solver(self,receipt):
        for key in ("constructor","solve","certificate"):
            for field in ("requested","completed","failed"): self.solver_cost[key][field]+=receipt[key][field]
            if not receipt[key]["wall_known"]: self.solver_cost[key].update(wall_s=None,wall_known=False)
            elif self.solver_cost[key]["wall_s"] is not None: self.solver_cost[key]["wall_s"]+=receipt[key]["wall_s"]
        if receipt["iterations_known"]: self.solver_cost["iterations"]+=receipt["iterations"]
        elif receipt["solve"]["requested"]: self.solver_cost["iterations_unknown_calls"]+=1
        base.require(self.solver_cost["constructor"]["requested"]<=1 and self.solver_cost["solve"]["requested"]<=1,"One solver constructor/solve;no retry")


def check_output(output):
    base.require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file())<=CONTRACT["output_bytes"],"E287 bounded output exceeded")


def persist(output,name,arrays,receipt):
    if arrays is not None: np.savez(output/(name+".npz"),**arrays)
    base.write_json(output/(name+".json"),receipt)
    check_output(output)


def measured_bytes(tree):
    """Include CSC buffers without double-counting shared NumPy owners."""
    arrays=[]
    def visit(value):
        if sparse.issparse(value): arrays.extend((value.data,value.indices,value.indptr))
        elif isinstance(value,np.ndarray): arrays.append(value)
        elif isinstance(value,dict):
            for item in value.values(): visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value: visit(item)
    visit(tree)
    return base.array_bytes(arrays)


def load_inputs(freeze,original):
    mappings={name:{k:base.load_ref(original,ref) for k,ref in ctx["mapping"].items()} for name,ctx in original["contexts"].items()}
    base.require(set(mappings)=={"startup","teacher"},"Two original contexts required")
    for name,mapping in mappings.items():
        mapping["stage_order"]=tuple(mapping["stage_order"].tolist())
        base.require(mapping["stage_total"].shape==(8000 if name=="startup" else 160000,391),"Full original native stage axes required")
        base.require([len(mapping[k]) for k in ("motor_ids","source_ids","edge_ids","fixed","rest","ltm")]==[391,51,1660,96,42,6],"Exact original motor/source/edge/muscle/q/LTM axes")
    base.validate_static_contexts(mappings["startup"],mappings["teacher"])
    legs={k:base.load_ref(original,ref) for k,ref in original["legs"].items()}; base.validate_leg_ownership(mappings["startup"],legs)
    base.require([int(np.sum(legs["edge"]==i)) for i in range(6)]==[300,282,254,261,268,295],"Exact original edge provenance")
    layout=fit.decoder_layout(mappings["startup"],legs)
    base.require(len(layout["free_motor"])==330 and len(layout["variable_muscle"])==77 and layout["variables"]==3960,"Original free decoder roster required")
    targets={k:base.load_ref(original,ref) for k,ref in original["targets"].items()}; base.validate_targets(targets)
    source=load_npz(freeze["source_candidate"]); coefficients32=source["coefficients32"].copy(); coefficients=coefficients32.astype(np.float64)
    references={name:{"q":np.empty((size,42)),"ltm":np.empty((size,6))} for name,size in (("startup",101),("teacher",2001))}
    baselines={name:np.empty((size,42)) for name,size in (("startup",101),("teacher",2001))}
    for i,row in enumerate(freeze["leg_results"]):
        q,l=np.flatnonzero(legs["q"]==i),np.flatnonzero(legs["ltm"]==i)
        for name,size in (("startup",101),("teacher",2001)):
            final=load_npz(row["final_"+name]); neutral=load_npz(row["neutral_"+name])
            for key,shape in (("q",(size,7)),("ltm",(size,1))):
                v=final[key]; base.require(v.dtype==np.float64 and v.shape==shape and np.isfinite(v).all(),"Exact saved E277 final axis")
            base.require(neutral["q"].dtype==np.float64 and neutral["q"].shape==(size,7) and np.isfinite(neutral["q"]).all(),"Exact neutral comparison baseline")
            references[name]["q"][:,q]=final["q"]; references[name]["ltm"][:,l]=final["ltm"]; baselines[name][:,q]=neutral["q"]
    return mappings,legs,targets,coefficients32,coefficients,references,baselines


def _capture_source(engine,name,coefficients,meter):
    """Publish independent host copies before releasing any device response."""
    device,primary,host=None,None,{}
    try:
        try:
            device=fit.metering.metered({"evaluation":meter.evaluation},"evaluation",
                lambda:engine.solve(coefficients,derivative=False,host=False),meter.deadline.clock)
        except Exception as error:
            primary=error
            device=getattr(error,"phase_partial_result",None)
            if not isinstance(device,dict): device=getattr(error,"partial_response",{})
            if not isinstance(device,dict): device={}
        for key in CONTRACT["host_copy_keys"]:
            try:
                base.require(key in device,"Missing current-response component: "+key)
                host[key]=engine.host_copy(device[key],label=name+":"+key)
            except Exception as error:
                partial=getattr(error,"phase_partial_result",None)
                if isinstance(partial,np.ndarray): host[key]=partial
                if primary is None: primary=error
                else: primary.add_note("Additional response copy failure "+key+": "+repr(error))
        receipt=copy.deepcopy(getattr(engine,"last_receipt",None) or device.get("receipt") or getattr(primary,"response_receipt",{}))
        if primary is not None:
            primary.response_receipt=receipt
            primary.partial_host_response=host
            primary.partial_response=host
            if hasattr(primary,"phase_partial_result"): del primary.phase_partial_result
            raise primary
        return host,receipt
    finally:
        device,engine=None,None


def acquire_context(name,mapping,coefficients,factory,meter):
    primary,engine=None,None; host={}; receipt={}; steps=(CONTRACT["engine_samples"][name]-1)*20
    def counted_factory(mapping):
        meter.engine_requests+=1
        base.require(meter.engine_requests<=2 and meter.active_contexts==0,"Exactly bounded one-live source engines")
        try:
            return fit.metering.metered({"constructor":meter.engine_creation},"constructor",lambda:factory(mapping),meter.deadline.clock)
        except Exception as error:
            interrupted=getattr(error,"phase_partial_result",None)
            if interrupted is not None:
                estimate=None
                try:
                    meter.cp=getattr(interrupted,"cp",None)
                    error.response_receipt=copy.deepcopy(getattr(interrupted,"last_receipt",interrupted.receipt))
                    estimate=meter.memory(interrupted,"interrupted-source-constructor")
                except Exception as secondary:
                    error.add_note("Interrupted constructor memory accounting failed: "+repr(secondary))
                finally:
                    if estimate is not None:
                        base.preserve_primary(error,"Interrupted constructor release failed",
                            lambda:meter.release(interrupted,"interrupted-source-constructor",estimate))
                    interrupted=None
                    del error.phase_partial_result
            raise
    try:
        with base.engine_scope(counted_factory,mapping,meter,"source-"+name) as engine:
            base.require(engine.problem["samples"]==CONTRACT["engine_samples"][name],"Full source acquisition axes")
            meter.cp=getattr(engine,"cp",None); meter.forward_requested+=steps
            base.require(meter.forward_requested<=42000,"Fixed forward neural tick cap")
            host,receipt=_capture_source(engine,name,coefficients,meter)
            persist(meter.output,"source-"+name,host,receipt)
            meter.deadline.check()
        return host
    except Exception as error:
        primary=error
        host=getattr(error,"partial_host_response",host)
        error.partial_host_response=host
        error.partial_response=host
        if host: base.preserve_primary(error,"Partial acquired source host persistence failed",lambda error=error:persist(meter.output,"source-"+name+"-partial",host,getattr(error,"response_receipt",{})))
        raise
    finally:
        if not receipt: receipt=getattr(primary,"response_receipt",{})
        meter.forward_completed+=int(receipt.get("completed_steps",0))
        engine=None
        if primary is not None: traceback.clear_frames(primary.__traceback__)
        gc.collect()
        if primary is None: meter.cleanup()
        else: base.preserve_primary(primary,"Acquired source cleanup failed",meter.cleanup)


def candidate_arrays(mapping,layout,x,coefficients32):
    low,high=fit.candidate_slopes(layout,x); n,m=len(low),len(mapping["fixed"])
    full=np.zeros((m,n,3,2)); full[mapping["membership"],np.arange(n),:,0]=low; full[mapping["membership"],np.arange(n),:,1]=high
    return {**{k:np.array(mapping[k],copy=True) for k in (*fit.DECODER_KEYS,"source_ids","edge_ids") if k not in ("low","high")},
        "tau":fit.FILTER_TAU.copy(),"low":low,"high":high,"slopes":full,"coefficients32":coefficients32.copy(),
        "coefficients64":coefficients32.astype(np.float64),"free_motor":layout["free_motor"].copy(),"x":x.copy(),"x0":layout["x0"].copy(),
        "lower":x[layout["slope_count"]:layout["slope_count"]+layout["pairs"]].reshape(-1,3).copy(),
        "upper":x[layout["slope_count"]+layout["pairs"]:].reshape(-1,3).copy()}


def fit_saved(mapping,legs,responses,targets,baseline,coefficients32,meter,*,solver_factory=clarabel.DefaultSolver,
              wall_clock=time.perf_counter,solve=fit.solve_decoder):
    """No full teacher/development object crosses the optimizer boundary."""
    result={"schema":SCHEMA,"run_valid":False,"selection_pass":False,"checks":{},"candidate_sha256":None,
        "body_promoted":False,"candidate_adopted":False,"F6_exposed":False}; primary=None
    try:
        meter.deadline.check()
        training={k:responses["teacher"][k][1:1201].copy() for k in ("bank","original_activation")}
        start={k:responses["startup"][k][1:100].copy() for k in ("bank","original_activation")}
        optimizer_targets={"train_q":targets["teacher_q"][1:1201].copy(),"startup_q":targets["startup_q"].copy(),"startup_ltm":targets["startup_ltm"].copy()}
        base.require(len(training["bank"])==1200 and len(start["bank"])==99,"Exact optimizer prefix copies")
        problem=fit.decoder_problem(mapping,legs,training,start,optimizer_targets)
        persist(meter.output,"decoder-problem",fit.problem_arrays(problem),{"variables":problem["layout"]["variables"],"training_only":True})
        # Include reconstruction/sparse preparation workspace as well as the
        # persistent matrices. The original waveforms remain counted separately.
        meter.host_bytes+=2*measured_bytes((problem,training,start,optimizer_targets)); meter.peak_estimate=max(meter.peak_estimate,meter.host_bytes); meter.enforce_memory()
        fixed_ltm=problem["startup_ltm_error"]
        if np.max(abs(fixed_ltm))>.025 or np.sqrt(base.mean_square(fixed_ltm))>.009:
            result.update(run_valid=True,status="fixed_ltm_rejection",checks={"fixed_ltm":False})
        else:
            meter.deadline.check(); limit=min(20.,meter.deadline.remaining())
            if limit<=0: raise base.BudgetExceeded("original")
            base.require(meter.solver_cost["constructor"]["requested"]==0 and meter.solver_cost["solve"]["requested"]==0,
                         "One decoder fit per run;no second constructor/solve")
            try:
                solved=solve(problem,limit,solver_factory=solver_factory,wall_clock=wall_clock); meter.account_solver(solved["receipt"])
            except Exception as error:
                receipt=getattr(error,"decoder_solve_receipt",None)
                if receipt is not None: meter.account_solver(receipt)
                raw=getattr(error,"partial_decoder_solution",None)
                if raw is not None:
                    base.preserve_primary(error,"Partial decoder raw solution persistence failed",lambda:persist(meter.output,"decoder-solve",
                        {k:np.asarray(getattr(raw,a),np.float64) for k,a in (("x","x"),("solver_s","s"),("solver_y","z"))},receipt))
                raise
            persist(meter.output,"decoder-solve",{k:solved[k] for k in ("x","solver_s","solver_y","primal_slack") if solved[k] is not None},solved["receipt"])
            if not solved["valid"]: result.update(run_valid=True,status="bounded_uncertified_decoder",checks={"solver":False})
            else:
                arrays=candidate_arrays(mapping,problem["layout"],solved["x"],coefficients32)
                physical=fit.physical_slopes(mapping,arrays["low"],arrays["high"])
                if not physical["pass"]:
                    persist(meter.output,"rejected-decoder",arrays,{"physical":physical})
                    result.update(run_valid=True,status="strict_physical_slope_rejection",checks={"solver":True,"slope":False})
                else:
                    # Candidate parameters and SHA are fixed before development.
                    np.savez(meter.output/"candidate.npz",**arrays); digest=base.sha256(meter.output/"candidate.npz")
                    base.write_json(meter.output/"candidate.json",{"candidate_sha256":digest,"decoder_identity":fit.decoder_identity(arrays)})
                    check_output(meter.output)
                    result["candidate_sha256"]=digest; meter.deadline.check()
                    predicted={name:fit.decode_candidate(mapping,arrays["low"],arrays["high"],response["bank"],response["original_activation"]) for name,response in responses.items()}
                    persist(meter.output,"candidate-startup",{k:predicted["startup"][k] for k in ("q","ltm")},{"candidate_sha256":digest})
                    persist(meter.output,"candidate-teacher",{k:predicted["teacher"][k] for k in ("q","ltm")},{"candidate_sha256":digest})
                    base.require(meter.development["requested"]==0,"One held-out development assessment;no reselection")
                    assessment=fit.metering.metered({"development":meter.development},"development",lambda:fit.assess_candidate(mapping,legs,
                        arrays["low"],arrays["high"],predicted["startup"],predicted["teacher"],targets,baseline),wall_clock)
                    result.update(run_valid=True,status="selected" if assessment["selection_pass"] else "valid_reject",
                        selection_pass=assessment["selection_pass"],checks={"solver":True,**assessment["checks"]},metrics=assessment)
        meter.deadline.check(); meter.selection=result
        persist(meter.output,"selection-result",None,result); meter.save(); return result
    except Exception as error:
        primary=error; result.update(run_valid=False,selection_pass=False,status="incomplete" if isinstance(error,base.BudgetExceeded) else "invalid")
        error.decoder_fit_result=result; meter.selection=result
        base.preserve_primary(error,"Partial decoder selection persistence failed",lambda:persist(meter.output,"selection-result",None,result))
        base.preserve_primary(error,"Partial decoder fit cost persistence failed",meter.save); raise
    finally:
        if primary is not None: primary.decoder_fit_cost=meter.receipt()


def execute_original(freeze,original,output,started,*,factory=None,solver_factory=clarabel.DefaultSolver):
    meter=DecoderMeter(output,ScienceDeadline(started)); primary=None
    try:
        base.require(shutil.disk_usage(output).free>=CONTRACT["free_disk_bytes"],"Two GiB free disk required")
        mappings,legs,targets,c32,c64,references,baselines=load_inputs(freeze,original)
        meter.host_bytes=base.array_bytes((mappings,targets,references,baselines)); meter.peak_estimate=meter.host_bytes; meter.enforce_memory()
        if factory is None:
            from tools.banc_m3_source_coefficient_engine import (
                CudaSourceCoefficientEngine,
            )
            factory=lambda mapping:CudaSourceCoefficientEngine(mapping,forward_implementation="precomputed")
        responses={}; reproduction={}
        for name in ("startup","teacher"):
            response=acquire_context(name,mappings[name],c64,factory,meter); responses[name]=response
            for key in ("q","ltm"): reproduction[name+":"+key]=base.numerical_gate(response[key],references[name][key],1e-10)
            reproduction[name+":zero_bank"]=bool(np.all(response["bank"][0]==0))
            reproduction[name+":zero_original_activation"]=bool(np.all(response["original_activation"][0]==0))
            old=fit.decode_candidate(mappings[name],mappings[name]["low"],mappings[name]["high"],response["bank"],response["original_activation"])
            for key in ("q","ltm"): reproduction[name+":old_decoder:"+key]=base.numerical_gate(old[key],response[key],1e-10)
            reproduction[name+":zero_activation"]=bool(np.all(old["activation"][0]==0))
            persist(output,"reproduction",None,reproduction)
            base.require(all(v if isinstance(v,bool) else v["pass"] for v in reproduction.values()),"E277 source/old decoder/zero state reproduction failed;solver0")
        persist(output,"old-source-decoder-metrics",None,{"train_mse":base.mean_square(responses["teacher"]["q"][1:1201]-targets["teacher_q"][1:1201]),
            "baseline_train_mse":base.mean_square(baselines["teacher"][1:1201]-targets["teacher_q"][1:1201]),"past_rejection_preserved":True})
        meter.host_bytes+=base.array_bytes(responses)
        return fit_saved(mappings["startup"],legs,responses,targets,baselines["teacher"],c32,meter,solver_factory=solver_factory)
    except Exception as error:
        primary=error; base.preserve_primary(error,"E287 original failure raw/cost failed",lambda error=error:base.failure_record(error,None,output,"e287-original",meter.receipt())); raise
    finally:
        if primary is None: meter.cleanup(); meter.save()
        else:
            base.preserve_primary(primary,"E287 final cleanup failed",meter.cleanup); base.preserve_primary(primary,"E287 final cost failed",meter.save)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("freeze","freeze-sha256","source-review","source-review-sha256","output"): parser.add_argument("--"+key,required=True)
    args=parser.parse_args(argv); output=Path(args.output).resolve(); base.require(not output.exists(),"Fresh E287 output directory required")
    started=time.perf_counter(); output.mkdir(parents=True)
    base.write_json(output/"cost.json",{"phase":"freeze_validation","original_calls":0,"solver_calls":0})
    try:
        freeze,original=validate_freeze(Path(args.freeze),args.freeze_sha256,Path(args.source_review),args.source_review_sha256)
        result=execute_original(freeze,original,output,started); base.write_json(output/"result.json",result); return 0
    except Exception as error:
        result={"schema":SCHEMA,"run_valid":False,"selection_pass":False,"checks":{},"candidate_sha256":None,
            "status":"incomplete" if isinstance(error,base.BudgetExceeded) else "invalid","error":str(error),"exception":type(error).__name__,
            "notes":getattr(error,"__notes__",[]),"partial_result":getattr(error,"decoder_fit_result",None),"body_promoted":False,"F6_exposed":False}
        base.preserve_primary(error,"Terminal E287 result persistence failed",lambda:base.write_json(output/"result.json",result)); return 1


if __name__=="__main__": raise SystemExit(main())
