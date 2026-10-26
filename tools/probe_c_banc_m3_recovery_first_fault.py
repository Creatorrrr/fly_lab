"""E302: observe the first frozen recovery RK fault; never repair the state."""
from __future__ import annotations

# The original exception owns the failure; publication and cleanup are secondary.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OWNED = ("tools/probe_c_banc_m3_recovery_first_fault.py", "tests/test_c_banc_recovery_first_fault.py")
FREEZE_SCHEMA = "flylab.m3.e302.recovery-first-fault.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e302.recovery-first-fault.review.v1"
RESULT_SCHEMA = "flylab.m3.e302.recovery-first-fault.result.v1"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e302-recovery-first-fault"
DESIGN = "docs/BANC_M3_E302_RECOVERY_FIRST_FAULT_DESIGN_20260928.md"
DESIGN_SHA = "e048213a0edb4b6f31928576f2c74fa0fe6bf868bc7ba1b13f1c49523a936478"
SOURCE_IDENTITY = "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
INPUTS = ("actual_csr", "baseline_csr", "native_configuration", "calibration", "candidate_start",
          "neutral_start", "neutral_final", "protected_roster", "original_start")
RECEIPTS = ("e301_freeze", "e301_review", "e301_result", "e301_ledger", "e301_root_audit",
            "e301_output_manifest", "e301_source_manifest")
LIMITS = {"supervisor_wall_s": 60, "internal_wall_s": 55, "science_wall_s": 50,
          "host_peak_bytes": 4*1024**3, "additional_gpu_peak_bytes": 2*1024**3,
          "output_bytes": 128*1024**2, "minimum_free_disk_bytes": 4*1024**3,
          "cpu_threads": 1, "models": 1, "original_attempts": 1, "retries": 0,
          "completed_ticks": 20, "partial_ticks": 1, "factor_stages": 80,
          "body_constructors": 0, "physics_steps": 0, "solver_calls": 0}
INFERENCE = "Neutral-final drive is inferred equal from byte/value-exact original whole starts and identical frozen first-control producer; candidate failed drive was not measured."


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):h.update(chunk)
    return h.hexdigest()


def safe(name, root=ROOT):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Literal workspace-relative path required")
    result = (root/name).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError("Path escapes workspace")
    return result


def same(left, right, label):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if (not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray) or left.dtype != right.dtype
                or left.shape != right.shape or left.tobytes(order="C") != right.tobytes(order="C")):
            raise ValueError("Exact array differs: "+label)
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():raise ValueError("Exact fields differ: "+label)
        for key in left:same(left[key], right[key], label+"."+key)
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):raise ValueError("Exact length differs: "+label)
        for i, (a,b) in enumerate(zip(left,right,strict=True)):same(a,b,label+str(i))
    elif type(left) is not type(right) or left != right:raise ValueError("Exact value differs: "+label)


def json_safe(value):
    if isinstance(value, dict):return {key: json_safe(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):return json_safe(value.tolist())
    if isinstance(value, np.generic):return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def validate_e301_evidence(freeze, result, ledger, audit, output_manifest, source_manifest):
    if (freeze.get("schema") != "flylab.m3.e301.recovery-resource-body.freeze.v1"
            or freeze.get("diagnostic_id") != "E301" or freeze.get("source_network_identity") != SOURCE_IDENTITY
            or result.get("diagnostic_id") != "E301" or result.get("run_valid") is not False
            or result.get("neutral_exact") is not True or audit.get("schema") != "flylab.m3.e301.root-original-raw-audit.v1"
            or audit.get("pass") is not True or audit.get("run_valid") is not False):
        raise ValueError("Exact failed E301 and passing failure-evidence audit required")
    for receipt in (result,audit):
        if any(receipt.get(key) is not False for key in ("candidate_adopted","F1_F8_complete","F6_exposed")):
            raise ValueError("E301 failure must remain unpromoted")
        if receipt.get("cost",{}).get("scientific_neural_ticks") != 40 or receipt["cost"].get("scientific_physics_steps") != 50:
            raise ValueError("Original E301 cost differs")
        if receipt.get("first_error",{}).get("message") != "Sticky recovery resource fault: 29":
            raise ValueError("Original E301 sticky29 required")
    arm = audit["arms"]["candidate"]
    if (result["arms"]["candidate"].get("completed_controls") != 0 or arm.get("controls") != 0
            or arm.get("native_samples") != 0 or arm.get("actual_neural_ticks") != 20
            or arm.get("whole_start_exact") is not True or arm.get("whole_final_available") is not False
            or audit["arms"]["neutral"].get("whole_scientific_endpoint_exact") is not True):
        raise ValueError("Failed-first-control E301 boundary differs")
    faults = arm.get("resource_faults",[])
    if len(faults)!=1 or faults[0].get("flags")!=29 or faults[0].get("actual_control_tick")!=0:
        raise ValueError("Actual E301 fault provenance differs")
    clock = ledger["arms"]["candidate"]["clock_events"]
    if (len(clock)!=1 or clock[0].get("neural_tick_before")!=0 or clock[0].get("neural_tick_after")!=20
            or clock[0].get("physics_time_after_s")!=0 or clock[0].get("control_tick_after")!=0):
        raise ValueError("E301 original advance ledger differs")
    if source_manifest.get("diagnostic_id") != "E301" or not isinstance(output_manifest.get("files"),dict):
        raise ValueError("E301 actual manifests required")


def gate(freeze_path, freeze_sha, review_path, review_sha, output, *, root=ROOT, version_reader=None):
    if sha256(freeze_path)!=freeze_sha or sha256(review_path)!=review_sha:raise ValueError("E302 freeze/review SHA differs")
    freeze,review = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (freeze_path,review_path))
    if (freeze.get("schema")!=FREEZE_SCHEMA or freeze.get("diagnostic_id")!="E302" or freeze.get("owned_files")!=list(OWNED)
            or set(freeze.get("inputs",{}))!=set(INPUTS) or freeze.get("limits")!=LIMITS or freeze.get("output")!=OUTPUT_PATH
            or freeze.get("design")!=DESIGN or freeze.get("design_sha256")!=DESIGN_SHA
            or freeze.get("source_network_identity")!=SOURCE_IDENTITY or freeze.get("drive_inference")!=INFERENCE
            or freeze.get("candidate_adopted") is not False):raise ValueError("Exact E302 prospective contract required")
    pins=freeze["source_sha256"]
    if (review.get("schema")!=REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects")!=[]
            or review.get("freeze_sha256")!=freeze_sha or review.get("source_sha256")!={key:pins[key] for key in OWNED}):
        raise ValueError("Approved exact E302 source review required")
    all_pins={}
    for group in (pins,freeze["input_sha256"],freeze["preservation_sha256"],{DESIGN:DESIGN_SHA}):
        for name,digest in group.items():
            if name in all_pins and all_pins[name]!=digest:raise ValueError("Conflicting pin")
            if not isinstance(digest,str) or len(digest)!=64 or sha256(safe(name,root))!=digest:raise ValueError("Pinned byte differs: "+name)
            all_pins[name]=digest
    def receipt(alias):
        spec=freeze[alias]
        if set(spec)!={"path","sha256"} or all_pins.get(spec["path"])!=spec["sha256"]:raise ValueError("Receipt not independently pinned: "+alias)
        return json.loads(safe(spec["path"],root).read_text(encoding="utf-8"))
    evidence={name:receipt(name) for name in RECEIPTS}
    old=evidence["e301_freeze"]
    validate_e301_evidence(old,evidence["e301_result"],evidence["e301_ledger"],evidence["e301_root_audit"],
                           evidence["e301_output_manifest"],evidence["e301_source_manifest"])
    old_review=evidence["e301_review"]
    if (old_review.get("schema")!="flylab.m3.e301.recovery-resource-body.review.v1" or old_review.get("approved") is not True
            or old_review.get("blocking_defects")!=[] or old_review.get("freeze_sha256")!=freeze["e301_freeze"]["sha256"]):
        raise ValueError("Exact E301 source approval required")
    for group in (old["reused_source_sha256"],old_review["source_sha256"],evidence["e301_source_manifest"]["source_sha256"]):
        if any(pins.get(name)!=digest for name,digest in group.items()):raise ValueError("Frozen E301 source chain incomplete")
    for group in (old["input_sha256"],old["preservation_sha256"]):
        if any(all_pins.get(name)!=digest for name,digest in group.items()):raise ValueError("Frozen E301 input/preservation chain incomplete")
    from tools import verify_c_banc_m3_recovery_resource_body as e301
    if not {*OWNED,*e301.SOURCE_FILES} <= set(pins):raise ValueError("All execution sources required")
    if freeze.get("candidate_network_identity")!=old["candidate_network_identity"]:raise ValueError("Candidate identity differs")
    if freeze["inputs"]["original_start"]!=old["paths"]["neutral_checkpoint"] or freeze["inputs"]["protected_roster"]!=old["paths"]["protected_selection"]:
        raise ValueError("Literal E150 initial/E296 roster paths required")
    for key,suffix in (("candidate_start","candidate/observed-start"),("neutral_start","neutral/observed-start"),("neutral_final","neutral/final")):
        if freeze["inputs"][key]!="verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body/"+suffix:
            raise ValueError("Literal E301 checkpoint required: "+key)
    for name in freeze["inputs"].values():
        path=safe(name,root);files=sorted(path.rglob("*")) if path.is_dir() else [path]
        if not files or any(file.relative_to(root).as_posix() not in freeze["input_sha256"] for file in files if file.is_file()):
            raise ValueError("Every loaded input/checkpoint byte required")
    # Result is deliberately a separate receipt, not inferred from output manifest.
    manifest_path=safe(freeze["e301_output_manifest"]["path"],root).parent
    for name,spec in evidence["e301_output_manifest"]["files"].items():
        path=safe(manifest_path.relative_to(root).as_posix()+"/"+name,root)
        if sha256(path)!=spec["sha256"] or path.stat().st_size!=spec["bytes"]:raise ValueError("E301 raw manifest differs")
        if all_pins.get(path.relative_to(root).as_posix())!=spec["sha256"]:raise ValueError("E301 raw byte not pinned")
    versions=version_reader or (lambda name:platform.python_version() if name=="python" else importlib.metadata.version(name))
    if not {"python","numpy","scipy","torch","cupy-cuda13x","psutil","threadpoolctl"}<=freeze["packages"].keys():raise ValueError("Actual runtime package pins required")
    if any(versions(name)!=version for name,version in freeze["packages"].items()):raise ValueError("Actual package differs")
    if output.resolve()!=safe(OUTPUT_PATH,root) or output.exists() or not output.parent.is_dir():raise ValueError("Fresh literal E302 output required")
    if shutil.disk_usage(output.parent).free<LIMITS["minimum_free_disk_bytes"]:raise OSError("E302 disk reserve unavailable")
    return freeze


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}


def load_inputs(freeze, *, checkpoint_loader=None, npz_loader=load_npz, root=ROOT):
    from tools import banc_m3_recovery_resource_synapses as engine
    from tools import banc_m3_recovery_resource_trial as wrapper
    if checkpoint_loader is None:
        from flylab.c.storage import StateStore
        checkpoint_loader=StateStore.load
    paths={key:safe(name,root) for key,name in freeze["inputs"].items()}
    original=checkpoint_loader(paths["original_start"])
    candidate=checkpoint_loader(paths["candidate_start"])
    neutral=checkpoint_loader(paths["neutral_start"])
    final=checkpoint_loader(paths["neutral_final"])
    for name,state in (("candidate",candidate),("neutral",neutral)):
        same(wrapper.remove_resource_layer(state),original,name+" layer-removed whole initial")
    wrapper.validate_neutral(original["network"])
    net=original["network"]
    if len(net["rate"])!=158706 or net["model_hash"]!=SOURCE_IDENTITY or net["dt"]!=.00025 or net["device"]!="cuda":raise ValueError("Exact original full native checkpoint required")
    if final["network"]["tick"]!=20 or final["control_tick"]!=1:raise ValueError("One completed original neutral control required")
    drive=final["network"]["drive"]
    if drive.shape!=(158706,) or drive.dtype!=np.float32 or not np.isfinite(drive).all():raise ValueError("Full finite native first held drive required")
    csr,baseline,configuration,calibration,roster=(npz_loader(paths[key]) for key in
        ("actual_csr","baseline_csr","native_configuration","calibration","protected_roster"))
    for key in ("indptr","indices","data"):same(csr[key],baseline[key],"Original E150/E257 CSR "+key)
    from scipy.sparse import csr_matrix
    weights=csr_matrix((csr["data"],csr["indices"],csr["indptr"]),shape=(158706,158706),copy=True)
    if weights.data.dtype!=np.float32 or not weights.has_canonical_format:raise ValueError("Canonical original f32 CSR required")
    for key in ("indptr","indices","data"):same(getattr(weights,key),csr[key],"CSR construction storage "+key)
    tau=configuration["membrane_tau"]
    if tau.shape!=(158706,) or tau.dtype!=np.float32 or not np.isfinite(tau).all() or np.any(tau<=0):raise ValueError("Original membrane tau required")
    from tools.banc_m3_intrinsic_recruitment import patch_tau
    same(tau,patch_tau(np.full(158706,.02,np.float32),calibration["tau_ids"],calibration["tau_values"]),
         "Original E150 calibration membrane tau")
    gain,timescale=net["motor_adaptation"],net["motor_timescale"]
    for name,actual in (("motor_ids",gain["ids"]),("motor_gains",gain["gains"]),("motor_log_tau",timescale["log_tau"])):
        same(configuration[name],actual,"Original native configuration "+name)
    for name,actual in (("actual_motor_gain32",configuration["motor_gains"]),
            ("actual_motor_tau32",timescale["tau32"][gain["ids"]]),("actual_motor_inverse32",timescale["inverse32"][gain["ids"]])):
        same(configuration[name],actual,"Native actual physiology "+name)
    selection=engine.checked_selection(weights,roster["edge_ids"],roster["premotor_ids"],roster["utilization"],roster["recovery_tau"],gain["ids"])
    for key,value in selection.items():same(value,roster[key],"Protected original selection "+key)
    metadata=engine.resource_metadata(SOURCE_IDENTITY,selection)
    if engine.resource_identity(metadata)!=freeze["candidate_network_identity"] or candidate["network"]["model_hash"]!=freeze["candidate_network_identity"]:raise ValueError("Actual candidate identity differs")
    same(candidate["network"]["synaptic_resource"],np.ones(len(selection["source_ids"]),np.float64),"Initial resource one")
    return {"original":original,"network":net,"weights":weights,"configuration":configuration,
            "selection":selection,"drive":drive.copy(),"freeze":freeze,"drive_inference":INFERENCE}


class DiagnosticStop(RuntimeError):
    def __init__(self, row):
        super().__init__("E302 first actual recovery fault before continuation")
        self.row=copy.deepcopy(row)
        self.resource_fault_flags=row["actual_flags"]


def host_array(value):
    return value.detach().cpu().numpy().copy()


def charge_wall(record,key,started,clock):
    primary=sys.exception()
    try:record[key]+=clock()-started
    except Exception as error:
        record.setdefault("wall_errors",[]).append(repr(error))
        if primary is None:raise
        primary.add_note("Secondary wall-clock read: "+repr(error))


def independent_factor(rate,resource,sources,u,tau):
    """Literal f64 formula and per-source original CUDA guards, including bad raw."""
    r=rate[sources];active=u>0;flags=np.zeros(len(sources),np.int32)
    flags[(~np.isfinite(resource))|(resource<0)|(resource>1)]|=1
    flags[(u==0)&(resource!=1)]|=2
    flags[active&((~np.isfinite(r))|(r<0))]|=4
    with np.errstate(over="ignore",invalid="ignore",under="ignore"):
        a=np.ones(len(sources),np.float64)
        a[active]=resource[active]*(1.+(tau[active]*u[active])*r[active].astype(np.float64))
        flags[(~np.isfinite(a))|(a<0)]|=8
        g=np.where(a<1.,a,1.).astype(np.float32)
    flags[(~np.isfinite(g))|(g<0)|(g>1)]|=8
    g[flags!=0]=np.array(0x7fffffff,np.uint32).view(np.float32)
    return g,int(np.bitwise_or.reduce(flags,initial=np.int32(0)))


class StageObserver:
    def __init__(self,network,*,host_copy=host_array,persist=lambda:None,clock=time.perf_counter,deadline=None,resource_check=lambda phase:None):
        self.network=network;self.host=host_copy;self.persist=persist;self.clock=clock;self.deadline=deadline
        self.resource_check=resource_check
        self.original_launch=network._recovery_launch;self.had_override="_recovery_launch" in vars(network)
        self.graph=network.graph;self.rows=[];self.ring=deque(maxlen=2);self.bad=None;self.pending=None
        self.first=None;self.stage=0;self.attempted_tick=0;self.completed_tick=network.tick
        self.publication_errors=[];self.publication_failures=[];self.installed=False
        self.cost={"factor_requested":0,"factor_completed":0,"factor_failed":0,"final_requested":0,
                   "final_completed":0,"final_failed":0,"host_copy_requested":0,"host_copy_completed":0,
                   "host_copy_failed":0,"observer_wall_s":0.}

    def _copy(self,value):
        self.cost["host_copy_requested"]+=1
        try:
            result=self.host(value);self.cost["host_copy_completed"]+=1;return result
        except Exception:self.cost["host_copy_failed"]+=1;raise

    def install(self):
        if self.installed:raise RuntimeError("Observer already installed")
        self.network.graph=None;self.network._recovery_launch=self.launch;self.installed=True

    def restore(self):
        if self.installed:
            if self.had_override:self.network._recovery_launch=self.original_launch
            else:del self.network._recovery_launch
            self.network.graph=self.graph;self.installed=False

    def release(self):
        self.restore();self.network=None;self.original_launch=None;self.graph=None

    def _raw(self,rate,resource,stage):
        net=self.network;selection=net.selection
        adaptation=net.adaptation if stage==0 else net._fused_buffers[2]
        return {"rate":self._copy(rate),"adaptation":self._copy(adaptation),"drive":self._copy(net.drive),
                "output_mask":self._copy(net.output_mask),"synaptic_resource":self._copy(resource),
                **{key:selection[key].copy() for key in ("source_ids","utilization","recovery_tau")}}

    def _row(self,raw,actual_flags,expected_flags,phase,stage):
        active=raw["utilization"]>0;r=raw["rate"][raw["source_ids"]];R=raw["synaptic_resource"]
        bad=active&((~np.isfinite(r))|(r<0))
        return json_safe({"attempted_tick":self.attempted_tick,"completed_tick":self.completed_tick,
            "stage_index":stage,"phase":phase,"valid":actual_flags==0,"actual_flags":actual_flags,"expected_flags":expected_flags,
            "source_count":len(r),"active_count":int(active.sum()),"active_rate_min":float(r[active].min()) if active.any() else None,
            "negative_source_ids":raw["source_ids"][active&(r<0)],"negative_source_values":r[active&(r<0)],
            "invalid_rate_source_ids":raw["source_ids"][bad],"invalid_resource_source_ids":raw["source_ids"][(~np.isfinite(R))|(R<0)|(R>1)],
            "resource_min":float(R.min()),"resource_max":float(R.max()),
            "factor_unity_count":int(np.count_nonzero(raw["actual_factor"]==1)),
            "factor_nonunity_count":int(np.count_nonzero(raw["actual_factor"]!=1)),
            "raw_kind":"RK internal observation; NOT a valid neural checkpoint"})

    def _stop(self,raw,row):
        self.bad=raw;self.first=row
        error=DiagnosticStop(row)
        try:self.persist()
        except Exception as secondary:
            self.publication_errors.append(repr(secondary));self.publication_failures.append(secondary)
            error.add_note("Secondary first-fault publication: "+repr(secondary))
        raise error

    def launch(self,name,tensors,step=None):
        if name not in ("recovery_factor","recovery_finish"):return self.original_launch(name,tensors,step)
        began=self.clock()
        if self.deadline is not None and began>=self.deadline:raise TimeoutError("E302 science deadline before stage")
        self.resource_check("before_"+name)
        kind="factor" if name=="recovery_factor" else "final"
        self.cost[kind+"_requested"]+=1
        try:
            if kind=="factor":
                if self.stage>=4 or self.cost["factor_requested"]>80:raise RuntimeError("Frozen RK factor stage count exceeded")
                raw=self._raw(tensors[1],tensors[2],self.stage);self.pending=raw
                expected,flags=independent_factor(raw["rate"],raw["synaptic_resource"],raw["source_ids"],raw["utilization"],raw["recovery_tau"])
                self.original_launch(name,tensors,step)
                raw["actual_factor"]=self._copy(self.network.resource_factor)
                actual=int(self._copy(self.network._resource_fault)[0]);self.cost[kind+"_completed"]+=1
                self.resource_check("after_factor")
                row=self._row(raw,actual,flags,"before_CSR_after_factor",self.stage);self.rows.append(row)
                if actual!=flags:raise ValueError("Actual sticky factor flags differ from independent stage guards")
                if flags:self._stop(raw,row)
                same(raw["actual_factor"],expected,"Actual valid stage factor32")
                self.ring.append((copy.deepcopy(row),raw));self.pending=None;self.stage+=1
            else:
                if self.stage!=4:raise RuntimeError("Resource finish before four factors")
                self.original_launch(name,tensors,step)
                raw=self._raw(tensors[1],tensors[8],3);self.pending=raw
                raw["actual_factor"]=self._copy(self.network.resource_factor)
                actual=int(self._copy(self.network._resource_fault)[0]);self.cost[kind+"_completed"]+=1
                self.resource_check("after_resource_finish")
                R,u=raw["synaptic_resource"],raw["utilization"]
                flags=(16 if np.any((~np.isfinite(R))|(R<0)|(R>1)) else 0)|(2 if np.any((u==0)&(R!=1)) else 0)
                if actual!=flags:raise ValueError("Actual resource finish flags differ")
                if flags:
                    row=self._row(raw,actual,flags,"after_resource_finish_before_neural_finish",4)
                    row["factor_scope"]="preceding stage3 factor; not recomputed at final R"
                    self.rows.append(row);self._stop(raw,row)
                self.pending=None
        except DiagnosticStop:raise
        except Exception:self.cost[kind+"_failed"]+=1;raise
        finally:charge_wall(self.cost,"observer_wall_s",began,self.clock)


def probe(network,drive,*,observer,max_ticks=20,clock=time.perf_counter):
    if type(max_ticks) is not int or not 1<=max_ticks<=20:raise ValueError("Single bounded prefix required")
    counts={"advance_requested":0,"advance_completed":0,"advance_failed":0,"completed_ticks":0,"partial_ticks":0,"advance_wall_s":0.}
    observer.install()
    try:
        for _ in range(max_ticks):
            observer.stage=0;observer.completed_tick=network.tick;observer.attempted_tick=network.tick+1
            counts["advance_requested"]+=1;began=clock()
            try:
                network.advance(drive,steps=1)
                if network.tick!=observer.attempted_tick or observer.stage!=4:raise ValueError("Frozen single tick/four stage completion differs")
                counts["advance_completed"]+=1;counts["completed_ticks"]+=1
            except DiagnosticStop as stopped:
                counts["advance_failed"]+=1;counts["partial_ticks"]+=1
                if network.tick!=observer.completed_tick:raise ValueError("Diagnostic stop unexpectedly completed native tick")
                # All raw evidence lives in the observer. The expected stop's
                # traceback must not retain the owned engine through a nested
                # publication exception or its exception context.
                traceback.clear_frames(stopped.__traceback__)
                stopped.__traceback__=None
                break
            except Exception:
                counts["advance_failed"]+=1;counts["partial_ticks"]+=1;raise
            finally:charge_wall(counts,"advance_wall_s",began,clock)
    except Exception as error:
        error.e302_probe_counts=counts
        raise
    finally:
        observer.restore()
        # A saved publication traceback can retain this completed frame through
        # f_back even when its own traceback frames were cleared.
        network=None
    return counts


class NativeDependencies:
    def __init__(self,*,engine=None,profiler=None):
        self.engine=engine;self.profiler=profiler;self.gpu_free_before=None;self.cupy_pool_high_water=0
        self.gpu_global_delta_high_water=0

    def load(self,freeze):return load_inputs(freeze)

    def construct(self,data,rows,persist,on_return):
        engine=self.engine
        if engine is None:
            from tools import banc_m3_recovery_resource_synapses as engine
        profiler=self.profiler
        if profiler is None:
            from tools.verify_c_banc_m3_native_timescale_transfer import (
                profiled_construction,
            )
            profiler=profiled_construction
        net=data["network"];s=data["selection"];tau=data["configuration"]["membrane_tau"]
        core,motor,timescale=net["core_adaptation"],net["motor_adaptation"],net["motor_timescale"]
        def callback():
            return engine.RecoveryResourceNetwork(data["weights"],tau,np.ones(len(tau),np.float32),
                np.full(len(tau),7.5,np.float32),np.full(len(tau),200.,np.float32),
                adaptation_gain=8.,adaptation_tau_s=.3,dt=.00025,device="cuda",cuda_implementation="packed",capture_steps=20,
                core_ids=core["ids"],core_gain=core["gain"],motor_ids=motor["ids"],motor_gains=motor["gains"],
                motor_log_tau=timescale["log_tau"],edge_ids=s["edge_ids"],premotor_ids=s["premotor_ids"],
                utilization=s["utilization"],recovery_tau=s["recovery_tau"])
        return profiler("e302_recovery_constructor",callback,rows,persist,on_return=on_return)

    def validate_initial(self,network,data):
        same(network.selection,data["selection"],"Actual native protected selection")
        if (network.source_timescale_identity!=SOURCE_IDENTITY or network.identity!=data["freeze"]["candidate_network_identity"]
                or network.device!="cuda" or network.cuda_implementation!="packed" or network.interval_steps!=20):raise ValueError("Actual original constructor identity/implementation differs")
        for key in ("indptr","indices","data"):same(getattr(network._source_weights,key),getattr(data["weights"],key),"Constructed full CSR "+key)
        network.import_base_snapshot(copy.deepcopy(data["network"]))
        saved=network.snapshot()
        for key in ("rate","adaptation","drive","output_mask","tick"):same(saved[key],data["network"][key],"Imported exact initial "+key)
        same(saved["synaptic_resource"],np.ones(len(data["selection"]["source_ids"]),np.float64),"Imported resource one")
        for name,expected in (("tau",data["configuration"]["membrane_tau"]),("_cell_gain",data["configuration"]["cell_gain"]),
            ("_cell_adaptation_tau",data["network"]["motor_timescale"]["tau32"]),("_cell_inverse_tau",data["network"]["motor_timescale"]["inverse32"])):
            same(host_array(getattr(network,name)),expected,"Actual native tensor "+name)
        # Decode the real packed CSR, retaining explicit-zero positions.
        packed=host_array(network._columns).view(np.uint32);lookup=host_array(network._weight_values)
        same((packed & 0x3ffff).astype(data["weights"].indices.dtype),data["weights"].indices,"Packed original column decode")
        same(lookup[packed>>18],data["weights"].data,"Packed original weight decode")

    def host_copy(self,value):return host_array(value)

    def resource_check(self,output):
        import cupy
        import psutil
        import torch
        memory=psutil.Process().memory_info()
        host=max(memory.rss,getattr(memory,"peak_wset",0))
        free,total=cupy.cuda.runtime.memGetInfo()
        if self.gpu_free_before is None:self.gpu_free_before=free
        pool=cupy.get_default_memory_pool().total_bytes()
        self.cupy_pool_high_water=max(self.cupy_pool_high_water,pool)
        self.gpu_global_delta_high_water=max(self.gpu_global_delta_high_water,self.gpu_free_before-free)
        gpu=torch.cuda.max_memory_reserved()+self.cupy_pool_high_water
        size=sum(file.stat().st_size for file in output.rglob("*") if file.is_file())
        if (host>LIMITS["host_peak_bytes"] or gpu>LIMITS["additional_gpu_peak_bytes"]
                or self.gpu_global_delta_high_water>LIMITS["additional_gpu_peak_bytes"] or size>LIMITS["output_bytes"]):
            raise RuntimeError("E302 resource capacity exceeded")
        return {"host_peak_bytes":host,"torch_allocator_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "cupy_pool_current_total_bytes":pool,"cupy_pool_sampled_high_water_bytes":self.cupy_pool_high_water,
                "gpu_allocator_peak_bound_bytes":gpu,"global_gpu_free_bytes":free,"global_gpu_total_bytes":total,
                "additional_gpu_sampled_high_water_bytes":self.gpu_global_delta_high_water,
                "global_gpu_peak_continuously_measured":False,"output_bytes":size}

    def cleanup(self):
        import cupy
        import torch
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
        cupy.get_default_memory_pool().free_all_blocks();cupy.get_default_pinned_memory_pool().free_all_blocks()


class Writer:
    def __init__(self,output):self.output=output

    def json(self,name,value):
        encoded=(json.dumps(json_safe(value),sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
        self._check(len(encoded));path=self.output/name;temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_bytes(encoded);temporary.replace(path)

    def raw(self,name,arrays):
        self._check(sum(value.nbytes for value in arrays.values())+8192)
        path=self.output/name;temporary=path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as stream:np.savez(stream,**arrays)
        temporary.replace(path)

    def _check(self,additional):
        size=sum(file.stat().st_size for file in self.output.rglob("*") if file.is_file())
        if size+additional>LIMITS["output_bytes"]:raise OSError("E302 output cap exceeded")


def execute(freeze,output,*,dependencies=None,writer=None,clock=time.perf_counter):
    deps=NativeDependencies() if dependencies is None else dependencies
    output.mkdir(exist_ok=False);writer=Writer(output) if writer is None else writer
    began=clock();network=None;observer=None;primary=None
    ledger={"construction_requested":0,"construction_completed":0,"construction_failed":0,"construction_wall_s":0.,
            "construction_profiles":[],"initial_validation_requested":0,"initial_validation_completed":0,
            "initial_validation_failed":0,"cleanup_requested":0,"cleanup_completed":0,"cleanup_failed":0,
            "body_constructors":0,"physics_steps":0,"solver_calls":0,"resource_samples":[],
            "resource_check_requested":0,"resource_check_completed":0,"resource_check_failed":0}
    result={"schema":RESULT_SCHEMA,"diagnostic_id":"E302","run_valid":False,"first_fault_observed":False,
            "candidate_adopted":False,"F1_F8_complete":False,"F2_complete":False,"F6_exposed":False,"biological_claim":False,
            "drive_inference":INFERENCE,"candidate_failed_drive_directly_measured":False,"e301_cumulative_flags":29,
            "rows":[],"raw_kind":"RK internal observations; NOT neural checkpoints"}
    def persist():writer.json("ledger.json",ledger)
    def sample(phase):
        ledger["resource_check_requested"]+=1
        try:
            row=deps.resource_check(output);ledger["resource_samples"].append({"phase":phase,**row})
            ledger["resource_check_completed"]+=1
        except Exception:ledger["resource_check_failed"]+=1;raise
    def output_manifest():
        files={file.relative_to(output).as_posix():{"sha256":sha256(file),"bytes":file.stat().st_size}
               for file in sorted(output.rglob("*")) if file.is_file() and file.name not in ("result.json","output-manifest.json")}
        writer.json("output-manifest.json",{"schema":"flylab.m3.e302.outputs.v1","files":files,
            "result_published_separately":True,"bytes":sum(spec["bytes"] for spec in files.values())})
    def publication():
        result["rows"]=observer.rows;result["first_fault"]=observer.first
        result["first_fault_observed"]=observer.first is not None
        for i,(row,raw) in enumerate(observer.ring):
            writer.raw("last-valid-"+str(i)+".npz",raw)
        if observer.bad is not None:writer.raw("first-invalid.npz",observer.bad)
        elif observer.pending is not None:writer.raw("partial-stage.npz",observer.pending)
        writer.json("stage-summary.json",{"rows":observer.rows,"last_valid_rows":[row for row,_ in observer.ring],
            "first_fault":observer.first,"raw_kind":result["raw_kind"]})
    def returned(value):
        nonlocal network
        network=value;ledger["construction_completed"]=1
    try:
        writer.json("source-manifest.json",{"schema":"flylab.m3.e302.sources.v1","diagnostic_id":"E302",
            "owned_sha256":{name:sha256(ROOT/name) for name in OWNED},"source_sha256":freeze.get("source_sha256",{}),
            "input_sha256":freeze.get("input_sha256",{}),"packages":freeze.get("packages",{}),
            "receipt_pins":{key:freeze.get(key) for key in RECEIPTS},"design":DESIGN,"design_sha256":DESIGN_SHA,
            "limits":LIMITS,"drive_inference":INFERENCE,"module_global_changes":False,
            "observer_scope":"one actual instance, factor boundary and resource finish boundary only"})
        data=deps.load(freeze)
        sample("before_constructor")
        if clock()-began>=LIMITS["science_wall_s"]:raise TimeoutError("E302 preconstruction deadline")
        persist();ledger["construction_requested"]=1;persist();started=clock()
        try:network=deps.construct(data,ledger["construction_profiles"],persist,returned)
        except Exception:
            ledger["construction_failed"]=1;raise
        finally:charge_wall(ledger,"construction_wall_s",started,clock)
        sample("after_constructor_while_owned")
        ledger["initial_validation_requested"]=1
        try:deps.validate_initial(network,data);ledger["initial_validation_completed"]=1
        except Exception:ledger["initial_validation_failed"]=1;raise
        sample("after_initial_validation_while_owned")
        observer=StageObserver(network,host_copy=deps.host_copy,persist=publication,clock=clock,
            deadline=began+LIMITS["science_wall_s"],resource_check=sample)
        counts=probe(network,data["drive"],observer=observer,clock=clock)
        ledger["science"]=counts;publication()
        if observer.publication_errors:raise OSError("First-fault publication failed: "+repr(observer.publication_errors))
        result["first_fault_observed"]=observer.first is not None
        result["all_valid_factors_unity_before_first_fault"]=all(row["factor_nonunity_count"]==0 for row in observer.rows if row["valid"])
        result["run_valid"]=True
    except Exception as error:
        primary=error
        ledger["science"]=getattr(error,"e302_probe_counts",ledger.get("science",{}))
    finally:
        if observer is not None:
            ledger["observer"]=observer.cost.copy()
            try:publication()
            except Exception as error:
                if primary is None:primary=error
                else:primary.add_note("Secondary partial raw publication: "+repr(error))
                traceback.clear_frames(error.__traceback__)
            try:observer.release()
            except Exception as error:
                if primary is None:primary=error
                else:primary.add_note("Secondary observer restoration: "+repr(error))
            for error in observer.publication_failures:traceback.clear_frames(error.__traceback__)
        if primary is not None:traceback.clear_frames(primary.__traceback__)
        network=None;observer=None
        if "data" in locals():data=None
        gc.collect();ledger["cleanup_requested"]+=1
        try:deps.cleanup();ledger["cleanup_completed"]+=1
        except Exception as error:
            ledger["cleanup_failed"]+=1
            if primary is None:primary=error
            else:primary.add_note("Secondary runtime cleanup: "+repr(error))
        try:sample("after_cleanup")
        except Exception as error:
            if primary is None:primary=error
            else:primary.add_note("Secondary resource audit: "+repr(error))
        result["run_valid"]=bool(result["run_valid"] and primary is None and clock()-began<LIMITS["internal_wall_s"])
        if primary is None and not result["run_valid"]:primary=TimeoutError("E302 final deadline")
        result["elapsed_s"]=clock()-began;result["cost"]=copy.deepcopy(ledger)
        if primary is not None:result["first_error"]={"type":type(primary).__name__,"message":str(primary),"notes":getattr(primary,"__notes__",[])}
        for callback in (persist,output_manifest,lambda:writer.json("result.json",result)):
            try:callback()
            except Exception as error:
                result["run_valid"]=False
                if primary is None:primary=error
                else:primary.add_note("Secondary terminal publication: "+repr(error))
    if primary is not None:
        primary.e302_partial_result=result;primary.e302_ledger=ledger
    return result,primary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("freeze","source-review","output"):parser.add_argument("--"+key,required=True,type=Path)
    for key in ("freeze-sha256","source-review-sha256"):parser.add_argument("--"+key,required=True)
    args=parser.parse_args(argv)
    freeze=gate(args.freeze,args.freeze_sha256,args.source_review,args.source_review_sha256,args.output)
    import torch
    from threadpoolctl import threadpool_limits
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            result,error=execute(freeze,args.output)
        return 0 if error is None and result["run_valid"] else 1
    finally:torch.set_num_threads(previous)


if __name__=="__main__":raise SystemExit(main())
