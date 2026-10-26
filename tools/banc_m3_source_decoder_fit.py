"""E287 one conditional monotone decoder fit, with immutable source dynamics."""
from __future__ import annotations

# Preserve primary faults through timing/storage failures.
import copy
import hashlib
import time

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from tools import banc_m3_conic_certificate as certificate
from tools import banc_m3_feasibility_restoration as metering
from tools import fit_c_banc_m3_source_coefficients as base

Q_PEAK = .04-.0015168425596647772-.0001
Q_RMS = .018-.00013667128182327153-.0001
REGULARIZATION = 1e-8
FILTER_TAU = np.array([.005, .010, .040], np.float64)
DECODER_KEYS = ("motor_ids", "membership", "pool", "low", "high", "kappa", "fixed", "moment", "rest", "ltm",
                "bank0", "activation0", "original_activation0")
PROBLEM_KEYS={"P","q","A","b","cones","cone_dimensions","layout","train_design","train_error_offset",
              "startup_design","startup_error_offset","startup_ltm_error","objective_constant"}
CONTRACT = {"source_coefficients": "E277 coefficients32 exactly expanded to float64;fixed",
    "parameter_order": "free motor ascending local index/filter0,1,2/interleaved low,high;all lower;all upper",
    "tau_s": FILTER_TAU.tolist(), "regularization": REGULARIZATION, "muscle_lower": .500001, "muscle_upper": 1.999999,
    "motor_floor_fraction": 1e-6, "slope_upper": 2., "q_peak": Q_PEAK, "q_rms": Q_RMS,
    "ltm_peak": .025, "ltm_rms": .009, "train_samples": 1200, "startup_samples": 99,
    "train_ratio": .8, "development_ratio": 1., "leg_ratio": 1.05, "solver_version": "0.11.1",
    "max_solver_calls": 1, "solver_wall_s": 20., "max_iter": 100, "max_threads": 1, "solver_tolerance": 1e-8,
    "fixed_transfer": "canonical LTM and singleton slopes;exact original_activation;no added muscle EMA",
    "optimality_claim": False, "retry_calls": 0, "repair_calls": 0, "F6_exposed": False}


def decoder_layout(mapping, legs):
    """Small generated rosters are allowed here; runner enforces full original axes."""
    base.require(all(k in mapping for k in DECODER_KEYS), "Complete original decoder mapping required")
    d = {k: np.array(mapping[k], copy=True) for k in DECODER_KEYS}
    n, m, q = len(d["motor_ids"]), len(d["fixed"]), len(d["rest"])
    base.require(1 <= n <= 391 and 1 <= m <= 96 and 1 <= q <= 42, "Bounded explicit decoder axes")
    shapes = {"membership": (n,), "pool": (m,n), "low": (n,3), "high": (n,3), "kappa": (n,), "fixed": (m,),
        "moment": (q,m), "rest": (q,), "bank0": (n,3), "activation0": (m,), "original_activation0": (m,)}
    for k, shape in shapes.items():
        v=d[k]; dtype=np.int64 if k=="membership" else np.bool_ if k=="fixed" else np.float64
        base.require(v.shape==shape and v.dtype==dtype and np.isfinite(v).all(), "Exact finite decoder array: "+k)
    base.require(d["motor_ids"].dtype==np.int64 and np.all(d["motor_ids"]>=0) and len(np.unique(d["motor_ids"]))==n
        and d["ltm"].dtype==np.int64 and d["ltm"].ndim==1 and len(np.unique(d["ltm"]))==len(d["ltm"])
        and np.all((d["ltm"]>=0)&(d["ltm"]<m)), "Explicit unique native MN/LTM axes")
    member=d["membership"]
    base.require(np.all((member>=0)&(member<m)) and np.all(d["pool"]>=0)
        and np.all(np.count_nonzero(d["pool"],axis=0)==1) and np.array_equal(np.argmax(d["pool"],axis=0),member)
        and np.allclose(d["pool"].sum(1),1,atol=1e-15,rtol=0), "Original anatomical pool partition required")
    counts=np.bincount(member,minlength=m); fixed=counts==1; fixed[d["ltm"]]=True
    base.require(np.array_equal(d["fixed"],fixed) and np.all(d["kappa"]>=0) and np.all(d["low"]>=0)
        and np.all(d["high"]>=0), "Exact LTM/singleton fixed rows and monotone slopes required")
    canonical=np.zeros((n,3)); canonical[:,1]=d["pool"][member,np.arange(n)]
    mask=fixed[member]
    base.require(np.array_equal(d["low"][mask],canonical[mask]) and np.array_equal(d["high"][mask],canonical[mask]),
                 "Fixed rows must preserve canonical original transfer")
    base.require(all(np.all(d[k]==0) for k in ("bank0","activation0","original_activation0")), "Exact zero initial bank/activation required")
    ownership={k:np.array(legs[k],copy=True) for k in ("motor","muscle","q")}
    ownership["ltm"]=np.array(legs.get("ltm",ownership["muscle"][d["ltm"]]),copy=True)
    for k,size in (("motor",n),("muscle",m),("q",q),("ltm",len(d["ltm"]))):
        v=ownership[k]; base.require(v.dtype==np.int64 and v.shape==(size,) and np.all((v>=0)&(v<6)), "Explicit decoder leg membership: "+k)
    base.require(np.array_equal(ownership["motor"],ownership["muscle"][member])
        and np.array_equal(ownership["ltm"],ownership["muscle"][d["ltm"]])
        and np.all(d["moment"][ownership["q"][:,None]!=ownership["muscle"][None,:]]==0), "No cross-leg decoder connection")
    free=np.flatnonzero(~mask).astype(np.int64); pairs=3*len(free)
    base.require(len(free)>0, "At least one original nonfixed motor required")
    slopes=np.stack((d["low"][free],d["high"][free]),axis=-1)
    x0=np.r_[slopes.ravel(),np.min(slopes,axis=-1).ravel(),np.max(slopes,axis=-1).ravel()]
    layout={"mapping":d,"legs":ownership,"free_motor":free,"variable_muscle":np.flatnonzero(~fixed).astype(np.int64),
        "fixed_muscle":np.flatnonzero(fixed).astype(np.int64), "pairs":pairs,"slope_count":2*pairs,"variables":4*pairs,"x0":x0}
    base.require(physical_slopes(d,d["low"],d["high"])["pass"], "Original slope budgets/floors invalid")
    return layout


def physical_slopes(mapping, low, high):
    member=mapping["membership"]; m=len(mapping["fixed"]); counts=np.bincount(member,minlength=m)
    finite=low.shape==high.shape==mapping["low"].shape and low.dtype==high.dtype==np.float64 and np.isfinite(low).all() and np.isfinite(high).all()
    if not finite: return {"pass":False,"checks":{"finite":False}}
    minimum=np.minimum(low,high).sum(1); maximum=np.maximum(low,high).sum(1)
    lower=np.bincount(member,weights=minimum,minlength=m); upper=np.bincount(member,weights=maximum,minlength=m)
    checks={"finite":True,"nonnegative":bool(np.all(low>=0)&np.all(high>=0)),"slope_upper":bool(np.all(low<=2)&np.all(high<=2)),
        "muscle_lower":bool(np.all(lower>=.5)),"muscle_upper":bool(np.all(upper<=2)),
        "motor_floor":bool(np.all(minimum>=1e-6/counts[member])),
        "fixed":bool(np.array_equal(low[mapping["fixed"][member]],mapping["low"][mapping["fixed"][member]])
            and np.array_equal(high[mapping["fixed"][member]],mapping["high"][mapping["fixed"][member]]))}
    return {"pass":bool(all(checks.values())),"checks":checks,"lower":lower,"upper":upper,"motor_minimum":minimum}


def _response_inputs(mapping, bank, original):
    n,m=len(mapping["motor_ids"]),len(mapping["fixed"])
    base.require(isinstance(bank,np.ndarray) and bank.dtype==np.float64 and bank.ndim==3 and bank.shape[1:]==(n,3)
        and 1<=len(bank)<=2001 and np.isfinite(bank).all() and np.all(bank>=0)
        and isinstance(original,np.ndarray) and original.dtype==np.float64 and original.shape==(len(bank),m)
        and np.isfinite(original).all() and np.all(original>=0), "Finite saved bank/original activation required")


def affine_q(mapping, layout, bank, original_activation):
    """C-order q affine map in the free interleaved slope columns only."""
    _response_inputs(mapping,bank,original_activation)
    free=layout["free_motor"]; q=len(mapping["rest"]); member=mapping["membership"]; count=len(bank)
    hinge=np.maximum(bank[:,free]-mapping["kappa"][free,None],0)
    basis=np.stack((np.minimum(bank[:,free],mapping["kappa"][free,None]),hinge),axis=-1).reshape(count,len(free),6)
    pieces=[]; row_indices=[]; col_indices=[]
    for leg in np.unique(layout["legs"]["q"]):
        joints=np.flatnonzero(layout["legs"]["q"]==leg); columns=np.flatnonzero(layout["legs"]["motor"][free]==leg)
        if not len(columns): continue
        gain=.02*mapping["moment"][np.ix_(joints,member[free[columns]])]
        values=np.einsum("sif,ji->sjif",basis[:,columns],gain).reshape(count*len(joints),len(columns)*6)
        local=sparse.coo_matrix(values)
        pieces.append(local.data); row_indices.append((local.row//len(joints))*q+joints[local.row%len(joints)])
        col_indices.append((columns[:,None]*6+np.arange(6)).ravel()[local.col])
    design=sparse.csc_matrix((np.concatenate(pieces), (np.concatenate(row_indices),np.concatenate(col_indices))),shape=(count*q,layout["slope_count"]))
    fixed=layout["fixed_muscle"]
    offset=mapping["rest"]+.02*(original_activation[:,fixed]@mapping["moment"][:,fixed].T)
    return design,offset


def candidate_slopes(layout, vector):
    x=np.asarray(vector); base.require(x.dtype==np.float64 and x.shape==(layout["variables"],) and np.isfinite(x).all(), "Complete finite original decoder vector required")
    low,high=(layout["mapping"][k].copy() for k in ("low","high")); value=x[:layout["slope_count"]].reshape(-1,3,2)
    low[layout["free_motor"]],high[layout["free_motor"]]=value[:,:,0],value[:,:,1]
    return low,high


def decode_candidate(mapping, low, high, bank, original_activation):
    _response_inputs(mapping,bank,original_activation)
    base.require(low.shape==high.shape==mapping["low"].shape and low.dtype==high.dtype==np.float64
        and np.isfinite(low).all() and np.isfinite(high).all(), "Finite explicit decoder slopes required")
    value=low[None,:,:]*bank+(high-low)[None,:,:]*np.maximum(bank-mapping["kappa"][None,:,None],0)
    activation=np.zeros_like(original_activation)
    for muscle in range(len(mapping["fixed"])):
        if mapping["fixed"][muscle]: activation[:,muscle]=original_activation[:,muscle]
        else: activation[:,muscle]=value[:,mapping["membership"]==muscle,:].sum(axis=(1,2))
    return {"activation":activation,"q":mapping["rest"]+.02*(activation@mapping["moment"].T),
            "ltm":.2*np.tanh(activation[:,mapping["ltm"]]/10)}


def _equations(layout, train_design, train_offset, startup_design, startup_offset):
    n=layout["variables"]; slopes=layout["slope_count"]; pairs=layout["pairs"]; free=layout["free_motor"]
    P=sparse.eye(n,format="csc",dtype=np.float64)*(2*REGULARIZATION); linear=np.zeros(n)
    member=layout["mapping"]["membership"]; qlegs=layout["legs"]["q"]; joints=len(qlegs)
    grams=[]; grows=[]; gcols=[]
    with threadpool_limits(limits=1):
        for leg in np.unique(qlegs):
            cols=(np.flatnonzero(layout["legs"]["motor"][free]==leg)[:,None]*6+np.arange(6)).ravel()
            if not len(cols): continue
            rows=np.flatnonzero(np.tile(qlegs,len(train_offset)//joints)==leg)
            D=train_design[rows][:,cols].toarray(); gram=2*(D.T@D)/len(train_offset)
            grams.append(gram.ravel()); grows.append(np.repeat(cols,len(cols))); gcols.append(np.tile(cols,len(cols)))
            linear[cols]=2*(D.T@train_offset[rows])/len(train_offset)
    P+=sparse.csc_matrix((np.concatenate(grams),(np.concatenate(grows),np.concatenate(gcols))),shape=(n,n))
    linear-=2*REGULARIZATION*layout["x0"]
    rr=[]; cc=[]; vv=[]; rhs=[]
    def row(indices,values,bound):
        rr.extend([len(rhs)]*len(indices)); cc.extend(indices); vv.extend(values); rhs.append(bound)
    for pair in range(pairs):
        lo,hi,lower,upper=2*pair,2*pair+1,slopes+pair,slopes+pairs+pair
        row([lower],[-1.],0.); row([upper],[1.],2.)
        row([lower,lo],[1.,-1.],0.); row([lower,hi],[1.,-1.],0.)
        row([lo,upper],[1.,-1.],0.); row([hi,upper],[1.,-1.],0.)
    for muscle in layout["variable_muscle"]:
        motors=np.flatnonzero(member[free]==muscle); pp=(motors[:,None]*3+np.arange(3)).ravel()
        row((slopes+pp).tolist(),[-1.]*len(pp),-.500001); row((slopes+pairs+pp).tolist(),[1.]*len(pp),1.999999)
    counts=np.bincount(member,minlength=len(layout["mapping"]["fixed"]))
    for index,motor in enumerate(free): row((slopes+3*index+np.arange(3)).tolist(),[-1.]*3,-1e-6/counts[member[motor]])
    envelope=sparse.csc_matrix((vv,(rr,cc)),shape=(len(rhs),n))
    pad=sparse.hstack((startup_design,sparse.csc_matrix((len(startup_offset),n-slopes))),format="csc")
    A=sparse.vstack((envelope,pad,-pad,sparse.csc_matrix((1,n)),-pad),format="csc")
    radius=np.sqrt(len(startup_offset))*Q_RMS
    b=np.r_[rhs,Q_PEAK-startup_offset,Q_PEAK+startup_offset,radius,startup_offset]
    dims=(len(rhs)+2*len(startup_offset),1+len(startup_offset))
    return P,linear,A,b,dims


def decoder_problem(mapping, legs, training, startup, targets):
    base.require(set(training)==set(startup)=={"bank","original_activation"}
        and set(targets)=={"train_q","startup_q","startup_ltm"}, "Training-only/startup-only optimizer objects required;no development")
    layout=decoder_layout(mapping,legs)
    for data,limit in ((training,1200),(startup,99)):
        _response_inputs(mapping,data["bank"],data["original_activation"]); base.require(len(data["bank"])<=limit,"Optimizer sample extent exceeds frozen training/startup prefix")
    td,to=affine_q(mapping,layout,training["bank"],training["original_activation"])
    sd,so=affine_q(mapping,layout,startup["bank"],startup["original_activation"])
    for key,shape in (("train_q",to.shape),("startup_q",so.shape),("startup_ltm",(len(startup["bank"]),len(mapping["ltm"])))):
        v=targets[key]; base.require(v.dtype==np.float64 and v.shape==shape and np.isfinite(v).all(), "Exact optimizer target: "+key)
    to=(to-targets["train_q"]).ravel(order="C"); so=(so-targets["startup_q"]).ravel(order="C")
    P,q,A,b,dims=_equations(layout,td,to,sd,so)
    le=.2*np.tanh(startup["original_activation"][:,mapping["ltm"]]/10)-targets["startup_ltm"]
    return {"P":P,"q":q,"A":A,"b":b,"cones":[clarabel.NonnegativeConeT(dims[0]),clarabel.SecondOrderConeT(dims[1])],
        "cone_dimensions":dims,"layout":layout,"train_design":td,"train_error_offset":to,"startup_design":sd,
        "startup_error_offset":so,"startup_ltm_error":le,"objective_constant":float(base.mean_square(to)+REGULARIZATION*np.dot(layout["x0"],layout["x0"]))}


def validate_problem(p):
    base.require(set(p)==PROBLEM_KEYS,"Exact training/startup decoder problem;no hidden development metadata")
    layout=decoder_layout(p["layout"]["mapping"],p["layout"]["legs"])
    for k in ("free_motor","variable_muscle","fixed_muscle","x0"): base.require(np.array_equal(layout[k],p["layout"][k]),"Original layout changed: "+k)
    for k in ("pairs","slope_count","variables"): base.require(layout[k]==p["layout"][k],"Original layout count changed")
    n=layout["variables"]; qn=len(layout["mapping"]["rest"])
    for name,max_rows in (("train",1200*qn),("startup",99*qn)):
        D,r=p[name+"_design"],p[name+"_error_offset"]
        base.require(sparse.isspmatrix_csc(D) and D.dtype==np.float64 and D.shape==(len(r),layout["slope_count"])
            and 0<len(r)<=max_rows and len(r)%qn==0 and D.has_canonical_format and np.isfinite(D.data).all()
            and r.dtype==np.float64 and r.shape==(len(r),) and np.isfinite(r).all(), "Original finite affine metadata required")
    exact=_equations(layout,p["train_design"],p["train_error_offset"],p["startup_design"],p["startup_error_offset"])
    for name,v in zip(("P","q","A","b"),exact[:4],strict=True):
        actual=p[name]
        if sparse.issparse(v):
            base.require(sparse.isspmatrix_csc(actual) and actual.dtype==np.float64 and actual.shape==v.shape
                and actual.has_canonical_format and np.isfinite(actual.data).all(),"Original finite sparse decoder equation required")
            delta=actual-v; delta.eliminate_zeros(); base.require(delta.nnz==0,"Original decoder equation changed: "+name)
        else: base.require(actual.dtype==np.float64 and np.array_equal(actual,v),"Original decoder RHS changed: "+name)
    dims=exact[4]
    base.require(tuple(p["cone_dimensions"])==dims and len(p["cones"])==2 and isinstance(p["cones"][0],clarabel.NonnegativeConeT)
        and isinstance(p["cones"][1],clarabel.SecondOrderConeT) and tuple(v.dim for v in p["cones"])==dims,"Original joint startup cones changed")
    le=p["startup_ltm_error"]
    base.require(le.dtype==np.float64 and le.shape==(len(p["startup_error_offset"])//qn,len(layout["mapping"]["ltm"])) and np.isfinite(le).all()
        and p["objective_constant"]==float(base.mean_square(p["train_error_offset"])+REGULARIZATION*np.dot(layout["x0"],layout["x0"])),"Original fixed LTM/objective constant changed")
    return n,dims


def primal_certificate(p, x, solver_s, solver_y, status):
    n,dims=validate_problem(p); x,s,y=(np.asarray(v) for v in (x,solver_s,solver_y))
    finite=all(v.dtype==np.float64 and v.shape==shape and np.isfinite(v).all() for v,shape in ((x,(n,)),(s,p["b"].shape),(y,p["b"].shape)))
    receipt={"status":str(status),"finite_raw":finite,"valid":False,"optimality_claim":False}
    if not finite: return {"valid":False,"receipt":receipt,"primal_slack":None}
    with np.errstate(over="ignore",invalid="ignore"):
        exact=p["b"]-p["A"]@x; slacks=certificate._cone_slacks(exact,dims)
        residual=float(min(slacks)); affine=float(np.max(abs(exact-s))); objective=float(.5*x@(p["P"]@x)+p["q"]@x)
    number=certificate._number
    receipt.update(cone_slacks=[number(v) for v in slacks],minimum_residual=number(residual),
        raw_affine_infinity_norm=number(affine),raw_objective=number(objective),objective_with_constant=number(objective+p["objective_constant"]))
    receipt["valid"]=bool(str(status) in ("Solved","AlmostSolved") and np.isfinite(exact).all()
        and all(np.isfinite(v) for v in (*slacks,affine,objective)) and residual>=-1e-8)
    return {"valid":receipt["valid"],"receipt":receipt,"primal_slack":exact}


def solve_decoder(p,time_limit,*,solver_factory=clarabel.DefaultSolver,wall_clock=time.perf_counter):
    validate_problem(p)
    base.require(clarabel.__version__=="0.11.1" and np.isfinite(time_limit) and 0<time_limit<=20,"Frozen bounded single decoder solve required")
    settings=clarabel.DefaultSettings(); applied={"verbose":False,"max_threads":1,"max_iter":100,"time_limit":float(time_limit),
        "tol_gap_abs":1e-8,"tol_gap_rel":1e-8,"tol_feas":1e-8}
    for k,v in applied.items(): setattr(settings,k,v); base.require(getattr(settings,k)==v,"Decoder solver setting changed")
    receipt={k:metering.phase() for k in ("constructor","solve","certificate")}
    receipt.update(settings=applied,status="requested",iterations=None,iterations_known=False); raw=None
    try:
        solver=metering.metered(receipt,"constructor",lambda:solver_factory(p["P"],p["q"],p["A"],p["b"],p["cones"],settings),wall_clock)
        raw=metering.metered(receipt,"solve",solver.solve,wall_clock)
        receipt.update(status=str(raw.status),iterations=int(raw.iterations),iterations_known=True)
        x,s,y=(np.asarray(v,np.float64) for v in (raw.x,raw.s,raw.z))
        cert=metering.metered(receipt,"certificate",lambda:primal_certificate(p,x,s,y,raw.status),wall_clock)
        receipt.update(certificate_receipt=cert["receipt"],solution_valid=cert["valid"])
        return {**cert,"x":x,"solver_s":s,"solver_y":y,"receipt":receipt}
    except Exception as error:
        if raw is None and getattr(error,"restoration_phase",None)=="solve":
            raw=getattr(error,"phase_partial_result",None)
            if raw is not None: receipt.update(status=str(raw.status),iterations=int(raw.iterations),iterations_known=True)
        error.decoder_solve_receipt,error.partial_decoder_solution=copy.deepcopy(receipt),raw; raise


def problem_arrays(p):
    arrays={k:v for k,v in p.items() if isinstance(v,np.ndarray)}
    for k in ("P","A","train_design","startup_design"):
        v=p[k]; arrays.update({k+"_data":v.data,k+"_indices":v.indices,k+"_indptr":v.indptr,k+"_shape":np.array(v.shape,np.int64)})
    arrays.update({k:v for k,v in p["layout"].items() if isinstance(v,np.ndarray)})
    arrays.update(cone_dimensions=np.array(p["cone_dimensions"],np.int64),objective_constant=np.array(p["objective_constant"]))
    return arrays


def decoder_identity(arrays):
    digest=hashlib.sha256(b"flylab.m3.e287.decoder-payload.v1")
    for name in sorted(arrays):
        v=np.ascontiguousarray(arrays[name]); digest.update(name.encode()); digest.update(v.dtype.str.encode())
        digest.update(np.asarray(v.shape,np.int64).tobytes()); digest.update(v.tobytes())
    return digest.hexdigest()


def assess_candidate(mapping,legs,low,high,startup,teacher,targets,baseline):
    base.require(teacher["q"].shape==baseline.shape==targets["teacher_q"].shape==(2001,len(mapping["rest"]))
        and startup["q"].shape==(101,len(mapping["rest"])) and startup["ltm"].shape==(101,len(mapping["ltm"])),"Exact held-out assessment axes required")
    qe=startup["q"][1:100]-targets["startup_q"]; le=startup["ltm"][1:100]-targets["startup_ltm"]
    rms=lambda v:float(np.sqrt(base.mean_square(v)))
    checks={"startup":base.startup_ok(qe,le)}
    for name,sl,ratio in (("train",slice(1,1201),.8),("development",slice(1201,2001),1.)):
        checks[name]=bool(rms(teacher["q"][sl]-targets["teacher_q"][sl])<=ratio*rms(baseline[sl]-targets["teacher_q"][sl]))
        for leg in np.unique(legs["q"]):
            cols=legs["q"]==leg
            checks[base.LEGS[leg]+"_"+name]=bool(rms(teacher["q"][sl][:,cols]-targets["teacher_q"][sl][:,cols])<=1.05*rms(baseline[sl][:,cols]-targets["teacher_q"][sl][:,cols]))
    result={"checks":checks,"body_promotion":False}
    physical=physical_slopes(mapping,low,high)
    result["checks"].update(internal_startup=bool(np.max(abs(qe))<=Q_PEAK and np.sqrt(base.mean_square(qe))<=Q_RMS
        and np.max(abs(le))<=.025 and np.sqrt(base.mean_square(le))<=.009),slope=physical["pass"])
    result.update(selection_pass=bool(all(result["checks"].values())),physical=physical,
        startup_metrics=base.startup_metrics(qe,le),train_mse=base.mean_square(teacher["q"][1:1201]-targets["teacher_q"][1:1201]),
        development_mse=base.mean_square(teacher["q"][1201:]-targets["teacher_q"][1201:]))
    result["conditional_support"]=result["selection_pass"]
    return result
