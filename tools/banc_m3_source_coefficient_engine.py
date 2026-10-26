"""E269 bounded conditional CUDA response; no network, body, solver or fallback.

Each parameter owns one edge/channel, one MN and one muscle. Recorded source
rates, masks and histories are fixed conditional inputs, including in tangents.
The NumPy oracle is explicitly a small generated-test helper, never a fallback.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping

import numpy as np

DT = .00025
CONTROL_DT = .005
STEPS_PER_CONTROL = 20
MAX_CHUNK_STEPS = 1000
MAX_CURRENT_ELEMENTS = 65536
MAX_GPU_BYTES = 4 * 1024**3
FILTER_ALPHA = -np.expm1(-CONTROL_DT / np.array([.005, .010, .040]))
FIXED_ALPHA = -np.expm1(-CONTROL_DT / .010)
STAGE_ORDER = ("start", "half1", "half2", "end")
_NATIVE = ("initial_rate", "initial_adaptation", "tau", "coefficient",
           "threshold", "cap", "cell_gain", "cell_inverse_adaptation_tau")
_FLOAT64 = ("pool", "low", "high", "kappa", "bank0", "activation0",
            "original_activation0", "moment", "rest")
_IDS = ("motor_ids", "source_ids", "edge_ids", "edge_post", "edge_source",
        "membership", "ltm", "stage_index")
_REQUIRED = ("stage_total", "source_rate", "source_mask", "source_history",
             "weights", "fixed", *_NATIVE, *_FLOAT64, *_IDS)


def _require(test, message):
    if not test:
        raise ValueError(message)


def checked_coefficients(coefficients, edges, *, rounded=False):
    """Check edge-major triples; optionally round once for native deployment."""
    c = np.asarray(coefficients)
    _require(c.dtype == np.float64 and c.shape == (edges, 3),
             "Explicit float64 edge-major coefficient triples required")
    _require(np.isfinite(c).all() and np.all(c >= 0)
             and np.all((c.sum(1) >= .125) & (c.sum(1) <= 8)),
             "Coefficient triples must be nonnegative with sum in [.125,8]")
    if rounded:
        c = c.astype(np.float32)
        sums = c.astype(np.float64).sum(1)
        _require(np.isfinite(c).all() and np.all(c >= 0)
                 and np.all((sums >= .125) & (sums <= 8)),
                 "Rounded native coefficients exceed triple bounds")
    return np.array(c, order="C", copy=True)


def prepare_problem(mapping):
    """Validate a complete, already explicitly contracted context on CPU.

    ``stage_index`` must be the ordered local stage ordinal arange(S), and
    ``stage_order`` must name the four native RK4 stages. IDs are native IDs;
    edge_post/source and membership/ltm are explicit local indices. Pool rows
    remain normalized; contraction must retain all MNs of each selected muscle.
    Input arrays are detached only after all bounds and validity checks pass.
    """
    _require(isinstance(mapping, Mapping) and all(k in mapping for k in _REQUIRED),
             "Complete E269 context mapping required")
    _require(tuple(mapping.get("stage_order", ())) == STAGE_ORDER,
             "Native RK4 stage_order start/half1/half2/end required")
    x = {k: np.asarray(mapping[k]) for k in _REQUIRED}
    total, source = x["stage_total"], x["source_rate"]
    _require(total.ndim == 2 and total.dtype == np.float32
             and 1 <= total.shape[0] <= 160000 and total.shape[0] % 80 == 0
             and 1 <= total.shape[1] <= 391, "Bounded native float32 stage totals required")
    stages, n = total.shape
    _require(source.ndim == 2 and source.shape[0] == stages
             and 1 <= source.shape[1] <= 51, "Bounded native source extent required")
    sources = source.shape[1]
    e = x["edge_ids"].size
    pool, moment = x["pool"], x["moment"]
    _require(pool.ndim == 2 and pool.shape[1] == n and 1 <= pool.shape[0] <= 96,
             "Explicit contracted muscle pool required")
    m = pool.shape[0]
    _require(moment.ndim == 2 and moment.shape[1] == m and 1 <= moment.shape[0] <= 42
             and 1 <= e <= 1660, "Bounded command and edge extents required")
    q = moment.shape[0]
    shapes = {"source_rate": (stages, sources), "source_mask": (stages, sources),
              "source_history": (stages, sources, 2), "weights": (e,),
              "motor_ids": (n,), "source_ids": (sources,), "edge_ids": (e,),
              "edge_post": (e,), "edge_source": (e,), "membership": (n,),
              "low": (n, 3), "high": (n, 3), "kappa": (n,),
              "fixed": (m,), "bank0": (n, 3), "activation0": (m,),
              "original_activation0": (m,), "rest": (q,), "stage_index": (stages,)}
    shapes.update({k: (n,) for k in _NATIVE})
    _require(all(x[k].shape == shape for k, shape in shapes.items()),
             "Context shapes disagree")
    _require(x["ltm"].ndim == 1 and x["ltm"].size <= 6, "Explicit bounded LTM list required")
    for k in (*_NATIVE, "stage_total", "source_rate", "source_mask", "weights"):
        _require(x[k].dtype == np.float32, f"Native float32 {k} required")
    for k in (*_FLOAT64, "source_history"):
        _require(x[k].dtype == np.float64, f"Exact float64 {k} required")
    for k in _IDS:
        _require(x[k].dtype == np.int64, f"Explicit int64 {k} required")
    _require(x["fixed"].dtype == np.bool_, "Explicit bool fixed muscles required")
    _require(np.array_equal(x["stage_index"], np.arange(stages, dtype=np.int64)),
             "Stage ordinals must be complete and increasing")
    for k in ("motor_ids", "source_ids", "edge_ids"):
        _require(np.all(x[k] >= 0) and np.unique(x[k]).size == x[k].size,
                 f"Unique nonnegative native {k} required")
    for k, bound in (("edge_post", n), ("edge_source", sources),
                     ("membership", m), ("ltm", m)):
        _require(np.all((x[k] >= 0) & (x[k] < bound)), f"Invalid local {k}")
    _require(np.unique(x["ltm"]).size == x["ltm"].size,
             "Unique local LTM muscles required")
    _require(np.unique(np.stack((x["edge_post"], x["edge_source"]), 1), axis=0).shape[0] == e,
             "Duplicate source/post edges forbidden")
    # Extents and dtypes have been checked before scanning or copying stage data.
    _require(all(np.isfinite(v).all() for v in x.values()), "Nonfinite context input")
    for k in ("source_rate", "source_history", "initial_rate", "initial_adaptation",
              "cell_gain", "coefficient", "pool", "low", "high", "kappa",
              "bank0", "activation0", "original_activation0"):
        _require(np.all(x[k] >= 0), f"Nonnegative {k} required")
    _require(all(np.all(x[k] > 0) for k in ("tau", "cap", "cell_inverse_adaptation_tau")),
             "Positive membrane tau, cap and native inverse adaptation tau required")
    _require(np.all(x["weights"] != 0) and np.all((x["source_mask"] == 0)
             | (x["source_mask"] == 1)), "Nonzero edges and native binary masks required")
    _require(np.all(np.count_nonzero(pool, axis=0) == 1)
             and np.array_equal(np.argmax(pool, axis=0), x["membership"])
             and np.allclose(pool.sum(1), 1, rtol=0, atol=1e-15),
             "Pool must partition MNs according to membership with normalized rows")
    _require(np.all(x["fixed"][x["ltm"]]), "All LTM muscles must use original fixed transfer")
    singleton = np.bincount(x["membership"], minlength=m) == 1
    _require(np.all(x["fixed"][singleton]), "Singleton muscles must use original fixed transfer")
    fixed_mn = x["fixed"][x["membership"]]
    canonical = np.zeros((n, 3))
    canonical[:, 1] = pool[x["membership"], np.arange(n)]
    _require(np.array_equal(x["low"][fixed_mn], canonical[fixed_mn])
             and np.array_equal(x["high"][fixed_mn], canonical[fixed_mn]),
             "Fixed-muscle slopes must preserve anatomical transfer")
    minimum = np.minimum(x["low"], x["high"]).sum(1)
    maximum = np.maximum(x["low"], x["high"]).sum(1)
    lower = np.bincount(x["membership"], weights=minimum, minlength=m)
    upper = np.bincount(x["membership"], weights=maximum, minlength=m)
    counts = np.bincount(x["membership"], minlength=m)
    _require(np.all(lower >= .5-1e-12) and np.all(upper <= 2+1e-12)
             and np.all(minimum >= 1e-6/counts[x["membership"]]-1e-12),
             "Original monotone slope budgets and per-MN minimum must be preserved")
    expected = _activation(x, x["bank0"], x["original_activation0"])
    _require(np.allclose(x["activation0"], expected, rtol=0, atol=1e-12),
             "Initial activation disagrees with bank and original fixed activation")
    samples, params = stages // 80 + 1, e * 3
    # Include all fixed upload arrays, forward/decoder/tangent buffers, largest
    # selected full command Jacobian, transfer temporaries and allocator margin.
    fixed_bytes = sum(v.nbytes for k, v in x.items() if k != "source_history")
    fixed_bytes += stages * sources * 2 * 4 + 8 * (n + 1 + e + params)
    output_bytes = (8 * samples * (5*n + 2*m + q + x["ltm"].size + params)
                    + 9 * stages * n + 8 * samples * 2*params
                    + 8 * (2*n + 6*params) + 8*stages*n)
    estimate = 2 * (fixed_bytes + output_bytes) + 64 * 1024**2
    _require(estimate <= MAX_GPU_BYTES, "Estimated engine allocation exceeds 4 GiB")
    _require(np.max(x["source_history"]) <= np.finfo(np.float32).max,
             "Native-rounded history overflows float32")
    result = {k: np.array(v, order="C", copy=True) for k, v in x.items()}
    result["history32"] = result["source_history"].astype(np.float32)
    _require(np.isfinite(result["history32"]).all(), "Native-rounded history overflows float32")
    order = np.argsort(result["edge_post"], kind="stable").astype(np.int64)
    result["edge_order"] = order
    result["edge_ptr"] = np.r_[0, np.cumsum(np.bincount(x["edge_post"], minlength=n))].astype(np.int64)
    result["parameter_muscle"] = np.repeat(x["membership"][x["edge_post"]], 3)
    result.update(stages=stages, neurons=n, sources=sources, edges=e, muscles=m,
                  commands=q, samples=samples, params=params, gpu_bytes_estimate=estimate,
                  current_bytes=8*stages*n,
                  stage_order=STAGE_ORDER)
    for v in result.values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return result


def _activation(p, bank, original):
    result = np.bincount(p["membership"], weights=np.sum(
        p["low"]*bank + (p["high"]-p["low"])*np.maximum(bank-p["kappa"][:, None], 0), axis=1),
        minlength=p["pool"].shape[0])
    result[p["fixed"]] = original[p["fixed"]]
    return result


def numpy_decoder(mapping, rates):
    """Pure CPU E150 replay of endpoint rates, retaining the initial row."""
    p = prepare_problem(mapping)
    r = np.asarray(rates)
    _require(r.shape == (p["samples"], p["neurons"]) and r.dtype in (np.float32, np.float64)
             and np.isfinite(r).all() and np.all(r >= 0), "Finite endpoint rate matrix required")
    _require(np.array_equal(r[0], p["initial_rate"].astype(r.dtype)), "Initial endpoint rate differs")
    bank = np.empty((p["samples"], p["neurons"], 3))
    activation = np.empty((p["samples"], p["muscles"]))
    original = np.empty_like(activation)
    bank[0], activation[0], original[0] = p["bank0"], p["activation0"], p["original_activation0"]
    for s in range(1, p["samples"]):
        bank[s] = bank[s-1] + FILTER_ALPHA*(r[s, :, None]-bank[s-1])
        original[s] = original[s-1] + FIXED_ALPHA*(p["pool"]@r[s]-original[s-1])
        activation[s] = _activation(p, bank[s], original[s])
    return {"bank": bank, "activation": activation, "original_activation": original,
            "q": p["rest"] + .02*(activation@p["moment"].T),
            "ltm": .2*np.tanh(activation[:, p["ltm"]]/10)}


# Backward-compatible pure helper name; the engine's decode_rates is GPU-only.
decode_rates = numpy_decoder


def numpy_oracle(mapping, coefficients, *, derivative=True):
    """Independent small NumPy RK4 oracle restricted to generated short inputs."""
    p = prepare_problem(mapping)
    _require(p["stages"] <= 8000 and p["neurons"] <= 16 and p["edges"] <= 32,
             "NumPy oracle is restricted to small generated fixtures")
    c = checked_coefficients(coefficients, p["edges"])
    n, params = p["neurons"], p["params"]
    r, z = p["initial_rate"].astype(float), p["initial_adaptation"].astype(float)
    out_r, out_z = np.empty((p["samples"], n)), np.empty((p["samples"], n))
    phi, branch = np.empty((p["stages"], n)), np.empty((p["stages"], n), bool)
    out_r[0], out_z[0] = r, z
    v, w = np.zeros(params), np.zeros(params)
    vr = np.zeros((p["samples"], params))
    wz = np.zeros_like(vr)
    rows = np.repeat(p["edge_post"], 3)
    gain, inv, tau = [p[k].astype(float) for k in ("cell_gain", "cell_inverse_adaptation_tau", "tau")]
    for step in range(p["stages"]//4):
        kr, kz, kv, kw = [], [], [], []
        for stage in range(4):
            idx = 4*step+stage
            h = DT if stage == 3 else DT/2
            rr, zz = (r, z) if stage == 0 else (r+h*kr[-1], z+h*kz[-1])
            u = np.column_stack((p["source_rate"][idx], p["history32"][idx])).astype(float)
            scale = p["weights"].astype(float)*p["source_mask"][idx, p["edge_source"]].astype(float)
            current = p["stage_total"][idx].astype(float)
            for edge in range(p["edges"]):
                value = ((c[edge, 0]-1)*u[p["edge_source"][edge], 0]
                         + c[edge, 1]*u[p["edge_source"][edge], 1]
                         + c[edge, 2]*u[p["edge_source"][edge], 2])
                current[p["edge_post"][edge]] += scale[edge]*value
            t = np.tanh(p["coefficient"].astype(float)*(current-p["threshold"].astype(float)-zz))
            branch[idx] = t > 0
            phi[idx] = np.where(t > 0, p["cap"].astype(float)*p["coefficient"].astype(float)*(1-t*t), 0)
            kr.append((p["cap"].astype(float)*np.maximum(t, 0)-rr)/tau)
            kz.append((gain*rr-zz)*inv)
            if derivative:
                vv, ww = (v, w) if stage == 0 else (v+h*kv[-1], w+h*kw[-1])
                di = (scale[:, None]*u[p["edge_source"]]).ravel()
                kv.append((phi[idx, rows]*(di-ww)-vv)/tau[rows])
                kw.append((gain[rows]*vv-ww)*inv[rows])
        r += DT/6*(kr[0]+2*kr[1]+2*kr[2]+kr[3])
        z += DT/6*(kz[0]+2*kz[1]+2*kz[2]+kz[3])
        if derivative:
            v += DT/6*(kv[0]+2*kv[1]+2*kv[2]+kv[3])
            w += DT/6*(kw[0]+2*kw[1]+2*kw[2]+kw[3])
        if (step+1) % 20 == 0:
            sample = (step+1)//20
            out_r[sample], out_z[sample] = r, z
            vr[sample], wz[sample] = v, w
    result = {"rate": out_r, "adaptation": out_z, "phi": phi, "branch": branch,
              **numpy_decoder(mapping, out_r)}
    if derivative:
        db = np.zeros((params, 3))
        da = np.zeros(params)
        b = np.zeros((p["samples"], params))
        muscle = p["parameter_muscle"]
        for sample in range(1, p["samples"]):
            db += FILTER_ALPHA*(vr[sample, :, None]-db)
            da += FIXED_ALPHA*(p["pool"][muscle, rows]*vr[sample]-da)
            slope = p["low"][rows] + (p["high"][rows]-p["low"][rows])*(
                result["bank"][sample, rows] > p["kappa"][rows, None])
            b[sample] = np.where(p["fixed"][muscle], da, np.sum(slope*db, axis=1))
        result.update(B=b, parameter_muscle=muscle.copy(), rate_sensitivity=vr,
                      adaptation_sensitivity=wz)
    return result


def activation_cotangent_product(result, cotangent):
    """CPU helper for compressed B; GPU callers use the engine method."""
    b, muscle = result["B"], result["parameter_muscle"]
    cot = np.asarray(cotangent)
    _require(cot.dtype == np.float64 and cot.ndim == 2 and cot.shape[0] == b.shape[0]
             and np.all(muscle < cot.shape[1]) and np.isfinite(cot).all(),
             "Finite float64 activation cotangent required")
    return np.sum(b*cot[:, muscle], axis=0)


def command_jacobian(mapping, result, samples):
    """CPU selected q and LTM Jacobians from compressed activation tangents."""
    idx = _sample_indices(samples, result["B"].shape[0])
    muscle = result["parameter_muscle"]
    b = result["B"][idx]
    q = .02*np.asarray(mapping["moment"])[:, muscle][None, :, :]*b[:, None, :]
    ltm = np.asarray(mapping["ltm"])
    factors = .02*(1-np.tanh(result["activation"][idx][:, ltm]/10)**2)
    tendon = factors[:, :, None]*(ltm[:, None] == muscle[None, :])[None, :, :]*b[:, None, :]
    return {"q": q, "ltm": tendon}


def _sample_indices(samples, count):
    idx = np.asarray(samples)
    _require(idx.ndim == 1 and idx.dtype == np.int64 and idx.size > 0
             and np.all((idx >= 0) & (idx < count)), "Explicit valid int64 sample indices required")
    return np.array(idx, copy=True)


CUDA_SOURCE = r'''
extern "C" __global__ void forward(
 const float* total,const float* src,const float* mask,const float* hist,
 const float* weights,const long long* post,const long long* cols,
 const long long* ptr,const long long* order,const double* c,
 const float* tau,const float* coeff,const float* threshold,const float* cap,
 const float* gain,const float* inv,double* state,double* rate,double* adapt,
 double* phi,unsigned char* branch,int n,int sources,int start,int count) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double r=state[2*i],z=state[2*i+1];
 if(start==0){rate[i]=r;adapt[i]=z;}
 for(int step=start;step<start+count;++step){
  double kr[4],kz[4];
  for(int st=0;st<4;++st){
   int row=step*4+st;double h=st==3?.00025:.000125;
   double rr=r,zz=z;if(st){rr+=h*kr[st-1];zz+=h*kz[st-1];}
   double input=(double)total[row*n+i];
   for(long long j=ptr[i];j<ptr[i+1];++j){
    long long e=order[j],s=row*sources+cols[e];
    double u=((c[3*e]-1.)*(double)src[s]+c[3*e+1]*(double)hist[2*s])
                 +c[3*e+2]*(double)hist[2*s+1];
    input+=((double)weights[e]*(double)mask[s])*u;
   }
   double t=tanh((double)coeff[i]*(input-(double)threshold[i]-zz));
   branch[row*n+i]=t>0.;
   phi[row*n+i]=t>0.?(double)cap[i]*(double)coeff[i]*(1.-t*t):0.;
   kr[st]=((double)cap[i]*(t>0.?t:0.)-rr)/(double)tau[i];
   kz[st]=((double)gain[i]*rr-zz)*(double)inv[i];
  }
  r+=.00025/6.*(((kr[0]+2.*kr[1])+2.*kr[2])+kr[3]);
  z+=.00025/6.*(((kz[0]+2.*kz[1])+2.*kz[2])+kz[3]);
  if((step+1)%20==0){int out=(step+1)/20*n+i;rate[out]=r;adapt[out]=z;}
 }
 state[2*i]=r;state[2*i+1]=z;
}
extern "C" __global__ void bank_decode(const double* rate,const double* bank0,
 double* bank,int n,int first,int count){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n*3)return;
 int cell=i/3,ch=i%3;double tau=ch==0?.005:(ch==1?.010:.040);
 double alpha=-expm1(-.005/tau),b;
 if(first==1){b=bank0[i];bank[i]=b;}else b=bank[(first-1)*n*3+i];
 for(int s=first;s<first+count;++s){b+=alpha*(rate[s*n+cell]-b);bank[s*n*3+i]=b;}
}
extern "C" __global__ void activation_decode(const double* rate,const double* bank,
 const double* pool,const long long* member,const double* low,const double* high,
 const double* kappa,const bool* fixed,const double* act0,const double* orig0,
 double* act,double* orig,int n,int m,int first,int count){
 int muscle=blockDim.x*blockIdx.x+threadIdx.x;if(muscle>=m)return;
 double a;if(first==1){a=orig0[muscle];orig[muscle]=a;act[muscle]=act0[muscle];}
 else a=orig[(first-1)*m+muscle];
 double alpha=-expm1(-.005/.010);
 for(int s=first;s<first+count;++s){
  double pooled=0.,v=0.;
  for(int i=0;i<n;++i){pooled+=pool[muscle*n+i]*rate[s*n+i];
   if(member[i]==muscle)for(int ch=0;ch<3;++ch){
    double b=bank[(s*n+i)*3+ch],hinge=b-kappa[i];
    v+=low[3*i+ch]*b+(high[3*i+ch]-low[3*i+ch])*(hinge>0.?hinge:0.);
   }
  }
  a+=alpha*(pooled-a);orig[s*m+muscle]=a;act[s*m+muscle]=fixed[muscle]?a:v;
 }
}
extern "C" __global__ void tangent(
 const float* src,const float* mask,const float* hist,const float* weights,
 const long long* post,const long long* cols,const long long* pm,
 const double* phi,const float* tau,const float* gain,const float* inv,
 const double* pool,const bool* fixed,const double* low,const double* high,
 const double* kappa,const double* bank,double* state,double* B,double* vr,double* wz,
 int n,int sources,int params,int start,int count){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=params)return;
 int e=p/3,ch=p%3,i=post[e],m=pm[p];
 double v=state[6*p],w=state[6*p+1],db[3]={state[6*p+2],state[6*p+3],state[6*p+4]},a=state[6*p+5];
 double al[3]={-expm1(-1.),-expm1(-.5),-expm1(-.125)},af=-expm1(-.5);
 if(start==0){B[p]=0.;vr[p]=0.;wz[p]=0.;}
 for(int step=start;step<start+count;++step){
  double kv[4],kw[4];
  for(int st=0;st<4;++st){int row=4*step+st,s=row*sources+cols[e];
   double h=st==3?.00025:.000125,vv=v,ww=w;
   if(st){vv+=h*kv[st-1];ww+=h*kw[st-1];}
   double u=ch==0?(double)src[s]:(double)hist[2*s+ch-1];
   double di=((double)weights[e]*(double)mask[s])*u;
   kv[st]=(phi[row*n+i]*(di-ww)-vv)/(double)tau[i];
   kw[st]=((double)gain[i]*vv-ww)*(double)inv[i];
  }
  v+=.00025/6.*(((kv[0]+2.*kv[1])+2.*kv[2])+kv[3]);
  w+=.00025/6.*(((kw[0]+2.*kw[1])+2.*kw[2])+kw[3]);
  if((step+1)%20==0){int sample=(step+1)/20,out=sample*params+p;double b=0.;
   for(int k=0;k<3;++k){db[k]+=al[k]*(v-db[k]);
    double slope=low[3*i+k]+(high[3*i+k]-low[3*i+k])*(bank[(sample*n+i)*3+k]>kappa[i]);
    b+=slope*db[k];
   }
   a+=af*(pool[m*n+i]*v-a);B[out]=fixed[m]?a:b;vr[out]=v;wz[out]=w;
  }
 }
 state[6*p]=v;state[6*p+1]=w;for(int k=0;k<3;++k)state[6*p+2+k]=db[k];state[6*p+5]=a;
}
'''


def _precomputed_source(reference):
    """Append E273 kernels while preserving the approved reference verbatim."""
    forward = reference[:reference.index('extern "C" __global__ void bank_decode')]
    header_end = forward.index(" const float* tau")
    forward = ('extern "C" __global__ void forward_precomputed(\n'
               ' const double* prepared,\n'+forward[header_end:])
    index = "int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;"
    _require(forward.count(index) == 1, "Frozen forward thread index changed")
    forward = forward.replace(index, "int i=blockIdx.x;if(i>=n)return;")
    start = forward.index("   double input=(double)total[row*n+i];")
    end = forward.index("   double t=tanh", start)
    current = forward[start:end]
    forward = forward[:start]+"   double input=prepared[row*n+i];\n"+forward[end:]
    # The exact current expression, including edge order and parentheses, comes
    # directly from the approved reference. Only the element index/store differ.
    prepare = r'''
extern "C" __global__ void current_precompute(
 const float* total,const float* src,const float* mask,const float* hist,
 const float* weights,const long long* cols,const long long* ptr,
 const long long* order,const double* c,double* prepared,
 int n,int sources,int first,int count) {
 int local=blockDim.x*blockIdx.x+threadIdx.x;if(local>=count)return;
 int element=first+local,row=element/n,i=element%n;
'''+current+r'''
 prepared[element]=input;
}
'''
    return prepare+forward


CUDA_SOURCE += _precomputed_source(CUDA_SOURCE)


def _receipt():
    return {"requested_count": 0, "completed_count": 0, "requested_bytes": 0,
            "completed_bytes": 0, "wall_s": 0.0}


def _current_receipt():
    return {**_receipt(), "requested_elements": 0, "completed_elements": 0}


class CudaSourceCoefficientEngine:
    """One owned conditional context, CUDA double and a fixed private stream.

    Validation occurs before importing CuPy or creating a GPU object. Failures
    raise the primary exception with response_receipt and partial_response;
    completion ranges always refer to synchronized chunks, never guessed work.
    No kernel retries or automatic host rescue are performed.
    """

    def __init__(self, mapping, *, chunk_steps=MAX_CHUNK_STEPS,
                 forward_implementation="precomputed"):
        _require(forward_implementation in ("reference", "precomputed"),
                 "Explicit reference or precomputed forward implementation required")
        self.problem = prepare_problem(mapping)
        _require(type(chunk_steps) is int and 20 <= chunk_steps <= MAX_CHUNK_STEPS
                 and chunk_steps % 20 == 0, "Chunk size must be 20..1000 steps, multiple of 20")
        self.chunk_steps = chunk_steps
        self.forward_implementation = forward_implementation
        self.receipt = {"upload": _receipt(), "solve": _receipt(),
                        "derivative": _receipt(), "host_copy": _receipt()}
        self.receipt["products"] = {k: _receipt() for k in ("cotangent", "command_jacobian")}
        self.receipt["current"] = _current_receipt()
        self._current_response = None
        self._current_arrays = {}
        self._product_bytes = 0
        self._active_receipt = None
        self.device_data = {}
        self.jit_wall_s = 0.0
        try:
            import cupy as cp
            self.cp = cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("Actual CUDA GPU required; no CPU fallback")
            self.device = cp.cuda.Device()
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.kernels = {k: cp.RawKernel(CUDA_SOURCE, k, options=("--std=c++11", "--fmad=false"))
                            for k in ("forward", "current_precompute", "forward_precomputed",
                                      "bank_decode", "activation_decode", "tangent")}
            with self.device, self.stream:
                for name, v in self.problem.items():
                    if isinstance(v, np.ndarray) and name != "source_history":
                        self.device_data[name] = self._upload(v)
                for kernel in self.kernels.values():
                    began = time.perf_counter()
                    try:
                        kernel.compile()
                    finally:
                        self.jit_wall_s += time.perf_counter()-began
        except Exception as error:
            error.response_receipt = copy.deepcopy(self.receipt)
            error.partial_response = self.device_data.copy()
            raise

    def _upload(self, value, receipt=None):
        bucket = self.receipt["upload"] if receipt is None else receipt
        began = time.perf_counter()
        bucket["requested_count"] += 1
        bucket["requested_bytes"] += value.nbytes
        completed = False
        try:
            result = self.cp.asarray(np.array(value, order="C", copy=True))
            self.stream.synchronize()
            bucket["completed_count"] += 1
            bucket["completed_bytes"] += value.nbytes
            completed = True
            return result
        finally:
            elapsed = time.perf_counter()-began
            bucket["wall_s"] += elapsed
            if receipt is None and hasattr(self, "last_receipt"):
                current = self.last_receipt["upload"]
                current["requested_count"] += 1
                current["requested_bytes"] += value.nbytes
                if completed:
                    current["completed_count"] += 1
                    current["completed_bytes"] += value.nbytes
                current["wall_s"] += elapsed

    def host_copy(self, value, receipt=None, *, label="unlabelled"):
        """Meter selected device copies in lifetime and current-call receipts.

        ``engine.receipt`` contains lifetime totals; ``engine.last_receipt``
        is the most recent solve/decode receipt, shared with its returned dict.
        Labels retain per-copy requested/completed byte/count and wall detail.
        Only current-response arrays (or their views) may be copied. A newer
        solve/decode invalidates old response ownership before any GPU work.
        """
        bucket = self.receipt["host_copy"] if receipt is None else receipt
        _require(isinstance(label, str) and bool(label), "Nonempty host-copy label required")
        if receipt is None:
            self._require_current_array(value)
        else:
            _require(self._active_receipt is not None
                     and receipt is self._active_receipt["host_copy"],
                     "Explicit copy receipt must belong to the active response")
        event = {"label": label, "bytes": int(value.nbytes), "completed": False}
        events = getattr(self, "host_copy_events", None)
        if events is None:
            self.host_copy_events = events = []
        events.append(event)
        began = time.perf_counter()
        failure = None
        bucket["requested_count"] += 1
        bucket["requested_bytes"] += value.nbytes
        try:
            with self.device, self.stream:
                result = self.cp.asnumpy(value, stream=self.stream)
                self.stream.synchronize()
            bucket["completed_count"] += 1
            bucket["completed_bytes"] += value.nbytes
            event["completed"] = True
            return result
        except Exception as error:
            failure = error
            raise
        finally:
            elapsed = time.perf_counter()-began
            event["wall_s"] = elapsed
            bucket["wall_s"] += elapsed
            if receipt is None and hasattr(self, "last_receipt"):
                current = self.last_receipt["host_copy"]
                current["requested_count"] += 1
                current["requested_bytes"] += value.nbytes
                if event["completed"]:
                    current["completed_count"] += 1
                    current["completed_bytes"] += value.nbytes
                current["wall_s"] += elapsed
                self.last_receipt.setdefault("host_copy_events", []).append(event.copy())
            if failure is not None:
                failure.response_receipt = {
                    "operation": "host_copy", "label": label,
                    "input_response_receipt": self._receipt_snapshot(self.last_receipt),
                    "host_copy": copy.deepcopy(bucket)}
                failure.partial_response = {"copy_input": value}

    @staticmethod
    def _receipt_snapshot(receipt):
        return copy.deepcopy({k: v for k, v in receipt.items()
                              if k not in ("host_copy_events", "product_events")})

    def _start_response(self, receipt):
        receipt["products"] = {k: _receipt() for k in ("cotangent", "command_jacobian")}
        receipt["current"] = _current_receipt()
        receipt.update(forward_implementation=self.forward_implementation,
                       planned_current_elements=0, completed_current_elements=0,
                       pending_current_range=None)
        self.last_receipt = receipt
        self._active_receipt = receipt
        self._current_response = None
        self._current_arrays = {}
        self._product_bytes = 0

    def _register_arrays(self, values, *, product=False):
        for value in values.values():
            if isinstance(value, self.cp.ndarray):
                if product and id(value) not in self._current_arrays:
                    self._product_bytes += value.nbytes
                self._current_arrays[id(value)] = value

    def _bind_response(self, result, receipt):
        self._register_arrays(result)
        self.last_receipt = receipt
        self._current_response = result
        self._active_receipt = None

    def _require_current_array(self, value):
        original, visited = value, set()
        while value is not None and id(value) not in visited:
            if self._current_arrays.get(id(value)) is value:
                return
            visited.add(id(value))
            value = getattr(value, "base", None)
        error = ValueError("Host copy requires an array owned by the current response")
        error.response_receipt = {"operation": "host_copy_validation",
                                  "current_response_receipt": self._receipt_snapshot(self.last_receipt)}
        error.partial_response = {"rejected_copy_input": original}
        raise error

    def solve(self, coefficients, *, derivative=True, host=False):
        """Return rate/z, actual E150 outputs, branch/phi and optional compressed B.

        Every call starts at the context's complete initial state. Parameter
        order is edge-major, channels raw/fast/slow. Receipt separates upload,
        forward+decoder, derivative and host-copy requests/completions.
        """
        p, cp, d = self.problem, self.cp, self.device_data
        c = checked_coefficients(coefficients, p["edges"])
        _require(type(derivative) is bool and type(host) is bool, "Explicit boolean solve flags required")
        began = time.perf_counter()
        receipt = {k: _receipt() for k in ("upload", "solve", "derivative", "host_copy")}
        receipt.update(requested_steps=p["stages"]//4, completed_steps=0,
                       derivative_completed_steps=0, completed_samples=0,
                       derivative_completed_samples=0,
                       status="requested", chunk_steps=self.chunk_steps,
                       gpu_bytes_estimate=p["gpu_bytes_estimate"])
        self._start_response(receipt)
        out = {}
        active_bucket = None
        n, m, samples, params = [p[k] for k in ("neurons", "muscles", "samples", "params")]
        try:
            with self.device, self.stream:
                dc = self._upload(c, receipt["upload"])
                if self.forward_implementation == "precomputed":
                    elements = p["stages"]*n
                    receipt["planned_current_elements"] = elements
                    out["current"] = cp.full((p["stages"], n), cp.nan, cp.float64)
                    for first in range(0, elements, MAX_CURRENT_ELEMENTS):
                        count = min(MAX_CURRENT_ELEMENTS, elements-first)
                        tick, active_bucket = time.perf_counter(), receipt["current"]
                        active_bucket["requested_count"] += 1
                        active_bucket["requested_elements"] += count
                        active_bucket["requested_bytes"] += count*8
                        receipt["status"] = "current_requested"
                        receipt["pending_current_range"] = [first, first+count]
                        self.kernels["current_precompute"](((count+255)//256,), (256,), (
                            *(d[k] for k in ("stage_total", "source_rate", "source_mask", "history32",
                                            "weights", "edge_source", "edge_ptr", "edge_order")), dc,
                            out["current"], np.int32(n), np.int32(p["sources"]),
                            np.int32(first), np.int32(count)))
                        self.stream.synchronize()
                        active_bucket["completed_count"] += 1
                        active_bucket["completed_elements"] += count
                        active_bucket["completed_bytes"] += count*8
                        active_bucket["wall_s"] += time.perf_counter()-tick
                        active_bucket = None
                        receipt["completed_current_elements"] = first+count
                        receipt["pending_current_range"] = None
                shape = (samples, n)
                for key in ("rate", "adaptation"):
                    out[key] = cp.full(shape, cp.nan, cp.float64)
                out["phi"] = cp.full((p["stages"], n), cp.nan, cp.float64)
                out["branch"] = cp.full((p["stages"], n), 255, cp.uint8)
                out["bank"] = cp.full((samples, n, 3), cp.nan, cp.float64)
                for key in ("activation", "original_activation"):
                    out[key] = cp.full((samples, m), cp.nan, cp.float64)
                state = cp.stack((d["initial_rate"], d["initial_adaptation"]), 1).astype(cp.float64)
                if derivative:
                    for key in ("B", "rate_sensitivity", "adaptation_sensitivity"):
                        out[key] = cp.full((samples, params), cp.nan, cp.float64)
                    out["parameter_muscle"] = d["parameter_muscle"]
                    tangent_state = cp.zeros((params, 6), cp.float64)
                for start in range(0, p["stages"]//4, self.chunk_steps):
                    count = min(self.chunk_steps, p["stages"]//4-start)
                    first, controls = start//20+1, count//20
                    tick = time.perf_counter()
                    bucket = receipt["solve"]
                    active_bucket = bucket
                    bucket["requested_count"] += 1
                    bucket["requested_bytes"] += count*4*n*9 + controls*(5*n+2*m)*8
                    receipt["status"] = "forward_requested"
                    common = (*(d[k] for k in ("tau", "coefficient", "threshold", "cap", "cell_gain",
                                             "cell_inverse_adaptation_tau")), state,
                              out["rate"], out["adaptation"], out["phi"], out["branch"],
                              np.int32(n), np.int32(p["sources"]), np.int32(start), np.int32(count))
                    if self.forward_implementation == "reference":
                        self.kernels["forward"](((n+31)//32,), (32,), (
                            *(d[k] for k in ("stage_total", "source_rate", "source_mask", "history32",
                              "weights", "edge_post", "edge_source", "edge_ptr", "edge_order")), dc, *common))
                    else:
                        self.kernels["forward_precomputed"]((n,), (1,), (out["current"], *common))
                    self.kernels["bank_decode"](((3*n+31)//32,), (32,),
                        (out["rate"], d["bank0"], out["bank"], np.int32(n), np.int32(first), np.int32(controls)))
                    self.kernels["activation_decode"](((m+31)//32,), (32,), (
                        out["rate"], out["bank"], *(d[k] for k in ("pool", "membership", "low",
                        "high", "kappa", "fixed", "activation0", "original_activation0")),
                        out["activation"], out["original_activation"], np.int32(n), np.int32(m),
                        np.int32(first), np.int32(controls)))
                    self.stream.synchronize()
                    bucket["completed_count"] += 1
                    bucket["completed_bytes"] += count*4*n*9 + controls*(5*n+2*m)*8
                    bucket["wall_s"] += time.perf_counter()-tick
                    active_bucket = None
                    receipt.update(completed_steps=start+count, completed_samples=(start+count)//20+1)
                    if derivative:
                        tick = time.perf_counter()
                        bucket = receipt["derivative"]
                        active_bucket = bucket
                        bucket["requested_count"] += 1
                        bucket["requested_bytes"] += controls*params*3*8
                        receipt["status"] = "derivative_requested"
                        self.kernels["tangent"](((params+31)//32,), (32,), (
                            *(d[k] for k in ("source_rate", "source_mask", "history32", "weights",
                                             "edge_post", "edge_source", "parameter_muscle")), out["phi"],
                            *(d[k] for k in ("tau", "cell_gain", "cell_inverse_adaptation_tau", "pool",
                                             "fixed", "low", "high", "kappa")), out["bank"], tangent_state,
                            out["B"], out["rate_sensitivity"], out["adaptation_sensitivity"],
                            np.int32(n), np.int32(p["sources"]), np.int32(params), np.int32(start), np.int32(count)))
                        self.stream.synchronize()
                        bucket["completed_count"] += 1
                        bucket["completed_bytes"] += controls*params*3*8
                        bucket["wall_s"] += time.perf_counter()-tick
                        active_bucket = None
                        receipt.update(derivative_completed_steps=start+count,
                                       derivative_completed_samples=(start+count)//20+1)
                out["q"] = d["rest"] + .02*(out["activation"]@d["moment"].T)
                out["ltm"] = .2*cp.tanh(out["activation"][:, d["ltm"]]/10)
                self.stream.synchronize()
                if host:
                    copied = {}
                    for k, v in out.items():
                        copied[k] = self.host_copy(v, receipt["host_copy"], label=k)
                    out = copied
                receipt["status"] = "complete"
        except Exception as error:
            if active_bucket is not None:
                active_bucket["wall_s"] += time.perf_counter()-tick
            receipt["status"] = "failed"
            receipt["wall_s"] = time.perf_counter()-began
            error.response_receipt = copy.deepcopy(receipt)
            error.partial_response = out.copy()
            self._bind_response(out, receipt)
            raise
        finally:
            receipt["wall_s"] = time.perf_counter()-began
            for k in ("upload", "solve", "derivative", "host_copy", "current"):
                for field in self.receipt[k]:
                    self.receipt[k][field] += receipt[k][field]
        out["receipt"] = receipt
        self._bind_response(out, receipt)
        return out

    def numpy_decoder(self, rates):
        """Explicit CPU E150 reference replay, without neural integration."""
        return numpy_decoder(self.problem, rates)

    def decode_rates(self, rates, *, host=False):
        """Replay supplied recorded endpoints through the actual GPU decoder.

        Uses the constructor's CUDA stream and the fixed E150 initial state.
        This gate advances no conditional neural steps and computes no tangent.
        """
        p, cp, d = self.problem, self.cp, self.device_data
        r = np.asarray(rates)
        _require(r.shape == (p["samples"], p["neurons"])
                 and r.dtype in (np.float32, np.float64) and np.isfinite(r).all()
                 and np.all(r >= 0), "Finite recorded endpoint rate matrix required")
        _require(np.array_equal(r[0], p["initial_rate"].astype(r.dtype)),
                 "Recorded initial endpoint differs")
        _require(type(host) is bool, "Explicit boolean host flag required")
        receipt = {k: _receipt() for k in ("upload", "solve", "derivative", "host_copy")}
        receipt.update(requested_steps=0, completed_steps=0, derivative_completed_steps=0,
                       completed_samples=0, derivative_completed_samples=0,
                       status="decode_requested", chunk_steps=self.chunk_steps,
                       gpu_bytes_estimate=p["gpu_bytes_estimate"])
        self._start_response(receipt)
        began, out = time.perf_counter(), {}
        active_bucket = None
        n, m = p["neurons"], p["muscles"]
        try:
            with self.device, self.stream:
                out["rate"] = self._upload(r.astype(np.float64), receipt["upload"])
                out["bank"] = cp.full((p["samples"], n, 3), cp.nan, cp.float64)
                for key in ("activation", "original_activation"):
                    out[key] = cp.full((p["samples"], m), cp.nan, cp.float64)
                for first in range(1, p["samples"], self.chunk_steps//20):
                    count = min(self.chunk_steps//20, p["samples"]-first)
                    tick, active_bucket = time.perf_counter(), receipt["solve"]
                    active_bucket["requested_count"] += 1
                    active_bucket["requested_bytes"] += count*(3*n+2*m)*8
                    self.kernels["bank_decode"](((3*n+31)//32,), (32,),
                        (out["rate"], d["bank0"], out["bank"], np.int32(n), np.int32(first), np.int32(count)))
                    self.kernels["activation_decode"](((m+31)//32,), (32,), (
                        out["rate"], out["bank"], *(d[k] for k in ("pool", "membership", "low",
                        "high", "kappa", "fixed", "activation0", "original_activation0")),
                        out["activation"], out["original_activation"], np.int32(n), np.int32(m),
                        np.int32(first), np.int32(count)))
                    self.stream.synchronize()
                    active_bucket["completed_count"] += 1
                    active_bucket["completed_bytes"] += count*(3*n+2*m)*8
                    active_bucket["wall_s"] += time.perf_counter()-tick
                    active_bucket = None
                    receipt["completed_samples"] = first+count
                out["q"] = d["rest"] + .02*(out["activation"]@d["moment"].T)
                out["ltm"] = .2*cp.tanh(out["activation"][:, d["ltm"]]/10)
                self.stream.synchronize()
                if host:
                    copied = {}
                    for k, v in out.items():
                        copied[k] = self.host_copy(v, receipt["host_copy"], label="recorded_"+k)
                    out = copied
                receipt["status"] = "complete"
        except Exception as error:
            if active_bucket is not None:
                active_bucket["wall_s"] += time.perf_counter()-tick
            receipt["status"] = "failed"
            receipt["wall_s"] = time.perf_counter()-began
            error.response_receipt = copy.deepcopy(receipt)
            error.partial_response = out.copy()
            self._bind_response(out, receipt)
            raise
        finally:
            receipt["wall_s"] = time.perf_counter()-began
            for k in ("upload", "solve", "derivative", "host_copy", "current"):
                for field in self.receipt[k]:
                    self.receipt[k][field] += receipt[k][field]
        out["receipt"] = receipt
        self._bind_response(out, receipt)
        return out

    def _product(self, name, result, build, output_bytes, *, host):
        """One bounded product with input ownership, metering and failure data."""
        began = time.perf_counter()
        operation = {"operation": name, "status": "validation", **_receipt(),
                     "input_response_receipt": self._receipt_snapshot(result.get("receipt", {}))}
        outputs, copied = {}, {}
        charged = False
        try:
            _require(result is self._current_response
                     and result.get("receipt") is self.last_receipt,
                     "Products require the engine's current response")
            _require(type(host) is bool, "Explicit boolean host flag required")
            p, cp = self.problem, self.cp
            _require("B" in result and isinstance(result["B"], cp.ndarray)
                     and result["B"].shape == (p["samples"], p["params"])
                     and self.last_receipt.get("status") == "complete",
                     "Completed device derivative result required")
            _require(p["gpu_bytes_estimate"] + 2*(self._product_bytes+output_bytes) <= MAX_GPU_BYTES,
                     "Retained current-response products exceed 4 GiB engine limit")
            operation.update(status="requested", requested_count=1, requested_bytes=output_bytes)
            charged = True
            with self.device, self.stream:
                build(outputs)
                self._register_arrays(outputs, product=True)
                self.stream.synchronize()
                operation.update(status="device_complete", completed_count=1,
                                 completed_bytes=output_bytes)
                if host:
                    for key, value in outputs.items():
                        copied[key] = self.host_copy(value, label=name+":"+key)
            operation["status"] = "complete"
            return copied if host else outputs
        except Exception as error:
            self._register_arrays(outputs, product=True)
            operation["status"] = "failed"
            operation["wall_s"] = time.perf_counter()-began
            if result is self._current_response:
                operation["input_response_receipt"] = self._receipt_snapshot(self.last_receipt)
            error.response_receipt = copy.deepcopy(operation)
            error.partial_response = {"input_"+k: v for k, v in result.items()
                                      if isinstance(v, self.cp.ndarray)}
            error.partial_response.update(outputs)
            error.partial_host_response = copied.copy()
            raise
        finally:
            operation["wall_s"] = time.perf_counter()-began
            if charged:
                for bucket in (self.last_receipt["products"][name], self.receipt["products"][name]):
                    for field in _receipt():
                        bucket[field] += operation[field]
                self.last_receipt.setdefault("product_events", []).append(copy.deepcopy(operation))

    def activation_cotangent_product(self, result, cotangent, *, host=True):
        p, cp = self.problem, self.cp
        cot = np.asarray(cotangent)
        def build(outputs):
            _require(cot.dtype == np.float64 and cot.shape == (p["samples"], p["muscles"])
                     and np.isfinite(cot).all(), "Finite float64 activation cotangent required")
            dc = self._upload(cot)
            outputs["gradient"] = cp.sum(result["B"]*dc[:, self.device_data["parameter_muscle"]], axis=0)
        return self._product("cotangent", result, build, p["params"]*8, host=host)["gradient"]

    def command_jacobian(self, result, samples, *, host=True):
        p, cp, d = self.problem, self.cp, self.device_data
        selected = np.asarray(samples)
        output_bytes = 8*selected.size*p["params"]*(p["commands"]+p["ltm"].size)
        def build(outputs):
            idx = _sample_indices(selected, p["samples"])
            jacobian_bytes = 8*idx.size*p["params"]*(p["commands"]+p["ltm"].size+2)
            _require(p["gpu_bytes_estimate"] + 2*jacobian_bytes <= MAX_GPU_BYTES,
                     "Selected command Jacobian exceeds 4 GiB engine limit")
            di = self._upload(idx)
            b, muscle = result["B"][di], d["parameter_muscle"]
            outputs["q"] = .02*d["moment"][:, muscle][None, :, :]*b[:, None, :]
            factors = .02*(1-cp.tanh(result["activation"][di][:, d["ltm"]]/10)**2)
            outputs["ltm"] = factors[:, :, None]*(d["ltm"][:, None] == muscle[None, :])[None, :, :]*b[:, None, :]
        return self._product("command_jacobian", result, build, output_bytes, host=host)
