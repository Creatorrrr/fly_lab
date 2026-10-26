"""E217 muscle-shared 64-anchor conditional neural response basis.

The anchor selection is offline. Each neuron's runtime gate and RBF reads only
its own signed input and own fast/slow memories; coefficients are shared solely
through the original anatomical muscle owner.
"""

from __future__ import annotations

import time

import numpy as np

from tools.banc_m3_causal_neural_basis import (
    DT,
    HISTORY_TAU,
    _x,
    normalized_basis,
    reaction,
)
from tools.banc_m3_multiscale_neural_basis import validate_inverse_slow

BASIS_COUNT = 64
PROBE = (np.arange(BASIS_COUNT, dtype=np.float64) + 1) / 65.


def select_grouped_prototypes(descriptor, membership, pool, cap, *, count=64):
    """Weighted group FPS with row0 and first-unselected ties, own MN centers."""
    descriptor = np.asarray(descriptor, np.float64)
    membership = np.asarray(membership, np.int64)
    pool = np.asarray(pool, np.float64)
    cap = np.asarray(cap, np.float64)
    if (descriptor.ndim != 3 or descriptor.shape[2] != 9
            or type(count) is not int or count < 1 or len(descriptor) < count):
        raise ValueError("E217 finite endpoint descriptor and unique-anchor count required")
    samples, motors, _ = descriptor.shape
    groups = pool.shape[0]
    if (membership.shape != (motors,) or pool.shape != (groups, motors)
            or cap.shape != (motors,) or groups < 1
            or np.any((membership < 0) | (membership >= groups))
            or np.any(pool < 0) or np.any(cap <= 0)
            or not all(np.isfinite(a).all() for a in (descriptor, pool, cap))):
        raise ValueError("E217 finite original muscle/MN ownership required")
    for m, owner in enumerate(membership):
        if np.any(pool[np.arange(groups) != owner, m] != 0):
            raise ValueError("E217 cross-muscle pool ownership")
    owned = pool[membership, np.arange(motors)] * cap
    group_cap = np.bincount(membership, weights=owned, minlength=groups)
    if np.any(group_cap <= 0):
        raise ValueError("E217 each muscle requires positive pooled rate capacity")
    weights = owned / group_cap[membership]
    anchor_rows = np.empty((groups, count), np.int64)
    for group in range(groups):
        cols = np.flatnonzero(membership == group)
        if len(cols) == 0:
            raise ValueError("E217 empty original muscle")
        values = descriptor[:, cols, :] * np.sqrt(weights[cols])[None, :, None]
        nearest = np.sum((values - values[0]) ** 2, axis=(1, 2))
        chosen = np.zeros(samples, bool)
        anchor_rows[group, 0] = 0
        chosen[0] = True
        for j in range(1, count):
            candidate = np.where(chosen, -np.inf, nearest)
            row = int(np.argmax(candidate))
            anchor_rows[group, j] = row
            chosen[row] = True
            distance = np.sum((values - values[row]) ** 2, axis=(1, 2))
            nearest = np.minimum(nearest, distance)
    centers = descriptor[anchor_rows[membership], np.arange(motors)[:, None], :]
    sigma2 = np.empty(motors, np.float64)
    for m in range(motors):
        own = centers[m]
        squared = np.sum((own[:, None, :] - own[None, :, :]) ** 2, axis=-1)
        squared[np.diag_indices(count)] = np.inf
        nearest_other = np.min(squared, axis=1)
        usable = nearest_other[(nearest_other > 0) & np.isfinite(nearest_other)]
        sigma2[m] = max(1e-4, float(np.median(usable))) if len(usable) else 1.
    return {"anchor_rows": anchor_rows, "centers": centers,
        "sigma2": sigma2, "owner_weights": weights,
        "group_cap": group_cap, "membership": membership}


def simulate_grouped_basis_numpy(positive, negative, drive, scale, inverse_slow,
                                 centers, sigma2, initial_rate, tau, cap,
                                 *, sample_every=20):
    """Small generated float64 reference with explicit 64-column RK stages."""
    arrays = [np.asarray(a) for a in (positive, negative, drive)]
    if (any(a.shape != arrays[0].shape for a in arrays)
            or arrays[0].ndim != 2 or len(arrays[0]) == 0
            or type(sample_every) is not int or sample_every < 1
            or len(arrays[0]) % (4 * sample_every)):
        raise ValueError("E217 aligned complete four-stage inputs required")
    n = arrays[0].shape[1]
    inverse = validate_inverse_slow(inverse_slow, n)
    scale = np.asarray(scale, np.float64)
    centers = np.asarray(centers, np.float64)
    sigma2 = np.asarray(sigma2, np.float64)
    initial = np.asarray(initial_rate, np.float64)
    tau = np.asarray(tau, np.float64)
    cap = np.asarray(cap, np.float64)
    if (scale.shape != (n, 3) or centers.shape != (n, BASIS_COUNT, 9)
            or sigma2.shape != (n,) or np.any(sigma2 <= 0)
            or any(v.shape != (n,) for v in (initial, tau, cap))
            or np.any(tau <= 0) or np.any(cap <= 0)
            or not all(np.isfinite(v).all() for v in (
                scale, centers, sigma2, initial, tau, cap))):
        raise ValueError("E217 own 64-center original physiology required")
    x = _x(*arrays, scale)
    samples = len(x) // (4 * sample_every) + 1
    fast = np.zeros((n, 3), np.float64)
    slow = np.zeros((n, 3), np.float64)
    b = np.zeros((n, BASIS_COUNT), np.float64)
    H = initial.copy()
    factor = np.ones(n, np.float64)
    probe = initial.copy()
    output = {"h_fast": np.empty((samples, n, 3), np.float64),
        "h_slow": np.empty((samples, n, 3), np.float64),
        "descriptor": np.empty((samples, n, 9), np.float64),
        "B": np.empty((samples, n, BASIS_COUNT), np.float64),
        "H": np.empty((samples, n), np.float64),
        "H_factor": np.empty((samples, n), np.float64),
        "probe_rate": np.empty((samples, n), np.float64)}

    def store(row, last_x):
        output["h_fast"][row] = fast
        output["h_slow"][row] = slow
        output["descriptor"][row] = np.concatenate((last_x, fast, slow), axis=1)
        output["B"][row] = b
        output["H"][row] = H
        output["H_factor"][row] = factor
        output["probe_rate"][row] = probe

    store(0, x[0])
    for tick in range(len(x) // 4):
        xs = x[4*tick:4*tick+4]
        slopes = []
        for stage, step in enumerate((0., .5*DT, .5*DT, DT)):
            if stage == 0:
                fs, ss, bs, hr, ff, rr = fast, slow, b, H, factor, probe
            else:
                prior = slopes[stage-1]
                fs = fast + step*prior[0]
                ss = slow + step*prior[1]
                bs = b + step*prior[2]
                hr = H + step*prior[3]
                ff = factor + step*prior[4]
                rr = probe + step*prior[5]
            descriptor = np.concatenate((xs[stage], fs, ss), axis=1)
            weights = normalized_basis(descriptor, centers, sigma2[:, None])
            target = reaction(xs[stage], fs)[:, None] * weights
            slopes.append(((xs[stage] - fs) / HISTORY_TAU,
                (xs[stage] - ss) * inverse[:, None],
                (target - bs) / tau[:, None], -hr / tau, -ff / tau,
                (cap * np.sum(target * PROBE, axis=1) - rr) / tau))
        fast, slow, b, H, factor, probe = (
            current + DT / 6 * (slopes[0][i] + 2*slopes[1][i]
                                + 2*slopes[2][i] + slopes[3][i])
            for i, current in enumerate((fast, slow, b, H, factor, probe)))
        if (tick + 1) % sample_every == 0:
            store((tick + 1)//sample_every, xs[3])
    return output


CUDA_SOURCE = r"""
__device__ __forceinline__ void input_x(const float* e,const float* i,const float* d,
    const double* scale,int row,int m,int n,double* x) {
    int index=row*n+m;
    x[0]=log1p((double)e[index])/scale[3*m];
    x[1]=log1p((double)i[index])/scale[3*m+1];
    x[2]=asinh((double)d[index])/scale[3*m+2];
}
__device__ __forceinline__ void grouped_target(const double* x,const double* fast,
    const double* slow,const double* center,double sigma,double* target,double* weighted) {
    double logits[64],largest=-1.0e300;
    for(int j=0;j<64;j++) {
        double squared=0.0;
        for(int k=0;k<3;k++) {
            double a=x[k]-center[9*j+k],b=fast[k]-center[9*j+k+3];
            double c=slow[k]-center[9*j+k+6];
            squared+=a*a+b*b+c*c;
        }
        logits[j]=-squared/(2.0*sigma);
        if(logits[j]>largest) largest=logits[j];
    }
    double sum=0.0;
    for(int j=0;j<64;j++) {target[j]=exp(logits[j]-largest);sum+=target[j];}
    double gate=-expm1(-8.0*(x[0]+fabs(x[2])+fmax(fast[0],0.0)+fabs(fast[2])));
    *weighted=0.0;
    for(int j=0;j<64;j++) {
        target[j]=gate*target[j]/sum;
        *weighted+=((double)(j+1)/65.0)*target[j];
    }
}
extern "C" __global__ void grouped_basis_kernel(
    const float* e,const float* i,const float* d,
    const double* scale,const double* inverse,const double* center,const double* sigma,
    const double* initial,const double* tau,const double* cap,
    double* fast_out,double* slow_out,double* descriptor,
    double* B,double* H,double* F,double* probe,int n,int ticks,int every) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=n) return;
    double fast[3]={0.,0.,0.},slow[3]={0.,0.,0.},b[64]={0.};
    double homogeneous=initial[m],factor=1.,r=initial[m];
    double x[4][3],kf[4][3],ks[4][3],kb[4][64],kH[4],kF[4],kr[4];
    double target[64],weighted;
    const double* local_center=center+576*m;
    input_x(e,i,d,scale,0,m,n,x[0]);
    for(int a=0;a<3;a++) {
        fast_out[m*3+a]=0.;slow_out[m*3+a]=0.;
        descriptor[m*9+a]=x[0][a];
        descriptor[m*9+a+3]=0.;descriptor[m*9+a+6]=0.;
    }
    for(int j=0;j<64;j++) B[m*64+j]=0.;
    H[m]=homogeneous;F[m]=factor;probe[m]=r;
    for(int tick=0;tick<ticks;tick++) {
        for(int stage=0;stage<4;stage++)
            input_x(e,i,d,scale,4*tick+stage,m,n,x[stage]);
        for(int stage=0;stage<4;stage++) {
            double step=(stage==0?0.:(stage==3?.00025:.000125));
            double fs[3],ss[3],bs[64],hr=homogeneous,ff=factor,rr=r;
            for(int a=0;a<3;a++) {
                fs[a]=fast[a]+(stage==0?0.:step*kf[stage-1][a]);
                ss[a]=slow[a]+(stage==0?0.:step*ks[stage-1][a]);
            }
            if(stage>0) {
                hr+=step*kH[stage-1];ff+=step*kF[stage-1];rr+=step*kr[stage-1];
            }
            for(int j=0;j<64;j++) bs[j]=b[j]+(stage==0?0.:step*kb[stage-1][j]);
            grouped_target(x[stage],fs,ss,local_center,sigma[m],target,&weighted);
            for(int a=0;a<3;a++) {
                kf[stage][a]=(x[stage][a]-fs[a])/.016;
                ks[stage][a]=(x[stage][a]-ss[a])*inverse[m];
            }
            for(int j=0;j<64;j++) kb[stage][j]=(target[j]-bs[j])/tau[m];
            kH[stage]=-hr/tau[m];kF[stage]=-ff/tau[m];
            kr[stage]=(cap[m]*weighted-rr)/tau[m];
        }
        for(int a=0;a<3;a++) {
            fast[a]+=.00025/6.*(kf[0][a]+2*kf[1][a]+2*kf[2][a]+kf[3][a]);
            slow[a]+=.00025/6.*(ks[0][a]+2*ks[1][a]+2*ks[2][a]+ks[3][a]);
        }
        for(int j=0;j<64;j++)
            b[j]+=.00025/6.*(kb[0][j]+2*kb[1][j]+2*kb[2][j]+kb[3][j]);
        homogeneous+=.00025/6.*(kH[0]+2*kH[1]+2*kH[2]+kH[3]);
        factor+=.00025/6.*(kF[0]+2*kF[1]+2*kF[2]+kF[3]);
        r+=.00025/6.*(kr[0]+2*kr[1]+2*kr[2]+kr[3]);
        if((tick+1)%every==0) {
            int row=(tick+1)/every,offset=row*n+m;
            for(int a=0;a<3;a++) {
                fast_out[offset*3+a]=fast[a];slow_out[offset*3+a]=slow[a];
                descriptor[offset*9+a]=x[3][a];
                descriptor[offset*9+a+3]=fast[a];
                descriptor[offset*9+a+6]=slow[a];
            }
            for(int j=0;j<64;j++) B[offset*64+j]=b[j];
            H[offset]=homogeneous;F[offset]=factor;probe[offset]=r;
        }
    }
}
"""


def planned_gpu_bytes(shape, *, sample_every=20):
    stages, n = shape
    if (type(sample_every) is not int or sample_every < 1 or stages < 1
            or n < 1 or stages % (4*sample_every)):
        raise ValueError("E217 complete grouped conditional dimensions required")
    samples = stages // (4*sample_every) + 1
    channels = 3*stages*n*4
    outputs = samples*n*(9+64+3+3+3)*8
    parameters = n*(3+1+576+1+3)*8
    return {"channel_bytes": channels, "output_bytes": outputs,
        "parameter_bytes": parameters,
        "peak_bytes": channels+outputs+parameters}


class PersistentGroupedInputs:
    """Three channel uploads, one grouped-basis launch, partial raw recovery."""

    def __init__(self, positive, negative, drive, *, gpu_limit=2147483648,
                 sample_every=20):
        import cupy as cp

        if (type(sample_every) is not int or sample_every < 1
                or positive.shape != negative.shape or positive.shape != drive.shape
                or positive.ndim != 2 or positive.shape[0] == 0
                or positive.shape[0] % (4*sample_every)
                or any(a.dtype != np.float32 for a in (
                    positive, negative, drive))):
            raise ValueError("E217 aligned native signed channels required")
        self.cp = cp
        self.n = positive.shape[1]
        self.ticks = positive.shape[0] // 4
        self.sample_every = sample_every
        self.samples = self.ticks // sample_every + 1
        self.gpu_limit = gpu_limit
        plan = planned_gpu_bytes(positive.shape, sample_every=sample_every)
        self.upload_bytes = plan["channel_bytes"]
        self.planned_output_bytes = plan["output_bytes"]
        self.planned_parameter_bytes = plan["parameter_bytes"]
        self.planned_peak_bytes = plan["peak_bytes"]
        if self.planned_peak_bytes > gpu_limit:
            raise MemoryError("E217 planned GPU allocation exceeds 2 GiB")
        pool = cp.get_default_memory_pool()
        self.baseline_pool_bytes = pool.total_bytes()
        free, total = cp.cuda.runtime.memGetInfo()
        self.baseline_device_used_bytes = total - free
        self.peak_owned_pool_bytes = 0
        self.peak_device_used_bytes = self.baseline_device_used_bytes
        self.transfer_count = 0
        self.parameter_transfer_count = 0
        self.parameter_upload_bytes = 0
        self.e = cp.asarray(np.ascontiguousarray(positive))
        self.transfer_count += 1
        self.i = cp.asarray(np.ascontiguousarray(negative))
        self.transfer_count += 1
        self.d = cp.asarray(np.ascontiguousarray(drive))
        self.transfer_count += 1
        self._measure()
        self.kernel = cp.RawKernel(CUDA_SOURCE, "grouped_basis_kernel",
                                   options=("--fmad=false",))
        self.calls_requested = 0
        self.calls_completed = 0
        self.launch_events = []
        self.current_output = None
        self.partial_host = {}

    def _measure(self):
        pool = self.cp.get_default_memory_pool()
        self.peak_owned_pool_bytes = max(
            self.peak_owned_pool_bytes, pool.total_bytes() - self.baseline_pool_bytes)
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.peak_device_used_bytes = max(self.peak_device_used_bytes, total - free)
        if self.peak_owned_pool_bytes > self.gpu_limit:
            raise MemoryError("E217 measured CuPy pool exceeds 2 GiB")

    def partial_readback(self):
        if self.current_output is None:
            return None
        values = dict(self.partial_host)
        errors = []
        for key, value in self.current_output.items():
            if key in values:
                continue
            try:
                values[key] = self.cp.asnumpy(value)
            except BaseException as error:  # noqa: BLE001 - preserve siblings
                errors.append({"array": key, "type": type(error).__name__,
                               "message": str(error)})
        return values, errors


def run_grouped_basis_cuda(context, scale, inverse_slow, prototypes,
                           initial_rate, tau, cap, *, sample_every=20):
    """One explicit 64-column kernel with raw host readback per array."""
    if context.calls_requested or sample_every != context.sample_every:
        raise RuntimeError("E217 one basis CUDA launch or frozen sample interval changed")
    n, ticks, cp = context.n, context.ticks, context.cp
    count = ticks // sample_every + 1
    scale = np.asarray(scale, np.float64)
    inverse = validate_inverse_slow(inverse_slow, n)
    centers = np.asarray(prototypes["centers"], np.float64)
    sigma2 = np.asarray(prototypes["sigma2"], np.float64)
    initial = np.asarray(initial_rate, np.float64)
    tau = np.asarray(tau, np.float64)
    cap = np.asarray(cap, np.float64)
    if (scale.shape != (n, 3) or centers.shape != (n, 64, 9)
            or sigma2.shape != (n,) or np.any(sigma2 <= 0)
            or any(v.shape != (n,) for v in (initial, tau, cap))
            or np.any(tau <= 0) or np.any(cap <= 0)
            or not all(np.isfinite(v).all() for v in (
                scale, inverse, centers, sigma2, initial, tau, cap))):
        raise ValueError("E217 exact grouped GPU parameters required")
    output = {"h_fast": cp.empty((count, n, 3), cp.float64),
              "h_slow": cp.empty((count, n, 3), cp.float64),
              "descriptor": cp.empty((count, n, 9), cp.float64),
              "B": cp.empty((count, n, 64), cp.float64),
              "H": cp.empty((count, n), cp.float64),
              "H_factor": cp.empty((count, n), cp.float64),
              "probe_rate": cp.empty((count, n), cp.float64)}
    parameters = [cp.asarray(np.ascontiguousarray(v), dtype=cp.float64)
        for v in (scale, inverse, centers, sigma2, initial, tau, cap)]
    context.parameter_transfer_count += 7
    context.parameter_upload_bytes += sum(v.nbytes for v in parameters)
    context._measure()
    context.calls_requested = 1
    context.current_output = output
    context.partial_host = {}
    event = {"name": "grouped_basis", "requested": True,
        "synchronized": False, "readback_complete": False,
        "readback_keys_completed": [], "kernel_wall_s": None}
    context.launch_events.append(event)
    began = time.perf_counter()
    context.kernel(((n+127)//128,), (128,),
        (context.e, context.i, context.d, *parameters,
         *(output[key] for key in (
             "h_fast", "h_slow", "descriptor", "B", "H", "H_factor",
             "probe_rate")), np.int32(n), np.int32(ticks), np.int32(sample_every)))
    cp.cuda.get_current_stream().synchronize()
    context.calls_completed = 1
    event["synchronized"] = True
    event["kernel_wall_s"] = time.perf_counter() - began
    context._measure()
    host = {}
    for key, value in output.items():
        host[key] = cp.asnumpy(value)
        context.partial_host[key] = host[key]
        event["readback_keys_completed"].append(key)
    event["readback_complete"] = True
    context.current_output = None
    context.partial_host = {}
    return host, time.perf_counter() - began


def group_affine_readout(basis, cap, membership, pool, moment, rest, ltm,
                         groups, free, fixed, original_rates, old_affine):
    """Pool 64 basis columns by muscle; reconstruct canonical probe directly."""
    B = np.asarray(basis["B"], np.float64)
    H = np.asarray(basis["H"], np.float64)
    cap = np.asarray(cap, np.float64)
    membership = np.asarray(membership, np.int64)
    pool = np.asarray(pool, np.float64)
    moment = np.asarray(moment, np.float64)
    rest = np.asarray(rest, np.float64)
    ltm = np.asarray(ltm, np.int64)
    free = np.asarray(free, np.int64)
    fixed = np.asarray(fixed, np.int64)
    original_rates = np.asarray(original_rates, np.float64)
    samples, n, width = B.shape
    intercept = np.asarray(old_affine["rate_intercept"], np.float64)
    q0 = np.asarray(old_affine["q_intercept"], np.float64)
    tendon0 = np.asarray(old_affine["tendon_intercept"], np.float64)
    active0 = np.asarray(old_affine["activation_intercept"], np.float64)
    names = [groups[int(muscle)]["leg"] + "_tarsus" for muscle in ltm]
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    if len(groups) != 96 or set(names) != {leg + "_tarsus" for leg in leg_order}:
        raise ValueError("E217 original six LTM anatomical tendon names changed")
    tendon_order = [names.index(leg + "_tarsus") for leg in leg_order]
    if (n != 391 or width != 64 or cap.shape != (n,)
            or membership.shape != (n,) or pool.shape != (96, n)
            or moment.shape != (42, 96) or rest.shape != (42,)
            or ltm.shape != (6,) or free.shape != (343,)
            or fixed.shape != (48,) or intercept.shape != (samples, n)
            or q0.shape != (samples, 42) or tendon0.shape != (samples, 6)
            or active0.shape != (samples, 96)
            or original_rates.shape != (samples, n)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(n))
            or not np.array_equal(intercept[:, fixed], original_rates[:, fixed])
            or not np.allclose(intercept[:, free], H[:, free], atol=2e-12, rtol=2e-12)
            or not all(np.isfinite(v).all() for v in (
                B, H, cap, pool, moment, rest, intercept, q0, active0))):
        raise ValueError("E217 original E213 readout/anatomy contract changed")
    for m, owner in enumerate(membership):
        nonzero = np.flatnonzero(pool[:, m])
        if len(nonzero) != 1 or nonzero[0] != owner:
            raise ValueError("E217 canonical one-muscle owner changed")
    d_group = np.zeros((samples, 96, 64), np.float64)
    filtered = np.zeros((n, 64), np.float64)
    alpha = -np.expm1(-.005 / .01)
    owned_weight = pool[membership, np.arange(n)] * cap
    owned_weight[fixed] = 0.
    for t in range(1, samples):
        filtered += alpha * (B[t] - filtered)
        np.add.at(d_group[t], membership, filtered * owned_weight[:, None])
    delta_muscle = np.sum(d_group * PROBE[None, None, :], axis=2)
    q_probe_affine = q0 + .02 * (delta_muscle @ moment.T)
    q_probe_affine[0] = q0[0]
    rate = intercept + cap[None] * np.sum(B * PROBE[None, None, :], axis=2)
    rate[:, fixed] = original_rates[:, fixed]
    active = active0[0].copy()
    q_probe_direct = np.empty_like(q0)
    activation_probe = np.empty_like(active0)
    tendon_probe = np.empty_like(tendon0)
    q_probe_direct[0] = q0[0]
    activation_probe[0] = active
    tendon_probe[0] = tendon0[0]
    for t in range(1, samples):
        active += alpha * (pool @ rate[t] - active)
        activation_probe[t] = active
        q_probe_direct[t] = rest + .02 * (moment @ active)
        tendon_probe[t] = .2 * np.tanh(active[ltm][tendon_order] / 10)
    return {"D_group": d_group, "q_intercept": q0,
        "tendon_intercept": tendon0, "rate_intercept": intercept,
        "activation_intercept": active0,
        "q_probe_affine": q_probe_affine,
        "q_probe_direct": q_probe_direct,
        "probe_rate_conditional": rate,
        "activation_probe": activation_probe,
        "tendon_probe": tendon_probe}
