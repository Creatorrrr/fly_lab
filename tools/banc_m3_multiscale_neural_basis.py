"""E213 conditional two-timescale history and positive response basis.

Recorded signed RK stages drive both kernels. Neither output feeds the BANC graph.
"""

from __future__ import annotations

import time

import numpy as np

from tools.banc_m3_causal_neural_basis import (
    BASIS_COUNT,
    DT,
    HISTORY_TAU,
    _x,
    compute_scales,
    normalized_basis,
    reaction,
)


def validate_inverse_slow(value, n):
    inverse = np.asarray(value)
    if (inverse.shape != (n,) or inverse.dtype != np.float64
            or not np.isfinite(inverse).all() or np.any(inverse <= 0)):
        raise ValueError("E213 actual positive float32-cast slow inverse required")
    return inverse


def simulate_history_numpy(positive, negative, drive, scale, inverse_slow,
                           *, sample_every=20):
    arrays = [np.asarray(a) for a in (positive, negative, drive)]
    if (any(a.shape != arrays[0].shape for a in arrays)
            or arrays[0].ndim != 2 or len(arrays[0]) == 0
            or len(arrays[0]) % (4 * sample_every)):
        raise ValueError("E213 aligned complete four-stage intervals required")
    n = arrays[0].shape[1]
    if np.asarray(scale).shape != (n, 3):
        raise ValueError("E213 frozen E209 scale shape required")
    inverse = validate_inverse_slow(inverse_slow, n)
    x = _x(*arrays, np.asarray(scale))
    count = len(x) // (4 * sample_every) + 1
    fast = np.zeros((n, 3), np.float64)
    slow = np.zeros((n, 3), np.float64)
    output = {"h_fast": np.empty((count, n, 3), np.float64),
              "h_slow": np.empty((count, n, 3), np.float64),
              "descriptor": np.empty((count, n, 9), np.float64)}

    def store(row, last_x):
        output["h_fast"][row] = fast
        output["h_slow"][row] = slow
        output["descriptor"][row] = np.concatenate((last_x, fast, slow), axis=1)

    store(0, x[0])
    for tick in range(len(x) // 4):
        xs = x[4*tick:4*tick+4]
        kf = []
        ks = []
        for stage, step in enumerate((0., .5*DT, .5*DT, DT)):
            fs = fast if stage == 0 else fast + step*kf[stage-1]
            ss = slow if stage == 0 else slow + step*ks[stage-1]
            kf.append((xs[stage] - fs) / HISTORY_TAU)
            ks.append((xs[stage] - ss) * inverse[:, None])
        fast = fast + DT / 6 * (kf[0] + 2*kf[1] + 2*kf[2] + kf[3])
        slow = slow + DT / 6 * (ks[0] + 2*ks[1] + 2*ks[2] + ks[3])
        if (tick + 1) % sample_every == 0:
            store((tick + 1) // sample_every, xs[3])
    return output


def select_prototypes_9d(descriptor, *, count=BASIS_COUNT):
    """E209 row-zero-first FPS and first-tie rule, now in nine dimensions."""
    descriptor = np.asarray(descriptor, np.float64)
    if (descriptor.ndim != 3 or descriptor.shape[2] != 9
            or len(descriptor) < 1 or count < 1
            or not np.isfinite(descriptor).all()):
        raise ValueError("E213 finite endpoint descriptors [sample,motor,9] required")
    _, n, _ = descriptor.shape
    rows = np.zeros((n, count), np.int64)
    centers = np.empty((n, count, 9), np.float64)
    sigma2 = np.empty(n, np.float64)
    for m in range(n):
        d = descriptor[:, m, :]
        centers[m, 0] = d[0]
        nearest = np.sum((d - d[0])**2, axis=1)
        for j in range(1, count):
            row = int(np.argmax(nearest)) if np.max(nearest) > 0 else 0
            rows[m, j] = row
            centers[m, j] = d[row]
            nearest = np.minimum(nearest, np.sum((d - d[row])**2, axis=1))
        pairwise = np.sum((centers[m, :, None] - centers[m, None, :])**2, axis=-1)
        pairwise[np.diag_indices(count)] = np.inf
        closest = np.empty(count, np.float64)
        for j in range(count):
            positive = pairwise[j, (pairwise[j] > 0) & np.isfinite(pairwise[j])]
            closest[j] = np.min(positive) if len(positive) else np.nan
        usable = closest[np.isfinite(closest)]
        sigma2[m] = max(1e-4, float(np.median(usable))) if len(usable) else 1.
    return {"centers": centers, "sigma2": sigma2, "rows": rows}


def simulate_basis_numpy(positive, negative, drive, scale, inverse_slow,
                         centers, sigma2, initial_rate, tau, cap,
                         *, sample_every=20):
    arrays = [np.asarray(a) for a in (positive, negative, drive)]
    if (any(a.shape != arrays[0].shape for a in arrays)
            or arrays[0].ndim != 2 or len(arrays[0]) == 0
            or len(arrays[0]) % (4 * sample_every)):
        raise ValueError("E213 aligned complete four-stage inputs required")
    n = arrays[0].shape[1]
    inverse = validate_inverse_slow(inverse_slow, n)
    if (np.asarray(centers).shape != (n, BASIS_COUNT, 9)
            or np.asarray(sigma2).shape != (n,)
            or any(np.asarray(v).shape != (n,) for v in (initial_rate, tau, cap))):
        raise ValueError("E213 per-MN original physiology and nine-D basis required")
    x = _x(*arrays, np.asarray(scale))
    count = len(x) // (4*sample_every) + 1
    fast = np.zeros((n, 3), np.float64)
    slow = np.zeros((n, 3), np.float64)
    b = np.zeros((n, BASIS_COUNT), np.float64)
    H = np.asarray(initial_rate, np.float64).copy()
    factor = np.ones(n, np.float64)
    probe = H.copy()
    beta = (np.arange(BASIS_COUNT) + 1) / 17
    tau = np.asarray(tau, np.float64)
    cap = np.asarray(cap, np.float64)
    centers = np.asarray(centers, np.float64)
    sigma2 = np.asarray(sigma2, np.float64)
    output = {"h_fast": np.empty((count, n, 3), np.float64),
              "h_slow": np.empty((count, n, 3), np.float64),
              "descriptor": np.empty((count, n, 9), np.float64),
              "B": np.empty((count, n, BASIS_COUNT), np.float64),
              "H": np.empty((count, n), np.float64),
              "H_factor": np.empty((count, n), np.float64),
              "probe_rate": np.empty((count, n), np.float64)}

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
                (cap * np.sum(target * beta, axis=1) - rr) / tau))
        fast, slow, b, H, factor, probe = (
            current + DT / 6 * (slopes[0][i] + 2*slopes[1][i]
                                + 2*slopes[2][i] + slopes[3][i])
            for i, current in enumerate((fast, slow, b, H, factor, probe)))
        if (tick + 1) % sample_every == 0:
            store((tick + 1) // sample_every, xs[3])
    return output


CUDA_SOURCE = r"""
__device__ __forceinline__ void input_x(const float* e,const float* i,const float* d,
    const double* scale,int row,int m,int n,double* x) {
    int index=row*n+m;
    x[0]=log1p((double)e[index])/scale[3*m];
    x[1]=log1p((double)i[index])/scale[3*m+1];
    x[2]=asinh((double)d[index])/scale[3*m+2];
}
__device__ __forceinline__ void basis_target(const double* x,const double* fast,
    const double* slow,const double* center,double sigma,double* target,double* weighted) {
    double logits[16],largest=-1.0e300;
    for(int j=0;j<16;j++) {
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
    for(int j=0;j<16;j++) {target[j]=exp(logits[j]-largest);sum+=target[j];}
    double gate=-expm1(-8.0*(x[0]+fabs(x[2])+fmax(fast[0],0.0)+fabs(fast[2])));
    *weighted=0.0;
    for(int j=0;j<16;j++) {
        target[j]=gate*target[j]/sum;
        *weighted+=((double)(j+1)/17.0)*target[j];
    }
}
extern "C" __global__ void history_kernel(const float* e,const float* i,const float* d,
    const double* scale,const double* inverse,double* fast_out,double* slow_out,
    double* descriptor,int n,int ticks,int every) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=n) return;
    double fast[3]={0.,0.,0.},slow[3]={0.,0.,0.},x[4][3],kf[4][3],ks[4][3];
    input_x(e,i,d,scale,0,m,n,x[0]);
    for(int a=0;a<3;a++) {
        fast_out[m*3+a]=0.;slow_out[m*3+a]=0.;
        descriptor[m*9+a]=x[0][a];
        descriptor[m*9+a+3]=0.;descriptor[m*9+a+6]=0.;
    }
    for(int tick=0;tick<ticks;tick++) {
        for(int stage=0;stage<4;stage++)
            input_x(e,i,d,scale,4*tick+stage,m,n,x[stage]);
        for(int stage=0;stage<4;stage++) {
            double step=(stage==0?0.:(stage==3?.00025:.000125));
            for(int a=0;a<3;a++) {
                double fs=fast[a]+(stage==0?0.:step*kf[stage-1][a]);
                double ss=slow[a]+(stage==0?0.:step*ks[stage-1][a]);
                kf[stage][a]=(x[stage][a]-fs)/.016;
                ks[stage][a]=(x[stage][a]-ss)*inverse[m];
            }
        }
        for(int a=0;a<3;a++) {
            fast[a]+=.00025/6.*(kf[0][a]+2*kf[1][a]+2*kf[2][a]+kf[3][a]);
            slow[a]+=.00025/6.*(ks[0][a]+2*ks[1][a]+2*ks[2][a]+ks[3][a]);
        }
        if((tick+1)%every==0) {
            int row=(tick+1)/every,offset=row*n+m;
            for(int a=0;a<3;a++) {
                fast_out[offset*3+a]=fast[a];slow_out[offset*3+a]=slow[a];
                descriptor[offset*9+a]=x[3][a];
                descriptor[offset*9+a+3]=fast[a];
                descriptor[offset*9+a+6]=slow[a];
            }
        }
    }
}
extern "C" __global__ void basis_kernel(const float* e,const float* i,const float* d,
    const double* scale,const double* inverse,const double* center,const double* sigma,
    const double* initial,const double* tau,const double* cap,
    double* fast_out,double* slow_out,double* descriptor,
    double* B,double* H,double* F,double* probe,int n,int ticks,int every) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=n) return;
    double fast[3]={0.,0.,0.},slow[3]={0.,0.,0.},b[16]={0.};
    double homogeneous=initial[m],factor=1.,r=initial[m];
    double x[4][3],kf[4][3],ks[4][3],kb[4][16],kH[4],kF[4],kr[4];
    double target[16],weighted;
    const double* local_center=center+144*m;
    input_x(e,i,d,scale,0,m,n,x[0]);
    for(int a=0;a<3;a++) {
        fast_out[m*3+a]=0.;slow_out[m*3+a]=0.;
        descriptor[m*9+a]=x[0][a];
        descriptor[m*9+a+3]=0.;descriptor[m*9+a+6]=0.;
    }
    for(int j=0;j<16;j++) B[m*16+j]=0.;
    H[m]=homogeneous;F[m]=factor;probe[m]=r;
    for(int tick=0;tick<ticks;tick++) {
        for(int stage=0;stage<4;stage++)
            input_x(e,i,d,scale,4*tick+stage,m,n,x[stage]);
        for(int stage=0;stage<4;stage++) {
            double step=(stage==0?0.:(stage==3?.00025:.000125));
            double fs[3],ss[3],bs[16],hr=homogeneous,ff=factor,rr=r;
            for(int a=0;a<3;a++) {
                fs[a]=fast[a]+(stage==0?0.:step*kf[stage-1][a]);
                ss[a]=slow[a]+(stage==0?0.:step*ks[stage-1][a]);
            }
            if(stage>0) {
                hr+=step*kH[stage-1];ff+=step*kF[stage-1];rr+=step*kr[stage-1];
            }
            for(int j=0;j<16;j++) bs[j]=b[j]+(stage==0?0.:step*kb[stage-1][j]);
            basis_target(x[stage],fs,ss,local_center,sigma[m],target,&weighted);
            for(int a=0;a<3;a++) {
                kf[stage][a]=(x[stage][a]-fs[a])/.016;
                ks[stage][a]=(x[stage][a]-ss[a])*inverse[m];
            }
            for(int j=0;j<16;j++) kb[stage][j]=(target[j]-bs[j])/tau[m];
            kH[stage]=-hr/tau[m];kF[stage]=-ff/tau[m];
            kr[stage]=(cap[m]*weighted-rr)/tau[m];
        }
        for(int a=0;a<3;a++) {
            fast[a]+=.00025/6.*(kf[0][a]+2*kf[1][a]+2*kf[2][a]+kf[3][a]);
            slow[a]+=.00025/6.*(ks[0][a]+2*ks[1][a]+2*ks[2][a]+ks[3][a]);
        }
        for(int j=0;j<16;j++)
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
            for(int j=0;j<16;j++) B[offset*16+j]=b[j];
            H[offset]=homogeneous;F[offset]=factor;probe[offset]=r;
        }
    }
}
"""


def planned_gpu_bytes(shape, *, sample_every=20):
    stages, n = shape
    if (type(sample_every) is not int or sample_every < 1 or n < 1
            or stages < 1 or stages % (4*sample_every)):
        raise ValueError("E213 complete conditional input dimensions required")
    samples = stages // (4*sample_every) + 1
    channel = 3*stages*n*4
    output = samples*n*(15 + 34)*8
    parameter = n*(3+1 + 3+1+144+1+3)*8
    return {"channel_bytes": channel, "output_bytes": output,
            "parameter_bytes": parameter,
            "peak_bytes": channel + output + parameter}


class PersistentMultiscaleInputs:
    """Exactly one upload per signed channel and at most two CUDA launches."""

    def __init__(self, positive, negative, drive, *, gpu_limit=1610612736,
                 sample_every=20):
        import cupy as cp

        if (type(sample_every) is not int or sample_every < 1
                or positive.shape != negative.shape
                or positive.shape != drive.shape
                or positive.ndim != 2 or positive.shape[0] == 0
                or positive.shape[0] % (4*sample_every)
                or any(a.dtype != np.float32 for a in
                       (positive, negative, drive))):
            raise ValueError("E213 aligned native signed channels required")
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
            raise MemoryError("E213 planned GPU buffer budget exceeded")
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
        self.history_kernel = cp.RawKernel(CUDA_SOURCE, "history_kernel",
                                           options=("--fmad=false",))
        self.basis_kernel = cp.RawKernel(CUDA_SOURCE, "basis_kernel",
                                         options=("--fmad=false",))
        self.calls_requested = 0
        self.calls_completed = 0
        self.current_output = None
        self.current_name = None
        self.launch_events = []
        self.partial_host = {}

    def _measure(self):
        pool = self.cp.get_default_memory_pool()
        self.peak_owned_pool_bytes = max(
            self.peak_owned_pool_bytes, pool.total_bytes() - self.baseline_pool_bytes)
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.peak_device_used_bytes = max(self.peak_device_used_bytes, total - free)
        if self.peak_owned_pool_bytes > self.gpu_limit:
            raise MemoryError("E213 measured owned CuPy pool exceeds 1.5 GiB")

    def _launch(self, kernel, args, output, *, name):
        if self.calls_requested >= 2:
            raise RuntimeError("E213 two conditional CUDA launch cap")
        self._measure()
        self.calls_requested += 1
        self.current_output = output
        self.current_name = name
        self.partial_host = {}
        event = {"name": name, "requested": True, "synchronized": False,
                 "readback_complete": False, "readback_keys_completed": [],
                 "kernel_wall_s": None}
        self.launch_events.append(event)
        began = time.perf_counter()
        kernel(((self.n + 127)//128,), (128,), args)
        self.cp.cuda.get_current_stream().synchronize()
        self.calls_completed += 1
        event["synchronized"] = True
        event["kernel_wall_s"] = time.perf_counter() - began
        self._measure()
        result = {}
        for key, value in output.items():
            result[key] = self.cp.asnumpy(value)
            self.partial_host[key] = result[key]
            event["readback_keys_completed"].append(key)
        event["readback_complete"] = True
        self.current_output = None
        self.current_name = None
        self.partial_host = {}
        return result, time.perf_counter() - began

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


def run_history_cuda(context, scale, inverse_slow, *, sample_every=20):
    cp = context.cp
    n, ticks = context.n, context.ticks
    if sample_every != context.sample_every or ticks % sample_every:
        raise ValueError("E213 complete history interval required")
    count = ticks // sample_every + 1
    output = {"h_fast": cp.empty((count, n, 3), cp.float64),
              "h_slow": cp.empty((count, n, 3), cp.float64),
              "descriptor": cp.empty((count, n, 9), cp.float64)}
    device_scale = cp.asarray(scale, dtype=cp.float64)
    device_inverse = cp.asarray(inverse_slow, dtype=cp.float64)
    context.parameter_transfer_count += 2
    context.parameter_upload_bytes += device_scale.nbytes + device_inverse.nbytes
    return context._launch(context.history_kernel,
        (context.e, context.i, context.d, device_scale, device_inverse,
         output["h_fast"], output["h_slow"], output["descriptor"],
         np.int32(n), np.int32(ticks), np.int32(sample_every)),
        output, name="history")


def run_basis_cuda(context, scale, inverse_slow, prototypes,
                   initial_rate, tau, cap, *, sample_every=20):
    cp = context.cp
    n, ticks = context.n, context.ticks
    if sample_every != context.sample_every or ticks % sample_every:
        raise ValueError("E213 complete basis interval required")
    count = ticks // sample_every + 1
    output = {"h_fast": cp.empty((count, n, 3), cp.float64),
              "h_slow": cp.empty((count, n, 3), cp.float64),
              "descriptor": cp.empty((count, n, 9), cp.float64),
              "B": cp.empty((count, n, BASIS_COUNT), cp.float64),
              "H": cp.empty((count, n), cp.float64),
              "H_factor": cp.empty((count, n), cp.float64),
              "probe_rate": cp.empty((count, n), cp.float64)}
    parameters = [cp.asarray(value, dtype=cp.float64) for value in
        (scale, inverse_slow, prototypes["centers"], prototypes["sigma2"],
         initial_rate, tau, cap)]
    context.parameter_transfer_count += len(parameters)
    context.parameter_upload_bytes += sum(value.nbytes for value in parameters)
    return context._launch(context.basis_kernel,
        (context.e, context.i, context.d, *parameters,
         *(output[key] for key in ("h_fast", "h_slow", "descriptor", "B",
                                   "H", "H_factor", "probe_rate")),
         np.int32(n), np.int32(ticks), np.int32(sample_every)),
        output, name="basis")


__all__ = (
    "CUDA_SOURCE",
    "PersistentMultiscaleInputs",
    "compute_scales",
    "normalized_basis",
    "planned_gpu_bytes",
    "reaction",
    "run_basis_cuda",
    "run_history_cuda",
    "select_prototypes_9d",
    "simulate_basis_numpy",
    "simulate_history_numpy",
    "validate_inverse_slow",
)
