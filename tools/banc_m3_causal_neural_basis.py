"""Conditional signed-input history and positive response basis for E209.

All dynamics here consume recorded RK stage inputs. Nothing feeds BANC back.
"""

from __future__ import annotations

import time

import numpy as np

DT = .00025
HISTORY_TAU = .016
BASIS_COUNT = 16
GATE_GAIN = 8.


def compute_scales(positive, negative, drive, *, block=4096):
    """Use native maxima once; never revise scales from later states."""
    if not (positive.shape == negative.shape == drive.shape and
            positive.ndim == 2 and positive.shape[0] > 0 and
            positive.shape[0] % 4 == 0):
        raise ValueError("Aligned four-stage signed inputs required")
    n = positive.shape[1]
    maxima = np.zeros((3, n), np.float64)
    for lo in range(0, len(positive), block):
        pieces = [np.asarray(item[lo:lo + block]) for item in
                  (positive, negative, drive)]
        if any(item.dtype != np.float32 or not np.isfinite(item).all() for item in pieces):
            raise ValueError("Finite native float32 signed inputs required")
        if np.any(pieces[0] < 0) or np.any(pieces[1] < 0):
            raise ValueError("Nonnegative recurrent channel magnitudes required")
        maxima[0] = np.maximum(maxima[0], np.max(pieces[0], axis=0))
        maxima[1] = np.maximum(maxima[1], np.max(pieces[1], axis=0))
        maxima[2] = np.maximum(maxima[2],
            np.max(np.abs(np.arcsinh(pieces[2].astype(np.float64))), axis=0))
    return np.stack((np.maximum(1., np.log1p(maxima[0])),
                     np.maximum(1., np.log1p(maxima[1])),
                     np.maximum(1., maxima[2])), axis=1)


def _x(positive, negative, drive, scale):
    return np.stack((np.log1p(np.asarray(positive, np.float64)) / scale[:, 0],
        np.log1p(np.asarray(negative, np.float64)) / scale[:, 1],
        np.arcsinh(np.asarray(drive, np.float64)) / scale[:, 2]), axis=-1)


def reaction(x, h):
    """The frozen gate uses max(hE,0), without changing the history state."""
    argument = x[..., 0] + np.abs(x[..., 2]) + np.maximum(h[..., 0], 0.) + np.abs(h[..., 2])
    return -np.expm1(-GATE_GAIN * argument)


def normalized_basis(descriptor, centers, sigma2):
    distance2 = np.sum((descriptor[..., None, :] - centers)**2, axis=-1)
    logits = -distance2 / (2 * sigma2)
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return weights / np.sum(weights, axis=-1, keepdims=True)


def simulate_history_numpy(positive, negative, drive, scale, *, sample_every=20):
    """Generated-data float64 RK4 oracle, including row-zero descriptor."""
    positive, negative, drive = (np.asarray(a) for a in (positive, negative, drive))
    if positive.shape != negative.shape or positive.shape != drive.shape or positive.ndim != 2:
        raise ValueError("Aligned native channels required")
    if len(positive) == 0 or len(positive) % (4 * sample_every):
        raise ValueError("Complete control intervals required")
    n = positive.shape[1]
    if np.asarray(scale).shape != (n, 3):
        raise ValueError("Actual per-MN scale shape required")
    x = _x(positive, negative, drive, np.asarray(scale))
    count = len(positive) // (4 * sample_every) + 1
    history = np.empty((count, n, 3), np.float64)
    descriptor = np.empty((count, n, 6), np.float64)
    h = np.zeros((n, 3), np.float64)
    history[0] = h
    descriptor[0] = np.concatenate((x[0], h), axis=1)
    for tick in range(len(positive) // 4):
        x1, x2, x3, x4 = x[4 * tick:4 * tick + 4]
        k1 = (x1 - h) / HISTORY_TAU
        k2 = (x2 - (h + .5 * DT * k1)) / HISTORY_TAU
        k3 = (x3 - (h + .5 * DT * k2)) / HISTORY_TAU
        k4 = (x4 - (h + DT * k3)) / HISTORY_TAU
        h = h + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (tick + 1) % sample_every == 0:
            row = (tick + 1) // sample_every
            history[row] = h
            descriptor[row] = np.concatenate((x4, h), axis=1)
    return {"h": history, "descriptor": descriptor}


def select_prototypes(descriptor, *, count=BASIS_COUNT):
    """Deterministic per-cell row0-first farthest points, first-row ties."""
    descriptor = np.asarray(descriptor, np.float64)
    if descriptor.ndim != 3 or descriptor.shape[2] != 6 or len(descriptor) < 1:
        raise ValueError("Endpoint descriptors [sample,motor,6] required")
    _, n, _ = descriptor.shape
    if not np.isfinite(descriptor).all() or count < 1:
        raise ValueError("Finite descriptors and positive basis count required")
    rows = np.zeros((n, count), np.int64)
    centers = np.empty((n, count, 6), np.float64)
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
        nearest_positive = np.empty(count, np.float64)
        for j in range(count):
            candidates = pairwise[j, (pairwise[j] > 0) & np.isfinite(pairwise[j])]
            nearest_positive[j] = np.min(candidates) if len(candidates) else np.nan
        usable = nearest_positive[np.isfinite(nearest_positive)]
        sigma2[m] = max(1e-4, float(np.median(usable))) if len(usable) else 1.
    return {"centers": centers, "sigma2": sigma2, "rows": rows}


def simulate_basis_numpy(positive, negative, drive, scale, centers, sigma2,
                         initial_rate, tau, cap, *, sample_every=20):
    """Independent generated RK4 oracle for B, homogeneous H, and probe rate."""
    arrays = [np.asarray(a) for a in (positive, negative, drive)]
    n = arrays[0].shape[1]
    if any(a.shape != arrays[0].shape for a in arrays) or len(arrays[0]) % (4 * sample_every):
        raise ValueError("Aligned complete native stage inputs required")
    if (np.asarray(centers).shape != (n, BASIS_COUNT, 6)
            or np.asarray(sigma2).shape != (n,)
            or any(np.asarray(v).shape != (n,) for v in (initial_rate, tau, cap))):
        raise ValueError("Per-MN basis and original physiology required")
    x = _x(*arrays, np.asarray(scale))
    count = len(arrays[0]) // (4 * sample_every) + 1
    h = np.zeros((n, 3), np.float64)
    b = np.zeros((n, BASIS_COUNT), np.float64)
    homogeneous = np.asarray(initial_rate, np.float64).copy()
    factor = np.ones(n, np.float64)
    probe = homogeneous.copy()
    beta = (np.arange(BASIS_COUNT) + 1) / 17
    output = {"h": np.empty((count, n, 3), np.float64),
        "descriptor": np.empty((count, n, 6), np.float64),
        "B": np.empty((count, n, BASIS_COUNT), np.float64),
        "H": np.empty((count, n), np.float64),
        "H_factor": np.empty((count, n), np.float64),
        "probe_rate": np.empty((count, n), np.float64)}

    def store(row, last_x):
        output["h"][row] = h
        output["descriptor"][row] = np.concatenate((last_x, h), axis=1)
        output["B"][row] = b
        output["H"][row] = homogeneous
        output["H_factor"][row] = factor
        output["probe_rate"][row] = probe

    store(0, x[0])
    tau = np.asarray(tau, np.float64)
    cap = np.asarray(cap, np.float64)
    centers = np.asarray(centers, np.float64)
    sigma2 = np.asarray(sigma2, np.float64)
    for tick in range(len(arrays[0]) // 4):
        xs = x[4 * tick:4 * tick + 4]
        ks = []
        for stage, step in enumerate((0., .5 * DT, .5 * DT, DT)):
            if stage == 0:
                hs, bs, hs_rate, fs, rs = h, b, homogeneous, factor, probe
            else:
                prev = ks[stage - 1]
                hs = h + step * prev[0]
                bs = b + step * prev[1]
                hs_rate = homogeneous + step * prev[2]
                fs = factor + step * prev[3]
                rs = probe + step * prev[4]
            weights = normalized_basis(np.concatenate((xs[stage], hs), axis=1),
                centers, sigma2[:, None])
            target = reaction(xs[stage], hs)[:, None] * weights
            dh = (xs[stage] - hs) / HISTORY_TAU
            db = (target - bs) / tau[:, None]
            dH = -hs_rate / tau
            dF = -fs / tau
            dr = (cap * np.sum(target * beta, axis=1) - rs) / tau
            ks.append((dh, db, dH, dF, dr))
        h, b, homogeneous, factor, probe = (
            current + DT / 6 * (ks[0][i] + 2 * ks[1][i] + 2 * ks[2][i] + ks[3][i])
            for i, current in enumerate((h, b, homogeneous, factor, probe)))
        if (tick + 1) % sample_every == 0:
            store((tick + 1) // sample_every, xs[3])
    return output


CUDA_SOURCE = r"""
__device__ __forceinline__ void input_x(const float* e, const float* i,
    const float* d, const double* scale, int row, int m, int n, double* x) {
    int index = row*n+m;
    x[0] = log1p((double)e[index])/scale[3*m];
    x[1] = log1p((double)i[index])/scale[3*m+1];
    x[2] = asinh((double)d[index])/scale[3*m+2];
}
__device__ __forceinline__ void basis_target(const double* x, const double* h,
    const double* center, double sigma, double* target, double* weighted) {
    double logits[16], largest = -1.0e300;
    for (int j=0;j<16;j++) {
        double squared=0.0;
        for (int k=0;k<3;k++) {
            double a=x[k]-center[6*j+k], b=h[k]-center[6*j+k+3];
            squared += a*a+b*b;
        }
        logits[j] = -squared/(2.0*sigma);
        if (logits[j] > largest) largest=logits[j];
    }
    double sum=0.0;
    for (int j=0;j<16;j++) { target[j]=exp(logits[j]-largest); sum+=target[j]; }
    double gate = -expm1(-8.0*(x[0]+fabs(x[2])+fmax(h[0],0.0)+fabs(h[2])));
    *weighted=0.0;
    for (int j=0;j<16;j++) {
        target[j] = gate*target[j]/sum;
        *weighted += ((double)(j+1)/17.0)*target[j];
    }
}
extern "C" __global__ void history_kernel(const float* e,const float* i,const float* d,
    const double* scale,double* history,double* descriptor,int n,int ticks,int every) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    if (m>=n) return;
    double h[3]={0.,0.,0.},x[4][3],k[4][3], hs[3];
    input_x(e,i,d,scale,0,m,n,x[0]);
    for (int a=0;a<3;a++) { history[m*3+a]=0.; descriptor[m*6+a]=x[0][a]; descriptor[m*6+a+3]=0.; }
    for (int tick=0;tick<ticks;tick++) {
        for (int stage=0;stage<4;stage++) input_x(e,i,d,scale,4*tick+stage,m,n,x[stage]);
        for (int stage=0;stage<4;stage++) {
            double step=(stage==0?0.:(stage==3?.00025:.000125));
            for (int a=0;a<3;a++) {
                hs[a]=h[a]+(stage==0?0.:step*k[stage-1][a]);
                k[stage][a]=(x[stage][a]-hs[a])/.016;
            }
        }
        for (int a=0;a<3;a++) h[a]+=.00025/6.*(k[0][a]+2*k[1][a]+2*k[2][a]+k[3][a]);
        if ((tick+1)%every==0) {
            int row=(tick+1)/every, offset=(row*n+m);
            for (int a=0;a<3;a++) { history[offset*3+a]=h[a]; descriptor[offset*6+a]=x[3][a]; descriptor[offset*6+a+3]=h[a]; }
        }
    }
}
extern "C" __global__ void basis_kernel(const float* e,const float* i,const float* d,
    const double* scale,const double* center,const double* sigma,const double* initial,
    const double* tau,const double* cap,double* history,double* descriptor,
    double* B,double* H,double* F,double* probe,int n,int ticks,int every) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    if (m>=n) return;
    double h[3]={0.,0.,0.}, b[16]={0.}, homogeneous=initial[m], factor=1., r=initial[m];
    double x[4][3], kh[4][3], kb[4][16], kH[4],kF[4],kr[4],target[16],weighted;
    const double* local_center=center+96*m;
    input_x(e,i,d,scale,0,m,n,x[0]);
    for (int a=0;a<3;a++) { history[m*3+a]=0.; descriptor[m*6+a]=x[0][a]; descriptor[m*6+a+3]=0.; }
    for (int j=0;j<16;j++) B[m*16+j]=0.;
    H[m]=homogeneous; F[m]=factor; probe[m]=r;
    for (int tick=0;tick<ticks;tick++) {
        for (int stage=0;stage<4;stage++) input_x(e,i,d,scale,4*tick+stage,m,n,x[stage]);
        for (int stage=0;stage<4;stage++) {
            double step=(stage==0?0.:(stage==3?.00025:.000125));
            double hs[3],bs[16], hr=homogeneous, ff=factor, rr=r;
            for (int a=0;a<3;a++) hs[a]=h[a]+(stage==0?0.:step*kh[stage-1][a]);
            if (stage>0) { hr+=step*kH[stage-1]; ff+=step*kF[stage-1]; rr+=step*kr[stage-1]; }
            for (int j=0;j<16;j++) bs[j]=b[j]+(stage==0?0.:step*kb[stage-1][j]);
            basis_target(x[stage],hs,local_center,sigma[m],target,&weighted);
            for (int a=0;a<3;a++) kh[stage][a]=(x[stage][a]-hs[a])/.016;
            for (int j=0;j<16;j++) kb[stage][j]=(target[j]-bs[j])/tau[m];
            kH[stage]=-hr/tau[m]; kF[stage]=-ff/tau[m];
            kr[stage]=(cap[m]*weighted-rr)/tau[m];
        }
        for (int a=0;a<3;a++) h[a]+=.00025/6.*(kh[0][a]+2*kh[1][a]+2*kh[2][a]+kh[3][a]);
        for (int j=0;j<16;j++) b[j]+=.00025/6.*(kb[0][j]+2*kb[1][j]+2*kb[2][j]+kb[3][j]);
        homogeneous+=.00025/6.*(kH[0]+2*kH[1]+2*kH[2]+kH[3]);
        factor+=.00025/6.*(kF[0]+2*kF[1]+2*kF[2]+kF[3]);
        r+=.00025/6.*(kr[0]+2*kr[1]+2*kr[2]+kr[3]);
        if ((tick+1)%every==0) {
            int row=(tick+1)/every, offset=row*n+m;
            for (int a=0;a<3;a++) { history[offset*3+a]=h[a]; descriptor[offset*6+a]=x[3][a]; descriptor[offset*6+a+3]=h[a]; }
            for (int j=0;j<16;j++) B[offset*16+j]=b[j];
            H[offset]=homogeneous; F[offset]=factor; probe[offset]=r;
        }
    }
}
"""


def planned_gpu_bytes(shape, *, sample_every=20):
    """Upper bound for one persistent upload and both resident output sets."""
    stages, n = shape
    if (type(sample_every) is not int or sample_every < 1 or n < 1
            or stages < 1 or stages % (4 * sample_every)):
        raise ValueError("Complete conditional input dimensions required")
    samples = stages // (4 * sample_every) + 1
    channel = 3 * stages * n * 4
    output = samples * n * (9 + 28) * 8
    parameter = n * (6 + 96 + 1 + 3) * 8
    return {"channel_bytes": channel, "output_bytes": output,
        "parameter_bytes": parameter, "peak_bytes": channel + output + parameter}


class PersistentSignedInputs:
    """One E/I/drive upload, shared by exactly two conditional kernel launches."""

    def __init__(self, positive, negative, drive, *, gpu_limit=1610612736,
                 sample_every=20):
        import cupy as cp

        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("Positive complete conditional sampling required")
        if (positive.shape != negative.shape or positive.shape != drive.shape
                or positive.ndim != 2 or positive.shape[0] == 0
                or positive.shape[0] % (4 * sample_every)
                or any(value.dtype != np.float32 for value in
                       (positive, negative, drive))):
            raise ValueError("Aligned native channels required")
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
            raise MemoryError("E209 exact planned GPU buffer budget exceeded")
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
        self.history_kernel = cp.RawKernel(CUDA_SOURCE, "history_kernel", options=("--fmad=false",))
        self.basis_kernel = cp.RawKernel(CUDA_SOURCE, "basis_kernel", options=("--fmad=false",))
        self.calls_requested = 0
        self.calls_completed = 0
        self.current_output = None
        self.current_name = None
        self.launch_events = []
        self.partial_host = {}

    def _measure(self):
        pool = self.cp.get_default_memory_pool()
        self.peak_owned_pool_bytes = max(self.peak_owned_pool_bytes,
            pool.total_bytes() - self.baseline_pool_bytes)
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.peak_device_used_bytes = max(self.peak_device_used_bytes, total - free)
        if self.peak_owned_pool_bytes > self.gpu_limit:
            raise MemoryError("E209 measured owned CuPy pool exceeds 1.5 GiB")

    def _launch(self, kernel, args, output, *, name):
        if self.calls_requested >= 2:
            raise RuntimeError("E209 two conditional CUDA launch cap")
        self._measure()
        self.calls_requested += 1
        self.current_output = output
        self.current_name = name
        self.partial_host = {}
        began = time.perf_counter()
        event = {"name": name, "requested": True, "synchronized": False,
                 "readback_complete": False, "readback_keys_completed": [],
                 "kernel_wall_s": None}
        self.launch_events.append(event)
        kernel(((self.n + 127) // 128,), (128,), args)
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
            except BaseException as error:  # noqa: BLE001 - keep sibling arrays
                errors.append({"array": key, "type": type(error).__name__,
                               "message": str(error)})
        return values, errors


def run_history_cuda(context, scale, *, sample_every=20):
    cp = context.cp
    n, ticks = context.n, context.ticks
    if sample_every != context.sample_every or ticks % sample_every:
        raise ValueError("Complete conditional control intervals required")
    count = ticks // sample_every + 1
    output = {"h": cp.empty((count, n, 3), cp.float64),
        "descriptor": cp.empty((count, n, 6), cp.float64)}
    device_scale = cp.asarray(scale, dtype=cp.float64)
    context.parameter_transfer_count += 1
    context.parameter_upload_bytes += device_scale.nbytes
    result, wall = context._launch(context.history_kernel,
        (context.e, context.i, context.d, device_scale,
         output["h"], output["descriptor"], np.int32(n), np.int32(ticks),
         np.int32(sample_every)), output, name="history")
    return result, wall


def run_basis_cuda(context, scale, prototypes, initial_rate, tau, cap,
                   *, sample_every=20):
    cp = context.cp
    n, ticks = context.n, context.ticks
    if sample_every != context.sample_every or ticks % sample_every:
        raise ValueError("Complete conditional control intervals required")
    count = ticks // sample_every + 1
    output = {"h": cp.empty((count, n, 3), cp.float64),
        "descriptor": cp.empty((count, n, 6), cp.float64),
        "B": cp.empty((count, n, BASIS_COUNT), cp.float64),
        "H": cp.empty((count, n), cp.float64),
        "H_factor": cp.empty((count, n), cp.float64),
        "probe_rate": cp.empty((count, n), cp.float64)}
    parameters = [cp.asarray(value, dtype=cp.float64) for value in
        (scale, prototypes["centers"], prototypes["sigma2"],
         initial_rate, tau, cap)]
    context.parameter_transfer_count += len(parameters)
    context.parameter_upload_bytes += sum(value.nbytes for value in parameters)
    result, wall = context._launch(context.basis_kernel,
        (context.e, context.i, context.d, *parameters,
         *(output[key] for key in ("h", "descriptor", "B", "H", "H_factor", "probe_rate")),
         np.int32(n), np.int32(ticks), np.int32(sample_every)), output, name="basis")
    return result, wall
