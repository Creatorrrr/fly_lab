"""Conditional E150 MN response to already recorded native RK4 inputs.

This is a local learning calculator. The recorded current is never substituted
for the outgoing-feedback current of a whole BANC model.
"""

from __future__ import annotations

import time

import numpy as np

WEIGHT_COUNT = 7
DT = .00025
CONTROL_STEPS = 20


def validate_inputs(current, parameters, weights, *, sample_every=CONTROL_STEPS):
    current = np.asarray(current)
    if (current.ndim != 2 or current.dtype != np.float32
            or current.shape[0] == 0 or current.shape[0] % 4 != 0
            or current.shape[1] == 0
            or not np.isfinite(current).all()):
        raise ValueError("Finite float32 four-stage native current required")
    n = current.shape[1]
    steps = current.shape[0] // 4
    if type(sample_every) is not int or sample_every < 1 or steps % sample_every:
        raise ValueError("Complete fixed control intervals required")
    expected = ("initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain", "inverse_adaptation_tau")
    values = {}
    for key in expected:
        value = np.asarray(parameters[key])
        if value.shape != (n,) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("Finite actual per-MN parameter required: " + key)
        values[key] = np.ascontiguousarray(value, dtype=np.float64)
    if np.any(values["tau"] <= 0) or np.any(values["cap"] <= 0):
        raise ValueError("Positive membrane tau and cap required")
    if np.any(values["inverse_adaptation_tau"] <= 0):
        raise ValueError("Recorded positive inverse adaptation tau required")
    w = np.asarray(weights)
    if (w.shape != (n, WEIGHT_COUNT) or w.dtype.kind != "f"
            or not np.isfinite(w).all() or np.any(np.abs(w) > 1)):
        raise ValueError("Finite per-MN seven weights within [-1,1] required")
    return (np.ascontiguousarray(current), values,
            np.ascontiguousarray(w, dtype=np.float64), steps)


def _rhs_numpy(i, r, z, R, Z, p, w):
    c = p["coefficient"]
    cap = p["cap"]
    s = c * (i - p["threshold"] - z)
    v = np.tanh(s)
    x = r / cap
    y = np.tanh(z / cap)
    f = np.stack((np.ones_like(v), v, v * v, v * v * v,
                  x, x * x, y), axis=-1)
    o = np.sum(w * f, axis=-1)
    pscale = .5 * (1 - np.tanh(o)**2)
    h = s + .5 * np.tanh(o)
    positive = h > 0
    h_slope = np.where(positive, cap * (1 - np.tanh(h)**2), 0.)
    target = cap * np.maximum(np.tanh(h), 0.)
    dr = (target - r) / p["tau"]
    dz = p["inverse_adaptation_tau"] * (p["adaptation_gain"] * r - z)
    fr = np.stack((np.zeros_like(v), np.zeros_like(v), np.zeros_like(v),
                   np.zeros_like(v), 1 / cap, 2 * x / cap,
                   np.zeros_like(v)), axis=-1)
    q = (1 - v * v) * c
    fz = np.stack((np.zeros_like(v), -q, -2 * v * q, -3 * v * v * q,
                   np.zeros_like(v), np.zeros_like(v),
                   (1 - y * y) / cap), axis=-1)
    phir = h_slope * pscale * np.sum(w * fr, axis=-1)
    phiz = h_slope * (-c + pscale * np.sum(w * fz, axis=-1))
    phiw = h_slope[:, None] * pscale[:, None] * f
    dR = (phir[:, None] * R + phiz[:, None] * Z + phiw - R) / p["tau"][:, None]
    dZ = p["inverse_adaptation_tau"][:, None] * (
        p["adaptation_gain"][:, None] * R - Z)
    return dr, dz, dR, dZ


def simulate_numpy(current, parameters, weights, *, sample_every=CONTROL_STEPS):
    """Independent vectorized float64 reference, intended for generated tests."""
    current, p, w, steps = validate_inputs(current, parameters, weights,
                                           sample_every=sample_every)
    n = current.shape[1]
    rate = np.empty((steps // sample_every + 1, n), np.float64)
    adaptation = np.empty_like(rate)
    rate_sensitivity = np.empty((len(rate), n, 7), np.float64)
    adaptation_sensitivity = np.empty_like(rate_sensitivity)
    r, z = p["initial_rate"].copy(), p["initial_adaptation"].copy()
    R, Z = np.zeros((n, 7)), np.zeros((n, 7))
    rate[0], adaptation[0] = r, z
    rate_sensitivity[0], adaptation_sensitivity[0] = R, Z
    for step in range(steps):
        i1, i2, i3, i4 = current[4 * step:4 * step + 4]
        a = _rhs_numpy(i1, r, z, R, Z, p, w)
        b = _rhs_numpy(i2, r + .5 * DT * a[0], z + .5 * DT * a[1],
                       R + .5 * DT * a[2], Z + .5 * DT * a[3], p, w)
        c = _rhs_numpy(i3, r + .5 * DT * b[0], z + .5 * DT * b[1],
                       R + .5 * DT * b[2], Z + .5 * DT * b[3], p, w)
        d = _rhs_numpy(i4, r + DT * c[0], z + DT * c[1],
                       R + DT * c[2], Z + DT * c[3], p, w)
        r += (DT / 6) * (a[0] + 2 * b[0] + 2 * c[0] + d[0])
        z += (DT / 6) * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
        R += (DT / 6) * (a[2] + 2 * b[2] + 2 * c[2] + d[2])
        Z += (DT / 6) * (a[3] + 2 * b[3] + 2 * c[3] + d[3])
        if (step + 1) % sample_every == 0:
            row = (step + 1) // sample_every
            rate[row], adaptation[row] = r, z
            rate_sensitivity[row], adaptation_sensitivity[row] = R, Z
    return {"rate": rate, "adaptation": adaptation,
            "rate_sensitivity": rate_sensitivity,
            "adaptation_sensitivity": adaptation_sensitivity}


CUDA_SOURCE = r"""
__device__ __forceinline__ void rhs(
    double I, double r, double z, const double* R, const double* Z,
    const double* w, double tau, double theta, double cap, double coefficient,
    double gain, double inv_adapt_tau,
    double* dr, double* dz, double* dR, double* dZ) {
    double s = coefficient*(I-theta-z);
    double v = tanh(s), x = r/cap, y = tanh(z/cap);
    double f[7] = {1., v, v*v, v*v*v, x, x*x, y};
    double o = 0.;
    for (int j=0; j<7; ++j) o += w[j]*f[j];
    double to = tanh(o), p = .5*(1.-to*to);
    double h = s+.5*to, th = tanh(h);
    double target = cap*fmax(th, 0.);
    *dr = (target-r)/tau;
    *dz = inv_adapt_tau*(gain*r-z);
    double hs = h > 0. ? cap*(1.-th*th) : 0.;
    double q = (1.-v*v)*coefficient;
    double fr[7] = {0.,0.,0.,0.,1./cap,2.*x/cap,0.};
    double fz[7] = {0.,-q,-2.*v*q,-3.*v*v*q,0.,0.,(1.-y*y)/cap};
    double wr=0., wz=0.;
    for (int j=0; j<7; ++j) { wr += w[j]*fr[j]; wz += w[j]*fz[j]; }
    double phir = hs*p*wr, phiz = hs*(-coefficient+p*wz);
    for (int j=0; j<7; ++j) {
        dR[j] = (phir*R[j]+phiz*Z[j]+hs*p*f[j]-R[j])/tau;
        dZ[j] = inv_adapt_tau*(gain*R[j]-Z[j]);
    }
}
extern "C" __global__ void local_response(
    const float* current, const double* tau, const double* theta,
    const double* cap, const double* coefficient, const double* gain,
    const double* inv_adapt_tau, const double* initial_rate,
    const double* initial_adaptation, const double* weights,
    double* rate, double* adaptation, double* rate_sensitivity,
    double* adaptation_sensitivity, int neurons, int steps, int sample_every,
    double dt) {
    int m = blockIdx.x*blockDim.x+threadIdx.x;
    if (m >= neurons) return;
    double r=initial_rate[m], z=initial_adaptation[m];
    double R[7]={0.,0.,0.,0.,0.,0.,0.}, Z[7]={0.,0.,0.,0.,0.,0.,0.};
    rate[m]=r; adaptation[m]=z;
    for (int j=0; j<7; ++j) {
        rate_sensitivity[m*7+j]=0.; adaptation_sensitivity[m*7+j]=0.;
    }
    const double* w = weights + m*7;
    for (int step=0; step<steps; ++step) {
        double kr[4], kz[4], kR[4][7], kZ[4][7];
        double sr=r, sz=z, SR[7], SZ[7];
        for (int j=0; j<7; ++j) { SR[j]=R[j]; SZ[j]=Z[j]; }
        for (int stage=0; stage<4; ++stage) {
            if (stage == 1) {
                sr=r+.5*dt*kr[0]; sz=z+.5*dt*kz[0];
                for (int j=0; j<7; ++j) {
                    SR[j]=R[j]+.5*dt*kR[0][j]; SZ[j]=Z[j]+.5*dt*kZ[0][j];
                }
            } else if (stage == 2) {
                sr=r+.5*dt*kr[1]; sz=z+.5*dt*kz[1];
                for (int j=0; j<7; ++j) {
                    SR[j]=R[j]+.5*dt*kR[1][j]; SZ[j]=Z[j]+.5*dt*kZ[1][j];
                }
            } else if (stage == 3) {
                sr=r+dt*kr[2]; sz=z+dt*kz[2];
                for (int j=0; j<7; ++j) {
                    SR[j]=R[j]+dt*kR[2][j]; SZ[j]=Z[j]+dt*kZ[2][j];
                }
            }
            double I=(double)current[(step*4+stage)*neurons+m];
            rhs(I,sr,sz,SR,SZ,w,tau[m],theta[m],cap[m],coefficient[m],
                gain[m],inv_adapt_tau[m],&kr[stage],&kz[stage],
                kR[stage],kZ[stage]);
        }
        r += (dt/6.)*(kr[0]+2.*kr[1]+2.*kr[2]+kr[3]);
        z += (dt/6.)*(kz[0]+2.*kz[1]+2.*kz[2]+kz[3]);
        for (int j=0; j<7; ++j) {
            R[j] += (dt/6.)*(kR[0][j]+2.*kR[1][j]+2.*kR[2][j]+kR[3][j]);
            Z[j] += (dt/6.)*(kZ[0][j]+2.*kZ[1][j]+2.*kZ[2][j]+kZ[3][j]);
        }
        if ((step+1)%sample_every == 0) {
            int row=(step+1)/sample_every;
            rate[row*neurons+m]=r; adaptation[row*neurons+m]=z;
            for (int j=0; j<7; ++j) {
                rate_sensitivity[(row*neurons+m)*7+j]=R[j];
                adaptation_sensitivity[(row*neurons+m)*7+j]=Z[j];
            }
        }
    }
}
"""


def simulate_cuda(current, parameters, weights, *, sample_every=CONTROL_STEPS,
                  timing=None):
    """One CuPy float64 local forward/tangent launch; input staged once."""
    import cupy as cp

    current, p, w, steps = validate_inputs(current, parameters, weights,
                                           sample_every=sample_every)
    n = current.shape[1]
    rows = steps // sample_every + 1
    output_bytes = 16 * rows * n * (1 + WEIGHT_COUNT)
    required = current.nbytes + output_bytes + 15 * n * 8
    if required > 768 * 1024**2:
        raise ValueError("E202 calculator exceeds 768 MiB GPU buffer limit")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("Actual CUDA GPU required; no CPU fallback")
    if timing is not None:
        timing.update(generated_warmup_calls=0, generated_warmup_ticks=0,
            cuda_launches_requested=0, cuda_launches_completed=0,
            conditional_ticks_completed=None, stage_inputs_consumed=None)
    compile_began = time.perf_counter()
    kernel = cp.RawKernel(CUDA_SOURCE, "local_response",
                          options=("--std=c++11", "--fmad=false"))
    kernel.compile()
    if timing is not None:
        timing["compile_s"] = time.perf_counter() - compile_began
    transfer_began = time.perf_counter()
    ci = cp.asarray(current)
    gpu = {key: cp.asarray(value) for key, value in p.items()}
    gw = cp.asarray(w)
    if timing is not None:
        timing["input_transfer_s"] = time.perf_counter() - transfer_began
    rate = cp.empty((rows, n), cp.float64)
    adaptation = cp.empty_like(rate)
    R = cp.empty((rows, n, 7), cp.float64)
    Z = cp.empty_like(R)
    main_began = time.perf_counter()
    if timing is not None:
        timing["cuda_launches_requested"] = 1
        timing["cuda_launches_completed"] = None
    kernel(((n + 127) // 128,), (128,), (
        ci, gpu["tau"], gpu["threshold"], gpu["cap"],
        gpu["coefficient"], gpu["adaptation_gain"],
        gpu["inverse_adaptation_tau"], gpu["initial_rate"],
        gpu["initial_adaptation"], gw, rate, adaptation, R, Z,
        np.int32(n), np.int32(steps), np.int32(sample_every), np.float64(DT)))
    cp.cuda.get_current_stream().synchronize()
    if timing is not None:
        timing["kernel_main_s"] = time.perf_counter() - main_began
        timing["cuda_launches_completed"] = 1
        timing["conditional_ticks_completed"] = steps
        timing["stage_inputs_consumed"] = len(current)
    readback_began = time.perf_counter()
    result = {"rate": cp.asnumpy(rate), "adaptation": cp.asnumpy(adaptation),
            "rate_sensitivity": cp.asnumpy(R),
            "adaptation_sensitivity": cp.asnumpy(Z)}
    if timing is not None:
        timing["output_readback_s"] = time.perf_counter() - readback_began
        timing["conditional_cuda_launches"] = 1
    return result
