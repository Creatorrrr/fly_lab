"""Conditional MN adaptation response to recorded, piecewise-linear current.

This is a feed-forward diagnostic. It does not rerun recurrent BANC, body, or
sensory dynamics, and its gain derivatives are conditional on fixed current.
"""

from __future__ import annotations

import math
import time

import numpy as np

DT = .00025
SAMPLE_DT = .005
SUBSTEPS = 20
ADAPTATION_TAU = .3
MUSCLE_TAU = .01


def prepare_problem(mapping):
    """Copy and validate an arbitrary small or full batched response fixture."""
    required = ('time_s', 'current', 'initial_rate', 'initial_adaptation',
                'initial_activation', 'tau', 'a', 'threshold', 'cap', 'pool',
                'moment', 'rest', 'ltm', 'membership', 'motor_ids')
    if any(key not in mapping for key in required):
        raise ValueError('Incomplete conditional response fixture')
    out = {}
    for key in required:
        value = np.asarray(mapping[key])
        if key in ('ltm', 'membership', 'motor_ids'):
            if value.dtype.kind not in 'iu':
                raise ValueError(key+' must contain native integer indices')
            out[key] = np.array(value, dtype=np.int64, order='C', copy=True)
        else:
            if value.dtype.kind not in 'fiu':
                raise ValueError(key+' must contain numeric values')
            out[key] = np.array(value, dtype=np.float64, order='C', copy=True)
            if not np.isfinite(out[key]).all():
                raise ValueError(key+' must be finite')
    time_s = out['time_s']
    current = out['current']
    if (time_s.ndim != 1 or len(time_s) < 2 or time_s[0] != 0.
            or not np.allclose(np.diff(time_s), SAMPLE_DT, rtol=0, atol=1e-12)
            or current.ndim != 3 or current.shape[1] != len(time_s)):
        raise ValueError('Current needs uniformly sampled .005 s PWL intervals')
    contexts, _, neurons = current.shape
    muscles, pool_neurons = out['pool'].shape if out['pool'].ndim == 2 else (0, 0)
    if (contexts < 1 or neurons < 1 or muscles < 1 or pool_neurons != neurons
            or out['initial_rate'].shape != (contexts, neurons)
            or out['initial_adaptation'].shape != (contexts, neurons)
            or out['initial_activation'].shape != (contexts, muscles)
            or any(out[key].shape != (neurons,)
                   for key in ('tau', 'a', 'threshold', 'cap', 'membership',
                               'motor_ids'))
            or out['moment'].ndim != 2
            or out['moment'].shape[1] != muscles
            or out['rest'].shape != (out['moment'].shape[0],)
            or out['ltm'].ndim != 1):
        raise ValueError('Conditional response dimensions disagree')
    if (np.any(out['tau'] <= 0) or np.any(out['cap'] <= 0)
            or np.any(out['a'] < 0) or np.any(out['pool'] < 0)
            or np.any(out['initial_rate'] < 0)
            or np.any(out['initial_adaptation'] < 0)
            or np.any(out['initial_activation'] < 0)
            or np.any(out['membership'] < 0)
            or np.any(out['membership'] >= muscles)
            or np.any(out['ltm'] < 0) or np.any(out['ltm'] >= muscles)
            or np.any(out['motor_ids'] < 0)
            or len(np.unique(out['motor_ids'])) != neurons
            or np.any(np.count_nonzero(out['pool'], axis=0) != 1)
            or not np.array_equal(np.argmax(out['pool'], axis=0),
                                  out['membership'])):
        raise ValueError('Invalid physiology or single-muscle MN membership')
    return out


def checked_gain(problem, gain96):
    raw = np.asarray(gain96)
    muscles = problem['pool'].shape[0]
    if raw.shape != (muscles,) or raw.dtype.kind not in 'fiu':
        raise ValueError('Exactly one numeric gain per muscle is required')
    gain = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(gain).all() or np.any((gain < 0) | (gain > 8)):
        raise ValueError('Finite muscle gains in [0,8] required')
    return np.ascontiguousarray(gain[problem['membership']])


def _derivative(r, z, v, w, current, gain, problem):
    argument = problem['a']*(current-problem['threshold']-z)/problem['cap']
    tanh = np.tanh(argument)
    positive = tanh > 0
    phi = np.where(positive, problem['cap']*tanh, 0.)
    phi_prime = np.where(positive, problem['a']*(1.-tanh*tanh), 0.)
    dr = (phi-r)/problem['tau']
    dz = (gain*r-z)/ADAPTATION_TAU
    dv = (-phi_prime*w-v)/problem['tau']
    dw = (r+gain*v-w)/ADAPTATION_TAU
    return dr, dz, dv, dw


def _decode_endpoints(problem, rates, sensitivity):
    contexts, samples, neurons = rates.shape
    muscles = problem['pool'].shape[0]
    flat_rate = rates.reshape(contexts*samples, neurons)
    flat_sensitivity = sensitivity.reshape(contexts*samples, neurons)
    pooled = (flat_rate @ problem['pool'].T).reshape(contexts, samples, muscles)
    pooled_sensitivity = (flat_sensitivity @ problem['pool'].T).reshape(
        contexts, samples, muscles)
    activation = np.empty((contexts, samples, muscles), np.float64)
    activation_sensitivity = np.zeros_like(activation)
    activation[:, 0] = problem['initial_activation']
    ema_fraction = -math.expm1(-SAMPLE_DT/MUSCLE_TAU)
    for index in range(1, samples):
        activation[:, index] = activation[:, index-1] + ema_fraction*(
            pooled[:, index]-activation[:, index-1])
        activation_sensitivity[:, index] = (
            activation_sensitivity[:, index-1] + ema_fraction*(
                pooled_sensitivity[:, index]-activation_sensitivity[:, index-1]))
    command = problem['rest'] + .02*(activation @ problem['moment'].T)
    tendon = .2*np.tanh(activation[:, :, problem['ltm']]/10.)
    return {'activation': activation, 'activation_sensitivity':
            activation_sensitivity, 'command': command, 'tendon': tendon}


def directional_sensitivity(problem, result):
    """Reconstruct the all-muscle gain direction from compact local factors."""
    act = result['activation']
    act_sensitivity = result['activation_sensitivity']
    ltm = problem['ltm']
    tanh = np.tanh(act[:, :, ltm]/10.)
    return {'rate': result['rate_sensitivity'],
            'adaptation': result['adaptation_sensitivity'],
            'activation': act_sensitivity,
            'command': .02*(act_sensitivity @ problem['moment'].T),
            'tendon': .02*(1.-tanh*tanh)*act_sensitivity[:, :, ltm]}


def solve_cpu(problem, gain96):
    """Independent float64 NumPy RK4 reference for small synthetic fixtures."""
    gain = checked_gain(problem, gain96)
    current = problem['current']
    contexts, samples, neurons = current.shape
    shape = (contexts, samples, neurons)
    rate = np.empty(shape, np.float64)
    adaptation = np.empty(shape, np.float64)
    v = np.zeros(shape, np.float64)
    w = np.zeros(shape, np.float64)
    rate[:, 0] = problem['initial_rate']
    adaptation[:, 0] = problem['initial_adaptation']
    started = time.perf_counter()
    for index in range(samples-1):
        r, z, vr, wz = (field[:, index].copy() for field in
                        (rate, adaptation, v, w))
        left, delta = current[:, index], current[:, index+1]-current[:, index]
        for substep in range(SUBSTEPS):
            first = left + (substep/SUBSTEPS)*delta
            middle = left + ((substep+.5)/SUBSTEPS)*delta
            last = left + ((substep+1)/SUBSTEPS)*delta
            k1 = _derivative(r, z, vr, wz, first, gain, problem)
            k2 = _derivative(*(state+DT*.5*k for state, k in
                               zip((r, z, vr, wz), k1)), middle, gain, problem)
            k3 = _derivative(*(state+DT*.5*k for state, k in
                               zip((r, z, vr, wz), k2)), middle, gain, problem)
            k4 = _derivative(*(state+DT*k for state, k in
                               zip((r, z, vr, wz), k3)), last, gain, problem)
            r, z, vr, wz = tuple(state+DT*(k1i+2*k2i+2*k3i+k4i)/6.
                                   for state, k1i, k2i, k3i, k4i in
                                   zip((r, z, vr, wz), k1, k2, k3, k4))
        rate[:, index+1], adaptation[:, index+1] = r, z
        v[:, index+1], w[:, index+1] = vr, wz
    result = {'rate': rate, 'adaptation': adaptation,
              'rate_sensitivity': v, 'adaptation_sensitivity': w}
    result.update(_decode_endpoints(problem, rate, v))
    return result, {'backend': 'cpu_float64_reference',
                    'context_rk4_ticks': contexts*(samples-1)*SUBSTEPS,
                    'conditional_model_s': contexts*(samples-1)*SAMPLE_DT,
                    'wall_s': time.perf_counter()-started}


CUDA_SOURCE = r'''
__device__ __forceinline__ void rhs(double r, double z, double v, double w,
    double input, double gain, double tau, double a, double threshold, double cap,
    double *dr, double *dz, double *dv, double *dw) {
    double x = a*(input-threshold-z)/cap;
    double t = tanh(x);
    double phi = t > 0.0 ? cap*t : 0.0;
    double prime = t > 0.0 ? a*(1.0-t*t) : 0.0;
    *dr = (phi-r)/tau;
    *dz = (gain*r-z)/0.3;
    *dv = (-prime*w-v)/tau;
    *dw = (r+gain*v-w)/0.3;
}
extern "C" __global__ void adaptive_response(
    const double *current, const double *initial_rate, const double *initial_z,
    const double *gain, const double *tau, const double *a,
    const double *threshold, const double *cap,
    double *out_rate, double *out_z, double *out_v, double *out_w,
    int contexts, int samples, int neurons) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index >= contexts*neurons) return;
    int context = index/neurons;
    int neuron = index%neurons;
    int first_index = context*samples*neurons+neuron;
    double r = initial_rate[index], z = initial_z[index], v = 0.0, w = 0.0;
    double g = gain[neuron], t = tau[neuron], slope = a[neuron];
    double th = threshold[neuron], c = cap[neuron];
    out_rate[first_index]=r; out_z[first_index]=z;
    out_v[first_index]=v; out_w[first_index]=w;
    for (int sample=0; sample<samples-1; ++sample) {
        int offset=first_index+sample*neurons;
        double left=current[offset], difference=current[offset+neurons]-left;
        for (int step=0; step<20; ++step) {
            double i1=left+difference*((double)step/20.0);
            double i2=left+difference*((double)step+0.5)/20.0;
            double i4=left+difference*((double)step+1.0)/20.0;
            double r1,z1,v1,w1,r2,z2,v2,w2,r3,z3,v3,w3,r4,z4,v4,w4;
            rhs(r,z,v,w,i1,g,t,slope,th,c,&r1,&z1,&v1,&w1);
            rhs(r+0.000125*r1,z+0.000125*z1,v+0.000125*v1,w+0.000125*w1,
                i2,g,t,slope,th,c,&r2,&z2,&v2,&w2);
            rhs(r+0.000125*r2,z+0.000125*z2,v+0.000125*v2,w+0.000125*w2,
                i2,g,t,slope,th,c,&r3,&z3,&v3,&w3);
            rhs(r+0.00025*r3,z+0.00025*z3,v+0.00025*v3,w+0.00025*w3,
                i4,g,t,slope,th,c,&r4,&z4,&v4,&w4);
            r += 0.00025*(r1+2.0*r2+2.0*r3+r4)/6.0;
            z += 0.00025*(z1+2.0*z2+2.0*z3+z4)/6.0;
            v += 0.00025*(v1+2.0*v2+2.0*v3+v4)/6.0;
            w += 0.00025*(w1+2.0*w2+2.0*w3+w4)/6.0;
        }
        int next=offset+neurons;
        out_rate[next]=r; out_z[next]=z; out_v[next]=v; out_w[next]=w;
    }
}
'''

_KERNEL = None
_EMA_KERNEL = None

CUDA_EMA_SOURCE = r'''
extern "C" __global__ void muscle_ema(
    const double *pooled, const double *pooled_v, const double *initial,
    double *activation, double *sensitivity,
    int contexts, int samples, int muscles, double fraction) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if (index>=contexts*muscles) return;
    int context=index/muscles, muscle=index%muscles;
    int first=context*samples*muscles+muscle;
    double a=initial[index], da=0.0;
    activation[first]=a; sensitivity[first]=da;
    for (int sample=1; sample<samples; ++sample) {
        int offset=first+sample*muscles;
        a += fraction*(pooled[offset]-a);
        da += fraction*(pooled_v[offset]-da);
        activation[offset]=a;
        sensitivity[offset]=da;
    }
}
'''


def _kernel():
    global _KERNEL
    import cupy as cp

    if _KERNEL is None:
        _KERNEL = cp.RawKernel(CUDA_SOURCE, 'adaptive_response',
                               options=('--std=c++11',))
    return _KERNEL


def _ema_kernel():
    global _EMA_KERNEL
    import cupy as cp

    if _EMA_KERNEL is None:
        _EMA_KERNEL = cp.RawKernel(CUDA_EMA_SOURCE, 'muscle_ema',
                                   options=('--std=c++11',))
    return _EMA_KERNEL


def _synchronize():
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


class CudaAdaptiveResponseEngine:
    """Own one fixed device upload reused for all 96-gain evaluations."""

    FIXED_KEYS = ('current', 'initial_rate', 'initial_adaptation',
                  'initial_activation', 'tau', 'a', 'threshold', 'cap',
                  'pool', 'moment', 'rest', 'ltm', 'membership')

    def __init__(self, problem):
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Actual CUDA GPU required; no CPU fallback')
        self.problem = prepare_problem(problem)  # Owned input copies.
        self.device = cp.cuda.Device(0)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.fixed_upload_count = 0
        self.fixed_upload_bytes = 0
        self.device_data = {}
        try:
            with self.device, self.stream:
                for key in self.FIXED_KEYS:
                    array = self.problem[key]
                    self.device_data[key] = cp.asarray(array)
                    self.fixed_upload_count += 1
                    self.fixed_upload_bytes += array.nbytes
        except Exception as error:
            error.fixed_upload_count_completed = self.fixed_upload_count
            error.fixed_upload_bytes_completed = self.fixed_upload_bytes
            raise
        self.contexts, self.samples, self.neurons = self.problem['current'].shape
        self.muscles = self.problem['pool'].shape[0]
        self.solve_calls = 0
        self.gain_upload_count_completed = 0
        self.gain_upload_bytes_completed = 0

    def metadata(self):
        return {'backend': 'cupy_cuda_float64',
                'decoder_backend': 'cupy_cuda_float64_pool_ema_q_tendon',
                'fixed_upload_count': self.fixed_upload_count,
                'fixed_upload_bytes': self.fixed_upload_bytes,
                'fixed_upload_keys': list(self.FIXED_KEYS),
                'per_gain_upload_count': 1,
                'per_gain_upload_bytes': self.neurons*8,
                'gain_upload_count_completed': self.gain_upload_count_completed,
                'gain_upload_bytes_completed': self.gain_upload_bytes_completed,
                'device_index': 0, 'stream': 'dedicated_nonblocking',
                'solve_calls': self.solve_calls}

    def _gpu_decode(self, neural):
        import cupy as cp

        rate = neural['rate']
        sensitivity = neural['rate_sensitivity']
        c, t, n = self.contexts, self.samples, self.neurons
        m = self.muscles
        pool = self.device_data['pool']
        pooled = (rate.reshape(c*t, n) @ pool.T).reshape(c, t, m)
        pooled_v = (sensitivity.reshape(c*t, n) @ pool.T).reshape(c, t, m)
        activation = cp.full((c, t, m), cp.nan, dtype=cp.float64)
        neural['activation'] = activation
        act_sensitivity = cp.full((c, t, m), cp.nan, dtype=cp.float64)
        neural['activation_sensitivity'] = act_sensitivity
        _ema_kernel()(((c*m+31)//32,), (32,), (
            pooled, pooled_v, self.device_data['initial_activation'],
            activation, act_sensitivity, np.int32(c), np.int32(t),
            np.int32(m), np.float64(-math.expm1(-SAMPLE_DT/MUSCLE_TAU))))
        command = self.device_data['rest'] + .02*(
            activation.reshape(c*t, m) @ self.device_data['moment'].T).reshape(
                c, t, self.problem['rest'].size)
        neural['command'] = command
        tendon = .2*cp.tanh(activation[:, :, self.device_data['ltm']]/10.)
        neural['tendon'] = tendon
        return {'activation': activation,
                'activation_sensitivity': act_sensitivity,
                'command': command, 'tendon': tendon}

    def solve(self, gain96):
        import cupy as cp

        gain = checked_gain(self.problem, gain96)
        began = time.perf_counter()
        c, t, n = self.contexts, self.samples, self.neurons
        requested_ticks = c*(t-1)*SUBSTEPS
        requested_s = c*(t-1)*SAMPLE_DT
        progress = {'stage': 'requested',
                    'requested_context_rk4_ticks': requested_ticks,
                    'requested_conditional_model_s': requested_s,
                    'requested_decoder_endpoint_ticks': c*(t-1),
                    'completion_class': 'not_enqueued',
                    'completed_context_rk4_ticks': None,
                    'completed_conditional_model_s': None,
                    'decoder_endpoint_ticks_completed': None,
                    'decoder_completion_class': 'not_enqueued'}
        device_output = {}
        host_output = {}
        self.solve_calls += 1
        try:
            with self.device, self.stream:
                d_gain = cp.asarray(gain)  # Only per-batch input transfer.
                self.gain_upload_count_completed += 1
                self.gain_upload_bytes_completed += gain.nbytes
                shape = (c, t, n)
                for key in ('rate', 'adaptation', 'rate_sensitivity',
                            'adaptation_sensitivity'):
                    device_output[key] = cp.full(shape, cp.nan,
                                                  dtype=cp.float64)
                data = self.device_data
                arguments = (data['current'], data['initial_rate'],
                    data['initial_adaptation'], d_gain,
                    *(data[key] for key in ('tau', 'a', 'threshold', 'cap')),
                    *(device_output[key] for key in ('rate', 'adaptation',
                        'rate_sensitivity', 'adaptation_sensitivity')),
                    np.int32(c), np.int32(t), np.int32(n))
                # One warp/block gives 37 independent resident block
                # candidates for the original 3*391 long temporal chains.
                _kernel()(((c*n+31)//32,), (32,), arguments)
                progress.update(stage='enqueued', completion_class='unknown')
                _synchronize()
                progress.update(stage='neural_synchronized',
                    completion_class='known_complete',
                    completed_context_rk4_ticks=requested_ticks,
                    completed_conditional_model_s=requested_s)
                progress.update(stage='decoder_requested',
                                decoder_completion_class='unknown')
                device_output.update(self._gpu_decode(device_output))
                progress.update(stage='decoder_enqueued',
                                decoder_completion_class='unknown')
                _synchronize()
                progress.update(stage='decoder_synchronized',
                                decoder_completion_class='known_complete',
                                decoder_endpoint_ticks_completed=c*(t-1))
                for key, value in device_output.items():
                    host_output[key] = cp.asnumpy(value, stream=self.stream)
                progress['stage'] = 'host_transfer_complete'
        except Exception as error:
            error.response_progress = progress.copy()
            error.requested_context_rk4_ticks = requested_ticks
            error.requested_conditional_model_s = requested_s
            if progress['completed_context_rk4_ticks'] is not None:
                error.completed_context_rk4_ticks = requested_ticks
                error.completed_conditional_model_s = requested_s
            for key, value in device_output.items():
                if key not in host_output:
                    try:
                        with self.device:
                            host_output[key] = cp.asnumpy(value, stream=self.stream)
                    except Exception as partial_error:  # noqa: BLE001
                        error.add_note(key+' partial read: '+repr(partial_error))
            error.partial_response = host_output
            raise
        return host_output, {'backend': 'cupy_cuda_float64',
            'decoder_backend': 'cupy_cuda_float64_pool_ema_q_tendon',
            'context_rk4_ticks': requested_ticks,
            'conditional_model_s': requested_s,
            'decoder_endpoint_ticks': c*(t-1),
            'progress': progress,
            'wall_s': time.perf_counter()-began}


def cuda_warmup():
    """Compile and execute one separate 20-step synthetic GPU response."""
    import cupy as cp

    began = time.perf_counter()
    tiny = prepare_problem({'time_s': [0., SAMPLE_DT],
        'current': np.zeros((1, 2, 1)), 'initial_rate': [[0.]],
        'initial_adaptation': [[0.]], 'initial_activation': [[0.]],
        'tau': [.02], 'a': [1.], 'threshold': [7.5], 'cap': [200.],
        'pool': [[1.]], 'moment': [[1.]], 'rest': [0.],
        'ltm': np.array([0]), 'membership': np.array([0]),
        'motor_ids': np.array([0])})
    engine = result = timing = None
    try:
        engine = CudaAdaptiveResponseEngine(tiny)
        result, timing = engine.solve([0.])
        if (result['rate'].shape != (1, 2, 1)
                or any(not np.isfinite(value).all() for value in result.values())):
            raise RuntimeError('Synthetic CUDA compile/warmup failed')
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    except Exception as error:
        if engine is not None:
            error.fixed_upload_count_completed = engine.fixed_upload_count
            error.fixed_upload_bytes_completed = engine.fixed_upload_bytes
        if timing is not None:
            error.response_progress = timing['progress'].copy()
            error.completed_context_rk4_ticks = timing['context_rk4_ticks']
            error.completed_conditional_model_s = timing['conditional_model_s']
            error.partial_response = result
        elif not hasattr(error, 'response_progress'):
            error.response_progress = {'stage': 'setup_before_solve',
                                       'completion_class': 'not_enqueued'}
        raise
    return {'backend': 'cupy_cuda_float64', 'gpu': gpu_name,
            'decoder_backend': timing['decoder_backend'],
            'setup_context_rk4_ticks': timing['context_rk4_ticks'],
            'setup_conditional_model_s': timing['conditional_model_s'],
            'setup_decoder_endpoint_ticks': timing['decoder_endpoint_ticks'],
            'setup_fixed_upload_count': engine.fixed_upload_count,
            'setup_fixed_upload_bytes': engine.fixed_upload_bytes,
            'wall_s': time.perf_counter()-began}


def solve_cuda(problem, gain96):
    """Convenience solver for tiny independent tests; original runner reuses an engine."""
    return CudaAdaptiveResponseEngine(problem).solve(gain96)
