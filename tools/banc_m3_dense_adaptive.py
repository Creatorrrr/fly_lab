"""Conditional MN adaptation on the four recorded native RK4 stage currents.

The inputs are fixed presynaptic currents.  Gain derivatives do not include
recurrent BANC, sensory, or body feedback.  No original experiment is run here.
"""

from __future__ import annotations

import math
import time

import numpy as np

DT = .00025
SAMPLE_DT = .005
TICKS_PER_SAMPLE = 20
STAGES_PER_TICK = 4
ADAPTATION_TAU = .3
MUSCLE_TAU = .01
MAX_CONTEXTS = 3
MAX_STAGES = 160000
MAX_NEURONS = 391
MAX_MUSCLES = 96

_FLOAT_KEYS = ('time_s', 'initial_rate', 'initial_adaptation',
               'initial_activation', 'tau', 'a', 'threshold', 'cap',
               'pool', 'moment', 'rest')
_INT_KEYS = ('ltm', 'membership', 'motor_ids')
_REQUIRED = ('stage_current',) + _FLOAT_KEYS + _INT_KEYS


def prepare_problem(mapping):
    """Reject oversized/malformed inputs before owning a full stage copy."""
    if any(key not in mapping for key in _REQUIRED):
        raise ValueError('Incomplete dense conditional response fixture')
    raw = {key: np.asarray(mapping[key]) for key in _REQUIRED}
    stage = raw['stage_current']
    if (stage.ndim != 3 or stage.dtype != np.float32
            or not 1 <= stage.shape[0] <= MAX_CONTEXTS
            or not 1 <= stage.shape[1] <= MAX_STAGES
            or not 1 <= stage.shape[2] <= MAX_NEURONS
            or stage.shape[1] % (STAGES_PER_TICK*TICKS_PER_SAMPLE)):
        raise ValueError('Native stage current must be bounded [C,80*k,N] float32')
    contexts, stages, neurons = stage.shape
    samples = stages//(STAGES_PER_TICK*TICKS_PER_SAMPLE)+1
    for key in _FLOAT_KEYS:
        if raw[key].dtype.kind not in 'fiu':
            raise ValueError(key+' must contain numeric values')
    for key in _INT_KEYS:
        if raw[key].dtype.kind not in 'iu':
            raise ValueError(key+' must contain native integer indices')
    time_s = raw['time_s']
    pool = raw['pool']
    if pool.ndim != 2 or not 1 <= pool.shape[0] <= MAX_MUSCLES:
        raise ValueError('Bounded muscle pool required')
    muscles = pool.shape[0]
    moment = raw['moment']
    rest = raw['rest']
    ltm = raw['ltm']
    if (time_s.shape != (samples,)
            or raw['initial_rate'].shape != (contexts, neurons)
            or raw['initial_adaptation'].shape != (contexts, neurons)
            or raw['initial_activation'].shape != (contexts, muscles)
            or any(raw[key].shape != (neurons,) for key in
                   ('tau', 'a', 'threshold', 'cap', 'membership', 'motor_ids'))
            or pool.shape != (muscles, neurons)
            or moment.ndim != 2 or not 1 <= moment.shape[0] <= 42
            or moment.shape[1] != muscles
            or rest.shape != (moment.shape[0],)
            or ltm.ndim != 1 or not 1 <= ltm.size <= 6):
        raise ValueError('Dense conditional response dimensions disagree')
    if any(not np.isfinite(raw[key]).all() for key in ('stage_current',)+_FLOAT_KEYS):
        raise ValueError('Nonfinite dense conditional input')
    if (time_s[0] != 0.
            or not np.allclose(np.diff(time_s), SAMPLE_DT, rtol=0, atol=1e-12)):
        raise ValueError('Control endpoints must be uniformly sampled at .005 s')
    if (np.any(raw['tau'] <= 0) or np.any(raw['cap'] <= 0)
            or np.any(raw['a'] < 0) or np.any(pool < 0)
            or np.any(raw['initial_rate'] < 0)
            or np.any(raw['initial_adaptation'] < 0)
            or np.any(raw['initial_activation'] < 0)
            or np.any(raw['membership'] < 0)
            or np.any(raw['membership'] >= muscles)
            or np.any(ltm < 0) or np.any(ltm >= muscles)
            or np.any(raw['motor_ids'] < 0)
            or len(np.unique(raw['motor_ids'])) != neurons
            or np.any(np.count_nonzero(pool, axis=0) != 1)
            or not np.array_equal(np.argmax(pool, axis=0), raw['membership'])):
        raise ValueError('Invalid physiology or single-muscle MN membership')
    out = {'stage_current': np.array(stage, dtype=np.float32, order='C', copy=True)}
    out.update({key: np.array(raw[key], dtype=np.float64, order='C', copy=True)
                for key in _FLOAT_KEYS})
    out.update({key: np.array(raw[key], dtype=np.int64, order='C', copy=True)
                for key in _INT_KEYS})
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


def directional_sensitivity(problem, result):
    """The simultaneous unit direction for every muscle gain."""
    act = result['activation']
    derivative = result['activation_sensitivity']
    ltm = problem['ltm']
    tanh = np.tanh(act[:, :, ltm]/10.)
    return {'rate': result['rate_sensitivity'],
            'adaptation': result['adaptation_sensitivity'],
            'activation': derivative,
            'command': .02*(derivative @ problem['moment'].T),
            'tendon': .02*(1.-tanh*tanh)*derivative[:, :, ltm]}


CUDA_SOURCE = r'''
__device__ __forceinline__ unsigned char rhs(
    double r, double z, double v, double w, double input, double gain,
    double tau, double a, double threshold, double cap,
    double *dr, double *dz, double *dv, double *dw) {
    double x = a*(input-threshold-z)/cap;
    double t = tanh(x);
    unsigned char positive = t > 0.0;
    double phi = positive ? cap*t : 0.0;
    double prime = positive ? a*(1.0-t*t) : 0.0;
    *dr = (phi-r)/tau;
    *dz = (gain*r-z)/0.3;
    *dv = (-prime*w-v)/tau;
    *dw = (r+gain*v-w)/0.3;
    return positive;
}
extern "C" __global__ void dense_adaptive_response(
    const float *stage_current, const double *initial_rate,
    const double *initial_z, const double *gain, const double *tau,
    const double *a, const double *threshold, const double *cap,
    double *out_rate, double *out_z, double *out_v, double *out_w,
    unsigned char *out_branch, int contexts, int ticks, int neurons) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index >= contexts*neurons) return;
    int context=index/neurons, neuron=index%neurons;
    int samples=ticks/20+1;
    int endpoint=context*samples*neurons+neuron;
    double r=initial_rate[index], z=initial_z[index], v=0.0, w=0.0;
    double g=gain[neuron], t=tau[neuron], slope=a[neuron];
    double th=threshold[neuron], c=cap[neuron];
    out_rate[endpoint]=r; out_z[endpoint]=z;
    out_v[endpoint]=v; out_w[endpoint]=w;
    for (int tick=0; tick<ticks; ++tick) {
        int stage=(context*ticks+tick)*4*neurons+neuron;
        double i1=(double)stage_current[stage];
        double i2=(double)stage_current[stage+neurons];
        double i3=(double)stage_current[stage+2*neurons];
        double i4=(double)stage_current[stage+3*neurons];
        double r1,z1,v1,w1,r2,z2,v2,w2,r3,z3,v3,w3,r4,z4,v4,w4;
        unsigned char b1=rhs(r,z,v,w,i1,g,t,slope,th,c,&r1,&z1,&v1,&w1);
        unsigned char b2=rhs(r+0.000125*r1,z+0.000125*z1,
            v+0.000125*v1,w+0.000125*w1,i2,g,t,slope,th,c,
            &r2,&z2,&v2,&w2);
        unsigned char b3=rhs(r+0.000125*r2,z+0.000125*z2,
            v+0.000125*v2,w+0.000125*w2,i3,g,t,slope,th,c,
            &r3,&z3,&v3,&w3);
        unsigned char b4=rhs(r+0.00025*r3,z+0.00025*z3,
            v+0.00025*v3,w+0.00025*w3,i4,g,t,slope,th,c,
            &r4,&z4,&v4,&w4);
        out_branch[(context*ticks+tick)*neurons+neuron] =
            b1 | (b2<<1) | (b3<<2) | (b4<<3);
        r += 0.00025*(r1+2.0*r2+2.0*r3+r4)/6.0;
        z += 0.00025*(z1+2.0*z2+2.0*z3+z4)/6.0;
        v += 0.00025*(v1+2.0*v2+2.0*v3+v4)/6.0;
        w += 0.00025*(w1+2.0*w2+2.0*w3+w4)/6.0;
        if ((tick+1)%20 == 0) {
            int next=endpoint+((tick+1)/20)*neurons;
            out_rate[next]=r; out_z[next]=z;
            out_v[next]=v; out_w[next]=w;
        }
    }
}
'''

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

_KERNEL = None
_EMA_KERNEL = None


def _kernel():
    global _KERNEL
    import cupy as cp
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(CUDA_SOURCE, 'dense_adaptive_response',
                               options=('--std=c++11',))
    return _KERNEL


def _ema_kernel():
    global _EMA_KERNEL
    import cupy as cp
    if _EMA_KERNEL is None:
        _EMA_KERNEL = cp.RawKernel(CUDA_EMA_SOURCE, 'muscle_ema',
                                   options=('--std=c++11',))
    return _EMA_KERNEL


class CudaDenseAdaptiveEngine:
    """One owned fixed upload; later solves transfer only MN gain values."""

    FIXED_KEYS = ('stage_current', 'initial_rate', 'initial_adaptation',
                  'initial_activation', 'tau', 'a', 'threshold', 'cap',
                  'pool', 'moment', 'rest', 'ltm', 'membership')

    def __init__(self, problem):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Actual CUDA GPU required; no CPU fallback')
        self.problem = prepare_problem(problem)
        self.device = cp.cuda.Device(0)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.fixed_upload_count_requested = 0
        self.fixed_upload_bytes_requested = 0
        self.fixed_upload_count_enqueued = 0
        self.fixed_upload_bytes_enqueued = 0
        self.fixed_upload_count_synchronized = 0
        self.fixed_upload_bytes_synchronized = 0
        self.fixed_upload_count_unknown_requested = 0
        self.fixed_upload_bytes_unknown_requested = 0
        self.fixed_upload_count = 0
        self.fixed_upload_bytes = 0
        self.device_data = {}
        try:
            with self.device, self.stream:
                for key in self.FIXED_KEYS:
                    array = self.problem[key]
                    self.fixed_upload_count_requested += 1
                    self.fixed_upload_bytes_requested += array.nbytes
                    self.device_data[key] = cp.asarray(array)
                    self.fixed_upload_count_enqueued += 1
                    self.fixed_upload_bytes_enqueued += array.nbytes
                self.stream.synchronize()
                self.fixed_upload_count_synchronized = self.fixed_upload_count_enqueued
                self.fixed_upload_bytes_synchronized = self.fixed_upload_bytes_enqueued
                self.fixed_upload_count = self.fixed_upload_count_synchronized
                self.fixed_upload_bytes = self.fixed_upload_bytes_synchronized
        except Exception as error:
            self.fixed_upload_count_unknown_requested = (
                self.fixed_upload_count_requested-self.fixed_upload_count_synchronized)
            self.fixed_upload_bytes_unknown_requested = (
                self.fixed_upload_bytes_requested-self.fixed_upload_bytes_synchronized)
            for field in ('requested', 'enqueued', 'synchronized',
                          'unknown_requested'):
                setattr(error, 'fixed_upload_count_'+field,
                        getattr(self, 'fixed_upload_count_'+field))
                setattr(error, 'fixed_upload_bytes_'+field,
                        getattr(self, 'fixed_upload_bytes_'+field))
            error.fixed_upload_count_completed = self.fixed_upload_count_synchronized
            error.fixed_upload_bytes_completed = self.fixed_upload_bytes_synchronized
            raise
        self.contexts, self.stages, self.neurons = self.problem['stage_current'].shape
        self.ticks = self.stages//STAGES_PER_TICK
        self.samples = self.ticks//TICKS_PER_SAMPLE+1
        self.muscles = self.problem['pool'].shape[0]
        self.solve_calls = 0
        self.gain_upload_count_requested = 0
        self.gain_upload_bytes_requested = 0
        self.gain_upload_count_enqueued = 0
        self.gain_upload_bytes_enqueued = 0
        self.gain_upload_count_synchronized = 0
        self.gain_upload_bytes_synchronized = 0
        self.gain_upload_count_unknown_requested = 0
        self.gain_upload_bytes_unknown_requested = 0
        self.gain_upload_count_completed = 0
        self.gain_upload_bytes_completed = 0

    def metadata(self):
        return {'backend': 'cupy_cuda_native_stage_float64',
                'decoder_backend': 'cupy_cuda_float64_pool_ema_q_tendon',
                'fixed_upload_count': self.fixed_upload_count,
                'fixed_upload_bytes': self.fixed_upload_bytes,
                'fixed_upload_count_requested': self.fixed_upload_count_requested,
                'fixed_upload_bytes_requested': self.fixed_upload_bytes_requested,
                'fixed_upload_count_enqueued': self.fixed_upload_count_enqueued,
                'fixed_upload_bytes_enqueued': self.fixed_upload_bytes_enqueued,
                'fixed_upload_count_synchronized': self.fixed_upload_count_synchronized,
                'fixed_upload_bytes_synchronized': self.fixed_upload_bytes_synchronized,
                'fixed_upload_count_unknown_requested':
                    self.fixed_upload_count_unknown_requested,
                'fixed_upload_bytes_unknown_requested':
                    self.fixed_upload_bytes_unknown_requested,
                'fixed_upload_keys': list(self.FIXED_KEYS),
                'per_gain_upload_count': 1,
                'per_gain_upload_bytes': self.neurons*8,
                'gain_upload_count_completed': self.gain_upload_count_completed,
                'gain_upload_bytes_completed': self.gain_upload_bytes_completed,
                'gain_upload_count_requested': self.gain_upload_count_requested,
                'gain_upload_bytes_requested': self.gain_upload_bytes_requested,
                'gain_upload_count_enqueued': self.gain_upload_count_enqueued,
                'gain_upload_bytes_enqueued': self.gain_upload_bytes_enqueued,
                'gain_upload_count_synchronized': self.gain_upload_count_synchronized,
                'gain_upload_bytes_synchronized': self.gain_upload_bytes_synchronized,
                'gain_upload_count_unknown_requested':
                    self.gain_upload_count_unknown_requested,
                'gain_upload_bytes_unknown_requested':
                    self.gain_upload_bytes_unknown_requested,
                'device_index': 0, 'stream': 'dedicated_nonblocking',
                'solve_calls': self.solve_calls}

    def _gpu_decode(self, neural):
        import cupy as cp
        c, s, n, m = self.contexts, self.samples, self.neurons, self.muscles
        pool = self.device_data['pool']
        pooled = (neural['rate'].reshape(c*s, n) @ pool.T).reshape(c, s, m)
        pooled_v = (neural['rate_sensitivity'].reshape(c*s, n) @ pool.T).reshape(c, s, m)
        activation = cp.full((c, s, m), cp.nan, dtype=cp.float64)
        neural['activation'] = activation
        sensitivity = cp.full((c, s, m), cp.nan, dtype=cp.float64)
        neural['activation_sensitivity'] = sensitivity
        _ema_kernel()(((c*m+31)//32,), (32,), (
            pooled, pooled_v, self.device_data['initial_activation'],
            activation, sensitivity, np.int32(c), np.int32(s), np.int32(m),
            np.float64(-math.expm1(-SAMPLE_DT/MUSCLE_TAU))))
        command = self.device_data['rest'] + .02*(
            activation.reshape(c*s, m) @ self.device_data['moment'].T).reshape(
                c, s, self.problem['rest'].size)
        neural['command'] = command
        tendon = .2*cp.tanh(activation[:, :, self.device_data['ltm']]/10.)
        neural['tendon'] = tendon
        return neural

    def solve(self, gain96):
        import cupy as cp
        gain = checked_gain(self.problem, gain96)
        began = time.perf_counter()
        c, ticks, n, samples = self.contexts, self.ticks, self.neurons, self.samples
        requested_ticks = c*ticks
        requested_s = requested_ticks*DT
        decoder_ticks = c*(samples-1)
        progress = {'stage': 'requested',
                    'requested_context_rk4_ticks': requested_ticks,
                    'requested_conditional_model_s': requested_s,
                    'requested_decoder_endpoint_ticks': decoder_ticks,
                    'completion_class': 'not_enqueued',
                    'completed_context_rk4_ticks': None,
                    'completed_conditional_model_s': None,
                    'decoder_endpoint_ticks_completed': None,
                    'decoder_completion_class': 'not_enqueued',
                    'gain_upload_count_requested': 0,
                    'gain_upload_bytes_requested': 0,
                    'gain_upload_count_enqueued': 0,
                    'gain_upload_bytes_enqueued': 0,
                    'gain_upload_count_synchronized': 0,
                    'gain_upload_bytes_synchronized': 0,
                    'gain_upload_count_unknown_requested': 0,
                    'gain_upload_bytes_unknown_requested': 0}
        device_output = {}
        host_output = {}
        self.solve_calls += 1
        try:
            with self.device, self.stream:
                progress['stage'] = 'gain_upload_requested'
                progress['gain_upload_count_requested'] = 1
                progress['gain_upload_bytes_requested'] = gain.nbytes
                self.gain_upload_count_requested += 1
                self.gain_upload_bytes_requested += gain.nbytes
                d_gain = cp.asarray(gain)
                progress['stage'] = 'gain_upload_enqueued'
                progress['gain_upload_count_enqueued'] = 1
                progress['gain_upload_bytes_enqueued'] = gain.nbytes
                self.gain_upload_count_enqueued += 1
                self.gain_upload_bytes_enqueued += gain.nbytes
                self.stream.synchronize()
                progress['stage'] = 'gain_upload_synchronized'
                progress['gain_upload_count_synchronized'] = 1
                progress['gain_upload_bytes_synchronized'] = gain.nbytes
                self.gain_upload_count_synchronized += 1
                self.gain_upload_bytes_synchronized += gain.nbytes
                self.gain_upload_count_completed += 1
                self.gain_upload_bytes_completed += gain.nbytes
                shape = (c, samples, n)
                for key in ('rate', 'adaptation', 'rate_sensitivity',
                            'adaptation_sensitivity'):
                    device_output[key] = cp.full(shape, cp.nan, dtype=cp.float64)
                device_output['branch_bits'] = cp.full((c, ticks, n),
                                                       255, dtype=cp.uint8)
                data = self.device_data
                args = (data['stage_current'], data['initial_rate'],
                        data['initial_adaptation'], d_gain,
                        *(data[key] for key in ('tau', 'a', 'threshold', 'cap')),
                        *(device_output[key] for key in ('rate', 'adaptation',
                           'rate_sensitivity', 'adaptation_sensitivity',
                           'branch_bits')),
                        np.int32(c), np.int32(ticks), np.int32(n))
                # The call wrapper may enqueue and then raise; until a stream
                # synchronization succeeds, all requested ticks are unknown.
                progress.update(stage='launch_requested', completion_class='unknown')
                _kernel()(((c*n+31)//32,), (32,), args)
                progress['stage'] = 'enqueued'
                self.stream.synchronize()
                progress.update(stage='neural_synchronized',
                    completion_class='known_complete',
                    completed_context_rk4_ticks=requested_ticks,
                    completed_conditional_model_s=requested_s)
                progress.update(stage='decoder_requested',
                                decoder_completion_class='unknown')
                self._gpu_decode(device_output)
                progress.update(stage='decoder_enqueued',
                                decoder_completion_class='unknown')
                self.stream.synchronize()
                progress.update(stage='decoder_synchronized',
                                decoder_completion_class='known_complete',
                                decoder_endpoint_ticks_completed=decoder_ticks)
                for key, value in device_output.items():
                    host_output[key] = cp.asnumpy(value, stream=self.stream)
                progress['stage'] = 'host_transfer_complete'
        except Exception as error:
            progress['gain_upload_count_unknown_requested'] = (
                progress['gain_upload_count_requested']
                -progress['gain_upload_count_synchronized'])
            progress['gain_upload_bytes_unknown_requested'] = (
                progress['gain_upload_bytes_requested']
                -progress['gain_upload_bytes_synchronized'])
            self.gain_upload_count_unknown_requested += (
                progress['gain_upload_count_unknown_requested'])
            self.gain_upload_bytes_unknown_requested += (
                progress['gain_upload_bytes_unknown_requested'])
            error.response_progress = progress.copy()
            error.gain_upload_progress = {key: value for key, value in progress.items()
                                          if key.startswith('gain_upload_')}
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
                    except Exception as partial_error:  # noqa: BLE001 - keep the original CUDA error
                        error.add_note(key+' partial read: '+repr(partial_error))
            error.partial_response = host_output
            raise
        return host_output, {'backend': 'cupy_cuda_native_stage_float64',
            'decoder_backend': 'cupy_cuda_float64_pool_ema_q_tendon',
            'context_rk4_ticks': requested_ticks,
            'conditional_model_s': requested_s,
            'decoder_endpoint_ticks': decoder_ticks,
            'progress': progress, 'wall_s': time.perf_counter()-began}


def cuda_warmup():
    """Compile and execute one separate 20-tick synthetic GPU response."""
    import cupy as cp
    began = time.perf_counter()
    tiny = prepare_problem({
        'stage_current': np.zeros((1, 80, 1), np.float32),
        'time_s': [0., SAMPLE_DT], 'initial_rate': [[0.]],
        'initial_adaptation': [[0.]], 'initial_activation': [[0.]],
        'tau': [.02], 'a': [1.], 'threshold': [7.5], 'cap': [200.],
        'pool': [[1.]], 'moment': [[1.]], 'rest': [0.],
        'ltm': np.array([0]), 'membership': np.array([0]),
        'motor_ids': np.array([0])})
    engine = result = timing = None
    try:
        engine = CudaDenseAdaptiveEngine(tiny)
        result, timing = engine.solve([0.])
        if (result['rate'].shape != (1, 2, 1)
                or result['branch_bits'].shape != (1, 20, 1)
                or any(not np.isfinite(value).all() for value in result.values())):
            raise RuntimeError('Synthetic CUDA compile/warmup failed')
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    except Exception as error:
        if engine is not None:
            for name, value in engine.metadata().items():
                if name.startswith(('fixed_upload_', 'gain_upload_')):
                    setattr(error, name, value)
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
    return {'backend': 'cupy_cuda_native_stage_float64', 'gpu': gpu_name,
            'decoder_backend': timing['decoder_backend'],
            'setup_context_rk4_ticks': timing['context_rk4_ticks'],
            'setup_conditional_model_s': timing['conditional_model_s'],
            'setup_decoder_endpoint_ticks': timing['decoder_endpoint_ticks'],
            'setup_fixed_upload_count': engine.fixed_upload_count,
            'setup_fixed_upload_bytes': engine.fixed_upload_bytes,
            **{'setup_'+name: value for name, value in engine.metadata().items()
               if name.startswith('gain_upload_')},
            **{'setup_'+name: engine.metadata()[name] for name in (
                'fixed_upload_count_requested', 'fixed_upload_bytes_requested',
                'fixed_upload_count_enqueued', 'fixed_upload_bytes_enqueued',
                'fixed_upload_count_synchronized', 'fixed_upload_bytes_synchronized',
                'fixed_upload_count_unknown_requested',
                'fixed_upload_bytes_unknown_requested')},
            'wall_s': time.perf_counter()-began}


def solve_cuda(problem, gain96):
    """Convenience solver for tiny synthetic tests; the runner reuses an engine."""
    return CudaDenseAdaptiveEngine(problem).solve(gain96)
