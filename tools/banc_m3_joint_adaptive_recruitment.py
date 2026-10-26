"""E128 conditional four-stage MN response and explicit edge variational bank.

Recorded presynaptic rates are fixed. This module never advances a BANC
network or a body. All neural integration and its sensitivities use CUDA.
"""

# ruff: noqa: BLE001  # preserve primary CUDA faults and secondary rescue errors

from __future__ import annotations

import math
import time

import numpy as np

from tools.banc_m3_dense_adaptive import (
    DT,
    MAX_CONTEXTS,
    MAX_MUSCLES,
    MAX_NEURONS,
    MAX_STAGES,
    MUSCLE_TAU,
    SAMPLE_DT,
)

MAX_SOURCES = 69
MAX_UNION_EDGES = 2444
MAX_FIT_EDGES = 1572
FIELDS = ('rate', 'adaptation', 'rate_sensitivity',
          'adaptation_sensitivity', 'activation',
          'activation_sensitivity', 'command', 'tendon')


def owned(value, dtype=None):
    """Always detach a validated numerical array from its caller."""
    return np.array(value, dtype=dtype, order='C', copy=True)


def prepare_joint_problem(mapping, motor_indices=None, muscle_indices=None,
                          fit_indices=None):
    """Validate all extents before copying any large native stage array."""
    required = ('stage_current', 'source_rate', 'time_s', 'initial_rate',
                'initial_adaptation', 'initial_activation', 'tau', 'a',
                'threshold', 'cap', 'pool', 'moment', 'rest', 'ltm',
                'membership', 'motor_ids', 'source_ids', 'union_edge_ids',
                'union_rows', 'union_cols', 'union_values',
                'fit_union_positions', 'edge_ids', 'edge_rows', 'edge_cols',
                'base_fit_weights')
    if any(k not in mapping for k in required):
        raise ValueError('Incomplete E128 joint input')
    x = {k: np.asarray(mapping[k]) for k in required}
    current, source = x['stage_current'], x['source_rate']
    if (current.ndim != 3 or source.ndim != 3 or
            current.dtype != np.float32 or
            not 1 <= current.shape[0] <= MAX_CONTEXTS or
            not 1 <= current.shape[1] <= MAX_STAGES or
            current.shape[1] % 80 or
            not 1 <= current.shape[2] <= MAX_NEURONS or
            source.shape[:2] != current.shape[:2] or
            source.dtype != np.float32 or
            not 1 <= source.shape[2] <= MAX_SOURCES):
        raise ValueError('Bounded native four-stage current/source required')
    c, stages, n = current.shape
    s = source.shape[2]
    samples = stages // 80 + 1
    m = x['pool'].shape[0] if x['pool'].ndim == 2 else 0
    u = x['union_edge_ids'].size
    e = x['edge_ids'].size
    if (not 1 <= m <= MAX_MUSCLES or not 1 <= u <= MAX_UNION_EDGES or
            not 1 <= e <= MAX_FIT_EDGES or
            x['time_s'].shape != (samples,) or
            x['initial_rate'].shape != (c, n) or
            x['initial_adaptation'].shape != (c, n) or
            x['initial_activation'].shape != (c, m) or
            any(x[k].shape != (n,) for k in
                ('tau', 'a', 'threshold', 'cap', 'membership', 'motor_ids')) or
            x['pool'].shape != (m, n) or x['moment'].ndim != 2 or
            not 1 <= x['moment'].shape[0] <= 42 or
            x['moment'].shape[1] != m or
            x['rest'].shape != (x['moment'].shape[0],) or
            x['ltm'].ndim != 1 or not 1 <= x['ltm'].size <= 6 or
            x['source_ids'].shape != (s,) or
            any(x[k].shape != (u,) for k in
                ('union_rows', 'union_cols')) or
            x['union_values'].shape != (c, u) or
            x['fit_union_positions'].shape != (e,) or
            any(x[k].shape != (e,) for k in
                ('edge_rows', 'edge_cols', 'base_fit_weights'))):
        raise ValueError('Joint input dimensions disagree')
    integers = ('ltm', 'membership', 'motor_ids', 'source_ids',
                'union_edge_ids', 'union_rows', 'union_cols',
                'fit_union_positions', 'edge_ids', 'edge_rows', 'edge_cols')
    numeric = ('time_s', 'initial_rate', 'initial_adaptation',
               'initial_activation', 'tau', 'a', 'threshold', 'cap',
               'pool', 'moment', 'rest', 'union_values', 'base_fit_weights')
    if any(x[k].dtype.kind not in 'fiu' for k in numeric):
        raise ValueError('Numerical physiology and edge values required')
    if any(x[k].dtype.kind not in 'iu' for k in integers):
        raise ValueError('Explicit edge and neuron IDs must be integers')
    if (np.any(x['tau'] <= 0) or np.any(x['cap'] <= 0) or
            np.any(x['a'] < 0) or np.any(x['pool'] < 0) or
            np.any(x['membership'] < 0) or np.any(x['membership'] >= m) or
            np.any(x['ltm'] < 0) or np.any(x['ltm'] >= m) or
            np.any(x['union_rows'] < 0) or np.any(x['union_rows'] >= n) or
            np.any(x['union_cols'] < 0) or np.any(x['union_cols'] >= s) or
            np.any(x['motor_ids'] < 0) or np.any(x['source_ids'] < 0) or
            np.any(x['initial_rate'] < 0) or
            np.any(x['initial_adaptation'] < 0) or
            np.any(x['initial_activation'] < 0) or
            np.any(x['fit_union_positions'] < 0) or
            np.any(x['fit_union_positions'] >= u) or
            len(np.unique(x['union_edge_ids'])) != u or
            len(np.unique(x['edge_ids'])) != e or
            len(np.unique(x['source_ids'])) != s or
            len(np.unique(x['motor_ids'])) != n or
            len(np.unique(np.stack((x['union_rows'],x['union_cols']),axis=1),
                axis=0)) != u or
            len(np.unique(x['fit_union_positions'])) != e or
            not np.array_equal(x['edge_ids'],
                               x['union_edge_ids'][x['fit_union_positions']]) or
            not np.array_equal(x['edge_rows'],
                               x['union_rows'][x['fit_union_positions']]) or
            not np.array_equal(x['edge_cols'],
                               x['union_cols'][x['fit_union_positions']]) or
            not np.array_equal(x['base_fit_weights'],
                x['union_values'][-1,x['fit_union_positions']].astype(np.float64)) or
            np.any(x['base_fit_weights']==0) or
            not np.array_equal(np.argmax(x['pool'], axis=0), x['membership']) or
            np.any(np.count_nonzero(x['pool'], axis=0) != 1) or
            not np.allclose(np.diff(x['time_s']), SAMPLE_DT, rtol=0, atol=1e-12)
            or x['time_s'][0] != 0):
        raise ValueError('Invalid joint mapping or physiology')
    small = tuple(k for k in required if k not in ('stage_current', 'source_rate'))
    if any(not np.isfinite(x[k]).all() for k in small if x[k].dtype.kind in 'f'):
        raise ValueError('Nonfinite joint input')
    # Large arrays are checked after extents, before any owned full copy.
    if (not np.isfinite(current).all() or not np.isfinite(source).all() or
            np.any(source < 0)):
        raise ValueError('Nonfinite recorded stage input')
    motors = np.arange(n) if motor_indices is None else np.asarray(motor_indices)
    muscles = np.arange(m) if muscle_indices is None else np.asarray(muscle_indices)
    fits = np.arange(e) if fit_indices is None else np.asarray(fit_indices)
    for ids, width, label in ((motors, n, 'motor'), (muscles, m, 'muscle'),
                              (fits, e, 'fit edge')):
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or ids.size == 0 or
                np.any(ids < 0) or np.any(ids >= width) or
                len(np.unique(ids)) != ids.size):
            raise ValueError('Explicit unique '+label+' indices required')
    if not np.all(np.isin(x['membership'][motors], muscles)):
        raise ValueError('Selected MNs require their owned muscles')
    if not np.all(np.isin(x['edge_rows'][fits], motors)):
        raise ValueError('Selected fit edges require their post MNs')
    local_motor = np.full(n, -1, dtype=np.int32)
    local_motor[motors] = np.arange(len(motors), dtype=np.int32)
    local_muscle = np.full(m, -1, dtype=np.int32)
    local_muscle[muscles] = np.arange(len(muscles), dtype=np.int32)
    keep = np.isin(x['union_rows'], motors)
    union_pos = np.flatnonzero(keep)
    order = np.argsort(x['union_rows'][union_pos], kind='stable')
    union_pos = union_pos[order]
    row = local_motor[x['union_rows'][union_pos]]
    row_order = np.argsort(row, kind='stable')
    union_pos, row = union_pos[row_order], row[row_order]
    ptr = np.r_[0, np.cumsum(np.bincount(row, minlength=len(motors)))].astype(np.int32)
    global_fit_to_local = np.full(e, -1, dtype=np.int32)
    global_fit_to_local[fits] = np.arange(len(fits), dtype=np.int32)
    union_to_fit = np.full(u, -1, dtype=np.int32)
    union_to_fit[x['fit_union_positions']] = np.arange(e, dtype=np.int32)
    union_fit = global_fit_to_local[union_to_fit[union_pos].clip(min=0)]
    union_fit[union_to_fit[union_pos] < 0] = -1
    local_fit_union = np.full(u, -1, dtype=np.int32)
    local_fit_union[union_pos] = np.arange(len(union_pos), dtype=np.int32)
    local_fit_pos = local_fit_union[x['fit_union_positions'][fits]]
    if np.any(local_fit_pos < 0):
        raise ValueError('Fit edge absent from selected union')
    local_ltm = local_muscle[x['ltm']]
    kept_ltm = local_ltm >= 0
    return {
        'stage_current': owned(current[:, :, motors], np.float32),
        'source_rate': owned(source, np.float32),
        'initial_rate': owned(x['initial_rate'][:, motors], np.float64),
        'initial_adaptation': owned(x['initial_adaptation'][:, motors], np.float64),
        'initial_activation': owned(x['initial_activation'][:, muscles], np.float64),
        'tau': owned(x['tau'][motors], np.float64),
        'a': owned(x['a'][motors], np.float64),
        'threshold': owned(x['threshold'][motors], np.float64),
        'cap': owned(x['cap'][motors], np.float64),
        'pool': owned(x['pool'][np.ix_(muscles, motors)], np.float64),
        'moment': owned(x['moment'][:, muscles], np.float64),
        'rest': owned(x['rest'], np.float64),
        'ltm': owned(local_ltm[kept_ltm], np.int32),
        'ltm_global_indices': np.flatnonzero(kept_ltm),
        'membership': owned(local_muscle[x['membership'][motors]], np.int32),
        'motor_indices': owned(motors, np.int32),
        'muscle_indices': owned(muscles, np.int32),
        'fit_indices': owned(fits, np.int32),
        'edge_ids': owned(x['edge_ids'][fits], np.int64),
        'edge_rows': owned(local_motor[x['edge_rows'][fits]], np.int32),
        'edge_cols': owned(x['edge_cols'][fits], np.int32),
        'edge_muscles': owned(local_muscle[x['membership'][x['edge_rows'][fits]]], np.int32),
        'fit_union_local': owned(local_fit_pos, np.int32),
        'union_ptr': ptr,
        'union_cols': owned(x['union_cols'][union_pos], np.int32),
        'union_fit': owned(union_fit, np.int32),
        'union_base': owned(x['union_values'][-1, union_pos], np.float64),
        'union_diff': owned(
            x['union_values'][-1, union_pos][None, :].astype(np.float64)-
            x['union_values'][:, union_pos].astype(np.float64)),
        'base_fit_weights': owned(x['base_fit_weights'][fits], np.float64),
        'time_s': owned(x['time_s'], np.float64),
        'contexts': c, 'stages': stages, 'sources': s,
    }


CUDA_SOURCE = r'''
__device__ __forceinline__ unsigned char rhs(
    double r,double z,double v,double w,double input,double gain,
    double tau,double a,double threshold,double cap,
    double *dr,double *dz,double *dv,double *dw,double *prime) {
    double x=a*(input-threshold-z)/cap;
    double t=tanh(x);
    unsigned char positive=t>0.;
    double phi=positive ? cap*t : 0.;
    *prime=positive ? a*(1.-t*t) : 0.;
    *dr=(phi-r)/tau;
    *dz=(gain*r-z)/0.3;
    *dv=(-*prime*w-v)/tau;
    *dw=(r+gain*v-w)/0.3;
    return positive;
}
extern "C" __global__ void joint_forward(
    const float *native,const float *source,const double *initial_r,
    const double *initial_z,const double *gain,const double *delta,
    const double *tau,const double *a,const double *threshold,const double *cap,
    const int *ptr,const int *cols,const int *fit,
    const double *base,const double *diff,
    double *out_r,double *out_z,double *out_v,double *out_w,
    unsigned char *branch,double *prime,
    int contexts,int ticks,int neurons,int sources,int edges,int native_mode,
    int record_prime) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*neurons) return;
    int c=index/neurons,n=index%neurons,samples=ticks/20+1;
    int endpoint=c*samples*neurons+n;
    double r=initial_r[index],z=initial_z[index],v=0.,w=0.;
    double g=gain[n],t=tau[n],slope=a[n],th=threshold[n],capn=cap[n];
    out_r[endpoint]=r;out_z[endpoint]=z;out_v[endpoint]=v;out_w[endpoint]=w;
    for(int tick=0;tick<ticks;++tick) {
      double k[4][4]; unsigned char bits=0;
      for(int stage=0;stage<4;++stage) {
        int stage_index=(c*ticks+tick)*4+stage;
        double input=(double)native[stage_index*neurons+n];
        if(!native_mode) {
          for(int j=ptr[n];j<ptr[n+1];++j) {
            double s=(double)source[stage_index*sources+cols[j]];
            input+=s*diff[c*edges+j];
            if(fit[j]>=0) input+=s*base[j]*expm1(delta[fit[j]]);
          }
        }
        double rr=r,zz=z,vv=v,ww=w;
        if(stage>0) {
          double h=stage==3 ? 0.00025 : 0.000125;
          rr+=h*k[stage-1][0];zz+=h*k[stage-1][1];
          vv+=h*k[stage-1][2];ww+=h*k[stage-1][3];
        }
        double p;
        bits|=rhs(rr,zz,vv,ww,input,g,t,slope,th,capn,
                  &k[stage][0],&k[stage][1],&k[stage][2],&k[stage][3],&p)<<stage;
        if(record_prime) prime[stage_index*neurons+n]=p;
      }
      branch[(c*ticks+tick)*neurons+n]=bits;
      r+=0.00025*(k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])/6.;
      z+=0.00025*(k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])/6.;
      v+=0.00025*(k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])/6.;
      w+=0.00025*(k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])/6.;
      if((tick+1)%20==0) {
        int next=endpoint+(tick+1)/20*neurons;
        out_r[next]=r;out_z[next]=z;out_v[next]=v;out_w[next]=w;
      }
    }
}
extern "C" __global__ void edge_bank(
    const float *source,const double *prime,const double *gain,
    const double *tau,const double *pool,const int *rows,
    const int *cols,const int *muscles,const double *base,
    const double *delta,double *B,
    int contexts,int ticks,int neurons,int sources,int edge_count,
    int muscle_count,int params,double fraction) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*edge_count) return;
    int c=index/edge_count,e=index%edge_count;
    int n=rows[e],m=muscles[e],samples=ticks/20+1;
    double p=0.,q=0.,act=0.,g=gain[n],t=tau[n];
    double weight=base[e]*exp(delta[e]);
    double factor=pool[m*neurons+n];
    B[(c*samples)*params+e]=0.;
    for(int tick=0;tick<ticks;++tick) {
      double kp[4],kq[4];
      for(int stage=0;stage<4;++stage) {
        int stage_index=(c*ticks+tick)*4+stage;
        double pp=p,qq=q;
        if(stage>0) {
          double h=stage==3 ? 0.00025 : 0.000125;
          pp+=h*kp[stage-1];qq+=h*kq[stage-1];
        }
        double ph=prime[stage_index*neurons+n];
        double drive=weight*(double)source[stage_index*sources+cols[e]];
        kp[stage]=(ph*(drive-qq)-pp)/t;
        kq[stage]=(g*pp-qq)/0.3;
      }
      p+=0.00025*(kp[0]+2*kp[1]+2*kp[2]+kp[3])/6.;
      q+=0.00025*(kq[0]+2*kq[1]+2*kq[2]+kq[3])/6.;
      if((tick+1)%20==0) {
        act+=fraction*(factor*p-act);
        B[(c*samples+(tick+1)/20)*params+e]=act;
      }
    }
}
extern "C" __global__ void muscle_ema(
    const double *pooled,const double *pooled_v,const double *initial,
    double *activation,double *sensitivity,
    int contexts,int samples,int muscles,double fraction) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*muscles) return;
    int c=index/muscles,m=index%muscles;
    int first=c*samples*muscles+m;
    double a=initial[index],da=0.;
    activation[first]=a;sensitivity[first]=da;
    for(int sample=1;sample<samples;++sample) {
      int offset=first+sample*muscles;
      a+=fraction*(pooled[offset]-a);
      da+=fraction*(pooled_v[offset]-da);
      activation[offset]=a;sensitivity[offset]=da;
    }
}
'''

_KERNEL = {}


def _transfer_bucket():
    return {state: {'count': 0, 'bytes': 0} for state in
            ('requested', 'enqueued', 'synchronized', 'unknown_requested')}


def _add_transfer(bucket, state, nbytes):
    bucket[state]['count'] += 1
    bucket[state]['bytes'] += int(nbytes)


def _kernel(name):
    import cupy as cp
    if name not in _KERNEL:
        _KERNEL[name] = cp.RawKernel(CUDA_SOURCE, name, options=('--std=c++11',))
    return _KERNEL[name]


class CudaJointAdaptiveEngine:
    """One fixed CUDA upload, then only float64 delta and gain uploads."""

    FIXED_KEYS = ('stage_current', 'source_rate', 'initial_rate',
                  'initial_adaptation', 'initial_activation', 'tau', 'a',
                  'threshold', 'cap', 'pool', 'moment', 'rest', 'ltm',
                  'membership', 'union_ptr', 'union_cols', 'union_fit',
                  'union_base', 'union_diff', 'edge_rows', 'edge_cols',
                  'edge_muscles', 'base_fit_weights')

    def __init__(self, mapping, motor_indices=None, muscle_indices=None,
                 fit_indices=None):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Actual CUDA GPU required')
        self.problem = prepare_joint_problem(mapping, motor_indices,
                                             muscle_indices, fit_indices)
        self.device = cp.cuda.Device(0)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.upload = _transfer_bucket()
        self.dynamic_upload = _transfer_bucket()
        self.fixed_upload_by_key = {}
        self.dynamic_upload_by_name = {
            'delta':_transfer_bucket(),'gain':_transfer_bucket()}
        self.host_transfer = _transfer_bucket()
        self.host_transfer_by_label = {}
        self.cotangent_upload = _transfer_bucket()
        self.index_upload = _transfer_bucket()
        self.index_upload_by_label = {}
        self.index_cache = {}
        self.index_cache_hits = 0
        self.torch_index_cache = {}
        self.torch_index_cache_hits = 0
        self.torch_input_transfer = _transfer_bucket()
        self.torch_input_transfer_by_label = {}
        self.torch_host_transfer = _transfer_bucket()
        self.torch_host_transfer_by_label = {}
        self.torch_cupy_shared_bytes = 0
        self.torch_contiguous_copy_bytes = 0
        self.torch_activation_shared_bytes = 0
        self.geometry_cost = {'forward_rows_requested': 0,
            'forward_rows_enqueued': 0, 'forward_rows_completed': 0,
            'backward_rows_requested': 0, 'backward_rows_enqueued': 0,
            'backward_rows_completed': 0}
        self.device_data = {}
        try:
            with self.device, self.stream:
                for key in self.FIXED_KEYS:
                    array = self.problem[key]
                    event=self.fixed_upload_by_key.setdefault(key,
                        _transfer_bucket())
                    _add_transfer(self.upload, 'requested', array.nbytes)
                    _add_transfer(event,'requested',array.nbytes)
                    self.device_data[key] = cp.asarray(array)
                    _add_transfer(self.upload, 'enqueued', array.nbytes)
                    _add_transfer(event,'enqueued',array.nbytes)
                self.stream.synchronize()
                self.upload['synchronized'] = self.upload['enqueued'].copy()
                for event in self.fixed_upload_by_key.values():
                    event['synchronized']=event['enqueued'].copy()
        except Exception as error:
            self.upload['unknown_requested'] = {
                k: self.upload['requested'][k]-self.upload['synchronized'][k]
                for k in ('count', 'bytes')}
            for event in self.fixed_upload_by_key.values():
                event['unknown_requested']={k:event['requested'][k]-
                    event['synchronized'][k] for k in ('count','bytes')}
            error.fixed_upload_progress = self.upload.copy()
            error.fixed_upload_by_key=self.fixed_upload_by_key
            raise
        p = self.problem
        self.contexts = p['contexts']
        self.ticks = p['stages']//4
        self.samples = self.ticks//20+1
        self.neurons = p['stage_current'].shape[2]
        self.muscles = p['pool'].shape[0]
        self.edges = p['edge_ids'].size
        self.solve_calls = 0
        self.bank_calls = 0

    def metadata(self):
        geometry = self.geometry_cost.copy()
        geometry['forward_rows_unknown_requested'] = (
            geometry['forward_rows_requested']-
            geometry['forward_rows_completed'])
        geometry['backward_rows_unknown_requested'] = (
            geometry['backward_rows_requested']-
            geometry['backward_rows_completed'])
        return {'backend': 'cupy_cuda_native_four_stage_float64_joint',
                'stream': 'dedicated_nonblocking',
                'fixed_upload': self.upload,
                'fixed_upload_by_key':self.fixed_upload_by_key,
                'dynamic_upload': self.dynamic_upload,
                'dynamic_upload_by_name':self.dynamic_upload_by_name,
                'host_transfer': self.host_transfer,
                'host_transfer_by_label':self.host_transfer_by_label,
                'cotangent_upload': self.cotangent_upload,
                'index_upload': self.index_upload,
                'index_upload_by_label': self.index_upload_by_label,
                'index_cache_entries': len(self.index_cache),
                'index_cache_hits': self.index_cache_hits,
                'torch_index_cache_entries': len(self.torch_index_cache),
                'torch_index_cache_hits': self.torch_index_cache_hits,
                'torch_input_transfer': self.torch_input_transfer,
                'torch_input_transfer_by_label':
                    self.torch_input_transfer_by_label,
                'torch_host_transfer': self.torch_host_transfer,
                'torch_host_transfer_by_label':
                    self.torch_host_transfer_by_label,
                'torch_cupy_shared_bytes': self.torch_cupy_shared_bytes,
                'torch_contiguous_copy_bytes':self.torch_contiguous_copy_bytes,
                'torch_activation_shared_bytes':
                    self.torch_activation_shared_bytes,
                'geometry_cost': geometry,
                'geometry_backend':getattr(self,'geometry_backend',None),
                'geometry_device':getattr(self,'geometry_device',None),
                'solve_calls': self.solve_calls,
                'bank_calls': self.bank_calls,
                'contexts': self.contexts, 'ticks_per_context': self.ticks,
                'neurons': self.neurons, 'muscles': self.muscles,
                'edges': self.edges}

    def device_indices(self, indices, label):
        """Explicitly count and cache index copies used by CUDA gathers."""
        import cupy as cp
        values = np.asarray(indices)
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise ValueError('One-dimensional integer CUDA indices required')
        values = owned(values, np.int64)
        key = (label, values.shape, values.tobytes())
        if key in self.index_cache:
            self.index_cache_hits += 1
            return self.index_cache[key]
        labeled = self.index_upload_by_label.setdefault(label, _transfer_bucket())
        for bucket in (self.index_upload, labeled):
            _add_transfer(bucket, 'requested', values.nbytes)
        try:
            with self.device, self.stream:
                result = cp.asarray(values)
                for bucket in (self.index_upload, labeled):
                    _add_transfer(bucket, 'enqueued', values.nbytes)
                self.stream.synchronize()
        except Exception:
            for bucket in (self.index_upload, labeled):
                _add_transfer(bucket, 'unknown_requested', values.nbytes)
            raise
        for bucket in (self.index_upload, labeled):
            _add_transfer(bucket, 'synchronized', values.nbytes)
        self.index_cache[key] = result
        return result

    def host_copy(self, value, *, progress=None, label='device_output'):
        """Count one synchronous CUDA-to-host copy, including uncertainty."""
        import cupy as cp
        nbytes=int(value.nbytes)
        event=self.host_transfer_by_label.setdefault(label,_transfer_bucket())
        _add_transfer(self.host_transfer, 'requested', nbytes)
        _add_transfer(event,'requested',nbytes)
        if progress is not None:
            progress['host_copy_bytes_requested'] += nbytes
            progress['host_copy_count_requested'] += 1
            progress['host_copy_active_label'] = label
        try:
            with self.device, self.stream:
                result=cp.asnumpy(value, stream=self.stream)
        except Exception:
            _add_transfer(self.host_transfer, 'unknown_requested', nbytes)
            _add_transfer(event,'unknown_requested',nbytes)
            raise
        for state in ('enqueued','synchronized'):
            _add_transfer(self.host_transfer, state, nbytes)
            _add_transfer(event,state,nbytes)
        if progress is not None:
            progress['host_copy_bytes_completed'] += nbytes
            progress['host_copy_count_completed'] += 1
            progress['host_copy_active_label'] = None
        return result

    def rescue_device_state(self, device_result, indices=None):
        """Materialize available numeric buffers without another integration."""
        output = {}
        secondary = []
        for key in (*FIELDS, 'branch_bits'):
            if key not in device_result:
                continue
            try:
                output[key]=self.host_copy(device_result[key],label='rescue_'+key)
            except Exception as error:
                secondary.append({'buffer':key,'error':repr(error)})
        if 'B' in device_result:
            selected=np.arange(1,min(self.samples,100),dtype=np.int64) if indices is None \
                else np.asarray(indices,dtype=np.int64)
            try:
                selected = self.device_indices(selected, 'initial_endpoints')
                output['B_initial99']=self.host_copy(
                    device_result['B'][:,selected,:],label='rescue_B_initial99')
            except Exception as error:
                secondary.append({'buffer':'B_initial99','error':repr(error)})
        return output,secondary

    def _check_vector(self, delta, gain):
        delta = np.asarray(delta, dtype=np.float64)
        gain = np.asarray(gain, dtype=np.float64)
        if (delta.shape != (self.edges,) or
                gain.shape != (self.muscles,) or
                not np.isfinite(delta).all() or
                not np.isfinite(gain).all() or
                np.any((gain < 0) | (gain > 8))):
            raise ValueError('Finite local delta and muscle gain vectors required')
        return np.ascontiguousarray(delta), np.ascontiguousarray(gain)

    def solve(self, delta, gain, *, native_mode=False, derivative=False,
              host=True):
        import cupy as cp
        delta, gain = self._check_vector(delta, gain)
        began = time.perf_counter()
        p = self.problem
        c,n,m,ticks,samples,e = (self.contexts,self.neurons,self.muscles,
                                 self.ticks,self.samples,self.edges)
        progress = {'stage': 'requested', 'completion_class': 'not_enqueued',
                    'forward_completion_class': 'not_enqueued',
                    'decoder_completion_class': 'not_enqueued',
                    'bank_completion_class': 'not_enqueued' if derivative
                        else 'not_requested',
                    'requested_context_rk4_ticks': c*ticks,
                    'completed_context_rk4_ticks': None,
                    'requested_conditional_model_s': c*ticks*DT,
                    'completed_conditional_model_s': None,
                    'requested_decoder_endpoint_ticks': c*(samples-1),
                    'completed_decoder_endpoint_ticks': None,
                    'derivative_bank_requested': bool(derivative),
                    'derivative_bank_complete': False,
                    'requested_bank_edge_stage_ticks':
                        c*ticks*4*e if derivative else 0,
                    'completed_bank_edge_stage_ticks': None,
                    'host_copy_count_requested': 0,
                    'host_copy_count_completed': 0,
                    'host_copy_bytes_requested': 0,
                    'host_copy_bytes_completed': 0,
                    'host_copy_active_label': None,
                    'dynamic_upload_requested': {'count': 0, 'bytes': 0},
                    'dynamic_upload_enqueued': {'count': 0, 'bytes': 0},
                    'dynamic_upload_synchronized': {'count': 0, 'bytes': 0},
                    'dynamic_upload_events':[]}
        d_out = {}
        h_out = {}
        self.solve_calls += 1
        try:
            with self.device, self.stream:
                uploaded=[]
                for name,item in (('delta',delta),('gain',gain)):
                    event={'name':name,'bytes':int(item.nbytes),
                        'requested':True,'enqueued':False,
                        'synchronized':False,'unknown_requested':False}
                    progress['dynamic_upload_events'].append(event)
                    _add_transfer(self.dynamic_upload,'requested',item.nbytes)
                    _add_transfer(self.dynamic_upload_by_name[name],
                                  'requested',item.nbytes)
                    progress['dynamic_upload_requested']['count']+=1
                    progress['dynamic_upload_requested']['bytes']+=item.nbytes
                    uploaded.append(cp.asarray(item))
                    _add_transfer(self.dynamic_upload,'enqueued',item.nbytes)
                    _add_transfer(self.dynamic_upload_by_name[name],
                                  'enqueued',item.nbytes)
                    event['enqueued']=True
                    progress['dynamic_upload_enqueued']['count']+=1
                    progress['dynamic_upload_enqueued']['bytes']+=item.nbytes
                d_delta,d_gain_muscle=uploaded
                self.stream.synchronize()
                progress['dynamic_upload_synchronized'] = progress['dynamic_upload_enqueued'].copy()
                for k in ('count', 'bytes'):
                    self.dynamic_upload['synchronized'][k] += progress['dynamic_upload_synchronized'][k]
                for event in progress['dynamic_upload_events']:
                    if event['enqueued']:
                        event['synchronized']=True
                        _add_transfer(self.dynamic_upload_by_name[event['name']],
                                      'synchronized',event['bytes'])
                data = self.device_data
                d_gain = d_gain_muscle[data['membership']]
                shape = (c,samples,n)
                for key in FIELDS[:4]:
                    d_out[key] = cp.full(shape, cp.nan, dtype=cp.float64)
                d_out['branch_bits'] = cp.full((c,ticks,n), 255, dtype=cp.uint8)
                phi = (cp.full((c,ticks*4,n), cp.nan, dtype=cp.float64)
                       if derivative else cp.empty(1, dtype=cp.float64))
                progress.update(stage='launch_requested', completion_class='unknown',
                                forward_completion_class='unknown')
                _kernel('joint_forward')(((c*n+31)//32,), (32,), (
                    data['stage_current'],data['source_rate'],
                    data['initial_rate'],data['initial_adaptation'],
                    d_gain,d_delta,data['tau'],data['a'],data['threshold'],
                    data['cap'],data['union_ptr'],data['union_cols'],
                    data['union_fit'],data['union_base'],data['union_diff'],
                    *(d_out[k] for k in FIELDS[:4]),d_out['branch_bits'],phi,
                    np.int32(c),np.int32(ticks),np.int32(n),
                    np.int32(p['sources']),np.int32(data['union_base'].size),
                    np.int32(bool(native_mode)),np.int32(bool(derivative))))
                progress['stage'] = 'forward_enqueued'
                self.stream.synchronize()
                progress.update(stage='forward_synchronized',
                    completion_class='known_complete',
                    forward_completion_class='known_complete',
                    completed_context_rk4_ticks=c*ticks,
                    completed_conditional_model_s=c*ticks*DT)
                progress.update(stage='decoder_requested',
                                decoder_completion_class='unknown')
                pooled = (d_out['rate'].reshape(c*samples,n) @ data['pool'].T).reshape(c,samples,m)
                pooled_v = (d_out['rate_sensitivity'].reshape(c*samples,n) @ data['pool'].T).reshape(c,samples,m)
                d_out['activation'] = cp.full((c,samples,m), cp.nan, dtype=cp.float64)
                d_out['activation_sensitivity'] = cp.full((c,samples,m), cp.nan, dtype=cp.float64)
                fraction = -math.expm1(-SAMPLE_DT/MUSCLE_TAU)
                _kernel('muscle_ema')(((c*m+31)//32,), (32,), (
                    pooled,pooled_v,data['initial_activation'],
                    d_out['activation'],d_out['activation_sensitivity'],
                    np.int32(c),np.int32(samples),np.int32(m),np.float64(fraction)))
                d_out['command'] = data['rest'] + .02*(
                    d_out['activation'].reshape(c*samples,m) @ data['moment'].T).reshape(
                    c,samples,p['rest'].size)
                d_out['tendon'] = .2*cp.tanh(d_out['activation'][:,:,data['ltm']]/10.)
                progress['stage'] = 'decoder_enqueued'
                self.stream.synchronize()
                progress['completed_decoder_endpoint_ticks'] = c*(samples-1)
                progress['decoder_completion_class'] = 'known_complete'
                progress['stage'] = 'decoder_synchronized'
                if derivative:
                    B = cp.full((c,samples,e+m), cp.nan, dtype=cp.float64)
                    d_out['B'] = B
                    progress['stage'] = 'bank_launch_requested'
                    progress['bank_completion_class'] = 'unknown'
                    self.bank_calls += 1
                    _kernel('edge_bank')(((c*e+31)//32,), (32,), (
                        data['source_rate'],phi,d_gain,data['tau'],data['pool'],
                        data['edge_rows'],data['edge_cols'],data['edge_muscles'],
                        data['base_fit_weights'],d_delta,B,
                        np.int32(c),np.int32(ticks),np.int32(n),
                        np.int32(p['sources']),np.int32(e),np.int32(m),
                        np.int32(e+m),np.float64(fraction)))
                    B[:,:,e:] = d_out['activation_sensitivity']
                    progress['stage'] = 'bank_enqueued'
                    self.stream.synchronize()
                    progress['derivative_bank_complete'] = True
                    progress['bank_completion_class'] = 'known_complete'
                    progress['completed_bank_edge_stage_ticks'] = c*ticks*4*e
                    progress['stage'] = 'bank_synchronized'
                if host:
                    for key in (*FIELDS, 'branch_bits'):
                        h_out[key] = self.host_copy(d_out[key],progress=progress,
                                                    label=key)
                progress['stage'] = 'host_complete' if host else 'device_complete'
        except Exception as error:
            progress['dynamic_upload_unknown_requested'] = {
                k: progress['dynamic_upload_requested'][k]-
                   progress['dynamic_upload_synchronized'][k]
                for k in ('count', 'bytes')}
            for k in ('count', 'bytes'):
                self.dynamic_upload['unknown_requested'][k] += (
                    progress['dynamic_upload_unknown_requested'][k])
            for event in progress['dynamic_upload_events']:
                if not event['synchronized']:
                    event['unknown_requested']=True
                    _add_transfer(self.dynamic_upload_by_name[event['name']],
                                  'unknown_requested',event['bytes'])
            rescued,secondary=self.rescue_device_state(d_out)
            for key,value in rescued.items():
                h_out.setdefault(key,value)
            for item in secondary:
                error.add_note('Secondary device rescue '+item['buffer']+': '+
                               item['error'])
            progress['host_copy_count_unknown_requested'] = (
                progress['host_copy_count_requested']-
                progress['host_copy_count_completed'])
            progress['host_copy_bytes_unknown_requested'] = (
                progress['host_copy_bytes_requested']-
                progress['host_copy_bytes_completed'])
            error.response_progress = progress.copy()
            error.partial_response = h_out
            error.secondary_copy_errors = secondary
            raise
        return d_out, h_out, {'progress': progress,
            'context_rk4_ticks': c*ticks,
            'conditional_model_s': c*ticks*DT,
            'decoder_endpoint_ticks': c*(samples-1),
            'wall_s': time.perf_counter()-began}


def activation_parameter_map(problem):
    """Columns of B map to one owned muscle each, edges then gains."""
    return np.r_[problem['edge_muscles'],
                 np.arange(problem['pool'].shape[0], dtype=np.int32)]


def activation_cotangent_product(engine, device_result, cotangent):
    """Accumulate per-context loss cotangents on GPU; return only gradients."""
    import cupy as cp
    import torch
    b = device_result['B']
    mapping = activation_parameter_map(engine.problem)
    shape=(engine.contexts, engine.samples, engine.muscles)
    if isinstance(cotangent,torch.Tensor):
        if (cotangent.shape != shape or cotangent.dtype != torch.float64 or
                cotangent.device.type != 'cuda'):
            raise ValueError('CUDA float64 activation cotangent shape mismatch')
        tensor=cotangent.detach()
        if not tensor.is_contiguous():
            tensor=tensor.contiguous()
            engine.torch_contiguous_copy_bytes += int(tensor.numel()*8)
        torch.cuda.synchronize(tensor.device)
        with engine.device, engine.stream:
            d_cotangent=cp.from_dlpack(tensor)
            engine.torch_cupy_shared_bytes += int(tensor.numel()*8)
    else:
        host=np.asarray(cotangent,dtype=np.float64)
        if host.shape != shape or not np.isfinite(host).all():
            raise ValueError('Activation cotangent shape mismatch')
        host=owned(host,np.float64)
        with engine.device, engine.stream:
            _add_transfer(engine.cotangent_upload,'requested',host.nbytes)
            try:
                d_cotangent=cp.asarray(host)
            except Exception:
                _add_transfer(engine.cotangent_upload,'unknown_requested',host.nbytes)
                raise
            _add_transfer(engine.cotangent_upload,'enqueued',host.nbytes)
            try:
                engine.stream.synchronize()
            except Exception:
                _add_transfer(engine.cotangent_upload,'unknown_requested',host.nbytes)
                raise
            _add_transfer(engine.cotangent_upload,'synchronized',host.nbytes)
    if d_cotangent.shape != shape:
        raise ValueError('Activation cotangent shape mismatch')
    with engine.device, engine.stream:
        mapping = engine.device_indices(mapping, 'parameter_muscles')
        gradient = cp.sum(b*d_cotangent[:,:,mapping], axis=1)
        engine.stream.synchronize()
        return engine.host_copy(gradient,label='loss_gradient')


def initial_activation_bank(engine, device_result, indices):
    index = np.asarray(indices)
    if (index.ndim != 1 or index.dtype.kind not in 'iu' or
            np.any(index < 0) or np.any(index >= engine.samples)):
        raise ValueError('Valid initial endpoint indices required')
    with engine.device, engine.stream:
        index = engine.device_indices(index, 'initial_endpoints')
        result = engine.host_copy(device_result['B'][:, index, :],
                                  label='initial_B')
    return result
