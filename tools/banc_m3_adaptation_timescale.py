"""E134 native CUDA response with per-muscle adaptation timescale sensitivities.

Recorded four-stage inputs are conditional data. This module does not advance a
network or body and does not patch the frozen E128 CUDA factory.
"""

# ruff: noqa: BLE001

from __future__ import annotations

import copy
import math
import time

import numpy as np

from tools import banc_m3_joint_adaptive_recruitment as neural
from tools.banc_m3_dense_adaptive import DT, MUSCLE_TAU, SAMPLE_DT

FIELDS = (*neural.FIELDS, 'rate_log_tau_sensitivity',
          'adaptation_log_tau_sensitivity',
          'activation_log_tau_sensitivity')
FORWARD_FIELDS = ('rate', 'adaptation', 'rate_sensitivity',
                  'adaptation_sensitivity', 'rate_log_tau_sensitivity',
                  'adaptation_log_tau_sensitivity')
LOG_TAU_LOWER = -math.log(3)
LOG_TAU_UPPER = math.log(10)

CUDA_SOURCE = r'''
__device__ __forceinline__ unsigned char rhs_tau(
    double r,double z,double v,double w,double u,double h,
    double input,double gain,double membrane_tau,double adapt_tau,
    double a,double threshold,double cap,
    double *dr,double *dz,double *dv,double *dw,double *du,double *dh,
    double *prime) {
    double x=a*(input-threshold-z)/cap;
    double t=tanh(x);
    unsigned char positive=t>0.;
    double phi=positive ? cap*t : 0.;
    *prime=positive ? a*(1.-t*t) : 0.;
    *dr=(phi-r)/membrane_tau;
    *dz=(gain*r-z)/adapt_tau;
    *dv=(-*prime*w-v)/membrane_tau;
    *dw=(r+gain*v-w)/adapt_tau;
    *du=(-*prime*h-u)/membrane_tau;
    *dh=(gain*u-h)/adapt_tau-(gain*r-z)/adapt_tau;
    return positive;
}
extern "C" __global__ void prepare_vectors(
    const double *delta,const double *log_tau,
    double *prepared_expm1,double *adapt_tau,int edges,int muscles) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<edges) prepared_expm1[index]=expm1(delta[index]);
    if(index<muscles) adapt_tau[index]=0.3*exp(log_tau[index]);
}
extern "C" __global__ void timescale_forward(
    const float *native,const float *source,const double *initial_r,
    const double *initial_z,const double *gain,const double *prepared_expm1,
    const double *adapt_tau,const int *membership,
    const double *membrane_tau,const double *a,const double *threshold,
    const double *cap,const int *ptr,const int *cols,const int *fit,
    const double *base,const double *diff,
    double *out_r,double *out_z,double *out_v,double *out_w,
    double *out_u,double *out_h,unsigned char *branch,double *prime,
    int contexts,int ticks,int neurons,int sources,int edges,int native_mode,
    int record_prime) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*neurons) return;
    int c=index/neurons,n=index%neurons,samples=ticks/20+1;
    int endpoint=c*samples*neurons+n;
    double r=initial_r[index],z=initial_z[index],v=0.,w=0.,u=0.,h=0.;
    double g=gain[n],t=membrane_tau[n],T=adapt_tau[membership[n]];
    double slope=a[n],th=threshold[n],capn=cap[n];
    out_r[endpoint]=r;out_z[endpoint]=z;out_v[endpoint]=v;
    out_w[endpoint]=w;out_u[endpoint]=u;out_h[endpoint]=h;
    for(int tick=0;tick<ticks;++tick) {
      double k[4][6]; unsigned char bits=0;
      for(int stage=0;stage<4;++stage) {
        int stage_index=(c*ticks+tick)*4+stage;
        double input=(double)native[stage_index*neurons+n];
        if(!native_mode) {
          for(int j=ptr[n];j<ptr[n+1];++j) {
            double s=(double)source[stage_index*sources+cols[j]];
            input+=s*diff[c*edges+j];
            if(fit[j]>=0) input+=s*base[j]*prepared_expm1[fit[j]];
          }
        }
        double rr=r,zz=z,vv=v,ww=w,uu=u,hh=h;
        if(stage>0) {
          double step=stage==3 ? 0.00025 : 0.000125;
          rr+=step*k[stage-1][0];zz+=step*k[stage-1][1];
          vv+=step*k[stage-1][2];ww+=step*k[stage-1][3];
          uu+=step*k[stage-1][4];hh+=step*k[stage-1][5];
        }
        double p;
        bits|=rhs_tau(rr,zz,vv,ww,uu,hh,input,g,t,T,slope,th,capn,
            &k[stage][0],&k[stage][1],&k[stage][2],&k[stage][3],
            &k[stage][4],&k[stage][5],&p)<<stage;
        if(record_prime) prime[stage_index*neurons+n]=p;
      }
      branch[(c*ticks+tick)*neurons+n]=bits;
      r+=0.00025*(k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])/6.;
      z+=0.00025*(k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])/6.;
      v+=0.00025*(k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])/6.;
      w+=0.00025*(k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])/6.;
      u+=0.00025*(k[0][4]+2*k[1][4]+2*k[2][4]+k[3][4])/6.;
      h+=0.00025*(k[0][5]+2*k[1][5]+2*k[2][5]+k[3][5])/6.;
      if((tick+1)%20==0) {
        int next=endpoint+(tick+1)/20*neurons;
        out_r[next]=r;out_z[next]=z;out_v[next]=v;
        out_w[next]=w;out_u[next]=u;out_h[next]=h;
      }
    }
}
extern "C" __global__ void timescale_edge_bank(
    const float *source,const double *prime,const double *gain,
    const double *membrane_tau,const double *adapt_tau,
    const double *pool,const int *rows,const int *cols,const int *muscles,
    const double *base,const double *prepared_expm1,double *B,
    int contexts,int ticks,int neurons,int sources,int edge_count,
    int muscle_count,int params,double fraction) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*edge_count) return;
    int c=index/edge_count,e=index%edge_count;
    int n=rows[e],m=muscles[e],samples=ticks/20+1;
    double p=0.,q=0.,act=0.,g=gain[n],t=membrane_tau[n],T=adapt_tau[m];
    double weight=base[e]*(1.+prepared_expm1[e]);
    double factor=pool[m*neurons+n];
    B[(c*samples)*params+e]=0.;
    for(int tick=0;tick<ticks;++tick) {
      double kp[4],kq[4];
      for(int stage=0;stage<4;++stage) {
        int stage_index=(c*ticks+tick)*4+stage;
        double pp=p,qq=q;
        if(stage>0) {
          double step=stage==3 ? 0.00025 : 0.000125;
          pp+=step*kp[stage-1];qq+=step*kq[stage-1];
        }
        double ph=prime[stage_index*neurons+n];
        double drive=weight*(double)source[stage_index*sources+cols[e]];
        kp[stage]=(ph*(drive-qq)-pp)/t;
        kq[stage]=(g*pp-qq)/T;
      }
      p+=0.00025*(kp[0]+2*kp[1]+2*kp[2]+kp[3])/6.;
      q+=0.00025*(kq[0]+2*kq[1]+2*kq[2]+kq[3])/6.;
      if((tick+1)%20==0) {
        act+=fraction*(factor*p-act);
        B[(c*samples+(tick+1)/20)*params+e]=act;
      }
    }
}
extern "C" __global__ void timescale_muscle_ema(
    const double *pooled,const double *pooled_v,const double *pooled_u,
    const double *initial,double *activation,double *sensitivity,
    double *tau_sensitivity,int contexts,int samples,int muscles,
    double fraction) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=contexts*muscles) return;
    int c=index/muscles,m=index%muscles,first=c*samples*muscles+m;
    double value=initial[index],v=0.,u=0.;
    activation[first]=value;sensitivity[first]=v;tau_sensitivity[first]=u;
    for(int sample=1;sample<samples;++sample) {
      int offset=first+sample*muscles;
      value+=fraction*(pooled[offset]-value);
      v+=fraction*(pooled_v[offset]-v);
      u+=fraction*(pooled_u[offset]-u);
      activation[offset]=value;sensitivity[offset]=v;tau_sensitivity[offset]=u;
    }
}

'''

_KERNEL = {}


def _kernel(name):
    import cupy as cp
    if name not in _KERNEL:
        _KERNEL[name] = cp.RawKernel(CUDA_SOURCE, name, options=('--std=c++11',))
    return _KERNEL[name]


def activation_parameter_map(problem):
    """B columns: fitted edge's muscle, then gain and logtau per muscle."""
    m = problem['pool'].shape[0]
    return np.r_[problem['edge_muscles'], np.arange(m, dtype=np.int32),
                 np.arange(m, dtype=np.int32)]


class CudaAdaptationTimescaleEngine(neural.CudaJointAdaptiveEngine):
    """Own the E134 kernels while reusing E128 owned input and transfer APIs."""

    def __init__(self, mapping, motor_indices=None, muscle_indices=None,
                 fit_indices=None):
        super().__init__(mapping, motor_indices, muscle_indices, fit_indices)
        self.dynamic_upload_by_name['log_tau'] = neural._transfer_bucket()
        self.last_progress = None

    def metadata(self):
        result = super().metadata()
        result['backend'] = 'cupy_cuda_native_four_stage_float64_adaptation_timescale'
        result['parameter_columns'] = self.edges+2*self.muscles
        result['last_progress'] = copy.deepcopy(self.last_progress)
        return result

    def _check_timescale_vectors(self, delta, gain, log_tau):
        # Validate the new variable before allocating or requesting GPU work.
        delta, gain = self._check_vector(delta, gain)
        raw = np.asarray(log_tau)
        if (raw.shape != (self.muscles,) or raw.dtype.kind not in 'fiu'
                or not np.isfinite(raw).all()
                or np.any(raw < LOG_TAU_LOWER) or np.any(raw > LOG_TAU_UPPER)):
            raise ValueError('Finite bounded per-muscle log adaptation tau required')
        return (neural.owned(delta, np.float64), neural.owned(gain, np.float64),
                neural.owned(raw, np.float64))

    def rescue_device_state(self, device_result, indices=None):
        output, secondary = {}, []
        for key in (*FIELDS, 'branch_bits'):
            if key not in device_result:
                continue
            try:
                output[key] = self.host_copy(device_result[key], label='rescue_'+key)
            except BaseException as error:
                secondary.append({'buffer': key, 'error': repr(error)})
        if 'B' in device_result:
            selected = (np.arange(1, min(self.samples, 100), dtype=np.int64)
                        if indices is None else np.asarray(indices, dtype=np.int64))
            try:
                selected = self.device_indices(selected, 'initial_endpoints')
                output['B_initial99'] = self.host_copy(
                    device_result['B'][:, selected, :], label='rescue_B_initial99')
            except BaseException as error:
                secondary.append({'buffer': 'B_initial99', 'error': repr(error)})
        return output, secondary

    def solve(self, delta, gain, log_tau, *, native_mode=False,
              derivative=False, host=True):
        delta, gain, log_tau = self._check_timescale_vectors(delta, gain, log_tau)
        import cupy as cp

        began = time.perf_counter()
        p = self.problem
        c, n, m, ticks, samples, e = (self.contexts, self.neurons,
            self.muscles, self.ticks, self.samples, self.edges)
        progress = {'stage': 'requested', 'completion_class': 'not_enqueued',
            'forward_completion_class': 'not_enqueued',
            'decoder_completion_class': 'not_enqueued',
            'bank_completion_class': 'not_enqueued' if derivative else 'not_requested',
            'prepare_completion_class': 'not_enqueued',
            'requested_context_rk4_ticks': c*ticks,
            'completed_context_rk4_ticks': None,
            'requested_conditional_model_s': c*ticks*DT,
            'completed_conditional_model_s': None,
            'requested_decoder_endpoint_ticks': c*(samples-1),
            'completed_decoder_endpoint_ticks': None,
            'derivative_bank_requested': bool(derivative),
            'derivative_bank_complete': False,
            'requested_bank_edge_stage_ticks': c*ticks*4*e if derivative else 0,
            'completed_bank_edge_stage_ticks': None,
            'host_copy_count_requested': 0, 'host_copy_count_completed': 0,
            'host_copy_bytes_requested': 0, 'host_copy_bytes_completed': 0,
            'host_copy_active_label': None,
            'dynamic_upload_requested': {'count': 0, 'bytes': 0},
            'dynamic_upload_enqueued': {'count': 0, 'bytes': 0},
            'dynamic_upload_synchronized': {'count': 0, 'bytes': 0},
            'dynamic_upload_events': [], 'kernel_events': []}
        d_out, h_out = {}, {}
        self.solve_calls += 1

        def event(name):
            value = {'name': name, 'requested': True, 'enqueued': False,
                     'synchronized': False, 'unknown_requested': False}
            progress['kernel_events'].append(value)
            return value

        def synchronized(value):
            value['synchronized'] = True
            value['unknown_requested'] = False

        try:
            with self.device, self.stream:
                uploaded = []
                for name, item in (('delta', delta), ('gain', gain),
                                   ('log_tau', log_tau)):
                    receipt = {'name': name, 'bytes': int(item.nbytes),
                        'requested': True, 'enqueued': False,
                        'synchronized': False, 'unknown_requested': False}
                    progress['dynamic_upload_events'].append(receipt)
                    for bucket in (self.dynamic_upload,
                                   self.dynamic_upload_by_name[name]):
                        neural._add_transfer(bucket, 'requested', item.nbytes)
                    progress['dynamic_upload_requested']['count'] += 1
                    progress['dynamic_upload_requested']['bytes'] += item.nbytes
                    uploaded.append(cp.asarray(item))
                    for bucket in (self.dynamic_upload,
                                   self.dynamic_upload_by_name[name]):
                        neural._add_transfer(bucket, 'enqueued', item.nbytes)
                    receipt['enqueued'] = True
                    progress['dynamic_upload_enqueued']['count'] += 1
                    progress['dynamic_upload_enqueued']['bytes'] += item.nbytes
                d_delta, d_gain_muscle, d_log_tau = uploaded
                self.stream.synchronize()
                progress['dynamic_upload_synchronized'] = (
                    progress['dynamic_upload_enqueued'].copy())
                for key in ('count', 'bytes'):
                    self.dynamic_upload['synchronized'][key] += (
                        progress['dynamic_upload_synchronized'][key])
                for receipt in progress['dynamic_upload_events']:
                    receipt['synchronized'] = True
                    neural._add_transfer(
                        self.dynamic_upload_by_name[receipt['name']],
                        'synchronized', receipt['bytes'])

                data = self.device_data
                d_gain = d_gain_muscle[data['membership']]
                prep = event('prepare')
                progress['prepare_completion_class'] = 'unknown'
                prepared_expm1 = cp.empty(e, dtype=cp.float64)
                adapt_tau = cp.empty(m, dtype=cp.float64)
                _kernel('prepare_vectors')(((max(e, m)+31)//32,), (32,), (
                    d_delta, d_log_tau, prepared_expm1, adapt_tau,
                    np.int32(e), np.int32(m)))
                prep['enqueued'] = True
                progress['stage'] = 'prepare_enqueued'
                self.stream.synchronize()
                synchronized(prep)
                progress['prepare_completion_class'] = 'known_complete'
                progress['stage'] = 'prepare_synchronized'

                shape = (c, samples, n)
                forward = event('forward')
                progress.update(stage='forward_requested',
                    completion_class='unknown', forward_completion_class='unknown')
                for key in FORWARD_FIELDS:
                    d_out[key] = cp.full(shape, cp.nan, dtype=cp.float64)
                d_out['branch_bits'] = cp.full((c, ticks, n), 255, dtype=cp.uint8)
                prime = (cp.full((c, ticks*4, n), cp.nan, dtype=cp.float64)
                         if derivative else cp.empty(1, dtype=cp.float64))
                _kernel('timescale_forward')(((c*n+31)//32,), (32,), (
                    data['stage_current'], data['source_rate'],
                    data['initial_rate'], data['initial_adaptation'],
                    d_gain, prepared_expm1, adapt_tau, data['membership'],
                    data['tau'], data['a'], data['threshold'], data['cap'],
                    data['union_ptr'], data['union_cols'], data['union_fit'],
                    data['union_base'], data['union_diff'],
                    *(d_out[k] for k in FORWARD_FIELDS),
                    d_out['branch_bits'], prime, np.int32(c), np.int32(ticks),
                    np.int32(n), np.int32(p['sources']),
                    np.int32(data['union_base'].size),
                    np.int32(bool(native_mode)), np.int32(bool(derivative))))
                forward['enqueued'] = True
                progress['stage'] = 'forward_enqueued'
                self.stream.synchronize()
                synchronized(forward)
                progress.update(stage='forward_synchronized',
                    completion_class='known_complete',
                    forward_completion_class='known_complete',
                    completed_context_rk4_ticks=c*ticks,
                    completed_conditional_model_s=c*ticks*DT)

                decoder = event('decoder')
                progress.update(stage='decoder_requested',
                                decoder_completion_class='unknown')
                pooled = (d_out['rate'].reshape(c*samples, n)
                          @ data['pool'].T).reshape(c, samples, m)
                pooled_v = (d_out['rate_sensitivity'].reshape(c*samples, n)
                            @ data['pool'].T).reshape(c, samples, m)
                pooled_u = (d_out['rate_log_tau_sensitivity'].reshape(c*samples, n)
                            @ data['pool'].T).reshape(c, samples, m)
                for key in ('activation', 'activation_sensitivity',
                            'activation_log_tau_sensitivity'):
                    d_out[key] = cp.full((c, samples, m), cp.nan, dtype=cp.float64)
                fraction = -math.expm1(-SAMPLE_DT/MUSCLE_TAU)
                _kernel('timescale_muscle_ema')(((c*m+31)//32,), (32,), (
                    pooled, pooled_v, pooled_u, data['initial_activation'],
                    d_out['activation'], d_out['activation_sensitivity'],
                    d_out['activation_log_tau_sensitivity'],
                    np.int32(c), np.int32(samples), np.int32(m),
                    np.float64(fraction)))
                d_out['command'] = data['rest'] + .02*(
                    d_out['activation'].reshape(c*samples, m) @ data['moment'].T
                    ).reshape(c, samples, p['rest'].size)
                d_out['tendon'] = .2*cp.tanh(
                    d_out['activation'][:, :, data['ltm']]/10.)
                decoder['enqueued'] = True
                progress['stage'] = 'decoder_enqueued'
                self.stream.synchronize()
                synchronized(decoder)
                progress['completed_decoder_endpoint_ticks'] = c*(samples-1)
                progress['decoder_completion_class'] = 'known_complete'
                progress['stage'] = 'decoder_synchronized'

                if derivative:
                    bank = event('bank')
                    progress.update(stage='bank_requested',
                                    bank_completion_class='unknown')
                    B = cp.full((c, samples, e+2*m), cp.nan, dtype=cp.float64)
                    d_out['B'] = B
                    self.bank_calls += 1
                    _kernel('timescale_edge_bank')(((c*e+31)//32,), (32,), (
                        data['source_rate'], prime, d_gain, data['tau'],
                        adapt_tau, data['pool'], data['edge_rows'],
                        data['edge_cols'], data['edge_muscles'],
                        data['base_fit_weights'], prepared_expm1, B,
                        np.int32(c), np.int32(ticks), np.int32(n),
                        np.int32(p['sources']), np.int32(e), np.int32(m),
                        np.int32(e+2*m), np.float64(fraction)))
                    B[:, :, e:e+m] = d_out['activation_sensitivity']
                    B[:, :, e+m:] = d_out['activation_log_tau_sensitivity']
                    bank['enqueued'] = True
                    progress['stage'] = 'bank_enqueued'
                    self.stream.synchronize()
                    synchronized(bank)
                    progress.update(stage='bank_synchronized',
                        derivative_bank_complete=True,
                        bank_completion_class='known_complete',
                        completed_bank_edge_stage_ticks=c*ticks*4*e)
                if host:
                    for key in (*FIELDS, 'branch_bits'):
                        h_out[key] = self.host_copy(d_out[key], progress=progress,
                                                    label=key)
                progress['stage'] = 'host_complete' if host else 'device_complete'
        except BaseException as error:
            progress['dynamic_upload_unknown_requested'] = {
                key: progress['dynamic_upload_requested'][key]
                     - progress['dynamic_upload_synchronized'][key]
                for key in ('count', 'bytes')}
            for key in ('count', 'bytes'):
                self.dynamic_upload['unknown_requested'][key] += (
                    progress['dynamic_upload_unknown_requested'][key])
            for receipt in progress['dynamic_upload_events']:
                if not receipt['synchronized']:
                    receipt['unknown_requested'] = True
                    neural._add_transfer(
                        self.dynamic_upload_by_name[receipt['name']],
                        'unknown_requested', receipt['bytes'])
            for receipt in progress['kernel_events']:
                receipt['unknown_requested'] = not receipt['synchronized']
            try:
                rescued, secondary = self.rescue_device_state(d_out)
            except BaseException as rescue_error:
                rescued, secondary = {}, [{'buffer': 'rescue_protocol',
                                            'error': repr(rescue_error)}]
            for key, value in rescued.items():
                h_out.setdefault(key, value)
            for item in secondary:
                error.add_note('Secondary device rescue '+item['buffer']+': '
                               +item['error'])
            progress['host_copy_count_unknown_requested'] = (
                progress['host_copy_count_requested']
                - progress['host_copy_count_completed'])
            progress['host_copy_bytes_unknown_requested'] = (
                progress['host_copy_bytes_requested']
                - progress['host_copy_bytes_completed'])
            self.last_progress = copy.deepcopy(progress)
            error.response_progress = copy.deepcopy(progress)
            error.partial_response = h_out
            error.secondary_copy_errors = secondary
            raise
        self.last_progress = copy.deepcopy(progress)
        return d_out, h_out, {'progress': progress,
            'context_rk4_ticks': c*ticks,
            'conditional_model_s': c*ticks*DT,
            'decoder_endpoint_ticks': c*(samples-1),
            'wall_s': time.perf_counter()-began}
