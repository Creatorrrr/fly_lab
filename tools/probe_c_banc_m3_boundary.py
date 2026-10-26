"""Check reduced-model numerical fidelity before fitting its parameters."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json


def rk4(rate, adaptation, current, weights, dt):
    def f(r, z, b):
        activation = np.maximum(200 * np.tanh((b + weights @ r - 7.5-z)/200), 0)
        return (activation-r)/.02, (8*r-z)/.3

    b0,b1,b2,b3=current if np.ndim(current)==2 else (current,)*4
    k1,a1=f(rate,adaptation,b0)
    k2,a2=f(rate+dt/2*k1,adaptation+dt/2*a1,b1)
    k3,a3=f(rate+dt/2*k2,adaptation+dt/2*a2,b2)
    k4,a4=f(rate+dt*k3,adaptation+dt*a3,b3)
    return rate+dt/6*(k1+2*k2+2*k3+k4), adaptation+dt/6*(a1+2*a2+2*a3+a4)


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    d=np.load(source/'signals.npz')
    boundary_dt=float(np.diff(d['time_s'])[0])
    stride=round(.005/boundary_dt)
    if stride not in (1,5) or not np.allclose(np.diff(d['time_s']),boundary_dt):
        raise ValueError('Unsupported boundary sample times')
    write_json(out/'spec.json',{
        'experiment':'M3-E08-boundary-fidelity',
        'hypothesis':'Causal left-held external boundary current approximates the selected full-network dynamics sufficiently for a reduced calibration screen',
        'competing_explanation':'Fast outside dynamics and recurrent amplification invalidate the reduced training model',
        'intervention':'Unchanged source parameters; native .25ms RK4, boundary-resolution RK4, and whole5ms RK4 with stage-time current samples. No future sample applied at an earlier stage.',
        'boundary_dt_s':boundary_dt,
        'comparator':'E07 full-network recorded rate and adaptation',
        'state_contract':'One-interval resets are explicitly numerical diagnostics only. Separate continuous replay has exactly one initial state and no resets.',
        'screen':'Native one-interval rate RMS<=1 model unit and p99 absolute error<=5; coarse/native one-interval RMS difference<=1. Also report continuous deviation; no autonomous gait claim.',
        'limits':'CPU only, <=120wall sec, no parameter fitting',
        'source_sha256':sha256(source/'signals.npz'), 'tool_sha256':sha256(__file__),
    })
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    w=csr_matrix(d['internal_weights'].astype(float))
    saved_r,saved_z=d['rate'][::stride].astype(float),d['adaptation'][::stride].astype(float)
    boundary=d['outside_current'][:-1].astype(float)+d['drive']
    start=time.perf_counter()
    paths={}
    results={}
    for label,substeps in (('native',round(boundary_dt/.00025)),('coarse',1),('rk4_control',1)):
        one_r,one_z=[],[]
        continuous_r,continuous_z=[saved_r[0].copy()],[saved_z[0].copy()]
        for t in range(len(saved_r)-1):
            if time.perf_counter()-start>120:
                raise TimeoutError('Boundary fidelity wall limit')
            r,z=saved_r[t].copy(),saved_z[t].copy()
            cr,cz=continuous_r[-1].copy(),continuous_z[-1].copy()
            bs=boundary[t*stride:(t+1)*stride]
            if label=='rk4_control':
                stages=bs[[0,stride//2,stride//2,stride-1]]
                r,z=rk4(r,z,stages,w,.005)
                cr,cz=rk4(cr,cz,stages,w,.005)
            else:
                for b in bs:
                    for _ in range(substeps):
                        r,z=rk4(r,z,b,w,boundary_dt/substeps)
                        cr,cz=rk4(cr,cz,b,w,boundary_dt/substeps)
            one_r.append(r);one_z.append(z)
            continuous_r.append(cr);continuous_z.append(cz)
        one_r=np.asarray(one_r);continuous_r=np.asarray(continuous_r)
        error=one_r-saved_r[1:]
        results[label]={
            'one_interval_rate_rms':float(np.sqrt(np.mean(error**2))),
            'one_interval_abs_error_p99':float(np.percentile(abs(error),99)),
            'continuous_rate_rms':float(np.sqrt(np.mean((continuous_r-saved_r)**2))),
            'one_interval_adaptation_rms':float(np.sqrt(np.mean((np.asarray(one_z)-saved_z[1:])**2))),
        }
        paths[label+'_one_rate']=one_r
        paths[label+'_continuous_rate']=continuous_r
        paths[label+'_continuous_adaptation']=np.asarray(continuous_z)
    delta=float(np.sqrt(np.mean((paths['native_one_rate']-paths['coarse_one_rate'])**2)))
    stage_delta=float(np.sqrt(np.mean((paths['native_one_rate']-paths['rk4_control_one_rate'])**2)))
    passed=results['native']['one_interval_rate_rms']<=1 and results['native']['one_interval_abs_error_p99']<=5 and delta<=1
    np.savez_compressed(out/'replays.npz',**paths)
    report={'experiment':'M3-E08-boundary-fidelity','run_status':'completed',
        'validity':'valid_numerical_diagnostic','hypothesis_verdict':'supported' if passed else 'refuted_at_5ms_boundary_resolution',
        'screen_passed':passed,'arms':results,'coarse_native_one_interval_rms':delta,
        'whole_control_native_one_interval_rms':stage_delta,
        'whole_control_solver_screen_passed':stage_delta<=1,
        'wall_s':time.perf_counter()-start,'new_full_neural_model_s':0,
        'adoption':'not_applicable','milestone_outcome':'pending'}
    write_json(out/'result.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
