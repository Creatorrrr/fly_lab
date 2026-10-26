"""Test one smooth-gradient continuation from E96 with native final readout."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flygym_demo.complex_terrain import FlyBodyPreprogrammedSteps

from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.fit_c_banc_m3_native_all_leg import (
    own_leg_targets,
    project_efficacies,
    reference_at_phase,
)
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks
from tools.verify_c_banc_observation import exact


def training_forcing(z,kappa=None):
    """Smooth only training gradients; None or zero is the exact native ReLU."""
    import torch

    if kappa is None or kappa==0:
        return torch.relu(z)
    if kappa<=0:
        raise ValueError('Positive softplus temperature required')
    return kappa*torch.nn.functional.softplus(z/kappa)


def native_shape_screen(final_metrics,baseline_metrics):
    """The adoption screen consumes only exact native float32-edge metrics."""
    screen={'q_ratio':final_metrics['q_rms']/baseline_metrics['q_rms'],
        'foot_ratio':final_metrics['foot_rms']/baseline_metrics['foot_rms'],
        'RM_row29_ratio':final_metrics['RM_row29_rms']/baseline_metrics['RM_row29_rms'],
        'per_leg_q_ratios':(np.asarray(final_metrics['per_leg_q_rms'])
            /np.asarray(baseline_metrics['per_leg_q_rms'])).tolist()}
    screen['supported']=bool(screen['q_ratio']<=.8 and screen['foot_ratio']<=.8
        and screen['RM_row29_ratio']<=.7
        and all(value<=1.05 for value in screen['per_leg_q_ratios']))
    return screen


def observed_roster(cal, data_ids, meta_ids, meta_sources):
    """Use only the calibration's 48 sources, never extra diagnostic readouts."""
    sources=np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']])
    motor=cal['motor_ids']
    ids=np.asarray(data_ids)
    if (len(sources)!=48 or len(motor)!=391 or len(ids)!=445
            or len(np.unique(ids))!=len(ids)
            or not np.array_equal(ids,meta_ids)
            or len(meta_sources)!=54
            or not np.all(np.isin(sources,meta_sources))):
        raise ValueError('Exact E95 calibration48 sources and diagnostic54 roster required')
    lookup={int(value):i for i,value in enumerate(ids)}
    return sources,motor,np.array([lookup[int(i)] for i in sources]),np.array(
        [lookup[int(i)] for i in motor])


def checked_initial_state(prepared, muscle_count=96):
    """Read actual stored zero state before prepending any t=0 observation."""
    activation=np.asarray(prepared['adapter']['activation'])
    if (activation.shape!=(muscle_count,) or not np.isfinite(activation).all()
            or np.any(activation!=0)):
        raise ValueError('Actual candidate adapter activation must be finite zero')
    network=prepared['network']
    if network['tick']!=0:
        raise ValueError('Actual candidate neural tick must be zero')
    for key in ('rate','drive','adaptation'):
        value=np.asarray(network[key])
        if value.shape!=(158706,) or not np.isfinite(value).all() or np.any(value!=0):
            raise ValueError('Actual candidate neural '+key+' must be finite zero')
    return activation.copy()


def frozen_tripod_offsets(reference_phase):
    """Require the six recorded E85 offsets themselves to match the contract."""
    expected=np.array([0.,.5,0.,.5,0.,.5])
    if reference_phase.ndim!=2 or reference_phase.shape[1]!=6:
        raise ValueError('Six recorded E85 leg phases required')
    delta=np.mod((reference_phase-reference_phase[:,[0]])/(2*np.pi),1)
    distance=np.abs(np.mod(delta-expected[None,:]+.5,1)-.5)
    if not np.isfinite(delta).all() or np.max(distance)>1e-10:
        raise ValueError('E85 recorded tripod offsets differ from frozen 0/.5 values')
    return expected


def wall_budget_ok(elapsed_s,prior_wall_s,limit_s):
    if not np.isfinite(prior_wall_s) or prior_wall_s<0:
        raise ValueError('Explicit prior execution wall cost must be nonnegative')
    return prior_wall_s+elapsed_s<=limit_s


def run(root,graph_path,out,*,prior_wall_s=0.):
    if not wall_budget_ok(0,prior_wall_s,180):
        raise TimeoutError('No remaining shared E97 wall allowance')
    # Claim this output before installing a finalizer that may update it.
    out.mkdir(parents=True,exist_ok=False)
    bodies=[]
    began=time.perf_counter()
    try:
        return _run(root,graph_path,out,bodies,prior_wall_s=prior_wall_s)
    finally:
        for body in bodies:
            body.close()
        if (out/'result.json').exists():
            result=json.loads((out/'result.json').read_text())
            result['wall_s']=time.perf_counter()-began
            result['prior_execution_wall_s']=prior_wall_s
            result['cumulative_execution_wall_s']=prior_wall_s+result['wall_s']
            result['resource_limit_met']=wall_budget_ok(result['wall_s'],prior_wall_s,180)
            if not result['resource_limit_met']:
                result['validity']='wall_limit_incomplete'
            write_json(out/'result.json',result)


def _run(root,graph_path,out,bodies,*,prior_wall_s):
    began=time.perf_counter()
    e95=root/'e95-rm-e2-premotor-source'
    e96=root/'e96-observed-native-recruitment'
    body_source=e95/'body'
    e85=root/'e85-space-compatible-cadence-reference'
    spec={'experiment':'M3-E97-softplus-training-native-recruitment',
        'hypothesis':'The exact ReLU gradient-zero trap of LF tibia extensor muscle13 at the E96 best point may be escaped by training-only softplus continuation without changing the native readout.',
        'intervention':'Start exactly at E96 best96x48 theta. Three continuing Adam250-update stages at softplus kappa10,1,.1, each with exact native objective observed before every update and once after all750. Then one native exact L-BFGS-B <=250 actual calls. Original590 positive efficacy/1536 signed existing edges, all other physiology/body fixed.',
        'labels':'Actual simultaneous E95 source/current samples, own-leg quadratic E1 phase minus frozen E85 tripod offsets; all-six bracketed .5-to-last-peak interval. E85 late recorded commands and 84-bin inverse offline only.',
        'objective':'E89 normalized 42joint+6x3foot+6LTM near.2+1e-5 efficacy prior. Smooth forcing only for training gradients, native ReLU for best selection, L-BFGS-B, actual float32 candidate and shape screen.',
        'limits':'Fit<=150wall, shared<=180wall, full neural/physics advance0, one CUDA owner, no body trial or F gate claim.',
        'prior_execution_wall_s':prior_wall_s,
        'e95_calibration_sha256':sha256(e95/'calibration.npz'),
        'e95_signals_sha256':sha256(body_source/'recruitment-signals.npz'),
        'e95_metadata_sha256':sha256(body_source/'recruitment-metadata.npz'),
        'e95_trace_sha256':sha256(body_source/'trace.json'),
        'e95_parent_review_sha256':sha256(body_source/'parent-second-review.json'),
        'e96_best_coefficients_sha256':sha256(e96/'fit-coefficients.npz'),
        'e96_result_sha256':sha256(e96/'result.json'),
        'e96_collapse_diagnostic_sha256':sha256(e96/'parent-muscle-collapse-diagnostic.json'),
        'e85_reference_sha256':sha256(e85/'commands.npz'),
        'causal_helper_sha256':sha256(Path(__file__).with_name('banc_m3_causal_recruitment.py')),
        'static_geometry_helper_sha256':sha256(Path(__file__).with_name('banc_m3_kinematic_fit.py')),
        'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'causal-helper.py').write_bytes(Path(__file__).with_name(
        'banc_m3_causal_recruitment.py').read_bytes())
    cal={key:value.copy() for key,value in np.load(e95/'calibration.npz').items()}
    old_cal={key:value.copy() for key,value in cal.items()}
    data=np.load(body_source/'recruitment-signals.npz')
    meta=np.load(body_source/'recruitment-metadata.npz')
    ref=np.load(e85/'commands.npz')
    graph=GraphStore.load(graph_path)
    original=StateStore.load(root/'e32-core-adaptation/candidate/start')
    prepared=StateStore.load(e95/'candidate-neutral')
    initial_activation=checked_initial_state(prepared)
    exact(original['body'],prepared['body'],'E96 original neutral body')
    for key in ('rate','adaptation','drive','output_mask','tick'):
        exact(original['network'][key],prepared['network'][key],
            'E96 original zero neural '+key)
    sources,motor,source_cols,motor_cols=observed_roster(cal,data['observed_ids'],
        meta['observed_ids'],meta['source_ids'])
    if len(cal['tau_ids'])!=38 or cal['theta'].shape!=(96,47):
        raise ValueError('E95 38tau and legacy47-source fit provenance required')
    times=np.arange(2001)*.005
    np.testing.assert_array_equal(data['time_s'],times[1:])
    raw_rate=data['rate'].astype(float)
    raw_current=(data['recurrent_current']+data['drive']-data['adaptation']).astype(float)
    measured_sources=np.vstack((np.zeros((1,48)),raw_rate[:,source_cols]))
    full_current=np.vstack((np.zeros((1,391)),raw_current[:,motor_cols]))
    if not np.isfinite(measured_sources).all() or not np.isfinite(full_current).all():
        raise ValueError('Finite E95 actual source and MN current required')
    if np.any(data['adaptation'][:,motor_cols]!=0) or np.any(meta['_cell_gain'][motor_cols]!=0):
        raise ValueError('Actual E95 MN adaptation must be zero')
    trace=json.loads((body_source/'trace.json').read_text())['samples']
    if len(trace)!=2001:
        raise ValueError('Complete original E95 2001-control trace required')
    actual_activation=np.vstack((initial_activation[None,:],
        data['muscle_activation'].astype(float)))
    decoded=cal['rest']+.02*actual_activation@cal['moment'].T
    recorded=np.asarray([row['joint_targets_rad'] for row in trace])
    np.testing.assert_allclose(decoded[1:],recorded[1:],rtol=0,atol=1e-6)
    leg_order=tuple(FlyBodyPreprogrammedSteps().legs)
    if leg_order!=('lf','lm','lh','rf','rm','rh'):
        raise ValueError('Frozen six-leg reference order required')
    group_by_leg={'rh':0,'rf':1,'lf':2,'rm':3,'lh':4,'lm':5}
    offsets=frozen_tripod_offsets(ref['phase'])
    phases=np.empty((2001,6))
    last=[]
    for leg,name in enumerate(leg_order):
        ids=cal['core_ids'][(cal['core_groups']==group_by_leg[name])
            &(cal['core_cell_types']==0)]
        if len(ids)!=1:
            raise ValueError('One actual E1 per original leg required')
        col=int(np.flatnonzero(sources==ids[0])[0])
        peaks=quadratic_peaks(times,measured_sources[:,col])
        if len(peaks)<20 or peaks[0]>.5 or peaks[-1]<9.5:
            raise ValueError('Complete actual E95 own-leg E1 peaks required')
        phases[:,leg]=phase_trace(times,peaks)
        last.append(peaks[-1])
    mask=(times>=.5)&(times<min(last))&np.isfinite(phases).all(axis=1)
    target=own_leg_targets(ref,phases,offsets,mask,0)
    reference_train=(ref['time_s']>=5)&(ref['time_s']<8)
    reference_holdout=(ref['time_s']>=8)&(ref['time_s']<10)
    reference_reconstructed=reference_at_phase(ref,
        ref['phase'][reference_holdout,0]/(2*np.pi),0,reference_train)
    reference_rms=float(np.sqrt(np.mean((reference_reconstructed
        -ref['command'][reference_holdout])**2)))
    np.savez_compressed(out/'reference-validation.npz',
        reconstructed=reference_reconstructed,
        heldout_command=ref['command'][reference_holdout],rms=reference_rms)
    if reference_rms>.002:
        raise ValueError('E85 held-out interpolation exceeds fixed .002rad gate')
    base,_=rate_weights(graph)
    base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,motor)
    current=patch_weights(base,cal['edge_ids'],cal['source_values'],cal['values'])
    original_w=base[motor][:,sources].toarray().astype(float)
    current_w=current[motor][:,sources].toarray().astype(float)
    if np.count_nonzero(original_w)!=1536:
        raise ValueError('Exactly1536 original signed source-MN edges required')
    background=full_current-measured_sources@current_w.T
    tau=meta['tau'][motor_cols].astype(float)
    a=meta['a'][motor_cols].astype(float)
    threshold=meta['threshold'][motor_cols].astype(float)
    cap=meta['cap'][motor_cols].astype(float)
    if (not np.allclose(a,1) or not np.allclose(threshold,7.5)
            or not np.allclose(cap,200)
            or not np.all(np.isin(np.round(tau,7),
                np.round([.02,.014046887330582595],7)))):
        raise ValueError('Actual E95 MN native physiology required')
    pool,moment,rest=cal['pool'],cal['moment'],cal['rest']
    membership,ltm=cal['membership'],cal['ltm']
    present=(pool@(original_w!=0).astype(float))>0
    if present.shape!=(96,48) or int(present.sum())!=590:
        raise ValueError('Exactly590 existing anatomical source parameters required')
    selected=np.flatnonzero(mask)[::20]
    body=new_body(original)
    bodies.append(body)
    body_before=body.snapshot()
    geometry=StaticFootGeometry(body,np.repeat(body.d.qpos[None,:],len(selected),axis=0))
    target_feet=geometry.evaluate(target[np.searchsorted(np.flatnonzero(mask),selected)])
    joint_scale=np.maximum(target.std(axis=0),.025)
    foot_scale=np.maximum(target_feet.std(axis=0),.05)
    np.savez_compressed(out/'training-inputs.npz',time_s=times,
        source_ids=sources,motor_ids=motor,source_rate=measured_sources,
        current=full_current,background=background,original_w=original_w,
        current_w=current_w,phase=phases,offsets=offsets,train_mask=mask,
        target=target,target_feet=target_feet,selected=selected,
        joint_scale=joint_scale,foot_scale=foot_scale,
        tau=tau,a=a,threshold=threshold,cap=cap,present=present,
        membership=membership,pool=pool,moment=moment,rest=rest,
        actual_activation=actual_activation,actual_command=decoded)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('E96 CUDA fit required')
    torch.set_num_threads(1)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device='cuda')
    source_t,background_t,original_t=map(tensor,(measured_sources,background,original_w))
    pool_t,moment_t,rest_t=map(tensor,(pool,moment,rest))
    tau_t,a_t,threshold_t,cap_t=map(tensor,(tau,a,threshold,cap))
    target_t,target_feet_t=map(tensor,(target,target_feet))
    joint_scale_t,foot_scale_t=map(tensor,(joint_scale,foot_scale))
    membership_t=torch.as_tensor(membership,device='cuda')
    mask_t=torch.as_tensor(mask,device='cuda')
    selected_t=torch.as_tensor(selected,device='cuda')
    ltm_t=torch.as_tensor(ltm,device='cuda')
    present_t=torch.as_tensor(present,device='cuda')

    def predict(theta=None,*,direct_weights=None,with_feet=True,kappa=None):
        weights=direct_weights if direct_weights is not None else original_t*theta[membership_t]
        current_t=source_t@weights.T+background_t
        z=cap_t*torch.tanh(a_t*(current_t-threshold_t)/cap_t)
        forcing=training_forcing(z,kappa)
        rate=linear_response(forcing,.005,tau_t)
        activation=endpoint_response(rate@pool_t.T,.005,.01)
        command=rest_t+.02*activation@moment_t.T
        tendon=.2*torch.tanh(activation[:,ltm_t]/10)
        feet=geometry.tensor(command[selected_t]) if with_feet else None
        return command,tendon,feet,rate,activation

    with torch.no_grad():
        bc,bt,_,br,ba=predict(direct_weights=tensor(current_w),with_feet=False)
    baseline_command=bc.cpu().numpy()
    transfer_rms=float(np.sqrt(np.mean((baseline_command[mask]-decoded[mask])**2)))
    reference_discrepancy=float(np.sqrt(np.mean((decoded[mask]-target)**2)))
    fidelity=transfer_rms/reference_discrepancy
    np.savez_compressed(out/'baseline-prediction.npz',command=baseline_command,
        tendon=bt.cpu().numpy(),motor_rate=br.cpu().numpy(),
        activation=ba.cpu().numpy(),actual_command=decoded,
        transfer_rms=transfer_rms,reference_rms=reference_discrepancy,ratio=fidelity)
    if not np.isfinite(fidelity) or fidelity>.25:
        write_json(out/'result.json',{'validity':'surrogate_method_failure',
            'baseline_transfer_ratio':fidelity,'actual_neural_s':0,
            'actual_physical_s':0,'wall_s':time.perf_counter()-began})
        return
    actual_tendon=.2*np.tanh(actual_activation[:,ltm]/10)
    baseline_feet=geometry.evaluate(decoded[selected])

    def metrics(command,feet,tendon):
        joint=(command[mask]-target)/joint_scale
        foot=(feet-target_feet)/foot_scale
        tendon_error=(tendon[mask]-.2)/.2
        return {'combined':float(np.mean(joint**2)+np.mean(foot**2)+np.mean(tendon_error**2)),
            'q_rms':float(np.sqrt(np.mean((command[mask]-target)**2))),
            'foot_rms':float(np.sqrt(np.mean((feet-target_feet)**2))),
            'RM_row29_rms':float(np.sqrt(np.mean((command[mask,29]-target[:,29])**2))),
            'per_leg_q_rms':[float(np.sqrt(np.mean((command[mask,7*i:7*i+7]
                -target[:,7*i:7*i+7])**2))) for i in range(6)],
            'joint_normalized':float(np.mean(joint**2)),
            'foot_normalized':float(np.mean(foot**2)),
            'tendon_normalized':float(np.mean(tendon_error**2))}

    baseline_metrics=metrics(decoded,baseline_feet,actual_tendon)
    write_json(out/'baseline-output.json',baseline_metrics)
    start_coefficients=np.load(e96/'fit-coefficients.npz')
    if (start_coefficients['theta'].shape!=(96,48)
            or not np.array_equal(start_coefficients['source_ids'],sources)
            or not np.array_equal(start_coefficients['present'],present)):
        raise ValueError('Exact E96 best96x48 coefficients and source order required')
    initial_theta=start_coefficients['theta'].astype(float)
    if not np.isfinite(initial_theta).all() or np.any(initial_theta[present]<.001) or np.any(initial_theta[present]>10):
        raise ValueError('E96 selected original bounded efficacy start required')
    np.savez_compressed(out/'initial-E96-coefficients.npz',theta=initial_theta,
        source_ids=sources,present=present)
    if not wall_budget_ok(time.perf_counter()-began,prior_wall_s,150):
        raise TimeoutError('E97 preparation left insufficient total wall reserve')
    log_theta=torch.nn.Parameter(tensor(np.log(initial_theta)))
    optimizer=torch.optim.Adam([log_theta],lr=.03)
    fit_began=time.perf_counter()
    history=[]
    best_loss=math.inf
    best=None
    adam_updates=0
    lbfgs_calls=0
    stop='adam_max_updates'

    def objective(theta,*,kappa=None):
        command,tendon,feet,_,_=predict(theta,kappa=kappa)
        joint=((command[mask_t]-target_t)/joint_scale_t).square().mean()
        foot=((feet-target_feet_t)/foot_scale_t).square().mean()
        tendon_loss=((tendon[mask_t]-.2)/.2).square().mean()
        prior=1e-5*(theta[present_t]-1).square().mean()
        return joint+foot+tendon_loss+prior,(joint,foot,tendon_loss,prior)

    native_observations=0
    smooth_gradient_calls=0
    for update in range(751):
        native_observations+=1
        write_json(out/'execution-progress.json',{'native_started_observations':native_observations,
            'smooth_started_gradient_calls':smooth_gradient_calls,
            'lbfgsb_started_loss_gradient_calls':lbfgs_calls,
            'prior_execution_wall_s':prior_wall_s,
            'execution_wall_s':time.perf_counter()-began})
        with torch.no_grad():
            native_loss,native_parts=objective(log_theta.exp(),kappa=None)
        value=float(native_loss.detach())
        if not math.isfinite(value):
            raise ValueError('Nonfinite exact-native observation')
        if value<best_loss:
            best_loss=value
            best=log_theta.detach().clone()
            np.savez_compressed(out/'best-fit-checkpoint.npz',
                best_log_theta=best.cpu().numpy(),best_loss=best_loss,
                adam_updates=adam_updates,lbfgs_calls=lbfgs_calls,
                source_ids=sources,present=present)
        item={'stage':'softplus_adam','update':update,'native_loss':value,
            'native_parts':[float(part.detach()) for part in native_parts],
            'fit_wall_s':time.perf_counter()-fit_began}
        if update==750:
            history.append(item)
            write_json(out/'history.json',history)
            break
        if (time.perf_counter()-fit_began>130
                or not wall_budget_ok(time.perf_counter()-began,prior_wall_s,150)):
            stop='wall_reserve_before_lbfgs'
            history.append(item)
            write_json(out/'history.json',history)
            break
        kappa=(10.,1.,.1)[update//250]
        optimizer.zero_grad(set_to_none=True)
        smooth_gradient_calls+=1
        write_json(out/'execution-progress.json',{'native_started_observations':native_observations,
            'smooth_started_gradient_calls':smooth_gradient_calls,
            'lbfgsb_started_loss_gradient_calls':lbfgs_calls,
            'prior_execution_wall_s':prior_wall_s,
            'execution_wall_s':time.perf_counter()-began})
        loss,parts=objective(log_theta.exp(),kappa=kappa)
        if not math.isfinite(float(loss.detach())):
            raise ValueError('Nonfinite training-only softplus objective')
        item.update(kappa=kappa,smooth_loss=float(loss.detach()),
            smooth_parts=[float(part.detach()) for part in parts])
        history.append(item)
        write_json(out/'history.json',history)
        if update%25==0:
            print(json.dumps(item),flush=True)
        loss.backward()
        optimizer.step()
        adam_updates+=1
        with torch.no_grad():
            log_theta.clamp_(math.log(.001),math.log(10))

    class FitLimit(Exception):
        pass

    lower=np.full((96,48),math.log(.001))
    upper=np.full((96,48),math.log(10))
    start=best.cpu().numpy().copy()
    if stop=='adam_max_updates':
        def scipy_objective(flat):
            nonlocal lbfgs_calls,best_loss,best,stop
            if (lbfgs_calls>=250 or time.perf_counter()-fit_began>150
                    or not wall_budget_ok(time.perf_counter()-began,prior_wall_s,150)):
                stop='lbfgs_call_or_wall_limit'
                raise FitLimit()
            lbfgs_calls+=1
            write_json(out/'execution-progress.json',{
                'native_started_observations':native_observations,
                'smooth_started_gradient_calls':smooth_gradient_calls,
                'lbfgsb_started_loss_gradient_calls':lbfgs_calls,
                'prior_execution_wall_s':prior_wall_s,
                'execution_wall_s':time.perf_counter()-began})
            parameter=torch.tensor(flat.reshape(96,48),dtype=torch.float64,
                device='cuda',requires_grad=True)
            loss,parts=objective(parameter.exp())
            value=float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Nonfinite L-BFGS-B objective')
            loss.backward()
            gradient=parameter.grad.detach().cpu().numpy().reshape(-1)
            if value<best_loss:
                best_loss=value
                best=parameter.detach().clone()
                np.savez_compressed(out/'best-fit-checkpoint.npz',
                    best_log_theta=best.cpu().numpy(),best_loss=best_loss,
                    adam_updates=adam_updates,lbfgs_calls=lbfgs_calls,
                    source_ids=sources,present=present)
            history.append({'stage':'lbfgsb','call':lbfgs_calls,'loss':value,
                'parts':[float(part.detach()) for part in parts],
                'fit_wall_s':time.perf_counter()-fit_began})
            write_json(out/'history.json',history)
            return value,gradient
        try:
            minimize(scipy_objective,start.reshape(-1),method='L-BFGS-B',
                jac=True,bounds=list(zip(lower.reshape(-1),upper.reshape(-1))),
                options={'maxfun':250,'maxiter':250,'ftol':1e-12})
            stop='lbfgs_converged_or_scipy_limit'
        except FitLimit:
            pass
    fit_wall=time.perf_counter()-fit_began
    with torch.no_grad():
        raw_theta=best.exp().cpu().numpy()
        command,tendon,feet,rate,activation=predict(tensor(raw_theta))
    np.savez_compressed(out/'raw-fit-coefficients.npz',
        theta=raw_theta,log_theta=best.cpu().numpy(),best_loss=best_loss,
        adam_updates=adam_updates,lbfgs_calls=lbfgs_calls,
        source_ids=sources,present=present)
    np.savez_compressed(out/'unprojected-prediction.npz',
        command=command.cpu().numpy(),tendon=tendon.cpu().numpy(),
        feet=feet.cpu().numpy(),motor_rate=rate.cpu().numpy(),
        activation=activation.cpu().numpy(),target=target,target_feet=target_feet)
    theta=project_efficacies(raw_theta,present)
    np.savez_compressed(out/'fit-coefficients.npz',theta=theta,
        raw_theta=raw_theta,source_ids=sources,present=present,
        roundoff_projection_max_abs=float(np.max(abs(theta-raw_theta))))
    values=cal['values'].copy()
    positions={int(edge):i for i,edge in enumerate(cal['edge_ids'])}
    fitted_edges=[]
    for row,motor_id in enumerate(motor):
        for edge in range(base.indptr[motor_id],base.indptr[motor_id+1]):
            pre=int(base.indices[edge])
            col=np.searchsorted(sources,pre)
            if col>=len(sources) or sources[col]!=pre or base.data[edge]==0:
                continue
            if edge not in positions:
                raise ValueError('E95 original source-MN edge absent from existing patch')
            values[positions[edge]]=np.float32(base.data[edge]*theta[membership[row],col])
            fitted_edges.append(edge)
    if len(fitted_edges)!=1536:
        raise ValueError('Exactly1536 existing source-MN edges must be fitted')
    changed=patch_weights(base,cal['edge_ids'],cal['source_values'],values)
    if (not np.array_equal(changed.indptr,current.indptr)
            or not np.array_equal(changed.indices,current.indices)
            or not np.array_equal(np.sign(changed.data[fitted_edges]),
                np.sign(base.data[fitted_edges]))):
        raise ValueError('Original CSR topology and signed edges required')
    unchanged=current.copy()
    unchanged.data[fitted_edges]=changed.data[fitted_edges]
    np.testing.assert_array_equal(unchanged.data,changed.data)
    actual_w=changed[motor][:,sources].toarray().astype(float)
    with torch.no_grad():
        fc,ft,ff,fr,fa=predict(direct_weights=tensor(actual_w))
    final_command=fc.cpu().numpy()
    final_feet=ff.cpu().numpy()
    final_tendon=ft.cpu().numpy()
    np.savez_compressed(out/'prediction.npz',command=final_command,
        tendon=final_tendon,feet=final_feet,motor_rate=fr.cpu().numpy(),
        muscle_activation=fa.cpu().numpy(),RM_muscle67_activation=fa.cpu().numpy()[:,67],
        LF_muscle13_activation=fa.cpu().numpy()[:,13],
        actual_float32_source_weights=actual_w,target=target,target_feet=target_feet)
    def lf_extensor_diagnostic(weights,activation_values):
        local_current=measured_sources@weights.T+background
        columns=[int(np.flatnonzero(motor==cell)[0]) for cell in (36295,141845)]
        margins=local_current[:,columns]-threshold[columns]
        values=activation_values[:,13]
        return {'MN_ids':[36295,141845],
            'max_margin_by_MN':margins.max(axis=0).tolist(),
            'fraction_above_threshold_by_MN':(margins>0).mean(axis=0).tolist(),
            'activation_min':float(values.min()),
            'activation_max':float(values.max()),
            'activation_p95_minus_p05':float(np.percentile(values,95)-np.percentile(values,5))}
    raw_w=original_w*raw_theta[membership]
    initial_w=original_w*initial_theta[membership]
    with torch.no_grad():
        _,_,_,_,initial_activation=predict(tensor(initial_theta),with_feet=False)
    write_json(out/'LF-muscle13-native-diagnostic.json',{
        'initial_E96':lf_extensor_diagnostic(initial_w,initial_activation.cpu().numpy()),
        'best_unprojected':lf_extensor_diagnostic(raw_w,activation.cpu().numpy()),
        'final_actual_float32_edges':lf_extensor_diagnostic(actual_w,fa.cpu().numpy()),
        'interpretation':'Exact native ReLU readout only; softplus never enters runtime'})
    final_metrics=metrics(final_command,final_feet,final_tendon)
    screen=native_shape_screen(final_metrics,baseline_metrics)
    result={'experiment':spec['experiment'],
        'validity':'valid_conditional_fit' if wall_budget_ok(
            time.perf_counter()-began,prior_wall_s,180) else 'wall_limit_incomplete',
        'hypothesis_verdict':'conditional_shape_supported' if screen['supported'] else 'not_supported',
        'adoption':'not_applicable','body_validated':False,
        'baseline_transfer_ratio':fidelity,'reference_extraction_rms_rad':reference_rms,
        'baseline_output':baseline_metrics,'fitted_output':final_metrics,
        'shape_screen':screen,'adam_updates':adam_updates,'lbfgs_actual_calls':lbfgs_calls,
        'smooth_gradient_calls':smooth_gradient_calls,
        'native_observations_during_adam':native_observations,
        'final_float32_native_checks':1,
        'best_loss':best_loss,'stop':stop,
        'fit_parameters':int(present.sum()),'fitted_edges':len(fitted_edges),
        'fit_wall_s':fit_wall,'wall_s':time.perf_counter()-began,
        'smooth_started_gradient_calls':smooth_gradient_calls,
        'native_started_observations':native_observations,
        'lbfgsb_started_loss_gradient_calls':lbfgs_calls,
        'actual_neural_s':0,'actual_physical_s':0,
        'roundoff_projection_max_abs':float(np.max(abs(theta-raw_theta))),
        'nonfit_weight_exact':True,'CSR_topology_exact':True,
        'source_sign_exact':True,'all_38_tau_exact':True}
    write_json(out/'result.json',result)
    if not screen['supported'] or result['validity']!='valid_conditional_fit':
        return
    cal['legacy_theta_e89']=cal['theta'].copy()
    cal['legacy_theta_source_ids_e89']=cal['theta_source_ids'].copy()
    cal['theta']=theta
    cal['theta_source_ids']=sources
    cal['values']=values
    for key,value in old_cal.items():
        if key not in ('values','theta','theta_source_ids'):
            exact(cal[key],value,'E97 other calibration '+key)
    np.savez_compressed(out/'calibration.npz',**cal)
    anatomy=json.loads((e95/'muscle-provenance.json').read_text())
    anatomy['E97_softplus_gradient_recruitment']={'source_ids':sources.tolist(),
        'existing_signed_edges':len(fitted_edges),'parameters':int(present.sum()),
        'conditional_only':True,'runtime_reference_phase':False}
    write_json(out/'muscle-provenance.json',anatomy)
    base_trial=EfficacyTrial(graph,original,cal['edge_ids'],cal['source_values'],
        cal['values'],zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial,cal,
        anatomy['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        neutral=session.snapshot()
        exact(original['body'],neutral['body'],'Original E97 neutral body')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(original['network'][key],neutral['network'][key],
                'Original E97 neutral neural '+key)
        exact(body_before,body.snapshot(),'Shadow static geometry leaves body unchanged')
        StateStore.save(out/'candidate-neutral',neutral)
        clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out/'candidate-neutral'))
        try:
            exact(neutral,clone.snapshot(),'E97 whole neutral disk exact')
        finally:
            clone.close()
    finally:
        session.close()
    result['neutral_disk_exact']=True
    result['wall_s']=time.perf_counter()-began
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prior-execution-wall-s',type=float,default=0.)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: '+str(args.out))
    started=time.perf_counter()
    try:
        run(args.root,args.graph,args.out,prior_wall_s=args.prior_execution_wall_s)
    except Exception as error:
        if args.out.exists():
            progress_path=args.out/'execution-progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {
                'native_started_observations':0,
                'smooth_started_gradient_calls':0,
                'lbfgsb_started_loss_gradient_calls':0}
            write_json(args.out/'execution-error.json',{'validity':'incomplete',
                'error':type(error).__name__,'message':str(error),
                **progress,
                'execution_wall_s':time.perf_counter()-started,
                'prior_execution_wall_s':args.prior_execution_wall_s,
                'cumulative_execution_wall_s':args.prior_execution_wall_s+
                    time.perf_counter()-started,
                'actual_neural_s':0,'actual_physical_s':0})
        raise
