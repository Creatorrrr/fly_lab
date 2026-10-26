"""Preserve actual healthy startup while fitting E101 independent motor edges."""

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
        'LF_row6_ratio':final_metrics['LF_row6_rms']/baseline_metrics['LF_row6_rms'],
        'per_leg_q_ratios':(np.asarray(final_metrics['per_leg_q_rms'])
            /np.asarray(baseline_metrics['per_leg_q_rms'])).tolist()}
    screen['supported']=bool(screen['q_ratio']<=.8 and screen['foot_ratio']<=.8
        and screen['LF_row6_ratio']<=.5
        and all(value<=1.05 for value in screen['per_leg_q_ratios']))
    return screen


def startup_grid(times,mask,target_q,target_tendon):
    """Require every original 5ms sample from .005 through .495 seconds."""
    times=np.asarray(times)
    mask=np.asarray(mask)
    target_q=np.asarray(target_q)
    target_tendon=np.asarray(target_tendon)
    expected=np.arange(1,100)*.005
    if (times.shape!=(2001,) or not np.array_equal(times,np.arange(2001)*.005)
            or mask.shape!=(2001,) or mask.dtype!=bool
            or not np.array_equal(times[mask],expected)
            or target_q.shape!=(99,42) or target_tendon.shape!=(99,6)
            or not np.isfinite(target_q).all() or not np.isfinite(target_tendon).all()):
        raise ValueError('Complete finite original99 startup command/tendon samples required')


def startup_screen(command,tendon,target_q,target_tendon,times,mask):
    """Use exact native float32-edge outputs, never smoothed training outputs."""
    startup_grid(times,mask,target_q,target_tendon)
    command=np.asarray(command)
    tendon=np.asarray(tendon)
    if (command.shape!=(2001,42) or tendon.shape!=(2001,6)
            or not np.isfinite(command[mask]).all() or not np.isfinite(tendon[mask]).all()):
        raise ValueError('Complete finite native startup prediction required')
    q=command[mask]-target_q
    ltm=tendon[mask]-target_tendon
    result={'q_RMS_rad':float(np.sqrt(np.mean(q**2))),
        'q_max_abs_rad':float(np.max(abs(q))),
        'LTM_RMS':float(np.sqrt(np.mean(ltm**2))),
        'LTM_max_abs':float(np.max(abs(ltm)))}
    result['supported']=bool(result['q_RMS_rad']<=.02
        and result['q_max_abs_rad']<=.05 and result['LTM_RMS']<=.01
        and result['LTM_max_abs']<=.03)
    return result


def independent_edge_order(graph, motor, sources, original_w, *, expected_count=1572):
    """Map each present dense entry to its one existing signed CSR edge."""
    rows,cols=np.nonzero(original_w)
    edges=[]
    for row,col in zip(rows,cols):
        post=int(motor[row]);pre=int(sources[col])
        positions=np.arange(graph.indptr[post],graph.indptr[post+1])
        selected=positions[graph.indices[positions]==pre]
        if len(selected)!=1 or original_w[row,col]==0:
            raise ValueError('Exactly one existing nonzero signed edge per parameter required')
        edges.append(int(selected[0]))
    if len(edges)!=expected_count or len(np.unique(edges))!=expected_count:
        raise ValueError('Exact independent original source-to-MN edge count required')
    return rows.astype(np.int64),cols.astype(np.int64),np.asarray(edges,dtype=np.int64)


def current_edge_start(original, current):
    """Allow only native float32 ratio roundoff at existing efficacy bounds."""
    original=np.asarray(original,dtype=np.float32)
    current=np.asarray(current,dtype=np.float32)
    if (original.shape!=current.shape or not np.isfinite(original).all()
            or not np.isfinite(current).all() or np.any(original==0)
            or np.any(current==0) or np.any(np.signbit(original)!=np.signbit(current))):
        raise ValueError('Finite nonzero original-sign edge values required')
    raw=current.astype(float)/original.astype(float)
    eps=8*np.finfo(np.float32).eps
    if np.any(raw<.001*(1-eps)) or np.any(raw>10*(1+eps)):
        raise ValueError('Current original-edge efficacy outside native roundoff bounds')
    projected=np.clip(raw,.001,10.)
    reconstructed=np.asarray(original.astype(float)*projected,dtype=np.float32)
    ulp=np.abs(np.spacing(current))
    if np.any(abs(reconstructed.astype(float)-current.astype(float))>ulp):
        raise ValueError('Current float32 edge values fail one-ULP reconstruction')
    return raw,projected,float(np.max(abs(projected-raw)))


def observed_roster(cal, data_ids, meta_ids, meta_sources):
    """Use only the calibration's 49 sources, never extra diagnostic readouts."""
    sources=np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']])
    motor=cal['motor_ids']
    ids=np.asarray(data_ids)
    if (len(sources)!=49 or len(motor)!=391 or len(ids)!=446
            or len(np.unique(ids))!=len(ids)
            or not np.array_equal(ids,meta_ids)
            or len(meta_sources)!=55
            or not np.all(np.isin(sources,meta_sources))):
        raise ValueError('Exact E100 calibration49 sources and diagnostic55 roster required')
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
        raise TimeoutError('No remaining shared E101 wall allowance')
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
    e99=root/'e99-lf-tarsal-source-preparation'
    e101=root/'e101-individual-edge-native-recruitment'
    body_source=root/'e100-lf-tarsal-native-body-v2'
    e85=root/'e85-space-compatible-cadence-reference'
    spec={'experiment':'M3-E103-startup-preserving-recruitment',
        'hypothesis':'The E101 steady-fit candidate failed actual startup because its first.5s outputs were unconstrained; penalize departures from the healthy E100 initial command and six LTM tendons.',
        'intervention':'One E101 actualfloat32 efficacy1572-edge start, no inverse warm start. Same750 continuing Adamloglr.03 smooth-gradient updates kappa10/1/.1,751 exact-native full-objective observations and at most250 exact-native L-BFGS-B calls; best on full native objective only. Original signed1572 edges, all49sourceinputs/38tau/body/readout fixed.',
        'labels':'Actual simultaneous E100 source/current samples, own-leg quadratic E1 phase minus frozen E85 tripod offsets; all-six bracketed .5-to-last-peak interval. E85 late recorded commands offline only.',
        'objective':'Unchanged E101 normalized steady42joint+6x3foot+6LTM near.2+1e-5 efficacy prior plus original99sample startup command/.02 squared, tendon/.01 squared and q absolute>.05 hinge squared. Smooth forcing only for training gradients, native ReLU for best selection, final screens.',
        'limits':'Fit<=150wall, shared<=180wall, full neural/physics advance0, one CUDA owner, no body trial or F gate claim.',
        'prior_execution_wall_s':prior_wall_s,
        'e99_calibration_sha256':sha256(e99/'calibration.npz'),
        'e99_neutral_manifest_sha256':sha256(e99/'candidate-neutral/manifest.json'),
        'e100_signals_sha256':sha256(body_source/'recruitment-signals.npz'),
        'e100_metadata_sha256':sha256(body_source/'recruitment-metadata.npz'),
        'e100_trace_sha256':sha256(body_source/'trace.json'),
        'e100_parent_review_sha256':sha256(body_source/'parent-first-review.json'),
        'e101_calibration_sha256':sha256(e101/'calibration.npz'),
        'e101_coefficients_sha256':sha256(e101/'fit-coefficients.npz'),
        'e101_training_inputs_sha256':sha256(e101/'training-inputs.npz'),
        'e101_parent_review_sha256':sha256(e101/'parent-first-review.json'),
        'e102_startup_diagnostic_sha256':sha256(root/'e102-independent-edge-native-body-v2/parent-startup-diagnostic.json'),
        'e85_reference_sha256':sha256(e85/'commands.npz'),
        'causal_helper_sha256':sha256(Path(__file__).with_name('banc_m3_causal_recruitment.py')),
        'static_geometry_helper_sha256':sha256(Path(__file__).with_name('banc_m3_kinematic_fit.py')),
        'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'causal-helper.py').write_bytes(Path(__file__).with_name(
        'banc_m3_causal_recruitment.py').read_bytes())
    cal={key:value.copy() for key,value in np.load(e99/'calibration.npz').items()}
    old_cal={key:value.copy() for key,value in cal.items()}
    data=np.load(body_source/'recruitment-signals.npz')
    meta=np.load(body_source/'recruitment-metadata.npz')
    ref=np.load(e85/'commands.npz')
    graph=GraphStore.load(graph_path)
    original=StateStore.load(root/'e32-core-adaptation/candidate/start')
    prepared=StateStore.load(e99/'candidate-neutral')
    initial_activation=checked_initial_state(prepared)
    exact(original['body'],prepared['body'],'E101 original neutral body')
    for key in ('rate','adaptation','drive','output_mask','tick'):
        exact(original['network'][key],prepared['network'][key],
            'E101 original zero neural '+key)
    sources,motor,source_cols,motor_cols=observed_roster(cal,data['observed_ids'],
        meta['observed_ids'],meta['source_ids'])
    if len(cal['tau_ids'])!=38 or cal['theta'].shape!=(96,48):
        raise ValueError('E99 38tau and legacy48-source fit provenance required')
    times=np.arange(2001)*.005
    np.testing.assert_array_equal(data['time_s'],times[1:])
    raw_rate=data['rate'].astype(float)
    raw_current=(data['recurrent_current']+data['drive']-data['adaptation']).astype(float)
    measured_sources=np.vstack((np.zeros((1,49)),raw_rate[:,source_cols]))
    full_current=np.vstack((np.zeros((1,391)),raw_current[:,motor_cols]))
    if not np.isfinite(measured_sources).all() or not np.isfinite(full_current).all():
        raise ValueError('Finite E100 actual source and MN current required')
    if np.any(data['adaptation'][:,motor_cols]!=0) or np.any(meta['_cell_gain'][motor_cols]!=0):
        raise ValueError('Actual E100 MN adaptation must be zero')
    trace=json.loads((body_source/'trace.json').read_text())['samples']
    if len(trace)!=2001:
        raise ValueError('Complete original E100 2001-control trace required')
    actual_activation=np.vstack((initial_activation[None,:],
        data['muscle_activation'].astype(float)))
    decoded=cal['rest']+.02*actual_activation@cal['moment'].T
    recorded=np.asarray([row['joint_targets_rad'] for row in trace])
    np.testing.assert_allclose(decoded[1:],recorded[1:],rtol=0,atol=1e-6)
    startup_mask=(times>0)&(times<.5)
    startup_q=recorded[startup_mask].copy()
    actual_tendon=.2*np.tanh(actual_activation[:,cal['ltm']]/10)
    tendon_names=tuple(f'{leg}_tarsus' for leg in ('lf','lm','lh','rf','rm','rh'))
    recorded_tendon=np.asarray([[row['tendon_inputs'][name] for name in tendon_names]
        for row in trace[1:]])
    np.testing.assert_allclose(actual_tendon[1:],recorded_tendon,rtol=0,atol=1e-6)
    startup_tendon=actual_tendon[startup_mask].copy()
    startup_grid(times,startup_mask,startup_q,startup_tendon)
    np.savez_compressed(out/'startup-inputs.npz',time_s=times,
        mask=startup_mask,target_command=startup_q,target_tendon=startup_tendon,
        recorded_command=recorded,recorded_tendon=recorded_tendon,
        actual_E100_activation=actual_activation)
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
            raise ValueError('Complete actual E100 own-leg E1 peaks required')
        phases[:,leg]=phase_trace(times,peaks)
        last.append(peaks[-1])
    mask=(times>=.5)&(times<min(last))&np.isfinite(phases).all(axis=1)
    if int(mask.sum())!=1826:
        raise ValueError('Exact E100 all-six bracketed1826 training samples required')
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
    if np.count_nonzero(original_w)!=1572:
        raise ValueError('Exactly1572 original signed source-MN edges required')
    edge_rows,edge_cols,edge_ids=independent_edge_order(graph,motor,sources,original_w)
    edge_original=base.data[edge_ids].astype(np.float32)
    edge_current=current.data[edge_ids].astype(np.float32)
    np.testing.assert_array_equal(edge_original,original_w[edge_rows,edge_cols].astype(np.float32))
    np.testing.assert_array_equal(edge_current,current_w[edge_rows,edge_cols].astype(np.float32))
    start_cal=np.load(e101/'calibration.npz')
    start_coeff=np.load(e101/'fit-coefficients.npz')
    exact(edge_ids,start_cal['individual_edge_ids'],'E103 E101 independent edge order')
    exact(edge_ids,start_coeff['edge_ids'],'E103 E101 saved fit edge order')
    exact(sources,start_cal['individual_source_basis_ids'],'E103 E101 source basis')
    for key,value in cal.items():
        if key!='values':
            exact(value,start_cal[key],'E103 E101 unchanged base calibration '+key)
    start_weights=patch_weights(base,start_cal['edge_ids'],
        start_cal['source_values'],start_cal['values'])
    nonfit=np.ones(len(base.data),dtype=bool)
    nonfit[edge_ids]=False
    exact(current.data[nonfit],start_weights.data[nonfit],
        'E103 E101 only1572 motor edges differ from E100')
    starting_edge_weights=start_weights.data[edge_ids].astype(np.float32)
    raw_initial,initial_theta,initial_projection=current_edge_start(
        edge_original,starting_edge_weights)
    np.savez_compressed(out/'initial-E101-edge-efficacies.npz',edge_ids=edge_ids,
        motor_ids=motor[edge_rows],source_ids=sources[edge_cols],
        motor_rows=edge_rows,source_columns=edge_cols,
        original_signed_weights=edge_original,
        current_E100_signed_weights=edge_current,
        starting_E101_signed_weights=starting_edge_weights,
        raw_ratio=raw_initial,initial_efficacy=initial_theta,
        roundoff_projection_max_abs=initial_projection)
    background=full_current-measured_sources@current_w.T
    tau=meta['tau'][motor_cols].astype(float)
    a=meta['a'][motor_cols].astype(float)
    threshold=meta['threshold'][motor_cols].astype(float)
    cap=meta['cap'][motor_cols].astype(float)
    if (not np.allclose(a,1) or not np.allclose(threshold,7.5)
            or not np.allclose(cap,200)
            or not np.all(np.isin(np.round(tau,7),
                np.round([.02,.014046887330582595],7)))):
        raise ValueError('Actual E100 MN native physiology required')
    pool,moment,rest=cal['pool'],cal['moment'],cal['rest']
    membership,ltm=cal['membership'],cal['ltm']
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
        tau=tau,a=a,threshold=threshold,cap=cap,
        edge_ids=edge_ids,edge_rows=edge_rows,edge_cols=edge_cols,
        original_signed_edge_weights=edge_original,current_signed_edge_weights=edge_current,
        membership=membership,pool=pool,moment=moment,rest=rest,
        actual_activation=actual_activation,actual_command=decoded)
    frozen_inputs=np.load(e101/'training-inputs.npz')
    rebuilt_inputs=np.load(out/'training-inputs.npz')
    for key in frozen_inputs.files:
        exact(frozen_inputs[key],rebuilt_inputs[key],
            'E103 unchanged E101 steady training input '+key)
    background=frozen_inputs['background'].copy()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('E103 CUDA fit required')
    torch.set_num_threads(1)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device='cuda')
    source_t,background_t,original_t=map(tensor,(measured_sources,background,original_w))
    pool_t,moment_t,rest_t=map(tensor,(pool,moment,rest))
    tau_t,a_t,threshold_t,cap_t=map(tensor,(tau,a,threshold,cap))
    target_t,target_feet_t=map(tensor,(target,target_feet))
    joint_scale_t,foot_scale_t=map(tensor,(joint_scale,foot_scale))
    edge_rows_t=torch.as_tensor(edge_rows,device='cuda')
    edge_cols_t=torch.as_tensor(edge_cols,device='cuda')
    edge_original_t=tensor(edge_original.astype(float))
    mask_t=torch.as_tensor(mask,device='cuda')
    selected_t=torch.as_tensor(selected,device='cuda')
    ltm_t=torch.as_tensor(ltm,device='cuda')
    startup_mask_t=torch.as_tensor(startup_mask,device='cuda')
    startup_q_t=tensor(startup_q)
    startup_tendon_t=tensor(startup_tendon)

    def predict(theta=None,*,direct_weights=None,with_feet=True,kappa=None):
        if direct_weights is not None:
            weights=direct_weights
        else:
            weights=original_t.clone()
            weights[edge_rows_t,edge_cols_t]=edge_original_t*theta
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
    startup_fidelity_q=baseline_command[startup_mask]-startup_q
    startup_fidelity={'q_RMS_rad':float(np.sqrt(np.mean(startup_fidelity_q**2))),
        'q_max_abs_rad':float(np.max(abs(startup_fidelity_q)))}
    write_json(out/'baseline-startup-fidelity.json',startup_fidelity)
    if startup_fidelity['q_RMS_rad']>.002 or startup_fidelity['q_max_abs_rad']>.01:
        write_json(out/'result.json',{'validity':'startup_surrogate_method_failure',
            'baseline_startup_fidelity':startup_fidelity,'actual_neural_s':0,
            'actual_physical_s':0,'wall_s':time.perf_counter()-began})
        return
    transfer_rms=float(np.sqrt(np.mean((baseline_command[mask]-decoded[mask])**2)))
    reference_discrepancy=float(np.sqrt(np.mean((decoded[mask]-target)**2)))
    fidelity=transfer_rms/reference_discrepancy
    np.savez_compressed(out/'baseline-prediction.npz',command=baseline_command,
        tendon=bt.cpu().numpy(),motor_rate=br.cpu().numpy(),
        activation=ba.cpu().numpy(),actual_command=decoded,
        transfer_rms=transfer_rms,reference_rms=reference_discrepancy,ratio=fidelity)
    if not np.isfinite(fidelity) or fidelity>.1:
        write_json(out/'result.json',{'validity':'surrogate_method_failure',
            'baseline_transfer_ratio':fidelity,'actual_neural_s':0,
            'actual_physical_s':0,'wall_s':time.perf_counter()-began})
        return
    baseline_feet=geometry.evaluate(decoded[selected])

    def metrics(command,feet,tendon):
        joint=(command[mask]-target)/joint_scale
        foot=(feet-target_feet)/foot_scale
        tendon_error=(tendon[mask]-.2)/.2
        return {'combined':float(np.mean(joint**2)+np.mean(foot**2)+np.mean(tendon_error**2)),
            'q_rms':float(np.sqrt(np.mean((command[mask]-target)**2))),
            'foot_rms':float(np.sqrt(np.mean((feet-target_feet)**2))),
            'LF_row6_rms':float(np.sqrt(np.mean((command[mask,6]-target[:,6])**2))),
            'per_leg_q_rms':[float(np.sqrt(np.mean((command[mask,7*i:7*i+7]
                -target[:,7*i:7*i+7])**2))) for i in range(6)],
            'joint_normalized':float(np.mean(joint**2)),
            'foot_normalized':float(np.mean(foot**2)),
            'tendon_normalized':float(np.mean(tendon_error**2))}

    baseline_metrics=metrics(decoded,baseline_feet,actual_tendon)
    write_json(out/'baseline-output.json',baseline_metrics)
    if not np.isfinite(initial_theta).all() or np.any(initial_theta<.001) or np.any(initial_theta>10):
        raise ValueError('Exact current-W bounded independent edge start required')
    if not wall_budget_ok(time.perf_counter()-began,prior_wall_s,150):
        raise TimeoutError('E103 preparation left insufficient total wall reserve')
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
        prior=1e-5*(theta-1).square().mean()
        startup_q_error=command[startup_mask_t]-startup_q_t
        startup_ltm_error=tendon[startup_mask_t]-startup_tendon_t
        startup_joint=(startup_q_error/.02).square().mean()
        startup_tendon_loss=(startup_ltm_error/.01).square().mean()
        startup_hinge=(startup_q_error.abs()/.05-1).relu().square().mean()
        return (joint+foot+tendon_loss+prior+startup_joint+startup_tendon_loss
            +startup_hinge),(joint,foot,tendon_loss,prior,startup_joint,
                startup_tendon_loss,startup_hinge)

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
                edge_ids=edge_ids,motor_ids=motor[edge_rows],source_ids=sources[edge_cols])
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

    lower=np.full(1572,math.log(.001))
    upper=np.full(1572,math.log(10))
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
            parameter=torch.tensor(flat,dtype=torch.float64,
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
                    edge_ids=edge_ids,motor_ids=motor[edge_rows],source_ids=sources[edge_cols])
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
        edge_ids=edge_ids,motor_ids=motor[edge_rows],source_ids=sources[edge_cols])
    np.savez_compressed(out/'unprojected-prediction.npz',
        command=command.cpu().numpy(),tendon=tendon.cpu().numpy(),
        feet=feet.cpu().numpy(),motor_rate=rate.cpu().numpy(),
        activation=activation.cpu().numpy(),target=target,target_feet=target_feet,
        startup_mask=startup_mask,startup_target_q=startup_q,
        startup_target_tendon=startup_tendon)
    theta=project_efficacies(raw_theta,np.ones_like(raw_theta,dtype=bool))
    np.savez_compressed(out/'fit-coefficients.npz',theta=theta,
        raw_theta=raw_theta,edge_ids=edge_ids,motor_ids=motor[edge_rows],
        source_ids=sources[edge_cols],source_basis_ids=sources,
        roundoff_projection_max_abs=float(np.max(abs(theta-raw_theta))))
    values=cal['values'].copy()
    positions={int(edge):i for i,edge in enumerate(cal['edge_ids'])}
    fitted_edges=edge_ids
    for index,edge in enumerate(edge_ids):
        if int(edge) not in positions:
            raise ValueError('E99 original source-MN edge absent from existing patch')
        values[positions[int(edge)]]=np.float32(base.data[edge]*theta[index])
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
        actual_float32_source_weights=actual_w,target=target,target_feet=target_feet,
        startup_mask=startup_mask,startup_target_q=startup_q,
        startup_target_tendon=startup_tendon)
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
    raw_w=original_w.copy()
    raw_w[edge_rows,edge_cols]=edge_original.astype(float)*raw_theta
    initial_w=original_w.copy()
    initial_w[edge_rows,edge_cols]=edge_original.astype(float)*initial_theta
    with torch.no_grad():
        initial_command,initial_tendon,_,_,initial_activation=predict(
            tensor(initial_theta),with_feet=False)
    initial_startup=startup_screen(initial_command.cpu().numpy(),
        initial_tendon.cpu().numpy(),startup_q,startup_tendon,times,startup_mask)
    write_json(out/'LF-muscle13-native-diagnostic.json',{
        'initial_E101_actual_float32':lf_extensor_diagnostic(initial_w,initial_activation.cpu().numpy()),
        'best_unprojected':lf_extensor_diagnostic(raw_w,activation.cpu().numpy()),
        'final_actual_float32_edges':lf_extensor_diagnostic(actual_w,fa.cpu().numpy()),
        'interpretation':'Exact native ReLU readout only; softplus never enters runtime'})
    final_metrics=metrics(final_command,final_feet,final_tendon)
    screen=native_shape_screen(final_metrics,baseline_metrics)
    final_startup=startup_screen(final_command,final_tendon,
        startup_q,startup_tendon,times,startup_mask)
    local_supported=bool(screen['supported'] and final_startup['supported'])
    result={'experiment':spec['experiment'],
        'validity':'valid_conditional_fit' if wall_budget_ok(
            time.perf_counter()-began,prior_wall_s,180) else 'wall_limit_incomplete',
        'hypothesis_verdict':'conditional_startup_and_shape_supported' if local_supported else 'not_supported',
        'adoption':'not_applicable','body_validated':False,
        'baseline_transfer_ratio':fidelity,'reference_extraction_rms_rad':reference_rms,
        'baseline_output':baseline_metrics,'fitted_output':final_metrics,
        'shape_screen':screen,'initial_startup_screen':initial_startup,
        'startup_screen':final_startup,'local_supported':local_supported,
        'baseline_startup_fidelity':startup_fidelity,
        'adam_updates':adam_updates,'lbfgs_actual_calls':lbfgs_calls,
        'smooth_gradient_calls':smooth_gradient_calls,
        'native_observations_during_adam':native_observations,
        'final_float32_native_checks':1,
        'best_loss':best_loss,'stop':stop,
        'fit_parameters':len(edge_ids),'fitted_edges':len(fitted_edges),
        'fit_wall_s':fit_wall,'wall_s':time.perf_counter()-began,
        'smooth_started_gradient_calls':smooth_gradient_calls,
        'native_started_observations':native_observations,
        'lbfgsb_started_loss_gradient_calls':lbfgs_calls,
        'actual_neural_s':0,'actual_physical_s':0,
        'roundoff_projection_max_abs':float(np.max(abs(theta-raw_theta))),
        'nonfit_weight_exact':True,'CSR_topology_exact':True,
        'source_sign_exact':True,'all_38_tau_exact':True}
    write_json(out/'result.json',result)
    if not local_supported or result['validity']!='valid_conditional_fit':
        return
    cal['individual_edge_ids']=edge_ids.copy()
    cal['individual_edge_efficacies']=theta.copy()
    cal['individual_edge_motor_ids']=motor[edge_rows].copy()
    cal['individual_edge_source_ids']=sources[edge_cols].copy()
    cal['individual_source_basis_ids']=sources.copy()
    cal['values']=values
    for key,value in old_cal.items():
        if key!='values':
            exact(cal[key],value,'E103 other calibration '+key)
    np.savez_compressed(out/'calibration.npz',**cal)
    anatomy=json.loads((e99/'muscle-provenance.json').read_text())
    anatomy['E103_startup_preserving_recruitment']={'source_ids':sources.tolist(),
        'existing_signed_edges':len(fitted_edges),'parameters':len(edge_ids),
        'legacy_theta_shape':list(old_cal['theta'].shape),
        'conditional_only':True,'runtime_reference_phase':False}
    write_json(out/'muscle-provenance.json',anatomy)
    base_trial=EfficacyTrial(graph,original,cal['edge_ids'],cal['source_values'],
        cal['values'],zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial,cal,
        anatomy['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        neutral=session.snapshot()
        exact(original['body'],neutral['body'],'Original E103 neutral body')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(original['network'][key],neutral['network'][key],
                'Original E103 neutral neural '+key)
        exact(body_before,body.snapshot(),'Shadow static geometry leaves body unchanged')
        StateStore.save(out/'candidate-neutral',neutral)
        clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out/'candidate-neutral'))
        try:
            exact(neutral,clone.snapshot(),'E103 whole neutral disk exact')
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
