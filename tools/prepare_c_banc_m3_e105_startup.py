"""Prepare the E104 failed-shape readout for one direct startup body diagnosis."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact


def accepted_e104_failed_fit_review(review):
    """Only the explicit, defect-free E104 second review authorizes E105."""
    return (isinstance(review,dict)
        and review.get('accepted_as_valid_failed_conditional_fit') is True
        and review.get('blocking_defects')==[])


def checked_e104_status(result,selection,review):
    if (result.get('validity')!='valid_conditional_fit'
            or result.get('resource_limit_met') is not True
            or result.get('startup_screen',{}).get('supported') is not True
            or result.get('shape_screen',{}).get('supported') is not False
            or result.get('local_supported') is not False
            or result.get('fit_parameters')!=1572 or result.get('fitted_edges')!=1572
            or result.get('selected_record')!='lowest_S_startup_feasible'
            or selection.get('selected_record')!='lowest_S_startup_feasible'
            or not accepted_e104_failed_fit_review(review)):
        raise ValueError('Reviewed E104 startup-pass/steady-fail result required')


def equal_integer_ids(left,right,label):
    left=np.asarray(left)
    right=np.asarray(right)
    if (left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu'
            or left.shape!=right.shape or not np.array_equal(left,right)):
        raise ValueError(label)


def run(root,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    e99=root/'e99-lf-tarsal-source-preparation'
    e104=root/'e104-constrained-startup-recruitment'
    previous=json.loads((e104/'result.json').read_text())
    selection=json.loads((e104/'native-selection.json').read_text())
    review=json.loads((e104/'parent-second-review.json').read_text())
    checked_e104_status(previous,selection,review)
    spec={'experiment':'M3-E105-startup-native-body-diagnostic-preparation',
        'status':'diagnostic_from_failed_conditional_shape',
        'intervention':'E104 selected actualfloat32 efficacy on1572 existing signed E99 source-to-MN edges; E104 steady/local failure remains. Original E32 zero-state, no runtime smoothing, phase, teacher or prewarm.',
        'limits':'Preparation<=60wall, no neural/physical advance; separate one actual body trial.',
        'e99_calibration_sha256':sha256(e99/'calibration.npz'),
        'e99_neutral_manifest_sha256':sha256(e99/'candidate-neutral/manifest.json'),
        'e104_result_sha256':sha256(e104/'result.json'),
        'e104_coefficients_sha256':sha256(e104/'fit-coefficients.npz'),
        'e104_prediction_sha256':sha256(e104/'prediction.npz'),
        'e104_selection_sha256':sha256(e104/'native-selection.json'),
        'e104_parent_second_review_sha256':sha256(e104/'parent-second-review.json'),
        'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    graph=GraphStore.load(graph_path)
    old={key:value.copy() for key,value in np.load(e99/'calibration.npz').items()}
    cal={key:value.copy() for key,value in old.items()}
    coeff=np.load(e104/'fit-coefficients.npz')
    predicted=np.load(e104/'prediction.npz')
    source_ids=np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']])
    edges=coeff['edge_ids']
    theta=coeff['theta']
    if (len(source_ids)!=49 or len(cal['tau_ids'])!=38 or cal['theta'].shape!=(96,48)
            or len(cal['theta_source_ids'])!=48 or len(edges)!=1572
            or len(np.unique(edges))!=1572 or theta.shape!=(1572,)
            or not np.isfinite(theta).all() or np.any(theta<.001)
            or np.any(theta>10)):
        raise ValueError('E104 exact49/38/1572 bounded provenance required')
    exact(source_ids,coeff['source_basis_ids'],'E104 source basis order')
    base,_=rate_weights(graph)
    base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,cal['motor_ids'])
    before=patch_weights(base,cal['edge_ids'],cal['source_values'],cal['values'])
    positions={int(edge):index for index,edge in enumerate(cal['edge_ids'])}
    motor=np.searchsorted(graph.indptr,edges,side='right')-1
    equal_integer_ids(motor,coeff['motor_ids'],'Original CSR MN receiver identities')
    equal_integer_ids(graph.indices[edges],coeff['source_ids'],
        'Original CSR source identities')
    if (not np.all(np.isin(motor,cal['motor_ids']))
            or not np.all(np.isin(graph.indices[edges],source_ids))
            or not all(int(edge) in positions for edge in edges)):
        raise ValueError('Only existing E99 selected source-to-MN edges allowed')
    for index,edge in enumerate(edges):
        cal['values'][positions[int(edge)]]=np.float32(base.data[edge]*theta[index])
    after=patch_weights(base,cal['edge_ids'],cal['source_values'],cal['values'])
    exact(before.indptr,after.indptr,'Original CSR row topology')
    exact(before.indices,after.indices,'Original CSR column topology')
    other=np.ones(len(before.data),dtype=bool)
    other[edges]=False
    exact(before.data[other],after.data[other],'All nonselected and source receiving weights')
    exact(np.sign(after.data[edges]),np.sign(base.data[edges]),'Original signed MN edges')
    exact(after.data[edges],np.asarray(base.data[edges].astype(float)*theta,dtype=np.float32),
        'Actual E104 original-signed float32 edge values')
    exact(after[cal['motor_ids']][:,source_ids].toarray().astype(float),
        predicted['actual_float32_source_weights'],'E104 stored actualfloat32 dense prediction weights')
    for key,value in old.items():
        if key!='values':
            exact(cal[key],value,'E99 unchanged calibration '+key)
    cal['individual_edge_ids']=edges.copy()
    cal['individual_edge_efficacies']=theta.copy()
    cal['individual_edge_motor_ids']=coeff['motor_ids'].copy()
    cal['individual_edge_source_ids']=coeff['source_ids'].copy()
    cal['individual_source_basis_ids']=source_ids.copy()
    anatomy=json.loads((e99/'muscle-provenance.json').read_text())
    anatomy['E105_E104_startup_diagnostic']={'E104_local_supported':False,
        'E104_steady_supported':False,'E104_startup_supported':True,
        'original_signed_MN_edges':1572,'selected_sources':49,
        'legacy_theta_shape':list(cal['theta'].shape),'runtime_smoothing':False}
    np.savez_compressed(out/'calibration.npz',**cal)
    write_json(out/'muscle-provenance.json',anatomy)
    initial=StateStore.load(root/'e32-core-adaptation/candidate/start')
    base_trial=EfficacyTrial(graph,initial,cal['edge_ids'],cal['source_values'],
        cal['values'],zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial,cal,
        anatomy['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        if session.network.device!='cuda':
            raise ValueError('E105 actual CUDA constructor required')
        actual=source_matrix(session.network)
        exact(actual.indptr,after.indptr,'Runtime CSR row topology')
        exact(actual.indices,after.indices,'Runtime CSR column topology')
        exact(actual.data,after.data,'Runtime exact selected and nonselected weights')
        neutral=session.snapshot()
        exact(initial['body'],neutral['body'],'Original neutral body/world')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(initial['network'][key],neutral['network'][key],'Original neutral neural '+key)
        StateStore.save(out/'candidate-neutral',neutral)
    finally:
        session.close()
    clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,StateStore.load(out/'candidate-neutral'))
    try:
        exact(neutral,clone.snapshot(),'Whole neutral disk exact')
    finally:
        clone.close()
    wall=time.perf_counter()-began
    result={'experiment':spec['experiment'],
        'validity':'valid_prepared_startup_diagnostic' if wall<=60 else 'wall_limit_incomplete',
        'status':'diagnostic_from_failed_conditional_shape',
        'E104_startup_supported':True,'E104_steady_supported':False,
        'E104_local_supported':False,'fitted_edges':1572,'fit_parameters':1572,
        'selected_sources':49,'tau_values_exact':True,'nonfit_weight_exact':True,
        'neutral_disk_exact':True,'actual_neural_s':0.,'actual_physical_s':0.,
        'fit_calls':0,'wall_s':wall,'resource_limit_met':wall<=60}
    write_json(out/'result.json',result)
    if wall>60:
        raise TimeoutError('E105 preparation exceeded60 wall seconds')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: '+str(args.out))
    started=time.perf_counter()
    try:
        run(args.root,args.graph,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete',
                'error':type(error).__name__,'message':str(error),
                'execution_wall_s':time.perf_counter()-started,
                'actual_neural_s':0.,'actual_physical_s':0.,'fit_calls':0})
        raise
