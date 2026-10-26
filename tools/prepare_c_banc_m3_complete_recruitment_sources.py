"""Prepare existing premotor source inputs for a native all-joint recruitment study."""

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
from tools.verify_c_banc_observation import exact


def run(root,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'e72-front-knee-delivery'
    rh=root/'e80-RH-positive-source-recruitment'
    spec={'experiment':'M3-E83-existing-premotor-source-basis',
        'question':'Can existing positive RHpremotor and LFlevator-source paths provide native waveforms for a joint all-leg calibration while preserving the E73motor interface?',
        'basis':'E82 measured-current transfer error is1.699% of reference command discrepancy; jointtracking errors are small. E79/E81twoRHcoxa fits reduced localerror but failed direction. E80selected46sources cover94/96muscles positively; LFlevator has none, is actuallysilent with J-.118..+.175<threshold7.5, and LFtarget exceeds fixedrest byup to.182764rad. A source basis needs this missing positive direction before joint refitting.',
        'LF_path':['720575941571519483','720575941433046798','720575941611361811'],
        'LF_secondary_tonic_path':['720575941500851362','720575941433046798','720575941611361811'],
        'intervention':'Keep E72all391motor incoming weights, moment/rest/pool/tendons/body/world/taus/input/initialstate EXACT. Reuse only E80twoRHrelay incoming patches and nonadaptation, not E80motor weights or momentgains. Add original LF IN03A013: retain existingLFtonic(+.69) andleftDNg100(+.18) incoming edges; other nonzero inputsx.001, adaptation8to0, tau20msunchanged. Its existing output toLFlevator stays attenuated+.00036 until later fitting.',
        'expression':'LFpositive reference target after MN/muscle lag anticorrelates with E1-following positive source (-.969 with oneextra20msrelay). Therefore use steadypositive LFlevator recruitment as a potential counterpart to varying existingdepressor, not E1-followingexcitation. No LFmotor current is forced and no restoffset, waveform/phase/root correction is introduced.',
        'validity':'Originalnodes/positiveedgecontacts/provenance andno nav overlap; allmotor rows exactlyE72; onlythree declared premotor incoming rows and adaptation gains differ; whole original neutralinitial arrays exact and disk restoration exact.',
        'next_test':'One actual fullBANC/CUDA/body5s+.1clone under originalworld/healthguards. Record all selected47source plus391MN endpointcurrents/rates/drive/adaptation and96muscleactivation. Inspect relay stability/transmission and retained source rhythms before deriving a joint fit. This preparatory5s test cannotpass original10sF3 or be F6.',
        'limits':'OneCUDA preparation<=90wallseconds,0experimentalneural/physicaladvance. Next actualbody<=5.1model/180wallseconds; M2/UI/F6/F1-F8unchanged.',
        'source_sha256':sha256(source/'calibration.npz'),'RH_sha256':sha256(rh/'calibration.npz'),
        'audit_sha256':sha256(root/'e82-native-all-leg-waveforms/result.json'),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    graph=GraphStore.load(graph_path)
    cal=dict(np.load(source/'calibration.npz'))
    original={k:v.copy() for k,v in cal.items()}
    rhcal=np.load(rh/'calibration.npz')
    encoder=BancTargetEncoder(graph,forward_drive=5000)
    matrix,_=rate_weights(graph)
    matrix=encoder.calibrate_weights(matrix,cal['motor_ids'])
    old_matrix=patch_weights(matrix,cal['edge_ids'],cal['source_values'],cal['values'])
    tonic,relay,mn=map(int,graph.resolve(['flywire:banc:888:'+rid for rid in spec['LF_path']]))
    dn=int(graph.resolve(['flywire:banc:888:'+spec['LF_secondary_tonic_path'][0]])[0])
    rh_ids=np.array(graph.resolve(['flywire:banc:888:720575941526736740',
        'flywire:banc:888:720575941390358788']))
    ids=np.r_[rh_ids,relay]
    np.testing.assert_allclose([matrix[relay,tonic],matrix[relay,dn],matrix[mn,relay]],[.69,.18,.36],rtol=1e-6,atol=0)
    if np.intersect1d(ids,np.r_[cal['input_ids'],encoder.ports,cal['tau_ids']]).size:
        raise ValueError('Three new original nonnavigation unmodifiedtau source cells required')
    k=len(cal['edge_ids'])
    np.testing.assert_array_equal(rhcal['edge_ids'][:k],cal['edge_ids'])
    rh_edges=rhcal['edge_ids'][k:]
    np.testing.assert_array_equal(np.unique(np.searchsorted(matrix.indptr,rh_edges,side='right')-1),np.sort(rh_ids))
    lf_edges=np.arange(matrix.indptr[relay],matrix.indptr[relay+1])
    lf_edges=lf_edges[matrix.data[lf_edges]!=0]
    if np.intersect1d(lf_edges,np.r_[cal['edge_ids'],rh_edges]).size:
        raise ValueError('LFpremotor input patch must have distinct ownership')
    lf_values=matrix.data[lf_edges]*np.float32(.001)
    retain=np.isin(matrix.indices[lf_edges],[tonic,dn])
    lf_values[retain]=matrix.data[lf_edges[retain]]
    cal.update(edge_ids=np.r_[cal['edge_ids'],rh_edges,lf_edges],
        source_values=np.r_[cal['source_values'],matrix.data[rh_edges],matrix.data[lf_edges]],
        values=np.r_[cal['values'],rhcal['values'][k:],lf_values],
        zero_adaptation_ids=np.union1d(cal['zero_adaptation_ids'],ids),
        input_ids=np.r_[cal['input_ids'],ids])
    changed_matrix=patch_weights(matrix,cal['edge_ids'],cal['source_values'],cal['values'])
    old_motor,new_motor=old_matrix[cal['motor_ids']],changed_matrix[cal['motor_ids']]
    for field in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(old_motor,field),getattr(new_motor,field))
    for key,value in original.items():
        if key not in ('edge_ids','source_values','values','zero_adaptation_ids','input_ids'):
            np.testing.assert_array_equal(value,cal[key])
    anatomy=json.loads((source/'muscle-provenance.json').read_text())
    anatomy['input_nodes'] += [graph.nodes[int(i)] for i in ids]
    hypothesis={'source_ids':ids.tolist(),'source_nodes':[graph.nodes[int(i)] for i in ids],
        'LFpaths':[spec['LF_path'],spec['LF_secondary_tonic_path']],
        'LFsource_weights':[float(matrix[relay,tonic]),float(matrix[relay,dn]),float(matrix[mn,relay])],
        'unchanged_MN_rows':391,'unchanged_motor_edge_data_exact':True,
        'unchanged_anatomical_moment_and_rest':True,'actual_neural_s':0,'actual_physical_s':0,
        'physiological_validation':'Engineered steady/rhythmic source hypothesis; not measured biology'}
    anatomy['premotor_basis_hypothesis']=hypothesis
    np.savez_compressed(out/'calibration.npz',**cal)
    write_json(out/'muscle-provenance.json',anatomy)
    write_json(out/'source-basis-hypothesis.json',hypothesis)
    initial=StateStore.load(root/'e32-core-adaptation/candidate/start')
    base=EfficacyTrial(graph,initial,cal['edge_ids'],cal['source_values'],cal['values'],
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    session=IntrinsicRecruitmentTrial(RecruitmentTrial(base,cal,anatomy['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        neutral=session.snapshot()
        exact(initial['body'],neutral['body'],'Original sourcebasis neutralbody')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(initial['network'][key],neutral['network'][key],'Original neural '+key)
        StateStore.save(out/'candidate-neutral',neutral)
    finally:
        session.close()
    clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,StateStore.load(out/'candidate-neutral'))
    try:
        exact(neutral,clone.snapshot(),'Entire sourcebasis diskstate')
    finally:
        clone.close()
    result={'experiment':spec['experiment'],'validity':'valid_prepared_existing_source_basis',
        'hypothesis_verdict':'not_tested','adoption':'not_applicable','milestone_outcome':'pending',
        'selected_source_count':len(np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']])),
        'new_source_ids':ids.tolist(),'unchanged391motor_rows_exact':True,
        'neutral_disk_exact':True,'actual_neural_s':0,'actual_physical_s':0,'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.root,args.graph,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete_preparation',
                'hypothesis_verdict':'not_tested','error':type(error).__name__,'message':str(error)})
        raise
