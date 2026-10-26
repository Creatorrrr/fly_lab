"""Finish E105 diagnostics from the preserved complete trace without model stepping."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.verify_c_banc_m3_intrinsic_body import e105_startup_diagnostic


def e105_resource_ledger(e104_wall,preparation_wall,preflight_wall,body_wall,
        recovery_wall,model_s):
    """Account for data-only recovery within body and shared execution limits."""
    body_total=body_wall+recovery_wall
    shared=e104_wall+preparation_wall+preflight_wall+body_total
    return {'body_including_recovery_wall_s':body_total,
        'shared_including_recovery_wall_s':shared,
        'resource_limit_met':bool(preparation_wall<=60 and body_total<=285
            and shared<=375 and model_s<=10.1000001)}


def run(root,source,body):
    if (body/'recovered-result.json').exists():
        raise FileExistsError('Preserve existing E105 recovery')
    started=time.perf_counter()
    original={name:sha256(body/name) for name in ('spec.json','runner.py',
        'execution-error.json','physical-result.json','quality-summary.json',
        'source-phase-summary.json','actual-LF-source-summary.json',
        'trace.json','native-contact.npz','actual-startup-vs-E104-raw.npz')}
    rows=json.loads((body/'trace.json').read_text())['samples']
    contact=np.load(body/'native-contact.npz')
    physical=json.loads((body/'physical-result.json').read_text())
    quality=json.loads((body/'quality-summary.json').read_text())
    source_phase=json.loads((body/'source-phase-summary.json').read_text())
    lf_source=json.loads((body/'actual-LF-source-summary.json').read_text())
    prediction=np.load(root/'e104-constrained-startup-recruitment/prediction.npz')
    original_startup=np.load(body/'actual-startup-vs-E104-raw.npz')
    if (len(rows)!=2001 or physical['controls']!=2000
            or physical['native_samples']!=100000
            or len(contact['time_s'])!=100000
            or physical['moving_disk_future_exact'] is not True
            or quality['complete_10s_native_health'] is not True):
        raise ValueError('Preserved complete healthy E105 raw body trace required')
    native=[(None,None,None,bool(a),float(b)) for a,b in zip(
        contact['nonfoot_contact'][:5000],contact['upright'][:5000])]
    transfer,times,q,ltm,dq,dltm=e105_startup_diagnostic(rows,native,prediction)
    for key,value in {'time_s':times,'actual_command':q,'actual_LTM':ltm,
            'command_difference':dq,'LTM_difference':dltm}.items():
        np.testing.assert_array_equal(original_startup[key],value)
    first100=bool(len(rows)>=101 and len(native)==5000
        and np.array_equal(np.asarray([row['time_s'] for row in rows[1:101]]),
            np.arange(1,101)*.005))
    health=bool(first100 and not contact['nonfoot_contact'][:5000].any()
        and np.min(contact['upright'][:5000])>=.9
        and all(not row['fault'] for row in rows[1:101]))
    diagnostic={'complete_first_half_second_native_health':health,
        'complete_first_100_controls':first100,
        'first_half_second_native_samples':5000,
        'first_half_second_nonfoot_samples':int(contact['nonfoot_contact'][:5000].sum()),
        'first_half_second_minimum_upright':float(np.min(contact['upright'][:5000])),
        'actual_99_command_LTM_transfer':transfer,
        'E104_startup_supported':True,'E104_steady_supported':False,
        'E104_local_supported':False,'full_10s_native_health_separate':True,
        'original_physical_F3_F5_separate':quality['development_screens']}
    write_json(body/'recovered-startup-diagnostic.json',diagnostic)
    prep=json.loads((source/'result.json').read_text())
    e104=json.loads((root/'e104-constrained-startup-recruitment/result.json').read_text())
    first_error=json.loads((root/'e105-startup-native-body-diagnostic/execution-error.json').read_text())
    body_error=json.loads((body/'execution-error.json').read_text())
    recovery_wall=time.perf_counter()-started
    ledger=e105_resource_ledger(e104['wall_s'],prep['wall_s'],
        first_error['execution_wall_s'],body_error['execution_wall_s'],
        recovery_wall,physical['neural_physical_model_s'])
    recovered={'experiment':'M3-E105-startup-native-body-diagnostic',
        'validity':'complete_raw_body_recovered_postprocessing',
        'original_execution_error_retained':True,
        'postprocessing_model_advance_s':0.,
        'physical':physical,'quality_summary':quality,
        'source_phase_summary':source_phase,'actual_LF_source_summary':lf_source,
        'startup_diagnostic':diagnostic,
        'E104_shape_supported':False,'E104_local_supported':False,
        'actual_neural_s':physical['neural_physical_model_s'],
        'actual_physical_s':physical['neural_physical_model_s'],
        'E104_wall_s':e104['wall_s'],'E105_preparation_wall_s':prep['wall_s'],
        'E105_failed_preflight_wall_s':first_error['execution_wall_s'],
        'E105_body_including_failed_postprocess_wall_s':body_error['execution_wall_s'],
        'E105_postprocessing_recovery_wall_s':recovery_wall,
        'shared_execution_wall_s':ledger['shared_including_recovery_wall_s'],
        **ledger,
        'F6_unused_poses_tested':False,'final_adoption':False}
    write_json(body/'recovered-result.json',recovered)
    write_json(body/'postprocessing-recovery.json',{
        'cause':'LF native_physiology diagnostic dict shadowed 100000 physical native tuple list',
        'original_artifact_sha256':original,
        'recovery_tool_sha256':sha256(__file__),
        'preserved_startup_raw_exact':True,'model_reexecution':False,
        'postprocessing_wall_s':recovery_wall})
    print(json.dumps({'startup':diagnostic,'shared_wall_s':ledger['shared_including_recovery_wall_s'],
        'resource_limit_met':recovered['resource_limit_met']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','body'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.source,args.body)
