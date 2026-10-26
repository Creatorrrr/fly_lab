"""Measure learned full-BANC state-field shift under identical body teaching.

No motor neural states are forced. Body teaching remains an explicit diagnostic
intervention; the decoded output is recorded, never used to move the body.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.target_sense import TargetObservation
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.train_c_banc_m3_recurrent import command_metrics
from tools.verify_c_banc_m3_full_trial import CalibratedTrial

JOINTS=np.array([[0,2,1,4,3,5,6],[7,9,8,11,10,12,13],[14,16,15,18,17,19,20],
                [21,23,22,25,24,26,27],[28,30,29,32,31,33,34],[35,37,36,39,38,40,41]])


def motor_replay(boundary,initial_boundary,internal,pool,moment,rest,initial_state):
    """Independent float64 native RK4, linearly interpolated 2.5ms boundary."""
    r,z,a=[np.asarray(v,dtype=float).copy() for v in initial_state]
    previous=np.asarray(initial_boundary,float);commands=[];rates=[]
    def rhs(rr,zz,b):
        u=np.maximum(200*np.tanh((b+internal@rr-7.5-zz)/200),0)
        return (u-rr)/.02,(8*rr-zz)/.3
    for half,end in enumerate(boundary):
        for j in range(10):
            b0=previous+(end-previous)*(j/10)
            bm=previous+(end-previous)*((j+.5)/10)
            b1=previous+(end-previous)*((j+1)/10)
            k1,a1=rhs(r,z,b0)
            k2,a2=rhs(r+.000125*k1,z+.000125*a1,bm)
            k3,a3=rhs(r+.000125*k2,z+.000125*a2,bm)
            k4,a4=rhs(r+.00025*k3,z+.00025*a3,b1)
            r+=.00025/6*(k1+2*k2+2*k3+k4)
            z+=.00025/6*(a1+2*a2+2*a3+a4)
        previous=end
        if half%2==1:
            a+=(1-math.exp(-.5))*(pool@r-a)
            commands.append(rest+.02*moment@a);rates.append(r.copy())
    return np.asarray(commands),np.asarray(rates)


def run(source,candidate,teacher,labels,graph_path,out,reference_source=None):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E19-closed-neural-body-teacher',
        'hypothesis':'Quantify intrinsic motor reconstruction error and additional nonmotor-feedback field shift when teacher motor forcing is removed',
        'intervention':'Same frozen E17b positive motor incoming efficacy, all158706 neurons/11584852source pairs/native.25ms CUDA. Replay E11b physical commands only; no motor r/z forcing. Record decoder output without applying it. Same E05 afferent18/fixed22tonic/no runtime CPG/root edits.',
        'arms':{'neutral':{'start_s':0,'duration_s':2.2},'release':{'start_s':1,'duration_s':.7}},
        'initialization':'Neutral exactly source initial body/neural r,z. Release ONCE restores E12 t1s full neural/body/receptor/muscle state. No subsequent state replacement.',
        'comparison':'Use SAME learned W_MO on E12 vs E19 nonmotor states. Independent native motor-only replay starts at same arm initial r/z/A and initial boundary. E12 source_current is not used because it combines original weights and motor inputs.',
        'fidelity_gate':'E19 boundary replay versus actual full decoder output RMS<=.02rad, finite; otherwise no causal field attribution. 2.5ms boundary interpolation is diagnostic approximation.',
        'interpretation':'Base motor mismatch and outside feedback amplification can coexist. These arms still use a body teacher; no autonomous/F1-F8 pass or adoption.',
        'limits':'One GPU job, total2.9modelsec/300wallsec, no fitting or altered parameters',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'teacher_sha256':sha256(teacher/'signals.npz'),
        'labels_sha256':sha256(labels/'trace.json'),'tool_sha256':sha256(__file__)}
    if reference_source:
        spec.update(experiment='M3-E21-self-consistency-check',
            hypothesis='The frozen one-refit E20 candidate reduces full-network decoded-command error on the same body input and initial state, without motor forcing',
            intervention='Same E19 body/afferent/tonic/native full neural dynamics; only frozen E20 absolute efficacies replace E17b. No compound scaling, motor clamp or runtime controller.',
            arms={'neutral':{'start_s':0,'duration_s':2.2}},
            comparison='Use SAME newly learned W_MO on E19 recorded nonmotor states and this new full neural trajectory. Compare new full command error and boundary-replay defect with E19. All data are exposed development, not final gait.',
            limits='One GPU job,2.2modelsec/300wallsec; no additional fit or candidate selection',
            reference_sha256=sha256(reference_source/'signals.npz'))
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    origin=StateStore.load(source/'start');taught=StateStore.load(teacher/'teacher-state-200')
    g=GraphStore.load(graph_path);cal=dict(np.load(candidate/'candidate.npz'));td=dict(np.load(teacher/'signals.npz'))
    reference=dict(np.load(reference_source/'signals.npz')) if reference_source else td
    if not np.array_equal(reference['observed_ids'],td['observed_ids']) or not np.array_equal(reference['time_s'],td['time_s']):raise ValueError('Reference observation identities or clocks changed')
    rows=json.loads((labels/'trace.json').read_text())['samples'][1:441]
    weights=csr_matrix((cal['incoming_data'],cal['incoming_indices'],cal['incoming_indptr']),shape=(391,g.n))
    outside=weights.copy();outside.data[np.isin(outside.indices,cal['motor_ids'])]=0;outside.eliminate_zeros()
    internal=weights[:,cal['motor_ids']].astype(float)
    observed=td['observed_ids'];boundary_matrix=outside[:,observed].astype(float)
    if boundary_matrix.nnz!=outside.nnz:raise ValueError('Missing outside presynaptic source')
    teacher_boundary=np.asarray(boundary_matrix@reference['presynaptic'].T).T
    began=time.perf_counter();summary={}
    arms=(('neutral',0,440),) if reference_source else (('neutral',0,440),('release',200,340))
    for name,lo,hi in arms:
        arm=out/name;arm.mkdir();s=CalibratedTrial(g,origin,candidate/'candidate.npz',preserve_preload_on_cut=True)
        record={k:[] for k in ('time_s','presynaptic','rate','adaptation','external_drive','source_current','decoded_command')}
        arm_start=time.perf_counter()
        try:
            if lo:s.release_teacher(taught)
            n,body,adapter=s.network,s.body,s.adapter
            initial=n.snapshot();StateStore.save(arm/'start',s.snapshot())
            initial_state=(initial['rate'][cal['motor_ids']],initial['adaptation'][cal['motor_ids']],adapter.activation.copy())
            initial_boundary=outside.astype(float)@initial['rate']
            pool=adapter.pool.copy()
            # Only recapture observation interval; warmup state is restored.
            n.interval_steps=10;n.graph=None
            with n.torch.inference_mode():
                n._interval();n._interval();n.torch.cuda.synchronize()
                n.graph=n.torch.cuda.CUDAGraph()
                with n.torch.cuda.graph(n.graph):n._interval()
            n.restore(initial)
            tonic=s.navigation.encoder.drive(TargetObservation(0,20,True))
            if np.count_nonzero(tonic)!=22 or not np.all(tonic[s.descending]==5000) or not np.all(tonic[s.navigation.encoder.ports]==-5000):
                raise ValueError('Unexpected forward input')
            for tick in range(lo,hi):
                if time.perf_counter()-began>280:raise TimeoutError('E19 collection wall limit')
                drive=adapter.encode()+tonic
                for half in (1,2):
                    n.advance(drive,10);full=n.readout()
                    record['time_s'].append(tick*.005+half*.0025)
                    record['presynaptic'].append(full[observed].copy())
                    record['rate'].append(full[cal['motor_ids']].copy())
                    record['adaptation'].append(n.adaptation[adapter.motor_ids].cpu().numpy().copy())
                    record['external_drive'].append(drive[cal['motor_ids']].copy())
                    record['source_current'].append(weights@full)
                command,_=adapter.decode(full[cal['motor_ids']]);record['decoded_command'].append(command.copy())
                row=rows[tick]
                body.step_joint_targets(np.asarray(row['joint_targets_rad']),np.asarray(row['adhesion'],bool),tendon_inputs=row['tendon_inputs'])
                for key in ('qpos','qvel','ctrl'):
                    if getattr(body.d,key).tobytes()!=np.asarray(row[key]).tobytes():raise AssertionError(f'Physical teaching mismatch {tick+1}/{key}')
                s.control_tick=tick+1
                if (tick+1)%100==0:print(json.dumps({'arm':name,'tick':tick+1,'wall_s':time.perf_counter()-began}),flush=True)
            s._clocks()
            record={k:np.asarray(v) for k,v in record.items()}
            if any(not np.isfinite(v).all() for v in record.values()) or np.any(record['external_drive']):raise ValueError('Nonfinite record or new direct motor drive')
            # Source topology/weights remain the learning reference; collection
            # model uses the separately identified learned candidate.
            np.savez_compressed(arm/'signals.npz',**record,motor_ids=cal['motor_ids'],observed_ids=observed,
                incoming_indptr=cal['incoming_indptr'],incoming_indices=cal['incoming_indices'],incoming_data=cal['incoming_source_data'],
                motor_group=cal['motor_group'],moment=cal['moment'],rest=cal['rest'],command=td['command'][lo:hi],
                initial_rate=initial_state[0],initial_adaptation=initial_state[1],initial_activation=initial_state[2])
            StateStore.save(arm/'final-diagnostic',{'body':body.snapshot(),'network':n.snapshot(),'adapter':adapter.snapshot(),
                'schema':'flylab.m3-body-teaching-diagnostic.v1','control_tick':s.control_tick})
            actual_boundary=np.asarray(boundary_matrix@record['presynaptic'].T).T
            q19,r19=motor_replay(actual_boundary,initial_boundary,internal,pool,cal['moment'],cal['rest'],initial_state)
            q12,r12=motor_replay(teacher_boundary[2*lo:2*hi],initial_boundary,internal,pool,cal['moment'],cal['rest'],initial_state)
            np.savez_compressed(arm/'comparison.npz',actual_boundary=actual_boundary,teacher_boundary=teacher_boundary[2*lo:2*hi],
                command_actual_boundary=q19,command_teacher_boundary=q12,rate_actual_boundary=r19,rate_teacher_boundary=r12,
                initial_boundary=initial_boundary)
            times=record['time_s'][1::2];mask=times>=max(.5,lo*.005)
            fidelity=float(np.sqrt(np.mean((q19-record['decoded_command'])**2)))
            initial_teacher=outside.astype(float)@(origin if not lo else taught)['network']['rate']
            boundary_delta=actual_boundary-teacher_boundary[2*lo:2*hi]
            result={'run_status':'completed','validity':'valid_body_teaching_collection','adoption':'not_applicable','milestone_outcome':'pending',
                'model_s':(hi-lo)*.005,'wall_s':time.perf_counter()-arm_start,'device':n.device,'cuda_implementation':n.cuda_implementation,
                'all_physical_steps_byte_exact':hi-lo,'model_hash':n.identity,'initial_boundary_difference':float(np.max(np.abs(initial_boundary-initial_teacher))),
                'boundary_replay_fidelity_rms_rad':fidelity,'field_attribution_gate_passed':fidelity<=.02,
                'actual_decoder':command_metrics(record['decoded_command'],td['command'][lo:hi],mask,JOINTS),
                ('matched_reference_boundary' if reference_source else 'matched_teacher_boundary'):command_metrics(q12,td['command'][lo:hi],mask,JOINTS),
                'matched_actual_boundary':command_metrics(q19,td['command'][lo:hi],mask,JOINTS),
                'outside_field_rms':float(np.sqrt(np.mean(boundary_delta**2))),
                'first_20ms_outside_field_rms':float(np.sqrt(np.mean(boundary_delta[:8]**2))),
                'candidate_sha256':spec['candidate_sha256'],'signals_sha256':sha256(arm/'signals.npz')}
            write_json(arm/'result.json',result);summary[name]=result;print(json.dumps({'arm':name,**result}),flush=True)
        except Exception as exc:
            np.savez_compressed(arm/'partial-signals.npz',**{k:np.asarray(v) for k,v in record.items()})
            write_json(arm/'execution-error.json',{'type':type(exc).__name__,'message':str(exc),'adoption':'not_applicable'})
            raise
        finally:s.close()
    write_json(out/'result.json',{'experiment':spec['experiment'],'run_status':'completed','adoption':'not_applicable',
        'milestone_outcome':'pending','wall_s':time.perf_counter()-began,'arms':summary})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','teacher','labels','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-source',type=Path)
    a=p.parse_args();run(a.source,a.candidate,a.teacher,a.labels,a.graph,a.out,a.reference_source)
