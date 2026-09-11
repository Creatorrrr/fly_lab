#!/usr/bin/env python3
"""Fixed-posture neural recruitment probes; no physical trajectory is executed."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.integrity import finite, write_json
from flylab.c.model_config import model_ticks
from flylab.c.muscles import MuscleRig
from flylab.c.neural import LIFParameters, create_backend
from flylab.c.receptors import ReceptorParameters, JointReceptors
from flylab.c.single_joint import JointParameters, build_profile, encode_feedback


def run(graph_path,out,seconds=.3,backend='auto',integration='exact-exponential-held-drive-v1',angles=(1.2,1.862,2.2),gains=(18.,36.,72.)):
    seconds=finite(seconds,'probe seconds',.05,2.)
    count=model_ticks(seconds,.001,maximum=2000)
    parameters=LIFParameters(integration=integration)
    if not 1<=len(angles)<=3 or not 1<=len(gains)<=3:raise ValueError('Use 1..3 fixed angles/gains')
    angles=[finite(q,'probe angle',.01,3.) for q in angles]
    gains=[finite(g,'probe gain',0.,100.) for g in gains]
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=dict(schema='flylab.joint-feedback-probe.v1',seconds=seconds,backend=backend,
        held_angles_rad=angles,sensory_gains_mV=gains,mute_outgoing=[False,True],integration=integration,
        held_velocity_rad_s=0.,direct_motor_stimulation=False,physicalExecuted=False,
        question='Are fast motor neurons recruited by these fixed FeCO hypotheses? Does recruitment disappear when sensory outgoing signals are muted?',
        adoption_policy='No default parameter changes; no force or behavior validation')
    write_json(out/'spec.json',spec)
    report=dict(status='RUNNING',physicalExecuted=False,biological_validation=False,cases=[])
    write_json(out/'report.json',report);start=time.perf_counter()
    try:
        graph=GraphStore.load(graph_path);body=MuscleRig()
        write_json(out/'graph_manifest.json',graph.manifest)
        for q in spec['held_angles_rad']:
            for gain in spec['sensory_gains_mV']:
                for muted in spec['mute_outgoing']:
                    name=f'q_{q:g}_gain_{gain:g}'+('_muted' if muted else '')
                    directory=out/name;directory.mkdir()
                    profile=build_profile(graph,body,JointParameters(initial_q_rad=q,sensory_gain_mV=gain),parameters)
                    write_json(directory/'profile.json',profile)
                    neural=create_backend(graph,LIFParameters(**profile['neural_parameters']),backend)
                    receptor=JointReceptors(ReceptorParameters(**profile['receptor_parameters']))
                    ports=[(r['feature'],graph.resolve(r['ids'],maximum=10000)) for r in profile['bindings']['sensory']]
                    sensory=np.unique(np.concatenate([ids for _,ids in ports]))
                    motor=graph.resolve([r['ids'][0] for r in profile['bindings']['motor']])
                    if muted: neural.set_interventions(mute=sensory.tolist())
                    rows=[]
                    for _ in range(count):
                        sensed=receptor.step(q,0.)
                        drive=encode_feedback(graph.n,ports,sensed,gain)
                        neural.advance(drive,model_ticks(.001,neural.p.dt),capture=motor)
                        read=neural.readout(motor)
                        rows.append(dict(neural_tick=neural.tick,seconds=neural.tick*neural.p.dt,
                            max_feedback_drive_mV=float(drive.max()),spike_events=neural.last_events,
                            **{k:v.tolist() for k,v in read.items()}))
                    state=neural.snapshot()
                    np.savez_compressed(directory/'final_neural.npz',**{key:state[key] for key in ('v','h','rate','spike_count')})
                    write_json(directory/'trace.json',rows)
                    incoming=[]
                    for index in motor:
                        a,b=graph.indptr[index:index+2];pre=graph.indices[a:b];weights=graph.weights[a:b]
                        active=state['spike_count'][pre]>0
                        emitted=weights*state['spike_count'][pre]
                        if muted: emitted[np.isin(pre,sensory)]=0.
                        incoming.append(dict(motor_id=graph.nodes[index]['id'],incoming_anatomical_pairs=int(b-a),
                            active_presynaptic_cells=int(active.sum()),
                            positive_weighted_emitted_mV=float(emitted[emitted>0].sum()),
                            negative_weighted_emitted_mV=float(emitted[emitted<0].sum()),
                            interpretation='Cumulative weighted emissions on incoming edges, not instantaneous current; late events may still be in transit'))
                    result=dict(name=name,q_rad=q,gain_mV=gain,muted=muted,simulated_neurons=graph.n,
                        neural_ticks=neural.tick,motor_spike_counts=state['spike_count'][motor].tolist(),
                        sensory_spike_count=int(state['spike_count'][sensory].sum()),
                        global_spike_count=int(state['spike_count'].sum()),
                        motor_sampled_peak_voltage_mV=np.max([r['voltage_mV'] for r in rows],axis=0).tolist(),
                        voltage_observation_interval_s=.001,spike_time_resolution_s=neural.p.dt,
                        motor_final_current_mV=state['h'][motor].tolist(),incoming=incoming)
                    write_json(directory/'result.json',result);report['cases'].append(result)
                    write_json(out/'report.json',report)
                    print(name,result['motor_spike_counts'],result['motor_sampled_peak_voltage_mV'],flush=True)
        report.update(status='COMPLETE',neural_backend=backend,neural_runtime=state.get('backend_runtime'),
            simulated_neurons=graph.n,simulated_seconds=seconds*len(report['cases']),
            interpretation='Fixed stimuli only. Subthreshold responses and absent motor recruitment remain failures of the proposed feedback mechanism.',
            motor_recruitment_detected=any(any(r['motor_spike_counts']) for r in report['cases']))
    except Exception as exc:
        report.update(status='FAIL' if report['cases'] else 'BLOCKED',error=repr(exc))
        if 'directory' in locals() and 'rows' in locals():
            write_json(directory/'partial_trace.json',rows)
            if 'neural' in locals():
                try:
                    failed=neural.snapshot()
                    np.savez_compressed(directory/'failed_neural.npz',**{k:failed[k] for k in ('v','h','rate','spike_count')})
                except Exception as save_error: report['failure_state_error']=repr(save_error)
    report['wall_seconds']=time.perf_counter()-start;write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/acquisitions/banc888-v2-20260909/bundle')
    p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=.3)
    p.add_argument('--backend',choices=('auto','exp_lif_mps','exp_lif_cpu_reference','exp_lif_cuda'),default='auto')
    p.add_argument('--integration',default='exact-exponential-held-drive-v1',choices=('exact-exponential-held-drive-v1','exact-exponential-reset-current-v1','exact-exponential-voltage-events-v1'))
    p.add_argument('--angles',type=float,nargs='+',default=[1.2,1.862,2.2]);p.add_argument('--gains',type=float,nargs='+',default=[18.,36.,72.])
    a=p.parse_args();report=run(a.graph,a.out,a.seconds,a.backend,a.integration,a.angles,a.gains)
    print(report['status']);raise SystemExit(0 if report['status']=='COMPLETE' else 2 if report['status']=='BLOCKED' else 1)
