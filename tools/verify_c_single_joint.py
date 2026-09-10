#!/usr/bin/env python3
"""Matched whole-BANC/MPS + native Hill-muscle experiments on one joint.

Technical execution, causal effects and physiological validation are reported
separately. Failed and partial runs are retained; existing outputs are refused.
"""
import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.body_identity import source_identity
from flylab.c.graph import GraphStore
from flylab.c.integrity import digest, finite, write_json
from flylab.c.muscles import MuscleRig
from flylab.c.single_joint import JointParameters, SingleJointLoop, build_profile
from flylab.c.storage import StateStore


def same(a,b):
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def exit_code(report, require_feedback=False):
    if report['status']=='BLOCKED': return 2
    if report.get('technical_status')!='PASS': return 1
    if require_feedback and not report.get('sensorimotor',{}).get('feedback_effect_detected'): return 1
    return 0


def cases():
    rows = [dict(name='feedback'),dict(name='open_loop',feedback=False),
            dict(name='polarity_swapped',parameters=dict(polarity_swapped=True)),
            dict(name='delay_5ms',parameters=dict(receptor_delay_steps=5)),
            dict(name='sensory_outgoing_muted',mute_sensory=True),
            dict(name='flexor',feedback=False,stimulate='flexor'),
            dict(name='extensor',feedback=False,stimulate='extensor'),
            dict(name='coactivation',feedback=False,stimulate='both'),
            dict(name='motor_suppressed',feedback=False,stimulate='both',suppress_motor=True),
            dict(name='motor_disconnected',feedback=False,stimulate='both',motor_connected=False)]
    for sign, torque in (('positive',.01),('negative',-.01)):
        rows += [dict(name='external_'+sign,torque=torque),
                 dict(name='external_'+sign+'_open',torque=torque,feedback=False)]
    for label, q in (('low',1.2),('high',2.2)):
        rows += [dict(name='heldout_'+label,parameters=dict(initial_q_rad=q)),
                 dict(name='heldout_'+label+'_open',parameters=dict(initial_q_rad=q),feedback=False)]
    for half in (50.,200.):
        for open_loop in (False,True):
            rows.append(dict(name=f'half_{int(half)}'+('_open' if open_loop else ''),
                parameters=dict(rate_half_Hz=half),feedback=not open_loop))
    return rows


def command(case, step, loop):
    t = step*loop.p.control_dt
    motors = [m['ids'][0] for m in loop.profile['bindings']['motor']]
    selected = {'flexor':motors[:1],'extensor':motors[1:],'both':motors}.get(case.get('stimulate'),[])
    return dict(stimulation={i:20. for i in selected} if .01 <= t < .05 else {},
        suppress_ids=motors if case.get('suppress_motor') else [],
        mute_ids=[i for p in loop.profile['bindings']['sensory'] for i in p['ids']] if case.get('mute_sensory') else [],
        feedback=case.get('feedback',True),motor_connected=case.get('motor_connected',True),
        torque=case.get('torque',0.) if .05 <= t < .08 else 0.)


def evaluate(traces, results, spec):
    completed = all(r['status']=='COMPLETE' for r in results)
    all_rows = [row for rows in traces.values() for row in rows]
    checks = dict(all_cases_complete=completed,
        whole_graph_clock=all(r['clock_agreement'] for r in results),
        activation_bounded=all(0 <= a <= 1 for row in all_rows for a in row['physics']['activation']),
        force_projection=all(row['physics']['torque_projection_error'] < 1e-9 for row in all_rows),
        dynamics_balance=all(row['physics']['dynamics_balance_error'] < 1e-8 for row in all_rows),
        correct_direction=all(row['direction']['status']=='CONSISTENT' for row in all_rows),
        exact_continuation=all(r['exact_continuation'] for r in results),
        feedback_off_zero=all(row['feedback_drive_max_mV']==0. for row in traces['open_loop']),
        disconnected_floor_only=all(row['physics']['applied_excitation']==row['physics']['minimum_excitation'] for row in traces['motor_disconnected']),
        suppression_zero_spikes=all(r['motor_spike_counts']==[0,0] for r in results if r['name']=='motor_suppressed'),
        no_contact=all(row['physics']['contacts']==0 for row in all_rows))
    def delta(a,b,key):
        if len(traces[a]) != len(traces[b]): return None
        return float(np.max(np.abs(np.array([r['physics'][key] for r in traces[a]])-
                                   np.array([r['physics'][key] for r in traces[b]]))))
    checks['disconnected_matches_no_drive'] = delta('motor_disconnected','open_loop','q_rad') == 0.
    checks['suppressed_matches_no_drive'] = delta('motor_suppressed','open_loop','q_rad') == 0.
    by_name={r['name']:r for r in results}
    for name,label in (('motor_disconnected','disconnected'),('motor_suppressed','suppressed')):
        checks[label+'_physical_state_exact'] = (by_name[name].get('physical_state_hash') is not None and
            by_name[name].get('physical_state_hash')==by_name['open_loop'].get('physical_state_hash'))
    physical = {name:delta(name,'open_loop','q_rad') for name in
                ('feedback','sensory_outgoing_muted','polarity_swapped','delay_5ms','flexor','extensor')}
    final = {name:rows[-1]['physics']['q_rad'] for name,rows in traces.items()}
    direction = final['flexor'] > final['open_loop'] > final['extensor'] if completed else None
    recovery = {}
    for name,base in (('external_positive','feedback'),('external_negative','feedback'),
                      ('external_positive_open','open_loop'),('external_negative_open','open_loop')):
        rows,control = traces[name],traces[base]
        if len(rows)!=len(control): recovery[name]=dict(status='INCOMPLETE');continue
        error = np.array([r['physics']['q_rad']-c['physics']['q_rad'] for r,c in zip(rows,control)])
        times = np.array([r['seconds'] for r in rows]); challenged = error[(times>=.05)&(times<=.08+1e-10)]
        peak = float(np.max(np.abs(challenged))); residual = abs(float(error[-1]))
        ratio = residual/peak if peak >= .001 else None
        recovery[name]=dict(peak_perturbation_rad=peak,final_residual_rad=residual,residual_ratio=ratio,
            status='INCOMPLETE' if ratio is None else 'PASS' if ratio <= .5 else 'FAIL')
    effect=physical['feedback'] is not None and physical['feedback'] > spec['gates']['causal_angle_difference_rad']
    return dict(technical_checks=checks, technical_status='PASS' if all(checks.values()) else 'FAIL',
        sensorimotor_status='EFFECT_DETECTED_NOT_CALIBRATED' if effect else 'FAIL_NO_FEEDBACK_MOTOR_EFFECT',
        sensorimotor=dict(max_angle_difference_vs_open_rad=physical,opponent_stimulation_direction=direction,
            feedback_effect_detected=effect,
            outgoing_path_effect_rad=delta('feedback','sensory_outgoing_muted','q_rad')),
        recovery=recovery,
        recovery_attribution={sign:dict(closed_vs_open_angle_difference_rad=delta('external_'+sign,'external_'+sign+'_open','q_rad'),
            attribution='BASELINE_MECHANICS_ONLY' if delta('external_'+sign,'external_'+sign+'_open','q_rad')==0. else 'REQUIRES_CAUSAL_ASSESSMENT',
            baseline='Passive mechanics plus upstream minimum excitation')
            for sign in ('positive','negative')},
        physiological_status='CALIBRATION_REQUIRED',
        interpretation='A causal effect is not proof of stabilizing feedback or calibrated biology')


def run(graph_path,out,seconds=.15,backend='exp_lif_mps',sensory_gain=18.):
    seconds=finite(seconds,'assay seconds',.1,2.)
    if abs(seconds/.001-round(seconds/.001))>1e-8: raise ValueError('Integral 1 ms duration required')
    defaults=asdict(JointParameters(sensory_gain_mV=sensory_gain))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=dict(schema='flylab.single-joint-assay.v1',seconds=seconds,control_dt=.001,backend=backend,cases=cases(),
        stimulation=dict(amplitude_mV=20.,start_s=.01,end_s=.05),external_torque=dict(start_s=.05,end_s=.08),
        gates=dict(projection=1e-9,dynamics=1e-8,causal_angle_difference_rad=1e-6,
                   recovery_min_challenge_rad=.001,recovery_max_residual_ratio=.5),
        partial_is_fail=True,parameters_selected_before_execution=True,default_parameters=defaults)
    write_json(out/'spec.json',spec)
    report=dict(schema='flylab.single-joint-results.v1',status='RUNNING',spec_hash=digest(spec),
                cases=[],physicalExecuted=False,biological_validation=False)
    write_json(out/'report.json',report); started=time.perf_counter(); traces={}
    try:
        graph=GraphStore.load(graph_path)
        write_json(out/'graph_manifest.json',graph.manifest);write_json(out/'source_identity.json',source_identity())
        shutil.copytree(Path(__file__).resolve().parents[1]/'flylab',out/'source/flylab',ignore=shutil.ignore_patterns('__pycache__'))
        for case in spec['cases']:
            directory=out/case['name'];directory.mkdir();rows=[]
            body=MuscleRig(); profile=build_profile(graph,body,JointParameters(**{**defaults,**case.get('parameters',{})}))
            write_json(directory/'profile.json',profile)
            loop=SingleJointLoop(graph,profile,backend,body=body)
            rows=[loop.frame()]
            try:
                for step in range(round(seconds/.001)):
                    rows.append(loop.step(**command(case,step,loop)))
                    report['physicalExecuted'] |= loop.body.tick>0
                    if loop.fault: break
                final=loop.snapshot();StateStore.save(directory/'final',final)
                exact=False
                if not loop.fault:
                    kw=command(case,loop.control_tick,loop)
                    loop.step(**kw);expected=loop.snapshot()
                    loop.restore(StateStore.load(directory/'final'));loop.step(**kw)
                    exact=same(expected,loop.snapshot())
                motor_counts=final['neural']['spike_count'][loop.motor_indices].tolist()
                result=dict(name=case['name'],status='FAILED_GUARD' if final['fault'] else 'COMPLETE',fault=final['fault'],
                    elapsed_s=rows[-1]['seconds'],simulated_nodes=graph.n,neural_ticks=final['neural']['tick'],
                    physics_ticks=final['body']['tick'],selected_neurons=len(loop.observed),
                    clock_agreement=(final['neural']['tick']==round(seconds/loop.neural.p.dt)
                        and final['body']['tick']==round(seconds/body.model.opt.timestep)
                        and final['receptors']['tick']==round(seconds/.001)),
                    exact_continuation=exact,motor_spike_counts=motor_counts,
                    final_angle_rad=rows[-1]['physics']['q_rad'],
                    max_motor_rates_Hz=np.max([r['motor_rates_Hz'] for r in rows],axis=0).tolist(),
                    final_all_neuron_spike_count=int(final['neural']['spike_count'].sum()),profile_hash=digest(profile))
                result['physical_state_hash']=hashlib.sha256(final['body']['state'].tobytes()).hexdigest()
            except Exception as exc:
                write_json(directory/'failure.json',dict(error=repr(exc),control_tick=loop.control_tick))
                StateStore.save(directory/'failed_state',loop.snapshot())
                raise
            finally: write_json(directory/'trace.json',rows)
            write_json(directory/'result.json',result);traces[case['name']]=rows;report['cases'].append(result)
            write_json(out/'report.json',report)
            print(case['name'],result['status'],'q',round(result['final_angle_rad'],6),'spikes',motor_counts,'replay',exact,flush=True)
        report.update(evaluate(traces,report['cases'],spec))
        report.update(status='COMPLETE_WITH_LIMITATIONS' if report['technical_status']=='PASS' else 'FAIL',
            simulated_seconds=sum(r['elapsed_s'] for r in report['cases']),
            continuation_simulated_seconds=2*.001*sum(r['fault'] is None for r in report['cases']),
            completed_cases=sum(r['status']=='COMPLETE' for r in report['cases']),
            graph_hash=graph.hash,neural_backend=backend,body_model_hash=body.identity,
            neural_runtime=loop.neural.snapshot().get('backend_runtime'),
            further_validation=['Measured motor recruitment/FeCO tuning/SI units','Other muscles and contacts','Gait and natural behavior'])
    except Exception as exc:
        report.update(status='FAIL' if report['physicalExecuted'] else 'BLOCKED',error=repr(exc))
    report['wall_seconds']=time.perf_counter()-started
    write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/acquisitions/banc888-v2-20260909/bundle')
    p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=.15)
    p.add_argument('--sensory-gain',type=float,default=18.,help='Explicit uncalibrated FeCO input gain in mV; kept in the profile identity')
    p.add_argument('--require-feedback-effect',action='store_true',help='Exit with failure when technical checks pass but no causal motor effect of feedback is detected')
    p.add_argument('--backend',choices=('exp_lif_mps','exp_lif_cpu_reference'),default='exp_lif_mps')
    a=p.parse_args();report=run(a.graph,a.out,a.seconds,a.backend,a.sensory_gain)
    print(report['status'],report.get('sensorimotor_status','NOT_EVALUATED'),report.get('error',''),flush=True)
    raise SystemExit(exit_code(report,a.require_feedback_effect))
