"""One slower measured-step body reference within the unchanged physical world."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control, contact_diagnostics
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    reference=root/'e42-source-aligned-reference'
    old=json.loads((reference/'trace.json').read_text())['samples']
    t=np.array([row['time_s'] for row in old])
    pos=np.array([row['body']['position'] for row in old])
    selected=(t>=.5)&(t<2.2)
    original_speed=float((pos[selected][-1,0]-pos[selected][0,0])/(t[selected][-1]-t[selected][0]))
    target_speed=1.5
    period=.15*original_speed/target_speed
    if not .3<period<.8:
        raise ValueError('One observed-data derived cadence expected in[.3,.8]s')
    spec={'experiment':'M3-E85-space-compatible-cadence-reference',
        'question':'Can the same measured-step body waveform sustain healthy10s forward walking at a slower physical cadence in the original world?',
        'basis':'Verified original walls have inner faces x+-24/y+-18mm, startthoraxx.555mm. E42already observed[.5,2.2) net+Xspeed4.1968mm/s; if maintained it would traverse about42mm in10s, beyond forward wall clearance. Thus a correct fast straight waveform alone cannot complete this original world/duration. Curved E73 is not a success. This is a speed/space consistency diagnosis, not a world or gate change.',
        'intervention':'Keep original M2neutralbody/world/servo/contact, same FlyBody measured-step amplitude.7/startupenvelope50ms, originalE37leg offsets, noadhesion/all8tendons.2. Change only reference period once from.15 to .15*(observednetXspeed/1.5). 1.5mm/s is a prospective engineering setpoint with margin above10mm/9.5s and below original wall clearance; cadence-speed proportionality is a hypothesis to test, not assumed performance.',
        'scope':'Externaltemplate/body-only developmentreference. Not a BANC controller, biologicalspeedvalidation, F6 or finaladoption. If useful, a subsequent constant kinetic scaling of existing neural source dynamics and native all-jointfit must produce motion without this runtimeclock.',
        'prediction':'At original10s/[.5,10) fixedheading netforward>=10mm, net/path>=.5, nativeunsupported0/upright>=.9; at least10completecyclesperleg, LRmeans within.15of.5/R>=.5, fouroriginalknee widths/CV gates. Original F1-F8 remain required from actual wholeBANC; reference failure prompts a changed method, no cadence grid or wall move.',
        'limits':'OneCPUphysicalbody <=10modelseconds/100wallseconds, native100ushealthguard. Noactualneuraladvance/M2UI/F6changes.',
        'original_speed_mm_s':original_speed,'target_speed_mm_s':target_speed,'derived_period_s':period,
        'reference_sha256':sha256(reference/'trace.json'),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    saved=StateStore.load(root/'e32-core-adaptation/candidate/start')
    body=new_body(saved)
    wave=json.loads((root/'e37-core-teacher/teaching-waveform.json').read_text())
    phase_by_leg=dict(zip(wave['group_leg_by_direct_motor_projection'],wave['group_training_phase_cycles']))
    offsets=np.array([phase_by_leg[leg] for leg in body.steps.legs])
    tendons={name:.2 for name in body.tendon_control.names}
    observer=SimpleNamespace(body=body,control_tick=0,rate=np.zeros(391),tendon_inputs=tendons,fault=None)
    native_step=body._step_physics
    shadow=copy.copy(body);shadow.d=body.mj.MjData(body.m)
    integration_type=body.mj.mjtState.mjSTATE_INTEGRATION
    integration_state=np.empty(body.mj.mj_stateSize(body.m,integration_type))
    native=[];rows=[];commands=[];phases=[];reason='duration'
    def observe_step():
        native_step()
        body.mj.mj_getState(body.m,body.d,integration_state,integration_type)
        body.mj.mj_setState(body.m,shadow.d,integration_state,integration_type)
        body.mj.mj_forward(body.m,shadow.d)
        feet=contact_diagnostics(shadow)
        _,rotation,_=shadow.pose()
        native.append((float(body.physics_time()),feet['normal_load_bw'],
            [np.nan if v is None else v for v in feet['slip_mm_s']],
            bool(shadow.nonfoot_contact()),float(rotation[2,2])))
    body._step_physics=observe_step
    try:
        exact(saved['body'],body.snapshot(),'Original cadence reference neutralbody')
        rows.append(capture_control(observer))
        with (out/'trace.jsonl').open('w',encoding='utf8') as stream:
            stream.write(json.dumps(rows[0])+'\n')
            for tick in range(1,2001):
                if time.perf_counter()-began>95:
                    reason='wall_limit';break
                phase=2*np.pi*((tick*.005/period+offsets)%1)
                amplitude=np.full(6,.7*(1-np.exp(-tick*.005/.05)))
                q=body.steps.get_joint_angles_by_dof_order(phase,amplitude,body.order)
                before=len(native)
                body.step_joint_targets(q,np.zeros(6,dtype=bool),tendon_inputs=tendons)
                observer.control_tick=tick
                rows.append(capture_control(observer));commands.append(q.copy());phases.append(phase.copy())
                stream.write(json.dumps(rows[-1])+'\n')
                if len(native)-before!=50:
                    raise ValueError('Exactly50nativephysicalsteps/control required')
                if body.fault or any(row[3] or row[4]<.9 for row in native[before:]):
                    reason='health_guard';break
                if tick%400==0:
                    print(json.dumps({'time_s':tick*.005,'position':rows[-1]['body']['position'],
                        'wall_s':time.perf_counter()-began}),flush=True)
        if not body.fault:
            StateStore.save(out/'final-body',body.snapshot())
    finally:
        write_json(out/'trace.json',{'schema':'flylab.m3-cadence-reference.v1',
            'controller':'External measured-step slowercadence reference; no BANC execution',
            'graph_hash':saved['graph_hash'],'model_hash':'no-neural-model','body_hash':body.model_hash,
            'adapter_hash':sha256(out/'spec.json'),'parameters':spec,'samples':rows})
        if commands:
            np.savez_compressed(out/'commands.npz',time_s=np.arange(1,len(commands)+1)*.005,
                command=commands,phase=phases,period_s=period,neutral=body.neutral)
        if native:
            np.savez_compressed(out/'native-contact.npz',time_s=[r[0] for r in native],
                normal_load_bw=[r[1] for r in native],slip_mm_s=[r[2] for r in native],
                nonfoot_contact=[r[3] for r in native],upright=[r[4] for r in native])
        body.close()
    quality,_,_=analyze(load_model(out/'trace.json'))
    times=np.array([row['time_s'] for row in rows]);position=np.array([row['body']['position'] for row in rows])
    selected=(times>=.5)&(times<10)
    delta=position[selected][-1,[0,2]]-position[selected][0,[0,2]]
    forward=np.array(rows[0]['body']['basis'])[[0,2],0];forward/=np.linalg.norm(forward)
    distance=float(np.linalg.norm(delta));path=float(np.linalg.norm(np.diff(position[selected][:,[0,2]],axis=0),axis=1).sum())
    physical={'physical_s':(len(rows)-1)*.005,'stop_reason':reason,'fault':body.fault,
        'native_samples':len(native),'native_unsupported_samples':sum(r[3] for r in native),
        'minimum_upright':min(r[4] for r in native),'fixed_heading_net_forward_mm':float(delta@forward),
        'net_displacement_mm':distance,'planar_path_mm':path,'net_over_path':distance/path if path>0 else 0}
    f3=(reason=='duration' and len(rows)==2001 and physical['native_unsupported_samples']==0
        and physical['minimum_upright']>=.9 and physical['fixed_heading_net_forward_mm']>=10 and physical['net_over_path']>=.5)
    f4=(all(row['complete_cycles']>=10 for row in quality['events'].values())
        and all(abs((quality['pair_phase'][key]['mean_cycles']-.5+.5)%1-.5)<=.15
            and quality['pair_phase'][key]['resultant_length']>=.5 for key in ('lf-rf','lm-rm','lh-rh')))
    widths={leg:v['p95']-v['p05'] for leg,v in quality['knee_interior_deg'].items()}
    f5=(all(widths[leg]>=value for leg,value in {'lf':25,'rf':30,'lh':40,'rh':35}.items())
        and all(v['period_cv'] is not None and v['period_cv']<=.3 for v in quality['events'].values()))
    result={'experiment':spec['experiment'],'validity':'valid_external_reference_body_trial',
        'hypothesis_verdict':'supported_for_reference_only' if f3 and f4 and f5 else 'not_supported',
        'adoption':'not_applicable','milestone_outcome':'pending','physical':physical,
        'reference_screens':{'F3_numeric':f3,'F4_numeric':f4,'F5_numeric':f5},
        'knee_width_deg':widths,'quality':quality,'actual_neural_s':0,
        'actual_physical_s':physical['physical_s'],'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='quality'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.root,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete_reference_trial',
                'hypothesis_verdict':'not_tested','error':type(error).__name__,'message':str(error)})
        raise
