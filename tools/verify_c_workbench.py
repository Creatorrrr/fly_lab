#!/usr/bin/env python3
"""Native GPU/physics workbench evidence; retain every diagnostic outcome."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.engine import CEngine
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings, SensoryEncoder
from flylab.c.storage import StateStore
from flylab.c.integrity import read_json, write_json
from flylab.c.experiments import replay_recording, paired
from flylab.c.behavior import CRITERIA, trace_sample, behavior_metrics
from flylab.sensors import default_world, SensorAdapter
from tools.benchmark_c_runtime import exact_differences
from flylab.c.neural import BACKEND_CHOICES
from flylab.c.backend_selection import resolve_backend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=.2);p.add_argument('--matrix',action='store_true')
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto')
    p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--bindings',default='data/fafb783/bindings.json')
    p.add_argument('--bilateral-bindings',default='data/fafb783/bindings-bilateral-geosmin-v1.json')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    a.backend=resolve_backend(a.backend)
    if not .1<=a.seconds<=3 or abs(a.seconds/.005-round(a.seconds/.005))>1e-8:raise ValueError('Use 5ms periods between .1 and 3s')
    g=GraphStore.load(a.graph)
    original=PortBindings(g,read_json(a.bindings))
    b=PortBindings(g,read_json(a.bilateral_bindings))
    report=dict(schema='flylab.workbench_validation.v1',status='RUNNING',gates={},cases=[],criteria=CRITERIA,
                biologicalValidation=False,physicalExecuted=False,binding=b.summary())
    def persist():write_json(a.out/'report.json',report)
    persist()
    e=CEngine(g,original,mode='C_STRICT',backend=a.backend)
    try:
        e.step(2);before=e.checkpoint();body=e.body
        e.command('place',dict(kind='food',position=[2,.7,-1],strength=2.))
        odor=SensorAdapter(42).observe(e.body,e.world,.005,e.config)['odor']
        report['gates']['source_edit']=dict(status='PASS' if body is e.body and odor!=before['last_sensors']['odor'] and
            not exact_differences(before['body'],e.body.snapshot()) and not exact_differences(before['neural'],e.neural.snapshot()) else 'FAIL',odor=odor)
        sensor_before=SensorAdapter(42).observe(e.body,e.world,.005,e.config)
        pos=e.body.frame()[0]['position'];basis=e.body.frame()[0]['basis'];axis=np.array([row[0] for row in basis]);axis[1]=0;axis/=np.linalg.norm(axis)
        obstacle=(np.array(pos)+axis*6).tolist();obstacle[1]=1.5
        e.command('place',dict(kind='obstacle',position=obstacle,radius=1.5))
        sensor_after=SensorAdapter(42).observe(e.body,e.world,.005,e.config)
        report['gates']['physical_geometry_edit']=dict(status='PASS' if sensor_after['nearRanges'][4]<sensor_before['nearRanges'][4] and
            not exact_differences(before['body'],e.body.snapshot()) else 'FAIL',before_mm=sensor_before['nearRanges'][4],after_mm=sensor_after['nearRanges'][4])
        e.command('update_object',dict(id='user-2',position=[obstacle[0]+2,1.2,obstacle[2]+4],radius=1.2))
        e.command('delete_object',dict(id='user-2'));e.command('undo_environment',{})
        state=e.checkpoint()
        try:e.command('place',dict(kind='obstacle',position=e.body.frame()[0]['position']))
        except ValueError:pass
        else:raise AssertionError('Invalid physical placement accepted')
        report['gates']['invalid_edit_preserves_state']=dict(status='PASS' if not exact_differences(state,e.checkpoint()) else 'FAIL')
        cohort=[n['id'] for n in g.nodes[:512]];e.subscribe(cohort)
        e.start_recording(a.out/'selected512',cohort)
        item=e.schedule(dict(kind='stimulate',ids=original.spec['motor']['forward']['ids'],amplitude_mV=20.,duration_controls=20))
        start=time.perf_counter();e.step(10);elapsed=time.perf_counter()-start
        e.subscribe(original.spec['motor']['forward']['ids']);e.command('cancel_intervention',{'serial':item['serial']})
        e.command('place',dict(kind='hazard',position=[-3,.7,-3],strength=2.))
        e.command('undo_environment',{});e.step(10);e.stop_recording()
        final=e.checkpoint();StateStore.save(a.out/'workbench_final',final)
        replay=replay_recording(g,original,a.out/'selected512',e.body_factory)
        try:
            differences=[]
            for key in ('body','neural','encoder','sensors','world','environment_history','intervention_history','record_cohort_ids'):
                differences+=exact_differences(final[key],replay.checkpoint()[key],key)
            report['gates']['edit_cancel_recording_replay']=dict(status='PASS' if not differences else 'FAIL',differences=differences)
        finally:replay.close()
        manifest=read_json(a.out/'selected512/manifest.json')
        report['gates']['fixed512_recording']=dict(status='PASS' if len(manifest['cohort_ids'])==512 and
            sum(c['rows'] for c in manifest['chunks'])==10 and manifest['dropped_records']==0 else 'FAIL',
            channels=512,signal_rows=sum(c['rows'] for c in manifest['chunks']),dropped=manifest['dropped_records'],
            first_50ms_wall_seconds=elapsed)
        report['physicalExecuted']=True;persist()
    finally:e.close()
    e=CEngine(g,b,mode='C_STRICT',backend=a.backend)
    try:
        packet=copy.deepcopy(e.last_sensors);packet['odor']=[.8,.1];packet['danger']=.6
        enc=SensoryEncoder(b,42);drive,_,ports=enc.encode(packet,.005,.0001)
        off,_,_=SensoryEncoder(b,42).encode(packet,.005,.0001,['*'])
        means={p['name']:float(np.mean(drive[ids])) for p,ids in b.sensory}
        report['gates']['reviewed_input_boundaries']=dict(status='PASS' if means['odor_DM1_left']>means['odor_DM1_right']>0 and
            means['geosmin_DA2']>0 and not np.any(off) else 'FAIL',mean_drive_mV=means,disabled_drive_zero=not np.any(off),ports=ports,
            scope='synthetic sensor boundary; separate from native behavior')
        comparison=paired(e,a.out/'sensory_off',dict(kind='sensor_off',channels=['*']),seconds=.15,onset=.05,
                          ids=b.spec['sensory'][0]['ids']+b.spec['sensory'][1]['ids']+b.spec['sensory'][2]['ids'])
        report['gates']['native_sensory_off_pair']=comparison
        control=StateStore.load(a.out/'sensory_off/control/final_checkpoint')['neural']
        inhibited=StateStore.load(a.out/'sensory_off/intervention/final_checkpoint')['neural']
        report['gates']['sensory_neural_effect']=dict(status='PASS' if not np.array_equal(control['v'],inhibited['v']) else 'FAIL',
             voltage_max_abs_mV=float(np.max(np.abs(control['v']-inhibited['v']))),behavioral_approval=False)
        persist()
    finally:e.close()
    if a.matrix:
        for mode in ('C_STRICT','C_SHADOW','C_ASSISTED'):
            for seed in (7,19,42):
                base=CEngine(g,b,mode=mode,seed=seed,backend=a.backend)
                try:initial=base.checkpoint();StateStore.save(a.out/f'initial-{mode}-{seed}',initial)
                finally:base.close()
                for case in ('baseline','food_left','food_right','hazard_left','hazard_right','obstacle_front'):
                    name=f'{mode}-{seed}-{case}';directory=a.out/name;started=time.perf_counter()
                    e=CEngine.from_checkpoint(g,b,initial);trace=[]
                    try:
                        e.start_recording(directory,[n['id'] for n in g.nodes if n['cell_type'] in ('ORN_DM1','ORN_DA2','DNp09','DNa02','MDN')])
                        world=default_world();world['sources']=[];world['obstacles']=[]
                        body=e.body.frame()[0];pos=np.asarray(body['position']);forward=np.asarray([row[0] for row in body['basis']]);forward[1]=0;forward/=np.linalg.norm(forward);left=np.array([forward[2],0,-forward[0]])
                        if case.startswith(('food_','hazard_')):
                            kind,side=case.split('_');point=(pos+5*forward+(3 if side=='left' else -3)*left).tolist();point[1]=.7
                            world['sources']=[dict(id='task-source',kind=kind,p=point,strength=2.)]
                        if case=='obstacle_front':
                            point=(pos+6*forward).tolist();point[1]=1.5;world['obstacles']=[dict(id='task-obstacle',p=point,r=1.5)]
                        e.command('load_environment',{'world':world});trace.append(trace_sample(e.frame()))
                        for _ in range(round(a.seconds/.005)):
                            f=e.step(1);sample=trace_sample(f);sample['contact']=bool(e.body.nonfoot_contact());trace.append(sample)
                            if f['fault']:break
                        metrics=behavior_metrics(trace,e.world)
                        supported=case!='obstacle_front' or mode!='C_STRICT'
                        r=dict(name=name,mode=mode,seed=seed,case=case,status='FAILED' if e.fault or metrics['observation_gaps'] else 'COMPLETE',
                               technical_status='FAIL' if e.fault else 'PASS',task_status='NOT_EVALUATED' if supported else 'UNSUPPORTED_SENSORY_CHANNEL',
                               physicalExecuted=True,behavior=metrics,binding_hash=b.hash,seconds=a.seconds,wall_seconds=time.perf_counter()-started)
                        if not e.fault:StateStore.save(directory/'final_checkpoint',e.checkpoint())
                        write_json(directory/'trace.json',trace);write_json(directory/'diagnostic.json',r);report['cases'].append(r)
                    except Exception as exc:
                        report['cases'].append(dict(name=name,status='FAILED',error=str(exc)));persist();raise
                    finally:e.close()
                    persist();print(name,report['cases'][-1]['status'],flush=True)
    report['status']='PASS' if all(v.get('status') in ('PASS','COMPLETE') for v in report['gates'].values()) and all(c['status']=='COMPLETE' for c in report['cases']) else 'FAIL'
    report['task_approval']='NOT_EVALUATED: short diagnostics cannot establish navigation or biological validity'
    persist();print(report['status'],flush=True)
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
