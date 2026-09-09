#!/usr/bin/env python3
"""Alternate immutable source checkouts; compare dynamics and observer/record overhead."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--case',choices=('default','512-recording'),default='default');p.add_argument('--seconds',type=float,default=.15)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.source.resolve() if a.worker else ROOT))
    from flylab.c.storage import StateStore
    from flylab.c.integrity import read_json, write_json
    if a.worker:
        from flylab.c.graph import GraphStore
        from flylab.c.ports import PortBindings
        from flylab.c.engine import CEngine
        from flylab.c.protocol import signal_frame
        g=GraphStore.load(ROOT/'data/fafb783/bundle');b=PortBindings(g,read_json(ROOT/'data/fafb783/bindings.json'))
        cp=StateStore.load(a.checkpoint);e=CEngine.from_checkpoint(g,b,cp)
        try:e.step(1)
        finally:e.close()
        e=CEngine.from_checkpoint(g,b,cp)
        try:
            if a.case=='512-recording':
                ids=[n['id'] for n in g.nodes[:512]];e.subscribe(ids);e.start_recording(a.out/'recording',ids)
            start=time.perf_counter();frames=[];binary_bytes=0
            for _ in range(round(a.seconds/.005)):
                f=e.step(1);binary_bytes+=len(signal_frame(e));json.dumps(f,allow_nan=False,separators=(',',':'))
                frames.append({k:f[k] for k in ('tick','body','physics','sensors','command','motor_rates_Hz')})
            seconds=time.perf_counter()-start;e.stop_recording()
            StateStore.save(a.out/'final',e.checkpoint());write_json(a.out/'trace.json',frames)
            write_json(a.out/'timing.json',dict(wall_seconds=seconds,model_seconds=a.seconds,binary_bytes=binary_bytes,
                                              physicalExecuted=True,case=a.case))
        finally:e.close()
        return
    from tools.benchmark_c_runtime import exact_differences, recording_differences
    report=dict(status='RUNNING',cases=[],scope='warm CEngine + compact JSON + FLC3, no network/browser',
                ignored_checkpoint_metadata=['environment_history','environment_updated_tick','intervention_history','record_cohort_ids'],
                core_fields=['body','neural','encoder','sensors','legacy','supervisor','world','config','last_command','last_sensors','last_motor_rates','pending','active'])
    for case in ('default','512-recording'):
        runs={};deltas=[]
        for trial in range(2):
            for label in (('baseline','current') if trial==0 else ('current','baseline')):
                out=a.out/f'{case}-{trial}-{label}'
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--baseline',str(a.baseline),'--source',str(a.baseline if label=='baseline' else ROOT),
                     '--checkpoint',str(a.checkpoint),'--out',str(out),'--case',case,'--seconds',str(a.seconds)]
                with (a.out/(out.name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                runs[trial,label]=out;print(out.name,read_json(out/'timing.json')['wall_seconds'],flush=True)
        reference=StateStore.load(runs[0,'baseline']/'final');trace=read_json(runs[0,'baseline']/'trace.json')
        for (trial,label),path in runs.items():
            state=StateStore.load(path/'final')
            for key in report['core_fields']:deltas+=exact_differences(reference[key],state[key],f'{trial}-{label}.{key}')
            deltas+=exact_differences(trace,read_json(path/'trace.json'),f'{trial}-{label}.trace')
            if case=='512-recording':deltas+=recording_differences(runs[0,'baseline']/'recording',path/'recording')
        times={label:[read_json(runs[trial,label]/'timing.json')['wall_seconds'] for trial in range(2)] for label in ('baseline','current')}
        medians={k:statistics.median(v) for k,v in times.items()}
        report['cases'].append(dict(case=case,status='PASS' if not deltas else 'FAIL',core_differences=deltas,
                                   wall_seconds=times,median_wall_seconds=medians,current_over_baseline=medians['current']/medians['baseline']))
        write_json(a.out/'report.json',report)
    report['status']='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL';write_json(a.out/'report.json',report)


if __name__=='__main__':main()
