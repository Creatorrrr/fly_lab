#!/usr/bin/env python3
"""Alternate frozen and current MPS runtimes from identical physical checkpoints.

Workers use separate processes to avoid mixing module versions. Initialization,
shader/JIT warmup and artifact export are excluded from continuation timing.
All model steps, native contacts, recordings and full neural states are retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
import copy

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def worker(args):
    sys.path.insert(0,str(args.source_root.resolve()))
    from flylab.c.graph import GraphStore
    from flylab.c.ports import PortBindings
    from flylab.c.engine import CEngine
    from flylab.c.storage import StateStore
    args.out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    graph=GraphStore.load(ROOT/'data/fafb783/bundle')
    bindings=PortBindings(graph,json.loads((ROOT/'data/fafb783/bindings.json').read_text()))
    source=StateStore.load(args.checkpoint)
    if args.case=='assisted':source['mode']='C_ASSISTED'
    override=None if source['neural']['backend']=='exp_lif_mps' else 'exp_lif_mps'
    warm=CEngine.from_checkpoint(graph,bindings,source,backend_override=override)
    try:warm.step(1)
    finally:warm.close()
    engine=CEngine.from_checkpoint(graph,bindings,source,backend_override=override)
    initialization=time.perf_counter()-began
    report=dict(status='RUNNING',source_root=str(args.source_root.resolve()),
                source_checkpoint=str(args.checkpoint.resolve()),
                source_manifest_sha256=hashlib.sha256((args.checkpoint/'manifest.json').read_bytes()).hexdigest(),
                mode=engine.mode,case=args.case,backend=engine.neural.backend,
                graph_hash=graph.hash,neurons=graph.n,edges=len(graph.weights),
                body_model_hash=source['body_model_hash'],
                initialization_and_warmup_seconds=initialization,model_seconds=args.seconds)
    write(args.out/'report.json',report)
    try:
        if args.case=='recording':engine.start_recording(args.out/'recording')
        trace=[];began=time.perf_counter()
        for _ in range(round(args.seconds/.005)):
            f=engine.step(1)
            if f['fault']:raise RuntimeError(f['fault'])
            trace.append({key:f[key] for key in ('tick','body','physics','sensors','command','motor_rates_Hz')})
            trace[-1]['selected_events']=engine.selected_events.copy()
        report['wall_seconds']=time.perf_counter()-began
        report['sim_wall_ratio']=args.seconds/report['wall_seconds']
        report['performance']=f['performance']
        engine.stop_recording()
        StateStore.save(args.out/'final_checkpoint',engine.checkpoint())
        write(args.out/'trace.json',trace)
        report.update(status='PASS',physical_executed=True,end_tick=engine.tick)
        write(args.out/'report.json',report)
    except Exception as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
        write(args.out/'report.json',report);raise
    finally:engine.close()
    return 0


def exact_differences(a,b,path=''):
    import numpy as np
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):
            return [path]
    elif isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path+'.keys']
        return [p for k in a for p in exact_differences(a[k],b[k],path+'.'+k)]
    elif isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b):return [path+'.length']
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in exact_differences(x,y,f'{path}[{i}]')]
    elif a!=b:return [path]
    return []

def baseline_state(state):
    """Compare unchanged dynamics across an additive telemetry schema change.

    Only the legacy concentration model with metabolism disabled is eligible.
    Every neural/body/encoder/RNG/control field remains in the exact comparison.
    """
    state=copy.deepcopy(state)
    if state['app_version'] not in ('0.3.0','0.4.0'):raise ValueError('Unreviewed application migration')
    state['app_version']='0.3-to-0.4-validated-baseline'
    if state.pop('metabolism',None) is not None:raise ValueError('Metabolism is outside legacy equivalence scope')
    sensors=state['sensors']
    if sensors.pop('transduction',{'kind':'legacy-clipped-v1'})!={'kind':'legacy-clipped-v1'}:
        raise ValueError('Changed sensory model cannot use legacy equivalence')
    if sensors.pop('previous_silhouette',None) is not None:raise ValueError('Optical state cannot be excluded')
    sensors.pop('diagnostics',None)
    for row in state['last_ports']:
        for key in ('feature','requested','clipped'):row.pop(key,None)
    return state


def recording_differences(a,b):
    import numpy as np
    failures=[]
    for filename in ('events.jsonl','body.jsonl'):
        rows=[]
        for directory in (a,b):
            data=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
            for row in data:
                if row.get('kind')=='recording_start':row['details']['path']='<run directory>'
            rows.append(data)
        failures+=exact_differences(*rows,filename)
    manifests=[json.loads((d/'manifest.json').read_text()) for d in (a,b)]
    for m in manifests:
        if m['status']!='COMPLETE' or m['dropped_records']!=0:failures.append('incomplete recording')
    filenames=[[c[k] for c in m['chunks'] for k in ('file','ticks_file')] for m in manifests]
    if filenames[0]!=filenames[1]:return failures+['recording chunk layout']
    for filename in filenames[0]:
        failures+=exact_differences(np.load(a/filename),np.load(b/filename),filename)
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--strict-checkpoint',type=Path)
    p.add_argument('--baseline-root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=.3)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--source-root',type=Path,default=ROOT)
    p.add_argument('--case',choices=('shadow','strict','assisted','recording'),default='shadow')
    p.add_argument('--additive-telemetry',action='store_true',help='Compare legacy dynamics with explicit additive telemetry exclusions')
    args=p.parse_args()
    if not 0<args.seconds<=5 or abs(args.seconds/.005-round(args.seconds/.005))>1e-8 or not 1<=args.repeats<=10:
        p.error('Use 5ms multiples up to five model seconds and 1–10 repeats')
    if args.worker:return worker(args)
    if not args.baseline_root:p.error('--baseline-root is required')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT))
    from flylab.c.storage import StateStore
    cases=['shadow','assisted','recording']
    if args.strict_checkpoint:cases.insert(1,'strict')
    report=dict(schema='flylab.c.runtime_benchmark.v1',status='RUNNING',platform=platform.platform(),
                timing_scope='end-to-end CEngine.step(1), warm runtime, one physical world, no browser',
                initialization_excluded=True,baseline='previous MPS implementation',cases=[])
    write(args.out/'report.json',report)
    for case in cases:
        rows={'baseline':[],'optimized':[]};runs={};source=args.strict_checkpoint if case=='strict' else args.checkpoint
        for trial in range(args.repeats):
            order=('baseline','optimized') if trial%2==0 else ('optimized','baseline')
            for label in order:
                path=args.out/f'{case}-{trial}-{label}'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--source-root',
                         str(args.baseline_root if label=='baseline' else ROOT),
                         '--checkpoint',str(source),'--case',case,'--seconds',str(args.seconds),'--out',str(path)]
                with (args.out/f'{path.name}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                row=json.loads((path/'report.json').read_text());row['artifacts']=str(path.resolve())
                rows[label].append(row);runs[trial,label]=path
                print(json.dumps(dict(case=case,trial=trial,version=label,wall_seconds=row['wall_seconds'])),flush=True)
        diffs=[]
        reference=StateStore.load(runs[0,'baseline']/'final_checkpoint')
        if args.additive_telemetry:reference=baseline_state(reference)
        reference_trace=json.loads((runs[0,'baseline']/'trace.json').read_text())
        for (trial,label),path in runs.items():
            current=StateStore.load(path/'final_checkpoint')
            if args.additive_telemetry:current=baseline_state(current)
            diff=exact_differences(reference,current,'checkpoint')
            diff+=exact_differences(reference_trace,json.loads((path/'trace.json').read_text()),'trace')
            if case=='recording':diff+=recording_differences(runs[0,'baseline']/'recording',path/'recording')
            if diff:diffs.append(dict(trial=trial,version=label,paths=diff[:50],total=len(diff)))
        medians={label:statistics.median(row['wall_seconds'] for row in values) for label,values in rows.items()}
        result=dict(case=case,mode=rows['optimized'][0]['mode'],runs=rows,median_wall_seconds=medians,
                    model_seconds=args.seconds,speedup=medians['baseline']/medians['optimized'],
                    optimized_sim_wall_ratio=args.seconds/medians['optimized'],
                    status='PASS' if not diffs else 'FAIL',exact_state_and_trace=not diffs,differences=diffs)
        result['excluded_observational_fields']=['app_version=0.3.0/0.4.0 compatibility migration','metabolism=None','sensors.transduction=legacy-clipped-v1','sensors.previous_silhouette=None','sensors.diagnostics','last_ports.feature/requested/clipped'] if args.additive_telemetry else []
        result['wall_distribution']={label:dict(min=min(v),max=max(v),p95=sorted(v)[max(0,__import__('math').ceil(.95*len(v))-1)]) for label,values in rows.items() for v in [[r['wall_seconds'] for r in values]]}
        report['cases'].append(result);write(args.out/'report.json',report)
        print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)
    report['status']='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL'
    write(args.out/'report.json',report)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
