#!/usr/bin/env python3
"""Compare CPU and a real CUDA/MPS backend from identical states."""
from pathlib import Path
import argparse
import copy
import json
import platform
import statistics
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.neural import create_backend, BACKEND_CHOICES
from flylab.c.backend_selection import resolve_backend
from flylab.c.engine import CEngine
from flylab.c.storage import StateStore
from flylab.c.integrity import file_hash


def neural_state(source, backend):
    state=copy.deepcopy(source)
    state['backend']=backend.backend
    state.pop('backend_runtime',None)
    if hasattr(backend,'runtime_identity'):state['backend_runtime']=dict(backend.runtime_identity)
    return state


def differences(a,b):
    result={}
    for key in ('v','h','rate','queue','spike_count','refractory_until'):
        result[key]=dict(max_abs=float(np.max(np.abs(a[key].astype(np.float64)-b[key]))),
                         exact=bool(np.array_equal(a[key],b[key])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--graph',type=Path,default=Path('data/fafb783/bundle'))
    p.add_argument('--bindings',type=Path,default=Path('data/fafb783/bindings.json'))
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto')
    p.add_argument('--strict-checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--neural-seconds',type=float,default=.05)
    p.add_argument('--physics-seconds',type=float,default=.1)
    args=p.parse_args()
    args.backend=resolve_backend(args.backend)
    if args.backend=='exp_lif_cpu_reference':p.error('A working GPU backend is required for this comparison')
    if not 1<=args.repeats<=10 or not 0<args.neural_seconds<=1 or not 0<=args.physics_seconds<=1:p.error('Invalid bounded benchmark duration')
    if any(abs(v/.005-round(v/.005))>1e-8 for v in (args.neural_seconds,args.physics_seconds)):p.error('Use integer 5ms periods')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='flylab.c.gpu_benchmark.v1',status='RUNNING',platform=platform.platform(),
                source_checkpoint=str(args.checkpoint),source_manifest_sha256=file_hash(args.checkpoint/'manifest.json'),
                physicalExecuted=False,biologicalValidation=False,neural=[],physical=[])
    def save(): (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        graph=GraphStore.load(args.graph)
        bindings=PortBindings(graph,json.loads(args.bindings.read_text()))
        source=StateStore.load(args.checkpoint)
        report.update(graph_hash=graph.hash,simulated_neurons=graph.n,edges=len(graph.weights))
        capture=graph.resolve(source['subscription'])
        drive=np.zeros(graph.n,dtype=np.float32)
        drive[graph.resolve(bindings.spec['sensory'][0]['ids'])]=10.
        report['neural_input']='Same fixed 10mV ORN_DM1 drive, starting from active user checkpoint; no body'
        cpu=create_backend(graph);gpu=create_backend(graph,backend=args.backend)
        report['gpu_runtime']=dict(gpu.runtime_identity)
        backends=(cpu,gpu);snapshots={};events={}
        for backend in backends:
            state=neural_state(source['neural'],backend)
            backend.restore(state)
            began=time.perf_counter();backend.advance(drive,50,capture);cold=time.perf_counter()-began
            times=[]
            for trial in range(args.repeats):
                backend.restore(state);all_events=[];began=time.perf_counter()
                for _ in range(round(args.neural_seconds/.005)):
                    backend.advance(drive,50,capture);all_events+=backend.last_events
                elapsed=time.perf_counter()-began;times.append(elapsed)
                final=backend.snapshot()
                if trial and not all(np.array_equal(final[k],snapshots[backend.backend][k]) for k in ('v','h','rate','queue','spike_count')):
                    raise AssertionError('Repeated execution of one backend was not deterministic')
                snapshots[backend.backend]=final;events[backend.backend]=all_events
            row=dict(backend=backend.backend,model_seconds=args.neural_seconds,
                     cold_5ms_seconds=cold,wall_seconds=times,median_wall_seconds=statistics.median(times),
                     sim_wall_ratio=args.neural_seconds/statistics.median(times))
            report['neural'].append(row);print(json.dumps(row),flush=True);save()
        a,b=snapshots[cpu.backend],snapshots[gpu.backend]
        diff=differences(a,b)
        parity=(all(diff[k]['max_abs']<=1e-4 for k in ('v','h','queue')) and diff['rate']['max_abs']<=1e-3
                and diff['spike_count']['exact'] and diff['refractory_until']['exact'] and events[cpu.backend]==events[gpu.backend])
        report['neural_parity']=dict(status='PASS' if parity else 'FAIL',differences=diff,
            selected_spike_times_equal=events[cpu.backend]==events[gpu.backend],
            limits=dict(voltage_current_queue_atol=1e-4,rate_atol_Hz=1e-3,spike_counts_timers_events='exact'))
        report['neural_speedup']=report['neural'][0]['median_wall_seconds']/report['neural'][1]['median_wall_seconds']
        save()
        if args.physics_seconds:
            sources=[('user_continuation',source)]
            if args.strict_checkpoint:sources.append(('strict_continuation',StateStore.load(args.strict_checkpoint)))
            for name,state in sources:
                trials=[];finals={}
                for backend_name in (cpu.backend,gpu.backend):
                    engine=CEngine.from_checkpoint(graph,bindings,state,backend_override=backend_name)
                    try:
                        began=time.perf_counter();trace=[]
                        for _ in range(round(args.physics_seconds/.005)):
                            f=engine.step(1)
                            trace.append(dict(tick=f['tick'],position=f['body']['position'],yaw=f['body']['yaw'],
                                              command=f['command'],fault=f['fault']))
                            if f['fault']:raise RuntimeError(f['fault'])
                        elapsed=time.perf_counter()-began
                        row=dict(backend=backend_name,mode=engine.mode,model_seconds=args.physics_seconds,
                            wall_seconds=elapsed,sim_wall_ratio=args.physics_seconds/elapsed,
                            performance=f['performance'],fault=f['fault'],trace=trace)
                        finals[backend_name]=engine.checkpoint()
                        StateStore.save(args.out/(name+'-'+backend_name),finals[backend_name])
                        trials.append(row);report['physicalExecuted']=True
                        print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
                    finally:engine.close()
                left,right=trials
                max_position=max(float(np.linalg.norm(np.asarray(a['position'])-b['position'])) for a,b in zip(left['trace'],right['trace']))
                max_yaw=max(abs(a['yaw']-b['yaw']) for a,b in zip(left['trace'],right['trace']))
                body_state=float(np.max(np.abs(np.asarray(finals[cpu.backend]['body']['state'])-finals[gpu.backend]['body']['state'])))
                item=dict(name=name,runs=trials,speedup=left['wall_seconds']/right['wall_seconds'],
                          max_position_difference_mm=max_position,max_yaw_difference_rad=max_yaw,
                          body_integration_state_max_abs=body_state,
                          neural_difference=differences(finals[cpu.backend]['neural'],finals[gpu.backend]['neural']),
                          status='PASS' if max_position<=.01 and max_yaw<=.01 else 'FAIL',
                          limits=dict(position_mm=.01,yaw_rad=.01),scope='Short physical continuation; not long-horizon behavioral equivalence')
                report['physical'].append(item);save()
        report['status']='PASS' if parity and all(r['status']=='PASS' for r in report['physical']) else 'FAIL'
        save();print('Result:',report['status'],'neural speedup',report['neural_speedup'],flush=True)
        return 0 if report['status']=='PASS' else 1
    except Exception as exc:
        report.update(status='ERROR',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':raise SystemExit(main())
