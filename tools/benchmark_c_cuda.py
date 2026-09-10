#!/usr/bin/env python3
"""Bounded CUDA ablation and repeated CPU/GPU physical continuations.

All variants start from the same preserved checkpoint. Compilation, graph
warmup and snapshot I/O are outside neural timing. Physical timing includes
the first control-period graph capture and every body/observation step.
"""
import argparse
import copy
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.neural import ExpLIF,LIFParameters
from flylab.c.neural_cuda import CudaLIF
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.engine import CEngine
from flylab.c.storage import StateStore,runtime_versions
from flylab.c.integrity import read_json,write_json,file_hash
from tools.benchmark_c_backends import neural_state,differences


def parity(reference,actual,reference_events,events):
    errors=differences(reference,actual)
    passed=(all(errors[k]['max_abs']<=1e-4 for k in ('v','h','queue')) and
            errors['rate']['max_abs']<=1e-3 and
            all(errors[k]['exact'] for k in ('spike_count','refractory_until')) and reference_events==events)
    return dict(status='PASS' if passed else 'FAIL',differences=errors,selected_events_equal=reference_events==events,
                limits=dict(voltage_current_queue_mV=1e-4,rate_Hz=1e-3,counts_timers_events='exact'))


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',schema='flylab.cuda-benchmark.v1',platform=platform.platform(),
        versions=runtime_versions(),source_checkpoint=str(args.checkpoint),
        source_checkpoint_manifest_sha256=file_hash(args.checkpoint/'manifest.json'),
        model_seconds=args.seconds,repeats=args.repeats,neural={},physical=[],
        physicalExecuted=False,biological_validation=False)
    def save():write_json(args.out/'report.json',report)
    save()
    try:
        graph=GraphStore.load(args.graph);bindings=PortBindings(graph,read_json(args.bindings))
        source=StateStore.load(args.checkpoint);parameters=LIFParameters(**source['parameters'])
        report.update(graph_hash=graph.hash,neurons=graph.n,edges=len(graph.weights),parameters=source['parameters'])
        capture=graph.resolve(source['subscription']);steps=round(.005/parameters.dt);periods=round(args.seconds/.005)
        drive=np.zeros(graph.n,np.float32);drive[graph.resolve(bindings.spec['sensory'][0]['ids'])]=10.
        np.savez(args.out/'inputs.npz',drive=drive,capture=capture)
        report['neural_input']='Fixed 10 mV first sensory population; identical active checkpoint and every neural tick'
        reference=ExpLIF(graph,parameters);reference.restore(neural_state(source['neural'],reference));reference_events=[]
        for _ in range(periods):
            reference.advance(drive,steps,capture);reference_events+=reference.last_events
        expected=reference.snapshot()
        implementations={
            'generic_cupy':ExpLIF(graph,parameters,'exp_lif_cuda'),
            'cuda_kernels':CudaLIF(graph,parameters,use_graphs=False),
            'cuda_graphs':CudaLIF(graph,parameters),
        }
        times={name:[] for name in implementations};finals={};event_rows={};repeat_equal={name:True for name in implementations}
        for name,neural in implementations.items():
            neural.restore(neural_state(source['neural'],neural));neural.advance(drive,steps,capture)
        for trial in range(args.repeats):
            order=list(implementations) if trial%2==0 else list(implementations)[::-1]
            for name in order:
                neural=implementations[name];neural.restore(neural_state(source['neural'],neural));events=[]
                began=time.perf_counter()
                for _ in range(periods):
                    neural.advance(drive,steps,capture);events+=neural.last_events
                elapsed=time.perf_counter()-began;times[name].append(elapsed)
                final=neural.snapshot()
                if name in finals:
                    repeat_equal[name]&=all(np.array_equal(final[k],finals[name][k]) for k in ('v','h','rate','queue','spike_count','refractory_until')) and events==event_rows[name]
                finals[name]=final;event_rows[name]=events
                print(name,trial,round(elapsed,6),flush=True)
        for name,neural in implementations.items():
            row=dict(wall_seconds=times[name],median_wall_seconds=statistics.median(times[name]),
                repeated_state_and_events_exact=bool(repeat_equal[name]),parity=parity(expected,finals[name],reference_events,event_rows[name]),
                runtime=getattr(neural,'runtime_identity',{}),graph_builds=getattr(neural,'graph_builds',0),
                graph_replays=getattr(neural,'graph_replays',0))
            report['neural'][name]=row
            StateStore.save(args.out/('neural-'+name),finals[name]);write_json(args.out/(name+'-events.json'),event_rows[name])
        report['speedup_over_generic_cuda']=report['neural']['generic_cupy']['median_wall_seconds']/report['neural']['cuda_graphs']['median_wall_seconds']
        report['graph_replay_speedup']=report['neural']['cuda_kernels']['median_wall_seconds']/report['neural']['cuda_graphs']['median_wall_seconds']
        save()
        if args.physics:
            for mode in ('C_SHADOW','C_STRICT','C_ASSISTED'):
                initial=copy.deepcopy(source);initial['mode']=mode
                timings={name:[] for name in ('exp_lif_cpu_reference','exp_lif_cuda')}
                traces={};states={};repeat_deterministic=True
                for trial in range(args.repeats):
                    order=list(timings) if trial%2==0 else list(timings)[::-1]
                    for backend in order:
                        engine=CEngine.from_checkpoint(graph,bindings,initial,backend_override=backend)
                        try:
                            trace=[];began=time.perf_counter()
                            for _ in range(periods):
                                frame=engine.step(1)
                                if frame['fault']:raise RuntimeError(str(frame['fault']))
                                trace.append(dict(tick=frame['tick'],position=frame['body']['position'],yaw=frame['body']['yaw'],
                                                  command=frame['command'],events=copy.deepcopy(engine.selected_events)))
                            elapsed=time.perf_counter()-began;timings[backend].append(elapsed)
                            final=engine.checkpoint()
                            if backend in states:
                                repeat_deterministic&=np.array_equal(states[backend]['body']['state'],final['body']['state'])
                                repeat_deterministic&=traces[backend]==trace
                            states[backend]=final;traces[backend]=trace
                            path=args.out/f'{mode}-{trial}-{backend}'
                            StateStore.save(path,final)
                            write_json(args.out/(path.name+'-trace.json'),trace)
                            print(mode,backend,trial,round(elapsed,6),flush=True)
                        finally:engine.close()
                cpu,gpu=timings
                position=max(float(np.linalg.norm(np.array(a['position'])-b['position'])) for a,b in zip(traces[cpu],traces[gpu]))
                yaw=max(abs(a['yaw']-b['yaw']) for a,b in zip(traces[cpu],traces[gpu]))
                events=lambda backend:[event for row in traces[backend] for event in row['events']]
                check=parity(states[cpu]['neural'],states[gpu]['neural'],events(cpu),events(gpu))
                medians={name:statistics.median(values) for name,values in timings.items()}
                row=dict(mode=mode,wall_seconds=timings,median_wall_seconds=medians,speedup=medians[cpu]/medians[gpu],
                    sim_wall_ratio=args.seconds/medians[gpu],max_position_difference_mm=position,max_yaw_difference_rad=yaw,
                    final_body_state_exact=bool(np.array_equal(states[cpu]['body']['state'],states[gpu]['body']['state'])),
                    repeated_body_and_trace_exact=bool(repeat_deterministic),neural_parity=check,
                    status='PASS' if check['status']=='PASS' and position<=.01 and yaw<=.01 and repeat_deterministic else 'FAIL')
                report['physical'].append(row);report['physicalExecuted']=True;save()
        optimized=report['neural']['cuda_graphs']
        report['status']='PASS' if optimized['parity']['status']=='PASS' and optimized['repeated_state_and_events_exact'] and all(r['status']=='PASS' for r in report['physical']) else 'FAIL'
        report['scope']='Short fixed-input and physical continuation; not long-horizon behavior or biological validation'
        save();return report
    except Exception as exc:
        report.update(status='ERROR',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','bindings','checkpoint','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=.05);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--physics',action='store_true');args=parser.parse_args()
    if not .005<=args.seconds<=.2 or abs(args.seconds/.005-round(args.seconds/.005))>1e-8 or not 1<=args.repeats<=7:
        parser.error('Use 5 ms multiples up to 0.2 model seconds and 1..7 repeats')
    raise SystemExit(0 if run(args)['status']=='PASS' else 1)
