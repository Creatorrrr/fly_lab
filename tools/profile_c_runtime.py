#!/usr/bin/env python3
"""Separate steady throughput, instrumented diagnostics and exact input replay."""
from pathlib import Path
import argparse, copy, cProfile, json, pstats, sys, time

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=root)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=.1)
    p.add_argument('--warmup-seconds', type=float, default=.02)
    p.add_argument('--repeats', type=int, default=3)
    local = root/'flylab.local.json'
    defaults = json.loads(local.read_text(encoding='utf-8')) if local.is_file() else {}
    p.add_argument('--graph', type=Path, default=root/defaults.get('graph','data/fafb783/bundle'))
    p.add_argument('--bindings', type=Path, default=root/defaults.get('bindings','data/fafb783/bindings.json'))
    args = p.parse_args()
    for value in (args.seconds,args.warmup_seconds):
        if not .005 <= value <= 5 or abs(value/.005-round(value/.005)) > 1e-8:
            p.error('Use 5 ms multiples in .005..5 seconds')
    if not 1 <= args.repeats <= 20: p.error('Use 1..20 repeats')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.source_root.resolve()))
    import numpy as np
    from flylab.c.graph import GraphStore
    from flylab.c.ports import PortBindings
    from flylab.c.engine import CEngine
    from flylab.c.storage import StateStore
    from flylab.c.integrity import file_hash
    graph = GraphStore.load(args.graph)
    bindings = PortBindings(graph,json.loads(args.bindings.read_text(encoding='utf-8')))
    state = StateStore.load(args.checkpoint)
    controls = round(args.seconds/.005)

    def new_engine():
        start=time.perf_counter()
        engine=CEngine.from_checkpoint(graph,bindings,state)
        build=time.perf_counter()-start
        start=time.perf_counter()
        engine.step(round(args.warmup_seconds/.005))
        if engine.fault or engine.body.fault:
            engine.close()
            raise RuntimeError('Warm-up fault')
        return engine,build,time.perf_counter()-start

    runs=[]
    for repeat in range(args.repeats):
        e,built,warm=new_engine()
        try:
            latency=[]
            start=time.perf_counter()
            for _ in range(controls):
                t=time.perf_counter()
                e.step(1)
                latency.append(time.perf_counter()-t)
                if e.fault or e.body.fault: raise RuntimeError('Steady execution fault')
            elapsed=time.perf_counter()-start
            runs.append(dict(repeat=repeat,build_s=built,warmup_s=warm,wall_s=elapsed,
                sim_per_wall=args.seconds/elapsed,control_latency_median_s=float(np.median(latency)),
                control_latency_p95_s=float(np.percentile(latency,95))))
        finally: e.close()

    e,built,warm=new_engine()
    b,n=e.body,e.neural
    try:
        body_start=b.snapshot()
        neural_start=n.snapshot() if n else None
        physical_inputs,neural_inputs,cuda_events=[],[],[]
        original_step=b._step_physics
        def traced_step():
            physical_inputs.append(tuple(np.array(getattr(b.d,k),copy=True)
                                         for k in ('ctrl','xfrc_applied','qfrc_applied')))
            return original_step()
        b._step_physics=traced_step
        latest={}
        if n:
            intervention_method=n.set_interventions
            def interventions(*a,**kw):
                latest.update(args=copy.deepcopy(a),kwargs=copy.deepcopy(kw))
                return intervention_method(*a,**kw)
            n.set_interventions=interventions
            method_name='begin_advance' if hasattr(n,'begin_advance') else 'advance'
            neural_method=getattr(n,method_name)
            def traced_neural(drive,steps,capture=(),pulses=None):
                neural_inputs.append((copy.deepcopy((drive,steps,capture,pulses)),copy.deepcopy(latest)))
                if n.backend!='exp_lif_cuda': return neural_method(drive,steps,capture,pulses)
                cp=n.xp
                begin,end=cp.cuda.Event(),cp.cuda.Event()
                cp.cuda.nvtx.RangePush('FLY LAB neural submit')
                try:
                    begin.record(n.stream)
                    result=neural_method(drive,steps,capture,pulses)
                    end.record(n.stream)
                    cuda_events.append((begin,end))
                    return result
                finally: cp.cuda.nvtx.RangePop()
            setattr(n,method_name,traced_neural)
        mj=b.mj
        old_timer=mj.get_mjcb_time()
        profiler=cProfile.Profile()
        try:
            mj.set_mjcb_time(time.perf_counter)
            b.d.timer.duration[:]=0.
            b.d.timer.number[:]=0
            start=time.perf_counter()
            profiler.enable()
            for _ in range(controls):
                e.step(1)
                if e.fault or b.fault: raise RuntimeError('Instrumented execution fault')
            profiler.disable()
            profiled_s=time.perf_counter()-start
            timer=[dict(name=mj.mjtTimer(i).name,seconds=float(t.duration),calls=int(t.number))
                   for i,t in enumerate(b.d.timer)]
            target_q,target_v=b.d.qpos.copy(),b.d.qvel.copy()
            target_neural=n.snapshot() if n else None
            gpu_ms=[float(n.xp.cuda.get_elapsed_time(a,z)) for a,z in cuda_events]
            StateStore.save(args.out/'final_checkpoint',e.checkpoint())
            profiler.dump_stats(str(args.out/'runtime.pstats'))
            with (args.out/'runtime.txt').open('w',encoding='utf-8') as out:
                pstats.Stats(profiler,stream=out).sort_stats('cumulative').print_stats(60)
            identity=b.physics_identity()
        finally:
            profiler.disable()
            mj.set_mjcb_time(old_timer)
            b._step_physics=original_step
            if n:
                setattr(n,method_name,neural_method)
                n.set_interventions=intervention_method

        b.restore(body_start)
        start=time.perf_counter()
        for i,inputs in enumerate(physical_inputs):
            for key,value in zip(('ctrl','xfrc_applied','qfrc_applied'),inputs):
                getattr(b.d,key)[:]=value
            b._step_physics()
            if (i+1)%50==0: b._forward_physics()
        body_s=time.perf_counter()-start
        body_equal=bool(np.array_equal(target_q,b.d.qpos) and np.array_equal(target_v,b.d.qvel))
        neural_s,neural_equal=None,None
        if n:
            n.restore(neural_start)
            start=time.perf_counter()
            for inputs,intervention in neural_inputs:
                n.set_interventions(*intervention.get('args',()),**intervention.get('kwargs',{}))
                n.advance(*inputs)
            neural_s=time.perf_counter()-start
            after=n.snapshot()
            neural_equal=all(np.array_equal(after[k],target_neural[k]) for k in
                             ('v','h','rate','queue','refractory_until','spike_count'))
        np.savez_compressed(args.out/'physical-inputs.npz',
            **{key:np.asarray([row[i] for row in physical_inputs])
               for i,key in enumerate(('ctrl','xfrc_applied','qfrc_applied'))})
        median=float(np.median([r['wall_s'] for r in runs]))
        result=dict(schema='flylab.runtime-profile.v2',
            status='PASS' if body_equal and neural_equal is not False else 'FAIL_REPLAY',
            source_root=str(args.source_root.resolve()),checkpoint=str(args.checkpoint),
            checkpoint_manifest_sha256=file_hash(args.checkpoint/'manifest.json'),
            model_seconds=args.seconds,warmup_model_seconds=args.warmup_seconds,
            steady_uninstrumented=dict(runs=runs,wall_median_s=median,sim_per_wall=args.seconds/median),
            instrumented=dict(wall_s=profiled_s,mujoco_timers=timer,cuda_stream_intervals_ms=gpu_ms,
                semantics='Diagnostic only: includes tracing/profiling overhead. CUDA events include stream transfers; CPU neural_s is submission plus residual wait.'),
            isolated_replay=dict(physics_wall_s=body_s,physics_exact=body_equal,
                neural_wall_s=neural_s,neural_exact=neural_equal,
                semantics='Recorded inputs, integration only. Do not add overlapping component times to predict end-to-end speed.'),
            physics=identity,real_time_60s_validation=False)
        (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='instrumented'},indent=2))
        return 0 if result['status']=='PASS' else 1
    finally: e.close()

if __name__=='__main__':
    raise SystemExit(main())
