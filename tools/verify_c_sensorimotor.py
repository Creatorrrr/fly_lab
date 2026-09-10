#!/usr/bin/env python3
"""Actual BANC/MPS/MuJoCo loop, ablations, joint direction and continuation."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.engine import CEngine
from flylab.c.integrity import read_json, write_json
from flylab.c.storage import StateStore
from flylab.c.body_identity import source_identity
from flylab.c.behavior import trace_sample
from flylab.c.tasks import evaluate
from flylab.sensors import default_world


def same(a,b):
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def run(graph_path, binding_path, out, seconds=.5):
    if not .05 <= seconds <= 30 or abs(seconds/.005-round(seconds/.005))>1e-8:
        raise ValueError('Duration must be 0.05..30 s on a control boundary')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=source_identity()
    shutil.copytree(Path(__file__).resolve().parents[1]/'flylab',out/'source/flylab',ignore=shutil.ignore_patterns('__pycache__'))
    graph=GraphStore.load(graph_path);bindings=PortBindings(graph,read_json(binding_path))
    write_json(out/'bindings.json',bindings.spec)
    reports=[]
    for case in ('feedback','feedback_off','motor_disconnected','flexor','extensor','DNg100'):
        world=default_world();world['sources']=[];world['obstacles']=[]
        engine=CEngine(graph,bindings,world=world,mode='C_STRICT',backend='exp_lif_mps')
        directory=out/case;directory.mkdir()
        try:
            duration=round((seconds+.02)/.005)
            if case in ('feedback_off','flexor','extensor'):
                engine.schedule(dict(kind='sensor_off',channels=['*'],duration_controls=duration))
            if case=='motor_disconnected':
                engine.schedule(dict(kind='motor_disconnect',duration_controls=duration))
            if case in ('flexor','extensor'):
                muscle='tibia_'+case+'_muscle'
                ids=next(m['ids'] for m in bindings.spec['neuromuscular']['rows'][0]['muscles'] if m['muscle']==muscle)
                engine.schedule(dict(kind='stimulate',ids=ids,amplitude_mV=20.,duration_controls=duration))
            if case=='DNg100':
                ids=[n['id'] for n in graph.nodes if n['cell_type']=='DNg100']
                engine.schedule(dict(kind='stimulate',ids=ids,amplitude_mV=20.,duration_controls=duration))
            ids=engine.neuromuscular.motor_indices
            engine.subscribe([graph.nodes[int(i)]['id'] for i in ids])
            initial=engine.checkpoint()
            trace=[trace_sample(engine.frame())];signals=[]
            for k in range(round(seconds/.005)):
                frame=engine.step();trace.append(trace_sample(frame))
                r=engine.neural.readout(ids)
                signals.append(dict(tick=engine.tick,events=engine.neural.last_events,
                                    **{key:v.tolist() for key,v in r.items()}))
                if frame['fault']:break
            final=engine.checkpoint() if not engine.fault else None
            restored_equal=None
            if final:
                StateStore.save(directory/'final',final)
                other=CEngine.from_checkpoint(graph,bindings,final)
                try:
                    engine.step(4);other.step(4)
                    restored_equal=all(same(a,b) for a,b in ((engine.neural.snapshot(),other.neural.snapshot()),
                        (engine.body.snapshot(),other.body.snapshot()),(engine.neuromuscular.snapshot(),other.neuromuscular.snapshot())))
                finally:other.close()
            points=np.array(frame['body']['legs']['lf']);a=points[1]-points[2];b=points[3]-points[2]
            knee=float(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
            result=dict(case=case,physicalExecuted=True,source=source,provenance=engine.provenance(),
                        simulated_nodes=graph.n,elapsed_s=trace[-1]['simTime'],fault=frame['fault'],
                        restore_equal=restored_equal,knee_interior_rad=knee,
                        cpg_unchanged=same(initial['body']['cpg'],engine.body.snapshot()['cpg']),
                        sensory_input_max_mV=max(p['value'] for r in trace for p in r['sensory_ports']) if len(trace)>1 else 0.,
                        motor_offset_max_rad=max(abs(v) for r in trace for v in r['neuromuscular']['offset_rad']),
                        behavior=evaluate(trace,world,required_seconds=seconds),biological_validation=False)
            write_json(directory/'trace.json',trace);write_json(directory/'signals.json',dict(ids=[graph.nodes[int(i)]['id'] for i in ids],rows=signals))
            write_json(directory/'result.json',result);reports.append(result)
            write_json(out/'report.json',dict(status='RUNNING',cases=reports))
            print(case,result['elapsed_s'],result['fault'],'restore',restored_equal,'knee',knee,'offset',result['motor_offset_max_rad'],flush=True)
        except Exception as exc:
            write_json(directory/'failure.json',dict(error=str(exc),physicalExecuted=engine.control_tick>0))
            raise
        finally:engine.close()
    lookup={r['case']:r for r in reports}
    checks=dict(no_physics_fault=all(not r['fault'] for r in reports),
                exact_continuation=all(r['restore_equal'] is True for r in reports),
                no_predefined_cpg=all(r['cpg_unchanged'] for r in reports),
                feedback_enters_neurons=lookup['feedback']['sensory_input_max_mV']>0,
                feedback_off_zero=lookup['feedback_off']['sensory_input_max_mV']==0,
                disconnected_zero_offsets=lookup['motor_disconnected']['motor_offset_max_rad']==0,
                tibia_direction=lookup['flexor']['knee_interior_rad']<lookup['feedback_off']['knee_interior_rad']<lookup['extensor']['knee_interior_rad'])
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,cases=reports,
                physicalExecuted=True,biological_validation=False,autonomous_walking='NOT_VALIDATED')
    write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/acquisitions/banc888-v2-20260909/bundle')
    p.add_argument('--bindings',default='data/banc888/bindings-neuromuscular-v1.json')
    p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=.5)
    a=p.parse_args();r=run(a.graph,a.bindings,a.out,a.seconds)
    print(r['status'],r['checks'])
    raise SystemExit(0 if r['status']=='PASS' else 1)
