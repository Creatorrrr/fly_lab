#!/usr/bin/env python3
"""Run full-connectome CUDA worlds, recording, selected rendering and recovery."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.batch import BatchSession
from flylab.gpu_profile import DeviceMemorySampler,nvtx_range


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','bindings','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--worlds',type=int,default=2);p.add_argument('--seconds',type=float,default=.05)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--mode',choices=('C_STRICT','C_SHADOW','C_ASSISTED'),default='C_SHADOW')
    p.add_argument('--restore',type=Path);p.add_argument('--events',type=Path)
    p.add_argument('--record-channels',type=int,default=512);p.add_argument('--render-world',type=int,default=0)
    a=p.parse_args()
    if not 1<=a.worlds<=32 or not .005<=a.seconds<=60 or abs(a.seconds/.005-round(a.seconds/.005))>1e-8:p.error('Use 1..32 worlds and integral 5 ms periods')
    if not 0<=a.record_channels<=512:p.error('Use 0..512 recording channels')
    a.out.mkdir(parents=True,exist_ok=False)
    graph=GraphStore.load(a.graph);bindings=PortBindings(graph,json.loads(a.bindings.read_text(encoding='utf-8')))
    events=json.loads(a.events.read_text(encoding='utf-8')) if a.events else []
    if any(set(event)!={'at_control','action','world'} or event['action'] not in ('pause','resume','cancel') for event in events):p.error('Invalid batch events')
    session=None
    with DeviceMemorySampler() as memory:
        try:
            with nvtx_range('batch_setup'):
                session=BatchSession.restore(graph,bindings,a.restore) if a.restore else BatchSession(graph,bindings,[a.seed+i for i in range(a.worlds)],mode=a.mode)
            if a.record_channels:
                ids=[n['id'] for n in graph.nodes[:min(a.record_channels,graph.n)]]
                for key,e in session.engines.items():
                    if session.status[key]!='CANCELLED':e.start_recording(a.out/('world-'+key),ids)
            latency=[];began=time.perf_counter()
            with nvtx_range('batch_run'):
                for tick in range(round(a.seconds/.005)):
                    for event in events:
                        if event['at_control']==tick:getattr(session,event['action'])(str(event['world']))
                    start=time.perf_counter();session.step(1);latency.append(time.perf_counter()-start)
            elapsed=time.perf_counter()-began
            checkpoint=session.checkpoint(a.out/'checkpoint')
            rendered=None
            if a.render_world>=0:
                from PIL import Image
                rendered=str(a.render_world)
                if rendered not in session.engines:raise ValueError('Unknown render world')
                with nvtx_range('batch_render'):
                    pixels=session.physics.render(session.active.index(rendered)) if rendered in session.active else session.engines[rendered].body.preview()
                    Image.fromarray(pixels).save(a.out/'selected-world.png')
            import numpy as np
            output=dict(summary=session.summary(),wall_s=elapsed,period_latency_median_s=float(np.median(latency)),
                period_latency_p95_s=float(np.percentile(latency,95)),checkpoint=checkpoint,rendered_world=rendered,
                recording_channels=a.record_channels,performance_scope='includes first graph capture; use profiler for warmed throughput')
        finally:
            if session:session.close()
    output['memory']=memory.report()
    (a.out/'report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
