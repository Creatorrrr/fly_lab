#!/usr/bin/env python3
"""Collect Windows/Linux Nsight CUDA/NVTX timeline and memory evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def find_nsys():
    found=shutil.which('nsys')
    if found:return found
    roots=[Path(os.environ.get('ProgramFiles','C:/Program Files'))/'NVIDIA Corporation']
    candidates=[p for root in roots for p in root.glob('Nsight Systems */target-windows-x64/nsys.exe')]
    if not candidates:raise RuntimeError('Nsight Systems CLI not found; provide --nsys')
    return str(sorted(candidates)[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','bindings','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--seconds',type=float,default=.05);p.add_argument('--warmup',type=float,default=.02)
    p.add_argument('--physics',choices=('cpu','warp','resident'),default='resident')
    p.add_argument('--nsys');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if any(not .005<=x<=60 or abs(x/.005-round(x/.005))>1e-8 for x in (a.seconds,a.warmup)):p.error('Use 5 ms multiples in .005..60 s')
    if not a.worker:
        a.out.mkdir(parents=True,exist_ok=False)
        nsys=a.nsys or find_nsys()
        command=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=true','--cuda-graph-trace=node',
            '--export=sqlite','--output='+str(a.out/'timeline'),sys.executable,'-X','utf8',str(Path(__file__).resolve()),
            '--worker','--graph',str(a.graph),'--bindings',str(a.bindings),'--out',str(a.out),
            '--seconds',str(a.seconds),'--warmup',str(a.warmup),'--physics',a.physics]
        env=dict(os.environ,PYTHONUTF8='1')
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        (a.out/'nsys.log').write_text(result.stdout,encoding='utf-8')
        from flylab.gpu_profile import summarize_timeline
        db=a.out/'timeline.sqlite';timeline=summarize_timeline(db) if db.is_file() else dict(timeline_tables={})
        output=dict(command=command,returncode=result.returncode,**timeline,
            cuda_trace_collected=timeline['timeline_tables'].get('CUPTI_ACTIVITY_KIND_KERNEL',0)>0,
            worker_report=(a.out/'worker.json').is_file())
        (a.out/'capture.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output))
        return 0 if result.returncode==0 and output['cuda_trace_collected'] else 2
    from flylab.c.engine import CEngine
    from flylab.c.graph import GraphStore
    from flylab.c.ports import PortBindings
    from flylab.gpu_profile import DeviceMemorySampler,nvtx_range
    import numpy as np
    graph=GraphStore.load(a.graph);bindings=PortBindings(graph,json.loads(a.bindings.read_text(encoding='utf-8')))
    profile=dict(backend='cpu' if a.physics=='cpu' else 'warp',control_backend='cuda' if a.physics=='resident' else 'cpu')
    with nvtx_range('setup'):e=CEngine(graph,bindings,mode='C_SHADOW',backend='exp_lif_cuda',physics_profile=profile)
    try:
        with nvtx_range('warmup'):e.step(round(a.warmup/.005))
        times=[]
        with DeviceMemorySampler() as memory,nvtx_range('steady_run'):
            for _ in range(round(a.seconds/.005)):
                start=time.perf_counter()
                with nvtx_range('control'):e.step(1)
                times.append(time.perf_counter()-start)
                if e.fault or e.body.fault:raise RuntimeError(e.fault or e.body.fault)
        output=dict(physics=e.body.physics_identity(),model_seconds=a.seconds,wall_s=sum(times),
            sim_per_wall=a.seconds/sum(times),control_median_s=float(np.median(times)),control_p95_s=float(np.percentile(times,95)),
            memory=memory.report(),scope='instrumented Nsight run; compare uninstrumented measurements separately')
        (a.out/'worker.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    finally:e.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
