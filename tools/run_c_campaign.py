#!/usr/bin/env python3
"""Run fresh physical experiments in an independent process; preserve failures."""
import argparse
from pathlib import Path
import sys
import os
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.campaign import run_campaign, pilot_spec, validate_spec
from flylab.c.neural import BACKEND_CHOICES
from flylab.c.integrity import read_json, write_json
from flylab.c.locking import lease

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--bindings',default='data/fafb783/bindings-bilateral-geosmin-v2.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--spec',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto');p.add_argument('--cancel-file',type=Path)
    p.add_argument('--worker-lock-fd',type=int,help=argparse.SUPPRESS)
    p.add_argument('--worker-lock-path',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker-ready-file',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--write-pilot',type=Path);a=p.parse_args()
    if bool(a.worker_lock_path)!=bool(a.worker_ready_file):p.error('Worker lease and ready paths must be supplied together')
    with ExitStack() as stack:
        if a.worker_lock_path:
            stack.enter_context(lease(a.worker_lock_path))
            write_json(a.worker_ready_file,dict(pid=os.getpid()))
        return run(a)


def run(a):
    g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings))
    spec=read_json(a.spec) if a.spec else pilot_spec(bindings=b)
    spec=validate_spec(spec,b,a.backend)
    if a.write_pilot:write_json(a.write_pilot,spec);return 0
    if a.worker_lock_fd is not None:
        os.fstat(a.worker_lock_fd)  # Keep the inherited lease alive until exit.
    with lease(a.out.parent/('.'+a.out.name+'.lock')):
        report=run_campaign(g,b,spec,a.out,backend=a.backend,resume=a.resume,
                            cancelled=lambda:bool(a.cancel_file and a.cancel_file.exists()))
    print(report['status'],len(report['cases']),report['completed_model_s'],flush=True)
    return 0 if report['status']=='COMPLETE' else 1 if report['status']=='COMPLETE_WITH_FAILURES' else 2
if __name__=='__main__':raise SystemExit(main())
