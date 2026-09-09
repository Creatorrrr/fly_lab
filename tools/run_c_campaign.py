#!/usr/bin/env python3
"""Run fresh physical experiments in an independent process; preserve failures."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.campaign import run_campaign, pilot_spec
from flylab.c.neural import NEURAL_BACKENDS
from flylab.c.integrity import read_json, write_json
from flylab.c.locking import lease

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--bindings',default='data/fafb783/bindings-bilateral-geosmin-v2.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--spec',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--backend',choices=NEURAL_BACKENDS,default='exp_lif_mps');p.add_argument('--cancel-file',type=Path)
    p.add_argument('--worker-lock-fd',type=int,help=argparse.SUPPRESS)
    p.add_argument('--write-pilot',type=Path);a=p.parse_args()
    if a.write_pilot:write_json(a.write_pilot,pilot_spec());return 0
    spec=read_json(a.spec) if a.spec else pilot_spec()
    if a.worker_lock_fd is not None:
        import os
        os.fstat(a.worker_lock_fd)  # Keep the inherited lease alive until exit.
    with lease(a.out.parent/('.'+a.out.name+'.lock')):
        g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings))
        report=run_campaign(g,b,spec,a.out,backend=a.backend,resume=a.resume,
                            cancelled=lambda:bool(a.cancel_file and a.cancel_file.exists()))
    print(report['status'],len(report['cases']),report['completed_model_s'],flush=True)
    return 0 if report['status']=='COMPLETE' else 1 if report['status']=='COMPLETE_WITH_FAILURES' else 2
if __name__=='__main__':raise SystemExit(main())
