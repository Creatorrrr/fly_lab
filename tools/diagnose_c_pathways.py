#!/usr/bin/env python3
"""Independent fixed-input pathway diagnostics; never a behavior PASS."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json, write_json
from flylab.c.pathways import reachability, propagate, diagnostic_cases
from flylab.c.neural import BACKEND_CHOICES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--bindings',default='data/fafb783/bindings-bilateral-geosmin-v2.json')
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto');p.add_argument('--seconds',type=float,default=.2)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings))
    write_json(a.out/'reachability.json',reachability(g,b));summaries=[]
    for case in diagnostic_cases():
        report=propagate(g,b,case,backend=a.backend,seconds=a.seconds)
        write_json(a.out/(case['name']+'.json'),report)
        summaries.append({k:v for k,v in report.items() if k not in ('trace','capture_ids')})
        write_json(a.out/'report.json',dict(cases=summaries,status='RUNNING',physicalExecuted=False))
        print(case['name'],report['active_nodes'],report['trace'][-1]['command'],flush=True)
    write_json(a.out/'report.json',dict(cases=summaries,status='COMPLETE',physicalExecuted=False,behavior='NOT_EVALUATED'))
if __name__=='__main__':main()
