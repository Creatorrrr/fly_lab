#!/usr/bin/env python3
"""Write fixed pose/friction cases; preparation does not execute or approve them."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.sensorimotor_campaign import fixed_spec
from flylab.c.campaign import validate_spec, SCENES
from flylab.c.integrity import write_json, read_json
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--bindings',default='data/fafb783/bindings-bilateral-geosmin-v2.json')
    p.add_argument('--scenes',nargs='+',choices=SCENES)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new fixed spec path')
    bindings=PortBindings(GraphStore.load(a.graph),read_json(a.bindings))
    spec=validate_spec(fixed_spec(bindings=bindings,scenes=a.scenes),bindings)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out,spec)
    print(len(spec['cases']),'prepared cases; physicalExecuted=False')
