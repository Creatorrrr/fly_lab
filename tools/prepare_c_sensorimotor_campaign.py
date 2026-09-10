#!/usr/bin/env python3
"""Write fixed pose/friction cases; preparation does not execute or approve them."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.sensorimotor_campaign import fixed_spec
from flylab.c.campaign import validate_spec
from flylab.c.integrity import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new fixed spec path')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    spec=validate_spec(fixed_spec());write_json(a.out,spec)
    print(len(spec['cases']),'prepared cases; physicalExecuted=False')
