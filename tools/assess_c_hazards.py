#!/usr/bin/env python3
"""Reassess committed hazard, hazard-free and sensory-off campaign traces."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.sensorimotor_campaign import assess_hazards
from flylab.c.integrity import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assay',choices=('natural','evoked'),default='natural')
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new assessment path')
    result=assess_hazards(a.campaign,assay=a.assay);a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,result)
    print(result['task_status'],len(result['cases']),'matched groups')
    raise SystemExit(0 if result['task_status']=='PASS' else 1)
