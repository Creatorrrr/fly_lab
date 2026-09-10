#!/usr/bin/env python3
"""Create an explicit four-site odor profile without inventing neural mappings."""
from pathlib import Path
import argparse
import copy
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings


def build(graph,base,field='gaussian'):
    from flylab.flygym_senses import FourSiteOdor
    PortBindings(graph,base)
    result=copy.deepcopy(base)
    result['sensor_model']=dict(FourSiteOdor(field=field).model,
        extended_observations=base.get('sensor_model',{}).get('extended_observations',False))
    result['profile']=base['profile']+'-four-site-'+field
    result['profile_version']=int(base.get('profile_version',1))+1
    result['biological_validation']=False
    result['sensor_model']['mapping_policy']='Existing antenna odor ports only; palp readouts and retina are observation-only'
    result['unused_observations']=list(dict.fromkeys(base.get('unused_observations',[])+
        ['palp readouts: anatomical mapping review required','compound-eye image: retinotopy mapping required']))
    PortBindings(graph,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',required=True,type=Path)
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--field',choices=('gaussian','inverse-square'),default='gaussian')
    a=p.parse_args()
    result=build(GraphStore.load(a.graph),json.loads(a.base.read_text(encoding='utf-8')),a.field)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(a.out)
