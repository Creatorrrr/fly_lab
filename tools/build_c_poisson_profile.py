#!/usr/bin/env python3
"""Explicit ORN current-pulse hypothesis, not a calibrated odor model."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json,write_json

def build(graph,base):
    spec=copy.deepcopy(base)
    spec.update(profile='fafb783-odor-poisson-current-hypothesis-v1',profile_version=1,
                parent_binding_hash=PortBindings(graph,base).hash,biological_validation=False)
    for p in spec['sensory']:
        p.update(method='poisson_Hz',gain=150.,cap=150.,baseline=0.,pulse_mV=68.75)
        p['evidence']=list(p.get('evidence',[]))+['https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py']
        p['uncertainty']+=' Input hypothesis only: 150 Hz/unit and 68.75 mV pulses enter this model synaptic drive h. Shiu et al inject voltage v and remove target refractoriness; this profile does neither and is not their model reproduction. No background or direct motor input.'
    PortBindings(graph,spec);return spec
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--base',default='data/fafb783/bindings-bilateral-geosmin-v2.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit('Choose a new immutable profile path')
    write_json(a.out,build(GraphStore.load(a.graph),read_json(a.base)))
