#!/usr/bin/env python3
"""Versioned bilateral food / geosmin-proxy engineering profile, never a gain fit."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json, write_json

SOURCES = ['https://codex.flywire.ai/faq', 'https://www.nature.com/articles/nature11747',
           'https://www.nature.com/articles/srep21841', 'https://doi.org/10.1016/j.cell.2012.09.046']


def build(graph, base):
    spec=copy.deepcopy(base);sensory=[]
    for side in ('left','right'):
        ids=[n['id'] for n in graph.nodes if n.get('cell_type')=='ORN_DM1' and n.get('soma_side')==side]
        if not ids: raise ValueError('Missing reviewed DM1 side: '+side)
        sensory.append(dict(copy.deepcopy(base['sensory'][0]), name='odor_DM1_'+side,
                            channel='odor_'+side, ids=ids, input_side=side,
                            boundary='AN afferent ORN_DM1; peripheral side from biological soma annotation',
                            evidence=SOURCES[:3], uncertainty='Peripheral side is assigned from the archived biological soma_side annotation. '
                            'Receptor tuning, bilateral release asymmetry, gain/cap and filter are not experimentally calibrated. '
                            'The virtual food field is an engineering proxy for DM1/Or42b input.'))
    ids=[n['id'] for n in graph.nodes if n.get('cell_type')=='ORN_DA2']
    if not ids: raise ValueError('Missing DA2 afferents for the geosmin-proxy input')
    sensory.append(dict(copy.deepcopy(base['sensory'][0]), name='geosmin_DA2', channel='danger',ids=ids,
                        input_side='bilateral_scalar',boundary='AN afferent ORN_DA2 / Or56a geosmin candidate',
                        evidence=[SOURCES[0],SOURCES[3]],uncertainty='The generic hazard field is treated only in this profile as a geosmin proxy. '
                        'Its concentration, 18 mV/unit gain, 24 mV cap and 20 ms filter are engineering assumptions inherited without fitting. '
                        'A head-center scalar is delivered bilaterally; lateral hazard information and avoidance success are not validated.'))
    for p in sensory:
        for index in graph.resolve(p['ids']):
            n=graph.nodes[int(index)]
            if n['super_class']!='sensory' or n['flow']!='afferent' or n.get('nerve')!='AN':
                raise ValueError('Port target no longer matches the reviewed antennal afferent boundary')
    spec.update(profile='fafb783-bilateral-food-geosmin-engineering-v1', profile_version=1,
                parent_binding_hash=PortBindings(graph,base).hash, sensory=sensory,
                review_date='2026-09-09', biological_validation=False,
                hazard_semantics='geosmin proxy; not generic danger',
                unused_observations=[v for v in base['unused_observations'] if v!='danger'])
    PortBindings(graph,spec)
    return spec


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--base',default='data/fafb783/bindings.json');p.add_argument('--out',type=Path,default=Path('data/fafb783/bindings-bilateral-geosmin-v1.json'))
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Use a new versioned filename; existing bindings are immutable')
    g=GraphStore.load(a.graph);s=build(g,read_json(a.base));write_json(a.out,s)
    print({p['name']:len(p['ids']) for p in s['sensory']});print('binding_hash:',PortBindings(g,s).hash)
