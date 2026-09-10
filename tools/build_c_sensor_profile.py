#!/usr/bin/env python3
"""Explicit coarse feature injections into reviewed real cell-type cohorts.

This profile bypasses retinal and bristle transduction. It is an experimental
engineering boundary, never a reconstruction of all sensory organs.
"""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json, write_json

def build(graph,base):
    spec=copy.deepcopy(base)
    # A Poisson odor template has Hz gains and a pulse amplitude. Visual and
    # contact proxies are held mV drives, independently of the parent's units.
    template=dict(method='drive_mV',baseline=0.,offset=0.,scale=1.,tau_s=.02)
    source='https://www.sciencedirect.com/science/article/pii/S0960982219301381'
    for side in ('left','right'):
        ids=[n['id'] for n in graph.nodes if n.get('cell_type')=='LC4' and n.get('soma_side')==side and n.get('super_class')=='visual_projection']
        if not ids:raise ValueError('LC4 hemisphere annotation unavailable')
        spec['sensory'].append(dict(copy.deepcopy(template),name='loom_LC4_'+side,channel='loom_'+side,ids=ids,
            input_kind='direct_injection',input_side=side,gain=8.,cap=24.,tau_s=.02,delay_controls=1,
            boundary='Horizontal obstacle silhouette expansion injected into hemisphere LC4 cohort',
            review_status='engineering_reviewed',evidence=[source,'https://codex.flywire.ai/faq'],
            uncertainty='Hemisphere-wide proxy; bypasses retinal processing and LC4 receptive-field tiling. '
                        '64 horizontal rays encode obstacles, not textured walls; injected gain is uncalibrated. '
                        'Escape-related LC4 activity does not establish walking avoidance.'))
    ids=[n['id'] for n in graph.nodes if n.get('cell_type') in ('BM_Fr','BM_FrOr') and n.get('class')=='mechanosensory']
    if not ids:raise ValueError('Frontal mechanosensory annotations unavailable')
    spec['sensory'].append(dict(copy.deepcopy(template),name='head_contact_frontal_bristle_proxy',channel='head_contact',ids=ids,
        input_kind='direct_injection',gain=18.,cap=24.,delay_controls=1,input_side='bilateral_scalar',
        boundary='Core head/world contact injected into annotated frontal bristle afferents',
        evidence=['https://codex.flywire.ai/faq','https://www.nature.com/articles/s41586-024-07686-5'],
        uncertainty='Contact area, force and bristle mechanics are not reconstructed. This is an explicit coarse head-contact injection; '
                    'leg contact and leg proprioception are not mapped to these head neurons.',review_status='engineering_reviewed'))
    spec['sensor_model']=dict(spec.get('sensor_model',{}),extended_observations=True)
    spec.update(profile='fafb783-coarse-visual-head-contact-research-v2',profile_version=2,
                biological_validation=False,parent_binding_hash=PortBindings(graph,base).hash)
    PortBindings(graph,spec);return spec

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--base',default='data/fafb783/bindings-bilateral-geosmin-v2.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit('Profile exists; choose a new version')
    write_json(a.out,build(GraphStore.load(a.graph),read_json(a.base)))
