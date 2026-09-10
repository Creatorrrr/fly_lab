#!/usr/bin/env python3
"""Build bilateral population inputs from archived anatomical annotations.

No retinotopy is invented. Optional individual ommatidium ports require a
separately supplied, reviewed mapping file. Palp geometry and tuning retain
explicit engineering uncertainty unless external calibration is provided.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings


def build(graph,base,*,hz=100,field='gaussian',retinotopy=None,site_calibration=None):
    PortBindings(graph,base)
    result=copy.deepcopy(base)
    result.update(profile='flygym-multimodal-v2-'+field,profile_version=2,biological_validation=False)
    result['sensor_model']=dict(kind='flygym-multimodal-v2',field=field,half_concentration=1.,
        retina=dict(sample_hz=hz,tau_s=.02,dark_threshold=.25),adaptation_tau_s=.5,
        adaptation_strength=.5,extended_observations=True)
    if site_calibration is not None:result['sensor_model']['site_calibration']=site_calibration
    ports=[]
    def add(name,channel,predicate,gain,uncertainty):
        ids=[n['id'] for n in graph.nodes if predicate(n)]
        if not ids:raise ValueError('No anatomically annotated population: '+name)
        ports.append(dict(name=name,channel=channel,ids=ids,input_kind='sensory',method='drive_mV',
            gain=gain,baseline=0.,cap=24.,tau_s=.02,delay_controls=0,offset=0.,scale=1.,
            review_status='engineering_reviewed',evidence=['https://codex.flywire.ai/faq',
                'archived graph '+graph.hash+'; cell_type, nerve, class, soma_side reviewed'],
            uncertainty=uncertainty))
    for side in ('left','right'):
        for site,nerve,cell in [('antenna','AN','ORN_DM1'),('palp','MxLbN',None)]:
            add(f'{site}_food_{side}',f'odor_{site}_{side}_food',
                lambda n,s=side,v=nerve,c=cell:n.get('soma_side')==s and n.get('nerve')==v and n.get('class')=='olfactory' and (c is None or n.get('cell_type')==c),18.,
                'Archived peripheral side and nerve annotation; food chemistry, gain, adaptation and palp offsets are engineering assumptions. Palp ORNs receive a broad food proxy, not calibrated receptor-specific chemistry.')
        add('antenna_hazard_'+side,'odor_antenna_'+side+'_hazard',
            lambda n,s=side:n.get('soma_side')==s and n.get('nerve')=='AN' and n.get('cell_type')=='ORN_DA2',18.,
            'DA2/geosmin candidate. Generic hazard treated as geosmin only in this profile; concentration, gain and behavioral response uncalibrated.')
        add('retina_luminance_'+side,'retina_luminance_'+side,
            lambda n,s=side:n.get('soma_side')==s and n.get('cell_type')=='R1-6' and n.get('sub_class')=='photo_receptor',12.,
            'Eye-wide intensity population proxy. No column/ommatidium correspondence or physiological phototransduction is asserted. Pale/yellow channel intensity is pooled.')
    if retinotopy is not None:
        if retinotopy.get('graph_hash')!=graph.hash or not isinstance(retinotopy.get('ports'),list):
            raise ValueError('Retinotopy graph identity and explicit ports required')
        # Explicit retinotopy replaces pooled visual inputs to avoid double drive.
        ports=[p for p in ports if not p['channel'].startswith('retina_')]
        if not retinotopy['ports']:raise ValueError('Empty retinotopy mapping')
        for p in retinotopy['ports']:
            if p.get('channel')!='retina_ommatidium':raise ValueError('Expected ommatidium port')
            ports.append(copy.deepcopy(p))
    result['sensory']=ports
    result['hazard_semantics']='bilateral geosmin engineering proxy'
    result['unused_observations']=['Retina motion/loom features are available for reviewed experimental ports',
        'Palp hazard chemistry unassigned; no invented aversive receptor mapping']
    result['mapping_status']=dict(retinotopy='external mapping' if retinotopy else 'population proxy only',
        palp_geometry='external calibration' if site_calibration else 'engineering offsets; anatomical calibration pending')
    PortBindings(graph,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','base','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--hz',choices=(100,200),type=int,default=100)
    p.add_argument('--field',choices=('gaussian','inverse-square','plume'),default='gaussian')
    p.add_argument('--retinotopy',type=Path);p.add_argument('--site-calibration',type=Path)
    a=p.parse_args();read=lambda path:json.loads(path.read_text(encoding='utf-8')) if path else None
    result=build(GraphStore.load(a.graph),read(a.base),hz=a.hz,field=a.field,
        retinotopy=read(a.retinotopy),site_calibration=read(a.site_calibration))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(a.out)
