#!/usr/bin/env python3
"""Resolve halting cohorts from the archived Sapkal et al. 2024 Table 3.

Only exact FAFB root IDs are accepted. FANC/MANC IDs are not transplanted into
BANC. A BB forward-only output gate is an engineering adapter, distinct from
the real synaptic paths that remain simulated for BB, FG and BRK.
"""
import argparse
import copy
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.integrity import read_json,write_json,file_hash

SOURCE='https://www.nature.com/articles/s41586-024-07854-7'
TABLE_URL='https://pmc-oa-opendata.s3.amazonaws.com/PMC11446846.1/41586_2024_7854_MOESM4_ESM.xlsx'
TABLE_SHA256='41c0f799193faf13477a7b537fa3d37d72d88c8cc048ad56f229536169c04d04'
EXPECTED={'FG':('CB0890','central','GABA'),'BB':('DNg60','descending','GABA'),
          'BRK1, BRK2':('AN_GNG_53','ascending','ACH'),
          'BRK3, BRK4':('AN_GNG_54','ascending','ACH'),
          'BRK5, BRK6':('AN_GNG_76','ascending','ACH')}
NAMES={'FG':'Foxglove / CB0890 (brain Walk-OFF)', 'BB':'Bluebell / DNg60 (Walk-OFF)',
       'BRK1, BRK2':'BRK1-2 (ascending brain segment)',
       'BRK3, BRK4':'BRK3-4 (ascending brain segment)',
       'BRK5, BRK6':'BRK5-6 (ascending brain segment)'}


def identities(table_path):
    if file_hash(table_path)!=TABLE_SHA256:raise ValueError('Halting source table checksum mismatch')
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(table_path) as z:
        strings=[''.join(e.itertext()).strip() for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        rows=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
            cells={}
            for c in row.findall('m:c',ns):
                v=c.find('m:v',ns)
                if v is None:continue
                # IDs are strings in the source workbook. Never round through a
                # float; 18-digit roots exceed IEEE-754 exact integer precision.
                value=strings[int(v.text)] if c.get('t')=='s' else v.text
                cells[''.join(k for k in c.get('r') if k.isalpha())]=value.strip()
            if cells.get('E') in EXPECTED:
                rows.append(dict(label=cells['E'],roots=[cells['A'],cells['B']],source_row=int(row.get('r'))))
    if {r['label'] for r in rows}!=set(EXPECTED) or len(rows)!=5:
        raise ValueError('Halting table layout differs')
    return rows


def build(graph,base,table_path):
    if (graph.manifest['dataset_id'],graph.manifest['snapshot_id'])!=('flywire_fafb','783'):
        raise ValueError('Exact FAFB v783 identity review required')
    original=PortBindings(graph,base);table={n['root_id']:n for n in graph.nodes}
    cohorts=[];resolved=[]
    for row in identities(table_path):
        cells=[]
        for side,root in zip(('left','right'),row['roots']):
            n=table.get(root)
            expected=EXPECTED[row['label']]
            if not n or tuple(n.get(k) for k in ('cell_type','super_class','nt_type'))!=expected or n['soma_side']!=side:
                raise ValueError('Current graph differs from reviewed halting identity: '+root)
            cells.append(n)
        uncertainty=('Exact paper root IDs and current type/side/NT annotations match. '
                     'Physiology and model firing-rate gains are not calibrated. '
                     'FAFB contains the brain segment of ascending BRK cells, not their local VNC outputs.')
        cohorts.append(dict(name=NAMES[row['label']],ids=[n['id'] for n in cells],
            review_status='engineering_reviewed',evidence=[SOURCE,TABLE_URL],uncertainty=uncertainty))
        resolved.append(dict(row,cells=[{k:n[k] for k in ('id','root_id','cell_type','super_class','soma_side','nt_type')} for n in cells]))
    spec=copy.deepcopy(base)
    bb=next(c for c in cohorts if c['name']==NAMES['BB'])
    spec['motor']['stop']=dict(copy.deepcopy(bb),gain=.02,
        uncertainty='Exact Bluebell/DNg60 IDs from Table 3. Forward-only linear gate reaches zero at 50 Hz mean BB activity; this is an uncalibrated engineering output adapter, not a reconstructed VNC stop circuit.')
    previous=spec.get('motor_decoder',{})
    spec['motor_decoder']=dict(kind='bounded-walk-off-v2',speed_limit=previous.get('speed_limit',1.),
                              yaw_limit=previous.get('yaw_limit',1.5),stop_full_drive=1.)
    spec.update(profile='fafb783-walking-visual-contact-walk-off-v6',profile_version=6,
        parent_binding_hash=original.hash,biological_validation=False,research_cohorts=cohorts,
        halting_source=dict(url=TABLE_URL,sha256=TABLE_SHA256,paper=SOURCE,table='Supplementary Table 3, FlyWire',
                            exact_root_matches=10,scope='FAFB brain; no FANC/MANC ID substitution'))
    PortBindings(graph,spec)
    return spec,dict(source=spec['halting_source'],graph_hash=graph.hash,resolved=resolved,
                     physicalExecuted=False,biological_validation=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--base',default='data/fafb783/bindings-walking-visual-contact-v5.json')
    p.add_argument('--table',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--review-out',required=True)
    a=p.parse_args()
    if a.out.exists() or Path(a.review_out).exists():raise SystemExit('Choose new output files')
    spec,review=build(GraphStore.load(a.graph),read_json(a.base),a.table)
    write_json(a.out,spec);write_json(a.review_out,review)
