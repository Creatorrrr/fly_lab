#!/usr/bin/env python3
"""BANC v888 neurons/edges from the paper's public, generation-pinned files.

Uses root_888, NEVER mutable root_id. Unclassified, unproofread objects and
explicit non-neurons are audited, not inferred from edge endpoints.
"""
import argparse
import math
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore, external_id, normalized_nt
from flylab.c.integrity import file_hash, read_json, write_json

SOURCE='https://www.nature.com/articles/s41586-026-10735-w'
NON_NEURONS={'glia','not_a_neuron','trachea'}

def build(raw, out):
    import pyarrow as pa
    import pyarrow.feather as feather
    raw=Path(raw);out=Path(out)
    provenance=read_json(raw/'download_manifest.json')
    for entry in provenance['files']:
        if file_hash(raw/entry['name'])!=entry['sha256']:raise ValueError('BANC raw checksum mismatch')
    table=feather.read_table(raw/'banc_888_meta.feather')
    fields=('root_888','proofread','roughly_proofread','super_class','cell_class','cell_sub_class','cell_type','fafb_cell_type',
            'side','region','nerve','flow','body_part_sensory','body_part_effector','peripheral_target_type',
            'cell_function','cell_function_detailed','neurotransmitter_predicted','neurotransmitter_verified','neurotransmitter_score')
    rows=table.select(fields).to_pylist();nodes=[];excluded=Counter()
    for row in rows:
        row={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in row.items()}
        cls=row['super_class']
        if cls in NON_NEURONS:excluded[cls]+=1;continue
        if not cls and row['proofread']!='TRUE' and row['roughly_proofread']!='TRUE':
            excluded['unclassified_unproofread_object']+=1;continue
        root=row['root_888']
        verified=row['neurotransmitter_verified']
        nt=normalized_nt(verified or row['neurotransmitter_predicted'])
        node=dict(id=external_id(root,'banc','888'),root_id=root,cell_type=row['cell_type'] or '',
                  super_class=cls or 'unknown_neuronal_class',**{'class':row['cell_class'] or ''},
                  flow=row['flow'] or '',soma_side=row['side'] if row['side'] in ('left','right','center') else 'unknown',
                  nt_type=nt,regions=[row['region']] if row['region'] else [],source_annotations=row)
        nodes.append(node)
    nodes.sort(key=lambda n:int(n['root_id']))
    roots=np.array([int(n['root_id']) for n in nodes],dtype=np.uint64)
    if len(np.unique(roots))!=len(roots):raise ValueError('Duplicate BANC v888 IDs')
    edges=feather.read_table(raw/'banc_888_edgelist_simple_v2.feather',columns=['pre','post','count'])
    pre=edges['pre'].cast(pa.uint64()).to_numpy();post=edges['post'].cast(pa.uint64()).to_numpy();counts=edges['count'].to_numpy()
    pi=np.searchsorted(roots,pre);qi=np.searchsorted(roots,post)
    valid=(pi<len(roots))&(qi<len(roots))
    valid &= (roots[np.minimum(pi,len(roots)-1)]==pre)&(roots[np.minimum(qi,len(roots)-1)]==post)
    metadata=dict(dataset_id='flywire_banc',specimen_id='BANC',sex='female',snapshot_id='888',
                  scope='declared_neuronal_roster',graph_hash_schema='content-v2',
                  annotation_release='generation-pinned in download_manifest.json',sources=[SOURCE]+[r['url'] for r in provenance['files']],
                  source_node_count=len(nodes),source_object_count=len(rows),excluded_node_count=0,excluded_node_ids=[],
                  excluded_objects=dict(excluded),excluded_connection_rows=int((~valid).sum()),
                  excluded_contacts=int(counts[~valid].sum()),
                  raw_file_hashes={r['name']:r['sha256'] for r in provenance['files']},
                  upstream_filters=['BANC v888 simple v2 edge list; no additional contact-count threshold'],
                  applied_filters=['master: annotated neuronal class OR proofread/roughly-proofread; explicit non-neurons excluded',
                                   'edges require both endpoints in declared master roster'],
                  scope_limitations=['BANC lacks lamina and ocellar ganglion','Current annotations projected onto v888 IDs',
                                     'Neuronal roster completeness requires independent review; not a full-snapshot PASS',
                                     'Unknown and modulatory chemical effects remain masked; no receptor inference'],
                  side_conventions=dict(annotation='BANC annotated biological side',body='UI x right, y up, z; positive yaw clockwise'))
    graph=GraphStore.from_edges(nodes,pi[valid].astype(np.int32),qi[valid].astype(np.int32),counts[valid].astype(np.int64),
                                metadata=metadata,unknown_policy='mask_zero')
    graph.save(out)
    audit=dict(status='LOADED_DECLARED_ROSTER',manifest=graph.manifest,
               super_classes=dict(Counter(n['super_class'] for n in nodes)),physicalExecuted=False,full_cns_validation=False)
    write_json(out.parent/'banc_acquisition_audit.json',audit)
    print({k:graph.manifest[k] for k in ('simulated_node_count','pair_edge_count','anatomical_synapse_count','masked_edge_count')},flush=True)
    return graph

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.raw,a.out)
