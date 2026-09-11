"""Build auditable sensory candidates from archived DoOR and FAFB data."""
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def verified_assets(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    evidence=[]
    for row in manifest['sources']:
        file=(directory/row['path']).resolve()
        if not file.is_relative_to(directory.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Sensory source hash/path mismatch')
        evidence.append(row['url']+' SHA256 '+row['sha256'])
    return evidence


def r_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f,delimiter=';');header=next(reader)
        return [(row[0],dict(zip(header,row[1:],strict=True))) for row in reader]


def chemical_profile(graph,base,directory,odor_names,gain=18.):
    from .c.ports import PortBindings
    directory=Path(directory);evidence=verified_assets(directory)
    odors=[row for _,row in r_csv(directory/'door/odor.csv')]
    selected=[]
    for name in odor_names:
        matches=[r for r in odors if name in (r['Name'],r['InChIKey'])]
        if len(matches)!=1:raise ValueError('Odor name must resolve uniquely in DoOR: '+name)
        selected.append(dict(name=matches[0]['Name'],inchikey=matches[0]['InChIKey']))
    table=dict(r_csv(directory/'door/door_response_matrix.csv'))
    mappings=[row for _,row in r_csv(directory/'door/door_mappings.csv')]
    rows=[];ports=[];excluded=[];seen=set()
    for item in mappings:
        receptor=item['receptor'];glom=item['glomerulus'];sensillum=item['sensillum']
        if receptor in seen or receptor not in table['SFR'] or item['adult']!='TRUE':continue
        baseline=table['SFR'][receptor]
        responses={o['inchikey']:float(table[o['inchikey']][receptor]) for o in selected
            if o['inchikey'] in table and table[o['inchikey']][receptor]!='NA'}
        if baseline=='NA' or not responses:continue
        site='palp' if sensillum.startswith('pb') else 'antenna' if sensillum.startswith(('ab','ac','at','ai')) else None
        if site is None:continue
        targets={side:[n['id'] for n in graph.nodes if n.get('cell_type')=='ORN_'+glom and n.get('soma_side')==side
            and n.get('nerve')==('MxLbN' if site=='palp' else 'AN')] for side in ('left','right')}
        if not all(targets.values()):excluded.append(dict(receptor=receptor,glomerulus=glom,reason='Bilateral annotated graph targets missing'));continue
        seen.add(receptor)
        rows.append(dict(receptor=receptor,glomerulus=glom,site=site,baseline=float(baseline),responses=responses))
        for side,ids in targets.items():
            ports.append(dict(name=f'door_{receptor}_{side}',channel='chemical_odor',receptor=receptor,site=site+'_'+side,
                ids=ids,input_kind='sensory',method='drive_mV',gain=gain,baseline=gain*float(baseline),cap=24.,
                tau_s=.02,delay_controls=0,offset=0.,scale=1.,review_status='engineering_reviewed',evidence=evidence,
                uncertainty='DoOR normalized measured responses and adult glomerulus mapping. Field occupancy, neural gain, mixture interactions and palp positions require calibration.'))
    # Some DoOR entries map to the same annotated ORN (coexpressed receptors).
    # Independent measured response rows cannot be summed as independent cells:
    # that would count spontaneous firing and shared transduction twice.
    target_receptors={}
    for port in ports:
        for node in port['ids']:target_receptors.setdefault(node,set()).add(port['receptor'])
    ambiguous={receptor for receptors in target_receptors.values() if len(receptors)>1 for receptor in receptors}
    for receptor in sorted(ambiguous):
        excluded.append(dict(receptor=receptor,reason='Shared annotated ORN targets; joint receptor response calibration unavailable'))
    rows=[row for row in rows if row['receptor'] not in ambiguous]
    ports=[port for port in ports if port['receptor'] not in ambiguous]
    if not rows:raise ValueError('No unambiguous measured DoOR receptors match this graph')
    result=copy.deepcopy(base)
    result['sensor_model']['chemical_odor']=dict(schema='flylab.chemical-odor.v1',odorants=selected,receptors=rows,evidence=evidence,
        missing_data_policy='unmeasured pairs are reported; never imputed as measured zero',biological_validation=False)
    result['sensory']=[p for p in result['sensory'] if not (p['channel'].startswith('odor_') or p['channel']=='chemical_odor')]+ports
    result.update(profile='flygym-door-named-odorants-v2',profile_version=4,
        hazard_semantics='source kind is scene metadata; receptor response requires explicit odorant InChIKey',
        chemical_mapping_exclusions=excluded,biological_validation=False)
    result['mapping_status']['odor_response']='DoOR normalized response measurements; absolute concentration/gain uncalibrated'
    result['unused_observations']=[v for v in result.get('unused_observations',[]) if 'Palp hazard' not in v]
    PortBindings(graph,result)
    return result


def column_retinotopy(graph,base,directory,*,registration=None):
    """Use anatomical column assignments; cross-specimen optics remain explicit."""
    from flygym.vision.retina import Retina
    from .c.ports import PortBindings
    directory=Path(directory);evidence=verified_assets(directory)
    root_to_index={n['root_id']:i for i,n in enumerate(graph.nodes)}
    columns={};coordinates={'left':set(),'right':set()}
    with gzip.open(directory/'optic/column_assignment.csv.gz','rt',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            value=(row['hemisphere'],int(row['p']),int(row['q']))
            if value[0] not in coordinates:raise ValueError('Unknown annotated eye')
            coordinates[value[0]].add(value[1:])
            if row['root_id'] in root_to_index:columns[root_to_index[row['root_id']]]=value
    photo=np.array([n.get('cell_type')=='R1-6' for n in graph.nodes])
    post=np.repeat(np.arange(graph.n,dtype=np.int32),np.diff(graph.indptr))
    known=np.zeros(graph.n,bool);known[list(columns)]=True
    edges=np.flatnonzero(photo[graph.indices]&known[post]);scores={}
    for edge in edges:
        pre=int(graph.indices[edge]);column=columns[int(post[edge])]
        scores.setdefault(pre,{})[column]=scores.get(pre,{}).get(column,0)+int(graph.counts[edge])
    selected={};excluded=[]
    for index in np.flatnonzero(photo):
        votes=scores.get(int(index),{});maximum=max(votes.values(),default=0)
        best=[key for key,value in votes.items() if value==maximum]
        if len(best)!=1:excluded.append(graph.nodes[index]['id']);continue
        selected[int(index)]=best[0]
    retina=Retina();mask=retina.ommatidia_id_map
    yy,xx=np.indices(mask.shape);positive=mask>0
    counts=np.bincount(mask[positive],minlength=722)[1:]
    centers=np.stack([np.bincount(mask[positive],weights=xx[positive],minlength=722)[1:]/counts,
                      np.bincount(mask[positive],weights=yy[positive],minlength=722)[1:]/counts],axis=1)
    transforms={}
    if registration is not None:
        if registration.get('graph_hash')!=graph.hash or not registration.get('evidence') or not registration.get('uncertainty'):
            raise ValueError('Retinal registration identity/evidence required')
        for side in coordinates:
            value=np.asarray(registration.get('eye_to_pixel',{}).get(side))
            if value.shape!=(2,3) or not np.isfinite(value).all():raise ValueError('A 2x3 p/q-to-pixel transform is required per eye')
            transforms[side]=value
    else:
        for side,points in coordinates.items():
            pq=np.asarray(sorted(points),float)
            basis=np.array([[-np.sqrt(3)/2,np.sqrt(3)/2],[-.5,-.5]])
            xy=pq@basis.T;lo=xy.min(axis=0);hi=xy.max(axis=0)
            span=centers.max(axis=0)-centers.min(axis=0)
            scale=span/(hi-lo)
            transforms[side]=np.column_stack([basis*scale[:,None],centers.min(axis=0)-lo*scale])
    groups={};assignments=[]
    for index,(side,p,q) in selected.items():
        uv=transforms[side]@np.array([p,q,1.])
        ommatidium=int(np.argmin(np.sum((centers-uv)**2,axis=1)))
        groups.setdefault((side,ommatidium),[]).append(graph.nodes[index]['id'])
        assignments.append(dict(id=graph.nodes[index]['id'],eye=side,p=p,q=q,ommatidium=ommatidium))
    if not groups:raise ValueError('No photoreceptor columns could be resolved')
    result=copy.deepcopy(base)
    result['sensory']=[p for p in result['sensory'] if not p['channel'].startswith('retina_')]
    for (eye,index),ids in sorted(groups.items()):
        result['sensory'].append(dict(name=f'retina_{eye}_{index}',channel='retina_ommatidium',eye=eye,ommatidium=index,
            ids=ids,input_kind='sensory',method='drive_mV',gain=12.,baseline=0.,cap=24.,tau_s=.02,delay_controls=0,offset=0.,scale=1.,
            review_status='engineering_reviewed',evidence=evidence,retinotopy_evidence=evidence,
            uncertainty='R1-6 column inferred from maximal retained anatomical postsynaptic counts. 796-column FAFB eye and 721-element model require optical registration; no one-to-one anatomical equivalence asserted.'))
    result['retinotopy_registration']=dict(method='external affine registration' if registration else 'explicit normalized geometric candidate; not calibrated optics',
        eye_to_pixel={k:v.tolist() for k,v in transforms.items()},external=registration,
        assignments=assignments,excluded_ambiguous_or_unmapped_ids=excluded)
    result['mapping_status']['retinotopy']='column-resolved experimental projection; physiological registration unvalidated'
    result['profile']+='-columns'
    PortBindings(graph,result)
    return result
