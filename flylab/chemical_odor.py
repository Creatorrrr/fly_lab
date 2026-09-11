"""Named odorants and measured receptor response tables, with explicit field units."""
import copy
import numpy as np


class ChemicalOdor:
    def __init__(self,spec,field):
        if not isinstance(spec,dict) or spec.get('schema')!='flylab.chemical-odor.v1' or not spec.get('evidence'):
            raise ValueError('Chemical odor response evidence required')
        rows=spec.get('receptors');odors=spec.get('odorants')
        if not isinstance(rows,list) or not 1<=len(rows)<=128 or not isinstance(odors,list) or not 1<=len(odors)<=64:
            raise ValueError('Invalid odorant/receptor inventory')
        if any(not isinstance(o,dict) for o in odors) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Odorant and receptor objects required')
        names=[o.get('inchikey') for o in odors]
        if any(not isinstance(k,str) or not k for k in names) or len(set(names))!=len(names):raise ValueError('Invalid odorant identity')
        receptors=set()
        for row in rows:
            if row.get('receptor') in receptors or not isinstance(row.get('receptor'),str) or row.get('site') not in ('antenna','palp'):
                raise ValueError('Invalid chemical receptor/site')
            receptors.add(row['receptor'])
            values=[row.get('baseline')]+list(row.get('responses',{}).values())
            if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Finite measured normalized responses required')
            if not row.get('responses') or set(row['responses'])-set(names):raise ValueError('Unknown receptor odorant')
        # Preserve source-table order for repeatable floating-point mixture sums.
        self.spec=copy.deepcopy(spec);self.field=field;self.odorants=tuple(names)

    def observe(self,body,world):
        concentrations={};unlabelled=[]
        for source in world['sources']:
            name=source.get('odorant')
            if name is None:unlabelled.append(source['id']);continue
            if name not in self.odorants:raise ValueError('Odorant absent from calibrated response table: '+str(name))
        for name in self.odorants:
            selected=dict(world,sources=[s for s in world['sources'] if s.get('odorant')==name])
            sample=self.field.observe(body,selected)
            raw=np.asarray(sample['concentration']).sum(axis=1)
            concentrations[name]=raw/(raw+self.field.model['half_concentration'])
        features={};missing=[]
        for row in self.spec['receptors']:
            delta=np.zeros(4)
            for name,occupancy in concentrations.items():
                if name not in row['responses']:
                    if np.any(occupancy):missing.append(dict(receptor=row['receptor'],odorant=name))
                    continue
                delta+=occupancy*(row['responses'][name]-row['baseline'])
            for side,site_index in (('left',0),('right',1)):
                site_index+=2 if row['site']=='palp' else 0
                features[row['receptor']+':'+row['site']+'_'+side]=float(np.clip(delta[site_index],-row['baseline'],1-row['baseline']))
        return dict(features=features,time_s=body.physics_time(),unlabelled_sources=unlabelled,
            unmeasured_pairs=missing,concentration_unit='analytic field occupancy; molar concentration and neural gain not calibrated',
            mixture_model='bounded additive response differences; mixture interactions unvalidated',
            biological_validation=False)
