"""Locate projected-gait information in E07 recorded neural inputs, CPU only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import LEGS
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_m3_presynaptic import readout


def run(source,identification,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'spec.json',{
        'experiment':'M3-E10-information',
        'hypothesis':'Projected knee commands can be read from actual incoming neural activity after afferent correction, even if restricted internal recurrent training failed',
        'competing_explanation':'Temporal instability or insufficient transmitted information still prevents an output readout',
        'models':['all actual direct presynaptic rates','same rates with four causal control samples at0,5,10,15ms','direct sensory presynaptic rates with the same causal memory'],
        'fit':'Unrestricted ridge with train-only centering, fixed scale200, alpha10, no lag search or feature-SD amplification; intercept allowed for diagnosis only',
        'split':'Existing exposed development: train [.5,1.2),validation[1.5,2.2); not independent confirmation',
        'screen':'Every knee validation R2>=.5 for a model before using it as support for incoming-weight calibration',
        'limits':'CPU only, no new neural/physics run or controller adoption',
        'source_sha256':sha256(source/'signals.npz'),'tool_sha256':sha256(__file__),
    })
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    d=np.load(source/'signals.npz')
    c=json.loads((identification/'collection-result.json').read_text())
    g=GraphStore.load(graph_path)
    if g.hash!=c['graph_hash']:
        raise ValueError('Source graph mismatch')
    motor=g.resolve(c['motor_roster'])
    times=np.arange(1,len(d['target'])+1)*.005
    train=(times>=.5)&(times<1.2);valid=(times>=1.5)&(times<2.2)
    # Neural end state was collected before this row's body command.
    all_rates=d['observed_rates'][1:].astype(float)
    ids=d['observed_ids']
    results,predictions={},{}
    for leg,j in zip(LEGS,d['joints'][:,5]):
        muscles=[x for x in c['muscles'] if x['leg']==leg and 'tibia_' in x['muscle']]
        neurons=motor[np.unique(np.concatenate([x['columns'] for x in muscles]))]
        pre=np.unique(np.concatenate([g.indices[g.indptr[i]:g.indptr[i+1]] for i in neurons]))
        cols=np.searchsorted(ids,pre)
        if not np.array_equal(ids[cols],pre):
            raise ValueError('Missing presynaptic states')
        x=all_rates[:,cols]
        sensory=np.array([g.nodes[i].get('super_class')=='sensory' for i in pre])
        def history(values):
            return np.concatenate([np.vstack((np.repeat(values[:1],lag,axis=0),values))[:len(values)] for lag in range(4)],axis=1)
        cases={'incoming':x,'incoming_history':history(x),'sensory_history':history(x[:,sensory])}
        results[leg]={}
        for name,features in cases.items():
            report,pred=readout(features,d['target'][:,j],train,valid,200.)
            results[leg][name]=report
            predictions[leg+'_'+name]=pred
    score={name:[results[leg][name]['validation']['r2'] for leg in LEGS] for name in cases}
    report={'experiment':'M3-E10-information','run_status':'completed','validity':'valid',
        'validation_knee_r2':score,'screen_passed':{name:all(v>=.5 for v in values) for name,values in score.items()},
        'results':results,'adoption':'not_applicable','milestone_outcome':'pending'}
    write_json(out/'result.json',report)
    np.savez_compressed(out/'predictions.npz',**predictions)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','identification','graph','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.identification,a.graph,a.out)
