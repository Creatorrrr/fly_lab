#!/usr/bin/env python3
"""Compare optional LIF equations on the entire FAFB graph with identical pulses."""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.neural import LIFParameters, ExpLIF, create_backend
from flylab.c.integrity import write_json, file_hash
from flylab.c.body_identity import source_identity


def run(graph_path,out,ticks=1000):
    if not 50<=ticks<=10000:raise ValueError('50..10000 neural ticks required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load(graph_path)
    ids=np.array([i for i,n in enumerate(g.nodes) if n['cell_type']=='ORN_DM1'],np.int32)
    if len(ids)!=68:raise ValueError('Expected 68 FAFB DM1 inputs')
    rng=np.random.default_rng(4205);pulses=(rng.poisson(100*.0001,(ticks,len(ids)))*68.75).astype(np.float32)
    drive=np.zeros(g.n,np.float32);np.savez(out/'inputs.npz',ids=ids,pulses=pulses,drive=drive)
    source=source_identity();shutil.copytree(Path(__file__).resolve().parents[1]/'flylab',out/'source/flylab',ignore=shutil.ignore_patterns('__pycache__'))
    rows=[]
    for name in ('exact-exponential-held-drive-v1','exact-exponential-reset-current-v1','exact-exponential-voltage-events-v1'):
        p=LIFParameters(integration=name);cpu=ExpLIF(g,p);gpu=create_backend(g,p,'exp_lif_mps')
        capture=np.unique(np.concatenate((ids,np.array([i for i,n in enumerate(g.nodes) if n['super_class']=='descending'],np.int32))))[:512]
        aevents=[];bevents=[]
        for start in range(0,ticks,50):
            count=min(50,ticks-start);stim=(ids,pulses[start:start+count])
            cpu.advance(drive,count,capture,stim);gpu.advance(drive,count,capture,stim)
            aevents.extend(cpu.last_events);bevents.extend(gpu.last_events)
        a,b=cpu.snapshot(),gpu.snapshot();errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('v','h','rate','queue')}
        discrete={k:bool(np.array_equal(a[k],b[k])) for k in ('spike_count','refractory_until')}
        passed=all(errors[k]<=(1e-3 if k=='rate' else 1e-4) for k in errors) and all(discrete.values()) and aevents==bevents
        np.savez_compressed(out/(name+'-states.npz'),**{prefix+'_'+k:s[k] for prefix,s in (('cpu',a),('mps',b)) for k in (*errors,*discrete)})
        write_json(out/(name+'-events.json'),dict(cpu=aevents,mps=bevents,capture=capture.tolist()))
        row=dict(integration=name,parameters=asdict(p),max_absolute_error=errors,discrete_equal=discrete,
                 selected_events_equal=aevents==bevents,total_spikes_cpu=int(a['spike_count'].sum()),
                 total_spikes_mps=int(b['spike_count'].sum()),status='PASS' if passed else 'FAIL')
        rows.append(row);print(row,flush=True)
        del cpu,gpu
    report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',cases=rows,
        graph_hash=g.hash,simulated_nodes=g.n,neural_ticks=ticks,seconds=ticks*.0001,
        input_sha256=file_hash(out/'inputs.npz'),source=source,physicalExecuted=False,biological_validation=False)
    write_json(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--out',required=True);p.add_argument('--ticks',type=int,default=1000);a=p.parse_args()
    raise SystemExit(0 if run(a.graph,a.out,a.ticks)['status']=='PASS' else 1)
