#!/usr/bin/env python3
"""Actual full-graph CPU/MPS comparison and fixed-input neural-dt refinement."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.neural import create_backend,LIFParameters
from flylab.c.integrity import read_json,write_json

def run(graph_path,binding_path,out,seconds=.05):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load(graph_path);b=PortBindings(g,read_json(binding_path))
    sensory=np.unique(np.concatenate([ids for _,ids in b.sensory]));capture=np.unique(np.r_[sensory[:64],b.motor_indices]).astype(np.int32)
    reports=[]
    for name,indices,amplitude in [('quiet',[],0.),('odor',sensory,18.),('descending',b.motor_indices,20.)]:
        drive=np.zeros(g.n,np.float32);drive[np.asarray(indices,dtype=int)]=amplitude
        cpu=create_backend(g);gpu=create_backend(g,backend='exp_lif_mps');first=None;maxima={k:0. for k in ('v','h','queue','rate')};events=[]
        for _ in range(round(seconds/.005)):
            cpu.advance(drive,50,capture);gpu.advance(drive,50,capture)
            a=cpu.snapshot();z=gpu.snapshot();diff={k:float(np.max(np.abs(a[k]-z[k]))) for k in maxima}
            maxima={k:max(maxima[k],v) for k,v in diff.items()}
            discrete=all(np.array_equal(a[k],z[k]) for k in ('refractory_until','spike_count','suppress','mute'))
            same_events=cpu.last_events==gpu.last_events
            passed=discrete and same_events and all(v<=(1e-3 if k=='rate' else 1e-4) for k,v in diff.items())
            if not passed and first is None:first=dict(control_boundary_tick=cpu.tick,differences=diff,discrete_equal=discrete,selected_events_equal=same_events)
            events.append(dict(tick=cpu.tick,cpu=cpu.last_events,mps=gpu.last_events))
        report=dict(name=name,status='PASS' if first is None else 'FAIL',max_absolute_error=maxima,first_differing_boundary=first,
                    computed_nodes=g.n,ticks=cpu.tick,seconds=seconds,selected_ids=[g.nodes[int(i)]['id'] for i in capture])
        write_json(out/(name+'-events.json'),events);reports.append(report);print(report,flush=True)
        del cpu,gpu
    refinement=[]
    for dt in (.0001,.00005,.000025):
        n=create_backend(g,LIFParameters(dt=dt));drive=np.zeros(g.n,np.float32);drive[sensory]=18.
        events=[];began=time.perf_counter()
        for _ in range(round(seconds/.005)):
            n.advance(drive,round(.005/dt),capture)
            events.extend(dict(seconds=e['tick']*dt,index=e['index']) for e in n.last_events)
        r=n.readout(capture)
        refinement.append(dict(dt=dt,seconds=seconds,ticks=n.tick,wall_seconds=time.perf_counter()-began,
                         selected_spike_count=r['spike_count'].tolist(),selected_events=events,
                         selected_voltage_mV=r['voltage_mV'].tolist(),computed_nodes=g.n))
        del n
    fine=refinement[-1]
    for row in refinement:
        row['max_selected_voltage_difference_from_finest_mV']=float(np.max(np.abs(np.array(row['selected_voltage_mV'])-fine['selected_voltage_mV'])))
        row['selected_spike_count_equal_finest']=row['selected_spike_count']==fine['selected_spike_count']
    result=dict(schema='flylab.numerics.v1',graph_hash=g.hash,binding_hash=b.hash,physicalExecuted=False,
                device_comparison=reports,refinement=refinement,
                backend_gate='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',
                convergence_gate='MEASURED_NOT_UNIVERSAL_APPROVAL',cuda='BLOCKED_NO_NVIDIA_DEVICE',
                interpretation='Fixed-input neural comparison only; dt refinement changes discrete spike timing and is not a physical timestep benchmark.')
    write_json(out/'report.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--bindings',default='data/fafb783/bindings.json');p.add_argument('--out',required=True);a=p.parse_args();run(a.graph,a.bindings,a.out)
