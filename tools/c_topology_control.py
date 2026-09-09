#!/usr/bin/env python3
"""Connectome topology sensitivity under exactly shared sensory input.

Permute presynaptic identities within transmitter classes. This retains every
neuron, each postsynaptic neuron's total anatomical input, the weight multiset,
and the sign at every contact; it changes which source identity carries an
outgoing connectivity profile. It is not a uniform degree-preserving edge swap.
No claim that the original graph performs better is built into the test.
"""
import argparse
import copy
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore,normalized_nt
from flylab.c.ports import PortBindings,SensoryEncoder,MotorDecoder
from flylab.c.neural import ExpLIF
from flylab.c.integrity import read_json,write_json,digest


def permute_sources(graph,seed):
    rng=np.random.default_rng(seed)
    permutation=np.arange(graph.n,dtype=np.int32)
    groups={}
    for i,n in enumerate(graph.nodes):groups.setdefault(normalized_nt(n.get('nt_type')),[]).append(i)
    for ids in groups.values():permutation[ids]=rng.permutation(ids)
    pre=permutation[graph.indices]
    post=np.repeat(np.arange(graph.n,dtype=np.int32),np.diff(graph.indptr))
    metadata={k:v for k,v in graph.manifest.items() if k not in ('graph_hash','bundle_files')}
    metadata.update(scope='rewired_control',control_parent_graph_hash=graph.hash,
                    control=dict(kind='source_identity_permutation_within_nt',seed=seed,permutation_hash=digest(permutation.tolist())))
    rewired=GraphStore.from_edges(graph.nodes,pre,post,graph.counts,metadata=metadata,
                                 unit_weight=graph.manifest['weight_model']['unit_weight_mV'],
                                 unknown_policy=graph.manifest['weight_model']['unknown_policy'])
    return rewired


def compare(graph,bindings,output,seconds=.05,seed=42):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    count=round(seconds/.005)
    if count<1 or abs(count*.005-seconds)>1e-9:raise ValueError('Duration must be positive 5ms periods')
    # Recorded input trace is itself explicit and identical in all conditions.
    packet=dict(schema='flylab.sensors.v2',panorama=[0.]*64,nearRanges=[10.]*9,odor=[.8,.8],
                odorChange=0.,danger=0.,angularVelocity=0.,forwardSpeed=0.,clearanceDown=1.,clearanceUp=10.,contact=0)
    rows=[]
    for arm in ('original','rewired','sensor_off'):
        g=permute_sources(graph,seed) if arm=='rewired' else graph
        spec=copy.deepcopy(bindings.spec);spec['graph_hash']=g.hash
        b=PortBindings(g,spec);e=SensoryEncoder(b,seed);n=ExpLIF(g);decoder=MotorDecoder(b)
        trace=[];start=time.perf_counter()
        for _ in range(count):
            drive,pulses,values=e.encode(packet,.005,.0001,['*'] if arm=='sensor_off' else [])
            n.advance(drive,50,b.motor_indices,pulses)
            command,rates=decoder.decode(n)
            trace.append(dict(tick=n.tick,command=command,motor_rates_Hz=rates,ports=values))
        np.save(output/(arm+'_voltage.npy'),n.v,allow_pickle=False)
        np.save(output/(arm+'_spike_counts.npy'),n.spike_count,allow_pickle=False)
        row=dict(arm=arm,graph_hash=g.hash,binding_hash=b.hash,input_hash=digest(packet),
                 total_spikes=int(n.spike_count.sum()),summary=n.summary(),wall_seconds=time.perf_counter()-start,
                 physicalExecuted=False,trace=trace)
        rows.append(row);write_json(output/(arm+'.json'),row)
        del n
    original=np.load(output/'original_spike_counts.npy',allow_pickle=False)
    rewired=np.load(output/'rewired_spike_counts.npy',allow_pickle=False)
    sensor_off=np.load(output/'sensor_off_spike_counts.npy',allow_pickle=False)
    result=dict(schema='flylab.topology-control.v1',status='COMPLETE',scope='full graph open-loop sensory sensitivity',
                seconds=seconds,seed=seed,physicalExecuted=False,biologicalValidation=False,
                original_vs_rewired_changed_neurons=int(np.count_nonzero(original!=rewired)),
                original_vs_sensor_off_changed_neurons=int(np.count_nonzero(original!=sensor_off)),
                no_behavioral_superiority_claim=True,arms=[{k:v for k,v in r.items() if k!='trace'} for r in rows])
    write_json(output/'comparison.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,default=Path('data/fafb783/bundle'))
    p.add_argument('--bindings',type=Path,default=Path('data/fafb783/bindings.json'))
    p.add_argument('--out',type=Path,default=Path('verification/c_topology'))
    p.add_argument('--seconds',type=float,default=.05)
    a=p.parse_args();g=GraphStore.load(a.graph);b=PortBindings(g,read_json(a.bindings))
    print(compare(g,b,a.out,a.seconds))
