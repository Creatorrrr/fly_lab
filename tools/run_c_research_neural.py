#!/usr/bin/env python3
"""Explicit physiology/learning trials; rewards never carry a steering target."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.research_neural import ResearchLIF
from flylab.c.storage import StateStore
from flylab.c.integrity import read_json,write_json,finite,bounded_int

def run(graph,profile,inputs,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load(graph);spec=read_json(profile);sequence=read_json(inputs)
    n=ResearchLIF(g,spec,device);capture=g.resolve(sequence['capture_ids']);trace=[]
    if len(capture)>512:raise ValueError('Maximum 512 recorded cells')
    write_json(out/'profile.json',spec);write_json(out/'inputs.json',sequence)
    StateStore.save(out/'initial',n.snapshot())
    for segment in sequence['segments']:
        ticks=bounded_int(segment['ticks'],'segment ticks',1,10000)
        drive=np.zeros(g.n,np.float32)
        for stimulus in segment.get('stimuli',[]):
            np.add.at(drive,g.resolve(stimulus['ids']),finite(stimulus['amplitude_mV'],'stimulus',-100.,100.))
        n.advance(drive,ticks,capture,reward=segment.get('reward',0.))
        trace.append(dict(tick=n.tick,spikes=n.last_events,**{k:v.tolist() for k,v in n.readout(capture).items()}))
    StateStore.save(out/'final',n.snapshot())
    write_json(out/'report.json',dict(status='COMPLETE',backend=n.backend,profile_hash=n.hash,graph_hash=g.hash,
                                     simulated_nodes=g.n,neural_ticks=n.tick,trace=trace,physicalExecuted=False,
                                     learned_behavior='NOT_EVALUATED',biological_validation=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','profile','inputs','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=('cpu','mps'),default='mps');a=p.parse_args();run(**vars(a))
