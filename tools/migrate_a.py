#!/usr/bin/env python3
"""Explicit cold-start migration: preserve graph/task/seed, NOT physical state."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.brain import validate_graph
from flylab.engine import config_values
from flylab.sensors import default_world,validate_world

def migrate(s):
    if s.get('schema')!='flylab.checkpoint.v1':raise ValueError('A checkpoint v1 required')
    graph=s.get('graph') or s.get('connectome')
    if graph is None:raise ValueError('Graph missing in A checkpoint')
    validate_graph(graph)
    c={k:v for k,v in s.get('config',{}).items() if k in config_values()};c['mode']='walk';c['altitude']=.8
    c=config_values(c);world=default_world()
    # Keep behavioral settings, but build the standardized B physical arena.
    w=s.get('world',{})
    for key in ('cueOn','foodOn'):
        if type(w.get(key)) is bool:world[key]=w[key]
    return dict(schema='flylab.experiment.v2',graph=graph,seed=s['seed'],config=c,world=validate_world(world),migrationNotes=[
        'A neural/kinematic/CPG state is NOT transplanted.',
        'A graph, seed and compatible task settings are preserved.',
        'Arena/obstacles/source positions use B defaults; mode is physical walk.',
        'Simulation begins from a fresh neutral physical state.'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise SystemExit('Output exists; choose a new path.')
    result=migrate(json.loads(Path(a.input).read_text(encoding='utf-8-sig')));dest.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(dest)
