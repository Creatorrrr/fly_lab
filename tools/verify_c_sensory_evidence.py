#!/usr/bin/env python3
"""Execute measured-odor and column-resolved retinal ports on an actual graph."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings
from flylab.c.engine import CEngine
from flylab.c.integrity import write_json
from flylab.sensors import default_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','bindings','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    graph=GraphStore.load(a.graph);bindings=PortBindings(graph,json.loads(a.bindings.read_text(encoding='utf-8')))
    world=default_world();odorants=bindings.spec['sensor_model']['chemical_odor']['odorants']
    for i,source in enumerate(world['sources']):source['odorant']=odorants[i%len(odorants)]['inchikey']
    write_json(a.out/'world.json',world)
    e=CEngine(graph,bindings,mode='C_STRICT',backend='auto',world=world)
    try:
        for _ in range(4):e.step(1)
        saved=e.checkpoint();e.step(2);expected=e.neural.snapshot()['v'];expected_q=e.body.d.qpos.copy()
        restored=CEngine.from_checkpoint(graph,bindings,saved)
        try:
            restored.step(2)
            neural_exact=bool(np.array_equal(expected,restored.neural.snapshot()['v']))
            body_exact=bool(np.array_equal(expected_q,restored.body.d.qpos))
            deltas=dict(neural_max_abs=float(np.max(np.abs(expected-restored.neural.snapshot()['v']))),
                retinal_max_abs=float(np.max(np.abs(e.sensors.retina.filtered-restored.sensors.retina.filtered))),
                encoder_max_abs=float(np.max(np.abs(e.encoder.filtered-restored.encoder.filtered))),
                retinal_raw_equal=bool(np.array_equal(e.sensors.retina.previous,restored.sensors.retina.previous)))
        finally:restored.close()
        e.encoder.restore(saved['encoder'])
        packet=e.last_sensors;features=e.sensors.diagnostics['features']
        on=e.encoder.encode(packet,.005,.0001,supplemental=features)[0]
        off=e.encoder.encode(packet,.005,.0001,supplemental=features,disabled=['*'])[0]
        report=dict(graph_hash=graph.hash,binding_hash=bindings.hash,neurons=graph.n,ports=len(bindings.sensory),
            backend=e.neural.backend,retina_compute=e.body._eye_observer.compute.backend,
            body_future_exact=body_exact,neural_future_exact=neural_exact,deltas=deltas,
            enabled_drive_sum=float(on.sum()),disabled_drive_sum=float(off.sum()),
            distinct_retinal_features=len(set(features['retina_ommatidia'][0])),
            chemical=e.sensors.diagnostics['chemical_odor'],biological_validation=False)
        write_json(a.out/'report.json',report);print(json.dumps(report))
        if not body_exact or not neural_exact or not on.sum()>0 or off.any():raise RuntimeError('Sensory execution/restore failed; see report.json')
    finally:e.close()


if __name__=='__main__':main()
