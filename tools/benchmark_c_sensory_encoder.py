"""Check ordered vector encoding against the retained scalar/Poisson path."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.ports import PortBindings,SensoryEncoder
from flylab.c.integrity import write_json


def main():
    from flylab.body import FlyGymBody
    from flylab.sensors import SensorAdapter,default_world
    from flylab.engine import config_values
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('graph','bindings','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    g=GraphStore.load(a.graph);b=PortBindings(g,json.loads(a.bindings.read_text(encoding='utf-8')))
    fast,scalar=SensoryEncoder(b,42),SensoryEncoder(b,42)
    if not fast.all_drives:raise ValueError('A held-drive profile required')
    scalar.all_drives=False
    world=default_world();body=FlyGymBody(42,world,config_values())
    try:packet=SensorAdapter(42).observe(body,world,.005,config_values())
    finally:body.close()
    rng=np.random.default_rng(42)
    features=dict(retina_ommatidia=rng.uniform(0,1,(2,721)).tolist(),chemical_odor={
        p['receptor']+':'+p['site']:float(rng.uniform(-.1,.5)) for p,_ in b.sensory if p['channel']=='chemical_odor'})
    exact=True
    for i in range(30):
        disabled=['*'] if i%4==0 else [b.sensory[0][0]['name']] if i%4==1 else []
        dt=.005 if i%3 else .01
        left=fast.encode(packet,dt,.0001,disabled,features);right=scalar.encode(packet,dt,.0001,disabled,features)
        exact &= bool(np.array_equal(left[0],right[0]) and left[2]==right[2] and np.array_equal(fast.filtered,scalar.filtered) and fast.delays==scalar.delays)
    timings={}
    for name,encoder in [('scalar',scalar),('vectorized',fast)]:
        rows=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(30):encoder.encode(packet,.005,.0001,supplemental=features)
            rows.append((time.perf_counter()-start)/30)
        timings[name]=float(np.median(rows))
    report=dict(exact=exact,ports=len(b.sensory),graph_hash=g.hash,binding_hash=b.hash,
        encoding_seconds=timings,speedup=timings['scalar']/timings['vectorized'],
        scope='encoder only; synthetic retinal/chemical features on real anatomical targets, no neural/physics/render timing')
    write_json(a.out/'report.json',report);print(json.dumps(report))
    if not exact:raise RuntimeError('Vectorized port encoding differs from ordered scalar reference')


if __name__=='__main__':main()
