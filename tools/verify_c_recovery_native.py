#!/usr/bin/env python3
"""Real GPU/MuJoCo dispatcher recovery after an explicitly injected fault."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.server import CDispatcher
from flylab.c import PROTOCOL
from flylab.c.integrity import write_json
from flylab.c.storage import StateStore
from flylab.c.neural import BACKEND_CHOICES

def run(out,graph='data/fafb783/bundle',bindings='data/fafb783/bindings.json',backend='auto'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    d=CDispatcher(graph,bindings,out/'artifacts',backend=backend)
    def call(op,**payload):return d.handle(dict(protocol=PROTOCOL,requestId=1,op=op,payload=payload))['result']
    try:
        call('init',mode='C_STRICT');call('advance',steps=10)
        call('record_start',name='before-fault')
        # This isolates the recovery contract; it is not a naturally occurring physics fault.
        d.engine.body.fault='INJECTED_RECOVERY_VALIDATION_FAULT'
        recovered=call('init',mode='C_STRICT')
        after=call('advance',steps=10)
        call('checkpoint',name='after-recovery')
        saved=StateStore.load(out/'artifacts/checkpoints/after-recovery')
        recovery=recovered['recovery']
        passed=recovery['kind']=='fault_reset' and recovery['diagnostic_saved'] and after['tick']==500 and not after['fault']
        passed &= recovery['diagnostic']['control_tick']==10 and recovery['diagnostic']['model_tick']==500
        result=dict(status='PASS' if passed else 'FAIL',physicalExecuted=True,neural_backend=d.backend,
                    injected_fault=True,recovery=recovery,after_tick=after['tick'],checkpoint_tick=saved['control_tick']*50,
                    natural_fault_incidence='NOT_EVALUATED',biologicalValidation=False)
        write_json(out/'report.json',result);print({k:v for k,v in result.items() if k!='recovery'})
    finally:d.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--graph',default='data/fafb783/bundle');p.add_argument('--bindings',default='data/fafb783/bindings.json')
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto');run(**vars(p.parse_args()))
