#!/usr/bin/env python3
"""Real flapping-wing fluid-force and exact restore tests; not flight approval."""
import argparse
from pathlib import Path
import sys
import shutil
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.flight import FlightRig
from flylab.c.storage import StateStore
from flylab.c.integrity import write_json

def run(assets,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);results=[]
    for fluid in (False,True):
        rig=FlightRig(assets,fluid=fluid);trace=[rig.frame()]
        for _ in range(100):trace.append(rig.step(4))
        saved=rig.snapshot();StateStore.save(out/('fluid-'+str(fluid)+'-checkpoint'),saved)
        expected=rig.step(20);expected_state=rig.snapshot();rig.restore(saved);actual=rig.step(20)
        equal=np.array_equal(expected_state['state'],rig.snapshot()['state'])
        if not equal:raise AssertionError('Flight restore changed physical integration')
        result=dict(fluid=fluid,trace=trace,checkpoint_restore_equal=equal,physicalExecuted=True,
                    final=actual,scope='20 ms articulated wing response; no stable flight, neural control, takeoff or landing approval')
        write_json(out/('fluid-'+str(fluid)+'.json'),result);results.append(result)
        print(fluid,trace[-1]['position_mm'],equal,flush=True)
    difference=np.linalg.norm(np.array(results[0]['final']['position_mm'])-results[1]['final']['position_mm'])
    with tempfile.TemporaryDirectory(prefix='flylab-relocated-wings-') as temporary:
        copied=Path(temporary)/'assets'
        shutil.copytree(assets,copied)
        relocated=FlightRig(copied,fluid=True)
        relocated.restore(saved);relocated.step(20)
        relocation_equal=np.array_equal(expected_state['state'],relocated.snapshot()['state'])
        if not relocation_equal:raise AssertionError('Asset relocation changed restored integration')
    report=dict(status='PASS' if difference>1e-9 else 'FAIL',
               physicalExecuted=True,fluid_effect_position_difference_mm=float(difference),
               checkpoint_restore_equal=all(r['checkpoint_restore_equal'] for r in results),
               relocated_assets_restore_equal=relocation_equal,
               neural_flight='BLOCKED_UNVALIDATED_MAPPING',stable_flight='NOT_VALIDATED')
    write_json(out/'report.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();raise SystemExit(0 if run(a.assets,a.out)['status']=='PASS' else 1)
