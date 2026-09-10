#!/usr/bin/env python3
"""Fixed, mirrored sensory assays on a full graph. No body or goal feedback.

Reports bias and response separately. It does not automatically change gains
or select a profile based on a successful food trajectory.
"""
import argparse
import shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.graph import GraphStore
from flylab.c.neural import create_backend, LIFParameters
from flylab.c.ports import PortBindings, SensoryEncoder, MotorDecoder
from flylab.c.body_identity import source_identity
from flylab.c.integrity import read_json, write_json, file_hash

STIMULI=(('silent',0.,0.),('symmetric_low',.1,.1),('symmetric',.3,.3),
         ('symmetric_high',.6,.6),('left',.35,.2),('right',.2,.35),
         ('left_holdout',.45,.3),('right_holdout',.3,.45))


def packet(left,right):
    return dict(schema='flylab.sensors.v2',panorama=[0.]*64,nearRanges=[10.]*9,
                odor=[left,right],odorChange=0.,danger=0.,angularVelocity=0.,
                forwardSpeed=0.,clearanceDown=1.,clearanceUp=10.,contact=0.)


def run(graph_path, binding_paths, out, seconds=2., seeds=(42,197)):
    if not 1.<=seconds<=10. or abs(seconds/.005-round(seconds/.005))>1e-8:
        raise ValueError('1..10 seconds on a control boundary required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    graph=GraphStore.load(graph_path);source=source_identity()
    shutil.copytree(Path(__file__).resolve().parents[1]/'flylab',out/'source/flylab',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,out/'source/probe_c_steering.py')
    reports=[]
    for path in binding_paths:
        bindings=PortBindings(graph,read_json(path));directory=out/Path(path).stem
        directory.mkdir();write_json(directory/'bindings.json',bindings.spec)
        params=LIFParameters(**bindings.spec.get('neural_parameters',{}))
        rows=[]
        for seed in seeds:
            for label,left,right in STIMULI:
                neural=create_backend(graph,params,'exp_lif_mps')
                encoder=SensoryEncoder(bindings,seed);decoder=MotorDecoder(bindings)
                trace=[]
                for k in range(round(seconds/.005)):
                    drive,pulses,ports=encoder.encode(packet(left,right),.005,params.dt,
                        supplemental=dict(loom_left=0.,loom_right=0.,head_contact=0.))
                    neural.advance(drive,round(.005/params.dt),bindings.motor_indices,pulses)
                    command,rates=decoder.decode(neural)
                    trace.append(dict(time_s=(k+1)*.005,command=command,rates_Hz=rates,
                                      input_values=[p['value'] for p in ports]))
                late=trace[len(trace)//2:]
                forward=float(np.mean([r['command']['forwardSpeed'] for r in late]))
                yaw=float(np.mean([r['command']['yawRate'] for r in late]))
                row=dict(name=label,seed=seed,odor=[left,right],mean_forward=forward,mean_yaw=yaw,
                    motor_mean_Hz={role:float(np.mean([r['rates_Hz'][role] for r in late])) for role in bindings.motor},
                    motor_spikes=neural.readout(bindings.motor_indices)['spike_count'].tolist())
                target=directory/f'{seed}-{label}.json';write_json(target,trace)
                row.update(trace_file=target.name,trace_sha256=file_hash(target));rows.append(row)
                print(Path(path).stem,seed,label,round(forward,4),round(yaw,4),flush=True)
                write_json(directory/'progress.json',dict(status='RUNNING',rows=rows))
                del neural
        checks=[]
        for seed in seeds:
            lookup={r['name']:r for r in rows if r['seed']==seed}
            checks.append(dict(seed=seed,
                silent_zero=abs(lookup['silent']['mean_forward'])<1e-6 and abs(lookup['silent']['mean_yaw'])<1e-6,
                symmetric_unbiased=all(abs(lookup[n]['mean_yaw'])<.05 for n in ('symmetric_low','symmetric','symmetric_high')),
                both_directions=lookup['left']['mean_yaw']<-.02 and lookup['right']['mean_yaw']>.02,
                holdout_both_directions=lookup['left_holdout']['mean_yaw']<-.02 and lookup['right_holdout']['mean_yaw']>.02))
        report=dict(profile=bindings.spec.get('profile'),binding_hash=bindings.hash,rows=rows,checks=checks,
            status='PASS' if all(all(v for k,v in c.items() if k!='seed') for c in checks) else 'FAIL')
        write_json(directory/'report.json',report);reports.append(report)
    report=dict(schema='flylab.steering-probe.v1',source=source,graph_hash=graph.hash,simulated_nodes=graph.n,
        seconds_per_assay=seconds,seeds=list(seeds),stimuli=STIMULI,profiles=reports,
        physicalExecuted=False,biological_validation=False,automatic_profile_adoption=False)
    write_json(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',default='data/fafb783/bundle')
    p.add_argument('--bindings',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--seeds',type=int,nargs='+',default=[42,197])
    a=p.parse_args();run(a.graph,a.bindings,a.out,a.seconds,a.seeds)
