#!/usr/bin/env python3
"""C acceptance evidence with separate neural, physical and behavioral gates.

The default acceptance scope requires an independently pinned full snapshot.
--scope loaded-graph permits numerical fixtures without granting full status.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import platform
import resource
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,default=Path('data/fafb783/bundle'))
    parser.add_argument('--bindings',type=Path,default=Path('data/fafb783/bindings.json'))
    parser.add_argument('--out',type=Path,default=Path('verification/c_native'))
    parser.add_argument('--physics',action='store_true')
    parser.add_argument('--seconds',type=float,default=.3)
    parser.add_argument('--compute-seconds',type=float,default=.05)
    parser.add_argument('--navigation-seconds',type=float,default=3.2)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--backend', choices=('exp_lif_cpu_reference','exp_lif_mps','exp_lif_cuda'), default='exp_lif_cpu_reference')
    parser.add_argument('--scope', choices=('full-snapshot','loaded-graph'), default='full-snapshot')
    parser.add_argument('--reference', type=Path, help='Independent snapshot reference')
    args=parser.parse_args()
    if not .1<=args.seconds<=10 or not 0<=args.navigation_seconds<=60: parser.error('Invalid experiment duration')
    if args.out.exists(): parser.error('Use a new output directory to preserve previous evidence')
    args.out.mkdir(parents=True)
    result=dict(schema='flylab.c.validation.v2',status='RUNNING',physicalExecuted=False,
                biologicalValidation=False,platform=platform.platform(),gates={},cases=[])
    def persist(): (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    persist()
    try:
        import numpy as np
        from flylab.c.graph import GraphStore
        from flylab.c.ports import PortBindings
        from flylab.c.neural import ExpLIF,LIFParameters,create_backend
        from flylab.c.model_config import resolve_parameters, model_identity, model_ticks
        from flylab.c.data_identity import verify_snapshot, validation_status
        from flylab.c.engine import CEngine
        from flylab.c.integrity import read_json,write_json
        from flylab.c.storage import StateStore
        from flylab.c.experiments import paired
        from tools.navigation_checks import NavigationMonitor
        from flylab.dependencies import dependency_report
        t=time.perf_counter();g=GraphStore.load(args.graph);bindings=PortBindings(g,read_json(args.bindings))
        result['gates']['data']=dict(status='PASS',load_seconds=time.perf_counter()-t,manifest=g.manifest)
        result['gates']['bindings']=dict(status='ENGINEERING_REVIEWED',summary=bindings.summary())
        result['requested_scope']=args.scope
        required=['data','loaded_graph_compute']
        if args.scope=='full-snapshot': required.extend(['full_snapshot_membership','full_snapshot_compute'])
        if args.cuda: required.append('cuda')
        if args.physics: required.extend(['closed_loop_contribution','physical_restore'])
        if args.physics and args.navigation_seconds: required.append('strict_navigation')
        result['required_gates']=required
        result['gates']['full_snapshot_membership']=verify_snapshot(g,args.reference)
        persist()
        parameters=resolve_parameters(bindings)
        computed_ticks=model_ticks(args.compute_seconds,parameters.dt)
        n=create_backend(g,parameters,args.backend); drive=np.zeros(g.n,dtype=np.float32);drive[bindings.motor_indices]=12.
        t=time.perf_counter();n.advance(drive,computed_ticks,capture=bindings.motor_indices)
        duration=time.perf_counter()-t
        model_seconds=computed_ticks*parameters.dt
        result['gates']['loaded_graph_compute']=dict(status='PASS',model_seconds=model_seconds,requested_model_seconds=args.compute_seconds,
            computed_ticks=computed_ticks,model=model_identity(parameters),wall_seconds=duration,backend=n.backend,
            sim_wall_ratio=model_seconds/duration,summary=n.summary(),sparse_bytes=int(g.indptr.nbytes+g.indices.nbytes+g.counts.nbytes+g.weights.nbytes),
            neural_state_bytes=sum(g.n*(n.slots if k=='queue' else 1)*(8 if k in ('refractory_until','spike_count') else 1 if k in ('suppress','mute') else np.dtype(n.p.dtype).itemsize) for k in ['v','h','rate','queue','refractory_until','spike_count','suppress','mute']),
            cpu_peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_native_unit='bytes' if sys.platform=='darwin' else 'KiB')
        membership=result['gates']['full_snapshot_membership']
        result['gates']['full_snapshot_compute']=dict(status=membership['status'],simulated_nodes=n.n,computed_ticks=n.tick,
            reason='Full source membership and actual state/ticks required',backend=n.backend,model=model_identity(parameters))
        if membership['status']=='PASS' and (n.n!=g.n or n.tick!=computed_ticks): result['gates']['full_snapshot_compute']['status']='FAIL'
        del n
        result['gates']['cuda']=dict(status='NOT_RUN',reason='No CUDA verification requested; CPU success is not GPU validation')
        if args.cuda:
            from tests.c_fixtures import graph_fixture
            toy=graph_fixture();cpu=ExpLIF(toy)
            try:
                gpu=ExpLIF(toy,backend='exp_lif_cuda')
                x=np.zeros(toy.n,dtype=np.float32);x[0]=30.
                cpu.advance(x,1000,list(range(toy.n)));gpu.advance(x,1000,list(range(toy.n)))
                delta=float(np.max(np.abs(cpu.v-gpu.host(gpu.v))))
                spike_equal=cpu.last_events==gpu.last_events
                result['gates']['cuda']=dict(status='PASS' if delta<1e-3 and spike_equal else 'FAIL',
                                              voltage_max_abs_mV=delta,spike_times_equal=spike_equal,scope='synthetic numerical circuit')
            except Exception as ex: result['gates']['cuda']=dict(status='BLOCKED',reason=str(ex))
        persist();print('Loaded-graph compute:',result['gates']['loaded_graph_compute'],flush=True)
        if args.physics:
            report=dependency_report()
            if not report['ready']: raise RuntimeError('BLOCKED_PHYSICS: '+str(report))
            base=CEngine(g,bindings,mode='C_STRICT',backend=args.backend)
            try:
                cp=base.checkpoint();StateStore.save(args.out/'initial_checkpoint',cp)
                # Calibrate actual body response to bounded direct DN input and
                # compare motor disconnection from exactly the same state.
                base.schedule(dict(kind='stimulate',ids=bindings.spec['motor']['forward']['ids'],amplitude_mV=20.,
                                   duration_controls=round((args.seconds+.1)/.005)))
                comparison=paired(base,args.out/'motor_contribution',dict(kind='motor_disconnect'),
                                  seconds=args.seconds,onset=.1)
                result['gates']['closed_loop_contribution']=dict(status='PASS' if comparison['status']=='COMPLETE'
                    and comparison['pre_intervention_matched'] and (comparison['mean_path_separation_mm'] or 0)>.01 else 'FAIL',
                    evidence=comparison,scope='experimental direct DN stimulation; not natural navigation')
                # Restore while delay currents and rate filters are active.
                base.step(10);saved=base.checkpoint();StateStore.save(args.out/'active_checkpoint',saved)
                base.step(10); expected_neural=base.neural.snapshot();expected_body=np.asarray(base.body.snapshot()['state'])
                restored=CEngine.from_checkpoint(g,bindings,StateStore.load(args.out/'active_checkpoint'))
                try:
                    restored.step(10)
                    actual_neural=restored.neural.snapshot()
                    neural_error=float(np.max(np.abs(expected_neural['v']-actual_neural['v'])))
                    queue_equal=bool(np.array_equal(expected_neural['queue'],actual_neural['queue']))
                    body_error=float(np.max(np.abs(expected_body-np.asarray(restored.body.snapshot()['state']))))
                    result['gates']['physical_restore']=dict(status='PASS' if neural_error==0 and body_error<1e-7 and queue_equal else 'FAIL',
                        neural_voltage_max_abs_mV=neural_error,queue_equal=queue_equal,body_state_max_abs=body_error,continuation_seconds=.05)
                finally:restored.close()
            finally:base.close()
            persist()
            cases=[dict(name='seed7_friction_low',seed=7,friction=.5,mode='C_STRICT',role='forward'),
                   dict(name='seed19_friction_high',seed=19,friction=1.5,mode='C_STRICT',role='forward'),
                   dict(name='seed42_backward',seed=42,friction=1.,mode='C_STRICT',role='backward'),
                   dict(name='seed42_yaw_left',seed=42,friction=1.,mode='C_STRICT',role='yaw_left'),
                   dict(name='seed42_yaw_right',seed=42,friction=1.,mode='C_STRICT',role='yaw_right'),
                   dict(name='seed7_obstacle_assisted',seed=7,friction=1.,mode='C_ASSISTED',role='forward',obstacle=True),
                   dict(name='seed19_sensor_off',seed=19,friction=1.,mode='C_STRICT',role=None,sensor_off=True),
                   dict(name='seed42_shadow',seed=42,friction=1.,mode='C_SHADOW',role=None)]
            for case in cases:
                print('Native case:',case['name'],flush=True)
                e=CEngine(g,bindings,mode=case['mode'],seed=case['seed'],config=dict(friction=case['friction']),backend=args.backend)
                directory=args.out/case['name']; trace=[]; begin=time.perf_counter();start=e.body.frame()[0]
                e.start_recording(directory)
                try:
                    if case.get('obstacle'):e.command('place',dict(kind='obstacle',position=[4.6,.8,0],radius=.8))
                    if case.get('sensor_off'):e.schedule(dict(kind='sensor_off',channels=['*'],duration_controls=round(args.seconds/.005)))
                    if case['role']:e.schedule(dict(kind='stimulate',ids=bindings.spec['motor'][case['role']]['ids'],amplitude_mV=20.,duration_controls=round(args.seconds/.005)))
                    for _ in range(round(args.seconds/.05)):
                        f=e.step(10);trace.append(dict(t=f['simTime'],body=f['body'],physics=f['physics'],command=f['command'],neural=f['neural']))
                        if f['fault']:break
                    end=e.body.frame()[0]
                    row=dict(**case,status='PASS' if not e.fault and abs(e.control_tick*.005-args.seconds)<1e-8 else 'FAIL',
                             physicalExecuted=e.control_tick>0,model_seconds=e.control_tick*.005,wall_seconds=time.perf_counter()-begin,
                             fault=e.fault,horizontal_displacement_mm=float(np.linalg.norm((np.asarray(end['position'])-start['position'])[[0,2]])),
                             yaw_change_rad=math.atan2(math.sin(end['yaw']-start['yaw']),math.cos(end['yaw']-start['yaw'])),
                             joint_range_rad=float(np.max(np.ptp(np.asarray([r['physics']['jointAngles'] for r in trace]),axis=0))) if trace else 0.,
                             max_contact_BW=max(max(r['physics']['contactsBW']) for r in trace) if trace else 0.,
                             command_sources=sorted({r['command']['command_source'] for r in trace}),
                             interpretation='Short physical integration case; no long-horizon navigation claim')
                    write_json(directory/'case.json',dict(summary=row,trajectory=trace,provenance=e.provenance()))
                    if not e.fault:StateStore.save(directory/'final_checkpoint',e.checkpoint())
                    result['cases'].append(row);result['physicalExecuted']|=row['physicalExecuted']
                finally:e.close()
                persist()
            if args.navigation_seconds:
                print('Native C_STRICT natural-sensory navigation:',args.navigation_seconds,'model seconds',flush=True)
                e=CEngine(g,bindings,mode='C_STRICT',seed=42,backend=args.backend);monitor=NavigationMonitor();trace=[]
                e.start_recording(args.out/'strict_navigation')
                try:
                    for _ in range(round(args.navigation_seconds/.1)):
                        f=e.step(20)
                        row=dict(t=f['simTime'],p=f['body']['position'],yaw=f['body']['yaw'],avoiding=False,recovering=False,
                                 motorCoupled=f['command']['motor_coupled'],motionExpected=True,
                                 forwardCommand=f['command']['u_final']['forwardSpeed'],yawCommand=f['command']['u_final']['yawRate'])
                        monitor.add(row);trace.append(row)
                        if f['fault']:break
                    gate=monitor.result()
                    result['gates']['strict_navigation']=dict(status='FAIL' if e.fault else gate['status'],
                        fault=e.fault,navigation=gate,model_seconds=e.control_tick*.005,
                        scope='seed42, natural scalar odor, strict mode; short stationary monitor only',
                        stable_exploration_validated=False,
                        forward_command_samples=sum(abs(row['forwardCommand'])>.01 for row in trace),
                        interpretation='A stationary-monitor PASS does not establish forward exploration or obstacle avoidance.')
                    write_json(args.out/'strict_navigation/navigation.json',dict(gate=result['gates']['strict_navigation'],trajectory=trace))
                finally:e.close()
            else:result['gates']['strict_navigation']=dict(status='NOT_RUN')
        else:
            result['gates']['closed_loop_contribution']=dict(status='NOT_RUN')
            result['gates']['strict_navigation']=dict(status='NOT_RUN')
        verdict=validation_status(result['gates'],required,result['cases'])
        result.update(status=verdict['status'],acceptance=verdict)
        persist();print(json.dumps({k:v for k,v in result.items() if k not in ('gates','cases')},indent=2),flush=True)
        return verdict['exit_code']
    except Exception as e:
        result.update(status='BLOCKED' if 'BLOCKED' in str(e) or isinstance(e,ImportError) else 'FAIL',reason=str(e),traceback=traceback.format_exc())
        persist();print(str(e),file=sys.stderr);return 2 if result['status']=='BLOCKED' else 1


if __name__=='__main__':raise SystemExit(main())
