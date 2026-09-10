"""Independent, resumable, chunked physical campaigns; no shared live engine."""
import copy
import os
import time
from pathlib import Path
from . import CONTROL_DT, MODES
from .engine import CEngine
from .body_identity import source_identity
from .behavior import trace_sample
from .tasks import evaluate, control_parameters
from .storage import StateStore, runtime_versions
from .integrity import canonical, read_json, write_json, digest, finite, bounded_int, checked_name, file_hash
from .diagnostics import fault_report
from ..body import FlyGymBody
from ..sensors import default_world

SCENES=('baseline','food_left','food_right','hazard_left','hazard_right','front_obstacle')

def pilot_spec(seconds=10., seeds=(7,19,42), modes=('C_STRICT','C_SHADOW','C_ASSISTED')):
    return dict(schema='flylab.campaign.v1',cases=[dict(name=f'{mode}-{seed}-{scene}',seed=seed,mode=mode,
                scene=scene,seconds=seconds) for seed in seeds for scene in SCENES for mode in modes])

def validate_spec(spec):
    if spec.get('schema')!='flylab.campaign.v1' or set(spec)!={'schema','cases'}: raise ValueError('Campaign v1 required')
    if not isinstance(spec['cases'],list) or not 1<=len(spec['cases'])<=1000: raise ValueError('1..1000 cases required')
    names=set()
    for c in spec['cases']:
        if set(c)-{'name','seed','mode','scene','seconds','config','intervention','interventions','world','task','task_parameters','initial_pose'}: raise ValueError('Unknown case field')
        if 'intervention' in c and 'interventions' in c: raise ValueError('Use one intervention field')
        if 'interventions' in c and (not isinstance(c['interventions'],list) or len(c['interventions'])>100 or any(not isinstance(e,dict) for e in c['interventions'])):
            raise ValueError('At most 100 intervention objects required')
        name=checked_name(c['name'])
        if name in names:raise ValueError('Duplicate case name')
        names.add(name);bounded_int(c['seed'],'seed',0,2**32-1)
        if c['mode'] not in MODES or c['scene'] not in SCENES:raise ValueError('Unknown campaign scene/mode')
        if c.get('task','diagnostic') not in ('walking','backward','food','hazard','obstacle','diagnostic','yaw_left','yaw_right','stop_resume'):
            raise ValueError('Unknown campaign task')
        control_parameters(c.get('task','diagnostic'),c.get('task_parameters'))
        seconds=finite(c['seconds'],'case model seconds',.005,3600.)
        if abs(seconds/CONTROL_DT-round(seconds/CONTROL_DT))>1e-8:raise ValueError('Integral control duration required')
    return copy.deepcopy(spec)

def scene_world(scene):
    w=default_world();w['sources']=[];w['obstacles']=[]
    if scene.startswith(('food_','hazard_')):
        kind,side=scene.split('_');w['sources']=[dict(id='target',kind=kind,p=[8.,.7,-4. if side=='left' else 4.],strength=1.2)]
    if scene=='front_obstacle':w['obstacles']=[dict(id='target',p=[7.,1.,0.],r=1.)]
    return w

def result_summary(result, path, root):
    """Keep frequent progress writes independent of full trajectory metrics.

    Detailed results stay immutable in each case directory. The manifest pins
    their content instead of rewriting every stationary window on every tick.
    """
    fields=('name','seconds','execution_status','case','physical','technical_status',
            'task_status','task','elapsed_s','horizontal_net_mm','criteria_hash',
            'biological_validation','error')
    return dict(**{key:result[key] for key in fields if key in result},
                result_file=path.relative_to(root).as_posix(),result_sha256=file_hash(path))

def verify_result_files(manifest, root):
    for row in manifest['cases']:
        if 'result_file' in row:
            path=(root/row['result_file']).resolve();path.relative_to(root.resolve())
            if file_hash(path)!=row['result_sha256']:raise ValueError('Case result hash mismatch')

def run_campaign(graph, bindings, spec, out, *, backend='exp_lif_mps', resume=False,
                 body_factory=FlyGymBody, cancelled=lambda:False, checkpoint_controls=1000):
    spec=validate_spec(spec);out=Path(out)
    bounded_int(checkpoint_controls,'checkpoint controls',1,24000)
    identity=dict(graph_hash=graph.hash,binding_hash=bindings.hash,backend=backend,versions=runtime_versions(),
                  source=source_identity(),spec_hash=digest(spec))
    if resume:
        manifest=read_json(out/'campaign.json')
        if manifest['identity']!=identity:raise ValueError('Cannot resume with changed data, source, runtime, backend or cases')
        verify_result_files(manifest,out)
        manifest.setdefault('attempts',[]).append(dict(previous_status=manifest['status'],started=time.time()))
    else:
        out.mkdir(parents=True,exist_ok=False)
        source_root=Path(__file__).resolve().parents[2]
        import shutil
        for path in (source_root/'flylab').rglob('*'):
            if path.suffix in ('.py','.metal'):
                target=out/'source'/path.relative_to(source_root);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
        for path in [source_root/'data/circuit.json',source_root/'tools/run_c_campaign.py',
                     *sorted((source_root/'data/releases').glob('*.json'))]:
            target=out/'source'/path.relative_to(source_root);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        write_json(out/'source/bindings.json',bindings.spec)
        write_json(out/'spec.json',spec)
        manifest=dict(schema='flylab.campaign-result.v1',identity=identity,status='RUNNING',cases=[],attempts=[],
                      physicalExecuted=False,started=time.time(),completed_model_s=0.,total_model_s=sum(c['seconds'] for c in spec['cases']))
    manifest['status']='RUNNING';write_json(out/'campaign.json',manifest)
    try:
        completed={r['name'] for r in manifest['cases'] if r['execution_status']=='COMPLETE'}
        for case in spec['cases']:
            if case['name'] in completed:continue
            directory=out/case['name'];directory.mkdir(exist_ok=True)
            progress_path=directory/'progress.json';e=None
            manifest.update(current_case=case['name'],current_model_s=0.);write_json(out/'campaign.json',manifest)
            try:
                rows=[]
                if resume and progress_path.exists():
                    progress=read_json(progress_path)
                    e=CEngine.from_checkpoint(graph,bindings,StateStore.load(directory/progress['checkpoint']),body_factory)
                    chunks=progress['chunks']
                    for chunk in chunks:
                        if file_hash(directory/chunk['file'])!=chunk['sha256']:raise ValueError('Trace chunk hash mismatch')
                        if file_hash(directory/chunk['signals_file'])!=chunk['signals_sha256']:raise ValueError('Signal chunk hash mismatch')
                else:
                    e=CEngine(graph,bindings,mode=case['mode'],seed=case['seed'],config=case.get('config'),
                              world=case.get('world',scene_world(case['scene'])),backend=backend,body_factory=body_factory,
                              initial_pose=case.get('initial_pose'))
                    for event in case.get('interventions', [case['intervention']] if case.get('intervention') else []):e.schedule(event)
                    StateStore.save(directory/'initial',e.checkpoint());chunks=[]
                    write_json(progress_path,dict(checkpoint='initial',chunks=chunks,control_tick=0))
                manifest['physicalExecuted'] |= not e.body.test_double
                total=round(case['seconds']/CONTROL_DT)
                while e.control_tick<total:
                    if cancelled():
                        manifest['status']='CANCELLED';return manifest
                    start=e.control_tick;rows=[trace_sample(e.frame())] if not chunks else []
                    signals=[]
                    end=min(total,start+checkpoint_controls)
                    for _ in range(start,end):
                        if cancelled():break
                        f=e.step(1);rows.append(trace_sample(f))
                        if e.control_tick % 20 == 0:
                            manifest['current_model_s']=e.control_tick*CONTROL_DT
                            write_json(out/'campaign.json',manifest)
                        if e.neural:
                            r=e.neural.readout(bindings.motor_indices)
                            signals.append(dict(tick=e.tick,events=e.neural.last_events,
                                                **{k:v.tolist() for k,v in r.items()}))
                        if f['fault']:raise RuntimeError(f['fault'])
                    if e.control_tick==start:manifest['status']='CANCELLED';return manifest
                    # New attempt suffix retains any uncommitted output from a crash.
                    name=f'chunk-{start:09}-{e.control_tick:09}-{len(manifest["attempts"]):03}'
                    path=directory/(name+'.jsonl')
                    with path.open('xb') as stream:
                        for row in rows:stream.write(canonical(row)+b'\n')
                        stream.flush();os.fsync(stream.fileno())
                    write_json(directory/(name+'-signals.json'),dict(ids=[graph.nodes[int(i)]['id'] for i in bindings.motor_indices],rows=signals))
                    checkpoint='checkpoint-'+name;StateStore.save(directory/checkpoint,e.checkpoint())
                    chunks.append(dict(file=path.name,sha256=file_hash(path),start_control=start,end_control=e.control_tick,
                                       signals_file=name+'-signals.json',signals_sha256=file_hash(directory/(name+'-signals.json'))))
                    write_json(progress_path,dict(checkpoint=checkpoint,chunks=chunks,control_tick=e.control_tick))
                    manifest.update(current_model_s=e.control_tick*CONTROL_DT,
                                    completed_model_s=sum(r['seconds'] for r in manifest['cases'] if r['execution_status']=='COMPLETE'))
                    write_json(out/'campaign.json',manifest)
                trace=[]
                for chunk in chunks:
                    with (directory/chunk['file']).open() as f:
                        import json
                        trace.extend(json.loads(line) for line in f)
                task=case.get('task',{'baseline':'walking','front_obstacle':'obstacle'}.get(case['scene'],case['scene'].split('_')[0]))
                verdict=evaluate(trace,e.world,task,required_seconds=case['seconds'],task_parameters=case.get('task_parameters'))
                result=dict(name=case['name'],seconds=case['seconds'],execution_status='COMPLETE',case=case,
                            physical=not e.body.test_double,provenance=e.provenance(),performance=e.performance(),**verdict)
                write_json(directory/'result.json',result)
                summary=result_summary(result,directory/'result.json',out)
                manifest['cases']=[r for r in manifest['cases'] if r['name']!=case['name']]+[summary]
            except Exception as exc:
                stamp=str(time.time_ns())
                write_json(directory/('failed-'+stamp+'.json'),dict(error=str(exc),trace=rows,
                            diagnostic=fault_report(e) if e else None,restorable=False))
                result=dict(name=case['name'],seconds=e.control_tick*CONTROL_DT if e else 0.,execution_status='FAILED',
                            case=case,technical_status='FAIL',task_status='INCOMPLETE',error=str(exc),
                            physical=e is not None and not e.body.test_double)
                failure_path=directory/('failure-result-'+stamp+'.json')
                write_json(failure_path,result)
                summary=result_summary(result,failure_path,out)
                manifest['cases']=[r for r in manifest['cases'] if r['name']!=case['name']]+[summary]
            finally:
                if e:e.close()
        failed=any(c['execution_status']=='FAILED' for c in manifest['cases'])
        manifest.update(status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',
                        completed_model_s=sum(c['seconds'] for c in manifest['cases'] if c['execution_status']=='COMPLETE'))
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=str(exc));raise
    finally:
        manifest['updated']=time.time();write_json(out/'campaign.json',manifest)
