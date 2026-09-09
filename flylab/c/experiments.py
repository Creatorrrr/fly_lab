"""Sequential same-origin controls and streaming replay of scientific records."""
from pathlib import Path
import copy
import json
import math
import time
import numpy as np
from .engine import CEngine
from .storage import StateStore
from .integrity import finite, file_hash, read_json, write_json


def paired(original, path, intervention, *, seconds=.5, onset=.2):
    seconds = finite(seconds, 'seconds', .01, 10)
    onset = finite(onset, 'onset', 0, seconds-.005)
    for value in (seconds, onset):
        if abs(value/.005-round(value/.005)) > 1e-8: raise ValueError('Use integer control periods')
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    initial = original.checkpoint()
    initial_identity = StateStore.save(path/'initial_checkpoint', initial)
    runs = []
    for variant in ('control', 'intervention'):
        e = CEngine.from_checkpoint(original.graph, original.bindings, initial, original.body_factory)
        try:
            e.stopped = False
            e.start_recording(path/variant)
            if variant == 'intervention':
                spec = dict(intervention, at_tick=e.tick+round(onset/e.parameters.dt),
                            duration_controls=round((seconds-onset)/.005))
                e.schedule(spec)
            start = e.body.frame()[0]; trace = []; began = time.perf_counter()
            remaining = round(seconds/.005)
            while remaining:
                batch = min(10, remaining)
                before = e.control_tick
                f = e.step(batch)
                advanced = e.control_tick-before
                remaining -= advanced
                trace.append(dict(simTime=f['simTime'], tick=f['tick'], position=f['body']['position'],
                                  yaw=f['body']['yaw'], command=f['command'], fault=f['fault']))
                if f['fault'] or advanced != batch: break
            end = e.body.frame()[0]
            run = dict(variant=variant, status='COMPLETE' if not remaining and not e.fault else 'FAILED',
                       physicalExecuted=not e.body.test_double and e.control_tick>initial['control_tick'],
                       model_seconds=(e.control_tick-initial['control_tick'])*.005, wall_seconds=time.perf_counter()-began,
                       horizontal_displacement_mm=float(np.linalg.norm(np.asarray(end['position'])[[0,2]]-np.asarray(start['position'])[[0,2]])),
                       yaw_change_rad=math.atan2(math.sin(end['yaw']-start['yaw']), math.cos(end['yaw']-start['yaw'])),
                       initial_state_hash=initial_identity['state_hash'], fault=e.fault, trace=trace)
            if not e.fault: StateStore.save(path/variant/'final_checkpoint', e.checkpoint())
            runs.append(run)
        finally: e.close()
    a, b = runs
    start_s = initial['control_tick']*.005
    before_a = [r for r in a['trace'] if r['simTime']<=start_s+onset+1e-10]
    before_b = [r for r in b['trace'] if r['simTime']<=start_s+onset+1e-10]
    matched = len(before_a)==len(before_b) and all(x['position']==y['position'] and x['yaw']==y['yaw'] for x,y in zip(before_a,before_b))
    separation = None
    if len(a['trace'])==len(b['trace']) and a['trace']:
        separation = float(np.mean([np.linalg.norm(np.asarray(x['position'])-y['position']) for x,y in zip(a['trace'],b['trace'])]))
    report = dict(schema='flylab.paired.v3', status='COMPLETE' if matched and all(r['status']=='COMPLETE' for r in runs) else 'FAILED',
                  intervention=intervention, seconds=seconds, onset=onset, initial_state_hash=initial_identity['state_hash'],
                  pre_intervention_matched=matched, mean_path_separation_mm=separation,
                  physicalExecuted=all(r['physicalExecuted'] for r in runs),
                  biologicalValidation=False, provenance=original.provenance(), runs=runs)
    write_json(path/'comparison.json', report)
    return dict(schema=report['schema'], status=report['status'], path=str(path.resolve()),
                pre_intervention_matched=matched, mean_path_separation_mm=separation,
                physicalExecuted=report['physicalExecuted'], biologicalValidation=False,
                runs=[{k:v for k,v in r.items() if k!='trace'} for r in runs])


def replay_recording(graph, bindings, path, body_factory):
    path = Path(path)
    manifest = read_json(path/'manifest.json')
    if manifest.get('schema')!='flylab.recording.v3' or manifest.get('status')!='COMPLETE':
        raise ValueError('A complete recording is required for replay')
    if file_hash(path/'events.jsonl')!=manifest.get('events_sha256') or file_hash(path/'body.jsonl')!=manifest.get('body_sha256'):
        raise ValueError('Recording hash mismatch')
    for chunk in manifest.get('chunks', []):
        from .integrity import checked_name
        for name, sha in ((chunk['file'],chunk['sha256']), (chunk['ticks_file'],chunk['ticks_sha256'])):
            checked_name(name)
            if file_hash(path/name)!=sha: raise ValueError('Recording chunk hash mismatch')
    e = CEngine.from_checkpoint(graph, bindings, StateStore.load(path/'initial_checkpoint'), body_factory)
    e.stopped = False
    try:
        end = manifest['end_tick']
        def advance_to(tick):
            # Bound work batches, not total replay duration. Every tick executes.
            while e.tick < tick:
                before = e.tick
                e.step(min(1000, (tick-e.tick)//e.substeps))
                if e.tick == before or e.fault:
                    raise RuntimeError('Replay stopped before requested model time')
        with (path/'events.jsonl').open() as f:
            for line in f:
                event = json.loads(line)
                kind = event['kind']
                if kind not in ('intervention_scheduled','release_all','stop','resume','configure','cue','food','push','place','clear_added'):
                    continue
                at = event['tick']
                if type(at) is not int or at<e.tick or (at-e.tick)%e.substeps or at>end:
                    raise ValueError('Replay command clock mismatch')
                advance_to(at)
                if e.tick != at or e.fault: raise RuntimeError('Replay stopped before command boundary')
                details = event['details']
                if kind=='intervention_scheduled':
                    data = {k:v for k,v in details.items() if k in ('kind','ids','channels','edges','amplitude_mV','at_tick')}
                    data['duration_controls'] = (details['expires_tick']-details['at_tick'])//e.substeps
                    e.schedule(data)
                else: e.command(kind,details)
        if type(end) is not int or end<e.tick or (end-e.tick)%e.substeps:
            raise ValueError('Replay end clock mismatch')
        advance_to(end)
        if e.tick!=end or e.fault: raise RuntimeError('Incomplete replay')
        return e
    except Exception:
        e.close()
        raise
