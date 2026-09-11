"""Sequential same-origin controls and streaming replay of scientific records."""
from pathlib import Path
import copy
import json
import math
import time
import numpy as np
from .engine import CEngine
from .storage import StateStore
from .integrity import finite, bounded_int, file_hash, read_json, write_json
from .workbench import ENVIRONMENT_COMMANDS
from .behavior import CRITERIA, trace_sample, behavior_metrics


def paired(original, path, intervention, *, seconds=.5, onset=.2, ids=None):
    seconds = finite(seconds, 'seconds', .01, 10)
    onset = finite(onset, 'onset', 0, seconds-.005)
    for value in (seconds, onset):
        if abs(value/.005-round(value/.005)) > 1e-8: raise ValueError('Use integer control periods')
    original._validate_intervention(dict(intervention, at_tick=original.tick+round(onset/original.parameters.dt),
                                         duration_controls=round((seconds-onset)/.005)))
    if ids is not None and not len(original.graph.resolve(ids)): raise ValueError('Fixed recording cohort required')
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    initial = original.checkpoint()
    initial_identity = StateStore.save(path/'initial_checkpoint', initial)
    runs = []
    for variant in ('control', 'intervention'):
        e = CEngine.from_checkpoint(original.graph, original.bindings, initial, original.body_factory)
        try:
            e.stopped = False
            e.start_recording(path/variant, ids)
            if variant == 'intervention':
                spec = dict(intervention, at_tick=e.tick+round(onset/e.parameters.dt),
                            duration_controls=round((seconds-onset)/.005))
                e.schedule(spec)
            start = e.body.frame()[0]; trace = [trace_sample(e.frame())]; began = time.perf_counter()
            remaining = round(seconds/.005)
            while remaining:
                batch = 1  # Behavior metrics sample every physical control interval.
                before = e.control_tick
                f = e.step(batch)
                advanced = e.control_tick-before
                remaining -= advanced
                sample = trace_sample(f)
                sample['contact'] = bool(e.body.nonfoot_contact())
                trace.append(sample)
                if f['fault'] or advanced != batch: break
            end = e.body.frame()[0]
            run = dict(variant=variant, status='COMPLETE' if not remaining and not e.fault else 'FAILED',
                       physicalExecuted=not e.body.test_double and e.control_tick>initial['control_tick'],
                       model_seconds=(e.control_tick-initial['control_tick'])*.005, wall_seconds=time.perf_counter()-began,
                       horizontal_displacement_mm=float(np.linalg.norm(np.asarray(end['position'])[[0,2]]-np.asarray(start['position'])[[0,2]])),
                       yaw_change_rad=math.atan2(math.sin(end['yaw']-start['yaw']), math.cos(end['yaw']-start['yaw'])),
                       initial_state_hash=initial_identity['state_hash'], fault=e.fault, trace=trace,
                       behavior=behavior_metrics(trace, e.world, motion_expected=e.motion_expected))
            if not e.fault: StateStore.save(path/variant/'final_checkpoint', e.checkpoint())
            runs.append(run)
        finally: e.close()
    a, b = runs
    start_s = initial['control_tick']*.005
    before_a = [r for r in a['trace'] if r['simTime']<=start_s+onset+1e-10]
    before_b = [r for r in b['trace'] if r['simTime']<=start_s+onset+1e-10]
    matched = len(before_a)==len(before_b) and all(x['position']==y['position'] and x['yaw']==y['yaw'] for x,y in zip(before_a,before_b))
    matched = matched and all(x['sensors']==y['sensors'] and x['motor_rates_Hz']==y['motor_rates_Hz'] and
                             x['command']==y['command'] for x,y in zip(before_a,before_b))
    separation = None
    if len(a['trace'])==len(b['trace']) and a['trace']:
        separation = float(np.mean([np.linalg.norm(np.asarray(x['position'])-y['position']) for x,y in zip(a['trace'],b['trace'])]))
    report = dict(schema='flylab.paired.v3', status='COMPLETE' if matched and all(r['status']=='COMPLETE' for r in runs) else 'FAILED',
                  intervention=intervention, seconds=seconds, onset=onset, initial_state_hash=initial_identity['state_hash'],
                  pre_intervention_matched=matched, mean_path_separation_mm=separation,
                  physicalExecuted=all(r['physicalExecuted'] for r in runs),
                  biologicalValidation=False, provenance=original.provenance(), runs=runs,
                  criteria=CRITERIA, task_status='NOT_EVALUATED',
                  technical_status='PASS' if matched and all(r['status']=='COMPLETE' and not r['behavior']['observation_gaps'] for r in runs) else 'FAILED')
    write_json(path/'comparison.json', report)
    preview = []
    for run in runs:
        points = []; signed = 0.; trace = run['trace']; stride = max(1,math.ceil(len(trace)/200))
        for i,row in enumerate(trace):
            if i:
                prev=trace[i-1];signed+=(row['position'][0]-prev['position'][0])*prev['forward_axis'][0]+(row['position'][2]-prev['position'][2])*prev['forward_axis'][2]
            if i%stride==0 or i==len(trace)-1:
                points.append(dict(model_s=row['simTime']-start_s,signed_forward_mm=signed,
                                   command_mm_s=row['command']['u_final']['forwardSpeed'],
                                   forward_rate_Hz=row['motor_rates_Hz'].get('forward',0.)))
        preview.append(dict(variant=run['variant'],points=points))
    return dict(schema=report['schema'], status=report['status'], path=str(path.resolve()),
                pre_intervention_matched=matched, mean_path_separation_mm=separation,
                physicalExecuted=report['physicalExecuted'], biologicalValidation=False,
                technical_status=report['technical_status'], task_status=report['task_status'], criteria=CRITERIA, preview=preview,
                runs=[{k:v for k,v in r.items() if k!='trace'} for r in runs])


def paired_campaign(original, path, intervention, *, seconds=.5, onset=.2, ids=None,
                    repeats=1, origin='current', seed=None):
    repeats = bounded_int(repeats, 'repeats', 1, 5)
    seconds = finite(seconds, 'seconds', .01, 10)
    if 2*seconds*repeats>20: raise ValueError('One comparison request is limited to 20 total model seconds')
    if origin not in ('current','fresh'): raise ValueError('Unknown comparison origin')
    if origin=='fresh': seed=bounded_int(seed, 'first seed', 0, 2**32-repeats)
    elif seed is not None: raise ValueError('A current checkpoint preserves its original RNG; use fresh for a new seed')
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    reports = []
    try:
        for i in range(repeats):
            source = original if origin=='current' else CEngine(original.graph, original.bindings,
                mode=original.mode, seed=seed+i, world=original.world, config=original.config,
                backend=original.neural.backend if original.neural else 'exp_lif_cpu_reference', parameters=original.parameters,
                body_factory=original.body_factory, motion_expected=original.motion_expected,
                metabolism=original.metabolism.snapshot()["parameters"] if original.metabolism else None)
            try: reports.append(dict(repeat=i+1, seed=source.seed, **paired(source,path/f'repeat-{i+1}',intervention,seconds=seconds,onset=onset,ids=ids)))
            finally:
                if source is not original: source.close()
        report = dict(schema='flylab.paired_campaign.v1', status='COMPLETE' if all(r['technical_status']=='PASS' for r in reports) else 'FAILED',
                      origin=origin, repeat_policy='identical-checkpoint determinism' if origin=='current' else 'consecutive seeds',
                      mode=original.mode, binding_hash=original.bindings.hash, path=str(path.resolve()),
                      comparisons=reports, task_status='NOT_EVALUATED', biologicalValidation=False)
    except Exception as exc:
        write_json(path/'campaign.json', dict(status='FAILED', error=str(exc), comparisons=reports))
        raise
    write_json(path/'campaign.json', report)
    return report


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
                if kind not in ENVIRONMENT_COMMANDS | {'intervention_scheduled','cancel_intervention','release_all','stop','resume','configure','cue','food','push','body_actuation'}:
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
