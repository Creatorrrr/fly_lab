"""Bounded, transactional environment edits and intervention presentation.

These operations run in the same worker and at the same control boundary as
the integrators. They never advance the simulation or rewrite delayed spikes.
"""
import copy
import math
from ..engine import Engine as BEngine
from ..sensors import validate_world
from .integrity import bounded_int, finite
from .inputs import validate_schedule

ENVIRONMENT_COMMANDS = {'place', 'update_object', 'delete_object', 'clear_added',
                        'load_environment', 'undo_environment'}
MAX_UNDO = 32
MAX_INTERVENTION_HISTORY = 256


def checked_world(world):
    validate_world(world)
    if set(world) != {'bounds', 'cueOn', 'cueAngle', 'foodOn', 'sources', 'obstacles'}:
        raise ValueError('Unknown environment field')
    for group in ('sources', 'obstacles'):
        fields = {'id', 'p', 'r'} if group == 'obstacles' else {'id', 'p', 'kind', 'strength'}
        for obj in world[group]:
            if set(obj) != fields: raise ValueError('Unknown object field')
            if group == 'obstacles':
                if obj['p'][1] != obj['r']: raise ValueError('Obstacles must rest on the floor')
                if abs(obj['p'][0])+obj['r'] > 24 or abs(obj['p'][2])+obj['r'] > 18:
                    raise ValueError('Obstacle intersects an arena wall')
    return world


def edit_environment(engine, kind, payload):
    allowed = {
        'place': {'kind', 'position', 'strength', 'radius'},
        'update_object': {'id', 'position', 'strength', 'radius'},
        'delete_object': {'id'}, 'clear_added': set(),
        'load_environment': {'world'}, 'undo_environment': set(),
    }
    if set(payload)-allowed[kind]: raise ValueError('Unknown environment command field')
    before = copy.deepcopy(engine.world)
    world = copy.deepcopy(before)
    serial = engine.object_serial
    if kind == 'undo_environment':
        if not engine.environment_history: raise ValueError('No environment edit to undo')
        world = copy.deepcopy(engine.environment_history[-1])
    elif kind == 'load_environment':
        world = copy.deepcopy(payload.get('world'))
    elif kind == 'clear_added':
        for group in ('sources', 'obstacles'):
            world[group] = [o for o in world[group] if not o['id'].startswith('user-')]
    else:
        if kind == 'place':
            category = payload.get('kind')
            if category not in ('food', 'hazard', 'obstacle'): raise ValueError('Unknown object kind')
            existing = {o['id'] for o in world['sources']+world['obstacles']}
            while True:
                serial = bounded_int(serial+1, 'object serial', 1, 1000000)
                if 'user-'+str(serial) not in existing: break
            obj = dict(id='user-'+str(serial), p=[0., .7, 0.])
            if category == 'obstacle': obj['r'] = .8
            else: obj.update(kind=category, strength=1.2)
            world['obstacles' if category == 'obstacle' else 'sources'].append(obj)
        else:
            found = [(g, o) for g in ('sources', 'obstacles') for o in world[g] if o['id'] == payload.get('id')]
            if not found: raise ValueError('Unknown environment object ID')
            group, obj = found[0]
            if kind == 'delete_object': world[group].remove(obj)
        if kind != 'delete_object':
            obstacle = 'r' in obj
            if (obstacle and 'strength' in payload) or (not obstacle and 'radius' in payload):
                raise ValueError('Strength is for odor sources; radius is for obstacles')
            p = payload.get('position', None if kind == 'place' else obj['p'])
            if not isinstance(p, list) or len(p) != 3: raise ValueError('Position [x, height, z] required')
            obj['p'] = [finite(v, 'position', -100, 100) for v in p]
            if obstacle:
                obj['r'] = finite(payload.get('radius', obj['r']), 'radius', .2, 3)
                obj['p'][1] = obj['r']
            else: obj['strength'] = finite(payload.get('strength', obj['strength']), 'strength', 0, 5)
    checked_world(world)
    old_objects = {o['id']: o for o in before['obstacles']}
    position = engine.body.frame()[0]['position']
    for obj in world['obstacles']:
        if old_objects.get(obj['id']) == obj: continue
        if math.hypot(obj['p'][0]-position[0], obj['p'][2]-position[2]) < obj['r']+3:
            raise ValueError('Place obstacle at least 3 mm clear of body in the horizontal plane')
        for other in world['obstacles']:
            if other['id'] != obj['id'] and math.dist(obj['p'], other['p']) < obj['r']+other['r']:
                raise ValueError('Obstacle overlaps another obstacle')
    # _set_world compiles/restores a candidate before swapping the live body.
    BEngine._set_world(engine, world)
    engine.object_serial = serial
    if kind == 'undo_environment': engine.environment_history.pop()
    elif before != world:
        engine.environment_history.append(before)
        engine.environment_history = engine.environment_history[-MAX_UNDO:]
    engine.environment_updated_tick = engine.tick


def terminal_intervention(engine, item, status, tick):
    engine.intervention_history.append(dict(copy.deepcopy(item), status=status, ended_tick=tick))
    engine.intervention_history = engine.intervention_history[-MAX_INTERVENTION_HISTORY:]


def intervention_rows(engine, offset=0, limit=50):
    offset = bounded_int(offset, 'offset', 0, 1000000)
    limit = bounded_int(limit, 'limit', 1, 200)
    rows = [dict(e, status='pending') for e in engine.pending]
    rows += [dict(e, status='active' if engine.tick < e['expires_tick'] else 'expired',
                  **({} if engine.tick < e['expires_tick'] else {'ended_tick': e['expires_tick']})) for e in engine.active]
    rows += engine.intervention_history
    rows.sort(key=lambda e: e['serial'], reverse=True)
    result = []
    for row in rows[offset:offset+limit]:
        row = copy.deepcopy(row)
        row.pop('indices', None)
        if 'edges' in row:
            row['edge_count'] = len(row['edges']); row.pop('edges')
        row['remaining_model_s'] = max(0, row['expires_tick']-engine.tick)*engine.parameters.dt if row['status'] in ('pending','active') else 0.
        result.append(row)
    return dict(items=result, total=len(rows), offset=offset, limit=limit,
                history_limit=MAX_INTERVENTION_HISTORY)


def cancel_intervention(engine, serial):
    serial = bounded_int(serial, 'intervention serial', 1)
    found = [e for e in engine.pending+engine.active if e['serial'] == serial]
    if not found or found[0]['expires_tick'] <= engine.tick:
        raise ValueError('Only pending or active interventions can be cancelled')
    item = found[0]
    validate_schedule([e for e in engine.pending+engine.active if e['serial'] != serial], engine.tick)
    engine.pending = [e for e in engine.pending if e['serial'] != serial]
    engine.active = [e for e in engine.active if e['serial'] != serial]
    terminal_intervention(engine, item, 'cancelled', engine.tick)
    # Remove only future spike/output masks. Do not touch the synaptic queue.
    if engine.neural:
        def targets(kind, field):
            return sorted({i for e in engine.active if e['kind'] == kind and e['expires_tick'] > engine.tick for i in e[field]})
        engine.neural.set_interventions(targets('suppress_spiking', 'indices'),
                                       targets('mute_outgoing', 'indices'), targets('mute_edges', 'edges'))


def restore_workbench(engine, state):
    history = copy.deepcopy(state.get('environment_history', []))
    if not isinstance(history, list) or len(history) > MAX_UNDO: raise ValueError('Invalid environment history')
    for world in history: checked_world(world)
    engine.environment_history = history
    tick = state.get('environment_updated_tick')
    if tick is not None:
        bounded_int(tick, 'environment updated tick', 0, engine.tick)
        if tick % engine.substeps: raise ValueError('Environment update clock mismatch')
    engine.environment_updated_tick = tick
    history = copy.deepcopy(state.get('intervention_history', []))
    if not isinstance(history, list) or len(history) > MAX_INTERVENTION_HISTORY: raise ValueError('Invalid intervention history')
    serials = {e['serial'] for e in engine.pending+engine.active}
    for item in history:
        serial = bounded_int(item.get('serial'), 'history serial', 1, engine.event_serial)
        if serial in serials: raise ValueError('Duplicate history serial')
        serials.add(serial)
        at = bounded_int(item.get('at_tick'), 'history start')
        end = bounded_int(item.get('expires_tick'), 'history expiry', at+engine.substeps, at+720000*engine.substeps)
        ended = bounded_int(item.get('ended_tick'), 'history end', 0, engine.tick)
        if any(t % engine.substeps for t in (at, end, ended)): raise ValueError('History clock mismatch')
        if item.get('status') not in ('cancelled', 'expired') or ended > end or (item['status'] == 'expired' and ended != end):
            raise ValueError('Invalid terminal intervention')
        spec = {k: v for k, v in item.items() if k in ('kind','ids','edges','channels','amplitude_mV')}
        spec.update(at_tick=engine.tick, duration_controls=(end-at)//engine.substeps)
        if engine._validate_intervention(spec).get('indices') != item.get('indices'):
            raise ValueError('History target mismatch')
    engine.intervention_history = history
    engine.record_indices = engine.graph.resolve(state.get('record_cohort_ids',
        [engine.graph.nodes[int(i)]['id'] for i in engine.bindings.motor_indices]))
