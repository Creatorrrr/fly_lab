#!/usr/bin/env python3
"""Real-physics wall escape gate. No fixture, no pose correction while walking.

Default-world trials cover four seeds for 20 model seconds. Four additional
trials start facing each arena wall at 4 mm clearance and run for 8 seconds.
Initial placement is an experimental setup, never a neural input or runtime fix.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.dependencies import dependency_report
from tools.navigation_checks import NavigationMonitor


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def trial(case, seed, out):
    import numpy as np
    from flylab import CONTROL_DT
    from flylab.engine import Engine
    from flylab.sensors import default_world
    from flylab.common import to_physics
    result = dict(case=case, seed=seed, status='RUNNING', physicalExecuted=False,
                  checks=[], trajectory=[], metrics={})
    engine = clone = None
    monitor = NavigationMonitor()
    start = time.monotonic()

    def check(name, condition):
        result['checks'].append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    try:
        world = default_world()
        if case != 'explore':
            world.update(obstacles=[], sources=[], foodOn=False)
        engine = Engine(seed=seed, world=world)
        b = engine.body
        result['physicalExecuted'] = True
        check('real FlyGym backend', not b.test_double)
        check('all body IDs resolve', (b.body_ids >= 0).all())
        head_id = int(b.body_ids[b.body_indices['c_head']])
        check('head has its own compiled frame', head_id != b.thorax)
        if case != 'explore':
            # Rigidly place the whole articulated fly ONCE at trial initialization.
            yaw, target = {'east': (0, [20, 0]), 'west': (math.pi, [-20, 0]),
                           'south': (math.pi/2, [0, 14]),
                           'north': (-math.pi/2, [0, -14])}[case]
            free = next(j for j in range(b.m.njnt)
                        if b.m.jnt_type[j] == b.mj.mjtJoint.mjJNT_FREE)
            adr = b.m.jnt_qposadr[free]
            q = np.array([math.cos(yaw/2), 0, 0, -math.sin(yaw/2)])
            orientation = np.zeros(4)
            b.mj.mju_mulQuat(orientation, q, b.d.qpos[adr+3:adr+7].copy())
            b.d.qpos[adr+3:adr+7] = orientation
            b.mj.mj_forward(b.m, b.d)
            p = b.pose()[0]
            b.d.qpos[adr:adr+3] += to_physics([target[0], p[2], target[1]]) - p
            b.d.qvel[:] = 0
            b.mj.mj_forward(b.m, b.d)
            b.last_position = b.pose()[0].copy()
            engine.last_sensors = engine.sensors.observe(b, world, CONTROL_DT, engine.config)
            engine.history.clear()
            engine.initial = engine.checkpoint()
        duration = 20 if case == 'explore' else 8
        max_head_error = 0.
        check_tick = None
        for _ in range(round(duration / CONTROL_DT)):
            f = engine.step(1)
            if f['physics']['fault']:
                raise AssertionError(f['physics']['fault'])
            if not np.isfinite(b.d.qpos).all():
                raise AssertionError('non-finite physical state')
            # Compare rendered head telemetry against its actual compiled body.
            head = to_physics(f['body']['segments']['c_head'])
            max_head_error = max(max_head_error, float(np.linalg.norm(head-b.d.xpos[head_id])))
            if seed == 42 and case == 'explore' and clone is None and b.walk_ticks > 20 and engine.brain.turnMemory:
                clone = Engine.from_checkpoint(engine.checkpoint())
                clone.step(20)
                check_tick = engine.tick + 20
            if check_tick == engine.tick:
                error = float(np.max(np.abs(np.asarray(b.snapshot()['state']) - clone.body.snapshot()['state'])))
                result['metrics']['activeTurnRestoreMaxError'] = error
                check('checkpoint continuation during active avoidance', error < 1e-7)
            if engine.tick % 20 == 0:
                row = dict(t=f['simTime'], p=f['body']['position'], yaw=f['body']['yaw'],
                           front=min(f['sensors']['nearRanges'][3:6]), contact=f['body']['contact'],
                           avoiding=f['neural']['avoidanceActive'],
                           recovering=f['neural']['recoveryActive'],
                           motorCoupled=f['config']['motorCoupled'], motionExpected=True,
                           forwardCommand=f['neural']['output']['forwardSpeed'],
                           yawCommand=f['neural']['output']['yawRate'], drive=f['physics']['descending'])
                result['trajectory'].append(row)
                monitor.add(row)
                audit=monitor.result()
                if audit['stationaryWindows']:
                    bad=audit['stationaryWindows'][-1]
                    raise AssertionError('Stationary enabled motion for 3 seconds within a 1 mm envelope: '+str(bad))
                if audit['samplingGaps']:
                    raise AssertionError('Navigation sampling gap; continuous monitoring is not established')
        rows = result['trajectory']
        positions = np.array([r['p'] for r in rows])[:, [0, 2]]
        result['metrics'].update(modelSeconds=engine.tick*CONTROL_DT,
                                 headPositionMaxErrorMM=max_head_error,
                                 planarSpanMM=float(np.linalg.norm(np.ptp(positions, axis=0))),
                                 collisionEntries=b.collisions,
                                 avoidanceSamples=sum(r['avoiding'] for r in rows))
        check('head telemetry matches actual body', max_head_error < 1e-10)
        check('completed requested time without fault or prolonged blocking', engine.tick*CONTROL_DT >= duration-1e-8)
        check('stays inside arena', (np.abs(positions[:, 0]) < 24).all() and (np.abs(positions[:, 1]) < 18).all())
        check('avoidance was exercised', any(r['avoiding'] for r in rows))
        check('gap-free enabled-motion stationary monitor', monitor.result()['status']=='PASS')
        check('measured forward walking resumes after avoidance', monitor.result()['walkingResumed'])
        if case == 'explore':
            check('explores a spatial span > 10 mm', result['metrics']['planarSpanMM'] > 10)
            if seed == 42:
                check('active-turn restoration was exercised', 'activeTurnRestoreMaxError' in result['metrics'])
        else:
            axis = 0 if case in ('east','west') else 1
            sign = 1 if case in ('east','south') else -1
            wall = 24 if axis == 0 else 18
            gaps = wall-sign*positions[:, axis]
            result['metrics']['maximumWallClearanceMM'] = float(max(gaps))
            check('moves at least 2 mm away from starting wall', max(gaps) > 6)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', reason=str(error), traceback=traceback.format_exc())
    finally:
        result['navigationAudit'] = monitor.result()
        if engine:
            write(out/'final_checkpoint.json', engine.checkpoint())
            engine.close()
        if clone:
            clone.close()
        result['wallSeconds'] = time.monotonic()-start
        write(out/'result.json', result)
    return 0 if result['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'verification/navigation')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--worker', choices=['explore','east','west','south','north'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error('--jobs must be 1..8')
    deps = dependency_report()
    if not deps['ready']:
        write(args.out/'report.json', dict(status='BLOCKED', physicalExecuted=False, dependencies=deps))
        return 2
    if args.worker:
        args.out.mkdir(parents=True, exist_ok=True)
        return trial(args.worker, args.seed, args.out)
    cases = [('explore', seed) for seed in [42,7,19,43]] + [(case,42) for case in ['east','west','south','north']]
    report = dict(status='RUNNING', dependencies=deps, cases=[])
    write(args.out/'report.json', report)
    def run(case):
        name, seed = case
        folder = args.out/f'{name}-{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'console.log').open('w') as log:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name,
                                    '--seed', str(seed), '--out', str(folder.resolve())],
                                   stdout=log, stderr=log, timeout=1800)
        path = folder/'result.json'
        item = json.loads(path.read_text()) if path.exists() else dict(case=name, seed=seed, status='FAIL', reason='No result')
        if child.returncode:
            item['status'] = 'FAIL'
        return {k:v for k,v in item.items() if k != 'trajectory'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, case) for case in cases]):
            try:
                item = future.result()
            except Exception as error:
                item = dict(status='FAIL', reason=str(error))
            report['cases'].append(item)
            write(args.out/'report.json', report)
            print(json.dumps(item), flush=True)
    report['status'] = 'PASS' if len(report['cases']) == len(cases) and all(c['status']=='PASS' for c in report['cases']) else 'FAIL'
    write(args.out/'report.json', report)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
