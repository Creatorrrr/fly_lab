#!/usr/bin/env python3
"""실제 FlyGym만 허용하는 보행/개입 배치 검사.

먼저 verify_physics.py가 PASS해야 조건별 실험을 실행한다.
물리 의존성이 없거나 실제 몸이 아니면 BLOCKED/FAIL을 남기며 대역으로 바꾸지 않는다.
--plan은 실행하지 않은 계획임을 명시한다. 이 파일은 진단까지 표준 라이브러리만 사용한다.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.dependencies import dependency_report

CASES = {
    'baseline': '기준: 평지, 시각 켜짐, 목표 0 rad',
    'turn-positive': '3초에 목표 방향 +π/3 rad',
    'turn-negative': '3초에 목표 방향 -π/3 rad',
    'friction-low': '3초에 마찰 배율 0.5',
    'friction-high': '3초에 마찰 배율 1.5',
    'obstacle': '3초에 몸 주변 5 mm의 유효 위치에 장애물 추가 (전방 우선)',
    'push': '3초에 측면 외력 0.5 BW, 50 ms',
    'cue-off': '3초에 시각 기준 제거',
    'left-suppression': '3초에 설계 PFL3-L 집단 억제',
    'graph-off': '3초에 발췌된 실제 연결의 재귀 전달 차단',
    'motor-off': '3초에 신경–운동 연결 차단; 강제 속도 초기화 없음',
}
ONSET = 3.0


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def build_plan(seeds: list[int], seconds: float) -> list[dict]:
    return [dict(seed=s, case=c, description=d, durationSeconds=seconds,
                 interventionSeconds=None if c == 'baseline' else ONSET)
            for s in seeds for c, d in CASES.items()]


def apply_intervention(engine, case: str) -> dict | None:
    if case == 'baseline':
        return
    settings = {'turn-positive': {'goalAngle': math.pi / 3},
                'turn-negative': {'goalAngle': -math.pi / 3},
                'friction-low': {'friction': .5},
                'friction-high': {'friction': 1.5},
                'graph-off': {'graphGain': 0.},
                'motor-off': {'motorCoupled': False}}
    if case in settings:
        engine.command(dict(type='configure', payload=settings[case]))
    elif case == 'cue-off':
        engine.command(dict(type='cue', payload={'enabled': False}))
    elif case == 'left-suppression':
        ids = [n['id'] for n in engine.circuit.nodes if n['type'] == 'PFL3-L']
        engine.command(dict(type='intervene', payload=dict(ids=ids, kind='suppress')))
    elif case == 'push':
        engine.command(dict(type='push', payload=dict(bw=.5, duration=.05)))
    elif case == 'obstacle':
        # 실험자의 환경 개입: actual pose is NOT sent to the neural controller.
        body = engine.frame()['body']
        p, yaw = body['position'], body['yaw']
        # Near a wall, blindly placing 5 mm ahead can leave the arena.
        # Use the nearest legal direction, keeping the full 5 mm body clearance.
        # The actual offset is recorded; a lateral placement is not called frontal.
        for offset in (0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, -3*math.pi/4, math.pi):
            point=[p[0]+5*math.cos(yaw+offset), .8, p[2]+5*math.sin(yaw+offset)]
            if -23<=point[0]<=23 and -17<=point[2]<=17:
                engine.command(dict(type='place', payload=dict(kind='obstacle', radius=.8, position=point)))
                return dict(position=point, angleOffsetDegrees=math.degrees(offset), distanceMM=5)
        raise RuntimeError('No legal placement among the eight 5 mm obstacle candidates')

    else:
        raise ValueError('Unknown case')


def run_native_case(case: str, seed: int, seconds: float, out: Path) -> int:
    """Production backend only. No fixture/body_factory argument is accepted."""
    import numpy as np
    from flylab import CONTROL_DT
    from flylab.body import FlyGymBody
    from flylab.engine import Engine
    from flylab.sensors import default_world
    result = dict(status='FAIL', case=case, seed=seed, physicalAttempted=False,
                  physicalExecuted=False, physicalSeconds=0., checks=[], trajectory=[], metrics={})
    engine = None
    start = time.perf_counter()
    def check(name, condition):
        result['checks'].append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    try:
        world = default_world()
        world['obstacles'] = []
        world['sources'] = []
        world['foodOn'] = False
        result['physicalAttempted'] = True
        engine = Engine(seed=seed, world=world,
                        config=dict(task='heading', goalAngle=0., sensorNoise=.018))
        check('native FlyGymBody; no fixture', type(engine.body) is FlyGymBody and not engine.body.test_double)
        cp = engine.checkpoint()
        result['initialStateSHA256'] = hashlib.sha256(json.dumps(cp, sort_keys=True).encode()).hexdigest()
        write_json(out / 'initial_checkpoint.json', cp)
        onset_tick = round(ONSET / CONTROL_DT)
        ticks = round(seconds / CONTROL_DT)
        previous = np.asarray(engine.frame()['body']['position'])[[0, 2]]
        planar_distance = 0.
        peak_contact = 0.
        min_upright = 1.
        max_drive_after = 0.
        max_suppressed = 0.
        max_rec_after = 0.
        max_cue_after = 0.
        all_q = []
        for i in range(ticks):
            if i == onset_tick:
                details=apply_intervention(engine, case)
                if details is not None: result['interventionDetails']=details
                result['interventionAppliedAt'] = engine.tick * CONTROL_DT if case != 'baseline' else None
            frame = engine.step(1)
            result['physicalExecuted'] = True
            result['physicalSeconds'] = frame['simTime']
            if engine.tick != i + 1 or frame['physics']['fault']:
                raise RuntimeError('Simulation stopped or fell: ' + str(frame['physics']['fault']))
            if not (np.isfinite(engine.body.d.qpos).all() and np.isfinite(engine.body.d.qvel).all()
                    and np.isfinite(engine.brain.a).all()):
                raise RuntimeError('Non-finite physical/neural state')
            if abs(frame['physics']['physicsTime'] - frame['simTime']) > 1e-5:
                raise RuntimeError('Physics/control time mismatch')
            body, phy = frame['body'], frame['physics']
            position = np.asarray(body['position'])[[0, 2]]
            planar_distance += float(np.linalg.norm(position - previous)); previous = position
            peak_contact = max(peak_contact, max(phy['contactsBW']))
            # Native rotation only for stability measurement, never neural input.
            min_upright = min(min_upright, float(engine.body.pose()[1][2, 2]))
            if i >= onset_tick:
                max_drive_after = max(max_drive_after, max(abs(x) for x in phy['descending']))
                max_rec_after = max(max_rec_after, abs(engine.brain.recNorm))
                max_cue_after = max(max_cue_after, max(abs(x) for x in engine.last_sensors['panorama']))
                max_suppressed = max(max_suppressed, max(abs(engine.brain.a[j]) for j in engine.circuit.pop['PFL3-L']))
            if engine.tick % 20 == 0:
                all_q.append(phy['jointAngles'])
                result['trajectory'].append(dict(time=frame['simTime'], position=body['position'],
                    yaw=body['yaw'], speed=body['speed'], headingEstimate=engine.brain.heading,
                    goal=engine.brain.goal, contactsBW=phy['contactsBW'], descending=phy['descending'],
                    avoidanceActive=bool(engine.brain.turnMemory),
                    recoveryActive=bool(engine.brain.turnMemory and engine.brain.recoveryTime),
                    motorCoupled=bool(engine.config['motorCoupled']),
                    front=min(frame['sensors']['nearRanges'][3:6]), contact=body['contact'],
                    motorCommand=dict(engine.brain.output)))
        result['metrics'].update(horizontalPathMM=planar_distance, peakFootContactBW=peak_contact,
            minUprightCosine=min_upright, maxDriveAfterIntervention=max_drive_after,
            maxRecurrenceAfterIntervention=max_rec_after, maxCueAfterIntervention=max_cue_after,
            maxSuppressedGroupAfterIntervention=max_suppressed,
            planarDisplacementMM=float(np.linalg.norm(previous-np.asarray(cp['body']['last_position'])[:2] * [1,-1])))
        check('completed requested model time', abs(result['physicalSeconds']-seconds)<1e-8)
        check('42 active joint values', bool(all_q) and len(all_q[-1]) == 42)
        check('contact support detected', peak_contact > .01)
        check('upright throughout (>0.15 cosine)', min_upright > .15)
        if case == 'baseline':
            check('planar movement exceeds 0.1 mm', planar_distance > .1)
            check('active joint variation exceeds .001 rad', np.max(np.ptp(all_q,axis=0)) > .001)
        if case == 'motor-off':
            check('motor command disconnected', max_drive_after == 0.)
        elif case == 'left-suppression':
            check('suppressed nodes remain zero', max_suppressed == 0.)
        elif case == 'graph-off':
            check('anatomical recurrent contribution zero', max_rec_after < 1e-12)
        elif case == 'cue-off':
            check('dark cue input zero', max_cue_after == 0.)
        result['status'] = 'PASS'
    except Exception as error:
        result['reason'] = str(error)
        result['traceback'] = traceback.format_exc()
    finally:
        if engine is not None:
            try:
                (out/'neural.csv').write_text(engine.csv(),encoding='utf-8')
                (out/'physics.csv').write_text(engine.csv(True),encoding='utf-8')
                write_json(out/'final_checkpoint.json',engine.checkpoint())
                write_json(out/'replay.json',engine.replay_file())
            except Exception as error:
                result['recordingError'] = str(error); result['status']='FAIL'
            try:
                engine.close()
            except Exception as error:
                result['cleanupError']=str(error); result['status']='FAIL'
        result['wallSeconds']=time.perf_counter()-start
        write_json(out/'result.json',result)
    return 0 if result['status']=='PASS' else 1


def compare_results(results: list[dict]) -> list[dict]:
    comparisons=[]
    for seed in sorted({r['seed'] for r in results}):
        base=next((r for r in results if r['seed']==seed and r['case']=='baseline'),None)
        for run in [r for r in results if r['seed']==seed and r['case']!='baseline']:
            c=dict(seed=seed,case=run['case'],status='NOT_EVALUATED')
            if base and base['status']=='PASS' and run['status']=='PASS':
                # Compare equal model timestamps, not rendering frames/wall time.
                pairs=list(zip(base['trajectory'],run['trajectory']))
                if any(abs(a['time']-b['time'])>1e-8 for a,b in pairs):
                    c.update(status='FAIL',reason='Timestamp mismatch');comparisons.append(c);continue
                def dist(a,b):
                    return math.sqrt(sum((a['position'][i]-b['position'][i])**2 for i in (0,2)))
                pre=[dist(a,b) for a,b in pairs if a['time']<=ONSET+1e-9]
                post=[dist(a,b) for a,b in pairs if a['time']>ONSET+1e-9]
                yaw=[abs(math.atan2(math.sin(a['yaw']-b['yaw']),math.cos(a['yaw']-b['yaw']))) for a,b in pairs if a['time']>ONSET+1e-9]
                same_initial=base['initialStateSHA256']==run['initialStateSHA256']
                c.update(initialStateIdentical=same_initial,maxPreOnsetSeparationMM=max(pre,default=0),
                         maxPostOnsetSeparationMM=max(post,default=0),maxPostOnsetYawDifferenceRad=max(yaw,default=0),
                         effectObserved=max(post,default=0)>.05 or max(yaw,default=0)>.05)
                c['status']='PASS' if same_initial and max(pre,default=0)<1e-6 else 'FAIL'
                # This establishes causal coupling in this engineered model, not biological validity.
                if run['case'] in ('motor-off','left-suppression') and not c['effectObserved']:
                    c.update(status='FAIL',reason='Intervention applied but required behavioral effect not detected')
            comparisons.append(c)
    return comparisons


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('verification/native_campaign'))
    p.add_argument('--seeds',type=int,nargs='+',default=[7,19,43])
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--case-timeout',type=int,default=1800)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--worker-case',choices=list(CASES),help=argparse.SUPPRESS)
    p.add_argument('--worker-seed',type=int,help=argparse.SUPPRESS)
    args=p.parse_args()
    if not math.isfinite(args.seconds) or not 4<=args.seconds<=60:
        p.error('--seconds must be 4..60')
    if not args.seeds or len(args.seeds)>20 or len(set(args.seeds))!=len(args.seeds) or any(not 0<=x<2**32 for x in args.seeds):
        p.error('--seeds must contain 1..20 unique uint32 values')
    if args.case_timeout<1:p.error('--case-timeout must be positive')
    if args.worker_case:
        if args.worker_seed is None or not 0<=args.worker_seed<2**32:p.error('Invalid worker seed')
        args.out.mkdir(parents=True,exist_ok=True)
        return run_native_case(args.worker_case,args.worker_seed,args.seconds,args.out)
    plan=build_plan(args.seeds,args.seconds)
    if args.plan:
        print(json.dumps(dict(status='PLAN_ONLY',physicalExecuted=False,cases=plan),ensure_ascii=False,indent=2))
        return 0
    args.out.mkdir(parents=True,exist_ok=True)
    # An interrupted rerun must not leave an old PASS report visible.
    report=dict(schema='flylab.native-campaign.v1',status='RUNNING',physicalExecuted=False,
                completedCases=0,plannedCases=len(plan),dependencies=dependency_report(),
                python=platform.python_version(),platform=platform.platform(),
                createdAtUTC=datetime.now(timezone.utc).isoformat(),cases=[],comparisons=[],
                interpretation='Software and simulated behavior verification only; not in-vivo validation.')
    report_path=args.out/'report.json'
    write_json(report_path,report)
    gate_path=args.out/'gate.json'
    gate_path.unlink(missing_ok=True)
    try:
        gate=subprocess.run([sys.executable,*(['-S'] if sys.flags.no_site else []),str(ROOT/'tools/verify_physics.py'),'--seconds','3','--output',str(gate_path.resolve())],
                            cwd=ROOT,text=True,capture_output=True,timeout=args.case_timeout)
        (args.out/'gate.log').write_text(gate.stdout+'\n'+gate.stderr,encoding='utf-8')
        gate_result=json.loads(gate_path.read_text()) if gate_path.exists() else {}
        report['gate']=dict(exitCode=gate.returncode,**gate_result)
        report['physicalExecuted']=bool(gate_result.get('physicalExecuted',False))
        if gate.returncode!=0 or gate_result.get('status')!='PASS' or not gate_result.get('physicalValidation'):
            report['status']='BLOCKED' if gate.returncode==2 else 'FAIL'
            report['reason']='Native gate did not pass; no campaign case was run.'
            report['cases']=[dict(**spec,status='BLOCKED',physicalExecuted=False) for spec in plan]
            write_json(report_path,report)
            print(json.dumps(dict(status=report['status'],completedCases=0,plannedCases=len(plan),report=str(report_path)),ensure_ascii=False))
            return 2 if report['status']=='BLOCKED' else 1
        results=[]
        for spec in plan:
            name=f"seed-{spec['seed']}_{spec['case']}";folder=args.out/name;folder.mkdir(exist_ok=True)
            result_path=folder/'result.json';result_path.unlink(missing_ok=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker-case',spec['case'],'--worker-seed',str(spec['seed']),
                 '--seconds',str(args.seconds),'--out',str(folder.resolve())]
            try:
                run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=args.case_timeout)
                (folder/'console.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
                item=json.loads(result_path.read_text()) if result_path.exists() else dict(**spec,status='FAIL',reason='Worker did not produce result')
                if run.returncode!=0:item['status']='FAIL'
            except subprocess.TimeoutExpired:
                item=dict(**spec,status='FAIL',reason='Per-case wall-clock timeout')
                write_json(result_path,item)
            results.append(item)
            report['physicalExecuted'] |= bool(item.get('physicalExecuted',False))
            report['completedCases']+=1
            report['cases']=[{k:v for k,v in r.items() if k!='trajectory'} for r in results]
            write_json(report_path,report)
            print(f"{name}: {item['status']}",flush=True)
        report['comparisons']=compare_results(results)
        report['status']='PASS' if all(r['status']=='PASS' for r in results) and all(r['status']=='PASS' for r in report['comparisons']) else 'FAIL'
        report['passCases']=sum(r['status']=='PASS' for r in results)
    except Exception as error:
        report.update(status='FAIL',reason=str(error),traceback=traceback.format_exc())
    write_json(report_path,report)
    print(json.dumps(dict(status=report['status'],completedCases=report['completedCases'],report=str(report_path)),ensure_ascii=False))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
