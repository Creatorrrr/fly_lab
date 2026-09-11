#!/usr/bin/env python3
"""REAL PHYSICS acceptance gate. Missing dependencies FAIL (exit 2), not skip.
Outputs are local and self-contained; it never silently uses the test fixture.
"""
from pathlib import Path
import sys,json,time,argparse,platform,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.dependencies import dependency_report
from flylab import CONTROL_DT

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=3);p.add_argument('--output',default='physics_verification.json');p.add_argument('--render',action='store_true');args=p.parse_args()
    if not .5<=args.seconds<=300:p.error('--seconds must be .5..300')
    result=dict(schema='flylab.physics-gate.v1',status='BLOCKED',physicalValidation=False,physicalAttempted=False,physicalExecuted=False,platform=platform.platform(),python=platform.python_version(),dependencies=dependency_report(),checks=[],metrics={})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);engine=None;clone=None
    def check(name,ok):
        result['checks'].append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    try:
        if not result['dependencies']['ready']:
            result['reason']='Required FlyGym 2.1.0 / MuJoCo 3.9.x are not installed';return 2
        # Imports are intentionally delayed until diagnostics can be persisted.
        import numpy as np
        from flylab.body import FlyGymBody
        from flylab.engine import Engine
        result['physicalAttempted']=True
        engine=Engine(config=dict(task='heading',goalAngle=0.))
        result['physicalExecuted']=True
        check('real backend, not fixture',isinstance(engine.body,FlyGymBody) and not engine.body.test_double)
        start=engine.body.frame()[0]['position'];states=[];t=time.perf_counter()
        for _ in range(round(args.seconds/CONTROL_DT)):
            frame=engine.step(1);check_fault=frame['physics']['fault']
            if check_fault:raise RuntimeError('Physical fall/instability: '+str(check_fault))
            if engine.tick%20==0:states.append(frame['physics'])
        elapsed=time.perf_counter()-t
        result['metrics'].update(modelSeconds=engine.tick*CONTROL_DT,wallSeconds=elapsed,modelTimePerWallTime=engine.tick*CONTROL_DT/elapsed,
           travelMM=engine.body.travel,netDisplacementMM=float(np.linalg.norm(np.asarray(engine.body.frame()[0]['position'])-start)),
           horizontalDisplacementMM=float(np.linalg.norm((np.asarray(engine.body.frame()[0]['position'])-start)[[0,2]])))
        check('integration and neural clocks',abs(engine.body.frame()[1]['physicsTime']-engine.tick*CONTROL_DT)<1e-5)
        check('finite neural states',np.isfinite(engine.brain.a).all())
        check('42 active joint telemetry',len(states[-1]['jointAngles'])==42)
        check('joint angles change',np.max(np.ptp(np.asarray([s['jointAngles'] for s in states]),axis=0))>.001)
        check('ground support force detected',max(max(s['contactsBW']) for s in states)>.01)
        check('nontrivial movement',engine.body.travel>.1)
        check('horizontal displacement, not just vertical settling',result['metrics']['horizontalDisplacementMM']>.1)
        cp=engine.checkpoint();clone=Engine.from_checkpoint(json.loads(json.dumps(cp)))
        engine.step(20);clone.step(20)
        check('continuation has no physics fault',not engine.body.fault and not clone.body.fault)
        delta=float(np.max(np.abs(np.asarray(engine.body.snapshot()['state'])-clone.body.snapshot()['state'])))
        result['metrics']['restoredIntegrationMaxError']=delta
        check('checkpoint continuation 100ms',delta<1e-7)
        # Virtual perturbation and joint target readback, not an in-vivo experiment.
        engine.command(dict(type='push',payload=dict(bw=.2,duration=.02)));engine.step(10)
        check('explicit force path remains finite',np.isfinite(engine.body.d.qpos).all())
        check('no fault after virtual push',not engine.body.fault)
        engine.command(dict(type='configure',payload=dict(motorCoupled=False)));engine.step(10)
        check('motor disconnect parks drive',np.max(np.abs(engine.body.descending))==0)
        check('no fault after motor disconnect',not engine.body.fault)
        if args.render:
            from PIL import Image
            image=engine.body.preview();image_path=out.with_suffix('.png');Image.fromarray(image).save(image_path);result['nativeImage']=str(image_path)
        result['status']='PASS';result['physicalValidation']=True;return 0
    except Exception as e:
        result['status']='FAIL';result['reason']=str(e);result['traceback']=traceback.format_exc();return 1
    finally:
        for resource in (engine,clone):
            if resource:
                try:resource.close()
                except Exception as cleanup_error:
                    result.setdefault('cleanupErrors',[]).append(str(cleanup_error))
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        # The JSON stream also works in legacy Windows console encodings.
        print(json.dumps(result,ensure_ascii=True,indent=2))
if __name__=='__main__':raise SystemExit(main())
