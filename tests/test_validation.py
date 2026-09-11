"""진단·배치 오케스트레이션 회귀 검사. 실제 물리 검사가 아님."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('campaign', ROOT/'tools/run_physics_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class ValidationTests(unittest.TestCase):
    def legacy_cli(self,script,*args):
        """Force the Windows encoding that previously lost diagnostic JSON."""
        env=dict(os.environ,PYTHONUTF8='0',PYTHONIOENCODING='cp949')
        return subprocess.run([sys.executable,'-X','utf8=0','-S',str(ROOT/'tools'/script),*map(str,args)],
                              cwd=ROOT,capture_output=True,text=True,encoding='cp949',env=env,timeout=20)

    def test_01_standard_library_preflight(self):
        """-S: site-packages 없이 진단 모듈만 실행 가능해야 한다."""
        p=subprocess.run([sys.executable,'-S','-c',
            'import json;from flylab.dependencies import dependency_report;print(json.dumps(dependency_report()))'],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertFalse(json.loads(p.stdout)['ready'])

    def test_02_missing_numpy_produces_blocked_json(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'newdir'/'gate.json'
            p=self.legacy_cli('verify_physics.py','--output',out)
            self.assertEqual(p.returncode,2,p.stderr)
            data=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['status'],'BLOCKED')
            self.assertFalse(data['physicalExecuted'])
            self.assertEqual(data['checks'],[])
            self.assertIn('3.12–3.14',data['dependencies']['hint'])

    def test_03_gate_blocks_all_campaign_cases(self):
        with tempfile.TemporaryDirectory() as t:
            p=self.legacy_cli('run_physics_campaign.py','--out',t)
            self.assertEqual(p.returncode,2,p.stderr)
            data=json.loads((Path(t)/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(data['status'],'BLOCKED')
            self.assertEqual(data['completedCases'],0)
            self.assertEqual(len(data['cases']),33)
            self.assertTrue(all(x['status']=='BLOCKED' and not x['physicalExecuted'] for x in data['cases']))

    def test_04_plan_has_no_execution_claim(self):
        p=self.legacy_cli('run_physics_campaign.py','--plan')
        self.assertEqual(p.returncode,0,p.stderr)
        data=json.loads(p.stdout)
        self.assertEqual(data['status'],'PLAN_ONLY')
        self.assertFalse(data['physicalExecuted'])
        self.assertEqual(len(data['cases']),33)
        self.assertEqual(next(item['description'] for item in data['cases'] if item['case']=='motor-off'),campaign.CASES['motor-off'])

    def test_05_campaign_plan_unique(self):
        plan=campaign.build_plan([7,19,43],10)
        self.assertEqual(len({(x['seed'],x['case']) for x in plan}),33)
        self.assertEqual({x['durationSeconds'] for x in plan},{10})

    def test_06_reject_bad_duration(self):
        p=subprocess.run([sys.executable,'-S',str(ROOT/'tools/run_physics_campaign.py'),'--seconds','nan'],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(p.returncode,0)

    def _results(self,case='motor-off',pre=0,post=1,initial='same'):
        def traj(shift):
            return [dict(time=1.,position=[pre if shift else 0,1,0],yaw=0.),
                    dict(time=5.,position=[post if shift else 0,1,0],yaw=0.)]
        return [dict(case='baseline',seed=7,status='PASS',initialStateSHA256='same',trajectory=traj(False)),
                dict(case=case,seed=7,status='PASS',initialStateSHA256=initial,trajectory=traj(True))]

    def test_07_matched_pair_effect(self):
        data=campaign.compare_results(self._results())[0]
        self.assertEqual(data['status'],'PASS')
        self.assertTrue(data['effectObserved'])

    def test_08_preexisting_divergence_fails(self):
        data=campaign.compare_results(self._results(pre=.1))[0]
        self.assertEqual(data['status'],'FAIL')

    def test_09_absent_motor_effect_fails(self):
        data=campaign.compare_results(self._results(post=0))[0]
        self.assertEqual(data['status'],'FAIL')
        self.assertFalse(data['effectObserved'])

    def test_10_different_initial_state_fails(self):
        data=campaign.compare_results(self._results(initial='different'))[0]
        self.assertEqual(data['status'],'FAIL')

    def test_11_failed_native_run_not_compared(self):
        data=self._results();data[1]['status']='FAIL'
        self.assertEqual(campaign.compare_results(data)[0]['status'],'NOT_EVALUATED')

    def test_12_stale_pass_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t);(path/'gate.json').write_text('{"status":"PASS","physicalValidation":true}')
            (path/'report.json').write_text('{"status":"PASS"}')
            p=self.legacy_cli('run_physics_campaign.py','--out',t)
            self.assertEqual(p.returncode,2,p.stderr)
            self.assertEqual(json.loads((path/'report.json').read_text(encoding='utf-8'))['status'],'BLOCKED')
            self.assertEqual(json.loads((path/'gate.json').read_text(encoding='utf-8'))['status'],'BLOCKED')

    def test_13_obstacle_near_boundary_has_valid_explicit_placement(self):
        from types import SimpleNamespace
        from flylab.sensors import default_world, validate_world
        world=default_world()
        def place(message):
            point=message['payload']['position']
            world['obstacles'].append(dict(id='test-placement',p=point,r=.8))
            validate_world(world)
        engine=SimpleNamespace(frame=lambda:dict(body=dict(position=[22,1,0],yaw=0)),command=place)
        details=campaign.apply_intervention(engine,'obstacle')
        self.assertNotEqual(details['angleOffsetDegrees'],0)
        self.assertAlmostEqual((details['position'][0]-22)**2+details['position'][2]**2,25)

if __name__=='__main__':
    unittest.main(verbosity=2)
