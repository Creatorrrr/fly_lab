"""Regression coverage for fix_2. Synthetic tests are NOT native physics runs."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import math
import subprocess
import unittest
import numpy as np
from flylab.brain import Circuit,NeuralController
from flylab.engine import default_graph,config_values,Engine
from tests.fixture_body import FixtureBody
from tools.navigation_checks import NavigationMonitor,audit_trajectory
from tools.run_physics_campaign import apply_intervention
from flylab.sensors import default_world,validate_world

ROOT=Path(__file__).resolve().parents[1]

def rows(n=61,step=.1,speed=0.,**override):
    return [dict(t=i*step,p=[speed*i*step,1.,0.],yaw=0.,front=4.,contact=0,
                 avoiding=True,recovering=False,motorCoupled=True,motionExpected=True,
                 **override) for i in range(n)] if not override else [
        {**r,**override} for r in rows(n,step,speed)]


class NavigationFixTests(unittest.TestCase):
    def test_4mm_gap_stationary_detected(self):
        a=audit_trajectory(rows(31));self.assertEqual(a['status'],'FAIL');self.assertEqual(a['stationaryWindows'][0]['spanMM'],0)
    def test_one_clear_sample_does_not_disable_monitor(self):
        rs=rows(31);rs[15].update(front=8.,contact=0,avoiding=False)
        self.assertEqual(audit_trajectory(rs)['status'],'FAIL')
    def test_pure_spin_is_stuck(self):
        rs=rows(31)
        for r in rs:r['yaw']=2*r['t']
        a=audit_trajectory(rs);self.assertEqual(a['status'],'FAIL');self.assertAlmostEqual(a['stationaryWindows'][0]['yawSweepRad'],6.)
    def test_oscillating_retreat_stays_in_small_envelope(self):
        rs=rows(61)
        for r in rs:r.update(p=[.2*math.sin(r['t']*20),1,0],recovering=True)
        self.assertEqual(audit_trajectory(rs)['status'],'FAIL')
    def test_zero_drive_is_not_assumed_intentional_stop(self):
        self.assertEqual(audit_trajectory(rows(31,drive=[0,0],avoiding=False))['status'],'FAIL')
    def test_explicit_motor_off_excluded(self):
        a=audit_trajectory(rows(61,motorCoupled=False));self.assertEqual(a['status'],'NOT_EVALUATED');self.assertEqual(a['evaluatedWindows'],0)
    def test_explicit_motion_not_expected_excluded(self):
        self.assertEqual(audit_trajectory(rows(61,motionExpected=False))['evaluatedWindows'],0)
    def test_resume_gets_full_3sec_window(self):
        rs=rows(61)
        for r in rs[:20]:r['motorCoupled']=False
        a=audit_trajectory(rs);self.assertEqual(a['stationaryWindows'][0]['startTime'],2.);self.assertEqual(a['stationaryWindows'][0]['endTime'],5.)
    def test_translation_passes(self):self.assertEqual(audit_trajectory(rows(speed=2))['status'],'PASS')
    def test_window_is_time_based(self):
        a=audit_trajectory(rows(61,step=.05));self.assertEqual(len(a['stationaryWindows']),1);self.assertEqual(a['stationaryWindows'][0]['endTime'],3)
    def test_missing_samples_not_silently_pass(self):
        rs=rows(81,speed=2);del rs[20:24]
        self.assertEqual(audit_trajectory(rs)['status'],'INCOMPLETE')
    def test_nonmonotone_rejected(self):
        rs=rows(5);rs[3]['t']=rs[2]['t']
        with self.assertRaises(ValueError):audit_trajectory(rs)
    def test_nonfinite_rejected(self):
        rs=rows(5);rs[2]['p'][0]=float('nan')
        with self.assertRaises(ValueError):audit_trajectory(rs)
    def test_flags_must_be_explicit(self):
        rs=rows(5);del rs[0]['motorCoupled']
        with self.assertRaises(ValueError):audit_trajectory(rs)
    def test_flag_change_alone_not_walking(self):
        rs=rows(61)
        for r in rs[10:]:r['avoiding']=False
        a=audit_trajectory(rs);self.assertFalse(a['walkingResumed']);self.assertTrue(any(e['status']=='FAILED_NO_RESUMPTION' for e in a['resumptionEpisodes']))
    def test_actual_forward_net_movement_resumes(self):
        rs=rows(61,speed=2)
        for r in rs[10:]:r['avoiding']=False
        a=audit_trajectory(rs);self.assertTrue(a['walkingResumed'])
    def test_reverse_only_not_forward_resume(self):
        rs=rows(61,speed=-2)
        for r in rs[10:]:r['avoiding']=False
        self.assertFalse(audit_trajectory(rs)['walkingResumed'])
    def test_closed_jitter_not_resume(self):
        rs=rows(61)
        for r in rs:
            r['p'][0]=.2*math.sin(r['t']*30)
            if r['t']>=1:r['avoiding']=False
        self.assertFalse(audit_trajectory(rs)['walkingResumed'])
    def test_recovery_flag_keeps_episode_active(self):
        rs=rows(61,speed=2)
        for r in rs[10:]:r.update(avoiding=False,recovering=True)
        self.assertFalse(audit_trajectory(rs)['walkingResumed'])
    def test_end_of_record_is_censored(self):
        rs=rows(12,speed=2);rs[-1]['avoiding']=False
        a=audit_trajectory(rs);self.assertFalse(a['walkingResumed']);self.assertEqual(a['resumptionEpisodes'][-1]['status'],'CENSORED_END_OF_RECORD')

class ObstacleFixTests(unittest.TestCase):
    def place(self,p,yaw):
        world=default_world();world['obstacles']=[]
        def command(msg):
            point=msg['payload']['position'];self.assertGreater(math.dist(p,point),3.8)
            world['obstacles'].append(dict(id='test',p=point,r=.8));validate_world(world)
        fake=SimpleNamespace(frame=lambda:dict(body=dict(position=p,yaw=yaw)),command=command)
        return apply_intervention(fake,'obstacle')
    def test_reported_corner_plus135(self):
        d=self.place([23.35,.8,17.35],math.radians(93.9))
        self.assertAlmostEqual(d['angleOffsetDegrees'],135);self.assertAlmostEqual(math.hypot(d['position'][0]-23.35,d['position'][2]-17.35),5)
    def test_mirrored_corner_minus135(self):
        self.assertAlmostEqual(self.place([23.35,.8,-17.35],math.radians(-93.9))['angleOffsetDegrees'],-135)
    def test_front_priority_unchanged(self):self.assertEqual(self.place([0,.8,0],0)['angleOffsetDegrees'],0)
    def test_corner_sweep(self):
        for x in [-23.9,23.9]:
            for z in [-17.9,17.9]:
                for deg in range(0,360,5):
                    d=self.place([x,.8,z],math.radians(deg));self.assertLessEqual(abs(d['position'][0]),23);self.assertLessEqual(abs(d['position'][2]),17)

class RestoreFixTests(unittest.TestCase):
    def setUp(self):
        self.brain=NeuralController(Circuit(default_graph()),42)
        self.config=config_values();self.sensor=dict(panorama=[0]*64,nearRanges=[1]*9,contact=1,odor=[0,0],odorChange=0,danger=0,angularVelocity=.5,forwardSpeed=0,clearanceDown=1,clearanceUp=10)
    def test_timer_range_and_atomicity(self):
        original=self.brain.snapshot()
        for k,v in [('contactTime',-1e-6),('contactTime',2),('recoveryTime',-.01),('recoveryTime',10),('recoveryTime',.40001),('turnMemory',math.tau+.01),('avoidSide',0),('avoidSide',.5),('prevContact',.3),('clock',-1),('rng',0)]:
            with self.subTest(k=k,v=v):
                s=copy.deepcopy(original);s[k]=v;s['a'][0]=.8
                with self.assertRaises(ValueError):self.brain.restore(s)
                self.assertEqual(original,self.brain.snapshot())
    def test_neural_array_coercions_rejected(self):
        for v in [True,'0.1']:
            s=self.brain.snapshot();s['a'][0]=v
            with self.assertRaises(ValueError):self.brain.restore(s)
    def test_nonfinite_and_boolean_rejected(self):
        for k in ['contactTime','recoveryTime','turnMemory','avoidSide']:
            for v in [float('nan'),float('inf'),True]:
                s=self.brain.snapshot();s[k]=v
                with self.assertRaises(ValueError):self.brain.restore(s)
    def test_exact_valid_boundaries(self):
        for c in [0,1]:
            for r in [0,.4,.4+5e-13]:
                s=self.brain.snapshot();s.update(contactTime=c,recoveryTime=r,turnMemory=math.tau,avoidSide=-1)
                self.brain.restore(s);self.assertEqual(self.brain.snapshot(),s)
    def test_restores_active_recovery_and_continues(self):
        for _ in range(65):self.brain.step(self.sensor,.005,self.config)
        self.assertGreater(self.brain.recoveryTime,0)
        other=NeuralController(self.brain.circuit,100);other.restore(json.loads(json.dumps(self.brain.snapshot())))
        for _ in range(160):
            self.brain.step(self.sensor,.005,self.config);other.step(self.sensor,.005,self.config)
        self.assertEqual(self.brain.snapshot(),other.snapshot())
    def test_malformed_intervention_atomic(self):
        s=self.brain.snapshot();s['stim']=[[0,dict(amplitude=.1,remaining=1)],[0,dict(amplitude=.2,remaining=2)]];old=self.brain.snapshot()
        with self.assertRaises(ValueError):self.brain.restore(s)
        self.assertEqual(self.brain.snapshot(),old)
    def test_engine_failed_restore_keeps_live_state(self):
        e=Engine(body_factory=FixtureBody)
        try:
            original=e.checkpoint();bad=copy.deepcopy(original);bad['brain']['recoveryTime']=10
            with self.assertRaises(ValueError):Engine.from_checkpoint(bad,body_factory=FixtureBody)
            self.assertEqual(e.checkpoint(),original)
        finally:e.close()
    def test_js_restore_rejects_and_preserves(self):
        js="""const fs=require('fs'),vm=require('vm');for(const k of ['math','connectome','brain'])vm.runInThisContext(fs.readFileSync('src/core/'+k+'.js','utf8'));const b=new Fly.NeuralController(new Fly.Circuit(JSON.parse(fs.readFileSync('data/circuit.json'))),42);const saved=JSON.stringify(b.snapshot());let n=0;for(const [k,v] of [['contactTime',2],['recoveryTime',10],['turnMemory',7],['avoidSide',0],['rng',0]]){const s=JSON.parse(saved);s[k]=v;s.a[0]=.8;try{b.restore(s);}catch(e){n++;}if(JSON.stringify(b.snapshot())!==saved)throw Error('not atomic');}const s=JSON.parse(saved);s.recoveryTime=.4;s.contactTime=1;b.restore(s);console.log(JSON.stringify({rejected:n,valid:b.snapshot().recoveryTime}));"""
        p=subprocess.run(['node','-e',js],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout),dict(rejected=5,valid=.4))

if __name__=='__main__':unittest.main()
