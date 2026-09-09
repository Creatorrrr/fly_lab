"""Tests of Python core and protocols. Fixture tests are NOT physical validation."""
import unittest,math,json,subprocess,io,csv
from pathlib import Path
import numpy as np
from flylab import CONTROL_DT,PROTOCOL
from flylab.engine import Engine,Dispatcher,default_graph,config_values,validate_sensor_packet
from flylab.brain import Circuit,NeuralController,validate_graph
from flylab.common import S,to_ui,to_physics,RNG,clone
from flylab.body import MotorAdapter,dependency_report,FlyGymBody
from flylab.sensors import default_world,validate_world
from tests.fixture_body import FixtureBody
ROOT=Path(__file__).resolve().parents[1]

class PureTests(unittest.TestCase):
    def test_01_graph_scope(self):
        c=Circuit(default_graph());self.assertEqual((c.real_count,c.N,len(c.pre)),(98,170,177));self.assertTrue(all(isinstance(n['id'],str) for n in c.nodes))
    def test_02_sparse_normalization(self):
        c=Circuit(default_graph());a=c.propagate(np.ones(c.N));self.assertTrue((a<=1+1e-12).all());self.assertEqual(float(c.propagate(np.ones(c.N),0).sum()),0)
    def test_03_invalid_graph(self):
        g=default_graph();g['edges'][0]['pre']='bogus'
        with self.assertRaises(ValueError):validate_graph(g)
    def test_04_coordinates(self):
        v=[1,2,3];self.assertEqual(to_ui(v),[1,3,-2]);np.testing.assert_equal(to_physics(to_ui(v)),v);self.assertAlmostEqual(np.linalg.det(S),1)
    def test_05_motor_direction(self):
        m=MotorAdapter();x=m.map(dict(forwardSpeed=2,yawRate=1));y=m.map(dict(forwardSpeed=2,yawRate=-1))
        self.assertGreater(x[0],x[1]);np.testing.assert_allclose(x,y[::-1]);np.testing.assert_equal(m.map(dict(forwardSpeed=0,yawRate=0)),[0,0])
    def test_06_reject_flight(self):
        with self.assertRaises(ValueError):config_values({'mode':'flight'})
    def test_07_world_validation(self):
        w=default_world();w['bounds'][0]=25
        with self.assertRaises(ValueError):validate_world(w)
    def test_08_rng(self):
        a=RNG(42);b=RNG(42);self.assertEqual([a.next() for _ in range(1000)],[b.next() for _ in range(1000)])
    def test_09_python_js_parity(self):
        g=default_graph();cfg=config_values();b=NeuralController(Circuit(g),42);steps=[]
        for i in range(750):
            yaw=math.sin(i*.008);s=dict(schema='flylab.sensors.v2',panorama=[.92*math.exp(8*(math.cos(k/64*math.tau-math.pi+yaw)-1)) if i<350 else 0 for k in range(64)],nearRanges=[2+abs(math.sin(i*.01+k))*8 for k in range(9)],odor=[.1,.2],odorChange=.01,danger=.15,angularVelocity=math.cos(i*.008)*.4,forwardSpeed=2,clearanceDown=.9,clearanceUp=10,contact=int(200<i<205))
            event=dict(ids=['model:PFL3-L:0'],kind='stimulate',amplitude=.3,duration=.2) if i==100 else None
            if event:b.intervene(**event)
            b.step(s,CONTROL_DT,cfg);steps.append(dict(sensor=s,event=event))
        result=subprocess.run(['node',str(ROOT/'tests/parity.cjs')],input=json.dumps(dict(graph=g,config=cfg,seed=42,steps=steps,dt=CONTROL_DT)),text=True,capture_output=True,check=True)
        js=json.loads(result.stdout);np.testing.assert_allclose(b.a,js['a'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(list(b.output.values()),list(js['output'].values()),atol=1e-12)
        (ROOT/'tests/parity_result.json').write_text(json.dumps(dict(passed=True,steps=750,maxAbsDifference=float(np.abs(b.a-np.array(js['a'])).max()),physics=False),indent=2))
    def test_10_backend_not_silent_fallback(self):
        if not dependency_report()['ready']:
            with self.assertRaisesRegex(RuntimeError,'의존성'):FlyGymBody(42,default_world(),config_values())

class FixtureIntegrationTests(unittest.TestCase):
    def setUp(self):self.e=Engine(body_factory=FixtureBody)
    def tearDown(self):self.e.close()
    def test_11_all_signals_finite(self):
        self.e.step(800);self.assertTrue(np.isfinite(self.e.brain.a).all());self.assertGreaterEqual(self.e.brain.a.min(),0);self.assertLessEqual(self.e.brain.a.max(),1)
    def test_12_no_truth_packet(self):
        validate_sensor_packet(self.e.last_sensors);p=clone(self.e.last_sensors);p['yaw']=1
        with self.assertRaises(ValueError):validate_sensor_packet(p)
    def test_13_clocks(self):
        f=self.e.step(100);self.assertAlmostEqual(f['simTime'],.5);self.assertAlmostEqual(f['physics']['physicsTime'],.5)
    def test_14_suppression(self):
        self.e.command(dict(type='intervene',payload=dict(ids=['model:DN:0'],kind='suppress')));self.e.step(20);self.assertEqual(self.e.brain.val('DN',0),0)
    def test_15_stimulus_expiry(self):
        self.e.command(dict(type='intervene',payload=dict(ids=['model:DN:0'],kind='stimulate',duration=.01)));self.e.step(3);self.assertEqual(len(self.e.brain.stim),0)
    def test_16_motor_disconnect(self):
        self.e.step(100);self.e.command(dict(type='configure',payload=dict(motorCoupled=False)));self.e.step(10);self.assertEqual(self.e.brain.output,dict(yawRate=0,forwardSpeed=0,verticalSpeed=0))
    def test_17_checkpoint_continuation(self):
        self.e.step(100);cp=json.loads(json.dumps(self.e.checkpoint()));other=Engine.from_checkpoint(cp,FixtureBody)
        try:self.e.step(100);other.step(100);self.assertEqual(self.e.checkpoint(),other.checkpoint())
        finally:other.close()
    def test_18_restore_reject_unknown_fields(self):
        cp=self.e.checkpoint();cp['brain']['__dict__']={}
        with self.assertRaises(ValueError):Engine.from_checkpoint(cp,FixtureBody)
    def test_19_neural_csv(self):
        self.e.step(40);rows=list(csv.reader(io.StringIO(self.e.csv())));self.assertEqual(len(rows[0]),177);self.assertEqual(len(rows[1]),177)
    def test_20_physical_csv(self):
        self.e.step(40);rows=list(csv.reader(io.StringIO(self.e.csv(True))));self.assertEqual(len(rows[0]),135)
    def test_21_subscription_does_not_disable_compute(self):
        self.e.subscribe([]);self.e.step(100);f=self.e.frame();self.assertEqual(f['neural']['values'],[]);self.assertGreater(self.e.brain.a.max(),.1)
    def test_22_invalid_placement_transactional(self):
        w=clone(self.e.world)
        with self.assertRaises(ValueError):self.e.command(dict(type='place',payload=dict(kind='obstacle',position=[0,1,0])))
        self.assertEqual(w,self.e.world)
    def test_23_static_world_change_preserves_states(self):
        self.e.step(20);s=self.e.body.snapshot();self.e.command(dict(type='place',payload=dict(kind='obstacle',position=[15,1,10])));self.assertEqual(s,self.e.body.snapshot())
    def test_24_bad_intervention_does_not_mutate(self):
        a=self.e.brain.snapshot()
        with self.assertRaises(ValueError):self.e.command(dict(type='intervene',payload=dict(ids=['unknown'],kind='suppress')))
        self.assertEqual(a,self.e.brain.snapshot())
    def test_25_push_is_explicit(self):
        self.e.command(dict(type='push',payload=dict(bw=.5,duration=.05)));self.assertGreater(self.e.body.push,0);self.assertEqual(self.e.log[-1]['command']['type'],'push')
    def test_26_bounded_history(self):
        # Exercise deque without a fake long-duration physics benchmark.
        for _ in range(1300):self.e._record()
        self.assertEqual(len(self.e.history),1201)

class DispatchTests(unittest.TestCase):
    def setUp(self):self.d=Dispatcher(FixtureBody);self.n=0;self.rpc('init')
    def tearDown(self):self.d.close()
    def rpc(self,op,p=None):self.n+=1;return self.d.handle(dict(protocol=PROTOCOL,requestId=self.n,op=op,payload=p or {}))['result']
    def test_27_capabilities_honest(self):
        r=self.d.ready()['capabilities'];self.assertFalse(r['physics']);self.assertTrue(r['testDouble']);self.assertFalse(r['fullBrain'])
    def test_28_step_limit(self):
        with self.assertRaises(ValueError):self.rpc('advance',dict(steps=11))
    def test_29_restore_transactional(self):
        cp=self.rpc('checkpoint');bad=clone(cp);bad['body']['backend']='fake'
        with self.assertRaises(ValueError):self.rpc('restore',bad)
        self.assertEqual(cp,self.rpc('checkpoint'))
    def test_30_replay(self):
        for _ in range(10):self.rpc('advance',dict(steps=10))
        self.rpc('command',dict(type='configure',payload=dict(goalAngle=-1.)))
        for _ in range(10):self.rpc('advance',dict(steps=10))
        target=self.rpc('checkpoint');replay=self.rpc('exportReplay');self.rpc('replay',replay);self.assertEqual(target,self.rpc('checkpoint'))
    def test_31_catalog_paging(self):
        r=self.rpc('catalog',dict(query='EPG',limit=10,offset=10));self.assertEqual(len(r['items']),10);self.assertEqual(r['total'],45)
    def test_32_pair_does_not_mutate(self):
        before=self.rpc('checkpoint');r=self.rpc('paired',dict(kind='motor-off',seconds=2));self.assertEqual(before,self.rpc('checkpoint'));self.assertGreater(r['meanPathSeparation'],0);self.assertFalse(r['physical'])
    def test_33_unknown_protocol(self):
        with self.assertRaises(ValueError):self.d.handle(dict(protocol='flylab.protocol.v1',requestId=1,op='frame'))

if __name__=='__main__':unittest.main(verbosity=2)
