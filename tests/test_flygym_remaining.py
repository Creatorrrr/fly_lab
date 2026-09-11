import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from flylab.retinal_input import RetinalInput
from flylab.flygym_senses import FourSiteOdor


class ClockedSenses(unittest.TestCase):
    def test_retinal_cadence_intensity_motion_and_restore(self):
        tick=[0];calls=[];light=[.5]
        def image():
            calls.append(tick[0]);a=np.zeros((2,721,2),np.float32);a[:,:,0]=light[0]
            return dict(time_s=tick[0]/10000,ommatidia=a)
        body=SimpleNamespace(physics_time=lambda:tick[0]/10000,compound_eye_observation=image)
        sensor=RetinalInput(dict(sample_hz=100,tau_s=0.))
        self.assertAlmostEqual(sensor.observe(body)['features']['retina_luminance_left'],.5)
        tick[0]=50;sensor.observe(body);self.assertEqual(calls,[0])
        saved=sensor.snapshot();tick[0]=100;light[0]=.1
        expected=sensor.observe(body)
        self.assertGreater(expected['features']['retina_motion_left'],39.)
        self.assertGreater(expected['features']['retina_loom_left'],99.)
        restored=RetinalInput(dict(sample_hz=100,tau_s=0.));restored.restore(saved)
        self.assertEqual(restored.observe(body),expected)
        bad=copy.deepcopy(saved);bad['next_tick']=105
        before=restored.snapshot()
        with self.assertRaises(ValueError):restored.restore(bad)
        self.assertEqual(restored.snapshot(),before)
        tick[0]=250
        with self.assertRaisesRegex(ValueError,'missed'):restored.observe(body)

    def test_plume_direction_and_site_mapping(self):
        from flylab.sensors import default_world
        from flylab.common import to_ui
        clock=[0.]
        b=SimpleNamespace(body_indices={'l_funiculus':0,'r_funiculus':1,'c_rostrum':2},body_ids=np.arange(3),
            d=SimpleNamespace(xpos=np.array([[1.,0.,0.],[1.,.2,0.],[1.,.1,0.]]),xmat=np.tile(np.eye(3).ravel(),(3,1))),physics_time=lambda:clock[0])
        w=default_world();w['sources']=[dict(id='odor',kind='food',p=to_ui([0.,0.,0.]),strength=1.)]
        odor=FourSiteOdor(field='plume',wind_mm_s=(2.,0.,0.))
        first=odor.observe(b,w);clock[0]=.25
        second=odor.observe(b,w)
        self.assertGreater(np.max(np.abs(np.asarray(first['response'])-second['response'])),.1)
        b.d.xpos[:,0]=-1.
        self.assertFalse(np.any(odor.observe(b,w)['response']))
        with self.assertRaises(ValueError):FourSiteOdor(field='plume',wind_mm_s=(0.,0.,0.))

    def test_retina_ports_require_explicit_mapping(self):
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.ports import PortBindings,feature
        g=graph_fixture();spec=copy.deepcopy(bindings_fixture(g).spec)
        spec['sensory'][0].update(channel='retina_ommatidium',eye='left',ommatidium=2)
        with self.assertRaises(ValueError):PortBindings(g,spec)
        spec['sensor_model']={'retina':{'sample_hz':100},'kind':'flygym-multimodal-v2'}
        with self.assertRaises(ValueError):PortBindings(g,spec)
        spec['sensory'][0]['retinotopy_evidence']=['SYNTHETIC TEST ONLY']
        binding=PortBindings(g,spec)
        self.assertEqual(feature({},binding.spec['sensory'][0],{'retina_ommatidia':[[0.,0.,.7],[0.,0.,0.]]}),.7)


@unittest.skipUnless(os.environ.get('FLYLAB_NATIVE_TESTS')=='1','Explicit native physics verification')
class NativeRemaining(unittest.TestCase):
    def test_lf_platform_has_actual_contact_and_preserves_restore(self):
        from flylab.c.muscle_imitation import MuscleImitation
        rig=MuscleImitation(contact_platform=True)
        airborne=MuscleImitation()
        try:
            frame=rig.step(teacher=True)
            self.assertTrue(any(c['normal_force']>0 for c in frame['contacts']))
            saved=rig.snapshot();expected=rig.step(teacher=True);rig.restore(saved)
            self.assertEqual(expected,rig.step(teacher=True))
            with self.assertRaises(ValueError):airborne.restore(saved)
            self.assertFalse(frame['biological_validation'])
        finally:rig.close();airborne.close()

    def test_retinal_pixels_repeat_and_restore_exactly(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        b=FlyGymBody(42,default_world(),config_values(None));clone=None
        try:
            for _ in range(3):b.step(dict(forwardSpeed=0.,yawRate=0.))
            saved=b.snapshot();first=b.compound_eye_observation()
            for _ in range(4):
                actual=b.compound_eye_observation()
                np.testing.assert_array_equal(first['raw_rgb'],actual['raw_rgb'])
                np.testing.assert_array_equal(first['ommatidia'],actual['ommatidia'])
            clone=FlyGymBody(42,default_world(),config_values(None));clone.restore(saved)
            actual=clone.compound_eye_observation()
            np.testing.assert_array_equal(first['raw_rgb'],actual['raw_rgb'])
            np.testing.assert_array_equal(first['ommatidia'],actual['ommatidia'])
        finally:
            b.close()
            if clone:clone.close()

    def test_multimodal_sensor_restore_and_off(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        from flylab.c.sensors import CSensorAdapter
        from flylab.c.ports import SensoryEncoder,PortBindings
        from tests.c_fixtures import graph_fixture,bindings_fixture
        b=FlyGymBody(42,default_world(),config_values(None))
        try:
            model=dict(kind='flygym-multimodal-v2',field='plume',retina=dict(sample_hz=100),adaptation_tau_s=.5)
            a=CSensorAdapter(42,model);a.observe(b,default_world(),.005,config_values(None))
            b.step(dict(forwardSpeed=0.,yawRate=0.));a.observe(b,default_world(),.005,config_values(None))
            saved=a.snapshot();clone=CSensorAdapter(1,model);clone.restore(saved)
            b.step(dict(forwardSpeed=0.,yawRate=0.))
            packet=a.observe(b,default_world(),.005,config_values(None))
            self.assertEqual(packet,clone.observe(b,default_world(),.005,config_values(None)))
            self.assertEqual(a.snapshot(),clone.snapshot())
            g=graph_fixture();spec=copy.deepcopy(bindings_fixture(g).spec);spec['sensor_model']=model
            spec['sensory'][0].update(channel='retina_luminance_left',gain=20.)
            enc=SensoryEncoder(PortBindings(g,spec),42)
            enabled=enc.encode(packet,.005,.0001,supplemental=a.diagnostics['features'])[0]
            off=enc.encode(packet,.005,.0001,disabled=['*'],supplemental=a.diagnostics['features'])[0]
            self.assertGreater(enabled.sum(),0.);self.assertEqual(off.sum(),0.)
        finally:b.close()

    def test_muscle_actions_and_future_restore(self):
        from flylab.c.muscle_imitation import MuscleImitation
        rig=MuscleImitation()
        try:
            for _ in range(4):rig.step(teacher=True)
            saved=rig.snapshot();expected=rig.step(teacher=True)
            rig.restore(saved);actual=rig.step(teacher=True)
            np.testing.assert_array_equal(expected['q'],actual['q'])
            before=rig.snapshot()
            with self.assertRaises(ValueError):rig.step(np.full(15,np.nan))
            np.testing.assert_array_equal(before['state'],rig.snapshot()['state'])
        finally:rig.close()


class BatchLifecycle(unittest.TestCase):
    def test_close_cleans_every_resource_even_when_one_close_fails(self):
        from flylab.c.batch import BatchSession
        closed=[]
        def close_physics():
            closed.append('physics');raise RuntimeError('test close failure')
        session=BatchSession.__new__(BatchSession)
        session.closed=False;session.active=['0','1'];session.neural=object()
        session.physics=SimpleNamespace(close=close_physics)
        session.engines={key:SimpleNamespace(close=lambda key=key:closed.append(key)) for key in ('0','1')}
        with self.assertRaisesRegex(RuntimeError,'test close failure'):session.close()
        self.assertEqual(closed,['physics','0','1']);self.assertTrue(session.closed)
        self.assertEqual(session.active,[]);self.assertIsNone(session.physics);self.assertIsNone(session.neural)
        session.close();self.assertEqual(closed,['physics','0','1'])


@unittest.skipUnless(os.environ.get('FLYLAB_CUDA_BATCH_TESTS')=='1','Explicit CUDA batch verification')
class CudaRemaining(unittest.TestCase):
    def test_closed_batch_rejects_mutation_before_any_clock_advances(self):
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.batch import BatchSession
        graph=graph_fixture();session=BatchSession(graph,bindings_fixture(graph),[42])
        try:
            session.step(1);engine=session.engines['0']
            before=(engine.tick,engine.neural.tick,engine.encoder.control_tick,engine.body.physics_time())
            voltage=engine.neural.snapshot()['v'].copy();position=engine.body.d.qpos.copy()
            session.close()
            with tempfile.TemporaryDirectory() as tmp:
                for call in (lambda:session.step(1),lambda:session.pause('0'),lambda:session.resume('0'),
                             lambda:session.cancel('0'),lambda:session.checkpoint(Path(tmp)/'checkpoint'),
                             lambda:engine.step(1),
                             lambda:engine.start_recording(Path(tmp)/'recording',[graph.nodes[0]['id']])):
                    with self.assertRaisesRegex(RuntimeError,'Closed'):call()
                self.assertEqual(list(Path(tmp).iterdir()),[])
            self.assertEqual(before,(engine.tick,engine.neural.tick,engine.encoder.control_tick,engine.body.physics_time()))
            np.testing.assert_array_equal(voltage,engine.neural.snapshot()['v'])
            np.testing.assert_array_equal(position,engine.body.d.qpos)
            self.assertTrue(session.summary()['closed']);self.assertEqual(session.active,[])
            self.assertIsNone(session.physics);self.assertIsNone(session.neural)
            self.assertEqual(session.status['0'],'RUNNING')  # Last state remains diagnostic metadata.
        finally:session.close()

    def test_subset_restore_preserves_world_mapping_clocks_and_controller_state(self):
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.batch import BatchSession
        from flylab.c.storage import StateStore
        graph=graph_fixture();binding=bindings_fixture(graph)
        session=BatchSession(graph,binding,[42,43,44]);restored=None
        try:
            session.step(2);session.pause('1');session.step(1)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'source';session.checkpoint(path);saved=StateStore.load(path,max_files=4096)
                restored=BatchSession.restore(graph,binding,path,worlds=['2','1'])
                self.assertEqual(restored.epoch,session.epoch+1)
                self.assertEqual(restored.status,{'0':'RUNNING','1':'PAUSED'})
                self.assertEqual(restored.regroup_origin['worlds'],['2','1'])
                for new,old in (('0','2'),('1','1')):
                    actual=restored.engines[new].checkpoint();expected=saved['engines'][old]
                    self.assertEqual(actual['seed'],expected['seed'])
                    self.assertEqual(actual['control_tick'],expected['control_tick'])
                    for field in ('v','h','rate','queue','spike_count','refractory_until','suppress','mute'):
                        np.testing.assert_array_equal(actual['neural'][field],expected['neural'][field])
                    np.testing.assert_array_equal(actual['encoder']['filtered'],expected['encoder']['filtered'])
                    self.assertEqual(actual['pending'],expected['pending']);self.assertEqual(actual['active'],expected['active'])
                gpu=restored.physics.snapshot()
                for field in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','time'):
                    np.testing.assert_array_equal(gpu['arrays'][field],saved['physics']['arrays'][field][[1]])
                for field in ('state','drive','mode','ticks','push_left','push'):
                    np.testing.assert_array_equal(gpu['controller'][field],saved['physics']['controller'][field][[1]])
                restored.step(1)
                self.assertEqual(restored.engines['0'].tick,200);self.assertEqual(restored.engines['1'].tick,100)
                checkpoint=Path(tmp)/'regrouped';restored.checkpoint(checkpoint);expected=restored.summary();restored.close()
                restored=BatchSession.restore(graph,binding,checkpoint)
                self.assertEqual(restored.summary(),expected)
                for worlds in ([],['2','2'],['3']):
                    with self.assertRaises(ValueError):BatchSession.restore(graph,binding,path,worlds=worlds)
        finally:
            if restored is not None:restored.close()
            session.close()

    def test_failed_batch_does_not_advance_and_preparation_is_rolled_back(self):
        from unittest.mock import patch
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.batch import BatchSession
        g=graph_fixture();session=BatchSession(g,bindings_fixture(g),[42,43])
        try:
            session.step(1)
            before={key:(e.tick,e.encoder.control_tick,e.body.d.qpos.copy(),e.neural.snapshot()['v'].copy()) for key,e in session.engines.items()}
            with patch.object(session.engines['1'],'_prepare_control',side_effect=ValueError('test preparation failure')):
                with self.assertRaisesRegex(ValueError,'test preparation failure'):session.step(1)
            self.assertEqual(session.active,[])
            with self.assertRaisesRegex(RuntimeError,'Faulted batch'):session.step(1)
            for key,e in session.engines.items():
                self.assertEqual(session.status[key],'FAILED')
                self.assertEqual((e.tick,e.encoder.control_tick),before[key][:2])
                np.testing.assert_array_equal(e.body.d.qpos,before[key][2])
                np.testing.assert_array_equal(e.neural.snapshot()['v'],before[key][3])
        finally:session.close()

    def test_paused_checkpoint_preserves_epoch_through_repeated_restore(self):
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.batch import BatchSession
        g=graph_fixture();binding=bindings_fixture(g);session=BatchSession(g,binding,[42])
        try:
            session.step(1);session.pause('0');expected=session.summary()
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(2):
                    path=Path(tmp)/str(i);session.checkpoint(path);session.close()
                    session=BatchSession.restore(g,binding,path)
                    self.assertEqual(session.summary(),expected)
                session.resume('0');session.step(1)
                self.assertEqual(session.engines['0'].tick,100)
        finally:session.close()

    def test_neural_batch_world_independence_and_pulses(self):
        from tests.c_fixtures import graph_fixture
        from flylab.c.neural_cuda import CudaLIF
        from flylab.c.neural_batch import CudaLIFBatch
        g=graph_fixture();singles=[CudaLIF(g) for _ in range(2)];neurals=[CudaLIF(g) for _ in range(2)]
        batch=CudaLIFBatch(neurals)
        self.assertIs(neurals[0].indptr,neurals[1].indptr)
        self.assertNotEqual(neurals[0].W.data.data.ptr,neurals[1].W.data.data.ptr)
        for phase in range(3):
            for n in (singles[1],neurals[1]):n.set_interventions(edges=[0] if phase%2==0 else [])
            drive=[np.full(g.n,30,np.float32),np.arange(g.n,dtype=np.float32)*10]
            capture=[np.arange(g.n,dtype=np.int32)]*2
            pulses=[None,(np.array([0,0],np.int32),np.full((50,2),2.,np.float32))]
            batch.advance(drive,50,capture,pulses)
            for one,other,x,c,p in zip(singles,neurals,drive,capture,pulses):
                one.advance(x,50,c,p)
                for key in ('v','h','rate','queue','spike_count','suppress','mute'):
                    np.testing.assert_array_equal(one.snapshot()[key],other.snapshot()[key])
                self.assertEqual(one.last_events,other.last_events)

    def test_batch_pause_cancel_checkpoint_and_render(self):
        from tests.c_fixtures import graph_fixture,bindings_fixture
        from flylab.c.batch import BatchSession
        g=graph_fixture();binding=bindings_fixture(g);session=BatchSession(g,binding,[42,43])
        restored=None
        try:
            metadata=session.engines['0'].body.physics_identity()
            self.assertAlmostEqual(metadata['tolerance'],1e-6,places=12)
            self.assertEqual(metadata['cpu_mirror_options']['tolerance'],1e-8)
            self.assertNotEqual(metadata['disableflags'],metadata['cpu_mirror_options']['disableflags'])
            session.step(2);session.pause('1');frozen=session.engines['1'].checkpoint();session.step(1)
            self.assertEqual(frozen['control_tick'],session.engines['1'].control_tick)
            np.testing.assert_array_equal(frozen['neural']['v'],session.engines['1'].neural.snapshot()['v'])
            session.resume('1');session.step(1)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'batch';session.checkpoint(path)
                restored=BatchSession.restore(g,binding,path)
                self.assertEqual(session.summary(),restored.summary())
                for key in session.engines:
                    np.testing.assert_array_equal(session.engines[key].body.d.qpos,restored.engines[key].body.d.qpos)
                image=restored.physics.render(0)
                self.assertEqual(image.shape,(240,400,3));self.assertGreater(float(np.std(image)),5.)
                restored.step(1)
                restored.cancel('1');tick=restored.engines['1'].tick;restored.step(1)
                self.assertEqual(tick,restored.engines['1'].tick)
        finally:
            if restored:restored.close()
            session.close()


if __name__=='__main__':unittest.main()
