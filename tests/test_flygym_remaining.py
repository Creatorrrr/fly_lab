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


@unittest.skipUnless(os.environ.get('FLYLAB_CUDA_BATCH_TESTS')=='1','Explicit CUDA batch verification')
class CudaRemaining(unittest.TestCase):
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
