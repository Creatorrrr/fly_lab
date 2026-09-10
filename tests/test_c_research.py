import copy
import importlib.util
import unittest
import numpy as np
from flylab.c.research_neural import ResearchLIF
from flylab.c.neural import ExpLIF
from flylab.c.metabolism import Metabolism
from flylab.c.engine import CEngine
from flylab.sensors import default_world
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state
from tests.test_c_mps import mps_available

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.g=graph_fixture();self.spec=dict(schema='flylab.physiology.v1',graph_hash=self.g.hash,
                                           evidence=['SYNTHETIC TEST'],uncertainty='Test-only model')
    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Optional Torch required; install requirements-research.txt')
    def test_empty_profile_matches_reference_spikes_and_voltage(self):
        n=ResearchLIF(self.g,self.spec);r=ExpLIF(self.g);x=np.full(self.g.n,20.,np.float32)
        n.advance(x,200,range(self.g.n));r.advance(x,200,range(self.g.n))
        self.assertEqual(n.last_events,r.last_events)
        np.testing.assert_allclose(n.snapshot()['arrays']['v'],r.v,atol=1e-4,rtol=0)
    @unittest.skipUnless(mps_available(),'Actual MPS required')
    def test_heterogeneity_delays_gaps_plasticity_cpu_mps_and_restore(self):
        self.check_heterogeneity('mps')

    def check_heterogeneity(self,device):
        ids=[n['id'] for n in self.g.nodes]
        spec=dict(self.spec,cells=[dict(ids=[ids[1]],parameters={'tau_m_s':.015},evidence=['test'])],
                  synapses=[dict(pre=ids[0],post=ids[1],weight_mV=2.,delay_s=.001,receptor='test effect',evidence=['test'])],
                  gap_junctions=[dict(a=ids[3],b=ids[4],coupling=.2,evidence=['test'])],
                  plasticity=dict(rule='reward_stdp_v1',eta=.01,tau_s=.02,max_abs_mV=10.,
                                  edges=[dict(pre=ids[0],post=ids[1])],evidence=['test']))
        cpu=ResearchLIF(self.g,spec);gpu=ResearchLIF(self.g,spec,device);x=np.zeros(self.g.n,np.float32);x[[0,3]]=30.
        cpu.advance(x,150,list(range(6)),reward=1.);gpu.advance(x,150,list(range(6)),reward=1.)
        self.assertEqual(cpu.last_events,gpu.last_events)
        for k,v in cpu.snapshot()['arrays'].items():
            np.testing.assert_allclose(v,gpu.snapshot()['arrays'][k],rtol=0,atol=1e-3 if k=='rate' else 1e-4,err_msg=k)
        saved=gpu.snapshot();cpu.restore(saved);gpu.advance(x,100);cpu.advance(x,100)
        np.testing.assert_array_equal(gpu.snapshot()['arrays']['spike_count'],cpu.snapshot()['arrays']['spike_count'])
    def test_energy_conservation_depletion_food_off_and_restore(self):
        m=Metabolism(dict(capacity=1.,initial_energy=.2,basal_per_s=.001,intake_per_s=.1))
        w=default_world();w['sources']=[dict(id='food',kind='food',p=[0.,0.,0.],strength=.001)]
        for _ in range(10):m.step(.005,[0,0,0],0.,w)
        self.assertAlmostEqual(m.intake,.001);self.assertAlmostEqual(m.energy,.2+m.intake-m.expenditure)
        saved=m.snapshot();m.step(.005,[0,0,0],0.,w);target=m.snapshot();m.restore(saved);m.step(.005,[0,0,0],0.,w)
        same_state(target,m.snapshot())
        bad=copy.deepcopy(saved);bad['energy']+=.1
        with self.assertRaises(ValueError):m.restore(bad)
        same_state(target,m.snapshot())
        w['sources'][0]['strength']=1.;w['foodOn']=False
        self.assertEqual(m.step(.005,[0,0,0],0.,w),0.)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Optional Torch required; install requirements-research.txt')
    def test_reward_gated_learning_changes_only_selected_existing_edge(self):
        ids=[n['id'] for n in self.g.nodes]
        spec=dict(self.spec,cells=[dict(ids=ids[:2],parameters={'tau_m_s':.001},evidence=['test'])],
            plasticity=dict(rule='reward_stdp_v1',eta=.01,tau_s=.02,max_abs_mV=10.,
                            edges=[dict(pre=ids[0],post=ids[1])],evidence=['test']))
        learned=ResearchLIF(self.g,spec);frozen=ResearchLIF(self.g,spec)
        for target in (0,1):
            drive=np.zeros(self.g.n,np.float32);drive[target]=30.
            learned.advance(drive,10,reward=1.);frozen.advance(drive,10,reward=0.)
        a=learned.snapshot()['arrays']['weights'];b=frozen.snapshot()['arrays']['weights']
        index=int(learned.pe[0]);self.assertGreater(a[index],b[index]);self.assertEqual(np.count_nonzero(a!=b),1)
    def test_engine_metabolism_checkpoint_and_no_hidden_motor_control(self):
        b=bindings_fixture(self.g);plain=CEngine(self.g,b,body_factory=FixtureBody)
        fed=CEngine(self.g,b,body_factory=FixtureBody,metabolism={})
        try:
            plain.step(10);fed.step(10);same_state(plain.neural.snapshot(),fed.neural.snapshot());same_state(plain.body.snapshot(),fed.body.snapshot())
            restored=CEngine.from_checkpoint(self.g,b,fed.checkpoint(),FixtureBody)
            try:
                restored.step(5);fed.step(5);same_state(fed.metabolism.snapshot(),restored.metabolism.snapshot())
            finally:restored.close()
        finally:plain.close();fed.close()
if __name__=='__main__':unittest.main()
