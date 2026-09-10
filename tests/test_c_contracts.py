"""Regression counterexamples from the independent C0.6 review."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from flylab.c.campaign import pilot_spec, validate_spec, run_campaign
from flylab.c.engine import CEngine
from flylab.c.jobs import CampaignJobs
from flylab.c.model_config import resolve_parameters, model_ticks, execution_capabilities
from flylab.c.neural import ExpLIF, LIFParameters
from flylab.c.neural_api import NeuralBackend
from flylab.c.ports import PortBindings
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state
from tests.test_c_mps import mps_available

ROOT = Path(__file__).resolve().parents[1]


def banc_bindings():
    from tests.test_c_sensorimotor import banc_fixture
    from flylab.c.neuromuscular import build_spec
    graph = banc_fixture()
    review = dict(review_status='engineering_reviewed', evidence=['SYNTHETIC TEST'], uncertainty='Test only')
    spec = dict(schema='flylab.bindings.v3', graph_hash=graph.hash, neuromuscular=build_spec(graph),
        sensory=[dict(name='odor', channel='odor_mean',
                      ids=[next(n['id'] for n in graph.nodes if n['super_class']=='sensory')],
                      input_kind='sensory', method='drive_mV', gain=0., baseline=0., cap=24.,
                      tau_s=0., delay_controls=0, offset=0., scale=1., **review)],
        motor={role:dict(ids=[graph.nodes[-i-1]['id']], gain=.01,
                        output_side=role[4:] if role.startswith('yaw_') else None, **review)
               for i,role in enumerate(('forward','backward','stop','yaw_left','yaw_right'))})
    return graph, PortBindings(graph, spec)


class ModelContractTests(unittest.TestCase):
    def test_doctor_uses_the_same_resolved_model(self):
        import contextlib
        import io
        from flylab.c.server import main
        graph = graph_fixture(); spec = copy.deepcopy(bindings_fixture(graph).spec)
        spec['neural_parameters'] = dict(rest_mV=-60.,reset_mV=-60.,threshold_mV=-30.)
        binding = PortBindings(graph,spec); stream = io.StringIO()
        with patch('sys.argv',['run_c.py','--doctor','--mode','C_STRICT','--backend','exp_lif_mps']), \
             patch('flylab.c.server.GraphStore.load',return_value=graph), \
             patch('flylab.c.server.read_json',return_value=spec), \
             patch('flylab.c.server.dependency_report',return_value={'ready':True}), \
             patch('flylab.c.server.create_backend') as backend, contextlib.redirect_stdout(stream):
            self.assertEqual(main(),0)
        parameters = backend.call_args.args[1]
        self.assertEqual(parameters.hash,resolve_parameters(binding).hash)
        self.assertEqual(json.loads(stream.getvalue())['model']['parameter_hash'],parameters.hash)

    def test_verifier_uses_engine_parameters_and_actual_dt(self):
        graph = graph_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);graph.save(root/'graph')
            for i, params in enumerate(({'rest_mV':-60.,'reset_mV':-60.,'threshold_mV':-30.},
                                       {'dt':.00005,'integration':'exact-exponential-reset-current-v1'})):
                with self.subTest(parameters=params):
                    spec = copy.deepcopy(bindings_fixture(graph).spec);spec['neural_parameters']=params
                    binding = PortBindings(graph, spec);path=root/f'binding-{i}.json';path.write_text(json.dumps(spec))
                    proc = subprocess.run([sys.executable,str(ROOT/'tools/verify_c.py'),'--scope','loaded-graph',
                        '--graph',str(root/'graph'),'--bindings',str(path),'--out',str(root/f'out-{i}')],
                        cwd=ROOT,capture_output=True,text=True,timeout=60)
                    self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                    gate=json.loads((root/f'out-{i}/report.json').read_text())['gates']['loaded_graph_compute']
                    engine=CEngine(graph,binding,mode='C_STRICT',body_factory=FixtureBody)
                    try:
                        ticks=model_ticks(.05,engine.parameters.dt);drive=np.zeros(graph.n,np.float32);drive[binding.motor_indices]=12.
                        engine.neural.advance(drive,ticks,binding.motor_indices)
                        self.assertEqual(gate['summary'],engine.neural.summary())
                        self.assertEqual(gate['model']['parameter_hash'],engine.provenance()['model_parameter_hash'])
                        self.assertEqual(gate['computed_ticks'],ticks)
                        self.assertEqual(gate['model_seconds'],ticks*engine.parameters.dt)
                    finally:engine.close()

    def test_parameter_override_precedence_and_fractional_clock_rejection(self):
        graph=graph_fixture();spec=copy.deepcopy(bindings_fixture(graph).spec)
        spec['neural_parameters']={'rest_mV':-60.,'reset_mV':-60.,'threshold_mV':-30.}
        binding=PortBindings(graph,spec)
        self.assertEqual(resolve_parameters(binding,{'threshold_mV':-25.}).rest_mV,-60.)
        self.assertEqual(resolve_parameters(binding,{'threshold_mV':-25.}).threshold_mV,-25.)
        self.assertEqual(resolve_parameters(binding,LIFParameters()).rest_mV,-52.)
        with self.assertRaises(ValueError):model_ticks(.00015,.0001)

    def test_banc_default_pilot_and_explicit_incompatible_modes(self):
        graph,binding=banc_bindings()
        self.assertEqual(execution_capabilities(binding)['modes'],['C_STRICT'])
        default=pilot_spec(bindings=binding)
        self.assertEqual(len(default['cases']),18)
        validate_spec(default,binding)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'bad';jobs=CampaignJobs(Path(tmp)/'jobs',Path(tmp)/'graph')
            for mode in ('C_SHADOW','C_ASSISTED','B_COMPAT'):
                spec=pilot_spec(.005,seeds=(7,),modes=(mode,))
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(ValueError,'Campaign case.*requires C_STRICT'):
                        run_campaign(graph,binding,spec,out,body_factory=FixtureBody)
                    self.assertFalse(out.exists())
                    with patch.object(jobs,'_launch') as launch:
                        with self.assertRaisesRegex(ValueError,'requires C_STRICT'):jobs.start(binding,'exp_lif_cpu_reference',spec)
                        launch.assert_not_called()
                    self.assertFalse(jobs.root.exists())
            with patch.object(jobs,'_launch') as launch:
                with self.assertRaises(ValueError):jobs.start(binding,'exp_lif_cpu_reference',{})
                launch.assert_not_called()

    def test_server_declares_profile_modes_and_rejects_before_saving_live_state(self):
        from flylab.c.server import CDispatcher
        from flylab.c import PROTOCOL
        from tests.test_c_sensorimotor import LegBody
        graph,binding=banc_bindings()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);graph.save(root/'graph');(root/'bindings.json').write_text(json.dumps(binding.spec))
            d=CDispatcher(root/'graph',root/'bindings.json',root/'artifacts',body_factory=LegBody,default_mode='C_STRICT')
            try:
                ready=d.handle(dict(protocol=PROTOCOL,requestId=1,op='init',payload={}))
                # Dispatcher wraps responses in its protocol envelope.
                data=ready.get('result',ready)
                self.assertEqual(data['capabilities']['modes'],['C_STRICT'])
                self.assertEqual(data['profiles'][0]['execution']['pilot_modes'],['C_STRICT'])
                before=d.engine.checkpoint()
                with self.assertRaisesRegex(ValueError,'requires C_STRICT'):
                    d.handle(dict(protocol=PROTOCOL,requestId=2,op='init',payload={'mode':'C_SHADOW'}))
                same_state(before,d.engine.checkpoint())
                self.assertFalse((root/'artifacts/checkpoints').exists())
            finally:
                if d.engine:d.engine.close()


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Optional Torch required; install requirements-research.txt')
class ResearchContractTests(unittest.TestCase):
    def setUp(self):
        from flylab.c.research_neural import ResearchLIF
        self.factory=ResearchLIF;self.graph=graph_fixture()
        self.spec=dict(schema='flylab.physiology.v1',graph_hash=self.graph.hash,
                       evidence=['SYNTHETIC TEST'],uncertainty='No physiological claim')

    def compare(self,research,reference,tolerance=1e-4):
        actual=research.snapshot()['arrays'];expected=reference.snapshot()
        self.assertEqual(research.last_events,reference.last_events)
        for name in ('v','h','rate','queue','spike_count','suppress','mute'):
            np.testing.assert_allclose(actual[name],expected[name],atol=tolerance,rtol=0,err_msg=name)
        np.testing.assert_array_equal(actual['refractory'],expected['refractory_until'])

    def check_integrations(self,device):
        for scheme in ('exact-exponential-held-drive-v1','exact-exponential-reset-current-v1','exact-exponential-voltage-events-v1'):
            with self.subTest(integration=scheme,device=device):
                spec=dict(self.spec,parameters={'integration':scheme})
                actual=self.factory(self.graph,spec,device);expected=ExpLIF(self.graph,LIFParameters(integration=scheme))
                actual.h[:]=50.;expected.h[:]=50.;drive=np.full(self.graph.n,20.,np.float32)
                actual.advance(drive,30,range(self.graph.n));expected.advance(drive,30,range(self.graph.n));self.compare(actual,expected)
                pi=np.array([0,0,1],np.int32);pv=np.zeros((80,3),np.float32);pv[::4]=[8.,9.,20.]
                for n in (actual,expected):
                    n.set_interventions(suppress=[2],mute=[0],edges=[1])
                    n.advance(drive,80,range(self.graph.n),(pi,pv))
                self.compare(actual,expected,1e-3 if device=='mps' else 1e-4)
                for n in (actual,expected):n.set_interventions();n.advance(np.zeros(self.graph.n,np.float32),70,range(self.graph.n))
                self.compare(actual,expected,1e-3 if device=='mps' else 1e-4)

    def test_all_integrations_pulses_refractory_and_interventions_cpu(self):
        self.check_integrations('cpu')

    @unittest.skipUnless(mps_available(),'Actual MPS required')
    def test_all_integrations_pulses_refractory_and_interventions_mps(self):
        self.check_integrations('mps')

    def test_common_api_and_invalid_input_are_atomic(self):
        n=self.factory(self.graph,self.spec);self.assertIsInstance(n,NeuralBackend)
        self.assertIsInstance(ExpLIF(self.graph),NeuralBackend)
        before=n.snapshot();drive=np.zeros(self.graph.n,np.float32)
        for call in (lambda:n.advance(drive,10,pulses=(np.array([0]),np.zeros((9,1)))),
                     lambda:n.advance(drive,10,reward=2.),lambda:n.set_interventions(edges=[99999])):
            with self.assertRaises(ValueError):call()
            same_state(before,n.snapshot())
        self.assertEqual(n.summary()['simulated_node_count'],self.graph.n)
        self.assertEqual(n.region_summary({'one':[0]})[0]['mean_rate_Hz'],0.)

    def plastic(self):
        ids=[n['id'] for n in self.graph.nodes]
        return self.factory(self.graph,dict(self.spec,plasticity=dict(rule='reward_stdp_v1',eta=.01,
            tau_s=.02,max_abs_mV=10.,edges=[dict(pre=ids[0],post=ids[1])],evidence=['SYNTHETIC TEST'])))

    def test_unreachable_state_rejected_without_partial_application(self):
        n=self.plastic();before=n.snapshot()
        for name,index,value in (('spike_count',0,100),('refractory',0,-1),('refractory',0,23),
                                 ('rate',0,10002.),('pre_trace',0,-1.),('post_trace',0,100.),('v',0,1e8)):
            with self.subTest(field=name,value=value):
                bad=copy.deepcopy(before);bad['arrays'][name][index]=value
                with self.assertRaises(ValueError):n.restore(bad)
                same_state(before,n.snapshot())
        bad=copy.deepcopy(before);bad['schema']='flylab.research-neural-state.v1'
        with self.assertRaisesRegex(ValueError,'archived runtime'):n.restore(bad)
        same_state(before,n.snapshot())

    def test_device_copy_failure_preserves_old_state_and_good_restore_continues(self):
        n=self.plastic();drive=np.full(self.graph.n,20.,np.float32);n.advance(drive,70,reward=1.)
        saved=n.snapshot();n.advance(drive,40,reward=.5);before=n.snapshot();tensor=n.tensor;calls=[0]
        def fail(a,*args):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('simulated allocation failure')
            return tensor(a,*args)
        with patch.object(n,'tensor',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'allocation'):n.restore(saved)
        same_state(before,n.snapshot())
        n.restore(saved);n.advance(drive,40,reward=.5);same_state(before,n.snapshot())

    def test_equal_and_near_equal_time_constants_have_finite_analytic_limit(self):
        for ts in (.02,.020000000001,.019999999999):
            n=self.factory(self.graph,dict(self.spec,parameters={'tau_m_s':.02,'tau_syn_s':ts}))
            n.h[:]=1.;drive=np.zeros(self.graph.n,np.float32);n.advance(drive,1)
            expected=-52.+.0001/.02*np.exp(-.0001/.02)
            np.testing.assert_allclose(n.snapshot()['arrays']['v'],expected,atol=4e-6,rtol=0)
        # The allowed dt/tau ratio can reach 5000. Computing exp(+5000)
        # and multiplying by exp(-5000) would manufacture NaN from a finite solution.
        for tm,ts in ((1e-6,1.),(1.,1e-6),(1e-6,1e-6)):
            parameters=dict(dt=.005,delay_s=.005,refractory_s=.005,tau_m_s=tm,tau_syn_s=ts)
            n=self.factory(self.graph,dict(self.spec,parameters=parameters))
            reference=ExpLIF(self.graph,LIFParameters(**parameters))
            n.h[:]=1.;reference.h[:]=1.;drive=np.zeros(self.graph.n,np.float32)
            n.advance(drive,1);reference.advance(drive,1)
            np.testing.assert_allclose(n.snapshot()['arrays']['v'],reference.v,atol=4e-6,rtol=0)
