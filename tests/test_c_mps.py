"""Actual Apple GPU checks. Skipped explicitly on hosts without MPS."""
import copy
import importlib.util
import unittest
import numpy as np
from flylab.c.neural import ExpLIF, LIFParameters, create_backend
from flylab.c.engine import CEngine
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


def mps_available():
    if importlib.util.find_spec('torch') is None: return False
    import torch
    return torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')


@unittest.skipUnless(mps_available(), 'Apple MPS hardware and torch compile_shader required')
class MetalTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.cpu = ExpLIF(self.graph)
        self.gpu = create_backend(self.graph, backend='exp_lif_mps')
        self.drive = np.array([30., 12., 8., 0., 14., 0.], dtype=np.float32)
        self.capture = np.arange(self.graph.n, dtype=np.int32)

    def check_equal(self, events=True):
        a,b = self.cpu.snapshot(),self.gpu.snapshot()
        for key in ('v','h','queue'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-4, rtol=0, err_msg=key)
        np.testing.assert_allclose(a['rate'],b['rate'],atol=1e-3,rtol=0)
        for key in ('spike_count','refractory_until','suppress','mute'):
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        self.assertEqual(a['tick'],b['tick']); self.assertEqual(a['slot'],b['slot'])
        if events:self.assertEqual(self.cpu.last_events,self.gpu.last_events)

    def run_both(self, steps, **kwargs):
        self.cpu.advance(self.drive, steps, capture=self.capture, **kwargs)
        self.gpu.advance(self.drive, steps, capture=self.capture, **kwargs)
        self.check_equal()

    def test_refractory_delay_and_continuous_state_match_reference(self):
        for _ in range(20): self.run_both(50)

    def test_inflight_events_survive_mute_and_suppress(self):
        self.run_both(80)
        for suppress,mute,edges in [([0],[],[]),([],[1],[0]),([2],[0],[1]),([],[],[])]:
            self.cpu.set_interventions(suppress,mute,edges)
            self.gpu.set_interventions(suppress,mute,edges)
            self.run_both(100)

    def test_repeated_pulse_targets_keep_addition_order(self):
        rng=np.random.default_rng(271)
        pi=np.array([2,0,2,1,2],dtype=np.int32)
        values=rng.uniform(-4,8,size=(300,len(pi))).astype(np.float32)
        self.run_both(300,pulses=(pi,values))

    def test_duplicate_and_changed_display_subscription(self):
        self.capture=np.array([2,0,2,0],dtype=np.int32);self.run_both(100)
        self.capture=np.array([],dtype=np.int32);self.run_both(50)
        self.capture=np.array([1,3],dtype=np.int32);self.run_both(100)

    def test_gpu_checkpoint_continuation_is_exact(self):
        self.gpu.advance(self.drive,100,self.capture)
        self.gpu.set_interventions(mute=[0],edges=[1]);saved=self.gpu.snapshot()
        other=create_backend(self.graph,backend='exp_lif_mps');other.restore(saved)
        self.gpu.advance(self.drive,250,self.capture);other.advance(self.drive,250,self.capture)
        same_state(self.gpu.snapshot(),other.snapshot())
        self.assertEqual(self.gpu.last_events,other.last_events)

    def test_runtime_mismatch_and_float64_fail_explicitly(self):
        state=self.gpu.snapshot();state['backend_runtime']['shader_sha256']='bad'
        with self.assertRaises(ValueError):create_backend(self.graph,backend='exp_lif_mps').restore(state)
        with self.assertRaises(ValueError):create_backend(self.graph,LIFParameters(dtype='float64'),backend='exp_lif_mps')

    def test_backend_transfer_preserves_state_and_source_engine(self):
        bindings=bindings_fixture(self.graph)
        source=CEngine(self.graph,bindings,mode='C_STRICT',body_factory=FixtureBody)
        try:
            source.schedule(dict(kind='stimulate',ids=[self.graph.nodes[0]['id']],amplitude_mV=30.))
            source.step(10);saved=source.checkpoint()
            target=CEngine.from_checkpoint(self.graph,bindings,saved,FixtureBody,backend_override='exp_lif_mps')
            try:
                same_state(source.checkpoint(),saved)
                for key in ('v','h','queue','rate','spike_count','refractory_until'):
                    np.testing.assert_array_equal(saved['neural'][key],target.neural.snapshot()[key])
                same_state(source.body.snapshot(),target.body.snapshot())
                self.assertEqual(target.frame()['performance']['measured_model_s'],0.)
                target.step(2)
                self.assertEqual(target.frame()['performance']['measured_model_s'],.01)
            finally:target.close()
        finally:source.close()

    def test_malformed_pulses_do_not_advance_clock(self):
        for pulses in [(np.array([0.5]),np.zeros((3,1))),
                       (np.array([self.graph.n]),np.zeros((3,1))),
                       (np.array([0]),np.full((3,1),np.nan))]:
            with self.assertRaises(ValueError):self.gpu.advance(self.drive,3,pulses=pulses)
            self.assertEqual(self.gpu.tick,0)


if __name__=='__main__':unittest.main()
