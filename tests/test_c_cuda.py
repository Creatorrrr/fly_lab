"""Actual NVIDIA GPU checks, including CUDA graph replay and CPU parity."""
import copy
import importlib.util
import unittest
import numpy as np
from flylab.c.neural import ExpLIF, LIFParameters, create_backend
from flylab.c.engine import CEngine
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


def cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(cuda_available(), 'Actual NVIDIA CUDA hardware and CuPy required')
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.cpu = ExpLIF(self.graph)
        self.gpu = create_backend(self.graph, backend='exp_lif_cuda')
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
        other=create_backend(self.graph,backend='exp_lif_cuda');other.restore(saved)
        self.gpu.advance(self.drive,250,self.capture);other.advance(self.drive,250,self.capture)
        same_state(self.gpu.snapshot(),other.snapshot())
        self.assertEqual(self.gpu.last_events,other.last_events)

    def test_reused_pulses_preserve_duplicate_order_and_changing_cohorts(self):
        rng = np.random.default_rng(162)
        for ids, steps, capture in [([2,0,2],50,[0,2,0]), ([2,0,2],50,[2]),
                                    ([0,2,2],30,[1,3]), ([4],25,[]), ([],0,[0,5]),
                                    ([2,0,2],50,[5,0,2])]:
            self.capture = np.asarray(capture, np.int32)
            pulses = (np.array(ids, np.int32), rng.uniform(0,15,(steps,len(ids))))
            self.run_both(steps, pulses=pulses)

    def test_async_upload_owns_inputs_and_old_readouts(self):
        ids = np.array([2,0,2], np.int32)
        values = np.full((50,3), 5., np.float32)
        drive = self.drive.copy()
        self.cpu.advance(drive,50,self.capture,(ids,values))
        self.gpu.begin_advance(drive,50,self.capture,(ids,values))
        drive.fill(900.); values.fill(900.); ids[:]=4
        self.gpu.finish_advance(); self.check_equal()
        readout = self.gpu.readout(self.capture)
        saved = copy.deepcopy(readout)
        self.run_both(50)
        same_state(readout,saved)

    def test_cached_interventions_keep_validation_and_restore_masks(self):
        self.cpu.set_interventions([0]); self.gpu.set_interventions([0])
        self.run_both(50)
        for invalid in ([False], [np.int64(0)]):
            before = self.gpu.snapshot()
            with self.assertRaises(ValueError): self.gpu.set_interventions(invalid)
            same_state(before,self.gpu.snapshot())
        saved = self.gpu.snapshot()
        self.gpu.set_interventions(); self.gpu.set_interventions()
        self.gpu.restore(saved)
        self.gpu.set_interventions([0]); self.run_both(50)

    def test_runtime_mismatch_and_float64_fail_explicitly(self):
        state=self.gpu.snapshot();state['backend_runtime']['shader_sha256']='bad'
        with self.assertRaises(ValueError):create_backend(self.graph,backend='exp_lif_cuda').restore(state)
        from flylab.c.neural_cuda import CudaLIF
        with self.assertRaises(ValueError):CudaLIF(self.graph,LIFParameters(dtype='float64'))

    def test_backend_transfer_preserves_state_and_source_engine(self):
        bindings=bindings_fixture(self.graph)
        source=CEngine(self.graph,bindings,mode='C_STRICT',body_factory=FixtureBody)
        try:
            source.schedule(dict(kind='stimulate',ids=[self.graph.nodes[0]['id']],amplitude_mV=30.))
            source.step(10);saved=source.checkpoint()
            target=CEngine.from_checkpoint(self.graph,bindings,saved,FixtureBody,backend_override='exp_lif_cuda')
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


@unittest.skipUnless(cuda_available(), 'Actual NVIDIA CUDA hardware required')
class ResearchCudaTests(unittest.TestCase):
    def test_all_integrations_pulses_refractory_and_interventions(self):
        from tests.test_c_contracts import ResearchContractTests
        helper=ResearchContractTests(); helper.setUp(); helper.check_integrations('cuda')

    def test_heterogeneity_delays_gaps_plasticity_and_restore(self):
        from tests.test_c_research import ResearchTests
        helper=ResearchTests(); helper.setUp(); helper.check_heterogeneity('cuda')

    def test_all_baseline_integrations_and_float64(self):
        for scheme in ('exact-exponential-held-drive-v1','exact-exponential-reset-current-v1','exact-exponential-voltage-events-v1'):
            for dtype in ('float32','float64'):
                with self.subTest(scheme=scheme,dtype=dtype):
                    g=graph_fixture(); p=LIFParameters(dtype=dtype,integration=scheme)
                    cpu=ExpLIF(g,p); gpu=create_backend(g,p,'exp_lif_cuda')
                    drive=np.full(g.n,25.,dtype); ids=np.array([0,0,2],np.int32)
                    pulses=np.full((100,3),.1,dtype)
                    for _ in range(2):
                        cpu.advance(drive,100,[0,1,0],(ids,pulses)); gpu.advance(drive,100,[0,1,0],(ids,pulses))
                        self.assertEqual(cpu.last_events,gpu.last_events)
                        for k in ('v','h','queue','rate','spike_count'):
                            np.testing.assert_allclose(cpu.snapshot()[k],gpu.snapshot()[k],atol=1e-4,rtol=0,err_msg=k)


@unittest.skipUnless(cuda_available(), 'Actual NVIDIA CUDA hardware required')
class CudaObservationTests(unittest.TestCase):
    def test_packed_large_counts_and_partial_blocks(self):
        g=graph_fixture(n=513); b=create_backend(g,backend='exp_lif_cuda')
        counts=np.arange(g.n,dtype=np.int64)+2**40
        # Mutations, like production uploads, use the owned nonblocking stream.
        with b.stream:
            b.spike_count[:]=b.xp.asarray(counts)
            b.rate[:]=b.xp.asarray(np.linspace(0,77,g.n,dtype=np.float32))
        b.advance(np.zeros(g.n,np.float32),0,[512,0,256,512])
        for ids in ([512,0,256,512],[511,2,3],[]):
            np.testing.assert_array_equal(b.readout(ids)['spike_count'],counts[list(ids)])
        self.assertEqual(b.summary()['cumulative_spikes'],int(counts.sum()))
        self.assertAlmostEqual(b.summary()['mean_rate_Hz'],float(b.host(b.rate).mean()),places=5)

    def test_body_failure_drains_pending_cuda_period(self):
        class BrokenBody(FixtureBody):
            def step(self,*args,**kwargs):raise RuntimeError('injected physical failure')
        g=graph_fixture(); e=CEngine(g,bindings_fixture(g),backend='exp_lif_cuda',body_factory=BrokenBody)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected physical failure'):e.step()
            self.assertIsNone(e.neural._pending_advance)
            e.neural.snapshot()
        finally:e.close()

    @unittest.skipUnless(importlib.util.find_spec('mujoco') and importlib.util.find_spec('flygym'),
                         'Native FlyGym/MuJoCo required')
    def test_single_joint_physics_and_checkpoint_match_cpu(self):
        from tests.test_c_single_joint import SingleJointTests
        helper=SingleJointTests(); helper.setUp(); helper.check_gpu_physics('exp_lif_cuda')


if __name__=='__main__':unittest.main()
