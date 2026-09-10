"""Equivalence checks for cached physical control and batched MPS observation."""
import copy
import unittest
import numpy as np
from flylab.body import FlyGymBody, dependency_report
from flylab.engine import config_values
from flylab.sensors import default_world, SensorAdapter
from flylab.c.engine import CEngine
from flylab.c.neural import create_backend
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state
from tests.test_c_mps import mps_available


@unittest.skipUnless(dependency_report()['ready'], 'Pinned physical runtime required')
class PhysicalOptimizationTests(unittest.TestCase):
    def test_compiled_splines_match_knots_periods_and_extreme_phases(self):
        from flylab._spline import BatchedSplines
        from flygym_demo.complex_terrain import PreprogrammedSteps
        steps=PreprogrammedSteps();fast=BatchedSplines(steps,steps.legs)
        rng=np.random.default_rng(17)
        phases=np.concatenate([rng.uniform(-1e4,1e4,(150,6)),
                               np.tile(fast.x[:,None],(1,6)),np.full((1,6),-2*np.pi)])
        for x in phases:
            expected=np.array([steps._psi_funcs[leg](v) for leg,v in zip(steps.legs,x)])
            np.testing.assert_array_equal(fast.evaluate(x),expected)

    def test_cached_controller_preserves_reflexes_and_reverse_cpg(self):
        from flylab.locomotion import CachedHybridStepper
        from flygym_demo.complex_terrain import HybridTurningController, HybridControllerObservation,PreprogrammedSteps
        from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
        a=HybridTurningController(.0001,preprogrammed_steps=PreprogrammedSteps(),output_dof_order=get_default_locomotion_dof_order())
        b=copy.deepcopy(a);a.reset(seed=83);b.reset(seed=83)
        fast=CachedHybridStepper(b);rng=np.random.default_rng(61)
        for i in range(600):
            if i in (200,400):
                a.enable_adhesion=b.enable_adhesion=(i==400)
                a.swing_extension=b.swing_extension=np.pi/(8 if i==200 else 4)
            heights=np.full(6,.1);heights[(i//45)%6]=-.8
            forces=rng.normal(0,2,(6,3,3))
            obs=HybridControllerObservation(.9,heights,forces,np.array([1.,0.,0.]))
            signal=np.array([[1.2,.4],[-.5,.6],[0.,0.],[-.7,-.9]][(i//75)%4])
            x=a.step(signal,obs);y=fast.step(signal,obs)
            np.testing.assert_array_equal(x.joint_angles,y.joint_angles)
            np.testing.assert_array_equal(x.adhesion_onoff,y.adhesion_onoff)
            for key in ('retraction_correction','stumbling_correction','retraction_persistence_counter'):
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            for key in ('curr_phases','curr_magnitudes'):
                np.testing.assert_array_equal(getattr(a.cpg_network,key),getattr(b.cpg_network,key))
            for key in ('net_corrections','retraction_correction','stumbling_correction','stumbling_mask'):
                np.testing.assert_array_equal(a.last_info[key],b.last_info[key])

    def test_native_body_contact_sensor_and_checkpoint_equivalence(self):
        world=default_world();config=config_values()
        a=FlyGymBody(42,world,config,optimized=False);b=FlyGymBody(42,world,config)
        try:
            b.restore(a.snapshot());sa=SensorAdapter(42);sb=SensorAdapter(42)
            for i in range(50):
                command=dict(forwardSpeed=[3.7,-1.5,0.,0.,3.][i//10],yawRate=[.8,-1.2,3.,0.,-.5][i//10],verticalSpeed=0.)
                if i==35:a.perturb();b.perturb()
                a.step(command);b.step(command)
                self.assertEqual(a.snapshot(),b.snapshot())
                self.assertEqual(sa.observe(a,world,.005,config),sb.observe(b,world,.005,config))
                np.testing.assert_array_equal(b._contact_forces('feet'),a.sim.get_bodysegment_contact_forces(
                    a.fly.name,[a.BodySegment(f'{leg}_{link}') for leg in ('lf','lm','lh','rf','rm','rh')
                                for link in ('tarsus1','tarsus2','tarsus3','tarsus4','tarsus5')],ground_only=True))
            saved=b.snapshot();b.step(command);after=b.snapshot();b.restore(saved);b.step(command)
            self.assertEqual(after,b.snapshot())
            rng=np.random.default_rng(471)
            for _ in range(100):
                origin=rng.uniform([-23,-17,.1],[23,17,4.])
                theta=rng.uniform(-np.pi,np.pi)
                r=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
                x=a.sensor_rays(origin,r);y=b.sensor_rays(origin,r)
                np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        finally:a.close();b.close()


class _Synchronous:
    def __init__(self,backend):self.wrapped=backend
    def __getattr__(self,name):
        if name=='begin_advance':raise AttributeError(name)
        return getattr(self.wrapped,name)


@unittest.skipUnless(mps_available(), 'Real Apple MPS required')
class PackedMPSObservationTests(unittest.TestCase):
    def test_packed_arrays_keep_large_integer_counts_and_partial_blocks(self):
        g=graph_fixture(n=513);b=create_backend(g,backend='exp_lif_mps')
        counts=np.arange(g.n,dtype=np.int64)+2**40
        b.spike_count[:]=b.xp.asarray(counts)
        b.v[:]=b.xp.asarray(np.linspace(-77,-46,g.n,dtype=np.float32))
        b.rate[:]=b.xp.asarray(np.linspace(0,77,g.n,dtype=np.float32))
        b.advance(np.zeros(g.n,np.float32),0,[512,0,256,512])
        for ids in ([512,0,256,512],[511,2,3],[]):
            result=b.readout(ids)
            np.testing.assert_array_equal(result['spike_count'],counts[list(ids)])
            np.testing.assert_array_equal(result['voltage_mV'],b.host(b.v)[list(ids)])
        summary=b.summary()
        self.assertEqual(summary['cumulative_spikes'],int(counts.sum()))
        self.assertAlmostEqual(summary['mean_rate_Hz'],float(b.host(b.rate).mean()),places=5)
        saved=b.snapshot();saved['v']=saved['v']-2
        # The oversized counters above test binary packing, not a reachable
        # tick-zero checkpoint. Restore a valid state to test cache invalidation.
        saved['spike_count'].fill(0);b.restore(saved)
        np.testing.assert_array_equal(b.readout([512,0])['voltage_mV'],saved['v'][[512,0]])

    def test_rejected_begin_does_not_corrupt_cached_readout(self):
        g=graph_fixture();b=create_backend(g,backend='exp_lif_mps')
        b.advance(np.arange(g.n,dtype=np.float32)*20,100,[0,1])
        expected=b.host(b.v).copy();tick=b.tick
        with self.assertRaises(ValueError):
            b.begin_advance(np.zeros(g.n),10,[4,5],(np.array([1]),np.ones((9,1))))
        self.assertEqual(tick,b.tick)
        np.testing.assert_array_equal(b.readout([4,5])['voltage_mV'],expected[[4,5]])

    def test_async_boundary_requires_completion(self):
        g=graph_fixture();b=create_backend(g,backend='exp_lif_mps');drive=np.full(g.n,30.,np.float32)
        b.begin_advance(drive,50,[0,1])
        for call in (b.snapshot,b.summary,lambda:b.readout([0]),lambda:b.begin_advance(drive,50),lambda:b.set_interventions([0])):
            with self.assertRaises(RuntimeError):call()
        b.finish_advance();self.assertTrue(b.submitted_event.query())
        self.assertEqual(b.tick,50)

    def test_body_failure_drains_the_pending_gpu_period(self):
        class BrokenBody(FixtureBody):
            def step(self,*args,**kwargs):raise RuntimeError('injected physical failure')
        g=graph_fixture();e=CEngine(g,bindings_fixture(g),backend='exp_lif_mps',body_factory=BrokenBody)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected physical failure'):e.step()
            self.assertIsNone(e.neural._pending_advance)
        finally:e.close()

    def test_overlapped_execution_keeps_closed_loop_causality(self):
        for mode in ('C_SHADOW','C_STRICT','C_ASSISTED'):
            g=graph_fixture();bindings=bindings_fixture(g)
            initial=CEngine(g,bindings,mode=mode,backend='exp_lif_mps',body_factory=FixtureBody)
            state=initial.checkpoint();initial.close()
            a=CEngine.from_checkpoint(g,bindings,state,FixtureBody)
            b=CEngine.from_checkpoint(g,bindings,state,FixtureBody)
            b.neural=_Synchronous(b.neural)
            try:
                for e in (a,b):e.schedule(dict(kind='stimulate',ids=[g.nodes[1]['id']],amplitude_mV=30.,duration_controls=10))
                for _ in range(20):a.step();b.step()
                same_state(a.checkpoint(),b.checkpoint())
            finally:a.close();b.close()


if __name__=='__main__':unittest.main()
