import copy
import math
import unittest
import numpy as np
from flylab.c.neuromuscular import NeuromuscularLoop, build_spec, typed_receptor_feature
from flylab.c.neural import ExpLIF
from tests.test_c_sensorimotor import banc_fixture, LegBody
from tests.test_c import same_state
from flylab.sensors import default_world
from flylab.engine import config_values
from flylab.c.ports import PortBindings,MotorDecoder,SensoryEncoder
from tests.c_fixtures import graph_fixture,bindings_fixture


class ContactBody(LegBody):
    def leg_observation(self):
        return dict(super().leg_observation(),support_load_bw=self.loads.copy(),
                    non_support_load_bw=np.zeros(6),adhesion_force_bw=np.zeros(6),
                    floor_normal_load_bw=self.loads.copy())


class RepairTests(unittest.TestCase):
    def test_obstacle_pass_requires_matched_opportunity_and_sensory_effect(self):
        from tools.assess_c_obstacles import assess,CHANNELS
        from flylab.c.tasks import evaluate
        from flylab.c.behavior import trace_sample
        from flylab.c.campaign import scene_world
        from flylab.c.engine import CEngine
        from tests.fixture_body import FixtureBody
        g=graph_fixture();e=CEngine(g,bindings_fixture(g),mode='C_STRICT',body_factory=FixtureBody)
        try:first=trace_sample(e.frame());provenance=e.provenance()
        finally:e.close()
        provenance['bindings']['sensory']=[dict(name=c,channel=c) for c in CHANNELS]
        world=scene_world('front_obstacle')
        def arm(avoid=False,off=False,free=False):
            rows=[]
            for i in range(2001):
                t=i*.005;r=copy.deepcopy(first)
                r.update(simTime=t,tick=i*50,position=[t,.9,3*math.sin(math.pi*t/10) if avoid else 0.],
                    obstacle_engaged=1.<=t<4.,sensory_ports=[dict(name=c,value=0. if off else 1.,enabled=not off) for c in CHANNELS])
                rows.append(r)
            w=copy.deepcopy(world)
            if free:w['obstacles']=[]
            return dict(trace=rows,world=w,provenance=copy.deepcopy(provenance),interventions=[dict(kind='sensor_off',channels=CHANNELS,duration_controls=2000)] if off else [])
        active,free,off=arm(True),arm(free=True),arm(off=True)
        single=evaluate(active['trace'],world,'obstacle',required_seconds=10.)
        self.assertEqual(single['trajectory_status'],'PASS');self.assertEqual(single['task_status'],'NOT_EVALUATED')
        self.assertEqual(assess(active,free,off)['task_status'],'PASS')
        off['trace'][100]['sensory_ports'][0]['enabled']=True
        self.assertEqual(assess(active,free,off)['task_status'],'INCOMPLETE')
        off['trace'][100]['sensory_ports'][0]['enabled']=False
        for a,b in zip(free['trace'],active['trace']):a['position']=b['position']
        self.assertEqual(assess(active,free,off)['task_status'],'NOT_APPLICABLE')

    def test_visual_contact_units_are_independent_of_poisson_template(self):
        from tools.build_c_sensor_profile import build
        from tools.probe_c_steering import packet
        g=graph_fixture(n=9)
        for i,side in ((6,'left'),(7,'right')):
            g.nodes[i].update(cell_type='LC4',soma_side=side,super_class='visual_projection')
        g.nodes[8].update(cell_type='BM_Fr',super_class='sensory',**{'class':'mechanosensory'})
        base=bindings_fixture(g,'poisson_Hz').spec;base['sensory'][0].update(baseline=10.,offset=.5,scale=7.)
        spec=build(g,base);b=PortBindings(g,spec)
        for p in spec['sensory'][1:]:
            self.assertEqual(p['method'],'drive_mV');self.assertNotIn('pulse_mV',p)
            self.assertEqual((p['baseline'],p['offset'],p['scale']),(0.,0.,1.))
        e=SensoryEncoder(b,42)
        for _ in range(30):drive,_,ports=e.encode(packet(0.,0.),.005,.0001,supplemental=dict(loom_left=1.,loom_right=0.,head_contact=1.))
        self.assertGreater(drive[6],7.9);self.assertEqual(drive[7],0.);self.assertGreater(drive[8],17.9)
        self.assertTrue(all(p['unit']=='mV' for p in ports[1:]))

    def test_walk_off_gates_only_positive_forward_and_cannot_create_reverse(self):
        g=graph_fixture();spec=bindings_fixture(g).spec
        spec['motor_decoder']=dict(kind='bounded-walk-off-v2',speed_limit=1.,yaw_limit=1.5,stop_full_drive=1.)
        spec['motor']['stop']['gain']=.02;b=PortBindings(g,spec);n=ExpLIF(g);decoder=MotorDecoder(b)
        n.rate[1]=10.;n.rate[2]=2.;n.rate[5]=3.
        before=decoder.decode(n)[0];n.rate[3]=50.;after=decoder.decode(n)[0]
        self.assertGreater(before['forwardSpeed'],0.);self.assertEqual(after['forwardSpeed'],0.)
        self.assertEqual(before['yawRate'],after['yawRate'])
        n.rate[1]=0.;self.assertLess(decoder.decode(n)[0]['forwardSpeed'],0.)
        n.rate[3]=0.;self.assertEqual(decoder.decode(n)[0]['forwardSpeed'],after['forwardSpeed']-.16)

    def test_research_cohorts_reject_unknown_ids_and_duplicate_names(self):
        g=graph_fixture();base=bindings_fixture(g).spec
        c=dict(base['motor']['forward'],name='test-reviewed');base['research_cohorts']=[c]
        self.assertEqual(len(PortBindings(g,base).research_cohorts),1)
        bad=copy.deepcopy(base);bad['research_cohorts']*=2
        with self.assertRaises(ValueError):PortBindings(g,bad)
        bad=copy.deepcopy(base);bad['research_cohorts'][0]['ids']=['invalid']
        with self.assertRaises(ValueError):PortBindings(g,bad)

    def test_assay_geometry_and_halting_interventions_are_explicit(self):
        from tools.prepare_c_control_assays import build
        from tools.prepare_c_repair_assays import build as repaired
        g=graph_fixture();spec=bindings_fixture(g).spec
        spec['research_cohorts']=[dict(spec['motor']['stop'],name='Foxglove test')]
        b=PortBindings(g,spec)
        forward=build(b)['cases'][0]
        self.assertEqual(forward['seconds'],10.);self.assertLess(forward['initial_pose']['position'][0],-20.)
        halt=repaired(b,'halting')['cases'];self.assertEqual(len(halt),4)
        self.assertTrue(any(e['kind']=='mute_outgoing' for e in halt[-1]['interventions']))
        active,free,off=repaired(b,'hazard')['cases']
        self.assertEqual(active['interventions'],free['interventions'])
        self.assertEqual(off['interventions'][:-1],active['interventions'])
        self.assertEqual(off['interventions'][-1]['channels'],['danger'])

    def setup_loop(self):
        graph=banc_fixture();body=ContactBody(42,default_world(),config_values())
        spec=build_spec(graph,2);return graph,body,NeuromuscularLoop(graph,spec,body)

    def test_receptors_distinguish_direction_and_opponent_position(self):
        spec=dict(load_half_bw=.5,touch_half_bw=.05)
        def feature(kind,typ,q,v,slow=0):return typed_receptor_feature(kind,typ,q,v,slow,.2,0,spec)
        self.assertGreater(feature('claw','SNpp50',2.,0),feature('claw','SNpp50',1.,0))
        self.assertLess(feature('claw','SNpp51',2.,0),feature('claw','SNpp51',1.,0))
        self.assertGreater(feature('hook','SNpp41',1,10),0)
        self.assertEqual(feature('hook','SNpp41',1,-10),0)
        self.assertGreater(feature('hook','SNpp39',1,-10),0)
        self.assertEqual(feature('hook','SNpp39',1,10),0)
        self.assertEqual(feature('hook','unresolved',1,10),0)
        self.assertEqual(feature('club','SNpp60',1,10,10),0)
        self.assertGreater(feature('club','SNpp60',1,-10,10),0)

    def test_floor_support_does_not_broadcast_bristle_contact(self):
        g,b,c=self.setup_loop();c.encode();drive,ports=c.encode()
        self.assertTrue(all(p['value']==0 for p in ports if p['receptor_kind']=='touch'))
        self.assertTrue(any(p['value']>0 for p in ports if p['receptor_kind']=='load'))
        self.assertTrue(all(p['feature']<1 for p in ports if p['receptor_kind']=='load'))
        c.encode(['leg_lf']);self.assertTrue(all(p['value']==0 for p in c.last_sensory if p['name'].startswith('leg_lf_')))
        before=c.snapshot();b.loads[0]=float('nan')
        with self.assertRaises(ValueError):c.encode()
        same_state(before,c.snapshot())

    def test_silent_motors_release_adhesion_and_active_balance_controls_it(self):
        g,b,c=self.setup_loop();n=ExpLIF(g)
        targets,adhesion=c.decode(n)
        self.assertFalse(adhesion.any());np.testing.assert_array_equal(targets,b.neutral)
        def ids(muscle):return g.resolve(next(m['ids'] for m in c.spec['rows'][0]['muscles'] if m['muscle']==muscle))
        n.rate[ids('tarsus_depressor_muscle')]=30
        self.assertTrue(c.decode(n)[1][0])
        n.rate[ids('tarsus_levator_muscle')]=60
        self.assertFalse(c.decode(n)[1][0])
        targets,adhesion=c.decode(n,disconnected=True)
        np.testing.assert_array_equal(targets,b.neutral);self.assertFalse(adhesion.any())

    def test_v2_receptor_state_restores_exactly_and_rejects_atomically(self):
        g,b,c=self.setup_loop();b.velocities[5]=12;c.encode();c.encode()
        cp=c.snapshot();c.encode();expected=c.snapshot();c.restore(cp);c.encode();same_state(expected,c.snapshot())
        bad=copy.deepcopy(cp);bad['velocity_lowpass'][3]=float('nan');before=c.snapshot()
        with self.assertRaises(ValueError):c.restore(bad)
        same_state(before,c.snapshot())
        bad=copy.deepcopy(cp);bad['last_contact'].pop('adhesion_force_bw')
        with self.assertRaises(ValueError):c.restore(bad)
        same_state(before,c.snapshot())
        bad=copy.deepcopy(cp);bad['last_contact']['support_load_bw']=[-1]*6
        with self.assertRaises(ValueError):c.restore(bad)
        same_state(before,c.snapshot())

    def test_actual_adhesion_compensation_and_contact_restoration(self):
        from flylab.body import FlyGymBody,dependency_report
        if not dependency_report()['ready']:self.skipTest('Pinned physical runtime required')
        world=default_world();world['sources']=[];world['obstacles']=[]
        b=FlyGymBody(42,world,config_values())
        try:
            cp=b.snapshot();obs=b.leg_observation()
            self.assertTrue(np.all(obs['load_bw']>3))
            self.assertTrue(np.all(obs['support_load_bw']<.5))
            self.assertAlmostEqual(float(obs['support_load_bw'].sum()),1.,places=3)
            np.testing.assert_array_equal(obs['non_support_load_bw'],np.zeros(6))
            b.sim.set_leg_adhesion_states(b.fly.name,np.zeros(6,bool));b.mj.mj_forward(b.m,b.d)
            off=b.leg_observation();self.assertTrue(np.all(off['adhesion_force_bw']==0))
            self.assertTrue(np.all(off['load_bw']<.5))
            b.restore(cp);restored=b.leg_observation()
            for key in obs:np.testing.assert_array_equal(obs[key],restored[key])
        finally:b.close()


if __name__=='__main__':unittest.main()
