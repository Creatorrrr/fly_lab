import copy
import math
import unittest
import numpy as np
from flylab.c.graph import GraphStore, external_id
from flylab.c.neural import ExpLIF, LIFParameters, create_backend
from flylab.c.ports import PortBindings, MotorDecoder
from flylab.c.engine import CEngine
from flylab.c.neuromuscular import NeuromuscularLoop, build_spec, joint_suffixes, MUSCLE_VECTORS, LEGS, PARTS
from tests.c_fixtures import graph_fixture, bindings_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


def banc_fixture():
    nodes = []
    def node(super_class, side, part, target, kind=None):
        root = str(100+len(nodes))
        a = dict(peripheral_target_type=target, body_part_effector=part if super_class=='motor' else None,
                 body_part_sensory=part if super_class=='sensory' else None,
                 cell_sub_class=f'{part}_{kind}_chordotonal_organ_neuron' if kind else None,
                 cell_function='tactile' if kind=='touch' else 'proprioception')
        cls = 'chordotonal_organ_neuron' if kind in ('claw','hook','club') else 'campaniform_sensillum_neuron' if kind=='load' else 'bristle_neuron'
        nodes.append(dict(id=external_id(root,'banc','888'), root_id=root, super_class=super_class,
                          soma_side=side, cell_type='SYNTHETIC_ONLY', nt_type='ACH', regions=[],
                          source_annotations=a, **{'class':cls}))
    for leg in LEGS:
        side, part = ('left' if leg[0]=='l' else 'right'), PARTS[leg[1]]
        for muscle in MUSCLE_VECTORS: node('motor', side, part, muscle)
        for kind in ('claw','hook','club','load','touch'):
            node('sensory', side, part, 'chordotonal_organ' if kind in ('claw','hook','club') else 'campaniform_sensillum' if kind=='load' else 'bristle', kind)
    for _ in range(6): node('descending','left',None,None)
    # Actual graph propagation is exercised: each sensory neuron excites the
    # preceding motor neuron through a strong test-only synapse.
    pre=[i for i,n in enumerate(nodes) if n['super_class']=='sensory']
    post=[max(0,i-1) for i in pre]
    return GraphStore.from_edges(nodes,np.array(pre,np.int32),np.array(post,np.int32),np.full(len(pre),100,np.int64),
        metadata=dict(dataset_id='flywire_banc',snapshot_id='888',specimen_id='TEST_ONLY',scope='fixture'))


class LegBody(FixtureBody):
    """Test-only actuator boundary. No physical-behavior claim."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.joint_names=['fixture'+s for leg in LEGS for s in joint_suffixes(leg)]
        self.neutral=np.tile([0,0,0,0,0,math.pi/2,0],6).astype(float)
        self.angles=self.neutral.copy();self.velocities=np.zeros(42);self.loads=np.full(6,.2)
    def leg_observation(self):
        return dict(angles_rad=self.angles.copy(),velocities_rad_s=self.velocities.copy(),load_bw=self.loads.copy())
    def step(self,*args,**kwargs): raise AssertionError('Predefined gait must not be used')
    def step_joint_targets(self,targets,adhesion,dt):
        self.velocities=(targets-self.angles)/dt;self.angles=targets.copy();self.time+=dt
    def snapshot(self):
        return dict(super().snapshot(),angles=self.angles.copy(),velocities=self.velocities.copy(),loads=self.loads.copy())
    def restore(self,s):
        super().restore(s)
        for k in ('angles','velocities','loads'):setattr(self,k,np.array(s[k]))


class SensorimotorTests(unittest.TestCase):
    def test_initial_pose_actual_body_transform_and_checkpoint_identity(self):
        from flylab.body import FlyGymBody,dependency_report
        from flylab.sensors import default_world
        from flylab.engine import config_values
        if not dependency_report()['ready']:self.skipTest('Pinned FlyGym/MuJoCo runtime required')
        world=default_world();world['sources']=[];world['obstacles']=[]
        pose=dict(position=[.5,.8,-.5],yaw_rad=.3)
        bodies=[]
        try:
            bodies.append(FlyGymBody(42,world,config_values()))
            bodies.append(FlyGymBody(42,world,config_values(),initial_pose=pose))
            bodies.append(FlyGymBody(42,world,config_values(),initial_pose=pose))
            base,posed,restored=bodies;frame=posed.frame()[0];x,y,z=base.frame()[0]['position']
            # The origin of the articulated asset differs from the thorax.
            expected=[.5+math.cos(.3)*x-math.sin(.3)*z,y,-.5+math.sin(.3)*x+math.cos(.3)*z]
            np.testing.assert_allclose(frame['position'],expected,rtol=0,atol=1e-4)
            self.assertAlmostEqual(frame['yaw'],.3,places=4)
            state=posed.snapshot();before=base.snapshot()
            with self.assertRaises(ValueError):base.restore(state)
            same_state(before,base.snapshot());restored.restore(state)
            for b in (posed,restored):b.step_joint_targets(b.neutral,np.ones(6,bool),.005)
            same_state(posed.snapshot(),restored.snapshot())
        finally:
            for b in bodies:b.close()

    def test_hazard_requires_matched_opportunity_and_excludes_direct_drives(self):
        from flylab.c.sensorimotor_campaign import evaluate_hazard, fixed_spec
        from flylab.c.behavior import trace_sample
        from flylab.c.campaign import scene_world, validate_spec
        g=graph_fixture();e=CEngine(g,bindings_fixture(g),mode='C_STRICT',body_factory=FixtureBody)
        try:first=trace_sample(e.frame());provenance=e.provenance()
        finally:e.close()
        provenance['bindings']['sensory']=[dict(name='test_danger',channel='danger')]
        world=scene_world('hazard_right');world['sources'][0]['p']=[8.,.7,0.]
        def arm(avoid=False,off=False,free=False):
            rows=[]
            for i in range(2001):
                r=copy.deepcopy(first);t=i*.005
                r.update(simTime=t,tick=i*50,position=[t,.9,-t*.4 if avoid else 0.],
                         sensory_ports=[dict(name='test_danger',enabled=not off,value=0. if off else 1.)])
                rows.append(r)
            w=copy.deepcopy(world)
            if free:w['sources']=[]
            return dict(trace=rows,world=w,provenance=copy.deepcopy(provenance),
                interventions=[dict(kind='sensor_off',channels=['danger'],duration_controls=2000)] if off else [])
        active,free,off=arm(True),arm(free=True),arm(off=True)
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'PASS')
        evoked=copy.deepcopy((active,free,off))
        stimulus=dict(kind='stimulate',ids=[g.nodes[1]['id']],amplitude_mV=20.,duration_controls=2000)
        for r in evoked:r['interventions'].insert(0,copy.deepcopy(stimulus))
        self.assertEqual(evaluate_hazard(*evoked)['task_status'],'INCOMPLETE')
        verdict=evaluate_hazard(*evoked,assay='evoked')
        self.assertEqual(verdict['task_status'],'PASS');self.assertFalse(verdict['natural_behavior'])
        evoked[2]['interventions'][0]['amplitude_mV']=19.
        self.assertEqual(evaluate_hazard(*evoked,assay='evoked')['task_status'],'INCOMPLETE')
        off['trace'][100]['sensory_ports'][0]['enabled']=True
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'INCOMPLETE')
        off['trace'][100]['sensory_ports'][0]['enabled']=False
        active['interventions']=[dict(kind='stimulate')]
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'INCOMPLETE')
        active['interventions']=[]
        for r in active['trace']:r['position']=[0.,.9,0.]
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'FAIL')
        for run in (free,off):
            for r in run['trace']:r['position']=[0.,.9,0.]
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'NOT_APPLICABLE')
        off['provenance']['seed']+=1
        self.assertEqual(evaluate_hazard(active,free,off)['task_status'],'INCOMPLETE')
        spec=validate_spec(fixed_spec());self.assertEqual(len(spec['cases']),120)
        self.assertEqual(len({tuple(c['initial_pose']['position'])+(c['initial_pose']['yaw_rad'],c['config']['friction']) for c in spec['cases']}),10)

    def test_voltage_events_can_emit_during_target_refractory_and_mps_matches(self):
        g=graph_fixture();p=LIFParameters(integration='exact-exponential-voltage-events-v1')
        cpu=ExpLIF(g,p);drive=np.zeros(g.n,np.float32)
        pi=np.array([0,0],np.int32);pv=np.zeros((30,2),np.float32);pv[:,0]=68.75
        cpu.advance(drive,30,range(g.n),(pi,pv))
        self.assertEqual(cpu.spike_count[0],30)
        self.assertEqual(cpu.h[0],0.)
        from tests.test_c_mps import mps_available
        if mps_available():
            gpu=create_backend(g,p,'exp_lif_mps');gpu.advance(drive,30,range(g.n),(pi,pv))
            self.assertEqual(cpu.last_events,gpu.last_events)
            for key in ('v','h','rate','queue'):
                np.testing.assert_allclose(cpu.snapshot()[key],gpu.snapshot()[key],atol=1e-4,rtol=0,err_msg=key)

    def test_reset_current_is_explicit_and_cpu_mps_agree(self):
        from tests.test_c_mps import mps_available
        g=graph_fixture();p=LIFParameters(integration='exact-exponential-reset-current-v1')
        cpu=ExpLIF(g,p);legacy=ExpLIF(g);drive=np.full(g.n,20.,np.float32)
        cpu.h[:]=50;legacy.h[:]=50
        cpu.advance(drive,30,range(g.n));legacy.advance(drive,30,range(g.n))
        self.assertTrue(np.all(cpu.h<legacy.h))
        if mps_available():
            gpu=create_backend(g,p,'exp_lif_mps');ref=ExpLIF(g,p)
            gpu.advance(drive,200,range(g.n));ref.advance(drive,200,range(g.n))
            self.assertEqual(gpu.last_events,ref.last_events)
            for key in ('v','h','rate','queue'):
                np.testing.assert_allclose(gpu.snapshot()[key],ref.snapshot()[key],atol=1e-4,rtol=0,err_msg=key)
            saved=gpu.snapshot();gpu.advance(drive,100);expected=gpu.snapshot();gpu.restore(saved);gpu.advance(drive,100)
            same_state(expected,gpu.snapshot())

    def test_yaw_and_stop_contracts_reject_stationary_or_missing_opportunity(self):
        from flylab.c.tasks import evaluate
        from flylab.c.behavior import trace_sample
        from flylab.sensors import default_world
        g=graph_fixture();e=CEngine(g,bindings_fixture(g),mode='C_STRICT',body_factory=FixtureBody)
        try:
            first=trace_sample(e.frame());trace=[]
            for k in range(1101):
                r=copy.deepcopy(first);r['simTime']=k*.005;r['tick']=k*50;trace.append(r)
            result=evaluate(trace,default_world(),'stop_resume',task_parameters=dict(stop_onset_s=2.,stop_release_s=3.5))
            self.assertEqual(result['task_status'],'FAIL')
            for r in trace:r['yaw']=-r['simTime']*.2
            self.assertEqual(evaluate(trace,default_world(),'yaw_left')['task_status'],'PASS')
            self.assertEqual(evaluate(trace,default_world(),'yaw_right')['task_status'],'FAIL')
            self.assertEqual(evaluate(trace[:101],default_world(),'yaw_left')['task_status'],'INCOMPLETE')
            with self.assertRaises(ValueError):evaluate(trace,default_world(),'stop_resume')
        finally:e.close()

    def test_stop_never_causes_backward_or_turning_and_legacy_is_preserved(self):
        g=graph_fixture();b=bindings_fixture(g);n=ExpLIF(g)
        n.rate[3]=100.
        legacy=MotorDecoder(b).decode(n)[0]
        self.assertLess(legacy['forwardSpeed'],0)
        spec=copy.deepcopy(b.spec);spec['motor_decoder']=dict(kind='bounded-opponent-v1',speed_limit=1.,yaw_limit=1.5,stop_half_drive=1.)
        d=MotorDecoder(PortBindings(g,spec));self.assertEqual(d.decode(n)[0]['forwardSpeed'],0.)
        n.rate[1]=100.;n.rate[5]=30.;braked=d.decode(n)[0]
        n.rate[3]=0;moving=d.decode(n)[0]
        self.assertLess(braked['forwardSpeed'],moving['forwardSpeed'])
        self.assertLess(braked['yawRate'],moving['yawRate'])

    def test_decoder_spec_rejects_nonfinite_and_unknown_models(self):
        g=graph_fixture();s=bindings_fixture(g).spec
        for value in (float('nan'),0,100):
            p=dict(s,motor_decoder=dict(kind='bounded-opponent-v1',speed_limit=value,yaw_limit=1.,stop_half_drive=1.))
            with self.assertRaises(ValueError):PortBindings(g,p)

    def setup_loop(self):
        from flylab.sensors import default_world
        from flylab.engine import config_values
        g=banc_fixture();body=LegBody(42,default_world(),config_values());loop=NeuromuscularLoop(g,build_spec(g),body)
        return g,body,loop

    def test_physical_feedback_delay_side_disable_and_atomic_rejection(self):
        g,b,c=self.setup_loop()
        first,_=c.encode();self.assertFalse(np.any(first))
        second,_=c.encode();self.assertTrue(np.any(second))
        drive,rows=c.encode(['leg_lf'])
        self.assertTrue(all(r['value']==0 for r in rows if r['name'].startswith('leg_lf_')))
        self.assertTrue(any(r['value']>0 for r in rows if r['name'].startswith('leg_rf_')))
        c.encode(['*']);self.assertTrue(all(r['value']==0 for r in c.last_sensory))
        state=c.snapshot();b.velocities[0]=np.nan
        with self.assertRaises(ValueError):c.encode()
        same_state(state,c.snapshot())

    def test_muscle_flexion_extension_and_immediate_motor_disconnect(self):
        g,b,c=self.setup_loop();n=ExpLIF(g)
        rows=c.spec['rows'][0]['muscles']
        flex=next(r for r in rows if r['muscle']=='tibia_flexor_muscle')
        ext=next(r for r in rows if r['muscle']=='tibia_extensor_muscle')
        n.rate[g.resolve(flex['ids'])]=100
        target,_=c.decode(n);self.assertGreater(target[5],b.neutral[5])
        target,_=c.decode(n,disconnected=True);np.testing.assert_array_equal(target,b.neutral)
        n.rate.fill(0);n.rate[g.resolve(ext['ids'])]=100
        target,_=c.decode(n);self.assertLess(target[5],b.neutral[5])

    def test_mapping_and_restore_are_identity_checked_before_mutation(self):
        g,b,c=self.setup_loop();s=copy.deepcopy(c.spec)
        s['rows'][0]['muscles'][0]['ids']=s['rows'][1]['muscles'][0]['ids']
        with self.assertRaises(ValueError):NeuromuscularLoop(g,s,b)
        c.encode();saved=c.snapshot();bad=copy.deepcopy(saved);bad['offset'][0]=2.
        with self.assertRaises(ValueError):c.restore(bad)
        same_state(saved,c.snapshot())

    def test_engine_closed_loop_checkpoint_interventions_and_no_cpg(self):
        g=banc_fixture();n=g.nodes;review=dict(review_status='engineering_reviewed',evidence=['SYNTHETIC TEST'],uncertainty='Test only')
        s=dict(schema='flylab.bindings.v3',graph_hash=g.hash,neuromuscular=build_spec(g),
            sensory=[dict(name='odor',channel='odor_mean',ids=[next(x['id'] for x in n if x['super_class']=='sensory')],
                          input_kind='sensory',method='drive_mV',gain=0.,baseline=0.,cap=24.,tau_s=0.,delay_controls=0,offset=0.,scale=1.,**review)],
            motor={role:dict(ids=[n[-i-1]['id']],gain=.01,output_side=role[4:] if role.startswith('yaw_') else None,**review)
                   for i,role in enumerate(('forward','backward','stop','yaw_left','yaw_right'))})
        b=PortBindings(g,s);e=CEngine(g,b,mode='C_STRICT',body_factory=LegBody)
        try:
            e.step(10);cp=e.checkpoint();e2=CEngine.from_checkpoint(g,b,cp,LegBody)
            try:
                e.step(5);e2.step(5);same_state(e.neural.snapshot(),e2.neural.snapshot());same_state(e.neuromuscular.snapshot(),e2.neuromuscular.snapshot())
                same_state(e.body.snapshot(),e2.body.snapshot())
                e.schedule(dict(kind='sensor_off',channels=['leg_feedback'],duration_controls=5));e.step()
                self.assertTrue(all(x['value']==0 for x in e.neuromuscular.last_sensory))
                self.assertFalse(e.frame()['command']['high_level_command_applied'])
            finally:e2.close()
            before=e.checkpoint();original=e.neuromuscular.encode
            def excessive_feedback(*args,**kwargs):
                drive,ports=original(*args,**kwargs)
                drive[0]=1001.
                return drive,ports
            e.neuromuscular.encode=excessive_feedback
            from flylab.c.inputs import InputRejected
            try:
                with self.assertRaises(InputRejected):e.step()
                same_state(before,e.checkpoint())
            finally:e.neuromuscular.encode=original
            with self.assertRaises(ValueError):CEngine(g,b,mode='C_SHADOW',body_factory=LegBody)
        finally:e.close()


if __name__=='__main__':unittest.main()
