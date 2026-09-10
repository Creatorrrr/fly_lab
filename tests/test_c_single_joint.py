import copy
import importlib.util
import unittest
import numpy as np
from flylab.c.graph import GraphStore, external_id
from flylab.c.single_joint import JointParameters, SingleJointLoop, build_profile
from flylab.c.muscle_calibration import direction_status, sweep
from flylab.c.muscles import MuscleRig
from tests.test_c import same_state
from tests.test_c_mps import mps_available


def joint_graph():
    """Synthetic BANC-shaped roster, never used by production defaults."""
    types = ['tibia_flexor_Fast','tibia_extensor_FETi','SNpp50','SNpp51','SNpp41','SNpp39','SNpp43']
    nodes = []
    for i, cell in enumerate(types):
        motor = i < 2; kind = 'claw' if i in (2,3) else 'hook' if i in (4,5) else 'club'
        a = dict(body_part_effector='front_leg' if motor else None,
            body_part_sensory=None if motor else 'front_leg',
            peripheral_target_type=('tibia_flexor_muscle' if i == 0 else 'tibia_extensor_muscle') if motor else 'chordotonal_organ',
            cell_sub_class='test_'+kind+'_neuron')
        nodes.append(dict(id=external_id(str(100+i),'banc','888'),root_id=str(100+i),cell_type=cell,
            nt_type='ACH',soma_side='left',super_class='motor' if motor else 'sensory',
            **{'class':'leg_motor_neuron' if motor else 'chordotonal_organ_neuron'},
            source_annotations=a,regions=['TEST_ONLY']))
    return GraphStore.from_edges(nodes,np.array([2,3,4,5]),np.array([1,0,1,0]),np.array([1000]*4),
        metadata=dict(dataset_id='flywire_banc',snapshot_id='888',scope='SYNTHETIC TEST ONLY'))


class JointWorkflowTests(unittest.TestCase):
    def test_technical_pass_does_not_satisfy_required_feedback_gate(self):
        from tools.verify_c_single_joint import exit_code
        report=dict(status='COMPLETE_WITH_LIMITATIONS',technical_status='PASS',sensorimotor=dict(feedback_effect_detected=False))
        self.assertEqual(exit_code(report),0)
        self.assertEqual(exit_code(report,require_feedback=True),1)
        self.assertEqual(exit_code(dict(status='BLOCKED')),2)
        self.assertEqual(exit_code(dict(status='FAIL',technical_status='FAIL')),1)
        report['sensorimotor']['feedback_effect_detected']=True
        self.assertEqual(exit_code(report,require_feedback=True),0)


@unittest.skipUnless(importlib.util.find_spec('mujoco') and importlib.util.find_spec('flygym'),
                     'Native FlyGym/MuJoCo required')
class SingleJointTests(unittest.TestCase):
    def setUp(self): self.graph = joint_graph()

    def loop(self, **kwargs):
        body = MuscleRig(); profile = build_profile(self.graph,body,JointParameters(**kwargs))
        return SingleJointLoop(self.graph,profile,body=body)

    def test_source_and_analytic_geometry_agree_through_reversal(self):
        rig = MuscleRig(); before = rig.snapshot()
        report = sweep(rig,samples=51)
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(len(report['zero_crossings']),1)
        self.assertAlmostEqual(report['zero_crossings'][0]['zero_angle_rad'],.61343969465,places=9)
        same_state(before,rig.snapshot())
        self.assertEqual(direction_status(dict(moment_arm_mm=[-.01,-.001]))['status'],'REVERSED')
        self.assertEqual(direction_status(dict(moment_arm_mm=[-.01,0.]))['status'],'SINGULAR')

    def test_profile_rejects_wrong_anatomy_before_neural_allocation(self):
        body = MuscleRig(); profile = build_profile(self.graph,body)
        profile['bindings']['motor'][0]['ids']=[self.graph.nodes[1]['id']]
        with self.assertRaises(ValueError): SingleJointLoop(self.graph,profile,body=body)
        with self.assertRaises(ValueError): self.loop(initial_q_rad=.5)
        bad = copy.deepcopy(self.graph); bad.nodes[0]['source_annotations']['peripheral_target_type']='other'
        with self.assertRaises(ValueError): build_profile(bad,body)

    def test_causal_motor_delay_and_disconnection(self):
        loop = self.loop(); target = self.graph.nodes[0]['id']
        frame = loop.step(stimulation={target:100.},feedback=False)
        self.assertEqual(frame['requested_excitation'],[0.,0.])
        self.assertEqual(frame['motor_rate_sample_tick'],0)
        for _ in range(45): frame = loop.step(stimulation={target:30.},feedback=False)
        self.assertIsNone(frame['fault']); self.assertGreater(frame['physics']['activation'][0],.01)
        self.assertEqual(frame['motor_rate_sample_tick'],frame['neural_tick']-10)
        off = loop.step(stimulation={target:30.},feedback=False,motor_connected=False)
        self.assertEqual(off['physics']['applied_excitation'],off['physics']['minimum_excitation'])
        self.assertGreater(off['requested_excitation'][0],0.)
        self.assertGreater(off['physics']['activation'][0],0.)  # residual force is not erased

    def test_feedback_direction_delay_and_outgoing_ablation(self):
        on, off, swap, delayed, muted = (self.loop(),self.loop(),self.loop(polarity_swapped=True),
                                       self.loop(receptor_delay_steps=5),self.loop())
        sensory_ids = [n['id'] for n in self.graph.nodes[2:]]
        for k in range(60):
            a=on.step(); b=off.step(feedback=False); c=swap.step(); d=delayed.step()
            e=muted.step(mute_ids=sensory_ids)
            if k < 5: self.assertEqual(d['feedback_drive_max_mV'],0.)
        self.assertGreater(a['feedback_drive_max_mV'],0.); self.assertEqual(b['feedback_drive_max_mV'],0.)
        self.assertNotEqual(a['signals']['spike_count'],b['signals']['spike_count'])
        self.assertNotEqual(a['motor_rates_Hz'],e['motor_rates_Hz'])
        self.assertNotEqual(a['physics']['q_rad'],c['physics']['q_rad'])

    def test_invalid_input_and_late_restore_failure_are_atomic(self):
        loop=self.loop()
        for _ in range(25): loop.step()
        saved=loop.snapshot()
        for command in (dict(stimulation={'missing':1}),dict(torque=float('nan')),dict(feedback=1),dict(mute_ids=['missing'])):
            with self.assertRaises(ValueError): loop.step(**command)
            same_state(saved,loop.snapshot())
        for kind in ('clock','body','receptor','neural','diagnostic','magnitude','delay','motor_command','suppression'):
            bad=copy.deepcopy(saved)
            if kind=='clock': bad['control_tick']+=1
            if kind=='body': bad['body']['requested'][0]=2.
            if kind=='receptor': bad['receptors']['queue'][0,0]=2.
            if kind=='neural': bad['neural']['spike_count'][0]=-1
            if kind=='diagnostic': bad['diagnostics']['last_motor_tick']=0
            if kind=='magnitude': bad['diagnostics']['last_drive_max']=0.
            if kind=='delay': bad['diagnostics']['last_receptors']['output']['claw_positive']=0.
            if kind=='motor_command': bad['diagnostics']['last_input']['motor_connected']=False
            if kind=='suppression': bad['diagnostics']['last_input']['suppress_ids']=[self.graph.nodes[0]['id']]
            with self.assertRaises(ValueError): loop.restore(bad)
            same_state(saved,loop.snapshot())
        for _ in range(3): loop.step(torque=.01)
        expected=loop.snapshot(); loop.restore(saved)
        for _ in range(3): loop.step(torque=.01)
        same_state(expected,loop.snapshot())

    def test_geometry_guard_stops_with_partial_clock_and_evidence(self):
        loop=self.loop(initial_q_rad=.65)
        frame=loop.step(torque=-1.,feedback=False)
        for _ in range(100):
            if frame['fault']: break
            frame=loop.step(torque=-1.,feedback=False)
        self.assertEqual(frame['fault']['code'],'FAULT_MUSCLE_DIRECTION')
        self.assertEqual(frame['direction']['status'],'REVERSED')
        self.assertLessEqual(frame['seconds'],frame['neural_seconds'])
        self.assertLessEqual(frame['neural_seconds']-frame['seconds'],.001+1e-10)
        saved=loop.snapshot()
        with self.assertRaises(RuntimeError): loop.step()
        with self.assertRaises(ValueError): loop.restore(saved)
        same_state(saved,loop.snapshot())

    @unittest.skipUnless(mps_available(),'Actual MPS required')
    def test_mps_physics_and_selected_spikes_match_cpu(self):
        cpu=self.loop(); gpu=SingleJointLoop(self.graph,cpu.profile,'exp_lif_mps')
        for _ in range(50): a=cpu.step();b=gpu.step()
        self.assertEqual(a['signals']['spike_count'],b['signals']['spike_count'])
        self.assertAlmostEqual(a['physics']['q_rad'],b['physics']['q_rad'],places=6)
        saved=gpu.snapshot();gpu.step();expected=gpu.snapshot();gpu.restore(saved);gpu.step()
        same_state(expected,gpu.snapshot())


if __name__=='__main__': unittest.main()
