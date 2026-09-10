import copy
import json
import os
import unittest
from types import SimpleNamespace
import numpy as np
from flylab.physics import PhysicsProfile,body_factory
from flylab.flygym_senses import FourSiteOdor


class ProfilesAndOdor(unittest.TestCase):
    def test_doctor_rejects_unavailable_selected_physics(self):
        import contextlib
        import io
        from unittest.mock import patch
        from flylab.c.server import main
        output=io.StringIO()
        unavailable={'cpu':{'available':True},'warp':{'available':False,'reason':'Test missing Warp'}}
        with patch('sys.argv',['run_c.py','--doctor','--backend','exp_lif_cpu_reference','--physics-backend','warp']), \
             patch('flylab.physics.physics_availability',return_value=unavailable), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),2)
        result=json.loads(output.getvalue())
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIn('Test missing Warp',result['error'])

    def test_physics_default_and_incompatible_settings(self):
        from flylab.body import FlyGymBody
        self.assertIs(body_factory(),FlyGymBody)
        self.assertEqual(PhysicsProfile().noslip_iterations,5)
        self.assertFalse(PhysicsProfile(backend='warp').multiccd)
        for value in ({'backend':'mps'},{'backend':'warp','noslip_iterations':5},
                      {'backend':'warp','multiccd':True},{'max_contacts':-1},{'cuda_graph':1}):
            with self.assertRaises(ValueError):PhysicsProfile(**value)

    def test_four_site_field_rotation_and_food_switch(self):
        from flylab.common import to_ui,to_physics
        from flylab.sensors import default_world
        positions=np.array([[1.,.1,1.],[1.,-.1,1.],[.9,0.,.8]])
        body=SimpleNamespace(test_double=False,body_indices={'l_funiculus':0,'r_funiculus':1,'c_rostrum':2},
            body_ids=np.arange(3),d=SimpleNamespace(xpos=positions.copy(),xmat=np.tile(np.eye(3).ravel(),(3,1))),
            physics_time=lambda: .01)
        world=default_world()
        for field in ('gaussian','inverse-square'):
            sensor=FourSiteOdor(field=field)
            baseline=sensor.observe(body,world)
            # The observation RPC must use the selected field and its parameters.
            from flylab.c.server import CDispatcher,PROTOCOL
            dispatcher=CDispatcher.__new__(CDispatcher)
            dispatcher.engine=SimpleNamespace(body=body,world=world,sensors=SimpleNamespace(four_site_odor=sensor))
            reply=dispatcher.handle(dict(protocol=PROTOCOL,requestId=1,op='physical_observation',payload={'kind':'odor'}))
            self.assertEqual(reply['result'],baseline)
            rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
            moved=copy.deepcopy(body);moved.d.xpos=positions@rotation.T
            moved.d.xmat=np.tile(rotation.ravel(),(3,1))
            changed=copy.deepcopy(world)
            for source in changed['sources']:source['p']=to_ui(rotation@np.asarray(to_physics(source['p'])))
            rotated=sensor.observe(moved,changed)
            np.testing.assert_allclose(baseline['concentration'],rotated['concentration'],atol=1e-12)
            changed=copy.deepcopy(world);changed['foodOn']=False
            disabled=np.asarray(sensor.observe(body,changed)['response'])
            np.testing.assert_array_equal(disabled[:,0],np.zeros(4))
            np.testing.assert_array_equal(disabled[:,1],np.asarray(baseline['response'])[:,1])


@unittest.skipUnless(os.environ.get('FLYLAB_NATIVE_TESTS')=='1','Explicit native physics verification')
class NativeImprovements(unittest.TestCase):
    def test_retina_is_observation_only_and_kinematics_named(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        from flylab.kinematics import spotlight_trace
        b=FlyGymBody(42,default_world(),config_values(None))
        try:
            before=json.dumps(b.snapshot(),sort_keys=True)
            observation=b.compound_eye_observation()
            self.assertEqual(observation['ommatidia'].shape,(2,721,2))
            self.assertGreater(float(np.ptp(observation['ommatidia'])),.1)
            self.assertEqual(before,json.dumps(b.snapshot(),sort_keys=True))
            clip=spotlight_trace(b)
            self.assertEqual(clip['q_rad'].shape[1],42)
            self.assertEqual(clip['metadata']['joint_names'],b.joint_names)
            self.assertEqual(before,json.dumps(b.snapshot(),sort_keys=True))
        finally:b.close()

    def test_teacher_recovery_and_atomic_restore(self):
        from flylab.c.joint_teacher import JointTeacher
        teacher=JointTeacher()
        for i in range(100):teacher.step(1.862,torque=.01 if 50<=i<80 else 0.)
        self.assertLess(abs(teacher.body.frame()['q_rad']-1.862),.03)
        state=teacher.snapshot()
        expected=teacher.step(1.862)
        teacher.restore(state)
        self.assertEqual(expected,teacher.step(1.862))
        before=teacher.body.data.qpos.copy()
        invalid=copy.deepcopy(state);invalid['receptors']['queue'][0,0]=float('nan')
        with self.assertRaises(ValueError):teacher.restore(invalid)
        np.testing.assert_array_equal(before,teacher.body.data.qpos)

    def test_four_site_profile_roundtrip(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        from flylab.c.sensors import CSensorAdapter
        b=FlyGymBody(42,default_world(),config_values(None))
        try:
            model={'kind':'four-site-odor-v1','half_concentration':1.}
            a=CSensorAdapter(42,model)
            a.observe(b,default_world(),.005,config_values(None))
            saved=a.snapshot()
            c=CSensorAdapter(1,model);c.restore(saved)
            self.assertEqual(a.observe(b,default_world(),.005,config_values(None)),
                             c.observe(b,default_world(),.005,config_values(None)))
        finally:b.close()


if __name__=='__main__':unittest.main()
