import math
import os
import unittest
import numpy as np
from flylab.body_options import BodyOptions,outside_physical_domain


class TerrainDomain(unittest.TestCase):
    def test_downhill_negative_global_height_is_above_ground(self):
        for degrees in (-10,10):
            options=BodyOptions(terrain='slope',slope_degrees=degrees)
            angle=math.radians(degrees)
            normal=np.array([math.sin(angle),0.,math.cos(angle)])
            position=np.array([math.copysign(8.,degrees),0.,-.4])
            self.assertGreater(float(position@normal),.9)
            self.assertFalse(outside_physical_domain(position,np.eye(3),options))
            self.assertTrue(outside_physical_domain(position-2*normal,np.eye(3),options))
            self.assertTrue(outside_physical_domain(position,np.diag([1.,-1.,-1.]),options))

    def test_uprightness_uses_terrain_frame_and_domain_stays_bounded(self):
        options=BodyOptions(terrain='slope',slope_degrees=30)
        angle=math.radians(95)
        rotation=np.array([[math.cos(angle),0.,math.sin(angle)],[0.,1.,0.],[-math.sin(angle),0.,math.cos(angle)]])
        self.assertFalse(outside_physical_domain(np.array([0.,0.,1.]),rotation,options))
        self.assertTrue(outside_physical_domain(np.array([0.,0.,1.]),rotation,BodyOptions()))
        for options in (BodyOptions(),options):
            self.assertTrue(outside_physical_domain(np.array([81.,0.,1.]),np.eye(3),options))
            self.assertTrue(outside_physical_domain(np.array([0.,0.,-1.]),np.eye(3),options))
            self.assertTrue(outside_physical_domain(np.array([np.nan,0.,1.]),np.eye(3),options))

    @unittest.skipUnless(os.environ.get('FLYLAB_NATIVE_TESTS')=='1','Explicit native physics verification')
    def test_native_flybody_walks_past_zero_global_height(self):
        from flylab.body import FlyGymBody
        from flylab.sensors import default_world
        from flylab.engine import config_values
        from flylab.locomotion_experiments import run_controller
        body=FlyGymBody(42,default_world(),config_values(),body_options=BodyOptions(model='flybody',terrain='slope',terrain_seed=42))
        try:
            result,trace=run_controller(body,'hybrid',.7,42)
            self.assertTrue(result['completed'],result)
            self.assertLess(float(trace['position'][:,2].min()),-.1)
        finally:body.close()
