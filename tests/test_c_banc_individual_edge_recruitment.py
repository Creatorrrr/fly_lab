"""E101 independent existing-edge and native screen boundaries."""

import unittest
from types import SimpleNamespace

import numpy as np

from tools.fit_c_banc_m3_individual_edge_recruitment import (
    current_edge_start,
    independent_edge_order,
    native_shape_screen,
)


class IndependentEdgeTests(unittest.TestCase):
    def test_independent_edges_keep_distinct_parameters_for_same_motor(self):
        graph=SimpleNamespace(indptr=np.array([0,0,2]),
            indices=np.array([2,3]),data=np.array([.5,-.25]))
        rows,cols,edges=independent_edge_order(graph,np.array([1]),
            np.array([2,3]),np.array([[.5,-.25]]),expected_count=2)
        np.testing.assert_array_equal(rows,[0,0])
        np.testing.assert_array_equal(cols,[0,1])
        np.testing.assert_array_equal(edges,[0,1])
        with self.assertRaises(ValueError):
            independent_edge_order(graph,np.array([1]),np.array([2,4]),
                np.array([[.5,-.25]]),expected_count=2)

    def test_float32_current_roundtrip_and_bounds(self):
        original=np.array([.5,-.25],dtype=np.float32)
        current=np.array([np.float32(.5*.001),np.float32(-.25*10)],dtype=np.float32)
        raw,projected,difference=current_edge_start(original,current)
        self.assertTrue(np.isfinite(raw).all())
        self.assertLessEqual(difference,8*np.finfo(np.float32).eps*10)
        np.testing.assert_array_equal((original.astype(float)*projected).astype(np.float32),current)
        with self.assertRaises(ValueError):
            current_edge_start(original,np.array([.6,.25],dtype=np.float32))
        with self.assertRaises(ValueError):
            current_edge_start(original,np.array([.0001,-.25],dtype=np.float32))

    def test_final_screen_uses_native_LF_not_smooth_proxy(self):
        baseline={'q_rms':1.,'foot_rms':1.,'LF_row6_rms':1.,
            'per_leg_q_rms':[1.]*6}
        final={'q_rms':.7,'foot_rms':.7,'LF_row6_rms':.6,
            'per_leg_q_rms':[.9]*6}
        self.assertFalse(native_shape_screen(final,baseline)['supported'])
        final['LF_row6_rms']=.4
        self.assertTrue(native_shape_screen(final,baseline)['supported'])


if __name__=='__main__':
    unittest.main()
