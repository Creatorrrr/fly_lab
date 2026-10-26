"""E97 smooth training is distinct from exact native evaluation."""

import unittest

import numpy as np
import torch

from tools.fit_c_banc_m3_softplus_recruitment import (
    native_shape_screen,
    training_forcing,
)


class SoftplusRecruitmentTests(unittest.TestCase):
    def test_negative_native_input_has_training_gradient(self):
        z=torch.tensor([-4.],dtype=torch.float64,requires_grad=True)
        smooth=training_forcing(z,1.)
        smooth.backward()
        self.assertGreater(float(z.grad.item()),0)
        z.grad.zero_()
        native=training_forcing(z,0.)
        native.backward()
        self.assertEqual(float(native.item()),0)
        self.assertEqual(float(z.grad.item()),0)
        torch.testing.assert_close(training_forcing(z,None),native)

    def test_screen_uses_native_metrics_even_if_smooth_metrics_pass(self):
        baseline={'q_rms':1.,'foot_rms':1.,'RM_row29_rms':1.,
            'per_leg_q_rms':[1.]*6}
        smooth={'q_rms':.5,'foot_rms':.5,'RM_row29_rms':.5,
            'per_leg_q_rms':[.5]*6}
        native={**smooth,'q_rms':1.1,'per_leg_q_rms':[1.4]+[.5]*5}
        self.assertTrue(native_shape_screen(smooth,baseline)['supported'])
        self.assertFalse(native_shape_screen(native,baseline)['supported'])
        np.testing.assert_allclose(native_shape_screen(native,baseline)[
            'per_leg_q_ratios'][0],1.4)


if __name__=='__main__':
    unittest.main()
