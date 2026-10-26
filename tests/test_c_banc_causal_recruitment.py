"""Causal offline recruitment response contracts."""

import unittest

import numpy as np
import torch

from tools.banc_m3_causal_recruitment import endpoint_response, linear_response
from tools.fit_c_banc_m3_native_all_leg import (
    own_leg_targets,
    own_leg_warm_target,
    project_efficacies,
)


class CausalRecruitmentTests(unittest.TestCase):
    def test_own_leg_labels_and_muscle_warm_use_own_phase(self):
        grid=np.arange(200)/200
        offsets=np.array([0, .5, 0, .5, 0, .5])
        reference={'time_s':5+grid*2,
            'phase':2*np.pi*(grid[:,None]+offsets[None,:]),
            'command':np.column_stack([np.sin(2*np.pi*(grid+offsets[leg]))
                for leg in range(6) for _ in range(7)])}
        phases=np.array([[.1,.3,.6,.8,.2,.7],[.4,.6,.9,1.1,.5,1.0]])
        selected=np.array([True,True])
        labels=own_leg_targets(reference,phases,offsets,selected,0)
        for leg in range(6):
            expected=np.sin(2*np.pi*phases[:,leg])
            np.testing.assert_allclose(labels[:,7*leg:7*(leg+1)],
                np.repeat(expected[:,None],7,axis=1),atol=.001,rtol=0)
        self.assertGreater(abs(labels[0,7]-labels[0,0]),.1)
        bins=np.arange(100)/100
        warm=np.column_stack((bins,2*bins))
        aligned=own_leg_warm_target(bins,warm,phases,offsets,selected,
            np.array([0,1]))
        np.testing.assert_allclose(aligned[:,0],[.1,.4],atol=1e-12)
        np.testing.assert_allclose(aligned[:,1],[1.6,.2],atol=1e-12)

    def test_efficacy_roundoff_projection_preserves_strict_bounds(self):
        source = np.array([np.nextafter(.001, 0), np.exp(np.log(10.)), .2])
        present = np.ones(3, dtype=bool)
        projected = project_efficacies(source, present)
        np.testing.assert_array_equal(projected, [.001, 10., .2])
        for invalid in (0., .00099, 10.00001, np.nan):
            with self.assertRaises(ValueError):
                project_efficacies(np.array([invalid]), np.array([True]))

    def test_two_motor_taus_and_gradcheck(self):
        values = torch.tensor([[0.0, 0.0], [2.0, 1.0], [1.0, 4.0],
                               [3.0, 2.0], [0.5, 5.0]], dtype=torch.float64,
                              requires_grad=True)
        taus = torch.tensor([.02, .014046887330582595], dtype=torch.float64)
        result = linear_response(values, .005, taus).detach().numpy()
        for column, tau in enumerate(taus.numpy()):
            beta = np.exp(-.005 / tau)
            alpha = 1 - beta
            c = 1 - alpha / (.005 / tau)
            expected = [0.]
            for prior, current in zip(values.detach().numpy()[:-1, column],
                                      values.detach().numpy()[1:, column]):
                expected.append(beta * expected[-1] + (alpha - c) * prior + c * current)
            np.testing.assert_allclose(result[:, column], expected, atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: linear_response(x, .005, taus), (values,)))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: endpoint_response(x, .005, .01), (values,)))

    def test_piecewise_linear_exact_recurrence_and_no_future_leak(self):
        values = torch.tensor([[0.], [2.], [1.], [3.], [0.]],
                              dtype=torch.float64, requires_grad=True)
        dt, tau = .005, .02
        result = linear_response(values, dt, torch.tensor([tau]))
        beta = np.exp(-dt / tau)
        alpha = 1 - beta
        c = 1 - alpha / (dt / tau)
        expected = [0.]
        for previous, current in zip(values.detach().numpy()[:-1, 0],
                                     values.detach().numpy()[1:, 0]):
            expected.append(beta * expected[-1] + (alpha - c) * previous + c * current)
        np.testing.assert_allclose(result.detach().numpy()[:, 0], expected, atol=1e-12)
        result[2].sum().backward()
        self.assertLess(float(values.grad[3:].abs().sum()), 1e-14)

    def test_endpoint_zero_state_and_gradient(self):
        values = torch.tensor([[9.], [2.], [1.], [3.]],
                              dtype=torch.float64, requires_grad=True)
        result = endpoint_response(values, .005, .01)
        beta = np.exp(-.5)
        expected = [0.]
        for value in values.detach().numpy()[1:, 0]:
            expected.append(beta * expected[-1] + (1 - beta) * value)
        np.testing.assert_allclose(result.detach().numpy()[:, 0], expected, atol=1e-12)
        result[2].sum().backward()
        self.assertLess(abs(float(values.grad[0])), 1e-14)
        self.assertLess(abs(float(values.grad[3])), 1e-14)
        self.assertGreater(float(values.grad[1]), 0)


if __name__ == '__main__':
    unittest.main()
