"""Generated E319 closed-polyline and exact-startup contract checks."""

import unittest

import numpy as np
import torch

from tools.banc_m3_phase_free_shape import PhaseFreeShapeObjective


def scalar_shape(curves, q):
    """Independent NumPy point-to-segment formula, including the closing edge."""
    per_leg = np.zeros(6)
    for leg in range(6):
        distances = []
        for row in q[100:]:
            x = row[7 * leg:7 * leg + 7]
            segments = []
            for vertex in range(curves.shape[1]):
                a = curves[leg, vertex]
                b = curves[leg, (vertex + 1) % curves.shape[1]]
                d = b - a
                length_sq = float(np.dot(d, d))
                u = 0.0 if length_sq == 0 else float(np.clip(np.dot(x - a, d) / length_sq, 0, 1))
                residual = x - (a + u * d)
                segments.append(float(np.dot(residual, residual) / 7))
            distances.append(min(segments))
        per_leg[leg] = np.mean(distances)
    return per_leg


class PhaseFreeShapeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31901)
        self.curves = rng.normal(size=(6, 5, 7)).astype(np.float64)
        self.start_q = rng.normal(size=(99, 42)).astype(np.float64)
        self.start_t = rng.normal(size=(99, 6)).astype(np.float64)
        self.q = np.concatenate((self.start_q, rng.normal(size=(21, 42))), axis=0)
        self.tendon = np.concatenate((self.start_t, rng.normal(size=(21, 6))), axis=0)

    def objective(self, curves=None):
        return PhaseFreeShapeObjective(
            self.curves if curves is None else curves, self.start_q, self.start_t,
        )

    def evaluate(self, q=None, tendon=None, objective=None):
        return (objective or self.objective()).evaluate(
            torch.tensor(self.q if q is None else q, dtype=torch.float64),
            torch.tensor(self.tendon if tendon is None else tendon, dtype=torch.float64),
        )

    def test_scalar_formula_closing_edge_duplicates_and_cyclic_shift(self):
        curves = self.curves.copy()
        curves[:, 2] = curves[:, 1]  # Six zero-length segments.
        q = self.q.copy()
        q[100:, :] = 0
        # Points on the last-to-first closing edge, shifted off the curve.
        for leg in range(6):
            q[100:, 7 * leg:7 * leg + 7] = .35 * curves[leg, -1] + .65 * curves[leg, 0] + .01
        expected = scalar_shape(curves, q)
        actual = self.evaluate(q=q, objective=self.objective(curves))
        np.testing.assert_allclose(actual["per_leg_shape_loss"].detach().numpy(), expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(actual["shape_loss"].item(), expected.mean(), delta=1e-12)
        shifted = self.evaluate(q=q, objective=self.objective(np.roll(curves, 2, axis=1)))
        self.assertAlmostEqual(actual["shape_loss"].item(), shifted["shape_loss"].item(), delta=1e-12)
        self.assertTrue(actual["startup_pass"].item())

    def test_random_forward_and_smooth_directional_derivative(self):
        objective = self.objective()
        expected = scalar_shape(self.curves, self.q)
        q = torch.tensor(self.q, dtype=torch.float64, requires_grad=True)
        tendon = torch.tensor(self.tendon, dtype=torch.float64, requires_grad=True)
        result = objective.evaluate(q, tendon)
        np.testing.assert_allclose(result["per_leg_shape_loss"].detach().numpy(), expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(result["shape_loss"].item(), expected.mean(), delta=1e-12)
        rng = np.random.default_rng(31902)
        direction = np.zeros_like(self.q)
        direction[100:] = rng.normal(size=direction[100:].shape)
        result["shape_loss"].backward()
        analytic = float(np.sum(q.grad.detach().numpy() * direction))
        h = 1e-6
        plus = self.evaluate(q=self.q + h * direction, objective=objective)["shape_loss"].item()
        minus = self.evaluate(q=self.q - h * direction, objective=objective)["shape_loss"].item()
        numeric = (plus - minus) / (2 * h)
        self.assertLessEqual(abs(analytic - numeric), max(1e-9, 1e-6 * abs(numeric)))
        self.assertEqual(float(q.grad[:100].abs().max()), 0.0)

    def test_control_99_100_101_boundary_and_four_startup_gates(self):
        objective = self.objective()
        base = self.evaluate(objective=objective)
        self.assertTrue(base["startup_pass"].item())
        q = self.q.copy()
        t = self.tendon.copy()
        q[99] += 10
        t[99] += 10
        ignored = self.evaluate(q=q, tendon=t, objective=objective)
        self.assertTrue(ignored["startup_pass"].item())
        self.assertEqual(base["shape_loss"].item(), ignored["shape_loss"].item())
        q[100] += 10
        active = self.evaluate(q=q, tendon=t, objective=objective)
        self.assertNotEqual(base["shape_loss"].item(), active["shape_loss"].item())
        q = self.q.copy()
        q[98, 0] += .041
        self.assertFalse(self.evaluate(q=q, objective=objective)["startup_pass"].item())
        q = self.q.copy()
        q[:99] += .019
        self.assertFalse(self.evaluate(q=q, objective=objective)["startup_pass"].item())
        t = self.tendon.copy()
        t[98, 0] += .026
        self.assertFalse(self.evaluate(tendon=t, objective=objective)["startup_pass"].item())
        t = self.tendon.copy()
        t[:99] += .010
        self.assertFalse(self.evaluate(tendon=t, objective=objective)["startup_pass"].item())

    def test_reference_copy_input_unchanged_and_invalid_contracts(self):
        curves = self.curves.copy()
        start_q = self.start_q.copy()
        start_t = self.start_t.copy()
        objective = self.objective()
        baseline = self.evaluate(objective=objective)["shape_loss"].item()
        self.curves[:] = 1000
        self.start_q[:] = 1000
        self.start_t[:] = 1000
        self.assertEqual(baseline, self.evaluate(objective=objective)["shape_loss"].item())
        np.testing.assert_array_equal(self.q[:99], start_q)
        np.testing.assert_array_equal(self.tendon[:99], start_t)
        with self.assertRaises((ValueError, RuntimeError)):
            q = self.q.copy()
            q[100, 0] = np.nan
            self.evaluate(q=q, objective=objective)
        with self.assertRaises((ValueError, RuntimeError)):
            t = self.tendon.copy()
            t[0, 0] = np.inf
            self.evaluate(tendon=t, objective=objective)
        with self.assertRaises(ValueError):
            objective.evaluate(torch.tensor(self.q[:100]), torch.tensor(self.tendon[:100]))
        with self.assertRaises(TypeError):
            objective.evaluate(torch.tensor(self.q, dtype=torch.float32), torch.tensor(self.tendon))
        with self.assertRaises(ValueError):
            PhaseFreeShapeObjective(curves[:, :2], start_q, start_t)
        with self.assertRaises(ValueError):
            PhaseFreeShapeObjective(curves, start_q[:-1], start_t)

    def test_exact_startup_rms_has_finite_zero_gradient(self):
        objective = self.objective()
        self.assertEqual(objective.device, objective.curves.device)
        q = torch.tensor(self.q, dtype=torch.float64, requires_grad=True)
        tendon = torch.tensor(self.tendon, dtype=torch.float64, requires_grad=True)
        result = objective.evaluate(q, tendon)
        self.assertEqual(result["startup_q_rms"].item(), 0.0)
        self.assertEqual(result["startup_tendon_rms"].item(), 0.0)
        (result["startup_q_rms"] + result["startup_tendon_rms"]).backward()
        self.assertTrue(torch.isfinite(q.grad).all().item())
        self.assertTrue(torch.isfinite(tendon.grad).all().item())
        self.assertEqual(q.grad.abs().max().item(), 0.0)
        self.assertEqual(tendon.grad.abs().max().item(), 0.0)

    def test_finite_inputs_that_overflow_derived_math_are_rejected(self):
        curves = self.curves.copy()
        curves[0, 0, 0] = 1e308
        with self.assertRaises(ValueError):
            self.objective(curves)
        q = self.q.copy()
        q[100, 0] = 1e308
        with self.assertRaises((ValueError, RuntimeError)):
            self.evaluate(q=q)


if __name__ == "__main__":
    unittest.main()
