"""CPU numerical contracts; these tiny models do not establish BANC walking."""

import copy
import importlib.util
import unittest
from typing import ClassVar

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.research_rate import ResearchRateNetwork


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch runtime")
class AdaptiveRateNumerics(unittest.TestCase):
    reference_errors: ClassVar[dict] = {}

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    @staticmethod
    def arguments():
        # Distinct time constants and asymmetric E/I edges expose orientation
        # mistakes and incomplete RK4 updates of the second state variable.
        return (
            csr_matrix([[0, -0.8, 0.15], [1.3, 0, -0.4], [0.2, 0.9, 0]]),
            np.array([0.018, 0.025, 0.040]),
            np.array([1.1, 0.9, 1.3]),
            np.array([4.0, 6.0, 5.0]),
            np.array([80.0, 100.0, 120.0]),
        )

    def make_network(self, **options):
        return AdaptiveRateNetwork(
            *self.arguments(),
            adaptation_gain=options.pop("adaptation_gain", 2.5),
            adaptation_tau_s=0.06,
            dt=0.0001,
            device="cpu",
            **options,
        )

    def test_timestep_resolves_both_neural_states(self):
        with self.assertRaisesRegex(ValueError, "adaptation time constant"):
            AdaptiveRateNetwork(
                csr_matrix((1, 1)),
                np.array([20.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([200.0]),
                adaptation_gain=1.0,
                adaptation_tau_s=0.01,
                dt=0.1,
                device="cpu",
            )

    def assert_same_state(self, before, after):
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            else:
                self.assertEqual(before[key], after[key], key)

    def test_zero_gain_reproduces_original_rate_model_exactly(self):
        base = ResearchRateNetwork(*self.arguments(), device="cpu", dt=0.0001)
        adaptive = self.make_network(adaptation_gain=0.0)
        weights_before = self.arguments()[0].toarray()
        for drive, steps, cuts in (
            ([35, 0, 7], 7, []),
            ([35, 0, 7], 31, [1]),
            ([0, 20, 0], 87, [0]),
            ([0, 0, 0], 250, []),
        ):
            base.set_muted(cuts)
            adaptive.set_muted(cuts)
            base.advance(drive, steps)
            adaptive.advance(drive, steps)
            np.testing.assert_array_equal(adaptive.readout(), base.readout())
            np.testing.assert_array_equal(adaptive.snapshot()["adaptation"], 0)
            self.assertEqual(adaptive.tick, base.tick)
        np.testing.assert_array_equal(
            adaptive.weights.to_dense().numpy(), weights_before.astype(np.float32)
        )

    def test_coupled_rk4_matches_independent_float64_ode(self):
        network = self.make_network()
        state = network.snapshot()
        state["rate"] = np.array([12.0, 8.0, 4.0], np.float32)
        state["adaptation"] = np.array([0.4, 0.8, 0.2], np.float32)
        network.restore(state)
        reference = np.concatenate((state["rate"], state["adaptation"])).astype(
            np.float64
        )
        # Compare equations, not a second hand-written RK4. Quantized model
        # parameters are promoted to float64 before scipy's DOP853 integration.
        weights = network.weights.to_dense().numpy().astype(np.float64)
        tau, gain, threshold, cap = (
            value.numpy().astype(np.float64)
            for value in (network.tau, network.a, network.threshold, network.cap)
        )
        errors = []
        for drive, steps, cuts in (
            ([30.0, 0.0, 10.0], 500, []),
            ([0.0, 20.0, 0.0], 350, [1]),
            ([0.0, 0.0, 0.0], 200, []),
        ):
            held_drive = np.asarray(drive, np.float64)
            mask = np.ones(3)
            mask[cuts] = 0

            def equation(_time, value, held_drive=held_drive, mask=mask):
                rates, adaptation = value[:3], value[3:]
                current = held_drive + weights @ (rates * mask)
                target = np.maximum(
                    cap * np.tanh(gain * (current - threshold - adaptation) / cap),
                    0.0,
                )
                return np.concatenate(
                    (
                        (target - rates) / tau,
                        (2.5 * rates - adaptation) / 0.06,
                    )
                )

            solution = solve_ivp(
                equation,
                (0.0, steps * network.dt),
                reference,
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
                max_step=0.001,
            )
            self.assertTrue(solution.success, solution.message)
            reference = solution.y[:, -1]
            network.set_muted(cuts)
            network.advance(drive, steps)
            actual_state = network.snapshot()
            actual = np.concatenate((actual_state["rate"], actual_state["adaptation"]))
            errors.append(np.abs(actual - reference))
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        self.reference_errors.update(
            rate_max_abs=float(np.max(np.asarray(errors)[:, :3])),
            adaptation_max_abs=float(np.max(np.asarray(errors)[:, 3:])),
            simulated_seconds=network.tick * network.dt,
        )

    def test_outgoing_cut_preserves_source_adaptation_and_blocks_recruitment(self):
        args = (
            csr_matrix([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.full(3, 0.02),
            np.ones(3),
            np.full(3, 7.5),
            np.full(3, 200.0),
        )
        network = AdaptiveRateNetwork(
            *args, adaptation_gain=1.5, adaptation_tau_s=0.1, device="cpu"
        )
        network.advance([50, 0, 0], 600)
        intact = network.snapshot()
        self.assertGreater(intact["rate"][1], 1.0)
        self.assertGreater(intact["adaptation"][0], 1.0)
        network.reset()
        network.set_muted([0])
        network.advance([50, 0, 0], 600)
        cut = network.snapshot()
        self.assertEqual(intact["rate"][0], cut["rate"][0])
        self.assertEqual(intact["adaptation"][0], cut["adaptation"][0])
        np.testing.assert_array_equal(cut["rate"][1:], 0)
        np.testing.assert_array_equal(cut["adaptation"][1:], 0)

    def test_checkpoint_restores_cuts_adaptation_and_exact_future(self):
        network = self.make_network()
        network.set_muted([2])
        network.advance([35, 5, 0], 120)
        saved = network.snapshot()
        restored = self.make_network()
        restored.restore(saved)
        self.assert_same_state(saved, restored.snapshot())
        for drive, steps, cuts in (
            ([10, 0, 30], 87, [2]),
            ([0, 25, 0], 73, [0]),
            ([0, 0, 0], 60, []),
        ):
            for model in (network, restored):
                model.set_muted(cuts)
                model.advance(drive, steps)
            self.assert_same_state(network.snapshot(), restored.snapshot())
        restored.reset()
        reset = restored.snapshot()
        self.assertEqual(reset["tick"], 0)
        for key in ("rate", "drive", "adaptation"):
            np.testing.assert_array_equal(reset[key], 0)
        np.testing.assert_array_equal(reset["output_mask"], 1)

    def test_invalid_adaptive_checkpoint_rejection_is_atomic(self):
        network = self.make_network()
        network.advance([30, 5, 0], 40)
        baseline = network.snapshot()
        candidate = copy.deepcopy(baseline)
        # Distinct valid parent fields detect mutation before adaptation fails.
        candidate["tick"] += 100
        candidate["rate"] += 1
        candidate["drive"] += 2
        candidate["output_mask"][0] = 0
        invalid_fields = [
            ("adaptation", np.array([1e100, 0, 0], np.float64)),
            ("adaptation", np.ones(3, dtype=bool)),
            ("adaptation", np.array([1j, 0, 0], np.complex128)),
            ("adaptation", np.array(["invalid", "state", "data"], object)),
            ("adaptation", np.array([np.nan, 0, 0])),
            ("adaptation", np.array([-1.0, 0, 0])),
            ("adaptation", np.zeros(2)),
            ("adaptation", None),
            ("adaptation_gain", 2.0),
            ("adaptation_tau_s", 0.1),
            ("output_mask", np.array([1.0, 0.5, 1.0])),
        ]
        for key, value in invalid_fields:
            with self.subTest(field=key, value=repr(value)):
                network.restore(baseline)
                invalid = copy.deepcopy(candidate)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    network.restore(invalid)
                self.assert_same_state(baseline, network.snapshot())


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("cupy"),
    "Optional CUDA runtime",
)
class AdaptiveRateCudaNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA device unavailable")
        cls.torch = torch

    def test_packed_graph_refreshes_rates_and_cuts_after_restore_without_adaptation(
        self,
    ):
        reference = AdaptiveRateNetwork(
            *AdaptiveRateNumerics.arguments(),
            dt=0.00025,
            capture_steps=20,
            device="cuda",
        )
        packed = AdaptiveRateNetwork(
            *AdaptiveRateNumerics.arguments(),
            dt=0.00025,
            capture_steps=20,
            device="cuda",
            cuda_implementation="packed",
        )
        self.assertEqual(packed.cuda_implementation, "packed")
        initial = reference.snapshot()
        initial["rate"] = np.array([31, 5, 12], np.float32)
        initial["output_mask"] = np.array([0, 1, 1], np.float32)
        for network in (reference, packed):
            network.restore(initial)
        for drive, steps, cut in (
            ([35, 0, 7], 40, [0]),
            ([0, 20, 0], 21, [2]),
            ([0, 0, 0], 1, [0, 1, 2]),
            ([4, 0, 30], 60, []),
        ):
            for network in (reference, packed):
                network.set_muted(cut)
                saved = network.snapshot()
                network.advance(drive, steps)
                future = network.snapshot()
                # Captured temporary buffers must be recomputed after reset
                # and restore, even when their old values are still allocated.
                network.reset()
                network.restore(saved)
                network.advance(drive, steps)
                for key in ("rate", "adaptation", "drive", "output_mask"):
                    np.testing.assert_array_equal(
                        network.snapshot()[key].view(np.uint32),
                        future[key].view(np.uint32),
                    )
            self.assertEqual(reference.tick, packed.tick)
            for key in ("rate", "adaptation", "drive", "output_mask"):
                np.testing.assert_array_equal(
                    reference.snapshot()[key].view(np.uint32),
                    packed.snapshot()[key].view(np.uint32),
                )

    def test_fixed_warp_sum_matches_independent_matrix_and_repeats(self):
        rng = np.random.default_rng(91)
        weights = rng.normal(0, 0.015, (67, 67)).astype(np.float32)
        weights[0] = 0
        weights[weights < -0.02] = 0
        args = (
            csr_matrix(weights),
            np.full(67, 0.02),
            np.ones(67),
            np.full(67, 1.0),
            np.full(67, 80.0),
        )
        gpu = AdaptiveRateNetwork(*args, adaptation_gain=2.0, device="cuda")
        cpu = AdaptiveRateNetwork(*args, adaptation_gain=2.0, device="cpu")
        ids = np.arange(0, 67, 3)
        gpu.set_muted(ids)
        cpu.set_muted(ids)
        rates = rng.uniform(0, 100, 67).astype(np.float32)
        source = self.torch.as_tensor(rates, device="cuda")
        actual = gpu._current(source).cpu().numpy()
        mask = np.ones(67)
        mask[ids] = 0
        expected = weights.astype(np.float64) @ (rates.astype(np.float64) * mask)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
        np.testing.assert_array_equal(gpu._current(source).cpu().numpy(), actual)
        drive = rng.uniform(0, 40, 67).astype(np.float32)
        gpu.advance(drive, 200)
        cpu.advance(drive, 200)
        np.testing.assert_allclose(gpu.readout(), cpu.readout(), rtol=2e-5, atol=1e-5)
        saved = gpu.snapshot()
        gpu.advance(drive, 100)
        future = gpu.snapshot()
        gpu.restore(saved)
        gpu.advance(drive, 100)
        for key in ("rate", "adaptation", "drive", "output_mask"):
            np.testing.assert_array_equal(gpu.snapshot()[key], future[key])

    def test_packed_integrator_preserves_tensor_and_scalar_division_rounding(self):
        rng = np.random.default_rng(36)
        n = 67
        weights = rng.normal(0, 0.1, (n, n)).astype(np.float32)
        args = (
            csr_matrix(weights),
            rng.uniform(0.02, 0.08, n),
            rng.uniform(0.5, 2, n),
            rng.uniform(1, 10, n),
            rng.uniform(50, 250, n),
        )
        # These nonbinary scalar values distinguish true float division from
        # reciprocal multiplication and expose unintended fused multiply-add.
        for adaptation_tau in (0.03, 0.0602):
            with self.subTest(adaptation_tau=adaptation_tau):
                options = {
                    "adaptation_gain": 2.7,
                    "adaptation_tau_s": adaptation_tau,
                    "dt": 0.00025,
                    "capture_steps": 20,
                    "device": "cuda",
                }
                reference = AdaptiveRateNetwork(*args, **options)
                packed = AdaptiveRateNetwork(
                    *args, cuda_implementation="packed", **options
                )
                state = reference.snapshot()
                state["rate"] = rng.uniform(0, 200, n).astype(np.float32)
                state["adaptation"] = rng.uniform(0, 80, n).astype(np.float32)
                state["rate"][-4:] = [0, -0.0, 1e-40, 1e-35]
                state["adaptation"][-4:] = [0, -0.0, 1e-40, 1e-35]
                state["drive"] = rng.uniform(-100, 150, n).astype(np.float32)
                state["output_mask"][::3] = 0
                outputs = []
                for network in (reference, packed):
                    network.restore(state)
                    outputs.append(
                        tuple(
                            value.cpu().numpy()
                            for value in network._derivative_pair(
                                network.rate, network.adaptation
                            )
                        )
                    )
                for left, right in zip(*outputs):
                    np.testing.assert_array_equal(
                        left.view(np.uint32), right.view(np.uint32)
                    )
                for network in (reference, packed):
                    network.advance(state["drive"], 43)
                for key in ("rate", "adaptation"):
                    np.testing.assert_array_equal(
                        reference.snapshot()[key].view(np.uint32),
                        packed.snapshot()[key].view(np.uint32),
                    )

    def test_packed_weights_and_changed_capture_size_preserve_exact_future(self):
        rng = np.random.default_rng(82)
        weights = (rng.integers(-70, 100, size=(263, 263)) * 0.03).astype(np.float32)
        weights[rng.uniform(size=weights.shape) < 0.7] = 0
        weights[0] = 0
        args = (
            csr_matrix(weights),
            np.full(263, 0.02),
            np.ones(263),
            np.full(263, 7.5),
            np.full(263, 200.0),
        )
        reference = AdaptiveRateNetwork(*args, adaptation_gain=4, device="cuda")
        packed = AdaptiveRateNetwork(
            *args,
            adaptation_gain=4,
            device="cuda",
            cuda_implementation="packed",
            capture_steps=25,
        )
        self.assertEqual(packed.cuda_implementation, "packed")
        self.assertEqual(reference.identity, packed.identity)
        initial = reference.snapshot()
        initial["rate"] = rng.uniform(0, 200, 263).astype(np.float32)
        initial["adaptation"] = rng.uniform(0, 30, 263).astype(np.float32)
        reference.restore(initial)
        packed.restore(initial)
        for steps, cut in (
            (63, []),
            (75, [0, 26, 160]),
            (100, list(range(263))),
            (7, []),
        ):
            drive = rng.uniform(0, 150, 263).astype(np.float32)
            for net in (reference, packed):
                net.set_muted(cut)
                net.advance(drive, steps)
            for key in ("rate", "adaptation", "drive", "output_mask"):
                np.testing.assert_array_equal(
                    reference.snapshot()[key].view(np.uint32),
                    packed.snapshot()[key].view(np.uint32),
                )
        altered = AdaptiveRateNetwork(
            *args,
            adaptation_gain=4,
            device="cuda",
            dt=0.00025,
            capture_steps=20,
            cuda_implementation="packed",
        )
        before = altered.snapshot()
        with self.assertRaisesRegex(ValueError, "Incompatible research rate"):
            altered.restore(packed.snapshot())
        for key in ("rate", "adaptation", "drive", "output_mask"):
            np.testing.assert_array_equal(before[key], altered.snapshot()[key])


if __name__ == "__main__":
    unittest.main()
