"""The combined readout must preserve every neural float and reject hidden faults."""

import importlib.util
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch")
class CombinedReadout(unittest.TestCase):
    def test_exact_cpu_future_across_cuts_restore_and_changed_inputs(self):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        args = (
            csr_matrix([[0.0, -0.2, 0.3], [1.0, 0.0, -0.1], [0.0, 1.0, 0.0]]),
            np.full(3, 0.02),
            np.ones(3),
            np.full(3, 7.5),
            np.full(3, 200.0),
        )
        for dt in (0.0001, 0.00025):
            reference = AdaptiveRateNetwork(
                *args, adaptation_gain=4.0, dt=dt, device="cpu"
            )
            combined = AdaptiveRateNetwork(
                *args, adaptation_gain=4.0, dt=dt, device="cpu"
            )
            for drive, steps, cuts, ids in (
                ([35, 2, 1], 20, [], [2, 0]),
                ([0, 12, 3], 7, [1], [1, 1]),
                ([0, 0, 0], 60, [0, 1, 2], []),
                ([7, 8, 20], 25, [], [0, 2]),
            ):
                reference.set_muted(cuts)
                combined.set_muted(cuts)
                saved = reference.snapshot()
                reference.advance(drive, steps)
                actual = combined.advance_readout(drive, ids, steps)
                np.testing.assert_array_equal(
                    actual.view(np.uint32),
                    reference.readout(np.asarray(ids, dtype=int)).view(np.uint32),
                )
                for key in ("rate", "adaptation", "drive", "output_mask"):
                    np.testing.assert_array_equal(
                        reference.snapshot()[key].view(np.uint32),
                        combined.snapshot()[key].view(np.uint32),
                    )
                combined.restore(saved)
                again = combined.advance_readout(drive, ids, steps)
                np.testing.assert_array_equal(
                    actual.view(np.uint32), again.view(np.uint32)
                )
            before = combined.tick
            for ids in ([3], [-1], [True], [1.5], [[0]]):
                with self.assertRaises(ValueError):
                    combined.advance_readout([0, 0, 0], ids, 1)
            self.assertEqual(combined.tick, before)
            # A bad neuron outside selected motor IDs must still stop execution.
            combined.rate[1] = float("nan")
            with self.assertRaisesRegex(RuntimeError, "complete-network"):
                combined.advance_readout([0, 0, 0], [0], 1)

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "Optional CUDA runtime")
    def test_cuda_combined_readout_bitwise(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("No CUDA device")
        args = (
            csr_matrix([[0.0, -0.3], [0.7, 0.0]]),
            np.full(2, 0.02),
            np.ones(2),
            np.full(2, 7.5),
            np.full(2, 200.0),
        )
        reference = AdaptiveRateNetwork(
            *args,
            adaptation_gain=4.0,
            device="cuda",
            dt=0.00025,
            capture_steps=20,
            cuda_implementation="packed",
        )
        combined = AdaptiveRateNetwork(
            *args,
            adaptation_gain=4.0,
            device="cuda",
            dt=0.00025,
            capture_steps=20,
            cuda_implementation="packed",
        )
        for drive, steps, cut in (
            ([20, 4], 20, []),
            ([0, 50], 43, [0]),
            ([10, 0], 7, [1]),
            ([0, 0], 60, []),
        ):
            for net in (reference, combined):
                net.set_muted(cut)
            reference.advance(drive, steps)
            output = combined.advance_readout(drive, [1, 0], steps)
            np.testing.assert_array_equal(
                output.view(np.uint32),
                reference.readout(np.array([1, 0])).view(np.uint32),
            )
            for key in ("rate", "adaptation", "drive", "output_mask"):
                np.testing.assert_array_equal(
                    reference.snapshot()[key].view(np.uint32),
                    combined.snapshot()[key].view(np.uint32),
                )


if __name__ == "__main__":
    unittest.main()
