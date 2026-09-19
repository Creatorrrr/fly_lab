import copy
import unittest

import numpy as np

from tools.analyze_c_banc_substep_quality import SCHEMA, compare_v2, summarize


class FullPhaseQualityTests(unittest.TestCase):
    def test_transient_is_counted_and_unsupported_is_not_zero_filled(self):
        load = np.full((100, 6), .1)
        slip = np.zeros((100, 6))
        slip[0::50] = 50.
        self.assertEqual(summarize(load, slip, .02)["balanced_mean_slip_mm_s"], 1.)
        load[1::2] = 0
        slip[1::2] = np.nan
        self.assertEqual(summarize(load, slip, .02)["balanced_mean_slip_mm_s"], 2.)

    def test_full_phase_comparison_preserves_original_progress_guard(self):
        old = dict(observation_valid=True, balanced_mean_slip_mm_s=1.,
                   mean_support_legs=3., moving_root_mean_speed_mm_s=4.,
                   first_arrival_s=6., approach_speed_mm_s=.8)
        base = dict(old, schema=SCHEMA, evaluator_hash="same", sample_dt_s=.0001,
                    control_dt_s=.005, support_threshold_bw=.02,
                    comparison_context={}, control_boundary_v1=old)
        candidate = copy.deepcopy(base)
        candidate["balanced_mean_slip_mm_s"] = .5
        candidate["control_boundary_v1"]["approach_speed_mm_s"] = .3
        result = compare_v2(base, candidate)
        self.assertTrue(result["Q1"])
        self.assertFalse(result["Q2"])
        candidate["evaluator_hash"] = "different"
        with self.assertRaises(ValueError):
            compare_v2(base, candidate)


if __name__ == "__main__":
    unittest.main()
