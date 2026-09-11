"""Leakage, ID-join and missing-domain contracts for the E17 diagnostic."""

import unittest

import numpy as np

from flylab.c.research_morphology import extend_sizes
from tools.probe_c_morphology_size import bucket, fit_proxy, validate_ids


class MorphologySizeTests(unittest.TestCase):
    def test_size_extension_preserves_measurements_and_unvalidated_cells(self):
        baseline = np.array([4.5, 1, 1, 1], np.float32)
        measured = np.array([True, False, False, False])
        eligible = np.array([True, True, False, False])
        prediction = np.array([900, 30, 7000, np.nan])
        candidate, fill = extend_sizes(baseline, measured, prediction, eligible, 10.0)
        np.testing.assert_array_equal(candidate, [4.5, 3, 1, 1])
        np.testing.assert_array_equal(fill, [False, True, False, False])
        np.testing.assert_array_equal(baseline, [4.5, 1, 1, 1])
        prediction[1] = np.nan
        with self.assertRaises(ValueError):
            extend_sizes(baseline, measured, prediction, eligible, 10.0)
        np.testing.assert_array_equal(baseline, [4.5, 1, 1, 1])

    def test_holdout_labels_cannot_select_or_fit_the_model(self):
        rng = np.random.default_rng(418)
        ids = [str(720575941000000000 + i) for i in range(2200)]
        length = np.exp(rng.uniform(3, 8, len(ids)))
        volume = np.exp(rng.uniform(-2, 5, len(ids))) * 1e9
        area = np.exp(1 + 0.7 * np.log(length) + 0.3 * np.log(volume / 1e9))
        regions = ["vnc"] * 2000 + ["unlabeled_brain"] * 200
        area[2000:] = np.nan
        report, arrays = fit_proxy(ids, length, volume, area, regions)
        self.assertTrue(report["holdout_metrics"]["supported"])
        self.assertEqual(report["selected_model"], "log_length_volume")
        np.testing.assert_allclose(report["coefficients"], [1, 0.7, 0.3], atol=1e-12)
        changed = area.copy()
        changed[arrays["holdout"]] *= 100
        rejected, second = fit_proxy(ids, length, volume, changed, regions)
        self.assertEqual(rejected["coefficients"], report["coefficients"])
        self.assertEqual(rejected["cv"], report["cv"])
        self.assertFalse(rejected["holdout_metrics"]["supported"])
        self.assertFalse(second["empirical_domain_eligible"].any())
        self.assertEqual(report["regions"]["unlabeled_brain"]["labeled"], 0)
        self.assertFalse(arrays["empirical_domain_eligible"][2000:].any())
        self.assertTrue(np.isnan(arrays["reference_area_um2"][2000:]).all())

    def test_missing_data_does_not_reassign_split_or_impute_labels(self):
        rng = np.random.default_rng(119)
        ids = [str(720575941000000000 + i) for i in range(2200)]
        length = np.exp(rng.uniform(3, 8, len(ids)))
        volume = np.exp(rng.uniform(-2, 5, len(ids))) * 1e9
        area = length**0.8
        length[-4:] = [np.nan, 0, -1, np.inf]
        volume[-5] = np.nan
        report, arrays = fit_proxy(ids, length, volume, area, ["vnc"] * len(ids))
        self.assertEqual(report["morphology_available"], len(ids) - 5)
        self.assertEqual(report["labeled_intersection"], len(ids) - 5)
        self.assertTrue(np.isnan(arrays["predicted_area_um2"][-5:]).all())
        self.assertFalse(arrays["train"][-5:].any())
        self.assertFalse(arrays["holdout"][-5:].any())
        self.assertEqual(arrays["partition_bucket"][-1], bucket(ids[-1], "holdout"))

    def test_ambiguous_and_inexact_ids_rejected(self):
        for ids in (
            ["1", "1"],
            [720575941481179066],
            ["7.20575941481179e17"],
            ["01"],
            ["NaN"],
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_ids(ids)
