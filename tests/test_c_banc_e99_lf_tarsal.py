"""Focused E99 analytical and conditional-source guards."""

import unittest

import numpy as np

from tools.prepare_c_banc_m3_e99_lf_tarsal import (
    deterministic_levator_map,
    inverse_native_rate,
    source_screen,
)


class E99SourceTests(unittest.TestCase):
    def test_inverse_native_rate_and_one_analytical_map(self):
        rate = np.array([5., 95.])
        self.assertTrue(np.allclose(200 * np.tanh((inverse_native_rate(rate) - 7.5) / 200), rate))
        i2 = np.linspace(10., 70., 1001)
        tonic = np.full_like(i2, 20.)
        background = np.full_like(i2, 3.)
        result = deterministic_levator_map(i2, tonic, background)
        low, high = np.percentile(i2, [5, 95])
        self.assertAlmostEqual(-result['negative_weight'] * (high-low),
            inverse_native_rate(95) - inverse_native_rate(5))
        self.assertAlmostEqual(result['tonic_factor'] * 20 + 3
            + result['negative_weight'] * low, inverse_native_rate(95))

    def test_invalid_anchors_or_tonic_factor_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_levator_map(np.ones(100), np.ones(100), np.zeros(100))
        with self.assertRaises(ValueError):
            deterministic_levator_map(np.linspace(0, 10, 100), -np.ones(100), np.zeros(100))

    def test_source_screen_requires_cadence_and_correlations(self):
        times = np.arange(2001) * .005
        period = .42
        wave = 50 + 40 * np.sin(2 * np.pi * times / period)
        target = wave.copy()
        good = source_screen(times, wave, wave, target, period, sign=1)
        self.assertTrue(good['screen'])
        wrong_target = 50 - 40 * np.sin(2 * np.pi * times / period)
        bad = source_screen(times, wave, wave, wrong_target, period, sign=1)
        self.assertFalse(bad['screen'])
        self.assertLess(bad['own_LF_muscle_target_Pearson'], 0)

    def test_source_screen_rejects_missing_target_and_truncated_window(self):
        times = np.arange(2001) * .005
        wave = 50 + 40 * np.sin(2 * np.pi * times / .42)
        target = wave.copy()
        target[600] = np.nan
        with self.assertRaisesRegex(ValueError, 'Every fixed-window'):
            source_screen(times, wave, wave, target, .42, sign=1)
        with self.assertRaisesRegex(ValueError, 'Complete original'):
            source_screen(times[:-1], wave[:-1], wave[:-1], wave[:-1], .42, sign=1)
        wave[700] = np.inf
        with self.assertRaisesRegex(ValueError, 'Every fixed-window'):
            source_screen(times, wave, np.ones_like(times), np.ones_like(times), .42, sign=1)


if __name__ == '__main__':
    unittest.main()
