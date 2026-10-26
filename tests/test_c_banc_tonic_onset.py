"""Reduced native onset dynamics and phase extraction contracts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.probe_c_banc_m3_pair_coupling import rate_step
from tools.probe_c_banc_m3_tonic_onset import quadratic_peaks, reduced_rk4


class TonicOnsetTests(unittest.TestCase):
    def test_existing_output_is_untouched_by_cli_error(self):
        tool = Path(__file__).resolve().parents[1] / 'tools/probe_c_banc_m3_tonic_onset.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'completed'
            out.mkdir()
            sentinel = out / 'result.json'
            sentinel.write_bytes(b'original completed evidence')
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            process = subprocess.run([sys.executable, '-X', 'utf8', str(tool),
                '--root', str(Path(directory) / 'missing-root'),
                '--graph', str(Path(directory) / 'missing-graph'),
                '--out', str(out)], capture_output=True, text=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('FileExistsError', process.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_constant_background_matches_existing_native_rk4(self):
        weights = np.array([[.2, -.3], [.4, .1]])
        tau = np.array([.02, .04])
        background = np.tile([30., 20.], (3, 1))
        result = reduced_rk4(weights, background, tau, 2)
        rate = np.zeros(2)
        for control in range(2):
            for _ in range(20):
                rate = rate_step(rate, weights, background[control], tau)
            np.testing.assert_allclose(result[control + 1], rate, atol=1e-12)

    def test_linear_background_is_used_at_both_half_stages(self):
        weights = np.zeros((1, 1))
        tau = np.array([.02])
        background = np.array([[8.], [28.]])
        result = reduced_rk4(weights, background, tau, 1)
        rate = 0.
        for substep in range(20):
            fraction = substep / 20
            def derivative(value, portion):
                current = 8 + 20 * portion
                return (max(200 * np.tanh((current - 7.5) / 200), 0) - value) / .02
            k1 = derivative(rate, fraction)
            k2 = derivative(rate + .000125*k1, fraction + .025)
            k3 = derivative(rate + .000125*k2, fraction + .025)
            k4 = derivative(rate + .00025*k3, fraction + .05)
            rate += .00025/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.assertAlmostEqual(float(result[1, 0]), rate, places=12)

    def test_quadratic_peak_refines_native_sample(self):
        times = np.arange(100) * .005
        apex = .2113
        values = 100 - 1000 * (times - apex) ** 2
        np.testing.assert_allclose(quadratic_peaks(times, values, prominence=1),
            [apex], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
