"""Fixed later-cycle tonic phase objective and output preservation."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.probe_c_banc_m3_tonic_phase import phase_objective, project_tonic_tau


class TonicPhaseTests(unittest.TestCase):
    def test_only_exact_native_float32_initial_tau_projects(self):
        period=.41968137649044307
        native = float(np.float32(.02))
        original=np.array([native,period])
        np.testing.assert_array_equal(project_tonic_tau(original,period,
            native_initial=True),[.02,period])
        with self.assertRaises(ValueError):
            project_tonic_tau([native],period)
        for invalid in (0., np.nextafter(native,0.), .01999,
                        np.nextafter(period,np.inf), period+.00001,np.nan):
            with self.assertRaises(ValueError):
                project_tonic_tau([invalid],period,native_initial=True)

    def test_antiphase_zero_and_synchrony_nonzero(self):
        period = .42
        times = np.arange(2001) * .005
        leading = 80 + 50 * np.sin(2 * np.pi * ((times - .2) / period + .25))
        lagging = 80 - 50 * np.sin(2 * np.pi * ((times - .2) / period + .25))
        rates = np.column_stack((lagging, lagging, leading, leading,
                                 leading, lagging))
        queries = np.arange(.5, 9.0001, .025)
        residual, _, _, _ = phase_objective(times, rates, np.arange(6),
                                           period, queries)
        self.assertLess(float(np.linalg.norm(residual)), 1e-3)
        synchronized = np.repeat(leading[:, None], 6, axis=1)
        wrong, _, _, _ = phase_objective(times, synchronized, np.arange(6),
                                        period, queries)
        self.assertGreater(float(np.linalg.norm(wrong)), 5)

    def test_missing_phase_bracket_rejected(self):
        times = np.arange(2001) * .005
        rates = np.tile((80 + 50 * np.sin(2*np.pi*times/.42))[:, None], (1, 6))
        rates[:, 0] = 0
        with self.assertRaises(ValueError):
            phase_objective(times, rates, np.arange(6), .42,
                            np.arange(.5, 9.0001, .025))

    def test_existing_output_byte_identical(self):
        script = Path(__file__).resolve().parents[1] / 'tools/probe_c_banc_m3_tonic_phase.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'completed'
            out.mkdir()
            (out / 'result.json').write_bytes(b'preserved result')
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            completed = subprocess.run([sys.executable, '-X', 'utf8', str(script),
                '--root', str(Path(directory) / 'missing-root'),
                '--graph', str(Path(directory) / 'missing-graph'), '--out', str(out)],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('FileExistsError', completed.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == '__main__':
    unittest.main()
