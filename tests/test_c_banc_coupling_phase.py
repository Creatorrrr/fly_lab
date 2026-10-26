"""Selected existing coupling-output bounds and immutable output guard."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.probe_c_banc_m3_coupling_phase import output_bounds, resource_budget


class CouplingPhaseTests(unittest.TestCase):
    def test_shared_reduced_budget_reserves_baseline_and_recheck(self):
        self.assertEqual(resource_budget(0), 240)
        self.assertEqual(resource_budget(10), 239)
        self.assertEqual(10 + 10 + 239 * 10 + 10, 2420)
        with self.assertRaises(ValueError):
            resource_budget(5)

    def test_intersection_uses_original_and_current_efficacy(self):
        original = np.array([1., 2., 3.])
        current = np.array([.01, 5., 20.])
        lower, upper = output_bounds(original, current)
        np.testing.assert_allclose(lower, [.005, 2.5, 10.])
        np.testing.assert_allclose(upper, [.02, 10., 30.])

    def test_invalid_source_or_empty_intersection_rejected(self):
        for original, current in (([0.], [1.]), ([1.], [0.]),
                                  ([1.], [100.])):
            with self.assertRaises(ValueError):
                output_bounds(np.array(original), np.array(current))

    def test_existing_output_preserved(self):
        script = Path(__file__).resolve().parents[1] / 'tools/probe_c_banc_m3_coupling_phase.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'completed'
            out.mkdir()
            (out / 'result.json').write_bytes(b'original evidence')
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            completed = subprocess.run([sys.executable, '-X', 'utf8', str(script),
                '--root', str(Path(directory) / 'missing'),
                '--graph', str(Path(directory) / 'graph'), '--out', str(out)],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('FileExistsError', completed.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == '__main__':
    unittest.main()
