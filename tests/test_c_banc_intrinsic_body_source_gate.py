"""E94 full-source trace and output preservation guards."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.verify_c_banc_m3_intrinsic_body import (
    complete_source_trace,
    equal_integer_ids,
    own_rm_inverse_at_actual_phase,
)


class E94SourceGateTests(unittest.TestCase):
    def test_saved_int64_and_graph_int32_ids_compare_by_value(self):
        equal_integer_ids(np.array([60038,11088],dtype=np.int64),
            np.array([60038,11088],dtype=np.int32),'same IDs')
        with self.assertRaisesRegex(ValueError,'different IDs'):
            equal_integer_ids(np.array([60038,11088],dtype=np.int64),
                np.array([60038,11089],dtype=np.int32),'different IDs')
        with self.assertRaisesRegex(ValueError,'integer IDs'):
            equal_integer_ids(np.array([60038.],dtype=float),
                np.array([60038],dtype=np.int32),'integer IDs')

    def test_own_phase_target_changes_when_other_run_has_different_phase(self):
        times = np.arange(2001) * .005
        mask = (times >= 2.5) & (times < 9.5)
        bins = np.arange(84) / 84
        muscle = 30 + 20 * np.cos(2*np.pi*bins)
        first_e1 = 100 + 60*np.cos(2*np.pi*(times-.2)/.42)
        second_e1 = 100 + 60*np.cos(2*np.pi*(times-.32)/.42)
        first_target, first_phase, _ = own_rm_inverse_at_actual_phase(
            times, first_e1, mask, bins, muscle, 0.)
        second_target, second_phase, _ = own_rm_inverse_at_actual_phase(
            times, second_e1, mask, bins, muscle, 0.)
        self.assertTrue(np.isfinite(first_phase).all())
        self.assertTrue(np.isfinite(second_phase).all())
        self.assertGreater(float(np.corrcoef(second_target, second_target)[0,1]), .999)
        self.assertLess(float(np.corrcoef(second_target, first_target)[0,1]), .5)

    def test_partial_prefix_cannot_pass_full_source_gate(self):
        ids = np.array([1, 2], dtype=np.int64)
        times = np.arange(1, 2001) * .005
        rates = np.ones((2000, 2))
        self.assertTrue(complete_source_trace(times, rates, ids, ids, 2000, 'duration'))
        self.assertFalse(complete_source_trace(times[:1500], rates[:1500],
            ids, ids, 1500, 'health_guard'))
        self.assertFalse(complete_source_trace(times, rates, ids, ids, 2000,
            'health_guard'))
        bad = rates.copy()
        bad[200, 0] = np.nan
        self.assertFalse(complete_source_trace(times, bad, ids, ids, 2000,
            'duration'))
        self.assertFalse(complete_source_trace(times, rates, ids[::-1], ids,
            2000, 'duration'))

    def test_existing_output_not_modified_by_cli_error(self):
        script = Path(__file__).resolve().parents[1] / 'tools/verify_c_banc_m3_intrinsic_body.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'completed'
            out.mkdir()
            (out / 'result.json').write_bytes(b'frozen')
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            run = subprocess.run([sys.executable, '-X', 'utf8', str(script),
                '--source', directory, '--root', directory,
                '--graph', directory, '--out', str(out),
                '--experiment', 'M3-E94-native-coupling-phase-body'],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('FileExistsError', run.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == '__main__':
    unittest.main()
