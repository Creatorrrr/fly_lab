"""E96 input roster must not mistake diagnostic readouts for fit sources."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.fit_c_banc_m3_observed_recruitment import (
    checked_initial_state,
    frozen_tripod_offsets,
    observed_roster,
    run,
    wall_budget_ok,
)


class ObservedRecruitmentTests(unittest.TestCase):
    def test_existing_output_preserved_by_direct_api_call(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'completed'
            out.mkdir()
            result=out/'result.json'
            result.write_bytes(b'{"validity":"frozen"}\n')
            before={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaises(FileExistsError):
                run(Path(folder),Path(folder),out)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})

    def test_final_wall_limit_keeps_measured_hypothesis_result(self):
        def completed_fit(root,graph_path,out,bodies,*,prior_wall_s):
            (out/'result.json').write_text(json.dumps({
                'validity':'valid_conditional_fit',
                'hypothesis_verdict':'conditional_shape_supported'}),encoding='utf-8')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'trial'
            with patch('tools.fit_c_banc_m3_observed_recruitment._run',completed_fit), \
                    patch('tools.fit_c_banc_m3_observed_recruitment.time.perf_counter',
                          side_effect=[0.,181.]):
                run(Path(folder),Path(folder),out)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['validity'],'wall_limit_incomplete')
            self.assertFalse(result['resource_limit_met'])
            self.assertEqual(result['hypothesis_verdict'],'conditional_shape_supported')

    def test_actual_checkpoint_nonzero_initial_activation_rejected(self):
        network={key:np.zeros(158706) for key in ('rate','drive','adaptation')}
        network['tick']=0
        state={'adapter':{'activation':np.zeros(96)},'network':network}
        np.testing.assert_array_equal(checked_initial_state(state),np.zeros(96))
        state['adapter']['activation'][67]=1
        with self.assertRaises(ValueError):
            checked_initial_state(state)

    def test_constant_quarter_phase_rejected(self):
        phase=np.tile(2*np.pi*np.array([0.,.5,0.,.5,0.,.5]),(4,1))
        np.testing.assert_array_equal(frozen_tripod_offsets(phase),
            [0.,.5,0.,.5,0.,.5])
        phase[:,4]=2*np.pi*.25
        with self.assertRaises(ValueError):
            frozen_tripod_offsets(phase)

    def test_prior_wall_cost_reduces_shared_allowance(self):
        self.assertTrue(wall_budget_ok(18,0,180))
        self.assertFalse(wall_budget_ok(18,163,180))
        with self.assertRaises(ValueError):
            wall_budget_ok(1,-1,180)

    def test_calibration_48_sources_selected_from_54_diagnostic_sources(self):
        source = np.arange(48)
        motor = np.arange(100, 491)
        diagnostics = np.arange(48, 54)
        ids = np.r_[source, motor, diagnostics]
        self.assertEqual(len(ids), 445)
        cal = {'input_ids': source, 'intrinsic_tonic_ids': np.array([], dtype=int),
            'coupling_ids': np.array([], dtype=int), 'motor_ids': motor}
        selected, mn, source_cols, motor_cols = observed_roster(cal, ids, ids,
            np.r_[source, diagnostics])
        np.testing.assert_array_equal(selected, source)
        np.testing.assert_array_equal(mn, motor)
        np.testing.assert_array_equal(source_cols, np.arange(48))
        np.testing.assert_array_equal(motor_cols, np.arange(48, 439))
        with self.assertRaises(ValueError):
            observed_roster(cal, ids, ids, source)


if __name__ == '__main__':
    unittest.main()
