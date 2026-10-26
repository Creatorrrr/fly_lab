"""E105 complete-prefix transfer and frozen E104 failure guards."""

import json
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_c_banc_m3_e105_startup import checked_e104_status, equal_integer_ids
from tools.recover_c_banc_m3_e105_postprocess import e105_resource_ledger
from tools.verify_c_banc_m3_intrinsic_body import e105_startup_diagnostic


class E105StartupTests(unittest.TestCase):
    @staticmethod
    def prefix():
        rows=[{'time_s':0.}]+[{'time_s':i*.005,
            'joint_targets_rad':[0.]*42,
            'tendon_inputs':{f'{leg}_tarsus':0. for leg in ('lf','lm','lh','rf','rm','rh')}}
            for i in range(1,100)]
        return rows,{'command':np.zeros((2001,42)),
            'tendon':np.zeros((2001,6))}

    def test_original_99_sample_transfer_requires_complete_grid(self):
        rows,predicted=self.prefix()
        result,*_=e105_startup_diagnostic(rows,[(0,)]*4950,predicted)
        self.assertTrue(result['complete_99_control_prefix'])
        self.assertTrue(result['conditional_transfer_screen'])
        truncated,*_=e105_startup_diagnostic(rows[:-1],[(0,)]*4900,predicted)
        self.assertFalse(truncated['complete_99_control_prefix'])
        self.assertFalse(truncated['conditional_transfer_screen'])
        rows[20]['joint_targets_rad'][0]=.011
        failed,*_=e105_startup_diagnostic(rows,[(0,)]*4950,predicted)
        self.assertFalse(failed['conditional_transfer_screen'])

    def test_nonfinite_reference_or_actual_fails_with_raw_difference_retained(self):
        rows,predicted=self.prefix()
        predicted['tendon'][20,0]=np.nan
        result,_,_,_,_,dltm=e105_startup_diagnostic(rows,[(0,)]*4950,predicted)
        self.assertFalse(result['complete_99_control_prefix'])
        self.assertFalse(result['conditional_transfer_screen'])
        self.assertIsNone(result['LTM_RMS'])
        self.assertTrue(np.isnan(dltm[19,0]))
        json.dumps(result,allow_nan=False)
        rows,predicted=self.prefix()
        rows[20]['joint_targets_rad'][0]=float('nan')
        result,_,actual,_,dq,_=e105_startup_diagnostic(rows,[(0,)]*4950,predicted)
        self.assertFalse(result['complete_99_control_prefix'])
        self.assertFalse(result['conditional_transfer_screen'])
        self.assertTrue(np.isnan(actual[19,0]))
        self.assertTrue(np.isnan(dq[19,0]))
        json.dumps(result,allow_nan=False)

    def test_empty_and_partial_prefix_fail_with_available_raw_samples(self):
        rows,predicted=self.prefix()
        for sample,contact in ((rows[:1],[]),(rows[:3],[(0,)]*100)):
            result,times,actual,_,dq,_=e105_startup_diagnostic(
                sample,contact,predicted)
            self.assertFalse(result['complete_99_control_prefix'])
            self.assertFalse(result['conditional_transfer_screen'])
            self.assertIsNone(result['q_RMS_rad'])
            self.assertEqual(len(times),len(sample)-1)
            self.assertEqual(actual.shape,(len(sample)-1,42))
            self.assertEqual(dq.shape,(len(sample)-1,42))
            json.dumps(result,allow_nan=False)
        predicted['command']=predicted['command'][:2]
        result,_,actual,_,dq,_=e105_startup_diagnostic(rows,[(0,)]*4950,predicted)
        self.assertFalse(result['complete_99_control_prefix'])
        self.assertEqual(actual.shape,(99,42))
        self.assertEqual(dq.shape,(1,42))

    def test_arithmetic_overflow_remains_a_serializable_failed_diagnostic(self):
        for actual,prediction in ((1e308,-1e308),(1e200,0.)):
            with self.subTest(actual=actual,prediction=prediction):
                rows,predicted=self.prefix()
                rows[20]['joint_targets_rad'][0]=actual
                predicted['command'][20,0]=prediction
                result,_,raw,_,dq,_=e105_startup_diagnostic(
                    rows,[(0,)]*4950,predicted)
                self.assertFalse(result['conditional_transfer_screen'])
                self.assertIsNone(result['q_RMS_rad'])
                self.assertEqual(raw[19,0],actual)
                self.assertEqual(dq.shape,(99,42))
                json.dumps(result,allow_nan=False)

    def test_failed_steady_screen_is_required_not_relabelled(self):
        valid={'validity':'valid_conditional_fit','resource_limit_met':True,
            'startup_screen':{'supported':True},'shape_screen':{'supported':False},
            'local_supported':False,'fit_parameters':1572,'fitted_edges':1572,
            'selected_record':'lowest_S_startup_feasible'}
        selection={'selected_record':'lowest_S_startup_feasible'}
        accepted={'accepted_as_valid_failed_conditional_fit':True,
            'blocking_defects':[]}
        checked_e104_status(valid,selection,accepted)
        root=Path(__file__).resolve().parents[1]
        real_review=json.loads((root/'verification/banc-natural-gait-m3-20260919'
            '/e104-constrained-startup-recruitment/parent-second-review.json').read_text())
        checked_e104_status(valid,selection,real_review)
        invalid=dict(valid,shape_screen={'supported':True})
        with self.assertRaises(ValueError):
            checked_e104_status(invalid,selection,accepted)
        for review in ({'accepted_as_valid_failed_conditional_fit':False,
                'blocking_defects':[]},
                {'accepted_as_valid_failed_conditional_fit':True,
                    'blocking_defects':['pending']},
                {'blocking_defects':[]},
                {'accepted_as_valid_failed_conditional_fit':True}):
            with self.subTest(review=review),self.assertRaises(ValueError):
                checked_e104_status(valid,selection,review)

    def test_recovery_wall_is_charged_to_body_and_shared_limit(self):
        ledger=e105_resource_ledger(50,10,1,284.8,.3,10.1)
        self.assertAlmostEqual(ledger['body_including_recovery_wall_s'],285.1)
        self.assertFalse(ledger['resource_limit_met'])

    def test_graph_int32_and_saved_int64_id_values_match_only_as_integers(self):
        equal_integer_ids(np.array([2,5],dtype=np.int32),
            np.array([2,5],dtype=np.int64),'ids')
        with self.assertRaises(ValueError):
            equal_integer_ids(np.array([2,5],dtype=np.int32),
                np.array([2,6],dtype=np.int64),'ids')
        with self.assertRaises(ValueError):
            equal_integer_ids(np.array([2.,5.]),np.array([2,5]),'ids')


if __name__=='__main__':
    unittest.main()
