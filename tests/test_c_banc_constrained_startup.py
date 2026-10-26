"""Focused checks for E104 native startup constraints and selection."""

import json
import unittest

import numpy as np
import torch
from scipy.optimize import OptimizeResult

from tools.fit_c_banc_m3_constrained_startup import (
    choose_native_record,
    conservative_startup,
    export_allowed,
    scipy_stop_record,
    startup_g,
    startup_screen,
)


class ConstrainedStartupTests(unittest.TestCase):
    def test_single_joint_maximum_is_not_diluted_by_mean(self):
        command=torch.zeros((100,42),dtype=torch.float64)
        command[1,0]=.06
        tendon=torch.zeros((100,6),dtype=torch.float64)
        target_q=torch.zeros((99,42),dtype=torch.float64)
        target_tendon=torch.zeros((99,6),dtype=torch.float64)
        mask=torch.zeros(100,dtype=torch.bool)
        mask[1:]=True
        g=startup_g(command,tendon,target_q,target_tendon,mask)
        self.assertLess(float(g[0]),0)
        self.assertGreater(float(g[1]),0)
        self.assertEqual(float(g[1]),.06**2/.05**2-1)
        self.assertTrue(all(float(value)<=0 for value in startup_g(
            torch.zeros_like(command),tendon,target_q,target_tendon,mask)))

    def test_relative_margin_only_affects_checkpoint_selection(self):
        limits={'q_RMS_rad':.02,'q_max_abs_rad':.05,
            'LTM_RMS':.01,'LTM_max_abs':.03}
        self.assertFalse(conservative_startup(limits))
        self.assertTrue(conservative_startup({k:v*(1-2e-4)
            for k,v in limits.items()}))
        times=np.arange(2001)*.005
        mask=np.zeros(2001,dtype=bool)
        mask[1:100]=True
        q=np.zeros((2001,42))
        q[1,0]=.05
        actual=startup_screen(q,np.zeros((2001,6)),np.zeros((99,42)),
            np.zeros((99,6)),times,mask)
        self.assertTrue(actual['supported'])
        self.assertFalse(conservative_startup(actual))

    def test_native_hierarchy_ignores_training_loss(self):
        self.assertEqual(choose_native_record({'lowest_P':{'smooth_loss':0}}),'lowest_P')
        self.assertEqual(choose_native_record({'lowest_P':{},
            'lowest_S_startup_feasible':{}}),'lowest_S_startup_feasible')
        self.assertEqual(choose_native_record({'lowest_P':{},
            'lowest_S_startup_feasible':{},'lowest_S_both_qualified':{}}),
            'lowest_S_both_qualified')
        with self.assertRaises(ValueError):
            choose_native_record({})

    def test_failed_final_native_screen_cannot_export(self):
        passing={'supported':True}
        failing={'supported':False}
        self.assertTrue(export_allowed(passing,passing,'valid_conditional_fit'))
        self.assertFalse(export_allowed(passing,failing,'valid_conditional_fit'))
        self.assertFalse(export_allowed(failing,passing,'valid_conditional_fit'))
        self.assertFalse(export_allowed(passing,passing,'wall_limit_incomplete'))

    def test_abnormal_scipy_return_is_not_reported_as_convergence(self):
        abnormal=OptimizeResult(success=False,status=2,
            message='ABNORMAL_TERMINATION_IN_LNSRCH',nit=4,nfev=63,njev=63,
            fun=.7735,jac=np.array([.1,-.4]))
        record=scipy_stop_record(abnormal)
        self.assertEqual(record['stop'],'lbfgs_scipy_failure')
        self.assertFalse(record['success'])
        self.assertEqual(record['nfev'],63)
        self.assertEqual(record['gradient_max_abs'],.4)
        normal=OptimizeResult(success=True,status=0,message='CONVERGENCE',
            nit=5,nfev=8,njev=8,fun=.6,jac=np.array([0.]))
        self.assertEqual(scipy_stop_record(normal)['stop'],'lbfgs_scipy_success')

    def test_nonfinite_return_keeps_failure_evidence_in_valid_json(self):
        failed=OptimizeResult(success=False,status=2,message='ABNORMAL',
            nit=1,nfev=3,njev=3,fun=np.nan,
            jac=np.array([np.nan,np.inf,-np.inf]))
        record=scipy_stop_record(failed)
        saved=json.loads(json.dumps(record,allow_nan=False))
        self.assertEqual(saved['stop'],'lbfgs_scipy_nonfinite_result')
        self.assertEqual(saved['message'],'ABNORMAL')
        self.assertEqual(saved['nfev'],3)
        self.assertIsNone(saved['objective'])
        self.assertEqual(saved['objective_nonfinite'],'nan')
        self.assertEqual(saved['gradient_nonfinite_counts'],{
            'nan':1,'positive_infinity':1,'negative_infinity':1})


if __name__=='__main__':
    unittest.main()
