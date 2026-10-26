"""E103 startup targets and native-only delivery gate."""

import unittest

import numpy as np

from tools.fit_c_banc_m3_startup_recruitment import startup_grid, startup_screen


class StartupRecruitmentTests(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(2001)*.005
        self.mask=(self.times>0)&(self.times<.5)
        self.q=np.zeros((99,42))
        self.tendon=np.zeros((99,6))

    def test_complete_99_sample_original_grid_required(self):
        startup_grid(self.times,self.mask,self.q,self.tendon)
        bad_mask=self.mask.copy()
        bad_mask[50]=False
        with self.assertRaisesRegex(ValueError,'Complete finite original99'):
            startup_grid(self.times,bad_mask,self.q[:-1],self.tendon[:-1])
        with self.assertRaisesRegex(ValueError,'Complete finite original99'):
            startup_grid(self.times+.005,self.mask,self.q,self.tendon)
        bad_q=self.q.copy()
        bad_q[20,3]=np.nan
        with self.assertRaisesRegex(ValueError,'Complete finite original99'):
            startup_grid(self.times,self.mask,bad_q,self.tendon)

    def test_late_only_good_candidate_fails_unsafe_startup(self):
        command=np.zeros((2001,42))
        tendon=np.zeros((2001,6))
        command[self.mask,25]=.1
        result=startup_screen(command,tendon,self.q,self.tendon,self.times,self.mask)
        self.assertFalse(result['supported'])
        self.assertGreater(result['q_max_abs_rad'],.05)
        command[self.mask,25]=0
        self.assertTrue(startup_screen(command,tendon,self.q,self.tendon,
            self.times,self.mask)['supported'])

    def test_nonfinite_native_prediction_rejected(self):
        command=np.zeros((2001,42))
        tendon=np.zeros((2001,6))
        command[30,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite native startup'):
            startup_screen(command,tendon,self.q,self.tendon,self.times,self.mask)


if __name__=='__main__':
    unittest.main()
