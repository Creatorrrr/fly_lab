import os
import unittest
import numpy as np
from flylab.c.flybody_flight import make_flight_env


class FlightArguments(unittest.TestCase):
    def test_invalid_inputs_fail_before_source_loading(self):
        for options in (dict(task='walk'),dict(seed=True),dict(seconds=float('nan')),dict(seconds=-1)):
            with self.assertRaises(ValueError):make_flight_env('unused',**options)


@unittest.skipUnless(os.environ.get('FLYLAB_FLIGHT_SOURCE'),'Explicit official FlyBody source required')
class FlightNative(unittest.TestCase):
    def test_all_tasks_integrate_contact_metrics_and_reject_invalid_actions(self):
        for task in ('flight','takeoff','landing'):
            env=make_flight_env(os.environ['FLYLAB_FLIGHT_SOURCE'],task=task,seconds=.002)
            try:
                env.reset();before=env.physics.data.qpos.copy()
                with self.assertRaises(ValueError):env.step(np.full(env.action_spec().shape,np.nan))
                np.testing.assert_array_equal(before,env.physics.data.qpos)
                self.assertEqual(env.physics.time(),0.)
                env.reset()
                for _ in range(10):t=env.step(np.zeros(env.action_spec().shape))
                self.assertTrue(t.last());self.assertAlmostEqual(env.physics.time(),.002)
                self.assertTrue(np.isfinite(env.physics.data.qpos).all())
                before=dict(env.task.last_metrics);env.task.get_reward_factors(env.physics)
                self.assertEqual(before,env.task.last_metrics)
                self.assertFalse(before['success'])
            finally:env.close()


if __name__=='__main__':unittest.main()
