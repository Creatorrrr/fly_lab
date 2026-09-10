import os
from pathlib import Path
import tempfile
import unittest
from flylab.dependencies import dependency_report
from flylab.c.jobs import CampaignJobs
from flylab.c.campaign import pilot_spec
from tests.c_fixtures import graph_fixture, bindings_fixture


@unittest.skipUnless(os.name=='nt' and dependency_report()['ready'],'Windows and the pinned physical runtime required')
class WindowsWorkerTests(unittest.TestCase):
    def test_child_owns_lease_and_accepts_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); graph=graph_fixture(); graph.save(root/'graph')
            bindings=bindings_fixture(graph)
            jobs=CampaignJobs(root/'jobs',root/'graph')
            spec=pilot_spec(.05,seeds=(42,),modes=('C_STRICT',));spec['cases']=spec['cases'][:1]
            result=jobs.start(bindings,'exp_lif_cpu_reference',spec)
            name=result['id']; process=jobs.processes[name]
            try:
                self.assertTrue(jobs.active())
                with self.assertRaises(ValueError):
                    CampaignJobs(jobs.root,root/'graph').start(bindings,'exp_lif_cpu_reference',spec)
                jobs.cancel(name)
                process.wait(timeout=30)
                self.assertFalse(jobs.active())
                state=jobs.status(name)
                self.assertEqual(state['status'],'CANCELLED',state)
            finally:
                if process.poll() is None:
                    jobs.cancel(name);process.wait(timeout=30)
