import copy
import unittest
import numpy as np
from flylab.c.engine import CEngine
from flylab.c.ports import SensoryEncoder,PortBindings
from tests.c_fixtures import graph_fixture,bindings_fixture
from tests.fixture_body import FixtureBody


class OrderedDriveTests(unittest.TestCase):
    def test_small_overlapping_drives_are_rounded_in_port_order(self):
        graph=graph_fixture();spec=copy.deepcopy(bindings_fixture(graph).spec)
        template=spec['sensory'][0]
        spec['sensory']=[dict(template,name=str(i),baseline=1000. if i==0 else 1e-6,cap=1000.) for i in range(257)]
        binding=PortBindings(graph,spec)
        engine=CEngine(graph,bindings_fixture(graph),body_factory=FixtureBody)
        try:
            encoder=SensoryEncoder(binding,42)
            drive,_,_=encoder.encode(engine.last_sensors,.005,.0001)
            self.assertEqual(drive[0],np.float32(1000.))
            self.assertFalse(np.any(drive[1:]))
            saved=encoder.snapshot();off,_,_=encoder.encode(engine.last_sensors,.005,.0001,['*'])
            self.assertFalse(np.any(off));encoder.restore(saved)
            np.testing.assert_array_equal(drive,encoder.encode(engine.last_sensors,.005,.0001)[0])
        finally:engine.close()


if __name__=='__main__':unittest.main()
