"""Neural/controller regressions; real navigation is checked separately."""
import unittest
import numpy as np
from flylab.brain import Circuit, NeuralController
from flylab.body import MotorAdapter
from flylab.engine import default_graph, config_values


class AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.brain = NeuralController(Circuit(default_graph()), 42)
        self.config = config_values(dict(task='heading', goalAngle=0))
        self.sensor = dict(nearRanges=[.08]*9, contact=0, panorama=[0]*64,
                           odor=[0, 0], odorChange=0, danger=0, angularVelocity=0,
                           forwardSpeed=0, clearanceDown=1, clearanceUp=10)

    def advance(self, steps=200):
        for _ in range(steps):
            self.brain.step(self.sensor, .005, self.config)

    def test_close_wall_keeps_turn_and_stops_forward_command(self):
        self.advance()
        self.assertGreater(self.brain.output['yawRate'], 2)
        self.assertEqual(self.brain.output['forwardSpeed'], 0)

    def test_turn_commitment_survives_side_noise(self):
        self.advance()
        self.sensor['nearRanges'] = [10]*4 + [.08]*5
        self.advance()
        self.assertGreater(self.brain.turnMemory, 0)
        self.assertGreater(self.brain.output['yawRate'], 2)

    def test_clearance_alone_does_not_erase_turn_memory(self):
        self.advance(1)
        self.sensor['nearRanges'] = [10]*9
        self.advance()
        self.assertGreater(self.brain.turnMemory, 0)
        self.sensor['angularVelocity'] = 2
        self.advance(150)
        self.assertEqual(self.brain.turnMemory, 0)
        self.assertGreater(self.brain.output['forwardSpeed'], 0)

    def test_contact_alone_triggers_avoidance(self):
        self.sensor.update(nearRanges=[10]*9, contact=1)
        self.advance()
        self.assertNotEqual(self.brain.turnMemory, 0)
        self.assertLessEqual(self.brain.output['forwardSpeed'], 0)

    def test_dn_suppression_is_not_bypassed_by_avoidance(self):
        ids = [self.brain.circuit.nodes[i]['id'] for i in self.brain.circuit.pop['DN']]
        self.brain.intervene(ids, 'suppress')
        self.advance()
        self.assertNotEqual(self.brain.turnMemory, 0)
        self.assertEqual(self.brain.output['yawRate'], 0)

    def test_disconnection_parks_avoidance(self):
        self.config['motorCoupled'] = False
        self.advance()
        self.assertEqual(self.brain.output, dict(yawRate=0, forwardSpeed=0, verticalSpeed=0))

    def test_pivot_uses_opposite_cpg_directions(self):
        motor = MotorAdapter()
        right = motor.map(dict(forwardSpeed=0, yawRate=3))
        left = motor.map(dict(forwardSpeed=0, yawRate=-3))
        self.assertGreater(right[0], 0)
        self.assertLess(right[1], 0)
        np.testing.assert_allclose(left, -right)

    def test_persistent_contact_backs_away(self):
        self.sensor['contact'] = 1
        self.advance(65)
        self.assertGreater(self.brain.recoveryTime, 0)
        self.assertLess(self.brain.output['forwardSpeed'], 0)
        self.assertTrue((MotorAdapter().map(self.brain.output) < 0).all())

    def test_checkpoint_preserves_active_turn(self):
        self.advance(40)
        other = NeuralController(self.brain.circuit, 0)
        other.restore(self.brain.snapshot())
        for _ in range(100):
            self.sensor['angularVelocity'] = .5
            self.brain.step(self.sensor, .005, self.config)
            other.step(self.sensor, .005, self.config)
        self.assertEqual(self.brain.snapshot(), other.snapshot())
