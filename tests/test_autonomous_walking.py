import copy
import unittest
from unittest.mock import patch

import numpy as np

from flylab.autonomy import AUTONOMOUS_MODE, SensoryWalkingPolicy
from flylab.c.engine import CEngine
from flylab.c.protocol import decode_signals, signal_frame
from flylab.c.sensors import CSensorAdapter
from tests.c_fixtures import bindings_fixture, graph_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state
from tools.verify_autonomous_walking import evaluate_trace


def packet(odor=(0.05, 0.05), ranges=None, contact=0):
    return {"odor": list(odor), "nearRanges": ranges or [10.0] * 9, "contact": contact}


class AutonomousFixtureBody(FixtureBody):
    """Protocol test double; never evidence of real gait or sensory placement."""

    def __init__(self, seed, world, config, **kwargs):
        super().__init__(seed, world, config)


class AutonomousWalkingTests(unittest.TestCase):
    def test_walking_gate_rejects_jitter_but_excludes_scheduled_rest(self):
        t = np.arange(1, 2001) * 0.005
        position = np.column_stack(
            [np.minimum(t * 5, 25), np.zeros(len(t)), np.ones(len(t))]
        )
        position[t > 5, 0] += 0.1 * np.sin(t[t > 5] * 600)
        rows = {
            "time_s": t,
            "position": position,
            "heading": np.tile([1.0, 0.0, 0.0], (len(t), 1)),
            "state": np.full(len(t), "walking", dtype="U10"),
            "joint_angles": np.zeros((len(t), 42)),
            "upright": np.ones(len(t)),
            "support": np.tile((np.arange(len(t)) % 2)[:, None], (1, 6)),
        }
        spec = {
            "seconds": 10,
            "control_dt_s": 0.005,
            "policy": {"parameters": {"task": "walk"}},
            "pause_at_s": -1,
            "resume_at_s": -1,
        }
        result = evaluate_trace(rows, spec)
        self.assertGreater(result["first_10s_net_mm"], 5)
        self.assertFalse(result["checks"]["no_stuck_active_window"])
        self.assertFalse(result["passed"])
        rows["state"][t > 5] = "stopped"
        self.assertTrue(evaluate_trace(rows, spec)["checks"]["no_stuck_active_window"])

    def test_bilateral_turn_and_odor_ablation(self):
        left = SensoryWalkingPolicy().step(packet((0.1, 0.02)), 0.005)
        right = SensoryWalkingPolicy().step(packet((0.02, 0.1)), 0.005)
        self.assertLess(left["yawRate"], 0)
        self.assertEqual(left["yawRate"], -right["yawRate"])
        cut = SensoryWalkingPolicy().step(packet((0.1, 0.02)), 0.005, ["odor_mean"])
        self.assertEqual(cut["yawRate"], 0)
        self.assertGreater(cut["forwardSpeed"], 0)

    def test_arrival_dwell_hysteresis_and_departure(self):
        policy = SensoryWalkingPolicy()
        for _ in range(19):
            self.assertGreater(
                policy.step(packet((0.4, 0.4)), 0.005)["forwardSpeed"], 0
            )
        self.assertEqual(policy.step(packet((0.4, 0.4)), 0.005)["forwardSpeed"], 0)
        self.assertEqual(policy.step(packet((0.22, 0.22)), 0.005)["forwardSpeed"], 0)
        self.assertGreater(policy.step(packet((0.05, 0.05)), 0.005)["forwardSpeed"], 0)

    def test_stop_overrides_obstacle_and_invalid_packet_preserves_state(self):
        policy = SensoryWalkingPolicy()
        policy.enabled = False
        self.assertEqual(policy.step(packet(contact=1), 0.005)["forwardSpeed"], 0)
        before = policy.snapshot()
        with self.assertRaises(ValueError):
            policy.step(packet((float("nan"), 0.1)), 0.005)
        same_state(before, policy.snapshot())

    def test_policy_restore_preserves_avoidance_and_arrival_future(self):
        a = SensoryWalkingPolicy()
        a.step(packet(contact=1), 0.005)
        b = SensoryWalkingPolicy()
        b.restore(a.snapshot())
        for value in [(0.4, 0.4)] * 40 + [(0.1, 0.1)] * 50:
            self.assertEqual(a.step(packet(value), 0.005), b.step(packet(value), 0.005))
        same_state(a.snapshot(), b.snapshot())

    def test_engine_source_cuts_checkpoint_and_strict_boundary(self):
        graph = graph_fixture()
        binding = bindings_fixture(graph)
        # This test exercises protocol/ownership with an explicit body and
        # sensor test double. Actual antenna placement is tested in MuJoCo.
        with patch(
            "flylab.c.engine.CSensorAdapter",
            side_effect=lambda seed, model: CSensorAdapter(seed),
        ):
            engine = CEngine(
                graph, binding, mode=AUTONOMOUS_MODE, body_factory=AutonomousFixtureBody
            )
            restored = None
            try:
                frame = engine.step(20)
                self.assertIsNone(engine.neural)
                self.assertFalse(frame["scope"]["fullBrain"])
                self.assertEqual(frame["scope"]["simulatedNodes"], 0)
                header, *_ = decode_signals(signal_frame(engine))
                self.assertEqual(frame["subscription"]["ids"], header["ids"])
                self.assertEqual(header["ids"], [])
                with self.assertRaisesRegex(ValueError, "does not simulate neurons"):
                    engine.subscribe([graph.nodes[0]["id"]])
                self.assertEqual(
                    frame["command"]["command_source"], "flygym_sensory_policy"
                )
                self.assertGreater(frame["body"]["travel"], 0)
                with self.assertRaisesRegex(ValueError, "Neural interventions"):
                    engine.schedule(
                        {"kind": "stimulate", "ids": [graph.nodes[0]["id"]]}
                    )
                saved = engine.checkpoint()
                restored = CEngine.from_checkpoint(
                    graph, binding, saved, body_factory=AutonomousFixtureBody
                )
                engine.step(25)
                restored.step(25)
                same_state(engine.body.snapshot(), restored.body.snapshot())
                same_state(engine.autonomy.snapshot(), restored.autonomy.snapshot())
                engine.schedule({"kind": "motor_disconnect", "duration_controls": 4})
                self.assertEqual(
                    engine.step(1)["command"]["u_final"]["forwardSpeed"], 0
                )
                self.assertGreater(
                    engine.step(4)["command"]["u_final"]["forwardSpeed"], 0
                )
                engine.command("autonomy", {"enabled": False})
                self.assertEqual(
                    engine.step(1)["command"]["u_final"]["forwardSpeed"], 0
                )
                tampered = copy.deepcopy(saved)
                tampered["autonomy"]["tick"] += 1
                with self.assertRaisesRegex(RuntimeError, "Autonomous/control clock"):
                    CEngine.from_checkpoint(
                        graph, binding, tampered, body_factory=AutonomousFixtureBody
                    )
            finally:
                engine.close()
                if restored:
                    restored.close()

    def test_autonomy_options_cannot_change_strict_or_use_a_tether(self):
        graph = graph_fixture()
        binding = bindings_fixture(graph)
        with self.assertRaisesRegex(ValueError, "require FLYGYM_AUTONOMOUS"):
            CEngine(graph, binding, mode="C_STRICT", autonomy={})
        with self.assertRaisesRegex(ValueError, "requires a free body"):
            CEngine(
                graph,
                binding,
                mode=AUTONOMOUS_MODE,
                body_options={"attachment": "tethered"},
            )


if __name__ == "__main__":
    unittest.main()
