"""BANC workbench admission and isolation from the existing main experiment."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from flylab.c import PROTOCOL
from flylab.c.banc_walking import (
    BancWalkingParameters,
    BancWalkingSession,
    validate_banc_roster,
)
from flylab.c.integrity import write_json
from flylab.c.server import CDispatcher
from tests.c_fixtures import bindings_fixture, graph_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


class BancWorkbenchContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        graph = graph_fixture()
        graph.save(self.path / "graph")
        write_json(self.path / "bindings.json", bindings_fixture(graph).spec)
        self.dispatcher = CDispatcher(
            self.path / "graph",
            self.path / "bindings.json",
            self.path / "artifacts",
            FixtureBody,
        )
        self.call("init", {"mode": "C_STRICT"})

    def tearDown(self):
        self.dispatcher.close()
        self.temp.cleanup()

    def call(self, op, payload=None):
        return self.dispatcher.handle(
            {"protocol": PROTOCOL, "requestId": 1, "op": op, "payload": payload or {}}
        )

    def test_non_banc_graph_is_rejected_without_replacing_main(self):
        main = self.dispatcher.engine
        before = main.checkpoint()
        with self.assertRaisesRegex(ValueError, "BANC v888"):
            self.call("banc_walking_init")
        self.assertIs(self.dispatcher.engine, main)
        self.assertIsNone(self.dispatcher.banc_walking)
        same_state(before, main.checkpoint())

    def test_rate_controls_are_bounded_and_do_not_advance_main(self):
        called = []
        self.dispatcher.banc_walking = SimpleNamespace(
            advance=called.append, close=lambda: None
        )
        self.dispatcher.banc_walking_result = lambda: {"observation": {"time_s": 0.1}}
        before = self.dispatcher.engine.checkpoint()
        for count in (-1, 0, 21, True, 1.5):
            with self.assertRaises(ValueError):
                self.call("banc_walking_advance", {"steps": count})
        self.assertFalse(called)
        self.call("banc_walking_advance", {"steps": 10})
        self.assertEqual(called, [10])
        same_state(before, self.dispatcher.engine.checkpoint())

    def test_model_parameters_reject_nonfinite_and_wrong_types(self):
        for field, value in (
            ("motor_gain", float("nan")),
            ("adaptation_gain", True),
            ("sensory_gain", 401),
            ("descending_drive", -1),
            ("pooling", "CPG"),
        ):
            with self.assertRaises(ValueError):
                BancWalkingParameters(**{field: value})

    def test_small_graph_cannot_claim_the_banc_research_roster(self):
        graph = graph_fixture()
        graph.manifest.update(
            dataset_id="flywire_banc",
            snapshot_id="888",
            scope="declared_neuronal_roster",
        )
        with self.assertRaisesRegex(ValueError, "Pinned BANC neuronal roster"):
            validate_banc_roster(graph)

    def test_nonfinite_physics_clock_is_rejected(self):
        session = object.__new__(BancWalkingSession)
        session.control_tick = 0
        session.network = SimpleNamespace(tick=0)
        session.body = SimpleNamespace(physics_time=lambda: float("nan"))
        with self.assertRaisesRegex(RuntimeError, "clocks diverged"):
            session._clocks()


if __name__ == "__main__":
    unittest.main()
