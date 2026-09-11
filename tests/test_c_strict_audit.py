"""Counterexamples for experiment setup and portable scientific records."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flylab.c.campaign import pilot_spec, run_campaign, validate_spec
from flylab.c.engine import CEngine
from flylab.c.experiments import replay_recording
from flylab.c.integrity import read_json
from flylab.c.ports import PortBindings
from flylab.c.sensorimotor_campaign import fixed_spec
from tests.c_fixtures import bindings_fixture, graph_fixture
from tests.fixture_body import FixtureBody
from tests.test_c import same_state


class StrictExperimentTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_fixture()
        self.binding = bindings_fixture(self.graph)

    def test_nondefault_startup_profile_keeps_the_base_profile_selectable(self):
        from flylab.c.integrity import write_json
        from flylab.c.server import CDispatcher

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.graph.save(root / "graph")
            write_json(root / "bindings.json", self.binding.spec)
            alternate = dict(self.binding.spec, profile="test-alternate")
            write_json(root / "bindings-alternate.json", alternate)
            dispatcher = CDispatcher(
                root / "graph",
                root / "bindings-alternate.json",
                root / "artifacts",
                FixtureBody,
            )
            try:
                self.assertIn("bindings.json", dispatcher.profiles())
            finally:
                dispatcher.close()

    def test_unknown_sensory_cut_rejected_before_campaign_creates_files(self):
        spec = pilot_spec(0.01, seeds=(42,), modes=("C_STRICT",))
        spec["cases"] = spec["cases"][:1]
        spec["cases"][0]["intervention"] = {
            "kind": "sensor_off",
            "channels": ["odor_left"],
            "duration_controls": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "not-started"
            with self.assertRaisesRegex(ValueError, "sensory channels"):
                run_campaign(
                    self.graph, self.binding, spec, out, body_factory=FixtureBody
                )
            self.assertFalse(out.exists())

    def test_food_control_uses_profile_channels_including_mean_and_sites(self):
        for channels, model in (
            (["odor_mean"], {}),
            (
                ["odor_antenna_left_food", "odor_antenna_right_food"],
                {"kind": "four-site-odor-v1"},
            ),
        ):
            spec = copy.deepcopy(self.binding.spec)
            spec["sensor_model"] = model
            spec["sensory"] = [
                dict(spec["sensory"][0], name=f"port-{i}", channel=channel)
                for i, channel in enumerate(channels)
            ]
            binding = PortBindings(self.graph, spec)
            cases = fixed_spec(
                0.01,
                conditions=({"seed": 42},),
                bindings=binding,
                scenes=("food_left",),
            )
            validate_spec(cases, binding)
            self.assertEqual(cases["cases"][1]["intervention"]["channels"], channels)

    def test_missing_hazard_ports_do_not_produce_an_invalid_control_campaign(self):
        with self.assertRaisesRegex(ValueError, "hazard"):
            fixed_spec(
                0.01,
                conditions=({"seed": 42},),
                bindings=self.binding,
                scenes=("hazard_left",),
            )

    def test_declared_leg_selectors_match_the_running_neuromuscular_loop(self):
        from tests.test_c_contracts import banc_bindings
        from tests.test_c_sensorimotor import LegBody

        graph, binding = banc_bindings()
        engine = CEngine(graph, binding, mode="C_STRICT", body_factory=LegBody)
        try:
            actual = (
                {p["name"] for p, _ in binding.sensory}
                | {p["channel"] for p, _ in binding.sensory}
                | {"*"}
                | engine.neuromuscular.channels
            )
            self.assertEqual(actual, binding.sensory_selectors)
        finally:
            engine.close()

    def test_named_odor_profile_does_not_silently_get_a_generic_food_assay(self):
        from types import SimpleNamespace

        binding = SimpleNamespace(spec={"sensory": [{"channel": "chemical_odor"}]})
        with self.assertRaisesRegex(ValueError, "named-odorant"):
            fixed_spec(
                0.01,
                conditions=({"seed": 42},),
                bindings=binding,
                scenes=("food_left",),
            )

    def test_hazard_control_must_cover_every_side_in_the_raw_records(self):
        from flylab.c.behavior import trace_sample
        from flylab.c.campaign import scene_world
        from flylab.c.sensorimotor_campaign import evaluate_hazard

        engine = CEngine(
            self.graph, self.binding, mode="C_STRICT", body_factory=FixtureBody
        )
        try:
            first = trace_sample(engine.frame())
            provenance = engine.provenance()
        finally:
            engine.close()
        channels = ["odor_antenna_left_hazard", "odor_antenna_right_hazard"]
        provenance["bindings"]["sensory"] = [
            {"name": name, "channel": channel}
            for name, channel in zip(("left", "right"), channels)
        ]
        world = scene_world("hazard_right")
        world["sources"][0]["p"] = [8.0, 0.7, 0.0]

        def arm(avoid=False, off=False, free=False):
            rows = []
            for i in range(2001):
                t = i * 0.005
                rows.append(
                    dict(
                        copy.deepcopy(first),
                        simTime=t,
                        tick=i * 50,
                        position=[t, 0.9, -0.4 * t if avoid else 0.0],
                        sensory_ports=[
                            {
                                "name": name,
                                "enabled": not off,
                                "value": 0.0 if off else 1.0,
                            }
                            for name in ("left", "right")
                        ],
                    )
                )
            w = copy.deepcopy(world)
            if free:
                w["sources"] = []
            return {
                "trace": rows,
                "world": w,
                "provenance": copy.deepcopy(provenance),
                "interventions": [
                    {
                        "kind": "sensor_off",
                        "channels": channels,
                        "duration_controls": 2000,
                    }
                ]
                if off
                else [],
            }

        active, free, off = arm(avoid=True), arm(free=True), arm(off=True)
        self.assertEqual(evaluate_hazard(active, free, off)["task_status"], "PASS")
        off["trace"][100]["sensory_ports"].pop()
        result = evaluate_hazard(active, free, off)
        self.assertEqual(result["task_status"], "INCOMPLETE")
        self.assertEqual(
            result["reasons"], ["Hazard sensory-off control is unverified"]
        )

    def test_replay_reads_utf8_environment_ids_on_a_non_utf8_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "record"
            engine = CEngine(
                self.graph, self.binding, mode="C_STRICT", body_factory=FixtureBody
            )
            try:
                engine.start_recording(out)
                world = copy.deepcopy(engine.world)
                world["sources"][0]["id"] = "먹이-왼쪽"
                engine.command("load_environment", {"world": world})
                engine.step(2)
                engine.stop_recording()
                expected = engine.checkpoint()
                original_open = Path.open

                def locale_open(path, *args, **kwargs):
                    if path.name == "events.jsonl" and "encoding" not in kwargs:
                        kwargs["encoding"] = "cp949"
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, "open", locale_open):
                    replay = replay_recording(
                        self.graph, self.binding, out, FixtureBody
                    )
                try:
                    actual = replay.checkpoint()
                    for key in ("world", "body", "neural", "sensors", "encoder"):
                        same_state(expected[key], actual[key])
                finally:
                    replay.close()
                self.assertEqual(read_json(out / "manifest.json")["status"], "COMPLETE")
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
