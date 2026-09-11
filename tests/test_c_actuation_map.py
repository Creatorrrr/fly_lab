"""Coverage follows instantiated physical order, never anatomical names alone."""

import copy
import os
import unittest
from pathlib import Path

import numpy as np

from flylab.c.neuromuscular import NeuromuscularLoop, build_spec
from flylab.engine import config_values
from flylab.sensors import default_world
from tests.test_c_sensorimotor import LegBody, banc_fixture


class ActuationMapTests(unittest.TestCase):
    def test_reordered_physical_axes_and_isolated_return_value(self):
        graph = banc_fixture()
        body = LegBody(42, default_world(), config_values())
        body.joint_names.reverse()
        loop = NeuromuscularLoop(graph, build_spec(graph), body)
        before = loop.snapshot()
        mapping = loop.summary()["actuation"]
        self.assertEqual([row["name"] for row in mapping["axes"]], body.joint_names)
        self.assertEqual(mapping["mapped_axes"], loop.spec["covered_dofs"])
        for row in mapping["axes"]:
            self.assertGreater(row["motor_neurons"], 0)
            self.assertEqual(row["status"], "engineering_map")
            self.assertIn("/" + row["region"], row["name"].replace("c_thorax-", ""))
        mapping["axes"][0]["status"] = "corrupted"
        self.assertEqual(
            loop.summary()["actuation"]["axes"][0]["status"], "engineering_map"
        )
        after = loop.snapshot()
        for key in ("filtered", "delayed", "offset"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(before["hash"], after["hash"])

    def test_nonleg_motor_is_not_mapped_by_anatomical_region(self):
        graph = banc_fixture()
        source = next(node for node in graph.nodes if node["super_class"] == "motor")
        extra = copy.deepcopy(source)
        extra.update(id="fixture:unbound-wing", cell_type="wing motor")
        extra["source_annotations"]["body_part_effector"] = "wing"
        graph.nodes.append(extra)
        body = LegBody(42, default_world(), config_values())
        loop = NeuromuscularLoop(graph, build_spec(graph), body)
        mapping = loop.summary()["actuation"]
        self.assertEqual(mapping["unmapped_motor_neurons"], 1)
        self.assertFalse(mapping["biological_validation"])


@unittest.skipUnless(
    os.environ.get("FLYLAB_NATIVE_TESTS") == "1", "Opt-in native body and local graph"
)
class NativeActuationMapTests(unittest.TestCase):
    def test_actual_42_and_78_axis_mapping_preserves_body(self):
        from flylab.body import FlyGymBody
        from flylab.body_options import BodyOptions
        from flylab.c.graph import GraphStore

        path = Path("data/acquisitions/banc888-windows-20260910/bundle")
        if not path.is_dir():
            self.skipTest("Local BANC v888 graph is required")
        graph = GraphStore.load(path)
        for scope, expected in (("legs", (42, 1)), ("whole_body", (78, 37))):
            options = BodyOptions(
                model="flybody",
                **(
                    {"actuation": scope, "servo_profile": "tracking_all"}
                    if scope == "whole_body"
                    else {}
                ),
            )
            body = FlyGymBody(
                42, default_world(), config_values(), body_options=options
            )
            try:
                before = body.snapshot()
                loop = NeuromuscularLoop(graph, build_spec(graph, 2), body)
                mapping = loop.summary()["actuation"]
                self.assertEqual(
                    (mapping["active_axes"], mapping["unmapped_axes"]), expected
                )
                self.assertEqual(mapping["mapped_axes"], 41)
                self.assertEqual(
                    (
                        mapping["graph_motor_neurons"],
                        mapping["mapped_motor_neurons"],
                        mapping["unmapped_motor_neurons"],
                    ),
                    (805, 391, 414),
                )
                rm = next(
                    row
                    for row in mapping["axes"]
                    if row["name"].endswith("c_thorax-rm_coxa-pitch")
                )
                self.assertEqual(rm["status"], "unmapped")
                if scope == "whole_body":
                    self.assertEqual(mapping["passive_joints"], 24)
                    self.assertEqual(
                        body.model_hash,
                        "0716877f304c75b77d0b25ce97a9899cd1ce1495399f6ff63bbd9c9df1cdf076",
                    )
                    self.assertEqual(
                        [row["name"] for row in mapping["axes"]],
                        body.whole_body_observation()["names"],
                    )
                    self.assertTrue(
                        all(
                            row["status"] == "unmapped"
                            for row in mapping["axes"]
                            if row["region"] not in ("lf", "lm", "lh", "rf", "rm", "rh")
                        )
                    )
                else:
                    self.assertEqual(
                        body.model_hash,
                        "24cc93922f734bc72d11d1b9bd139318d7331399626cc51c3abc60c4f0e28bdc",
                    )
                self.assertEqual(body.snapshot(), before)
            finally:
                body.close()
