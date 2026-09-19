"""Source identity, anatomical side and explicit hypothesis contracts."""
import unittest
from types import SimpleNamespace

import numpy as np

from flylab.c.banc_walking import BancWalkingParameters
from flylab.c.coxa_hair_plate import ROOTS, anterior_feature, front_coxa_ports


def fixture():
    nodes = []
    for side, roots in zip(("left", "right"), ROOTS):
        for index, root in enumerate(roots):
            nodes.append({
                "id": "flywire:banc:888:" + root, "class": "hair_plate_neuron",
                "super_class": "sensory", "soma_side": side, "nt_type": "ACH",
                "cell_type": "SNpp45" if index < 4 else "SNpp52",
                "source_annotations": {
                    "body_part_sensory": "coxa,front_leg",
                    "peripheral_target_type": "CoHP8,hair_plate",
                    "nerve": side + "_ventral_prothoracic_nerve",
                },
            })
    lookup = {node["id"]: i for i, node in enumerate(nodes)}
    return SimpleNamespace(nodes=nodes, resolve=lambda ids: [lookup[i] for i in ids])


class FrontHairPlateContracts(unittest.TestCase):
    def test_source_memberships_and_types_are_preserved(self):
        graph = fixture()
        rows = front_coxa_ports(graph)
        self.assertEqual(set(rows), {0, 3})
        for leg, roots in zip((0, 3), ROOTS):
            actual = {i for port in rows[leg] for i in port["ids"]}
            self.assertEqual(actual, {"flywire:banc:888:" + root for root in roots})
            for port in rows[leg]:
                for i in graph.resolve(port["ids"]):
                    self.assertEqual(port["cell_type"], graph.nodes[i]["cell_type"])

    def test_conflicting_nerve_or_side_or_organ_is_rejected(self):
        for key, value in (("nerve", "left_prothoracic_leg_nerve"),
                           ("body_part_sensory", "front_leg,trochanter"),
                           ("peripheral_target_type", "hair_plate")):
            graph = fixture()
            graph.nodes[0]["source_annotations"][key] = value
            with self.assertRaises(ValueError):
                front_coxa_ports(graph)
        for key, value in (("soma_side", "right"), ("nt_type", "GABA")):
            graph = fixture()
            graph.nodes[0][key] = value
            with self.assertRaises(ValueError):
                front_coxa_ports(graph)

    def test_mirrored_physical_axis_and_nonanterior_motion(self):
        np.testing.assert_array_equal(
            anterior_feature(np.array([.25, -.25]), np.zeros(2), np.array([1., -1.])),
            np.array([.5, .5]),
        )
        self.assertEqual(anterior_feature(-.1, 0., 1.), 0.)
        self.assertEqual(anterior_feature(1., 0., 1.), 1.)
        self.assertEqual(anterior_feature(.3, .3, -1.), 0.)

    def test_option_is_explicit_and_needs_physical_direction(self):
        self.assertNotIn("front_hair_plate", BancWalkingParameters().state_dict())
        enabled = BancWalkingParameters(coxa_geometry=True, front_hair_plate=True)
        self.assertTrue(enabled.state_dict()["front_hair_plate"])
        self.assertEqual(enabled, BancWalkingParameters(**enabled.state_dict()))
        for args in ({"front_hair_plate": True}, {"front_hair_plate": 1}):
            with self.assertRaises(ValueError):
                BancWalkingParameters(**args)


if __name__ == "__main__":
    unittest.main()
