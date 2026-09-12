"""Sensory causality and final-actuator cut contracts on a tiny anatomical fixture."""

import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from flylab.c.neuromuscular import LEGS, PARTS, joint_suffixes
from flylab.c.rate_body import RateBodyAdapter
from flylab.kinematic_senses import segment_flexion


def fixture():
    nodes = []
    for leg in LEGS:
        for kind, cell_type, target in (
            ("motor", "flexor", "tibia_flexor_muscle"),
            ("motor", "extensor", "tibia_extensor_muscle"),
            ("motor", "depressor", "long_tendon_muscle"),
            ("sensory", "SNpp50", "chordotonal_organ"),
            ("sensory", "SNpp51", "chordotonal_organ"),
        ):
            nodes.append(
                {
                    "id": str(len(nodes)),
                    "cell_type": cell_type,
                    "super_class": kind,
                    "soma_side": "left" if leg[0] == "l" else "right",
                    "class": "chordotonal_organ_neuron"
                    if kind == "sensory"
                    else "motor",
                    "source_annotations": {
                        "body_part_effector"
                        if kind == "motor"
                        else "body_part_sensory": PARTS[leg[1]],
                        "peripheral_target_type": target,
                        "cell_sub_class": "femoral_chordotonal_claw_neuron",
                    },
                }
            )
    graph = SimpleNamespace(
        n=len(nodes),
        nodes=nodes,
        hash="fixture",
        manifest={"dataset_id": "flywire_banc", "snapshot_id": "888"},
        resolve=lambda ids, **_: np.asarray([int(i) for i in ids], dtype=int),
    )
    angles = np.tile([0, 0, 0, 0, 0, math.pi / 4, 0.0], 6)
    body = SimpleNamespace(
        model_hash="fixture-body",
        neutral=angles.copy(),
        knee_kinematics=lambda: {
            "flexion_rad": angles.reshape(6, 7)[:, 5].copy(),
            "velocity_rad_s": np.zeros(6),
            "d_flexion_dq": np.ones(6),
        },
        joint_names=["fly" + suffix for leg in LEGS for suffix in joint_suffixes(leg)],
        foot_kinematics=lambda _: {"target_lift_mm": np.zeros(6)},
        leg_observation=lambda: {
            "angles_rad": angles.copy(),
            "velocities_rad_s": np.zeros(42),
            "support_load_bw": np.ones(6),
            "non_support_load_bw": np.zeros(6),
        },
    )
    return graph, body, angles


class RateBodyContracts(unittest.TestCase):
    def setUp(self):
        self.graph, self.body, self.angles = fixture()
        self.adapter = RateBodyAdapter(self.graph, self.body)

    def test_real_angle_drives_correct_typed_sensory_cells_with_latency(self):
        np.testing.assert_array_equal(self.adapter.encode(), np.zeros(self.graph.n))
        response = self.adapter.encode()
        scale = 18 * (1 - math.exp(-0.005 / 0.02))
        for leg in range(6):
            self.assertAlmostEqual(response[leg * 5 + 3], scale * 0.25, places=6)
            self.assertAlmostEqual(response[leg * 5 + 4], scale * 0.75, places=6)
        self.angles[5] = math.pi
        self.adapter.encode()
        changed = self.adapter.encode()
        self.assertGreater(changed[3], changed[8])
        self.assertLess(changed[4], changed[9])
        np.testing.assert_array_equal(
            self.adapter.encode(sensory_cut=True), np.zeros(self.graph.n)
        )

    def test_motor_antagonists_and_disconnect_at_final_actuator(self):
        rates = np.zeros(len(self.adapter.motor_ids))
        neutral, adhesive = self.adapter.decode(rates)
        np.testing.assert_array_equal(neutral, self.body.neutral)
        self.assertFalse(adhesive.any())
        rates[self.adapter.motor_columns[0]] = 100
        rates[self.adapter.motor_columns[7]] = 100
        target, _ = self.adapter.decode(rates)
        self.assertGreater(target[5], neutral[5])
        self.assertLess(target[13], neutral[13])
        rates.fill(200)
        disconnected, adhesive = self.adapter.decode(rates, motor_cut=True)
        np.testing.assert_array_equal(disconnected, self.body.neutral)
        self.assertFalse(adhesive.any())
        np.testing.assert_array_equal(self.adapter.offset, np.zeros(42))

    def test_physical_hinge_sign_changes_flexor_target_direction(self):
        self.body.knee_kinematics = lambda: {
            "flexion_rad": np.ones(6),
            "velocity_rad_s": np.zeros(6),
            "d_flexion_dq": -np.ones(6),
        }
        adapter = RateBodyAdapter(self.graph, self.body)
        rates = np.zeros(len(adapter.motor_ids))
        rates[adapter.motor_columns[0]] = 100.0
        target, _ = adapter.decode(rates)
        self.assertLess(target[5], self.body.neutral[5])

    def test_ltm_reaches_distal_tendon_without_double_joint_drive(self):
        names = [leg + "_tarsus" for leg in LEGS]
        self.body.tendon_control = SimpleNamespace(
            index={name: i for i, name in enumerate(names)},
            limits=np.tile([-1.0, 1.0], (6, 1)),
        )
        adapter = RateBodyAdapter(self.graph, self.body, ltm_tendons=True)
        rates = np.zeros(len(adapter.motor_ids))
        rates[adapter.motor_columns[2]] = 100.0
        targets, _ = adapter.decode(rates)
        np.testing.assert_array_equal(targets, self.body.neutral)
        self.assertGreater(adapter.tendon_inputs["lf_tarsus"], 0.0)
        self.assertEqual(adapter.tendon_inputs["rf_tarsus"], 0.0)
        saved = adapter.snapshot()
        other = RateBodyAdapter(self.graph, self.body, ltm_tendons=True)
        other.restore(saved)
        adapter.decode(rates)
        other.decode(rates)
        self.assertEqual(adapter.tendon_inputs, other.tendon_inputs)
        with self.assertRaises(ValueError):
            self.adapter.restore(saved)
        invalid = copy.deepcopy(saved)
        invalid["tendon_activation"][0] = np.nan
        before = other.snapshot()
        with self.assertRaises(ValueError):
            other.restore(invalid)
        np.testing.assert_array_equal(
            before["tendon_activation"], other.tendon_activation
        )
        adapter.decode(rates, motor_cut=True)
        self.assertTrue(all(value == 0.0 for value in adapter.tendon_inputs.values()))

    def test_anterior_coxa_rotator_uses_physical_direction_and_distinct_state(self):
        neuron = copy.deepcopy(self.graph.nodes[0])
        neuron["id"] = str(len(self.graph.nodes))
        neuron["source_annotations"]["peripheral_target_type"] = (
            "sternal_anterior_rotator_muscle"
        )
        self.graph.nodes.append(neuron)
        self.graph.n = len(self.graph.nodes)
        self.body.coxa_rotation_kinematics = lambda: {"d_anterior_dq": -np.ones(6)}
        adapter = RateBodyAdapter(self.graph, self.body, coxa_geometry=True)
        rates = np.zeros(len(adapter.motor_ids))
        rates[adapter.motor_columns[int(neuron["id"])]] = 100.0
        target, _ = adapter.decode(rates)
        self.assertLess(target[0], self.body.neutral[0])
        with self.assertRaises(ValueError):
            RateBodyAdapter(self.graph, self.body).restore(adapter.snapshot())
        self.body.coxa_rotation_kinematics = lambda: {"d_anterior_dq": np.zeros(6)}
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            RateBodyAdapter(self.graph, self.body, coxa_geometry=True)

    def test_flexion_measurement_is_invariant_to_rigid_motion(self):
        p = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0]])
        v = np.zeros((3, 3))
        v[2, 0] = 1.0
        angle, speed = segment_flexion(p, v)
        self.assertAlmostEqual(angle, math.pi / 2)
        self.assertAlmostEqual(speed, 1.0)
        rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        angular = np.array([0.2, 0.3, 0.4])
        translated = p @ rotation.T + np.array([5.0, -2.0, 3.0])
        moving = (
            v @ rotation.T + np.cross(angular, translated) + np.array([0.3, 0.4, 0.7])
        )
        transformed = segment_flexion(translated, moving)
        np.testing.assert_allclose(transformed, [angle, speed], atol=1e-12)
        with self.assertRaises(ValueError):
            segment_flexion(np.zeros((3, 3)), v)

    def test_restore_retains_future_and_rejects_partial_state(self):
        self.adapter.encode()
        self.adapter.decode(np.arange(len(self.adapter.motor_ids)))
        saved = self.adapter.snapshot()
        other = RateBodyAdapter(self.graph, self.body)
        other.restore(saved)
        np.testing.assert_array_equal(self.adapter.encode(), other.encode())
        rates = np.full(len(self.adapter.motor_ids), 50.0)
        for a, b in zip(self.adapter.decode(rates), other.decode(rates)):
            np.testing.assert_array_equal(a, b)
        before = self.adapter.snapshot()
        invalid = copy.deepcopy(before)
        invalid["offset"].fill(0.3)
        invalid["filtered"][0] = np.inf
        with self.assertRaises(ValueError):
            self.adapter.restore(invalid)
        for key in ("offset", "filtered", "velocity_lowpass"):
            np.testing.assert_array_equal(before[key], self.adapter.snapshot()[key])

    def test_actual_flybody_knee_jacobian_matches_segment_finite_difference(self):
        import mujoco

        from flylab.body import FlyGymBody
        from flylab.body_options import BodyOptions
        from flylab.engine import config_values
        from flylab.sensors import default_world

        body = FlyGymBody(
            42,
            default_world(),
            config_values(None),
            body_options=BodyOptions(
                model="flybody",
                actuation="whole_body",
                servo_profile="tracking_all",
                tendons="all",
            ),
        )
        try:
            original = body.d.qpos.copy()
            calibration = body.knee_kinematics()
            derivatives = []
            for index, leg in enumerate(LEGS):
                joint = body.joint_names.index(
                    next(
                        name
                        for name in body.joint_names
                        if name.endswith(joint_suffixes(leg)[5])
                    )
                )
                address = body.qpos_ids[joint]
                values = []
                for delta in (-1e-6, 1e-6):
                    body.d.qpos[:] = original
                    body.d.qpos[address] += delta
                    mujoco.mj_forward(body.m, body.d)
                    values.append(body.knee_kinematics()["flexion_rad"][index])
                derivatives.append((values[1] - values[0]) / 2e-6)
            np.testing.assert_allclose(
                derivatives, calibration["d_flexion_dq"], atol=1e-8
            )
            self.assertTrue(np.all(np.asarray(derivatives) < -0.99))
            body.d.qpos[:] = original
            mujoco.mj_forward(body.m, body.d)
            anterior = body.coxa_rotation_kinematics()["d_anterior_dq"]
            forward = body.pose()[1][:, 0].copy()
            coxa_differences = []
            for leg in LEGS:
                endpoint = int(
                    body.body_ids[body.body_indices[f"{leg}_trochanterfemur"]]
                )
                joint = next(
                    i
                    for i, name in enumerate(body.joint_names)
                    if name.endswith(joint_suffixes(leg)[0])
                )
                samples = []
                for delta in (-1e-6, 1e-6):
                    body.d.qpos[:] = original
                    body.d.qpos[body.qpos_ids[joint]] += delta
                    mujoco.mj_forward(body.m, body.d)
                    samples.append(float(forward @ body.d.xpos[endpoint]))
                coxa_differences.append((samples[1] - samples[0]) / 2e-6)
            np.testing.assert_allclose(coxa_differences, anterior, atol=1e-8)
            self.assertTrue(np.all(anterior < -0.1))
        finally:
            body.close()

    def test_endpoint_proxy_moves_actual_coxa_in_declared_directions(self):
        import mujoco

        from flylab.body import FlyGymBody
        from flylab.body_options import BodyOptions
        from flylab.engine import config_values
        from flylab.sensors import default_world

        directions = {
            "tergopleural_promotor_muscle": np.array([1.0, 0.0]),
            "pleural_remotor_and_abductor_muscle": np.array([-1.0, -1.0]),
            "sternal_adductor_muscle": np.array([0.0, 1.0]),
        }
        for leg in range(6):
            for muscle in directions:
                node = copy.deepcopy(self.graph.nodes[leg * 5])
                node["id"] = str(len(self.graph.nodes))
                node["source_annotations"]["peripheral_target_type"] = muscle
                self.graph.nodes.append(node)
        self.graph.n = len(self.graph.nodes)
        body = FlyGymBody(
            42,
            default_world(),
            config_values(None),
            body_options=BodyOptions(
                model="flybody",
                actuation="whole_body",
                servo_profile="tracking_all",
                tendons="all",
            ),
        )
        try:
            adapter = RateBodyAdapter(
                self.graph, body, coxa_geometry=True, coxa_endpoint=True
            )
            original = body.d.qpos.copy()
            rotation = body.pose()[1]
            for leg, groups in enumerate(adapter.muscles):
                endpoint = body.body_ids[
                    body.body_indices[LEGS[leg] + "_trochanterfemur"]
                ]
                axes = rotation[:, [0, 1]].copy()
                axes[:, 1] *= -1 if leg < 3 else 1
                for name, _, vector in groups:
                    if name not in directions:
                        continue
                    positions = []
                    for scale in (-1e-6, 1e-6):
                        body.d.qpos[:] = original
                        body.d.qpos[body.qpos_ids[adapter.joints[leg, :3]]] += (
                            scale * vector[:3]
                        )
                        mujoco.mj_forward(body.m, body.d)
                        positions.append(axes.T @ body.d.xpos[endpoint])
                    observed = (positions[1] - positions[0]) / 2e-6
                    observed /= np.max(np.abs(observed))
                    np.testing.assert_allclose(observed, directions[name], atol=1e-7)
            self.assertEqual(adapter.schema, "flylab.rate-body.v5")
        finally:
            body.close()


if __name__ == "__main__":
    unittest.main()
