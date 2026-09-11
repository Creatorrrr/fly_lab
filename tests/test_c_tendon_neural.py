"""Neural causality, channel isolation, ownership and restart counterexamples."""

import copy
import os
import unittest
from types import SimpleNamespace

import numpy as np

from flylab.body import FlyGymBody
from flylab.c.engine import CEngine
from flylab.c.graph import GraphStore, external_id
from flylab.c.neural import ExpLIF
from flylab.c.neuromuscular import LEGS, NeuromuscularLoop, build_spec
from flylab.c.ports import PortBindings
from flylab.engine import config_values
from flylab.sensors import default_world
from flylab.tendon_control import TendonControl
from tests.test_c import same_state
from tests.test_c_contracts import banc_bindings
from tests.test_c_motor_units import PadBody
from tests.test_c_sensorimotor import banc_fixture


def tendon_graph():
    original = banc_fixture()
    nodes = copy.deepcopy(original.nodes)
    for side in ("left", "right"):
        root = str(1000 + len(nodes))
        nodes.append(
            dict(
                id=external_id(root, "banc", "888"),
                root_id=root,
                cell_type="MNad26",
                soma_side=side,
                super_class="motor",
                nt_type="ACH",
                regions=[],
                **{"class": "abdomen_motor_neuron"},
                source_annotations={
                    "body_part_effector": "abdomen",
                    "nerve": f"{side}_first_abdominal_nerve",
                },
            )
        )
    return GraphStore.from_edges(
        nodes,
        original.indices,
        np.repeat(np.arange(original.n, dtype=np.int32), np.diff(original.indptr)),
        original.counts,
        metadata={
            "dataset_id": "flywire_banc",
            "snapshot_id": "888",
            "scope": "fixture",
        },
    )


def tendon_bindings(graph):
    _, original = banc_bindings()
    spec = copy.deepcopy(original.spec)
    spec.update(graph_hash=graph.hash, neuromuscular=build_spec(graph, 2))
    return PortBindings(graph, spec)


class TendonBody(PadBody):
    """Test actuator boundary; native motion is checked separately."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whole_body_control = None
        self.d = SimpleNamespace(ctrl=np.zeros(8))
        control = self.tendon_control = TendonControl.__new__(TendonControl)
        control.body = self
        control.names = ["abdomen_pitch", "abdomen_yaw"] + [
            leg + "_tarsus" for leg in LEGS
        ]
        control.index = {name: i for i, name in enumerate(control.names)}
        control.limits = np.array([[-1.05, 0.7], [-0.7, 0.7]] + [[-0.9, 0.9]] * 6)
        control.act_ids = np.arange(8)
        control.joint_ids = np.arange(36)

    set_body_actuation = FlyGymBody.set_body_actuation

    def step_joint_targets(self, targets, adhesion, dt, *, tendon_inputs=None):
        if tendon_inputs:
            self.d.ctrl[:] = self.tendon_control.command_vector(tendon_inputs)
        super().step_joint_targets(targets, adhesion, dt)

    def snapshot(self):
        return dict(super().snapshot(), tendon_ctrl=self.d.ctrl.copy())

    def restore(self, state):
        super().restore(state)
        self.d.ctrl[:] = state["tendon_ctrl"]


class NeuralTendonTests(unittest.TestCase):
    def loop(self, version=2):
        graph = tendon_graph()
        body = TendonBody(42, default_world(), config_values())
        return graph, body, NeuromuscularLoop(graph, build_spec(graph, version), body)

    def test_ltm_reroutes_without_first_joint_double_drive_and_preserves_pooling(self):
        for version in (1, 2, 3):
            graph, body, loop = self.loop(version)
            neural = ExpLIF(graph)
            ids = next(
                ids for name, ids, _ in loop.muscles[0] if name == "long_tendon_muscle"
            )
            neural.rate[ids] = 100.0
            target, _ = loop.decode(neural)
            np.testing.assert_array_equal(target, body.neutral)
            self.assertGreater(loop.tendon_inputs["lf_tarsus"], 0)
            self.assertTrue(
                all(
                    value == 0
                    for name, value in loop.tendon_inputs.items()
                    if name != "lf_tarsus"
                )
            )
            source = next(
                r for r in loop.tendon_adapter.last if r["name"] == "lf_tarsus"
            )["sources"][0]
            self.assertEqual(source["output_Hz"], 50.0 if version == 3 else 100.0)
            self.assertEqual(loop.summary()["actuation"]["mapped_tendon_controls"], 8)
            # All inactive rates stay silent; no tonic tendon activation.
            loop.decode(neural, disconnected=True)
            self.assertTrue(all(v == 0 for v in loop.tendon_inputs.values()))

    def test_first_tarsal_joint_depressor_and_levator_remain_position_controls(self):
        for muscle, sign in (
            ("tarsus_depressor_muscle", -1),
            ("tarsus_levator_muscle", 1),
        ):
            graph, body, loop = self.loop()
            neural = ExpLIF(graph)
            ids = next(ids for name, ids, _ in loop.muscles[0] if name == muscle)
            neural.rate[ids] = 100.0
            target, _ = loop.decode(neural)
            self.assertGreater(
                sign * (target[loop.joints[0, 6]] - body.neutral[loop.joints[0, 6]]), 0
            )
            self.assertEqual(loop.tendon_inputs["lf_tarsus"], 0)

    def test_abdominal_bilateral_hypothesis_has_explicit_signs_and_no_leg_drive(self):
        results = []
        for sides in (("left",), ("right",), ("left", "right")):
            graph, body, loop = self.loop()
            neural = ExpLIF(graph)
            for i, node in enumerate(graph.nodes):
                if node["cell_type"] == "MNad26" and node["soma_side"] in sides:
                    neural.rate[i] = 100.0
            target, _ = loop.decode(neural)
            np.testing.assert_array_equal(target, body.neutral)
            self.assertTrue(
                all(loop.tendon_inputs[leg + "_tarsus"] == 0 for leg in LEGS)
            )
            results.append(loop.tendon_inputs.copy())
        self.assertGreater(results[0]["abdomen_pitch"], 0)
        self.assertEqual(results[0]["abdomen_pitch"], results[1]["abdomen_pitch"])
        self.assertLess(results[0]["abdomen_yaw"], 0)
        self.assertEqual(results[0]["abdomen_yaw"], -results[1]["abdomen_yaw"])
        self.assertEqual(results[2]["abdomen_yaw"], 0)
        self.assertGreater(results[2]["abdomen_pitch"], results[0]["abdomen_pitch"])

    def test_unresolved_or_contradictory_abdominal_identity_is_not_auto_connected(self):
        for field, value in (
            ("cell_type", "UNRESOLVED"),
            ("nerve", "left_second_abdominal_nerve"),
            ("cell_sub_class", "abdomen_neurosecretory_cell"),
        ):
            graph, body, _ = self.loop()
            node = next(
                n
                for n in graph.nodes
                if n["cell_type"] == "MNad26" and n["soma_side"] == "left"
            )
            (node if field == "cell_type" else node["source_annotations"])[field] = (
                value
            )
            loop = NeuromuscularLoop(graph, build_spec(graph, 2), body)
            rows = loop.summary()["actuation"]["tendon_controls"]
            self.assertTrue(all(row["status"] == "unmapped" for row in rows[:2]))
            self.assertEqual(loop.tendon_adapter.modes["abdomen_pitch"], "manual")
            with self.assertRaises(ValueError):
                loop.tendon_adapter.set_modes({"abdomen_pitch": "neural"})

    def test_invalid_rates_and_corrupt_filter_restore_are_atomic(self):
        graph, _, loop = self.loop()
        neural = ExpLIF(graph)
        neural.rate[loop.motor_indices] = 100.0
        loop.decode(neural)
        saved = loop.snapshot()
        for value in (float("nan"), -1.0):
            neural.rate[loop.motor_indices[0]] = value
            with self.assertRaises(ValueError):
                loop.decode(neural)
            same_state(saved, loop.snapshot())
        for key, value in (
            ("activation", np.full(8, 0.5)),
            ("modes", {"lf_tarsus": "neural"}),
            ("hash", "wrong"),
        ):
            bad = copy.deepcopy(saved)
            bad["tendons"][key] = value
            with self.assertRaises(ValueError):
                loop.restore(bad)
            same_state(saved, loop.snapshot())

    def test_engine_spikes_precede_inputs_disconnect_is_immediate_and_restore_repeats(
        self,
    ):
        graph = tendon_graph()
        bindings = tendon_bindings(graph)
        engine = CEngine(graph, bindings, mode="C_STRICT", body_factory=TendonBody)
        restored = None
        try:
            ids = next(
                g["ids"]
                for g in engine.neuromuscular.spec["rows"][0]["muscles"]
                if g["muscle"] == "long_tendon_muscle"
            )
            engine.schedule(
                {
                    "kind": "stimulate",
                    "ids": ids,
                    "amplitude_mV": 30.0,
                    "duration_controls": 30,
                }
            )
            engine.step(1)
            self.assertEqual(engine.body.d.ctrl[2], 0.0)
            engine.step(8)
            self.assertGreater(engine.body.d.ctrl[2], 0.0)
            self.assertGreater(
                engine.neural.readout(graph.resolve(ids))["spike_count"].sum(), 0
            )
            restored = CEngine.from_checkpoint(
                graph, bindings, engine.checkpoint(), TendonBody
            )
            corrupt = engine.checkpoint()
            corrupt["body"]["tendon_ctrl"][2] = 0.0
            with self.assertRaisesRegex(ValueError, "restored physical inputs"):
                CEngine.from_checkpoint(graph, bindings, corrupt, TendonBody)
            for e in (engine, restored):
                e.step(2)
                e.schedule({"kind": "motor_disconnect", "duration_controls": 2})
                e.step(1)
                np.testing.assert_array_equal(e.body.d.ctrl, 0)
            same_state(engine.body.snapshot(), restored.body.snapshot())
            same_state(
                engine.neuromuscular.snapshot(), restored.neuromuscular.snapshot()
            )
            same_state(engine.neural.snapshot(), restored.neural.snapshot())
        finally:
            engine.close()
            if restored:
                restored.close()

    def test_manual_ownership_and_return_to_neural_are_recordable_and_atomic(self):
        graph = tendon_graph()
        engine = CEngine(
            graph, tendon_bindings(graph), mode="C_STRICT", body_factory=TendonBody
        )
        try:
            tick = engine.tick
            engine.command("body_actuation", {"tendon_inputs": {"lf_tarsus": 0.07}})
            self.assertEqual(engine.tick, tick)
            engine.step(2)
            self.assertEqual(engine.body.d.ctrl[2], 0.07)
            self.assertEqual(
                engine.neuromuscular.tendon_adapter.modes["lf_tarsus"], "manual"
            )
            before = engine.checkpoint()
            for payload in (
                {
                    "tendon_inputs": {"lf_tarsus": 0.02},
                    "tendon_modes": {"bad": "manual"},
                },
                {
                    "tendon_inputs": {"lf_tarsus": 0.02},
                    "tendon_modes": {"lf_tarsus": "neural"},
                },
                {
                    "tendon_inputs": {"lf_tarsus": 100},
                    "tendon_modes": {"rf_tarsus": "manual"},
                },
            ):
                with self.assertRaises(ValueError):
                    engine.command("body_actuation", payload)
                same_state(before, engine.checkpoint())
            engine.command("body_actuation", {"tendon_modes": {"lf_tarsus": "neural"}})
            self.assertEqual(engine.body.d.ctrl[2], 0)
            self.assertEqual(
                engine.neuromuscular.tendon_adapter.modes["lf_tarsus"], "neural"
            )
        finally:
            engine.close()


@unittest.skipUnless(
    os.environ.get("FLYLAB_NATIVE_TESTS") == "1", "Native MuJoCo tendon response"
)
class NativeNeuralTendons(unittest.TestCase):
    def test_neural_tendon_commands_apply_to_actual_distal_hinges(self):
        from flylab.body_options import BodyOptions

        graph = tendon_graph()
        body = FlyGymBody(
            42,
            default_world(),
            config_values(),
            body_options=BodyOptions(
                model="flybody",
                actuation="whole_body",
                servo_profile="tracking_all",
                attachment="tethered",
                tendons="all",
            ),
        )
        try:
            loop = NeuromuscularLoop(graph, build_spec(graph, 2), body)
            neural = ExpLIF(graph)
            group = next(
                g
                for g in loop.spec["rows"][0]["muscles"]
                if g["muscle"] == "long_tendon_muscle"
            )
            neural.rate[graph.resolve(group["ids"])] = 100.0
            tendon = body.tendon_control.index["lf_tarsus"]
            addresses = body.m.jnt_qposadr[body.tendon_control.joints[tendon]]
            start = body.d.qpos[addresses].copy()
            for _ in range(20):
                target, adhesion = loop.decode(neural)
                body.step_joint_targets(
                    target, adhesion, tendon_inputs=loop.tendon_inputs
                )
            self.assertGreater(body.d.ctrl[body.tendon_control.act_ids[tendon]], 0)
            self.assertTrue(np.all(body.d.qpos[addresses] - start > 0.01))
            self.assertEqual(
                body.tendon_control.observation()["neural_mapping"],
                "connected_experimental",
            )
            self.assertFalse(body.fault)
        finally:
            body.close()
