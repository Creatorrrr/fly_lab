"""Explicit experimental BANC walking session, independent of C/LIF and CPG modes.

The full source graph supplies recurrence. Local leg receptors are the default
external input; optional DNg100 drive supports explicit circuit experiments. Activity
adaptation is a continuous cellular model hypothesis. No gait oscillator,
periodic reset, phase drive, or automatic root-pose correction is present.
"""

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..body import FlyGymBody
from ..body_options import BodyOptions
from ..common import to_ui
from ..engine import config_values
from ..sensors import default_world, validate_world
from .adaptive_rate import AdaptiveRateNetwork
from .integrity import bounded_int, digest
from .rate_body import RateBodyAdapter
from .research_annotations import rate_weights


def validate_banc_roster(graph):
    """Admit the pinned research roster without relabeling it a full snapshot."""
    m = graph.manifest
    if (m.get("dataset_id"), m.get("snapshot_id")) != ("flywire_banc", "888"):
        raise ValueError("BANC v888 source graph required")
    full_snapshot = graph.full_brain
    declared = (
        m.get("scope") == "declared_neuronal_roster"
        and graph.hash
        == "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
        and m.get("source_node_count") == graph.n == 158706
        and len(graph.indices) == 11584852
        and m.get("excluded_node_count") == 0
        and m.get("excluded_node_ids") == []
    )
    if not (full_snapshot or declared):
        raise ValueError(
            "Pinned BANC neuronal roster or verified full snapshot required"
        )
    return {
        "scope": m["scope"],
        "full_snapshot_verified": bool(full_snapshot),
        "all_declared_neurons_simulated": True,
        "limitations": copy.deepcopy(m.get("scope_limitations", [])),
    }


@dataclass(frozen=True)
class BancWalkingParameters:
    adaptation_gain: float = 4.0
    adaptation_tau_s: float = 0.15
    sensory_gain: float = 18.0
    motor_gain: float = 0.002
    descending_drive: float = 0.0
    pooling: str = "mean"

    def __post_init__(self):
        if self.pooling not in ("mean", "unit_response"):
            raise ValueError("Unknown BANC motor pooling")
        for name, lo, hi in (
            ("adaptation_gain", 0.0, 100.0),
            ("adaptation_tau_s", 0.01, 10.0),
            ("sensory_gain", 0.0, 400.0),
            ("motor_gain", 0.0, 0.02),
            ("descending_drive", 0.0, 400.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lo <= value <= hi
            ):
                raise ValueError("Invalid BANC rate parameter: " + name)


class BancWalkingSession:
    schema = "flylab.banc-walking.v1"

    def __init__(self, graph, *, seed=42, parameters=None, world=None, device="auto"):
        self.source_scope = validate_banc_roster(graph)
        self.graph = graph
        self.seed = bounded_int(seed, "seed", 0, 2**32 - 1)
        self.p = BancWalkingParameters(**(parameters or {}))
        if world is None:
            world = default_world()
            world["sources"], world["obstacles"] = [], []
        self.world = copy.deepcopy(validate_world(world))
        self.options = BodyOptions(
            model="flybody",
            actuation="whole_body",
            servo_profile="tracking_all",
            tendons="all",
        )
        self.body = FlyGymBody(
            seed, self.world, config_values(None), body_options=self.options
        )
        try:
            self.adapter = RateBodyAdapter(
                graph,
                self.body,
                sensory_gain=self.p.sensory_gain,
                motor_gain=self.p.motor_gain,
                pooling=self.p.pooling,
            )
            weights, _ = rate_weights(graph)
            self.network = AdaptiveRateNetwork(
                weights,
                *[
                    np.full(graph.n, value, np.float32)
                    for value in (0.02, 1.0, 7.5, 200.0)
                ],
                adaptation_gain=self.p.adaptation_gain,
                adaptation_tau_s=self.p.adaptation_tau_s,
                device=device,
            )
            self.descending = np.array(
                [
                    i
                    for i, node in enumerate(graph.nodes)
                    if node["cell_type"] == "DNg100"
                ]
            )
            if len(self.descending) != 2:
                raise ValueError("Bilateral BANC v888 DNg100 required")
        except Exception:
            self.body.close()
            raise
        self.tendon_inputs = {name: 0.2 for name in self.body.tendon_control.names}
        self.cpg_hash = digest(self.body.snapshot()["cpg"])
        self.cuts = {
            "sensory": False,
            "descending": False,
            "circuit": False,
            "motor": False,
        }
        self.control_tick = 0
        self.fault = None
        self.closed = False
        self.wall_s = 0.0
        self.initial_position = self.body.pose()[0]
        self.signed_forward_mm = 0.0
        self.rate = np.zeros(len(self.adapter.motor_ids), np.float32)

    def set_cuts(self, cuts):
        if (
            not isinstance(cuts, dict)
            or set(cuts) - set(self.cuts)
            or any(type(value) is not bool for value in cuts.values())
        ):
            raise ValueError(
                "Boolean sensory, descending, circuit, or motor cuts required"
            )
        self.cuts.update(cuts)
        self.network.set_muted(
            np.arange(self.graph.n)
            if self.cuts["circuit"]
            else self.descending
            if self.cuts["descending"]
            else ()
        )

    def advance(self, controls=20):
        bounded_int(controls, "BANC controls", 1, 20)
        if self.closed or self.fault or self.body.fault:
            raise RuntimeError(self.fault or self.body.fault or "Closed BANC session")
        began = time.perf_counter()
        try:
            for _ in range(controls):
                prior, rotation, _ = self.body.pose()
                drive = self.adapter.encode(sensory_cut=self.cuts["sensory"])
                drive[self.descending] += self.p.descending_drive
                self.network.advance(drive, 50)
                self.rate = self.network.readout(self.adapter.motor_ids)
                target, adhesion = self.adapter.decode(
                    self.rate, motor_cut=self.cuts["motor"]
                )
                self.body.step_joint_targets(
                    target, adhesion, tendon_inputs=self.tendon_inputs
                )
                current = self.body.pose()[0]
                self.signed_forward_mm += float((current - prior)[:2] @ rotation[:2, 0])
                self.control_tick += 1
                self.fault = self.body.fault
                if self.fault:
                    break
            self._clocks()
        except Exception as exc:
            self.fault = str(exc)
            raise
        finally:
            self.wall_s += time.perf_counter() - began
        return self.frame()

    def _clocks(self):
        physical_time = self.body.physics_time()
        if (
            self.network.tick != self.control_tick * 50
            or not math.isfinite(physical_time)
            or abs(physical_time - self.control_tick * 0.005) > 1e-5
        ):
            raise RuntimeError("BANC neural/body clocks diverged")
        if digest(self.body.snapshot()["cpg"]) != self.cpg_hash:
            raise RuntimeError("BANC walking must not advance the external CPG")

    def frame(self):
        position, rotation, _ = self.body.pose()
        offset = position - self.initial_position
        return {
            "schema": self.schema,
            "time_s": self.control_tick * 0.005,
            "control_tick": self.control_tick,
            "neural_tick": self.network.tick,
            "position_mm": to_ui(position),
            "horizontal_net_mm": float(np.linalg.norm(offset[:2])),
            "signed_forward_mm": self.signed_forward_mm,
            "upright": float(rotation[2, 2]),
            "fault": self.fault or self.body.fault,
            "parameters": asdict(self.p),
            "cuts": self.cuts.copy(),
            "wall_s": self.wall_s,
            "graph_hash": self.graph.hash,
            "source_scope": copy.deepcopy(self.source_scope),
            "neurons": self.graph.n,
            "edges": len(self.graph.indices),
            "model_hash": self.network.identity,
            "body_hash": self.body.model_hash,
            "motor_rate_unit": "model rate units",
            "motor_rate_mean": float(self.rate.mean()),
            "muscles": copy.deepcopy(self.adapter.last_motor),
            "tendon_inputs": self.tendon_inputs.copy(),
            "feet": self.body.contact_probe(),
            "sensory_port_count": len(self.adapter.ports),
            "feedback_feature_mean": float(self.adapter.last_features.mean()),
            "input_model": "Measured leg receptor feedback only"
            if self.p.descending_drive == 0
            else "Constant bilateral DNg100 drive and measured leg receptor feedback",
            "descending_drive": self.p.descending_drive,
            "device": self.network.device,
            "walking_status": "EXPERIMENTAL_RUN_NOT_ASSESSED",
            "biological_validation": False,
            "cpg_advanced": digest(self.body.snapshot()["cpg"]) != self.cpg_hash,
            "imposed_gait": False,
            "root_pose_correction": False,
            "motor_model": "Anatomical muscle pooling and bounded position servos; not calibrated muscle forces",
        }

    def snapshot(self):
        if self.fault or self.body.fault or self.closed:
            raise ValueError("Only a healthy BANC session can be checkpointed")
        self._clocks()
        return {
            "schema": self.schema,
            "graph_hash": self.graph.hash,
            "parameters": asdict(self.p),
            "seed": self.seed,
            "world": copy.deepcopy(self.world),
            "body": self.body.snapshot(),
            "network": self.network.snapshot(),
            "adapter": self.adapter.snapshot(),
            "cuts": self.cuts.copy(),
            "control_tick": self.control_tick,
            "initial_position": self.initial_position.copy(),
            "signed_forward_mm": self.signed_forward_mm,
            "cpg_hash": self.cpg_hash,
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get("schema") != cls.schema or saved.get("graph_hash") != graph.hash:
            raise ValueError("BANC rate checkpoint graph/schema mismatch")
        if not isinstance(saved.get("cuts"), dict) or set(saved["cuts"]) != {
            "sensory",
            "descending",
            "circuit",
            "motor",
        }:
            raise ValueError("Complete BANC causal cut state required")
        controls = bounded_int(
            saved.get("control_tick"), "BANC checkpoint controls", 0, 10**10
        )
        position = np.asarray(saved.get("initial_position"))
        if (
            position.shape != (3,)
            or position.dtype.kind not in "fiu"
            or not np.isfinite(position).all()
        ):
            raise ValueError("Invalid BANC checkpoint initial position")
        session = cls(
            graph,
            seed=saved["seed"],
            parameters=saved["parameters"],
            world=saved["world"],
            device=saved["network"]["device"],
        )
        try:
            session.body.restore(saved["body"])
            session.network.restore(saved["network"])
            session.adapter.restore(saved["adapter"])
            # Validate the saved causal mask before setting any new one.
            expected = np.ones(graph.n, np.float32)
            if saved["cuts"].get("circuit") is True:
                expected.fill(0)
            elif saved["cuts"].get("descending") is True:
                expected[session.descending] = 0
            if not np.array_equal(expected, saved["network"]["output_mask"]):
                raise ValueError("BANC checkpoint cut and neural mask disagree")
            session.set_cuts(saved["cuts"])
            session.control_tick = controls
            session.initial_position = position.copy()
            session.signed_forward_mm = float(saved["signed_forward_mm"])
            session.wall_s = float(saved["wall_s"])
            if (
                not math.isfinite(session.signed_forward_mm)
                or not math.isfinite(session.wall_s)
                or session.wall_s < 0
            ):
                raise ValueError("Invalid BANC checkpoint metrics")
            if session.cpg_hash != saved["cpg_hash"]:
                raise ValueError("BANC checkpoint changed CPG state")
            session.rate = session.network.readout(session.adapter.motor_ids)
            session._clocks()
        except Exception:
            session.close()
            raise
        return session

    def close(self):
        self.closed = True
        self.body.close()
        self.network = None
