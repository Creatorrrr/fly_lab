"""Experimental full-BANC rate/body interface, with no gait clock.

Anatomical neuron memberships come from BANC v888. Receptor tuning, signed
muscle moment vectors and the position-servo decoder are engineering models.
Rate-model input/output units are deliberately not labeled mV or measured Hz.
"""

import copy
import math

import numpy as np

from .integrity import digest
from .neuromuscular import (
    DOFS,
    LEGS,
    build_spec,
    joint_suffixes,
    typed_receptor_feature,
)


class RateBodyAdapter:
    def __init__(
        self, graph, body, *, sensory_gain=18.0, motor_gain=0.002, pooling="mean"
    ):
        if pooling not in ("mean", "unit_response"):
            raise ValueError("Unknown rate motor pooling")
        self.pooling = pooling
        if not np.isfinite([sensory_gain, motor_gain]).all() or not (
            0 <= sensory_gain <= 400 and 0 <= motor_gain <= 0.02
        ):
            raise ValueError("Invalid rate/body interface gains")
        self.graph, self.body = graph, body
        self.sensory_gain, self.motor_gain = float(sensory_gain), float(motor_gain)
        self.spec = build_spec(graph, 2)
        self.joints, self.muscles, self.ports = [], [], []
        for leg_index, row in enumerate(self.spec["rows"]):
            joints = []
            for suffix in joint_suffixes(row["leg"]):
                matches = [
                    i
                    for i, name in enumerate(body.joint_names)
                    if name.endswith(suffix)
                ]
                if len(matches) != 1:
                    raise ValueError("Unique physical joint required: " + suffix)
                joints.append(matches[0])
            self.joints.append(joints)
            self.muscles.append(
                [
                    (
                        group["muscle"],
                        graph.resolve(group["ids"]),
                        np.array([group["vector"].get(dof, 0.0) for dof in DOFS]),
                    )
                    for group in row["muscles"]
                ]
            )
            for port in row["sensory"]:
                self.ports.append(
                    (
                        leg_index,
                        port["kind"],
                        port["cell_type"],
                        graph.resolve(port["ids"], maximum=10000),
                    )
                )
        self.joints = np.asarray(self.joints, dtype=int)
        self.motor_ids = np.array(
            sorted(
                {int(i) for groups in self.muscles for _, ids, _ in groups for i in ids}
            )
        )
        if len(np.unique(self.joints)) != 42:
            raise ValueError("Six complete and distinct leg joint mappings required")
        self.motor_columns = {int(i): k for k, i in enumerate(self.motor_ids)}
        self.offset = np.zeros(42)
        self.filtered = np.zeros(len(self.ports))
        self.velocity_lowpass = np.zeros(42)
        self.last_motor = []
        self.last_features = np.zeros(len(self.ports))
        calibration = body.knee_kinematics()
        slope = np.asarray(calibration["d_flexion_dq"])
        if (
            slope.shape != (6,)
            or not np.isfinite(slope).all()
            or np.any(np.abs(slope) < 0.5)
        ):
            raise ValueError("Unresolved physical knee flexion direction")
        self.knee_sign = np.sign(slope)
        self.identity = digest(
            {
                "graph": graph.hash,
                "body": body.model_hash,
                "sensory_gain": self.sensory_gain,
                "motor_gain": self.motor_gain,
                "pooling": self.pooling,
                "schema": "flylab.rate-body.v2",
                "knee_geometry": "physical-segment-origins-and-jacobians-v1",
                "knee_motor_sign": self.knee_sign.tolist(),
                "anatomy": self.spec["rows"],
                "maximum_offset_rad": 0.5,
                "activation_tau_s": 0.03,
                "receptor_tau_s": 0.02,
                "receptor_latency_controls": 1,
                "adhesion": "tarsal-balance-and-lift-veto",
            }
        )

    def encode(self, *, sensory_cut=False, dt=0.005):
        if not np.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError("Invalid receptor timestep")
        observation = self.body.leg_observation()
        q, velocity, support, touch = [
            np.asarray(observation[key], float)
            for key in (
                "angles_rad",
                "velocities_rad_s",
                "support_load_bw",
                "non_support_load_bw",
            )
        ]
        if (
            q.shape != (42,)
            or velocity.shape != (42,)
            or support.shape != (6,)
            or touch.shape != (6,)
            or not all(np.isfinite(x).all() for x in (q, velocity, support, touch))
        ):
            raise ValueError("Finite physical receptor observations required")
        knee = self.body.knee_kinematics()
        flexion, flexion_velocity = [
            np.asarray(knee[key]) for key in ("flexion_rad", "velocity_rad_s")
        ]
        if (
            flexion.shape != (6,)
            or flexion_velocity.shape != (6,)
            or not np.isfinite(flexion).all()
            or not np.isfinite(flexion_velocity).all()
        ):
            raise ValueError("Invalid measured knee geometry")
        q, velocity = q.copy(), velocity.copy()
        q[self.joints[:, 5]], velocity[self.joints[:, 5]] = flexion, flexion_velocity
        features = np.array(
            [
                typed_receptor_feature(
                    kind,
                    cell_type,
                    q[self.joints[leg, 5]],
                    velocity[self.joints[leg, 5]],
                    self.velocity_lowpass[self.joints[leg, 5]],
                    support[leg],
                    touch[leg],
                    self.spec,
                )
                for leg, kind, cell_type, _ in self.ports
            ]
        )
        drive = np.zeros(self.graph.n, np.float32)
        if not sensory_cut:
            for value, (_, _, _, ids) in zip(self.filtered, self.ports):
                np.add.at(drive, ids, value)
        self.filtered += (1 - math.exp(-dt / 0.02)) * (
            self.sensory_gain * features - self.filtered
        )
        self.velocity_lowpass += (1 - math.exp(-dt / 0.03)) * (
            velocity - self.velocity_lowpass
        )
        self.last_features = features
        return drive

    def decode(self, rates, *, motor_cut=False, dt=0.005):
        rates = np.asarray(rates)
        if (
            rates.shape != self.motor_ids.shape
            or rates.dtype.kind not in "fiu"
            or not np.isfinite(rates).all()
            or np.any(rates < 0)
        ):
            raise ValueError("Finite nonnegative BANC motor rate units required")
        if not np.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError("Invalid muscle timestep")
        requested, adhesion, diagnostic = np.zeros(42), np.zeros(6, bool), []
        for leg, groups in enumerate(self.muscles):
            vector, output = np.zeros(7), {}
            for name, ids, moment in groups:
                units = rates[[self.motor_columns[int(i)] for i in ids]]
                rate = float(
                    np.mean(units)
                    if self.pooling == "mean"
                    else np.sum(100.0 * units / (100.0 + units))
                )
                vector += rate * moment * self.motor_gain
                output[name] = rate
            vector[5] *= self.knee_sign[leg]
            requested[self.joints[leg]] = 0.5 * np.tanh(vector / 0.5)
            adhesion[leg] = (
                output.get("tarsus_depressor_muscle", 0)
                + output.get("long_tendon_muscle", 0)
                > output.get("tarsus_levator_muscle", 0) + 1.0
            )
            diagnostic.append({"leg": LEGS[leg], "muscle_rate_units": output})
        if motor_cut:
            # Enforce disconnection at the final actuator boundary, after pooling.
            self.offset.fill(0)
            adhesion.fill(False)
        else:
            self.offset += (1 - math.exp(-dt / 0.03)) * (requested - self.offset)
        targets = self.body.neutral + self.offset
        lift = np.asarray(self.body.foot_kinematics(targets)["target_lift_mm"])
        if lift.shape != (6,) or not np.isfinite(lift).all():
            raise ValueError("Invalid local pad lift observation")
        adhesion[lift > 0.02] = False
        self.last_motor = diagnostic
        return targets, adhesion

    def snapshot(self):
        return {
            "schema": "flylab.rate-body.v2",
            "hash": self.identity,
            "offset": self.offset.copy(),
            "filtered": self.filtered.copy(),
            "velocity_lowpass": self.velocity_lowpass.copy(),
            "last_features": self.last_features.copy(),
            "last_motor": copy.deepcopy(self.last_motor),
        }

    def restore(self, state):
        if (
            state.get("schema") != "flylab.rate-body.v2"
            or state.get("hash") != self.identity
        ):
            raise ValueError("Incompatible rate/body state")
        values = {}
        for key, lo, hi in (
            ("offset", -0.5, 0.5),
            ("filtered", 0, self.sensory_gain),
            ("velocity_lowpass", -1e10, 1e10),
            ("last_features", 0.0, 1.0),
        ):
            value = np.asarray(state.get(key))
            if (
                value.dtype.kind not in "fiu"
                or value.shape != getattr(self, key).shape
                or not np.isfinite(value).all()
                or np.any((value < lo) | (value > hi))
            ):
                raise ValueError("Invalid rate/body state: " + key)
            values[key] = value
        diagnostics = copy.deepcopy(state.get("last_motor"))
        if not isinstance(diagnostics, list) or len(diagnostics) not in (0, 6):
            raise ValueError("Invalid rate/body motor diagnostics")
        for leg, row in enumerate(diagnostics):
            if (
                not isinstance(row, dict)
                or set(row) != {"leg", "muscle_rate_units"}
                or row["leg"] != LEGS[leg]
            ):
                raise ValueError("Invalid rate/body motor leg")
            muscles = row["muscle_rate_units"]
            if not isinstance(muscles, dict) or set(muscles) != {
                group[0] for group in self.muscles[leg]
            }:
                raise ValueError("Invalid rate/body motor muscles")
            raw = np.asarray(list(muscles.values()))
            if (
                raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all()
                or np.any(raw < 0)
            ):
                raise ValueError("Invalid rate/body motor values")
        for key, value in values.items():
            getattr(self, key)[:] = value
        self.last_motor = diagnostics
