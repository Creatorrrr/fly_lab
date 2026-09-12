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
        self,
        graph,
        body,
        *,
        sensory_gain=18.0,
        motor_gain=0.002,
        pooling="mean",
        coxa_geometry=False,
        ltm_tendons=False,
        coxa_endpoint=False,
    ):
        if pooling not in ("mean", "unit_response"):
            raise ValueError("Unknown rate motor pooling")
        self.pooling = pooling
        if type(coxa_endpoint) is not bool or (coxa_endpoint and not coxa_geometry):
            raise ValueError("Endpoint direction model requires physical coxa mapping")
        self.coxa_endpoint = coxa_endpoint
        if type(ltm_tendons) is not bool:
            raise ValueError("Boolean neural LTM tendon option required")
        self.ltm_tendons = ltm_tendons
        self.tendon_activation = np.zeros(6)
        if type(coxa_geometry) is not bool:
            raise ValueError("Boolean physical coxa direction option required")
        self.schema = "flylab.rate-body.v3" if coxa_geometry else "flylab.rate-body.v2"
        self.coxa_sign = np.ones(6)
        if coxa_geometry:
            slope = np.asarray(body.coxa_rotation_kinematics()["d_anterior_dq"])
            if (
                slope.shape != (6,)
                or not np.isfinite(slope).all()
                or np.any(np.abs(slope) < 1e-4)
            ):
                raise ValueError("Unresolved physical coxa anterior rotation direction")
            self.coxa_sign = np.sign(slope)
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
        endpoint_vectors = []
        if coxa_endpoint:
            geometry = body.coxa_endpoint_kinematics()
            anterior, inward = [np.asarray(geometry[k]) for k in ("anterior", "inward")]
            if any(
                x.shape != (6, 3) or not np.isfinite(x).all()
                for x in (anterior, inward)
            ):
                raise ValueError("Finite physical coxa endpoint Jacobians required")
            directions = {
                "tergopleural_promotor_muscle": (1.0, 0.0),
                "pleural_remotor_and_abductor_muscle": (-1.0, -1.0),
                "sternal_adductor_muscle": (0.0, 1.0),
            }
            for leg, groups in enumerate(self.muscles):
                jacobian = np.stack([anterior[leg], inward[leg]])
                if np.linalg.matrix_rank(jacobian, tol=1e-5) != 2:
                    raise ValueError("Unresolved coxa movement plane")
                for name, _, moment in groups:
                    moment[0] *= self.coxa_sign[leg]
                    if name not in directions:
                        continue
                    displacement = np.linalg.lstsq(
                        jacobian, directions[name], rcond=1e-5
                    )[0]
                    displacement /= np.max(np.abs(displacement))
                    moment[:3] = displacement
                    endpoint_vectors.append(
                        {"leg": LEGS[leg], "muscle": name, "vector": moment.tolist()}
                    )
        self.motor_columns = {int(i): k for k, i in enumerate(self.motor_ids)}
        if ltm_tendons:
            control = getattr(body, "tendon_control", None)
            if control is None:
                raise ValueError("LTM motor mapping requires physical tarsal tendons")
            for leg, groups in zip(LEGS, self.muscles):
                name = leg + "_tarsus"
                if name not in control.index:
                    raise ValueError("Missing physical LTM tendon: " + name)
                lo, hi = control.limits[control.index[name]]
                if lo > 0.0 or hi < 0.1:
                    raise ValueError("Physical LTM envelope mismatch")
                matches = [
                    moment
                    for muscle, _, moment in groups
                    if muscle == "long_tendon_muscle"
                ]
                if len(matches) != 1:
                    raise ValueError("Unique anatomical LTM motor group required")
                # The distal tendon replaces the first-tarsus position proxy.
                # Tarsus depressor/levator and LTM adhesion requests remain.
                matches[0].fill(0.0)
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
        if coxa_geometry:
            self.identity = digest(
                {
                    "schema": self.schema,
                    "rate_body_v2_hash": self.identity,
                    "coxa_direction": "distal-coxa-forward-jacobian-v1",
                    "coxa_motor_sign": self.coxa_sign.tolist(),
                    "source": "https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf",
                }
            )
        if ltm_tendons:
            self.schema = "flylab.rate-body.v4"
            self.identity = digest(
                {
                    "schema": self.schema,
                    "base_adapter_hash": self.identity,
                    "ltm": "six-distal-tendons-no-first-joint-double-count",
                    "maximum_tendon_input": 0.1,
                    "rate_scale_model_units": 100.0,
                    "activation_tau_s": 0.03,
                    "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11348827/",
                }
            )
        if coxa_endpoint:
            self.schema = "flylab.rate-body.v5"
            self.identity = digest(
                {
                    "schema": self.schema,
                    "base_adapter_hash": self.identity,
                    "model": "minimum-norm-coxa-endpoint-direction-proxy-v1",
                    "vectors": endpoint_vectors,
                    "anatomical_scope": "Foreleg action labels generalized as an engineering hypothesis; not measured six-leg moments",
                }
            )

    @property
    def tendon_inputs(self):
        return (
            {
                leg + "_tarsus": float(value)
                for leg, value in zip(LEGS, self.tendon_activation)
            }
            if self.ltm_tendons
            else {}
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
            if not self.coxa_endpoint:
                vector[0] *= self.coxa_sign[leg]
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
        if self.ltm_tendons:
            requested_tendons = 0.1 * np.tanh(
                np.array(
                    [
                        row["muscle_rate_units"]["long_tendon_muscle"]
                        for row in diagnostic
                    ]
                )
                / 100.0
            )
            if motor_cut:
                self.tendon_activation.fill(0.0)
            else:
                self.tendon_activation += (1 - math.exp(-dt / 0.03)) * (
                    requested_tendons - self.tendon_activation
                )
        self.last_motor = diagnostic
        return targets, adhesion

    def snapshot(self):
        state = {
            "schema": self.schema,
            "hash": self.identity,
            "offset": self.offset.copy(),
            "filtered": self.filtered.copy(),
            "velocity_lowpass": self.velocity_lowpass.copy(),
            "last_features": self.last_features.copy(),
            "last_motor": copy.deepcopy(self.last_motor),
        }
        if self.ltm_tendons:
            state["tendon_activation"] = self.tendon_activation.copy()
        return state

    def restore(self, state):
        if state.get("schema") != self.schema or state.get("hash") != self.identity:
            raise ValueError("Incompatible rate/body state")
        values = {}
        fields = (
            ("offset", -0.5, 0.5),
            ("filtered", 0, self.sensory_gain),
            ("velocity_lowpass", -1e10, 1e10),
            ("last_features", 0.0, 1.0),
        )
        if self.ltm_tendons:
            fields += (("tendon_activation", 0.0, 0.1),)
        for key, lo, hi in fields:
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
