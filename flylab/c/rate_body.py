"""Experimental full-BANC rate/body interface, with no gait clock.

Anatomical neuron memberships come from BANC v888. Receptor tuning, signed
muscle moment vectors and the position-servo decoder are engineering models.
Rate-model input/output units are deliberately not labeled mV or measured Hz.
"""

import copy
import math

import numpy as np

from .integrity import digest
from .target_activation import RELEASE_TAU_S, advance_target_activation
from .coxa_hair_plate import INPUT_GAIN, SOURCE as HAIR_PLATE_SOURCE, anterior_feature, front_coxa_ports
from .neuromuscular import (
    DOFS,
    LEGS,
    build_spec,
    joint_suffixes,
    typed_receptor_feature,
)


class OrderedMoments:
    """Prepared muscle terms with the original float64 addition order."""

    def __init__(self, muscles):
        self.counts = np.array([len(groups) for groups in muscles])
        self.legs = np.arange(len(muscles))
        self.moments = np.asarray(
            [moment for groups in muscles for _, _, moment in groups], dtype=np.float64
        )
        self.row_leg = np.repeat(self.legs, self.counts)
        self.row_col = np.concatenate([np.arange(count) + 1 for count in self.counts])
        self.weighted = np.empty_like(self.moments)
        # Leading +0 preserves the scalar loop's signed-zero semantics. Padding
        # after a leg's last muscle is never selected as that leg's result.
        self.terms = np.zeros((len(muscles), int(self.counts.max()) + 1, 7))
        self.accumulated = np.empty_like(self.terms)

    def __call__(self, pooled, gain):
        rates = np.asarray(pooled, dtype=np.float64)
        np.multiply(rates[:, None], self.moments, out=self.weighted)
        np.multiply(self.weighted, gain, out=self.weighted)
        self.terms[self.row_leg, self.row_col] = self.weighted
        np.add.accumulate(self.terms, axis=1, dtype=np.float64, out=self.accumulated)
        return self.accumulated[self.legs, self.counts]


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
        adhesion_model="legacy",
        joint_activation_tau_s=0.03,
        feasible_activation=False,
        joint_gain_multipliers=(1., 1., 1., 1., 1., 1., 1.),
        load_gain_multiplier=1.0,
        active_length=False,
        muscle_activation_first=False,
        fast_coxa_activation=False,
        front_hair_plate=False,
        fast_joint_release=False,
        joint_release_mode="all",
    ):
        if joint_release_mode not in ("all", "reversal_only") or (
            joint_release_mode != "all" and not fast_joint_release
        ):
            raise ValueError("Explicit fast release required for the selected joint release mode")
        self.joint_release_mode = joint_release_mode
        if type(fast_joint_release) is not bool or (fast_joint_release and (
            type(joint_activation_tau_s) not in (int, float)
            or joint_activation_tau_s <= RELEASE_TAU_S or muscle_activation_first
        )):
            raise ValueError("Fast joint target release requires slower post-saturation activation")
        self.fast_joint_release = fast_joint_release
        if type(front_hair_plate) is not bool or (front_hair_plate and not coxa_geometry):
            raise ValueError("Front hair plate input requires physical coxa direction")
        self.front_hair_plate = front_hair_plate
        if type(fast_coxa_activation) is not bool or (fast_coxa_activation and (
            type(joint_activation_tau_s) not in (int, float) or joint_activation_tau_s <= .03 or muscle_activation_first
        )):
            raise ValueError("Fast coxa activation requires slower joint activation after saturation")
        self.fast_coxa_activation = fast_coxa_activation
        if type(muscle_activation_first) is not bool:
            raise ValueError("Boolean muscle activation order required")
        self.muscle_activation_first = muscle_activation_first
        if type(active_length) is not bool:
            raise ValueError("Boolean active length required")
        self.active_length = active_length
        self.total_activation = np.zeros(42)
        if type(load_gain_multiplier) not in (int, float) or not math.isfinite(load_gain_multiplier) or not 0 <= load_gain_multiplier <= 32:
            raise ValueError("Invalid load receptor gain multiplier")
        self.load_gain_multiplier = float(load_gain_multiplier)
        self.joint_gain_multipliers = np.asarray(joint_gain_multipliers, dtype=float)
        if self.joint_gain_multipliers.shape != (7,) or not np.isfinite(self.joint_gain_multipliers).all() or np.any((self.joint_gain_multipliers < .25) | (self.joint_gain_multipliers > 4)):
            raise ValueError("Seven bounded joint gain multipliers required")
        self._joint_gain_changed = not np.all(self.joint_gain_multipliers == 1.)
        if type(feasible_activation) is not bool:
            raise ValueError("Boolean feasible activation required")
        self.feasible_activation = feasible_activation
        if isinstance(joint_activation_tau_s, bool) or not isinstance(joint_activation_tau_s, (int, float)) or not math.isfinite(joint_activation_tau_s) or not .005 <= joint_activation_tau_s <= .25:
            raise ValueError("Invalid joint activation time constant")
        self.joint_activation_tau_s = joint_activation_tau_s
        if adhesion_model not in ("legacy", "feasible_lift", "activation", "graded", "request_lift"):
            raise ValueError("Unknown neural adhesion model")
        if adhesion_model == "request_lift" and (muscle_activation_first or active_length):
            raise ValueError("Request lift requires the post-saturation joint filter")
        self.adhesion_model = adhesion_model
        self.adhesion_activation = np.zeros(6)
        self.adhesion_control = np.zeros(6)
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
        if self.front_hair_plate:
            for leg, ports in front_coxa_ports(graph).items():
                self.spec["rows"][leg]["sensory"].extend(ports)
                self.spec["sensory_neurons"] += sum(len(port["ids"]) for port in ports)
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
        self._muscle_columns = [
            [
                np.asarray([self.motor_columns[int(i)] for i in ids], dtype=int)
                for _, ids, _ in groups
            ]
            for groups in self.muscles
        ]
        pool_groups = {}
        for index, columns in enumerate(
            c for groups in self._muscle_columns for c in groups
        ):
            pool_groups.setdefault(len(columns), []).append((index, columns))
        self._pool_groups = [
            (np.asarray([i for i, _ in group]), np.stack([c for _, c in group]))
            for group in pool_groups.values()
        ]
        self._pool_count = sum(len(groups) for groups in self.muscles)
        # Reuse only receptor computations with identical arguments/behavior.
        # Membership and the order of repeated additions remain unchanged.
        feature_keys = {}
        self._receptor_keys, port_features = [], []
        for leg, kind, cell_type, _ in self.ports:
            key = (leg, kind, cell_type if kind in ("claw", "hook") else "")
            if key not in feature_keys:
                feature_keys[key] = len(self._receptor_keys)
                self._receptor_keys.append(key)
            port_features.append(feature_keys[key])
        self._port_features = np.asarray(port_features, dtype=int)
        self._sensory_ids = np.concatenate([ids for _, _, _, ids in self.ports])
        self._sensory_unique = len(np.unique(self._sensory_ids)) == len(self._sensory_ids)
        self._sensory_ports = np.repeat(
            np.arange(len(self.ports)), [len(ids) for _, _, _, ids in self.ports]
        )
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
        # Build after both endpoint and LTM adjustments to the fixed moments.
        self._moment_plan = OrderedMoments(self.muscles)
        self.muscle_activation = np.zeros(self._pool_count)
        if self.active_length:
            self._absolute_moment_plan = OrderedMoments(self.muscles)
            self._absolute_moment_plan.moments = np.abs(self._absolute_moment_plan.moments)
        self.offset = np.zeros(42)
        self.filtered = np.zeros(len(self.ports))
        self._load_ports = np.asarray([kind == 'load' for _, kind, _, _ in self.ports])
        self._hair_plate_ports = np.asarray([kind == 'coxa_limit' for _, kind, _, _ in self.ports])
        self._receptor_input_gains = np.full(len(self.ports), self.sensory_gain)
        self._receptor_input_gains[self._load_ports] *= self.load_gain_multiplier
        self._receptor_input_gains[self._hair_plate_ports] = INPUT_GAIN
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
        if adhesion_model in ("feasible_lift", "request_lift") or feasible_activation:
            self._control_limited = body.m.actuator_ctrllimited[body.act_ids].astype(bool)
            self._control_ranges = body.m.actuator_ctrlrange[body.act_ids].copy()
            if self._control_ranges.shape != (42, 2) or not np.isfinite(self._control_ranges).all():
                raise ValueError("Finite actuator ranges required for feasible lift")
        if adhesion_model == "feasible_lift":
            self.schema = "flylab.rate-body.v6"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "adhesion_model": "actuator-feasible-first-order-lift-v1",
                "control_limited": self._control_limited.tolist(),
                "control_ranges": self._control_ranges.tolist(),
                "lift_release_mm": 0.02,
            })
        if adhesion_model == "activation":
            self.schema = "flylab.rate-body.v7"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "adhesion_model": "tarsal-balance-activation-v1",
                "activation_tau_s": .03, "balance_threshold": 1.,
            })
        if joint_activation_tau_s != .03:
            self.schema = "flylab.rate-body.v8"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "joint_activation_tau_s": joint_activation_tau_s,
                "model": "continuous-muscle-position-activation-v1",
            })
        if feasible_activation:
            self.schema = "flylab.rate-body.v9"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "actuator-feasible-muscle-activation-v1",
                "control_limited": self._control_limited.tolist(),
                "control_ranges": self._control_ranges.tolist(),
            })
        if self._joint_gain_changed:
            self.schema = "flylab.rate-body.v10"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "relative-anatomical-joint-moment-v1",
                "dofs": list(DOFS), "multipliers": self.joint_gain_multipliers.tolist(),
            })
        if self.load_gain_multiplier != 1.:
            self.schema = "flylab.rate-body.v11"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "peripheral-load-receptor-gain-v1",
                "load_gain_multiplier": self.load_gain_multiplier,
            })
        if self.active_length:
            self.schema = "flylab.rate-body.v12"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "total-muscle-active-length-v1",
                "activation_tau_s": .03, "rate_to_length_scale": .5,
                "maximum_stiffness_scale": 2.,
            })
        if self.muscle_activation_first:
            self.schema = "flylab.rate-body.v13"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "muscle-activation-before-joint-saturation-v1",
                "activation_tau_s": self.joint_activation_tau_s,
                "adhesion_and_ltm": "original raw pooled rates",
            })
        if self.adhesion_model == "graded":
            self.schema = "flylab.rate-body.v14"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "adhesion": "LP30ms[tanh(max(depressor+LTM-levator,0)/100)] * original_on",
                "rate_units": "model units, not measured Hz",
            })
        if self.fast_coxa_activation:
            self.schema = "flylab.rate-body.v15"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "coxa_rotation_activation_tau_s": .03,
                "other_joint_activation_tau_s": self.joint_activation_tau_s,
                "model": "fast-coxa-rotation-muscle-activation-v1",
                "biological_validation": False,
            })
        if self.front_hair_plate:
            self.schema = "flylab.rate-body.v16"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "front-coxa-anterior-limit-proxy-v1",
                "input_gain": INPUT_GAIN, "anterior_range_rad": .5,
                "activation_tau_s": .02, "delay_controls": 1,
                "source": HAIR_PLATE_SOURCE, "biological_validation": False,
            })
        if self.fast_joint_release:
            self.schema = "flylab.rate-body.v17"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "signed-target-fast-release-v1",
                "release_tau_s": RELEASE_TAU_S,
                "growth_tau_s": self.joint_activation_tau_s,
                "zero_crossing": "exact held-input split integration",
                "fast_coxa_response_preserved": self.fast_coxa_activation,
                "biological_validation": False,
            })
            if self.joint_release_mode == "reversal_only":
                self.schema = "flylab.rate-body.v18"
                self.identity = digest({
                    "schema": self.schema, "base_adapter_hash": self.identity,
                    "model": "signed-target-reversal-release-v1",
                    "same_sign_decrease_and_zero_request": "original growth time constant",
                })
        if self.adhesion_model == "request_lift":
            self.schema = "flylab.rate-body.v19"
            self.identity = digest({
                "schema": self.schema, "base_adapter_hash": self.identity,
                "model": "neural-request-additional-lift-veto-v1",
                "control_limited": self._control_limited.tolist(),
                "control_ranges": self._control_ranges.tolist(),
                "lift_release_mm": .02,
                "existing_filtered_lift_veto": "preserved",
                "biological_validation": False,
            })

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
        knee = self.body.knee_kinematics(include_direction=False)
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
                anterior_feature(q[self.joints[leg, 0]], self.body.neutral[self.joints[leg, 0]], self.coxa_sign[leg])
                if kind == "coxa_limit" else typed_receptor_feature(
                    kind,
                    cell_type,
                    q[self.joints[leg, 5]],
                    velocity[self.joints[leg, 5]],
                    self.velocity_lowpass[self.joints[leg, 5]],
                    support[leg],
                    touch[leg],
                    self.spec,
                )
                for leg, kind, cell_type in self._receptor_keys
            ]
        )[self._port_features]
        drive = np.zeros(self.graph.n, np.float32)
        if not sensory_cut:
            if self._sensory_unique:
                # Preserve +0 addition and float32 storage, including signed zero.
                drive[self._sensory_ids] += self.filtered[self._sensory_ports]
            else:
                np.add.at(drive, self._sensory_ids, self.filtered[self._sensory_ports])
        request = self.sensory_gain * features
        if self.load_gain_multiplier != 1.:
            request[self._load_ports] *= self.load_gain_multiplier
        if self.front_hair_plate:
            request[self._hair_plate_ports] = INPUT_GAIN * features[self._hair_plate_ports]
        self.filtered += (1 - math.exp(-dt / 0.02)) * (request - self.filtered)
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
        pooled = np.empty(self._pool_count)
        for indices, columns in self._pool_groups:
            units = rates[columns]
            pooled[indices] = (
                np.mean(units, axis=1)
                if self.pooling == "mean"
                else np.sum(100.0 * units / (100.0 + units), axis=1)
            )
        joint_rates = pooled
        if self.muscle_activation_first:
            if motor_cut:
                self.muscle_activation.fill(0)
            else:
                self.muscle_activation += (1 - math.exp(-dt / self.joint_activation_tau_s)) * (pooled - self.muscle_activation)
            joint_rates = self.muscle_activation
        vectors = self._moment_plan(joint_rates, self.motor_gain)
        if self.active_length:
            absolute = self._absolute_moment_plan(pooled, self.motor_gain)
            if self._joint_gain_changed:
                absolute *= self.joint_gain_multipliers
            stiffness_request = np.zeros(42)
            for leg, joints in enumerate(self.joints):
                stiffness_request[joints] = np.tanh(absolute[leg] / .5)
        rates_by_muscle = iter(pooled)
        requested, adhesion, diagnostic = np.zeros(42), np.zeros(6, bool), []
        for leg, groups in enumerate(self.muscles):
            vector, output = vectors[leg], {}
            for name, ids, moment in groups:
                rate = float(next(rates_by_muscle))
                output[name] = rate
            if not self.coxa_endpoint:
                vector[0] *= self.coxa_sign[leg]
            vector[5] *= self.knee_sign[leg]
            if self._joint_gain_changed:
                vector *= self.joint_gain_multipliers
            requested[self.joints[leg]] = 0.5 * np.tanh(vector / 0.5)
            adhesion[leg] = (
                output.get("tarsus_depressor_muscle", 0)
                + output.get("long_tendon_muscle", 0)
                > output.get("tarsus_levator_muscle", 0) + 1.0
            )
            diagnostic.append({"leg": LEGS[leg], "muscle_rate_units": output})
        if self.adhesion_model in ("activation", "graded"):
            balance = np.asarray([
                row['muscle_rate_units'].get('tarsus_depressor_muscle', 0)
                + row['muscle_rate_units'].get('long_tendon_muscle', 0)
                - row['muscle_rate_units'].get('tarsus_levator_muscle', 0)
                for row in diagnostic])
            request = np.tanh(np.maximum(balance, 0) / 100) if self.adhesion_model == "graded" else balance
            self.adhesion_activation += (1 - math.exp(-dt / .03)) * (request - self.adhesion_activation)
            if self.adhesion_model == "activation":
                adhesion[:] = self.adhesion_activation > 1.
        if motor_cut:
            # Enforce disconnection at the final actuator boundary, after pooling.
            self.offset.fill(0)
            adhesion.fill(False)
            self.adhesion_activation.fill(0)
            if self.active_length:
                self.total_activation.fill(0)
        else:
            if self.feasible_activation:
                target_request = self.body.neutral + requested
                requested = np.where(
                    self._control_limited,
                    np.clip(target_request, self._control_ranges[:, 0], self._control_ranges[:, 1]),
                    target_request,
                ) - self.body.neutral
            if self.muscle_activation_first:
                self.offset[:] = requested
            else:
                activation_alpha = 1 - math.exp(-dt / self.joint_activation_tau_s)
                if self.fast_coxa_activation:
                    activation_alpha = np.full(42, activation_alpha)
                    activation_alpha[self.joints[:, 0]] = 1 - math.exp(-dt / .03)
                if self.fast_joint_release:
                    next_offset = advance_target_activation(
                        self.offset, requested, dt, self.joint_activation_tau_s,
                        reversals_only=self.joint_release_mode == "reversal_only",
                    )
                    if self.fast_coxa_activation:
                        yaw = self.joints[:, 0]
                        next_offset[yaw] = self.offset[yaw] + activation_alpha[yaw] * (
                            requested[yaw] - self.offset[yaw]
                        )
                    self.offset[:] = next_offset
                else:
                    self.offset += activation_alpha * (requested - self.offset)
            if self.active_length:
                self.total_activation += (1 - math.exp(-dt / .03)) * (stiffness_request - self.total_activation)
        targets = self.body.neutral + (self.offset / (1 + self.total_activation) if self.active_length else self.offset)
        prediction_targets = targets
        if self.adhesion_model == "feasible_lift":
            # Only the shadow displacement changes. Raw motor requests, servo
            # limits and offset dynamics retain their original interpretation.
            prediction_targets = np.where(
                self._control_limited,
                np.clip(targets, self._control_ranges[:, 0], self._control_ranges[:, 1]),
                targets,
            )
        lift = np.asarray(self.body.foot_kinematics(prediction_targets)["target_lift_mm"])
        if lift.shape != (6,) or not np.isfinite(lift).all():
            raise ValueError("Invalid local pad lift observation")
        adhesion[lift > 0.02] = False
        if self.adhesion_model == "request_lift" and not motor_cut:
            # Release may anticipate the joint filter using a request within
            # actuator control bounds. This does not predict actual clearance.
            request_targets = self.body.neutral + requested
            request_targets = np.where(
                self._control_limited,
                np.clip(request_targets, self._control_ranges[:, 0], self._control_ranges[:, 1]),
                request_targets,
            )
            request_lift = np.asarray(self.body.foot_kinematics(request_targets)["target_lift_mm"])
            if request_lift.shape != (6,) or not np.isfinite(request_lift).all():
                raise ValueError("Invalid neural request pad lift observation")
            adhesion[request_lift > .02] = False
        if self.adhesion_model == "graded":
            self.adhesion_control[:] = self.adhesion_activation * adhesion
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
        if self.adhesion_model in ("activation", "graded"):
            state["adhesion_activation"] = self.adhesion_activation.copy()
        if self.adhesion_model == "graded":
            state["adhesion_control"] = self.adhesion_control.copy()
        if self.active_length:
            state["total_activation"] = self.total_activation.copy()
        if self.muscle_activation_first:
            state["muscle_activation"] = self.muscle_activation.copy()
        return state

    def restore(self, state):
        if state.get("schema") != self.schema or state.get("hash") != self.identity:
            raise ValueError("Incompatible rate/body state")
        values = {}
        fields = (
            ("offset", -0.5, 0.5),
            ("filtered", 0, self._receptor_input_gains),
            ("velocity_lowpass", -1e10, 1e10),
            ("last_features", 0.0, 1.0),
        )
        if self.ltm_tendons:
            fields += (("tendon_activation", 0.0, 0.1),)
        if self.adhesion_model == "activation":
            fields += (("adhesion_activation", -1e12, 1e12),)
        if self.adhesion_model == "graded":
            fields += (("adhesion_activation", 0., 1.), ("adhesion_control", 0., 1.))
        if self.active_length:
            fields += (("total_activation", 0., 1.),)
        if self.muscle_activation_first:
            fields += (("muscle_activation", 0., 1e12),)
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
