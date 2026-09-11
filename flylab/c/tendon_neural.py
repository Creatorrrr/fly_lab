"""Explicit BANC rate-to-tendon hypotheses, separate from muscle anatomy.

The long-tendon muscle roster is anatomical. Its conversion to four equal
tarsal torques is an engineering approximation. MNad26 has no resolved muscle
target in BANC v888: the abdominal common/opponent projection is a population
experiment, never an inferred anatomical pitch/yaw muscle assignment.
"""

import copy
import math

import numpy as np

from .integrity import digest, finite

LTM_EVIDENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11348827/"
ABDOMEN_EVIDENCE = "https://www.virtualflybrain.org/term/mnad26-fbbt_20011146/"


def build_tendon_spec(graph, leg_spec, control):
    leg_rows = {row["leg"]: row for row in leg_spec["rows"]}
    abdominal = {side: [] for side in ("left", "right")}
    abdominal_ids = []
    for node in graph.nodes:
        a = node.get("source_annotations", {})
        if node["super_class"] != "motor" or a.get("body_part_effector") != "abdomen":
            continue
        abdominal_ids.append(node["id"])
        side = node["soma_side"]
        if (
            node.get("cell_type") == "MNad26"
            and node.get("class") == "abdomen_motor_neuron"
            and side in abdominal
            and a.get("nerve") == f"{side}_first_abdominal_nerve"
            and a.get("cell_sub_class") != "abdomen_neurosecretory_cell"
        ):
            abdominal[side].append(node["id"])
    rows = []
    for name, limits in zip(control.names, control.limits):
        row = {
            "name": name,
            "status": "unmapped",
            "groups": [],
            "limits": limits.tolist(),
        }
        if name.endswith("_tarsus") and name[:-7] in leg_rows:
            leg = name[:-7]
            muscle = next(
                (
                    g
                    for g in leg_rows[leg]["muscles"]
                    if g["muscle"] == "long_tendon_muscle"
                ),
                None,
            )
            row.update(
                maximum_input=0.1,
                hypothesis="Positive LTM output drives all four distal tarsal hinges; passive springs provide release. The first tarsal joint keeps its depressor/levator servo.",
                evidence=[LTM_EVIDENCE],
            )
            if muscle:
                row.update(
                    status="engineering_map",
                    groups=[
                        {
                            "source": "leg_muscle",
                            "leg": leg,
                            "muscle": muscle["muscle"],
                            "ids": muscle["ids"].copy(),
                            "coefficient": 1.0,
                        }
                    ],
                )
        elif name in ("abdomen_pitch", "abdomen_yaw"):
            row.update(
                maximum_input=0.05,
                hypothesis="MNad26 bilateral mean drives positive pitch only; (right-left)/2 drives signed yaw. Target muscles, moment arms and physiological signs are unresolved. No tonic drive or invented antagonist.",
                evidence=[ABDOMEN_EVIDENCE],
            )
            if all(abdominal.values()):
                row.update(
                    status="population_hypothesis",
                    groups=[
                        {
                            "source": "population_mean",
                            "side": side,
                            "cell_type": "MNad26",
                            "ids": abdominal[side].copy(),
                            "coefficient": -0.5
                            if name == "abdomen_yaw" and side == "left"
                            else 0.5,
                        }
                        for side in ("left", "right")
                    ],
                )
        else:
            raise ValueError("Unsupported neural tendon: " + name)
        if not row["groups"]:
            row["reason"] = (
                "Required same-specimen muscle or bilateral MNad26 cohort is absent"
            )
        if not limits[0] <= -row["maximum_input"] < row["maximum_input"] <= limits[1]:
            raise ValueError("Neural tendon envelope exceeds physical limits")
        rows.append(row)
    assigned_abdominal = {
        identity
        for row in rows
        for group in row["groups"]
        if group["source"] == "population_mean"
        for identity in group["ids"]
    }
    return {
        "schema": "flylab.tendon-neural.v1",
        "graph_hash": graph.hash,
        "rows": rows,
        "rate_scale_Hz": 100.0,
        "activation_tau_s": 0.03,
        "input_model": "maximum_input * tanh(signed_output_Hz / rate_scale_Hz)",
        "tarsal_pooling": "Reuse the selected leg adapter's LTM pooling",
        "abdominal_pooling": "Mean within each side, then equal bilateral weights",
        "abdominal_unassigned_ids": [
            identity for identity in abdominal_ids if identity not in assigned_abdominal
        ],
        "biological_validation": False,
        "manual_policy": "Named manual inputs override only that tendon until returned to neural mode",
        "disconnect_policy": "Immediately zero neural-mode inputs and their filter state; explicit manual inputs remain independent",
    }


class NeuralTendonAdapter:
    def __init__(self, graph, leg_spec, body):
        self.control = body.tendon_control
        self.spec = build_tendon_spec(graph, leg_spec, self.control)
        self.hash = digest(self.spec)
        self.rows = self.spec["rows"]
        self.groups = [
            [(group, graph.resolve(group["ids"])) for group in row["groups"]]
            for row in self.rows
        ]
        self.motor_indices = np.array(
            sorted(
                {int(i) for groups in self.groups for _, ids in groups for i in ids}
            ),
            np.int32,
        )
        self.modes = {
            r["name"]: "neural" if r["groups"] else "manual" for r in self.rows
        }
        self.activation = np.zeros(len(self.rows))
        self.last = []
        # Observation metadata only; this does not change physical controls.
        self.control.neural_adapter = self

    @property
    def rerouted_legs(self):
        return {
            g["leg"]
            for row in self.rows
            for g in row["groups"]
            if g["source"] == "leg_muscle"
        }

    def validate_modes(self, modes):
        if not isinstance(modes, dict) or not modes:
            raise ValueError("Provide nonempty named tendon modes")
        mapped = {r["name"] for r in self.rows if r["groups"]}
        for name, mode in modes.items():
            if name not in self.modes or mode not in ("neural", "manual"):
                raise ValueError("Unknown tendon name or control mode")
            if mode == "neural" and name not in mapped:
                raise ValueError("This tendon has no neural mapping")

    def set_modes(self, modes):
        self.validate_modes(modes)
        for name, mode in modes.items():
            self.modes[name] = mode
            self.activation[self.control.index[name]] = 0.0

    def decode(self, rate_table, leg_outputs, disconnected, dt):
        finite(dt, "tendon dt", 0.0001, 0.05)
        next_activation = self.activation.copy()
        commands, diagnostics = {}, []
        alpha = 1 - math.exp(-dt / self.spec["activation_tau_s"])
        for i, (row, groups) in enumerate(zip(self.rows, self.groups)):
            drive, sources = 0.0, []
            for group, indices in groups:
                rates = np.array([rate_table[int(j)] for j in indices])
                if group["source"] == "leg_muscle":
                    output = leg_outputs[group["leg"]][group["muscle"]]
                else:
                    output = float(np.mean(rates))
                drive += group["coefficient"] * output
                sources.append(
                    {
                        "ids": group["ids"].copy(),
                        "rates_Hz": rates.tolist(),
                        "output_Hz": output,
                        "coefficient": group["coefficient"],
                    }
                )
            requested = row["maximum_input"] * math.tanh(
                drive / self.spec["rate_scale_Hz"]
            )
            mode = self.modes[row["name"]]
            if mode == "neural":
                next_activation[i] = (
                    0.0
                    if disconnected
                    else self.activation[i] + alpha * (requested - self.activation[i])
                )
                commands[row["name"]] = float(next_activation[i])
            else:
                next_activation[i] = 0.0
            diagnostics.append(
                {
                    "name": row["name"],
                    "status": row["status"],
                    "mode": mode,
                    "sources": sources,
                    "requested_input": requested,
                    "neural_input": float(next_activation[i]),
                    "motor_disconnected": bool(disconnected),
                }
            )
        self.activation[:] = next_activation
        self.last = diagnostics
        return commands

    def summary(self):
        return {
            "schema": self.spec["schema"],
            "hash": self.hash,
            "connected_controls": sum(bool(r["groups"]) for r in self.rows),
            "modes": self.modes.copy(),
            "controls": copy.deepcopy(self.rows),
            "last": copy.deepcopy(self.last),
            "biological_validation": False,
            "abdominal_unassigned_neurons": len(self.spec["abdominal_unassigned_ids"]),
            "disconnect_policy": self.spec["disconnect_policy"],
        }

    def snapshot(self):
        return {
            "hash": self.hash,
            "activation": self.activation.copy(),
            "modes": self.modes.copy(),
            "last": copy.deepcopy(self.last),
        }

    def validate_state(self, state):
        if not isinstance(state, dict) or state.get("hash") != self.hash:
            raise ValueError("Neural tendon checkpoint identity mismatch")
        values = np.asarray(state.get("activation"))
        limits = np.array([row["maximum_input"] for row in self.rows])
        lower = np.array(
            [
                -row["maximum_input"]
                if any(g["coefficient"] < 0 for g in row["groups"])
                else 0.0
                for row in self.rows
            ]
        )
        if (
            values.dtype != self.activation.dtype
            or values.shape != self.activation.shape
            or not np.isfinite(values).all()
            or np.any(values < lower)
            or np.any(values > limits)
        ):
            raise ValueError("Invalid neural tendon activation state")
        modes = state.get("modes")
        self.validate_modes(modes)
        if set(modes) != set(self.modes):
            raise ValueError("Incomplete tendon modes")
        if any(
            values[i] != 0
            for i, row in enumerate(self.rows)
            if modes[row["name"]] == "manual"
        ):
            raise ValueError("Manual tendons cannot retain a neural activation")
        if not isinstance(state.get("last"), list):
            raise TypeError("Invalid tendon diagnostic state")
        return {
            "activation": values.copy(),
            "modes": modes.copy(),
            "last": copy.deepcopy(state["last"]),
        }

    def restore(self, state):
        validated = self.validate_state(state)
        self.activation[:] = validated["activation"]
        self.modes = validated["modes"]
        self.last = validated["last"]

    def verify_applied_inputs(self):
        """A restored engine boundary must agree with the physical motor state."""
        applied = self.control.body.d.ctrl[self.control.act_ids]
        if any(
            applied[i] != self.activation[i]
            for i, name in enumerate(self.control.names)
            if self.modes[name] == "neural"
        ):
            raise ValueError(
                "Neural tendon state differs from restored physical inputs"
            )
