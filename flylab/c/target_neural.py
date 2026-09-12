"""Experimental target cue -> anatomical BANC inhibitory input ports.

The cell identities and existing edges come from BANC. Input currents,
projection efficacy and cue thresholds are engineering hypotheses. This is
neither a reconstructed visual pathway nor a verified biological stop circuit.
The encoder never creates joint targets, gait phases, or body poses.
"""

import math

import numpy as np

from .integrity import digest
from .target_sense import TargetObservation

# Ordered as DNg105, IN08(F/M/H), IN19(F/M/H), IN09(F/M/H).
# DNg105 soma is CONTRALATERAL to the motor targets used by each input port.
LEFT = (
    "720575941535975873",
    "720575941537865165",
    "720575941504382999",
    "720575941553598180",
    "720575941541544028",
    "720575941594854251",
    "720575941567768423",
    "720575941429526743",
    "720575941450559566",
    "720575941592869928",
)
RIGHT = (
    "720575941480799395",
    "720575941496630400",
    "720575941509326430",
    "720575941438399807",
    "720575941496314697",
    "720575941572062729",
    "720575941561135887",
    "720575941623146186",
    "720575941613448668",
    "720575941503464587",
)


class BancTargetEncoder:
    """A nonperiodic, hysteretic target-cue controller for explicit experiments."""

    schema = "flylab.banc-target-encoder.v2"

    def __init__(self, graph, *, forward_drive=0.0):
        if type(forward_drive) not in (int, float) or forward_drive not in (
            0.0,
            5000.0,
        ):
            raise ValueError("Use a declared experimental descending input profile")
        self.n = graph.n
        self.forward_drive = float(forward_drive)
        self.ports = graph.resolve(
            ["flywire:banc:888:" + root for root in LEFT + RIGHT]
        )
        self.descending = graph.resolve(
            [
                "flywire:banc:888:720575941500851362",
                "flywire:banc:888:720575941626500746",
            ]
        )
        # Existing source projections inhibit opposite coxa rotators. The
        # selected front/hind assignment is an engineering input hypothesis;
        # the middle-leg response did not support this turning assignment.
        self.turn_left = self.ports[[4, 6, 11, 13]]
        self.turn_right = self.ports[[1, 3, 14, 16]]
        for side in (self.ports[:10], self.ports[10:]):
            types = (
                ("DNg105",) + ("IN08A006",) * 3 + ("IN19A003",) * 3 + ("IN09A002",) * 3
            )
            for i, expected in zip(side, types):
                node = graph.nodes[i]
                if (
                    node["cell_type"] != expected
                    or node["nt_type"] != "GABA"
                    or node["super_class"] == "motor"
                ):
                    raise ValueError("BANC inhibitory port anatomy mismatch")
        if any(graph.nodes[i]["cell_type"] != "DNg100" for i in self.descending):
            raise ValueError("BANC bilateral descending identity mismatch")
        self.identity = digest(
            {
                "schema": self.schema,
                "graph_hash": graph.hash,
                "left": LEFT,
                "right": RIGHT,
                "turn_left": [LEFT[4], LEFT[6], RIGHT[1], RIGHT[3]],
                "turn_right": [LEFT[1], LEFT[3], RIGHT[4], RIGHT[6]],
                "turn_model": "front-hind-antagonist-inhibition-v1",
                "projection_efficacy": 64.0,
                "projection_scope": "existing leg motor edges only",
                "background_input": -5000.0,
                "command_increment": 10000.0,
                "forward_drive": self.forward_drive,
                "turn_enter_rad": 0.15,
                "turn_leave_rad": 0.06,
                "brake_enter_mm": 1.25,
                "brake_leave_mm": 1.75,
                "rear_tie_rad": 2.7,
            }
        )
        self.turn = 0
        self.braking = False

    def calibrate_weights(self, weights, motor_ids):
        """Scale only existing inhibitory port -> anatomical motor entries."""
        result = weights.copy()
        motor, ports = np.zeros(self.n, bool), np.zeros(self.n, bool)
        motor[motor_ids], ports[self.ports] = True, True
        affected = np.repeat(motor, np.diff(result.indptr)) & ports[result.indices]
        if not np.any(affected) or np.any(result.data[affected] >= 0):
            raise ValueError("Existing negative motor projections required")
        result.data[affected] *= np.float32(64.0)
        return result

    def drive(self, observation):
        if (
            not isinstance(observation, TargetObservation)
            or type(observation.visible) is not bool
            or not math.isfinite(observation.bearing_rad)
            or not math.isfinite(observation.distance_mm)
            or abs(observation.bearing_rad) > math.pi
            or observation.distance_mm < 0
        ):
            raise ValueError("Finite local target observation required")
        result = np.zeros(self.n, np.float32)
        result[self.ports] = -5000.0
        result[self.descending] = self.forward_drive
        if not observation.visible:
            # Remove all target-dependent request memory immediately. Tonic
            # port background/arousal are independent of the target cue.
            self.turn, self.braking = 0, False
            return result
        distance, bearing = observation.distance_mm, observation.bearing_rad
        self.braking = distance <= (1.75 if self.braking else 1.25)
        if self.braking:
            self.turn = 0
            result[self.ports] += 10000.0
            return result
        if abs(bearing) <= 0.06:
            self.turn = 0
        elif abs(bearing) >= 0.15:
            self.turn = -1 if abs(bearing) >= 2.7 else (1 if bearing > 0 else -1)
        if self.turn:
            selected = self.turn_left if self.turn == 1 else self.turn_right
            result[selected] += 10000.0
        return result

    def snapshot(self):
        return {
            "schema": self.schema,
            "hash": self.identity,
            "turn": self.turn,
            "braking": self.braking,
        }

    def restore(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "hash", "turn", "braking"}
            or state["schema"] != self.schema
            or state["hash"] != self.identity
            or type(state["turn"]) is not int
            or state["turn"] not in (-1, 0, 1)
            or type(state["braking"]) is not bool
            or (state["braking"] and state["turn"] != 0)
        ):
            raise ValueError("Invalid target encoder checkpoint")
        self.turn, self.braking = state["turn"], state["braking"]
