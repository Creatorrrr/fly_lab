"""Explicit ideal target-bearing sensor for engineering navigation experiments.

This is simulated perception, not a reconstruction of BANC visual processing.
Only this sensor knows target coordinates. A neural encoder receives local
bearing and proximity; the motor decoder must never receive a target or pose.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..common import to_physics


@dataclass(frozen=True)
class TargetObservation:
    bearing_rad: float
    distance_mm: float
    visible: bool


def validate_target(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or abs(v) > 100
            for v in value
        )
    ):
        raise ValueError("Target requires finite X/Z millimetres within [-100, 100]")
    return tuple(float(v) for v in value)


class TargetSensor:
    schema = "flylab.ideal-target-sensor.v1"

    def __init__(self, target_xz):
        self.target_xz = validate_target(target_xz)

    def observe(self, position, rotation, *, cut=False):
        """Positive bearing means anatomical left; UI Z increases to right."""
        p, r = np.asarray(position, float), np.asarray(rotation, float)
        if (
            p.shape != (3,)
            or r.shape != (3, 3)
            or not np.isfinite(p).all()
            or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
        ):
            raise ValueError("Finite physical position and proper rotation required")
        if type(cut) is not bool:
            raise ValueError("Boolean target sensory cut required")
        if cut:
            return TargetObservation(0.0, 0.0, False)
        target = to_physics([self.target_xz[0], 0.0, self.target_xz[1]])
        delta = target - p
        delta[2] = 0.0
        # Horizontal forward/left avoid converting pitch to a goal cue.
        yaw = math.atan2(r[1, 0], r[0, 0])
        bearing = math.atan2(delta[1], delta[0]) - yaw
        return TargetObservation(
            math.atan2(math.sin(bearing), math.cos(bearing)),
            float(np.linalg.norm(delta[:2])),
            True,
        )

    def snapshot(self):
        return {"schema": self.schema, "target_xz_mm": list(self.target_xz)}

    @classmethod
    def from_snapshot(cls, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "target_xz_mm"}
            or state["schema"] != cls.schema
        ):
            raise ValueError("Invalid target sensor checkpoint")
        return cls(state["target_xz_mm"])
