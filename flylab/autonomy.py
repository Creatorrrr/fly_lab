"""Sensory navigation for the explicit FlyGym hybrid-controller baseline.

The policy receives receptor observations, never world/target coordinates.
Its commands drive the existing physical CPG/reflex controller, not BANC.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .common import number

AUTONOMOUS_MODE = "FLYGYM_AUTONOMOUS"
SENSOR_MODEL = {
    "kind": "four-site-odor-v1",
    "field": "inverse-square",
    "half_concentration": 1.0,
    "core_radius_mm": 0.5,
}


def autonomous_world():
    return {
        "bounds": [24, 16, 18],
        "cueOn": True,
        "cueAngle": 0.0,
        "foodOn": True,
        "sources": [
            {"id": "food-1", "kind": "food", "p": [12.0, 0.7, -6.0], "strength": 1.0}
        ],
        "obstacles": [],
    }


@dataclass(frozen=True)
class AutonomyParameters:
    task: str = "odor_seek"
    speed: float = 2.0
    odor_gain: float = 40.0
    stop_odor: float = 0.26
    resume_odor: float = 0.18
    proximity_mm: float = 6.0

    def __post_init__(self):
        if self.task not in ("odor_seek", "walk"):
            raise ValueError("Unknown autonomous task")
        for name, lo, hi in (
            ("speed", 0.1, 3.3),
            ("odor_gain", 0.0, 100.0),
            ("stop_odor", 0.01, 0.95),
            ("resume_odor", 0.0, 0.94),
            ("proximity_mm", 1.0, 6.0),
        ):
            number(getattr(self, name), name, lo, hi)
        if self.resume_odor >= self.stop_odor:
            raise ValueError(
                "Resume concentration must be below the stop concentration"
            )


class SensoryWalkingPolicy:
    version = "flygym-sensory-walking-v3"

    def __init__(self, parameters=None):
        self.p = AutonomyParameters(**(parameters or {}))
        self.enabled = True
        self.arrived = False
        self.arrival_s = 0.0
        self.turn_side = 1
        self.avoid_s = 0.0
        self.tick = 0
        self.last = {"state": "ready", "odor": [0.0, 0.0], "contrast": 0.0}

    def step(self, packet, dt, disabled=()):
        number(dt, "policy timestep", 0.0001, 0.1)
        odor = np.asarray(packet["odor"], dtype=float)
        ranges = np.asarray(packet["nearRanges"], dtype=float)
        if (
            odor.shape != (2,)
            or ranges.shape != (9,)
            or not np.isfinite(odor).all()
            or not np.isfinite(ranges).all()
            or np.any(odor < 0)
            or np.any(odor > 1)
            or np.any(ranges < 0)
        ):
            raise ValueError("Invalid autonomous sensory observations")
        disabled = set(disabled)
        if "*" in disabled or "odor" in disabled or "odor_mean" in disabled:
            odor = np.zeros(2)
        else:
            odor = odor.copy()
            for i, side in enumerate(("left", "right")):
                if "odor_" + side in disabled:
                    odor[i] = 0.0
        mean = float(odor.mean())
        contrast = float((odor[0] - odor[1]) / max(mean, 1e-9))
        if mean < self.p.resume_odor:
            self.arrived = False
            self.arrival_s = 0.0
        elif mean >= self.p.stop_odor:
            self.arrival_s += dt
            if self.arrival_s >= 0.1:
                self.arrived = True
        else:
            self.arrival_s = 0.0

        state = "walking" if self.p.task == "walk" else "seeking"
        speed = self.p.speed
        yaw = (
            float(np.clip(-self.p.odor_gain * contrast, -2.8, 2.8))
            if self.p.task == "odor_seek"
            else 0.0
        )
        contact_enabled = "*" not in disabled and "contact" not in disabled
        proximity_enabled = "*" not in disabled and "proximity" not in disabled
        obstructed = (contact_enabled and bool(packet["contact"])) or (
            proximity_enabled and float(np.min(ranges[3:6])) < self.p.proximity_mm
        )
        if obstructed:
            if not self.avoid_s:
                self.turn_side = 1 if ranges[5:].sum() >= ranges[:4].sum() else -1
            self.avoid_s = 0.35
        elif self.avoid_s:
            self.avoid_s = max(0.0, self.avoid_s - dt)
        if self.avoid_s:
            # Keep both CPG drives nonnegative while shortening the inside
            # stride. Opposite frequency signs did not reliably pivot FlyBody
            # and could trap it against a wall. Start this turn before contact.
            state, speed, yaw = "avoiding", 1.32, self.turn_side * 2.8
        if self.p.task == "odor_seek" and self.arrived:
            state, speed, yaw = "at_food", 0.0, 0.0
        if not self.enabled:
            state, speed, yaw = "stopped", 0.0, 0.0
        self.tick += 1
        self.last = {"state": state, "odor": odor.tolist(), "contrast": contrast}
        return {"forwardSpeed": speed, "yawRate": yaw, "verticalSpeed": 0.0}

    def summary(self):
        return dict(
            enabled=True,
            walking_enabled=self.enabled,
            version=self.version,
            parameters=asdict(self.p),
            **copy.deepcopy(self.last),
            controller="FlyGym HybridTurningController",
            neural_control=False,
            biological_validation=False,
            input_scope="bilateral odor, proximity and contact",
        )

    def snapshot(self):
        return {
            "version": self.version,
            "parameters": asdict(self.p),
            "enabled": self.enabled,
            "arrived": self.arrived,
            "arrival_s": self.arrival_s,
            "turn_side": self.turn_side,
            "avoid_s": self.avoid_s,
            "tick": self.tick,
            "last": copy.deepcopy(self.last),
        }

    def restore(self, state):
        if state.get("version") != self.version or state.get("parameters") != asdict(
            self.p
        ):
            raise ValueError("Autonomous policy identity mismatch")
        if (
            type(state.get("enabled")) is not bool
            or type(state.get("arrived")) is not bool
        ):
            raise ValueError("Invalid autonomous state flags")
        if type(state.get("tick")) is not int or state["tick"] < 0:
            raise ValueError("Invalid autonomous clock")
        if type(state.get("turn_side")) is not int or state["turn_side"] not in (-1, 1):
            raise ValueError("Invalid autonomous turn side")
        number(state.get("arrival_s"), "arrival dwell", 0, 1e8)
        number(state.get("avoid_s"), "avoidance dwell", 0, 0.35)
        for key in (
            "enabled",
            "arrived",
            "arrival_s",
            "turn_side",
            "avoid_s",
            "tick",
            "last",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
