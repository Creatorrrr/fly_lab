"""Checkpointable experimental target sensing and neural input coordination."""

import copy
import math

import numpy as np

from ..common import to_ui
from .target_neural import BancTargetEncoder
from .target_sense import TargetSensor


class BancTargetNavigation:
    schema = "flylab.banc-target-navigation.v1"

    def __init__(self, graph, config):
        if not isinstance(config, dict) or set(config) != {
            "target_xz_mm",
            "forward_drive",
        }:
            raise ValueError(
                "Explicit target coordinates and descending profile required"
            )
        self.sensor = TargetSensor(config["target_xz_mm"])
        self.encoder = BancTargetEncoder(graph, forward_drive=config["forward_drive"])
        self.cut = False
        self.history = []
        self.start_time = 0.0
        self.start_distance = 0.0

    def set_target(self, target, time_s, position, rotation):
        sensor = TargetSensor(target)
        observation = sensor.observe(position, rotation)
        self.sensor = sensor
        self.encoder.turn, self.encoder.braking = 0, False
        self.start_time, self.start_distance = float(time_s), observation.distance_mm
        self.history = []
        self.record(time_s, position, True)

    def set_cut(self, value):
        if type(value) is not bool:
            raise ValueError("Boolean target cue cut required")
        self.cut = value

    def drive(self, position, rotation):
        return self.encoder.drive(self.sensor.observe(position, rotation, cut=self.cut))

    def record(self, time_s, position, healthy):
        p = to_ui(position)
        self.history.append([float(time_s), float(p[0]), float(p[2]), bool(healthy)])
        self.history = self.history[-301:]

    def view(self, position, rotation):
        observation = self.sensor.observe(position, rotation)
        held, stop_path = False, None
        if len(self.history) == 301:
            points = np.asarray([r[1:3] for r in self.history])
            distance = np.linalg.norm(points - self.sensor.target_xz, axis=1)
            stop_path = float(
                np.linalg.norm(np.diff(points[100:], axis=0), axis=1).sum()
            )
            held = bool(
                np.all(distance <= 2.0)
                and stop_path <= 0.5
                and all(r[3] for r in self.history)
            )
        return {
            "target_xz_mm": list(self.sensor.target_xz),
            "bearing_rad": observation.bearing_rad,
            "distance_mm": observation.distance_mm,
            "cue_cut": self.cut,
            "turn_request": self.encoder.turn,
            "brake_request": self.encoder.braking,
            "physical_hold_confirmed": held,
            "stop_window_path_mm": stop_path,
            "goal_elapsed_s": (self.history[-1][0] - self.start_time)
            if self.history
            else 0.0,
            "started_outside_arrival_radius": self.start_distance > 2.0,
            "status": "EXPERIMENTAL_NAVIGATION_NOT_VALIDATED",
            "input_model": "Ideal target bearing/range and calibrated inhibitory BANC input ports",
            "biological_validation": False,
            "encoder_hash": self.encoder.identity,
            "recent_path_xz_mm": [r[1:3] for r in self.history[::10]],
        }

    def snapshot(self):
        return {
            "schema": self.schema,
            "config": {
                "target_xz_mm": list(self.sensor.target_xz),
                "forward_drive": self.encoder.forward_drive,
            },
            "encoder": self.encoder.snapshot(),
            "cut": self.cut,
            "history": copy.deepcopy(self.history),
            "start_time": self.start_time,
            "start_distance": self.start_distance,
        }

    def restore(self, state, *, time_s):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema",
                "config",
                "encoder",
                "cut",
                "history",
                "start_time",
                "start_distance",
            }
            or state["schema"] != self.schema
        ):
            raise ValueError("Invalid BANC target state")
        if (
            state["config"] != self.snapshot()["config"]
            or type(state["cut"]) is not bool
        ):
            raise ValueError("Target checkpoint configuration mismatch")
        for key in ("start_time", "start_distance"):
            value = state[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid target checkpoint metric")
        rows = state["history"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 301:
            raise ValueError("Bounded physical target history required")
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 4
                or type(row[3]) is not bool
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in row[:3]
                )
            ):
                raise ValueError("Invalid physical target observation")
        times = np.asarray([r[0] for r in rows])
        if (
            times[0] < state["start_time"] - 1e-9
            or abs(times[-1] - time_s) > 1e-9
            or not np.allclose(np.diff(times), 0.005, atol=1e-9, rtol=0)
        ):
            raise ValueError("Target history and physical clock disagree")
        self.encoder.restore(state["encoder"])
        self.cut = state["cut"]
        self.history = copy.deepcopy(rows)
        self.start_time, self.start_distance = (
            state["start_time"],
            state["start_distance"],
        )
