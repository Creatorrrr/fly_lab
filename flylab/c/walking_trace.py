"""Read-only 200 Hz BANC instrumentation, independent of neural dynamics.

Records measured states, not interpolated poses. No stepping, forward dynamics,
pose edits, or writes to mjData are allowed here. Contact slip uses both bodies'
velocities at the actual contact point, not the origin of the last tarsus.
"""

import copy
import math
import uuid
from collections import deque
from dataclasses import asdict

import numpy as np

from ..body import LEGS, LINKS
from ..common import S, to_ui
from .integrity import bounded_int

TRACE_SCHEMA = "flylab.banc-trace.v1"
CONTROL_DT_S = 0.005


def command_interval(tick):
    """The command on row t produced the already-measured state at t."""
    return None if tick == 0 else {
        "start_tick": tick - 1, "end_tick": tick,
        "start_s": (tick - 1) * CONTROL_DT_S, "end_s": tick * CONTROL_DT_S,
    }


def capture_navigation(session, position, rotation):
    navigation = getattr(session, "navigation", None)
    if navigation is None:
        return None
    # observe() is pure; drive() would change hysteresis and must never be used.
    cue = navigation.sensor.observe(position, rotation)
    return {
        "target_xz_mm": list(navigation.sensor.target_xz),
        "bearing_rad": cue.bearing_rad, "distance_mm": cue.distance_mm,
        "cue_cut": navigation.cut,
        "turn_request": navigation.encoder.turn,
        "brake_request": navigation.encoder.braking,
    }


def capture_neural_request(session, drive, position, rotation):
    """Copy the existing host drive after encoding; no GPU readback or encode."""
    navigation = getattr(session, "navigation", None)
    return {
        "start_tick": session.control_tick,
        "end_tick": session.control_tick + 1,
        "navigation_at_start": capture_navigation(session, position, rotation),
        "descending_drive": np.asarray(drive)[session.descending].tolist(),
        "steering_port_drive": np.asarray(drive)[navigation.encoder.ports].tolist()
        if navigation else [],
        "cuts": session.cuts.copy(),
        "external_force_at_start": {
            "push_left_s": float(session.body.push_left),
            "push_force": session.body.push_force.tolist(),
        },
    }


def capture_initial_state(session):
    """One CPU physics snapshot at the recording anchor, never a neural snapshot."""
    body = session.body
    physical_time = float(body.physics_time())
    if not math.isfinite(physical_time) or abs(physical_time - session.control_tick * CONTROL_DT_S) > 1e-5:
        raise ValueError("Trace initial physics clock mismatch")
    def port(index):
        node = session.graph.nodes[int(index)]
        return {key: node.get(key) for key in ("id", "cell_type", "soma_side", "output_side")}
    navigation = getattr(session, "navigation", None)
    return {
        "control_tick": session.control_tick,
        "time_s": session.control_tick * CONTROL_DT_S,
        "body": body.snapshot(),
        "body_hash": body.model_hash,
        "graph_hash": session.graph.hash,
        "model_hash": session.network.identity,
        "adapter_hash": session.adapter.identity,
        "physics_identity": body.physics_identity(),
        "body_options": asdict(session.options),
        "world": copy.deepcopy(session.world),
        "parameters": session.p.state_dict() if hasattr(session.p, "state_dict") else asdict(session.p),
        "joint_commands": {
            "names": list(body.joint_names),
            "joint_range_rad": body.m.jnt_range[body.joint_ids].tolist(),
            "joint_limited": body.m.jnt_limited[body.joint_ids].astype(bool).tolist(),
            "actuator_control_range_rad": body.m.actuator_ctrlrange[body.act_ids].tolist(),
            "actuator_control_limited": body.m.actuator_ctrllimited[body.act_ids].astype(bool).tolist(),
            "interpretation": "joint_targets_rad and ctrl retain raw requests; MuJoCo applies configured control limits internally. Tracking error is requested minus measured angle, and can include an unreachable request.",
        },
        "descending_ports": [port(i) for i in session.descending],
        "steering_ports": [port(i) for i in navigation.encoder.ports] if navigation else [],
        "units": {"time": "model seconds", "position": "mm", "angle": "rad",
                  "neural_drive": "model units", "force": "MuJoCo model units; not SI calibrated"},
        "command_contract": {
            "control_dt_s": CONTROL_DT_S,
            "physics_dt_s": float(body.m.opt.timestep),
            "state_timing": "row t is the state after integrating the command in (t-1,t]",
            "replay": "restore initial.body with FlyGymBody.restore; skip the anchor row; restore each row's neural_request.external_force_at_start push_force/push_left_s, then apply its joint_targets_rad, adhesion and tendon_inputs for one control_dt_s",
            "neural_input_timing": "neural_request is copied before neural integration; navigation is measured at the row end",
            "descending_order": "descending_drive follows descending_ports; soma_side is source anatomy, not inferred steering side",
            "initial_state": "body.snapshot includes mjSTATE_INTEGRATION, held action and pending push force/duration",
            "external_updates": "subsequent external state edits require their own recorded intervention; external_force_at_start records the held push request",
            "verification": "complete data is a replay candidate, not a verified numerical PASS",
        },
    }


def contact_diagnostics(body, *, support_threshold_bw=0.02):
    """Force-weighted tangential relative speed at compressive foot contacts.

    The threshold selects load-bearing samples for diagnostics only. Adhesion
    reaction is included in the normal force: this is not tissue strain or
    net weight support. Sampling at 5 ms can miss sub-control contact events.
    """
    if not math.isfinite(support_threshold_bw) or support_threshold_bw <= 0:
        raise ValueError("Positive finite diagnostic support threshold required")
    mapping, _ = body._force_maps["feet"]
    normal = np.zeros(6)
    weighted_speed = np.zeros(6)
    peak_speed = np.zeros(6)
    counts = np.zeros(6, dtype=int)
    jac_a, jac_b = np.zeros((3, body.m.nv)), np.zeros((3, body.m.nv))
    wrench = np.zeros(6)
    weight = max(float(body.weight0), 1e-30)
    for index, contact in enumerate(body.d.contact[: body.d.ncon]):
        if contact.exclude:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        segment = (
            mapping[a]
            if b in body.support_geom_ids
            else mapping[b]
            if a in body.support_geom_ids
            else -1
        )
        if segment < 0:
            continue
        body.mj.mj_contactForce(body.m, body.d, index, wrench)
        force_bw = max(0.0, float(wrench[0])) / weight
        if force_bw == 0:
            continue
        body.mj.mj_jac(
            body.m, body.d, jac_a, None, contact.pos, int(body.m.geom_bodyid[a])
        )
        body.mj.mj_jac(
            body.m, body.d, jac_b, None, contact.pos, int(body.m.geom_bodyid[b])
        )
        relative = (jac_a - jac_b) @ body.d.qvel
        axis = np.asarray(contact.frame).reshape(3, 3)[0]
        tangent = relative - float(relative @ axis) * axis
        speed = float(np.linalg.norm(tangent))
        leg = int(segment) // 5
        normal[leg] += force_bw
        weighted_speed[leg] += force_bw * speed
        peak_speed[leg] = max(peak_speed[leg], speed)
        counts[leg] += 1
    slip = np.divide(weighted_speed, normal, out=np.zeros(6), where=normal > 0)
    if not all(np.isfinite(v).all() for v in (normal, slip, peak_speed)):
        raise ValueError("Nonfinite contact diagnostic")
    loaded = normal >= support_threshold_bw
    return {
        "legs": list(LEGS),
        "normal_load_bw": normal.tolist(),
        "load_bearing": loaded.tolist(),
        "contact_count": counts.tolist(),
        "slip_mm_s": [float(slip[i]) if loaded[i] else None for i in range(6)],
        "peak_contact_slip_mm_s": [
            float(peak_speed[i]) if loaded[i] else None for i in range(6)
        ],
        "support_threshold_bw": support_threshold_bw,
        "source": "force-weighted tangential relative velocity at actual MuJoCo contact points",
        "caveat": "normal force includes adhesion; 5ms boundary sampling; not a gait-success verdict",
    }


def capture_control(session):
    """Copy state at an already-integrated control boundary; never mj_forward."""
    body = session.body
    p, rotation, velocity = body.pose()
    feet = contact_diagnostics(body)
    positions = body.d.xpos
    segments = {
        name: to_ui(positions[int(body.body_ids[index])])
        for name, index in body.body_indices.items()
        if name in ("c_thorax", "c_head") or name.startswith("c_abdomen")
    }
    if "c_abdomen12" in segments:
        segments["c_abdomen"] = segments["c_abdomen12"]
    legs = {
        leg: [
            to_ui(positions[int(body.body_ids[body.body_indices[f"{leg}_{link}"]])])
            for link in LINKS
            if f"{leg}_{link}" in body.body_indices
        ]
        for leg in LEGS
    }
    actual = body.d.qpos[body.qpos_ids]
    targets = np.asarray(body.last_action.joint_angles)
    sample = {
        "control_tick": session.control_tick,
        "time_s": session.control_tick * 0.005,
        "command_interval": command_interval(session.control_tick),
        "body": {
            "position": to_ui(p),
            "basis": (S @ rotation).tolist(),
            "yaw": math.atan2(-rotation[1, 0], rotation[0, 0]),
            "speed": float(velocity[3:] @ rotation[:, 0]),
            "legs": legs,
            "segments": segments,
            "contactsBW": feet["normal_load_bw"],
        },
        "qpos": body.d.qpos.tolist(),
        "qvel": body.d.qvel.tolist(),
        "ctrl": body.d.ctrl.tolist(),
        "joint_angles_rad": actual.tolist(),
        "joint_targets_rad": targets.tolist(),
        "joint_tracking_error_rad": (targets - actual).tolist(),
        "adhesion": np.asarray(body.last_action.adhesion_onoff, dtype=bool).tolist(),
        "tendon_inputs": dict(session.tendon_inputs),
        "motor_rates": session.rate.tolist(),
        "upright": float(rotation[2, 2]),
        "feet": feet,
        "fault": session.fault or body.fault,
        "navigation": capture_navigation(session, p, rotation),
    }
    request = getattr(session, "_trace_neural_request", None)
    if hasattr(body, 'active_length_scale'):
        sample['joint_stiffness_scale'] = body.active_length_scale.tolist()
    if getattr(body, 'graded_adhesion', False):
        sample['adhesion_control'] = body.d.ctrl[body.sim._intern_adhesionactuatorids_by_fly[body.fly.name]].tolist()
    sample["neural_request"] = copy.deepcopy(request) if request is not None and request["end_tick"] == session.control_tick else None
    if any(not np.isfinite(sample[key]).all() for key in ("qpos", "qvel", "ctrl")):
        raise ValueError("Nonfinite recorded physical state")
    return sample


class WalkingTrace:
    """Bounded observer history; intentionally excluded from model checkpoints."""

    def __init__(self, capacity=6000):
        self.capacity = bounded_int(capacity, "trace capacity", 2, 60000)
        self.samples = deque(maxlen=capacity)
        self.epoch = uuid.uuid4().hex
        self.evicted = 0
        self.initial = None
        self.missing_intervals = 0
        self.incomplete_commands = 0

    def start(self, initial):
        if self.initial is not None or self.samples:
            raise ValueError("Trace initial state is fixed at its first sample")
        bounded_int(initial.get("control_tick"), "initial trace tick", 0, 10**10)
        if initial.get("time_s") != initial["control_tick"] * CONTROL_DT_S:
            raise ValueError("Trace initial clock mismatch")
        self.initial = copy.deepcopy(initial)

    def append(self, sample):
        tick = bounded_int(sample.get("control_tick"), "trace tick", 0, 10**10)
        if sample.get("time_s") != tick * 0.005:
            raise ValueError("Trace clock mismatch")
        if self.samples and tick <= self.samples[-1]["control_tick"]:
            raise ValueError(
                "Trace samples must advance strictly; restore starts a new epoch"
            )
        if not self.samples and self.initial is not None and tick != self.initial["control_tick"]:
            raise ValueError("Initial trace state and first sample disagree")
        if self.samples:
            self.missing_intervals += max(0, tick - self.samples[-1]["control_tick"] - 1)
            if (sample.get("command_interval") != command_interval(tick)
                    or any(key not in sample for key in ("joint_targets_rad", "adhesion", "tendon_inputs"))
                    or sample.get("neural_request") is None):
                self.incomplete_commands += 1
        self.evicted += int(len(self.samples) == self.capacity)
        self.samples.append(copy.deepcopy(sample))

    def export_metadata(self, *, include_initial=False, trace_error=None):
        first = self.samples[0]["control_tick"] if self.samples else None
        reasons = []
        if self.initial is None:
            reasons.append("INITIAL_PHYSICS_STATE_NOT_CAPTURED")
        elif first != self.initial["control_tick"]:
            reasons.append("INITIAL_SAMPLE_EVICTED_OR_UNAVAILABLE")
        if self.missing_intervals:
            reasons.append("MISSING_CONTROL_INTERVALS")
        if trace_error:
            reasons.append("TRACE_CAPTURE_ERROR")
        if self.incomplete_commands:
            reasons.append("INCOMPLETE_COMMAND_RECORDS")
        aligned = self.initial is not None and first == self.initial["control_tick"]
        return {
            "command_replay": {
                "eligible": not reasons, "verified": False,
                "status": "DATA_COMPLETE_NOT_VERIFIED" if not reasons else "UNAVAILABLE",
                "reasons": reasons, "first_available_tick": first,
                "initial_tick": self.initial["control_tick"] if self.initial else None,
                "missing_control_intervals": self.missing_intervals,
            },
            "trace_contract": copy.deepcopy({key: value for key, value in self.initial.items()
                                             if key not in ("body", "control_tick", "time_s")}) if self.initial else None,
            # Never attach an old physical anchor to a later retained first row.
            **({"initial_physics": copy.deepcopy(self.initial) if aligned else None} if include_initial else {}),
        }

    def page(self, *, after=-1, limit=200):
        bounded_int(after, "trace cursor", -1, 10**10)
        bounded_int(limit, "trace page size", 1, 200)
        rows = [s for s in self.samples if s["control_tick"] > after]
        selected = rows[:limit]
        first = self.samples[0]["control_tick"] if self.samples else None
        return {
            "schema": TRACE_SCHEMA,
            "epoch": self.epoch,
            "sample_dt_s": 0.005,
            "capacity": self.capacity,
            "evicted_samples": self.evicted,
            "first_available_tick": first,
            "last_available_tick": self.samples[-1]["control_tick"]
            if self.samples
            else None,
            "gap_before_page": after >= 0 and first is not None and first > after + 1,
            "has_more": len(rows) > limit,
            "next_after": selected[-1]["control_tick"] if selected else after,
            "samples": copy.deepcopy(selected),
        }


def summarize_trace(samples):
    """Descriptive diagnostics only. Missing contact time is not zero slip."""
    if not samples:
        return {"status": "NO_SAMPLES", "gait_verdict": "NOT_ASSESSED"}
    times = np.asarray([row["time_s"] for row in samples])
    ticks = np.asarray([row["control_tick"] for row in samples])
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("Trace order must be strictly increasing")
    continuous = np.diff(ticks) == 1
    legs = []
    for leg in range(6):
        stance = np.asarray([r["feet"]["load_bearing"][leg] for r in samples], bool)
        measured = [
            r["feet"]["slip_mm_s"][leg]
            for r in samples
            if r["feet"]["load_bearing"][leg]
        ]
        legs.append(
            {
                "leg": LEGS[leg],
                "loaded_samples": len(measured),
                "loaded_sample_fraction": float(stance.mean()),
                "stance_transitions": int(
                    np.count_nonzero((stance[1:] != stance[:-1]) & continuous)
                ),
                "loaded_slip_mean_mm_s": float(np.mean(measured)) if measured else None,
                "loaded_slip_p95_mm_s": float(np.percentile(measured, 95))
                if measured
                else None,
            }
        )
    positions = np.asarray([s["body"]["position"] for s in samples])[:, [0, 2]]
    return {
        "status": "MEASURED",
        "gait_verdict": "NOT_ASSESSED",
        "samples": len(samples),
        "start_s": float(times[0]),
        "end_s": float(times[-1]),
        "missing_control_intervals": int(np.maximum(np.diff(ticks) - 1, 0).sum()),
        "horizontal_net_mm": float(np.linalg.norm(positions[-1] - positions[0])),
        "observed_path_mm": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1)[continuous].sum()
        ),
        "upright_min": min(s["upright"] for s in samples),
        "legs": legs,
        "joint_tracking_error_rms_rad": float(
            np.sqrt(
                np.mean(np.square([s["joint_tracking_error_rad"] for s in samples]))
            )
        ),
        "note": "Descriptive 5ms sampled contact diagnostics; no biological or walking PASS inferred",
    }
