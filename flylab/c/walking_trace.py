"""Read-only 200 Hz BANC instrumentation, independent of neural dynamics.

Records measured states, not interpolated poses. No stepping, forward dynamics,
pose edits, or writes to mjData are allowed here. Contact slip uses both bodies'
velocities at the actual contact point, not the origin of the last tarsus.
"""

import copy
import math
import uuid
from collections import deque

import numpy as np

from ..body import LEGS, LINKS
from ..common import S, to_ui
from .integrity import bounded_int

TRACE_SCHEMA = "flylab.banc-trace.v1"


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
    }
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

    def append(self, sample):
        tick = bounded_int(sample.get("control_tick"), "trace tick", 0, 10**10)
        if sample.get("time_s") != tick * 0.005:
            raise ValueError("Trace clock mismatch")
        if self.samples and tick <= self.samples[-1]["control_tick"]:
            raise ValueError(
                "Trace samples must advance strictly; restore starts a new epoch"
            )
        self.evicted += int(len(self.samples) == self.capacity)
        self.samples.append(copy.deepcopy(sample))

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
