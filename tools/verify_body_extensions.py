"""Bounded physical tendon responses, free walking A/B, and shared contacts."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.integrity import write_json
from flylab.engine import config_values
from flylab.sensors import default_world


def body(tendons, attachment="tethered", seed=42):
    return FlyGymBody(
        seed,
        default_world(),
        config_values(),
        body_options=BodyOptions(
            model="flybody",
            actuation="whole_body",
            servo_profile="tracking_all",
            attachment=attachment,
            tendons=tendons,
        ),
    )


def run(out, seconds=2.0):
    from PIL import Image

    from flylab.shared_arena import SharedFlyArena

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        "schema": "flylab.body-extensions-verification.v1",
        "tendon_pulse_input": 0.2,
        "pulse_seconds": 0.1,
        "minimum_signed_tendon_response": 1e-6,
        "tether_root_drift_limit_mm": 1e-12,
        "free_seconds": seconds,
        "seeds": [42, 43],
        "walking_input": {"forwardSpeed": 2.0, "yawRate": 0.0},
        "walking_tendon_input": 0.2,
        "physical_dt_s": 0.0001,
        "comparisons": ["none", "tarsi", "abdomen", "all"],
        "interpretation": "Mechanical response and engineering gait comparison; no neural tendon mapping or biological claim",
        "wall_limit_s": 600,
    }
    write_json(out / "spec.json", spec)
    report = {
        "status": "RUNNING",
        "structural": [],
        "responses": [],
        "walking": [],
        "shared": [],
        "biological_validation": False,
    }
    started = time.perf_counter()

    def save():
        write_json(out / "report.json", report)

    try:
        for option in spec["comparisons"]:
            b = body(option)
            try:
                initial = b.snapshot()
                root = b.pose()[0].copy()
                ctl = b.tendon_control
                report["structural"].append(
                    {
                        "tendons": option,
                        "position_servos": len(b.whole_body_control.names),
                        "passive_joints": len(b.whole_body_control.passive_ids),
                        "tendon_inputs": len(ctl.names) if ctl else 0,
                        "free_joints": int(
                            np.sum(b.m.jnt_type == b.mj.mjtJoint.mjJNT_FREE)
                        ),
                        "root_fixed": b.whole_body_observation()["root_fixed"],
                        "model_hash": b.model_hash,
                    }
                )
                names = ctl.names if ctl else ["c_thorax-c_head-yaw"]
                for name in names:
                    samples = []
                    for sign in (-1, 1):
                        b.restore(initial)
                        command = (
                            {"tendon_inputs": {name: sign * 0.2}}
                            if ctl
                            else {"targets": {name: sign * 0.1}}
                        )
                        for _ in range(20):
                            b.step_body_targets(**command)
                        value = (
                            float(b.d.ten_length[ctl.tendon_ids[ctl.index[name]]])
                            if ctl
                            else float(
                                b.d.qpos[
                                    b.whole_body_control.qpos_ids[
                                        b.whole_body_control.index[name]
                                    ]
                                ]
                            )
                        )
                        samples.append(
                            {
                                "sign": sign,
                                "value": value,
                                "root_drift_mm": float(
                                    np.linalg.norm(b.pose()[0] - root)
                                ),
                                "warnings": b.d.warning.number.tolist(),
                                "fault": b.fault,
                            }
                        )
                    response = samples[1]["value"] - samples[0]["value"]
                    saved = b.snapshot()
                    b.step_body_targets(**command)
                    expected = b.snapshot()
                    b.restore(saved)
                    b.step_body_targets(**command)
                    row = {
                        "profile": option,
                        "name": name,
                        "samples": samples,
                        "signed_response": response,
                        "cpu_future_restored": (expected == b.snapshot()),
                        "passed": bool(
                            response > spec["minimum_signed_tendon_response"]
                            and all(
                                s["root_drift_mm"] <= spec["tether_root_drift_limit_mm"]
                                and not s["fault"]
                                and not any(s["warnings"])
                                for s in samples
                            )
                        ),
                    }
                    report["responses"].append(row)
                Image.fromarray(b.preview()).save(out / f"tethered-{option}.png")
                write_json(out / f"inventory-{option}.json", b.whole_body_observation())
            finally:
                b.close()
            save()
        for seed in spec["seeds"]:
            for option in spec["comparisons"]:
                if time.perf_counter() - started > spec["wall_limit_s"]:
                    raise TimeoutError("Verification wall limit")
                b = body(option, "free", seed)
                try:
                    if b.tendon_control:
                        b.set_body_actuation(
                            tendon_inputs={n: 0.2 for n in b.tendon_control.names}
                        )
                    p0, R0, _ = b.pose()
                    trace = []
                    slip_integral = 0.0
                    contact_samples = 0
                    for _ in range(round(seconds / 0.005)):
                        b.step(spec["walking_input"])
                        p, _R, _ = b.pose()
                        contact = np.asarray(b.frame()[1]["contactsBW"]) > 0.05
                        speed = []
                        for bid in b._tip_ids:
                            v = np.zeros(6)
                            b.mj.mj_objectVelocity(
                                b.m, b.d, b.mj.mjtObj.mjOBJ_BODY, int(bid), v, 0
                            )
                            speed.append(float(np.linalg.norm(v[3:5])))
                        slip_integral += (
                            float(np.sum(np.asarray(speed)[contact])) * 0.005
                        )
                        contact_samples += int(np.sum(contact))
                        trace.append(
                            {
                                "time_s": b.physics_time(),
                                "position_mm": p.tolist(),
                                "contacts": contact.tolist(),
                                "tip_speed_mm_s": speed,
                                "joint_angles": b.d.qpos.copy().tolist(),
                                "fault": b.fault,
                            }
                        )
                        if b.fault or any(b.d.warning.number):
                            break
                    displacement = b.pose()[0] - p0
                    row = {
                        "seed": seed,
                        "tendons": option,
                        "steps": len(trace),
                        "complete": len(trace) == round(seconds / 0.005),
                        "fault": b.fault,
                        "warnings": b.d.warning.number.tolist(),
                        "forward_mm": float(displacement @ R0[:, 0]),
                        "displacement_mm": float(np.linalg.norm(displacement[:2])),
                        "loaded_tip_travel_mm": slip_integral,
                        "loaded_samples": contact_samples,
                        "mean_loaded_tip_speed_mm_s": slip_integral
                        / (contact_samples * 0.005)
                        if contact_samples
                        else None,
                        "metric_note": "Horizontal distal segment speed while foot load >0.05 BW; rolling/contact changes also contribute",
                    }
                    report["walking"].append(row)
                    write_json(out / f"walking-{seed}-{option}.json", trace)
                finally:
                    b.close()
                save()
        for collisions in (False, True):
            arena = SharedFlyArena(spacing_mm=3.0, collisions=collisions)
            try:
                arena.set_drives({n: [0.6, 0.6] for n in arena.names})
                trace = []
                for _ in range(100):
                    trace.append(arena.advance(1))
                saved = arena.snapshot()
                arena.advance(2)
                future = arena.snapshot()
                arena.restore(saved)
                arena.advance(2)
                row = {
                    "collisions": collisions,
                    "steps": 100,
                    "contact_count": sum(len(r["interfly_contacts"]) for r in trace),
                    "cpu_future_restored": future == arena.snapshot(),
                    "finite": bool(np.isfinite(arena.d.qpos).all()),
                    "fault": arena.fault,
                }
                report["shared"].append(row)
                write_json(out / f"shared-{collisions}.json", trace)
                Image.fromarray(arena.preview()).save(out / f"shared-{collisions}.png")
            finally:
                arena.close()
        report.update(
            status="COMPLETE",
            wall_s=time.perf_counter() - started,
            mechanical_response_pass=all(
                r["passed"] and r["cpu_future_restored"] for r in report["responses"]
            ),
            gait_improvement_validated=False,
            adoption="Optional experimental profiles; default unchanged. Free gait outcomes reported per condition.",
        )
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=2.0)
    a = p.parse_args()
    if (
        not 0.1 <= a.seconds <= 5
        or abs(a.seconds / 0.005 - round(a.seconds / 0.005)) > 1e-9
    ):
        p.error("Use 0.1..5 seconds in 0.005s increments")
    run(a.out, a.seconds)
