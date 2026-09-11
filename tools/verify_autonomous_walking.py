"""Physical sensory walking probes with preserved trajectories and honest gates."""

import argparse
import hashlib
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.autonomy import SENSOR_MODEL, SensoryWalkingPolicy, autonomous_world
from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.integrity import write_json
from flylab.c.sensors import CSensorAdapter
from flylab.common import to_physics
from flylab.engine import config_values

GATES = {
    "version": "sensory-walking-physical-v1",
    "first_10s_net_min_mm": 5.0,
    "first_10s_signed_forward_min_mm": 5.0,
    "upright_min": 0.5,
    "stuck_window_s": 3.0,
    "stuck_radius_mm": 1.0,
    "arrival_distance_max_mm": 2.0,
    "stopped_path_max_mm": 0.5,
    "resume_2s_net_min_mm": 1.0,
}


def evaluate_trace(arrays, spec):
    """A jittering body against a wall must not pass on path length alone."""
    p, t = arrays["position"][:, :2], arrays["time_s"]
    active = ~np.isin(arrays["state"], ["stopped", "at_food"])
    first = t <= min(10.0, spec["seconds"])
    displacement = np.diff(p, axis=0)
    signed = (displacement * arrays["heading"][:-1, :2]).sum(axis=1)
    first_net = float(np.linalg.norm(p[first][-1] - p[0]))
    first_forward = float(signed[first[1:]].sum())
    stuck = []
    window = round(GATES["stuck_window_s"] / spec["control_dt_s"])
    # Every 50 ms; only fully active 3 s windows count. Intentional resting
    # at food and the explicitly scheduled pause are not locomotion failure.
    for start in range(0, len(p) - window + 1, 10):
        if active[start : start + window].all():
            radius = np.linalg.norm(p[start : start + window] - p[start], axis=1).max()
            if radius < GATES["stuck_radius_mm"]:
                stuck.append(float(t[start]))
    walking = spec["policy"]["parameters"]["task"] == "walk"
    checks = {
        "finite_joints": bool(np.isfinite(arrays["joint_angles"]).all()),
        "upright": bool(arrays["upright"].min() > GATES["upright_min"]),
        "no_stuck_active_window": not stuck,
    }
    result = {
        "first_10s_net_mm": first_net,
        "first_10s_signed_forward_mm": first_forward,
        "stuck_window_start_s": stuck,
        "signed_forward_mm": float(signed.sum()),
    }
    if walking:
        checks["net_progress"] = first_net >= GATES["first_10s_net_min_mm"]
        checks["forward_progress"] = (
            first_forward >= GATES["first_10s_signed_forward_min_mm"]
        )
        support = arrays["support"] > 0.05
        transitions = (np.diff(support.astype(int), axis=0) != 0).sum(axis=0)
        result["support_transitions_per_leg"] = transitions.tolist()
        checks["six_leg_load_cycles"] = bool(np.all(transitions >= 4))
    else:
        target = to_physics(spec["world"]["sources"][0]["p"])[:2]
        checks["arrival"] = bool(
            arrays["state"][-1] == "at_food"
            and np.linalg.norm(p[-1] - target) <= GATES["arrival_distance_max_mm"]
        )
        last = t[1:] > t[-1] - 1.0
        result["last_1s_path_mm"] = float(
            np.linalg.norm(displacement[last], axis=1).sum()
        )
        checks["arrival_stop"] = (
            result["last_1s_path_mm"] <= GATES["stopped_path_max_mm"]
        )
    pause, resume = spec["pause_at_s"], spec["resume_at_s"]
    if 0 <= pause < resume < t[-1] - 2:
        stop = (t[1:] > pause + 0.5) & (t[1:] <= resume - 0.5)
        moved = (t > resume) & (t <= resume + 2)
        result["paused_path_mm"] = float(
            np.linalg.norm(displacement[stop], axis=1).sum()
        )
        result["resumed_2s_net_mm"] = float(np.linalg.norm(p[moved][-1] - p[moved][0]))
        checks["pause"] = result["paused_path_mm"] <= GATES["stopped_path_max_mm"]
        checks["resume"] = result["resumed_2s_net_mm"] >= GATES["resume_2s_net_min_mm"]
    return dict(result, checks=checks, passed=all(checks.values()))


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    world = autonomous_world()
    world["sources"][0]["p"][2] = args.target_side
    options = BodyOptions(
        model=args.model,
        actuation="whole_body" if args.model == "flybody" else "legs",
        servo_profile="tracking_all" if args.model == "flybody" else "asset",
        tendons=args.tendons,
    )
    config = config_values({"sensorNoise": 0.0, "friction": args.friction})
    policy = SensoryWalkingPolicy(
        {"task": args.task, "speed": args.speed, "odor_gain": args.gain}
    )
    spec = {
        "seed": args.seed,
        "seconds": args.seconds,
        "body": asdict(options),
        "config": config,
        "world": world,
        "policy": policy.snapshot(),
        "sensor_model": SENSOR_MODEL,
        "physical_dt_s": 0.0001,
        "control_dt_s": 0.005,
        "push_at_s": args.push_at,
        "pause_at_s": args.pause_at,
        "resume_at_s": args.resume_at,
        "hypothesis": "Sensory policy and existing hybrid controller produce physical walking and odor approach.",
        "scope": "Engineered controller; no neural or biological validation.",
        "acceptance_gates": GATES,
    }
    spec["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path("flylab/autonomy.py"), Path(__file__))
    }
    write_json(out / "spec.json", spec)
    body = None
    rows = []
    began = time.perf_counter()
    error = None
    try:
        body = FlyGymBody(args.seed, world, config, body_options=options)
        if body.tendon_control:
            body.set_body_actuation(
                tendon_inputs={name: 0.2 for name in body.tendon_control.names}
            )
        sensors = CSensorAdapter(args.seed, SENSOR_MODEL)
        for tick in range(round(args.seconds / 0.005)):
            elapsed = tick * 0.005
            if time.perf_counter() - began > 600:
                raise TimeoutError("600 second physical probe wall limit")
            if args.pause_at >= 0 and elapsed >= args.pause_at:
                policy.enabled = args.resume_at >= 0 and elapsed >= args.resume_at
            if args.push_at >= 0 and abs(elapsed - args.push_at) < 1e-7:
                body.perturb(bw=0.5, duration=0.05)
            packet = sensors.observe(body, world, 0.005, config)
            command = policy.step(packet, 0.005)
            body.step(command)
            p, rotation, _velocity = body.pose()
            contact = body.contact_probe()
            rows.append(
                {
                    "time_s": (tick + 1) * 0.005,
                    "position": p.copy(),
                    "upright": rotation[2, 2],
                    "heading": rotation[:, 0].copy(),
                    "joint_angles": body.d.qpos[body.qpos_ids].copy(),
                    "support": np.asarray(contact["floor_normal_bw"]),
                    "slip": np.asarray(contact["slip_speed_mm_s"]),
                    "odor": packet["odor"],
                    "command": [command["forwardSpeed"], command["yawRate"]],
                    "state": policy.last["state"],
                    "contact": packet["contact"],
                    "ranges": packet["nearRanges"],
                }
            )
            if body.fault:
                break
    except Exception as exc:  # noqa: BLE001 - persist the failed physical trajectory.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        result = {
            "error": error,
            "physical_fault": body.fault if body else None,
            "completed": len(rows) == round(args.seconds / 0.005)
            and error is None
            and not (body and body.fault),
            "seconds_executed": len(rows) * 0.005,
            "wall_s": time.perf_counter() - began,
            "neural_control": False,
            "biological_validation": False,
        }
        if rows:
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            np.savez_compressed(out / "trace.npz", **arrays)
            p = arrays["position"]
            target = to_physics(world["sources"][0]["p"])
            distances = np.linalg.norm((p - target)[:, :2], axis=1)
            result.update(
                net_mm=float(np.linalg.norm((p[-1] - p[0])[:2])),
                forward_mm=float((p[-1] - p[0]) @ arrays["heading"][0]),
                path_mm=float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()),
                initial_distance_mm=float(distances[0]),
                final_distance_mm=float(distances[-1]),
                nearest_distance_mm=float(distances.min()),
                min_upright=float(arrays["upright"].min()),
                policy_states={
                    str(s): int(np.count_nonzero(arrays["state"] == s))
                    for s in np.unique(arrays["state"])
                },
                support_fraction=(arrays["support"] > 0.05).mean(axis=0).tolist(),
                final_position=p[-1].tolist(),
                final_heading=arrays["heading"][-1].tolist(),
                model_hash=body.model_hash if body else None,
                evaluation=evaluate_trace(arrays, spec),
            )
        result["acceptance_passed"] = result["completed"] and result.get(
            "evaluation", {}
        ).get("passed", False)
        write_json(out / "report.json", result)
        print(result, flush=True)
        if body:
            body.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=("walk", "odor_seek"), default="odor_seek")
    parser.add_argument(
        "--model", choices=("flybody", "neuromechfly"), default="flybody"
    )
    parser.add_argument(
        "--tendons", choices=("none", "all", "tarsi", "abdomen"), default="all"
    )
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--gain", type=float, default=40.0)
    parser.add_argument("--target-side", type=float, default=-6.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--push-at", type=float, default=-1.0)
    parser.add_argument("--pause-at", type=float, default=-1.0)
    parser.add_argument("--resume-at", type=float, default=-1.0)
    args = parser.parse_args()
    if (
        not 0.005 <= args.seconds <= 60
        or abs(args.seconds / 0.005 - round(args.seconds / 0.005)) > 1e-8
    ):
        parser.error("Use 5 ms periods within 0.005..60 seconds")
    result = run(args)
    raise SystemExit(0 if result["acceptance_passed"] else 1)
