"""Bounded predictive neural-input experiment; actor state is never rewound.

Only a separate forecast world is restored between candidates. The actor
executes the selected neural current through its own continuous dynamics.
This is an external engineering planner, not BANC goal computation.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import canonical, digest, file_hash, write_json
from flylab.c.target_neural import LEFT, RIGHT
from tools.probe_c_banc_navigation import measure, restore
from tools.verify_c_banc_rate_walking import sample


def action_drive(session, action):
    enc = session.navigation.encoder
    drive = np.zeros(session.graph.n, np.float32)
    drive[enc.ports] = -5000.0
    drive[enc.descending] = 5000.0
    selected = session.graph.resolve(["flywire:banc:888:" + root for root in action])
    drive[selected] += 10000.0
    return drive


def row(session):
    result = sample(session.body, session.control_tick)
    result["upright"] = float(session.body.pose()[1][2, 2])
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    params = {"coxa_geometry": True, "ltm_tendons": True}
    config = {"target_xz_mm": [6.0, -3.0], "forward_drive": 5000.0}
    actions = {
        "quiet": (),
        "lf": tuple(LEFT[i] for i in (1, 4, 7)),
        "lf-rm": tuple(LEFT[i] for i in (1, 4, 7)) + tuple(RIGHT[i] for i in (2, 5, 8)),
        "lm": tuple(LEFT[i] for i in (2, 5, 8)),
        "right": RIGHT,
    }
    write_json(
        args.out / "spec.json",
        {
            "experiment": "NAV-E20",
            "seconds_per_direction": args.seconds,
            "prediction_horizon_s": 0.25,
            "decision_interval_s": 0.25,
            "directions": ["left", "right"],
            "actions": actions,
            "wall_limit_per_direction_s": 180.0,
            "seed": 42,
            "hypothesis": "State-dependent selection among existing neural inputs can control yaw when fixed input cannot",
            "criterion": "Healthy opposite signed yaw>=0.25rad over2s; actor/forecast physical future must match exactly",
            "claim_boundary": "External engineering neural-input planner, not BANC intrinsic target computation",
            "graph_hash": graph.hash,
            "parameters": params,
            "navigation": config,
            "source_hashes": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("flylab/c/banc_walking.py"),
                    Path("flylab/c/target_neural.py"),
                    Path("flylab/c/rate_body.py"),
                )
            },
        },
    )
    actor = BancWalkingSession(
        graph, parameters=params, navigation=config, device="cuda"
    )
    forecast = BancWalkingSession(
        graph, parameters=params, navigation=config, device="cuda"
    )
    initial = actor.snapshot()
    report = []
    try:
        for direction, sign in (("left", -1.0), ("right", 1.0)):
            # A fresh independent trial is allowed; never restore actor within it.
            restore(actor, initial)
            actor.navigation.restore(initial["navigation"], time_s=0.0)
            trace, decisions = [row(actor)], []
            began = time.perf_counter()
            for decision in range(round(args.seconds / 0.25)):
                if time.perf_counter() - began > 180:
                    raise TimeoutError("Predictive assay direction wall limit")
                start = actor.snapshot()
                start_yaw = row(actor)["yaw"]
                candidates = []
                for name, roots in actions.items():
                    restore(forecast, start)
                    forecast.navigation.restore(
                        start["navigation"], time_s=actor.control_tick * 0.005
                    )
                    current = action_drive(forecast, roots)
                    forecast.navigation.drive = lambda _p, _r, held=current: held
                    for count in (20, 20, 10):
                        forecast.advance(count)
                        if forecast.fault:
                            break
                    change = math.atan2(
                        math.sin(row(forecast)["yaw"] - start_yaw),
                        math.cos(row(forecast)["yaw"] - start_yaw),
                    )
                    healthy = not forecast.fault and all(
                        r[3] for r in forecast.navigation.history[-50:]
                    )
                    score = -sign * change if healthy else math.inf
                    candidates.append((score, name, roots, forecast.body.d.qpos.copy()))
                selected = min(candidates, key=lambda x: x[0])
                if not math.isfinite(selected[0]):
                    raise RuntimeError("No healthy predicted neural action")
                current = action_drive(actor, selected[2])
                actor.navigation.drive = lambda _p, _r, held=current: held
                for _ in range(50):
                    actor.advance(1)
                    trace.append(row(actor))
                    if actor.fault:
                        break
                equal = bool(np.array_equal(actor.body.d.qpos, selected[3]))
                record = {
                    "time_s": actor.control_tick * 0.005,
                    "selected": selected[1],
                    "candidate_scores": {
                        c[1]: c[0] if math.isfinite(c[0]) else None for c in candidates
                    },
                    "forecast_actor_exact": equal,
                    "yaw": row(actor)["yaw"],
                }
                decisions.append(record)
                print(json.dumps({"direction": direction, **record}), flush=True)
                if not equal:
                    raise RuntimeError(
                        "Actor execution differs from selected physical prediction"
                    )
                if actor.fault:
                    break
            result = {
                "direction": direction,
                "measurements": measure(trace),
                "decisions": decisions,
                "wall_seconds": time.perf_counter() - began,
                "cpg_unchanged": digest(actor.body.snapshot()["cpg"]) == actor.cpg_hash,
                "actor_restores_during_trial": 0,
                "validity": "valid",
                "milestone": "pending",
            }
            (args.out / (direction + ".jsonl")).write_bytes(
                b"".join(canonical(r) + b"\n" for r in trace)
            )
            report.append(result)
            write_json(args.out / "results.json", report)
    finally:
        actor.close()
        forecast.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    run(parser.parse_args())
