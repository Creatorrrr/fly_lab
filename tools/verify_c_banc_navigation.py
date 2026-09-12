"""Real BANC target feedback, checkpoint and causal-cut evidence.

Development targets are separate from the fixed six-case acceptance matrix.
This tool never steers through joint targets, changes the root, or resets the
actor during a target trial. Each branch is a separate checkpoint experiment.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import canonical, digest, file_hash, write_json
from flylab.c.navigation_tasks import CASES, CRITERIA, evaluate_navigation
from flylab.c.storage import StateStore
from flylab.common import to_ui
from tools.verify_c_banc_rate_walking import sample


def physical_state_hash(state):
    def pack(value):
        if isinstance(value, np.ndarray):
            return {
                "dtype": value.dtype.str,
                "shape": value.shape,
                "bytes": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return digest(pack({key: value for key, value in state.items() if key != "wall_s"}))


def checkpoint_checks(graph, out, session):
    for _ in range(5):
        session.advance(20)
    state = session.snapshot()
    StateStore.save(out / "checkpoint", state)
    for _ in range(3):
        session.advance(20)
    future = session.snapshot()
    reference_hash = physical_state_hash(future)
    session.close()
    results = []
    for cut in ("intact", "goal_cue", "motor", "circuit"):
        branch = BancWalkingSession.from_checkpoint(
            graph, StateStore.load(out / "checkpoint")
        )
        try:
            if cut == "goal_cue":
                branch.set_navigation_cut(True)
            elif cut != "intact":
                branch.set_cuts({cut: True})
            for _ in range(3):
                branch.advance(20)
            current = branch.snapshot()
            result = {
                "cut": cut,
                "physical_state_hash": physical_state_hash(current),
                "maximum_joint_offset": float(np.max(np.abs(branch.adapter.offset))),
                "maximum_tendon_activation": float(
                    branch.adapter.tendon_activation.max()
                ),
                "maximum_motor_rate": float(branch.rate.max()),
                "maximum_neural_difference": float(
                    np.max(
                        np.abs(current["network"]["rate"] - future["network"]["rate"])
                    )
                ),
                "maximum_physical_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(current["body"]["state"])
                            - np.asarray(future["body"]["state"])
                        )
                    )
                ),
                "cpg_unchanged": not branch.frame()["cpg_advanced"],
            }
            if cut == "intact":
                result["future_exact"] = result["physical_state_hash"] == reference_hash
                if not result["future_exact"]:
                    raise AssertionError("Saved target/physics/neural future diverged")
            elif cut == "motor":
                if (
                    result["maximum_joint_offset"]
                    or result["maximum_tendon_activation"]
                ):
                    raise AssertionError("Motor cut left a neural actuator request")
            elif cut == "goal_cue":
                if (
                    not result["maximum_neural_difference"]
                    or not result["maximum_physical_difference"]
                ):
                    raise AssertionError(
                        "Goal cue had no causal physical effect in this branch"
                    )
            elif result["maximum_motor_rate"] > 0.001:
                raise AssertionError(
                    "Motor activity persists without synaptic transmission"
                )
            results.append(result)
            write_json(out / "checks.json", results)
            print(json.dumps(result), flush=True)
        finally:
            branch.close()
    invalid = copy.deepcopy(state)
    invalid["navigation"]["history"][-1][0] += 0.005
    try:
        branch = BancWalkingSession.from_checkpoint(graph, invalid)
    except ValueError as exc:
        results.append({"malformed_history_rejected": True, "error": str(exc)})
    else:
        branch.close()
        raise AssertionError("Malformed physical target clock accepted")
    write_json(out / "checks.json", results)


def restart_check(graph, out, saved_path):
    session = BancWalkingSession.from_checkpoint(graph, StateStore.load(saved_path))
    try:
        if not session.frame()["navigation"]["physical_hold_confirmed"]:
            raise ValueError(
                "Restart check requires an observed stopped target session"
            )
        before = session.snapshot()
        position, rotation, _ = session.body.pose()
        forward = np.asarray(to_ui(rotation[:, 0]))[[0, 2]]
        forward /= np.linalg.norm(forward)
        target = np.asarray(to_ui(position))[[0, 2]] + 8.0 * forward
        session.set_navigation_target(target.tolist())
        after = session.snapshot()
        preserved = ("body", "network", "adapter", "control_tick")
        if physical_state_hash(
            {k: before[k] for k in preserved}
        ) != physical_state_hash({k: after[k] for k in preserved}):
            raise AssertionError("Changing target reset the neural/body state")
        trace = [sample(session.body, session.control_tick)]
        for _ in range(400):
            session.advance(1)
            trace.append(sample(session.body, session.control_tick))
            if session.fault or trace[-1]["contact"]:
                break
        positions = np.asarray([r["position"] for r in trace])[:, [0, 2]]
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        healthy = not any(r["fault"] or r["contact"] for r in trace)
        result = {
            "source_checkpoint": str(saved_path),
            "target_xz_mm": target.tolist(),
            "preserved_body_neural_adapter_on_target_change": True,
            "elapsed_s": trace[-1]["simTime"] - trace[0]["simTime"],
            "net_mm": net,
            "healthy": healthy,
            "task_status": "PASS"
            if len(trace) == 401 and healthy and net >= 1.0
            else "FAIL",
            "cpg_unchanged": not session.frame()["cpg_advanced"],
        }
        with (out / "restart_trace.jsonl").open("wb") as stream:
            for row in trace:
                stream.write(canonical(row) + b"\n")
        write_json(out / "restart_result.json", result)
        if not session.fault:
            StateStore.save(out / "restart_final", session.snapshot())
        print(json.dumps(result), flush=True)
    finally:
        session.close()


def trial(graph, out, name, seed, offset, args):
    out.mkdir()
    session = BancWalkingSession(
        graph,
        seed=seed,
        device="cuda",
        parameters={"coxa_geometry": True, "ltm_tendons": True},
        navigation={"target_xz_mm": [0.0, 0.0], "forward_drive": 5000.0},
    )
    try:
        initial = to_ui(session.body.pose()[0])
        target = [float(initial[0] + offset[0]), float(initial[2] + offset[1])]
        session.set_navigation_target(target)
        identity = {
            "name": name,
            "seed": seed,
            "offset_xz_mm": list(offset),
            "target_xz_mm": target,
            "initial_position": initial,
            "encoder_hash": session.navigation.encoder.identity,
            "adapter_hash": session.adapter.identity,
            "network_hash": session.network.identity,
        }
        write_json(out / "identity.json", identity)
        if args.checks_only:
            checkpoint_checks(graph, out, session)
            return {"name": name, "checks": "PASS", "navigation": "NOT_TESTED"}
        StateStore.save(out / "initial", session.snapshot())
        trace = [sample(session.body, 0)]
        began, error = time.perf_counter(), None
        with (out / "trace.jsonl").open("wb") as stream:
            stream.write(canonical(trace[0]) + b"\n")
            for _ in range(round(args.seconds / 0.005)):
                if time.perf_counter() - began > args.wall_limit:
                    error = "Declared wall limit reached"
                    break
                session.advance(1)
                row = sample(session.body, session.control_tick)
                row["navigation"] = session.navigation.view(*session.body.pose()[:2])
                stream.write(canonical(row) + b"\n")
                trace.append(row)
                if session.fault or row["contact"]:
                    break
                if session.control_tick % 200 == 0:
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "time_s": row["simTime"],
                                "distance_mm": row["navigation"]["distance_mm"],
                                "bearing_rad": row["navigation"]["bearing_rad"],
                                "turn": row["navigation"]["turn_request"],
                                "brake": row["navigation"]["brake_request"],
                            }
                        ),
                        flush=True,
                    )
                if row["navigation"]["physical_hold_confirmed"]:
                    break
        verdict = evaluate_navigation(trace, target)
        verdict.update(
            name=name,
            wall_seconds=time.perf_counter() - began,
            runtime_error=error,
            cpg_unchanged=not session.frame()["cpg_advanced"],
            imposed_gait=False,
            root_correction=False,
            final_navigation=session.frame()["navigation"],
        )
        write_json(out / "result.json", verdict)
        if not session.fault:
            StateStore.save(out / "final", session.snapshot())
        print(json.dumps(verdict), flush=True)
        return verdict
    finally:
        session.close()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "verify_source.py")
    cases = CASES if args.matrix else [("development", args.seed, args.target)]
    write_json(
        args.out / "spec.json",
        {
            "experiment": args.experiment,
            "cases": cases,
            "seconds": args.seconds,
            "checks_only": args.checks_only,
            "restart_from": str(args.restart_from) if args.restart_from else None,
            "wall_limit_per_case": args.wall_limit,
            "criteria": CRITERIA,
            "criteria_hash": digest(CRITERIA),
            "prediction": "Target feedback closes physical bearing/range and holds the arrival region; fixed physical evaluator decides",
            "adoption": "Experimental until complete fixed acceptance criteria pass",
            "sources": {
                str(p): file_hash(p)
                for p in [
                    Path(__file__),
                    *[
                        Path("flylab/c") / (name + ".py")
                        for name in (
                            "banc_walking",
                            "rate_body",
                            "target_sense",
                            "target_neural",
                            "target_navigation",
                            "navigation_tasks",
                            "adaptive_rate",
                        )
                    ],
                    Path("flylab/body.py"),
                ]
            },
        },
    )
    graph = GraphStore.load(args.graph)
    if args.restart_from:
        restart_check(graph, args.out, args.restart_from)
        return
    results = []
    for name, seed, offset in cases:
        results.append(trial(graph, args.out / name, name, seed, offset, args))
        write_json(args.out / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--target", type=float, nargs=2, default=(6.0, -3.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--wall-limit", type=float, default=240.0)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--matrix", action="store_true")
    arguments = parser.parse_args()
    if not 0 < arguments.seconds <= 31.5 or not 0 < arguments.wall_limit <= 300:
        parser.error("Bounded trial seconds/wall limit required")
    if arguments.matrix and (arguments.checks_only or arguments.seconds != 31.5):
        parser.error("Acceptance matrix requires the full fixed 31.5s opportunity")
    if arguments.restart_from and (arguments.matrix or arguments.checks_only):
        parser.error("Restart verification is a separate physical experiment")
    run(arguments)
