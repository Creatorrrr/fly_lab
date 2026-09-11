"""Bounded real-body checkpoint and matched causal-cut verification."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    write_json(
        args.out / "spec.json",
        {
            "schema": "flylab.banc-session-check.v1",
            "warmup_s": 1.0,
            "branch_s": 0.3,
            "branches": ["intact", "sensory", "descending", "circuit", "motor"],
            "future_tolerance": 1e-8,
            "same_start_checkpoint": True,
            "decision": "Checkpoint future must match; circuit and motor cuts must remove their downstream contribution. Sensory/descending effects are measured, not assumed. This is not a walking task pass.",
            "graph_hash": graph.hash,
            "neural_dt_s": args.neural_dt,
            "cuda_implementation": args.cuda_implementation,
            "sources": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("flylab/c/banc_walking.py"),
                    Path("flylab/c/rate_body.py"),
                    Path("flylab/c/adaptive_rate.py"),
                )
            },
        },
    )
    original = BancWalkingSession(
        graph,
        parameters={"neural_dt_s": args.neural_dt},
        cuda_implementation=args.cuda_implementation,
    )
    try:
        for _ in range(10):
            original.advance(20)
        state = original.snapshot()
        StateStore.save(args.out / "start", state)
        original.advance(20)
        original.advance(20)
        original.advance(20)
        future = original.snapshot()
        future_qpos = original.body.d.qpos.copy()
    finally:
        original.close()
    results = []
    reference = None
    for cut in ("intact", "sensory", "descending", "circuit", "motor"):
        session = BancWalkingSession.from_checkpoint(
            graph,
            StateStore.load(args.out / "start"),
            cuda_implementation=args.cuda_implementation,
        )
        try:
            if cut != "intact":
                session.set_cuts({cut: True})
            for _ in range(3):
                session.advance(20)
            current = session.snapshot()
            frame = session.frame()
            if cut == "intact":
                reference = current
                reference_qpos = session.body.d.qpos.copy()
                errors = {
                    key: float(
                        np.max(np.abs(current["network"][key] - future["network"][key]))
                    )
                    for key in ("rate", "adaptation", "drive", "output_mask")
                }
                errors["body_qpos"] = float(
                    np.max(np.abs(session.body.d.qpos - future_qpos))
                )
                errors["body_integration_state"] = float(
                    np.max(
                        np.abs(
                            np.asarray(current["body"]["state"])
                            - np.asarray(future["body"]["state"])
                        )
                    )
                )
                for key in ("filtered", "offset", "velocity_lowpass", "last_features"):
                    errors["adapter_" + key] = float(
                        np.max(np.abs(current["adapter"][key] - future["adapter"][key]))
                    )
                if max(errors.values()) > 1e-8:
                    raise AssertionError("Checkpoint future diverged: " + str(errors))
            else:
                errors = {}
            row = {
                "cut": cut,
                "future_errors": errors,
                "motor_rate_max": float(session.rate.max()),
                "motor_rate_difference": float(
                    np.max(
                        np.abs(
                            session.rate
                            - reference["network"]["rate"][session.adapter.motor_ids]
                        )
                    )
                ),
                "offset_max_rad": float(np.max(np.abs(session.adapter.offset))),
                "joint_difference_rad": float(
                    np.max(
                        np.abs(
                            session.body.d.qpos[session.body.qpos_ids]
                            - reference_qpos[session.body.qpos_ids]
                        )
                    )
                ),
                "frame": frame,
            }
            if cut == "circuit" and row["motor_rate_max"] > 0.001:
                raise AssertionError(
                    "Motor activity persists without any synaptic transmission"
                )
            if cut == "motor" and row["offset_max_rad"] != 0.0:
                raise AssertionError("Disconnected motor signal reached joint targets")
            results.append(row)
            write_json(args.out / "results.json", results)
            print(
                {key: value for key, value in row.items() if key != "frame"}, flush=True
            )
            if cut == "intact":
                from PIL import Image

                Image.fromarray(session.body.preview()).save(args.out / "native.png")
            StateStore.save(args.out / cut, current)
        finally:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--neural-dt", type=float, choices=(0.0001, 0.0002, 0.00025), default=0.0001
    )
    parser.add_argument(
        "--cuda-implementation", choices=("reference", "packed"), default="reference"
    )
    run(parser.parse_args())
