"""Compare production CUDA against the frozen 2026-09-12 packed baseline."""

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from tools.banc_cuda_reference import LegacyPackedNetwork, UnfusedPackedNetwork
from tools.benchmark_c_banc_session import reset_session


def run(args):
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    saved = StateStore.load(args.checkpoint)
    graph = GraphStore.load(args.graph)
    write_json(
        args.out / "spec.json",
        {
            "scope": "Frozen packed baseline vs current production CUDA; same full BANC graph, saved RK4 timestep, sensory/physical coupling and checkpoint",
            "change": "Per-stage pre-mask; unchanged Torch pointwise derivatives and fixed lane sum",
            "baseline": args.baseline,
            "repeats": 5,
            "model_s_per_repeat": 0.15,
            "warmup_s": 0.05,
            "order": "baseline,candidate alternating/reversed",
            "checkpoint": str(args.checkpoint),
            "decision": "Require bit-exact neural/body/adapter future and repeatability; separately validate long-run walking and causal cuts.",
            "sources": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("tools/banc_cuda_reference.py"),
                    Path("flylab/c/adaptive_rate.py"),
                    Path("flylab/c/banc_walking.py"),
                )
            },
        },
    )
    baseline_class = (
        LegacyPackedNetwork if args.baseline == "legacy" else UnfusedPackedNetwork
    )
    with patch("flylab.c.banc_walking.AdaptiveRateNetwork", baseline_class):
        baseline = BancWalkingSession.from_checkpoint(graph, saved)
    candidate = None
    try:
        candidate = BancWalkingSession.from_checkpoint(graph, saved)
        sessions, start = (
            [baseline, candidate],
            [baseline.snapshot(), candidate.snapshot()],
        )
        walls, ends, records = [[], []], [None, None], []
        for repeat in range(5):
            for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                session = sessions[index]
                reset_session(session, start[index])
                session.advance(10)
                reset_session(session, start[index])
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _ in range(3):
                    session.advance(10)
                elapsed = time.perf_counter() - began
                walls[index].append(elapsed)
                current = session.snapshot()
                if session.fault or session.body.fault:
                    raise AssertionError("Measured session faulted")
                if ends[index] is not None:
                    compare_states(ends[index], current)
                ends[index] = current
                records.append(
                    {"repeat": repeat, "candidate": bool(index), "wall_s": elapsed}
                )
                write_json(args.out / "runs.json", records)
                print(records[-1], flush=True)
        errors = compare_states(ends[0], ends[1])
        medians = [float(np.median(v)) for v in walls]
        report = {
            "wall_samples": walls,
            "median_wall_s": medians,
            "speedup": medians[0] / medians[1],
            "future_errors": errors,
            "exact_future": True,
            "repeat_futures_bit_exact": True,
            "same_model_hash": baseline.network.identity == candidate.network.identity,
            "model_hash": baseline.network.identity,
            "graph_hash": graph.hash,
            "candidate": "production AdaptiveRateNetwork",
            "baseline": args.baseline,
        }
        write_json(args.out / "result.json", report)
        print(report, flush=True)
    finally:
        baseline.close()
        if candidate is not None:
            candidate.close()


def compare_states(left, right):
    errors = {}
    for section, keys in (
        ("network", ("rate", "adaptation", "drive", "output_mask")),
        ("adapter", ("filtered", "offset", "velocity_lowpass", "last_features")),
        ("body", ("state",)),
    ):
        for key in keys:
            a, b = np.asarray(left[section][key]), np.asarray(right[section][key])
            np.testing.assert_equal(a.dtype, b.dtype)
            np.testing.assert_equal(a.shape, b.shape)
            np.testing.assert_array_equal(
                a.view(np.uint8), b.view(np.uint8), err_msg=section + ":" + key
            )
            errors[section + ":" + key] = float(np.abs(a - b).max())
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/neural-tendons/banc-walking-checkpoints/banc-e8d36e000dfe"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", choices=("legacy", "premask"), default="legacy")
    run(parser.parse_args())
