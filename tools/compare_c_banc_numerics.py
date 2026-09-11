"""Fixed-input BANC numerical comparison; explicit experimental dt regridding."""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore


def run(args):
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    saved = StateStore.load(args.checkpoint)["network"]
    weights, _ = rate_weights(graph)
    cases = [
        ("reference", 0.0001),
        ("packed", 0.0001),
        ("packed", 0.0002),
        ("packed", 0.00025),
    ]
    write_json(
        args.out / "spec.json",
        {
            "experiment": "E-PERF-1/E-PERF-2",
            "version": 1,
            "checkpoint": str(args.checkpoint),
            "graph_hash": graph.hash,
            "cases": cases,
            "held_input_s": 0.1,
            "initial_state": "Exact saved rate, adaptation, drive and output_mask; dt candidates explicitly regrid the integer tick to the same model time, using their own identity. This is an experiment, not an implicit checkpoint migration.",
            "criteria": {
                "same_dt": "bit-exact rate and adaptation",
                "different_dt_rms_max": 0.2,
                "different_dt_absolute_max": 2.0,
            },
            "source_hashes": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("flylab/c/adaptive_rate.py"),
                    Path("flylab/c/research_rate.py"),
                )
            },
        },
    )
    records, reference = [], None
    for implementation, dt in cases:
        net = AdaptiveRateNetwork(
            weights,
            *[np.full(graph.n, x, np.float32) for x in (0.02, 1.0, 7.5, 200.0)],
            adaptation_gain=4.0,
            adaptation_tau_s=0.15,
            dt=dt,
            device="cuda",
            capture_steps=10 if dt == 0.0001 else round(0.005 / dt),
            cuda_implementation=implementation,
        )
        state = net.snapshot()
        for key in ("rate", "adaptation", "drive", "output_mask"):
            state[key] = saved[key].copy()
        state["tick"] = round(saved["tick"] * saved["dt"] / dt)
        net.restore(state)
        start = time.perf_counter()
        net.advance(saved["drive"], round(0.1 / dt))
        wall = time.perf_counter() - start
        result = net.snapshot()
        if reference is None:
            reference = result
        errors = {}
        for key in ("rate", "adaptation"):
            difference = result[key].astype(np.float64) - reference[key]
            errors[key] = {
                "rms": float(np.sqrt(np.mean(difference**2))),
                "max": float(np.abs(difference).max()),
                "bit_equal": bool(
                    np.array_equal(
                        result[key].view(np.uint32), reference[key].view(np.uint32)
                    )
                ),
            }
        passed = all(
            e["bit_equal"] if dt == 0.0001 else e["rms"] <= 0.2 and e["max"] <= 2.0
            for e in errors.values()
        )
        row = {
            "implementation": implementation,
            "dt": dt,
            "model_hash": net.identity,
            "wall_s": wall,
            "errors": errors,
            "pass": passed,
        }
        records.append(row)
        write_json(args.out / "results.json", records)
        print(row, flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()


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
            "artifacts/neural-tendons/banc-walking-checkpoints/banc-757d5d9fe381"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
