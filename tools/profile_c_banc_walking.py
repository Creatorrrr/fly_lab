"""Bounded stage timings for the actual BANC sensory walking session."""

import argparse
import base64
import io
import json
import sys
import time
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
    saved = StateStore.load(args.checkpoint)
    session = BancWalkingSession.from_checkpoint(graph, saved)
    import torch
    from PIL import Image

    stages = {}

    def wrap(obj, name, label):
        original = getattr(obj, name)

        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                stages[label] = stages.get(label, 0.0) + time.perf_counter() - start

        setattr(obj, name, timed)

    try:
        session.advance(10)
        session.body.preview()
        for obj, name, label in (
            (session.adapter, "encode", "sensory_encoding"),
            (session.network, "advance", "neural_advance"),
            (session.network, "readout", "motor_readout"),
            (session.adapter, "decode", "motor_decoding"),
            (session.body, "step_joint_targets", "physical_advance"),
            (session, "frame", "frame"),
            (session, "_clocks", "clock_checks"),
        ):
            wrap(obj, name, label)
        walls = []
        for _ in range(3):
            begin = time.perf_counter()
            session.advance(10)
            walls.append(time.perf_counter() - begin)
        previews, encodings = [], []
        for _ in range(5):
            begin = time.perf_counter()
            pixels = session.body.preview()
            previews.append(time.perf_counter() - begin)
            begin = time.perf_counter()
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            json.dumps({"image": base64.b64encode(stream.getvalue()).decode("ascii")})
            encodings.append(time.perf_counter() - begin)
        stages["other"] = sum(walls) - sum(stages.values())
        lengths = np.diff(graph.indptr)
        report = {
            "scope": "Separate session; three sequential 50ms controls after warmup; excludes setup and browser/network time",
            "checkpoint": str(args.checkpoint),
            "graph_hash": graph.hash,
            "model_hash": session.network.identity,
            "parameters": saved["parameters"],
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "model_seconds": 0.15,
            "wall_seconds": sum(walls),
            "repeats_wall_seconds": walls,
            "stage_seconds": stages,
            "stage_percent": {k: v / sum(walls) * 100 for k, v in stages.items()},
            "preview_median_seconds": float(np.median(previews)),
            "png_json_median_seconds": float(np.median(encodings)),
            "edges": len(graph.indices),
            "row_length_percentiles": np.percentile(
                lengths, [0, 50, 90, 99, 100]
            ).tolist(),
            "neural_dt": session.network.dt,
            "matvec_per_model_second": 4 / session.network.dt,
            "sources": {
                str(p): file_hash(p)
                for p in (Path(__file__), Path("flylab/c/adaptive_rate.py"))
            },
        }
        write_json(args.out / "profile.json", report)
        print(json.dumps(report, indent=2))
    finally:
        session.close()


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
