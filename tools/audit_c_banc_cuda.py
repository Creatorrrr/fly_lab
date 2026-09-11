"""Isolated CUDA audit worker; never changes the live BANC implementation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from flylab.gpu_profile import nvtx_range


def run(args):
    import cupy as cp
    import torch
    import torch.cuda.profiler

    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    saved = StateStore.load(args.checkpoint)
    session = BancWalkingSession.from_checkpoint(graph, saved)
    try:
        session.advance(2)
        net = session.network
        tensor_names = (
            "rate",
            "adaptation",
            "drive",
            "output_mask",
            "weights",
            "tau",
            "a",
            "threshold",
            "cap",
            "coefficient",
            "_rowptr",
            "_columns",
            "_weight_values",
        )
        buffers = {
            name: {
                "device": str(getattr(net, name).device),
                "dtype": str(getattr(net, name).dtype),
                "elements": getattr(net, name).numel(),
            }
            for name in tensor_names
        }
        lengths = np.diff(graph.indptr)
        rates = net.readout()
        row = {
            "scope": "Isolated saved-state worker, not live server; profiled timings are diagnostic only",
            "kind": args.kind,
            "checkpoint": str(args.checkpoint),
            "graph_hash": graph.hash,
            "model_hash": net.identity,
            "neurons": graph.n,
            "edges": len(graph.indices),
            "neural_dt_s": net.dt,
            "capture_steps": net.interval_steps,
            "cuda_graph_present": net.graph is not None,
            "implementation": net.cuda_implementation,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cupy": cp.__version__,
            "buffers": buffers,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "row_length_percentiles": np.percentile(
                lengths, [0, 25, 50, 75, 90, 99, 100]
            ).tolist(),
            "zero_rows": int(np.count_nonzero(lengths == 0)),
            "exact_zero_rates": int(np.count_nonzero(rates == 0)),
            "rates_above_one": int(np.count_nonzero(rates > 1)),
            "kernel_launch": {
                "grid": [(graph.n + 7) // 8, 1, 1],
                "block": [256, 1, 1],
                "attributes": net._csr_kernel.attributes,
            },
            "sources": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("flylab/c/adaptive_rate.py"),
                    Path("flylab/c/research_rate.py"),
                    Path("flylab/c/banc_walking.py"),
                )
            },
        }
        write_json(args.out / "metadata.json", row)

        def annotate(obj, name, label):
            original = getattr(obj, name)

            def wrapped(*a, **kw):
                with nvtx_range(label):
                    return original(*a, **kw)

            setattr(obj, name, wrapped)

        for obj, name, label in (
            (session.adapter, "encode", "sensory"),
            (net, "advance", "neural"),
            (net, "readout", "readout"),
            (session.adapter, "decode", "motor"),
            (session.body, "step_joint_targets", "physics"),
            (session, "frame", "frame"),
            (session, "_clocks", "clocks"),
        ):
            annotate(obj, name, label)
        torch.cuda.synchronize()
        with torch.cuda.profiler.profile():
            if args.kind == "timeline":
                with nvtx_range("audit_control"):
                    session.advance(1)
                    torch.cuda.synchronize()
            else:
                with nvtx_range("audit_matvec"):
                    result = net._current(net.rate)
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(result).all()):
                        raise AssertionError("Invalid current")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "kind",
                        "neurons",
                        "edges",
                        "cuda_graph_present",
                        "capture_steps",
                        "implementation",
                        "torch_allocated_bytes",
                        "kernel_launch",
                    )
                }
            ),
            flush=True,
        )
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
            "artifacts/neural-tendons/banc-walking-checkpoints/banc-e8d36e000dfe"
        ),
    )
    parser.add_argument("--kind", choices=("timeline", "matvec"), default="timeline")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
