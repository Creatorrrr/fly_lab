"""Bounded, bit-exact CUDA row scheduling probe; no model parameter changes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.adaptive_rate import _PACKED_CSR_CURRENT
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore


def run(args):
    import cupy as cp
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    variants = (
        ("original", "shared_lookup")
        if args.probe == "shared"
        else (
            ("original", "degree_schedule", "degree_repack")
            if args.probe == "schedule"
            else ("original", "restrict", "unroll4", "unroll8", "stream_entries")
        )
    )
    write_json(
        args.out / "spec.json",
        {
            "experiment": "OBS-E2-" + args.probe.upper(),
            "hypothesis": "Degree scheduling reduces tail imbalance, explicit read-only pointers/unrolling overlap loads, or the observed 815-entry weight lookup fits shared memory and reduces dependent lookup latency; within-row addition order is fixed",
            "alternative": "Lost input/output locality costs more than scheduling saves",
            "variants": variants,
            "repeats": 7,
            "matvecs_per_capture": 80,
            "seed": 731,
            "judgment": "Adopt candidate for full-session validation only if frozen and random-mask currents are bit exact on four replays and median CUDA event time improves at least 10%; walking parity and 15% overall time reduction remain separate requirements",
            "stop": "One bounded variant set, single GPU, 300 wall seconds; retain all failures; no accuracy relaxation",
            "source_hash": file_hash(Path(__file__)),
            "checkpoint": str(args.checkpoint),
        },
    )
    graph = GraphStore.load(args.graph)
    weights, _ = rate_weights(graph)
    saved = StateStore.load(args.checkpoint)["network"]
    tensor = lambda x: torch.as_tensor(x, device="cuda")
    unique, inverse = np.unique(weights.data.view(np.uint32), return_inverse=True)
    assert graph.n <= 2**18 and len(unique) <= 2**14 and weights.nnz < 2**31
    entries = weights.indices.astype(np.uint32) | (inverse.astype(np.uint32) << 18)
    degrees = np.diff(weights.indptr)
    order = np.argsort(-degrees, kind="stable").astype(np.int32)
    lengths = degrees[order]
    packed_ptr = np.r_[0, lengths.cumsum()].astype(np.int32)
    edge_order = np.repeat(
        weights.indptr[order] - packed_ptr[:-1], lengths
    ) + np.arange(weights.nnz)
    lookup = tensor(unique.view(np.float32))
    row_order = tensor(order)
    rate, mask = tensor(saved["rate"]), tensor(saved["output_mask"])
    masked, result = torch.empty_like(rate), torch.empty_like(rate)
    resources, calls = [], []
    for name in variants:
        source = _PACKED_CSR_CURRENT
        row = tensor(
            packed_ptr if name == "degree_repack" else weights.indptr.astype(np.int32)
        )
        edge = tensor(
            (entries[edge_order] if name == "degree_repack" else entries).view(np.int32)
        )
        if name in ("degree_schedule", "degree_repack"):
            source = source.replace(
                "float* current, int n", "const int* row_order, float* current, int n"
            )
            if name == "degree_schedule":
                source = source.replace(
                    "if (row >= n) return;",
                    "if (row >= n) return;\n    row = row_order[row];",
                )
            else:
                source = source.replace(
                    "current[row] = value", "current[row_order[row]] = value"
                )
        elif name == "shared_lookup":
            source = source.replace(
                "    int lane",
                f"    __shared__ float local_lookup[{len(unique)}];\n    for (int i = threadIdx.x; i < {len(unique)}; i += blockDim.x) local_lookup[i] = lookup[i];\n    __syncthreads();\n    int lane",
            )
            source = source.replace(
                "float weight = lookup[item >> 18]",
                "float weight = local_lookup[item >> 18]",
            )
        elif name != "original":
            source = (
                source.replace("const int* ", "const int* __restrict__ ")
                .replace("const unsigned int* ", "const unsigned int* __restrict__ ")
                .replace("const float* ", "const float* __restrict__ ")
                .replace("float* current", "float* __restrict__ current")
            )
            if name.startswith("unroll"):
                source = source.replace(
                    "    for (int edge",
                    "    #pragma unroll " + name[6:] + "\n    for (int edge",
                )
            if name == "stream_entries":
                source = source.replace(
                    "unsigned int item = entries[edge];",
                    'unsigned int item; asm("ld.global.cs.u32 %0, [%1];" : "=r"(item) : "l"(entries + edge));',
                )
        kernel = cp.RawKernel(source, "csr_current", options=("--fmad=false",))
        kernel.compile()
        inputs = (
            (row, edge, lookup, masked)
            + ((row_order,) if name in ("degree_schedule", "degree_repack") else ())
            + (result,)
        )
        pointers = tuple(np.uint64(x.data_ptr()) for x in inputs) + (np.int32(graph.n),)
        resources.extend((row, edge, kernel))

        def call(kernel=kernel, pointers=pointers):
            torch.mul(rate, mask, out=masked)
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                kernel(((graph.n + 7) // 8,), (256,), pointers)

        calls.append((name, call))
    captures, records = [], []
    with torch.inference_mode():
        reference = None
        for name, call in calls:
            call()
            output = result.cpu().numpy().copy()
            if reference is None:
                reference = output
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                for _ in range(80):
                    call()
            captures.append(capture)
            records.append(
                {
                    "name": name,
                    "frozen_bit_equal": bool(
                        np.array_equal(
                            output.view(np.uint32), reference.view(np.uint32)
                        )
                    ),
                    "times_ms": [],
                }
            )
        for repeat in range(7):
            for index in (
                range(len(variants))
                if repeat % 2 == 0
                else reversed(range(len(variants)))
            ):
                capture = captures[index]
                capture.replay()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                capture.replay()
                end.record()
                end.synchronize()
                records[index]["times_ms"].append(begin.elapsed_time(end) / 80)
        rng = np.random.default_rng(731)
        rate.copy_(tensor(rng.uniform(0, 200, graph.n).astype(np.float32)))
        mask.copy_(tensor((rng.uniform(size=graph.n) > 0.2).astype(np.float32)))
        reference = None
        for capture, record in zip(captures, records):
            outputs = []
            for _ in range(4):
                capture.replay()
                outputs.append(result.cpu().numpy().copy())
            if reference is None:
                reference = outputs[0]
            record["random_bit_equal"] = all(
                np.array_equal(x.view(np.uint32), reference.view(np.uint32))
                for x in outputs
            )
            record["random_max_error"] = float(np.max(np.abs(outputs[0] - reference)))
            record["median_ms"] = float(np.median(record["times_ms"]))
    write_json(
        args.out / "result.json",
        {
            "validity": "valid",
            "graph_hash": graph.hash,
            "device": torch.cuda.get_device_name(),
            "variants": records,
        },
    )
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--probe", choices=("schedule", "memory", "shared"), default="schedule"
    )
    run(p.parse_args())
