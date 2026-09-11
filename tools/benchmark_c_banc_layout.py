"""Compare lossless CSR layouts without changing the live walking implementation."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.adaptive_rate import _CSR_CURRENT
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore

PACKED = r"""
extern "C" __global__ void csr_current(
    const int* rowptr, const unsigned int* entries,
    const float* lookup, const float* rate, const float* mask,
    float* current, int n) {
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= n) return;
    float value = 0.0f;
    for (int edge = rowptr[row] + lane; edge < rowptr[row + 1]; edge += 32) {
        unsigned int item = entries[edge];
        int column = item & 0x3ffff;
        float weight = lookup[item >> 18];
        value += weight * (rate[column] * mask[column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) current[row] = value;
}
"""


def run(args):
    import cupy as cp
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    weights, _ = rate_weights(graph)
    saved = StateStore.load(args.checkpoint)["network"]
    unique, inverse = np.unique(weights.data.view(np.uint32), return_inverse=True)
    if graph.n > 2**18 or len(unique) > 2**14 or weights.nnz >= 2**31:
        raise ValueError("This graph exceeds the exact packed layout bounds")
    packed = weights.indices.astype(np.uint32) | (inverse.astype(np.uint32) << 18)
    np.testing.assert_array_equal(packed & 0x3FFFF, weights.indices)
    np.testing.assert_array_equal(unique[packed >> 18], weights.data.view(np.uint32))

    def tensor(array):
        return torch.as_tensor(array, device="cuda")

    row64 = tensor(weights.indptr.astype(np.int64))
    row32 = tensor(weights.indptr.astype(np.int32))
    col64 = tensor(weights.indices.astype(np.int64))
    col32 = tensor(weights.indices.astype(np.int32))
    weight = tensor(weights.data)
    entries, lookup = tensor(packed.view(np.int32)), tensor(unique.view(np.float32))
    row_order = tensor(
        np.argsort(-np.diff(weights.indptr), kind="stable").astype(np.int32)
    )
    sorted_source = PACKED.replace(
        "const float* lookup, const float* rate",
        "const float* lookup, const int* row_order, const float* rate",
    ).replace(
        "int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;\n    if (row >= n) return;",
        "int slot = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;\n    if (slot >= n) return;\n    int row = row_order[slot];",
    )
    rate, mask = tensor(saved["rate"]), tensor(saved["output_mask"])
    result = torch.empty_like(rate)
    layouts = [
        ("original_64bit", _CSR_CURRENT, (row64, col64, weight)),
        (
            "indices_32bit",
            _CSR_CURRENT.replace("long long", "int"),
            (row32, col32, weight),
        ),
        ("exact_dictionary_32bit", PACKED, (row32, entries, lookup)),
        ("dictionary_sorted_rows", sorted_source, (row32, entries, lookup, row_order)),
    ]
    records, captures = [], []
    reference = None
    for name, source, buffers in layouts:
        kernel = cp.RawKernel(source, "csr_current", options=("--fmad=false",))
        kernel.compile()
        pointers = tuple(
            np.uint64(a.data_ptr()) for a in (*buffers, rate, mask, result)
        )

        def call(kernel=kernel, pointers=pointers):
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                kernel(((graph.n + 7) // 8,), (256,), pointers + (np.int32(graph.n),))

        call()
        actual = result.cpu().numpy().copy()
        if reference is None:
            reference = actual
        np.testing.assert_array_equal(actual, reference)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            for _ in range(100):
                call()
        captures.append((name, capture))
        records.append(
            {
                "name": name,
                "exact_saved_state_current": True,
                "bytes": sum(a.numel() * a.element_size() for a in buffers),
                "times_ms": [],
            }
        )
    # Interleave layouts to reduce the influence of GPU clocks and temperature.
    for repeat in range(7):
        for index in (
            range(len(layouts)) if repeat % 2 == 0 else reversed(range(len(layouts)))
        ):
            capture = captures[index][1]
            capture.replay()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            capture.replay()
            end.record()
            end.synchronize()
            records[index]["times_ms"].append(begin.elapsed_time(end) / 100)
    rng = np.random.default_rng(192)
    rate.copy_(tensor(rng.uniform(0, 200, graph.n).astype(np.float32)))
    mask.copy_(tensor((rng.uniform(0, 1, graph.n) > 0.2).astype(np.float32)))
    outputs = []
    for (_, capture), record in zip(captures, records):
        capture.replay()
        outputs.append(result.cpu().numpy().copy())
        np.testing.assert_array_equal(outputs[-1], outputs[0])
        record["exact_random_rate_and_mask_current"] = True
        record["median_ms"] = float(np.median(record["times_ms"]))
    report = {
        "scope": "Frozen-vector matvec microbenchmark only; not complete RK4 or physical walking speedup",
        "graph_hash": graph.hash,
        "neurons": graph.n,
        "edges": weights.nnz,
        "unique_float_bit_patterns": len(unique),
        "device": torch.cuda.get_device_name(),
        "layouts": records,
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(args.out / "result.json", report)
    for r in records:
        print({k: v for k, v in r.items() if k != "times_ms"}, flush=True)


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
