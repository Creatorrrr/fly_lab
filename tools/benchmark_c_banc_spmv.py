"""CUDA-only candidate audit; does not replace the production numerical model."""

import argparse
import ctypes
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
from tools.banc_cuda_reference import LEGACY_PACKED_CSR_CURRENT


def run(args):
    import cupy as cp
    import torch
    from cupy_backends.cuda.libs import cusparse as cs

    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    weights, _ = rate_weights(graph)
    saved = StateStore.load(args.checkpoint)["network"]
    write_json(
        args.out / "spec.json",
        {
            "scope": "Frozen-vector matvec candidate audit, not RK4/walking adoption; all source entries preserved",
            "graph_hash": graph.hash,
            "checkpoint": str(args.checkpoint),
            "repeats": 7,
            "matvecs_per_capture": 80,
            "order": "forward/reverse interleaved",
            "candidates": "packed block64/128/256/512, packed pre-mask, cuSPARSE CSR_ALG1 and CSR_ALG2 with/without preprocessing",
            "masking": "Each pre-mask and cuSPARSE call includes the same rate*mask elementwise operation",
            "judgment": "Compare GPU event medians, exact currents and repeated bit equality. Different reduction order needs new numerical identity plus full validation before adoption.",
            "source_hash": file_hash(Path(__file__)),
            "kernel_sources": {
                str(p): file_hash(p)
                for p in (
                    Path("flylab/c/adaptive_rate.py"),
                    Path("tools/banc_cuda_reference.py"),
                )
            },
        },
    )
    tensor = lambda a: torch.as_tensor(a, device="cuda")
    unique, inverse = np.unique(weights.data.view(np.uint32), return_inverse=True)
    assert graph.n <= 2**18 and len(unique) <= 2**14 and weights.nnz < 2**31
    packed = weights.indices.astype(np.uint32) | (inverse.astype(np.uint32) << 18)
    row = tensor(weights.indptr.astype(np.int32))
    entries = tensor(packed.view(np.int32))
    lookup = tensor(unique.view(np.float32))
    columns, values = tensor(weights.indices.astype(np.int32)), tensor(weights.data)
    rate, mask = tensor(saved["rate"]), tensor(saved["output_mask"])
    masked, result = torch.empty_like(rate), torch.empty_like(rate)
    variants, resources = [], []

    def packed_call(block, premask=False):
        source = _PACKED_CSR_CURRENT if premask else LEGACY_PACKED_CSR_CURRENT
        kernel = cp.RawKernel(source, "csr_current", options=("--fmad=false",))
        kernel.compile()
        inputs = (
            (row, entries, lookup, masked, result)
            if premask
            else (row, entries, lookup, rate, mask, result)
        )
        pointers = tuple(np.uint64(t.data_ptr()) for t in inputs) + (np.int32(graph.n),)

        def call():
            if premask:
                torch.mul(rate, mask, out=masked)
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                kernel(
                    ((graph.n + block // 32 - 1) // (block // 32),), (block,), pointers
                )

        return call

    for block in (256, 64, 128, 512):
        variants.append((f"packed_block{block}", packed_call(block)))
    variants.append(("packed_premask_block256", packed_call(256, True)))
    handle = cs.create()
    cs.setPointerMode(handle, cs.CUSPARSE_POINTER_MODE_HOST)
    alpha, beta = np.array(1, np.float32), np.array(0, np.float32)
    # The CuPy wrapper resolves the already-loaded Torch cuSPARSE on this host.
    # Use that exact DLL for the missing preprocess binding, not another wheel's
    # identically named DLL. The version guard below verifies the pairing.
    lib = ctypes.CDLL(
        str(Path(torch.__file__).resolve().parent / "lib/cusparse64_12.dll")
    )
    preprocess = lib.cusparseSpMV_preprocess
    preprocess.restype = ctypes.c_int
    preprocess.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    raw_spmv = lib.cusparseSpMV
    raw_spmv.restype = ctypes.c_int
    raw_spmv.argtypes = preprocess.argtypes
    raw_stream = lib.cusparseSetStream
    raw_stream.restype = ctypes.c_int
    raw_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    version = ctypes.c_int()
    lib.cusparseGetVersion.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    assert lib.cusparseGetVersion(handle, ctypes.byref(version)) == 0
    assert version.value == cs.getVersion(handle), "Mismatched cuSPARSE DLLs"
    descriptors = []
    try:
        for algorithm, name in (
            (cs.CUSPARSE_CSRMV_ALG1, "alg1"),
            (cs.CUSPARSE_CSRMV_ALG2, "alg2"),
        ):
            for prepared in (False, True):
                desc = cs.createCsr(
                    graph.n,
                    graph.n,
                    weights.nnz,
                    row.data_ptr(),
                    columns.data_ptr(),
                    values.data_ptr(),
                    cs.CUSPARSE_INDEX_32I,
                    cs.CUSPARSE_INDEX_32I,
                    cs.CUSPARSE_INDEX_BASE_ZERO,
                    0,
                )
                dx, dy = (
                    cs.createDnVec(graph.n, masked.data_ptr(), 0),
                    cs.createDnVec(graph.n, result.data_ptr(), 0),
                )
                descriptors.append((desc, dx, dy))
                op_args = (
                    handle,
                    cs.CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha.ctypes.data,
                    desc,
                    dx,
                    beta.ctypes.data,
                    dy,
                    0,
                    algorithm,
                )
                cs.setStream(handle, torch.cuda.current_stream().cuda_stream)
                workspace = torch.empty(
                    cs.spMV_bufferSize(*op_args), dtype=torch.uint8, device="cuda"
                )
                resources.append(workspace)
                args_mv = (*op_args, workspace.data_ptr())
                if prepared:
                    status = preprocess(*args_mv)
                    if status != 0:
                        raise RuntimeError(f"cusparseSpMV_preprocess status {status}")

                def call(args_mv=args_mv):
                    torch.mul(rate, mask, out=masked)
                    # Installed CuPy wrappers reject stream capture despite
                    # native cuSPARSE supporting it. Keep the same DLL/handle.
                    if raw_stream(handle, torch.cuda.current_stream().cuda_stream) != 0:
                        raise RuntimeError("cusparseSetStream failed")
                    status = raw_spmv(*args_mv)
                    if status != 0:
                        raise RuntimeError(f"cusparseSpMV status {status}")

                variants.append(
                    (f"cusparse_{name}" + ("_preprocessed" if prepared else ""), call)
                )
        captures, records = [], []
        reference = None
        with torch.inference_mode():
            for name, call in variants:
                call()
                actual = result.cpu().numpy().copy()
                if reference is None:
                    reference = actual
                error = actual.astype(np.float64) - reference
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    for _ in range(80):
                        call()
                captures.append(capture)
                records.append(
                    {
                        "name": name,
                        "bit_equal_to_reference": bool(
                            np.array_equal(
                                actual.view(np.uint32), reference.view(np.uint32)
                            )
                        ),
                        "max_error": float(np.abs(error).max()),
                        "rms_error": float(np.sqrt(np.mean(error**2))),
                        "times_ms": [],
                    }
                )
            for repeat in range(7):
                for index in (
                    range(len(captures))
                    if repeat % 2 == 0
                    else reversed(range(len(captures)))
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
                record["random_repeats_bit_equal"] = all(
                    np.array_equal(a.view(np.uint32), outputs[0].view(np.uint32))
                    for a in outputs
                )
                record["random_bit_equal_to_reference"] = bool(
                    np.array_equal(
                        reference.view(np.uint32), outputs[0].view(np.uint32)
                    )
                )
                record["random_max_error"] = float(np.abs(outputs[0] - reference).max())
                record["median_ms"] = float(np.median(record["times_ms"]))
        write_json(
            args.out / "result.json",
            {
                "graph_hash": graph.hash,
                "device": torch.cuda.get_device_name(),
                "cusparse_version": version.value,
                "variants": records,
            },
        )
        for record in records:
            print(
                json.dumps({k: v for k, v in record.items() if k != "times_ms"}),
                flush=True,
            )
    finally:
        torch.cuda.synchronize()
        for desc, dx, dy in descriptors:
            cs.destroyDnVec(dx)
            cs.destroyDnVec(dy)
            cs.destroySpMat(desc)
        cs.destroy(handle)


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
    run(parser.parse_args())
