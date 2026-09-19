"""Bounded experimental RK fusion verification, separate from production."""

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from tools.banc_exact_rk_candidate import SOURCE, ExactRKCandidate
from tools.verify_c_banc_observation import exact
from tools.verify_c_banc_source_parity import observe, reference_class, restore_same


def bit_equal(a, b, name):
    aa, bb = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    indices = np.flatnonzero(aa.view(np.uint32) != bb.view(np.uint32))
    if indices.size:
        at = int(indices[0])
        raise AssertionError(
            f"{name}: {len(indices)} differing bits, first {at}: {aa[at]!r} / {bb[at]!r}"
        )


def tanh_primitive(out):
    import cupy as cp
    import torch

    rng = np.random.default_rng(8431)
    bits = rng.integers(0, 2**32, 1048576, dtype=np.uint32)
    dense = np.linspace(-20, 20, 1048576, dtype=np.float32)
    edges = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x7F800000,
            0xFF800000,
            0x7FFFFFFF,
            0xFFC00000,
        ],
        dtype=np.uint32,
    )
    values = np.concatenate([bits.view(np.float32), dense, edges.view(np.float32)])
    source = (
        SOURCE
        + """\nextern "C" __global__ void tanh_probe(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = matching_tanh(x[i]); }"""
    )
    kernel = cp.RawKernel(source, "tanh_probe", options=("--fmad=false",))
    kernel.compile()
    x = torch.tensor(values, device="cuda")
    y = torch.empty_like(x)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        kernel(
            ((len(values) + 255) // 256,),
            (256,),
            (np.uint64(x.data_ptr()), np.uint64(y.data_ptr()), np.int32(len(values))),
        )
    bit_equal(torch.tanh(x), y, "dense/random-bits tanh primitive")
    write_json(
        out / "tanh-primitive.json",
        {
            "samples": len(values),
            "bit_exact": True,
            "distributions": "uniform[-20,20], uniform 32-bit patterns, signed zero/subnormal/Inf/NaN edges",
        },
    )


def primitive(out):
    import cupy as cp
    import torch

    rng = np.random.default_rng(9731)
    count = 4096
    values = [
        torch.tensor(rng.uniform(-300, 300, count).astype(np.float32), device="cuda")
        for _ in range(8)
    ]
    activation, cap, rate, adaptation, tau, br, bz, mask = values
    cap.abs_().add_(1)
    tau.abs_().add_(0.02)
    edge = np.array(
        [
            0.0,
            -0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            -np.nextafter(np.float32(0), np.float32(1)),
            np.inf,
            -np.inf,
            np.nan,
        ],
        np.float32,
    )
    rate[: len(edge)] = torch.tensor(edge, device="cuda")
    activation[: len(edge)] = torch.tensor(edge, device="cuda")
    mask[::3] = 0
    mask[1::3] = 1
    kernel = cp.RawKernel(SOURCE, "stage", options=("--fmad=false",))
    kernel.compile()
    output = [torch.empty_like(rate) for _ in range(5)]
    reports = []
    for gain, at, step in (
        (4.0, 0.15, 0.000125),
        (2.7, 0.03, 0.0001),
        (2.7, 0.0602, 0.00025),
    ):
        target = torch.clamp_min(cap * activation, 0.0)
        kr = (target - rate) / tau
        ka = (gain * rate - adaptation) / at
        nr, na = br + step * kr, bz + step * ka
        expected = [kr, ka, nr, na, nr * mask]
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            kernel(
                ((count + 255) // 256,),
                (256,),
                tuple(np.uint64(t.data_ptr()) for t in (*values, *output))
                + (
                    np.float32(gain),
                    np.float32(1 / at),
                    np.float32(step),
                    np.int32(count),
                ),
            )
        for index, (a, b) in enumerate(zip(expected, output)):
            bit_equal(a, b, f"primitive gain={gain} tau={at} output={index}")
        reports.append({"gain": gain, "adaptation_tau": at, "bit_exact": True})
    write_json(out / "primitive.json", reports)


def small_network(out):
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(9632)
    n = 263
    weight = rng.normal(0, 0.1, (n, n)).astype(np.float32)
    weight[rng.random((n, n)) < 0.7] = 0
    weight[0] = 0
    parameters = (
        csr_matrix(weight),
        rng.uniform(0.02, 0.08, n),
        rng.uniform(0.5, 2, n),
        rng.uniform(1, 10, n),
        rng.uniform(50, 250, n),
    )
    results = []
    for dt, adaptation_tau in ((0.0001, 0.15), (0.00025, 0.03), (0.0002, 0.0602)):
        options = {
            "adaptation_gain": 2.7,
            "adaptation_tau_s": adaptation_tau,
            "dt": dt,
            "capture_steps": 10,
            "device": "cuda",
            "cuda_implementation": "packed",
        }
        base, candidate = (
            cls(*parameters, **options)
            for cls in (AdaptiveRateNetwork, ExactRKCandidate)
        )
        initial = base.snapshot()
        initial["rate"] = rng.uniform(0, 200, n).astype(np.float32)
        initial["adaptation"] = rng.uniform(0, 50, n).astype(np.float32)
        for net in (base, candidate):
            net.restore(initial)
        for steps, cut in ((1, []), (21, [0, 42, 83]), (43, list(range(n))), (20, [])):
            drive = rng.uniform(-100, 150, n).astype(np.float32)
            for net in (base, candidate):
                net.set_muted(cut)
                before = net.snapshot()
                net.advance(drive, steps)
                after = net.snapshot()
                net.reset()
                net.restore(before)
                net.advance(drive, steps)
                exact(after, net.snapshot())
            exact(base.snapshot(), candidate.snapshot())
        for net in (base, candidate):
            net.rate[150] = float("nan")
            try:
                net.advance_readout(np.zeros(n), [0], 10)
            except RuntimeError as exc:
                if "Invalid raw complete-network" not in str(exc):
                    raise
            else:
                raise AssertionError(
                    "Nonfinite neuron outside readout was not rejected"
                )
        results.append(
            {
                "dt": dt,
                "adaptation_tau": adaptation_tau,
                "future_restore_cuts_errors_exact": True,
            }
        )
    write_json(out / "small-network.json", results)


def run(args):
    import torch

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "spec.json",
        {
            "hypothesis": (
                "Fuse CSR pre-tanh arithmetic and post-tanh RK state construction while preserving every original rounded operation. "
                + (
                    "Fuse validated CUDA tanhf into post-RK kernels, preserving subnormal inputs."
                    if args.tanh
                    else "Torch tanh retained."
                )
            ),
            "source": file_hash(ROOT / "tools/banc_exact_rk_candidate.py"),
            "prior_failure": "fusion-rejected remains unchanged; prior partial derivative fusion was inexact and only about 3 percent faster.",
            "adoption": "Primitive exceptional value parity, 0.1 s complete-bundle exactness; if passed, 5 paired 0.5 s full-session timings. No production adoption or tolerances changed.",
            "wall_bound_s": 720,
            "gpu_bound_s": 480,
            "fused_tanh": args.tanh,
        },
    )
    (out / "candidate.py").write_bytes(
        (ROOT / "tools/banc_exact_rk_candidate.py").read_bytes()
    )
    (out / "probe.py").write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    result = {"status": "running"}
    sessions = []
    try:
        if args.tanh:
            tanh_primitive(out)
            ExactRKCandidate.fuse_tanh = True
            print("dense/random tanh primitive exact", flush=True)
        primitive(out)
        print("primitive exact", flush=True)
        small_network(out)
        print(
            "small networks exact with cuts, tails, resets and nonfinite errors",
            flush=True,
        )
        base_class = reference_class(
            ROOT / "verification/banc-observation-m1-20260912/baseline"
        )
        graph = GraphStore.load(
            ROOT / "data/acquisitions/banc888-windows-20260910/bundle"
        )
        saved = StateStore.load(
            ROOT / "verification/banc-observation-m1-20260912/e2-pr-timing/start"
        )
        base = base_class.from_checkpoint(graph, saved)
        sessions.append(base)
        module = importlib.import_module("flylab.c.banc_walking")
        original = module.AdaptiveRateNetwork
        try:
            module.AdaptiveRateNetwork = ExactRKCandidate
            candidate = module.BancWalkingSession.from_checkpoint(graph, saved)
        finally:
            module.AdaptiveRateNetwork = original
        sessions.append(candidate)
        for tick in range(20):
            base.advance(1)
            candidate.advance(1)
            left, right = observe(base), observe(candidate)
            for key in left:
                if left[key].tobytes() != right[key].tobytes():
                    raise AssertionError(f"pilot control {tick + 1}: {key}")
            exact(base.snapshot(), candidate.snapshot())
        print("full pilot exact", flush=True)
        result["pilot"] = "20 control samples and complete future bit exact"
        rows = []
        for pair in range(5):
            for name, session in (
                (("baseline", base), ("candidate", candidate))
                if pair % 2 == 0
                else (("candidate", candidate), ("baseline", base))
            ):
                restore_same(session, saved)
                session.advance(10)
                restore_same(session, saved)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    session.advance(10)
                torch.cuda.synchronize()
                wall = time.perf_counter() - started
                row = {"pair": pair, "arm": name, "wall_s": wall}
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(out / "runs.json", rows)
            exact(base.snapshot(), candidate.snapshot())
        medians = {
            arm: statistics.median(row["wall_s"] for row in rows if row["arm"] == arm)
            for arm in ("baseline", "candidate")
        }
        result.update(
            status="completed",
            validity="valid",
            medians=medians,
            reduction=1 - medians["candidate"] / medians["baseline"],
            final_exact=True,
        )
    except AssertionError as exc:
        result.update(
            status="completed", validity="valid", hypothesis="refuted", error=str(exc)
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        result.update(status="aborted", validity="invalid", error=repr(exc))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - began
        write_json(out / "result.json", result)
        for session in sessions:
            session.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tanh", action="store_true")
    run(parser.parse_args())
