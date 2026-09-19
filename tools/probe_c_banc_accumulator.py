"""One sequential RK accumulator candidate: exactness and three-arm timing.

Preparing/importing this module does not initialize CUDA. Execution requires
the explicit --run-cuda flag after the shared GPU has been allocated.
"""

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

from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore, runtime_versions
from tools import probe_c_banc_exact_rk as checks
from tools.banc_accumulator_candidate import SOURCE, AccumulatorCandidate
from tools.verify_c_banc_observation import exact
from tools.verify_c_banc_source_parity import (
    observe,
    reference_class,
    restore_same,
    source_identity,
)

PROFILES = ("general", "target")
ARMS = ("baseline", "current", "accumulator")
PILOT_CONTROLS = 20
TIMING_CONTROLS = 100
REPEATS = 5


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError("Bounded accumulator candidate experiment expired")


def exceptional_probe(out):
    import cupy as cp
    import torch

    kernels = {
        name: cp.RawKernel(SOURCE, name, options=("--fmad=false",))
        for name in ("stage", "finish")
    }
    for kernel in kernels.values():
        kernel.compile()
    rng = np.random.default_rng(12463)
    n = 4096
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
    reports = []
    for dt, tau_a in ((0.0001, 0.15), (0.00025, 0.03), (0.0002, 0.0602)):

        def tensor(low, high):
            return torch.tensor(
                rng.uniform(low, high, n).astype(np.float32), device="cuda"
            )

        base_r, base_z, cap, tau = (
            tensor(0, 200),
            tensor(0, 50),
            tensor(1, 250),
            tensor(0.02, 0.2),
        )
        base_r[: len(edge)] = torch.tensor(edge, device="cuda")
        original_r, original_z = base_r.clone(), base_z.clone()
        reference_r, reference_z = original_r, original_z
        work_r, work_z, masked, total_r, total_z = [
            torch.empty_like(base_r) for _ in range(5)
        ]
        mask = torch.tensor((rng.random(n) > 0.2).astype(np.float32), device="cuda")
        r, z = base_r, base_z
        expected_sum_r, expected_sum_z = None, None
        for stage in range(4):
            activation = tensor(-1, 1)
            activation[: len(edge)] = torch.tensor(edge, device="cuda")
            kr = (torch.clamp_min(cap * activation, 0.0) - reference_r) / tau
            ka = (2.7 * reference_r - reference_z) / tau_a
            if stage < 3:
                coefficient = dt * (0.5 if stage < 2 else 1)
                expected_sum_r = kr if stage == 0 else expected_sum_r + 2 * kr
                expected_sum_z = ka if stage == 0 else expected_sum_z + 2 * ka
                next_r = original_r + coefficient * kr
                next_z = original_z + coefficient * ka
                tensors = (
                    activation,
                    cap,
                    r,
                    z,
                    tau,
                    base_r,
                    base_z,
                    mask,
                    total_r,
                    total_z,
                    work_r,
                    work_z,
                    masked,
                )
                args = tuple(np.uint64(t.data_ptr()) for t in tensors) + (
                    np.float32(2.7),
                    np.float32(1 / tau_a),
                    np.float32(coefficient),
                    np.int32(stage),
                    np.int32(n),
                )
                with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                    kernels["stage"](((n + 255) // 256,), (256,), args)
                for label, expected, actual in (
                    ("sum-rate", expected_sum_r, total_r),
                    ("sum-adaptation", expected_sum_z, total_z),
                    ("next-rate", next_r, work_r),
                    ("next-adaptation", next_z, work_z),
                    ("premask", next_r * mask, masked),
                ):
                    checks.bit_equal(expected, actual, f"dt{dt} stage{stage} {label}")
                reference_r, reference_z = next_r, next_z
                r, z = work_r, work_z
            else:
                final_r = original_r + (dt / 6) * (expected_sum_r + kr)
                final_z = original_z + (dt / 6) * (expected_sum_z + ka)
                tensors = (activation, cap, r, z, tau, total_r, total_z, base_r, base_z)
                args = tuple(np.uint64(t.data_ptr()) for t in tensors) + (
                    np.float32(2.7),
                    np.float32(1 / tau_a),
                    np.float32(dt / 6),
                    np.int32(n),
                )
                with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                    kernels["finish"](((n + 255) // 256,), (256,), args)
                checks.bit_equal(final_r, base_r, f"dt{dt} final-rate")
                checks.bit_equal(final_z, base_z, f"dt{dt} final-adaptation")
        reports.append(
            {"dt": dt, "adaptation_tau": tau_a, "every_stage_and_final_bit_exact": True}
        )
    write_json(out / "exceptional.json", reports)


def run_profile(profile, graph, base_class, out, deadline):
    import torch

    out.mkdir()
    sessions = {}
    result = {"status": "running", "profile": profile}
    try:
        parameters = {"neural_dt_s": 0.00025}
        navigation = None
        if profile == "target":
            parameters.update(coxa_geometry=True, ltm_tendons=True)
            navigation = {
                "target_xz_mm": [6.555187432018627, -2.9940043430431347],
                "forward_drive": 5000.0,
            }
        sessions["baseline"] = base_class(
            graph, parameters=parameters, device="cuda", navigation=navigation
        )
        saved = sessions["baseline"].snapshot()
        StateStore.save(out / "start", saved)
        module = importlib.import_module("flylab.c.banc_walking")
        sessions["current"] = module.BancWalkingSession.from_checkpoint(graph, saved)
        original = module.AdaptiveRateNetwork
        try:
            module.AdaptiveRateNetwork = AccumulatorCandidate
            sessions["accumulator"] = module.BancWalkingSession.from_checkpoint(
                graph, saved
            )
        finally:
            module.AdaptiveRateNetwork = original
        for session in sessions.values():
            exact(saved, session.snapshot())
        net = sessions["accumulator"].network
        result["buffers"] = {
            "neurons": net.n,
            "candidate_arrays": len(net._fused_buffers),
            "current_arrays": len(sessions["current"].network._fused_buffers),
            "candidate_bytes": sum(
                t.numel() * t.element_size() for t in net._fused_buffers
            ),
            "current_bytes": sum(
                t.numel() * t.element_size()
                for t in sessions["current"].network._fused_buffers
            ),
        }
        write_json(out / "buffers.json", result["buffers"])
        for control in range(PILOT_CONTROLS):
            sessions["baseline"].advance(1)
            reference_signals = observe(sessions["baseline"])
            reference_state = sessions["baseline"].snapshot()
            for arm in ("current", "accumulator"):
                sessions[arm].advance(1)
                exact(
                    reference_signals,
                    observe(sessions[arm]),
                    f"{profile}.{arm}.control{control + 1}",
                )
                exact(
                    reference_state,
                    sessions[arm].snapshot(),
                    f"{profile}.{arm}.full{control + 1}",
                )
            check_deadline(deadline)
        for arm, session in sessions.items():
            StateStore.save(out / f"{arm}-pilot-final", session.snapshot())
        result["pilot"] = {
            "model_s": 0.1,
            "control_samples": 20,
            "every_control_and_full_state_bit_exact": True,
        }

        print(
            json.dumps(
                {"profile": profile, "pilot": "bit exact", "buffers": result["buffers"]}
            ),
            flush=True,
        )
        rows = []
        canonical_future = None
        for pair in range(REPEATS):
            for arm in ARMS if pair % 2 == 0 else ARMS[::-1]:
                session = sessions[arm]
                restore_same(session, saved)
                session.advance(10)
                restore_same(session, saved)
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _ in range(TIMING_CONTROLS // 10):
                    session.advance(10)
                torch.cuda.synchronize()
                wall = time.perf_counter() - began
                now = session.snapshot()
                if canonical_future is None:
                    canonical_future = now
                else:
                    exact(canonical_future, now, f"{profile}.{arm}.timing_future{pair}")
                row = {
                    "profile": profile,
                    "pair": pair,
                    "arm": arm,
                    "model_s": 0.5,
                    "wall_s": wall,
                }
                rows.append(row)
                write_json(out / "runs.json", rows)
                print(json.dumps(row), flush=True)
                check_deadline(deadline)
        medians = {
            arm: statistics.median(row["wall_s"] for row in rows if row["arm"] == arm)
            for arm in ARMS
        }
        result.update(
            status="completed",
            validity="valid",
            medians=medians,
            accumulator_reduction_vs_baseline=1
            - medians["accumulator"] / medians["baseline"],
            accumulator_reduction_vs_current=1
            - medians["accumulator"] / medians["current"],
            current_reduction_vs_baseline=1 - medians["current"] / medians["baseline"],
            every_timing_future_bit_exact=True,
        )
    except AssertionError as exc:
        result.update(
            status="completed", validity="valid", hypothesis="refuted", error=str(exc)
        )
    except Exception as exc:  # noqa: BLE001
        # Preserve unexpected runtime/tool failures as invalid experiments,
        # distinct from valid numerical counterexamples handled above.
        result.update(status="aborted", validity="invalid", error=repr(exc))
    finally:
        write_json(out / "result.json", result)
        for session in sessions.values():
            session.close()
    print(json.dumps(result), flush=True)
    return result


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA device required")
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    deadline = began + args.max_wall_seconds
    write_json(
        args.out / "spec.json",
        {
            "hypothesis": "Replace six derivative arrays with two running sums, keeping (((k1+2*k2)+2*k3)+k4) rounded exactly in the original order; no matrix or algorithm changes.",
            "profiles": args.profiles,
            "arms": ARMS,
            "order": "forward/reverse alternating by pair",
            "pilot_control_samples": PILOT_CONTROLS,
            "pilot_model_seconds": 0.1,
            "exceptional": "Every intermediate running sum, next stage and final reduction checked against original Torch operation order with subnormal, signed-zero, infinity and NaN values; original-source small network cuts and restore tests.",
            "repeats": REPEATS,
            "timing_model_seconds": 0.5,
            "timing_excludes": "Initialization, compilation, capture, warmup, restore, snapshots, PNG, trace, server and UI",
            "decision": "Report parity and headless timing only. Final15% and recording acceptance belong to the main source-parity tool.",
            "original_source": source_identity(args.baseline),
            "current_source": source_identity(ROOT),
            "candidate_sha256": file_hash(ROOT / "tools/banc_accumulator_candidate.py"),
            "probe_sha256": file_hash(Path(__file__)),
            "runtime": runtime_versions(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "max_wall_seconds": args.max_wall_seconds,
        },
    )
    (args.out / "candidate.py").write_bytes(
        (ROOT / "tools/banc_accumulator_candidate.py").read_bytes()
    )
    (args.out / "probe.py").write_bytes(Path(__file__).read_bytes())
    base_class = reference_class(args.baseline)
    exceptional_probe(args.out)
    original_neural = sys.modules[base_class.__module__].AdaptiveRateNetwork

    class OriginalForChecks(original_neural):
        def advance_readout(self, drive, ids, steps):
            self.advance(drive, steps)
            return self.readout(ids)

    checks.AdaptiveRateNetwork = OriginalForChecks
    checks.ExactRKCandidate = AccumulatorCandidate
    checks.small_network(args.out)
    checks.tanh_primitive(args.out)
    print("exceptional sums, original-source small networks and tanh exact", flush=True)
    graph = GraphStore.load(args.graph)
    results = []
    for profile in args.profiles:
        check_deadline(deadline)
        result = run_profile(profile, graph, base_class, args.out / profile, deadline)
        results.append(result)
        summary = {"profiles": results, "wall_s": time.perf_counter() - began}
        write_json(args.out / "result.json", summary)
        if result.get("validity") != "valid" or result.get("hypothesis") == "refuted":
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "verification/banc-observation-m1-20260912/baseline",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/acquisitions/banc888-windows-20260910/bundle",
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    parser.add_argument("--max-wall-seconds", type=int, default=175)
    parser.add_argument("--run-cuda", action="store_true")
    args = parser.parse_args()
    if not args.run_cuda:
        parser.error(
            "CUDA execution is disabled; use --run-cuda only after GPU allocation"
        )
    if (
        len(set(args.profiles)) != len(args.profiles)
        or not 60 <= args.max_wall_seconds <= 480
    ):
        parser.error("Distinct profiles and a60..480second wall bound required")
    raise SystemExit(run(args))
