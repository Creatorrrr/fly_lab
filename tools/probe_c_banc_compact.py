"""One24-bit candidate: full-source parity and three-arm headless timing.

Preparing/importing this module does not initialize CUDA. Execution requires
the explicit --run-cuda flag after the shared GPU has been allocated.
"""

import argparse
import hashlib
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
from tools.banc_compact_candidate import SOURCE as COMPACT_SOURCE
from tools.banc_compact_candidate import CompactCandidate
from tools.verify_c_banc_observation import exact
from tools.verify_c_banc_source_parity import (
    observe,
    reference_class,
    restore_same,
    source_identity,
)

PROFILES = ("general", "target")
ARMS = ("baseline", "current", "compact")
PILOT_CONTROLS = 20
TIMING_CONTROLS = 100
REPEATS = 5


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError("Bounded compact candidate experiment expired")


def current_only_source():
    """Derive the same compact reduction, exposing its pre-pointwise value."""
    epilogue = """    if (lane == 0) {
        value = add_exact(drive[row], value);
        value = sub_exact(value, threshold[row]);
        value = sub_exact(value, adaptation[row]);
        input[row] = mul_exact(coefficient[row], value);
    }"""
    if COMPACT_SOURCE.count(epilogue) != 1:
        raise RuntimeError("Compact current verifier source no longer matches")
    return COMPACT_SOURCE.replace(
        "compact_csr_input(", "compact_csr_current(", 1
    ).replace(epilogue, "    if (lane == 0) input[row] = value;", 1)


def check_current_bits(sessions, out, deadline):
    """Compare actual compact decoding, not its inherited32-bit _current()."""
    import cupy as cp
    import torch

    base, candidate = sessions["baseline"].network, sessions["compact"].network
    snapshots = {arm: session.network.snapshot() for arm, session in sessions.items()}
    frozen_rate = snapshots["baseline"]["rate"]
    if not np.count_nonzero(frozen_rate):
        raise AssertionError("The frozen matvec state must be nontrivial")
    kernel = cp.RawKernel(
        current_only_source(), "compact_csr_current", options=("--fmad=false",)
    )
    kernel.compile()
    result = torch.empty_like(candidate.rate)
    masked = torch.empty_like(candidate.rate)
    rng = np.random.default_rng(9931)
    fixtures = (
        ("frozen-after-0.1s", frozen_rate, snapshots["baseline"]["output_mask"]),
        (
            "random-rate-and-outgoing-cuts",
            rng.uniform(0, 200, base.n).astype(np.float32),
            (rng.random(base.n) >= 0.2).astype(np.float32),
        ),
    )
    checks = []
    try:
        with torch.inference_mode():
            for fixture, rate, mask in fixtures:
                device_rate = torch.as_tensor(rate, device="cuda")
                device_mask = torch.as_tensor(mask, device="cuda")
                for session in sessions.values():
                    session.network.output_mask.copy_(device_mask)
                expected = base._current(device_rate).detach().cpu().numpy().copy()
                for repeat in range(4):
                    for arm in ("current", "compact"):
                        net = sessions[arm].network
                        if arm == "current":
                            actual = net._current(device_rate)
                        else:
                            torch.mul(device_rate, candidate.output_mask, out=masked)
                            tensors = (
                                candidate._rowptr,
                                candidate._compact_words,
                                candidate._weight_values,
                                candidate._compact_hot,
                                candidate._compact_escapes,
                                masked,
                                candidate.drive,
                                candidate.threshold,
                                candidate.adaptation,
                                candidate.coefficient,
                                result,
                            )
                            with cp.cuda.ExternalStream(
                                torch.cuda.current_stream().cuda_stream
                            ):
                                kernel(
                                    ((candidate.n + 7) // 8,),
                                    (256,),
                                    tuple(np.uint64(t.data_ptr()) for t in tensors)
                                    + (np.int32(candidate.n),),
                                )
                            actual = result
                        value = actual.detach().cpu().numpy()
                        if (
                            value.dtype != expected.dtype
                            or value.tobytes() != expected.tobytes()
                        ):
                            differing = np.flatnonzero(
                                value.view(np.uint32) != expected.view(np.uint32)
                            )
                            raise AssertionError(
                                f"{fixture} {arm} repeat{repeat}: current bits differ at neuron{int(differing[0])}"
                            )
                        check_deadline(deadline)
                checks.append(
                    {
                        "fixture": fixture,
                        "four_repeats_bit_exact": True,
                        "nonzero_rates": int(np.count_nonzero(rate)),
                        "muted_sources": int(np.count_nonzero(mask == 0)),
                        "rate_sha256": hashlib.sha256(rate.tobytes()).hexdigest(),
                        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                        "current_sha256": hashlib.sha256(
                            expected.tobytes()
                        ).hexdigest(),
                    }
                )
    finally:
        for arm, session in sessions.items():
            session.network.restore(snapshots[arm])
    write_json(out / "matvec.json", checks)
    return checks


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
            module.AdaptiveRateNetwork = CompactCandidate
            sessions["compact"] = module.BancWalkingSession.from_checkpoint(
                graph, saved
            )
        finally:
            module.AdaptiveRateNetwork = original
        for session in sessions.values():
            exact(saved, session.snapshot())
        net = sessions["compact"].network
        result["matrix"] = {
            "graph_hash": graph.hash,
            "network_hash": net.identity,
            "neurons": net.n,
            "original_entries": net._columns.numel(),
            "original_lookup_entries": net._weight_values.numel(),
            "hot_lookup_entries": net._compact_hot.numel(),
            "escape_entries": net._compact_escapes.numel(),
            "compact_padded_bytes": net._compact_words.numel() * 4,
            "all_entries_initial_cpu_roundtrip": True,
        }
        write_json(out / "matrix.json", result["matrix"])
        for control in range(PILOT_CONTROLS):
            sessions["baseline"].advance(1)
            reference_signals = observe(sessions["baseline"])
            reference_state = sessions["baseline"].snapshot()
            for arm in ("current", "compact"):
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
        result["matvec"] = check_current_bits(sessions, out, deadline)
        print(
            json.dumps(
                {
                    "profile": profile,
                    "pilot_and_matvec": "bit exact",
                    "matrix": result["matrix"],
                }
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
            compact_reduction_vs_baseline=1 - medians["compact"] / medians["baseline"],
            compact_reduction_vs_current=1 - medians["compact"] / medians["current"],
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
            "hypothesis": "One top63+escape24-bit entry candidate reduces cache traffic while preserving every CSR and RK operation.",
            "profiles": args.profiles,
            "arms": ARMS,
            "order": "forward/reverse alternating by pair",
            "pilot_control_samples": PILOT_CONTROLS,
            "pilot_model_seconds": 0.1,
            "matvec": "Nontrivial frozen state after pilot and seeded random rates with20% outgoing cuts; four bit-exact repeats, actual compact parser exercised.",
            "repeats": REPEATS,
            "timing_model_seconds": 0.5,
            "timing_excludes": "Initialization, compilation, capture, warmup, restore, snapshots, PNG, trace, server and UI",
            "decision": "Report parity and headless timing only. Final15% and recording acceptance belong to the main source-parity tool.",
            "original_source": source_identity(args.baseline),
            "current_source": source_identity(ROOT),
            "candidate_sha256": file_hash(ROOT / "tools/banc_compact_candidate.py"),
            "probe_sha256": file_hash(Path(__file__)),
            "runtime": runtime_versions(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "max_wall_seconds": args.max_wall_seconds,
        },
    )
    (args.out / "candidate.py").write_bytes(
        (ROOT / "tools/banc_compact_candidate.py").read_bytes()
    )
    (args.out / "probe.py").write_bytes(Path(__file__).read_bytes())
    base_class = reference_class(args.baseline)
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
    parser.add_argument("--max-wall-seconds", type=int, default=240)
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
