"""One preregistered CSR-stage fusion candidate; no production edits."""

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from tools import probe_c_banc_exact_rk as checks
from tools.banc_stage_fusion_candidate import StageFusionCandidate
from tools.verify_c_banc_observation import exact
from tools.verify_c_banc_source_parity import observe, reference_class, restore_same


def run(args):
    import torch

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "spec.json",
        {
            "hypothesis": "Merge the existing CSR reduction and pointwise RK stage into one kernel per stage; lane0 post-processing and two distinct ping-pong masked buffers prevent cross-row read/write races.",
            "arithmetic": "Original fixed-lane CSR float order and FTZ retained, existing exact PTX pointwise arithmetic and matching_tanh unchanged.",
            "adoption": "Primitive and random small network checks, full-source 0.1s per-control/full-state parity, then five alternating paired0.5s full-session timings. No tolerance relaxation or variant search.",
            "candidate": file_hash(ROOT / "tools/banc_stage_fusion_candidate.py"),
            "pointwise": file_hash(ROOT / "flylab/c/adaptive_fused.py"),
            "wall_bound_s": 480,
            "gpu_bound_s": 300,
            "include_current_production": args.current,
        },
    )
    for source, destination in (
        (ROOT / "tools/banc_stage_fusion_candidate.py", "candidate.py"),
        (Path(__file__), "probe.py"),
        (ROOT / "flylab/c/adaptive_fused.py", "pointwise.py"),
    ):
        (out / destination).write_bytes(source.read_bytes())
    sessions = []
    result = {"status": "running"}
    began = time.perf_counter()
    try:
        base_class = reference_class(
            ROOT / "verification/banc-observation-m1-20260912/baseline"
        )
        original_neural = sys.modules[base_class.__module__].AdaptiveRateNetwork

        class OriginalForChecks(original_neural):
            def advance_readout(self, drive, ids, steps):
                self.advance(drive, steps)
                return self.readout(ids)

        checks.AdaptiveRateNetwork = OriginalForChecks
        checks.ExactRKCandidate = StageFusionCandidate
        checks.primitive(out)
        checks.tanh_primitive(out)
        checks.small_network(out)
        print("primitives, original-source small networks exact", flush=True)
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
            module.AdaptiveRateNetwork = StageFusionCandidate
            candidate = module.BancWalkingSession.from_checkpoint(graph, saved)
        finally:
            module.AdaptiveRateNetwork = original
        sessions.append(candidate)
        current = None
        if args.current:
            current = module.BancWalkingSession.from_checkpoint(graph, saved)
            sessions.append(current)
        for tick in range(20):
            base.advance(1)
            candidate.advance(1)
            left, right = observe(base), observe(candidate)
            for key in left:
                if left[key].tobytes() != right[key].tobytes():
                    raise AssertionError(f"pilot control {tick + 1}: {key}")
            exact(base.snapshot(), candidate.snapshot())
        print("full 0.1s pilot exact", flush=True)
        result["pilot"] = "20 controls and complete snapshots bit exact"
        rows = []
        for pair in range(5):
            order = (
                (("baseline", base), ("current", current), ("candidate", candidate))
                if current
                else (("baseline", base), ("candidate", candidate))
            )
            for name, session in order if pair % 2 == 0 else order[::-1]:
                restore_same(session, saved)
                session.advance(10)
                restore_same(session, saved)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    session.advance(10)
                torch.cuda.synchronize()
                row = {"pair": pair, "arm": name, "wall_s": time.perf_counter() - start}
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(out / "runs.json", rows)
            exact(base.snapshot(), candidate.snapshot())
            if current:
                exact(base.snapshot(), current.snapshot())
        medians = {
            arm: statistics.median(row["wall_s"] for row in rows if row["arm"] == arm)
            for arm in (
                ("baseline", "current", "candidate")
                if current
                else ("baseline", "candidate")
            )
        }
        result.update(
            status="completed",
            validity="valid",
            medians=medians,
            reduction=1 - medians["candidate"] / medians["baseline"],
            final_exact=True,
        )
        if current:
            result.update(
                reduction_vs_current=1 - medians["candidate"] / medians["current"],
                current_reduction_vs_baseline=1
                - medians["current"] / medians["baseline"],
            )
    except AssertionError as exc:
        result.update(
            status="completed", validity="valid", hypothesis="refuted", error=str(exc)
        )
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
    parser.add_argument("--current", action="store_true")
    run(parser.parse_args())
