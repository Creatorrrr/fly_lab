"""NW1: bounded full-BANC adaptation probe before physical integration."""

import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.graph import GraphStore
from flylab.c.integrity import write_json
from flylab.c.neuromuscular import build_spec
from flylab.c.research_annotations import rate_weights


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    weights, audit = rate_weights(graph)
    muscles = build_spec(graph, 2)["rows"]
    dn = np.array([i for i, n in enumerate(graph.nodes) if n["cell_type"] == "DNg100"])
    if len(dn) != 2:
        raise ValueError("Expected bilateral annotated DNg100 input cohort")
    motor_ids = sorted(
        {
            i
            for row in muscles
            for muscle in row["muscles"]
            for i in graph.resolve(muscle["ids"])
        }
    )
    local = {value: i for i, value in enumerate(motor_ids)}
    drive = np.zeros(graph.n, np.float32)
    drive[dn] = 400.0
    gains = (0.0, 4.0, 20.0)
    write_json(
        args.out / "spec.json",
        {
            "experiment": "NW1",
            "hypothesis": "Continuous activity adaptation may sustain motor variation under held descending input",
            "competing_explanation": "Adaptation yields another tonic or globally synchronous state without six-leg rhythms",
            "graph_hash": graph.hash,
            "all_nodes": graph.n,
            "anatomical_edges": len(graph.indices),
            "weights": audit,
            "gains": gains,
            "tau_s": 0.15,
            "dn_ids": [graph.nodes[i]["id"] for i in dn],
            "held_dn_input": 400.0,
            "neural_seconds_per_case": 3.0,
            "wall_limit_per_case_s": 180.0,
            "dt": 0.0001,
            "record_dt": 0.005,
            "parameter_status": "Engineering hypothesis, not cell-specific physiological calibration",
            "decision": "Require sustained last-2s tibia opponent variation >2 rate units on all six legs before physical adoption; no gait success from this probe",
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path("flylab/c/adaptive_rate.py"))
            },
            "cpg": False,
            "timed_reset": False,
            "phase_stimulus": False,
            "physical_executed": False,
        },
    )
    for gain in gains:
        began = time.perf_counter()
        network = AdaptiveRateNetwork(
            weights,
            *[np.full(graph.n, value, np.float32) for value in (0.02, 1.0, 7.5, 200.0)],
            adaptation_gain=gain,
        )
        trace = []
        for _ in range(600):
            if time.perf_counter() - began > 180:
                raise TimeoutError("Bounded neural probe exceeded 180 seconds")
            network.advance(drive, 50)
            trace.append(network.readout(motor_ids))
        rates = np.asarray(trace)
        signals, summary = [], []
        for row in muscles:
            sides = []
            for names in (
                ("tibia_flexor_muscle", "accessory_tibia_flexor_muscle"),
                ("tibia_extensor_muscle",),
            ):
                ids = [
                    local[i]
                    for m in row["muscles"]
                    if m["muscle"] in names
                    for i in graph.resolve(m["ids"])
                ]
                sides.append(rates[:, ids].mean(axis=1))
            diff = sides[0] - sides[1]
            signals.append(diff)
            summary.append(
                {
                    "leg": row["leg"],
                    "opponent_std": float(diff[200:].std()),
                    "opponent_range": float(np.ptp(diff[200:])),
                    "flexor_mean": float(sides[0][200:].mean()),
                    "extensor_mean": float(sides[1][200:].mean()),
                }
            )
        np.savez_compressed(
            args.out / f"gain-{gain:g}.npz",
            time_s=np.arange(1, 601) * 0.005,
            rates=rates,
            motor_ids=np.array([graph.nodes[i]["id"] for i in motor_ids]),
            opponent=np.array(signals).T,
            final_rates=network.readout(),
            adaptation=network.adaptation.detach().cpu().numpy(),
        )
        report = {
            "gain": gain,
            "model_hash": network.identity,
            "seconds": 3.0,
            "wall_s": time.perf_counter() - began,
            "finite": bool(np.isfinite(rates).all()),
            "legs": summary,
            "all_six_variable": all(r["opponent_std"] > 2 for r in summary),
            "physical_executed": False,
            "walking_verified": False,
        }
        write_json(args.out / f"gain-{gain:g}.json", report)
        print(report, flush=True)
        del network
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
