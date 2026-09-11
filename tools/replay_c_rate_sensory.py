"""E8: isolate static and dynamic afferent inputs using a preserved E7 trace."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import write_json
from flylab.c.receptors import CHANNELS, JointReceptors, ReceptorParameters
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(graph_path, reference, out):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    trace_path = reference / "uniform-dn0-q1.4-feedback1-torque0.json"
    original = json.loads(trace_path.read_text())[:80]
    prior_spec = json.loads((reference / "spec.json").read_text())
    prior_report = json.loads((reference / "report.json").read_text())
    graph = GraphStore.load(graph_path)
    binding = anatomical_bindings(graph)
    if (
        prior_spec["graph_hash"] != graph.hash
        or prior_report["status"] != "COMPLETE"
        or binding != prior_spec["binding"]
        or len(original) != 80
        or [row["control_tick"] for row in original] != list(range(1, 81))
    ):
        raise ValueError("Complete matching E7 trace and binding required")
    motors = graph.resolve([row["ids"][0] for row in binding["motor"]])
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    weights = csr_matrix(
        (np.sign(graph.weights) * graph.counts * 0.03, graph.indices, graph.indptr),
        shape=(graph.n, graph.n),
        dtype=np.float32,
    )
    network = ResearchRateNetwork(
        weights,
        np.full(graph.n, 0.020),
        np.ones(graph.n),
        np.full(graph.n, 7.5),
        np.full(graph.n, 200),
        device="cuda",
    )
    spec = {
        "test": "E8 static/dynamic sensory replay, no physical simulation",
        "source_trace_sha256": sha256(trace_path),
        "source_spec_sha256": sha256(reference / "spec.json"),
        "script_sha256": sha256(Path(__file__)),
        "kernel_sha256": sha256(Path("flylab/c/research_rate.py")),
        "graph_hash": graph.hash,
        "simulated_neurons": graph.n,
        "model_hash": network.identity,
        "device": network.device,
        "conditions": ["replay", "fixed_posture", "angle_only", "reverse_velocity"],
        "controls_per_case": 80,
        "control_dt_s": 0.001,
        "neural_dt_s": network.dt,
        "model_limit_s": 0.32,
        "wall_limit_s": 120,
        "validity_max_rate_error": 1e-4,
        "validity_max_receptor_error": 1e-12,
        "main_contribution_ratios": [0.20, 0.80],
        "sensory_gain": 250.0,
        "rate_units": "author rate-model output, not an observed spike count",
        "descending_input": 0,
        "biological_validation": False,
        "full_goal_completed": False,
    }
    write_json(out / "spec.json", spec)
    reports = {}
    rate_reference = np.asarray([row["motor_rates"] for row in original])
    receptor_reference = np.asarray(
        [[row["receptors"]["output"][c] for c in CHANNELS] for row in original]
    )
    low, high = original[0]["physics"]["joint_range_rad"]
    for condition in spec["conditions"]:
        receptors = JointReceptors(ReceptorParameters(low, high))
        if receptors.identity != original[0]["receptors"]["profile_hash"]:
            raise ValueError("Receptor implementation drift since E7")
        network.reset()
        rows = []
        for tick, source in enumerate(original):
            rates = network.readout(motors)
            stimulus = source["receptors"]["stimulus"]
            q, velocity = stimulus["q_rad"], stimulus["velocity_rad_s"]
            if condition == "fixed_posture":
                q, velocity = 1.4, 0.0
            elif condition == "angle_only":
                velocity = 0.0
            elif condition == "reverse_velocity":
                velocity = -velocity
            sample = receptors.step(q, velocity)
            network.advance(encode_feedback(graph.n, ports, sample, 250.0))
            rows.append(
                {
                    "control_tick": tick + 1,
                    "motor_rate_sample_tick": network.tick - 10,
                    "motor_rates": rates.tolist(),
                    "receptors": sample,
                }
            )
            if time.perf_counter() - began > spec["wall_limit_s"]:
                raise TimeoutError("E8 cumulative wall limit")
        write_json(out / (condition + ".json"), rows)
        final = network.readout()
        np.savez_compressed(out / (condition + "-neural.npz"), rates=final)
        values = np.asarray([row["motor_rates"] for row in rows])
        summary = {
            "controls": len(rows),
            "model_seconds": network.tick * network.dt,
            "rate_integral": (values.sum(axis=0) * 0.001).tolist(),
            "rate_max": values.max(axis=0).tolist(),
            "rate_final_sample": values[-1].tolist(),
            "onset_gt_1_ms": [
                int(np.flatnonzero(column > 1)[0]) if np.any(column > 1) else None
                for column in values.T
            ],
            "all_neurons_finite": bool(np.isfinite(final).all()),
        }
        if condition == "replay":
            observed = np.asarray(
                [[row["receptors"]["output"][c] for c in CHANNELS] for row in rows]
            )
            summary.update(
                max_rate_error=float(np.max(np.abs(values - rate_reference))),
                max_receptor_error=float(np.max(np.abs(observed - receptor_reference))),
            )
        reports[condition] = summary
        print(condition, json.dumps(summary), flush=True)
    valid = (
        reports["replay"]["max_rate_error"] <= spec["validity_max_rate_error"]
        and reports["replay"]["max_receptor_error"]
        <= spec["validity_max_receptor_error"]
        and all(r["all_neurons_finite"] for r in reports.values())
    )
    baseline = reports["replay"]["rate_integral"][1]
    for condition, result in reports.items():
        result["FETi_integral_ratio_to_replay"] = (
            result["rate_integral"][1] / baseline if baseline > 0 else None
        )
    write_json(
        out / "report.json",
        {
            "status": "COMPLETE",
            "validity": "valid" if valid else "invalid",
            "cases": reports,
            "wall_s": time.perf_counter() - began,
            "neural_model_seconds": sum(r["model_seconds"] for r in reports.values()),
            "physical_model_seconds": 0,
            "full_goal_completed": False,
            "controller_adopted": False,
        },
    )
    if not valid:
        raise RuntimeError("E8 replay validity failed; interventions uninterpretable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.graph, args.reference, args.out)
