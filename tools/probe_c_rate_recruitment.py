"""Bounded CSR reference check and whole-BANC rate size-assumption probe."""

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_rate import ResearchRateNetwork


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def reference_candidate(reference, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with np.load(reference / "inputs.npz") as data:
        network = ResearchRateNetwork(
            csr_matrix(data["weights"]),
            data["tau"],
            data["a"],
            data["threshold"],
            data["fr_cap"],
            device="cuda",
        )
        times = data["time_s"].copy()
    write(
        out / "spec.json",
        {
            "test": "E5 sparse CSR candidate on preserved author reference inputs",
            "reference": str(reference),
            "inputs_sha256": hashlib.sha256(
                (reference / "inputs.npz").read_bytes()
            ).hexdigest(),
            "kernel_sha256": hashlib.sha256(
                Path("flylab/c/research_rate.py").read_bytes()
            ).hexdigest(),
            "model_hash": network.identity,
            "device": network.device,
            "model_limit_s": 4.0,
            "wall_limit_s": 120,
        },
    )
    report = {"status": "RUNNING", "conditions": {}, "kernel": "CSR CUDA RK4"}
    for name, value in (("unstimulated", 0), ("stimulated", 250)):
        began = time.perf_counter()
        network.reset()
        drive = np.zeros(network.n, np.float32)
        trace = np.zeros((network.n, 2001), np.float32)
        for tick in range(2000):
            if tick == 20:
                drive[31] = value
            if tick == 1999:
                drive.fill(0)
            network.advance(drive)
            trace[:, tick + 1] = network.readout()
            if time.perf_counter() - start > 120:
                raise TimeoutError("Sparse reference unit wall limit")
        np.savez_compressed(out / f"cuda-{name}.npz", raw=trace, time_s=times)
        report["conditions"][name] = {
            "raw_finite": bool(np.isfinite(trace).all()),
            "last_time_s": network.tick * network.dt,
            "raw_min": float(trace.min()),
            "raw_max": float(trace.max()),
            "wall_s": time.perf_counter() - began,
        }
        write(out / "cuda-report.json", report)
        print(name, json.dumps(report["conditions"][name]), flush=True)
    report.update(status="COMPLETE", wall_s=time.perf_counter() - start)
    write(out / "cuda-report.json", report)


def banc(graph_path, metadata_path, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    graph = GraphStore.load(graph_path)
    with metadata_path.open(encoding="utf8", newline="") as file:
        rows = list(csv.DictReader(file))
    areas = {}
    for row in rows:
        root_id = row["pt_root_id"]
        if root_id in areas:
            raise ValueError(
                "Duplicate author root ID; refuse positional/ambiguous join"
            )
        try:
            value = float(row["surf_area_um2"])
        except ValueError:
            value = float("nan")
        areas[root_id] = value
    aligned = np.array(
        [areas.get(node["id"].rsplit(":", 1)[-1], np.nan) for node in graph.nodes]
    )
    valid = np.isfinite(aligned) & (aligned > 0)
    median = float(np.median(aligned[valid]))
    sizes = np.ones(graph.n, np.float32)
    sizes[valid] = aligned[valid] / median
    if int(valid.sum()) != 4747:
        raise ValueError("Unexpected area coverage for fixed author/current graph pair")
    # Historical E6 input: left soma, RIGHT VNC target. Preserve its exact
    # intervention; use probe_c_banc_descending for explicit target-side tests.
    dn = [
        i
        for i, n in enumerate(graph.nodes)
        if n.get("cell_type") == "DNg100"
        and n.get("soma_side") == "left"
        and n["id"].rsplit(":", 1)[-1] == "720575941626500746"
    ]
    if len(dn) != 1:
        raise ValueError("Historical E6 DNg100 (left soma/right VNC) identity mismatch")
    lf = np.array(
        [
            i
            for i, n in enumerate(graph.nodes)
            if n.get("super_class") == "motor"
            and n.get("soma_side") == "left"
            and n.get("source_annotations", {}).get("body_part_effector") == "front_leg"
        ],
        dtype=np.int32,
    )
    tibia = np.array(
        [
            j
            for j, i in enumerate(lf)
            if graph.nodes[i]
            .get("source_annotations", {})
            .get("peripheral_target_type")
            in (
                "tibia_flexor_muscle",
                "tibia_extensor_muscle",
                "accessory_tibia_flexor_muscle",
            )
        ],
        dtype=np.int32,
    )
    if len(lf) != 69 or len(tibia) != 19:
        raise ValueError("Fixed motor readout coverage mismatch")
    # Preserve the graph's zero/NT policies, replace the explicitly identified
    # contact-to-input-unit multiplier for this separately labelled rate model.
    weights = csr_matrix(
        (np.sign(graph.weights) * graph.counts * 0.03, graph.indices, graph.indptr),
        shape=(graph.n, graph.n),
        dtype=np.float32,
    )
    spec = {
        "test": "E6 complete BANC rate-model size-assumption comparison",
        "graph_hash": graph.hash,
        "neurons": graph.n,
        "retained_pairs": len(graph.counts),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "size_measured_nodes": int(valid.sum()),
        "size_assumed_one_nodes": int((~valid).sum()),
        "normalization_area_median_um2": median,
        "stimulus_id": graph.nodes[dn[0]]["id"],
        "stimulus_soma_side": "left",
        "stimulus_target_vnc_side": "right",
        "stimulus_input_units": 250,
        "stimulus_start_s": 0.020,
        "stimulus_end_s": 1.999,
        "tau_s": 0.020,
        "a_before_size": 1.0,
        "threshold_before_size": 7.5,
        "cap": 200.0,
        "weight_multiplier": 0.03,
        "weight_unit": "rate-model input per anatomical contact; not mV",
        "graph_nt_and_zero_policy_preserved": True,
        "dt_s": 0.0001,
        "saved_interval_s": 0.001,
        "lf_motor_ids": [graph.nodes[i]["id"] for i in lf],
        "tibia_indices_in_lf": tibia.tolist(),
        "active_threshold": 0.01,
        "wall_limit_s": 600,
        "model_limit_s": 8,
        "kernel_sha256": hashlib.sha256(
            Path("flylab/c/research_rate.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "author_banc_reproduction": False,
        "biological_validation": False,
    }
    write(out / "spec.json", spec)
    np.savez_compressed(
        out / "size-alignment.npz", sizes=sizes, measured=valid, area_um2=aligned
    )
    report = {"status": "RUNNING", "conditions": {}, "full_goal_completed": False}
    for profile, scale in (
        ("uniform", np.ones(graph.n, np.float32)),
        ("partial_area", sizes),
    ):
        network = ResearchRateNetwork(
            weights,
            np.full(graph.n, 0.020),
            1.0 / scale,
            7.5 * scale,
            np.full(graph.n, 200),
            device="cuda",
        )
        for name, value in (("unstimulated", 0), ("stimulated", 250)):
            label = profile + "-" + name
            began = time.perf_counter()
            network.reset()
            drive = np.zeros(graph.n, np.float32)
            trace = np.zeros((69, 2001), np.float32)
            incoming = np.zeros_like(trace)
            dn_trace = np.zeros(2001, np.float32)
            for tick in range(2000):
                if tick == 20:
                    drive[dn[0]] = value
                if tick == 1999:
                    drive.fill(0)
                network.advance(drive)
                trace[:, tick + 1] = network.readout(lf)
                dn_trace[tick + 1] = network.readout(dn)[0]
                with network.torch.inference_mode():
                    current = network.torch.sparse.mm(
                        network.weights, network.rate.unsqueeze(1)
                    ).squeeze(1)
                    incoming[:, tick + 1] = current[lf].cpu().numpy()
                if time.perf_counter() - start > 600:
                    raise TimeoutError("E6 cumulative wall limit")
            state = network.snapshot()
            np.savez_compressed(
                out / f"{label}.npz",
                rates=trace,
                incoming_rate_current=incoming,
                dn_rates=dn_trace,
                final_rates=state["rate"],
                time_s=np.arange(2001) * 0.001,
            )
            mask = trace > 0.01
            maxima = np.max(trace, axis=1)
            onset = [
                float(np.flatnonzero(x)[0] * 0.001) if x.any() else None for x in mask
            ]
            condition = {
                "valid": bool(
                    np.isfinite(state["rate"]).all() and network.tick == 20000
                ),
                "all_neurons_computed": network.n,
                "model_hash": network.identity,
                "active_lf_motors": int(np.count_nonzero(maxima > 0.01)),
                "active_tibia_motors": int(np.count_nonzero(maxima[tibia] > 0.01)),
                "motor_max_rates": maxima.tolist(),
                "onset_s": onset,
                "active_sample_duration_s": (mask.sum(axis=1) * 0.001).tolist(),
                "all_network_zero": bool(np.count_nonzero(state["rate"]) == 0),
                "whole_network_max_rate": float(state["rate"].max()),
                "dn_max_rate": float(dn_trace.max()),
                "wall_s": time.perf_counter() - began,
            }
            report["conditions"][label] = condition
            write(out / "report.json", report)
            print(
                label,
                json.dumps(
                    {
                        k: v
                        for k, v in condition.items()
                        if k
                        not in (
                            "motor_max_rates",
                            "onset_s",
                            "active_sample_duration_s",
                        )
                    }
                ),
                flush=True,
            )
        del network
        gc.collect()
    report.update(
        status="COMPLETE",
        validity="valid",
        wall_s=time.perf_counter() - start,
        neural_model_seconds=8,
        full_goal_completed=False,
    )
    write(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "banc"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.phase == "reference":
        if args.reference is None:
            parser.error("reference requires --reference")
        reference_candidate(args.reference, args.out)
    else:
        if args.metadata is None:
            parser.error("banc requires --metadata")
        banc(args.graph, args.metadata, args.out)
