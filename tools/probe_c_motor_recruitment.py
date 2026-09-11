"""E2: isolated FeCO cohort recruitment on the unchanged complete BANC graph."""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import write_json
from flylab.c.neural import LIFParameters, create_backend
from flylab.c.receptors import JointReceptors, ReceptorParameters
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(graph_path, reference_profile, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(graph_path)
    profile = json.loads(Path(reference_profile).read_text(encoding="utf-8"))
    binding = anatomical_bindings(graph)
    if profile["graph_hash"] != graph.hash or profile["bindings"] != binding:
        raise ValueError("Reference graph or anatomical bindings mismatch")
    parameters = LIFParameters(**profile["neural_parameters"])
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
    tibia = [
        j
        for j, i in enumerate(lf)
        if graph.nodes[i]["source_annotations"].get("peripheral_target_type")
        in ("tibia_flexor_muscle", "tibia_extensor_muscle")
    ]
    ports = [
        (r["feature"], graph.resolve(r["ids"], maximum=10000))
        for r in binding["sensory"]
    ]
    cohorts = {
        r["cell_type"]: graph.resolve(r["ids"], maximum=10000)
        for r in binding["sensory"]
    }
    sensory = np.unique(np.concatenate([ids for _, ids in ports]))
    cases = [
        {
            "name": cell + ("_cut" if cut else ""),
            "cell": cell,
            "cut": cut,
            "direct": False,
        }
        for cell in ("SNpp50", "SNpp51", "SNpp41", "SNpp39")
        for cut in (False, True)
    ]
    cases += [
        {
            "name": "q2.2_gain36" + ("_cut" if cut else ""),
            "cell": None,
            "cut": cut,
            "direct": False,
        }
        for cut in (False, True)
    ]
    cases += [
        {
            "name": "direct_" + r["cell_type"],
            "ids": r["ids"],
            "cell": None,
            "cut": False,
            "direct": True,
        }
        for r in binding["motor"]
    ]
    spec = {
        "schema": "flylab.motor-recruitment.v1",
        "graph_hash": graph.hash,
        "parameters": asdict(parameters),
        "seconds_per_case": 0.3,
        "per_cell_drive_mV": 20.0,
        "cases": cases,
        "observed_nodes": [graph.nodes[int(i)] for i in lf],
        "tibia_observed_indices": tibia,
        "record_interval_s": 0.001,
        "spike_time_resolution_s": parameters.dt,
        "physical_executed": False,
        "backend": "exp_lif_cuda",
        "wall_limit_s": 600,
        "purpose": "Separate cohort recruitment from downstream muscle hypotheses; GOAL_PLAN E2 v1",
        "adoption": "Only a sensory candidate, if direct controls respond and matching sensory-cut removes tibia recruitment",
    }
    write_json(out / "spec.json", spec)
    write_json(out / "graph_manifest.json", graph.manifest)
    report = {
        "status": "RUNNING",
        "validity": "pending",
        "hypothesis": "not_tested",
        "adoption": "defer",
        "milestone": "pending",
        "physical_executed": False,
        "biological_validation": False,
        "cases": [],
    }
    write_json(out / "report.json", report)
    started = time.perf_counter()
    try:
        for case in cases:
            if time.perf_counter() - started > spec["wall_limit_s"]:
                raise TimeoutError("E2 shared wall limit reached")
            directory = out / case["name"]
            directory.mkdir()
            neural = create_backend(graph, parameters, "exp_lif_cuda")
            receptor = JointReceptors(
                ReceptorParameters(**profile["receptor_parameters"])
            )
            drive = np.zeros(graph.n, np.float32)
            stimulated = (
                graph.resolve(case["ids"])
                if case["direct"]
                else cohorts[case["cell"]]
                if case["cell"]
                else sensory
            )
            if case["cut"]:
                neural.set_interventions(mute=stimulated.tolist())
            if case["cell"] or case["direct"]:
                drive[stimulated] = 20.0
            rows, events = [], []
            capture = np.unique(np.concatenate([lf, stimulated])).astype(np.int32)
            lf_device = neural.xp.asarray(lf)
            for k in range(300):
                if k % 25 == 0 and time.perf_counter() - started > spec["wall_limit_s"]:
                    raise TimeoutError("E2 shared wall limit reached")
                if not case["cell"] and not case["direct"]:
                    drive = encode_feedback(
                        graph.n, ports, receptor.step(2.2, 0.0), 36.0
                    )
                neural.advance(drive, round(0.001 / parameters.dt), capture=capture)
                read = neural.readout(lf)
                rows.append(
                    [
                        read["voltage_mV"],
                        neural.host(neural.h[lf_device]),
                        read["rate_Hz"],
                        read["spike_count"],
                    ]
                )
                events.extend(neural.last_events)
            state = neural.snapshot()
            np.savez_compressed(
                directory / "motor_trace.npz",
                samples=np.asarray(rows),
                ticks=np.arange(1, 301) * round(0.001 / parameters.dt),
                observed_indices=lf,
            )
            np.savez_compressed(
                directory / "final_neural.npz",
                **{key: state[key] for key in ("v", "h", "rate", "spike_count")},
            )
            write_json(directory / "events.json", events)
            counts = state["spike_count"][lf]
            result = dict(
                **case,
                neural_ticks=neural.tick,
                simulated_neurons=graph.n,
                stimulated_neurons=len(stimulated),
                motor_spike_counts=counts.tolist(),
                tibia_spikes=int(counts[tibia].sum()),
                lf_spikes=int(counts.sum()),
                active_lf_motors=int(np.count_nonzero(counts)),
                active_tibia_motors=int(np.count_nonzero(counts[tibia])),
                stimulated_spikes=int(state["spike_count"][stimulated].sum()),
                global_spikes=int(state["spike_count"].sum()),
                backend=state["backend"],
            )
            write_json(directory / "result.json", result)
            report["cases"].append(result)
            write_json(out / "report.json", report)
            print(
                case["name"],
                "tibia",
                result["tibia_spikes"],
                "LF",
                result["lf_spikes"],
                flush=True,
            )
            del neural, state, lf_device
            gc.collect()
        direct_ok = all(
            row["stimulated_spikes"] > 0 for row in report["cases"] if row["direct"]
        )
        candidate = []
        for i in range(0, 8, 2):
            intact, cut = report["cases"][i : i + 2]
            if intact["tibia_spikes"] > 0 and cut["tibia_spikes"] == 0:
                candidate.append(intact["cell"])
        report.update(
            status="COMPLETE",
            validity="valid" if direct_ok else "invalid",
            hypothesis="supported"
            if direct_ok and candidate
            else "refuted"
            if direct_ok
            else "not_tested",
            adoption="candidate_for_dynamic_test"
            if direct_ok and candidate
            else "reject_cohort_only"
            if direct_ok
            else "defer",
            candidate_cohorts=candidate,
            positive_controls_pass=direct_ok,
            simulated_neurons=graph.n,
            simulated_seconds=3.6,
            interpretation="Recruitment alone does not demonstrate muscle or behavioral control; held-drive cohort test only",
        )
    except Exception as exc:  # noqa: BLE001 -- Preserve partial evidence; CLI reports failure.
        report.update(status="FAILED", validity="invalid", error=repr(exc))
        if "rows" in locals():
            np.savez_compressed(
                directory / "partial_motor_trace.npz", samples=np.asarray(rows)
            )
            write_json(directory / "partial_events.json", events)
    report["wall_s"] = time.perf_counter() - started
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", default="data/acquisitions/banc888-windows-20260910/bundle"
    )
    parser.add_argument(
        "--reference-profile",
        default="verification/flygym-completion-20260910/joint-voltage-events-01/q_2.2_gain_36/profile.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.graph, args.reference_profile, args.out)
    print(result["status"], result["hypothesis"])
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)
