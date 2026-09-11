"""E9: complete-BANC steady-posture transfer into all 19 tibia motor cells."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.muscle_calibration import interior_angle
from flylab.c.muscles import MuscleRig
from flylab.c.receptors import ClawPositionTuning, JointReceptors, ReceptorParameters
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(graph_path, rate_reference, out, *, claw_profile="legacy"):
    if claw_profile not in ("legacy", "angle_tuned"):
        raise ValueError("Unknown claw profile")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    prior = json.loads((rate_reference / "spec.json").read_text())
    if graph.hash != prior["graph_hash"]:
        raise ValueError("E6 graph mismatch")
    with np.load(rate_reference / "size-alignment.npz") as data:
        sizes = data["sizes"].copy()
    binding = anatomical_bindings(graph)
    observed = [
        i
        for i, n in enumerate(graph.nodes)
        if n.get("super_class") == "motor"
        and n.get("soma_side") == "left"
        and n.get("source_annotations", {}).get("body_part_effector") == "front_leg"
    ]
    labels = [
        {
            "index": i,
            "id": graph.nodes[i]["id"],
            "cell_type": graph.nodes[i].get("cell_type"),
            "target": graph.nodes[i]["source_annotations"]["peripheral_target_type"],
        }
        for i in observed
    ]
    tibia = [i for i, n in enumerate(labels) if "tibia" in n["target"]]
    if len(observed) != 69 or len(tibia) != 19:
        raise ValueError("Declared LF69/tibia19 observation scope mismatch")
    motors = graph.resolve([row["ids"][0] for row in binding["motor"]])
    motor_columns = [observed.index(i) for i in motors]
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    body = MuscleRig()
    low, high = map(float, body.model.jnt_range[0])
    weights = csr_matrix(
        (np.sign(graph.weights) * graph.counts * 0.03, graph.indices, graph.indptr),
        shape=(graph.n, graph.n),
        dtype=np.float32,
    )
    spec = {
        "test": "E9/E10 synthetic static-posture motor transfer, no physical integration",
        "claw_profile": claw_profile,
        "receptor_source_sha256": file_hash(Path("flylab/c/receptors.py")),
        "graph_hash": graph.hash,
        "simulated_neurons": graph.n,
        "body_hash": body.identity,
        "profiles": ["uniform", "partial_area"],
        "angles_rad": [1.0, 1.4, 2.0, 2.3],
        "velocity_rad_s": 0,
        "descending_input": 0,
        "sensory_gain": 250,
        "seconds_per_case": 0.2,
        "controls": 200,
        "tail_controls": 50,
        "model_limit_s": 1.6,
        "wall_limit_s": 120,
        "observed": labels,
        "tibia_columns": tibia,
        "fast_feti_columns": motor_columns,
        "source_alignment_sha256": file_hash(rate_reference / "size-alignment.npz"),
        "script_sha256": file_hash(Path(__file__)),
        "unit_contract": body.metadata["unit_contract"],
        "biological_validation": False,
    }
    write_json(out / "spec.json", spec)
    reports = []
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
        for angle in spec["angles_rad"]:
            network.reset()
            body.data.qpos[0] = angle
            body.data.qvel[0] = 0
            body.mj.mj_forward(body.model, body.data)
            geometry = body.frame()
            gains = np.array(
                [
                    body.mj.mju_muscleGain(
                        body.data.actuator_length[i],
                        0.0,
                        body.model.actuator_lengthrange[i],
                        body.model.actuator_acc0[i],
                        body.model.actuator_gainprm[i, :9],
                    )
                    for i in body.act_ids
                ]
            )
            torque_coefficients = np.array(geometry["moment_arm_mm"]) * gains
            receptors = JointReceptors(
                ReceptorParameters(low, high),
                claw_tuning=ClawPositionTuning()
                if claw_profile == "angle_tuned"
                else None,
            )
            rows, inputs = [], []
            for _ in range(spec["controls"]):
                sample = receptors.step(
                    angle,
                    0.0,
                    **(
                        {"interior_angle_rad": interior_angle(body.model, body.data)}
                        if claw_profile == "angle_tuned"
                        else {}
                    ),
                )
                network.advance(encode_feedback(graph.n, ports, sample, 250.0))
                rows.append(network.readout(observed))
                inputs.append(sample["output"])
                if time.perf_counter() - began > spec["wall_limit_s"]:
                    raise TimeoutError("E9 cumulative wall limit")
            name = f"{profile}-q{angle:g}"
            rates = np.asarray(rows)
            np.savez_compressed(
                out / (name + ".npz"),
                motor_rates=rates,
                final_neural_rates=network.readout(),
            )
            write_json(out / (name + "-inputs.json"), inputs)
            tail = rates[-spec["tail_controls"] :]
            pair = tail[:, motor_columns]
            excitation = pair / (100 + pair)
            summary = {
                "name": name,
                "profile": profile,
                "angle_rad": angle,
                "interior_angle_rad": interior_angle(body.model, body.data),
                "model_hash": network.identity,
                "model_seconds": network.tick * network.dt,
                "tail_mean": tail.mean(axis=0).tolist(),
                "tail_std": tail.std(axis=0).tolist(),
                "tail_fraction_ge_90pct_cap": (tail >= 180).mean(axis=0).tolist(),
                "fast_feti_mean": pair.mean(axis=0).tolist(),
                "torque_per_activation": torque_coefficients.tolist(),
                "active_torque_mean": float((excitation @ torque_coefficients).mean()),
                "measured_physical_movement": False,
            }
            reports.append(summary)
            print(
                name,
                summary["fast_feti_mean"],
                "active_torque",
                summary["active_torque_mean"],
                flush=True,
            )
        del network
        gc.collect()
    write_json(
        out / "report.json",
        {
            "status": "COMPLETE",
            "cases": reports,
            "neural_model_seconds": sum(r["model_seconds"] for r in reports),
            "wall_s": time.perf_counter() - began,
            "full_goal_completed": False,
            "controller_adopted": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--rate-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--claw-profile", choices=["legacy", "angle_tuned"], default="legacy"
    )
    args = parser.parse_args()
    run(args.graph, args.rate_reference, args.out, claw_profile=args.claw_profile)
