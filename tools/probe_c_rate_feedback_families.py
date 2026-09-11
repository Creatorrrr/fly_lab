"""E12: causal outgoing cuts of dynamic sensory families in the moving rig."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.motor_recruitment import TorqueBalancedRecruitment
from flylab.c.muscle_calibration import interior_angle
from flylab.c.muscles import MuscleRig
from flylab.c.receptors import ClawPositionTuning, JointReceptors, ReceptorParameters
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(graph_path, reference, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    prior = json.loads((reference / "spec.json").read_text())
    if (
        graph.hash != prior["graph_hash"]
        or prior["claw_profile"] != "angle_tuned"
        or prior["recruitment"] != "torque_balanced"
    ):
        raise ValueError("Matching E11 comparison required")
    binding = anatomical_bindings(graph)
    motors = graph.resolve([row["ids"][0] for row in binding["motor"]])
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    cuts = {}
    for mode in ("connected", "hook", "club", "hook_club"):
        families = mode.split("_")
        cuts[mode] = sorted(
            {
                int(i)
                for feature, ids in ports
                if feature.split("_")[0] in families
                for i in ids
            }
        )
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
    calibration = TorqueBalancedRecruitment(MuscleRig())
    if calibration.identity != prior["motor_calibration_hash"]:
        raise ValueError("E11 calibration drift")
    spec = {
        "test": "E12 dynamic afferent-family outgoing cuts, moving body",
        "graph_hash": graph.hash,
        "simulated_neurons": graph.n,
        "neural_model_hash": network.identity,
        "calibration_hash": calibration.identity,
        "source_spec_sha256": file_hash(reference / "spec.json"),
        "angles_rad": [1.4, 2.0],
        "controls": 600,
        "tail_controls": 150,
        "cut_ids": {
            mode: [graph.nodes[i]["id"] for i in ids] for mode, ids in cuts.items()
        },
        "model_limit_s": 4.8,
        "wall_limit_s": 180,
        "validity_max_q_error": 1e-5,
        "maximum_tail_range_ratio": 0.2,
        "maximum_final_velocity_rad_s": 1.0,
        "script_sha256": file_hash(Path(__file__)),
        "biological_validation": False,
    }
    write_json(out / "spec.json", spec)
    report = {"status": "RUNNING", "cases": [], "full_goal_completed": False}
    for angle in spec["angles_rad"]:
        old = json.loads(
            (reference / f"uniform-dn0-q{angle:g}-feedback1-torque0.json").read_text()
        )
        for mode, ids in cuts.items():
            body = MuscleRig()
            body.data.qpos[0] = angle
            body.mj.mj_forward(body.model, body.data)
            low, high = map(float, body.model.jnt_range[0])
            receptors = JointReceptors(
                ReceptorParameters(low, high), claw_tuning=ClawPositionTuning()
            )
            network.reset()
            network.set_muted(ids)
            rows = []
            fault = None
            try:
                for tick in range(spec["controls"]):
                    rates = network.readout(motors)
                    sample = receptors.step(
                        float(body.data.qpos[0]),
                        float(body.data.qvel[0]),
                        interior_angle_rad=interior_angle(body.model, body.data),
                    )
                    network.advance(encode_feedback(graph.n, ports, sample, 250.0))
                    physics = body.step(
                        calibration.encode(rates), steps=100, direction_guard=True
                    )
                    if abs(body.data.time - network.tick * network.dt) > 1e-9:
                        raise RuntimeError("Neural/physical clock mismatch")
                    rows.append(
                        {
                            "control_tick": tick + 1,
                            "motor_rates": rates.tolist(),
                            "physics": physics,
                            "receptors": sample,
                        }
                    )
                    if time.perf_counter() - began > spec["wall_limit_s"]:
                        raise TimeoutError("E12 cumulative wall limit")
            except RuntimeError as exc:
                fault = str(exc)
            name = f"q{angle:g}-{mode}"
            write_json(out / (name + ".json"), rows)
            np.savez_compressed(
                out / (name + "-neural.npz"), rates=network.readout(), tick=network.tick
            )
            q = np.array([r["physics"]["q_rad"] for r in rows])
            result = {
                "name": name,
                "angle": angle,
                "cut": mode,
                "controls": len(rows),
                "fault": fault,
                "neural_model_seconds": network.tick * network.dt,
                "physics_model_seconds": body.data.time,
                "q_range": [float(q.min()), float(q.max())],
                "q_last": float(body.data.qpos[0]),
                "velocity_last": float(body.data.qvel[0]),
                "tail_q_range": float(np.ptp(q[-spec["tail_controls"] :]))
                if len(rows) == spec["controls"]
                else None,
            }
            if mode == "connected":
                result["reference_max_q_error"] = float(
                    np.max(
                        np.abs(
                            q[:300]
                            - np.array(
                                [r["physics"]["q_rad"] for r in old[: len(q[:300])]]
                            )
                        )
                    )
                )
            report["cases"].append(result)
            write_json(out / "report.json", report)
            print(json.dumps(result), flush=True)
    valid = all(
        c.get("reference_max_q_error", 0) <= spec["validity_max_q_error"]
        for c in report["cases"]
    )
    candidates = []
    for mode in ("hook", "club", "hook_club"):
        passed = True
        for angle in spec["angles_rad"]:
            group = {c["cut"]: c for c in report["cases"] if c["angle"] == angle}
            baseline, candidate = group["connected"], group[mode]
            passed = passed and (
                candidate["controls"] == 600
                and baseline["controls"] == 600
                and candidate["fault"] is None
                and candidate["tail_q_range"] <= 0.2 * baseline["tail_q_range"]
                and abs(candidate["velocity_last"]) <= 1
            )
        if valid and passed:
            candidates.append(mode)
    report.update(
        status="COMPLETE",
        validity="valid" if valid else "invalid",
        candidates=candidates,
        wall_s=time.perf_counter() - began,
        neural_model_seconds=sum(c["neural_model_seconds"] for c in report["cases"]),
    )
    write_json(out / "report.json", report)


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
