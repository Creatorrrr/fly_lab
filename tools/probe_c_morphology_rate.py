"""E18: full-BANC sensory transfer and decay with a frozen size hypothesis."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.muscle_calibration import interior_angle
from flylab.c.muscles import MuscleRig
from flylab.c.receptors import ClawPositionTuning, JointReceptors, ReceptorParameters
from flylab.c.research_annotations import rate_weights
from flylab.c.research_morphology import load_morphology_sizes
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(
    graph_path,
    rate_reference,
    posture_reference,
    morphology_reference,
    expected_manifest,
    out,
):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    graph = GraphStore.load(graph_path)
    candidate, fill, audit = load_morphology_sizes(
        graph, rate_reference, morphology_reference, expected_manifest
    )
    with np.load(rate_reference / "size-alignment.npz") as data:
        partial = data["sizes"].copy()
    np.savez_compressed(
        out / "size-profiles.npz", baseline=partial, candidate=candidate, fill=fill
    )
    write_json(out / "size-audit.json", audit)
    previous = json.loads((rate_reference / "report.json").read_text())
    posture = json.loads((posture_reference / "spec.json").read_text())
    if posture["graph_hash"] != graph.hash or posture["claw_profile"] != "angle_tuned":
        raise ValueError("Matching E10 reference required")
    observed = np.array([row["index"] for row in posture["observed"]], np.int64)
    if any(
        graph.nodes[i]["id"] != row["id"]
        for i, row in zip(observed, posture["observed"])
    ):
        raise ValueError("Observation root order mismatch")
    binding = anatomical_bindings(graph)
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    sensory_ids = np.unique(np.concatenate([indices for _, indices in ports]))
    body = MuscleRig()
    low, high = map(float, body.model.jnt_range[0])
    weights, _ = rate_weights(graph)
    spec = {
        "test": "E18 fixed morphology size hypothesis: recruitment and input removal",
        "graph_hash": graph.hash,
        "simulated_neurons": graph.n,
        "observed": posture["observed"],
        "fast_feti_columns": posture["fast_feti_columns"],
        "profiles": ["partial_area", "morphology_area"],
        "angles_rad": [None, 1.0, 1.4, 2.0, 2.3],
        "stimulus_controls": 200,
        "controls": 600,
        "control_dt_s": 0.001,
        "neural_dt_s": 0.0001,
        "rate_tau_s": 0.020,
        "sensory_gain": 250,
        "descending_input": 0,
        "wall_limit_s": 180,
        "neural_model_limit_s": 6,
        "physics_model_seconds": 0,
        "biological_validation": False,
        "e17_manifest_sha256": expected_manifest,
        "sources": {
            name: file_hash(Path(name))
            for name in (
                str(Path(__file__)),
                "flylab/c/research_morphology.py",
                "flylab/c/research_rate.py",
                "flylab/c/research_annotations.py",
                "flylab/c/receptors.py",
                "GOAL_PLAN.md",
            )
        },
    }
    write_json(out / "spec.json", spec)
    cases = []
    zero = np.zeros(graph.n, np.float32)
    for profile, sizes in (("partial_area", partial), ("morphology_area", candidate)):
        network = ResearchRateNetwork(
            weights,
            np.full(graph.n, 0.02),
            1 / sizes,
            7.5 * sizes,
            np.full(graph.n, 200),
            device="cuda",
        )
        if (
            profile == "partial_area"
            and network.identity
            != previous["conditions"]["partial_area-unstimulated"]["model_hash"]
        ):
            raise ValueError("E6 partial-area model identity changed")
        for angle in spec["angles_rad"]:
            network.reset()
            receptors = JointReceptors(
                ReceptorParameters(low, high), claw_tuning=ClawPositionTuning()
            )
            if angle is not None:
                body.data.qpos[0] = angle
                body.data.qvel[0] = 0
                body.mj.mj_forward(body.model, body.data)
                measured_angle = interior_angle(body.model, body.data)
            rates, inputs = [], []
            states = {"rate_at_0ms": network.readout()}
            for tick in range(spec["controls"]):
                if angle is not None and tick < spec["stimulus_controls"]:
                    sample = receptors.step(
                        angle, 0.0, interior_angle_rad=measured_angle
                    )
                    drive = encode_feedback(
                        graph.n, ports, sample, spec["sensory_gain"]
                    )
                else:
                    drive = zero
                network.advance(drive)
                rates.append(network.readout(observed))
                inputs.append(drive[sensory_ids].copy())
                if tick + 1 in (200, 400, 600):
                    states[f"rate_at_{tick + 1}ms"] = network.readout()
                if time.perf_counter() - began > spec["wall_limit_s"]:
                    raise TimeoutError("E18 cumulative wall limit")
            rates = np.asarray(rates)
            pair = rates[:, posture["fast_feti_columns"]]
            valid = bool(
                network.tick == 6000
                and np.isfinite(rates).all()
                and np.all(rates >= 0)
                and all(
                    np.isfinite(s).all() and np.all(s >= 0) for s in states.values()
                )
                and not np.count_nonzero(network.drive.cpu().numpy())
                and np.all(network.output_mask.cpu().numpy() == 1)
                and (
                    angle is not None
                    or all(not np.count_nonzero(s) for s in states.values())
                )
            )
            name = (
                f"{profile}-q{angle:g}"
                if angle is not None
                else f"{profile}-unstimulated"
            )
            # Store the actual executed sparse drives with explicit neuron IDs.
            np.savez_compressed(
                out / (name + ".npz"),
                motor_rates=rates,
                motor_rate_sample_neural_ticks=np.arange(1, 601) * 10,
                sensory_indices=sensory_ids,
                sensory_drive=np.asarray(inputs),
                final_output_mask=network.output_mask.cpu().numpy(),
                neural_tick=network.tick,
                **states,
            )
            item = {
                "name": name,
                "profile": profile,
                "angle_rad": angle,
                "valid": valid,
                "model_hash": network.identity,
                "neural_model_seconds": network.tick * network.dt,
                "stimulus_tail_fast_feti": pair[150:200]
                .astype(np.float64)
                .mean(axis=0)
                .tolist(),
                "decay_tail_fast_feti": pair[-50:]
                .astype(np.float64)
                .mean(axis=0)
                .tolist(),
                "final_active_neurons_ge_2": int((states["rate_at_600ms"] >= 2).sum()),
            }
            cases.append(item)
            write_json(out / "cases.json", cases)
            print(json.dumps(item, allow_nan=False), flush=True)
        del network
        gc.collect()
    qualification = {}
    for profile in spec["profiles"]:
        selected = [case for case in cases if case["profile"] == profile]
        angles = {case["angle_rad"]: case for case in selected}
        fast, other = angles[1.0]["stimulus_tail_fast_feti"]
        antagonist, feti = angles[2.0]["stimulus_tail_fast_feti"]
        directional = (
            fast >= 2 and other <= 0.2 * fast and feti >= 2 and antagonist <= 0.2 * feti
        )
        decayed = all(
            max(c["decay_tail_fast_feti"]) < 2
            for c in selected
            if c["angle_rad"] is not None
        )
        qualification[profile] = {
            "bidirectional_transfer": directional,
            "all_four_inputs_decay_below_2": decayed,
            "next_physical_test_qualified": directional
            and decayed
            and all(c["valid"] for c in selected),
        }
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(c["valid"] for c in cases) else "invalid",
        "cases": cases,
        "qualification": qualification,
        "neural_model_seconds": sum(c["neural_model_seconds"] for c in cases),
        "wall_seconds": time.perf_counter() - began,
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {"qualification": qualification, "wall_seconds": report["wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--rate-reference", type=Path, required=True)
    parser.add_argument("--posture-reference", type=Path, required=True)
    parser.add_argument("--morphology-reference", type=Path, required=True)
    parser.add_argument("--morphology-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(
            args.graph,
            args.rate_reference,
            args.posture_reference,
            args.morphology_reference,
            args.morphology_manifest_sha256,
            args.out,
        )
    except Exception as exc:
        if (
            args.out.exists()
            and not (args.out / "report.json").exists()
            and not (args.out / "invalid.json").exists()
        ):
            write_json(
                args.out / "invalid.json",
                {"validity": "invalid", "error": f"{type(exc).__name__}: {exc}"},
            )
        raise
