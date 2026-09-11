"""E15: distinguish autonomous recurrent activity from sensory/rate decay."""

import argparse
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
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(
    graph_path,
    rate_reference,
    posture_reference,
    out,
    *,
    start_mode="replay",
    cut_profile="global",
):
    if start_mode not in ("replay", "saved_state"):
        raise ValueError("Unknown E15 start mode")
    if cut_profile not in ("global", "regional"):
        raise ValueError("Unknown outgoing-cut profile")
    if cut_profile == "regional" and start_mode != "saved_state":
        raise ValueError("Regional comparisons require the fixed source state")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    graph = GraphStore.load(graph_path)
    prior = json.loads((posture_reference / "spec.json").read_text())
    rate_prior = json.loads((rate_reference / "report.json").read_text())
    if prior["graph_hash"] != graph.hash or prior["claw_profile"] != "angle_tuned":
        raise ValueError("Matching E10 angle-tuned reference required")
    observed = np.array([row["index"] for row in prior["observed"]], np.int64)
    if any(
        graph.nodes[i]["id"] != row["id"] for i, row in zip(observed, prior["observed"])
    ):
        raise ValueError("Observation roster changed")
    weights, _ = rate_weights(graph)
    network = ResearchRateNetwork(
        weights,
        np.full(graph.n, 0.02),
        np.ones(graph.n),
        np.full(graph.n, 7.5),
        np.full(graph.n, 200),
        device="cuda",
    )
    if (
        network.identity
        != rate_prior["conditions"]["uniform-unstimulated"]["model_hash"]
    ):
        raise ValueError("E6 uniform network drift")
    binding = anatomical_bindings(graph)
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    body = MuscleRig()
    low, high = map(float, body.model.jnt_range[0])
    cut_indices = {"none": np.array([], np.int64)}
    if cut_profile == "regional":
        regions = np.array(
            [n.get("source_annotations", {}).get("region") for n in graph.nodes],
            dtype=object,
        )
        for name, members in (
            ("brain", ("central_brain", "optic_lobe")),
            ("vnc", ("ventral_nerve_cord",)),
            ("central_brain", ("central_brain",)),
            ("optic_lobe", ("optic_lobe",)),
        ):
            cut_indices[name] = np.flatnonzero(np.isin(regions, members))
    cut_indices["all_outgoing"] = np.arange(graph.n, dtype=np.int64)
    spec = {
        "test": "E16 regional contribution to autonomous rate persistence"
        if cut_profile == "regional"
        else "E15 post-stimulus autonomous rate persistence",
        "start_mode": start_mode,
        "cut_profile": cut_profile,
        "outgoing_cut_counts": {name: len(ids) for name, ids in cut_indices.items()},
        "cut_selection": "Graph source_annotations.region; unclassified cells excluded from regional cuts",
        "prefix_controls_replayed": 200 if start_mode == "replay" else 0,
        "source_snapshot_seconds": 0.2,
        "graph_hash": graph.hash,
        "model_hash": network.identity,
        "simulated_neurons": graph.n,
        "observed": prior["observed"],
        "tibia_columns": prior["tibia_columns"],
        "fast_feti_columns": prior["fast_feti_columns"],
        "angles_rad": [1.0, 2.0],
        "stimulus_controls": 200,
        "zero_input_controls": 400,
        "control_dt_s": 0.001,
        "neural_dt_s": network.dt,
        "tau_s": 0.02,
        "sensory_gain": 250,
        "descending_input": 0,
        "cuts_after_stimulus": list(cut_indices),
        "wall_limit_s": 180
        if start_mode == "replay" or cut_profile == "regional"
        else 120,
        "model_limit_s": len(cut_indices) * (1.2 if start_mode == "replay" else 0.8),
        "physics_model_seconds": 0,
        "posture_reference_spec_sha256": file_hash(posture_reference / "spec.json"),
        "source_sha256": {
            str(path): file_hash(path)
            for path in (
                Path(__file__),
                Path("flylab/c/receptors.py"),
                Path("flylab/c/research_rate.py"),
                Path("flylab/c/research_annotations.py"),
            )
        },
        "biological_validation": False,
    }
    write_json(out / "spec.json", spec)
    np.savez_compressed(out / "cut-selection.npz", **cut_indices)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    reports = []
    zero = np.zeros(graph.n, np.float32)
    for angle in spec["angles_rad"]:
        body.data.qpos[0] = angle
        body.data.qvel[0] = 0
        body.mj.mj_forward(body.model, body.data)
        anatomical_angle = interior_angle(body.model, body.data)
        with np.load(posture_reference / f"uniform-q{angle:g}.npz") as data:
            reference = data["motor_rates"].copy()
            source_state = data["final_neural_rates"].copy()
        if reference.shape != (200, len(observed)):
            raise ValueError("E10 rate trace shape mismatch")
        if (
            source_state.shape != (graph.n,)
            or source_state.dtype != np.float32
            or not np.isfinite(source_state).all()
            or np.any(source_state < 0)
            or not np.array_equal(source_state[observed], reference[-1])
        ):
            raise ValueError("E10 full state and observed terminal rates disagree")
        for cut in spec["cuts_after_stimulus"]:
            network.reset()
            network.set_muted([])
            receptors = JointReceptors(
                ReceptorParameters(low, high), claw_tuning=ClawPositionTuning()
            )
            rates, analytic_errors = [], []
            initial = None
            first_control = 0 if start_mode == "replay" else 200
            restore_error = None
            if start_mode == "saved_state":
                staged = network.snapshot()
                staged.update(rate=source_state.copy(), tick=2000)
                # The external drive is removed at this boundary. Rate is the
                # entire dynamic state of this declared first-order model.
                network.restore(staged)
                restore_error = float(np.max(np.abs(network.readout() - source_state)))
            for tick in range(first_control, 600):
                if tick < 200:
                    sample = receptors.step(
                        angle, 0.0, interior_angle_rad=anatomical_angle
                    )
                    drive = encode_feedback(graph.n, ports, sample, 250.0)
                else:
                    drive = zero
                    if tick == 200:
                        initial = network.readout().astype(np.float64)
                        network.set_muted(cut_indices[cut])
                network.advance(drive)
                rates.append(network.readout(observed))
                if cut == "all_outgoing" and tick in (219, 299, 599):
                    expected = initial * np.exp(-((tick + 1 - 200) * 0.001) / 0.02)
                    error = float(np.max(np.abs(network.readout() - expected)))
                    analytic_errors.append(
                        {"control_tick": tick + 1, "max_error": error}
                    )
                if time.perf_counter() - started > spec["wall_limit_s"]:
                    raise TimeoutError("E15 cumulative wall limit")
            rates = np.asarray(rates)
            final = network.readout()
            pre_error = (
                float(np.max(np.abs(rates[:200] - reference)))
                if start_mode == "replay"
                else None
            )
            clock_valid = network.tick == 6000
            analytic_valid = all(row["max_error"] <= 2e-4 for row in analytic_errors)
            initial_valid = (
                pre_error <= 1e-4 if start_mode == "replay" else restore_error == 0
            )
            valid = initial_valid and analytic_valid and clock_valid
            summaries = []
            for column in prior["tibia_columns"]:
                prefix = rates[:200] if start_mode == "replay" else reference
                before = float(prefix[150:200, column].mean())
                after = float(rates[-50:, column].mean())
                ratio = after / before if before >= 2 else None
                summaries.append(
                    {
                        "id": prior["observed"][column]["id"],
                        "cell_type": prior["observed"][column]["cell_type"],
                        "stimulus_tail_mean": before,
                        "zero_input_tail_mean": after,
                        "tail_ratio": ratio,
                        "persistence": "not_tested"
                        if not valid or ratio is None
                        else "supported"
                        if ratio >= 0.2
                        else "refuted"
                        if ratio < 0.01
                        else "inconclusive",
                    }
                )
            name = f"q{angle:g}-{cut}"
            np.savez_compressed(
                out / (name + ".npz"),
                motor_rates=rates,
                motor_rate_sample_neural_ticks=np.arange(first_control + 1, 601) * 10,
                rates_at_input_removal=initial,
                final_rates=final,
                final_input=network.drive.cpu().numpy(),
                final_output_mask=network.output_mask.cpu().numpy(),
                neural_tick=network.tick,
            )
            reports.append(
                {
                    "name": name,
                    "angle_rad": angle,
                    "cut": cut,
                    "muted_neurons": len(cut_indices[cut]),
                    "valid": valid,
                    "baseline_pre_error": pre_error,
                    "source_state_restore_error": restore_error,
                    "source_snapshot_sha256": file_hash(
                        posture_reference / f"uniform-q{angle:g}.npz"
                    ),
                    "cut_decay_checks": analytic_errors,
                    "tibia": summaries,
                    "final_active_neurons_above_2": int((final >= 2).sum()),
                    "input_removed": bool(
                        np.count_nonzero(network.drive.cpu().numpy()) == 0
                    ),
                    "executed_controls": 600 - first_control,
                    "neural_seconds": (600 - first_control) * 0.001,
                    "final_neural_clock_seconds": network.tick * network.dt,
                }
            )
            write_json(out / "cases.json", reports)
            print(
                name,
                "valid",
                valid,
                "active",
                reports[-1]["final_active_neurons_above_2"],
                flush=True,
            )
    valid = all(row["valid"] and row["input_removed"] for row in reports)
    report = {
        "status": "COMPLETE",
        "validity": "valid" if valid else "invalid",
        "cases": reports,
        "neural_model_seconds": sum(row["neural_seconds"] for row in reports),
        "wall_seconds": time.perf_counter() - started,
        "implementation_adoption": "not_applicable",
        "F3_F6": "fail",
        "full_goal_completed": False,
    }
    write_json(out / "report.json", report)
    print(
        spec["test"],
        report["validity"],
        "wall seconds",
        report["wall_seconds"],
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--start-mode", choices=["replay", "saved_state"], default="replay"
    )
    parser.add_argument(
        "--cut-profile", choices=["global", "regional"], default="global"
    )
    args = parser.parse_args()
    run(
        args.graph,
        args.rate_reference,
        args.posture_reference,
        args.out,
        start_mode=args.start_mode,
        cut_profile=args.cut_profile,
    )
