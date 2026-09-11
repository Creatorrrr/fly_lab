"""E7: whole-BANC rate output into the existing two-muscle force fixture."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.integrity import write_json
from flylab.c.motor_recruitment import TorqueBalancedRecruitment
from flylab.c.muscle_calibration import interior_angle
from flylab.c.muscles import MuscleRig
from flylab.c.receptors import ClawPositionTuning, JointReceptors, ReceptorParameters
from flylab.c.research_annotations import rate_weights, validate_reference
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.single_joint import anatomical_bindings, encode_feedback


def run(
    graph_path,
    rate_reference,
    out,
    *,
    claw_profile="legacy",
    descending_inputs=(0, 250),
    recruitment="common_rate",
    temporal_profile="author",
    profiles=("uniform", "partial_area"),
    annotation_reference=None,
    annotation_sha256=None,
):
    validate_reference(annotation_reference, annotation_sha256)
    if annotation_reference is not None and temporal_profile != "author":
        raise ValueError("Annotation comparison requires fixed author timing")
    if claw_profile not in ("legacy", "angle_tuned"):
        raise ValueError("Unknown claw profile")
    if recruitment not in ("common_rate", "torque_balanced"):
        raise ValueError("Unknown motor recruitment profile")
    if temporal_profile not in ("author", "fast_hypothesis"):
        raise ValueError("Unknown rate temporal profile")
    if (
        not isinstance(profiles, (tuple, list))
        or not profiles
        or any(p not in ("uniform", "partial_area") for p in profiles)
        or len(set(profiles)) != len(profiles)
    ):
        raise ValueError("Distinct declared rate profiles required")
    if (
        not isinstance(descending_inputs, (tuple, list))
        or not descending_inputs
        or any(
            type(value) is not int or value not in (0, 250)
            for value in descending_inputs
        )
        or len(set(descending_inputs)) != len(descending_inputs)
    ):
        raise ValueError("Distinct declared descending inputs required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    (out / "receptors.py").write_bytes(Path("flylab/c/receptors.py").read_bytes())
    (out / "motor_recruitment.py").write_bytes(
        Path("flylab/c/motor_recruitment.py").read_bytes()
    )
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    prior_spec = json.loads((rate_reference / "spec.json").read_text())
    prior_report = json.loads((rate_reference / "report.json").read_text())
    if prior_spec["graph_hash"] != graph.hash or prior_report["status"] != "COMPLETE":
        raise ValueError("A complete matching E6 comparison is required")
    with np.load(rate_reference / "size-alignment.npz") as data:
        sizes = data["sizes"].copy()
    binding = anatomical_bindings(graph)
    motors = graph.resolve([row["ids"][0] for row in binding["motor"]])
    ports = [
        (row["feature"], graph.resolve(row["ids"], maximum=10000))
        for row in binding["sensory"]
    ]
    dn = graph.resolve([prior_spec["stimulus_id"]])[0]
    weights, _ = rate_weights(graph)
    candidate_weights, annotation_audit = rate_weights(
        graph, reference=annotation_reference, expected_sha256=annotation_sha256
    )
    if annotation_audit is not None:
        write_json(out / "annotation-audit.json", annotation_audit)
        source_bytes = Path(annotation_reference).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != annotation_sha256:
            raise ValueError("Annotation source changed while preparing experiment")
        (out / "annotation-reference.csv").write_bytes(source_bytes)
        (out / "research_annotations.py").write_bytes(
            Path("flylab/c/research_annotations.py").read_bytes()
        )
    body_probe = MuscleRig()
    motor_calibration = (
        TorqueBalancedRecruitment(body_probe)
        if recruitment == "torque_balanced"
        else None
    )
    fast = temporal_profile == "fast_hypothesis"
    rate_tau = 0.002 if fast else 0.020
    response_tau = 0.002 if fast else 0.020
    neural_dt = 0.00002 if fast else 0.0001
    neural_steps = round(0.001 / neural_dt)
    spec = {
        "test": "E7/E10/E11/E13/E14 BANC rate hypothesis / two-MTU force response",
        "annotation_audit_hash": annotation_audit["audit_hash"]
        if annotation_audit
        else None,
        "temporal_profile": temporal_profile,
        "rate_tau_s": rate_tau,
        "receptor_response_tau_s": response_tau,
        "claw_profile": claw_profile,
        "recruitment": recruitment,
        "motor_calibration": motor_calibration.metadata if motor_calibration else None,
        "motor_calibration_hash": motor_calibration.identity
        if motor_calibration
        else None,
        "graph_hash": graph.hash,
        "simulated_neurons": graph.n,
        "body_hash": body_probe.identity,
        "unit_contract": body_probe.metadata["unit_contract"],
        "rate_reference_spec_sha256": hashlib.sha256(
            (rate_reference / "spec.json").read_bytes()
        ).hexdigest(),
        "profiles": list(profiles),
        "descending_input_units": list(descending_inputs),
        "initial_angles_rad": [1.4, 2.0],
        "torques": [0.0, 0.01, -0.01],
        "torque_interval_s": [0.05, 0.08],
        "recovery_interval_s": [0.08, 0.3],
        "seconds_per_case": 0.3,
        "sensor_gain_input_units": 250,
        "rate_to_excitation": "fixed_scale*rate/(100+rate)"
        if motor_calibration
        else "rate/(100+rate)",
        "motor_lag_controls": 1,
        "control_dt_s": 0.001,
        "neural_dt_s": neural_dt,
        "physics_dt_s": 0.00001,
        "binding": binding,
        "mechanical_teacher": False,
        "cpg": False,
        "minimum_iae_reduction": 0.20,
        "maximum_peak_ratio": 1.05,
        "wall_limit_s": 180
        if annotation_audit
        else 600
        if len(descending_inputs) == 2
        else 300,
        "model_limit_s": 3.6 * len(profiles) * len(descending_inputs),
        "biological_validation": False,
        "scope": "fixed-body left tibia, no contacts",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            Path("flylab/c/research_rate.py").read_bytes()
        ).hexdigest(),
        "receptors_sha256": hashlib.sha256(
            Path("flylab/c/receptors.py").read_bytes()
        ).hexdigest(),
    }
    write_json(out / "spec.json", spec)
    report = {
        "status": "RUNNING",
        "cases": [],
        "comparisons": [],
        "model_hashes": {},
        "full_goal_completed": False,
    }
    traces = {}
    for profile, scale in (
        ("uniform", np.ones(graph.n, np.float32)),
        ("partial_area", sizes),
    ):
        if profile not in profiles:
            continue
        network = ResearchRateNetwork(
            weights,
            np.full(graph.n, rate_tau),
            1.0 / scale,
            7.5 * scale,
            np.full(graph.n, 200),
            device="cuda",
            dt=neural_dt,
        )
        if (
            not fast
            and network.identity
            != prior_report["conditions"][profile + "-unstimulated"]["model_hash"]
        ):
            raise ValueError("Rate profile drift since E6")
        report["model_hashes"][profile] = {"before_annotation": network.identity}
        if annotation_audit is not None:
            # Verify the E6 baseline identity before constructing a distinct
            # candidate; a new annotation profile must not hide baseline drift.
            del network
            gc.collect()
            network = ResearchRateNetwork(
                candidate_weights,
                np.full(graph.n, rate_tau),
                1.0 / scale,
                7.5 * scale,
                np.full(graph.n, 200),
                device="cuda",
                dt=neural_dt,
            )
        report["model_hashes"][profile]["executed"] = network.identity
        for descending in descending_inputs:
            for angle in (1.4, 2.0):
                for feedback in (False, True):
                    for torque in (0.0, 0.01, -0.01):
                        case_start = time.perf_counter()
                        name = f"{profile}-dn{descending}-q{angle:g}-feedback{int(feedback)}-torque{torque:g}"
                        body = MuscleRig()
                        body.data.qpos[0] = angle
                        body.mj.mj_forward(body.model, body.data)
                        low, high = map(float, body.model.jnt_range[0])
                        receptors = JointReceptors(
                            ReceptorParameters(low, high, response_tau_s=response_tau),
                            claw_tuning=ClawPositionTuning()
                            if claw_profile == "angle_tuned"
                            else None,
                        )
                        network.reset()
                        rows = []
                        fault = None
                        try:
                            for tick in range(300):
                                rates = network.readout(motors)
                                excitation = (
                                    motor_calibration.encode(rates)
                                    if motor_calibration
                                    else rates / (100.0 + rates)
                                )
                                sample = receptors.step(
                                    float(body.data.qpos[0]),
                                    float(body.data.qvel[0]),
                                    enabled=feedback,
                                    **(
                                        {
                                            "interior_angle_rad": interior_angle(
                                                body.model, body.data
                                            )
                                        }
                                        if claw_profile == "angle_tuned"
                                        else {}
                                    ),
                                )
                                drive = encode_feedback(graph.n, ports, sample, 250.0)
                                drive[dn] += descending
                                network.advance(drive, neural_steps)
                                frame = body.step(
                                    excitation,
                                    steps=100,
                                    torque=torque if 50 <= tick < 80 else 0.0,
                                    direction_guard=True,
                                )
                                if (
                                    abs(body.data.time - network.tick * network.dt)
                                    > 1e-9
                                ):
                                    raise RuntimeError("Neural/physical clock mismatch")
                                rows.append(
                                    {
                                        "control_tick": tick + 1,
                                        "neural_tick": network.tick,
                                        "physics": frame,
                                        "motor_rate_sample_tick": network.tick
                                        - neural_steps,
                                        "motor_rates": rates.tolist(),
                                        "excitation": excitation.tolist(),
                                        "receptors": sample,
                                    }
                                )
                                if time.perf_counter() - began > spec["wall_limit_s"]:
                                    raise TimeoutError("E7 cumulative wall limit")
                        except RuntimeError as exc:
                            fault = str(exc)
                        write_json(out / (name + ".json"), rows)
                        np.savez_compressed(
                            out / (name + "-neural.npz"),
                            rates=network.rate.cpu().numpy(),
                            drive=network.drive.cpu().numpy(),
                            tick=network.tick,
                        )
                        key = (profile, descending, angle, feedback, torque)
                        traces[key] = rows
                        report["cases"].append(
                            {
                                "name": name,
                                "controls": len(rows),
                                "fault": fault,
                                "neural_tick": network.tick,
                                "physics_tick": body.tick,
                                "wall_s": time.perf_counter() - case_start,
                                "free_drift_rad": float(body.data.qpos[0] - angle)
                                if torque == 0
                                else None,
                            }
                        )
                        write_json(out / "report.json", report)
                        print(name, "controls", len(rows), "fault", fault, flush=True)
        del network
        gc.collect()
    adopted = []
    for profile in profiles:
        for descending in descending_inputs:
            passes = []
            for angle in (1.4, 2.0):
                for torque in (0.01, -0.01):
                    responses = {}
                    for feedback in (False, True):
                        forced = traces[(profile, descending, angle, feedback, torque)]
                        free = traces[(profile, descending, angle, feedback, 0.0)]
                        if len(forced) != 300 or len(free) != 300:
                            responses[str(feedback)] = {"complete": False}
                            continue
                        difference = np.array(
                            [r["physics"]["q_rad"] for r in forced]
                        ) - np.array([r["physics"]["q_rad"] for r in free])
                        responses[str(feedback)] = {
                            "complete": True,
                            "iae": float(np.abs(difference[80:]).sum() * 0.001),
                            "peak": float(np.abs(difference[80:]).max()),
                        }
                    on, off = responses["True"], responses["False"]
                    passed = (
                        on["complete"]
                        and off["complete"]
                        and off["iae"] > 0
                        and on["iae"] <= 0.8 * off["iae"]
                        and on["peak"] <= 1.05 * off["peak"]
                    )
                    passes.append(passed)
                    report["comparisons"].append(
                        {
                            "profile": profile,
                            "descending": descending,
                            "angle": angle,
                            "torque": torque,
                            "responses": responses,
                            "passed": bool(passed),
                        }
                    )
            if all(passes):
                adopted.append({"profile": profile, "descending": descending})
    report.update(
        status="COMPLETE",
        candidate_profiles=adopted,
        hypothesis="supported" if adopted else "refuted",
        full_goal_completed=False,
        all_cases_complete=all(
            c["controls"] == 300 and c["fault"] is None for c in report["cases"]
        ),
        neural_model_seconds=sum(c["neural_tick"] * neural_dt for c in report["cases"]),
        wall_s=time.perf_counter() - began,
    )
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("cases", "comparisons")}
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotation-reference", type=Path)
    parser.add_argument("--annotation-sha256")
    parser.add_argument(
        "--claw-profile", choices=["legacy", "angle_tuned"], default="legacy"
    )
    parser.add_argument(
        "--descending-input", type=int, nargs="+", choices=[0, 250], default=[0, 250]
    )
    parser.add_argument(
        "--recruitment",
        choices=["common_rate", "torque_balanced"],
        default="common_rate",
    )
    parser.add_argument(
        "--temporal-profile", choices=["author", "fast_hypothesis"], default="author"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["uniform", "partial_area"],
        default=["uniform", "partial_area"],
    )
    args = parser.parse_args()
    if len(set(args.descending_input)) != len(args.descending_input):
        parser.error("Descending inputs must be unique")
    run(
        args.graph,
        args.rate_reference,
        args.out,
        claw_profile=args.claw_profile,
        descending_inputs=tuple(args.descending_input),
        recruitment=args.recruitment,
        temporal_profile=args.temporal_profile,
        profiles=tuple(args.profiles),
        annotation_reference=args.annotation_reference,
        annotation_sha256=args.annotation_sha256,
    )
