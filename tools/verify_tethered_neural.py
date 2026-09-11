"""Whole BANC circuit on a tethered FlyBody: intact, sensory cut and motor drive."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(out, graph_path, bindings_path):
    from flylab.body import FlyGymBody
    from flylab.body_options import BodyOptions
    from flylab.c.engine import CEngine
    from flylab.c.experiments import replay_recording
    from flylab.c.graph import GraphStore
    from flylab.c.integrity import write_json
    from flylab.c.ports import PortBindings
    from flylab.c.storage import StateStore

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(graph_path)
    bindings = PortBindings(
        graph, json.loads(Path(bindings_path).read_text(encoding="utf-8"))
    )
    options = BodyOptions(
        model="flybody",
        actuation="whole_body",
        servo_profile="tracking_all",
        attachment="tethered",
        tendons="all",
    )
    report = {
        "schema": "flylab.tethered-neural-verification.v1",
        "status": "RUNNING",
        "graph_hash": graph.hash,
        "bindings_hash": bindings.hash,
        "neurons": graph.n,
        "backend": "exp_lif_cuda",
        "body_options": options.model_identity(),
        "seconds_per_condition": 0.1,
        "tendon_neural_mapping": "not_initialized",
        "biological_validation": False,
        "cases": [],
    }
    write_json(out / "spec.json", report)
    start = time.perf_counter()
    engine = None
    try:
        engine = CEngine(
            graph,
            bindings,
            mode="C_STRICT",
            backend="exp_lif_cuda",
            body_options=options,
        )
        initial = engine.checkpoint()
        report["tendon_neural_mapping"] = engine.body.tendon_control.observation()[
            "neural_mapping"
        ]
        report["tendon_validation_tool"] = (
            "tools/verify_c_tendon_neural.py; this assay directly stimulates the tibia flexor"
        )
        StateStore.save(out / "initial", initial)
        # Use reviewed peripheral target annotations already in the decoder.
        ids = [
            graph.nodes[int(i)]["id"]
            for name, indices, _ in engine.neuromuscular.muscles[0]
            if name == "tibia_flexor_muscle"
            for i in indices
        ]
        if not ids:
            raise ValueError("No annotated LF tibia flexor motors")
        report["direct_motor_ids"] = ids
        motor_indices = engine.neuromuscular.motor_indices
        traces = {}
        for condition in ("intact", "sensory_cut", "motor_direct"):
            if condition != "intact":
                engine.close()
                engine = CEngine.from_checkpoint(graph, bindings, initial)
            if condition == "sensory_cut":
                engine.schedule(
                    {
                        "kind": "sensor_off",
                        "channels": ["leg_feedback"],
                        "duration_controls": 20,
                    }
                )
            if condition == "motor_direct":
                engine.schedule(
                    {
                        "kind": "stimulate",
                        "ids": ids,
                        "amplitude_mV": 30.0,
                        "duration_controls": 20,
                    }
                )
            root = engine.body.pose()[0].copy()
            rows = []
            for _ in range(20):
                engine.step(1)
                read = engine.neural.readout(motor_indices)
                rows.append(
                    {
                        "time_s": engine.body.physics_time(),
                        "angles_rad": engine.body.joint_observation()[
                            "angles_rad"
                        ].tolist(),
                        "motor_rate_Hz": read["rate_Hz"].tolist(),
                        "motor_spikes": read["spike_count"].tolist(),
                        "root_drift_mm": float(
                            np.linalg.norm(engine.body.pose()[0] - root)
                        ),
                        "body_fault": engine.body.fault,
                        "engine_fault": engine.fault,
                    }
                )
            write_json(out / (condition + ".json"), rows)
            traces[condition] = rows
            report["cases"].append(
                {
                    "condition": condition,
                    "steps": engine.control_tick,
                    "max_root_drift_mm": max(r["root_drift_mm"] for r in rows),
                    "max_motor_rate_Hz": max(max(r["motor_rate_Hz"]) for r in rows),
                    "fault": engine.fault or engine.body.fault,
                    "motion_expected": engine.motion_expected,
                }
            )
        # Checkpoint + recorded nonleg/tendon commands must follow engine time.
        engine.close()
        engine = CEngine.from_checkpoint(graph, bindings, initial)
        engine.start_recording(out / "recording")
        tick = engine.tick
        engine.command(
            "body_actuation",
            {
                "targets": {"c_thorax-c_head-yaw": 0.1},
                "tendon_inputs": {"lf_tarsus": 0.1},
            },
        )
        if engine.tick != tick:
            raise AssertionError("Held command advanced the engine clock")
        engine.step(4)
        engine.stop_recording()
        repeated = replay_recording(graph, bindings, out / "recording", FlyGymBody)
        try:
            if engine.body.snapshot() != repeated.body.snapshot():
                raise AssertionError(
                    "Recorded physical commands did not replay exactly"
                )
            np.testing.assert_equal(
                engine.neural.snapshot(), repeated.neural.snapshot()
            )
        finally:
            repeated.close()
        baseline = np.asarray([r["angles_rad"] for r in traces["intact"]])
        report.update(
            status="COMPLETE",
            recording_replayed=True,
            maximum_angle_difference_rad={
                condition: float(
                    np.max(
                        np.abs(np.asarray([r["angles_rad"] for r in rows]) - baseline)
                    )
                )
                for condition, rows in traces.items()
            },
            wall_s=time.perf_counter() - start,
            interpretation="Tethered diagnostic contrasts only; angle differences do not establish beneficial sensory feedback or walking",
        )
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write_json(out / "report.json", report)
        if engine:
            engine.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    p.add_argument(
        "--bindings",
        type=Path,
        default=Path(
            "data/acquisitions/banc888-windows-20260910/bindings-neuromuscular-v2.json"
        ),
    )
    a = p.parse_args()
    run(a.out, a.graph, a.bindings)
