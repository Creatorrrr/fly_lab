"""Stimulate actual BANC motor IDs and verify native tendon control causally."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.engine import CEngine
from flylab.c.experiments import replay_recording
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, read_json, write_json
from flylab.c.ports import PortBindings
from flylab.c.storage import StateStore, runtime_versions


def run(out, graph_path, bindings_path, backend):
    out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(graph_path)
    bindings = PortBindings(graph, read_json(bindings_path))
    options = BodyOptions(
        model="flybody",
        actuation="whole_body",
        servo_profile="tracking_all",
        attachment="tethered",
        tendons="all",
    )
    report = {
        "schema": "flylab.neural-tendon-verification.v1",
        "status": "RUNNING",
        "graph_hash": graph.hash,
        "binding_hash": bindings.hash,
        "neurons": graph.n,
        "backend": backend,
        "body_options": options.model_identity(),
        "cases": [],
        "biological_validation": False,
        "seconds_per_case": 0.1,
        "interpretation": "Evoked neural-to-physical transmission only; not spontaneous walking or validated abdominal muscle anatomy",
    }
    engine = None
    started = time.perf_counter()
    try:
        engine = CEngine(
            graph, bindings, mode="C_STRICT", backend=backend, body_options=options
        )
        initial = engine.checkpoint()
        StateStore.save(out / "initial", initial)
        adapter = engine.neuromuscular.tendon_adapter
        write_json(out / "mapping.json", adapter.spec)
        report["adapter_hash"] = engine.neuromuscular.hash
        report["mapping"] = engine.neuromuscular.summary()["actuation"]
        tendon_names = adapter.control.names.copy()
        cohorts = {
            row["name"]: [
                identity for group in row["groups"] for identity in group["ids"]
            ]
            for row in adapter.rows
            if row["name"].endswith("_tarsus")
        }
        abdomen = next(row for row in adapter.rows if row["name"] == "abdomen_pitch")
        for group in abdomen["groups"]:
            cohorts["abdomen_" + group["side"]] = group["ids"]
        cohorts["abdomen_bilateral"] = (
            cohorts["abdomen_left"] + cohorts["abdomen_right"]
        )
        report["cohorts"] = cohorts
        cases = [("baseline", [], None)] + [
            (name, ids, None) for name, ids in cohorts.items()
        ]
        cases += [
            ("lf_motor_cut", cohorts["lf_tarsus"], "motor_disconnect"),
            ("lf_spike_suppression", cohorts["lf_tarsus"], "suppress_spiking"),
            ("lf_sensory_cut", cohorts["lf_tarsus"], "sensor_off"),
        ]
        write_json(out / "spec.json", report)
        traces = {}
        for name, ids, intervention in cases:
            if name != "baseline":
                engine.close()
                engine = CEngine.from_checkpoint(graph, bindings, initial)
            if ids:
                engine.schedule(
                    {
                        "kind": "stimulate",
                        "ids": ids,
                        "amplitude_mV": 30.0,
                        "duration_controls": 20,
                    }
                )
            if intervention:
                event = {"kind": intervention, "duration_controls": 20}
                if intervention == "suppress_spiking":
                    event["ids"] = ids
                if intervention == "sensor_off":
                    event["channels"] = ["leg_feedback"]
                engine.schedule(event)
            indices = graph.resolve(ids) if ids else engine.neuromuscular.motor_indices
            root = engine.body.pose()[0].copy()
            rows = []
            for _ in range(20):
                engine.step(1)
                obs = engine.body.tendon_control.observation()
                neural = engine.neural.readout(indices)
                rows.append(
                    {
                        "time_s": engine.body.physics_time(),
                        "inputs": obs["inputs"],
                        "angles_rad": obs["angles_rad"],
                        "forces": obs["actuator_forces"],
                        "rates_Hz": neural["rate_Hz"].tolist(),
                        "spikes": neural["spike_count"].tolist(),
                        "root_drift_mm": float(
                            np.linalg.norm(engine.body.pose()[0] - root)
                        ),
                    }
                )
            traces[name] = rows
            write_json(out / (name + ".json"), rows)
            if engine.fault or engine.body.fault or any(engine.body.d.warning.number):
                raise AssertionError("Fault or numerical warning: " + name)
            if any(row["root_drift_mm"] != 0 for row in rows):
                raise AssertionError("Tethered root drifted")
            report["cases"].append(
                {
                    "name": name,
                    "max_rate_Hz": max(max(r["rates_Hz"]) for r in rows),
                    "final_spikes": sum(rows[-1]["spikes"]),
                    "final_inputs": dict(zip(tendon_names, rows[-1]["inputs"])),
                    "root_drift_mm": 0.0,
                    "status": "EXECUTED",
                }
            )
            print(name + " executed", flush=True)
        checks = {}
        for name in cohorts:
            rows = traces[name]
            targets = (
                [name] if name.endswith("_tarsus") else ["abdomen_pitch", "abdomen_yaw"]
            )
            if name == "abdomen_bilateral":
                targets = ["abdomen_pitch"]
            for target in targets:
                index = tendon_names.index(target)
                sign = -1 if name == "abdomen_left" and target == "abdomen_yaw" else 1
                command = rows[-1]["inputs"][index]
                delta = (
                    np.array(rows[-1]["angles_rad"][index])
                    - traces["baseline"][-1]["angles_rad"][index]
                )
                if sign * command <= 0 or sign * float(np.mean(delta)) <= 0.001:
                    raise AssertionError(
                        f"No signed physical response: {name} -> {target}"
                    )
                checks[name + "_" + target] = {
                    "final_input": command,
                    "mean_angle_change_rad": float(np.mean(delta)),
                }
        for case in ("lf_motor_cut", "lf_spike_suppression"):
            if np.any(np.asarray([row["inputs"] for row in traces[case]]) != 0):
                raise AssertionError("Cut pathway retained a tendon command: " + case)
        if max(max(r["rates_Hz"]) for r in traces["lf_motor_cut"]) <= 0:
            raise AssertionError("Motor cut did not preserve the neural stimulus")
        if any(sum(r["spikes"]) for r in traces["lf_spike_suppression"]):
            raise AssertionError("Suppressed source neurons still spiked")
        if traces["lf_sensory_cut"][-1]["inputs"][tendon_names.index("lf_tarsus")] <= 0:
            raise AssertionError(
                "Direct motor transmission failed without sensory input"
            )
        report["signed_responses"] = checks
        report["causal_controls"] = {
            "motor_disconnect_zero": True,
            "source_suppression_zero": True,
            "direct_motor_with_sensory_cut": True,
        }
        # Resume an active neural filter and its physical model from the same boundary.
        engine.close()
        engine = CEngine.from_checkpoint(graph, bindings, initial)
        engine.schedule(
            {
                "kind": "stimulate",
                "ids": cohorts["lf_tarsus"],
                "amplitude_mV": 30.0,
                "duration_controls": 30,
            }
        )
        engine.step(10)
        active = engine.checkpoint()
        StateStore.save(out / "active", active)
        engine.step(5)
        future = engine.checkpoint()
        engine.close()
        engine = CEngine.from_checkpoint(graph, bindings, active)
        engine.step(5)
        for key in ("body", "neural", "neuromuscular"):
            np.testing.assert_equal(engine.checkpoint()[key], future[key])
        report["active_checkpoint_exact"] = True
        # Also record manual ownership and its explicit return to the neural path.
        engine.start_recording(out / "recording")
        engine.command("body_actuation", {"tendon_inputs": {"lf_tarsus": 0.04}})
        engine.step(4)
        engine.command("body_actuation", {"tendon_modes": {"lf_tarsus": "neural"}})
        engine.step(4)
        engine.stop_recording()
        replayed = replay_recording(graph, bindings, out / "recording", FlyGymBody)
        try:
            for key in ("body", "neural", "neuromuscular"):
                np.testing.assert_equal(
                    engine.checkpoint()[key], replayed.checkpoint()[key]
                )
        finally:
            replayed.close()
        report.update(
            status="PASS",
            recording_replay_exact=True,
            source_hashes={
                str(p): file_hash(p)
                for p in (
                    Path("flylab/c/tendon_neural.py"),
                    Path("flylab/c/neuromuscular.py"),
                    Path("flylab/c/engine.py"),
                    Path("flylab/body.py"),
                    Path(__file__),
                )
            },
            versions=runtime_versions(),
        )
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        report["wall_s"] = time.perf_counter() - started
        write_json(out / "report.json", report)
        if engine:
            engine.close()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "neurons",
                    "active_checkpoint_exact",
                    "recording_replay_exact",
                    "wall_s",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--graph", default="data/acquisitions/banc888-windows-20260910/bundle"
    )
    parser.add_argument(
        "--bindings",
        default="data/acquisitions/banc888-windows-20260910/bindings-neuromuscular-v2.json",
    )
    parser.add_argument("--backend", default="exp_lif_cuda")
    args = parser.parse_args()
    run(args.out, args.graph, args.bindings, args.backend)
