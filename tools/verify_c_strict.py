#!/usr/bin/env python3
"""Audit real C_STRICT input, motor causality, and task outcomes independently."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.backend_selection import resolve_backend
from flylab.c.behavior import trace_sample
from flylab.c.body_identity import source_identity
from flylab.c.campaign import scene_world
from flylab.c.engine import CEngine
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, read_json, write_json
from flylab.c.ports import PortBindings
from flylab.c.tasks import evaluate
from flylab.sensors import default_world

CONTROL_CRITERIA = {
    "min_direct_horizontal_net_mm": 0.1,
    "max_disconnected_horizontal_net_mm": 0.001,
    "interpretation": "Detectable response to direct neural stimulation; not a walking or navigation criterion",
}


def control_verdict(results):
    by_case = {row["case"]: row for row in results}
    needed = {"sensory_off", "direct_motor", "motor_disconnect"}
    if not needed <= by_case.keys():
        return {"status": "NOT_RUN", "criteria": CONTROL_CRITERIA}
    off, direct, cut = (
        by_case[key] for key in ("sensory_off", "direct_motor", "motor_disconnect")
    )
    passed = all(row["technical_status"] == "PASS" for row in (off, direct, cut))
    passed = (
        passed
        and off["total_spikes"] == 0
        and off["sensory_off_verified"]
        and direct["motor_spikes"]["forward"] > 0
    )
    passed = (
        passed
        and direct["horizontal_net_mm"]
        >= CONTROL_CRITERIA["min_direct_horizontal_net_mm"]
        and cut["motor_spikes"]["forward"] > 0
    )
    passed = (
        passed
        and cut["max_abs_command"] == 0
        and cut["horizontal_net_mm"]
        <= CONTROL_CRITERIA["max_disconnected_horizontal_net_mm"]
    )
    return {"status": "PASS" if passed else "FAIL", "criteria": CONTROL_CRITERIA}


def run_case(graph, binding, case, seconds, backend, out):
    world = (
        scene_world(case) if case in ("food_left", "food_right") else default_world()
    )
    if case in ("named_food", "food_upwind", "named_food_upwind"):
        world = scene_world("food_left")
    if case.endswith("_upwind"):
        # Fixed upstream source for the profiles' +x wind, not a steering command.
        world["sources"][0]["p"][0] = -8.0
    if case.startswith("named_food"):
        # Explicit diagnostic odorant, never reinterpret generic hazard as geosmin.
        world["sources"][0]["odorant"] = binding.spec["sensor_model"]["chemical_odor"][
            "odorants"
        ][0]["inchikey"]
    started = time.perf_counter()
    engine = None
    try:
        engine = CEngine(
            graph, binding, mode="C_STRICT", seed=42, world=world, backend=backend
        )
        n = round(seconds / 0.005)
        if case == "sensory_off":
            engine.schedule(
                {"kind": "sensor_off", "channels": ["*"], "duration_controls": n}
            )
        if case in ("direct_motor", "motor_disconnect"):
            engine.schedule(
                {
                    "kind": "stimulate",
                    "ids": binding.spec["motor"]["forward"]["ids"],
                    "amplitude_mV": 12.0,
                    "duration_controls": n,
                }
            )
        if case == "motor_disconnect":
            engine.schedule({"kind": "motor_disconnect", "duration_controls": n})
        frame = engine.frame()
        trace = [trace_sample(frame)]
        trace[0]["sensory_ports"] = []
        trace[0]["sensor_diagnostics"] = {}
        initial = np.array(frame["body"]["position"])
        max_rates = {key: 0.0 for key in binding.motor}
        max_command = 0.0
        peak_input_features = {}
        motor_causality = True
        sensory_off_verified = True
        for _ in range(n):
            frame = engine.step(1)
            for (port, _), value in zip(binding.sensory, frame["sensory_ports"]):
                channel = port["channel"]
                peak_input_features[channel] = max(
                    peak_input_features.get(channel, 0.0), abs(value["feature"])
                )
            row = trace_sample(frame)
            # Full ports retained at the final boundary; avoid duplicating 1122 ports per tick.
            row["sensory_ports"] = []
            row["sensor_diagnostics"] = {}
            row["motor_rates_Hz"] = frame["motor_rates_Hz"]
            trace.append(row)
            for key, rate in frame["motor_rates_Hz"].items():
                max_rates[key] = max(max_rates[key], rate)
            cmd = frame["command"]
            max_command = max(max_command, *(abs(v) for v in cmd["u_final"].values()))
            expected = (
                "motor_disconnect" if case == "motor_disconnect" else "connectome_lif"
            )
            motor_causality &= (
                cmd["command_source"] == expected
                and cmd["u_assist"] is None
                and cmd["u_legacy"] is None
            )
            motor_causality &= cmd["u_final"] == (
                {"forwardSpeed": 0.0, "yawRate": 0.0, "verticalSpeed": 0.0}
                if case == "motor_disconnect"
                else cmd["u_neural"]
            )
            if case == "sensory_off":
                sensory_off_verified &= all(
                    not p["enabled"] and p["value"] == 0 for p in frame["sensory_ports"]
                )
            if frame["fault"]:
                break
        neural = engine.neural.snapshot()
        channels = {}
        for (port, ids), value in zip(binding.sensory, frame["sensory_ports"]):
            row = channels.setdefault(
                port["channel"],
                {"ports": 0, "target_connections": 0, "max_input": 0.0, "spike_sum": 0},
            )
            row["ports"] += 1
            row["target_connections"] += len(ids)
            row["max_input"] = max(row["max_input"], abs(value["value"]))
            # A neuron may appear in several ports; this is explicitly a connection sum.
            row["spike_sum"] += int(neural["spike_count"][ids].sum())
        result = {
            "case": case,
            "seconds": seconds,
            "seed": 42,
            "physical": True,
            "backend": backend,
            "elapsed_wall_s": time.perf_counter() - started,
            "fault": frame["fault"],
            "completed": engine.control_tick == n,
            "mode": engine.mode,
            "graph_hash": graph.hash,
            "binding_hash": binding.hash,
            "model_parameter_hash": engine.parameters.hash,
            "body_model_hash": frame["physics"]["bodyModelHash"],
            "world": copy.deepcopy(world),
            "strict_causality": motor_causality,
            "sensory_off_verified": sensory_off_verified
            if case == "sensory_off"
            else None,
            "total_spikes": int(neural["spike_count"].sum()),
            "max_motor_rates_Hz": max_rates,
            "max_abs_command": max_command,
            "channels": channels,
            "peak_input_features": peak_input_features,
            "motor_spikes": {
                key: int(neural["spike_count"][ids].sum())
                for key, (_, ids) in binding.motor.items()
            },
            "horizontal_net_mm": float(
                np.linalg.norm((np.array(frame["body"]["position"]) - initial)[[0, 2]])
            ),
            "chemical_diagnostics": frame["sensor_diagnostics"].get("chemical_odor"),
            "final_ports": frame["sensory_ports"],
            "trace": trace,
            "task": evaluate(
                trace,
                world,
                "food"
                if case
                in (
                    "food_left",
                    "food_right",
                    "named_food",
                    "food_upwind",
                    "named_food_upwind",
                )
                else "diagnostic",
                required_seconds=seconds,
            ),
            "biological_validation": False,
        }
        result["technical_status"] = (
            "PASS"
            if result["completed"]
            and not result["fault"]
            and motor_causality
            and sensory_off_verified
            else "FAIL"
        )
    except Exception as exc:  # noqa: BLE001 - preserve each native failure and continue the bounded audit
        result = {
            "case": case,
            "technical_status": "FAIL",
            "error": str(exc),
            "elapsed_wall_s": time.perf_counter() - started,
        }
    finally:
        if engine:
            engine.close()
    write_json(out / (case + ".json"), result)
    return {
        k: v
        for k, v in result.items()
        if k not in ("final_ports", "trace", "world", "chemical_diagnostics", "task")
    } | {
        "task_status": result.get("task", {}).get("task_status", "INCOMPLETE"),
        "unlabelled_sources": (result.get("chemical_diagnostics") or {}).get(
            "unlabelled_sources", []
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph")
    p.add_argument("--bindings-dir")
    p.add_argument("--profiles", nargs="+")
    p.add_argument(
        "--cases",
        nargs="+",
        default=["default", "sensory_off", "direct_motor", "motor_disconnect"],
        choices=[
            "default",
            "sensory_off",
            "direct_motor",
            "motor_disconnect",
            "food_left",
            "food_right",
            "named_food",
            "food_upwind",
            "named_food_upwind",
        ],
    )
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--backend", default="auto")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (
        not 0.005 <= a.seconds <= 30
        or abs(a.seconds / 0.005 - round(a.seconds / 0.005)) > 1e-8
    ):
        p.error("Use an integral control duration between .005 and 30 seconds")
    local = read_json(Path(__file__).resolve().parents[1] / "flylab.local.json")
    directory = Path(a.bindings_dir or Path(local["bindings"]).parent)
    paths = (
        [directory / name for name in a.profiles]
        if a.profiles
        else sorted(directory.glob("bindings*.json"))
    )
    if not paths:
        p.error("No binding profiles found")
    graph = GraphStore.load(a.graph or local["graph"])
    backend = resolve_backend(a.backend)
    a.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "flylab.strict-audit.v1",
        "scope": "Independent real-body experiments; live session unchanged",
        "backend": backend,
        "graph_hash": graph.hash,
        "seconds": a.seconds,
        "cases": a.cases,
        "tool_sha256": file_hash(Path(__file__)),
        "source": source_identity(),
        "profiles": [],
    }
    for path in paths:
        binding = PortBindings(graph, read_json(path))
        if binding.spec.get("neuromuscular"):
            raise ValueError(
                "This assay covers descending-to-CPG profiles; BANC needs joint-specific positive controls"
            )
        output = a.out / path.stem
        output.mkdir()
        row = {
            "file": path.name,
            "profile": binding.spec.get("profile"),
            "binding_hash": binding.hash,
            "results": [],
        }
        report["profiles"].append(row)
        for case in a.cases:
            result = run_case(graph, binding, case, a.seconds, backend, output)
            row["results"].append(result)
            write_json(a.out / "report.json", report)
            print(
                json.dumps(
                    dict(
                        profile=path.name,
                        **{
                            k: result.get(k)
                            for k in (
                                "case",
                                "technical_status",
                                "error",
                                "total_spikes",
                                "max_abs_command",
                                "horizontal_net_mm",
                                "elapsed_wall_s",
                                "unlabelled_sources",
                            )
                        },
                    )
                ),
                flush=True,
            )
        row["controls"] = control_verdict(row["results"])
    report["technical_status"] = (
        "PASS"
        if all(
            r["technical_status"] == "PASS"
            for row in report["profiles"]
            for r in row["results"]
        )
        else "FAIL"
    )
    controls = [row["controls"]["status"] for row in report["profiles"]]
    report["control_status"] = (
        "FAIL"
        if "FAIL" in controls
        else "NOT_RUN"
        if all(v == "NOT_RUN" for v in controls)
        else "PASS"
    )
    report["behavioral_approval"] = False
    write_json(a.out / "report.json", report)
    return (
        0
        if report["technical_status"] == "PASS" and report["control_status"] != "FAIL"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
