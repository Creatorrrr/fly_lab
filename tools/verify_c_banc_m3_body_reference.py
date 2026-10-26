"""M3-E01: same physical FlyBody, FlyGym teacher commands, no BANC claim."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def analyze_recordings(out):
    result = json.loads((out / "physical-result.json").read_text())
    results = {}
    for name, path in (
        ("teacher", out / "teacher-trace.json"),
        ("m2", out / "m2-prefix-trace.json"),
    ):
        payload = json.loads(path.read_text())
        data = load_model(path)
        if not np.any(data["window"]):
            results[name] = {"analysis_status": "no_post_settling_samples"}
            continue
        measured, _, _ = analyze(data)
        exploratory, _, _ = analyze({**data, "moving": data["window"]})
        measured["exploratory_all_prebrake"] = exploratory
        arr = payload["samples"]
        q = np.array([r["joint_angles_rad"] for r in arr])
        target = np.array([r["joint_targets_rad"] for r in arr])
        measured["tracking_rms_rad"] = float(np.sqrt(np.mean((target - q) ** 2)))
        measured["joint_command_p05_p95_width_rad"] = (
            np.percentile(target, 95, axis=0) - np.percentile(target, 5, axis=0)
        ).tolist()
        measured["joint_actual_p05_p95_width_rad"] = (
            np.percentile(q, 95, axis=0) - np.percentile(q, 5, axis=0)
        ).tolist()
        results[name] = measured
    result["comparison"] = results
    result["analyzer_sha256"] = sha256("tools/analyze_c_banc_biological_gait.py")
    write_json(out / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "comparison"}), flush=True
    )
    for name, measured in results.items():
        print(
            json.dumps(
                {
                    "arm": name,
                    "physical": measured.get("physical"),
                    "knee_widths_deg": {
                        k: v["p95"] - v["p05"]
                        for k, v in measured.get("knee_interior_deg", {}).items()
                    },
                    "tracking_rms_rad": measured.get("tracking_rms_rad"),
                }
            ),
            flush=True,
        )


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    saved = StateStore.load(source / "start")
    original = json.loads((source / "trace.json").read_text())
    write_json(
        out / "spec.json",
        {
            "experiment": "M3-E01",
            "plan": "docs/BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md",
            "source_trace_sha256": sha256(source / "trace.json"),
            "tool_sha256": sha256(__file__),
            "hypothesis": "Identical body can express larger knee excursions under FlyBody-specific teacher commands",
            "competing_explanation": "Physical actuation/contact or template mapping blocks those excursions",
            "intervention": "Existing FlyGym hybrid teacher, forwardSpeed request 3.3, yawRate 0, controller dt .005, adhesion disabled as in FlyBody default; all tendons held .2",
            "comparator": "Same saved model and exact initial state; M2 first 3 seconds",
            "validity": "Exact initial body, contiguous .005 records, finite data; faults preserved",
            "decision": "Compare command range versus actual range and physical motion; teacher performance alone is not BANC improvement",
            "limits": "3 model seconds, no neural network execution, one body at a time",
        },
    )
    body = new_body(saved)
    began = time.perf_counter()
    try:
        body.controller = type(body.controller)(
            timestep=0.005, preprogrammed_steps=body.steps, output_dof_order=body.order
        )
        body.controller.reset(seed=saved["seed"])
        body.controller.enable_adhesion = False
        body.restore(saved["body"])
        exact(saved["body"], body.snapshot(), "M3 teacher initial body")
        tendons = {name: 0.2 for name in body.tendon_control.names}
        observer = SimpleNamespace(
            body=body,
            control_tick=0,
            rate=np.zeros(391),
            tendon_inputs=tendons,
            fault=None,
        )
        rows = [capture_control(observer)]
        drive = body.motor.map({"forwardSpeed": 3.3, "yawRate": 0.0})
        for tick in range(1, 601):
            action = body.controller.step(drive, body._observation())
            body.step_joint_targets(
                action.joint_angles,
                np.asarray(action.adhesion_onoff, bool),
                tendon_inputs=tendons,
            )
            observer.control_tick = tick
            rows.append(capture_control(observer))
            if body.fault or body.nonfoot_contact():
                break
        payload = {
            "schema": "flylab.m3-reference-teacher.v1",
            "controller": "FlyGym hybrid teacher; BANC not executed",
            "body_hash": body.model_hash,
            "model_hash": "flygym-hybrid-teacher-no-neural-model",
            "adapter_hash": "preprogrammed-flybody-steps",
            "graph_hash": None,
            "parameters": saved["parameters"],
            "samples": rows,
        }
        write_json(out / "teacher-trace.json", payload)
        subset = {**original, "samples": original["samples"][: len(rows)]}
        write_json(out / "m2-prefix-trace.json", subset)
        result = {
            "experiment": "M3-E01",
            "run_status": "completed",
            "validity": "valid",
            "hypothesis_verdict": "pending interpretation",
            "adoption": "not_applicable",
            "milestone_outcome": "pending",
            "initial_body_bit_exact": True,
            "body_hash": body.model_hash,
            "model_s": rows[-1]["time_s"],
            "wall_s": time.perf_counter() - began,
            "body_fault": body.fault,
            "nonfoot_contact": bool(body.nonfoot_contact()),
        }
        write_json(out / "physical-result.json", result)
        StateStore.save(out / "final-body", body.snapshot())
    finally:
        body.close()
    analyze_recordings(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze_recordings(args.out)
    elif args.source:
        run(args.source, args.out)
    else:
        parser.error("--source is required for physical execution")
