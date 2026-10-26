"""M3-E06: anatomical muscle-command expressivity, with no neural gait claim.

The known teacher commands are labels in this body-only experiment. Nonnegative
muscle rates, a fixed anatomical moment matrix, and a 10 ms activation filter
must express them before they are suitable neural-training objectives.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def run(source, identification, teacher, out, resume=False):
    out.mkdir(parents=True, exist_ok=resume)
    spec = {
        "experiment": "M3-E06",
        "hypothesis": "Fixed anatomical muscle directions plus a calibrated constant rest posture can express useful teacher gait without a sign-changing or clock-based decoder",
        "competing_explanation": "Missing/coupled actuation directions or finite muscle rate/activation constraints still prevent it",
        "arms": ["original-rest", "calibrated-rest"],
        "intervention": "Teacher target inverse fit to nonnegative muscle group rates in [0,200]; fixed .02 rad/rate-unit moment scale; 10ms activation; no .5rad tanh; same original physical servo limits. Calibrated arm sets single-sign row rest at training-label extreme, unsupported row rest at training mean; bidirectional rows retain original neutral.",
        "calibration_window_s": [.5, 1.2],
        "validation_window_s": [1.5, 2.2],
        "replay": "2.2s, same exact initial M2 body; teacher labels each control, original teacher tendon inputs and no adhesion; neural network not run",
        "validity": "Nonnegative bounded rates, finite arrays, constant anatomical signs/rest, complete contiguous physical record",
        "screen": "Validation 42-joint command RMS <=.05rad, physical net displacement>=10mm, upright>=.9, no fault/nonfoot contact; useful feasibility only, not M3 adoption",
        "limits": "2 arms x 2.2model sec CPU, <=120wall sec each, no parameter search",
        "tool_sha256": sha256(__file__),
        "signals_sha256": sha256(identification / "signals.npz"),
        "teacher_sha256": sha256(teacher),
    }
    if resume:
        previous = json.loads((out / "spec.json").read_text())
        if any(previous[k] != spec[k] for k in ("signals_sha256", "teacher_sha256")):
            raise ValueError("Resume source mismatch")
        write_json(out / "recovery.json", {
            "reason": "Physical first arm saved before analysis metadata error; resume that analysis and collect only missing second arm",
            "tool_sha256": spec["tool_sha256"],
            "replayed_completed_arms": False,
        })
    else:
        write_json(out / "spec.json", spec)
    d = np.load(identification / "signals.npz")
    collection = json.loads((identification / "collection-result.json").read_text())
    rows = json.loads(teacher.read_text())["samples"][1:441]
    saved = StateStore.load(source / "start")
    moment = np.column_stack([d["moment"][:, g["columns"][0]] for g in collection["muscles"]])
    # Constant calibration uses training labels only; no held-out extrema.
    train = (d["time_s"] >= .5) & (d["time_s"] < 1.2)
    valid = (d["time_s"] >= 1.5) & (d["time_s"] < 2.2)
    target = d["target"]
    alpha = 1 - math.exp(-.005 / .01)
    results = {}
    for arm in ("original-rest", "calibrated-rest"):
        arm_out = out / arm
        if resume and (arm_out / "physical-result.json").exists():
            for name in ("signals.npz", "trace.json", "final-body"):
                if not (arm_out / name).exists():
                    raise ValueError("Incomplete physical arm cannot be resumed")
            continue
        arm_out.mkdir()
        rest = d["neutral"].copy()
        if arm == "calibrated-rest":
            for j, v in enumerate(moment):
                v = v[v != 0]
                if not len(v):
                    rest[j] = np.mean(target[train, j])
                elif np.all(v < 0):
                    rest[j] = np.max(target[train, j])
                elif np.all(v > 0):
                    rest[j] = np.min(target[train, j])
        body = new_body(saved)
        began = time.perf_counter()
        try:
            exact(saved["body"], body.snapshot(), "E06 initial body")
            activation = np.zeros(moment.shape[1])
            observer = SimpleNamespace(body=body, control_tick=0,
                rate=np.zeros(391), tendon_inputs={}, fault=None)
            trace = [capture_control(observer)]
            commands, muscle_rates, errors = [], [], []
            with (arm_out / "trace.jsonl").open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(trace[0]) + "\n")
                for tick, row in enumerate(rows, 1):
                    if time.perf_counter() - began > 120:
                        raise TimeoutError("E06 wall limit")
                    # The only optimized quantity is a nonnegative input rate.
                    residual = target[tick - 1] - rest - .02 * moment @ ((1-alpha) * activation)
                    fit = lsq_linear(.02 * alpha * moment, residual,
                        bounds=(0, 200), method="bvls", tol=1e-8, max_iter=300)
                    if not fit.success:
                        np.savez_compressed(arm_out / "solver-failure.npz",
                            residual=residual, moment=moment, solution=fit.x,
                            activation=activation)
                        write_json(arm_out / "solver-failure.json", {
                            "tick": tick, "message": fit.message,
                            "status": int(fit.status), "optimality": float(fit.optimality),
                            "cost": float(fit.cost), "physical_steps": tick-1,
                            "wall_s": time.perf_counter()-began,
                            "validity": "invalid_incomplete_solver_execution",
                            "hypothesis_verdict": "inconclusive",
                        })
                        StateStore.save(arm_out / "failure-body", body.snapshot())
                        raise ValueError("Nonnegative muscle solver failed")
                    rates = fit.x
                    activation += alpha * (rates - activation)
                    command = rest + .02 * moment @ activation
                    if not np.isfinite(command).all() or np.any(rates < 0) or np.any(rates > 200):
                        raise ValueError("Invalid muscle command")
                    observer.tendon_inputs = row["tendon_inputs"]
                    body.step_joint_targets(command, np.zeros(6, bool),
                        tendon_inputs=observer.tendon_inputs)
                    observer.control_tick = tick
                    trace.append(capture_control(observer))
                    stream.write(json.dumps(trace[-1]) + "\n")
                    commands.append(command.copy())
                    muscle_rates.append(rates.copy())
                    errors.append(command - target[tick - 1])
                    if body.fault or body.nonfoot_contact():
                        break
            physical = {
                "run_status": "completed", "validity": "valid_body_only_diagnostic",
                "model_s": observer.control_tick * .005,
                "native_time_s": body.d.time, "wall_s": time.perf_counter()-began,
                "fault": body.fault, "nonfoot_contact": bool(body.nonfoot_contact()),
                "initial_body_bit_exact": True, "neural_steps": 0,
                "external_teacher": True, "body_hash": body.model_hash,
            }
            write_json(arm_out / "physical-result.json", physical)
            StateStore.save(arm_out / "final-body", body.snapshot())
            np.savez_compressed(arm_out / "signals.npz", rest=rest, moment=moment,
                rates=muscle_rates, command=commands, error=errors,
                time_s=d["time_s"][:len(commands)])
            write_json(arm_out / "trace.json", {
                "schema": "flylab.m3-actuation-space.v1", "body_hash": body.model_hash,
                "controller": "teacher labels through anatomical nonnegative muscle fit",
                "model_hash": "no-neural-model", "adapter_hash": arm,
                "graph_hash": saved["graph_hash"], "samples": trace,
                "parameters": {"rest": rest.tolist(), "moment_scale": .02,
                    "activation_tau_s": .01, "rate_cap": 200},
            })
        finally:
            body.close()
    for arm in ("original-rest", "calibrated-rest"):
        arm_out = out / arm
        signal = np.load(arm_out / "signals.npz")
        physical = json.loads((arm_out / "physical-result.json").read_text())
        payload = json.loads((arm_out / "trace.json").read_text())
        trace_path = arm_out / "trace.json"
        if "parameters" not in payload:
            # Version the metadata repair; the original physical samples stay intact.
            payload["parameters"] = {"rest": signal["rest"].tolist(),
                "moment_scale": .02, "activation_tau_s": .01, "rate_cap": 200}
            trace_path = arm_out / "trace-analysis-v2.json"
            write_json(trace_path, payload)
        measured, _, _ = analyze(load_model(trace_path))
        error = signal["error"]
        mask = valid[:len(error)]
        rms = float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
        report = {**physical, "model_s": len(error)*.005,
            "native_time_s": physical.get("native_time_s", physical["model_s"]),
            "quality": measured, "validation_command_rms_rad": rms,
            "per_joint_validation_rms_rad": np.sqrt(np.mean(error[mask]**2, axis=0)).tolist()
                if mask.any() else None,
            "adoption": "not_applicable", "milestone_outcome": "pending"}
        write_json(arm_out / "result.json", report)
        results[arm] = report
        print(json.dumps({"arm": arm, "rms_rad": rms, **physical,
            "model_s": len(error)*.005,
            "physical": measured["physical"]}), flush=True)
    write_json(out / "result.json", {"experiment": "M3-E06", "arms": results,
        "adoption": "not_applicable", "milestone_outcome": "pending"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--identification", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.source, a.identification, a.teacher, a.out, a.resume)
