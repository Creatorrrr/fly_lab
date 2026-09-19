"""Posthoc command-range and load-threshold diagnostics on verified recordings.

Replays only recorded physical commands. It changes neither BANC nor actuator
limits, checks every physical future, and separates braking from moving samples.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from flylab.c.walking_trace import contact_diagnostics
from tools.analyze_c_banc_observation import new_body, validate_recording


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "spec.json",
        {
            "source": str(args.source),
            "source_hash": file_hash(Path(__file__)),
            "question": "Do recorded commands exceed native joint/actuator limits, and does support threshold alter observed slip?",
            "design": "Posthoc descriptive replay of all recorded commands with 0.01/0.02/0.05 BW thresholds; no gait candidate or causal adoption",
            "validity": "Every integration state must match the original recording byte for byte",
            "stop": "One source, up to its recorded duration; abort on mismatch",
        },
    )
    payload = json.loads((args.source / "trace.json").read_text(encoding="utf-8"))
    saved = StateStore.load(args.source / "start")
    validate_recording(payload, saved)
    rows = payload["samples"]
    signals = np.load(args.source / "recorded-signals.npz")
    body = new_body(saved)
    try:
        limits = body.m.jnt_range[body.joint_ids]
        limited = body.m.jnt_limited[body.joint_ids].astype(bool)
        control_limits = body.m.actuator_ctrlrange[body.act_ids]
        control_limited = body.m.actuator_ctrllimited[body.act_ids].astype(bool)
        typ = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, typ))
        records = []
        for index, row in enumerate(rows[1:]):
            request = (row.get("neural_request") or {}).get("external_force_at_start")
            if request:
                body.push_left = request["push_left_s"]
                body.push_force = np.asarray(request["push_force"])
            body.step_joint_targets(
                np.asarray(row["joint_targets_rad"]),
                np.asarray(row["adhesion"], bool),
                tendon_inputs=row["tendon_inputs"],
                interpolate=not (row.get("neural_request") or {}).get("cuts", {}).get("motor", False),
                **({'joint_stiffness_scale': row['joint_stiffness_scale']} if 'joint_stiffness_scale' in row else {}),
                **({'adhesion_control': row['adhesion_control']} if 'adhesion_control' in row else {}),
            )
            body.mj.mj_getState(body.m, body.d, state, typ)
            if state.tobytes() != signals["physical"][index].tobytes():
                raise AssertionError(f"Physical replay differs at row {index + 1}")
            obs = body.joint_observation()
            feet = contact_diagnostics(body, support_threshold_bw=0.01)
            cue = (row.get("neural_request") or {}).get("navigation_at_start")
            records.append(
                {
                    "braking": bool(cue and cue["brake_request"]),
                    "targets": row["joint_targets_rad"],
                    "errors": row["joint_tracking_error_rad"],
                    "constraint": obs["constraint_torque"],
                    "actuator": obs["actuator_torque"],
                    "feet": feet,
                }
            )
        result = {
            "validity": "valid",
            "command_replay_bit_exact": True,
            "samples": len(records),
            "phases": {},
        }
        for phase in ("all", "moving_request", "braking_request"):
            selected = [
                r
                for r in records
                if phase == "all" or r["braking"] == (phase == "braking_request")
            ]
            if not selected:
                continue
            targets = np.asarray([r["targets"] for r in selected])
            errors = np.asarray([r["errors"] for r in selected])
            constraint = np.asarray([r["constraint"] for r in selected])
            load = np.asarray([r["feet"]["normal_load_bw"] for r in selected])
            slip = np.asarray(
                [
                    [np.nan if x is None else x for x in r["feet"]["slip_mm_s"]]
                    for r in selected
                ]
            )
            exceed = (
                np.maximum(
                    np.maximum(limits[:, 0] - targets, targets - limits[:, 1]), 0
                )
                * limited
            )
            ctrl_exceed = (
                np.maximum(
                    np.maximum(
                        control_limits[:, 0] - targets, targets - control_limits[:, 1]
                    ),
                    0,
                )
                * control_limited
            )
            rms = np.sqrt(np.mean(errors**2, axis=0))
            joints = []
            for i in np.argsort(-rms)[:6]:
                joints.append(
                    {
                        "name": body.joint_names[i],
                        "rms_rad": float(rms[i]),
                        "target_range_rad": [
                            float(targets[:, i].min()),
                            float(targets[:, i].max()),
                        ],
                        "joint_limited": bool(limited[i]),
                        "joint_range_rad": limits[i].tolist(),
                        "outside_joint_range_fraction": float(
                            np.mean(exceed[:, i] > 0)
                        ),
                        "max_joint_range_excess_rad": float(exceed[:, i].max()),
                        "outside_control_range_fraction": float(
                            np.mean(ctrl_exceed[:, i] > 0)
                        ),
                        "constraint_torque_absolute_p95_model_units": float(
                            np.percentile(np.abs(constraint[:, i]), 95)
                        ),
                    }
                )
            sensitivity = {}
            for threshold in (0.01, 0.02, 0.05):
                support = load >= threshold
                sensitivity[str(threshold)] = [
                    {
                        "leg": leg,
                        "support_fraction": float(support[:, i].mean()),
                        "slip_mean_mm_s": float(slip[support[:, i], i].mean())
                        if support[:, i].any()
                        else None,
                        "slip_p95_mm_s": float(
                            np.percentile(slip[support[:, i], i], 95)
                        )
                        if support[:, i].any()
                        else None,
                    }
                    for i, leg in enumerate(LEGS)
                ]
            result["phases"][phase] = {
                "samples": len(selected),
                "largest_joint_errors": joints,
                "support_threshold_bw_sensitivity": sensitivity,
            }
        result["limitations"] = (
            "Posthoc boundary observations; moving/braking is the neural request at interval start, not measured speed. Range violation and constraint torque do not identify a sole cause or prove that clipping improves walking."
        )
        write_json(args.out / "result.json", result)
        print(
            json.dumps(
                {
                    "validity": result["validity"],
                    "samples": len(records),
                    "phases": {k: v["samples"] for k, v in result["phases"].items()},
                }
            )
        )
    finally:
        body.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
