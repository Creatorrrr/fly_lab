"""Measure the complete moving window at every native physics phase.

v1 remains a control-boundary diagnostic. This v2 measurement fixes its
phase-locked sampling bias without changing the slip formula or success ratios.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.body import LEGS
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore
from flylab.c.walking_trace import contact_diagnostics
from tools.analyze_c_banc_gait_quality import compare, quality
from tools.analyze_c_banc_observation import new_body, validate_recording

SCHEMA = "flylab.banc-gait-quality.v2"


def summarize(load, slip, threshold):
    support = load >= threshold
    if np.any(support & ~np.isfinite(slip)):
        raise ValueError("Missing supported microstep velocity")
    legs = []
    for i, leg in enumerate(LEGS):
        values = slip[support[:, i], i]
        legs.append({
            "leg": leg, "supported_samples": len(values),
            "supported_time_s": len(values) * .0001,
            "support_fraction": float(support[:, i].mean()),
            "mean_slip_mm_s": float(values.mean()) if len(values) else None,
            "p95_slip_mm_s": float(np.percentile(values, 95)) if len(values) else None,
        })
    return {
        "legs": legs,
        "balanced_mean_slip_mm_s": float(np.mean([r["mean_slip_mm_s"] for r in legs]))
        if all(r["mean_slip_mm_s"] is not None for r in legs) else None,
        "mean_support_legs": float(support.sum(axis=1).mean()),
    }


def compare_v2(base, candidate):
    if base.get("schema") != SCHEMA or candidate.get("schema") != SCHEMA:
        raise ValueError("Both arms must use the full-substep v2 evaluator")
    for key in ("evaluator_hash", "sample_dt_s", "control_dt_s", "support_threshold_bw", "comparison_context"):
        if key not in base or key not in candidate or base[key] != candidate[key]:
            raise ValueError("Incompatible measurement comparison: " + key)
    result = compare(base, candidate)
    boundary = compare(base["control_boundary_v1"], candidate["control_boundary_v1"])
    result["v1_progress_Q2"] = boundary["Q2"]
    result["Q2"] = result["Q2"] and boundary["Q2"]
    result["v1_Q1_unchanged"] = boundary["Q1"]
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    evaluator_hash = file_hash(Path(__file__))
    write_json(args.out / "spec.json", {
        "experiment": args.experiment, "source": str(args.source),
        "evaluator_hash": evaluator_hash,
        "v1_evaluator_hash": file_hash(ROOT / "tools/analyze_c_banc_gait_quality.py"),
        "question": __doc__, "sample_dt_s": .0001,
        "selection": "Every physics step after .5s through first brake start, at most 10s",
        "validity": "All selected recorded physical futures and endpoint contacts exactly equal",
        "decision": "Q1 decrease >=30%; original v1 Q2 AND v2 Q2; same arrival/stop contract",
        "additional_support_guard": "Each leg loaded >=.5s, retaining v1's >=100 supported control samples",
        "wall_limit_s": args.wall_limit,
    })
    began = time.perf_counter()
    payload = json.loads((args.source / "trace.json").read_text(encoding="utf-8"))
    saved = StateStore.load(args.source / "start")
    validate_recording(payload, saved)
    coarse = quality(payload)
    signals = np.load(args.source / "recorded-signals.npz")["physical"]
    body = new_body(saved)
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    loads, slips, times = [], [], []
    original_step = body._step_physics

    def observed_step():
        original_step()
        body.mj.mj_getState(body.m, body.d, state, typ)
        body.mj.mj_setState(body.m, shadow.d, state, typ)
        body.mj.mj_forward(body.m, shadow.d)
        contact = contact_diagnostics(shadow, support_threshold_bw=.01)
        loads.append(contact["normal_load_bw"])
        slips.append([np.nan if v is None else v for v in contact["slip_mm_s"]])
        times.append(float(body.d.time - body.t0))

    body._step_physics = observed_step
    selected = 0
    try:
        for index, row in enumerate(payload["samples"][1:], 1):
            if not .5 < row["time_s"] <= coarse["window_end_s"] + 1e-9:
                continue
            previous = payload["samples"][index - 1]
            if 'joint_stiffness_scale' in previous:
                from flylab.c.banc_mechanics import set_active_length_scale
                set_active_length_scale(body, previous['joint_stiffness_scale'])
            body.mj.mj_setState(body.m, body.d, signals[index - 2], typ)
            body.mj.mj_forward(body.m, body.d)
            body.fault = None
            previous = payload["samples"][index - 1]
            body.last_action = body.Action(joint_angles=np.asarray(previous["joint_targets_rad"]),
                                           adhesion_onoff=np.asarray(previous["adhesion"], bool))
            request = row["neural_request"]["external_force_at_start"]
            body.push_left, body.push_force = request["push_left_s"], np.asarray(request["push_force"])
            body.step_joint_targets(np.asarray(row["joint_targets_rad"]),
                                    np.asarray(row["adhesion"], bool),
                                    tendon_inputs=row["tendon_inputs"],
                                    interpolate=not row["neural_request"]["cuts"]["motor"],
                                    **({'joint_stiffness_scale': row['joint_stiffness_scale']} if 'joint_stiffness_scale' in row else {}),
                                    **({'adhesion_control': row['adhesion_control']} if 'adhesion_control' in row else {}))
            body.mj.mj_getState(body.m, body.d, state, typ)
            if state.tobytes() != signals[index - 1].tobytes():
                raise AssertionError(f"Physical future mismatch at control {index}")
            expected = np.asarray(row["feet"]["normal_load_bw"])
            if not np.array_equal(expected, np.asarray(loads[-1])):
                raise AssertionError(f"Endpoint load mismatch at control {index}")
            observed = np.asarray(slips[-1])
            expected_slip = np.asarray([np.nan if v is None else v for v in row["feet"]["slip_mm_s"]])
            mask = expected >= .02
            if not np.array_equal(expected_slip[mask], observed[mask]):
                raise AssertionError(f"Endpoint slip mismatch at control {index}")
            selected += 1
            if selected % 200 == 0:
                print(json.dumps({"controls_verified": selected, "wall_s": time.perf_counter() - began}), flush=True)
            if time.perf_counter() - began > args.wall_limit:
                raise TimeoutError("Full-phase measurement wall limit")
        if not selected or len(loads) != selected * 50:
            raise ValueError("Complete 50-phase moving windows required")
        expected_controls = round((coarse["window_end_s"] - .5) / .005)
        if selected != expected_controls or abs(times[-1] - coarse["window_end_s"]) > 1e-8:
            raise ValueError("Incomplete moving window")
        if file_hash(Path(__file__)) != evaluator_hash:
            raise RuntimeError("Evaluator changed during measurement")
        load, slip = np.asarray(loads), np.asarray(slips)
        result = copy.deepcopy(coarse)
        result.update(summarize(load, slip, .02))
        result.update({
            "schema": SCHEMA, "source": str(args.source),
            "sample_dt_s": .0001, "control_dt_s": .005,
            "moving_observed_s": selected * .005,
            "control_boundary_v1": coarse,
            "command_replay_bit_exact": True, "endpoint_contact_bit_exact": True,
            "verified_controls": selected, "substep_samples": len(load),
            "sensitivity": {str(t): summarize(load, slip, t) for t in (.01, .02, .05)},
            "phase_means_mm_s": [summarize(load[i::50], slip[i::50], .02)["balanced_mean_slip_mm_s"] for i in range(50)],
            "limitations": "Full native .1ms sampling; normal load includes adhesion reaction; engineering model, not biological gait validation",
            "evaluator_hash": evaluator_hash,
            "comparison_context": {
                "graph_hash": payload["graph_hash"], "seed": saved["seed"],
                "world": saved["world"],
                "initial_position": payload["samples"][0]["body"]["position"],
                "target_xz_mm": payload["samples"][0]["navigation"]["target_xz_mm"],
            },
            "wall_s": time.perf_counter() - began,
        })
        result["observation_valid"] = bool(coarse["observation_valid"] and all(r["supported_time_s"] >= .5 for r in result["legs"]))
        if args.baseline:
            result["comparison"] = compare_v2(json.loads(args.baseline.read_text()), result)
        np.savez_compressed(args.out / "contacts.npz", time_s=times, load_bw=load, slip_mm_s=slip)
        write_json(args.out / "quality.json", result)
        print(json.dumps({k: result[k] for k in ("balanced_mean_slip_mm_s", "mean_support_legs", "observation_valid", "wall_s")}
                         | {"comparison": result.get("comparison")}), flush=True)
        return result
    finally:
        body._step_physics = original_step
        body.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--wall-limit", type=float, default=300)
    parser.add_argument("--experiment", default="M2-E21")
    run(parser.parse_args())
