"""M3-E03: constrained readout identifiability on teacher-driven BANC states.

The teacher controls the body in this diagnostic. No fit is evidence of an
autonomous neural attractor. All final walking gates remain separate.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import LEGS
from flylab.c.storage import StateStore
from flylab.c.target_sense import TargetObservation
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_m3_motor_transfer import pool
from tools.verify_c_banc_observation import exact


def collect(source, teacher, out, graph_path):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "spec.json",
        {
            "experiment": "M3-E03b",
            "tool_sha256": sha256(__file__),
            "source": "exact M2 initial full state and E01 teacher command prefix",
            "teacher_sha256": sha256(teacher),
            "source_trace_sha256": sha256(source / "trace.json"),
            "hypothesis": "Anatomically restricted static motor-unit capacities can express teacher commands without arbitrary signs",
            "competing_explanation": "Reflex signals have the wrong relation to position targets or lack the required dynamic information; torque labels may differ",
            "intervention": "Full CUDA BANC receives its normal leg receptors plus the existing target encoder's held forward command (bearing=0, distance=20): bilateral DNg100=5000 and all 20 inhibitory port backgrounds=-5000. Body is driven only by recorded teacher commands. Existing M2 weights/dynamics unchanged.",
            "alignment": "Motor(t) computed from pre-interval sensory state predicts command/mean actuator torque applied during [t,t+.005); no future motor, external phase or lag search",
            "train_window_s": [0.5, 1.2],
            "validation_window_s": [1.5, 2.2],
            "split_scope": "Contiguous temporal blocks separated by .3s, not shuffled samples; repeated cycles of one teacher clip, not an independent behavioral validation",
            "models": [
                "M2 decoder",
                "nonnegative per-unit coefficients with existing muscle support/signs",
                "same support, unrestricted-sign diagnostic",
            ],
            "fit": "42 separate ridge least squares, ridge=.01 on rate/200 features, no intercept, no old range/filter limits in this expression test; torque standardized by training SD for conditioning",
            "screen": "Position NNLS validation R2>=.5 for every knee and median>=.5 over all 42 joints with nonconstant labels before considering a joint-command runtime",
            "validity": "All replay qpos/qvel/ctrl bit exact; 50 native torque samples/control; actual CUDA and neural/physical clocks consistent; finite arrays",
            "decision": "If both fits fail, static output rescaling is insufficient. If only constrained fit fails, examine fixed signs/target variable. No adoption from this screen; full teacher removal required for any candidate.",
            "limits": "2.2model sec / 300wall sec, one CUDA job; no parameter search",
        },
    )
    saved = StateStore.load(source / "start")
    commands = json.loads(teacher.read_text())["samples"]
    graph = GraphStore.load(graph_path)
    session = BancWalkingSession.from_checkpoint(graph, saved)
    began = time.perf_counter()
    try:
        if session.network.device != "cuda":
            raise ValueError("Actual CUDA required for this experiment")
        body, adapter = session.body, session.adapter
        exact(saved["body"], body.snapshot(), "teacher-forcing initial")
        tonic = session.navigation.encoder.drive(TargetObservation(0.0, 20.0, True))
        if not (
            np.all(tonic[session.descending] == 5000.0)
            and np.all(tonic[session.navigation.encoder.ports] == -5000.0)
            and np.count_nonzero(tonic) == 22
        ):
            raise AssertionError("Held-forward input contract differs from M2 encoder")
        presyn = np.unique(
            np.concatenate(
                [
                    graph.indices[graph.indptr[i] : graph.indptr[i + 1]]
                    for i in adapter.motor_ids
                ]
            )
        )
        observed = np.union1d(adapter.motor_ids, presyn)
        motor_columns = np.searchsorted(observed, adapter.motor_ids)
        presyn_columns = np.searchsorted(observed, presyn)
        substep = body._step_physics
        torques = []

        def observe_step():
            substep()
            torques.append(body.d.qfrc_actuator[body.qvel_ids].copy())

        body._step_physics = observe_step
        rates, pooled, legacy, target, force, features, presyn_rates = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for tick, row in enumerate(commands[1:441], 1):
            if time.perf_counter() - began > 300:
                raise TimeoutError("E03 wall limit")
            drive = adapter.encode()
            drive += tonic
            observation = session.network.advance_readout(
                drive, observed, session.neural_steps_per_control
            )
            motor = observation[motor_columns]
            presyn_rates.append(observation[presyn_columns])
            prediction, _ = adapter.decode(motor)
            torques.clear()
            body.step_joint_targets(
                np.asarray(row["joint_targets_rad"]),
                np.asarray(row["adhesion"], bool),
                tendon_inputs=row["tendon_inputs"],
            )
            for name, actual in (
                ("qpos", body.d.qpos),
                ("qvel", body.d.qvel),
                ("ctrl", body.d.ctrl),
            ):
                if actual.tobytes() != np.asarray(row[name]).tobytes():
                    raise AssertionError(f"Teacher physics mismatch {tick} {name}")
            if len(torques) != 50:
                raise AssertionError("Expected 50 native substeps")
            rates.append(motor.copy())
            pooled.append(pool(adapter, motor))
            legacy.append(prediction.copy())
            target.append(row["joint_targets_rad"])
            force.append(np.mean(torques, axis=0))
            features.append(adapter.last_features.copy())
            session.control_tick = tick
        session._clocks()
        # Fixed anatomical membership and existing engineering moment directions.
        moment = np.zeros((42, len(adapter.motor_ids)))
        labels = []
        for leg, groups in enumerate(adapter.muscles):
            for name, ids, vector in groups:
                vector = vector.copy()
                vector[0] *= adapter.coxa_sign[leg]
                vector[5] *= adapter.knee_sign[leg]
                columns = [adapter.motor_columns[int(i)] for i in ids]
                moment[np.ix_(adapter.joints[leg], columns)] = vector[:, None]
                labels.append({"leg": LEGS[leg], "muscle": name, "columns": columns})
        data = {
            "time_s": np.arange(1, len(rates) + 1) * 0.005,
            "motor": np.asarray(rates),
            "pooled": np.asarray(pooled),
            "legacy": np.asarray(legacy),
            "target": np.asarray(target),
            "torque": np.asarray(force),
            "sensory": np.asarray(features),
            "moment": moment,
            "neutral": body.neutral,
            "joints": adapter.joints,
            "presyn_ids": presyn,
            "presyn_rates": np.asarray(presyn_rates),
            "tonic_ids": np.flatnonzero(tonic),
            "tonic_values": tonic[tonic != 0],
        }
        np.savez_compressed(out / "signals.npz", **data)
        StateStore.save(out / "teacher-driven-final", session.snapshot())
        write_json(
            out / "collection-result.json",
            {
                "run_status": "completed",
                "validity": "valid",
                "exact_physical_controls": len(rates),
                "native_samples": len(rates) * 50,
                "model_s": len(rates) * 0.005,
                "wall_s": time.perf_counter() - began,
                "device": session.network.device,
                "graph_hash": graph.hash,
                "model_hash": session.network.identity,
                "adapter_hash": adapter.identity,
                "body_hash": body.model_hash,
                "neurons": graph.n,
                "edges": len(graph.indices),
                "motor_roster": [graph.nodes[i]["id"] for i in adapter.motor_ids],
                "muscles": labels,
                "control_status": "teacher-driven diagnostic; saved state is not evidence of autonomous gait",
            },
        )
    finally:
        session.close()
    fit(out)


def quality(prediction, y, mask):
    error = np.mean((prediction[mask] - y[mask]) ** 2)
    variance = np.var(y[mask])
    return {
        "rmse": float(np.sqrt(error)),
        "r2": float(1 - error / variance) if variance > 1e-12 else None,
    }


def fit(out):
    d = np.load(out / "signals.npz")
    t = d["time_s"]
    train = (t >= 0.5) & (t < 1.2)
    valid = (t >= 1.5) & (t < 2.2)
    records, coefficients = {}, {}
    position = d["target"] - d["neutral"]
    for label, y in (("position", position), ("torque", d["torque"])):
        reports = []
        for j in range(42):
            cols = np.flatnonzero(d["moment"][j])
            x = d["motor"][:, cols].astype(float) / 200 * d["moment"][j, cols]
            scale = max(float(np.std(y[train, j])), 1e-9) if label == "torque" else 1.0
            a = np.vstack((x[train], 0.01 * np.eye(len(cols))))
            b = np.r_[y[train, j] / scale, np.zeros(len(cols))]
            constrained = lsq_linear(a, b, bounds=(0, np.inf), tol=1e-10, max_iter=1000)
            if not constrained.success:
                raise ValueError("Constrained solver did not converge")
            free = np.linalg.lstsq(a, b, rcond=1e-10)[0]
            report = {
                "joint": j,
                "motor_columns": cols.tolist(),
                "rank": int(np.linalg.matrix_rank(x[train])),
                "columns": len(cols),
                "condition_number": float(np.linalg.cond(a)) if len(cols) else None,
                "constrained_max_coefficient": float(
                    np.max(constrained.x * scale, initial=0.0)
                ),
                "motor_support_status": "mapped"
                if len(cols)
                else "no_position_muscle_mapping",
            }
            for name, coef in (
                ("constrained", constrained.x * scale),
                ("free_sign_diagnostic", free * scale),
            ):
                prediction = x @ coef
                report[name] = {
                    "train": quality(prediction, y[:, j], train),
                    "validation": quality(prediction, y[:, j], valid),
                }
                full = np.zeros(len(d["motor"][0]))
                full[cols] = coef * d["moment"][j, cols] / 200
                coefficients.setdefault(label + "_" + name, []).append(full)
            if label == "position":
                report["legacy"] = {
                    "validation": quality(
                        d["legacy"][:, j] - d["neutral"][j], y[:, j], valid
                    )
                }
            reports.append(report)
        records[label] = reports
    knee = d["joints"][:, 5]
    all_scores = [r["constrained"]["validation"]["r2"] for r in records["position"]]
    all_scores = [s for s in all_scores if s is not None]
    scores = [records["position"][j]["constrained"]["validation"]["r2"] for j in knee]
    passed = (
        all(s is not None and s >= 0.5 for s in scores) and np.median(all_scores) >= 0.5
    )
    result = {
        "experiment": json.loads((out / "spec.json").read_text())["experiment"],
        "run_status": "completed",
        "validity": "valid",
        "hypothesis_verdict": "position_static_screen_supported"
        if passed
        else "position_static_screen_refuted",
        "adoption": "not_applicable",
        "milestone_outcome": "pending",
        "position_screen_pass": bool(passed),
        "knee_position_r2": dict(zip(LEGS, scores)),
        "all_joint_median_r2": float(np.median(all_scores)),
        "signals_sha256": sha256(out / "signals.npz"),
        "analysis_source_sha256": sha256(__file__),
        "results": records,
    }
    np.savez_compressed(out / "coefficients.npz", **coefficients)
    write_json(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    for label in ("position", "torque"):
        print(
            label,
            json.dumps(
                {
                    leg: {
                        key: records[label][j][key]["validation"]
                        for key in ("constrained", "free_sign_diagnostic")
                    }
                    for leg, j in zip(LEGS, knee)
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    p.add_argument("--fit-only", action="store_true")
    a = p.parse_args()
    if a.fit_only:
        fit(a.out)
    elif a.source and a.teacher:
        collect(a.source, a.teacher, a.out, a.graph)
    else:
        p.error("--source and --teacher required for collection")
