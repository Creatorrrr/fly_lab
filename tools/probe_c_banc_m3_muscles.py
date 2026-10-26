"""M3-E02b: equal motor-unit inputs to anatomical muscles on one free body.

Diagnostic inputs bypass BANC. Forces are MuJoCo servo forces, not measured
biological muscle forces. Every arm restores the exact same physical state.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import LEGS
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_m3_motor_transfer import adapter_for
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def run(source, out, graph_path):
    out.mkdir(parents=True, exist_ok=False)
    arms = [("zero", None, ())]
    for leg in LEGS:
        for muscles in (
            ("tibia_flexor_muscle",),
            ("accessory_tibia_flexor_muscle",),
            ("tibia_extensor_muscle",),
            ("tibia_flexor_muscle", "tibia_extensor_muscle"),
        ):
            arms.append((leg + "-" + "+".join(muscles), leg, muscles))
    for leg in ("lf", "rf"):
        for muscle in (
            "sternal_anterior_rotator_muscle",
            "sternal_posterior_rotator_muscle",
            "tergopleural_promotor_muscle",
            "pleural_remotor_and_abductor_muscle",
            "sternal_adductor_muscle",
        ):
            arms.append((leg + "-" + muscle, leg, (muscle,)))
    write_json(
        out / "spec.json",
        {
            "experiment": "M3-E02b",
            "tool_sha256": sha256(__file__),
            "source_trace_sha256": sha256(source / "trace.json"),
            "hypothesis": "Anatomical single-muscle inputs reach the expected joint/geometry, while equal antagonist activity cancels in the position decoder",
            "competing_explanation": "Actuator ranges, physical coupling, wrong direction or receptor response block the intended movement",
            "intervention": "Each listed motor unit receives 100 model units for .2s; all others zero. Same M2 initial free body restored in each arm; .2 tendon support retained. No network integration.",
            "arms": arms,
            "validity": "Every initial body bit exact; finite raw joint/force/foot/receptor records, preserve collisions",
            "decision": "Identify direction and cancellation, not fit biological forces or claim gait improvement",
            "budget": "35 arms, each <=.2model sec, CPU only, no parameter search",
        },
    )
    began = time.perf_counter()
    saved = StateStore.load(source / "start")
    graph = GraphStore.load(graph_path)
    body = new_body(saved)
    result, arrays = [], {}
    try:
        adapter = adapter_for(graph, body, saved)
        initial_flexion = body.knee_kinematics(include_direction=False)[
            "flexion_rad"
        ].copy()
        for name, leg, names in arms:
            body.restore(saved["body"])
            adapter.restore(saved["adapter"])
            exact(saved["body"], body.snapshot(), name + " initial")
            rates = np.zeros(len(adapter.motor_ids), np.float32)
            members = []
            if leg is not None:
                groups = adapter.muscles[LEGS.index(leg)]
                lookup = {n: ids for n, ids, _ in groups}
                missing = set(names) - lookup.keys()
                if missing:
                    result.append(
                        {
                            "arm": name,
                            "status": "anatomical_group_absent",
                            "missing": sorted(missing),
                        }
                    )
                    continue
                for muscle in names:
                    ids = lookup[muscle]
                    rates[[adapter.motor_columns[int(i)] for i in ids]] = 100.0
                    members.extend(graph.nodes[i]["id"] for i in ids)
            recording = []
            for tick in range(40):
                target, adhesion = adapter.decode(rates)
                tendons = {key: 0.2 for key in body.tendon_control.names}
                tendons.update(adapter.tendon_inputs)
                body.step_joint_targets(target, adhesion, tendon_inputs=tendons)
                adapter.encode()
                obs = body.leg_observation()
                knee = body.knee_kinematics(include_direction=False)["flexion_rad"]
                recording.append(
                    np.concatenate(
                        (
                            body.d.qpos,
                            body.d.qvel,
                            target,
                            knee,
                            body.d.qfrc_actuator[body.qvel_ids],
                            adapter.last_features,
                        )
                    )
                )
                if body.fault or body.nonfoot_contact():
                    break
            arrays[name] = np.asarray(recording)
            result.append(
                {
                    "arm": name,
                    "status": "completed",
                    "model_s": len(recording) * 0.005,
                    "motor_ids": members,
                    "fault": body.fault,
                    "nonfoot_contact": body.nonfoot_contact(),
                    "upright": float(body.pose()[1][2, 2]),
                    "target_delta_rad": (target - body.neutral).tolist(),
                    "flexion_delta_deg": np.degrees(knee - initial_flexion).tolist(),
                    "joint_tracking_error_deg": np.degrees(
                        target - obs["angles_rad"]
                    ).tolist(),
                    "receptors": adapter.last_features.tolist(),
                }
            )
        np.savez_compressed(out / "raw-arms.npz", **arrays)
        paired = {}
        for leg in LEGS:
            name = leg + "-tibia_flexor_muscle+tibia_extensor_muscle"
            paired[leg] = bool(np.array_equal(arrays[name], arrays["zero"]))
        report = {
            "experiment": "M3-E02b",
            "run_status": "completed",
            "validity": "valid",
            "hypothesis_verdict": "pending interpretation",
            "adoption": "not_applicable",
            "milestone_outcome": "pending",
            "wall_s": time.perf_counter() - began,
            "body_hash": body.model_hash,
            "adapter_hash": adapter.identity,
            "raw_layout": {
                "nq": body.m.nq,
                "nv": body.m.nv,
                "after_qpos_qvel": [
                    "target42",
                    "flexion6",
                    "actuator_torque42",
                    "receptor_features",
                ],
            },
            "zero_vs_equal_antagonist_bit_exact": paired,
            "arms": result,
        }
        write_json(out / "result.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "arms"}), flush=True)
        for r in result:
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "arm",
                            "status",
                            "flexion_delta_deg",
                            "fault",
                            "nonfoot_contact",
                        )
                        if k in r
                    }
                ),
                flush=True,
            )
    finally:
        body.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    a = p.parse_args()
    run(a.source, a.out, a.graph)
