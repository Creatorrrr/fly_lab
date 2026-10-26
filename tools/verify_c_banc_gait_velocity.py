"""Check boundary-velocity aliasing against bit-exact native physical replay."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.storage import StateStore
from flylab.common import to_ui
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_observation import new_body, validate_recording


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    payload = json.loads((source / "trace.json").read_text())
    saved = StateStore.load(source / "start")
    validate_recording(payload, saved)
    physical = np.load(source / "recorded-signals.npz")["physical"]
    body = new_body(saved)
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    original_step = body._step_physics
    measured = []

    def sample():
        body.mj.mj_getState(body.m, body.d, state, typ)
        body.mj.mj_setState(body.m, shadow.d, state, typ)
        body.mj.mj_forward(body.m, shadow.d)
        p, rotation, velocity = shadow.pose()
        jacp, jacr = np.zeros((3, body.m.nv)), np.zeros((3, body.m.nv))
        body.mj.mj_jacBody(body.m, shadow.d, jacp, jacr, body.thorax)
        measured.append(
            [
                float(shadow.d.time - body.t0),
                *to_ui(p),
                *to_ui(rotation[:, 0]),
                *to_ui(jacp @ shadow.d.qvel),
                float(velocity[3:] @ rotation[:, 0]),
            ]
        )

    def observed_step():
        original_step()
        sample()

    body._step_physics = observed_step
    verified = 0
    try:
        for index, row in enumerate(payload["samples"][1:], 1):
            if not 0.5 < row["time_s"] <= 1.5:
                continue
            previous = payload["samples"][index - 1]
            body.mj.mj_setState(body.m, body.d, physical[index - 2], typ)
            body.mj.mj_forward(body.m, body.d)
            body.fault = None
            body.last_action = body.Action(
                joint_angles=np.asarray(previous["joint_targets_rad"]),
                adhesion_onoff=np.asarray(previous["adhesion"], bool),
            )
            request = row["neural_request"]["external_force_at_start"]
            body.push_left, body.push_force = (
                request["push_left_s"],
                np.asarray(request["push_force"]),
            )
            if not measured:
                sample()
            body.step_joint_targets(
                np.asarray(row["joint_targets_rad"]),
                np.asarray(row["adhesion"], bool),
                tendon_inputs=row["tendon_inputs"],
                interpolate=not row["neural_request"]["cuts"]["motor"],
            )
            body.mj.mj_getState(body.m, body.d, state, typ)
            if state.tobytes() != physical[index - 1].tobytes():
                raise AssertionError(f"Physical future differs at {index}")
            verified += 1
        if verified != 200 or len(measured) != 10001:
            raise AssertionError("Incomplete 1 s native replay")
        a = np.asarray(measured)
        t, p, forward, velocity = a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7:10]
        actual = np.sum(velocity * forward, axis=1)
        fd = np.sum(np.gradient(p, t, axis=0) * forward, axis=1)
        control = np.arange(0, len(t), 50)
        cfd = np.sum(
            np.gradient(p[control], t[control], axis=0) * forward[control], axis=1
        )
        result = {
            "source_sha256": sha256(source / "trace.json"),
            "tool_sha256": sha256(__file__),
            "native_samples": len(t),
            "window_s": [0.5, 1.5],
            "bit_exact_physical_controls": verified,
            "native_world_displacement_ui_mm": (p[-1] - p[0]).tolist(),
            "native_integrated_world_velocity_ui_mm": np.trapezoid(
                velocity, t, axis=0
            ).tolist(),
            "native_displacement_integration_error_mm": float(
                np.linalg.norm(p[-1] - p[0] - np.trapezoid(velocity, t, axis=0))
            ),
            "signed_forward_integral_mm": {
                "native_origin_jacobian": float(np.trapezoid(actual, t)),
                "native_finite_difference": float(np.trapezoid(fd, t)),
                "control_boundary_reported": float(
                    np.trapezoid(a[control, 10], t[control])
                ),
                "control_boundary_finite_difference": float(
                    np.trapezoid(cfd, t[control])
                ),
            },
            "interpretation": "Compare native integration and position-derived velocity before selecting walking bouts; boundary instantaneous speeds are not interval averages",
        }
        np.savez_compressed(
            out / "native-velocity.npz",
            time_s=t,
            position_ui_mm=p,
            forward_ui=forward,
            origin_velocity_ui_mm_s=velocity,
            com_forward_mm_s=a[:, 10],
        )
        write_json(out / "result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        body._step_physics = original_step
        body.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
