"""Measured gait, actual mesh replay and matched FlyGym controller comparison.

Reconstructs a separate body from an already verified recording. No network is
loaded, and no pose is assigned to a running actor. High-frequency contact
observations forward a separate mjData copy; command replay retains all physics
substeps and checks its future against the recorded integration state.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.body import LEGS, LINKS, FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.integrity import file_hash, write_json
from flylab.c.banc_mechanics import configure_activation_trajectory, configure_active_length, configure_adhesion_units, configure_finite_pad_contact, configure_torque_capacity
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control, contact_diagnostics, summarize_trace
from flylab.common import to_ui
from flylab.engine import config_values
from tools.verify_c_banc_observation import exact


def new_body(saved):
    body = FlyGymBody(
        saved["seed"],
        saved["world"],
        config_values(None),
        body_options=BodyOptions(
            model="flybody",
            actuation="whole_body",
            servo_profile=saved["parameters"].get("servo_profile", "tracking_all"),
            tendons="all",
        ),
    )
    configure_torque_capacity(body, saved["parameters"].get("torque_limit_bw_mm", 0.0))
    configure_activation_trajectory(body, saved["parameters"].get("joint_activation_tau_s", .03) if saved["parameters"].get("continuous_joint_activation", False) else 0.0)
    configure_active_length(body, saved["parameters"].get("active_length", False))
    configure_finite_pad_contact(body, saved["parameters"].get("finite_pad_contact", False))
    configure_adhesion_units(body, saved["parameters"].get("adhesion_unit_conversion", False))
    if saved["parameters"].get("adhesion_model") == "graded":
        body.graded_adhesion = True
    body.restore(saved["body"])
    return body


def metrics(rows):
    result = summarize_trace(rows)
    load = np.asarray([r["feet"]["load_bearing"] for r in rows])
    slip = np.asarray(
        [[np.nan if v is None else v for v in r["feet"]["slip_mm_s"]] for r in rows]
    )
    error = np.asarray([r["joint_tracking_error_rad"] for r in rows])
    p = np.asarray([r["body"]["position"] for r in rows])[:, [0, 2]]
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    net = float(np.linalg.norm(p[-1] - p[0]))
    result.update(
        horizontal_path_mm=path,
        net_horizontal_mm=net,
        net_to_path_ratio=net / path if path else None,
        total_absolute_yaw_rad=float(
            np.abs(np.diff(np.unwrap([r["body"]["yaw"] for r in rows]))).sum()
        ),
        joint_rms_rad=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        exact_tripod_support_fraction=float(
            np.mean(
                np.all(load == [1, 0, 1, 0, 1, 0], axis=1)
                | np.all(load == [0, 1, 0, 1, 0, 1], axis=1)
            )
        ),
        measured_legs={},
    )
    for i, leg in enumerate(LEGS):
        finite = slip[:, i][np.isfinite(slip[:, i])]
        result["measured_legs"][leg] = {
            "support_fraction": float(load[:, i].mean()),
            "support_transitions": int(
                np.count_nonzero(np.diff(load[:, i].astype(int)))
            ),
            "loaded_slip_percentiles_mm_s": np.percentile(
                finite, [50, 95, 100]
            ).tolist()
            if len(finite)
            else None,
        }
    nav = rows[-1].get("navigation")
    if nav is not None:
        result["final_navigation"] = nav
    result["interpretation"] = (
        "Descriptive boundary samples, not a natural-gait or biological-validity verdict; normal load includes adhesion reaction."
    )
    return result


def plot_rows(rows, out, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([r["time_s"] for r in rows])
    load = np.asarray([r["feet"]["normal_load_bw"] for r in rows])
    slip = np.asarray(
        [[np.nan if v is None else v for v in r["feet"]["slip_mm_s"]] for r in rows]
    )
    error = np.asarray([r["joint_tracking_error_rad"] for r in rows])
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    for i, leg in enumerate(LEGS):
        axes[0].plot(t, load[:, i], lw=0.6, label=leg.upper())
        axes[1].plot(t, slip[:, i], lw=0.6)
    axes[0].legend(ncol=6, loc="upper right")
    axes[0].set_ylabel("Normal load (BW)")
    axes[1].set_ylabel("Loaded contact slip (mm/s)")
    axes[2].plot(t, np.sqrt(np.mean(error**2, axis=1)), color="#2e759c")
    axes[2].set_ylabel("42-joint RMS error (rad)")
    axes[2].set_xlabel("Model time (s)")
    axes[0].set_title(label + " | 5 ms measurements; normal load includes adhesion")
    fig.savefig(out / "contact-tracking.png", dpi=150)
    plt.close(fig)


def validate_recording(payload, saved):
    rows = payload["samples"]
    if len(rows) < 2 or not payload.get("command_replay", {}).get("eligible"):
        raise ValueError("Complete command-replay recording required")
    anchor = payload.get("initial_physics")
    if not anchor or anchor["control_tick"] != saved["control_tick"]:
        raise ValueError("Recording and checkpoint anchor ticks differ")
    exact(anchor["body"], saved["body"], "initial physics")
    if payload["body_hash"] != saved["body"]["modelHash"]:
        raise ValueError("Recording and checkpoint physics models differ")
    for offset, row in enumerate(rows):
        tick = saved["control_tick"] + offset
        if row["control_tick"] != tick or row["time_s"] != tick * 0.005:
            raise ValueError(
                "Noncontiguous trace; analysis must not bridge missing time"
            )


def native_video(body, payload, out):
    import imageio.v2 as iio
    from PIL import Image, ImageDraw

    rows = payload["samples"]
    shadow = body.mj.MjData(body.m)
    renderer = body.mj.Renderer(body.m, height=480, width=800)
    camera = body.mj.MjvCamera()
    camera.type = body.mj.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 9.0, 135.0, -28.0
    max_segment_error = 0.0
    selected = list(range(0, len(rows), 4))  # 50fps, real recorded states only.
    try:
        with iio.get_writer(
            out / "native-mesh.mp4", fps=50, codec="libx264", quality=7
        ) as writer:
            for index in selected:
                row = rows[index]
                shadow.qpos[:] = row["qpos"]
                shadow.qvel[:] = row["qvel"]
                shadow.ctrl[:] = row["ctrl"]
                body.mj.mj_forward(body.m, shadow)
                for leg in LEGS:
                    positions = [
                        to_ui(
                            shadow.xpos[
                                int(body.body_ids[body.body_indices[f"{leg}_{link}"]])
                            ]
                        )
                        for link in LINKS
                        if f"{leg}_{link}" in body.body_indices
                    ]
                    max_segment_error = max(
                        max_segment_error,
                        float(
                            np.max(
                                np.abs(np.asarray(positions) - row["body"]["legs"][leg])
                            )
                        ),
                    )
                camera.lookat[:] = shadow.xpos[body.thorax]
                renderer.update_scene(shadow, camera)
                frame = Image.fromarray(renderer.render())
                ImageDraw.Draw(frame).text(
                    (12, 12),
                    f"Native FlyBody mesh | recorded t={row['time_s']:.3f} model s",
                    fill=(255, 255, 255),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
                writer.append_data(np.asarray(frame))
                if index in (selected[0], selected[len(selected) // 2], selected[-1]):
                    frame.save(out / f"native-{index:04d}.png")
    finally:
        renderer.close()
    if max_segment_error > 1e-12:
        raise AssertionError("Native replay mesh and recorded segment positions differ")
    return {
        "frames": len(selected),
        "fps": 50,
        "sampled_every_s": 0.02,
        "maximum_segment_error_mm": max_segment_error,
        "reconstruction": "Separate mjData, recorded qpos/qvel/ctrl; no actor integration or interpolation",
    }


def comparison_html(payload, out):
    scripts = "\n".join(
        (ROOT / p).read_text(encoding="utf-8")
        for p in ("src/core/math.js", "src/ui/world-view.js", "src/c/banc-trace.js")
    )
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = """<!doctype html><meta charset="utf-8"><title>BANC 실제 메시와 측정 자세</title>
<style>body{background:#0b151d;color:#e6eff5;font:15px system-ui;margin:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;background:#000}#wrap{height:480px;position:relative}canvas{position:absolute;width:100%;height:100%}button{padding:10px;margin:8px}input{width:95%}pre{white-space:pre-wrap}img{max-width:100%}</style>
<h1>BANC 실제 메시와 5ms 측정 자세</h1><p>왼쪽은 기록된 관절 상태로 별도 MuJoCo 데이터에서 렌더한 실제 메시, 오른쪽은 같은 시각의 실제 분절 위치입니다. 보간·신경 계산·물리 적분 없이 재생합니다.</p>
<button id="play">1× 재생</button><button id="slow">0.25× 재생</button><button id="pause">정지</button>
<main><div><video id="native" preload="metadata" src="native-mesh.mp4"></video><p>실제 메시: 50 FPS 기록 영상</p></div><div id="wrap"><canvas id="world"></canvas><canvas id="overlay"></canvas></div></main>
<input id="seek" type="range" min="0" max="1" step=".001" value="0"><p id="status"></p><pre id="feet"></pre>
<img src="contact-tracking.png" alt="접촉 하중, 실제 접촉점 미끄러짐, 관절 추종 오차"><script>"""
    html += (
        scripts
        + "\nconst payload="
        + encoded
        + """;
const el=id=>document.getElementById(id),video=el('native'),v=Fly.createBancTraceView({canvas:el('world'),overlay:el('overlay'),status:el('status'),diagnostic:el('feet'),slider:el('seek')});
v.ingest({trace:payload,trace_epoch:payload.epoch,world:payload.world});
const span=payload.samples.at(-1).time_s-payload.samples[0].time_s;
el('play').onclick=()=>{video.playbackRate=1;video.play();};el('slow').onclick=()=>{video.playbackRate=.25;video.play();};el('pause').onclick=()=>video.pause();
el('seek').oninput=()=>{video.pause();video.currentTime=Number(el('seek').value)*span;};
function draw(now){requestAnimationFrame(draw);if(!document.hidden){v.seek(Math.min(video.currentTime,span)/span);v.draw(now);}}requestAnimationFrame(draw);
</script>"""
    )
    (out / "comparison.html").write_text(html, encoding="utf-8")


def high_frequency(body, saved, rows, signals, out):
    # Exploratory selection around the largest recorded loaded slip, not a holdout.
    maxima = [
        max((v for v in r["feet"]["slip_mm_s"] if v is not None), default=0.0)
        for r in rows
    ]
    center = int(np.argmax(maxima[1:])) + 1
    begin, end = max(1, center - 10), min(len(rows) - 1, center + 10)
    body.restore(saved["body"])
    shadow_body = copy.copy(body)
    shadow_body.d = body.mj.MjData(body.m)
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    original = body._step_physics
    selected = False
    samples = []

    def step_observed():
        original()
        if selected:
            body.mj.mj_getState(body.m, body.d, state, typ)
            body.mj.mj_setState(body.m, shadow_body.d, state, typ)
            body.mj.mj_forward(body.m, shadow_body.d)
            samples.append(
                {
                    "time_s": float(body.d.time - body.t0),
                    "feet": contact_diagnostics(shadow_body),
                }
            )

    body._step_physics = step_observed
    try:
        for index, row in enumerate(rows[1:], 1):
            selected = begin <= index <= end
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
            if state.tobytes() != signals["physical"][index - 1].tobytes():
                raise AssertionError(
                    f"Diagnostic shadow changed physical future at {index}"
                )
            if index >= end:
                break
    finally:
        body._step_physics = original
    coarse = rows[begin : end + 1]
    values = lambda data: np.asarray(
        [[np.nan if v is None else v for v in r["feet"]["slip_mm_s"]] for r in data]
    )
    a, b = values(samples), values(coarse)
    result = {
        "selection": "Exploratory maximum boundary-slip window",
        "first_control": begin,
        "last_control": end,
        "shadow_sample_dt_s": 0.0001,
        "samples": len(samples),
        "command_replay_through_window_bit_exact": True,
        "legs": {
            leg: {
                "fine_max_mm_s": float(np.nanmax(a[:, i]))
                if np.isfinite(a[:, i]).any()
                else None,
                "coarse_max_mm_s": float(np.nanmax(b[:, i]))
                if np.isfinite(b[:, i]).any()
                else None,
            }
            for i, leg in enumerate(LEGS)
        },
    }
    write_json(out / "substep-contact.json", {"result": result, "samples": samples})
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    payload = json.loads((args.source / "trace.json").read_text(encoding="utf-8"))
    saved = StateStore.load(args.source / "start")
    rows = payload["samples"]
    validate_recording(payload, saved)
    write_json(
        args.out / "spec.json",
        {
            "experiment": "OBS-E3",
            "source": str(args.source),
            "source_hash": file_hash(Path(__file__)),
            "comparison": "Identical saved FlyBody physics, 10s general/target measured BANC versus existing FlyGym hybrid CPG at fixed 3.3mm/s request, all 0.2 tendon inputs; CPG is comparison only, not introduced into BANC",
            "judgment": "Descriptive slip/tracking/support/path comparison; no automatic gait pass or causal mechanism claim. Mesh segments must match recorded positions and shadow diagnostics must preserve integration bits.",
            "high_frequency": "Posthoc highest measured loaded-slip window up to 105ms, 0.1ms shadow mjData forward, separate from running actor",
            "bounds": "One body replay/video per source, optional matched controller up to source duration; no BANC GPU compute; stop on fault",
        },
    )
    result = {"validity": "pending"}
    body = new_body(saved)
    try:
        result["joint_names"] = list(body.joint_names)
        if body.d.qpos.tobytes() != np.asarray(rows[0]["qpos"]).tobytes():
            raise ValueError("First recorded pose and physical anchor differ")
        if body.d.qvel.tobytes() != np.asarray(rows[0]["qvel"]).tobytes():
            raise ValueError("First recorded velocity and physical anchor differ")
        if body.d.ctrl.tobytes() != np.asarray(rows[0]["ctrl"]).tobytes():
            raise ValueError("First recorded control and physical anchor differ")
        result["banc"] = metrics(rows)
        plot_rows(rows, args.out, args.source.name)
        result["native_mesh"] = native_video(body, payload, args.out)
        comparison_html(payload, args.out)
        result["substep_contact"] = high_frequency(
            body, saved, rows, np.load(args.source / "recorded-signals.npz"), args.out
        )
        if args.compare_flygym:
            body.controller = type(body.controller)(
                timestep=0.005,
                preprogrammed_steps=body.steps,
                output_dof_order=body.order,
            )
            body.controller.reset(seed=saved["seed"])
            body.controller.enable_adhesion = False
            body.restore(saved["body"])
            exact(saved["body"], body.snapshot(), "matched comparison start")
            tendons = {name: 0.2 for name in body.tendon_control.names}
            observer = SimpleNamespace(
                body=body,
                control_tick=saved["control_tick"],
                rate=np.zeros(391),
                tendon_inputs=tendons,
                fault=None,
            )
            cpg_rows = [capture_control(observer)]
            drive = body.motor.map({"forwardSpeed": 3.3, "yawRate": 0.0})
            for tick in range(1, len(rows)):
                action = body.controller.step(drive, body._observation())
                body.step_joint_targets(
                    action.joint_angles,
                    np.asarray(action.adhesion_onoff, bool),
                    tendon_inputs=tendons,
                )
                observer.control_tick = saved["control_tick"] + tick
                cpg_rows.append(capture_control(observer))
                if body.fault:
                    break
            result["flygym_comparison"] = metrics(cpg_rows)
            result["flygym_comparison"]["same_initial_physics_model_hash"] = (
                body.model_hash == saved["body"]["modelHash"]
            )
            result["flygym_comparison"]["same_initial_physical_state_bit_exact"] = True
            result["flygym_comparison"]["control_dt_s"] = 0.005
            result["flygym_comparison"]["limitations"] = (
                "Different controller and held tendon/adhesion policy; this measures whether the same physical body can walk, not which individual BANC parameter caused a difference."
            )
            write_json(args.out / "flygym-trace.json", {"samples": cpg_rows})
        result["validity"] = "valid"
        write_json(args.out / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        body.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare-flygym", action="store_true")
    run(p.parse_args())
