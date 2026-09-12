#!/usr/bin/env python3
"""Full-bundle gate: exact state preservation, command replay and measured gait.

Uses a private session, never an open server or user checkpoint directory.
Run on the CUDA machine with its existing BANC bundle. No gait PASS is inferred
from this tool's numerical PASS. All outputs go into a new directory.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import (
    TRACE_SCHEMA,
    summarize_trace,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def exact(left, right, path="state"):
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(path + " keys differ")
        for key in left:
            if key != "wall_s":
                exact(left[key], right[key], path + "." + key)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(path + " length differs")
        for i, (a, b) in enumerate(zip(left, right)):
            exact(a, b, f"{path}[{i}]")
    elif isinstance(left, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes() != right.tobytes()
        ):
            raise AssertionError(path + " bits differ")
    elif isinstance(left, float):
        if np.float64(left).tobytes() != np.float64(right).tobytes():
            raise AssertionError(path + " float bits differ")
    elif left != right:
        raise AssertionError(path + " differs")


def measured(session):
    body = session.body
    return np.concatenate(
        (
            body.d.qpos,
            body.d.qvel,
            body.d.ctrl,
            session.rate.astype(float),
            session.adapter.offset,
            session.adapter.filtered,
        )
    )


def write_replay(path, payload):
    root = Path(__file__).resolve().parents[1]
    scripts = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("src/core/math.js", "src/ui/world-view.js", "src/c/banc-trace.js")
    )
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = """<!doctype html><meta charset="utf-8"><title>BANC measured-state replay</title>
<style>body{background:#0a1115;color:#dce9ed;font:14px sans-serif;margin:24px}button{padding:10px;margin-right:8px}#wrap{position:relative;height:460px;max-width:1100px}canvas{position:absolute;width:100%;height:100%}#overlay{pointer-events:none}pre{white-space:pre-wrap;line-height:1.7}input{width:80%}</style>
<h1>BANC 측정 상태 재생</h1><p>저장된 5ms 물리 자세입니다. 신경·물리를 다시 계산하거나 보간하지 않습니다.</p>
<button id="normal">1× 재생</button><button id="slow">0.25× 재생</button><div id="wrap"><canvas id="world"></canvas><canvas id="overlay"></canvas></div>
<p id="status"></p><input id="seek" type="range" min="0" max="1" step=".001" value="0"><pre id="feet"></pre><script>"""
    html += (
        scripts
        + "\nconst payload="
        + encoded
        + """;
const el=id=>document.getElementById(id),v=Fly.createBancTraceView({canvas:el('world'),overlay:el('overlay'),status:el('status'),diagnostic:el('feet'),slider:el('seek')});
v.ingest({trace:payload,trace_epoch:payload.epoch,world:payload.world});v.replay(.25);
el('normal').onclick=()=>v.replay(1);el('slow').onclick=()=>v.replay(.25);el('seek').oninput=()=>v.seek(Number(el('seek').value));
function draw(t){requestAnimationFrame(draw);if(!document.hidden)v.draw(t);}requestAnimationFrame(draw);
</script>"""
    )
    path.write_text(html, encoding="utf-8")


def run(args):
    if not (args.graph / "manifest.json").is_file():
        raise FileNotFoundError("BANC bundle missing: " + str(args.graph))
    if (
        not np.isfinite(args.seconds)
        or not 0.005 <= args.seconds <= 30
        or abs(args.seconds / 0.005 - round(args.seconds / 0.005)) > 1e-8
    ):
        raise ValueError("seconds must be a multiple of 5ms, in [0.005,30]")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "full_banc_gate": True,
        "gait_verdict": "NOT_ASSESSED",
    }
    write_json(args.out / "result.json", report)
    session = None
    try:
        graph = GraphStore.load(args.graph)
        parameters = {"neural_dt_s": args.neural_dt}
        navigation = None
        if args.target is not None:
            parameters.update(coxa_geometry=True, ltm_tendons=True)
            navigation = {"target_xz_mm": args.target, "forward_drive": 5000.0}
        session = BancWalkingSession(
            graph,
            seed=args.seed,
            parameters=parameters,
            device=args.device,
            navigation=navigation,
        )
        start = session.snapshot()
        StateStore.save(args.out / "initial-checkpoint", start)
        count = round(args.seconds / 0.005)
        session.fused_readout = False
        reference = []
        for _ in range(count):
            session.advance(1, capture=False)
            reference.append(measured(session))
        final_reference = session.snapshot()
        reference = np.asarray(reference)
        session.close()
        session = BancWalkingSession.from_checkpoint(graph, start)
        rows = []
        with (args.out / "trace.jsonl").open("w", encoding="utf-8") as out:
            for i in range(count):
                session.advance(1, capture=True)
                if session.trace_error:
                    raise RuntimeError("Observation failed: " + session.trace_error)
                actual = measured(session)
                if actual.tobytes() != reference[i].tobytes():
                    raise AssertionError(f"Control {i + 1}: physical/motor bits differ")
                if i == 0:
                    rows.append(session.trace.samples[0])
                    out.write(
                        json.dumps(rows[-1], ensure_ascii=False, allow_nan=False) + "\n"
                    )
                rows.append(session.trace.samples[-1])
                out.write(
                    json.dumps(rows[-1], ensure_ascii=False, allow_nan=False) + "\n"
                )
        exact(final_reference, session.snapshot())
        metadata = {
            "schema": TRACE_SCHEMA,
            "epoch": session.trace.epoch,
            "sample_dt_s": 0.005,
            "graph_hash": graph.hash,
            "model_hash": session.network.identity,
            "body_hash": session.body.model_hash,
            "adapter_hash": session.adapter.identity,
            "world": session.world,
            "parameters": parameters,
            "samples": rows,
        }
        write_json(args.out / "trace.json", metadata)
        write_json(args.out / "gait-summary.json", summarize_trace(rows))
        write_replay(args.out / "replay.html", metadata)
        np.savez_compressed(args.out / "reference-signals.npz", states=reference)
        # Hold recorded controls at their original control boundaries, with no
        # neural stepping; the same body receives targets/adhesion/tendons.
        session.body.restore(start["body"])
        nq, nv, nu = session.body.m.nq, session.body.m.nv, session.body.m.nu
        for i, sample in enumerate(rows[1:]):
            session.body.step_joint_targets(
                np.asarray(sample["joint_targets_rad"]),
                np.asarray(sample["adhesion"], dtype=bool),
                tendon_inputs=sample["tendon_inputs"],
            )
            physical = np.concatenate(
                (session.body.d.qpos, session.body.d.qvel, session.body.d.ctrl)
            )
            if physical.tobytes() != reference[i, : nq + nv + nu].tobytes():
                raise AssertionError(
                    f"Command-only physical replay differs at control {i + 1}"
                )
        session.close()
        session = BancWalkingSession.from_checkpoint(graph, start)
        # Paired timing measures only the current unfused/fused readout paths,
        # not a claim about the old source tree or browser/network throughput.
        measurements = {"unfused": [], "fused": [], "fused_recorded": []}
        session.advance(10)
        warmed = session.snapshot()
        for pair in range(3):
            for mode in (
                ("unfused", "fused", "fused_recorded")
                if pair % 2 == 0
                else ("fused_recorded", "fused", "unfused")
            ):
                session.close()
                session = BancWalkingSession.from_checkpoint(graph, warmed)
                session.fused_readout = mode != "unfused"
                began = time.perf_counter()
                session.advance(20, capture=mode == "fused_recorded")
                measurements[mode].append(time.perf_counter() - began)
                if session.trace_error:
                    raise RuntimeError(session.trace_error)
        report.update(
            status="PASS",
            exact_control_samples=count,
            final_state_bitwise=True,
            command_replay_bitwise=True,
            device=session.network.device,
            timing_scope="current runtime, 0.1model seconds; PNG, browser/network, restore/compile excluded; recording measured separately",
            unfused_wall_s=measurements["unfused"],
            fused_wall_s=measurements["fused"],
            fused_recorded_wall_s=measurements["fused_recorded"],
            fused_speedup=statistics.median(measurements["unfused"])
            / statistics.median(measurements["fused"]),
            recording_overhead_ratio=statistics.median(measurements["fused_recorded"])
            / statistics.median(measurements["fused"]),
            note="PASS is numerical/observer/replay validation, not a natural-gait verdict",
        )
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out / "result.json", report)
        if session is not None:
            session.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--neural-dt", type=float, choices=(0.0001, 0.0002, 0.00025), default=0.00025
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--target", nargs=2, type=float)
    run(parser.parse_args())
