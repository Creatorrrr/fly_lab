"""Bounded, same-state neural command assays; does not modify live sessions."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import canonical, digest, file_hash, write_json
from flylab.c.neuromuscular import build_spec
from tools.verify_c_banc_rate_walking import sample


def restore(session, state):
    session.body.restore(state["body"])
    session.network.restore(state["network"])
    session.adapter.restore(state["adapter"])
    session.tendon_inputs.update(session.adapter.tendon_inputs)
    session.control_tick = state["control_tick"]
    session.signed_forward_mm = state["signed_forward_mm"]
    session.fault = None


def measure(trace):
    p = np.asarray([r["position"] for r in trace])[:, [0, 2]]
    heading = np.unwrap([r["yaw"] for r in trace])
    elapsed = np.array([r["simTime"] - trace[0]["simTime"] for r in trace])
    dwell = p[(elapsed >= 0.5 - 1e-10) & (elapsed <= 1.5 + 1e-10)]
    return {
        "seconds": trace[-1]["simTime"] - trace[0]["simTime"],
        "signed_yaw_ui_rad": float(heading[-1] - heading[0]),
        "net_mm": float(np.linalg.norm(p[-1] - p[0])),
        "delta_xz_mm": (p[-1] - p[0]).tolist(),
        "path_mm": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
        "stop_window_path_mm": float(
            np.linalg.norm(np.diff(dwell, axis=0), axis=1).sum()
        )
        if elapsed[-1] >= 1.5 - 1e-10
        else None,
        "nonfoot_contact": any(r["contact"] for r in trace),
        "faults": [r["fault"] for r in trace if r["fault"]],
        "min_upright": min(r["upright"] for r in trace),
    }


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "probe_source.py")
    graph = GraphStore.load(args.graph)
    port_profile = (
        json.loads(args.port_profile.read_text(encoding="utf-8"))
        if args.port_profile
        else None
    )
    port_ids = (
        graph.resolve(["flywire:banc:888:" + root for root in port_profile["roots"]])
        if port_profile
        else np.array([], dtype=int)
    )
    if port_profile and any(
        graph.nodes[i]["super_class"] == "motor" or graph.nodes[i]["nt_type"] != "GABA"
        for i in port_ids
    ):
        raise ValueError(
            "This inhibitory port assay requires annotated non-motor GABA cells"
        )
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    else:
        left, right = "720575941500851362", "720575941626500746"
        cases = [{"name": "intact", "roots": [], "amplitude": 0}]
        for side, root in (("left", left), ("right", right)):
            for level in (20, 80, 200):
                cases.append(
                    {"name": f"{side}-{level}", "roots": [root], "amplitude": level}
                )
        for level in (20, 80):
            cases.append(
                {"name": f"both-{level}", "roots": [left, right], "amplitude": level}
            )
    write_json(
        args.out / "spec.json",
        {
            "experiment": args.experiment,
            "cases": cases,
            "seconds": args.seconds,
            "warmup_seconds": args.warmup,
            "seed": args.seed,
            "graph_hash": graph.hash,
            "wall_limit_seconds_per_case": 90,
            "hypothesis": "Anatomically selected non-periodic neural input controls physical steering or stopping",
            "comparator": "Identical neural, receptor, muscle and physical checkpoint; intact control",
            "prediction": "Opposite commands cause opposite signed yaw >=0.25 rad over 2s without fault/contact; short screens are exploratory only",
            "adoption": "Defer target-mode adoption until fixed goal criteria pass; never infer locomotion from neural change alone",
            "dn_incoming_scale": args.dn_incoming_scale,
            "dn_outgoing_scale": args.dn_outgoing_scale,
            "coxa_geometry": args.coxa_geometry,
            "ltm_tendons": args.ltm_tendons,
            "coxa_endpoint": args.coxa_endpoint,
            "neutral_margin_rad": args.neutral_margin,
            "middle_coxa_neutral_offset_rad": args.middle_coxa_offset,
            "port_profile": port_profile,
            "bilateral_DNg100_forward_drive": args.forward_drive,
            "sources": {
                str(p): file_hash(p)
                for p in [
                    Path(__file__),
                    Path("flylab/body.py"),
                    Path("flylab/c/banc_walking.py"),
                    Path("flylab/c/rate_body.py"),
                    Path("flylab/c/adaptive_rate.py"),
                ]
            },
        },
    )
    if args.dn_incoming_scale != 1.0 or args.dn_outgoing_scale != 1.0 or port_profile:
        import flylab.c.banc_walking as walking_module

        incoming, outgoing = np.ones(graph.n, np.float32), np.ones(graph.n, np.float32)
        dn = [
            i
            for i, n in enumerate(graph.nodes)
            if n["cell_type"] in ("DNa02", "DNg13", "DNg100")
        ]
        incoming[dn], outgoing[dn] = args.dn_incoming_scale, args.dn_outgoing_scale
        if port_profile:
            if port_profile.get("projection_scope", "all") == "all":
                outgoing[port_ids] = port_profile["outgoing_scale"]
            elif port_profile["projection_scope"] != "leg_motors":
                raise ValueError("Unknown static projection calibration scope")

        def calibrated(weights, *values, **kw):
            modified = weights.copy()
            modified.data *= outgoing[modified.indices]
            modified.data *= np.repeat(incoming, np.diff(modified.indptr))
            if port_profile and port_profile.get("projection_scope") == "leg_motors":
                motor = np.zeros(graph.n, bool)
                ids = sorted(
                    {
                        cell
                        for row in build_spec(graph, 2)["rows"]
                        for group in row["muscles"]
                        for cell in group["ids"]
                    }
                )
                motor[graph.resolve(ids)] = True
                port = np.zeros(graph.n, bool)
                port[port_ids] = True
                selected_edges = (
                    np.repeat(motor, np.diff(modified.indptr)) & port[modified.indices]
                )
                modified.data[selected_edges] *= port_profile["outgoing_scale"]
            return AdaptiveRateNetwork(modified, *values, **kw)

        walking_module.AdaptiveRateNetwork = calibrated
    s = BancWalkingSession(
        graph,
        seed=args.seed,
        device="cuda",
        parameters={
            "coxa_geometry": args.coxa_geometry,
            "ltm_tendons": args.ltm_tendons,
            "coxa_endpoint": args.coxa_endpoint,
        },
    )
    if port_profile or args.forward_drive:
        original_encode = s.adapter.encode

        def encode_port_background(**kwargs):
            drive = original_encode(**kwargs)
            if port_profile:
                drive[port_ids] += port_profile["background_input"]
            drive[s.descending] += args.forward_drive
            return drive

        s.adapter.encode = encode_port_background
    if args.neutral_margin or args.middle_coxa_offset:
        limits = s.body.m.jnt_range[s.body.joint_ids]
        original_neutral = s.body.neutral.copy()
        if args.neutral_margin:
            s.body.neutral = np.clip(
                s.body.neutral,
                limits[:, 0] + args.neutral_margin,
                limits[:, 1] - args.neutral_margin,
            )
        s.body.neutral[s.adapter.joints[[1, 4], 0]] += args.middle_coxa_offset
        changed = s.body.neutral != original_neutral
        if np.any(s.body.neutral[changed] < limits[changed, 0]) or np.any(
            s.body.neutral[changed] > limits[changed, 1]
        ):
            raise ValueError("Candidate neutral exceeds physical joint limits")
        s.adapter.identity = digest(
            {
                "base_adapter": s.adapter.identity,
                "neutral_target_rad": s.body.neutral.tolist(),
            }
        )
    warmup = [sample(s.body, s.control_tick)]
    for _ in range(round(args.warmup / 0.005)):
        s.advance(1)
        warmup.append(sample(s.body, s.control_tick))
        if s.fault:
            report = {
                "phase": "warmup",
                "fault": s.fault,
                "seconds": s.control_tick * 0.005,
                "validity": "valid",
                "hypothesis_verdict": "refuted",
                "claim": "Candidate reaches the required healthy initial state",
                "adoption": "reject",
                "milestone": "pending",
                "command_assays": "not_run",
            }
            (args.out / "warmup.jsonl").write_bytes(
                b"".join(canonical(r) + b"\n" for r in warmup)
            )
            write_json(args.out / "results.json", report)
            print(json.dumps(report), flush=True)
            s.close()
            return
    checkpoint = s.snapshot()
    summary = []
    try:
        for case in cases:
            restore(s, checkpoint)
            selected = (
                graph.resolve(
                    [
                        graph.nodes[graph.index["flywire:banc:888:" + root]]["id"]
                        for root in case["roots"]
                    ]
                )
                if case["roots"]
                else np.array([], dtype=int)
            )
            if case.get("outgoing_scale", 1.0) != 1.0:
                # Diagnostic only: scaling outgoing transmission is equivalent
                # to efficacy scaling on those existing weight columns. This
                # is NOT a binary causal-cut checkpoint or a source annotation.
                s.network.output_mask[selected] = float(case["outgoing_scale"])
            observed = port_ids if port_profile else selected
            trace, muscle, neural = [], [], []
            start = time.perf_counter()
            for i in range(round(args.seconds / 0.005) + 1):
                row = sample(s.body, s.control_tick)
                row["upright"] = float(s.body.pose()[1][2, 2])
                trace.append(row)
                if i == round(args.seconds / 0.005) or s.body.fault:
                    break
                if time.perf_counter() - start > 90:
                    raise TimeoutError("Per-condition assay limit")
                drive = s.adapter.encode()
                drive[selected] += case["amplitude"]
                s.network.advance(drive, s.neural_steps_per_control)
                if len(observed) and i % 20 == 0:
                    net = s.network
                    current = net._current(net.rate)[observed].detach().cpu().numpy()
                    actual_drive = net.drive[observed].detach().cpu().numpy()
                    adaptation = net.adaptation[observed].detach().cpu().numpy()
                    neural.append(
                        np.stack(
                            [net.readout(observed), actual_drive, current, adaptation]
                        )
                    )
                s.rate = s.network.readout(s.adapter.motor_ids)
                targets, adhesion = s.adapter.decode(s.rate)
                s.tendon_inputs.update(s.adapter.tendon_inputs)
                s.body.step_joint_targets(
                    targets, adhesion, tendon_inputs=s.tendon_inputs
                )
                s.control_tick += 1
                muscle.append(s.rate.copy())
            s._clocks()
            report = {
                "name": case["name"],
                "measurements": measure(trace),
                "cpg_unchanged": digest(s.body.snapshot()["cpg"]) == s.cpg_hash,
                "wall_seconds": time.perf_counter() - start,
                "validity": "valid",
                "hypothesis_verdict": "inconclusive",
                "adoption": "defer",
                "milestone": "pending",
            }
            report["diagnostic_outgoing_efficacy"] = case.get("outgoing_scale", 1.0)
            report["observed_neuron_ids"] = [graph.nodes[i]["id"] for i in observed]
            report["late_motor_mean"] = float(np.mean(muscle[-200:]))
            report.update(
                network_model_hash=s.network.identity,
                motor_adapter_hash=s.adapter.identity,
                coxa_motor_sign=s.adapter.coxa_sign.tolist(),
            )
            if neural:
                observation = np.array(neural)
                report["selected_neural"] = {
                    "rate_min": observation[:, 0].min(axis=0).tolist(),
                    "rate_max": observation[:, 0].max(axis=0).tolist(),
                    "drive_mean": observation[:, 1].mean(axis=0).tolist(),
                    "current_mean": observation[:, 2].mean(axis=0).tolist(),
                    "adaptation_mean": observation[:, 3].mean(axis=0).tolist(),
                    "net_margin_min": (
                        observation[:, 1] + observation[:, 2] - observation[:, 3] - 7.5
                    )
                    .min(axis=0)
                    .tolist(),
                    "net_margin_max": (
                        observation[:, 1] + observation[:, 2] - observation[:, 3] - 7.5
                    )
                    .max(axis=0)
                    .tolist(),
                }
            (args.out / (case["name"] + ".jsonl")).write_bytes(
                b"".join(canonical(r) + b"\n" for r in trace)
            )
            np.savez_compressed(
                args.out / (case["name"] + ".npz"),
                motor_rates=np.asarray(muscle),
                selected_neural=np.asarray(neural),
            )
            summary.append(report)
            write_json(args.out / "results.json", summary)
            print(json.dumps(report), flush=True)
    finally:
        s.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=Path)
    p.add_argument("--experiment", default="NAV-E1")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--warmup", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dn-incoming-scale", type=float, default=1.0)
    p.add_argument("--dn-outgoing-scale", type=float, default=1.0)
    p.add_argument("--coxa-geometry", action="store_true")
    p.add_argument("--neutral-margin", type=float, default=0.0)
    p.add_argument("--ltm-tendons", action="store_true")
    p.add_argument("--port-profile", type=Path)
    p.add_argument("--middle-coxa-offset", type=float, default=0.0)
    p.add_argument("--forward-drive", type=float, default=0.0)
    p.add_argument("--coxa-endpoint", action="store_true")
    run(p.parse_args())
