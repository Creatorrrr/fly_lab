"""NW2: real full-BANC adaptive neural/body trials, without a CPG fallback."""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.banc_walking import validate_banc_roster
from flylab.c.graph import GraphStore
from flylab.c.integrity import canonical, digest, file_hash, write_json
from flylab.c.rate_body import RateBodyAdapter
from flylab.c.research_annotations import rate_weights
from flylab.c.tasks import CRITERIA, evaluate
from flylab.common import to_ui
from flylab.engine import config_values
from flylab.sensors import default_world


def sample(body, tick):
    position, rotation, _ = body.pose()
    forward = to_ui(rotation[:, 0])
    return {
        "tick": tick,
        "simTime": tick * 0.005,
        "position": to_ui(position),
        "yaw": float(np.arctan2(forward[2], forward[0])),
        "forward_axis": forward,
        "contact": bool(body.nonfoot_contact()),
        "avoidance": False,
        "motion_enabled": True,
        "fault": body.fault,
    }


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    source_scope = validate_banc_roster(graph)
    weights, _ = rate_weights(graph)
    descending = np.array(
        [i for i, node in enumerate(graph.nodes) if node["cell_type"] == "DNg100"]
    )
    if len(descending) != 2:
        raise ValueError("Bilateral annotated DNg100 required")
    world = default_world()
    world["sources"], world["obstacles"] = [], []
    options = BodyOptions(
        model="flybody",
        actuation="whole_body",
        servo_profile="tracking_all",
        tendons="all",
    )
    cases = (
        [tuple(map(float, value.split(":"))) for value in args.case]
        if args.case
        else [(4.0, 0.0), (20.0, 0.0), (4.0, 18.0), (20.0, 18.0)]
    )
    write_json(
        args.out / "spec.json",
        {
            "experiment": "NW2",
            "graph_hash": graph.hash,
            "source_scope": source_scope,
            "neurons": graph.n,
            "edges": len(graph.indices),
            "body": options.model_identity(),
            "cases": [
                {"adaptation_gain": gain, "feedback_gain": sensory}
                for gain, sensory in cases
            ],
            "seconds": args.seconds,
            "seed": args.seed,
            "cut": args.cut,
            "wall_limit_s_per_case": 240.0,
            "dn_input": args.descending_drive,
            "input_status": "Measured leg receptors only"
            if args.descending_drive == 0
            else "Held descending input to isolate gait generation, not sensory autonomy",
            "tendon_input": 0.2,
            "motor_gain": args.motor_gain,
            "pooling": args.pooling,
            "adaptation_tau_s": args.adaptation_tau,
            "neural_dt_s": args.neural_dt,
            "cuda_implementation": args.cuda_implementation,
            "weights": "Source counts * .03 and existing NT signs; no edge normalization, deletion or rewiring",
            "cpg": False,
            "timed_reset": False,
            "phase_stimulus": False,
            "root_override": False,
            "hypothesis": "Sustained BANC motor variation with adaptation can yield stable forward gait through anatomical muscle mappings",
            "competing_explanation": "Uncoordinated activity, tonic antagonist bias, or incorrect muscle coupling yields immobility or falling",
            "decision": "Preserve fixed walking criteria. A shorter trial is only a screen; no adoption from neural variability or incomplete walking tests.",
            "walking_criteria": CRITERIA,
            "biological_validation": False,
            "sources": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path("flylab/c/rate_body.py"),
                    Path("flylab/c/adaptive_rate.py"),
                )
            },
        },
    )
    reports = []
    for gain, sensory in cases:
        name = f"gain-{gain:g}-feedback-{sensory:g}"
        directory = args.out / name
        directory.mkdir()
        began = time.perf_counter()
        body = FlyGymBody(args.seed, world, config_values(None), body_options=options)
        adapter = RateBodyAdapter(
            graph,
            body,
            sensory_gain=sensory,
            motor_gain=args.motor_gain,
            pooling=args.pooling,
        )
        cpg_before = digest(body.snapshot()["cpg"])
        net = AdaptiveRateNetwork(
            weights,
            *[np.full(graph.n, value, np.float32) for value in (0.02, 1.0, 7.5, 200.0)],
            adaptation_gain=gain,
            adaptation_tau_s=args.adaptation_tau,
            dt=args.neural_dt,
            capture_steps=10
            if args.neural_dt == 0.0001
            else round(0.005 / args.neural_dt),
            cuda_implementation=args.cuda_implementation,
        )
        if args.cut == "descending":
            net.set_muted(descending)
        trace, rates, targets, support, measured, contacts = (
            [sample(body, 0)],
            [],
            [],
            [],
            [],
            [],
        )
        tendon = {key: 0.2 for key in body.tendon_control.names}
        error = None
        try:
            with (directory / "trace.jsonl").open("wb") as stream:
                stream.write(canonical(trace[0]) + b"\n")
                for tick in range(1, round(args.seconds / 0.005) + 1):
                    if time.perf_counter() - began > 240:
                        raise TimeoutError("Bounded physical neural trial wall limit")
                    drive = adapter.encode(sensory_cut=args.cut == "sensory")
                    drive[descending] += args.descending_drive
                    net.advance(drive, round(0.005 / args.neural_dt))
                    rate = net.readout(adapter.motor_ids)
                    target, adhesion = adapter.decode(
                        rate, motor_cut=args.cut == "motor"
                    )
                    body.step_joint_targets(target, adhesion, tendon_inputs=tendon)
                    row = sample(body, tick)
                    stream.write(canonical(row) + b"\n")
                    trace.append(row)
                    rates.append(rate)
                    targets.append(target)
                    measured.append(body.d.qpos[body.qpos_ids].copy())
                    contact = body.contact_probe()
                    support.append(contact["floor_normal_bw"])
                    contacts.append(contact["floor_contact"])
                    if body.fault:
                        break
                    if tick % 200 == 0:
                        print(name, tick * 0.005, "model s", flush=True)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            error = str(exc)
        report = evaluate(trace, world, required_seconds=10.0)
        report.update(
            name=name,
            neural_seconds=net.tick * net.dt,
            wall_s=time.perf_counter() - began,
            error=error,
            cpg_unchanged=cpg_before == digest(body.snapshot()["cpg"]),
            model_hash=net.identity,
            body_hash=body.model_hash,
            adapter_hash=adapter.identity,
        )
        report["walking_verified"] = (
            report["task_status"] == "PASS"
            and error is None
            and report["cpg_unchanged"]
        )
        write_json(directory / "result.json", report)
        np.savez_compressed(
            directory / "signals.npz",
            motor_rates=np.asarray(rates),
            motor_ids=adapter.motor_ids,
            targets=np.asarray(targets),
            joint_angles=np.asarray(measured),
            joint_mapping=adapter.joints,
            support_bw=np.asarray(support),
            contacts=np.asarray(contacts),
            final_rates=net.readout(),
            final_adaptation=net.adaptation.cpu().numpy(),
        )
        print(
            {
                key: report[key]
                for key in (
                    "name",
                    "elapsed_s",
                    "horizontal_net_mm",
                    "task_status",
                    "error",
                    "cpg_unchanged",
                )
            },
            flush=True,
        )
        reports.append(report)
        write_json(args.out / "results.json", reports)
        body.close()
        del net, body, adapter
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--neural-dt", type=float, choices=(0.0001, 0.0002, 0.00025), default=0.0001
    )
    parser.add_argument(
        "--cuda-implementation", choices=("reference", "packed"), default="reference"
    )
    parser.add_argument("--seconds", type=float, choices=(3.0, 10.0), default=3.0)
    parser.add_argument(
        "--case", action="append", choices=("0:18", "4:0", "20:0", "4:18", "20:18")
    )
    parser.add_argument(
        "--cut", choices=("none", "sensory", "descending", "motor"), default="none"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--descending-drive", type=float, choices=(0.0, 400.0), default=0.0
    )
    parser.add_argument("--pooling", choices=("mean", "unit_response"), default="mean")
    parser.add_argument(
        "--motor-gain",
        type=float,
        choices=(0.0003, 0.001, 0.0015, 0.002, 0.003, 0.0045, 0.006),
        default=0.002,
    )
    parser.add_argument(
        "--adaptation-tau", type=float, choices=(0.03, 0.05, 0.15), default=0.15
    )
    run(parser.parse_args())
