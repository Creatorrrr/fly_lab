"""Matched full-session throughput with explicit experimental dt regridding."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore


def reset_session(session, saved):
    """Restore a validated same-model snapshot outside the measured interval."""
    session.body.restore(saved["body"])
    session.network.restore(saved["network"])
    session.adapter.restore(saved["adapter"])
    session.set_cuts(saved["cuts"])
    session.control_tick = saved["control_tick"]
    session.initial_position = np.asarray(saved["initial_position"]).copy()
    session.signed_forward_mm = saved["signed_forward_mm"]
    session.wall_s = saved["wall_s"]
    session.rate = session.network.readout(session.adapter.motor_ids)
    session._clocks()


def experimental_common_start(graph, saved, dt, implementation):
    """Same r/z/body state with a deliberately different integrator identity.

    This is confined to the benchmark. Production checkpoint restore continues
    to reject incompatible models/dt. Integer clocks preserve the physical time.
    """
    parameters = dict(saved["parameters"], neural_dt_s=dt)
    session = BancWalkingSession(
        graph,
        seed=saved["seed"],
        parameters=parameters,
        world=saved["world"],
        device="cuda",
        cuda_implementation=implementation,
    )
    candidate = copy.deepcopy(saved)
    candidate["parameters"] = parameters
    candidate["network"] = session.network.snapshot()
    for key in ("rate", "adaptation", "drive", "output_mask"):
        candidate["network"][key] = saved["network"][key].copy()
    candidate["network"]["tick"] = (
        saved["control_tick"] * session.neural_steps_per_control
    )
    reset_session(session, candidate)
    return session, session.snapshot()


def run(args):
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    graph = GraphStore.load(args.graph)
    saved = StateStore.load(args.checkpoint)
    spec = {
        "experiment": "E-PERF-3",
        "evaluator": "flylab.banc-session-throughput.v1",
        "graph_hash": graph.hash,
        "checkpoint": str(args.checkpoint),
        "candidate_dt_s": args.neural_dt,
        "candidate_implementation": args.cuda_implementation,
        "baseline": "Original reference fixed-warp CSR, 0.1ms RK4, capture10",
        "warmup_controls_per_run": 10,
        "measured_controls_per_run": 30,
        "controls_per_call": 10,
        "repeats": 5,
        "order": "AB,BA,AB,BA,AB",
        "common_start": "Identical saved neural arrays, sensor/motor buffers, physical state and model time; experimental explicit integer-tick regrid for changed dt, own integrator identity; checkpoint files not modified",
        "includes": "Neural/physical integration, sensor and motor adaptation, readout, diagnostics and frame; all declared neurons and pair edges",
        "excludes": "Construction, compilation, warmup, restore, browser render/network/PNG",
        "acceptance": "baseline median wall / candidate median wall >= 2.0, no faults; bit exact entire future for same-dt candidate, otherwise separate P2/P3/P4 gates",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "sources": {
            str(p): file_hash(p)
            for p in (
                Path(__file__),
                Path("flylab/c/adaptive_rate.py"),
                Path("flylab/c/research_rate.py"),
                Path("flylab/c/banc_walking.py"),
            )
        },
    }
    write_json(args.out / "spec.json", spec)
    baseline = BancWalkingSession.from_checkpoint(
        graph, saved, cuda_implementation="reference"
    )
    candidate = None
    try:
        candidate, candidate_start = experimental_common_start(
            graph, saved, args.neural_dt, args.cuda_implementation
        )
        sessions = [baseline, candidate]
        starts = [baseline.snapshot(), candidate_start]
        walls, finals, rows = [[], []], [None, None], []
        for repeat in range(5):
            for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                session = sessions[index]
                reset_session(session, starts[index])
                session.advance(10)
                reset_session(session, starts[index])
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _ in range(3):
                    session.advance(10)
                wall = time.perf_counter() - began
                final = session.snapshot()
                if session.fault or session.body.fault:
                    raise AssertionError("Measured session faulted")
                if finals[index] is not None:
                    for key in ("rate", "adaptation", "drive", "output_mask"):
                        np.testing.assert_array_equal(
                            final["network"][key], finals[index]["network"][key]
                        )
                    np.testing.assert_array_equal(
                        final["body"]["state"], finals[index]["body"]["state"]
                    )
                finals[index] = final
                walls[index].append(wall)
                row = {
                    "repeat": repeat,
                    "variant": "baseline" if index == 0 else "candidate",
                    "wall_s": wall,
                    "model_s": 0.15,
                }
                rows.append(row)
                write_json(args.out / "runs.json", rows)
                print(json.dumps(row), flush=True)
        errors = {}
        for key in ("rate", "adaptation", "drive", "output_mask"):
            errors["network_" + key] = float(
                np.max(np.abs(finals[0]["network"][key] - finals[1]["network"][key]))
            )
        for key in ("filtered", "offset", "velocity_lowpass", "last_features"):
            errors["adapter_" + key] = float(
                np.max(np.abs(finals[0]["adapter"][key] - finals[1]["adapter"][key]))
            )
        errors["body_state"] = float(
            np.max(
                np.abs(
                    np.asarray(finals[0]["body"]["state"])
                    - np.asarray(finals[1]["body"]["state"])
                )
            )
        )
        if args.neural_dt == 0.0001 and max(errors.values()) != 0:
            raise AssertionError(
                "Same-model kernel changed the physical future: " + str(errors)
            )
        medians = [float(np.median(values)) for values in walls]
        ratio = medians[0] / medians[1]
        result = dict(
            spec,
            wall_samples=walls,
            median_wall_s=medians,
            throughput_model_s_per_wall_s=[0.15 / x for x in medians],
            speedup=ratio,
            performance_pass=ratio >= 2.0,
            final_errors=errors,
            repeat_futures_exact=True,
            model_hashes=[s.network.identity for s in sessions],
            implementation=[s.network.cuda_implementation for s in sessions],
        )
        write_json(args.out / "result.json", result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "median_wall_s",
                        "speedup",
                        "performance_pass",
                        "final_errors",
                    )
                }
            ),
            flush=True,
        )
    finally:
        baseline.close()
        if candidate is not None:
            candidate.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/neural-tendons/banc-walking-checkpoints/banc-757d5d9fe381"
        ),
    )
    parser.add_argument(
        "--neural-dt", type=float, choices=(0.0001, 0.0002, 0.00025), default=0.00025
    )
    parser.add_argument(
        "--cuda-implementation", choices=("reference", "packed"), default="packed"
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
