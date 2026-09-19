"""Compare an immutable source worktree with the current complete BANC session.

Loads the reference package under a distinct name (all relative dependencies
remain in that worktree). Both models execute sequentially on the same GPU.
Numerical recording and throughput measurement are separate experiments.
"""

import argparse
import copy
import importlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.storage import StateStore, runtime_versions
from flylab.c.walking_trace import TRACE_SCHEMA, WalkingTrace, summarize_trace
from tools.verify_c_banc_observation import exact, write_replay


def reference_class(root):
    root = root.resolve()
    name = "banc_source_reference"
    spec = importlib.util.spec_from_file_location(
        name,
        root / "flylab/__init__.py",
        submodule_search_locations=[str(root / "flylab")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    module = importlib.import_module(name + ".c.banc_walking")
    assert Path(module.__file__).resolve() == root / "flylab/c/banc_walking.py"
    assert module.FlyGymBody.__module__ == name + ".body"
    return module.BancWalkingSession


def source_identity(root):
    files = (
        "flylab/body.py",
        "flylab/c/banc_walking.py",
        "flylab/c/banc_mechanics.py",
        "flylab/c/rate_body.py",
        "flylab/c/neuromuscular.py",
        "flylab/kinematic_senses.py",
        "flylab/c/research_rate.py",
        "flylab/c/adaptive_rate.py",
        "flylab/c/adaptive_fused.py",
        "flylab/c/target_navigation.py",
        "flylab/c/target_neural.py",
        "flylab/c/walking_trace.py",
    )
    return {
        "root": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {p: file_hash(root / p) for p in files if (root / p).is_file()},
    }


def restore_same(session, saved):
    """Same-model reset outside timing, including navigation and tendon state."""
    session.body.restore(saved["body"])
    session.network.restore(saved["network"])
    session.adapter.restore(saved["adapter"])
    session.tendon_inputs.update(session.adapter.tendon_inputs)
    session.set_cuts(saved["cuts"])
    for key in ("control_tick", "signed_forward_mm", "wall_s"):
        setattr(session, key, saved[key])
    session.initial_position = saved["initial_position"].copy()
    session.rate = session.network.readout(session.adapter.motor_ids)
    if session.navigation:
        session.navigation.restore(
            saved["navigation"], time_s=session.control_tick * 0.005
        )
    session.fault = session.body.fault
    if hasattr(session, "trace"):
        session.trace = WalkingTrace()
        session.trace_error = None
    session._clocks()


def observe(session):
    body = session.body
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    body.mj.mj_getState(body.m, body.d, state, typ)
    return {
        "physical": state,
        "qpos": body.d.qpos.copy(),
        "qvel": body.d.qvel.copy(),
        "ctrl": body.d.ctrl.copy(),
        "motor": session.rate.copy(),
        "targets": np.asarray(body.last_action.joint_angles).copy(),
        "adhesion": np.asarray(body.last_action.adhesion_onoff).copy(),
    }


def advance(session, count, arm):
    if arm == "baseline":
        return session.advance(count)
    session.fused_readout = arm != "unfused"
    return session.advance(count, capture=arm == "recorded")


def record_run(session, start, arm, controls, out, reference=None):
    restore_same(session, start)
    records = {}
    rows = []
    began = time.perf_counter()
    for index in range(controls):
        advance(session, 1, arm)
        current = observe(session)
        for key, value in current.items():
            if reference is not None:
                expected = reference[key][index]
                if (
                    value.dtype != expected.dtype
                    or value.tobytes() != expected.tobytes()
                ):
                    raise AssertionError(
                        f"{arm} control {index + 1} {key}: bit mismatch"
                    )
            records.setdefault(key, []).append(value)
        if arm == "recorded":
            if session.trace_error:
                raise RuntimeError(session.trace_error)
            if index == 0:
                rows.append(copy.deepcopy(session.trace.samples[0]))
            rows.append(copy.deepcopy(session.trace.samples[-1]))
        if (index + 1) % 200 == 0:
            print(json.dumps({"arm": arm, "model_s": (index + 1) * 0.005}), flush=True)
        if time.perf_counter() - began > 300:
            raise TimeoutError("Numerical comparison exceeded 300 seconds per arm")
    arrays = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(out / (arm + "-signals.npz"), **arrays)
    final = session.snapshot()
    StateStore.save(out / (arm + "-final"), final)
    if rows:
        payload = dict(
            schema=TRACE_SCHEMA,
            epoch=session.trace.epoch,
            sample_dt_s=0.005,
            world=session.world,
            parameters=vars(session.p),
            graph_hash=session.graph.hash,
            model_hash=session.network.identity,
            body_hash=session.body.model_hash,
            adapter_hash=session.adapter.identity,
            samples=rows,
            **session.trace.export_metadata(
                include_initial=True, trace_error=session.trace_error
            ),
        )
        write_json(out / "trace.json", payload)
        write_json(out / "gait-summary.json", summarize_trace(rows))
        write_replay(out / "replay.html", payload)
        # The interval ending at each row uses that row's recorded command.
        session.body.restore(start["body"])
        for index, row in enumerate(rows[1:]):
            external = (row.get("neural_request") or {}).get("external_force_at_start")
            if external:
                session.body.push_left = external["push_left_s"]
                session.body.push_force = np.asarray(external["push_force"])
            session.body.step_joint_targets(
                np.asarray(row["joint_targets_rad"]),
                np.asarray(row["adhesion"], bool),
                tendon_inputs=row["tendon_inputs"],
                interpolate=not (row.get("neural_request") or {}).get("cuts", {}).get("motor", False),
                **({'joint_stiffness_scale': row['joint_stiffness_scale']} if 'joint_stiffness_scale' in row else {}),
                **({'adhesion_control': row['adhesion_control']} if 'adhesion_control' in row else {}),
            )
            actual = observe(session)["physical"]
            if actual.tobytes() != arrays["physical"][index].tobytes():
                raise AssertionError(
                    f"Command replay control {index + 1}: integration state differs"
                )
    return arrays, final, time.perf_counter() - began


def run(args):
    import torch

    variant_path = getattr(args, "candidate_parameters", None)
    if variant_path and (args.mode != "timing" or args.checkpoint):
        raise ValueError("Different model parameters require timing from fresh states")
    overrides = json.loads(variant_path.read_text(encoding="utf-8")) if variant_path else {}
    if not isinstance(overrides, dict):
        raise ValueError("Candidate parameters must be an object")
    if overrides.get("neural_dt_s", args.neural_dt) != args.neural_dt:
        raise ValueError("Use the same neural integration interval for this comparison")
    controls = round(args.seconds / 0.005)
    if not 1 <= controls <= 6000 or abs(controls * 0.005 - args.seconds) > 1e-10:
        raise ValueError("seconds must be a multiple of 5ms, at most 30s")
    args.out.mkdir(parents=True, exist_ok=False)
    reference = reference_class(args.baseline)
    spec = {
        "experiment": args.experiment,
        "mode": args.mode,
        "profile": args.profile,
        "baseline": source_identity(args.baseline),
        "candidate": source_identity(ROOT),
        "tool_sha256": file_hash(Path(__file__)),
        "runtime": runtime_versions(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seconds": args.seconds,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "repeats": args.repeats,
        "neural_dt_s": args.neural_dt,
        "arms": args.arms,
        "arm_semantics": "unfused means separate fault/motor readout; current CUDA integration and CPU optimizations are common to unfused/candidate/recorded",
        "order": "listed/reversed alternately",
        "decision": "Exact per-control motor/complete physics and final full state. Throughput: candidate <= 0.85 baseline; recorded <= 1.10 candidate. No gait verdict.",
        "excluded_state": ["wall_s"],
        "timing_excludes": "initialization, capture, warmup, restore, detailed parity snapshots, PNG, network transport, browser",
    }
    if variant_path:
        spec.update(
            candidate_parameters=overrides,
            candidate_parameters_sha256=file_hash(variant_path),
            decision="Same initial body; different declared neural/body-interface model. Within-arm future exactness and candidate median <=1.10 baseline. No cross-model future equality or gait pass inferred.",
        )
    write_json(args.out / "spec.json", spec)
    report = {"run_status": "running", "validity": "pending"}
    sessions = []
    try:
        graph = GraphStore.load(args.graph)
        params = {"neural_dt_s": args.neural_dt}
        navigation = None
        if args.profile == "target":
            params.update(coxa_geometry=True, ltm_tendons=True)
            navigation = {
                "target_xz_mm": [6.555187432018627, -2.9940043430431347],
                "forward_drive": 5000.0,
            }
        if args.checkpoint:
            original_saved = StateStore.load(args.checkpoint)
            expected_dt = original_saved["parameters"].get("neural_dt_s", 0.0001)
            if expected_dt != args.neural_dt:
                raise ValueError("Checkpoint and declared neural dt differ")
            base = reference.from_checkpoint(graph, original_saved)
        else:
            base = reference(
                graph, parameters=params, device="cuda", navigation=navigation
            )
        sessions.append(base)
        saved = base.snapshot()
        StateStore.save(args.out / "start", saved)
        candidate = (
            BancWalkingSession(
                graph, parameters={**params, **overrides}, device="cuda", navigation=navigation
            )
            if variant_path
            else BancWalkingSession.from_checkpoint(graph, saved)
        )
        sessions.append(candidate)
        candidate_saved = candidate.snapshot()
        if variant_path:
            # Mechanical capacity changes the model identity, not the shared
            # settled initial integration/controller state used for timing.
            exact({k: v for k, v in saved["body"].items() if k != "modelHash"},
                  {k: v for k, v in candidate_saved["body"].items() if k != "modelHash"})
            StateStore.save(args.out / "candidate-start", candidate_saved)
        else:
            exact(saved, candidate_saved)
        runtime_model = {
            "neurons": graph.n, "edges": len(graph.indices),
            "packed_lookup_entries": int(candidate.network._weight_values.numel()),
            "graph_hash": graph.hash, "network_hash": candidate.network.identity,
            "body_hash": candidate.body.model_hash,
        }
        write_json(args.out / "model.json", runtime_model)
        print(json.dumps(runtime_model), flush=True)
        if args.mode == "parity":
            arrays, final, duration = record_run(
                base, saved, "baseline", controls, args.out
            )
            results = [{"arm": "baseline", "wall_s_including_recording": duration}]
            for arm in args.arms:
                if arm == "baseline":
                    continue
                _, actual, duration = record_run(
                    candidate, saved, arm, controls, args.out, arrays
                )
                exact(final, actual)
                results.append(
                    {
                        "arm": arm,
                        "wall_s_including_recording": duration,
                        "bit_exact": True,
                    }
                )
                write_json(args.out / "arms.json", results)
            report.update(
                run_status="completed",
                validity="valid",
                hypothesis="supported",
                control_samples=controls,
                final_full_state_bit_exact=True,
                command_replay_bit_exact="recorded" in args.arms,
                arms=results,
            )
        else:
            rows, end_states = [], {}
            for pair in range(args.repeats):
                for arm in args.arms if pair % 2 == 0 else args.arms[::-1]:
                    session = base if arm == "baseline" else candidate
                    initial = saved if arm == "baseline" else candidate_saved
                    restore_same(session, initial)
                    advance(session, 10, arm)
                    restore_same(session, initial)
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    remaining = controls
                    while remaining:
                        count = min(10, remaining)
                        advance(session, count, arm)
                        remaining -= count
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - began
                    now = session.snapshot()
                    if variant_path and arm in end_states:
                        exact(end_states[arm], now)
                    elif not variant_path and end_states:
                        exact(next(iter(end_states.values())), now)
                    end_states[arm] = now
                    if getattr(session, "trace_error", None):
                        raise RuntimeError(session.trace_error)
                    row = {
                        "pair": pair,
                        "arm": arm,
                        "model_s": args.seconds,
                        "wall_s": wall,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    write_json(args.out / "runs.json", rows)
            medians = {
                arm: statistics.median(r["wall_s"] for r in rows if r["arm"] == arm)
                for arm in args.arms
            }
            report.update(
                run_status="completed",
                validity="valid",
                repeat_future_bit_exact=True,
                future_comparison="within each arm" if variant_path else "across all arms",
                medians=medians,
                rows=rows,
            )
            if "baseline" in medians and "candidate" in medians:
                report.update(
                    time_reduction=1 - medians["candidate"] / medians["baseline"],
                    performance_pass=medians["candidate"]
                    <= (1.10 if variant_path else 0.85) * medians["baseline"],
                )
            if "recorded" in medians and "candidate" in medians:
                report.update(
                    recording_overhead=medians["recorded"] / medians["candidate"] - 1,
                    recording_pass=medians["recorded"] <= 1.1 * medians["candidate"],
                )
    except AssertionError as exc:
        report.update(
            run_status="completed_with_mismatch",
            validity="valid",
            hypothesis="refuted",
            error=repr(exc),
        )
        raise
    except Exception as exc:
        report.update(run_status="aborted", validity="invalid", error=repr(exc))
        raise
    finally:
        write_json(args.out / "result.json", report)
        for session in sessions:
            session.close()
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("rows", "arms")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--candidate-parameters", type=Path,
        help="Timing only: different declared parameters, same fresh initial body; compare repeated futures within each arm",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/acquisitions/banc888-windows-20260910/bundle",
    )
    parser.add_argument("--profile", choices=("general", "target"), default="general")
    parser.add_argument("--mode", choices=("parity", "timing"), default="parity")
    parser.add_argument(
        "--neural-dt", type=float, choices=(0.0001, 0.0002, 0.00025), default=0.00025
    )
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--experiment", default="OBS-E1")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline", "unfused", "candidate", "recorded"),
        default=["baseline", "unfused", "candidate", "recorded"],
    )
    run(parser.parse_args())
