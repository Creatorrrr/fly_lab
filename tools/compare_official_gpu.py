"""Compare official GPUSimulation, existing WarpWorldBatch, and CPU physics."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(out, worlds=2, steps=10):
    import warp as wp

    from flylab.body import FlyGymBody
    from flylab.body_options import BodyOptions
    from flylab.c.integrity import write_json
    from flylab.engine import config_values
    from flylab.official_gpu import OfficialGPUComparison
    from flylab.physics import PhysicsProfile, warp_option_metadata
    from flylab.sensors import default_world
    from flylab.warp_batch import WarpWorldBatch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "flylab.official-gpu-comparison.v1",
        "status": "RUNNING",
        "worlds": worlds,
        "steps": steps,
        "control_dt_s": 0.005,
        "scope": "identical held joint targets; physics and adapter costs only",
        "gpu_pair_tolerances": {"qpos": 1e-5, "qvel": 1e-4},
        "cpu_tolerances": {"qpos": 0.001, "qvel": 0.1},
        "default_changed": False,
        "neural_simulation": False,
        "biological_validation": False,
    }
    write_json(out / "spec.json", report)
    bodies = []
    official = current = None
    try:
        for i in range(worlds):
            bodies.append(
                FlyGymBody(
                    42 + i,
                    default_world(),
                    config_values(),
                    body_options=BodyOptions(model="flybody"),
                    physics_profile=PhysicsProfile(noslip_iterations=0, multiccd=False),
                )
            )
        b = bodies[0]
        targets = np.asarray([x.neutral for x in bodies])
        adhesion = np.zeros((worlds, 6), bool)
        commands = [{"forwardSpeed": 0.0, "yawRate": 0.0} for _ in bodies]
        current = WarpWorldBatch(bodies)
        official = OfficialGPUComparison(bodies)
        report["effective_options"] = {
            n: warp_option_metadata(m)
            for n, m in [
                ("official", official.sim.mjw_model),
                ("current", current.model),
            ]
        }
        saved_current = current.snapshot()
        saved_official = official.snapshot()
        # Both paths compile and instantiate their graphs before timing.
        for _ in range(2):
            current.advance(commands, targets, adhesion)
            official.advance(targets, adhesion)
        wp.synchronize()
        current.restore(saved_current)
        official.restore(saved_official)
        timings = {n: [] for n in ("official", "current", "cpu")}
        traces = {n: [] for n in timings}
        for _ in range(steps):
            for name, advance in [
                ("current", lambda: current.advance(commands, targets, adhesion)),
                ("official", lambda: official.advance(targets, adhesion)),
            ]:
                start = time.perf_counter()
                advance()
                wp.synchronize()
                timings[name].append(time.perf_counter() - start)
                d = current.data if name == "current" else official.sim.mjw_data
                traces[name].append((d.qpos.numpy().copy(), d.qvel.numpy().copy()))
            start = time.perf_counter()
            for x in bodies:
                x.step_joint_targets(x.neutral, np.zeros(6, bool))
            timings["cpu"].append(time.perf_counter() - start)
            traces["cpu"].append(
                (
                    np.asarray([x.d.qpos.copy() for x in bodies]),
                    np.asarray([x.d.qvel.copy() for x in bodies]),
                )
            )
            current.control.check_health()
            official.audit()
        arrays = {}
        for name, t in traces.items():
            arrays[name + "_qpos"] = np.asarray([x[0] for x in t])
            arrays[name + "_qvel"] = np.asarray([x[1] for x in t])
        np.savez_compressed(out / "trajectories.npz", **arrays)
        differences = {}
        for left, right in [
            ("official", "current"),
            ("official", "cpu"),
            ("current", "cpu"),
        ]:
            differences[left + "_vs_" + right] = {
                key: float(
                    np.max(np.abs(arrays[left + "_" + key] - arrays[right + "_" + key]))
                )
                for key in ("qpos", "qvel")
            }
        saved_official = official.snapshot()
        official.advance(targets, adhesion)
        official.audit()
        future = official.snapshot()
        official.restore(saved_official)
        official.advance(targets, adhesion)
        official.audit()
        repeated = official.snapshot()
        report.update(
            status="COMPLETE",
            model_hash=b.model_hash,
            differences=differences,
            timing_s=timings,
            median_control_wall_s={n: float(np.median(v)) for n, v in timings.items()},
            official_restore_max_abs={
                k: float(np.max(np.abs(future[k] - repeated[k])))
                for k in ("qpos", "qvel")
            },
            gpu_pair_pass=all(
                differences["official_vs_current"][k] <= v
                for k, v in report["gpu_pair_tolerances"].items()
            ),
            cpu_equivalence_pass=all(
                differences["official_vs_cpu"][k] <= v
                for k, v in report["cpu_tolerances"].items()
            ),
            timing_note="Synchronized samples after warmup. Official adapter includes host control transfer; current includes GPU controller/audit. No general speedup claim.",
        )
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write_json(out / "report.json", report)
        if official:
            official.close()
        if current:
            current.close()
        for b in bodies:
            b.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", type=int, choices=(1, 2, 4, 8), default=2)
    p.add_argument("--steps", type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.steps <= 100:
        p.error("Use 1..100 steps")
    run(a.out, a.worlds, a.steps)
