"""Physical whole-body command coverage and checkpoint comparison (GOAL_PLAN E1)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.c.integrity import write_json
from flylab.engine import config_values
from flylab.sensors import default_world


def run(
    out,
    servo_profile="asset",
    tracking=False,
    seed=42,
    include_legs=False,
    holdout=False,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        "schema": "flylab.whole-body-verification.v1",
        "seed": seed,
        "terrain": "flat",
        "physics": "CPU MuJoCo",
        "physical_dt_s": 0.0001,
        "command_dt_s": 0.005,
        "evaluator_revision": 2,
        "seconds_per_target": 0.12 if tracking else 0.03,
        "maximum_offset_rad": 0.1,
        "minimum_signed_response_rad": 1e-5,
        "comparison": "Opposite in-range targets from the identical integration state",
        "criterion": "Every active joint responds in target direction; no numerical warnings/fault, invalid requests atomic, CPU future restored",
        "interpretation": "Servo connectivity and local response, not calibrated tracking, neural control or flight",
        "wall_limit_s": 120 if tracking else 600,
        "model_limit_s": 20.0 if tracking else 10.0,
        "servo_profile": servo_profile,
        "include_legs": include_legs,
        "command_holdout": holdout,
        "command_seed": 20260911 if holdout else None,
        "maximum_center_shift_rad": 0.04 if holdout else 0.0,
        "test": "E4 all-joint tracking"
        if tracking and include_legs
        else "E3 nonleg tracking"
        if tracking
        else "E1 actuator coverage",
        "minimum_response_ratio": 0.90 if tracking else None,
        "maximum_target_error_rad": 0.02 if tracking else None,
    }
    write_json(out / "spec.json", spec)
    command_rng = np.random.default_rng(20260911)
    report = {
        "status": "RUNNING",
        "validity": "pending",
        "hypothesis": "not_tested",
        "adoption": "defer",
        "milestone": "pending",
        "cases": [],
        "physical_executed": True,
        "biological_validation": False,
    }
    start = time.perf_counter()
    body = None
    try:
        body = FlyGymBody(
            seed,
            default_world(),
            config_values(),
            body_options=BodyOptions(
                model="flybody", actuation="whole_body", servo_profile=servo_profile
            ),
        )
        ctl = body.whole_body_control
        initial = body.snapshot()
        write_json(out / "initial.json", initial)
        write_json(out / "inventory.json", ctl.observation())
        report.update(
            model_hash=body.model_hash,
            active=len(ctl.names),
            passive=len(ctl.passive_ids),
            initial_warning_counts=body.d.warning.number.tolist(),
            free_joints=int(
                np.count_nonzero(body.m.jnt_type == body.mj.mjtJoint.mjJNT_FREE)
            ),
        )
        traces = []
        for index, name in enumerate(ctl.names):
            if tracking and not include_legs and ctl.order[index].child.is_leg():
                continue
            if time.perf_counter() - start > spec["wall_limit_s"]:
                raise TimeoutError("E1 wall limit")
            lo, hi = ctl.limits[index]
            span = hi - lo
            center = np.clip(ctl.neutral[index], lo + 0.2 * span, hi - 0.2 * span)
            delta = min(0.1, 0.15 * span)
            if holdout:
                center = np.clip(
                    center + command_rng.uniform(-0.04, 0.04),
                    lo + 0.2 * span,
                    hi - 0.2 * span,
                )
                delta = min(0.07, 0.105 * span)
            results = []
            for sign in (-1, 1):
                body.restore(initial)
                target = float(center + sign * delta)
                before_time = body.physics_time()
                for _ in range(
                    round(spec["seconds_per_target"] / spec["command_dt_s"])
                ):
                    body.step_body_targets({name: target}, dt=spec["command_dt_s"])
                observation = ctl.observation()
                results.append(
                    {
                        "target_rad": target,
                        "angle_rad": observation["angles_rad"][index],
                        "actuator_force": observation["actuator_force"][index],
                        "fault": body.fault,
                        "finite": bool(
                            np.isfinite(body.d.qpos).all()
                            and np.isfinite(body.d.qvel).all()
                        ),
                        "elapsed_s": body.physics_time() - before_time,
                        "warning_counts": body.d.warning.number.tolist(),
                    }
                )
                traces.append({"joint": name, "sign": sign, "observation": observation})
            response = results[1]["angle_rad"] - results[0]["angle_rad"]
            passed = response > spec["minimum_signed_response_rad"] and all(
                r["finite"]
                and not r["fault"]
                and not any(r["warning_counts"])
                and abs(r["elapsed_s"] - spec["seconds_per_target"]) < 1e-10
                for r in results
            )
            target_error = max(abs(r["angle_rad"] - r["target_rad"]) for r in results)
            if tracking:
                passed = bool(
                    passed
                    and response / (2 * delta) >= spec["minimum_response_ratio"]
                    and target_error <= spec["maximum_target_error_rad"]
                )
            report["cases"].append(
                {
                    "name": name,
                    "region": observation["regions"][index],
                    "signed_response_rad": response,
                    "response_ratio": response / (2 * delta),
                    "target_error_rad": target_error,
                    "passed": passed,
                    "samples": results,
                }
            )
            write_json(out / "report.json", report)
        body.restore(initial)
        invalid_checks = []
        for bad in (
            {ctl.names[0]: float("nan")},
            {ctl.names[0]: True},
            {"unknown": 0.0},
            {ctl.names[0]: float(ctl.limits[0, 1] + 1)},
            {},
        ):
            before = body.snapshot()
            try:
                body.step_body_targets(bad)
                invalid_checks.append(False)
            except ValueError:
                invalid_checks.append(before == body.snapshot())
        before = body.snapshot()
        try:
            body.step_body_targets({ctl.names[0]: 0.0}, dt=0.00015)
            invalid_checks.append(False)
        except ValueError:
            invalid_checks.append(before == body.snapshot())
        # Simultaneous small commands, with sparse subsequent updates preserving them.
        commands = {
            name: float(np.clip(ctl.neutral[i] + 0.01, *ctl.limits[i]))
            for i, name in enumerate(ctl.names)
        }
        body.step_body_targets(commands, dt=0.01)
        saved = body.snapshot()
        body.step_body_targets(commands, dt=0.01)
        expected = body.snapshot()
        body.restore(saved)
        body.step_body_targets(commands, dt=0.01)
        actual = body.snapshot()
        future_equal = expected == actual
        future_max = float(
            np.max(np.abs(np.array(expected["state"]) - np.array(actual["state"])))
        )
        write_json(out / "traces.json", traces)
        write_json(
            out / "checkpoint-future.json",
            {"saved": saved, "expected": expected, "actual": actual},
        )
        passed = (
            all(r["passed"] for r in report["cases"])
            and all(invalid_checks)
            and future_equal
        )
        report.update(
            status="COMPLETE",
            validity="valid",
            hypothesis="supported" if passed else "refuted",
            adoption="adopt_coverage" if passed else "defer",
            coverage_passed=sum(r["passed"] for r in report["cases"]),
            invalid_requests_atomic=all(invalid_checks),
            future_exact=future_equal,
            future_max_integration_delta=future_max,
            simultaneous_fault=body.fault,
            simulated_seconds=2 * len(report["cases"]) * spec["seconds_per_target"]
            + 0.02,
            full_goal_completed=False,
        )
    except Exception as exc:  # noqa: BLE001 -- Preserve partial evidence; CLI reports failure.
        report.update(status="FAILED", validity="invalid", error=repr(exc))
    finally:
        if body is not None:
            body.close()
    report["wall_s"] = time.perf_counter() - start
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--servo-profile",
        choices=("asset", "tracking", "tracking_all"),
        default="asset",
    )
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-legs", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    result = run(
        args.out,
        args.servo_profile,
        args.tracking,
        args.seed,
        args.include_legs,
        args.holdout,
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=True)
    )
    if result.get("cases"):
        print(
            "Unpassed joints:", [r["name"] for r in result["cases"] if not r["passed"]]
        )
    raise SystemExit(0 if result.get("adoption") == "adopt_coverage" else 1)
