"""E185: one gated E181 baseline with shadow ground-reaction telemetry.

Importing this file does not load a graph, checkpoint, MuJoCo, or CUDA.
The new force is recomputed by shadow mj_forward after integration; it is not
the preceding solver impulse. Native coordinates are z-up and model units.
"""

# Failure paths deliberately preserve the first error and partial raw records.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ENTRY_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

OUTPUT = ROOT / "verification/banc-natural-gait-m3-20260919/e185-ground-reaction"
DESIGN = "docs/BANC_M3_E185_GROUND_REACTION_DESIGN_20260924.md"
OWNED = (
    "tools/observe_c_banc_m3_ground_reaction.py",
    "tests/test_c_banc_ground_reaction_observer.py",
)
SCHEMA = "flylab.m3.e185.ground-reaction-freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e185.parent-source-review.v1"
INHERITED_S = 4236.676628199639
LIMIT_S = 4994.0
EVENT_INDICES = (131, 219, 308, 391, 433, 480, 521, 564)
EXPECTED_STEPS = 30000
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LINKS = (
    "coxa",
    "trochanterfemur",
    "tibia",
    "tarsus1",
    "tarsus2",
    "tarsus3",
    "tarsus4",
    "tarsus5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError("Repository-relative path required")
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Repository-relative path required")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Path escaped repository")
    return resolved


def sealed(path: Path, expected: str) -> dict[str, Any]:
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError("Exact sealed bytes required: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Sealed JSON object required")
    return value


def checked_map(mapping: Any) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Nonempty frozen SHA256 map required")
    for name, digest in mapping.items():
        sealed_bytes = safe(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or sha256(sealed_bytes) != digest
        ):
            raise ValueError("Frozen bytes changed: " + str(name))


def source_gate(args: argparse.Namespace) -> dict[str, Any]:
    freeze = sealed(args.freeze, args.freeze_sha256)
    required = {
        "schema": SCHEMA,
        "design_path": DESIGN,
        "owned_files": list(OWNED),
        "output_directory": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "theta": [0.0] * 6,
        "duration_s": 3.0,
        "controls": 600,
        "neural_steps_per_control": 20,
        "physics_steps_per_control": 50,
        "physics_dt_s": 0.0001,
        "native_samples": EXPECTED_STEPS,
        "event_indices": list(EVENT_INDICES),
        "inherited_cumulative_s": INHERITED_S,
        "cumulative_limit_s": LIMIT_S,
        "driver_wall_s": 150,
        "child_internal_s": 100,
        "child_external_s": 105,
        "cleanup_s": 5,
        "native_processes_max": 1,
        "automatic_retries": 0,
        "cpu_workers": 1,
        "gpu_workers": 1,
        "minimum_free_disk_bytes": 20 << 30,
        "baseline_raw_bytes_max": 1 << 30,
        "new_output_bytes_max": 2 << 30,
        "candidate_adopted": False,
        "goal_status": "active",
    }
    if any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError("E185 fixed observation contract changed")
    if sha256(safe(DESIGN)) != freeze.get("design_sha256"):
        raise ValueError("E185 design changed")
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or not set(OWNED).issubset(sources):
        raise ValueError("E185 source inventory incomplete")
    checked_map(sources)
    checked_map(freeze.get("preservation_sha256"))
    checked_map(freeze.get("e182_baseline_sha256"))
    metadata = sealed(safe(freeze["metadata_path"]), freeze["metadata_sha256"])
    if (
        metadata.get("e184_final_cost_s") is None
        or metadata.get("e184_raw_review") is None
    ):
        raise ValueError("Parent E184 metadata incomplete")
    old = freeze.get("e181_gate")
    if not isinstance(old, dict) or set(old) != {
        "freeze_path",
        "freeze_sha256",
        "source_review_path",
        "source_review_sha256",
    }:
        raise ValueError("Exact E181 gate paths required")
    for name, digest in (
        (old["freeze_path"], old["freeze_sha256"]),
        (old["source_review_path"], old["source_review_sha256"]),
    ):
        sealed(safe(name), digest)
    review = sealed(args.source_review, args.source_review_sha256)
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("freeze_sha256") != args.freeze_sha256
        or review.get("accepts_ground_reaction") is not True
        or review.get("blocking_defects") != []
        or review.get("source_sha256") != {name: sources[name] for name in OWNED}
    ):
        raise ValueError("Accepted E185 parent source review required")
    if (
        freeze["e182_baseline_directory"]
        != "verification/banc-natural-gait-m3-20260919/e182-native-timing-recovery/baseline"
    ):
        raise ValueError("Exact E182 baseline required")
    if not all(
        Path(name).is_relative_to(freeze["e182_baseline_directory"])
        for name in freeze["e182_baseline_sha256"]
    ):
        raise ValueError("E182 baseline inventory escaped baseline")
    return freeze


def fly_mass_inventory(body: Any) -> dict[str, Any]:
    ids = np.asarray(body.body_ids)
    masses = np.asarray(body.m.body_mass, dtype=float)
    parents = np.asarray(body.m.body_parentid, dtype=int)
    if (
        ids.ndim != 1
        or ids.size == 0
        or ids.dtype.kind not in "iu"
        or np.any(ids <= 0)
        or np.any(ids >= len(masses))
        or len(np.unique(ids)) != len(ids)
        or len(parents) != len(masses)
        or not np.isfinite(masses).all()
        or np.any(masses < 0)
    ):
        raise ValueError("Invalid fly body IDs or model masses")
    own = set(map(int, ids))
    roots = [int(i) for i in ids if int(parents[int(i)]) not in own]
    if len(roots) != 1:
        raise ValueError("Fly body IDs require one ancestry root")
    root = roots[0]
    for body_id in range(1, len(masses)):
        ancestor = body_id
        seen = set()
        while ancestor > 0 and ancestor not in seen and ancestor != root:
            seen.add(ancestor)
            ancestor = int(parents[ancestor])
        if ancestor == root and masses[body_id] > 0 and body_id not in own:
            raise ValueError("Positive-mass fly descendant missing from body IDs")
    total = float(masses[ids].sum())
    gravity = np.asarray(body.m.opt.gravity, dtype=float)
    if (
        not math.isfinite(total)
        or total <= 0
        or not math.isclose(total, float(body.mass), rel_tol=1e-10, abs_tol=1e-12)
        or gravity.shape != (3,)
        or not np.isfinite(gravity).all()
        or float(np.linalg.norm(gravity)) <= 0
        or not math.isfinite(float(body.weight0))
        or float(body.weight0) <= 0
        or not math.isclose(
            float(body.weight0),
            total * float(np.linalg.norm(gravity)),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Fly mass or weight identity changed")
    return {
        "body_ids": ids.astype(int).tolist(),
        "body_mass": masses[ids].tolist(),
        "body_mass_total": total,
        "body_mass_reported": float(body.mass),
        "weight0": float(body.weight0),
        "gravity": gravity.tolist(),
        "fly_root_body_id": root,
        "coordinates": "native MuJoCo z-up; mm",
        "force_unit": "MuJoCo model units; not SI calibrated",
        "force_timing": "post-integration shadow mj_forward instantaneous force, not prior solver impulse",
        "leg_order": list(LEGS),
        "leg_link_order": list(LINKS),
        "physics_identity": body.physics_identity(),
        "body_model_hash": body.model_hash,
    }


def shadow_force_sample(
    shadow: Any, inventory: dict[str, Any], rotation: np.ndarray
) -> dict[str, Any]:
    mapping, size = shadow._force_maps["feet"]
    mapping = np.asarray(mapping, dtype=int)
    leg_mapping, leg_size = shadow._force_maps["legs"]
    leg_mapping = np.asarray(leg_mapping, dtype=int)
    geom_bodyid = np.asarray(shadow.m.geom_bodyid, dtype=int)
    if (
        size != 30
        or leg_size != 48
        or len(mapping) != shadow.m.ngeom
        or len(leg_mapping) != shadow.m.ngeom
        or len(geom_bodyid) != shadow.m.ngeom
    ):
        raise ValueError("Exact 30-foot/48-leg force mappings required")
    segments = np.zeros((30, 3), dtype=float)
    leg_segments = np.zeros((48, 3), dtype=float)
    fly_force = np.zeros(3, dtype=float)
    wrench = np.zeros(6, dtype=float)
    fly_ids = set(inventory["body_ids"])
    for index, contact in enumerate(shadow.d.contact[: shadow.d.ncon]):
        if contact.exclude:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        if a < 0 or b < 0 or a >= len(mapping) or b >= len(mapping):
            raise ValueError("Contact geom out of range")
        foot_a, foot_b = int(mapping[a]), int(mapping[b])
        leg_a, leg_b = int(leg_mapping[a]), int(leg_mapping[b])
        fly_a, fly_b = int(geom_bodyid[a]) in fly_ids, int(geom_bodyid[b]) in fly_ids
        floor_a = a in shadow.support_geom_ids and shadow._ground_mask[a]
        floor_b = b in shadow.support_geom_ids and shadow._ground_mask[b]
        if not ((fly_a and floor_b) or (fly_b and floor_a)):
            continue
        shadow.mj.mj_contactForce(shadow.m, shadow.d, index, wrench)
        force = np.asarray(contact.frame, dtype=float).reshape(3, 3).T @ wrench[:3]
        if floor_b:
            if foot_a >= 0:
                segments[foot_a] -= force
            if leg_a >= 0:
                leg_segments[leg_a] -= force
            if fly_a:
                fly_force -= force
        if floor_a:
            if foot_b >= 0:
                segments[foot_b] += force
            if leg_b >= 0:
                leg_segments[leg_b] += force
            if fly_b:
                fly_force += force
    legs = segments.reshape(6, 5, 3).sum(axis=1)
    all_legs = leg_segments.reshape(6, 8, 3).sum(axis=1)
    residual = fly_force - all_legs.sum(axis=0)
    heading = np.asarray(rotation[:, 0], dtype=float).copy()
    planar = heading.copy()
    planar[2] = 0
    norm = float(np.linalg.norm(planar))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Undefined native XY heading")
    planar /= norm
    ids = np.asarray(inventory["body_ids"], dtype=int)
    mass = np.asarray(inventory["body_mass"], dtype=float)
    xipos = np.asarray(shadow.d.xipos, dtype=float)
    if len(xipos) <= int(ids.max()):
        raise ValueError("Missing shadow body xipos")
    com = np.average(xipos[ids], axis=0, weights=mass)
    position = np.asarray(shadow.d.xpos[shadow.thorax], dtype=float).copy()
    arrays = (
        segments,
        legs,
        leg_segments,
        all_legs,
        fly_force,
        residual,
        heading,
        planar,
        com,
        position,
        rotation,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite shadow force or pose")
    weight = inventory["weight0"]
    return {
        "segment_force": segments,
        "leg_force": legs,
        "segment_force_bw": segments / weight,
        "leg_force_bw": legs / weight,
        "all_leg_segment_force": leg_segments,
        "all_leg_force": all_legs,
        "full_fly_floor_force": fly_force,
        "nonleg_floor_force_residual": residual,
        "all_leg_segment_force_bw": leg_segments / weight,
        "all_leg_force_bw": all_legs / weight,
        "full_fly_floor_force_bw": fly_force / weight,
        "nonleg_floor_force_residual_bw": residual / weight,
        "thorax_position": position,
        "thorax_rotation": rotation.copy(),
        "forward": heading,
        "heading_xy": planar,
        "com_position": com,
        "leg_forward_force": legs @ heading,
        "leg_heading_xy_force": legs @ planar,
        "leg_forward_force_bw": (legs @ heading) / weight,
        "leg_heading_xy_force_bw": (legs @ planar) / weight,
        "all_leg_forward_force": all_legs @ heading,
        "all_leg_heading_xy_force": all_legs @ planar,
        "all_leg_forward_force_bw": (all_legs @ heading) / weight,
        "all_leg_heading_xy_force_bw": (all_legs @ planar) / weight,
        "full_fly_forward_force": float(fly_force @ heading),
        "full_fly_heading_xy_force": float(fly_force @ planar),
        "full_fly_forward_force_bw": float(fly_force @ heading) / weight,
        "full_fly_heading_xy_force_bw": float(fly_force @ planar) / weight,
    }


def install_native_observer(
    body: Any,
    samples: list,
    forces: list,
    inventory_out: dict,
    contact_diagnostics: Any,
    health_stop_type: type[Exception],
):
    """Copy E113's step/health tuple; add force from the same shadow forward."""
    native_step = body._step_physics
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    integration_type = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, integration_type))
    inventory = fly_mass_inventory(body)
    inventory_out.update(inventory)

    def step():
        native_step()
        body.mj.mj_getState(body.m, body.d, state, integration_type)
        body.mj.mj_setState(body.m, shadow.d, state, integration_type)
        body.mj.mj_forward(body.m, shadow.d)
        feet = contact_diagnostics(shadow)
        _, rotation, _ = shadow.pose()
        rotation = np.asarray(rotation, dtype=float)
        loads = np.asarray(feet["normal_load_bw"], dtype=float)
        slip = np.asarray(
            [np.nan if value is None else value for value in feet["slip_mm_s"]],
            dtype=float,
        )
        nonfoot = bool(shadow.nonfoot_contact())
        upright = float(rotation[2, 2])
        now = float(body.physics_time())
        samples.append(
            (now, loads.copy(), slip.copy(), np.isfinite(slip), nonfoot, upright)
        )
        health_bad = (
            not np.isfinite(loads).all()
            or not math.isfinite(upright)
            or not np.isfinite(body.d.qpos).all()
            or not np.isfinite(body.d.qvel).all()
            or nonfoot
            or upright < 0.9
            or body.fault
        )
        force_error = None
        try:
            row = shadow_force_sample(shadow, inventory, rotation)
            row["time_s"] = now
            row["available"] = True
            row["error"] = ""
            forces.append(row)
        except Exception as error:
            forces.append({"time_s": now, "available": False, "error": repr(error)})
            force_error = error
        if health_bad:
            raise health_stop_type("Native 100us health guard at " + str(now))
        if force_error is not None:
            raise force_error

    body._step_physics = step
    return lambda: setattr(body, "_step_physics", native_step)


FORCE_SHAPES = {
    "segment_force": (30, 3),
    "leg_force": (6, 3),
    "segment_force_bw": (30, 3),
    "leg_force_bw": (6, 3),
    "all_leg_segment_force": (48, 3),
    "all_leg_force": (6, 3),
    "full_fly_floor_force": (3,),
    "nonleg_floor_force_residual": (3,),
    "all_leg_segment_force_bw": (48, 3),
    "all_leg_force_bw": (6, 3),
    "full_fly_floor_force_bw": (3,),
    "nonleg_floor_force_residual_bw": (3,),
    "thorax_position": (3,),
    "thorax_rotation": (3, 3),
    "forward": (3,),
    "heading_xy": (3,),
    "com_position": (3,),
    "leg_forward_force": (6,),
    "leg_heading_xy_force": (6,),
    "leg_forward_force_bw": (6,),
    "leg_heading_xy_force_bw": (6,),
    "all_leg_forward_force": (6,),
    "all_leg_heading_xy_force": (6,),
    "all_leg_forward_force_bw": (6,),
    "all_leg_heading_xy_force_bw": (6,),
    "full_fly_forward_force": (),
    "full_fly_heading_xy_force": (),
    "full_fly_forward_force_bw": (),
    "full_fly_heading_xy_force_bw": (),
}


def force_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays = {
        "time_s": np.asarray([r["time_s"] for r in rows], dtype=float),
        "available": np.asarray([r["available"] for r in rows], dtype=bool),
        "error": np.asarray([r["error"] for r in rows], dtype=str),
    }
    for key, shape in FORCE_SHAPES.items():
        arrays[key] = np.asarray(
            [r[key] if r["available"] else np.full(shape, np.nan) for r in rows],
            dtype=float,
        ).reshape((len(rows), *shape))
    arrays["finite"] = np.asarray(
        [
            r["available"] and all(np.isfinite(r[key]).all() for key in FORCE_SHAPES)
            for r in rows
        ],
        dtype=bool,
    )
    return arrays


def write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_force_clock(
    arrays: dict[str, np.ndarray], original_time: np.ndarray
) -> dict[str, Any]:
    n = len(arrays["time_s"])
    clock = (
        n == EXPECTED_STEPS
        and len(original_time) == n
        and np.allclose(arrays["time_s"], original_time, rtol=0, atol=2e-8)
    )
    if clock:
        clock = np.allclose(
            arrays["time_s"], np.arange(1, n + 1) * 0.0001, rtol=0, atol=2e-8
        )
    return {
        "native_samples": n,
        "expected_native_samples": EXPECTED_STEPS,
        "clock_valid": bool(clock),
        "all_available": bool(arrays["available"].all()),
        "all_finite": bool(arrays["finite"].all()),
        "force_complete": bool(
            clock and arrays["available"].all() and arrays["finite"].all()
        ),
    }


def compare_full_control_rows(candidate: Path, reference: Path) -> dict[str, Any]:
    left = json.loads((candidate / "trace.json").read_text(encoding="utf-8"))["samples"]
    right = json.loads((reference / "trace.json").read_text(encoding="utf-8"))[
        "samples"
    ]
    if len(left) != 601 or len(right) != 601:
        raise ValueError("Exact E182 baseline requires 601 control rows")
    mismatches = []
    for index, (new, old) in enumerate(zip(left, right, strict=True)):
        if not math.isclose(new["time_s"], old["time_s"], rel_tol=0, abs_tol=2e-8):
            mismatches.append(f"row{index}.time_s")
        if {k: v for k, v in new.items() if k != "time_s"} != {
            k: v for k, v in old.items() if k != "time_s"
        }:
            mismatches.append(f"row{index}.all_nonclock_fields")
    return {
        "rows": 601,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "exact_reproduction": not mismatches,
    }


def child(args: argparse.Namespace, freeze: dict[str, Any]) -> dict[str, Any]:
    # Heavy E181 import and every original asset access begin only here.
    from flylab.c.walking_trace import contact_diagnostics
    from tools import optimize_c_banc_m3_native_timing as e181

    readiness = sealed(OUTPUT / "readiness.json", args.readiness_sha256)
    if (
        readiness.get("schema") != "flylab.m3.e185.readiness.v1"
        or readiness.get("freeze_sha256") != args.freeze_sha256
        or readiness.get("source_review_sha256") != args.source_review_sha256
        or len(readiness.get("observations", [])) != 2
        or not all(row.get("ready") is True for row in readiness["observations"])
    ):
        raise ValueError("E185 two-observation readiness not permitted")
    old = freeze["e181_gate"]
    gate = e181.verified_inputs(
        safe(old["freeze_path"]),
        old["freeze_sha256"],
        safe(old["source_review_path"]),
        old["source_review_sha256"],
        mode="baseline",
    )
    if gate.mode != "baseline":
        raise ValueError("Actual E181 baseline gate required")
    force_rows: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    old_hook = e181.protocol.install_native_observer
    original_native_samples: list | None = None
    body_restore = None

    def hook(body, samples):
        nonlocal original_native_samples, body_restore
        original_native_samples = samples
        body_restore = install_native_observer(
            body,
            samples,
            force_rows,
            inventory,
            contact_diagnostics,
            e181.protocol.NativeHealthStop,
        )
        return body_restore

    threads = None
    result = None
    primary = None
    persistence_errors: list[dict[str, str]] = []
    arrays: dict[str, np.ndarray] | None = None
    try:
        if shutil.disk_usage(OUTPUT.parent).free < 20 << 30:
            raise RuntimeError("E185 requires 20 GiB free")
        threads, _ = e181.protocol.joint_fit.one_cpu_thread()
        startup = e181._load_startup(gate.freeze)
        graph = e181.GraphStore.load(safe(gate.freeze["graph_path"]))
        if graph.hash != gate.freeze["graph_hash"]:
            raise ValueError("E181 graph identity changed")
        neutral = e181.StateStore.load(safe(gate.freeze["neutral_checkpoint"]))
        e181.protocol.install_native_observer = hook
        result = e181.run_arm(
            graph,
            neutral,
            (0.0,) * 6,
            OUTPUT / "baseline",
            startup=startup,
            limits=e181.LIMITS,
            wall_limit_s=100,
            gate=gate,
            process_began=ENTRY_STARTED,
        )
    except Exception as error:
        primary = error
        raise
    finally:
        e181.protocol.install_native_observer = old_hook
        cleanup_errors = []
        for action in (
            body_restore,
            threads.restore_original_limits if threads is not None else None,
        ):
            if action is not None:
                try:
                    action()
                except Exception as error:
                    cleanup_errors.append(repr(error))
        try:
            arrays = force_arrays(force_rows)
        except Exception as error:
            persistence_errors.append(
                {"artifact": "force_arrays", "error": repr(error)}
            )
        if arrays is not None:
            try:
                np.savez_compressed(OUTPUT / "ground-reaction.npz", **arrays)
            except Exception as error:
                persistence_errors.append(
                    {"artifact": "ground-reaction.npz", "error": repr(error)}
                )
                try:
                    np.savez(OUTPUT / "ground-reaction-uncompressed.npz", **arrays)
                except Exception as fallback_error:
                    persistence_errors.append(
                        {
                            "artifact": "ground-reaction-uncompressed.npz",
                            "error": repr(fallback_error),
                        }
                    )
        try:
            write_new(
                OUTPUT / "ground-reaction-metadata.json",
                {
                    "schema": "flylab.m3.e185.ground-reaction-metadata.v1",
                    "inventory": inventory,
                    "force_samples": len(force_rows),
                    "original_native_samples": len(original_native_samples)
                    if original_native_samples is not None
                    else 0,
                    "primary_error": repr(primary) if primary is not None else None,
                    "cleanup_errors": cleanup_errors,
                    "raw_persistence_errors": persistence_errors,
                },
            )
        except Exception as error:
            persistence_errors.append(
                {"artifact": "ground-reaction-metadata.json", "error": repr(error)}
            )
        if persistence_errors:
            # Streaming stderr remains available even if a metadata write fails.
            print(
                json.dumps({"observer_persistence_errors": persistence_errors}),
                file=sys.stderr,
            )
    original = (
        np.asarray([row[0] for row in original_native_samples], dtype=float)
        if original_native_samples is not None
        else np.empty(0)
    )
    force_check = (
        validate_force_clock(arrays, original)
        if arrays is not None
        else {
            "native_samples": len(force_rows),
            "force_complete": False,
            "error": "force_arrays_unavailable",
        }
    )
    complete_result = bool(
        result
        and result.get("validity") == "valid_3s_diagnostic"
        and result.get("complete_3s") is True
        and result.get("clock_valid") is True
        and result.get("startup_pass") is True
        and result.get("measured_health_stop") is False
        and result.get("technical_failure") is False
        and result.get("completed_controls") == 600
        and result.get("native_samples") == EXPECTED_STEPS
        and abs(float(result.get("actual_neural_s", -1)) - 3.0) <= 1e-10
        and abs(float(result.get("actual_physical_s", -1)) - 3.0) <= 1e-8
        and float(result.get("process_wall_s", math.inf)) <= 100
    )
    comparison = complete_rows = None
    comparison_errors: list[str] = []
    if (
        complete_result
        and force_check["force_complete"]
        and not persistence_errors
        and not cleanup_errors
    ):
        reference = safe(freeze["e182_baseline_directory"])
        try:
            comparison = e181.compare_baseline_prefix(OUTPUT / "baseline", reference)
            complete_rows = compare_full_control_rows(OUTPUT / "baseline", reference)
        except Exception as error:
            comparison_errors.append(repr(error))
    baseline_bytes = e181._directory_bytes(OUTPUT / "baseline")
    total_bytes = e181._directory_bytes(OUTPUT)
    check = {
        "schema": "flylab.m3.e185.observation-check.v1",
        "force": force_check,
        "e181_result": result,
        "e182_exact_comparison": comparison,
        "all_control_fields_comparison": complete_rows,
        "raw_persistence_errors": persistence_errors,
        "cleanup_errors": cleanup_errors,
        "comparison_errors": comparison_errors,
        "e181_complete_valid_3s": complete_result,
        "baseline_raw_bytes": baseline_bytes,
        "new_output_bytes": total_bytes,
        "child_elapsed_including_force_save_and_comparison_s": time.perf_counter()
        - ENTRY_STARTED,
        "observation_valid": bool(
            complete_result
            and not cleanup_errors
            and not persistence_errors
            and not comparison_errors
            and force_check["force_complete"]
            and comparison is not None
            and comparison["exact_reproduction"]
            and complete_rows is not None
            and complete_rows["exact_reproduction"]
            and baseline_bytes <= 1 << 30
            and total_bytes <= 2 << 30
            and time.perf_counter() - ENTRY_STARTED <= 100
        ),
        "candidate_adopted": False,
        "F1_F8_complete": False,
    }
    write_new(OUTPUT / "observation-check.json", check)
    return check


def supervise_child(
    command: list[str], e181: Any, monitor_stop: threading.Event
) -> dict[str, Any]:
    """Stream one owned child; inspect resources while it runs and reap its tree."""
    started = time.perf_counter()
    process = None
    known: dict[int, float] = {}
    status = "spawn_error"
    exit_code = None
    error = None
    termination = None
    cap_reason = None
    cleanup_wall_s = 0.0
    runtime_wall_s = 0.0
    live_after_exit_snapshot: list[int] = []
    exit_snapshot_cleared = False
    with (
        (OUTPUT / "child-stdout.txt").open("x", encoding="utf-8") as stdout,
        (OUTPUT / "child-stderr.txt").open("x", encoding="utf-8") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            root = e181.psutil.Process(process.pid)
            known[root.pid] = root.create_time()
            write_new(
                OUTPUT / "child-started.json",
                {
                    "schema": "flylab.m3.e185.child-started.v1",
                    "pid": process.pid,
                    "pid_created_s": known[process.pid],
                    "readiness_sha256": sha256(OUTPUT / "readiness.json"),
                },
            )
            status = "running"
            while True:
                e181._owned_children(process.pid, known)
                if time.perf_counter() - started >= 105:
                    cap_reason = "child_105s_timeout"
                elif time.perf_counter() - ENTRY_STARTED >= 125:
                    cap_reason = "driver_125s_cap"
                elif shutil.disk_usage(OUTPUT.parent).free < 20 << 30:
                    cap_reason = "free_disk_below_20GiB"
                elif e181._directory_bytes(OUTPUT / "baseline") > 1 << 30:
                    cap_reason = "baseline_raw_above_1GiB"
                elif e181._directory_bytes(OUTPUT) > 2 << 30:
                    cap_reason = "new_output_above_2GiB"
                if cap_reason:
                    status = "resource_guard"
                    break
                try:
                    exit_code = process.wait(timeout=0.5)
                    status = "completed" if exit_code == 0 else "child_nonzero"
                    break
                except subprocess.TimeoutExpired:
                    continue
            e181._owned_children(process.pid, known)
            remaining = e181._live_owned(known)
            if remaining and status == "completed":
                live_after_exit_snapshot = [p.pid for p in remaining]
                status = "unexpected_live_descendant"
        except Exception as exc:
            error = repr(exc)
            status = "supervisor_error" if process is not None else "spawn_error"
        finally:
            monitor_stop.set()
            runtime_wall_s = time.perf_counter() - started
            cleanup_began = time.perf_counter()
            if process is not None:
                e181._owned_children(process.pid, known)
                if termination is None:
                    termination = e181._terminate_owned_tree(process.pid, known)
                if termination["alive_pids"] or termination["errors"]:
                    status = "termination_uncertain"
                elif (
                    status == "unexpected_live_descendant"
                    and exit_code == 0
                    and not termination["terminated_pids"]
                ):
                    # A Windows exit snapshot may outlive wait() briefly. Only
                    # accept natural exit when final cleanup touched no process.
                    status = "completed"
                    exit_snapshot_cleared = True
                try:
                    process.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError):
                    status = "termination_uncertain"
            cleanup_wall_s = time.perf_counter() - cleanup_began
            if cleanup_wall_s > 5:
                status = "cleanup_budget_exceeded"
    receipt = {
        "schema": "flylab.m3.e185.child-supervisor.v1",
        "status": status,
        "exit_code": exit_code,
        "pid": process.pid if process else None,
        "owned_pid_birth_s": known,
        "termination": termination,
        "cap_reason": cap_reason,
        "error": error,
        "child_wall_s": runtime_wall_s,
        "cleanup_wall_s": cleanup_wall_s,
        "live_after_exit_snapshot": live_after_exit_snapshot,
        "exit_snapshot_cleared_without_termination": exit_snapshot_cleared,
        "supervisor_wall_s": time.perf_counter() - started,
        "stdout_path": "child-stdout.txt",
        "stderr_path": "child-stderr.txt",
    }
    write_new(OUTPUT / "child-supervisor.json", receipt)
    return receipt


def assess_monitor(
    rows: list[dict[str, Any]], errors: list[str], alive: bool
) -> dict[str, Any]:
    bad = []
    for index, row in enumerate(rows):
        if (
            row.get("queue_error")
            or row.get("gpu_error")
            or row.get("queue") != {"running": 0, "pending": 0}
            or not row.get("gpu_csv")
        ):
            bad.append(index)
    return {
        "samples": len(rows),
        "bad_sample_indices": bad,
        "monitor_errors": errors,
        "thread_alive_after_cleanup": alive,
        "resource_clear": bool(rows and not bad and not errors and not alive),
    }


def driver(args: argparse.Namespace, freeze: dict[str, Any]) -> dict[str, Any]:
    from tools import optimize_c_banc_m3_native_timing as e181
    from tools import recover_c_banc_m3_native_timing as e182

    readiness = e182._readiness()
    receipt = {
        "schema": "flylab.m3.e185.readiness.v1",
        "freeze_sha256": args.freeze_sha256,
        "source_review_sha256": args.source_review_sha256,
        "observations": readiness,
        "native_permitted": all(r.get("ready") is True for r in readiness),
        "parent_pid": os.getpid(),
        "parent_created_s": e181.psutil.Process(os.getpid()).create_time(),
    }
    write_new(OUTPUT / "readiness.json", receipt)
    if not receipt["native_permitted"]:
        raise RuntimeError("E185 readiness blocked; native not started")
    stop = threading.Event()
    monitor_errors: list[str] = []
    monitor = threading.Thread(
        target=e182._monitor,
        args=(OUTPUT / "resources.jsonl", stop, monitor_errors),
        daemon=True,
    )
    monitor.start()
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--freeze",
        str(args.freeze),
        "--freeze-sha256",
        args.freeze_sha256,
        "--source-review",
        str(args.source_review),
        "--source-review-sha256",
        args.source_review_sha256,
        "--child",
        "--readiness-sha256",
        sha256(OUTPUT / "readiness.json"),
    ]
    supervisor = None
    try:
        supervisor = supervise_child(command, e181, stop)
    finally:
        stop.set()
        monitor.join(
            timeout=max(0.0, 5.0 - supervisor["cleanup_wall_s"])
            if supervisor is not None
            else 5.0
        )
        if monitor.is_alive():
            monitor_errors.append("monitor_thread_did_not_stop")
        write_new(
            OUTPUT / "monitor-errors.json",
            {
                "errors": monitor_errors,
                "thread_alive_after_cleanup": monitor.is_alive(),
            },
        )
    monitor_check = assess_monitor(
        e182._resource_rows(OUTPUT / "resources.jsonl"),
        monitor_errors,
        monitor.is_alive(),
    )
    child_result = (
        json.loads((OUTPUT / "observation-check.json").read_text(encoding="utf-8"))
        if (OUTPUT / "observation-check.json").exists()
        else None
    )
    result = {
        "schema": "flylab.m3.e185.driver-assessment.v1",
        "child_supervisor": supervisor,
        "resource_monitor": monitor_check,
        "child_result": child_result,
        "driver_wall_s": time.perf_counter() - ENTRY_STARTED,
        "observation_valid": bool(
            supervisor["status"] == "completed"
            and supervisor["child_wall_s"] <= 105
            and supervisor["cleanup_wall_s"] <= 5
            and monitor_check["resource_clear"]
            and child_result
            and child_result.get("observation_valid") is True
            and time.perf_counter() - ENTRY_STARTED <= 125
        ),
        "candidate_adopted": False,
        "F1_F8_complete": False,
    }
    write_new(OUTPUT / "driver-assessment.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256"):
        parser.add_argument(
            "--" + key,
            required=True,
            type=Path if key in ("freeze", "source-review") else str,
        )
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--readiness-sha256")
    args = parser.parse_args(argv)
    if args.child:
        if not OUTPUT.is_dir() or not args.readiness_sha256:
            raise ValueError("Exact E185 child readiness required")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
    receipt_path = OUTPUT / ("child-attempt.json" if args.child else "attempt.json")
    write_new(
        receipt_path,
        {
            "schema": "flylab.m3.e185.attempt.v1",
            "stage": "started",
            "freeze_sha256": args.freeze_sha256,
            "source_review_sha256": args.source_review_sha256,
            "native_processes_max": 1,
            "candidate_adopted": False,
        },
    )
    stage = "source_gate"
    error = None
    result = None
    try:
        freeze = source_gate(args)
        stage = "child" if args.child else "driver"
        result = child(args, freeze) if args.child else driver(args, freeze)
        return 0 if result.get("observation_valid") is True else 1
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        return 1
    finally:
        write_new(
            OUTPUT / ("child-completion.json" if args.child else "completion.json"),
            {
                "schema": "flylab.m3.e185.completion.v1",
                "stage": stage,
                "error": error,
                "result": result,
                "elapsed_wall_s": time.perf_counter() - ENTRY_STARTED,
                "inherited_cumulative_s": INHERITED_S,
                "cumulative_with_this_attempt_s": INHERITED_S
                + time.perf_counter()
                - ENTRY_STARTED,
                "native_processes_max": 1,
                "candidate_adopted": False,
            },
        )
        if error:
            print(json.dumps(error), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
