"""E156 body-only replay with the frozen causal substep recruitment path.

The original entry point validates the complete saved E150 recruitment history,
then changes only the 42 held joint controls during the frozen 1--2 s body
window.  It never constructs or advances a neural model.
"""

# ruff: noqa: I001  # Thread limits must precede numerical imports.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_info


FREEZE_SHA = "535afbfd25d6d210e1dba56bfab1923ca366694b77566a4f325a7100fbd335c4"
DESIGN_SHA = "3e03864e1f11915dc8c2ac69d04c823bdb11241f27f36a1a71c3c197b915f2eb"
OWNED = (
    "tools/verify_c_banc_m3_substep_body.py",
    "tests/test_c_banc_substep_body.py",
)
SCHEMA = "flylab.m3.e156.causal-substep-body-result.v1"
TRACE_SCHEMA = "flylab.m3.e150.native-monotone-body-trace.v1"
STATE_SCHEMA = "flylab.m3.e150.native-monotone-state.v1"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
CONTROL_DT = 0.005
PHYSICS_DT = 0.0001
SUBSTEPS = 50
ANCHOR_TICK = 200
FINAL_TICK = 400
SUPPORT_BW = 0.02
CLOCK_ATOL = 2e-8
UI_FROM_PHYSICS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_json(path: Path, value: Any) -> None:
    temporary = _temporary(Path(path))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = _temporary(Path(path))
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(relative: str) -> Path:
    value = Path(relative)
    path = (ROOT / value).resolve()
    require(
        not value.is_absolute()
        and ".." not in value.parts
        and path.is_relative_to(ROOT),
        "E156 frozen paths must be repository-relative",
    )
    return path


def verify_hash_map(entries: dict[str, str]) -> None:
    require(isinstance(entries, dict), "Frozen hash map required")
    for relative, digest in entries.items():
        require(
            sha256(safe_path(relative)) == digest, "Frozen hash mismatch: " + relative
        )


def load_freeze(path: Path, expected_sha: str) -> dict[str, Any]:
    require(
        expected_sha == FREEZE_SHA and sha256(path) == FREEZE_SHA,
        "Exact E156 freeze required",
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        value.get("schema") == "flylab.m3.e156.causal-substep-body.v1",
        "E156 freeze schema changed",
    )
    require(
        value.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(value["design"])) == DESIGN_SHA,
        "E156 design changed",
    )
    require(value.get("owned_files") == list(OWNED), "E156 owned files changed")
    require(
        len(value.get("input_sha256", {})) == 91
        and len(value.get("reused_source_sha256", {})) == 18
        and len(value.get("preservation_sha256", {})) == 2973,
        "E156 frozen hash inventory changed",
    )
    require(
        value.get("replay")
        == {
            "anchor_tick": 200,
            "final_tick": 400,
            "control_dt": 0.005,
            "physics_dt": 0.0001,
            "substeps_per_control": 50,
        },
        "E156 replay contract changed",
    )
    require(
        value.get("counts")
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_constructor": 1,
            "shadow_mjdata": 1,
            "settling_steps": 2000,
            "scientific_steps": 10000,
            "shadow_forward_calls": 10000,
            "neural_steps": 0,
            "fit_calls": 0,
            "helper_control_intervals": 2000,
        },
        "E156 execution counts changed",
    )
    require(
        value.get("resource_limits")
        == {
            "internal_s": 65,
            "external_s": 75,
            "supervisor_s": 85,
            "raw_bytes": 536870912,
            "free_bytes": 4294967296,
            "cpu_threads": 1,
            "gpu_count": 0,
        }
        and value.get("prior_conservative_s") == 2536
        and value.get("prospective_cumulative_s") == 2621,
        "E156 resource contract changed",
    )
    require(
        value.get("decision")
        == {
            "max_slip_ratio": 0.8,
            "min_forward_ratio": 0.9,
            "max_net_path_ratio_drop": 0.05,
            "min_knee_width_ratio": 0.9,
            "min_upright": 0.9,
            "nonfoot_allowed": 0,
        }
        and value.get("candidate_adopted") is False
        and value.get("whole_goal_complete") is False
        and value.get("F6_exposed") is False
        and value.get("active_UI_changed") is False,
        "E156 decision boundary changed",
    )
    return value


def validate_source_gate(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha256(path) == expected_sha, "Parent E156 source-review SHA changed")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_sources = {name: sha256(ROOT / name) for name in OWNED}
    require(
        review.get("accepts_single_substep_body") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1
        and review.get("source_sha256") == expected_sources,
        "Accepted exact E156 two-source gate required",
    )
    return review


def deadline(started: float, limit_s: float, operation: str) -> float:
    elapsed = time.perf_counter() - started
    if elapsed > limit_s:
        raise TimeoutError(f"E156 {operation} exceeded {limit_s}s: {elapsed:.6f}s")
    return elapsed


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def validate_parameters(parameters: dict[str, np.ndarray]) -> None:
    expected = {
        "tau": ((3,), np.float64),
        "kappa": ((391,), np.float64),
        "slopes": ((96, 391, 3, 2), np.float64),
        "membership": ((391,), np.int64),
        "motor_ids": ((391,), np.int64),
    }
    require(
        set(parameters) == set(expected),
        "Complete frozen E150 parameter bundle required",
    )
    for key, (shape, dtype) in expected.items():
        value = parameters[key]
        require(
            value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
            "Invalid E150 parameter: " + key,
        )


def make_helper_adapter(
    saved: dict[str, Any], parameters: dict[str, np.ndarray], neutral: Any
) -> Any:
    """Build a state-only adapter view; it has no decode or neural owner."""
    snapshot = saved.get("adapter")
    require(isinstance(snapshot, dict), "Checkpoint monotone adapter snapshot required")
    required = {
        "schema",
        "hash",
        "parameters",
        "calibration",
        "groups",
        "original",
        "bank",
        "activation",
        "last_motor",
    }
    require(
        set(snapshot) == required,
        "Complete checkpoint monotone adapter snapshot required",
    )
    fixed = np.asarray(snapshot["parameters"].get("fixed_muscles"))
    for key in ("tau", "kappa", "slopes", "membership"):
        require(
            np.array_equal(
                np.asarray(snapshot["parameters"].get(key)), parameters[key]
            ),
            "Checkpoint parameter differs: " + key,
        )
    calibration = snapshot["calibration"]
    require(
        np.array_equal(
            np.asarray(calibration.get("motor_ids")), parameters["motor_ids"]
        ),
        "Checkpoint motor IDs differ",
    )
    roster = snapshot["parameters"].get("motor_roster")
    require(isinstance(roster, list), "Checkpoint motor roster required")
    original_snapshot = snapshot.get("original")
    require(
        isinstance(original_snapshot, dict)
        and isinstance(original_snapshot.get("hash"), str),
        "Original adapter identity required",
    )
    original = SimpleNamespace(
        identity=original_snapshot["hash"],
        activation=np.array(
            original_snapshot.get("activation"), dtype=np.float64, copy=True
        ),
    )
    return SimpleNamespace(
        schema=snapshot["schema"],
        identity=snapshot["hash"],
        original=original,
        body=SimpleNamespace(neutral=np.array(neutral, dtype=np.float64, copy=True)),
        calibration={
            key: np.array(value, copy=True) for key, value in calibration.items()
        },
        groups=copy.deepcopy(snapshot["groups"]),
        tau=np.array(parameters["tau"], copy=True),
        kappa=np.array(parameters["kappa"], copy=True),
        slopes=np.array(parameters["slopes"], copy=True),
        membership=np.array(parameters["membership"], copy=True),
        motor_roster=tuple(roster),
        fixed_muscles=np.array(fixed, dtype=np.int64, copy=True),
        variable_muscles=np.setdiff1d(np.arange(96, dtype=np.int64), fixed),
        lower_budget=np.array(snapshot["parameters"].get("lower_budget"), copy=True),
        upper_budget=np.array(snapshot["parameters"].get("upper_budget"), copy=True),
        motor_minimum_slope=np.array(
            snapshot["parameters"].get("motor_minimum_slope"), copy=True
        ),
        bank=np.array(snapshot["bank"], dtype=np.float64, copy=True),
        activation=np.array(snapshot["activation"], dtype=np.float64, copy=True),
    )


def validate_saved_inputs(
    trace: dict[str, Any],
    native: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    body_hash: str,
) -> list[dict[str, Any]]:
    validate_parameters(parameters)
    rows = trace.get("samples")
    require(
        trace.get("schema") == TRACE_SCHEMA
        and trace.get("body_hash") == body_hash
        and trace.get("E150_provenance", {}).get("numeric_protocol")
        == "unchanged E141 native body protocol"
        and isinstance(rows, list)
        and len(rows) == 2001,
        "Complete frozen E150 trace required",
    )
    ticks = np.asarray([row.get("control_tick") for row in rows])
    times = np.asarray([row.get("time_s") for row in rows], dtype=np.float64)
    require(
        np.array_equal(ticks, np.arange(2001)),
        "Exact trace tick/index identity required",
    )
    require(
        np.array_equal(times, np.arange(2001) * CONTROL_DT),
        "Exact trace control clock required",
    )
    required_state = {
        "schema",
        "control_tick",
        "network_tick",
        "physics_time_s",
        "complete",
        "bank",
        "bank_available",
        "activation",
        "activation_available",
        "original_activation",
        "original_activation_available",
        "motor_rate",
        "motor_rate_available",
    }
    require(
        set(native) == required_state, "Complete E150 native recruitment state required"
    )
    schema = np.asarray(native["schema"])
    require(
        schema.ndim == 0 and str(schema) == STATE_SCHEMA,
        "E150 recruitment state schema changed",
    )
    shapes = {
        "control_tick": (2001,),
        "network_tick": (2001,),
        "physics_time_s": (2001,),
        "complete": (2001,),
        "bank": (2001, 391, 3),
        "bank_available": (2001,),
        "activation": (2001, 96),
        "activation_available": (2001,),
        "original_activation": (2001, 96),
        "original_activation_available": (2001,),
        "motor_rate": (2001, 391),
        "motor_rate_available": (2001,),
    }
    for key, shape in shapes.items():
        require(native[key].shape == shape, "E150 native shape differs: " + key)
    require(
        native["control_tick"].dtype == np.int64
        and native["network_tick"].dtype == np.int64
        and all(
            native[key].dtype == np.dtype(bool)
            for key in (
                "complete",
                "bank_available",
                "activation_available",
                "original_activation_available",
                "motor_rate_available",
            )
        )
        and all(
            native[key].dtype == np.float64
            for key in (
                "physics_time_s",
                "bank",
                "activation",
                "original_activation",
                "motor_rate",
            )
        ),
        "E150 native dtype differs",
    )
    require(
        np.array_equal(native["control_tick"], np.arange(2001)),
        "Native control ticks differ",
    )
    require(
        np.array_equal(native["network_tick"], 20 * np.arange(2001)),
        "Native network ticks differ",
    )
    require(
        np.allclose(native["physics_time_s"], times, atol=CLOCK_ATOL, rtol=0),
        "Native physical clocks differ",
    )
    require(
        all(
            np.all(native[key])
            for key in (
                "complete",
                "bank_available",
                "activation_available",
                "original_activation_available",
                "motor_rate_available",
            )
        ),
        "Incomplete E150 recruitment row",
    )
    require(
        all(
            np.isfinite(native[key]).all() and np.all(native[key] >= 0)
            for key in ("bank", "activation", "original_activation", "motor_rate")
        ),
        "Invalid E150 recruitment values",
    )
    motor_ids = parameters["motor_ids"]
    for index, row in enumerate(rows):
        rates = np.asarray(row.get("motor_rates"), dtype=np.float64)
        require(
            rates.ndim == 1 and np.isfinite(rates).all(),
            f"Trace motor rates invalid at {index}",
        )
        selected = (
            rates
            if rates.shape == (391,)
            else rates[motor_ids]
            if len(rates) > int(motor_ids[-1])
            else None
        )
        require(
            selected is not None
            and np.array_equal(selected, native["motor_rate"][index]),
            f"Trace/native motor rate differs at {index}",
        )
        target = np.asarray(row.get("joint_targets_rad"), dtype=np.float64)
        require(
            target.shape == (42,) and np.isfinite(target).all(),
            f"Trace joint target invalid at {index}",
        )
        require(
            np.asarray(row.get("qpos"), dtype=np.float64).ndim == 1
            and np.isfinite(row["qpos"]).all(),
            f"Trace qpos invalid at {index}",
        )
        require(
            np.asarray(row.get("qvel"), dtype=np.float64).ndim == 1
            and np.isfinite(row["qvel"]).all(),
            f"Trace qvel invalid at {index}",
        )
        require(
            np.asarray(row.get("ctrl"), dtype=np.float64).ndim == 1
            and np.isfinite(row["ctrl"]).all(),
            f"Trace ctrl invalid at {index}",
        )
        require(
            np.asarray(row.get("adhesion")).shape == (6,)
            and np.asarray(row["adhesion"]).dtype == np.dtype(bool),
            f"Trace adhesion invalid at {index}",
        )
        require(
            isinstance(row.get("tendon_inputs"), dict)
            and all(np.isfinite(value) for value in row["tendon_inputs"].values()),
            f"Trace tendon inputs invalid at {index}",
        )
        require(row.get("fault") in (None, False, ""), f"Trace fault at {index}")
    return rows


def new_helper_registry() -> dict[str, list[Any]]:
    registry = {
        key: []
        for key in (
            "tick",
            "complete",
            "selected",
            "bank_before",
            "fixed_before",
            "rate",
            "endpoint_bank",
            "endpoint_fixed",
            "endpoint_activation",
            "endpoint_target",
            "endpoint_error",
        )
    }
    registry["selected_tick"] = []
    registry["selected_joint_target"] = []
    return registry


def _append_helper_attempt(
    registry: dict[str, list[Any]],
    tick: int,
    native: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    values = {
        "tick": tick,
        "complete": False,
        "selected": ANCHOR_TICK < tick <= FINAL_TICK,
        "bank_before": native["bank"][tick - 1],
        "fixed_before": native["original_activation"][tick - 1],
        "rate": native["motor_rate"][tick],
        "endpoint_bank": native["bank"][tick],
        "endpoint_fixed": native["original_activation"][tick],
        "endpoint_activation": native["activation"][tick],
        "endpoint_target": np.asarray(row["joint_targets_rad"], dtype=np.float64),
        "endpoint_error": np.full(4, np.nan),
    }
    for key, value in values.items():
        registry[key].append(
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )


def helper_arrays(registry: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    count = len(registry["tick"])
    shapes = {
        "bank_before": (391, 3),
        "fixed_before": (96,),
        "rate": (391,),
        "endpoint_bank": (391, 3),
        "endpoint_fixed": (96,),
        "endpoint_activation": (96,),
        "endpoint_target": (42,),
        "endpoint_error": (4,),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, values in registry.items():
        if key in ("selected_tick", "selected_joint_target"):
            continue
        dtype = (
            bool
            if key in ("complete", "selected")
            else np.int64
            if key == "tick"
            else np.float64
        )
        array = np.asarray(values, dtype=dtype)
        if key in shapes:
            array = array.reshape(count, *shapes[key])
        arrays[key] = array
    selected_count = len(registry["selected_tick"])
    arrays["selected_tick"] = np.asarray(registry["selected_tick"], dtype=np.int64)
    arrays["selected_joint_target"] = np.asarray(
        registry["selected_joint_target"], dtype=np.float64
    ).reshape(selected_count, SUBSTEPS, 42)
    return arrays


def reconstruct_endpoints(
    helper: Any,
    adapter: Any,
    rows: list[dict[str, Any]],
    native: dict[str, np.ndarray],
    registry: dict[str, list[Any]],
    ledger: dict[str, Any],
    *,
    on_interval: Callable[[int], None] | None = None,
) -> np.ndarray:
    error_keys = (
        "bank_max_abs",
        "fixed_activation_max_abs",
        "activation_max_abs",
        "target_max_abs",
    )
    for tick in range(1, 2001):
        row = rows[tick]
        _append_helper_attempt(registry, tick, native, row)
        ledger["helper_intervals_requested"] += 1
        adapter.bank = np.array(native["bank"][tick], copy=True)
        adapter.original.activation = np.array(
            native["original_activation"][tick], copy=True
        )
        adapter.activation = np.array(native["activation"][tick], copy=True)
        result = helper.compute(
            native["bank"][tick - 1],
            native["original_activation"][tick - 1],
            native["motor_rate"][tick],
            row["joint_targets_rad"],
            motor_cut=False,
            dt=CONTROL_DT,
            physics_dt=PHYSICS_DT,
        )
        ledger["helper_intervals_returned"] += 1
        registry["endpoint_error"][-1] = np.asarray(
            [result.endpoint_error[key] for key in error_keys], dtype=np.float64
        )
        registry["complete"][-1] = True
        ledger["helper_intervals_completed"] += 1
        if ANCHOR_TICK < tick <= FINAL_TICK:
            require(
                result.joint_targets.shape == (SUBSTEPS, 42),
                "Selected helper trajectory shape differs",
            )
            registry["selected_tick"].append(tick)
            registry["selected_joint_target"].append(
                np.array(result.joint_targets, copy=True)
            )
        if on_interval is not None:
            on_interval(tick)
    trajectory = np.asarray(registry["selected_joint_target"], dtype=np.float64)
    require(
        trajectory.shape == (200, 50, 42),
        "Complete selected helper trajectory required",
    )
    return trajectory


def validate_checkpoint_anchor(
    saved: dict[str, Any], native: dict[str, np.ndarray], adapter: Any
) -> None:
    require(
        saved.get("control_tick") == ANCHOR_TICK
        and saved.get("network", {}).get("tick") == 20 * ANCHOR_TICK,
        "Checkpoint clock differs",
    )
    snapshot = saved["adapter"]
    require(
        np.array_equal(snapshot["bank"], native["bank"][ANCHOR_TICK])
        and np.array_equal(snapshot["activation"], native["activation"][ANCHOR_TICK])
        and np.array_equal(
            snapshot["original"]["activation"],
            native["original_activation"][ANCHOR_TICK],
        )
        and snapshot["hash"] == adapter.identity
        and snapshot["original"]["hash"] == adapter.original.identity,
        "Checkpoint/native recruitment anchor differs",
    )


def validate_baseline(
    baseline_native: dict[str, np.ndarray],
    baseline_controls: dict[str, np.ndarray],
    baseline_context: dict[str, Any],
    body_hash: str,
) -> None:
    native_shapes = {
        "time_s": (10000,),
        "rotation_world": (10000, 3, 3),
        "heading_horizontal": (10000, 3),
        "force_world_bw": (10000, 6, 3),
        "force_forward_bw": (10000, 6),
        "normal_load_bw": (10000, 6),
        "slip_mm_s": (10000, 6),
        "slip_valid": (10000, 6),
        "nonfoot_contact": (10000,),
        "upright": (10000,),
        "control_tick": (10000,),
    }
    require(
        set(baseline_native) == set(native_shapes),
        "Complete E154 baseline native raw required",
    )
    for key, shape in native_shapes.items():
        require(
            baseline_native[key].shape == shape,
            "E154 baseline native shape differs: " + key,
        )
    require(
        set(baseline_controls)
        == {"tick", "time_s", "qpos", "qvel", "ctrl", "verified"},
        "Complete E154 controls required",
    )
    require(
        np.array_equal(
            baseline_controls["tick"], np.arange(ANCHOR_TICK, FINAL_TICK + 1)
        )
        and np.all(baseline_controls["verified"])
        and np.allclose(
            baseline_controls["time_s"],
            np.arange(ANCHOR_TICK, FINAL_TICK + 1) * CONTROL_DT,
            atol=CLOCK_ATOL,
            rtol=0,
        ),
        "E154 baseline control window differs",
    )
    require(
        baseline_context.get("body_hash") == body_hash,
        "E154 baseline body identity differs",
    )
    require(
        all(
            np.isfinite(baseline_native[key]).all()
            for key in (
                "time_s",
                "rotation_world",
                "heading_horizontal",
                "force_world_bw",
                "force_forward_bw",
                "normal_load_bw",
                "upright",
            )
        ),
        "Nonfinite E154 baseline raw",
    )
    require(
        np.array_equal(
            np.isfinite(baseline_native["slip_mm_s"]), baseline_native["slip_valid"]
        ),
        "E154 slip validity differs",
    )
    require(
        np.allclose(
            baseline_native["time_s"],
            ANCHOR_TICK * CONTROL_DT
            + np.arange(1, 10001, dtype=np.float64) * PHYSICS_DT,
            atol=CLOCK_ATOL,
            rtol=0,
        )
        and np.array_equal(
            baseline_native["control_tick"], np.repeat(np.arange(201, 401), 50)
        ),
        "E154 native clock/tick alignment differs",
    )


def new_body_registry(nq: int, nv: int, nu: int) -> dict[str, Any]:
    return {
        "shape": (int(nq), int(nv), int(nu)),
        "trace_rows": [],
        "applied": {
            key: []
            for key in (
                "tick",
                "substep",
                "time_before_s",
                "ctrl",
                "integration_complete",
                "observation_complete",
            )
        },
        "native": {
            key: []
            for key in (
                "tick",
                "substep",
                "time_s",
                "rotation_world",
                "normal_load_bw",
                "slip_mm_s",
                "slip_valid",
                "nonfoot_contact",
                "upright",
                "observation_complete",
            )
        },
        "native_attempts": {
            key: []
            for key in (
                "tick",
                "substep",
                "time_s",
                "rotation_shape",
                "load_shape",
                "slip_shape",
                "rotation_world",
                "normal_load_bw",
                "slip_mm_s",
                "returned",
                "complete",
            )
        },
        "controls": {
            key: []
            for key in (
                "tick",
                "time_s",
                "qpos",
                "qvel",
                "ctrl",
                "position",
                "rotation_world",
                "joint_target",
                "knee_flexion",
                "captured",
            )
        },
    }


def body_arrays(
    registry: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    nq, nv, nu = registry["shape"]
    applied_shapes = {"ctrl": (nu,)}
    native_shapes = {
        "rotation_world": (3, 3),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
        "slip_valid": (6,),
    }
    attempt_shapes = {
        "rotation_shape": (2,),
        "load_shape": (2,),
        "slip_shape": (2,),
        "rotation_world": (3, 3),
        "normal_load_bw": (6,),
        "slip_mm_s": (6,),
    }
    control_shapes = {
        "qpos": (nq,),
        "qvel": (nv,),
        "ctrl": (nu,),
        "position": (3,),
        "rotation_world": (3, 3),
        "joint_target": (42,),
        "knee_flexion": (6,),
    }
    result = []
    for section, shapes in (
        ("applied", applied_shapes),
        ("native", native_shapes),
        ("native_attempts", attempt_shapes),
        ("controls", control_shapes),
    ):
        count = len(registry[section]["tick"])
        arrays = {}
        for key, values in registry[section].items():
            dtype = (
                bool
                if key
                in (
                    "integration_complete",
                    "observation_complete",
                    "slip_valid",
                    "nonfoot_contact",
                    "observation_complete",
                    "returned",
                    "complete",
                    "captured",
                )
                else np.int64
                if key in ("tick", "substep")
                else np.float64
            )
            array = np.asarray(values, dtype=dtype)
            if key in shapes:
                array = array.reshape(count, *shapes[key])
            arrays[key] = array
        result.append(arrays)
    return tuple(result)  # type: ignore[return-value]


def save_raw(
    output: Path,
    helper_registry: dict[str, list[Any]],
    trajectory: np.ndarray,
    body_registry: dict[str, Any],
) -> None:
    helper = helper_arrays(helper_registry)
    if np.asarray(trajectory).size:
        require(
            np.array_equal(
                np.asarray(trajectory, dtype=np.float64),
                helper["selected_joint_target"],
            ),
            "Returned trajectory differs from completed selected helper rows",
        )
    atomic_npz(output / "helper-endpoints.npz", **helper)
    atomic_npz(
        output / "joint-trajectory.npz",
        tick=helper["selected_tick"],
        joint_target=helper["selected_joint_target"],
    )
    applied, native, attempts, controls = body_arrays(body_registry)
    atomic_npz(output / "applied-substeps.npz", **applied)
    atomic_npz(output / "candidate-native.npz", **native)
    atomic_npz(output / "native-attempts.npz", **attempts)
    atomic_npz(output / "candidate-controls.npz", **controls)
    atomic_json(
        output / "candidate-trace.json",
        {
            "schema": "flylab.m3.e156.candidate-trace.v1",
            "samples": body_registry["trace_rows"],
        },
    )


def append_control_attempt(
    registry: dict[str, Any],
    body: Any,
    tick: int,
    target: np.ndarray,
) -> int:
    values = {
        "tick": tick,
        "time_s": float(body.physics_time()),
        "qpos": np.array(body.d.qpos, copy=True),
        "qvel": np.array(body.d.qvel, copy=True),
        "ctrl": np.array(body.d.ctrl, copy=True),
        "position": np.full(3, np.nan),
        "rotation_world": np.full((3, 3), np.nan),
        "joint_target": np.array(target, copy=True),
        "knee_flexion": np.full(6, np.nan),
        "captured": False,
    }
    index = len(registry["controls"]["tick"])
    for key, value in values.items():
        registry["controls"][key].append(value)
    return index


def complete_control_capture(
    registry: dict[str, Any],
    index: int,
    body: Any,
    tick: int,
    captured: dict[str, Any],
) -> None:
    require(
        captured.get("control_tick") == tick
        and captured.get("fault") in (None, False, ""),
        "Captured control invalid",
    )
    body_row = captured.get("body")
    require(isinstance(body_row, dict), "Captured body geometry required")
    ui_position = np.asarray(body_row.get("position"), dtype=np.float64)
    ui_basis = np.asarray(body_row.get("basis"), dtype=np.float64)
    require(
        ui_position.shape == (3,)
        and ui_basis.shape == (3, 3)
        and np.isfinite(ui_position).all()
        and np.isfinite(ui_basis).all(),
        "Finite captured position and basis required",
    )
    registry["controls"]["position"][index] = UI_FROM_PHYSICS.T @ ui_position
    registry["controls"]["rotation_world"][index] = UI_FROM_PHYSICS.T @ ui_basis
    registry["controls"]["knee_flexion"][index] = knee_angles_from_legs(
        body_row.get("legs")
    )
    registry["controls"]["captured"][index] = True
    registry["trace_rows"].append(copy.deepcopy(captured))


def _shape2(value: np.ndarray) -> np.ndarray:
    result = np.full(2, -1, dtype=np.int64)
    result[: min(2, value.ndim)] = value.shape[:2]
    return result


def _fixed_attempt(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64).reshape(-1)
    result = np.full(int(np.prod(shape)), np.nan, dtype=np.float64)
    count = min(len(source), len(result))
    result[:count] = source[:count]
    return result.reshape(shape)


def execute_intervention(
    *,
    body: Any,
    shadow: Any,
    rows: list[dict[str, Any]],
    native_state: dict[str, np.ndarray],
    trajectory: np.ndarray,
    body_registry: dict[str, Any],
    ledger: dict[str, Any],
    journal_path: Path,
    contact_diagnostics: Callable[..., dict[str, Any]],
    capture_control: Callable[[Any], dict[str, Any]],
    replay_command: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    implicit_request: dict[str, Any] | None,
    deadline_check: Callable[[], None] | None = None,
) -> None:
    original_step = body._step_physics
    state_type = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, state_type))
    joint_mask = np.zeros(int(body.m.nu), dtype=bool)
    joint_mask[np.asarray(body.act_ids, dtype=np.int64)] = True
    active_tick = -1
    active_offset = -1
    substep = 0
    expected_nonjoint: np.ndarray | None = None
    journal = Path(journal_path).open("a", encoding="utf-8")  # noqa: SIM115 - closed while preserving the primary error below

    def write_transition(value: dict[str, Any], *, flush: bool) -> None:
        journal.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        if flush:
            journal.flush()

    def observed_step() -> None:
        nonlocal substep
        require(
            0 <= active_offset < 200 and 0 <= substep < SUBSTEPS,
            "Unexpected scientific physics call",
        )
        body.d.ctrl[body.act_ids] = trajectory[active_offset, substep]
        if expected_nonjoint is not None:
            require(
                np.array_equal(body.d.ctrl[~joint_mask], expected_nonjoint),
                "Non-joint control changed during substeps",
            )
        applied = body_registry["applied"]
        for key, value in (
            ("tick", active_tick),
            ("substep", substep),
            ("time_before_s", float(body.physics_time())),
            ("ctrl", np.array(body.d.ctrl, copy=True)),
            ("integration_complete", False),
            ("observation_complete", False),
        ):
            applied[key].append(value)
        ledger["scientific_steps_requested"] += 1
        write_transition(
            {
                "tick": active_tick,
                "substep": substep,
                "time_before_s": float(body.physics_time()),
                "ctrl": np.asarray(body.d.ctrl, dtype=np.float64).tolist(),
                "integration_complete": False,
            },
            flush=True,
        )
        expected_before = (
            ANCHOR_TICK * CONTROL_DT
            + (ledger["scientific_steps_requested"] - 1) * PHYSICS_DT
        )
        require(
            abs(float(body.physics_time()) - expected_before) <= CLOCK_ATOL,
            "Scientific substep start clock differs",
        )
        original_step()
        ledger["scientific_steps_completed"] += 1
        applied["integration_complete"][-1] = True
        expected_after = (
            ANCHOR_TICK * CONTROL_DT + ledger["scientific_steps_completed"] * PHYSICS_DT
        )
        require(
            abs(float(body.physics_time()) - expected_after) <= CLOCK_ATOL,
            "Scientific substep completion clock differs",
        )
        write_transition(
            {
                "tick": active_tick,
                "substep": substep,
                "phase": "integration_complete",
                "time_s": float(body.physics_time()),
            },
            flush=False,
        )
        body.mj.mj_getState(body.m, body.d, state, state_type)
        body.mj.mj_setState(body.m, shadow.d, state, state_type)
        ledger["shadow_forward_requested"] += 1
        body.mj.mj_forward(body.m, shadow.d)
        ledger["shadow_forward_completed"] += 1
        diagnostic = contact_diagnostics(shadow)
        position, rotation, _ = shadow.pose()
        del position
        load = np.asarray(diagnostic["normal_load_bw"], dtype=np.float64)
        slip = np.asarray(
            [
                np.nan if value is None else value
                for value in np.asarray(diagnostic["slip_mm_s"], dtype=object).reshape(
                    -1
                )
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(rotation, dtype=np.float64)
        attempts = body_registry["native_attempts"]
        for key, value in (
            ("tick", active_tick),
            ("substep", substep),
            ("time_s", float(body.physics_time())),
            ("rotation_shape", _shape2(rotation)),
            ("load_shape", _shape2(load)),
            ("slip_shape", _shape2(slip)),
            ("rotation_world", _fixed_attempt(rotation, (3, 3))),
            ("normal_load_bw", _fixed_attempt(load, (6,))),
            ("slip_mm_s", _fixed_attempt(slip, (6,))),
            ("returned", True),
            ("complete", False),
        ):
            attempts[key].append(value)
        require(
            rotation.shape == (3, 3) and load.shape == slip.shape == (6,),
            "Six-leg native contact diagnostic required",
        )
        valid = np.isfinite(slip)
        nonfoot = bool(shadow.nonfoot_contact())
        upright = float(rotation[2, 2])
        native = body_registry["native"]
        for key, value in (
            ("tick", active_tick),
            ("substep", substep),
            ("time_s", float(body.physics_time())),
            ("rotation_world", np.array(rotation, copy=True)),
            ("normal_load_bw", load),
            ("slip_mm_s", slip),
            ("slip_valid", valid),
            ("nonfoot_contact", nonfoot),
            ("upright", upright),
            ("observation_complete", False),
        ):
            native[key].append(value)
        require(
            np.isfinite(load).all()
            and np.all(load >= 0)
            and math.isfinite(upright)
            and np.isfinite(rotation).all(),
            "Finite native observation required",
        )
        ledger["scientific_samples_observed"] += 1
        attempts["complete"][-1] = True
        native["observation_complete"][-1] = True
        applied["observation_complete"][-1] = True
        write_transition(
            {
                "tick": active_tick,
                "substep": substep,
                "phase": "observation_complete",
                "time_s": float(body.physics_time()),
            },
            flush=False,
        )
        substep += 1

    body._step_physics = observed_step
    try:
        for offset, row in enumerate(rows[201:401]):
            active_offset = offset
            active_tick = int(row["control_tick"])
            substep = 0
            endpoint, adhesion, command = replay_command(
                row, implicit_request=implicit_request
            )
            require(
                np.array_equal(endpoint, trajectory[offset, -1]),
                f"Trajectory endpoint differs at {active_tick}",
            )
            body.push_left = command["push_left_s"]
            body.push_force = command["push_force"]
            expected_full = np.asarray(row["ctrl"], dtype=np.float64)
            require(
                expected_full.shape == (int(body.m.nu),),
                "Recorded ctrl width differs from body",
            )
            expected_nonjoint = expected_full[~joint_mask].copy()
            kwargs = copy.deepcopy(command["step_kwargs"])
            kwargs["interpolate"] = False
            ledger["step_joint_targets_requested"] += 1
            body.step_joint_targets(endpoint, adhesion, **kwargs)
            ledger["step_joint_targets_completed"] += 1
            require(
                substep == SUBSTEPS,
                f"Control {active_tick} did not execute 50 physical steps",
            )
            require(
                np.array_equal(body.d.ctrl[body.act_ids], endpoint),
                f"Final joint ctrl differs at {active_tick}",
            )
            require(
                np.array_equal(body.d.ctrl, expected_full),
                f"Final full ctrl differs at {active_tick}",
            )
            require(
                abs(float(body.physics_time()) - active_tick * CONTROL_DT)
                <= CLOCK_ATOL,
                f"Control {active_tick} physical clock differs",
            )
            control_index = append_control_attempt(
                body_registry, body, active_tick, endpoint
            )
            view = SimpleNamespace(
                body=body,
                control_tick=active_tick,
                tendon_inputs=copy.deepcopy(row["tendon_inputs"]),
                rate=np.array(native_state["motor_rate"][active_tick], copy=True),
                fault=None,
                navigation=None,
            )
            ledger["control_capture_requested"] += 1
            captured = capture_control(view)
            complete_control_capture(
                body_registry, control_index, body, active_tick, captured
            )
            ledger["control_capture_completed"] += 1
            require(not body.fault, "Body fault: " + str(body.fault))
            warning = np.asarray(body.d.warning.number)
            require(not np.any(warning), "MuJoCo warning during E156 intervention")
            if deadline_check is not None:
                deadline_check()
    finally:
        active_error = sys.exc_info()[1]
        try:
            journal.flush()
            os.fsync(journal.fileno())
        except Exception as error:
            if active_error is not None:
                active_error.add_note(
                    "Secondary E156 substep journal flush failure: " + repr(error)
                )
            else:
                raise
        finally:
            journal.close()
            try:
                body._step_physics = original_step
            except BaseException as error:
                if active_error is not None:
                    active_error.add_note(
                        "Secondary E156 step restore failure: " + repr(error)
                    )
                else:
                    raise


def knee_angles_from_legs(legs: dict[str, Any]) -> np.ndarray:
    values = []
    for leg in LEGS:
        points = np.asarray(legs.get(leg), dtype=np.float64)
        require(
            points.ndim == 2
            and points.shape[0] >= 4
            and points.shape[1] == 3
            and np.isfinite(points).all(),
            "Complete geometric leg points required: " + leg,
        )
        # WalkingTrace follows LINKS: coxa, trochanterfemur, tibia, tarsus1, ...
        # Match FlyGymBody.knee_kinematics' three physical segment origins.
        a = points[1] - points[2]
        b = points[3] - points[2]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        require(denominator > 0, "Degenerate geometric knee: " + leg)
        values.append(math.acos(float(np.clip((a @ b) / denominator, -1, 1))))
    return np.asarray(values)


def trace_geometry(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(rows) == 201, "Anchor plus 200 captured geometry rows required")
    position = np.empty((201, 3), dtype=np.float64)
    rotation = np.empty((201, 3, 3), dtype=np.float64)
    knee = np.empty((201, 6), dtype=np.float64)
    for index, row in enumerate(rows):
        body = row.get("body")
        require(isinstance(body, dict), "Captured body geometry row required")
        ui_position = np.asarray(body.get("position"), dtype=np.float64)
        ui_basis = np.asarray(body.get("basis"), dtype=np.float64)
        require(
            ui_position.shape == (3,)
            and ui_basis.shape == (3, 3)
            and np.isfinite(ui_position).all()
            and np.isfinite(ui_basis).all(),
            "Finite captured body position/basis required",
        )
        position[index] = UI_FROM_PHYSICS.T @ ui_position
        rotation[index] = UI_FROM_PHYSICS.T @ ui_basis
        knee[index] = knee_angles_from_legs(body.get("legs"))
    return position, rotation, knee


def movement_metrics(
    positions: Any,
    rotations: Any,
    knee_flexion: Any,
    native_load: Any,
    native_slip: Any,
    native_valid: Any,
    native_nonfoot: Any,
    native_upright: Any,
) -> dict[str, Any]:
    position = np.asarray(positions, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.float64)
    knee = np.asarray(knee_flexion, dtype=np.float64)
    load = np.asarray(native_load, dtype=np.float64)
    slip = np.asarray(native_slip, dtype=np.float64)
    valid = np.asarray(native_valid)
    nonfoot = np.asarray(native_nonfoot)
    upright = np.asarray(native_upright, dtype=np.float64)
    require(
        position.shape == (201, 3)
        and rotation.shape == (201, 3, 3)
        and knee.shape == (201, 6),
        "Anchor plus 200 geometric controls required",
    )
    require(
        load.shape == slip.shape == valid.shape == (10000, 6)
        and nonfoot.shape == upright.shape == (10000,),
        "Exact 10000 native samples required",
    )
    require(
        valid.dtype == np.dtype(bool) and nonfoot.dtype == np.dtype(bool),
        "Boolean native masks required",
    )
    require(
        np.isfinite(position).all()
        and np.isfinite(rotation).all()
        and np.isfinite(knee).all()
        and np.isfinite(load).all()
        and np.all(load >= 0)
        and np.isfinite(upright).all()
        and np.array_equal(np.isfinite(slip), valid),
        "Finite aligned movement inputs required",
    )
    require(
        np.allclose(
            np.einsum("nji,njk->nik", rotation, rotation),
            np.eye(3)[None],
            atol=1e-9,
            rtol=0,
        ),
        "Physical rotation basis is not orthogonal",
    )
    # Use the measured physical forward basis for signed displacement and
    # continuity.  Horizontal projection belongs only to net/path geometry.
    heading = rotation[:-1, :, 0]
    delta = np.diff(position, axis=0)
    signed = np.sum(delta * heading, axis=1)
    horizontal = delta[:, :2]
    path = float(np.linalg.norm(horizontal, axis=1).sum())
    net = float(np.linalg.norm(position[-1, :2] - position[0, :2]))
    supported_load = load >= SUPPORT_BW
    require(
        np.all(valid[supported_load]),
        "Every load-bearing native sample requires a finite slip value",
    )
    support = supported_load
    denominator = float(load[support].sum())
    require(
        denominator > 0,
        "At least one load-bearing native sample is required",
    )
    weighted_slip = float(np.sum(load[support] * slip[support]) / denominator)
    raw_speed = np.sum(
        np.gradient(position, CONTROL_DT, axis=0) * rotation[:, :, 0], axis=1
    )
    speed = gaussian_filter1d(raw_speed, sigma=0.02 / CONTROL_DT, mode="nearest")
    moving = speed[1:] > 0.5
    interruptions = int(np.sum((~moving)[1:] & moving[:-1])) + int(not moving[0])
    longest = 0
    current = 0
    for value in moving:
        current = current + 1 if value else 0
        longest = max(longest, current)
    widths = np.percentile(knee, 95, axis=0) - np.percentile(knee, 5, axis=0)
    return {
        "load_weighted_slip_mm_s": weighted_slip,
        "supported_leg_samples": int(support.sum()),
        "signed_forward_mm": float(signed.sum()),
        "horizontal_net_mm": net,
        "horizontal_path_mm": path,
        "net_path_ratio": net / path if path > 0 else None,
        "knee_width_rad": {leg: float(widths[index]) for index, leg in enumerate(LEGS)},
        "knee_width_deg": {
            leg: float(np.degrees(widths[index])) for index, leg in enumerate(LEGS)
        },
        "continuity": {
            "forward_fraction": float(moving.mean()),
            "interruptions": interruptions,
            "longest_forward_bout_s": longest * CONTROL_DT,
            "gaussian_sigma_s": 0.02,
            "threshold_mm_s": 0.5,
            "speed_ticks": list(range(ANCHOR_TICK, FINAL_TICK + 1)),
            "speed_mm_s": speed.tolist(),
            "summary_ticks": [ANCHOR_TICK + 1, FINAL_TICK],
        },
        "nonfoot_samples": int(nonfoot.sum()),
        "minimum_upright": float(upright.min()),
    }


def compare_effect(
    candidate: dict[str, Any], baseline: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any]:
    candidate_slip = candidate["load_weighted_slip_mm_s"]
    baseline_slip = baseline["load_weighted_slip_mm_s"]
    slip_ratio = None
    if candidate_slip is not None and baseline_slip is not None:
        if baseline_slip > 0:
            slip_ratio = candidate_slip / baseline_slip
        else:
            slip_ratio = 0.0 if candidate_slip == 0 else math.inf
    forward_ratio = (
        candidate["signed_forward_mm"] / baseline["signed_forward_mm"]
        if baseline["signed_forward_mm"] != 0
        else None
    )
    candidate_ratio = candidate["net_path_ratio"]
    baseline_ratio = baseline["net_path_ratio"]
    knee_ratios = {
        leg: candidate["knee_width_rad"][leg] / baseline["knee_width_rad"][leg]
        if baseline["knee_width_rad"][leg] > 0
        else None
        for leg in LEGS
    }
    gates = {
        "slip": candidate_slip is not None
        and baseline_slip is not None
        and candidate_slip <= decision["max_slip_ratio"] * baseline_slip,
        "forward": forward_ratio is not None
        and candidate["signed_forward_mm"] > 0
        and forward_ratio >= decision["min_forward_ratio"],
        "net_path": candidate_ratio is not None
        and baseline_ratio is not None
        and candidate_ratio >= baseline_ratio - decision["max_net_path_ratio_drop"],
        "nonfoot": candidate["nonfoot_samples"] <= decision["nonfoot_allowed"],
        "upright": candidate["minimum_upright"] >= decision["min_upright"],
        "knees": all(
            value is not None and value >= decision["min_knee_width_ratio"]
            for value in knee_ratios.values()
        ),
    }
    return {
        "slip_ratio": slip_ratio,
        "forward_ratio": forward_ratio,
        "net_path_ratio_drop": baseline_ratio - candidate_ratio
        if baseline_ratio is not None and candidate_ratio is not None
        else None,
        "knee_width_ratio": knee_ratios,
        "gates": gates,
        "advance_to_closed_loop": all(gates.values()),
    }


def output_manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and item.name != "output-manifest.json"
    ):
        files[path.relative_to(output).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "flylab.m3.e156.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def _save_failure(
    output: Path,
    helper_registry: dict[str, list[Any]],
    trajectory: np.ndarray,
    body_registry: dict[str, Any],
    ledger: dict[str, Any],
    result: dict[str, Any],
    started: float,
    limit_s: float,
    primary: BaseException,
    secondary: list[str],
) -> None:
    operations: list[tuple[str, Callable[[], None]]] = [
        ("raw", lambda: save_raw(output, helper_registry, trajectory, body_registry)),
        ("ledger", lambda: atomic_json(output / "ledger.json", ledger)),
        (
            "timing",
            lambda: atomic_json(
                output / "timing.json",
                {"wall_s": time.perf_counter() - started, "internal_limit_s": limit_s},
            ),
        ),
        (
            "error",
            lambda: atomic_json(
                output / "error.json",
                {
                    "primary": result.get("error"),
                    "traceback": "".join(traceback.format_exception(primary)),
                },
            ),
        ),
    ]
    for name, operation in operations:
        try:
            operation()
        except Exception as error:  # noqa: BLE001 - secondary publication error
            secondary.append(name + ": " + repr(error))
    result["resource_limit_met"] = (time.perf_counter() - started) <= limit_s
    result["secondary_errors"] = list(secondary)
    try:
        atomic_json(output / "result.json", result)
    except Exception as error:  # noqa: BLE001 - secondary publication error
        secondary.append("result: " + repr(error))
    try:
        manifest = output_manifest(output)
        atomic_json(output / "output-manifest.json", manifest)
    except Exception as error:  # noqa: BLE001 - secondary publication error
        secondary.append("manifest: " + repr(error))
    for note in secondary:
        primary.add_note("Secondary E156 cleanup/publication failure: " + note)


def run(
    freeze_path: Path,
    freeze_sha: str,
    source_review: Path,
    source_review_sha: str,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    freeze = load_freeze(freeze_path, freeze_sha)
    validate_source_gate(source_review, source_review_sha)
    verify_hash_map(freeze["input_sha256"])
    verify_hash_map(freeze["reused_source_sha256"])
    verify_hash_map(freeze["preservation_sha256"])
    require(not output.exists(), "E156 output must be new")
    require(
        shutil.disk_usage(output.parent).free
        >= freeze["resource_limits"]["free_bytes"],
        "E156 free-space reserve unavailable",
    )
    pools = threadpool_info()
    require(
        all(item.get("num_threads") == 1 for item in pools),
        "Actual CPU threadpool exceeds one thread",
    )
    limit_s = float(freeze["resource_limits"]["internal_s"])
    deadline(started, limit_s, "preflight")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA,
        "accepted": False,
        "advance_to_closed_loop": False,
        "publication_complete": False,
        "resource_limit_met": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "interpretation_limit": "One-second body-only replay of saved BANC output; no neural feedback, model fitting, biological gait adoption, or F1-F8 completion.",
    }
    atomic_json(output / "result.json", result)
    append_jsonl(
        output / "progress.jsonl",
        {"stage": "preflight_complete", "wall_s": time.perf_counter() - started},
    )
    ledger = {
        "original_processes": 1,
        "automatic_retries": 0,
        "CPU_threads": 1,
        "GPU": 0,
        "network_models": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "body_models": 0,
        "settling_physics_steps": 0,
        "settling_clock_derived_steps": 0,
        "shadow_mjdata": 0,
        "helper_intervals_requested": 0,
        "helper_intervals_returned": 0,
        "helper_intervals_completed": 0,
        "step_joint_targets_requested": 0,
        "step_joint_targets_completed": 0,
        "scientific_steps_requested": 0,
        "scientific_steps_completed": 0,
        "scientific_samples_observed": 0,
        "shadow_forward_requested": 0,
        "shadow_forward_completed": 0,
        "control_capture_requested": 0,
        "control_capture_completed": 0,
        "checkpoint_reads": 0,
        "trace_reads": 0,
        "recruitment_state_reads": 0,
        "parameter_reads": 0,
        "baseline_native_reads": 0,
        "baseline_controls_reads": 0,
        "baseline_context_reads": 0,
        "thread_pools": pools,
    }
    helper_registry = new_helper_registry()
    trajectory = np.empty((0, 50, 42), dtype=np.float64)
    body_registry = new_body_registry(0, 0, 0)
    body = None
    primary: BaseException | None = None
    secondary: list[str] = []
    try:
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import capture_control, contact_diagnostics
        from tools import analyze_c_banc_m3_signed_contact as e154
        from tools import analyze_c_banc_observation as observation
        from tools.banc_m3_substep_recruitment import CausalSubstepRecruitment

        trace = json.loads(safe_path(freeze["trace"]).read_text(encoding="utf-8"))
        ledger["trace_reads"] = 1
        native_state = load_npz(safe_path(freeze["recruitment_state"]))
        ledger["recruitment_state_reads"] = 1
        parameters = load_npz(safe_path(freeze["parameter_bundle"]))
        ledger["parameter_reads"] = 1
        rows = validate_saved_inputs(
            trace, native_state, parameters, freeze["body_hash"]
        )
        baseline_native = load_npz(safe_path(freeze["baseline_native"]))
        ledger["baseline_native_reads"] = 1
        baseline_controls = load_npz(safe_path(freeze["baseline_controls"]))
        ledger["baseline_controls_reads"] = 1
        baseline_context = json.loads(
            safe_path(freeze["baseline_context"]).read_text(encoding="utf-8")
        )
        ledger["baseline_context_reads"] = 1
        validate_baseline(
            baseline_native, baseline_controls, baseline_context, freeze["body_hash"]
        )
        saved = StateStore.load(safe_path(freeze["checkpoint"]))
        ledger["checkpoint_reads"] = 1
        require(
            saved.get("body", {}).get("modelHash") == freeze["body_hash"],
            "Checkpoint body identity differs",
        )
        require(
            e154.validate_body_parameters(saved)
            == baseline_context.get("body_parameters"),
            "Checkpoint body parameters differ from E154 baseline",
        )
        implicit_request = e154.implicit_request_for_trace(trace, saved)
        deadline(started, limit_s, "before body construction")
        body = e154.construct_body_safely(saved, observation, ledger)
        ledger["body_models"] = 1
        require(
            getattr(body, "model_hash", freeze["body_hash"]) == freeze["body_hash"],
            "Constructed body identity differs",
        )
        actual_feet_map, _ = body._force_maps["feet"]
        require(
            np.array_equal(
                np.asarray(actual_feet_map, dtype=np.int64),
                np.asarray(baseline_context["feet_map"], dtype=np.int64),
            )
            and sorted(int(value) for value in body.support_geom_ids)
            == list(baseline_context["support_geom_ids"])
            and float(body.weight0) == float(baseline_context["weight0"]),
            "Constructed body contact provenance differs from E154 baseline",
        )
        require(
            float(body.m.opt.timestep) == PHYSICS_DT, "Actual physics timestep differs"
        )
        ledger["settling_clock_derived_steps"] = e154.validate_settling_count(
            float(body.t0), PHYSICS_DT, 2000
        )
        require(
            ledger["settling_physics_steps"] == 2000,
            "Actual neutral settling count differs",
        )
        adapter = make_helper_adapter(saved, parameters, body.neutral)
        validate_checkpoint_anchor(saved, native_state, adapter)
        helper = CausalSubstepRecruitment(adapter)
        trajectory = reconstruct_endpoints(
            helper,
            adapter,
            rows,
            native_state,
            helper_registry,
            ledger,
            on_interval=lambda _tick: deadline(
                started, limit_s, "saved recruitment reconstruction"
            ),
        )
        save_raw(output, helper_registry, trajectory, body_registry)
        append_jsonl(
            output / "progress.jsonl",
            {
                "stage": "helper_complete",
                "wall_s": time.perf_counter() - started,
                "intervals": 2000,
            },
        )
        for key in ("qpos", "qvel", "ctrl"):
            require(
                np.array_equal(
                    np.asarray(getattr(body.d, key)), np.asarray(rows[200][key])
                )
                and np.array_equal(
                    np.asarray(getattr(body.d, key)), baseline_controls[key][0]
                ),
                "Checkpoint/trace/baseline anchor differs: " + key,
            )
        require(
            abs(float(body.physics_time()) - 1.0) <= CLOCK_ATOL,
            "Checkpoint physics clock differs",
        )
        actuator_ids = np.asarray(body.act_ids, dtype=np.int64)
        require(
            actuator_ids.shape == (42,)
            and len(np.unique(actuator_ids)) == 42
            and np.all((actuator_ids >= 0) & (actuator_ids < int(body.m.nu))),
            "Actual 42-joint actuator mapping is invalid",
        )
        atomic_json(
            output / "context.json",
            {
                "schema": "flylab.m3.e156.body-context.v1",
                "body_hash": body.model_hash,
                "nq": int(body.m.nq),
                "nv": int(body.m.nv),
                "nu": int(body.m.nu),
                "joint_actuator_ids": actuator_ids.tolist(),
                "nonjoint_actuator_ids": np.setdiff1d(
                    np.arange(int(body.m.nu)), actuator_ids
                ).tolist(),
                "neutral_joint_targets": np.asarray(body.neutral).tolist(),
                "physics_dt": float(body.m.opt.timestep),
                "anchor_time_s": float(body.physics_time()),
                "anchor_control_tick": ANCHOR_TICK,
                "final_control_tick": FINAL_TICK,
                "ui_from_physics": UI_FROM_PHYSICS.tolist(),
            },
        )
        body_registry = new_body_registry(
            int(body.m.nq), int(body.m.nv), int(body.m.nu)
        )
        anchor_index = append_control_attempt(
            body_registry,
            body,
            200,
            np.asarray(rows[200]["joint_targets_rad"], dtype=np.float64),
        )
        anchor_capture = capture_control(
            SimpleNamespace(
                body=body,
                control_tick=200,
                tendon_inputs=copy.deepcopy(rows[200]["tendon_inputs"]),
                rate=np.array(native_state["motor_rate"][200], copy=True),
                fault=None,
                navigation=None,
            )
        )
        complete_control_capture(body_registry, anchor_index, body, 200, anchor_capture)
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        ledger["shadow_mjdata"] = 1
        execute_intervention(
            body=body,
            shadow=shadow,
            rows=rows,
            native_state=native_state,
            trajectory=trajectory,
            body_registry=body_registry,
            ledger=ledger,
            journal_path=output / "applied-substeps.jsonl",
            contact_diagnostics=contact_diagnostics,
            capture_control=capture_control,
            replay_command=e154.replay_command,
            implicit_request=implicit_request,
            deadline_check=lambda: deadline(started, limit_s, "scientific body replay"),
        )
        require(
            ledger["scientific_steps_requested"]
            == ledger["scientific_steps_completed"]
            == ledger["scientific_samples_observed"]
            == 10000
            and ledger["shadow_forward_requested"]
            == ledger["shadow_forward_completed"]
            == 10000
            and ledger["step_joint_targets_requested"]
            == ledger["step_joint_targets_completed"]
            == 200
            and ledger["control_capture_completed"] == 200,
            "Exact E156 scientific work counts required",
        )
        StateStore.save(output / "final-actor", body.snapshot())
        save_raw(output, helper_registry, trajectory, body_registry)
        _, candidate_native, candidate_attempts, candidate_controls = body_arrays(
            body_registry
        )
        require(
            len(candidate_attempts["tick"]) == 10000
            and np.all(candidate_attempts["complete"])
            and np.all(candidate_native["observation_complete"]),
            "Exact complete native observation attempts required",
        )
        expected_native_time = (
            ANCHOR_TICK * CONTROL_DT
            + np.arange(1, 10001, dtype=np.float64) * PHYSICS_DT
        )
        require(
            np.allclose(
                candidate_native["time_s"],
                expected_native_time,
                atol=CLOCK_ATOL,
                rtol=0,
            )
            and np.allclose(
                candidate_attempts["time_s"],
                expected_native_time,
                atol=CLOCK_ATOL,
                rtol=0,
            )
            and np.array_equal(
                candidate_native["tick"], np.repeat(np.arange(201, 401), 50)
            ),
            "Candidate native clock/tick alignment differs",
        )
        baseline_position, baseline_rotation, baseline_knee = trace_geometry(
            rows[200:401]
        )
        candidate_position, candidate_rotation, candidate_knee = trace_geometry(
            body_registry["trace_rows"]
        )
        require(
            np.array_equal(candidate_controls["position"], candidate_position)
            and np.array_equal(candidate_controls["rotation_world"], candidate_rotation)
            and np.array_equal(candidate_controls["knee_flexion"], candidate_knee),
            "Candidate trace and compact geometry differ",
        )
        require(
            np.array_equal(candidate_controls["captured"], np.ones(201, dtype=bool)),
            "Anchor plus 200 complete candidate captures required",
        )
        baseline_metrics = movement_metrics(
            baseline_position,
            baseline_rotation,
            baseline_knee,
            baseline_native["normal_load_bw"],
            baseline_native["slip_mm_s"],
            baseline_native["slip_valid"],
            baseline_native["nonfoot_contact"],
            baseline_native["upright"],
        )
        candidate_metrics = movement_metrics(
            candidate_position,
            candidate_rotation,
            candidate_knee,
            candidate_native["normal_load_bw"],
            candidate_native["slip_mm_s"],
            candidate_native["slip_valid"],
            candidate_native["nonfoot_contact"],
            candidate_native["upright"],
        )
        comparison = compare_effect(
            candidate_metrics, baseline_metrics, freeze["decision"]
        )
        atomic_json(
            output / "summary.json",
            {
                "schema": "flylab.m3.e156.causal-substep-body-summary.v1",
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "comparison": comparison,
                "closed_loop": False,
            },
        )
        body.close()
        body = None
        verify_hash_map(freeze["input_sha256"])
        verify_hash_map(freeze["reused_source_sha256"])
        verify_hash_map(freeze["preservation_sha256"])
        deadline(started, limit_s, "before final publication")
        atomic_json(output / "ledger.json", ledger)
        atomic_json(
            output / "timing.json",
            {
                "wall_s_before_final_publication": time.perf_counter() - started,
                "internal_limit_s": limit_s,
                "external_limit_s": freeze["resource_limits"]["external_s"],
                "supervisor_limit_s": freeze["resource_limits"]["supervisor_s"],
                "prior_actual_original_s": freeze["prior_actual_original_s"],
                "prior_conservative_s": freeze["prior_conservative_s"],
                "previous_allocation_s": freeze["previous_allocation_s"],
                "prospective_cumulative_s": freeze["prospective_cumulative_s"],
            },
        )
        result.update(
            {
                "accepted": True,
                "advance_to_closed_loop": comparison["advance_to_closed_loop"],
                "publication_complete": True,
                "resource_limit_met": True,
                "helper_intervals": 2000,
                "scientific_physics_steps": 10000,
                "shadow_forward_calls": 10000,
                "effect_gates": comparison["gates"],
            }
        )
        append_jsonl(
            output / "progress.jsonl",
            {"stage": "complete", "wall_s": time.perf_counter() - started},
        )
        atomic_json(output / "result.json", result)
        manifest = output_manifest(output)
        require(
            manifest["total_bytes"] <= freeze["resource_limits"]["raw_bytes"],
            "E156 output exceeded raw byte limit",
        )
        atomic_json(output / "output-manifest.json", manifest)
        deadline(started, limit_s, "after final manifest publication")
        return result
    except Exception as error:  # noqa: BLE001 - preserve exact primary failure
        primary = error
        result.update(
            {
                "accepted": False,
                "advance_to_closed_loop": False,
                "publication_complete": False,
                "resource_limit_met": (time.perf_counter() - started) <= limit_s,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
    finally:
        if body is not None:
            try:
                body.close()
            except Exception as error:  # noqa: BLE001 - secondary cleanup error
                secondary.append("body_close: " + repr(error))
        if primary is not None:
            _save_failure(
                output,
                helper_registry,
                trajectory,
                body_registry,
                ledger,
                result,
                started,
                limit_s,
                primary,
                secondary,
            )
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    raise RuntimeError("Unreachable E156 state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.freeze,
                arguments.freeze_sha256,
                arguments.source_review,
                arguments.source_review_sha256,
                arguments.output,
            ),
            indent=2,
        )
    )
