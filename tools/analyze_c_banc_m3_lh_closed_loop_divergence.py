"""E252: saved E251 endpoint localization, without advancing a model."""

# Broad catches preserve the first analysis or publication failure and restore errors.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "12e53d29bd0bdb417b04840e16847513673a5155b974688048da2e9bbe95f7ae"
SCHEMA = "flylab.m3.e252.lh-closed-loop-divergence"
OWNED = (
    "tools/analyze_c_banc_m3_lh_closed_loop_divergence.py",
    "tests/test_c_banc_lh_closed_loop_divergence.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DOFS = ("coxa_yaw", "coxa_pitch", "coxa_roll", "femur_pitch", "femur_roll",
        "tibia_pitch", "tarsus_pitch")
LIMITS = {"internal_s": 30, "finalization_reserve_s": 5, "external_s": 40,
          "supervisor_s": 45, "cpu_threads": 1, "gpu_count": 0,
          "original_attempts": 1, "retries": 0, "body_constructors": 0,
          "neural_constructors": 0, "neural_ticks": 0, "physics_steps": 0,
          "fit_calls": 0, "solver_calls": 0, "resident_bytes": 2147483648,
          "output_bytes": 67108864, "minimum_free_bytes": 536870912}
TOL = {"sensory": 1e-6, "neural": 1e-5, "bank": 1e-5,
       "activation": 1e-6, "command": 1e-7, "actual_joint": 1e-7,
       "body_position": 1e-7, "native": 1e-7}
ARM_KEYS = {
    "sensory": ("control_tick_before", "complete", "source_ids", "old_filtered",
                "new_filtered", "last_features", "old_velocity_lowpass",
                "new_velocity_lowpass", "returned_source_drive"),
    "activity": ("observed_ids", "source_ids", "bridge_ids",
                 "reachable_motor_ids", "rate", "drive", "adaptation"),
    "control": ("control_tick", "network_tick", "physics_time_s",
                "motor_rate32", "activation", "command", "tendon"),
    "recruitment": ("control_tick", "network_tick", "physics_time_s", "complete",
                    "bank", "bank_available", "activation", "activation_available",
                    "original_activation", "original_activation_available",
                    "motor_rate", "motor_rate_available"),
    "native": ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid",
               "nonfoot_contact", "upright"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def save_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def pinned(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash path required")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Unsafe pinned path")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("Pinned file missing or outside repository")
    return resolved


def gate(freeze_path: str, freeze_sha: str, review_path: str,
         review_sha: str) -> tuple[dict, dict]:
    if freeze_sha != FREEZE_SHA or sha(Path(freeze_path)) != FREEZE_SHA:
        raise ValueError("E252 freeze hash mismatch")
    if sha(Path(review_path)) != review_sha:
        raise ValueError("E252 review hash mismatch")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != SCHEMA + ".freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS
            or freeze.get("windows") != {"startup": [1, 100],
                                          "steady": [100, 300], "full": [0, 301]}
            or freeze.get("legs") != list(LEGS)
            or freeze.get("stage_tolerances") != TOL
            or freeze.get("controls") != 300
            or freeze.get("native_steps") != 15000
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("control_dt_s") != .005
            or freeze.get("native_dt_s") != .0001
            or freeze.get("thresholds") != {
                "startup_q_max": .04, "startup_q_rms": .018,
                "startup_tendon_rms": .009, "startup_tendon_max": .025,
                "lh_positive_sse_fraction": .5}):
        raise ValueError("E252 frozen contract differs")
    if (review.get("schema") != SCHEMA + ".review.v1"
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != {name: sha(pinned(name)) for name in OWNED}):
        raise ValueError("E252 parent source review invalid")
    if sha(pinned(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E252 design hash differs")
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Hash map required")
        for name, expected in mapping.items():
            if sha(pinned(name)) != expected:
                raise ValueError("Pinned bytes differ: " + name)
    if freeze["packages"] != {name: importlib.metadata.version(name)
                              for name in freeze["packages"]}:
        raise ValueError("Pinned packages differ")
    if (set(freeze["paths"]) != {"calibration", "parameters", "muscles",
                                  "startup", "edge_patch", "selection", "e251_result",
                                  "e251_manifest", "e251_output_review",
                                  "e251_final_review", "e251_supervisor"}
            or set(freeze["arms"]) != {"baseline", "candidate"}):
        raise ValueError("E252 input roles differ")
    return freeze, review


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if not set(keys).issubset(saved.files):
            raise ValueError("Required NPZ keys missing: " + str(path))
        return {key: saved[key].copy() for key in keys}


def normalize_selection(value: dict) -> dict:
    """Extract the frozen E248 selected route without losing wrapper provenance."""
    if not isinstance(value, dict):
        raise TypeError("E248 selection JSON object required")
    if "selected" in value:
        if not isinstance(value.get("rule"), str) or not value["rule"]:
            raise ValueError("E248 selection rule missing")
        value = value["selected"]
    if (not isinstance(value, dict) or value.get("port") != 85
            or value.get("kind") != "claw" or value.get("reachable_count") != 55
            or not all(key in value for key in
                       ("source_rows", "bridge_rows", "reachable_motor_rows"))):
        raise ValueError("E248 selected LH claw route differs")
    return value


def load_inputs(freeze: dict, progress=None) -> dict:
    inputs: dict = {}
    for role, keys in {
        "calibration": ("motor_ids", "moment", "rest", "pool", "membership", "ltm"),
        "parameters": ("motor_ids", "membership"),
        "startup": ("target_command", "target_tendon"),
        "edge_patch": ("edge_ids", "pre_ids", "post_ids", "source_values",
                       "target_values"),
    }.items():
        inputs[role] = load_npz(pinned(freeze["paths"][role]), keys)
        if progress:
            progress(role)
    for role in ("muscles", "selection", "e251_result", "e251_manifest",
                 "e251_output_review", "e251_final_review", "e251_supervisor"):
        inputs[role] = json.loads(pinned(freeze["paths"][role]).read_text(encoding="utf-8"))
        if role == "selection":
            inputs[role] = normalize_selection(inputs[role])
        if progress:
            progress(role)
    for arm, folder in freeze["arms"].items():
        root = (ROOT / folder).resolve()
        if not root.is_relative_to(ROOT):
            raise ValueError("Unsafe E251 arm path")
        data = {}
        for key, filename in (("sensory", "sensory-input.npz"),
                              ("activity", "activity.npz"),
                              ("control", "control-raw.npz"),
                              ("recruitment", "native-recruitment-state.npz"),
                              ("native", "native-contact.npz")):
            data[key] = load_npz(pinned(f"{folder}/{filename}"), ARM_KEYS[key])
            if progress:
                progress(arm + "/" + key)
        for key, filename in (("sensory_metadata", "sensory-metadata.json"),
                              ("trace", "trace.json"), ("arm_result", "arm-result.json")):
            data[key] = json.loads(pinned(f"{folder}/{filename}").read_text(encoding="utf-8"))
            if progress:
                progress(arm + "/" + key)
        data["output_mask"] = np.load(pinned(f"{folder}/observed-start/array_00002.npy"),
                                      allow_pickle=False)
        if progress:
            progress(arm + "/output_mask")
        inputs[arm] = data
    return inputs


def require_array(value, shape, dtype=None, *, finite=True) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or (dtype is not None and array.dtype != dtype):
        raise ValueError(f"Array shape/dtype differs: {array.shape}, {array.dtype}, expected {shape}/{dtype}")
    if finite and array.dtype.kind in "fiu" and not np.isfinite(array).all():
        raise ValueError("Nonfinite required array")
    return array


def _clock(value, expected, *, atol=0.0) -> None:
    if not np.allclose(value, expected, rtol=0, atol=atol):
        raise ValueError("Saved clock alignment differs")


def validate(inputs: dict, *, rows=301, native_rows=15000,
             motors=391, muscles=96, neurons=158706) -> dict:
    if inputs["e251_result"].get("validity") != "valid_short_native_body_diagnostic":
        raise ValueError("Valid E251 receipt required")
    c, p, target, patch = (inputs[key] for key in
                           ("calibration", "parameters", "startup", "edge_patch"))
    ids = require_array(c["motor_ids"], (motors,), np.dtype("int64"))
    if (not np.array_equal(ids, p["motor_ids"])
            or len(np.unique(ids)) != motors or not np.array_equal(ids, np.sort(ids))):
        raise ValueError("E150 motor roster differs")
    member = require_array(c["membership"], (motors,), np.dtype("int64"))
    if (not np.array_equal(member, p["membership"])
            or np.any((member < 0) | (member >= muscles))):
        raise ValueError("E150 calibration/parameter membership differs")
    pool = require_array(c["pool"], (muscles, motors), np.dtype("float64"))
    require_array(c["moment"], (42, muscles), np.dtype("float64"))
    require_array(c["rest"], (42,), np.dtype("float64"))
    require_array(c["ltm"], (6,), np.dtype("int64"))
    if (np.any(pool < 0) or not np.allclose(pool.sum(axis=1), 1, rtol=0, atol=1e-12)
            or not np.array_equal(np.argmax(pool, axis=0), member)
            or not np.array_equal((pool > 0).sum(axis=0), np.ones(motors))):
        raise ValueError("E150 anatomical pool differs")
    groups = inputs["muscles"]["muscles"]
    if len(groups) != muscles:
        raise ValueError("Exactly 96 original muscles required")
    for m, group in enumerate(groups):
        if (group["leg"] not in LEGS or not group["muscle"]
                or group["columns"] != np.flatnonzero(member == m).tolist()):
            raise ValueError("Original named muscle columns differ")
    require_array(target["target_command"], (99, 42), np.dtype("float64"))
    require_array(target["target_tendon"], (99, 6), np.dtype("float64"))
    for key, dtype in (("edge_ids", "int64"), ("pre_ids", "int64"),
                       ("post_ids", "int64"), ("source_values", "float32"),
                       ("target_values", "float32")):
        require_array(patch[key], (25,), np.dtype(dtype))
    if (len(np.unique(patch["edge_ids"])) != 25
            or np.any((patch["pre_ids"] < 0) | (patch["pre_ids"] >= neurons))
            or np.any((patch["post_ids"] < 0) | (patch["post_ids"] >= neurons))
            or np.any(patch["source_values"] == 0)
            or np.any(patch["target_values"] == 0)
            or not np.array_equal(np.sign(patch["source_values"]),
                                  np.sign(patch["target_values"]))):
        raise ValueError("E250 25-edge identities/weights differ")
    selection = normalize_selection(inputs["selection"])
    source = np.asarray(selection["source_rows"], np.int64)
    bridge = np.asarray(selection["bridge_rows"], np.int64)
    reachable = np.asarray(selection["reachable_motor_rows"], np.int64)
    if (source.shape != (23,) or bridge.shape != (4,) or reachable.shape != (55,)
            or len(np.unique(np.r_[source, bridge, reachable])) != 82
            or not np.isin(patch["pre_ids"], source).all()
            or not np.isin(patch["post_ids"], bridge).all()):
        raise ValueError("E251 selection/patch rows differ")
    observed = np.r_[source, bridge, reachable]
    joint_columns = None
    for arm in ("baseline", "candidate"):
        data = inputs[arm]
        if (data["arm_result"].get("status") != "complete_healthy_short_diagnostic"
                or data["arm_result"].get("complete") is not True):
            raise ValueError("Complete healthy E251 arm required")
        s, a, r, n = (data[key] for key in ("sensory", "activity", "recruitment", "native"))
        k = data["control"]
        if not np.array_equal(data["output_mask"].shape, (neurons,)):
            raise ValueError("Original neural output mask width differs")
        if (data["output_mask"].dtype != np.float32
                or not np.isfinite(data["output_mask"]).all()
                or not np.isin(data["output_mask"], [0, 1]).all()):
            raise ValueError("Original output mask invalid")
        require_array(s["control_tick_before"], (rows-1,), np.dtype("int64"))
        _clock(s["control_tick_before"], np.arange(rows-1))
        require_array(s["complete"], (rows-1,), np.dtype("bool"))
        if not s["complete"].all() or not np.array_equal(s["source_ids"], source):
            raise ValueError("Incomplete/misaligned sensory rows")
        for name, width, dtype in (("old_filtered", 241, "float64"),
                                   ("new_filtered", 241, "float64"),
                                   ("last_features", 241, "float64"),
                                   ("old_velocity_lowpass", 42, "float64"),
                                   ("new_velocity_lowpass", 42, "float64"),
                                   ("returned_source_drive", 23, "float32")):
            require_array(s[name], (rows-1, width), np.dtype(dtype))
        for name, expected in (("observed_ids", observed), ("source_ids", source),
                               ("bridge_ids", bridge),
                               ("reachable_motor_ids", reachable)):
            if not np.array_equal(a[name], expected):
                raise ValueError("Activity ID order differs: " + name)
        for name in ("rate", "drive", "adaptation"):
            require_array(a[name], (rows, 82), np.dtype("float32"))
        if not np.array_equal(a["drive"][1:, :23], s["returned_source_drive"]):
            raise ValueError("Source drive does not align to next endpoint")
        for name, width, dtype in (("motor_rate32", motors, "float32"),
                                   ("activation", muscles, "float64"),
                                   ("command", 42, "float64"),
                                   ("tendon", 6, "float64")):
            require_array(k[name], (rows, width), np.dtype(dtype))
        for name, width in (("bank", (motors, 3)),
                            ("activation", (muscles,)),
                            ("original_activation", (muscles,)),
                            ("motor_rate", (motors,))):
            require_array(r[name], (rows, *width), np.dtype("float64"))
        for name in ("control_tick", "network_tick"):
            require_array(k[name], (rows,), np.dtype("int64"))
            require_array(r[name], (rows,), np.dtype("int64"))
        _clock(k["control_tick"], np.arange(rows))
        _clock(k["network_tick"], np.arange(rows)*20)
        _clock(r["control_tick"], k["control_tick"])
        _clock(r["network_tick"], k["network_tick"])
        for name in ("physics_time_s",):
            require_array(k[name], (rows,), np.dtype("float64"))
            require_array(r[name], (rows,), np.dtype("float64"))
            _clock(k[name], np.arange(rows)*.005, atol=2e-8)
            _clock(r[name], k[name], atol=2e-8)
        for name in ("complete", "bank_available", "activation_available",
                     "original_activation_available", "motor_rate_available"):
            require_array(r[name], (rows,), np.dtype("bool"))
            if not r[name].all():
                raise ValueError("Incomplete E251 recruitment rows")
        if (not np.allclose(r["motor_rate"], k["motor_rate32"], rtol=0, atol=1e-5)
                or not np.array_equal(r["activation"], k["activation"])):
            raise ValueError("Duplicate E251 recruitment/control values disagree")
        loc = np.searchsorted(ids, reachable)
        if (np.any(loc >= motors) or not np.array_equal(ids[loc], reachable)
                or not np.allclose(a["rate"][:, 27:], k["motor_rate32"][:, loc],
                                   rtol=0, atol=1e-5)):
            raise ValueError("Reachable activity/MN control alignment differs")
        for name, shape, dtype in (("time_s", (native_rows,), "float64"),
                                   ("normal_load_bw", (native_rows, 6), "float64"),
                                   ("slip_mm_s", (native_rows, 6), "float64"),
                                   ("slip_valid", (native_rows, 6), "bool"),
                                   ("nonfoot_contact", (native_rows,), "bool"),
                                   ("upright", (native_rows,), "float64")):
            require_array(n[name], shape, np.dtype(dtype), finite=name != "slip_mm_s")
        _clock(n["time_s"], np.arange(1, native_rows+1)*.0001, atol=2e-8)
        if (not np.isfinite(n["slip_mm_s"][n["slip_valid"]]).all()
                or not np.isnan(n["slip_mm_s"][~n["slip_valid"]]).all()):
            raise ValueError("Native slip availability/finite values differ")
        meta = data["sensory_metadata"]
        columns = np.asarray(meta["joint_columns"], np.int64)
        if (columns.shape != (6, 7) or set(columns.flat) != set(range(42))
                or not np.array_equal(meta["source_rows"], source)):
            raise ValueError("Physical joint columns or source IDs differ")
        if joint_columns is None:
            joint_columns = columns
        elif not np.array_equal(joint_columns, columns):
            raise ValueError("Arm joint mapping differs")
        samples = data["trace"].get("samples")
        if not isinstance(samples, list) or len(samples) != rows:
            raise ValueError("Complete E251 trace rows required")
        for t, row in enumerate(samples):
            if row["control_tick"] != t or abs(row["time_s"] - t*.005) > 2e-8:
                raise ValueError("Trace clock differs")
        q = require_array([row["joint_angles_rad"] for row in samples],
                          (rows, 42), np.dtype("float64"))
        command = require_array([row["joint_targets_rad"] for row in samples],
                                (rows, 42), np.dtype("float64"))
        position = require_array([row["body"]["position"] for row in samples],
                                 (rows, 3), np.dtype("float64"))
        if not np.array_equal(command, k["command"]):
            raise ValueError("Trace/control command differs")
        data["actual_joint"] = q
        data["body_position"] = position
    if not np.array_equal(inputs["baseline"]["output_mask"],
                          inputs["candidate"]["output_mask"]):
        raise ValueError("E251 arm output masks differ")
    if not np.array_equal(inputs["baseline"]["native"]["time_s"],
                          inputs["candidate"]["native"]["time_s"]):
        raise ValueError("E251 arm native sample times differ")
    return {"motor_ids": ids, "membership": member,
            "joint_columns": joint_columns, "source_ids": source,
            "bridge_ids": bridge, "reachable_motor_ids": reachable}


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def stage_stats(delta: np.ndarray, *, start: int, tol: float,
                availability: np.ndarray | None = None) -> dict:
    value = np.asarray(delta, np.float64)
    if availability is not None:
        mask = np.asarray(availability, bool)
        if mask.shape != value.shape:
            raise ValueError("Stage availability shape differs")
        value = np.where(mask, value, 0.)
        valid = mask
    else:
        valid = np.ones(value.shape, bool)
    if not valid.any():
        return {"available": False, "first_exact_control": None,
                "first_above_tolerance_control": None, "rms": None,
                "max_abs": None, "valid_values": 0}
    axes = tuple(range(1, value.ndim))
    exact_rows = np.any((value != 0) & valid, axis=axes) if axes else (value != 0) & valid
    tol_rows = np.any((np.abs(value) > tol) & valid, axis=axes) if axes else (np.abs(value) > tol) & valid
    exact_index = np.flatnonzero(exact_rows)
    tol_index = np.flatnonzero(tol_rows)
    return {"available": True,
            "first_exact_control": int(start+exact_index[0]) if len(exact_index) else None,
            "first_above_tolerance_control": int(start+tol_index[0]) if len(tol_index) else None,
            "rms": float(np.sqrt(np.sum(value*value)/int(valid.sum()))),
            "max_abs": float(np.max(np.abs(value))),
            "valid_values": int(valid.sum())}


def analyze_arrays(inputs: dict, *, raw: dict | None = None,
                   progress=None, sizes=(301, 15000, 391, 96, 158706)) -> tuple[dict, dict]:
    """Pure saved-array analysis; raw is caller-owned for partial preservation."""
    if raw is None:
        raw = {}
    rows, native_rows, motors, muscles, neurons = sizes
    mapping = validate(inputs, rows=rows, native_rows=native_rows,
                       motors=motors, muscles=muscles, neurons=neurons)
    raw["motor_ids"] = mapping["motor_ids"]
    raw["membership"] = mapping["membership"]
    raw["joint_columns"] = mapping["joint_columns"]
    raw["source_ids"] = mapping["source_ids"]
    raw["bridge_ids"] = mapping["bridge_ids"]
    raw["reachable_motor_ids"] = mapping["reachable_motor_ids"]
    report = {"schema": SCHEMA + ".analysis.v1", "stages": {}, "windows": {},
              "first_new_violation": None, "maximum_candidate_error": None,
              "hypothesis_supported": None,
              "claim_scope": "saved endpoint localization, not causal latency or gait improvement"}
    b, c = inputs["baseline"], inputs["candidate"]
    observed_control = b["sensory"]["control_tick_before"].copy()
    endpoint_control = b["control"]["control_tick"].copy()
    native_time = b["native"]["time_s"].copy()
    raw["sensory_observed_control_tick"] = observed_control
    raw["sensory_observed_time_s"] = observed_control.astype(np.float64)*.005
    raw["sensory_applied_endpoint_control_tick"] = observed_control+1
    raw["sensory_applied_endpoint_time_s"] = (observed_control+1).astype(np.float64)*.005
    raw["endpoint_control_tick"] = endpoint_control
    raw["endpoint_time_s"] = b["control"]["physics_time_s"].copy()
    raw["native_sample_index_1_based"] = np.arange(1, native_rows+1, dtype=np.int64)
    raw["native_time_s"] = native_time
    stages = {
        "old_filtered": (b["sensory"]["old_filtered"], c["sensory"]["old_filtered"], 0, "sensory"),
        "new_filtered": (b["sensory"]["new_filtered"], c["sensory"]["new_filtered"], 0, "sensory"),
        "last_features": (b["sensory"]["last_features"], c["sensory"]["last_features"], 0, "sensory"),
        "old_velocity_lowpass": (b["sensory"]["old_velocity_lowpass"], c["sensory"]["old_velocity_lowpass"], 0, "sensory"),
        "source_drive": (b["sensory"]["returned_source_drive"], c["sensory"]["returned_source_drive"], 1, "sensory"),
        "source_rate": (b["activity"]["rate"][:, :23], c["activity"]["rate"][:, :23], 0, "neural"),
        "bridge_rate": (b["activity"]["rate"][:, 23:27], c["activity"]["rate"][:, 23:27], 0, "neural"),
        "reachable_motor_rate": (b["activity"]["rate"][:, 27:], c["activity"]["rate"][:, 27:], 0, "neural"),
        "source_adaptation": (b["activity"]["adaptation"][:, :23], c["activity"]["adaptation"][:, :23], 0, "neural"),
        "bridge_adaptation": (b["activity"]["adaptation"][:, 23:27], c["activity"]["adaptation"][:, 23:27], 0, "neural"),
        "motor_rate391": (b["control"]["motor_rate32"], c["control"]["motor_rate32"], 0, "neural"),
        "bank": (b["recruitment"]["bank"], c["recruitment"]["bank"], 0, "bank"),
        "activation": (b["control"]["activation"], c["control"]["activation"], 0, "activation"),
        "command": (b["control"]["command"], c["control"]["command"], 0, "command"),
        "actual_joint": (b["actual_joint"], c["actual_joint"], 0, "actual_joint"),
        "body_position": (b["body_position"], c["body_position"], 0, "body_position"),
    }
    for name, (left, right, start, category) in stages.items():
        delta = np.asarray(right, np.float64) - np.asarray(left, np.float64)
        raw["difference_" + name] = delta
        report["stages"][name] = stage_stats(delta, start=start, tol=TOL[category])
        if progress:
            progress(name)
    for name in ("normal_load_bw", "upright"):
        delta = c["native"][name] - b["native"][name]
        raw["difference_native_" + name] = delta
        report["stages"]["native_" + name] = stage_stats(
            delta, start=1, tol=TOL["native"])
    slip_available = b["native"]["slip_valid"] & c["native"]["slip_valid"]
    raw["native_slip_baseline_available"] = b["native"]["slip_valid"]
    raw["native_slip_candidate_available"] = c["native"]["slip_valid"]
    slip_delta = np.where(slip_available,
                          c["native"]["slip_mm_s"] - b["native"]["slip_mm_s"], 0.)
    raw["difference_native_slip_mm_s"] = slip_delta
    raw["native_slip_jointly_available"] = slip_available
    report["stages"]["native_slip_mm_s"] = stage_stats(
        slip_delta, start=1, tol=TOL["native"], availability=slip_available)
    report["native_slip_availability"] = {
        "baseline_valid_values": int(b["native"]["slip_valid"].sum()),
        "candidate_valid_values": int(c["native"]["slip_valid"].sum()),
        "joint_valid_values": int(slip_available.sum()),
        "baseline_only_values": int((b["native"]["slip_valid"] & ~c["native"]["slip_valid"]).sum()),
        "candidate_only_values": int((c["native"]["slip_valid"] & ~b["native"]["slip_valid"]).sum())}
    nonfoot_diff = c["native"]["nonfoot_contact"].astype(np.int8) - b["native"]["nonfoot_contact"].astype(np.int8)
    raw["difference_native_nonfoot_contact"] = nonfoot_diff
    report["stages"]["native_nonfoot_contact"] = stage_stats(nonfoot_diff, start=1, tol=0)
    if progress:
        progress("native")

    patch = inputs["edge_patch"]
    source_ids = mapping["source_ids"]
    bridge_ids = mapping["bridge_ids"]
    source_lookup = {int(value): i for i, value in enumerate(source_ids)}
    bridge_lookup = {int(value): i for i, value in enumerate(bridge_ids)}
    source_column = np.asarray([source_lookup[int(value)] for value in patch["pre_ids"]])
    bridge_column = np.asarray([bridge_lookup[int(value)] for value in patch["post_ids"]])
    mask = b["output_mask"][patch["pre_ids"]].astype(np.float64)
    rb = b["activity"]["rate"][:, source_column].astype(np.float64)*mask
    rc = c["activity"]["rate"][:, source_column].astype(np.float64)*mask
    wb = patch["source_values"].astype(np.float64)
    wc = patch["target_values"].astype(np.float64)
    edge_baseline = rb*wb
    edge_candidate = rc*wc
    edge_intervention = rb*(wc-wb)
    edge_feedback = (rc-rb)*wc
    if not np.allclose(edge_candidate-edge_baseline,
                       edge_intervention+edge_feedback, rtol=0, atol=1e-10):
        raise ValueError("25-edge arithmetic identity differs")
    for key, value in (("edge_id", patch["edge_ids"]),
                       ("edge_pre", patch["pre_ids"]), ("edge_post", patch["post_ids"]),
                       ("edge_baseline_current", edge_baseline),
                       ("edge_candidate_current", edge_candidate),
                       ("edge_intervention", edge_intervention),
                       ("edge_feedback", edge_feedback)):
        raw[key] = value
    for name, value in (("baseline", edge_baseline), ("candidate", edge_candidate),
                        ("intervention", edge_intervention), ("feedback", edge_feedback)):
        grouped = np.zeros((rows, 4), np.float64)
        for j in range(4):
            grouped[:, j] = value[:, bridge_column == j].sum(axis=1, dtype=np.float64)
        raw["bridge_edge_" + name] = grouped
    if progress:
        progress("edges")

    calibration = inputs["calibration"]
    moment = calibration["moment"]
    rest = calibration["rest"]
    for arm in ("baseline", "candidate"):
        predicted = rest + .02 * inputs[arm]["control"]["activation"] @ moment.T
        if not np.allclose(predicted[1:], inputs[arm]["control"]["command"][1:],
                           rtol=0, atol=1e-12):
            raise ValueError("Original E150 activation-to-command reconstruction differs")
        raw[arm + "_row0_command_reconstruction_difference"] = (
            inputs[arm]["control"]["command"][0] - predicted[0])
    activation_delta = raw["difference_activation"]
    contribution = .02 * activation_delta[:, None, :] * moment[None, :, :]
    raw["muscle_joint_contribution"] = contribution
    if not np.allclose(contribution.sum(axis=2)[1:], raw["difference_command"][1:],
                       rtol=0, atol=1e-12):
        raise ValueError("96-muscle command contribution sum differs")
    if progress:
        progress("muscle_command")

    target = inputs["startup"]["target_command"]
    be = b["control"]["command"][1:100] - target
    ce = c["control"]["command"][1:100] - target
    dq = ce-be
    signed = ce*ce-be*be
    linear = 2*be*dq
    quadratic = dq*dq
    if not np.allclose(signed, linear+quadratic, rtol=0, atol=1e-12):
        raise ValueError("Startup signed SSE identity differs")
    positive = np.maximum(signed, 0)
    negative = np.minimum(signed, 0)
    violations = (np.abs(be) <= .04) & (np.abs(ce) > .04)
    for key, value in (("startup_baseline_error", be), ("startup_candidate_error", ce),
                       ("startup_signed_sse_change", signed),
                       ("startup_linear_sse_term", linear),
                       ("startup_quadratic_sse_term", quadratic),
                       ("startup_positive_sse_change", positive),
                       ("startup_negative_sse_change", negative),
                       ("startup_new_violation", violations)):
        raw[key] = value
    columns = mapping["joint_columns"]
    inverse_leg = np.empty(42, np.int64)
    inverse_dof = np.empty(42, np.int64)
    for leg in range(6):
        inverse_leg[columns[leg]] = leg
        inverse_dof[columns[leg]] = np.arange(7)
    raw["joint_leg_index"] = inverse_leg
    raw["joint_dof_index"] = inverse_dof
    per_leg = {}
    for leg, name in enumerate(LEGS):
        j = columns[leg]
        per_leg[name] = {"signed_sse_change": float(signed[:, j].sum()),
                         "linear_sse_term": float(linear[:, j].sum()),
                         "quadratic_sse_term": float(quadratic[:, j].sum()),
                         "positive_sse_change": float(positive[:, j].sum()),
                         "negative_sse_change": float(negative[:, j].sum()),
                         "new_violations": int(violations[:, j].sum()),
                         "baseline_q_rms_rad": _rms(be[:, j]),
                         "candidate_q_rms_rad": _rms(ce[:, j]),
                         "baseline_q_max_abs_rad": float(np.max(np.abs(be[:, j]))),
                         "candidate_q_max_abs_rad": float(np.max(np.abs(ce[:, j])))}
    all_violations = np.argwhere(violations)
    if len(all_violations):
        t, j = map(int, all_violations[0])
        report["first_new_violation"] = {"control": t+1, "joint_column": j,
            "leg": LEGS[inverse_leg[j]], "dof": DOFS[inverse_dof[j]],
            "baseline_error_rad": float(be[t, j]),
            "candidate_error_rad": float(ce[t, j])}
    top = int(np.argmax(np.abs(ce)))
    t, j = np.unravel_index(top, ce.shape)
    report["maximum_candidate_error"] = {"control": int(t+1),
        "joint_column": int(j), "leg": LEGS[inverse_leg[j]],
        "dof": DOFS[inverse_dof[j]], "candidate_error_rad": float(ce[t, j]),
        "baseline_error_rad": float(be[t, j])}
    denominator = float(positive.sum())
    fraction = (float(positive[:, columns[2]].sum())/denominator
                if denominator > 0 else None)
    tendon_target = inputs["startup"]["target_tendon"]
    startup = {"baseline_q_rms_rad": _rms(be), "candidate_q_rms_rad": _rms(ce),
               "baseline_q_max_abs_rad": float(np.max(np.abs(be))),
               "candidate_q_max_abs_rad": float(np.max(np.abs(ce))),
               "signed_sse_change": float(signed.sum()),
               "linear_sse_term": float(linear.sum()),
               "quadratic_sse_term": float(quadratic.sum()),
               "positive_sse_change": denominator,
               "negative_sse_change": float(negative.sum()),
               "lh_positive_sse_fraction": fraction,
               "new_violation_count": int(violations.sum()), "legs": per_leg}
    for arm in ("baseline", "candidate"):
        te = inputs[arm]["control"]["tendon"][1:100] - tendon_target
        startup[arm + "_tendon_rms"] = _rms(te)
        startup[arm + "_tendon_max_abs"] = float(np.max(np.abs(te)))
    local = bool(len(all_violations) and np.all(inverse_leg[all_violations[:, 1]] == 2)
                 and fraction is not None and fraction >= .5)
    report["hypothesis_supported"] = local
    report["startup"] = startup
    report["windows"] = {"startup": [1, 100], "steady": [100, 300],
                         "full": [0, 301], "native_steady_s": [.5, 1.5]}
    for label, sl in (("startup", slice(1, 100)),
                      ("steady", slice(100, 300)), ("full", slice(0, 301))):
        report["windows"][label + "_difference"] = {
            name: {"rms": _rms(raw["difference_" + name][sl]),
                   "max_abs": float(np.max(np.abs(raw["difference_" + name][sl])))}
            for name in ("motor_rate391", "bank", "activation", "command",
                         "actual_joint", "body_position")}
    native_sl = (native_time >= .5) & (native_time < 1.5)
    raw["native_steady_mask"] = native_sl
    raw["native_steady_selected_time_s"] = native_time[native_sl]
    report["windows"]["native_steady_selected_samples"] = int(native_sl.sum())
    report["windows"]["native_steady_first_sample_index_1_based"] = (
        int(np.flatnonzero(native_sl)[0]+1) if native_sl.any() else None)
    report["windows"]["native_steady_last_sample_index_1_based"] = (
        int(np.flatnonzero(native_sl)[-1]+1) if native_sl.any() else None)
    if not native_sl.any():
        raise ValueError("No raw-time native steady samples")
    report["windows"]["native_steady_difference"] = {
        name: {"rms": _rms(raw["difference_native_" + name][native_sl]),
               "max_abs": float(np.max(np.abs(raw["difference_native_" + name][native_sl])))}
        for name in ("normal_load_bw", "upright")}
    report["windows"]["native_steady_slip"] = {
        "joint_valid_values": int(slip_available[native_sl].sum()),
        "baseline_valid_values": int(b["native"]["slip_valid"][native_sl].sum()),
        "candidate_valid_values": int(c["native"]["slip_valid"][native_sl].sum()),
        "difference_rms_valid_only": (float(np.sqrt(np.sum(slip_delta[native_sl]**2)
            /slip_available[native_sl].sum())) if slip_available[native_sl].any() else None)}
    for stage in report["stages"]:
        report["stages"][stage]["index_kind"] = (
            "native_sample_1_based" if stage.startswith("native_")
            else "observed_control_boundary_0_based" if stage in (
                "old_filtered", "new_filtered", "last_features",
                "old_velocity_lowpass") else "applied_endpoint_control_1_based"
            if stage == "source_drive" else "endpoint_control_0_based")
    report["timing"] = {"sensory_observed_control_tick_t": "features and old/new filters observed at t",
                        "source_drive_control_tick_before_t": "returned drive applies to endpoint t+1",
                        "endpoint_rate": "stored after 20 neural ticks",
                        "native_sample": "100us post-integration; no stage-level causal ordering"}
    if progress:
        progress("startup")
    return report, raw


def record_error(error: BaseException, stage: str) -> dict:
    return {"stage": stage, "type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


def invalidate(result: dict) -> None:
    result["validity"] = "incomplete_or_invalid"
    result["H252_local_startup_supported"] = None
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        analysis["hypothesis_supported"] = None


def save_raw_once(out: Path, raw: dict) -> None:
    if not raw or (out / "diagnostic-raw.npz").exists():
        return
    temporary = out / "diagnostic-raw.npz.part"
    primary = None
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **raw)
        os.replace(temporary, out / "diagnostic-raw.npz")
    except BaseException as error:
        primary = error
    try:
        temporary.unlink(missing_ok=True)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E252 raw cleanup: " + repr(error))
    if primary is not None:
        raise primary


def publish(out: Path, result: dict, raw: dict, ledger: dict,
            elapsed, deadline: float) -> None:
    save_raw_once(out, raw)
    ledger["process_wall_s"] = elapsed()
    save_json(out / "errors.json", ledger["errors"])
    save_json(out / "ledger.json", ledger)
    files = {item.name: {"sha256": sha(item), "bytes": item.stat().st_size}
             for item in sorted(out.iterdir()) if item.is_file()
             and item.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": SCHEMA + ".manifest.v1", "files": files,
                "total_bytes": sum(row["bytes"] for row in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_payload = json_bytes(manifest)
    result_payload = json_bytes(result)
    if manifest["total_bytes"] + len(manifest_payload) + len(result_payload) > LIMITS["output_bytes"]:
        raise OSError("E252 output exceeds 64 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E252 deadline before manifest")
    (out / "output-manifest.json").write_bytes(manifest_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E252 deadline before result")
    (out / "result.json").write_bytes(result_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E252 deadline after result")


def settle(out: Path, result: dict, raw: dict, ledger: dict,
           elapsed, deadline: float) -> BaseException | None:
    first = None
    for attempt in range(3):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, ledger, elapsed, deadline)
            result["process_wall_s"] = elapsed()
            save_json(out / "result.json", result)
            if time.perf_counter() >= deadline:
                raise TimeoutError("E252 deadline after final receipt")
            return first
        except BaseException as error:
            if first is None:
                first = error
            ledger["errors"].append(record_error(error, f"publication_{attempt+1}"))
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            invalidate(result)
            if time.perf_counter() >= deadline:
                deadline = float("inf")
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, raw, ledger, elapsed, float("inf"))
        return first
    except BaseException as error:
        ledger["errors"].append(record_error(error, "terminal_publication"))
        result["finalization_pending"] = True
        result["errors"] = ledger["errors"]
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
            (out / "result.json").unlink(missing_ok=True)
            save_json(out / "publication-failure.json", ledger["errors"])
            save_json(out / "result.json", result)
        except BaseException as secondary:
            ledger["errors"].append(record_error(secondary, "terminal_receipt"))
        return first


def run(freeze_path: str, freeze_sha: str, review_path: str,
        review_sha: str, output: str) -> dict:
    began = time.perf_counter()
    out = Path(output).resolve()
    if out != (BASE / "e252-lh-closed-loop-divergence").resolve():
        raise ValueError("Exact write-once E252 output path required")
    if out.exists():
        raise FileExistsError("E252 output already exists")
    out.mkdir(parents=True)
    ledger = {"schema": SCHEMA + ".ledger.v1", "phase": "gate",
              "loaded_input_roles": [], "analysis_requested_count": 0,
              "analysis_entered_count": 0, "analysis_completed_count": 0,
              "original_calculation_count": 0, "completed_stages": [],
              "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
              "neural_constructors": 0, "body_constructors": 0,
              "neural_ticks": 0, "physics_steps": 0, "retries": 0,
              "errors": []}
    result = {"schema": SCHEMA + ".result.v1", "validity": "incomplete_or_invalid",
              "H252_local_startup_supported": None, "candidate_adopted": False,
              "F2_complete": False, "F6_exposed": False,
              "whole_goal_complete": False, "active_UI_changed": False,
              "H251_rejected_preserved": True, "finalization_pending": True}
    raw: dict = {}
    primary = None
    restore_error = None
    def elapsed():
        return time.perf_counter()-began
    def check_budget(stage):
        if elapsed() >= LIMITS["internal_s"]-LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E252 prepublication deadline at " + stage)
        import psutil
        if psutil.Process().memory_info().rss > LIMITS["resident_bytes"]:
            raise MemoryError("E252 RSS exceeds 2 GiB")
    try:
        freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < LIMITS["minimum_free_bytes"]:
            raise OSError("E252 requires 512 MiB free")
        limiter = None
        work_error = None
        try:
            ledger["threadpool_before"] = threadpool_info()
            limiter = threadpool_limits(limits=1)
            ledger["threadpool_observed"] = threadpool_info()
            if any(item.get("num_threads") != 1 for item in ledger["threadpool_observed"]):
                raise RuntimeError("E252 CPU thread limit differs")
            ledger["phase"] = "load"
            def loaded(role):
                ledger["loaded_input_roles"].append(role)
                check_budget(role)
            inputs = load_inputs(freeze, progress=loaded)
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            def progress(stage):
                ledger["completed_stages"].append(stage)
                check_budget(stage)
            analysis, raw = analyze_arrays(inputs, raw=raw, progress=progress)
            ledger["analysis_completed_count"] = 1
        except BaseException as error:
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = analysis
        result["H252_local_startup_supported"] = analysis["hypothesis_supported"]
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:
        primary = error
        invalidate(result)
        ledger["errors"].append(record_error(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append(record_error(restore_error, "threadpool_restore"))
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
                  analysis_requested_count=ledger["analysis_requested_count"],
                  analysis_entered_count=ledger["analysis_entered_count"],
                  analysis_completed_count=ledger["analysis_completed_count"],
                  original_calculation_count=ledger["original_calculation_count"],
                  completed_stages=ledger["completed_stages"],
                  fit_calls=0, solver_calls=0, cuda_calls=0,
                  neural_constructors=0, body_constructors=0,
                  neural_ticks=0, physics_steps=0, errors=ledger["errors"])
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle(out, result, raw, ledger, elapsed,
                               began+LIMITS["internal_s"])
    if primary is None:
        primary = publication_error
    if primary is not None:
        raise primary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
