"""E258 saved E257 startup command and original24 endpoint localization."""

# Preserve primary analysis, restoration, and publication faults independently.
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
FREEZE_SHA = "f7adb87b8d6f227a8c7327581a30c17280247845e36da088ad5450082c0850bc"
SCHEMA = "flylab.m3.e258.lh-partition-startup"
OWNED = ("tools/analyze_c_banc_m3_lh_partition_startup.py",
         "tests/test_c_banc_lh_partition_startup.py")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DOFS = ("coxa_yaw", "coxa_pitch", "coxa_roll", "femur_pitch",
        "femur_roll", "tibia_pitch", "tarsus_pitch")
PARTITION_EDGES = (8191121, 8191220, 8191247, 8191292, 8191340, 8191396)
WINDOWS = {"full": (0, 301), "startup": (1, 100),
           "steady": (100, 300), "initial_200ms": (1, 41)}
LIMITS = {"internal_s": 30, "finalization_reserve_s": 5,
          "external_s": 40, "supervisor_s": 45, "cpu_threads": 1,
          "gpu_count": 0, "original_attempts": 1, "retries": 0,
          "body_constructors": 0, "neural_constructors": 0,
          "neural_ticks": 0, "physics_steps": 0, "fit_calls": 0,
          "solver_calls": 0, "resident_bytes": 2147483648,
          "output_bytes": 67108864, "minimum_free_bytes": 536870912}


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


def pinned(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Repository-relative slash path required")
    relative = Path(name)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ValueError("Unsafe frozen path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Pinned input missing/outside repository")
    return path


def gate(freeze_path: str, freeze_sha: str, review_path: str,
         review_sha: str) -> tuple[dict, dict]:
    if freeze_sha != FREEZE_SHA or sha(Path(freeze_path)) != FREEZE_SHA:
        raise ValueError("E258 freeze SHA mismatch")
    if sha(Path(review_path)) != review_sha:
        raise ValueError("E258 review SHA mismatch")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != SCHEMA + ".freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("arms") != {arm: f"verification/banc-natural-gait-m3-20260919/e257-lh-partition-body/{arm}"
                                      for arm in ("baseline", "candidate")}
            or freeze.get("controls") != 300 or freeze.get("n") != 158706
            or freeze.get("control_dt_s") != .005
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("native_steps") != 15000
            or freeze.get("windows") != {k: list(v) for k, v in WINDOWS.items()}
            or freeze.get("legs") != list(LEGS)
            or freeze.get("resource_limits") != LIMITS
            or freeze.get("thresholds") != {
                "startup_q_max": .04, "startup_q_rms": .018,
                "startup_tendon_rms": .009, "startup_tendon_max": .025,
                "lh_positive_sse_fraction": .5, "fixed_muscle_index": 41,
                "fixed_muscle_leg": "lh", "fixed_muscle_name": "femur_reductor_muscle",
                "muscle_positive_projected_fraction": .8,
                "command_reconstruction_atol": 1e-12,
                "core_exact_tolerance": 1e-5, "rate_cap": 200.0}
            or len(freeze.get("core_observer_rows", [])) != 24
            or freeze.get("candidate_identity") != "4d5bae3386bed2dce2ca8a2add5130449c60a70e3b0d95a51947feeb03c76354"
            or len(freeze.get("partition_edge_ids", [])) != 6
            or len(freeze.get("preserved_edge_ids", [])) != 19):
        raise ValueError("E258 frozen scientific/resource contract differs")
    if (review.get("schema") != SCHEMA + ".review.v1"
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != {name: sha(pinned(name)) for name in OWNED}):
        raise ValueError("E258 approved source review required")
    if sha(pinned(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E258 design SHA differs")
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("E258 pinned hash map required")
        for name, expected in mapping.items():
            if sha(pinned(name)) != expected:
                raise ValueError("E258 pinned bytes differ: " + name)
    if freeze["packages"] != {name: importlib.metadata.version(name)
                              for name in freeze["packages"]}:
        raise ValueError("E258 package versions differ")
    return freeze, review


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("E258 NPZ keys missing: " + str(path))
        return {key: archive[key].copy() for key in keys}


def load_inputs(freeze: dict, progress=None) -> dict:
    result = {}
    for role, keys in {
        "calibration": ("motor_ids", "moment", "rest", "pool", "membership", "ltm"),
        "parameters": ("motor_ids", "membership"),
        "startup": ("target_command", "target_tendon"),
        "edge_patch": ("edge_ids", "pre_ids", "post_ids", "source_values", "target_values"),
        "e256_q": ("original24_rows", "six_normalized", "full25_normalized"),
    }.items():
        result[role] = load_npz(pinned(freeze["paths"][role]), keys)
        if progress:
            progress(role)
    for role in ("muscles", "selection", "roster", "e256_result", "e256_output_review",
                 "e256_final_review", "e257_result", "e257_manifest",
                 "e257_output_review", "e257_final_review", "e257_supervisor"):
        result[role] = json.loads(pinned(freeze["paths"][role]).read_text(encoding="utf-8"))
        if progress:
            progress(role)
    for arm, folder in freeze["arms"].items():
        data = {}
        for role, filename, keys in (
            ("activity", "activity.npz", ("observed_ids", "source_ids", "bridge_ids",
                                         "reachable_motor_ids", "rate", "drive", "adaptation")),
            ("core", "core-activity.npz", ("core_ids", "control_tick", "network_tick",
                                           "physics_time_s", "rate", "drive", "adaptation")),
            ("control", "control-raw.npz", ("control_tick", "network_tick",
                                             "physics_time_s", "motor_rate32",
                                             "activation", "command", "tendon")),
            ("recruitment", "native-recruitment-state.npz",
             ("control_tick", "network_tick", "physics_time_s", "complete", "bank_available",
              "activation_available", "motor_rate_available", "bank", "activation",
              "motor_rate")),
            ("sensory", "sensory-input.npz", ("control_tick_before", "complete", "source_ids",
                                               "old_filtered", "new_filtered", "last_features",
                                               "old_velocity_lowpass", "new_velocity_lowpass",
                                               "returned_source_drive")),
        ):
            data[role] = load_npz(pinned(f"{folder}/{filename}"), keys)
            if progress:
                progress(arm + "/" + role)
        for role, filename in (("sensory_metadata", "sensory-metadata.json"),
                               ("trace", "trace.json"), ("arm_result", "arm-result.json")):
            data[role] = json.loads(pinned(f"{folder}/{filename}").read_text(encoding="utf-8"))
            if progress:
                progress(arm + "/" + role)
        data["output_mask"] = np.load(pinned(f"{folder}/observed-start/array_00002.npy"),
                                      allow_pickle=False)
        if progress:
            progress(arm + "/output_mask")
        result[arm] = data
    return result


def checked(value, shape: tuple, dtype: str | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or (dtype is not None and array.dtype != np.dtype(dtype)):
        raise ValueError(f"E258 shape/dtype differs: {array.shape}/{array.dtype}, {shape}/{dtype}")
    if array.dtype.kind in "fiu" and not np.isfinite(array).all():
        raise ValueError("E258 nonfinite required array")
    return array


def aligned(value: np.ndarray, expected: np.ndarray, *, atol=0.) -> None:
    if not np.allclose(value, expected, rtol=0, atol=atol):
        raise ValueError("E258 saved clocks disagree")


def route(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("E258 E248 route object required")
    if "selected" in value:
        if not isinstance(value.get("rule"), str) or not value["rule"]:
            raise ValueError("E258 E248 route wrapper malformed")
        value = value["selected"]
    if (not isinstance(value, dict) or value.get("port") != 85
            or value.get("kind") != "claw" or value.get("reachable_count") != 55):
        raise ValueError("E258 E248 route differs")
    return value


def validate(inputs: dict, *, sizes=(301, 391, 96, 158706),
             core_rows: tuple[int, ...] | None = None) -> dict:
    rows, motors, muscles, neurons = sizes
    if rows != 301 or motors != 391 or muscles != 96 or neurons != 158706:
        raise ValueError("E258 original fixed dimensions required")
    e257 = inputs["e257_result"]
    if (e257.get("validity") != "valid_short_native_body_diagnostic"
            or e257.get("finalization_pending") is not False
            or e257.get("baseline") != "complete_healthy_short_diagnostic"
            or e257.get("candidate") != "complete_healthy_short_diagnostic"
            or e257.get("H257", {}).get("H257_supported") is not False
            or inputs["e257_output_review"].get("all_passed") is not True):
        raise ValueError("Valid complete rejected E257 with parent audit required")
    e256 = inputs["e256_result"]
    if (e256.get("validity") != "valid_six_existing_edge_partition_contrast"
            or e256.get("H256_supported") is not True
            or e256.get("finalization_pending") is not False
            or inputs["e256_output_review"].get("validity") != "independently_valid"
            or inputs["e256_output_review"].get("H256_supported") is not True):
        raise ValueError("Valid static E256 receipt required")
    c = inputs["calibration"]
    p = inputs["parameters"]
    motor_ids = checked(c["motor_ids"], (motors,), "int64")
    membership = checked(c["membership"], (motors,), "int64")
    if (not np.array_equal(motor_ids, checked(p["motor_ids"], (motors,), "int64"))
            or not np.array_equal(membership, checked(p["membership"], (motors,), "int64"))
            or len(np.unique(motor_ids)) != motors
            or not np.array_equal(motor_ids, np.sort(motor_ids))
            or np.any((membership < 0) | (membership >= muscles))):
        raise ValueError("E258 original motor roster/membership differs")
    moment = checked(c["moment"], (42, muscles), "float64")
    checked(c["rest"], (42,), "float64")
    pool = checked(c["pool"], (muscles, motors), "float64")
    checked(c["ltm"], (6,), "int64")
    if (np.any(pool < 0) or not np.allclose(pool.sum(axis=1), 1, rtol=0, atol=1e-12)
            or not np.array_equal(np.argmax(pool, axis=0), membership)
            or not np.array_equal((pool > 0).sum(axis=0), np.ones(motors))):
        raise ValueError("E258 original muscle pool differs")
    groups = inputs["muscles"]["muscles"]
    if len(groups) != muscles:
        raise ValueError("E258 requires 96 named muscles")
    for m, group in enumerate(groups):
        if (group["leg"] not in LEGS or not group["muscle"]
                or group["columns"] != np.flatnonzero(membership == m).tolist()):
            raise ValueError("E258 named muscle ownership differs")
    if groups[41]["leg"] != "lh" or groups[41]["muscle"] != "femur_reductor_muscle":
        raise ValueError("E258 fixed muscle41 provenance differs")
    target = inputs["startup"]
    checked(target["target_command"], (99, 42), "float64")
    checked(target["target_tendon"], (99, 6), "float64")
    roster = inputs["roster"]
    ids24 = np.asarray(core_rows if core_rows is not None else roster["original_24_node_rows"],
                       np.int64)
    if (ids24.shape != (24,) or len(np.unique(ids24)) != 24
            or not np.array_equal(ids24, np.asarray(roster["original_24_node_rows"]))
            or not np.array_equal(ids24, inputs["e256_q"]["original24_rows"])
            or not np.array_equal(ids24, inputs["e256_result"]["metrics"]["Q"]["original24_rows"])):
        raise ValueError("E258 original24 roster/Q alignment differs")
    for key in ("six_normalized", "full25_normalized"):
        checked(inputs["e256_q"][key], (2, 40, 24), "float64")
    patch = inputs["edge_patch"]
    for key, dtype in (("edge_ids", "int64"), ("pre_ids", "int64"),
                       ("post_ids", "int64")):
        checked(patch[key], (6,), dtype)
    for key in ("source_values", "target_values"):
        value = np.asarray(patch[key])
        if (value.shape != (6,) or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(value).all()):
            raise ValueError("E258 frozen patch value shape/dtype/finite differs")
        if (value.dtype == np.float64
                and not np.array_equal(value, value.astype(np.float32).astype(np.float64))):
            raise ValueError("E258 frozen patch f64 is not exactly representable as f32")
    if len(np.unique(patch["edge_ids"])) != 6:
        raise ValueError("E258 six edge IDs differ")
    selection = route(inputs["selection"])
    source = checked(selection["source_rows"], (23,), "int64")
    bridge = checked(selection["bridge_rows"], (4,), "int64")
    reachable = checked(selection["reachable_motor_rows"], (55,), "int64")
    if (not np.array_equal(patch["edge_ids"], np.asarray(PARTITION_EDGES))
            or not np.isin(patch["pre_ids"], source).all()
            or not np.all(patch["post_ids"] == 111970)
            or len(np.unique(np.r_[source, bridge, reachable])) != 82
            or not np.isin(reachable, motor_ids).all()):
        raise ValueError("E258 six-edge/observed route differs")
    joint_columns = None
    for arm in ("baseline", "candidate"):
        data = inputs[arm]
        if (data["arm_result"].get("status") != "complete_healthy_short_diagnostic"
                or data["arm_result"].get("complete") is not True):
            raise ValueError("E258 complete healthy arm required")
        a, k, r, s, core = (data[key] for key in
                            ("activity", "control", "recruitment", "sensory", "core"))
        for name, expected in (("observed_ids", np.r_[source, bridge, reachable]),
                               ("source_ids", source), ("bridge_ids", bridge),
                               ("reachable_motor_ids", reachable)):
            if not np.array_equal(a[name], expected):
                raise ValueError("E258 activity ID order differs: " + name)
        for field in ("rate", "drive", "adaptation"):
            checked(a[field], (rows, 82), "float32")
            checked(core[field], (rows, 24), "float32")
        if not np.array_equal(checked(core["core_ids"], (24,), "int64"), ids24):
            raise ValueError("E258 core24 IDs differ")
        for key in ("control_tick", "network_tick"):
            clock = np.arange(rows, dtype=np.int64)*(20 if key == "network_tick" else 1)
            aligned(checked(k[key], (rows,), "int64"), clock)
            aligned(checked(r[key], (rows,), "int64"), clock)
            aligned(checked(core[key], (rows,), "int64"), clock)
        expected_time = np.arange(rows, dtype=np.float64)*.005
        for store in (k, r, core):
            aligned(checked(store["physics_time_s"], (rows,), "float64"),
                    expected_time, atol=2e-8)
        for key, width, dtype in (("motor_rate32", motors, "float32"),
                                  ("activation", muscles, "float64"),
                                  ("command", 42, "float64"),
                                  ("tendon", 6, "float64")):
            checked(k[key], (rows, width), dtype)
        checked(r["bank"], (rows, motors, 3), "float64")
        checked(r["activation"], (rows, muscles), "float64")
        checked(r["motor_rate"], (rows, motors), "float64")
        for key in ("complete", "bank_available", "activation_available",
                    "motor_rate_available"):
            if not checked(r[key], (rows,), "bool").all():
                raise ValueError("E258 incomplete recruitment availability")
        if (not np.array_equal(k["activation"], r["activation"])
                or not np.allclose(k["motor_rate32"], r["motor_rate"], rtol=0, atol=1e-5)):
            raise ValueError("E258 duplicate recruitment/control differs")
        motor_lookup = {int(value): j for j, value in enumerate(motor_ids)}
        reachable_columns = np.asarray([motor_lookup[int(value)] for value in reachable])
        if not np.allclose(a["rate"][:, 27:], k["motor_rate32"][:, reachable_columns],
                           rtol=0, atol=1e-5):
            raise ValueError("E258 activity/reachable motor rate differs")
        checked(s["control_tick_before"], (rows-1,), "int64")
        aligned(s["control_tick_before"], np.arange(rows-1, dtype=np.int64))
        if not checked(s["complete"], (rows-1,), "bool").all():
            raise ValueError("E258 sensory record incomplete")
        if not np.array_equal(checked(s["source_ids"], (23,), "int64"), source):
            raise ValueError("E258 sensory source IDs differ")
        for key, width, dtype in (("old_filtered", 241, "float64"),
                                  ("new_filtered", 241, "float64"),
                                  ("last_features", 241, "float64"),
                                  ("old_velocity_lowpass", 42, "float64"),
                                  ("new_velocity_lowpass", 42, "float64"),
                                  ("returned_source_drive", 23, "float32")):
            checked(s[key], (rows-1, width), dtype)
        expected_source_drive = np.repeat(
            s["old_filtered"][:, 85, None].astype(np.float32), 23, axis=1)
        if not np.array_equal(s["returned_source_drive"], expected_source_drive):
            raise ValueError("E258 old_filtered port85 does not produce source drive")
        if not np.array_equal(a["drive"][1:, :23], s["returned_source_drive"]):
            raise ValueError("E258 source drive must apply at next endpoint")
        if not np.array_equal(checked(data["output_mask"], (neurons,), "float32"),
                              inputs["baseline"]["output_mask"]):
            raise ValueError("E258 output mask differs")
        meta = data["sensory_metadata"]
        physical = checked(meta["joint_columns"], (6, 7), "int64")
        if (set(physical.flat) != set(range(42))
                or not np.array_equal(meta["source_rows"], source)):
            raise ValueError("E258 physical joint columns/source differ")
        if joint_columns is None:
            joint_columns = physical
        elif not np.array_equal(joint_columns, physical):
            raise ValueError("E258 arm physical joint mapping differs")
        samples = data["trace"].get("samples")
        if not isinstance(samples, list) or len(samples) != rows:
            raise ValueError("E258 complete trace rows required")
        for t, row in enumerate(samples):
            if row["control_tick"] != t or abs(row["time_s"]-t*.005) > 2e-8:
                raise ValueError("E258 trace endpoint clock differs")
        if not np.array_equal(checked([row["joint_targets_rad"] for row in samples],
                                      (rows, 42), "float64"), k["command"]):
            raise ValueError("E258 trace/control command differs")
    return {"motor_ids": motor_ids, "membership": membership,
            "moment": moment, "joint_columns": joint_columns,
            "source_ids": source, "bridge_ids": bridge,
            "reachable_motor_ids": reachable, "core_ids": ids24,
            "muscles": groups}


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, np.float64)
    return float(np.sqrt(np.mean(array*array)))


def difference(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Promote both saved operands before subtracting, including float32 states."""
    return (np.asarray(candidate, dtype=np.float64)
            - np.asarray(baseline, dtype=np.float64))


def stage_summary(delta: np.ndarray, ticks: np.ndarray, *, tolerance=1e-5,
                  rate=False) -> dict:
    values = np.asarray(delta, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(ticks) or not len(ticks):
        raise ValueError("E258 stage/tick shape differs")
    exact = np.flatnonzero(np.any(values != 0, axis=1))
    above = np.flatnonzero(np.any(np.abs(values) > tolerance, axis=1))
    result = {"rms": rms(values), "max_abs": float(np.max(np.abs(values))),
              "signed_sum": float(values.sum()),
              "positive_sum": float(np.maximum(values, 0).sum()),
              "negative_sum": float(np.minimum(values, 0).sum()),
              "first_exact_control": int(ticks[exact[0]]) if len(exact) else None,
              "first_abs_gt_1e_minus_5_control": int(ticks[above[0]]) if len(above) else None,
              "sample_count": len(ticks), "value_count": int(values.size)}
    if rate:
        result["rate_cap_normalized_rms"] = result["rms"]/200.
    return result


def _endpoint_projection(contribution: np.ndarray, error: np.ndarray, control: int,
                         joint: int, leg: np.ndarray, dof: np.ndarray,
                         muscles: list[dict]) -> dict:
    sign = float(np.sign(error[control-1, joint]))
    projected = sign*contribution[control, joint]
    positive = np.maximum(projected, 0)
    negative = np.minimum(projected, 0)
    denominator = float(positive.sum())
    fraction = float(positive[41]/denominator) if denominator > 0 else None
    return {"control": control, "time_s": control*.005, "joint_column": joint,
            "leg": LEGS[leg[joint]], "dof": DOFS[dof[joint]],
            "candidate_error_rad": float(error[control-1, joint]),
            "candidate_error_sign": sign,
            "signed_by_muscle": contribution[control, joint].tolist(),
            "projected_by_muscle": projected.tolist(),
            "projected_positive_by_muscle": positive.tolist(),
            "projected_negative_by_muscle": negative.tolist(),
            "projected_signed_sum": float(projected.sum()),
            "projected_positive_sum": denominator,
            "projected_negative_sum": float(negative.sum()),
            "muscle41_name": muscles[41]["muscle"],
            "muscle41_positive_contribution": float(positive[41]),
            "muscle41_positive_fraction": fraction,
            "muscle41_dominates": bool(fraction is not None and fraction >= .8)}


def analyze_arrays(inputs: dict, *, raw: dict | None = None,
                   progress=None, sizes=(301, 391, 96, 158706),
                   core_rows: tuple[int, ...] | None = None) -> tuple[dict, dict]:
    """Analyze already-loaded immutable endpoint arrays; caller owns partial raw."""
    if raw is None:
        raw = {}
    mapping = validate(inputs, sizes=sizes, core_rows=core_rows)
    for name in ("motor_ids", "membership", "joint_columns", "source_ids",
                 "bridge_ids", "reachable_motor_ids", "core_ids"):
        raw[name] = mapping[name].copy()
    baseline, candidate = inputs["baseline"], inputs["candidate"]
    raw["endpoint_control_tick"] = baseline["control"]["control_tick"].copy()
    raw["endpoint_time_s"] = baseline["control"]["physics_time_s"].copy()
    observed = baseline["sensory"]["control_tick_before"].copy()
    raw["sensory_observed_control_tick"] = observed
    raw["sensory_observed_time_s"] = observed.astype(np.float64)*.005
    raw["sensory_applied_endpoint_control_tick"] = observed+1
    raw["sensory_applied_endpoint_time_s"] = (observed+1).astype(np.float64)*.005
    raw["source_drive_applied_delta"] = difference(
        candidate["sensory"]["returned_source_drive"],
        baseline["sensory"]["returned_source_drive"])
    for name, key, sl in (("source_rate", "rate", slice(0, 23)),
                           ("source_drive", "drive", slice(0, 23)),
                           ("source_adaptation", "adaptation", slice(0, 23)),
                           ("bridge_rate", "rate", slice(23, 27)),
                           ("bridge_drive", "drive", slice(23, 27)),
                           ("bridge_adaptation", "adaptation", slice(23, 27)),
                           ("reachable_rate", "rate", slice(27, None)),
                           ("reachable_drive", "drive", slice(27, None)),
                           ("reachable_adaptation", "adaptation", slice(27, None))):
        raw["difference_" + name] = difference(
            candidate["activity"][key][:, sl], baseline["activity"][key][:, sl])
    for key in ("rate", "drive", "adaptation"):
        raw["difference_core_" + key] = difference(candidate["core"][key],
                                                    baseline["core"][key])
    raw["difference_motor_rate391"] = difference(
        candidate["control"]["motor_rate32"], baseline["control"]["motor_rate32"])
    raw["difference_activation96"] = difference(
        candidate["control"]["activation"], baseline["control"]["activation"])
    raw["difference_muscle41_activation"] = raw["difference_activation96"][:, 41].copy()
    raw["difference_command"] = difference(candidate["control"]["command"],
                                            baseline["control"]["command"])
    raw["difference_old_filtered"] = difference(
        candidate["sensory"]["old_filtered"], baseline["sensory"]["old_filtered"])
    if progress:
        progress("stages")

    rest = inputs["calibration"]["rest"]
    moment = mapping["moment"]
    for arm in ("baseline", "candidate"):
        predicted = rest + .02 * inputs[arm]["control"]["activation"] @ moment.T
        actual = inputs[arm]["control"]["command"]
        raw[arm + "_command_reconstruction"] = predicted
        raw[arm + "_row0_command_residual"] = actual[0]-predicted[0]
        if not np.allclose(predicted[1:], actual[1:], rtol=0, atol=1e-12):
            raise ValueError("E258 original activation-to-command reconstruction differs")
    contribution = (.02*raw["difference_activation96"][:, None, :]
                    * moment[None, :, :])
    raw["muscle_joint_contribution"] = contribution
    if not np.allclose(contribution.sum(axis=2)[1:], raw["difference_command"][1:],
                       rtol=0, atol=1e-12):
        raise ValueError("E258 96-muscle command difference differs")
    if progress:
        progress("muscle_command")

    target = inputs["startup"]["target_command"]
    be = baseline["control"]["command"][1:100]-target
    ce = candidate["control"]["command"][1:100]-target
    change = ce-be
    signed = ce*ce-be*be
    linear = 2*be*change
    quadratic = change*change
    if not np.allclose(signed, linear+quadratic, rtol=0, atol=1e-12):
        raise ValueError("E258 signed SSE decomposition differs")
    positive = np.maximum(signed, 0)
    negative = np.minimum(signed, 0)
    violations = (np.abs(be) <= .04) & (np.abs(ce) > .04)
    for name, value in (("baseline_error", be), ("candidate_error", ce),
                        ("signed_sse_change", signed), ("linear_sse_term", linear),
                        ("quadratic_sse_term", quadratic),
                        ("positive_sse_change", positive),
                        ("negative_sse_change", negative),
                        ("new_violation", violations)):
        raw["startup_" + name] = value
    joint_columns = mapping["joint_columns"]
    leg_index = np.empty(42, np.int64)
    dof_index = np.empty(42, np.int64)
    for i in range(6):
        leg_index[joint_columns[i]] = i
        dof_index[joint_columns[i]] = np.arange(7)
    raw["joint_leg_index"] = leg_index
    raw["joint_dof_index"] = dof_index
    per_leg = {}
    for i, name in enumerate(LEGS):
        j = joint_columns[i]
        per_leg[name] = {"new_violation_count": int(violations[:, j].sum()),
                         "signed_sse_change": float(signed[:, j].sum()),
                         "linear_sse_term": float(linear[:, j].sum()),
                         "quadratic_sse_term": float(quadratic[:, j].sum()),
                         "positive_sse_change": float(positive[:, j].sum()),
                         "negative_sse_change": float(negative[:, j].sum()),
                         "baseline_q_rms_rad": rms(be[:, j]),
                         "candidate_q_rms_rad": rms(ce[:, j]),
                         "baseline_q_max_abs_rad": float(np.max(np.abs(be[:, j]))),
                         "candidate_q_max_abs_rad": float(np.max(np.abs(ce[:, j])))}
    violation_rows = np.argwhere(violations)
    first = None
    if len(violation_rows):
        t, j = map(int, violation_rows[0])
        first = {"control": t+1, "joint_column": j,
                 "leg": LEGS[leg_index[j]], "dof": DOFS[dof_index[j]],
                 "baseline_error_rad": float(be[t, j]),
                 "candidate_error_rad": float(ce[t, j])}
    peak_t, peak_j = np.unravel_index(int(np.argmax(np.abs(ce))), ce.shape)
    peak = {"control": int(peak_t+1), "joint_column": int(peak_j),
            "leg": LEGS[leg_index[peak_j]], "dof": DOFS[dof_index[peak_j]],
            "baseline_error_rad": float(be[peak_t, peak_j]),
            "candidate_error_rad": float(ce[peak_t, peak_j])}
    denominator = float(positive.sum())
    lh_fraction = (float(positive[:, joint_columns[2]].sum())/denominator
                   if denominator > 0 else None)
    endpoints = {}
    if first is not None:
        endpoints["first_new_violation"] = _endpoint_projection(
            contribution, ce, first["control"], first["joint_column"],
            leg_index, dof_index, mapping["muscles"])
    endpoints["maximum_candidate_error"] = _endpoint_projection(
        contribution, ce, peak["control"], peak["joint_column"],
        leg_index, dof_index, mapping["muscles"])
    local = bool(len(violation_rows)
                 and np.all(leg_index[violation_rows[:, 1]] == 2)
                 and lh_fraction is not None and lh_fraction >= .5)
    muscle_dominant = bool(first is not None and all(
        endpoints[key]["muscle41_dominates"]
        for key in ("first_new_violation", "maximum_candidate_error")))
    startup = {"baseline_q_rms_rad": rms(be), "candidate_q_rms_rad": rms(ce),
               "baseline_q_max_abs_rad": float(np.max(np.abs(be))),
               "candidate_q_max_abs_rad": float(np.max(np.abs(ce))),
               "new_violation_count": int(violations.sum()),
               "signed_sse_change": float(signed.sum()),
               "linear_sse_term": float(linear.sum()),
               "quadratic_sse_term": float(quadratic.sum()),
               "positive_sse_change": denominator,
               "negative_sse_change": float(negative.sum()),
               "lh_positive_sse_fraction": lh_fraction, "legs": per_leg,
               "all_new_violations_lh_and_lh_positive_sse_at_least_half": local}
    tendon = inputs["startup"]["target_tendon"]
    for arm in ("baseline", "candidate"):
        error = inputs[arm]["control"]["tendon"][1:100]-tendon
        startup[arm + "_tendon_rms"] = rms(error)
        startup[arm + "_tendon_max_abs"] = float(np.max(np.abs(error)))
    if progress:
        progress("startup")

    stages = {name: raw["difference_" + name] for name in
              ("core_rate", "core_drive", "core_adaptation",
               "source_rate", "source_drive", "source_adaptation",
               "bridge_rate", "bridge_drive", "bridge_adaptation",
               "reachable_rate", "reachable_drive", "reachable_adaptation",
               "motor_rate391", "activation96", "muscle41_activation")}
    stages["old_filtered"] = raw["difference_old_filtered"]
    stages["source_drive_applied"] = raw["source_drive_applied_delta"]
    report_windows = {}
    for window, (lo, hi) in WINDOWS.items():
        entry = {"endpoint_controls": [lo, hi], "stages": {}}
        for name, delta in stages.items():
            if name == "old_filtered":
                tick = raw["sensory_observed_control_tick"]
                selected = (tick >= lo) & (tick < hi)
            elif name == "source_drive_applied":
                tick = raw["sensory_applied_endpoint_control_tick"]
                selected = (tick >= lo) & (tick < hi)
            else:
                tick = raw["endpoint_control_tick"]
                selected = (tick >= lo) & (tick < hi)
            if selected.any():
                entry["stages"][name] = stage_summary(delta[selected], tick[selected],
                    rate=name in ("core_rate", "source_rate", "bridge_rate",
                                  "reachable_rate", "motor_rate391"))
            else:
                entry["stages"][name] = {"available": False, "sample_count": 0}
        report_windows[window] = entry
    q = inputs["e256_result"]["metrics"]["Q"]
    report = {"schema": SCHEMA + ".analysis.v1", "first_new_violation": first,
              "maximum_candidate_error": peak, "endpoint_projections": endpoints,
              "muscle_roster": [
                  {"muscle_index": i, "leg": group["leg"],
                   "muscle": group["muscle"], "columns": group["columns"]}
                  for i, group in enumerate(mapping["muscles"])],
              "startup": startup, "windows": report_windows,
              "E256_static_Q6": float(q["Q6"]),
              "E256_static_Q25": float(q["Q25"]),
              "E256_comparison_scope": "body-free held-input tick40000; not comparable as causal ratio to E257 control0 body dynamics",
              "H258_same_muscle": bool(local and muscle_dominant),
              "timing": {"old_filtered": "observed at control t",
                         "source_drive_applied": "encoder at t, endpoint after t+1 control",
                         "other_stages": "endpoint after neural ticks; no within-5ms causal latency"},
              "claim_scope": "saved endpoint muscle attribution and observed stage differences; no biological or neural causal proof"}
    if progress:
        progress("summary")
    return report, raw


def error_record(error: BaseException, stage: str) -> dict:
    return {"stage": stage, "type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


def invalidate(result: dict) -> None:
    result["validity"] = "incomplete_or_invalid"
    result["H258_same_muscle"] = None
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        analysis["H258_same_muscle"] = None
        startup = analysis.get("startup")
        if isinstance(startup, dict):
            startup["all_new_violations_lh_and_lh_positive_sse_at_least_half"] = None
        for entry in analysis.get("endpoint_projections", {}).values():
            entry["muscle41_dominates"] = None


def raw_once(out: Path, raw: dict) -> None:
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
            primary.add_note("Secondary E258 raw cleanup: " + repr(error))
    if primary is not None:
        raise primary


def publish(out: Path, result: dict, raw: dict, ledger: dict,
            elapsed, deadline: float) -> None:
    raw_once(out, raw)
    ledger["process_wall_s"] = elapsed()
    save_json(out / "errors.json", ledger["errors"])
    save_json(out / "ledger.json", ledger)
    files = {item.name: {"sha256": sha(item), "bytes": item.stat().st_size}
             for item in sorted(out.iterdir()) if item.is_file()
             and item.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": SCHEMA + ".manifest.v1", "files": files,
                "total_bytes": sum(item["bytes"] for item in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_payload = json_bytes(manifest)
    result_payload = json_bytes(result)
    if manifest["total_bytes"] + len(manifest_payload) + len(result_payload) > LIMITS["output_bytes"]:
        raise OSError("E258 output exceeds 64 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E258 deadline before manifest")
    (out / "output-manifest.json").write_bytes(manifest_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E258 deadline before result")
    (out / "result.json").write_bytes(result_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E258 deadline after result")


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
                raise TimeoutError("E258 deadline after final receipt")
            return first
        except BaseException as error:
            if first is None:
                first = error
            ledger["errors"].append(error_record(error, f"publication_{attempt+1}"))
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
        ledger["errors"].append(error_record(error, "terminal_publication"))
        result["finalization_pending"] = True
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
            (out / "result.json").unlink(missing_ok=True)
        except BaseException as secondary:
            ledger["errors"].append(error_record(secondary, "terminal_receipt"))
        for name, value in (("errors.json", ledger["errors"]),
                            ("ledger.json", ledger),
                            ("publication-failure.json", ledger["errors"]),
                            ("result.json", result)):
            try:
                result["errors"] = ledger["errors"]
                save_json(out / name, value)
            except BaseException as secondary:
                ledger["errors"].append(error_record(secondary, "terminal_" + name))
        return first


def run(freeze_path: str, freeze_sha: str, review_path: str,
        review_sha: str, output: str) -> dict:
    began = time.perf_counter()
    out = Path(output).resolve()
    if out != (BASE / "e258-lh-partition-startup").resolve():
        raise ValueError("Exact E258 write-once output path required")
    if out.exists():
        raise FileExistsError("E258 output already exists")
    out.mkdir(parents=True)
    ledger = {"schema": SCHEMA + ".ledger.v1", "phase": "gate",
              "loaded_input_roles": [], "analysis_requested_count": 0,
              "analysis_entered_count": 0, "analysis_completed_count": 0,
              "original_calculation_count": 0, "completed_stages": [],
              "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
              "neural_constructors": 0, "body_constructors": 0,
              "neural_ticks": 0, "physics_steps": 0, "retries": 0, "errors": []}
    result = {"schema": SCHEMA + ".result.v1", "validity": "incomplete_or_invalid",
              "H258_same_muscle": None, "candidate_adopted": False,
              "F2_complete": False, "F6_exposed": False,
              "whole_goal_complete": False, "active_UI_changed": False,
              "H257_rejected_preserved": True, "finalization_pending": True}
    raw: dict = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter()-began

    def check_budget(stage):
        if elapsed() >= LIMITS["internal_s"]-LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E258 prepublication deadline at " + stage)
        import psutil
        if psutil.Process().memory_info().rss > LIMITS["resident_bytes"]:
            raise MemoryError("E258 RSS exceeds 2 GiB")

    try:
        freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < LIMITS["minimum_free_bytes"]:
            raise OSError("E258 requires 512 MiB free")
        limiter = None
        work_error = None
        try:
            ledger["threadpool_before"] = threadpool_info()
            limiter = threadpool_limits(limits=1)
            ledger["threadpool_observed"] = threadpool_info()
            if any(item.get("num_threads") != 1 for item in ledger["threadpool_observed"]):
                raise RuntimeError("E258 CPU thread limit differs")
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

            analysis, raw = analyze_arrays(inputs, raw=raw, progress=progress,
                                          core_rows=tuple(freeze["core_observer_rows"]))
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
        result["H258_same_muscle"] = analysis["H258_same_muscle"]
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:
        primary = error
        invalidate(result)
        ledger["errors"].append(error_record(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append(error_record(restore_error, "threadpool_restore"))
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
