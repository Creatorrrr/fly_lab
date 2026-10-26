"""E194: read saved native states to diagnose knee-velocity boundary sampling.

No neural model, command replay, or scientific physics step is performed here.
The body constructor's neutral settling is counted separately.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.body import LEGS
from flylab.c.storage import StateStore
from tools import analyze_c_banc_observation as observation
from tools.analyze_c_banc_m3_signed_contact import construct_body_safely

BASE = Path("verification/banc-natural-gait-m3-20260919")
METADATA = BASE / "parent-e194-frozen-metadata.json"
FREEZE = BASE / "e194-native-knee-velocity-freeze.json"
REVIEW = BASE / "parent-e194-source-review.json"
OUTPUT = BASE / "e194-native-knee-velocity"
SCHEMA = "flylab.m3.e194.native-knee-velocity"
ROWS = 10001
NATIVE_ROWS = 10000
CONTROL_INTERVALS = 200
STEPS_PER_CONTROL = 50
INTERNAL_S = 50
GEOMETRY_ATOL = 1e-12
MAX_OUTPUT_BYTES = 256 * 1024**2
MIN_FREE_BYTES = 1024**3
FLAGS = (
    "step_requested", "integration_complete", "state_captured",
    "state_copy_complete", "shadow_forward_complete", "observer_returned",
    "diagnostics_returned", "nonfoot_returned", "observation_complete",
    "source_verified",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_path(value: str) -> Path:
    require(isinstance(value, str) and value, "path missing")
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "missing or escaping input: " + value)
    return path


def verify_gate(freeze_path: Path, review_path: Path, output: Path) -> dict:
    freeze = read_json(freeze_path)
    review = read_json(review_path)
    require(freeze.get("schema") == SCHEMA + ".freeze.v1", "freeze schema")
    require(review.get("schema") == SCHEMA + ".source-review.v1", "review schema")
    require(review.get("approved") is True or review.get("accepted") is True,
            "parent source approval absent")
    require(review.get("freeze_sha256") == sha256(freeze_path), "review freeze hash")
    metadata_path = safe_path(freeze["metadata_path"])
    require(sha256(metadata_path) == freeze["metadata_sha256"], "metadata hash")
    metadata = read_json(metadata_path)
    require(metadata.get("schema") == SCHEMA + ".metadata.v1", "metadata schema")
    require(freeze.get("design_sha256") == metadata.get("design_sha256")
            == sha256(safe_path(metadata["design_path"])), "design hash")
    require(Path(freeze["output_path"]) == OUTPUT and output.resolve() == (ROOT / OUTPUT).resolve(),
            "output path")
    require(metadata.get("original_runs") == 0 and metadata.get("candidate_adopted") is False,
            "metadata prior-run contract")
    require(metadata.get("reserved_s") == 60 and metadata.get("cumulative_before_s", 0)
            + 60 <= metadata.get("cumulative_cap_s", 0), "cumulative cap")
    params = freeze.get("parameters", {})
    for key, value in (("native_rows", NATIVE_ROWS), ("output_rows", ROWS),
                       ("control_intervals", CONTROL_INTERVALS),
                       ("steps_per_control", STEPS_PER_CONTROL),
                       ("internal_wall_s", INTERNAL_S), ("cpu_threads", 1),
                       ("blas_threads", 1), ("gpu_count", 0),
                       ("automatic_retries", 0)):
        require(params.get(key) == value, "parameter: " + key)
    require(params.get("maximum_output_bytes") == MAX_OUTPUT_BYTES
            and params.get("minimum_free_bytes") == MIN_FREE_BYTES, "storage cap")
    source_hashes = freeze.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes, "source inventory")
    require(review.get("source_hashes") == source_hashes, "review source inventory")
    for name, expected in source_hashes.items():
        require(sha256(safe_path(name)) == expected, "source changed: " + name)
    for name, expected in metadata["inputs"].items():
        require(source_hashes.get(name) == expected, "input unpinned: " + name)
    for name, expected in metadata["reused_sources"].items():
        require(source_hashes.get(name) == expected, "reused source unpinned: " + name)
    for field in ("source_native", "source_layout", "source_result", "source_review",
                  "encoder", "provenance", "trace", "e193_review"):
        require(metadata[field] in source_hashes, "missing required hash: " + field)
    checkpoint = (ROOT / metadata["checkpoint"]).resolve()
    require(checkpoint.is_relative_to(ROOT) and checkpoint.is_dir(), "checkpoint path")
    for path in checkpoint.iterdir():
        if path.is_file():
            require(str(path.relative_to(ROOT)).replace("\\", "/") in source_hashes,
                    "checkpoint file unpinned")
    require(not output.exists(), "output already exists")
    return metadata


def load_sources(metadata: dict) -> tuple[dict, dict, dict, dict, dict]:
    layout = read_json(safe_path(metadata["source_layout"]))
    result = read_json(safe_path(metadata["source_result"]))
    review = read_json(safe_path(metadata["source_review"]))
    trace = read_json(safe_path(metadata["trace"]))
    provenance = read_json(safe_path(metadata["provenance"]))
    require(layout.get("body_hash") == metadata["body_hash"]
            == trace.get("body_hash") == provenance.get("body_model_hash"), "body hash")
    require(result.get("accepted") is True and result.get("publication_complete") is True
            and result.get("resource_limit_met") is True, "E157 result not accepted")
    require(review.get("accepted") is True and not review.get("failed_checks"),
            "E157 parent review not accepted")
    e193_review = read_json(safe_path(metadata["e193_review"]))
    require(e193_review.get("validity") == "valid_observational_diagnostic"
            and e193_review.get("original_runs") == 1, "E193 review")
    with np.load(safe_path(metadata["source_native"]), allow_pickle=False) as source:
        native = {key: source[key] for key in source.files}
    with np.load(safe_path(metadata["encoder"]), allow_pickle=False) as source:
        encoder = {key: source[key] for key in source.files}
    return layout, trace, native, encoder, provenance


def validate_alignment(metadata: dict, layout: dict, trace: dict,
                       native: dict, encoder: dict, saved: dict) -> None:
    headers = metadata["headers"]
    for key, header in headers.items():
        require(key in native and native[key].shape == tuple(header["shape"])
                and str(native[key].dtype) == header["dtype"], "native header: " + key)
    require(set(FLAGS) <= native.keys(), "native completion flags missing")
    for key in FLAGS:
        require(np.all(native[key]), "incomplete native source: " + key)
    require(np.array_equal(native["native_index"], np.arange(NATIVE_ROWS)), "native_index")
    require(np.array_equal(native["control_tick"],
                           np.repeat(np.arange(201, 401), STEPS_PER_CONTROL)),
            "control_tick repetition")
    require(np.isfinite(native["time_s"]).all() and
            np.all(np.diff(native["time_s"]) > 0), "native clock")
    require(np.allclose(native["time_s"], 1.0 +
                        (np.arange(NATIVE_ROWS) + 1) * 0.0001,
                        rtol=0, atol=1e-11), "native clock grid")
    for key in ("qpos", "qvel", "ctrl"):
        require(np.isfinite(native[key]).all(), "nonfinite native " + key)
    rows = trace.get("samples")
    require(isinstance(rows, list) and len(rows) == 601, "trace count")
    require(saved["control_tick"] == 200 and saved["body"]["modelHash"]
            == metadata["body_hash"], "checkpoint identity")
    require(rows[200]["control_tick"] == 200 and rows[200]["time_s"] == 1.0,
            "checkpoint trace anchor")
    state = np.asarray(saved["body"]["state"], dtype=np.float64)
    require(np.isfinite(state).all(), "checkpoint state")
    # mjSTATE_INTEGRATION begins with time, qpos, qvel, act, ctrl. The exact
    # qpos/qvel/ctrl anchor is checked after body construction using MjData.
    for tick in range(201, 401):
        row = rows[tick]
        require(row["control_tick"] == tick and row["time_s"] == tick * 0.005,
                "trace clock at " + str(tick))
        index = (tick - 200) * STEPS_PER_CONTROL - 1
        require(abs(native["time_s"][index] - row["time_s"]) <= 1e-11,
                "native boundary clock")
        for key in ("qpos", "qvel", "ctrl"):
            require(np.array_equal(native[key][index], np.asarray(row[key])),
                    f"native/trace {key} tick {tick}")
    require(encoder["control_tick"].shape == (600,) and
            np.array_equal(encoder["control_tick"][200:401],
                           np.arange(200, 401)), "encoder ticks")
    for key in ("complete", "flexion_rad_available", "flexion_velocity_rad_s_available"):
        require(np.all(encoder[key][200:401]), "encoder completion: " + key)
    for key in ("flexion_rad", "flexion_velocity_rad_s"):
        require(encoder[key].shape == (600, 6) and
                np.isfinite(encoder[key][200:401]).all(), "encoder values: " + key)
    require(layout["nq"] == native["qpos"].shape[1]
            and layout["nv"] == native["qvel"].shape[1]
            and layout["nu"] == native["ctrl"].shape[1], "layout dimensions")


def _rms(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.sqrt(np.mean(np.square(values), axis=axis))


@contextmanager
def protect_actor(body):
    """Check the complete integration vector on success and observer failure."""
    mj = body.mj
    typ = mj.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mj.mj_stateSize(body.m, typ))
    mj.mj_getState(body.m, body.d, before, typ)
    try:
        yield
    finally:
        after = np.empty_like(before)
        mj.mj_getState(body.m, body.d, after, typ)
        if not np.array_equal(before, after):
            primary = sys.exc_info()[1]
            if primary is None:
                raise ValueError("actor integration state changed")
            primary.add_note("Actor integration state changed during original observation failure")


def summarize(time_s: np.ndarray, flexion: np.ndarray, velocity: np.ndarray) -> tuple[dict, dict]:
    require(time_s.shape == (ROWS,) and flexion.shape == velocity.shape == (ROWS, 6),
            "complete sample shapes")
    require(np.isfinite(time_s).all() and np.isfinite(flexion).all()
            and np.isfinite(velocity).all() and np.all(np.diff(time_s) > 0),
            "finite ordered samples")
    interval = np.diff(time_s).reshape(CONTROL_INTERVALS, STEPS_PER_CONTROL)
    v0 = velocity[:-1].reshape(CONTROL_INTERVALS, STEPS_PER_CONTROL, 6)
    v1 = velocity[1:].reshape(CONTROL_INTERVALS, STEPS_PER_CONTROL, 6)
    control = np.arange(0, ROWS, STEPS_PER_CONTROL)
    change = np.diff(flexion[control], axis=0)
    native_trapezoid = np.sum((v0 + v1) * 0.5 * interval[:, :, None], axis=1)
    native_right = np.sum(v1 * interval[:, :, None], axis=1)
    boundary_trapezoid = (velocity[control[:-1]] + velocity[control[1:]]) * 0.5 * np.diff(time_s[control])[:, None]
    arrays = {"actual_change_rad": change,
              "native_trapezoid_rad": native_trapezoid,
              "native_right_rad": native_right,
              "boundary_trapezoid_rad": boundary_trapezoid,
              "native_trapezoid_error_rad": native_trapezoid - change,
              "native_right_error_rad": native_right - change,
              "boundary_trapezoid_error_rad": boundary_trapezoid - change}
    legs = {}
    supportive = 0
    for index, leg in enumerate(LEGS):
        displacement = float(_rms(change[:, index]))
        nt_error = float(_rms(arrays["native_trapezoid_error_rad"][:, index]))
        nr_error = float(_rms(arrays["native_right_error_rad"][:, index]))
        bt_error = float(_rms(arrays["boundary_trapezoid_error_rad"][:, index]))
        nt_rel = nt_error / displacement if displacement else None
        nr_rel = nr_error / displacement if displacement else None
        bt_rel = bt_error / displacement if displacement else None
        ratio = bt_error / nt_error if nt_error else (float("inf") if bt_error else None)
        supported = bool(nt_rel is not None and nt_rel <= 0.1 and
                         ratio is not None and ratio >= 4 and bt_rel >= 0.25)
        supportive += supported
        legs[leg] = {
            "stationary": displacement == 0,
            "actual_change_rms_rad": displacement,
            "native_trapezoid_error_rms_rad": nt_error,
            "native_right_error_rms_rad": nr_error,
            "boundary_trapezoid_error_rms_rad": bt_error,
            "native_trapezoid_relative_error": nt_rel,
            "native_right_relative_error": nr_rel,
            "boundary_trapezoid_relative_error": bt_rel,
            "boundary_to_native_error_ratio": ratio if ratio is None or np.isfinite(ratio) else None,
            "native_velocity_rms_rad_s": float(_rms(velocity[:, index])),
            "native_velocity_peak_abs_rad_s": float(np.max(np.abs(velocity[:, index]))),
            "boundary_velocity_rms_rad_s": float(_rms(velocity[control, index])),
            "boundary_velocity_peak_abs_rad_s": float(np.max(np.abs(velocity[control, index]))),
            "supports_boundary_information_loss": supported,
        }
    global_metrics = {name + "_rms_rad": float(_rms(values)) for name, values in arrays.items()}
    verdict = ("supports_boundary_information_loss_in_saved_trajectory"
               if supportive >= 5 else "unresolved_or_weak_boundary_information_loss")
    return arrays, {"interval_count": CONTROL_INTERVALS, "legs": legs,
                    "global": global_metrics, "supporting_legs": supportive,
                    "verdict": verdict}


def collect(body, saved: dict, layout: dict, trace: dict, native: dict,
            encoder: dict, deadline: float, arrays: dict,
            counters: dict | None = None) -> None:
    mj = body.mj
    typ = mj.mjtState.mjSTATE_INTEGRATION
    actor_before = np.empty(mj.mj_stateSize(body.m, typ))
    mj.mj_getState(body.m, body.d, actor_before, typ)
    shadow = copy.copy(body)
    shadow.d = mj.MjData(body.m)
    if counters is not None:
        counters["shadow_mjdata"] += 1
    mj.mj_setState(body.m, shadow.d, actor_before, typ)
    mj.mj_forward(body.m, shadow.d)
    anchor = trace["samples"][200]
    require(abs(float(body.d.time - body.t0) - 1.0) <= 1e-12, "body anchor time")
    for key in ("qpos", "qvel", "ctrl"):
        require(np.array_equal(getattr(body.d, key), np.asarray(anchor[key])),
                "checkpoint/trace anchor " + key)
    require(np.array_equal(np.asarray(saved["body"]["state"]), actor_before),
            "checkpoint integration state")
    hinge_dofs = [int(group[5]) for group in layout["leg_dof_addresses"]]
    require(len(hinge_dofs) == 6, "six hinge dofs")
    for row_index in range(ROWS):
        require(time.perf_counter() < deadline, "internal wall deadline")
        if row_index:
            source_index = row_index - 1
            shadow.d.time = float(native["time_s"][source_index]) + body.t0
            shadow.d.qpos[:] = native["qpos"][source_index]
            shadow.d.qvel[:] = native["qvel"][source_index]
            shadow.d.ctrl[:] = native["ctrl"][source_index]
            mj.mj_forward(body.m, shadow.d)
            arrays["time_s"][row_index] = native["time_s"][source_index]
            arrays["source_index"][row_index] = source_index
        else:
            arrays["time_s"][row_index] = 1.0
            arrays["source_index"][row_index] = -1
        knee = shadow.knee_kinematics(include_direction=False)
        flexion = np.asarray(knee["flexion_rad"], dtype=np.float64)
        velocity = np.asarray(knee["velocity_rad_s"], dtype=np.float64)
        require(flexion.shape == velocity.shape == (6,) and
                np.isfinite(flexion).all() and np.isfinite(velocity).all(),
                "invalid knee kinematics row " + str(row_index))
        arrays["flexion_rad"][row_index] = flexion
        arrays["analytic_velocity_rad_s"][row_index] = velocity
        arrays["hinge_qvel_rad_s"][row_index] = shadow.d.qvel[hinge_dofs]
        if row_index % STEPS_PER_CONTROL == 0:
            tick = 200 + row_index // STEPS_PER_CONTROL
            encoder_index = tick
            require(np.allclose(flexion, encoder["flexion_rad"][encoder_index],
                                rtol=0, atol=GEOMETRY_ATOL),
                    "encoder angle boundary " + str(tick))
            require(np.allclose(velocity, encoder["flexion_velocity_rad_s"][encoder_index],
                                rtol=0, atol=GEOMETRY_ATOL),
                    "encoder velocity boundary " + str(tick))
        arrays["complete"][row_index] = True
    actor_after = np.empty_like(actor_before)
    mj.mj_getState(body.m, body.d, actor_after, typ)
    require(np.array_equal(actor_before, actor_after), "actor integration state changed")


def run(freeze_path: Path, review_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    deadline = started + INTERNAL_S
    metadata = verify_gate(freeze_path, review_path, output)
    layout, trace, native, encoder, _ = load_sources(metadata)
    saved = StateStore.load((ROOT / metadata["checkpoint"]).resolve())
    validate_alignment(metadata, layout, trace, native, encoder, saved)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be disabled")
    require(os.environ.get("OMP_NUM_THREADS") == "1"
            and os.environ.get("OPENBLAS_NUM_THREADS") == "1", "CPU/BLAS thread cap")
    import shutil
    require(shutil.disk_usage(output.parent).free >= MIN_FREE_BYTES, "free disk cap")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"time_s": np.full(ROWS, np.nan),
              "source_index": np.full(ROWS, -2, dtype=np.int64),
              "flexion_rad": np.full((ROWS, 6), np.nan),
              "analytic_velocity_rad_s": np.full((ROWS, 6), np.nan),
              "hinge_qvel_rad_s": np.full((ROWS, 6), np.nan),
              "complete": np.zeros(ROWS, dtype=bool),
              "error": np.full(ROWS, "", dtype="U512")}
    counters = {"body_constructor_attempts": 0, "body_constructor_completed": 0,
                "settling_physics_steps": 0, "scientific_physics_steps": 0,
                "body_models": 0, "shadow_mjdata": 0}
    body = None
    primary_error = None
    result = {"schema": SCHEMA + ".result.v1", "complete": False,
              "source_native_sha256": sha256(safe_path(metadata["source_native"])),
              "body_hash": metadata["body_hash"], "error": None,
              "rows_complete": 0, "counters": counters}
    try:
        body = construct_body_safely(saved, observation, counters)
        counters["body_models"] = 1
        require(body.model_hash == metadata["body_hash"], "constructed body hash")
        require(counters["settling_physics_steps"] == 2000, "constructor settling count")
        servo = np.concatenate((np.asarray(body.m.actuator_gainprm).ravel(),
                                np.asarray(body.m.actuator_biasprm).ravel(),
                                np.asarray(body.m.actuator_ctrlrange).ravel()))
        result["model"] = {"timestep_s": float(body.m.opt.timestep),
                           "integrator": int(body.m.opt.integrator),
                           "servo_profile": saved["parameters"].get("servo_profile", "tracking_all"),
                           "servo_sha256": hashlib.sha256(servo.tobytes()).hexdigest(),
                           "integration_state_sha256": hashlib.sha256(
                               np.asarray(saved["body"]["state"], dtype=np.float64).tobytes()).hexdigest()}
        original_step = body._step_physics

        def forbidden_step():
            raise RuntimeError("E194 forbids actor physics steps after construction")

        body._step_physics = forbidden_step
        try:
            with protect_actor(body):
                collect(body, saved, layout, trace, native, encoder, deadline, arrays,
                        counters)
        finally:
            body._step_physics = original_step
        intervals, summary = summarize(arrays["time_s"], arrays["flexion_rad"],
                                       arrays["analytic_velocity_rad_s"])
        arrays.update(intervals)
        result.update(summary)
        result["complete"] = True
    except Exception as exc:
        primary_error = exc
        first = int(np.count_nonzero(arrays["complete"]))
        arrays["error"][min(first, ROWS - 1)] = repr(exc)
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        raise
    finally:
        result["rows_complete"] = int(np.count_nonzero(arrays["complete"]))
        result["wall_s"] = time.perf_counter() - started
        cleanup_error = None
        if body is not None:
            try:
                body.close()
            except Exception as exc:  # noqa: BLE001
                result["cleanup_error"] = repr(exc)
                result["complete"] = False
                cleanup_error = exc
                if primary_error is None:
                    result["error"] = {"type": type(exc).__name__,
                                       "message": str(exc), "phase": "cleanup"}
        output_error = None
        try:
            np.savez_compressed(output / "native-knee-velocity.npz", **arrays)
            write_json(output / "result.json", result)
            require(sum(path.stat().st_size for path in output.iterdir()) <= MAX_OUTPUT_BYTES,
                    "output byte cap")
        except Exception as exc:  # noqa: BLE001
            output_error = exc
        if primary_error is not None:
            for secondary in (cleanup_error, output_error):
                if secondary is not None:
                    primary_error.add_note("Secondary E194 finalization failure: " + repr(secondary))
        elif cleanup_error is not None:
            if output_error is not None:
                cleanup_error.add_note("Secondary E194 output failure: " + repr(output_error))
            raise cleanup_error
        elif output_error is not None:
            raise output_error
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.freeze, args.review, args.output)), flush=True)
