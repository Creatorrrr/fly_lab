"""E212 bounded anatomical box capacity diagnostic on frozen E209 response data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fit_c_banc_m3_local_response import load_targets
from tools.verify_c_banc_m3_local_response import LEGS
from tools.verify_c_banc_m3_native_current import (
    configure_single_thread_runtime,
    file_sha,
    save_json,
)

FREEZE_SHA = "a64f0324550d963b72ee51828fdf9b637c3c2e08ddca8f4ae7d09dd6218c058b"
FREEZE_SCHEMA = "flylab.m3.e212.anatomical-capacity.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e212.anatomical-capacity.review.v1"
OWNED = ("tools/analyze_c_banc_m3_anatomical_capacity.py",
         "tests/test_c_banc_anatomical_capacity.py")
SHAPE = {"motors": 391, "free_motors": 343, "fixed_motors": 48,
    "free_muscles": 90, "all_muscles": 96, "samples": 2001,
    "selected_samples": 1923, "startup_rows": 99, "main_rows": 1824,
    "legs": 6, "problems_max": 11538}
SOLVER = {"name": "scipy.optimize.lsq_linear", "method": "bvls", "tol": 1e-10,
    "max_iter": 100, "bounds": [0.0, 1.0],
    "order": "leg-major LF LM LH RF RM RH then frame1..1923",
    "parameter": "independent per-frame muscle activation fraction",
    "zero_cap": "exclude exact zero only; no weak positive pruning"}
CERTIFICATE = {"minorant": "f-gradient.dot(v)+minimum(gradient,0).sum()",
    "lower_nonnegative": True, "upper_slack_tolerance": 1e-10,
    "gap_tolerance_relative": 1e-8}
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "solver_calls_max": 11538, "retries": 0, "internal_s": 60,
    "external_s": 80, "supervisor_s": 90,
    "resident_bytes": 2147483648, "output_bytes": 67108864,
    "minimum_free_bytes": 4294967296}
FLUSH_EVERY = 32
FLOAT_FIELDS = ("fraction", "activation", "gradient", "q_feasible",
                "residual", "f", "minorant", "lower", "upper", "gap",
                "optimality", "wall_s")
INT_FIELDS = ("status", "nit", "active_count")
BOOL_FIELDS = ("completed", "called", "certified")


def repository_path(relative, *, file=True):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("E212 repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or "." in name.parts or ".." in name.parts:
        raise ValueError("Unsafe E212 repository path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError("Missing or escaped E212 path")
    if file and not path.is_file():
        raise ValueError("E212 frozen file required")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha, *, elapsed=None):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E212 frozen SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E212 source review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("design") != "docs/BANC_M3_E212_ANATOMICAL_CAPACITY_DESIGN_20260924.md"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("shape") != SHAPE or freeze.get("solver") != SOLVER
            or freeze.get("certificate") != CERTIFICATE
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("new_cuda") != 0 or freeze.get("new_neural") != 0
            or freeze.get("new_physics") != 0
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False
            or freeze.get("whole_goal_complete") is not False):
        raise ValueError("E212 frozen scientific/resource contract changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E212 design SHA changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256", {})
    if not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != expected
            for name, expected in extra.items()):
        raise ValueError("E212 additional reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E212 parent source review does not authorize these sources")
    for label, count in (("input_sha256", 186),
                         ("reused_source_sha256", 23),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("E212 frozen hash cardinality changed")
        for relative, expected in hashes.items():
            if file_sha(repository_path(relative)) != expected:
                raise ValueError("E212 frozen byte changed: " + relative)
            if elapsed is not None and elapsed() >= RESOURCE["internal_s"]:
                raise TimeoutError("E212 preflight exceeded internal deadline")
    paths = freeze.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("E212 frozen path map absent")
    return freeze, review, {key: repository_path(value, file=key != "start")
                            for key, value in paths.items()}


def sum_group_caps(D, membership, free, free_muscles):
    """Pure group cap aggregation; D can be a small generated frame array."""
    D = np.asarray(D, np.float64)
    if (D.ndim != 3 or D.shape[1:] != (len(membership), 16)
            or not np.isfinite(D).all() or np.any(D < 0)):
        raise ValueError("E212 finite nonnegative 16-basis D required")
    U = np.zeros((len(D), len(free_muscles)), np.float64)
    muscle_to_column = {int(muscle): j for j, muscle in enumerate(free_muscles)}
    for motor in free:
        U[:, muscle_to_column[int(membership[motor])]] += D[:, motor].sum(axis=1)
    return U


def anatomical_leg_columns(free_muscles, groups, moment, *, expected_muscles=90):
    """Validate original whole-muscle leg ownership and no cross-leg moment."""
    if len(groups) != 96 or moment.shape != (42, 96):
        raise ValueError("E212 original muscle anatomy changed")
    leg_columns = []
    for leg_index, leg in enumerate(LEGS):
        for muscle, group in enumerate(groups):
            if group["leg"] != leg:
                continue
            other = np.r_[0:7 * leg_index, 7 * (leg_index + 1):42]
            if np.any(moment[other, muscle] != 0):
                raise ValueError("E212 moment crosses anatomical leg")
        own = np.asarray([j for j, muscle in enumerate(free_muscles)
                          if groups[int(muscle)]["leg"] == leg], np.int64)
        if len(own) == 0:
            raise ValueError("E212 leg has no free anatomical muscle")
        leg_columns.append(own)
    if (len(np.unique(np.concatenate(leg_columns))) != expected_muscles
            or sum(len(v) for v in leg_columns) != expected_muscles):
        raise ValueError("E212 free muscles not owned by exactly one leg")
    return leg_columns


def group_caps(D, membership, free, groups, moment):
    """Sum nonnegative E209 basis widths into 90 anatomical muscle caps."""
    D = np.asarray(D, np.float64)
    membership = np.asarray(membership, np.int64)
    free = np.asarray(free, np.int64)
    moment = np.asarray(moment, np.float64)
    if (D.shape != (2001, 391, 16) or membership.shape != (391,)
            or free.shape != (343,) or len(groups) != 96
            or moment.shape != (42, 96)
            or not np.array_equal(np.sort(free), free)
            or np.any((membership < 0) | (membership >= 96))
            or not np.isfinite(D).all() or np.any(D < 0)
            or not np.isfinite(moment).all()):
        raise ValueError("E212 original nonnegative D/anatomy changed")
    free_muscles = np.unique(membership[free])
    if len(free_muscles) != 90:
        raise ValueError("E212 expected exactly 90 free muscles")
    leg_columns = anatomical_leg_columns(free_muscles, groups, moment)
    U = sum_group_caps(D[1:1924], membership, free, free_muscles)
    if not np.isfinite(U).all() or np.any(U < 0):
        raise ValueError("E212 nonfinite or negative muscle cap")
    return U, free_muscles, leg_columns


def certificate(C, y, v):
    """Convex affine minorant on the closed [0,1]^n box."""
    C = np.asarray(C, np.float64)
    y = np.asarray(y, np.float64)
    v = np.asarray(v, np.float64)
    if (C.ndim != 2 or y.shape != (C.shape[0],)
            or v.shape != (C.shape[1],)
            or not np.isfinite(C).all() or not np.isfinite(y).all()
            or not np.isfinite(v).all()
            or np.any(v < 0) or np.any(v > 1)):
        raise ValueError("E212 strict feasible box certificate input required")
    residual = C @ v - y
    f = float(residual @ residual)
    gradient = 2 * C.T @ residual
    minorant = float(f - gradient @ v + np.minimum(gradient, 0).sum())
    lower = max(0., minorant)
    gap = f - lower
    certified = bool(lower <= f + 1e-10 and gap <= 1e-8 * max(1., f))
    return {"v": v.copy(), "residual": residual, "gradient": gradient,
        "f": f, "minorant": minorant, "lower": lower, "upper": f,
        "gap": gap, "certified": certified}


def solve_frame(C, y, *, active=None, solver=lsq_linear, raw_sink=None):
    """One frame: exact-zero columns excluded, no pruning or box clipping."""
    C = np.asarray(C, np.float64)
    y = np.asarray(y, np.float64)
    if (C.ndim != 2 or C.shape[0] != 7 or y.shape != (7,)
            or not np.isfinite(C).all() or not np.isfinite(y).all()):
        raise ValueError("E212 finite seven-joint frame required")
    if active is None:
        active = np.any(C != 0, axis=0)
    active = np.asarray(active, bool)
    if active.shape != (C.shape[1],):
        raise ValueError("E212 active cap mask shape changed")
    positive = np.flatnonzero(active)
    v = np.zeros(C.shape[1], np.float64)
    if len(positive) == 0:
        raw = certificate(C, y, v)
        raw.update(status=-2, optimality=0., nit=0, called=False,
                   active_count=0, wall_s=0.)
        return raw
    started = time.perf_counter()
    solved = solver(C[:, positive], y, method="bvls", bounds=(0., 1.),
                    tol=1e-10, max_iter=100)
    wall = time.perf_counter() - started
    raw_v = np.asarray(solved.x, np.float64)
    if raw_sink is not None:
        raw_sink(raw_v, solved, wall, positive)
    if raw_v.shape != (len(positive),):
        raise ValueError("E212 BVLS raw fraction shape changed")
    v[positive] = raw_v
    raw = certificate(C, y, v)
    raw.update(status=int(solved.status), optimality=float(solved.optimality),
               nit=int(solved.nit), called=True, active_count=len(positive),
               wall_s=wall)
    if not np.isfinite(raw["optimality"]) or raw["nit"] < 0:
        raise ValueError("E212 BVLS diagnostic nonfinite")
    if raw["status"] not in (1, 2, 3):
        raw["certified"] = False
    return raw


def create_raw_store(out, U, q0, targets):
    """NaN/sentinel files make every unfinished leg-frame distinguishable."""
    out = Path(out)
    out.mkdir(parents=True)
    shapes = {"U": (1923, 90), "fraction": (1923, 90),
        "activation": (1923, 90), "gradient": (1923, 90),
        "q_feasible": (1923, 42), "residual": (1923, 42),
        "f": (6, 1923), "minorant": (6, 1923), "lower": (6, 1923),
        "upper": (6, 1923), "gap": (6, 1923),
        "optimality": (6, 1923), "wall_s": (6, 1923),
        "status": (6, 1923), "nit": (6, 1923), "active_count": (6, 1923),
        "completed": (6, 1923), "called": (6, 1923),
        "certified": (6, 1923)}
    arrays = {}
    for name, shape in shapes.items():
        dtype = np.float64 if name in FLOAT_FIELDS or name == "U" else (
            np.int32 if name in INT_FIELDS else np.bool_)
        value = open_memmap(out / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        value[:] = np.nan if np.issubdtype(dtype, np.floating) else (
            -999 if np.issubdtype(dtype, np.integer) else False)
        arrays[name] = value
    arrays["U"][:] = U
    np.save(out / "q_intercept.npy", q0[1:1924])
    np.save(out / "target.npy", targets)
    np.save(out / "frame_indices.npy", np.arange(1, 1924, dtype=np.int64))
    save_json(out / "layout.json", {"schema": "flylab.m3.e212.raw-layout.v1",
        "shape": {name: list(shape) for name, shape in shapes.items()},
        "indexing": "leg-major status/cost; frame axis row1..1923; muscle axis free_muscles",
        "unfinished_float": "NaN", "unfinished_integer": -999,
        "completed_mask_is_authoritative": True})
    flush_raw(arrays)
    return arrays


def flush_raw(arrays):
    for value in arrays.values():
        value.flush()


def write_frame(arrays, leg_index, row_index, muscle_columns, cap, record):
    """Publish a complete in-memory frame then mark the durable completion bit."""
    fraction = record["v"]
    activation = cap * fraction
    arrays["fraction"][row_index, muscle_columns] = fraction
    arrays["activation"][row_index, muscle_columns] = activation
    arrays["gradient"][row_index, muscle_columns] = record["gradient"]
    for key in ("f", "minorant", "lower", "upper", "gap", "optimality",
                "wall_s", "status", "nit", "active_count", "called", "certified"):
        arrays[key][leg_index, row_index] = record[key]


def retain_solver_return(arrays, out, ledger, leg_index, frame, columns,
                         raw_v, solved, wall, positive):
    """Keep solver bytes before box, status, or certificate validation."""
    if raw_v.shape == (len(positive),):
        arrays["fraction"][frame, columns[positive]] = raw_v
    else:
        np.savez(Path(out) / "malformed-solver-return.npz",
                 x=raw_v, leg_index=leg_index, frame=frame + 1)
    arrays["status"][leg_index, frame] = int(solved.status)
    arrays["optimality"][leg_index, frame] = float(solved.optimality)
    arrays["nit"][leg_index, frame] = int(solved.nit)
    arrays["active_count"][leg_index, frame] = len(positive)
    arrays["wall_s"][leg_index, frame] = wall
    ledger["solver_returned_calls"] += 1
    ledger["solver_wall_s_observed"] += wall


def summarize_complete(arrays, target, baseline):
    """Only summarize a fully persisted 6x1923 diagnostic."""
    if not np.all(arrays["completed"]):
        raise ValueError("E212 incomplete frames cannot be summarized")
    all_certified = bool(np.all(arrays["certified"]))
    lower = np.asarray(arrays["lower"])
    upper = np.asarray(arrays["upper"])
    residual = np.asarray(arrays["residual"])
    q = np.asarray(arrays["q_feasible"])
    if not all(np.isfinite(v).all() for v in (lower, upper, residual, q)):
        raise ValueError("E212 complete raw contains nonfinite values")
    main_lower = float(np.sqrt(np.sum(lower[:, 99:]) / (1824 * 42)))
    main_upper = float(np.sqrt(np.sum(upper[:, 99:]) / (1824 * 42)))
    startup_lower = float(np.sqrt(np.sum(lower[:, :99]) / (99 * 42)))
    startup_upper = float(np.sqrt(np.sum(upper[:, :99]) / (99 * 42)))
    per_leg = []
    for i, leg in enumerate(LEGS):
        per_leg.append({"leg": leg,
            "main_lower_rms": float(np.sqrt(np.sum(lower[i, 99:]) / (1824 * 7))),
            "main_feasible_upper_rms":
                float(np.sqrt(np.sum(upper[i, 99:]) / (1824 * 7))),
            "startup_lower_rms":
                float(np.sqrt(np.sum(lower[i, :99]) / (99 * 7))),
            "startup_feasible_upper_rms":
                float(np.sqrt(np.sum(upper[i, :99]) / (99 * 7))),
            "startup_feasible_max":
                float(np.max(np.abs(residual[:99, 7*i:7*(i+1)])))})
    main_q = q[99:]
    ev = (main_q[target["pair_end"] - 100]
          - main_q[target["pair_start"] - 100]) / .005 - target["velocity"]
    velocity_rms = float(np.sqrt(np.mean(ev * ev)))
    return {"main_lower_rms": main_lower,
        "main_feasible_upper_rms": main_upper,
        "startup_lower_rms": startup_lower,
        "startup_feasible_upper_rms": startup_upper,
        "startup_feasible_max": float(np.max(np.abs(residual[:99]))),
        "feasible_posture_velocity_rms_descriptive": velocity_rms,
        "legs": per_leg,
        "main_lower_above_original_20_percent":
            bool(main_lower > .8 * .056686385349786386),
        "main_feasible_upper_within_original_20_percent":
            bool(main_upper <= .8 * .056686385349786386),
        "startup_lower_above_original_rms": bool(startup_lower > .018),
        "startup_feasible_upper_within_original_rms":
            bool(startup_upper <= .018),
        "startup_feasible_max_within_original": bool(
            np.max(np.abs(residual[:99])) <= .04),
        "all_frame_gaps_certified": all_certified,
        "baseline": baseline,
        "interpretation": "independent framewise anatomical box relaxation only"}


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    manifest = {"schema": "flylab.m3.e212.output-manifest.v1",
        "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    save_json(Path(out) / "output-manifest.json", manifest)
    return manifest


def _save_attempt(errors, stage, operation):
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - preserve original failure
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        return None


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E212 output exists: " + str(out))
    out.mkdir(parents=True)
    freeze = None
    primary = None
    errors = []
    arrays = None
    ledger = {"schema": "flylab.m3.e212.ledger.v1",
        "solver_calls": 0, "solver_returned_calls": 0,
        "solver_wall_s_observed": 0.,
        "completed_frames": 0, "certified_frames": 0,
        "zero_cap_frames": 0, "last_persisted_solver_calls": 0,
        "last_persisted_completed_frames": 0,
        "full_banc_calls": 0, "cuda_calls": 0, "physics_steps": 0,
        "retries": 0, "current_leg": None, "current_frame": None}
    result = {"schema": "flylab.m3.e212.anatomical-capacity.result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "anatomical_relaxation_only": True}

    def elapsed():
        return time.perf_counter() - began

    def persist_progress():
        if arrays is not None:
            flush_raw(arrays)
        ledger["last_persisted_solver_calls"] = ledger["solver_calls"]
        ledger["last_persisted_completed_frames"] = ledger["completed_frames"]
        save_json(out / "ledger.json", ledger)

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha, elapsed=elapsed)
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E212 requires 4 GiB free disk")
        save_json(out / "provenance.json", {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        save_json(out / "threading.json", configure_single_thread_runtime())
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        from flylab.c.storage import StateStore
        with threadpool_limits(limits=1):
            start = StateStore.load(paths["start"])
            trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
            target = load_targets(paths["targets"], trace)
            with (np.load(paths["affine"], allow_pickle=False) as affine,
                  np.load(paths["mask"], allow_pickle=False) as mask):
                D = affine["D"].copy()
                q0 = affine["q_intercept"].copy()
                free = mask["free_columns"].astype(np.int64)
                fixed = mask["fixed_columns"].astype(np.int64)
                if (q0.shape != (2001, 42) or fixed.shape != (48,)
                        or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                        or not np.isfinite(q0).all()
                        or np.any(D[:, fixed] != 0)
                        or not np.array_equal(mask["membership"],
                            start["adapter"]["parameters"]["membership"])):
                    raise ValueError("E212 frozen affine/mask changed")
            membership = np.asarray(start["adapter"]["parameters"]["membership"], np.int64)
            moment = np.asarray(start["adapter"]["calibration"]["moment"], np.float64)
            groups = start["adapter"]["groups"]
            U, free_muscles, leg_columns = group_caps(
                D, membership, free, groups, moment)
            del D
            frame_targets = np.vstack((target["startup_command"], target["target"]))
            if (frame_targets.shape != (1923, 42)
                    or not np.array_equal(target["indices"], np.arange(100, 1924))):
                raise ValueError("E212 fixed startup/main frame order changed")
            padded_columns = np.full((6, max(map(len, leg_columns))), -1, np.int64)
            for i, columns in enumerate(leg_columns):
                padded_columns[i, :len(columns)] = columns
            np.savez(out / "anatomy.npz", membership=membership, moment=moment,
                free_muscles=free_muscles, leg_group_columns=padded_columns,
                leg_group_counts=np.asarray([len(v) for v in leg_columns], np.int64),
                free_motor_columns=free, fixed_motor_columns=fixed)
            arrays = create_raw_store(out / "capacity-raw", U, q0, frame_targets)
            persist_progress()
            for leg_index, leg in enumerate(LEGS):
                columns = leg_columns[leg_index]
                muscles = free_muscles[columns]
                moment_leg = .02 * moment[7*leg_index:7*(leg_index+1), muscles]
                for frame in range(1923):
                    if (elapsed() >= RESOURCE["internal_s"]
                            or ledger["solver_calls"] >= RESOURCE["solver_calls_max"]):
                        raise TimeoutError("E212 internal time or BVLS call cap reached")
                    ledger["current_leg"] = leg
                    ledger["current_frame"] = frame + 1
                    cap = U[frame, columns]
                    C = moment_leg * cap[None, :]
                    y = frame_targets[frame, 7*leg_index:7*(leg_index+1)] - (
                        q0[frame + 1, 7*leg_index:7*(leg_index+1)])
                    will_call = bool(np.any(cap > 0))
                    if will_call:
                        ledger["solver_calls"] += 1
                        arrays["called"][leg_index, frame] = True

                    record = solve_frame(C, y, active=cap > 0,
                        raw_sink=partial(retain_solver_return, arrays, out, ledger,
                                         leg_index, frame, columns))
                    if will_call != record["called"]:
                        raise ValueError("E212 exact zero-cap call accounting changed")
                    if not record["called"]:
                        ledger["zero_cap_frames"] += 1
                    write_frame(arrays, leg_index, frame, columns, cap, record)
                    q_feasible = (frame_targets[frame, 7*leg_index:7*(leg_index+1)]
                                  + record["residual"])
                    arrays["q_feasible"][frame, 7*leg_index:7*(leg_index+1)] = q_feasible
                    arrays["residual"][frame, 7*leg_index:7*(leg_index+1)] = (
                        record["residual"])
                    if record["status"] not in (-2, 1, 2, 3):
                        raise RuntimeError("E212 BVLS incomplete or failed status")
                    arrays["completed"][leg_index, frame] = True
                    ledger["completed_frames"] += 1
                    ledger["certified_frames"] += int(record["certified"])
                    if ledger["completed_frames"] % FLUSH_EVERY == 0:
                        persist_progress()
                persist_progress()
            persist_progress()
            if (ledger["completed_frames"] != 11538
                    or ledger["solver_calls"] > 11538):
                raise ValueError("E212 full coverage/call count changed")
            metrics = summarize_complete(arrays, target, target["baseline"])
            save_json(out / "metrics.json", metrics)
            result.update(validity=("valid_certified_anatomical_relaxation"
                if ledger["certified_frames"] == 11538
                else "valid_complete_uncertified_diagnostic"), metrics=metrics)
    except BaseException as error:  # noqa: BLE001 - preserve partial raw and primary
        primary = error
        errors.append({"stage": "execution", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    _save_attempt(errors, "raw-flush", lambda: flush_raw(arrays) if arrays else None)
    ledger["process_wall_s"] = elapsed()
    ledger["call_count_exact_if_clean_exit"] = primary is None
    _save_attempt(errors, "ledger-final", lambda: save_json(out / "ledger.json", ledger))
    result.update(process_wall_s=elapsed(), raw_errors=errors,
        solver_calls=ledger["solver_calls"],
        solver_returned_calls=ledger["solver_returned_calls"],
        solver_wall_s_observed=ledger["solver_wall_s_observed"],
        completed_frames=ledger["completed_frames"],
        certified_frames=ledger["certified_frames"],
        zero_cap_frames=ledger["zero_cap_frames"],
        prior_cumulative_s=(freeze or {}).get("prior_cumulative_s"),
        conservative_cumulative_s=(freeze or {}).get("prior_cumulative_s", 0) + elapsed(),
        retries=0)
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    _save_attempt(errors, "execution-errors.json",
                  lambda: save_json(out / "execution-errors.json", errors))
    _save_attempt(errors, "result-preliminary", lambda: save_json(
        out / "result.json", dict(result, validity="incomplete_or_invalid",
                                  finalization_pending=True)))
    manifest = _save_attempt(errors, "output-manifest.json",
                             lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > RESOURCE["output_bytes"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E212 output manifest missing or over 64 MiB"
        if primary is None:
            primary = RuntimeError(result["resource_failure"])
    if elapsed() >= RESOURCE["internal_s"]:
        result["validity"] = "incomplete_or_invalid"
        result["resource_failure"] = "E212 hard internal deadline exceeded"
        if primary is None:
            primary = TimeoutError(result["resource_failure"])
    if errors:
        result["validity"] = "incomplete_or_invalid"
    result["finalization_pending"] = False
    try:
        save_json(out / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - preserve primary
        errors.append({"stage": "result-final", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if primary is None:
            primary = error
    if errors:
        _save_attempt(errors, "publication-failure.json",
                      lambda: save_json(out / "publication-failure.json", errors))
        if primary is None:
            primary = RuntimeError("E212 raw/publication failure")
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
