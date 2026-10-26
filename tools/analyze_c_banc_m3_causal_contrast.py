"""E244 saved-only contrast of each MN's present and causal input memories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_c_banc_m3_matched_generalization import anatomy_weights

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
RESERVED_OUT = (BASE / "e244-causal-contrast").resolve()
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
FREEZE_SHA = "d3be0cf6d21a282760b07e75bd0215a0b3c2a0ebad66674593f5472793a23961"
FREEZE_SCHEMA = "flylab.m3.e244.causal-contrast.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e244.causal-contrast.review.v1"
OWNED = ("tools/analyze_c_banc_m3_causal_contrast.py",
         "tests/test_c_banc_causal_contrast.py")
RESOURCE = {"cpu_threads": 1, "gpu_count": 0, "original_attempts": 1,
    "retries": 0, "fit_calls": 0, "solver_calls": 0,
    "generated_solver_calls": 0, "generated_cuda_calls": 0,
    "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
    "internal_s": 60, "finalization_reserve_s": 10, "external_s": 75,
    "supervisor_s": 85, "resident_bytes": 3221225472,
    "output_bytes": 134217728, "minimum_free_bytes": 2147483648}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def repository_path(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("E244 repository-relative slash path required")
    relative = Path(name)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ValueError("unsafe E244 frozen path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("E244 frozen path missing or outside repository")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E244 freeze SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E244 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("shape") != {"samples": 2001, "motors": 391,
                "free_motors": 343, "fixed_motors": 48, "muscles": 96,
                "joints": 42, "legs": 6, "descriptor_width": 9,
                "contrast_width": 6}
            or freeze.get("split") != {"train_first": 1, "train_last": 1200,
                "steady_first": 100, "steady_last": 1200,
                "observed_tail_first": 1201, "observed_tail_last": 2000,
                "tail_independent_validation": False, "F6_exposed": False}
            or freeze.get("representation", {}).get("contrast") != "concat(x-fast,fast-slow)"
            or freeze.get("representation", {}).get("neighbor_metric") != "euclidean"
            or freeze.get("representation", {}).get("std_floor") != 1e-4
            or freeze.get("representation", {}).get("exclusion_inclusive_controls") != 20
            or freeze.get("representation", {}).get("batch_queries_max") != 128
            or freeze.get("hypothesis_criteria", {}).get("global_q_rms_fraction") != .8
            or freeze.get("hypothesis_criteria", {}).get("each_leg_q_rms_fraction") != 1.
            or any(freeze.get(name) is not False for name in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E244 frozen scientific/resource contract changed")
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own):
        raise ValueError("E244 parent source review invalid")
    if freeze.get("packages") != {name: version(name) for name in freeze["packages"]}:
        raise ValueError("E244 frozen package versions changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E244 design changed")
    for hashes in (freeze["input_sha256"], freeze["reused_source_sha256"],
                   freeze["preservation_sha256"],
                   review.get("additional_reused_source_sha256"),
                   review.get("additional_input_sha256")):
        if not isinstance(hashes, dict):
            raise TypeError("E244 review/freeze hash map missing")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E244 pinned byte changed: " + name)
    paths = {role: repository_path(name) for role, name in freeze["paths"].items()}
    if set(paths) != {"parameters", "descriptor", "targets", "groups",
                      "baseline_raw", "baseline_result", "baseline_ledger",
                      "source_proof", "source_supervisor"}:
        raise ValueError("E244 frozen role set changed")
    return freeze, review, paths


def load_inputs(paths, *, progress=None):
    values = {}
    selection = {
        "parameters": ("free_columns", "fixed_columns", "membership",
                       "pool", "moment", "cap", "ltm"),
        "descriptor": ("descriptor",),
        "targets": ("target_command",),
        "baseline_raw": ("free_columns", "sensitivity_weight",
                         "nearest_train_control", "nearest_train_distance",
                         "neighbor_teacher_q_difference"),
    }
    for role, keys in selection.items():
        with np.load(paths[role], allow_pickle=False) as saved:
            values[role] = {key: saved[key].copy() for key in keys}
        if progress:
            progress(role)
    for role in ("groups", "baseline_result", "baseline_ledger", "source_proof",
                 "source_supervisor"):
        value = json.loads(paths[role].read_text(encoding="utf-8"))
        values[role] = value["groups"] if role == "groups" else value
        if progress:
            progress(role)
    return values


def rms(value):
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def q_stats(values):
    return {"q_rms_rad": rms(values),
            "q_max_abs_rad": float(np.max(np.abs(values)))}


def distance_stats(values):
    return {"mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def contrast_and_scale(descriptor, *, train_last=1200):
    """Six present/history contrasts; std uses only controls 1..train_last."""
    descriptor = np.asarray(descriptor, np.float64)
    if (descriptor.ndim != 3 or descriptor.shape[2] != 9
            or len(descriptor) <= train_last or train_last < 22
            or not np.isfinite(descriptor).all()):
        raise ValueError("E244 complete finite own nine-channel descriptor required")
    z = np.concatenate((descriptor[:, :, :3] - descriptor[:, :, 3:6],
                        descriptor[:, :, 3:6] - descriptor[:, :, 6:9]), axis=2)
    scale = np.maximum(np.std(z[1:train_last+1], axis=0, ddof=0), 1e-4)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("E244 training-only standard deviation invalid")
    return z, scale


def verify_baseline(baseline, baseline_result, target, free, weights,
                    *, train_last=1200):
    """Do not search old neighbors again; verify their anatomy and q identity."""
    n = len(target)
    indices = np.asarray(baseline["nearest_train_control"], np.int64)
    distances = np.asarray(baseline["nearest_train_distance"], np.float64)
    difference = np.asarray(baseline["neighbor_teacher_q_difference"], np.float64)
    if (not np.array_equal(baseline["free_columns"], free)
            or not np.allclose(baseline["sensitivity_weight"], weights,
                               atol=2e-10, rtol=2e-12)
            or indices.shape != (n, 6) or distances.shape != (n, 6)
            or difference.shape != (n, 42)
            or not np.array_equal(indices[0], np.full(6, -1))
            or not np.array_equal(distances[0], np.zeros(6))
            or not np.array_equal(difference[0], np.zeros(42))
            or np.any(indices[1:] < 1) or np.any(indices[1:] > train_last)
            or np.any(np.abs(indices[1:] - np.arange(1, n)[:, None]) <= 20)
            or not np.isfinite(distances).all() or np.any(distances < 0)
            or not np.isfinite(difference).all()):
        raise ValueError("E244 E243 baseline anatomy/neighbor identity changed")
    for i, leg in enumerate(LEGS):
        sl = slice(7*i, 7*(i+1))
        if not np.array_equal(difference[1:, sl], target[indices[1:, i], sl] - target[1:, sl]):
            raise ValueError("E244 E243 baseline teacher q provenance changed: " + leg)
        for name, lo, hi in (("train", 1, train_last+1),
                             ("steady", 100, train_last+1),
                             ("observed_tail", train_last+1, n)):
            stored = baseline_result["analysis"]["segments"][name][leg][
                "neighbor_teacher_q_difference_rms"]
            if not np.isclose(rms(difference[lo:hi, sl]), stored,
                              atol=2e-10, rtol=2e-12):
                raise ValueError("E244 E243 baseline segment RMS changed: " + leg)
    return indices, distances, difference


def analyze_arrays(parameters, descriptor_data, targets, groups, baseline_raw,
                   baseline_result, *, raw=None, progress=None, train_last=1200):
    """Pure saved-array comparison; caller-owned raw keeps completed prefixes."""
    raw = {} if raw is None else raw
    descriptor = np.asarray(descriptor_data["descriptor"], np.float64)
    target = np.asarray(targets["target_command"], np.float64)
    free = np.asarray(parameters["free_columns"], np.int64)
    n, motors = descriptor.shape[:2]
    if (descriptor.shape != (n, motors, 9) or n <= train_last
            or target.shape != (n, 42) or len(np.unique(free)) != len(free)
            or np.any((free < 0) | (free >= motors))
            or not np.isfinite(target).all()):
        raise ValueError("E244 original descriptor/teacher alignment invalid")
    weights = anatomy_weights(parameters, groups)
    z, scale = contrast_and_scale(descriptor, train_last=train_last)
    old_indices, old_distances, old_difference = verify_baseline(
        baseline_raw, baseline_result, target, free, weights, train_last=train_last)
    raw.update(scale=scale, free_columns=free.copy(),
        sensitivity_weight=weights.copy(),
        nearest_train_control=np.full((n, 6), -1, np.int64),
        nearest_train_distance=np.full((n, 6), np.nan),
        contrast_teacher_q_difference=np.full((n, 42), np.nan),
        baseline_teacher_q_difference=old_difference.copy(),
        baseline_nearest_train_control=old_indices.copy(),
        baseline_nearest_train_distance=old_distances.copy(),
        completed_neighbor_batches=np.zeros(6, np.int64),
        completed_query_rows=np.zeros(6, np.int64),
        completed_neighbor_legs=np.asarray(0, np.int64))
    raw["nearest_train_distance"][0] = 0
    raw["contrast_teacher_q_difference"][0] = 0
    train = np.arange(1, train_last+1, dtype=np.int64)
    for leg_index, leg in enumerate(LEGS):
        selected = np.flatnonzero(weights[leg_index] > 0)
        own = free[selected]
        factors = np.sqrt(weights[leg_index, selected])[:, None] / scale[own]
        feature = (z[:, own, :] * factors[None]).reshape(n, -1)
        reference = np.ascontiguousarray(feature[train])
        for first in range(1, n, 128):
            query = np.arange(first, min(first+128, n), dtype=np.int64)
            distances = cdist(np.ascontiguousarray(feature[query]), reference,
                              metric="euclidean")
            distances[np.abs(query[:, None] - train[None]) <= 20] = np.inf
            nearest = np.argmin(distances, axis=1)
            values = distances[np.arange(len(query)), nearest]
            if not np.isfinite(values).all():
                raise ValueError("E244 no admissible training neighbor: " + leg)
            chosen = train[nearest]
            raw["nearest_train_control"][query, leg_index] = chosen
            raw["nearest_train_distance"][query, leg_index] = values
            sl = slice(7*leg_index, 7*(leg_index+1))
            raw["contrast_teacher_q_difference"][query, sl] = (
                target[chosen, sl] - target[query, sl])
            raw["completed_neighbor_batches"][leg_index] += 1
            raw["completed_query_rows"][leg_index] += len(query)
            if progress:
                progress("neighbor_batch", leg_index, int(raw["completed_query_rows"][leg_index]))
        raw["completed_neighbor_legs"] = np.asarray(leg_index + 1, np.int64)
        if progress:
            progress("neighbor_leg", leg_index, n-1)

    if not all(np.isfinite(value).all() for value in raw.values()
               if isinstance(value, np.ndarray) and value.dtype.kind == "f"):
        raise ValueError("E244 completed diagnostic has nonfinite arrays")
    segments = (("train", 1, train_last+1), ("startup", 1, 100),
                ("steady", 100, train_last+1),
                ("observed_tail", train_last+1, n))
    report = {"schema": "flylab.m3.e244.causal-contrast.analysis.v1",
              "descriptive_only": True, "observed_tail_independent_validation": False,
              "distance_scale_comparable_across_representations": False,
              "segments": {}, "bins": [], "hypothesis": {}}
    for name, lo, hi in segments:
        section = {}
        for label, difference in (("baseline", old_difference),
                                  ("contrast", raw["contrast_teacher_q_difference"])):
            section[label] = {"global": q_stats(difference[lo:hi]), "legs": {}}
            for i, leg in enumerate(LEGS):
                section[label]["legs"][leg] = q_stats(
                    difference[lo:hi, 7*i:7*(i+1)])
        for label, distances in (("baseline", old_distances),
                                 ("contrast", raw["nearest_train_distance"])):
            section[label]["distance"] = {
                "global": distance_stats(distances[lo:hi]),
                "legs": {leg: distance_stats(distances[lo:hi, i])
                         for i, leg in enumerate(LEGS)}}
        report["segments"][name] = section
    for bin_index, first in enumerate(range(1, n, 200)):
        last = min(first+200, n)
        section = {"bin": bin_index+1, "first_control": first,
                   "last_control": last-1}
        for label, difference in (("baseline", old_difference),
                                  ("contrast", raw["contrast_teacher_q_difference"])):
            section[label] = {"global": q_stats(difference[first:last]), "legs": {}}
            for i, leg in enumerate(LEGS):
                section[label]["legs"][leg] = q_stats(
                    difference[first:last, 7*i:7*(i+1)])
        report["bins"].append(section)
    predicates = {}
    for name in ("steady", "observed_tail"):
        section = report["segments"][name]
        predicates[name + "/global"] = (section["contrast"]["global"]["q_rms_rad"]
            <= .8*section["baseline"]["global"]["q_rms_rad"])
        for leg in LEGS:
            predicates[name + "/" + leg] = (
                section["contrast"]["legs"][leg]["q_rms_rad"]
                <= section["baseline"]["legs"][leg]["q_rms_rad"])
    report["hypothesis"] = {"supported": bool(all(predicates.values())),
        "predicates": predicates,
        "failed": [name for name, passed in predicates.items() if not passed],
        "scope": "saved same-input teacher neighbor comparison only"}
    return report, raw


def validate_original_dimensions(inputs):
    if (inputs["descriptor"]["descriptor"].shape != (2001, 391, 9)
            or inputs["targets"]["target_command"].shape != (2001, 42)
            or inputs["parameters"]["free_columns"].shape != (343,)
            or inputs["parameters"]["fixed_columns"].shape != (48,)
            or inputs["baseline_raw"]["nearest_train_control"].shape != (2001, 6)):
        raise ValueError("E244 original frozen array dimensions changed")


def publish(out, result, raw, ledger, *, deadline=None, elapsed=None):
    """Atomic raw first; errors, ledger, manifest, then exact-byte result."""
    raw_path = out / "diagnostic-raw.npz"
    if raw and not raw_path.is_file():
        temporary = out / "diagnostic-raw.npz.part"
        first = None
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **raw)
            os.replace(temporary, raw_path)
        except BaseException as error:  # noqa: BLE001 - cleanup cannot mask first storage error
            first = error
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as error:  # noqa: BLE001 - retain both storage errors
            if first is None:
                first = error
            else:
                ledger.setdefault("raw_secondary_errors", []).append({
                    "stage": "raw_temporary_cleanup", "type": type(error).__name__,
                    "message": str(error)})
        if first is not None:
            raise first
    if elapsed:
        ledger["process_wall_s"] = elapsed()
    write_json(out / "errors.json", ledger["errors"])
    write_json(out / "ledger.json", ledger)
    files = {path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file()
             and path.name not in ("result.json", "output-manifest.json")}
    manifest = {"schema": "flylab.m3.e244.causal-contrast.manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    manifest_bytes = json_bytes(manifest)
    if elapsed:
        result["process_wall_s"] = elapsed()
    result_bytes = json_bytes(result)
    if manifest["total_bytes"] + len(manifest_bytes) + len(result_bytes) > RESOURCE["output_bytes"]:
        raise OSError("E244 output exceeds 128 MiB")
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E244 deadline before result publication")
    (out / "output-manifest.json").write_bytes(manifest_bytes)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E244 deadline before result publication")
    (out / "result.json").write_bytes(result_bytes)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E244 deadline after result publication")


def settle_publication(out, result, raw, ledger, *, elapsed, deadline):
    """At most three terminal refreshes; no success marker on late failure."""
    first = None
    started = time.perf_counter()
    ledger["publication_started_elapsed_s"] = elapsed()
    for attempt in range(3):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, ledger, deadline=deadline, elapsed=elapsed)
            result["process_wall_s"] = elapsed()
            result["publication_wall_s"] = time.perf_counter() - started
            write_json(out / "result.json", result)
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("E244 deadline after final time receipt")
            return first
        except BaseException as error:  # noqa: BLE001 - preserve first and secondary errors
            if first is None:
                first = error
            ledger["errors"].append({"stage": "publication", "attempt": attempt + 1,
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
            result["validity"] = "incomplete_or_invalid"
            result["hypothesis_supported"] = None
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            if deadline is not None and time.perf_counter() >= deadline:
                deadline = None
    result["finalization_pending"] = True
    result["hypothesis_supported"] = None
    result["errors"] = ledger["errors"]
    result["raw_secondary_errors"] = ledger.get("raw_secondary_errors", [])
    try:
        write_json(out / "publication-failure.json", ledger["errors"])
        write_json(out / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - preserve primary even if disk stays broken
        ledger["errors"].append({"stage": "terminal_publication",
            "type": type(error).__name__, "message": str(error)})
    return first


def run(freeze_path, freeze_sha, review_path, review_sha, output,
        *, generated_test_output=False):
    began = time.perf_counter()
    out = Path(output).resolve()
    if not generated_test_output and out != RESERVED_OUT:
        raise ValueError("E244 original output must use reserved directory")
    if out.exists():
        raise FileExistsError("one-shot E244 output already exists")
    out.mkdir(parents=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    process = psutil.Process()
    ledger = {"schema": "flylab.m3.e244.causal-contrast.ledger.v1",
        "phase": "gate", "loaded_input_roles": [], "analysis_requested_count": 0,
        "analysis_entered_count": 0, "analysis_completed_count": 0,
        "original_calculation_count": 0, "completed_neighbor_legs": 0,
        "completed_neighbor_batches": [0]*6, "completed_query_rows": [0]*6,
        "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
        "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
        "process_id": process.pid,
        "child_process_ids_start": [child.pid for child in process.children(recursive=True)],
        "retries": 0, "errors": []}
    result = {"schema": "flylab.m3.e244.causal-contrast.result.v1",
        "validity": "incomplete_or_invalid", "neighbor_distance_kind": "euclidean",
        "hypothesis_supported": None, "descriptive_only": True,
        "observed_tail_independent_validation": False,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "active_UI_changed": False, "finalization_pending": True}
    raw = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter() - began

    def progress(kind, leg, rows):
        if kind == "neighbor_batch":
            ledger["completed_neighbor_batches"][leg] += 1
            ledger["completed_query_rows"][leg] = rows
        elif kind == "neighbor_leg":
            ledger["completed_neighbor_legs"] = leg + 1
        if elapsed() >= 50:
            raise TimeoutError("E244 calculation exceeded 50-second work deadline")
        if psutil.Process().memory_info().rss > RESOURCE["resident_bytes"]:
            raise MemoryError("E244 RSS exceeds 3 GiB")

    try:
        freeze, review, paths = verify_gate(
            freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "additional_reused_source_sha256": review["additional_reused_source_sha256"],
            "additional_input_sha256": review["additional_input_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < RESOURCE["minimum_free_bytes"]:
            raise OSError("E244 requires 2 GiB free disk")
        if elapsed() >= 50:
            raise TimeoutError("E244 hash gate exhausted work deadline")
        ledger["phase"] = "input"

        def loaded(role):
            ledger["loaded_input_roles"].append(role)
            if elapsed() >= 50:
                raise TimeoutError("E244 input load exceeded work deadline")

        ledger["threadpool_before"] = threadpool_info()
        limiter = None
        work_error = None
        try:
            limiter = threadpool_limits(limits=1)
            observed = threadpool_info()
            ledger["threadpool_observed"] = observed
            if any(pool.get("num_threads") != 1 for pool in observed):
                raise RuntimeError("E244 imported threadpool exceeds one thread")
            inputs = load_inputs(paths, progress=loaded)
            validate_original_dimensions(inputs)
            if (inputs["baseline_result"].get("validity") != "valid_saved_diagnostic"
                    or inputs["baseline_result"].get("neighbor_distance_kind") != "euclidean"
                    or inputs["baseline_ledger"].get("analysis_completed_count") != 1
                    or inputs["source_proof"].get("validity") != "independently_valid"):
                raise ValueError("E244 valid independently audited E243 baseline required")
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            report, raw = analyze_arrays(*(inputs[key] for key in (
                "parameters", "descriptor", "targets", "groups",
                "baseline_raw", "baseline_result")), raw=raw, progress=progress)
            ledger["analysis_completed_count"] = 1
        except BaseException as error:  # noqa: BLE001 - restore cannot mask work failure
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:  # noqa: BLE001 - separately retain restore failure
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = report
        result["hypothesis_supported"] = report["hypothesis"]["supported"]
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:  # noqa: BLE001 - keep first failure and returned raw
        primary = error
        ledger["errors"].append({"stage": ledger["phase"],
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append({"stage": "threadpool_restore",
                "type": type(restore_error).__name__, "message": str(restore_error),
                "traceback": "".join(traceback.format_exception(
                    type(restore_error), restore_error, restore_error.__traceback__))})
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
        analysis_requested_count=ledger["analysis_requested_count"],
        analysis_entered_count=ledger["analysis_entered_count"],
        analysis_completed_count=ledger["analysis_completed_count"],
        original_calculation_count=ledger["original_calculation_count"],
        completed_neighbor_batches=ledger["completed_neighbor_batches"],
        completed_query_rows=ledger["completed_query_rows"],
        completed_neighbor_legs=ledger["completed_neighbor_legs"],
        fit_calls=0, solver_calls=0, cuda_calls=0,
        full_banc_calls=0, body_constructors=0, physics_steps=0,
        errors=ledger["errors"], process_wall_s=elapsed())
    ledger["child_process_ids_end"] = [child.pid for child in process.children(recursive=True)]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle_publication(out, result, raw, ledger,
        elapsed=elapsed, deadline=began + RESOURCE["internal_s"])
    if primary is None:
        primary = publication_error
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
