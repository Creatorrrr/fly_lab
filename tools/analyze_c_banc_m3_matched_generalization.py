"""E243 saved-only diagnosis of the frozen matched-teacher response family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
FREEZE_SHA = "464ed18eb279823f7bbdaf7371b4e3a690b675f1fdaec4a08946477002e8a919"
FREEZE_SCHEMA = "flylab.m3.e243.matched-generalization.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e243.matched-generalization.review.v1"
OWNED = ("tools/analyze_c_banc_m3_matched_generalization.py",
         "tests/test_c_banc_matched_generalization.py")
RESERVED_OUT = (ROOT / "verification/banc-natural-gait-m3-20260919/"
                "e243-matched-generalization").resolve()
LIMITS = {"internal_s": 60, "finalization_reserve_s": 10,
          "output_bytes": 134217728, "minimum_free_bytes": 2147483648,
          "resident_bytes": 3221225472}
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
        raise ValueError("repository-relative slash path required")
    relative = Path(name)
    if relative.is_absolute() or any(p in (".", "..") for p in relative.parts):
        raise ValueError("unsafe frozen path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("frozen path missing or outside repository")
    return path


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("E243 freeze SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E243 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != RESOURCE
            or freeze.get("shape") != {"samples": 2001, "motors": 391,
                "free_motors": 343, "fixed_motors": 48, "muscles": 96,
                "free_muscles": 90, "fixed_muscles": 6, "basis_count": 64,
                "variables": 5760, "leg_groups": [16, 14, 15, 16, 14, 15],
                "leg_widths": [1024, 896, 960, 1024, 896, 960],
                "box_rows": 11520}
            or freeze.get("diagnostic", {}).get("neighbor_exclusion_inclusive_controls") != 20
            or freeze.get("split", {}).get("tail_independent_validation") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E243 frozen scientific contract changed")
    own_hashes = {name: file_sha(repository_path(name)) for name in OWNED}
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or review.get("source_sha256") != own_hashes):
        raise ValueError("E243 parent source review invalid")
    if freeze.get("packages") != {name: version(name) for name in freeze["packages"]}:
        raise ValueError("E243 frozen package versions changed")
    if file_sha(repository_path(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E243 design changed")
    # All frozen source/data identities are checked before any NPZ opens.
    for field in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        for name, expected in freeze[field].items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E243 frozen input changed: " + name)
    for field in ("additional_reused_source_sha256", "additional_input_sha256"):
        hashes = review.get(field)
        if not isinstance(hashes, dict):
            raise TypeError("E243 review additional hashes missing: " + field)
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E243 review additional byte changed: " + name)
    paths = {role: repository_path(name) for role, name in freeze["paths"].items()}
    if set(paths) != {"parameters", "geometry", "targets", "groups",
                      "canonical_original", "descriptor", "candidate",
                      "metrics", "source_result", "source_ledger",
                      "source_proof", "source_supervisor"}:
        raise ValueError("E243 frozen role set changed")
    return freeze, paths


def load_inputs(paths, *, progress=None):
    values = {}
    selections = {
        "parameters": ("free_columns", "fixed_columns", "membership", "pool", "moment", "cap", "ltm"),
        "geometry": ("centers", "sigma2", "anchor_rows"),
        "descriptor": ("descriptor",),
        "targets": ("target_command",),
        "canonical_original": ("command",),
        "candidate": ("beta", "q32"),
    }
    for role, keys in selections.items():
        with np.load(paths[role], allow_pickle=False) as saved:
            values[role] = {key: saved[key].copy() for key in keys}
        if progress:
            progress(role)
    values["groups"] = json.loads(paths["groups"].read_text(encoding="utf-8"))["groups"]
    if progress:
        progress("groups")
    values["metrics"] = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if progress:
        progress("metrics")
    return values


def validate_original_dimensions(inputs):
    if (inputs["descriptor"]["descriptor"].shape != (2001, 391, 9)
            or inputs["geometry"]["centers"].shape != (391, 64, 9)
            or inputs["parameters"]["free_columns"].shape != (343,)
            or inputs["parameters"]["fixed_columns"].shape != (48,)
            or inputs["targets"]["target_command"].shape != (2001, 42)
            or inputs["candidate"]["beta"].shape != (96, 64)):
        raise ValueError("E243 original frozen array dimensions changed")


def rms(value):
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def error_stats(command, target):
    error = command - target
    velocity = np.diff(error, axis=0) / .005
    def stats(q, v):
        return {"q_rms_rad": rms(q), "q_max_abs_rad": float(np.max(np.abs(q))),
                "velocity_rms_rad_s": rms(v),
                "velocity_max_abs_rad_s": float(np.max(np.abs(v)))}
    result = {"global": {}, "legs": {}, "startup": {
        "q_rms_rad": rms(error[1:100]),
        "q_max_abs_rad": float(np.max(np.abs(error[1:100])))}}
    for phase, lo, hi in (("train", 1, 1201), ("validation", 1201, 2001)):
        result["global"][phase] = stats(error[lo:hi], velocity[lo-1:hi-1])
        for i, leg in enumerate(LEGS):
            sl = slice(i*7, (i+1)*7)
            result["legs"].setdefault(leg, {})[phase] = stats(
                error[lo:hi, sl], velocity[lo-1:hi-1, sl])
    return result


def compare_metrics(actual, stored):
    maximum = 0.
    for scope in ("global", "legs"):
        names = LEGS if scope == "legs" else (None,)
        for name in names:
            for phase in ("train", "validation"):
                a = actual[scope][name][phase] if name else actual[scope][phase]
                b = stored[scope][name][phase] if name else stored[scope][phase]
                for key, value in a.items():
                    difference = abs(value - float(b[key]))
                    maximum = max(maximum, difference)
                    if not np.isclose(value, b[key], atol=2e-10, rtol=2e-12):
                        raise ValueError("E242 saved metric mismatch: " + str((scope, name, phase, key)))
    for key, value in actual["startup"].items():
        maximum = max(maximum, abs(value - float(stored["startup"][key])))
        if not np.isclose(value, stored["startup"][key], atol=2e-10, rtol=2e-12):
            raise ValueError("E242 saved startup metric mismatch: " + key)
    return maximum


def anatomy_weights(parameters, groups):
    free = np.asarray(parameters["free_columns"], np.int64)
    fixed = np.asarray(parameters["fixed_columns"], np.int64)
    owner = np.asarray(parameters["membership"], np.int64)
    pool = np.asarray(parameters["pool"], np.float64)
    moment = np.asarray(parameters["moment"], np.float64)
    cap = np.asarray(parameters["cap"], np.float64)
    if (pool.shape != (96, len(owner)) or moment.shape != (42, 96)
            or cap.shape != (len(owner),) or len(free) + len(fixed) != len(owner)
            or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(len(owner)))
            or np.any((owner < 0) | (owner >= 96))):
        raise ValueError("original anatomy identity invalid")
    weights = np.zeros((6, len(free)), np.float64)
    for index, leg in enumerate(LEGS):
        ids = [j for j, m in enumerate(free) if groups[int(owner[m])]["leg"] == leg]
        motors = free[ids]
        muscles = owner[motors]
        values = .02 * cap[motors] * pool[muscles, motors] * np.linalg.norm(
            moment[7*index:7*(index+1), muscles], axis=0)
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("leg sensitivity is nonpositive/nonfinite: " + leg)
        weights[index, ids] = values / values.sum()
    return weights


def summarize(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def analyze_arrays(parameters, geometry, descriptor_data, targets,
                   canonical, candidate_data, groups, metrics_data,
                   *, raw=None, progress=None, train_last=1200):
    """Analyze aligned saved arrays. Caller-owned raw retains completed prefixes on failure."""
    inputs = {"parameters": parameters, "geometry": geometry,
              "descriptor": descriptor_data, "targets": targets,
              "canonical_original": canonical, "candidate": candidate_data,
              "groups": groups, "metrics": metrics_data}
    raw = {} if raw is None else raw
    p, geo = inputs["parameters"], inputs["geometry"]
    descriptor = np.asarray(inputs["descriptor"]["descriptor"], np.float64)
    target = np.asarray(inputs["targets"]["target_command"], np.float64)
    original = np.asarray(inputs["canonical_original"]["command"], np.float64)
    candidate = np.asarray(inputs["candidate"]["q32"], np.float64)
    beta = np.asarray(inputs["candidate"]["beta"], np.float64)
    ltm = np.asarray(p["ltm"], np.int64)
    free = np.asarray(p["free_columns"], np.int64)
    centers = np.asarray(geo["centers"], np.float64)
    sigma2 = np.asarray(geo["sigma2"], np.float64)
    anchors = np.asarray(geo["anchor_rows"], np.int64)
    n, motors = descriptor.shape[:2]
    k = centers.shape[1]
    if (descriptor.shape != (n, motors, 9) or n < train_last + 1
            or train_last < 22 or target.shape != (n, 42)
            or original.shape != (n, 42) or candidate.shape != (n, 42)
            or beta.shape != (96, k) or centers.shape != (motors, k, 9)
            or ltm.shape != (6,) or np.any((ltm < 0) | (ltm >= 96))
            or np.any(beta[ltm] != 0)
            or sigma2.shape != (motors,) or anchors.shape != (96, k)
            or np.any(sigma2 <= 0) or np.any(anchors < 0)
            or np.any(anchors > train_last) or np.any((free < 0) | (free >= motors))
            or len(np.unique(free)) != len(free)
            or not all(np.isfinite(v).all() for v in
                       (descriptor, centers, sigma2, target, original, candidate, beta))):
        raise ValueError("E243 saved descriptor/geometry/readout alignment invalid")
    weights = anatomy_weights(p, inputs["groups"])
    raw.update(free_columns=free.copy(),
        min_distance=np.full((n, len(free)), np.nan),
        nearest_center=np.full((n, len(free)), -1, np.int64),
        anchor_control=np.full((n, len(free)), -1, np.int64),
        entropy=np.full((n, len(free)), np.nan),
        effective_centers=np.full((n, len(free)), np.nan),
        gate=np.full((n, len(free)), np.nan),
        instantaneous_target=np.full((n, len(free)), np.nan),
        leg_weighted_min_distance=np.full((n, 6), np.nan),
        leg_weighted_entropy=np.full((n, 6), np.nan),
        leg_weighted_effective_centers=np.full((n, 6), np.nan),
        leg_weighted_gate=np.full((n, 6), np.nan),
        leg_weighted_instantaneous_target=np.full((n, 6), np.nan),
        leg_coordinate_contribution=np.full((n, 6, 9), np.nan),
        nearest_train_control=np.full((n, 6), -1, np.int64),
        nearest_train_distance=np.full((n, 6), np.nan),
        neighbor_teacher_q_difference=np.full((n, 42), np.nan),
        candidate_q_error=candidate-target,
        original_q_error=original-target,
        sensitivity_weight=weights)
    raw["nearest_train_distance"][0] = 0
    raw["neighbor_teacher_q_difference"][0] = 0
    raw["completed_mn"] = np.asarray(0, np.int32)
    raw["completed_legs"] = np.asarray(0, np.int32)
    raw["completed_neighbor_legs"] = np.asarray(0, np.int32)

    for position, motor in enumerate(free):
        delta = descriptor[:, motor, None, :] - centers[motor, None, :, :]
        coordinates = delta * delta / (2*sigma2[motor])
        distances = coordinates.sum(axis=2)
        closest = np.argmin(distances, axis=1)
        logits = -distances
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropy = -np.sum(np.where(probabilities > 0,
            probabilities*np.log(np.maximum(probabilities, np.finfo(float).tiny)), 0), axis=1)
        x = descriptor[:, motor]
        gate = -np.expm1(-8*(x[:, 0] + np.abs(x[:, 2])
            + np.maximum(x[:, 3], 0) + np.abs(x[:, 5])))
        owner = int(p["membership"][motor])
        raw["min_distance"][:, position] = distances[np.arange(n), closest]
        raw["nearest_center"][:, position] = closest
        raw["anchor_control"][:, position] = anchors[owner, closest]
        raw["entropy"][:, position] = entropy
        raw["effective_centers"][:, position] = 1/np.sum(probabilities**2, axis=1)
        raw["gate"][:, position] = gate
        raw["instantaneous_target"][:, position] = (
            p["cap"][motor] * gate * (probabilities @ beta[owner]))
        # The nine coordinate contributions belong to the closest own center.
        if position == 0:
            contributions = np.full((n, len(free), 9), np.nan)
        contributions[:, position] = coordinates[np.arange(n), closest]
        raw["completed_mn"] = np.asarray(position + 1, np.int32)
        if progress:
            progress("mn", position + 1)

    for leg_index, leg in enumerate(LEGS):
        w = weights[leg_index]
        raw["leg_weighted_min_distance"][:, leg_index] = raw["min_distance"] @ w
        raw["leg_weighted_entropy"][:, leg_index] = raw["entropy"] @ w
        raw["leg_weighted_effective_centers"][:, leg_index] = raw["effective_centers"] @ w
        raw["leg_weighted_gate"][:, leg_index] = raw["gate"] @ w
        raw["leg_weighted_instantaneous_target"][:, leg_index] = (
            raw["instantaneous_target"] @ w)
        raw["leg_coordinate_contribution"][:, leg_index] = np.einsum(
            "tmj,m->tj", contributions, w, optimize=True)
        raw["completed_legs"] = np.asarray(leg_index + 1, np.int32)
        if progress:
            progress("leg", leg_index + 1)
    del contributions

    train = np.arange(1, train_last+1)
    query = np.arange(1, n)
    for leg_index, leg in enumerate(LEGS):
        selected = np.flatnonzero(weights[leg_index] > 0)
        motors_leg = free[selected]
        scales = np.sqrt(weights[leg_index, selected] / (2*sigma2[motors_leg]))
        features = (descriptor[:, motors_leg, :] * scales[None, :, None]).reshape(n, -1)
        reference = np.ascontiguousarray(features[train])
        for begin in range(0, len(query), 128):
            qs = query[begin:begin+128]
            distances = cdist(np.ascontiguousarray(features[qs]), reference)
            distances[np.abs(qs[:, None]-train[None, :]) <= 20] = np.inf
            nearest = np.argmin(distances, axis=1)
            if not np.isfinite(distances[np.arange(len(qs)), nearest]).all():
                raise ValueError("no admissible train neighbor: " + leg)
            chosen = train[nearest]
            raw["nearest_train_control"][qs, leg_index] = chosen
            raw["nearest_train_distance"][qs, leg_index] = distances[
                np.arange(len(qs)), nearest]
            sl = slice(7*leg_index, 7*(leg_index+1))
            raw["neighbor_teacher_q_difference"][qs, sl] = target[chosen, sl] - target[qs, sl]
            if progress:
                progress("neighbor_batch", leg_index + 1)
        raw["completed_neighbor_legs"] = np.asarray(leg_index + 1, np.int32)
        if progress:
            progress("neighbor_leg", leg_index + 1)

    metrics = inputs["metrics"]
    original_metrics = error_stats(original, target)
    candidate_metrics = error_stats(candidate, target)
    metric_diff = max(compare_metrics(original_metrics, metrics["original"]),
                      compare_metrics(candidate_metrics, metrics["projected"]))
    segments = {"train": slice(1, train_last+1),
                "steady": slice(100, train_last+1),
                "observed_tail": slice(train_last+1, n)}
    report = {"schema": "flylab.m3.e243.matched-generalization.analysis.v1",
        "descriptive_only": True, "observed_tail_independent_validation": False,
        "instantaneous_target_is_RK_rate": False,
        "e242_metric_max_abs_difference": metric_diff,
        "beta_boundary_fraction": {"lower": float(np.mean(beta[np.unique(
            p["membership"][free])] <= 1e-8)),
                                  "upper": float(np.mean(beta[np.unique(
            p["membership"][free])] >= 1-1e-8))},
        "fixed_ltm_beta_exact_zero": True,
        "segments": {}, "bins": []}
    for name, rows in segments.items():
        per_leg = {}
        for leg_index, leg in enumerate(LEGS):
            values = raw["leg_weighted_min_distance"][rows, leg_index]
            train_p95 = float(np.quantile(
                raw["leg_weighted_min_distance"][segments["train"], leg_index], .95))
            coordinate = raw["leg_coordinate_contribution"][rows, leg_index].mean(axis=0)
            neighbor = raw["neighbor_teacher_q_difference"][rows, 7*leg_index:7*(leg_index+1)]
            per_leg[leg] = {"minimum_distance": summarize(values),
                "fraction_over_train_p95": float(np.mean(values > train_p95)),
                "entropy": summarize(raw["leg_weighted_entropy"][rows, leg_index]),
                "effective_centers": summarize(raw["leg_weighted_effective_centers"][rows, leg_index]),
                "gate": summarize(raw["leg_weighted_gate"][rows, leg_index]),
                "instantaneous_target": summarize(
                    raw["leg_weighted_instantaneous_target"][rows, leg_index]),
                "coordinate_contribution_mean": coordinate.tolist(),
                "x_fast_slow_contribution_mean": [float(coordinate[i:i+3].sum())
                                                  for i in (0, 3, 6)],
                "neighbor_distance": summarize(raw["nearest_train_distance"][rows, leg_index]),
                "neighbor_teacher_q_difference_rms": rms(neighbor)}
        report["segments"][name] = per_leg
    for index, first in enumerate(range(1, n, 200)):
        last = min(first+200, n)
        row = {"bin": index+1, "first_control": first, "last_control": last-1,
               "global": {}, "legs": {}}
        for label, error in (("original", raw["original_q_error"]),
                             ("candidate", raw["candidate_q_error"])):
            q = error[first:last]
            velocity = np.diff(error[first-1:last], axis=0)/.005
            row["global"][label] = {"q_rms_rad": rms(q),
                "q_max_abs_rad": float(np.max(np.abs(q))),
                "velocity_rms_rad_s": rms(velocity),
                "velocity_max_abs_rad_s": float(np.max(np.abs(velocity)))}
            for leg_index, leg in enumerate(LEGS):
                sl = slice(leg_index*7, (leg_index+1)*7)
                row["legs"].setdefault(leg, {})[label] = {
                    "q_rms_rad": rms(q[:, sl]),
                    "q_max_abs_rad": float(np.max(np.abs(q[:, sl]))),
                    "velocity_rms_rad_s": rms(velocity[:, sl]),
                    "velocity_max_abs_rad_s": float(np.max(np.abs(velocity[:, sl])))}
        report["bins"].append(row)
    if not all(np.isfinite(value).all() for value in raw.values()
               if isinstance(value, np.ndarray) and value.dtype.kind == "f"):
        raise ValueError("E243 completed diagnostic contains nonfinite values")
    return report, raw


def publish(out, result, raw, ledger, *, deadline=None, elapsed=None):
    """Keep raw independently, then refresh every terminal receipt in order."""
    raw_path = out / "diagnostic-raw.npz"
    if raw and not raw_path.is_file():
        temporary = out / "diagnostic-raw.npz.part"
        raw_error = None
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **raw)
            os.replace(temporary, raw_path)
        except BaseException as error:  # noqa: BLE001 - cleanup cannot replace storage failure
            raw_error = error
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as error:  # noqa: BLE001 - retain both storage failures
            if raw_error is None:
                raw_error = error
            else:
                ledger.setdefault("raw_secondary_errors", []).append({
                    "stage": "raw_temporary_cleanup", "type": type(error).__name__,
                    "message": str(error)})
        if raw_error is not None:
            raise raw_error
    if elapsed is not None:
        ledger["process_wall_s"] = elapsed()
    write_json(out / "errors.json", ledger["errors"])
    write_json(out / "ledger.json", ledger)
    files = {path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file()
             and path.name not in ("result.json", "output-manifest.json")}
    manifest = {"schema": "flylab.m3.e243.matched-generalization.manifest.v1",
                "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    manifest_bytes = json_bytes(manifest)
    if elapsed is not None:
        result["process_wall_s"] = elapsed()
    result_bytes = json_bytes(result)
    if (manifest["total_bytes"] + len(manifest_bytes) + len(result_bytes)
            > LIMITS["output_bytes"]):
        raise OSError("E243 output exceeds 128 MiB")
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E243 deadline before result publication")
    (out / "output-manifest.json").write_bytes(manifest_bytes)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E243 deadline before result publication")
    (out / "result.json").write_bytes(result_bytes)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("E243 deadline after result publication")


def settle_publication(out, result, raw, ledger, *, elapsed, deadline):
    """Bounded invalid refresh after first/late receipt failures."""
    first_publication_error = None
    publication_started = time.perf_counter()
    ledger["publication_started_elapsed_s"] = elapsed()
    for attempt in range(3):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, ledger, deadline=deadline, elapsed=elapsed)
            # Capture the first complete publication. This result is excluded
            # from the manifest, so the final time observation does not stale it.
            result["process_wall_s"] = elapsed()
            result["publication_wall_s"] = time.perf_counter() - publication_started
            write_json(out / "result.json", result)
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("E243 deadline after final time receipt")
            return first_publication_error
        except BaseException as error:  # noqa: BLE001 - retain primary and latest secondary
            if first_publication_error is None:
                first_publication_error = error
            ledger["errors"].append({"stage": "publication", "attempt": attempt + 1,
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
            result["validity"] = "incomplete_or_invalid"
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            # A missed deadline cannot authorize success; failure receipts may
            # still need a brief bounded write after the clock has expired.
            if deadline is not None and time.perf_counter() >= deadline:
                deadline = None
    result["finalization_pending"] = True
    result["errors"] = ledger["errors"]
    result["raw_secondary_errors"] = ledger.get("raw_secondary_errors", [])
    try:
        write_json(out / "publication-failure.json", ledger["errors"])
        write_json(out / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - retain primary even if storage stays broken
        ledger["errors"].append({"stage": "terminal_publication",
            "type": type(error).__name__, "message": str(error)})
    return first_publication_error


def run(freeze_path, freeze_sha, review_path, review_sha, output,
        *, generated_test_output=False):
    began = time.perf_counter()
    out = Path(output).resolve()
    if not generated_test_output and out != RESERVED_OUT:
        raise ValueError("E243 original output must use reserved directory")
    if out.exists():
        raise FileExistsError("one-shot E243 output already exists")
    out.mkdir(parents=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ledger = {"schema": "flylab.m3.e243.matched-generalization.ledger.v1",
        "phase": "gate", "loaded_input_roles": [], "completed_mn": 0,
        "completed_legs": 0, "completed_neighbor_legs": 0,
        "original_calculation_count": 0, "analysis_requested": False,
        "analysis_requested_count": 0, "analysis_entered": False,
        "analysis_entered_count": 0, "analysis_completed_count": 0,
        "analysis_completed": False, "fit_calls": 0, "solver_calls": 0,
        "cuda_calls": 0, "neural_calls": 0, "body_calls": 0,
        "full_banc_calls": 0, "body_constructors": 0, "physics_steps": 0,
        "retries": 0,
        "errors": []}
    result = {"schema": "flylab.m3.e243.matched-generalization.result.v1",
              "validity": "incomplete_or_invalid", "descriptive_only": True,
              "neighbor_distance_kind": "euclidean",
              "candidate_adopted": False, "F2_complete": False,
              "F6_exposed": False, "whole_goal_complete": False,
              "active_UI_changed": False, "finalization_pending": True}
    raw = {}
    primary = None
    restore_error = None

    def elapsed():
        return time.perf_counter() - began

    def progress(kind, number):
        if kind == "mn":
            ledger["completed_mn"] = number
        elif kind == "leg":
            ledger["completed_legs"] = number
        elif kind == "neighbor_leg":
            ledger["completed_neighbor_legs"] = number
        if elapsed() > LIMITS["internal_s"] - LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E243 calculation exceeded 50-second work deadline")
        import psutil
        if psutil.Process().memory_info().rss > LIMITS["resident_bytes"]:
            raise MemoryError("E243 RSS exceeds 3 GiB")

    try:
        freeze, paths = verify_gate(freeze_path, freeze_sha, review_path, review_sha)
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha,
            "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "additional_reused_source_sha256": review["additional_reused_source_sha256"],
            "additional_input_sha256": review["additional_input_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < LIMITS["minimum_free_bytes"]:
            raise OSError("E243 requires 2 GiB free disk")
        if elapsed() > 50:
            raise TimeoutError("E243 hash gate exhausted work deadline")
        ledger["phase"] = "input"
        source_result = json.loads(paths["source_result"].read_text(encoding="utf-8"))
        source_ledger = json.loads(paths["source_ledger"].read_text(encoding="utf-8"))
        source_proof = json.loads(paths["source_proof"].read_text(encoding="utf-8"))
        if (source_result.get("validity") != "valid_conditional_matched_teacher_fit"
                or source_result.get("candidate_eligible") is not False
                or source_ledger.get("solver_calls") != 1
                or source_proof.get("validity") != "independently_valid"):
            raise ValueError("E243 pinned rejected E242 fit/proof changed")
        ledger["loaded_input_roles"].extend(("source_result", "source_ledger", "source_proof"))

        def loaded(role):
            ledger["loaded_input_roles"].append(role)
            if elapsed() > 50:
                raise TimeoutError("E243 input load exceeded work deadline")

        ledger["threadpool_before"] = threadpool_info()
        limiter = None
        work_error = None
        try:
            limiter = threadpool_limits(limits=1)
            observed = threadpool_info()
            ledger["threadpool_observed"] = observed
            if any(pool.get("num_threads") != 1 for pool in observed):
                raise RuntimeError("E243 imported threadpool exceeds one thread")
            inputs = load_inputs(paths, progress=loaded)
            validate_original_dimensions(inputs)
            ledger["phase"] = "analysis"
            ledger["analysis_requested"] = True
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered"] = True
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            report, raw = analyze_arrays(*(inputs[key] for key in (
                "parameters", "geometry", "descriptor", "targets",
                "canonical_original", "candidate", "groups", "metrics")),
                raw=raw, progress=progress)
            ledger["analysis_completed"] = True
            ledger["analysis_completed_count"] = 1
        except BaseException as error:  # noqa: BLE001 - restore must not replace work failure
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:  # noqa: BLE001 - retain secondary failure
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = report
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:  # noqa: BLE001 - retain first failure and partial raw
        primary = error
        ledger["errors"].append({"stage": ledger["phase"],
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append({"stage": "threadpool_restore",
                "type": type(restore_error).__name__, "message": str(restore_error),
                "traceback": "".join(traceback.format_exception(
                    type(restore_error), restore_error, restore_error.__traceback__))})
    ledger["process_wall_s"] = elapsed()
    result.update(completed_mn=ledger["completed_mn"],
        completed_legs=ledger["completed_legs"],
        completed_neighbor_legs=ledger["completed_neighbor_legs"],
        loaded_input_roles=ledger["loaded_input_roles"],
        analysis_requested=ledger["analysis_requested"],
        analysis_requested_count=ledger["analysis_requested_count"],
        analysis_entered=ledger["analysis_entered"],
        analysis_entered_count=ledger["analysis_entered_count"],
        analysis_completed=ledger["analysis_completed"],
        analysis_completed_count=ledger["analysis_completed_count"],
        original_calculation_count=ledger["original_calculation_count"],
        full_banc_calls=0, body_constructors=0, physics_steps=0,
        process_wall_s=elapsed(), errors=ledger["errors"])
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle_publication(out, result, raw, ledger,
        elapsed=elapsed, deadline=began + LIMITS["internal_s"])
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
