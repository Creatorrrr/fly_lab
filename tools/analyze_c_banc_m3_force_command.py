"""E186 bounded, saved-data force-to-command attribution; no model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "docs/BANC_M3_E186_FORCE_COMMAND_DESIGN_20260924.md"
METADATA = "verification/banc-natural-gait-m3-20260919/parent-e186-frozen-metadata.json"
OUTPUT = "verification/banc-natural-gait-m3-20260919/e186-force-command"
OWNED = (
    "tools/analyze_c_banc_m3_force_command.py",
    "tests/test_c_banc_force_command.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
SIGNS = ("negative", "zero", "positive")
BANDS = {"pre": (-20, -10), "lead": (-10, 0), "post": (0, 10), "recover": (10, 21)}
PARAMETERS = {
    "control_dt_s": 0.005,
    "native_dt_s": 0.0001,
    "control_rows": 601,
    "native_rows": 30000,
    "cpu_threads": 1,
    "gpu_count": 0,
    "internal_wall_s": 40,
    "supervisor_wall_s": 45,
    "automatic_retries": 0,
    "inherited_cumulative_wall_s": 4339.688065899652,
    "cumulative_cap_s": 4994.0,
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(relative):
    require(
        isinstance(relative, str) and relative and "\\" not in relative, "unsafe path"
    )
    p = PurePosixPath(relative)
    require(
        not p.is_absolute()
        and ":" not in relative
        and all(x not in ("", ".", "..") for x in p.parts),
        "unsafe path",
    )
    target = ROOT.joinpath(*p.parts)
    require(
        not any(x.is_symlink() for x in (target, *target.parents) if x != ROOT.parent),
        "symlink path",
    )
    require(target.resolve().is_relative_to(ROOT.resolve()), "path escaped root")
    return target


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate(freeze_path, review_path, output):
    """Complete source/input gate before any original numeric file is opened."""
    freeze = read_json(freeze_path)
    review = read_json(review_path)
    require(
        freeze.get("schema") == "flylab.m3.e186.force-command.freeze.v1",
        "freeze schema",
    )
    require(
        review.get("schema") == "flylab.m3.e186.force-command.source-review.v1",
        "review schema",
    )
    freeze_sha = digest(freeze_path)
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == freeze_sha,
        "parent source approval",
    )
    require(
        freeze.get("design_path") == DESIGN
        and freeze.get("design_sha256") == digest(safe_path(DESIGN)),
        "design identity",
    )
    require(
        freeze.get("metadata_path") == METADATA
        and freeze.get("metadata_sha256") == digest(safe_path(METADATA)),
        "metadata identity",
    )
    require(
        freeze.get("output_path") == OUTPUT
        and output.resolve() == safe_path(OUTPUT).resolve(),
        "output identity",
    )
    require(freeze.get("parameters") == PARAMETERS, "resource contract")
    supervisor = "verification/banc-natural-gait-m3-20260919/parent-e186-supervise.py"
    require(
        freeze.get("parent_supervisor_sha256") == digest(safe_path(supervisor)),
        "supervisor changed",
    )
    metadata = read_json(safe_path(METADATA))
    require(
        metadata.get("schema") == "flylab.m3.e186.force-command.metadata.v1"
        and metadata["design_path"] == DESIGN
        and metadata["design_sha256"] == freeze["design_sha256"],
        "metadata contract",
    )
    require(
        metadata["cumulative_before_s"] == PARAMETERS["inherited_cumulative_wall_s"]
        and metadata["stage_reserve_s"] == 45
        and metadata["original_analysis_runs"] == 0,
        "stage accounting",
    )
    inputs = metadata.get("inputs")
    reused = metadata.get("source_hashes")
    require(
        isinstance(inputs, dict)
        and len(inputs) == 199
        and isinstance(reused, dict)
        and len(reused) == 17,
        "full inventories required",
    )
    sources = {name: digest(safe_path(name)) for name in (*OWNED, *reused)}
    require(
        len(sources) == 19
        and freeze.get("source_hashes") == sources
        and review.get("source_hashes") == sources
        and reused == {k: sources[k] for k in reused},
        "source changed",
    )
    require(
        review.get("design_sha256") == freeze["design_sha256"]
        and review.get("metadata_sha256") == freeze["metadata_sha256"],
        "review identity",
    )
    require(freeze.get("input_sha256") == inputs, "input inventory differs")
    for name, sha in inputs.items():
        require(
            isinstance(sha, str) and len(sha) == 64 and digest(safe_path(name)) == sha,
            "input changed: " + name,
        )
    prior = read_json(safe_path(metadata["previous_review_path"]))
    require(
        prior.get("saved_raw_analysis_accepted") is True
        and prior.get("child_observation_valid") is True
        and prior.get("original_driver_accepted") is False,
        "E185 acceptance differs",
    )
    return metadata, freeze_sha


def f64(value, shape, label, *, finite=True):
    a = np.asarray(value)
    require(
        a.shape == shape
        and a.dtype == np.float64
        and (not finite or np.isfinite(a).all()),
        "invalid " + label,
    )
    return a


def i64(value, shape, label):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype == np.int64, "invalid " + label)
    return a


def check_residual(expected, observed, label, atol=1e-10):
    require(
        expected.shape == observed.shape
        and np.isfinite(expected).all()
        and np.isfinite(observed).all(),
        "invalid comparison " + label,
    )
    delta = np.abs(expected - observed)
    maximum = float(np.max(delta))
    require(maximum <= atol, f"{label} residual {maximum:.17g} exceeds {atol}")
    return maximum


def reconstruct(rate, bank, activation, original, cal):
    """Full 391-MN contributions, including fixed original tau=.01 transfer."""
    t, n = rate.shape
    m = activation.shape[1]
    require(
        bank.shape == (t, n, 3) and original.shape == (t, m), "reconstruction shape"
    )
    tau = f64(cal["tau"], (3,), "tau")
    require(np.array_equal(tau, [0.005, 0.01, 0.04]), "tau differs")
    membership = i64(cal["membership"], (n,), "membership")
    fixed = i64(cal["fixed_muscles"], (19,), "fixed")
    pool = f64(cal["pool"], (m, n), "pool")
    slopes = f64(cal["slopes"], (m, n, 3, 2), "slopes")
    kappa = f64(cal["kappa"], (n,), "kappa")
    require(
        np.array_equal(np.argmax(pool, axis=0), membership)
        and np.all(np.count_nonzero(pool, axis=0) == 1),
        "pool owner",
    )
    require(
        np.all(pool >= 0) and np.allclose(pool.sum(axis=1), 1, atol=1e-15, rtol=0),
        "pool weights",
    )
    require(np.all(slopes >= 0) and np.all(kappa >= 0), "monotone parameters")
    owners = membership == np.arange(m)[:, None]
    require(
        np.all(slopes[~owners] == 0)
        and np.all(np.any(slopes > 0, axis=(2, 3)) == owners),
        "slope owner",
    )
    require(
        len(np.unique(fixed)) == 19 and np.all((fixed >= 0) & (fixed < m)), "fixed rows"
    )
    canonical = np.zeros_like(slopes)
    canonical[:, :, 1, 0] = pool
    canonical[:, :, 1, 1] = pool
    require(np.array_equal(slopes[fixed], canonical[fixed]), "fixed slopes")
    require(
        np.array_equal(
            fixed,
            np.unique(
                np.r_[
                    cal["ltm"],
                    np.flatnonzero(np.bincount(membership, minlength=m) == 1),
                ]
            ),
        ),
        "fixed union",
    )
    require(
        np.all(rate >= 0)
        and np.all(bank >= 0)
        and np.all(activation >= 0)
        and np.all(original >= 0),
        "negative saved state",
    )
    require(
        np.array_equal(bank[0], cal["initial_bank"])
        and np.array_equal(activation[0], cal["initial_activation"])
        and np.array_equal(original[0], cal["initial_original_activation"]),
        "anchor state",
    )
    require(
        np.all(bank[0] == 0)
        and np.all(activation[0] == 0)
        and np.all(original[0] == 0),
        "nonzero anchor",
    )
    alpha = -np.expm1(-0.005 / tau)
    bank_residual = check_residual(
        bank[:-1] + alpha[None, None, :] * (rate[1:, :, None] - bank[:-1]),
        bank[1:],
        "bank",
    )
    # The original adapter sums pool-weighted rates, then applies tau=.01;
    # linearity gives the same per-MN filtered contribution below.
    original_expected = original[:-1] + alpha[1] * (rate[1:] @ pool.T - original[:-1])
    original_residual = check_residual(
        original_expected, original[1:], "original activation"
    )
    by_mn = np.empty((t, n), dtype=np.float64)
    low = slopes[membership, np.arange(n), :, 0]
    high = slopes[membership, np.arange(n), :, 1]
    hinge = (high - low)[None, :, :] * np.maximum(bank - kappa[None, :, None], 0)
    by_mn[:] = np.sum(low[None] * bank + hinge, axis=2)
    fixed_mn = np.isin(membership, fixed)
    by_mn[:, fixed_mn] = (
        pool[membership[fixed_mn], np.flatnonzero(fixed_mn)][None, :]
        * bank[:, fixed_mn, 1]
    )
    rebuilt = np.zeros_like(activation)
    for muscle in range(m):
        rebuilt[:, muscle] = by_mn[:, membership == muscle].sum(axis=1)
    activation_residual = check_residual(rebuilt, activation, "activation")
    check_residual(activation[:, fixed], original[:, fixed], "fixed original")
    return by_mn, {
        "bank": bank_residual,
        "original_activation": original_residual,
        "activation": activation_residual,
    }


def native_sign_bins(force, load, slip, slip_valid, time_s, original_time):
    """Mask native samples before 50-sample sums; retain exact zeros."""
    v, legs = force.shape
    require(
        legs == 6
        and v % 50 == 0
        and load.shape == force.shape
        and slip.shape == force.shape
        and slip_valid.shape == force.shape,
        "native shape",
    )
    require(
        time_s.shape == (v,)
        and original_time.shape == (v,)
        and np.max(np.abs(time_s - np.arange(1, v + 1) * 0.0001)) <= 2e-8
        and np.max(np.abs(time_s - original_time)) <= 2e-8,
        "native clock",
    )
    require(
        np.isfinite(force).all() and np.isfinite(load).all() and np.all(load >= 0),
        "native force/load",
    )
    require(
        slip_valid.dtype == np.bool_
        and np.all(slip_valid[load >= 0.02] & np.isfinite(slip[load >= 0.02])),
        "supported slip missing",
    )
    support = load >= 0.02
    masks = np.stack(
        (support & (force < 0), support & (force == 0), support & (force > 0)), axis=-1
    )
    shape = (v // 50, 50, 6, 3)
    count = masks.reshape(shape).sum(axis=1)
    signed = np.where(masks, force[..., None], 0).reshape(shape).sum(axis=1)
    normal = np.where(masks, load[..., None], 0).reshape(shape).sum(axis=1)
    magnitude = np.where(masks[..., 0], -force, 0).reshape(v // 50, 50, 6).sum(axis=1)
    require(
        np.array_equal(count.sum(axis=2), support.reshape(v // 50, 50, 6).sum(axis=1)),
        "native count conservation",
    )
    require(
        np.max(
            np.abs(
                signed.sum(axis=2)
                - np.where(support, force, 0).reshape(v // 50, 50, 6).sum(axis=1)
            )
        )
        <= 1e-12,
        "native force conservation",
    )
    return {
        "count": np.pad(count, ((1, 0), (0, 0), (0, 0))),
        "signed_force_sum_bw": np.pad(signed, ((1, 0), (0, 0), (0, 0))),
        "normal_load_sum_bw": np.pad(normal, ((1, 0), (0, 0), (0, 0))),
        "negative_magnitude_sum_bw": np.pad(magnitude, ((1, 0), (0, 0))),
    }


def validate_force_semantics(force, force_metadata, expected_body_hash):
    inventory = force_metadata["inventory"]
    mass = np.asarray(inventory["body_mass"], dtype=float)
    gravity = f64(np.asarray(inventory["gravity"], dtype=float), (3,), "gravity")
    ids = np.asarray(inventory["body_ids"])
    require(
        ids.ndim == 1
        and ids.dtype.kind in "iu"
        and len(ids) == len(mass)
        and len(np.unique(ids)) == len(ids)
        and np.all(ids > 0),
        "fly body inventory",
    )
    require(
        np.isfinite(mass).all()
        and np.all(mass >= 0)
        and mass.sum() > 0
        and np.linalg.norm(gravity) > 0,
        "mass/gravity",
    )
    weight = float(inventory["weight0"])
    require(
        np.isfinite(weight)
        and weight > 0
        and np.isclose(mass.sum(), inventory["body_mass_total"], atol=1e-12, rtol=1e-10)
        and np.isclose(
            mass.sum(), inventory["body_mass_reported"], atol=1e-12, rtol=1e-10
        )
        and np.isclose(
            weight, mass.sum() * np.linalg.norm(gravity), atol=1e-12, rtol=1e-10
        ),
        "body weight identity",
    )
    require(
        inventory["body_model_hash"] == expected_body_hash
        and inventory["leg_order"] == list(LEGS),
        "force body identity",
    )
    require(
        force_metadata["schema"] == "flylab.m3.e185.ground-reaction-metadata.v1"
        and force_metadata["force_samples"]
        == force_metadata["original_native_samples"]
        == 30000
        and force_metadata["primary_error"] is None
        and force_metadata["cleanup_errors"] == []
        and force_metadata["raw_persistence_errors"] == [],
        "force metadata validity",
    )
    require(
        force["available"].shape
        == force["finite"].shape
        == force["error"].shape
        == (30000,)
        and force["available"].dtype == force["finite"].dtype == np.bool_
        and np.all(force["available"] & force["finite"])
        and np.all(force["error"] == ""),
        "force error flags",
    )
    for bw in (name for name in force if name.endswith("_bw")):
        original = bw[:-3]
        require(original in force, "missing physical force " + original)
        check_residual(force[original] / weight, force[bw], "force/BW " + original)
    rotation = f64(force["thorax_rotation"], (30000, 3, 3), "native rotation")
    require(
        np.max(np.abs(np.einsum("tji,tjk->tik", rotation, rotation) - np.eye(3)))
        <= 1e-8,
        "native rotation orthogonality",
    )
    forward = f64(force["forward"], (30000, 3), "native forward")
    check_residual(rotation[:, :, 0], forward, "native forward", 1e-8)
    planar = forward.copy()
    planar[:, 2] = 0
    norms = np.linalg.norm(planar, axis=1)
    require(np.all(norms > 1e-12), "native heading missing")
    planar /= norms[:, None]
    check_residual(planar, force["heading_xy"], "native planar heading", 1e-8)
    for prefix, vector, count, segment in (
        ("leg", "leg_force", 5, "segment_force"),
        ("all_leg", "all_leg_force", 8, "all_leg_segment_force"),
    ):
        f64(force[vector], (30000, 6, 3), vector)
        check_residual(
            force[segment].reshape(30000, 6, count, 3).sum(axis=2),
            force[vector],
            prefix + " segment sum",
        )
        for projection, axis in (("forward", forward), ("heading_xy", planar)):
            check_residual(
                np.einsum("tlj,tj->tl", force[vector], axis),
                force[prefix + "_" + projection + "_force"],
                prefix + " " + projection,
            )
    check_residual(
        force["all_leg_force"].sum(axis=1) + force["nonleg_floor_force_residual"],
        force["full_fly_floor_force"],
        "fly force sum",
    )
    for projection, axis in (("forward", forward), ("heading_xy", planar)):
        check_residual(
            np.einsum("tj,tj->t", force["full_fly_floor_force"], axis),
            force["full_fly_" + projection + "_force"],
            "fly " + projection,
        )
    return {
        "body_weight": weight,
        "mass_total": float(mass.sum()),
        "gravity_norm": float(np.linalg.norm(gravity)),
    }


def conditional_mean(values, count, magnitude, legs_for_columns=None):
    """Rows x features and rows x legs weights; NaNs remove their weight."""
    values = np.asarray(values, dtype=float)
    count = np.asarray(count, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    require(
        values.ndim == 2
        and count.shape == (len(values), 6)
        and magnitude.shape == count.shape,
        "conditional shape",
    )
    require(
        legs_for_columns is not None and len(legs_for_columns) == values.shape[1],
        "feature ownership",
    )
    weights = count[:, legs_for_columns]
    magweights = magnitude[:, legs_for_columns]
    finite = np.isfinite(values)
    effective = np.where(finite, weights, 0)
    numerator = np.sum(effective * np.where(finite, values, 0), axis=0)
    denominator = effective.sum(axis=0)
    mean = np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )
    effective_mag = np.where(finite, magweights, 0)
    mag_denominator = effective_mag.sum(axis=0)
    mag_mean = np.divide(
        np.sum(effective_mag * np.where(finite, values, 0), axis=0),
        mag_denominator,
        out=np.full(values.shape[1], np.nan),
        where=mag_denominator > 0,
    )
    return mean, denominator, mag_mean, mag_denominator


def projected_foot_motion(position, basis, feet):
    require(
        position.shape[1:] == (3,)
        and basis.shape == (len(position), 3, 3)
        and feet.shape == (len(position), 6, 3),
        "body shape",
    )
    require(
        np.isfinite(position).all()
        and np.isfinite(basis).all()
        and np.isfinite(feet).all(),
        "body finite",
    )
    require(
        np.max(np.abs(np.einsum("tji,tjk->tik", basis, basis) - np.eye(3))) <= 1e-8,
        "body basis",
    )
    heading = basis[:, :, 0].copy()
    heading[:, 1] = 0
    norms = np.linalg.norm(heading, axis=1)
    require(np.all(norms > 1e-12), "missing planar heading")
    heading /= norms[:, None]
    midpoint = heading[1:] + heading[:-1]
    midpoint_norm = np.linalg.norm(midpoint, axis=1)
    require(np.all(midpoint_norm > 1e-12), "opposed interval headings")
    midpoint /= midpoint_norm[:, None]
    world = np.full((len(position), 6), np.nan)
    world[1:] = np.einsum("tlj,tj->tl", np.diff(feet, axis=0) / 0.005, midpoint)
    relative = np.einsum("tlj,tj->tl", feet - position[:, None, :], heading)
    relative_rate = np.full_like(world, np.nan)
    relative_rate[1:] = np.diff(relative, axis=0) / 0.005
    return heading, world, relative, relative_rate


def joint_motion(target, actual, contribution, row_leg):
    t = len(target)
    require(
        target.shape == actual.shape == (t, 42) and contribution.shape == (t, 96, 42),
        "joint shape",
    )
    measured = np.full_like(actual, np.nan)
    measured[1:] = np.diff(actual, axis=0) / 0.005
    target_rate = np.full_like(target, np.nan)
    target_rate[2:] = np.diff(target, axis=0)[1:] / 0.005
    contribution_rate = np.full_like(contribution, np.nan)
    contribution_rate[2:] = np.diff(contribution, axis=0)[1:] / 0.005
    start_discrepancy = np.full_like(target, np.nan)
    start_discrepancy[1:] = target[1:] - actual[:-1]
    cosine = np.full((t, 6), np.nan)
    for leg in range(6):
        rows = row_leg == leg
        a, b = target_rate[:, rows], measured[:, rows]
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        denom = norm_a * norm_b
        valid = (norm_a > 1e-12) & (norm_b > 1e-12) & np.isfinite(denom)
        np.divide(np.sum(a * b, axis=1), denom, out=cosine[:, leg], where=valid)
    return {
        "joint_measured_velocity_rad_s": measured,
        "joint_target_rate_rad_s": target_rate,
        "joint_contribution_rate_rad_s": contribution_rate,
        "joint_tracking_error_rad": target - actual,
        "joint_start_discrepancy_rad": start_discrepancy,
        "joint_alignment_cosine": cosine,
    }


def event_indices(position, basis):
    # E184: np.gradient thorax in UI, dot body-forward, symmetric Gaussian 20ms.
    velocity = np.gradient(position, 0.005, axis=0)
    speed = gaussian_filter1d(
        np.sum(velocity * basis[:, :, 0], axis=1), 4.0, mode="nearest"
    )
    require(speed.shape == (601,) and np.isfinite(speed).all(), "forward speed")
    events = []
    previous_end = -1
    for i in range(100, 600):
        if speed[i - 1] > 0.5 and speed[i] <= 0.5:
            first, last = i - 20, i + 20
            reason = (
                "edge_window"
                if first < 100 or last >= 600
                else "overlap_selected_window"
                if first <= previous_end
                else None
            )
            if reason is None:
                previous_end = last
            events.append(
                {"index": i, "selected": reason is None, "exclusion_reason": reason}
            )
    return speed, events


def _windows(events):
    selected = [e["index"] for e in events if e["selected"]]
    windows = {
        "whole": np.arange(1, 601),
        "evaluation": np.arange(100, 600),
        "startup99": np.arange(1, 100),
        "selected8": np.concatenate([np.arange(i - 20, i + 21) for i in selected]),
    }
    for label, (a, b) in BANDS.items():
        windows["selected8:" + label] = np.concatenate(
            [np.arange(i + a, i + b) for i in selected]
        )
    for e in events:
        i = e["index"]
        for label, (a, b) in BANDS.items():
            windows[f"event{i}:{label}"] = np.arange(i + a, i + b)
    return windows


def aggregate(windows, bins, signals, owners):
    labels = list(windows)
    w, l, s = len(labels), 6, 3
    arrays = {
        "window_labels": np.asarray(labels),
        "sign_labels": np.asarray(SIGNS),
        "leg_labels": np.asarray(LEGS),
    }
    arrays["condition_native_count"] = np.zeros((w, l, s), dtype=np.int64)
    arrays["condition_signed_force_sum_bw"] = np.zeros((w, l, s))
    arrays["condition_normal_load_sum_bw"] = np.zeros((w, l, s))
    arrays["condition_negative_magnitude_sum_bw"] = np.zeros((w, l, s))
    for name in signals:
        feature = signals[name].shape[1]
        arrays[f"mean_{name}"] = np.full((w, s, feature), np.nan)
        arrays[f"valid_native_count_{name}"] = np.zeros((w, s, feature))
        arrays[f"negative_magnitude_mean_{name}"] = np.full((w, feature), np.nan)
        arrays[f"negative_magnitude_weight_{name}"] = np.zeros((w, feature))
    for wi, rows in enumerate(windows.values()):
        count = bins["count"][rows]
        arrays["condition_native_count"][wi] = count.sum(axis=0)
        arrays["condition_signed_force_sum_bw"][wi] = bins["signed_force_sum_bw"][
            rows
        ].sum(axis=0)
        arrays["condition_normal_load_sum_bw"][wi] = bins["normal_load_sum_bw"][
            rows
        ].sum(axis=0)
        arrays["condition_negative_magnitude_sum_bw"][wi, :, 0] = bins[
            "negative_magnitude_sum_bw"
        ][rows].sum(axis=0)
        for name, value in signals.items():
            for si in range(3):
                mean, valid, magmean, magvalid = conditional_mean(
                    value[rows],
                    count[:, :, si],
                    bins["negative_magnitude_sum_bw"][rows]
                    if si == 0
                    else np.zeros_like(count[:, :, si]),
                    owners[name],
                )
                arrays[f"mean_{name}"][wi, si] = mean
                arrays[f"valid_native_count_{name}"][wi, si] = valid
                if si == 0:
                    arrays[f"negative_magnitude_mean_{name}"][wi] = magmean
                    arrays[f"negative_magnitude_weight_{name}"][wi] = magvalid
    return arrays


def analyze(metadata):
    base = safe_path(metadata["baseline_directory"])
    start = base / "start"
    cal = {
        name: np.load(safe_path(path), allow_pickle=False)
        for name, path in metadata["fields"].items()
    }
    for name, shape in metadata["shapes"].items():
        (i64 if name in {"membership", "fixed_muscles", "motor_ids", "ltm"} else f64)(
            cal[name], tuple(shape), name
        )
    state = read_json(start / "state.json")
    manifest = read_json(start / "manifest.json")["files"]
    adapter = state["adapter"]
    patch = state["monotone_recruitment_patch"]
    require(
        adapter["schema"] == "flylab.m3-monotone-recruitment-adapter.v1"
        and adapter["hash"] == metadata["expected"]["adapter_hash"]
        and patch["adapter_identity"] == adapter["hash"],
        "adapter identity",
    )
    require(
        state["control_tick"] == state["network"]["tick"] == 0
        and state["network"]["dt"] == 0.00025,
        "anchor clock",
    )
    require(
        manifest["state.json"]
        == metadata["inputs"][(start / "state.json").relative_to(ROOT).as_posix()],
        "state manifest",
    )
    for name, path in metadata["fields"].items():
        require(
            manifest[Path(path).name] == metadata["inputs"][path],
            "field manifest " + name,
        )
    for name in ("tau", "kappa", "slopes", "membership", "fixed_muscles"):
        require(
            adapter["parameters"][name]
            == {"__npy__": Path(metadata["fields"][name]).name},
            "adapter parameter ref " + name,
        )
    for name in ("moment", "rest", "pool", "motor_ids", "ltm"):
        require(
            adapter["calibration"][name]
            == {"__npy__": Path(metadata["fields"][name]).name},
            "calibration ref " + name,
        )
    for name in ("tau", "kappa", "slopes", "membership", "fixed_muscles", "motor_ids"):
        ref = patch[name]["__npy__"]
        require(
            manifest[ref] == metadata["inputs"][metadata["fields"][name]]
            and digest(start / ref) == digest(safe_path(metadata["fields"][name])),
            "patch copy " + name,
        )
    groups = adapter["groups"]
    roster = adapter["parameters"]["motor_roster"]
    require(
        len(groups) == 96
        and len(roster) == 391
        and roster == patch["motor_roster"]
        and len(set(roster)) == 391
        and all(isinstance(name, str) and name for name in roster),
        "anatomical roots",
    )
    require(
        np.array_equal(cal["motor_ids"], np.sort(np.unique(cal["motor_ids"]))),
        "sorted unique motor IDs",
    )
    membership = cal["membership"]
    muscle_leg = np.array([LEGS.index(g["leg"]) for g in groups], dtype=np.int64)
    require(tuple(dict.fromkeys(g["leg"] for g in groups)) == LEGS, "muscle leg order")
    for m, group in enumerate(groups):
        require(
            group["columns"] == np.flatnonzero(membership == m).tolist()
            and isinstance(group["muscle"], str)
            and group["muscle"],
            "group membership",
        )
    mn_leg = muscle_leg[membership]
    joint_info = read_json(safe_path(metadata["joint_path"]))
    row_leg = np.asarray(joint_info["joint_row_leg_index"], dtype=np.int64)
    require(
        row_leg.shape == (42,)
        and all(np.count_nonzero(row_leg == i) == 7 for i in range(6))
        and row_leg[30] == 4
        and joint_info["zero_moment_rows"] == [30],
        "joint order",
    )
    joint_names = joint_info["joint_name_suffixes"]
    require(
        len(joint_names) == 42 and joint_names[30] == joint_info["zero_row_name"],
        "joint names",
    )
    moment = cal["moment"]
    require(
        np.array_equal(np.flatnonzero(np.all(moment == 0, axis=1)), [30]),
        "zero moment row",
    )
    require(
        np.all((moment != 0) <= (row_leg[:, None] == muscle_leg[None, :])),
        "moment crosses leg",
    )
    with np.load(base / "native-recruitment-state.npz", allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    for key, shape in {
        "motor_rate": (601, 391),
        "bank": (601, 391, 3),
        "activation": (601, 96),
        "original_activation": (601, 96),
    }.items():
        f64(rec[key], shape, key)
        require(np.all(rec[key + "_available"]), key + " unavailable")
    require(
        np.array_equal(rec["control_tick"], np.arange(601))
        and np.array_equal(rec["network_tick"], np.arange(601) * 20)
        and np.max(np.abs(rec["physics_time_s"] - np.arange(601) * 0.005)) <= 2e-8
        and np.all(rec["complete"]),
        "recruitment clock",
    )
    by_mn, residuals = reconstruct(
        rec["motor_rate"],
        rec["bank"],
        rec["activation"],
        rec["original_activation"],
        cal,
    )
    trace = read_json(base / "trace.json")["samples"]
    require(len(trace) == 601, "trace rows")
    for i, row in enumerate(trace):
        require(
            row["control_tick"] == i and abs(row["time_s"] - i * 0.005) <= 2e-8,
            "trace clock",
        )
        interval = row["command_interval"]
        require(
            interval is None
            if i == 0
            else interval
            == {
                "start_tick": i - 1,
                "end_tick": i,
                "start_s": (i - 1) * 0.005,
                "end_s": i * 0.005,
            },
            "command interval",
        )
    target = np.asarray([r["joint_targets_rad"] for r in trace], dtype=np.float64)
    actual = np.asarray([r["joint_angles_rad"] for r in trace], dtype=np.float64)
    position = np.asarray([r["body"]["position"] for r in trace], dtype=np.float64)
    basis = np.asarray([r["body"]["basis"] for r in trace], dtype=np.float64)
    feet = np.asarray(
        [[r["body"]["legs"][leg][-1] for leg in LEGS] for r in trace], dtype=np.float64
    )
    f64(target, (601, 42), "target")
    f64(actual, (601, 42), "actual")
    adhesion = np.asarray([r["adhesion"] for r in trace])
    require(
        adhesion.shape == (601, 6)
        and adhesion.dtype == np.bool_
        and not adhesion[1:].any(),
        "recorded original adhesion path",
    )
    tendon = np.asarray(
        [[r["tendon_inputs"][leg + "_tarsus"] for leg in LEGS] for r in trace],
        dtype=np.float64,
    )
    f64(tendon, (601, 6), "tendon inputs")
    ltm = cal["ltm"]
    require([groups[int(m)]["leg"] for m in ltm] == list(LEGS), "LTM leg order")
    check_residual(
        0.2 * np.tanh(rec["activation"][1:, ltm] / 10), tendon[1:], "LTM tendon formula"
    )
    check_residual(
        np.asarray([r["motor_rates"] for r in trace], dtype=np.float64),
        rec["motor_rate"],
        "trace motor",
    )
    contribution = 0.02 * rec["activation"][:, :, None] * moment.T[None]
    residuals["target"] = check_residual(
        cal["rest"][None] + contribution[1:].sum(axis=1), target[1:], "target"
    )
    motion = joint_motion(target, actual, contribution, row_leg)
    heading, foot_world, foot_relative, foot_relative_rate = projected_foot_motion(
        position, basis, feet
    )
    speed, events = event_indices(position, basis)
    prior = read_json(safe_path(metadata["prior_events_path"]))
    saved_events = prior["arms"]["baseline"]["events"]
    require(
        events
        == [
            {
                "index": e["index"],
                "selected": e["selected"],
                "exclusion_reason": e["exclusion_reason"],
            }
            for e in saved_events
        ]
        and [e["index"] for e in events if e["selected"]]
        == metadata["expected"]["selected_events"]
        and len(events) == 11,
        "E184 event selection changed",
    )
    with np.load(base / "native-contact.npz", allow_pickle=False) as z:
        native = {k: z[k] for k in z.files}
    with np.load(safe_path(metadata["force_path"]), allow_pickle=False) as z:
        force = {k: z[k] for k in z.files}
    force_semantics = validate_force_semantics(
        force,
        read_json(safe_path(metadata["force_metadata_path"])),
        metadata["expected"]["body_hash"],
    )
    native_time = f64(native["time_s"], (30000,), "native time")
    force_time = f64(force["time_s"], (30000,), "force time")
    require(
        np.array_equal(force["available"], np.ones(30000, dtype=bool))
        and np.array_equal(force["finite"], np.ones(30000, dtype=bool)),
        "force availability",
    )
    f64(force["all_leg_heading_xy_force_bw"], (30000, 6), "force heading")
    f64(force["all_leg_force_bw"], (30000, 6, 3), "force vectors")
    f64(force["heading_xy"], (30000, 3), "native headings")
    require(
        np.max(np.abs(np.linalg.norm(force["heading_xy"], axis=1) - 1)) <= 1e-8
        and np.max(np.abs(force["heading_xy"][:, 2])) <= 1e-8,
        "native heading",
    )
    require(
        np.max(
            np.abs(
                np.einsum("tlj,tj->tl", force["all_leg_force_bw"], force["heading_xy"])
                - force["all_leg_heading_xy_force_bw"]
            )
        )
        <= 1e-10,
        "heading force projection",
    )
    require(
        np.max(
            np.abs(
                force["all_leg_segment_force_bw"].reshape(30000, 6, 8, 3).sum(axis=2)
                - force["all_leg_force_bw"]
            )
        )
        <= 1e-10,
        "leg segment sum",
    )
    require(
        np.max(
            np.abs(
                force["all_leg_force_bw"].sum(axis=1)
                + force["nonleg_floor_force_residual_bw"]
                - force["full_fly_floor_force_bw"]
            )
        )
        <= 1e-10,
        "fly force sum",
    )
    bins = native_sign_bins(
        force["all_leg_heading_xy_force_bw"],
        native["normal_load_bw"],
        native["slip_mm_s"],
        native["slip_valid"],
        force_time,
        native_time,
    )
    startup = np.asarray([r["joint_targets_rad"] for r in trace[1:100]], dtype=float)
    with np.load(base / "actual-startup-original99-raw.npz", allow_pickle=False) as z:
        startup_keys = z.files
        require(
            "actual_command" in startup_keys and "actual_LTM" in startup_keys,
            "startup raw keys",
        )
        check_residual(z["actual_command"], startup, "startup command")
        check_residual(z["actual_LTM"], tendon[1:100], "startup LTM")
    # Rows are command intervals. Row 0 has no native observations.
    arrays = {
        "control_time_s": np.arange(601) * 0.005,
        "native_interval_end_control_row": np.arange(1, 601),
        "motor_ids": cal["motor_ids"],
        "motor_roots_391": np.asarray(roster),
        "muscle_names_96": np.asarray([g["muscle"] for g in groups]),
        "joint_names_42": np.asarray(joint_names),
        "mn_membership": membership,
        "mn_leg_index": mn_leg,
        "muscle_leg_index": muscle_leg,
        "joint_row_leg_index": row_leg,
        "ltm_muscle_index_6": ltm,
        "motor_rate_391": rec["motor_rate"],
        "mn_activation_contribution_391": by_mn,
        "muscle_activation_96": rec["activation"],
        "joint_target_rad_42": target,
        "joint_actual_rad_42": actual,
        "joint_muscle_contribution_rad_96_42": contribution,
        "adhesion_6": adhesion,
        "tendon_input_6": tendon,
        "foot_world_forward_mm_s": foot_world,
        "foot_relative_forward_mm": foot_relative,
        "foot_relative_forward_mm_s": foot_relative_rate,
        "body_heading_ui": heading,
        "smoothed_body_forward_mm_s": speed,
        "event_indices_all": np.array([e["index"] for e in events]),
        "event_indices_selected": np.array(
            [e["index"] for e in events if e["selected"]]
        ),
    }
    arrays.update(motion)
    arrays.update({"native_" + k: v for k, v in bins.items()})
    mn_change = np.full_like(by_mn, np.nan)
    mn_change[2:] = np.diff(by_mn, axis=0)[1:] / 0.005
    rate_change = np.full_like(rec["motor_rate"], np.nan)
    rate_change[2:] = np.diff(rec["motor_rate"], axis=0)[1:] / 0.005
    muscle_change = np.full_like(rec["activation"], np.nan)
    muscle_change[2:] = np.diff(rec["activation"], axis=0)[1:] / 0.005
    one_leg = np.arange(6, dtype=np.int64)
    rel_positive = np.where(
        np.isfinite(foot_relative_rate), (foot_relative_rate > 0).astype(float), np.nan
    )
    align_positive = np.where(
        np.isfinite(motion["joint_alignment_cosine"]),
        (motion["joint_alignment_cosine"] > 0).astype(float),
        np.nan,
    )
    simultaneous = np.where(
        np.isfinite(rel_positive) & np.isfinite(align_positive),
        rel_positive * align_positive,
        np.nan,
    )
    signals = {
        "mn_rate_391": rec["motor_rate"],
        "mn_rate_change_391": rate_change,
        "mn_contribution_391": by_mn,
        "mn_contribution_rate_391": mn_change,
        "muscle_activation_96": rec["activation"],
        "muscle_activation_rate_96": muscle_change,
        "joint_target_42": target,
        "joint_actual_42": actual,
        "joint_measured_velocity_42": motion["joint_measured_velocity_rad_s"],
        "joint_target_rate_42": motion["joint_target_rate_rad_s"],
        "joint_tracking_error_42": motion["joint_tracking_error_rad"],
        "joint_start_discrepancy_42": motion["joint_start_discrepancy_rad"],
        "adhesion_6": adhesion.astype(float),
        "tendon_input_6": tendon,
        "foot_world_forward_6": foot_world,
        "foot_relative_forward_6": foot_relative_rate,
        "joint_alignment_cosine_6": motion["joint_alignment_cosine"],
        "forward_relative_fraction_6": rel_positive,
        "alignment_fraction_6": align_positive,
        "simultaneous_fraction_6": simultaneous,
    }
    owners = {
        name: mn_leg
        if name.endswith("391")
        else muscle_leg
        if name.endswith("96")
        else row_leg
        if name.endswith("42")
        else one_leg
        for name in signals
    }
    # Joint-specific muscle contributions retain both joint and muscle indices.
    q_flat = contribution.transpose(0, 2, 1).reshape(601, 42 * 96)
    dq_flat = (
        motion["joint_contribution_rate_rad_s"].transpose(0, 2, 1).reshape(601, 42 * 96)
    )
    signals["joint_muscle_contribution_42x96"] = q_flat
    signals["joint_muscle_contribution_rate_42x96"] = dq_flat
    owners["joint_muscle_contribution_42x96"] = np.repeat(row_leg, 96)
    owners["joint_muscle_contribution_rate_42x96"] = np.repeat(row_leg, 96)
    windows = _windows(events)
    aggregate_arrays = aggregate(windows, bins, signals, owners)
    arrays.update(aggregate_arrays)
    summary = {
        "schema": "flylab.m3.e186.force-command.analysis.v1",
        "selected_events": metadata["expected"]["selected_events"],
        "events": events,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "legs": LEGS,
        "signs": SIGNS,
        "force_inventory": force_semantics,
        "motor_roots": roster,
        "muscle_names": [g["muscle"] for g in groups],
        "joint_names": joint_names,
        "windows": [
            {
                "label": label,
                "control_rows": len(rows),
                "repeated_one_trajectory": label.startswith("selected8"),
            }
            for label, rows in windows.items()
        ],
        "reconstruction_max_abs": residuals,
        "native_counts_by_condition": aggregate_arrays[
            "condition_native_count"
        ].tolist(),
        "negative_magnitude_sum_bw": aggregate_arrays[
            "condition_negative_magnitude_sum_bw"
        ].tolist(),
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "interpretation": "saved single-trajectory temporal agreement only; no causal A/B verdict, candidate adoption, physical muscle force, torque, joint power, or natural gait success",
        "limitations": [
            "requested target may exceed actuator limits",
            "tarsus5 origin is not the contact point",
            "body-relative coordinate rate includes changing heading",
            "Gaussian event speed uses future samples",
            "native force is post-integration shadow instantaneous force, not solver impulse",
            "selected events overlap one trajectory and are not independent",
        ],
    }
    compact = {}
    for name in (
        "forward_relative_fraction_6",
        "alignment_fraction_6",
        "simultaneous_fraction_6",
        "foot_relative_forward_6",
        "joint_alignment_cosine_6",
    ):
        mean = aggregate_arrays["mean_" + name]
        valid = aggregate_arrays["valid_native_count_" + name]
        compact[name] = {
            label: {
                sign: [
                    {
                        "mean": None
                        if not np.isfinite(mean[wi, si, leg])
                        else float(mean[wi, si, leg]),
                        "valid_native_count": int(valid[wi, si, leg]),
                    }
                    for leg in range(6)
                ]
                for si, sign in enumerate(SIGNS)
            }
            for wi, label in enumerate(windows)
        }
    summary["leg_condition_summaries"] = compact
    return summary, arrays


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def execute(freeze_path, review_path, out):
    began = time.perf_counter()
    require(out.resolve() == safe_path(OUTPUT).resolve(), "fixed output required")
    require(not out.exists() and out.parent.is_dir(), "output exists or parent missing")
    out.mkdir(exist_ok=False)
    terminal = {
        "schema": "flylab.m3.e186.force-command.terminal.v1",
        "status": "failed",
        "attempted": True,
        "native_runs": 0,
        "model_constructions": 0,
        "neural_steps": 0,
        "physics_steps": 0,
        "gpu_count": 0,
        "inherited_cumulative_wall_s": PARAMETERS["inherited_cumulative_wall_s"],
        "supervisor_wall_limit_s": 45,
    }
    try:
        metadata, freeze_sha = gate(freeze_path, review_path, out)
        terminal.update(
            freeze_sha256=freeze_sha, source_review_sha256=digest(review_path)
        )
        with threadpool_limits(limits=1):
            summary, arrays = analyze(metadata)
        require(
            time.perf_counter() - began <= PARAMETERS["internal_wall_s"],
            "internal wall budget",
        )
        write_json_new(out / "analysis.json", summary)
        np.savez_compressed(out / "aligned.npz", **arrays)
        require(
            time.perf_counter() - began <= PARAMETERS["internal_wall_s"],
            "internal wall budget",
        )
        terminal.update(
            status="complete",
            output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
            array_count=len(arrays),
        )
    except Exception as exc:
        terminal["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        terminal["elapsed_wall_s"] = time.perf_counter() - began
        write_json_new(out / "terminal.json", terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    execute(args.freeze, args.source_review, args.out)


if __name__ == "__main__":
    main()
