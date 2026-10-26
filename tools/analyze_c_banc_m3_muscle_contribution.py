"""E179 saved-data muscle contribution audit; no model or simulation imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "9cfe48ce51202b52e7d062269abf57b7ca0a872be9c6d9d9246c1b66663231ce"
SCHEMA = "flylab.m3.e179.muscle-contribution.v1"
OWNED = (
    "tools/analyze_c_banc_m3_muscle_contribution.py",
    "tests/test_c_banc_muscle_contribution.py",
)
OUTPUTS = (
    "contributions.npz",
    "summary.json",
    "events.json",
    "input-receipt.json",
    "result.json",
)


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(relative: str) -> Path:
    require(
        isinstance(relative, str) and relative and "\\" not in relative,
        "Invalid frozen path",
    )
    path = (ROOT / relative).resolve(strict=True)
    require(
        path.is_relative_to(ROOT.resolve()) and path.is_file(),
        "Frozen path escapes workspace",
    )
    return path


def verify_hash_map(entries: dict[str, str], label: str) -> None:
    require(isinstance(entries, dict) and entries, f"Missing {label} inventory")
    for name, digest in entries.items():
        require(isinstance(digest, str) and len(digest) == 64, f"Malformed {label} SHA")
        require(sha256(source_path(name)) == digest, f"{label} hash mismatch: {name}")


def verified_gate(
    freeze_path: Path, freeze_sha256: str, review_path: Path, review_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """No original value is opened before this exact source and input gate."""
    require(
        freeze_sha256 == FREEZE_SHA and sha256(freeze_path) == FREEZE_SHA,
        "E179 freeze SHA mismatch",
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == SCHEMA and freeze.get("owned_files") == list(OWNED),
        "E179 freeze schema/ownership mismatch",
    )
    require(
        sha256(source_path(freeze["design_path"])) == freeze["design_sha256"],
        "E179 design changed",
    )
    c = freeze["contract"]
    require(
        c
        == {
            "control_rows": 2001,
            "motor_cells": 391,
            "muscles": 96,
            "joint_targets": 42,
            "control_dt_s": 0.005,
            "neural_steps_per_control": 20,
            "native_rows": 100000,
            "native_dt_s": 0.0001,
            "native_steps_per_control": 50,
            "clock_atol_s": 1e-8,
            "reconstruction_atol": 1e-10,
            "tau": [0.005, 0.01, 0.04],
            "rm_group": 74,
            "rm_leg_index": 4,
            "legs": ["lf", "lm", "lh", "rf", "rm", "rh"],
            "silent_post_ids": [108052, 153939],
            "support_threshold_bw": 0.02,
            "pre_support_steps": 100,
            "post_nonsupport_steps": 100,
            "event_half_window_steps": 200,
            "windows": {"whole": [0, 10], "startup": [0, 1], "late": [1, 10]},
        },
        "E179 analysis contract changed",
    )
    for key, value in {
        "internal_wall_limit_s": 90,
        "external_wall_limit_s": 105,
        "supervisor_limit_s": 120,
        "maximum_raw_bytes": 134217728,
        "minimum_free_disk_bytes": 1073741824,
        "cpu_threads": 1,
        "gpu_count": 0,
        "model_constructions": 0,
        "neural_steps": 0,
        "physics_steps": 0,
        "fit_calls": 0,
        "original_processes": 1,
        "automatic_retries": 0,
        "original_values_loaded_at_freeze": 0,
        "F6_exposed": False,
        "active_UI_changed": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
    }.items():
        require(
            freeze.get(key) == value, f"E179 resource/outcome contract differs: {key}"
        )
    require(
        len(freeze["input_sha256"]) == 40 and len(freeze["reused_source_sha256"]) == 9,
        "E179 inventory size changed",
    )
    require(
        len(review_sha256) == 64 and sha256(review_path) == review_sha256,
        "Exact E179 source review SHA required",
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    require(
        review.get("schema") == freeze["source_gate_schema"],
        "E179 source gate schema differs",
    )
    require(
        review.get("freeze_sha256") == FREEZE_SHA, "E179 source gate freeze differs"
    )
    require(
        review.get("accepts_muscle_contribution") is True
        and review.get("blocking_defects") == [],
        "E179 source gate blocks analysis",
    )
    require(
        review.get("source_sha256")
        == {name: sha256(source_path(name)) for name in OWNED},
        "E179 owned source changed after review",
    )
    verify_hash_map(freeze["input_sha256"], "input")
    verify_hash_map(freeze["reused_source_sha256"], "reused source")
    return freeze, review


def f64(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    a = np.asarray(value)
    require(
        a.shape == shape and a.dtype == np.float64 and np.isfinite(a).all(),
        f"Invalid {label} shape/dtype/finite",
    )
    return a


def i64(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    a = np.asarray(value)
    require(a.shape == shape and a.dtype == np.int64, f"Invalid {label} shape/dtype")
    return a


def bank_recurrence(
    rate: np.ndarray, bank: np.ndarray, tau: np.ndarray, dt: float = 0.005
) -> np.ndarray:
    """Expected adjacent bank rows, using the saved preceding row at each step."""
    rate = np.asarray(rate)
    bank = np.asarray(bank)
    tau = np.asarray(tau)
    require(
        rate.ndim == 2 and bank.shape == (*rate.shape, 3), "Bank/rate shape differs"
    )
    f64(rate, rate.shape, "motor rate")
    f64(bank, bank.shape, "bank")
    f64(tau, (3,), "tau")
    require(np.all(tau > 0) and np.isfinite(dt) and dt > 0, "Invalid bank clock")
    alpha = -np.expm1(-dt / tau)
    return bank[:-1] + alpha[None, None, :] * (rate[1:, :, None] - bank[:-1])


def selected_contributions(
    bank: np.ndarray,
    columns: np.ndarray,
    kappa: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> dict[str, np.ndarray]:
    bank = np.asarray(bank)
    columns = np.asarray(columns)
    require(bank.ndim == 3 and bank.shape[2] == 3, "Bank must be [row,motor,filter]")
    f64(bank, bank.shape, "bank")
    require(
        columns.ndim == 1
        and np.issubdtype(columns.dtype, np.integer)
        and np.all((0 <= columns) & (columns < bank.shape[1])),
        "Invalid selected columns",
    )
    require(len(np.unique(columns)) == len(columns), "Duplicate selected columns")
    kappa = f64(kappa, (bank.shape[1],), "kappa")
    low = f64(low, (len(columns), 3), "low slope")
    high = f64(high, (len(columns), 3), "high slope")
    b = bank[:, columns, :]
    lo = low[None, :, :] * b
    hinge = (high - low)[None, :, :] * np.maximum(
        b - kappa[columns][None, :, None], 0.0
    )
    return {"low": lo, "hinge": hinge, "total": lo + hinge}


def reconstruct_activation(
    bank: np.ndarray,
    original_activation: np.ndarray,
    membership: np.ndarray,
    fixed: np.ndarray,
    kappa: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Copy fixed muscles; calculate only variable muscle monotone output."""
    bank = np.asarray(bank)
    require(bank.ndim == 3 and bank.shape[2] == 3, "Bank shape differs")
    t, n, _ = bank.shape
    f64(bank, (t, n, 3), "bank")
    original_activation = np.asarray(original_activation)
    require(
        original_activation.ndim == 2 and original_activation.shape[0] == t,
        "Original activation rows differ",
    )
    m = original_activation.shape[1]
    f64(original_activation, (t, m), "original activation")
    membership = i64(membership, (n,), "membership")
    fixed = i64(fixed, (len(fixed),), "fixed")
    kappa = f64(kappa, (n,), "kappa")
    low = f64(low, (m, n, 3), "low slopes")
    high = f64(high, (m, n, 3), "high slopes")
    require(
        np.all((membership >= 0) & (membership < m))
        and np.all((fixed >= 0) & (fixed < m))
        and len(np.unique(fixed)) == len(fixed),
        "Invalid muscle mapping",
    )
    out = np.array(original_activation, copy=True)
    for muscle in range(m):
        if muscle in fixed:
            continue
        cols = np.flatnonzero(membership == muscle)
        require(len(cols) > 0, "Empty variable muscle")
        terms = selected_contributions(
            bank, cols, kappa, low[muscle, cols], high[muscle, cols]
        )
        out[:, muscle] = np.sum(terms["low"], axis=(1, 2)) + np.sum(
            terms["hinge"], axis=(1, 2)
        )
    return out


def residual(
    expected: np.ndarray, observed: np.ndarray, label: str, atol: float = 1e-10
) -> dict[str, Any]:
    require(
        expected.shape == observed.shape
        and np.isfinite(expected).all()
        and np.isfinite(observed).all(),
        f"Invalid {label} comparison",
    )
    delta = np.abs(expected - observed)
    where = np.unravel_index(int(np.argmax(delta)), delta.shape)
    maximum = float(delta[where])
    require(
        maximum <= atol, f"{label} residual {maximum:.17g} at {where} exceeds {atol}"
    )
    return {"max_abs": maximum, "index": [int(i) for i in where]}


def native_events(
    load: np.ndarray,
    time_s: np.ndarray,
    *,
    threshold: float = 0.02,
    pre: int = 100,
    post: int = 100,
    half_window: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load = np.asarray(load)
    time_s = np.asarray(time_s)
    require(load.ndim == 1 and len(load) == len(time_s), "Native contact shape differs")
    f64(load, load.shape, "native load")
    f64(time_s, time_s.shape, "native time")
    require(
        np.all(load >= 0)
        and threshold > 0
        and pre > 0
        and post > 0
        and half_window >= max(pre, post),
        "Invalid native contact contract",
    )
    support = load >= threshold
    crossing = np.flatnonzero(support[:-1] & ~support[1:]) + 1
    events = []
    for i in crossing:
        i = int(i)
        if i - half_window < 0 or i + half_window >= len(load):
            reason = "incomplete_alignment_window"
        elif not np.all(support[i - pre : i]):
            reason = "unstable_pre_support"
        elif np.any(support[i : i + post]):
            reason = "unstable_post_nonsupport"
        else:
            reason = "included"
        events.append(
            {
                "native_index": i,
                "time_s": float(time_s[i]),
                "reason": reason,
                "included": reason == "included",
            }
        )
    padded = np.r_[True, support, True]
    starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    ends = np.flatnonzero(~padded[:-1] & padded[1:])
    stats = {
        "downcrossing_count": len(events),
        "included_count": sum(e["included"] for e in events),
        "reason_counts": dict(Counter(e["reason"] for e in events)),
        "support_fraction": float(np.mean(support)),
        "longest_nonsupport_samples": int(np.max(ends - starts)) if len(starts) else 0,
    }
    return events, stats


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_arm(freeze: dict[str, Any], arm: str) -> dict[str, Any]:
    spec = freeze["arms"][arm]
    data: dict[str, Any] = {
        k: np.load(source_path(v), allow_pickle=False)
        for k, v in spec["fields"].items()
    }
    for k in ("state", "manifest", "muscle_provenance", "output_manifest", "trace"):
        data[k] = _json(source_path(spec["paths"][k]))
    for k in ("native_recruitment_state", "native_contact"):
        with np.load(source_path(spec["paths"][k]), allow_pickle=False) as z:
            data[k] = {name: z[name] for name in z.files}
    return data


def _metadata(freeze: dict[str, Any], data: dict[str, Any], arm: str) -> dict[str, Any]:
    spec = freeze["arms"][arm]
    state = data["state"]
    adapter = state["adapter"]
    patch = state["monotone_recruitment_patch"]
    require(
        adapter["schema"] == "flylab.m3-monotone-recruitment-adapter.v1"
        and adapter["hash"] == freeze["expected_identity"]["adapter_hash"],
        "Monotone adapter identity differs",
    )
    require(
        patch["adapter_identity"] == adapter["hash"], "Patch/adapter identity differs"
    )
    require(state["control_tick"] == 0, "Start checkpoint is not anchor")
    manifest = data["manifest"]["files"]
    require(
        state["network"]["tick"] == 0 and state["network"]["dt"] == 0.00025,
        "Start neural clock is not the saved anchor",
    )
    require(
        manifest.get("state.json") == freeze["input_sha256"][spec["paths"]["state"]],
        "Start state manifest entry differs",
    )

    def pinned_copy(reference: Any, field: str) -> None:
        require(
            isinstance(reference, dict) and set(reference) == {"__npy__"},
            f"{field} copy reference malformed",
        )
        name = reference["__npy__"]
        require(
            isinstance(name, str) and Path(name).name == name and name.endswith(".npy"),
            f"{field} copy path invalid",
        )
        relative = (Path(spec["paths"]["state"]).parent / name).as_posix()
        expected = freeze["input_sha256"][spec["fields"][field]]
        require(
            manifest.get(name) == expected
            and freeze["preservation_sha256"].get(relative) == expected,
            f"{field} copy manifest/preservation differs",
        )
        require(sha256(source_path(relative)) == expected, f"{field} copy bytes differ")

    for name in ("tau", "kappa", "slopes", "membership", "fixed_muscles"):
        ref = adapter["parameters"][name]
        require(
            ref == {"__npy__": Path(spec["fields"][name]).name},
            f"{name} adapter reference differs",
        )
        pinned_copy(patch[name], name)
    pinned_copy(patch["motor_ids"], "motor_ids")
    for name in ("motor_ids", "moment", "rest", "pool", "ltm"):
        ref = adapter["calibration"][name]
        require(
            ref == {"__npy__": Path(spec["fields"][name]).name},
            f"{name} calibration reference differs",
        )
    for name, key in (("bank", "initial_bank"), ("activation", "initial_activation")):
        require(
            adapter[name] == {"__npy__": Path(spec["fields"][key]).name},
            f"{name} initial state reference differs",
        )
    original_ref = adapter["original"]["activation"]
    require(
        original_ref
        == {"__npy__": Path(spec["fields"]["initial_original_activation"]).name},
        "Original activation reference differs",
    )
    roster = adapter["parameters"]["motor_roster"]
    require(
        roster == patch["motor_roster"] == data["muscle_provenance"]["motor_roster"],
        "Motor roster provenance differs",
    )
    groups = adapter["groups"]
    require(
        groups == data["muscle_provenance"]["muscles"] and len(groups) == 96,
        "Named muscle provenance differs",
    )
    require(
        groups[74] == freeze["expected_identity"]["group"],
        "RM tibia flexor group differs",
    )
    require(
        [roster[i] for i in groups[74]["columns"]]
        == freeze["expected_identity"]["roots"],
        "RM root roster differs",
    )
    for relative in spec["fields"].values():
        name = Path(relative).name
        require(
            manifest.get(name) == freeze["input_sha256"][relative],
            f"Start manifest differs: {name}",
        )
    return {"groups": groups, "roster": roster}


def _validate_arm(
    freeze: dict[str, Any], data: dict[str, Any], meta: dict[str, Any]
) -> None:
    n, m, j, t, v = 391, 96, 42, 2001, 100000
    for name, shape in {
        "tau": (3,),
        "kappa": (n,),
        "slopes": (m, n, 3, 2),
        "pool": (m, n),
        "moment": (j, m),
        "rest": (j,),
        "initial_bank": (n, 3),
        "initial_activation": (m,),
        "initial_original_activation": (m,),
    }.items():
        f64(data[name], shape, name)
    for name, shape in {
        "motor_ids": (n,),
        "membership": (n,),
        "fixed_muscles": (19,),
        "ltm": (6,),
    }.items():
        i64(data[name], shape, name)
    require(np.array_equal(data["tau"], freeze["contract"]["tau"]), "Tau differs")
    for name in (
        "kappa",
        "slopes",
        "pool",
        "initial_bank",
        "initial_activation",
        "initial_original_activation",
    ):
        require(np.all(data[name] >= 0), f"Negative {name}")
    require(
        np.allclose(data["pool"].sum(axis=1), 1, atol=1e-15, rtol=0),
        "Pool rows are not normalized",
    )
    require(
        np.max(np.abs(data["rest"])) <= 10, "Rest target outside calibration bounds"
    )
    require(
        np.all((data["membership"] >= 0) & (data["membership"] < m)),
        "Membership range differs",
    )
    require(
        np.array_equal(
            np.count_nonzero(data["pool"] > 0, axis=0), np.ones(n, dtype=int)
        )
        and np.array_equal(np.argmax(data["pool"], axis=0), data["membership"]),
        "Pool is not single membership",
    )
    support = np.arange(m)[:, None] == data["membership"][None, :]
    require(np.all(data["slopes"][~support] == 0), "Off-owner slope support")
    require(
        np.array_equal(np.any(data["slopes"] > 0, axis=(2, 3)), support),
        "Anatomical slope owner has no positive support",
    )
    require(
        len(np.unique(data["fixed_muscles"])) == len(data["fixed_muscles"])
        and np.all((data["fixed_muscles"] >= 0) & (data["fixed_muscles"] < m)),
        "Invalid fixed muscle rows",
    )
    canonical = np.zeros_like(data["slopes"])
    canonical[:, :, 1, 0] = data["pool"]
    canonical[:, :, 1, 1] = data["pool"]
    require(
        np.array_equal(
            data["slopes"][data["fixed_muscles"]], canonical[data["fixed_muscles"]]
        ),
        "Fixed slopes differ from original pool",
    )
    require(
        len(np.unique(data["ltm"])) == 6
        and np.all((data["ltm"] >= 0) & (data["ltm"] < m)),
        "Invalid LTM rows",
    )
    require(
        np.array_equal(
            data["fixed_muscles"],
            np.unique(
                np.r_[
                    data["ltm"],
                    np.flatnonzero(np.bincount(data["membership"], minlength=m) == 1),
                ]
            ),
        ),
        "Fixed muscle union differs",
    )
    require(
        np.array_equal(data["motor_ids"], np.sort(np.unique(data["motor_ids"])))
        and len(meta["roster"]) == n
        and len(set(meta["roster"])) == n
        and all(isinstance(root, str) and root for root in meta["roster"]),
        "Motor IDs/roster incomplete or unordered",
    )
    leg_names = freeze["contract"]["legs"]
    group_legs = [group.get("leg") for group in meta["groups"]]
    require(
        set(group_legs) == set(leg_names)
        and group_legs == sorted(group_legs, key=leg_names.index),
        "Named muscle leg ordering differs",
    )
    for muscle, group in enumerate(meta["groups"]):
        require(
            group["columns"] == np.flatnonzero(data["membership"] == muscle).tolist()
            and isinstance(group["muscle"], str)
            and bool(group["muscle"]),
            f"Muscle {muscle} columns/name/leg differ",
        )
        leg_index = leg_names.index(group["leg"])
        active_rows = np.flatnonzero(data["moment"][:, muscle])
        require(
            np.all(active_rows // 7 == leg_index),
            f"Muscle {muscle} moment crosses its named leg",
        )
    require(
        set(freeze["contract"]["silent_post_ids"]).issubset(
            set(data["motor_ids"][meta["groups"][74]["columns"]].tolist())
        ),
        "E178 selected IDs absent",
    )
    rec = data["native_recruitment_state"]
    for key, shape in {
        "control_tick": (t,),
        "network_tick": (t,),
        "physics_time_s": (t,),
        "bank": (t, n, 3),
        "activation": (t, m),
        "original_activation": (t, m),
        "motor_rate": (t, n),
    }.items():
        (i64 if key in ("control_tick", "network_tick") else f64)(rec[key], shape, key)
    for key in ("bank", "activation", "original_activation", "motor_rate"):
        require(np.all(rec[key] >= 0), f"Negative saved {key}")
    require(
        np.array_equal(rec["control_tick"], np.arange(t))
        and np.array_equal(rec["network_tick"], 20 * np.arange(t)),
        "Recruitment tick differs",
    )
    require(
        np.max(np.abs(rec["physics_time_s"] - 0.005 * np.arange(t))) <= 1e-8,
        "Recruitment clock differs",
    )
    for key in (
        "complete",
        "bank_available",
        "activation_available",
        "original_activation_available",
        "motor_rate_available",
    ):
        a = np.asarray(rec[key])
        require(
            a.shape == (t,) and a.dtype == np.bool_ and np.all(a), f"{key} unavailable"
        )
    for key, first in (
        ("bank", "initial_bank"),
        ("activation", "initial_activation"),
        ("original_activation", "initial_original_activation"),
    ):
        require(np.array_equal(rec[key][0], data[first]), f"{key} anchor differs")
    native = data["native_contact"]
    f64(native["time_s"], (v,), "native time")
    f64(native["normal_load_bw"], (v, 6), "native load")
    require(np.all(native["normal_load_bw"] >= 0), "Negative native load")
    require(
        np.max(np.abs(native["time_s"] - (np.arange(v) + 1) * 0.0001)) <= 1e-8,
        "Native clock differs",
    )
    trace = data["trace"]
    require(
        isinstance(trace.get("samples"), list) and len(trace["samples"]) == t,
        "Trace rows differ",
    )
    for row_index, row in enumerate(trace["samples"]):
        require(
            row.get("control_tick") == row_index
            and abs(row.get("time_s", float("nan")) - row_index * 0.005) <= 1e-8,
            "Trace clock differs",
        )
        expected_interval = (
            None
            if row_index == 0
            else {
                "start_tick": row_index - 1,
                "end_tick": row_index,
                "start_s": (row_index - 1) * 0.005,
                "end_s": row_index * 0.005,
            }
        )
        require(
            row.get("command_interval") == expected_interval,
            "Trace command interval differs",
        )
        require(row.get("fault") is None, "Trace fault recorded")
        require(
            np.array_equal(
                f64(
                    np.asarray(row["motor_rates"], dtype=np.float64),
                    (n,),
                    "trace rates",
                ),
                rec["motor_rate"][row_index],
            ),
            "Trace endpoint rate differs",
        )
        f64(
            np.asarray(row["joint_targets_rad"], dtype=np.float64), (j,), "trace target"
        )
        f64(
            np.asarray(row["joint_angles_rad"], dtype=np.float64),
            (j,),
            "trace actual joint",
        )
        require(
            np.isfinite(row["body"]["speed"]) and np.isfinite(row["upright"]),
            "Trace body diagnostic invalid",
        )
        feet = row["feet"]
        require(
            isinstance(feet, dict) and "normal_load_bw" in feet,
            "Trace feet unavailable",
        )
        foot_load = f64(
            np.asarray(feet["normal_load_bw"], dtype=np.float64),
            (6,),
            "trace foot load",
        )
        require(np.all(foot_load >= 0), "Negative trace foot load")
        require(
            np.array_equal(
                f64(
                    np.asarray(row["body"]["contactsBW"], dtype=np.float64),
                    (6,),
                    "body foot load",
                ),
                foot_load,
            ),
            "Trace body/feet loads differ",
        )


def _stats(a: np.ndarray, dt: float = 0.005) -> dict[str, Any]:
    a = np.asarray(a)
    return {
        "mean": np.mean(a, axis=0).tolist(),
        "min": np.min(a, axis=0).tolist(),
        "max": np.max(a, axis=0).tolist(),
        "exact_zero_rows": np.sum(a == 0, axis=0).tolist(),
        "integral": (dt * np.sum(a, axis=0)).tolist(),
    }


def analyze_arm(
    freeze: dict[str, Any], arm: str, data: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Validate one saved arm and return compact raw arrays, summary, events."""
    if data is None:
        data = _load_arm(freeze, arm)
    meta = _metadata(freeze, data, arm)
    _validate_arm(freeze, data, meta)
    rec = data["native_recruitment_state"]
    columns = np.asarray(meta["groups"][74]["columns"], dtype=np.int64)
    bank = rec["bank"]
    rate = rec["motor_rate"]
    activation = rec["activation"]
    original = rec["original_activation"]
    slopes = data["slopes"]
    fixed = data["fixed_muscles"]
    low, high = slopes[..., 0], slopes[..., 1]
    errors = {
        "bank": residual(bank_recurrence(rate, bank, data["tau"]), bank[1:], "bank")
    }
    reconstructed = reconstruct_activation(
        bank, original, data["membership"], fixed, data["kappa"], low, high
    )
    errors["activation"] = residual(reconstructed, activation, "activation")
    terms = selected_contributions(
        bank, columns, data["kappa"], low[74, columns], high[74, columns]
    )
    errors["selected_sum"] = residual(
        np.sum(terms["low"], axis=(1, 2)) + np.sum(terms["hinge"], axis=(1, 2)),
        activation[:, 74],
        "selected sum",
    )
    target = data["rest"][None, :] + 0.02 * (reconstructed @ data["moment"].T)
    trace_rows = data["trace"]["samples"]
    observed_target = np.asarray(
        [row["joint_targets_rad"] for row in trace_rows], dtype=np.float64
    )
    errors["target"] = residual(target[1:], observed_target[1:], "target")
    rm_rows = slice(28, 35)
    rm_muscle_contribution = (
        0.02 * activation[:, 74, None] * data["moment"][rm_rows, 74][None, :]
    )
    rm_muscles = np.asarray(
        [i for i, group in enumerate(meta["groups"]) if group["leg"] == "rm"],
        dtype=np.int64,
    )
    all_rm_contributions = (
        0.02
        * activation[:, rm_muscles, None]
        * data["moment"][rm_rows, rm_muscles].T[None, :, :]
    )
    errors["rm_muscle_command_sum"] = residual(
        data["rest"][rm_rows][None, :] + np.sum(all_rm_contributions[1:], axis=1),
        observed_target[1:, rm_rows],
        "RM muscle command sum",
    )
    load = data["native_contact"]["normal_load_bw"][:, 4]
    native_time = data["native_contact"]["time_s"]
    native, contact_stats = native_events(load, native_time)
    support = (load >= 0.02).reshape(2000, 50)
    support_fraction = support.mean(axis=1)
    native_support = load >= 0.02
    release_indices = np.flatnonzero(native_support[:-1] & ~native_support[1:]) + 1
    release_count = np.bincount(release_indices // 50, minlength=2000)
    for event in native:
        i = event["native_index"]
        row = i // 50 + 1
        require(
            1 <= row <= 2000 and abs(event["time_s"] - (i + 1) * 0.0001) <= 1e-8,
            "Event-to-command clock differs",
        )
        event.update(
            {
                "control_row": row,
                "activation": float(activation[row, 74]),
                "motor_ids": data["motor_ids"][columns].tolist(),
                "rate": rate[row, columns].tolist(),
                "contribution_by_filter": terms["total"][row].tolist(),
            }
        )
    selected = [e for e in native if e["included"]]
    event_summary = {
        **contact_stats,
        "longest_nonsupport_s": contact_stats["longest_nonsupport_samples"] * 0.0001,
        "conditional": {
            "status": "available",
            "selected_count": len(selected),
            "mean_activation": float(np.mean([e["activation"] for e in selected])),
        }
        if selected
        else {"status": "unavailable", "reason": "no_stable_downcrossing"},
    }
    windows = {}
    for name, (start, end) in freeze["contract"]["windows"].items():
        idx = np.arange(1, 2001)
        idx = idx[(idx * 0.005 > start) & (idx * 0.005 <= end)]
        total_integral = 0.005 * float(np.sum(terms["total"][idx]))
        by_cell_filter = 0.005 * np.sum(terms["total"][idx], axis=0)
        ratio = (
            None if total_integral == 0 else (by_cell_filter / total_integral).tolist()
        )
        by_motor = np.sum(terms["total"][idx], axis=2)
        by_filter = np.sum(terms["total"][idx], axis=1)
        motor_integral = 0.005 * np.sum(by_motor, axis=0)
        filter_integral = 0.005 * np.sum(by_filter, axis=0)
        windows[name] = {
            "rows": len(idx),
            "rate": _stats(rate[idx][:, columns]),
            "low": _stats(terms["low"][idx]),
            "hinge": _stats(terms["hinge"][idx]),
            "total": _stats(terms["total"][idx]),
            "contribution_fraction": ratio,
            "per_motor_total": _stats(by_motor),
            "per_filter_total": _stats(by_filter),
            "per_motor_integral_fraction": None
            if total_integral == 0
            else (motor_integral / total_integral).tolist(),
            "per_filter_integral_fraction": None
            if total_integral == 0
            else (filter_integral / total_integral).tolist(),
            "activation": _stats(activation[idx, 74]),
            "original_activation": _stats(original[idx, 74]),
            "activation_minus_original": _stats(
                activation[idx, 74] - original[idx, 74]
            ),
        }
    raw = {
        "control_tick": rec["control_tick"],
        "time_s": rec["physics_time_s"],
        "columns": columns,
        "motor_ids": data["motor_ids"][columns],
        "root_ids": np.asarray([meta["roster"][i] for i in columns]),
        "rate": rate[:, columns],
        "bank": bank[:, columns],
        "low": terms["low"],
        "hinge": terms["hinge"],
        "total": terms["total"],
        "activation": activation[:, 74],
        "reconstructed_activation": reconstructed[:, 74],
        "original_activation": original[:, 74],
        "is_fixed": np.asarray(74 in fixed),
        "command_control_tick": rec["control_tick"][1:],
        "target_residual": target[1:] - observed_target[1:],
        "rm_actual_joint_angles": np.asarray(
            [row["joint_angles_rad"][rm_rows] for row in trace_rows]
        ),
        "rm_requested_joint_targets": observed_target[:, rm_rows],
        "rm_muscle_command_contribution": rm_muscle_contribution[1:],
        "rm_muscle_indices": rm_muscles,
        "rm_muscle_names": np.asarray(
            [meta["groups"][i]["muscle"] for i in rm_muscles]
        ),
        "all_rm_muscle_command_contributions": all_rm_contributions[1:],
        "native_interval_control_tick": rec["control_tick"][1:],
        "rm_native_support_fraction": support_fraction,
        "rm_native_release_count": release_count,
    }
    summary = {
        "arm": arm,
        "motor_ids": data["motor_ids"][columns].tolist(),
        "columns": columns.tolist(),
        "roots": [meta["roster"][i] for i in columns],
        "muscle_group": 74,
        "muscle_name": meta["groups"][74],
        "fixed": bool(74 in fixed),
        "reconstruction_residuals": errors,
        "windows": windows,
        "contact": event_summary,
    }
    return raw, summary, {"arm": arm, "events": native, "summary": event_summary}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_remove_stage(stage: Path, output: Path) -> None:
    parent = output.parent.resolve(strict=True)
    resolved = stage.resolve(strict=False)
    require(
        resolved.parent == parent
        and resolved.name.startswith(".e179-stage-")
        and resolved != output.resolve(),
        "Unsafe E179 staging cleanup path",
    )
    if stage.exists():
        shutil.rmtree(stage)


def _publish_stage(stage: Path, output: Path) -> None:
    require(not output.exists(), "E179 output appeared before publication")
    require(
        stage.resolve().parent == output.parent.resolve()
        and stage.name.startswith(".e179-stage-"),
        "Unsafe E179 staging publication path",
    )
    try:
        stage.rename(output)
    except BaseException:
        _safe_remove_stage(stage, output)
        raise


def run(
    freeze_path: Path,
    freeze_sha256: str,
    source_review_path: Path,
    source_review_sha256: str,
    output: Path,
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        require(
            pools and all(pool["num_threads"] == 1 for pool in pools),
            "Numerical threadpool is not single-threaded",
        )
        observed_pools = [
            {"internal_api": pool["internal_api"], "num_threads": pool["num_threads"]}
            for pool in pools
        ]
        return _run_single_thread(
            freeze_path,
            freeze_sha256,
            source_review_path,
            source_review_sha256,
            output,
            observed_pools,
        )


def _run_single_thread(
    freeze_path: Path,
    freeze_sha256: str,
    source_review_path: Path,
    source_review_sha256: str,
    output: Path,
    observed_pools: list[dict[str, Any]],
) -> dict[str, Any]:
    began = time.monotonic()
    freeze, review = verified_gate(
        freeze_path, freeze_sha256, source_review_path, source_review_sha256
    )
    require(
        output.resolve() == (ROOT / freeze["output_directory"]).resolve(),
        "E179 output path differs",
    )
    require(not output.exists(), "E179 output already exists")
    require(
        shutil.disk_usage(output.parent).free >= freeze["minimum_free_disk_bytes"],
        "Insufficient free disk",
    )
    baseline = freeze["arms"]["baseline"]
    candidate = freeze["arms"]["candidate"]
    for field in baseline["fields"]:
        require(
            freeze["input_sha256"][baseline["fields"][field]]
            == freeze["input_sha256"][candidate["fields"][field]],
            f"Two-arm {field} source differs",
        )
    require(
        freeze["input_sha256"][baseline["paths"]["muscle_provenance"]]
        == freeze["input_sha256"][candidate["paths"]["muscle_provenance"]],
        "Two-arm muscle provenance differs",
    )
    stage = Path(tempfile.mkdtemp(prefix=".e179-stage-", dir=output.parent))
    try:
        raws = {}
        summaries = {}
        events = {}
        for arm in ("baseline", "candidate"):
            require(
                time.monotonic() - began < freeze["internal_wall_limit_s"],
                "E179 wall limit reached",
            )
            raw, summaries[arm], events[arm] = analyze_arm(freeze, arm)
            raws.update({f"{arm}_{name}": value for name, value in raw.items()})
        for key in ("motor_ids", "columns", "roots", "fixed"):
            require(
                summaries["baseline"][key] == summaries["candidate"][key],
                f"Two-arm {key} differs",
            )
        a = raws["baseline_activation"]
        b = raws["candidate_activation"]
        unequal = np.flatnonzero(a != b)
        where = int(np.argmax(np.abs(a - b)))
        summaries["two_arm_difference"] = {
            "first_different_control_row": int(unequal[0]) if len(unequal) else None,
            "max_abs_activation_difference": float(np.max(np.abs(a - b))),
            "max_abs_control_row": where,
        }
        raw_uncompressed_bytes = sum(
            np.asarray(value).nbytes for value in raws.values()
        )
        require(
            raw_uncompressed_bytes <= freeze["maximum_raw_bytes"],
            "E179 uncompressed raw limit exceeded",
        )
        np.savez_compressed(stage / "contributions.npz", **raws)
        require(
            (stage / "contributions.npz").stat().st_size <= freeze["maximum_raw_bytes"],
            "E179 raw limit exceeded",
        )
        _write_json(stage / "summary.json", summaries)
        _write_json(stage / "events.json", events)
        verify_hash_map(freeze["input_sha256"], "post-run input")
        verify_hash_map(freeze["reused_source_sha256"], "post-run reused source")
        require(
            sha256(freeze_path) == FREEZE_SHA
            and sha256(source_review_path) == source_review_sha256,
            "E179 gate files changed during run",
        )
        require(
            review["source_sha256"]
            == {name: sha256(source_path(name)) for name in OWNED},
            "Owned source changed during run",
        )
        _write_json(
            stage / "input-receipt.json",
            {
                "schema": SCHEMA,
                "freeze_sha256": FREEZE_SHA,
                "source_review_sha256": source_review_sha256,
                "input_sha256": freeze["input_sha256"],
                "reused_source_sha256": freeze["reused_source_sha256"],
                "owned_source_sha256": review["source_sha256"],
                "verified_before_and_after": True,
                "observed_threadpools": observed_pools,
                "raw_uncompressed_bytes": raw_uncompressed_bytes,
            },
        )
        elapsed = time.monotonic() - began
        require(elapsed <= freeze["internal_wall_limit_s"], "E179 wall limit exceeded")
        result = {
            "schema": SCHEMA,
            "status": "valid_saved_data_diagnostic",
            "wall_s": elapsed,
            "cpu_threads": 1,
            "observed_threadpools": observed_pools,
            "raw_uncompressed_bytes": raw_uncompressed_bytes,
            "original_processes": 1,
            "automatic_retries": 0,
            "gpu_count": 0,
            "model_constructions": 0,
            "neural_steps": 0,
            "physics_steps": 0,
            "fit_calls": 0,
            "F6_exposed": False,
            "candidate_adopted": False,
            "whole_goal_complete": False,
            "interpretation_limit": "Saved output presence does not prove foot lift, normal gait, biological mechanism, force, or causal recovery.",
        }
        _write_json(stage / "result.json", result)
        manifest = {
            name: {
                "sha256": sha256(stage / name),
                "bytes": (stage / name).stat().st_size,
            }
            for name in OUTPUTS
        }
        _write_json(
            stage / "output-manifest.json", {"schema": SCHEMA, "files": manifest}
        )
        total_published_bytes = sum(
            path.stat().st_size for path in stage.iterdir() if path.is_file()
        )
        require(
            total_published_bytes <= freeze["maximum_raw_bytes"],
            "E179 total publication size limit exceeded",
        )
        require(
            time.monotonic() - began <= freeze["internal_wall_limit_s"],
            "E179 publication deadline exceeded",
        )
        _publish_stage(stage, output)
        return result
    except BaseException:
        _safe_remove_stage(stage, output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
