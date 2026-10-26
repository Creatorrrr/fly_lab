"""E171: one-shot, saved-QR-only affine capacity diagnosis.

The original archives are opened only after the parent source gate and all
metadata/hash checks. No feature, target, or original run is reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tools/diagnose_c_banc_m3_saved_qr_capacity.py"
TESTS = "tests/test_c_banc_saved_qr_capacity.py"
KINDS = ("command", "velocity", "startup")
ROWS = {"command": 1824 * 7, "velocity": 1823 * 7, "startup": 99 * 7}
DENOMINATORS = {"command": 1824 * 42, "velocity": 1823 * 42, "startup": 99 * 42}
EPS = np.finfo(np.float64).eps
ARCHIVE_KEY_MAP = {
    "pair_start": "affine_rows_velocity_start",
    "pair_end": "affine_rows_velocity_end",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exact_array(
    array: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, label: str
) -> None:
    require(array.shape == shape, f"{label}: shape {array.shape} != {shape}")
    require(array.dtype == dtype, f"{label}: dtype {array.dtype} != {dtype}")
    if np.issubdtype(dtype, np.floating):
        require(bool(np.isfinite(array).all()), f"{label}: nonfinite")


def preflight(freeze_path: Path, gate_path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Metadata-only gate. In particular, this function never calls np.load."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require(freeze["schema"] == "flylab.m3.e171.saved-qr-capacity.v2", "freeze schema")
    require(freeze["family_order"] == ["static", "history"], "family order")
    require(freeze["design_sha256"] == sha256(root / freeze["design"]), "design hash")
    require(
        gate.get("schema") == "flylab.m3.e171.parent-source-review.v2", "gate schema"
    )
    require(gate.get("accepts_single_saved_qr_diagnostic") is True, "parent acceptance")
    require(gate.get("freeze_sha256") == sha256(freeze_path), "gate freeze hash")
    require(gate.get("design_sha256") == freeze["design_sha256"], "gate design hash")
    require(set(gate.get("source_sha256", {})) == {SOURCE, TESTS}, "gate source names")
    for name in (SOURCE, TESTS):
        require(
            gate["source_sha256"][name] == sha256(root / name),
            f"gate source hash: {name}",
        )
    require(
        set(freeze["allowed_keys"])
        == {
            "pair_motors",
            "pair_filters",
            "pair_muscles",
            "w0",
            "target_indices",
            "pair_start",
            "pair_end",
        }
        | {
            f"qr_{kind}_{leg}{suffix}"
            for kind in KINDS
            for leg in freeze["sample_contract"]["legs"]
            for suffix in ("", "_active")
        },
        "freeze allowed keys",
    )
    for name, expected in freeze["input_sha256"].items():
        require(sha256(root / name) == expected, f"input hash: {name}")
    require(freeze["archive_key_map"] == ARCHIVE_KEY_MAP, "archive key aliases")
    for family in freeze["family_order"]:
        validate_archive_members(
            root / freeze["families"][family]["raw"],
            freeze["allowed_keys"],
            freeze["archive_key_map"],
        )
    originals = [
        json.loads((root / name).read_text(encoding="utf-8"))
        for name in (
            "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json",
            "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
        )
    ]
    for prior in originals:
        require(
            prior["sample_contract"] == freeze["sample_contract"],
            "sample contract mismatch",
        )
        for name, expected in freeze["shared_input_hashes"].items():
            require(
                prior["input_sha256"].get(name) == expected,
                f"shared input hash: {name}",
            )
    require(
        freeze["shared_input_hash_count"] == len(freeze["shared_input_hashes"]),
        "shared hash count",
    )
    require(np.__version__ == freeze["method"]["version"], "NumPy version")
    require(freeze["method"]["dtype"] == "float64", "method dtype")
    pools = threadpool_info()
    require(
        all(pool.get("num_threads") == 1 for pool in pools), f"BLAS threads: {pools}"
    )
    return {"freeze": freeze, "gate": gate, "threadpool_info": pools}


def validate_archive_members(
    path: Path, allowed: list[str], aliases: dict[str, str]
) -> None:
    """Check ZIP directory metadata before any original array value is opened."""
    with zipfile.ZipFile(path) as archive:
        for name in allowed:
            member = aliases.get(name, name) + ".npy"
            archive.getinfo(member)


def read_selected(
    path: Path, allowed: list[str], aliases: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Access only the fixed whitelist; never enumerate or copy the archive."""
    aliases = {} if aliases is None else aliases
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[aliases.get(name, name)] for name in allowed}


def validate_saved(
    static: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    c = freeze["sample_contract"]
    legs = c["legs"]
    require(legs == ["lf", "lm", "lh", "rf", "rm", "rh"], "legs")
    require(
        (
            c["target_count"],
            c["target_first_tick"],
            c["target_last_tick"],
            c["startup_first_tick"],
            c["startup_last_tick"],
        )
        == (1824, 100, 1923, 1, 99),
        "tick contract",
    )
    motors = np.asarray(c["variable_motors"], dtype=np.int64)
    variable = np.asarray(c["variable_muscles"], dtype=np.int64)
    group_legs = np.asarray(c["group_legs"], dtype=np.int64)
    exact_array(motors, (330,), np.dtype("int64"), "contract motors")
    require(np.array_equal(motors, np.sort(np.unique(motors))), "motor order")
    require(len(variable) == 77 and len(c["fixed_muscles"]) == 19, "muscle counts")
    require(
        np.array_equal(np.sort(np.r_[variable, c["fixed_muscles"]]), np.arange(96)),
        "muscle partition",
    )
    require(
        np.array_equal(np.bincount(group_legs, minlength=6), c["group_counts"]),
        "leg group counts",
    )
    expected_motors = np.repeat(motors, 3)
    expected_filters = np.tile(np.arange(3, dtype=np.int64), 330)
    for family, saved in (("static", static), ("history", history)):
        exact_array(
            saved["pair_motors"], (990,), np.dtype("int64"), f"{family} pair_motors"
        )
        exact_array(
            saved["pair_filters"], (990,), np.dtype("int64"), f"{family} pair_filters"
        )
        exact_array(
            saved["pair_muscles"], (990,), np.dtype("int64"), f"{family} pair_muscles"
        )
        exact_array(saved["w0"], (1980,), np.dtype("float64"), f"{family} w0")
        require(
            np.array_equal(saved["pair_motors"], expected_motors),
            f"{family} motor mapping",
        )
        require(
            np.array_equal(saved["pair_filters"], expected_filters),
            f"{family} filter mapping",
        )
        require(
            np.array_equal(
                saved["pair_muscles"].reshape(330, 3),
                saved["pair_muscles"][::3, None] * np.ones((1, 3), dtype=np.int64),
            ),
            f"{family} muscle/filter mapping",
        )
        require(
            np.isin(saved["pair_muscles"], variable).all(), f"{family} muscle ownership"
        )
        for name, length in (
            ("target_indices", 1824),
            ("pair_start", 1823),
            ("pair_end", 1823),
        ):
            exact_array(saved[name], (length,), np.dtype("int64"), f"{family} {name}")
        require(
            np.array_equal(saved["target_indices"], np.arange(100, 1924)),
            f"{family} target ticks",
        )
        require(
            np.array_equal(saved["pair_start"], np.arange(100, 1923)),
            f"{family} velocity starts",
        )
        require(
            np.array_equal(saved["pair_end"], np.arange(101, 1924)),
            f"{family} velocity ends",
        )
        require(
            array_sha256(saved["target_indices"]) == c["target_indices_sha256"],
            f"{family} target hash",
        )
        owners = np.repeat(saved["pair_muscles"], 2)
        covered: list[np.ndarray] = []
        for leg_index, leg in enumerate(legs):
            alpha_active = np.flatnonzero(group_legs[owners] == leg_index)
            expected_active = (
                alpha_active
                if family == "static"
                else np.r_[alpha_active, alpha_active + 1980]
            )
            covered.append(expected_active)
            for kind in KINDS:
                key = f"qr_{kind}_{leg}"
                active = saved[f"{key}_active"]
                exact_array(
                    active,
                    expected_active.shape,
                    np.dtype("int64"),
                    f"{family} {key}_active",
                )
                require(
                    np.array_equal(active, expected_active),
                    f"{family} {key} active mapping",
                )
                qr = saved[key]
                exact_array(
                    qr,
                    (min(ROWS[kind], active.size + 1), active.size + 1),
                    np.dtype("float64"),
                    f"{family} {key}",
                )
        require(
            np.array_equal(
                np.sort(np.concatenate(covered)),
                np.arange(freeze["families"][family]["coefficient_count"]),
            ),
            f"{family} active coverage",
        )
    for key in (
        "pair_motors",
        "pair_filters",
        "pair_muscles",
        "w0",
        "target_indices",
        "pair_start",
        "pair_end",
    ):
        require(np.array_equal(static[key], history[key]), f"family mismatch: {key}")
    return {
        "legs": legs,
        "group_legs": group_legs.tolist(),
        "variable_muscles": variable.tolist(),
    }


def solve_block(
    qr: np.ndarray, baseline: np.ndarray, raw: dict[str, np.ndarray]
) -> dict[str, Any]:
    """One SVD, fixed cutoff, no retry. Raw arrays are populated before assertions."""
    matrix, offset = qr[:, :-1], qr[:, -1]
    raw["qr"] = qr
    raw["baseline_local"] = baseline
    raw["baseline_residual"] = matrix @ baseline + offset
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    raw.update(U=u, s=singular, Vh=vh)
    cutoff = float(EPS * max(matrix.shape) * singular[0]) if singular.size else 0.0
    keep = singular > cutoff
    raw["keep"] = keep
    raw["cutoff"] = np.asarray(cutoff, dtype=np.float64)
    x = (
        -(vh[keep, :].T @ ((u[:, keep].T @ offset) / singular[keep]))
        if keep.any()
        else np.zeros(matrix.shape[1], dtype=np.float64)
    )
    raw["x"] = x
    actual = matrix @ x + offset
    projected = offset - u[:, keep] @ (u[:, keep].T @ offset)
    raw.update(actual_residual=actual, projected_residual=projected)
    rank = int(keep.sum())
    reconstruction = float(
        np.linalg.norm(u @ (singular[:, None] * vh) - matrix)
        / max(np.linalg.norm(matrix), np.finfo(float).tiny)
    )
    u_orth = float(np.max(np.abs(u.T @ u - np.eye(u.shape[1])))) if u.shape[1] else 0.0
    v_orth = (
        float(np.max(np.abs(vh @ vh.T - np.eye(vh.shape[0])))) if vh.shape[0] else 0.0
    )
    normal = matrix.T @ actual
    stationarity = u[:, keep].T @ actual
    raw.update(normal_residual=normal, retained_stationarity=stationarity)
    resolution = float(
        128
        * EPS
        * max(matrix.shape)
        * (np.linalg.norm(matrix) * np.linalg.norm(x) + np.linalg.norm(offset))
        + 1e-12
    )
    norm_gap = float(abs(np.linalg.norm(actual) - np.linalg.norm(projected)))

    def finite_condition(first: float, last: float) -> float | None:
        if last <= 0:
            return None
        with np.errstate(over="ignore", divide="ignore"):
            value = float(first / last)
        return value if math.isfinite(value) else None

    details = {
        "rank": rank,
        "singular_count": int(singular.size),
        "cutoff": cutoff,
        "near_cutoff_count": int(
            np.count_nonzero((singular >= cutoff / 10) & (singular <= cutoff * 10))
        ),
        "dropped_count": int(singular.size - rank),
        "full_condition": finite_condition(float(singular[0]), float(singular[-1]))
        if singular.size
        else None,
        "retained_condition": finite_condition(
            float(singular[0]), float(singular[rank - 1])
        )
        if rank
        else None,
        "coefficient_max_abs": float(np.max(np.abs(x))) if x.size else 0.0,
        "coefficient_l2": float(np.linalg.norm(x)),
        "reconstruction_relative": reconstruction,
        "u_orthogonality_max": u_orth,
        "v_orthogonality_max": v_orth,
        "normal_residual_l2": float(np.linalg.norm(normal)),
        "retained_stationarity_l2": float(np.linalg.norm(stationarity)),
        "actual_projected_norm_gap": norm_gap,
        "resolution_indicator": resolution,
        "baseline_norm": float(np.linalg.norm(raw["baseline_residual"])),
        "actual_norm": float(np.linalg.norm(actual)),
        "projected_norm": float(np.linalg.norm(projected)),
        "qr_sha256": array_sha256(qr),
    }
    require(
        np.isfinite(singular).all()
        and np.isfinite(x).all()
        and np.isfinite(actual).all()
        and np.isfinite(projected).all(),
        "nonfinite SVD return",
    )
    require(
        bool(np.all(singular >= 0) and np.all(np.diff(singular) <= 0)),
        "singular ordering",
    )
    require(
        reconstruction <= 1e-10 and u_orth <= 1e-10 and v_orth <= 1e-10,
        "SVD reconstruction/orthogonality",
    )
    require(norm_gap <= resolution, "actual/projected numerical gap")
    require(
        details["actual_norm"]
        <= details["baseline_norm"] + 1e-12 + 1e-10 * details["baseline_norm"],
        "command fit worse than baseline",
    )
    return details


def evaluate_qr(
    qr: np.ndarray, local: np.ndarray, baseline: np.ndarray, rows: int
) -> dict[str, float]:
    residual = qr[:, :-1] @ local + qr[:, -1]
    base_residual = qr[:, :-1] @ baseline + qr[:, -1]
    return {
        "norm": float(np.linalg.norm(residual)),
        "baseline_norm": float(np.linalg.norm(base_residual)),
        "rms": float(np.linalg.norm(residual) / math.sqrt(rows)),
        "baseline_rms": float(np.linalg.norm(base_residual) / math.sqrt(rows)),
    }


def coefficient_diagnostics(
    coefficients: np.ndarray,
    pair_muscles: np.ndarray,
    variable_muscles: list[int],
    family: str,
    contract: dict[str, Any],
    pair_motors: np.ndarray | None = None,
) -> dict[str, Any]:
    alpha = coefficients[:1980].reshape(990, 2)
    beta = coefficients[1980:].reshape(990, 2) if family == "history" else None
    branches = {"alpha": alpha}
    if beta is not None:
        branches["beta"] = beta
        branches["alpha_plus_beta"] = alpha + beta
    branch_stats = {}
    for name, values in branches.items():
        branch_stats[name] = {
            "below_zero_count": int(np.count_nonzero(values < 0)),
            "above_two_count": int(np.count_nonzero(values > 2)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "below_zero_magnitude": float(max(0.0, -values.min())),
            "above_two_magnitude": float(max(0.0, values.max() - 2)),
        }
    endpoints = (
        np.concatenate((alpha, alpha + beta), axis=1) if beta is not None else alpha
    )
    pair_min = endpoints.min(axis=1)
    pair_max = endpoints.max(axis=1)
    n_owned = np.bincount(pair_muscles[::3], minlength=96)
    floor = float(contract["motor_floor_fraction"])
    lower = float(contract["muscle_lower"])
    upper = float(contract["muscle_upper"])
    atol = float(contract["physical_atol"])
    motors = []
    for i in range(330):
        muscle = int(pair_muscles[3 * i])
        value = float(pair_min[3 * i : 3 * i + 3].sum())
        threshold = floor / int(n_owned[muscle])
        motors.append(
            {
                "motor_local": i,
                "motor_original": int(pair_motors[3 * i])
                if pair_motors is not None
                else None,
                "muscle": muscle,
                "minimum": value,
                "floor": threshold,
                "slack": value - threshold,
                "violated": bool(value < threshold - atol),
            }
        )
    muscles = []
    for muscle in variable_muscles:
        selected = pair_muscles == muscle
        min_total = float(pair_min[selected].sum())
        max_total = float(pair_max[selected].sum())
        muscles.append(
            {
                "muscle": int(muscle),
                "minimum": min_total,
                "maximum": max_total,
                "lower_slack": min_total - lower,
                "upper_slack": upper - max_total,
                "lower_violated": bool(min_total < lower - atol),
                "upper_violated": bool(max_total > upper + atol),
            }
        )
    return {
        "branches": branch_stats,
        "pair_min": pair_min.tolist(),
        "pair_max": pair_max.tolist(),
        "motors": motors,
        "muscles": muscles,
        "motor_floor_violation_count": sum(item["violated"] for item in motors),
        "muscle_lower_violation_count": sum(item["lower_violated"] for item in muscles),
        "muscle_upper_violation_count": sum(item["upper_violated"] for item in muscles),
    }


def aggregate_family(
    legs: dict[str, dict[str, Any]], resolution: list[float], ratio_limit: float = 0.8
) -> dict[str, Any]:
    metrics = {}
    for kind in KINDS:
        actual = math.sqrt(
            sum(leg["metrics"][kind]["norm"] ** 2 for leg in legs.values())
        )
        baseline = math.sqrt(
            sum(leg["metrics"][kind]["baseline_norm"] ** 2 for leg in legs.values())
        )
        metrics[kind] = {
            "norm": actual,
            "baseline_norm": baseline,
            "rms": actual / math.sqrt(DENOMINATORS[kind]),
            "baseline_rms": baseline / math.sqrt(DENOMINATORS[kind]),
        }
    command = metrics["command"]
    indicator = math.sqrt(sum(value * value for value in resolution)) / math.sqrt(
        DENOMINATORS["command"]
    )
    threshold = ratio_limit * command["baseline_rms"]
    unresolved = (
        indicator > 0.001 * command["baseline_rms"]
        or abs(command["rms"] - threshold) <= indicator
    )
    return {
        "metrics": metrics,
        "command_ratio": command["rms"] / command["baseline_rms"]
        if command["baseline_rms"]
        else None,
        "resolution_indicator_rms": indicator,
        "numerically_unresolved": unresolved,
        "attains_20_percent": bool(not unresolved and command["rms"] <= threshold),
    }


def classify_conclusion(results: list[dict[str, Any]]) -> str:
    if any(item["attains_20_percent"] for item in results):
        return "supported_returned_saved_affine_witness"
    if any(item["numerically_unresolved"] for item in results):
        return "inconclusive_numerical_resolution"
    return "inconclusive_not_attained_at_fixed_cutoff"


def output_bytes(out: Path) -> int:
    return sum(path.stat().st_size for path in out.iterdir() if path.is_file())


def resource_check(out: Path, started: float, limits: dict[str, Any]) -> None:
    require(
        time.monotonic() - started <= limits["internal_wall_s"], "internal wall limit"
    )
    require(output_bytes(out) <= limits["max_output_bytes"], "raw output size limit")


def run(
    freeze_path: Path, gate_path: Path, out: Path, root: Path = ROOT
) -> dict[str, Any]:
    require(not out.exists(), "output already exists")
    started = time.monotonic()
    pre = preflight(freeze_path, gate_path, root)
    freeze = pre["freeze"]
    require(
        shutil.disk_usage(out.parent).free
        >= freeze["resource_limits"]["min_free_bytes"],
        "free disk",
    )
    out.mkdir(parents=False, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "flylab.m3.e171.saved-qr-capacity-result.v2",
        "status": "invalid_not_tested",
        "adoption": "not_applicable",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F1_F8": "pending",
        "original_processes": 1,
        "new_svd_fits": 0,
        "new_qr_factorizations": 0,
        "retries": 0,
        "startup_pointwise_max": "not_evaluated",
        "clipping": "not_evaluated",
        "activation": "not_evaluated",
        "native_gate": "not_evaluated",
        "threadpool_info": pre["threadpool_info"],
        "numpy_version": np.__version__,
        "provenance": {
            "freeze_sha256": sha256(freeze_path),
            "design_sha256": freeze["design_sha256"],
            "gate_sha256": sha256(gate_path),
            "source_sha256": {name: sha256(root / name) for name in (SOURCE, TESTS)},
        },
        "families": {},
        "primary_error": None,
        "secondary_publication_errors": [],
    }
    try:
        selected = {
            family: read_selected(
                root / freeze["families"][family]["raw"],
                freeze["allowed_keys"],
                freeze["archive_key_map"],
            )
            for family in freeze["family_order"]
        }
        validation = validate_saved(selected["static"], selected["history"], freeze)
        report["validation"] = validation
        # A per-leg comparison catches redistribution of baseline errors that a
        # family-wide squared norm could conceal. This consumes no fit.
        baseline_comparison: dict[str, dict[str, Any]] = {}
        for leg in validation["legs"]:
            baseline_comparison[leg] = {}
            for kind in KINDS:
                norms = []
                for family in freeze["family_order"]:
                    saved = selected[family]
                    baseline = (
                        np.r_[saved["w0"], np.zeros(1980)]
                        if family == "history"
                        else saved["w0"]
                    )
                    key = f"qr_{kind}_{leg}"
                    qr = saved[key]
                    local = baseline[saved[f"{key}_active"]]
                    norms.append(float(np.linalg.norm(qr[:, :-1] @ local + qr[:, -1])))
                compatible = abs(norms[0] - norms[1]) <= 1e-12 + 1e-10 * abs(norms[0])
                baseline_comparison[leg][kind] = {
                    "static_norm": norms[0],
                    "history_norm": norms[1],
                    "compatible": compatible,
                }
                require(compatible, f"per-leg family baseline mismatch: {leg}/{kind}")
        report["baseline_comparison"] = baseline_comparison
        for family in freeze["family_order"]:
            saved = selected[family]
            size = freeze["families"][family]["coefficient_count"]
            baseline = (
                np.r_[saved["w0"], np.zeros(1980)]
                if family == "history"
                else np.asarray(saved["w0"])
            )
            full = np.empty(size, dtype=np.float64)
            legs: dict[str, dict[str, Any]] = {}
            resolution: list[float] = []
            for leg in validation["legs"]:
                resource_check(out, started, freeze["resource_limits"])
                command = f"qr_command_{leg}"
                active = saved[f"{command}_active"]
                raw: dict[str, np.ndarray] = {}
                raw_path = out / f"{family}-{leg}-svd-raw.npz"
                block_error: Exception | None = None
                details: dict[str, Any] | None = None
                try:
                    report["new_svd_fits"] += 1
                    details = solve_block(saved[command], baseline[active], raw)
                except Exception as exc:  # noqa: BLE001 - preserve the primary SVD/numerical failure
                    block_error = exc
                # Includes a completed return even if its numerical checks fail.
                try:
                    np.savez_compressed(raw_path, **raw)
                except Exception as exc:  # noqa: BLE001 - publication failure is separately recorded
                    report["secondary_publication_errors"].append(
                        {"path": str(raw_path), "error": repr(exc)}
                    )
                    fallback = out / f"{family}-{leg}-svd-raw.fallback.npz"
                    try:
                        np.savez_compressed(fallback, **raw)
                        report.setdefault("partial_blocks", []).append(
                            {
                                "family": family,
                                "leg": leg,
                                "raw": fallback.name,
                                "sha256": sha256(fallback),
                                "returned_x_sha256": array_sha256(raw["x"])
                                if "x" in raw
                                else None,
                            }
                        )
                    except Exception as fallback_exc:  # noqa: BLE001 - retain both I/O errors
                        report["secondary_publication_errors"].append(
                            {"path": str(fallback), "error": repr(fallback_exc)}
                        )
                    if block_error is None:
                        block_error = exc
                if block_error is not None:
                    raise block_error
                assert details is not None
                require("x" in raw, f"no SVD return for {family}/{leg}")
                full[active] = raw["x"]
                resolution.append(details["resolution_indicator"])
                leg_metrics = {
                    kind: evaluate_qr(
                        saved[f"qr_{kind}_{leg}"],
                        raw["x"],
                        baseline[active],
                        ROWS[kind],
                    )
                    for kind in KINDS
                }
                legs[leg] = {
                    "active": active.tolist(),
                    "svd": details,
                    "svd_raw": raw_path.name,
                    "svd_raw_sha256": sha256(raw_path),
                    "metrics": leg_metrics,
                }
                report["families"].setdefault(family, {"legs": {}})["legs"][leg] = legs[
                    leg
                ]
                resource_check(out, started, freeze["resource_limits"])
            np.save(out / f"{family}-coefficients.npy", full)
            family_result = aggregate_family(legs, resolution)
            family_result["coefficient_file"] = f"{family}-coefficients.npy"
            family_result["coefficient_sha256"] = sha256(
                out / family_result["coefficient_file"]
            )
            family_result["coefficient_diagnostics"] = coefficient_diagnostics(
                full,
                saved["pair_muscles"],
                validation["variable_muscles"],
                family,
                freeze["contract"],
                saved["pair_motors"],
            )
            report["families"][family].update(family_result)
            resource_check(out, started, freeze["resource_limits"])
        for kind in KINDS:
            a = report["families"]["static"]["metrics"][kind]["baseline_norm"]
            b = report["families"]["history"]["metrics"][kind]["baseline_norm"]
            require(
                abs(a - b) <= 1e-12 + 1e-10 * abs(a),
                f"family baseline mismatch: {kind}",
            )
        results = [report["families"][name] for name in freeze["family_order"]]
        report["status"] = "valid_diagnostic"
        report["conclusion"] = classify_conclusion(results)
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - preserve any original numerical or I/O failure
        report["status"] = "invalid_not_tested"
        report["primary_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["conclusion"] = "invalid_not_tested"
    report["elapsed_s"] = time.monotonic() - started
    try:
        json_write(out / "result.json", report)
    except Exception as exc:
        report["secondary_publication_errors"].append(
            {"path": str(out / "result.json"), "error": repr(exc)}
        )
        report["status"] = "invalid_not_tested"
        report["conclusion"] = "invalid_not_tested"
        if report["primary_error"] is None:
            report["primary_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        try:
            json_write(out / "result.fallback.json", report)
        except Exception as fallback_exc:  # noqa: BLE001 - primary failure remains authoritative
            print(f"fallback publication failed: {fallback_exc!r}", file=sys.stderr)
        raise
    try:
        resource_check(out, started, freeze["resource_limits"])
    except (ValueError, OSError) as exc:
        report["status"] = "invalid_not_tested"
        report["conclusion"] = "invalid_not_tested"
        resource_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if report["primary_error"] is None:
            report["primary_error"] = resource_error
        else:
            report.setdefault("secondary_resource_errors", []).append(resource_error)
        report["elapsed_s"] = time.monotonic() - started
        try:
            json_write(out / "result.json", report)
        except Exception as publication_exc:  # noqa: BLE001 - preserve complete report
            report["secondary_publication_errors"].append(
                {"path": str(out / "result.json"), "error": repr(publication_exc)}
            )
            json_write(out / "result.fallback.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.source_gate, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "conclusion": result["conclusion"],
                "new_svd_fits": result["new_svd_fits"],
            }
        )
    )
    return 0 if result["status"] == "valid_diagnostic" else 1


if __name__ == "__main__":
    sys.exit(main())
