"""E311 saved startup algebra only: no solver, model, teacher, or controller."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
SCHEMA = RESULT_SCHEMA = "flylab.m3.e311.decoder-obstruction.result.v1"
FREEZE_SCHEMA = SCHEMA.replace("result", "freeze")
REVIEW_SCHEMA = SCHEMA.replace("result", "review")
DESIGN = "docs/BANC_M3_E311_DECODER_OBSTRUCTION_DESIGN_20260928.md"
DESIGN_SHA = "68664a147990babdb4750502804cf5ae94bc319b76a2ab3e07f327dd65f7da0b"
OUTPUT_PATH = BASE + "/e311-decoder-obstruction"
OWNED = (
    "tools/analyze_c_banc_m3_decoder_obstruction.py",
    "tests/test_c_banc_decoder_obstruction.py",
)
SOURCE_FILES = OWNED + (
    "tools/banc_m3_source_decoder_fit.py",
    "tools/fit_c_banc_m3_source_decoder.py",
    BASE + "/parent-e287-saved-ray-audit.py",
)
INPUT_PATHS = {
    k: BASE + "/" + v
    for k, v in {
        "result": "e287-source-decoder-fit/result.json",
        "solve_json": "e287-source-decoder-fit/decoder-solve.json",
        "problem": "e287-source-decoder-fit/decoder-problem.npz",
        "solve_raw": "e287-source-decoder-fit/decoder-solve.npz",
        "startup": "e287-source-decoder-fit/source-startup.npz",
        "startup_receipt": "e287-source-decoder-fit/source-startup.json",
        "cost": "e287-source-decoder-fit/cost.json",
        "e287_freeze": "e287-source-decoder-fit-freeze.json",
        "e287_review": "e287-source-decoder-fit-source-review.json",
        "e287_root_audit": "parent-e287-original-final-review.json",
        "ray_audit": "parent-e287-saved-ray-audit.json",
        "output_manifest": "parent-e287-output-manifest.json",
        "mapping": "e269-decoder-reference/engine-physiology-decoder.npz",
        "axes": "e270-prepared-inputs/axes.npz",
        "e310_audit": "parent-e310-original-raw-audit.json",
    }.items()
}
INPUT_PATHS["e310_report"] = (
    "docs/BANC_M3_E310_RESOURCE_CONTINUATION_RESULTS_20260928.md"
)
LIMITS = {
    "supervisor_s": 20,
    "internal_s": 15,
    "finalization_s": 5,
    "cpu_threads": 1,
    "host_bytes": 1536 * 1024**2,
    "output_bytes": 32 * 1024**2,
    "free_disk_bytes": 256 * 1024**2,
    "attempts": 1,
    "retries": 0,
    "model_calls": 0,
    "neural_ticks": 0,
    "physics_steps": 0,
    "cuda_calls": 0,
    "fit_calls": 0,
    "solver_calls": 0,
    "qr_calls": 0,
}
Q_PEAK = 0.03838315744033522
Q_RMS = 0.017763328718176728
CONTRACT = {
    "rows": 99,
    "motors": 391,
    "muscles": 96,
    "joints": 42,
    "legs": 6,
    "fixed": 19,
    "variable": 77,
    "free_motors": 330,
    "variables": 3960,
    "q_peak": Q_PEAK,
    "q_rms": Q_RMS,
    "native_q_peak": 0.04,
    "native_q_rms": 0.018,
    "reconstruction_atol": 1e-13,
    "certificate_allowance_scale": 1e-10,
    "source": "fixed_E277_E287_startup_only",
    "activation_box": "independent a[t,m] in [0,2 max member saved bank]",
    "limits": LIMITS,
}
MAPPING_KEYS = (
    "motor_ids",
    "membership",
    "pool",
    "low",
    "high",
    "kappa",
    "fixed",
    "moment",
    "rest",
    "ltm",
    "leg_motor",
    "leg_muscle",
    "target_startup_command",
)
PROBLEM_KEYS = (
    "b",
    "startup_error_offset",
    "free_motor",
    "variable_muscle",
    "fixed_muscle",
    "cone_dimensions",
)
SCOPE = "Only fixed E277 source bank and original fixed19 within this independent activation box; no E150/SSP-neutral, arbitrary activation range, different source, feasibility, causal, biological, or gait claim."


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def safe(root, name):
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and not Path(name).is_absolute(),
        "Relative POSIX path required",
    )
    p = (Path(root) / name).resolve(strict=True)
    require(
        p.is_relative_to(Path(root).resolve())
        and p.is_file()
        and not (Path(root) / name).is_symlink(),
        "Unsafe file path",
    )
    return p


def packages():
    return {
        "python": platform.python_version(),
        **{
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "threadpoolctl")
        },
    }


def checked_array(value, shape, dtype, label):
    require(
        isinstance(value, np.ndarray)
        and value.shape == shape
        and value.dtype == np.dtype(dtype)
        and np.isfinite(value).all(),
        "Invalid " + label,
    )
    return value


def csc(data, prefix):
    shape = checked_array(data[prefix + "_shape"], (2,), np.int64, prefix + " shape")
    require(np.all(shape > 0), "Positive sparse dimensions required")
    values = data[prefix + "_data"]
    ids = data[prefix + "_indices"]
    ptr = data[prefix + "_indptr"]
    checked_array(values, (len(values),), np.float64, prefix + " data")
    require(
        ids.dtype in (np.int32, np.int64)
        and ptr.dtype in (np.int32, np.int64)
        and ids.shape == values.shape
        and ptr.shape == (int(shape[1]) + 1,)
        and ptr[0] == 0
        and ptr[-1] == len(values)
        and np.all(np.diff(ptr) >= 0)
        and np.all((ids >= 0) & (ids < shape[0])),
        "Invalid sparse storage",
    )
    matrix = sparse.csc_matrix((values, ids, ptr), shape=tuple(shape))
    require(matrix.has_canonical_format, "Canonical saved CSC required")
    return matrix


def sparse_residual(a, b):
    require(a.shape == b.shape, "Sparse reconstruction shape differs")
    difference = a - b
    maximum = float(np.max(abs(difference.data))) if difference.nnz else 0.0
    return {"max_abs": maximum, "pass": maximum <= 1e-13}


def dense_residual(a, b):
    require(
        a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
        "Finite reconstruction comparison required",
    )
    maximum = float(np.max(abs(a - b)))
    return {"max_abs": maximum, "pass": maximum <= 1e-13}


def layout(mapping, axes):
    n, m, j = len(mapping["motor_ids"]), len(mapping["fixed"]), len(mapping["rest"])
    for key, shape, dtype in (
        ("motor_ids", (n,), np.int64),
        ("membership", (n,), np.int64),
        ("pool", (m, n), np.float64),
        ("low", (n, 3), np.float64),
        ("high", (n, 3), np.float64),
        ("kappa", (n,), np.float64),
        ("fixed", (m,), bool),
        ("moment", (j, m), np.float64),
        ("rest", (j,), np.float64),
        ("leg_motor", (n,), np.int64),
        ("leg_muscle", (m,), np.int64),
    ):
        checked_array(mapping[key], shape, dtype, key)
    member = mapping["membership"]
    pool = mapping["pool"]
    require(
        len(np.unique(mapping["motor_ids"])) == n
        and np.all(mapping["motor_ids"] >= 0)
        and np.all((member >= 0) & (member < m)),
        "Motor membership identity differs",
    )
    require(
        np.all(pool >= 0)
        and np.all(np.count_nonzero(pool, axis=0) == 1)
        and np.array_equal(np.argmax(pool, axis=0), member)
        and np.allclose(pool.sum(axis=1), 1, rtol=0, atol=1e-15),
        "Original anatomical pool differs",
    )
    counts = np.bincount(member, minlength=m)
    fixed = counts == 1
    ltm = mapping["ltm"]
    require(
        ltm.dtype == np.int64
        and ltm.ndim == 1
        and len(np.unique(ltm)) == len(ltm)
        and np.all((ltm >= 0) & (ltm < m)),
        "LTM roster differs",
    )
    fixed[ltm] = True
    require(
        np.array_equal(fixed, mapping["fixed"]),
        "Original fixed singleton/LTM definition differs",
    )
    for key in ("low", "high", "kappa"):
        require(np.all(mapping[key] >= 0), "Negative monotone parameters")
    original = np.zeros((n, 3))
    original[:, 1] = pool[member, np.arange(n)]
    require(
        np.array_equal(mapping["low"][fixed[member]], original[fixed[member]])
        and np.array_equal(mapping["high"][fixed[member]], original[fixed[member]]),
        "Original fixed transfer differs",
    )
    upper = np.bincount(
        member,
        weights=np.maximum(mapping["low"], mapping["high"]).sum(axis=1),
        minlength=m,
    )
    require(np.all(upper <= 2.0), "Original family exceeds activation relaxation bound")
    for key, size in (("q", j), ("ltm", len(ltm))):
        checked_array(axes[key], (size,), np.int64, key + " leg axes")
    require(
        all(
            np.all((v >= 0) & (v < 6))
            for v in (
                axes["q"],
                axes["ltm"],
                mapping["leg_motor"],
                mapping["leg_muscle"],
            )
        )
        and np.array_equal(mapping["leg_motor"], mapping["leg_muscle"][member])
        and np.array_equal(axes["ltm"], mapping["leg_muscle"][ltm])
        and np.all(
            mapping["moment"][axes["q"][:, None] != mapping["leg_muscle"][None, :]] == 0
        ),
        "Original leg support differs",
    )
    return (
        np.flatnonzero(~fixed[member]).astype(np.int64),
        np.flatnonzero(~fixed).astype(np.int64),
        np.flatnonzero(fixed).astype(np.int64),
    )


def reconstruct(mapping, bank, original, target):
    """No temporal integration: independent literal saved-row hinge coefficients."""
    free = np.flatnonzero(~mapping["fixed"][mapping["membership"]])
    fixed = np.flatnonzero(mapping["fixed"])
    t, _n, _ = bank.shape
    j = len(mapping["rest"])
    rows = []
    cols = []
    values = []
    basis = np.stack(
        (
            np.minimum(bank[:, free], mapping["kappa"][free, None]),
            np.maximum(bank[:, free] - mapping["kappa"][free, None], 0),
        ),
        axis=-1,
    ).reshape(t, len(free), 6)
    for col, motor in enumerate(free):
        v = (
            0.02
            * mapping["moment"][:, mapping["membership"][motor]][None, :, None]
            * basis[:, col, None, :]
        )
        nonzero = np.nonzero(v)
        rows.append(nonzero[0] * j + nonzero[1])
        cols.append(col * 6 + nonzero[2])
        values.append(v[nonzero])
    design = sparse.csc_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(t * j, len(free) * 6),
    )
    offset = (
        mapping["rest"]
        + 0.02 * (original[:, fixed] @ mapping["moment"][:, fixed].T)
        - target
    )
    value = mapping["low"][None, :, :] * bank + (mapping["high"] - mapping["low"])[
        None, :, :
    ] * np.maximum(bank - mapping["kappa"][None, :, None], 0)
    activation = original.copy()
    for muscle in np.flatnonzero(~mapping["fixed"]):
        activation[:, muscle] = value[:, mapping["membership"] == muscle].sum(
            axis=(1, 2)
        )
    q = mapping["rest"] + 0.02 * (activation @ mapping["moment"].T)
    return design, offset, activation, q, basis


def original_equations(design, offset, mapping, free, variable):
    """Rebuild only original feasibility rows, never P/teacher/objective."""
    pairs = 3 * len(free)
    slopes = 2 * pairs
    n = 4 * pairs
    rr = []
    cc = []
    vv = []
    rhs = []

    def row(columns, values, bound):
        rr.extend([len(rhs)] * len(columns))
        cc.extend(columns)
        vv.extend(values)
        rhs.append(bound)

    for pair in range(pairs):
        lo, hi, lower, upper = (
            2 * pair,
            2 * pair + 1,
            slopes + pair,
            slopes + pairs + pair,
        )
        row([lower], [-1.0], 0.0)
        row([upper], [1.0], 2.0)
        row([lower, lo], [1.0, -1.0], 0.0)
        row([lower, hi], [1.0, -1.0], 0.0)
        row([lo, upper], [1.0, -1.0], 0.0)
        row([hi, upper], [1.0, -1.0], 0.0)
    member = mapping["membership"]
    for muscle in variable:
        pp = (
            np.flatnonzero(member[free] == muscle)[:, None] * 3 + np.arange(3)
        ).ravel()
        row((slopes + pp).tolist(), [-1.0] * len(pp), -0.500001)
        row((slopes + pairs + pp).tolist(), [1.0] * len(pp), 1.999999)
    counts = np.bincount(member, minlength=len(mapping["fixed"]))
    for index, motor in enumerate(free):
        row(
            (slopes + 3 * index + np.arange(3)).tolist(),
            [-1.0] * 3,
            -1e-6 / counts[member[motor]],
        )
    envelope = sparse.csc_matrix((vv, (rr, cc)), shape=(len(rhs), n))
    pad = sparse.hstack(
        (design, sparse.csc_matrix((len(offset.ravel()), n - slopes))), format="csc"
    )
    A = sparse.vstack(
        (envelope, pad, -pad, sparse.csc_matrix((1, n)), -pad), format="csc"
    )
    o = offset.ravel()
    b = np.r_[rhs, Q_PEAK - o, Q_PEAK + o, np.sqrt(len(o)) * Q_RMS, o]
    return A, b, (len(rhs) + 2 * len(o), 1 + len(o)), len(rhs)


def old_ray(A, b, y, dims):
    require(
        A.dtype == np.float64
        and A.shape[0] == len(b) == len(y) == sum(dims)
        and len(dims) == 2,
        "Original cone layout differs",
    )
    checked_array(b, (A.shape[0],), np.float64, "b")
    checked_array(y, (A.shape[0],), np.float64, "solver_y")
    linear = int(dims[0])
    require(linear > 0 and dims[1] >= 2, "Original cones differ")
    margins = [
        float(np.min(y[:linear])),
        float(y[linear] - np.linalg.norm(y[linear + 1 :])),
    ]
    aty = np.asarray(A.T @ y)
    dot = float(b @ y)
    upper = dot + 2 * float(np.maximum(-aty, 0).sum())
    allowance = 1e-10 * (
        1 + float(np.sum(abs(b * y))) + 2 * float(np.sum(abs(A).T @ abs(y)))
    )
    return {
        "dual_cone_margins": margins,
        "b_dot_y": dot,
        "stationarity_max_abs": float(abs(aty).max()),
        "max_possible_weighted_slack": upper,
        "conservative_rounding_allowance": allowance,
        "bounded_affine_infeasibility_supported": bool(
            min(margins) >= 0 and upper + allowance < 0
        ),
        "variable_box": [0.0, 2.0],
    }, aty


def activation_bounds(offset, moment, upper):
    gain = 0.02 * moment
    lower = offset + upper @ np.minimum(gain, 0).T
    higher = offset + upper @ np.maximum(gain, 0).T
    scale = 1e-10 * (1 + abs(offset) + upper @ abs(gain).T)
    minimum = np.where(lower > 0, lower, np.where(higher < 0, -higher, 0.0))
    lower_rms = float(np.sqrt(np.mean(np.maximum(minimum - scale, 0) ** 2)))
    return {
        "lower": lower,
        "upper": higher,
        "allowance": scale,
        "optimistic_min_abs_error": minimum,
        "internal_peak_excluded": (lower > Q_PEAK + scale) | (higher < -Q_PEAK - scale),
        "native_peak_excluded": (lower > 0.04 + scale) | (higher < -0.04 - scale),
    }, lower_rms


def projected_ray(
    offset, moment, upper, positive, negative, soc_scalar, soc_vector, dual_cone_valid
):
    """Discard envelope duals; retain literal original peak/SOC signs, no projection."""
    w = positive - negative - soc_vector
    c = 0.02 * (w @ moment)
    peak_term = Q_PEAK * float((positive + negative).sum())
    soc_term = np.sqrt(offset.size) * Q_RMS * float(soc_scalar)
    offset_term = float(np.sum(w * offset))
    box_term = float(np.sum(upper * np.maximum(-c, 0)))
    bound = peak_term + soc_term - offset_term + box_term
    allowance = 1e-10 * (
        1
        + abs(peak_term)
        + abs(soc_term)
        + float(np.sum(abs(w * offset)))
        + float(np.sum(upper * abs(c)))
    )
    return {
        "peak_rhs": peak_term,
        "soc_rhs": soc_term,
        "weighted_offset": offset_term,
        "box_support": box_term,
        "max_possible_weighted_slack": bound,
        "conservative_rounding_allowance": allowance,
        "activation_box_excluded": bool(dual_cone_valid and bound + allowance < 0),
        "dual_cone_valid": bool(dual_cone_valid),
        "envelope_discarded": True,
        "feasibility_proven": False,
    }, {"projected_w": w, "activation_c": c}


def compute(
    mapping, startup, problem, solver_y, axes, *, ray_audit=None, check=lambda: None
):
    free, variable, fixed = layout(mapping, axes)
    t = 99
    j = len(mapping["rest"])
    n = len(mapping["motor_ids"])
    m = len(mapping["fixed"])
    require(len(free) > 0, "Original variable MN support required")
    for key, shape in (
        ("bank", (101, n, 3)),
        ("rate", (101, n)),
        ("original_activation", (101, m)),
        ("q", (101, j)),
        ("ltm", (101, len(mapping["ltm"]))),
    ):
        checked_array(startup[key], shape, np.float64, key)
        if key in ("bank", "rate", "original_activation"):
            require(np.all(startup[key] >= 0), "Negative saved source state")
    target = checked_array(
        mapping["target_startup_command"], (99, j), np.float64, "Target99"
    )
    for key, expected in (
        ("free_motor", free),
        ("variable_muscle", variable),
        ("fixed_muscle", fixed),
    ):
        require(
            np.array_equal(problem[key], expected) and problem[key].dtype == np.int64,
            "Original layout differs: " + key,
        )
    check()
    bank = startup["bank"][1:100]
    original = startup["original_activation"][1:100]
    design, offset, activation, q, basis = reconstruct(mapping, bank, original, target)
    saved_design = csc(problem, "startup_design")
    saved_A = csc(problem, "A")
    original_A, b, dims, envelope = original_equations(
        design, offset, mapping, free, variable
    )
    require(
        np.array_equal(
            checked_array(problem["cone_dimensions"], (2,), np.int64, "cones"), dims
        ),
        "Original cone dimensions differ",
    )
    checks = {
        "startup_design": sparse_residual(design, saved_design),
        "startup_offset": dense_residual(
            offset.ravel(), problem["startup_error_offset"]
        ),
        "original_A": sparse_residual(original_A, saved_A),
        "original_b": dense_residual(b, problem["b"]),
        "old_q": dense_residual(q, startup["q"][1:100]),
    }
    ray, aty = old_ray(saved_A, problem["b"], solver_y, dims)
    if ray_audit is not None:
        for key, value in ray.items():
            expected = ray_audit[key]
            if isinstance(value, bool):
                require(value is expected, "Saved ray outcome differs")
            else:
                require(
                    np.allclose(
                        value,
                        expected,
                        rtol=0,
                        atol=1e-13 * (1 + np.max(abs(np.asarray(expected)))),
                    ),
                    "Saved ray arithmetic differs: " + key,
                )
    check()
    upper = np.stack(
        [
            2 * np.max(bank[:, mapping["membership"] == muscle, :], axis=(1, 2))
            for muscle in variable
        ],
        axis=1,
    )
    moment = mapping["moment"][:, variable]
    bounds, rms = activation_bounds(offset, moment, upper)
    nn = int(dims[0])
    positive = solver_y[envelope : envelope + offset.size].reshape(t, j)
    negative = solver_y[envelope + offset.size : nn].reshape(t, j)
    soc = solver_y[nn + 1 :].reshape(t, j)
    projected, proj_arrays = projected_ray(
        offset,
        moment,
        upper,
        positive,
        negative,
        solver_y[nn],
        soc,
        min(ray["dual_cone_margins"]) >= 0,
    )
    anatomical = moment != 0
    adjustable = (upper > 0).astype(np.int64) @ anatomical.T.astype(np.int64) > 0
    counts = np.diff(design.tocsr().indptr).reshape(t, j)
    mass = positive + negative
    rank = np.lexsort((np.arange(mass.size), -mass.ravel()))[:20]
    violations = np.flatnonzero(
        bounds["internal_peak_excluded"].ravel()
        | bounds["native_peak_excluded"].ravel()
    )

    def provenance(row):
        time_index, joint = divmod(int(row), j)
        supported = variable[anatomical[joint]]
        fixed_supported = fixed[mapping["moment"][joint, fixed] != 0]
        motors = free[
            (mapping["moment"][joint, mapping["membership"][free]] != 0)
            & np.any(basis[time_index] > 0, axis=1)
        ]
        return {
            "row_index": int(row),
            "control_tick": time_index + 1,
            "joint": joint,
            "leg": int(axes["q"][joint]),
            "dual_peak_mass": float(mass[time_index, joint]),
            "variable_muscles": supported.tolist(),
            "instant_nonzero_U_muscles": supported[
                upper[time_index, np.flatnonzero(anatomical[joint])] > 0
            ].tolist(),
            "fixed_muscles": [
                {
                    "index": int(k),
                    "leg": int(mapping["leg_muscle"][k]),
                    "motor_ids": mapping["motor_ids"][
                        mapping["membership"] == k
                    ].tolist(),
                }
                for k in fixed_supported
            ],
            "additive_motor_ids": mapping["motor_ids"][motors].tolist(),
            "offset": float(offset[time_index, joint]),
            "error_interval": [
                float(bounds["lower"][time_index, joint]),
                float(bounds["upper"][time_index, joint]),
            ],
            "anatomical_variable_support_zero": bool(not anatomical[joint].any()),
            "instantaneous_adjustment_zero": bool(not adjustable[time_index, joint]),
        }

    valid = (
        all(v["pass"] for v in checks.values())
        and ray["bounded_affine_infeasibility_supported"]
    )
    excluded = bool(
        bounds["internal_peak_excluded"].any()
        or rms > Q_RMS
        or projected["activation_box_excluded"]
    )
    summary = {
        "run_valid": bool(valid),
        "checks": checks,
        "old_ray": ray,
        "projected_ray": projected,
        "optimistic_rms_lower_bound": rms,
        "internal_rms_excluded": rms > Q_RMS,
        "native_rms_excluded": rms > 0.018,
        "internal_direct_violation_rows": int(bounds["internal_peak_excluded"].sum()),
        "native_direct_violation_rows": int(bounds["native_peak_excluded"].sum()),
        "top20_dual_mass_rows": [provenance(k) for k in rank],
        "all_direct_violation_rows": [provenance(k) for k in violations],
        "activation_box_exclusion_supported": bool(valid and excluded),
        "outcome": "invalid"
        if not valid
        else "activation_box_excluded"
        if excluded
        else "inconclusive_not_feasibility",
        "raw_optimistic_rms_lower_bound": float(
            np.sqrt(np.mean(bounds["optimistic_min_abs_error"] ** 2))
        ),
        "conservative_optimistic_rms_lower_bound": rms,
        "optimistic_rms_lower_bound_basis": "per-coordinate allowance reduced before RMS",
        "limits": SCOPE,
        "no_controller_generated": True,
        "teacher_is_not_feasibility_condition": True,
    }
    raw = {
        "offset": offset,
        "U": upper,
        "old_q_reconstructed": q,
        "old_q_residual": q - startup["q"][1:100],
        "activation_reconstructed": activation,
        "dual_peak_mass": mass,
        "dual_positive": positive,
        "dual_negative": negative,
        "dual_soc_vector": soc,
        "dual_soc_scalar": np.array(solver_y[nn]),
        "original_A_transpose_y": aty,
        "anatomical_variable_support": anatomical,
        "instant_adjustable_support": adjustable,
        "additive_basis_support_count": counts,
        "fixed_support": mapping["moment"][:, fixed] != 0,
        "membership": mapping["membership"],
        "motor_ids": mapping["motor_ids"],
        "free_motor": free,
        "variable_muscle": variable,
        "fixed_muscle": fixed,
        "top20_row_indices": rank,
        "direct_violation_row_indices": violations,
        "initial_q_anchor": startup["q"][0].copy(),
        "initial_bank_anchor": startup["bank"][0].copy(),
        **bounds,
        **proj_arrays,
    }
    return summary, raw


def validate_producer(
    result, solve, cost, startup_receipt, root_audit, ray_audit, branch_audit
):
    require(
        result.get("schema") == "flylab.m3.e287.source-decoder-fit.v1"
        and result.get("run_valid") is True
        and result.get("selection_pass") is False
        and result.get("candidate_sha256") is None
        and result.get("status") == "bounded_uncertified_decoder"
        and result.get("checks") == {"solver": False},
        "Original valid E287 rejection required",
    )
    require(
        all(
            result.get(k) is False
            for k in ("body_promoted", "candidate_adopted", "F6_exposed")
        ),
        "Producer promoted",
    )
    require(
        solve.get("status") == "PrimalInfeasible"
        and solve.get("iterations") == 15
        and solve.get("iterations_known") is True
        and solve.get("solution_valid") is False,
        "Original15iteration infeasible solve required",
    )
    for k in ("constructor", "solve", "certificate"):
        receipt = solve[k]
        require(
            receipt.get("requested") == receipt.get("completed") == 1
            and receipt.get("failed") == 0
            and receipt.get("wall_known") is True,
            "Original solver receipt differs",
        )
    require(
        cost.get("engine_requests") == 2
        and cost["engine_creation"].get("completed") == 2
        and cost["evaluation"].get("completed") == 2
        and cost.get("forward_requested_steps")
        == cost.get("forward_completed_steps")
        == 42000
        and cost.get("derivative_steps") == 0
        and cost["solver"]["solve"].get("requested")
        == cost["solver"]["solve"].get("completed")
        == 1
        and cost["solver"].get("iterations") == 15
        and cost["development"].get("requested") == 0
        and cost.get("physics_steps") == cost.get("whole_BANC_calls") == 0
        and cost.get("source_fit_calls") == 0
        and cost.get("decoder_fit_calls") == 1
        and cost.get("F6_exposed") is False,
        "Original measured costs differ",
    )
    require(
        startup_receipt.get("status") == "complete"
        and startup_receipt.get("requested_steps")
        == startup_receipt.get("completed_steps")
        == 2000
        and startup_receipt.get("completed_samples") == 101
        and startup_receipt.get("derivative_completed_steps") == 0
        and startup_receipt.get("derivative_completed_samples") == 0,
        "Complete fixed-source startup receipt required",
    )
    require(
        root_audit.get("schema") == "flylab.m3.e287.root-original-audit.v1"
        and root_audit.get("run_valid") is True
        and root_audit.get("verdict") == "valid_reject"
        and root_audit.get("selection_pass") is False
        and root_audit.get("candidate_sha256") is None
        and root_audit.get("cost") == cost,
        "Original root audited rejection/cost required",
    )
    require(
        all(
            root_audit.get(k) is False
            for k in (
                "candidate_adopted",
                "F1_F8_complete",
                "F6_exposed",
                "natural_gait_complete",
            )
        ),
        "Root audit promoted",
    )
    require(
        ray_audit.get("schema") == "flylab.m3.e287.saved-global-ray-audit.v1"
        and ray_audit.get("raw_solver_status") == "PrimalInfeasible"
        and ray_audit.get("bounded_affine_infeasibility_supported") is True
        and ray_audit.get("new_optimization") is False
        and ray_audit.get("original_calls") == ray_audit.get("solver_calls") == 0,
        "Saved scoped ray required",
    )
    require(
        branch_audit.get("schema") == "flylab.m3.e310.root-original-raw-audit.v2"
        and branch_audit.get("pass") is True
        and branch_audit.get("diagnostic_run_valid") is True
        and branch_audit.get("decision") == "fixed_resource_branch_not_expanded"
        and branch_audit.get("interval_resource_benefit_observed") is False
        and all(
            branch_audit.get(k) is False
            for k in (
                "candidate_adopted",
                "F1_F8_complete",
                "F6_exposed",
                "biological_claim",
            )
        ),
        "E310 rejected branch must remain closed",
    )


def gate(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    *,
    root=ROOT,
    check=lambda: None,
):
    require(
        sha256(freeze_path) == freeze_sha256 and sha256(review_path) == review_sha256,
        "Freeze/review SHA differs",
    )
    freeze, review = read_json(freeze_path), read_json(review_path)
    require(
        freeze.get("schema") == FREEZE_SCHEMA
        and freeze.get("contract") == CONTRACT
        and freeze.get("output") == OUTPUT_PATH
        and freeze.get("design") == {"path": DESIGN, "sha256": DESIGN_SHA}
        and freeze.get("packages") == packages(),
        "Exact E311 contract/version required",
    )
    require(
        set(freeze["approved_sources"]) == set(OWNED)
        and review.get("schema") == REVIEW_SCHEMA
        and review.get("verdict") == "approved"
        and review.get("approved_sources") == freeze["approved_sources"]
        and review.get("freeze_sha256") == freeze_sha256,
        "E311 source review differs",
    )
    require(
        set(INPUT_PATHS.values()) <= set(freeze["input_sha256"])
        and set(SOURCE_FILES) - set(OWNED) <= set(freeze["inherited_sources"]),
        "Required input/source inventory missing",
    )
    hashes = {}
    for inventory in (
        freeze["approved_sources"],
        freeze["inherited_sources"],
        freeze["input_sha256"],
        freeze["preservation_sha256"],
        {DESIGN: DESIGN_SHA},
    ):
        for name, digest in inventory.items():
            require(name not in hashes or hashes[name] == digest, "Conflicting pin")
            hashes[name] = digest
    for name, digest in hashes.items():
        check()
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and sha256(safe(root, name)) == digest,
            "Pinned bytes differ: " + name,
        )

    def load(key):
        return read_json(safe(root, INPUT_PATHS[key]))

    prior = load("e287_freeze")
    prior_review = load("e287_review")
    require(
        prior.get("schema") == "flylab.m3.e287.source-decoder-fit.v1"
        and prior_review.get("schema") == "flylab.m3.e287.source-review.v1"
        and prior_review.get("verdict") == "approved"
        and prior_review.get("approved_sources") == prior.get("approved_sources")
        and prior_review.get("freeze_sha256") == hashes[INPUT_PATHS["e287_freeze"]],
        "Original reviewed E287 source freeze required",
    )
    for name, digest in {
        **prior["approved_sources"],
        **prior["inherited_sources"],
    }.items():
        require(
            freeze["inherited_sources"].get(name) == digest,
            "Inherited E287 source changed",
        )
    for alias, key in (("mapping", "static"), ("axes", "axes")):
        require(
            prior["inputs"][key]
            == {"path": INPUT_PATHS[alias], "sha256": hashes[INPUT_PATHS[alias]]},
            "Original static provenance differs",
        )
    for key, value in packages().items():
        if key in prior["packages"]:
            require(prior["packages"][key] == value, "Original package differs")
    ray = load("ray_audit")
    for name, digest in ray["inputs"].items():
        require(hashes.get(name) == digest, "Original ray input pin differs")
    manifest = load("output_manifest")
    for alias in (
        "result",
        "solve_json",
        "problem",
        "solve_raw",
        "startup",
        "startup_receipt",
        "cost",
    ):
        name = INPUT_PATHS[alias]
        entry = manifest["files"][Path(name).name]
        require(
            entry["sha256"] == hashes[name]
            and entry["bytes"] == safe(root, name).stat().st_size,
            "Original output manifest differs",
        )
    validate_producer(
        load("result"),
        load("solve_json"),
        load("cost"),
        load("startup_receipt"),
        load("e287_root_audit"),
        ray,
        load("e310_audit"),
    )
    return freeze


def selected_npz(path, keys):
    """Never iterate/unpack P, train, teacher, development, or native-stage values."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in keys}


def load_inputs(freeze, *, root=ROOT):
    def path(key):
        return safe(root, INPUT_PATHS[key])

    mapping = selected_npz(path("mapping"), MAPPING_KEYS)
    axes = selected_npz(path("axes"), ("q", "ltm"))
    keys = PROBLEM_KEYS + tuple(
        prefix + suffix
        for prefix in ("A", "startup_design")
        for suffix in ("_data", "_indices", "_indptr", "_shape")
    )
    problem = selected_npz(path("problem"), keys)
    startup = selected_npz(
        path("startup"), ("bank", "rate", "original_activation", "q", "ltm")
    )
    free, variable, fixed = layout(mapping, axes)
    require(
        (
            len(mapping["motor_ids"]),
            len(mapping["fixed"]),
            len(mapping["rest"]),
            len(mapping["ltm"]),
            len(free),
            len(variable),
            len(fixed),
        )
        == (391, 96, 42, 6, 330, 77, 19)
        and problem["A_shape"][1] == 3960,
        "Original full roster/layout required",
    )
    return (
        mapping,
        startup,
        problem,
        selected_npz(path("solve_raw"), ("solver_y",))["solver_y"],
        axes,
    )


def write_json(path, value):
    p = Path(path)
    temp = p.with_name(p.name + ".tmp")
    temp.write_bytes(canonical(value))
    os.replace(temp, p)


def rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (k, ctypes.c_size_t)
                for k in (
                    "peak",
                    "working",
                    "qpp",
                    "qp",
                    "qpnp",
                    "qnp",
                    "page",
                    "peakpage",
                )
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        f = ctypes.windll.psapi.GetProcessMemoryInfo
        f.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        require(
            f(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb) != 0, "RSS unavailable"
        )
        return int(c.peak)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def verify_publication(output):
    output = Path(output)
    require(
        not (output / "invalidation.json").exists()
        and not (output / "failure.json").exists(),
        "Explicitly invalid publication",
    )
    result = read_json(output / "result.json")
    seal = read_json(output / "result-sha256.json")
    manifest = read_json(output / "output-manifest.json")
    require(
        result.get("schema") == SCHEMA
        and result.get("publication_status") == "complete"
        and seal.get("sha256") == sha256(output / "result.json"),
        "Invalid result seal",
    )
    for name, entry in manifest["files"].items():
        require(
            Path(name).name == name
            and sha256(output / name) == entry["sha256"]
            and (output / name).stat().st_size == entry["bytes"],
            "Output manifest differs",
        )
    return result


def run(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    wall_clock=time.perf_counter,
    gate_fn=gate,
    loader=load_inputs,
    compute_fn=compute,
    writer=write_json,
    rss=rss_bytes,
):
    started = wall_clock()
    output = Path(output)
    require(
        output.resolve() == (Path(root) / OUTPUT_PATH).resolve()
        and not output.exists(),
        "Literal fresh E311 output required",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cost = {
        "analysis_requested": 1,
        "analysis_completed": 0,
        "gate_requested": 0,
        "gate_completed": 0,
        "load_requested": 0,
        "load_completed": 0,
        "compute_requested": 0,
        "compute_completed": 0,
        "retries": 0,
        **{
            k: 0
            for k in (
                "model_calls",
                "neural_ticks",
                "physics_steps",
                "cuda_calls",
                "fit_calls",
                "solver_calls",
                "qr_calls",
            )
        },
    }
    partial = {
        "schema": SCHEMA,
        "run_valid": False,
        "publication_status": "partial",
        "cost": cost,
        "scope": SCOPE,
    }

    def check():
        require(wall_clock() - started <= 15, "E311 internal deadline exceeded")
        require(rss() <= LIMITS["host_bytes"], "RSS cap exceeded")

    def final_check(final):
        now = wall_clock()
        require(now - final <= 5, "Finalization deadline exceeded")
        require(now - started <= 20, "E311 supervisor deadline exceeded")
        require(rss() <= LIMITS["host_bytes"], "RSS cap exceeded")
        require(
            sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            <= LIMITS["output_bytes"],
            "Output byte cap exceeded",
        )

    try:
        require(
            shutil.disk_usage(output.parent).free >= LIMITS["free_disk_bytes"],
            "Disk reserve exceeded",
        )
        cost["gate_requested"] += 1
        freeze = gate_fn(
            freeze_path,
            freeze_sha256,
            review_path,
            review_sha256,
            root=root,
            check=check,
        )
        cost["gate_completed"] += 1
        writer(
            output / "source-manifest.json",
            {
                "schema": SCHEMA.replace("result", "sources"),
                "approved_sources": freeze["approved_sources"],
                "inherited_sources": freeze["inherited_sources"],
                "design": freeze["design"],
                "packages": freeze["packages"],
            },
        )
        writer(
            output / "input-receipt.json",
            {
                "schema": SCHEMA.replace("result", "inputs"),
                "input_sha256": freeze["input_sha256"],
                "freeze_sha256": freeze_sha256,
                "review_sha256": review_sha256,
            },
        )
        check()
        cost["load_requested"] += 1
        inputs = loader(freeze, root=root)
        cost["load_completed"] += 1
        check()
        cost["compute_requested"] += 1
        began = wall_clock()
        ray = read_json(safe(root, INPUT_PATHS["ray_audit"]))
        with threadpool_limits(limits=1):
            summary, raw = compute_fn(*inputs, ray_audit=ray, check=check)
        cost["compute_completed"] += 1
        cost["compute_wall_s"] = wall_clock() - began
        partial["summary"] = summary
        check()
        final = wall_clock()
        final_check(final)
        np.savez_compressed(output / "obstruction.npz", **raw)
        writer(output / "summary.json", summary)
        cost["analysis_completed"] += 1
        cost["wall_s"] = wall_clock() - started
        cost["peak_rss_bytes"] = rss()
        result = {
            **partial,
            "run_valid": summary["run_valid"],
            "publication_status": "complete",
            "candidate_generated": False,
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "F6_exposed": False,
            "F2_complete": False,
            "M2_changed": False,
            "UI_changed": False,
            "biological_claim": False,
        }
        writer(output / "result.json", result)
        writer(
            output / "result-sha256.json",
            {"sha256": sha256(output / "result.json"), "path": "result.json"},
        )
        files = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in output.iterdir()
            if p.is_file()
        }
        writer(
            output / "output-manifest.json",
            {"schema": SCHEMA.replace("result", "output-manifest"), "files": files},
        )
        final_check(final)
        return result
    except BaseException as error:
        partial.update(error=repr(error), publication_status="invalid_partial")
        error.partial_result = partial
        error.partial_raw = locals().get("raw", {})
        error.cost_receipt = cost
        try:
            cost.update(wall_s=wall_clock() - started, peak_rss_bytes=rss())
        except BaseException as secondary:  # noqa: BLE001 -- retain first failure
            error.add_note("Failure meter error: " + repr(secondary))
        if error.partial_raw:
            try:
                # Reserve ZIP headers without relying on a favorable compression ratio.
                used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                raw_bytes = sum(
                    np.asarray(v).nbytes for v in error.partial_raw.values()
                )
                require(
                    used + raw_bytes + 65536 + 1024 * len(error.partial_raw)
                    <= LIMITS["output_bytes"],
                    "Partial raw output byte cap exceeded",
                )
                np.savez_compressed(
                    output / "partial-obstruction.npz", **error.partial_raw
                )
                partial["partial_raw_path"] = "partial-obstruction.npz"
                partial["partial_raw_semantically_valid"] = False
            except BaseException as secondary:  # noqa: BLE001 -- retain first failure
                error.add_note("Partial raw publication failure: " + repr(secondary))
        for name, value in (
            ("invalidation.json", {"run_valid": False, "error": repr(error)}),
            ("failure.json", partial),
            ("result.json", partial),
        ):
            try:
                writer(output / name, value)
            except BaseException as secondary:  # noqa: BLE001 -- retain first failure
                error.add_note("Partial publication failure: " + repr(secondary))
        for name in ("result-sha256.json", "output-manifest.json"):
            try:
                (output / name).unlink(missing_ok=True)
            except BaseException as secondary:  # noqa: BLE001 -- retain first failure
                error.add_note("Seal withdrawal failure: " + repr(secondary))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "source-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "source-review-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    result = run(
        a.freeze, a.freeze_sha256, a.source_review, a.source_review_sha256, a.output
    )
    print(
        json.dumps(
            {"run_valid": result["run_valid"], "outcome": result["summary"]["outcome"]}
        )
    )
    return 0 if result["run_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
