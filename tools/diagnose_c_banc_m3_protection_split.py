"""E174: two frozen protection splits of the saved E169 static SOCP.

The source gate and ZIP directory are checked before any archive value is read.
This is a saved affine diagnostic, never a candidate or native gait test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import clarabel
import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tools/diagnose_c_banc_m3_protection_split.py"
TESTS = "tests/test_c_banc_protection_split.py"
KINDS = ("command", "velocity", "startup")
EPS = np.finfo(np.float64).eps


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _member_names(freeze: dict[str, Any]) -> list[str]:
    names = freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
    aliases = freeze["archive_key_map"]
    return [aliases.get(name, name) + ".npy" for name in names]


def preflight(freeze_path: Path, gate_path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Verify frozen inputs, source and runtime before np.load is possible."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require(freeze["schema"] == "flylab.m3.e174.protection-split.v1", "freeze schema")
    require(freeze["family_order"] == ["startup", "muscle"], "family order")
    require(
        gate.get("schema") == "flylab.m3.e174.parent-source-review.v1", "gate schema"
    )
    require(
        gate.get("accepts_single_protection_split_diagnostic") is True,
        "parent acceptance",
    )
    require(sha256(freeze_path) == gate.get("freeze_sha256"), "gate freeze hash")
    require(
        sha256(root / freeze["design"])
        == freeze["design_sha256"]
        == gate.get("design_sha256"),
        "design hash",
    )
    require(
        sha256(root / freeze["base_freeze"]) == freeze["base_freeze_sha256"],
        "base freeze hash",
    )
    require(set(gate.get("source_sha256", {})) == {SOURCE, TESTS}, "gate source names")
    for name in (SOURCE, TESTS):
        require(
            sha256(root / name) == gate["source_sha256"][name],
            f"gate source hash: {name}",
        )
    require(len(freeze["allowed_qr_keys"]) == 43, "43 QR keys")
    require(len(freeze["allowed_problem_keys"]) == 10, "10 problem keys")
    require(len(set(_member_names(freeze))) == 53, "53 distinct selected members")
    require(
        freeze["archive_key_map"]
        == {
            "pair_start": "affine_rows_velocity_start",
            "pair_end": "affine_rows_velocity_end",
        },
        "archive aliases",
    )
    require(
        freeze["shared_input_hash_count"] == len(freeze["shared_input_hashes"]),
        "shared input count",
    )
    require(len(freeze["input_sha256"]) == 64, "64 input pins")
    require(freeze["owned_files"] == [SOURCE, TESTS], "owned files")
    for name, expected in freeze["reused_sources"].items():
        require(sha256(root / name) == expected, f"reused source hash: {name}")
    for name, expected in freeze["input_sha256"].items():
        require(sha256(root / name) == expected, f"input hash: {name}")
    base = json.loads((root / freeze["base_freeze"]).read_text(encoding="utf-8"))
    require(
        base["sample_contract"] == freeze["sample_contract"], "base sample contract"
    )
    for prior_name in (
        "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json",
        "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
    ):
        prior = json.loads((root / prior_name).read_text(encoding="utf-8"))
        require(
            prior["sample_contract"] == freeze["sample_contract"],
            f"prior sample contract: {prior_name}",
        )
        for name, expected in freeze["shared_input_hashes"].items():
            require(
                prior["input_sha256"].get(name) == expected,
                f"prior shared hash: {name}",
            )
    with zipfile.ZipFile(root / freeze["raw"]) as archive:
        infos = [archive.getinfo(name) for name in _member_names(freeze)]
        require(
            all(info.file_size >= 0 and info.compress_size >= 0 for info in infos),
            "ZIP metadata",
        )
    require(np.__version__ == freeze["versions"]["numpy"], "NumPy version")
    require(scipy.__version__ == freeze["versions"]["scipy"], "SciPy version")
    require(clarabel.__version__ == freeze["versions"]["clarabel"], "Clarabel version")
    pools = threadpool_info()
    require(
        freeze["resource_limits"]["cpu_threads"] == 1
        and all(p.get("num_threads") == 1 for p in pools),
        f"CPU1: {pools}",
    )
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        require(os.environ.get(key) == "1", f"CPU1 environment: {key}")
    return {"freeze": freeze, "threadpool_info": pools}


def load_saved(freeze: dict[str, Any], root: Path = ROOT) -> dict[str, np.ndarray]:
    """Call only after preflight; read exactly the selected 53 values."""
    with np.load(root / freeze["raw"], allow_pickle=False) as archive:
        return {
            name: archive[freeze["archive_key_map"].get(name, name)]
            for name in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        }


def _exact(a: np.ndarray, shape: tuple[int, ...], dtype: str, label: str) -> None:
    require(a.shape == shape and a.dtype == np.dtype(dtype), f"{label} shape/dtype")
    require(np.isfinite(a).all(), f"{label} finite")


def validate_saved(
    saved: dict[str, np.ndarray], freeze: dict[str, Any]
) -> dict[str, Any]:
    """Validate original geometry and independent QR/startup row identities."""
    c = freeze["sample_contract"]
    rc = freeze["row_contract"]
    require(c["legs"] == ["lf", "lm", "lh", "rf", "rm", "rh"], "legs")
    require((c["target_count"], c["startup_last_tick"]) == (1824, 99), "sample counts")
    for name, count in (
        ("pair_motors", 990),
        ("pair_filters", 990),
        ("pair_muscles", 990),
        ("w0", 1980),
        ("target_indices", 1824),
        ("pair_start", 1823),
        ("pair_end", 1823),
    ):
        _exact(saved[name], (count,), "float64" if name == "w0" else "int64", name)
    require(
        np.array_equal(saved["pair_motors"], np.repeat(c["variable_motors"], 3)),
        "motor map",
    )
    require(
        np.array_equal(saved["pair_filters"], np.tile(np.arange(3), 330)), "filter map"
    )
    require(
        np.array_equal(
            saved["pair_muscles"].reshape(330, 3),
            np.repeat(saved["pair_muscles"][::3, None], 3, axis=1),
        ),
        "muscle map",
    )
    require(
        np.isin(saved["pair_muscles"], c["variable_muscles"]).all(), "muscle ownership"
    )
    require(
        np.array_equal(saved["target_indices"], np.arange(100, 1924)), "target ticks"
    )
    require(
        np.array_equal(saved["pair_start"], np.arange(100, 1923))
        and np.array_equal(saved["pair_end"], np.arange(101, 1924)),
        "velocity ticks",
    )
    require(
        hashlib.sha256(
            np.ascontiguousarray(saved["target_indices"]).tobytes()
        ).hexdigest()
        == c["target_indices_sha256"],
        "target index hash",
    )
    _exact(saved["problem_A_shape"], (2,), "int64", "A shape")
    require(
        saved["problem_A_shape"].tolist()[1] == rc["original_variable_count"],
        "original columns",
    )
    _exact(
        saved["problem_A_indices"],
        (saved["problem_A_data"].size,),
        "int32",
        "A indices",
    )
    _exact(saved["problem_A_indptr"], (3962,), "int32", "A indptr")
    require(
        saved["problem_A_data"].dtype == np.float64
        and np.isfinite(saved["problem_A_data"]).all(),
        "A data",
    )
    require(
        saved["problem_A_indptr"][0] == 0
        and saved["problem_A_indptr"][-1] == saved["problem_A_data"].size,
        "CSC pointers",
    )
    require(np.all(np.diff(saved["problem_A_indptr"]) >= 0), "CSC monotone")
    dims = np.asarray(saved["problem_cone_dims"], dtype=np.int64)
    kinds = [str(v) for v in saved["problem_cone_kinds"]]
    require(
        dims.size == rc["original_cone_count"] == 16 and len(kinds) == 16, "cone count"
    )
    require(
        dims[0] == rc["original_linear_rows"] == 20682
        and kinds[0] == "NonnegativeConeT",
        "linear cone",
    )
    require(
        all(v == "SecondOrderConeT" for v in kinds[1:]) and np.all(dims[1:] > 1),
        "SOC kinds",
    )
    require(
        saved["problem_A_shape"].tolist() == [int(dims.sum()), 3961], "full A shape"
    )
    for name, count in (
        ("problem_b", int(dims.sum())),
        ("problem_q", 3961),
        ("problem_lower", 3961),
        ("problem_upper", 3961),
    ):
        _exact(saved[name], (count,), "float64", name)
    require(
        np.array_equal(saved["problem_q"], np.r_[np.zeros(3960), -1.0]),
        "original objective",
    )
    require(
        np.array_equal(saved["problem_lower"], np.r_[np.zeros(3960), -1.0]),
        "original lower",
    )
    require(
        np.array_equal(saved["problem_upper"], np.r_[np.full(3960, 2.0), 1.0]),
        "original upper",
    )
    A = sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    eye = sparse.eye(3961, format="csc")
    require(
        (A[:3961] + eye).nnz == 0 and (A[3961:7922] - eye).nnz == 0, "original box rows"
    )
    require(
        np.array_equal(
            saved["problem_b"][:7922],
            np.r_[-saved["problem_lower"], saved["problem_upper"]],
        ),
        "original box rhs",
    )
    owners = np.repeat(saved["pair_muscles"], 2)
    group_legs = np.asarray(c["group_legs"])
    require(
        group_legs.shape == (96,)
        and np.array_equal(np.bincount(group_legs, minlength=6), c["group_counts"]),
        "leg groups",
    )
    for i, leg in enumerate(c["legs"]):
        expected = np.flatnonzero(group_legs[owners] == i)
        for kind in KINDS:
            key = f"qr_{kind}_{leg}"
            active = saved[f"{key}_active"]
            _exact(active, expected.shape, "int64", key + " active")
            require(np.array_equal(active, expected), key + " ownership")
            nrows = min(
                {"command": 1824 * 7, "velocity": 1823 * 7, "startup": 99 * 7}[kind],
                active.size + 1,
            )
            _exact(saved[key], (nrows, active.size + 1), "float64", key)
    require(np.all((saved["w0"] >= 0) & (saved["w0"] <= 2)), "baseline box")
    # The pointwise rows are uncompressed; QR is used only in the SOC.
    plus = A[12366:16524, :]
    minus = A[16524:20682, :]
    require(
        np.allclose(
            (plus[:, :1980] + minus[:, :1980]).toarray(), 0, atol=1e-12, rtol=1e-10
        ),
        "startup opposite rows",
    )
    require(
        plus[:, 1980:3960].nnz == minus[:, 1980:3960].nnz == 0,
        "startup auxiliary columns",
    )
    require(
        np.array_equal(plus[:, 3960].toarray().ravel(), np.ones(4158))
        and np.array_equal(minus[:, 3960].toarray().ravel(), np.ones(4158)),
        "startup t columns",
    )
    require(
        np.allclose(
            saved["problem_b"][12366:16524] + saved["problem_b"][16524:20682],
            2,
            atol=1e-12,
            rtol=1e-10,
        ),
        "startup opposite rhs",
    )
    soc_starts = np.r_[20682, 20682 + np.cumsum(dims[1:])]
    command_M, command_off = _qr_global(saved, "command", freeze)
    command_limit = freeze["contract"]["command_RMS_ratio"] * float(
        np.linalg.norm(command_M @ saved["w0"] + command_off)
    )
    for kind, index, limit in (
        ("command", 0, command_limit),
        ("startup", 14, freeze["contract"]["startup_RMS"] * math.sqrt(99 * 42)),
    ):
        M, off = _qr_global(saved, kind, freeze)
        lo, hi = int(soc_starts[index]), int(soc_starts[index + 1])
        require(hi - lo == M.shape[0] + 1, f"{kind} SOC dimension")
        rows = A[lo:hi]
        require(limit > 0, f"{kind} positive SOC limit")
        require(
            np.allclose(
                rows[1:, :1980].toarray(), -M.toarray() / limit, atol=1e-12, rtol=1e-10
            ),
            f"{kind} SOC QR",
        )
        require(
            rows[1:, 1980:].nnz == 0 and rows[0, :3960].nnz == 0,
            f"{kind} SOC extra columns",
        )
        require(
            np.allclose(
                saved["problem_b"][lo + 1 : hi], off / limit, atol=1e-12, rtol=1e-10
            ),
            f"{kind} SOC offset",
        )
        require(
            abs(saved["problem_b"][lo] - 1) <= 1e-12
            and abs(rows[0, 3960] - 1) <= 1e-12,
            f"{kind} SOC head/t",
        )
    return {
        "A": A,
        "cone_dims": dims,
        "soc_starts": soc_starts,
        "variable_muscles": c["variable_muscles"],
    }


def _qr_global(
    saved: dict[str, np.ndarray], kind: str, freeze: dict[str, Any]
) -> tuple[sparse.csc_matrix, np.ndarray]:
    pieces = []
    offsets = []
    for leg in freeze["sample_contract"]["legs"]:
        key = f"qr_{kind}_{leg}"
        qr = saved[key]
        active = saved[f"{key}_active"]
        loc = sparse.coo_matrix(qr[:, :-1])
        pieces.append(
            sparse.csc_matrix(
                (loc.data, (loc.row, active[loc.col])), shape=(qr.shape[0], 1980)
            )
        )
        offsets.append(qr[:, -1])
    return sparse.vstack(pieces, format="csc"), np.concatenate(offsets)


def build_problem(
    saved: dict[str, np.ndarray], family: str, freeze: dict[str, Any]
) -> dict[str, Any]:
    require(family in ("startup", "muscle"), "family")
    original = sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    b0 = saved["problem_b"]
    dims = np.asarray(saved["problem_cone_dims"], dtype=int)
    starts = np.r_[20682, 20682 + np.cumsum(dims[1:])]
    cols = np.r_[np.arange(1980), 3960] if family == "startup" else np.arange(3961)
    n = cols.size
    guard = freeze["contract"]["inner_guard"]
    lower = np.r_[np.zeros(n - 1), -1.0]
    upper = np.r_[np.full(n - 1, 2.0), 1.0]
    inner_lower = lower.copy()
    inner_upper = upper.copy()
    inner_lower[:1980] = guard
    inner_upper[:1980] -= guard
    box_A = sparse.vstack(
        (-sparse.eye(n, format="csc"), sparse.eye(n, format="csc")), format="csc"
    )
    box_b = np.r_[-inner_lower, inner_upper]
    old_lo, old_hi = (12366, 20682) if family == "startup" else (7922, 12366)
    extra_A = original[old_lo:old_hi, :][:, cols]
    require(
        family != "startup" or original[old_lo:old_hi, 1980:3960].nnz == 0,
        "startup auxiliary omission",
    )
    linear_A = sparse.vstack((box_A, extra_A), format="csc")
    linear_b = np.r_[box_b, b0[old_lo:old_hi]]
    selected_soc = [0, 14] if family == "startup" else [0]
    soc_dims = [int(dims[1 + i]) for i in selected_soc]
    soc_A = [
        original[int(starts[i]) : int(starts[i + 1]), :][:, cols] for i in selected_soc
    ]
    soc_b = [b0[int(starts[i]) : int(starts[i + 1])] for i in selected_soc]
    A = sparse.vstack([linear_A, *soc_A], format="csc")
    b = np.concatenate([linear_b, *soc_b])
    true_b = b.copy()
    true_b[:1980] += guard
    true_b[n : n + 1980] += guard
    if family == "muscle":
        # Original rows 11882:12366 become 7922+(11882-7922):12366.
        start = 2 * n + (11882 - 7922)
        true_b[start : 2 * n + (12366 - 7922)] += freeze["contract"]["physical_atol"]
    q = np.zeros(n)
    q[-1] = -1
    shape = freeze["row_contract"][family]
    require(
        (n, linear_A.shape[0], len(soc_dims))
        == (shape["variable_count"], shape["linear_rows"], shape["soc_count"]),
        "family shape",
    )
    return {
        "A": A,
        "b": b,
        "true_b": true_b,
        "q": q,
        "P": sparse.csc_matrix((n, n)),
        "cones": [
            clarabel.NonnegativeConeT(linear_A.shape[0]),
            *[clarabel.SecondOrderConeT(dim) for dim in soc_dims],
        ],
        "linear_rows": linear_A.shape[0],
        "soc_dims": soc_dims,
        "lower": lower,
        "upper": upper,
        "inner_lower": inner_lower,
        "inner_upper": inner_upper,
        "family": family,
        "old_rows": [old_lo, old_hi],
    }


def physical_diagnostics(
    saved: dict[str, np.ndarray], c: np.ndarray, freeze: dict[str, Any]
) -> dict[str, Any]:
    pair = c.reshape(990, 2)
    pair_min = pair.min(axis=1)
    pair_max = pair.max(axis=1)
    muscle = saved["pair_muscles"]
    owned = np.bincount(muscle[::3], minlength=96)
    mn_min = pair_min.reshape(330, 3).sum(axis=1)
    mn_floor = freeze["contract"]["motor_floor_fraction"] / owned[muscle[::3]]
    mus_min = np.bincount(muscle, weights=pair_min, minlength=96)
    mus_max = np.bincount(muscle, weights=pair_max, minlength=96)
    variable = np.asarray(freeze["sample_contract"]["variable_muscles"])
    tol = freeze["contract"]["physical_atol"]
    return {
        "mn_min": mn_min,
        "mn_floor": mn_floor,
        "muscle_min": mus_min[variable],
        "muscle_max": mus_max[variable],
        "mn_floor_ok": bool(np.all(mn_min >= mn_floor - tol)),
        "muscle_min_ok": bool(
            np.all(mus_min[variable] >= freeze["contract"]["muscle_lower"] - tol)
        ),
        "muscle_max_ok": bool(
            np.all(mus_max[variable] <= freeze["contract"]["muscle_upper"] + tol)
        ),
        "pair_min": pair_min,
        "pair_max": pair_max,
    }


def returned_slacks(problem: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    require(x.shape == problem["q"].shape and np.isfinite(x).all(), "slack point")
    linear = problem["linear_rows"]
    ax = np.asarray(problem["A"] @ x).ravel()
    inner = problem["b"][:linear] - ax[:linear]
    true = problem["true_b"][:linear] - ax[:linear]
    soc = []
    index = linear
    for dim in problem["soc_dims"]:
        value = problem["b"][index : index + dim] - ax[index : index + dim]
        soc.append(
            {
                "head": float(value[0]),
                "tail_norm": float(np.linalg.norm(value[1:])),
                "slack": float(value[0] - np.linalg.norm(value[1:])),
            }
        )
        index += dim
    require(index == problem["b"].size, "SOC partitions")
    return {
        "inner": inner,
        "true": true,
        "soc": soc,
        "summary": {
            "inner_min": float(inner.min()),
            "inner_row": int(inner.argmin()),
            "true_min": float(true.min()),
            "true_row": int(true.argmin()),
            "soc": soc,
        },
    }


def inner_witness(
    saved: dict[str, np.ndarray],
    problem: dict[str, Any],
    family: str,
    freeze: dict[str, Any],
    record: dict[str, Any] | None = None,
) -> np.ndarray:
    record = {} if record is None else record
    baseline = saved["w0"]
    record["baseline_coefficients"] = baseline.copy()
    base_phys = physical_diagnostics(saved, baseline, freeze)
    record["baseline_physical"] = base_phys
    muscles = saved["pair_muscles"]
    owned = np.bincount(muscles[::3], minlength=96)
    g = freeze["contract"]["inner_guard"]
    lam = 6 * g * int(owned.max())
    record["anchor_lambda"] = np.asarray(lam)
    a = np.repeat(1 / (3 * owned[muscles[::3]]), 6)
    record["anchor_a"] = a.copy()
    c = (1 - lam) * baseline + lam * a
    record["anchor_coefficients"] = c.copy()
    if family == "startup":
        x = np.r_[c, freeze["contract"]["inner_witness_margin"]]
    else:
        pair = c.reshape(990, 2)
        x = np.r_[
            c,
            pair.min(axis=1),
            pair.max(axis=1),
            freeze["contract"]["inner_witness_margin"],
        ]
    record["anchor_x"] = x.copy()
    slacks = returned_slacks(problem, x)
    record["anchor_inner_linear_slack"] = slacks["inner"].copy()
    record["anchor_true_linear_slack"] = slacks["true"].copy()
    record["anchor_soc_slack"] = np.asarray([v["slack"] for v in slacks["soc"]])
    record["anchor_slack_summary"] = slacks["summary"]
    require(
        all(
            base_phys[key] for key in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok")
        ),
        "baseline physical",
    )
    require(0 < lam < 1, "anchor lambda")
    require(a.shape == baseline.shape, "anchor shape")
    require(np.all((c >= g) & (c <= 2 - g)), "anchor inner c")
    require(
        slacks["summary"]["inner_min"] >= -1e-12
        and all(v["slack"] >= -1e-12 for v in slacks["soc"]),
        "anchor inner feasibility",
    )
    return x


def project_dual(
    z: np.ndarray, linear_rows: int, soc_dims: list[int], pad: float
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    require(
        z.ndim == 1 and z.size == linear_rows + sum(soc_dims) and np.isfinite(z).all(),
        "dual shape/finite",
    )
    pieces = [np.maximum(z[:linear_rows], 0)]
    index = linear_rows
    for dim in soc_dims:
        cone = z[index : index + dim]
        head, tail = cone[0], cone[1:]
        norm = np.linalg.norm(tail)
        if norm <= head:
            projected = cone.copy()
        elif norm <= -head:
            projected = np.zeros_like(cone)
        else:
            projected = np.r_[0.5 * (norm + head), 0.5 * (1 + head / norm) * tail]
        projected[0] = max(projected[0], np.linalg.norm(projected[1:])) + pad * (
            1 + np.linalg.norm(projected[1:])
        )
        pieces.append(projected)
        index += dim
    return np.concatenate(pieces)


def dual_bound(
    problem: dict[str, Any], z: np.ndarray, true_family: bool, pad: float
) -> dict[str, Any]:
    b = problem["true_b"] if true_family else problem["b"]
    lower = problem["lower"] if true_family else problem["inner_lower"]
    upper = problem["upper"] if true_family else problem["inner_upper"]
    h = np.asarray(problem["q"] + problem["A"].T @ z).ravel()
    terms = np.minimum(h * lower, h * upper)
    b_dot = float(b @ z)
    correction = pad * (1 + abs(b_dot) + float(np.sum(np.abs(terms))))
    result = b_dot - float(terms.sum()) + correction
    require(np.isfinite(result) and np.isfinite(h).all(), "dual bound finite")
    return {
        "upper": result,
        "b_dot_z": b_dot,
        "box_min_sum": float(terms.sum()),
        "pad": correction,
        "stationarity": h,
        "box_terms": terms,
    }


def metrics(
    saved: dict[str, np.ndarray],
    c: np.ndarray,
    freeze: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    output = {}
    for kind in KINDS:
        n_samples = {"command": 1824, "velocity": 1823, "startup": 99}[kind]
        per_leg = {}
        resolutions = []
        for leg in freeze["sample_contract"]["legs"]:
            key = f"qr_{kind}_{leg}"
            qr = saved[key]
            active = saved[key + "_active"]
            M, off = qr[:, :-1], qr[:, -1]
            residual = M @ c[active] + off
            base = M @ saved["w0"][active] + off
            if arrays is not None:
                arrays[f"{kind}_{leg}_returned"] = residual.copy()
                arrays[f"{kind}_{leg}_baseline"] = base.copy()
            norm = float(np.linalg.norm(residual))
            base_norm = float(np.linalg.norm(base))
            per_leg[leg] = {
                "norm": norm,
                "rms": norm / math.sqrt(n_samples * 7),
                "baseline_norm": base_norm,
                "baseline_rms": base_norm / math.sqrt(n_samples * 7),
                "max_abs": float(np.max(np.abs(residual))),
            }
            resolutions.append(
                freeze["contract"]["resolution_factor"]
                * EPS
                * max(M.shape)
                * (np.linalg.norm(M) * np.linalg.norm(c[active]) + np.linalg.norm(off))
                + freeze["contract"]["resolution_absolute_pad"]
            )
        norm = math.sqrt(sum(v["norm"] ** 2 for v in per_leg.values()))
        base = math.sqrt(sum(v["baseline_norm"] ** 2 for v in per_leg.values()))
        output[kind] = {
            "norm": norm,
            "rms": norm / math.sqrt(n_samples * 42),
            "baseline_norm": base,
            "baseline_rms": base / math.sqrt(n_samples * 42),
            "max_abs": max(v["max_abs"] for v in per_leg.values()),
            "resolution_rms": float(
                np.linalg.norm(resolutions) / math.sqrt(n_samples * 42)
            ),
            "legs": per_leg,
        }
    return output


def startup_pointwise(
    saved: dict[str, np.ndarray], c: np.ndarray, freeze: dict[str, Any]
) -> np.ndarray:
    """Recover original 4158 uncompressed startup values from its plus rows."""
    A = sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    scale = freeze["contract"]["startup_max_abs"]
    return scale * (
        np.asarray(A[12366:16524, :1980] @ c).ravel()
        + 1
        - saved["problem_b"][12366:16524]
    )


def returned_witness(
    saved: dict[str, np.ndarray], x: np.ndarray, family: str, freeze: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    c = x[:1980]
    qr_arrays: dict[str, np.ndarray] = {}
    measured = metrics(saved, c, freeze, qr_arrays)
    phys = physical_diagnostics(saved, c, freeze)
    contract = freeze["contract"]
    cmd = measured["command"]
    resolution_ok = (
        cmd["resolution_rms"]
        <= contract["resolution_baseline_fraction"] * cmd["baseline_rms"]
    )
    command_ok = (
        cmd["rms"] + cmd["resolution_rms"]
        <= contract["command_RMS_ratio"] * cmd["baseline_rms"]
    )
    startup = measured["startup"]
    pointwise = startup_pointwise(saved, c, freeze)
    baseline_pointwise = startup_pointwise(saved, saved["w0"], freeze)
    startup["direct_pointwise_max_abs"] = float(np.max(np.abs(pointwise)))
    protect = (
        (
            startup["rms"] + startup["resolution_rms"] <= contract["startup_RMS"]
            and startup["direct_pointwise_max_abs"] <= contract["startup_max_abs"]
        )
        if family == "startup"
        else all(phys[key] for key in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok"))
    )
    finite = bool(np.isfinite(x).all())
    box = finite and bool(np.all((c >= 0) & (c <= 2)))
    t_box = finite and bool(-1 <= x[-1] <= 1)
    witness = {
        "finite": finite,
        "coefficient_box": box,
        "margin_box": t_box,
        "resolution_ok": resolution_ok,
        "command_ok": command_ok,
        "protection_ok": protect,
        "valid": bool(
            finite and box and t_box and resolution_ok and command_ok and protect
        ),
    }
    return witness, {
        "qr": measured,
        "physical": phys,
        "qr_arrays": qr_arrays,
        "startup_pointwise": pointwise,
        "startup_baseline_pointwise": baseline_pointwise,
    }


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    return value


def settings_record(settings: Any) -> dict[str, Any]:
    return {
        key: _serializable(getattr(settings, key))
        for key in dir(settings)
        if not key.startswith("_") and not callable(getattr(settings, key))
    }


def bare_return(solution: Any) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray(solution.x, dtype=np.float64).copy(),
        "s": np.asarray(solution.s, dtype=np.float64).copy(),
        "z": np.asarray(solution.z, dtype=np.float64).copy(),
        "status": np.asarray(str(solution.status)),
        "iterations": np.asarray(solution.iterations),
        "solve_time": np.asarray(solution.solve_time),
    }


def save_raw_first(
    out: Path, family: str, raw: dict[str, np.ndarray], report: dict[str, Any]
) -> Path:
    path = out / f"{family}-solver-raw.npz"
    try:
        np.savez_compressed(path, **raw)
        return path
    except Exception as exc:
        report["secondary_publication_errors"].append(
            {"path": str(path), "error": repr(exc)}
        )
        fallback = out / f"{family}-solver-raw.fallback.npz"
        try:
            np.savez_compressed(fallback, **raw)
            report["families"].setdefault(family, {})["raw_fallback"] = fallback.name
        except Exception as second:  # noqa: BLE001 - preserve failed raw publication
            report["secondary_publication_errors"].append(
                {"path": str(fallback), "error": repr(second)}
            )
        raise


def resource_check(out: Path, started: float, limits: dict[str, Any]) -> None:
    require(
        time.monotonic() - started <= limits["internal_wall_s"], "internal wall limit"
    )
    require(
        sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        <= limits["max_output_bytes"],
        "output size limit",
    )
    require(shutil.disk_usage(out).free >= limits["min_free_bytes"], "free disk")


def save_preparation(out: Path, family: str, record: dict[str, Any]) -> Path:
    """Persist each computed raw preparation array, including failed anchors."""
    path = out / f"{family}-preparation-raw.npz"
    arrays = {
        key: value for key, value in record.items() if isinstance(value, np.ndarray)
    }
    np.savez_compressed(path, **arrays)
    return path


def run(
    freeze_path: Path,
    gate_path: Path,
    out: Path,
    root: Path = ROOT,
    solver_factory: Any = clarabel.DefaultSolver,
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
    out.mkdir(parents=False)
    report: dict[str, Any] = {
        "schema": "flylab.m3.e174.protection-split-result.v1",
        "status": "invalid_not_tested",
        "conclusion": "invalid_not_tested",
        "adoption": "not_applicable",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F1_F8": "pending",
        "new_conic_fits": 0,
        "retries": 0,
        "families": {},
        "primary_error": None,
        "secondary_publication_errors": [],
        "secondary_resource_errors": [],
        "threadpool_info": pre["threadpool_info"],
        "provenance": {
            "freeze_sha256": sha256(freeze_path),
            "gate_sha256": sha256(gate_path),
            "design_sha256": freeze["design_sha256"],
            "input_sha256": freeze["input_sha256"],
            "source_sha256": {name: sha256(root / name) for name in (SOURCE, TESTS)},
        },
    }
    fit_total = 0.0
    try:
        for name in (SOURCE, TESTS):
            shutil.copyfile(root / name, out / ("source-" + Path(name).name))
        saved = load_saved(freeze, root)
        validation = validate_saved(saved, freeze)
        report["validation"] = {
            "cone_dims": validation["cone_dims"].tolist(),
            "variable_muscles": validation["variable_muscles"],
        }
        for family in freeze["family_order"]:
            resource_check(out, started, freeze["resource_limits"])
            problem = build_problem(saved, family, freeze)
            baseline = (
                np.r_[saved["w0"], freeze["contract"]["inner_witness_margin"]]
                if family == "startup"
                else np.r_[
                    saved["w0"],
                    np.min(saved["w0"].reshape(990, 2), axis=1),
                    np.max(saved["w0"].reshape(990, 2), axis=1),
                    freeze["contract"]["inner_witness_margin"],
                ]
            )
            np.savez_compressed(
                out / f"{family}-problem.npz",
                b=problem["b"],
                true_b=problem["true_b"],
                q=problem["q"],
                lower=problem["lower"],
                upper=problem["upper"],
                inner_lower=problem["inner_lower"],
                inner_upper=problem["inner_upper"],
                cone_dims=np.r_[problem["linear_rows"], problem["soc_dims"]],
                baseline=baseline,
            )
            sparse.save_npz(out / f"{family}-A.npz", problem["A"])
            report["families"][family] = {
                "status": "prepared",
                "shape": list(problem["A"].shape),
                "linear_rows": problem["linear_rows"],
                "soc_dims": problem["soc_dims"],
            }
            preparation: dict[str, Any] = {}
            try:
                inner_witness(saved, problem, family, freeze, preparation)
            except Exception:
                report["families"][family]["status"] = "preparation_failed"
                try:
                    path = save_preparation(out, family, preparation)
                    report["families"][family]["preparation_raw_file"] = path.name
                except Exception as publication:  # noqa: BLE001 - retain original failure
                    report["secondary_publication_errors"].append(
                        {
                            "path": str(out / f"{family}-preparation-raw.npz"),
                            "error": repr(publication),
                        }
                    )
                raise
            path = save_preparation(out, family, preparation)
            report["families"][family].update(
                {
                    "preparation_raw_file": path.name,
                    "baseline_physical": _serializable(
                        preparation["baseline_physical"]
                    ),
                    "anchor_slacks": preparation["anchor_slack_summary"],
                }
            )
            settings = clarabel.DefaultSettings()
            for key, value in freeze["contract"]["solver"].items():
                setattr(settings, key, value)
            require(report["new_conic_fits"] < 2, "max two fits")
            report["new_conic_fits"] += 1
            fit_started = time.monotonic()
            solver = solver_factory(
                problem["P"],
                problem["q"],
                problem["A"],
                problem["b"],
                problem["cones"],
                settings,
            )
            solution = solver.solve()
            fit_wall = time.monotonic() - fit_started
            fit_total += fit_wall
            raw = bare_return(solution)
            raw_path = save_raw_first(out, family, raw, report)
            report["families"][family]["raw_file"] = raw_path.name
            report["families"][family]["raw_sha256"] = sha256(raw_path)
            require(
                fit_wall <= freeze["resource_limits"]["family_fit_wall_s"]
                and fit_total <= freeze["resource_limits"]["total_fit_wall_s"],
                "fit wall limit",
            )
            require(
                raw["x"].shape == problem["q"].shape
                and raw["s"].shape == raw["z"].shape == problem["b"].shape,
                "return shape",
            )
            require(
                all(
                    np.isfinite(raw[k]).all()
                    for k in ("x", "s", "z", "iterations", "solve_time")
                ),
                "return finite",
            )
            require(
                str(raw["status"]) in freeze["contract"]["allowed_solver_status"],
                "solver status",
            )
            actual_settings = settings_record(solver.get_settings())
            for key, expected in freeze["contract"]["solver"].items():
                require(
                    actual_settings[key] == expected, f"actual solver setting: {key}"
                )
            x = raw["x"]
            witness, diagnostics = returned_witness(saved, x, family, freeze)
            slacks = returned_slacks(problem, x)
            z = project_dual(
                raw["z"],
                problem["linear_rows"],
                problem["soc_dims"],
                freeze["contract"]["dual_cone_pad"],
            )
            duals = {
                name: dual_bound(
                    problem, z, name == "true", freeze["contract"]["dual_upper_pad"]
                )
                for name in ("inner", "true")
            }
            np.savez_compressed(
                out / f"{family}-derived.npz",
                returned_x=x,
                z_projected=z,
                inner_linear_slack=slacks["inner"],
                true_linear_slack=slacks["true"],
                soc_slack=np.array([item["slack"] for item in slacks["soc"]]),
                stationarity_inner=duals["inner"]["stationarity"],
                stationarity_true=duals["true"]["stationarity"],
                box_terms_inner=duals["inner"]["box_terms"],
                box_terms_true=duals["true"]["box_terms"],
                pair_min=diagnostics["physical"]["pair_min"],
                pair_max=diagnostics["physical"]["pair_max"],
                startup_pointwise=diagnostics["startup_pointwise"],
                startup_baseline_pointwise=diagnostics["startup_baseline_pointwise"],
                **{
                    f"qr_{name}": value
                    for name, value in diagnostics["qr_arrays"].items()
                },
            )
            direct_norm = float(np.linalg.norm(diagnostics["startup_pointwise"]))
            baseline_direct_norm = float(
                np.linalg.norm(diagnostics["startup_baseline_pointwise"])
            )
            qr_norm = diagnostics["qr"]["startup"]["norm"]
            baseline_qr_norm = diagnostics["qr"]["startup"]["baseline_norm"]
            tol = (
                freeze["contract"]["QR_absolute_atol"]
                + freeze["contract"]["QR_relative_atol"] * direct_norm
            )
            baseline_tol = (
                freeze["contract"]["QR_absolute_atol"]
                + freeze["contract"]["QR_relative_atol"] * baseline_direct_norm
            )
            report["families"][family]["startup_direct_qr_norm_check"] = {
                "direct_norm": direct_norm,
                "qr_norm": qr_norm,
                "absolute_gap": abs(direct_norm - qr_norm),
                "allowed_gap": tol,
                "baseline_direct_norm": baseline_direct_norm,
                "baseline_qr_norm": baseline_qr_norm,
                "baseline_absolute_gap": abs(baseline_direct_norm - baseline_qr_norm),
                "baseline_allowed_gap": baseline_tol,
            }
            require(
                abs(direct_norm - qr_norm) <= tol
                and abs(baseline_direct_norm - baseline_qr_norm) <= baseline_tol,
                "startup direct/QR norm mismatch",
            )
            contradiction = witness["valid"] and duals["true"]["upper"] < 0
            conclusion = (
                "invalid_numerical_contradiction"
                if contradiction
                else "supported_saved_affine_witness"
                if witness["valid"]
                else "refuted_saved_protection_family"
                if duals["true"]["upper"] < 0
                else "inconclusive"
            )
            report["families"][family].update(
                {
                    "status": "invalid" if contradiction else "valid",
                    "conclusion": conclusion,
                    "solver_status": str(raw["status"]),
                    "iterations": int(raw["iterations"]),
                    "solve_time_s": float(raw["solve_time"]),
                    "fit_wall_s": fit_wall,
                    "actual_solver_settings": actual_settings,
                    "witness": witness,
                    "metrics": diagnostics["qr"],
                    "physical": _serializable(diagnostics["physical"]),
                    "returned_slacks": slacks["summary"],
                    "duals": {
                        name: {
                            key: value
                            for key, value in dual.items()
                            if key not in ("stationarity", "box_terms")
                        }
                        for name, dual in duals.items()
                    },
                }
            )
            resource_check(out, started, freeze["resource_limits"])
            require(not contradiction, "numerical contradiction")
        report["status"] = "valid_diagnostic"
        report["conclusion"] = {
            name: report["families"][name]["conclusion"]
            for name in freeze["family_order"]
        }
    except Exception as exc:  # noqa: BLE001 - preserve any solver or diagnostic failure
        report["status"] = report["conclusion"] = "invalid_not_tested"
        report["primary_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    report["fit_wall_total_s"] = fit_total
    report["elapsed_s"] = time.monotonic() - started
    try:
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - resource failure invalidates output
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_resource_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
    try:
        json_write(out / "result.json", report)
    except Exception as exc:  # noqa: BLE001 - publication failure invalidates output
        report["secondary_publication_errors"].append(
            {"path": str(out / "result.json"), "error": repr(exc)}
        )
        if report["primary_error"] is None:
            report["primary_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.fallback.json", report)
        except Exception as fallback:  # noqa: BLE001 - retain primary error
            report["secondary_publication_errors"].append(
                {"path": str(out / "result.fallback.json"), "error": repr(fallback)}
            )
    try:
        resource_check(out, started, freeze["resource_limits"])
        manifest = {
            p.name: sha256(p)
            for p in out.iterdir()
            if p.is_file() and p.name != "output-manifest.json"
        }
        json_write(
            out / "output-manifest.json",
            {"schema": "flylab.m3.e174.output-manifest.v1", "files": manifest},
        )
        resource_check(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - late publication failure invalidates output
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_publication_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.json", report)
            manifest = {
                p.name: sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "output-manifest.json"
            }
            json_write(
                out / "output-manifest.json",
                {"schema": "flylab.m3.e174.output-manifest.v1", "files": manifest},
            )
        except Exception as fallback:  # noqa: BLE001 - retain earlier failure
            report["secondary_publication_errors"].append(
                {"path": str(out), "error": repr(fallback)}
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--source-gate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.freeze, args.source_gate, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "conclusion": result["conclusion"],
                "new_conic_fits": result["new_conic_fits"],
            }
        )
    )
    return 0 if result["status"] == "valid_diagnostic" else 1


if __name__ == "__main__":
    sys.exit(main())
