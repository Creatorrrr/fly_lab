"""E175 single saved positive endpoint history SOCP diagnostic.

Original values are unavailable until the parent source gate and metadata pass.
This diagnostic neither refits history nor asserts native gait improvement.
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_c_banc_m3_protection_split as reused

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tools/diagnose_c_banc_m3_endpoint_history.py"
TESTS = "tests/test_c_banc_endpoint_history.py"
KINDS = ("command", "velocity", "startup")
SAMPLES = {"command": 1824, "velocity": 1823, "startup": 99}
N = 5941
LINEAR = 28602
EPS = np.finfo(np.float64).eps


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return reused.sha256(path)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    reused.json_write(path, payload)


def _header(stream: Any) -> dict[str, Any]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported NPY header")
    return {"shape": list(shape), "fortran_order": fortran, "dtype": str(dtype)}


def preflight(freeze_path: Path, gate_path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Check every frozen pin and selected header before opening array values."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require(freeze["schema"] == "flylab.m3.e175.endpoint-history.v1", "freeze schema")
    require(
        gate.get("schema") == "flylab.m3.e175.parent-source-review.v1", "gate schema"
    )
    require(
        gate.get("accepts_single_endpoint_history_diagnostic") is True,
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
    require(freeze["owned_files"] == [SOURCE, TESTS], "owned files")
    require(set(gate.get("source_sha256", {})) == {SOURCE, TESTS}, "gate source names")
    for name in (SOURCE, TESTS):
        require(
            sha256(root / name) == gate["source_sha256"][name],
            f"gate source hash: {name}",
        )
    require(len(freeze["input_sha256"]) == 70, "70 input pins")
    require(
        freeze["shared_input_hash_count"] == len(freeze["shared_input_hashes"]),
        "shared input count",
    )
    for name, expected in freeze["reused_sources"].items():
        require(sha256(root / name) == expected, f"reused source hash: {name}")
    for name, expected in freeze["input_sha256"].items():
        require(sha256(root / name) == expected, f"input hash: {name}")
    base = json.loads((root / freeze["base_freeze"]).read_text(encoding="utf-8"))
    require(
        base["sample_contract"] == freeze["sample_contract"], "base sample contract"
    )
    for name in (
        "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json",
        "verification/banc-natural-gait-m3-20260919/e170-history-recruitment-freeze-v2.json",
    ):
        prior = json.loads((root / name).read_text(encoding="utf-8"))
        require(
            prior["sample_contract"] == freeze["sample_contract"],
            f"sample contract: {name}",
        )
        for pinned, digest in freeze["shared_input_hashes"].items():
            require(
                prior["input_sha256"].get(pinned) == digest, f"shared input: {pinned}"
            )
    require(
        len(freeze["allowed_qr_keys"]) == 43
        and len(freeze["allowed_problem_keys"]) == 10,
        "selected key counts",
    )
    require(
        freeze["archive_key_map"]
        == {
            "pair_start": "affine_rows_velocity_start",
            "pair_end": "affine_rows_velocity_end",
        },
        "archive aliases",
    )
    names = freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
    require(
        len(set(names)) == 53 and set(freeze["selected_header_metadata"]) == set(names),
        "53 selected headers",
    )
    uncompressed = 0
    with zipfile.ZipFile(root / freeze["raw"]) as archive:
        require(
            len(set(archive.namelist())) == len(archive.namelist()),
            "duplicate ZIP members",
        )
        for name in names:
            member = freeze["archive_key_map"].get(name, name) + ".npy"
            info = archive.getinfo(member)
            with archive.open(member) as stream:
                require(
                    _header(stream) == freeze["selected_header_metadata"][name],
                    f"NPY header: {name}",
                )
            uncompressed += info.file_size
    require(
        uncompressed == freeze["selected_uncompressed_bytes"], "selected byte count"
    )
    require(np.__version__ == freeze["versions"]["numpy"], "NumPy version")
    require(scipy.__version__ == freeze["versions"]["scipy"], "SciPy version")
    require(clarabel.__version__ == freeze["versions"]["clarabel"], "Clarabel version")
    pools = threadpool_info()
    require(
        freeze["resource_limits"]["cpu_threads"] == 1
        and all(v.get("num_threads") == 1 for v in pools),
        "CPU1 pools",
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
    """Only call after the complete parent and source preflight succeeds."""
    with np.load(root / freeze["raw"], allow_pickle=False) as archive:
        return {
            name: archive[freeze["archive_key_map"].get(name, name)]
            for name in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]
        }


def transform() -> sparse.csc_matrix:
    """old=(alpha=a,beta=b-a,ell,u,t)."""
    eye = sparse.eye(N, format="csc")
    rows = np.r_[np.arange(1980, 3960), np.arange(1980, 3960)]
    cols = np.r_[np.arange(1980), np.arange(1980, 3960)]
    data = np.r_[-np.ones(1980), np.ones(1980)]
    correction = sparse.csc_matrix((data, (rows, cols)), shape=(N, N))
    return (
        eye
        + correction
        - sparse.diags(
            np.r_[np.zeros(1980), np.ones(1980), np.zeros(1981)], format="csc"
        )
    )


def _global_qr(
    saved: dict[str, np.ndarray], kind: str, legs: list[str]
) -> tuple[sparse.csc_matrix, np.ndarray]:
    pieces, offsets = [], []
    for leg in legs:
        key = f"qr_{kind}_{leg}"
        qr, active = saved[key], saved[key + "_active"]
        half = active.size // 2
        low = active[:half]
        require(np.array_equal(active[half:], low + 1980), f"QR active halves: {key}")
        local = np.c_[qr[:, :half] - qr[:, half:-1], qr[:, half:-1]]
        local_coo = sparse.coo_matrix(local)
        endpoint_active = np.r_[low, low + 1980]
        pieces.append(
            sparse.csc_matrix(
                (local_coo.data, (local_coo.row, endpoint_active[local_coo.col])),
                shape=(qr.shape[0], 3960),
            )
        )
        offsets.append(qr[:, -1])
    return sparse.vstack(pieces, format="csc"), np.concatenate(offsets)


def _criterion_specs(freeze: dict[str, Any]) -> list[tuple[str, list[str]]]:
    legs = freeze["sample_contract"]["legs"]
    return [
        ("command", legs),
        ("velocity", legs),
        *[(kind, [leg]) for leg in legs for kind in ("command", "velocity")],
        ("startup", legs),
    ]


def validate_saved(
    saved: dict[str, np.ndarray], freeze: dict[str, Any]
) -> dict[str, Any]:
    """Validate array geometry, ownership, source rows and all 15 SOC identities."""
    c, rc = freeze["sample_contract"], freeze["row_contract"]
    require(c["legs"] == ["lf", "lm", "lh", "rf", "rm", "rh"], "leg order")
    require((c["target_count"], c["startup_last_tick"]) == (1824, 99), "sample counts")
    for name in freeze["allowed_qr_keys"] + freeze["allowed_problem_keys"]:
        meta = freeze["selected_header_metadata"][name]
        value = saved[name]
        require(
            list(value.shape) == meta["shape"] and str(value.dtype) == meta["dtype"],
            f"saved shape/dtype: {name}",
        )
        if value.dtype.kind in "fi":
            require(np.isfinite(value).all(), f"saved finite: {name}")
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
    muscles = np.asarray(c["variable_muscles"])
    require(
        muscles.size == 77 and np.isin(saved["pair_muscles"], muscles).all(),
        "muscle ownership",
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
        "target hash",
    )
    require(
        saved["problem_A_shape"].tolist()[1] == N
        and saved["problem_A_indptr"].size == N + 1,
        "original columns",
    )
    require(
        saved["problem_A_indptr"][0] == 0
        and saved["problem_A_indptr"][-1] == saved["problem_A_data"].size
        and np.all(np.diff(saved["problem_A_indptr"]) >= 0),
        "CSC pointers",
    )
    dims = np.asarray(saved["problem_cone_dims"], dtype=int)
    require(
        dims.size == 16 and dims[0] == rc["original_linear_rows"] == 24642,
        "original cones",
    )
    require(
        list(saved["problem_cone_kinds"])
        == ["NonnegativeConeT"] + ["SecondOrderConeT"] * 15,
        "original cone kinds",
    )
    require(
        saved["problem_A_shape"].tolist() == [int(dims.sum()), N], "original A shape"
    )
    require(
        np.array_equal(saved["problem_q"], np.r_[np.zeros(N - 1), -1.0]),
        "original objective",
    )
    require(
        np.array_equal(
            saved["problem_lower"], np.r_[np.full(3960, 1e-9), np.zeros(1980), -1.0]
        )
        and np.array_equal(saved["problem_upper"], np.r_[np.full(5940, 2.0), 1.0]),
        "original support",
    )
    require(np.all((saved["w0"] >= 0) & (saved["w0"] <= 2)), "baseline support")
    A = sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    eye = sparse.eye(N, format="csc")
    require(
        (A[:N] + eye).nnz == 0 and (A[N : 2 * N] - eye).nnz == 0, "original box rows"
    )
    require(
        np.array_equal(
            saved["problem_b"][: 2 * N],
            np.r_[-saved["problem_lower"], saved["problem_upper"]],
        ),
        "original box RHS",
    )
    old_p = np.arange(990)
    old_ell, old_u = 3960 + old_p, 4950 + old_p
    old_a = (2 * old_p, 2 * old_p + 1)
    old_beta = (1980 + 2 * old_p, 1981 + 2 * old_p)
    rows, cols, values = [], [], []
    for block, endpoint in enumerate(old_a):
        row = 11882 + 990 * block + old_p
        rows.extend((row, row))
        cols.extend((old_ell, endpoint))
        values.extend((np.ones(990), -np.ones(990)))
    for block, (endpoint, beta) in enumerate(zip(old_a, old_beta, strict=True)):
        row = 13862 + 990 * block + old_p
        rows.extend((row, row, row))
        cols.extend((endpoint, beta, old_u))
        values.extend((np.ones(990), np.ones(990), -np.ones(990)))
    expected_epi = sparse.csc_matrix(
        (np.concatenate(values), (np.concatenate(rows) - 11882, np.concatenate(cols))),
        shape=(3960, N),
    )
    require(
        (A[11882:15842] - expected_epi).nnz == 0
        and np.array_equal(saved["problem_b"][11882:15842], np.zeros(3960)),
        "original epigraph",
    )
    count = np.bincount(saved["pair_muscles"][::3], minlength=96)
    selected = np.asarray(c["variable_muscles"])
    mus_index = np.searchsorted(selected, saved["pair_muscles"])
    require(
        np.array_equal(selected[mus_index], saved["pair_muscles"]), "physical owner map"
    )
    expected_phys = sparse.vstack(
        (
            sparse.csc_matrix(
                (-np.ones(990), (np.repeat(np.arange(330), 3), old_ell)), shape=(330, N)
            ),
            sparse.csc_matrix((-np.ones(990), (mus_index, old_ell)), shape=(77, N)),
            sparse.csc_matrix((np.ones(990), (mus_index, old_u)), shape=(77, N)),
        ),
        format="csc",
    )
    require((A[15842:16326] - expected_phys).nnz == 0, "original physical rows")
    expected_rhs = np.r_[
        -freeze["contract"]["motor_floor_fraction"] / count[saved["pair_muscles"][::3]],
        np.full(77, -freeze["contract"]["muscle_lower"]),
        np.full(77, freeze["contract"]["muscle_upper"]),
    ]
    atol, rtol = (
        freeze["contract"]["QR_absolute_atol"],
        freeze["contract"]["QR_relative_atol"],
    )
    require(
        np.all(
            np.abs(saved["problem_b"][15842:16326] - expected_rhs)
            <= atol + rtol * np.abs(expected_rhs)
        ),
        "original physical RHS",
    )
    owners = np.repeat(saved["pair_muscles"], 2)
    group_legs = np.asarray(c["group_legs"])
    require(
        group_legs.shape == (96,)
        and np.array_equal(np.bincount(group_legs, minlength=6), c["group_counts"]),
        "leg groups",
    )
    for j, leg in enumerate(c["legs"]):
        expected = np.flatnonzero(group_legs[owners] == j)
        for kind in KINDS:
            key = f"qr_{kind}_{leg}"
            active = saved[key + "_active"]
            require(
                np.array_equal(active, np.r_[expected, expected + 1980]),
                f"QR ownership: {key}",
            )
            nrows = min(SAMPLES[kind] * 7, active.size + 1)
            require(saved[key].shape == (nrows, active.size + 1), f"QR shape: {key}")
    plus = A[16326:20484]
    minus = A[20484:24642]
    difference = (plus[:, :3960] + minus[:, :3960]).tocoo()
    require(
        np.allclose(difference.data, 0, atol=atol, rtol=rtol),
        "startup +/- coefficients",
    )
    require(
        plus[:, 3960:5940].nnz == minus[:, 3960:5940].nnz == 0, "startup auxiliary zero"
    )
    require(
        np.array_equal(plus[:, -1].toarray().ravel(), np.ones(4158))
        and np.array_equal(minus[:, -1].toarray().ravel(), np.ones(4158)),
        "startup t columns",
    )
    require(
        np.all(
            np.abs(
                saved["problem_b"][16326:20484] + saved["problem_b"][20484:24642] - 2
            )
            <= atol + rtol * 2
        ),
        "startup RHS sum",
    )
    starts = np.r_[24642, 24642 + np.cumsum(dims[1:])]
    specs = _criterion_specs(freeze)
    baseline = np.r_[saved["w0"], saved["w0"]]
    limits = []
    for j, (kind, legs) in enumerate(specs):
        M, offset = _global_qr(saved, kind, legs)
        base = np.asarray(M @ baseline + offset).ravel()
        if j == 0:
            limit = freeze["contract"]["command_RMS_ratio"] * np.linalg.norm(base)
        elif j == 14:
            limit = freeze["contract"]["startup_RMS"] * math.sqrt(99 * 42)
        else:
            limit = np.linalg.norm(base)
        require(np.isfinite(limit) and limit > 0, f"SOC limit: {j}")
        lo, hi = int(starts[j]), int(starts[j + 1])
        require(hi - lo == M.shape[0] + 1, f"SOC dimension: {j}")
        rows = A[lo:hi]
        require(
            abs(saved["problem_b"][lo] - 1) <= atol
            and abs(rows[0, -1] - 1) <= atol
            and rows[0, :-1].nnz == 0,
            f"SOC head: {j}",
        )
        require(rows[1:, 3960:].nnz == 0, f"SOC extra columns: {j}")
        old_M = sparse.hstack((M[:, :1980] + M[:, 1980:], M[:, 1980:]), format="csc")
        expected_matrix = -old_M / limit
        difference = (rows[1:, :3960] - expected_matrix).tocoo()
        if difference.nnz:
            reference = np.asarray(
                expected_matrix.tocsr()[difference.row, difference.col]
            ).ravel()
            require(
                np.all(np.abs(difference.data) <= atol + rtol * np.abs(reference)),
                f"SOC QR: {j}",
            )
        expected_offset = offset / limit
        require(
            np.all(
                np.abs(saved["problem_b"][lo + 1 : hi] - expected_offset)
                <= atol + rtol * np.abs(expected_offset)
            ),
            f"SOC offset: {j}",
        )
        limits.append(float(limit))
    return {"A": A, "dims": dims, "starts": starts, "limits": limits, "specs": specs}


def physical(
    saved: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray, freeze: dict[str, Any]
) -> dict[str, Any]:
    endpoints = np.stack((a.reshape(990, 2), b.reshape(990, 2)), axis=0)
    minimum, maximum = endpoints.min(axis=(0, 2)), endpoints.max(axis=(0, 2))
    owner = saved["pair_muscles"]
    count = np.bincount(owner[::3], minlength=96)
    mn = minimum.reshape(330, 3).sum(axis=1)
    floor = freeze["contract"]["motor_floor_fraction"] / count[owner[::3]]
    mus_min = np.bincount(owner, weights=minimum, minlength=96)
    mus_max = np.bincount(owner, weights=maximum, minlength=96)
    selected = np.asarray(freeze["sample_contract"]["variable_muscles"])
    c = freeze["contract"]
    tol = c["physical_atol"]
    return {
        "four_endpoints": endpoints,
        "pair_min": minimum,
        "pair_max": maximum,
        "mn_min": mn,
        "mn_floor": floor,
        "muscle_min": mus_min[selected],
        "muscle_max": mus_max[selected],
        "mn_floor_ok": bool(np.all(mn >= floor - tol)),
        "muscle_min_ok": bool(np.all(mus_min[selected] >= c["muscle_lower"] - tol)),
        "muscle_max_ok": bool(np.all(mus_max[selected] <= c["muscle_upper"] + tol)),
    }


def build_problem(
    saved: dict[str, np.ndarray],
    freeze: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validate_saved(saved, freeze) if validation is None else validation
    rc, c = freeze["row_contract"], freeze["contract"]
    old, T = validation["A"], transform()
    lower = np.r_[np.zeros(5940), -1.0]
    upper = np.r_[np.full(5940, 2.0), 1.0]
    inner_lower, inner_upper = lower.copy(), upper.copy()
    inner_lower[:3960] = c["inner_guard"]
    inner_upper[:3960] -= c["inner_guard"]
    box = sparse.vstack(
        (-sparse.eye(N, format="csc"), sparse.eye(N, format="csc")), format="csc"
    )
    p = np.arange(990)
    endpoint_cols = (2 * p, 2 * p + 1, 1980 + 2 * p, 1981 + 2 * p)
    ell, u = 3960 + p, 4950 + p
    rr, cc, dd = [], [], []
    for side, aux in ((0, ell), (1, u)):
        for block, endpoint in enumerate(endpoint_cols):
            row = (side * 4 + block) * 990 + p
            rr.extend((row, row))
            cc.extend((aux, endpoint))
            dd.extend(
                (
                    np.full(990, 1 if side == 0 else -1),
                    np.full(990, -1 if side == 0 else 1),
                )
            )
    envelope = sparse.csc_matrix(
        (np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))), shape=(7920, N)
    )
    require(
        rc["envelope_rows"] == [11882, 19802]
        and rc["physical_rows"] == [19802, 20286]
        and rc["startup_rows"] == [20286, 28602],
        "new row contract",
    )
    physical_rows = old[15842:16326] @ T
    require(physical_rows[:, :3960].nnz == 0, "physical coefficients")
    startup_rows = old[16326:24642] @ T
    soc_rows = old[24642:] @ T
    A = sparse.vstack(
        (box, envelope, physical_rows, startup_rows, soc_rows), format="csc"
    )
    bvec = np.r_[
        -inner_lower,
        inner_upper,
        np.full(7920, -c["envelope_guard"]),
        saved["problem_b"][15842:16326],
        saved["problem_b"][16326:],
    ]
    bvec[19802:20286] -= c["physical_guard"]
    true_b = bvec.copy()
    true_b[:3960] += c["inner_guard"]
    true_b[N : N + 3960] += c["inner_guard"]
    true_b[11882:19802] += c["envelope_guard"]
    true_b[19802:20286] += c["physical_guard"] + c["physical_atol"]
    q = np.r_[np.zeros(N - 1), -1.0]
    dims = [int(v) for v in validation["dims"][1:]]
    require(
        A.shape == (LINEAR + sum(dims), N) and bvec.shape == (A.shape[0],),
        "new A/b shape",
    )
    return {
        "A": A,
        "b": bvec,
        "true_b": true_b,
        "q": q,
        "P": sparse.csc_matrix((N, N)),
        "cones": [
            clarabel.NonnegativeConeT(LINEAR),
            *[clarabel.SecondOrderConeT(v) for v in dims],
        ],
        "linear_rows": LINEAR,
        "soc_dims": dims,
        "lower": lower,
        "upper": upper,
        "inner_lower": inner_lower,
        "inner_upper": inner_upper,
        "T": T,
    }


def slacks(problem: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    return reused.returned_slacks(problem, x)


def anchor(
    saved: dict[str, np.ndarray], problem: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    c = freeze["contract"]
    owner = saved["pair_muscles"]
    count = np.bincount(owner[::3], minlength=96)
    uniform = np.repeat(1 / (3 * count[owner]), 2)
    v = (1 - c["inner_witness_mix"]) * saved["w0"] + c["inner_witness_mix"] * uniform
    endpoints = v.reshape(990, 2)
    x = np.r_[
        v,
        v,
        endpoints.min(axis=1) - c["envelope_guard"],
        endpoints.max(axis=1) + c["envelope_guard"],
        c["inner_witness_margin"],
    ]
    base = physical(saved, saved["w0"], saved["w0"], freeze)
    derived = physical(saved, v, v, freeze)
    checked = slacks(problem, x)
    return {
        "x": x,
        "uniform": uniform,
        "v": v,
        "baseline_physical": base,
        "anchor_physical": derived,
        "slacks": checked,
    }


def inner_witness(
    saved: dict[str, np.ndarray],
    problem: dict[str, Any],
    freeze: dict[str, Any],
    record: dict[str, Any] | None = None,
) -> np.ndarray:
    prepared = anchor(saved, problem, freeze)
    if record is not None:
        record.update(prepared)
    require(anchor_valid(prepared), "anchor inner feasibility")
    return prepared["x"]


def anchor_valid(record: dict[str, Any]) -> bool:
    base = record["baseline_physical"]
    checks = record["slacks"]
    return bool(
        all(base[k] for k in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok"))
        and np.isfinite(record["x"]).all()
        and checks["summary"]["inner_min"] >= -1e-12
        and all(v["slack"] >= -1e-12 for v in checks["soc"])
    )


def startup_pointwise(
    saved: dict[str, np.ndarray], c: np.ndarray, freeze: dict[str, Any]
) -> np.ndarray:
    """Recover uncompressed startup vector; c contains a then b endpoints."""
    require(c.shape == (3960,), "startup coefficient shape")
    old_a = sparse.csc_matrix(
        (
            saved["problem_A_data"],
            saved["problem_A_indices"],
            saved["problem_A_indptr"],
        ),
        shape=tuple(saved["problem_A_shape"]),
    )
    old_coeff = np.r_[c[:1980], c[1980:] - c[:1980]]
    scale = freeze["contract"]["startup_max_abs"]
    return scale * (
        np.asarray(old_a[16326:20484, :3960] @ old_coeff).ravel()
        + 1
        - saved["problem_b"][16326:20484]
    )


def metrics(
    saved: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray, freeze: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result, arrays = {}, {}
    endpoint = np.r_[a, b]
    baseline = np.r_[saved["w0"], saved["w0"]]
    c = freeze["contract"]
    for kind in KINDS:
        legs = {}
        for leg in freeze["sample_contract"]["legs"]:
            key = f"qr_{kind}_{leg}"
            qr, active = saved[key], saved[key + "_active"]
            half = active.size // 2
            M = np.c_[qr[:, :half] - qr[:, half:-1], qr[:, half:-1]]
            off = qr[:, -1]
            cols = np.r_[active[:half], active[:half] + 1980]
            residual = M @ endpoint[cols] + off
            base = M @ baseline[cols] + off
            arrays[f"{kind}_{leg}_returned"] = residual
            arrays[f"{kind}_{leg}_baseline"] = base
            denom = math.sqrt(SAMPLES[kind] * 7)
            resolution = (
                c["resolution_factor"]
                * EPS
                * max(M.shape)
                * (
                    np.linalg.norm(M) * np.linalg.norm(endpoint[cols])
                    + np.linalg.norm(off)
                )
                + c["resolution_absolute_pad"]
            )
            legs[leg] = {
                "rms": float(np.linalg.norm(residual) / denom),
                "baseline_rms": float(np.linalg.norm(base) / denom),
                "resolution_rms": float(resolution / denom),
                "norm": float(np.linalg.norm(residual)),
                "baseline_norm": float(np.linalg.norm(base)),
                "resolution_norm": float(resolution),
            }
        denom = math.sqrt(SAMPLES[kind] * 42)
        result[kind] = {
            "rms": math.sqrt(sum(v["norm"] ** 2 for v in legs.values())) / denom,
            "baseline_rms": math.sqrt(
                sum(v["baseline_norm"] ** 2 for v in legs.values())
            )
            / denom,
            "resolution_rms": math.sqrt(
                sum(v["resolution_norm"] ** 2 for v in legs.values())
            )
            / denom,
            "legs": legs,
        }
    return result, arrays


def _returned_witness(
    saved: dict[str, np.ndarray], x: np.ndarray, freeze: dict[str, Any]
) -> dict[str, Any]:
    a, b = x[:1980], x[1980:3960]
    measured, arrays = metrics(saved, a, b, freeze)
    base_x = np.r_[saved["w0"], saved["w0"], np.zeros(1980), 0.0]
    direct = startup_pointwise(saved, x[:3960], freeze)
    direct_base = startup_pointwise(saved, base_x[:3960], freeze)
    qr_norm = measured["startup"]["rms"] * math.sqrt(99 * 42)
    baseline_qr_norm = measured["startup"]["baseline_rms"] * math.sqrt(99 * 42)
    atol, rtol = (
        freeze["contract"]["QR_absolute_atol"],
        freeze["contract"]["QR_relative_atol"],
    )
    direct_norm = float(np.linalg.norm(direct))
    baseline_direct_norm = float(np.linalg.norm(direct_base))
    norm_ok = bool(
        abs(direct_norm - qr_norm) <= atol + rtol * qr_norm
        and abs(baseline_direct_norm - baseline_qr_norm)
        <= atol + rtol * baseline_qr_norm
    )
    phys = physical(saved, a, b, freeze)
    c = freeze["contract"]
    resolutions = []
    rms_checks = []
    for kind in ("command", "velocity"):
        item = measured[kind]
        resolutions.append(
            item["resolution_rms"]
            <= c["resolution_baseline_fraction"] * item["baseline_rms"]
        )
        ratio = (
            c["command_RMS_ratio"]
            if kind == "command"
            else c["global_velocity_RMS_ratio"]
        )
        rms_checks.append(
            item["rms"] + item["resolution_rms"] <= ratio * item["baseline_rms"]
        )
        for leg in freeze["sample_contract"]["legs"]:
            entry = item["legs"][leg]
            resolutions.append(
                entry["resolution_rms"]
                <= c["resolution_baseline_fraction"] * entry["baseline_rms"]
            )
            ratio = (
                c["per_leg_command_RMS_ratio"]
                if kind == "command"
                else c["per_leg_velocity_RMS_ratio"]
            )
            rms_checks.append(
                entry["rms"] + entry["resolution_rms"] <= ratio * entry["baseline_rms"]
            )
    start = measured["startup"]
    physical_ok = all(
        phys[k] for k in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok")
    )
    support = bool(np.all((x[:3960] >= 0) & (x[:3960] <= 2)) and -1 <= x[-1] <= 1)
    auxiliary_support = bool(np.all((x[3960:5940] >= 0) & (x[3960:5940] <= 2)))
    valid = bool(
        np.isfinite(x).all()
        and support
        and physical_ok
        and norm_ok
        and all(resolutions)
        and all(rms_checks)
        and start["rms"] + start["resolution_rms"] <= c["startup_RMS"]
        and np.max(np.abs(direct)) <= c["startup_max_abs"]
    )
    return {
        "valid": valid,
        "finite": bool(np.isfinite(x).all()),
        "support": support,
        "auxiliary_support": auxiliary_support,
        "physical_ok": physical_ok,
        "resolution_ok": bool(all(resolutions)),
        "rms_ok": bool(all(rms_checks)),
        "startup_norm_ok": norm_ok,
        "startup_norm_comparison": {
            "returned_direct_norm": direct_norm,
            "returned_qr_norm": qr_norm,
            "returned_reference_norm": qr_norm,
            "baseline_direct_norm": baseline_direct_norm,
            "baseline_qr_norm": baseline_qr_norm,
            "baseline_reference_norm": baseline_qr_norm,
            "absolute_atol": atol,
            "relative_atol": rtol,
        },
        "startup_max_abs": float(np.max(np.abs(direct))),
        "metrics": measured,
        "physical": phys,
        "qr_arrays": arrays,
        "startup_direct": direct,
        "startup_baseline_direct": direct_base,
    }


def returned_witness(
    saved: dict[str, np.ndarray], x: np.ndarray, freeze: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _returned_witness(saved, x, freeze)
    diagnostics = {
        k: result[k]
        for k in (
            "metrics",
            "physical",
            "qr_arrays",
            "startup_direct",
            "startup_baseline_direct",
        )
    }
    witness_flags = {k: v for k, v in result.items() if k not in diagnostics}
    return witness_flags, diagnostics


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _resource(out: Path, started: float, limits: dict[str, Any]) -> None:
    reused.resource_check(out, started, limits)


def _publish(
    out: Path, report: dict[str, Any], limits: dict[str, Any], started: float
) -> None:
    """Keep the first failure authoritative across result and manifest updates."""
    try:
        json_write(out / "result.json", report)
    except Exception as exc:  # noqa: BLE001
        report["secondary_publication_errors"].append(
            {"path": "result.json", "error": repr(exc)}
        )
        report["primary_error"] = report["primary_error"] or {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.fallback.json", report)
        except Exception as second:  # noqa: BLE001
            report["secondary_publication_errors"].append(
                {"path": "result.fallback.json", "error": repr(second)}
            )
    try:
        _resource(out, started, limits)
    except Exception as exc:  # noqa: BLE001
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_resource_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.json", report)
        except Exception as second:  # noqa: BLE001
            report["secondary_publication_errors"].append(
                {"path": "result.json", "error": repr(second)}
            )
    try:
        files = {
            p.name: sha256(p)
            for p in out.iterdir()
            if p.is_file() and p.name != "output-manifest.json"
        }
        json_write(
            out / "output-manifest.json",
            {"schema": "flylab.m3.e175.output-manifest.v1", "files": files},
        )
        _resource(out, started, limits)
    except Exception as exc:  # noqa: BLE001
        item = {"type": type(exc).__name__, "message": str(exc)}
        if report["primary_error"] is None:
            report["primary_error"] = item
        else:
            report["secondary_publication_errors"].append(item)
        report["status"] = report["conclusion"] = "invalid_not_tested"
        try:
            json_write(out / "result.json", report)
            files = {
                p.name: sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "output-manifest.json"
            }
            json_write(
                out / "output-manifest.json",
                {"schema": "flylab.m3.e175.output-manifest.v1", "files": files},
            )
        except Exception as second:  # noqa: BLE001
            report["secondary_publication_errors"].append(
                {"path": "final publication", "error": repr(second)}
            )


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
        "schema": "flylab.m3.e175.endpoint-history-result.v1",
        "status": "invalid_not_tested",
        "conclusion": "invalid_not_tested",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F1_F8": "pending",
        "new_conic_fits": 0,
        "new_qr_factorizations": 0,
        "new_svd_fits": 0,
        "retries": 0,
        "original_values_loaded_before_gate": 0,
        "primary_error": None,
        "secondary_publication_errors": [],
        "secondary_resource_errors": [],
        "families": {},
        "threadpool_info": pre["threadpool_info"],
        "provenance": {
            "freeze_sha256": sha256(freeze_path),
            "gate_sha256": sha256(gate_path),
            "design_sha256": freeze["design_sha256"],
            "source_sha256": {name: sha256(root / name) for name in (SOURCE, TESTS)},
            "reused_sources": freeze["reused_sources"],
            "raw_sha256": freeze["input_sha256"][freeze["raw"]],
        },
    }
    fit_wall = 0.0
    try:
        for name in (SOURCE, TESTS):
            shutil.copyfile(root / name, out / ("source-" + Path(name).name))
        for name in freeze["reused_sources"]:
            shutil.copyfile(root / name, out / ("source-" + Path(name).name))
        saved = load_saved(freeze, root)
        validation = validate_saved(saved, freeze)
        problem = build_problem(saved, freeze, validation)
        sparse.save_npz(out / "endpoint-A.npz", problem["A"])
        _save_npz(
            out / "endpoint-problem.npz",
            {
                **{
                    k: problem[k]
                    for k in (
                        "b",
                        "true_b",
                        "q",
                        "lower",
                        "upper",
                        "inner_lower",
                        "inner_upper",
                    )
                },
                "cone_dims": np.r_[LINEAR, problem["soc_dims"]],
            },
        )
        prepared = anchor(saved, problem, freeze)
        checks = prepared["slacks"]
        _save_npz(
            out / "endpoint-preparation-raw.npz",
            {
                "baseline_coefficients": saved["w0"],
                "anchor_uniform": prepared["uniform"],
                "anchor_a": prepared["v"],
                "anchor_b": prepared["v"],
                "anchor_x": prepared["x"],
                "anchor_inner_linear_slack": checks["inner"],
                "anchor_true_linear_slack": checks["true"],
                "anchor_soc_slack": np.asarray(
                    [item["slack"] for item in checks["soc"]]
                ),
                **{
                    f"baseline_physical_{key}": value
                    for key, value in prepared["baseline_physical"].items()
                    if isinstance(value, np.ndarray)
                },
                **{
                    f"anchor_physical_{key}": value
                    for key, value in prepared["anchor_physical"].items()
                    if isinstance(value, np.ndarray)
                },
            },
        )
        report["preparation"] = {
            "anchor_valid": anchor_valid(prepared),
            "anchor_slack": checks["summary"],
            "baseline_physical_ok": bool(
                all(
                    prepared["baseline_physical"][k]
                    for k in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok")
                )
            ),
        }
        require(report["preparation"]["baseline_physical_ok"], "baseline physical")
        require(report["preparation"]["anchor_valid"], "anchor inner feasibility")
        _resource(out, started, freeze["resource_limits"])
        settings = clarabel.DefaultSettings()
        for key, value in freeze["contract"]["solver"].items():
            setattr(settings, key, value)
        report["new_conic_fits"] = 1
        fit_start = time.monotonic()
        solver = solver_factory(
            problem["P"],
            problem["q"],
            problem["A"],
            problem["b"],
            problem["cones"],
            settings,
        )
        solution = solver.solve()
        fit_wall = time.monotonic() - fit_start
        raw = reused.bare_return(solution)
        raw_path = reused.save_raw_first(out, "endpoint", raw, report)
        report["solver_raw_file"] = raw_path.name
        report["solver_raw_sha256"] = sha256(raw_path)
        require(
            fit_wall <= freeze["resource_limits"]["family_fit_wall_s"], "fit wall limit"
        )
        require(
            raw["x"].shape == problem["q"].shape
            and raw["s"].shape == problem["b"].shape
            and raw["z"].shape == problem["b"].shape,
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
        settings_actual = reused.settings_record(solver.get_settings())
        for key, expected in freeze["contract"]["solver"].items():
            require(settings_actual[key] == expected, f"actual setting: {key}")
        x = raw["x"]
        checked = slacks(problem, x)
        witness_flags, observed = returned_witness(saved, x, freeze)
        projected = reused.project_dual(
            raw["z"], LINEAR, problem["soc_dims"], freeze["contract"]["dual_cone_pad"]
        )
        dual = {
            name: reused.dual_bound(
                problem,
                projected if name.startswith("projected") else raw["z"],
                name.endswith("true"),
                freeze["contract"]["dual_upper_pad"],
            )
            for name in ("raw_inner", "raw_true", "projected_inner", "projected_true")
        }
        _save_npz(
            out / "endpoint-derived.npz",
            {
                "returned_x": x,
                "inner_linear_slack": checked["inner"],
                "true_linear_slack": checked["true"],
                "soc_slack": np.asarray([item["slack"] for item in checked["soc"]]),
                "startup_direct": observed["startup_direct"],
                "startup_baseline_direct": observed["startup_baseline_direct"],
                "z_projected": projected,
                **{f"qr_{key}": value for key, value in observed["qr_arrays"].items()},
                **{
                    f"physical_{key}": value
                    for key, value in observed["physical"].items()
                    if isinstance(value, np.ndarray)
                },
                **{
                    f"stationarity_{name}": value["stationarity"]
                    for name, value in dual.items()
                },
                **{
                    f"box_terms_{name}": value["box_terms"]
                    for name, value in dual.items()
                },
            },
        )
        report["witness"] = witness_flags
        report["startup_norm_comparison"] = witness_flags["startup_norm_comparison"]
        require(witness_flags["startup_norm_ok"], "startup direct/QR norm mismatch")
        upper = dual["projected_true"]["upper"]
        require(not (witness_flags["valid"] and upper < 0), "numerical contradiction")
        conclusion = (
            "supported_saved_endpoint_witness"
            if witness_flags["valid"]
            else "refuted_saved_endpoint_family"
            if math.isfinite(upper) and upper < 0
            else "inconclusive"
        )
        report.update(
            {
                "status": "valid_diagnostic",
                "conclusion": conclusion,
                "solver_status": str(raw["status"]),
                "solver_iterations": int(raw["iterations"]),
                "solver_time_s": float(raw["solve_time"]),
                "actual_solver_settings": settings_actual,
                "returned_slacks": checked["summary"],
                "witness": witness_flags,
                "metrics": observed["metrics"],
                "physical_ok": {
                    key: observed["physical"][key]
                    for key in ("mn_floor_ok", "muscle_min_ok", "muscle_max_ok")
                },
                "dual": {
                    name: {
                        k: v
                        for k, v in value.items()
                        if k not in ("stationarity", "box_terms")
                    }
                    for name, value in dual.items()
                },
            }
        )
        _resource(out, started, freeze["resource_limits"])
    except Exception as exc:  # noqa: BLE001 - preserve first numerical, solver or publication error
        report["status"] = report["conclusion"] = "invalid_not_tested"
        report["primary_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    report["fit_wall_s"] = fit_wall
    report["elapsed_s"] = time.monotonic() - started
    _publish(out, report, freeze["resource_limits"], started)
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
                "new_conic_fits": result["new_conic_fits"],
            }
        )
    )
    return 0 if result["status"] == "valid_diagnostic" else 1


if __name__ == "__main__":
    sys.exit(main())
