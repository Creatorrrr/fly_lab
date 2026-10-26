"""E145 single global SOCP for a conditional static BANC motor-pool proposal.

This program only reads frozen saved arrays.  It performs one CPU Clarabel
solve and no neural, CUDA, physics, EMA, or LP work.  A returned optimizer
point is assessed again with the original physical expressions; solver status
alone never makes it a proposal or completes F1--F8.
"""

# ruff: noqa: BLE001, I001  # env limits must precede numerical imports

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import clarabel  # thread limits must precede numerical imports
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from fit_c_banc_m3_static_pool import (
    LEGS,
    assess_composite,
    build_affine_design,
    build_leg_layout,
    composite_metrics_to_raw,
    evaluate_composite,
    load_inputs,
    validate_pool_contract,
    validate_saved_inputs,
    velocity_affine_design,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "abc3d2c27dfa1554019c1552e83aa525daf46019242c4a4721eca8ea1cef3da7"
DESIGN_SHA = "291aa6998043403dd114433e75e494b07e2c66c83308793592293bcbea99fac6"
SCHEMA = "flylab.m3.e145.global-conic-pool.v1"
OWNED = ("tools/fit_c_banc_m3_conic_pool.py", "tests/test_c_banc_conic_pool.py")
DEFAULT_FREEZE = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e145-global-conic-freeze.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_path(relative: str, expected: str) -> None:
    path = Path(relative)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"Unsafe frozen path: {relative}",
    )
    require(sha256(ROOT / path) == expected, f"Frozen hash mismatch: {relative}")


def save_json_new(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz_new(path: Path, arrays: dict[str, Any]) -> None:
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def add_secondary(
    primary: BaseException, operation: str, secondary: BaseException
) -> None:
    primary.add_note(
        f"Secondary E145 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E145_GLOBAL_CONIC_POOL_DESIGN_20260921.md",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "cpu_threads": 1,
        "gpu_count": 0,
        "original_processes": 1,
        "retries": 0,
        "max_optimizer_calls": 1,
        "solver": "Clarabel",
        "solver_version": "0.11.1",
        "solver_settings": {
            "max_threads": 1,
            "direct_solve_method": "qdldl",
            "max_iter": 100,
            "time_limit": 30.0,
            "verbose": False,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "tol_infeas_abs": 1e-10,
            "tol_infeas_rel": 1e-10,
        },
        "margin_bounds": [-1.0, 1.0],
        "total_fit_limit_s": 30,
        "internal_wall_limit_s": 60,
        "external_wall_limit_s": 65,
        "supervisor_limit_s": 70,
        "prior_conservative_process_s": 2179,
        "prospective_cumulative_limit_s": 2249,
        "original_program_cumulative_limit_s": 2271,
        "maximum_raw_bytes": 134217728,
        "minimum_free_disk_bytes": 1073741824,
        "original_neural_steps": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 0,
        "LP_calls": 0,
        "candidate_adopted": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
        "dual_interior_pad": 1e-12,
        "dual_upper_bound_pad": 1e-8,
        "QR_relative_atol": 1e-10,
        "QR_absolute_atol": 1e-12,
        "pool_floor_fraction": 1e-6,
        "row_sum_atol": 1e-12,
        "lower_bound_atol": 1e-12,
        "proposal_command_RMS_ratio": 0.8,
        "proposal_velocity_RMS_ratio": 1.0,
        "per_leg_command_and_velocity_nonincrease": True,
        "fixed_LTM_rows": True,
        "fixed_singleton_rows": True,
        "leg_order": list(LEGS),
        "startup_limits": {
            "q_RMS": 0.018,
            "q_max_abs": 0.04,
            "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
    }


def reviewed_inputs(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every frozen hash and the exact accepted two-source review."""
    require(freeze_sha == FREEZE_SHA, "Exact E145 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E145 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E145 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E145 design changed")
    for mapping_name in (
        "input_sha256",
        "dependency_sha256",
        "reused_source_sha256",
        "preservation_sha256",
    ):
        mapping = freeze.get(mapping_name)
        require(
            isinstance(mapping, dict) and mapping,
            f"Missing frozen mapping: {mapping_name}",
        )
        for relative, digest in mapping.items():
            checked_path(relative, digest)
    require(
        str(clarabel.__version__) == freeze["solver_version"],
        "Clarabel version changed",
    )

    require(sha256(review_path) == review_sha, "E145 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_global_conic_fit") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E145 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


@dataclass(frozen=True)
class AnchorMap:
    """Affine map from free weights to every variable supported pool entry."""

    columns: np.ndarray
    muscles: np.ndarray
    anchors: np.ndarray
    free_columns: np.ndarray
    free_muscles: np.ndarray
    constant: np.ndarray
    matrix: np.ndarray
    baseline_free: np.ndarray
    lower: np.ndarray
    anchor_rows: np.ndarray


def build_anchor_map(
    pool: Any,
    membership: Any,
    ltm: Any,
    *,
    floor_fraction: float = 1e-6,
) -> AnchorMap:
    """Use the last sorted member as anchor for each nonfixed muscle."""
    pool = np.asarray(pool, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)
    fixed = {int(value) for value in np.asarray(ltm)}
    variable_muscles = np.asarray(
        [
            m
            for m in range(pool.shape[0])
            if m not in fixed and np.count_nonzero(membership == m) > 1
        ],
        dtype=np.int64,
    )
    grouped = [np.flatnonzero(membership == m) for m in variable_muscles]
    columns = np.concatenate(grouped) if grouped else np.empty(0, dtype=np.int64)
    muscles = membership[columns]
    anchors = np.asarray([members[-1] for members in grouped], dtype=np.int64)
    free_columns = (
        np.concatenate([members[:-1] for members in grouped])
        if grouped
        else np.empty(0, dtype=np.int64)
    )
    free_muscles = membership[free_columns]
    constant = np.zeros(columns.size, dtype=np.float64)
    matrix = np.zeros((columns.size, free_columns.size), dtype=np.float64)
    anchor_rows = np.empty((variable_muscles.size, free_columns.size), dtype=np.float64)
    anchor_rows.fill(0)
    column_to_local = {int(column): index for index, column in enumerate(columns)}
    free_to_index = {int(column): index for index, column in enumerate(free_columns)}
    lower = np.empty(free_columns.size, dtype=np.float64)
    for row, (muscle, members) in enumerate(zip(variable_muscles, grouped)):
        anchor_local = column_to_local[int(members[-1])]
        constant[anchor_local] = 1.0
        floor = float(floor_fraction) / members.size
        for column in members[:-1]:
            free_index = free_to_index[int(column)]
            member_local = column_to_local[int(column)]
            matrix[member_local, free_index] = 1.0
            matrix[anchor_local, free_index] = -1.0
            anchor_rows[row, free_index] = 1.0
            lower[free_index] = floor
        require(
            np.array_equal(members, np.sort(members)),
            f"E145 unsorted members for muscle {muscle}",
        )
    baseline_free = pool[free_muscles, free_columns].copy()
    reconstructed = constant + matrix @ baseline_free
    require(
        np.max(np.abs(reconstructed - pool[muscles, columns])) <= 1e-15,
        "E145 baseline anchor reconstruction differs",
    )
    return AnchorMap(
        columns,
        muscles,
        anchors,
        free_columns,
        free_muscles,
        constant,
        matrix,
        baseline_free,
        lower,
        anchor_rows,
    )


def thin_qr_compress(matrix: Any, offset: Any) -> np.ndarray:
    """Return the untruncated thin-QR R for [matrix, offset]."""
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    require(
        matrix.ndim == 2 and matrix.shape[0] == offset.size, "E145 QR input mismatch"
    )
    augmented = np.column_stack((matrix, offset))
    return np.linalg.qr(augmented, mode="reduced")[1]


def compressed_affine(r: Any) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=np.float64)
    return r[:, :-1], r[:, -1]


@dataclass(frozen=True)
class AffineCriterion:
    name: str
    matrix: sparse.csr_matrix
    offset: np.ndarray
    limit: float
    direct_matrix: sparse.csr_matrix
    direct_offset: np.ndarray


@dataclass(frozen=True)
class ConicProblem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    b: np.ndarray
    cones: tuple[Any, ...]
    cone_kinds: tuple[str, ...]
    cone_dims: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    anchor_map: AnchorMap
    criteria: tuple[AffineCriterion, ...]
    startup_matrix: sparse.csr_matrix
    startup_offset: np.ndarray


def _embed_local(
    matrix: np.ndarray, indices: np.ndarray, width: int
) -> sparse.csr_matrix:
    local = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    selector = sparse.csr_matrix(
        (np.ones(indices.size), (np.arange(indices.size), indices)),
        shape=(indices.size, width),
    )
    return local @ selector


def make_soc_rows(
    matrix: sparse.spmatrix | np.ndarray,
    offset: Any,
    limit: float,
    n_free: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Encode ||M z + e|| <= limit * (1-t) as A y+s=b."""
    require(np.isfinite(limit) and limit > 0, "E145 SOC limit must be positive")
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    require(matrix.shape == (offset.size, n_free), "E145 SOC affine shape mismatch")
    head = sparse.csr_matrix(([1.0], ([0], [n_free])), shape=(1, n_free + 1))
    tail = sparse.hstack(
        (-matrix / limit, sparse.csr_matrix((offset.size, 1))), format="csr"
    )
    return sparse.vstack((head, tail), format="csr"), np.concatenate(
        ([1.0], offset / limit)
    )


def _make_nonnegative_rows(
    anchor: AnchorMap,
    startup_matrix: sparse.csr_matrix,
    startup_offset: np.ndarray,
    startup_max: float,
    margin_bounds: tuple[float, float],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = anchor.baseline_free.size
    eye = sparse.eye(n, format="csr")
    zero = sparse.csr_matrix((n, 1))
    rows = [sparse.hstack((-eye, zero)), sparse.hstack((eye, zero))]
    rhs = [-anchor.lower, np.ones(n)]
    # Each anchor is 1-sum(free), so sum(free) <= 1-anchor_floor.
    anchor_floor = np.asarray(
        [
            1e-6 / np.count_nonzero(anchor.muscles == muscle)
            for muscle in np.unique(anchor.muscles)
        ]
    )
    rows.append(
        sparse.hstack(
            (
                sparse.csr_matrix(anchor.anchor_rows),
                sparse.csr_matrix((anchor.anchor_rows.shape[0], 1)),
            )
        )
    )
    rhs.append(1.0 - anchor_floor)
    low, high = margin_bounds
    rows.extend(
        [
            sparse.csr_matrix(([-1.0], ([0], [n])), shape=(1, n + 1)),
            sparse.csr_matrix(([1.0], ([0], [n])), shape=(1, n + 1)),
        ]
    )
    rhs.extend([np.asarray([-low]), np.asarray([high])])
    t_column = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    rows.extend(
        [
            sparse.hstack((startup_matrix / startup_max, t_column)),
            sparse.hstack((-startup_matrix / startup_max, t_column)),
        ]
    )
    rhs.extend([1.0 - startup_offset / startup_max, 1.0 + startup_offset / startup_max])
    return sparse.vstack(rows, format="csr"), np.concatenate(rhs)


def _leg_affine(
    saved: dict[str, Any], data: dict[str, Any], anchor: AnchorMap, leg: str
) -> dict[str, Any]:
    layout = build_leg_layout(
        leg,
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        data["anatomy"],
        saved["moment"],
    )
    lookup = {int(column): i for i, column in enumerate(anchor.columns)}
    rows = np.asarray([lookup[int(column)] for column in layout.variable_columns])
    local_free = np.flatnonzero(np.isin(anchor.free_columns, layout.variable_columns))
    require(
        np.array_equal(anchor.columns[rows], layout.variable_columns),
        f"E145 {leg} map mismatch",
    )
    local_constant = anchor.constant[rows]
    local_matrix = anchor.matrix[np.ix_(rows, local_free)]
    full_x_design, full_x_offset = build_affine_design(
        saved["b"], saved["baseline"], saved["moment"], layout, np.arange(2000)
    )
    full_z_design = np.einsum("tjn,nk->tjk", full_x_design, local_matrix)
    full_z_offset = full_x_offset + np.einsum(
        "tjn,n->tj", full_x_design, local_constant
    )
    endpoint_rows = saved["target_indices"] - 1
    command_matrix = full_z_design[endpoint_rows].reshape(-1, local_free.size)
    command_offset = (
        full_z_offset[endpoint_rows] - saved["target"][:, layout.joint_columns]
    ).reshape(-1)
    velocity_design, velocity_offset = velocity_affine_design(
        full_z_design, full_z_offset, saved["pair_end"] - 2
    )
    velocity_matrix = velocity_design.reshape(-1, local_free.size)
    velocity_residual = (
        velocity_offset - saved["target_velocity"][:, layout.joint_columns]
    ).reshape(-1)
    startup_matrix = full_z_design[:99].reshape(-1, local_free.size)
    startup_offset = (
        full_z_offset[:99] - saved["startup_target"][:, layout.joint_columns]
    ).reshape(-1)
    return {
        "layout": layout,
        "free_indices": local_free,
        "command_matrix": command_matrix,
        "command_offset": command_offset,
        "velocity_matrix": velocity_matrix,
        "velocity_offset": velocity_residual,
        "startup_matrix": startup_matrix,
        "startup_offset": startup_offset,
    }


def build_conic_problem(
    saved: dict[str, Any], data: dict[str, Any], freeze: dict[str, Any]
) -> tuple[ConicProblem, dict[str, np.ndarray]]:
    """Build the one frozen global SOCP without invoking Clarabel."""
    anchor = build_anchor_map(
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        floor_fraction=freeze["pool_floor_fraction"],
    )
    n = anchor.baseline_free.size
    legs = [_leg_affine(saved, data, anchor, leg) for leg in LEGS]
    criteria: list[AffineCriterion] = []
    raw: dict[str, np.ndarray] = {}

    def add(
        name: str, blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], limit: float
    ) -> None:
        compressed_blocks: list[sparse.csr_matrix] = []
        compressed_offsets: list[np.ndarray] = []
        direct_blocks: list[sparse.csr_matrix] = []
        direct_offsets: list[np.ndarray] = []
        for local_matrix, offset, indices in blocks:
            r = thin_qr_compress(local_matrix, offset)
            r_matrix, r_offset = compressed_affine(r)
            compressed_blocks.append(_embed_local(r_matrix, indices, n))
            compressed_offsets.append(r_offset)
            direct_blocks.append(_embed_local(local_matrix, indices, n))
            direct_offsets.append(offset)
            raw[f"qr_{name}_{len(compressed_blocks) - 1}"] = r
        matrix = sparse.vstack(compressed_blocks, format="csr")
        offset = np.concatenate(compressed_offsets)
        direct_matrix = sparse.vstack(direct_blocks, format="csr")
        direct_offset = np.concatenate(direct_offsets)
        criterion = AffineCriterion(
            name, matrix, offset, float(limit), direct_matrix, direct_offset
        )
        criteria.append(criterion)
        raw[f"criterion_{name}_limit"] = np.asarray(limit)
        for suffix, z in (
            ("baseline", anchor.baseline_free),
            ("generated_equal", _equal_free_vector(anchor)),
        ):
            direct_norm = np.linalg.norm(direct_matrix @ z + direct_offset)
            compressed_norm = np.linalg.norm(matrix @ z + offset)
            raw[f"criterion_{name}_{suffix}_direct_norm"] = np.asarray(direct_norm)
            raw[f"criterion_{name}_{suffix}_compressed_norm"] = np.asarray(
                compressed_norm
            )
            tolerance = (
                freeze["QR_absolute_atol"] + freeze["QR_relative_atol"] * direct_norm
            )
            require(
                abs(direct_norm - compressed_norm) <= tolerance,
                f"E145 {name} QR norm mismatch",
            )

    baseline = evaluate_composite(saved, saved["pool"])["metrics"]["baseline"]
    command_blocks = [
        (leg["command_matrix"], leg["command_offset"], leg["free_indices"])
        for leg in legs
    ]
    velocity_blocks = [
        (leg["velocity_matrix"], leg["velocity_offset"], leg["free_indices"])
        for leg in legs
    ]
    startup_blocks = [
        (leg["startup_matrix"], leg["startup_offset"], leg["free_indices"])
        for leg in legs
    ]
    add(
        "overall_command",
        command_blocks,
        np.sqrt(1827 * 42)
        * freeze["proposal_command_RMS_ratio"]
        * baseline["command_RMS"],
    )
    add(
        "overall_velocity",
        velocity_blocks,
        np.sqrt(1826 * 42)
        * freeze["proposal_velocity_RMS_ratio"]
        * baseline["velocity_RMS"],
    )
    for i, leg_name in enumerate(LEGS):
        add(
            f"{leg_name}_command",
            [command_blocks[i]],
            np.sqrt(1827 * 7) * baseline["by_leg"][leg_name]["command_RMS"],
        )
        add(
            f"{leg_name}_velocity",
            [velocity_blocks[i]],
            np.sqrt(1826 * 7) * baseline["by_leg"][leg_name]["velocity_RMS"],
        )
    add(
        "overall_startup",
        startup_blocks,
        np.sqrt(99 * 42) * freeze["startup_limits"]["q_RMS"],
    )

    startup_matrix = sparse.vstack(
        [_embed_local(m, idx, n) for m, _e, idx in startup_blocks], format="csr"
    )
    startup_offset = np.concatenate([e for _m, e, _idx in startup_blocks])
    nonnegative_A, nonnegative_b = _make_nonnegative_rows(
        anchor,
        startup_matrix,
        startup_offset,
        freeze["startup_limits"]["q_max_abs"],
        tuple(freeze["margin_bounds"]),
    )
    A_blocks: list[sparse.csr_matrix] = [nonnegative_A]
    b_blocks: list[np.ndarray] = [nonnegative_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(nonnegative_A.shape[0])]
    cone_kinds = ["NonnegativeConeT"]
    cone_dims = [nonnegative_A.shape[0]]
    for criterion in criteria:
        rows, rhs = make_soc_rows(
            criterion.matrix, criterion.offset, criterion.limit, n
        )
        A_blocks.append(rows)
        b_blocks.append(rhs)
        cones.append(clarabel.SecondOrderConeT(rows.shape[0]))
        cone_kinds.append("SecondOrderConeT")
        cone_dims.append(rows.shape[0])
    A = sparse.vstack(A_blocks, format="csc")
    b = np.concatenate(b_blocks)
    q = np.zeros(n + 1, dtype=np.float64)
    q[-1] = -1.0
    P = sparse.csc_matrix((n + 1, n + 1), dtype=np.float64)
    lower = np.concatenate((np.zeros(n), np.asarray(freeze["margin_bounds"][:1])))
    upper = np.concatenate((np.ones(n), np.asarray(freeze["margin_bounds"][1:])))
    problem = ConicProblem(
        P,
        q,
        A,
        b,
        tuple(cones),
        tuple(cone_kinds),
        np.asarray(cone_dims),
        lower,
        upper,
        anchor,
        tuple(criteria),
        startup_matrix,
        startup_offset,
    )
    raw.update(_problem_arrays(problem))
    _verify_baseline_feasible_range(problem, freeze)
    return problem, raw


def _equal_free_vector(anchor: AnchorMap) -> np.ndarray:
    result = np.empty(anchor.free_columns.size, dtype=np.float64)
    for muscle in np.unique(anchor.free_muscles):
        count = np.count_nonzero(anchor.muscles == muscle)
        result[anchor.free_muscles == muscle] = 1.0 / count
    return result


def _problem_arrays(problem: ConicProblem) -> dict[str, np.ndarray]:
    return {
        "P_data": problem.P.data,
        "P_indices": problem.P.indices,
        "P_indptr": problem.P.indptr,
        "P_shape": np.asarray(problem.P.shape),
        "q": problem.q,
        "A_data": problem.A.data,
        "A_indices": problem.A.indices,
        "A_indptr": problem.A.indptr,
        "A_shape": np.asarray(problem.A.shape),
        "b": problem.b,
        "cone_kinds": np.asarray(problem.cone_kinds),
        "cone_dims": problem.cone_dims,
        "variable_columns": problem.anchor_map.columns,
        "variable_column_muscles": problem.anchor_map.muscles,
        "anchor_columns": problem.anchor_map.anchors,
        "free_columns": problem.anchor_map.free_columns,
        "free_column_muscles": problem.anchor_map.free_muscles,
        "anchor_constant": problem.anchor_map.constant,
        "anchor_matrix": problem.anchor_map.matrix,
        "baseline_free": problem.anchor_map.baseline_free,
        "free_lower_bounds": problem.anchor_map.lower,
        "anchor_sum_rows": problem.anchor_map.anchor_rows,
        "box_lower": problem.lower,
        "box_upper": problem.upper,
        "criterion_names": np.asarray([item.name for item in problem.criteria]),
        "criterion_limits": np.asarray([item.limit for item in problem.criteria]),
    }


def _verify_baseline_feasible_range(
    problem: ConicProblem, freeze: dict[str, Any]
) -> None:
    z0 = problem.anchor_map.baseline_free
    normalized = {
        item.name: np.linalg.norm(item.direct_matrix @ z0 + item.direct_offset)
        / item.limit
        for item in problem.criteria
    }
    require(
        abs(normalized["overall_command"] - 1.25) <= 1e-10,
        "E145 baseline command ratio changed",
    )
    startup_max = np.max(np.abs(problem.startup_matrix @ z0 + problem.startup_offset))
    require(
        max([*normalized.values(), startup_max / freeze["startup_limits"]["q_max_abs"]])
        <= 1.25 + 1e-10,
        "E145 t=-0.25 baseline feasibility premise changed",
    )


def solver_settings(freeze: dict[str, Any]) -> Any:
    settings = clarabel.DefaultSettings()
    for key, value in freeze["solver_settings"].items():
        setattr(settings, key, value)
    return settings


def settings_record(settings: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in dir(settings):
        if key.startswith("_") or key == "default":
            continue
        value = getattr(settings, key)
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not np.isfinite(value):
                result[key] = str(value)
            else:
                result[key] = value
    return result


def _info_record(info: Any) -> dict[str, Any]:
    keys = (
        "mu",
        "sigma",
        "step_length",
        "iterations",
        "cost_primal",
        "cost_dual",
        "res_primal",
        "res_dual",
        "res_primal_inf",
        "res_dual_inf",
        "gap_abs",
        "gap_rel",
        "ktratio",
        "solve_time",
        "status",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = getattr(info, key, None)
        if key == "status":
            result[key] = str(value)
        elif value is None:
            result[key] = None
        else:
            number = float(value)
            result[key] = number if np.isfinite(number) else None
    return result


def _history_arrays(history: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    history_keys = (
        "mu",
        "sigma",
        "step_length",
        "iterations",
        "cost_primal",
        "cost_dual",
        "res_primal",
        "res_dual",
        "res_primal_inf",
        "res_dual_inf",
        "gap_abs",
        "gap_rel",
        "ktratio",
        "solve_time",
    )
    result: dict[str, np.ndarray] = {
        "history_json": np.asarray(json.dumps(history, allow_nan=False)),
        "history_status": np.asarray([row["status"] for row in history]),
    }
    for key in history_keys:
        result[f"history_{key}"] = np.asarray(
            [np.nan if row.get(key) is None else row[key] for row in history],
            dtype=np.float64,
        )
    return result


def _bare_solution_arrays(solution: Any) -> dict[str, np.ndarray]:
    """Copy solver-owned return values before any derived calculation."""
    return {
        "solver_x": np.asarray(solution.x, dtype=np.float64).copy(),
        "solver_s": np.asarray(solution.s, dtype=np.float64).copy(),
        "solver_z_raw": np.asarray(solution.z, dtype=np.float64).copy(),
        "solver_status": np.asarray(str(solution.status)),
        "solver_iterations": np.asarray(solution.iterations),
        "solver_solve_time_s": np.asarray(solution.solve_time),
        "solver_primal_objective": np.asarray(solution.obj_val),
        "solver_dual_objective": np.asarray(solution.obj_val_dual),
        "solver_reported_primal_residual": np.asarray(solution.r_prim),
        "solver_reported_dual_residual": np.asarray(solution.r_dual),
    }


def _publish_solver_state(
    partial: dict[str, Any],
    progress: dict[str, Any],
    state: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    progress["solver_state"] = dict(state)
    progress["fit_calls"] = int(state["solve_entries"])
    partial["solver_state_json"] = np.asarray(
        json.dumps(state, sort_keys=True, allow_nan=False)
    )
    partial.update(_history_arrays(history))


def solve_conic_problem(
    problem: ConicProblem,
    freeze: dict[str, Any],
    *,
    solver_factory: Callable[..., Any] = clarabel.DefaultSolver,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Invoke exactly one Clarabel solver and retain callback iteration data."""
    if partial is None:
        partial = {}
    if progress is None:
        progress = {}
    if history is None:
        history = []
    if state is None:
        state = {}
    state.update(
        {
            "attempted": True,
            "constructed": False,
            "callback_registration_attempted": False,
            "callback_registered": False,
            "solve_entered": False,
            "solve_returned": False,
            "solve_entries": 0,
            "phase": "settings",
        }
    )
    settings = solver_settings(freeze)
    initial_settings = settings_record(settings)
    partial["solver_settings_json"] = np.asarray(
        json.dumps(initial_settings, sort_keys=True, allow_nan=False)
    )
    _publish_solver_state(partial, progress, state, history)
    state["phase"] = "constructor_entered"
    _publish_solver_state(partial, progress, state, history)
    try:
        solver = solver_factory(
            problem.P,
            problem.q,
            problem.A,
            problem.b,
            list(problem.cones),
            settings,
        )
    except BaseException as error:
        state.update(
            {
                "phase": "constructor_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _publish_solver_state(partial, progress, state, history)
        raise
    state["constructed"] = True
    state["phase"] = "constructed"
    _publish_solver_state(partial, progress, state, history)

    def callback(info: Any) -> bool:
        history.append(_info_record(info))
        _publish_solver_state(partial, progress, state, history)
        return False

    state["callback_registration_attempted"] = True
    state["phase"] = "callback_registration_entered"
    _publish_solver_state(partial, progress, state, history)
    try:
        solver.set_termination_callback(callback)
    except BaseException as error:
        state.update(
            {
                "phase": "callback_registration_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _publish_solver_state(partial, progress, state, history)
        raise
    state["callback_registered"] = True
    state["solve_entered"] = True
    state["solve_entries"] = 1
    state["phase"] = "solve_entered"
    _publish_solver_state(partial, progress, state, history)
    try:
        solution = solver.solve()
    except BaseException as error:
        state.update(
            {
                "phase": "solve_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _publish_solver_state(partial, progress, state, history)
        raise
    # Bare returned values become caller-owned before settings retrieval or any
    # numerical diagnostic can fail.
    partial.update(_bare_solution_arrays(solution))
    state["solve_returned"] = True
    state["phase"] = "solve_returned"
    _publish_solver_state(partial, progress, state, history)
    try:
        final_settings = settings_record(solver.get_settings())
    except BaseException as error:
        state.update(
            {
                "phase": "settings_retrieval_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _publish_solver_state(partial, progress, state, history)
        raise
    partial["solver_settings_json"] = np.asarray(
        json.dumps(final_settings, sort_keys=True, allow_nan=False)
    )
    state["phase"] = "complete"
    _publish_solver_state(partial, progress, state, history)
    return solution, history, final_settings


def project_soc(vector: Any, interior_pad: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).copy()
    head, tail = float(value[0]), value[1:]
    tail_norm = float(np.linalg.norm(tail))
    if tail_norm <= head:
        projected = value
    elif tail_norm <= -head:
        projected = np.zeros_like(value)
    else:
        projected = np.concatenate(
            ([(tail_norm + head) / 2], ((tail_norm + head) / (2 * tail_norm)) * tail)
        )
    projected[0] = max(projected[0], np.linalg.norm(projected[1:]))
    projected[0] += interior_pad * (1 + np.linalg.norm(projected[1:]))
    return projected


def project_conic_dual(
    dual: Any, cone_kinds: tuple[str, ...], cone_dims: Any, *, interior_pad: float
) -> tuple[np.ndarray, np.ndarray]:
    dual = np.asarray(dual, dtype=np.float64)
    dims = np.asarray(cone_dims, dtype=np.int64)
    require(dual.size == int(dims.sum()), "E145 dual/cone dimension mismatch")
    projected = np.empty_like(dual)
    margins: list[float] = []
    start = 0
    for kind, dim in zip(cone_kinds, dims):
        section = dual[start : start + dim]
        if kind == "NonnegativeConeT":
            corrected = np.maximum(section, 0)
            margins.extend(corrected.tolist())
        elif kind == "SecondOrderConeT":
            corrected = project_soc(section, interior_pad)
            margins.append(float(corrected[0] - np.linalg.norm(corrected[1:])))
        else:
            raise ValueError(f"E145 unsupported dual cone: {kind}")
        projected[start : start + dim] = corrected
        start += int(dim)
    return projected, np.asarray(margins)


def dual_margin_upper_bound(
    q: Any,
    A: sparse.spmatrix,
    b: Any,
    dual: Any,
    lower: Any,
    upper: Any,
    *,
    pad_factor: float,
) -> dict[str, Any]:
    """Compute the frozen box-corrected lower bound on min q'x."""
    q = np.asarray(q, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dual = np.asarray(dual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    stationarity = q + A.T @ dual
    box_terms = np.minimum(stationarity * lower, stationarity * upper)
    dual_dot = float(b @ dual)
    box_sum = float(np.sum(box_terms))
    objective_lower = -dual_dot + box_sum
    numerical_pad = float(pad_factor * (1 + abs(dual_dot) + np.sum(np.abs(box_terms))))
    return {
        "stationarity": np.asarray(stationarity),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "dual_b_dot": dual_dot,
        "box_terms": box_terms,
        "box_term_sum": box_sum,
        "objective_lower_bound": objective_lower,
        "margin_upper_bound_unpadded": -objective_lower,
        "numerical_pad": numerical_pad,
        "margin_upper_bound": -objective_lower + numerical_pad,
    }


def candidate_pool_from_free(
    baseline_pool: Any, anchor: AnchorMap, free: Any
) -> np.ndarray:
    """Construct exactly a+Bz, with no clipping or normalization."""
    free = np.asarray(free, dtype=np.float64)
    require(
        free.shape == anchor.baseline_free.shape,
        "E145 returned free vector shape mismatch",
    )
    candidate = np.asarray(baseline_pool, dtype=np.float64).copy()
    candidate[anchor.muscles, anchor.columns] = anchor.constant + anchor.matrix @ free
    return candidate


def assess_physical_candidate(
    saved: dict[str, Any],
    candidate_pool: np.ndarray,
    composite: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Apply pool floors and then the unchanged E144 physical gates."""
    validate_pool_contract(
        candidate_pool,
        saved["membership"],
        saved["ltm"],
        row_sum_atol=freeze["row_sum_atol"],
    )
    floor_margins = []
    for muscle in range(96):
        columns = np.flatnonzero(saved["membership"] == muscle)
        floor_margins.extend(
            candidate_pool[muscle, columns]
            - freeze["pool_floor_fraction"] / columns.size
        )
    floor_min = float(np.min(floor_margins))
    strictly_positive = bool(
        np.all(candidate_pool[saved["membership"], np.arange(391)] > 0)
    )
    base = assess_composite(saved, candidate_pool, composite, freeze)
    gates = dict(base.get("gates", {}))
    gates["strictly_positive_same_support"] = strictly_positive
    gates["pool_floor_with_tolerance"] = floor_min >= -freeze["lower_bound_atol"]
    proposal = bool(gates and all(gates.values()))
    return {
        **base,
        "gates": gates,
        "pool_floor_min_margin": floor_min,
        "conditional_static_pool_proposal": proposal,
        "solver_status_used_as_proposal_gate": False,
    }


def solution_arrays(
    problem: ConicProblem,
    solution: Any,
    history: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    x = np.asarray(solution.x, dtype=np.float64)
    slack = np.asarray(solution.s, dtype=np.float64)
    dual = np.asarray(solution.z, dtype=np.float64)
    projected, cone_margins = project_conic_dual(
        dual,
        problem.cone_kinds,
        problem.cone_dims,
        interior_pad=freeze["dual_interior_pad"],
    )
    raw_bound = dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        dual,
        problem.lower,
        problem.upper,
        pad_factor=freeze["dual_upper_bound_pad"],
    )
    projected_bound = dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        projected,
        problem.lower,
        problem.upper,
        pad_factor=freeze["dual_upper_bound_pad"],
    )
    result: dict[str, Any] = {
        **_bare_solution_arrays(solution),
        **_history_arrays(history),
        "solver_z_projected": projected,
        "primal_residual": problem.A @ x + slack - problem.b,
        "stationarity_raw": problem.q + problem.A.T @ dual,
        "stationarity_projected": problem.q + problem.A.T @ projected,
        "projected_dual_cone_margins": cone_margins,
    }
    for prefix, values in (
        ("dual_raw", raw_bound),
        ("dual_projected", projected_bound),
    ):
        for key, value in values.items():
            result[f"{prefix}_{key}"] = np.asarray(value)
    return result


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e145.global-conic-result.v1",
        "experiment": "M3-E145-global-conic-pool",
        "status": "failed",
        "validity": "invalid_incomplete",
        "conditional_static_pool_proposal": False,
        "candidate_adopted": False,
        "F1_F8": "pending",
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
        "failure": progress.get(
            "primary_failure", {"type": type(primary).__name__, "message": str(primary)}
        ),
    }


def calculate(
    data: dict[str, Any],
    freeze: dict[str, Any],
    progress: dict[str, Any],
    partial: dict[str, Any],
    prepared: tuple[ConicProblem, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ConicProblem]:
    """Build, solve once, and return raw evidence before any proposal verdict."""
    partial.update(
        {
            "unvalidated_b": np.asarray(data["e143"].get("filtered_motor_rate", [])),
            "unvalidated_pool": np.asarray(data["e143"].get("pool", [])),
            "unvalidated_membership": np.asarray(data["e143"].get("membership", [])),
            "unvalidated_moment": np.asarray(data["e143"].get("moment", [])),
            "unvalidated_rest": np.asarray(data["e143"].get("rest", [])),
            "unvalidated_ltm": np.asarray(data["e143"].get("ltm", [])),
        }
    )
    saved = validate_saved_inputs(data, freeze)
    partial.update(
        {key: value for key, value in saved.items() if isinstance(value, np.ndarray)}
    )
    problem, pre_solve_raw = prepared or build_conic_problem(saved, data, freeze)
    partial.update(pre_solve_raw)
    progress["problem_built"] = True
    raw: dict[str, Any] = {
        **pre_solve_raw,
        "filtered_motor_rate": saved["b"],
        "baseline_pool": saved["pool"],
        "membership": saved["membership"],
        "moment": saved["moment"],
        "rest": saved["rest"],
        "ltm": saved["ltm"],
        "target_indices": saved["target_indices"],
        "target_rad": saved["target"],
        "velocity_pair_start_indices": saved["pair_start"],
        "velocity_pair_end_indices": saved["pair_end"],
        "target_velocity_rad_s": saved["target_velocity"],
        "startup_target_rad": saved["startup_target"],
        "startup_target_tendon": saved["startup_tendon_target"],
    }
    solver_state: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    fit_started = time.perf_counter()
    try:
        solution, history, final_settings = solve_conic_problem(
            problem,
            freeze,
            partial=partial,
            progress=progress,
            state=solver_state,
            history=history,
        )
        # Retain this publication in calculate as an ownership boundary even
        # when a generated test double replaces solve_conic_problem.
        partial.update(_bare_solution_arrays(solution))
        partial.update(_history_arrays(history))
        partial["solver_settings_json"] = np.asarray(
            json.dumps(final_settings, sort_keys=True, allow_nan=False)
        )
    finally:
        progress["timings_s"]["fit"] = time.perf_counter() - fit_started
        progress["fit_wall_s"] = progress["timings_s"]["fit"]
    require(
        progress["timings_s"]["fit"] <= freeze["total_fit_limit_s"],
        "E145 fit exceeded 30 seconds",
    )
    result_arrays = solution_arrays(problem, solution, history, freeze)
    raw.update(result_arrays)
    raw["solver_settings_json"] = np.asarray(
        json.dumps(final_settings, sort_keys=True, allow_nan=False)
    )
    partial.update(result_arrays)
    candidate = candidate_pool_from_free(
        saved["pool"], problem.anchor_map, np.asarray(solution.x)[:-1]
    )
    raw["candidate_pool"] = candidate
    raw["returned_margin_t"] = np.asarray(solution.x[-1])
    composite = evaluate_composite(saved, candidate)
    raw.update(composite["arrays"])
    raw.update(composite_metrics_to_raw(composite["metrics"]))
    facts = {
        "solver": {
            "status": str(solution.status),
            "iterations": int(solution.iterations),
            "solve_time_s": float(solution.solve_time),
            "primal_objective": float(solution.obj_val),
            "dual_objective": float(solution.obj_val_dual),
            "returned_margin_t": float(solution.x[-1]),
            "calls": 1,
        },
        "metrics": composite["metrics"],
    }
    return (
        raw,
        facts,
        {"saved": saved, "candidate_pool": candidate, "composite": composite},
        problem,
    )


def run(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "schema": "flylab.m3.e145.global-conic-execution.v1",
        "status": "created",
        "original_neural_steps": 0,
        "CUDA_calls": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 0,
        "LP_calls": 0,
        "fit_calls": 0,
        "original_processes": 1,
        "automatic_retries": 0,
        "cpu_threads_requested": 1,
        "gpu_count": 0,
        "candidate_adopted": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
        "timings_s": {},
        "failed_raw_operations": [],
        "failed_persistence_operations": [],
        "failed_resource_checks": [],
    }
    partial: dict[str, Any] = {}
    freeze: dict[str, Any] | None = None
    primary: BaseException | None = None
    summary: dict[str, Any] = {}
    raw_saved = False
    problem_saved = False

    def capture_failure(error: BaseException, operation: str) -> None:
        nonlocal primary
        if primary is None:
            primary = error
            progress["primary_failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        else:
            add_secondary(primary, operation, error)

    def check_resources(stage: str) -> None:
        elapsed = time.perf_counter() - started
        progress[f"wall_s_{stage}"] = elapsed
        progress["wall_s"] = elapsed
        if freeze is None:
            return
        try:
            size = output_bytes(output)
        except BaseException as error:
            size = None
            progress["failed_resource_checks"].append(
                f"{stage}:output_bytes_unavailable"
            )
            capture_failure(error, f"{stage} output byte accounting")
        progress[f"bytes_{stage}"] = size
        progress["raw_bytes"] = size
        checks = (
            (
                elapsed > freeze["internal_wall_limit_s"],
                "internal_wall_limit_s",
                TimeoutError(f"E145 exceeded internal wall during {stage}"),
            ),
            (
                size is not None and size > freeze["maximum_raw_bytes"],
                "maximum_raw_bytes",
                RuntimeError(f"E145 exceeded raw byte limit during {stage}"),
            ),
        )
        for failed, label, error in checks:
            marker = f"{stage}:{label}"
            if failed and marker not in progress["failed_resource_checks"]:
                progress["failed_resource_checks"].append(marker)
                capture_failure(error, f"{stage} resource check")

    try:
        stage = time.perf_counter()
        freeze, review = reviewed_inputs(
            freeze_path, freeze_sha, review_path, review_sha
        )
        progress["timings_s"]["source_gate"] = time.perf_counter() - stage
        progress["source_review_sha256"] = review_sha
        progress["source_sha256"] = review["source_sha256"]
        save_json_new(output / "freeze-snapshot.json", freeze)
        save_json_new(output / "source-review-snapshot.json", review)
        free = shutil.disk_usage(output.parent).free
        progress["starting_free_disk_bytes"] = free
        require(
            free >= freeze["minimum_free_disk_bytes"],
            "E145 insufficient starting free disk",
        )
        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(
                all(
                    int(item.get("num_threads", 1)) <= 1
                    for item in progress["thread_pools"]
                ),
                "E145 CPU thread pool exceeds one thread",
            )
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            # Build and persist the exact conic problem before the single solve.
            saved = validate_saved_inputs(data, freeze)
            pre_problem, pre_raw = build_conic_problem(saved, data, freeze)
            try:
                save_npz_new(output / "conic-problem.npz", pre_raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save conic-problem.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            problem_saved = True
            progress["problem_persisted_before_solve"] = True
            raw, facts, outcome, _problem = calculate(
                data, freeze, progress, partial, prepared=(pre_problem, pre_raw)
            )
            try:
                save_npz_new(output / "conic-pool-raw.npz", raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save conic-pool-raw.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            raw_saved = True
            assessment = assess_physical_candidate(
                outcome["saved"],
                outcome["candidate_pool"],
                outcome["composite"],
                freeze,
            )
        proposal = bool(assessment["conditional_static_pool_proposal"])
        summary = {
            "schema": "flylab.m3.e145.global-conic-result.v1",
            "experiment": "M3-E145-global-conic-pool",
            "status": "pass",
            "validity": "conditional_saved_E142_single_global_SOCP",
            "hypothesis_verdict": "conditional_static_pool_proposal"
            if proposal
            else "global_conic_no_conditional_proposal",
            "analysis": facts,
            "assessment": assessment,
            "conditional_static_pool_proposal": proposal,
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
            "interpretation": {
                "scope": "one saved-data static-pool SOCP; no neural or body execution",
                "numerical_upper_bound": "floating-point diagnostic, not an exact impossibility proof",
                "biology": "failure does not establish impossibility of BANC walking",
                "gait_claim": "natural gait remains unverified",
            },
            "required_persistence_included_in_resource_check": True,
        }
    except BaseException as error:
        primary = error
        progress["status"] = "failed"
        progress["primary_failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if partial and not raw_saved:
            try:
                save_npz_new(output / "partial-raw.npz", partial)
                progress["partial_raw_saved"] = True
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save partial-raw.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                capture_failure(error, "partial raw")
        progress["problem_saved"] = problem_saved
        check_resources("before_required_receipts")
        if primary is not None:
            summary = failure_summary(primary, progress)
        try:
            save_json_new(output / "execution-raw.json", progress)
        except BaseException as error:
            progress["failed_persistence_operations"].append(
                {
                    "operation": "save execution-raw.json",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            capture_failure(error, "save execution-raw.json")
        check_resources("before_summary")
        if primary is not None:
            summary = failure_summary(primary, progress)
        summary_was_success = summary.get("status") == "pass"
        try:
            save_json_new(output / "summary.json", summary)
        except BaseException as error:
            progress["failed_persistence_operations"].append(
                {
                    "operation": "save summary.json",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            capture_failure(error, "save summary.json")
        check_resources("after_summary")
        if primary is None:
            progress["status"] = "pass"
        elif summary_was_success:
            try:
                if (output / "summary.json").exists():
                    (output / "summary.json").rename(
                        output / "invalidated-success-summary.json"
                    )
                summary = failure_summary(primary, progress)
                save_json_new(output / "summary.json", summary)
            except BaseException as error:
                capture_failure(error, "invalidate authoritative success summary")
        if primary is not None:
            progress["status"] = "failed"
            try:
                save_json_new(
                    output / "finalization-errors.json",
                    {
                        "status": "failed",
                        "primary_failure": progress.get("primary_failure"),
                        "failed_raw_operations": progress["failed_raw_operations"],
                        "failed_persistence_operations": progress[
                            "failed_persistence_operations"
                        ],
                        "failed_resource_checks": progress["failed_resource_checks"],
                        "wall_s_at_finalization_receipt": time.perf_counter() - started,
                        "bytes_last_observed": progress.get("raw_bytes"),
                    },
                )
            except BaseException as error:
                add_secondary(primary, "save finalization-errors.json", error)
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    summary["final_resources"] = {
        "wall_s_after_required_persistence": progress["wall_s_after_summary"],
        "bytes_after_required_persistence": progress["bytes_after_summary"],
        "all_required_persistence_and_resource_checks_passed": True,
    }
    return summary


def main() -> None:
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
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
