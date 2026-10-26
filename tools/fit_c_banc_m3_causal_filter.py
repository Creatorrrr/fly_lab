"""E146 bounded saved-signal causal muscle-filter SOCP.

The program reads frozen arrays only.  It builds three causal activation
branches for every recorded motor neuron and performs one CPU Clarabel solve.
No neural, CUDA, physics, LP, or runtime-controller work is performed here.
"""

# ruff: noqa: BLE001, I001  # thread limits must precede numerical imports

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

import fit_c_banc_m3_conic_pool as conic
from fit_c_banc_m3_static_pool import (
    COMMAND_GAIN,
    DT,
    LEGS,
    build_leg_layout,
    composite_metrics_to_raw,
    finite_array,
    load_inputs,
    validate_saved_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "e7cb061237321ccf6e881ecfa1afc849849affa45cc489a489a591141b304db7"
DESIGN_SHA = "277540f46b9a77982b88c598654209d67e8a21ac4c590cd58fc40e2df915ab6c"
SCHEMA = "flylab.m3.e146.causal-filter.v1"
OWNED = (
    "tools/fit_c_banc_m3_causal_filter.py",
    "tests/test_c_banc_causal_filter.py",
)
TAU_S = np.asarray([0.005, 0.01, 0.04], dtype=np.float64)
MIDDLE_BRANCH = 1
DEFAULT_FREEZE = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e146-causal-filter-freeze.json"
)

# Public names intentionally point at the frozen E145 numerical implementation.
# Tests inject factories through the wrapper argument; this module never changes
# the imported module's globals.
make_soc_rows = conic.make_soc_rows
solve_conic_problem = conic.solve_conic_problem
solution_arrays = conic.solution_arrays
thin_qr_compress = conic.thin_qr_compress
compressed_affine = conic.compressed_affine


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
        f"Secondary E146 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E146_CAUSAL_FILTER_DESIGN_20260921.md",
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
        "prior_conservative_process_s": 2188,
        "prior_conservative_actual_s": 2187.4898345999536,
        "prospective_cumulative_limit_s": 2258,
        "original_program_cumulative_limit_s": 2271,
        "maximum_raw_bytes": 134217728,
        "minimum_free_disk_bytes": 1073741824,
        "original_neural_steps": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 2000,
        "recorded_EMA_scalar_updates": 2346000,
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
        "tau_s": [0.005, 0.01, 0.04],
        "control_dt_s": 0.005,
        "branch_order": "motor-major:3*i+k",
        "branch_guard_fraction": 1e-6,
        "original_MN_floor_fraction": 1e-6,
        "baseline_feasibility_margin": -0.251,
        "anchor_branch": 1,
        "middle_branch": 1,
    }


def reviewed_inputs(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete freeze and accepted two-source review first."""
    require(freeze_sha == FREEZE_SHA, "Exact E146 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E146 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E146 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E146 design changed")
    for mapping_name in (
        "input_sha256",
        "dependency_sha256",
        "reused_source_sha256",
        "preservation_sha256",
    ):
        mapping = freeze.get(mapping_name)
        require(isinstance(mapping, dict) and mapping, f"Missing {mapping_name}")
        for relative, digest in mapping.items():
            checked_path(relative, digest)
    require(
        str(clarabel.__version__) == freeze["solver_version"],
        "Clarabel version changed",
    )
    require(sha256(review_path) == review_sha, "E146 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_causal_filter_fit") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E146 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


def causal_filter_bank(
    recorded_rate: Any,
    *,
    dt: float = DT,
    tau_s: Any = TAU_S,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply the exact zero-state, future-free filter to every motor column."""
    rate = np.asarray(recorded_rate, dtype=np.float64)
    tau = np.asarray(tau_s, dtype=np.float64)
    require(rate.ndim == 2, "E146 recorded rate must be two-dimensional")
    require(np.isfinite(rate).all(), "E146 recorded rate contains nonfinite values")
    require(
        tau.ndim == 1 and tau.size == 3 and np.all(tau > 0),
        "E146 tau contract changed",
    )
    alpha = -np.expm1(-float(dt) / tau)
    state = np.zeros((rate.shape[1], tau.size), dtype=np.float64)
    bank = np.full((rate.shape[0], rate.shape[1], tau.size), np.nan, dtype=np.float64)
    if partial is not None:
        partial["recorded_motor_rate"] = rate
        partial["filter_bank"] = bank
        partial["filter_tau_s"] = tau.copy()
    if progress is not None:
        progress["recorded_EMA_updates"] = 0
        progress["recorded_EMA_scalar_updates"] = 0
    for index, row in enumerate(rate):
        state = state + alpha * (row[:, None] - state)
        bank[index] = state
        if progress is not None:
            progress["recorded_EMA_updates"] = index + 1
            progress["recorded_EMA_scalar_updates"] = (
                (index + 1) * rate.shape[1] * tau.size
            )
    return bank


def validate_causal_saved_inputs(
    data: dict[str, Any],
    freeze: dict[str, Any],
    *,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate E144 inputs, then the newly derived causal branch bank."""
    saved = validate_saved_inputs(data, freeze)
    rate = finite_array(
        data["e143"]["recorded_motor_rate"], (2000, 391), "recorded motor rate"
    )
    if partial is not None:
        partial["recorded_motor_rate"] = rate
    bank = causal_filter_bank(
        rate,
        dt=freeze["control_dt_s"],
        tau_s=freeze["tau_s"],
        partial=partial,
        progress=progress,
    )
    require(bank.shape == (2000, 391, 3), "E146 filter bank shape mismatch")
    require(
        np.array_equal(bank[:, :, MIDDLE_BRANCH], saved["b"]),
        "E146 10ms branch differs from E143 saved filtered motor rate",
    )
    saved.update(recorded_rate=rate, bank=bank)
    return saved


@dataclass(frozen=True)
class CausalAnchorMap:
    """Affine supported-entry map with a last-MN middle-branch anchor."""

    supported_flat: np.ndarray
    supported_motors: np.ndarray
    supported_branches: np.ndarray
    supported_muscles: np.ndarray
    anchors_flat: np.ndarray
    free_flat: np.ndarray
    free_motors: np.ndarray
    free_branches: np.ndarray
    free_muscles: np.ndarray
    constant: np.ndarray
    matrix: np.ndarray
    canonical_free: np.ndarray
    lifted_free: np.ndarray
    lower: np.ndarray
    anchor_rows: np.ndarray
    variable_muscles: np.ndarray
    fixed_muscles: np.ndarray


def canonical_weights(pool: Any) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.float64)
    weights = np.zeros((*pool.shape, 3), dtype=np.float64)
    weights[:, :, MIDDLE_BRANCH] = pool
    return weights


def build_causal_anchor_map(
    pool: Any,
    membership: Any,
    ltm: Any,
    *,
    guard_fraction: float = 1e-6,
) -> CausalAnchorMap:
    """Build motor-major free coordinates for every nonfixed muscle."""
    pool = np.asarray(pool, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)
    ltm_set = {int(value) for value in np.asarray(ltm)}
    fixed = np.asarray(
        [
            muscle
            for muscle in range(pool.shape[0])
            if muscle in ltm_set or np.count_nonzero(membership == muscle) == 1
        ],
        dtype=np.int64,
    )
    variable = np.asarray(
        [muscle for muscle in range(pool.shape[0]) if muscle not in set(fixed)],
        dtype=np.int64,
    )
    supported_flat: list[int] = []
    supported_muscles: list[int] = []
    anchors: list[int] = []
    for muscle in variable:
        members = np.flatnonzero(membership == muscle)
        require(np.array_equal(members, np.sort(members)), "E146 unsorted members")
        for motor in members:
            for branch in range(3):
                supported_flat.append(3 * int(motor) + branch)
                supported_muscles.append(int(muscle))
        anchors.append(3 * int(members[-1]) + MIDDLE_BRANCH)
    supported = np.asarray(supported_flat, dtype=np.int64)
    muscles = np.asarray(supported_muscles, dtype=np.int64)
    anchors_array = np.asarray(anchors, dtype=np.int64)
    anchor_set = set(anchors)
    free = np.asarray([value for value in supported if value not in anchor_set])
    free_motors, free_branches = np.divmod(free, 3)
    supported_motors, supported_branches = np.divmod(supported, 3)
    free_muscles = membership[free_motors]
    constant = np.zeros(supported.size, dtype=np.float64)
    matrix = np.zeros((supported.size, free.size), dtype=np.float64)
    anchor_rows = np.zeros((variable.size, free.size), dtype=np.float64)
    support_lookup = {int(value): i for i, value in enumerate(supported)}
    free_lookup = {int(value): i for i, value in enumerate(free)}
    for row, (muscle, anchor_flat) in enumerate(zip(variable, anchors_array)):
        anchor_local = support_lookup[int(anchor_flat)]
        constant[anchor_local] = 1.0
        muscle_free = free[free_muscles == muscle]
        for flat in muscle_free:
            free_index = free_lookup[int(flat)]
            matrix[support_lookup[int(flat)], free_index] = 1.0
            matrix[anchor_local, free_index] = -1.0
            anchor_rows[row, free_index] = 1.0
    canonical = canonical_weights(pool)
    canonical_free = canonical[free_muscles, free_motors, free_branches].copy()
    lifted = canonical.copy()
    for muscle in variable:
        members = np.flatnonzero(membership == muscle)
        require(
            np.array_equal(
                pool[muscle, members],
                np.full(members.size, 1.0 / members.size),
            ),
            f"E146 original pool is not exactly uniform for muscle {muscle}",
        )
        lifted[muscle, members, 0] = pool[muscle, members] * guard_fraction
        lifted[muscle, members, 1] = pool[muscle, members] * (
            1.0 - 2.0 * guard_fraction
        )
        lifted[muscle, members, 2] = pool[muscle, members] * guard_fraction
    lifted_free = lifted[free_muscles, free_motors, free_branches].copy()
    reconstructed = constant + matrix @ canonical_free
    expected = canonical[muscles, supported_motors, supported_branches]
    require(
        np.max(np.abs(reconstructed - expected)) <= 1e-15,
        "E146 canonical anchor reconstruction differs",
    )
    lower = np.asarray(
        [
            guard_fraction / np.count_nonzero(membership == muscle)
            for muscle in free_muscles
        ]
    )
    return CausalAnchorMap(
        supported,
        supported_motors,
        supported_branches,
        muscles,
        anchors_array,
        free,
        free_motors,
        free_branches,
        free_muscles,
        constant,
        matrix,
        canonical_free,
        lifted_free,
        lower,
        anchor_rows,
        variable,
        fixed,
    )


def weights_from_free(
    baseline_pool: Any, anchor: CausalAnchorMap, free: Any
) -> np.ndarray:
    """Construct the exact affine W without clipping or normalization."""
    free = np.asarray(free, dtype=np.float64)
    require(
        free.shape == anchor.canonical_free.shape, "E146 free vector shape mismatch"
    )
    weights = canonical_weights(baseline_pool)
    weights[
        anchor.supported_muscles,
        anchor.supported_motors,
        anchor.supported_branches,
    ] = anchor.constant + anchor.matrix @ free
    return weights


def activation_from_weights(
    bank: Any,
    weights: Any,
    canonical_activation: Any,
    variable_muscles: Any,
) -> np.ndarray:
    """Evaluate W while reusing the 391-term canonical fixed outputs exactly."""
    bank = np.asarray(bank, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    activation = np.asarray(canonical_activation, dtype=np.float64).copy()
    for muscle in np.asarray(variable_muscles, dtype=np.int64):
        members = np.flatnonzero(np.any(weights[muscle] != 0, axis=1))
        activation[:, muscle] = np.einsum(
            "tik,ik->t",
            bank[:, members],
            weights[muscle, members],
            optimize=False,
        )
    return activation


def activation_affine(
    bank: Any,
    pool: Any,
    anchor: CausalAnchorMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Expose the exact activation affine map for generated-data checks."""
    bank = np.asarray(bank, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    middle = bank[:, :, MIDDLE_BRANCH]
    constant = middle @ pool.T
    design = np.zeros(
        (bank.shape[0], pool.shape[0], anchor.free_flat.size), dtype=np.float64
    )
    for muscle, anchor_flat in zip(anchor.variable_muscles, anchor.anchors_flat):
        motor, branch = divmod(int(anchor_flat), 3)
        constant[:, muscle] = bank[:, motor, branch]
    for local, (motor, branch, muscle) in enumerate(
        zip(
            anchor.free_motors,
            anchor.free_branches,
            anchor.free_muscles,
        )
    ):
        anchor_flat = anchor.anchors_flat[
            np.flatnonzero(anchor.variable_muscles == muscle)[0]
        ]
        anchor_motor, anchor_branch = divmod(int(anchor_flat), 3)
        design[:, muscle, local] = (
            bank[:, motor, branch] - bank[:, anchor_motor, anchor_branch]
        )
    return design, constant


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def evaluate_causal_composite(
    saved: dict[str, Any], candidate_weights: np.ndarray | None
) -> dict[str, Any]:
    """Evaluate canonical and candidate physical arrays directly."""
    canonical_activation = saved["b"] @ saved["pool"].T
    canonical = canonical_weights(saved["pool"])
    variants = {"baseline": canonical}
    if candidate_weights is not None:
        variants["candidate"] = candidate_weights
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    endpoint_rows = saved["target_indices"] - 1
    velocity_rows = saved["pair_end"] - 2
    for name, weights in variants.items():
        if name == "baseline":
            activation = canonical_activation
        else:
            activation = activation_from_weights(
                saved["bank"],
                weights,
                canonical_activation,
                saved["anchor"].variable_muscles,
            )
        command = saved["rest"] + COMMAND_GAIN * (activation @ saved["moment"].T)
        selected = command[endpoint_rows]
        velocity = np.diff(command, axis=0)[velocity_rows] / DT
        startup = command[:99]
        tendon = 0.2 * np.tanh(activation[:99, saved["ltm"]] / 10)
        arrays.update(
            {
                f"{name}_activation": activation,
                f"{name}_selected_command_rad": selected,
                f"{name}_selected_velocity_rad_s": velocity,
                f"{name}_startup_command_rad": startup,
                f"{name}_startup_tendon": tendon,
            }
        )
        command_error = selected - saved["target"]
        velocity_error = velocity - saved["target_velocity"]
        startup_error = startup - saved["startup_target"]
        tendon_error = tendon - saved["startup_tendon_target"]
        metrics[name] = {
            "command_RMS": _rms(command_error),
            "velocity_RMS": _rms(velocity_error),
            "startup_q_RMS": _rms(startup_error),
            "startup_q_max_abs": float(np.max(np.abs(startup_error))),
            "startup_LTM_RMS": _rms(tendon_error),
            "startup_LTM_max_abs": float(np.max(np.abs(tendon_error))),
            "by_leg": {
                leg: {
                    "command_RMS": _rms(command_error[:, 7 * i : 7 * (i + 1)]),
                    "velocity_RMS": _rms(velocity_error[:, 7 * i : 7 * (i + 1)]),
                }
                for i, leg in enumerate(LEGS)
            },
        }
    return {"arrays": arrays, "metrics": metrics}


def _embed_local(matrix: Any, indices: np.ndarray, width: int) -> sparse.csr_matrix:
    local = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    selector = sparse.csr_matrix(
        (np.ones(indices.size), (np.arange(indices.size), indices)),
        shape=(indices.size, width),
    )
    return local @ selector


def constant_activation(saved: dict[str, Any], anchor: CausalAnchorMap) -> np.ndarray:
    """Return the affine constant, keeping every fixed 391-term sum untouched."""
    activation = (saved["b"] @ saved["pool"].T).copy()
    for muscle, anchor_flat in zip(anchor.variable_muscles, anchor.anchors_flat):
        motor, branch = divmod(int(anchor_flat), 3)
        activation[:, muscle] = saved["bank"][:, motor, branch]
    return activation


def _leg_affine(
    saved: dict[str, Any],
    data: dict[str, Any],
    anchor: CausalAnchorMap,
    leg: str,
) -> dict[str, Any]:
    layout = build_leg_layout(
        leg,
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        data["anatomy"],
        saved["moment"],
    )
    joint_columns = layout.joint_columns
    local_free = np.flatnonzero(np.isin(anchor.free_muscles, layout.variable_muscles))
    local_supported = np.flatnonzero(
        np.isin(anchor.supported_muscles, layout.variable_muscles)
    )
    signal = saved["bank"][
        :,
        anchor.supported_motors[local_supported],
        anchor.supported_branches[local_supported],
    ]
    moment = saved["moment"][
        np.ix_(joint_columns, anchor.supported_muscles[local_supported])
    ].T
    supported_design = COMMAND_GAIN * signal[:, :, None] * moment[None, :, :]
    local_map = anchor.matrix[np.ix_(local_supported, local_free)]
    design = np.einsum("tsj,sf->tjf", supported_design, local_map, optimize=True)
    offset = saved["rest"][joint_columns] + COMMAND_GAIN * (
        saved["constant_activation"] @ saved["moment"][joint_columns].T
    )
    endpoint_rows = saved["target_indices"] - 1
    command_matrix = design[endpoint_rows].reshape(-1, local_free.size)
    command_offset = (
        offset[endpoint_rows] - saved["target"][:, joint_columns]
    ).reshape(-1)
    velocity_design = np.diff(design, axis=0)[saved["pair_end"] - 2] / DT
    velocity_offset = (
        np.diff(offset, axis=0)[saved["pair_end"] - 2] / DT
        - saved["target_velocity"][:, joint_columns]
    )
    startup_matrix = design[:99].reshape(-1, local_free.size)
    startup_offset = (offset[:99] - saved["startup_target"][:, joint_columns]).reshape(
        -1
    )
    return {
        "layout": layout,
        "free_indices": local_free,
        "command_matrix": command_matrix,
        "command_offset": command_offset,
        "velocity_matrix": velocity_design.reshape(-1, local_free.size),
        "velocity_offset": velocity_offset.reshape(-1),
        "startup_matrix": startup_matrix,
        "startup_offset": startup_offset,
    }


def _make_nonnegative_rows(
    anchor: CausalAnchorMap,
    membership: np.ndarray,
    startup_matrix: sparse.csr_matrix,
    startup_offset: np.ndarray,
    startup_max: float,
    margin_bounds: tuple[float, float],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = anchor.free_flat.size
    eye = sparse.eye(n, format="csr")
    zero = sparse.csr_matrix((n, 1))
    rows = [sparse.hstack((-eye, zero)), sparse.hstack((eye, zero))]
    rhs = [-anchor.lower, np.ones(n)]
    anchor_floor = np.asarray(
        [
            1e-6 / np.count_nonzero(membership == muscle)
            for muscle in anchor.variable_muscles
        ]
    )
    rows.append(
        sparse.hstack(
            (
                sparse.csr_matrix(anchor.anchor_rows),
                sparse.csr_matrix((anchor.variable_muscles.size, 1)),
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


def _problem_arrays(
    problem: conic.ConicProblem, saved: dict[str, Any]
) -> dict[str, np.ndarray]:
    anchor = problem.anchor_map
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
        "recorded_motor_rate": saved["recorded_rate"],
        "filter_bank": saved["bank"],
        "filter_tau_s": TAU_S,
        "baseline_pool": saved["pool"],
        "membership": saved["membership"],
        "moment": saved["moment"],
        "rest": saved["rest"],
        "ltm": saved["ltm"],
        "baseline_endpoint_rad": saved["baseline"],
        "target_indices": saved["target_indices"],
        "target_rad": saved["target"],
        "velocity_pair_start_indices": saved["pair_start"],
        "velocity_pair_end_indices": saved["pair_end"],
        "target_velocity_rad_s": saved["target_velocity"],
        "startup_target_rad": saved["startup_target"],
        "startup_target_tendon": saved["startup_tendon_target"],
        "fixed_muscle_mask": np.isin(np.arange(96), anchor.fixed_muscles),
        "variable_muscles": anchor.variable_muscles,
        "supported_flat": anchor.supported_flat,
        "supported_motors": anchor.supported_motors,
        "supported_branches": anchor.supported_branches,
        "supported_muscles": anchor.supported_muscles,
        "anchor_flat": anchor.anchors_flat,
        "free_flat": anchor.free_flat,
        "free_motors": anchor.free_motors,
        "free_branches": anchor.free_branches,
        "free_muscles": anchor.free_muscles,
        "anchor_constant": anchor.constant,
        "anchor_matrix": anchor.matrix,
        "canonical_free": anchor.canonical_free,
        "lifted_free": anchor.lifted_free,
        "free_lower_bounds": anchor.lower,
        "anchor_sum_rows": anchor.anchor_rows,
        "canonical_weights": canonical_weights(saved["pool"]),
        "box_lower": problem.lower,
        "box_upper": problem.upper,
        "criterion_names": np.asarray([item.name for item in problem.criteria]),
        "criterion_limits": np.asarray([item.limit for item in problem.criteria]),
    }


def _verify_lifted_feasibility(
    problem: conic.ConicProblem, freeze: dict[str, Any], raw: dict[str, np.ndarray]
) -> None:
    point = np.concatenate(
        (problem.anchor_map.lifted_free, [freeze["baseline_feasibility_margin"]])
    )
    slack = problem.b - problem.A @ point
    start = 0
    margins: list[float] = []
    for kind, dim in zip(problem.cone_kinds, problem.cone_dims):
        section = slack[start : start + dim]
        if kind == "NonnegativeConeT":
            margins.append(float(np.min(section)))
        else:
            margins.append(float(section[0] - np.linalg.norm(section[1:])))
        start += int(dim)
    raw["lifted_baseline_t"] = np.asarray(point[-1])
    raw["lifted_baseline_cone_margins"] = np.asarray(margins)
    require(np.min(margins) >= -1e-10, "E146 lifted baseline t=-.251 is infeasible")


def build_conic_problem(
    saved: dict[str, Any], data: dict[str, Any], freeze: dict[str, Any]
) -> tuple[conic.ConicProblem, dict[str, np.ndarray]]:
    """Build the exact 15-SOC guarded causal-filter problem."""
    anchor = build_causal_anchor_map(
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        guard_fraction=freeze["branch_guard_fraction"],
    )
    saved["anchor"] = anchor
    saved["constant_activation"] = constant_activation(saved, anchor)
    n = anchor.free_flat.size
    legs = [_leg_affine(saved, data, anchor, leg) for leg in LEGS]
    anatomy_muscles = np.concatenate([leg["layout"].muscles for leg in legs])
    require(
        np.array_equal(np.sort(anatomy_muscles), np.arange(96))
        and np.unique(anatomy_muscles).size == 96,
        "E146 anatomy does not partition all muscles exactly once",
    )
    baseline = evaluate_causal_composite(saved, None)["metrics"]["baseline"]
    criteria: list[conic.AffineCriterion] = []
    raw: dict[str, np.ndarray] = {}

    def add(name: str, blocks: list[tuple[Any, Any, Any]], limit: float) -> None:
        compressed_blocks = []
        compressed_offsets = []
        direct_blocks = []
        direct_offsets = []
        for block_index, (matrix, offset, indices) in enumerate(blocks):
            r = thin_qr_compress(matrix, offset)
            r_matrix, r_offset = compressed_affine(r)
            compressed_blocks.append(_embed_local(r_matrix, indices, n))
            compressed_offsets.append(r_offset)
            direct_blocks.append(_embed_local(matrix, indices, n))
            direct_offsets.append(offset)
            raw[f"qr_{name}_{block_index}"] = r
        compressed_matrix = sparse.vstack(compressed_blocks, format="csr")
        compressed_offset = np.concatenate(compressed_offsets)
        direct_matrix = sparse.vstack(direct_blocks, format="csr")
        direct_offset = np.concatenate(direct_offsets)
        criterion = conic.AffineCriterion(
            name,
            compressed_matrix,
            compressed_offset,
            float(limit),
            direct_matrix,
            direct_offset,
        )
        criteria.append(criterion)
        raw[f"criterion_{name}_limit"] = np.asarray(limit)
        for suffix, vector in (
            ("canonical", anchor.canonical_free),
            ("lifted", anchor.lifted_free),
        ):
            direct_norm = np.linalg.norm(direct_matrix @ vector + direct_offset)
            compressed_norm = np.linalg.norm(
                compressed_matrix @ vector + compressed_offset
            )
            raw[f"criterion_{name}_{suffix}_direct_norm"] = np.asarray(direct_norm)
            raw[f"criterion_{name}_{suffix}_compressed_norm"] = np.asarray(
                compressed_norm
            )
            tolerance = (
                freeze["QR_absolute_atol"] + freeze["QR_relative_atol"] * direct_norm
            )
            require(
                abs(direct_norm - compressed_norm) <= tolerance,
                f"E146 {name} QR norm mismatch",
            )

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
    require(len(criteria) == 15, "E146 SOC count changed")
    startup_matrix = sparse.vstack(
        [_embed_local(m, idx, n) for m, _offset, idx in startup_blocks], format="csr"
    )
    startup_offset = np.concatenate([offset for _m, offset, _idx in startup_blocks])
    nonnegative_A, nonnegative_b = _make_nonnegative_rows(
        anchor,
        saved["membership"],
        startup_matrix,
        startup_offset,
        freeze["startup_limits"]["q_max_abs"],
        tuple(freeze["margin_bounds"]),
    )
    A_blocks = [nonnegative_A]
    b_blocks = [nonnegative_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(nonnegative_A.shape[0])]
    kinds = ["NonnegativeConeT"]
    dims = [nonnegative_A.shape[0]]
    for criterion in criteria:
        rows, rhs = make_soc_rows(
            criterion.matrix, criterion.offset, criterion.limit, n
        )
        A_blocks.append(rows)
        b_blocks.append(rhs)
        cones.append(clarabel.SecondOrderConeT(rows.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(rows.shape[0])
    A = sparse.vstack(A_blocks, format="csc")
    b = np.concatenate(b_blocks)
    q = np.zeros(n + 1, dtype=np.float64)
    q[-1] = -1.0
    problem = conic.ConicProblem(
        sparse.csc_matrix((n + 1, n + 1)),
        q,
        A,
        b,
        tuple(cones),
        tuple(kinds),
        np.asarray(dims),
        np.concatenate((np.zeros(n), np.asarray(freeze["margin_bounds"][:1]))),
        np.concatenate((np.ones(n), np.asarray(freeze["margin_bounds"][1:]))),
        anchor,
        tuple(criteria),
        startup_matrix,
        startup_offset,
    )
    raw.update(_problem_arrays(problem, saved))
    lifted_weights = weights_from_free(saved["pool"], anchor, anchor.lifted_free)
    lifted_composite = evaluate_causal_composite(saved, lifted_weights)
    lifted_delta = (
        lifted_composite["arrays"]["candidate_selected_command_rad"]
        - lifted_composite["arrays"]["baseline_selected_command_rad"]
    )
    raw["lifted_weights"] = lifted_weights
    raw["lifted_minus_canonical_selected_command_rad"] = lifted_delta
    raw["lifted_minus_canonical_command_RMS"] = np.asarray(_rms(lifted_delta))
    raw["lifted_minus_canonical_command_max_abs"] = np.asarray(
        np.max(np.abs(lifted_delta))
    )
    _verify_lifted_feasibility(problem, freeze, raw)
    return problem, raw


def assess_physical_candidate(
    saved: dict[str, Any],
    weights: np.ndarray,
    composite: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    anchor = saved["anchor"]
    weights = np.asarray(weights, dtype=np.float64)
    canonical = canonical_weights(saved["pool"])
    support = np.zeros_like(weights, dtype=bool)
    support[saved["membership"], np.arange(391), :] = True
    fixed_rows_exact = all(
        np.array_equal(weights[muscle], canonical[muscle])
        for muscle in anchor.fixed_muscles
    )
    row_sums = weights.sum(axis=(1, 2))
    motor_sums = weights[saved["membership"], np.arange(391)].sum(axis=1)
    motor_floors = np.asarray(
        [
            freeze["original_MN_floor_fraction"]
            / np.count_nonzero(saved["membership"] == muscle)
            for muscle in saved["membership"]
        ]
    )
    candidate = composite["metrics"]["candidate"]
    baseline = composite["metrics"]["baseline"]
    arrays = composite["arrays"]
    limits = freeze["startup_limits"]
    gates = {
        "finite_weights": bool(np.isfinite(weights).all()),
        "same_anatomical_support": bool(np.all(weights[~support] == 0)),
        "all_branches_nonnegative": bool(np.all(weights[support] >= 0)),
        "row_sum_atol_1e_12": bool(
            np.allclose(row_sums, 1.0, atol=freeze["row_sum_atol"], rtol=0)
        ),
        "original_motor_floor_with_tolerance": bool(
            np.all(motor_sums >= motor_floors - freeze["lower_bound_atol"])
        ),
        "fixed_LTM_and_singleton_rows_exact": fixed_rows_exact,
        "fixed_activation_output_exact": bool(
            np.array_equal(
                arrays["candidate_activation"][:, anchor.fixed_muscles],
                arrays["baseline_activation"][:, anchor.fixed_muscles],
            )
        ),
        "fixed_tendon_output_exact": bool(
            np.array_equal(
                arrays["candidate_startup_tendon"], arrays["baseline_startup_tendon"]
            )
        ),
        "startup_q_RMS": candidate["startup_q_RMS"] <= limits["q_RMS"],
        "startup_q_max_abs": candidate["startup_q_max_abs"] <= limits["q_max_abs"],
        "startup_LTM_RMS": candidate["startup_LTM_RMS"] <= limits["LTM_RMS"],
        "startup_LTM_max_abs": candidate["startup_LTM_max_abs"]
        <= limits["LTM_max_abs"],
        "overall_command_20pct_improvement": candidate["command_RMS"]
        <= freeze["proposal_command_RMS_ratio"] * baseline["command_RMS"],
        "overall_velocity_nonincrease": candidate["velocity_RMS"]
        <= freeze["proposal_velocity_RMS_ratio"] * baseline["velocity_RMS"],
        "each_leg_command_nonincrease": all(
            candidate["by_leg"][leg]["command_RMS"]
            <= baseline["by_leg"][leg]["command_RMS"]
            for leg in LEGS
        ),
        "each_leg_velocity_nonincrease": all(
            candidate["by_leg"][leg]["velocity_RMS"]
            <= baseline["by_leg"][leg]["velocity_RMS"]
            for leg in LEGS
        ),
    }
    proposal = bool(all(gates.values()))
    return {
        "conditional_causal_filter_proposal": proposal,
        "gates": gates,
        "row_sum_max_abs_error": float(np.max(np.abs(row_sums - 1))),
        "runtime_row_sum_compatible_1e_15": bool(
            np.allclose(row_sums, 1.0, atol=1e-15, rtol=0)
        ),
        "minimum_branch_weight": float(np.min(weights[support])),
        "minimum_original_motor_floor_margin": float(np.min(motor_sums - motor_floors)),
        "solver_status_used_as_proposal_gate": False,
        "bounded_failure_is_global_impossibility": False,
    }


def calculate(
    data: dict[str, Any],
    freeze: dict[str, Any],
    progress: dict[str, Any],
    partial: dict[str, Any],
    prepared: tuple[conic.ConicProblem, dict[str, np.ndarray], dict[str, Any]]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], conic.ConicProblem]:
    """Solve once and return raw evidence before any verdict/resource check."""
    if prepared is None:
        saved = validate_causal_saved_inputs(
            data, freeze, partial=partial, progress=progress
        )
        problem, pre_raw = build_conic_problem(saved, data, freeze)
    else:
        problem, pre_raw, saved = prepared
    partial.update(pre_raw)
    progress["problem_built"] = True
    history: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    fit_started = time.perf_counter()
    try:
        solution, history, final_settings = solve_conic_problem(
            problem,
            freeze,
            partial=partial,
            progress=progress,
            state=state,
            history=history,
        )
        partial.update(conic._bare_solution_arrays(solution))
        partial.update(conic._history_arrays(history))
        partial["solver_settings_json"] = np.asarray(
            json.dumps(final_settings, sort_keys=True, allow_nan=False)
        )
    finally:
        progress["timings_s"]["fit"] = time.perf_counter() - fit_started
        progress["fit_wall_s"] = progress["timings_s"]["fit"]
    require(
        progress["timings_s"]["fit"] <= freeze["total_fit_limit_s"],
        "E146 fit exceeded 30 seconds",
    )
    result_arrays = solution_arrays(problem, solution, history, freeze)
    raw = {**pre_raw, **result_arrays}
    raw["solver_settings_json"] = np.asarray(
        json.dumps(final_settings, sort_keys=True, allow_nan=False)
    )
    partial.update(result_arrays)
    weights = weights_from_free(
        saved["pool"], problem.anchor_map, np.asarray(solution.x)[:-1]
    )
    raw["candidate_weights"] = weights
    raw["returned_margin_t"] = np.asarray(solution.x[-1])
    partial["candidate_weights"] = weights
    partial["returned_margin_t"] = np.asarray(solution.x[-1])
    composite = evaluate_causal_composite(saved, weights)
    raw.update(composite["arrays"])
    raw.update(composite_metrics_to_raw(composite["metrics"]))
    partial.update(raw)
    facts = {
        "solver": {
            "status": str(solution.status),
            "iterations": int(solution.iterations),
            "solve_time_s": float(solution.solve_time),
            "primal_objective": float(solution.obj_val),
            "dual_objective": float(solution.obj_val_dual),
            "returned_margin_t": float(solution.x[-1]),
            "calls": int(progress["fit_calls"]),
        },
        "metrics": composite["metrics"],
    }
    return (
        raw,
        facts,
        {"saved": saved, "weights": weights, "composite": composite},
        problem,
    )


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e146.causal-filter-result.v1",
        "experiment": "M3-E146-causal-filter",
        "status": "failed",
        "validity": "invalid_incomplete",
        "conditional_causal_filter_proposal": False,
        "candidate_adopted": False,
        "F1_F8": "pending",
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
        "failure": progress.get(
            "primary_failure", {"type": type(primary).__name__, "message": str(primary)}
        ),
    }


def run(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "schema": "flylab.m3.e146.causal-filter-execution.v1",
        "status": "created",
        "original_neural_steps": 0,
        "CUDA_calls": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 0,
        "recorded_EMA_scalar_updates": 0,
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
                TimeoutError(f"E146 exceeded internal wall during {stage}"),
            ),
            (
                size is not None and size > freeze["maximum_raw_bytes"],
                "maximum_raw_bytes",
                RuntimeError(f"E146 exceeded raw byte limit during {stage}"),
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
            "E146 insufficient starting free disk",
        )
        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(
                all(
                    int(item.get("num_threads", 1)) <= 1
                    for item in progress["thread_pools"]
                ),
                "E146 CPU thread pool exceeds one thread",
            )
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            saved = validate_causal_saved_inputs(
                data, freeze, partial=partial, progress=progress
            )
            pre_problem, pre_raw = build_conic_problem(saved, data, freeze)
            save_npz_new(output / "conic-problem.npz", pre_raw)
            problem_saved = True
            progress["problem_persisted_before_solve"] = True
            raw, facts, outcome, _problem = calculate(
                data,
                freeze,
                progress,
                partial,
                prepared=(pre_problem, pre_raw, saved),
            )
            save_npz_new(output / "causal-filter-raw.npz", raw)
            raw_saved = True
            assessment = assess_physical_candidate(
                outcome["saved"], outcome["weights"], outcome["composite"], freeze
            )
        proposal = bool(assessment["conditional_causal_filter_proposal"])
        projected_bound = float(raw["dual_projected_margin_upper_bound"])
        verdict = (
            "conditional_causal_filter_proposal"
            if proposal
            else "guarded_causal_filter_refuted"
            if projected_bound < 0
            else "guarded_causal_filter_inconclusive"
        )
        summary = {
            "schema": "flylab.m3.e146.causal-filter-result.v1",
            "experiment": "M3-E146-causal-filter",
            "status": "pass",
            "validity": "conditional_saved_E143_single_global_SOCP",
            "hypothesis_verdict": verdict,
            "analysis": facts,
            "assessment": assessment,
            "conditional_causal_filter_proposal": proposal,
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
            "interpretation": {
                "scope": "one saved-data three-branch causal-filter SOCP",
                "numerical_upper_bound": "floating-point guarded-family diagnostic",
                "biology": "engineering activation-response hypothesis only",
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
