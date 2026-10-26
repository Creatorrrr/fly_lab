"""E147 bounded muscle-gain SOCP over the saved E146 causal filter bank."""

# ruff: noqa: BLE001, I001  # thread limits must precede numerical imports

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

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

import fit_c_banc_m3_causal_filter as causal
import fit_c_banc_m3_conic_pool as conic
import fit_c_banc_m3_static_pool as static

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "5b5b8792ca74fdf64a7e4527daaa9308d65eee46af7831582b3cdd854ef23812"
DESIGN_SHA = "dd685f004575c9f7d9f6ae40ab10dfc61664f0b91372ded531c2e7f52b3399a5"
SCHEMA = "flylab.m3.e147.muscle-gain.v1"
OWNED = ("tools/fit_c_banc_m3_muscle_gain.py", "tests/test_c_banc_muscle_gain.py")
LEGS = static.LEGS
DT = static.DT
COMMAND_GAIN = static.COMMAND_GAIN
MIDDLE_BRANCH = causal.MIDDLE_BRANCH
DEFAULT_FREEZE = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e147-muscle-gain-freeze.json"
)

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
        f"Secondary E147 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E147_MUSCLE_GAIN_DESIGN_20260921.md",
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
        "prior_conservative_process_s": 2200,
        "prior_conservative_actual_s": 2199.860127400025,
        "prospective_cumulative_limit_s": 2270,
        "original_program_cumulative_limit_s": 2271,
        "maximum_raw_bytes": 134217728,
        "minimum_free_disk_bytes": 1073741824,
        "original_neural_steps": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 0,
        "recorded_EMA_scalar_updates": 0,
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
        "middle_branch": 1,
        "reused_filter_vector_updates": 2000,
        "gain_bounds": [0.5, 2.0],
        "solver_gain_bounds": [0.500001, 1.999999],
        "weight_upper_bound": 2.0,
        "parameter_order": "muscle-id ascending, then motor-major flat 3*i+k",
        "anchor_parameterization": False,
    }


def reviewed_inputs(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete freeze and accepted two-source review before input load."""
    require(freeze_sha == FREEZE_SHA, "Exact E147 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E147 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E147 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E147 design changed")
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
    require(str(clarabel.__version__) == "0.11.1", "Clarabel version changed")
    require(sha256(review_path) == review_sha, "E147 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_muscle_gain_fit") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E147 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def load_inputs(freeze: dict[str, Any]) -> dict[str, Any]:
    data = static.load_inputs(freeze)
    raw_path = next(
        key
        for key in freeze["input_sha256"]
        if key.endswith("e146-causal-filter/causal-filter-raw.npz")
    )
    data["e146"] = _load_npz(ROOT / raw_path)
    return data


def validate_saved_inputs(
    data: dict[str, Any],
    freeze: dict[str, Any],
    *,
    base_validator: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = static.validate_saved_inputs,
    partial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate unchanged E143 inputs and the reused E146 bank without filtering."""
    raw = data["e146"]
    if partial is not None:
        partial["unvalidated_filter_bank"] = np.asarray(raw.get("filter_bank", []))
        partial["unvalidated_filter_tau_s"] = np.asarray(raw.get("filter_tau_s", []))
    saved = base_validator(data, freeze)
    bank = static.finite_array(raw["filter_bank"], (2000, 391, 3), "E146 bank")
    tau = static.finite_array(raw["filter_tau_s"], (3,), "E146 tau")
    require(np.array_equal(tau, np.asarray(freeze["tau_s"])), "E147 tau changed")
    require(
        np.array_equal(bank[:, :, MIDDLE_BRANCH], saved["b"]),
        "E147 reused bank middle branch differs from E143",
    )
    exact_pairs = {
        "baseline_pool": "pool",
        "membership": "membership",
        "moment": "moment",
        "rest": "rest",
        "ltm": "ltm",
        "target_indices": "target_indices",
        "target_rad": "target",
        "velocity_pair_start_indices": "pair_start",
        "velocity_pair_end_indices": "pair_end",
        "target_velocity_rad_s": "target_velocity",
        "startup_target_rad": "startup_target",
        "startup_target_tendon": "startup_tendon_target",
    }
    for raw_key, saved_key in exact_pairs.items():
        require(
            np.array_equal(raw[raw_key], saved[saved_key]),
            f"E147 E146 raw differs for {raw_key}",
        )
    saved.update(bank=bank, tau=tau)
    if partial is not None:
        partial["filter_bank"] = bank
        partial["filter_tau_s"] = tau
    return saved


@dataclass(frozen=True)
class GainMap:
    variable_flat: np.ndarray
    variable_motors: np.ndarray
    variable_branches: np.ndarray
    variable_muscle_for_coefficient: np.ndarray
    variable_muscles: np.ndarray
    fixed_muscles: np.ndarray
    canonical_vector: np.ndarray
    lifted_vector: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    gain_rows: sparse.csr_matrix


def canonical_weights(pool: Any) -> np.ndarray:
    return causal.canonical_weights(pool)


def build_gain_map(
    pool: Any,
    membership: Any,
    ltm: Any,
    *,
    guard_fraction: float = 1e-6,
    upper_bound: float = 2.0,
) -> GainMap:
    """Order direct coefficients by muscle, then motor-major flat branch."""
    pool = np.asarray(pool, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)
    fixed_set = {int(value) for value in np.asarray(ltm)}
    fixed = np.asarray(
        [
            muscle
            for muscle in range(pool.shape[0])
            if muscle in fixed_set or np.count_nonzero(membership == muscle) == 1
        ],
        dtype=np.int64,
    )
    variable = np.asarray(
        [muscle for muscle in range(pool.shape[0]) if muscle not in set(fixed)],
        dtype=np.int64,
    )
    flats: list[int] = []
    muscles: list[int] = []
    for muscle in variable:
        members = np.flatnonzero(membership == muscle)
        require(
            np.array_equal(
                pool[muscle, members], np.full(members.size, 1.0 / members.size)
            ),
            f"E147 original pool is not exactly uniform for muscle {muscle}",
        )
        for motor in members:
            for branch in range(3):
                flats.append(3 * int(motor) + branch)
                muscles.append(int(muscle))
    variable_flat = np.asarray(flats, dtype=np.int64)
    coefficient_muscles = np.asarray(muscles, dtype=np.int64)
    motors, branches = np.divmod(variable_flat, 3)
    canonical = canonical_weights(pool)
    canonical_vector = canonical[coefficient_muscles, motors, branches].copy()
    lifted = canonical.copy()
    for muscle in variable:
        members = np.flatnonzero(membership == muscle)
        lifted[muscle, members, 0] = pool[muscle, members] * guard_fraction
        lifted[muscle, members, 1] = pool[muscle, members] * (1 - 2 * guard_fraction)
        lifted[muscle, members, 2] = pool[muscle, members] * guard_fraction
    lifted_vector = lifted[coefficient_muscles, motors, branches].copy()
    lower = np.asarray(
        [
            guard_fraction / np.count_nonzero(membership == muscle)
            for muscle in coefficient_muscles
        ]
    )
    rows = np.searchsorted(variable, coefficient_muscles)
    gain_rows = sparse.csr_matrix(
        (np.ones(variable_flat.size), (rows, np.arange(variable_flat.size))),
        shape=(variable.size, variable_flat.size),
    )
    require(
        np.max(np.abs(gain_rows @ canonical_vector - 1)) <= 1e-15,
        "E147 canonical gains differ from one",
    )
    require(
        np.allclose(gain_rows @ lifted_vector, 1, atol=1e-15, rtol=0),
        "E147 lifted gains differ from one",
    )
    return GainMap(
        variable_flat,
        motors,
        branches,
        coefficient_muscles,
        variable,
        fixed,
        canonical_vector,
        lifted_vector,
        lower,
        np.full(variable_flat.size, upper_bound),
        gain_rows,
    )


def weights_from_vector(pool: Any, mapping: GainMap, vector: Any) -> np.ndarray:
    """Copy direct coefficients into W without clipping or normalization."""
    vector = np.asarray(vector, dtype=np.float64)
    require(
        vector.shape == mapping.canonical_vector.shape, "E147 vector shape mismatch"
    )
    weights = canonical_weights(pool)
    weights[
        mapping.variable_muscle_for_coefficient,
        mapping.variable_motors,
        mapping.variable_branches,
    ] = vector
    return weights


def gains_from_vector(mapping: GainMap, vector: Any) -> np.ndarray:
    gains = np.ones(96, dtype=np.float64)
    gains[mapping.variable_muscles] = mapping.gain_rows @ np.asarray(vector)
    return gains


def activation_from_weights(
    bank: Any, weights: Any, canonical_activation: Any, variable_muscles: Any
) -> np.ndarray:
    return causal.activation_from_weights(
        bank, weights, canonical_activation, variable_muscles
    )


def evaluate_composite(
    saved: dict[str, Any], candidate_weights: np.ndarray | None
) -> dict[str, Any]:
    return causal.evaluate_causal_composite(saved, candidate_weights)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def fixed_activation(saved: dict[str, Any], mapping: GainMap) -> np.ndarray:
    activation = saved["b"] @ saved["pool"].T
    activation[:, mapping.variable_muscles] = 0.0
    return activation


def _embed_local(matrix: Any, indices: np.ndarray, width: int) -> sparse.csr_matrix:
    local = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    selector = sparse.csr_matrix(
        (np.ones(indices.size), (np.arange(indices.size), indices)),
        shape=(indices.size, width),
    )
    return local @ selector


def _leg_affine(
    saved: dict[str, Any], data: dict[str, Any], mapping: GainMap, leg: str
) -> dict[str, Any]:
    layout = static.build_leg_layout(
        leg,
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        data["anatomy"],
        saved["moment"],
    )
    local = np.flatnonzero(
        np.isin(mapping.variable_muscle_for_coefficient, layout.variable_muscles)
    )
    motors = mapping.variable_motors[local]
    branches = mapping.variable_branches[local]
    muscles = mapping.variable_muscle_for_coefficient[local]
    signal = saved["bank"][:, motors, branches]
    moment = saved["moment"][np.ix_(layout.joint_columns, muscles)].T
    design = (COMMAND_GAIN * signal[:, :, None] * moment[None, :, :]).transpose(0, 2, 1)
    offset = saved["rest"][layout.joint_columns] + COMMAND_GAIN * (
        saved["fixed_activation"] @ saved["moment"][layout.joint_columns].T
    )
    endpoint_rows = saved["target_indices"] - 1
    command_matrix = design[endpoint_rows].reshape(-1, local.size)
    command_offset = (
        offset[endpoint_rows] - saved["target"][:, layout.joint_columns]
    ).reshape(-1)
    velocity_matrix = (np.diff(design, axis=0)[saved["pair_end"] - 2] / DT).reshape(
        -1, local.size
    )
    velocity_offset = (
        np.diff(offset, axis=0)[saved["pair_end"] - 2] / DT
        - saved["target_velocity"][:, layout.joint_columns]
    ).reshape(-1)
    startup_matrix = design[:99].reshape(-1, local.size)
    startup_offset = (
        offset[:99] - saved["startup_target"][:, layout.joint_columns]
    ).reshape(-1)
    return {
        "layout": layout,
        "indices": local,
        "command_matrix": command_matrix,
        "command_offset": command_offset,
        "velocity_matrix": velocity_matrix,
        "velocity_offset": velocity_offset,
        "startup_matrix": startup_matrix,
        "startup_offset": startup_offset,
    }


def _make_nonnegative_rows(
    mapping: GainMap,
    startup_matrix: sparse.csr_matrix,
    startup_offset: np.ndarray,
    startup_max: float,
    margin_bounds: tuple[float, float],
    solver_gain_bounds: tuple[float, float],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = mapping.variable_flat.size
    zero = sparse.csr_matrix((n, 1))
    eye = sparse.eye(n, format="csr")
    rows = [sparse.hstack((-eye, zero)), sparse.hstack((eye, zero))]
    rhs = [-mapping.lower, mapping.upper]
    gain_zero = sparse.csr_matrix((mapping.gain_rows.shape[0], 1))
    gain_low, gain_high = solver_gain_bounds
    rows.extend(
        [
            sparse.hstack((-mapping.gain_rows, gain_zero)),
            sparse.hstack((mapping.gain_rows, gain_zero)),
        ]
    )
    rhs.extend(
        [
            np.full(mapping.gain_rows.shape[0], -gain_low),
            np.full(mapping.gain_rows.shape[0], gain_high),
        ]
    )
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
    rhs.extend([1 - startup_offset / startup_max, 1 + startup_offset / startup_max])
    return sparse.vstack(rows, format="csr"), np.concatenate(rhs)


def _problem_arrays(
    problem: conic.ConicProblem, saved: dict[str, Any]
) -> dict[str, np.ndarray]:
    mapping = problem.anchor_map
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
        "filter_bank": saved["bank"],
        "filter_tau_s": saved["tau"],
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
        "fixed_muscle_mask": np.isin(np.arange(96), mapping.fixed_muscles),
        "variable_muscles": mapping.variable_muscles,
        "variable_flat": mapping.variable_flat,
        "variable_motors": mapping.variable_motors,
        "variable_branches": mapping.variable_branches,
        "variable_muscle_for_coefficient": mapping.variable_muscle_for_coefficient,
        "variable_muscle_ids": mapping.variable_muscle_for_coefficient,
        "fitted_muscle_ids": mapping.variable_muscles,
        "gain_row_data": mapping.gain_rows.data,
        "gain_row_indices": mapping.gain_rows.indices,
        "gain_row_indptr": mapping.gain_rows.indptr,
        "gain_row_shape": np.asarray(mapping.gain_rows.shape),
        "row_sum_matrix": mapping.gain_rows.toarray(),
        "canonical_vector": mapping.canonical_vector,
        "lifted_vector": mapping.lifted_vector,
        "canonical_weights": canonical_weights(saved["pool"]),
        "lifted_weights": weights_from_vector(
            saved["pool"], mapping, mapping.lifted_vector
        ),
        "coefficient_solver_lower": mapping.lower,
        "coefficient_solver_upper": mapping.upper,
        "weight_lower_bounds": mapping.lower,
        "box_lower": problem.lower,
        "box_upper": problem.upper,
        "criterion_names": np.asarray([item.name for item in problem.criteria]),
        "criterion_limits": np.asarray([item.limit for item in problem.criteria]),
    }


def _verify_lifted_feasibility(
    problem: conic.ConicProblem, freeze: dict[str, Any], raw: dict[str, np.ndarray]
) -> None:
    point = np.concatenate(
        (problem.anchor_map.lifted_vector, [freeze["baseline_feasibility_margin"]])
    )
    slack = problem.b - problem.A @ point
    margins: list[float] = []
    start = 0
    for kind, dim in zip(problem.cone_kinds, problem.cone_dims):
        section = slack[start : start + dim]
        margin = (
            np.min(section)
            if kind == "NonnegativeConeT"
            else section[0] - np.linalg.norm(section[1:])
        )
        margins.append(float(margin))
        start += int(dim)
    raw["lifted_baseline_t"] = np.asarray(point[-1])
    raw["lifted_baseline_cone_margins"] = np.asarray(margins)
    require(np.min(margins) >= -1e-10, "E147 lifted baseline t=-.251 is infeasible")


def build_conic_problem(
    saved: dict[str, Any], data: dict[str, Any], freeze: dict[str, Any]
) -> tuple[conic.ConicProblem, dict[str, np.ndarray]]:
    mapping = build_gain_map(
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        guard_fraction=freeze["branch_guard_fraction"],
        upper_bound=freeze["weight_upper_bound"],
    )
    saved["anchor"] = mapping
    saved["fixed_activation"] = fixed_activation(saved, mapping)
    legs = [_leg_affine(saved, data, mapping, leg) for leg in LEGS]
    anatomy_muscles = np.concatenate([leg["layout"].muscles for leg in legs])
    require(
        np.array_equal(np.sort(anatomy_muscles), np.arange(96))
        and np.unique(anatomy_muscles).size == 96,
        "E147 anatomy does not partition all muscles exactly once",
    )
    baseline = evaluate_composite(saved, None)["metrics"]["baseline"]
    n = mapping.variable_flat.size
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
        criteria.append(
            conic.AffineCriterion(
                name,
                compressed_matrix,
                compressed_offset,
                float(limit),
                direct_matrix,
                direct_offset,
            )
        )
        raw[f"criterion_{name}_limit"] = np.asarray(limit)
        for suffix, vector in (
            ("canonical", mapping.canonical_vector),
            ("lifted", mapping.lifted_vector),
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
                f"E147 {name} QR norm mismatch",
            )

    command_blocks = [
        (leg["command_matrix"], leg["command_offset"], leg["indices"]) for leg in legs
    ]
    velocity_blocks = [
        (leg["velocity_matrix"], leg["velocity_offset"], leg["indices"]) for leg in legs
    ]
    startup_blocks = [
        (leg["startup_matrix"], leg["startup_offset"], leg["indices"]) for leg in legs
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
    for index, leg in enumerate(LEGS):
        add(
            f"{leg}_command",
            [command_blocks[index]],
            np.sqrt(1827 * 7) * baseline["by_leg"][leg]["command_RMS"],
        )
        add(
            f"{leg}_velocity",
            [velocity_blocks[index]],
            np.sqrt(1826 * 7) * baseline["by_leg"][leg]["velocity_RMS"],
        )
    add(
        "overall_startup",
        startup_blocks,
        np.sqrt(99 * 42) * freeze["startup_limits"]["q_RMS"],
    )
    require(len(criteria) == 15, "E147 SOC count changed")
    startup_matrix = sparse.vstack(
        [
            _embed_local(matrix, indices, n)
            for matrix, _offset, indices in startup_blocks
        ],
        format="csr",
    )
    startup_offset = np.concatenate(
        [offset for _matrix, offset, _indices in startup_blocks]
    )
    nonnegative_A, nonnegative_b = _make_nonnegative_rows(
        mapping,
        startup_matrix,
        startup_offset,
        freeze["startup_limits"]["q_max_abs"],
        tuple(freeze["margin_bounds"]),
        tuple(freeze["solver_gain_bounds"]),
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
    q = np.zeros(n + 1, dtype=np.float64)
    q[-1] = -1.0
    problem = conic.ConicProblem(
        sparse.csc_matrix((n + 1, n + 1)),
        q,
        sparse.vstack(A_blocks, format="csc"),
        np.concatenate(b_blocks),
        tuple(cones),
        tuple(kinds),
        np.asarray(dims),
        np.concatenate((np.zeros(n), np.asarray(freeze["margin_bounds"][:1]))),
        np.concatenate(
            (
                np.full(n, freeze["weight_upper_bound"]),
                np.asarray(freeze["margin_bounds"][1:]),
            )
        ),
        mapping,
        tuple(criteria),
        startup_matrix,
        startup_offset,
    )
    raw.update(_problem_arrays(problem, saved))
    lifted_weights = weights_from_vector(saved["pool"], mapping, mapping.lifted_vector)
    lifted_composite = evaluate_composite(saved, lifted_weights)
    delta = (
        lifted_composite["arrays"]["candidate_selected_command_rad"]
        - lifted_composite["arrays"]["baseline_selected_command_rad"]
    )
    raw["lifted_minus_canonical_selected_command_rad"] = delta
    raw["lifted_minus_canonical_command_RMS"] = np.asarray(_rms(delta))
    raw["lifted_minus_canonical_command_max_abs"] = np.asarray(np.max(np.abs(delta)))
    _verify_lifted_feasibility(problem, freeze, raw)
    return problem, raw


def assess_physical_candidate(
    saved: dict[str, Any],
    weights: np.ndarray,
    candidate_gain: np.ndarray,
    composite: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    mapping = saved["anchor"]
    canonical = canonical_weights(saved["pool"])
    support = np.zeros_like(weights, dtype=bool)
    support[saved["membership"], np.arange(saved["membership"].size), :] = True
    fixed_exact = all(
        np.array_equal(weights[muscle], canonical[muscle])
        for muscle in mapping.fixed_muscles
    )
    motor_sums = weights[saved["membership"], np.arange(saved["membership"].size)].sum(
        axis=1
    )
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
    gain_low, gain_high = freeze["gain_bounds"]
    tolerance = freeze["lower_bound_atol"]
    physical_gain = weights.sum(axis=(1, 2))
    gates = {
        "finite_weights_and_gain": bool(
            np.isfinite(weights).all() and np.isfinite(candidate_gain).all()
        ),
        "same_anatomical_support": bool(np.all(weights[~support] == 0)),
        "all_branches_nonnegative": bool(np.all(weights[support] >= 0)),
        "gain_bounds_with_tolerance": bool(
            np.all(physical_gain >= gain_low - tolerance)
            and np.all(physical_gain <= gain_high + tolerance)
        ),
        "candidate_gain_matches_effective_weights": bool(
            np.allclose(candidate_gain, physical_gain, atol=1e-15, rtol=0)
        ),
        "original_motor_floor_with_tolerance": bool(
            np.all(motor_sums >= motor_floors - tolerance)
        ),
        "fixed_LTM_and_singleton_rows_exact": fixed_exact,
        "fixed_activation_output_exact": bool(
            np.array_equal(
                arrays["candidate_activation"][:, mapping.fixed_muscles],
                arrays["baseline_activation"][:, mapping.fixed_muscles],
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
    return {
        "conditional_muscle_gain_proposal": bool(all(gates.values())),
        "gates": gates,
        "minimum_branch_weight": float(np.min(weights[support])),
        "minimum_original_motor_floor_margin": float(np.min(motor_sums - motor_floors)),
        "minimum_gain": float(np.min(physical_gain)),
        "maximum_gain": float(np.max(physical_gain)),
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
    if prepared is None:
        saved = validate_saved_inputs(data, freeze, partial=partial)
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
        progress["fit_wall_s"] <= freeze["total_fit_limit_s"],
        "E147 fit exceeded 30 seconds",
    )
    result_arrays = solution_arrays(problem, solution, history, freeze)
    raw = {**pre_raw, **result_arrays}
    raw["solver_settings_json"] = np.asarray(
        json.dumps(final_settings, sort_keys=True, allow_nan=False)
    )
    partial.update(result_arrays)
    vector = np.asarray(solution.x)[:-1]
    weights = weights_from_vector(saved["pool"], problem.anchor_map, vector)
    constraint_gain_csr = gains_from_vector(problem.anchor_map, vector)
    candidate_gain = weights.sum(axis=(1, 2))
    raw.update(
        candidate_vector=vector,
        candidate_weights=weights,
        candidate_gain=candidate_gain,
        candidate_gain_constraint_csr=constraint_gain_csr,
        returned_margin_t=np.asarray(solution.x[-1]),
    )
    partial.update(
        candidate_vector=vector,
        candidate_weights=weights,
        candidate_gain=candidate_gain,
        candidate_gain_constraint_csr=constraint_gain_csr,
        returned_margin_t=np.asarray(solution.x[-1]),
    )
    composite = evaluate_composite(saved, weights)
    raw.update(composite["arrays"])
    partial.update(composite["arrays"])
    metrics_raw = static.composite_metrics_to_raw(composite["metrics"])
    raw.update(metrics_raw)
    partial.update(metrics_raw)
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
        {
            "saved": saved,
            "weights": weights,
            "candidate_gain": candidate_gain,
            "composite": composite,
        },
        problem,
    )


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e147.muscle-gain-result.v1",
        "experiment": "M3-E147-muscle-gain",
        "status": "failed",
        "validity": "invalid_incomplete",
        "conditional_muscle_gain_proposal": False,
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
        "schema": "flylab.m3.e147.muscle-gain-execution.v1",
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
                TimeoutError(f"E147 exceeded internal wall during {stage}"),
            ),
            (
                size is not None and size > freeze["maximum_raw_bytes"],
                "maximum_raw_bytes",
                RuntimeError(f"E147 exceeded raw byte limit during {stage}"),
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
            free >= freeze["minimum_free_disk_bytes"], "E147 insufficient free disk"
        )
        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(
                all(
                    int(item.get("num_threads", 1)) <= 1
                    for item in progress["thread_pools"]
                ),
                "E147 CPU thread pool exceeds one thread",
            )
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            saved = validate_saved_inputs(data, freeze, partial=partial)
            partial.update(
                filter_bank=saved["bank"],
                filter_tau_s=saved["tau"],
                baseline_pool=saved["pool"],
                membership=saved["membership"],
            )
            problem, pre_raw = build_conic_problem(saved, data, freeze)
            partial.update(pre_raw)
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
                data,
                freeze,
                progress,
                partial,
                prepared=(problem, pre_raw, saved),
            )
            try:
                save_npz_new(output / "muscle-gain-raw.npz", raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save muscle-gain-raw.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            raw_saved = True
            assessment = assess_physical_candidate(
                outcome["saved"],
                outcome["weights"],
                outcome["candidate_gain"],
                outcome["composite"],
                freeze,
            )
        proposal = bool(assessment["conditional_muscle_gain_proposal"])
        projected_bound = float(raw["dual_projected_margin_upper_bound"])
        verdict = (
            "conditional_muscle_gain_proposal"
            if proposal
            else "guarded_bounded_muscle_gain_refuted"
            if projected_bound < 0
            else "guarded_bounded_muscle_gain_inconclusive"
        )
        summary = {
            "schema": "flylab.m3.e147.muscle-gain-result.v1",
            "experiment": "M3-E147-muscle-gain",
            "status": "pass",
            "validity": "conditional_saved_E146_single_global_SOCP",
            "hypothesis_verdict": verdict,
            "analysis": facts,
            "assessment": assessment,
            "conditional_muscle_gain_proposal": proposal,
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
            "interpretation": {
                "scope": "one saved-data bounded muscle-gain SOCP",
                "numerical_upper_bound": "floating-point guarded-family diagnostic",
                "biology": "engineering activation-to-joint gain hypothesis only",
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
