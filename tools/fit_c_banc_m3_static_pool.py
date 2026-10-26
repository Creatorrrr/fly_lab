"""E144 bounded CPU fit of one static, normalized MN pool per muscle.

The program reads only frozen E143/E103 arrays.  It performs at most one
analytic-Jacobian SLSQP solve per leg and never runs neural, CUDA, body, EMA,
or LP work.  A successful process may still report that the bounded search
found no conditional proposal; that is not a proof of global impossibility.
"""

# ruff: noqa: BLE001  # preserve the first failure while saving partial evidence

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

import numpy as np  # thread limits must be set before NumPy import
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "f9c6920d94c87dd9b384cec8359f2c0264a9231f7ad5d801b5f8fdf0e6d22000"
DESIGN_SHA = "ce7214626bf5b15b0d599f588285c25410080778197e60ac9f4b9cae8a0754b5"
SCHEMA = "flylab.m3.e144.static-pool-feasibility.v1"
OWNED = ("tools/fit_c_banc_m3_static_pool.py", "tests/test_c_banc_static_pool.py")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
DT = 0.005
COMMAND_GAIN = 0.02
DEFAULT_FREEZE = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e144-static-pool-freeze.json"
)
DEFAULT_OUTPUT = (
    ROOT / "verification/banc-natural-gait-m3-20260919/e144-static-pool-fit"
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
        f"Secondary E144 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E144_STATIC_POOL_FEASIBILITY_DESIGN_20260921.md",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "cpu_threads": 1,
        "gpu_count": 0,
        "original_processes": 1,
        "retries": 0,
        "max_optimizer_calls": 6,
        "max_calls_per_leg": 140,
        "max_total_calls": 840,
        "leg_soft_limit_s": 9,
        "total_fit_limit_s": 54,
        "internal_wall_limit_s": 70,
        "external_wall_limit_s": 75,
        "supervisor_limit_s": 80,
        "prior_conservative_process_s": 2170,
        "prior_conservative_actual_s": 2169.4090214,
        "prospective_cumulative_limit_s": 2250,
        "original_program_cumulative_limit_s": 2271,
        "maximum_raw_bytes": 67108864,
        "minimum_free_disk_bytes": 1073741824,
        "original_neural_steps": 0,
        "physics_steps": 0,
        "recorded_EMA_updates": 0,
        "LP_calls": 0,
        "solver": "SLSQP",
        "solver_options": {"maxiter": 100, "ftol": 1e-10, "disp": False},
        "leg_order": list(LEGS),
        "pool_floor_fraction": 1e-6,
        "row_sum_atol": 1e-12,
        "lower_bound_atol": 1e-12,
        "startup_optimizer_margin": 1e-9,
        "relative_pool_prior": 1e-6,
        "objective_scale_floor": 1e-12,
        "startup_limits": {
            "q_RMS": 0.018,
            "q_max_abs": 0.04,
            "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
        "proposal_command_RMS_ratio": 0.8,
        "proposal_velocity_RMS_ratio": 1.0,
        "per_leg_command_and_velocity_nonincrease": True,
        "fixed_LTM_rows": True,
        "fixed_singleton_rows": True,
        "candidate_adopted": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "whole_goal_complete": False,
    }


def reviewed_inputs(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete source gate before opening numerical input arrays."""
    require(freeze_sha == FREEZE_SHA, "Exact E144 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E144 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E144 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E144 design changed")
    for mapping_name in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(mapping_name)
        require(
            isinstance(mapping, dict) and mapping,
            f"Missing frozen mapping: {mapping_name}",
        )
        for relative, digest in mapping.items():
            checked_path(relative, digest)

    require(sha256(review_path) == review_sha, "E144 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_static_pool_fit") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E144 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    require(array.shape == shape, f"E144 {name} shape mismatch: {array.shape}")
    require(np.issubdtype(array.dtype, np.number), f"E144 {name} is not numeric")
    require(np.isfinite(array).all(), f"E144 {name} contains nonfinite values")
    return array.astype(np.float64, copy=False)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def load_inputs(freeze: dict[str, Any]) -> dict[str, Any]:
    paths = freeze["input_sha256"]
    raw_path = next(
        key for key in paths if key.endswith("e143-pool-capacity/pool-capacity-raw.npz")
    )
    startup_path = next(
        key
        for key in paths
        if key.endswith("e103-startup-preserving-recruitment/startup-inputs.npz")
    )
    anatomy_path = next(
        key
        for key in paths
        if key.endswith("e142-cached-native-body/muscle-provenance.json")
    )
    cal_path = next(
        key
        for key in paths
        if key.endswith("e142-cached-native-body/assembled-calibration.npz")
    )
    return {
        "e143": _load_npz(ROOT / raw_path),
        "startup": _load_npz(ROOT / startup_path),
        "anatomy": json.loads((ROOT / anatomy_path).read_text(encoding="utf-8")),
        "calibration": _load_npz(ROOT / cal_path),
    }


def validate_pool_contract(
    pool: Any, membership: Any, ltm: Any, *, row_sum_atol: float = 1e-12
) -> None:
    pool = finite_array(pool, (96, 391), "pool")
    membership = np.asarray(membership)
    ltm = np.asarray(ltm)
    require(
        membership.shape == (391,) and np.issubdtype(membership.dtype, np.integer),
        "E144 membership shape/type mismatch",
    )
    require(
        np.all((membership >= 0) & (membership < 96)), "E144 membership out of range"
    )
    require(
        ltm.shape == (6,)
        and np.issubdtype(ltm.dtype, np.integer)
        and len(np.unique(ltm)) == 6
        and np.all((ltm >= 0) & (ltm < 96)),
        "E144 LTM indices invalid",
    )
    require(np.all(pool >= 0), "E144 pool contains negative weights")
    require(
        np.allclose(pool.sum(axis=1), 1, atol=row_sum_atol, rtol=0),
        "E144 pool rows are not normalized",
    )
    require(
        np.array_equal(np.sum(pool > 0, axis=0), np.ones(391, dtype=int))
        and np.array_equal(np.argmax(pool, axis=0), membership),
        "E144 pool support differs from membership",
    )


def validate_saved_inputs(
    data: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    """Validate frozen schemas and exact affine reconstruction without solving."""
    raw, startup, cal = data["e143"], data["startup"], data["calibration"]
    b = finite_array(raw["filtered_motor_rate"], (2000, 391), "saved filtered b")
    pool = finite_array(raw["pool"], (96, 391), "saved pool")
    membership = np.asarray(raw["membership"])
    moment = finite_array(raw["moment"], (42, 96), "saved moment")
    rest = finite_array(raw["rest"], (42,), "saved rest")
    ltm = np.asarray(raw["ltm"])
    validate_pool_contract(pool, membership, ltm, row_sum_atol=freeze["row_sum_atol"])
    for key in ("pool", "membership", "moment", "rest", "ltm"):
        require(
            np.array_equal(raw[key], cal[key]),
            f"E144 calibration differs from E143 raw: {key}",
        )
    baseline = finite_array(
        raw["baseline_endpoint_rad"], (2000, 42), "baseline command"
    )
    activation = b @ pool.T
    reconstructed = rest + COMMAND_GAIN * (activation @ moment.T)
    require(
        np.max(np.abs(reconstructed - baseline)) <= 1e-12,
        "E144 baseline affine reconstruction differs",
    )
    indices = np.asarray(raw["target_indices"])
    require(
        indices.shape == (1827,)
        and np.issubdtype(indices.dtype, np.integer)
        and np.all(indices >= 1)
        and np.all(np.diff(indices) == 1),
        "E144 target indices changed or are nonadjacent",
    )
    target = finite_array(raw["target_rad"], (1827, 42), "target command")
    pair_start = np.asarray(raw["velocity_pair_start_indices"])
    pair_end = np.asarray(raw["velocity_pair_end_indices"])
    require(
        np.array_equal(pair_start, indices[:-1])
        and np.array_equal(pair_end, indices[1:]),
        "E144 saved velocity pairs differ from target adjacency",
    )
    target_velocity = finite_array(
        raw["target_velocity_rad_s"], (1826, 42), "target velocity"
    )
    require(
        np.max(np.abs(target_velocity - np.diff(target, axis=0) / DT)) <= 1e-12,
        "E144 target velocity indexing differs",
    )
    startup_mask = np.asarray(startup["mask"])
    require(
        startup_mask.shape == (2001,)
        and startup_mask.dtype == np.bool_
        and np.array_equal(np.flatnonzero(startup_mask), np.arange(1, 100)),
        "E144 startup mask is not exact grid rows 1..99",
    )
    startup_target = finite_array(startup["target_command"], (99, 42), "startup target")
    startup_tendon = finite_array(
        startup["target_tendon"], (99, 6), "startup tendon target"
    )
    return {
        "b": b,
        "pool": pool,
        "membership": membership.astype(np.int64, copy=False),
        "moment": moment,
        "rest": rest,
        "ltm": ltm.astype(np.int64, copy=False),
        "baseline": baseline,
        "target_indices": indices.astype(np.int64, copy=False),
        "target": target,
        "pair_start": pair_start.astype(np.int64, copy=False),
        "pair_end": pair_end.astype(np.int64, copy=False),
        "target_velocity": target_velocity,
        "startup_target": startup_target,
        "startup_tendon_target": startup_tendon,
    }


@dataclass(frozen=True)
class LegLayout:
    leg: str
    leg_index: int
    joint_columns: np.ndarray
    muscles: np.ndarray
    variable_muscles: np.ndarray
    variable_columns: np.ndarray
    variable_column_muscles: np.ndarray
    x0: np.ndarray
    lower: np.ndarray
    row_constraint: np.ndarray


def build_leg_layout(
    leg: str,
    pool: Any,
    membership: Any,
    ltm: Any,
    anatomy: dict[str, Any],
    moment: Any,
    floor_fraction: float = 1e-6,
) -> LegLayout:
    """Return only non-LTM, non-singleton variables for one anatomical leg."""
    require(leg in LEGS, f"Unknown E144 leg: {leg}")
    leg_index = LEGS.index(leg)
    pool = np.asarray(pool, dtype=np.float64)
    membership = np.asarray(membership)
    ltm_set = {int(value) for value in np.asarray(ltm)}
    moment = np.asarray(moment, dtype=np.float64)
    muscles_meta = anatomy.get("muscles")
    require(
        isinstance(muscles_meta, list) and len(muscles_meta) == 96,
        "E144 anatomy muscle list changed",
    )
    muscles = np.asarray(
        [index for index, item in enumerate(muscles_meta) if item.get("leg") == leg]
    )
    require(muscles.size > 0, f"E144 no anatomy muscles for {leg}")
    for muscle in muscles:
        expected = np.flatnonzero(membership == muscle)
        observed = np.asarray(
            muscles_meta[int(muscle)].get("columns", []), dtype=np.int64
        )
        require(
            np.array_equal(observed, expected),
            f"E144 anatomy columns differ for muscle {muscle}",
        )
    joints = np.arange(7 * leg_index, 7 * (leg_index + 1), dtype=np.int64)
    outside = np.setdiff1d(np.arange(42), joints)
    require(
        np.count_nonzero(moment[np.ix_(outside, muscles)]) == 0,
        f"E144 {leg} muscles have cross-leg moment support",
    )
    variable_muscles = np.asarray(
        [
            m
            for m in muscles
            if int(m) not in ltm_set and np.count_nonzero(membership == m) > 1
        ],
        dtype=np.int64,
    )
    columns = np.concatenate(
        [np.flatnonzero(membership == m) for m in variable_muscles]
    )
    groups = membership[columns].astype(np.int64, copy=False)
    x0 = pool[groups, columns]
    lower = np.empty(columns.size, dtype=np.float64)
    row_constraint = np.zeros((variable_muscles.size, columns.size), dtype=np.float64)
    for row, muscle in enumerate(variable_muscles):
        local = np.flatnonzero(groups == muscle)
        lower[local] = floor_fraction / local.size
        row_constraint[row, local] = 1.0
    require(np.all(x0 > 0), f"E144 {leg} original support is not strictly positive")
    require(
        np.max(np.abs(row_constraint @ x0 - 1)) <= 1e-12,
        f"E144 {leg} x0 row sum differs",
    )
    return LegLayout(
        leg,
        leg_index,
        joints,
        muscles,
        variable_muscles,
        columns,
        groups,
        x0.astype(np.float64, copy=True),
        lower,
        row_constraint,
    )


def build_affine_design(
    b: Any,
    baseline: Any,
    moment: Any,
    layout: LegLayout,
    rows: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build q = D @ x + c with shape D=(time,joint,variable)."""
    b = np.asarray(b, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    coefficients = np.asarray(moment)[
        np.ix_(layout.joint_columns, layout.variable_column_muscles)
    ]
    design = (
        COMMAND_GAIN
        * b[np.ix_(rows, layout.variable_columns)][:, None, :]
        * coefficients[None, :, :]
    )
    offset = baseline[np.ix_(rows, layout.joint_columns)] - np.einsum(
        "tjn,n->tj", design, layout.x0
    )
    return design, offset


def velocity_affine_design(
    endpoint_design: Any, endpoint_offset: Any, velocity_rows: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Difference actual adjacent endpoint affine rows, then select pair_end-2."""
    design = np.asarray(endpoint_design, dtype=np.float64)
    offset = np.asarray(endpoint_offset, dtype=np.float64)
    velocity_rows = np.asarray(velocity_rows, dtype=np.int64)
    require(
        design.ndim == 3
        and offset.shape == design.shape[:2]
        and np.all((velocity_rows >= 0) & (velocity_rows < design.shape[0] - 1)),
        "E144 velocity affine shape/index mismatch",
    )
    return (
        (design[velocity_rows + 1] - design[velocity_rows]) / DT,
        (offset[velocity_rows + 1] - offset[velocity_rows]) / DT,
    )


@dataclass(frozen=True)
class QuadraticObjective:
    hessian: np.ndarray
    linear: np.ndarray
    constant: float


def build_quadratic_objective(
    command_design: Any,
    command_offset: Any,
    command_target: Any,
    velocity_design: Any,
    velocity_offset: Any,
    velocity_target: Any,
    x0: Any,
    baseline_command_mse: float,
    baseline_velocity_mse: float,
    *,
    scale_floor: float = 1e-12,
    prior_weight: float = 1e-6,
) -> QuadraticObjective:
    """Precompute the exact normalized least-squares Gram and linear terms."""
    x0 = np.asarray(x0, dtype=np.float64)
    matrices = (
        (
            np.asarray(command_design).reshape(-1, x0.size),
            (np.asarray(command_offset) - np.asarray(command_target)).reshape(-1),
            max(float(baseline_command_mse), scale_floor),
        ),
        (
            np.asarray(velocity_design).reshape(-1, x0.size),
            (np.asarray(velocity_offset) - np.asarray(velocity_target)).reshape(-1),
            max(float(baseline_velocity_mse), scale_floor),
        ),
    )
    hessian = np.zeros((x0.size, x0.size), dtype=np.float64)
    linear = np.zeros(x0.size, dtype=np.float64)
    constant = 0.0
    for matrix, residual, scale in matrices:
        factor = 1.0 / (residual.size * scale)
        hessian += 2 * factor * (matrix.T @ matrix)
        linear += 2 * factor * (matrix.T @ residual)
        constant += factor * float(residual @ residual)
    prior_factor = prior_weight / x0.size
    inverse_square = 1 / np.square(x0)
    hessian += 2 * prior_factor * np.diag(inverse_square)
    linear += -2 * prior_factor / x0
    constant += prior_weight
    return QuadraticObjective(hessian, linear, constant)


def quadratic_value_grad(model: QuadraticObjective, x: Any) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    gradient = model.hessian @ x + model.linear
    value = (
        0.5 * float(x @ model.hessian @ x) + float(model.linear @ x) + model.constant
    )
    return value, gradient


def startup_rms_constraint_value_jac(
    design: Any, offset: Any, target: Any, x: Any, budget: float
) -> tuple[float, np.ndarray]:
    """Return budget^2 - mean(error^2) and its analytic Jacobian."""
    design = np.asarray(design, dtype=np.float64)
    error = (
        np.einsum("tjn,n->tj", design, np.asarray(x))
        + np.asarray(offset)
        - np.asarray(target)
    )
    value = float(budget * budget - np.mean(np.square(error)))
    jacobian = -2 * np.einsum("tjn,tj->n", design, error) / error.size
    return value, jacobian


def proportional_startup_budgets(
    baseline_error: Any, global_limit: float, *, zero_atol: float = 0.0
) -> np.ndarray:
    """Allocate per-leg RMS bounds in proportion to baseline leg RMS."""
    error = np.asarray(baseline_error, dtype=np.float64)
    require(
        error.ndim == 2 and error.shape[1] == 42, "E144 startup error shape mismatch"
    )
    total = float(np.sqrt(np.mean(np.square(error))))
    by_leg = np.asarray(
        [np.sqrt(np.mean(np.square(error[:, 7 * i : 7 * (i + 1)]))) for i in range(6)]
    )
    if total <= zero_atol:
        return np.zeros(6, dtype=np.float64)
    return by_leg * (float(global_limit) / total)


@dataclass(frozen=True)
class LegProblem:
    layout: LegLayout
    command_design: np.ndarray
    command_offset: np.ndarray
    command_target: np.ndarray
    velocity_design: np.ndarray
    velocity_offset: np.ndarray
    velocity_target: np.ndarray
    startup_design: np.ndarray
    startup_offset: np.ndarray
    startup_target: np.ndarray
    objective: QuadraticObjective
    startup_rms_budget: float
    startup_optimizer_max_limit: float
    startup_selector_max_limit: float
    row_sum_atol: float
    lower_bound_atol: float


def evaluate_leg_vector(problem: LegProblem, x: Any) -> dict[str, Any]:
    """Evaluate one vector without clipping, normalization, or mutation."""
    x = np.asarray(x, dtype=np.float64)
    require(x.shape == problem.layout.x0.shape, "E144 leg vector shape mismatch")
    objective, objective_jac = quadratic_value_grad(problem.objective, x)
    command = np.einsum("tjn,n->tj", problem.command_design, x) + problem.command_offset
    velocity = (
        np.einsum("tjn,n->tj", problem.velocity_design, x) + problem.velocity_offset
    )
    startup = np.einsum("tjn,n->tj", problem.startup_design, x) + problem.startup_offset
    command_error = command - problem.command_target
    velocity_error = velocity - problem.velocity_target
    startup_error = startup - problem.startup_target
    row_error = problem.layout.row_constraint @ x - 1
    lower_margin = x - problem.layout.lower
    upper_margin = 1 - x
    startup_rms = float(np.sqrt(np.mean(np.square(startup_error))))
    startup_max = float(np.max(np.abs(startup_error)))
    feasible = bool(
        np.isfinite(x).all()
        and np.isfinite(objective)
        and np.all(x > 0)
        and float(np.min(lower_margin)) >= -problem.lower_bound_atol
        and float(np.min(upper_margin)) >= 0
        and float(np.max(np.abs(row_error))) <= problem.row_sum_atol
        and startup_rms <= problem.startup_rms_budget
        and startup_max <= problem.startup_selector_max_limit
    )
    rms_value, rms_jac = startup_rms_constraint_value_jac(
        problem.startup_design,
        problem.startup_offset,
        problem.startup_target,
        x,
        problem.startup_rms_budget,
    )
    return {
        "x": x.copy(),
        "objective": float(objective),
        "objective_jac": objective_jac,
        "command_rms": float(np.sqrt(np.mean(np.square(command_error)))),
        "velocity_rms": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "startup_rms": startup_rms,
        "startup_max_abs": startup_max,
        "row_sum_max_abs": float(np.max(np.abs(row_error))),
        "lower_margin": float(np.min(lower_margin)),
        "upper_margin": float(np.min(upper_margin)),
        "startup_rms_constraint": rms_value,
        "startup_rms_jac": rms_jac,
        "startup_linear_constraints": np.concatenate(
            (
                problem.startup_optimizer_max_limit - startup_error.reshape(-1),
                problem.startup_optimizer_max_limit + startup_error.reshape(-1),
            )
        ),
        "feasible": feasible,
    }


class EvaluationLimit(RuntimeError):
    pass


class EvaluationLedger:
    """Cache exact vectors while separately counting objective requests."""

    def __init__(
        self,
        problem: LegProblem,
        *,
        max_calls: int,
        deadline: float,
        total_counter: list[int],
        total_limit: int,
    ) -> None:
        self.problem = problem
        self.max_calls = int(max_calls)
        self.deadline = float(deadline)
        self.total_counter = total_counter
        self.total_limit = int(total_limit)
        self.records: list[dict[str, Any]] = []
        self.interrupted_attempts: list[dict[str, Any]] = []
        self._cache: dict[bytes, int] = {}
        self.objective_calls = 0
        self.objective_calls_requested = 0
        self.started = time.perf_counter()

    def _record_interruption(
        self, vector: np.ndarray, origin: str, message: str
    ) -> None:
        self.interrupted_attempts.append(
            {
                "x": vector.copy(),
                "origin": origin,
                "reason": message,
                "elapsed_s": time.perf_counter() - self.started,
            }
        )

    def _guard_deadline(self, vector: np.ndarray, origin: str) -> None:
        if time.perf_counter() >= self.deadline:
            message = f"E144 {self.problem.layout.leg} soft deadline reached"
            self._record_interruption(vector, origin, message)
            raise EvaluationLimit(message)

    def _guard_new_evaluation(self, vector: np.ndarray, origin: str) -> None:
        if (
            len(self.records) >= self.max_calls
            or self.total_counter[0] >= self.total_limit
        ):
            message = f"E144 {self.problem.layout.leg} evaluation cap reached"
            self._record_interruption(vector, origin, message)
            raise EvaluationLimit(message)

    def get(self, x: Any, *, origin: str = "solver") -> dict[str, Any]:
        vector = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        require(
            vector.shape == self.problem.layout.x0.shape,
            "E144 cached vector shape mismatch",
        )
        # Time is a work limit, so cached reads stop after it too.  Count caps
        # apply only when a new analytic evaluation would be performed.
        self._guard_deadline(vector, origin)
        key = vector.tobytes()
        if key not in self._cache:
            self._guard_new_evaluation(vector, origin)
            try:
                record = evaluate_leg_vector(self.problem, vector)
            except BaseException as error:
                self._record_interruption(
                    vector,
                    origin,
                    f"evaluation_error:{type(error).__name__}:{error}",
                )
                raise
            record["origin"] = origin
            record["elapsed_s"] = time.perf_counter() - self.started
            self._cache[key] = len(self.records)
            self.records.append(record)
            self.total_counter[0] += 1
        return self.records[self._cache[key]]

    def fun(self, x: Any) -> float:
        self.objective_calls_requested += 1
        vector = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        self._guard_deadline(vector, "solver_objective")
        if self.objective_calls >= self.max_calls:
            message = f"E144 {self.problem.layout.leg} objective-call cap reached"
            self._record_interruption(vector, "solver_objective", message)
            raise EvaluationLimit(message)
        self.objective_calls += 1
        return float(self.get(vector, origin="solver_objective")["objective"])

    def jac(self, x: Any) -> np.ndarray:
        return np.asarray(self.get(x, origin="solver_jac")["objective_jac"])


def select_earliest_feasible(records: list[dict[str, Any]]) -> int | None:
    """Return the zero-based earliest strict-feasible minimum objective."""
    selected: int | None = None
    best = np.inf
    for index, record in enumerate(records):
        value = float(record["objective"])
        if bool(record["feasible"]) and np.isfinite(value) and value < best:
            selected, best = index, value
    return selected


def solve_leg(
    problem: LegProblem,
    freeze: dict[str, Any],
    deadline: float,
    total_counter: list[int],
    *,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> tuple[EvaluationLedger, dict[str, Any]]:
    ledger = EvaluationLedger(
        problem,
        max_calls=freeze["max_calls_per_leg"],
        deadline=deadline,
        total_counter=total_counter,
        total_limit=freeze["max_total_calls"],
    )
    state: dict[str, Any] = {
        "preparation_completed": True,
        "baseline_attempted": True,
        "baseline_completed": False,
        "optimizer_attempted": False,
        "optimizer_entered": False,
        "optimizer_completed": False,
        "optimizer_bounded_interruption": False,
        "phase": "baseline",
    }

    def publish(result: dict[str, Any] | None = None) -> None:
        if partial is not None:
            publish_leg_partial(partial, problem, ledger, state, result)
        if progress is not None:
            leg = problem.layout.leg
            progress.setdefault("leg_unique_evaluations", {})[leg] = len(ledger.records)
            progress.setdefault("leg_objective_calls", {})[leg] = ledger.objective_calls
            progress.setdefault("leg_objective_calls_requested", {})[leg] = (
                ledger.objective_calls_requested
            )
            progress.setdefault("leg_interrupted_attempts", {})[leg] = len(
                ledger.interrupted_attempts
            )
            progress.setdefault("leg_optimizer_phase", {})[leg] = state["phase"]

    publish()
    try:
        ledger.get(problem.layout.x0, origin="baseline")
    except BaseException as error:
        state["phase"] = "baseline_error"
        state["error_type"] = type(error).__name__
        state["error_message"] = str(error)
        publish()
        raise
    state["baseline_completed"] = True
    state["phase"] = "baseline_completed"
    publish()
    constraints = (
        {
            "type": "eq",
            "fun": lambda x: problem.layout.row_constraint @ x - 1,
            "jac": lambda _x: problem.layout.row_constraint,
        },
        {
            "type": "ineq",
            "fun": lambda x: ledger.get(x)["startup_linear_constraints"],
            "jac": lambda _x: np.concatenate(
                (
                    -problem.startup_design.reshape(-1, problem.layout.x0.size),
                    problem.startup_design.reshape(-1, problem.layout.x0.size),
                ),
                axis=0,
            ),
        },
        {
            "type": "ineq",
            "fun": lambda x: ledger.get(x)["startup_rms_constraint"],
            "jac": lambda x: ledger.get(x)["startup_rms_jac"],
        },
    )
    result: dict[str, Any]
    state["optimizer_attempted"] = True
    state["optimizer_entered"] = True
    state["phase"] = "optimizer_entered"
    if progress is not None:
        progress["fit_calls"] = int(progress.get("fit_calls", 0)) + 1
    publish()
    try:
        optimization = minimize(
            ledger.fun,
            problem.layout.x0.copy(),
            method="SLSQP",
            jac=ledger.jac,
            bounds=list(zip(problem.layout.lower, np.ones(problem.layout.x0.size))),
            constraints=constraints,
            options=dict(freeze["solver_options"]),
        )
        state["optimizer_completed"] = True
        state["phase"] = "optimizer_returned"
        result = {
            "returned": True,
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "nit": int(optimization.nit),
            "nfev": int(optimization.nfev),
            "njev": int(optimization.njev),
        }
    except EvaluationLimit as error:
        state["optimizer_bounded_interruption"] = True
        state["phase"] = "optimizer_bounded_interruption"
        result = {
            "returned": False,
            "success": False,
            "status": None,
            "message": str(error),
            "interrupted": type(error).__name__,
        }
    except BaseException as error:
        state["phase"] = "optimizer_error"
        state["error_type"] = type(error).__name__
        state["error_message"] = str(error)
        publish()
        raise
    result["objective_calls"] = ledger.objective_calls
    result["objective_calls_requested"] = ledger.objective_calls_requested
    result["unique_evaluations"] = len(ledger.records)
    result["interrupted_attempts"] = len(ledger.interrupted_attempts)
    result["selected_index"] = select_earliest_feasible(ledger.records)
    publish(result)
    return ledger, result


def apply_selected_weights(
    baseline_pool: Any,
    selections: list[tuple[LegLayout, np.ndarray] | None],
) -> np.ndarray | None:
    """Compose the exact selected vectors; never clip or renormalize."""
    if any(item is None for item in selections):
        return None
    candidate = np.asarray(baseline_pool, dtype=np.float64).copy()
    for item in selections:
        assert item is not None
        layout, vector = item
        candidate[layout.variable_column_muscles, layout.variable_columns] = vector
    return candidate


def _rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def evaluate_composite(
    saved: dict[str, Any], candidate_pool: np.ndarray | None
) -> dict[str, Any]:
    """Compute baseline/candidate arrays and scalar diagnostics before verdict checks."""
    b, pool, moment, rest, ltm = (
        saved["b"],
        saved["pool"],
        saved["moment"],
        saved["rest"],
        saved["ltm"],
    )
    endpoint_rows = saved["target_indices"] - 1
    velocity_rows = saved["pair_end"] - 2
    pools = {"baseline": pool}
    if candidate_pool is not None:
        pools["candidate"] = candidate_pool
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, current_pool in pools.items():
        activation = b @ current_pool.T
        command = rest + COMMAND_GAIN * (activation @ moment.T)
        selected = command[endpoint_rows]
        velocity = np.diff(command, axis=0)[velocity_rows] / DT
        startup = command[:99]
        tendon = 0.2 * np.tanh(activation[:99, ltm] / 10)
        arrays.update(
            {
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


def assess_composite(
    saved: dict[str, Any],
    candidate_pool: np.ndarray | None,
    composite: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen strict proposal gates to already-computed raw values."""
    if candidate_pool is None:
        return {
            "conditional_static_pool_proposal": False,
            "reason": "at_least_one_leg_has_no_evaluated_feasible_vector",
            "bounded_failure_is_global_impossibility": False,
        }
    validate_pool_contract(
        candidate_pool,
        saved["membership"],
        saved["ltm"],
        row_sum_atol=freeze["row_sum_atol"],
    )
    fixed = {int(x) for x in saved["ltm"]}
    fixed.update(
        muscle
        for muscle in range(96)
        if np.count_nonzero(saved["membership"] == muscle) == 1
    )
    fixed_rows_exact = all(
        np.array_equal(candidate_pool[m], saved["pool"][m]) for m in fixed
    )
    candidate = composite["metrics"]["candidate"]
    baseline = composite["metrics"]["baseline"]
    tendon_exact = np.array_equal(
        composite["arrays"]["candidate_startup_tendon"],
        composite["arrays"]["baseline_startup_tendon"],
    )
    limits = freeze["startup_limits"]
    gates = {
        "anatomical_pool_valid": True,
        "fixed_LTM_and_singleton_rows_exact": fixed_rows_exact,
        "fixed_tendon_output_exact": tendon_exact,
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
        "conditional_static_pool_proposal": all(gates.values()),
        "gates": gates,
        "bounded_failure_is_global_impossibility": False,
    }


def composite_metrics_to_raw(metrics: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten scalar diagnostics so they are durable before proposal checks."""
    raw: dict[str, np.ndarray] = {}
    scalar_names = (
        "command_RMS",
        "velocity_RMS",
        "startup_q_RMS",
        "startup_q_max_abs",
        "startup_LTM_RMS",
        "startup_LTM_max_abs",
    )
    for variant, values in metrics.items():
        for name in scalar_names:
            raw[f"{variant}_{name}"] = np.asarray(values[name])
        for leg, leg_values in values["by_leg"].items():
            raw[f"{variant}_{leg}_command_RMS"] = np.asarray(leg_values["command_RMS"])
            raw[f"{variant}_{leg}_velocity_RMS"] = np.asarray(
                leg_values["velocity_RMS"]
            )
    return raw


def _problem_for_leg(
    saved: dict[str, Any],
    data: dict[str, Any],
    freeze: dict[str, Any],
    leg: str,
    budget: float,
) -> LegProblem:
    layout = build_leg_layout(
        leg,
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        data["anatomy"],
        saved["moment"],
        freeze["pool_floor_fraction"],
    )
    all_rows = np.arange(2000)
    full_design, full_offset = build_affine_design(
        saved["b"], saved["baseline"], saved["moment"], layout, all_rows
    )
    endpoint_rows = saved["target_indices"] - 1
    command_design = full_design[endpoint_rows]
    command_offset = full_offset[endpoint_rows]
    command_target = saved["target"][:, layout.joint_columns]
    velocity_design, velocity_offset = velocity_affine_design(
        full_design, full_offset, saved["pair_end"] - 2
    )
    velocity_target = saved["target_velocity"][:, layout.joint_columns]
    startup_design = full_design[:99]
    startup_offset = full_offset[:99]
    startup_target = saved["startup_target"][:, layout.joint_columns]
    baseline_command = saved["baseline"][endpoint_rows][:, layout.joint_columns]
    baseline_velocity = np.diff(saved["baseline"], axis=0)[saved["pair_end"] - 2]
    baseline_velocity = baseline_velocity[:, layout.joint_columns] / DT
    command_mse = float(np.mean(np.square(baseline_command - command_target)))
    velocity_mse = float(np.mean(np.square(baseline_velocity - velocity_target)))
    objective = build_quadratic_objective(
        command_design,
        command_offset,
        command_target,
        velocity_design,
        velocity_offset,
        velocity_target,
        layout.x0,
        command_mse,
        velocity_mse,
        scale_floor=freeze["objective_scale_floor"],
        prior_weight=freeze["relative_pool_prior"],
    )
    return LegProblem(
        layout,
        command_design,
        command_offset,
        command_target,
        velocity_design,
        velocity_offset,
        velocity_target,
        startup_design,
        startup_offset,
        startup_target,
        objective,
        float(budget),
        freeze["startup_limits"]["q_max_abs"] - freeze["startup_optimizer_margin"],
        freeze["startup_limits"]["q_max_abs"],
        freeze["row_sum_atol"],
        freeze["lower_bound_atol"],
    )


def _records_to_arrays(
    prefix: str, records: list[dict[str, Any]], width: int | None = None
) -> dict[str, np.ndarray]:
    fields = (
        "objective",
        "command_rms",
        "velocity_rms",
        "startup_rms",
        "startup_max_abs",
        "row_sum_max_abs",
        "lower_margin",
        "upper_margin",
        "startup_rms_constraint",
    )
    if records:
        vectors = np.stack([row["x"] for row in records])
    else:
        require(width is not None, "E144 empty evaluation history requires width")
        vectors = np.empty((0, width), dtype=np.float64)
    result = {f"{prefix}_evaluation_x": vectors}
    for field in fields:
        result[f"{prefix}_evaluation_{field}"] = np.asarray(
            [row[field] for row in records]
        )
    result[f"{prefix}_evaluation_feasible"] = np.asarray(
        [row["feasible"] for row in records]
    )
    result[f"{prefix}_evaluation_origin"] = np.asarray(
        [row["origin"] for row in records]
    )
    result[f"{prefix}_evaluation_elapsed_s"] = np.asarray(
        [row["elapsed_s"] for row in records]
    )
    return result


def _interruptions_to_arrays(
    prefix: str, attempts: list[dict[str, Any]], width: int
) -> dict[str, np.ndarray]:
    if attempts:
        vectors = np.stack([row["x"] for row in attempts])
    else:
        vectors = np.empty((0, width), dtype=np.float64)
    return {
        f"{prefix}_interrupted_x": vectors,
        f"{prefix}_interrupted_origin": np.asarray([row["origin"] for row in attempts]),
        f"{prefix}_interrupted_reason": np.asarray([row["reason"] for row in attempts]),
        f"{prefix}_interrupted_elapsed_s": np.asarray(
            [row["elapsed_s"] for row in attempts]
        ),
    }


def publish_leg_partial(
    target: dict[str, Any],
    problem: LegProblem,
    ledger: EvaluationLedger,
    state: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> None:
    """Copy one active leg's durable history into the caller-owned raw record."""
    leg = problem.layout.leg
    target.update(
        {
            f"{leg}_variable_columns": problem.layout.variable_columns,
            f"{leg}_variable_muscles": problem.layout.variable_column_muscles,
            f"{leg}_x0": problem.layout.x0,
            f"{leg}_lower_bounds": problem.layout.lower,
            f"{leg}_row_constraint": problem.layout.row_constraint,
            f"{leg}_startup_rms_budget": np.asarray(problem.startup_rms_budget),
            f"{leg}_preparation_completed": np.asarray(state["preparation_completed"]),
            f"{leg}_baseline_attempted": np.asarray(state["baseline_attempted"]),
            f"{leg}_baseline_completed": np.asarray(state["baseline_completed"]),
            f"{leg}_optimizer_attempted": np.asarray(state["optimizer_attempted"]),
            f"{leg}_optimizer_entered": np.asarray(state["optimizer_entered"]),
            f"{leg}_optimizer_completed": np.asarray(state["optimizer_completed"]),
            f"{leg}_optimizer_bounded_interruption": np.asarray(
                state["optimizer_bounded_interruption"]
            ),
            f"{leg}_optimizer_phase": np.asarray(state["phase"]),
            f"{leg}_unique_evaluations": np.asarray(len(ledger.records)),
            f"{leg}_objective_calls": np.asarray(ledger.objective_calls),
            f"{leg}_objective_calls_requested": np.asarray(
                ledger.objective_calls_requested
            ),
        }
    )
    if "error_type" in state:
        target[f"{leg}_error_type"] = np.asarray(state["error_type"])
        target[f"{leg}_error_message"] = np.asarray(state["error_message"])
    target.update(_records_to_arrays(leg, ledger.records, problem.layout.x0.size))
    target.update(
        _interruptions_to_arrays(
            leg, ledger.interrupted_attempts, problem.layout.x0.size
        )
    )
    if result is not None:
        selected = result["selected_index"]
        target[f"{leg}_selected_evaluation_index"] = np.asarray(
            -1 if selected is None else selected
        )
        target[f"{leg}_solver_returned"] = np.asarray(result["returned"])
        target[f"{leg}_solver_success"] = np.asarray(result["success"])


def calculate(
    data: dict[str, Any],
    freeze: dict[str, Any],
    progress: dict[str, Any],
    partial: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if partial is None:
        partial = {}
    # Preserve unvalidated frozen inputs before schema/numerical checks.
    partial.update(
        {
            "unvalidated_b": np.asarray(data["e143"].get("filtered_motor_rate", [])),
            "unvalidated_pool": np.asarray(data["e143"].get("pool", [])),
            "unvalidated_membership": np.asarray(data["e143"].get("membership", [])),
            "unvalidated_moment": np.asarray(data["e143"].get("moment", [])),
            "unvalidated_rest": np.asarray(data["e143"].get("rest", [])),
            "unvalidated_ltm": np.asarray(data["e143"].get("ltm", [])),
            "unvalidated_startup_target": np.asarray(
                data["startup"].get("target_command", [])
            ),
        }
    )
    saved = validate_saved_inputs(data, freeze)
    partial.update(
        {key: value for key, value in saved.items() if isinstance(value, np.ndarray)}
    )
    startup_error = saved["baseline"][:99] - saved["startup_target"]
    budgets = proportional_startup_budgets(
        startup_error,
        freeze["startup_limits"]["q_RMS"] - freeze["startup_optimizer_margin"],
    )
    raw: dict[str, Any] = {
        "leg_order": np.asarray(LEGS),
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
        "startup_rms_budgets": budgets,
    }
    partial.update(raw)
    fit_started = time.perf_counter()
    total_counter = [0]
    selections: list[tuple[LegLayout, np.ndarray] | None] = []
    leg_results: dict[str, Any] = {}
    progress["fit_calls"] = 0
    progress["leg_unique_evaluations"] = {}
    progress["leg_objective_calls"] = {}
    progress["leg_objective_calls_requested"] = {}
    progress["leg_interrupted_attempts"] = {}
    progress["leg_optimizer_phase"] = {}
    try:
        for leg_index, leg in enumerate(LEGS):
            if time.perf_counter() - fit_started >= freeze["total_fit_limit_s"]:
                raise TimeoutError("E144 total fit limit reached")
            partial[f"{leg}_preparation_attempted"] = np.asarray(True)
            partial[f"{leg}_preparation_completed"] = np.asarray(False)
            partial[f"{leg}_optimizer_attempted"] = np.asarray(False)
            partial[f"{leg}_optimizer_entered"] = np.asarray(False)
            partial[f"{leg}_optimizer_completed"] = np.asarray(False)
            partial[f"{leg}_optimizer_phase"] = np.asarray("preparation")
            problem = _problem_for_leg(saved, data, freeze, leg, budgets[leg_index])
            leg_started = time.perf_counter()
            deadline = min(
                leg_started + freeze["leg_soft_limit_s"],
                fit_started + freeze["total_fit_limit_s"],
            )
            ledger, result = solve_leg(
                problem,
                freeze,
                deadline,
                total_counter,
                partial=partial,
                progress=progress,
            )
            leg_wall = time.perf_counter() - leg_started
            result["wall_s"] = leg_wall
            selected_index = result["selected_index"]
            if selected_index is None:
                selections.append(None)
            else:
                selections.append(
                    (problem.layout, ledger.records[selected_index]["x"].copy())
                )
            leg_results[leg] = result
            publish_leg_partial(
                raw,
                problem,
                ledger,
                {
                    "preparation_completed": True,
                    "baseline_attempted": True,
                    "baseline_completed": True,
                    "optimizer_attempted": True,
                    "optimizer_entered": True,
                    "optimizer_completed": result["returned"],
                    "optimizer_bounded_interruption": not result["returned"],
                    "phase": (
                        "optimizer_returned"
                        if result["returned"]
                        else "optimizer_bounded_interruption"
                    ),
                },
                result,
            )
            # Caller-owned partial receives the completed raw block before any
            # post-solve wall check can turn the run into a resource failure.
            partial.update(raw)
            partial[f"{leg}_wall_s"] = np.asarray(leg_wall)
            if result["returned"] and leg_wall > freeze["leg_soft_limit_s"]:
                raise TimeoutError(f"E144 {leg} solver returned after leg soft limit")
            if time.perf_counter() - fit_started > freeze["total_fit_limit_s"]:
                raise TimeoutError("E144 total fit limit reached after completed solve")
    finally:
        fit_wall = time.perf_counter() - fit_started
        progress["timings_s"]["fit"] = fit_wall
        progress["unique_evaluations"] = total_counter[0]
        progress["objective_calls"] = sum(progress["leg_objective_calls"].values())
        progress["objective_calls_requested"] = sum(
            progress["leg_objective_calls_requested"].values()
        )
        progress["interrupted_attempts"] = sum(
            progress["leg_interrupted_attempts"].values()
        )
        partial["fit_wall_s"] = np.asarray(fit_wall)
        partial["unique_evaluations"] = np.asarray(total_counter[0])
        partial["objective_calls"] = np.asarray(progress["objective_calls"])
        partial["objective_calls_requested"] = np.asarray(
            progress["objective_calls_requested"]
        )
        partial["fit_calls"] = np.asarray(progress["fit_calls"])
    candidate_pool = apply_selected_weights(saved["pool"], selections)
    raw["candidate_pool"] = (
        np.empty((0, 391), dtype=np.float64)
        if candidate_pool is None
        else candidate_pool
    )
    raw["proposal_available"] = np.asarray(candidate_pool is not None)
    composite = evaluate_composite(saved, candidate_pool)
    raw.update(composite["arrays"])
    raw.update(composite_metrics_to_raw(composite["metrics"]))
    facts = {
        "legs": leg_results,
        "metrics": composite["metrics"],
        "optimizer_calls": progress["fit_calls"],
        "unique_evaluations": total_counter[0],
        "objective_calls": progress["objective_calls"],
        "objective_calls_requested": progress["objective_calls_requested"],
    }
    return (
        raw,
        facts,
        {"saved": saved, "candidate_pool": candidate_pool, "composite": composite},
    )


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e144.static-pool-result.v1",
        "experiment": "M3-E144-static-pool-feasibility",
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


def run(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "schema": "flylab.m3.e144.static-pool-execution.v1",
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
                TimeoutError(f"E144 exceeded internal wall limit during {stage}"),
            ),
            (
                size is not None and size > freeze["maximum_raw_bytes"],
                "maximum_raw_bytes",
                RuntimeError(f"E144 exceeded raw byte limit during {stage}"),
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
            "E144 insufficient starting free disk",
        )
        require(
            time.perf_counter() - started < freeze["internal_wall_limit_s"],
            "E144 source gate exhausted internal wall limit",
        )
        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(
                all(
                    int(item.get("num_threads", 1)) <= 1
                    for item in progress["thread_pools"]
                ),
                "E144 CPU thread pool exceeds one thread",
            )
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            raw, facts, outcome_inputs = calculate(data, freeze, progress, partial)
            partial.update(raw)
            stage = time.perf_counter()
            try:
                save_npz_new(output / "static-pool-raw.npz", raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save static-pool-raw.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            raw_saved = True
            progress["timings_s"]["raw_persistence"] = time.perf_counter() - stage
            # Final numerical outcome gates occur only after raw persistence.
            assessment = assess_composite(
                outcome_inputs["saved"],
                outcome_inputs["candidate_pool"],
                outcome_inputs["composite"],
                freeze,
            )
        progress["status"] = "pending_finalization"
        proposal = bool(assessment["conditional_static_pool_proposal"])
        summary = {
            "schema": "flylab.m3.e144.static-pool-result.v1",
            "experiment": "M3-E144-static-pool-feasibility",
            "status": "pass",
            "validity": "conditional_saved_E142_static_pool_fit",
            "hypothesis_verdict": (
                "conditional_static_pool_proposal"
                if proposal
                else "bounded_search_no_conditional_proposal"
            ),
            "analysis": facts,
            "assessment": assessment,
            "conditional_static_pool_proposal": proposal,
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
            "interpretation": {
                "scope": "one bounded static saved-data fit; no neural or body execution",
                "bounded_failure": "does not establish global mathematical impossibility",
                "biology": "proportional force allocation is an engineering interface hypothesis",
                "gait_claim": "natural gait improvement remains unverified",
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
                if primary is None:
                    capture_failure(error, "partial raw")
                else:
                    add_secondary(primary, "partial raw", error)
        check_resources("before_required_receipts")
        if primary is not None:
            progress["status"] = "failed"
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
        if (
            progress["failed_raw_operations"]
            or progress["failed_persistence_operations"]
        ):
            try:
                save_json_new(
                    output / "secondary-errors.json",
                    {
                        "raw": progress["failed_raw_operations"],
                        "persistence": progress["failed_persistence_operations"],
                        "resource": progress["failed_resource_checks"],
                    },
                )
            except BaseException as error:
                capture_failure(error, "save secondary-errors.json")
        check_resources("before_summary")
        if primary is not None:
            progress["status"] = "failed"
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
                save_json_new(
                    output / "summary.json", failure_summary(primary, progress)
                )
                summary = failure_summary(primary, progress)
            except BaseException as error:
                capture_failure(error, "invalidate authoritative success summary")
        if primary is not None:
            progress["status"] = "failed"
            finalization = {
                "status": "failed",
                "primary_failure": progress.get("primary_failure"),
                "failed_raw_operations": progress["failed_raw_operations"],
                "failed_persistence_operations": progress[
                    "failed_persistence_operations"
                ],
                "failed_resource_checks": progress["failed_resource_checks"],
                "wall_s_at_finalization_receipt": time.perf_counter() - started,
                "bytes_last_observed": progress.get("raw_bytes"),
            }
            try:
                save_json_new(output / "finalization-errors.json", finalization)
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
