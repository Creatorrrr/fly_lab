"""E164 fixed-bank onset recruitment SOCP and conditional proposal check."""

# ruff: noqa: BLE001
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
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fit_c_banc_m3_conic_pool as e145

FREEZE_SHA = "2431c31ff39374946dd4b894dc021465e29ac404cf8ef8a9cf35080cb3932884"
DESIGN_SHA = "786842e3bee117c064573b2856daeb41e474208bcb7ed0f28e2774a331a3119d"
SCHEMA = "flylab.m3.e164.onset-recruitment.v1"
OWNED = (
    "tools/fit_c_banc_m3_onset_recruitment.py",
    "tests/test_c_banc_onset_recruitment.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numerical_check(
    condition: bool,
    message: str,
    pending: list[tuple[bool, str]] | None,
) -> None:
    """Defer numerical gates when the caller must checkpoint their raw evidence."""
    if pending is None:
        require(condition, message)
    else:
        pending.append((bool(condition), message))


def save_check_evidence(
    path: Path,
    raw: dict[str, np.ndarray],
    stage: str,
    pending: list[tuple[bool, str]],
) -> None:
    raw[f"{stage}_check_names"] = np.asarray([name for _, name in pending], dtype=str)
    raw[f"{stage}_check_passed"] = np.asarray([ok for ok, _ in pending], dtype=bool)
    atomic_npz(path, raw)


def enforce_checks(pending: list[tuple[bool, str]]) -> None:
    for condition, message in pending:
        require(condition, message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative: str) -> Path:
    value = Path(relative)
    path = (ROOT / value).resolve()
    require(
        not value.is_absolute()
        and ".." not in value.parts
        and path.is_relative_to(ROOT),
        "E164 repository-relative path required",
    )
    return path


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def merged_frozen_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for map_name in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(map_name)
        require(
            isinstance(mapping, dict) and mapping, "Missing E164 hash map: " + map_name
        )
        for name, digest in mapping.items():
            if name in merged:
                require(
                    merged[name] == digest, "Conflicting E164 duplicate hash: " + name
                )
            else:
                merged[name] = digest
    return merged


def verify_hashes_once(mapping: dict[str, str]) -> None:
    for name, digest in mapping.items():
        require(sha256(safe_path(name)) == digest, "E164 frozen hash changed: " + name)


def validate_source_review_contract(review: dict[str, Any]) -> None:
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and type(review.get("original_processes_allowed")) is int
        and review["original_processes_allowed"] == 1
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E164 two-source review required",
    )


def preflight_contract(
    freeze_path: str | Path,
    freeze_sha256: str,
    review_path: str | Path,
    review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Hash/contract-only gate; this function never opens an original NPZ."""
    require(freeze_sha256 == FREEZE_SHA, "Exact E164 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E164 freeze bytes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == SCHEMA
        and freeze.get("design_sha256") == DESIGN_SHA
        and sha256(safe_path(freeze["design"])) == DESIGN_SHA
        and freeze.get("owned_files") == list(OWNED)
        and freeze.get("solver") == "Clarabel"
        and freeze.get("solver_version") == "0.11.1"
        and freeze.get("new_onset_features_evaluated_before_freeze") is False
        and freeze.get("fit_results_seen_before_freeze") is False,
        "E164 fixed contract changed",
    )
    require(
        freeze.get("counts")
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "solver_calls_max": 1,
            "body_models": 0,
            "graph_store_instances": 0,
            "state_store_instances": 0,
            "physics_steps": 0,
            "neural_steps": 0,
            "new_EMA_rollouts": 0,
            "recorded_recurrence_comparisons": 2000,
            "F6_exposed": False,
        },
        "E164 count contract changed",
    )
    require(
        freeze.get("resource_limits")
        == {
            "cpu_threads": 1,
            "gpu_count": 0,
            "fit_s": 20,
            "internal_s": 80,
            "external_s": 95,
            "supervisor_s": 105,
            "raw_bytes": 201326592,
            "free_bytes": 1073741824,
        },
        "E164 resource contract changed",
    )
    require(
        str(clarabel.__version__) == freeze["solver_version"],
        "Clarabel version changed",
    )
    require(
        len(freeze.get("input_sha256", {})) == 230
        and len(freeze.get("reused_source_sha256", {})) == 38
        and len(freeze.get("preservation_sha256", {})) == 3193,
        "E164 frozen hash-map counts changed",
    )
    require(
        sha256(review_path) == review_sha256, "Exact E164 source review SHA required"
    )
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    validate_source_review_contract(review)
    for name, digest in review["source_sha256"].items():
        require(
            sha256(safe_path(name)) == digest, "Reviewed E164 source changed: " + name
        )
    hashes = merged_frozen_hashes(freeze)
    verify_hashes_once(hashes)
    return freeze, review, hashes


def onset_feature(bank: Any) -> np.ndarray:
    bank = np.asarray(bank, dtype=np.float64)
    require(bank.ndim == 3 and bank.shape[2] == 3, "E164 bank shape invalid")
    require(np.isfinite(bank).all(), "E164 bank must be finite")
    return np.maximum(bank[:, :, 0] - bank[:, :, 2], 0.0)


def reconstruct_activation(
    bank: Any,
    original_activation: Any,
    slopes: Any,
    kappa: Any,
    membership: Any,
    fixed_muscles: Any,
) -> np.ndarray:
    bank = np.asarray(bank, dtype=np.float64)
    original = np.asarray(original_activation, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)
    fixed = np.asarray(fixed_muscles, dtype=np.int64)
    require(
        bank.shape[1:] == (391, 3)
        and original.shape == (bank.shape[0], 96)
        and slopes.shape == (96, 391, 3, 2)
        and kappa.shape == (391,)
        and membership.shape == (391,),
        "E164 activation reconstruction shape mismatch",
    )
    result = np.empty_like(original)
    result[:, fixed] = original[:, fixed]
    variable = np.setdiff1d(np.arange(96), fixed)
    hinge = np.maximum(bank - kappa[None, :, None], 0.0)
    for muscle in variable:
        members = np.flatnonzero(membership == muscle)
        low = slopes[muscle, members, :, 0]
        delta = slopes[muscle, members, :, 1] - low
        result[:, muscle] = np.sum(low[None] * bank[:, members], axis=(1, 2)) + np.sum(
            delta[None] * hinge[:, members], axis=(1, 2)
        )
    return result


@dataclass(frozen=True)
class ParameterLayout:
    variable_muscles: np.ndarray
    variable_motors: np.ndarray
    membership: np.ndarray
    g_index: np.ndarray
    beta_index: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mu: np.ndarray
    tonic_lower: np.ndarray
    tonic_upper: np.ndarray

    @property
    def parameter_count(self) -> int:
        return self.variable_muscles.size + self.variable_motors.size


def build_parameter_layout(
    membership: Any,
    slopes: Any,
    fixed_muscles: Any,
    frozen_variable_muscles: Any | None = None,
    frozen_variable_motors: Any | None = None,
) -> ParameterLayout:
    membership = np.asarray(membership, dtype=np.int64)
    slopes = np.asarray(slopes, dtype=np.float64)
    fixed = np.asarray(fixed_muscles, dtype=np.int64)
    variable_muscles = np.setdiff1d(np.arange(96), fixed)
    variable_motors = np.flatnonzero(np.isin(membership, variable_muscles))
    if frozen_variable_muscles is not None:
        require(
            np.array_equal(variable_muscles, frozen_variable_muscles),
            "E164 variable muscles differ",
        )
    if frozen_variable_motors is not None:
        require(
            np.array_equal(variable_motors, frozen_variable_motors),
            "E164 variable motors differ",
        )
    require(
        variable_muscles.size == 77 and variable_motors.size == 330,
        "E164 variable dimension differs",
    )
    g_index = np.full(96, -1, dtype=np.int64)
    g_index[variable_muscles] = np.arange(variable_muscles.size)
    beta_index = np.full(391, -1, dtype=np.int64)
    beta_index[variable_motors] = variable_muscles.size + np.arange(
        variable_motors.size
    )
    low = slopes[..., 0]
    high = slopes[..., 1]
    mu = np.sum(np.minimum(low, high), axis=2)
    tonic_lower = np.asarray([np.sum(mu[m, membership == m]) for m in variable_muscles])
    tonic_upper = np.asarray(
        [
            np.sum(np.maximum(low[m, membership == m], high[m, membership == m]))
            for m in variable_muscles
        ]
    )
    lower = np.r_[np.full(variable_muscles.size, 0.5), np.zeros(variable_motors.size)]
    upper = np.r_[np.ones(variable_muscles.size), np.full(variable_motors.size, 2.0)]
    return ParameterLayout(
        variable_muscles,
        variable_motors,
        membership,
        g_index,
        beta_index,
        lower,
        upper,
        mu,
        tonic_lower,
        tonic_upper,
    )


def apply_onset_transfer(
    baseline_activation: Any,
    onset: Any,
    layout: ParameterLayout,
    vector: Any,
) -> np.ndarray:
    baseline = np.asarray(baseline_activation, dtype=np.float64)
    onset = np.asarray(onset, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    require(
        vector.shape == (layout.parameter_count,),
        "E164 parameter vector shape mismatch",
    )
    result = baseline.copy()
    for muscle in layout.variable_muscles:
        result[:, muscle] *= vector[layout.g_index[muscle]]
    for motor in layout.variable_motors:
        result[:, layout.membership[motor]] += (
            vector[layout.beta_index[motor]] * onset[:, motor]
        )
    return result


def reference_at_phase(
    reference: dict[str, np.ndarray], phase: Any, lf_column: int, mask: Any
) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    coordinates = np.mod(reference["phase"][selected, lf_column] / (2 * np.pi), 1)
    values = reference["command"][selected]
    order = np.argsort(coordinates)
    x, y = coordinates[order], values[order]
    extended_x = np.r_[x[-1] - 1, x, x[0] + 1]
    extended_y = np.vstack((y[-1], y, y[0]))
    query = np.mod(phase, 1)
    return np.column_stack(
        [np.interp(query, extended_x, extended_y[:, column]) for column in range(42)]
    )


def own_leg_targets(
    reference: dict[str, np.ndarray],
    phases: Any,
    offsets: Any,
    mask: Any,
    lf_column: int,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    source = reference["time_s"] >= 5
    target = np.empty((int(selected.sum()), 42), dtype=np.float64)
    for leg in range(6):
        rows = reference_at_phase(
            reference, phases[selected, leg] - offsets[leg], lf_column, source
        )
        target[:, 7 * leg : 7 * (leg + 1)] = rows[:, 7 * leg : 7 * (leg + 1)]
    return target


def thin_qr(matrix: Any, offset: Any) -> np.ndarray:
    return e145.thin_qr_compress(matrix, offset)


@dataclass(frozen=True)
class Criterion:
    name: str
    matrix: sparse.csr_matrix
    offset: np.ndarray
    limit: float
    qr_blocks: tuple[np.ndarray, ...]


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
    criteria: tuple[Criterion, ...]
    startup_matrix: sparse.csr_matrix
    startup_offset: np.ndarray


def _leg_affine(
    activation: np.ndarray,
    onset: np.ndarray,
    moment: np.ndarray,
    rest: np.ndarray,
    layout: ParameterLayout,
    leg: int,
    rows: np.ndarray,
    gain: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    # Anatomy ownership is fixed by muscle group, while cross-leg moment must be zero.
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    muscles = layout.variable_muscles[group_legs[layout.variable_muscles] == leg]
    motors = layout.variable_motors[
        np.isin(layout.membership[layout.variable_motors], muscles)
    ]
    indices = np.r_[layout.g_index[muscles], layout.beta_index[motors]]
    joints = np.arange(7 * leg, 7 * (leg + 1))
    local = np.empty((rows.size, 7, indices.size), dtype=np.float64)
    for column, muscle in enumerate(muscles):
        local[:, :, column] = (
            gain * activation[rows, muscle, None] * moment[joints, muscle]
        )
    for offset, motor in enumerate(motors, start=muscles.size):
        muscle = layout.membership[motor]
        local[:, :, offset] = gain * onset[rows, motor, None] * moment[joints, muscle]
    baseline = rest[joints] + gain * (activation[rows] @ moment[joints].T)
    constant = baseline - np.sum(local[:, :, : muscles.size], axis=2)
    selector = sparse.csr_matrix(
        (np.ones(indices.size), (np.arange(indices.size), indices)),
        shape=(indices.size, layout.parameter_count),
    )
    return (
        sparse.csr_matrix(local.reshape(-1, indices.size)) @ selector,
        constant.reshape(-1),
        indices,
    )


def _compress_blocks(
    name: str,
    blocks: list[tuple[sparse.csr_matrix, np.ndarray]],
    limit: float,
    nparam: int,
    atol: float,
    rtol: float,
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]] | None = None,
) -> Criterion:
    matrices: list[sparse.csr_matrix] = []
    offsets: list[np.ndarray] = []
    qrs: list[np.ndarray] = []
    witnesses = (
        np.r_[np.ones(77), np.zeros(nparam - 77)],
        np.linspace(-0.375, 0.625, nparam),
        np.cos(np.arange(nparam, dtype=np.float64) * 0.173),
    )
    for matrix, offset in blocks:
        active = np.unique(matrix.indices)
        local = matrix[:, active].toarray()
        r = thin_qr(local, offset)
        rm, ro = e145.compressed_affine(r)
        selector = sparse.csr_matrix(
            (np.ones(active.size), (np.arange(active.size), active)),
            shape=(active.size, nparam),
        )
        compressed = sparse.csr_matrix(rm) @ selector
        block = len(qrs)
        raw[f"qr_{name}_{block}"] = r
        raw[f"qr_{name}_{block}_active"] = active
        direct_norms = np.asarray(
            [np.linalg.norm(matrix @ witness + offset) for witness in witnesses]
        )
        compressed_norms = np.asarray(
            [np.linalg.norm(compressed @ witness + ro) for witness in witnesses]
        )
        raw[f"qr_{name}_{block}_direct_norms"] = direct_norms
        raw[f"qr_{name}_{block}_compressed_norms"] = compressed_norms
        numerical_check(
            np.all(
                np.abs(direct_norms - compressed_norms) <= atol + rtol * direct_norms
            ),
            "E164 QR witness mismatch: " + name,
            pending,
        )
        matrices.append(compressed)
        offsets.append(ro)
        qrs.append(r)
    return Criterion(
        name,
        sparse.vstack(matrices, format="csr"),
        np.concatenate(offsets),
        float(limit),
        tuple(qrs),
    )


def build_conic_problem(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray] | None = None,
    pending: list[tuple[bool, str]] | None = None,
) -> tuple[ConicProblem, dict[str, np.ndarray]]:
    layout: ParameterLayout = saved["layout"]
    nparam = layout.parameter_count
    indices = saved["target_indices"]
    pair_start = saved["pair_start"]
    pair_end = saved["pair_end"]
    command_blocks = []
    velocity_blocks = []
    startup_blocks = []
    if raw is None:
        raw = {}
    for leg in range(6):
        command_matrix, command_base, _ = _leg_affine(
            saved["activation"],
            saved["onset"],
            saved["moment"],
            saved["rest"],
            layout,
            leg,
            indices,
            0.02,
        )
        command_offset = command_base - saved["target"][
            :, 7 * leg : 7 * (leg + 1)
        ].reshape(-1)
        a0, b0, _ = _leg_affine(
            saved["activation"],
            saved["onset"],
            saved["moment"],
            saved["rest"],
            layout,
            leg,
            pair_start,
            0.02,
        )
        a1, b1, _ = _leg_affine(
            saved["activation"],
            saved["onset"],
            saved["moment"],
            saved["rest"],
            layout,
            leg,
            pair_end,
            0.02,
        )
        velocity_matrix = (a1 - a0) / 0.005
        velocity_offset = (b1 - b0) / 0.005 - saved["target_velocity"][
            :, 7 * leg : 7 * (leg + 1)
        ].reshape(-1)
        startup_matrix, startup_base, _ = _leg_affine(
            saved["activation"],
            saved["onset"],
            saved["moment"],
            saved["rest"],
            layout,
            leg,
            np.arange(1, 100),
            0.02,
        )
        startup_offset = startup_base - saved["startup_target"][
            :, 7 * leg : 7 * (leg + 1)
        ].reshape(-1)
        command_blocks.append((command_matrix, command_offset))
        velocity_blocks.append((velocity_matrix, velocity_offset))
        startup_blocks.append((startup_matrix, startup_offset))
        for prefix, matrix, offset in (
            ("command", command_matrix, command_offset),
            ("velocity", velocity_matrix, velocity_offset),
            ("startup", startup_matrix, startup_offset),
        ):
            raw[f"{prefix}_{LEGS[leg]}_data"] = matrix.data
            raw[f"{prefix}_{LEGS[leg]}_indices"] = matrix.indices
            raw[f"{prefix}_{LEGS[leg]}_indptr"] = matrix.indptr
            raw[f"{prefix}_{LEGS[leg]}_shape"] = np.asarray(matrix.shape)
            raw[f"{prefix}_{LEGS[leg]}_offset"] = offset
    baseline = saved["baseline_metrics"]
    c = freeze["contract"]
    criteria = [
        _compress_blocks(
            "overall_command",
            command_blocks,
            math.sqrt(indices.size * 42)
            * baseline["overall_command_RMS"]
            * c["proposal_command_RMS_ratio"],
            nparam,
            c["QR_absolute_atol"],
            c["QR_relative_atol"],
            raw,
            pending,
        ),
        _compress_blocks(
            "overall_velocity",
            velocity_blocks,
            math.sqrt(pair_start.size * 42) * baseline["overall_velocity_RMS"],
            nparam,
            c["QR_absolute_atol"],
            c["QR_relative_atol"],
            raw,
            pending,
        ),
    ]
    for leg in range(6):
        criteria.append(
            _compress_blocks(
                f"{LEGS[leg]}_command",
                [command_blocks[leg]],
                math.sqrt(indices.size * 7)
                * baseline["by_leg"][LEGS[leg]]["command_RMS"],
                nparam,
                c["QR_absolute_atol"],
                c["QR_relative_atol"],
                raw,
                pending,
            )
        )
        criteria.append(
            _compress_blocks(
                f"{LEGS[leg]}_velocity",
                [velocity_blocks[leg]],
                math.sqrt(pair_start.size * 7)
                * baseline["by_leg"][LEGS[leg]]["velocity_RMS"],
                nparam,
                c["QR_absolute_atol"],
                c["QR_relative_atol"],
                raw,
                pending,
            )
        )
    criteria.append(
        _compress_blocks(
            "startup",
            startup_blocks,
            math.sqrt(99 * 42) * c["startup_limits"]["q_RMS"],
            nparam,
            c["QR_absolute_atol"],
            c["QR_relative_atol"],
            raw,
            pending,
        )
    )
    require(len(criteria) == c["soc_count"] == 15, "E164 SOC count differs")
    startup_matrix = sparse.vstack([item[0] for item in startup_blocks], format="csr")
    startup_offset = np.concatenate([item[1] for item in startup_blocks])
    lower = np.r_[layout.lower, c["margin_bounds"][0]]
    upper = np.r_[layout.upper, c["margin_bounds"][1]]
    rows = [-sparse.eye(nparam + 1, format="csr"), sparse.eye(nparam + 1, format="csr")]
    rhs = [-lower, upper]
    for muscle in layout.variable_muscles:
        members = np.flatnonzero(layout.membership == muscle)
        g = layout.g_index[muscle]
        n = members.size
        for motor in members:
            row = sparse.csr_matrix(
                ([-layout.mu[muscle, motor]], ([0], [g])), shape=(1, nparam + 1)
            )
            rows.append(row)
            rhs.append(np.asarray([-c["motor_floor_fraction"] / n]))
        rows.append(
            sparse.csr_matrix(
                ([-layout.tonic_lower[g]], ([0], [g])), shape=(1, nparam + 1)
            )
        )
        rhs.append(np.asarray([-c["tonic_lower_budget"]]))
        beta = layout.beta_index[members]
        columns = np.r_[g, beta]
        values = np.r_[layout.tonic_upper[g], np.ones(beta.size)]
        rows.append(
            sparse.csr_matrix(
                (values, (np.zeros(columns.size), columns)), shape=(1, nparam + 1)
            )
        )
        rhs.append(np.asarray([c["total_upper_budget"]]))
    tcol = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    rows.extend(
        [
            sparse.hstack((startup_matrix / 0.04, tcol)),
            sparse.hstack((-startup_matrix / 0.04, tcol)),
        ]
    )
    rhs.extend([1 - startup_offset / 0.04, 1 + startup_offset / 0.04])
    nonnegative_A = sparse.vstack(rows, format="csr")
    nonnegative_b = np.concatenate(rhs)
    A_parts = [nonnegative_A]
    b_parts = [nonnegative_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(nonnegative_A.shape[0])]
    kinds = ["NonnegativeConeT"]
    dims = [nonnegative_A.shape[0]]
    for criterion in criteria:
        soc_A, soc_b = e145.make_soc_rows(
            criterion.matrix, criterion.offset, criterion.limit, nparam
        )
        A_parts.append(soc_A)
        b_parts.append(soc_b)
        cones.append(clarabel.SecondOrderConeT(soc_A.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(soc_A.shape[0])
    problem = ConicProblem(
        sparse.csc_matrix((nparam + 1, nparam + 1)),
        np.r_[np.zeros(nparam), -1.0],
        sparse.vstack(A_parts, format="csc"),
        np.concatenate(b_parts),
        tuple(cones),
        tuple(kinds),
        np.asarray(dims),
        lower,
        upper,
        tuple(criteria),
        startup_matrix,
        startup_offset,
    )
    witness = np.r_[
        np.ones(layout.variable_muscles.size),
        np.zeros(layout.variable_motors.size),
        c["baseline_witness_margin"],
    ]
    slack = problem.b - problem.A @ witness
    raw.update(
        problem_A_data=problem.A.data,
        problem_A_indices=problem.A.indices,
        problem_A_indptr=problem.A.indptr,
        problem_A_shape=np.asarray(problem.A.shape),
        problem_b=problem.b,
        problem_q=problem.q,
        lower=lower,
        upper=upper,
        baseline_witness=witness,
        baseline_witness_slack=slack,
        cone_dims=problem.cone_dims,
        cone_kinds=np.asarray(problem.cone_kinds),
    )
    numerical_check(
        np.all(slack[: nonnegative_A.shape[0]] >= -c["physical_atol"]),
        "E164 baseline witness linear infeasible",
        pending,
    )
    start = nonnegative_A.shape[0]
    for criterion, dim in zip(criteria, problem.cone_dims[1:], strict=True):
        section = slack[start : start + dim]
        numerical_check(
            section[0] >= np.linalg.norm(section[1:]) - c["physical_atol"],
            "E164 baseline witness SOC infeasible: " + criterion.name,
            pending,
        )
        start += dim
    return problem, raw


def direct_metrics(
    saved: dict[str, Any], vector: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    activation = apply_onset_transfer(
        saved["activation"], saved["onset"], saved["layout"], vector
    )
    q = saved["rest"] + 0.02 * (activation @ saved["moment"].T)
    metrics, arrays = metrics_from_recorded(saved, q)
    arrays.update(
        candidate_activation=activation,
        candidate_command=q,
        candidate_tendon=np.asarray(saved["tendon"], dtype=np.float64),
    )
    return metrics, arrays


def metrics_from_recorded(
    saved: dict[str, Any], q: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate physical metrics from an explicit recorded command trace."""
    q = np.asarray(q, dtype=np.float64)
    require(
        q.ndim == 2 and q.shape[1] == 42 and q.shape[0] >= 100 and np.isfinite(q).all(),
        "E164 command trace invalid",
    )
    target_q = q[saved["target_indices"]]
    velocity = (q[saved["pair_end"]] - q[saved["pair_start"]]) / 0.005
    command_error = target_q - saved["target"]
    velocity_error = velocity - saved["target_velocity"]
    startup_error = q[1:100] - saved["startup_target"]
    metrics: dict[str, Any] = {
        "overall_command_RMS": float(np.sqrt(np.mean(command_error**2))),
        "overall_velocity_RMS": float(np.sqrt(np.mean(velocity_error**2))),
        "startup_q_RMS": float(np.sqrt(np.mean(startup_error**2))),
        "startup_q_max_abs": float(np.max(np.abs(startup_error))),
        "by_leg": {},
    }
    for leg, name in enumerate(LEGS):
        columns = slice(7 * leg, 7 * (leg + 1))
        metrics["by_leg"][name] = {
            "command_RMS": float(np.sqrt(np.mean(command_error[:, columns] ** 2))),
            "velocity_RMS": float(np.sqrt(np.mean(velocity_error[:, columns] ** 2))),
        }
    arrays = {
        "candidate_velocity": velocity,
        "command_error": command_error,
        "velocity_error": velocity_error,
        "startup_error": startup_error,
    }
    return metrics, arrays


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_frozen_npy_reference(
    state_path: Path, reference: Any, frozen_hashes: dict[str, str]
) -> np.ndarray:
    """Resolve one StateStore JSON array reference without constructing StateStore."""
    require(
        isinstance(reference, dict)
        and set(reference) == {"__npy__"}
        and isinstance(reference["__npy__"], str),
        "E164 frozen checkpoint array reference required",
    )
    name = Path(reference["__npy__"])
    require(
        name.name == reference["__npy__"] and not name.is_absolute(),
        "E164 checkpoint array basename required",
    )
    path = (state_path.parent / name).resolve()
    require(path.is_relative_to(ROOT), "E164 checkpoint array escaped repository")
    relative = path.relative_to(ROOT).as_posix()
    require(
        relative in frozen_hashes and sha256(path) == frozen_hashes[relative],
        "E164 checkpoint array is not frozen: " + relative,
    )
    return np.load(path, allow_pickle=False)


def load_original_inputs(
    freeze: dict[str, Any], frozen_hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    values = {
        name: _load_npz(safe_path(path))
        for name, path in freeze["paths"].items()
        if path.endswith(".npz")
    }
    values["trace"] = json.loads(
        safe_path(freeze["paths"]["trace"]).read_text(encoding="utf-8")
    )
    state_path = safe_path(freeze["paths"]["state_json"])
    values["state"] = json.loads(state_path.read_text(encoding="utf-8"))
    hashes = (
        frozen_hashes if frozen_hashes is not None else merged_frozen_hashes(freeze)
    )
    state_adapter = values["state"]["adapter"]
    values["state_initial_bank"] = load_frozen_npy_reference(
        state_path, state_adapter["bank"], hashes
    )
    values["state_initial_activation"] = load_frozen_npy_reference(
        state_path, state_adapter["activation"], hashes
    )
    state_parameters = state_adapter.get("parameters", {})
    values["state_parameters"] = {
        key: load_frozen_npy_reference(state_path, state_parameters[key], hashes)
        for key in ("fixed_muscles", "kappa", "slopes", "membership", "tau")
    }
    return values


def prepare_saved(
    data: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    c = freeze["sample_contract"]
    recruitment, adapter, parameters = (
        data["recruitment"],
        data["adapter"],
        data["parameters"],
    )
    bank = np.asarray(recruitment["bank"], dtype=np.float64)
    activation_saved = np.asarray(recruitment["activation"], dtype=np.float64)
    original_activation = np.asarray(
        recruitment["original_activation"], dtype=np.float64
    )
    require(
        bank.shape == (2001, 391, 3)
        and activation_saved.shape == original_activation.shape == (2001, 96)
        and np.isfinite(bank).all()
        and np.isfinite(activation_saved).all()
        and np.isfinite(original_activation).all(),
        "E164 saved signal shape/finite receipt differs",
    )
    state_adapter = data["state"]["adapter"]
    state_bank = data.get("state_initial_bank", state_adapter.get("bank"))
    state_activation = data.get(
        "state_initial_activation", state_adapter.get("activation")
    )
    require(
        np.array_equal(bank[0], np.asarray(state_bank, dtype=np.float64))
        and np.array_equal(
            activation_saved[0], np.asarray(state_activation, dtype=np.float64)
        ),
        "E164 frozen initial bank/activation differs",
    )
    for key in (
        "complete",
        "bank_available",
        "activation_available",
        "original_activation_available",
        "motor_rate_available",
    ):
        receipt = np.asarray(recruitment[key])
        require(
            receipt.shape == (2001,) and receipt.dtype == np.bool_ and receipt.all(),
            "E164 incomplete receipt: " + key,
        )
    require(
        np.array_equal(recruitment["control_tick"], np.arange(2001))
        and np.array_equal(recruitment["network_tick"], np.arange(2001) * 20),
        "E164 tick receipt differs",
    )
    require(
        np.allclose(
            recruitment["physics_time_s"],
            np.arange(2001) * 0.005,
            atol=c["clock_atol"],
            rtol=0,
        ),
        "E164 clock differs",
    )
    for key in ("membership", "motor_ids"):
        require(
            np.array_equal(adapter[key], parameters[key]),
            "E164 adapter/parameter mapping differs: " + key,
        )
    membership_source = np.asarray(adapter["membership"])
    membership = np.asarray(membership_source, dtype=np.int64)
    motor_ids = np.asarray(adapter["motor_ids"])
    pool = np.asarray(adapter["pool"], dtype=np.float64)
    calibration = data["calibration"]
    raw.update(
        adapter_membership=membership,
        adapter_motor_ids=motor_ids,
        adapter_pool=pool,
        calibration_membership_residual=np.asarray(calibration["membership"])
        - membership,
        calibration_moment_residual=np.asarray(calibration["moment"], dtype=np.float64)
        - np.asarray(adapter["moment"], dtype=np.float64),
        calibration_rest_residual=np.asarray(calibration["rest"], dtype=np.float64)
        - np.asarray(adapter["rest"], dtype=np.float64),
        calibration_pool_residual=np.asarray(calibration["pool"], dtype=np.float64)
        - pool,
        calibration_ltm_residual=np.asarray(calibration["ltm"], dtype=np.int64)
        - np.asarray(adapter["ltm"], dtype=np.int64),
    )
    require(
        membership.shape == (391,)
        and membership_source.dtype == np.int64
        and np.all((membership >= 0) & (membership < 96))
        and motor_ids.shape == (391,)
        and motor_ids.dtype.kind in "iu"
        and len(np.unique(motor_ids)) == 391
        and np.array_equal(motor_ids, np.sort(motor_ids))
        and pool.shape == (96, 391)
        and np.isfinite(pool).all()
        and np.all(pool >= 0),
        "E164 adapter identity/support shape differs",
    )
    expected_support = np.zeros_like(pool, dtype=bool)
    expected_support[membership, np.arange(391)] = True
    require(
        np.array_equal(pool != 0, expected_support)
        and np.allclose(np.sum(pool, axis=1), 1.0, atol=1e-14, rtol=0),
        "E164 normalized single-muscle pool support differs",
    )
    require(
        all(
            np.array_equal(np.asarray(calibration[key]), np.asarray(adapter[key]))
            for key in ("membership", "motor_ids", "moment", "rest", "pool", "ltm")
        ),
        "E164 calibration/adapter identity differs",
    )
    state_parameters = state_adapter.get("parameters", {})
    roster = state_parameters.get("motor_roster")
    patch_roster = (
        data["state"].get("monotone_recruitment_patch", {}).get("motor_roster")
    )
    require(
        isinstance(roster, list)
        and len(roster) == 391
        and all(isinstance(item, str) and item for item in roster)
        and len(set(roster)) == 391
        and roster == patch_roster,
        "E164 checkpoint root motor roster differs",
    )
    require(
        isinstance(state_adapter.get("hash"), str)
        and bool(state_adapter["hash"])
        and data["trace"].get("adapter_hash") == state_adapter["hash"]
        and isinstance(data["state"].get("graph_hash"), str)
        and bool(data["state"]["graph_hash"])
        and data["trace"].get("graph_hash") == data["state"]["graph_hash"],
        "E164 trace/checkpoint graph or adapter identity differs",
    )
    raw.update(
        motor_root_roster=np.asarray(roster),
        patch_motor_root_roster=np.asarray(patch_roster),
        graph_hash=np.asarray(data["state"]["graph_hash"]),
        adapter_hash=np.asarray(state_adapter["hash"]),
    )
    fixed = np.asarray(c["fixed_muscles"], dtype=np.int64)
    slopes_source = np.asarray(parameters["slopes"])
    slopes = np.asarray(slopes_source, dtype=np.float64)
    tau_source = np.asarray(parameters["tau"])
    tau = np.asarray(tau_source, dtype=np.float64)
    kappa = np.asarray(parameters["kappa"])
    require(
        slopes.shape == (96, 391, 3, 2)
        and slopes_source.dtype == np.float64
        and np.isfinite(slopes).all()
        and np.all(slopes >= 0)
        and tau_source.dtype == np.float64
        and np.array_equal(tau, np.asarray(c["tau_s"], dtype=np.float64)),
        "E164 slope/tau receipt differs",
    )
    require(
        kappa.shape == (391,)
        and kappa.dtype == np.float64
        and np.isfinite(kappa).all()
        and np.all(kappa >= 0),
        "E164 kappa receipt differs",
    )
    slope_support = np.any(slopes > 0, axis=(2, 3))
    supported_slopes = slopes[membership, np.arange(391)]
    require(
        np.array_equal(slope_support, expected_support)
        and np.all(np.sum(np.min(supported_slopes, axis=-1), axis=1) > 0),
        "E164 positive slope ownership differs",
    )
    if "state_parameters" in data:
        state_values = data["state_parameters"]
        raw.update(
            state_kappa_residual=np.asarray(state_values["kappa"])
            - np.asarray(parameters["kappa"]),
            state_slopes_residual=np.asarray(state_values["slopes"]) - slopes,
            state_membership_residual=np.asarray(state_values["membership"])
            - membership,
            state_tau_residual=np.asarray(state_values["tau"]) - tau,
        )
        require(
            np.array_equal(np.asarray(state_values["fixed_muscles"]), fixed)
            and np.array_equal(np.asarray(state_values["kappa"]), parameters["kappa"])
            and np.array_equal(np.asarray(state_values["slopes"]), slopes)
            and np.array_equal(np.asarray(state_values["membership"]), membership)
            and np.array_equal(np.asarray(state_values["tau"]), tau),
            "E164 checkpoint parameter receipt differs",
        )
    reconstructed_activation = reconstruct_activation(
        bank,
        original_activation,
        parameters["slopes"],
        parameters["kappa"],
        adapter["membership"],
        fixed,
    )
    onset = onset_feature(bank)
    alpha = -np.expm1(-0.005 / np.asarray(parameters["tau"], dtype=np.float64))
    predicted_bank = bank[:-1] + alpha[None, None, :] * (
        recruitment["motor_rate"][1:, :, None] - bank[:-1]
    )
    recurrence_residual = bank[1:] - predicted_bank
    rows = data["trace"]["samples"]
    trace_ticks = np.asarray([row["control_tick"] for row in rows])
    trace_clock = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    trace_motor_rate = np.asarray(
        [row.get("motor_rates", row.get("motor_rate", [])) for row in rows],
        dtype=np.float64,
    )
    raw.update(
        trace_clock=trace_clock,
        trace_ticks=trace_ticks,
        trace_motor_rate=trace_motor_rate,
    )
    require(
        len(rows) == 2001
        and data["trace"].get("body_hash") == freeze["body_hash"]
        and np.array_equal(trace_ticks, np.arange(2001))
        and np.isfinite(trace_clock).all()
        and np.allclose(
            trace_clock,
            np.arange(2001) * 0.005,
            atol=c["clock_atol"],
            rtol=0,
        ),
        "E164 trace clock/ticks differ",
    )
    require(
        trace_motor_rate.shape == (2001, 391)
        and np.isfinite(trace_motor_rate).all()
        and np.array_equal(trace_motor_rate, np.asarray(recruitment["motor_rate"])),
        "E164 trace motor-rate receipt differs",
    )
    recorded = np.asarray([row["joint_targets_rad"] for row in rows], dtype=np.float64)
    tendon = np.asarray(
        [[row["tendon_inputs"][name + "_tarsus"] for name in LEGS] for row in rows],
        dtype=np.float64,
    )
    reconstructed_q = adapter["rest"] + c["command_gain"] * (
        reconstructed_activation @ adapter["moment"].T
    )
    reconstructed_tendon = c["tendon_max"] * np.tanh(
        reconstructed_activation[:, adapter["ltm"]] / c["tendon_rate_scale"]
    )
    raw.update(
        bank=bank,
        onset_feature=onset,
        reconstructed_activation=reconstructed_activation,
        activation_residual=reconstructed_activation - activation_saved,
        recurrence_residual=recurrence_residual,
        reconstructed_target=reconstructed_q,
        target_residual=reconstructed_q - recorded,
        reconstructed_tendon=reconstructed_tendon,
        tendon_residual=reconstructed_tendon - tendon,
        fixed_activation_recorded_residual=activation_saved[:, fixed]
        - original_activation[:, fixed],
    )
    numerical_check(
        np.max(np.abs(raw["activation_residual"])) <= c["reconstruction_atol"],
        "E164 activation reconstruction differs",
        pending,
    )
    numerical_check(
        np.max(np.abs(raw["recurrence_residual"])) <= c["reconstruction_atol"],
        "E164 bank recurrence differs",
        pending,
    )
    numerical_check(
        np.max(np.abs(raw["target_residual"][1:])) <= c["reconstruction_atol"],
        "E164 target reconstruction differs",
        pending,
    )
    numerical_check(
        np.max(np.abs(raw["tendon_residual"])) <= c["reconstruction_atol"],
        "E164 tendon reconstruction differs",
        pending,
    )
    numerical_check(
        np.array_equal(
            reconstructed_activation[:, fixed], original_activation[:, fixed]
        )
        and np.array_equal(activation_saved[:, fixed], original_activation[:, fixed]),
        "E164 fixed activation differs",
        pending,
    )
    mask_data = data["target_mask"]
    mask = np.asarray(mask_data["mask"])
    indices = np.asarray(mask_data["target_indices"], dtype=np.int64)
    require(
        mask.shape == (2001,)
        and mask.dtype == np.bool_
        and np.array_equal(np.flatnonzero(mask), indices),
        "E164 target mask differs",
    )
    require(
        indices.size == c["target_count"]
        and indices[0] == c["target_first_tick"]
        and indices[-1] == c["target_last_tick"]
        and np.all(np.diff(indices) == 1),
        "E164 target index contract differs",
    )
    phase = data["phase_diagnostic"]
    phase_clock = np.asarray(phase["control_time_s"], dtype=np.float64)
    mask_clock = np.asarray(mask_data["control_time_s"], dtype=np.float64)
    phase_values = np.asarray(phase["e1_phase"], dtype=np.float64)
    phase_available = np.asarray(phase["e1_phase_available"])
    raw.update(
        phase_clock_residual=phase_clock - np.arange(2001) * 0.005,
        mask_clock_residual=mask_clock - np.arange(2001) * 0.005,
        phase_available=phase_available,
    )
    require(
        phase_clock.shape == mask_clock.shape == (2001,)
        and np.allclose(phase_clock, mask_clock, atol=c["clock_atol"], rtol=0)
        and np.allclose(
            phase_clock, np.arange(2001) * 0.005, atol=c["clock_atol"], rtol=0
        ),
        "E164 phase/mask control clocks differ",
    )
    require(
        phase_values.shape == phase_available.shape == (2001, 6)
        and phase_available.dtype == np.bool_
        and np.array_equal(phase_available, np.isfinite(phase_values))
        and phase_available[indices].all()
        and np.isfinite(phase_values[indices]).all(),
        "E164 phase unavailable",
    )
    reference = data["reference"]
    source_holdout = (reference["time_s"] >= 5) & (reference["time_s"] < 8)
    holdout = reference["time_s"] >= 8
    holdout_reconstructed = reference_at_phase(
        reference,
        reference["phase"][holdout, c["lf_column"]] / (2 * np.pi),
        c["lf_column"],
        source_holdout,
    )
    holdout_residual = holdout_reconstructed - reference["command"][holdout]
    holdout_rms = float(np.sqrt(np.mean(holdout_residual**2)))
    reference_offsets = np.mod(
        reference["phase"] / (2 * np.pi)
        - np.asarray(reference["phase"])[:, [c["lf_column"]]] / (2 * np.pi),
        1.0,
    )
    expected_offsets = np.mod(
        np.asarray(c["offsets"]) - c["offsets"][c["lf_column"]], 1.0
    )
    offset_residual = np.minimum(
        np.mod(reference_offsets - expected_offsets, 1.0),
        np.mod(expected_offsets - reference_offsets, 1.0),
    )
    raw.update(
        reference_holdout_reconstructed=holdout_reconstructed,
        reference_holdout_residual=holdout_residual,
        reference_holdout_RMS=np.asarray(holdout_rms),
        reference_offset_residual=offset_residual,
    )
    numerical_check(
        holdout_rms <= freeze["contract"]["reference_holdout_RMS_limit"]
        and np.max(offset_residual) <= c["reconstruction_atol"],
        "E164 reference interpolation holdout differs",
        pending,
    )
    target = own_leg_targets(
        reference, phase["e1_phase"], np.asarray(c["offsets"]), mask, c["lf_column"]
    )
    target_velocity = np.diff(target, axis=0) / 0.005
    require(
        np.array_equal(mask_data["pair_start"], indices[:-1])
        and np.array_equal(mask_data["pair_end"], indices[1:]),
        "E164 velocity pairs differ",
    )
    startup = data["startup"]
    require(
        np.array_equal(np.flatnonzero(startup["mask"]), np.arange(1, 100)),
        "E164 startup mask differs",
    )
    require(
        np.asarray(startup["target_command"]).shape == (99, 42)
        and np.asarray(startup["target_tendon"]).shape == (99, 6),
        "E164 startup target shape differs",
    )
    layout = build_parameter_layout(
        adapter["membership"],
        parameters["slopes"],
        fixed,
        c["variable_muscles"],
        c["variable_motors"],
    )
    group_legs = np.asarray(c["group_legs"])
    groups = state_adapter["groups"]
    require(
        len(groups) == 96
        and all(
            groups[muscle]["leg"] == LEGS[group_legs[muscle]] for muscle in range(96)
        )
        and all(
            np.array_equal(
                np.asarray(groups[muscle]["columns"], dtype=np.int64),
                np.flatnonzero(membership == muscle),
            )
            for muscle in range(96)
        ),
        "E164 state group leg ownership differs",
    )
    singleton = np.flatnonzero(np.bincount(membership, minlength=96) == 1)
    canonical = np.zeros_like(slopes)
    canonical[:, :, 1, 0] = pool
    canonical[:, :, 1, 1] = pool
    require(
        np.array_equal(
            np.sort(fixed),
            np.union1d(np.asarray(adapter["ltm"], dtype=np.int64), singleton),
        )
        and np.array_equal(slopes[fixed], canonical[fixed]),
        "E164 fixed LTM/singleton identity differs",
    )
    for muscle in range(96):
        own = int(group_legs[muscle])
        outside = np.r_[np.arange(0, 7 * own), np.arange(7 * (own + 1), 42)]
        require(
            np.array_equal(adapter["moment"][outside, muscle], np.zeros(outside.size)),
            "E164 cross-leg moment differs",
        )
    baseline_saved = {
        # The affine family is anchored to the verified recorded E150 activation.
        # Reconstruction remains an independent residual check above.
        "activation": activation_saved,
        "onset": onset,
        "layout": layout,
        "moment": adapter["moment"],
        "rest": adapter["rest"],
        "target_indices": indices,
        "pair_start": indices[:-1],
        "pair_end": indices[1:],
        "target": target,
        "target_velocity": target_velocity,
        "startup_target": startup["target_command"],
        "startup_tendon": startup["target_tendon"],
        "tendon": tendon,
        "recorded_command": recorded,
    }
    baseline_vector = np.r_[
        np.ones(layout.variable_muscles.size), np.zeros(layout.variable_motors.size)
    ]
    baseline_metrics, baseline_arrays = metrics_from_recorded(baseline_saved, recorded)
    baseline_saved["baseline_metrics"] = baseline_metrics
    ltm_error = tendon[1:100] - startup["target_tendon"]
    baseline_saved["startup_ltm_RMS"] = float(np.sqrt(np.mean(ltm_error**2)))
    baseline_saved["startup_ltm_max_abs"] = float(np.max(np.abs(ltm_error)))
    raw.update(
        target_indices=indices,
        target=target,
        target_velocity=target_velocity,
        baseline_vector=baseline_vector,
        mu=layout.mu,
        tonic_lower=layout.tonic_lower,
        tonic_upper=layout.tonic_upper,
        recorded_command=recorded,
        baseline_velocity=baseline_arrays["candidate_velocity"],
        baseline_command_error=baseline_arrays["command_error"],
        baseline_velocity_error=baseline_arrays["velocity_error"],
        baseline_startup_error=baseline_arrays["startup_error"],
    )
    return baseline_saved


def problem_arrays(problem: ConicProblem) -> dict[str, np.ndarray]:
    return {
        "P_data": problem.P.data,
        "P_indices": problem.P.indices,
        "P_indptr": problem.P.indptr,
        "P_shape": np.asarray(problem.P.shape),
        "A_data": problem.A.data,
        "A_indices": problem.A.indices,
        "A_indptr": problem.A.indptr,
        "A_shape": np.asarray(problem.A.shape),
        "b": problem.b,
        "q": problem.q,
        "cone_dims": problem.cone_dims,
        "cone_kinds": np.asarray(problem.cone_kinds),
        "lower": problem.lower,
        "upper": problem.upper,
    }


def solver_raw(
    problem: ConicProblem,
    solution: Any,
    history: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> dict[str, np.ndarray]:
    x = np.asarray(solution.x, dtype=np.float64)
    s = np.asarray(solution.s, dtype=np.float64)
    z = np.asarray(solution.z, dtype=np.float64)
    projected, margins = e145.project_conic_dual(
        z,
        problem.cone_kinds,
        problem.cone_dims,
        interior_pad=freeze["dual_interior_pad"],
    )
    raw_bound = e145.dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        z,
        problem.lower,
        problem.upper,
        pad_factor=freeze["dual_upper_bound_pad"],
    )
    projected_bound = e145.dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        projected,
        problem.lower,
        problem.upper,
        pad_factor=freeze["dual_upper_bound_pad"],
    )
    return {
        "solver_x": x,
        "solver_s": s,
        "solver_z": z,
        "solver_status": np.asarray(str(solution.status)),
        "solver_iterations": np.asarray(solution.iterations),
        "solver_solve_time": np.asarray(solution.solve_time),
        "history_json": np.asarray(
            json.dumps(history, sort_keys=True, allow_nan=False)
        ),
        "primal_residual": problem.A @ x + s - problem.b,
        "stationarity_raw": problem.q + problem.A.T @ z,
        "solver_z_projected": projected,
        "projected_cone_margins": margins,
        **{f"raw_dual_{key}": np.asarray(value) for key, value in raw_bound.items()},
        **{
            f"projected_dual_{key}": np.asarray(value)
            for key, value in projected_bound.items()
        },
    }


def output_manifest(out: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(
        item
        for item in out.rglob("*")
        if item.is_file() and item.name != "output-manifest.json"
    ):
        files[path.relative_to(out).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "flylab.m3.e164.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def run(
    freeze_path: str | Path,
    freeze_sha256: str,
    source_review: str | Path,
    source_review_sha256: str,
    out_path: str | Path,
    *,
    input_loader: Callable[[dict[str, Any]], dict[str, Any]] = load_original_inputs,
    solver_factory: Callable[..., Any] = clarabel.DefaultSolver,
) -> dict[str, Any]:
    began = time.perf_counter()
    freeze, _review, _frozen_hashes = preflight_contract(
        freeze_path, freeze_sha256, source_review, source_review_sha256
    )
    out = Path(out_path).resolve()
    require(not out.exists(), "E164 output must be new")
    require(
        shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
        "E164 free-space reserve unavailable",
    )
    out.mkdir(parents=True)
    snapshot = out / "source-snapshot"
    snapshot.mkdir()
    snapshot_bytes: dict[str, bytes] = {}
    for name in OWNED:
        source = ROOT / name
        target = snapshot / Path(name).name
        shutil.copyfile(source, target)
        snapshot_bytes[name] = source.read_bytes()
    for label, source in (
        ("freeze.json", Path(freeze_path)),
        ("source-review.json", Path(source_review)),
        ("design.md", safe_path(freeze["design"])),
    ):
        target = snapshot / label
        shutil.copyfile(source, target)
        snapshot_bytes[label] = source.read_bytes()
    progress: dict[str, Any] = {
        "schema": "flylab.m3.e164.ledger.v1",
        "status": "created",
        "fit_calls": 0,
        "solver_constructor_attempts": 0,
        "solver_constructor_completed": 0,
        "callback_registration_attempts": 0,
        "callback_registration_completed": 0,
        "solver_returns": 0,
        "body_models": 0,
        "graph_store_instances": 0,
        "state_store_instances": 0,
        "physics_steps": 0,
        "neural_steps": 0,
        "new_EMA_rollouts": 0,
        "secondary_errors": [],
    }
    result: dict[str, Any] = {
        "schema": "flylab.m3.e164.result.v1",
        "validity": "incomplete",
        "conditional_proposal": False,
        "hypothesis_verdict": "invalid_not_tested",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
    }
    primary: BaseException | None = None
    thread_limit = None
    pre_raw: dict[str, np.ndarray] = {}
    pre_raw_persisted = False

    def persist() -> None:
        atomic_json(out / "ledger.json", progress)

    def deadline(stage: str) -> None:
        if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E164 internal deadline: " + stage)

    def secondary(error: BaseException, stage: str) -> None:
        progress["secondary_errors"].append({"stage": stage, "error": repr(error)})
        if primary is not None:
            primary.add_note("Secondary E164 " + stage + ": " + repr(error))

    try:
        thread_limit = threadpool_limits(limits=1)
        progress["cpu_thread_environment"] = {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        progress["threadpools_during"] = threadpool_info()
        require(
            all(
                item.get("num_threads") == 1 for item in progress["threadpools_during"]
            ),
            "E164 CPU threads exceed one",
        )
        persist()
        data = (
            input_loader(freeze, _frozen_hashes)
            if input_loader is load_original_inputs
            else input_loader(freeze)
        )
        input_checks: list[tuple[bool, str]] = []
        saved = prepare_saved(data, freeze, pre_raw, input_checks)
        save_check_evidence(out / "pre-solve-raw.npz", pre_raw, "input", input_checks)
        pre_raw_persisted = True
        enforce_checks(input_checks)
        deadline("after input checkpoint")
        pre_raw_persisted = False
        problem_checks: list[tuple[bool, str]] = []
        problem, _affine_raw = build_conic_problem(
            saved, freeze, pre_raw, problem_checks
        )
        pre_raw.update(problem_arrays(problem))
        save_check_evidence(
            out / "pre-solve-raw.npz", pre_raw, "problem", problem_checks
        )
        pre_raw_persisted = True
        enforce_checks(problem_checks)
        progress["status"] = "pre_solve_saved"
        persist()
        deadline("before solver")
        partial: dict[str, Any] = {}
        solver_state: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        fit_began = time.perf_counter()

        def sync_solver_counts() -> None:
            progress["fit_calls"] = max(
                progress["fit_calls"], int(solver_state.get("solve_entries", 0))
            )
            progress["solver_constructor_attempts"] = max(
                progress["solver_constructor_attempts"],
                int(solver_state.get("attempted", False)),
            )
            progress["solver_constructor_completed"] = max(
                progress["solver_constructor_completed"],
                int(solver_state.get("constructed", False)),
            )
            progress["callback_registration_attempts"] = max(
                progress["callback_registration_attempts"],
                int(solver_state.get("callback_registration_attempted", False)),
            )
            progress["callback_registration_completed"] = max(
                progress["callback_registration_completed"],
                int(solver_state.get("callback_registered", False)),
            )
            progress["solver_returns"] = max(
                progress["solver_returns"],
                int(solver_state.get("solve_returned", False)),
            )

        def instrumented_factory(*args: Any) -> Any:
            progress["solver_constructor_attempts"] = 1
            progress["status"] = "solver_constructor_requested"
            persist()
            solver = solver_factory(*args)
            progress["solver_constructor_completed"] = 1
            progress["status"] = "solver_constructed"
            persist()

            class InstrumentedSolver:
                def set_termination_callback(self, callback: Any) -> None:
                    progress["callback_registration_attempts"] = 1
                    progress["status"] = "callback_registration_requested"
                    persist()
                    solver.set_termination_callback(callback)
                    progress["callback_registration_completed"] = 1
                    progress["status"] = "callback_registered"
                    persist()

                def solve(self) -> Any:
                    progress["fit_calls"] = 1
                    progress["status"] = "solve_entered"
                    persist()
                    returned = solver.solve()
                    progress["solver_returns"] = 1
                    # A ledger write can fail before the outer wrapper sees the
                    # return. Own the bare values before any post-return I/O.
                    partial.update(e145._bare_solution_arrays(returned))
                    solver_state["solve_returned"] = True
                    progress["status"] = "solve_returned"
                    persist()
                    return returned

                def get_settings(self) -> Any:
                    return solver.get_settings()

            return InstrumentedSolver()

        try:
            solution, history, settings = e145.solve_conic_problem(
                problem,
                freeze,
                solver_factory=instrumented_factory,
                partial=partial,
                progress=progress,
                state=solver_state,
                history=history,
            )
        except BaseException as solve_error:
            sync_solver_counts()
            if partial:
                try:
                    atomic_npz(
                        out / "solver-partial.npz",
                        {key: np.asarray(value) for key, value in partial.items()},
                    )
                except BaseException as save_error:
                    solve_error.add_note(
                        "Secondary E164 solver partial save: " + repr(save_error)
                    )
                    progress["secondary_errors"].append(
                        {"stage": "solver partial raw", "error": repr(save_error)}
                    )
            raise
        sync_solver_counts()
        progress["fit_wall_s"] = time.perf_counter() - fit_began
        bare = {key: np.asarray(value) for key, value in partial.items()}
        atomic_npz(out / "solver-return-raw.npz", bare)
        require(
            progress["fit_wall_s"] <= freeze["resource_limits"]["fit_s"],
            "E164 fit deadline exceeded",
        )
        derived = solver_raw(problem, solution, history, freeze)
        atomic_npz(out / "solver-derived-raw.npz", derived)
        x = np.asarray(solution.x, dtype=np.float64)
        require(
            x.shape == (408,) and np.isfinite(x).all(), "E164 returned vector invalid"
        )
        metrics, candidate_raw = direct_metrics(saved, x[:-1])
        candidate_raw.update(returned_vector=x)
        atomic_npz(out / "candidate-raw.npz", candidate_raw)
        baseline = saved["baseline_metrics"]
        startup_limit = freeze["contract"]["startup_limits"]
        budget_ok = True
        layout: ParameterLayout = saved["layout"]
        for muscle in layout.variable_muscles:
            g = x[layout.g_index[muscle]]
            members = np.flatnonzero(layout.membership == muscle)
            budget_ok &= bool(
                np.all(
                    g * layout.mu[muscle, members]
                    >= freeze["contract"]["motor_floor_fraction"] / members.size
                    - freeze["contract"]["physical_atol"]
                )
            )
            budget_ok &= bool(
                g * layout.tonic_lower[layout.g_index[muscle]]
                >= freeze["contract"]["tonic_lower_budget"]
                - freeze["contract"]["physical_atol"]
            )
            budget_ok &= bool(
                g * layout.tonic_upper[layout.g_index[muscle]]
                + np.sum(x[layout.beta_index[members]])
                <= freeze["contract"]["total_upper_budget"]
                + freeze["contract"]["physical_atol"]
            )
        fixed_exact = np.array_equal(
            candidate_raw["candidate_activation"][
                :, freeze["sample_contract"]["fixed_muscles"]
            ],
            saved["activation"][:, freeze["sample_contract"]["fixed_muscles"]],
        )
        beta = x[layout.variable_muscles.size : layout.parameter_count]
        gates = {
            "returned_vector_finite_structural": bool(
                x.shape == (408,) and np.isfinite(x).all()
            ),
            "overall_command_20pct": metrics["overall_command_RMS"]
            <= 0.8 * baseline["overall_command_RMS"],
            "overall_velocity_nonincrease": metrics["overall_velocity_RMS"]
            <= baseline["overall_velocity_RMS"],
            "all_leg_nonincrease": all(
                metrics["by_leg"][leg][kind] <= baseline["by_leg"][leg][kind]
                for leg in LEGS
                for kind in ("command_RMS", "velocity_RMS")
            ),
            "startup_q": metrics["startup_q_RMS"] <= startup_limit["q_RMS"]
            and metrics["startup_q_max_abs"] <= startup_limit["q_max_abs"],
            "startup_LTM": saved["startup_ltm_RMS"] <= startup_limit["LTM_RMS"]
            and saved["startup_ltm_max_abs"] <= startup_limit["LTM_max_abs"],
            "budgets": budget_ok,
            "fixed_exact": fixed_exact,
            "beta_nonnegative_exact": bool(np.all(beta >= 0.0)),
            "bounds_with_numeric_atol": bool(
                np.all(x >= problem.lower - freeze["contract"]["physical_atol"])
                and np.all(x <= problem.upper + freeze["contract"]["physical_atol"])
            ),
        }
        proposal = all(gates.values())
        projected_upper = float(derived["projected_dual_margin_upper_bound"])
        verdict = (
            "conditional_onset_proposal"
            if proposal
            else (
                "refuted_fixed_onset_family"
                if projected_upper < 0
                else "valid_rejection_inconclusive"
            )
        )
        result.update(
            validity="valid",
            conditional_proposal=proposal,
            hypothesis_verdict=verdict,
            solver_status=str(solution.status),
            solver_status_is_solved=str(solution.status) == "Solved",
            margin=float(x[-1]),
            metrics=metrics,
            baseline_metrics=baseline,
            gates=gates,
            dual_projected_margin_upper_bound=projected_upper,
            solver_settings=settings,
        )
        progress["status"] = "complete"
    except BaseException as error:
        primary = error
        progress["status"] = "technical_failed"
        if pre_raw and not pre_raw_persisted:
            try:
                atomic_npz(out / "input-partial-raw.npz", pre_raw)
            except BaseException as save_error:
                error.add_note("Secondary E164 input partial save: " + repr(save_error))
                progress["secondary_errors"].append(
                    {"stage": "input partial raw", "error": repr(save_error)}
                )
    finally:
        if thread_limit is not None:
            try:
                thread_limit.restore_original_limits()
                progress["threadpools_after"] = threadpool_info()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    secondary(error, "threadpool restore")
        try:
            preflight_contract(
                freeze_path,
                freeze_sha256,
                source_review,
                source_review_sha256,
            )
            live = {
                **{name: (ROOT / name).read_bytes() for name in OWNED},
                "freeze.json": Path(freeze_path).read_bytes(),
                "source-review.json": Path(source_review).read_bytes(),
                "design.md": safe_path(freeze["design"]).read_bytes(),
            }
            require(live == snapshot_bytes, "E164 immutable snapshot bytes changed")
            require(
                all(
                    (snapshot / Path(name).name).read_bytes() == snapshot_bytes[name]
                    for name in OWNED
                )
                and all(
                    (snapshot / name).read_bytes() == snapshot_bytes[name]
                    for name in ("freeze.json", "source-review.json", "design.md")
                ),
                "E164 saved snapshot bytes changed",
            )
            progress["post_hashes_verified"] = True
        except BaseException as error:
            progress["post_hashes_verified"] = False
            if primary is None:
                primary = error
            else:
                secondary(error, "terminal hashes/snapshots")

        def set_failure_receipt() -> None:
            progress["status"] = "technical_failed"
            result.update(
                validity="incomplete",
                conditional_proposal=False,
                hypothesis_verdict="invalid_not_tested",
                technical_failure=True,
            )
            if primary is not None:
                error_record = {
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "traceback": "".join(traceback.format_exception(primary)),
                }
                result["primary_error"] = error_record
                progress["primary_error"] = error_record

        if primary is not None:
            set_failure_receipt()
        else:
            result["technical_failure"] = False
        result.update(
            process_wall_s=time.perf_counter() - began,
            prior_actual_original_s=freeze["prior_actual_original_s"],
            prior_conservative_s=freeze["prior_conservative_s"],
            conservative_cumulative_s=freeze["prior_conservative_s"]
            + time.perf_counter()
            - began,
            counts={
                key: progress[key]
                for key in (
                    "fit_calls",
                    "solver_constructor_attempts",
                    "solver_constructor_completed",
                    "callback_registration_attempts",
                    "callback_registration_completed",
                    "solver_returns",
                    "body_models",
                    "graph_store_instances",
                    "state_store_instances",
                    "physics_steps",
                    "neural_steps",
                    "new_EMA_rollouts",
                )
            },
        )

        def quarantine_success_receipts() -> None:
            for name in ("result.json", "ledger.json", "output-manifest.json"):
                path = out / name
                if not path.exists():
                    continue
                try:
                    path.replace(out / f"{name}.invalid-stale-{uuid.uuid4().hex}")
                except BaseException as error:
                    secondary(error, "quarantine " + name)

        try:
            atomic_json(out / "ledger.json", progress)
            atomic_json(out / "result.json", result)
            atomic_json(out / "output-manifest.json", output_manifest(out))
            total = sum(
                path.stat().st_size for path in out.rglob("*") if path.is_file()
            )
            require(
                total <= freeze["resource_limits"]["raw_bytes"],
                "E164 output byte limit exceeded",
            )
            require(
                shutil.disk_usage(out).free >= freeze["resource_limits"]["free_bytes"],
                "E164 terminal free-space reserve unavailable",
            )
            deadline("after terminal publication")
            disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
            require(
                disk["conditional_proposal"] == result["conditional_proposal"]
                and disk["validity"] == result["validity"],
                "E164 disk result differs",
            )
        except BaseException as error:
            had_success = primary is None
            if primary is None:
                primary = error
            else:
                secondary(error, "terminal publication")
            set_failure_receipt()
            if had_success:
                quarantine_success_receipts()
            # Bounded fail-closed rewrite. Failed atomic temporary and stale files
            # are evidence and are intentionally retained.
            failure_published = False
            for attempt in range(2):
                try:
                    atomic_json(out / "ledger.json", progress)
                    atomic_json(out / "result.json", result)
                    atomic_json(out / "output-manifest.json", output_manifest(out))
                    disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
                    require(
                        disk["validity"] == "incomplete"
                        and disk["conditional_proposal"] is False
                        and disk.get("primary_error") == result.get("primary_error"),
                        "E164 fail-closed disk receipt differs",
                    )
                    failure_published = True
                    break
                except BaseException as rewrite_error:
                    secondary(rewrite_error, f"fail-closed publication {attempt + 1}")
            progress["failure_receipt_published"] = failure_published
    if primary is not None:
        raise primary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.freeze_sha256,
                args.source_review,
                args.source_review_sha256,
                args.out,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
