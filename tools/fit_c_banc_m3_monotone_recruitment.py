"""E148 saved-bank monotone single-knot recruitment SOCP."""

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
import fit_c_banc_m3_muscle_gain as gain
import fit_c_banc_m3_static_pool as static

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "0d739d5cf32c30faf67a2132688f82196c99227c6dbdb28c4011bdc378be65e2"
DESIGN_SHA = "0108a4ac5fbe5b0159cb4dc417e9d5d0787699fcfc793e63882dbab72eac9273"
SCHEMA = "flylab.m3.e148.monotone-recruitment.v1"
OWNED = (
    "tools/fit_c_banc_m3_monotone_recruitment.py",
    "tests/test_c_banc_monotone_recruitment.py",
)
LEGS = static.LEGS
DT = static.DT
COMMAND_GAIN = static.COMMAND_GAIN
MIDDLE_BRANCH = causal.MIDDLE_BRANCH

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
        f"Secondary E148 {operation} failure: {type(secondary).__name__}: {secondary}"
    )


def _expected_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "design_path": "docs/BANC_M3_E148_MONOTONE_RECRUITMENT_DESIGN_20260921.md",
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
            "time_limit": 20.0,
            "verbose": False,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "tol_infeas_abs": 1e-10,
            "tol_infeas_rel": 1e-10,
        },
        "margin_bounds": [-1.0, 1.0],
        "total_fit_limit_s": 20,
        "internal_wall_limit_s": 45,
        "external_wall_limit_s": 50,
        "supervisor_limit_s": 59,
        "prior_conservative_process_s": 2212,
        "prior_conservative_actual_s": 2211.2814538000384,
        "prospective_cumulative_limit_s": 2271,
        "original_program_cumulative_limit_s": 2271,
        "maximum_raw_bytes": 201326592,
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
        "parameter_order": "muscle/motor/filter pairs; [interleaved low-high slopes, lower envelopes, upper envelopes, margin]",
        "anchor_parameterization": False,
        "knot_rule": "max(1Hz, median of strictly positive original10ms values over2000 samples); empty positive set ->1Hz",
        "knot_minimum_hz": 1.0,
        "segments_per_branch": 2,
        "envelope_auxiliaries_per_pair": 2,
        "derived_basis_scalar_values": 4692000,
        "E147_inclusion_output_atol": 1e-12,
        "nonlinear_transfer": "nonnegative continuous monotone single-knot piecewise-linear slopes",
    }


def reviewed_inputs(
    freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(freeze_sha == FREEZE_SHA, "Exact E148 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E148 freeze file required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key, expected in _expected_contract().items():
        require(freeze.get(key) == expected, f"E148 frozen contract changed: {key}")
    require(sha256(ROOT / freeze["design_path"]) == DESIGN_SHA, "E148 design changed")
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
    require(sha256(review_path) == review_sha, "E148 source review SHA mismatch")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        review.get("accepts_single_monotone_recruitment_fit") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == FREEZE_SHA
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E148 two-source review required",
    )
    for relative, digest in review["source_sha256"].items():
        checked_path(relative, digest)
    return freeze, review


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def load_inputs(freeze: dict[str, Any]) -> dict[str, Any]:
    data = static.load_inputs(freeze)
    for label, suffix in (
        ("e146", "e146-causal-filter/causal-filter-raw.npz"),
        ("e147", "e147-muscle-gain/muscle-gain-raw.npz"),
    ):
        path = next(key for key in freeze["input_sha256"] if key.endswith(suffix))
        data[label] = _load_npz(ROOT / path)
    return data


def knot_and_basis(
    bank: Any,
    minimum_hz: float = 1.0,
    *,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build target-free motor knots and the continuous low/high hinge basis."""
    bank = np.asarray(bank, dtype=np.float64)
    require(bank.ndim == 3 and bank.shape[2] == 3, "E148 bank shape invalid")
    require(np.isfinite(bank).all() and np.all(bank >= 0), "E148 bank invalid")
    middle = bank[:, :, MIDDLE_BRANCH]
    kappa = np.empty(bank.shape[1], dtype=np.float64)
    for motor in range(bank.shape[1]):
        positive = middle[:, motor][middle[:, motor] > 0]
        median = float(np.median(positive)) if positive.size else minimum_hz
        kappa[motor] = max(float(minimum_hz), median)
    low = np.minimum(bank, kappa[None, :, None])
    high = np.maximum(bank - kappa[None, :, None], 0)
    basis = np.stack((low, high), axis=-1)
    if partial is not None:
        partial.update(kappa=kappa, recruitment_basis=basis)
    if progress is not None:
        progress.update(
            derived_knot_values=int(kappa.size),
            derived_basis_scalar_values=int(basis.size),
            derived_candidate_coefficients=0,
        )
    reconstruction_error = np.max(np.abs(basis.sum(axis=-1) - bank))
    reconstruction_scale = max(1.0, float(np.max(bank)))
    require(
        reconstruction_error <= 1e-15 * reconstruction_scale,
        "E148 hinge basis does not reconstruct bank",
    )
    return kappa, basis


def validate_saved_inputs(
    data: dict[str, Any],
    freeze: dict[str, Any],
    *,
    base_validator: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = static.validate_saved_inputs,
    partial: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw146 = data["e146"]
    raw147 = data["e147"]
    if partial is not None:
        partial.update(
            unvalidated_filter_bank=np.asarray(raw146.get("filter_bank", [])),
            unvalidated_filter_tau_s=np.asarray(raw146.get("filter_tau_s", [])),
            unvalidated_e147_candidate_weights=np.asarray(
                raw147.get("candidate_weights", [])
            ),
            unvalidated_e147_candidate_activation=np.asarray(
                raw147.get("candidate_activation", [])
            ),
            unvalidated_e147_candidate_selected_command_rad=np.asarray(
                raw147.get("candidate_selected_command_rad", [])
            ),
            unvalidated_e147_candidate_selected_velocity_rad_s=np.asarray(
                raw147.get("candidate_selected_velocity_rad_s", [])
            ),
            unvalidated_e147_candidate_startup_command_rad=np.asarray(
                raw147.get("candidate_startup_command_rad", [])
            ),
            unvalidated_e147_candidate_startup_tendon=np.asarray(
                raw147.get("candidate_startup_tendon", [])
            ),
        )
    saved = base_validator(data, freeze)
    bank = static.finite_array(raw146["filter_bank"], (2000, 391, 3), "E146 bank")
    tau = static.finite_array(raw146["filter_tau_s"], (3,), "E146 tau")
    require(np.array_equal(tau, np.asarray(freeze["tau_s"])), "E148 tau changed")
    require(np.array_equal(bank[:, :, 1], saved["b"]), "E148 middle bank changed")
    exact_pairs = {
        "baseline_pool": "pool",
        "membership": "membership",
        "moment": "moment",
        "rest": "rest",
        "ltm": "ltm",
        "baseline_endpoint_rad": "baseline",
        "target_indices": "target_indices",
        "target_rad": "target",
        "velocity_pair_start_indices": "pair_start",
        "velocity_pair_end_indices": "pair_end",
        "target_velocity_rad_s": "target_velocity",
        "startup_target_rad": "startup_target",
        "startup_target_tendon": "startup_tendon_target",
    }
    for label, raw in (("E146", raw146), ("E147", raw147)):
        for raw_key, saved_key in exact_pairs.items():
            require(
                np.array_equal(raw[raw_key], saved[saved_key]),
                f"E148 {label} raw differs for {raw_key}",
            )
    require(
        np.array_equal(raw147["filter_bank"], bank)
        and np.array_equal(raw147["filter_tau_s"], tau),
        "E148 E147 reused filter metadata changed",
    )
    if partial is not None:
        partial.update(filter_bank=bank, filter_tau_s=tau)
    if progress is not None:
        progress["reused_filter_vector_updates"] = int(bank.shape[0])
    kappa, basis = knot_and_basis(
        bank,
        freeze["knot_minimum_hz"],
        partial=partial,
        progress=progress,
    )
    if partial is not None:
        partial.update(kappa=kappa, recruitment_basis=basis)
    if progress is not None:
        progress.update(
            derived_knot_values=int(kappa.size),
            derived_basis_scalar_values=int(basis.size),
            derived_candidate_coefficients=0,
        )
    witness147 = static.finite_array(
        raw147["candidate_weights"], (96, 391, 3), "E147 candidate weights"
    )
    witness147_outputs = {
        "activation": static.finite_array(
            raw147["candidate_activation"],
            (2000, 96),
            "E147 saved candidate activation",
        ),
        "selected_command_rad": static.finite_array(
            raw147["candidate_selected_command_rad"],
            (1827, 42),
            "E147 saved selected command",
        ),
        "selected_velocity_rad_s": static.finite_array(
            raw147["candidate_selected_velocity_rad_s"],
            (1826, 42),
            "E147 saved selected velocity",
        ),
        "startup_command_rad": static.finite_array(
            raw147["candidate_startup_command_rad"],
            (99, 42),
            "E147 saved startup command",
        ),
        "startup_tendon": static.finite_array(
            raw147["candidate_startup_tendon"],
            (99, 6),
            "E147 saved startup tendon",
        ),
    }
    if partial is not None:
        partial.update(
            {
                f"e147_saved_candidate_{key}": value
                for key, value in witness147_outputs.items()
            }
        )
    saved.update(
        bank=bank,
        tau=tau,
        kappa=kappa,
        basis=basis,
        witness147=witness147,
        witness147_outputs=witness147_outputs,
    )
    return saved


def publish_derived_counts(progress: dict[str, Any], saved: dict[str, Any]) -> None:
    """Publish completed target-independent derivations for failure receipts."""
    progress.update(
        reused_filter_vector_updates=int(saved["bank"].shape[0]),
        derived_knot_values=int(saved["kappa"].size),
        derived_basis_scalar_values=int(saved["basis"].size),
        derived_candidate_coefficients=0,
    )


@dataclass(frozen=True)
class RecruitmentMap:
    pair_flat: np.ndarray
    pair_motors: np.ndarray
    pair_branches: np.ndarray
    pair_muscles: np.ndarray
    fitted_muscle_ids: np.ndarray
    fixed_muscles: np.ndarray
    coefficient_low_indices: np.ndarray
    coefficient_high_indices: np.ndarray
    coefficient_slice: slice
    lower_slice: slice
    upper_slice: slice
    gain_rows: sparse.csr_matrix
    coefficient_lower: np.ndarray
    canonical_vector: np.ndarray
    lifted_vector: np.ndarray
    e147_vector: np.ndarray


def _transfer_vector(pair_slopes: np.ndarray) -> np.ndarray:
    pair_slopes = np.asarray(pair_slopes, dtype=np.float64)
    lower = np.min(pair_slopes, axis=1)
    upper = np.max(pair_slopes, axis=1)
    return np.concatenate((pair_slopes.reshape(-1), lower, upper))


def build_recruitment_map(
    pool: Any,
    membership: Any,
    ltm: Any,
    e147_weights: Any,
    *,
    guard_fraction: float = 1e-6,
) -> RecruitmentMap:
    pool = np.asarray(pool, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)
    e147_weights = np.asarray(e147_weights, dtype=np.float64)
    base = gain.build_gain_map(
        pool, membership, ltm, guard_fraction=guard_fraction, upper_bound=2.0
    )
    pair_flat = base.variable_flat
    pair_motors = base.variable_motors
    pair_branches = base.variable_branches
    pair_muscles = base.variable_muscle_for_coefficient
    n = pair_flat.size
    low_indices = 2 * np.arange(n, dtype=np.int64)
    high_indices = low_indices + 1
    canonical_pair = base.canonical_vector
    lifted_pair = base.lifted_vector
    e147_pair = e147_weights[pair_muscles, pair_motors, pair_branches]
    canonical = _transfer_vector(np.column_stack((canonical_pair, canonical_pair)))
    lifted = _transfer_vector(np.column_stack((lifted_pair, lifted_pair)))
    witness147 = _transfer_vector(np.column_stack((e147_pair, e147_pair)))
    return RecruitmentMap(
        pair_flat,
        pair_motors,
        pair_branches,
        pair_muscles,
        base.variable_muscles,
        base.fixed_muscles,
        low_indices,
        high_indices,
        slice(0, 2 * n),
        slice(2 * n, 3 * n),
        slice(3 * n, 4 * n),
        base.gain_rows,
        base.lower,
        canonical,
        lifted,
        witness147,
    )


def canonical_transfer_weights(pool: Any) -> np.ndarray:
    base = causal.canonical_weights(pool)
    return np.repeat(base[..., None], 2, axis=-1)


def piecewise_response(values: Any, kappa: Any, slopes: Any) -> np.ndarray:
    """Evaluate the continuous two-slope transfer for matching final axes."""
    values = np.asarray(values, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    require(slopes.shape[-1] == 2, "E148 slope segment axis invalid")
    low = np.minimum(values, kappa)
    high = np.maximum(values - kappa, 0.0)
    return slopes[..., 0] * low + slopes[..., 1] * high


def weights_from_vector(pool: Any, mapping: RecruitmentMap, vector: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    require(
        vector.shape == mapping.canonical_vector.shape, "E148 vector shape mismatch"
    )
    weights = canonical_transfer_weights(pool)
    slopes = vector[mapping.coefficient_slice].reshape(-1, 2)
    weights[
        mapping.pair_muscles,
        mapping.pair_motors,
        mapping.pair_branches,
        :,
    ] = slopes
    return weights


def activation_from_weights(
    saved: dict[str, Any], weights: np.ndarray, mapping: RecruitmentMap
) -> np.ndarray:
    """Evaluate the hinge form with an exact old-linear floating path."""
    canonical_activation = saved["b"] @ saved["pool"].T
    activation = canonical_activation.copy()
    for muscle in mapping.fitted_muscle_ids:
        members = np.flatnonzero(saved["membership"] == muscle)
        slopes = weights[muscle, members]
        low_slope = slopes[:, :, 0]
        slope_delta = slopes[:, :, 1] - low_slope
        linear = np.einsum(
            "tik,ik->t",
            saved["bank"][:, members],
            low_slope,
            optimize=False,
        )
        hinge = np.einsum(
            "tik,ik->t",
            saved["basis"][:, members][:, :, :, 1],
            slope_delta,
            optimize=False,
        )
        activation[:, muscle] = linear + hinge
    return activation


def activation_outputs(saved: dict[str, Any], activation: Any) -> dict[str, np.ndarray]:
    """Evaluate frozen physical outputs using the E146/E147 operation order."""
    activation = np.asarray(activation, dtype=np.float64)
    command = saved["rest"] + COMMAND_GAIN * (activation @ saved["moment"].T)
    return {
        "activation": activation,
        "complete_command_rad": command,
        "selected_command_rad": command[saved["target_indices"] - 1],
        "selected_velocity_rad_s": np.diff(command, axis=0)[saved["pair_end"] - 2] / DT,
        "startup_command_rad": command[:99],
        "startup_tendon": 0.2 * np.tanh(activation[:99, saved["ltm"]] / 10),
    }


def evaluate_composite(
    saved: dict[str, Any], weights: np.ndarray | None
) -> dict[str, Any]:
    mapping = saved["anchor"]
    variants: dict[str, np.ndarray | None] = {"baseline": None}
    if weights is not None:
        variants["candidate"] = weights
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    canonical_activation = saved["b"] @ saved["pool"].T
    for name, current in variants.items():
        activation = (
            canonical_activation
            if current is None
            else activation_from_weights(saved, current, mapping)
        )
        outputs = activation_outputs(saved, activation)
        arrays.update({f"{name}_{key}": value for key, value in outputs.items()})
        selected = outputs["selected_command_rad"]
        velocity = outputs["selected_velocity_rad_s"]
        startup = outputs["startup_command_rad"]
        tendon = outputs["startup_tendon"]
        command_error = selected - saved["target"]
        velocity_error = velocity - saved["target_velocity"]
        startup_error = startup - saved["startup_target"]
        tendon_error = tendon - saved["startup_tendon_target"]
        metrics[name] = {
            "command_RMS": float(np.sqrt(np.mean(command_error**2))),
            "velocity_RMS": float(np.sqrt(np.mean(velocity_error**2))),
            "startup_q_RMS": float(np.sqrt(np.mean(startup_error**2))),
            "startup_q_max_abs": float(np.max(np.abs(startup_error))),
            "startup_LTM_RMS": float(np.sqrt(np.mean(tendon_error**2))),
            "startup_LTM_max_abs": float(
                np.max(np.abs(tendon_error)),
            ),
            "by_leg": {
                leg: {
                    "command_RMS": float(
                        np.sqrt(np.mean(command_error[:, 7 * i : 7 * (i + 1)] ** 2))
                    ),
                    "velocity_RMS": float(
                        np.sqrt(np.mean(velocity_error[:, 7 * i : 7 * (i + 1)] ** 2))
                    ),
                }
                for i, leg in enumerate(LEGS)
            },
        }
    return {"arrays": arrays, "metrics": metrics}


def check_e147_linear_inclusion(
    saved: dict[str, Any],
    mapping: RecruitmentMap,
    tolerance: float,
    raw: dict[str, np.ndarray],
    partial: dict[str, Any] | None = None,
) -> None:
    """Check the corrected hinge path against actual saved E147 outputs."""
    saved_outputs = saved["witness147_outputs"]
    direct_activation = causal.activation_from_weights(
        saved["bank"],
        saved["witness147"],
        saved["b"] @ saved["pool"].T,
        mapping.fitted_muscle_ids,
    )
    transfer_activation = activation_from_weights(
        saved,
        weights_from_vector(saved["pool"], mapping, mapping.e147_vector),
        mapping,
    )
    reference_outputs = activation_outputs(saved, saved_outputs["activation"])
    transfer_outputs = activation_outputs(saved, transfer_activation)
    comparisons = {
        "direct_activation_vs_saved": (
            saved_outputs["activation"],
            direct_activation,
        ),
        "saved_selected_command_reconstruction": (
            saved_outputs["selected_command_rad"],
            reference_outputs["selected_command_rad"],
        ),
        "saved_selected_velocity_reconstruction": (
            saved_outputs["selected_velocity_rad_s"],
            reference_outputs["selected_velocity_rad_s"],
        ),
        "saved_startup_command_reconstruction": (
            saved_outputs["startup_command_rad"],
            reference_outputs["startup_command_rad"],
        ),
        "saved_startup_tendon_reconstruction": (
            saved_outputs["startup_tendon"],
            reference_outputs["startup_tendon"],
        ),
        "activation": (saved_outputs["activation"], transfer_activation),
        "complete_command": (
            reference_outputs["complete_command_rad"],
            transfer_outputs["complete_command_rad"],
        ),
        "selected_command": (
            saved_outputs["selected_command_rad"],
            transfer_outputs["selected_command_rad"],
        ),
        "selected_velocity": (
            saved_outputs["selected_velocity_rad_s"],
            transfer_outputs["selected_velocity_rad_s"],
        ),
        "startup_command": (
            saved_outputs["startup_command_rad"],
            transfer_outputs["startup_command_rad"],
        ),
        "startup_tendon": (
            saved_outputs["startup_tendon"],
            transfer_outputs["startup_tendon"],
        ),
    }
    errors: dict[str, float] = {}
    diagnostics: dict[str, np.ndarray] = {}
    for name, (reference, observed) in comparisons.items():
        difference = np.abs(np.asarray(observed) - np.asarray(reference))
        flat_index = int(np.argmax(difference))
        errors[name] = float(difference.flat[flat_index])
        diagnostics[f"e147_inclusion_{name}_max_abs_error"] = np.asarray(errors[name])
        diagnostics[f"e147_inclusion_{name}_max_abs_index"] = np.asarray(
            np.unravel_index(flat_index, difference.shape), dtype=np.int64
        )
    raw.update(diagnostics)
    if partial is not None:
        partial.update(diagnostics)
    require(
        max(errors.values()) <= tolerance,
        "E148 does not include all saved E147 outputs within tolerance: "
        + ", ".join(f"{name}={value:.17g}" for name, value in errors.items()),
    )


def fixed_activation(saved: dict[str, Any], mapping: RecruitmentMap) -> np.ndarray:
    activation = saved["b"] @ saved["pool"].T
    activation[:, mapping.fitted_muscle_ids] = 0.0
    return activation


def _embed_local(matrix: Any, indices: np.ndarray, width: int) -> sparse.csr_matrix:
    local = sparse.csr_matrix(matrix, dtype=np.float64)
    selector = sparse.csr_matrix(
        (np.ones(indices.size), (np.arange(indices.size), indices)),
        shape=(indices.size, width),
    )
    return local @ selector


def _leg_affine(
    saved: dict[str, Any], data: dict[str, Any], mapping: RecruitmentMap, leg: str
) -> dict[str, Any]:
    layout = static.build_leg_layout(
        leg,
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        data["anatomy"],
        saved["moment"],
    )
    local_pairs = np.flatnonzero(np.isin(mapping.pair_muscles, layout.variable_muscles))
    local_coefficients = np.column_stack(
        (
            mapping.coefficient_low_indices[local_pairs],
            mapping.coefficient_high_indices[local_pairs],
        )
    ).reshape(-1)
    signal = saved["basis"][
        :, mapping.pair_motors[local_pairs], mapping.pair_branches[local_pairs], :
    ].reshape(saved["basis"].shape[0], -1)
    coefficient_muscles = np.repeat(mapping.pair_muscles[local_pairs], 2)
    moment = saved["moment"][np.ix_(layout.joint_columns, coefficient_muscles)].T
    design = (COMMAND_GAIN * signal[:, :, None] * moment[None, :, :]).transpose(0, 2, 1)
    offset = saved["rest"][layout.joint_columns] + COMMAND_GAIN * (
        saved["fixed_activation"] @ saved["moment"][layout.joint_columns].T
    )
    endpoints = saved["target_indices"] - 1
    return {
        "layout": layout,
        "coefficient_indices": local_coefficients,
        "command_matrix": design[endpoints].reshape(-1, local_coefficients.size),
        "command_offset": (
            offset[endpoints] - saved["target"][:, layout.joint_columns]
        ).reshape(-1),
        "velocity_matrix": (
            np.diff(design, axis=0)[saved["pair_end"] - 2] / DT
        ).reshape(-1, local_coefficients.size),
        "velocity_offset": (
            np.diff(offset, axis=0)[saved["pair_end"] - 2] / DT
            - saved["target_velocity"][:, layout.joint_columns]
        ).reshape(-1),
        "startup_matrix": design[:99].reshape(-1, local_coefficients.size),
        "startup_offset": (
            offset[:99] - saved["startup_target"][:, layout.joint_columns]
        ).reshape(-1),
    }


def _make_nonnegative_rows(
    mapping: RecruitmentMap,
    startup_matrix: sparse.csr_matrix,
    startup_offset: np.ndarray,
    startup_max: float,
    margin_bounds: tuple[float, float],
    solver_gain_bounds: tuple[float, float],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = mapping.pair_flat.size
    width = 4 * n + 1

    def block(matrix: sparse.spmatrix, columns: np.ndarray) -> sparse.csr_matrix:
        return _embed_local(matrix, columns, width)

    coeff = np.arange(2 * n)
    ell = np.arange(2 * n, 3 * n)
    upper = np.arange(3 * n, 4 * n)
    eye2 = sparse.eye(2 * n, format="csr")
    eyen = sparse.eye(n, format="csr")
    rows = [block(-eye2, coeff), block(eye2, coeff)]
    rhs = [-np.repeat(mapping.coefficient_lower, 2), np.full(2 * n, 2.0)]
    rows.extend(
        [block(-eyen, ell), block(eyen, ell), block(-eyen, upper), block(eyen, upper)]
    )
    rhs.extend([np.zeros(n), np.full(n, 2.0), np.zeros(n), np.full(n, 2.0)])
    select_low = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), mapping.coefficient_low_indices)),
        shape=(n, 2 * n),
    )
    select_high = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), mapping.coefficient_high_indices)),
        shape=(n, 2 * n),
    )
    envelope_rows = (
        block(-select_low, coeff) + block(eyen, ell),
        block(-select_high, coeff) + block(eyen, ell),
        block(select_low, coeff) - block(eyen, upper),
        block(select_high, coeff) - block(eyen, upper),
    )
    rows.extend(envelope_rows)
    rhs.extend([np.zeros(n) for _ in envelope_rows])
    gain_low, gain_high = solver_gain_bounds
    rows.extend([block(-mapping.gain_rows, ell), block(mapping.gain_rows, upper)])
    rhs.extend(
        [
            np.full(mapping.fitted_muscle_ids.size, -gain_low),
            np.full(mapping.fitted_muscle_ids.size, gain_high),
        ]
    )
    t = 4 * n
    low_t, high_t = margin_bounds
    rows.extend(
        [
            sparse.csr_matrix(([-1.0], ([0], [t])), shape=(1, width)),
            sparse.csr_matrix(([1.0], ([0], [t])), shape=(1, width)),
        ]
    )
    rhs.extend([np.asarray([-low_t]), np.asarray([high_t])])
    t_column = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    startup_full = sparse.hstack(
        (startup_matrix, sparse.csr_matrix((startup_matrix.shape[0], 2 * n)))
    )
    rows.extend(
        [
            sparse.hstack((startup_full / startup_max, t_column)),
            sparse.hstack((-startup_full / startup_max, t_column)),
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
        "kappa": saved["kappa"],
        "recruitment_basis": saved["basis"],
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
        **{
            f"e147_saved_candidate_{key}": value
            for key, value in saved["witness147_outputs"].items()
        },
        "pair_flat": mapping.pair_flat,
        "pair_motors": mapping.pair_motors,
        "pair_branches": mapping.pair_branches,
        "pair_muscles": mapping.pair_muscles,
        "fitted_muscle_ids": mapping.fitted_muscle_ids,
        "fixed_muscle_ids": mapping.fixed_muscles,
        "fixed_muscle_mask": np.isin(np.arange(96), mapping.fixed_muscles),
        "coefficient_low_indices": mapping.coefficient_low_indices,
        "coefficient_high_indices": mapping.coefficient_high_indices,
        "coefficient_slice": np.asarray([0, mapping.coefficient_slice.stop]),
        "lower_slice": np.asarray(
            [mapping.lower_slice.start, mapping.lower_slice.stop]
        ),
        "upper_slice": np.asarray(
            [mapping.upper_slice.start, mapping.upper_slice.stop]
        ),
        "row_sum_matrix": mapping.gain_rows.toarray(),
        "weight_lower_bounds": mapping.coefficient_lower,
        "canonical_vector": mapping.canonical_vector,
        "lifted_vector": mapping.lifted_vector,
        "e147_witness_vector": mapping.e147_vector,
        "canonical_weights": weights_from_vector(
            saved["pool"], mapping, mapping.canonical_vector
        ),
        "lifted_weights": weights_from_vector(
            saved["pool"], mapping, mapping.lifted_vector
        ),
        "e147_witness_weights": weights_from_vector(
            saved["pool"], mapping, mapping.e147_vector
        ),
        "box_lower": problem.lower,
        "box_upper": problem.upper,
        "criterion_names": np.asarray([item.name for item in problem.criteria]),
        "criterion_limits": np.asarray([item.limit for item in problem.criteria]),
    }


def _mapping_arrays(
    mapping: RecruitmentMap, saved: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Return mapping and witness evidence available before affine construction."""
    return {
        "pair_flat": mapping.pair_flat,
        "pair_motors": mapping.pair_motors,
        "pair_branches": mapping.pair_branches,
        "pair_muscles": mapping.pair_muscles,
        "fitted_muscle_ids": mapping.fitted_muscle_ids,
        "fixed_muscle_ids": mapping.fixed_muscles,
        "fixed_muscle_mask": np.isin(np.arange(96), mapping.fixed_muscles),
        "coefficient_low_indices": mapping.coefficient_low_indices,
        "coefficient_high_indices": mapping.coefficient_high_indices,
        "coefficient_slice": np.asarray([0, mapping.coefficient_slice.stop]),
        "lower_slice": np.asarray(
            [mapping.lower_slice.start, mapping.lower_slice.stop]
        ),
        "upper_slice": np.asarray(
            [mapping.upper_slice.start, mapping.upper_slice.stop]
        ),
        "row_sum_matrix": mapping.gain_rows.toarray(),
        "weight_lower_bounds": mapping.coefficient_lower,
        "canonical_vector": mapping.canonical_vector,
        "lifted_vector": mapping.lifted_vector,
        "e147_witness_vector": mapping.e147_vector,
        "canonical_weights": weights_from_vector(
            saved["pool"], mapping, mapping.canonical_vector
        ),
        "lifted_weights": weights_from_vector(
            saved["pool"], mapping, mapping.lifted_vector
        ),
        "e147_witness_weights": weights_from_vector(
            saved["pool"], mapping, mapping.e147_vector
        ),
    }


def _verify_witness_feasibility(
    problem: conic.ConicProblem,
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    partial: dict[str, Any] | None = None,
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
    if partial is not None:
        partial.update(
            lifted_baseline_t=raw["lifted_baseline_t"],
            lifted_baseline_cone_margins=raw["lifted_baseline_cone_margins"],
        )
    require(np.min(margins) >= -1e-10, "E148 lifted witness infeasible")


def build_conic_problem(
    saved: dict[str, Any],
    data: dict[str, Any],
    freeze: dict[str, Any],
    *,
    partial: dict[str, Any] | None = None,
) -> tuple[conic.ConicProblem, dict[str, np.ndarray]]:
    mapping = build_recruitment_map(
        saved["pool"],
        saved["membership"],
        saved["ltm"],
        saved["witness147"],
        guard_fraction=freeze["branch_guard_fraction"],
    )
    saved["anchor"] = mapping
    saved["fixed_activation"] = fixed_activation(saved, mapping)
    raw: dict[str, np.ndarray] = _mapping_arrays(mapping, saved)
    if partial is not None:
        partial.update(raw)
    legs = [_leg_affine(saved, data, mapping, leg) for leg in LEGS]
    anatomy_muscles = np.concatenate([leg["layout"].muscles for leg in legs])
    require(
        np.array_equal(np.sort(anatomy_muscles), np.arange(96))
        and np.unique(anatomy_muscles).size == 96,
        "E148 anatomy does not partition all muscles exactly once",
    )
    baseline = evaluate_composite(saved, None)["metrics"]["baseline"]
    n_pair = mapping.pair_flat.size
    n_parameter = 4 * n_pair
    criteria: list[conic.AffineCriterion] = []

    def add(name: str, blocks: list[tuple[Any, Any, Any]], limit: float) -> None:
        compressed_blocks = []
        compressed_offsets = []
        direct_blocks = []
        direct_offsets = []
        for block_index, (matrix, offset, indices) in enumerate(blocks):
            r = thin_qr_compress(matrix, offset)
            r_matrix, r_offset = compressed_affine(r)
            compressed_blocks.append(_embed_local(r_matrix, indices, n_parameter))
            compressed_offsets.append(r_offset)
            direct_blocks.append(_embed_local(matrix, indices, n_parameter))
            direct_offsets.append(offset)
            raw[f"qr_{name}_{block_index}"] = r
            if partial is not None:
                partial[f"qr_{name}_{block_index}"] = r
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
        if partial is not None:
            partial[f"criterion_{name}_limit"] = raw[f"criterion_{name}_limit"]
        for suffix, vector in (
            ("canonical", mapping.canonical_vector),
            ("lifted", mapping.lifted_vector),
            ("e147", mapping.e147_vector),
        ):
            direct_norm = np.linalg.norm(direct_matrix @ vector + direct_offset)
            compressed_norm = np.linalg.norm(
                compressed_matrix @ vector + compressed_offset
            )
            raw[f"criterion_{name}_{suffix}_direct_norm"] = np.asarray(direct_norm)
            raw[f"criterion_{name}_{suffix}_compressed_norm"] = np.asarray(
                compressed_norm
            )
            if partial is not None:
                partial.update(
                    {
                        f"criterion_{name}_{suffix}_direct_norm": raw[
                            f"criterion_{name}_{suffix}_direct_norm"
                        ],
                        f"criterion_{name}_{suffix}_compressed_norm": raw[
                            f"criterion_{name}_{suffix}_compressed_norm"
                        ],
                    }
                )
            tolerance = (
                freeze["QR_absolute_atol"] + freeze["QR_relative_atol"] * direct_norm
            )
            require(
                abs(direct_norm - compressed_norm) <= tolerance,
                f"E148 {name} QR mismatch",
            )

    command_blocks = [
        (leg["command_matrix"], leg["command_offset"], leg["coefficient_indices"])
        for leg in legs
    ]
    velocity_blocks = [
        (leg["velocity_matrix"], leg["velocity_offset"], leg["coefficient_indices"])
        for leg in legs
    ]
    startup_blocks = [
        (leg["startup_matrix"], leg["startup_offset"], leg["coefficient_indices"])
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
    require(len(criteria) == 15, "E148 SOC count changed")
    startup_coeff = sparse.vstack(
        [
            _embed_local(matrix, indices, 2 * n_pair)
            for matrix, _offset, indices in startup_blocks
        ],
        format="csr",
    )
    startup_offset = np.concatenate(
        [offset for _matrix, offset, _indices in startup_blocks]
    )
    nonnegative_A, nonnegative_b = _make_nonnegative_rows(
        mapping,
        startup_coeff,
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
            criterion.matrix, criterion.offset, criterion.limit, n_parameter
        )
        A_blocks.append(rows)
        b_blocks.append(rhs)
        cones.append(clarabel.SecondOrderConeT(rows.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(rows.shape[0])
    q = np.zeros(n_parameter + 1)
    q[-1] = -1
    startup_parameter_matrix = sparse.hstack(
        (startup_coeff, sparse.csr_matrix((startup_coeff.shape[0], 2 * n_pair))),
        format="csr",
    )
    problem = conic.ConicProblem(
        sparse.csc_matrix((n_parameter + 1, n_parameter + 1)),
        q,
        sparse.vstack(A_blocks, format="csc"),
        np.concatenate(b_blocks),
        tuple(cones),
        tuple(kinds),
        np.asarray(dims),
        np.concatenate((np.zeros(n_parameter), [-1.0])),
        np.concatenate((np.full(n_parameter, 2.0), [1.0])),
        mapping,
        tuple(criteria),
        startup_parameter_matrix,
        startup_offset,
    )
    raw.update(_problem_arrays(problem, saved))
    if partial is not None:
        partial.update(raw)
    for name, vector in (
        ("canonical", mapping.canonical_vector),
        ("lifted", mapping.lifted_vector),
        ("e147", mapping.e147_vector),
    ):
        weights = weights_from_vector(saved["pool"], mapping, vector)
        composite = evaluate_composite(saved, weights)
        delta = (
            composite["arrays"]["candidate_selected_command_rad"]
            - composite["arrays"]["baseline_selected_command_rad"]
        )
        raw[f"{name}_minus_canonical_selected_command_rad"] = delta
        raw[f"{name}_minus_canonical_command_max_abs"] = np.asarray(
            np.max(np.abs(delta))
        )
        if partial is not None:
            partial.update(
                {
                    f"{name}_minus_canonical_selected_command_rad": raw[
                        f"{name}_minus_canonical_selected_command_rad"
                    ],
                    f"{name}_minus_canonical_command_max_abs": raw[
                        f"{name}_minus_canonical_command_max_abs"
                    ],
                }
            )
    check_e147_linear_inclusion(
        saved,
        mapping,
        freeze["E147_inclusion_output_atol"],
        raw,
        partial,
    )
    _verify_witness_feasibility(problem, freeze, raw, partial)
    return problem, raw


def physical_budgets(
    weights: np.ndarray, mapping: RecruitmentMap, muscle_count: int = 96
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slopes = weights[
        mapping.pair_muscles,
        mapping.pair_motors,
        mapping.pair_branches,
        :,
    ]
    pair_lower = np.min(slopes, axis=1)
    pair_upper = np.max(slopes, axis=1)
    lower = np.ones(muscle_count)
    upper = np.ones(muscle_count)
    for muscle in mapping.fitted_muscle_ids:
        mask = mapping.pair_muscles == muscle
        lower[muscle] = np.sum(pair_lower[mask])
        upper[muscle] = np.sum(pair_upper[mask])
    return pair_lower, lower, upper


def assess_physical_candidate(
    saved: dict[str, Any],
    weights: np.ndarray,
    lower_budget: np.ndarray,
    upper_budget: np.ndarray,
    composite: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    mapping = saved["anchor"]
    canonical = canonical_transfer_weights(saved["pool"])
    support = np.zeros_like(weights, dtype=bool)
    support[saved["membership"], np.arange(saved["membership"].size), :, :] = True
    _pair_lower, physical_lower, physical_upper = physical_budgets(weights, mapping)
    actual_motor_min_slopes = np.min(
        weights[saved["membership"], np.arange(saved["membership"].size)], axis=2
    ).sum(axis=1)
    motor_floors = np.asarray(
        [
            freeze["original_MN_floor_fraction"]
            / np.count_nonzero(saved["membership"] == muscle)
            for muscle in saved["membership"]
        ]
    )
    motor_floor_margins = actual_motor_min_slopes - motor_floors
    fixed_exact = all(
        np.array_equal(weights[muscle], canonical[muscle])
        for muscle in mapping.fixed_muscles
    )
    candidate = composite["metrics"]["candidate"]
    baseline = composite["metrics"]["baseline"]
    arrays = composite["arrays"]
    tolerance = freeze["lower_bound_atol"]
    limits = freeze["startup_limits"]
    gates = {
        "finite_weights_and_budgets": bool(
            np.isfinite(weights).all()
            and np.isfinite(lower_budget).all()
            and np.isfinite(upper_budget).all()
        ),
        "same_anatomical_support": bool(np.all(weights[~support] == 0)),
        "all_slopes_nonnegative": bool(np.all(weights[support] >= 0)),
        "authoritative_lower_budget_matches_weights": bool(
            np.array_equal(lower_budget, physical_lower)
        ),
        "authoritative_upper_budget_matches_weights": bool(
            np.array_equal(upper_budget, physical_upper)
        ),
        "muscle_lower_budget": bool(
            np.all(physical_lower[mapping.fitted_muscle_ids] >= 0.5 - tolerance)
        ),
        "muscle_upper_budget": bool(
            np.all(physical_upper[mapping.fitted_muscle_ids] <= 2.0 + tolerance)
        ),
        "original_motor_min_slope_floor": bool(
            np.min(motor_floor_margins) >= -tolerance
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
        "conditional_monotone_recruitment_proposal": bool(all(gates.values())),
        "gates": gates,
        "minimum_slope": float(np.min(weights[support])),
        "minimum_motor_floor_margin": float(np.min(motor_floor_margins)),
        "minimum_lower_budget": float(np.min(physical_lower)),
        "maximum_upper_budget": float(np.max(physical_upper)),
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
        saved = validate_saved_inputs(data, freeze, partial=partial, progress=progress)
        publish_derived_counts(progress, saved)
        problem, pre_raw = build_conic_problem(saved, data, freeze, partial=partial)
    else:
        problem, pre_raw, saved = prepared
        publish_derived_counts(progress, saved)
    partial.update(pre_raw)
    progress["problem_built"] = True
    history: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        solution, history, settings = solve_conic_problem(
            problem,
            freeze,
            partial=partial,
            progress=progress,
            state=state,
            history=history,
        )
        partial.update(conic._bare_solution_arrays(solution))
        partial.update(conic._history_arrays(history))
    finally:
        progress["timings_s"]["fit"] = time.perf_counter() - started
        progress["fit_wall_s"] = progress["timings_s"]["fit"]
    require(progress["fit_wall_s"] <= freeze["total_fit_limit_s"], "E148 fit timeout")
    result_arrays = solution_arrays(problem, solution, history, freeze)
    raw = {**pre_raw, **result_arrays}
    raw["solver_settings_json"] = np.asarray(
        json.dumps(settings, sort_keys=True, allow_nan=False)
    )
    partial.update(result_arrays)
    vector = np.asarray(solution.x[:-1])
    weights = weights_from_vector(saved["pool"], problem.anchor_map, vector)
    pair_lower, lower_budget, upper_budget = physical_budgets(
        weights, problem.anchor_map
    )
    candidate_coefficients = vector[problem.anchor_map.coefficient_slice].reshape(-1, 2)
    pair_upper = np.max(candidate_coefficients, axis=1)
    solver_lower = vector[problem.anchor_map.lower_slice]
    solver_upper = vector[problem.anchor_map.upper_slice]
    candidate_raw = {
        "candidate_vector": vector,
        "candidate_coefficients": candidate_coefficients,
        "candidate_weights": weights,
        "candidate_pair_min_slope": pair_lower,
        "candidate_pair_max_slope": pair_upper,
        "candidate_lower_budget": lower_budget,
        "candidate_upper_budget": upper_budget,
        "candidate_solver_lower_aux": solver_lower,
        "candidate_solver_upper_aux": solver_upper,
        "returned_margin_t": np.asarray(solution.x[-1]),
    }
    raw.update(candidate_raw)
    partial.update(candidate_raw)
    progress["derived_candidate_coefficients"] = int(
        vector[problem.anchor_map.coefficient_slice].size
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
            "lower_budget": lower_budget,
            "upper_budget": upper_budget,
            "composite": composite,
        },
        problem,
    )


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def failure_summary(primary: BaseException, progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "flylab.m3.e148.monotone-recruitment-result.v1",
        "experiment": "M3-E148-monotone-recruitment",
        "status": "failed",
        "validity": "invalid_incomplete",
        "conditional_monotone_recruitment_proposal": False,
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
        "schema": "flylab.m3.e148.monotone-recruitment-execution.v1",
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
        for failed, label, error in (
            (
                elapsed > freeze["internal_wall_limit_s"],
                "internal_wall_limit_s",
                TimeoutError(f"E148 exceeded internal wall during {stage}"),
            ),
            (
                size is not None and size > freeze["maximum_raw_bytes"],
                "maximum_raw_bytes",
                RuntimeError(f"E148 exceeded raw byte limit during {stage}"),
            ),
        ):
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
            free >= freeze["minimum_free_disk_bytes"], "E148 insufficient free disk"
        )
        with threadpool_limits(limits=1):
            progress["thread_pools"] = threadpool_info()
            require(
                all(
                    int(item.get("num_threads", 1)) <= 1
                    for item in progress["thread_pools"]
                ),
                "E148 CPU thread pool exceeds one thread",
            )
            stage = time.perf_counter()
            data = load_inputs(freeze)
            progress["timings_s"]["input_load"] = time.perf_counter() - stage
            saved = validate_saved_inputs(
                data, freeze, partial=partial, progress=progress
            )
            publish_derived_counts(progress, saved)
            problem, pre_raw = build_conic_problem(saved, data, freeze, partial=partial)
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
                data, freeze, progress, partial, prepared=(problem, pre_raw, saved)
            )
            try:
                save_npz_new(output / "monotone-recruitment-raw.npz", raw)
            except BaseException as error:
                progress["failed_raw_operations"].append(
                    {
                        "operation": "save monotone-recruitment-raw.npz",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            raw_saved = True
            assessment = assess_physical_candidate(
                outcome["saved"],
                outcome["weights"],
                outcome["lower_budget"],
                outcome["upper_budget"],
                outcome["composite"],
                freeze,
            )
        proposal = bool(assessment["conditional_monotone_recruitment_proposal"])
        projected_bound = float(raw["dual_projected_margin_upper_bound"])
        verdict = (
            "conditional_monotone_recruitment_proposal"
            if proposal
            else "guarded_monotone_recruitment_refuted"
            if projected_bound < 0
            else "guarded_monotone_recruitment_inconclusive"
        )
        summary = {
            "schema": "flylab.m3.e148.monotone-recruitment-result.v1",
            "experiment": "M3-E148-monotone-recruitment",
            "status": "pass",
            "validity": "conditional_saved_E146_E147_single_global_SOCP",
            "hypothesis_verdict": verdict,
            "analysis": facts,
            "assessment": assessment,
            "conditional_monotone_recruitment_proposal": proposal,
            "candidate_adopted": False,
            "F1_F8": "pending",
            "F6_exposed": False,
            "active_UI_changed": False,
            "whole_goal_complete": False,
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
