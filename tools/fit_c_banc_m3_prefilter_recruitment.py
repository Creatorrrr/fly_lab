"""E197: one saved-signal fit of a causal prefilter recruitment hinge."""

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
from pathlib import Path
from typing import Any

for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
import fit_c_banc_m3_conic_pool as e145
import fit_c_banc_m3_onset_recruitment as e164
import fit_c_banc_m3_resource_attenuation as e168

SCHEMA = "flylab.m3.e197.prefilter-recruitment.v1"
GATE_SCHEMA = "flylab.m3.e197.parent-source-review.v1"
RESULT_SCHEMA = "flylab.m3.e197.result.v1"
OWNED = (
    "tools/fit_c_banc_m3_prefilter_recruitment.py",
    "tests/test_c_banc_prefilter_recruitment.py",
)
DESIGN = "docs/BANC_M3_E197_PREFILTER_RECRUITMENT_DESIGN_20260924.md"
LEGS = e164.LEGS
DT = 0.005
TAU = np.asarray([0.005, 0.01, 0.04], dtype=np.float64)
N_COEFF = 1980
N_PARAM = 3961
FIRST_CONE = 20682
PHYSICAL_ROWS = (11882, 12366)


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError("E197 " + message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(name: str) -> Path:
    require(isinstance(name, str) and bool(name), "path must be nonempty text")
    relative = Path(name)
    result = (ROOT / relative).resolve()
    require(
        not relative.is_absolute() and ".." not in relative.parts and result.is_relative_to(ROOT),
        "unsafe frozen path",
    )
    return result


def frozen_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for section in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze[section]
        require(isinstance(mapping, dict) and bool(mapping), "missing " + section)
        for name, digest in mapping.items():
            require(
                isinstance(digest, str) and len(digest) == 64
                and (name not in merged or merged[name] == digest),
                "conflicting frozen hash: " + name,
            )
            merged[name] = digest
    return merged


def validate_freeze(path: Path, digest: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Read only metadata and hashes. No original numerical array is loaded here."""
    require(len(digest) == 64 and sha256(path) == digest, "freeze digest differs")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == SCHEMA
        and freeze.get("design") == DESIGN
        and freeze.get("owned_files") == list(OWNED)
        and freeze.get("solver") == "Clarabel"
        and freeze.get("solver_version") == str(clarabel.__version__) == "0.11.1",
        "freeze identity differs",
    )
    require(
        sha256(repo_path(DESIGN)) == freeze.get("design_sha256"),
        "design changed",
    )
    base_path = repo_path(freeze["base_freeze"])
    require(
        freeze.get("base_freeze_sha256") == e164.FREEZE_SHA
        and sha256(base_path) == e164.FREEZE_SHA,
        "E164 base freeze differs",
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    require(
        base["schema"] == e164.SCHEMA
        and base["sample_contract"] == freeze["sample_contract"]
        and base["paths"] == freeze["paths"]
        and base["body_hash"] == freeze["body_hash"],
        "E164 sample provenance differs",
    )
    transfer, contract, limits = (
        freeze["transfer"], freeze["contract"], freeze["resource_limits"]
    )
    require(
        transfer["pair_count"] == 990
        and transfer["coefficient_count"] == N_COEFF
        and transfer["total_parameter_count"] == N_PARAM
        and transfer["first_cone_row_count"] == FIRST_CONE
        and transfer["unique_qr_blocks"] == 18
        and transfer["branch_order"] == ["low", "high"]
        and transfer["pair_order"] == "ascending_original_graph_column_then_filter"
        and transfer["initial_bank"] == "zero"
        and transfer["initial_tick_advanced"] is False
        and transfer["hinge_before_filter"] is True
        and transfer["command_anchor"] == "unanchored_rest_plus_moment_current_activation"
        and transfer["native_expression"] == "sum_positive_slopes_times_EMA_of_rate_hinge_branches"
        and transfer["baseline_witness"] == "none_no_claim_of_old_E150_inclusion",
        "prefilter transfer contract differs",
    )
    require(
        transfer["tau_s"] == TAU.tolist()
        and transfer["dt_s"] == DT
        and contract["margin_bounds"] == [-1.0, 1.0]
        and contract["coefficient_bounds"] == contract["envelope_bounds"] == [0.0, 2.0]
        and contract["inner_coefficient_guard"] == 1e-9
        and contract["inner_physical_guard"] == 1e-7
        and contract["physical_atol"] == 1e-12
        and contract["physical_dual_relaxation_rows"] == list(PHYSICAL_ROWS)
        and contract["proposal_command_RMS_ratio"] == 0.8
        and contract["tonic_lower_budget"] == 0.5
        and contract["total_upper_budget"] == 2.0
        and contract["motor_floor_fraction"] == 1e-6
        and contract["soc_count"] == 15
        and contract["startup_limits"] == {
            "q_RMS": 0.018, "q_max_abs": 0.04, "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
        "mathematical contract differs",
    )
    require(
        (
            limits["original_processes"], limits["automatic_retries"],
            limits["cpu_threads"], limits["gpu_count"], limits["fit_s"],
            limits["internal_s"], limits["external_s"], limits["supervisor_s"],
            limits["raw_bytes"], limits["free_bytes"],
        ) == (1, 0, 1, 0, 30, 95, 110, 120, 402653184, 2147483648)
        and freeze["prior_conservative_original_s"] == 4997.566624599716
        and freeze["prospective_cumulative_s"] == 5144
        and freeze["prior_allocated_bound_s"] == 5144
        and freeze["supervisor_reservation_s"] == 120
        and freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "resource/outcome contract differs",
    )
    merged = frozen_hashes(freeze)
    old_hashes = e164.merged_frozen_hashes(base)
    require(
        merged.get(freeze["base_freeze"]) == e164.FREEZE_SHA
        and all(merged.get(name) == value for name, value in old_hashes.items()),
        "base input hash coverage differs",
    )
    require(not set(OWNED).intersection(merged), "owned source must be gate-pinned")
    for name, expected in merged.items():
        require(sha256(repo_path(name)) == expected, "frozen hash changed: " + name)
    return freeze, base, merged


def validate_gate(path: Path, digest: str, freeze: dict[str, Any], freeze_digest: str) -> dict[str, Any]:
    require(len(digest) == 64 and sha256(path) == digest, "source review digest differs")
    gate = json.loads(path.read_text(encoding="utf-8"))
    require(
        gate.get("schema") == GATE_SCHEMA
        and gate.get("approved") is True
        and gate.get("blocking_defects") == []
        and gate.get("accepts_single_prefilter_recruitment_fit") is True
        and gate.get("original_processes_allowed") == 1
        and gate.get("automatic_retries") == 0
        and gate.get("freeze_sha256") == freeze_digest
        and gate.get("design_sha256") == freeze["design_sha256"]
        and gate.get("source_sha256") == {
            name: sha256(repo_path(name)) for name in OWNED
        },
        "source review blocks execution",
    )
    return gate


def snapshot(out: Path, freeze_path: Path, gate_path: Path) -> dict[str, str]:
    sources = {
        "design.md": repo_path(DESIGN), "freeze.json": freeze_path,
        "source-review.json": gate_path, "fitter.py": repo_path(OWNED[0]),
        "tests.py": repo_path(OWNED[1]),
    }
    directory = out / "source-snapshot"
    directory.mkdir()
    receipt = {}
    for name, path in sources.items():
        digest = sha256(path)
        (directory / name).write_bytes(path.read_bytes())
        require(sha256(directory / name) == digest, "snapshot differs: " + name)
        receipt[name] = digest
    return receipt


def prefilter_state(
    rate: Any, kappa: Any, tau: Any = TAU, dt: float = DT
) -> tuple[np.ndarray, np.ndarray]:
    """Exact held-input zero-state low/high banks; row zero is never advanced."""
    r, knot, times = np.asarray(rate), np.asarray(kappa), np.asarray(tau)
    require(
        r.ndim == 2 and r.shape[0] >= 1 and knot.shape == (r.shape[1],)
        and times.shape == (3,) and r.dtype == knot.dtype == times.dtype == np.float64
        and np.isfinite(r).all() and np.isfinite(knot).all()
        and np.isfinite(times).all() and np.all(r >= 0)
        and np.all(knot >= 0) and np.all(times > 0)
        and isinstance(dt, (float, int, np.floating)) and math.isfinite(dt) and dt > 0,
        "prefilter input invalid",
    )
    alpha = -np.expm1(-float(dt) / times)
    low = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float64)
    high = np.zeros_like(low)
    for tick in range(1, r.shape[0]):
        lo_input = np.minimum(r[tick], knot)
        hi_input = np.maximum(r[tick] - knot, 0.0)
        low[tick] = low[tick - 1] + alpha[None, :] * (
            lo_input[:, None] - low[tick - 1]
        )
        high[tick] = high[tick - 1] + alpha[None, :] * (
            hi_input[:, None] - high[tick - 1]
        )
    return low, high


def native_activation(
    low: Any, high: Any, slopes: Any, membership: Any,
    fixed: Any, original_activation: Any,
) -> np.ndarray:
    """Direct per-muscle equation, independent of the flattened affine basis."""
    lo, hi, weight = np.asarray(low), np.asarray(high), np.asarray(slopes)
    owner, fixed_index = np.asarray(membership), np.asarray(fixed)
    original = np.asarray(original_activation)
    require(
        lo.shape == hi.shape and lo.ndim == 3 and lo.shape[1:] == (391, 3)
        and weight.shape == (96, 391, 3, 2) and owner.shape == (391,)
        and original.shape == (lo.shape[0], 96)
        and all(x.dtype == np.float64 for x in (lo, hi, weight, original))
        and owner.dtype == fixed_index.dtype == np.int64
        and all(np.isfinite(x).all() for x in (lo, hi, weight, original))
        and np.all(lo >= 0) and np.all(hi >= 0)
        and np.all(original >= 0),
        "native activation input invalid",
    )
    result = np.zeros_like(original)
    result[:, fixed_index] = original[:, fixed_index]
    variable = np.setdiff1d(np.arange(96), fixed_index)
    for muscle in variable:
        motors = np.flatnonzero(owner == muscle)
        result[:, muscle] = np.sum(
            weight[muscle, motors, :, 0][None] * lo[:, motors]
            + weight[muscle, motors, :, 1][None] * hi[:, motors],
            axis=(1, 2),
        )
    return result


def prepare_transfer(
    data: dict[str, Any], saved: dict[str, Any], freeze: dict[str, Any],
    raw: dict[str, np.ndarray], progress: dict[str, Any],
) -> dict[str, Any]:
    contract = freeze["sample_contract"]
    rate = np.asarray(data["recruitment"]["motor_rate"])
    bank = np.asarray(data["recruitment"]["bank"])
    knot = np.asarray(data["parameters"]["kappa"])
    times = np.asarray(data["parameters"]["tau"])
    slopes = np.asarray(data["parameters"]["slopes"])
    owner = np.asarray(data["adapter"]["membership"])
    pool = np.asarray(data["adapter"]["pool"])
    motors = np.asarray(contract["variable_motors"], dtype=np.int64)
    require(
        rate.shape == (2001, 391) and rate.dtype == np.float64
        and bank.shape == (2001, 391, 3) and bank.dtype == np.float64
        and knot.shape == (391,) and knot.dtype == np.float64
        and times.dtype == np.float64 and np.array_equal(times, TAU)
        and slopes.shape == (96, 391, 3, 2) and slopes.dtype == np.float64
        and owner.shape == (391,) and owner.dtype == np.int64
        and pool.shape == (96, 391) and pool.dtype == np.float64
        and np.isfinite(rate).all() and np.all(rate >= 0)
        and np.array_equal(motors, np.sort(motors)) and len(motors) == 330
        and np.array_equal(
            motors, np.flatnonzero(np.isin(owner, contract["variable_muscles"]))
        ),
        "source rate/anatomy/clock invalid",
    )
    low, high = prefilter_state(rate, knot, times)
    progress["prefilter_update_steps"] = 2000
    alpha = -np.expm1(-DT / times)
    old_prediction = bank[:-1] + alpha[None, None, :] * (
        rate[1:, :, None] - bank[:-1]
    )
    old_residual = bank[1:] - old_prediction
    sum_residual = low + high - bank
    pair_motors = np.repeat(motors, 3)
    pair_filters = np.tile(np.arange(3), 330)
    pair_muscles = owner[pair_motors]
    selected_low = low[:, pair_motors, pair_filters]
    selected_high = high[:, pair_motors, pair_filters]
    basis = np.empty((2001, N_COEFF), dtype=np.float64)
    basis[:, 0::2] = selected_low
    basis[:, 1::2] = selected_high
    selected_bank = bank[:, pair_motors, pair_filters]
    old_basis = np.empty_like(basis)
    old_basis[:, 0::2] = np.minimum(selected_bank, knot[pair_motors])
    old_basis[:, 1::2] = np.maximum(selected_bank - knot[pair_motors], 0.0)
    w0 = slopes[pair_muscles, pair_motors, pair_filters].reshape(-1)
    canonical_pair = pool[pair_muscles, pair_motors] * (pair_filters == 1)
    canonical = np.repeat(canonical_pair, 2)
    saved.update(
        low=low, high=high, branch_basis=basis, old_branch_basis=old_basis,
        w0=w0, pair_motors=pair_motors, pair_filters=pair_filters,
        pair_muscles=pair_muscles, membership=owner, slopes=slopes,
        canonical=canonical, pool=pool,
        fixed_muscles=np.asarray(contract["fixed_muscles"], dtype=np.int64),
        group_legs=np.asarray(contract["group_legs"], dtype=np.int64),
        original_activation=np.asarray(data["recruitment"]["original_activation"]),
    )
    raw.update(
        endpoint_rate=rate, original_bank=bank, kappa=knot, tau=times,
        low_bank=low, high_bank=high, low_plus_high_minus_original_bank=sum_residual,
        original_bank_recurrence_residual=old_residual,
        pair_motors=pair_motors, pair_filters=pair_filters,
        pair_muscles=pair_muscles, pair_branches=np.tile(np.arange(2), 990),
        branch_basis=basis, old_postfilter_basis=old_basis,
        basis_difference=basis - old_basis, w0=w0,
        canonical_equal_branch=canonical,
        affine_moment=np.asarray(saved["moment"]),
        affine_rows_command=np.asarray(saved["target_indices"]),
        affine_rows_velocity_start=np.asarray(saved["pair_start"]),
        affine_rows_velocity_end=np.asarray(saved["pair_end"]),
        affine_rows_startup=np.arange(1, 100),
    )
    # Retain both the actual E150 baseline and the changed response at its w0.
    _, w0_arrays = candidate(saved, w0)
    raw.update(
        w0_prefilter_activation=w0_arrays["candidate_activation"],
        w0_prefilter_command=w0_arrays["candidate_command"],
        w0_command_minus_recorded=w0_arrays["candidate_command"]
        - saved["recorded_command"],
    )
    _, canonical_arrays = candidate(saved, canonical)
    canonical_linear = bank[:, :, 1] @ pool.T
    variable = np.asarray(contract["variable_muscles"], dtype=np.int64)
    canonical_residual = (
        canonical_arrays["candidate_activation"][:, variable]
        - canonical_linear[:, variable]
    )
    raw.update(
        canonical_equal_branch_activation=canonical_arrays["candidate_activation"],
        canonical_original_linear_activation=canonical_linear,
        canonical_linear_residual=canonical_residual,
        canonical_equal_branch_command=canonical_arrays["candidate_command"],
    )
    require(
        np.max(np.abs(old_residual)) <= 1e-12
        and np.all(np.abs(sum_residual) <= freeze["contract"]["raw_bank_sum_atol"]
                   + freeze["contract"]["raw_bank_sum_rtol"] * np.abs(bank))
        and np.all(low[0] == 0) and np.all(high[0] == 0)
        and np.array_equal(saved["original_activation"][0], np.zeros(96))
        and np.array_equal(saved["activation"][0], np.zeros(96)),
        "old/new state recurrence or zero initial state differs",
    )
    require(
        np.all(np.abs(canonical_residual) <= 1e-12 + 1e-12 * np.abs(
            canonical_linear[:, variable]
        )),
        "canonical equal-branch response differs from linear EMA",
    )
    return saved


def affine_activation(saved: dict[str, Any], weights: Any) -> np.ndarray:
    w = np.asarray(weights)
    require(w.shape == (N_COEFF,) and w.dtype == np.float64 and np.isfinite(w).all(),
            "coefficient vector invalid")
    activation = np.zeros_like(saved["activation"])
    fixed = saved["fixed_muscles"]
    activation[:, fixed] = saved["original_activation"][:, fixed]
    for muscle in np.setdiff1d(np.arange(96), fixed):
        columns = np.flatnonzero(np.repeat(saved["pair_muscles"], 2) == muscle)
        activation[:, muscle] = np.sum(
            saved["branch_basis"][:, columns] * w[columns][None, :], axis=1
        )
    return activation


def candidate(
    saved: dict[str, Any], weights: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    activation = affine_activation(saved, weights)
    command = np.empty_like(saved["recorded_command"])
    command[0] = saved["recorded_command"][0]
    for tick in range(1, command.shape[0]):
        command[tick] = saved["rest"] + 0.02 * (
            saved["moment"] @ activation[tick]
        )
    metrics, arrays = e164.metrics_from_recorded(saved, command)
    arrays.update(
        candidate_activation=activation, candidate_command=command,
        candidate_tendon=np.asarray(saved["tendon"]),
        decoder_residual=command[1:] - (
            saved["rest"] + 0.02 * (activation[1:] @ saved["moment"].T)
        ),
    )
    return metrics, arrays


def independent_candidate(saved: dict[str, Any], weights: Any) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.zeros_like(saved["slopes"])
    indices = (saved["pair_muscles"], saved["pair_motors"], saved["pair_filters"])
    slopes[indices] = np.asarray(weights).reshape(990, 2)
    activation = native_activation(
        saved["low"], saved["high"], slopes, saved["membership"],
        saved["fixed_muscles"], saved["original_activation"],
    )
    command = np.empty_like(saved["recorded_command"])
    command[0] = saved["recorded_command"][0]
    for tick in range(1, command.shape[0]):
        command[tick] = saved["rest"] + 0.02 * (saved["moment"] @ activation[tick])
    return activation, command


def local_affine(
    saved: dict[str, Any], leg: int, rows: np.ndarray, kind: str,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    owners = np.repeat(saved["pair_muscles"], 2)
    active = np.flatnonzero(saved["group_legs"][owners] == leg)
    joints = np.arange(7 * leg, 7 * (leg + 1))
    moment = saved["moment"][np.ix_(joints, owners[active])]
    basis = saved["branch_basis"]
    fixed = saved["fixed_muscles"]
    fixed_command = saved["rest"][joints][None, :] + 0.02 * (
        saved["original_activation"][:, fixed]
        @ saved["moment"][np.ix_(joints, fixed)].T
    )
    if kind == "command":
        values = basis[rows][:, active]
        offset = fixed_command[rows] - saved["target"][:, joints]
    elif kind == "velocity":
        values = (basis[saved["pair_end"]][:, active] - basis[saved["pair_start"]][:, active]) / DT
        offset = (
            fixed_command[saved["pair_end"]] - fixed_command[saved["pair_start"]]
        ) / DT - saved["target_velocity"][:, joints]
    elif kind == "startup":
        values = basis[rows][:, active]
        offset = fixed_command[rows] - saved["startup_target"][:, joints]
    else:
        raise ValueError("E197 unknown affine kind")
    direct = (0.02 * values[:, None, :] * moment[None, :, :]).reshape(
        len(rows) * 7, len(active)
    )
    return sparse.csr_matrix(direct), offset.reshape(-1), active


def compressed_blocks(
    saved: dict[str, Any], freeze: dict[str, Any], raw: dict[str, np.ndarray],
) -> tuple[dict[str, tuple[sparse.csr_matrix, np.ndarray]], sparse.csr_matrix, np.ndarray]:
    blocks: dict[str, tuple[sparse.csr_matrix, np.ndarray]] = {}
    startup_matrices, startup_offsets = [], []
    witnesses = (
        saved["w0"], np.zeros(N_COEFF),
        0.5 + 0.5 * np.cos(np.arange(N_COEFF) * 0.173),
    )
    for leg, name in enumerate(LEGS):
        for kind in ("command", "velocity", "startup"):
            rows = (
                saved["target_indices"] if kind == "command"
                else saved["pair_start"] if kind == "velocity"
                else np.arange(1, 100)
            )
            direct, offset, active = local_affine(saved, leg, rows, kind)
            qr = e145.thin_qr_compress(direct.toarray(), offset)
            local_matrix, local_offset = e145.compressed_affine(qr)
            selector = sparse.csr_matrix(
                (np.ones(len(active)), (np.arange(len(active)), active)),
                shape=(len(active), 3960),
            )
            matrix = sparse.csr_matrix(local_matrix) @ selector
            norms = np.asarray([np.linalg.norm(direct @ w[active] + offset) for w in witnesses])
            compressed_norms = np.asarray([
                np.linalg.norm(matrix @ np.r_[w, np.zeros(1980)] + local_offset)
                for w in witnesses
            ])
            key = f"{kind}_{name}"
            raw[f"qr_{key}"] = qr
            raw[f"qr_{key}_active"] = active
            raw[f"qr_{key}_direct_norms"] = norms
            raw[f"qr_{key}_compressed_norms"] = compressed_norms
            require(
                np.all(np.abs(norms - compressed_norms)
                       <= freeze["contract"]["QR_absolute_atol"]
                       + freeze["contract"]["QR_relative_atol"] * norms),
                "QR/direct mismatch: " + key,
            )
            blocks[key] = matrix, local_offset
            if kind == "startup":
                startup_matrices.append(direct @ selector)
                startup_offsets.append(offset)
    require(len(blocks) == 18, "QR block count differs")
    return blocks, sparse.vstack(startup_matrices, format="csr"), np.concatenate(startup_offsets)


def build_problem(
    saved: dict[str, Any], freeze: dict[str, Any], raw: dict[str, np.ndarray]
) -> e164.ConicProblem:
    c = freeze["contract"]
    blocks, startup_matrix, startup_offset = compressed_blocks(saved, freeze, raw)
    baseline = saved["baseline_metrics"]

    def criterion(name: str, keys: list[str], limit: float) -> e164.Criterion:
        require(np.isfinite(limit) and limit > 0, "nonpositive SOC limit")
        return e164.Criterion(
            name,
            sparse.vstack([blocks[key][0] for key in keys], format="csr"),
            np.concatenate([blocks[key][1] for key in keys]),
            float(limit), tuple(raw[f"qr_{key}"] for key in keys),
        )

    criteria = [
        criterion("overall_command", [f"command_{leg}" for leg in LEGS],
                  math.sqrt(1824 * 42) * baseline["overall_command_RMS"] * 0.8),
        criterion("overall_velocity", [f"velocity_{leg}" for leg in LEGS],
                  math.sqrt(1823 * 42) * baseline["overall_velocity_RMS"]),
    ]
    for leg in LEGS:
        criteria.append(criterion(
            leg + "_command", [f"command_{leg}"],
            math.sqrt(1824 * 7) * baseline["by_leg"][leg]["command_RMS"],
        ))
        criteria.append(criterion(
            leg + "_velocity", [f"velocity_{leg}"],
            math.sqrt(1823 * 7) * baseline["by_leg"][leg]["velocity_RMS"],
        ))
    criteria.append(criterion(
        "startup", [f"startup_{leg}" for leg in LEGS],
        math.sqrt(99 * 42) * c["startup_limits"]["q_RMS"],
    ))
    require(len(criteria) == 15, "SOC count differs")

    guard = c["inner_coefficient_guard"]
    lower = np.r_[np.full(N_COEFF, guard), np.zeros(1980), -1.0]
    upper = np.r_[np.full(N_COEFF, 2.0 - guard), np.full(1980, 2.0), 1.0]
    eye = sparse.eye(N_PARAM, format="csr")
    rows = [-eye, eye]
    rhs = [-lower, upper]
    p = np.arange(990)
    wlo, whi, ell, u = 2 * p, 2 * p + 1, 1980 + p, 2970 + p
    ecols = np.concatenate([
        np.column_stack((ell, wlo)).reshape(-1),
        np.column_stack((ell, whi)).reshape(-1),
        np.column_stack((wlo, u)).reshape(-1),
        np.column_stack((whi, u)).reshape(-1),
    ])
    envelope = sparse.csr_matrix(
        (np.tile([1.0, -1.0], 3960), (np.repeat(np.arange(3960), 2), ecols)),
        shape=(3960, N_PARAM),
    )
    rows.append(envelope)
    rhs.append(np.zeros(3960))
    counts = np.bincount(saved["pair_muscles"][::3], minlength=96)
    rows.append(sparse.csr_matrix(
        (-np.ones(990), (np.repeat(np.arange(330), 3), ell)),
        shape=(330, N_PARAM),
    ))
    physical_guard = c["inner_physical_guard"]
    rhs.append(-c["motor_floor_fraction"] / counts[saved["pair_muscles"][::3]]
               - physical_guard)
    variable = np.asarray(freeze["sample_contract"]["variable_muscles"], dtype=np.int64)
    muscle_index = np.searchsorted(variable, saved["pair_muscles"])
    require(np.array_equal(variable[muscle_index], saved["pair_muscles"]),
            "muscle ownership differs")
    rows.extend((
        sparse.csr_matrix((-np.ones(990), (muscle_index, ell)), shape=(77, N_PARAM)),
        sparse.csr_matrix((np.ones(990), (muscle_index, u)), shape=(77, N_PARAM)),
    ))
    rhs.extend((
        -np.full(77, c["tonic_lower_budget"] + physical_guard),
        np.full(77, c["total_upper_budget"] - physical_guard),
    ))
    tcol = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    max_abs = c["startup_limits"]["q_max_abs"]
    rows.extend((
        sparse.hstack((startup_matrix / max_abs, tcol), format="csr"),
        sparse.hstack((-startup_matrix / max_abs, tcol), format="csr"),
    ))
    rhs.extend((1 - startup_offset / max_abs, 1 + startup_offset / max_abs))
    nonnegative_A = sparse.vstack(rows, format="csr")
    nonnegative_b = np.concatenate(rhs)
    require(nonnegative_A.shape == (FIRST_CONE, N_PARAM), "linear row order differs")
    A_parts, b_parts = [nonnegative_A], [nonnegative_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(FIRST_CONE)]
    kinds, dims = ["NonnegativeConeT"], [FIRST_CONE]
    for item in criteria:
        A, b = e145.make_soc_rows(item.matrix, item.offset, item.limit, 3960)
        A_parts.append(A)
        b_parts.append(b)
        cones.append(clarabel.SecondOrderConeT(A.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(A.shape[0])
    problem = e164.ConicProblem(
        sparse.csc_matrix((N_PARAM, N_PARAM)),
        np.r_[np.zeros(3960), -1.0],
        sparse.vstack(A_parts, format="csc"), np.concatenate(b_parts),
        tuple(cones), tuple(kinds), np.asarray(dims), lower, upper,
        tuple(criteria), startup_matrix, startup_offset,
    )
    raw.update({"problem_" + key: value for key, value in e164.problem_arrays(problem).items()})
    for label, w in (
        ("w0", saved["w0"]),
        ("equal_branch", saved["canonical"]),
    ):
        pair = w.reshape(990, 2)
        trial = np.r_[w, pair.min(axis=1), pair.max(axis=1), -0.251]
        raw[label + "_diagnostic_vector"] = trial
        raw[label + "_diagnostic_slack"] = problem.b - problem.A @ trial
    raw["cone_dims"] = problem.cone_dims
    raw["cone_kinds"] = np.asarray(problem.cone_kinds)
    return problem


def physical_gates(saved: dict[str, Any], weights: np.ndarray, freeze: dict[str, Any]) -> dict[str, bool]:
    c = freeze["contract"]
    pair = weights.reshape(990, 2)
    low, high = pair.min(axis=1), pair.max(axis=1)
    muscles = saved["pair_muscles"]
    counts = np.bincount(muscles[::3], minlength=96)
    motor_min = low.reshape(330, 3).sum(axis=1)
    muscle_min = np.bincount(muscles, weights=low, minlength=96)
    muscle_max = np.bincount(muscles, weights=high, minlength=96)
    variable = np.asarray(freeze["sample_contract"]["variable_muscles"], dtype=np.int64)
    return {
        "coefficient_exact_bounds": bool(np.all((weights >= 0) & (weights <= 2))),
        "motor_sum_min_floor": bool(np.all(
            motor_min >= c["motor_floor_fraction"] / counts[muscles[::3]]
            - c["physical_atol"])),
        "muscle_sum_min_budget": bool(np.all(
            muscle_min[variable] >= c["tonic_lower_budget"] - c["physical_atol"])),
        "muscle_sum_max_budget": bool(np.all(
            muscle_max[variable] <= c["total_upper_budget"] + c["physical_atol"])),
    }


def candidate_gates(
    saved: dict[str, Any], freeze: dict[str, Any], x: np.ndarray,
    metrics: dict[str, Any], arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    c, baseline = freeze["contract"], saved["baseline_metrics"]
    gates = physical_gates(saved, x[:N_COEFF], freeze)
    gates.update(
        margin_bounds=bool(-1 <= x[-1] <= 1),
        fixed_exact=bool(np.array_equal(
            arrays["candidate_activation"][:, saved["fixed_muscles"]],
            saved["original_activation"][:, saved["fixed_muscles"]])),
        activation_nonnegative=bool(np.all(arrays["candidate_activation"] >= -c["physical_atol"])),
        decoder_reconstruction=bool(np.max(np.abs(arrays["decoder_residual"]))
                                    <= 1e-12 + 1e-12 * np.max(np.abs(arrays["candidate_command"][1:]))),
        overall_command_20pct=bool(metrics["overall_command_RMS"]
                                   <= 0.8 * baseline["overall_command_RMS"]),
        overall_velocity_nonincrease=bool(metrics["overall_velocity_RMS"]
                                          <= baseline["overall_velocity_RMS"]),
        all_leg_nonincrease=bool(all(
            metrics["by_leg"][leg][kind] <= baseline["by_leg"][leg][kind]
            for leg in LEGS for kind in ("command_RMS", "velocity_RMS"))),
        startup_q=bool(metrics["startup_q_RMS"] <= c["startup_limits"]["q_RMS"]
                       and metrics["startup_q_max_abs"] <= c["startup_limits"]["q_max_abs"]),
        startup_LTM=bool(saved["startup_ltm_RMS"] <= c["startup_limits"]["LTM_RMS"]
                         and saved["startup_ltm_max_abs"] <= c["startup_limits"]["LTM_max_abs"]),
    )
    return gates


def auxiliary_diagnostics(x: np.ndarray) -> dict[str, np.ndarray]:
    """Record solver-only envelope residuals without changing runtime adoption."""
    pair = x[:N_COEFF].reshape(990, 2)
    ell, upper = x[1980:2970], x[2970:3960]
    return {
        "auxiliary_lower": ell,
        "auxiliary_upper": upper,
        "auxiliary_lower_slack": pair.min(axis=1) - ell,
        "auxiliary_upper_slack": upper - pair.max(axis=1),
        "auxiliary_box_lower_slack": x[N_COEFF:3960],
        "auxiliary_box_upper_slack": 2.0 - x[N_COEFF:3960],
    }


def returned_qr_diagnostics(
    saved: dict[str, Any], problem: e164.ConicProblem,
    x: np.ndarray, freeze: dict[str, Any], raw: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compare the actual returned vector against every direct and QR residual."""
    names: list[str] = []
    direct_norms: list[float] = []
    qr_norms: list[float] = []
    direct_vectors: dict[str, np.ndarray] = {}
    for leg, name in enumerate(LEGS):
        for kind in ("command", "velocity", "startup"):
            rows = (
                saved["target_indices"] if kind == "command"
                else saved["pair_start"] if kind == "velocity"
                else np.arange(1, 100)
            )
            direct, offset, active = local_affine(saved, leg, rows, kind)
            vector = np.asarray(direct @ x[active] + offset)
            key = f"{kind}_{name}"
            direct_vectors[key] = vector
            names.append(key)
            direct_norms.append(float(np.linalg.norm(vector)))
            qr = raw[f"qr_{key}"]
            qr_matrix, qr_offset = e145.compressed_affine(qr)
            qr_norms.append(float(np.linalg.norm(qr_matrix @ x[active] + qr_offset)))
    criterion_direct: list[float] = []
    criterion_qr: list[float] = []
    for item in problem.criteria:
        if item.name == "overall_command":
            keys = [f"command_{leg}" for leg in LEGS]
        elif item.name == "overall_velocity":
            keys = [f"velocity_{leg}" for leg in LEGS]
        elif item.name == "startup":
            keys = [f"startup_{leg}" for leg in LEGS]
        else:
            leg, kind = item.name.split("_", 1)
            keys = [f"{kind}_{leg}"]
        criterion_direct.append(float(np.linalg.norm(np.concatenate(
            [direct_vectors[key] for key in keys]
        ))))
        criterion_qr.append(float(np.linalg.norm(
            item.matrix @ x[:3960] + item.offset
        )))
    direct_array = np.asarray(direct_norms + criterion_direct)
    qr_array = np.asarray(qr_norms + criterion_qr)
    tolerance = (
        freeze["contract"]["QR_absolute_atol"]
        + freeze["contract"]["QR_relative_atol"] * direct_array
    )
    return {
        "returned_qr_names": np.asarray(names + [item.name for item in problem.criteria]),
        "returned_direct_norms": direct_array,
        "returned_qr_norms": qr_array,
        "returned_qr_absolute_residual": np.abs(direct_array - qr_array),
        "returned_qr_tolerance": tolerance,
    }


def dual_diagnostics(
    problem: e164.ConicProblem, solution: Any, history: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> dict[str, np.ndarray]:
    raw = e164.solver_raw(problem, solution, history, freeze)
    true_b = problem.b.copy()
    guard = freeze["contract"]["inner_coefficient_guard"]
    true_b[:N_COEFF] += guard
    true_b[N_PARAM:N_PARAM + N_COEFF] += guard
    first, last = PHYSICAL_ROWS
    true_b[first:last] += freeze["contract"]["inner_physical_guard"]
    true_b[first:last] += freeze["contract"]["physical_atol"]
    true_lower = problem.lower.copy()
    true_upper = problem.upper.copy()
    true_lower[:N_COEFF] = 0.0
    true_upper[:N_COEFF] = 2.0
    bound = e145.dual_margin_upper_bound(
        problem.q, problem.A, true_b, raw["solver_z_projected"],
        true_lower, true_upper, pad_factor=freeze["dual_upper_bound_pad"],
    )
    raw["true_b"] = true_b
    raw["true_lower"] = true_lower
    raw["true_upper"] = true_upper
    raw.update({"true_projected_" + key: np.asarray(value) for key, value in bound.items()})
    return raw


def output_manifest(out: Path) -> dict[str, Any]:
    files = {
        path.relative_to(out).as_posix(): {
            "sha256": sha256(path), "bytes": path.stat().st_size,
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "output-manifest.json"
    }
    return {"schema": "flylab.m3.e197.output-manifest.v1", "files": files,
            "total_bytes": sum(row["bytes"] for row in files.values())}


def run(
    freeze_path: str | Path, freeze_sha256: str, source_review: str | Path,
    source_review_sha256: str, out_path: str | Path, *,
    input_loader: Callable[..., dict[str, Any]] = e164.load_original_inputs,
    solver_factory: Callable[..., Any] = clarabel.DefaultSolver,
) -> dict[str, Any]:
    began = time.perf_counter()
    freeze_path, source_review = Path(freeze_path).resolve(), Path(source_review).resolve()
    freeze, base, merged = validate_freeze(freeze_path, freeze_sha256)
    validate_gate(source_review, source_review_sha256, freeze, freeze_sha256)
    out = Path(out_path).resolve()
    require(not out.exists() and out.parent.is_dir(), "output must be a new directory")
    require(shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
            "free-space reserve unavailable")
    out.mkdir()
    approved = snapshot(out, freeze_path, source_review)
    progress: dict[str, Any] = {
        "status": "initialized", "fit_calls": 0, "solver_constructor_attempts": 0,
        "solver_constructor_completed": 0, "callback_registration_attempts": 0,
        "callback_registration_completed": 0, "solver_returns": 0,
        "prefilter_update_steps": 0, "neural_steps": 0, "physics_steps": 0,
        "body_models": 0, "graph_store_instances": 0, "state_store_instances": 0,
        "secondary_errors": [],
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested", "candidate_adopted": False,
        "whole_goal_complete": False, "F6_exposed": False,
        "conditional_prefilter_proposal": False,
    }
    primary: BaseException | None = None
    pool_limit = None
    pre_raw: dict[str, np.ndarray] = {}
    partial: dict[str, Any] = {}
    derived: dict[str, np.ndarray] = {}
    candidate_raw: dict[str, np.ndarray] = {}

    def persist() -> None:
        e168.atomic_json(out / "ledger.json", progress)

    def deadline(stage: str) -> None:
        require(time.perf_counter() - began <= freeze["resource_limits"]["internal_s"],
                "internal deadline: " + stage)

    def retain(name: str, arrays: dict[str, Any]) -> None:
        e168.atomic_npz(out / name, {key: np.asarray(value) for key, value in arrays.items()})

    try:
        pool_limit = threadpool_limits(limits=1)
        progress["threadpools_during"] = threadpool_info()
        require(all(item.get("num_threads") == 1 for item in progress["threadpools_during"]),
                "CPU thread count exceeds one")
        persist()
        data = input_loader(base, merged) if input_loader is e164.load_original_inputs else input_loader(base)
        e168.capture_loaded_input(data, pre_raw)
        retain("pre-solve-raw.npz", pre_raw)
        e168.validate_loaded_input(data, pre_raw)
        checks: list[tuple[bool, str]] = []
        saved = e164.prepare_saved(data, base, pre_raw, checks)
        prepare_transfer(data, saved, freeze, pre_raw, progress)
        pre_raw["input_checks_json"] = np.asarray(
            json.dumps(checks, sort_keys=True, allow_nan=False)
        )
        require(all(ok for ok, _ in checks), "input numerical check failed")
        deadline("after input")
        problem = build_problem(saved, freeze, pre_raw)
        retain("pre-solve-raw.npz", pre_raw)
        progress["status"] = "pre_solve_saved"
        persist()
        deadline("before solver")
        state: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        fit_began = time.perf_counter()

        def instrumented_factory(*args: Any) -> Any:
            progress["solver_constructor_attempts"] = 1
            persist()
            solver = solver_factory(*args)
            progress["solver_constructor_completed"] = 1
            persist()

            class InstrumentedSolver:
                def set_termination_callback(self, callback: Any) -> None:
                    progress["callback_registration_attempts"] = 1
                    persist()
                    solver.set_termination_callback(callback)
                    progress["callback_registration_completed"] = 1
                    persist()

                def solve(self) -> Any:
                    progress["fit_calls"] = 1
                    progress["status"] = "solve_entered"
                    persist()
                    solution = solver.solve()
                    progress["solver_returns"] = 1
                    partial.update(e145._bare_solution_arrays(solution))
                    progress["status"] = "solve_returned"
                    return solution

                def get_settings(self) -> Any:
                    return solver.get_settings()

            return InstrumentedSolver()

        solution, history, settings = e145.solve_conic_problem(
            problem, freeze, solver_factory=instrumented_factory,
            partial=partial, progress=progress, state=state, history=history,
        )
        progress["fit_wall_s"] = time.perf_counter() - fit_began
        retain("solver-return-raw.npz", partial)
        require(progress["fit_wall_s"] <= freeze["resource_limits"]["fit_s"],
                "fit deadline exceeded")
        require(str(solution.status) in ("Solved", "AlmostSolved"),
                "solver did not return supported status")
        x = np.asarray(solution.x, dtype=np.float64)
        require(x.shape == (N_PARAM,) and np.isfinite(x).all(), "solver vector invalid")
        derived = dual_diagnostics(problem, solution, history, freeze)
        retain("solver-derived-raw.npz", derived)
        metrics, arrays = candidate(saved, x[:N_COEFF])
        arrays.update(returned_vector=x, candidate_slopes=x[:N_COEFF])
        candidate_raw = arrays
        retain("candidate-raw.npz", candidate_raw)
        native_activation_result, native_command_result = independent_candidate(
            saved, x[:N_COEFF]
        )
        native_velocity = (
            native_command_result[saved["pair_end"]]
            - native_command_result[saved["pair_start"]]
        ) / DT
        arrays.update(
            native_activation=native_activation_result,
            native_command=native_command_result,
            native_activation_residual=native_activation_result[1:]
            - arrays["candidate_activation"][1:],
            native_command_residual=native_command_result[1:]
            - arrays["candidate_command"][1:],
            native_velocity_residual=native_velocity - arrays["candidate_velocity"],
        )
        arrays.update(auxiliary_diagnostics(x))
        arrays.update(returned_qr_diagnostics(saved, problem, x, freeze, pre_raw))
        candidate_raw = arrays
        retain("candidate-raw.npz", candidate_raw)
        require(
            np.all(arrays["returned_qr_absolute_residual"]
                   <= arrays["returned_qr_tolerance"]),
            "returned direct/QR norm differs",
        )
        activation_tol = 1e-12 + 1e-12 * np.max(np.abs(native_activation_result[1:]))
        q_tol = 1e-12 + 1e-12 * np.max(np.abs(native_command_result[1:]))
        velocity_tol = 2 * q_tol / DT
        require(
            np.max(np.abs(arrays["native_activation_residual"])) <= activation_tol
            and np.max(np.abs(arrays["native_command_residual"])) <= q_tol
            and np.max(np.abs(arrays["native_velocity_residual"])) <= velocity_tol,
            "native/affine candidate differs",
        )
        gates = candidate_gates(saved, freeze, x, metrics, arrays)
        proposal = all(gates.values())
        true_upper = float(derived["true_projected_margin_upper_bound"])
        require(math.isfinite(true_upper), "true dual upper invalid")
        require(not (proposal and true_upper < 0), "proposal and dual contradict")
        verdict = (
            "supported_saved_prefilter_proposal" if proposal
            else "refuted_saved_prefilter_family" if true_upper < 0
            else "valid_rejection_inconclusive"
        )
        result.update(
            validity="valid", hypothesis_verdict=verdict,
            conditional_prefilter_proposal=proposal, solver_status=str(solution.status),
            margin=float(x[-1]), metrics=metrics,
            baseline_metrics=saved["baseline_metrics"], gates=gates,
            true_projected_margin_upper_bound=true_upper, solver_settings=settings,
        )
        progress["status"] = "complete"
    except BaseException as error:  # noqa: BLE001 - terminal evidence must survive interruptions
        primary = error
        progress["status"] = "technical_failed"
        for name, arrays in (
            ("pre-solve-raw.npz", pre_raw),
            ("solver-return-raw.npz", partial),
            ("solver-derived-raw.npz", derived),
            ("candidate-raw.npz", candidate_raw),
        ):
            if arrays:
                try:
                    retain(name, arrays)
                except BaseException as secondary:  # noqa: BLE001
                    try:
                        retain(name.replace(".npz", "-fallback.npz"), arrays)
                    except BaseException as tertiary:  # noqa: BLE001
                        error.add_note("Secondary E197 raw save: " + repr(tertiary))
                        progress["secondary_errors"].append(repr(secondary))
                        progress["secondary_errors"].append(repr(tertiary))
    finally:
        if pool_limit is not None:
            try:
                pool_limit.restore_original_limits()
                progress["threadpools_after"] = threadpool_info()
            except BaseException as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                else:
                    progress["secondary_errors"].append(repr(error))
        try:
            validate_freeze(freeze_path, freeze_sha256)
            validate_gate(source_review, source_review_sha256, freeze, freeze_sha256)
            for name, expected in approved.items():
                require(sha256(out / "source-snapshot" / name) == expected,
                        "snapshot changed: " + name)
            progress["post_hashes_verified"] = True
        except BaseException as error:  # noqa: BLE001
            progress["post_hashes_verified"] = False
            if primary is None:
                primary = error
            else:
                progress["secondary_errors"].append(repr(error))

        def fail_receipt() -> None:
            progress["status"] = "technical_failed"
            result.update(
                validity="incomplete", hypothesis_verdict="invalid_not_tested",
                conditional_prefilter_proposal=False, technical_failure=True,
            )
            if primary is not None:
                receipt = {
                    "type": type(primary).__name__, "message": str(primary),
                    "traceback": "".join(traceback.format_exception(primary)),
                }
                result["primary_error"] = receipt
                progress["primary_error"] = receipt

        if primary is not None:
            fail_receipt()
        else:
            result["technical_failure"] = False
        elapsed = time.perf_counter() - began
        result.update(
            elapsed_before_publication_s=elapsed,
            prior_conservative_original_s=freeze["prior_conservative_original_s"],
            conservative_cumulative_before_publication_s=
            freeze["prior_conservative_original_s"] + elapsed,
            counts={key: progress[key] for key in (
                "fit_calls", "solver_constructor_attempts", "solver_constructor_completed",
                "callback_registration_attempts", "callback_registration_completed",
                "solver_returns", "prefilter_update_steps", "neural_steps", "physics_steps",
                "body_models", "graph_store_instances", "state_store_instances",
            )},
        )
        try:
            persist()
            e168.atomic_json(out / "result.json", result)
            e168.atomic_json(out / "output-manifest.json", output_manifest(out))
            manifest = json.loads((out / "output-manifest.json").read_text(encoding="utf-8"))
            for name, receipt in manifest["files"].items():
                path = out / name
                require(sha256(path) == receipt["sha256"]
                        and path.stat().st_size == receipt["bytes"],
                        "output manifest changed: " + name)
            total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
            require(total <= freeze["resource_limits"]["raw_bytes"], "raw byte cap exceeded")
            require(shutil.disk_usage(out).free >= freeze["resource_limits"]["free_bytes"],
                    "terminal free-space reserve unavailable")
            deadline("after publication")
        except BaseException as error:  # noqa: BLE001
            if primary is None:
                primary = error
            else:
                progress["secondary_errors"].append(repr(error))
            fail_receipt()
            for name in ("result.json", "ledger.json", "output-manifest.json"):
                path = out / name
                if path.exists():
                    path.replace(out / f"{name}.invalid-stale-{uuid.uuid4().hex}")
            try:
                persist()
                e168.atomic_json(out / "result.json", result)
                e168.atomic_json(out / "output-manifest.json", output_manifest(out))
            except BaseException as secondary:  # noqa: BLE001
                primary.add_note("Secondary E197 failure publication: " + repr(secondary))
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
    run(args.freeze, args.freeze_sha256, args.source_review,
        args.source_review_sha256, args.out)


if __name__ == "__main__":
    main()
