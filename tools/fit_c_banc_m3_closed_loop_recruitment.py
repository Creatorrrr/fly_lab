"""E169 frozen conditional fit of positive recruitment branches on E150 signals."""

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
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import clarabel
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import fit_c_banc_m3_conic_pool as e145
import fit_c_banc_m3_onset_recruitment as e164
import fit_c_banc_m3_resource_attenuation as e168

FREEZE_SHA = "7d5f3b911e4952e988bbf740b5b253e1aa2c1b3faddaa66a9b7e443d5dff7884"
DESIGN_SHA = "8a7a840b655039efcda62477da08843f5a6bae696651f905a11f92b55d06a73f"
SCHEMA = "flylab.m3.e169.closed-loop-recruitment.v1"
GATE_SCHEMA = "flylab.m3.e169.parent-source-review.v1"
OWNED = (
    "tools/fit_c_banc_m3_closed_loop_recruitment.py",
    "tests/test_c_banc_closed_loop_recruitment.py",
)
LEGS = e164.LEGS


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_path(name: str) -> Path:
    require(isinstance(name, str), "E169 frozen path must be text")
    relative = Path(name)
    path = (ROOT / relative).resolve()
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and path.is_relative_to(ROOT),
        "E169 unsafe frozen path",
    )
    return path


def hashes(freeze: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for section in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze[section]
        require(
            isinstance(mapping, dict) and mapping, "E169 missing hash map: " + section
        )
        for name, digest in mapping.items():
            require(
                name not in merged or merged[name] == digest,
                "E169 conflicting hash: " + name,
            )
            merged[name] = digest
    return merged


def validate_freeze(
    path: Path, digest: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    require(
        digest == FREEZE_SHA and sha256(path) == FREEZE_SHA,
        "Exact E169 freeze bytes required",
    )
    freeze = json.loads(path.read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == SCHEMA
        and freeze.get("design_sha256") == DESIGN_SHA
        and freeze.get("owned_files") == list(OWNED),
        "E169 freeze identity differs",
    )
    require(sha256(frozen_path(freeze["design"])) == DESIGN_SHA, "E169 design changed")
    base_path = frozen_path(freeze["base_freeze"])
    require(
        freeze["base_freeze_sha256"] == e164.FREEZE_SHA
        and sha256(base_path) == e164.FREEZE_SHA,
        "E169 base freeze differs",
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    require(
        base["schema"] == e164.SCHEMA
        and base["sample_contract"] == freeze["sample_contract"]
        and base["body_hash"] == freeze["body_hash"]
        and base["paths"] == freeze["paths"],
        "E169/base sample contract differs",
    )
    t, c, limits = freeze["transfer"], freeze["contract"], freeze["resource_limits"]
    require(
        freeze["solver"] == "Clarabel"
        and freeze["solver_version"] == str(clarabel.__version__) == "0.11.1",
        "E169 solver version differs",
    )
    require(
        (
            t["pair_count"],
            t["coefficient_count"],
            t["total_parameter_count"],
            t["first_cone_row_count"],
            t["unique_qr_blocks"],
        )
        == (990, 1980, 3961, 20682, 18)
        and t["pair_order"] == "ascending_original_graph_column_then_filter"
        and t["branch_order"] == ["low", "high"],
        "E169 transfer contract differs",
    )
    require(
        c["margin_bounds"] == [-1.0, 1.0]
        and c["coefficient_bounds"] == c["envelope_bounds"] == [0.0, 2.0]
        and c["tonic_lower_budget"] == 0.5
        and c["total_upper_budget"] == 2.0
        and c["motor_floor_fraction"] == 1e-6
        and c["physical_atol"] == 1e-12
        and c["physical_dual_relaxation_rows"] == [11882, 12366]
        and c["soc_count"] == 15
        and c["baseline_witness_margin"] == -0.251,
        "E169 physical contract differs",
    )
    require(
        (
            limits["original_processes"],
            limits["automatic_retries"],
            limits["cpu_threads"],
            limits["gpu_count"],
            limits["fit_s"],
            limits["internal_s"],
            limits["external_s"],
            limits["supervisor_s"],
            limits["raw_bytes"],
            limits["free_bytes"],
        )
        == (1, 0, 1, 0, 20, 70, 85, 95, 201326592, 1073741824),
        "E169 resource contract differs",
    )
    require(
        freeze["prior_conservative_s"] == 2928
        and freeze["prospective_cumulative_s"] == 3023
        and freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "E169 prior/outcome contract differs",
    )
    merged = hashes(freeze)
    base_hashes = e164.merged_frozen_hashes(base)
    require(
        merged.get(freeze["base_freeze"]) == e164.FREEZE_SHA
        and all(merged.get(name) == value for name, value in base_hashes.items()),
        "E169 base hashes differ",
    )
    for name, value in merged.items():
        require(sha256(frozen_path(name)) == value, "E169 frozen hash changed: " + name)
    return freeze, base, merged


def validate_gate(path: Path, digest: str, freeze: dict[str, Any]) -> dict[str, Any]:
    require(sha256(path) == digest, "E169 source review SHA differs")
    gate = json.loads(path.read_text(encoding="utf-8"))
    require(
        gate.get("schema") == GATE_SCHEMA
        and gate.get("approved") is True
        and gate.get("blocking_defects") == []
        and gate.get("accepts_single_closed_loop_recruitment_fit") is True,
        "E169 source review blocks execution",
    )
    require(
        type(gate.get("original_processes_allowed")) is int
        and gate["original_processes_allowed"] == 1
        and type(gate.get("automatic_retries")) is int
        and gate["automatic_retries"] == 0,
        "E169 source review process contract differs",
    )
    require(
        gate.get("freeze_sha256") == FREEZE_SHA
        and gate.get("design_sha256") == DESIGN_SHA
        and gate.get("source_sha256")
        == {name: sha256(frozen_path(name)) for name in OWNED},
        "E169 source review provenance differs",
    )
    return gate


def snapshot(
    out: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    gate_path: Path,
    gate: dict[str, Any],
) -> dict[str, str]:
    sources = {
        "design.md": frozen_path(freeze["design"]),
        "freeze.json": freeze_path,
        "source-review.json": gate_path,
        "fitter.py": frozen_path(OWNED[0]),
        "tests.py": frozen_path(OWNED[1]),
    }
    directory = out / "source-snapshot"
    directory.mkdir()
    approved = {}
    for name, source in sources.items():
        digest = sha256(source)
        (directory / name).write_bytes(source.read_bytes())
        require(
            sha256(directory / name) == digest, "E169 snapshot write differs: " + name
        )
        approved[name] = digest
    require(
        approved["fitter.py"] == gate["source_sha256"][OWNED[0]]
        and approved["tests.py"] == gate["source_sha256"][OWNED[1]],
        "E169 approved snapshot differs",
    )
    return approved


def verify_snapshot(
    out: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    gate_path: Path,
    approved: dict[str, str],
) -> None:
    sources = {
        "design.md": frozen_path(freeze["design"]),
        "freeze.json": freeze_path,
        "source-review.json": gate_path,
        "fitter.py": frozen_path(OWNED[0]),
        "tests.py": frozen_path(OWNED[1]),
    }
    for name, source in sources.items():
        require(
            sha256(source) == approved[name]
            and sha256(out / "source-snapshot" / name) == approved[name],
            "E169 source/snapshot changed: " + name,
        )


atomic_json = e168.atomic_json
atomic_npz = e168.atomic_npz
capture_loaded_input = e168.capture_loaded_input
validate_loaded_input = e168.validate_loaded_input


def numerical(ok: Any, message: str, pending: list[tuple[bool, str]]) -> None:
    pending.append((bool(ok), message))


def enforce(pending: list[tuple[bool, str]]) -> None:
    for ok, message in pending:
        require(ok, message)


def prepare_transfer(
    data: dict[str, Any],
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
) -> dict[str, Any]:
    c = freeze["sample_contract"]
    motors = np.asarray(c["variable_motors"], dtype=np.int64)
    membership = np.asarray(data["adapter"]["membership"], dtype=np.int64)
    slopes = np.asarray(data["parameters"]["slopes"], dtype=np.float64)
    bank = np.asarray(data["recruitment"]["bank"], dtype=np.float64)
    kappa = np.asarray(data["parameters"]["kappa"], dtype=np.float64)
    require(
        motors.shape == (330,)
        and np.array_equal(motors, np.sort(motors))
        and np.unique(motors).size == 330,
        "E169 variable motor order differs",
    )
    require(
        np.array_equal(
            motors, np.flatnonzero(np.isin(membership, c["variable_muscles"]))
        ),
        "E169 variable motor ownership differs",
    )
    pair_motors = np.repeat(motors, 3)
    pair_filters = np.tile(np.arange(3), 330)
    pair_muscles = membership[pair_motors]
    branch_basis = np.empty((2001, 1980), dtype=np.float64)
    selected_bank = bank[:, pair_motors, pair_filters]
    selected_kappa = kappa[pair_motors]
    branch_basis[:, 0::2] = np.minimum(selected_bank, selected_kappa)
    branch_basis[:, 1::2] = np.maximum(selected_bank - selected_kappa, 0)
    w0 = slopes[pair_muscles, pair_motors, pair_filters].reshape(-1)
    saved.update(
        pair_motors=pair_motors,
        pair_filters=pair_filters,
        pair_muscles=pair_muscles,
        branch_basis=branch_basis,
        w0=w0,
        membership=membership,
        slopes=slopes,
        bank=bank,
        kappa=kappa,
        original_activation=np.asarray(data["recruitment"]["original_activation"]),
        group_legs=np.asarray(c["group_legs"], dtype=np.int64),
        fixed_muscles=np.asarray(c["fixed_muscles"], dtype=np.int64),
    )
    raw.update(
        pair_motors=pair_motors,
        pair_filters=pair_filters,
        pair_muscles=pair_muscles,
        pair_branches=np.tile(np.arange(2), 990),
        branch_basis=branch_basis,
        w0=w0,
        affine_moment=np.asarray(saved["moment"]),
        affine_recorded_anchor=np.asarray(saved["recorded_command"]),
        affine_rows_command=np.asarray(saved["target_indices"]),
        affine_rows_velocity_start=np.asarray(saved["pair_start"]),
        affine_rows_velocity_end=np.asarray(saved["pair_end"]),
        affine_rows_startup=np.arange(1, 100),
    )
    baseline, arrays = candidate(saved, w0)
    raw.update(
        baseline_delta_activation=arrays["candidate_delta_activation"],
        baseline_command_residual=arrays["candidate_command"]
        - saved["recorded_command"],
        baseline_activation_residual=arrays["candidate_activation"]
        - saved["activation"],
    )
    numerical(
        np.array_equal(arrays["candidate_activation"], saved["activation"])
        and np.array_equal(arrays["candidate_command"], saved["recorded_command"])
        and baseline == saved["baseline_metrics"],
        "E169 w0 baseline is not exact",
        pending,
    )
    native = native_candidate(saved, w0)
    raw["baseline_native_activation_residual"] = native - saved["activation"]
    numerical(
        np.max(np.abs(raw["baseline_native_activation_residual"][1:]))
        <= freeze["contract"]["native_compatibility_atol"],
        "E169 native baseline differs",
        pending,
    )
    return saved


def activation_delta(saved: dict[str, Any], w: Any) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    require(w.shape == (1980,) and np.isfinite(w).all(), "E169 slope vector invalid")
    delta = np.zeros_like(saved["activation"])
    np.add.at(
        delta.T,
        np.repeat(saved["pair_muscles"], 2),
        (saved["branch_basis"] * (w - saved["w0"])).T,
    )
    return delta


def native_candidate(saved: dict[str, Any], w: Any) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    slopes = np.array(saved["slopes"], copy=True)
    slopes[saved["pair_muscles"], saved["pair_motors"], saved["pair_filters"]] = (
        w.reshape(990, 2)
    )
    return e164.reconstruct_activation(
        saved["bank"],
        saved["original_activation"],
        slopes,
        saved["kappa"],
        saved["membership"],
        saved["fixed_muscles"],
    )


def candidate(
    saved: dict[str, Any], w: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    delta = activation_delta(saved, w)
    activation = saved["activation"] + delta
    command = saved["recorded_command"] + 0.02 * (delta @ saved["moment"].T)
    metrics, arrays = e164.metrics_from_recorded(saved, command)
    reconstructed = saved["rest"] + 0.02 * (activation @ saved["moment"].T)
    arrays.update(
        candidate_activation=activation,
        candidate_command=command,
        candidate_tendon=np.asarray(saved["tendon"]),
        candidate_delta_activation=delta,
        decoder_residual=command[1:] - reconstructed[1:],
    )
    return metrics, arrays


def physical_gates(
    saved: dict[str, Any], w: np.ndarray, freeze: dict[str, Any]
) -> dict[str, bool]:
    c = freeze["contract"]
    pair = w.reshape(990, 2)
    lower = np.min(pair, axis=1)
    upper = np.max(pair, axis=1)
    muscles = saved["pair_muscles"]
    n_motors = np.bincount(muscles[::3], minlength=96)
    mn_lower = lower.reshape(330, 3).sum(axis=1)
    by_muscle_lower = np.bincount(muscles, weights=lower, minlength=96)
    by_muscle_upper = np.bincount(muscles, weights=upper, minlength=96)
    variable = np.asarray(freeze["sample_contract"]["variable_muscles"], dtype=np.int64)
    return {
        "coefficient_exact_bounds": bool(np.all((w >= 0) & (w <= 2))),
        "motor_sum_min_floor": bool(
            np.all(
                mn_lower
                >= c["motor_floor_fraction"] / n_motors[muscles[::3]]
                - c["physical_atol"]
            )
        ),
        "muscle_sum_min_budget": bool(
            np.all(
                by_muscle_lower[variable]
                >= c["tonic_lower_budget"] - c["physical_atol"]
            )
        ),
        "muscle_sum_max_budget": bool(
            np.all(
                by_muscle_upper[variable]
                <= c["total_upper_budget"] + c["physical_atol"]
            )
        ),
    }


def candidate_gates(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    x: np.ndarray,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    c = freeze["contract"]
    baseline = saved["baseline_metrics"]
    gates = physical_gates(saved, x[:1980], freeze)
    gates.update(
        margin_bounds=bool(-1 <= x[-1] <= 1),
        fixed_exact=bool(
            np.array_equal(
                arrays["candidate_activation"][:, saved["fixed_muscles"]],
                saved["activation"][:, saved["fixed_muscles"]],
            )
        ),
        activation_nonnegative=bool(
            np.all(arrays["candidate_activation"] >= -c["physical_atol"])
        ),
        decoder_reconstruction=bool(
            np.max(np.abs(arrays["decoder_residual"])) <= c["physical_atol"]
        ),
        overall_command_20pct=bool(
            metrics["overall_command_RMS"]
            <= c["proposal_command_RMS_ratio"] * baseline["overall_command_RMS"]
        ),
        overall_velocity_nonincrease=bool(
            metrics["overall_velocity_RMS"] <= baseline["overall_velocity_RMS"]
        ),
        all_leg_nonincrease=bool(
            all(
                metrics["by_leg"][leg][kind] <= baseline["by_leg"][leg][kind]
                for leg in LEGS
                for kind in ("command_RMS", "velocity_RMS")
            )
        ),
        startup_q=bool(
            metrics["startup_q_RMS"] <= c["startup_limits"]["q_RMS"]
            and metrics["startup_q_max_abs"] <= c["startup_limits"]["q_max_abs"]
        ),
        startup_LTM=bool(
            saved["startup_ltm_RMS"] <= c["startup_limits"]["LTM_RMS"]
            and saved["startup_ltm_max_abs"] <= c["startup_limits"]["LTM_max_abs"]
        ),
    )
    return gates


def local_affine(
    saved: dict[str, Any], leg: int, rows: np.ndarray, kind: str
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build one direct local block without retaining a dense global affine copy."""
    owners = np.repeat(saved["pair_muscles"], 2)
    active = np.flatnonzero(saved["group_legs"][owners] == leg)
    joints = np.arange(7 * leg, 7 * (leg + 1))
    moment = saved["moment"][np.ix_(joints, owners[active])]
    basis = saved["branch_basis"]
    if kind == "command":
        values = basis[rows][:, active]
        anchor = (
            saved["recorded_command"][np.ix_(rows, joints)] - saved["target"][:, joints]
        )
    elif kind == "velocity":
        values = (
            basis[saved["pair_end"]][:, active] - basis[saved["pair_start"]][:, active]
        ) / 0.005
        anchor = (
            (
                saved["recorded_command"][saved["pair_end"]][:, joints]
                - saved["recorded_command"][saved["pair_start"]][:, joints]
            )
            / 0.005
        ) - saved["target_velocity"][:, joints]
    elif kind == "startup":
        values = basis[rows][:, active]
        anchor = (
            saved["recorded_command"][np.ix_(rows, joints)]
            - saved["startup_target"][:, joints]
        )
    else:
        raise ValueError("E169 unknown affine block")
    local = (0.02 * values[:, None, :] * moment[None, :, :]).reshape(
        rows.size * 7, active.size
    )
    local_matrix = sparse.csr_matrix(local)
    offset = anchor.reshape(-1) - local_matrix @ saved["w0"][active]
    return local_matrix, offset, active


def compressed_blocks(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
) -> tuple[
    dict[str, tuple[sparse.csr_matrix, np.ndarray]], sparse.csr_matrix, np.ndarray
]:
    blocks: dict[str, tuple[sparse.csr_matrix, np.ndarray]] = {}
    startup_matrices = []
    startup_offsets = []
    witnesses = (
        saved["w0"],
        np.zeros(1980),
        0.5 + 0.5 * np.cos(np.arange(1980) * 0.173),
    )
    for leg, name in enumerate(LEGS):
        for kind in ("command", "velocity", "startup"):
            rows = (
                saved["target_indices"]
                if kind == "command"
                else saved["pair_start"]
                if kind == "velocity"
                else np.arange(1, 100)
            )
            direct, offset, active = local_affine(saved, leg, rows, kind)
            qr = e145.thin_qr_compress(direct.toarray(), offset)
            qmatrix, qoffset = e145.compressed_affine(qr)
            selector = sparse.csr_matrix(
                (np.ones(active.size), (np.arange(active.size), active)),
                shape=(active.size, 3960),
            )
            compressed = sparse.csr_matrix(qmatrix) @ selector
            direct_norms = np.asarray(
                [np.linalg.norm(direct @ w[active] + offset) for w in witnesses]
            )
            compressed_norms = np.asarray(
                [
                    np.linalg.norm(compressed @ np.r_[w, np.zeros(1980)] + qoffset)
                    for w in witnesses
                ]
            )
            key = f"{kind}_{name}"
            raw[f"qr_{key}"] = qr
            raw[f"qr_{key}_active"] = active
            raw[f"qr_{key}_direct_norms"] = direct_norms
            raw[f"qr_{key}_compressed_norms"] = compressed_norms
            numerical(
                np.all(
                    np.abs(direct_norms - compressed_norms)
                    <= freeze["contract"]["QR_absolute_atol"]
                    + freeze["contract"]["QR_relative_atol"] * direct_norms
                ),
                "E169 QR norm mismatch: " + key,
                pending,
            )
            blocks[key] = compressed, qoffset
            if kind == "startup":
                startup_matrices.append(direct @ selector)
                startup_offsets.append(offset)
    require(
        len(blocks) == freeze["transfer"]["unique_qr_blocks"],
        "E169 QR block count differs",
    )
    return (
        blocks,
        sparse.vstack(startup_matrices, format="csr"),
        np.concatenate(startup_offsets),
    )


def build_problem(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
) -> e164.ConicProblem:
    c = freeze["contract"]
    blocks, startup_matrix, startup_offset = compressed_blocks(
        saved, freeze, raw, pending
    )
    baseline = saved["baseline_metrics"]

    def criterion(name: str, keys: list[str], limit: float) -> e164.Criterion:
        require(np.isfinite(limit) and limit > 0, "E169 nonpositive SOC limit: " + name)
        return e164.Criterion(
            name,
            sparse.vstack([blocks[key][0] for key in keys], format="csr"),
            np.concatenate([blocks[key][1] for key in keys]),
            float(limit),
            tuple(raw[f"qr_{key}"] for key in keys),
        )

    criteria = [
        criterion(
            "overall_command",
            [f"command_{leg}" for leg in LEGS],
            math.sqrt(1824 * 42)
            * baseline["overall_command_RMS"]
            * c["proposal_command_RMS_ratio"],
        ),
        criterion(
            "overall_velocity",
            [f"velocity_{leg}" for leg in LEGS],
            math.sqrt(1823 * 42) * baseline["overall_velocity_RMS"],
        ),
    ]
    for leg in LEGS:
        criteria.append(
            criterion(
                leg + "_command",
                [f"command_{leg}"],
                math.sqrt(1824 * 7) * baseline["by_leg"][leg]["command_RMS"],
            )
        )
        criteria.append(
            criterion(
                leg + "_velocity",
                [f"velocity_{leg}"],
                math.sqrt(1823 * 7) * baseline["by_leg"][leg]["velocity_RMS"],
            )
        )
    criteria.append(
        criterion(
            "startup",
            [f"startup_{leg}" for leg in LEGS],
            math.sqrt(99 * 42) * c["startup_limits"]["q_RMS"],
        )
    )
    require(len(criteria) == c["soc_count"] == 15, "E169 SOC count differs")

    n = 3961
    lower = np.r_[np.zeros(3960), c["margin_bounds"][0]]
    upper = np.r_[np.full(3960, 2.0), c["margin_bounds"][1]]
    eye = sparse.eye(n, format="csr")
    rows = [-eye, eye]
    rhs = [-lower, upper]
    p = np.arange(990)
    wlo, whi, ell, u = 2 * p, 2 * p + 1, 1980 + p, 2970 + p
    erows = np.repeat(np.arange(3960), 2)
    ecols = np.concatenate(
        [
            np.column_stack((ell, wlo)).reshape(-1),
            np.column_stack((ell, whi)).reshape(-1),
            np.column_stack((wlo, u)).reshape(-1),
            np.column_stack((whi, u)).reshape(-1),
        ]
    )
    edata = np.tile(np.array([1.0, -1.0]), 3960)
    epigraph = sparse.csr_matrix((edata, (erows, ecols)), shape=(3960, n))
    rows.append(epigraph)
    rhs.append(np.zeros(3960))
    motor_floor = sparse.csr_matrix(
        (-np.ones(990), (np.repeat(np.arange(330), 3), ell)), shape=(330, n)
    )
    n_motors = np.bincount(saved["pair_muscles"][::3], minlength=96)
    rows.append(motor_floor)
    rhs.append(-c["motor_floor_fraction"] / n_motors[saved["pair_muscles"][::3]])
    variable = np.asarray(freeze["sample_contract"]["variable_muscles"], dtype=np.int64)
    muscle_index = np.searchsorted(variable, saved["pair_muscles"])
    require(
        np.array_equal(variable[muscle_index], saved["pair_muscles"]),
        "E169 muscle map invalid",
    )
    muscle_lower = sparse.csr_matrix(
        (-np.ones(990), (muscle_index, ell)), shape=(77, n)
    )
    muscle_upper = sparse.csr_matrix((np.ones(990), (muscle_index, u)), shape=(77, n))
    rows.extend((muscle_lower, muscle_upper))
    rhs.extend(
        (-np.full(77, c["tonic_lower_budget"]), np.full(77, c["total_upper_budget"]))
    )
    tcol = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    rows.extend(
        (
            sparse.hstack(
                (startup_matrix / c["startup_limits"]["q_max_abs"], tcol), format="csr"
            ),
            sparse.hstack(
                (-startup_matrix / c["startup_limits"]["q_max_abs"], tcol), format="csr"
            ),
        )
    )
    rhs.extend(
        (
            1 - startup_offset / c["startup_limits"]["q_max_abs"],
            1 + startup_offset / c["startup_limits"]["q_max_abs"],
        )
    )
    nonnegative_A = sparse.vstack(rows, format="csr")
    nonnegative_b = np.concatenate(rhs)
    require(nonnegative_A.shape == (20682, 3961), "E169 first cone dimensions differ")
    A_parts = [nonnegative_A]
    b_parts = [nonnegative_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(20682)]
    kinds = ["NonnegativeConeT"]
    dims = [20682]
    for item in criteria:
        soc_A, soc_b = e145.make_soc_rows(item.matrix, item.offset, item.limit, 3960)
        A_parts.append(soc_A)
        b_parts.append(soc_b)
        cones.append(clarabel.SecondOrderConeT(soc_A.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(soc_A.shape[0])
    problem = e164.ConicProblem(
        sparse.csc_matrix((n, n)),
        np.r_[np.zeros(3960), -1.0],
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
    pair = saved["w0"].reshape(990, 2)
    witness = np.r_[
        saved["w0"],
        np.min(pair, axis=1),
        np.max(pair, axis=1),
        c["baseline_witness_margin"],
    ]
    slack = problem.b - problem.A @ witness
    raw.update(
        {"problem_" + key: value for key, value in e164.problem_arrays(problem).items()}
    )
    raw.update(
        baseline_witness=witness,
        baseline_witness_slack=slack,
        cone_dims=problem.cone_dims,
        cone_kinds=np.asarray(problem.cone_kinds),
    )
    gates = physical_gates(saved, saved["w0"], freeze)
    for key, value in gates.items():
        numerical(value, "E169 baseline physical failure: " + key, pending)
    numerical(
        np.all(slack[:20682] >= -c["physical_atol"]),
        "E169 witness linear infeasible",
        pending,
    )
    start = 20682
    for item, dim in zip(criteria, dims[1:], strict=True):
        section = slack[start : start + dim]
        numerical(
            section[0] >= np.linalg.norm(section[1:]) - c["physical_atol"],
            "E169 witness SOC infeasible: " + item.name,
            pending,
        )
        start += dim
    return problem


def solver_derived(
    problem: e164.ConicProblem,
    solution: Any,
    history: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> dict[str, np.ndarray]:
    raw = e164.solver_raw(problem, solution, history, freeze)
    b_relaxed = problem.b.copy()
    first, last = freeze["contract"]["physical_dual_relaxation_rows"]
    require(
        (first, last) == (11882, 12366) and problem.cone_dims[0] == 20682,
        "E169 physical dual row map differs",
    )
    b_relaxed[first:last] += freeze["contract"]["physical_atol"]
    relaxed = e145.dual_margin_upper_bound(
        problem.q,
        problem.A,
        b_relaxed,
        raw["solver_z_projected"],
        problem.lower,
        problem.upper,
        pad_factor=freeze["dual_upper_bound_pad"],
    )
    raw["physical_budget_relaxed_b"] = b_relaxed
    raw.update(
        {
            "relaxed_projected_" + key: np.asarray(value)
            for key, value in relaxed.items()
        }
    )
    return raw


def output_manifest(out: Path) -> dict[str, Any]:
    files = {
        path.relative_to(out).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "output-manifest.json"
    }
    return {
        "schema": "flylab.m3.e169.output-manifest.v1",
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }


def run(
    freeze_path: str | Path,
    freeze_sha256: str,
    source_review: str | Path,
    source_review_sha256: str,
    out_path: str | Path,
    *,
    input_loader: Callable[..., dict[str, Any]] = e164.load_original_inputs,
    solver_factory: Callable[..., Any] = clarabel.DefaultSolver,
) -> dict[str, Any]:
    began = time.perf_counter()
    freeze_path, source_review = (
        Path(freeze_path).resolve(),
        Path(source_review).resolve(),
    )
    freeze, base, merged = validate_freeze(freeze_path, freeze_sha256)
    gate = validate_gate(source_review, source_review_sha256, freeze)
    out = Path(out_path).resolve()
    require(
        not out.exists() and out.parent.is_dir(), "E169 output must be a new directory"
    )
    require(
        shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
        "E169 free-space reserve unavailable",
    )
    out.mkdir()
    approved = snapshot(out, freeze, freeze_path, source_review, gate)
    progress: dict[str, Any] = {
        "status": "initialized",
        "fit_calls": 0,
        "solver_constructor_attempts": 0,
        "solver_constructor_completed": 0,
        "callback_registration_attempts": 0,
        "callback_registration_completed": 0,
        "solver_returns": 0,
        "new_ema_rollout_steps": 0,
        "resource_state_update_steps": 0,
        "body_models": 0,
        "graph_store_instances": 0,
        "state_store_instances": 0,
        "physics_steps": 0,
        "neural_steps": 0,
        "secondary_errors": [],
    }
    result: dict[str, Any] = {
        "schema": "flylab.m3.e169.result.v1",
        "validity": "incomplete",
        "conditional_closed_loop_recruitment_proposal": False,
        "hypothesis_verdict": "invalid_not_tested",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
    }
    primary: BaseException | None = None
    thread_limit = None
    pre_raw: dict[str, np.ndarray] = {}
    pre_saved = False
    # Retain caller-owned completed arrays across failures of their first publication.
    solver_return_raw: dict[str, Any] = {}
    solver_return_saved = False
    solver_derived_raw: dict[str, np.ndarray] = {}
    solver_derived_saved = False
    candidate_raw: dict[str, np.ndarray] = {}
    candidate_saved = False

    def persist() -> None:
        atomic_json(out / "ledger.json", progress)

    def deadline(stage: str) -> None:
        if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E169 internal deadline: " + stage)

    def secondary(error: BaseException, stage: str) -> None:
        progress["secondary_errors"].append({"stage": stage, "error": repr(error)})
        if primary is not None:
            primary.add_note("Secondary E169 " + stage + ": " + repr(error))

    try:
        thread_limit = threadpool_limits(limits=1)
        progress["cpu_thread_environment"] = {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        progress["threadpools_during"] = threadpool_info()
        require(
            all(
                item.get("num_threads") == 1 for item in progress["threadpools_during"]
            ),
            "E169 CPU threads exceed one",
        )
        persist()
        data = (
            input_loader(base, merged)
            if input_loader is e164.load_original_inputs
            else input_loader(base)
        )
        capture_loaded_input(data, pre_raw)
        atomic_npz(out / "pre-solve-raw.npz", pre_raw)
        pre_saved = True
        validate_loaded_input(data, pre_raw)
        pre_saved = False
        checks: list[tuple[bool, str]] = []
        saved = e164.prepare_saved(data, base, pre_raw, checks)
        prepare_transfer(data, saved, freeze, pre_raw, checks)
        e164.save_check_evidence(out / "pre-solve-raw.npz", pre_raw, "input", checks)
        pre_saved = True
        enforce(checks)
        deadline("after input checkpoint")
        pre_saved = False
        problem_checks: list[tuple[bool, str]] = []
        problem = build_problem(saved, freeze, pre_raw, problem_checks)
        e164.save_check_evidence(
            out / "pre-solve-raw.npz", pre_raw, "problem", problem_checks
        )
        pre_saved = True
        enforce(problem_checks)
        progress["status"] = "pre_solve_saved"
        persist()
        deadline("before solver")
        partial = solver_return_raw
        state: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        fit_began = time.perf_counter()

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
                    persist()
                    solver.set_termination_callback(callback)
                    progress["callback_registration_completed"] = 1
                    persist()

                def solve(self) -> Any:
                    progress["fit_calls"] = 1
                    progress["status"] = "solve_entered"
                    persist()
                    returned = solver.solve()
                    progress["solver_returns"] = 1
                    partial.update(e145._bare_solution_arrays(returned))
                    state["solve_returned"] = True
                    progress["status"] = "solve_returned"
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
                state=state,
                history=history,
            )
        except BaseException as solve_error:
            progress["fit_calls"] = max(
                progress["fit_calls"], int(state.get("solve_entries", 0))
            )
            if partial:
                try:
                    atomic_npz(
                        out / "solver-partial.npz",
                        {key: np.asarray(value) for key, value in partial.items()},
                    )
                except BaseException as save_error:
                    solve_error.add_note(
                        "Secondary E169 solver partial save: " + repr(save_error)
                    )
                    secondary(save_error, "solver partial raw")
            raise
        progress["fit_calls"] = max(
            progress["fit_calls"], int(state.get("solve_entries", 0))
        )
        progress["fit_wall_s"] = time.perf_counter() - fit_began
        atomic_npz(
            out / "solver-return-raw.npz",
            {key: np.asarray(value) for key, value in partial.items()},
        )
        solver_return_saved = True
        require(
            progress["fit_wall_s"] <= freeze["resource_limits"]["fit_s"],
            "E169 fit deadline exceeded",
        )
        solver_derived_raw = solver_derived(problem, solution, history, freeze)
        atomic_npz(out / "solver-derived-raw.npz", solver_derived_raw)
        solver_derived_saved = True
        x = np.asarray(solution.x, dtype=np.float64)
        require(
            x.shape == (3961,) and np.isfinite(x).all(), "E169 returned vector invalid"
        )
        metrics, arrays = candidate(saved, x[:1980])
        native = native_candidate(saved, x[:1980])
        native_command = saved["rest"] + 0.02 * (native @ saved["moment"].T)
        native_velocity = (
            native_command[saved["pair_end"]] - native_command[saved["pair_start"]]
        ) / 0.005
        arrays.update(
            returned_vector=x,
            candidate_slopes=x[:1980],
            candidate_native_activation=native,
            candidate_native_activation_residual=native[1:]
            - arrays["candidate_activation"][1:],
            candidate_native_command_residual=native_command[1:]
            - arrays["candidate_command"][1:],
            candidate_native_velocity_residual=native_velocity
            - arrays["candidate_velocity"],
        )
        candidate_raw = arrays
        atomic_npz(out / "candidate-raw.npz", candidate_raw)
        candidate_saved = True
        require(
            np.max(np.abs(arrays["candidate_native_activation_residual"]))
            <= freeze["contract"]["native_compatibility_atol"]
            and np.max(np.abs(arrays["candidate_native_command_residual"]))
            <= freeze["contract"]["native_compatibility_atol"]
            and np.max(np.abs(arrays["candidate_native_velocity_residual"]))
            <= freeze["contract"]["native_compatibility_atol"],
            "E169 native candidate reconstruction differs",
        )
        gates = candidate_gates(saved, freeze, x, metrics, arrays)
        proposal = all(gates.values())
        strict_upper = float(solver_derived_raw["projected_dual_margin_upper_bound"])
        relaxed_upper = float(
            solver_derived_raw["relaxed_projected_margin_upper_bound"]
        )
        require(
            math.isfinite(strict_upper) and math.isfinite(relaxed_upper),
            "E169 projected dual bound invalid",
        )
        verdict = (
            "conditional_closed_loop_recruitment_proposal"
            if proposal
            else "refuted_current_positive_branch_family"
            if relaxed_upper < 0
            else "valid_rejection_inconclusive"
        )
        result.update(
            validity="valid",
            conditional_closed_loop_recruitment_proposal=proposal,
            hypothesis_verdict=verdict,
            solver_status=str(solution.status),
            solver_status_is_solved=str(solution.status) == "Solved",
            margin=float(x[-1]),
            metrics=metrics,
            baseline_metrics=saved["baseline_metrics"],
            gates=gates,
            strict_projected_margin_upper_bound=strict_upper,
            relaxed_projected_margin_upper_bound=relaxed_upper,
            solver_settings=settings,
        )
        progress["status"] = "complete"
    except BaseException as error:
        primary = error
        progress["status"] = "technical_failed"
        for name, arrays, already_saved in (
            ("solver-return", solver_return_raw, solver_return_saved),
            ("solver-derived", solver_derived_raw, solver_derived_saved),
            ("candidate", candidate_raw, candidate_saved),
        ):
            if arrays and not already_saved:
                try:
                    atomic_npz(
                        out / f"{name}-fallback.npz",
                        {key: np.asarray(value) for key, value in arrays.items()},
                    )
                except BaseException as save_error:
                    error.add_note(
                        "Secondary E169 " + name + " fallback: " + repr(save_error)
                    )
                    secondary(save_error, name + " fallback raw")
        if pre_raw and not pre_saved:
            try:
                atomic_npz(out / "input-partial-raw.npz", pre_raw)
            except BaseException as save_error:
                error.add_note("Secondary E169 input partial save: " + repr(save_error))
                secondary(save_error, "input partial raw")
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
            validate_freeze(freeze_path, freeze_sha256)
            validate_gate(source_review, source_review_sha256, freeze)
            verify_snapshot(out, freeze, freeze_path, source_review, approved)
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
                conditional_closed_loop_recruitment_proposal=False,
                hypothesis_verdict="invalid_not_tested",
                technical_failure=True,
            )
            if primary is not None:
                receipt = {
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "traceback": "".join(traceback.format_exception(primary)),
                }
                result["primary_error"] = receipt
                progress["primary_error"] = receipt

        if primary is not None:
            set_failure_receipt()
        else:
            result["technical_failure"] = False
        elapsed = time.perf_counter() - began
        result.update(
            elapsed_before_publication_s=elapsed,
            prior_conservative_original_s=freeze["prior_conservative_original_s"],
            prior_conservative_s=freeze["prior_conservative_s"],
            conservative_cumulative_before_publication_s=freeze["prior_conservative_s"]
            + elapsed,
            counts={
                key: progress[key]
                for key in (
                    "fit_calls",
                    "solver_constructor_attempts",
                    "solver_constructor_completed",
                    "callback_registration_attempts",
                    "callback_registration_completed",
                    "solver_returns",
                    "new_ema_rollout_steps",
                    "resource_state_update_steps",
                    "body_models",
                    "graph_store_instances",
                    "state_store_instances",
                    "physics_steps",
                    "neural_steps",
                )
            },
        )

        def quarantine_success() -> None:
            for name in ("result.json", "ledger.json", "output-manifest.json"):
                path = out / name
                if path.exists():
                    try:
                        path.replace(out / f"{name}.invalid-stale-{uuid.uuid4().hex}")
                    except BaseException as error:
                        secondary(error, "quarantine " + name)

        try:
            atomic_json(out / "ledger.json", progress)
            atomic_json(out / "result.json", result)
            atomic_json(out / "output-manifest.json", output_manifest(out))
            manifest = json.loads(
                (out / "output-manifest.json").read_text(encoding="utf-8")
            )
            for name, receipt in manifest["files"].items():
                path = out / name
                require(
                    sha256(path) == receipt["sha256"]
                    and path.stat().st_size == receipt["bytes"],
                    "E169 output manifest changed: " + name,
                )
            total = sum(
                path.stat().st_size for path in out.rglob("*") if path.is_file()
            )
            require(
                total <= freeze["resource_limits"]["raw_bytes"],
                "E169 output byte limit exceeded",
            )
            require(
                shutil.disk_usage(out).free >= freeze["resource_limits"]["free_bytes"],
                "E169 terminal free-space reserve unavailable",
            )
            disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
            require(
                disk["conditional_closed_loop_recruitment_proposal"]
                == result["conditional_closed_loop_recruitment_proposal"]
                and disk["validity"] == result["validity"],
                "E169 disk result differs",
            )
            deadline("after terminal publication")
        except BaseException as error:
            had_success = primary is None
            if primary is None:
                primary = error
            else:
                secondary(error, "terminal publication")
            set_failure_receipt()
            if had_success:
                quarantine_success()
            published = False
            for attempt in range(2):
                try:
                    atomic_json(out / "ledger.json", progress)
                    atomic_json(out / "result.json", result)
                    atomic_json(out / "output-manifest.json", output_manifest(out))
                    disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
                    require(
                        disk["validity"] == "incomplete"
                        and disk["conditional_closed_loop_recruitment_proposal"]
                        is False
                        and disk.get("primary_error") == result.get("primary_error"),
                        "E169 fail-closed receipt differs",
                    )
                    published = True
                    break
                except BaseException as rewrite_error:
                    secondary(rewrite_error, f"fail-closed publication {attempt + 1}")
            progress["failure_receipt_published"] = published
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
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.out,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
