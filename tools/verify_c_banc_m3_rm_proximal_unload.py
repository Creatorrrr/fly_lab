"""E162 analytic RM proximal-unload selection and one gated body contrast."""

# Broad catches preserve the first scientific/storage failure while attempting
# every remaining raw and receipt publication.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verify_c_banc_m3_rm_singleton_kinetics as e160

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "ce941db555ff29e33b45438accf8f3d40e50f1a727106fa7fbc236f82709ee34"
DESIGN_SHA = "f98b99b2cc2a696c8ddfec1f819c93218996884db1af4e50516cb42685389000"
OWNED = (
    "tools/verify_c_banc_m3_rm_proximal_unload.py",
    "tests/test_c_banc_rm_proximal_unload.py",
)
CONTROL_DT = 0.005
PHYSICS_DT = 0.0001
N_NATIVE = 100000
LEGS = e160.LEGS
TENDONS = e160.TENDONS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(relative: str) -> Path:
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("E162 repository-relative path required")
    return path


def atomic_json(path: Path, value: Any) -> None:
    temporary = Path(path).with_name(Path(path).name + f".{uuid.uuid4().hex}.tmp")
    # A failed write or replace leaves its unique temporary bytes as evidence.
    # Cleanup must not hide the primary storage error.
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = Path(path).with_name(Path(path).name + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def exact_hash_map(mapping: dict[str, str], label: str) -> None:
    require(isinstance(mapping, dict), f"E162 {label} map required")
    for name, digest in mapping.items():
        require(sha256(safe_path(name)) == digest, f"E162 {label} changed: {name}")


def verified_inputs(
    freeze_path: str | Path,
    freeze_sha256: str,
    review_path: str | Path,
    review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Own E162 gate; never call or patch E160's verified_inputs/run."""
    require(freeze_sha256 == FREEZE_SHA, "Exact E162 freeze SHA argument required")
    require(sha256(freeze_path) == FREEZE_SHA, "Exact E162 freeze bytes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    fixed = {
        "schema": "flylab.m3.e162.rm-proximal-unload.v1",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "body_hash": "1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f",
        "original99_selection_exposures_max": 1,
        "original99_independent_confirmation": False,
        "candidate_values_seen_before_freeze": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "original_runs_before_gate": 0,
        "implementation": {
            "model": "gpt-5.6-sol",
            "effort": "high",
            "agent": "/root/m3_sol_e132_lf_constrained",
        },
    }
    require(
        all(freeze.get(k) == v for k, v in fixed.items()),
        "E162 frozen contract changed",
    )
    require(
        freeze.get("intervention")
        == {
            "muscle": 70,
            "motor_columns": [192, 245],
            "graph_indices": [77368, 99838],
            "joint_row": 32,
            "moment": 1.0,
            "command_gain": 0.02,
            "lambda_min": 0.0,
            "lambda_max": 0.25,
            "positive_upper_margin": 1e-12,
            "selection": "largest analytic intersection of original99 q RMS/max constraints",
            "candidate_formula": "baseline_command - lambda * .02 * moment[:,70] * activation[:,70]",
            "startup_indices": [1, 99],
            "preserve_initial_observation_exact": True,
        },
        "E162 intervention changed",
    )
    require(
        freeze.get("counts")
        == {
            "original_processes": 1,
            "automatic_retries": 0,
            "body_constructor_max": 1,
            "shadow_mjdata_max": 1,
            "settling_steps_max": 2000,
            "scientific_steps_max": 100000,
            "shadow_forward_calls_max": 100000,
            "neural_steps": 0,
            "fit_calls": 0,
        },
        "E162 counts changed",
    )
    require(
        freeze.get("resource_limits")
        == {
            "internal_s": 260,
            "external_s": 280,
            "supervisor_s": 295,
            "raw_bytes": 1073741824,
            "free_bytes": 4294967296,
            "cpu_threads": 1,
            "gpu_count": 0,
        },
        "E162 resources changed",
    )
    require(len(freeze.get("input_sha256", {})) == 213, "E162 input count changed")
    require(
        len(freeze.get("reused_source_sha256", {})) == 27,
        "E162 reused-source count changed",
    )
    require(
        len(freeze.get("preservation_sha256", {})) == 3150,
        "E162 preservation count changed",
    )
    require(sha256(review_path) == review_sha256, "Exact E162 source review required")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    require(
        (
            review.get("approved") is True
            or review.get("accepts_single_rm_proximal_unload") is True
        )
        and review.get("freeze_sha256") == FREEZE_SHA
        and review.get("blocking_defects", []) == []
        and set(review.get("source_sha256", {})) == set(OWNED),
        "Accepted exact E162 two-source gate required",
    )
    for name, digest in review["source_sha256"].items():
        require(
            sha256(safe_path(name)) == digest, "Reviewed E162 source changed: " + name
        )
    for key in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        exact_hash_map(freeze[key], key)
    return freeze, review


def stable_quadratic_interval(
    a: float, b: float, c: float
) -> tuple[float, float, float]:
    """Solve ``a*x*x+b*x+c<=0`` for a nonnegative quadratic coefficient."""
    require(
        all(math.isfinite(v) for v in (a, b, c)) and a >= 0,
        "Finite convex quadratic required",
    )
    if a == 0.0:
        if b == 0.0:
            return (
                (-math.inf, math.inf, 0.0)
                if c <= 0.0
                else (math.inf, -math.inf, -math.inf)
            )
        root = -c / b
        return (-math.inf, root, math.inf) if b > 0 else (root, math.inf, math.inf)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return (
            (-math.inf, math.inf, discriminant)
            if c <= 0.0
            else (math.inf, -math.inf, discriminant)
        )
    root_disc = math.sqrt(max(0.0, discriminant))
    if root_disc == 0.0:
        root = -b / (2.0 * a)
        return root, root, discriminant
    q = -0.5 * (b + math.copysign(root_disc, b))
    if q == 0.0:
        lower = (-b - root_disc) / (2.0 * a)
        upper = (-b + root_disc) / (2.0 * a)
    else:
        roots = (q / a, c / q)
        lower, upper = min(roots), max(roots)
    return lower, upper, discriminant


def analytic_lambda_intersection(
    baseline99: Any,
    target99: Any,
    direction99: Any,
    *,
    max_abs: float = 0.04,
    rms: float = 0.02,
    lambda_min: float = 0.0,
    lambda_max: float = 0.25,
    positive_upper_margin: float = 1e-12,
) -> dict[str, Any]:
    """Return the largest predeclared lambda in all max/RMS intervals."""
    q = np.asarray(baseline99, dtype=np.float64)
    target = np.asarray(target99, dtype=np.float64)
    direction = np.asarray(direction99, dtype=np.float64)
    require(
        q.shape == target.shape == direction.shape == (99, 42),
        "Exact 99x42 lambda arrays required",
    )
    require(
        np.isfinite(q).all()
        and np.isfinite(target).all()
        and np.isfinite(direction).all(),
        "Finite lambda arrays required",
    )
    require(
        0 <= lambda_min <= lambda_max and max_abs > 0 and rms > 0,
        "Valid lambda bounds required",
    )
    residual = q - target
    baseline_max = float(np.max(np.abs(residual)))
    residual_square_mean = (
        math.fsum(float(value) * float(value) for value in residual.flat)
        / residual.size
    )
    baseline_rms = math.sqrt(residual_square_mean)
    baseline_feasible = baseline_max <= max_abs and residual_square_mean <= rms * rms
    lower = np.full(q.shape, -math.inf)
    upper = np.full(q.shape, math.inf)
    nonzero = direction != 0.0
    lo_root = np.empty(np.count_nonzero(nonzero))
    hi_root = np.empty(np.count_nonzero(nonzero))
    a0 = residual[nonzero]
    d0 = direction[nonzero]
    lo_root[:] = (-max_abs - a0) / d0
    hi_root[:] = (max_abs - a0) / d0
    lower[nonzero] = np.minimum(lo_root, hi_root)
    upper[nonzero] = np.maximum(lo_root, hi_root)
    impossible_zero = (~nonzero) & (np.abs(residual) > max_abs)
    lower[impossible_zero] = math.inf
    upper[impossible_zero] = -math.inf
    count = residual.size
    qa = math.fsum(float(value) * float(value) for value in direction.flat) / count
    qb = (
        2.0
        * math.fsum(
            float(error) * float(delta)
            for error, delta in zip(residual.flat, direction.flat, strict=True)
        )
        / count
    )
    qc = residual_square_mean - rms * rms
    qlower, qupper, discriminant = stable_quadratic_interval(qa, qb, qc)
    intersection_lower = max(lambda_min, float(np.max(lower)), qlower)
    intersection_upper = min(lambda_max, float(np.min(upper)), qupper)
    feasible = bool(
        baseline_feasible
        and intersection_lower <= intersection_upper
        and intersection_upper >= 0
    )
    unmargin_upper = max(0.0, intersection_upper) if feasible else 0.0
    selected = (
        unmargin_upper * (1.0 - positive_upper_margin) if unmargin_upper > 0 else 0.0
    )
    if selected < max(0.0, intersection_lower):
        selected = 0.0
        feasible = False
    atol = 32 * np.finfo(np.float64).eps * max(1.0, abs(unmargin_upper))
    active_element = np.isfinite(upper) & (np.abs(upper - unmargin_upper) <= atol)
    return {
        "lambda": float(selected),
        "gain_scale": float(1.0 - selected),
        "feasible": feasible,
        "baseline_feasible": baseline_feasible,
        "baseline_max_abs": baseline_max,
        "baseline_RMS": baseline_rms,
        "intersection_lower": float(intersection_lower),
        "intersection_upper_unmargined": float(intersection_upper),
        "element_lower": lower,
        "element_upper": upper,
        "element_active_upper": active_element,
        "quadratic_a": qa,
        "quadratic_b": qb,
        "quadratic_c": qc,
        "quadratic_discriminant": discriminant,
        "quadratic_lower": float(qlower),
        "quadratic_upper": float(qupper),
        "quadratic_active_upper": bool(
            math.isfinite(qupper) and abs(qupper - unmargin_upper) <= atol
        ),
        "hard_upper_active": bool(abs(lambda_max - unmargin_upper) <= atol),
        "residual": residual,
        "direction": direction,
        "selection_exposures": 1,
        "independent_confirmation": False,
    }


def validate_group70_mapping(
    parameters: dict[str, np.ndarray],
    provenance: dict[str, np.ndarray],
    intervention: dict[str, Any],
) -> None:
    membership = np.asarray(parameters["membership"])
    motor_ids = np.asarray(parameters["motor_ids"])
    columns = np.asarray(intervention["motor_columns"], dtype=np.int64)
    require(
        membership.shape == motor_ids.shape == (391,), "E162 391-MN mapping required"
    )
    require(
        np.array_equal(np.flatnonzero(membership == intervention["muscle"]), columns)
        and np.array_equal(motor_ids[columns], intervention["graph_indices"])
        and np.array_equal(membership, provenance["membership"])
        and np.array_equal(motor_ids, provenance["motor_ids"]),
        "E162 group70 global/local mapping differs",
    )
    moment = np.asarray(provenance["moment"], dtype=np.float64)
    column = moment[:, intervention["muscle"]]
    expected = np.zeros(42)
    expected[intervention["joint_row"]] = intervention["moment"]
    require(np.array_equal(column, expected), "E162 group70 moment column differs")


def build_proximal_candidate(
    baseline_command: Any,
    activation: Any,
    moment: Any,
    lambda_value: float,
    *,
    muscle: int = 70,
    gain: float = 0.02,
) -> dict[str, np.ndarray]:
    """Apply constant efficacy reduction on ticks1..2000, preserving tick0."""
    command = np.asarray(baseline_command)
    activation = np.asarray(activation)
    moment = np.asarray(moment)
    require(
        command.shape == (2001, 42) and command.dtype == np.float64,
        "Float64 2001x42 baseline required",
    )
    require(
        activation.shape == (2001, 96) and activation.dtype == np.float64,
        "Float64 2001x96 activation required",
    )
    require(
        moment.shape == (42, 96) and moment.dtype == np.float64,
        "Float64 42x96 moment required",
    )
    require(
        np.isfinite(command).all()
        and np.isfinite(activation).all()
        and np.isfinite(moment).all(),
        "Finite candidate inputs required",
    )
    require(
        math.isfinite(lambda_value) and 0 <= lambda_value <= 0.25,
        "Bounded E162 lambda required",
    )
    direction = -gain * activation[:, muscle, None] * moment[:, muscle][None, :]
    direction[0] = 0.0
    candidate = command + lambda_value * direction
    candidate[0] = command[0]
    require(np.array_equal(candidate[0], command[0]), "E162 tick0 observation changed")
    unaffected = moment[:, muscle] == 0
    require(
        np.array_equal(candidate[:, unaffected], command[:, unaffected]),
        "E162 non-target joints changed",
    )
    return {"candidate_command": candidate, "direction": direction}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _trace_arrays(
    rows: list[dict[str, Any]], clock_atol: float
) -> tuple[np.ndarray, np.ndarray]:
    command = np.asarray([row["joint_targets_rad"] for row in rows], dtype=np.float64)
    tendon = np.asarray(
        [[row["tendon_inputs"][key] for key in TENDONS] for row in rows],
        dtype=np.float64,
    )
    require(
        command.shape == (2001, 42) and tendon.shape == (2001, 6),
        "Complete E150 commands required",
    )
    require(
        np.isfinite(command).all() and np.isfinite(tendon).all(),
        "Finite E150 commands required",
    )
    ticks = np.asarray([row["control_tick"] for row in rows])
    clocks = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    require(
        ticks.shape == clocks.shape == (2001,), "Complete E150 trace receipts required"
    )
    require(np.array_equal(ticks, np.arange(2001)), "Exact control ticks required")
    require(
        np.isfinite(clocks).all()
        and np.allclose(
            clocks, np.arange(2001) * CONTROL_DT, rtol=0.0, atol=clock_atol
        ),
        "Control clock outside frozen tolerance",
    )
    return command, tendon


def _startup_targets(
    startup: dict[str, np.ndarray],
    baseline: np.ndarray,
    tendon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qkey = (
        "original_E100_E103_command"
        if "original_E100_E103_command" in startup
        else "target_command"
    )
    lkey = (
        "original_E100_E103_LTM"
        if "original_E100_E103_LTM" in startup
        else "target_tendon"
    )
    q, ltm = (
        np.asarray(startup[qkey], dtype=np.float64),
        np.asarray(startup[lkey], dtype=np.float64),
    )
    require(
        q.shape == (99, 42) and ltm.shape == (99, 6),
        "Exact original99 targets required",
    )
    require(
        np.array_equal(np.asarray(startup["actual_command"]), baseline[1:100])
        and np.array_equal(np.asarray(startup["actual_LTM"]), tendon[1:100]),
        "Stored actual startup does not equal E150 trace indices1:100",
    )
    return q, ltm


def json_selection(selection: dict[str, Any]) -> dict[str, Any]:
    """Represent unbounded scalar intervals explicitly without JSON infinities."""
    result: dict[str, Any] = {}
    for key, value in selection.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = {
                "unbounded": "positive_infinity" if value > 0 else "negative_infinity"
            }
        else:
            result[key] = value
    return result


def output_manifest(out: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "output-manifest.json"
    ):
        files[path.relative_to(out).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "flylab.m3.e162.output-manifest.v1",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "terminal_receipts": ["ledger.json", "result.json"],
    }


def run(
    freeze_path: str | Path,
    freeze_sha256: str,
    source_review: str | Path,
    source_review_sha256: str,
    output: str | Path,
    *,
    body_runner: Callable[
        ...,
        tuple[
            list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]
        ],
    ]
    | None = None,
    dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    began = time.perf_counter()
    freeze, _review = verified_inputs(
        freeze_path, freeze_sha256, source_review, source_review_sha256
    )
    out = Path(output).resolve()
    require(not out.exists(), "E162 output must be new")
    require(
        shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
        "E162 free-space reserve unavailable",
    )
    out.mkdir(parents=True)
    snapshot = out / "source-snapshot"
    snapshot.mkdir()
    snapshot_files: dict[str, str] = {}
    for name in OWNED:
        source = ROOT / name
        target = snapshot / Path(name).name
        shutil.copyfile(source, target)
        snapshot_files[target.name] = sha256(target)
    for label, source_name in (
        ("freeze.json", freeze_path),
        ("source-review.json", source_review),
    ):
        source = Path(source_name)
        if source.is_file():
            target = snapshot / label
            shutil.copyfile(source, target)
            snapshot_files[label] = sha256(target)
    atomic_json(
        snapshot / "snapshot-manifest.json",
        {"schema": "flylab.m3.e162.source-snapshot.v1", "files": snapshot_files},
    )
    ledger = {
        "schema": "flylab.m3.e162.ledger.v1",
        "status": "created",
        "original_processes": 1,
        "selection_exposures": 0,
        "neural_steps": 0,
        "fit_calls": 0,
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "settling_physics_steps": 0,
        "body_models": 0,
        "shadow_mjdata": 0,
        "scientific_steps_requested": 0,
        "scientific_steps_completed": 0,
        "scientific_samples_observed": 0,
        "shadow_forward_requested": 0,
        "shadow_forward_completed": 0,
        "cleanup_attempted": False,
        "cleanup_completed": False,
        "secondary_errors": [],
    }
    result: dict[str, Any] = {
        "schema": "flylab.m3.e162.result.v1",
        "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested",
        "advance_to_closed_loop": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "active_UI_changed": False,
        "body_executed": False,
        "physical_metrics": None,
    }
    primary: BaseException | None = None
    health_stop: e160.NativeHealthStop | None = None
    body = None
    thread_limits = None

    def persist() -> None:
        atomic_json(out / "ledger.json", ledger)

    def deadline(label: str) -> None:
        if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E162 internal deadline: " + label)

    def secondary(
        error: BaseException, stage: str, owner: BaseException | None = None
    ) -> None:
        ledger["secondary_errors"].append({"stage": stage, "error": repr(error)})
        if owner is not None:
            owner.add_note("Secondary E162 " + stage + ": " + repr(error))

    def save_partial(error: BaseException) -> None:
        failed = False
        for label, value, operation in (
            (
                "partial native",
                getattr(error, "e158_native", None),
                lambda x: atomic_npz(out / "candidate-native-partial.npz", **x),
            ),
            (
                "partial trace",
                getattr(error, "e158_trace_rows", None),
                lambda x: atomic_json(
                    out / "candidate-trace-partial.json",
                    {"schema": "flylab.m3.e162.partial-trace.v1", "samples": x},
                ),
            ),
        ):
            if value is not None:
                try:
                    operation(value)
                except BaseException as save_error:
                    failed = True
                    secondary(save_error, label, error)
        error.e162_partial_save_failed = failed

    try:
        thread_limits = threadpool_limits(limits=1)
        ledger["cpu_thread_environment"] = {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        ledger["threadpools_during"] = threadpool_info()
        require(
            all(value == "1" for value in ledger["cpu_thread_environment"].values())
            and all(
                pool.get("num_threads") == 1 for pool in ledger["threadpools_during"]
            ),
            "E162 CPU thread count exceeds one",
        )
        persist()
        payload = json.loads(safe_path(freeze["trace"]).read_text(encoding="utf-8"))
        rows = payload.get("samples")
        require(
            isinstance(rows, list) and len(rows) == 2001, "Complete E150 trace required"
        )
        recruitment = _load_npz(safe_path(freeze["recruitment"]))
        provenance = _load_npz(safe_path(freeze["provenance"]))
        parameters = _load_npz(safe_path(freeze["parameters"]))
        startup = _load_npz(safe_path(freeze["startup"]))
        validate_group70_mapping(parameters, provenance, freeze["intervention"])
        clock_atol = float(freeze["sample_contract"]["clock_atol"])
        baseline, tendon = _trace_arrays(rows, clock_atol)
        activation = np.asarray(recruitment["activation"])
        require(
            activation.shape == (2001, 96)
            and activation.dtype == np.float64
            and np.isfinite(activation).all(),
            "Complete E150 activation required",
        )
        for key in ("complete", "activation_available"):
            receipt = np.asarray(recruitment[key])
            require(
                receipt.shape == (2001,)
                and receipt.dtype == np.bool_
                and receipt.all(),
                "Incomplete E150 recruitment availability: " + key,
            )
        receipt_ticks = np.asarray(recruitment["control_tick"])
        receipt_clocks = np.asarray(recruitment["physics_time_s"])
        require(
            receipt_ticks.shape == receipt_clocks.shape == (2001,)
            and np.issubdtype(receipt_ticks.dtype, np.integer)
            and np.array_equal(receipt_ticks, np.arange(2001))
            and np.issubdtype(receipt_clocks.dtype, np.floating)
            and np.isfinite(receipt_clocks).all()
            and np.allclose(
                receipt_clocks,
                np.arange(2001) * CONTROL_DT,
                rtol=0.0,
                atol=clock_atol,
            ),
            "E150 recruitment clock/ticks differ",
        )
        rest = np.asarray(provenance["rest"], dtype=np.float64)
        moment = np.asarray(provenance["moment"], dtype=np.float64)
        require(
            rest.shape == (42,)
            and moment.shape == (42, 96)
            and np.isfinite(rest).all()
            and np.isfinite(moment).all(),
            "Finite E150 decoder provenance required",
        )
        reconstructed = rest[None, :] + freeze["intervention"]["command_gain"] * (
            activation @ moment.T
        )
        reconstruction_error = baseline - reconstructed
        atomic_npz(
            out / "baseline-reconstruction.npz",
            time_s=np.asarray([row["time_s"] for row in rows], dtype=np.float64),
            observed_command=baseline,
            reconstructed_command=reconstructed,
            reconstruction_error=reconstruction_error,
            observed_tick0=baseline[0],
            reconstructed_tick0=reconstructed[0],
        )
        require(
            np.allclose(
                baseline[1:],
                reconstructed[1:],
                rtol=0.0,
                atol=freeze["sample_contract"]["reconstruction_atol"],
            ),
            "E150 baseline command reconstruction differs on ticks1:2000",
        )
        original99, original_ltm99 = _startup_targets(startup, baseline, tendon)
        ledger["selection_exposures"] = 1
        ledger["status"] = "selection_requested"
        persist()
        unit = build_proximal_candidate(
            baseline, activation, provenance["moment"], 0.0, muscle=70, gain=0.02
        )
        selection = analytic_lambda_intersection(
            baseline[1:100],
            original99,
            unit["direction"][1:100],
            max_abs=freeze["startup_limits"]["q_max"],
            rms=freeze["startup_limits"]["q_RMS"],
            lambda_min=freeze["intervention"]["lambda_min"],
            lambda_max=freeze["intervention"]["lambda_max"],
            positive_upper_margin=freeze["intervention"]["positive_upper_margin"],
        )
        raw_selection = {
            key: value
            for key, value in selection.items()
            if isinstance(value, np.ndarray)
        }
        raw_selection.update(
            {
                key: np.asarray(value)
                for key, value in selection.items()
                if isinstance(value, (float, int, bool))
            }
        )
        atomic_npz(out / "lambda-constraints.npz", **raw_selection)
        candidate_values = build_proximal_candidate(
            baseline,
            activation,
            provenance["moment"],
            selection["lambda"],
            muscle=70,
            gain=0.02,
        )
        candidate = candidate_values["candidate_command"]
        effective_activation = activation.copy()
        effective_activation[1:, freeze["intervention"]["muscle"]] *= selection[
            "gain_scale"
        ]
        candidate_reconstructed = rest[None, :] + freeze["intervention"][
            "command_gain"
        ] * (effective_activation @ moment.T)
        atomic_npz(
            out / "candidate-commands.npz",
            time_s=np.arange(2001) * CONTROL_DT,
            baseline_command=baseline,
            candidate_command=candidate,
            measured_activation=activation,
            baseline_activation=activation,
            candidate_effective_activation=effective_activation,
            candidate_activation=effective_activation,
            candidate_reconstructed_command=candidate_reconstructed,
            candidate_reconstruction_error=candidate - candidate_reconstructed,
            candidate_direction=candidate_values["direction"],
            baseline_tendon=tendon,
            candidate_tendon=tendon.copy(),
            lambda_value=np.asarray(selection["lambda"]),
            gain_scale=np.asarray(selection["gain_scale"]),
        )
        require(
            np.allclose(
                candidate[1:],
                candidate_reconstructed[1:],
                rtol=0.0,
                atol=freeze["sample_contract"]["reconstruction_atol"],
            ),
            "E162 candidate command reconstruction differs on ticks1:2000",
        )
        screen = e160.startup_compatibility(
            candidate, tendon, original99, original_ltm99, freeze["startup_limits"]
        )
        atomic_npz(
            out / "startup-original99.npz",
            time_s=np.arange(1, 100) * CONTROL_DT,
            candidate_command=candidate[1:100],
            candidate_LTM=tendon[1:100],
            original_command=original99,
            original_LTM=original_ltm99,
            command_error=candidate[1:100] - original99,
            LTM_error=tendon[1:100] - original_ltm99,
        )
        atomic_json(out / "startup-screen.json", screen)
        result.update(
            lambda_value=selection["lambda"],
            gain_scale=selection["gain_scale"],
            selection=json_selection(selection),
            startup=screen,
            startup_pass=screen["startup_pass"],
            selection_exposures=1,
            original99_independent_confirmation=False,
        )
        ledger["status"] = "selection_complete"
        persist()
        if not selection["baseline_feasible"]:
            result.update(
                validity="valid_baseline_startup_failure",
                hypothesis_verdict="not_physically_tested_baseline_failure",
                stop_reason="baseline_startup_failure",
            )
            ledger["status"] = "baseline_startup_failure"
        elif selection["lambda"] == 0.0:
            result.update(
                validity="valid_lambda_zero_no_body",
                hypothesis_verdict="not_physically_tested_zero_intervention",
                stop_reason="lambda_zero",
            )
            ledger["status"] = "lambda_zero"
        elif not screen["startup_pass"]:
            result.update(
                validity="valid_candidate_startup_rejected",
                hypothesis_verdict="not_physically_tested_startup_rejected",
                stop_reason="candidate_startup_rejected",
            )
            ledger["status"] = "candidate_startup_rejected"
        else:
            deadline("before body")
            if body_runner is None:
                deps = (
                    e160._default_body_dependencies()
                    if dependencies is None
                    else dependencies
                )
                saved = deps["state_load"](safe_path(freeze["checkpoint"]))
                require(
                    saved.get("control_tick") == 0
                    and saved.get("body", {}).get("modelHash") == freeze["body_hash"],
                    "E150 start checkpoint differs",
                )
                implicit = deps["implicit_request"](payload, saved)
                body = deps["construct_body"](saved, deps["observation"], ledger)
                ledger["body_models"] = 1
                require(
                    abs(float(body.m.opt.timestep) - PHYSICS_DT) <= 1e-15,
                    "E162 physics dt differs",
                )
                for key in ("qpos", "qvel", "ctrl"):
                    require(
                        np.array_equal(
                            np.asarray(getattr(body.d, key)), np.asarray(rows[0][key])
                        ),
                        "E150 start state differs: " + key,
                    )
                require(
                    abs(float(body.physics_time())) <= e160.CLOCK_ATOL,
                    "E150 start clock differs",
                )
                shadow = copy.copy(body)
                shadow.d = body.mj.MjData(body.m)
                ledger["shadow_mjdata"] = 1
                try:
                    trace_rows, native = deps["execute_body"](
                        body=body,
                        shadow=shadow,
                        rows=rows,
                        candidate_command=candidate,
                        implicit_request=implicit,
                        replay_command=deps["replay_command"],
                        observe_shadow=deps["observe_shadow"],
                        contact_diagnostics=deps["contact_diagnostics"],
                        capture_control=deps["capture_control"],
                        ledger=ledger,
                        deadline_check=lambda: deadline("body replay"),
                    )
                except BaseException as error:
                    save_partial(error)
                    raise
                atomic_npz(out / "candidate-native.npz", **native)
                candidate_payload = copy.deepcopy(payload)
                candidate_payload.update(
                    schema="flylab.m3.e162.rm-proximal-unload-trace.v1",
                    controller="E150 saved activation with fixed group70 efficacy reduction",
                    samples=trace_rows,
                    E162_provenance={
                        "lambda": selection["lambda"],
                        "gain_scale": selection["gain_scale"],
                        "neural_steps": 0,
                    },
                )
                atomic_json(out / "candidate-trace.json", candidate_payload)
                quality_raw, _, _ = deps["gait"].analyze(
                    deps["gait"].load_model(out / "candidate-trace.json")
                )
                quality = deps["e113"].body_quality(
                    trace_rows,
                    list(
                        zip(
                            native["time_s"],
                            native["normal_load_bw"],
                            native["slip_mm_s"],
                            native["slip_valid"],
                            native["nonfoot_contact"],
                            native["upright"],
                            strict=True,
                        )
                    ),
                    quality_raw,
                    reason="duration",
                    fault=False,
                    clock_valid=True,
                    future_exact=False,
                )
                quality["longest_forward_bout_s"] = quality_raw["window"][
                    "longest_forward_bout_s"
                ]
                slip = deps["e113"].supported_slip(
                    native, complete=quality["complete_10s_native_health"]
                )
            else:
                try:
                    trace_rows, native, quality, slip = body_runner(
                        rows=rows,
                        candidate_command=candidate,
                        ledger=ledger,
                        deadline_check=lambda: deadline("stub body"),
                    )
                except BaseException as error:
                    save_partial(error)
                    raise
                atomic_npz(out / "candidate-native.npz", **native)
            require(
                len(trace_rows) == 2001 and len(native["time_s"]) == N_NATIVE,
                "Complete E162 body raw required",
            )
            e160.validate_complete_counts(ledger)
            impulses = e160.impulse_summary(
                native["time_s"], native["force_forward_bw"]
            )
            baseline_force = _load_npz(safe_path(freeze["baseline_forces"]))
            baseline_impulses = e160.impulse_summary(
                baseline_force["time_s"], baseline_force["force_forward_bw"]
            )
            baseline_dir = safe_path(freeze["baseline_directory"])
            baseline_quality = json.loads(
                (baseline_dir / "quality-summary.json").read_text(encoding="utf-8")
            )
            baseline_slip = json.loads(
                (baseline_dir / "supported-slip.json").read_text(encoding="utf-8")
            )
            decision = e160.advancement_decision(
                complete_health=quality.get("complete_10s_native_health") is True,
                quality=quality,
                slip=slip,
                baseline_quality=baseline_quality,
                baseline_slip=baseline_slip,
                impulses=impulses,
                baseline_impulses=baseline_impulses,
                thresholds=freeze["advancement"],
            )
            # The measured longest bout comes from NumPy, so its comparison
            # yields np.bool_. Preserve the frozen decisions as native scalars.
            require(
                all(
                    isinstance(value, (bool, np.bool_))
                    for value in decision["criteria"].values()
                ),
                "Boolean advancement criteria required",
            )
            decision["criteria"] = {
                name: bool(value) for name, value in decision["criteria"].items()
            }
            atomic_json(out / "quality-summary.json", quality)
            atomic_json(out / "supported-slip.json", slip)
            atomic_json(
                out / "impulses.json",
                {"candidate": impulses, "baseline": baseline_impulses},
            )
            atomic_json(out / "advancement.json", decision)
            result.update(
                validity="valid_body_contrast",
                hypothesis_verdict="supported_for_closed_loop_followup"
                if decision["advance_to_closed_loop"]
                else "advancement_criteria_not_met",
                stop_reason="duration",
                body_executed=True,
                physical_metrics={
                    "quality": quality,
                    "supported_slip": slip,
                    "impulses": impulses,
                    "advancement": decision,
                },
                advance_to_closed_loop=decision["advance_to_closed_loop"],
            )
            ledger["status"] = "body_complete"
    except e160.NativeHealthStop as error:
        if getattr(error, "e162_partial_save_failed", False):
            primary = error
            ledger["status"] = "technical_failed"
        else:
            health_stop = error
            ledger["status"] = "measured_native_health_stop"
    except BaseException as error:
        primary = error
        ledger["status"] = "technical_failed"
    finally:

        def record_primary(error: BaseException, stage: str) -> None:
            nonlocal primary
            if primary is None:
                primary = error
            else:
                secondary(error, stage, primary)
            ledger["status"] = "technical_failed"

        def update_receipts() -> None:
            elapsed = time.perf_counter() - began
            result.update(
                process_wall_s=elapsed,
                prior_conservative_actual_s=freeze["prior_actual_original_s"],
                prior_conservative_s=freeze["prior_conservative_s"],
                conservative_cumulative_s=freeze["prior_conservative_s"] + elapsed,
                counts={
                    key: ledger[key]
                    for key in (
                        "neural_steps",
                        "fit_calls",
                        "body_constructor_attempts",
                        "body_constructor_completed",
                        "settling_physics_steps",
                        "body_models",
                        "shadow_mjdata",
                        "scientific_steps_requested",
                        "scientific_steps_completed",
                        "scientific_samples_observed",
                        "shadow_forward_requested",
                        "shadow_forward_completed",
                    )
                },
                cpu_thread_environment=copy.deepcopy(
                    ledger.get("cpu_thread_environment")
                ),
                threadpools_during=copy.deepcopy(ledger.get("threadpools_during")),
                body_executed=bool(
                    result.get("body_executed")
                    or ledger["scientific_steps_completed"] > 0
                ),
                candidate_adopted=False,
                whole_goal_complete=False,
                F6_exposed=False,
                active_UI_changed=False,
            )
            if primary is not None:
                result["advance_to_closed_loop"] = False
                metrics = result.get("physical_metrics")
                decision = (
                    metrics.get("advancement") if isinstance(metrics, dict) else None
                )
                if isinstance(decision, dict):
                    decision["advance_to_closed_loop"] = False
                    decision["revoked_by"] = "technical_or_resource_failure"
                failure = {
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                }
                result.update(
                    validity="incomplete",
                    hypothesis_verdict="invalid_not_tested",
                    technical_failure=True,
                    primary_error=failure,
                )
                ledger["primary_error"] = failure
            elif health_stop is not None:
                result.update(
                    validity="measured_native_health_stop",
                    hypothesis_verdict="refuted_health_stop",
                    technical_failure=False,
                    stop_reason="native_health_stop",
                    advance_to_closed_loop=False,
                    physical_metrics={
                        "complete": False,
                        "native_samples_completed": ledger[
                            "scientific_steps_completed"
                        ],
                        "health_stop": str(health_stop),
                    },
                )
            else:
                result["technical_failure"] = False

        if body is not None:
            ledger["cleanup_attempted"] = True
            try:
                body.close()
                ledger["cleanup_completed"] = True
            except BaseException as error:
                record_primary(error, "body close")
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
                ledger["threadpools_after"] = threadpool_info()
            except BaseException as error:
                record_primary(error, "threadpool restore")

        # Integrity is terminal work, including measured health stops and all
        # partial/error paths. The gate also rechecks owned source/freeze/review.
        try:
            verified_inputs(
                freeze_path,
                freeze_sha256,
                source_review,
                source_review_sha256,
            )
            ledger["post_hashes_verified"] = True
        except BaseException as error:
            record_primary(error, "terminal immutable verification")

        def quarantine(path: Path, stage: str) -> None:
            require(
                path.parent == out, "Canonical receipt must stay in output directory"
            )
            if not path.exists():
                return
            target = path.with_name(
                path.name + f".invalid-stale-{uuid.uuid4().hex}.json"
            )
            try:
                path.replace(target)
            except BaseException as error:
                secondary(error, stage + " quarantine rename", primary)
                try:
                    target.write_bytes(path.read_bytes())
                    require(
                        sha256(target) == sha256(path),
                        "Quarantined receipt copy differs",
                    )
                    path.unlink()
                except BaseException as copy_error:
                    secondary(copy_error, stage + " quarantine copy/unlink", primary)

        def publish_terminal() -> None:
            update_receipts()
            decision = None
            metrics = result.get("physical_metrics")
            if isinstance(metrics, dict):
                decision = metrics.get("advancement")
            if primary is not None and isinstance(decision, dict):
                atomic_json(out / "advancement.json", decision)
            persist()
            atomic_json(out / "result.json", result)
            manifest = output_manifest(out)
            atomic_json(out / "output-manifest.json", manifest)

            # Publication is complete only after resource/deadline checks that
            # include every terminal receipt, followed by disk assertions.
            raw_bytes = sum(
                path.stat().st_size for path in out.rglob("*") if path.is_file()
            )
            require(
                raw_bytes <= freeze["resource_limits"]["raw_bytes"],
                "E162 raw-byte limit exceeded",
            )
            deadline("after terminal publication")
            disk_result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            require(
                disk_result["validity"] == result["validity"]
                and disk_result["advance_to_closed_loop"]
                == result["advance_to_closed_loop"],
                "E162 terminal result differs on disk",
            )
            if primary is not None:
                require(
                    disk_result["validity"] == "incomplete"
                    and disk_result["advance_to_closed_loop"] is False,
                    "E162 failed run left successful result",
                )
                if (out / "advancement.json").exists():
                    disk_advance = json.loads(
                        (out / "advancement.json").read_text(encoding="utf-8")
                    )
                    require(
                        disk_advance["advance_to_closed_loop"] is False,
                        "E162 failed run left successful advancement",
                    )
            disk_manifest = json.loads(
                (out / "output-manifest.json").read_text(encoding="utf-8")
            )
            for name, item in disk_manifest["files"].items():
                require(
                    sha256(out / name) == item["sha256"],
                    "E162 terminal manifest mismatch: " + name,
                )

        published = False
        for publication_attempt in range(2):
            try:
                publish_terminal()
                published = True
                break
            except BaseException as error:
                record_primary(error, f"terminal publication {publication_attempt + 1}")
                update_receipts()
                for name in (
                    "output-manifest.json",
                    "result.json",
                    "advancement.json",
                    "ledger.json",
                ):
                    quarantine(out / name, name)
        if not published:
            # No canonical receipt may retain a stale successful statement.
            for name in (
                "output-manifest.json",
                "result.json",
                "advancement.json",
                "ledger.json",
            ):
                quarantine(out / name, "terminal " + name)
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
