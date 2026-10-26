"""E168 one-shot saved-signal resource attenuation SOCP."""

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

FREEZE_SHA = "8c38176963ff0ec4537d22e5f84717d736a39d26dab16c038d1f4e652f581177"
DESIGN_SHA = "3a0b2e58c2b4b5da33bef6089621082df7c5b4ab4e81a9638a083481a3b3b58a"
SCHEMA = "flylab.m3.e168.resource-attenuation.v1"
GATE_SCHEMA = "flylab.m3.e168.parent-source-review.v1"
OWNED = (
    "tools/fit_c_banc_m3_resource_attenuation.py",
    "tests/test_c_banc_resource_attenuation.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


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
    require(isinstance(name, str), "E168 frozen path must be text")
    relative = Path(name)
    path = (ROOT / relative).resolve()
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and path.is_relative_to(ROOT),
        "E168 unsafe frozen path",
    )
    return path


def hashes(freeze: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze[section]
        require(
            isinstance(mapping, dict) and mapping, "E168 missing hash map: " + section
        )
        for name, digest in mapping.items():
            require(
                name not in result or result[name] == digest,
                "E168 conflicting hash: " + name,
            )
            result[name] = digest
    return result


def validate_freeze(
    path: Path, digest: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    require(
        digest == FREEZE_SHA and sha256(path) == FREEZE_SHA,
        "Exact E168 freeze bytes required",
    )
    freeze = json.loads(path.read_text(encoding="utf-8"))
    require(
        freeze.get("schema") == SCHEMA
        and freeze.get("design_sha256") == DESIGN_SHA
        and freeze.get("owned_files") == list(OWNED),
        "E168 freeze identity differs",
    )
    require(sha256(frozen_path(freeze["design"])) == DESIGN_SHA, "E168 design changed")
    base_path = frozen_path(freeze["base_freeze"])
    require(
        freeze["base_freeze_sha256"] == e164.FREEZE_SHA
        and sha256(base_path) == e164.FREEZE_SHA,
        "E168 base freeze differs",
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    require(
        base["schema"] == e164.SCHEMA
        and base["sample_contract"] == freeze["sample_contract"]
        and base["body_hash"] == freeze["body_hash"]
        and base["paths"] == freeze["paths"],
        "E168/base sample contract differs",
    )
    c, t, limits = freeze["contract"], freeze["transfer"], freeze["resource_limits"]
    require(
        freeze["solver"] == "Clarabel"
        and freeze["solver_version"] == str(clarabel.__version__) == "0.11.1",
        "E168 solver version differs",
    )
    require(
        t
        == {
            "resource_tau_s": 0.07,
            "initial_resource": 1.0,
            "resource_bounds": [0.5, 1.0],
            "resource_update": "exact_held_endpoint_rate_then_current_contribution",
            "zero_rate_zero_kappa_drive": 0.0,
            "positive_rate_zero_kappa_drive": 1.0,
            "lambda_parameter_count": 330,
            "total_parameter_count": 331,
            "command_anchor": "recorded_command_plus_moment_activation_delta",
            "all_history_worst_depletion": 0.5,
        },
        "E168 transfer contract differs",
    )
    require(
        c["margin_bounds"] == [-1.0, 1.0]
        and c["lambda_bounds"] == [0.0, 1.0]
        and c["baseline_witness_margin"] == -0.251
        and c["proposal_command_RMS_ratio"] == 0.8
        and c["soc_count"] == 15
        and c["physical_atol"] == 1e-12,
        "E168 optimization contract differs",
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
        == (1, 0, 1, 0, 20, 40, 50, 55, 201326592, 1073741824),
        "E168 resource limits differ",
    )
    require(
        freeze["counts"]
        == {
            "body_models": 0,
            "state_store_instances": 0,
            "graph_store_instances": 0,
            "neural_steps": 0,
            "physics_steps": 0,
            "new_ema_rollout_steps": 0,
            "resource_state_update_steps": 2000,
            "fit_calls_max": 1,
        },
        "E168 counts differ",
    )
    require(
        freeze["prior_conservative_s"] == 2898
        and freeze["prospective_cumulative_s"] == 2953
        and freeze["original_resource_evaluations_before_freeze"] == 0
        and freeze["original_fit_calls_before_freeze"] == 0
        and freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "E168 prior/outcome contract differs",
    )
    merged = hashes(freeze)
    require(
        merged[freeze["base_freeze"]] == e164.FREEZE_SHA,
        "E168 base freeze hash missing",
    )
    base_hashes = e164.merged_frozen_hashes(base)
    require(
        all(merged.get(name) == value for name, value in base_hashes.items()),
        "E168 base references differ",
    )
    for name, value in merged.items():
        require(sha256(frozen_path(name)) == value, "E168 frozen hash changed: " + name)
    return freeze, base, merged


def validate_gate(path: Path, digest: str, freeze: dict[str, Any]) -> dict[str, Any]:
    require(sha256(path) == digest, "E168 source review SHA differs")
    gate = json.loads(path.read_text(encoding="utf-8"))
    require(
        gate.get("schema") == GATE_SCHEMA
        and gate.get("approved") is True
        and gate.get("blocking_defects") == []
        and gate.get("accepts_single_resource_fit") is True,
        "E168 source review blocks execution",
    )
    require(
        type(gate.get("original_processes_allowed")) is int
        and gate["original_processes_allowed"] == 1
        and type(gate.get("automatic_retries")) is int
        and gate["automatic_retries"] == 0,
        "E168 source review process contract differs",
    )
    require(
        gate.get("freeze_sha256") == FREEZE_SHA
        and gate.get("design_sha256") == DESIGN_SHA,
        "E168 source review provenance differs",
    )
    require(
        gate.get("source_sha256")
        == {name: sha256(frozen_path(name)) for name in OWNED},
        "E168 source review source hashes differ",
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
    snap = out / "source-snapshot"
    snap.mkdir()
    approved = {}
    for name, source in sources.items():
        digest = sha256(source)
        (snap / name).write_bytes(source.read_bytes())
        require(sha256(snap / name) == digest, "E168 snapshot write differs: " + name)
        approved[name] = digest
    require(
        approved["fitter.py"] == gate["source_sha256"][OWNED[0]]
        and approved["tests.py"] == gate["source_sha256"][OWNED[1]],
        "E168 approved snapshot differs",
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
            "E168 live/snapshot changed: " + name,
        )


def atomic_json(path: Path, value: Any) -> None:
    e164.atomic_json(path, value)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    e164.atomic_npz(path, arrays)


def numerical(ok: Any, message: str, pending: list[tuple[bool, str]]) -> None:
    pending.append((bool(ok), message))


def enforce(pending: list[tuple[bool, str]]) -> None:
    for ok, message in pending:
        require(ok, message)


INPUT_SPECS: dict[str, dict[str, tuple[tuple[int, ...], Any, bool]]] = {
    "recruitment": {
        "bank": ((2001, 391, 3), np.float64, True),
        "activation": ((2001, 96), np.float64, True),
        "original_activation": ((2001, 96), np.float64, True),
        "motor_rate": ((2001, 391), np.float64, True),
        "control_tick": ((2001,), np.int64, False),
        "network_tick": ((2001,), np.int64, False),
        "physics_time_s": ((2001,), np.float64, False),
        **{
            key: ((2001,), np.bool_, False)
            for key in (
                "complete",
                "bank_available",
                "activation_available",
                "original_activation_available",
                "motor_rate_available",
            )
        },
    },
    "adapter": {
        "membership": ((391,), np.int64, False),
        "motor_ids": ((391,), np.int64, False),
        "moment": ((42, 96), np.float64, False),
        "rest": ((42,), np.float64, False),
        "pool": ((96, 391), np.float64, True),
        "ltm": ((6,), np.int64, False),
    },
    "parameters": {
        "membership": ((391,), np.int64, False),
        "motor_ids": ((391,), np.int64, False),
        "slopes": ((96, 391, 3, 2), np.float64, True),
        "kappa": ((391,), np.float64, True),
        "tau": ((3,), np.float64, True),
    },
    "calibration": {
        "membership": ((391,), np.int64, False),
        "motor_ids": ((391,), np.int64, False),
        "moment": ((42, 96), np.float64, False),
        "rest": ((42,), np.float64, False),
        "pool": ((96, 391), np.float64, True),
        "ltm": ((6,), np.int64, False),
    },
    "phase_diagnostic": {
        "control_time_s": ((2001,), np.float64, False),
        "e1_phase": ((2001, 6), np.float64, False),
        "e1_phase_available": ((2001, 6), np.bool_, False),
    },
    "target_mask": {
        "control_time_s": ((2001,), np.float64, False),
        "mask": ((2001,), np.bool_, False),
        "target_indices": ((1824,), np.int64, False),
        "pair_start": ((1823,), np.int64, False),
        "pair_end": ((1823,), np.int64, False),
    },
    "reference": {
        "time_s": ((2000,), np.float64, False),
        "phase": ((2000, 6), np.float64, False),
        "command": ((2000, 42), np.float64, False),
    },
    "startup": {
        "mask": ((2001,), np.bool_, False),
        "target_command": ((99, 42), np.float64, False),
        "target_tendon": ((99, 6), np.float64, False),
        "recorded_command": ((2001, 42), np.float64, False),
        "recorded_tendon": ((2000, 6), np.float64, False),
    },
}


def capture_loaded_input(data: dict[str, Any], raw: dict[str, np.ndarray]) -> None:
    """Retain source dtypes and values before any E164 float/int coercion."""

    def capture(name: str, value: Any) -> None:
        try:
            array = np.asarray(value)
            raw[name + "_dtype"] = np.asarray(str(array.dtype))
            if array.dtype.kind == "O":
                raw[name + "_repr"] = np.asarray(repr(value)[:4096])
            else:
                raw[name] = array
        except BaseException as error:
            raw[name + "_capture_error"] = np.asarray(repr(error))

    for group, fields in INPUT_SPECS.items():
        values = data.get(group)
        if isinstance(values, dict):
            for key in fields:
                if key in values:
                    capture(f"loaded_{group}_{key}", values[key])
    state = data.get("state")
    adapter = state.get("adapter", {}) if isinstance(state, dict) else {}
    for key in ("bank", "activation"):
        value = data.get("state_initial_" + key, adapter.get(key))
        if value is not None:
            capture("loaded_state_initial_" + key, value)
    parameters = data.get("state_parameters")
    if isinstance(parameters, dict):
        for key in ("fixed_muscles", "kappa", "slopes", "membership", "tau"):
            if key in parameters:
                capture("loaded_state_parameters_" + key, parameters[key])
    if isinstance(adapter, dict) and "groups" in adapter:
        capture(
            "loaded_state_group_columns_json",
            json.dumps([group.get("columns") for group in adapter["groups"]]),
        )
    trace = data.get("trace")
    rows = trace.get("samples") if isinstance(trace, dict) else None
    if isinstance(rows, list):
        for name, extractor in (
            ("control_tick", lambda row: row.get("control_tick")),
            ("time_s", lambda row: row.get("time_s")),
            ("motor_rate", lambda row: row.get("motor_rates", row.get("motor_rate"))),
            ("recorded_command", lambda row: row.get("joint_targets_rad")),
            (
                "tendon",
                lambda row: [
                    row.get("tendon_inputs", {}).get(leg + "_tarsus") for leg in LEGS
                ],
            ),
        ):
            try:
                capture("loaded_trace_" + name, [extractor(row) for row in rows])
            except BaseException as error:
                raw["loaded_trace_" + name + "_capture_error"] = np.asarray(repr(error))


def validate_loaded_input(data: dict[str, Any], raw: dict[str, np.ndarray]) -> None:
    for group, fields in INPUT_SPECS.items():
        for key, (shape, dtype, nonnegative) in fields.items():
            name = f"loaded_{group}_{key}"
            require(name in raw, "E168 missing/unserializable input: " + name)
            array = raw[name]
            require(
                array.shape == shape and array.dtype == np.dtype(dtype),
                "E168 input shape/dtype differs: " + name,
            )
            if name != "loaded_phase_diagnostic_e1_phase":
                require(np.isfinite(array).all(), "E168 nonfinite input: " + name)
            if nonnegative:
                require(np.all(array >= 0), "E168 negative input: " + name)
    for key, shape, dtype in (
        ("loaded_state_initial_bank", (391, 3), np.float64),
        ("loaded_state_initial_activation", (96,), np.float64),
        ("loaded_trace_control_tick", (2001,), np.int64),
        ("loaded_trace_time_s", (2001,), np.float64),
        ("loaded_trace_motor_rate", (2001, 391), np.float64),
        ("loaded_trace_recorded_command", (2001, 42), np.float64),
        ("loaded_trace_tendon", (2001, 6), np.float64),
    ):
        require(key in raw, "E168 missing/unserializable input: " + key)
        array = raw[key]
        require(
            array.shape == shape
            and array.dtype == np.dtype(dtype)
            and np.isfinite(array).all(),
            "E168 input shape/dtype/finite differs: " + key,
        )
    for key, shape, dtype in (
        ("fixed_muscles", (19,), np.int64),
        ("kappa", (391,), np.float64),
        ("slopes", (96, 391, 3, 2), np.float64),
        ("membership", (391,), np.int64),
        ("tau", (3,), np.float64),
    ):
        if "state_parameters" in data:
            name = "loaded_state_parameters_" + key
            require(
                name in raw
                and raw[name].shape == shape
                and raw[name].dtype == np.dtype(dtype)
                and np.isfinite(raw[name]).all(),
                "E168 checkpoint parameter shape/dtype differs: " + name,
            )
    groups = data["state"]["adapter"]["groups"]
    require(
        isinstance(groups, list)
        and len(groups) == 96
        and all(
            isinstance(group, dict)
            and isinstance(group.get("columns"), list)
            and all(type(column) is int for column in group["columns"])
            for group in groups
        ),
        "E168 state group column type differs",
    )
    membership = raw["loaded_adapter_membership"]
    require(
        np.all((membership >= 0) & (membership < 96))
        and np.array_equal(membership, raw["loaded_parameters_membership"]),
        "E168 source ownership differs",
    )
    require(
        np.array_equal(raw["loaded_recruitment_control_tick"], np.arange(2001))
        and np.array_equal(
            raw["loaded_recruitment_network_tick"], np.arange(2001) * 20
        ),
        "E168 source tick receipt differs",
    )
    phase = raw["loaded_phase_diagnostic_e1_phase"]
    available = raw["loaded_phase_diagnostic_e1_phase_available"]
    targets = raw["loaded_target_mask_target_indices"]
    require(
        np.array_equal(available, np.isfinite(phase))
        and np.all(targets >= 0)
        and np.all(targets < 2001)
        and available[targets].all(),
        "E168 phase availability differs",
    )


def held_resource(
    rate: Any,
    kappa: Any,
    tau: float = 0.07,
    dt: float = 0.005,
    *,
    progress: dict[str, Any] | None = None,
    raw: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rate = np.asarray(rate)
    kappa = np.asarray(kappa)
    require(
        rate.ndim == 2
        and kappa.shape == (rate.shape[1],)
        and rate.dtype == np.float64
        and kappa.dtype == np.float64
        and np.isfinite(rate).all()
        and np.isfinite(kappa).all()
        and np.all(rate >= 0)
        and np.all(kappa >= 0),
        "E168 rate/kappa invalid",
    )
    require(tau == 0.07 and dt == 0.005, "E168 resource clock changed")
    with np.errstate(over="ignore"):
        total = rate + kappa[None, :]
    require(np.isfinite(total).all(), "E168 drive denominator overflow")
    drive = np.divide(rate, total, out=np.zeros_like(rate), where=total > 0)
    resource = np.full_like(rate, np.nan)
    resource[0] = 1.0
    if raw is not None:
        raw["resource"] = resource
        raw["resource_drive"] = drive
    if progress is not None:
        progress["resource_state_update_steps"] = 0
    for tick in range(1, rate.shape[0]):
        s = drive[tick]
        steady = 1.0 / (1.0 + s)
        resource[tick] = steady + (resource[tick - 1] - steady) * np.exp(
            -(1.0 + s) * dt / tau
        )
        if progress is not None:
            progress["resource_state_update_steps"] = tick
    return resource, drive


def positive_contribution(
    bank: Any, slopes: Any, kappa: Any, membership: Any
) -> np.ndarray:
    bank, slopes, kappa, membership = map(np.asarray, (bank, slopes, kappa, membership))
    require(
        bank.ndim == 3
        and bank.shape[1:] == (391, 3)
        and slopes.shape == (96, 391, 3, 2)
        and kappa.shape == (391,)
        and membership.shape == (391,),
        "E168 contribution shape differs",
    )
    require(
        bank.dtype == slopes.dtype == kappa.dtype == np.float64
        and np.isfinite(bank).all()
        and np.isfinite(slopes).all()
        and np.isfinite(kappa).all()
        and np.all(bank >= 0)
        and np.all(slopes >= 0)
        and np.all(kappa >= 0),
        "E168 contribution input invalid",
    )
    own = slopes[membership, np.arange(391)]
    low = np.minimum(bank, kappa[None, :, None])
    high = np.maximum(bank - kappa[None, :, None], 0.0)
    result = np.sum(own[None, :, :, 0] * low + own[None, :, :, 1] * high, axis=2)
    require(
        np.isfinite(result).all() and np.all(result >= 0), "E168 contribution invalid"
    )
    return result


def resource_features(
    data: dict[str, Any],
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r, p, a = data["recruitment"], data["parameters"], data["adapter"]
    rate = np.asarray(r["motor_rate"])
    bank = np.asarray(r["bank"])
    kappa = np.asarray(p["kappa"])
    membership = np.asarray(a["membership"])
    raw.update(
        endpoint_rate=rate,
        bank=bank,
        kappa=kappa,
        membership=membership,
        slopes=np.asarray(p["slopes"]),
    )
    require(
        rate.shape == (2001, 391)
        and rate.dtype == np.float64
        and np.isfinite(rate).all()
        and np.all(rate >= 0),
        "E168 endpoint rate invalid",
    )
    require(
        bank.shape == (2001, 391, 3)
        and bank.dtype == np.float64
        and np.isfinite(bank).all()
        and np.all(bank >= 0),
        "E168 bank invalid",
    )
    require(
        np.array_equal(membership, np.asarray(p["membership"]))
        and membership.dtype == np.int64,
        "E168 ownership invalid",
    )
    resource, drive = held_resource(rate, kappa, progress=progress, raw=raw)
    c = positive_contribution(bank, p["slopes"], kappa, membership)
    phi = (1.0 - resource) * c
    variable = np.asarray(freeze["sample_contract"]["variable_motors"], dtype=np.int64)
    fixed = np.asarray(freeze["sample_contract"]["fixed_muscles"], dtype=np.int64)
    variable_muscles = np.asarray(
        freeze["sample_contract"]["variable_muscles"], dtype=np.int64
    )
    require(
        variable.shape == (330,)
        and np.array_equal(variable, np.sort(variable))
        and np.array_equal(np.unique(membership[variable]), variable_muscles),
        "E168 variable graph columns differ",
    )
    mu = np.sum(
        np.min(np.asarray(p["slopes"])[membership, np.arange(391)], axis=2), axis=1
    )
    upper = np.sum(
        np.max(np.asarray(p["slopes"])[membership, np.arange(391)], axis=2), axis=1
    )
    reconstructed = np.zeros((2001, 96), dtype=np.float64)
    np.add.at(reconstructed.T, membership, c.T)
    raw.update(
        endpoint_rate=rate,
        bank=bank,
        kappa=kappa,
        membership=membership,
        resource=resource,
        resource_drive=drive,
        positive_contribution=c,
        attenuation_feature=phi,
        contribution_reconstructed_activation=reconstructed,
        contribution_residual=reconstructed[:, variable_muscles]
        - saved["activation"][:, variable_muscles],
        mu=mu,
        upper_slope=upper,
        variable_motors=variable,
        fixed_muscles=fixed,
    )
    numerical(
        np.min(resource) >= 0.5
        and np.max(resource) <= 1.0
        and np.array_equal(resource[0], np.ones(391)),
        "E168 resource bounds differ",
        pending,
    )
    numerical(
        np.array_equal(phi[0], np.zeros(391))
        and np.all(phi >= 0)
        and np.all(phi <= 0.5 * c),
        "E168 attenuation feature bounds differ",
        pending,
    )
    numerical(
        np.max(np.abs(raw["contribution_residual"])) <= 1e-12,
        "E168 positive contribution reconstruction differs",
        pending,
    )
    n_by_muscle = np.bincount(membership, minlength=96)
    numerical(
        np.all(mu[variable] > 0)
        and np.all(mu[variable] >= 1e-6 / n_by_muscle[membership[variable]] - 1e-12),
        "E168 baseline per-MN floor unavailable",
        pending,
    )
    for muscle in variable_muscles:
        members = np.flatnonzero(membership == muscle)
        numerical(
            np.sum(mu[members]) >= 0.5 - 1e-12 and np.sum(upper[members]) <= 2 + 1e-12,
            "E168 baseline muscle budget unavailable",
            pending,
        )
    saved.update(
        resource=resource,
        drive=drive,
        contribution=c,
        phi=phi,
        membership=membership,
        variable_motors=variable,
        fixed_muscles=fixed,
        mu=mu,
        upper_slope=upper,
    )
    return saved


def candidate(
    saved: dict[str, Any], lam: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lam = np.asarray(lam, dtype=np.float64)
    variable = saved["variable_motors"]
    require(
        lam.shape == (330,) and np.isfinite(lam).all(), "E168 lambda vector invalid"
    )
    delta = np.zeros_like(saved["activation"])
    np.add.at(
        delta.T,
        saved["membership"][variable],
        (-saved["phi"][:, variable] * lam[None, :]).T,
    )
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


def leg_affine(
    saved: dict[str, Any], leg: int, rows: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    members = saved["variable_motors"]
    owners = saved["membership"][members]
    joints = np.arange(7 * leg, 7 * (leg + 1))
    active = np.flatnonzero(
        np.isin(owners, np.flatnonzero(np.asarray(saved["group_legs"]) == leg))
    )
    local = (
        -0.02
        * saved["phi"][np.ix_(rows, members[active])][:, None, :]
        * saved["moment"][np.ix_(joints, owners[active])][None, :, :]
    )
    selector = sparse.csr_matrix(
        (np.ones(active.size), (np.arange(active.size), active)),
        shape=(active.size, 330),
    )
    matrix = sparse.csr_matrix(local.reshape(rows.size * 7, active.size)) @ selector
    constant = saved["recorded_command"][np.ix_(rows, joints)].reshape(-1)
    return matrix, constant


def compress(
    name: str,
    blocks: list[tuple[sparse.csr_matrix, np.ndarray]],
    limit: float,
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
) -> e164.Criterion:
    matrices, offsets, qrs = [], [], []
    witnesses = (
        np.zeros(330),
        np.linspace(-0.375, 0.625, 330),
        np.cos(np.arange(330) * 0.173),
    )
    for number, (matrix, offset) in enumerate(blocks):
        active = np.unique(matrix.indices)
        q = e145.thin_qr_compress(matrix[:, active].toarray(), offset)
        qm, qo = e145.compressed_affine(q)
        selector = sparse.csr_matrix(
            (np.ones(active.size), (np.arange(active.size), active)),
            shape=(active.size, 330),
        )
        compressed = sparse.csr_matrix(qm) @ selector
        direct = np.asarray([np.linalg.norm(matrix @ w + offset) for w in witnesses])
        compact = np.asarray([np.linalg.norm(compressed @ w + qo) for w in witnesses])
        raw.update(
            {
                f"qr_{name}_{number}": q,
                f"qr_{name}_{number}_active": active,
                f"qr_{name}_{number}_direct_norms": direct,
                f"qr_{name}_{number}_compressed_norms": compact,
            }
        )
        numerical(
            np.all(
                np.abs(direct - compact)
                <= freeze["contract"]["QR_absolute_atol"]
                + freeze["contract"]["QR_relative_atol"] * direct
            ),
            "E168 QR witness differs: " + name,
            pending,
        )
        matrices.append(compressed)
        offsets.append(qo)
        qrs.append(q)
    return e164.Criterion(
        name,
        sparse.vstack(matrices, format="csr"),
        np.concatenate(offsets),
        float(limit),
        tuple(qrs),
    )


def build_problem(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    raw: dict[str, np.ndarray],
    pending: list[tuple[bool, str]],
) -> e164.ConicProblem:
    indices = saved["target_indices"]
    starts, ends = saved["pair_start"], saved["pair_end"]
    command_blocks, velocity_blocks, startup_blocks = [], [], []
    for leg, name in enumerate(LEGS):
        joints = slice(7 * leg, 7 * (leg + 1))
        cm, cb = leg_affine(saved, leg, indices)
        command_offset = cb - saved["target"][:, joints].reshape(-1)
        m0, b0 = leg_affine(saved, leg, starts)
        m1, b1 = leg_affine(saved, leg, ends)
        vm = (m1 - m0) / 0.005
        velocity_offset = (b1 - b0) / 0.005 - saved["target_velocity"][
            :, joints
        ].reshape(-1)
        sm, sb = leg_affine(saved, leg, np.arange(1, 100))
        startup_offset = sb - saved["startup_target"][:, joints].reshape(-1)
        command_blocks.append((cm, command_offset))
        velocity_blocks.append((vm, velocity_offset))
        startup_blocks.append((sm, startup_offset))
        for label, matrix, offset in (
            ("command", cm, command_offset),
            ("velocity", vm, velocity_offset),
            ("startup", sm, startup_offset),
        ):
            raw.update(
                {
                    f"{label}_{name}_data": matrix.data,
                    f"{label}_{name}_indices": matrix.indices,
                    f"{label}_{name}_indptr": matrix.indptr,
                    f"{label}_{name}_shape": np.asarray(matrix.shape),
                    f"{label}_{name}_offset": offset,
                }
            )
    baseline, c = saved["baseline_metrics"], freeze["contract"]
    criteria = [
        compress(
            "overall_command",
            command_blocks,
            math.sqrt(indices.size * 42) * baseline["overall_command_RMS"] * 0.8,
            freeze,
            raw,
            pending,
        ),
        compress(
            "overall_velocity",
            velocity_blocks,
            math.sqrt(starts.size * 42) * baseline["overall_velocity_RMS"],
            freeze,
            raw,
            pending,
        ),
    ]
    for leg, name in enumerate(LEGS):
        criteria.append(
            compress(
                f"{name}_command",
                [command_blocks[leg]],
                math.sqrt(indices.size * 7) * baseline["by_leg"][name]["command_RMS"],
                freeze,
                raw,
                pending,
            )
        )
        criteria.append(
            compress(
                f"{name}_velocity",
                [velocity_blocks[leg]],
                math.sqrt(starts.size * 7) * baseline["by_leg"][name]["velocity_RMS"],
                freeze,
                raw,
                pending,
            )
        )
    criteria.append(
        compress(
            "startup",
            startup_blocks,
            math.sqrt(99 * 42) * c["startup_limits"]["q_RMS"],
            freeze,
            raw,
            pending,
        )
    )
    require(len(criteria) == c["soc_count"] == 15, "E168 SOC count differs")
    startup_matrix = sparse.vstack([block[0] for block in startup_blocks], format="csr")
    startup_offset = np.concatenate([block[1] for block in startup_blocks])
    lower = np.r_[np.zeros(330), -1.0]
    upper = np.r_[np.ones(330), 1.0]
    rows = [-sparse.eye(331, format="csr"), sparse.eye(331, format="csr")]
    rhs = [-lower, upper]
    membership, variable, mu = (
        saved["membership"],
        saved["variable_motors"],
        saved["mu"],
    )
    n_by_muscle = np.bincount(membership, minlength=96)
    for column, motor in enumerate(variable):
        muscle = membership[motor]
        # -(mu - .5 mu lambda) <= -floor
        row = sparse.csr_matrix(([0.5 * mu[motor]], ([0], [column])), shape=(1, 331))
        rows.append(row)
        rhs.append(
            np.asarray([mu[motor] - c["motor_floor_fraction"] / n_by_muscle[muscle]])
        )
    for muscle in freeze["sample_contract"]["variable_muscles"]:
        local = np.flatnonzero(membership[variable] == muscle)
        motors = variable[local]
        row = sparse.csr_matrix(
            (0.5 * mu[motors], (np.zeros(local.size), local)), shape=(1, 331)
        )
        rows.append(row)
        rhs.append(np.asarray([np.sum(mu[motors]) - c["tonic_lower_budget"]]))
    tcol = sparse.csr_matrix(np.ones((startup_matrix.shape[0], 1)))
    scale = c["startup_limits"]["q_max_abs"]
    require(scale == 0.04, "E168 startup max changed")
    rows.extend(
        [
            sparse.hstack((startup_matrix / scale, tcol)),
            sparse.hstack((-startup_matrix / scale, tcol)),
        ]
    )
    rhs.extend([1 - startup_offset / scale, 1 + startup_offset / scale])
    linear_a = sparse.vstack(rows, format="csr")
    linear_b = np.concatenate(rhs)
    parts_a, parts_b = [linear_a], [linear_b]
    cones: list[Any] = [clarabel.NonnegativeConeT(linear_a.shape[0])]
    kinds, dims = ["NonnegativeConeT"], [linear_a.shape[0]]
    for criterion in criteria:
        require(
            criterion.limit > 0 and np.isfinite(criterion.limit),
            "E168 zero/invalid SOC limit: " + criterion.name,
        )
        soc_a, soc_b = e145.make_soc_rows(
            criterion.matrix, criterion.offset, criterion.limit, 330
        )
        parts_a.append(soc_a)
        parts_b.append(soc_b)
        cones.append(clarabel.SecondOrderConeT(soc_a.shape[0]))
        kinds.append("SecondOrderConeT")
        dims.append(soc_a.shape[0])
    problem = e164.ConicProblem(
        sparse.csc_matrix((331, 331)),
        np.r_[np.zeros(330), -1.0],
        sparse.vstack(parts_a, format="csc"),
        np.concatenate(parts_b),
        tuple(cones),
        tuple(kinds),
        np.asarray(dims),
        lower,
        upper,
        tuple(criteria),
        startup_matrix,
        startup_offset,
    )
    witness = np.r_[np.zeros(330), c["baseline_witness_margin"]]
    slack = problem.b - problem.A @ witness
    raw.update(
        {
            "baseline_witness": witness,
            "baseline_witness_slack": slack,
            "linear_row_count": np.asarray(linear_a.shape[0]),
            "cone_kinds": np.asarray(kinds),
            "cone_dims": np.asarray(dims),
            **{
                f"problem_{key}": value
                for key, value in e164.problem_arrays(problem).items()
            },
        }
    )
    numerical(
        np.all(slack[: linear_a.shape[0]] >= -c["physical_atol"]),
        "E168 baseline witness linear infeasible",
        pending,
    )
    position = linear_a.shape[0]
    for criterion, dim in zip(criteria, dims[1:], strict=True):
        section = slack[position : position + dim]
        numerical(
            section[0] >= np.linalg.norm(section[1:]) - c["physical_atol"],
            "E168 baseline witness SOC infeasible: " + criterion.name,
            pending,
        )
        position += dim
    return problem


def candidate_gates(
    saved: dict[str, Any],
    freeze: dict[str, Any],
    x: np.ndarray,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    c = freeze["contract"]
    baseline = saved["baseline_metrics"]
    lam = x[:330]
    variable, membership, mu = (
        saved["variable_motors"],
        saved["membership"],
        saved["mu"],
    )
    n_by_muscle = np.bincount(membership, minlength=96)
    floors = (1 - 0.5 * lam) * mu[variable]
    muscle_budget = [
        np.sum(floors[membership[variable] == muscle])
        >= c["tonic_lower_budget"] - c["physical_atol"]
        for muscle in freeze["sample_contract"]["variable_muscles"]
    ]
    return {
        "lambda_exact_bounds": bool(np.all((lam >= 0) & (lam <= 1))),
        "margin_bounds": bool(-1 <= x[-1] <= 1),
        "all_history_motor_floor": bool(
            np.all(
                floors
                >= c["motor_floor_fraction"] / n_by_muscle[membership[variable]]
                - c["physical_atol"]
            )
        ),
        "all_history_muscle_budget": bool(all(muscle_budget)),
        "upper_budget_invariant": bool(
            all(
                np.sum(saved["upper_slope"][membership == muscle])
                <= c["total_upper_budget"] + c["physical_atol"]
                for muscle in freeze["sample_contract"]["variable_muscles"]
            )
        ),
        "fixed_exact": bool(
            np.array_equal(
                arrays["candidate_activation"][:, saved["fixed_muscles"]],
                saved["activation"][:, saved["fixed_muscles"]],
            )
        ),
        "activation_nonnegative": bool(
            np.all(arrays["candidate_activation"] >= -c["physical_atol"])
        ),
        "decoder_reconstruction": bool(
            np.max(np.abs(arrays["decoder_residual"])) <= c["physical_atol"]
        ),
        "overall_command_20pct": metrics["overall_command_RMS"]
        <= c["proposal_command_RMS_ratio"] * baseline["overall_command_RMS"],
        "overall_velocity_nonincrease": metrics["overall_velocity_RMS"]
        <= baseline["overall_velocity_RMS"],
        "all_leg_nonincrease": all(
            metrics["by_leg"][leg][kind] <= baseline["by_leg"][leg][kind]
            for leg in LEGS
            for kind in ("command_RMS", "velocity_RMS")
        ),
        "startup_q": metrics["startup_q_RMS"] <= c["startup_limits"]["q_RMS"]
        and metrics["startup_q_max_abs"] <= c["startup_limits"]["q_max_abs"],
        "startup_LTM": saved["startup_ltm_RMS"] <= c["startup_limits"]["LTM_RMS"]
        and saved["startup_ltm_max_abs"] <= c["startup_limits"]["LTM_max_abs"],
    }


def output_manifest(out: Path) -> dict[str, Any]:
    files = {
        p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "output-manifest.json"
    }
    return {
        "schema": "flylab.m3.e168.output-manifest.v1",
        "files": files,
        "total_bytes": sum(v["bytes"] for v in files.values()),
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
        not out.exists() and out.parent.is_dir(), "E168 output must be a new directory"
    )
    require(
        shutil.disk_usage(out.parent).free >= freeze["resource_limits"]["free_bytes"],
        "E168 free-space reserve unavailable",
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
        "resource_state_update_steps": 0,
        "new_ema_rollout_steps": 0,
        "body_models": 0,
        "graph_store_instances": 0,
        "state_store_instances": 0,
        "physics_steps": 0,
        "neural_steps": 0,
        "secondary_errors": [],
    }
    result: dict[str, Any] = {
        "schema": "flylab.m3.e168.result.v1",
        "validity": "incomplete",
        "conditional_resource_proposal": False,
        "hypothesis_verdict": "invalid_not_tested",
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
    }
    primary: BaseException | None = None
    thread_limit = None
    pre_raw: dict[str, np.ndarray] = {}
    pre_saved = False

    def persist() -> None:
        atomic_json(out / "ledger.json", progress)

    def deadline(stage: str) -> None:
        if time.perf_counter() - began > freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E168 internal deadline: " + stage)

    def secondary(error: BaseException, stage: str) -> None:
        progress["secondary_errors"].append({"stage": stage, "error": repr(error)})
        if primary is not None:
            primary.add_note("Secondary E168 " + stage + ": " + repr(error))

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
            "E168 CPU threads exceed one",
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
        input_checks: list[tuple[bool, str]] = []
        saved = e164.prepare_saved(data, base, pre_raw, input_checks)
        saved["group_legs"] = np.asarray(
            freeze["sample_contract"]["group_legs"], dtype=np.int64
        )
        resource_features(data, saved, freeze, pre_raw, input_checks, progress)
        zero_metrics, zero_arrays = candidate(saved, np.zeros(330))
        pre_raw.update(
            lambda_zero_command_residual=zero_arrays["candidate_command"]
            - saved["recorded_command"],
            lambda_zero_activation_residual=zero_arrays["candidate_activation"]
            - saved["activation"],
            lambda_zero_decoder_residual=zero_arrays["decoder_residual"],
        )
        numerical(
            np.array_equal(zero_arrays["candidate_command"], saved["recorded_command"])
            and np.array_equal(
                zero_arrays["candidate_activation"], saved["activation"]
            ),
            "E168 lambda0 does not preserve baseline exactly",
            input_checks,
        )
        numerical(
            np.max(np.abs(zero_arrays["decoder_residual"]))
            <= freeze["contract"]["physical_atol"],
            "E168 baseline decoder residual differs",
            input_checks,
        )
        numerical(
            zero_metrics == saved["baseline_metrics"],
            "E168 lambda0 metrics differ",
            input_checks,
        )
        e164.save_check_evidence(
            out / "pre-solve-raw.npz", pre_raw, "input", input_checks
        )
        pre_saved = True
        enforce(input_checks)
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
        partial: dict[str, Any] = {}
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
                        "Secondary E168 solver partial save: " + repr(save_error)
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
        require(
            progress["fit_wall_s"] <= freeze["resource_limits"]["fit_s"],
            "E168 fit deadline exceeded",
        )
        derived = e164.solver_raw(problem, solution, history, freeze)
        atomic_npz(out / "solver-derived-raw.npz", derived)
        x = np.asarray(solution.x, dtype=np.float64)
        require(
            x.shape == (331,) and np.isfinite(x).all(), "E168 returned vector invalid"
        )
        metrics, arrays = candidate(saved, x[:330])
        arrays.update(
            returned_vector=x,
            candidate_resource=saved["resource"],
            candidate_attenuation_feature=saved["phi"],
        )
        atomic_npz(out / "candidate-raw.npz", arrays)
        gates = candidate_gates(saved, freeze, x, metrics, arrays)
        proposal = all(gates.values())
        projected_upper = float(derived["projected_dual_margin_upper_bound"])
        require(math.isfinite(projected_upper), "E168 projected dual bound invalid")
        verdict = (
            "conditional_resource_proposal"
            if proposal
            else (
                "refuted_fixed_resource_family"
                if projected_upper < 0
                else "valid_rejection_inconclusive"
            )
        )
        result.update(
            validity="valid",
            conditional_resource_proposal=proposal,
            hypothesis_verdict=verdict,
            solver_status=str(solution.status),
            solver_status_is_solved=str(solution.status) == "Solved",
            margin=float(x[-1]),
            metrics=metrics,
            baseline_metrics=saved["baseline_metrics"],
            gates=gates,
            dual_projected_margin_upper_bound=projected_upper,
            solver_settings=settings,
        )
        progress["status"] = "complete"
    except BaseException as error:
        primary = error
        progress["status"] = "technical_failed"
        if pre_raw and not pre_saved:
            try:
                atomic_npz(out / "input-partial-raw.npz", pre_raw)
            except BaseException as save_error:
                error.add_note("Secondary E168 input partial save: " + repr(save_error))
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
                conditional_resource_proposal=False,
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
        elapsed_before_publication = time.perf_counter() - began
        result.update(
            elapsed_before_publication_s=elapsed_before_publication,
            prior_conservative_original_s=freeze["prior_conservative_original_s"],
            prior_conservative_s=freeze["prior_conservative_s"],
            conservative_cumulative_before_publication_s=freeze["prior_conservative_s"]
            + elapsed_before_publication,
            counts={
                key: progress[key]
                for key in (
                    "fit_calls",
                    "solver_constructor_attempts",
                    "solver_constructor_completed",
                    "callback_registration_attempts",
                    "callback_registration_completed",
                    "solver_returns",
                    "resource_state_update_steps",
                    "new_ema_rollout_steps",
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
                    "E168 output manifest changed: " + name,
                )
            total = sum(
                path.stat().st_size for path in out.rglob("*") if path.is_file()
            )
            require(
                total <= freeze["resource_limits"]["raw_bytes"],
                "E168 output byte limit exceeded",
            )
            require(
                shutil.disk_usage(out).free >= freeze["resource_limits"]["free_bytes"],
                "E168 terminal free-space reserve unavailable",
            )
            disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
            require(
                disk["conditional_resource_proposal"]
                == result["conditional_resource_proposal"]
                and disk["validity"] == result["validity"],
                "E168 disk result differs",
            )
            # This deadline includes all final hashes, writes, size/free checks and readback.
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
                        and disk["conditional_resource_proposal"] is False
                        and disk.get("primary_error") == result.get("primary_error"),
                        "E168 fail-closed receipt differs",
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
    try:
        result = run(
            args.freeze,
            args.freeze_sha256,
            args.source_review,
            args.source_review_sha256,
            args.out,
        )
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "technical_failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
