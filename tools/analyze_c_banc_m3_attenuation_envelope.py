"""E167 optimistic, pointwise attenuation envelope on a frozen E150 recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e167.attenuation-envelope.v1"
GATE_SCHEMA = "flylab.m3.e167.parent-source-review.v1"
FREEZE_SHA = "99719219647887ec0feb59594749773f743b3c9ac568b5ec028c5978be4ebd34"
OWNED = (
    "tools/analyze_c_banc_m3_attenuation_envelope.py",
    "tests/test_c_banc_attenuation_envelope.py",
)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def require(ok: Any, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def frozen_path(name: str) -> Path:
    require(isinstance(name, str), "Frozen path must be a string")
    relative = Path(name)
    require(
        not relative.is_absolute() and ".." not in relative.parts, "Unsafe frozen path"
    )
    return ROOT / relative


def frozen_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for section in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze[section]
        require(
            isinstance(mapping, dict) and bool(mapping), "Frozen hash mapping missing"
        )
        for name, digest in mapping.items():
            require(
                name not in hashes or hashes[name] == digest, "Frozen hash conflict"
            )
            hashes[name] = digest
    return hashes


def verify_hashes(hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        require(sha256(frozen_path(name)) == digest, "Frozen file changed: " + name)


def validate_freeze(freeze: dict[str, Any], path: Path, digest: str) -> dict[str, Any]:
    require(
        digest == FREEZE_SHA and sha256(path) == FREEZE_SHA,
        "Exact E167 freeze required",
    )
    require(
        freeze.get("schema") == SCHEMA and freeze.get("owned_files") == list(OWNED),
        "E167 freeze identity differs",
    )
    require(
        sha256(frozen_path(freeze["design"])) == freeze["design_sha256"],
        "E167 design changed",
    )
    c = freeze["sample_contract"]
    require(
        c["legs"] == list(LEGS) and c["control_rows"] == 2001,
        "E167 clock contract changed",
    )
    require(
        (c["motor_cells"], c["muscles"], c["joint_targets"]) == (391, 96, 42),
        "E167 anatomy contract changed",
    )
    require(
        (c["control_dt"], c["command_gain"], c["maximum_attenuation"])
        == (0.005, 0.02, 0.5),
        "E167 scale contract changed",
    )
    require(
        (c["target_count"], c["target_first_tick"], c["target_last_tick"])
        == (1824, 100, 1923),
        "E167 target contract changed",
    )
    require(
        (
            c["padding_floor_rad"],
            c["padding_epsilon_multiplier"],
            c["decision_margin_rad"],
            c["required_rms_fraction"],
        )
        == (1e-10, 4096, 1e-9, 0.8),
        "E167 decision contract changed",
    )
    fixed = np.asarray(c["fixed_muscles"])
    variable = np.asarray(c["variable_muscles"])
    require(
        fixed.shape == (19,)
        and variable.shape == (77,)
        and np.array_equal(np.sort(np.r_[fixed, variable]), np.arange(96)),
        "E167 muscle partition differs",
    )
    limits = freeze["resource_limits"]
    require(
        (
            limits["original_processes"],
            limits["automatic_retries"],
            limits["cpu_threads"],
            limits["gpu_count"],
        )
        == (1, 0, 1, 0),
        "E167 resource contract changed",
    )
    require(
        (
            limits["internal_s"],
            limits["external_s"],
            limits["supervisor_s"],
            limits["raw_bytes"],
            limits["free_bytes"],
        )
        == (40, 60, 70, 100663296, 1073741824),
        "E167 resource limits changed",
    )
    require(
        freeze["candidate_adopted"] is False
        and freeze["whole_goal_complete"] is False
        and freeze["F6_exposed"] is False,
        "E167 outcome contract changed",
    )
    return c


def validate_gate(
    path: Path, digest: str, freeze_sha: str, design_sha: str
) -> dict[str, Any]:
    require(sha256(path) == digest, "E167 parent gate SHA differs")
    gate = json.loads(path.read_text(encoding="utf-8"))
    source = {name: sha256(ROOT / name) for name in OWNED}
    require(gate.get("schema") == GATE_SCHEMA, "E167 parent gate schema differs")
    require(
        gate.get("approved") is True and gate.get("blocking_defects") == [],
        "E167 parent gate blocks execution",
    )
    require(
        gate.get("accepts_single_attenuation_envelope") is True,
        "E167 parent did not authorize single envelope",
    )
    require(
        type(gate.get("original_processes_allowed")) is int
        and gate["original_processes_allowed"] == 1,
        "E167 process gate differs",
    )
    require(
        type(gate.get("automatic_retries")) is int and gate["automatic_retries"] == 0,
        "E167 retry gate differs",
    )
    require(
        gate.get("freeze_sha256") == freeze_sha
        and gate.get("design_sha256") == design_sha,
        "E167 parent gate provenance differs",
    )
    require(gate.get("source_sha256") == source, "E167 source gate differs")
    return gate


def snapshot(
    out: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    gate_path: Path,
    gate_sha: str,
    gate: dict[str, Any],
) -> dict[str, str]:
    files = {
        "design.md": (frozen_path(freeze["design"]), freeze["design_sha256"]),
        "freeze.json": (freeze_path, sha256(freeze_path)),
        "source-review.json": (gate_path, gate_sha),
        "analyzer.py": (ROOT / OWNED[0], gate["source_sha256"][OWNED[0]]),
        "tests.py": (ROOT / OWNED[1], gate["source_sha256"][OWNED[1]]),
    }
    directory = out / "source-snapshot"
    directory.mkdir()
    for name, (source, digest) in files.items():
        require(sha256(source) == digest, "E167 live source changed: " + name)
        (directory / name).write_bytes(source.read_bytes())
        require(
            sha256(directory / name) == digest, "E167 source snapshot differs: " + name
        )
    return {name: digest for name, (_, digest) in files.items()}


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
        "analyzer.py": ROOT / OWNED[0],
        "tests.py": ROOT / OWNED[1],
    }
    for name, source in sources.items():
        require(
            sha256(source) == approved[name], "E167 post live source changed: " + name
        )
        require(
            sha256(out / "source-snapshot" / name) == approved[name],
            "E167 post snapshot changed: " + name,
        )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_inputs(freeze: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: load_npz(frozen_path(path))
        for name, path in freeze["paths"].items()
        if path.endswith(".npz")
    }


def raw_inputs(
    data: dict[str, dict[str, np.ndarray]], c: dict[str, Any]
) -> dict[str, np.ndarray]:
    fields = {
        "bank": ("recruitment", "bank"),
        "activation": ("recruitment", "activation"),
        "original_activation": ("recruitment", "original_activation"),
        "complete": ("recruitment", "complete"),
        "bank_available": ("recruitment", "bank_available"),
        "activation_available": ("recruitment", "activation_available"),
        "original_activation_available": (
            "recruitment",
            "original_activation_available",
        ),
        "motor_rate_available": ("recruitment", "motor_rate_available"),
        "control_tick": ("recruitment", "control_tick"),
        "network_tick": ("recruitment", "network_tick"),
        "time_s": ("recruitment", "physics_time_s"),
        "prior_bank": ("prior_raw", "bank"),
        "prior_activation": ("prior_raw", "reconstructed_activation"),
        "prior_activation_residual": ("prior_raw", "activation_residual"),
        "target_indices": ("prior_raw", "target_indices"),
        "target": ("prior_raw", "target"),
        "recorded_command": ("prior_raw", "recorded_command"),
        "moment": ("adapter", "moment"),
        "rest": ("adapter", "rest"),
        "membership": ("adapter", "membership"),
        "parameter_membership": ("parameters", "membership"),
        "slopes": ("parameters", "slopes"),
        "kappa": ("parameters", "kappa"),
    }
    raw = {name: data[section][key] for name, (section, key) in fields.items()}
    raw["contract_fixed_muscles"] = np.asarray(c["fixed_muscles"], dtype=np.int64)
    return raw


def array(
    raw: dict[str, np.ndarray], key: str, shape: tuple[int, ...], kind: str
) -> np.ndarray:
    value = raw[key]
    require(value.shape == shape, key + " shape differs")
    valid = {
        "float": value.dtype == np.float64,
        "int": value.dtype == np.int64,
        "bool": value.dtype == np.bool_,
    }[kind]
    require(valid, key + " dtype differs")
    if kind == "float":
        require(np.isfinite(value).all(), key + " nonfinite")
    return value


def validate_inputs(raw: dict[str, np.ndarray], c: dict[str, Any]) -> None:
    array(raw, "bank", (2001, 391, 3), "float")
    array(raw, "activation", (2001, 96), "float")
    array(raw, "original_activation", (2001, 96), "float")
    array(raw, "prior_bank", (2001, 391, 3), "float")
    array(raw, "prior_activation", (2001, 96), "float")
    array(raw, "prior_activation_residual", (2001, 96), "float")
    for key in (
        "complete",
        "bank_available",
        "activation_available",
        "original_activation_available",
        "motor_rate_available",
    ):
        require(array(raw, key, (2001,), "bool").all(), key + " incomplete")
    ticks = array(raw, "control_tick", (2001,), "int")
    require(np.array_equal(ticks, np.arange(2001)), "E167 control ticks differ")
    network_ticks = array(raw, "network_tick", (2001,), "int")
    require(np.array_equal(network_ticks, ticks * 20), "E167 network ticks differ")
    clock = array(raw, "time_s", (2001,), "float")
    require(
        np.max(np.abs(clock - ticks * c["control_dt"])) <= c["clock_atol"],
        "E167 clock differs",
    )
    indices = array(raw, "target_indices", (1824,), "int")
    require(np.array_equal(indices, np.arange(100, 1924)), "E167 target indices differ")
    array(raw, "target", (1824, 42), "float")
    array(raw, "recorded_command", (2001, 42), "float")
    moment = array(raw, "moment", (42, 96), "float")
    array(raw, "rest", (42,), "float")
    membership = array(raw, "membership", (391,), "int")
    require(
        np.array_equal(membership, array(raw, "parameter_membership", (391,), "int")),
        "E167 membership sources differ",
    )
    require(
        np.all((membership >= 0) & (membership < 96)), "E167 membership range differs"
    )
    slopes = array(raw, "slopes", (96, 391, 3, 2), "float")
    kappa = array(raw, "kappa", (391,), "float")
    require(
        np.all(raw["bank"] >= 0) and np.all(slopes >= 0) and np.all(kappa >= 0),
        "E167 negative bank/slope/kappa",
    )
    fixed = array(raw, "contract_fixed_muscles", (19,), "int")
    require(
        np.array_equal(fixed, np.asarray(c["fixed_muscles"])), "E167 fixed list differs"
    )
    support = np.any(slopes > 0, axis=(2, 3))
    expected_support = np.arange(96)[:, None] == membership[None, :]
    require(np.array_equal(support, expected_support), "E167 slope ownership differs")
    require(
        np.array_equal(raw["prior_bank"], raw["bank"]), "E167 prior/E150 bank differs"
    )
    require(
        np.max(np.abs(raw["prior_activation"] - raw["activation"]))
        <= c["reconstruction_atol"],
        "E167 prior activation differs",
    )
    require(
        np.max(np.abs(raw["prior_activation_residual"])) <= c["reconstruction_atol"],
        "E167 prior residual differs",
    )
    require(
        np.array_equal(
            raw["activation"][:, fixed], raw["original_activation"][:, fixed]
        ),
        "E167 fixed activation differs",
    )
    for muscle in range(96):
        nonzero_legs = np.flatnonzero(
            np.any(moment[:, muscle].reshape(6, 7) != 0, axis=1)
        )
        require(nonzero_legs.size <= 1, "E167 cross-leg moment differs")


def contribution_sum(
    bank: np.ndarray, slopes: np.ndarray, kappa: np.ndarray, membership: np.ndarray
) -> np.ndarray:
    low_basis = np.minimum(bank, kappa[None, :, None])
    high_basis = np.maximum(bank - kappa[None, :, None], 0.0)
    supported = slopes[membership, np.arange(bank.shape[1])]
    each = np.sum(
        supported[..., 0][None] * low_basis + supported[..., 1][None] * high_basis,
        axis=2,
    )
    total = np.zeros((bank.shape[0], slopes.shape[0]), dtype=np.float64)
    for muscle in range(slopes.shape[0]):
        total[:, muscle] = np.sum(each[:, membership == muscle], axis=1)
    return total


def envelope(raw: dict[str, np.ndarray], c: dict[str, Any]) -> dict[str, np.ndarray]:
    activation = raw["activation"]
    moment = raw["moment"]
    rest = raw["rest"]
    total = contribution_sum(
        raw["bank"], raw["slopes"], raw["kappa"], raw["membership"]
    )
    variable = np.asarray(c["variable_muscles"], dtype=np.int64)
    lower_a = activation.copy()
    lower_a[:, variable] -= c["maximum_attenuation"] * total[:, variable]
    upper_a = activation.copy()
    positive = np.maximum(moment, 0.0)
    negative = np.minimum(moment, 0.0)
    gain = c["command_gain"]
    qlo = rest + gain * (lower_a @ positive.T + upper_a @ negative.T)
    qhi = rest + gain * (upper_a @ positive.T + lower_a @ negative.T)
    qbaseline = rest + gain * (activation @ moment.T)
    eps = np.finfo(np.float64).eps
    pad = np.maximum(
        c["padding_floor_rad"],
        c["padding_epsilon_multiplier"]
        * eps
        * (
            1
            + np.abs(rest)
            + gain * ((np.abs(lower_a) + np.abs(upper_a)) @ np.abs(moment).T)
        ),
    )
    pad += gain * c["reconstruction_atol"] * np.sum(np.abs(moment), axis=1)
    expanded_low = qlo - pad
    expanded_high = qhi + pad
    indices = raw["target_indices"]
    target = raw["target"]
    distance = np.maximum(
        np.maximum(expanded_low[indices] - target, target - expanded_high[indices]), 0.0
    )
    return {
        "contribution_sum": total,
        "activation_residual": total[:, variable] - activation[:, variable],
        "activation_lower": lower_a,
        "activation_upper": upper_a,
        "q_lower_unpadded": qlo,
        "q_upper_unpadded": qhi,
        "q_lower": expanded_low,
        "q_upper": expanded_high,
        "padding": pad,
        "q_baseline": qbaseline,
        "decoder_residual": qbaseline - raw["recorded_command"],
        "distance": distance,
        "baseline_error": raw["recorded_command"][indices] - target,
    }


def validate_numerics(
    raw: dict[str, np.ndarray], result: dict[str, np.ndarray], c: dict[str, Any]
) -> None:
    atol = c["reconstruction_atol"]
    require(
        np.isfinite(result["distance"]).all() and np.isfinite(result["padding"]).all(),
        "E167 envelope nonfinite",
    )
    require(np.min(result["contribution_sum"]) >= 0, "E167 contribution negative")
    require(
        np.max(np.abs(result["activation_residual"])) <= atol,
        "E167 contribution reconstruction differs",
    )
    require(
        np.max(np.abs(result["decoder_residual"][1:])) <= atol,
        "E167 decoder reconstruction differs",
    )
    require(
        np.array_equal(
            result["activation_lower"][:, c["fixed_muscles"]],
            raw["activation"][:, c["fixed_muscles"]],
        ),
        "E167 fixed lower differs",
    )
    require(
        np.array_equal(
            result["activation_upper"][:, c["fixed_muscles"]],
            raw["activation"][:, c["fixed_muscles"]],
        ),
        "E167 fixed upper differs",
    )
    require(
        np.all(result["q_lower"] <= result["q_upper"]), "E167 envelope ordering differs"
    )
    require(
        np.all(result["q_baseline"] >= result["q_lower"])
        and np.all(result["q_baseline"] <= result["q_upper"]),
        "E167 baseline outside envelope",
    )


def summary(result: dict[str, np.ndarray], c: dict[str, Any]) -> dict[str, Any]:
    baseline = result["baseline_error"]
    floor = result["distance"]

    def metric(a: np.ndarray, d: np.ndarray) -> dict[str, float]:
        return {
            "baseline_RMS_rad": float(np.sqrt(np.mean(a * a))),
            "optimistic_RMS_floor_rad": float(np.sqrt(np.mean(d * d))),
            "outside_fraction": float(np.count_nonzero(d > 0) / d.size),
        }

    overall = metric(baseline, floor)
    threshold = (
        c["required_rms_fraction"] * overall["baseline_RMS_rad"]
        + c["decision_margin_rad"]
    )
    return {
        "schema": SCHEMA + ".summary",
        "overall": overall,
        "per_leg": {
            leg: metric(baseline[:, 7 * i : 7 * (i + 1)], floor[:, 7 * i : 7 * (i + 1)])
            for i, leg in enumerate(LEGS)
        },
        "per_joint": [
            metric(baseline[:, i : i + 1], floor[:, i : i + 1]) for i in range(42)
        ],
        "screen_threshold_rad": threshold,
        "hypothesis_verdict": "excluded_from_conditional_20pct_screen"
        if overall["optimistic_RMS_floor_rad"] > threshold
        else "not_excluded_by_optimistic_envelope",
        "interpretation": "Pointwise joint intervals are optimistic necessary conditions for this frozen E150 attenuation family and conditional waveform screen only; they are not a gait requirement or a feasible controller.",
    }


def publish(
    out: Path, result: dict[str, Any], limits: dict[str, Any], started: float
) -> dict[str, Any]:
    # This timestamp precedes publication. The supervisor records complete wall
    # time; the final internal check below includes all writes and disk checks.
    result = {**result, "elapsed_before_publication_s": time.perf_counter() - started}
    files = sorted(
        path
        for path in out.rglob("*")
        if path.is_file() and path.name not in ("result.json", "manifest.json")
    )
    manifest = {path.relative_to(out).as_posix(): sha256(path) for path in files}
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "result.json", result)
    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    require(total <= limits["raw_bytes"], "E167 raw byte limit exceeded")
    require(
        shutil.disk_usage(out).free >= limits["free_bytes"],
        "E167 free disk limit exceeded",
    )
    require(
        time.perf_counter() - started <= limits["internal_s"],
        "E167 internal time exceeded",
    )
    return result


def run(
    freeze_path: Path,
    freeze_sha: str,
    gate_path: Path,
    gate_sha: str,
    out: Path,
    *,
    input_loader: Callable[
        [dict[str, Any]], dict[str, dict[str, np.ndarray]]
    ] = load_inputs,
) -> dict[str, Any]:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    freeze: dict[str, Any] | None = None
    hashes: dict[str, str] = {}
    approved: dict[str, str] | None = None
    error: BaseException | None = None
    secondary: list[str] = []
    stage = "preflight"
    try:
        with threadpool_limits(limits=1):
            require(
                all(item.get("num_threads") == 1 for item in threadpool_info()),
                "E167 CPU thread limit differs",
            )
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            c = validate_freeze(freeze, freeze_path, freeze_sha)
            hashes = frozen_hashes(freeze)
            verify_hashes(hashes)
            gate = validate_gate(
                gate_path, gate_sha, freeze_sha, freeze["design_sha256"]
            )
            approved = snapshot(out, freeze, freeze_path, gate_path, gate_sha, gate)
            stage = "input"
            data = input_loader(freeze)
            raw = raw_inputs(data, c)
            atomic_npz(out / "input-raw.npz", raw)
            validate_inputs(raw, c)
            stage = "envelope"
            calculated = envelope(raw, c)
            atomic_npz(out / "envelope-raw.npz", calculated)
            validate_numerics(raw, calculated, c)
            report = summary(calculated, c)
            atomic_json(out / "summary.json", report)
            verify_hashes(hashes)
            verify_snapshot(out, freeze, freeze_path, gate_path, approved)
            stage = "publication"
            return publish(
                out,
                {
                    "schema": SCHEMA + ".result",
                    "validity": "valid_diagnostic",
                    "hypothesis_verdict": report["hypothesis_verdict"],
                    "freeze_sha256": freeze_sha,
                    "source_review_sha256": gate_sha,
                    "original_processes": 1,
                    "automatic_retries": 0,
                    "candidate_adopted": False,
                    "whole_goal_complete": False,
                    "F6_exposed": False,
                    "counts": freeze["counts"],
                },
                freeze["resource_limits"],
                started,
            )
    except BaseException as exc:  # noqa: BLE001 - preserve first failure, including interrupts
        error = exc
    for name in ("result.json", "manifest.json"):
        (out / name).unlink(missing_ok=True)
    if hashes:
        try:
            verify_hashes(hashes)
        except BaseException as exc:  # noqa: BLE001 - secondary integrity failure
            secondary.append("post_hashes: " + repr(exc))
    if approved is not None and freeze is not None:
        try:
            verify_snapshot(out, freeze, freeze_path, gate_path, approved)
        except BaseException as exc:  # noqa: BLE001 - secondary snapshot failure
            secondary.append("post_snapshot: " + repr(exc))
    try:
        atomic_json(
            out / "failure.json",
            {
                "schema": SCHEMA + ".failure",
                "stage": stage,
                "primary_error": {"type": type(error).__name__, "message": str(error)},
                "secondary_errors": secondary,
                "elapsed_s": time.perf_counter() - started,
            },
        )
    except BaseException as exc:  # noqa: BLE001 - retain first error if disk is unavailable
        secondary.append("failure_persistence: " + repr(exc))
    try:
        (out / "traceback.txt").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
    except BaseException as exc:  # noqa: BLE001 - retain first error if disk is unavailable
        secondary.append("traceback_persistence: " + repr(exc))
    assert error is not None
    for note in secondary:
        error.add_note(note)
    raise error.with_traceback(error.__traceback__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
