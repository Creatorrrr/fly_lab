"""E336 fixed saved-only NumPy/geometry diagnostics; no runtime controller."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e336-saved-gait-transfer"
N, MOTORS, CONTROLS = 158706, 391, 300
INC = 2.382760007990865 * 0.005
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
S = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
OWNED = (
    "tools/analyze_c_banc_m3_saved_gait_transfer.py",
    "tests/test_c_banc_saved_gait_transfer.py",
)
SCHEMA = "flylab.m3.e336.saved-gait-transfer.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e336.saved-gait-transfer.review.v1"
QUAL_SCHEMA = "flylab.m3.e336.generated-qualification.v1"
RESULT_SCHEMA = "flylab.m3.e336.saved-gait-transfer.result.v1"
LIMITS = {
    "supervisor_wall_s": 40,
    "internal_wall_s": 35,
    "science_wall_s": 30,
    "finalization_reserve_s": 5,
    "host_bytes": 2 << 30,
    "output_bytes": 64 << 20,
    "minimum_free_disk_bytes": 1 << 30,
    "cpu_threads": 1,
    "retry_count": 0,
}
PATHS = {
    "prior_freeze": BASE + "/e335-smooth-secant-body-freeze.json",
    "prior_decision": BASE + "/parent-e335-final-decision.json",
    "prior_external": BASE + "/parent-e335-external-completion.json",
    "prior_audit": BASE + "/parent-e335-original-raw-audit-v3.json",
    "prior_audit_source": BASE + "/parent-e335-original-raw-audit-v3.py",
    "prior_audit_contract": BASE + "/parent-e335-audit-v3-contract.json",
    "prior_audit_supervision": BASE + "/parent-e335-audit-v3-supervision.json",
    "prior_contract": BASE + "/parent-e335-original-contract.json",
    "prior_supervision": BASE + "/parent-e335-original-supervision.json",
    "prior_review": BASE + "/parent-e335-original-source-review.json",
    "prior_accounting": BASE + "/parent-e335-original-accounting.json",
    "e333_freeze": BASE + "/e333-smooth-efficacy-direction-freeze.json",
    "e334_decision": BASE + "/parent-e334-final-decision.json",
    "e334_audit": BASE + "/parent-e334-original-raw-audit-v2.json",
    "e334_audit_source": BASE + "/parent-e334-original-raw-audit.py",
    "e334_audit_contract": BASE + "/parent-e334-audit-v2-contract.json",
    "e334_audit_supervision": BASE + "/parent-e334-audit-v2-supervision.json",
    "e334_contract": BASE + "/parent-e334-original-contract.json",
    "e334_supervision": BASE + "/parent-e334-original-supervision.json",
    "e334_freeze": BASE + "/e334-smooth-constrained-secant-freeze.json",
    "e334_review": BASE + "/parent-e334-original-source-review.json",
    "reader": BASE + "/parent-e335-audit-readers.py",
    "pure_gait": BASE + "/parent-e327-saved-metrics-v2.py",
    "design": "docs/BANC_M3_E336_SAVED_GAIT_TRANSFER_DESIGN_20261001.md",
    "astra": BASE + "/parent-e336-astra-design-review.md",
    "reflection": BASE + "/parent-e336-astra-reflection.json",
    "invocation": BASE + "/parent-e336-astra-invocation.json",
    "inventory": BASE + "/parent-e336-input-inventory.json",
    "inventory_source": BASE + "/parent-e336-inventory-v2.py",
    "inventory_contract": BASE + "/parent-e336-inventory-v2-contract.json",
    "inventory_supervision": BASE + "/parent-e336-inventory-v2-supervision.json",
    "workload": BASE + "/parent-e336-workload.json",
    "workload_source": BASE + "/parent-e336-workload.py",
    "workload_math": BASE + "/parent-e336-independent-values.py",
    "workload_contract": BASE + "/parent-e336-workload-contract.json",
    "workload_supervision": BASE + "/parent-e336-workload-supervision.json",
    "qualification": BASE + "/parent-e336-generated-qualification.json",
}
FIXED = {
    "design": "dfb957c6cc745672baa7ca76bd50c8596c44b64720bdcb8ece991aa9063d44af",
    "reflection": "63265e676395a39b9a500a009f3365749321925fb09c84379dbfe3b63015719b",
    "astra": "315e0ce730ed7d140ad4c2fae15f083f62185a19499d3ddf7e400a5a075a4924",
    "reader": "35ea9ac5156eff0eb195fd28d1a92938b6574879e81c373d80a211cfc7fc9201",
    "pure_gait": "b75f80920ba59886b93cf24fc157a3c483c51a92693fa43a3c33ff3055442117",
    "inventory": "019449e1d0741d05db15af0b867855b28e672c64759ac850bd55ee791a43ce61",
    "workload": "3e06e4c9707b1041996acc900ce8984c3c56fe98a3ecf1ba3f246ddf988ff10f",
    "prior_decision": "5c683044b19a87757255c4319a391a51671dab48159e13bf2e918f379e16688a",
    "e334_decision": "6838c9282c2c6fd919567dd65464e267a48057bad42862902cf0d9d56e226d2a",
}


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(Path(path).read_text("utf8"), parse_constant=reject)


def put_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", "utf8")


def sha256(path, timely=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            timely()
            h.update(block)
    timely()
    return h.hexdigest()


def bytes_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metadata(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "c_order_sha256": bytes_sha(array),
    }


def safe(name, root=ROOT):
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError("path string required")
    windows = PureWindowsPath(name)
    if windows.drive and not windows.is_absolute():
        raise ValueError("drive-relative path forbidden")
    path = Path(name.replace("\\", "/"))
    path = (path if path.is_absolute() else Path(root) / path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes repository")
    return path


def merge_pins(target, added, *, root=ROOT):
    for name, value in added.items():
        key = safe(name, root).relative_to(Path(root).resolve()).as_posix()
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("SHA256 pin required")
        if key in target and target[key] != value:
            raise ValueError("producer pin collision: " + key)
        target[key] = value


def exact(a, b, label="bytes"):
    if (
        a.dtype != b.dtype
        or a.shape != b.shape
        or a.tobytes(order="C") != b.tobytes(order="C")
    ):
        raise ValueError("exact correspondence differs: " + label)


def close(a, b, label="value", *, atol=1e-10, rtol=1e-10):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError("numeric correspondence differs: " + label)


def typed(a, shape, dtype, label, *, finite=True):
    if (
        not isinstance(a, np.ndarray)
        or a.shape != tuple(shape)
        or a.dtype != np.dtype(dtype)
        or a.dtype.hasobject
    ):
        raise ValueError("typed shape differs: " + label)
    if finite and not np.isfinite(a).all():
        raise ValueError("nonfinite input: " + label)
    return a


def npz_headers(path):
    """ZIP/NPY metadata only; never interpret numerical payload values."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if (
                not info.filename.endswith(".npy")
                or "/" in info.filename
                or info.filename in headers
            ):
                raise ValueError("unsafe or duplicate NPZ member")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("NPY header version unsupported")
                header_bytes = stream.tell()
            if dtype.hasobject:
                raise ValueError("object NPZ member forbidden")
            key = info.filename[:-4]
            if key in headers:
                raise ValueError("duplicate NPZ key")
            headers[key] = {
                "shape": list(shape),
                "dtype": dtype.str,
                "fortran": fortran,
                "compression": info.compress_type,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "npy_version": list(version),
                "header_bytes": header_bytes,
                "crc32": info.CRC,
            }
    return headers


def dtype_metadata(value):
    """Normalize only a scalar dtype descriptor, preserving its byte order."""
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty dtype metadata string required")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid dtype metadata") from error
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("scalar nonobject dtype metadata required")
    return dtype.str


def load_selected(path, schema, *, reads=None, timely=lambda: None):
    """Decode only enumerated typed keys after the caller's passive gate."""
    headers = npz_headers(path)
    for key, spec in schema.items():
        if (
            key not in headers
            or headers[key]["shape"] != spec["shape"]
            or headers[key]["dtype"] != dtype_metadata(spec["dtype"])
            or (
                "fortran_order" in spec
                and (
                    type(spec["fortran_order"]) is not bool
                    or headers[key]["fortran"] != spec["fortran_order"]
                )
            )
        ):
            raise ValueError("selected input header differs: " + key)
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        for key, spec in schema.items():
            timely()
            values[key] = typed(
                archive[key],
                spec["shape"],
                spec["dtype"],
                key,
                finite=spec.get("finite", True),
            )
            if reads is not None:
                reads.append(dict(path=str(path), key=key, **metadata(values[key])))
    timely()
    return values


def canonical_constants(curves):
    typed(curves, (6, 128, 7), "<f8", "curves")
    coordinate = np.remainder(np.arange(128, dtype=np.float64) / 128 + INC, 1.0) * 128
    floor = np.floor(coordinate)
    index = floor.astype(np.int64) % 128
    future = (
        curves[:, index]
        + (coordinate - floor)[None, :, None]
        * (np.roll(curves, -1, axis=1) - curves)[:, index]
    )
    n = np.mean((future - curves) ** 2, axis=(1, 2))
    angle = 2 * np.pi * (np.arange(128, dtype=np.float64) / 128)
    return {
        "curves": curves,
        "normalization": n,
        "temperature": 7 * n,
        "future_anchors": future,
        "phase_basis": np.stack((np.cos(angle), np.sin(angle)), axis=-1),
    }


class ValueEncoding:
    """Full128 value-only soft encoding with approved stored loss constants."""

    def __init__(self, constants):
        recomputed = canonical_constants(constants["curves"])
        shapes = {
            "curves": (6, 128, 7),
            "normalization": (6,),
            "temperature": (6,),
            "future_anchors": (6, 128, 7),
            "phase_basis": (128, 2),
        }
        self.constants = {}
        for key, shape in shapes.items():
            a = typed(constants[key], shape, "<f8", key)
            close(a, recomputed[key], key, rtol=1e-12, atol=1e-15)
            self.constants[key] = a.copy(order="C")
        if np.any(self.constants["normalization"] <= 0) or np.any(
            self.constants["temperature"] <= 0
        ):
            raise ValueError("positive reference step normalization required")

    def encode(self, q):
        typed(q, (len(q), 42), "<f8", "q")
        if not len(q):
            raise ValueError("nonempty q required")
        c = self.constants
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residual = q.reshape(-1, 6, 1, 7) - c["curves"][None]
            distance = np.einsum("tlsj,tlsj->tls", residual, residual)
            logits = -distance / c["temperature"][None, :, None]
            unnormalized = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            weights = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
            target = np.einsum("tls,lsj->tlj", weights, c["future_anchors"]).reshape(
                -1, 42
            )
            z = np.einsum("tls,sk->tlk", weights, c["phase_basis"])
            logs = np.zeros_like(weights)
            np.log(weights, out=logs, where=weights > 0)
            entropy = -(weights * logs).sum(axis=-1)
        return {
            "weights": weights,
            "target": target,
            "z": z,
            "R": np.linalg.norm(z, axis=-1),
            "entropy": entropy,
            "minimum_anchor_distance_rad": np.sqrt(distance.min(axis=-1) / 7),
        }


def phase_segments(z, R, *, controls=None):
    typed(z, (len(z), 6, 2), "<f8", "z")
    typed(R, (len(z), 6), "<f8", "R")
    controls = (
        np.arange(100, 100 + len(z), dtype=np.int64) if controls is None else controls
    )
    typed(controls, (len(z),), "<i8", "phase controls")
    if np.any(np.diff(controls) != 1):
        raise ValueError("contiguous phase controls required")
    cross = z[:-1, :, 0] * z[1:, :, 1] - z[:-1, :, 1] * z[1:, :, 0]
    dot = np.einsum("tlk,tlk->tl", z[:-1], z[1:])
    delta = np.arctan2(cross, dot) / (2 * np.pi)
    valid = (R[:-1] >= 0.5) & (R[1:] >= 0.5) & (np.abs(delta) != 0.5)
    saved = np.where(valid, delta, np.nan)
    legs = {}
    for leg, name in enumerate(LEGS):
        edges = np.diff(np.r_[False, valid[:, leg], False].astype(np.int8))
        segments = []
        for start, end in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        ):
            segments.append(
                {
                    "first_control": int(controls[start]),
                    "last_control": int(controls[end]),
                    "edge_count": int(end - start),
                    "elapsed_s": float((end - start) * 0.005),
                    "expected_progress": float((end - start) * INC),
                    "signed_progress": float(delta[start:end, leg].sum()),
                }
            )
        values = delta[valid[:, leg], leg]
        legs[name] = {
            "valid_edge_count": len(values),
            "unknown_edge_count": int((~valid[:, leg]).sum()),
            "reverse_count": int(np.count_nonzero(values < 0)),
            "delta_p05_median_p95": None
            if not len(values)
            else np.percentile(values, [5, 50, 95]).tolist(),
            "segments": segments,
            "signed_progress_sum_within_segments": float(values.sum()),
        }
    return {
        "phase": np.arctan2(z[:, :, 1], z[:, :, 0]) / (2 * np.pi),
        "delta": saved,
        "valid": valid,
    }, legs


def rms_leg(a):
    with np.errstate(over="raise", invalid="raise"):
        return np.sqrt(np.mean(a.reshape(-1, 6, 7) ** 2, axis=(0, 2)))


def ratios(numerator, denominator):
    return [
        None if b == 0 else float(a / b)
        for a, b in zip(np.ravel(numerator), np.ravel(denominator), strict=True)
    ]


def smooth_diagnostics(constants, q):
    typed(q, (201, 42), "<f8", "smooth c100..300")
    encoder = ValueEncoding(constants)
    encoded = encoder.encode(q)
    residual = (q[1:] - encoded["target"][:-1]).reshape(200, 6, 7)
    n = encoder.constants["normalization"]
    per_leg = np.mean(residual**2, axis=(0, 2)) / n
    z = encoded["z"][1:]
    pairs = np.array(
        [np.mean(1 - np.einsum("tk,tk->t", z[:, a], z[:, b])) for a, b in PAIRS]
    )
    phases, segments = phase_segments(encoded["z"], encoded["R"])
    static = q - encoded["target"]
    step = np.diff(q, axis=0)
    arrays = dict(
        q=q,
        **encoded,
        **{"phase_" + k: v for k, v in phases.items()},
        successor_residual=residual,
        static_residual=static,
        command_increment=step,
        per_leg_successor_loss=per_leg,
        coordination_pair_loss=pairs,
    )
    metrics = {
        "training_loss": float(per_leg.mean() + pairs.mean()),
        "successor_loss": float(per_leg.mean()),
        "coordination_loss": float(pairs.mean()),
        "lr_coordination_loss": float(pairs[[2, 7, 11]].mean()),
        "static_normalized_rms": (rms_leg(static) / np.sqrt(n)).tolist(),
        "increment_rms_rad": rms_leg(step).tolist(),
        "increment_over_reference_step": (rms_leg(step) / np.sqrt(n)).tolist(),
        "reference_step_scale_rad": np.sqrt(n).tolist(),
        "phases": segments,
        "controls": [100, 300],
        "boundaries": 201,
        "edges": 200,
    }
    return arrays, metrics


def tracking_statistics(command, actual, stored_error, normalization, curves):
    for name, a in (
        ("command", command),
        ("actual", actual),
        ("stored_error", stored_error),
    ):
        typed(a, (200, 42), "<f8", name)
    tracking = actual - command
    close(tracking, -stored_error, "target-minus-actual sign")
    span = lambda a: np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0)
    cspan, aspan, reference = (
        span(command),
        span(actual),
        np.percentile(curves, 95, axis=1) - np.percentile(curves, 5, axis=1),
    )
    ci, ai = np.diff(command, axis=0), np.diff(actual, axis=0)
    arrays = {
        "tracking": tracking,
        "command_increment": ci,
        "actual_increment": ai,
        "command_span": cspan,
        "actual_span": aspan,
        "reference_span": reference.reshape(42),
    }
    metrics = {
        "controls": [100, 299],
        "boundaries": 200,
        "edges": 199,
        "tracking_rms_rad": rms_leg(tracking).tolist(),
        "tracking_max_rad": np.max(
            np.abs(tracking.reshape(200, 6, 7)), axis=(0, 2)
        ).tolist(),
        "tracking_normalized_rms": (
            rms_leg(tracking) / np.sqrt(normalization)
        ).tolist(),
        "command_increment_rms_rad": rms_leg(ci).tolist(),
        "actual_increment_rms_rad": rms_leg(ai).tolist(),
        "command_increment_over_reference_step": (
            rms_leg(ci) / np.sqrt(normalization)
        ).tolist(),
        "actual_increment_over_reference_step": (
            rms_leg(ai) / np.sqrt(normalization)
        ).tolist(),
        "actual_span_over_command_span": ratios(aspan, cspan),
        "command_span_over_reference_span": ratios(cspan, reference),
        "actual_span_over_reference_span": ratios(aspan, reference),
    }
    return arrays, metrics


def difference_statistics(left, right):
    if (
        left.dtype != right.dtype
        or left.shape != right.shape
        or left.ndim != 2
        or left.dtype not in (np.float32, np.float64)
    ):
        raise ValueError("same typed difference matrix required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite difference inputs required")
    with np.errstate(over="raise", invalid="raise"):
        delta = left.astype(np.float64) - right.astype(np.float64)
        square = delta * delta
        result = {
            "changed_count": np.count_nonzero(left != right, axis=1).astype(np.int64),
            "rms": np.sqrt(square.mean(axis=1)),
            "l2": np.sqrt(square.sum(axis=1)),
            "maximum": np.abs(delta).max(axis=1),
        }
    return result


def native_statistics(native):
    t, load, slip, valid = (
        native[k] for k in ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid")
    )
    if (
        t.shape != (15000,)
        or load.shape != (15000, 6)
        or slip.shape != load.shape
        or valid.shape != load.shape
        or valid.dtype != np.bool_
    ):
        raise ValueError("native schema differs")
    if not np.isfinite(t).all() or not np.isfinite(load).all():
        raise ValueError("finite native time/load required")
    close(t, np.arange(1, 15001) * 0.0001, "native clock", rtol=0, atol=2e-8)
    window = (t >= 0.5) & (t < 1.5)
    support = window[:, None] & (load >= 0.02)
    if not np.all(valid[support]) or not np.isfinite(slip[support]).all():
        raise ValueError("every supported native slip must be valid/finite")
    legs = {}
    for leg, name in enumerate(LEGS):
        own = support[:, leg]
        den = float(load[own, leg].sum())
        legs[name] = {
            "support_samples": int(own.sum()),
            "load_denominator_bw": den,
            "weighted_slip_mm_s": None
            if den == 0
            else float((load[own, leg] * slip[own, leg]).sum() / den),
            "support_fraction": float(np.mean(own[window])),
            "contact_transitions": int(
                np.count_nonzero(np.diff(own[window].astype(np.int8)))
            ),
        }
    den = float(load[support].sum())
    nominal = (np.arange(1, 15001) * 0.0001 >= 0.5) & (
        np.arange(1, 15001) * 0.0001 < 1.5
    )
    ticks = np.flatnonzero(window) + 1
    return {
        "weighted_slip_mm_s": None
        if den == 0
        else float((load[support] * slip[support]).sum() / den),
        "load_denominator_bw": den,
        "per_leg": legs,
        "actual_mask_count": len(ticks),
        "first_native_tick": None if not len(ticks) else int(ticks[0]),
        "last_native_tick": None if not len(ticks) else int(ticks[-1]),
        "nominal_mask_count": int(nominal.sum()),
        "nominal_mask_difference_ticks": (
            np.flatnonzero(window != nominal) + 1
        ).tolist(),
    }


def verify_tonic(encoded, actual, config, profile):
    if (
        encoded.dtype != np.float32
        or actual.dtype != np.float32
        or encoded.shape != actual.shape
    ):
        raise ValueError("f32 live tonic inputs required")
    descending = np.asarray(config["descending_ids"], dtype=np.int64)
    if sorted(descending.tolist()) != sorted(profile["bilateral_descending"]):
        raise ValueError("tonic descending correspondence differs")
    expected = encoded.copy()
    expected[:, descending] += np.float32(config["descending_drive"])
    if config["navigation"]:
        tonic = np.zeros(encoded.shape[1], np.float32)
        tonic[profile["ports"]] = np.float32(profile["port_background"])
        tonic[profile["bilateral_descending"]] = np.float32(profile["forward_drive"])
        expected += tonic[None]
    exact(expected, actual, "ordered in-place f32 tonic")
    return {
        "verified": True,
        "rounding_order": "encoded f32 copy; descending indexed f32 addition; tonic f32 vector in-place addition",
    }


def verify_execution(report, descriptor, contract, supervision, *, root=ROOT):
    """Bind report/helper bytes to its sole completed exact Python command."""
    commands, attempts = contract.get("commands", []), supervision.get("attempts", [])
    if (
        supervision.get("schema") != "flylab.m3.e267.root-supervision.v1"
        or supervision.get("success") is not True
        or len(commands) != 1
        or len(attempts) != 1
        or attempts[0].get("exit_code") != 0
        or attempts[0].get("killed") is not False
        or not supervision.get("owned_pids")
    ):
        raise ValueError("sole successful completed producer required")
    argv = commands[0].get("argv")
    if (
        not isinstance(argv, list)
        or not all(isinstance(v, str) for v in argv)
        or argv != attempts[0].get("argv")
    ):
        raise ValueError("literal producer argv mismatch")
    helper = safe(report["audit_source_path"], root)
    pins = {}
    merge_pins(pins, contract["source_sha256"], root=root)
    helper_name = helper.relative_to(Path(root).resolve()).as_posix()
    if (
        pins.get(helper_name) != report["audit_source_sha256"]
        or sha256(helper) != report["audit_source_sha256"]
    ):
        raise ValueError("executed report helper source absent/mismatched")
    if (
        len(argv) < 7
        or argv[1:4] != ["-X", "utf8", "-B"]
        or safe(argv[0], root) != safe(".venv/Scripts/python.exe", root)
        or safe(argv[4], root) != helper
        or helper.suffix != ".py"
    ):
        raise ValueError("exact Python helper command required")
    outputs = []
    for i, arg in enumerate(argv[5:], 5):
        if arg == "--":
            raise ValueError("producer option terminator forbidden")
        if arg == "--output":
            if i + 1 == len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("missing report output")
            outputs.append(argv[i + 1])
        elif arg.startswith("--output="):
            outputs.append(arg.split("=", 1)[1])
    if len(outputs) != 1 or safe(outputs[0], root) != safe(descriptor["path"], root):
        raise ValueError("executed report output path differs")


def _pin_map(pins, root, timely):
    for name, digest in pins.items():
        if sha256(safe(name, root), timely) != digest:
            raise ValueError("frozen byte mismatch: " + name)


def verify_producer_closure(contract, freeze, *, root=ROOT):
    combined = freeze["source_sha256"].copy()
    merge_pins(combined, freeze["input_sha256"], root=root)
    executed = {}
    merge_pins(executed, contract["source_sha256"], root=root)
    if any(combined.get(name) != pin for name, pin in executed.items()):
        raise ValueError("executed helper/imported source omitted from frozen closure")
    for command in contract["commands"]:
        for key in ("stdout", "stderr"):
            name = safe(command[key], root).relative_to(Path(root).resolve()).as_posix()
            if combined.get(name) != sha256(safe(name, root)):
                raise ValueError("completed producer stdout/stderr omitted or changed")


def _inherited(root):
    prior = read(safe(PATHS["prior_freeze"], root))
    if len(prior["source_sha256"]) != 203 or len(prior["input_sha256"]) != 2418:
        raise ValueError("exact E335203source/2418input predecessor required")
    sources, inputs = {}, {}
    merge_pins(sources, prior["source_sha256"], root=root)
    merge_pins(inputs, prior["input_sha256"], root=root)
    decision = read(safe(PATHS["prior_decision"], root))
    merge_pins(inputs, decision["evidence_sha256"], root=root)
    merge_pins(
        inputs,
        {PATHS["prior_decision"]: sha256(safe(PATHS["prior_decision"], root))},
        root=root,
    )
    for folder in ("e318-full-drive-capture", "e335-smooth-secant-body"):
        prefix = BASE + "/" + folder
        manifest = read(safe(prefix + "/output-manifest.json", root))
        for name, desc in manifest["files"].items():
            if name in {
                "control-raw.npz",
                "native-contact.npz",
                "trace.json",
                "live-input-events.json",
            } or name.startswith(
                ("observed-start/", "full-drive-chunks/full-drive-", "live-input-")
            ):
                merge_pins(inputs, {prefix + "/" + name: desc["sha256"]}, root=root)
    return sources, inputs


def verify_inventory(inventory, sources, inputs, *, root=ROOT):
    if (
        inventory.get("schema") != "flylab.m3.e336.root-input-inventory.v1"
        or inventory.get("passed") is not True
        or inventory.get("array_payload_numeric_reads") != 0
    ):
        raise ValueError("passed metadata-only inventory required")
    for field, expected in (
        ("inherited_source_sha256", sources),
        ("inherited_input_sha256", inputs),
    ):
        normalized = {}
        merge_pins(normalized, inventory[field], root=root)
        if normalized != expected:
            raise ValueError("exact inventory inherited roster differs")
    combined = sources.copy()
    merge_pins(combined, inputs, root=root)
    if set(inventory["files"]) != set(combined):
        raise ValueError("inventory exact file roster required")
    for name, digest in combined.items():
        if inventory["files"][name] != {
            "sha256": digest,
            "bytes": safe(name, root).stat().st_size,
        }:
            raise ValueError("inventory file bytes/SHA differ")
    for protected in inventory["protected_F6"]["payload_paths"]:
        path = safe(protected, root)
        if any(
            safe(name, root) == path or safe(name, root).is_relative_to(path)
            for name in combined
        ):
            raise ValueError("required closure conflicts with protected F6 payload")
    for name, archive in inventory["archives"].items():
        if (
            name not in combined
            or archive["sha256"] != combined[name]
            or archive["bytes"] != safe(name, root).stat().st_size
        ):
            raise ValueError("inventory archive producer differs")
        headers = npz_headers(safe(name, root))
        for key, spec in archive["members"].items():
            h = headers[key]
            for a, b in (
                ("dtype", "dtype"),
                ("shape", "shape"),
                ("fortran", "fortran_order"),
                ("compression", "zip_compression"),
                ("compressed_bytes", "compressed_bytes"),
                ("uncompressed_bytes", "uncompressed_bytes"),
                ("npy_version", "npy_version"),
                ("header_bytes", "header_bytes"),
                ("crc32", "crc32"),
            ):
                expected = dtype_metadata(spec[b]) if a == "dtype" else spec[b]
                if (a == "fortran" and type(expected) is not bool) or h[a] != expected:
                    raise ValueError("inventory NPY/ZIP metadata differs")


def source_correspondence(sources, *, root=ROOT):
    """Concrete pinned source expressions establish recorded axes and order."""
    required = {
        "flylab/c/walking_trace.py": (
            "body.d.qpos[body.qpos_ids]",
            "targets - actual",
            "S @ rotation",
        ),
        "flylab/body.py": (
            "self.d.qpos[self.qpos_ids].copy()",
            "self.d.qvel[self.qvel_ids].copy()",
        ),
        "flylab/c/banc_walking.py": (
            "self.adapter.encode(sensory_cut=self.cuts['sensory'])",
            "self.network.advance_readout(drive, self.adapter.motor_ids, self.neural_steps_per_control)",
        ),
        "flylab/common.py": ("S @ np.asarray(v, dtype=float)",),
    }
    proof = {}
    for name, expressions in required.items():
        if name not in sources:
            raise ValueError("ordering source missing from inherited producer")
        tree = ast.dump(ast.parse(safe(name, root).read_text("utf8")))
        for expression in expressions:
            if ast.dump(ast.parse(expression, mode="eval").body) not in tree:
                raise ValueError("pinned source axis/call correspondence differs")
        proof[name] = {"sha256": sources[name], "exact_expressions": list(expressions)}
    return proof


def _completed(contract, supervision, digest, *, original_calls):
    if (
        supervision.get("success") is not True
        or supervision.get("schema") != "flylab.m3.e267.root-supervision.v1"
        or supervision.get("contract_sha256") != digest
        or supervision.get("original_calls") != original_calls
        or contract.get("original_calls") != original_calls
        or len(contract.get("commands", [])) != 1
        or len(supervision.get("attempts", [])) != 1
        or not supervision.get("owned_pids")
    ):
        raise ValueError("completed historical producer contract required")
    attempt = supervision["attempts"][0]
    if (
        attempt.get("exit_code") != 0
        or attempt.get("killed") is not False
        or attempt.get("argv") != contract["commands"][0]["argv"]
    ):
        raise ValueError("historical producer completed argv differs")


def verify_previous(inputs, *, root=ROOT):
    """Exact E334/E335 completed audits, terminal SHA and historical verdicts."""
    held_decision = read(safe(PATHS["e334_decision"], root))
    if (
        held_decision.get("schema") != "flylab.m3.e334.root-final-decision.v1"
        or held_decision.get("run_valid") is not True
        or held_decision.get("formal_verdict") != "conditional_neural_supported"
        or held_decision.get("root_independent_raw_qualification") is not True
        or held_decision.get("raw_checks") != 5094
        or held_decision.get("body_calls") != 0
        or held_decision.get("cost_charged_once") is not True
    ):
        raise ValueError("completed conditional E3345094 saved producer required")
    for name, pin in held_decision["evidence_sha256"].items():
        key = safe(name, root).relative_to(Path(root).resolve()).as_posix()
        if inputs.get(key) != pin:
            raise ValueError("E334 final decision producer evidence differs")
    decision = read(safe(PATHS["prior_decision"], root))
    marker = read(safe(PATHS["prior_external"], root))
    if (
        decision.get("schema") != "flylab.m3.e335.root-final-decision.v1"
        or decision.get("run_valid") is not True
        or decision.get("formal_verdict") != "rejected"
        or decision.get("root_independent_raw_qualification") is not True
        or decision.get("raw_checks") != 7082
        or decision.get("cost_charged_once") is not True
        or decision.get("original_reserved_s") != 0
        or decision.get("original_cumulative_s") != 9493.171036399524
        or marker.get("schema") != "flylab.m3.e335.root-external-completion.v1"
        or marker.get("original_completed") is not True
        or marker.get("audit_passed") is not True
        or marker.get("raw_checks") != 7082
        or marker.get("original_cost_already_charged_once") is not True
        or marker.get("formal_verdict") != "rejected"
    ):
        raise ValueError(
            "completed rejected E335 and once-charged external proof required"
        )
    for group in (
        "original_owned_ended_immediate_snapshot",
        "audit_owned_ended_immediate_snapshot",
    ):
        if not marker.get(group) or not all(
            value is True for value in marker[group].values()
        ):
            raise ValueError("historical owned-ended marker required")
    for name, pin in marker["evidence_sha256"].items():
        key = safe(name, root).relative_to(Path(root).resolve()).as_posix()
        if inputs.get(key) != pin:
            raise ValueError("external completion producer evidence differs")
    for label, prefix, count in (
        ("prior", "e335-smooth-secant-body", 7082),
        ("e334", "e334-smooth-constrained-secant", 5094),
    ):
        audit = read(safe(PATHS[label + "_audit"], root))
        checks = audit.get("checks", {})
        expected_schema = (
            "flylab.m3."
            + ("e335" if label == "prior" else "e334")
            + ".root-original-raw-audit.v1"
        )
        if (
            audit.get("schema") != expected_schema
            or audit.get("passed") is not True
            or len(checks) != count
            or not all(v is True for v in checks.values())
        ):
            raise ValueError("exact passed saved audit required")
        for field, alias in (
            ("audit_source_sha256", "_audit_source"),
            ("freeze_sha256", "_freeze"),
            ("review_sha256", "_review"),
            ("supervision_sha256", "_supervision"),
        ):
            if audit.get(field) != inputs.get(PATHS[label + alias]):
                raise ValueError("saved audit producer link differs")
        for terminal, pin in audit["terminal_sha256"].items():
            name = BASE + "/" + prefix + "/" + terminal
            if inputs.get(name) != pin or sha256(safe(name, root)) != pin:
                raise ValueError("completed audit terminal bytes differ")
        for role, calls in (("", 1), ("_audit", 0)):
            cp = PATHS[label + role + "_contract"]
            sp = PATHS[label + role + "_supervision"]
            _completed(
                read(safe(cp, root)),
                read(safe(sp, root)),
                inputs[cp],
                original_calls=calls,
            )
        result = read(safe(BASE + "/" + prefix + "/result.json", root))
        if (
            result.get("run_valid") is not True
            or result.get("first_error") is not None
            or result.get("errors") != []
        ):
            raise ValueError("current valid terminal result required")
    audit = read(safe(PATHS["prior_audit"], root))
    if any(
        audit["checks"].get(key) is not True
        for key in (
            "once_original_full_CSR_bits",
            "initial_actual_native_model_identity",
        )
    ):
        raise ValueError("actual original CSR topology/native identity proof required")
    return decision


def make_freeze(*, root=ROOT):
    sources, inputs = _inherited(root)
    inventory = read(safe(PATHS["inventory"], root))
    verify_inventory(inventory, sources, inputs, root=root)
    inherited_sources, inherited_inputs = sources.copy(), inputs.copy()
    merge_pins(sources, {name: sha256(safe(name, root)) for name in OWNED}, root=root)
    merge_pins(
        inputs, {name: sha256(safe(name, root)) for name in PATHS.values()}, root=root
    )
    qualification = read(safe(PATHS["qualification"], root))
    merge_pins(inputs, qualification["evidence_sha256"], root=root)
    for alias in ("inventory_contract", "workload_contract"):
        contract = read(safe(PATHS[alias], root))
        merge_pins(inputs, contract["source_sha256"], root=root)
        for command in contract["commands"]:
            merge_pins(
                inputs,
                {
                    command[key]: sha256(safe(command[key], root))
                    for key in ("stdout", "stderr")
                },
                root=root,
            )
    invocation = read(safe(PATHS["invocation"], root))
    merge_pins(
        inputs,
        {BASE + "/" + name: pin for name, pin in invocation["files_sha256"].items()},
        root=root,
    )
    old_paths = read(safe(PATHS["e333_freeze"], root))["paths"]
    reference = safe(old_paths["e332_result"], root).parent / "reference.npz"
    reference_name = reference.relative_to(Path(root).resolve()).as_posix()
    if reference_name not in inputs:
        raise ValueError("canonical reference absent from inherited producer")
    return {
        "schema": SCHEMA,
        "paths": PATHS.copy(),
        "limits": LIMITS.copy(),
        "output": OUTPUT,
        "source_sha256": sources,
        "input_sha256": inputs,
        "inherited_source_sha256": inherited_sources,
        "inherited_input_sha256": inherited_inputs,
        "input_members": inventory["archives"],
        "reference": reference_name,
        "startup": old_paths["e103_startup"],
        "ordering_source_metadata": source_correspondence(sources, root=root),
        "original_analysis_authorized": False,
        "allocation": None,
        "candidate_adopted": False,
        "original_analysis_supervisor_required": True,
    }


def gate(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    timely=lambda: None,
):
    if Path(output).exists() or Path(output).resolve() != safe(OUTPUT, root):
        raise ValueError("fresh literal E336 output required")
    if (
        sha256(freeze_path, timely) != freeze_sha256
        or sha256(review_path, timely) != review_sha256
    ):
        raise ValueError("freeze/review SHA differs")
    freeze, review = read(freeze_path), read(review_path)
    expected = make_freeze(root=root)
    for field in expected:
        if field in {"original_analysis_authorized", "allocation"}:
            continue
        if freeze.get(field) != expected[field]:
            raise ValueError("exact frozen E336 contract differs: " + field)
    allocation = freeze.get("allocation")
    if freeze.get("original_analysis_authorized") is not True or allocation != {
        "cumulative_before_s": 9493.171036399524,
        "cumulative_limit_s": 9660,
        "original_analysis_reserved_s": 40,
    }:
        raise ValueError("root prospective saved-analysis40 allocation required")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("approved") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha256
        or review.get("source_sha256") != freeze["source_sha256"]
    ):
        raise ValueError("approved current-source review required")
    for alias, pin in FIXED.items():
        if freeze["input_sha256"].get(PATHS[alias]) != pin:
            raise ValueError("frozen reviewed design/helper pin differs")
    _pin_map(freeze["source_sha256"], root, timely)
    _pin_map(freeze["input_sha256"], root, timely)
    verify_previous(freeze["input_sha256"], root=root)
    inventory = read(safe(PATHS["inventory"], root))
    for report, descriptor, cp, sp in (
        (
            {
                "audit_source_path": PATHS["inventory_source"],
                "audit_source_sha256": inventory["source_sha256"],
            },
            {"path": PATHS["inventory"]},
            PATHS["inventory_contract"],
            PATHS["inventory_supervision"],
        ),
        (
            {
                "audit_source_path": PATHS["workload_source"],
                "audit_source_sha256": sha256(safe(PATHS["workload_source"], root)),
            },
            {"path": PATHS["workload"]},
            PATHS["workload_contract"],
            PATHS["workload_supervision"],
        ),
    ):
        contract, supervision = read(safe(cp, root)), read(safe(sp, root))
        _completed(contract, supervision, freeze["input_sha256"][cp], original_calls=0)
        verify_execution(report, descriptor, contract, supervision, root=root)
        verify_producer_closure(contract, freeze, root=root)
        if descriptor["path"] == PATHS["workload"]:
            workload = read(safe(PATHS["workload"], root))
            if (
                workload.get("passed") is not True
                or not workload.get("checks")
                or not all(value is True for value in workload["checks"].values())
            ):
                raise ValueError("completed metadata/synthetic workload proof required")
            if (
                contract["source_sha256"].get(PATHS["workload_math"])
                != freeze["input_sha256"][PATHS["workload_math"]]
            ):
                raise ValueError("executed workload imported math helper unpinned")
    qual = read(safe(PATHS["qualification"], root))
    own = {name: freeze["source_sha256"][name] for name in OWNED}
    if (
        qual.get("schema") != QUAL_SCHEMA
        or qual.get("approved") is not True
        or qual.get("blocking_defects") != []
        or qual.get("source_sha256") != own
    ):
        raise ValueError("current generated qualification required")
    report = read(safe(qual["cpu_report"]["path"], root))
    checks = report.get("checks", {})
    if (
        report.get("schema") != "flylab.m3.e336.root-generated.v1"
        or report.get("passed") is not True
        or not checks
        or not all(v is True for v in checks.values())
        or report.get("source_sha256") != own
        or any(
            report.get(k) != 0
            for k in (
                "original_analysis_calls",
                "neural_calls",
                "body_calls",
                "CUDA_calls",
            )
        )
    ):
        raise ValueError("exact completed CPU report required")
    for desc in (qual["cpu_report"], qual["cpu_contract"], qual["cpu_supervision"]):
        name = safe(desc["path"], root).relative_to(Path(root).resolve()).as_posix()
        if freeze["input_sha256"].get(name) != desc["sha256"]:
            raise ValueError("generated descriptor evidence pin differs")
    contract, supervision = (
        read(safe(qual[k]["path"], root)) for k in ("cpu_contract", "cpu_supervision")
    )
    _completed(contract, supervision, qual["cpu_contract"]["sha256"], original_calls=0)
    normalized = {}
    merge_pins(normalized, contract["source_sha256"], root=root)
    if any(normalized.get(name) != pin for name, pin in own.items()):
        raise ValueError("current own2 absent from completed producer contract")
    verify_execution(report, qual["cpu_report"], contract, supervision, root=root)
    verify_producer_closure(contract, freeze, root=root)
    reflection = read(safe(PATHS["reflection"], root))
    if reflection.get("accepted_all_mandatory_amendments") is not True or set(
        reflection.get("mandatory", {})
    ) != {"M" + str(i) for i in range(1, 9)}:
        raise ValueError("accepted actual Astra M1-M8 required")
    timely()
    return freeze


@dataclass(frozen=True)
class TestDimensions:
    nodes: int
    motors: int = 391

    def __post_init__(self):
        if (
            type(self.nodes) is not int
            or self.nodes < 1
            or type(self.motors) is not int
            or self.motors < 1
        ):
            raise ValueError("explicit positive generated dimensions required")


LIVE_FIELDS = {
    "prior_position": (3,),
    "prior_rotation": (3, 3),
    "angles_rad": (42,),
    "velocities_rad_s": (42,),
    "support_load_bw": (6,),
    "non_support_load_bw": (6,),
    "knee_flexion_rad": (6,),
    "knee_velocity_rad_s": (6,),
}
LIVE_CLOCKS = (
    "control_before",
    "control_after",
    "neural_before",
    "neural_after",
    "body_before",
    "body_after",
    "physics_returned_before",
    "physics_returned_after",
)
EVENT_ORDER = (
    "control_requested",
    "encode_requested",
    "leg_observation_returned",
    "knee_kinematics_returned",
    "encode_returned",
    "advance_readout_requested",
    "advance_readout_returned",
    "original_control_call_returned",
    "control_returned",
)


def verify_live_chunk(
    chunk, descriptor, events, raw, trace_model, *, nodes=N, motors=MOTORS
):
    ids = typed(chunk["control_ids"], (25,), "<i8", "live IDs")
    exact(ids, np.asarray(descriptor["controls"], np.int64), "descriptor IDs")
    if np.any(np.diff(ids) != 1) or ids[0] < 1 or ids[-1] > 300:
        raise ValueError("contiguous live controls1..300 required")
    fields = {
        **LIVE_FIELDS,
        "encoded_drive": (nodes,),
        "actual_drive": (nodes,),
        "motor_readout": (motors,),
    }
    required = {"control_ids"}
    for key, shape in fields.items():
        required.update((key, key + "_control_ids"))
        typed(
            chunk[key],
            (25, *shape),
            "<f4"
            if key in {"encoded_drive", "actual_drive", "motor_readout"}
            else "<f8",
            key,
        )
        exact(chunk[key + "_control_ids"], ids, key + " IDs")
    for key in LIVE_CLOCKS:
        required.update((key, key + "_control_ids"))
        expected = (ids - 1 if key.endswith("before") else ids).copy()
        if key.startswith("neural"):
            expected *= 20
        elif key.startswith("physics_returned"):
            expected *= 50
        elif key.startswith("body"):
            expected = expected.astype(np.float64) * 0.005
        typed(chunk[key], (25,), expected.dtype, key)
        exact(chunk[key + "_control_ids"], ids, key + " IDs")
        if key.startswith("body"):
            close(chunk[key], expected, key, atol=2e-8, rtol=0)
        else:
            exact(chunk[key], expected, key)
    if set(chunk) != required or set(descriptor["arrays"]) != required:
        raise ValueError("exact live input field roster required")
    for key, array in chunk.items():
        if metadata(array) != descriptor["arrays"][key]:
            raise ValueError("live field actual producer digest differs")
    availability = descriptor["availability"]
    if len(availability) != 25:
        raise ValueError("complete live availability required")
    for c, row in zip(ids, availability, strict=True):
        if (
            row.get("control") != int(c)
            or row.get("control_returned") is not True
            or row.get("error") is not None
        ):
            raise ValueError("successful live return availability required")
        for key in LIVE_CLOCKS:
            observed = row.get(key)
            expected = chunk[key][int(c - ids[0])]
            if key.startswith("body"):
                close(observed, expected, "availability " + key, rtol=0, atol=2e-8)
            elif observed != expected:
                raise ValueError("live availability clock differs")
        own = [e for e in events if e["control"] == int(c)]
        names = [e["event"] for e in own]
        if any(names.count(name) != 1 for name in EVENT_ORDER) or [
            names.index(name) for name in EVENT_ORDER
        ] != sorted(names.index(name) for name in EVENT_ORDER):
            raise ValueError("single original live call order differs")
        if (
            next(e for e in own if e["event"] == "knee_kinematics_returned")[
                "include_direction"
            ]
            is not False
        ):
            raise ValueError("original knee call kwargs differ")
        for key in fields:
            acquired = [e for e in own if e["event"] == key + "_acquired"]
            if len(acquired) != 1 or acquired[0]["sha256"] != bytes_sha(
                chunk[key][int(c - ids[0])]
            ):
                raise ValueError("actual acquired row digest differs")
    exact(
        chunk["motor_readout"],
        raw["motor_rate32"][ids],
        "live returned MN/raw boundary",
    )
    close(
        chunk["angles_rad"],
        trace_model["actual"][ids - 1],
        "pre-neural boundary angles",
    )
    close(
        chunk["knee_flexion_rad"],
        np.pi - np.deg2rad(trace_model["knee"][ids - 1]),
        "geometric knee complement",
    )
    close(
        np.einsum("ij,tj->ti", S, chunk["prior_position"]),
        trace_model["position"][ids - 1],
        "UI prior position",
    )
    close(
        np.einsum("ij,tjk->tik", S, chunk["prior_rotation"]),
        trace_model["basis"][ids - 1],
        "UI prior basis",
    )
    return {
        "controls": ids.tolist(),
        "motor_raw_exact": True,
        "prior_boundary_geometry": True,
        "receptor_load_is_not_native_floor_load": True,
        "velocities_not_finite_difference": True,
    }


def trace_data(path, raw, *, gait_module=None):
    if gait_module is None:
        from tools import analyze_c_banc_biological_gait as gait_module
    payload = read(path)
    rows = payload["samples"]
    if len(rows) != 301 or [r["control_tick"] for r in rows] != list(range(301)):
        raise ValueError("trace complete301 boundary rows required")
    if any((row.get("navigation") or {}).get("brake_request") for row in rows[100:300]):
        raise ValueError("fixed physical window must have no brake request")
    for key, expected in (
        ("control_tick", np.arange(301, dtype=np.int64)),
        ("network_tick", np.arange(301, dtype=np.int64) * 20),
    ):
        exact(raw[key], expected, key)
    close(
        raw["physics_time_s"],
        np.arange(301) * 0.005,
        "boundary physics clock",
        rtol=0,
        atol=2e-8,
    )
    close(
        [row["time_s"] for row in rows],
        np.arange(301) * 0.005,
        "trace clock",
        rtol=0,
        atol=2e-8,
    )
    model = gait_module.load_model(path)
    actual = np.array([r["joint_angles_rad"] for r in rows], np.float64)
    targets = np.array([r["joint_targets_rad"] for r in rows], np.float64)
    error = np.array([r["joint_tracking_error_rad"] for r in rows], np.float64)
    for key, a in (("actual", actual), ("targets", targets), ("error", error)):
        typed(a, (301, 42), "<f8", key)
    exact(targets[1:], raw["command"][1:], "trace interval command targets")
    close(actual - targets, -error, "trace tracking sign")
    model.update(
        actual=actual,
        targets=targets,
        stored_error=error,
        rows=rows,
        basis=np.array([row["body"]["basis"] for row in rows], np.float64),
        adhesion=np.array([row["adhesion"] for row in rows], np.bool_),
        payload_metadata={
            k: payload[k]
            for k in ("graph_hash", "model_hash", "adapter_hash", "body_hash")
        },
    )
    return model


def legacy_shape(q, curves):
    losses = []
    x = q.reshape(-1, 6, 7)
    for leg in range(6):
        a = curves[leg]
        d = np.roll(a, -1, axis=0) - a
        den = (d * d).sum(axis=1)
        offset = x[:, leg, None] - a[None]
        u = np.zeros(offset.shape[:2])
        np.divide(
            np.einsum("tij,ij->ti", offset, d), den[None], out=u, where=den[None] > 0
        )
        nearest = a[None] + np.clip(u, 0, 1)[..., None] * d[None]
        losses.append(
            float(
                np.min(
                    np.sum((x[:, leg, None] - nearest) ** 2, axis=-1) / 7, axis=1
                ).mean()
            )
        )
    return float(np.mean(losses))


def physical_metrics(raw, native, model, curves, startup, *, gait_module=None):
    if gait_module is None:
        from tools import analyze_c_banc_biological_gait as gait_module
    analyzed, phi, _ = gait_module.analyze(model)
    window = (model["t"] >= 0.5) & (model["t"] < 1.5)
    if window.sum() != 200:
        raise ValueError("official200 physical boundary samples required")
    forward = model["basis"][0, [0, 2], 0].copy()
    if np.linalg.norm(forward) == 0:
        raise ValueError("nonzero initial physical forward required")
    forward /= np.linalg.norm(forward)
    points = model["position"][window][:, [0, 2]]
    dq = raw["command"][1:100] - startup["target_command"]
    dt = raw["tendon"][1:100] - startup["target_tendon"]
    values = {
        "shape_loss": legacy_shape(raw["command"][101:], curves),
        "startup_q_rms": float(np.sqrt(np.mean(dq * dq))),
        "startup_q_max": float(np.abs(dq).max()),
        "startup_tendon_rms": float(np.sqrt(np.mean(dt * dt))),
        "startup_tendon_max": float(np.abs(dt).max()),
        "forward_mm": float((points[-1] - points[0]) @ forward),
        "longest_continuous_forward_bout_s": float(
            analyzed["window"]["longest_forward_bout_s"]
        ),
        "complete_cycles": {
            leg: int(analyzed["events"][leg]["complete_cycles"]) for leg in LEGS
        },
    }
    contact = native_statistics(native)
    values["slip_mm_s"] = contact["weighted_slip_mm_s"]
    values["startup_pass"] = all(
        values[key] <= bound
        for key, bound in {
            "startup_q_rms": 0.018,
            "startup_q_max": 0.04,
            "startup_tendon_rms": 0.009,
            "startup_tendon_max": 0.025,
        }.items()
    )
    values["minimum_native_upright"] = float(np.min(native["upright"]))
    values["nonfoot_contact_count"] = int(np.count_nonzero(native["nonfoot_contact"]))
    values["healthy"] = bool(
        np.isfinite(native["upright"]).all()
        and values["minimum_native_upright"] >= 0.9
        and values["nonfoot_contact_count"] == 0
        and not model["faults"]
    )
    values["short_fixed_pass"] = bool(
        values["healthy"]
        and values["forward_mm"] >= 1.6790085779712631
        and values["slip_mm_s"] is not None
        and values["slip_mm_s"] <= 1.8026436968249955
        and values["longest_continuous_forward_bout_s"] >= 0.3
        and all(v > 0 for v in values["complete_cycles"].values())
    )
    arrays = {
        "geometric_knee_interior_deg": model["knee"],
        "foot_ap_leg_lengths": model["ap"],
        "official_phase": phi,
        "eligible_moving": model["moving"],
        "boundary_adhesion": model["adhesion"],
    }
    return arrays, {
        "official": values,
        "native": contact,
        "gait": analyzed,
        "geometry_knee_span_deg": (
            np.percentile(model["knee"][100:300], 95, axis=0)
            - np.percentile(model["knee"][100:300], 5, axis=0)
        ).tolist(),
        "foot_ap_span_leg_lengths": (
            np.percentile(model["ap"][100:300], 95, axis=0)
            - np.percentile(model["ap"][100:300], 5, axis=0)
        ).tolist(),
        "exploratory_sensitivity_is_descriptive_only": True,
    }


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Dependencies:
    """Original saved arrays are reached only from run after passive gate."""

    def load(self, freeze, *, root=ROOT, timely=lambda: None, reads=None):
        reader = module(safe(PATHS["reader"], root), "e336_limited_state")
        constants = self.selected(freeze["reference"], freeze, root, timely, reads)
        ValueEncoding(constants)
        startup = load_selected(
            safe(freeze["startup"], root),
            {
                "target_command": {"shape": [99, 42], "dtype": "<f8"},
                "target_tendon": {"shape": [99, 6], "dtype": "<f8"},
            },
            reads=reads,
            timely=timely,
        )
        bodies = {}
        for label, folder in (
            ("neutral", "e318-full-drive-capture"),
            ("live", "e335-smooth-secant-body"),
        ):
            prefix = BASE + "/" + folder
            raw = self.selected(
                prefix + "/control-raw.npz", freeze, root, timely, reads
            )
            native = self.selected(
                prefix + "/native-contact.npz", freeze, root, timely, reads
            )
            model = trace_data(safe(prefix + "/trace.json", root), raw)
            state = reader.LimitedState(
                safe(prefix + "/observed-start", root), timely=timely
            )
            motor = state.leaf(
                ["adapter", "calibration", "motor_ids"], dtype="<i8", shape=(391,)
            )
            reads.append(
                dict(
                    path=prefix + "/observed-start",
                    semantic_leaf=["adapter", "calibration", "motor_ids"],
                    **metadata(motor),
                )
            )
            bodies[label] = {
                "raw": raw,
                "native": native,
                "model": model,
                "motor_ids": motor,
                "state_metadata": state.state,
            }
        exact(
            bodies["neutral"]["motor_ids"],
            bodies["live"]["motor_ids"],
            "ordered observed MN leaves",
        )
        held = self.selected(
            BASE + "/e334-smooth-constrained-secant/continuation-forward.npz",
            freeze,
            root,
            timely,
            reads,
        )
        e334 = read(safe(BASE + "/e334-smooth-constrained-secant/result.json", root))[
            "partial"
        ]
        proof = e334["actual_input_proof"]
        if metadata(bodies["neutral"]["motor_ids"]) != proof["motor_ids"]:
            raise ValueError("E334 actual ordered MN input descriptor differs")
        prior = read(safe(PATHS["prior_freeze"], root))
        graph_hash = prior["e318_config"]["graph_hash"]
        for body in bodies.values():
            if (
                body["model"]["payload_metadata"]["graph_hash"] != graph_hash
                or body["state_metadata"]["graph_hash"] != graph_hash
            ):
                raise ValueError("same source graph/node ordering required")
        audit = read(safe(PATHS["prior_audit"], root))
        if (
            bodies["live"]["state_metadata"]["network"]["model_hash"]
            != audit["metrics"]["native_candidate_model_hash"]
        ):
            raise ValueError("actual candidate native model identity differs")
        events = read(
            safe(BASE + "/e335-smooth-secant-body/live-input-events.json", root)
        )
        return {
            "constants": constants,
            "startup": startup,
            "bodies": bodies,
            "held": held,
            "held_proof": proof,
            "live_events": events,
            "tonic_profile": audit["metrics"]["retained_cut_cue_tonic_profile"],
            "prior_metrics": audit["metrics"],
            "held_metrics": read(safe(PATHS["e334_audit"], root))["metrics"],
            "held_smooth": e334["metrics"],
            "held_descriptors": read(
                safe(BASE + "/e318-full-drive-capture/result.json", root)
            )["chunks"],
        }

    @staticmethod
    def selected(name, freeze, root, timely, reads):
        descriptor = freeze["input_members"][name]
        schema = {
            key: {
                "shape": spec["shape"],
                "dtype": spec["dtype"],
                "finite": key != "slip_mm_s",
                **(
                    {"fortran_order": spec["fortran_order"]}
                    if "fortran_order" in spec
                    else {}
                ),
            }
            for key, spec in descriptor["members"].items()
        }
        return load_selected(safe(name, root), schema, reads=reads, timely=timely)

    def compute(
        self,
        data,
        freeze,
        save,
        *,
        root=ROOT,
        timely=lambda: None,
        reads=None,
        nodes=N,
        motors=MOTORS,
    ):
        report = {
            "trajectories": {},
            "correspondence": [],
            "interpretation_limits": [
                "Three saved trajectories, not independent biological replicates; phase sampling can alias",
                "R measures circular weight concentration, not reference proximity or statistical confidence",
                "Reference encodes common global phase with anatomical offsets already included",
                "Static residual is descriptive; no fixed-point stability/absence claim",
                "Held/live differences combine input and state changes; no causal feedback or body Jacobian claim",
                "Contact transitions are not steps; receptor support, adhesion and floor load differ",
            ],
        }
        for label in ("neutral", "held", "live"):
            q = (
                data["held"]["q"][99:]
                if label == "held"
                else data["bodies"][label]["raw"]["command"][100:301]
            )
            arrays, smooth = smooth_diagnostics(data["constants"], q)
            save(label + "-smooth.npz", arrays)
            expected = (
                data["held_metrics"]["actual_training_loss"]
                if label == "held"
                else data["prior_metrics"]["smooth"][
                    "neutral" if label == "neutral" else "candidate"
                ]["training_loss"]
            )
            close(smooth["training_loss"], expected, label + " saved audited smooth")
            expected_terms = (
                data["held_smooth"]
                if label == "held"
                else data["prior_metrics"]["smooth"][
                    "neutral" if label == "neutral" else "candidate"
                ]
            )
            for key in (
                "training_loss",
                "successor_loss",
                "coordination_loss",
                "lr_coordination_loss",
            ):
                close(smooth[key], expected_terms[key], label + " audited " + key)
            result = {"smooth": smooth}
            if label != "held":
                body = data["bodies"][label]
                m = body["model"]
                a, t = tracking_statistics(
                    body["raw"]["command"][100:300],
                    m["actual"][100:300],
                    m["stored_error"][100:300],
                    data["constants"]["normalization"],
                    data["constants"]["curves"],
                )
                save(label + "-tracking.npz", a)
                a, physical = physical_metrics(
                    body["raw"],
                    body["native"],
                    m,
                    data["constants"]["curves"],
                    data["startup"],
                )
                save(label + "-geometry.npz", a)
                expected_physical = data["prior_metrics"]["gait"][
                    "neutral" if label == "neutral" else "candidate"
                ]
                for key, value in physical["official"].items():
                    if isinstance(value, (dict, bool)) or value is None:
                        if value != expected_physical[key]:
                            raise ValueError(
                                "audited official physical value differs: " + key
                            )
                    else:
                        close(value, expected_physical[key], "official " + key)
                result.update(tracking=t, physical=physical)
            report["trajectories"][label] = result
            timely()
        comparisons = {key: [] for key in ("drive", "motor")}
        live = data["live_events"]
        if (
            live.get("schema") != "flylab.m3.e335.live-input.v1"
            or live.get("nodes") != nodes
            or len(live.get("chunks", [])) != 12
        ):
            raise ValueError("complete live12chunk producer required")
        for start, desc, held_desc, actual_desc in zip(
            range(1, 301, 25),
            live["chunks"],
            data["held_descriptors"],
            data["held_proof"]["drives_chunks"],
            strict=True,
        ):
            end = start + 24
            if (
                desc["controls"] != list(range(start, end + 1))
                or held_desc["start"] != start
                or held_desc["end"] != end
                or held_desc["count"] != 25
            ):
                raise ValueError("held/live chunk label alignment differs")
            lname = BASE + "/e335-smooth-secant-body/" + desc["file"]
            hname = (
                BASE + "/e318-full-drive-capture/full-drive-chunks/" + held_desc["file"]
            )
            for name, descriptor in ((lname, desc), (hname, held_desc)):
                if (
                    freeze["input_sha256"].get(name) != descriptor["sha256"]
                    or safe(name, root).stat().st_size != descriptor["bytes"]
                ):
                    raise ValueError("actual chunk producer path/SHA/bytes differs")
            chunk = self.selected(lname, freeze, root, timely, reads)
            held_chunk = self.selected(hname, freeze, root, timely, reads)
            expected_ids = np.arange(start, end + 1, dtype=np.int64)
            exact(held_chunk["control_tick"], expected_ids, "held control labels")
            exact(held_chunk["network_tick"], expected_ids * 20, "held neural clock")
            typed(held_chunk["drive"], (25, nodes), "<f4", "held drive")
            typed(held_chunk["motor_rate"], (25, motors), "<f4", "held motor")
            exact(
                held_chunk["motor_rate"],
                data["bodies"]["neutral"]["raw"]["motor_rate32"][expected_ids],
                "held/raw MN",
            )
            if actual_desc != {"start_control": start, **metadata(held_chunk["drive"])}:
                raise ValueError(
                    "E334 actual-used held host/device chunk descriptor differs"
                )
            report["correspondence"].append(
                verify_live_chunk(
                    chunk,
                    desc,
                    live["events"],
                    data["bodies"]["live"]["raw"],
                    data["bodies"]["live"]["model"],
                    nodes=nodes,
                    motors=motors,
                )
            )
            verify_tonic(
                chunk["encoded_drive"],
                chunk["actual_drive"],
                live["configuration"],
                data["tonic_profile"],
            )
            comparisons["drive"].append(
                difference_statistics(chunk["actual_drive"], held_chunk["drive"])
            )
            comparisons["motor"].append(
                difference_statistics(chunk["motor_readout"], held_chunk["motor_rate"])
            )
            del chunk, held_chunk
            timely()
        arrays = {
            kind + "_" + key: np.concatenate([item[key] for item in comparisons[kind]])
            for kind in comparisons
            for key in comparisons[kind][0]
        }
        arrays["control_ids"] = np.arange(1, 301, dtype=np.int64)
        save("input-differences.npz", arrays)
        report["input_difference_percentiles"] = {
            key: np.percentile(value, [5, 50, 95]).tolist()
            for key, value in arrays.items()
            if key != "control_ids"
        }
        qdelta = data["bodies"]["live"]["raw"]["command"][1:] - data["held"]["q"]
        tdelta = data["bodies"]["live"]["raw"]["tendon"][1:] - data["held"]["tendon"]
        save(
            "held-live-command-differences.npz",
            {
                "control_ids": np.arange(1, 301, dtype=np.int64),
                "command_difference": qdelta,
                "tendon_difference": tdelta,
            },
        )
        report["held_live_command_rms_rad"] = rms_leg(qdelta).tolist()
        report["held_live_tendon_rms"] = np.sqrt(
            np.mean(tdelta * tdelta, axis=0)
        ).tolist()
        held_span = np.percentile(
            data["held"]["q"][99:299], 95, axis=0
        ) - np.percentile(data["held"]["q"][99:299], 5, axis=0)
        live_span = np.percentile(
            data["bodies"]["live"]["raw"]["command"][100:300], 95, axis=0
        ) - np.percentile(data["bodies"]["live"]["raw"]["command"][100:300], 5, axis=0)
        report["held_live_span"] = {
            "controls": [100, 299],
            "numerator_live_span_rad": live_span.tolist(),
            "denominator_held_span_rad": held_span.tolist(),
            "live_over_held": ratios(live_span, held_span),
        }
        return report


def recursive_bytes(output):
    return sum(
        path.stat().st_size for path in Path(output).rglob("*") if path.is_file()
    )


def file_manifest(output):
    return {
        path.relative_to(output).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(Path(output).rglob("*"))
        if path.is_file()
        and path.name
        not in {
            "result.json",
            "advance-ledger.json",
            "output-manifest.json",
            "emergency-result.json",
        }
    }


def usage():
    memory = psutil.Process().memory_info()
    return {
        "host_rss_bytes": memory.rss,
        "host_peak_bytes": getattr(memory, "peak_wset", memory.rss),
    }


def run(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    root=ROOT,
    dependencies=None,
    gatekeeper=gate,
    test_dimensions=None,
    clock=time.perf_counter,
    writer=put_json,
    emergency_writer=put_json,
    saver=np.savez,
    meter=usage,
):
    final_deadline = None
    if test_dimensions is None:
        if (
            dependencies is not None
            or gatekeeper is not gate
            or meter is not usage
            or clock is not time.perf_counter
            or writer is not put_json
            or emergency_writer is not put_json
            or saver is not np.savez
        ):
            raise ValueError("generated injection requires explicit TestDimensions")
        if any(
            os.environ.get(key) != "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            raise ValueError("CPUthread1 supervisor environment required")
        nodes, motors = N, MOTORS
    else:
        if not isinstance(test_dimensions, TestDimensions) or dependencies is None:
            raise ValueError("explicit generated dependencies/dimensions required")
        nodes, motors = test_dimensions.nodes, test_dimensions.motors
    s0 = clock()
    output = Path(output)
    ledger = {
        "schema": "flylab.m3.e336.advance-ledger.v1",
        "limits": LIMITS.copy(),
        "errors": [],
        "numerical_reads": [],
        "saved_arrays": [],
        "original_analysis_calls": int(test_dimensions is None),
        "neural_calls": 0,
        "body_calls": 0,
        "CUDA_calls": 0,
        "retry_count": 0,
        "phases": [],
    }
    result = {
        "schema": RESULT_SCHEMA,
        "analysis_valid": False,
        "classification": "not_tested",
        "candidate_adopted": False,
        "biological_gait_proven": False,
        "F1_F8_complete": False,
        "F6_exposed": False,
        "M2_modified": False,
        "original_analysis_calls": int(test_dimensions is None),
        "neural_calls": 0,
        "body_calls": 0,
        "CUDA_calls": 0,
        "generated_explicit_fixture": test_dimensions is not None,
        "root_independent_raw_audit_required": True,
        "external_completion_required": True,
        "first_error": None,
        "errors": [],
        "provenance": {"freeze_sha256": freeze_sha256, "review_sha256": review_sha256},
    }
    first_error, errors = None, []

    def fail(stage, error):
        nonlocal first_error
        row = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error)[:500],
        }
        errors.append(row)
        if first_error is None:
            first_error = row
        result.update(
            analysis_valid=False,
            classification="invalid",
            first_error=first_error,
            errors=copy.deepcopy(errors),
        )

    def timely():
        if clock() >= (s0 + 30 if final_deadline is None else final_deadline):
            raise TimeoutError("absolute E336 deadline exceeded")

    def check():
        timely()
        value = meter()
        ledger["host_peak_bytes"] = max(
            value["host_peak_bytes"], ledger.get("host_peak_bytes", 0)
        )
        if ledger["host_peak_bytes"] > LIMITS["host_bytes"]:
            raise MemoryError("owned host2GiB limit")
        if recursive_bytes(output) > LIMITS["output_bytes"]:
            raise OSError("recursive output64MiB limit")

    # Every numerical reader/module remains downstream of this passive gate.
    freeze = gatekeeper(
        freeze_path,
        freeze_sha256,
        review_path,
        review_sha256,
        output,
        root=root,
        timely=timely,
    )
    timely()
    output.mkdir(parents=True, exist_ok=False)

    def save(name, arrays):
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("owned diagnostic NPZ name required")
        pending = output / (name + ".pending.npz")
        descriptors = {key: metadata(value) for key, value in arrays.items()}
        ledger["saved_arrays"].append(
            {"file": name, "arrays": descriptors, "write_returned": False}
        )
        saver(pending, **arrays)
        ledger["saved_arrays"][-1]["write_returned"] = True
        with np.load(pending, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("saved diagnostic NPZ key mismatch")
            for key in arrays:
                if metadata(archive[key]) != descriptors[key]:
                    raise ValueError("saved diagnostic NPZ readback mismatch")
        pending.replace(output / name)
        check()

    try:
        if shutil.disk_usage(output).free < LIMITS["minimum_free_disk_bytes"]:
            raise OSError("free disk1GiB reserve required")
        check()
        writer(
            output / "source-manifest.json",
            dict(
                schema="flylab.m3.e336.source-manifest.v1",
                **result["provenance"],
                source_sha256=freeze["source_sha256"],
                input_sha256=freeze["input_sha256"],
            ),
        )
        deps = Dependencies() if dependencies is None else dependencies
        began = clock()
        data = deps.load(
            freeze, root=root, timely=check, reads=ledger["numerical_reads"]
        )
        ledger["phases"].append(
            {"phase": "selective_load", "elapsed_s": clock() - began}
        )
        report = deps.compute(
            data,
            freeze,
            save,
            root=root,
            timely=check,
            reads=ledger["numerical_reads"],
            nodes=nodes,
            motors=motors,
        )
        writer(output / "diagnosis.json", report)
        result.update(
            analysis_valid=True,
            classification="valid_saved_diagnosis",
            diagnosis=report,
        )
        check()
        _pin_map(freeze["source_sha256"], root, check)
        _pin_map(freeze["input_sha256"], root, check)
        check()
    except Exception as error:  # noqa: BLE001
        fail("science", error)
    final_begin = clock()
    final_deadline = min(s0 + 35, final_begin + 5)
    ledger["absolute_clock"] = {
        "s0": s0,
        "science_deadline": s0 + 30,
        "final_begin": final_begin,
        "final_deadline": final_deadline,
    }

    def publish(sink):
        result.update(first_error=first_error, errors=copy.deepcopy(errors))
        ledger.update(
            analysis_valid=result["analysis_valid"],
            first_error=first_error,
            errors=copy.deepcopy(errors),
            elapsed_s=clock() - s0,
        )
        sink(
            output / "receipts.json",
            {
                "schema": "flylab.m3.e336.receipts.v1",
                "numerical_reads": ledger["numerical_reads"],
                "saved_arrays": ledger["saved_arrays"],
                "original_analysis_calls": result["original_analysis_calls"],
                "neural_calls": 0,
                "body_calls": 0,
                "CUDA_calls": 0,
            },
        )
        manifest = {
            "schema": "flylab.m3.e336.output-manifest.v1",
            "files": file_manifest(output),
            "analysis_valid": result["analysis_valid"],
            "external_completion_required": True,
        }
        sink(output / "advance-ledger.json", ledger)
        sink(output / "result.json", result)
        sink(output / "output-manifest.json", manifest)
        if (
            read(output / "result.json") != result
            or read(output / "advance-ledger.json") != ledger
            or read(output / "output-manifest.json") != manifest
        ):
            raise ValueError("terminal disk/return readback disagreement")
        if file_manifest(output) != manifest["files"]:
            raise ValueError("raw manifest changed after publication")
        check()  # Late writes/readback/resource checks use the same absolute clock.

    try:
        publish(writer)
    except Exception as error:  # noqa: BLE001
        fail("publication", error)
        try:
            publish(emergency_writer)
        except Exception as secondary:  # noqa: BLE001
            fail("repair", secondary)
            try:
                publish(emergency_writer)
            except Exception as last:  # noqa: BLE001
                fail("emergency", last)
                result["disk_agreement_established"] = False
                try:
                    result.update(errors=copy.deepcopy(errors), first_error=first_error)
                    ledger.update(
                        analysis_valid=False,
                        first_error=first_error,
                        errors=copy.deepcopy(errors),
                    )
                    emergency_writer(output / "result.json", result)
                    emergency_writer(output / "advance-ledger.json", ledger)
                    emergency_writer(output / "emergency-result.json", result)
                    check()
                except Exception as final_error:  # noqa: BLE001
                    result["emergency_sink_error"] = repr(final_error)[:500]
                    try:
                        emergency_writer(output / "result.json", result)
                    except Exception:  # noqa: BLE001,S110 - explicit sink failure retained
                        pass
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(
        args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.output
    )
    print(
        json.dumps(
            {
                "analysis_valid": result["analysis_valid"],
                "first_error": result["first_error"],
            },
            allow_nan=False,
        )
    )
    return 0 if result["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
