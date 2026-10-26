"""E354 offline reference labels and q-only loss; no neural or body execution."""

from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNED = (
    "tools/banc_m3_reference_cadence_teacher.py",
    "tests/test_c_banc_reference_cadence_teacher.py",
)
REGISTRATION = BASE + "/parent-e354-teacher-input-registration.json"
REG_SHA = "86cbe13d14b43872fa64a8cd16332776aaefb87bcc38034ab6056f25988ac52f"
REFLECTION = BASE + "/parent-e354-reflected-design-registration.json"
REF_SHA = "400cae6d54dffbb49f07d33c6f2edadcefee65c40c6b23da1c53fe82c2b15a52"
HELPER_SHA = "a9a5804b9b48adffb41c43c91931c5d1664b43e17624f1f0c70478a7105c1a04"
OUTPUT = BASE + "/e354-reference-teacher-loss"
AUDIT_OUTPUT = BASE + "/e354-reference-teacher-audit"
STARTS = (100, 164, 228, 292)
CAP, EMERGENCY = 4 << 20, 64 << 10
CHECKS = (
    "CPU",
    "ruff",
    "full_window_derivatives",
    "mutation_isolation",
    "finite_device",
    "selected7",
    "producer_audit",
    "bad_raw_result",
    "late_failure",
    "publication_caps",
    "private_gate",
    "lazy_import_help",
)
DENIED = dict.fromkeys(
    (
        "parameter_VJP_qualified",
        "gradient_step_improved",
        "candidate_adopted",
        "BANC_autonomy",
        "F1_F8_complete",
        "biology",
        "natural_gait",
        "M2_UI_F6_changed",
        "actual_CUDA_qualified",
    ),
    False,
)
_HELPER = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def helper():
    global _HELPER
    if _HELPER is None:
        file = ROOT / "tools/recover_c_banc_m3_source_pose_report.py"
        require(sha(file.read_bytes()) == HELPER_SHA, "frozen minimal E351 helper")
        spec = importlib.util.spec_from_file_location("e354_reader_primitives", file)
        _HELPER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HELPER)
    return _HELPER


def numpy():
    import numpy as np

    return np


def torch_module():
    import torch

    return torch


class ReferenceCadenceTeacher:
    """Explicit-device f64 constants; whole300 and fixed four windows16."""

    def __init__(self, target_q, normalization, *, device):
        t = torch_module()
        require(
            isinstance(target_q, t.Tensor) and isinstance(normalization, t.Tensor),
            "tensor constants",
        )
        require(target_q.dtype == normalization.dtype == t.float64, "f64 constants")
        require(
            tuple(target_q.shape) == (300, 42) and tuple(normalization.shape) == (6,),
            "constant shape",
        )
        self.device = t.device(device)
        self._target = target_q.detach().to(self.device).clone()
        self.device = self._target.device
        self._norm = normalization.detach().to(self.device).clone()
        self._finite(self._target)
        self._finite(self._norm)
        t._assert_async((self._norm > 0).all(), "positive normalization")
        self._window_target = t.stack([self._target[s : s + 4] for s in STARTS]).clone()
        self._expanded_norm = self._norm.repeat_interleave(7)

    @staticmethod
    def _finite(value):
        t = torch_module()
        t._assert_async(t.isfinite(value).all(), "finite E354 tensor")

    def _input(self, q, shape):
        t = torch_module()
        require(isinstance(q, t.Tensor) and q.dtype == t.float64, "q f64 tensor")
        require(
            q.device == self.device and tuple(q.shape) == shape, "exact q device/shape"
        )
        self._finite(q)

    def full(self, q):
        self._input(q, (300, 42))
        loss = ((q - self._target).square() / self._expanded_norm).mean()
        self._finite(loss)
        return loss

    def window(self, q):
        self._input(q, (4, 4, 42))
        loss = ((q - self._window_target).square() / self._expanded_norm).mean()
        self._finite(loss)
        return loss

    def validate_gradient(self, value, *, window=False):
        self._input(value, (4, 4, 42) if window else (300, 42))
        return value


def descriptor(name, raw):
    return {"path": name, "bytes": len(raw), "sha256": sha(raw)}


def own_sources(root=ROOT):
    return {name: sha(helper().safe(name, root).read_bytes()) for name in OWNED}


def json_bytes(value):
    return json.dumps(
        helper().lower(value), allow_nan=False, separators=(",", ":")
    ).encode("utf8")


def error_record(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}


def raw_roster():
    return {
        **{k: ((300, 42), "<f8") for k in ("source_q", "target_q", "full_gq")},
        "normalization": ((6,), "<f8"),
        **{
            k: ((4, 4, 42), "<f8")
            for k in ("window_source_q", "window_target_q", "window_gq")
        },
        "window_control_tick": ((4, 4), "<i8"),
        **{
            k: ((), "<f8")
            for k in (
                "full_loss",
                "window_loss",
                "prefix_loss",
                "startup_loss",
                "teacher_self_loss",
            )
        },
    }


def generated_reader(root):
    root = Path(root).resolve()
    require(
        not root.is_relative_to(ROOT)
        or root.relative_to(ROOT).as_posix().startswith(BASE + "/e354-sol-FIRST-"),
        "owned E354 fixture",
    )
    reader = helper().EvidenceReader(root)
    reader.array_authorized = True
    return reader


def metadata_preflight(root=ROOT, reader=None):
    r = reader or helper().EvidenceReader(root)

    def pinned(name, digest):
        f = helper().safe(name, root)
        return r.json({"path": name, "bytes": f.stat().st_size, "sha256": digest})

    reg = pinned(REGISTRATION, REG_SHA)
    ref = pinned(REFLECTION, REF_SHA)
    require(
        reg["schema"] == "flylab.m3.e354.teacher-input-registration.v1"
        and reg["original_authorized"] is False,
        "E354 registration",
    )
    require(
        ref["registration"]["sha256"] == REG_SHA
        and ref["original_authorized"] is False,
        "reflected registration",
    )
    for spec in (reg["design"], reg["actual_review"], reg["review_md"]):
        r.bytes(spec)
    for name, pin in reg["consumed_sources"].items():
        require(
            name.endswith(".py") and sha(helper().safe(name, root).read_bytes()) == pin,
            "consumed PY pin",
        )
    for name, pin in reg["inherited_source_pins"].items():
        require(
            name.endswith(".py") and sha(helper().safe(name, root).read_bytes()) == pin,
            "historical text-only source pin",
        )
    meta = {
        name: r.json(spec) for name, spec in reg["inherited_teacher_provenance"].items()
    }
    decision = meta[BASE + "/parent-e353-final-decision.json"]
    report = meta[BASE + "/e353-reference-cadence-audit/report.json"]
    result = meta[BASE + "/e353-reference-cadence-body/result.json"]
    require(
        decision["diagnostic_completion_qualified"] is True
        and decision["H353_supported_for_later_learning_eligibility"] is True
        and all(
            v > 0 for v in decision["verified_metrics"]["complete_cycles"].values()
        ),
        "qualified all-six teacher",
    )
    require(
        report["audit_passed"] is True
        and result["status"] == "completed"
        and result["external_completion_qualified"] is False
        and report["external_completion_qualified"] is False,
        "historical provisional flags preserved",
    )
    require(
        decision["fixed_frequency_Hz"] == reg["teacher_frequency_Hz"]
        and decision["source_sha256"]
        == result["source_sha256"]
        == report["source_sha256"],
        "teacher identity/frequency",
    )
    cadence = reg["payloads"]["cadence"]
    require(
        result["raw"]["cadence-plan.npz"]["sha256"]
        == cadence["sha256"]
        == report["consumed_digests"][cadence["path"]],
        "cadence acquisition linkage",
    )
    require(
        set(reg["payloads"]) == {"control", "reference", "cadence"}
        and set(reg["allowed_original_npz_paths"])
        == {s["path"] for s in reg["payloads"].values()},
        "exact original3 whitelist",
    )
    for role, names in {
        "control": {"command", "control_tick", "physics_time_s"},
        "reference": {"normalization"},
        "cadence": {"command", "control_tick", "control_time_s"},
    }.items():
        s = reg["payloads"][role]
        require(
            set(s["selected_members"]) == names
            and helper().safe(s["path"], root).stat().st_size == s["bytes"],
            "selected7 stat-only",
        )
    require(
        reg["fixed_window_starts"] == list(STARTS)
        and reg["completed_raw_members"]
        == {k: {"shape": list(s), "dtype": d} for k, (s, d) in raw_roster().items()},
        "fixed windows/exact13",
    )
    return {"reader": r, "registration": reg, "authorized": False}


def load_inputs(ctx):
    require(ctx.get("authorized") is True, "own private gate before array read")
    r, reg, np = ctx["reader"], ctx["registration"], numpy()
    data = {}
    for role, spec in reg["payloads"].items():
        data[role] = r.npz(
            spec,
            {
                k: (tuple(v["shape"]), v["dtype"])
                for k, v in spec["selected_members"].items()
            },
            selected=spec["selected_members"],
        )
    control, cadence = data["control"], data["cadence"]
    for values, clock in ((control, "physics_time_s"), (cadence, "control_time_s")):
        require(
            all(np.isfinite(v).all() for v in values.values()), "finite selected values"
        )
        require(
            np.array_equal(values["control_tick"], np.arange(301, dtype=np.int64))
            and np.allclose(values[clock], np.arange(301) * 0.005, atol=2e-8, rtol=0),
            "selected indices/clock",
        )
    require(
        control["command"][:101].tobytes() == cadence["command"][:101].tobytes(),
        "literal c0 and prefix byte identity",
    )
    norm = data["reference"]["normalization"]
    require(np.isfinite(norm).all() and (norm > 0).all(), "finite positive norm")
    return {
        "source_q": control["command"][1:].copy(),
        "target_q": cadence["command"][1:].copy(),
        "normalization": norm.copy(),
    }


def produce(data, *, timely=lambda: None, gradient=None):
    np, t = numpy(), torch_module()
    acquired = {}
    receipt = {
        "schema": "flylab.m3.e354.teacher-receipt.v1",
        "status": "technical_invalid",
        "stage": "constants",
        "first_error": None,
        "secondary_errors": [],
        "full_requested": 0,
        "full_returned": 0,
        "window_requested": 0,
        "window_returned": 0,
        "teacher_self_loss_direct_evaluations": 0,
        "teacher_self_gradient_measured_autograd": 0,
        "teacher_self_gradient_scope": "algebraic_identity_zero",
        "device": "cpu",
        "dtype": "float64",
        "q_only": True,
        "NN_body_CPG_GPU_calls": 0,
    }

    def save(name, value):
        arr = (
            value.detach().numpy().copy()
            if isinstance(value, t.Tensor)
            else np.asarray(value).copy()
        )
        require(np.isfinite(arr).all(), "finite acquired " + name)
        acquired[name] = arr

    try:
        timely()
        target = t.from_numpy(data["target_q"])
        norm = t.from_numpy(data["normalization"])
        teacher = ReferenceCadenceTeacher(target, norm, device="cpu")
        receipt["target_requires_grad"] = teacher._target.requires_grad
        receipt["normalization_requires_grad"] = teacher._norm.requires_grad
        q = t.from_numpy(data["source_q"]).clone().requires_grad_(True)
        qw = (
            t.stack([q.detach()[s : s + 4] for s in STARTS])
            .clone()
            .requires_grad_(True)
        )
        receipt["full_is_leaf"] = q.is_leaf
        receipt["window_is_leaf"] = qw.is_leaf
        for name, value in (
            ("source_q", q),
            ("target_q", teacher._target),
            ("normalization", teacher._norm),
            ("window_source_q", qw),
            ("window_target_q", teacher._window_target),
        ):
            save(name, value)
        save(
            "window_control_tick",
            np.asarray([[s + k + 1 for k in range(4)] for s in STARTS], np.int64),
        )
        for mode, leaf, loss_fn in (
            ("full", q, teacher.full),
            ("window", qw, teacher.window),
        ):
            receipt["stage"] = mode + "_loss"
            timely()
            loss = loss_fn(leaf)
            save(mode + "_loss", loss)
            receipt["stage"] = mode + "_gradient"
            receipt[mode + "_requested"] += 1
            g = (gradient or t.autograd.grad)(loss, leaf)[0]
            receipt[mode + "_returned"] += 1
            teacher.validate_gradient(g, window=mode == "window")
            save(mode + "_gq", g)
            timely()
        receipt["stage"] = "label_diagnostics"
        residual = (q.detach() - teacher._target).square() / teacher._expanded_norm
        save("prefix_loss", residual[:100].mean())
        save("startup_loss", residual[:99].mean())
        receipt["teacher_self_loss_direct_evaluations"] += 1
        self_loss = teacher.full(teacher._target)
        save("teacher_self_loss", self_loss)
        timely()
        receipt["status"] = "completed"
        receipt["stage"] = "acquisition_completed"
    except Exception as error:  # noqa: BLE001 -- retain actual gradient request/return/acquired subset
        receipt["first_error"] = error_record(receipt["stage"], error)
    receipt["acquired"] = sorted(acquired)
    receipt["missing"] = sorted(set(raw_roster()) - set(acquired))
    return acquired, receipt


def independent_check(data, raw, receipt, result=None):
    """NumPy/math only; neither teacher methods nor Torch autograd are called."""
    np = numpy()
    require(set(raw) == set(raw_roster()), "completed exact13")
    for name, (shape, dtype) in raw_roster().items():
        require(
            raw[name].shape == shape
            and raw[name].dtype == np.dtype(dtype)
            and np.isfinite(raw[name]).all(),
            "raw finite dtype/shape " + name,
        )
    require(
        receipt["status"] == "completed"
        and receipt["first_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["missing"] == []
        and set(receipt["acquired"]) == set(raw),
        "completed acquisition receipt",
    )
    require(
        all(
            type(receipt[k]) is int and receipt[k] == 1
            for k in (
                "full_requested",
                "full_returned",
                "window_requested",
                "window_returned",
                "teacher_self_loss_direct_evaluations",
            )
        )
        and type(receipt["teacher_self_gradient_measured_autograd"]) is int
        and receipt["teacher_self_gradient_measured_autograd"] == 0
        and receipt["teacher_self_gradient_scope"] == "algebraic_identity_zero",
        "actual two independent gradients/self scope",
    )
    require(
        receipt["target_requires_grad"] is False
        and receipt["normalization_requires_grad"] is False
        and receipt["full_is_leaf"] is True
        and receipt["window_is_leaf"] is True
        and receipt["device"] == "cpu"
        and receipt["dtype"] == "float64"
        and receipt["q_only"] is True
        and receipt["NN_body_CPG_GPU_calls"] == 0,
        "CPU64 q-only leaves",
    )
    for name in ("source_q", "target_q", "normalization"):
        require(
            raw[name].tobytes() == data[name].tobytes(),
            "original label byte linkage " + name,
        )
    starts = (100, 164, 228, 292)
    indices = np.asarray([[s + k for k in range(4)] for s in starts], np.int64)
    require(
        np.array_equal(raw["window_control_tick"], indices + 1),
        "independent window index",
    )
    for name in ("source_q", "target_q"):
        require(
            raw["window_" + name].tobytes() == data[name][indices].tobytes(),
            "independent window labels",
        )
    norm = data["normalization"]
    require(np.isfinite(norm).all() and (norm > 0).all(), "positive norm")

    def formula(q, target):
        flatq, flatt = q.reshape(-1, 42), target.reshape(-1, 42)
        count = len(flatq) * 42
        terms, gradient = [], np.empty_like(flatq)
        for i in range(len(flatq)):
            for j in range(42):
                e = float(flatq[i, j]) - float(flatt[i, j])
                n = float(norm[j // 7])
                terms.append(e * e / n)
                gradient[i, j] = 2 * e / (count * n)
        loss = math.fsum(terms) / count
        require(
            math.isfinite(loss) and np.isfinite(gradient).all(),
            "finite independent math",
        )
        return loss, gradient.reshape(q.shape)

    full, gf = formula(data["source_q"], data["target_q"])
    window, gw = formula(data["source_q"][indices], data["target_q"][indices])
    prefix, _ = formula(data["source_q"][:100], data["target_q"][:100])
    startup, _ = formula(data["source_q"][:99], data["target_q"][:99])
    metrics = {
        "full_loss": full,
        "window_loss": window,
        "prefix_loss": prefix,
        "startup_loss": startup,
        "teacher_self_loss": 0.0,
    }
    for name, value in metrics.items():
        require(
            math.isclose(float(raw[name]), value, rel_tol=1e-11, abs_tol=1e-12),
            "independent loss " + name,
        )
    require(
        prefix == startup == 0 and float(raw["teacher_self_loss"]) == 0,
        "label-only zero diagnostics",
    )
    require(
        np.allclose(raw["full_gq"], gf, rtol=1e-11, atol=1e-12)
        and np.allclose(raw["window_gq"], gw, rtol=1e-11, atol=1e-12)
        and np.allclose(
            raw["window_gq"],
            (300 / 16) * raw["full_gq"][indices],
            rtol=1e-11,
            atol=1e-12,
        ),
        "independent derivatives/mean300:16",
    )
    if result is not None:
        require(
            result["status"] == "completed"
            and result["receipt"] == receipt
            and result["H354"] == "qualified_offline_teacher_loss"
            and result["external_completion_qualified"] is False
            and all(result[k] is False for k in DENIED),
            "stored result scope",
        )
        for name, value in metrics.items():
            require(
                math.isclose(
                    result["metrics"][name], value, rel_tol=1e-11, abs_tol=1e-12
                ),
                "stored result math",
            )
    return metrics


def npz_pack(raw):
    np = numpy()
    require(set(raw) <= set(raw_roster()), "actual acquired subset")
    stream = io.BytesIO()
    np.savez_compressed(stream, **raw)
    payload = stream.getvalue()
    members = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name, value in raw.items():
            shape, dtype = raw_roster()[name]
            require(
                value.shape == shape
                and value.dtype == np.dtype(dtype)
                and np.isfinite(value).all(),
                "staged finite shape/dtype",
            )
            info = archive.getinfo(name + ".npy")
            with archive.open(info) as child:
                version = np.lib.format.read_magic(child)
                actual_shape, fortran, actual_dtype = (
                    np.lib.format.read_array_header_1_0(child)
                )
                require(
                    version == (1, 0)
                    and actual_shape == shape
                    and not fortran
                    and actual_dtype == value.dtype,
                    "acquired new header",
                )
                members[name] = {
                    "member": info.filename,
                    "shape": list(shape),
                    "dtype": str(actual_dtype),
                    "fortran_order": False,
                    "npy_version": [1, 0],
                    "header_bytes": child.tell(),
                    "zip_compression": info.compress_type,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": info.CRC,
                }
    return payload, members


def write_raw(file, raw, *, timely=lambda: None):
    timely()
    with file.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    timely()


def failure(output, receipt, error, stage, progress):
    report = {
        "schema": "flylab.m3.e354.teacher-failure.v1",
        "status": "technical_invalid",
        "stage": stage,
        "first_error": receipt.get("first_error") or error_record(stage, error),
        "secondary_errors": (
            receipt.get("secondary_errors", []) + [error_record(stage, error)]
        )[:16],
        "receipt": receipt,
        "staged": progress.get("staged", {}),
        "published": progress.get("published", {}),
        "unpublished_hashes_are_saved_proof": False,
        "late_error_has_precedence": True,
        "H354": "not_evaluated",
        **DENIED,
    }
    payload = json_bytes(report)
    require(
        len(payload) <= EMERGENCY and helper().usage(output) + len(payload) <= CAP,
        "failure64KiB/total4MiB",
    )
    write_raw(output / "failure.json", payload)
    return report


def publish_worker(output, data, raw, receipt, *, timely=lambda: None, progress=None):
    progress = progress if progress is not None else {}
    try:
        require(
            set(raw) == set(receipt["acquired"])
            and set(receipt["missing"]) == set(raw_roster()) - set(raw),
            "honest acquired/missing",
        )
        payload, members = npz_pack(raw)
        spec = dict(descriptor("raw.npz", payload), selected_members=members)
        metrics = (
            {
                k: float(raw[k])
                for k in (
                    "full_loss",
                    "window_loss",
                    "prefix_loss",
                    "startup_loss",
                    "teacher_self_loss",
                )
            }
            if receipt["status"] == "completed"
            else None
        )
        result = {
            "schema": "flylab.m3.e354.teacher-result.v1",
            "status": receipt["status"],
            "source_sha256": own_sources(),
            "registration_sha256": REG_SHA,
            "reflection_sha256": REF_SHA,
            "raw": spec,
            "receipt": receipt,
            "metrics": metrics,
            "H354": "qualified_offline_teacher_loss"
            if metrics is not None
            else "not_evaluated",
            "external_completion_qualified": False,
            "first_error": receipt["first_error"],
            "secondary_errors": receipt["secondary_errors"],
            **DENIED,
        }
        staged = {
            "raw.npz": payload,
            "receipt.json": json_bytes(receipt),
            "result.json": json_bytes(result),
        }
        require(sum(len(v) for v in staged.values()) <= 64 << 20, "stage64MiB")
        progress.update(
            staged={n: descriptor(n, v) for n, v in staged.items()}, published={}
        )
        require(
            helper().usage(output) + sum(len(v) for v in staged.values()) + EMERGENCY
            <= CAP,
            "reserve raw/receipt/result/emergency before raw",
        )
        for name, value in staged.items():
            write_raw(output / name, value, timely=timely)
            progress["published"][name] = progress["staged"][name]
        timely()
        require(helper().usage(output) <= CAP, "closed total4MiB")
        require(receipt["status"] == "completed", "incomplete q-only acquisition")
        return result, 0 if receipt["status"] == "completed" else 1
    except Exception as error:  # noqa: BLE001 -- acquired receipt and raw survive publication/close failure
        return failure(output, receipt, error, "publication", progress), 1


def audit_worker(ctx, data):
    require(
        ctx.get("authorized") is True and ctx.get("mode") == "audit", "own audit gate"
    )
    seal, r = ctx["worker_seal"], ctx["reader"]
    require(
        type(seal.get("expected_exit_code")) is int and seal["expected_exit_code"] == 0,
        "completed worker exit0 before newraw",
    )
    roster = seal["artifacts"]
    require(
        set(roster) == {"raw.npz", "receipt.json", "result.json"},
        "complete saved worker roster",
    )
    require(
        sum(s["bytes"] for s in roster.values()) <= CAP
        and helper().usage(helper().safe(OUTPUT, r.root)) <= CAP,
        "sealed actual worker4MiB",
    )

    def spec(name):
        s = dict(roster[name])
        s["path"] = OUTPUT + "/" + name
        return s

    receipt, result = r.json(spec("receipt.json")), r.json(spec("result.json"))
    require(
        receipt["status"] == "completed" and result["status"] == "completed",
        "completed before newraw",
    )
    require(
        result["source_sha256"] == own_sources()
        and result["registration_sha256"] == REG_SHA
        and result["reflection_sha256"] == REF_SHA,
        "current worker linkage",
    )
    rawspec = result["raw"]
    require(
        all(rawspec[k] == roster["raw.npz"][k] for k in ("path", "bytes", "sha256"))
        and set(rawspec["selected_members"]) == set(raw_roster()),
        "saved raw descriptor/exact13",
    )
    raw = r.npz(spec("raw.npz"), raw_roster(), selected=rawspec["selected_members"])
    # selected metadata would allow extras; E354 completed raw explicitly forbids them.
    with zipfile.ZipFile(io.BytesIO(r.cache[OUTPUT + "/raw.npz"])) as archive:
        require(
            set(archive.namelist()) == {k + ".npy" for k in raw_roster()},
            "raw exact13 no extras",
        )
    metrics = independent_check(data, raw, receipt, result)
    return {
        "schema": "flylab.m3.e354.teacher-audit.v1",
        "audit_passed": True,
        "H354": "qualified_offline_teacher_loss",
        "metrics": metrics,
        "source_sha256": own_sources(),
        "registration_sha256": REG_SHA,
        "read_counts": r.read_counts,
        "consumed_digests": r.digests,
        "opened_members": r.opened_members,
        "Torch_autograd_body_NN_GPU_calls": 0,
        "external_completion_qualified": False,
        **DENIED,
    }


def check_execution(reader, execution, own, cap, *, original=0, argv=None):
    contract, supervision, identity, full = (
        reader.json(execution[k])
        for k in ("contract", "supervision", "identity", "full_exit")
    )
    wall = full["full_exit_elapsed_s"]
    require(
        full["schema"] == "flylab.m3.e354.actual-full-return.v1"
        and full["scope"] == "prevalidation-through-observer-final-exit"
        and full["source_sha256"] == own
        and full["exit_code"] == 0
        and full["all_owned_ended"] is True
        and full["known_identity_cleanup"] == []
        and full["first_error"] is None
        and full["secondary_errors"] == []
        and type(wall) in (float, int)
        and math.isfinite(wall)
        and 0 < wall <= cap
        and wall == full["actual_outer_full_return_s"] == full["charged_s"]
        and full["charged_once"] is True
        and full["original_calls"] == original,
        "actual outer full return",
    )
    require(
        all(full[k] == execution[k] for k in ("contract", "supervision", "identity")),
        "execution descriptor linkage",
    )
    outer, tool = (
        reader.json(full["outer_witness"]),
        reader.json(full["actual_tool_return"]),
    )
    require(
        tool["actual_returned_exit_code"] == outer["exit_code"] == 0
        and tool["actual_outer_full_return_s"] == wall
        and tool["tool_chunk_id"] == full["actual_tool_chunk_id"]
        and outer["all_owned_ended"] is True
        and outer["first_error"] is None
        and outer["secondary_errors"] == []
        and outer["known_identity_cleanup"] == [],
        "actual returned tool/outer linkage",
    )
    require(
        contract["original_calls"] == original
        and all(contract["source_sha256"].get(k) == v for k, v in own.items())
        and supervision["success"] is True
        and identity["qualified"] is True
        and identity["all_owned_pids_ended"] is True
        and identity["actual_supervisor_exit_code"] == 0
        and identity["errors"] == []
        and identity["contract"] == execution["contract"]
        and identity["supervision"] == execution["supervision"],
        "current supervised sources/exits",
    )
    require(
        supervision["max_tree_rss_bytes"] <= contract["max_tree_rss_bytes"] <= 2 << 30
        and len(contract["commands"]) == len(supervision["attempts"])
        and all(
            a["argv"] == c["argv"] and a["exit_code"] == 0 and not a["killed"]
            for a, c in zip(supervision["attempts"], contract["commands"], strict=True)
        ),
        "actual argv/RSS/no kills",
    )
    observed = {
        row["pid"]: row["create_time"]
        for row in outer["owned_identities"]
        if row["close"] == "ended"
    }
    require(
        identity["owned_pids"]
        and all(
            observed.get(pid) == identity["pid_create_times"][str(pid)]
            and identity["present_after_return"][str(pid)] is False
            and type(observed[pid]) in (int, float)
            and math.isfinite(observed[pid])
            for pid in identity["owned_pids"]
        ),
        "live observed creation identities closed",
    )
    if argv is not None:
        require(
            len(contract["commands"]) == 1 and contract["commands"][0]["argv"] == argv,
            "sole worker argv",
        )
    return full


def primary_cpu_role(reader, execution, own, *, root=ROOT, extra_cpu_tests=()):
    """Bind CPU claims to one complete current pytest suite, never process-only PASS."""
    contract = reader.json(execution["contract"])
    commands = contract.get("commands")
    require(type(commands) is list and len(commands) == 1, "sole primary pytest role")
    argv = commands[0].get("argv")
    prefix = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    require(
        type(argv) is list and all(type(v) is str for v in argv) and argv[:9] == prefix,
        "canonical primary pytest role",
    )
    tail = argv[9:]
    if tail[:1] == ["--basetemp"]:
        require(len(tail) >= 3, "explicit generated basetemp")
        folder = Path(tail[1])
        require(
            folder.is_absolute()
            and not folder.resolve().is_relative_to(root.resolve())
            and folder.name.startswith(
                ("e354-sol-FIRST-", "e354-sol-R1-", "e354-root-")
            )
            and ".." not in folder.parts,
            "owned external generated basetemp",
        )
        tail = tail[2:]
    require(
        tail and tail[0] == OWNED[1] and len(set(tail)) == len(tail),
        "complete owned current test suite",
    )
    require(type(extra_cpu_tests) in (list, tuple), "registered extra CPU tests")
    registered = {}
    for spec in extra_cpu_tests:
        name = spec["path"]
        require(
            type(name) is str
            and name.startswith(BASE + "/parent-e354-root-")
            and name.endswith(".py")
            and name not in registered,
            "controlled ROOT CPU test registration",
        )
        require(
            sha(reader.bytes(spec)) == spec["sha256"],
            "registered current extra test bytes",
        )
        registered[name] = spec["sha256"]
    require(
        set(tail[1:]) == set(registered),
        "only registered complete extra tests; no selection switches",
    )
    expected = {OWNED[1]: own[OWNED[1]], **registered}
    require(
        all(
            contract["source_sha256"].get(name) == pin for name, pin in expected.items()
        ),
        "actual current pytest test pins",
    )
    return contract


def make_freeze(qualification, root=ROOT, context=None):
    ctx = context or metadata_preflight(root)
    r = ctx["reader"]
    spec = descriptor(qualification, helper().safe(qualification, root).read_bytes())
    q, own = r.json(spec), own_sources(root)
    require(
        q["schema"] == "flylab.m3.e354.root-qualification.v1"
        and q["source_sha256"] == own
        and q["passed"] is True
        and type(q["original_calls"]) is int
        and q["original_calls"] == 0
        and set(q["checks"]) == set(CHECKS)
        and all(q["checks"][k] is True for k in CHECKS),
        "current ROOT qualification/checks",
    )
    check_execution(r, q["execution"], own, 20)
    primary_cpu_role(
        r, q["execution"], own, root=root, extra_cpu_tests=q.get("extra_cpu_tests", ())
    )
    return {
        "schema": "flylab.m3.e354.teacher-freeze.v1",
        "source_sha256": own,
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "qualification": spec,
        "design": ctx["registration"]["design"],
        "consumed_sources": ctx["registration"]["consumed_sources"],
        "original_authorized": False,
    }


def original_argv(mode, freeze, review):
    return [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--" + mode,
        "--freeze",
        freeze,
        "--review",
        review,
        "--output",
        OUTPUT if mode == "run" else AUDIT_OUTPUT,
    ]


def gate(mode, freeze_name, review_name, *, output, root=ROOT, reader=None):
    require(mode in ("run", "audit"), "own mode")
    ctx = metadata_preflight(root, reader)
    r, own = ctx["reader"], own_sources(root)

    def read(name):
        file = helper().safe(name, root)
        return r.json({"path": name, "bytes": file.stat().st_size, "sha256": None})

    frozen, review = read(freeze_name), read(review_name)
    require(
        frozen == make_freeze(frozen["qualification"]["path"], root, ctx),
        "current exact freeze",
    )
    target, cap, internal = (OUTPUT, 10, 8) if mode == "run" else (AUDIT_OUTPUT, 5, 4)
    require(
        output == target
        and review["schema"] == "flylab.m3.e354.root-original-review.v1"
        and review["mode"] == mode
        and review["approved"] is True
        and review["source_sha256"] == own
        and review["freeze_sha256"] == r.digests[freeze_name]
        and review["registration_sha256"] == REG_SHA
        and review["reflection_sha256"] == REF_SHA
        and review["output"] == target,
        "current ROOT own review",
    )
    reserve = r.json(review["reservation"])
    expected = {
        "schema": "flylab.m3.e354.root-reservation.v1",
        "mode": mode,
        "cumulative_limit_s": 9900,
        "reserved_s": cap,
        "wall_s": cap,
        "internal_s": internal,
        "original_calls": 1,
        "retry_count": 0,
        "body_calls": 0,
        "neural_calls": 0,
        "CPG_calls": 0,
        "GPU_calls": 0,
        "source_sha256": own,
        "freeze_sha256": r.digests[freeze_name],
        "registration_sha256": REG_SHA,
        "reflection_sha256": REF_SHA,
        "review_path": review_name,
        "output": target,
        "argv": original_argv(mode, freeze_name, review_name),
    }
    require(
        all(
            type(reserve.get(k)) is type(v) and reserve[k] == v
            for k, v in expected.items()
        ),
        "exact reviewed once reservation",
    )
    cumulative = 9868.929474399501
    if mode == "audit":
        seal = r.json(review["worker_seal"])
        require(
            seal["schema"] == "flylab.m3.e354.closed-worker-seal.v1"
            and seal["source_sha256"] == own
            and seal["registration_sha256"] == REG_SHA
            and seal["output"] == OUTPUT
            and type(seal["expected_exit_code"]) is int
            and seal["expected_exit_code"] == 0,
            "completed worker exit0 before array gate",
        )
        full = check_execution(
            r,
            seal["execution"],
            own,
            10,
            original=1,
            argv=original_argv("run", seal["freeze_path"], seal["review_path"]),
        )
        cumulative += full["charged_s"]
        ctx["worker_seal"] = seal
    require(
        type(reserve["cumulative_before_s"]) is float
        and reserve["cumulative_before_s"] == cumulative
        and cumulative + cap <= 9900,
        "current charged cumulative",
    )
    require(
        not helper().safe(target, root).exists()
        and shutil.disk_usage(root).free >= 512 << 20
        and all(
            os.environ.get(k) == "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "fresh output/disk/CPU1",
    )
    ctx.update(authorized=True, mode=mode)
    r.array_authorized = True
    return ctx


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--review")
    parser.add_argument("--output")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    a = parser.parse_args(argv)
    mode = "run" if a.run else "audit" if a.audit else None
    if mode:
        require(
            list(sys.argv[1:] if argv is None else argv)
            == original_argv(mode, a.freeze, a.review)[5:],
            "exact sole CLI",
        )

    def timely():
        require(
            time.perf_counter() - ENTRY <= (8 if mode == "run" else 4),
            "inclusive entry through final close deadline",
        )
        import psutil

        require(psutil.Process().memory_info().rss <= 2 << 30, "host2GiB")

    receipt, output, stage = None, None, "gate"
    progress = {}
    try:
        r = helper().EvidenceReader(timely=timely)
        if mode is None:
            require(a.qualification is not None, "passive qualification")
            value = make_freeze(a.qualification, context=metadata_preflight(reader=r))
            write_raw(helper().safe(a.freeze), json_bytes(value), timely=timely)
            return 0
        ctx = gate(mode, a.freeze, a.review, output=a.output, reader=r)
        output = helper().safe(a.output)
        stage = "input_loading"
        data = load_inputs(ctx)
        output.mkdir()
        if mode == "audit":
            stage = "independent_audit"
            report = audit_worker(ctx, data)
            payload = json_bytes(report)
            require(
                len(payload) + EMERGENCY <= CAP,
                "audit/emergency4MiB before publication",
            )
            write_raw(output / "report.json", payload, timely=timely)
            require(helper().usage(output) <= CAP, "closed audit4MiB")
            timely()
            return 0
        stage = "q_only_acquisition"
        raw, receipt = produce(data, timely=timely)
        stage = "publication"
        result, code = publish_worker(
            output, data, raw, receipt, timely=timely, progress=progress
        )
        if code:
            receipt = dict(receipt)
            receipt["first_error"] = receipt.get("first_error") or result["first_error"]
            receipt["secondary_errors"] = result["secondary_errors"]
            print(json_bytes(result).decode(), file=sys.stderr)
        stage = "main_finalization"
        timely()
        return code
    except Exception as error:  # noqa: BLE001 -- strict error receipt, no saved success promotion
        if receipt is None:
            receipt = {
                "stage": stage,
                "first_error": None,
                "secondary_errors": [],
                "counter_status": "not_instrumented_before_q_acquisition",
                "acquired": [],
                "missing": list(raw_roster()),
            }
        report = {
            "schema": "flylab.m3.e354.teacher-failure.v1",
            "status": "technical_invalid",
            "stage": stage,
            "first_error": receipt.get("first_error") or error_record(stage, error),
            "secondary_errors": (
                receipt.get("secondary_errors", []) + [error_record(stage, error)]
            )[:16],
            "receipt": receipt,
            "late_error_has_precedence": True,
            "H354": "not_evaluated",
            **DENIED,
        }
        if output is not None and output.exists():
            try:
                report = failure(output, receipt, error, stage, progress)
            except Exception as secondary:  # noqa: BLE001 -- preserve emergency sink error and actual receipt
                report["secondary_errors"] = (
                    report["secondary_errors"]
                    + [error_record("failure_sink", secondary)]
                )[:16]
        print(json_bytes(report).decode(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
