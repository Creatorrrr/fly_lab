"""E203 one bounded conditional MN response fit; never a whole-BANC model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_local_response import CUDA_SOURCE, validate_inputs
from tools.verify_c_banc_m3_local_response import (
    LEGS,
    file_sha,
    load_original,
    repository_path,
    write_json,
)

FREEZE_SCHEMA = "flylab.m3.e203.local-response-fit.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e203.local-response-fit.review.v1"
OWNED = ("tools/fit_c_banc_m3_local_response.py",
         "tests/test_c_banc_local_response_fit.py")
FREEZE_SHA = "a069feb1f6b740acb5eab10828ddaaafe616b3ec0111387e5ed7a0bea8b239c7"
MAX_EVALUATIONS = 60
RESPONSE_KEYS = ("rate", "adaptation", "rate_sensitivity",
                 "adaptation_sensitivity")


def validate_response(response, rows, motors):
    """Reject any incomplete/nonfinite device readback before it enters a fit."""
    expected = {"rate": (rows, motors), "adaptation": (rows, motors),
        "rate_sensitivity": (rows, motors, 7),
        "adaptation_sensitivity": (rows, motors, 7)}
    if not isinstance(response, dict):
        raise TypeError("E203 response is not a mapping")
    for key in RESPONSE_KEYS:
        value = response.get(key)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != expected[key] or not np.isfinite(value).all()):
            raise ValueError(f"E203 invalid float64 finite response: {key}")


def output_manifest(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_file() and path.name not in {"result.json", "output-manifest.json"}:
            files[path.relative_to(out).as_posix()] = {
                "sha256": file_sha(path), "bytes": path.stat().st_size}
    result = {"schema": "flylab.m3.e203.output-manifest.v1", "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(Path(out) / "output-manifest.json", result)
    return result


class EvaluationLimitReached(RuntimeError):
    """Finite bounded search exhausted its authorized actual CUDA evaluations."""


class PersistentLocalResponse:
    """One native-current upload and one reusable E202 CUDA buffer set."""

    def __init__(self, current, parameters, *, sample_every=20):
        import cupy as cp

        current, p, zero, steps = validate_inputs(current, parameters,
            np.zeros((np.asarray(current).shape[1], 7), np.float64),
            sample_every=sample_every)
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("Actual CUDA required; no CPU fallback")
        n = current.shape[1]
        rows = steps // sample_every + 1
        required = current.nbytes + 16 * rows * n * 8 + 15 * n * 8
        if required > 768 * 1024**2:
            raise ValueError("E203 GPU buffer exceeds 768 MiB")
        compile_began = time.perf_counter()
        self.kernel = cp.RawKernel(CUDA_SOURCE, "local_response",
            options=("--std=c++11", "--fmad=false"))
        self.kernel.compile()
        compile_s = time.perf_counter() - compile_began
        transfer_began = time.perf_counter()
        self.current = cp.asarray(current)
        self.parameters = {key: cp.asarray(value) for key, value in p.items()}
        self.weights = cp.asarray(zero)
        self.rate = cp.empty((rows, n), cp.float64)
        self.adaptation = cp.empty_like(self.rate)
        self.rate_sensitivity = cp.empty((rows, n, 7), cp.float64)
        self.adaptation_sensitivity = cp.empty_like(self.rate_sensitivity)
        self.cp = cp
        self.n, self.steps, self.rows, self.sample_every = n, steps, rows, sample_every
        self.stats = {"device_setup_s": time.perf_counter() - transfer_began,
            "kernel_compile_s": compile_s, "gpu_buffer_bytes": required,
            "current_uploads": 1, "parameter_uploads": 8,
            "weight_uploads": 0, "generated_warmup_calls": 0,
            "original_warmup_calls": 0, "launch_requested": 0,
            "launch_synchronized": 0, "launch_completion_unknown": 0,
            "conditional_ticks_completed": 0, "stage_inputs_consumed": 0,
            "readback_completed": 0, "kernel_main_s": 0.,
            "readback_s": 0.}

    def evaluate(self, weights):
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if (w.shape != (self.n, 7) or not np.isfinite(w).all()
                or np.any(np.abs(w) > 1)):
            raise ValueError("Finite bounded E203 full MN weight matrix required")
        cp, p = self.cp, self.parameters
        self.weights.set(w)
        self.stats["weight_uploads"] += 1
        self.stats["launch_requested"] += 1
        began = time.perf_counter()
        try:
            self.kernel(((self.n + 127) // 128,), (128,), (
                self.current, p["tau"], p["threshold"], p["cap"],
                p["coefficient"], p["adaptation_gain"],
                p["inverse_adaptation_tau"], p["initial_rate"],
                p["initial_adaptation"], self.weights, self.rate,
                self.adaptation, self.rate_sensitivity,
                self.adaptation_sensitivity, np.int32(self.n),
                np.int32(self.steps), np.int32(self.sample_every),
                np.float64(.00025)))
            cp.cuda.get_current_stream().synchronize()
        except BaseException:
            self.stats["launch_completion_unknown"] += 1
            raise
        self.stats["launch_synchronized"] += 1
        self.stats["conditional_ticks_completed"] += self.steps
        self.stats["stage_inputs_consumed"] += 4 * self.steps
        self.stats["kernel_main_s"] += time.perf_counter() - began
        read_began = time.perf_counter()
        result = {"rate": cp.asnumpy(self.rate),
            "adaptation": cp.asnumpy(self.adaptation),
            "rate_sensitivity": cp.asnumpy(self.rate_sensitivity),
            "adaptation_sensitivity": cp.asnumpy(self.adaptation_sensitivity)}
        self.stats["readback_s"] += time.perf_counter() - read_began
        self.stats["readback_completed"] += 1
        return result


def _ema(values, alpha, initial):
    """Time-vectorized exact affine EMA with immutable row-zero state."""
    values = np.asarray(values, np.float64)
    initial = np.asarray(initial, np.float64)
    rows = np.empty((len(values) + 1, *initial.shape), np.float64)
    rows[0] = initial
    beta = 1 - alpha
    rows[1:], _ = lfilter([alpha], [1, -beta], values, axis=0,
                          zi=(beta * initial)[None, ...])
    return rows


def _ema_adjoint(output_gradient, alpha):
    """Reverse y[t]=beta*y[t-1]+alpha*x[t], with fixed y[0]."""
    g = np.asarray(output_gradient, np.float64)
    result = np.zeros_like(g)
    carry = np.zeros_like(g[0])
    beta = 1 - alpha
    for tick in range(len(g) - 1, 0, -1):
        total = g[tick] + carry
        result[tick] = alpha * total
        carry = beta * total
    return result


class DecoderAdjoint:
    """Vectorized E150 anatomical+monotone decoder and local reverse pass."""

    def __init__(self, start):
        a = start["adapter"]
        c, p = a["calibration"], a["parameters"]
        self.pool = np.asarray(c["pool"], np.float64)
        self.moment = np.asarray(c["moment"], np.float64)
        self.rest = np.asarray(c["rest"], np.float64)
        self.ltm = np.asarray(c["ltm"], np.int64)
        self.membership = np.asarray(p["membership"], np.int64)
        self.kappa = np.asarray(p["kappa"], np.float64)
        self.slopes = np.asarray(p["slopes"], np.float64)
        self.fixed = np.asarray(p["fixed_muscles"], np.int64)
        self.variable = np.setdiff1d(np.arange(96), self.fixed)
        self.bank0 = np.asarray(a["bank"], np.float64).copy()
        self.original0 = np.asarray(a["original"]["activation"], np.float64).copy()
        self.activation0 = np.asarray(a["activation"], np.float64).copy()
        self.held_command = np.asarray(start["body"]["action"]["joint_angles"], np.float64)
        self.bank_alpha = -np.expm1(-.005 / np.asarray(p["tau"], np.float64))
        self.original_alpha = -np.expm1(-.005 / .01)
        self.groups = a["groups"]
        self.tendon_names = [self.groups[int(index)]["leg"] + "_tarsus"
                             for index in self.ltm]
        if (self.pool.shape != (96, 391) or self.moment.shape != (42, 96)
                or self.rest.shape != (42,) or self.ltm.shape != (6,)
                or self.membership.shape != (391,)
                or self.slopes.shape != (96, 391, 3, 2)
                or self.bank0.shape != (391, 3)
                or self.original0.shape != (96,)
                or self.activation0.shape != (96,)
                or self.held_command.shape != (42,)
                or set(self.tendon_names) != {leg + "_tarsus" for leg in LEGS}):
            raise ValueError("Complete frozen E150 decoder state required")
        self.tendon_order = [self.tendon_names.index(leg + "_tarsus") for leg in LEGS]

    def forward(self, rates):
        rates = np.asarray(rates, np.float64)
        if rates.ndim != 2 or rates.shape[1] != 391 or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Finite nonnegative local MN rates required")
        bank = np.empty((len(rates), 391, 3), np.float64)
        for k, alpha in enumerate(self.bank_alpha):
            bank[:, :, k] = _ema(rates[1:], alpha, self.bank0[:, k])
        pooled = rates[1:] @ self.pool.T
        original = _ema(pooled, self.original_alpha, self.original0)
        activation = np.empty((len(rates), 96), np.float64)
        activation[:, self.fixed] = original[:, self.fixed]
        activation[0] = self.activation0
        hinge = np.maximum(bank - self.kappa[None, :, None], 0.)
        for muscle in self.variable:
            members = np.flatnonzero(self.membership == muscle)
            low = self.slopes[muscle, members, :, 0]
            delta = self.slopes[muscle, members, :, 1] - low
            activation[1:, muscle] = (np.sum(low[None] * bank[1:, members], axis=(1, 2))
                + np.sum(delta[None] * hinge[1:, members], axis=(1, 2)))
        command = self.rest + .02 * (activation @ self.moment.T)
        command[0] = self.held_command
        tendon = .2 * np.tanh(activation[:, self.ltm] / 10)
        tendon = tendon[:, self.tendon_order]
        if not all(np.isfinite(v).all() for v in (bank, activation, command, tendon)):
            raise ValueError("Nonfinite E150 decoder output")
        return {"bank": bank, "activation": activation, "command": command,
                "tendon": tendon}

    def backward(self, command_gradient, decoded):
        gq = np.asarray(command_gradient, np.float64)
        if gq.shape != decoded["command"].shape:
            raise ValueError("Aligned command gradient required")
        gactive = .02 * (gq @ self.moment)
        gbank = np.zeros_like(decoded["bank"])
        for muscle in self.variable:
            members = np.flatnonzero(self.membership == muscle)
            low = self.slopes[muscle, members, :, 0]
            high = self.slopes[muscle, members, :, 1]
            slope = np.where(decoded["bank"][1:, members] > self.kappa[members][None, :, None],
                             high[None], low[None])
            gbank[1:, members] += gactive[1:, muscle, None, None] * slope
        grate = np.zeros((len(gq), 391), np.float64)
        for k, alpha in enumerate(self.bank_alpha):
            grate += _ema_adjoint(gbank[:, :, k], alpha)
        goriginal = np.zeros((len(gq), 96), np.float64)
        goriginal[:, self.fixed] = gactive[:, self.fixed]
        grate += _ema_adjoint(goriginal, self.original_alpha) @ self.pool
        grate[0] = 0
        return grate


def _rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, np.float64)**2)))


def objective_and_command_gradient(decoded, target_data, free_weights,
                                   *, penalty=10000., inner=.995):
    """Frozen normalized objective, fourteen guards, sample guards, and dq."""
    q = decoded["command"]
    indices = target_data["indices"]
    starts, ends = target_data["pair_start"], target_data["pair_end"]
    target, velocity = target_data["target"], target_data["velocity"]
    startup = target_data["startup_command"]
    eq = q[indices] - target
    ev = (q[ends] - q[starts]) / .005 - velocity
    es = q[1:100] - startup
    baseline = target_data["baseline"]
    denom = (.8 * baseline["command_rms"])**2
    if denom <= 0:
        raise ValueError("Positive original command baseline required")
    merit = float(np.mean(eq * eq) / denom)
    gq = np.zeros_like(q)
    gq[indices] += 2 * eq / (eq.size * denom)
    terms = {"main_command": merit}

    def guard(name, residual, limit, scatter):
        nonlocal merit
        if limit <= 0:
            raise ValueError("Positive original protection limit required: " + name)
        scale = (inner * limit)**2
        a = float(np.mean(residual * residual) / scale)
        excess = max(a - 1., 0.)
        value = penalty * excess**2
        merit += value
        terms[name] = {"A": a, "penalty": value,
                       "rms": _rms(residual), "limit": limit}
        if excess:
            scatter(4 * penalty * excess * residual / (residual.size * scale))

    def scatter_velocity(gradient, columns):
        np.add.at(gq, (ends[:, None], columns[None, :]), gradient / .005)
        np.add.at(gq, (starts[:, None], columns[None, :]), -gradient / .005)

    all_columns = np.arange(42)
    guard("overall_velocity", ev, baseline["velocity_rms"],
          lambda grad: scatter_velocity(grad, all_columns))
    for leg in range(6):
        columns = np.arange(7 * leg, 7 * leg + 7)
        guard(f"{LEGS[leg]}_command", eq[:, columns],
              baseline["leg_command_rms"][leg],
              lambda grad, cols=columns: np.add.at(
                  gq, (indices[:, None], cols[None, :]), grad))
        guard(f"{LEGS[leg]}_velocity", ev[:, columns],
              baseline["leg_velocity_rms"][leg],
              lambda grad, cols=columns: scatter_velocity(grad, cols))
    guard("startup_rms", es, .018,
          lambda grad: gq.__setitem__(slice(1, 100), gq[1:100] + grad))
    sample_scale = (inner * .04)**2
    b = es * es / sample_scale
    excess = np.maximum(b - 1., 0.)
    sample_penalty = float(penalty * np.sum(excess * excess))
    merit += sample_penalty
    terms["startup_sample_penalty"] = sample_penalty
    gq[1:100] += 4 * penalty * excess * es / sample_scale
    regularization = 1e-4 * float(np.mean(free_weights * free_weights))
    merit += regularization
    terms["regularization"] = regularization
    terms["total"] = merit
    if not np.isfinite(merit) or not np.isfinite(gq).all():
        raise ValueError("Nonfinite E203 objective or command gradient")
    return merit, gq, terms


def direct_metrics(decoded, target_data):
    q = decoded["command"]
    i = target_data["indices"]
    a, b = target_data["pair_start"], target_data["pair_end"]
    eq = q[i] - target_data["target"]
    ev = (q[b] - q[a]) / .005 - target_data["velocity"]
    es = q[1:100] - target_data["startup_command"]
    et = decoded["tendon"][1:100] - target_data["startup_tendon"]
    return {"command_rms": _rms(eq), "velocity_rms": _rms(ev),
        "leg_command_rms": [_rms(eq[:, 7*j:7*j+7]) for j in range(6)],
        "leg_velocity_rms": [_rms(ev[:, 7*j:7*j+7]) for j in range(6)],
        "startup_q_rms": _rms(es), "startup_q_max": float(np.max(np.abs(es))),
        "startup_tendon_rms": _rms(et),
        "startup_tendon_max": float(np.max(np.abs(et)))}


def feasible(metrics, baseline):
    return bool(metrics["command_rms"] <= .8 * baseline["command_rms"]
        and metrics["velocity_rms"] <= baseline["velocity_rms"]
        and all(a <= b for a, b in zip(metrics["leg_command_rms"],
                                       baseline["leg_command_rms"]))
        and all(a <= b for a, b in zip(metrics["leg_velocity_rms"],
                                       baseline["leg_velocity_rms"]))
        and metrics["startup_q_rms"] <= .018
        and metrics["startup_q_max"] <= .04
        and metrics["startup_tendon_rms"] <= .009
        and metrics["startup_tendon_max"] <= .025)


def load_targets(path, trace):
    with np.load(path, allow_pickle=False) as data:
        selected = {key: data[key].copy() for key in (
            "target_indices", "target", "target_velocity", "recorded_command",
            "loaded_target_mask_pair_start", "loaded_target_mask_pair_end",
            "loaded_startup_target_command", "loaded_startup_target_tendon")}
    indices = selected["target_indices"].astype(np.int64)
    starts = selected["loaded_target_mask_pair_start"].astype(np.int64)
    ends = selected["loaded_target_mask_pair_end"].astype(np.int64)
    recorded = selected["recorded_command"]
    if (indices.shape != (1824,) or not np.array_equal(indices, np.arange(100, 1924))
            or starts.shape != (1823,) or ends.shape != (1823,)
            or not np.array_equal(starts, indices[:-1])
            or not np.array_equal(ends, indices[1:])
            or selected["target"].shape != (1824, 42)
            or selected["target_velocity"].shape != (1823, 42)
            or selected["loaded_startup_target_command"].shape != (99, 42)
            or selected["loaded_startup_target_tendon"].shape != (99, 6)
            or recorded.shape != (2001, 42)):
        raise ValueError("Exact E197 target/startup shapes and clocks required")
    actual = np.asarray([row["joint_targets_rad"] for row in trace["samples"]])
    if not np.array_equal(recorded, actual):
        raise ValueError("E197 recorded command differs from frozen E150")
    target = {"indices": indices, "pair_start": starts, "pair_end": ends,
        "target": selected["target"], "velocity": selected["target_velocity"],
        "startup_command": selected["loaded_startup_target_command"],
        "startup_tendon": selected["loaded_startup_target_tendon"]}
    eq = recorded[indices] - target["target"]
    ev = (recorded[ends] - recorded[starts]) / .005 - target["velocity"]
    target["baseline"] = {"command_rms": _rms(eq),
        "velocity_rms": _rms(ev),
        "leg_command_rms": [_rms(eq[:, 7*j:7*j+7]) for j in range(6)],
        "leg_velocity_rms": [_rms(ev[:, 7*j:7*j+7]) for j in range(6)]}
    return target


def verify_gate(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or file_sha(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E203 SHA changed")
    if file_sha(review_path) != review_sha:
        raise ValueError("E203 review SHA changed")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design") != "docs/BANC_M3_E203_LOCAL_RESPONSE_FIT_DESIGN_20260924.md"
            or file_sha(repository_path(freeze["design"])) != freeze.get("design_sha256")
            or file_sha(repository_path(freeze["base_freeze"])) != freeze.get("base_freeze_sha256")
            or freeze.get("base_freeze_sha256") != "2771f9b21aae69f4f8a14252d7441c9842c3ec698e1d5e6f507d35ab8ea3f20c"
            or freeze.get("solver", {}).get("hard_cuda_evaluations") != 60
            or freeze.get("resource_limits", {}).get("internal_s") != 270
            or freeze.get("F6_exposed") is not False
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("E203 frozen contract changed")
    for label, count in (("input_sha256", 97),
                         ("reused_source_sha256", 8),
                         ("preservation_sha256", 1)):
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError("Frozen E203 hash count changed")
        for name, expected in hashes.items():
            if file_sha(repository_path(name)) != expected:
                raise ValueError("E203 frozen input/source changed: " + name)
    own = {name: file_sha(repository_path(name)) for name in OWNED}
    extra = review.get("additional_reused_source_sha256")
    if extra is not None and (not isinstance(extra, dict) or any(
            file_sha(repository_path(name)) != sha for name, sha in extra.items())):
        raise ValueError("Additional E203 reviewed source changed")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != own):
        raise ValueError("E203 parent source review does not authorize these files")
    return freeze, review


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("One-shot E203 output exists: " + str(out))
    out.mkdir(parents=True)
    ledger = {"schema": "flylab.m3.e203.fit-ledger.v1", "optimizer_calls": 0,
        "cuda_evaluations": 0, "cache_hits": 0, "new_neural_ticks": 0,
        "new_physics_steps": 0, "generated_warmup_calls": 0,
        "original_warmup_calls": 0, "optimizer_requests": [], "errors": []}
    result = {"schema": "flylab.m3.e203.fit-result.v1",
        "validity": "incomplete_or_invalid", "candidate_adopted": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    freeze = None
    cache = {}
    evaluations = []
    best = None
    device = None
    parameter = None
    target = None

    def elapsed():
        return time.perf_counter() - began

    def attempt(label, operation):
        nonlocal primary
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - retain partial raw
            ledger["errors"].append({"stage": label, "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc()})
            if primary is None:
                primary = error
            return None

    try:
        freeze, review = verify_gate(freeze_path, freeze_sha,
                                     review_path, review_sha)
        if shutil.disk_usage(out).free < 4 * 1024**3:
            raise OSError("E203 requires 4 GiB free disk")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            paths = {key: repository_path(value, file=key != "start")
                     for key, value in freeze["paths"].items()}
            write_json(out / "provenance.json", {"freeze_sha256": FREEZE_SHA,
                "review_sha256": review_sha, "source_sha256": review["source_sha256"],
                "input_sha256": freeze["input_sha256"]})
            current, parameter, _endpoints, start, trace, _native = load_original(paths)
            with np.load(paths["mask"], allow_pickle=False) as data:
                mask = {key: data[key].copy() for key in data.files}
            free = np.asarray(mask["free_columns"], np.int64)
            fixed = np.asarray(mask["fixed_columns"], np.int64)
            membership = np.asarray(parameter["membership"], np.int64)
            ltm = np.asarray(start["adapter"]["calibration"]["ltm"], np.int64)
            expected_fixed = np.flatnonzero(np.isin(membership, ltm))
            if (free.shape != (343,) or fixed.shape != (48,)
                    or not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(391))
                    or not np.array_equal(free, np.sort(free))
                    or not np.array_equal(fixed, expected_fixed)
                    or not np.array_equal(mask["trainable_mask"],
                                          np.isin(np.arange(391), free))
                    or not np.array_equal(mask["membership"], membership)
                    or not np.array_equal(mask["ltm"], ltm)
                    or not np.array_equal(mask["motor_ids"], parameter["motor_ids"])
                    or mask["motor_roster"].tolist() !=
                       start["monotone_recruitment_patch"]["motor_roster"]):
                raise ValueError("Frozen E203 anatomical free/fixed MN mask changed")
            target = load_targets(paths["targets"], trace)
            decoder = DecoderAdjoint(start)
            p = {key: parameter[key] for key in (
                "initial_rate", "initial_adaptation", "tau", "threshold",
                "cap", "coefficient", "adaptation_gain",
                "inverse_adaptation_tau")}
            device = PersistentLocalResponse(current, p)
            with np.load(paths["e202_response"], allow_pickle=False) as data:
                e202_response = {key: data[key].copy() for key in (
                    "rate", "adaptation", "rate_sensitivity",
                    "adaptation_sensitivity")}
            with np.load(paths["e202_decoded"], allow_pickle=False) as data:
                e202_decoded = {key: data[key].copy() for key in (
                    "bank", "activation", "command", "tendon")}

            def evaluate(free_vector):
                nonlocal best
                values = np.ascontiguousarray(free_vector, dtype=np.float64)
                if (values.shape != (2401,) or not np.isfinite(values).all()
                        or np.any(np.abs(values) > 1)):
                    raise ValueError("Finite bounded E203 free weights required")
                key = values.tobytes()
                request = {"order": len(ledger["optimizer_requests"]),
                    "wall_s": elapsed(), "cache_hit": key in cache,
                    "weight_sha256": hashlib.sha256(key).hexdigest()}
                ledger["optimizer_requests"].append(request)
                if key in cache:
                    ledger["cache_hits"] += 1
                    write_json(out / f"cache-hit-{request['order']:03}.json", request)
                    return cache[key]
                if elapsed() >= 240:
                    raise TimeoutError("E203 fit 240 s deadline exceeded")
                if len(evaluations) >= MAX_EVALUATIONS:
                    raise EvaluationLimitReached("E203 hard 60 CUDA evaluations reached")
                weights = np.zeros((391, 7), np.float64)
                weights[free] = values.reshape(343, 7)
                response = device.evaluate(weights)
                ledger["cuda_evaluations"] = device.stats["launch_synchronized"]
                try:
                    validate_response(response, e202_response["rate"].shape[0], 391)
                except (TypeError, ValueError) as invalid:
                    invalid_index = len(evaluations)
                    try:
                        raw = {key: value for key, value in response.items()
                               if isinstance(value, np.ndarray)}
                        np.savez_compressed(out / f"invalid-response-{invalid_index:03}.npz",
                                            weights=weights, **raw)
                        write_json(out / f"invalid-response-{invalid_index:03}.json",
                            {"validity": "incomplete_or_invalid",
                             "reason": str(invalid), "wall_s": elapsed()})
                    except BaseException as storage_error:  # noqa: BLE001
                        ledger["errors"].append({"stage": "invalid-response-raw",
                            "type": type(storage_error).__name__,
                            "message": str(storage_error)})
                    raise
                decoded = decoder.forward(response["rate"])
                merit, gq, terms = objective_and_command_gradient(
                    decoded, target, values)
                g_rate = decoder.backward(gq, decoded)
                gradient_all = np.einsum("tm,tmk->mk", g_rate,
                                         response["rate_sensitivity"], optimize=True)
                gradient = gradient_all[free].reshape(-1)
                gradient += 2e-4 * values / len(values)
                if not np.isfinite(gradient).all():
                    raise ValueError("Nonfinite E203 analytic gradient")
                metrics = direct_metrics(decoded, target)
                accepted = feasible(metrics, target["baseline"])
                serial = len(evaluations)
                record = {"index": serial, "merit": merit, "terms": terms,
                    "metrics": metrics, "feasible": accepted,
                    "wall_s": elapsed(), "cache_hit": False,
                    "weight_sha256": hashlib.sha256(key).hexdigest()}
                np.savez_compressed(out / f"evaluation-{serial:03}.npz",
                    weights=weights, free_weights=values, gradient=gradient,
                    command=decoded["command"])
                write_json(out / f"evaluation-{serial:03}.json", record)
                evaluations.append(record)
                cache[key] = (merit, gradient.copy())
                choose = (best is None
                    or accepted and not best["record"]["feasible"]
                    or accepted and best["record"]["feasible"]
                       and metrics["command_rms"] < best["record"]["metrics"]["command_rms"]
                    or not accepted and not best["record"]["feasible"]
                       and merit < best["record"]["merit"])
                if choose:
                    best = {"record": record, "response": response,
                            "decoded": decoded, "weights": weights,
                            "gradient": gradient.copy()}
                return cache[key]

            zero = np.zeros(2401, np.float64)
            evaluate(zero)
            first = best
            for key, original in e202_response.items():
                actual = first["response"][key]
                if (actual.dtype != original.dtype or actual.shape != original.shape
                        or actual.tobytes() != original.tobytes()):
                    raise ValueError("E203 w0 failed bit-exact E202 response: " + key)
            for key, original in e202_decoded.items():
                np.testing.assert_allclose(first["decoded"][key], original,
                    atol=2e-12, rtol=1e-12,
                    err_msg="E203 w0 decoder mismatch: " + key)
            ledger["w0_reconstruction_verified"] = True
            ledger["optimizer_calls"] = 1
            try:
                optimized = minimize(evaluate, zero, jac=True,
                    method="L-BFGS-B", bounds=[(-1., 1.)] * 2401,
                    options={"maxiter": 40, "maxfun": 60, "maxls": 10,
                             "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6})
                result["optimizer"] = {"success": bool(optimized.success),
                    "status": int(optimized.status), "message": str(optimized.message),
                    "nit": int(optimized.nit), "nfev": int(optimized.nfev)}
            except EvaluationLimitReached as bounded:
                result["optimizer"] = {"success": False,
                    "status": "hard_cuda_evaluation_limit", "message": str(bounded)}
            if elapsed() >= 240:
                raise TimeoutError("E203 fit 240 s deadline exceeded")
            if best is None:
                raise ValueError("No finite E203 evaluation selected")
            result["validity"] = ("supported_conditional_local_response_proposal"
                if best["record"]["feasible"] else "valid_bounded_search_no_proposal")
            result["selected_evaluation"] = best["record"]["index"]
            result["selected_metrics"] = best["record"]["metrics"]
            result["selected_merit"] = best["record"]["merit"]
    except BaseException as error:  # noqa: BLE001 - raw outlives failure
        primary = error
        ledger["errors"].append({"stage": "run", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    if best is not None:
        attempt("selected-state.npz", lambda: np.savez_compressed(
            out / "selected-state.npz", **best["response"],
            weights=best["weights"], motor_ids=parameter["motor_ids"]))
        attempt("selected-decoder.npz", lambda: np.savez_compressed(
            out / "selected-decoder.npz", **best["decoded"]))
        attempt("selection.json", lambda: write_json(out / "selection.json", {
            "selected": best["record"], "baseline": target["baseline"],
            "evaluation_count": len(evaluations),
            "status": ("verified_candidate" if primary is None
                       and ledger.get("w0_reconstruction_verified")
                       else "unverified_incomplete_raw"),
            "w0_reconstruction_verified": bool(
                ledger.get("w0_reconstruction_verified", False))}))
    if primary is not None:
        result["validity"] = "incomplete_or_invalid"
    if device is not None:
        ledger["cuda"] = device.stats
        ledger["cuda_evaluations"] = device.stats["launch_synchronized"]
    ledger["evaluation_count"] = len(evaluations)
    ledger["process_wall_s"] = elapsed()
    result["process_wall_s"] = elapsed()
    result["prior_cumulative_s"] = (freeze or {}).get("prior_cumulative_s")
    result["conditional_evaluations"] = ledger["cuda_evaluations"]
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    attempt("ledger.json", lambda: write_json(out / "ledger.json", ledger))
    preliminary = {**result, "validity": "incomplete_or_invalid",
                   "finalization_pending": True}
    attempt("result.json-preliminary", lambda: write_json(out / "result.json", preliminary))
    manifest = attempt("output-manifest.json", lambda: output_manifest(out))
    if manifest is None or manifest["total_bytes"] > 256 * 1024**2:
        primary = primary or RuntimeError("E203 output manifest/256 MiB limit failed")
        result["validity"] = "incomplete_or_invalid"
    if elapsed() >= 270:
        primary = primary or TimeoutError("E203 internal 270 s deadline exceeded")
        result["validity"] = "incomplete_or_invalid"
    if primary is None:
        result["finalization_pending"] = False
        attempt("result.json-final", lambda: write_json(out / "result.json", result))
    if primary is not None:
        attempt("publication-failure.json", lambda: write_json(
            out / "publication-failure.json", ledger["errors"]))
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha, args.review, args.review_sha, args.output)


if __name__ == "__main__":
    main()
