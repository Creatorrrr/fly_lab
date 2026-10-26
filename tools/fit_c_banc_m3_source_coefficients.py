"""E270 one-pass, source-only fit. Original execution requires a root freeze.

Freeze v1: schema, approved_sources (the four reviewed files and SHA256),
inputs {name:{path,sha256}}, contract (exact CONTRACT), contexts
{startup,teacher}:{mapping:{engine_key:{input,key?}}, reference:{rate,
adaptation,q,ltm,activation,bank}}, targets:{startup_q,startup_ltm,teacher_q},
legs:{motor,muscle,edge,q,ltm}, identity:{motors,sources,edges,muscles,joints},
native_validation:{input}, generated_cuda_gate:{input}.
Every array reference addresses an explicitly hashed .npy or .npz input.
The two gates are hashed JSON inputs, whose status must be 'pass'. Native
validation must assert counters_valid, raw_valid, identity_valid and decoder_valid.
Source review v1: schema, verdict='approved', approved_sources, freeze_sha256.
No original data is touched by importing these pure CPU helpers.
"""

from __future__ import annotations

# Errors are persisted at the CLI boundary without retrying original work.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import hashlib
import json
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e270.bounded-source-fit.v1"
REVIEW_SCHEMA = "flylab.m3.e270.source-review.v1"
SOURCES = (
    "tools/fit_c_banc_m3_source_coefficients.py",
    "tests/test_c_banc_source_coefficients_fit.py",
    "tools/banc_m3_source_coefficient_engine.py",
    "tests/test_c_banc_source_coefficient_engine.py",
)
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
CONTRACT = {
    "leg_order": list(LEGS),
    "maxiter": 60,
    "ftol": 1e-9,
    "max_unique_evaluations": 80,
    "fit_wall_s": 180.0,
    "leg_wall_s": 30.0,
    "original_wall_s": 290.0,
    "train_start": 1,
    "train_stop": 1201,
    "development_start": 1201,
    "development_stop": 2001,
    "startup_start": 1,
    "startup_stop": 100,
    "q_rms": 0.018,
    "ltm_rms": 0.009,
    "q_max": 0.04,
    "ltm_max": 0.025,
    "selector_margin": 1e-4,
    "triple_min": 0.125,
    "triple_max": 8.0,
    "bound_margin": 1e-6,
    "regularization": 1e-4,
    "train_ratio": 0.8,
    "development_ratio": 1.0,
    "leg_ratio": 1.05,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(name):
    require(isinstance(name, str), "Text relative path required")
    p = Path(name)
    result = (ROOT / p).resolve()
    require(
        not p.is_absolute()
        and ".." not in p.parts
        and result.is_relative_to(ROOT)
        and result.is_file(),
        "Unsafe/missing input path",
    )
    return result


def verified_json(path, digest):
    require(
        isinstance(digest, str) and len(digest) == 64 and sha256(path) == digest,
        "SHA256 mismatch",
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_freeze(path, digest, review_path, review_digest):
    freeze = verified_json(path, digest)
    review = verified_json(review_path, review_digest)
    require(
        freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT,
        "Frozen schema/contract differs",
    )
    sources = freeze.get("approved_sources")
    require(
        isinstance(sources, dict) and set(sources) == set(SOURCES),
        "Exactly four approved sources required",
    )
    for name, value in sources.items():
        require(sha256(safe_path(name)) == value, "Reviewed source changed: " + name)
    require(
        review.get("schema") == REVIEW_SCHEMA
        and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources
        and review.get("freeze_sha256") == digest,
        "Root source review differs",
    )
    inputs = freeze.get("inputs")
    require(isinstance(inputs, dict) and inputs, "Frozen inputs required")
    for spec in inputs.values():
        require(set(spec) == {"path", "sha256"}, "Input needs path and SHA256")
        require(sha256(safe_path(spec["path"])) == spec["sha256"], "Input SHA changed")
    for name in ("native_validation", "generated_cuda_gate"):
        ref = freeze[name]
        require(
            set(ref) == {"input"} and ref["input"] in inputs, "Hashed gate required"
        )
        gate = json.loads(
            safe_path(inputs[ref["input"]]["path"]).read_text(encoding="utf-8")
        )
        require(gate.get("status") == "pass", name + " did not pass")
        if name == "native_validation":
            require(
                all(
                    gate.get(k) is True
                    for k in (
                        "counters_valid",
                        "raw_valid",
                        "identity_valid",
                        "decoder_valid",
                    )
                ),
                "Native validations missing",
            )
    return freeze


def load_ref(freeze, ref):
    require(
        isinstance(ref, dict) and set(ref) in ({"input"}, {"input", "key"}),
        "Explicit hashed array reference required",
    )
    spec = freeze["inputs"][ref["input"]]
    loaded = np.load(safe_path(spec["path"]), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            require("key" in ref, "NPZ key required")
            return loaded[ref["key"]].copy()
        finally:
            loaded.close()
    require("key" not in ref, "NPY has no key")
    return loaded


def neutral(edges):
    c = np.zeros((edges, 3), np.float64)
    c[:, 0] = 1
    return c


def coefficient_valid(c, *, interior=False):
    c = np.asarray(c)
    lo = 0.125 * (1 + 1e-6) if interior else 0.125
    hi = 8 * (1 - 1e-6) if interior else 8
    return (
        c.dtype == np.float64
        and c.ndim == 2
        and c.shape[1] == 3
        and np.isfinite(c).all()
        and np.all(c >= 0)
        and np.all(c <= 8)
        and np.all(c.sum(1) >= lo)
        and np.all(c.sum(1) <= hi)
    )


def triple_constraint(edges):
    a = np.zeros((edges, edges * 3))
    for i in range(edges):
        a[i, i * 3 : i * 3 + 3] = 1
    return LinearConstraint(a, 0.125 * (1 + 1e-6), 8 * (1 - 1e-6))


class BudgetExceeded(RuntimeError):
    def __init__(self, scope):
        self.scope = scope
        super().__init__(scope + " budget exhausted")


@dataclass
class OriginalDeadline:
    started: float
    clock: object = time.perf_counter

    def check(self):
        if self.clock() - self.started >= 290:
            raise BudgetExceeded("original")

    def receipt(self):
        return {"original_wall_s": self.clock() - self.started, "limit_s": 290}


@dataclass
class Budget:
    clock: object = time.perf_counter
    deadline: OriginalDeadline | None = None
    started: float = field(init=False)
    leg_started: float = field(init=False)
    finished: float | None = None
    unique: int = 0
    total_unique: int = 0
    solver_calls: int = 0
    iterations: int = 0
    cache_hits: int = 0
    failed: int = 0
    incomplete: int = 0
    budget_stops: int = 0
    host_requested: int = 0
    host_completed: int = 0

    def __post_init__(self):
        self.started = self.leg_started = self.clock()

    def start_leg(self):
        self.leg_started = self.clock()
        self.unique = 0
        self.check()

    def check(self, *, new=False):
        if self.deadline is not None:
            self.deadline.check()
        now = self.clock()
        if now - self.started >= 180:
            raise BudgetExceeded("shared_fit")
        if now - self.leg_started >= 30:
            raise BudgetExceeded("leg")
        if new and self.unique >= 80:
            raise BudgetExceeded("unique_evaluations")

    def shared_exhausted(self):
        return (
            self.finished if self.finished is not None else self.clock()
        ) - self.started >= 180

    def stop(self):
        self.finished = self.clock()

    def receipt(self):
        return {
            k: getattr(self, k)
            for k in (
                "unique",
                "total_unique",
                "solver_calls",
                "iterations",
                "cache_hits",
                "failed",
                "incomplete",
                "budget_stops",
                "host_requested",
                "host_completed",
            )
        } | {
            "fit_wall_s": (self.finished if self.finished is not None else self.clock())
            - self.started,
            "leg_wall_s": (self.finished if self.finished is not None else self.clock())
            - self.leg_started,
        }


def training_mapping(mapping):
    """No development stages can be uploaded to an optimization engine."""
    require(len(mapping["stage_total"]) == 160000, "Full teacher context required")
    part = dict(mapping)
    for key in (
        "stage_total",
        "source_rate",
        "source_mask",
        "source_history",
        "stage_index",
    ):
        require(len(mapping[key]) == 160000, "Teacher stage axes differ: " + key)
        part[key] = mapping[key][:96000]
    require(
        np.array_equal(part["stage_index"], np.arange(96000, dtype=np.int64)),
        "Training stage order changed",
    )
    return part


def preserve_primary(error, label, callback):
    """Secondary persistence/cleanup failures must not replace the run error."""
    try:
        return callback()
    except Exception as secondary:
        error.add_note(label + ": " + repr(secondary))
        return None


def failure_record(error, engine, output, phase, cost):
    """Persist the primary error before attempting fallible partial rescue."""
    record = {
        "phase": phase,
        "exception": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "response_receipt": copy.deepcopy(getattr(error, "response_receipt", {})),
        "cost": cost,
    }
    preserve_primary(error, "Primary error receipt could not be saved",
                     lambda: write_json(output / (phase + "-error.json"), record))
    preserve_primary(error, "Failure cost could not be saved",
                     lambda: write_json(output / "cost.json", cost))
    rescue_errors = []
    partial = {}
    for key, value in getattr(error, "partial_response", {}).items():
        if not hasattr(value, "shape"):
            continue
        try:
            partial[key] = (
                value
                if isinstance(value, np.ndarray)
                else engine.host_copy(value, label=phase + ":partial:" + key)
            )
        except Exception as rescue_error:
            rescue_errors.append(
                {
                    "key": key,
                    "error": str(rescue_error),
                    "exception": type(rescue_error).__name__,
                }
            )
    try:
        if partial:
            np.savez(output / (phase + "-partial.npz"), **partial)
    except Exception as rescue_error:
        rescue_errors.append({"key": "save_partial", "error": str(rescue_error)})
    if rescue_errors:
        preserve_primary(error, "Rescue errors could not be saved",
                         lambda: write_json(output / (phase + "-rescue-errors.json"), rescue_errors))


def command_jacobians(result, mapping):
    b, muscle = result["B"], result["parameter_muscle"]
    q = 0.02 * mapping["moment"][:, muscle][None, :, :] * b[:, None, :]
    factor = 0.02 * (1 - np.tanh(result["activation"][:, mapping["ltm"]] / 10) ** 2)
    ltm = (
        factor[:, :, None]
        * (mapping["ltm"][:, None] == muscle[None, :])[None, :, :]
        * b[:, None, :]
    )
    return q, ltm


def mean_square(values):
    """Use one float64 C-order reduction for equal baseline/candidate values."""
    ordered = np.ascontiguousarray(values, dtype=np.float64)
    return float(np.mean(ordered * ordered))


def objective(c, q, jac, target, baseline_mse):
    require(
        np.isfinite(baseline_mse) and baseline_mse > 0, "Positive baseline MSE required"
    )
    error = q - target
    delta = c - neutral(len(c))
    mse = mean_square(error)
    value = mse / baseline_mse + 1e-4 * np.mean(delta**2)
    gradient = 2 * np.einsum("sq,sqp->p", error, jac) / error.size / baseline_mse
    gradient += 2e-4 * delta.ravel() / delta.size
    return float(value), gradient, float(mse)


def startup_constraints(qerr, lerr, qjac, ljac, fixed_sse, *, margin=1e-4):
    """Two global SSE budgets, then seven joint maxima and one LTM maximum.

    q/l error arrays contain only the current leg. fixed_sse contains all
    other legs, and global element counts always remain 99*42 and 99*6.
    """
    values, grads = [], []
    for error, jac, other, count, rms, peak in (
        (qerr, qjac, fixed_sse[0], 99 * 42, 0.018, 0.04),
        (lerr, ljac, fixed_sse[1], 99 * 6, 0.009, 0.025),
    ):
        values.append(count * (rms * (1 - margin)) ** 2 - other - np.sum(error**2))
        grads.append(-2 * np.einsum("sq,sqp->p", error, jac))
        indices = np.argmax(np.abs(error), axis=0)
        for col, row in enumerate(indices):
            values.append(peak * (1 - margin) - abs(error[row, col]))
            grads.append(-np.sign(error[row, col]) * jac[row, col])
    return np.asarray(values), np.asarray(grads)


def startup_ok(qerr, lerr, *, margin=0):
    return all(
        np.isfinite(x).all()
        and np.sqrt(np.mean(x * x)) <= rms * (1 - margin)
        and np.max(np.abs(x)) <= peak * (1 - margin)
        for x, rms, peak in ((qerr, 0.018, 0.04), (lerr, 0.009, 0.025))
    )


def choose_candidate(evaluations, fallback):
    eligible = [e for e in evaluations if e["valid"] and e["mse"] <= e["baseline_mse"]]
    return min(eligible, key=lambda e: e["objective"]) if eligible else fallback


class LegEvaluator:
    def __init__(
        self,
        startup,
        teacher,
        mapping,
        target_q,
        target_ltm,
        target_teacher,
        fixed_sse,
        baseline_mse,
        budget,
        output,
        *,
        retain_startup_jacobians=False,
    ):
        self.startup, self.teacher, self.mapping = startup, teacher, mapping
        require(
            target_teacher.shape == (1201, 7),
            "Optimization target must exclude development",
        )
        for engine, expected in ((startup, 101), (teacher, 1201)):
            require(
                engine.problem["samples"] == expected,
                "Optimization engine exposes wrong context rows",
            )
        self.targets = target_q, target_ltm, target_teacher
        self.fixed_sse, self.baseline_mse = fixed_sse, baseline_mse
        self.budget, self.output = budget, output
        self.cache, self.evaluations = {}, []
        require(type(retain_startup_jacobians) is bool, "Explicit Jacobian retention flag required")
        self.retain_startup_jacobians = retain_startup_jacobians

    def evaluate(self, flat):
        c = np.asarray(flat, np.float64).reshape(-1, 3)
        require(coefficient_valid(c), "Invalid optimizer coefficients")
        key = c.tobytes()
        if key in self.cache:
            self.budget.cache_hits += 1
            self.budget.check()
            return self.cache[key]
        self.budget.check(new=True)
        self.budget.unique += 1
        self.budget.total_unique += 1
        idx = len(self.evaluations)
        raw = {}
        active_engine = None
        active_device = None
        try:
            for name, engine in (("startup", self.startup), ("teacher", self.teacher)):
                active_engine = engine
                active_device = None
                self.budget.check()
                self.budget.host_requested += 1
                device = engine.solve(c, derivative=True, host=False)
                active_device = device
                raw[name] = {
                    k: engine.host_copy(device[k], label=f"eval-{idx}:{name}:{k}")
                    for k in ("q", "ltm")
                }
                raw[name]["receipt"] = device.get("receipt", {})
                require(
                    raw[name]["q"].shape == ((101 if name == "startup" else 1201), 7),
                    "Optimization response exposes development/wrong rows",
                )
                require(
                    raw[name]["ltm"].shape == ((101 if name == "startup" else 1201), 1),
                    "Optimization tendon response exposes development/wrong rows",
                )
                if name == "startup":
                    raw[name]["jacobian"] = engine.command_jacobian(
                        device, np.arange(1, 100, dtype=np.int64), host=True
                    )
                else:
                    error = raw[name]["q"][1:] - self.targets[2][1:]
                    cotangent = np.zeros(
                        (len(raw[name]["q"]), len(self.mapping["fixed"])), np.float64
                    )
                    cotangent[1:1201] = (0.04 / error.size / self.baseline_mse) * (
                        error @ self.mapping["moment"]
                    )
                    raw[name]["objective_gradient"] = (
                        engine.activation_cotangent_product(
                            device, cotangent, host=True
                        )
                    )
                self.budget.host_completed += 1
                del device
                active_device = None
                np.savez(
                    self.output / f"eval-{idx:03d}-{name}.npz",
                    coefficients=c,
                    **{k: v for k, v in raw[name].items() if isinstance(v, np.ndarray)},
                )
                write_json(
                    self.output / f"eval-{idx:03d}-{name}-receipt.json",
                    raw[name].get("receipt", {}),
                )
                require(
                    all(np.isfinite(raw[name][k]).all() for k in ("q", "ltm"))
                    and (
                        np.isfinite(raw[name]["objective_gradient"]).all()
                        if name == "teacher"
                        else all(
                            np.isfinite(v).all() for v in raw[name]["jacobian"].values()
                        )
                    ),
                    "Incomplete/nonfinite response",
                )
                if name == "startup":
                    self.budget.check()
            s, t = raw["startup"], raw["teacher"]
            qe, le = s["q"][1:100] - self.targets[0], s["ltm"][1:100] - self.targets[1]
            delta = c - neutral(len(c))
            mse = mean_square(t["q"][1:1201] - self.targets[2][1:1201])
            obj = mse / self.baseline_mse + 1e-4 * np.mean(delta**2)
            grad = t["objective_gradient"] + 2e-4 * delta.ravel() / delta.size
            constraints, cj = startup_constraints(
                qe, le, s["jacobian"]["q"], s["jacobian"]["ltm"], self.fixed_sse
            )
            entry = {
                "coefficients": c.copy(),
                "objective": obj,
                "gradient": grad,
                "mse": mse,
                "baseline_mse": self.baseline_mse,
                "constraints": constraints,
                "constraint_jacobian": cj,
                "qerr": qe,
                "lerr": le,
                "valid": coefficient_valid(c, interior=True)
                and np.all(constraints >= 0),
                "evaluation": idx,
            }
            if self.retain_startup_jacobians:
                entry["startup_q_jacobian"] = s["jacobian"]["q"]
                entry["startup_ltm_jacobian"] = s["jacobian"]["ltm"]
            self.evaluations.append(entry)
            self.cache[key] = entry
            write_json(self.output / "cost.json", self.budget.receipt())
            # Preserve a fully completed feasible evaluation before signaling timeout.
            self.budget.check()
            return entry
        except BudgetExceeded:
            self.budget.budget_stops += 1
            write_json(self.output / "cost.json", self.budget.receipt())
            raise
        except Exception as error:
            self.budget.failed += 1
            if not hasattr(error, "partial_response") and active_device is not None:
                error.partial_response = {
                    k: v for k, v in active_device.items() if k in ("q", "ltm", "B")
                }
            if hasattr(error, "partial_response"):
                self.budget.incomplete += 1
            if not hasattr(error, "response_receipt") and active_engine is not None:
                error.response_receipt = copy.deepcopy(
                    getattr(active_engine, "last_receipt", {})
                )
            failure_record(
                error,
                active_engine,
                self.output,
                f"eval-{idx:03d}",
                self.budget.receipt(),
            )
            raise


def fit_leg(evaluator, initial):
    """Exactly one SLSQP call; budgets never trigger a restart."""
    budget = evaluator.budget
    status = "complete"
    budget_scope = None
    primary_error = None
    budget.solver_calls += 1

    def callback(_):
        budget.iterations += 1
        budget.check()

    try:
        result = minimize(
            lambda x: evaluator.evaluate(x)["objective"],
            initial.ravel(),
            jac=lambda x: evaluator.evaluate(x)["gradient"],
            method="SLSQP",
            bounds=Bounds(0, 8),
            constraints=[
                triple_constraint(len(initial)),
                {
                    "type": "ineq",
                    "fun": lambda x: evaluator.evaluate(x)["constraints"],
                    "jac": lambda x: evaluator.evaluate(x)["constraint_jacobian"],
                },
            ],
            callback=callback,
            options={"maxiter": 60, "ftol": 1e-9, "disp": False},
        )
        status = str(result.message)
    except BudgetExceeded as error:
        status = str(error)
        budget_scope = error.scope
    except Exception as error:
        status = "execution_failure"
        primary_error = error
    fallback = {
        "coefficients": initial.copy(),
        "objective": 1.0,
        "mse": evaluator.baseline_mse,
        "baseline_mse": evaluator.baseline_mse,
        "valid": True,
        "evaluation": None,
    }
    selected = choose_candidate(evaluator.evaluations, fallback)
    selected = dict(selected, status=status, budget_scope=budget_scope)
    def persist_selection():
        np.savez(
            evaluator.output / "selected.npz",
            **{k: v for k, v in selected.items() if isinstance(v, np.ndarray)},
        )
        write_json(
            evaluator.output / "selection.json",
            {"status": status, "budget_scope": budget_scope,
             "evaluation": selected["evaluation"], "objective": selected["objective"],
             "mse": selected["mse"], "cost": budget.receipt()},
        )
    if primary_error is not None:
        preserve_primary(primary_error, "Failed fit selection could not be saved", persist_selection)
        raise primary_error
    persist_selection()
    return selected


def run_fit_pass(run_leg, budget, edge_indices, output):
    """Single fixed-order pass; untouched legs remain explicitly neutral."""
    combined = neutral(sum(len(e) for e in edge_indices))
    statuses = [{"leg": name, "status": "neutral_not_started"} for name in LEGS]
    for leg, edges in enumerate(edge_indices):
        if budget.shared_exhausted():
            break
        try:
            budget.start_leg()
            chosen = run_leg(leg)
            combined[edges] = chosen["coefficients"]
            statuses[leg] = {
                "leg": LEGS[leg],
                "status": chosen["status"],
                "budget_scope": chosen.get("budget_scope"),
                "evaluation": chosen["evaluation"],
            }
        except BudgetExceeded as error:
            if error.scope == "original" or getattr(error, "abort_run", False):
                raise
            budget.budget_stops += 1
            statuses[leg] = {
                "leg": LEGS[leg],
                "status": "neutral_budget_stop",
                "budget_scope": error.scope,
                "evaluation": None,
            }
        finally:
            pending = sys.exc_info()[1]
            try:
                np.savez(output / "partial-candidate.npz", coefficients=combined)
                write_json(
                    output / "fit-pass.json", {"legs": statuses, "cost": budget.receipt()}
                )
            except Exception as secondary:
                if pending is None:
                    raise
                pending.add_note("Partial candidate receipt failed: " + repr(secondary))
        if budget.shared_exhausted():
            break
    budget.stop()
    write_json(output / "fit-pass.json", {"legs": statuses, "cost": budget.receipt()})
    return combined, statuses


def round_and_finalize(coefficients, predict, deadline, output):
    """One rounding and one prediction; expired fit/leg clocks do not apply."""
    coefficients32 = coefficients.astype(np.float32)
    rounded = coefficients32.astype(np.float64)
    require(coefficient_valid(rounded), "Rounded E267 coefficient range failed")
    np.savez(
        output / "candidate.npz",
        coefficients64=coefficients,
        coefficients32=coefficients32,
    )
    deadline.check()
    result = predict(rounded)
    deadline.check()
    return result


def write_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )


def contract_leg(mapping, legs, leg):
    """Explicitly contract all axes, retaining original source and edge order."""
    n, m, e, q, l = [
        np.flatnonzero(legs[k] == leg) for k in ("motor", "muscle", "edge", "q", "ltm")
    ]
    require(len(q) == 7 and len(l) == 1 and len(e) > 0, "Invalid leg ownership")
    require(
        np.all(np.isin(mapping["edge_post"][e], n)), "Edge leg differs from motor leg"
    )
    require(
        np.all(np.isin(mapping["membership"][n], m)), "Motor muscle ownership differs"
    )
    result = dict(mapping)
    for k in (
        "initial_rate",
        "initial_adaptation",
        "tau",
        "coefficient",
        "threshold",
        "cap",
        "cell_gain",
        "cell_inverse_adaptation_tau",
        "motor_ids",
        "low",
        "high",
        "kappa",
        "bank0",
    ):
        result[k] = mapping[k][n].copy()
    result["stage_total"] = mapping["stage_total"][:, n].copy()
    result["membership"] = np.searchsorted(m, mapping["membership"][n]).astype(np.int64)
    result["pool"] = mapping["pool"][np.ix_(m, n)].copy()
    for k in ("fixed", "activation0", "original_activation0"):
        result[k] = mapping[k][m].copy()
    for k in ("edge_ids", "edge_source", "weights"):
        result[k] = mapping[k][e].copy()
    result["edge_post"] = np.searchsorted(n, mapping["edge_post"][e]).astype(np.int64)
    result["moment"] = mapping["moment"][np.ix_(q, m)].copy()
    require(
        np.all(
            mapping["moment"][
                np.ix_(q, np.setdiff1d(np.arange(len(legs["muscle"])), m))
            ]
            == 0
        ),
        "Cross-leg moment is not separable",
    )
    result["rest"] = mapping["rest"][q].copy()
    require(np.all(np.isin(mapping["ltm"][l], m)), "Cross-leg tendon ownership")
    result["ltm"] = np.searchsorted(m, mapping["ltm"][l]).astype(np.int64)
    return result, (n, m, e, q, l)


def adjudicate(q, baseline, target, startup_qerr, startup_lerr, leg_q):
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    train, development = slice(1, 1201), slice(1201, 2001)
    checks = {
        "startup": startup_ok(startup_qerr, startup_lerr),
        "train": rms(q[train] - target[train])
        <= 0.8 * rms(baseline[train] - target[train]),
        "development": rms(q[development] - target[development])
        <= rms(baseline[development] - target[development]),
    }
    for leg in range(6):
        cols = leg_q == leg
        for name, sl in (("train", train), ("development", development)):
            checks[f"{LEGS[leg]}_{name}"] = rms(
                q[sl][:, cols] - target[sl][:, cols]
            ) <= 1.05 * rms(baseline[sl][:, cols] - target[sl][:, cols])
    return {
        "checks": checks,
        "conditional_support": all(checks.values()),
        "body_promotion": False,
    }


STATIC_KEYS = (
    "motor_ids",
    "source_ids",
    "edge_ids",
    "edge_post",
    "edge_source",
    "weights",
    "tau",
    "coefficient",
    "threshold",
    "cap",
    "cell_gain",
    "cell_inverse_adaptation_tau",
    "membership",
    "pool",
    "low",
    "high",
    "kappa",
    "fixed",
    "moment",
    "rest",
    "ltm",
)


def validate_static_contexts(startup, teacher):
    for key in STATIC_KEYS:
        a, b = startup[key], teacher[key]
        require(
            a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
            "Context static mapping changed: " + key,
        )
    require(
        startup["stage_order"] == teacher["stage_order"], "Context stage order differs"
    )


def validate_targets(targets):
    expected = {"startup_q": (99, 42), "startup_ltm": (99, 6), "teacher_q": (2001, 42)}
    require(set(targets) == set(expected), "Exact targets required")
    for key, shape in expected.items():
        x = targets[key]
        require(
            x.dtype == np.float64 and x.shape == shape and np.isfinite(x).all(),
            "Finite float64 target shape required: " + key,
        )


def validate_leg_ownership(mapping, legs):
    sizes = {
        "motor": len(mapping["motor_ids"]),
        "muscle": len(mapping["fixed"]),
        "edge": len(mapping["edge_ids"]),
        "q": len(mapping["rest"]),
        "ltm": len(mapping["ltm"]),
    }
    require(set(legs) == set(sizes), "Exact explicit leg axes required")
    for key, size in sizes.items():
        x = legs[key]
        require(
            x.dtype == np.int64 and x.shape == (size,) and np.all((x >= 0) & (x < 6)),
            "Invalid leg labels: " + key,
        )
    require(
        np.array_equal(legs["motor"], legs["muscle"][mapping["membership"]]),
        "Motor leg differs from muscle membership",
    )
    require(
        np.array_equal(legs["edge"], legs["motor"][mapping["edge_post"]]),
        "Edge leg differs from postsynaptic motor",
    )
    require(
        np.array_equal(legs["ltm"], legs["muscle"][mapping["ltm"]]),
        "Tendon leg differs from muscle",
    )
    require(
        np.all(mapping["pool"][legs["muscle"][:, None] != legs["motor"][None, :]] == 0),
        "Cross-leg pool ownership",
    )
    require(
        np.all(mapping["moment"][legs["q"][:, None] != legs["muscle"][None, :]] == 0),
        "Cross-leg joint moment ownership",
    )
    require(
        all(
            np.sum(legs["q"] == i) == 7 and np.sum(legs["ltm"] == i) == 1
            for i in range(6)
        ),
        "Exact seven joints and one tendon per leg required",
    )


def startup_metrics(qerr, lerr):
    metrics = {}
    for name, x, rms_limit, max_limit in (
        ("q", qerr, 0.018, 0.04),
        ("ltm", lerr, 0.009, 0.025),
    ):
        metrics[name] = {
            "rms": float(np.sqrt(np.mean(x * x))),
            "max": float(np.max(np.abs(x))),
            "rms_tolerance": rms_limit,
            "max_tolerance": max_limit,
        }
    return {"metrics": metrics, "pass": startup_ok(qerr, lerr)}


def numerical_gate(actual, expected, tolerance):
    require(actual.shape == expected.shape, "Reproduction reference shape differs")
    delta = np.abs(actual - expected)
    error = float(np.max(delta))
    return {
        "max_absolute_error": error if np.isfinite(error) else None,
        "tolerance": tolerance,
        "pass": bool(np.isfinite(error) and error <= tolerance),
    }


def array_bytes(tree):
    roots = {}

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, np.ndarray):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            roots[id(owner)] = owner.nbytes

    visit(tree)
    return sum(roots.values())


class RunMeter:
    def __init__(self, output, deadline):
        self.output, self.deadline, self.budget = output, deadline, None
        self.engine_events, self.memory_events, self.cleanup_events = [], [], []
        self.phase = "input_validation"
        self.host_bytes = 0
        self.current_estimate = 0
        self.peak_estimate = 0
        self.active_contexts = 0
        self.peak_contexts = 0
        self.peak_rss = None
        self.save()

    def receipt(self):
        return {
            "phase": self.phase,
            **self.deadline.receipt(),
            "fit": self.budget.receipt() if self.budget else None,
            "engine_events": self.engine_events,
            "memory_events": self.memory_events,
            "cleanup_events": self.cleanup_events,
            "peak_contexts_live": self.peak_contexts,
            "aggregate_estimated_peak_bytes": self.peak_estimate,
            "observed_peak_rss_bytes": self.peak_rss,
        }

    def save(self):
        write_json(self.output / "cost.json", self.receipt())

    def memory(self, engine, phase):
        estimate = int(engine.problem["gpu_bytes_estimate"])
        # Estimate includes prepared host copies as well as device input/output bound.
        estimate += 2 * array_bytes(engine.problem)
        self.active_contexts += 1
        self.current_estimate += estimate
        self.peak_contexts = max(self.peak_contexts, self.active_contexts)
        require(self.active_contexts <= 2, "At most two fitting contexts may be live")
        self.peak_estimate = max(
            self.peak_estimate, self.host_bytes + self.current_estimate
        )
        self.memory_events.append(
            {
                "phase": phase,
                "host_base_bytes": self.host_bytes,
                "context_estimated_bytes": estimate,
                "aggregate_estimated_bytes": self.host_bytes + self.current_estimate,
            }
        )
        try:
            import psutil

            rss = psutil.Process().memory_info().rss
            self.peak_rss = max(self.peak_rss or 0, rss)
            self.memory_events[-1]["observed_rss_bytes"] = rss
        except (ImportError, OSError) as error:
            self.memory_events[-1]["rss_unavailable"] = str(error)
        return estimate

    def release(self, engine, phase, estimate):
        self.engine_events.append(
            {
                "phase": phase,
                "lifetime_receipt": copy.deepcopy(engine.receipt),
                "last_receipt": copy.deepcopy(getattr(engine, "last_receipt", {})),
                "host_copy_events": copy.deepcopy(
                    getattr(engine, "host_copy_events", [])
                ),
            }
        )
        self.current_estimate -= estimate
        self.active_contexts -= 1
        self.save()

    def cleanup(self, cp=None):
        started = self.deadline.clock()
        event = {"phase": self.phase, "status": "requested"}
        try:
            gc.collect()
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
            event["status"] = "complete"
        except Exception as error:
            event.update(status="failed", error=str(error))
        event["wall_s"] = self.deadline.clock() - started
        self.cleanup_events.append(event)
        self.save()


@contextmanager
def engine_scope(factory, mapping, meter, phase):
    engine, estimate = None, None
    primary = None
    meter.phase = phase
    meter.deadline.check()
    try:
        engine = factory(mapping)
        estimate = meter.memory(engine, phase)
        yield engine
    except BudgetExceeded as error:
        primary = error
        preserve_primary(error, "Budget-stop cost could not be saved", meter.save)
        raise
    except Exception as error:
        primary = error
        if not hasattr(error, "response_receipt") and engine is not None:
            error.response_receipt = copy.deepcopy(
                getattr(engine, "last_receipt", engine.receipt)
            )
        failure_record(error, engine, meter.output, phase, meter.receipt())
        raise
    finally:
        if engine is not None and estimate is not None:
            if primary is None:
                meter.release(engine, phase, estimate)
            else:
                preserve_primary(primary, "Engine release receipt failed",
                                 lambda: meter.release(engine, phase, estimate))
        if primary is None:
            meter.save()
        else:
            preserve_primary(primary, "Final failure cost could not be saved", meter.save)


def selective_copy(engine, result, keys, label):
    try:
        return {
            key: engine.host_copy(result[key], label=label + ":" + key) for key in keys
        }
    except Exception as error:
        if not hasattr(error, "partial_response"):
            error.partial_response = {k: result[k] for k in keys}
        if not hasattr(error, "response_receipt"):
            error.response_receipt = copy.deepcopy(getattr(engine, "last_receipt", {}))
        raise


def original_run(freeze, output, original_started):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.banc_m3_source_coefficient_engine import (
        CudaSourceCoefficientEngine,
        prepare_problem,
    )

    deadline = OriginalDeadline(original_started)
    meter = RunMeter(output, deadline)
    primary = None
    try:
        return execute_original(
            freeze, output, CudaSourceCoefficientEngine, prepare_problem, meter
        )
    except BudgetExceeded as error:
        primary = error
        preserve_primary(error, "Original budget receipt failed", lambda: write_json(
            output / "budget-stop.json", {"scope": primary.scope, "cost": meter.receipt()}))
        raise
    except Exception as error:
        primary = error
        failure_record(error, None, output, "original", meter.receipt())
        raise
    finally:
        if primary is None:
            meter.save()
        else:
            preserve_primary(primary, "Original final failure cost failed", meter.save)


def execute_original(freeze, output, factory, prepare_problem, meter, *, leg_fit=None,
                     evaluator_factory=LegEvaluator):
    deadline = meter.deadline
    mappings = {
        name: {k: load_ref(freeze, r) for k, r in ctx["mapping"].items()}
        for name, ctx in freeze["contexts"].items()
    }
    require(set(mappings) == {"startup", "teacher"}, "Exactly two contexts required")
    meter.host_bytes = array_bytes(mappings)
    meter.save()
    for name, mapping in mappings.items():
        mapping["stage_order"] = tuple(mapping["stage_order"].tolist())
        prepared = prepare_problem(mapping)
        meter.peak_estimate = max(
            meter.peak_estimate, meter.host_bytes + array_bytes(prepared)
        )
        require(
            len(mapping["stage_total"]) == (8000 if name == "startup" else 160000),
            "Context stage count",
        )
        del prepared
        deadline.check()
    validate_static_contexts(mappings["startup"], mappings["teacher"])
    identity = freeze["identity"]
    for mapping in mappings.values():
        require(
            [
                len(mapping[k])
                for k in ("motor_ids", "source_ids", "edge_ids", "fixed", "rest")
            ]
            == [
                identity[k] for k in ("motors", "sources", "edges", "muscles", "joints")
            ]
            == [391, 51, 1660, 96, 42],
            "Exact original identities/counts required",
        )
    legs = {k: load_ref(freeze, r) for k, r in freeze["legs"].items()}
    validate_leg_ownership(mappings["startup"], legs)
    require(
        [int(np.sum(legs["edge"] == i)) for i in range(6)]
        == [300, 282, 254, 261, 268, 295],
        "Original leg edge counts changed",
    )
    targets = {k: load_ref(freeze, r) for k, r in freeze["targets"].items()}
    validate_targets(targets)
    refs = {
        name: {
            k: load_ref(freeze, r)
            for k, r in freeze["contexts"][name]["reference"].items()
        }
        for name in mappings
    }
    for name, reference in refs.items():
        rows = 101 if name == "startup" else 2001
        for key, tail in (
            ("rate", (391,)),
            ("adaptation", (391,)),
            ("q", (42,)),
            ("ltm", (6,)),
            ("activation", (96,)),
            ("bank", (391, 3)),
        ):
            x = reference[key]
            require(
                x.shape == (rows, *tail)
                and x.dtype in (np.float32, np.float64)
                and np.isfinite(x).all(),
                "Invalid native reference: " + name + ":" + key,
            )
    meter.host_bytes = array_bytes((mappings, refs, targets, legs))
    baselines = {
        "startup": {"q": np.empty((101, 42)), "ltm": np.empty((101, 6))},
        "teacher": {"q": np.empty((2001, 42))},
    }
    indices = {
        leg: tuple(
            np.flatnonzero(legs[k] == leg)
            for k in ("motor", "muscle", "edge", "q", "ltm")
        )
        for leg in range(6)
    }
    gates = {}

    def neutral_context(leg, name):
        part, idx = contract_leg(mappings[name], legs, leg)
        n, m, _, q, l = idx
        phase = "neutral-" + LEGS[leg] + "-" + name
        with engine_scope(factory, part, meter, phase) as engine:
            response = engine.solve(neutral(len(idx[2])), derivative=False, host=False)
            host = selective_copy(
                engine,
                response,
                ("rate", "adaptation", "q", "ltm", "activation", "bank"),
                phase,
            )
            del response
            np.savez(output / (phase + ".npz"), **host)
            for key, cols, tolerance in (
                ("rate", n, 0.002),
                ("adaptation", n, 0.002),
                ("q", q, 0.00005),
                ("ltm", l, 0.00001),
                ("activation", m, 0.005),
            ):
                gates[phase + ":" + key] = numerical_gate(
                    host[key], refs[name][key][:, cols], tolerance
                )
            baselines[name]["q"][:, q] = host["q"]
            if name == "startup":
                baselines[name]["ltm"][:, l] = host["ltm"]
            del host
            decoded = engine.decode_rates(refs[name]["rate"][:, n], host=False)
            direct = selective_copy(
                engine, decoded, ("q", "ltm", "activation", "bank"), phase + ":direct"
            )
            del decoded
            np.savez(output / (phase + "-direct.npz"), **direct)
            for key, cols, tolerance in (
                ("q", q, 1e-11),
                ("ltm", l, 1e-11),
                ("activation", m, 1e-10),
                ("bank", n, 1e-10),
            ):
                gates[phase + ":direct:" + key] = numerical_gate(
                    direct[key], refs[name][key][:, cols], tolerance
                )
            cp = engine.cp
            deadline.check()
        return cp

    for leg in range(6):
        for name in mappings:
            cp = neutral_context(leg, name)
            meter.cleanup(cp)
            write_json(
                output / "neutral-gates.json", {"checks": gates, "optimizer_calls": 0}
            )
    qerr = baselines["startup"]["q"][1:100] - targets["startup_q"]
    lerr = baselines["startup"]["ltm"][1:100] - targets["startup_ltm"]
    gates["strict_global_startup"] = startup_metrics(qerr, lerr)
    gate_pass = all(g["pass"] for g in gates.values())
    write_json(
        output / "neutral-gates.json",
        {"checks": gates, "pass": gate_pass, "optimizer_calls": 0},
    )
    require(gate_pass, "Neutral/decoder reproduction gate failed; optimizer calls zero")
    meter.host_bytes = array_bytes((mappings, refs, targets, legs, baselines))
    # Full references have served fidelity; retain baseline q/LTM and training-only targets in evaluator.
    refs = {}
    meter.host_bytes = array_bytes((mappings, targets, legs, baselines))
    train = training_mapping(mappings["teacher"])
    meter.budget = Budget(clock=deadline.clock, deadline=deadline)
    budget = meter.budget
    fit_metadata = []

    def run_leg(leg):
        _, _, e, q, l = indices[leg]
        leg_output = output / LEGS[leg]
        leg_output.mkdir()
        fixed = (
            np.sum(qerr * qerr) - np.sum(qerr[:, q] ** 2),
            np.sum(lerr * lerr) - np.sum(lerr[:, l] ** 2),
        )
        baseline_mse = mean_square(
            baselines["teacher"]["q"][1:1201][:, q]
            - targets["teacher_q"][1:1201][:, q]
        )
        recorded_mse = mean_square(
            load_ref(freeze, freeze["contexts"]["teacher"]["reference"]["q"])[
                1:1201
            ][:, q]
            - targets["teacher_q"][1:1201][:, q]
        )
        fit_metadata.append(
            {
                "leg": LEGS[leg],
                "baseline_leg_mse": baseline_mse,
                "actual_decoder_leg_mse": recorded_mse,
                "difference": baseline_mse - recorded_mse,
            }
        )
        write_json(output / "baseline-leg-mse.json", fit_metadata)
        # Contraction and engine preparation belong to this leg/shared fit clocks.
        startup_part, _ = contract_leg(mappings["startup"], legs, leg)
        teacher_part, _ = contract_leg(train, legs, leg)
        budget.check()
        deadline.check()
        with engine_scope(
            factory, startup_part, meter, "fit-" + LEGS[leg] + "-startup"
        ) as es:
            budget.check()
            with engine_scope(
                factory, teacher_part, meter, "fit-" + LEGS[leg] + "-training"
            ) as et:
                budget.check()
                deadline.check()
                evaluator = evaluator_factory(
                    es,
                    et,
                    es.problem,
                    targets["startup_q"][:, q],
                    targets["startup_ltm"][:, l],
                    targets["teacher_q"][:1201, q],
                    fixed,
                    baseline_mse,
                    budget,
                    leg_output,
                )
                chosen = (leg_fit or fit_leg)(evaluator, neutral(len(e)))
                cache_bytes = array_bytes(evaluator.cache)
                meter.peak_estimate = max(
                    meter.peak_estimate,
                    meter.host_bytes + meter.current_estimate + cache_bytes,
                )
                meter.memory_events.append(
                    {
                        "phase": "fit-" + LEGS[leg] + ":cache",
                        "cache_host_bytes": cache_bytes,
                    }
                )
                if chosen["evaluation"] is not None:
                    qerr[:, q], lerr[:, l] = chosen["qerr"], chosen["lerr"]
                cp = es.cp
        return chosen, cp

    def bounded_leg(leg):
        cp = None
        try:
            selected, cp = run_leg(leg)
            return selected
        finally:
            # run_leg's local engines and all device results have left scope.
            meter.cleanup(cp)

    combined, statuses = run_fit_pass(
        bounded_leg, budget, [indices[i][2] for i in range(6)], output
    )
    final = {
        "startup": {"q": np.empty((101, 42)), "ltm": np.empty((101, 6))},
        "teacher": {"q": np.empty((2001, 42))},
    }

    def final_context(leg, name, rounded):
        part, idx = contract_leg(mappings[name], legs, leg)
        phase = "final-" + LEGS[leg] + "-" + name
        with engine_scope(factory, part, meter, phase) as engine:
            response = engine.solve(rounded[idx[2]], derivative=False, host=False)
            host = selective_copy(engine, response, ("q", "ltm"), phase)
            del response
            require(
                all(np.isfinite(v).all() for v in host.values()),
                "Final prediction nonfinite",
            )
            np.savez(output / (phase + ".npz"), **host)
            final[name]["q"][:, idx[3]] = host["q"]
            if name == "startup":
                final[name]["ltm"][:, idx[4]] = host["ltm"]
            cp = engine.cp
            deadline.check()
        return cp

    def predict(rounded):
        for leg in range(6):
            for name in mappings:
                deadline.check()
                cp = final_context(leg, name, rounded)
                meter.cleanup(cp)
        return adjudicate(
            final["teacher"]["q"],
            baselines["teacher"]["q"],
            targets["teacher_q"],
            final["startup"]["q"][1:100] - targets["startup_q"],
            final["startup"]["ltm"][1:100] - targets["startup_ltm"],
            legs["q"],
        )

    verdict = round_and_finalize(combined, predict, deadline, output)
    verdict["fit_budget_stopped"] = (
        any(s.get("budget_scope") for s in statuses) or budget.shared_exhausted()
    )
    verdict["leg_statuses"] = statuses
    write_json(output / "checks.json", verdict)
    meter.save()
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "freeze",
        "freeze-sha256",
        "source-review",
        "source-review-sha256",
        "output",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(not output.exists(), "Fresh output directory required")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "cost.json",
        {
            "phase": "freeze_validation",
            "original_wall_s": 0,
            "fit": None,
            "engine_events": [],
            "optimizer_calls": 0,
        },
    )
    try:
        freeze = validate_freeze(
            Path(args.freeze),
            args.freeze_sha256,
            Path(args.source_review),
            args.source_review_sha256,
        )
        verdict = original_run(freeze, output, started)
        write_json(
            output / "result.json",
            {
                "status": "complete",
                "original_wall_s": time.perf_counter() - started,
                **verdict,
            },
        )
        return 0
    except BudgetExceeded as error:
        cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
        cost["original_wall_s"] = time.perf_counter() - started
        write_json(output / "cost.json", cost)
        write_json(
            output / "result.json",
            {
                "status": "budget_stop",
                "scope": error.scope,
                "original_wall_s": time.perf_counter() - started,
            },
        )
        return 1
    except Exception as error:
        cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
        cost["original_wall_s"] = time.perf_counter() - started
        write_json(output / "cost.json", cost)
        write_json(
            output / "result.json",
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "original_wall_s": time.perf_counter() - started,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
