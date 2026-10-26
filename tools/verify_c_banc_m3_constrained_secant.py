"""E326: one output-constrained secant step and one original continuous rollout.

The affine startup interpolation is a hypothesis. Only the new full forward
trajectory determines whether the four unchanged E103 limits are met.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from tools import verify_c_banc_m3_full_horizon_direction as e323
from tools import verify_c_banc_m3_magnitude_direction as e325
from tools import verify_c_banc_m3_reset_window_direction as e324

ROOT, BASE = e323.ROOT, e323.BASE
N, GROUPS = e323.N, e323.GROUPS
E325 = e325.OUTPUT
OUTPUT = BASE + "/e326-constrained-secant"
SCHEMA = "flylab.m3.e326.constrained-secant.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e326.constrained-secant.review.v1"
RESULT_SCHEMA = "flylab.m3.e326.constrained-secant.result.v1"
OWNED = ("tools/verify_c_banc_m3_constrained_secant.py",
         "tests/test_c_banc_constrained_secant.py")
LIMITS = {"supervisor_wall_s": 60, "internal_wall_s": 50,
          "science_wall_s": 35, "finalization_reserve_s": 15,
          "host_bytes": 6 << 30, "additional_gpu_bytes": 8 << 30,
          "output_bytes": 1 << 30, "minimum_free_disk_bytes": 4 << 30,
          "forward_neural_ticks": 6000, "recompute_neural_ticks": 0,
          "backward_block_ticks": 0, "backward_blocks": 0,
          "decoder_replay_controls": 0, "body_calls": 0,
          "optimizer_steps": 0, "retry_count": 0}
PATHS = {**e325.PATHS,
         "e325_freeze": BASE + "/e325-magnitude-direction-freeze.json",
         "e325_review": BASE + "/e325-magnitude-direction-source-review.json",
         "e325_result": E325 + "/result.json",
         "e325_audit": BASE + "/parent-e325-original-raw-audit.json",
         "e325_source_manifest": E325 + "/source-manifest.json",
         "e325_output_manifest": E325 + "/output-manifest.json",
         "e325_ledger": E325 + "/advance-ledger.json",
         "e325_output": E325,
         "e323_neutral": e323.OUTPUT + "/neutral.npz"}


def _metrics(error: np.ndarray) -> dict:
    return {"rms": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
            "max": float(np.max(np.abs(error)))}


def _scalar_upper(error: np.ndarray, change: np.ndarray, cap: float) -> float:
    moving = change != 0
    if not np.any(moving):
        return 1.0
    e, d = error[moving], change[moving]
    bounds = np.where(d > 0, (cap - e) / d, (-cap - e) / d)
    return float(min(1.0, np.min(bounds)))


def _rms_upper(error: np.ndarray, change: np.ndarray, cap: float) -> float:
    a = float(np.mean(np.square(change), dtype=np.float64))
    if a == 0:
        return 1.0
    b = float(2 * np.mean(error * change, dtype=np.float64))
    c = float(np.mean(np.square(error), dtype=np.float64) - cap * cap)
    # c < 0 by the strict neutral check, so the positive root is unique.
    discriminant = b * b - 4 * a * c
    root = (-2 * c / (b + np.sqrt(discriminant)) if b >= 0
            else (-b + np.sqrt(discriminant)) / (2 * a))
    return float(min(1.0, root))


def feasible_alpha(q0: np.ndarray, t0: np.ndarray, q1: np.ndarray,
                   t1: np.ndarray, target_q: np.ndarray,
                   target_tendon: np.ndarray) -> dict:
    """Solve all scalar and mean-square startup inequalities on [0, 1]."""
    arrays = (q0, t0, q1, t1, target_q, target_tendon)
    if (any(not isinstance(x, np.ndarray) or x.dtype != np.float64
            or x.ndim != 2 or not np.isfinite(x).all() for x in arrays)
            or q0.shape != q1.shape or q0.shape != target_q.shape
            or t0.shape != t1.shape or t0.shape != target_tendon.shape
            or q0.shape[0] == 0 or t0.shape[0] != q0.shape[0]):
        raise ValueError("Finite matching f64 startup arrays required")
    qe, te = q0 - target_q, t0 - target_tendon
    qd, td = q1 - q0, t1 - t0
    neutral = {"q": _metrics(qe), "tendon": _metrics(te)}
    if (neutral["q"]["rms"] >= .018 or neutral["q"]["max"] >= .04
            or neutral["tendon"]["rms"] >= .009
            or neutral["tendon"]["max"] >= .025):
        raise ValueError("E323 neutral does not strictly pass E103 startup")
    bounds = {"q_max": _scalar_upper(qe, qd, .04),
              "q_rms": _rms_upper(qe, qd, .018),
              "tendon_max": _scalar_upper(te, td, .025),
              "tendon_rms": _rms_upper(te, td, .009)}
    if any(not np.isfinite(x) or x <= 0 or x > 1 for x in bounds.values()):
        raise ValueError("No positive feasible E326 alpha")
    maximum = min(bounds.values())
    alpha = maximum / 2
    predicted = {"q": _metrics(qe + alpha * qd),
                 "tendon": _metrics(te + alpha * td)}
    if (predicted["q"]["rms"] > .018 or predicted["q"]["max"] > .04
            or predicted["tendon"]["rms"] > .009
            or predicted["tendon"]["max"] > .025):
        raise ArithmeticError("Computed midpoint violates E103 constraints")
    return {"alpha_max": maximum, "alpha_star": alpha, "limits": bounds,
            "limiting_constraints": [key for key, value in bounds.items()
                                     if value == maximum],
            "neutral_metrics": neutral, "predicted_metrics": predicted}


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path,
         review_sha256: str, output: Path, *, root: Path = ROOT) -> dict:
    """Hash the complete closed evidence graph before loading NPZ or CUDA."""
    if output.exists() or output.resolve() != e323.safe(OUTPUT, root):
        raise ValueError("Fresh literal E326 output required")
    if (e323.sha256(freeze_path) != freeze_sha256
            or e323.sha256(review_path) != review_sha256):
        raise ValueError("E326 freeze/review hash differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact E326 frozen diagnostic required")
    pins, sources = freeze.get("input_sha256"), freeze.get("source_sha256")
    if (not isinstance(pins, dict) or not isinstance(sources, dict)
            or not set(OWNED).issubset(sources)
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources
            or any(name in pins and pins[name] != digest
                   for name, digest in sources.items())):
        raise ValueError("Root approved E326 source pins required")
    prior_freeze = json.loads(e323.safe(PATHS["e325_freeze"], root).read_text("utf8"))
    prior_review = json.loads(e323.safe(PATHS["e325_review"], root).read_text("utf8"))
    prior_result = json.loads(e323.safe(PATHS["e325_result"], root).read_text("utf8"))
    prior_audit = json.loads(e323.safe(PATHS["e325_audit"], root).read_text("utf8"))
    prior_source = json.loads(e323.safe(PATHS["e325_source_manifest"], root).read_text("utf8"))
    prior_manifest = json.loads(e323.safe(PATHS["e325_output_manifest"], root).read_text("utf8"))
    if (prior_freeze.get("schema") != e325.SCHEMA
            or prior_freeze.get("paths") != e325.PATHS
            or prior_freeze.get("source_sha256") != prior_source.get("source_sha256")
            or prior_source.get("input_sha256") != prior_freeze.get("input_sha256")
            or prior_source.get("freeze_sha256") != pins[PATHS["e325_freeze"]]
            or prior_source.get("review_sha256") != pins[PATHS["e325_review"]]
            or prior_review.get("schema") != e325.REVIEW_SCHEMA
            or prior_review.get("approved") is not True
            or prior_review.get("blocking_defects") != []
            or prior_review.get("freeze_sha256") != pins[PATHS["e325_freeze"]]
            or prior_review.get("source_sha256") != prior_freeze["source_sha256"]
            or prior_result.get("schema") != e325.RESULT_SCHEMA
            or prior_result.get("run_valid") is not True
            or prior_result.get("classification") != "supported"
            or prior_result.get("continuation_completed") is not True
            or prior_result.get("continuation_supported") is not False
            or prior_result.get("actual_work", {}).get("forward_neural_ticks") != 14000
            or prior_audit.get("schema") != "flylab.m3.e325.root-original-raw-audit.v1"
            or prior_audit.get("pass") is not True
            or prior_audit.get("diagnostic_run_valid") is not True
            or prior_audit.get("continuation_supported") is not False
            or prior_audit.get("result_sha256") != pins[PATHS["e325_result"]]
            or prior_audit.get("advance_ledger_sha256") != pins[PATHS["e325_ledger"]]
            or prior_audit.get("output_manifest_sha256") != pins[PATHS["e325_output_manifest"]]
            or prior_manifest.get("schema") != "flylab.m3.e325.output-manifest.v1"):
        raise ValueError("Audited E325 producer proof required")
    metrics = prior_result["continuation"]["metrics"]
    audited = prior_audit.get("continuation_metrics", {})
    if (metrics.get("startup_pass") is not False
            or metrics.get("startup_q_max", 0) <= .04
            or any(not np.isfinite(metrics.get(key, np.nan))
                   or abs(metrics[key] - audited.get(key, np.inf)) > 1e-12
                   for key in ("shape_loss", "startup_q_rms", "startup_q_max",
                               "startup_tendon_rms", "startup_tendon_max"))
            or not np.isfinite(prior_result.get("neutral_shape_loss", np.nan))
            or metrics["shape_loss"] >= prior_result["neutral_shape_loss"]):
        raise ValueError("E325 audited shape/startup evidence differs")
    for name, digest in prior_freeze["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError("E325 source closure differs: " + name)
    for name, digest in prior_freeze["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("E325 input closure differs: " + name)
    expected = e323._file_roster(PATHS, root) | set(prior_freeze["input_sha256"])
    if set(pins) != expected:
        raise ValueError("Exact E326 input pin roster required")
    for name, digest in {**pins, **sources}.items():
        if not isinstance(digest, str) or e323.sha256(e323.safe(name, root)) != digest:
            raise ValueError("E326 source/input differs: " + name)
    prior_dir = e323.safe(E325, root)
    files = {p.relative_to(prior_dir).as_posix() for p in prior_dir.rglob("*")
             if p.is_file() and p.name not in
             ("result.json", "advance-ledger.json", "output-manifest.json")}
    if set(prior_manifest.get("files", {})) != files:
        raise ValueError("E325 producer output roster differs")
    for relative in files:
        p = prior_dir / relative
        if prior_manifest["files"][relative] != {
                "sha256": pins[p.relative_to(root).as_posix()], "bytes": p.stat().st_size}:
            raise ValueError("E325 producer output file differs: " + relative)
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E326 free disk reserve unavailable")
    return freeze


class Dependencies(e325.Dependencies):
    def load(self, freeze: dict, root: Path = ROOT) -> dict:
        from tools.verify_c_banc_m3_native_adjoint import load_npz

        data = super().load(freeze, root=root)
        result = json.loads(e323.safe(PATHS["e325_result"], root).read_text("utf8"))
        e325_dir = e323.safe(E325, root)
        e323_neutral = load_npz(e323.safe(PATHS["e323_neutral"], root))
        e325_actual = load_npz(e325_dir / "continuation-minus.npz")
        theta = load_npz(e325_dir / "direction.npz")["theta_minus"]
        e324._valid_array(theta, (GROUPS,), np.float64, "E325 actual theta")
        for name, arrays in (("E323 neutral", e323_neutral),
                             ("E325 actual", e325_actual)):
            e324._valid_array(arrays["q"], (300, 42), np.float64, name + " q")
            e324._valid_array(arrays["tendon"], (300, 6), np.float64,
                              name + " tendon")
        if (not np.array_equal(theta, theta.astype(np.float32).astype(np.float64))
                or not np.any(theta != 1)
                or result["continuation"]["metrics"]["shape_loss"]
                   >= data["e323_neutral_shape_loss"]):
            raise ValueError("E325 actual efficacy/shape invalid")
        data.update(e325_theta_minus=theta, e325_q=e325_actual["q"],
                    e325_tendon=e325_actual["tendon"],
                    e323_q=e323_neutral["q"], e323_tendon=e323_neutral["tendon"],
                    e323_neutral_shape_loss=float(data["e323_neutral_shape_loss"]),
                    e325_audited_result_sha256=freeze["input_sha256"][PATHS["e325_result"]])
        return data


def compute(data: dict, *, dependencies=None, device="cuda", checkpoint=None,
            deadline=None, on_receipt=None, on_progress=None,
            on_boundary=None) -> tuple[dict, dict]:
    """Select the midpoint once, then run only the continuous 300-control trace."""
    import torch

    n = N if dependencies is None else data["drives"].shape[1]
    groups_count = GROUPS if dependencies is None else len(data["e325_theta_minus"])
    if (data["drives"].shape != (300, n)
            or data["motor_rates"].shape != (300, 391)):
        raise ValueError("Exactly 300 fixed E318 controls required")
    theta1 = data["e325_theta_minus"]
    e324._valid_array(theta1, (groups_count,), np.float64, "E325 theta minus")
    if not np.array_equal(theta1, theta1.astype(np.float32).astype(np.float64)):
        raise ValueError("Actual E325 theta must be float32-derived")
    alpha = feasible_alpha(data["e323_q"][:99], data["e323_tendon"][:99],
                           data["e325_q"][:99], data["e325_tendon"][:99],
                           data["startup"]["target_command"],
                           data["startup"]["target_tendon"])
    ideal = 1 + alpha["alpha_star"] * (theta1 - 1)
    theta = np.float32(ideal).astype(np.float64)
    if (not np.isfinite(theta).all() or np.min(theta) <= 0
            or not np.any(theta != 1)):
        raise ValueError("E326 actual f32 efficacy is unchanged or invalid")
    selected = data["native"]["csr"]["data"][data["native"]["mask"]["positions"]]
    edge_groups = data["native"]["mask"]["edge_parameters"]
    if len(selected) != len(edge_groups) or not len(selected):
        raise ValueError("Current CSR selected edge/group mismatch")
    modified = np.float32(selected.astype(np.float64) * theta[edge_groups])
    weight_delta = modified.astype(np.float64) - selected.astype(np.float64)
    save = checkpoint or (lambda _name, _arrays: None)
    timely = deadline or (lambda: None)
    report_receipt = on_receipt or (lambda _name, _value: None)
    progress = on_progress or (lambda _value: None)
    boundary = on_boundary or (lambda _name, _event: None)
    result = {"conditional_fixed_neural_input": True, "steps_per_control": 20,
              "e325_audited_result_sha256": data.get("e325_audited_result_sha256"),
              "neutral_shape_loss": float(data["e323_neutral_shape_loss"]),
              "alpha": alpha,
              "direction": {"changed_groups": int(np.count_nonzero(theta != 1)),
                           "min": float(np.min(theta)), "max": float(np.max(theta)),
                           "ideal_f32_max_abs_difference": float(np.max(np.abs(theta - ideal))),
                           "selected_weight_changed_count": int(np.count_nonzero(weight_delta)),
                           "selected_weight_delta_min": float(np.min(weight_delta)),
                           "selected_weight_delta_max": float(np.max(weight_delta))},
              "continuation_requested": False, "continuation_completed": False,
              "continuation_supported": False, "decoder_replay_controls": 0,
              "body_calls": 0, "optimizer_steps": 0}
    receipts = {}
    save("constrained-step", {"source_theta_minus": theta1, "theta_ideal": ideal,
                              "theta": theta,
                              "alpha_max": np.asarray(alpha["alpha_max"], np.float64),
                              "alpha_star": np.asarray(alpha["alpha_star"], np.float64),
                              "predicted_q": data["e323_q"][:99] +
                              alpha["alpha_star"] * (data["e325_q"][:99] - data["e323_q"][:99]),
                              "predicted_tendon": data["e323_tendon"][:99] +
                              alpha["alpha_star"] * (data["e325_tendon"][:99]
                                                     - data["e323_tendon"][:99])})
    progress(result.copy())

    def marked(name, action):
        timely()
        boundary(name, "requested")
        try:
            value = action()
        except BaseException as error:
            try:
                boundary(name, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E326 phase journal failure: " + repr(secondary))
            raise
        boundary(name, "host_returned")
        return value

    deps = Dependencies() if dependencies is None else dependencies
    rollout, objective, initial = marked("construct", lambda: deps.build(data, device))
    r0, z0, bank0, original0, all_drives, mask = initial
    if all_drives.shape != (300, n):
        raise ValueError("Built E326 drive controls differ")
    name, active, receipt = "continuation-constrained", "continuation-constrained_forward", None
    try:
        timely()
        boundary(active, "requested")
        result["continuation_requested"] = True
        progress(result.copy())
        with torch.no_grad():
            output = rollout.run(r0, z0, bank0, original0, all_drives,
                                 torch.as_tensor(np.ascontiguousarray(theta),
                                                 device=rollout.device), mask)
        receipt = output["receipt"]
        boundary(active, "host_returned")
        receipts[name] = e324._receipt(receipt)
        report_receipt(name, receipts[name])
        arrays = {key: e324._host(output[key]) for key in
                  ("q", "tendon", "rate", "adaptation", "bank",
                   "original_activation")}
        save(name, arrays)
        receipt.confirm()
        receipts[name] = e324._receipt(receipt)
        report_receipt(name, receipts[name])
        boundary(active, "device_confirmed")
        active = None
        for key, dimensions, dtype in (("q", (300, 42), np.float64),
                                       ("tendon", (300, 6), np.float64),
                                       ("rate", (n,), np.float32),
                                       ("adaptation", (n,), np.float32),
                                       ("bank", (391, 3), np.float64),
                                       ("original_activation", (96,), np.float64)):
            e324._valid_array(arrays[key], dimensions, dtype, "E326 " + key)
        confirmed = receipts[name]["confirmed"]
        expected = {"forward_controls": 300, "forward_neural_ticks": 6000,
                    "recompute_controls": 0, "recompute_neural_ticks": 0,
                    "backward_blocks": 0, "backward_block_ticks": 0}
        if confirmed != expected:
            raise ValueError("E326 unexpected work receipt")
        metrics = e323._metrics(objective.evaluate(output["q"], output["tendon"]))
        actual = {"q": _metrics(arrays["q"][:99] - data["startup"]["target_command"]),
                  "tendon": _metrics(arrays["tendon"][:99]
                                     - data["startup"]["target_tendon"])}
        predicted_q = data["e323_q"][:99] + alpha["alpha_star"] * (
            data["e325_q"][:99] - data["e323_q"][:99])
        predicted_t = data["e323_tendon"][:99] + alpha["alpha_star"] * (
            data["e325_tendon"][:99] - data["e323_tendon"][:99])
        result["affine_prediction_error"] = {
            "q_max_abs": float(np.max(np.abs(arrays["q"][:99] - predicted_q))),
            "tendon_max_abs": float(np.max(np.abs(arrays["tendon"][:99] - predicted_t)))}
        result["actual_startup_metrics"] = actual
        result["metrics"] = metrics
        result["continuation_completed"] = True
        result["continuation"] = {"metrics": metrics, "receipt": receipts[name],
                                  "raw_file": name + ".npz"}
        result["actual_work"] = expected | {
            "decoder_replay_controls": 0, "body_calls": 0,
            "optimizer_steps": 0, "retry_count": 0}
        result["continuation_supported"] = bool(
            np.isfinite(metrics["shape_loss"])
            and metrics["shape_loss"] < result["neutral_shape_loss"]
            and metrics["startup_q_rms"] <= .018
            and metrics["startup_q_max"] <= .04
            and metrics["startup_tendon_rms"] <= .009
            and metrics["startup_tendon_max"] <= .025
            and metrics["startup_pass"]
            and actual["q"]["rms"] <= .018 and actual["q"]["max"] <= .04
            and actual["tendon"]["rms"] <= .009
            and actual["tendon"]["max"] <= .025)
        result["constrained_continuation_supported"] = result["continuation_supported"]
        progress(result.copy())
        timely()
    except BaseException as error:
        if active is not None:
            try:
                boundary(active, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E326 phase journal failure: " + repr(secondary))
        if receipt is None:
            receipt = getattr(error, "receipt", None)
        if receipt is not None:
            try:
                receipt.confirm()
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E326 receipt confirmation failure: " + repr(secondary))
            try:
                receipts[name] = e324._receipt(receipt)
                report_receipt(name, receipts[name])
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E326 receipt journal failure: " + repr(secondary))
        raise
    return result, receipts


def run(freeze_path: Path, freeze_sha256: str, review_path: Path,
        review_sha256: str, output: Path, *, root: Path = ROOT,
        gatekeeper=gate, dependencies=None, clock=time.perf_counter,
        writer=e324.put_json, saver=np.savez_compressed, meter=e323.usage,
        device="cuda") -> dict:
    """Passively gate, then journal a single original attempt and its publication."""
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path,
                        review_sha256, output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E326",
              "run_valid": False, "candidate_adopted": False,
              "biological_gait_proven": False, "F1_F8_complete": False,
              "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested", "conditional_fixed_neural_input": True,
              "continuation_requested": False,
              "continuation_completed": False,
              "continuation_supported": False,
              "constrained_continuation_supported": False,
              "provenance": {"freeze_sha256": freeze_sha256,
                             "review_sha256": review_sha256,
                             "source_manifest": "source-manifest.json",
                             "e325_result_sha256": freeze["input_sha256"][PATHS["e325_result"]],
                             "e325_audit_sha256": freeze["input_sha256"][PATHS["e325_audit"]],
                             "e325_direction_sha256": freeze["input_sha256"][E325 + "/direction.npz"],
                             "e325_actual_sha256": freeze["input_sha256"][E325 + "/continuation-minus.npz"],
                             "e323_neutral_sha256": freeze["input_sha256"][PATHS["e323_neutral"]]},
              "publication": {"measured_boundary": "before_terminal_metadata_commit",
                              "external_audit_pins_required": ["result.json",
                                                               "advance-ledger.json"],
                              "metadata_excluded_from_manifest": True},
              "errors": []}
    ledger = {"limits": LIMITS.copy(), "retry_count": 0, "body_calls": 0,
              "optimizer_steps": 0, "attempts": [], "phases": {},
              "receipts": {}, "errors": []}
    first_error = None
    baseline = None
    science_began = clock()

    def failure(stage, error):
        nonlocal first_error
        entry = {"stage": stage, "type": type(error).__name__,
                 "message": str(error)[:500]}
        ledger["errors"].append(entry)
        if first_error is None:
            first_error = entry

    def persist():
        ledger["elapsed_s"] = clock() - began
        writer(output / "advance-ledger.json", ledger)

    def check_resource(value):
        if baseline is None:
            return
        if value["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E326 device identity changed")
        ledger["resource_latest"] = value
        ledger["additional_gpu_peak_bytes"] = max(
            ledger.get("additional_gpu_peak_bytes", 0),
            baseline["device_free_bytes"] - value["device_free_bytes"])
        ledger["additional_torch_peak_reserved_bytes"] = max(
            ledger.get("additional_torch_peak_reserved_bytes", 0),
            value["torch_peak_reserved_bytes"] - baseline["torch_reserved_bytes"])
        if (value["host_peak_bytes"] > LIMITS["host_bytes"]
                or ledger["additional_gpu_peak_bytes"] > LIMITS["additional_gpu_bytes"]
                or ledger["additional_torch_peak_reserved_bytes"]
                > LIMITS["additional_gpu_bytes"]):
            raise MemoryError("E326 host/device resource limit exceeded")

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E326 internal deadline exceeded")
        if clock() - science_began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E326 science deadline exceeded")

    def checkpoint(name, arrays):
        path = output / (name + ".npz")
        attempt = {"stage": "save_" + name, "requested": True,
                   "returned": None, "requested_elapsed_s": clock() - began}
        ledger["attempts"].append(attempt)
        persist()
        saver(path, **arrays)
        attempt["returned"] = True
        attempt["bytes"] = path.stat().st_size
        attempt["returned_elapsed_s"] = clock() - began
        persist()

    try:
        persist()
        baseline = meter()
        ledger["resource_baseline"] = baseline
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e326.source-manifest.v1",
            "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
            "source_sha256": freeze["source_sha256"],
            "input_sha256": freeze["input_sha256"]})
        deps = Dependencies() if dependencies is None else dependencies
        ledger["phases"]["input_load"] = {
            "requested": True, "host_returned": None,
            "device_confirmed": None, "failed": False,
            "requested_elapsed_s": clock() - began}
        persist()
        try:
            data = deps.load(freeze, root=root)
        except BaseException as error:
            ledger["phases"]["input_load"]["failed"] = True
            try:
                persist()
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E326 input-load journal failure: " + repr(secondary))
            raise
        ledger["phases"]["input_load"]["host_returned"] = True
        ledger["phases"]["input_load"]["last_event_elapsed_s"] = clock() - began
        persist()
        deadline()
        check_resource(meter())

        def on_receipt(name, value):
            ledger["receipts"][name] = value
            persist()

        def on_boundary(name, event):
            phases = ledger["phases"]
            if event == "requested":
                if name in phases:
                    raise ValueError("E326 phase requested twice: " + name)
                phases[name] = {"requested": True, "host_returned": None,
                                "device_confirmed": None, "failed": False,
                                "requested_elapsed_s": clock() - began}
            elif event in ("host_returned", "device_confirmed"):
                phases[name][event] = True
            elif event == "failed":
                phases[name]["failed"] = True
            else:
                raise ValueError("Unknown E326 phase event")
            phases[name]["last_event_elapsed_s"] = clock() - began
            persist()
            if ((event == "host_returned" and name == "construct")
                    or event == "device_confirmed"):
                check_resource(meter())
                deadline()

        def on_progress(value):
            result["partial"] = value
            persist()

        computed, receipts = compute(data, dependencies=deps, checkpoint=checkpoint,
                                     deadline=deadline, on_receipt=on_receipt,
                                     on_progress=on_progress, on_boundary=on_boundary,
                                     device=device)
        deadline()
        ledger["receipts"] = receipts
        result.update(computed)
        result["classification"] = ("supported" if
                                    computed["constrained_continuation_supported"]
                                    else "rejected")
        check_resource(meter())
    except Exception as error:  # noqa: BLE001
        failure("science", error)
        result["science_error"] = traceback.format_exc(limit=8)[-3000:]
    finally:
        ledger["science_wall_s"] = clock() - science_began
        finalization_began = clock()
        try:
            check_resource(meter())
        except Exception as error:  # noqa: BLE001
            failure("terminal_resources", error)
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                writer(output / name, value)
            except Exception as error:  # noqa: BLE001
                failure("publish_" + name, error)
        try:
            files = {path.relative_to(output).as_posix(): {
                "sha256": e323.sha256(path), "bytes": path.stat().st_size}
                for path in output.rglob("*") if path.is_file()
                and path.name not in ("output-manifest.json", "advance-ledger.json",
                                      "result.json", "publication-failure.json")
                and not path.name.endswith(".pending")}
            total_bytes = sum(path.stat().st_size for path in output.rglob("*")
                              if path.is_file())
            if total_bytes > LIMITS["output_bytes"]:
                raise OSError("E326 output size limit exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e326.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"],
                "external_audit_pins_required": True})
            if not (output / "output-manifest.json").is_file():
                raise OSError("E326 output manifest was not published")
        except Exception as error:  # noqa: BLE001
            failure("output_manifest", error)
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            failure("finalization_deadline", TimeoutError("E326 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            failure("internal_deadline", TimeoutError("E326 internal deadline exceeded"))
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        result["run_valid"] = first_error is None
        if first_error is not None:
            result["constrained_continuation_supported"] = False
            result["continuation_supported"] = False
        try:
            writer(output / "advance-ledger.json", ledger)
            writer(output / "result.json", result)
            if (json.loads((output / "advance-ledger.json").read_text("utf8")) != ledger
                    or json.loads((output / "result.json").read_text("utf8")) != result):
                raise OSError("E326 terminal metadata differs from return value")
        except Exception as error:  # noqa: BLE001
            failure("terminal_publication", error)
            result.update(run_valid=False, continuation_supported=False,
                          constrained_continuation_supported=False,
                          first_error=first_error,
                          errors=ledger["errors"].copy(), cost=ledger.copy())
            try:
                writer(output / "publication-failure.json", result)
            except Exception as secondary:  # noqa: BLE001
                failure("publication_failure_journal", secondary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "review-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
