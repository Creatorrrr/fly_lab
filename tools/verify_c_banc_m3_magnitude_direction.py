"""E325: one magnitude-weighted direction using audited E324 gradients.

The original entrypoint is gated by a new reviewed freeze. It never computes
another gradient, changes the body, fits weights, or searches step sizes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tools import verify_c_banc_m3_full_horizon_direction as e323
from tools import verify_c_banc_m3_reset_window_direction as e324

ROOT, BASE = e324.ROOT, e324.BASE
N, GROUPS, CONTROLS, STARTS = e324.N, e324.GROUPS, e324.CONTROLS, e324.STARTS
E324 = e324.OUTPUT
OUTPUT = BASE + "/e325-magnitude-direction"
SCHEMA = "flylab.m3.e325.magnitude-direction.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e325.magnitude-direction.review.v1"
RESULT_SCHEMA = "flylab.m3.e325.magnitude-direction.result.v1"
OWNED = ("tools/verify_c_banc_m3_magnitude_direction.py",
         "tests/test_c_banc_magnitude_direction.py")
LIMITS = {"supervisor_wall_s": 140, "internal_wall_s": 120,
          "science_wall_s": 100, "finalization_reserve_s": 20,
          "host_bytes": 6 << 30, "additional_gpu_bytes": 8 << 30,
          "output_bytes": 2 << 30, "minimum_free_disk_bytes": 4 << 30,
          "forward_neural_ticks": 14000, "recompute_neural_ticks": 0,
          "backward_block_ticks": 0, "backward_blocks": 0,
          "decoder_replay_controls": 300, "body_calls": 0,
          "optimizer_steps": 0, "retry_count": 0}
PATHS = {**e324.PATHS,
         "e324_freeze": BASE + "/e324-reset-window-direction-freeze.json",
         "e324_review": BASE + "/e324-reset-window-direction-source-review.json",
         "e324_qualification": BASE + "/parent-e324-generated-qualification.json",
         "e324_result": E324 + "/result.json",
         "e324_audit_v2": BASE + "/parent-e324-original-raw-audit-v2.json",
         "e324_audit_v2_source": BASE + "/parent-e324-raw-audit-v2.py",
         "e324_source_manifest": E324 + "/source-manifest.json",
         "e324_output_manifest": E324 + "/output-manifest.json",
         "e324_ledger": E324 + "/advance-ledger.json",
         "e324_output": E324}


def gate(freeze_path: Path, freeze_sha256: str, review_path: Path,
         review_sha256: str, output: Path, *, root: Path = ROOT) -> dict:
    """Verify current source, all producer files and root audit without payloads."""
    if output.exists() or output.resolve() != e323.safe(OUTPUT, root):
        raise ValueError("Fresh literal E325 output required")
    if (e323.sha256(freeze_path) != freeze_sha256
            or e323.sha256(review_path) != review_sha256):
        raise ValueError("E325 freeze/review hash differs")
    freeze = json.loads(freeze_path.read_text("utf8"))
    review = json.loads(review_path.read_text("utf8"))
    if (freeze.get("schema") != SCHEMA or freeze.get("paths") != PATHS
            or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact E325 frozen diagnostic required")
    pins, sources = freeze.get("input_sha256"), freeze.get("source_sha256")
    if (not isinstance(pins, dict) or not isinstance(sources, dict)
            or not set(OWNED).issubset(sources)
            or not e323._file_roster(PATHS, root).issubset(pins)
            or any(name in pins and pins[name] != digest for name, digest in sources.items())
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("source_sha256") != sources):
        raise ValueError("Approved complete E325 source/input pins required")
    prior_freeze = json.loads(e323.safe(PATHS["e324_freeze"], root).read_text("utf8"))
    prior_review = json.loads(e323.safe(PATHS["e324_review"], root).read_text("utf8"))
    prior_qual = json.loads(e323.safe(PATHS["e324_qualification"], root).read_text("utf8"))
    prior_result = json.loads(e323.safe(PATHS["e324_result"], root).read_text("utf8"))
    prior_audit = json.loads(e323.safe(PATHS["e324_audit_v2"], root).read_text("utf8"))
    prior_source = json.loads(e323.safe(PATHS["e324_source_manifest"], root).read_text("utf8"))
    prior_manifest = json.loads(e323.safe(PATHS["e324_output_manifest"], root).read_text("utf8"))
    if (prior_freeze.get("schema") != e324.SCHEMA
            or prior_freeze.get("paths") != e324.PATHS
            or prior_freeze.get("source_sha256") != prior_source.get("source_sha256")
            or prior_source.get("schema") != "flylab.m3.e324.source-manifest.v1"
            or prior_source.get("input_sha256") != prior_freeze.get("input_sha256")
            or prior_source.get("freeze_sha256") != pins[PATHS["e324_freeze"]]
            or prior_source.get("review_sha256") != pins[PATHS["e324_review"]]
            or prior_review.get("schema") != e324.REVIEW_SCHEMA
            or prior_review.get("approved") is not True
            or prior_review.get("blocking_defects") != []
            or prior_review.get("freeze_sha256") != pins[PATHS["e324_freeze"]]
            or prior_review.get("source_sha256") != prior_freeze["source_sha256"]
            or prior_qual.get("schema") != "flylab.m3.e324.generated-qualification.v1"
            or prior_qual.get("approved") is not True
            or prior_qual.get("blocking_defects") != []
            or prior_qual.get("original_calls") != 0
            or any(prior_freeze["source_sha256"].get(name) != digest for name, digest
                   in prior_qual.get("source_sha256", {}).items())
            or any(prior_freeze["source_sha256"].get(name) != digest for name, digest
                   in prior_qual.get("evidence_sha256", {}).items())
            or prior_result.get("schema") != e324.RESULT_SCHEMA
            or prior_result.get("run_valid") is not True
            or prior_result.get("classification") != "supported"
            or prior_result.get("comparison", {}).get("classification") != "supported"
            or prior_result.get("continuation_requested") is not True
            or prior_result.get("continuation_supported") is not False
            or prior_result.get("actual_work", {}).get("forward_neural_ticks") != 18000
            or prior_result.get("actual_work", {}).get("backward_block_ticks") != 4000
            or prior_audit.get("schema") != "flylab.m3.e324.root-original-raw-audit.v2"
            or prior_audit.get("pass") is not True
            or prior_audit.get("diagnostic_run_valid") is not True
            or prior_audit.get("verdict") != "supported"
            or prior_audit.get("continuation_supported") is not False
            or prior_audit.get("audit_source_sha256") != pins[
                PATHS["e324_audit_v2_source"]]
            or prior_audit.get("result_sha256") != pins[PATHS["e324_result"]]
            or prior_audit.get("advance_ledger_sha256") != pins[PATHS["e324_ledger"]]
            or prior_audit.get("output_manifest_sha256") != pins[PATHS["e324_output_manifest"]]
            or prior_manifest.get("schema") != "flylab.m3.e324.output-manifest.v1"):
        raise ValueError("Audited E324 producer proof required")
    audited_metrics = prior_audit.get("continuation_metrics", {})
    saved_metrics = prior_result.get("continuation", {}).get("metrics", {})
    if (prior_result.get("window_starts") != list(STARTS)
            or not np.isfinite(prior_result.get("neutral_shape_loss", np.nan))
            or abs(prior_result["neutral_shape_loss"] - prior_audit["neutral_shape_loss"]) > 1e-12
            or any(not np.isfinite(prior_result["comparison"].get(key, np.nan))
                   or abs(prior_result["comparison"][key] - prior_audit[key]) > 1e-12
                   for key in ("observed", "predicted", "relative_error"))
            or audited_metrics.get("startup_pass") is not False
            or saved_metrics.get("startup_pass") is not False
            or any(not np.isfinite(saved_metrics.get(key, np.nan))
                   or abs(saved_metrics[key] - audited_metrics[key]) > 1e-12
                   for key in ("shape_loss", "startup_q_rms", "startup_q_max",
                               "startup_tendon_rms", "startup_tendon_max"))):
        raise ValueError("E324 complete 50-window neutral source required")
    for name, digest in prior_freeze["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError("E324 source closure differs: " + name)
    for name, digest in prior_freeze["input_sha256"].items():
        if pins.get(name) != digest:
            raise ValueError("E324 input closure differs: " + name)
    for name, digest in {**pins, **sources}.items():
        if not isinstance(digest, str) or e323.sha256(e323.safe(name, root)) != digest:
            raise ValueError("E325 source/input differs: " + name)
    prior_dir = e323.safe(E324, root)
    expected_files = {path.relative_to(prior_dir).as_posix() for path in prior_dir.rglob("*")
                      if path.is_file() and path.name not in
                      ("result.json", "advance-ledger.json", "output-manifest.json")}
    if set(prior_manifest.get("files", {})) != expected_files:
        raise ValueError("E324 producer output roster differs")
    for relative in expected_files:
        path = prior_dir / relative
        if prior_manifest["files"][relative] != {
                "sha256": pins[path.relative_to(root).as_posix()],
                "bytes": path.stat().st_size}:
            raise ValueError("E324 producer output file differs: " + relative)
    required = e323._file_roster(PATHS, root) | set(prior_freeze["input_sha256"])
    if set(pins) != required:
        raise ValueError("Exact E325 input pin roster required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E325 free disk reserve unavailable")
    return freeze


class Dependencies(e324.Dependencies):
    """Load one E318 original dataset and all pinned E324 neutral evidence."""

    def load(self, freeze: dict, root: Path = ROOT) -> dict:
        from tools.verify_c_banc_m3_native_adjoint import load_npz

        data = super().load(freeze, root=root)
        prior_dir = e323.safe(E324, root)
        prior_result = json.loads(e323.safe(PATHS["e324_result"], root).read_text("utf8"))
        direction = load_npz(prior_dir / "direction.npz")
        gradient = direction["gradient"]
        windows = direction["window_gradients"]
        e324._valid_array(gradient, (GROUPS,), np.float64, "E324 gradient")
        e324._valid_array(windows, (50, GROUPS), np.float64, "E324 window gradients")
        if not np.array_equal(gradient, np.sum(windows, axis=0, dtype=np.float64)):
            raise ValueError("E324 shared gradient differs from 50 saved contributions")
        losses = []
        for index, c in enumerate(STARTS):
            name = f"window-{c:03d}-neutral"
            gradient_file = load_npz(prior_dir / (name + "-gradient.npz"))
            if not e323.exact(gradient_file["gradient"], windows[index]):
                raise ValueError("E324 saved window gradient differs: " + name)
            raw = load_npz(prior_dir / (name + ".npz"))
            for key, dimensions, dtype in (("q", (4, 42), np.float64),
                                           ("tendon", (4, 6), np.float64),
                                           ("rate", (N,), np.float32),
                                           ("adaptation", (N,), np.float32),
                                           ("bank", (391, 3), np.float64),
                                           ("original_activation", (96,), np.float64)):
                e324._valid_array(raw[key], dimensions, dtype, name + " " + key)
            if (not np.max(np.abs(raw["q"] - data["raw"]["command"][c + 1:c + 5])) <= 1e-11
                    or not np.max(np.abs(raw["tendon"]
                                          - data["raw"]["tendon"][c + 1:c + 5])) <= 1e-11
                    or not e323.exact(raw["rate"], data["rates"][c + 3])
                    or not e323.exact(raw["adaptation"], data["adaptations"][c + 3])):
                raise ValueError("E324 neutral original raw differs: " + name)
            row = prior_result["runs"][name]
            reproduction = row.get("neutral_reproduction", {})
            if (reproduction.get("end_rate_adaptation_dtype_bytes_exact") is not True
                    or any(reproduction.get(key, np.inf) > limit for key, limit in
                           (("q_max_abs_rad", 1e-11),
                            ("tendon_max_abs_command", 1e-11),
                            ("bank_max_abs", 1e-10),
                            ("original_max_abs", 1e-10)))
                    or row.get("receipt", {}).get("confirmed", {}).get(
                        "backward_block_ticks") != 80):
                raise ValueError("E324 neutral receipt/reproduction differs: " + name)
            losses.append(row["shape_loss"])
        losses = np.asarray(losses, dtype=np.float64)
        e324._valid_array(losses, (50,), np.float64, "E324 neutral losses")
        neutral = prior_result["neutral_shape_loss"]
        if abs(float(np.mean(losses)) - neutral) > 1e-12:
            raise ValueError("E324 mean neutral loss differs")
        data.update(e324_gradient=gradient, e324_window_gradients=windows,
                    e324_neutral_losses=losses,
                    e324_neutral_shape_loss=float(neutral),
                    e324_audited_result_sha256=freeze["input_sha256"][PATHS["e324_result"]])
        return data


def _counters(receipts: dict, replay_controls: int, continuation: bool,
              windows: int = 50) -> dict:
    if len(receipts) != 2 * windows + int(continuation):
        raise ValueError("E325 complete receipt roster required")
    rows = []
    for name, receipt in receipts.items():
        confirmed = receipt["confirmed"]
        if confirmed is None:
            raise ValueError("E325 unconfirmed neural work: " + name)
        if name == "continuation-minus":
            controls = 300
        elif name.endswith(("-plus", "-minus")):
            controls = 4
        else:
            raise ValueError("Unknown E325 receipt: " + name)
        expected = {"forward_controls": controls, "forward_neural_ticks": 20 * controls,
                    "recompute_controls": 0, "recompute_neural_ticks": 0,
                    "backward_blocks": 0, "backward_block_ticks": 0}
        if confirmed != expected:
            raise ValueError("E325 unexpected neural/backward work: " + name)
        rows.append(confirmed)
    totals = {key: sum(row[key] for row in rows) for key in
              ("forward_controls", "forward_neural_ticks", "recompute_controls",
               "recompute_neural_ticks", "backward_blocks", "backward_block_ticks")}
    if (totals["forward_controls"] != 8 * windows + (300 if continuation else 0)
            or replay_controls != 300):
        raise ValueError("E325 exact forward/replay work differs")
    totals.update(decoder_replay_controls=replay_controls, body_calls=0,
                  optimizer_steps=0, retry_count=0)
    for key in ("forward_neural_ticks", "recompute_neural_ticks", "backward_blocks",
                "backward_block_ticks", "decoder_replay_controls"):
        if totals[key] > LIMITS[key]:
            raise ValueError("E325 work cap exceeded: " + key)
    return totals


def compute(data: dict, *, dependencies=None, device="cuda", checkpoint=None,
            deadline=None, on_receipt=None, on_progress=None, on_boundary=None,
            starts: tuple[int, ...] = STARTS,
            short_objective: Callable | None = None) -> tuple[dict, dict]:
    """Run only fixed plus/minus windows with the pinned E324 gradient."""
    import torch

    if (not starts or any(type(c) is not int or c < 100 or c > 296 for c in starts)
            or tuple(sorted(set(starts))) != starts
            or dependencies is None and starts != STARTS):
        raise ValueError("Fixed ordered E325 starts required")
    if (data["drives"].shape != (300, N)
            or data["motor_rates"].shape != (300, 391)
            or data["rates"].shape != (300, N)
            or data["adaptations"].shape != (300, N)):
        raise ValueError("E325 requires all 300 recorded original controls")
    gradient = data["e324_gradient"]
    window_gradients = data["e324_window_gradients"]
    neutral_losses = data["e324_neutral_losses"]
    e324._valid_array(gradient, (GROUPS,), np.float64, "E324 aggregated gradient")
    e324._valid_array(window_gradients, (len(starts), GROUPS), np.float64,
                      "E324 window gradients")
    e324._valid_array(neutral_losses, (len(starts),), np.float64,
                      "E324 neutral window losses")
    if (not e323.exact(gradient, np.sum(window_gradients, axis=0, dtype=np.float64))
            or not np.isfinite(data["e324_neutral_shape_loss"])
            or abs(float(np.mean(neutral_losses)) - data["e324_neutral_shape_loss"]) > 1e-12
            or not np.any(gradient)):
        raise ValueError("Audited E324 gradient/neutral evidence differs")
    deps = Dependencies() if dependencies is None else dependencies
    save = checkpoint or (lambda _name, _arrays: None)
    timely = deadline or (lambda: None)
    report_receipt = on_receipt or (lambda _name, _value: None)
    progress = on_progress or (lambda _value: None)
    boundary = on_boundary or (lambda _name, _event: None)
    shape = short_objective or e324.short_shape
    receipts = {}
    result = {"conditional_fixed_neural_input": True,
              "reused_e324_neutral_and_gradient": True,
              "e324_audited_result_sha256": data.get("e324_audited_result_sha256"),
              "window_starts": list(starts), "window_controls": 4,
              "steps_per_control": 20, "direction_count": 1,
              "body_calls": 0, "optimizer_steps": 0,
              "decoder_replay_controls": 0,
              "decoder_replay_work": {"requested_controls": 0,
                                      "host_returned_controls": 0,
                                      "device_confirmed_controls": None},
              "neutral_shape_loss": float(data["e324_neutral_shape_loss"]),
              "windows": [], "runs": {}}

    def marked(name, action):
        timely()
        boundary(name, "requested")
        try:
            value = action()
        except BaseException as error:
            try:
                boundary(name, "failed")
            except Exception as secondary:  # noqa: BLE001
                error.add_note("E325 phase journal failure: " + repr(secondary))
            raise
        boundary(name, "host_returned")
        return value

    def retain_failure_receipt(name, receipt, error):
        if receipt is None:
            return
        try:
            receipt.confirm()
        except Exception as secondary:  # noqa: BLE001 - preserve neural error
            error.add_note("E325 receipt confirmation failure: " + repr(secondary))
        try:
            receipts[name] = e324._receipt(receipt)
            report_receipt(name, receipts[name])
        except Exception as secondary:  # noqa: BLE001 - preserve neural error
            error.add_note("E325 receipt journal failure: " + repr(secondary))

    rollout, objective, initial = marked("construct", lambda: deps.build(data, device))
    r0, z0, bank0, original0, all_drives, mask = initial
    if (all_drives.shape != (300, N) or len(data["motor_rates"]) != 300):
        raise ValueError("Built E325 drive/MN controls differ")

    def replay():
        banks, originals, q_rows, tendon_rows = [bank0], [original0], [], []
        with torch.no_grad():
            for control, motor in enumerate(data["motor_rates"], start=1):
                result["decoder_replay_work"]["requested_controls"] = control
                decoded = rollout.decoder.step(
                    torch.as_tensor(np.ascontiguousarray(motor), device=rollout.device),
                    banks[-1], originals[-1], dt=.005)
                banks.append(decoded["bank"])
                originals.append(decoded["original_activation"])
                q_rows.append(decoded["q"])
                tendon_rows.append(decoded["tendon"])
                result["decoder_replay_controls"] = control
                result["decoder_replay_work"]["host_returned_controls"] = control
        return banks, originals, torch.stack(q_rows), torch.stack(tendon_rows)

    try:
        banks, originals, replay_q, replay_tendon = marked("decoder_replay", replay)
        progress(result.copy())
        replay_arrays = {"q": e324._host(replay_q),
                         "tendon": e324._host(replay_tendon),
                         "bank100": e324._host(banks[100]),
                         "bank300": e324._host(banks[300]),
                         "original100": e324._host(originals[100]),
                         "original300": e324._host(originals[300])}
        save("decoder-replay", replay_arrays)
    except BaseException as error:
        try:
            progress(result.copy())
        except Exception as secondary:  # noqa: BLE001
            error.add_note("E325 replay progress journal failure: " + repr(secondary))
        raise
    result["decoder_replay_work"]["device_confirmed_controls"] = 300
    progress(result.copy())
    boundary("decoder_replay", "device_confirmed")
    for key, dimensions, raw_key in (("q", (300, 42), "command"),
                                     ("tendon", (300, 6), "tendon")):
        e324._valid_array(replay_arrays[key], dimensions, np.float64,
                          "E325 decoder " + key)
        e324._valid_array(data["raw"][raw_key], (301, dimensions[1]),
                          np.float64, "E318 raw " + key)
    q_error = float(np.max(np.abs(replay_arrays["q"]
                                  - data["raw"]["command"][1:])))
    tendon_error = float(np.max(np.abs(replay_arrays["tendon"]
                                       - data["raw"]["tendon"][1:])))
    state_errors = {}
    for control, whole in ((100, data["whole100"]), (300, data["final"])):
        for key, dimensions in (("bank", (391, 3)), ("original", (96,))):
            value = replay_arrays[key + str(control)]
            expected = (whole["adapter"]["bank"] if key == "bank" else
                        whole["adapter"]["original"]["activation"])
            e324._valid_array(value, dimensions, np.float64, "replayed " + key)
            e324._valid_array(expected, dimensions, np.float64, "E318 " + key)
            state_errors[key + str(control)] = float(np.max(np.abs(value - expected)))
    result["decoder_reproduction"] = {"q_max_abs_rad": q_error,
                                      "tendon_max_abs_command": tendon_error,
                                      **{key + "_max_abs": value for key, value
                                         in state_errors.items()}}
    progress(result.copy())
    if (q_error > 1e-11 or tendon_error > 1e-11
            or any(value > 1e-10 for value in state_errors.values())):
        raise ValueError("E325 original decoder replay reproduction failed")
    timely()

    direction = gradient / np.max(np.abs(gradient))
    plus = np.float32(1 + .001 * direction).astype(np.float64)
    minus = np.float32(1 - .001 * direction).astype(np.float64)
    if (not np.isfinite(plus).all() or not np.isfinite(minus).all()
            or not np.any(plus != 1) or not np.any(minus != 1)
            or plus.min() <= 0 or minus.min() <= 0):
        raise ValueError("E325 f32 magnitude efficacy direction invalid")
    save("direction", {"gradient": gradient, "direction": direction,
                       "theta_plus": plus, "theta_minus": minus,
                       "window_gradients": window_gradients,
                       "neutral_window_losses": neutral_losses})
    selected = data["native"]["csr"]["data"][data["native"]["mask"]["positions"]]
    groups = data["native"]["mask"]["edge_parameters"]
    if len(selected) != len(groups) or not len(selected):
        raise ValueError("E325 selected native edge/group mismatch")
    deltas = {}
    for label, theta in (("plus", plus), ("minus", minus)):
        modified = np.float32(selected.astype(np.float64) * theta[groups])
        delta = modified.astype(np.float64) - selected.astype(np.float64)
        deltas[label] = {"changed_count": int(np.count_nonzero(delta)),
                         "min": float(delta.min()), "max": float(delta.max())}
    result["direction"] = {"normalization": "gradient/max_abs_gradient",
                           "gradient_max_abs": float(np.max(np.abs(gradient))),
                           "nonzero_gradient_groups": int(np.count_nonzero(gradient)),
                           "theta_plus_changed_groups": int(np.count_nonzero(plus != 1)),
                           "theta_minus_changed_groups": int(np.count_nonzero(minus != 1)),
                           "theta_plus_unchanged_groups": int(np.count_nonzero(plus == 1)),
                           "theta_minus_unchanged_groups": int(np.count_nonzero(minus == 1)),
                           "theta_plus_min": float(plus.min()),
                           "theta_plus_max": float(plus.max()),
                           "theta_minus_min": float(minus.min()),
                           "theta_minus_max": float(minus.max()),
                           "selected_weight_delta": deltas}
    progress(result.copy())
    tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value), device=rollout.device)

    def start_state(c):
        if c == 0:
            return r0, z0, bank0, original0
        return (tensor(data["rates"][c - 1]),
                tensor(data["adaptations"][c - 1]), banks[c], originals[c])

    def perform(c, label, efficacy):
        name = f"window-{c:03d}-{label}"
        r, z, bank, original = start_state(c)
        receipt = None
        active = name + "_forward"
        try:
            timely()
            boundary(active, "requested")
            with torch.no_grad():
                output = rollout.run(r, z, bank, original,
                                     all_drives[c:c + 4], efficacy, mask)
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
            for key, dimensions, dtype in (("q", (4, 42), np.float64),
                                           ("tendon", (4, 6), np.float64),
                                           ("rate", (N,), np.float32),
                                           ("adaptation", (N,), np.float32),
                                           ("bank", (391, 3), np.float64),
                                           ("original_activation", (96,), np.float64)):
                e324._valid_array(arrays[key], dimensions, dtype, name + " " + key)
            loss, per_leg = shape(output["q"], objective.curves)
            loss_value = float(e324._host(loss))
            legs = e324._host(per_leg)
            if not np.isfinite(loss_value) or not np.isfinite(legs).all():
                raise ValueError("Nonfinite E325 short shape")
            result["runs"][name] = {"start_control": c, "label": label,
                                    "shape_loss": loss_value,
                                    "per_leg_shape_loss": legs.tolist(),
                                    "receipt": e324._receipt(receipt),
                                    "raw_file": name + ".npz"}
            progress(result.copy())
            timely()
            return loss_value
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E325 phase journal failure: " + repr(secondary))
            if receipt is None:
                receipt = getattr(error, "receipt", None)
            retain_failure_receipt(name, receipt, error)
            raise

    plus_losses, minus_losses = [], []
    for index, c in enumerate(starts):
        plus_loss = perform(c, "plus", tensor(plus))
        minus_loss = perform(c, "minus", tensor(minus))
        plus_losses.append(plus_loss)
        minus_losses.append(minus_loss)
        observed = (plus_loss - minus_loss) / len(starts)
        predicted = float(np.sum(window_gradients[index] * (plus - minus),
                                 dtype=np.float64))
        result["windows"].append({"start_control": c,
                                  "reused_neutral_shape_loss": float(neutral_losses[index]),
                                  "plus_shape_loss": plus_loss,
                                  "minus_shape_loss": minus_loss,
                                  "observed_contribution": observed,
                                  "predicted_contribution": predicted,
                                  "e324_gradient_file": f"window-{c:03d}-neutral-gradient.npz"})
        progress(result.copy())
    plus_mean = float(np.mean(plus_losses, dtype=np.float64))
    minus_mean = float(np.mean(minus_losses, dtype=np.float64))
    observed = plus_mean - minus_mean
    predicted = float(np.sum(gradient * (plus - minus), dtype=np.float64))
    comparison = e323.classify(observed, predicted)
    result.update(plus_shape_loss=plus_mean, minus_shape_loss=minus_mean,
                  comparison=comparison,
                  minus_improves_neutral=bool(minus_mean < result["neutral_shape_loss"]),
                  continuation_requested=False, continuation_completed=False)
    progress(result.copy())
    if (comparison["classification"] == "supported"
            and result["minus_improves_neutral"]):
        name = "continuation-minus"
        receipt = None
        active = name + "_forward"
        try:
            timely()
            boundary(active, "requested")
            result["continuation_requested"] = True
            progress(result.copy())
            with torch.no_grad():
                output = rollout.run(r0, z0, bank0, original0,
                                     all_drives, tensor(minus), mask)
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
                                           ("rate", (N,), np.float32),
                                           ("adaptation", (N,), np.float32),
                                           ("bank", (391, 3), np.float64),
                                           ("original_activation", (96,), np.float64)):
                e324._valid_array(arrays[key], dimensions, dtype,
                                  "continuation " + key)
            metrics = e323._metrics(objective.evaluate(output["q"], output["tendon"]))
            result["continuation_completed"] = True
            result["continuation"] = {"metrics": metrics,
                                      "receipt": e324._receipt(receipt),
                                      "raw_file": name + ".npz"}
            result["continuation_supported"] = bool(
                metrics["shape_loss"] < result["neutral_shape_loss"]
                and metrics["startup_q_rms"] <= .018
                and metrics["startup_q_max"] <= .04
                and metrics["startup_tendon_rms"] <= .009
                and metrics["startup_tendon_max"] <= .025
                and metrics["startup_pass"])
            progress(result.copy())
            timely()
        except BaseException as error:
            if active is not None:
                try:
                    boundary(active, "failed")
                except Exception as secondary:  # noqa: BLE001
                    error.add_note("E325 phase journal failure: " + repr(secondary))
            if receipt is None:
                receipt = getattr(error, "receipt", None)
            retain_failure_receipt(name, receipt, error)
            raise
    else:
        result["continuation_supported"] = None
    result["actual_work"] = _counters(receipts, result["decoder_replay_controls"],
                                      result["continuation_requested"], len(starts))
    return result, receipts


def run(freeze_path: Path, freeze_sha256: str, review_path: Path,
        review_sha256: str, output: Path, *, root: Path = ROOT,
        gatekeeper=gate, dependencies=None, clock=time.perf_counter,
        writer=e324.put_json, saver=np.savez_compressed, meter=e323.usage,
        device="cuda") -> dict:
    """Passively gate once, then journal all original work and publication."""
    began = clock()
    freeze = gatekeeper(freeze_path, freeze_sha256, review_path,
                        review_sha256, output, root=root)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E325",
              "run_valid": False, "candidate_adopted": False,
              "biological_gait_proven": False, "F1_F8_complete": False,
              "F6_exposed": False, "M2_modified": False,
              "classification": "not_tested",
              "conditional_fixed_neural_input": True,
              "provenance": {"freeze_sha256": freeze_sha256,
                             "review_sha256": review_sha256,
                             "source_manifest": "source-manifest.json",
                             "e324_result_sha256": freeze["input_sha256"][PATHS["e324_result"]],
                             "e324_audit_v2_sha256": freeze["input_sha256"][PATHS["e324_audit_v2"]],
                             "e324_direction_sha256": freeze["input_sha256"][E324 + "/direction.npz"]},
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
        return entry

    def persist():
        ledger["elapsed_s"] = clock() - began
        writer(output / "advance-ledger.json", ledger)

    def check_resource(value):
        if baseline is None:
            return
        if value["device_total_bytes"] != baseline["device_total_bytes"]:
            raise RuntimeError("E325 device identity changed")
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
            raise MemoryError("E325 host/device resource limit exceeded")

    def deadline():
        if clock() - began >= LIMITS["internal_wall_s"]:
            raise TimeoutError("E325 internal deadline exceeded")
        if clock() - science_began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E325 science deadline exceeded")

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
        # Torch CUDA initializes before E315/CuPy can enter through the loader.
        baseline = meter()
        ledger["resource_baseline"] = baseline
        writer(output / "source-manifest.json", {
            "schema": "flylab.m3.e325.source-manifest.v1",
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
            except Exception as secondary:  # noqa: BLE001 - keep original loader error
                error.add_note("E325 input-load journal failure: " + repr(secondary))
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
                    raise ValueError("E325 phase was requested twice: " + name)
                phases[name] = {"requested": True, "host_returned": None,
                                "device_confirmed": None, "failed": False,
                                "requested_elapsed_s": clock() - began}
            elif event in ("host_returned", "device_confirmed"):
                phases[name][event] = True
            elif event == "failed":
                phases[name]["failed"] = True
            else:
                raise ValueError("Unknown E325 phase event")
            phases[name]["last_event_elapsed_s"] = clock() - began
            persist()
            if ((event == "host_returned" and name == "construct")
                    or event == "device_confirmed"):
                check_resource(meter())
                deadline()

        def on_progress(value):
            result["partial"] = value
            if "decoder_replay_work" in value:
                ledger["decoder_replay_work"] = value["decoder_replay_work"].copy()
            persist()

        computed, receipts = compute(data, dependencies=deps, checkpoint=checkpoint,
                                     deadline=deadline, on_receipt=on_receipt,
                                     on_progress=on_progress, on_boundary=on_boundary,
                                     device=device)
        deadline()
        ledger["receipts"] = receipts
        result.update(computed)
        result["classification"] = computed["comparison"]["classification"]
        check_resource(meter())
    except Exception as error:  # noqa: BLE001 - preserve primary diagnostic failure
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
                raise OSError("E325 output size limit exceeded")
            writer(output / "output-manifest.json", {
                "schema": "flylab.m3.e325.output-manifest.v1", "files": files,
                "mutable_metadata_excluded": ["result.json", "advance-ledger.json",
                                              "publication-failure.json"],
                "external_audit_pins_required": True})
            if not (output / "output-manifest.json").is_file():
                raise OSError("E325 output manifest was not published")
            if sum(path.stat().st_size for path in output.rglob("*")
                   if path.is_file()) > LIMITS["output_bytes"]:
                raise OSError("E325 published output size limit exceeded")
        except Exception as error:  # noqa: BLE001
            failure("output_manifest", error)
        ledger["publication_boundary"] = "before_terminal_metadata_commit"
        ledger["finalization_wall_s"] = clock() - finalization_began
        ledger["elapsed_s"] = clock() - began
        if ledger["finalization_wall_s"] > LIMITS["finalization_reserve_s"]:
            failure("finalization_deadline", TimeoutError("E325 finalization reserve exceeded"))
        if ledger["elapsed_s"] >= LIMITS["internal_wall_s"]:
            failure("internal_deadline", TimeoutError("E325 internal deadline exceeded"))
        result["first_error"] = first_error
        result["errors"] = ledger["errors"].copy()
        result["cost"] = ledger.copy()
        result["run_valid"] = first_error is None
        try:
            writer(output / "advance-ledger.json", ledger)
            writer(output / "result.json", result)
            if (json.loads((output / "advance-ledger.json").read_text("utf8")) != ledger
                    or json.loads((output / "result.json").read_text("utf8")) != result):
                raise OSError("E325 terminal metadata differs from return value")
        except Exception as error:  # noqa: BLE001
            failure("terminal_publication", error)
            result.update(run_valid=False, first_error=first_error,
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
