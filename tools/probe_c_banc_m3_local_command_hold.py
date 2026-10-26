"""E228: preregistered, body-only RF/RH command holds at one saved break."""

# Preserve the first physical, resource, or storage failure across cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.storage import StateStore
from tools import analyze_c_banc_observation as observation
from tools import observe_c_banc_m3_effector_force_replay as replay
from tools import observe_c_banc_m3_ground_reaction as ground
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e228-local-command-hold"
FREEZE_SHA = "10c079d5635dcb48a77620e93303d82a525917ccc25f2e57fc43bad51b11b191"
FREEZE_SCHEMA = "flylab.m3.e228.local-command-hold.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e228.local-command-hold.review.v1"
OWNED = ("tools/probe_c_banc_m3_local_command_hold.py",
         "tests/test_c_banc_local_command_hold.py")
ARMS = ("sham", "rf_hold", "rh_hold")
START, END, ANCHOR = 1607, 1647, 1619
HOLD_FIRST, HOLD_LAST = 1620, 1635
DT = .005
STEPS = 50
NATIVE_FIELDS = ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid",
                 "nonfoot_contact", "upright")


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    """Check every frozen byte and limit before construction or replay."""
    if (freeze_sha != FREEZE_SHA or replay.sha256(freeze_path) != FREEZE_SHA
            or replay.sha256(review_path) != review_sha):
        raise ValueError("Exact E228 freeze/review hashes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or replay.sha256(replay.safe(freeze["design"])) != freeze.get("design_sha256")
            or freeze.get("arms") != list(ARMS)
            or freeze.get("event") != {
                "event_id": 25, "start_control": START, "end_control": END,
                "entry_control": 1623, "exit_control": 1631,
                "anchor_control": ANCHOR, "hold_start_control": HOLD_FIRST,
                "hold_end_control": HOLD_LAST,
                "rf_joint_indices": list(range(21, 28)),
                "rh_joint_indices": list(range(35, 42)),
                "velocity_summary_controls": list(range(1620, 1647))}
            or freeze.get("phases") != {"lead": [1608, 1620],
                "hold": [1620, 1636], "recovery": [1636, 1648],
                "during_break": [1623, 1631]}
            or freeze.get("native") != {"control_dt_s": DT,
                "physics_dt_s": .0001, "steps_per_control": STEPS,
                "controls_per_arm": 40, "native_samples_per_arm": 2000,
                "max_scientific_steps": 6000, "new_neural_ticks": 0,
                "support_threshold_bw": .02}
            or freeze.get("diagnostic_thresholds") != {
                "backward_distance_reduction_fraction": .25,
                "minimum_forward_speed_gain_mm_s": .5,
                "maximum_forward_distance_loss_mm": .02,
                "rf_negative_integral_reduction_fraction": .2}
            or freeze.get("force_fields") != ["time_s", "available", "error",
                *replay.FORCE_SHAPES]
            or freeze.get("resource_limits") != {"cpu_threads": 1,
                "gpu_count": 0, "original_attempts": 1, "retries": 0,
                "body_constructors": 1, "fit_calls": 0, "solver_calls": 0,
                "internal_s": 25., "finalization_reserve_s": 5.,
                "external_s": 30., "supervisor_s": 35.,
                "resident_bytes": 6442450944,
                "output_bytes": 33554432,
                "minimum_free_bytes": 2147483648}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E228 fixed arm, science, or resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Parent-approved E228 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E228 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Frozen E228 source/input changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E228 hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Reviewed E228 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Frozen E228 package version changed: " + package)
    if out != OUT.resolve() or out.exists() or not out.parent.is_dir():
        raise FileExistsError("Exact new E228 output directory required")
    if shutil.disk_usage(out.parent).free < freeze["resource_limits"][
            "minimum_free_bytes"]:
        raise OSError("E228 output disk reserve unavailable")
    return freeze, review


def load_reference(freeze):
    paths = {key: replay.safe(name) for key, name in freeze["paths"].items()}
    trace = json.loads(paths["reference_trace"].read_text(encoding="utf-8"))["samples"]
    with np.load(paths["reference_native"], allow_pickle=False) as archive:
        native = {key: archive[key].copy() for key in archive.files}
    with np.load(paths["reference_force"], allow_pickle=False) as archive:
        force = {key: archive[key].copy() for key in archive.files}
    if (len(trace) != 1664 or set(native) != set(NATIVE_FIELDS)
            or set(force) != set(freeze["force_fields"])
            or any(len(value) != 53150 for value in native.values())
            or any(len(value) != 53150 for value in force.values())
            or not np.all(force["available"])
            or not np.array_equal(force["time_s"], native["time_s"])
            or any(trace[tick]["control_tick"] != tick or not np.isclose(
                trace[tick]["time_s"], tick * DT, atol=2e-8, rtol=0)
                for tick in range(START, END + 1))
            or not np.allclose(native["time_s"],
                3. + (np.arange(53150) + 1) * .0001, atol=2e-8, rtol=0)):
        raise ValueError("Exact frozen E225/E227 trace/native/force required")
    if any(np.asarray(trace[tick]["joint_targets_rad"]).shape != (42,)
           or np.asarray(trace[tick]["adhesion"]).shape != (6,)
           or any(trace[tick]["adhesion"])
           or len(trace[tick]["tendon_inputs"]) != 8
           for tick in range(START, END + 1)):
        raise ValueError("E228 recorded 42-joint, all-off adhesion, eight-tendon input required")
    return paths, trace, native, force


def hold_command(original, anchor, arm, tick):
    """Return exact source command except one frozen seven-joint hold interval."""
    if arm not in ARMS or not START < tick <= END:
        raise ValueError("E228 preregistered arm/control required")
    source = np.asarray(original, np.float64)
    held = np.asarray(anchor, np.float64)
    if source.shape != (42,) or held.shape != (42,) or not (
            np.isfinite(source).all() and np.isfinite(held).all()):
        raise ValueError("Finite recorded 42-joint targets required")
    applied = source.copy()
    if HOLD_FIRST <= tick <= HOLD_LAST and arm != "sham":
        columns = range(21, 28) if arm == "rf_hold" else range(35, 42)
        applied[list(columns)] = held[list(columns)]
    validate_hold(source, applied, held, arm, tick)
    return applied


def validate_hold(original, applied, anchor, arm, tick):
    expected = np.asarray(original, np.float64).copy()
    if HOLD_FIRST <= tick <= HOLD_LAST and arm != "sham":
        columns = range(21, 28) if arm == "rf_hold" else range(35, 42)
        expected[list(columns)] = np.asarray(anchor, np.float64)[list(columns)]
    if not np.array_equal(np.asarray(applied), expected):
        raise ValueError("E228 command changed outside frozen leg/hold indices")


def source_slice(tick):
    if not START < tick <= END:
        raise ValueError("E228 source control outside local window")
    first = (tick - 601) * STEPS
    return slice(first, first + STEPS)


def local_slice(tick):
    if not START < tick <= END:
        raise ValueError("E228 local control outside window")
    first = (tick - START - 1) * STEPS
    return slice(first, first + STEPS)


def first_array_difference(left, right):
    """First row with byte difference, including differing NaN payloads."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("E228 comparison shape/dtype mismatch")
    for index in range(len(a)):
        if a[index].tobytes() != b[index].tobytes():
            return index
    return None


def first_native_divergence(samples, reference, *, source_start=50350):
    """Find the first retained native mismatch, including a partial control."""
    if not samples:
        return None
    actual = e113.native_arrays(samples)
    if set(actual) != set(reference):
        raise ValueError("E228 exact six native comparison fields required")
    first = None
    for key in NATIVE_FIELDS:
        source = reference[key][source_start:source_start + len(samples)]
        offset = first_array_difference(actual[key], source)
        if offset is not None:
            global_index = source_start + offset
            first = global_index if first is None else min(first, global_index)
    return first


def compare_force_chunk(rows, reference, tick):
    if len(rows) != STEPS:
        raise ValueError("E228 force interval must contain 50 native samples")
    offset = source_slice(tick).start
    for index, row in enumerate(rows):
        for key, expected in reference.items():
            actual = np.asarray(row[key])
            expected_value = np.asarray(expected[offset + index])
            if (actual.shape != expected_value.shape
                    or actual.dtype != expected_value.dtype
                    or actual.tobytes() != expected_value.tobytes()):
                raise ValueError("E228 exact lead/sham force differs: " + key)


def phase_metrics(native, force, control_rows, phases, threshold=.02,
                  initial_heading=None):
    """Completed-arm native support/force/slip summaries without zero-filling."""
    if (len(native["time_s"]) != 2000 or len(force["time_s"]) != 2000
            or len(control_rows) != 41 or not np.all(force["available"])):
        raise ValueError("Complete healthy E228 arm required for phase metrics")
    report = {}
    for name, (first_control, stop_control) in phases.items():
        start, stop = (first_control - START - 1) * STEPS, (
            stop_control - START - 1) * STEPS
        support = native["normal_load_bw"][start:stop] >= threshold
        horizontal = force["all_leg_heading_xy_force_bw"][start:stop]
        slip = native["slip_mm_s"][start:stop]
        slip_valid = native["slip_valid"][start:stop] & support
        positive = (np.maximum(horizontal, 0.) * support).sum(axis=0) * .0001
        negative = (np.maximum(-horizontal, 0.) * support).sum(axis=0) * .0001
        signed = (horizontal * support).sum(axis=0) * .0001
        support_count = support.sum(axis=0)
        mean_force = np.divide((horizontal * support).sum(axis=0),
            support_count, out=np.full(6, np.nan), where=support_count > 0)
        total_negative = float(negative.sum())
        slip_count = slip_valid.sum(axis=0)
        slip_sum = np.where(slip_valid, slip, 0.).sum(axis=0)
        slips = [float(slip_sum[i] / slip_count[i]) if slip_count[i] else None
                 for i in range(6)]
        rows = control_rows[first_control - START:stop_control - START]
        errors = np.asarray([row["joint_tracking_error_rad"] for row in rows],
                            np.float64)
        report[name] = {"controls": list(range(first_control, stop_control)),
            "native_sample_count": stop - start,
            "supported_time_s_by_leg": (support.sum(axis=0) * .0001).tolist(),
            "positive_supported_BW_s_by_leg": positive.tolist(),
            "negative_supported_BW_s_by_leg": negative.tolist(),
            "net_supported_BW_s_by_leg": signed.tolist(),
            "mean_supported_BW_by_leg": [float(mean_force[i]) if support_count[i]
                                          else None for i in range(6)],
            "negative_share_by_leg": (negative / total_negative).tolist()
                if total_negative > 0 else None,
            "supported_slip_mean_mm_s_by_leg": slips,
            "supported_slip_valid_count_by_leg": slip_count.tolist(),
            "upright_min": float(np.min(native["upright"][start:stop])),
            "nonfoot_count": int(native["nonfoot_contact"][start:stop].sum()),
            "joint_tracking_RMS_rad": float(np.sqrt(np.mean(errors**2)))}
        if initial_heading is not None:
            original = np.einsum("tlj,j->tl",
                force["all_leg_force_bw"][start:stop], initial_heading)
            original_positive = (np.maximum(original, 0.) * support).sum(
                axis=0) * .0001
            original_negative = (np.maximum(-original, 0.) * support).sum(
                axis=0) * .0001
            report[name]["original_initial_horizontal"] = {
                "positive_supported_BW_s_by_leg": original_positive.tolist(),
                "negative_supported_BW_s_by_leg": original_negative.tolist(),
                "net_supported_BW_s_by_leg": (
                    original_positive - original_negative).tolist()}
    return report


def trajectory_metrics(rows, heading, velocity_controls):
    """Use original control0 horizontal direction and the frozen gradient."""
    if len(rows) != 41 or [row["control_tick"] for row in rows] != list(
            range(START, END + 1)):
        raise ValueError("Complete E228 body rows required for trajectory")
    heading = np.asarray(heading, np.float64)
    if heading.shape != (3,) or not np.isclose(np.linalg.norm(heading), 1.):
        raise ValueError("Original unit horizontal heading required")
    ui_positions = np.asarray([row["body"]["position"] for row in rows],
                              np.float64)
    native_position = ui_positions @ replay.S
    forward = native_position @ heading
    velocity = np.gradient(forward, DT)
    indices = np.asarray(velocity_controls, int) - START
    if (not np.array_equal(indices, np.arange(13, 40))
            or not np.isfinite(velocity).all()):
        raise ValueError("Frozen E228 interior velocity controls required")
    selected = velocity[indices]
    return {"initial_forward_position_mm": forward.tolist(),
        "initial_forward_speed_mm_s": velocity.tolist(),
        "speed_controls": list(range(START, END + 1)),
        "summary_controls": list(velocity_controls),
        "initial_forward_displacement_mm": float(forward[-1] - forward[0]),
        "min_initial_forward_speed_mm_s": float(np.min(selected)),
        "backward_distance_mm": float(np.maximum(-selected, 0.).sum() * DT),
        "backward_time_s": float(np.count_nonzero(selected < 0.) * DT),
        "final_yaw_change_rad": float(rows[-1]["body"]["yaw"] -
                                      rows[0]["body"]["yaw"])}


def assess_local(sham, hold, *, arm, thresholds):
    """Only complete healthy arms receive the preregistered local verdict."""
    if not sham or not hold:
        return {"available": False, "reason": "incomplete_or_health_stop"}
    base = sham["trajectory"]
    other = hold["trajectory"]
    baseline_backward = base["backward_distance_mm"]
    reduction = ((baseline_backward - other["backward_distance_mm"])
                 / baseline_backward) if baseline_backward > 0 else None
    speed_gain = (other["min_initial_forward_speed_mm_s"] -
                  base["min_initial_forward_speed_mm_s"])
    distance_loss = (base["initial_forward_displacement_mm"] -
                     other["initial_forward_displacement_mm"])
    local = (reduction is not None and reduction >= thresholds[
        "backward_distance_reduction_fraction"] and speed_gain >= thresholds[
        "minimum_forward_speed_gain_mm_s"] and distance_loss <= thresholds[
        "maximum_forward_distance_loss_mm"])
    result = {"available": True, "arm": arm,
        "backward_distance_reduction_fraction": reduction,
        "minimum_forward_speed_gain_mm_s": speed_gain,
        "forward_distance_loss_mm": distance_loss,
        "local_body_improvement_supported": bool(local)}
    if arm == "rf_hold":
        rf = replay.LEGS.index("rf")
        original_negative = sham["phases"]["during_break"][
            "negative_supported_BW_s_by_leg"][rf]
        hold_negative = hold["phases"]["during_break"][
            "negative_supported_BW_s_by_leg"][rf]
        force_reduction = ((original_negative - hold_negative) /
                           original_negative) if original_negative > 0 else None
        result["RF_negative_force_reduction_fraction"] = force_reduction
        result["RF_command_mechanical_contribution_supported"] = bool(
            local and force_reduction is not None and force_reduction >= thresholds[
                "rf_negative_integral_reduction_fraction"])
    return result


def save_npz(path, arrays):
    replay.save_npz(path, arrays)


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e228.local-command-hold.ledger.v1",
        "body_constructors_requested": 0, "body_constructors_returned": 0,
        "constructor_settling_steps_nominal_source": 2000,
        "constructor_settling_steps_measured": None,
        "scientific_controls_requested": 0, "scientific_controls_returned": 0,
        "scientific_steps_requested": 0, "scientific_steps_observed": 0,
        "new_neural_ticks": 0, "cuda_calls": 0, "fit_calls": 0,
        "solver_calls": 0, "retries": 0, "arms": {}, "storage_errors": []}
    result = {"schema": "flylab.m3.e228.local-command-hold.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "body_only_local_diagnostic": True, "arm_order": list(ARMS),
        "arms": {}, "E225_10s_complete": False,
        "E225_startup_previously_failed": True,
        "E225_startup_recorded_command_RMS_rad": .018052184585,
        "E225_startup_recorded_command_max_abs_rad": .05806575858,
        "E225_all_six_eligible_cycles": 0,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    body = restore_hook = thread_limits = process = None
    current_arm = None
    native = []
    force_sink = replay.ForceSink(freeze["force_fields"])
    rows = []
    metrics = {}

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E228 " + name + ": " + repr(error))

    def persist():
        ledger["wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def budget():
        if time.perf_counter() - began >= limits["internal_s"] - limits[
                "finalization_reserve_s"]:
            raise TimeoutError("E228 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E228 resident memory cap exceeded")

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        paths, reference, source_native, source_force = load_reference(freeze)
        base = StateStore.load(paths["base_checkpoint"])
        start = StateStore.load(paths["body_checkpoint"])
        if start["modelHash"] != base["body"]["modelHash"]:
            raise ValueError("E228 exact event25 body checkpoint required")
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threading = one_cpu_thread()
        ledger["threading"] = threading
        import psutil
        process = psutil.Process()
        budget()
        ledger["body_constructors_requested"] = 1
        persist()
        constructor_start = time.perf_counter()
        body = observation.new_body(base)
        ledger["constructor_wall_s"] = time.perf_counter() - constructor_start
        ledger["body_constructors_returned"] = 1
        if body.model_hash != start["modelHash"]:
            raise ValueError("E228 physical model hash changed")
        heading = replay.initial_horizontal_heading(reference[0])
        (out / "arms").mkdir()
        for arm in ARMS:
            budget()
            current_arm = arm
            arm_out = out / "arms" / arm
            arm_out.mkdir()
            native = []
            force_sink = replay.ForceSink(freeze["force_fields"])
            rows = []
            arm_ledger = {"status": "restore_requested", "controls_requested": 0,
                "controls_returned": 0, "native_samples": 0,
                "force_samples": 0, "clock_events": []}
            ledger["arms"][arm] = arm_ledger
            arm_result = {"arm": arm, "status": "running", "complete_healthy": False,
                "first_qpos_difference_control": None,
                "first_ctrl_difference_control": None,
                "first_native_difference_sample": None,
                "first_physical_difference_control": None}
            result["arms"][arm] = arm_result
            try:
                body.restore(start)
                exact(start, body.snapshot(), "E228 restored start body " + arm)
                if not np.isclose(body.physics_time(), START * DT,
                                  atol=2e-8, rtol=0):
                    raise ValueError("E228 restored event25 body clock changed")
                start_row = replay.capture_physical_row(body, START,
                    reference[START]["tendon_inputs"])
                replay.compare_physical_row(start_row, reference[START])
                StateStore.save(arm_out / "start", body.snapshot())
                rows.append(start_row)
                arm_ledger["status"] = "start_exact"
                if restore_hook is not None:
                    restore_hook()
                    restore_hook = None
                restore_hook = ground.install_native_observer(body, native,
                    force_sink, {}, replay.contact_diagnostics, e113.NativeHealthStop)
                with ((arm_out / "commands.jsonl").open("x", encoding="utf-8",
                                                       buffering=1) as commands,
                      (arm_out / "control.jsonl").open("x", encoding="utf-8",
                                                      buffering=1) as controls):
                    controls.write(json.dumps(start_row, allow_nan=False) + "\n")
                    anchor = reference[ANCHOR]["joint_targets_rad"]
                    for tick in range(START + 1, END + 1):
                        budget()
                        source = reference[tick]
                        applied = hold_command(source["joint_targets_rad"],
                                               anchor, arm, tick)
                        command = {"control_tick": tick,
                            "requested_command": source["joint_targets_rad"],
                            "applied_command": applied.tolist(),
                            "adhesion": source["adhesion"],
                            "tendon_inputs": source["tendon_inputs"],
                            "joint_stiffness_scale": source.get(
                                "joint_stiffness_scale"),
                            "adhesion_control": source.get("adhesion_control")}
                        commands.write(json.dumps(command, allow_nan=False) + "\n")
                        before = float(body.physics_time())
                        old_count = len(native)
                        entry = {"control_tick": tick,
                            "physics_time_before_s": before,
                            "native_before": old_count, "returned": False}
                        arm_ledger["clock_events"].append(entry)
                        arm_ledger["controls_requested"] += 1
                        ledger["scientific_controls_requested"] += 1
                        ledger["scientific_steps_requested"] += STEPS
                        try:
                            body.step_joint_targets(applied,
                                np.asarray(source["adhesion"], bool),
                                tendon_inputs=source["tendon_inputs"],
                                interpolate=True,
                                **({"joint_stiffness_scale": source[
                                    "joint_stiffness_scale"]} if
                                    "joint_stiffness_scale" in source else {}),
                                **({"adhesion_control": source["adhesion_control"]}
                                   if "adhesion_control" in source else {}))
                            entry["returned"] = True
                            arm_ledger["controls_returned"] += 1
                            ledger["scientific_controls_returned"] += 1
                        finally:
                            after = float(body.physics_time())
                            entry["physics_time_after_s"] = after
                            interval = after - before
                            entry["actual_physical_s"] = (interval if math.isfinite(
                                interval) else None)
                            entry["native_after"] = len(native)
                            arm_ledger["native_samples"] = len(native)
                            arm_ledger["force_samples"] = len(force_sink)
                            ledger["scientific_steps_observed"] = sum(
                                item["native_samples"] for item in ledger[
                                    "arms"].values())
                        if len(native) != old_count + STEPS or len(force_sink) != len(native):
                            raise ValueError("E228 exactly50 native/force samples required")
                        if arm == "sham" or tick <= ANCHOR:
                            replay.compare_native_chunk(native[old_count:],
                                source_native, source_slice(tick).start,
                                source_slice(tick).stop)
                            compare_force_chunk(force_sink.rows[old_count:],
                                                source_force, tick)
                        else:
                            actual_chunk = e113.native_arrays(native[old_count:])
                            reference_chunk = {key: source_native[key][
                                source_slice(tick)] for key in NATIVE_FIELDS}
                            for key in NATIVE_FIELDS:
                                first = first_array_difference(actual_chunk[key],
                                                               reference_chunk[key])
                                if first is not None:
                                    global_index = source_slice(tick).start + first
                                    current = arm_result["first_native_difference_sample"]
                                    arm_result["first_native_difference_sample"] = (
                                        global_index if current is None else min(
                                            current, global_index))
                        row = replay.capture_physical_row(body, tick,
                            source["tendon_inputs"])
                        if arm == "sham" or tick <= ANCHOR:
                            replay.compare_physical_row(row, source)
                        else:
                            for key, receipt in (("qpos", "first_qpos_difference_control"),
                                                 ("ctrl", "first_ctrl_difference_control")):
                                if arm_result[receipt] is None:
                                    try:
                                        exact(source[key], row[key], key)
                                    except AssertionError:
                                        arm_result[receipt] = tick
                            if arm_result["first_physical_difference_control"] is None:
                                try:
                                    replay.compare_physical_row(row, source)
                                except AssertionError:
                                    arm_result["first_physical_difference_control"] = tick
                        rows.append(row)
                        controls.write(json.dumps(row, allow_nan=False) + "\n")
                if len(rows) != 41 or len(native) != 2000 or len(force_sink) != 2000:
                    raise ValueError("E228 completed arm count differs")
                native_arrays = e113.native_arrays(native)
                force_arrays = force_sink.arrays()
                if not np.all(force_arrays["available"]):
                    raise ValueError("E228 incomplete shadow force")
                StateStore.save(arm_out / "final", body.snapshot())
                arm_result["status"] = "complete_healthy"
                arm_result["complete_healthy"] = True
                arm_ledger["status"] = "complete_healthy"
                metrics[arm] = {"trajectory": trajectory_metrics(rows, heading,
                        freeze["event"]["velocity_summary_controls"]),
                    "phases": phase_metrics(native_arrays, force_arrays, rows,
                        freeze["phases"], freeze["native"]["support_threshold_bw"],
                        heading)}
                prior.write_receipt(arm_out / "metrics.json", metrics[arm])
            except e113.NativeHealthStop as error:
                arm_result.update(status="measured_native_health_stop",
                                  measured_error=repr(error))
                arm_ledger["status"] = "measured_native_health_stop"
            except (TimeoutError, MemoryError) as error:
                arm_result.update(status="resource_limited", error=repr(error))
                arm_ledger["status"] = "resource_limited"
                raise
            except BaseException as error:
                arm_result.update(status="technical_failure", error=repr(error))
                arm_ledger["status"] = "technical_failure"
                raise
            finally:
                attempt(arm + "/native.npz", lambda arm_out=arm_out,
                        samples=native: save_npz(
                    arm_out / "native.npz", e113.native_arrays(samples)))
                attempt(arm + "/force.npz", lambda arm_out=arm_out,
                        sink=force_sink: save_npz(
                    arm_out / "force.npz", sink.arrays()))
                if arm != "sham":
                    def retain_partial_divergence(samples=native, receipt=arm_result):
                        first = first_native_divergence(samples, source_native)
                        prior_first = receipt["first_native_difference_sample"]
                        if first is not None:
                            receipt["first_native_difference_sample"] = (
                                first if prior_first is None else min(first, prior_first))
                    attempt(arm + "/partial_native_difference",
                            retain_partial_divergence)
                arm_result["last_physics_time_s"] = (float(body.physics_time())
                    if body is not None and np.isfinite(body.physics_time()) else None)
                arm_result["native_samples"] = len(native)
                arm_result["force_samples"] = len(force_sink)
                attempt(arm + "/arm-result.json", lambda arm_out=arm_out,
                        arm_result=arm_result: prior.write_receipt(
                            arm_out / "arm-result.json", arm_result))
                persist()
            if primary is not None:
                raise primary
            if arm == "sham" and not arm_result["complete_healthy"]:
                result["validity"] = "measured_native_health_stop"
                result["interventions_not_executed_reason"] = "sham_health_stop"
                break
        result["local_comparisons"] = {arm: assess_local(metrics.get("sham"),
            metrics.get(arm), arm=arm,
            thresholds=freeze["diagnostic_thresholds"])
            for arm in ARMS[1:]}
        if result["validity"] == "incomplete":
            result["validity"] = ("valid_local_body_command_hold_diagnostic"
                if all(result["arms"][arm]["complete_healthy"] for arm in ARMS)
                else "measured_native_health_stop")
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        if current_arm is not None and restore_hook is not None:
            attempt("observer_cleanup", restore_hook)
        if body is not None:
            attempt("body_cleanup", body.close)
        if thread_limits is not None:
            attempt("cpu_thread_cleanup", thread_limits.restore_original_limits)
        ledger["wall_s"] = time.perf_counter() - began
        ledger["output_bytes_before_final_receipts"] = sum(
            path.stat().st_size for path in out.rglob("*") if path.is_file())
        if ledger["wall_s"] > limits["internal_s"] and primary is None:
            primary = TimeoutError("E228 internal wall cap exceeded")
            result["validity"] = "resource_limited"
        if ledger["output_bytes_before_final_receipts"] > limits[
                "output_bytes"] and primary is None:
            primary = OSError("E228 output cap exceeded")
            result["validity"] = "technical_storage_failure"
        if primary is not None and result["validity"] == (
                "valid_local_body_command_hold_diagnostic"):
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", replay.manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.freeze, args.freeze_sha256, args.review,
                     args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
